"""The row-local sub-layers and the pair CONSUMERS of a row-sharded pair stack — what needs no collective, and the one collective each consumer needs.

Row-local (no collective; class ``row_local``): the pair transition (per element), the layer norms / projections / gates of every pair sub-layer,
the trunk prologue's pair terms (relative-position one-hots, bond features, the previous-pair LayerNorm+Linear, the template-pair sum — each
row's value depends on that row's tokens and replicated per-token features), the outer-product mean's OUTPUT rows given its left operand sliced
to my rows (:func:`opm_operands`), the structure-module / diffusion consumers that read pair row ``i`` for query token ``i``, the predicted-aligned-
error logits per row. Consumers with ONE collective: the MSA row-attention pair bias and the single/sequence attention pair logits — a ``[N/P, N, H]``
projection of my rows gathered to ``[N, N, H]`` (:func:`pair_logits_full`; class ``moves_bytes``); the distogram / distance-error symmetrisation
``left + left^T`` (:func:`symmetrize`; ``moves_bytes`` + the stock's add); masked means over the whole map (:func:`masked_mean`; class ``reordered`` —
two psums); full ``[N, N]`` maps a writer needs on the host (:func:`rows_full`). The recycle carry: :func:`constrain_carry` pins the pair leaves of the
carried tree to the row layout at the loop boundary (outside shard_map, inside jit) so the loop state is never materialised replicated.

The single-device ROW-CHUNK of these sub-layers (evaluating a row-independent sub-layer over row blocks into one output to cap the transient) is
the kit's existing sub-batch seam (``subbatch`` / ``sharded_apply`` in the stock trees) and runs unchanged on the local rows inside a region.
Standard library at import.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

from . import LEVER, _lazy
from . import shard as _shard

ROW_LOCAL_SITES = ("pair_transition", "trimul_projections_and_gates", "triatt_qkv_and_gating", "prologue_pair_embeddings",
                   "outer_product_mean_output_rows", "structure_or_diffusion_pair_row_reads", "pae_logits_rows")


def opm_operands(left_full: Any, norm_mask_full: Any, axis: str, n_loc: int, res_dim: int = 1) -> Tuple[Any, Any]:
    """The outer-product mean's left activations ``[S, N, c]`` and its mask ``[S, N, 1]`` sliced along the residue dim to MY rows (``[S, N/P, c]``):
    with the right operand left whole (MSA replicated), the stock einsum then yields exactly my output rows ``[N/P, N, C]`` and the normaliser
    ``einsum(mask_local, mask_full)`` matches. No collective (a ``dynamic_slice`` of a replicated operand)."""
    return (_shard.local_block(left_full, axis, n_loc, dim=res_dim), _shard.local_block(norm_mask_full, axis, n_loc, dim=res_dim))


def pair_logits_full(logits_local: Any, axis: str, row_dim: int = 0):
    """A per-head pair projection of my rows (``[N/P, N, H]``) gathered over rows (``[N, N, H]``): the MSA row-attention bias, the single / sequence
    attention pair logits. The stock's transpose to ``[H, N, N]`` follows unchanged."""
    return _shard.gather(logits_local, axis, dim=row_dim)


def symmetrize(left_local: Any, axis: str, row_dim: int = 0, col_dim: int = 1):
    """``left + left^T`` for a row-sharded map (distogram half-logits, distance-error logits): my rows of the transpose come from
    :func:`shard.transpose_block` (one all_to_all); the add is the stock's."""
    return left_local + _shard.transpose_block(left_local, axis, row_dim, col_dim)


def rows_full(x_local: Any, axis: str, row_dim: int = 0):
    """A row-sharded ``[N/P, N, …]`` map assembled to ``[N, N, …]`` on every device (an output the host writer needs whole: contact probabilities,
    PAE, the returned pair embeddings). One all_gather; use at the OUTPUT boundary only — inside the stack it undoes the sharding."""
    return _shard.gather(x_local, axis, dim=row_dim)


def masked_mean(values_local: Any, mask_local: Any, axis: str, eps: float, axes: Optional[Sequence[int]] = None):
    """``sum(mask*values) / (sum(mask) + eps)`` over the WHOLE map from row blocks — the stock mask-mean form with the STOCK's ``eps`` (the adapter passes
    it; no default here): two psums. Class ``reordered`` covers the psum order only; everything else is the stock's arithmetic."""
    jnp = _lazy.jnp()
    ax = tuple(axes) if axes is not None else None
    num = _shard.psum(jnp.sum(values_local * mask_local, axis=ax), axis)
    den = _shard.psum(jnp.sum(mask_local, axis=ax), axis)
    return num / (den + eps)


def constrain_carry(tree: Any, rmesh: Any, pair_keys: Iterable[str] = ("pair",), row_dim: int = 0):
    """The recycle-loop carry with its pair leaves pinned to the row layout (``with_sharding_constraint``; dict leaves named in ``pair_keys``; other
    leaves untouched). Call on the loop's initial value and on the body's result (outside shard_map, inside jit)."""
    keys = set(pair_keys)
    if isinstance(tree, dict):
        return {k: (_shard.constrain(v, rmesh, row_dim) if k in keys else v) for k, v in tree.items()}
    return _shard.constrain(tree, rmesh, row_dim)
