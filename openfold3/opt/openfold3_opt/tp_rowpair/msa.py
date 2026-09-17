"""OpenFold3's MSA module on row shards (``latent/msa_module.py`` ``MSAModuleStack`` / ``MSAModuleBlock``, ``latent/base_blocks.py``
``MSABlock``) and the host-resident MSA-feature plumbing (``MSA_HOST`` / ``BONDS_ON_HOST``).

Every tensor statement below is either OpenFold3's own sub-module call (``OuterProductMean.layer_norm/linear_1/linear_2/_opm/_chunk``,
``MSAPairWeightedAveraging._prep_inputs/layer_norm_m/linear_v/linear_g/linear_o``, ``MSAModuleEmbedder._subsample_all_msa``) or the core's
row statement it is handed to (``opt_core.mem.rowpair.msa`` / ``.transition`` / ``.msa_host``); this module owns the attribute paths, the
predict-configuration refusals and OpenFold3's block order (``opm_first`` / ``skip_msa_update``), nothing else:

  * outer-product mean: OUTPUT ROWS ``a[rows] x b[all]`` (``msa.opm_rows_budgeted``: row block from ``ROWPAIR_OPM_ROWS`` | the row-block
    budget | the agreed free bytes, a multiple of the run's chunk so ``OuterProductMean._chunk`` walks its global chunk grid; ``z += OPM(m)``
    lands in place per row block when ``inplace_safe``);
  * pair-weighted averaging: the softmax logits of LOCAL query rows from the shard (``msa.pwa_bias_rows``: ``_prep_inputs`` on z rows —
    complete rows, so the softmax over ``j`` needs no communication), the m-side statements on the full replicated ``m`` chunk (stock M),
    ONE all-gather of the token rows per sequence chunk BEFORE ``linear_o`` (``msa.pwa_rows``); OpenFold3's softmax route is kept
    (triton ``fused_softmax`` on CUDA, ``softmax_no_cast`` otherwise);
  * MSA transition: replicated (OpenFold3's own call) or token-sharded + gathered under ``ROWPAIR_MSA_TRANS_SHARD=1`` (``msa.msa_transition_rows``);
  * each block's ``pair_stack`` through the ONE pair-block driver (``pairstack.pair_block_rows``).

REPLICATED BY DESIGN: ``m`` ``[1, S<=1024, N, 64]`` on every rank (the embedder's subsample indices are ONE draw per cycle: rank 0's broadcast to
every rank under ``ROWPAIR_DIFF_NOISE_SYNC=bcast``, or every rank's own identically seeded draw proven identical by name under ``guard`` —
``sync_policy``; ``msa_m=replicated`` in the schedule census). HOST-RESIDENT (``MSA_HOST`` = ``rank0``): the raw MSA features
``msa`` / ``has_deletion`` / ``deletion_value`` never reach the device whole (``opt_core.mem.rowpair.msa_host``): ``place_batch_features``
parks them (``rank0``: ranks > 0 hold zero-row placeholders), ``msa_module_embedder_forward_host`` (installed over ``MSAModuleEmbedder.forward``)
draws the subsample with OpenFold3's statements ON THE DEVICE and moves only the selected rows (``rank0``: rank 0 gathers, the rows are
broadcast in ``ROWPAIR_BCAST_CHUNK_GB`` pieces). ``BONDS_ON_HOST`` parks ``token_bonds`` ``[N, N]`` on every rank's host (row slabs
move in ``trunk.input_embedder_rows``). Predict configuration only for the host path (batch size 1, ``subsample_all_msa``, no
``subsample_main_msa``) — refused by name otherwise.
"""
from __future__ import annotations

import functools
import math
import os

import torch
from openfold3.core.utils.tensor_utils import add

from . import core as C
from . import pairstack as PS


class MsaRefused(RuntimeError):
    pass


HOST_KEYS_MSA = ("msa", "has_deletion", "deletion_value")
ROW_DIMS_MSA = {"msa": -3, "has_deletion": -2, "deletion_value": -2}          # the per-cycle subsample selects along the N_msa dim of each
HOST_KEYS_BONDS = ("token_bonds",)

C.register("msa", ("opm_rows_budgeted", "pwa_bias_rows", "pwa_rows", "msa_transition_rows", "msa_block_sharded", "msa_module_sharded"))
C.register("msa_host", ("host_mode", "park_features", "place_skipped", "rows_to_device", "residency_text"))


MSA_HOST = "rank0"                      # the raw MSA features' placement on the tp line, in the core's placement words (msa_host.host_mode): rank 0's host, the selected rows broadcast
BONDS_ON_HOST = True                    # token_bonds on every rank's host
FORCE_HOST = False                      # tests: take the host path even when the features already sit on the compute device of a one-rank group


def mode():
    """The MSA placement in the core's words (``msa_host.host_mode(MSA_HOST)`` = ``"rank0"``: rank 0's host; rows broadcast)."""
    return C.fn("msa_host", "host_mode")(MSA_HOST)


def forced() -> bool:
    return FORCE_HOST


def bonds_on_host() -> bool:
    return BONDS_ON_HOST


def host_keys():
    return (HOST_KEYS_MSA if mode() else ()) + (HOST_KEYS_BONDS if bonds_on_host() else ())


# ----------------------------------------------------------------------------------------------------------------- residency census (logs only)
def log_batch_residency(batch, tag, top=12):
    comm = C.comm()
    if comm.verbose:
        comm.log(C.fn("msa_host", "residency_text")(batch, tag, top=top))


def log_live_tensors(tag, min_gib=1.0):
    comm = C.comm()
    if comm.verbose and torch.cuda.is_available():
        comm.log(f"[live] {tag}: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB")


# ----------------------------------------------------------------------------------------------------------------- host-resident features
def place_batch_features(batch, device=None):
    """Park ``host_keys()`` on (pinned) host in place; idempotent (``msa_host.park_features``: mode ``rank0`` leaves zero-row placeholders of
    the MSA keys on ranks > 0; ``token_bonds`` is parked on every rank). Returns ``batch``."""
    MH = C.seam("msa_host")
    md = mode()
    if md is not None:
        MH.park_features(batch, HOST_KEYS_MSA, mode=md, row_dims=ROW_DIMS_MSA, log=C.comm().verbose)
    if bonds_on_host():
        MH.park_features(batch, HOST_KEYS_BONDS, mode="all", row_dims={"token_bonds": -2}, log=C.comm().verbose)
    return batch


def _feat_dtype(batch):
    """The dtype ``torch.cat([msa, has_deletion[..., None], deletion_value[..., None]], -1)`` promotes to (``MSAModuleEmbedder.forward``)."""
    return functools.reduce(torch.promote_types, [batch[k].dtype for k in HOST_KEYS_MSA])


SYNC_SWITCH = "ROWPAIR_DIFF_NOISE_SYNC"
"""The replicated-draw policy the line sets per determinism level (modes.replicated_sync_policy): ``bcast`` (rank 0's draw on every rank) |
``guard`` (every rank draws; proven identical by name; the default when unset). The MSA subsample and the diffusion state follow the same word."""


def sync_policy() -> str:
    """``guard`` when the switch is absent (the core's own default for its diffusion sync points); the tp line always sets it (bcast at det 0, guard at det 1)."""
    v = (os.environ.get(SYNC_SWITCH) or "guard").strip().lower()
    if v not in ("bcast", "guard"):
        raise MsaRefused(f"refused: {SYNC_SWITCH}={v!r} for the MSA subsample draw (bcast | guard)")
    return v


def msa_module_embedder_forward_host(self, batch: dict, s_input: torch.Tensor):
    """``MSAModuleEmbedder.forward`` (``input_embedders.py``) with the MSA features host-resident. The subsample size and the selection are
    OpenFold3's statements run ON THE DEVICE in their order (``torch.randint`` of the forward, then ``_subsample_all_msa`` applied to an index
    stand-in of ``msa_feat`` — it reads the stand-in's device and ``index_select``s it, so the drawn indices come back instead of feature rows);
    only the selected rows are gathered from the host copies and moved (``msa_host.rows_to_device``: mode ``rank0`` gathers on rank 0 and
    broadcasts). The stock forward runs unchanged when the lever is off or the features already live on the compute device."""
    md = mode()
    msa, device = batch["msa"], s_input.device
    use_host = md is not None and (msa.device != device or int(msa.shape[-3]) == 0 or (md == "rank0" and C.comm().world > 1) or forced())
    if not use_host:
        from . import model as M                              # the stock forward, as recorded by the install (model.PATCHES)
        return M.PATCHES.original(type(self), "forward")(self, batch=batch, s_input=s_input)
    batch_dims = tuple(msa.shape[:-3])
    if math.prod(batch_dims) > 1 or self.subsample_main_msa or not self.subsample_all_msa:
        raise MsaRefused("refused: the tp line's host-parked MSA supports the predict configuration only (batch size 1, subsample_all_msa=true, "
                         "subsample_main_msa=false)")
    msa_mask = batch["msa_mask"]
    if msa_mask.device != device:
        msa_mask = msa_mask.to(device)
    n_msa = int(msa_mask.shape[-2])
    stand_in = torch.arange(n_msa, device=device).view(*batch_dims, n_msa, 1, 1)          # index_select(-3, selected) of it == selected

    def draw() -> torch.Tensor:                               # OpenFold3's two RNG statements in their order: the subsample size, then the selection
        n_pick = torch.randint(low=self.min_subsampled_all_msa, high=int(self.max_subsampled_all_msa + 1), size=(1,), device=device).item()
        picked, _ = type(self)._subsample_all_msa(msa_feat=stand_in, msa_mask=msa_mask, no_subsampled_all_msa=n_pick)
        return picked.reshape(-1)

    policy = sync_policy()
    if policy == "bcast":                                     # rank 0 draws, every rank receives and proves the draw (the core's draw_replicated)
        selected = C.fn("msa", "draw_replicated")(draw, "msa_subsample", device).to(device=device, dtype=torch.long)
    else:                                                     # guard: every rank draws with its own (identically seeded) generator; the core proves them identical by name
        selected = draw()
        C.fn("trunk", "guard_replicated")(selected, "msa_subsample")
    mask_sub = msa_mask.index_select(-2, selected)            # == _subsample_all_msa's mask_sub for these indices
    N, n_sel = int(msa_mask.shape[-1]), int(selected.numel())
    c_feats = int(self.linear_m.in_features) if hasattr(self.linear_m, "in_features") else int(self.linear_m.weight.shape[1])
    dtype = _feat_dtype(batch)

    def build():                                              # the selected rows of cat([msa, has_deletion, deletion_value]) gathered ON THE HOST, then moved
        parts = []
        for k in HOST_KEYS_MSA:
            h = batch[k]
            rows = h.index_select(ROW_DIMS_MSA[k] % h.dim(), selected.to(h.device))
            parts.append(rows if k == "msa" else rows.unsqueeze(-1))
        return torch.cat(parts, dim=-1).to(device=device, dtype=dtype).contiguous()

    msa_feat = C.fn("msa_host", "rows_to_device")(build, shape=(*batch_dims, n_sel, N, c_feats), dtype=dtype, device=device, mode=md, name="msa_rows")
    m = self.linear_m(msa_feat)
    m = m + self.linear_s_input(s_input).unsqueeze(-3)
    return m, mask_sub


# ----------------------------------------------------------------------------------------------------------------- MSA module on shards
def _rows2(z_loc, lay):
    """The ``[n_loc, N, C]`` view of a ``[1, n_loc, N, C]`` shard (the core's row statements index rows on dim 0); refuses other leading shapes."""
    if z_loc.dim() != 4 or int(z_loc.shape[0]) != 1 or int(z_loc.shape[1]) != lay.R or int(z_loc.shape[2]) != lay.N:
        raise MsaRefused(f"refused: z shard {tuple(z_loc.shape)} is not [1, n_loc={lay.R}, N={lay.N}, C] (predict batches carry batch size 1)")
    return z_loc[0]


def opm_rows(opm, m, msa_mask, z_loc, lay, chunk_size=None, inplace_safe=False):
    """``z = add(z, OuterProductMean(m, mask, chunk_size), inplace)`` restricted to this rank's rows (``MSABlock._compute_opm``, no offload).
    The operands are OpenFold3's statements on the replicated ``m`` (``layer_norm``, ``linear_1/2`` x mask, transposed to ``[N, S, c]``);
    per output row block the engine's ``_chunk`` / ``_opm`` and mask normalisation run on ``a[rows] x b[all]``. Returns ``z_loc``."""
    MS = C.seam("msa")
    if m.dim() != 4 or int(m.shape[0]) != 1:
        raise MsaRefused(f"refused: m {tuple(m.shape)} is not [1, S, N, c_m] (predict batches carry batch size 1)")
    if msa_mask is None:
        msa_mask = m.new_ones(m.shape[:-1])
    ln = C.fn("shard", "ln_rows_guarded")(opm.layer_norm, m, rows_dim=-3)      # the whole replicated m [1, S, N, c]: sequence-row blocks from ROWPAIR_LN_GUARD_ELEMS elements (S x N x c >= 2^31 from N = 32,768 at S = 1,024)
    mask = msa_mask.unsqueeze(-1)
    a = (opm.linear_1(ln) * mask).transpose(-2, -3)[0]        # [N, S, c]
    b = (opm.linear_2(ln) * mask).transpose(-2, -3)[0]
    del ln
    mask2 = mask[0]                                           # [S, N, 1]
    c = int(a.shape[-1])
    C_z = int(opm.linear_out.out_features) if hasattr(opm.linear_out, "out_features") else int(opm.linear_out.weight.shape[0])

    def outer_fn(a_blk, b_all, g0, g1):                       # OuterProductMean._forward for token rows [g0, g1): its own chunk grid + norm
        outer = opm._chunk(a_blk, b_all, chunk_size) if chunk_size is not None else opm._opm(a_blk, b_all)
        norm = torch.einsum("...abc,...adc->...bdc", mask2[..., g0:g1, :], mask2) + opm.eps
        if inplace_safe:
            outer /= norm
            return outer
        return outer / norm

    N = lay.N
    per_row = N * int(a.element_size()) * (2 * C_z + (c * c if chunk_size is None else 0))   # the block's outer + normalised copy (+ the unchunked c*c einsum)
    z2 = _rows2(z_loc, lay)
    if inplace_safe:
        MS.opm_rows_budgeted(a, b, lay, outer_fn, C_z=C_z, out=z2, add=True, align=chunk_size, global_rows=True, row_dim=0, bytes_per_row=per_row)
        return z_loc
    rows = MS.opm_rows_budgeted(a, b, lay, outer_fn, C_z=C_z, out=None, add=False, align=chunk_size, global_rows=True, row_dim=0, bytes_per_row=per_row)
    return add(z_loc, rows.unsqueeze(0), inplace=False)


def _softmax_route():
    """OpenFold3's softmax of the pair-weighted averaging (``layers/msa.py`` ``_get_pair_weighted_avg``): triton ``fused_softmax`` on CUDA when
    triton is installed, ``softmax_no_cast`` otherwise."""
    from openfold3.core.model.layers import msa as _msa
    fused = getattr(_msa, "fused_softmax", None) if getattr(_msa, "triton_is_installed", False) else None
    return lambda x: fused(x) if (fused is not None and x.is_cuda) else _msa.softmax_no_cast(x, -1)   # noqa: E731


def pwa_rows(mod, m, z_loc, pair_mask_loc, lay, chunk_size=None):
    """``MSAPairWeightedAveraging.forward(m, z, mask, chunk_size)`` with z given as this rank's rows -> the full replicated m-update
    ``[1, S, N, c_m]``. OpenFold3's sequence chunk grid (``chunk_layer`` over the flattened ``(*, S)`` dims) is the core's ``s_chunk``."""
    MS = C.seam("msa")
    if m.dim() != 4 or int(m.shape[0]) != 1:
        raise MsaRefused(f"refused: m {tuple(m.shape)} is not [1, S, N, c_m] (predict batches carry batch size 1)")
    z2 = _rows2(z_loc, lay)
    mask2 = pair_mask_loc.reshape(lay.R, lay.N) if pair_mask_loc is not None else None
    H = int(mod.no_heads)

    def prep_fn(z_rows, g0, g1):                              # _prep_inputs on rows: LN_z, linear_z, permute to heads-first, + mask bias -> [H, rows, N]
        mk = mask2[g0 - lay.r0:g1 - lay.r0] if mask2 is not None else None
        return mod._prep_inputs(z=z_rows, mask=mk).reshape(H, g1 - g0, lay.N)

    def values_fn(m_chunk):                                   # layer_norm_m, linear_v (heads view), sigmoid(linear_g): the full [chunk, N, c_m] slab (stock M)
        m_in = mod.layer_norm_m(m_chunk)
        v = mod.linear_v(m_in)
        v = v.view(v.shape[:-1] + (H, -1)).transpose(-2, -3)  # [chunk, H, N, c]
        g = mod.sigmoid(mod.linear_g(m_in))
        g = g.view(g.shape[:-1] + (H, -1))                    # [chunk, N, H, c]
        return v, g

    def attend_fn(w, state, g0, g1):                          # weights x values in OpenFold3's einsum layout (w broadcast over the chunk), gate rows, flatten
        v, g = state
        n = int(v.shape[0])
        o = torch.einsum("...hqk,...hkc->...qhc", w.unsqueeze(0).expand(n, *w.shape), v)
        o = o * g[:, g0:g1]
        return o.reshape(n, g1 - g0, -1)                      # [chunk, q, H*c]

    bias = MS.pwa_bias_rows(prep_fn, z2, lay)                 # [H, n_loc, N]
    S, N, c_m = int(m.shape[-3]), int(m.shape[-2]), int(m.shape[-1])
    out = MS.pwa_rows(m.reshape(S, N, c_m), bias, lay, values_fn=values_fn, attend_fn=attend_fn, out_fn=mod.linear_o, softmax_fn=_softmax_route(),
                      s_chunk=chunk_size)
    return out.reshape(m.shape[:-1] + (out.shape[-1],))


def msa_transition_rows(blk, m, msa_trans_mask, lay, chunk_size, ckpt_chunk_size):
    """``blk.msa_transition(m, mask, chunk_size, ckpt_chunk_size)``: replicated, or token-sharded + gathered when ``ROWPAIR_MSA_TRANS_SHARD=1``."""
    def fn(m_tok, g0, g1):
        mk = msa_trans_mask[..., g0:g1] if msa_trans_mask is not None else None
        return blk.msa_transition(m_tok, mask=mk, chunk_size=chunk_size, ckpt_chunk_size=ckpt_chunk_size)
    return C.fn("msa", "msa_transition_rows")(fn, m, lay, shard_tokens=C.fn("dist", "env_int")("ROWPAIR_MSA_TRANS_SHARD", 0) == 1, token_dim=-2)


def _block_fns(blk, msa_mask, pair_mask_loc, lay, chunk_size, transition_ckpt_chunk_size, inplace_safe, _mask_trans, _attn_chunk_size, flags):
    """The per-block callables of ``msa.msa_module_sharded`` for one ``MSAModuleBlock`` (``MSAModuleBlock.forward``, no offload)."""
    msa_trans_mask = msa_mask if _mask_trans else None

    def opm(m, z_loc):
        return opm_rows(blk.outer_product_mean, m, msa_mask, z_loc, lay, chunk_size=chunk_size, inplace_safe=inplace_safe)

    def msa_update(m, z_loc):
        m = add(m, blk.msa_dropout_layer(pwa_rows(blk.msa_att_row, m, z_loc, pair_mask_loc, lay, chunk_size=chunk_size)), inplace=inplace_safe)
        return add(m, msa_transition_rows(blk, m, msa_trans_mask, lay, chunk_size, transition_ckpt_chunk_size), inplace=inplace_safe)

    def pair_block(z_loc):
        return PS.pair_block_rows(blk.pair_stack, z_loc, pair_mask_loc, lay, chunk_size=chunk_size, inplace_safe=inplace_safe, _mask_trans=_mask_trans,
                                  _attn_chunk_size=chunk_size if _attn_chunk_size is None else _attn_chunk_size, **flags)

    return dict(opm=opm, msa_update=None if blk.skip_msa_update else msa_update, pair_block=pair_block, opm_first=bool(blk.opm_first))


def msa_block_rows(blk, m, z_loc, msa_mask, pair_mask_loc, lay, chunk_size=None, transition_ckpt_chunk_size=None, inplace_safe=False, _mask_trans=True,
                   _attn_chunk_size=None, **flags):
    """``MSAModuleBlock.forward`` (no offload) with z row-sharded. Returns ``(m, z_loc)``."""
    fns = _block_fns(blk, msa_mask, pair_mask_loc, lay, chunk_size, transition_ckpt_chunk_size, inplace_safe, _mask_trans, _attn_chunk_size, flags)
    return C.fn("msa", "msa_block_sharded")(m, z_loc, lay, **fns)


def msa_module_rows(stack, m, z_loc, msa_mask, pair_mask_loc, lay, chunk_size=None, transition_ckpt_chunk_size=None, inplace_safe=False, _mask_trans=True,
                    _attn_chunk_size=None, **flags):
    """``MSAModuleStack.forward`` (eval: no activation checkpointing; ``tune_chunk_size`` pinned off) -> ``z_loc`` (the stack returns z only)."""
    if getattr(stack, "chunk_size_tuner", None) is not None:
        raise MsaRefused("refused: MSAModuleStack.tune_chunk_size=true under the tp line (runner yaml must pin tune_chunk_size=false; ranks would diverge)")
    if torch.is_grad_enabled() and getattr(stack, "blocks_per_ckpt", None):
        raise MsaRefused("refused: MSAModuleStack activation checkpointing (grad enabled) under the tp line — inference only")
    blocks = [_block_fns(blk, msa_mask, pair_mask_loc, lay, chunk_size, transition_ckpt_chunk_size, inplace_safe, _mask_trans, _attn_chunk_size, flags)
              for blk in stack.blocks]

    def between_blocks(i):
        if getattr(stack, "clear_cache_between_blocks", False) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return C.fn("msa", "msa_module_sharded")(m, z_loc, lay, blocks, between_blocks=between_blocks)
