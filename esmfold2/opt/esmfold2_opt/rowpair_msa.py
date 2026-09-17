"""The MSA encoder of the FULL model (variant ``full_msa``: ``ESMFold2Model.msa_encoder``, modeling_esmfold2.py ``MSAEncoder`` /
``MSAEncoderBlock``; modeling_esmfold2_common.py ``OuterProductMean`` / ``MSAPairWeightedAveraging``) under the row-sharded pair stack of
``pred --mode big --n_gpu P`` (P > 1): the pair track the encoder reads and writes is this rank's ROWS ``x_pair[B, R, N, 256]`` (rows
``r0:r1`` of :class:`opt_core.mem.rowpair.dist.Layout`) from entry to exit — nothing ``[N, N, *]`` is materialised on a rank and the pair is
never gathered. The engine's own sub-modules run its own statements; only WHICH rows a statement sees changes:

* ``m`` — the MSA representation ``[B, N, M, d_msa]`` (``embed(msa features) + project_inputs(x_inputs)``), M <= ``msa_max_depth`` — is
  REPLICATED BY DESIGN on every rank (``msa_m=replicated`` in the schedule census; O(N·M), not O(N²)); so are the MSA features the loop
  hands in (``msa_oh``, ``has_deletion``, ``deletion_value``, ``msa_attention_mask``: integer-derived, GUARDED identical across ranks once
  per call — position-weighted fingerprints through :func:`opt_core.mem.rowpair.dist.allreduce_checksum`, refused by name on a mismatch:
  ``msa_features=guarded``);
* ``pair = x_pair + OuterProductMean(m)`` — the OPM OUTPUT rows of this rank: ``einsum('bimc,bjmd->bijcd', a[:, rows], b)`` against the whole
  replicated ``b`` operand, ``Wout`` and the ``/ n_valid`` divide in the checkpoint's order (``divide_outer_before_proj``), accumulated IN PLACE
  into the rank's rows block by block (:func:`opt_core.mem.rowpair.msa.opm_rows_budgeted`, ``add=True``: the row block derives from the
  AGREED free bytes / ``ROWPAIR_OPM_ROWS`` / ``ROWPAIR_ROWBLK_MB`` and is printed ``opm_rows=… opm_rows_source=…``);
* ``m = m + MSAPairWeightedAveraging(m, pair)`` — the pair bias ``Linear(LN(pair rows))`` and the softmax over ``j`` are COMPLETE for local
  query rows ``i`` (row-local, no communication); ``out[i, m] = Σ_j w[i, j] v[j, m] · gate[i, m]`` for local rows against the replicated
  ``v``; the ``[R, M, h·dh]`` out rows are ALL-GATHERED over token rows (O(N·M)) and ``Wout`` runs at the stock shape
  (:func:`opt_core.mem.rowpair.msa.pwa_bias_rows` + :func:`~opt_core.mem.rowpair.msa.pwa_rows`; ``pwa_qblock=R pwa_qblock_source=given``
  unless ``ROWPAIR_PWA_QBLOCK`` pins it: this engine's per-block transient is ``[q, N, h] + [q, M, h·dh]``, no ``chunk·H·q·N`` tensor);
* ``m = m + msa_transition(m)`` — replicated, the engine's own call (:func:`opt_core.mem.rowpair.msa.msa_transition_rows` with
  ``shard_tokens=False``: ``msa_transition=replicated``; the kit's t11 kernel composes when installed on the instance), issued per block of
  ``MSA_TRANSITION_TOKENS`` tokens (a per-token statement: identical values, the 4x-hidden transient bounded by the block);
* ``pair = pair + tri_mul_out(pair, mask)``; ``+ tri_mul_in``; ``+ pair_transition`` — the block's three pair statements
  (``MSAEncoderBlock.forward``) as one ``block(pair, pair_attention_mask=)``-shaped callable (:func:`pair_ops`) handed with the rows to the
  kit's ONE row form of a pair block, ``esmfold2_opt.rowpair.pair_block_rows_`` (:data:`PAIR_BLOCK_NAME`): the stock sub-modules are CALLED
  on the rows inside the sharded context, where the class-level ``TriangleMultiplicativeBlock.forward`` dispatcher runs the triangle
  multiplication in its row-sharded form (:mod:`opt_core.mem.rowpair.trimul`) and the pair transition is row-local: the install sets each
  block's ``PairTransition`` dim-1 chunk (its own ``set_chunk_size``) to ``EF2_ROWPAIR_MSA_TRANSITION_ROWS`` rows (else the pair
  stack's ``EF2_ROWPAIR_TRANSITION_ROWS``, else ``MSA_PAIR_TRANSITION_ROWS_DEFAULT`` = 64; ``0`` = the model's chunk as loaded), so its LayerNorm + SwiGLU hidden is
  ``[rows, N, 4·256]``, never the whole shard ``[R, N, 4·256]`` — ``msa_pair_transition=rows:<n>x<blocks>`` in the census; a block without a
  ``pair_transition.set_chunk_size`` is refused at install and a chunk changed after the install is refused at the call, both by name
  (``msa_trimul=reference_rows``: the MSA blocks' triangle multiplications are ``TriangleMultiplicativeBlock`` modules under every kernel backend of the
  folding trunk). This module binds no triangle op itself; an install whose record lacks that dispatcher is refused by name at the first call.

Layout bridge: this engine's ``m`` is token-major (``[B, N, M, d]``); the core's MSA statements take ``[S, N, c]`` (sequence-major). The
callables receive transposed VIEWS and transpose back, so every LayerNorm / Linear / einsum / softmax launch runs on the STOCK operand layout
(no copies besides the core's own gather staging). Numerics: per element the dense statement (the contraction over the MSA depth per
``(i, j)``, the softmax over ``j`` per row, LayerNorm statistics per row stay WHOLE on one rank); GEMM / einsum extents differ from the
single-GPU statement only in the row count — bitwise iff the stack's kernels are M-invariant (fp32 on CPU is; the tier-2 word of ``big``
covers CUDA). ``B == 1`` only (one item per ``pred`` fold; refused by name otherwise).

Single-GPU levers of the MSA path under rows: t12 (OPM epilogue), t13 (PWA kernel) and the kit's fused-trimul
``MSAEncoderBlock.forward`` (ef2_opt M1: the MSA blocks' triangle multiplication on the fused kernel) are FORWARD-level patches of the
modules whose SUB-modules this file calls directly — they do not run on rows (``msa_p1_levers_replaced=t12,t13,m1``; with M1 on at
``--n_gpu 1`` the MSA blocks' trimul kernel therefore differs between P == 1 and P > 1: inside the tier-2 word of ``big``); t11 (the
msa_transition instance forward) and t14 inside it compose (``msa_p1_levers_kept=t11``); a CUDA-graph wrapper on ``msa_encoder.forward``
(ef2_opt G3) is superseded by the rows forward (``msa_graphs_off``). No size or availability gate selects a replicated / gathered path: a missing pair-block binding, a whole ``[N, N]``
pair argument, ``B != 1``, a layout that disagrees with the rows, or MSA features that differ across ranks REFUSE BY NAME
(:class:`opt_core.mem.rowpair.RowpairRefused`). At ``--n_gpu 1`` nothing here is installed (the core's statements refuse a P == 1 layout by
name; :func:`install_msa_rows` is called by :func:`esmfold2_opt.rowpair.install` only at P > 1).

API:
    install_msa_rows(model, layout=None, census=None, *, pair_block=None) -> dict
                                        rebinds ``model.msa_encoder.forward`` to :func:`msa_encoder_rows` for this rank process (one record in
                                        :data:`esmfold2_opt.rowpair.PATCHES`); ``layout``: a Layout (fixed N), a zero-argument callable returning
                                        the running fold's Layout, or None = ``esmfold2_opt.rowpair.CTX.layout`` at call time; ``census``: a
                                        mapping the caller's LEVER line reads — the named facts of this module are written into it (and into the
                                        core's schedule census); ``pair_block``: the pair-block callable (default :func:`resolve_pair_block`)
    msa_encoder_rows(enc, x_pair_loc, *, x_inputs, msa_oh, has_deletion, deletion_value, msa_attention_mask, layout, pair_block, ...)
                                        ``MSAEncoder.forward`` with ``x_pair`` given as ROWS -> the ``msa_pair`` rows ``[B, R, N, 256]`` (the caller's
                                        ``x_pair_loc`` is not modified: the loop reuses ``z_init`` rows every step); ``pair_block(ops, z_rows
                                        [1, R, N, C], mask_rows [1, R, N], layout) -> z_rows``
    pair_ops(block, pt_rows=0)                      the block's three pair statements as ``ops(pair, pair_attention_mask=None) -> pair``
                                                    (``pt_rows`` > 0: the transition's row chunk the install set, checked per call)
    uninstall_msa_rows(model)                       drops the install mark and restores the blocks' ``PairTransition`` chunk sizes
    pair_transition_rows(environ=None)              the MSA blocks' transition row chunk from the environment (0 = as loaded)
    opm_add_rows_(opm, m, msa_mask, z, layout)      ``z[R, N, C] += OuterProductMean rows`` in place (returns the shard; adopt it)
    pwa_delta_rows(pwa, m, z, vis, layout)          the ``MSAPairWeightedAveraging`` m-update ``[B, N, M, d_msa]`` from pair ROWS
    msa_transition_replicated(tr, m, layout)        ``tr(m)`` (replicated, named)
    guard_msa_features(feats)                       the cross-rank guard of the MSA features (raises by name)
    pair_mask_rows(tok_mask, g0, g1)                rows ``[g0, g1)`` of ``tok_mask[:, :, None] & tok_mask[:, None, :]``
"""
from __future__ import annotations

import collections
import os
import sys
import types
from typing import Callable, Mapping, MutableMapping, Optional

from opt_core.mem.rowpair import RowpairRefused

__all__ = ["install_msa_rows", "uninstall_msa_rows", "msa_encoder_rows", "pair_ops", "opm_add_rows_", "pwa_delta_rows", "msa_transition_replicated",
           "guard_msa_features", "feature_fingerprint", "pair_mask_rows", "resolve_pair_block", "census_facts", "PAIR_BLOCK_NAME", "TRIMUL_DISPATCH", "REPLICATED",
           "P1_LEVERS_REPLACED", "P1_LEVERS_KEPT", "GUARDED_FEATURES",
           "pair_transition_rows", "pair_transition_word", "ENV_PAIR_TRANSITION_ROWS", "MSA_PAIR_TRANSITION_ROWS_DEFAULT", "MSA_TRANSITION_TOKENS", "PWA_BIAS_ROWS"]

PAIR_BLOCK_NAME = "pair_block_rows_"                                   # esmfold2_opt.rowpair.<name>(block, z_rows[B,R,N,C], mask_rows[B,R,N], layout=None) -> z_rows: the kit's ONE row
                                                                       # form of a stock pair block (the block is CALLED on the rows inside the sharded context)
TRIMUL_DISPATCH = "TriangleMultiplicativeBlock.forward"                # the install record (esmfold2_opt.rowpair.PATCHES) entry the MSA blocks' triangle multiplication on rows needs
REPLICATED = ("msa_m", "msa_features", "msa_transition")              # named: whole on every rank by design (O(N·M) tensors; the pair track is rows)
P1_LEVERS_REPLACED = ("t12", "t13", "m1")                        # forward-level single-GPU levers of the MSA path that do not run on rows (module docstring)
P1_LEVERS_KEPT = ("t11",)                                              # the msa_transition instance forward: the same replicated call as at n_gpu=1
V2_LEVERS_ON_ROWS = ("m15", "t15msa")                                  # the second-generation single-GPU MSA kernels the row form ISSUES ITSELF when their module has them on:
#   m15 = ef2_msa_v2.mtr_fused (msa_transition + residual: an m-only, per-(token, row) statement -> issued per token block on the replicated m, in place),
#   t15msa = ef2_pair_v2._pair_transition_residual_v2 (PairTransition + residual through the t15 kernel: row-local -> issued per local row block, in place).
#   m16 / m17 (fused PWA / OPM block-forward kernels over ONE L) do not run on rows: esmfold2_opt.rowpair.NOT_FOR_ROUTE names them.
GUARDED_FEATURES = ("msa_oh", "msa_attention_mask", "has_deletion", "deletion_value")   # integer-derived loop inputs guarded replicated per call
STATS = collections.Counter()                                           # m15_row_calls / m15_row_fallthrough / t15msa_row_calls: the composing kernels' per-process counters (census words)
MSA_TRANSITION_TOKENS = 256                                             # token rows per msa_transition call on the replicated m: the transition is a per-(token, sequence) statement, so
                                                                        #   token blocks give identical values while the 4x hidden it materialises ([rows·M, 2·4·d] + [rows·M, 4·d] bf16)
                                                                        #   is bounded by the block instead of growing with N on every rank (census msa_transition_tokens=)
PWA_BIAS_ROWS = 256                                                     # query rows per pair-weighted-averaging logits call (opt_core pwa_bias_rows rows=): LN(z rows) + linear_z are
                                                                        #   per-row statements (identical values); bounds their [rows, N, C_z] transient (census pwa_bias_rows=)
ENV_PWA_QBLOCK = "ROWPAIR_PWA_QBLOCK"                                  # the core's pin of the PWA query block; unset -> this module passes q_block=R (whole local rows)
ENV_PAIR_TRANSITION_ROWS = "EF2_ROWPAIR_MSA_TRANSITION_ROWS"          # rows per MSA-encoder PairTransition block on this rank's shard (its own dim-1 chunk_size); unset: the pair stack's variable, else MSA_PAIR_TRANSITION_ROWS_DEFAULT (pair_transition_rows); 0 = the model's chunk_size as loaded
MSA_PAIR_TRANSITION_ROWS_DEFAULT = 64                                   # the MSA blocks' own default: their PairTransition is the stock module (LayerNorm + SwiGLU with the 2x4C hidden
                                                                        #   materialised per row block, [rows, N, 2·4·C]), unlike the folding trunk's fused in-place update; a quarter of
                                                                        #   the trunk's block keeps that transient a quarter the size (per-row statements: identical values)


def pair_transition_rows(environ=None) -> int:
    """Rows per MSA-encoder ``PairTransition`` block on a rank's shard: ``EF2_ROWPAIR_MSA_TRANSITION_ROWS``, else the pair stack's
    ``EF2_ROWPAIR_TRANSITION_ROWS`` (``esmfold2_opt.rowpair.ENV_TRANSITION_ROWS``, the folding trunk's transition row block), else the MSA
    blocks' own default (``MSA_PAIR_TRANSITION_ROWS_DEFAULT``, 64); ``0`` keeps the model's ``chunk_size`` as loaded (``None`` = the whole shard). Row-local
    statements (LayerNorm + SwiGLU per row block, concatenated along the row dim): each element's arithmetic is the whole-shard call's;
    equality with it is bitwise where the stack's kernels do not change reduction order with the leading size (see NUMERICS above)."""
    from . import rowpair as RP                                           # the pair stack's variable name and default: one definition (rowpair.py)
    environ = os.environ if environ is None else environ
    for name in (ENV_PAIR_TRANSITION_ROWS, RP.ENV_TRANSITION_ROWS):
        v = str(environ.get(name, "") or "").strip()
        if v:
            try:
                n = int(v)
            except ValueError:
                n = -1
            if n < 0:
                raise RowpairRefused(f"{name}={v!r}: a non-negative integer is required (0 = the model's chunk_size as loaded)")
            return n
    return MSA_PAIR_TRANSITION_ROWS_DEFAULT


def pair_transition_word(pt_rows: int, n_blocks: int) -> str:
    """The census value of ``msa_pair_transition``: ``rows:<n>x<blocks>`` when the install set a row chunk, else ``whole``."""
    return f"rows:{int(pt_rows)}x{int(n_blocks)}" if int(pt_rows) > 0 else "whole"


# ================================================================================================================ census
def census_facts(**extra) -> dict:
    """The named facts of this module for the caller's LEVER line / schedule census (constant part + ``extra``)."""
    facts = {"msa": "rows", "msa_m": "replicated", "msa_features": "guarded", "msa_transition": "replicated", "msa_opm": "rows_inplace",
             "msa_pwa": "rows_softmax_local+allgather_out_rows", "msa_trimul": "reference_rows", "msa_z_gathers": 0,
             "msa_p1_levers_replaced": ",".join(P1_LEVERS_REPLACED), "msa_p1_levers_kept": ",".join(P1_LEVERS_KEPT)}
    facts.update(kit_kernel_words())                                       # msa_m15 / msa_t15msa: the v2 kernels the row form issues itself when their module has them on
    facts.update(extra)
    return facts


def _record(census: Optional[MutableMapping], **facts) -> None:
    """Write ``facts`` into the caller's census mapping (when given) and into the core's schedule census."""
    from opt_core.mem.rowpair import evidence as EV
    EV.record_schedule(**facts)
    if census is not None:
        if hasattr(census, "update"):
            census.update(facts)
        else:
            for k, v in facts.items():
                census[k] = v


# ================================================================================================================ row-local pieces
def kit_msa_kernels() -> dict:
    """The single-GPU MSA kernels this rank's model has ON that the row form issues itself (:data:`V2_LEVERS_ON_ROWS`): ``{"m15": <ef2_msa_v2 module
    or None>, "t15msa": <ef2_pair_v2 module or None>}`` — read off the modules' own lever records (``levers_on()``), never off an environment."""
    out = {"m15": None, "t15msa": None}
    v2 = sys.modules.get("ef2_msa_v2")
    try:
        if v2 is not None and "m15" in (v2.levers_on() or []) and callable(getattr(v2, "mtr_fused", None)):
            out["m15"] = v2
    except Exception:  # noqa: BLE001 — a module without the record API contributes nothing (named m15=absent)
        pass
    p2 = sys.modules.get("ef2_pair_v2")
    try:
        if p2 is not None and (p2.levers_on() or {}).get("t15msa") and callable(getattr(p2, "_pair_transition_residual_v2", None)):
            out["t15msa"] = p2
    except Exception:  # noqa: BLE001
        pass
    return out


def kit_kernel_words(kernels: Optional[dict] = None) -> dict:
    """Census words for the composing kernels: ``msa_m15=rows_inplace|off``, ``msa_t15msa=row_blocks_inplace|off``."""
    k = kernels if kernels is not None else kit_msa_kernels()
    return {"msa_m15": "token_blocks_inplace" if k.get("m15") is not None else "off",
            "msa_t15msa": "row_blocks_inplace" if k.get("t15msa") is not None else "off"}


def pair_mask_rows(tok_mask, g0: int, g1: int):
    """Rows ``[g0, g1)`` of ``MSAEncoder.forward``'s ``pair_attention_mask = tok_mask.unsqueeze(2) & tok_mask.unsqueeze(1)`` (bool ``[B, g1-g0, N]``)
    — a row-local statement; the whole ``[N, N]`` mask is never built."""
    return tok_mask[:, g0:g1].unsqueeze(2) & tok_mask.unsqueeze(1)


def guard_msa_features(feats: Mapping[str, object], enabled: Optional[bool] = None) -> dict:
    """GUARD: the replicated MSA inputs are identical on every rank: a position-weighted fingerprint of each tensor of :data:`GUARDED_FEATURES`
    present in ``feats`` (:func:`feature_fingerprint`) is compared across ranks with :func:`opt_core.mem.rowpair.dist.allreduce_checksum`; a
    mismatch raises :class:`RowpairRefused` naming the tensor (the ranks' RNG streams left lockstep before the MSA subsample / column mask — a
    broken run). The fingerprint is position-weighted because these tensors are one-hot / {0, 1}-valued: a plain sum or xor of their words is
    the same for ANY two subsamples of equal size. Returns ``{name: checksums}``; ``enabled=False`` (a test's argument) returns ``{}``
    and the census then says ``msa_features=unguarded``."""
    from opt_core.mem.rowpair import dist as RD
    if enabled is None:
        enabled = True
    if not enabled:
        return {}
    out = {}
    for name in GUARDED_FEATURES:
        t = feats.get(name)
        if t is not None:
            out[name] = RD.allreduce_checksum(feature_fingerprint(t), name=f"msa_features.{name}", raise_on_mismatch=True)
    return out


def feature_fingerprint(t):
    """``[B, N]`` float64: per token row ``i``, ``sum_k (k + 1) * t[b, i, k]`` over the row's flattened trailing elements ``k`` — exact for one-hot /
    {0, 1} / float32-valued features (every product and partial sum is an integer or a float32 value scaled by a small integer, far inside float64's
    exact range at MSA sizes), and sensitive to WHERE each value sits, not only to how many there are."""
    import torch
    x = t.detach()
    if x.dtype == torch.bool:
        x = x.to(torch.uint8)
    x = x.reshape(int(x.shape[0]), int(x.shape[1]), -1).to(torch.float64)
    w = torch.arange(1, int(x.shape[-1]) + 1, dtype=torch.float64, device=x.device)
    return (x * w).sum(dim=-1)


def opm_add_rows_(opm, m, msa_mask, z, layout, *, rows: Optional[int] = None, budget_bytes: Optional[int] = None):
    """``z[R, N, C] += OuterProductMean(m, msa_attention_mask)`` restricted to this rank's rows, IN PLACE (the engine's ``pair = pair + opm(...)``).
    The m-side statements are ``OuterProductMean.forward``'s own on the replicated ``m`` (modeling_esmfold2_common.py:2760-2764); per row block
    ``[g0, g1)``: ``n_valid = (mask_f[:, g0:g1] @ mask_fᵀ).clamp(min=1)``, ``outer = einsum('bimc,bjmd->bijcd', a[:, g0:g1], b).flatten(-2)``,
    ``Wout`` and ``/ n_valid`` in the checkpoint's order — bound to :func:`opt_core.mem.rowpair.msa.opm_rows_budgeted` (row block: ``rows`` -> ``given``; else budgeted from the AGREED free bytes, aligned to the
    module's ``_chunk_size`` grid, printed). The dense statement is out of place (``pair = pair + delta`` promotes to ``result_type(pair, delta)``):
    when ``z``'s dtype is narrower than that, ``z`` is converted ONCE (a one-row probe of the OPM statements decides) and the converted shard is the one
    accumulated into and RETURNED — callers adopt the return value."""
    import torch
    from opt_core.mem.rowpair import msa as RM
    if m.dim() != 4 or int(m.shape[0]) != 1:
        raise RowpairRefused(f"opm_add_rows_: m {tuple(m.shape)} must be [1, N, M, d_msa] (B == 1)")
    if z.dim() != 3 or int(z.shape[0]) != layout.R or int(z.shape[1]) != layout.N:
        raise RowpairRefused(f"opm_add_rows_: z {tuple(z.shape)} is not this rank's shard [{layout.R}, {layout.N}, C]")
    m_norm = opm.norm(m)
    x = opm.W(m_norm) * msa_mask.unsqueeze(-1).to(m_norm.dtype)
    del m_norm
    a, b = x.chunk(2, dim=-1)                                              # [1, N, M, d_hidden] each (views of x)
    mask_f = msa_mask.to(a.dtype)                                          # [1, N, M]
    divide_first = bool(getattr(opm, "divide_outer_before_proj", False))

    def outer_fn(a_blk, b_all, g0: int, g1: int):                          # the OPM rows [g0, g1): [rows, N, C_z], token rows on dim 0
        n_valid = (mask_f[:, g0:g1] @ mask_f.transpose(-1, -2)).unsqueeze(-1).clamp(min=1.0)
        outer = torch.einsum("bimc,bjmd->bijcd", a_blk, b_all).flatten(-2)
        if divide_first:
            blk = opm.Wout(outer / n_valid)
        else:
            blk = opm.Wout(outer) / n_valid
        del outer, n_valid
        return blk[0]

    N, dh, C_z = int(a.shape[1]), int(a.shape[-1]), int(opm.Wout.out_features)
    rt = torch.promote_types(z.dtype, outer_fn(a.narrow(1, layout.r0, 1), b, layout.r0, layout.r0 + 1).dtype)   # the dense `pair + delta` dtype (one-row probe)
    if rt != z.dtype:
        z = z.to(rt)
    elt = int(a.element_size())
    bytes_per_row = N * elt * (2 * dh * dh + 2 * C_z)                      # outer [N, dh·dh] (+ its normalised copy) + the projected rows (+ the divided copy)
    align = getattr(opm, "_chunk_size", None)                              # the module's chunk grid over the left (i) axis (None: unchunked -> the layout's align)
    RM.opm_rows_budgeted(a, b, layout, outer_fn, rows, C_z=C_z, out=z, add=True, align=align, global_rows=True, row_dim=1,
                         bytes_per_row=bytes_per_row, budget_bytes=budget_bytes)
    del a, b, x, mask_f
    return z


def pwa_delta_rows(pwa, m, z, vis, layout, *, q_block: Optional[int] = None):
    """The m-update of ``MSAPairWeightedAveraging.forward`` (modeling_esmfold2_common.py:2801-2824) with the pair given as ROWS: ``bias =
    compute_bias(pair rows) [1, R, N, h]`` masked by the visibility rows, ``softmax over j`` (row-local), ``out_i = einsum('bijh,bjmhd,bimhd->bimhd',
    attn rows, v, gate rows)`` against the replicated ``v`` / ``gate`` (``norm_single``, ``Wv``, ``Wgate`` on the whole ``m``: stock shapes), the out
    rows ALL-GATHERED over token rows, ``Wout`` at the stock shape — bound to :func:`opt_core.mem.rowpair.msa.pwa_bias_rows` /
    :func:`~opt_core.mem.rowpair.msa.pwa_rows`. ``vis``: the bool visibility rows ``[1, R, N]``.
    Returns ``delta [1, N, M, d_msa]`` (a view; the caller adds it: ``m = m + delta``)."""
    import torch
    from opt_core.mem.rowpair import msa as RM
    if m.dim() != 4 or int(m.shape[0]) != 1:
        raise RowpairRefused(f"pwa_delta_rows: m {tuple(m.shape)} must be [1, N, M, d_msa] (B == 1)")
    B_, L, _M, _d = (int(v) for v in m.shape)
    h, dh = int(pwa.n_heads), int(pwa.head_width)
    r0 = layout.r0

    def prep_fn(z_rows, g0: int, g1: int):                                 # logits of query rows [g0, g1): stock layout [1, rows, N, h] -> the core's [h, rows, N] VIEW
        bias = pwa.compute_bias(z_rows.unsqueeze(0))
        bias.masked_fill_(~vis[:, g0 - r0:g1 - r0].unsqueeze(-1).bool(), -1e5)
        return bias[0].permute(2, 0, 1)

    bias_shard = RM.pwa_bias_rows(prep_fn, z, layout, rows=PWA_BIAS_ROWS) # [h, R, N], assembled from PWA_BIAS_ROWS-row pieces (per-row statements)

    def softmax_fn(logits):                                                # softmax over j on the STOCK layout [rows, N, h] (dim -2), handed on as the [h, rows, N] view
        return torch.softmax(logits.permute(1, 2, 0), dim=-2).permute(2, 0, 1)

    def values_fn(m_chunk):                                                # m_chunk: [Mc, N, d] view -> the stock statements on [1, N, Mc, d]
        mc = m_chunk.transpose(0, 1).unsqueeze(0)
        Mc = int(mc.shape[2])
        msa_normed = pwa.norm_single(mc)
        v = pwa.Wv(msa_normed).reshape(B_, L, Mc, h, dh)
        gate = torch.sigmoid(pwa.Wgate(msa_normed)).reshape(B_, L, Mc, h, dh)
        del msa_normed
        return v, gate

    def attend_fn(w, state, g0: int, g1: int):                             # -> [Mc, q, h·dh] (the core gathers token rows on dim -2)
        v, gate = state
        attn = w.permute(1, 2, 0).unsqueeze(0)                             # back to the stock [1, q, N, h]
        ob = torch.einsum("bijh,bjmhd,bimhd->bimhd", attn, v, gate[:, g0:g1])
        q = g1 - g0
        return ob[0].reshape(q, int(v.shape[2]), h * dh).transpose(0, 1)

    def out_fn(o_full):                                                    # o_full [Mc, N, h·dh] -> Wout at the stock shape [1, N, Mc, h·dh] -> [Mc, N, d_msa] view
        y = pwa.Wout(o_full.transpose(0, 1).unsqueeze(0))
        return y[0].transpose(0, 1)

    if q_block is None and not os.environ.get(ENV_PWA_QBLOCK, "").strip():
        q_block = layout.R                                                 # whole local rows: this engine materialises [q, N, h] + [q, M, h·dh] per block (printed pwa_qblock_source=given)
    m_core = m[0].transpose(0, 1)                                          # [M, N, d] view (sequence-major, as the core walks it)
    delta = RM.pwa_rows(m_core, bias_shard, layout, values_fn=values_fn, attend_fn=attend_fn, out_fn=out_fn, softmax_fn=softmax_fn, s_chunk=None,
                        q_block=q_block)                                   # [M, N, d_msa] in the statement's own dtype
    del bias_shard
    return delta.transpose(0, 1).unsqueeze(0)                              # the [1, N, M, d_msa] view the block adds to m


def msa_transition_replicated(tr, m, layout):
    """``msa_transition(m)`` on the replicated ``m`` — the engine's own call (:func:`opt_core.mem.rowpair.msa.msa_transition_rows`,
    ``shard_tokens=False``: exact by construction, ``msa_transition=replicated``). Returns the delta (the block adds it)."""
    from opt_core.mem.rowpair import msa as RM
    from opt_core.mem.torch_rowchunk import rowchunk_apply

    def whole(mm, _t0, _t1):                                               # the engine's transition on the whole replicated m, issued per token block: a per-(token,
        n_tok = int(mm.shape[1])                                           #   sequence) statement -> identical values; the transient is the block's, not N's
        RM.record_schedule(msa_transition_tokens=int(min(n_tok, MSA_TRANSITION_TOKENS)))
        if n_tok <= MSA_TRANSITION_TOKENS:
            return tr(mm)
        return rowchunk_apply(lambda i0, i1: tr(mm[:, i0:i1]), n_tok, MSA_TRANSITION_TOKENS, row_dim=1, lever="rowpair")
    return RM.msa_transition_rows(whole, m, layout, shard_tokens=False, token_dim=1)


def msa_transition_residual_replicated_(tr, m, layout, kernels: Optional[dict] = None):
    """``m = m + msa_transition(m)`` on the replicated ``m``. With ef2_msa_v2's lever m15 on (``kernels["m15"]``) the module's one-kernel
    ``mtr_fused`` (transition + residual, stock rounding points) is issued per block of :data:`MSA_TRANSITION_TOKENS` tokens IN PLACE over ``m``
    (a per-(token, row) statement: identical values whatever the blocking; ``msa_m15=token_blocks_inplace``, counted ``m15_row_calls``); a call the
    kernel does not serve (its own ``_mtr_ok``) takes the replicated stock statement and is counted ``m15_row_fallthrough``. Without m15: the
    engine's own transition (:func:`msa_transition_replicated`) and the residual add. Returns the updated ``m`` (may alias the argument)."""
    v2 = (kernels or {}).get("m15")
    if v2 is None:
        return m + msa_transition_replicated(tr, m, layout)
    ok = getattr(v2, "_mtr_ok", None)
    n_tok = int(m.shape[1])
    blk = max(1, int(MSA_TRANSITION_TOKENS))
    blocks_contiguous = m.is_contiguous() and (int(m.shape[0]) == 1 or n_tok <= blk)   # the kernel takes contiguous blocks: a token slice m[:, i0:i1] of a
    if not blocks_contiguous or (callable(ok) and not ok(tr, m)):                        # batch B > 1 is not contiguous (one block spanning every token is) —
        STATS["m15_row_fallthrough"] += 1                                                # such a call takes the replicated stock statement, counted
        return m + msa_transition_replicated(tr, m, layout)
    from opt_core.mem.rowpair import evidence as EV
    EV.record_schedule(msa_transition_tokens=int(min(n_tok, blk)), msa_m15="token_blocks_inplace")
    for i0 in range(0, n_tok, blk):                                          # rows of m are independent under the kernel: out may alias m (ef2_msa_v2.mtr_fused docstring)
        mb = m[:, i0:min(n_tok, i0 + blk)]
        v2.mtr_fused(tr, mb, out=mb)
        STATS["m15_row_calls"] += 1
    return m


def pair_transition_fused_rows_(p2, module, pair, rows: int):
    """``pair += PairTransition(pair)`` on this rank's shard through ef2_pair_v2's t15 kernel (lever t15msa): ``_pair_transition_residual_v2``
    (transition + residual fused, out of place per block) per local row block of ``rows`` rows, written back IN PLACE
    (``opt_core.mem.rowpair.pairstack.transition_update_``, ``add=False``) — row-local, no communication; a block the kernel does not serve takes
    the module's stock ``pair + pt(pair)`` inside that same function. Counted ``t15msa_row_calls``."""
    from . import rowpair as RPK
    from opt_core.mem.rowpair import pairstack as RPS
    lay = RPK.CTX.layout
    if not RPK._is_shard_of(pair, lay) or not pair.is_contiguous():
        raise RowpairRefused(f"pair_transition_fused_rows_: pair {tuple(pair.shape)} is not this rank's contiguous shard inside the sharded context")
    for b in range(int(pair.shape[0])):
        RPS.transition_update_(lambda zb, mb: p2._pair_transition_residual_v2(module, zb.unsqueeze(0))[0], pair[b], None, max(1, int(rows)), add=False)
        STATS["t15msa_row_calls"] += 1
    return pair


# ================================================================================================================ the pair block (the kit's row form; this module's statements)
def resolve_pair_block() -> Callable:
    """The kit's row form of a pair block, ``esmfold2_opt.rowpair.PAIR_BLOCK_NAME(block, z_rows, mask_rows, layout=None) -> z_rows``. Absent ->
    refused BY NAME (no triangle op of the MSA encoder ever runs on a shard through the dense statement)."""
    from . import rowpair as RP
    fn = getattr(RP, PAIR_BLOCK_NAME, None)
    if fn is None:
        raise RowpairRefused(f"rowpair_msa: esmfold2_opt.rowpair.{PAIR_BLOCK_NAME} is required (the kit's one row form of a pair block: the block called on "
                             "this rank's rows inside the sharded context); the MSA encoder's pair blocks have no other sharded form")
    return fn


def pair_ops(block, pt_rows: int = 0, kernels: Optional[dict] = None) -> Callable:
    """The three pair statements of ``MSAEncoderBlock.forward`` (modeling_esmfold2.py) as one ``ops(pair, pair_attention_mask=None) -> pair``
    callable — the shape :data:`PAIR_BLOCK_NAME` calls on the rows: ``pair = pair + block.tri_mul_out(pair, mask=m)``; ``+ block.tri_mul_in``;
    ``+ block.pair_transition(pair)``. The block's own sub-modules run (``TriangleMultiplicativeUpdate`` -> ``TriangleMultiplicativeBlock``,
    ``PairTransition``); inside the sharded context the class-level ``TriangleMultiplicativeBlock.forward`` dispatcher gives the triangle
    multiplication its row form and the transition is row-local. ``pt_rows > 0``: the row chunk :func:`install_msa_rows` set on
    ``block.pair_transition`` (its dim-1 ``_chunk_size``; dim 1 of the ``[B, R, N, C]`` rows IS the row dim) must still be in place when the
    statement runs — a later ``set_chunk_size`` (the model-level call cascades into every MSA block) is refused BY NAME, never run whole."""
    pt_rows = int(pt_rows or 0)
    p2 = (kernels or {}).get("t15msa")                                      # ef2_pair_v2 with lever t15msa on: the PairTransition residual through its kernel per row block

    def ops(pair, pair_attention_mask=None):                              # `pair` is the encoder's OWN row copy (msa_encoder_rows pair_loc): the three residual
        from . import rowpair as RPK                                        #   statements accumulate into it in place — no update copy, no `pair + …` result
        if pt_rows > 0:
            held = getattr(block.pair_transition, "_chunk_size", None)
            if held != pt_rows:
                raise RowpairRefused(f"msa_pair_ops: pair_transition._chunk_size={held!r} but install_msa_rows set rows:{pt_rows} (a set_chunk_size call after "
                                     f"the install changed it); refused — the MSA-encoder PairTransition does not run on the whole [R, N, 4·c_z] shard")
        if not (RPK.CTX.active and RPK.CTX.layout is not None):               # outside a sharded fold: the block's own three statements, as written
            pair = pair + block.tri_mul_out(pair, mask=pair_attention_mask)
            pair = pair + block.tri_mul_in(pair, mask=pair_attention_mask)
            return p2._pair_transition_residual_v2(block.pair_transition, pair) if p2 is not None else pair + block.pair_transition(pair)
        RPK.trimul_residual_rows_(block.tri_mul_out, pair, pair_attention_mask)   # pair += tri_mul_out(pair)
        RPK.trimul_residual_rows_(block.tri_mul_in, pair, pair_attention_mask)    # pair += tri_mul_in(pair)
        if p2 is not None:                                                         # pair += pair_transition(pair) per row block: through the t15 kernel (lever t15msa) ...
            pair_transition_fused_rows_(p2, block.pair_transition, pair, pt_rows or MSA_PAIR_TRANSITION_ROWS_DEFAULT)
        else:                                                                      # ... or the module's own forward
            RPK.transition_residual_rows_(block.pair_transition, pair, pt_rows or MSA_PAIR_TRANSITION_ROWS_DEFAULT)
        return pair
    ops.__name__ = "msa_pair_ops"
    ops.__qualname__ = "pair_ops.<msa_pair_ops>"
    return ops


def _dispatch_checked(pair_block: Callable) -> Callable:
    """``pair_block`` guarded by the install record: the MSA blocks' triangle multiplication runs through their ``TriangleMultiplicativeBlock``
    modules, so :data:`TRIMUL_DISPATCH` must be among :data:`esmfold2_opt.rowpair.PATCHES` when the rows reach it — else refused BY NAME (the
    dense statement would otherwise contract over this rank's rows only)."""
    def checked(ops, z_rows, mask_rows, layout):
        from . import rowpair as RP
        names = RP.PATCHES.names()
        if not any(n == TRIMUL_DISPATCH or n.endswith("." + TRIMUL_DISPATCH) for n in names):
            raise RowpairRefused(f"rowpair_msa: {TRIMUL_DISPATCH} is not in the lever's install record ({len(names)} patches) — the MSA encoder's "
                                 "tri_mul_out / tri_mul_in engines are TriangleMultiplicativeBlock modules under every kernel backend and need its row "
                                 "dispatcher installed (esmfold2_opt.rowpair.install) before a fold with an MSA reaches them")
        return pair_block(ops, z_rows, mask_rows, layout)
    checked.__name__ = getattr(pair_block, "__name__", PAIR_BLOCK_NAME)
    return checked


# ================================================================================================================ the encoder on rows
def msa_encoder_rows(enc, x_pair_loc, *, x_inputs, msa_oh, has_deletion, deletion_value, msa_attention_mask, layout, pair_block: Callable,
                     census: Optional[MutableMapping] = None, opm_rows: Optional[int] = None, opm_budget_bytes: Optional[int] = None,
                     q_block: Optional[int] = None, guard: Optional[bool] = None, probe: Optional[Callable] = None):
    """``MSAEncoder.forward`` (modeling_esmfold2.py:1187-1201) with ``x_pair`` given as this rank's ROWS ``[1, R, N, 256]`` and every other input
    replicated exactly as ``ESMFold2Model._run_one_loop`` builds it (``[1, N, d_inputs]``, ``[1, N, M, 33]``, ``[1, N, M]`` x3); returns the
    ``msa_pair`` rows ``[1, R, N, 256]`` — ``MSAEncoder.forward`` / ``MSAEncoderBlock.forward`` statement for statement with every pair
    statement bound to :mod:`opt_core.mem.rowpair.msa` on this rank's rows. ``pair_block(ops, z_rows [1, R, N, C], mask_rows [1, R, N] bool,
    layout) -> z_rows``: the kit's row form of a pair block (:func:`resolve_pair_block`) called with :func:`pair_ops` of each block;
    ``probe(k, tag, obj)``: test hook after each sub-op of block ``k`` (``tag`` in ``m0 opm pwa mtr blk``)."""
    import torch
    from opt_core.mem.rowpair import dist as RD
    enc = getattr(enc, "msa_encoder", enc)
    lay = RD.require_sharded(layout, "msa_encoder_rows")
    if x_pair_loc.dim() != 4 or int(x_pair_loc.shape[0]) != 1:
        raise RowpairRefused(f"msa_encoder_rows: x_pair {tuple(x_pair_loc.shape)} must be [1, R, N, C] (B == 1: one item per fold)")
    R, N = int(x_pair_loc.shape[1]), int(x_pair_loc.shape[2])
    if N != lay.N or R != lay.R:
        whole = " — a WHOLE pair reached the MSA encoder under n_gpu>1 (the loop must hand its z_inject rows)" if R == N else ""
        raise RowpairRefused(f"msa_encoder_rows: x_pair rows {tuple(x_pair_loc.shape)} vs layout R={lay.R} N={lay.N} rank={lay.rank}{whole}")
    if int(msa_oh.shape[1]) != N or int(msa_attention_mask.shape[1]) != N:
        raise RowpairRefused(f"msa_encoder_rows: msa_oh {tuple(msa_oh.shape)} / msa_attention_mask {tuple(msa_attention_mask.shape)} vs N={N} "
                             "([1, N, M, ...] token-major inputs expected)")
    proof = guard_msa_features({"msa_oh": msa_oh, "msa_attention_mask": msa_attention_mask, "has_deletion": has_deletion, "deletion_value": deletion_value},
                               enabled=guard)
    # ---- m = embed(cat[msa_oh, hd, dv]) + project_inputs(x_inputs)[:, :, None]       (replicated by design; MSAEncoder.forward verbatim)
    m_feat = torch.cat([msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)
    m = enc.embed(m_feat) + enc.project_inputs(x_inputs).unsqueeze(2)
    del m_feat
    tok_mask = msa_attention_mask[:, :, 0].bool()                            # [1, N]
    vis = pair_mask_rows(tok_mask, lay.r0, lay.r1)                           # [1, R, N] bool: this rank's rows of pair_attention_mask
    if probe is not None:
        probe(-1, "m0", m)
    # ---- pair = x_pair + ...  (x_pair itself stays intact: the loop reuses the z_init rows every step; the OPM rows accumulate in place into the
    #      copy, converted once to the dense sum's dtype by opm_add_rows_ when that is wider)
    pair_loc = x_pair_loc.clone(memory_format=torch.contiguous_format)
    n_blocks = 0
    pt_rows = int(vars(enc).get("_ef2_rowpair_msa_pt_rows", 0) or 0)      # the PairTransition row chunk install_msa_rows set (0: not installed here, or the model's chunk as loaded)
    kernels = kit_msa_kernels()                                              # the single-GPU kernels on this model the row form issues itself (m15, t15msa), read per call
    for k, block in enumerate(enc.blocks):
        z = opm_add_rows_(block.outer_product_mean, m, msa_attention_mask, pair_loc[0], lay, rows=opm_rows, budget_bytes=opm_budget_bytes)
        if z.data_ptr() != pair_loc.data_ptr():                             # converted to the dense sum's dtype: adopt
            pair_loc = z.unsqueeze(0)
        del z
        if probe is not None:
            probe(k, "opm", pair_loc)
        if not block.is_final_block:
            delta = pwa_delta_rows(block.msa_pair_weighted_averaging, m, pair_loc[0], vis, lay, q_block=q_block)
            m = m + delta
            del delta
            if probe is not None:
                probe(k, "pwa", m)
            m = msa_transition_residual_replicated_(block.msa_transition, m, lay, kernels)   # m + msa_transition(m): m15's kernel per token block in place, else the engine's own
            if probe is not None:
                probe(k, "mtr", m)
        pair_out = pair_block(pair_ops(block, pt_rows, kernels), pair_loc, vis, lay)  # pair + tri_mul_out; + tri_mul_in; + pair_transition — the stock sub-modules on the rows (the transition through t15msa's kernel when on; its row chunk checked per call)
        if pair_out.dim() != 4 or tuple(pair_out.shape) != tuple(pair_loc.shape):
            raise RowpairRefused(f"msa_encoder_rows: the pair block returned {tuple(pair_out.shape)} for the rows {tuple(pair_loc.shape)}")
        pair_loc = pair_out
        del pair_out
        if probe is not None:
            probe(k, "blk", pair_loc)
        n_blocks += 1
    del m
    _record(census, **census_facts(msa_blocks=n_blocks, msa_features="guarded" if proof else "unguarded", msa_depth=int(msa_oh.shape[2]),
                                   msa_pair_transition=pair_transition_word(pt_rows, n_blocks), **kit_kernel_words(kernels),
                                   **{k_: int(v_) for k_, v_ in STATS.items()}))
    return pair_loc


# ================================================================================================================ install
def _resolve_layout(spec, N: int):
    """The Layout of the running fold: ``spec`` (a Layout / a zero-argument callable / None = esmfold2_opt.rowpair.CTX.layout)."""
    lay = spec() if callable(spec) else spec
    if lay is None:
        from . import rowpair as RP
        lay = getattr(getattr(RP, "CTX", None), "layout", None)
    if lay is None:
        raise RowpairRefused("rowpair_msa: no Layout for this fold (esmfold2_opt.rowpair.CTX.layout is unset and install_msa_rows was given none)")
    if int(lay.N) != int(N):
        raise RowpairRefused(f"rowpair_msa: the fold's Layout has N={lay.N} but the pair rows have N={N}")
    return lay


def install_msa_rows(model, layout=None, census: Optional[MutableMapping] = None, *, pair_block: Optional[Callable] = None) -> dict:
    """Rebind ``model.msa_encoder.forward`` so the loop's ``self.msa_encoder(x_pair=<z_inject ROWS>, ...)`` statement returns ``msa_pair`` ROWS
    (:func:`msa_encoder_rows`); the original forward is kept in :data:`esmfold2_opt.rowpair.PATCHES` (restored by ``rowpair.uninstall``). A model
    without an MSA encoder (variants fast / full_nomsa) or in train() mode is refused by name. ``layout`` / ``census`` / ``pair_block``: module
    docstring. Writes the module's named facts into ``census`` and returns them (``msa_graphs_off``: an ef2_opt CUDA-graph wrapper found on the
    encoder's forward and superseded; ``msa_pair_block``: the row form's name; ``msa_m1_blocks``: blocks whose fused-trimul block forward (ef2_opt
    M1) is bypassed on rows; ``msa_trimul=reference_rows``)."""
    enc = getattr(model, "msa_encoder", None)
    if enc is None:
        raise RowpairRefused("install_msa_rows: the model has no msa_encoder (variant full_msa only; fast / full_nomsa install nothing here)")
    if getattr(model, "training", False):
        raise RowpairRefused("install_msa_rows: the model is in train() mode; n_gpu>1 runs inference only")
    if getattr(enc, "_ef2_rowpair_msa", False):
        raise RowpairRefused("install_msa_rows: already installed on this msa_encoder (one install per rank process)")
    from . import rowpair as RP
    pb = pair_block if pair_block is not None else _dispatch_checked(resolve_pair_block())   # a caller-supplied pair_block is used as given
    pt_rows = pair_transition_rows()                                                         # the blocks' PairTransition walks this rank's rows in blocks (its own dim-1 chunk): the SwiGLU
    if pt_rows > 0:                                                                           # hidden is [rows, N, 4·c_z], not [R, N, 4·c_z] (chunk_size=None as loaded leaves it whole)
        for k, b in enumerate(enc.blocks):                                                   # every block or none: refused BY NAME before anything is bound (fail-closed; =0 opts out by name)
            pt = getattr(b, "pair_transition", None)
            if pt is None or not callable(getattr(pt, "set_chunk_size", None)) or not hasattr(pt, "_chunk_size"):
                raise RowpairRefused(f"install_msa_rows: msa_encoder.blocks[{k}] has no pair_transition with set_chunk_size/_chunk_size "
                                     f"(modeling_esmfold2.PairTransition); the row chunk rows:{pt_rows} cannot be set — refused "
                                     f"({ENV_PAIR_TRANSITION_ROWS}=0 keeps the model's chunk_size as loaded, by name)")
    graphs_off = bool(getattr(enc, "_ef2opt_graphed", False) and "forward" in vars(enc))
    m1 = sum(1 for b in enc.blocks if getattr(b, "_ef2opt_fused", False))                   # blocks carrying the kit's fused-trimul block forward (does not run on rows)

    def forward(self, x_pair, x_inputs, msa_oh, has_deletion, deletion_value, msa_attention_mask):
        if getattr(RP.CTX, "below_floor", False):                                           # the fold is below the sharding floor: the kept encoder, whole
            return RP.PATCHES.original(self, "forward")(x_pair=x_pair, x_inputs=x_inputs, msa_oh=msa_oh, has_deletion=has_deletion,
                                                       deletion_value=deletion_value, msa_attention_mask=msa_attention_mask)
        lay = _resolve_layout(layout, int(x_pair.shape[-2]))
        return msa_encoder_rows(self, x_pair, x_inputs=x_inputs, msa_oh=msa_oh, has_deletion=has_deletion, deletion_value=deletion_value,
                                msa_attention_mask=msa_attention_mask, layout=lay, pair_block=pb, census=census)

    bound = types.MethodType(forward, enc)
    bound.__func__.__wrapped_forward__ = getattr(enc, "_ef2opt_eager_forward", None) or enc.forward
    pt_set = 0
    if pt_rows > 0:                                                                           # the chunks first (every block has the module: checked above), the bind after: a failing
        prev = []                                                                             # set_chunk_size leaves every block as it was and nothing bound
        try:
            for b in enc.blocks:
                prev.append(b.pair_transition._chunk_size)
                b.pair_transition.set_chunk_size(int(pt_rows)); pt_set += 1
        except Exception:
            for b, chunk in zip(enc.blocks, prev):
                b.pair_transition.set_chunk_size(chunk)
            raise
        enc.__dict__["_ef2_rowpair_msa_pt_prev"] = prev                                     # put back by uninstall_msa_rows
    RP._patch(enc, "forward", bound, "model.msa_encoder")
    enc.__dict__["_ef2_rowpair_msa"] = True
    enc.__dict__["_ef2_rowpair_msa_pt_rows"] = int(pt_rows)                                  # read per call by msa_encoder_rows -> pair_ops: the chunk must still hold when the statement runs
    facts = census_facts(msa_blocks=len(enc.blocks), msa_graphs_off=int(graphs_off), msa_pair_block=getattr(pb, "__name__", PAIR_BLOCK_NAME),
                         msa_m1_blocks=m1, msa_guard=1, msa_replicated=",".join(REPLICATED),
                         msa_pair_transition=pair_transition_word(pt_rows, pt_set))
    _record(census, **facts)
    return facts


def uninstall_msa_rows(model) -> bool:
    """Drop this module's install mark and put the MSA blocks' ``PairTransition`` chunk sizes back as they were before the install (the forward
    itself is restored with every other patch of the lever by ``esmfold2_opt.rowpair.uninstall`` = ``PATCHES.restore``). Returns whether a mark
    was present."""
    enc = getattr(model, "msa_encoder", None)
    if enc is None:
        return False
    prev = enc.__dict__.pop("_ef2_rowpair_msa_pt_prev", None)
    if prev is not None:
        for b, chunk in zip(enc.blocks, prev):
            b.pair_transition.set_chunk_size(chunk)
    enc.__dict__.pop("_ef2_rowpair_msa_pt_rows", None)
    return bool(enc.__dict__.pop("_ef2_rowpair_msa", False))
