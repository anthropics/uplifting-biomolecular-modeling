"""The MSA module under row sharding — the pair track ``z`` stays SHARDED through the MSA module (no all-gather of ``z`` at its boundary).
The MSA representation has TWO named layouts (``m_layout``, the adapter's explicit argument): ``replicated`` (default) — every rank holds
``m[S, N, c_m]`` (``S <= 1024`` rows: ``S·N·c_m·4`` B = 8.2 GB at N = 31,140, c_m = 64) — and ``token_sharded`` — rank q holds
``m[:, r0_q:r1_q, :]`` for deep MSAs (S = 16,384: 4.2 MB per token per rank replicated, 42 GB at N = 10,000), where only the NON-LOCAL
operand of each statement is all-gathered (the OPM's ``b``, the pair-weighted averaging's values source) and the m-updates stay on the
token shard (S-chunking of ``m`` across ranks is not a mechanism of this module). Every statement keeps each reduction
WHOLE on one rank: the contraction over sequences of the outer-product mean per ``(i, j)``, the softmax over ``j`` of the pair-weighted
averaging per query row ``i``, LayerNorm statistics per row — sharding changes only the M (row count) of row-local launches, never splits
a sum across ranks (numerics class of :mod:`opt_core.mem.rowpair.transition`: per-element identical to the dense statement; bit-exact iff the
stack's kernels are M-invariant). Engine sub-modules enter as CALLABLES; this module imports no engine.

    opm_rows_budgeted(a, b, layout, outer_fn, rows=None, *, C_z, out=, add=, align=, ...)
                                        :func:`opt_core.mem.rowpair.transition.opm_rows` (OPM OUTPUT rows: ``a[:, rows] x b[:, all]``) with the
                                        row block CHOSEN, ALIGNED and PRINTED here: ``rows`` -> ``opm_rows_source=given``; ``ROWPAIR_OPM_ROWS`` ->
                                        ``env``; else the ``ROWPAIR_ROWBLK_MB`` (512 MiB) target for the ``[rows, N, 2·C_z]`` output transient ->
                                        ``rowblk``, lowered to :data:`OPM_FRAC` of the AGREED free bytes (:func:`dist.agreed_free_bytes`) when
                                        that binds -> ``budget``; every choice rounded DOWN to a multiple of ``align`` (>= ``align``: the engine's
                                        chunk, so an engine statement chunking ``a_blk`` itself walks its global chunk grid) and kept below 2**31
                                        elements per launch. ``add=True, out=z_shard``: the engine's ``z += OPM(m)`` lands IN PLACE per row block.
    default_opm_rows(N, C_z, Rmax, ...) the ``(rows, source)`` rule above (pure)
    pwa_bias_rows(prep_fn, z_shard, layout, rows=None)
                                        the pair-weighted-averaging LOGITS of this rank's query rows, ``[H, R, N]`` = ``prep_fn(z_rows, g0, g1)``
                                        (engine: ``linear_z(LN(z_rows))`` permuted + the mask bias of rows ``[g0, g1)``) — row-local; evaluated in
                                        row blocks of ``rows`` only when the shard reaches ``ROWPAIR_LN_GUARD_ELEMS`` (2**31) elements (the
                                        LayerNorm int32 guard), else ONE call (byte-identical to the unblocked statement)
    pwa_rows(m, bias_shard, layout, *, values_fn, attend_fn, out_fn, softmax_fn=, s_chunk=, q_block=, out=)
                                        MSA pair-weighted averaging with ``z`` given as ROWS -> the replicated ``m``-update ``[S, N, c_out]``. The
                                        engine's chunk grid over sequences (``s_chunk``) is walked explicitly so EVERY m-side GEMM keeps the
                                        dense M: ``state = values_fn(m_chunk [chunk, N, c_m])`` (LN_m, linear_v, sigmoid(linear_g) on the full
                                        slab), the softmax over ``j`` of the LOCAL query rows (complete rows: no communication; ``softmax_fn`` =
                                        the engine's kernel route), ``o_rows = attend_fn(w [H, q, N], state, g0, g1) -> [chunk, q, F]`` (the
                                        engine's weights x values einsum in its own operand layout, gate rows, flatten — M = q local rows
                                        instead of N: the row-sharding premise), ONE all_gather of the token rows per S-chunk, then
                                        ``out_fn(o_full [chunk, N, F])`` (linear_o on the GATHERED slab: dense M). ``q_block``
                                        (``ROWPAIR_PWA_QBLOCK``; auto: whole unless ``chunk·H·R·N·elt`` exceeds ``ROWPAIR_PWA_MAX_GB`` GiB (8)
                                        -> 256 rows) bounds the per-block transient — softmax rows are independent, so any ``q_block`` is
                                        byte-identical; printed as ``pwa_qblock`` / ``pwa_qblock_source``.
    msa_transition_rows(fn, m, layout, *, shard_tokens, token_dim=-2)
                                        the MSA transition: ``shard_tokens=False`` -> REPLICATED (``fn(m, 0, N)`` — the engine's own call, exact by
                                        construction; printed ``msa_transition=replicated``); ``True`` -> TOKEN-SHARDED (``fn(m[.., r0:r1, :],
                                        r0, r1)`` on this rank's token columns — a per-token statement, M = S·R — then one all_gather; printed
                                        ``token_sharded``). The choice is the adapter's explicit argument (its lever), never a size default
    msa_block_sharded(m, z_shard, layout, *, opm, msa_update, pair_block, opm_first)
    msa_module_sharded(m, z_shard, layout, blocks)
                                        the MSA-module control flow on a row-sharded z with the engine's pieces as callables: per block
                                        ``z_shard = opm(m, z_shard)`` (``z += OPM(m)`` rows, in place), ``m = msa_update(m, z_shard)`` (pair-weighted
                                        averaging + transition; None = the engine's 'skip m-update' last block), ``z_shard = pair_block(z_shard)`` (the
                                        trunk's sharded pair-block driver); ``opm_first`` orders OPM before / after the m-update (both engine forms)
    draw_replicated(draw_fn, name)      a random draw the module consumes (the MSA subsample) drawn ONCE on rank 0, broadcast, and PROVEN replicated
                                        (:func:`dist.allreduce_checksum`). An engine whose ranks draw from identically seeded generators / one
                                        synchronous RNG stream instead keeps its own draw on every rank and proves it with the family's one guard,
                                        ``opt_core.mem.rowpair.trunk.guard_replicated``. ONE form per run: after a ``draw_replicated`` only rank 0's
                                        generator has advanced.

At P == 1 (no group / a P=1 or replicated layout) every sharded statement and driver here REFUSES BY NAME (:func:`dist.require_sharded`):
the kit's ``--n_gpu 1`` path runs the engine's own MSA module. ``draw_replicated`` passes through at P == 1.

Memory per rank: the pair-bias logits ``H·R·N`` (:func:`pwa_bias_rows`), the softmax transient ``chunk·H·q_block·N``, the OPM output block
``rows·N·2·C_z`` (budgeted, printed), the gathered ``o`` slab ``chunk·N·H·c`` per S-chunk; what does NOT shard under ``m_layout=replicated``
is ``m`` itself (``S·N·c_m``) and under both layouts the engine's ``v = linear_v(LN(m_chunk))`` (``chunk·N·H·c``: values need every token) —
named ``msa_m=replicated|token_sharded`` in the schedule census.

Environment (mechanism knobs; a kit's adapter maps its own lever names onto these): ``ROWPAIR_OPM_ROWS`` (int rows), ``ROWPAIR_ROWBLK_MB``
(row-block byte target, default 512), ``ROWPAIR_PWA_QBLOCK`` (int query rows), ``ROWPAIR_PWA_MAX_GB`` (auto threshold, default 8),
``ROWPAIR_LN_GUARD_ELEMS`` (default 2**31). Whether the MSA transition runs replicated or token-sharded is an explicit ARGUMENT
(``shard_tokens``) the adapter passes from its own lever — nothing here selects a replicated or gathered path from a size or an env default.
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, Mapping, Optional, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import Layout, agreed_free_bytes, allreduce_checksum, broadcast_obj, checksum, env_float, env_int, is_dist, require_sharded, world, zmeta, zrows
from .evidence import record_schedule
from .shard import choose_block_rows, iter_row_blocks, unshard_rows
from .transition import opm_rows
from .trimul import INT32_MAX, _share

__all__ = ["opm_rows_budgeted", "default_opm_rows", "pwa_bias_rows", "pwa_rows", "default_pwa_qblock", "msa_transition_rows",
           "msa_block_sharded", "msa_module_sharded", "draw_replicated", "M_LAYOUTS",
           "OPM_FRAC", "ENV_OPM_ROWS", "ENV_ROWBLK_MB", "ROWBLK_MB_DEFAULT", "ENV_PWA_QBLOCK", "ENV_PWA_MAX_GB", "PWA_MAX_GB_DEFAULT",
           "PWA_QBLOCK_AUTO", "ENV_LN_GUARD_ELEMS", "LN_GUARD_DEFAULT"]

OPM_FRAC = 0.25                        # share of the AGREED free bytes an auto-sized OPM row block may take (transient, freed per block)
ENV_OPM_ROWS = "ROWPAIR_OPM_ROWS"      # pins the OPM row block (reproducible schedules print opm_rows_source=env)
ENV_ROWBLK_MB = "ROWPAIR_ROWBLK_MB"    # byte target of a streaming row block [rows, N, C] (default 512 MiB)
ROWBLK_MB_DEFAULT = 512
ENV_PWA_QBLOCK = "ROWPAIR_PWA_QBLOCK"  # pins the pair-weighted-averaging query-row block
ENV_PWA_MAX_GB = "ROWPAIR_PWA_MAX_GB"  # auto threshold: the [chunk, H, R, N] softmax transient above this many GiB is walked in PWA_QBLOCK_AUTO rows
PWA_MAX_GB_DEFAULT = 8.0
PWA_QBLOCK_AUTO = 256
ENV_LN_GUARD_ELEMS = "ROWPAIR_LN_GUARD_ELEMS"
LN_GUARD_DEFAULT = 2 ** 31


def _env_float(name: str, default: float) -> float:
    try:
        return env_float(name, default)
    except ValueError as e:
        raise RowpairRefused(str(e)) from None


M_LAYOUTS = ("replicated", "token_sharded")


def _m_layout(m_layout: str, what: str) -> str:
    """The MSA representation's layout word: ``replicated`` (every rank holds ``m[S, N, c_m]``) | ``token_sharded`` (rank q holds
    ``m[:, r0_q:r1_q, :]``) — an explicit argument of the adapter, refused by name otherwise."""
    if m_layout not in M_LAYOUTS:
        raise RowpairRefused(f"{what}: m_layout={m_layout!r}: one of {M_LAYOUTS} is required")
    return m_layout


def _env_pinned_rows(name: str, what: str) -> Optional[int]:
    """A positive row count pinned through ``name``, or None when unset; malformed / non-positive values are refused BY NAME."""
    if not os.environ.get(name, "").strip():
        return None
    try:
        v = env_int(name, 0)
    except ValueError as e:
        raise RowpairRefused(f"{what}: {e}") from None
    if v < 1:
        raise RowpairRefused(f"{what}: {name}={v}: a positive row count is required")
    return int(v)


# ------------------------------------------------------------------------------------------------------------- OPM rows, budgeted
def default_opm_rows(N: int, C_z: int, Rmax: int, *, elt_bytes: int = 4, bytes_per_row: Optional[int] = None, align: Optional[int] = None,
                     budget_bytes: Optional[int] = None) -> Tuple[int, str]:
    """``(rows, source)`` for the OPM output-row block over this rank's ``Rmax`` rows — :func:`opt_core.mem.rowpair.shard.choose_block_rows`
    with the OPM policy: ``ROWPAIR_OPM_ROWS`` pins the block (``env:ROWPAIR_OPM_ROWS``; malformed / non-positive refused by name); else the
    row-block target (``ROWPAIR_ROWBLK_MB`` -> ``env:ROWPAIR_ROWBLK_MB``, default 512 MiB -> ``default``) for one row's transient ``bytes_per_row``
    (default ``N·2·C_z·elt``: the output block and its normalised copy — an engine statement that chunks its inner ``[rows, N, c·c]`` product itself
    passes its own), LOWERED to ``budget_bytes`` (the caller's share of the AGREED free bytes; None = no CUDA on any rank) when that gives fewer
    rows (``budget``). Every source: below 2**31 elements of ``[rows, N, 2·C_z]`` (``+cap``), rounded down to a multiple of ``align`` (>= ``align``,
    ``+align``), clipped to ``Rmax`` (``+n_max``)."""
    if Rmax is None or int(Rmax) < 1:
        raise RowpairRefused(f"default_opm_rows: Rmax={Rmax!r} (this rank's row capacity) is required and >= 1")
    N, C_z = int(N), int(C_z)
    epr = max(1, N * 2 * C_z)
    per_row = int(bytes_per_row) if bytes_per_row else epr * int(elt_bytes)
    kw = dict(bytes_per_row=per_row, elems_per_row=epr, cap_elems=INT32_MAX, align=int(align) if align else 1, n_max=int(Rmax))
    pinned = _env_pinned_rows(ENV_OPM_ROWS, "default_opm_rows")
    if pinned is not None:
        rows, source = choose_block_rows(rows=pinned, **kw)
        return rows, source.replace("given", f"env:{ENV_OPM_ROWS}", 1)
    rows, source = choose_block_rows(env=ENV_ROWBLK_MB, default_mb=ROWBLK_MB_DEFAULT, **kw)
    if budget_bytes is not None:
        rb, sb = choose_block_rows(budget_bytes=int(budget_bytes), env=None, **kw)
        if rb < rows:
            rows, source = rb, sb
    return rows, source


def opm_rows_budgeted(a, b, layout: Layout, outer_fn: Callable, rows: Optional[int] = None, *, C_z: int, out=None, add: bool = False,
                      align: Optional[int] = None, global_rows: bool = False, row_dim: int = 1, a_local: bool = False,
                      bytes_per_row: Optional[int] = None, budget_bytes: Optional[int] = None, allow_unsharded: bool = False):
    """:func:`~opt_core.mem.rowpair.transition.opm_rows` with the row block decided, aligned and RECORDED here (``evidence.schedule_fields``
    prints ``opm_rows=<r> opm_rows_source=<source> opm_align=<a> opm_inplace=<bool> msa_m=replicated|token_sharded``). ``a`` / ``b``: the
    OPM operands (token rows on ``row_dim``; ``a`` replicated, or with ``a_local=True`` this rank's token rows only — the token-sharded MSA
    representation, ``b`` all-gathered by the caller: :func:`opt_core.mem.rowpair.shard.unshard_rows`); ``C_z``: output channels (sizes the
    default transient); ``add=True, out=z_shard``: the rows are accumulated IN PLACE into the z shard block by block; ``align``: the engine's
    chunk size (row blocks are multiples of it). Without ``rows`` the budget is :data:`OPM_FRAC` of :func:`~opt_core.mem.rowpair.dist.agreed_free_bytes`
    (ONE value on every rank — a collective every rank reaches here — so every rank prints one schedule; ``budget_bytes`` overrides it for tests).
    ``allow_unsharded=True``: the single-GPU row-blocked OPM — an unsharded layout (``Layout(N, 1, 0)``) is accepted, the statement walks all
    ``N`` rows in blocks of the chosen size on this device and the census gains ``opm_layout=unsharded`` (a kit's own named lever, default
    off in every line); without it an unsharded layout is refused by name."""
    unsharded = bool(allow_unsharded) and (layout.P == 1 or layout.replicated or not is_dist())
    if not unsharded:
        require_sharded(layout, "opm_rows_budgeted")
    if out is not None and int(out.shape[0]) != layout.R:
        raise RowpairRefused(f"opm_rows_budgeted: out {tuple(out.shape)} must hold this rank's R={layout.R} rows on dim 0")
    if rows is not None:
        if int(rows) < 1:
            raise RowpairRefused(f"opm_rows_budgeted: rows={rows}: a positive row count is required")
        epr = max(1, layout.N * 2 * int(C_z))
        r_, source = choose_block_rows(rows=int(rows), bytes_per_row=epr * int(a.element_size()), elems_per_row=epr, cap_elems=INT32_MAX,
                                       align=int(align) if align else 1, n_max=layout.Rmax)
    else:
        budget = int(budget_bytes) if budget_bytes is not None else _share(OPM_FRAC, agreed_free_bytes())
        r_, source = default_opm_rows(layout.N, int(C_z), layout.Rmax, elt_bytes=int(a.element_size()), bytes_per_row=bytes_per_row, align=align,
                                      budget_bytes=budget)
    facts = dict(opm_rows=r_, opm_rows_source=source, opm_align=int(align) if align else 1, opm_inplace=bool(add),
                 msa_m="token_sharded" if a_local else "replicated")
    if unsharded:
        facts["opm_layout"] = "unsharded"
    record_schedule(**facts)
    return opm_rows(a, b, layout, outer_fn, r_, out=out, add=add, global_rows=global_rows, row_dim=row_dim, align=align, a_local=a_local,
                    allow_unsharded=allow_unsharded)


# ------------------------------------------------------------------------------------------------------------- pair-weighted averaging
def pwa_bias_rows(prep_fn: Callable[[object, int, int], object], z_shard, layout: Layout, rows: Optional[int] = None):
    """The softmax LOGITS of this rank's query rows: ``prep_fn(z_rows [rows, N, C_z], g0, g1) -> [..., H, rows, N]`` (row-local: the engine's
    ``linear_z(LN(z_rows))`` permuted to heads-first plus the mask bias of the GLOBAL rows ``[g0, g1)``) assembled into ``[..., H, R, N]``.
    ``rows=None``: ONE call unless the shard holds >= ``ROWPAIR_LN_GUARD_ELEMS`` (2**31) elements, in which case row blocks of
    ``2**30 // (N·C_z)`` rows keep every LayerNorm launch int32-indexable (per-row statements: identical values)."""
    require_sharded(layout, "pwa_bias_rows")
    R, N, C, _dtype, _device = zmeta(z_shard)
    if int(R) != layout.R or int(N) != layout.N:
        raise RowpairRefused(f"pwa_bias_rows: z shard {(int(R), int(N), int(C))} vs layout R={layout.R} N={layout.N}")
    if rows is None:
        guard = int(_env_float(ENV_LN_GUARD_ELEMS, LN_GUARD_DEFAULT))
        rows = R if int(R) * int(N) * int(C) < guard else max(1, (2 ** 30) // max(1, int(N) * int(C)))
    rows = max(1, min(int(rows), max(1, int(R))))
    record_schedule(pwa_bias_rows=int(rows))
    if rows >= R:
        y = prep_fn(zrows(z_shard, 0, R), layout.r0, layout.r1)
        if y.dim() < 2 or int(y.shape[-2]) != R or int(y.shape[-1]) != N:
            raise RowpairRefused(f"pwa_bias_rows: prep_fn returned {tuple(y.shape)}; [..., H, rows, N] (query rows on dim -2) is required")
        return y
    from ..torch_rowchunk import rowchunk_apply
    return rowchunk_apply(lambda i0, i1: prep_fn(zrows(z_shard, i0, i1), layout.r0 + i0, layout.r0 + i1), int(R), int(rows), row_dim=-2,
                          lever="rowpair")


def default_pwa_qblock(R: int, N: int, H: int, chunk: int, elt_bytes: int = 4, q_block: Optional[int] = None) -> Tuple[int, str]:
    """``(q_block, source)`` for the query-row block of the pair-weighted averaging over this rank's ``R`` rows: ``q_block`` -> ``given``;
    ``ROWPAIR_PWA_QBLOCK`` -> ``env``; else ``R`` (one block) when the softmax transient ``chunk·H·R·N·elt`` is at most ``ROWPAIR_PWA_MAX_GB`` GiB
    (default 8) -> ``whole``, else :data:`PWA_QBLOCK_AUTO` (256) -> ``auto``; then kept below 2**31 elements per softmax launch (``int32``) and
    clipped to ``[1, R]``. Softmax rows are independent: the value never changes the arithmetic, only the transient."""
    R, N, H, chunk = max(1, int(R)), int(N), int(H), max(1, int(chunk))
    if q_block is not None:
        if int(q_block) < 1:
            raise RowpairRefused(f"default_pwa_qblock: q_block={q_block}: a positive row count is required")
        q, source = int(q_block), "given"
    else:
        pinned = _env_pinned_rows(ENV_PWA_QBLOCK, "default_pwa_qblock")
        if pinned is not None:
            q, source = pinned, "env"
        else:
            nbytes = chunk * H * R * N * int(elt_bytes)
            if nbytes <= _env_float(ENV_PWA_MAX_GB, PWA_MAX_GB_DEFAULT) * 2 ** 30:
                q, source = R, "whole"
            else:
                q, source = PWA_QBLOCK_AUTO, "auto"
    cap = max(1, INT32_MAX // max(1, chunk * H * N))
    if q > cap:
        q, source = cap, source + "+int32"
    return max(1, min(q, R)), source


def pwa_rows(m, bias_shard, layout: Layout, *, values_fn: Callable[[object], object], attend_fn: Callable[[object, object, int, int], object],
             out_fn: Callable[[object], object], softmax_fn: Optional[Callable[[object], object]] = None, s_chunk: Optional[int] = None,
             q_block: Optional[int] = None, out=None, m_layout: str = "replicated"):
    """MSA pair-weighted averaging with the pair logits of LOCAL query rows -> the replicated ``m``-update ``[S, N, c_out]``.

    ``m``: the replicated MSA representation ``[S, N, c_m]`` (an engine's leading batch dims flattened into S, as its chunking over
    sequences does). ``bias_shard``: ``[H, R, N]`` logits of this rank's query rows (:func:`pwa_bias_rows`). Per S-chunk ``[s0, s1)`` of
    ``s_chunk`` sequences (None: one chunk) — the engine's own chunk grid, so every m-side GEMM keeps the dense M::

        state = values_fn(m[s0:s1])                   # the engine's LN_m / linear_v / sigmoid(linear_g) of the full [chunk, N, c_m] slab (any object)
        for query blocks [g0, g1) of the local rows:  # q_block rows: default_pwa_qblock (printed pwa_qblock / pwa_qblock_source)
            w = softmax_fn(bias[:, g0-r0:g1-r0, :])   # [H, q, N]   softmax over j of COMPLETE rows: no communication
            o[:, q-block] = attend_fn(w, state, g0, g1)          # -> [chunk, q, F]: the engine's weights x values einsum in ITS layout (w broadcast
                                                                 #    over the chunk), times its gate rows [g0, g1), flattened — M = q rows instead of N
        o_full = all_gather(o over token rows)        # [chunk, N, F]  ONE collective per S-chunk (bit-preserving)
        out[s0:s1] = out_fn(o_full)                   # the engine's linear_o on the gathered slab: dense M

    ``softmax_fn`` None: ``torch.softmax(x, -1)`` in the logits' dtype (an engine passes its own kernel route, e.g. a fused softmax on CUDA).
    Softmax rows are independent and ``attend_fn`` is per query row, so any ``q_block`` gives the same values; it bounds the transient an
    engine's einsum materialises for ``q`` rows (``chunk·H·q·N`` when ``w`` is expanded over the chunk). Returns ``out`` (allocated ``[S] +
    out_fn's block shape[1:]`` on the first chunk when None).

    ``m_layout="token_sharded"`` (named option; default ``replicated``): ``m`` is this rank's TOKEN shard ``[S, R, c_m]`` of a deep MSA
    representation (``S·N·c_m`` split over ranks instead of replicated). Per S-chunk the NON-LOCAL operand only is all-gathered — ``m_chunk =
    all_gather(m[s0:s1] over tokens) [chunk, N, c_m]`` feeds ``values_fn`` exactly as above (values need every token ``j``) — and the
    output stays token-sharded: ``out[s0:s1] = out_fn(o_rows [chunk, R, F])`` (linear_o at M = chunk·R: a per-token statement) ->
    ``[S, R, c_out]``. Printed ``msa_m=token_sharded``."""
    require_sharded(layout, "pwa_rows")
    tok_sharded = _m_layout(m_layout, "pwa_rows") == "token_sharded"
    m_rows = layout.R if tok_sharded else layout.N
    if m.dim() != 3 or int(m.shape[1]) != m_rows:
        raise RowpairRefused(f"pwa_rows: m {tuple(m.shape)} must be the {'token-sharded [S, R' if tok_sharded else 'replicated [S, N'}, c_m] MSA "
                             f"representation ({m_rows} tokens on dim 1; flatten leading dims into S)")
    if bias_shard.dim() != 3 or int(bias_shard.shape[1]) != layout.R or int(bias_shard.shape[2]) != layout.N:
        raise RowpairRefused(f"pwa_rows: bias_shard {tuple(bias_shard.shape)} must be [H, R, N] with R={layout.R} N={layout.N} (pwa_bias_rows)")
    S, N = int(m.shape[0]), layout.N
    H, R = int(bias_shard.shape[0]), layout.R
    step = S if s_chunk is None else max(1, min(int(s_chunk), S))
    q, q_source = default_pwa_qblock(R, N, H, step, int(bias_shard.element_size()), q_block)
    record_schedule(pwa_qblock=int(q), pwa_qblock_source=q_source, pwa_s_chunk=int(step), msa_m="token_sharded" if tok_sharded else "replicated")
    if softmax_fn is None:
        softmax_fn = lambda x: torch.softmax(x, dim=-1)  # noqa: E731
    for s0 in range(0, S, step):
        s1 = min(S, s0 + step)
        m_chunk = unshard_rows(m[s0:s1].contiguous(), layout, dim=1) if tok_sharded else m[s0:s1]   # token-sharded m: gather the NON-LOCAL operand only
        state = values_fn(m_chunk)
        del m_chunk

        def o_rows(g0: int, g1: int, state=state, n=s1 - s0):                    # softmax weights x values for query rows [g0, g1)
            w = softmax_fn(bias_shard[:, g0 - layout.r0:g1 - layout.r0, :])      # [H, q, N]
            y = attend_fn(w, state, g0, g1)
            del w
            if y.dim() != 3 or int(y.shape[0]) != n or int(y.shape[1]) != g1 - g0:
                raise RowpairRefused(f"pwa_rows: attend_fn returned {tuple(y.shape)} for chunk={n}, query rows [{g0}, {g1}); [chunk, q, F] is required")
            return y

        if q >= R:
            o = o_rows(layout.r0, layout.r1)                                     # ONE launch: byte-identical to the unblocked statement
        else:
            o = None
            for b0, b1, g0, g1 in iter_row_blocks(layout, q, 1):                # query blocks (unit 1: q_block is value-neutral, printed as chosen)
                y = o_rows(g0, g1)
                if o is None:
                    o = y.new_empty((s1 - s0, R, int(y.shape[2])))
                o[:, b0:b1].copy_(y)
                del y
        del state
        if tok_sharded:
            y = out_fn(o.contiguous())                                           # [chunk, R, c_out] the update of this rank's tokens (M = chunk*R)
            n_out = R
        else:
            o_full = unshard_rows(o.contiguous(), layout, dim=-2)                # [chunk, N, F]    all ranks' token rows (bit-exact data movement)
            del o
            y = out_fn(o_full)                                                   # linear_o on the gathered slab: dense M
            del o_full
            n_out = N
        if y.dim() < 2 or int(y.shape[0]) != s1 - s0 or int(y.shape[1]) != n_out:
            raise RowpairRefused(f"pwa_rows: out_fn returned {tuple(y.shape)}; [chunk={s1 - s0}, {n_out}, c_out] is required")
        if out is None:
            out = y.new_empty((S,) + tuple(int(x) for x in y.shape[1:]))
        elif int(out.shape[0]) != S or tuple(int(x) for x in out.shape[1:]) != tuple(int(x) for x in y.shape[1:]):
            raise RowpairRefused(f"pwa_rows: out {tuple(out.shape)} vs the update {(S,) + tuple(int(x) for x in y.shape[1:])}")
        out[s0:s1].copy_(y)
        del y
    return out


# ------------------------------------------------------------------------------------------------------------- MSA transition
def msa_transition_rows(fn: Callable[[object, int, int], object], m, layout: Layout, *, shard_tokens: bool, token_dim: int = -2,
                        m_layout: str = "replicated"):
    """The MSA transition of a replicated ``m``; ``shard_tokens`` is the adapter's explicit choice (its own lever; no default here).
    ``False``: ``fn(m, 0, N)`` — the engine's own statement on the whole ``m`` (exact by construction; ``msa_transition=replicated`` in the
    schedule census). ``True``: ``fn(m_tokens, r0, r1)`` on this rank's token block ``m.narrow(token_dim, r0, R)`` (its mask sliced by the
    callable with the global token range) then ONE all_gather along ``token_dim`` -> the replicated result (a per-token statement: M = S·R
    instead of S·N; ``msa_transition=token_sharded``). ``m_layout="token_sharded"``: ``m`` IS this rank's token shard (``R`` on ``token_dim``);
    ``fn(m, r0, r1)`` is returned as a shard (no gather; ``shard_tokens`` must be True — the statement is the engine's own on the shard)."""
    require_sharded(layout, "msa_transition_rows")
    if shard_tokens not in (True, False):
        raise RowpairRefused(f"msa_transition_rows: shard_tokens={shard_tokens!r}: True | False is required (the adapter's explicit choice)")
    td = token_dim % m.dim()
    if _m_layout(m_layout, "msa_transition_rows") == "token_sharded":
        if not shard_tokens:
            raise RowpairRefused("msa_transition_rows: m_layout='token_sharded' runs the statement on the token shard: shard_tokens=True is required")
        if int(m.shape[td]) != layout.R:
            raise RowpairRefused(f"msa_transition_rows: token-sharded m {tuple(m.shape)} must hold this rank's R={layout.R} tokens on dim {token_dim}")
        record_schedule(msa_transition="token_sharded", msa_m="token_sharded")
        y = fn(m, layout.r0, layout.r1)
        if int(y.shape[td]) != layout.R:
            raise RowpairRefused(f"msa_transition_rows: fn returned {tuple(y.shape)} for this rank's {layout.R} tokens on dim {token_dim}")
        return y
    if int(m.shape[td]) != layout.N:
        raise RowpairRefused(f"msa_transition_rows: m {tuple(m.shape)} token_dim={token_dim} vs layout.N={layout.N}")
    if not shard_tokens:
        record_schedule(msa_transition="replicated")
        return fn(m, 0, layout.N)
    record_schedule(msa_transition="token_sharded")
    y = fn(m.narrow(td, layout.r0, layout.R), layout.r0, layout.r1)
    if int(y.shape[td]) != layout.R:
        raise RowpairRefused(f"msa_transition_rows: fn returned {tuple(y.shape)} for this rank's {layout.R} tokens on dim {token_dim}")
    return unshard_rows(y.contiguous(), layout, dim=td)


# ------------------------------------------------------------------------------------------------------------- block / module drivers
def msa_block_sharded(m, z_shard, layout: Layout, *, opm: Callable[[object, object], object], pair_block: Callable[[object], object],
                      msa_update: Optional[Callable[[object, object], object]] = None, opm_first: bool = True):
    """ONE MSA-module block on a row-sharded z, the engine's pieces as callables: ``opm(m, z_shard) -> z_shard`` (``z += OPM(m)`` rows via
    :func:`opm_rows_budgeted` with ``add=True``), ``msa_update(m, z_shard) -> m`` (``m + PWA(m, z rows)`` then ``+ transition``: :func:`pwa_rows`,
    :func:`msa_transition_rows`; None = the block skips the m-update, as an engine's last block does), ``pair_block(z_shard) -> z_shard`` (the
    trunk's sharded pair-block driver). ``opm_first``: OPM, m-update, pair block; else the OPM follows the m-update
    INSIDE it — m-update, OPM, pair block — so a block without an m-update (``msa_update=None``) runs no OPM either (the engine statement
    order: the late OPM belongs to the update branch). ``m`` stays replicated (``msa_m=replicated``). Returns ``(m, z_shard)``."""
    require_sharded(layout, "msa_block_sharded")
    if opm_first:
        z_shard = opm(m, z_shard)
    if msa_update is not None:
        m = msa_update(m, z_shard)
        if not opm_first:
            z_shard = opm(m, z_shard)
    z_shard = pair_block(z_shard)
    return m, z_shard


def msa_module_sharded(m, z_shard, layout: Layout, blocks: Iterable[Mapping[str, object]], *, between_blocks: Optional[Callable[[int], None]] = None):
    """The MSA module: ``blocks`` yields per-block keyword dicts for :func:`msa_block_sharded` (``opm``, ``pair_block``, ``msa_update``,
    ``opm_first``); ``between_blocks(i)`` runs before block ``i`` (an engine's cache clearing). Records ``msa_blocks`` / ``msa_m=replicated``.
    Returns ``z_shard`` (the module's output is the pair track; ``m`` is consumed)."""
    require_sharded(layout, "msa_module_sharded")
    n = 0
    for i, blk in enumerate(blocks):
        if between_blocks is not None:
            between_blocks(i)
        m, z_shard = msa_block_sharded(m, z_shard, layout, **dict(blk))
        n += 1
    record_schedule(msa_blocks=n, msa_m="replicated")
    return z_shard


# ------------------------------------------------------------------------------------------------------------- replicated random draws
def draw_replicated(draw_fn: Callable[[], object], name: str = "msa_indices", device=None):
    """Rank 0 draws (``draw_fn()`` -> a tensor, e.g. the MSA subsample indices), every rank receives rank 0's draw (broadcast), and the
    replication is PROVEN before the value is used (:func:`~opt_core.mem.rowpair.dist.allreduce_checksum`, raising by name). ``draw_fn`` runs ONLY on rank 0, so only rank 0's generator
    advances: once ONE shared draw of a run goes through this function, EVERY later shared draw must too (per-rank generators are out of
    lockstep from here on) — an engine that keeps identically seeded per-rank generators instead proves each shared draw with the family's
    ONE guard (``opt_core.mem.rowpair.trunk.guard_replicated``) and uses this function on none. The result lives on ``device`` when given, else on the CPU on every rank (indices are small; the engine places them).
    Without a group: ``draw_fn()`` as is (moved to ``device`` when given)."""
    if not is_dist():
        t = draw_fn()
        return t.to(device) if device is not None else t
    P, r = world()
    drawn = None
    if r == 0:
        drawn = draw_fn()
        if not hasattr(drawn, "detach"):
            raise RowpairRefused(f"draw_replicated({name}): draw_fn must return a tensor (got {type(drawn).__name__})")
        drawn = drawn.detach().cpu()
    t = broadcast_obj(drawn, src=0)
    allreduce_checksum(t, name=name, raise_on_mismatch=True)   # always on: the broadcast is proven, not assumed
    n, s = checksum(t)[:2]
    record_schedule(msa_draw="rank0_bcast", msa_draw_n=n, msa_draw_sum=s)
    return t.to(device) if device is not None else t
