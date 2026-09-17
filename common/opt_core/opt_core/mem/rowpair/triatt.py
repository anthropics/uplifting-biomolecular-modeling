"""L2 — triangle attention under row sharding. The attention KERNEL is the caller's callable (the engine's stock ``_attention``, SDPA,
a fused kernel of the F1 family ...): this module only arranges that every query row's softmax over ALL keys happens whole on one rank.

Starting node (around row i: queries ``q_ij``, keys / values ``k_ik, v_ik``, logits ``q_ij . k_ik + tb[j, k, h]``): queries, keys and
values of row i are functions of row i alone — ROW-LOCAL — and the triangle bias ``tb = linear(LN(z))[j, k, h]`` is needed for ALL rows j:
the one collective is an all-gather of the small ``[N, N, H]`` bias, assembled from each rank's rows (:func:`gather_triangle_bias`).
Ending node = the same statement on the transposed pair tensor: :func:`triatt_ending` transposes the shard (all-to-all,
:func:`opt_core.mem.rowpair.dist.transpose_shards`), runs the starting form, and transposes the result back — or, for an engine whose
stack transposes outside the module (``x.transpose(-2, -3)`` before the call), the adapter calls :func:`triatt_starting` on
an already transposed shard.

Query row sub-blocks (``q_rows``, env ``ROWPAIR_TRIATT_QROWS``): the local rows are fed to the kernel in blocks so the logits transient
``rows x H x N x N`` of a non-fused kernel is bounded (the F1 fused kernels need no blocking; pass ``q_rows=None``).

REDUCTION ORDER: every softmax and every PV product is computed on one rank over all N keys — no cross-rank partial sums. The bias
gather and the transposes are bit-preserving. Sharded == dense bit for bit when the kernel's per-query-row arithmetic does not depend on
the number of query rows in the launch (true for the stock einsum/softmax statement and for fused kernels with fixed tiles; a kernel
with occupancy-keyed split-KV is NOT — the family's tests state which).

API:
    gather_triangle_bias(tb_shard, layout)                     ``[R, N, H] -> [N, N, H]`` on every rank (all_gather_rows; H small)
    triatt_starting(attn_rows, z_shard, tb_full, layout, ...)  ``out_shard[i] = attn_rows(z_rows, tb_full, (g0, g1))`` over local query-row
                                                               blocks; ``attn_rows`` is the engine statement 'attention of pair rows g0:g1 given
                                                               the whole triangle bias' (it projects q/k/v/gate from its rows itself)
    triatt_ending(attn_rows, z_shard, tb_of, layout, ...)      transpose_shards -> ``tb_full = gather(tb_of(zT rows))`` -> starting -> transpose back
    triatt_dense(attn_rows, z, tb_full)                        the dense statement (one call over all rows) — the tests' reference
    TriAttFns(ln, bias, attend) + triatt_update_(fns, z_shard, mask_shard, layout, rows=, stream=, tb_rows=)
                                                               the whole STARTING-node triangle-attention UPDATE of a row shard IN PLACE
                                                               (``z[i] += attend(LN(z[i]), mask[i], bias_full)``): the bias ``[N, N, H]`` is the
                                                               all-gather of ``bias(LN(rows))`` computed in row blocks; the update runs per
                                                               row-batch block of ``rows`` rows. ``stream=True`` (default): LN per block, no
                                                               shard-sized buffer (per block: LN once for the bias pass, once for the attention
                                                               pass — same values); ``stream=False``: the stock structure (``x = LN(shard)``
                                                               once into one ``[R, N, C]`` buffer, blocks read it). The ENDING node is the
                                                               same update on rows of z^T (the pair-block driver owns the transposes:
                                                               :mod:`opt_core.mem.rowpair.pairstack`).
    triangle_bias_shard(fns, z_shard) / triatt_update_jit_(fns, z_shard, mask_shard, layout, rows=, qblock=, group=, prefetch=) / BiasBlockExchange
                                                               ``ROWPAIR_TRIATT_BIAS=jit`` (default ``gather`` = the update above): the SAME update
                                                               with the bias kept SHARDED — this rank's plane rows ``[R, N, H]`` only; the row
                                                               windows are served per QUERY BLOCK of the global grid (``ROWPAIR_TRIATT_JIT_QBLOCK``,
                                                               2048) whose bias rows / columns cross the links just in time (per-owner broadcasts /
                                                               a column-slab all-gather, one block prefetched on a side stream,
                                                               ``ROWPAIR_TRIATT_JIT_PREFETCH``), ``ROWPAIR_TRIATT_JIT_GROUP`` windows per fetched
                                                               block, the kernel's operand of each block prepared once per window group
                                                               (``ROWPAIR_TRIATT_JIT_PREP=tile``; ``launch`` = per launch); needs the OPTIONAL
                                                               ``TriAttFns`` fields ``proj mask_bias wrap core``
                                                               (+ ``bias_transposed``) — :func:`opt_core.mem.rowpair.pairstack.bind` binds the
                                                               gather update by name without them; bitwise to the gather update with the flash kernel
    attend_query_blocks(core_fn, q, k, v, biases, qblock)      an attention core over QUERY sub-blocks: ``o[..., j0:j1, :] = core_fn(q[...,
                                                               j0:j1, :], k, v, [b or b[..., j0:j1, :]])`` (biases whose query dim is 1
                                                               broadcast) — bounds the logits transient ``2 * rows * H * qblock * N`` instead
                                                               of ``2 * rows * H * N * N`` elements; every query's softmax is whole
    blocks4(n, size) / lever_rows(name)                        the row-block policy of the query/row-block levers: blocks of ``size`` (a
                                                               multiple of 4) with the tail MERGED into the previous block (no launch with a
                                                               tiny ragged M); ``lever_rows`` reads ``ROWPAIR_<NAME>`` (unset/0 = lever off)
    attention_core(stock, kernel=, min_tokens=, ...)           a ``core_fn`` for :func:`attend_query_blocks` / ``TriAttFns.attend`` that serves
                                                               the row block with a FUSED triangle-attention kernel — ``flash_triattn`` (the
                                                               carried kernel, :mod:`opt_core.kernels.flash_triattn_serve`) or ``cueq``
                                                               (cuEquivariance's ``triangle_attention``) — or the engine's own core ``stock``
                                                               (``torch``), every fallback a NAMED event in :data:`CORE_LEDGER`; a row block with
                                                               the gathered bias IS the kernel's call shape ``q/k/v [1, rows, H, N, D]``, bias
                                                               ``[1, 1, H, N, N]``, key mask ``[1, rows, 1, 1, N]``; launches split so no operand
                                                               indexes 2**31 elements (rows per launch; query blocks on a plane whose bias exceeds
                                                               it, ``ROWPAIR_TRIATT_FLASH_QBLOCK``); Tier 2 (online softmax) — never bitwise to the eager core
    bias_hnn(tb_full, dtype) / emit_core_line(tag)             the ``[N, N, H] -> [1, 1, H, N, N]`` bias copy memoised per gathered-bias tensor
                                                               (one copy per attention call shared by every row batch) / the core's ONE LEVER line
"""
from __future__ import annotations

import contextlib
import os
from typing import Callable, List, Optional, Sequence, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import Layout, all_gather_rows, env_int, require_sharded, transpose_shards, zadd, zblocks, zlen, zrows, zwrite
from .evidence import record_schedule

__all__ = ["gather_triangle_bias", "triatt_starting", "triatt_ending", "triatt_dense", "q_row_block", "blocks4", "lever_rows",
           "attend_query_blocks", "TriAttFns", "triangle_bias_shard", "triangle_bias_rows", "triatt_update_",
           "ENV_TRIATT_BIAS", "BIAS_WORDS", "ENV_TRIATT_JIT_QBLOCK", "JIT_QBLOCK_DEFAULT", "ENV_TRIATT_JIT_GROUP", "JIT_GROUP_DEFAULT",
           "ENV_TRIATT_JIT_PREFETCH", "ENV_TRIATT_JIT_PREP", "JIT_PREP_WORDS", "JIT_PREP_DEFAULT", "JIT_FIELDS", "triatt_bias_word", "jit_fields_missing",
           "jit_qblock", "jit_group", "jit_prefetch", "jit_prep_word", "jit_qblocks",
           "BiasBlockExchange", "triatt_update_jit_",
           "CORE_KERNELS", "ENV_TRIATT_CORE", "ENV_TRIATT_INT32_GUARD", "ENV_TRIATT_FLASH_QBLOCK", "FLASH_QBLOCK_DEFAULT", "int32_guard", "flash_qblock",
           "CORE_LEDGER", "core_ledger", "attention_core", "bias_hnn", "emit_core_line", "describe_core",
           "ENV_TRIATT_STAGE", "STAGE_WORDS", "stage_word", "ENV_TRIATT_BIG", "BIG_WORDS", "big_word", "StagedBias", "StageSlot", "STAGE",
           "stage_bias_once", "stage_slot", "native_staged_bytes", "native_stage_member", "NATIVE_STAGED_MEMBERS", "ENV_TRIATT_STAGE_ROOM", "stage_need_bytes", "stage_room_guard", "stage_room_margin"]


def q_row_block(R: int, q_rows: Optional[int]) -> int:
    """The query-row block: ``q_rows`` when given, else ``ROWPAIR_TRIATT_QROWS`` (0 / unset = all local rows in one call)."""
    if q_rows is None:
        q_rows = env_int("ROWPAIR_TRIATT_QROWS", 0)
    q_rows = int(q_rows)
    return int(R) if q_rows <= 0 else max(1, min(int(R), q_rows))


def gather_triangle_bias(tb_shard, layout: Layout):
    """The whole triangle bias ``[N, N, H]`` from this rank's rows ``[R, N, H]`` (bit-preserving all-gather; equality at P == 1)."""
    require_sharded(layout, "gather_triangle_bias")
    if int(tb_shard.shape[0]) != layout.R or int(tb_shard.shape[1]) != layout.N:
        raise RowpairRefused(f"gather_triangle_bias: shard {tuple(tb_shard.shape)} vs layout R={layout.R} N={layout.N}")
    return all_gather_rows(tb_shard.contiguous(), layout)


def triatt_starting(attn_rows: Callable[[object, object, Tuple[int, int]], object], z_shard, tb_full, layout: Layout, *,
                    q_rows: Optional[int] = None, out=None, inplace_add: bool = False):
    """Starting-node triangle attention of this rank's rows. ``attn_rows(z_rows, tb_full, (g0, g1))`` is the engine's statement for pair rows
    with GLOBAL indices ``g0:g1`` (``z_rows = z_shard[g0-r0 : g1-r0]``, ``[rows, N, C]``) given the whole triangle bias ``tb_full [N, N, H]``; it
    returns the module's output rows ``[rows, N, C_out]``. Rows are fed in blocks of :func:`q_row_block`. ``inplace_add=True``: ``z_shard[rows]
    += result`` (the residual form; returns ``z_shard``); else the rows are written into ``out`` (allocated from the first block when None)
    and ``out`` is returned."""
    require_sharded(layout, "triatt_starting")
    R = zlen(z_shard)
    if R != layout.R:
        raise RowpairRefused(f"triatt_starting: shard has {R} rows, layout.R={layout.R}")
    step = q_row_block(R, q_rows)
    record_schedule(triatt_qrows=step)
    for i0, i1 in zblocks(R, step):
        rows = zrows(z_shard, i0, i1)
        y = attn_rows(rows, tb_full, (layout.r0 + i0, layout.r0 + i1))
        if inplace_add:
            from .dist import zadd
            zadd(z_shard, i0, i1, y)
        else:
            if out is None:
                shape = (R,) + tuple(int(s) for s in y.shape[1:])
                out = y.new_empty(shape)
            out[i0:i1].copy_(y)
        del y
    return z_shard if inplace_add else out


def triatt_ending(attn_rows: Callable[[object, object, Tuple[int, int]], object], z_shard, tb_of: Callable[[object], object],
                  layout: Layout, *, q_rows: Optional[int] = None, inplace_add: bool = False, a2a_chunks: Optional[int] = None):
    """Ending-node triangle attention as the starting statement on the transposed shard: ``zT = transpose_shards(z)``; ``tb_full =
    gather(tb_of(zT_rows))`` (``tb_of`` = the engine's ``linear(LN(rows)) -> [rows, N, H]``); ``yT = triatt_starting(attn_rows, zT, tb_full)``;
    result = transpose back. ``inplace_add=True`` adds the result into ``z_shard`` and returns it; else returns the output shard ``[R, N, C_out]``.
    Two all-to-all transposes + one bias all-gather per call."""
    require_sharded(layout, "triatt_ending")
    zT = transpose_shards(zrows(z_shard, 0, zlen(z_shard)), layout, chunks=a2a_chunks)
    tb_full = gather_triangle_bias(tb_of(zT), layout)
    yT = triatt_starting(attn_rows, zT, tb_full, layout, q_rows=q_rows, out=None, inplace_add=False)
    del zT, tb_full
    y = transpose_shards(yT, layout, chunks=a2a_chunks)
    del yT
    if inplace_add:
        from .dist import zadd
        zadd(z_shard, 0, zlen(z_shard), y)
        return z_shard
    return y


def triatt_dense(attn_rows: Callable[[object, object, Tuple[int, int]], object], z, tb_full):
    """The dense statement: one call over all rows (``(0, N)``)."""
    return attn_rows(z, tb_full, (0, int(z.shape[0])))


# ----------------------------------------------------------------------------------------------------------------- block policy of the levers
def blocks4(n: int, size: Optional[int]) -> List[Tuple[int, int]]:
    """Row blocks ``(i0, i1)`` of ``range(n)`` of ``size`` rows (``size`` must be a positive multiple of 4) with the ragged tail MERGED into the
    previous block, so no launch runs with fewer than ``size`` rows unless ``n`` itself is smaller (one block ``(0, n)`` when ``size`` is None/0
    or ``n <= size``)."""
    n = int(n)
    if not size or int(size) <= 0 or n <= int(size):
        return [(0, n)]
    size = int(size)
    if size % 4:
        raise RowpairRefused(f"row/query block size must be a positive multiple of 4, got {size}")
    k = n // size
    return [(i * size, (i + 1) * size if i < k - 1 else n) for i in range(k)]


def lever_rows(name: str) -> Optional[int]:
    """``ROWPAIR_<name>`` as a row/query block size: unset or 0 -> None (lever off); otherwise a positive multiple of 4 (refused by name if not)."""
    v = os.environ.get("ROWPAIR_" + name, "").strip()
    if not v or int(v) == 0:
        return None
    r = int(v)
    if r < 0 or r % 4:
        raise RowpairRefused(f"ROWPAIR_{name} must be a positive multiple of 4, got {v!r}")
    return r


def attend_query_blocks(core_fn: Callable[[object, object, object, Sequence[object]], object], q, k, v, biases: Sequence[object],
                        qblock: Optional[int]):
    """``core_fn(q, k, v, biases) -> o[..., Q, d]`` evaluated over QUERY sub-blocks of ``qblock`` rows (:func:`blocks4`): ``o[..., j0:j1, :] =
    core_fn(q[..., j0:j1, :], k, v, [b if b.shape[-2] == 1 else b[..., j0:j1, :] for b in biases])``. Keys/values whole, every query row's
    softmax whole: per-query arithmetic unchanged (M-only change); ``qblock`` None/0 or ``>= Q`` = one call."""
    Q = int(q.shape[-2])
    if not qblock or int(qblock) >= Q:
        return core_fn(q, k, v, list(biases))
    o = None
    for j0, j1 in blocks4(Q, int(qblock)):
        ob = core_fn(q[..., j0:j1, :], k, v, [b if int(b.shape[-2]) == 1 else b[..., j0:j1, :] for b in biases])
        if o is None:
            o = ob.new_empty(tuple(ob.shape[:-2]) + (Q,) + tuple(ob.shape[-1:]))
        o[..., j0:j1, :] = ob
        del ob
    return o


# ----------------------------------------------------------------------------------------------------------------- the module-shaped update
class TriAttFns(object):
    """The engine's STARTING-node triangle-attention statements as callables:

    ``ln(z_rows) -> x_rows``                       the input LayerNorm of a block of pair rows ``[rows, N, C]``
    ``bias(x_rows) -> tb_rows``                    the triangle-bias projection of LayerNorm'd rows: ``[rows, N, H]`` (channel-last)
    ``attend(x_rows, mask_rows, tb_full, (i0, i1)) -> o_rows``
                                                  the attention of the LayerNorm'd row batch ``i0:i1`` (LOCAL indices; ``[rows, N, C]``) given its
                                                  mask rows ``[rows, N]`` and the WHOLE triangle bias ``[N, N, H]``: q/k/v/gate projections,
                                                  softmax over all N keys of each row, output projection — ``[rows, N, C]`` (the delta the
                                                  residual adds). Query sub-blocks inside it: :func:`attend_query_blocks`.
    Replicated-by-design: the gathered bias ``[N, N, H]`` (``4*H*N^2`` bytes; H is the small per-head count).

    OPTIONAL (keyword-only, default None: today's three-callable construction is unchanged) — the pieces ``attend`` is made of, so the
    sharded-bias schedule :func:`triatt_update_jit_` (``ROWPAIR_TRIATT_BIAS=jit``) can own the QUERY-BLOCK loop and hand the kernel one
    exchanged bias block per launch instead of the whole plane (``attend == wrap(core(*proj(x), [mask_bias(m), view(tb_full)]), x)``):
    ``proj(x_rows) -> (q, k, v)``                   the row batch's projections in the dispatch's layout (``[rows, H, N, D]`` for ``bnhsd``), scaled
                                                  as the dispatch expects them (query-position-local: ``q[..., j0:j1, :]`` is the block's q)
    ``mask_bias(mask_rows) -> mb``                 the additive key-mask operand the dispatch reads at ``biases[0]`` (``[rows, 1, 1, N]``)
    ``wrap(o, x_rows) -> delta_rows``              gate + output projection of the assembled core output ``o`` (the dispatch's output layout,
                                                  ``[rows, H, N, D]``) — ``[rows, N, C]``
    ``core``                                       the row-block dispatch itself (:func:`attention_core` ``(q, k, v, [mb, tri_bias]) -> o``; the
                                                  triangle bias at ``biases[1]`` as ``[1, H, S_q, S_k]``)
    ``bias_transposed``                            the plane's orientation in the kernel operand: False (default) ``bias[h, j, k] = plane[j, k, h]``
                                                  (a query block = plane ROWS ``j0:j1``); True ``bias[h, j, k] = plane[k, j, h]`` (an engine that
                                                  projects the ending node's bias in z's own frame, ``transpose_bias``: a query block = plane
                                                  COLUMNS ``j0:j1`` of every rank's rows)."""

    def __init__(self, ln: Callable, bias: Callable, attend: Callable, *, proj: Optional[Callable] = None, mask_bias: Optional[Callable] = None,
                 wrap: Optional[Callable] = None, core: Optional[Callable] = None, bias_transposed: bool = False):
        self.ln, self.bias, self.attend = ln, bias, attend
        self.proj, self.mask_bias, self.wrap, self.core, self.bias_transposed = proj, mask_bias, wrap, core, bool(bias_transposed)


def triangle_bias_shard(fns: TriAttFns, z_shard, tb_rows: Optional[int] = None, x_shard=None):
    """This rank's ROWS of the triangle bias, ``[R, N, H]`` contiguous: ``bias(ln(rows))`` over LOCAL row blocks of ``tb_rows`` rows (:func:`blocks4`;
    the LayerNorm'd block is discarded — or read from ``x_shard`` = an already LayerNorm'd shard). No communication (the statements
    :func:`triangle_bias_rows` gathers from; :func:`triatt_update_jit_` keeps this shard and exchanges query blocks of it)."""
    R = zlen(z_shard if x_shard is None else x_shard)
    tb_shard = None
    for i0, i1 in blocks4(R, tb_rows):
        x = fns.ln(zrows(z_shard, i0, i1)) if x_shard is None else zrows(x_shard, i0, i1)
        tb = fns.bias(x)
        if tb_shard is None:
            tb_shard = tb.new_empty((R,) + tuple(tb.shape[1:]))
        tb_shard[i0:i1] = tb
        del x, tb
    return tb_shard.contiguous()


def triangle_bias_rows(fns: TriAttFns, z_shard, layout: Layout, tb_rows: Optional[int] = None, x_shard=None):
    """``[N, N, H]`` triangle bias on every rank: ``bias(ln(rows))`` over LOCAL row blocks of ``tb_rows`` rows (:func:`blocks4`; the LayerNorm'd
    block is discarded — or read from ``x_shard`` = an already LayerNorm'd shard) into ``[R, N, H]`` (:func:`triangle_bias_shard`), then
    :func:`gather_triangle_bias`."""
    tb_shard = triangle_bias_shard(fns, z_shard, tb_rows, x_shard=x_shard)
    tb_full = gather_triangle_bias(tb_shard, layout)
    del tb_shard
    return tb_full


def triatt_update_(fns: TriAttFns, z_shard, mask_shard, layout: Layout, *, rows: int, stream: bool = True, tb_rows: Optional[int] = None,
                   add: bool = True, stage: Optional[str] = None):
    """The starting-node triangle-attention update of this rank's pair rows IN PLACE: ``z_shard[i] (+)= attend(ln(z[i]), mask[i], tb_full)`` for
    every local row-batch block of ``rows`` rows (the engine's pinned attention chunk; LOCAL blocks ``range(0, R, rows)`` — with a layout aligned
    to ``rows`` they are the engine's global chunks), ``tb_full = triangle_bias_rows(...)`` from every rank's rows. ``mask_shard`` ``[R, N]`` or
    None (ones). ``stream=True``: no shard-sized buffer (LN per block in the bias pass and again in the attention pass); ``stream=False``: ``x =
    ln(shard)`` once into one ``[R, N, C]`` buffer (in blocks of ``tb_rows``), bias and attention read it. ``tb_rows`` (``ROWPAIR_TRIATT_ROWBLOCK``
    when None; unset = one block): rows per LN+bias launch. ``add=False``: the shard is overwritten with the delta instead. Returns ``z_shard``
    (== rows ``r0:r1`` of ``z + starting_attention(z)``). The ENDING node: run this on the rows of z^T (:mod:`opt_core.mem.rowpair.pairstack`).
    ``stage`` (``ROWPAIR_TRIATT_STAGE`` when None; ``per_call`` default): ``once`` arms :data:`STAGE` for this plane so a fused core built by
    :func:`attention_core` stages the kernel's bias operand ONCE per gathered plane (per orientation) instead of per row window (:class:`StagedBias`);
    the staged operand is released when the row loop ends."""
    require_sharded(layout, "triatt_update_")
    R = zlen(z_shard)
    N = int(z_shard.shape[1])
    if R != layout.R or N != layout.N:
        raise RowpairRefused(f"triatt_update_: z_shard {tuple(z_shard.shape)} vs layout R={layout.R} N={layout.N}")
    if mask_shard is None:
        mask_shard = z_shard.new_ones((R, N))
    tb_rows = tb_rows if tb_rows else lever_rows("TRIATT_ROWBLOCK")
    rows = max(1, int(rows))
    record_schedule(triatt_rows=rows, triatt_stream=bool(stream), triatt_tb_rows=int(tb_rows or 0))
    x_shard = None
    if not stream:                                                       # the stock structure: one LayerNorm'd copy of the shard
        x_shard = torch.empty_like(z_shard)
        for i0, i1 in blocks4(R, tb_rows):
            x_shard[i0:i1] = fns.ln(z_shard[i0:i1])
    tb_full = triangle_bias_rows(fns, z_shard, layout, tb_rows, x_shard=x_shard)
    with stage_slot().plane(stage_word(stage), mask=mask_shard, rows=rows) as _slot:   # per_call: nothing staged (today's per-window path); once: the core stages the kernel operand of THIS plane once
        for w, (i0, i1) in enumerate(zblocks(R, rows)):
            _slot.window(w)                                                  # the key plan's window index (decided once per plane, no per-window host readback)
            x = fns.ln(z_shard[i0:i1]) if x_shard is None else x_shard[i0:i1]
            o = fns.attend(x, mask_shard[i0:i1], tb_full, (i0, i1))
            del x
            if add:
                zadd(z_shard, i0, i1, o)
            else:
                zwrite(z_shard, i0, i1, o)
            del o
    del tb_full, x_shard
    return z_shard


# ----------------------------------------------------------------------------------------------------------------- sharded bias, just in time
ENV_TRIATT_BIAS = "ROWPAIR_TRIATT_BIAS"                                 # gather (default: today's replicated [N, N, H] plane, triatt_update_) | jit (triatt_update_jit_: the plane stays
BIAS_WORDS = ("gather", "jit")                                          #  SHARDED [R, N, H]; each query block's bias rows cross the links right before the launches that read them)
ENV_TRIATT_JIT_QBLOCK = "ROWPAIR_TRIATT_JIT_QBLOCK"                     # queries per exchanged bias block = per kernel launch (default JIT_QBLOCK_DEFAULT; a positive multiple of 16)
JIT_QBLOCK_DEFAULT = 2048
ENV_TRIATT_JIT_GROUP = "ROWPAIR_TRIATT_JIT_GROUP"                       # row windows (of `rows` rows) served per exchanged block (default JIT_GROUP_DEFAULT): the plane crosses the links
JIT_GROUP_DEFAULT = 4                                                   #  ceil(windows / group) times per call; each window of the group keeps its LN / q / k / v / o tiles across the blocks
ENV_TRIATT_JIT_PREFETCH = "ROWPAIR_TRIATT_JIT_PREFETCH"                 # 1 (default): the NEXT block's collective is issued on a side stream while this block's launches run (two block
                                                                        #  buffers); 0: every block is fetched in line on the compute stream (one buffer)
ENV_TRIATT_JIT_PREP = "ROWPAIR_TRIATT_JIT_PREP"                         # tile (default): the flash kernel's bias operand of each exchanged tile is PREPARED ONCE per (tile, window
JIT_PREP_WORDS = ("tile", "launch")                                     #  group) and the group's windows launch the kernel alone (StageSlot.tiles) | launch: every launch upcasts /
JIT_PREP_DEFAULT = "tile"                                               #  lays out its own tile (the per-launch form); census triatt_jit_prep=
JIT_FIELDS = ("proj", "mask_bias", "wrap", "core")                      # the OPTIONAL TriAttFns fields the jit schedule needs (a kit that binds jit provides them; else gather, by name)


def triatt_bias_word(word: Optional[str] = None) -> str:
    """``gather`` | ``jit``: the explicit ``word``, else ``ROWPAIR_TRIATT_BIAS`` (unset = ``gather``, today's schedule); anything else is refused by name."""
    w = (os.environ.get(ENV_TRIATT_BIAS, "") if word is None else str(word)).strip().lower() or "gather"
    if w not in BIAS_WORDS:
        raise RowpairRefused(f"{ENV_TRIATT_BIAS}={w!r}: one of {BIAS_WORDS} expected")
    return w


def jit_fields_missing(fns) -> List[str]:
    """The names of :data:`JIT_FIELDS` a ``TriAttFns`` does not carry ([] = the jit schedule can bind it)."""
    return [f for f in JIT_FIELDS if getattr(fns, f, None) is None]


def _env_pos_int(name: str, default: int, multiple: int = 1) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return int(default)
    try:
        n = int(v)
    except ValueError:
        n = 0
    if n <= 0 or n % int(multiple):
        raise RowpairRefused(f"{name}={v!r}: a positive integer" + (f" multiple of {multiple}" if multiple > 1 else "") + " expected")
    return n


def jit_qblock(qblock: Optional[int] = None) -> int:
    """Queries per exchanged block: ``qblock`` when given, else ``ROWPAIR_TRIATT_JIT_QBLOCK`` (default :data:`JIT_QBLOCK_DEFAULT`; a positive multiple of 16)."""
    if qblock is not None:
        q = int(qblock)
        if q <= 0 or q % 16:
            raise RowpairRefused(f"triatt jit: qblock={qblock!r}: a positive multiple of 16 expected")
        return q
    return _env_pos_int(ENV_TRIATT_JIT_QBLOCK, JIT_QBLOCK_DEFAULT, 16)


def jit_group(group: Optional[int] = None) -> int:
    """Row windows per exchanged block: ``group`` when given, else ``ROWPAIR_TRIATT_JIT_GROUP`` (default :data:`JIT_GROUP_DEFAULT`)."""
    if group is not None:
        g = int(group)
        if g <= 0:
            raise RowpairRefused(f"triatt jit: group={group!r}: a positive integer expected")
        return g
    return _env_pos_int(ENV_TRIATT_JIT_GROUP, JIT_GROUP_DEFAULT)


def jit_prefetch(prefetch: Optional[bool] = None) -> bool:
    """``prefetch`` when given, else ``ROWPAIR_TRIATT_JIT_PREFETCH`` (default on)."""
    from .dist import env_flag
    return bool(prefetch) if prefetch is not None else env_flag(ENV_TRIATT_JIT_PREFETCH, True)


def jit_prep_word(prep: Optional[str] = None) -> str:
    """``tile`` | ``launch``: the explicit ``prep``, else ``ROWPAIR_TRIATT_JIT_PREP`` (unset = ``tile``); anything else is refused by name."""
    w = (os.environ.get(ENV_TRIATT_JIT_PREP, "") if prep is None else str(prep)).strip().lower() or JIT_PREP_DEFAULT
    if w not in JIT_PREP_WORDS:
        raise RowpairRefused(f"{ENV_TRIATT_JIT_PREP}={w!r}: one of {JIT_PREP_WORDS} expected")
    return w


def jit_qblocks(N: int, qblock: int) -> List[Tuple[int, int]]:
    """The GLOBAL query-block grid ``[(j0, j1), ...]`` of ``range(N)``: blocks of ``qblock`` queries, the last one ragged (one block when ``qblock >= N``).
    Identical on every rank (a function of N and the lever only) — the exchange's lockstep rests on it."""
    N, qblock = int(N), int(qblock)
    if qblock <= 0 or qblock >= N:
        return [(0, N)]
    return [(j0, min(N, j0 + qblock)) for j0 in range(0, N, qblock)]


class BiasBlockExchange(object):
    """The just-in-time exchange of triangle-bias QUERY BLOCKS from a row-sharded bias ``tb_shard [R, N, H]`` (this rank's plane rows): fetch ``f``
    (``f = 0, 1, ..`` in order; block ``b = f % n_blocks`` of :func:`jit_qblocks`, the grid repeating once per row-window group) makes the kernel's bias
    operand of that query block resident on EVERY rank and returns it as the ``[1, H, j1-j0, N]`` view the dispatch takes at ``biases[1]``:

    * ``transposed=False`` (``bias[h, j, k] = plane[j, k, h]``): the block is plane ROWS ``j0:j1`` — for every rank q owning rows of ``[j0, j1)`` one
      ``bcast_`` of its region from q into the block buffer ``[qblock, N, H]`` (the owner copies its rows in first; a block inside one rank's rows is
      one broadcast) — :func:`opt_core.mem.rowpair.dist.all_gather_rows`' aligned form restricted to the block's rows;
    * ``transposed=True`` (``bias[h, j, k] = plane[k, j, h]``): the block is plane COLUMNS ``j0:j1`` of every rank's rows — each rank's column slab
      ``tb_shard[:, j0:j1, :]`` is copied contiguous and all-gathered into ``[N, j1-j0, H]`` (``all_gather_into`` of ``Rmax``-padded slabs on the grid /
      P region broadcasts on the aligned policy, exactly :func:`all_gather_rows`' two forms), viewed ``permute(2, 1, 0)``.

    Bit-preserving (copies and collectives only): the operand's values are the gathered plane's values at those queries. LOCKSTEP: every rank issues
    the same collectives in the same order — guaranteed when every rank constructs the exchange with the same ``(N, qblock, transposed, layout)`` and
    walks ``f = 0 .. total-1`` (the caller's cross-rank agreement check). ``prefetch`` (CUDA only): two block buffers; :meth:`get` of fetch f issues fetch
    f+1's copies + collective on a SIDE stream (after the event :meth:`done` recorded for that buffer's previous block), and makes the current stream
    wait for fetch f's arrival event — the links move block f+1 while block f's launches run. Without prefetch (or on CPU / the threaded test hub) every
    block is fetched in line on the current stream into one buffer. Resident bytes: ``nbuf * qblock * N * H`` elements (+ one ``Rmax * qblock * H``
    send slab when transposed on the grid); :attr:`stats` counts fetches, collectives and wire bytes received."""

    def __init__(self, tb_shard, layout: Layout, qblock: int, *, total: int, transposed: bool = False, prefetch: bool = True):
        from .dist import comm, is_dist
        if tb_shard.dim() != 3 or int(tb_shard.shape[0]) != layout.R or int(tb_shard.shape[1]) != layout.N:
            raise RowpairRefused(f"BiasBlockExchange: tb_shard {tuple(tb_shard.shape)} vs layout R={layout.R} N={layout.N} ([R, N, H] expected)")
        require_sharded(layout, "BiasBlockExchange")
        self.tb = tb_shard.contiguous()
        self.lay = layout
        self.N, self.H = int(tb_shard.shape[1]), int(tb_shard.shape[2])
        self.qb = max(1, min(int(qblock), self.N))
        self.blocks = jit_qblocks(self.N, self.qb)
        self.nb = len(self.blocks)
        self.total = int(total)
        self.transposed = bool(transposed)
        self.cm = comm()
        self.dist = bool(is_dist())
        self.cuda = bool(tb_shard.is_cuda)
        self.prefetch = bool(prefetch) and self.cuda
        self.nbuf = 2 if self.prefetch else 1
        P, Rmax, N, H, qb = layout.P, layout.Rmax, self.N, self.H, self.qb
        self.padded = bool(layout.padded_is_global)
        if self.transposed:
            self.rows_out = P * Rmax if self.padded else N
            self.send = tb_shard.new_empty((Rmax, qb, H)) if self.padded else None       # the own column slab, Rmax-padded (all_gather_into's equal blocks)
            self.bufs = [tb_shard.new_empty((self.rows_out, qb, H)) for _ in range(self.nbuf)]
        else:
            self.rows_out = qb
            self.send = None
            self.bufs = [tb_shard.new_empty((qb, N, H)) for _ in range(self.nbuf)]
        self.side = None
        self.ready = [None] * self.nbuf
        self.free = [None] * self.nbuf
        if self.prefetch:
            dev = tb_shard.device
            self.side = torch.cuda.Stream(device=dev)
            self.side.wait_stream(torch.cuda.current_stream(dev))                      # the shard (written on the compute stream) is complete before any side-stream copy reads it
            self.ready = [torch.cuda.Event() for _ in range(self.nbuf)]
        self.issued = set()
        self.cursor = -1
        self.stats = {"fetches": 0, "collectives": 0, "wire_bytes": 0, "resident_bytes": int(sum(b.numel() * b.element_size() for b in self.bufs)
                                                                                          + (self.send.numel() * self.send.element_size() if self.send is not None else 0))}

    # ---- buffers of a block of width w (a ragged last block uses a contiguous PREFIX of the buffer, reinterpreted)
    def _buf(self, slot: int, w: int):
        b = self.bufs[slot]
        if self.transposed:
            return b.view(-1)[: self.rows_out * w * self.H].view(self.rows_out, w, self.H)
        return b[:w]

    def _snd(self, w: int):
        R = int(self.send.shape[0])
        return self.send.view(-1)[: R * w * self.H].view(R, w, self.H)

    def _collect(self, slot: int, j0: int, j1: int) -> None:
        """Make plane block [j0, j1) resident in buffer ``slot`` (issued on the CURRENT stream — the side stream under prefetch)."""
        lay, cm, tb = self.lay, self.cm, self.tb
        w = j1 - j0
        buf = self._buf(slot, w)
        esz = tb.element_size()
        if not self.transposed:                                                          # plane rows j0:j1: one broadcast per owning rank of its region
            for q, (q0, q1) in enumerate(lay.bounds):
                a0, a1 = max(q0, j0), min(q1, j1)
                if a1 <= a0:
                    continue
                reg = buf[a0 - j0:a1 - j0]
                if q == lay.rank:
                    reg.copy_(tb[a0 - q0:a1 - q0])
                if self.dist:
                    cm.bcast_(reg, src=q)
                    self.stats["collectives"] += 1
                    if q != lay.rank:
                        self.stats["wire_bytes"] += int(reg.numel()) * esz
        elif self.padded:                                                                # plane columns j0:j1 of every rank's rows: all_gather_into of Rmax-padded slabs (global row i at padded i)
            snd = self._snd(w)
            R = lay.R
            if R > 0:
                snd[:R].copy_(tb[:, j0:j1])
            if int(snd.shape[0]) > R:
                snd[R:].zero_()
            cm.all_gather_into(buf, snd)
            self.stats["collectives"] += 1
            self.stats["wire_bytes"] += int(buf.numel() - snd.numel()) * esz
        else:                                                                            # aligned policy: P region broadcasts (all_gather_rows' second form)
            for q, (q0, q1) in enumerate(lay.bounds):
                if q1 <= q0:
                    continue
                reg = buf[q0:q1]
                if q == lay.rank:
                    reg.copy_(tb[:, j0:j1])
                if self.dist:
                    cm.bcast_(reg, src=q)
                    self.stats["collectives"] += 1
                    if q != lay.rank:
                        self.stats["wire_bytes"] += int(reg.numel()) * esz
        self.stats["fetches"] += 1

    def _issue(self, f: int) -> None:
        slot = f % self.nbuf
        j0, j1 = self.blocks[f % self.nb]
        if self.prefetch:
            if self.free[slot] is not None:                                              # the buffer's previous block: every launch that read it has completed (event from done())
                self.side.wait_event(self.free[slot])
            with torch.cuda.stream(self.side):
                self._collect(slot, j0, j1)
                self.ready[slot].record(self.side)
        else:
            self._collect(slot, j0, j1)
        self.issued.add(f)

    def view_of(self, slot: int, b: int):
        """The dispatch's triangle-bias operand of block ``b`` resident in ``slot``: ``[1, H, j1-j0, N]`` (a strided view, no copy)."""
        j0, j1 = self.blocks[b]
        buf = self._buf(slot, j1 - j0)
        if self.transposed:
            return buf[: self.N].permute(2, 1, 0).unsqueeze(0)                            # [N, w, H] -> [1, H, w, N]: bias[h, j, k] = plane[k, j0 + j, h]
        return buf.permute(2, 0, 1).unsqueeze(0)                                         # [w, N, H] -> [1, H, w, N]: bias[h, j, k] = plane[j0 + j, k, h]

    def get(self, f: int):
        """Fetch ``f`` (in order 0, 1, ..): block ``f % n_blocks`` resident and ordered before the current stream's next launches; under prefetch the
        next fetch is issued on the side stream before returning. Returns ``((j0, j1), view [1, H, j1-j0, N])``."""
        f = int(f)
        if f != self.cursor + 1 or f >= self.total:
            raise RowpairRefused(f"BiasBlockExchange.get({f}): fetches are sequential 0..{self.total - 1} (last was {self.cursor})")
        slot = f % self.nbuf
        if f not in self.issued:
            self._issue(f)
        if self.prefetch:
            torch.cuda.current_stream(self.tb.device).wait_event(self.ready[slot])
            if f + 1 < self.total:
                self._issue(f + 1)
        self.cursor = f
        return self.blocks[f % self.nb], self.view_of(slot, f % self.nb)

    def done(self, f: int) -> None:
        """Every launch reading fetch ``f`` has been ENQUEUED on the current stream: its buffer may be refilled once they complete (an event under prefetch)."""
        if self.prefetch:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.tb.device))
            self.free[int(f) % self.nbuf] = ev

    def close(self) -> dict:
        """Order the current stream after the side stream's last work, drop the buffers; returns :attr:`stats`."""
        if self.prefetch and self.side is not None:
            torch.cuda.current_stream(self.tb.device).wait_stream(self.side)
        self.bufs = []
        self.send = None
        self.tb = None
        return dict(self.stats)


def _jit_agree(facts: tuple, what: str) -> None:
    """Cross-rank agreement of the jit schedule's lockstep facts BEFORE its first collective (one object all-gather): every rank sees the same list, so a
    mismatch is refused by name on EVERY rank together (a mismatch inside the loop would be a collective some rank never posts)."""
    from .dist import comm
    allf = comm().allgather_obj(tuple(facts))
    if any(tuple(f) != tuple(allf[0]) for f in allf):
        raise RowpairRefused(f"{what}: ranks disagree on the jit schedule facts (rows, qblock, group, transposed, N, Rmax, H) = {allf}: "
                             f"every rank must run the same {ENV_TRIATT_JIT_QBLOCK} / {ENV_TRIATT_JIT_GROUP} / chunk")


def triatt_update_jit_(fns: TriAttFns, z_shard, mask_shard, layout: Layout, *, rows: int, stream: bool = True, tb_rows: Optional[int] = None,
                       add: bool = True, stage: Optional[str] = None, qblock: Optional[int] = None, group: Optional[int] = None, prefetch: Optional[bool] = None,
                       prep: Optional[str] = None):
    """The starting-node triangle-attention update of this rank's pair rows IN PLACE — :func:`triatt_update_`'s statement and result — with the
    triangle bias kept SHARDED (``ROWPAIR_TRIATT_BIAS=jit``): no ``[N, N, H]`` plane is ever resident. Schedule (``u`` = window of ``rows`` rows):

    1. ``tb_shard = triangle_bias_shard(fns, z)`` — this rank's plane rows ``[R, N, H]`` (the gather path's statements minus the all-gather);
    2. for every GROUP of ``group`` consecutive row windows (LOCAL windows ``range(0, R, rows)``; the group count is ``ceil(ceil(Rmax/rows)/group)`` on
       every rank — a rank past its own rows keeps serving the exchange): per window ``x = ln(z[u]); q, k, v = proj(x); mb = mask_bias(mask[u])``;
       then for every query block ``(j0, j1)`` of the GLOBAL grid :func:`jit_qblocks` ``(N, qblock)``: the block's bias operand arrives through
       :class:`BiasBlockExchange` and every window of the group runs ``o[u][..., j0:j1, :] = core(q[..., j0:j1, :], k, v, [mb, block_view])``; then per
       window ``z[u] (+)= wrap(o[u], x)``.
    Per window the statements are ``attend``'s own (projections once on the whole window, the dispatch per query block on the SAME q / k / v, the
    wrap once on the assembled output): with the flash kernel — whose per-query arithmetic does not depend on the queries sharing the launch
    (:func:`opt_core.kernels.flash_triattn_serve.triangle_attention_qblocks`) — the result equals :func:`triatt_update_`'s bit for bit at equal ``rows``.
    Traffic per call and rank: ``ceil(ceil(R/rows)/group) * (plane bytes) * (P-1)/P`` over the group's links (NVLink), issued one block ahead on a side
    stream under ``prefetch``; device: the shard ``[R, N, H]`` + the exchange's block buffers + ``group`` windows' ``x q k v o`` tiles — never the plane.
    ``fns`` must carry :data:`JIT_FIELDS` (:func:`jit_fields_missing`; :func:`opt_core.mem.rowpair.pairstack.bind` binds the gather path BY NAME when they
    are absent); ``fns.bias_transposed`` selects the block orientation. ``qblock`` / ``group`` / ``prefetch``: :func:`jit_qblock` / :func:`jit_group` /
    :func:`jit_prefetch` (arguments win over ``ROWPAIR_TRIATT_JIT_*``). ``stream`` / ``tb_rows`` / ``add`` as :func:`triatt_update_`. Lockstep: the facts
    ``(rows, qblock, group, transposed, N, Rmax, H)`` are agreed across ranks before the first block (refused by name on mismatch, on every rank).
    ``prep`` (``ROWPAIR_TRIATT_JIT_PREP``, :func:`jit_prep_word`): ``tile`` (default) — the loop runs under :meth:`StageSlot.tiles`: the flash kernel's
    operand of each exchanged tile (fp32 ``[1, 1, H, j1-j0, ceil16(N)]`` + its 16-bit copy + flags, read ONCE from the strided bf16 buffer view) is
    prepared at the group's first window and the group's other windows launch the kernel alone (:func:`attention_core`'s ``flash_tile`` kind); the
    schedule releases it per tile (a buffer refilled in place carries no new identity, so no operand outlives its tile); without device room for one
    tile's operand (``stage_need_bytes('flash_tile')`` + margin) the tile's windows prepare per launch BY NAME (``triatt_jit_prep=launch:no_room``);
    ``launch`` — every launch upcasts / lays out its own tile. Bitwise either way (the same prep and attention kernels on the
    same operands). ``stage`` (``ROWPAIR_TRIATT_STAGE``): the stage-once operand of a GATHERED plane (:func:`triatt_update_`) does not apply to tiles;
    ``once`` is recorded as ignored (``triatt_stage=per_call:jit``).
    Census: ``triatt_bias=jit:qb<qblock>:<blocks>``, ``triatt_jit_group``, ``triatt_jit_prefetch``, ``triatt_jit_fetches``, ``triatt_jit_gib`` (resident
    exchange buffers + window tiles), ``triatt_jit_wire_gib`` (bytes received by this call), ``triatt_jit_prep=tile|launch[:<reason>]`` (+ ``_gib``,
    ``_tiles``, ``_served`` under tile)."""
    require_sharded(layout, "triatt_update_jit_")
    missing = jit_fields_missing(fns)
    if missing:
        raise RowpairRefused(f"triatt_update_jit_: TriAttFns lacks {missing} (the kit's triatt binding provides proj / mask_bias / wrap / core for "
                             f"{ENV_TRIATT_BIAS}=jit; pairstack.bind binds the gather schedule by name without them)")
    R = zlen(z_shard)
    N = int(z_shard.shape[1])
    if R != layout.R or N != layout.N:
        raise RowpairRefused(f"triatt_update_jit_: z_shard {tuple(z_shard.shape)} vs layout R={layout.R} N={layout.N}")
    if mask_shard is None:
        mask_shard = z_shard.new_ones((R, N))
    tb_rows = tb_rows if tb_rows else lever_rows("TRIATT_ROWBLOCK")
    rows = max(1, int(rows))
    qb = min(jit_qblock(qblock), N)
    G = jit_group(group)
    pf = jit_prefetch(prefetch)
    transposed = bool(getattr(fns, "bias_transposed", False))
    x_shard = None
    if not stream:                                                       # the stock structure: one LayerNorm'd copy of the shard
        x_shard = torch.empty_like(z_shard)
        for i0, i1 in blocks4(R, tb_rows):
            x_shard[i0:i1] = fns.ln(z_shard[i0:i1])
    tb_shard = triangle_bias_shard(fns, z_shard, tb_rows, x_shard=x_shard)             # [R, N, H]: this rank's plane rows — never gathered
    H = int(tb_shard.shape[2])
    _jit_agree((rows, qb, G, int(transposed), N, int(layout.Rmax), H), "triatt_update_jit_")
    windows = list(zblocks(R, rows))                                                    # LOCAL windows; the group count is global (Rmax)
    n_win_global = -(-int(layout.Rmax) // rows)
    n_groups = -(-n_win_global // G)
    blocks = jit_qblocks(N, qb)
    nb = len(blocks)
    ex = BiasBlockExchange(tb_shard, layout, qb, total=n_groups * nb, transposed=transposed, prefetch=pf)
    record_schedule(triatt_rows=rows, triatt_stream=bool(stream), triatt_tb_rows=int(tb_rows or 0), triatt_bias=f"jit:qb{qb}:{nb}",
                    triatt_jit_group=G, triatt_jit_prefetch=int(ex.prefetch))
    tile_bytes = 0
    f = 0
    want_stage = stage_word(stage)
    pw = jit_prep_word(prep)
    sl = stage_slot()
    span = sl.tiles() if pw == "tile" else sl.plane("per_call")                            # tile: each tile's kernel operand prepared once per group (released per tile below); launch: every
    with span:                                                                          #  (bound to a name first: a parenthesised `with (expr):` head is read as the 3.9 multi-item form by
                                                                                        #  some 3.11 parsers under ast feature_version=(3, 8) — the Python floor test; no behaviour change)
        if pw == "launch":                                                              #  launch prepares its own (no plane is ever staged once: the buffers are refilled in place)
            record_schedule(triatt_jit_prep="launch")
        if want_stage != "per_call":
            record_schedule(triatt_stage="per_call:jit")                                  # ROWPAIR_TRIATT_STAGE=once asked: ignored under jit, by name
        for g in range(n_groups):
            wins = windows[g * G:(g + 1) * G]                                               # may be empty on a rank with fewer rows: it still serves every fetch
            xs, qkv, mbs, outs = [], [], [], []
            for i0, i1 in wins:
                x = fns.ln(z_shard[i0:i1]) if x_shard is None else x_shard[i0:i1]
                xs.append(x)
                qkv.append(tuple(fns.proj(x)))
                mbs.append(fns.mask_bias(mask_shard[i0:i1]))
                outs.append(None)
            if g == 0 and wins:
                q0_, k0_, v0_ = qkv[0]
                tile_bytes = G * (xs[0].numel() * xs[0].element_size() + sum(t.numel() * t.element_size() for t in (q0_, k0_, v0_)) + q0_.numel() * q0_.element_size())
            for b in range(nb):
                (j0, j1), view = ex.get(f)
                sl.release()                                                            # a tile refilled in place carries no new identity: no earlier tile's operand may serve this one
                for t, (i0, i1) in enumerate(wins):
                    q, k, v = qkv[t]
                    ob = fns.core(q[..., j0:j1, :], k, v, [mbs[t], view])                  # [rows, H, j1-j0, D] in the dispatch's layout
                    if outs[t] is None:
                        outs[t] = ob.new_empty(tuple(ob.shape[:-2]) + (N,) + tuple(ob.shape[-1:]))
                    outs[t][..., j0:j1, :] = ob
                    del ob
                ex.done(f)
                sl.release()                                                            # the tile's prepared operand is dropped before the next tile's is made (one at a time)
                f += 1
                del view
            for t, (i0, i1) in enumerate(wins):
                o = fns.wrap(outs[t], xs[t])
                if add:
                    zadd(z_shard, i0, i1, o)
                else:
                    zwrite(z_shard, i0, i1, o)
                outs[t] = None
                del o
            del xs, qkv, mbs, outs
    stats = ex.close()
    record_schedule(triatt_jit_fetches=int(stats["fetches"]), triatt_jit_gib=round((stats["resident_bytes"] + tile_bytes) / 2 ** 30, 3),
                    triatt_jit_wire_gib=round(stats["wire_bytes"] / 2 ** 30, 3))
    del ex, tb_shard, x_shard
    return z_shard


# ----------------------------------------------------------------------------------------------------------------- fused attention cores
CORE_KERNELS = ("flash_triattn", "cueq", "torch")                       # + "tier:<word>": the row kernels.triattn selects for a tier word (TIER_PREFIX) —
TIER_PREFIX = "tier:"                                                     # the kernel words (kit switches, census, env override use exactly these)
ENV_TRIATT_CORE = "ROWPAIR_TRIATT_CORE"                                 # engineering override of the kit's `kernel=` (printed as triatt_core_src=env)
INT32_MAX = 2 ** 31 - 1
ENV_TRIATT_INT32_GUARD = "ROWPAIR_TRIATT_INT32_GUARD"                   # =1: a pair plane whose bias has more than 2**31-1 elements (H*S*S; H=4: S > 23,170) is REFUSED by
                                                                        #  name (`unsupported:bias_elems>int32`, the engine core serves it);
                                                                        #  unset/0 (default): the flash kernel serves it per query block (FLASH_QBLOCK below), see attention_core
ENV_TRIATT_FLASH_QBLOCK = "ROWPAIR_TRIATT_FLASH_QBLOCK"                 # queries per flash launch on such planes (default FLASH_QBLOCK_DEFAULT; a positive int, a multiple of
FLASH_QBLOCK_DEFAULT = 2048                                             #  the kernel's 64/128-query tile recommended): the kernel's per-launch bias copy is H*qblock*ceil16(S) fp32

ENV_TRIATT_STAGE = "ROWPAIR_TRIATT_STAGE"                               # per_call (default: every row window stages the kernel's bias operand itself)
STAGE_WORDS = ("per_call", "once")                                      #  | once: the operand is staged ONCE per gathered plane per orientation (StagedBias) and every window
STAGE_DEFAULT = "per_call"                                              #  of the plane launches the kernel alone (triatt_update_(stage=) / STAGE)
ENV_TRIATT_BIG = "ROWPAIR_TRIATT_BIG"                                   # how a TIER word serves a plane above the int32 bias bound (H*S*S > 2**31-1): qblocks (default: the
BIG_WORDS = ("qblocks", "native", "auto")                                #  flash kernel per query block) | native: the tier's triattn_native row on the WHOLE
BIG_DEFAULT = "qblocks"                                                 #  window (its member stages a ~16*S*S-byte fp32 plane at H=4) | auto: native when this rank's free
BIG_MARGIN_BYTES = 2 << 30                                              #  device memory covers that staged plane + BIG_MARGIN_BYTES, else qblocks (census triatt_big=)
NATIVE_STAGED_PKGS = ("v11",)                                            # sealed triattn_native generations whose m1 member carries the staged-call contract this module drives
NATIVE_STAGED_MEMBERS = {"cuda_b": "triattn_m1", "cuda_c": "triattn_mw", "cuda_80": "triattn_sm80"}   # package router route -> member module (its top-level
                                                                        #  name) whose bias staging this module hoists: m1 (cc 9.0 bf16 D32, 512<=S<=3072 and S>4096), the
                                                                        #  high band (cc 9.0, 3072<S<=4096) and the sm_80 member (cc 8.0, bf16 D16/32/64, any S); every other
                                                                        #  route (k13 / cuda below 512 tokens, the Triton member for D16/D64/fp16 on 9.0) stages per call, by name
_NATIVE_EXT_MEMBER = {"triattn_m1_ext": "cuda_b", "triattn_mw_ext_g3x4": "cuda_c", "triattn_sm80_ext": "cuda_80"}
                                                                        #  (ext.stage_mask / ext.stage_bias / ext.fwd, module _tma_ok / FALLBACKS): other generations serve per call


def stage_word(stage: Optional[str] = None) -> str:
    """``stage`` or ``ROWPAIR_TRIATT_STAGE`` (unset = ``per_call``); refused by name when not one of :data:`STAGE_WORDS`."""
    w = (os.environ.get(ENV_TRIATT_STAGE, "") if stage is None else str(stage)).strip().lower() or STAGE_DEFAULT
    if w not in STAGE_WORDS:
        raise RowpairRefused(f"{ENV_TRIATT_STAGE} / stage= must be one of {STAGE_WORDS}, got {w!r}")
    return w


def big_word(big: Optional[str] = None) -> str:
    """``big`` or ``ROWPAIR_TRIATT_BIG`` (unset = ``qblocks``); refused by name when not one of :data:`BIG_WORDS`."""
    w = (os.environ.get(ENV_TRIATT_BIG, "") if big is None else str(big)).strip().lower() or BIG_DEFAULT
    if w not in BIG_WORDS:
        raise RowpairRefused(f"{ENV_TRIATT_BIG} / big= must be one of {BIG_WORDS}, got {w!r}")
    return w


def native_stage_member(cc, dtype: str = "bf16", head_dim: int = 32, S: int = 0, strided: bool = True) -> Optional[str]:
    """Which sealed triattn_native member :func:`attention_core` drives with the bias staging hoisted (``ROWPAIR_TRIATT_STAGE=once``) for a
    tri-attention call class -- ``cuda_b`` | ``cuda_c`` | ``cuda_80`` (:data:`NATIVE_STAGED_MEMBERS`) -- or None where the package routes the class
    to a member without a staged-call entry (then every window stages per call, census ``triatt_stage_aside=native_route:<route>``).  Pure table
    lookup (the face's ``route_extension``: no device, no import of the payload); ``cc`` as (8, 0) / "9.0" / 9.0."""
    from ...kernels.triattn import triattn_native as CC
    if isinstance(cc, (tuple, list)):
        ccp = (int(cc[0]), int(cc[1]))
    else:                                                               # "9.0" / 9.0 / "90"
        w = str(cc).strip()
        major, _, minor = w.partition(".") if "." in w else (w[:-1], ".", w[-1:])
        ccp = (int(major), int(minor or 0))
    return _NATIVE_EXT_MEMBER.get(CC.route_extension(str(dtype), int(head_dim), int(S), bool(strided), ccp))


ENV_TRIATT_STAGE_ROOM = "ROWPAIR_TRIATT_STAGE_ROOM"                     # 1 (default): stage a plane ONCE only when the device has room for the staged operand (its size law +
                                                                        #  margin) -- else the plane runs per call BY NAME (census triatt_stage_aside=no_room=<n>); 0: no check
                                                                        #  (engineering switch: the staging's own allocation decides, an OOM is the caller's)


def stage_room_guard() -> bool:
    return os.environ.get(ENV_TRIATT_STAGE_ROOM, "1").strip().lower() not in ("0", "off", "false", "no")


def stage_need_bytes(kind: str, H: int, S: int, *, member: Optional[str] = None, qblock: Optional[int] = None, B: int = 1) -> int:
    """Device bytes staging ONE ``[B, 1, H, S, S]`` plane once holds (+ the largest transient of making it), per staged kind -- the size law:
    ``triattn_native`` m1 (``cuda_b``): :func:`native_staged_bytes` = 65,536*H*nq*(nq+1), nq = ceil(S/128) (~16.5*S^2 at H=4) + the contiguous bf16 source
    copy of a strided view (2*H*S^2); ``cuda_c`` / ``cuda_80``: fp32 [B, H, ceil128(S), ceil64(S)] + that copy; ``flash_triattn``: the prepared plane
    fp32 [B, 1, H, S, ceil16(S)] + its 16-bit copy + flags = 6*H*S*ceil16(S) (+ the fp32 upcast of the source, 4*H*S^2, transient);
    ``flash_qblocks``: EVERY query block prepared = the same 6*H*S*ceil16(S) total + one block's fp32 source copy (4*H*qblock*S) transient;
    ``flash_tile``: ONE exchanged tile of the jit schedule (``[.., qblock, S]``: S = keys) = 6*H*qblock*ceil16(S) + flags + 4*H*qblock*S transient."""
    B, H, S = int(B), int(H), int(S)
    c16 = -(-S // 16) * 16
    if kind == "triattn_native":
        if member in ("cuda_c", "cuda_80"):
            need = 4 * B * H * (-(-S // 128) * 128) * (-(-S // 64) * 64)
        else:
            need = native_staged_bytes(H, S, B)
        return need + 2 * B * H * S * S
    if kind == "flash_triattn":
        return 6 * B * H * S * c16 + 4 * B * H * (-(-S // 32)) + 4 * B * H * S * S
    if kind == "flash_qblocks":
        qb = int(qblock) if qblock else 2048
        return 6 * B * H * S * c16 + 4 * B * H * (-(-S // 32)) + 4 * B * H * min(qb, S) * S
    if kind == "flash_tile":                                            # ONE exchanged tile [B, 1, H, qblock, S] of the jit schedule: its prepared operand + a fp32 source copy (transient)
        qb = min(int(qblock) if qblock else 2048, S)
        return 6 * B * H * qb * c16 + 4 * B * H * (-(-qb // 32)) + 4 * B * H * qb * S
    return 0


def stage_room_margin(need: int) -> int:
    """max(1 GiB, 5 % of the need)."""
    return max(1 << 30, int(need) // 20)


def _free_device_bytes(device) -> Optional[int]:
    """Free bytes on a CUDA device (the driver's number + what this process's caching allocator holds unused); None elsewhere (no check)."""
    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(device)
    try:
        st = torch.cuda.memory_stats(device)
        free += int(st.get("reserved_bytes.all.current", 0)) - int(st.get("allocated_bytes.all.current", 0))
    except Exception as e:  # noqa: BLE001 -- allocator statistics unavailable: the driver's number alone
        from ...oom import is_oom
        if is_oom(e):
            raise
    return int(free)


def native_staged_bytes(H: int, S: int, B: int = 1) -> int:
    """Device bytes of the sealed triattn_native package's m1 staging of one ``[B, 1, H, S, S]`` plane: ``[B*H, nq, 4*(nk+1), 4096]`` fp32 with
    ``nq = nk = ceil(S/128)`` (16.5 * S * S bytes at H=4, S=24,035: 8.7 GiB)."""
    nq = nk = -(-int(S) // 128)
    return int(B) * int(H) * nq * 4 * (nk + 1) * 4096 * 4


def _plane_key(tb) -> tuple:
    """Identity of a bias plane VIEW for staging: storage address + geometry + dtype + version (the kit re-creates the same view of the held
    gathered plane per window: same key; a new plane, another orientation's transpose or an in-place write: another key)."""
    try:                                                                # an inference-mode tensor has no version counter (RuntimeError): identity = address + geometry
        ver = int(tb._version)
    except RuntimeError:
        ver = -1
    return (int(tb.data_ptr()), tuple(int(s) for s in tb.shape), tuple(int(s) for s in tb.stride()), str(tb.dtype), str(tb.device), ver)


class StagedBias(object):
    """The kernel operand of ONE gathered triangle-bias plane (one orientation), staged once and read by every row window of that plane:

    ``kernel``  ``triattn_native``   -> ``ops`` = the m1 member's fragment-order fp32 staging (``ext.stage_bias``; masked-in-every-row keys folded per
                                   ``keyany``, so a window whose batch-OR key words differ is re-staged: identical to the member's per-call result)
                ``flash_triattn`` -> ``ops`` = :class:`opt_core.kernels.flash_triattn_serve.Prepped` of the whole plane (below the int32 bound)
                ``flash_qblocks`` -> ``ops`` = ``[(j0, j1, Prepped), ...]`` per query block (above the bound)
    ``key`` the plane view's identity (:func:`_plane_key`), ``nbytes`` device bytes held, ``n_staged`` stagings (1 + re-stagings), ``n_served``
    windows served from it. Built EMPTY (:func:`stage_bias_once`); the operand is made at the first window, which knows q's dtype / head dim /
    scale / mask; :meth:`drop` releases it."""
    __slots__ = ("kernel", "key", "S", "H", "ops", "keyany", "scale", "member", "nbytes", "n_staged", "n_served", "sig_idx")

    def __init__(self, kernel: str, key: tuple, S: int, H: int):
        self.kernel, self.key, self.S, self.H = str(kernel), key, int(S), int(H)
        self.ops = None; self.keyany = None; self.scale = None; self.member = None
        self.nbytes = 0; self.n_staged = 0; self.n_served = 0; self.sig_idx = None

    def drop(self) -> None:
        self.ops = None; self.keyany = None; self.nbytes = 0

    def same_keys(self, keyany) -> bool:
        """The m1 staging folds the keys no row of the CALL attends (``keyany`` words): reusable only for a window with the same words."""
        a, b = self.keyany, keyany
        if a is None and b is None:
            return True
        if (a is None) != (b is None) or tuple(a.shape) != tuple(b.shape):
            return False
        return bool((a == b).all().item())


def _room_or_aside(slot, sb, kind: str, H: int, S: int, device, *, member: Optional[str] = None, qblock: Optional[int] = None) -> None:
    """Before a plane is staged once: its size law (:func:`stage_need_bytes`) + margin against the device's free bytes; without room the plane
    runs per call BY NAME (:class:`_NoRoom`, census ``triatt_stage_aside=no_room=<n>``, one log line with need / free GiB). ``ROWPAIR_TRIATT_STAGE_ROOM=0``
    skips the check; no CUDA device (or no reading): no check."""
    if not stage_room_guard():
        return
    free = _free_device_bytes(device)
    if free is None:
        return
    need = stage_need_bytes(kind, H, S, member=member, qblock=qblock)
    margin = stage_room_margin(need)
    if int(free) >= need + margin:
        return
    slot.no_room = sb.key
    pre = "triatt_jit_prep" if kind == "flash_tile" else "triatt_stage"          # the jit schedule's tile (StageSlot.tiles) / a plane staged once
    record_schedule(**{pre + "_need_gib": round(need / 2.0 ** 30, 2), pre + "_free_gib": round(int(free) / 2.0 ** 30, 2)})
    _log_core_once("stage:no_room:%s:%d" % (kind, S), "%s: no room for the %s%s staging of the S=%d H=%d %s (need %.2f GiB + margin %.2f, free %.2f GiB) -> "
                   "%s per call, by name (%s=0 skips the check)" % ("jit prep=tile" if kind == "flash_tile" else "stage=once", kind, (":" + member) if member else "", S, H,
                                                                  "tile" if kind == "flash_tile" else "plane", need / 2.0 ** 30, margin / 2.0 ** 30, int(free) / 2.0 ** 30,
                                                                  "this tile's windows prepare" if kind == "flash_tile" else "this plane stages", ENV_TRIATT_STAGE_ROOM))
    raise _NoRoom("no_room")


def _host_bools(t) -> list:
    """ONE host readback of a small bool/uint8 device vector (the stage-once key plan's only synchronisation point; tests count its calls)."""
    return [bool(x) for x in t.tolist()]


def _window_sigs(mask, blocks):
    """``[n_windows, N]`` bool: per row window ``(i0, i1)`` of the plane mask ``[R, N]`` (True / nonzero = attend), the keys SOME row of the
    window attends -- the set every triattn_native member folds into its staged bias (its ``stage_mask`` keyany words), for all windows in ONE
    batched device statement (uniform windows; a ragged list falls to one small ``any`` per window)."""
    m = mask if mask.dtype == torch.bool else (mask != 0)
    R, N = int(m.shape[0]), int(m.shape[-1])
    m = m.reshape(R, N)
    n = len(blocks)
    rows = int(blocks[0][1] - blocks[0][0]) if n else 0
    uniform = n > 0 and all(int(i0) == w * rows and int(i1) == min(R, (w + 1) * rows) for w, (i0, i1) in enumerate(blocks)) and n * rows >= R > (n - 1) * rows
    if uniform:
        if n * rows != R:                                               # the tail window is shorter: pad with rows that attend nothing
            m = torch.cat([m, torch.zeros(n * rows - R, N, dtype=torch.bool, device=m.device)], dim=0)
        return m.reshape(n, rows, N).any(dim=1)
    return torch.stack([m[int(i0):int(i1)].any(dim=0) for i0, i1 in blocks], dim=0) if n else torch.zeros(0, N, dtype=torch.bool, device=m.device)


def _key_decision(slot, sb, sc: float, mask5, S: int):
    """Does THIS window (``slot.cur_window``) need the bias (re)staged for its attended-key set?  True / False from the plane's key plan with
    NO host readback (the plan was read back once per plane: :meth:`StageSlot._plan`); None when the slot has no plan for this window (a
    caller that armed the plane without ``mask=`` / ``rows=``, or never marked the window) -> the caller compares per window as before."""
    w = slot.cur_window
    plan = slot._plan(sb, mask5, S) if (w is not None and mask5 is not None) else None   # made (and checked against THIS window's own mask) at the first marked window
    if sb.ops is None or sb.scale != float(sc):
        return True
    if mask5 is None:
        return sb.keyany is not None                                     # an unmasked window after a masked staging: re-stage (never in the tp lines)
    if plan is None:
        return None
    eq0, eqprev = plan
    if w >= len(eq0):
        return None
    st = sb.sig_idx
    if st is None:                                                      # staged before any window was marked: unknown set -> re-stage by name
        return True
    if st == w:                                                         # the same window again (query sub-blocks of one window)
        return False
    if eq0[w] and eq0[st]:
        return False
    if eq0[w] != eq0[st]:
        return True
    return not (st == w - 1 and eqprev[w])                              # both differ from window 0: equal only provably for the immediately preceding window


class StageSlot(object):
    """Where a fused core (:func:`attention_core`) finds the :class:`StagedBias` of the plane it is serving. :func:`triatt_update_` arms it for
    ONE gathered plane (``with STAGE.plane(mode):`` around its row loop) and the operand is dropped when the loop ends; outside an armed
    ``once`` span every window stages per call (today's path). Census on release: ``triatt_stage=once:<kernel>``, ``triatt_stage_gib`` (peak
    bytes held), ``triatt_stage_planes / _served / _restaged / _aside``."""

    def __init__(self):
        self.mode = STAGE_DEFAULT; self.armed = False; self.staged = None
        self.planes = 0; self.served = 0; self.staged_n = 0; self.restaged = 0; self.aside_n = 0; self.peak_bytes = 0; self.kernels = set(); self.asides = {}
        self.plane_mask = None; self.plane_blocks = None; self.cur_window = None; self.keyplan = None; self.keyplan_word = None; self.keysyncs = 0
        self.no_room = None                                             # the plane key found without device room in this span (per call by name, not re-measured per window)
        self.tile_on = False                                            # armed by tiles(): the jit schedule's exchanged tiles are prepared once per (tile, window group)

    @contextlib.contextmanager
    def plane(self, mode: str = "once", *, mask=None, rows: Optional[int] = None, blocks=None):
        """Arm the slot for ONE gathered plane. ``mask`` = this rank's plane mask ``[R, N]`` (True / nonzero = attend) and ``rows`` = the row
        window of the loop inside (or ``blocks`` = its ``[(i0, i1), ...]``): with them the attended-key set of EVERY window is decided once per
        plane on device and read back ONCE (:func:`_key_decision`), so no window pays a host readback; the loop marks each window with
        :meth:`window` before its attend call (:func:`triatt_update_` does). Without them (or unmarked windows) each window compares its key
        set to the staged one (one small readback per window, counted ``triatt_stage_keysyncs``)."""
        mode = stage_word(mode)
        prev = (self.mode, self.armed, self.tile_on)
        self.release()
        self.mode, self.armed, self.tile_on = mode, True, False
        self.planes = 0; self.served = 0; self.staged_n = 0; self.restaged = 0; self.aside_n = 0; self.peak_bytes = 0; self.kernels = set(); self.asides = {}
        self.cur_window = None; self.keyplan = None; self.keyplan_word = None; self.keysyncs = 0; self.no_room = None
        self.plane_mask, self.plane_blocks = (None, None)
        if mode == "once" and mask is not None and getattr(mask, "dim", lambda: 0)() == 2 and (rows or blocks):
            self.plane_mask = mask
            self.plane_blocks = [(int(a), int(b)) for a, b in blocks] if blocks is not None else list(zblocks(int(mask.shape[0]), int(rows)))
        record_schedule(triatt_stage=mode)
        try:
            yield self
        finally:
            self.release()
            self.mode, self.armed, self.tile_on = prev
            self.plane_mask = None; self.plane_blocks = None; self.cur_window = None; self.keyplan = None

    @contextlib.contextmanager
    def tiles(self):
        """Arm the slot for the jit schedule's exchanged TILES (:func:`triatt_update_jit_`, ``ROWPAIR_TRIATT_JIT_PREP=tile``): a query-block call of
        :func:`attention_core` finds the flash kernel's operand of the tile it reads prepared ONCE (at the group's first window; kind ``flash_tile``)
        and launches the kernel alone; the schedule :meth:`release` s it per tile. Whole-row calls inside the span stage per call (mode
        ``per_call``: no plane is staged once). Census ``triatt_stage=per_call``, ``triatt_jit_prep=tile`` (``launch:<reason>`` when a tile steps
        aside), ``triatt_jit_prep_gib`` (bytes of one prepared tile), ``triatt_jit_prep_tiles`` (tiles prepared), ``triatt_jit_prep_served`` (launches
        served from prepared tiles)."""
        prev = (self.mode, self.armed, self.tile_on)
        self.release()
        self.mode, self.armed, self.tile_on = STAGE_DEFAULT, True, True
        self.planes = 0; self.served = 0; self.staged_n = 0; self.restaged = 0; self.aside_n = 0; self.peak_bytes = 0; self.kernels = set(); self.asides = {}
        self.cur_window = None; self.keyplan = None; self.keyplan_word = None; self.keysyncs = 0; self.no_room = None
        self.plane_mask, self.plane_blocks = (None, None)
        record_schedule(triatt_stage=STAGE_DEFAULT, triatt_jit_prep="tile", triatt_jit_prep_tiles=0)
        try:
            yield self
        finally:
            self.release()
            words = dict(triatt_jit_prep_tiles=self.staged_n, triatt_jit_prep_served=self.served)
            if self.aside_n and self.staged_n:                          # some tiles stepped aside (each reason logged once), the others were prepared once: the lever word stands
                words.update(triatt_jit_prep="tile", triatt_jit_prep_asides=self.aside_n)
            elif not self.aside_n:
                words.update(triatt_jit_prep="tile")
            record_schedule(**words)                                    # all tiles aside: the last `launch:<reason>` word stands
            self.mode, self.armed, self.tile_on = prev
            self.cur_window = None; self.keyplan = None

    def tile(self) -> bool:
        """Inside a :meth:`tiles` span (the jit schedule arms it; :func:`attention_core` gives query-block calls the ``flash_tile`` kind there)."""
        return bool(self.armed and self.tile_on)

    def window(self, w: Optional[int]) -> None:
        """Mark the row window the next attend call(s) serve (its index in the armed plane's blocks); None = unmarked."""
        self.cur_window = None if w is None else int(w)

    def _plan(self, sb: "StagedBias", mask5, S: int):
        """The plane's key plan ``(eq0, eqprev)`` -- per window: its attended-key set equals window 0's / the previous window's -- made at the
        first decision of the plane from the armed plane mask in one batched device statement and read back ONCE together with a check that
        the marked window's own mask (what the kernel folds) matches the plan's row slice; a mismatch (a kit whose attend transforms the mask)
        retires the plan for this plane BY NAME (``triatt_stage_keyplan=mismatch``: per-window compares as before). None without a plan."""
        if self.keyplan is not None:
            return self.keyplan if self.keyplan != "off" else None
        if self.plane_mask is None or not self.plane_blocks or mask5 is None:
            return None
        try:
            pm = self.plane_mask
            if int(pm.shape[-1]) != int(S) or pm.device != mask5.device:
                raise ValueError("plane_mask_shape")
            sig = _window_sigs(pm, self.plane_blocks)                     # [n_w, N]
            w = int(self.cur_window)
            if not 0 <= w < int(sig.shape[0]):
                raise ValueError("window_index")
            m5 = mask5 if mask5.dtype == torch.bool else (mask5 != 0)
            actual = m5.reshape(-1, int(S)).any(dim=0)                   # the marked window's own attended-key set, from the tensor the kernel folds
            eq0 = (sig == sig[0:1]).all(dim=1)
            eqprev = torch.cat([torch.ones(1, dtype=torch.bool, device=sig.device), (sig[1:] == sig[:-1]).all(dim=1)], dim=0)
            ok = (sig[w] == actual).all().reshape(1)
            host = _host_bools(torch.cat([eq0, eqprev, ok], dim=0).to(torch.uint8))   # the ONE readback of this plane
            self.keysyncs += 1
            n = int(sig.shape[0])
            if not host[-1]:
                raise ValueError("mismatch")
            self.keyplan = (host[:n], host[n:2 * n])
            self.keyplan_word = "plan"
        except Exception as e:  # noqa: BLE001 -- no usable plan for this plane: per-window compares, by name
            from ...oom import is_oom
            if is_oom(e):
                raise
            self.keyplan = "off"
            self.keyplan_word = "mismatch" if str(e) == "mismatch" else "percall:%s" % type(e).__name__
        record_schedule(triatt_stage_keyplan=self.keyplan_word, triatt_stage_keysyncs=self.keysyncs)
        return self.keyplan if self.keyplan != "off" else None

    def note_keysync(self) -> None:
        """A per-window key-set compare read back to the host (no plan for the window): counted."""
        self.keysyncs += 1
        if self.keyplan_word is None:
            self.keyplan_word = "percall"
        record_schedule(triatt_stage_keyplan=self.keyplan_word, triatt_stage_keysyncs=self.keysyncs)

    def once(self) -> bool:
        return bool(self.armed and self.mode == "once")

    def get(self, kernel: str, tb, S: int, H: int) -> "StagedBias":
        key = _plane_key(tb)
        sb = self.staged
        if sb is not None and sb.kernel == kernel and sb.key == key:
            return sb
        if sb is not None:                                              # another plane / view / kernel inside one armed span: re-stage (counted)
            self.restaged += 1
            sb.drop()
        self.staged = StagedBias(kernel, key, S, H)
        self.planes += 1
        return self.staged

    def note_staged(self, sb: "StagedBias") -> None:
        self.staged_n += 1
        self.kernels.add(sb.kernel)
        self.peak_bytes = max(self.peak_bytes, int(sb.nbytes))
        if sb.kernel == "flash_tile":                                   # the jit schedule's tiles: their own words (no plane is staged once)
            if self.staged_n == 1 or self.staged_n % 64 == 0:
                record_schedule(triatt_jit_prep="tile", triatt_jit_prep_gib=round(self.peak_bytes / 2.0 ** 30, 2), triatt_jit_prep_tiles=self.staged_n)
            return
        record_schedule(triatt_stage="once:" + "+".join(sorted(self.kernels)), triatt_stage_gib=round(self.peak_bytes / 2.0 ** 30, 2),
                        triatt_stage_planes=self.planes, triatt_stage_n=self.staged_n, triatt_stage_restaged=self.restaged)

    def note_served(self, sb: "StagedBias") -> None:
        sb.n_served += 1; self.served += 1
        if self.served == 1 or self.served % 64 == 0:
            record_schedule(**{("triatt_jit_prep_served" if sb.kernel == "flash_tile" else "triatt_stage_served"): self.served})

    def aside(self, reason: str) -> None:
        self.aside_n += 1
        self.asides[reason] = self.asides.get(reason, 0) + 1
        from .trunk import census_token                                              # the census VALUE as one token (an aside reason may carry an exception text)
        if self.tile_on:                                                # a tile of the jit schedule steps aside: its windows prepare per launch, by name
            record_schedule(triatt_jit_prep="launch:%s" % census_token(reason), triatt_jit_prep_asides=self.aside_n)
            return
        record_schedule(triatt_stage_aside="%s=%d" % (census_token(reason), self.asides[reason]))

    def release(self) -> None:
        if self.staged is not None:
            self.staged.drop()
            self.staged = None
        if self.tile_on:                                                # the next tile measures its room again (a plane span keeps its no-room key for the span)
            self.no_room = None
        if self.served:
            record_schedule(**{("triatt_jit_prep_served" if self.tile_on else "triatt_stage_served"): self.served})


STAGE = StageSlot()                                                     # the MAIN thread's slot (a rank process has one thread of statements): triatt_update_ arms it,
_TL = __import__("threading").local()                                   #  attention_core(staged=None) reads it; rank THREADS (an emulated line) each get their own (stage_slot())


def stage_slot() -> StageSlot:
    """The calling thread's :class:`StageSlot`: :data:`STAGE` on the main thread, a per-thread slot on rank threads (the emulated line runs P
    ranks as threads of one process; real ranks are processes and never share it)."""
    import threading
    if threading.current_thread() is threading.main_thread():
        return STAGE
    sl = getattr(_TL, "slot", None)
    if sl is None:
        sl = _TL.slot = StageSlot()
    return sl


def stage_bias_once(tb, kernel_word: str, S: Optional[int] = None, H: Optional[int] = None, *, slot: Optional[StageSlot] = None) -> StagedBias:
    """The :class:`StagedBias` of plane ``tb`` (``[.., H, S, S]`` view in the kernel's frame) for ``kernel_word`` (``triattn_native`` |
    ``flash_triattn`` | ``flash_qblocks``) in ``slot`` (default :data:`STAGE`): the held one when it was made from this very view for this kernel,
    else a new EMPTY one (the previous operand dropped). The operand itself is staged by the first window call (:func:`attention_core`)."""
    sl = slot if slot is not None else stage_slot()
    S_ = int(tb.shape[-1]) if S is None else int(S)
    H_ = int(tb.shape[-3]) if H is None and tb.dim() >= 3 else int(H or 1)
    return sl.get(str(kernel_word), tb, S_, H_)


class _StageAside(Exception):
    """The staged path cannot take THIS call (by name): the per-call path serves it."""
    sticky = True                                                       # the rest of this call class steps aside too (memoised by the core)


class _NoRoom(_StageAside):
    """The device has no room for the plane's staged operand right now: THIS plane runs per call by name; a later plane asks again."""
    sticky = False


def _native_member(q5, k5, v5, tb5, mask5):
    """``(route, member module, ext)`` of the sealed triattn_native member the package's router takes this call to, when that member carries a
    staged-call contract this module drives (:data:`NATIVE_STAGED_MEMBERS`: m1 = ``cuda_b``, the high band ``cuda_c``, the sm_80 member ``cuda_80``);
    :class:`_StageAside` naming the reason otherwise (payload bytes untouched: this is face glue over the members' public entry points)."""
    from ...kernels.triattn import triattn_native as CC
    if str(CC.ACTIVE_PKG) not in NATIVE_STAGED_PKGS:
        raise _StageAside("native_pkg:%s" % CC.ACTIVE_PKG)
    try:
        if "install" not in CC._STATE:
            CC.install(check=True)
        pkg = CC._module()
        name = pkg.route(q5, k5, v5, tb5, mask5)
    except Exception as e:  # noqa: BLE001 -- the package's typed refusal / no prebuilt: the per-call door names it
        from ...oom import is_oom
        if is_oom(e):
            raise
        raise _StageAside("native_route:%s" % type(e).__name__) from None
    modname = NATIVE_STAGED_MEMBERS.get(name)
    if modname is None:                                                 # k13 / cuda (S < 512), the Triton member (D16 / D64 / fp16 ...): no staged-call entry -> per call, by name
        raise _StageAside("native_route:%s" % name)
    try:
        import importlib
        face = importlib.import_module(pkg.__name__ + "._face")        # the payload's face: member resolution (prebuilt .so injected into the member module)
        face._resolve(name)
        M = importlib.import_module(modname)                            # the member, by the top-level name the router imports it under
        if name == "cuda_b":
            ext = M._EXT if getattr(M, "_EXT", None) is not None else M._build()
        else:
            ext = M._build()                                            # cuda_c: the default geometry's injected extension; cuda_80: its one extension
    except Exception as e:  # noqa: BLE001 -- no member module / no prebuilt member for this stack: per call by name
        from ...oom import is_oom
        if is_oom(e):
            raise
        raise _StageAside("native_member:%s:%s" % (type(e).__name__, str(e)[:60])) from None
    if name == "cuda_b":
        missing = [n for n in ("stage_mask", "stage_bias", "fwd") if not hasattr(ext, n)] + [n for n in ("_tma_ok", "FALLBACKS") if not hasattr(M, n)]
    else:                                                               # cuda_c / cuda_80: the member's OWN public staged-call entry (bias_staged=) and staging functions
        import inspect
        entry = getattr(M, "triangle_attention" if name == "cuda_c" else "triangle_attention_sm80", None)
        missing = [n for n in (("stage_mask", "stage_bias") + (("geometry",) if name == "cuda_80" else ())) if not callable(getattr(M, n, None))]
        try:
            if entry is None or "bias_staged" not in inspect.signature(entry).parameters:
                missing.append("bias_staged")
        except (TypeError, ValueError):
            missing.append("bias_staged")
    if missing:
        raise _StageAside("native_contract:%s:%s" % (name, ",".join(missing)))
    return name, M, ext


def _native_staged_call(sb: StagedBias, q5, k5, v5, tb5, mask5, sc: float, slot: StageSlot):
    """Serve one row window from the plane's staged operand: the m1 member (``cuda_b``) through :func:`_native_staged_call_m1`; the high-band
    member (``cuda_c``) and the sm_80 member (``cuda_80``) through their own public entry with ``bias_staged=`` (:func:`_native_staged_call_entry`)."""
    name = sb.member[0]
    record_schedule(triatt_stage_member=name)
    if name == "cuda_b":
        return _native_staged_call_m1(sb, q5, k5, v5, tb5, mask5, sc, slot)
    return _native_staged_call_entry(sb, q5, k5, v5, tb5, mask5, sc, slot)


def _key_signature(mask5, B: int, N: int, S: int):
    """``[B, S]`` bool: the keys SOME row of the call attends (what every member folds into its staged bias as -inf columns), or None unmasked."""
    if mask5 is None:
        return None
    m = mask5.reshape(B, N, S)
    return (m if m.dtype == torch.bool else (m != 0)).any(dim=1)


def _native_staged_call_entry(sb: StagedBias, q5, k5, v5, tb5, mask5, sc: float, slot: StageSlot):
    """``cuda_c`` (3072 < S <= 4096 on cc 9.0) / ``cuda_80`` (cc 8.0): the member's public ``triangle_attention(..., bias_staged=)`` with the
    staging made ONCE per plane by the member's own ``stage_mask`` -> ``stage_bias`` from the same contiguous plane the per-call door hands it
    (re-made only for a window whose attended-key set differs, or another scale); the member's forward statements otherwise unchanged --
    outputs bitwise the per-call member's."""
    name, M, ext = sb.member
    B, N, H, S, D = (int(x) for x in q5.shape)
    if q5.dtype != torch.bfloat16 or k5.dtype != torch.bfloat16 or v5.dtype != torch.bfloat16:
        raise _StageAside("native_%s:dtype" % name)
    if (name == "cuda_c" and (D != 32 or S > 4096)) or (name == "cuda_80" and D not in (16, 32, 64)):
        raise _StageAside("native_%s:class_D%d_S%d" % (name, D, S))
    if tuple(k5.shape) != (B, N, H, S, D) or tuple(v5.shape) != (B, N, H, S, D):
        raise _StageAside("native_%s:kv_shape" % name)
    bias5 = tb5 if tb5.dim() == 5 else tb5.unsqueeze(0)
    if tuple(bias5.shape) != (B, 1, H, S, S):
        raise _StageAside("native_%s:bias_shape" % name)
    if mask5 is not None and tuple(mask5.shape) != (B, N, 1, 1, S):
        raise _StageAside("native_%s:mask_shape" % name)
    keysig = _key_signature(mask5, B, N, S)
    need = _key_decision(slot, sb, sc, mask5, S)
    if need is None:                                                    # no plan for this window: compare its key set to the staged one (one small readback, counted)
        need = sb.ops is None or sb.scale != float(sc) or not sb.same_keys(keysig)
        if sb.ops is not None:
            slot.note_keysync()
    if need:
        restage = sb.ops is not None
        sb.drop()
        keyany = None
        if mask5 is not None:                                           # the member's own key words for THIS attended-key set (what its per-call path folds into the staging)
            if name == "cuda_c":
                keyany = M.stage_mask(mask5, B, N, S, ext)[1]
            else:
                g = M.geometry(D, False)
                dense = all(int(t.stride(3)) <= 512 for t in (q5, k5, v5))
                small = S <= int(g["small_max"]) or (dense and S <= int(g["small_max_rows"]))
                R = int(M.geometry(D, True)["R"]) if small else int(g["R"])
                keyany = M.stage_mask(mask5, B, N, S, R, q5.device)[0]
        src = bias5 if bias5.is_contiguous() else bias5.contiguous()   # the plane the per-call door hands the package (a contiguous copy of the view), made once per staging
        sb.ops = M.stage_bias(src, float(sc), keyany)
        del src
        sb.keyany, sb.scale, sb.sig_idx = keysig, float(sc), slot.cur_window
        sb.nbytes = int(sb.ops.numel()) * int(sb.ops.element_size())
        sb.n_staged += 1
        if restage:
            slot.restaged += 1
        slot.note_staged(sb)
    entry = M.triangle_attention if name == "cuda_c" else M.triangle_attention_sm80
    return entry(q5, k5, v5, bias5, mask5, float(sc), bias_staged=sb.ops)


def _native_staged_call_m1(sb: StagedBias, q5, k5, v5, tb5, mask5, sc: float, slot: StageSlot):
    """The m1 member's ``triangle_attention_m1`` statements with the bias staging HOISTED: ``ext.stage_bias`` runs for the first window of the
    plane (and again only for a window whose batch-OR key words differ), every other window zeroes the fix-list counter and launches
    ``ext.fwd`` on the held staging. Same operands into the same kernels: outputs bitwise the per-call member's."""
    _name, M1, ext = sb.member
    B, N, H, S, D = (int(x) for x in q5.shape)
    if not (q5.is_cuda and k5.is_cuda and v5.is_cuda and tb5.is_cuda and (mask5 is None or mask5.is_cuda)):
        raise _StageAside("native_m1:not_cuda")
    if D != 32 or q5.dtype != torch.bfloat16 or k5.dtype != torch.bfloat16 or v5.dtype != torch.bfloat16:
        raise _StageAside("native_m1:dtype_or_head_dim")
    if tuple(k5.shape) != (B, N, H, S, D) or tuple(v5.shape) != (B, N, H, S, D):
        raise _StageAside("native_m1:kv_shape")
    bias5 = tb5 if tb5.dim() == 5 else tb5.unsqueeze(0)
    if tuple(bias5.shape) != (B, 1, H, S, S):
        raise _StageAside("native_m1:bias_shape")
    qkv = []
    for t in (q5, k5, v5):
        if M1._tma_ok(t):
            qkv.append(t)
        else:
            M1.FALLBACKS["qkv_copy"] += 1
            qkv.append(t.contiguous())
    maskw = rowkind = kcend = kcstart = keyany = rowkc0 = rowkc1 = None
    mask = mask5
    if mask is not None:
        if tuple(mask.shape) != (B, N, 1, 1, S):
            raise _StageAside("native_m1:mask_shape")
        if mask.dtype != torch.bool:
            M1.FALLBACKS["mask_copy"] += 1
            mask = mask != 0
        if M1._MASK_COUNTS is None or M1._MASK_COUNTS.device != q5.device:
            M1._MASK_COUNTS = torch.zeros(2, dtype=torch.int32, device=q5.device)
        maskw, keyany, rowkind, kcend, kcstart, rowkc0, rowkc1 = ext.stage_mask(mask, M1._MASK_COUNTS)
    n_ctas = ((S + 127) // 128) * ((N + 2) // 3 + 1) * B * H
    fix = torch.empty(1 + 3 * n_ctas, dtype=torch.int32, device=q5.device)
    need = _key_decision(slot, sb, sc, mask, S)
    if need is None:                                                    # no plan for this window: compare its key words to the staged ones (one small readback, counted)
        need = sb.ops is None or sb.scale != float(sc) or not sb.same_keys(keyany)
        if sb.ops is not None:
            slot.note_keysync()
    if need:
        restage = sb.ops is not None
        sb.drop()
        sb.ops = ext.stage_bias(bias5, float(sc), keyany, fix)          # zeroes fix[0]; [B*H, nq, 4*(nk+1), 4096] fp32
        sb.keyany, sb.scale, sb.sig_idx = keyany, float(sc), slot.cur_window
        sb.nbytes = int(sb.ops.numel()) * int(sb.ops.element_size())
        sb.n_staged += 1
        if restage:
            slot.restaged += 1
        slot.note_staged(sb)
    else:
        fix[0:1].zero_()                                                # the attention kernel's fix-list counter starts at 0 (stage_bias does this on the staging call)
    out = torch.empty(B, N, H, S, D, dtype=q5.dtype, device=q5.device)
    if M1._FIX_TOTAL is None or M1._FIX_TOTAL.device != q5.device:
        M1._FIX_TOTAL = torch.zeros(2, dtype=torch.int32, device=q5.device)
    ext.fwd(qkv[0], qkv[1], qkv[2], sb.ops, float(sc), out, fix, M1._FIX_TOTAL, maskw, rowkind, kcend, kcstart, rowkc0, rowkc1, 0, None)
    return out



def int32_guard() -> bool:
    """``ROWPAIR_TRIATT_INT32_GUARD`` set (not ``""``/``0``): planes with ``H*S*S > 2**31-1`` bias elements are refused by name."""
    return os.environ.get(ENV_TRIATT_INT32_GUARD, "").strip() not in ("", "0")


def flash_qblock() -> int:
    """``ROWPAIR_TRIATT_FLASH_QBLOCK`` (unset = :data:`FLASH_QBLOCK_DEFAULT`): queries per flash launch on a plane above the int32 bias bound; refused if not a positive int."""
    v = os.environ.get(ENV_TRIATT_FLASH_QBLOCK, "").strip()
    if not v:
        return FLASH_QBLOCK_DEFAULT
    try:
        n = int(v)
    except ValueError:
        n = 0
    if n <= 0:
        raise RowpairRefused(f"{ENV_TRIATT_FLASH_QBLOCK} must be a positive integer (queries per flash launch), got {v!r}")
    return n
CORE_LEDGER = None                                                       # the process ledger of the core (name=F1.flash_triattn), created by core_ledger()
_CORE_STATE = {"served": 0, "fallback": 0, "kernel": None, "printed": set()}
_CORE_STATE_LOCK = __import__("threading").Lock()                         # served/fallback are read-modify-write: exact under rank THREADS too (an emulated line; real ranks are processes)
_BIAS_MEMO: dict = {}


def _log_core_once(key: str, msg: str) -> None:
    import sys
    if key not in _CORE_STATE["printed"]:
        _CORE_STATE["printed"].add(key)
        sys.stderr.write("[opt_core.rowpair] triatt core %s\n" % msg)
        sys.stderr.flush()


def core_ledger(min_tokens: int = 0, expected: Sequence[str] = ("below_gate",)):
    """The process ledger of :func:`attention_core` (``name=F1.flash_triattn``, ``origin=core``; :func:`opt_core.kernels.flash_triattn_serve.ledger`),
    created on first use with the kit's token gate and the fallback reasons its mode declares."""
    global CORE_LEDGER
    if CORE_LEDGER is None:
        from ...kernels import flash_triattn_serve as F1
        CORE_LEDGER = F1.ledger(min_tokens=int(min_tokens), expected=tuple(expected), origin="core")
        CORE_LEDGER.set("rowpair", 1)
    return CORE_LEDGER


def bias_hnn(tb_full, dtype=None):
    """``[N, N, H] -> [1, 1, H, N, N]`` contiguous (``dtype``: None = as is), memoised per gathered-bias TENSOR (identity of a held source, never an
    address): every row batch of one attention call shares one copy; the next call's bias replaces it (at most one extra ``[N, N, H]`` alive)."""
    dt = dtype if dtype is not None else tb_full.dtype
    if _BIAS_MEMO.get("src") is tb_full and _BIAS_MEMO.get("dtype") == dt:
        return _BIAS_MEMO["v"]
    _BIAS_MEMO.clear()
    v = tb_full.permute(2, 0, 1).to(dtype=dt).contiguous()[None, None]
    _BIAS_MEMO.update(src=tb_full, dtype=dt, v=v)
    return v


def _pick(biases: Sequence[object], which):
    if callable(which):
        return which(biases)
    if which in (None, "none"):
        return None
    idx = {"bias0": 0, "bias1": 1, "bias2": 2}[which]
    return biases[idx] if idx < len(biases) else None


def attention_core(stock: Callable[[object, object, object, Sequence[object]], object], *, kernel: str = "flash_triattn", min_tokens: int = 0,
                   ledger=None, scale: Optional[float] = None, layout: str = "bnhsd", mask_from="bias0", tri_bias="bias1",
                   serve_dtypes: Sequence[str] = ("bfloat16", "float16"), stock_qblock: Optional[int] = None,
                   expected: Sequence[str] = ("below_gate",), staged: Optional[StageSlot] = None, **kernel_kwargs) -> Callable:
    """A ``core_fn(q, k, v, biases) -> o`` (the signature :func:`attend_query_blocks` and every engine's eager attention core take) serving the
    ROW BLOCK with a fused triangle-attention kernel, else the engine's own core ``stock`` — every non-served call a NAMED event.

    ``kernel``: ``flash_triattn`` (the carried kernel through :func:`opt_core.kernels.flash_triattn_serve.triangle_attention`: the
    documented gates fall back to ``stock`` COUNTED — ``below_gate`` (``S < min_tokens``), ``unsupported:<why>`` (the kernel's own
    ``flash_supported``: dtype / head dim / rank classes, or ``bias_elems>int32`` under ``ROWPAIR_TRIATT_INT32_GUARD=1``); a card without a tuned row runs the kernel's safe settings
    (its own ``settings=safe:…`` word); where the kernel cannot run at all — q not on a CUDA device, torch / triton / the carried kernel not
    importable, cc < 8.0 (:func:`flash_triattn_serve.require`), its safe settings failing to build — the call raises :class:`RowpairRefused`
    naming the lever and the opt-out, never a silent engine-core path) | ``cueq`` (``cuequivariance_torch.triangle_attention``; not importable =
    :class:`RowpairRefused` at construction; a call it rejects = ``cueq_unsupported:<Type>``) | ``torch`` (``stock`` for every call, counted
    as ``kernel_torch`` — the opt-out) | ``tier:<word>`` (the row :mod:`opt_core.kernels.triattn` selects for a tier word — fast | exact |
    big — per row window through :func:`opt_core.attn.pair_fused.core_attention`, resolved once per shape class; where the provider refuses
    or cannot serve, the flash_triattn path serves BY NAME from then on: schedule word ``triatt_core_tier_off``, one stderr note; planes above
    the int32 bias bound always take the flash path per query block -- schedule word ``triatt_core_tier_big``, never the engine statement, no
    provider row asked there -- and dtypes outside ``serve_dtypes`` take the flash path's own named fallback).
    ``ROWPAIR_TRIATT_CORE`` overrides ``kernel`` (engineering; census ``triatt_core_src=env``).
    ``layout``: ``bnhsd`` = q/k/v ``[B, rows, H, S, D]`` (the row axis ahead of the heads — cuEquivariance's order; B may be absent) | ``bhnsd`` = ``[B, H, rows,
    S, D]``. ``mask_from``: ``bias0`` = ``biases[0]`` is the additive key-mask bias ``[.., rows, 1, 1, S]`` (0 keep / -inf or -1e9 drop) |
    ``none`` | a callable ``mask_of(biases) -> bool [B, rows, 1, 1, S]`` (an engine carrying explicit mask rows closes over them). ``tri_bias``:
    ``bias1`` | ``bias0`` | callable -> the triangle bias ``[B|1, 1, H, S, S]`` (cast to fp32 for the flash kernel). ``scale``: None = ``D**-0.5``.
    ``serve_dtypes``: the compute dtypes the ADAPTER serves (default bf16/fp16; an engine whose pair stack runs fp32 passes ``("bfloat16",
    "float16", "float32")`` — the flash kernel then computes its products at ``input_precision`` = ``tf32`` unless ``kernel_kwargs`` say otherwise).
    ``stock_qblock``: when the call falls back, run ``stock`` over query sub-blocks of this many rows (bounds the eager logits transient exactly as
    :func:`attend_query_blocks` does) — pass the engine's chunk; None = one stock call. Launch bounds: the row batch is split so ``rows*H*S*D <
    2**31`` per launch (the kernels index with int32); a pair plane whose bias has more than ``2**31 - 1`` elements (``H*S*S``; H=4: S > 23,170)
    is served by the flash kernel PER QUERY BLOCK (:func:`flash_triattn_serve.triangle_attention_qblocks`; ``ROWPAIR_TRIATT_FLASH_QBLOCK`` queries
    per launch, default 2048, on the caller's bias view — no whole-plane fp32 copy; the kernel's own per-launch bias copy ``H*qblock*ceil16(S)``
    stays far below its int32 row-pitch bound; per-query arithmetic identical to one launch: schedule word ``triatt_flash_qblock``) unless
    ``ROWPAIR_TRIATT_INT32_GUARD=1`` restores the ``unsupported:bias_elems>int32`` refusal instead (``cueq`` keeps that refusal).
    Planes at or below the bound are unaffected by either setting. QUERY-BLOCK calls (``q [.., S_q, D]`` with ``k / v [.., S_k, D]``, bias ``[.., S_q,
    S_k]``, mask over ``S_k``; ``S_q != S_k`` — the sharded-bias schedule :func:`triatt_update_jit_`): the key count is read from ``k``; the flash
    path serves them (its per-query arithmetic is launch-independent) — inside :meth:`StageSlot.tiles` (``ROWPAIR_TRIATT_JIT_PREP=tile``) from the
    tile's operand prepared ONCE per window group (kind ``flash_tile``: the first window prepares, the others launch the kernel alone; bitwise the
    per-call launch), else per call; a ``tier:<word>`` row is not asked for them (whole rows only: schedule word ``triatt_core_tier_rect``, one
    stderr note) — whole-row calls are decided exactly as before. Reduction order: every query row's
    softmax runs over all S keys on this rank (the row-pair rule); the fused kernels are Tier 2 against the eager core (online softmax),
    run-to-run repeatable.
    ``staged``: the :class:`StageSlot` the core reads (default :data:`STAGE`, armed by :func:`triatt_update_`): inside an armed ``once`` span
    the kernel's bias operand is staged ONCE per plane (:class:`StagedBias`) -- the flash kernel's prepared fp32/16-bit plane
    (:func:`flash_triattn_serve.triangle_attention_prepped`; per query block above the int32 bound) or, for a tier word kernels.triattn resolves
    to ``triattn_native``, the sealed package's m1 staging -- and every row window launches the kernel alone: no per-window plane copies; outputs
    bitwise the per-call path's; a window the staged path cannot take steps aside BY NAME to the per-call statements (census
    ``triatt_stage_aside``). ``ROWPAIR_TRIATT_BIG`` (:func:`big_word`; tier words only): ``qblocks`` = the flash path per query block above the
    bound (default, as before) | ``native`` = the tier's triattn_native row serves such windows WHOLE (its member stages :func:`native_staged_bytes`)
    | ``auto`` = native when free device memory covers that staging + :data:`BIG_MARGIN_BYTES`, else qblocks; census ``triatt_big=<policy>:<why>``.
    Census ``triatt_cells=measured:<cell>|nearest:<cell>``: the kernels.triattn cell the tier word resolved on at the window's call class."""
    from ...kernels import flash_triattn_serve as F1
    word = (os.environ.get(ENV_TRIATT_CORE, "") or kernel).strip().lower()
    src = "env" if os.environ.get(ENV_TRIATT_CORE, "") else "kit"
    tier = word[len(TIER_PREFIX):] if word.startswith(TIER_PREFIX) and len(word) > len(TIER_PREFIX) else None
    if word not in CORE_KERNELS and tier is None:
        raise RowpairRefused(f"attention_core: kernel {word!r} is not one of {CORE_KERNELS} or 'tier:<word>'")
    if layout not in ("bnhsd", "bhnsd"):
        raise RowpairRefused(f"attention_core: layout {layout!r} (bnhsd | bhnsd)")
    L = ledger if ledger is not None else core_ledger(min_tokens, expected)
    cueq_fn = None
    if word == "cueq":
        try:
            from cuequivariance_torch import triangle_attention as cueq_fn          # noqa: N813
        except Exception as e:  # noqa: BLE001
            raise RowpairRefused(f"attention_core: cueq_missing ({type(e).__name__}: {str(e)[:120]})") from None
        try:
            import cuequivariance_torch as _cq
            L.impl = "cueq@%s" % getattr(_cq, "__version__", "?")
        except Exception:  # noqa: BLE001
            L.impl = "cueq"
    elif word == "torch":
        L.impl = "torch"
    elif tier is not None:                                                  # the tier door: kernels.triattn's row for <word> per row window (resolved once per shape class by
        L.impl = "kernels.triattn:" + word                                  #  attn.pair_fused.core_attention; a refusal steps aside BY NAME to the flash path below, counted)
    L.set("kernel", word)
    L.set("rowpair", 1)
    _CORE_STATE["kernel"] = word
    record_schedule(triatt_core=word, triatt_core_src=src)
    serve_dt = tuple(serve_dtypes)
    guard_big = int32_guard()                                               # process constants, read at bind: ROWPAIR_TRIATT_INT32_GUARD (refuse planes above the int32 bias
    qblock_big = flash_qblock()                                             #  bound) and ROWPAIR_TRIATT_FLASH_QBLOCK (queries per flash launch on such planes)
    big_pref = big_word()                                                   # ROWPAIR_TRIATT_BIG: qblocks | native | auto (tier words above the bound)
    slot_pinned = staged                                                    # where the staged plane operand lives: the given slot, else the calling thread's (stage_slot())

    def _count(served: bool):
        key = "served" if served else "fallback"
        with _CORE_STATE_LOCK:                                              # `+= 1` on a dict slot is not atomic across threads: without the lock two rank threads finishing
            _CORE_STATE[key] += 1                                           #  together can lose one count (census `calls=6 served=5 fallback=0`); processes never shared it
            record_schedule(triatt_core_served=_CORE_STATE["served"], triatt_core_fallback=_CORE_STATE["fallback"])

    def _stock_call(q, k, v, biases):
        if stock_qblock:
            return attend_query_blocks(stock, q, k, v, biases, int(stock_qblock))
        return stock(q, k, v, list(biases))

    rows_dim = -4 if layout == "bnhsd" else -3                              # the row-batch axis in the ENGINE's layout (q [.., rows, H, S, D] | [.., H, rows, S, D])

    def _win(t, r0, r1, rows):
        """rows r0:r1 of a tensor in the engine's layout when it carries the row axis (size == rows there), else the tensor (broadcast)."""
        if t is None or t.dim() < -rows_dim or int(t.shape[rows_dim]) != rows or (r0 == 0 and r1 == rows):
            return t
        idx = [slice(None)] * t.dim()
        idx[rows_dim] = slice(r0, r1)
        return t[tuple(idx)]

    def _to5(t):
        while t.dim() < 5:
            t = t.unsqueeze(0)
        return t

    def _bn(t):                                                             # engine layout -> [B, rows, H, S, D] (5-D, a view)
        t5 = _to5(t)
        return t5.transpose(1, 2) if layout == "bhnsd" else t5

    def _fallback(reason, q, k, v, biases):
        L.fallback(reason)
        _count(False)
        return _stock_call(q, k, v, biases)

    ready = {"ok": False}
    tier_state = {"off": None, "big": None, "rect": None}                   # the tier door's step-aside reason once it refused in this process (then the flash path serves);
                                                                            #  "big": noted once when a plane above the int32 bias bound went to the flash path by name;
                                                                            #  "rect": noted once when a query-block call (triatt_update_jit_) went to the flash path by name

    def _slot() -> StageSlot:
        return slot_pinned if slot_pinned is not None else stage_slot()

    def _tier_sel(qs, ks, ms):
        """kernels.triattn's Selection for the tier word at this window's call class (memo per class; None when it refused) + the census word
        ``triatt_cells`` = measured:<cell> | nearest:<cell> | refused:<kind>."""
        from ...attn import pair_fused as _PF
        cc, dtype, D_, H_, S_ = _PF._call_class(qs, ks)
        form = "mask_bias" if ms is not None else "bias_only"
        key = ("sel", cc, dtype, D_, H_, S_, form)
        if key not in tier_state:
            try:
                sel = _PF.resolve_tier_core(tier, cc, dtype, D_, H_, S_, form=form)
                tier_state[key] = sel
                record_schedule(triatt_cells=("measured:" if getattr(sel, "measured", False) else "nearest:") + str(getattr(sel, "cell", None)),
                                triatt_tier_row=str(getattr(sel, "row", None)))
            except Exception as e:  # noqa: BLE001 -- a Refusal by name (or no table for this class): the door's own per-call resolution names it
                from ...oom import is_oom
                if is_oom(e):
                    raise
                tier_state[key] = None
                record_schedule(triatt_cells="refused:%s" % getattr(e, "kind", type(e).__name__))
        return tier_state[key]

    def _big_policy(qs, ks, ms, H_, S_):
        """qblocks | native for a tier word above the int32 bound at this S (ROWPAIR_TRIATT_BIG; memo per S; census triatt_big=<policy>:<why>)."""
        key = ("big", S_)
        if key in tier_state:
            return tier_state[key]
        pol, why = "qblocks", ("default" if not os.environ.get(ENV_TRIATT_BIG, "").strip() else "env")
        if big_pref in ("native", "auto"):
            sel = _tier_sel(qs, ks, ms)
            row = getattr(sel, "row", None)
            if row != "triattn_native":
                pol, why = "qblocks", "tier_row=%s" % (row or "refused")
            elif big_pref == "native":
                pol, why = "native", "env"
            else:
                need = native_staged_bytes(H_, S_) + BIG_MARGIN_BYTES
                try:
                    free = int(torch.cuda.mem_get_info(qs.device)[0])
                except Exception as e:  # noqa: BLE001 -- no memory info on this device: policy falls to qblocks by name
                    from ...oom import is_oom
                    if is_oom(e):
                        raise
                    free = 0
                if free >= need:
                    pol, why = "native", "auto:free_%.1fGiB>=need_%.1fGiB" % (free / 2.0 ** 30, need / 2.0 ** 30)
                else:
                    pol, why = "qblocks", "auto:free_%.1fGiB<need_%.1fGiB" % (free / 2.0 ** 30, need / 2.0 ** 30)
        tier_state[key] = pol
        record_schedule(triatt_big="%s:%s" % (pol, why))
        _log_core_once("big:%s" % S_, "kernel=%s: plane above the int32 bias bound at S=%d -> %s (%s)" % (word, S_, pol, why))
        return pol

    def _staged_window(kind, qs, ks, vs, tb_src, ms, sc, stock5, S_, H_, D_):
        """Serve ONE row window from the plane's staged operand (staging it at the first window): ``kind`` = triattn_native | flash_triattn |
        flash_qblocks | flash_tile (one exchanged tile of the jit schedule: the query-block call's ``[.., S_q, S_k]`` operand, prepared at the group's
        first window). Returns (o, served: bool) or raises _StageAside (the per-call statements serve this window, by name)."""
        off = tier_state.get(("stage_off", kind, S_, D_))
        if off is not None:
            raise _StageAside(off)
        slot = _slot()
        sb = slot.get(kind, tb_src, S_, H_)
        if sb.ops is None and slot.no_room is not None and slot.no_room == sb.key:   # this plane was found without room at an earlier window: per call, by name (no re-measure per window)
            raise _NoRoom("no_room")
        if kind == "triattn_native":
            if sb.member is None:
                sb.member = _native_member(qs, ks, vs, tb_src, ms)
            if sb.ops is None:
                _room_or_aside(slot, sb, kind, H_, S_, qs.device, member=sb.member[0])
            o = _native_staged_call(sb, qs, ks, vs, tb_src, ms, sc, slot)
            L.serve(F1.shape_key(qs))
            slot.note_served(sb)
            return o, True
        before = L.served
        try:
            if kind in ("flash_triattn", "flash_tile"):                   # a plane staged once / ONE exchanged tile of the jit schedule ([.., S_q, S_k] read once from the buffer view)
                if sb.ops is None:
                    if kind == "flash_tile":
                        _room_or_aside(slot, sb, kind, H_, int(ks.shape[3]), qs.device, qblock=S_)
                    else:
                        _room_or_aside(slot, sb, kind, H_, S_, qs.device)
                    sb.ops = F1.prep_bias(tb_src, head_dim=D_, dtype=qs.dtype)
                    sb.nbytes = int(sb.ops.nbytes); sb.n_staged += 1
                    slot.note_staged(sb)
                o = F1.triangle_attention_prepped(qs, ks, vs, tb_src, sb.ops, mask=ms, scale=sc, ledger=L, stock=stock5, serve_dtypes=serve_dt, **kernel_kwargs)
            else:
                if sb.ops is None:
                    _room_or_aside(slot, sb, kind, H_, S_, qs.device, qblock=qblock_big)
                    sb.ops = F1.prep_bias_qblocks(tb_src, qblock=qblock_big, head_dim=D_, dtype=qs.dtype)
                    sb.nbytes = int(sum(p.nbytes for _, _, p in sb.ops)); sb.n_staged += 1
                    slot.note_staged(sb)
                o = F1.triangle_attention_prepped_qblocks(qs, ks, vs, tb_src, sb.ops, mask=ms, scale=sc, ledger=L, stock=stock5, serve_dtypes=serve_dt, **kernel_kwargs)
        except F1.Refusal as r:
            if r.kind == "prepped_unavailable":                             # the kernel module in use predates the prepared launch: per call, by name
                raise _StageAside("flash:%s" % r.kind) from None
            raise RowpairRefused("%s (kernel=flash_triattn): cannot run in this process — %s; run the kit with `--mode off`, or opt this "
                                 "lever out with kernel='torch' / %s=torch" % (getattr(F1, "LEVER", "F1.flash_triattn"), str(r)[:200], ENV_TRIATT_CORE)) from r
        except ValueError as e:                                             # a plane the prepared launch cannot index (int32 pitch): per call, by name
            raise _StageAside("flash:%s" % str(e)[:40]) from None
        if L.served > before:
            slot.note_served(sb)
        return o, L.served > before

    def core(q, k, v, biases):
        biases = list(biases)
        if word == "torch":
            return _fallback("kernel_torch", q, k, v, biases)
        if not ready["ok"]:
            _lever_ready(F1, "flash_triattn" if tier is not None else word, q)   # the tier door steps aside to the flash path: that path must be able to run
            ready["ok"] = bool(q.is_cuda)
        q5 = _bn(q)
        B, rows, H, S, D = (int(x) for x in q5.shape)
        SK = int(_bn(k).shape[3])                                           # keys per row (== S for a whole-row call; a QUERY-BLOCK call of the sharded-bias schedule —
        rect = SK != S                                                      #  triatt_update_jit_ — hands q [.., S_q, D] with k / v / bias / mask over all S_k keys: rect)
        tb = _pick(biases, tri_bias)
        if tb is None:
            return _fallback("unsupported:no_triangle_bias", q, k, v, biases)
        tb5 = _to5(tb)
        if layout == "bhnsd" and int(tb5.shape[2]) != 1:                    # [B, H, 1, S, S] in the engine layout -> [B, 1, H, S, S]
            tb5 = tb5.transpose(1, 2)
        if int(tb5.shape[1]) != 1 or int(tb5.shape[-1]) != SK or int(tb5.shape[-2]) != S:
            return _fallback("unsupported:triangle_bias_shape", q, k, v, biases)
        big = int(tb5.shape[-3]) * S * SK > INT32_MAX                      # more bias elements than an int32 indexes (H=4: S > 23,170; a query block's operand: H*S_q*S_k)
        if big and (guard_big or word == "cueq"):                           # ROWPAIR_TRIATT_INT32_GUARD=1 (refused by name) | cueq (its own int32 indexing: refused as before)
            return _fallback("unsupported:bias_elems>int32", q, k, v, biases)
        mb = _pick(biases, mask_from)
        mask5 = None
        if mb is not None:
            m5 = _bn(mb) if not callable(mask_from) else _to5(mb)
            if m5.dtype != torch.bool:
                m5 = (m5 == 0)                                                # additive mask bias (0 keep / -inf drop) -> bool key mask
            if int(m5.shape[-2]) != 1 or int(m5.shape[-1]) != SK:
                return _fallback("unsupported:mask_shape", q, k, v, biases)
            mask5 = m5
        bigpol = "qblocks"
        if big and tier is not None and not rect:                           # a TIER word above the int32 bias bound: ROWPAIR_TRIATT_BIG -- the flash path per query block
            bigpol = _big_policy(q5, _bn(k), mask5, int(tb5.shape[-3]), S)  #  (qblocks, default) or the tier's triattn_native row on the whole window (native | auto with memory)
            if bigpol == "qblocks" and tier_state["big"] is None:           # the flash_triattn path per query block BY NAME (below) -- never the engine statement, and no
                tier_state["big"] = "flash_triattn:bias_elems>int32"        #  provider row is asked (none stages a whole-plane fp32 bias)
                record_schedule(triatt_core_tier_big=tier_state["big"])
                _log_core_once("tier:big", "kernel=%s: a pair plane above the int32 bias bound (H*S*S = %d > 2**31-1) -> the flash_triattn path serves it per "
                                           "query block (the provider rows are not asked there: no whole-plane bias staging)" % (word, int(tb5.shape[-3]) * S * S))
            elif bigpol == "native" and tier_state["big"] is None:
                tier_state["big"] = "triattn_native:whole_window"
                record_schedule(triatt_core_tier_big=tier_state["big"])
        if rect and tier is not None and tier_state["rect"] is None:       # a query-block call (S_q != S_k; triatt_update_jit_): the provider rows serve whole rows only -> the
            tier_state["rect"] = "flash_triattn:qblock"                     #  flash_triattn path BY NAME for such calls (whole-row calls of the process still ask the tier row)
            record_schedule(triatt_core_tier_rect=tier_state["rect"])
            _log_core_once("tier:rect", "kernel=%s: query-block calls (S_q=%d of S_k=%d keys; ROWPAIR_TRIATT_BIAS=jit) -> the flash_triattn path serves them "
                                        "(the provider rows take whole rows)" % (word, S, SK))
        slot = _slot()
        once = slot.once() and word != "cueq"                               # inside triatt_update_(stage='once'): the kernel operand of THIS plane is staged once (below)
        tiled = rect and slot.tile() and word != "cueq"                     # a query-block call inside triatt_update_jit_(prep='tile'): the tile's operand prepared once per group (below)
        tb_src = tb5                                                        # the caller's plane view: staging reads it through its own strides (no copy here)
        pc = {}

        def tb_percall():
            """The per-call operand of today's statements: below the bound the whole-plane fp32 (flash) / contiguous (tier, cueq) copy, made
            at most once per core call; above it the caller's view as is."""
            if "v" not in pc:
                t = tb_src
                if not big:                                                  # the whole-plane fp32 bias, shared by every row window below
                    if word == "flash_triattn" and t.dtype != torch.float32:
                        t = t.float()
                    t = t.contiguous()
                pc["v"] = t
            return pc["v"]

        if not big:
            if not once and not tiled:
                tb5 = tb_percall()                                           # per_call (default): the copies happen here, before the loop, exactly as before
        else:                                                               # a plane above the int32 bias bound: NO whole-plane copy (2 x 4*H*S*S bytes here + the kernel's
            record_schedule(triatt_flash_qblock=qblock_big)                 #  own per-launch copy) — the kernel is launched per QUERY block on the caller's bias view; each
                                                                            #  launch's slice [B, 1, H, qblock, S] is upcast / laid out by the kernel's prep pass (small)
        sc = float(scale) if scale is not None else float(D) ** -0.5
        per = max(1, INT32_MAX // max(1, H * SK * D))                        # rows per launch: rows*H*S_k*D < 2**31 (int32 offsets in the kernels)
        outs = []
        for r0 in range(0, rows, per):
            r1 = min(rows, r0 + per)
            qs, ks, vs = (_bn(_win(t, r0, r1, rows)) for t in (q, k, v))
            ms = None
            if mask5 is not None:
                ms = mask5[:, r0:r1] if int(mask5.shape[1]) == rows else mask5
                ms = ms.expand(B, r1 - r0, 1, 1, SK).contiguous()

            def stock5(q_, k_, v_, b_, m_, s_, r0=r0, r1=r1):                # this row window through the engine core in the ENGINE's layout, back as [B, rows, H, S, D]
                ob = _stock_call(_win(q, r0, r1, rows), _win(k, r0, r1, rows), _win(v, r0, r1, rows), [_win(bb, r0, r1, rows) for bb in biases])
                return _bn(ob)

            dt_ok = str(qs.dtype).replace("torch.", "") in serve_dt
            kind = None                                                     # the staged / whole-window kind this window can take: triattn_native | flash_triattn | flash_qblocks
            if tier is not None and dt_ok and tier_state["off"] is None and not rect:
                if big:
                    kind = "triattn_native" if bigpol == "native" else ("flash_qblocks" if once else None)
                elif once:                                                  # a tier word stages once where kernels.triattn serves triattn_native (the m1 staging hoisted); any other
                    kind = "triattn_native" if getattr(_tier_sel(qs, ks, ms), "row", None) == "triattn_native" else None   # row keeps the provider door per call, by name
                    if kind is None:
                        slot.aside("unstaged_row:%s" % getattr(_tier_sel(qs, ks, ms), "row", None))
            elif word == "flash_triattn" and once and dt_ok and not rect:
                kind = "flash_qblocks" if big else "flash_triattn"
            if kind is None and tiled and dt_ok and not big and (word == "flash_triattn" or tier is not None):
                kind = "flash_tile"                                         # the jit schedule's tile: prepared once per (tile, window group), every window launches the kernel alone
            if kind == "triattn_native" and not once:                           # ROWPAIR_TRIATT_BIG=native|auto, per call: the provider door on the WHOLE window (the package stages per call)
                try:
                    from ...attn import pair_fused as _PF
                    _tier_sel(qs, ks, ms)
                    o = _PF.core_attention(qs, ks, vs, tb_src, ms, core=word, scale=sc)
                except Exception as e:  # noqa: BLE001 — the provider refused / cannot serve here: the flash path per query block BY NAME from now on
                    from ...oom import is_oom
                    if is_oom(e):
                        raise
                    tier_state["off"] = "%s:%s" % (type(e).__name__, str(e)[:80])
                    record_schedule(triatt_core_tier_off=tier_state["off"], triatt_big="qblocks:native_refused")
                    _log_core_once("tier:%s" % type(e).__name__, "kernel=%s stepped aside above the bound (%s: %s) -> the flash_triattn path per query block serves" % (word, type(e).__name__, str(e)[:160]))
                else:
                    L.serve(F1.shape_key(qs))
                    _count(True)
                    outs.append(o)
                    continue
            elif kind is not None:                                          # stage='once': the plane's operand staged at the first window, every window launches the kernel alone
                try:
                    o, served = _staged_window(kind, qs, ks, vs, tb_src, ms, sc, stock5, S, H, D)
                except _StageAside as a:                                    # by name: this window (and the rest of this class, unless the aside is this plane's only: no_room) takes the per-call statements below
                    if a.sticky:
                        tier_state[("stage_off", kind, S, D)] = str(a)
                    slot.aside(str(a)[:60])
                    _log_core_once("stage:%s:%s" % (kind, str(a)[:20]), "kernel=%s %s stepped aside for %s (%s) -> per-call staging" % (word, "jit prep=tile" if kind == "flash_tile" else "stage=once", kind, str(a)[:120]))
                except Exception as e:  # noqa: BLE001 — an out-of-memory is the caller's; anything else the staged glue raised: per call by name from now on
                    from ...oom import is_oom
                    if is_oom(e):
                        raise
                    tier_state[("stage_off", kind, S, D)] = "%s:%s" % (type(e).__name__, str(e)[:60])
                    slot.aside("%s:%s" % (kind, type(e).__name__))
                    _log_core_once("stage:%s:%s" % (kind, type(e).__name__), "kernel=%s %s stepped aside for %s (%s: %s) -> per-call staging" % (word, "jit prep=tile" if kind == "flash_tile" else "stage=once", kind, type(e).__name__, str(e)[:160]))
                else:
                    _count(served)
                    outs.append(o)
                    continue
            if tier is not None and not big and not rect and tier_state["off"] is None and dt_ok:
                try:                                                        # kernels.triattn's row for the word on this window ([B, rows, H, S, D] = the block's [B, I, H, J, D])
                    from ...attn import pair_fused as _PF
                    _tier_sel(qs, ks, ms)
                    o = _PF.core_attention(qs, ks, vs, tb_percall(), ms, core=word, scale=sc)
                except Exception as e:  # noqa: BLE001 — the provider refused / cannot serve here: the flash path BY NAME from now on (an out-of-memory is the caller's)
                    from ...oom import is_oom
                    if is_oom(e):
                        raise
                    tier_state["off"] = "%s:%s" % (type(e).__name__, str(e)[:80])
                    record_schedule(triatt_core_tier_off=tier_state["off"])
                    _log_core_once("tier:%s" % type(e).__name__, "kernel=%s stepped aside (%s: %s) -> the flash_triattn path serves" % (word, type(e).__name__, str(e)[:160]))
                else:
                    L.serve(F1.shape_key(qs))
                    _count(True)
                    outs.append(o)
                    continue
            if word == "cueq":
                try:
                    o = cueq_fn(qs, ks, vs, tb_percall(), mask=ms, scale=sc)
                except Exception as e:  # noqa: BLE001 — a configuration cuEquivariance rejects is served by the engine core, named; an out-of-memory is the caller's
                    from ...oom import is_oom
                    if is_oom(e):
                        raise
                    L.fallback("cueq_unsupported:%s" % type(e).__name__)
                    _log_core_once("cueq:%s" % type(e).__name__, "cueq rejected a call (%s: %s) -> the engine core serves such calls" % (type(e).__name__, str(e)[:160]))
                    _count(False)
                    outs.append(stock5(qs, ks, vs, tb5, ms, sc))
                    continue
                L.serve(F1.shape_key(qs))
                _count(True)
            else:
                before = L.served
                try:
                    if big:                                                 # per query block on the caller's bias view (H*qblock*ceil16(S) << 2**31 per launch); gates / events / ONE serve per window as below
                        o = F1.triangle_attention_qblocks(qs, ks, vs, tb_percall(), mask=ms, scale=sc, qblock=qblock_big, ledger=L, stock=stock5, serve_dtypes=serve_dt, **kernel_kwargs)
                    else:
                        o = F1.triangle_attention(qs, ks, vs, tb_percall(), mask=ms, scale=sc, ledger=L, stock=stock5, serve_dtypes=serve_dt, **kernel_kwargs)
                except F1.Refusal as r:                                     # the kernel's picked AND safe settings failed to build: the lever cannot run here
                    raise RowpairRefused("%s (kernel=flash_triattn): cannot run in this process — %s; run the kit with `--mode off`, or opt this "
                                         "lever out with kernel='torch' / %s=torch" % (getattr(F1, "LEVER", "F1.flash_triattn"), str(r)[:200], ENV_TRIATT_CORE)) from r
                _count(L.served > before)
            outs.append(o)
        o5 = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)           # [B, rows, H, S, D]
        if layout == "bhnsd":
            o5 = o5.transpose(1, 2)
        return o5.reshape(tuple(q.shape[:-1]) + (int(o5.shape[-1]),))      # the engine's own rank / layout back

    core.kernel = word
    core.ledger = L
    return core


def _lever_ready(F1, word: str, q) -> None:
    """The fused lever must be ABLE to run in this process, else its refusal — never a silent engine-core path: ``q`` on a CUDA device, and,
    for the flash kernel, torch / triton / a cc >= 8.0 device / the carried kernel importable (:func:`flash_triattn_serve.require`)."""
    lever = getattr(F1, "LEVER", "F1.flash_triattn")
    if not q.is_cuda:
        raise RowpairRefused("%s (kernel=%s): cannot run in this process — q is on %s, not a CUDA device; run the kit with `--mode off`, or opt "
                             "this lever out with kernel='torch' / %s=torch" % (lever, word, q.device.type, ENV_TRIATT_CORE))
    if word == "flash_triattn":
        try:
            F1.require()
        except F1.Refusal as r:
            raise RowpairRefused("%s (kernel=flash_triattn): cannot run in this process — %s; run the kit with `--mode off`, or opt this lever out "
                                 "with kernel='torch' / %s=torch" % (lever, str(r)[:200], ENV_TRIATT_CORE)) from None


def emit_core_line(tag: str, ledger=None, **evidence) -> str:
    """Print the core's ONE triangle-attention LEVER line (``[<tag>] LEVER name=F1.flash_triattn … origin=core … kernel=<word> rowpair=1``)."""
    from ...kernels import flash_triattn_serve as F1
    from ...report import emit
    L = ledger if ledger is not None else core_ledger()
    if _CORE_STATE["kernel"] in (None, "flash_triattn"):
        return F1.emit_line(L, tag, **evidence)
    if L.origin is None:
        L.origin = "core"
    return emit(L.line(tag, **evidence))


def describe_core(ledger=None) -> dict:
    """Plain data for a manifest: the core ledger's fields + served/fallback call counts."""
    L = ledger if ledger is not None else core_ledger()
    d = dict(L.fields())
    d.update(kernel=_CORE_STATE["kernel"], core_served=_CORE_STATE["served"], core_fallback=_CORE_STATE["fallback"])
    return d

