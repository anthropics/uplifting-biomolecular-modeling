"""L2 driver — one PAIR BLOCK / PAIR STACK forward on a PRE-SHARDED pair tensor. Every triangle-update pair stack (Pairformer, MSA-module pair
stack, template pair stack, confidence pair stack) is the same block: triangle multiplication outgoing -> incoming -> triangle attention
starting -> ending -> pair transition (+ for Pairformer blocks: attention-pair-bias of the single representation and the single transition), each
``z += op(z)`` (row-wise dropout is equality at inference). This module sequences those updates on this rank's row shard ``z_shard[R, N, C]``
and owns the ORIENTATION of the ending attention (rows of z^T: distributed transpose -> the starting-form update -> transpose back); it never
gathers z. The engine's modules enter as the callables of :class:`PairBlockFns` (built by :func:`bind` from the module-shaped statements of
:mod:`.trimul` / :mod:`.triatt` and a row-local transition, or supplied by the adapter directly).

API:
    PairBlockFns(trimul_out, trimul_in, triatt_start, triatt_end, transition, apb=None, single_transition=None)
                                          the five pair updates as ``fn(z_shard, mask_shard, layout) -> z_shard`` (IN PLACE, residual included;
                                          ``triatt_end`` receives rows of z^T and the rows of mask^T) + the optional single-representation callables
                                          ``apb(s, z_shard, layout) -> s`` and ``single_transition(s) -> s`` (s REPLICATED)
    bind(trimul_out=TriMulFns, trimul_in=TriMulFns, triatt_start=TriAttFns, triatt_end=TriAttFns, transition=fn, *, chunk, ...)
                                          -> PairBlockFns bound to the streamed schedules of :func:`opt_core.mem.rowpair.trimul.trimul_update_` /
                                          :func:`opt_core.mem.rowpair.triatt.triatt_update_` and :func:`transition_update_`
    transition_update_(fn, z_shard, mask_shard, rows)
                                          ``z[i] += fn(z[i], mask[i, :, None])`` per LOCAL row block (the pair transition, streamed)
    pair_block_(fns, z_shard, mask_shard, layout, *, s=None, maskT_shard=None, ...) -> (z_shard, s)
    pair_stack_(blocks, z_shard, mask_shard, layout, *, s=None, block_callback=None, ...) -> (z_shard, s)
    mask_transposed(mask_shard, layout)   rows of mask^T for the ending attention (once per stack call)

Orientation + storage schedule of the ending attention (``u`` := bytes of the shard):
    default            ``zT = transpose(z)`` (:func:`opt_core.mem.rowpair.ring.transpose_shard`: +1u, staging per its budget) -> the input
                       shard's storage is RELEASED while the update runs on zT when this module holds its only owner (``end_free``,
                       ``ROWPAIR_TRIATT_END_FREE``, default on; skipped BY NAME in the census when the tensor does not own its whole storage —
                       ``skipped:not-storage-owner`` — or sits on foreign memory torch cannot release — ``skipped:storage_not_resizable``)
                       -> ``zT += att(zT)`` -> storage regrown, ``transpose(zT, out=z_shard)``: the result is the caller's tensor object
                       (``reuse_storage=True``; False = a fresh tensor as the stock statement returns). Peak 2u around the transposes, 1u +
                       the update's transients during the attention.
    transpose_inplace  (``ROWPAIR_TRANSPOSE_INPLACE=1``) pairwise block swaps into the shard's own storage both ways
                       (:func:`opt_core.mem.rowpair.ring.transpose_shard_inplace_`): no second shard at any point.
Replicated-by-design (named, checked by the adapter's checksum census): the single representation ``s``, the single mask, the gathered
triangle biases ``[N, N, H]`` (unless ``ROWPAIR_TRIATT_BIAS=jit``: :func:`bind` then binds :func:`opt_core.mem.rowpair.triatt.triatt_update_jit_`,
which keeps each rank's bias ROWS and exchanges query blocks of the plane just in time — no replicated plane); the pair mask travels as
row shards (``mask_shard = pair_mask[r0:r1, :]``, its transpose once per stack call).
Refusals by name: P == 1 / replicated layouts (the structural n_gpu=1 rule), shards whose shape disagrees with the layout, training mode is
the adapter's refusal (this driver applies no dropout).
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import Layout, require_sharded, transpose_shards, zadd, zblocks, zlen, env_flag as _env_flag
from .evidence import record_schedule
from .trimul import ContractStats, TriMulFns, trimul_update_

__all__ = ["PairBlockFns", "bind", "transition_update_", "pair_block_", "pair_stack_", "mask_transposed", "env_flag", "transition_rows_word", "ENV_TRANSITION_ROWS"]


def env_flag(name: str, default: bool) -> bool:
    """``ROWPAIR_<name>`` as a 0/1 flag (unset or empty -> ``default``): :func:`opt_core.mem.rowpair.dist.env_flag` on the prefixed name."""
    return _env_flag("ROWPAIR_" + name, default)


class PairBlockFns(object):
    """One pair block's updates. Pair updates: ``fn(z_shard, mask_shard, layout) -> z_shard`` — IN PLACE on the ``[R, N, C]`` shard they are given
    (residual included), returning it; ``triatt_end`` is handed this rank's rows of z^T and of mask^T (the driver transposes). Single representation
    (Pairformer blocks): ``apb(s, z_shard, layout) -> s`` (attention-pair-bias update of the replicated single representation from LOCAL pair
    rows — local query rows + all-gather, e.g. :func:`opt_core.mem.rowpair.transition.apb_local_queries`) and ``single_transition(s) -> s``."""

    def __init__(self, trimul_out: Callable, trimul_in: Callable, triatt_start: Callable, triatt_end: Callable, transition: Callable,
                 apb: Optional[Callable] = None, single_transition: Optional[Callable] = None):
        self.trimul_out, self.trimul_in, self.triatt_start, self.triatt_end, self.transition = trimul_out, trimul_in, triatt_start, triatt_end, transition
        self.apb, self.single_transition = apb, single_transition


def transition_update_(fn: Callable[[object, object], object], z_shard, mask_shard, rows: int, *, add: bool = True, residual_in_fn: bool = False):
    """The pair transition streamed over LOCAL row blocks of ``rows`` rows: ``z_shard[i0:i1] (+)= fn(z_shard[i0:i1], mask_u[i0:i1])`` with ``mask_u
    = mask_shard[..., None]`` (``[rows, N, 1]``; ones when ``mask_shard`` is None). Row-local: no communication, no shard-sized buffer.
    ``residual_in_fn`` (default False = the delta form above): ``fn`` returns the UPDATED rows ``x + delta`` — written in place into
    the block it was handed (a kernel's residual epilogue: one pass over the block instead of delta + add) or as a new tensor (then copied in);
    :func:`bind` sets it from the bound callable's ``residual_in_fn`` attribute (:func:`.transition.transition_face_fn` ``residual=True``)."""
    R = zlen(z_shard)
    mask_u = (z_shard.new_ones((R, int(z_shard.shape[1]))) if mask_shard is None else mask_shard).unsqueeze(-1)
    if residual_in_fn:
        for i0, i1 in zblocks(R, max(1, int(rows))):
            blk = z_shard[i0:i1]
            y = fn(blk, mask_u[i0:i1])
            if y is not blk and (getattr(y, "data_ptr", None) is None or y.data_ptr() != blk.data_ptr() or tuple(y.shape) != tuple(blk.shape)):
                blk.copy_(y)                                           # the callable answered out of place (e.g. its named engine fallback): write the rows
            del y, blk
        return z_shard
    for i0, i1 in zblocks(R, max(1, int(rows))):
        delta = fn(z_shard[i0:i1], mask_u[i0:i1])
        if add:
            zadd(z_shard, i0, i1, delta)
        else:
            z_shard[i0:i1] = delta
        del delta
    return z_shard


ENV_TRANSITION_ROWS = "ROWPAIR_TRANSITION_ROWS"        # bind(transition_rows=None) reads it: unset = the attention chunk (today); an int = the transition's own row block


def transition_rows_word(rows=None) -> Optional[int]:
    """``rows`` given wins; else ``ROWPAIR_TRANSITION_ROWS``: unset / empty / ``0`` / ``chunk`` -> None (the attention chunk, today); a positive integer
    -> that many rows; anything else refused by name."""
    if rows is not None:
        return max(1, int(rows))
    import os
    v = os.environ.get(ENV_TRANSITION_ROWS, "").strip().lower()
    if v in ("", "0", "chunk", "none", "off"):
        return None
    try:
        n = int(v)
    except ValueError:
        raise RowpairRefused(f"{ENV_TRANSITION_ROWS}={v!r}: a positive row count (or unset / chunk) is required") from None
    if n < 1:
        raise RowpairRefused(f"{ENV_TRANSITION_ROWS}={v!r}: a positive row count (or unset / chunk) is required")
    return n


def bind(*, trimul_out: TriMulFns, trimul_in: TriMulFns, triatt_start: TriAttFns, triatt_end: TriAttFns,
         transition: Callable[[object, object], object], chunk: int, apb: Optional[Callable] = None, single_transition: Optional[Callable] = None,
         stream: bool = True, tb_rows: Optional[int] = None, trimul_kw: Optional[dict] = None, stats: Optional[dict] = None,
         triatt_bias: Optional[str] = None, transition_rows: Optional[int] = None) -> PairBlockFns:
    """PairBlockFns whose pair updates are the streamed schedules of this package: tri-mult out/in = :func:`opt_core.mem.rowpair.trimul.trimul_update_`
    (``trimul_kw``: its schedule knobs ``RB grid mm_rows a2a_chunks`` and the REQUIRED ``inplace_chunk`` = the engine's in-place triangle-multiplication column chunk), tri-att start/end = :func:`opt_core.mem.rowpair.triatt.triatt_update_`
    (row-batch blocks of ``chunk`` rows = the engine's pinned attention chunk; ``stream`` / ``tb_rows`` as there), transition =
    :func:`transition_update_` over blocks of ``chunk`` rows (``transition(x_rows, mask_u_rows) -> delta``). ``stats`` (dict): per-op
    :class:`opt_core.mem.rowpair.trimul.ContractStats` are stored under ``trimul_out`` / ``trimul_in``.
    ``triatt_bias`` (``gather`` | ``jit``; None = ``ROWPAIR_TRIATT_BIAS``, unset = ``gather``): ``gather`` binds :func:`triatt_update_` (the replicated
    bias plane — today's schedule, nothing recorded); ``jit`` binds :func:`opt_core.mem.rowpair.triatt.triatt_update_jit_` (the plane stays sharded, query
    blocks exchanged just in time) when BOTH ``TriAttFns`` carry its optional fields — else the gather schedule BY NAME (schedule word
    ``triatt_bias=gather:fields_missing:<names>``).
    ``transition_rows`` (None = ``chunk``, today's schedule byte for byte): the transition's OWN row block, decoupled from the
    attention chunk — the transition is row-local (LayerNorm + three GEMMs per row), so e.g. 256-row blocks while the tri-attention stays at
    ``chunk`` = 16 cost only the block's hidden transient ``rows*N*4c`` and save ``R/chunk - R/rows`` launches per block; recorded as census
    ``transition_rows`` when given. A bound ``transition`` callable carrying ``residual_in_fn = True`` (:func:`.transition.transition_face_fn`
    with ``residual=True``) is driven in the residual form of :func:`transition_update_` (census ``transition_residual=fn``)."""
    from .triatt import triatt_update_, triatt_update_jit_, triatt_bias_word, jit_fields_missing   # noqa: WPS433 — the triangle-attention driver (and the kernel entry it serves with) loads with the pair stacks that bind one
    tkw = dict(trimul_kw or {})
    chunk = max(1, int(chunk))
    triatt_ = triatt_update_
    if triatt_bias_word(triatt_bias) == "jit":
        missing = sorted(set(jit_fields_missing(triatt_start)) | set(jit_fields_missing(triatt_end)))
        if missing:
            record_schedule(triatt_bias="gather:fields_missing:" + "+".join(missing))
        else:
            triatt_ = triatt_update_jit_                                   # same call shape; qblock / group / prefetch from ROWPAIR_TRIATT_JIT_*

    def _st(key):
        if stats is None:
            return None
        return stats.setdefault(key, ContractStats())

    res_in_fn = bool(getattr(transition, "residual_in_fn", False))
    if transition_rows is None:
        transition_rows = transition_rows_word()                       # ROWPAIR_TRANSITION_ROWS (unset -> None: the chunk, today)
    if transition_rows is None and not res_in_fn:
        transition_bound = lambda z, m, lay: transition_update_(transition, z, m, chunk)          # noqa: E731 — today's binding, byte for byte
    else:
        trows = chunk if transition_rows is None else max(1, int(transition_rows))
        facts = {}
        if transition_rows is not None:
            facts["transition_rows"] = trows
        if res_in_fn:
            facts["transition_residual"] = "fn"
        record_schedule(**facts)
        if res_in_fn:
            transition_bound = lambda z, m, lay: transition_update_(transition, z, m, trows, residual_in_fn=True)   # noqa: E731
        else:
            transition_bound = lambda z, m, lay: transition_update_(transition, z, m, trows)   # noqa: E731

    return PairBlockFns(
        trimul_out=lambda z, m, lay: trimul_update_(trimul_out, z, m, lay, outgoing=True, stats=_st("trimul_out"), **tkw),
        trimul_in=lambda z, m, lay: trimul_update_(trimul_in, z, m, lay, outgoing=False, stats=_st("trimul_in"), **tkw),
        triatt_start=lambda z, m, lay: triatt_(triatt_start, z, m, lay, rows=chunk, stream=stream, tb_rows=tb_rows),
        triatt_end=lambda zT, mT, lay: triatt_(triatt_end, zT, mT, lay, rows=chunk, stream=stream, tb_rows=tb_rows),
        transition=transition_bound,
        apb=apb, single_transition=single_transition)


def mask_transposed(mask_shard, layout: Layout):
    """This rank's rows of mask^T (``[R, N]``) from its rows of the pair mask — the mask the ending attention sees."""
    if mask_shard is None:
        return None
    return transpose_shards(mask_shard.unsqueeze(-1).contiguous(), layout)[..., 0].contiguous()


def _check(z_shard, mask_shard, layout: Layout, what: str) -> None:
    require_sharded(layout, what)
    if getattr(z_shard, "dim", lambda: 3)() != 3 or zlen(z_shard) != layout.R or int(z_shard.shape[1]) != layout.N:
        raise RowpairRefused(f"{what}: z_shard {tuple(z_shard.shape)} vs layout R={layout.R} N={layout.N} ([R, N, C] expected; a kit's "
                             "[1, R, N, C] shard passes z[0])")
    if mask_shard is not None and tuple(mask_shard.shape) != (layout.R, layout.N):
        raise RowpairRefused(f"{what}: mask_shard {tuple(mask_shard.shape)} vs (R, N)=({layout.R}, {layout.N})")


def pair_block_(fns: PairBlockFns, z_shard, mask_shard, layout: Layout, *, s=None, maskT_shard=None, transition_mask: bool = True,
                transpose_inplace: Optional[bool] = None, end_free: Optional[bool] = None, reuse_storage: bool = True, census: bool = True):
    """One pair block on this rank's rows: ``z = trimul_out(z); z = trimul_in(z); z = triatt_start(z); z = T^-1(triatt_end(T(z))); z =
    transition(z)`` (all in place on the shard), then — when ``fns.apb`` is set — ``s = apb(s, z); s = single_transition(s)``. Returns ``(z_shard,
    s)``; with ``reuse_storage`` (default) the returned shard IS the caller's tensor object. ``maskT_shard``: rows of mask^T
    (:func:`mask_transposed`; computed here when None and a mask is given). ``transition_mask=False`` runs the transition unmasked (for an engine whose
    pair transition takes no mask). ``transpose_inplace`` / ``end_free``: module docstring (None -> ``ROWPAIR_TRANSPOSE_INPLACE`` default off /
    ``ROWPAIR_TRIATT_END_FREE`` default on)."""
    _check(z_shard, mask_shard, layout, "pair_block_")
    tin = env_flag("TRANSPOSE_INPLACE", False) if transpose_inplace is None else bool(transpose_inplace)
    efree = env_flag("TRIATT_END_FREE", True) if end_free is None else bool(end_free)
    if maskT_shard is None and mask_shard is not None:
        maskT_shard = mask_transposed(mask_shard, layout)
    z = fns.trimul_out(z_shard, mask_shard, layout)
    z = fns.trimul_in(z, mask_shard, layout)
    z = fns.triatt_start(z, mask_shard, layout)
    # ---- ending attention on rows of z^T
    freed = "off"
    if tin:
        from .ring import transpose_shard_inplace_                        # noqa: WPS433
        zT = transpose_shard_inplace_(z, layout)
        form = "inplace"
    else:
        from .ring import transpose_shard                                 # noqa: WPS433
        from .shard import owns_whole_storage, regrow_storage_, release_storage_, storage_resizable   # noqa: WPS433
        zT = transpose_shard(z, layout)
        form = "p2p"
        if efree and reuse_storage:
            if not owns_whole_storage(z):
                freed = "skipped:not-storage-owner"
            elif not storage_resizable(z):
                freed = "skipped:storage_not_resizable"                    # foreign memory torch cannot release: a memory fact (named), never a failure
            else:
                nbytes = release_storage_(z)
                freed = "applied"
    zT = fns.triatt_end(zT, maskT_shard, layout)
    if tin:
        z = transpose_shard_inplace_(zT, layout)
    else:
        if freed == "applied":
            regrow_storage_(z, nbytes)
        z = transpose_shard(zT, layout, out=z if reuse_storage else None)
        del zT
    if census:
        record_schedule(pairstack_transpose=form, pairstack_end_free=freed, pairstack_reuse_storage=bool(reuse_storage))
    # ---- transition (+ single representation)
    z = fns.transition(z, mask_shard if transition_mask else None, layout)
    if fns.apb is not None:
        if s is None:
            raise RowpairRefused("pair_block_: fns.apb is set but s is None (Pairformer blocks carry the single representation)")
        s = fns.apb(s, z, layout)
        if fns.single_transition is not None:
            s = fns.single_transition(s)
    return z, s


def pair_stack_(blocks: Sequence[PairBlockFns], z_shard, mask_shard, layout: Layout, *, s=None,
                block_callback: Optional[Callable[[int, object, object], None]] = None, between_blocks: Optional[Callable[[int], None]] = None,
                **block_kw) -> Tuple[object, object]:
    """``for b in blocks: z, s = pair_block_(b, z, ...)`` on this rank's rows — the shard is never gathered between blocks. The rows of mask^T
    are computed once. ``between_blocks(i)`` runs before block i (e.g. the engine's cache clearing); ``block_callback(i, s, z_shard)`` after it
    (progress, trunk checkpoints). Returns ``(z_shard, s)``."""
    _check(z_shard, mask_shard, layout, "pair_stack_")
    maskT = mask_transposed(mask_shard, layout) if mask_shard is not None else None
    record_schedule(pairstack_blocks=len(blocks))
    times = _block_times_log(z_shard) if os.environ.get(ENV_PAIRSTACK_TIMES, "").strip() not in ("", "0") else None   # opt-in per-block wall log (probes); unset: nothing here
    z = z_shard
    for i, b in enumerate(blocks):
        if between_blocks is not None:
            between_blocks(i)
        z, s = pair_block_(b, z, mask_shard, layout, s=s, maskT_shard=maskT, census=(i == 0), **block_kw)
        if times is not None:
            times(i, len(blocks))
        if block_callback is not None:
            block_callback(i, s, z)
    return z, s


ENV_PAIRSTACK_TIMES = "ROWPAIR_PAIRSTACK_TIMES"                          # "1": pair_stack_ logs every block's wall time + allocator state on this rank (a device sync per
                                                                         #  block; probes / anchors only -- unset by default: no sync, no line)


def _block_times_log(z_shard):
    """-> ``f(i, n)`` logging ``[pairstack] block i/n <dt>s alloc <GiB> peak <GiB> reserved <GiB>`` per block through the rank's logger."""
    import time as _time
    from .dist import comm
    dev = getattr(z_shard, "device", None)
    cuda = bool(getattr(dev, "type", "") == "cuda")
    if cuda:
        torch.cuda.synchronize(dev)
    st = {"t": _time.perf_counter()}

    def f(i, n):
        if cuda:
            torch.cuda.synchronize(dev)
        t = _time.perf_counter()
        extra = ""
        if cuda:
            extra = " alloc %.2f GiB peak %.2f GiB reserved %.2f GiB" % (torch.cuda.memory_allocated(dev) / 2.0 ** 30, torch.cuda.max_memory_allocated(dev) / 2.0 ** 30,
                                                                      torch.cuda.memory_reserved(dev) / 2.0 ** 30)
        try:
            comm().log("[pairstack] block %d/%d %.2fs%s" % (i + 1, n, t - st["t"], extra))
        except Exception as e:  # noqa: BLE001 -- no communicator logger in this process: print
            from ..oom import is_oom
            if is_oom(e):
                raise
            print("[pairstack] block %d/%d %.2fs%s" % (i + 1, n, t - st["t"], extra), flush=True)
        st["t"] = _time.perf_counter()
    return f
