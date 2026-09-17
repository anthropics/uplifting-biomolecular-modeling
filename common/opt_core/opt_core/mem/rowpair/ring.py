"""Peer exchanges of a row-sharded pair tensor with LEADING dims — ``z_loc[*, n_loc, N, C]`` = rows ``r0:r1`` of ``z[*, N, N, C]`` — built only
on the comm surface's ``p2p`` / ``p2p_start`` (:mod:`.dist`): distributed transposes of the two token dims (all-at-once, peer-streamed above a
byte budget, in place, banded) and sub-block ring passes. Pure data movement: bit-preserving, no arithmetic. Every rank calls every primitive
the same number of times (ranks whose own block / window is empty included). The dim-0 ``[R, N, C]`` forms over all-to-all windows (store
duck-types, output row windows) are :func:`opt_core.mem.rowpair.dist.transpose_shards` / ``transpose_blocks`` / ``ring_blocks``.

API (``layout`` = :class:`opt_core.mem.rowpair.dist.Layout`; its ``parts r0 r1 n_loc n_max`` name the partition):
    transpose_shard(z_loc, layout, out=None)                  rows r0:r1 of ``z^T`` (``out[.., i, j, :] = z[.., j, i, :]``, ``[*, n_loc, N, C]``). Block
                                                              ``(r, s) = z[r-rows, s-cols]`` goes to rank s transposed; all P-1 blocks in ONE ``p2p``
                                                              (transient = out + send + recv staging, ~2(P-1)/P^2 z-eq) unless
                                                              :func:`transpose_budget_exceeded` -> :func:`transpose_shard_streamed`
    transpose_shard_streamed(z_loc, layout, peers_in_flight)  the same values, ``peers_in_flight`` peers per exchange on a shifted schedule (step d:
                                                              send to ``rank+d``, receive from ``rank-d``): transient = out + peers*(send+recv block)
    transpose_shard_inplace_(z_loc, layout, peers_in_flight)  ``z_loc`` (contiguous) overwritten with rows r0:r1 of ``z^T``: pairwise block swaps
                                                              on the XOR schedule (block (r,s) and its replacement (z[s-rows, r-cols])^T have the
                                                              same shape); transient = peers*(send+recv block) + one diagonal temp
    transpose_band(z_loc, layout, windows)                    a BAND of ``z^T`` rows per rank: ``windows[q] = (w0, w1)`` global columns of z rank q
                                                              wants as rows of ``z^T`` -> ``[*, w1-w0, N, C]`` (None when the own window is empty)
    ring_pass(own, layout, shapes, dtype=, device=)           one pass of a sub-block ring: yields ``(src, block)`` for ``src = rank, rank-1, ...``
                                                              skipping sources whose ``shapes[src]`` is None; the exchange for the next block is
                                                              STARTED before the current block is yielded (``p2p_start``) and waited after
    transpose_budget_exceeded(z_loc, layout) / transpose_send_bytes(z_loc, layout)

Schedule knobs (arguments or environment; every rank derives the same value): ``ROWPAIR_TRANSPOSE_BUDGET_GB`` (default 24: above it
``transpose_shard`` streams), ``ROWPAIR_TRANSPOSE_PEERS`` (peers in flight when streaming, default 1). The form used is recorded in the
schedule census (``transpose_form``).
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import Layout, comm, env_float, env_int, require_schedule_dtype

__all__ = ["transpose_shard", "transpose_shard_streamed", "transpose_shard_inplace_", "transpose_band", "ring_pass", "transpose_budget_exceeded",
           "transpose_send_bytes", "TRANSPOSE_BUDGET_GB"]

TRANSPOSE_BUDGET_GB = 24.0


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _lead_C(z_loc) -> Tuple[List[int], int]:
    if z_loc.dim() < 3:
        raise RowpairRefused(f"ring: a [*, n_loc, N, C] shard is required, got shape {tuple(z_loc.shape)}")
    return list(z_loc.shape[:-3]), int(z_loc.shape[-1])


def _check_shard(z_loc, layout: Layout, what: str) -> None:
    if int(z_loc.shape[-3]) != layout.n_loc or int(z_loc.shape[-2]) != layout.N:
        raise RowpairRefused(f"{what}: shard shape {tuple(z_loc.shape)} does not match layout rows {layout.r0}:{layout.r1} of N={layout.N}")


def _transpose_out(z_loc, layout: Layout, out):
    """Allocate — or validate a caller-provided — output buffer ``[*, n_loc, N, C]`` for the distributed transposes."""
    lead, C = _lead_C(z_loc)
    want = lead + [layout.n_loc, layout.N, C]
    if out is None:
        return torch.empty(want, dtype=z_loc.dtype, device=z_loc.device)
    if list(out.shape) != want or out.dtype != z_loc.dtype or not out.is_contiguous():
        raise RowpairRefused(f"transpose out= must be a contiguous {want} {z_loc.dtype} tensor, got {tuple(out.shape)} {out.dtype}")
    if out.untyped_storage().data_ptr() == z_loc.untyped_storage().data_ptr():
        raise RowpairRefused("transpose out= must not alias the input shard")
    return out


def transpose_send_bytes(z_loc, layout: Layout) -> int:
    """Bytes this rank stages for an all-at-once transpose (one direction): ``lead * n_loc * (N - n_loc) * C * esize``."""
    lead, C = _lead_C(z_loc)
    n_lead = 1
    for d in lead:
        n_lead *= int(d)
    return int(n_lead * layout.n_loc * (layout.N - layout.n_loc) * C * z_loc.element_size())


def transpose_budget_exceeded(z_loc, layout: Layout, budget_gb: Optional[float] = None) -> bool:
    """True when the all-at-once staging exceeds ``budget_gb`` (``ROWPAIR_TRANSPOSE_BUDGET_GB``, default 24) at P > 1. Ranks own different row
    counts, so the DECISION is taken on ``n_max`` (identical on every rank): every rank picks the same form."""
    if layout.P <= 1:
        return False
    lead, C = _lead_C(z_loc)
    n_lead = 1
    for d in lead:
        n_lead *= int(d)
    worst = n_lead * layout.n_max * (layout.N - min(q1 - q0 for q0, q1 in layout.parts)) * C * z_loc.element_size()
    budget = env_float("ROWPAIR_TRANSPOSE_BUDGET_GB", TRANSPOSE_BUDGET_GB) if budget_gb is None else float(budget_gb)
    return worst > budget * 1e9


def _record(form: str) -> None:
    from .evidence import record_schedule
    record_schedule(transpose_form=form)


def transpose_shard(z_loc, layout: Layout, out=None, *, budget_gb: Optional[float] = None, peers_in_flight: Optional[int] = None):
    """Distributed transpose of the two token dims: input rows r0:r1 of z (``[*, n_loc, N, C]``); output rows r0:r1 of ``z^T``
    (``[*, n_loc, N, C]``, ``out[.., i, j, :] = z[.., j, i, :]``). Block ``(r, s) = z[r-rows, s-cols]`` goes to rank s, which places its transpose at
    ``out[s-rows (local), r-cols]``. All P-1 blocks go in ONE ``p2p`` call: transient = out (1/P z-eq) + send + recv staging (2(P-1)/P^2 z-eq).
    Above the budget (:func:`transpose_budget_exceeded`) this dispatches to :func:`transpose_shard_streamed` (``peers_in_flight`` =
    ``ROWPAIR_TRANSPOSE_PEERS``, default 1): identical values, smaller transient. ``out``: optional pre-allocated contiguous result buffer
    (must not alias ``z_loc``). P == 1: the local transpose into ``out``."""
    _check_shard(z_loc, layout, "transpose_shard")
    if transpose_budget_exceeded(z_loc, layout, budget_gb):
        k = env_int("ROWPAIR_TRANSPOSE_PEERS", 1) if peers_in_flight is None else int(peers_in_flight)
        return transpose_shard_streamed(z_loc, layout, peers_in_flight=k, out=out)
    cm = comm()
    lead, C = _lead_C(z_loc)
    out = _transpose_out(z_loc, layout, out)
    st, site = None, None
    if layout.P > 1:
        _record("all")
        st = cm.staging                                     # (P-1) send + (P-1) recv blocks, freed at exit once the exchange completed
        site = st.enter("transpose_shard")
    send, recv = {}, {}
    try:
        for s, (c0, c1) in enumerate(layout.parts):
            if c1 <= c0:
                continue
            blk = z_loc[..., :, c0:c1, :]                       # [*, n_loc, n_s, C]  (my rows, rank-s columns)
            if s == layout.rank:
                out[..., :, layout.r0:layout.r1, :].copy_(blk.transpose(-2, -3))
            else:
                if layout.n_loc > 0:
                    sb = st.buffer(site, f"send{s}", lead + [c1 - c0, layout.n_loc, C], z_loc.dtype, z_loc.device)
                    sb.copy_(blk.transpose(-2, -3))                 # [*, n_s, n_loc, C] = already in the receiver's orientation
                    send[s] = sb
                    recv[s] = st.buffer(site, f"recv{s}", lead + [layout.n_loc, c1 - c0, C], z_loc.dtype, z_loc.device)
        if layout.P > 1:
            cm.p2p(send, recv, "transpose_shard")
        for s, t in recv.items():
            c0, c1 = layout.parts[s]
            out[..., :, c0:c1, :].copy_(t)
    finally:
        if st is not None:
            st.exit(site)
    del send, recv
    return out


def transpose_shard_streamed(z_loc, layout: Layout, peers_in_flight: int = 1, out=None):
    """Same semantics as :func:`transpose_shard`; the P-1 off-diagonal blocks are exchanged ``peers_in_flight`` peers at a time on a shifted
    schedule: at step d (1 <= d < P) this rank sends its block for rank ``(rank+d)%P`` and receives the block it needs from rank ``(rank-d)%P``
    (every rank runs the same step, so transfers pair up). Transient per rank = out (1/P z-eq) + peers_in_flight * (one send block + one recv
    block), versus out + 2(P-1)/P^2 z-eq for the all-at-once form."""
    _check_shard(z_loc, layout, "transpose_shard_streamed")
    P = layout.P
    lead, C = _lead_C(z_loc)
    out = _transpose_out(z_loc, layout, out)
    out[..., :, layout.r0:layout.r1, :].copy_(z_loc[..., :, layout.r0:layout.r1, :].transpose(-2, -3))      # diagonal block: local transpose
    if P == 1:
        return out
    cm = comm()
    k = max(1, int(peers_in_flight))
    _record(f"streamed:{k}")
    st = cm.staging                                           # k send + k recv blocks reused every step, freed at exit
    site = st.enter("transpose_shard_streamed")
    steps = list(range(1, P))
    cap = [_numel(lead) * layout.n_loc * layout.Rmax * C]       # every slot sized once to the largest peer block (ragged layouts re-view it)
    try:
        for i in range(min(k, len(steps))):
            st.buffer(site, f"send#{i}", cap, z_loc.dtype, z_loc.device); st.buffer(site, f"recv#{i}", cap, z_loc.dtype, z_loc.device)
        for g in range(0, len(steps), k):
            send, recv = {}, {}
            for i, d in enumerate(steps[g:g + k]):
                dst = (layout.rank + d) % P                       # I send z[my rows, dst-cols]^T to dst
                src = (layout.rank - d) % P                       # I receive z[src rows, my cols]^T from src -> out[:, src-cols]
                d0, d1 = layout.parts[dst]
                s0, s1 = layout.parts[src]                        # (P == 2 or d == P/2: dst == src -> one send + one recv entry each)
                if layout.n_loc > 0 and d1 > d0:
                    sb = st.buffer(site, f"send#{i}", lead + [d1 - d0, layout.n_loc, C], z_loc.dtype, z_loc.device)
                    sb.copy_(z_loc[..., :, d0:d1, :].transpose(-2, -3))                  # [*, n_dst, n_loc, C] (receiver's orientation)
                    send[dst] = sb
                if layout.n_loc > 0 and s1 > s0:
                    recv[src] = st.buffer(site, f"recv#{i}", lead + [layout.n_loc, s1 - s0, C], z_loc.dtype, z_loc.device)
            cm.p2p(send, recv, "transpose_shard_streamed")
            for src, t in recv.items():
                s0, s1 = layout.parts[src]
                out[..., :, s0:s1, :].copy_(t)
            del send, recv
    finally:
        st.exit(site)
    return out


def transpose_shard_inplace_(z_loc, layout: Layout, peers_in_flight: int = 1):
    """IN-PLACE distributed transpose: ``z_loc`` (contiguous ``[*, n_loc, N, C]``, rows r0:r1 of z) is overwritten with rows r0:r1 of ``z^T`` and
    returned (same tensor object). Works because block ``(r, s) = z[r-rows, s-cols]`` (``[n_r, n_s]``) and the block that must replace it,
    ``(z[s-rows, r-cols])^T``, have the same shape. Ranks swap blocks pairwise on a symmetric schedule (round d: ``peer = rank XOR d``,
    ``d = 1 .. next_pow2(P)-1``; rounds with ``peer >= P`` are idle but every rank still enters the exchange), so a slot is only overwritten
    in the same round in which its own content was staged and sent. The diagonal block is transposed through an ``n_loc x n_loc`` temporary.
    Transient per rank = peers_in_flight * (send + recv block) + the diagonal temp (~(2*peers+1)/P^2 z-eq) on top of z_loc itself — versus a
    whole second shard (1/P) for the out-of-place variants. Values identical to :func:`transpose_shard`."""
    _check_shard(z_loc, layout, "transpose_shard_inplace_")
    if not z_loc.is_contiguous():
        raise RowpairRefused("transpose_shard_inplace_ needs a contiguous shard")
    P = layout.P
    lead, C = _lead_C(z_loc)
    diag = z_loc[..., :, layout.r0:layout.r1, :]
    # The diagonal block through a REAL ``n_loc x n_loc`` temporary, whatever n_loc is: ``clone(contiguous_format)`` is the statement
    # ``.contiguous()`` executes on the transposed view when n_loc >= 2 (same kernel, same bytes); on a ONE-ROW shard (n_loc == 1 — e.g. the
    # grid of N=257 over P=2 leaves rows 256:257 on rank 1) the transposed view of the 1x1 block is already contiguous, ``.contiguous()`` would
    # hand back the block itself, and ``copy_`` refuses a source aliasing its destination with other strides.
    tmp = diag.transpose(-2, -3).clone(memory_format=torch.contiguous_format)
    diag.copy_(tmp)
    del tmp, diag
    if P == 1:
        return z_loc
    cm = comm()
    P2 = 1
    while P2 < P:
        P2 *= 2
    steps = list(range(1, P2))
    k = max(1, int(peers_in_flight))
    _record(f"inplace:{k}")
    st = cm.staging                                           # k send + k recv blocks reused every round, freed at exit
    site = st.enter("transpose_shard_inplace_")
    cap = [_numel(lead) * layout.n_loc * layout.Rmax * C]       # every slot sized once to the largest peer block (ragged layouts re-view it)
    try:
        for i in range(min(k, len(steps))):
            st.buffer(site, f"send#{i}", cap, z_loc.dtype, z_loc.device); st.buffer(site, f"recv#{i}", cap, z_loc.dtype, z_loc.device)
        for g in range(0, len(steps), k):
            send, recv = {}, {}
            for i, d in enumerate(steps[g:g + k]):
                peer = layout.rank ^ d
                if peer >= P:
                    continue
                p0, p1 = layout.parts[peer]
                if layout.n_loc > 0 and p1 > p0:
                    sb = st.buffer(site, f"send#{i}", lead + [p1 - p0, layout.n_loc, C], z_loc.dtype, z_loc.device)
                    sb.copy_(z_loc[..., :, p0:p1, :].transpose(-2, -3))                 # my block (r,peer) in the receiver's orientation
                    send[peer] = sb
                    recv[peer] = st.buffer(site, f"recv#{i}", lead + [layout.n_loc, p1 - p0, C], z_loc.dtype, z_loc.device)   # (z[peer rows, my cols])^T
            cm.p2p(send, recv, "transpose_shard_inplace_")   # every rank enters every round (idle rounds pass empty dicts)
            for peer, t in recv.items():
                p0, p1 = layout.parts[peer]
                z_loc[..., :, p0:p1, :].copy_(t)
            del send, recv
    finally:
        st.exit(site)
    return z_loc


def transpose_band(z_loc, layout: Layout, windows: Sequence[Tuple[int, int]]):
    """Banded distributed transpose. ``windows``: list over ranks q of ``(w0, w1)`` = the GLOBAL column window of z that rank q wants as rows of
    ``z^T`` (``w1 <= w0``: nothing). Input: my rows of z (``[*, n_loc, N, C]``). Output: my band of ``z^T`` rows ``[*, w1-w0, N, C]`` with
    ``out[.., i, j, :] = z[.., j, w0_me + i, :]`` (None if my window is empty). Rank p sends ``z_loc[..., :, w0_q:w1_q, :]`` transposed
    (``[*, w_q, n_p, C]``, receiver orientation) to q. Transient: ``sum_q w_q x n_loc x C`` sent + ``w_me x N x C`` received."""
    _check_shard(z_loc, layout, "transpose_band")
    lead, C = _lead_C(z_loc)
    if len(windows) != layout.P:
        raise RowpairRefused(f"transpose_band: {len(windows)} windows for P={layout.P}")
    if layout.P > 1:
        comm().check_replicated_arg([(int(a), int(b)) for a, b in windows], "transpose_band: windows")   # ROWPAIR_P2P_CHECK: identical on every rank
    w0m, w1m = windows[layout.rank]
    wm = max(0, int(w1m) - int(w0m))
    out = torch.empty(lead + [wm, layout.N, C], dtype=z_loc.dtype, device=z_loc.device) if wm > 0 else None
    send, recv = {}, {}
    st, site = None, None
    if layout.P > 1:
        st = comm().staging                                 # one send + one recv block per peer window, freed at exit
        site = st.enter("transpose_band")
    try:
        for q, (w0, w1) in enumerate(windows):
            w0, w1 = int(w0), int(w1)
            w = max(0, w1 - w0)
            if q == layout.rank:
                if w > 0 and layout.n_loc > 0:
                    out[..., :, layout.r0:layout.r1, :].copy_(z_loc[..., :, w0:w1, :].transpose(-2, -3))
                continue
            if w > 0:
                if not (0 <= w0 and w1 <= layout.N):
                    raise RowpairRefused(f"transpose_band: window {(w0, w1)} of rank {q} outside [0, {layout.N}]")
                if layout.n_loc > 0:
                    sb = st.buffer(site, f"send{q}", lead + [w, layout.n_loc, C], z_loc.dtype, z_loc.device)
                    sb.copy_(z_loc[..., :, w0:w1, :].transpose(-2, -3))                  # [*, w_q, n_loc, C]
                    send[q] = sb
            q0, q1 = layout.parts[q]
            if wm > 0 and q1 > q0:
                recv[q] = st.buffer(site, f"recv{q}", lead + [wm, q1 - q0, C], z_loc.dtype, z_loc.device)
        if layout.P > 1:
            comm().p2p(send, recv, "transpose_band")
        for q, t in recv.items():
            q0, q1 = layout.parts[q]
            out[..., :, q0:q1, :].copy_(t)
    finally:
        if st is not None:
            st.exit(site)
    del send, recv
    return out


def ring_pass(own, layout: Layout, shapes: Sequence[Optional[Sequence[int]]], dtype=None, device=None) -> Iterator[Tuple[int, object]]:
    """One pass of a sub-block ring: yield ``(src, block)`` for ``src = rank, rank-1, ..., rank-P+1 (mod P)``, skipping sources whose
    ``shapes[src]`` is None. ``own`` = my contiguous block (None iff ``shapes[rank]`` is None); ``shapes[src]`` = full shape of src's block
    (the identical list on all ranks). Tick ``step``: rank r holds the block of ``src = (r-step) % P``; the exchange (held -> r+1, next <- r-1) is
    STARTED before the block is yielded and waited after the consumer returns (overlap where the comm's ``p2p_start`` is asynchronous; with a
    synchronous one the transfer simply precedes the compute). Every rank issues exactly P-1 exchanges per pass; the own copy + two wire
    buffers (held + arriving, sized to the largest block, from the comm's :class:`opt_core.mem.rowpair.dist.Staging` scope of this pass) are
    resident, no step allocates, and they are freed at exhaustion once every transfer completed. A yielded block is valid until ``next()``: the
    consumer copies out what it keeps and must not modify yielded blocks. Unlike :func:`opt_core.mem.rowpair.dist.ring_blocks` (whole
    blocks padded to ``Rmax``) blocks here have per-source shapes and may be absent."""
    P, r = layout.P, layout.rank
    if len(shapes) != P or (own is None) != (shapes[r] is None):
        raise RowpairRefused(f"ring_pass: shapes has {len(shapes)} entries for P={P}, own is {'None' if own is None else 'given'} while shapes[rank] is "
                             f"{'None' if shapes[r] is None else 'given'}")
    if own is not None:
        require_schedule_dtype(own, dtype, device, "ring_pass: own block")             # the named schedule dtype/device, when given, is what peers allocate
        dtype, device = own.dtype, own.device
        if tuple(own.shape) != tuple(shapes[r]) or not own.is_contiguous():
            raise RowpairRefused(f"ring_pass: own block {tuple(own.shape)} contiguous={own.is_contiguous()} vs shapes[rank]={tuple(shapes[r])}")
    if P == 1:
        if own is not None:
            yield r, own
        return
    if dtype is None or device is None:
        raise RowpairRefused("ring_pass: dtype and device are required on a rank whose own block is absent")
    cm = comm()
    cm.check_replicated_arg([None if s is None else [int(x) for x in s] for s in shapes], "ring_pass: shapes")   # ROWPAIR_P2P_CHECK
    st = cm.staging                                                              # the pass's wire buffers: the own copy + two flat receive buffers
    site = st.enter("ring_pass")                                                 # sized to the largest block, alternating per step; freed at
    nmax = max([_numel(s) for s in shapes if s is not None] + [1])               # exhaustion once every transfer completed
    nxt, prv = (r + 1) % P, (r - 1) % P
    cur, cur_src = None, r
    try:
        if own is not None:                                                      # the caller's tensor never goes on the wire
            cur = st.buffer(site, "own", [nmax], dtype, device)[:_numel(shapes[r])].view([int(x) for x in shapes[r]])
            cur.copy_(own)
        for step in range(P):
            nsrc = (r - step - 1) % P
            h, incoming = None, None
            if step < P - 1:
                if shapes[nsrc] is not None:
                    flat = st.buffer(site, "AB"[step % 2], [nmax], dtype, device)
                    incoming = flat[:_numel(shapes[nsrc])].view([int(x) for x in shapes[nsrc]])
                h = cm.p2p_start({nxt: cur} if cur is not None else {}, {prv: incoming} if incoming is not None else {}, "ring_pass")
            if cur is not None:
                yield cur_src, cur
            if step < P - 1:
                h.wait()
            cur, cur_src = incoming, nsrc
    finally:
        st.exit(site)
