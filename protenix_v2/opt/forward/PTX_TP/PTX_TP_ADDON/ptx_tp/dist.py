"""
ptx_tp.dist -- the tensor-parallel (TP) communication contract: the row
layout and the collectives (tensors in / tensors out).

Conventions
-----------
* A "pair" tensor X[N, N, C] is ROW-SHARDED: rank q holds X[q0:q1, :, :]
  (contiguous), where (q0, q1) = Layout.bounds[q].
* The row grid is P-INVARIANT: global rows are grouped in blocks of B (=128)
  rows; rank q owns blocks [q*bpr, (q+1)*bpr) with bpr = ceil(ceil(N/B)/P).
  Every rank boundary is a multiple of B, so a chunk grid whose chunk size
  divides B never straddles a rank boundary.
* Everything here is PURE DATA MOVEMENT (bit-preserving); no arithmetic is
  performed on payload tensors (except the integer checksums).
* P == 1 (or no process group) is always supported: identity semantics.

Environment knobs
-----------------
PTX_TP_A2A_CHUNKS   number of channel groups used to stage all-to-all
                    transposes (default 4; bounds the send+recv transient to
                    2/chunks of the moved tensor).
"""
from __future__ import annotations

import os
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

__all__ = [
    "init_from_env", "Layout", "is_dist", "barrier",
    "all_gather_rows", "gather_rows_to_rank0",
    "transpose_shards", "alltoall_window", "ring_blocks",
    "allreduce_checksum", "broadcast_obj",
    "zrows", "zwrite", "zadd", "zmeta",
]


# --------------------------------------------------------------------------
# process-group setup
# --------------------------------------------------------------------------
def init_from_env(backend: Optional[str] = None, timeout_s: int = 1800):
    """Initialise torch.distributed from the torchrun environment.

    Returns (P, rank, device).  NCCL on CUDA machines (device=cuda:LOCAL_RANK,
    made current), gloo otherwise (CPU tests; also forced by PTX_TP_FORCE_CPU=1).
    With WORLD_SIZE unset/1 no process group is created (P=1 semantics).
    """
    rank = int(os.environ.get("RANK", "0"))
    P = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", str(rank)))
    use_cuda = torch.cuda.is_available() and os.environ.get("PTX_TP_FORCE_CPU", "0") != "1"
    if backend is None:
        backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if P > 1 and not dist.is_initialized():
        import datetime
        kw = dict(backend=backend, timeout=datetime.timedelta(seconds=timeout_s))
        if backend == "nccl":
            try:
                dist.init_process_group(device_id=device, **kw)
            except TypeError:  # older torch without device_id
                dist.init_process_group(**kw)
        else:
            dist.init_process_group(**kw)
    return P, rank, device


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def barrier():
    if is_dist():
        dist.barrier()


# --------------------------------------------------------------------------
# Layout: the P-invariant global row grid
# --------------------------------------------------------------------------
class Layout:
    """P-invariant row grid.

    nb = ceil(N/B) blocks of B rows; bpr = ceil(nb/P) blocks per rank; rank q
    owns global rows [q0,q1) = [min(N, q*bpr*B), min(N, (q+1)*bpr*B)).

    Attributes: N, P, rank, B, nb, bpr, r0, r1, R (=r1-r0), Rmax (=bpr*B),
    bounds (list of (q0,q1)), replicated (True iff N < P*B: then every tp_*
    function must run the stock single-device statement on every rank and
    all tensors stay whole; bounds become (0,N) for every rank).
    Trailing ranks may own ZERO rows when N is small relative to P*B; all
    primitives handle R == 0.
    """

    def __init__(self, N: int, P: int, rank: int, B: int = 128):
        assert N >= 1 and P >= 1 and 0 <= rank < P and B >= 1
        self.N, self.P, self.rank, self.B = int(N), int(P), int(rank), int(B)
        self.nb = -(-self.N // self.B)
        self.bpr = -(-self.nb // self.P)
        self.Rmax = self.bpr * self.B
        self.bounds: List[Tuple[int, int]] = [
            (min(self.N, q * self.Rmax), min(self.N, (q + 1) * self.Rmax)) for q in range(self.P)
        ]
        self.r0, self.r1 = self.bounds[self.rank]
        self.R = self.r1 - self.r0
        self.replicated = self.N < self.P * self.B
        if self.replicated:
            self.bounds = [(0, self.N) for _ in range(self.P)]
            self.r0, self.r1, self.R, self.Rmax = 0, self.N, self.N, self.N

    def rows(self, q: Optional[int] = None) -> slice:
        q0, q1 = self.bounds[self.rank if q is None else q]
        return slice(q0, q1)

    def nrows(self, q: Optional[int] = None) -> int:
        q0, q1 = self.bounds[self.rank if q is None else q]
        return q1 - q0

    def chunks(self, chunk: int, q: Optional[int] = None) -> Iterator[Tuple[int, int]]:
        """GLOBAL chunk grid range(0, N, chunk) clipped to rank q's rows:
        yields (c0, c1) global with q0 <= c0 < c1 <= q1.  If `chunk` divides
        B these are exactly stock's chunks that fall inside the rank's rows."""
        q0, q1 = self.bounds[self.rank if q is None else q]
        c = (q0 // chunk) * chunk
        while c < q1:
            c0, c1 = max(c, q0), min(c + chunk, q1)
            if c1 > c0:
                yield (c0, c1)
            c += chunk

    def local_blocks(self, rows_per_block: int, q: Optional[int] = None) -> Iterator[Tuple[int, int]]:
        """LOCAL row sub-blocks [i0, i1) (local coordinates) of rank q's shard."""
        R = self.nrows(q)
        i = 0
        rows_per_block = max(1, int(rows_per_block))
        while i < R:
            yield (i, min(R, i + rows_per_block))
            i += rows_per_block

    def owner(self, i: int) -> int:
        return self.rank if self.replicated else min(self.P - 1, i // self.Rmax)

    def __repr__(self) -> str:
        return (f"Layout(N={self.N}, P={self.P}, rank={self.rank}, B={self.B}, r0={self.r0}, r1={self.r1}, "
                f"R={self.R}, Rmax={self.Rmax}, replicated={self.replicated})")



# --------------------------------------------------------------------------
# storage-agnostic pair-shard accessors (tensor fallbacks; when a ptx_tp.zstore
# module is importable its storage-aware versions are bound instead, below)
# --------------------------------------------------------------------------
def _t_zrows(z, i0: int, i1: int) -> torch.Tensor:
    """rows i0:i1 (LOCAL indices) of the shard as bf16/stock-dtype [i1-i0, N, C]; a VIEW for tensors."""
    return z[i0:i1]


def _t_zwrite(z, i0: int, i1: int, x: torch.Tensor) -> None:
    """copy rows back (no-op when x is the view itself)."""
    dst = z[i0:i1]
    if x.data_ptr() == dst.data_ptr() and x.shape == dst.shape and x.stride() == dst.stride():
        return
    dst.copy_(x)


def _t_zadd(z, i0: int, i1: int, delta: torch.Tensor) -> None:
    """in-place z[i0:i1] += delta."""
    z[i0:i1] += delta


def zmeta(z):
    """(R, N, C, dtype, device) of a shard (tensor or ZStore duck-type)."""
    if hasattr(z, "zmeta"):
        return z.zmeta()
    return z.shape[0], z.shape[1], z.shape[2], z.dtype, z.device


try:  # real implementations (fp8 storage aware) when the zstore module is installed
    from ptx_tp.zstore import zrows, zwrite, zadd  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    def zrows(z, i0, i1):
        return z.zrows(i0, i1) if hasattr(z, "zrows") else _t_zrows(z, i0, i1)

    def zwrite(z, i0, i1, x):
        return z.zwrite(i0, i1, x) if hasattr(z, "zwrite") else _t_zwrite(z, i0, i1, x)

    def zadd(z, i0, i1, delta):
        return z.zadd(i0, i1, delta) if hasattr(z, "zadd") else _t_zadd(z, i0, i1, delta)

# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------
def _pad_rows(x: torch.Tensor, rows: int, dim: int = 0) -> torch.Tensor:
    """Contiguous tensor with `rows` entries along `dim`, leading entries = x
    (bit copy), padding zeroed.  No copy if x already has that shape and is
    contiguous."""
    n = x.shape[dim]
    if n == rows and x.is_contiguous():
        return x
    shape = list(x.shape)
    shape[dim] = rows
    buf = x.new_empty(shape)
    if n > 0:
        buf.narrow(dim, 0, n).copy_(x)
    if rows > n:
        buf.narrow(dim, n, rows - n).zero_()
    return buf


def _p2p_exchange(sends: Sequence[Tuple[torch.Tensor, int]], recvs: Sequence[Tuple[torch.Tensor, int]]):
    ops = [dist.P2POp(dist.isend, t, dst) for t, dst in sends]
    ops += [dist.P2POp(dist.irecv, t, src) for t, src in recvs]
    if not ops:
        return
    for r in dist.batch_isend_irecv(ops):
        r.wait()


def _all_to_all_single(out: torch.Tensor, inp: torch.Tensor):
    """all_to_all_single with P equal blocks along dim 0; P2P fallback."""
    P = dist.get_world_size()
    try:
        dist.all_to_all_single(out, inp)
        return
    except (RuntimeError, NotImplementedError, ValueError):  # backend dependent
        pass
    rank = dist.get_rank()
    out[rank].copy_(inp[rank])
    sends = [(inp[d].contiguous(), d) for d in range(P) if d != rank]
    recvs = [(out[s], s) for s in range(P) if s != rank]
    _p2p_exchange(sends, recvs)


# --------------------------------------------------------------------------
# gathers
# --------------------------------------------------------------------------
def all_gather_rows(x_shard: torch.Tensor, layout: Layout) -> torch.Tensor:
    """x_shard[R_rank, ...] -> full [N, ...] on every rank (rank order).
    Pad to Rmax, all_gather_into_tensor, return the leading-N view (global
    row i lands at padded position i, so no unpad copy is needed)."""
    if layout.replicated or not is_dist() or layout.P == 1:
        return x_shard
    assert x_shard.shape[0] == layout.R, (tuple(x_shard.shape), repr(layout))
    rest = list(x_shard.shape[1:])
    buf = _pad_rows(x_shard.contiguous(), layout.Rmax, 0)
    out = x_shard.new_empty([layout.P * layout.Rmax] + rest)
    try:
        dist.all_gather_into_tensor(out, buf)
    except (RuntimeError, NotImplementedError, ValueError):  # gloo fallback
        dist.all_gather([out[q * layout.Rmax:(q + 1) * layout.Rmax] for q in range(layout.P)], buf)
    return out[: layout.N]


def gather_rows_to_rank0(x_shard: torch.Tensor, layout: Layout) -> Optional[torch.Tensor]:
    """x_shard[R_rank, ...] -> full [N, ...] on rank 0 (None elsewhere);
    point-to-point, only rank 0 allocates the full tensor."""
    if layout.replicated or not is_dist() or layout.P == 1:
        return x_shard if layout.rank == 0 else None
    assert x_shard.shape[0] == layout.R
    rest = list(x_shard.shape[1:])
    if layout.rank == 0:
        out = x_shard.new_empty([layout.P * layout.Rmax] + rest)
        if layout.R > 0:
            out[: layout.R].copy_(x_shard)
        _p2p_exchange([], [(out[q * layout.Rmax:(q + 1) * layout.Rmax], q) for q in range(1, layout.P)])
        return out[: layout.N]
    _p2p_exchange([(_pad_rows(x_shard.contiguous(), layout.Rmax, 0), 0)], [])
    return None


# --------------------------------------------------------------------------
# all-to-all window transpose (behind transpose_shards and the column-layout
# contractions)
# --------------------------------------------------------------------------
def alltoall_window(
    provider: Callable[[int, int], torch.Tensor],
    layout: Layout,
    off: int = 0,
    width: Optional[int] = None,
    *,
    C: int,
    dtype: torch.dtype,
    device,
    out_layout: str = "rows",
    chunks: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generalised distributed transpose of a row-sharded X[N, N, C].

    provider(c0, c1) must return THIS rank's block X[r0:r1, c0:c1, :] as a
    tensor [R, c1-c0, C] (a view, or computed just in time); it is called once
    per destination rank with that destination's column window.
    For destination d the window is global columns [d0+off, min(d1, d0+off+width))
    ((d0,d1) = bounds[d]); width=None means d's whole row range (off must be 0).

    Every rank receives the data of ITS window rows i in [r0+off, min(r1, r0+off+width))
    (W_v rows) over all j in [0, N):
        out_layout "rows"   : out[w, j, c] = X[j, r0+off+w, c]  -> [W_v, N, C]
        out_layout "gemm_a" : out[c, w, j] = X[j, r0+off+w, c]  -> [C, W_v, N]
        out_layout "gemm_b" : out[c, j, w] = X[j, r0+off+w, c]  -> [C, N, W_v]
    Bit-preserving.  Staged in `chunks` channel groups (PTX_TP_A2A_CHUNKS,
    default 4): transient = provider blocks + out + 2/chunks*(W*N*C) send/recv.
    """
    P, rank, N = layout.P, layout.rank, layout.N
    if width is None:
        assert off == 0
        width = layout.Rmax
    W = int(width)

    def win(q):
        q0, q1 = layout.bounds[q]
        return min(q1, q0 + off), min(q1, q0 + off + W)

    my_a, my_b = win(rank)
    Wv = my_b - my_a
    if out is None:
        shape = {"rows": (Wv, N, C), "gemm_a": (C, Wv, N), "gemm_b": (C, N, Wv)}[out_layout]
        out = torch.empty(shape, dtype=dtype, device=device)

    def place(src_q: int, blk: torch.Tensor, c0: int, c1: int):
        # blk: [R_src (j range = bounds[src_q]), Wv, c1-c0]
        j0, j1 = layout.bounds[src_q]
        if j1 <= j0 or Wv == 0:
            return
        if out_layout == "rows":
            out[:, j0:j1, c0:c1].copy_(blk.transpose(0, 1))
        elif out_layout == "gemm_a":
            out[c0:c1, :, j0:j1].copy_(blk.permute(2, 1, 0))
        else:
            out[c0:c1, j0:j1, :].copy_(blk.permute(2, 0, 1))

    if layout.replicated or not is_dist() or P == 1:
        if Wv > 0:
            place(rank, provider(my_a, my_b), 0, C)
        return out

    blocks = []
    for d in range(P):
        a, b = win(d)
        blocks.append(provider(a, b) if b > a else None)
    nch = int(chunks if chunks is not None else int(os.environ.get("PTX_TP_A2A_CHUNKS", "4")))
    nch = max(1, min(nch, C))
    cs = -(-C // nch)
    Rmax = layout.Rmax
    for c0 in range(0, C, cs):
        c1 = min(C, c0 + cs)
        send = torch.zeros((P, Rmax, W, c1 - c0), dtype=dtype, device=device)
        for d in range(P):
            blk = blocks[d]
            if blk is not None and blk.shape[0] > 0 and blk.shape[1] > 0:
                send[d, : blk.shape[0], : blk.shape[1], :].copy_(blk[..., c0:c1])
        recv = torch.empty_like(send)
        _all_to_all_single(recv, send)
        del send
        for s in range(P):
            Rs = layout.nrows(s)
            if Rs > 0:
                place(s, recv[s, :Rs, :Wv, :], c0, c1)
        del recv
    del blocks
    return out


def transpose_shards(z_shard, layout: Layout, chunks: Optional[int] = None, *,
                     block_rows: Optional[int] = None, out=None):
    """z_shard[R, N, C] (rows r0:r1 of Z) -> zT_shard[R, N, C] with
    zT_shard[i-r0, j, :] = Z[j, i, :]  (all_to_all over padded
    [P, Rmax, W, C/chunks] blocks + local block transpose; bit-preserving).

    z_shard / out may be plain tensors or ZStore objects (accessed only via
    zrows / zwrite in row blocks).  block_rows (env PTX_TP_TRANSPOSE_BLOCK_ROWS,
    default 0 = whole shard in one exchange) streams the transpose in output
    row windows of that many rows (use multiples of 128) so that a full
    stock-dtype copy of the shard is never materialised at once:
    transient ~= 3 * block_rows * N * C elements (+ 2/chunks of one window)."""
    R, N, C, dtype, device = zmeta(z_shard)
    if layout.replicated or not is_dist() or layout.P == 1:
        zt = zrows(z_shard, 0, R).transpose(0, 1).contiguous()
        if out is None:
            return zt
        zwrite(out, 0, zt.shape[0], zt)
        return out
    assert R == layout.R and N == layout.N, ((R, N, C), repr(layout))
    if block_rows is None:
        block_rows = int(os.environ.get("PTX_TP_TRANSPOSE_BLOCK_ROWS", "0"))
    rb = int(os.environ.get("PTX_TP_ZROWS_BLOCK", "128"))  # read granularity for zrows

    def provider(c0, c1):
        # my rows, columns c0:c1, read through zrows in row blocks
        if c1 <= c0:
            return torch.empty((R, 0, C), dtype=dtype, device=device)
        if not hasattr(z_shard, "zrows") and torch.is_tensor(z_shard):
            return z_shard[:, c0:c1, :]
        blk = torch.empty((R, c1 - c0, C), dtype=dtype, device=device)
        for i0 in range(0, R, rb):
            i1 = min(R, i0 + rb)
            blk[i0:i1].copy_(zrows(z_shard, i0, i1)[:, c0:c1, :])
        return blk

    if not block_rows or block_rows >= layout.Rmax:
        res = alltoall_window(provider, layout, 0, None, C=C, dtype=dtype, device=device,
                              out_layout="rows", chunks=chunks)
        if out is None:
            return res
        zwrite(out, 0, res.shape[0], res)
        return out
    if out is None:
        out = torch.empty((R, N, C), dtype=dtype, device=device)
    for w0 in range(0, layout.Rmax, block_rows):      # same schedule on every rank
        w1 = min(layout.Rmax, w0 + block_rows)
        res = alltoall_window(provider, layout, w0, w1 - w0, C=C, dtype=dtype, device=device,
                              out_layout="rows", chunks=chunks)
        if res.shape[0] > 0:
            zwrite(out, w0, w0 + res.shape[0], res)
        del res
    return out


# --------------------------------------------------------------------------
# ring streaming
# --------------------------------------------------------------------------
def ring_blocks(x_shard: torch.Tensor, layout: Layout, *, rows: Optional[Tuple[int, int]] = None,
                dim: int = 0) -> Iterator[Tuple[int, torch.Tensor]]:
    """Stream every rank's block around the ring; yields (q, x_q) for EVERY
    rank q exactly once: t=0 the own block, then q = (rank - t) % P.

    rows=None    : x_shard is the rank's whole shard along `dim` (R_rank
                   entries); blocks travel padded to Rmax; x_q has R_q entries.
    rows=(j0,RB) : x_shard is the rank's LOCAL sub-slab [j0:j0+RB] (clipped to
                   R) along `dim`; blocks travel padded to RB; x_q has
                   clip(R_q - j0, 0, RB) entries (may be 0 -> still yielded).
    isend/irecv double-buffered on padded buffers.  The generator MUST be run to
    exhaustion on every rank (the prefetch for step t+1 is posted before x_t is
    yielded).  The consumer must be done READING x_q before calling next() (on CUDA ordinary stream ordering is
    enough: the later irecv into that buffer is ordered after kernels already
    enqueued on the current stream)."""
    P, rank = layout.P, layout.rank
    if rows is None:
        unit = layout.Rmax
        valid = lambda q: layout.nrows(q)  # noqa: E731
    else:
        j0, RB = int(rows[0]), int(rows[1])
        unit = RB
        valid = lambda q: max(0, min(layout.nrows(q) - j0, RB))  # noqa: E731
    if layout.replicated or not is_dist() or P == 1:
        yield rank, x_shard
        return
    assert x_shard.shape[dim] == valid(rank), (tuple(x_shard.shape), dim, valid(rank), rows)
    own = _pad_rows(x_shard, unit, dim)          # == x_shard itself when already padded & contiguous
    cur = own
    bufs = [torch.empty_like(own), None]         # recv targets alternate: step t receives into bufs[t % 2]
    nxt_rank, prv_rank = (rank + 1) % P, (rank - 1) % P
    for t in range(P):
        q = (rank - t) % P
        reqs = None
        if t < P - 1:
            if bufs[t % 2] is None:              # t == 1: reuse our private padded copy if we made one
                bufs[1] = own if own is not x_shard else torch.empty_like(own)
            tgt = bufs[t % 2]
            assert tgt.data_ptr() != cur.data_ptr()
            reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, cur, nxt_rank),
                                           dist.P2POp(dist.irecv, tgt, prv_rank)])
        yield q, cur.narrow(dim, 0, valid(q))
        if reqs is not None:
            for r in reqs:
                r.wait()
            cur = bufs[t % 2]


# --------------------------------------------------------------------------
# replicated-tensor checks / object broadcast
# --------------------------------------------------------------------------
def _int_view(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous().reshape(-1)
    nb = t.element_size() * t.numel()
    u8 = t.view(torch.uint8)
    if nb % 8 == 0:
        return u8.view(torch.int64)
    if nb % 4 == 0:
        return u8.view(torch.int32).to(torch.int64)
    if nb % 2 == 0:
        return u8.view(torch.int16).to(torch.int64)
    return u8.to(torch.int64)


def _xor_fold(v: torch.Tensor) -> int:
    v = v.clone()
    while v.numel() > 1:
        n = v.numel()
        if n % 2 == 1:
            v = torch.cat([v, v.new_zeros(1)])
            n += 1
        v = torch.bitwise_xor(v[: n // 2], v[n // 2:])
    return int(v.item()) if v.numel() else 0


def allreduce_checksum(t: torch.Tensor, name: str = "tensor", raise_on_mismatch: bool = True):
    """Assert a REPLICATED tensor is bitwise identical on all ranks.
    checksum = (numel, wrapping int64 sum of the int view, xor-fold).
    Returns the list of per-rank checksums."""
    v = _int_view(t)
    local = (int(t.numel()), int(v.sum().item()) if v.numel() else 0, _xor_fold(v))
    if not is_dist():
        return [local]
    allv = [None] * dist.get_world_size()
    dist.all_gather_object(allv, local)
    if raise_on_mismatch and not all(x == allv[0] for x in allv):
        raise AssertionError(f"allreduce_checksum mismatch for '{name}': {allv}")
    return allv


def broadcast_obj(obj, src: int = 0):
    """Broadcast a picklable object from src; returns it on every rank."""
    if not is_dist():
        return obj
    box = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(box, src=src)
    return box[0]


# =====================================================================================================================
# Additional helpers. (a) pure data-movement helpers used by the confidence seam -- zclone / zalloc_like / zblocks /
# gather_cat_to_rank0 / transpose_blocks (bit-preserving; transpose_shards above is the whole-shard form); (b) small driver
# conveniences used by the launcher / bcast / trimul code -- world(), allreduce_(), checksum(), zlen(). Nothing above is redefined
# except _int_view / _xor_fold: the later bindings are the ones every function of this module resolves at call time.
# =====================================================================================================================
ZBLOCK = 128


def _world() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _is_nccl() -> bool:
    return dist.is_initialized() and dist.get_backend() == "nccl"


# ----------------------------------------------------------------------------- layout


def world():
    """(P, rank) of the default group, or from the torchrun env without a process group."""
    if is_dist():
        return dist.get_world_size(), dist.get_rank()
    return int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))


REDUCE_OPS = ("sum", "max", "min")


def allreduce_(t: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """In place ``t = op over ranks (t)`` (``sum`` | ``max`` | ``min``) on the default group; returns ``t``. The identity at P == 1.
    Called at points every rank reaches (it is a collective)."""
    if op not in REDUCE_OPS:
        raise ValueError(f"allreduce_: op {op!r} not in {REDUCE_OPS}")
    if not is_dist():
        return t
    dist.all_reduce(t, op={"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}[op])
    return t


def zlen(z) -> int:
    """number of LOCAL rows of a shard (tensor or store-like with .R / __len__)."""
    return int(z.shape[0]) if hasattr(z, "shape") else int(getattr(z, "R", None) or len(z))


def _int_view(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    if t.dtype == torch.bool:
        return t.to(torch.int64)
    nb = t.element_size()
    if nb == 1:
        return t.view(torch.uint8).to(torch.int64)
    if nb == 2:
        return t.view(torch.int16).to(torch.int64)
    if nb == 4:
        return t.view(torch.int32).to(torch.int64)
    return t.view(torch.int64)


def checksum(t: torch.Tensor):
    """(numel, sum of the integer view, xor-fold of the integer view) -- cheap bitwise fingerprint (driver/bcast debug)."""
    iv = _int_view(t).reshape(-1)
    if iv.numel() == 0:
        return 0, 0, 0
    s = int(iv.sum().item())
    x = iv
    while x.numel() > 1:
        n = x.numel()
        if n % 2:
            x = torch.cat([x, x.new_zeros(1)])
            n += 1
        x = torch.bitwise_xor(x[: n // 2], x[n // 2:])
    return int(t.numel()), s, int(x.item())


def zclone(z):
    """Private per-sample copy of a shard (tensor fallback: .clone())."""
    return z.clone()


def zalloc_like(z, dtype=None, rows: Optional[int] = None):
    """New shard container shaped like z (tensor fallback: torch.empty)."""
    shape = list(z.shape)
    if rows is not None:
        shape[0] = rows
    return torch.empty(shape, dtype=dtype or z.dtype, device=z.device)


def zblocks(R: int, step: int = ZBLOCK):
    """LOCAL row blocks (i0, i1) of size `step` (tail shorter)."""
    for i0 in range(0, R, step):
        yield i0, min(R, i0 + step)


# ----------------------------------------------------------------------------- collectives


def gather_cat_to_rank0(piece: torch.Tensor, counts: list, device=None) -> Optional[torch.Tensor]:
    """Variable-length row concat to rank 0: rank q contributes `piece` with counts[q] rows
    (counts known on every rank).  Returns cat over ranks (rank order) on rank 0, None elsewhere."""
    P, rank = _world()
    piece = piece.contiguous()
    assert piece.shape[0] == counts[rank], (piece.shape, counts, rank)
    if P == 1:
        return piece
    if rank == 0:
        total = int(sum(counts))
        full = piece.new_empty((total,) + tuple(piece.shape[1:]))
        o = 0
        for q in range(P):
            n = int(counts[q])
            if n > 0:
                if q == 0:
                    full[o: o + n] = piece
                else:
                    dist.recv(full[o: o + n], src=q)
            o += n
        return full
    if counts[rank] > 0:
        dist.send(piece, dst=0)
    return None


def transpose_blocks(z_shard, layout: Layout, step: int = ZBLOCK):
    """Block-streamed transpose.  Yields (i0, i1, zT_blk) for this rank's LOCAL row blocks
    [i0, i1) (size `step`, ascending; tail shorter) with  zT_blk[i - i0, j, :] = z[j, r0 + i, :]
    for ALL global j.  zT_blk is a transposed view of a [N, i1-i0, C] column block assembled by
    ONE all_to_all_single per round (variable split sizes, no padding); peak transient is
    O(N * step * C) - a full transposed copy of the shard never exists.  Pure data movement
    (bit-preserving).  Collective: every rank runs max_q ceil(R_q / step) rounds."""
    P, R, N, r0 = layout.P, layout.R, layout.N, layout.r0
    if layout.replicated:
        full = zrows(z_shard, 0, N)
        for (i0, i1) in zblocks(N, step):
            yield i0, i1, full[:, i0:i1].transpose(0, 1)
        return
    probe = zrows(z_shard, 0, min(1, R))
    C = tuple(probe.shape[2:])
    Cn = 1
    for c in C:
        Cn *= int(c)
    n_rounds = max(-(-layout.nrows(q) // step) for q in range(P))
    if not _is_nccl():  # gloo / CPU debug fallback: all-gather once, slice column blocks
        mine = zrows(z_shard, 0, R)
        full = all_gather_rows(mine.contiguous(), layout)
        for (i0, i1) in zblocks(R, step):
            yield i0, i1, full[:, r0 + i0: r0 + i1].transpose(0, 1)
        return
    zr = zrows(z_shard, 0, R)                              # [R, N, C] view of my rows
    def _blk(q, t):
        q0, q1 = layout.bounds[q]
        g0 = q0 + t * step
        g1 = min(q1, g0 + step)
        return g0, max(0, g1 - g0)

    for t in range(n_rounds):
        in_pieces, in_sizes, out_sizes = [], [], []
        _, my_b = _blk(layout.rank, t)                     # width of MY t-th block (0 if none)
        for q in range(P):
            g0, b_q = _blk(q, t)
            if b_q > 0 and R > 0:
                in_pieces.append(zr[:, g0:g0 + b_q].reshape(R * b_q, Cn))   # copy: (row j, col) row-major
            in_sizes.append(R * b_q)
            out_sizes.append(layout.nrows(q) * my_b)
        inp = torch.cat(in_pieces, 0) if in_pieces else probe.new_zeros((0, Cn))
        out = probe.new_empty((N * my_b, Cn))
        dist.all_to_all_single(out, inp, output_split_sizes=out_sizes, input_split_sizes=in_sizes)
        del inp, in_pieces
        if my_b > 0:
            i0 = t * step
            colblock = out.view((N, my_b) + C)                # z[:, r0+i0 : r0+i0+my_b, :] (rank-ordered rows)
            yield i0, i0 + my_b, colblock.transpose(0, 1)
        del out


__all__ = list(__all__) + ["ZBLOCK", "world", "allreduce_", "zlen", "checksum", "zclone", "zalloc_like", "zblocks", "gather_cat_to_rank0", "transpose_blocks"]


def _xor_fold(v):     # same fold as above: odd lengths are padded with a zero element (xor-neutral) at every halving; this later binding is the one used
    v = v.reshape(-1)
    while v.numel() > 1:
        n = v.numel()
        if n % 2:
            v = torch.cat([v, v.new_zeros(1)])
            n += 1
        v = torch.bitwise_xor(v[: n // 2], v[n // 2:])
    return int(v.item()) if v.numel() else 0
