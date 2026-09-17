"""L2 — the sharded triangle-multiplication CONTRACTION over a row-sharded pair tensor (tensors in, tensors out; the projections,
gates, LayerNorms and the epilogue of an engine's TriangleMultiplication module are the caller's callables — see the ADAPTER GUIDE).

Two contraction patterns cover both directions:

(a) ROW-SHARD OUTER contraction (``outgoing``: ``x_ij = sum_k a_ik * b_jk``)
        ``a[i,k,c]``, ``b[j,k,c]`` are ROW-LOCAL functions of pair rows (i resp. j); rank(I) holds a rows I and b rows I; b row-blocks
        travel the ring; tile ``T[c, I_sub, J_sub] = A[c, I_sub, :] @ B[c, :, J_sub]`` (the FULL k in ONE batched matmul, batch =
        channels) -> ``epilogue_fn(T, rows, cols)``.
(b) COL-SHARD INNER contraction (``incoming``: ``x_ij = sum_k a_ki * b_kj``)
        ``a[k,i,c]``, ``b[k,j,c]`` are row-local functions of pair rows k (sharded); an all-to-all moves ``a[:, I_sub]`` -> rank(I) as
        ``A[c, I_sub, k]`` and ``b[:, J_slab]`` -> rank(J) as ``B[c, k, J_slab]``; then (a)'s ring/tile loop runs unchanged and tiles land
        directly in ROW layout on rank(I).

P-INVARIANCE RULE (the reduction-order class of this module): a kernel's shape may depend on P only through the row count of row-local
GEMMs (M) and the tile extents; every output element's full k-contraction happens inside ONE matmul on ONE rank; there is no split-K
across ranks and no partial-sum all-reduce anywhere. Sharded == dense bit for bit when the stack's batched GEMM is tile-extent invariant
(CPU fp32 and cuBLAS fp32/TF32/bf16 without split-K heuristics are in the family's tests; each stack is tested by the kit).

INT32 RULE: every elementwise / LayerNorm launch stays below 2**31 elements (fused kernels index with int32): :func:`max_rows_per_launch`
gives the row budget for ``[rows, N, C]`` blocks; providers / epilogues sub-chunk with it. Batched matmuls may exceed 2**31 total elements
(64-bit strides) but tile elements are asserted < 2**31.

GEMM operand layouts (they reproduce the cuBLAS call of the dense torch statement ``torch.matmul(a[c,i,k], b[c,k,j])``)::
    A block : [C, rows_i, N]  contiguous   (gemm_a_layout)
    B block : [C, N, rows_j]  contiguous   (gemm_b_layout)
    T tile  : [C, rows_i, rows_j] = torch.matmul(A, B)

API:
    trimul_outgoing(a_shard, b_shard, layout, epilogue_fn, RA=, RB=)   ``a_shard``/``b_shard``: ``[R, N, C]`` projected operands of THIS rank's
                                                                       rows (channel-last); tiles of x for local rows x global columns go to
                                                                       ``epilogue_fn(T[C, rows, cols], (i0, i1), (c0, c1))``
    trimul_incoming(a_shard, b_shard, layout, epilogue_fn, RA=, RB=)   the same for ``x_ij = sum_k a_ki b_kj`` (operands read from tensors the
                                                                       epilogue does not modify)
    trimul_dense(a, b, outgoing)                                       the dense statement both reproduce (``[N, N, C] -> [N, N, C]``), the tests' reference
    rowshard_outer_contract / colshard_inner_contract                  the general forms (providers instead of tensors: operands computed just
                                                                       in time per pass / slab — the streamed sub-chunked form, ``RA``/``RB`` rows;
                                                                       ``col_bounds`` / ``mm_rows`` / ``defer_pending_rows`` / ``a_window_rows`` select
                                                                       the STREAMED schedule below)
    TriMulFns(proj, out, gate, C_h) + trimul_update_(fns, z_shard, mask_shard, layout, outgoing=)
                                                                       the whole triangle-multiplication UPDATE of a row shard IN PLACE
                                                                       (``z += g * out(sum_k a b)``) at the streamed schedule: resident = z shard + the
                                                                       ``a`` operand of my rows in GEMM layout (``[C_h, R, N]``); ``b`` sub-blocks of
                                                                       ``RB`` rows are PRODUCED JUST IN TIME from ORIGINAL rows and travel the ring
                                                                       (outgoing) / are assembled by all-to-all windows of PROJECTED columns (incoming);
                                                                       one pass; the epilogue's tiles for local rows whose own b sub-block is not yet
                                                                       produced are DEFERRED (outgoing) so every source row is read before it is written
    slab_grid(layout, RB, col_bounds) / default_sub(N) / mm_row_pieces  the b sub-block grid (``col_bounds`` = extra GLOBAL boundaries no sub-block
                                                                       straddles, e.g. an engine's in-place column-chunk grid), the ``RB`` table
                                                                       (``ROWPAIR_TRIMUL_SUB``, 512 rows up to 16384 tokens else 128), the matmul M pieces
                                                                       (``ROWPAIR_TRIMUL_ROWS`` = auto | 0 | rows: split a tile's local rows only when the
                                                                       A operand piece would reach 2**31 elements)
    deferred_peak_bytes(layout, grid, C, elt)                          the exact peak of the outgoing deferral buffer for a grid (bytes)
    default_rows_a / default_rows_b(N, C, elt, Rmax, budget_bytes)     block rows from a byte budget (``opt_core.mem.budget``: the device's actual
                                                                       free memory when ``budget_bytes`` is None and CUDA is present); inside the
                                                                       contracts the budget is a share of the free bytes AGREED across ranks
                                                                       (``dist.agreed_free_bytes``: min over ranks) so every rank runs one schedule
    ContractStats                                                      passes / slabs / ring_steps / tiles / matmul_flops / bytes_streamed / mode /
                                                                       mm_pieces / deferred_tiles / deferred_peak_bytes / grid

STREAMED SCHEDULE (``trimul_update_``; u := bytes of the z shard = R*N*C_z*elt, a := R*N*C_h*elt):
    outgoing  1. ``A[C_h, R, N]`` built ONCE from original rows in row blocks (``a_rows``: the engine's global chunk grid clipped to my rows,
                 :meth:`Layout.chunks`).                                                        resident: u + a
              2. pass t = 0..T-1 over the sub-block index of :func:`slab_grid`: every rank produces its own b sub-block t (``[C_h, N, w<=RB]``, GEMM
                 layout) from ORIGINAL rows and the P sub-blocks travel once around the ring (:func:`opt_core.mem.rowpair.dist.ring_blocks`,
                 wire unit ``RB`` rows); for each held sub-block: ``T[C_h, rows, w] = A[:, rows] @ B`` over the FULL k = N locally (M pieces of
                 :func:`mm_row_pieces`) -> ``epilogue_fn`` (``out``: LN + projection of the tile, ``gate`` read from ORIGINAL ``z[rows, cols]``,
                 ``z[rows, cols] += x``). Local rows whose own sub-block t2 > t is not yet produced must stay original -> their tile rows are
                 cloned (``[C_h, rows, w]``) and handed to the epilogue right after sub-block t2 is produced (before that pass streams). Extra
                 resident <= ~a/4 (:func:`deferred_peak_bytes`). Per-rank ring traffic per call: (P-1)/P of the whole projected b.
    incoming  ``x_ij = sum_k a_ki b_kj`` needs z^T rows: 1. ``A[C_h, R, N(k)]`` built ONCE, sub-block by sub-block of the same grid, from my
              band of z^T rows (:func:`zT_band` = banded p2p transpose :func:`opt_core.mem.rowpair.ring.transpose_band`; the mask travels
              transposed once per call by all-to-all) projected with the engine's statements — z^T is never whole on a rank. 2. the same ring /
              tile loop as outgoing (:func:`rowshard_outer_contract`), each pass's b sub-block projected just in time from the z^T band of that
              pass's window (``slab_hook``; never cached), then the same epilogue. No deferral is needed: every column block is written exactly
              at its own sub-block index and the providers of later sub-blocks read other columns. resident: u + a; transient per sub-block: one
              z^T band ``[w<=RB, N, C_z]`` + its projection ``[C_h, N, w]``.
    A-BLOCK   ``RA`` (``trimul_update_(RA=)`` > ``ROWPAIR_TRIMUL_ROWS_A`` > whole; ``ROWPAIR_TRIMUL_ABLOCK=0`` forces whole): A resident ``RA`` rows
    schedule  at a time, ``passes = ceil(Rmax / RA)``, steps 2 of both directions once per pass. One pass (the default) IS the schedule above.
              More than one: z is READ-ONLY through the call (nothing deferred; own sub-blocks, the peers' band fetches and the gates read the
              original device shard in every pass), the epilogue's statements run on the pass's OUTPUT ROW BLOCK ``[RA, N, C_z]`` (pre-filled per
              tile with z's original block), the block goes to a host ROW MIRROR (u bytes, page-locked chunks, once per process) after the pass,
              and one copy mirror -> z ends the call. resident: u + 2 * RA*N*C (+ the transients above); traffic: ring (and bands) x passes,
              host D2H u + H2D u per call. Per element the arithmetic is the one-pass arithmetic (same units, same statements).
    Kernel launches: every elementwise / LayerNorm / projection launch is a row block or a tile (< 2**31 elements); matmul operand pieces
    < 2**31 elements under ``mm_rows=auto``.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import RowpairRefused
from ._torch import torch
from .dist import Layout, agreed_free_bytes, all_gather_rows, alltoall_window, env_int, require_schedule_dtype, require_sharded, ring_blocks, transpose_shards
from .ring import transpose_band
from .shard import choose_block_rows
from .evidence import record_schedule

__all__ = ["INT32_MAX", "gemm_a_layout", "gemm_b_layout", "max_rows_per_launch", "default_rows_a", "default_rows_b", "default_sub", "slab_grid",
           "inplace_chunk_bounds", "grid_bounds", "grid_windows", "zT_band", "mm_row_pieces", "deferred_peak_bytes", "ContractStats",
           "rowshard_outer_contract", "colshard_inner_contract", "trimul_outgoing", "trimul_incoming", "trimul_dense", "tile_writer", "TriMulFns",
           "trimul_update_", "ablock_rows", "require_ablock_agreement", "describe_ablock", "host_mirror", "host_mirror_bytes", "release_host_mirror", "release_leased_mirror", "pass_windows", "ABLOCK", "ENV_ABLOCK"]

INT32_MAX = 2 ** 31 - 1
ABUF_FRAC, BBUF_FRAC = 0.30, 0.20         # default share of the device's free bytes for the resident A block / the streamed B slabs


# ----------------------------------------------------------------------------------------------------------------- layouts / budgets
def gemm_a_layout(x_rows):
    """``[rows, N, C]`` (channel-last row block) -> A block ``[C, rows, N]`` contiguous."""
    return x_rows.permute(2, 0, 1).contiguous()


def gemm_b_layout(x_rows):
    """``[rows, N, C]`` (channel-last row block) -> B block ``[C, N, rows]`` contiguous."""
    return x_rows.permute(2, 1, 0).contiguous()


def max_rows_per_launch(N: int, C: int) -> int:
    """Largest row count r such that an ``[r, N, C]`` elementwise launch has < 2**31 elements (``ROWPAIR_ELEM_ROWS`` overrides; >= 1)."""
    env = os.environ.get("ROWPAIR_ELEM_ROWS")
    if env:
        return max(1, int(env))
    return max(1, INT32_MAX // (int(N) * int(C)))


def _budget(frac: float, budget_bytes: Optional[int]) -> Optional[int]:
    if budget_bytes is not None:
        return int(budget_bytes)
    from ..budget import budget_bytes as dev_budget
    return dev_budget(frac)                                   # None without CUDA


def _share(frac: float, free: Optional[int]) -> Optional[int]:
    """``frac`` of the AGREED free bytes (:func:`opt_core.mem.rowpair.dist.agreed_free_bytes`) as a byte budget — the rule of
    :func:`opt_core.mem.budget.budget_bytes` applied to the cross-rank value; None (no CUDA on any rank) keeps the table default."""
    return None if free is None else max(0, int(int(free) * float(frac)))


def default_rows_a(N: int, C: int, elt_bytes: int = 2, Rmax: Optional[int] = None, budget_bytes: Optional[int] = None) -> int:
    """RA = rows of the resident A block per pass: ``ROWPAIR_TRIMUL_ROWS_A``, else the largest multiple of 128 with ``RA*N*C*elt <= budget``
    (``budget_bytes``, default :data:`ABUF_FRAC` of the device's free bytes; without CUDA: ``Rmax``), clipped to ``Rmax``, >= 1."""
    env = os.environ.get("ROWPAIR_TRIMUL_ROWS_A")
    if env:
        ra = int(env)
    else:
        b = _budget(ABUF_FRAC, budget_bytes)
        ra = int(Rmax or 128) if b is None else max(128, (int(b // (N * C * elt_bytes)) // 128) * 128)
    if Rmax is not None:
        ra = min(ra, int(Rmax))
    return max(1, ra)


def default_rows_b(N: int, C: int, elt_bytes: int = 2, Rmax: Optional[int] = None, budget_bytes: Optional[int] = None) -> int:
    """RB = rows per streamed B slab: ``ROWPAIR_TRIMUL_ROWS_B``, else the largest multiple of 128 with ``2*RB*N*C*elt <= budget``
    (default :data:`BBUF_FRAC` of the device's free bytes; without CUDA: ``Rmax``), clipped to ``Rmax``, >= 1."""
    env = os.environ.get("ROWPAIR_TRIMUL_ROWS_B")
    if env:
        rb = int(env)
    else:
        b = _budget(BBUF_FRAC, budget_bytes)
        rb = int(Rmax or 128) if b is None else max(128, (int(b // (2 * N * C * elt_bytes)) // 128) * 128)
    if Rmax is not None:
        rb = min(rb, int(Rmax))
    return max(1, rb)


class ContractStats(object):
    """Accounting filled by the contractions (evidence: ``passes slabs ring_steps tiles matmul_flops bytes_streamed mode RA RB``)."""

    def __init__(self):
        self.passes = 0
        self.slabs = 0
        self.ring_steps = 0
        self.tiles = 0
        self.matmul_flops = 0.0
        self.bytes_streamed = 0
        self.mode = ""
        self.RA = None
        self.RB = None
        self.tiles_source = None
        self.mm_pieces = 0                 # matmul launches (== tiles unless mm_rows splits a tile's rows)
        self.deferred_tiles = 0            # outgoing streamed schedule: tile row pieces held back until their rows' own sub-block was produced
        self.deferred_peak_bytes = 0       # the most bytes such pieces held at once
        self.grid = None                   # "rank" (uniform RB sub-blocks from each rank start) | "bounds" (col_bounds honoured)

    def facts(self) -> dict:
        return {"passes": self.passes, "slabs": self.slabs, "ring_steps": self.ring_steps, "tiles": self.tiles,
                "matmul_tflop": round(self.matmul_flops / 1e12, 3), "gb_streamed": round(self.bytes_streamed / 1e9, 3), "mode": self.mode or None,
                "RA": self.RA, "RB": self.RB, "mm_pieces": self.mm_pieces, "deferred_tiles": self.deferred_tiles,
                "deferred_peak_gb": round(self.deferred_peak_bytes / 1e9, 3), "grid": self.grid}

    def __repr__(self):
        return "ContractStats(" + ", ".join(f"{k}={v!r}" for k, v in self.facts().items()) + ")"


def default_sub(N: int) -> int:
    """``RB`` of the streamed schedule when neither the argument nor ``ROWPAIR_TRIMUL_SUB`` gives it: 512 rows per b sub-block up to 16384
    tokens, 128 above (halves the ring / tile transients at very large N; same bytes moved, more messages). A pure function of N, hence
    identical on every rank."""
    return 512 if int(N) <= 16384 else 128


def _sub_rows(N: int, RB: Optional[int]) -> Tuple[int, str]:
    """(RB, source) for the streamed schedule: ``ROWPAIR_TRIMUL_SUB`` > the argument > :func:`default_sub`."""
    env = os.environ.get("ROWPAIR_TRIMUL_SUB", "").strip()
    if env:
        return max(1, int(env)), "env"
    if RB:
        return max(1, int(RB)), "fixed"
    return default_sub(N), "table"


def slab_grid(layout: Layout, RB: int, col_bounds: Optional[Sequence[int]] = None) -> List[List[Tuple[int, int]]]:
    """The b sub-block grid of one contraction call: ``grid[q]`` = rank q's sub-blocks as LOCAL row ranges ``(j0, j1)`` (ascending, each at most
    ``RB`` rows, together exactly ``[0, R_q)``). Sub-blocks start at every rank boundary and — when ``col_bounds`` lists extra GLOBAL boundaries
    (an engine's in-place column-chunk grid, so the einsum's column extents equal the engine's) — at those too; inside each interval they step
    by ``RB``. ``col_bounds=None`` is the uniform grid ``(t*RB, min(R_q, (t+1)*RB))`` of :func:`rowshard_outer_contract`. Identical on every
    rank (a pure function of the layout and the arguments)."""
    N, RB = layout.N, max(1, int(RB))
    bset = {0, N}
    for q0, q1 in layout.bounds:
        bset.add(q0)
        bset.add(q1)
    if col_bounds:
        bset |= {int(b) for b in col_bounds if 0 <= int(b) <= N}
    bounds = sorted(bset)
    blocks = []
    for g0, g1 in zip(bounds[:-1], bounds[1:]):
        for s0 in range(g0, g1, RB):
            blocks.append((s0, min(s0 + RB, g1)))
    grid = [[(j0 - q0, j1 - q0) for (j0, j1) in blocks if q0 <= j0 and j1 <= q1] for (q0, q1) in layout.bounds]
    if sum(len(g) for g in grid) != len(blocks):
        raise RowpairRefused("slab_grid: violated: " + repr((blocks, layout.bounds)))
    return grid


def inplace_chunk_bounds(N: int, chunk: int) -> List[int]:
    """The column-block boundaries of the stock IN-PLACE triangle-multiplication inference statement (``i_range = range(0, half_n, chunk)``
    with ``half_n = ceil(N/2)`` — the chunk before ``half_n`` is contracted — then ``range(half_n, N, chunk)`` truncated at N): the ``col_bounds`` under
    which the streamed schedule's einsum column extents equal that statement's (``grid="stock"``)."""
    N, chunk = int(N), max(1, int(chunk))
    half_n = N // 2 + N % 2
    return sorted(set(list(range(0, half_n, chunk)) + list(range(half_n, N, chunk)) + [N]))


def grid_bounds(N: int, grid: Optional[str] = None, inplace_chunk: Optional[int] = None) -> Tuple[Optional[List[int]], str]:
    """``(col_bounds, word)`` of the streamed schedule's b sub-block grid: ``grid`` (else ``ROWPAIR_TRIMUL_GRID``, default ``stock``) = ``stock``
    -> :func:`inplace_chunk_bounds` of ``inplace_chunk``; ``rank`` -> None (uniform ``RB`` sub-blocks from each rank start). Anything else is
    refused by name."""
    word = (grid or os.environ.get("ROWPAIR_TRIMUL_GRID", "") or "stock").strip().lower()
    if word == "stock":
        if inplace_chunk is None or int(inplace_chunk) < 1:
            raise RowpairRefused("grid_bounds: grid='stock' needs inplace_chunk (the engine's in-place column chunk)")
        return inplace_chunk_bounds(N, int(inplace_chunk)), word
    if word == "rank":
        return None, word
    raise RowpairRefused(f"ROWPAIR_TRIMUL_GRID/grid must be 'stock' or 'rank', got {word!r}")


def grid_windows(layout: Layout, grid: List[List[Tuple[int, int]]], t: int) -> List[Tuple[int, int]]:
    """GLOBAL row windows ``(w0, w1)`` of every rank's sub-block ``t`` (``(0, 0)`` for a rank without one) — the columns of z a rank wants as rows
    of z^T for that sub-block."""
    return [(layout.bounds[q][0] + g[t][0], layout.bounds[q][0] + g[t][1]) if t < len(g) else (0, 0) for q, g in enumerate(grid)]


def zT_band(z_shard, layout: Layout, windows: Sequence[Tuple[int, int]]):
    """Banded distributed transpose: my band of z^T rows ``[w_me, N, C]`` (``out[i, j, :] = z[j, w0_me + i, :]``; None when my window is empty) for
    per-rank GLOBAL windows (:func:`grid_windows`) — z^T is never materialised whole. EVERY rank calls it (it serves the other ranks' windows
    from its rows). Transport: :func:`opt_core.mem.rowpair.ring.transpose_band` (p2p pieces; bit-preserving; census ``trimul_band=ring``)."""
    record_schedule(trimul_band="ring")
    return transpose_band(z_shard, layout, list(windows))


def mm_row_pieces(rows: int, N: int, C: int, mm_rows=None) -> List[Tuple[int, int]]:
    """Relative row pieces ``(p0, p1)`` of a tile's ``rows`` local rows for the matmul launches. ``mm_rows``: None -> one piece (the A block as
    is); ``"auto"`` -> one piece unless the A piece ``[C, rows, N]`` would reach 2**31 elements, then pieces of the largest multiple of 256 rows
    below that; ``0`` -> never split; an int -> pieces of that many rows. ``ROWPAIR_TRIMUL_ROWS`` (auto | 0 | rows) overrides a non-None
    argument. M-only change: every output element's k-contraction stays whole inside one matmul."""
    rows = int(rows)
    if mm_rows is None or rows <= 0:
        return [(0, rows)]
    spec = (os.environ.get("ROWPAIR_TRIMUL_ROWS", "").strip().lower() or str(mm_rows)).strip().lower()
    if spec == "0":
        return [(0, rows)]
    if spec == "auto":
        capped, _ = choose_block_rows(int(N), int(C), rows=rows, env=None, cap_elems=INT32_MAX)   # THE chooser: the int32 launch cap only
        if capped >= rows:
            return [(0, rows)]
        Rp = (capped // 256) * 256 or capped                                 # matmul pieces on a 256-row grain when the cap allows one
    else:
        Rp = max(1, int(spec))
    return [(p, min(p + Rp, rows)) for p in range(0, rows, Rp)]


def deferred_peak_bytes(layout: Layout, grid: List[List[Tuple[int, int]]], C: int, elt: int, rank: Optional[int] = None) -> int:
    """The exact peak (bytes) of the outgoing streamed schedule's deferral buffer on ``rank`` (default: this rank) for ``grid``: at sub-block
    index t the tiles of local rows at or beyond the end of own sub-block t (rows of not-yet-produced own sub-blocks) x the columns of every
    rank's sub-block t are held (``C`` channels, ``elt`` bytes each) until their rows' sub-block is produced."""
    r = layout.rank if rank is None else int(rank)
    own = grid[r]
    R = layout.nrows(r)
    T = max(len(g) for g in grid)
    live, peak = 0, 0
    per_t2: Dict[int, int] = {}
    for t in range(T):
        live -= per_t2.pop(t, 0)
        hi = own[t][1] if t < len(own) else R
        cols = sum((g[t][1] - g[t][0]) for g in grid if t < len(g))
        for t2 in range(t + 1, len(own)):
            l0, l1 = own[t2]
            add = (l1 - l0) * cols * int(C) * int(elt) if l0 >= hi else max(0, l1 - max(l0, hi)) * cols * int(C) * int(elt)
            per_t2[t2] = per_t2.get(t2, 0) + add
            live += add
        peak = max(peak, live)
    return peak


def _ring_slabs(slab, layout: Layout, grid, t: int, RB: int, dim: int = 2, uniform: bool = True):
    """Every rank's sub-block ``t`` of ``grid`` around the ring: yields ``(q, block)`` for EVERY rank q (``block`` has ``w_q(t)`` entries along
    ``dim``, possibly 0 -> still yielded). Wire unit = ``RB`` rows (:func:`opt_core.mem.rowpair.dist.ring_blocks`, zero padded, double
    buffered). ``uniform=True`` (the grid of ``col_bounds=None``) is exactly ``ring_blocks(slab, layout, rows=(t*RB, RB))``; otherwise the own
    sub-block travels padded to the same unit and each yielded block is narrowed to its source's width. Must be exhausted on every rank."""
    if uniform:
        for q, blk in ring_blocks(slab, layout, rows=(t * RB, RB), dim=dim):       # the uniform-grid path
            yield q, blk
        return

    def width(q):
        g = grid[q]
        return (g[t][1] - g[t][0]) if t < len(g) else 0

    unit_me = min(layout.R, RB)                                                    # ring_blocks(rows=(0, RB)) expects min(R, RB) own entries
    own = slab
    if int(slab.shape[dim]) != unit_me:
        shape = list(slab.shape)
        shape[dim] = unit_me
        own = slab.new_zeros(shape)
        if slab.shape[dim] > 0:
            own.narrow(dim, 0, int(slab.shape[dim])).copy_(slab)
    for q, blk in ring_blocks(own, layout, rows=(0, RB), dim=dim):
        yield q, blk.narrow(dim, 0, width(q))


def _tiles(A, i0: int, i1: int, Bq, pieces: List[Tuple[int, int]], C: int, N: int, st: "ContractStats"):
    """The matmul launches of one held B block against the A block of local rows ``i0:i1``: yields ``(T, (p0, p1))`` per M piece (``T =
    [C, p1-p0, v]``, local rows ``p0:p1``)."""
    v = int(Bq.shape[2])
    for (a0, a1) in pieces:
        if a1 <= a0:
            continue
        if not (C * (a1 - a0) * v <= INT32_MAX):
            raise RowpairRefused("rowpair.trimul: violated: " + repr((("tile too large", C, a1 - a0, v))))
        Ap = A if (a0 == 0 and a1 == int(A.shape[1])) else A[:, a0:a1]
        T = torch.matmul(Ap, Bq)                              # [C, rows, v]  full k
        st.tiles += 1
        st.mm_pieces += 1
        st.matmul_flops += 2.0 * C * (a1 - a0) * v * N
        yield T, (i0 + a0, i0 + a1)


def _b_slab(b_source, jb: int, j0: int, j1: int, C: int, N: int, dtype, device):
    """This rank's B slab ``[C, N, j1-j0]`` (contiguous) for local rows ``j0:j1``."""
    if j1 <= j0:
        return torch.empty((C, N, 0), dtype=dtype, device=device)
    if callable(b_source):
        s = b_source(j0, j1)
    elif torch.is_tensor(b_source):          # whole b in gemm_b layout [C, N, R]
        s = b_source[:, :, j0:j1]
    else:                                     # list of cached slabs (gemm_b layout each)
        s = b_source[jb]
    if not (s.shape == (C, N, j1 - j0)):
        raise RowpairRefused("rowpair.trimul: violated: " + repr(((tuple(s.shape), (C, N, j1 - j0)))))
    require_schedule_dtype(s, dtype, device, "trimul: the b projection (proj(..., is_a=False))")
    return s.contiguous()


def _b_plane(b_t, z_shard, C_h: int, w: int, N: int, dtype, device):
    """The plane a fused provider's ``proj_into`` writes a b sub-block of ``w`` rows into: ``[C_h, w, N]`` contiguous (today's statement; ``_b_slab``
    makes its ``[C_h, N, w]`` transpose contiguous), or — when the provider answers ``b_transposed(z_shard)`` True — the ``[C_h, w, N]`` VIEW of a
    contiguous ``[C_h, N, w]`` buffer, so the projection lands in the GEMM-B layout directly and no transpose copy follows. A provider without the
    hook (every plain :class:`TriMulFns`) or answering False gets the first, byte for byte."""
    if b_t is not None and b_t(z_shard):
        return torch.empty((C_h, N, w), dtype=dtype, device=device).transpose(1, 2)
    return torch.empty((C_h, w, N), dtype=dtype, device=device)


# ----------------------------------------------------------------------------------------------------------------- (a) outer
def rowshard_outer_contract(a_provider: Callable[[int, int], object], b_source, layout: Layout,
                            epilogue_fn: Callable[[object, Tuple[int, int], Tuple[int, int]], None], *, N: int, C: int, dtype, device,
                            RA: Optional[int] = None, RB: Optional[int] = None, stats: Optional[ContractStats] = None,
                            col_bounds: Optional[Sequence[int]] = None, mm_rows=None, defer_pending_rows: bool = False,
                            slab_hook: Optional[Callable[[int], None]] = None,
                            pass_hook: Optional[Callable[[int, int, int], None]] = None) -> ContractStats:
    """``x[I, J]`` tiles of ``T[c,i,j] = sum_k A[c,i,k] * B[c,k,j]`` for this rank's rows I (all local row sub-blocks of RA rows) against EVERY
    rank's b rows J (streamed around the ring in slabs of RB rows).

    ``a_provider(i0, i1)`` -> A block ``[C, i1-i0, N]`` contiguous for LOCAL rows ``i0:i1`` (called once per pass; may return None when
    ``i1 <= i0``). ``b_source``: this rank's b in gemm_b layout ``[C, N, R]`` | list of slabs ``[C, N, RB_v]`` (jb-th slab = local rows
    ``jb*RB : ...``) | callable ``b_provider(j0, j1) -> [C, N, j1-j0]`` (recomputed once per (pass, slab)). ``epilogue_fn(T, (i0, i1), (c0, c1))``:
    ``T = [C, i1-i0, c1-c0]`` tile for LOCAL rows ``i0:i1`` and GLOBAL columns ``c0:c1`` (consumed before the next tile). All ranks execute
    the same (pass, slab, ring-step) schedule (counts derive from Rmax, not R), so ranks with fewer rows still relay blocks. Memory per rank
    (elements of dtype): A = RA*N*C (+ provider transients), ring = 2*RB*N*C recv buffers (+ RB*N*C if the own slab needs padding), tile =
    C*RA*RB (+ epilogue transients), plus b_source if cached (R*N*C).

    Streamed-schedule options (all default to the uniform schedule above): ``col_bounds`` — extra GLOBAL boundaries the b slabs honour
    (:func:`slab_grid`; slab index t then walks every rank's t-th sub-block, absent sub-blocks relay nothing); ``mm_rows`` — the matmul of a held
    slab runs in M pieces of the A block (:func:`mm_row_pieces`; the epilogue then receives row pieces); ``defer_pending_rows`` — for a callable
    ``b_source`` that reads the rows ``epilogue_fn`` WRITES (an in-place update of the source shard): tile row pieces for local rows whose own
    slab has not been produced yet are cloned and delivered to ``epilogue_fn`` right after that slab is produced (before its pass streams), so
    every source row is read by the provider before any epilogue write touches it — per-element values unchanged, the epilogue's M differs;
    requires ONE A pass (``RA >= layout.Rmax``), refused by name otherwise; ``slab_hook(jb)`` — called on EVERY rank at the start of slab index
    jb of every pass, before the own slab is produced (a per-slab collective prologue, e.g. the z^T band exchange of the incoming update);
    ``pass_hook(p, i0, i1)`` — called on every rank after the last tile of pass p (local rows ``i0:i1``, ``i1 <= i0`` on a rank with no rows in
    that pass): the row-blocked update flushes the pass's output rows there."""
    require_sharded(layout, "rowshard_outer_contract")
    st = stats or ContractStats()
    R, Rmax = layout.R, layout.Rmax
    elt = torch.empty((), dtype=dtype).element_size()
    src = "fixed" if (RA and RB) else ("env" if (os.environ.get("ROWPAIR_TRIMUL_ROWS_A") or os.environ.get("ROWPAIR_TRIMUL_ROWS_B")) else "budget")
    free = None if (RA and RB) else agreed_free_bytes()               # world size > 1 here: ONE free-bytes value for every rank's schedule
    RA = int(RA or default_rows_a(N, C, elt, Rmax, _share(ABUF_FRAC, free)))
    RB = int(RB or default_rows_b(N, C, elt, Rmax, _share(BBUF_FRAC, free)))
    st.RA, st.RB, st.tiles_source = RA, RB, src
    grid = slab_grid(layout, RB, col_bounds)
    st.grid = "bounds" if col_bounds else "rank"
    npass = -(-Rmax // RA)
    if defer_pending_rows and npass != 1:
        raise RowpairRefused(f"rowshard_outer_contract: defer_pending_rows needs one A pass (RA={RA} < Rmax={Rmax}): a later pass would re-read "
                             "source rows the epilogue already wrote")
    if defer_pending_rows and not callable(b_source):
        raise RowpairRefused("rowshard_outer_contract: defer_pending_rows is the schedule of a CALLABLE b_source (a cached b needs no deferral)")
    record_schedule(trimul_RA=RA, trimul_RB=RB, tiles_source=src, layout_B=layout.B, trimul_grid=st.grid,
                    trimul_mm_rows=("one" if mm_rows is None else (os.environ.get("ROWPAIR_TRIMUL_ROWS", "").strip().lower() or str(mm_rows))),
                    trimul_defer=bool(defer_pending_rows))
    T_slabs = max(len(g) for g in grid)                                # == ceil(Rmax / RB) for the uniform grid (rank 0 owns Rmax rows)
    own = grid[layout.rank]
    deferred: Dict[int, list] = {}
    live = 0
    for p in range(npass):
        i0, i1 = min(R, p * RA), min(R, (p + 1) * RA)
        A = a_provider(i0, i1) if i1 > i0 else None
        if A is not None:
            if not (A.shape == (C, i1 - i0, N) and A.is_contiguous()):
                raise RowpairRefused("rowpair.trimul: violated: " + repr(((tuple(A.shape), (C, i1 - i0, N)))))
        pieces = mm_row_pieces(i1 - i0, N, C, mm_rows)
        st.passes += 1
        for jb in range(T_slabs):
            j0, j1 = own[jb] if jb < len(own) else (R, R)
            if slab_hook is not None:
                slab_hook(jb)
            slab = _b_slab(b_source, jb, j0, j1, C, N, dtype, device)        # produced from ORIGINAL rows j0:j1 (streamed schedule)
            produced_hi = j1 if jb < len(own) else R                         # local rows < produced_hi may be written from now on
            if defer_pending_rows:
                for (cols, q0, q1, Tq) in deferred.pop(jb, []):              # rows of own slab jb are free now: deliver their held tiles
                    live -= Tq.numel() * Tq.element_size()
                    epilogue_fn(Tq, (q0, q1), cols)
                    del Tq
            st.slabs += 1
            for q, Bq in _ring_slabs(slab, layout, grid, jb, RB, dim=2, uniform=not col_bounds):
                st.ring_steps += 1
                if q != layout.rank:
                    st.bytes_streamed += C * N * RB * elt
                v = Bq.shape[2]
                if A is None or v == 0:
                    continue
                c0 = layout.bounds[q][0] + grid[q][jb][0]
                cols = (c0, c0 + v)
                for T, (p0, p1) in _tiles(A, i0, i1, Bq, pieces, C, N, st):
                    if not defer_pending_rows or produced_hi >= p1:
                        epilogue_fn(T, (p0, p1), cols)
                    else:
                        hi = max(p0, min(p1, produced_hi))
                        if hi > p0:                                          # immediate rows
                            epilogue_fn(T[:, : hi - p0], (p0, hi), cols)
                        d0 = max(p0, produced_hi)                            # rows of not-yet-produced own slabs -> defer per future slab
                        for t2 in range(jb + 1, len(own)):
                            l0, l1 = own[t2]
                            q0, q1 = max(l0, d0), min(l1, p1)
                            if q1 > q0:
                                Tq = T[:, q0 - p0: q1 - p0].clone()
                                live += Tq.numel() * Tq.element_size()
                                st.deferred_tiles += 1
                                st.deferred_peak_bytes = max(st.deferred_peak_bytes, live)
                                deferred.setdefault(t2, []).append((cols, q0, q1, Tq))
                    del T
            del slab
        del A
        if pass_hook is not None:
            pass_hook(p, i0, i1)
    if deferred:
        raise RowpairRefused(f"rowshard_outer_contract: internal: undelivered deferred tiles for slabs {sorted(deferred)}")
    st.mode = st.mode or ("b:tensor" if torch.is_tensor(b_source) else ("b:provider" if callable(b_source) else "b:slabs"))
    return st


# ----------------------------------------------------------------------------------------------------------------- (b) inner
def colshard_inner_contract(pa_window: Callable[[int, int], object], pb_window: Callable[[int, int], object], layout: Layout,
                            epilogue_fn: Callable[[object, Tuple[int, int], Tuple[int, int]], None], *, N: int, C: int, dtype, device,
                            RA: Optional[int] = None, RB: Optional[int] = None, cache_b: bool = True, b_slabs: Optional[List[object]] = None,
                            a2a_chunks: Optional[int] = None, stats: Optional[ContractStats] = None, mm_rows=None,
                            a_window_rows: Optional[int] = None) -> ContractStats:
    """``x[I, J]`` tiles of ``T[c,i,j] = sum_k a[k,i,c] * b[k,j,c]`` where a, b are ROW(k)-sharded ``[R_k, N, C]`` quantities available through
    column-window providers on their k-owner: ``pa_window(c0, c1) -> a[K_mine, c0:c1, :]`` (``[R, c1-c0, C]``), ``pb_window(c0, c1) ->
    b[K_mine, c0:c1, :]`` (providers must read from data that ``epilogue_fn`` does not modify during the call — e.g. the operand tensors or a
    pre-update copy — because other ranks' later passes still request windows of ALL local rows). Step 1 (once, or per pass if
    ``cache_b=False``): B slabs = all-to-all of ``b[:, J_slab]`` into gemm_b layout ``[C, N(k), RB_v(j)]`` on rank(J). Step 2 per pass: A =
    all-to-all of ``a[:, I_sub]`` into gemm_a layout ``[C, RA_v(i), N(k)]`` on rank(I); then the ring/tile loop of (a). ``b_slabs``: precomputed
    slabs (list, gemm_b layout) to skip step 1. Memory per rank: A = RA*N*C (+ all-to-all transient (1+2/chunks)*RA*N*C), B cache = R*N*C if
    cache_b else per-slab transient (1+2/chunks)*RB*N*C, ring = 2*RB*N*C, tile = C*RA*RB.

    Streamed-schedule options: ``mm_rows`` as in :func:`rowshard_outer_contract`; ``a_window_rows`` — the A block of a pass is assembled in
    all-to-all WINDOWS of that many rows (each written into its rows of A) instead of one exchange, so the provider blocks + staging of one
    window (~(1 + 2/chunks) * a_window_rows * N * C * P/P) are the transient, never a second A. With ONE pass (``RA >= Rmax``), ``cache_b=False``
    and providers that read the shard the epilogue updates in place, the schedule is hazard-free: A is assembled before any write, slab jb is
    assembled from column windows no earlier slab's tiles wrote, and each (rows, cols) tile is written exactly once."""
    require_sharded(layout, "colshard_inner_contract")
    st = stats or ContractStats()
    R, Rmax = layout.R, layout.Rmax
    elt = torch.empty((), dtype=dtype).element_size()
    src = "fixed" if (RA and RB) else ("env" if (os.environ.get("ROWPAIR_TRIMUL_ROWS_A") or os.environ.get("ROWPAIR_TRIMUL_ROWS_B")) else "budget")
    free = None if (RA and RB) else agreed_free_bytes()               # world size > 1 here: ONE free-bytes value for every rank's schedule
    RA = int(RA or default_rows_a(N, C, elt, Rmax, _share(ABUF_FRAC, free)))
    RB = int(RB or default_rows_b(N, C, elt, Rmax, _share(BBUF_FRAC, free)))
    st.RA, st.RB, st.tiles_source = RA, RB, src
    st.grid = "rank"
    record_schedule(trimul_RA=RA, trimul_RB=RB, tiles_source=src, layout_B=layout.B, trimul_grid=st.grid,
                    trimul_mm_rows=("one" if mm_rows is None else (os.environ.get("ROWPAIR_TRIMUL_ROWS", "").strip().lower() or str(mm_rows))),
                    trimul_a_window_rows=int(a_window_rows or 0))
    nslab = -(-Rmax // RB)

    def make_slab(jb):
        return alltoall_window(pb_window, layout, jb * RB, RB, C=C, dtype=dtype, device=device, out_layout="gemm_b", chunks=a2a_chunks)

    def make_A(p):
        i0, i1 = min(R, p * RA), min(R, (p + 1) * RA)
        if not a_window_rows or int(a_window_rows) >= RA:
            return alltoall_window(pa_window, layout, p * RA, RA, C=C, dtype=dtype, device=device, out_layout="gemm_a", chunks=a2a_chunks)
        Wb = max(1, int(a_window_rows))
        A = torch.empty((C, i1 - i0, N), dtype=dtype, device=device)
        for w0 in range(0, RA, Wb):                                      # same window schedule on every rank (offsets within the pass)
            W = min(Wb, RA - w0)
            l0, l1 = min(R, p * RA + w0), min(R, p * RA + w0 + W)         # my rows of this window (may be empty: still exchanged)
            alltoall_window(pa_window, layout, p * RA + w0, W, C=C, dtype=dtype, device=device, out_layout="gemm_a", chunks=a2a_chunks,
                            out=A[:, l0 - i0: l1 - i0, :])
        return A

    if b_slabs is None and cache_b:
        b_slabs = [make_slab(jb) for jb in range(nslab)]
        st.mode = "colshard:b-cached"
    elif b_slabs is not None:
        st.mode = "colshard:b-given"
    else:
        st.mode = "colshard:b-per-pass"
    npass = -(-Rmax // RA)
    for p in range(npass):
        A = make_A(p)
        i0, i1 = min(R, p * RA), min(R, (p + 1) * RA)
        if not (A.shape == (C, i1 - i0, N)):
            raise RowpairRefused("rowpair.trimul: violated: " + repr(((tuple(A.shape), (C, i1 - i0, N)))))
        pieces = mm_row_pieces(i1 - i0, N, C, mm_rows)
        st.passes += 1
        for jb in range(nslab):
            slab = b_slabs[jb] if b_slabs is not None else make_slab(jb)
            st.slabs += 1
            for q, Bq in ring_blocks(slab, layout, rows=(jb * RB, RB), dim=2):
                st.ring_steps += 1
                if q != layout.rank:
                    st.bytes_streamed += C * N * RB * elt
                v = Bq.shape[2]
                if i1 <= i0 or v == 0:
                    continue
                c0 = layout.bounds[q][0] + jb * RB
                for T, (p0, p1) in _tiles(A, i0, i1, Bq, pieces, C, N, st):
                    epilogue_fn(T, (p0, p1), (c0, c0 + v))
                    del T
            if b_slabs is None:
                del slab
        del A
    return st


# ----------------------------------------------------------------------------------------------------------------- tensor-level forms
def tile_writer(x_shard):
    """The epilogue that WRITES tiles into a channel-last output shard ``x_shard[R, N, C]``: ``x_shard[i0:i1, c0:c1, :] = T.permute(1, 2, 0)``.
    An engine's epilogue (LayerNorm-out, output projection, gating, residual add on the tile) wraps or replaces it."""
    def epilogue(T, rows: Tuple[int, int], cols: Tuple[int, int]) -> None:
        i0, i1 = rows
        c0, c1 = cols
        x_shard[i0:i1, c0:c1, :].copy_(T.permute(1, 2, 0))
    return epilogue


def trimul_outgoing(a_shard, b_shard, layout: Layout, epilogue_fn: Optional[Callable] = None, *, out=None, RA: Optional[int] = None,
                    RB: Optional[int] = None, stats: Optional[ContractStats] = None):
    """``x_ij = sum_k a_ik b_jk`` for this rank's rows i and all columns j from the projected operand shards ``a_shard``, ``b_shard``
    (``[R, N, C]``, channel-last, this rank's rows). Tiles go to ``epilogue_fn`` (default: :func:`tile_writer` into ``out``, allocated
    ``[R, N, C]`` when None). Returns ``(out_or_None, stats)`` — ``out`` is None when a caller-supplied epilogue consumed the tiles."""
    R, N, C = int(a_shard.shape[0]), int(a_shard.shape[1]), int(a_shard.shape[2])
    if tuple(b_shard.shape) != (R, N, C) or R != layout.R or N != layout.N:
        raise RowpairRefused(f"trimul_outgoing: a {tuple(a_shard.shape)} b {tuple(b_shard.shape)} vs layout R={layout.R} N={layout.N}")
    if epilogue_fn is None:
        if out is None:
            out = a_shard.new_empty((R, N, C))
        epilogue_fn = tile_writer(out)
    b_gemm = gemm_b_layout(b_shard)                                              # [C, N, R] once (the cached-b mode)
    st = rowshard_outer_contract(lambda i0, i1: gemm_a_layout(a_shard[i0:i1]), b_gemm, layout, epilogue_fn, N=N, C=C,
                                 dtype=a_shard.dtype, device=a_shard.device, RA=RA, RB=RB, stats=stats)
    return out, st


def trimul_incoming(a_shard, b_shard, layout: Layout, epilogue_fn: Optional[Callable] = None, *, out=None, RA: Optional[int] = None,
                    RB: Optional[int] = None, cache_b: bool = True, a2a_chunks: Optional[int] = None, stats: Optional[ContractStats] = None):
    """``x_ij = sum_k a_ki b_kj`` for this rank's rows i and all columns j; ``a_shard`` / ``b_shard`` hold this rank's rows k of the projected
    operands (``[R, N, C]``) and are only READ (the epilogue must not modify them). Same return as :func:`trimul_outgoing`."""
    R, N, C = int(a_shard.shape[0]), int(a_shard.shape[1]), int(a_shard.shape[2])
    if tuple(b_shard.shape) != (R, N, C) or R != layout.R or N != layout.N:
        raise RowpairRefused(f"trimul_incoming: a {tuple(a_shard.shape)} b {tuple(b_shard.shape)} vs layout R={layout.R} N={layout.N}")
    if epilogue_fn is None:
        if out is None:
            out = a_shard.new_empty((R, N, C))
        epilogue_fn = tile_writer(out)
    st = colshard_inner_contract(lambda c0, c1: a_shard[:, c0:c1, :], lambda c0, c1: b_shard[:, c0:c1, :], layout, epilogue_fn, N=N, C=C,
                                 dtype=a_shard.dtype, device=a_shard.device, RA=RA, RB=RB, cache_b=cache_b, a2a_chunks=a2a_chunks, stats=stats)
    return out, st


def trimul_dense(a, b, outgoing: bool):
    """The dense statement the sharded forms reproduce: ``a, b [N, N, C]`` channel-last -> ``x [N, N, C]`` with
    ``x = matmul(a.permute(2,0,1), b.permute(2,1,0))`` (outgoing) / ``matmul(a.permute(2,1,0), b.permute(2,0,1))`` (incoming), permuted back."""
    if outgoing:
        p = torch.matmul(a.permute(2, 0, 1), b.permute(2, 1, 0))
    else:
        p = torch.matmul(a.permute(2, 1, 0), b.permute(2, 0, 1))
    return p.permute(1, 2, 0)


# ----------------------------------------------------------------------------------------------------------------- the module-shaped update
class TriMulFns(object):
    """The engine's triangle-multiplication statements as callables (the gated structure: gated projections of LayerNorm'd pair rows, an
    output LayerNorm + projection of the contraction, an output gate read from the ORIGINAL pair values, a residual add):

    ``proj(z_block, mask_block, is_a) -> p``   the a (``is_a=True``) or b projection of a block of ORIGINAL pair values ``[..., C_z]`` with its
                                              mask block ``[..., 1]`` (per-element statements, e.g. ``LN_in -> sigmoid(linear_g) * linear_p *
                                              mask``); returns ``[..., C_h]`` (any strides) in z's DTYPE and on z's device — the pair
                                              schedule dtype every rank sizes its ring buffers with (an autocast projection casts back;
                                              refused by name otherwise: :func:`opt_core.mem.rowpair.dist.require_schedule_dtype`). Called on row blocks ``z[rows, :, :]``
                                              (outgoing; incoming's A/B windows: column blocks ``z[:, cols, :]``).
    ``out(x) -> y``                           ``x [rows, cols, C_h]`` (a permuted tile view) -> ``[rows, cols, C_z]`` (e.g. ``LN_out -> linear_z``);
                                              any dtype torch adds into z (local epilogue, never communicated) — likewise ``gate``
    ``gate(z_block) -> g`` | None             ``z[rows, cols, C_z]`` ORIGINAL values -> the multiplicative gate of ``y`` (e.g. ``sigmoid(linear_g(
                                              LN_in(z)))``); None = no gate
    ``C_h``                                   the hidden channel count of ``proj``'s output
    Optional (a provider such as :func:`opt_core.mem.rowpair.trimul_fused.fused_trimul_fns` adds them; absent = the callables above):
    ``proj_into(dst, z_block, mask_block, is_a) -> bool``
                                              write the projection of ``z_block [rows, N, C_z]`` (mask ``[rows, N]``) DIRECTLY into ``dst``
                                              ``[C_h, rows, N]`` (channel-major: the GEMM-A block of a, the row plane of b); False = declined,
                                              the driver runs ``proj``
    ``tile_epilogue(T, z_block, add) -> bool``  ``z_block [rows, w, C_z] (+)= gate(z_block) * out(T[C_h, rows, w])`` in place in one launch;
                                              False = declined, the driver runs ``out`` / ``gate`` / the add
    ``operand_dtype(z_shard) -> dtype|None``  the dtype of the GEMM operands and ring blocks the provider serves THIS shard with (e.g. bf16
                                              planes for an fp32 shard); absent or None = z's dtype (a provider that declines the shard answers
                                              z's dtype, so its declined units run the statements exactly as a plain TriMulFns does)
    ``b_transposed(z_shard) -> bool``          (optional, asked once per b sub-block) True = the provider's ``proj_into`` writes a b sub-block into the
                                              ``[C_h, w, N]`` VIEW of a contiguous ``[C_h, N, w]`` buffer — the GEMM-B layout directly, no transpose
                                              copy in ``_b_slab``; absent / False = the ``[C_h, w, N]`` plane of the statements above.
    Replicated-by-design: nothing (all operands are pair rows)."""

    def __init__(self, proj: Callable, out: Callable, gate: Optional[Callable], C_h: int):
        self.proj, self.out, self.gate, self.C_h = proj, out, gate, int(C_h)


ENV_ABLOCK = "ROWPAIR_TRIMUL_ABLOCK"            # 0: the update's A plane is whole (one pass) whatever ROWPAIR_TRIMUL_ROWS_A / RA= say (bisecting by name)
ABLOCK_WHOLE_FRAC = 0.80   # planner: the update's A plane is WHOLE (one pass, today's statements) when its u = Rmax*N*C*elt bytes fit this share of the agreed free device
#                            bytes (the rest is the update's own scratch). Set so every measured fold that ran whole keeps its schedule: the largest such ratio measured is
#                            u/free ~0.72 (a 60,120-token x8 forward; margin 0.08), the next 0.46, single-GPU folds <= 0.13. Above 0.80 the plane is row-blocked (the
#                            66,720 / 70,320-token x8 forwards sit at 1.06 / 1.34 — blocked under any threshold below 1)
ABLOCK_BLOCK_FRAC = 0.40   # planner: else the row cap RA is the largest multiple of 128 whose pass working set 2*RA*N*C*elt (A block + output row block) fits this share —
#                            calibrated on the two validated row-blocked forwards (66,720 x8: 134 GB free after the template stage -> 6 x 1,392 rows, the schedule that
#                            completed; 70,320 x8: 118 GB free -> 7 x 1,264 <= the 6 x 1,472 that ran): it errs toward MORE passes (a pass costs a few % of the pair
#                            stack's wall, an OOM costs the forward); the validated forwards used <= 0.45 of the free bytes for this working set
_AUTO_FREE: Dict[Tuple[str, int, int, int, int], Optional[int]] = {}   # (device, N, Rmax, C, elt) -> the agreed free device bytes probed at that shape's FIRST update of the process
ABLOCK: Dict[str, object] = {"calls": 0, "multipass_calls": 0, "ra": None, "passes": None, "source": None, "host": "none", "host_gib": 0.0,
                             "chunks": 0, "pinned_gib": 0.0, "d2h_bytes": 0, "h2d_bytes": 0}
"""This process's A-block record of :func:`trimul_update_` (the kit's TRIMUL census line quotes it through :func:`describe_ablock`): ``ra`` /
``passes`` / ``source`` of the last call, ``multipass_calls`` of ``calls``, the host row mirror's placement ``host`` (``none`` until a multi-pass
call allocates it | ``host_pinned`` | ``host_pageable:<kind>`` | ``host``), its ``host_gib`` / ``chunks`` / ``pinned_gib``, and the bytes it moved."""
_MIRRORS: Dict[Tuple[str, int, int], object] = {}
_LEASED_SAID: Dict[Tuple[str, int, int], str] = {}          # ROWPAIR_HOST_SLAB=lease: the placement word last SAID per mirror key (a line at first use / on change)
_MIRROR_LOCK = __import__("threading").Lock()


def ablock_auto_cap(Rmax: int, N: int, C: int, elt_bytes: int, free_bytes: Optional[int]) -> Tuple[int, str]:
    """The planner's row cap for the update's A block from the AGREED free device bytes at plan time (``free_bytes``; None = no CUDA on any rank):
    ``(Rmax, "whole")`` when the whole A plane ``u = Rmax*N*C*elt`` fits :data:`ABLOCK_WHOLE_FRAC` of them (one pass: today's statements) or without
    CUDA; else ``(RA, "auto")`` with ``RA`` the largest multiple of 128 (>= 128) whose pass working set ``2*RA*N*C*elt`` (the resident A block + the
    output row block; the host row mirror is host bytes) fits :data:`ABLOCK_BLOCK_FRAC` of them. A pure function of its arguments: every rank derives
    the same cap from the same agreed value (:func:`opt_core.mem.rowpair.dist.agreed_free_bytes`, the minimum over ranks)."""
    Rmax, N, C, elt = max(1, int(Rmax)), int(N), int(C), int(elt_bytes)
    if free_bytes is None:
        return Rmax, "whole"
    row = N * C * elt                                                       # bytes of one A row
    if Rmax * row <= ABLOCK_WHOLE_FRAC * int(free_bytes):
        return Rmax, "whole"
    cap = (int(ABLOCK_BLOCK_FRAC * int(free_bytes)) // (2 * row)) // 128 * 128
    cap = max(128, cap)
    return (Rmax, "whole") if cap >= Rmax else (cap, "auto")


def ablock_rows(Rmax: int, RA: Optional[int] = None, *, N: Optional[int] = None, C: Optional[int] = None, elt_bytes: int = 2,
                free_bytes: Optional[int] = None) -> Tuple[int, int, str]:
    """``(RA_eff, passes, source)`` of the update's A-block schedule. The row cap: the argument ``RA`` (source ``fixed``) > the diagnostic override
    ``ROWPAIR_TRIMUL_ROWS_A`` (``env``) > the PLANNER :func:`ablock_auto_cap` from ``N``, ``C``, ``elt_bytes`` and the agreed free device bytes
    ``free_bytes`` (``auto`` when it row-blocks, ``whole`` when the plane fits or no ``N`` / no CUDA is given); ``ROWPAIR_TRIMUL_ABLOCK=0`` forces
    ``Rmax`` whatever the others say (source ``off``). ``passes =
    ceil(Rmax / cap)``; the passes are BALANCED: ``RA_eff = ceil(Rmax / passes)`` rounded up to a multiple of 16 when that stays within the cap (so
    no pass is a sliver of rows: 8832 rows under a 1536 cap run 6 x 1472, not 5 x 1536 + 592), ``passes = ceil(Rmax / RA_eff)`` unchanged. A cap
    >= ``Rmax`` is one pass of ``Rmax`` rows (today's whole plane). A pure function of ``Rmax`` and the words: identical on every rank."""
    Rmax = max(1, int(Rmax))
    if os.environ.get(ENV_ABLOCK, "").strip().lower() in ("0", "off", "false", "no"):
        return Rmax, 1, "off"
    env = os.environ.get("ROWPAIR_TRIMUL_ROWS_A", "").strip()
    if RA is not None and int(RA) > 0:
        cap, src = int(RA), "fixed"
    elif env:
        try:
            cap = int(env)
        except ValueError:
            raise RowpairRefused(f"trimul_update_: ROWPAIR_TRIMUL_ROWS_A={env!r}: a row count is required") from None
        if cap < 1:
            raise RowpairRefused(f"trimul_update_: ROWPAIR_TRIMUL_ROWS_A={env!r}: must be >= 1")
        src = "env"
    elif N is not None and C is not None:
        cap, src = ablock_auto_cap(Rmax, int(N), int(C), int(elt_bytes), free_bytes)
    else:
        cap, src = Rmax, "whole"
    cap = min(cap, Rmax)
    npass = -(-Rmax // cap)
    if npass == 1:
        return Rmax, 1, src
    ra = -(-Rmax // npass)
    ra16 = -(-ra // 16) * 16
    ra = ra16 if ra16 <= cap else ra
    return ra, -(-Rmax // ra), src


def require_ablock_agreement(layout: Layout, ra: int, passes: int, source: str, gather: Optional[Callable[[object], list]] = None) -> None:
    """Refuse by name BEFORE any collective when the ranks' A-block schedules differ (``refused: trimul_ablock_ranks_differ: rank <r> ra=<..>
    passes=<..>; ...``): a per-rank ``RA`` (e.g. a rank-local row count handed to ``RA=``) would give the ranks different pass counts, hence
    different collective schedules — a hang, not an error, without this word. One object all-gather of ``(rank, ra, passes, source)``;
    nothing without a multi-rank group. ``gather``: the all-gather to use (tests), default the family's communicator."""
    if gather is None:
        from .dist import comm as _comm, is_dist
        if not is_dist():
            return
        c = _comm()
        if not getattr(c, "active", False) or int(getattr(c, "P", 1)) <= 1:
            return
        gather = c.allgather_obj
    allv = gather((int(layout.rank), int(ra), int(passes), str(source)))
    if len({(int(v[1]), int(v[2])) for v in allv}) > 1:
        detail = "; ".join(f"rank {int(v[0])} ra={int(v[1])} passes={int(v[2])} src={v[3]}" for v in sorted(allv))
        raise RowpairRefused(f"refused: trimul_ablock_ranks_differ: {detail} (RA must be ONE value on every rank: ROWPAIR_TRIMUL_ROWS_A or the same RA= "
                             f"argument; the schedule derives the passes from layout.Rmax)")


def describe_ablock() -> Dict[str, object]:
    """A copy of :data:`ABLOCK` plus ``words``, the census fragment a kit appends to its TRIMUL line: ``ablock=on ablock_ra=<RA>
    ablock_passes=<n> ablock_src=<fixed|env|whole|off> ablock_calls=<multi-pass calls>/<calls> ablock_host=<none|host_pinned|host_pageable:<kind>|host>
    ablock_host_gib=<g> ablock_chunks=<n>`` (``ablock=idle`` before any call)."""
    d = dict(ABLOCK)
    if not d["calls"]:
        d["words"] = "ablock=idle"
    else:
        d["words"] = (f"ablock=on ablock_ra={d['ra']} ablock_passes={d['passes']} ablock_src={d['source']} ablock_calls={d['multipass_calls']}/{d['calls']} "
                      f"ablock_host={d['host']} ablock_host_gib={d['host_gib']} ablock_chunks={d['chunks']}"
                      + (f" ablock_host_slab={d['host_slab']}" if "host_slab" in d else ""))   # ROWPAIR_HOST_SLAB=lease only (lease | second)
    return d


def _say(log: Optional[Callable[[str], None]], text: str) -> None:
    """A schedule / placement line at FIRST USE: through the caller's ``log`` when given, else the package's stderr line (``[rowpair] ...``,
    :func:`opt_core.report.emit`) — a schedule word is visible when it takes effect, not only in the exit census."""
    if log is not None:
        log(text)
        return
    from ...report import emit
    emit("[rowpair] " + text)


def host_mirror(z_shard, layout: Layout, log: Optional[Callable[[str], None]] = None):
    """The host ROW MIRROR of the multi-pass update: an exact-size chunked host copy (:func:`.trunk.host_alloc` from the family's park pool
    :func:`.trunk.park_pool` — page-locked ``ROWPAIR_PARK_CHUNK_GIB`` chunks for a CUDA shard, ``ROWPAIR_PARK_PIN_MAX_GB`` / ``ROWPAIR_PARK_STRICT``
    honoured, a refused page-lock a NAMED pageable buffer) of at least ``z_shard``'s bytes, allocated ONCE per process (and rank) and re-used by
    every later call (grown when a larger shard comes; a smaller shard uses a prefix). Returns the :class:`opt_core.mem.torch_hostpair.HostChunks`.
    ``ROWPAIR_HOST_SLAB=lease``: the mirror is instead a per-call LEASE of this rank's one host slab (:class:`.trunk.HostSlabLease`;
    a :class:`.trunk.LeasedHost` is returned and :func:`release_leased_mirror` hands it back at the call's copy-back) — or, while a z park holds the
    slab, a NAMED private buffer (census ``host_slab_second``, ``trimul_ablock_host_slab=second``) released likewise at the call's end."""
    from .trunk import host_alloc, host_slab_lease, host_slab_mode, park_pool
    key = (str(z_shard.device), int(layout.rank), int(layout.P))
    nbytes = int(z_shard.numel()) * int(z_shard.element_size())
    if host_slab_mode() == "lease":                                              # ROWPAIR_HOST_SLAB=lease: the mirror is a LEASE of this
        lease = host_slab_lease(z_shard.device, pin=bool(z_shard.is_cuda))       # rank's one host slab for the duration of ONE multi-pass call (released
        hc, where, _fallback, _pinned = host_alloc(lease, z_shard, "trimul row mirror", lambda m: _say(log, m))   # at the call's copy-back:
        leased = bool(getattr(hc, "leased", False))                              # release_leased_mirror); a busy slab (a z park holds it) answers
        with _MIRROR_LOCK:                                                       # with a NAMED private buffer
            said = ("lease" if leased else "second") + ":" + str(where)
            first = _LEASED_SAID.get(key) != said
            _LEASED_SAID[key] = said
        ABLOCK.update(host=where, host_gib=round(nbytes / 2 ** 30, 3), chunks=hc.n_chunks, pinned_gib=round(hc.pinned_bytes / 2 ** 30, 3),
                      host_slab=("lease" if leased else "second"))
        record_schedule(trimul_ablock_host=where, trimul_ablock_host_gib=round(nbytes / 2 ** 30, 3), trimul_ablock_chunks=hc.n_chunks,
                        trimul_ablock_pinned_gib=round(hc.pinned_bytes / 2 ** 30, 3), trimul_ablock_host_slab=("lease" if leased else "second"))
        if first:
            _say(log, f"[trimul] row mirror {tuple(z_shard.shape)} {nbytes / 2 ** 30:.2f} GiB -> {where} chunks={hc.n_chunks} pinned_gib={hc.pinned_bytes / 2 ** 30:.2f} "
                      f"({'LEASED host slab, per call' if leased else 'PRIVATE buffer: the host slab is held by ' + str(lease.holder)})")
        return hc
    with _MIRROR_LOCK:
        hc = _MIRRORS.get(key)
        if hc is not None and hc.nbytes >= nbytes and not hc.released:
            return hc
        if hc is not None:
            hc.release()
        pool = park_pool(pin=bool(z_shard.is_cuda))
        hc, where, _fallback, _pinned = host_alloc(pool, z_shard, "trimul row mirror", log)
        _MIRRORS[key] = hc
        ABLOCK.update(host=where, host_gib=round(nbytes / 2 ** 30, 3), chunks=hc.n_chunks, pinned_gib=round(hc.pinned_bytes / 2 ** 30, 3))
        record_schedule(trimul_ablock_host=where, trimul_ablock_host_gib=round(nbytes / 2 ** 30, 3), trimul_ablock_chunks=hc.n_chunks,
                        trimul_ablock_pinned_gib=round(hc.pinned_bytes / 2 ** 30, 3))
        _say(log, f"[trimul] row mirror {tuple(z_shard.shape)} {nbytes / 2 ** 30:.2f} GiB -> {where} chunks={hc.n_chunks} pinned_gib={hc.pinned_bytes / 2 ** 30:.2f}")
        return hc


def host_mirror_bytes() -> Tuple[int, int, int]:
    """``(mirrors, chunks, pinned bytes)`` of the row mirrors this process holds right now (the pinned-pool shrink line quotes it before releasing)."""
    with _MIRROR_LOCK:
        live = [hc for hc in _MIRRORS.values() if not hc.released]
        return len(live), sum(int(hc.n_chunks) for hc in live), sum(int(hc.pinned_bytes if hc.pinned else hc.alloc_bytes) for hc in live)


def release_leased_mirror(mirror) -> bool:
    """End of ONE multi-pass call under ``ROWPAIR_HOST_SLAB=lease``: hand the leased slab back (:meth:`.trunk.LeasedHost.release` records the
    hand-over event the next holder waits on) — or return a second-holder private buffer's chunks to torch's cache
    (:meth:`opt_core.mem.torch_hostpair.HostChunks.release`; torch's caching host allocator keeps a pinned block out of reuse until the copies it
    recorded on it completed). Returns True when something was released. A process-held mirror (the default placement) is left alone."""
    if mirror is None or getattr(mirror, "released", True):
        return False
    with _MIRROR_LOCK:
        held = any(mirror is hc for hc in _MIRRORS.values())
    if held:
        return False
    mirror.release()
    return True


def release_host_mirror() -> int:
    """Return every row mirror this process holds to its pool (the chunks go back to torch's caching host allocator); returns how many."""
    with _MIRROR_LOCK:
        n = 0
        for hc in _MIRRORS.values():
            hc.release()
            n += 1
        _MIRRORS.clear()
        return n


def pass_windows(layout: Layout, grid: List[List[Tuple[int, int]]], t: int, RA: int, p: int) -> List[Tuple[int, int]]:
    """:func:`grid_windows` of sub-block ``t`` restricted to pass ``p`` of an ``RA``-row A-block schedule: rank q's window is its WHOLE sub-block t
    when that sub-block intersects q's local rows ``[p*RA, (p+1)*RA)``, else ``(0, 0)``. One pass (``RA >= Rmax``) is :func:`grid_windows` itself."""
    out = []
    for q, g in enumerate(grid):
        if t < len(g):
            l0, l1 = g[t]
            i0, i1 = p * RA, (p + 1) * RA
            if min(l1, i1) > max(l0, i0):
                out.append((layout.bounds[q][0] + l0, layout.bounds[q][0] + l1))
                continue
        out.append((0, 0))
    return out


def trimul_update_(fns: TriMulFns, z_shard, mask_shard, layout: Layout, *, outgoing: bool, add: bool = True, RB: Optional[int] = None,
                   grid: Optional[str] = None, inplace_chunk: Optional[int] = None, mm_rows="auto", a2a_chunks: Optional[int] = None,
                   RA: Optional[int] = None, stats: Optional[ContractStats] = None, log: Optional[Callable[[str], None]] = None):
    """The triangle-multiplication update of this rank's pair rows IN PLACE at the streamed schedule (module docstring): ``z_shard[i, j] (+)=
    gate(z[i, j]) * out(sum_k a b)[i, j]`` for local rows i and all columns j — outgoing ``sum_k a_ik b_jk`` / incoming ``sum_k a_ki b_kj``.
    ``z_shard`` ``[R, N, C_z]`` contiguous (rows ``r0:r1``; a kit's ``[1, R, N, C]`` shard passes ``z[0]``); ``mask_shard`` ``[R, N]`` or None
    (ones). Returns ``z_shard`` (== rows ``r0:r1`` of the dense in-place update). ``add=False`` writes the update instead of adding it.

    outgoing: ``A[C_h, R, N]`` from my ORIGINAL rows, projected on the GLOBAL row grid ``range(0, N, inplace_chunk)`` clipped to my rows
    (:meth:`Layout.chunks`); own b sub-blocks projected just in time from ORIGINAL rows; deferral of pending rows (module docstring).
    incoming: ``a[c,i,k] = proj_a(z[k,i]) m[k,i]`` and ``b[c,k,j] = proj_b(z[k,j]) m[k,j]`` are functions of z^T rows: A is built band by band from
    :func:`zT_band` (the z^T rows of my own sub-blocks, one banded exchange per sub-block index, z^T never whole), each pass re-fetches the band of
    its sub-block index and projects the own b sub-block from it; the mask travels once as rows of mask^T; tiles land in ROW layout directly
    (no transpose back) and nothing is deferred (every column block is written exactly at its own sub-block index; bands and gates read other
    columns). Per call: 2 banded transposes of the shard + 1 of the mask + the ring.

    Schedule knobs (identical on every rank by construction; printed in the schedule census): ``RB`` rows per b sub-block (``ROWPAIR_TRIMUL_SUB``
    > ``RB`` > :func:`default_sub`); ``grid`` (``ROWPAIR_TRIMUL_GRID``: ``stock`` (default) = sub-blocks also break at :func:`inplace_chunk_bounds`
    of ``inplace_chunk`` so the einsum column extents are the in-place statement's, ``rank`` = uniform sub-blocks from each rank start);
    ``mm_rows`` (:func:`mm_row_pieces`, default ``auto``: M pieces only when an A piece would reach 2**31 elements); ``inplace_chunk`` is
    REQUIRED (the engine's in-place statement's column chunk — an engine value the adapter passes; refused by name when absent);
    ``a2a_chunks`` (staging groups of the mask^T all-to-all, incoming); ``RA`` — the A-BLOCK schedule (:func:`ablock_rows`: ``RA`` >
    ``ROWPAIR_TRIMUL_ROWS_A`` > whole; ``ROWPAIR_TRIMUL_ABLOCK=0`` forces whole): A is resident ``RA`` rows at a time and the contraction runs
    ``passes = ceil(Rmax / RA)`` passes (the ring — and, incoming, the band exchanges — once per pass). ONE pass (the default: ``RA >= Rmax``) is
    exactly the statements above (whole A from original rows, in-place epilogue, outgoing deferral). MORE than one pass: ``z_shard`` is READ-ONLY
    for the whole call (own b sub-blocks, the peers' band fetches and the gates read the original device shard in every pass; nothing is
    deferred); pass p projects A for local rows ``[p*RA, (p+1)*RA)`` just in time (outgoing: the same global row grid clipped to the block;
    incoming: the bands of the own sub-blocks that intersect the block — :func:`pass_windows`, whole sub-blocks; a chunk / sub-block that
    straddles the block is projected whole and the block's rows taken, so every projection launch has its one-pass extents); the epilogue runs
    the SAME statements on the pass's OUTPUT ROW BLOCK ``[RA, N, C_z]`` (each tile's block pre-filled with z's original values, so the fused
    epilogue / gate / ``+=`` read and write what they read and write in one pass); after the pass's last tile the block is copied to the host
    ROW MIRROR (:func:`host_mirror`: u bytes per rank, page-locked chunks, allocated once per process); after the last pass ONE copy mirror ->
    ``z_shard``. Per element the arithmetic is one pass's (bitwise, given the tile-extent invariance of the module docstring); device memory
    ``u + 2*RA*N*C`` + transients instead of ``2u``; traffic per call: ring x passes, host D2H u + H2D u. Census: ``trimul_ablock_ra`` /
    ``trimul_ablock_passes`` / ``trimul_ablock_source`` (+ ``trimul_ablock_host*`` once the mirror exists); :data:`ABLOCK` / :func:`describe_ablock`
    for a kit's TRIMUL line. ``log`` receives the mirror's placement line. Memory (one pass): resident z + A (``R*N*C_h``); transients one
    sub-block ``RB*N*C_h`` x 3 (own + ring pair) + one band ``RB*N*C_z`` (incoming) + tile/epilogue ``4 * rows*RB*C``; outgoing deferral <= ~A/4
    (:func:`deferred_peak_bytes`, reported in ``stats``)."""
    require_sharded(layout, "trimul_update_")
    if inplace_chunk is None or int(inplace_chunk) < 1:
        raise RowpairRefused("trimul_update_: inplace_chunk (the engine's in-place triangle-multiplication column chunk) is required")
    R, N, C_z = int(z_shard.shape[0]), int(z_shard.shape[1]), int(z_shard.shape[2])
    if R != layout.R or N != layout.N or z_shard.dim() != 3:
        raise RowpairRefused(f"trimul_update_: z_shard {tuple(z_shard.shape)} vs layout R={layout.R} N={layout.N} ([R, N, C] expected)")
    if not z_shard.is_contiguous():
        raise RowpairRefused("trimul_update_: z_shard must be contiguous (it is written in place)")
    if mask_shard is None:
        mask_shard = z_shard.new_ones((R, N))
    if tuple(mask_shard.shape) != (R, N):
        raise RowpairRefused(f"trimul_update_: mask_shard {tuple(mask_shard.shape)} vs (R, N)=({R}, {N})")
    mask_u = mask_shard.unsqueeze(-1)                                       # [R, N, 1]
    C_h = int(fns.C_h)
    proj_into = getattr(fns, "proj_into", None)                             # optional fused hooks (:mod:`opt_core.mem.rowpair.trimul_fused`): a projection written
    tile_epilogue = getattr(fns, "tile_epilogue", None)                     #   straight into its GEMM block / the tile epilogue in one launch; absent (every plain
    od = getattr(fns, "operand_dtype", None)                                #   TriMulFns) or returning False = the statements below. operand_dtype(z_shard): the GEMM / ring
    b_t = getattr(fns, "b_transposed", None)                                #   dtype; b_transposed(z_shard) -> bool: hand proj_into a b sub-block plane in the GEMM-B layout (_b_plane)
    served_dtype = od(z_shard) if callable(od) else od                      #   dtype a provider SERVES this shard with (bf16 planes for an fp32 shard); absent / declined = z's
    dtype = served_dtype or z_shard.dtype
    device = z_shard.device                                                 #   dtype, i.e. the statements' own
    RB_, sub_src = _sub_rows(N, RB)
    col_bounds, grid_word = grid_bounds(N, grid, inplace_chunk)
    st = stats or ContractStats()
    record_schedule(trimul_sub_source=sub_src, trimul_grid_word=grid_word, trimul_inplace_chunk=int(inplace_chunk))

    C_row, elt_row = int(z_shard.shape[-1]), int(torch.empty((), dtype=dtype).element_size())
    akey = (str(device), int(N), int(layout.Rmax), C_row, elt_row)                # per shape: the template pair stack (C=64) and the trunk pair stack (C=128) plan separately
    if akey not in _AUTO_FREE:                                                # the planner's probe: ONE agreed free-bytes value per shape and process (every rank reaches
        _AUTO_FREE[akey] = agreed_free_bytes(device) if z_shard.is_cuda else None   # this first update of the shape together; the min over ranks), so the schedule is one
    RA_, npass, ra_src = ablock_rows(layout.Rmax, RA, N=N, C=C_row, elt_bytes=elt_row, free_bytes=_AUTO_FREE[akey])   # value for the whole run, identical on every rank
    if ra_src in ("fixed", "env", "auto"):
        require_ablock_agreement(layout, RA_, npass, ra_src)                 # a per-rank RA is a named refusal before any collective, not a diverged schedule
    multipass = npass > 1
    if multipass and not int(ABLOCK["multipass_calls"]):                      # the first multi-pass call of the process: the schedule word at first use
        _say(log, f"[trimul] ablock schedule ra={RA_} passes={npass} src={ra_src} free_gib={(_AUTO_FREE.get(akey) or 0) / 2 ** 30:.1f} rank={layout.rank} R={layout.R} Rmax={layout.Rmax} N={int(z_shard.shape[-2])} "
                  f"({'outgoing' if outgoing else 'incoming'}; z read-only, output row block + host row mirror)")
    record_schedule(trimul_ablock_ra=RA_, trimul_ablock_passes=npass, trimul_ablock_source=ra_src, trimul_ablock_free_gib=round((_AUTO_FREE.get(akey) or 0) / 2 ** 30, 1))
    ABLOCK.update(calls=int(ABLOCK["calls"]) + 1, multipass_calls=int(ABLOCK["multipass_calls"]) + int(multipass), ra=RA_, passes=npass, source=ra_src)
    r0 = layout.r0
    elt_z = int(z_shard.element_size())
    mirror = host_mirror(z_shard, layout, log) if multipass else None         # u host bytes, once per process; z_shard stays ORIGINAL through the passes
    outblk = torch.empty((min(RA_, R), N, C_z), dtype=z_shard.dtype, device=device) if multipass else None   # the pass's output rows
    cur = {"o0": 0}                                                          # local row offset of the pass outblk holds

    def zdst(i0, i1, c0, c1):                                               # the block the epilogue reads (original values) and writes: z itself in
        blk = z_shard[i0:i1, c0:c1, :]                                      #   one pass; the pass's output block, pre-filled with z's original block,
        if not multipass:                                                   #   in a multi-pass call (z is read-only until the final copy back)
            return blk
        dst = outblk[i0 - cur["o0"]:i1 - cur["o0"], c0:c1, :]
        dst.copy_(blk)
        return dst

    def epilogue(T, rows, cols):                                            # T [C_h, rows, cols] -> x [rows, cols, C_z]; z[rows, cols] (+)= g * x
        i0, i1 = rows
        c0, c1 = cols
        zb = zdst(i0, i1, c0, c1)
        if tile_epilogue is not None:
            if tile_epilogue(T, zb, add):
                return
            if served_dtype is not None and T.dtype != z_shard.dtype:        # a declined tile of a SERVED-dtype schedule: the statements take z's dtype (a declined shard keeps the plain path's T as is)
                T = T.to(z_shard.dtype)
        x = fns.out(T.permute(1, 2, 0))
        if fns.gate is not None:
            g = fns.gate(zb)                                                # ORIGINAL values: this block is written exactly once, right below
            x = x.mul_(g) if x.is_contiguous() else x * g
            del g
        if add:
            zb += x
        else:
            zb.copy_(x)
        del x

    def pass_hook(p, i0, i1):                                               # multi-pass: the pass's finished rows -> the host mirror (stream-ordered)
        if multipass and i1 > i0:
            mirror.store_range(outblk[: i1 - i0], i0 * N * C_z * elt_z, non_blocking=True)
            ABLOCK["d2h_bytes"] = int(ABLOCK["d2h_bytes"]) + (i1 - i0) * N * C_z * elt_z

    def proj_rows(dst, l0, l1, i0, i1, zsrc, msrc, m_u):                   # A rows max(l0,i0):min(l1,i1) of the unit l0:l1 (a chunk / a band) into dst
        k0, k1 = max(l0, i0), min(l1, i1)                                   #   [C_h, i1-i0, N]; a unit that straddles the block is projected WHOLE
        if k1 <= k0:                                                        #   (one pass's extents) and its block rows taken
            return
        whole = (k0, k1) == (l0, l1)
        tgt = dst[:, k0 - i0:k1 - i0, :] if whole else torch.empty((C_h, l1 - l0, N), dtype=dtype, device=device)
        if proj_into is None or not proj_into(tgt, zsrc, msrc, True):
            tgt[...] = fns.proj(zsrc, m_u, True).permute(2, 0, 1)
        if not whole:
            dst[:, k0 - i0:k1 - i0, :] = tgt[:, k0 - l0:k1 - l0, :]
        del tgt

    if outgoing:
        def a_provider(i0, i1):                                             # A [C_h, i1-i0, N] of my ORIGINAL rows i0:i1 on the in-place statement's
            A = torch.empty((C_h, i1 - i0, N), dtype=dtype, device=device)  #   GLOBAL row grid restricted to my rows (every element written: no zero-fill)
            for g0, g1 in layout.chunks(max(1, int(inplace_chunk))):
                l0, l1 = g0 - r0, g1 - r0
                proj_rows(A, l0, l1, i0, i1, z_shard[l0:l1], mask_shard[l0:l1], mask_u[l0:l1])
            return A

        def b_provider(j0, j1):                                             # own sub-block from ORIGINAL rows j0:j1 -> [C_h, N, w] (GEMM layout)
            if proj_into is not None:
                plane = _b_plane(b_t, z_shard, C_h, j1 - j0, N, dtype, device)
                if proj_into(plane, z_shard[j0:j1], mask_shard[j0:j1], False):
                    return plane.transpose(1, 2)                            # [C_h, N, w]: a view of the [C_h, w, N] plane (made contiguous by _b_slab), or — b_transposed — the contiguous buffer itself
                b = gemm_b_layout(fns.proj(z_shard[j0:j1], mask_u[j0:j1], False))
                return b.to(dtype) if served_dtype is not None else b          # a declined shard: exactly the plain path's operand
            return gemm_b_layout(fns.proj(z_shard[j0:j1], mask_u[j0:j1], False))

        def pass_hook_out(p, i0, i1):
            pass_hook(p, i0, i1)
            cur["o0"] = (p + 1) * RA_

        rowshard_outer_contract(a_provider, b_provider, layout, epilogue, N=N, C=C_h, dtype=dtype, device=device, RA=RA_,
                                RB=RB_, stats=st, col_bounds=col_bounds, mm_rows=mm_rows, defer_pending_rows=not multipass, pass_hook=pass_hook_out)
    else:
        grid_ = slab_grid(layout, RB_, col_bounds)
        own = grid_[layout.rank]
        T_sub = max(len(g) for g in grid_)
        maskT_u = transpose_shards(mask_u.contiguous(), layout, chunks=a2a_chunks)      # [R, N, 1]: my rows of mask^T
        held = {}

        def a_provider(i0, i1):                                             # A [C_h, i1-i0, N] band by band: the z^T rows of my own sub-blocks that
            A = torch.empty((C_h, i1 - i0, N), dtype=dtype, device=device)  #   intersect rows i0:i1 (EVERY rank takes part in each exchange)
            p = i0 // RA_
            for t in range(T_sub):
                wins = pass_windows(layout, grid_, t, RA_, p)
                if all(w1 <= w0 for w0, w1 in wins):
                    continue
                band = zT_band(z_shard, layout, wins)                       # [w, N, C_z] z^T rows of my own sub-block t (None: not mine this pass)
                if band is not None:
                    l0, l1 = own[t]
                    proj_rows(A, l0, l1, i0, i1, band, maskT_u[l0:l1, :, 0], maskT_u[l0:l1])
                del band
            return A

        def a_provider_ranked(i0, i1):                                      # a rank with no rows in this pass still serves the peers' bands
            return a_provider(i0, i1) if i1 > i0 else None

        def slab_hook(jb):                                                  # every rank, every sub-block index: the band of pass jb
            if jb == 0 and cur.get("pending_pass") is not None:             # a rank without rows in this pass: its share of the A-band exchanges
                p = cur.pop("pending_pass")
                for t in range(T_sub):
                    wins = pass_windows(layout, grid_, t, RA_, p)
                    if all(w1 <= w0 for w0, w1 in wins):
                        continue
                    band = zT_band(z_shard, layout, wins)
                    del band
            held["band"] = zT_band(z_shard, layout, grid_windows(layout, grid_, jb))

        def b_provider(j0, j1):
            band = held.pop("band")
            if proj_into is not None:
                plane = _b_plane(b_t, z_shard, C_h, j1 - j0, N, dtype, device)
                if proj_into(plane, band, maskT_u[j0:j1, :, 0], False):
                    return plane.transpose(1, 2)
                b = gemm_b_layout(fns.proj(band, maskT_u[j0:j1], False))
                return b.to(dtype) if served_dtype is not None else b
            return gemm_b_layout(fns.proj(band, maskT_u[j0:j1], False))

        def pass_hook_in(p, i0, i1):
            pass_hook(p, i0, i1)
            cur["o0"] = (p + 1) * RA_
            if p + 1 < npass and min(R, (p + 2) * RA_) <= min(R, (p + 1) * RA_):   # no rows of mine in the next pass: serve its A bands from slab_hook(0)
                cur["pending_pass"] = p + 1

        if min(R, RA_) <= 0:
            cur["pending_pass"] = 0
        rowshard_outer_contract(a_provider_ranked, b_provider, layout, epilogue, N=N, C=C_h, dtype=dtype, device=device, RA=RA_,
                                RB=RB_, stats=st, col_bounds=col_bounds, mm_rows=mm_rows, defer_pending_rows=False, slab_hook=slab_hook,
                                pass_hook=pass_hook_in)
        held.clear()
        del maskT_u
    if multipass:
        del outblk
        mirror.load_range(z_shard, 0, non_blocking=True)                    # the finished update -> z (stream-ordered after the last pass's copy out)
        ABLOCK["h2d_bytes"] = int(ABLOCK["h2d_bytes"]) + R * N * C_z * elt_z
        release_leased_mirror(mirror)                                        # ROWPAIR_HOST_SLAB=lease: the slab goes back per call (a no-op for the process-held mirror)
    st.mode = ("update:outgoing:" if outgoing else "update:incoming:") + (st.mode or "")
    return z_shard
