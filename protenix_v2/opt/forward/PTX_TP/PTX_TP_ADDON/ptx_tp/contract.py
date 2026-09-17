"""
ptx_tp.contract -- distributed contractions over a
row-sharded pair tensor (tensors in / tensors out; no model imports).

Two contraction patterns cover every "pairwise over a shared axis" op:

(a) ROW-SHARD OUTER contraction  (TriMul "outgoing":  x_ij = sum_k a_ik * b_jk)
        a[i,k,c], b[j,k,c] are ROW-LOCAL functions of X rows (i resp. j);
        rank(I) holds a rows I and b rows I; b row-blocks travel the ring;
        tile T[c, I_sub, J_sub] = A[c, I_sub, :] @ B[c, :, J_sub]   (FULL k in
        ONE batched matmul, batch = channels) -> epilogue_fn(T, rows, cols).

(b) COL-SHARD INNER contraction  (TriMul "incoming":  x_ij = sum_k a_ki * b_kj)
        a[k,i,c], b[k,j,c] are row-local functions of X rows k (sharded);
        an all-to-all moves a[:, I_sub] -> rank(I) as A[c, I_sub, k] and
        b[:, J_slab] -> rank(J) as B[c, k, J_slab]; then (a)'s ring/tile loop
        runs unchanged and tiles land directly in ROW layout on rank(I).

P-INVARIANCE RULE: a kernel's shape / element coverage may depend on P only
through the row count of row-local GEMMs (M) and the tile extents; every
output element's full k-contraction happens inside ONE matmul on ONE rank;
there is no split-K across ranks and no partial-sum all-reduce anywhere.

INT32 RULE: keep every elementwise/LayerNorm launch below 2**31 elements
(many fused kernels index with int32).  `max_rows_per_launch(N, C)` gives the
row budget for [rows, N, C] blocks; providers/epilogues must sub-chunk with
it.  Batched matmuls may exceed 2**31 total elements (64-bit strides) but each
of m, n, k must be < 2**31 and we additionally assert tile elements < 2**31.

GEMM operand layouts (chosen to reproduce the cuBLAS call of the stock
single-device torch statement `torch.matmul(a[c,i,k], b[c,k,j])`):
    A block : [C, rows_i, N]  contiguous   (gemm_a)
    B block : [C, N, rows_j]  contiguous   (gemm_b)
    T tile  : [C, rows_i, rows_j] = torch.matmul(A, B)
"""
from __future__ import annotations

import math
import os
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

from .dist import Layout, all_gather_rows, alltoall_window, is_dist, ring_blocks

__all__ = [
    "gemm_a_layout", "gemm_b_layout", "max_rows_per_launch",
    "default_rows_a", "default_rows_b",
    "rowshard_outer_contract", "colshard_inner_contract", "rowsplit_attention",
    "ContractStats",
]

INT32_MAX = 2 ** 31 - 1


# --------------------------------------------------------------------------
# layouts / budgets
# --------------------------------------------------------------------------
def gemm_a_layout(x_rows: torch.Tensor) -> torch.Tensor:
    """[rows, N, C] (channel-last row block) -> A block [C, rows, N] contiguous."""
    return x_rows.permute(2, 0, 1).contiguous()


def gemm_b_layout(x_rows: torch.Tensor) -> torch.Tensor:
    """[rows, N, C] (channel-last row block) -> B block [C, N, rows] contiguous."""
    return x_rows.permute(2, 1, 0).contiguous()


def max_rows_per_launch(N: int, C: int) -> int:
    """Largest row count r such that an [r, N, C] elementwise launch has
    < 2**31 elements (env PTX_TP_ELEM_ROWS overrides; result >= 1)."""
    env = os.environ.get("PTX_TP_ELEM_ROWS")
    if env:
        return max(1, int(env))
    return max(1, INT32_MAX // (int(N) * int(C)))


def _gb(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def default_rows_a(N: int, C: int, elt_bytes: int = 2, Rmax: Optional[int] = None) -> int:
    """RA = rows of the resident A block per pass.  env PTX_TP_TRIMUL_ROWS_A, else
    the largest multiple of 128 with RA*N*C*elt <= PTX_TP_TRIMUL_ABUF_GB (24 GB)."""
    env = os.environ.get("PTX_TP_TRIMUL_ROWS_A")
    if env:
        ra = int(env)
    else:
        ra = int(_gb("PTX_TP_TRIMUL_ABUF_GB", 24.0) * 1e9 // (N * C * elt_bytes))
        ra = max(128, (ra // 128) * 128)
    if Rmax is not None:
        ra = min(ra, Rmax)
    return max(1, ra)


def default_rows_b(N: int, C: int, elt_bytes: int = 2, Rmax: Optional[int] = None) -> int:
    """RB = rows per streamed B slab.  env PTX_TP_TRIMUL_ROWS_B, else the largest
    multiple of 128 with 2*RB*N*C*elt <= PTX_TP_TRIMUL_BBUF_GB (16 GB)."""
    env = os.environ.get("PTX_TP_TRIMUL_ROWS_B")
    if env:
        rb = int(env)
    else:
        rb = int(_gb("PTX_TP_TRIMUL_BBUF_GB", 16.0) * 1e9 // (2 * N * C * elt_bytes))
        rb = max(128, (rb // 128) * 128)
    if Rmax is not None:
        rb = min(rb, Rmax)
    return max(1, rb)


class ContractStats:
    """Lightweight accounting filled by the contractions (for logging)."""

    def __init__(self):
        self.passes = 0
        self.slabs = 0
        self.ring_steps = 0
        self.tiles = 0
        self.matmul_flops = 0.0
        self.bytes_streamed = 0
        self.mode = ""

    def __repr__(self):
        return (f"ContractStats(mode={self.mode!r}, passes={self.passes}, slabs={self.slabs}, ring_steps={self.ring_steps}, "
                f"tiles={self.tiles}, matmul_TFLOP={self.matmul_flops / 1e12:.3f}, GB_streamed={self.bytes_streamed / 1e9:.3f})")


BSource = Union[torch.Tensor, Sequence[torch.Tensor], Callable[[int, int], torch.Tensor]]


def _b_slab(b_source: BSource, jb: int, j0: int, j1: int, C: int, N: int, dtype, device) -> torch.Tensor:
    """This rank's B slab [C, N, j1-j0] (contiguous) for local rows j0:j1."""
    if j1 <= j0:
        return torch.empty((C, N, 0), dtype=dtype, device=device)
    if callable(b_source):
        s = b_source(j0, j1)
    elif torch.is_tensor(b_source):          # whole b in gemm_b layout [C, N, R]
        s = b_source[:, :, j0:j1]
    else:                                     # list of cached slabs (gemm_b layout each)
        s = b_source[jb]
    assert s.shape == (C, N, j1 - j0), (tuple(s.shape), (C, N, j1 - j0))
    return s.contiguous()


# --------------------------------------------------------------------------
# (a) row-shard outer contraction
# --------------------------------------------------------------------------
def rowshard_outer_contract(
    a_provider: Callable[[int, int], Optional[torch.Tensor]],
    b_source: BSource,
    layout: Layout,
    epilogue_fn: Callable[[torch.Tensor, Tuple[int, int], Tuple[int, int]], None],
    *,
    N: int,
    C: int,
    dtype: torch.dtype,
    device,
    RA: Optional[int] = None,
    RB: Optional[int] = None,
    stats: Optional[ContractStats] = None,
) -> ContractStats:
    """x[I, J] tiles of  T[c,i,j] = sum_k A[c,i,k] * B[c,k,j]  for this rank's
    rows I (all local row sub-blocks of RA rows) against EVERY rank's b rows J
    (streamed around the ring in slabs of RB rows).

    a_provider(i0, i1) -> A block [C, i1-i0, N] contiguous for LOCAL rows i0:i1
        (called once per pass; may return None when i1 <= i0).
    b_source: this rank's b in gemm_b layout [C, N, R]  |  list of slabs
        [C, N, RB_v] (jb-th slab = local rows jb*RB : ...)  |  callable
        b_provider(j0, j1) -> [C, N, j1-j0] (recomputed once per (pass, slab)).
    epilogue_fn(T, (i0, i1), (c0, c1)): T = [C, i1-i0, c1-c0] tile for LOCAL
        rows i0:i1 and GLOBAL columns c0:c1 (consumed before the next tile).
    All ranks execute the same (pass, slab, ring-step) schedule (counts derive
    from Rmax, not R), so ranks with fewer/zero rows still relay blocks.
    Memory per rank (elements of dtype): A = RA*N*C (+ provider transients),
    ring = 2*RB*N*C recv buffers (+ RB*N*C if the own slab needs padding),
    tile = C*RA*RB (+ epilogue transients), plus b_source if cached (R*N*C).
    """
    st = stats or ContractStats()
    R, Rmax, P = layout.R, layout.Rmax, layout.P
    elt = torch.empty((), dtype=dtype).element_size()
    RA = int(RA or default_rows_a(N, C, elt, Rmax))
    RB = int(RB or default_rows_b(N, C, elt, Rmax))
    npass = -(-Rmax // RA)
    nslab = -(-Rmax // RB)
    assert C * N * RB <= INT32_MAX or True  # matmul operands may exceed; tiles asserted below
    for p in range(npass):
        i0, i1 = min(R, p * RA), min(R, (p + 1) * RA)
        A = a_provider(i0, i1) if i1 > i0 else None
        if A is not None:
            assert A.shape == (C, i1 - i0, N) and A.is_contiguous(), (tuple(A.shape), (C, i1 - i0, N))
        st.passes += 1
        for jb in range(nslab):
            j0, j1 = min(R, jb * RB), min(R, (jb + 1) * RB)
            slab = _b_slab(b_source, jb, j0, j1, C, N, dtype, device)
            st.slabs += 1
            for q, Bq in ring_blocks(slab, layout, rows=(jb * RB, RB), dim=2):
                st.ring_steps += 1
                if q != layout.rank:
                    st.bytes_streamed += C * N * RB * elt
                v = Bq.shape[2]
                if A is None or v == 0:
                    continue
                assert C * (i1 - i0) * v <= INT32_MAX, ("tile too large", C, i1 - i0, v)
                T = torch.matmul(A, Bq)                       # [C, RA_v, v]  full k
                st.tiles += 1
                st.matmul_flops += 2.0 * C * (i1 - i0) * v * N
                c0 = layout.bounds[q][0] + jb * RB
                epilogue_fn(T, (i0, i1), (c0, c0 + v))
                del T
            del slab
        del A
    st.mode = st.mode or ("b:tensor" if torch.is_tensor(b_source) else ("b:provider" if callable(b_source) else "b:slabs"))
    return st


# --------------------------------------------------------------------------
# (b) col-shard inner contraction
# --------------------------------------------------------------------------
def colshard_inner_contract(
    pa_window: Callable[[int, int], torch.Tensor],
    pb_window: Callable[[int, int], torch.Tensor],
    layout: Layout,
    epilogue_fn: Callable[[torch.Tensor, Tuple[int, int], Tuple[int, int]], None],
    *,
    N: int,
    C: int,
    dtype: torch.dtype,
    device,
    RA: Optional[int] = None,
    RB: Optional[int] = None,
    cache_b: bool = True,
    b_slabs: Optional[List[torch.Tensor]] = None,
    a2a_chunks: Optional[int] = None,
    stats: Optional[ContractStats] = None,
) -> ContractStats:
    """x[I, J] tiles of  T[c,i,j] = sum_k a[k,i,c] * b[k,j,c]  where a, b are
    ROW(k)-sharded [R_k, N, C] quantities available through column-window
    providers on their k-owner:
        pa_window(c0, c1) -> a[K_mine, c0:c1, :]   ([R, c1-c0, C])
        pb_window(c0, c1) -> b[K_mine, c0:c1, :]
    (providers must read from data that epilogue_fn does not modify during
    the call -- e.g. a pre-update copy -- because other ranks' later passes
    still request windows of ALL local rows).
    Step 1 (once, or per pass if cache_b=False): B slabs = all-to-all of
        b[:, J_slab] into gemm_b layout [C, N(k), RB_v(j)] on rank(J).
    Step 2 per pass: A = all-to-all of a[:, I_sub] into gemm_a layout
        [C, RA_v(i), N(k)] on rank(I); then the ring/tile loop of (a).
    b_slabs: optionally pass precomputed slabs (list, gemm_b layout) to skip step 1.
    Memory per rank: A = RA*N*C (+ all-to-all transient (1+2/chunks)*RA*N*C),
    B cache = R*N*C if cache_b else per-slab transient (1+2/chunks)*RB*N*C,
    ring = 2*RB*N*C, tile = C*RA*RB.
    """
    st = stats or ContractStats()
    R, Rmax = layout.R, layout.Rmax
    elt = torch.empty((), dtype=dtype).element_size()
    RA = int(RA or default_rows_a(N, C, elt, Rmax))
    RB = int(RB or default_rows_b(N, C, elt, Rmax))
    nslab = -(-Rmax // RB)

    def make_slab(jb):
        return alltoall_window(pb_window, layout, jb * RB, RB, C=C, dtype=dtype, device=device,
                               out_layout="gemm_b", chunks=a2a_chunks)      # [C, N, v]

    if b_slabs is None and cache_b:
        b_slabs = [make_slab(jb) for jb in range(nslab)]
        st.mode = "colshard:b-cached"
    elif b_slabs is not None:
        st.mode = "colshard:b-given"
    else:
        st.mode = "colshard:b-per-pass"

    def a_provider(i0, i1):
        # all ranks call this the same number of times (once per pass) -> collective-safe
        return None  # placeholder (not used)

    npass = -(-Rmax // RA)
    for p in range(npass):
        A = alltoall_window(pa_window, layout, p * RA, RA, C=C, dtype=dtype, device=device,
                            out_layout="gemm_a", chunks=a2a_chunks)          # [C, RA_v, N]
        i0, i1 = min(R, p * RA), min(R, (p + 1) * RA)
        assert A.shape == (C, i1 - i0, N), (tuple(A.shape), (C, i1 - i0, N))
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
                assert C * (i1 - i0) * v <= INT32_MAX, ("tile too large", C, i1 - i0, v)
                T = torch.matmul(A, Bq)
                st.tiles += 1
                st.matmul_flops += 2.0 * C * (i1 - i0) * v * N
                c0 = layout.bounds[q][0] + jb * RB
                epilogue_fn(T, (i0, i1), (c0, c0 + v))
                del T
            if b_slabs is None:
                del slab
        del A
    return st


# --------------------------------------------------------------------------
# (c) row-split attention helper
# --------------------------------------------------------------------------
def rowsplit_attention(
    kernel_fn: Callable[..., torch.Tensor],
    q_shard: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    layout: Layout,
    *,
    bias_shard: Optional[torch.Tensor] = None,
    gather: bool = True,
    **kw,
) -> torch.Tensor:
    """Queries = this rank's rows, keys/values = all N (replicated inputs):
    out_shard = kernel_fn(q_shard, k_full, v_full, bias_shard, **kw); optionally
    all_gather_rows -> replicated output.  CAVEAT (P-invariance): only exact if
    kernel_fn's per-query-row arithmetic is independent of the number of query
    rows (no split-KV heuristics keyed on occupancy) -- check per kernel; for
    small operands prefer running the stock statement replicated on every rank."""
    out = kernel_fn(q_shard, k_full, v_full, bias_shard, **kw)
    if gather:
        lead = out.shape[0]
        assert lead == layout.R, (tuple(out.shape), layout.R)
        return all_gather_rows(out.contiguous(), layout)
    return out
