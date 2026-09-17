"""K1 of the ESM-family TriMul line with tensor-descriptor (TMA) reads of the z tile and the weight blocks (cc 9.0 cells).  Same statements
as kernels._k1s after the loads; imported by kernels.launch_k1 only when a cell asks for it and the descriptor API exists (triton >= 3.4)."""
import triton
import triton.language as tl

from .kernels import _k1_gate_store, _k1_tile, _ln_rows


@triton.jit
def _k1s_tma(z_ptr, lnw_ptr, lnb_ptr, wgT_ptr, wpT_ptr, mask_ptr, ab_ptr,
             N, Np, NN, NC, plane_stride, eps,
             C: tl.constexpr, CK: tl.constexpr, NCK: tl.constexpr, D2: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr,
             HAS_MASK: tl.constexpr, TRANS: tl.constexpr, FASTSIG: tl.constexpr, STAGGER: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    r = tl.arange(0, BM)
    ck = tl.arange(0, CK)
    tok, lmask, obase, smask, mv = _k1_tile(pid0, pid1, r, mask_ptr, N, Np, BM, HAS_MASK, TRANS)
    # the z tile: TMA boxes [BM, CK] on the [N*N, C] (row tiling) / [N, N*C] (column tiling) views of z; out-of-range rows are zero-filled by the
    # unit and zeroed again by lmask at the store; the last row tile of the [N*N, C] view may wrap into the next pair row (masked the same way)
    if TRANS:
        zd = tl.make_tensor_descriptor(z_ptr, shape=[N, NC], strides=[NC, 1], block_shape=[BM, CK])
        zr0 = pid0 * BM
        zc0 = tl.minimum(pid1, N - 1) * C
    else:
        zd = tl.make_tensor_descriptor(z_ptr, shape=[NN, C], strides=[C, 1], block_shape=[BM, CK])
        zr0 = tl.minimum(pid1, N - 1) * N + pid0 * BM
        zc0 = 0
    z0 = zd.load([zr0, zc0]).to(tl.float32)
    z1 = z0
    z2 = z0
    z3 = z0
    if NCK > 1:
        z1 = zd.load([zr0, zc0 + CK]).to(tl.float32)
    if NCK > 2:
        z2 = zd.load([zr0, zc0 + 2 * CK]).to(tl.float32)
    if NCK > 3:
        z3 = zd.load([zr0, zc0 + 3 * CK]).to(tl.float32)
    x0, x1, x2, x3 = _ln_rows(z0, z1, z2, z3, lnw_ptr, lnb_ptr, ck, eps, C, CK, NCK)
    wg_desc = tl.make_tensor_descriptor(wgT_ptr, shape=[C, D2], strides=[D2, 1], block_shape=[CK, BN])
    wp_desc = tl.make_tensor_descriptor(wpT_ptr, shape=[C, D2], strides=[D2, 1], block_shape=[CK, BN])
    ncols = tl.arange(0, BN)
    NB: tl.constexpr = D2 // BN
    for nb in tl.range(0, NB, num_stages=STAGES):
        if STAGGER:
            n0 = ((nb + pid0 + pid1) % NB) * BN
        else:
            n0 = nb * BN
        wg = wg_desc.load([0, n0])
        wp = wp_desc.load([0, n0])
        acc_g = tl.dot(x0, wg)
        acc_p = tl.dot(x0, wp)
        if NCK > 1:
            wg = wg_desc.load([CK, n0])
            wp = wp_desc.load([CK, n0])
            acc_g = tl.dot(x1, wg, acc_g)
            acc_p = tl.dot(x1, wp, acc_p)
        if NCK > 2:
            wg = wg_desc.load([2 * CK, n0])
            wp = wp_desc.load([2 * CK, n0])
            acc_g = tl.dot(x2, wg, acc_g)
            acc_p = tl.dot(x2, wp, acc_p)
        if NCK > 3:
            wg = wg_desc.load([3 * CK, n0])
            wp = wp_desc.load([3 * CK, n0])
            acc_g = tl.dot(x3, wg, acc_g)
            acc_p = tl.dot(x3, wp, acc_p)
        _k1_gate_store(acc_g, acc_p, mv, lmask, smask, obase, n0, ncols, plane_stride, ab_ptr, HAS_MASK, FASTSIG)
