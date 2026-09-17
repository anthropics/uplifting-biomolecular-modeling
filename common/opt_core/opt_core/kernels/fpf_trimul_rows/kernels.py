"""Triton kernels of the ROW-BLOCK statements of the triangle multiplicative update (`opt_core.mem.rowpair.trimul_fused`, P > 1 only) for
pair widths the whole-plane unit `fpf_trimul_v4` is not written at: c_z / c_hidden in {64, 128, 256, 384} (384 = 3 x 128 channel chunks).

    _k1r   K1 of fpf_trimul_v4 (`kernels._k1c`: LayerNorm_in (fp32 statistics, two-pass, tile resident) -> ONE projection's gate|proj MMAs
           (bf16 x bf16 -> fp32) -> (+bias) -> sigmoid(g) * p * mask on the fp32 accumulators -> bf16) reading a CHANNEL-LAST z row block
           [rows, N, C] IN PLACE through two strides and writing channel-major bf16 planes through two strides + the plane stride:
             a-layout  dst [D, rows, N]   (the resident GEMM-A block / a plane written [D, w, N]): program = (BM tokens of block row i)
             b-layout  dst [D, N, w]      (the GEMM-B operand of a b sub-block, its CONTIGUOUS layout -- no transpose copy afterwards):
                                          program = (BM block rows at token k); z is read at row stride N*C, the tile is stored unit-stride along w
           One kernel, two stride sets (`launch_k1`).  No permute / contiguous / cast copy of z on either path.
    _k3r   K3 of fpf_trimul_v4 (`kernels._k3c`: LayerNorm_out(T tile) -> W_o -> (+b_o) ; LayerNorm_in(z) recomputed -> W_g -> (+b_g) -> sigmoid ;
           o = gate * out (-> bf16 when STOCK_ROUND) (+ z residual in fp32) -> z's dtype) on a COLUMN WINDOW z[rows, w, C] of the shard addressed
           IN PLACE (row stride N*C_z): no staging copy in, no copy out; T [CH, rows, w] read channel-strided.

Widths: C (LN_in, the K1 MMAs' K, the K3 gate's K, the K3 output-channel loop) and CH (LN_out, the K3 out-projection's K) are walked as
NCK x CK / NCH x CHK power-of-two chunks held resident (the chunk walk of `opt_core.kernels.trimul_esm_shapes.kernels`, see NOTICE):
a power of two <= 256 is ONE chunk -- then _k1r / _k3r are _k1c / _k3c statement for statement after the address arithmetic -- and
384 is 3 x 128.  z (and the K3 output / residual) may be bf16 or fp32: LN statistics fp32 either way, every MMA bf16 x bf16 -> fp32
accumulate, planes bf16, the K3 store in z's dtype.  Rounding points = fpf_trimul_v4's (cuEq order: gate on the fp32 accumulators, then
round).  Fixed tiles (the rows table `table_rows.json`, keyed capability x widths), no autotune, no atomics, no split-K: run-to-run
bit-exact.  int64 address arithmetic throughout (a shard row stride N*C_z times rows exceeds 2**31 elements at the sizes this serves).

Host launchers: `launch_k1(...)` / `launch_k3(...)` (layout detection from the destination's strides, grids, constexpr plumbing); the
weight pack is fpf_trimul_v4's `pack_generic` (one vocabulary for both units; the row-block provider slices one projection's columns).
"""
import torch
import triton
import triton.language as tl

from . import MAX_CHUNKS, WIDTHS, RowsUnsupported, chunking      # the face's pure width rule (one statement for the face, the provider and these launchers)  # noqa: F401

LAYOUT_A, LAYOUT_B = "a", "b"      # K1 destination layouts: [D, rows, N] (tile along tokens) | [D, N, w] contiguous (tile along block rows)


# =====================================================================================================================
# shared device helpers: two-pass LayerNorm over resident channel chunks (fp32 statistics; chunks k >= NCK alias chunk 0, unused)
# =====================================================================================================================
@triton.jit
def _ln_rows(z0, z1, z2, z3, w_ptr, b_ptr, ck, eps, C: tl.constexpr, CK: tl.constexpr, NCK: tl.constexpr):
    """[BM, CK] fp32 chunks (tokens x channels) -> LayerNorm over the channels -> bf16 chunks (the stock rounding point: LN output in the autocast dtype)."""
    s1 = tl.sum(z0, axis=1)
    if NCK > 1:
        s1 += tl.sum(z1, axis=1)
    if NCK > 2:
        s1 += tl.sum(z2, axis=1)
    if NCK > 3:
        s1 += tl.sum(z3, axis=1)
    mean = s1 / C
    zc_0 = z0 - mean[:, None]
    s2 = tl.sum(zc_0 * zc_0, axis=1)
    if NCK > 1:
        zc_1 = z1 - mean[:, None]
        s2 += tl.sum(zc_1 * zc_1, axis=1)
    if NCK > 2:
        zc_2 = z2 - mean[:, None]
        s2 += tl.sum(zc_2 * zc_2, axis=1)
    if NCK > 3:
        zc_3 = z3 - mean[:, None]
        s2 += tl.sum(zc_3 * zc_3, axis=1)
    var = s2 / C
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + ck)
    b = tl.load(b_ptr + ck)
    x0 = (zc_0 * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    x1 = x0
    x2 = x0
    x3 = x0
    if NCK > 1:
        w = tl.load(w_ptr + CK + ck)
        b = tl.load(b_ptr + CK + ck)
        x1 = (zc_1 * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    if NCK > 2:
        w = tl.load(w_ptr + 2 * CK + ck)
        b = tl.load(b_ptr + 2 * CK + ck)
        x2 = (zc_2 * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    if NCK > 3:
        w = tl.load(w_ptr + 3 * CK + ck)
        b = tl.load(b_ptr + 3 * CK + ck)
        x3 = (zc_3 * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    return x0, x1, x2, x3


@triton.jit
def _ln_cols(z0, z1, z2, z3, w_ptr, b_ptr, ck, eps, C: tl.constexpr, CK: tl.constexpr, NCK: tl.constexpr):
    """[CK, BM] fp32 chunks (channels x tokens) -> LayerNorm over the channel axis -> bf16 chunks."""
    s1 = tl.sum(z0, axis=0)
    if NCK > 1:
        s1 += tl.sum(z1, axis=0)
    if NCK > 2:
        s1 += tl.sum(z2, axis=0)
    if NCK > 3:
        s1 += tl.sum(z3, axis=0)
    mean = s1 / C
    zc_0 = z0 - mean[None, :]
    s2 = tl.sum(zc_0 * zc_0, axis=0)
    if NCK > 1:
        zc_1 = z1 - mean[None, :]
        s2 += tl.sum(zc_1 * zc_1, axis=0)
    if NCK > 2:
        zc_2 = z2 - mean[None, :]
        s2 += tl.sum(zc_2 * zc_2, axis=0)
    if NCK > 3:
        zc_3 = z3 - mean[None, :]
        s2 += tl.sum(zc_3 * zc_3, axis=0)
    var = s2 / C
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + ck)
    b = tl.load(b_ptr + ck)
    x0 = (zc_0 * rstd[None, :] * w[:, None] + b[:, None]).to(tl.bfloat16)
    x1 = x0
    x2 = x0
    x3 = x0
    if NCK > 1:
        w = tl.load(w_ptr + CK + ck)
        b = tl.load(b_ptr + CK + ck)
        x1 = (zc_1 * rstd[None, :] * w[:, None] + b[:, None]).to(tl.bfloat16)
    if NCK > 2:
        w = tl.load(w_ptr + 2 * CK + ck)
        b = tl.load(b_ptr + 2 * CK + ck)
        x2 = (zc_2 * rstd[None, :] * w[:, None] + b[:, None]).to(tl.bfloat16)
    if NCK > 3:
        w = tl.load(w_ptr + 3 * CK + ck)
        b = tl.load(b_ptr + 3 * CK + ck)
        x3 = (zc_3 * rstd[None, :] * w[:, None] + b[:, None]).to(tl.bfloat16)
    return x0, x1, x2, x3


# =====================================================================================================================
# K1: LN_in + ONE gated projection (+mask) of a channel-last z row block -> channel-major bf16 planes, through strides
# program = (BM positions t of the TILED axis, index i on grid axis 1):  z[t, i, c] at z_ptr + t*z_ts + i*z_is + c ;
# mask[t, i] at mask_ptr + t*m_ts + i*m_is ; out[n, i, t] at out_ptr + n*plane_stride + i*o_is + t  (unit stride along the tiled axis)
#   a-layout: t = token (z_ts = C, o unit stride along N), i = block row (z_is = N*C, o_is = N)            -> dst [D, rows, N]
#   b-layout: t = block row (z_ts = N*C, o unit stride along w), i = token (z_is = C, o_is = w)           -> dst [D, N, w] contiguous
# =====================================================================================================================
@triton.jit
def _k1r(z_ptr, lnw_ptr, lnb_ptr, wgT_ptr, wpT_ptr, bg_ptr, bp_ptr, mask_ptr, out_ptr,
         L, z_ts, z_is, m_ts, m_is, o_is, plane_stride, eps,
         C: tl.constexpr, CK: tl.constexpr, NCK: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid_t = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    ts = pid_t * BM + tl.arange(0, BM)
    ts64 = ts.to(tl.int64)
    lmask = ts < L
    ck = tl.arange(0, CK)
    ck64 = ck.to(tl.int64)
    zrow = z_ptr + (ts64 * z_ts + i * z_is)[:, None]                            # [BM, 1] row base pointers (int64 offsets)
    z0 = tl.load(zrow + ck64[None, :], mask=lmask[:, None], other=0.0).to(tl.float32)          # [BM, CK] (bf16 or fp32 z; LN statistics fp32)
    z1 = z0
    z2 = z0
    z3 = z0
    if NCK > 1:
        z1 = tl.load(zrow + (CK + ck64)[None, :], mask=lmask[:, None], other=0.0).to(tl.float32)
    if NCK > 2:
        z2 = tl.load(zrow + (2 * CK + ck64)[None, :], mask=lmask[:, None], other=0.0).to(tl.float32)
    if NCK > 3:
        z3 = tl.load(zrow + (3 * CK + ck64)[None, :], mask=lmask[:, None], other=0.0).to(tl.float32)
    x0, x1, x2, x3 = _ln_rows(z0, z1, z2, z3, lnw_ptr, lnb_ptr, ck, eps, C, CK, NCK)   # bf16 [BM, CK] chunks: the stock rounding point
    if HAS_MASK:
        mv = tl.load(mask_ptr + ts64 * m_ts + i * m_is, mask=lmask, other=0.0).to(tl.float32)
    obase = i * o_is + ts64                                                       # [BM] element offsets inside plane 0
    ncols = tl.arange(0, BN)
    for nb in range(0, D // BN):
        n0 = nb * BN
        woff = (ck64 * D)[:, None] + (n0 + ncols).to(tl.int64)[None, :]          # [CK, BN] of W^T [C, D] (unit stride along out-channels)
        wg = tl.load(wgT_ptr + woff)
        acc_g = tl.dot(x0, wg)                                                    # [BM, BN] fp32
        wp = tl.load(wpT_ptr + woff)
        acc_p = tl.dot(x0, wp)
        if NCK > 1:
            wg = tl.load(wgT_ptr + woff + CK * D)
            acc_g = tl.dot(x1, wg, acc_g)
            wp = tl.load(wpT_ptr + woff + CK * D)
            acc_p = tl.dot(x1, wp, acc_p)
        if NCK > 2:
            wg = tl.load(wgT_ptr + woff + 2 * CK * D)
            acc_g = tl.dot(x2, wg, acc_g)
            wp = tl.load(wpT_ptr + woff + 2 * CK * D)
            acc_p = tl.dot(x2, wp, acc_p)
        if NCK > 3:
            wg = tl.load(wgT_ptr + woff + 3 * CK * D)
            acc_g = tl.dot(x3, wg, acc_g)
            wp = tl.load(wpT_ptr + woff + 3 * CK * D)
            acc_p = tl.dot(x3, wp, acc_p)
        if HAS_BIAS:
            acc_g = acc_g + tl.load(bg_ptr + n0 + ncols)[None, :]
            acc_p = acc_p + tl.load(bp_ptr + n0 + ncols)[None, :]
        val = tl.sigmoid(acc_g) * acc_p                                           # gate on the fp32 accumulators, THEN round (cuEq order)
        if HAS_MASK:
            val = val * mv[:, None]
        val = tl.where(lmask[:, None], val, 0.0)
        planes = (n0 + ncols).to(tl.int64) * plane_stride
        tl.store(out_ptr + obase[:, None] + planes[None, :], val.to(tl.bfloat16), mask=lmask[:, None])


# =====================================================================================================================
# K3: the tile epilogue on a column window of the shard, in place
# program = (BM columns j of the window, window row i):  x[ch, i, j] at x_ptr + ch*x_cs + i*x_rs + j  (T [CH, rows, w], unit column stride);
# z[i, j, c] at z_ptr + i*z_rs + j*C + c  (z window [rows, w, C], row stride z_rs = N*C_z: the shard's own storage) ; out likewise (o_rs).
# In place (out == z) is safe: a program reads its whole z tile (every chunk) before its first store and programs are disjoint.
# =====================================================================================================================
@triton.jit
def _k3r(x_ptr, z_ptr, lnow_ptr, lnob_ptr, lniw_ptr, lnib_ptr, wz_ptr, wg_ptr, bo_ptr, bg_ptr, out_ptr,
         W, x_cs, x_rs, z_rs, o_rs, eps,
         C: tl.constexpr, CK: tl.constexpr, NCK: tl.constexpr, CH: tl.constexpr, CHK: tl.constexpr, NCH: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, RESIDUAL: tl.constexpr, STOCK_ROUND: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid_j = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    cols = pid_j * BM + tl.arange(0, BM)
    cmask = cols < W
    cols64 = cols.to(tl.int64)
    ch = tl.arange(0, CHK)
    ch64 = ch.to(tl.int64)
    cz = tl.arange(0, CK)
    cz64 = cz.to(tl.int64)
    # ---- x tile [CH, BM] as NCH resident chunks [CHK, BM] (channel-strided) -> LayerNorm_out
    xcol = (i * x_rs + cols64)[None, :]
    xt0 = tl.load(x_ptr + (ch64 * x_cs)[:, None] + xcol, mask=cmask[None, :], other=0.0).to(tl.float32)
    xt1 = xt0
    xt2 = xt0
    xt3 = xt0
    if NCH > 1:
        xt1 = tl.load(x_ptr + ((CHK + ch64) * x_cs)[:, None] + xcol, mask=cmask[None, :], other=0.0).to(tl.float32)
    if NCH > 2:
        xt2 = tl.load(x_ptr + ((2 * CHK + ch64) * x_cs)[:, None] + xcol, mask=cmask[None, :], other=0.0).to(tl.float32)
    if NCH > 3:
        xt3 = tl.load(x_ptr + ((3 * CHK + ch64) * x_cs)[:, None] + xcol, mask=cmask[None, :], other=0.0).to(tl.float32)
    xln0, xln1, xln2, xln3 = _ln_cols(xt0, xt1, xt2, xt3, lnow_ptr, lnob_ptr, ch, eps, CH, CHK, NCH)   # [CHK, BM] bf16: LN_out (stock rounding point)
    # ---- z tile transposed [C, BM] as NCK resident chunks -> LayerNorm_in recomputed (the output gate's input)
    ztok = i * z_rs + cols64 * C                                                  # [BM] token offsets of this window row (token-major z)
    zt0 = tl.load(z_ptr + cz64[:, None] + ztok[None, :], mask=cmask[None, :], other=0.0).to(tl.float32)
    zt1 = zt0
    zt2 = zt0
    zt3 = zt0
    if NCK > 1:
        zt1 = tl.load(z_ptr + (CK + cz64)[:, None] + ztok[None, :], mask=cmask[None, :], other=0.0).to(tl.float32)
    if NCK > 2:
        zt2 = tl.load(z_ptr + (2 * CK + cz64)[:, None] + ztok[None, :], mask=cmask[None, :], other=0.0).to(tl.float32)
    if NCK > 3:
        zt3 = tl.load(z_ptr + (3 * CK + cz64)[:, None] + ztok[None, :], mask=cmask[None, :], other=0.0).to(tl.float32)
    zln0, zln1, zln2, zln3 = _ln_cols(zt0, zt1, zt2, zt3, lniw_ptr, lnib_ptr, cz, eps, C, CK, NCK)     # [CK, BM] bf16: LN_in(z)
    otok = i * o_rs + cols64 * C
    nr = tl.arange(0, BN)
    for nb in range(0, C // BN):
        n0 = nb * BN
        n64 = (n0 + nr).to(tl.int64)
        wz = tl.load(wz_ptr + n64[:, None] * CH + ch64[None, :])                 # [BN, CHK] of W_o [C, CH] (native nn.Linear layout [out, in])
        acc_p = tl.dot(wz, xln0)                                                  # [BN, BM] fp32
        if NCH > 1:
            wz = tl.load(wz_ptr + n64[:, None] * CH + (CHK + ch64)[None, :])
            acc_p = tl.dot(wz, xln1, acc_p)
        if NCH > 2:
            wz = tl.load(wz_ptr + n64[:, None] * CH + (2 * CHK + ch64)[None, :])
            acc_p = tl.dot(wz, xln2, acc_p)
        if NCH > 3:
            wz = tl.load(wz_ptr + n64[:, None] * CH + (3 * CHK + ch64)[None, :])
            acc_p = tl.dot(wz, xln3, acc_p)
        wg = tl.load(wg_ptr + n64[:, None] * C + cz64[None, :])                  # [BN, CK] of W_g [C, C]
        acc_g = tl.dot(wg, zln0)
        if NCK > 1:
            wg = tl.load(wg_ptr + n64[:, None] * C + (CK + cz64)[None, :])
            acc_g = tl.dot(wg, zln1, acc_g)
        if NCK > 2:
            wg = tl.load(wg_ptr + n64[:, None] * C + (2 * CK + cz64)[None, :])
            acc_g = tl.dot(wg, zln2, acc_g)
        if NCK > 3:
            wg = tl.load(wg_ptr + n64[:, None] * C + (3 * CK + cz64)[None, :])
            acc_g = tl.dot(wg, zln3, acc_g)
        if HAS_BIAS:
            acc_p = acc_p + tl.load(bo_ptr + n0 + nr)[:, None]
            acc_g = acc_g + tl.load(bg_ptr + n0 + nr)[:, None]
        o = tl.sigmoid(acc_g) * acc_p
        if STOCK_ROUND:
            o = o.to(tl.bfloat16).to(tl.float32)                                  # stock: the kernel returns bf16; torch then adds the residual (fp32 opmath -> z's dtype)
        if RESIDUAL:
            o = o + tl.load(z_ptr + n64[:, None] + ztok[None, :], mask=cmask[None, :], other=0.0).to(tl.float32)
        tl.store(out_ptr + n64[:, None] + otok[None, :], o.to(out_ptr.dtype.element_ty), mask=cmask[None, :])


# =====================================================================================================================
# host launchers (no allocation, no copy: layouts are read off the tensors' strides; a layout the kernels do not address raises BY NAME)
# =====================================================================================================================
IO_DTYPES = (torch.bfloat16, torch.float32)
MASK_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def k1_layout(dst, rows: int, N: int, D: int):
    """The K1 destination layout word for ``dst``: ``'a'`` = ``[D, rows, N]`` with unit stride along N; ``'b'`` = a ``[D, rows, N]`` VIEW whose unit
    stride is along rows (``x.transpose(1, 2)`` of a contiguous ``[D, N, rows]`` buffer: the GEMM-B layout); RowsUnsupported otherwise."""
    if tuple(dst.shape) != (D, rows, N) or dst.dtype != torch.bfloat16:
        raise RowsUnsupported("k1_dst:shape=%s,dtype=%s(want [%d, %d, %d] bf16)" % (tuple(dst.shape), dst.dtype, D, rows, N))
    sp, sr, sn = (int(s) for s in dst.stride())
    if sn == 1 and sr >= N and sp >= rows * sr:
        return LAYOUT_A
    if sr == 1 and sn >= rows and sp >= N * sn:
        return LAYOUT_B
    raise RowsUnsupported("k1_dst:strides=%s" % (tuple(dst.stride()),))


def launch_k1(z_block, mask_block, dst, *, ln_w, ln_b, wgT, wpT, bg, bp, has_bias: bool, cell, eps: float):
    """K1 of ONE projection into ``dst`` from ``z_block [rows, N, C]`` (channel-last, unit channel stride, any row / token strides) and ``mask_block
    [rows, N]`` (fp32 | bf16 | fp16, any strides) or None.  ``dst``: layout 'a' or 'b' (:func:`k1_layout`).  ``wgT`` / ``wpT``: ``[C, D]`` bf16
    (this projection's columns of the pack's ``wgT_in`` / ``wpT_in``), ``bg`` / ``bp``: ``[D]`` fp32 or None.  ``cell``: ``{BM, BN, num_warps,
    num_stages}``.  Returns the layout word."""
    rows, N, C = (int(s) for s in z_block.shape)
    D = int(wgT.shape[1])
    if z_block.dtype not in IO_DTYPES or z_block.stride(2) != 1:
        raise RowsUnsupported("k1_z:dtype=%s,strides=%s" % (z_block.dtype, tuple(z_block.stride())))
    layout = k1_layout(dst, rows, N, D)
    CK, NCK = chunking(C)
    BM, BN = int(cell["BM"]), min(int(cell["BN"]), D)
    if D % BN:
        raise RowsUnsupported("k1_cell:BN=%d does not divide D=%d" % (BN, D))
    hm = mask_block is not None
    if hm:
        if tuple(mask_block.shape) != (rows, N) or mask_block.dtype not in MASK_DTYPES:
            raise RowsUnsupported("k1_mask:shape=%s,dtype=%s" % (tuple(mask_block.shape), mask_block.dtype))
        m, m_rs, m_ns = mask_block, int(mask_block.stride(0)), int(mask_block.stride(1))
    else:
        m, m_rs, m_ns = z_block, 0, 0
    z_rs, z_ns = int(z_block.stride(0)), int(z_block.stride(1))
    sp, sr, sn = (int(s) for s in dst.stride())
    if layout == LAYOUT_A:                                   # tile along tokens; grid axis 1 = block rows
        L, G1, z_ts, z_is, m_ts, m_is, o_is = N, rows, z_ns, z_rs, m_ns, m_rs, sr
    else:                                                    # tile along block rows; grid axis 1 = tokens
        L, G1, z_ts, z_is, m_ts, m_is, o_is = rows, N, z_rs, z_ns, m_rs, m_ns, sn
    if rows == 0 or N == 0:
        return layout
    grid = (triton.cdiv(L, BM), G1, 1)
    _k1r[grid](z_block, ln_w, ln_b, wgT, wpT, bg if has_bias else ln_w, bp if has_bias else ln_w, m, dst,
               L, z_ts, z_is, m_ts, m_is, o_is, sp, float(eps),
               C=C, CK=CK, NCK=NCK, D=D, BM=BM, BN=BN, HAS_MASK=hm, HAS_BIAS=bool(has_bias),
               num_warps=int(cell["num_warps"]), num_stages=int(cell["num_stages"]))
    return layout


def launch_k3(T, z_win, out_win, *, ln_out_w, ln_out_b, ln_in_w, ln_in_b, wz, wg_out, bo, bg, has_bias: bool, cell, residual: bool,
              stock_round: bool, eps: float):
    """K3 on the tile: ``out_win[i, j, :] = sigmoid(LN_in(z) W_g^T + b_g) * (LN_out(T[:, i, j]) W_o^T + b_o) (+ z_win[i, j, :])`` for the window
    ``z_win [rows, w, C]`` (unit channel stride, column stride C, ANY row stride: a column window of the shard in place) and ``T [CH, rows, w]``
    (bf16, unit column stride).  ``out_win``: same shape / dtype / stride class as ``z_win`` (may BE ``z_win``)."""
    rows, w, C = (int(s) for s in z_win.shape)
    CH = int(T.shape[0])
    if tuple(T.shape) != (CH, rows, w) or T.dtype != torch.bfloat16 or (w > 1 and T.stride(2) != 1):
        raise RowsUnsupported("k3_T:shape=%s,dtype=%s,strides=%s(want [%d, %d, %d] bf16, unit column stride)" % (tuple(T.shape), T.dtype, tuple(T.stride()), CH, rows, w))
    for name, t in (("z", z_win), ("out", out_win)):
        if tuple(t.shape) != (rows, w, C) or t.dtype not in IO_DTYPES or t.stride(2) != 1 or (w > 1 and t.stride(1) != C):
            raise RowsUnsupported("k3_%s:shape=%s,dtype=%s,strides=%s" % (name, tuple(t.shape), t.dtype, tuple(t.stride())))
    if out_win.dtype != z_win.dtype:
        raise RowsUnsupported("k3_out:dtype=%s!=z %s" % (out_win.dtype, z_win.dtype))
    CK, NCK = chunking(C)
    CHK, NCH = chunking(CH)
    BM, BN = int(cell["BM"]), min(int(cell["BN"]), C)
    if C % BN:
        raise RowsUnsupported("k3_cell:BN=%d does not divide C=%d" % (BN, C))
    if rows == 0 or w == 0:
        return
    grid = (triton.cdiv(w, BM), rows, 1)
    _k3r[grid](T, z_win, ln_out_w, ln_out_b, ln_in_w, ln_in_b, wz, wg_out, bo if has_bias else ln_in_w, bg if has_bias else ln_in_w, out_win,
               w, int(T.stride(0)), int(T.stride(1)), int(z_win.stride(0)), int(out_win.stride(0)), float(eps),
               C=C, CK=CK, NCK=NCK, CH=CH, CHK=CHK, NCH=NCH, BM=BM, BN=BN, RESIDUAL=bool(residual), STOCK_ROUND=bool(stock_round),
               HAS_BIAS=bool(has_bias), num_warps=int(cell["num_warps"]), num_stages=int(cell["num_stages"]))


# =====================================================================================================================
# the torch statements of the two pieces (the correctness yardstick of the sweep / tests; fp32 math with the kernels' rounding points)
# =====================================================================================================================
def ref_k1(z_block, mask_block, *, ln_w, ln_b, w_g, w_p, b_g=None, b_p=None, eps=1e-5):
    """``[rows, N, D]`` fp32: ``sigmoid(x w_g^T + b_g) * (x w_p^T + b_p) * mask`` with ``x = LN(z).bf16`` and bf16 weights (fp32 accumulate)."""
    C = int(z_block.shape[-1])
    x = torch.nn.functional.layer_norm(z_block.float(), (C,), ln_w.float(), ln_b.float(), eps).to(torch.bfloat16).float()
    g = x @ w_g.to(torch.bfloat16).float().t()
    p = x @ w_p.to(torch.bfloat16).float().t()
    if b_g is not None:
        g = g + b_g.float()
    if b_p is not None:
        p = p + b_p.float()
    v = torch.sigmoid(g) * p
    if mask_block is not None:
        v = v * mask_block.float().unsqueeze(-1)
    return v


def ref_k3(T, z_win, *, ln_out_w, ln_out_b, ln_in_w, ln_in_b, w_o, w_og, b_o=None, b_og=None, residual=True, stock_round=True, eps=1e-5):
    """``[rows, w, C]`` in z's dtype: the epilogue statements on ``T [CH, rows, w]`` / ``z_win [rows, w, C]``."""
    CH, C = int(T.shape[0]), int(z_win.shape[-1])
    x = torch.nn.functional.layer_norm(T.permute(1, 2, 0).float(), (CH,), ln_out_w.float(), ln_out_b.float(), eps).to(torch.bfloat16).float()
    zl = torch.nn.functional.layer_norm(z_win.float(), (C,), ln_in_w.float(), ln_in_b.float(), eps).to(torch.bfloat16).float()
    p = x @ w_o.to(torch.bfloat16).float().t()
    g = zl @ w_og.to(torch.bfloat16).float().t()
    if b_o is not None:
        p = p + b_o.float()
    if b_og is not None:
        g = g + b_og.float()
    o = torch.sigmoid(g) * p
    if stock_round:
        o = o.to(torch.bfloat16).float()
    if residual:
        o = o + z_win.float()
    return o.to(z_win.dtype)
