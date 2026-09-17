"""fpf_trimul.kernels — TriangleMultiplication as three fixed-tile Triton launches.

Function computed (through the kit's adapter): the same pipeline as the stock fused op, cuEquivariance 0.8/0.10 `triangle_multiplicative_update`
    x_in = LN_in(z) (fp32 stats, affine, rounded to z.dtype)
    ab   = sigmoid(x_in @ g_inT) * (x_in @ p_inT) * mask   -> a = ab[:C] (layout [d,i,k]), b = ab[C:]
    x    = outgoing: sum_k a[d,i,k] b[d,j,k] | incoming: sum_k a[d,k,i] b[d,k,j]    (fp32 accumulate, x.dtype out)
    out  = sigmoid(x_in @ g_outT) * (LN_out(x) @ p_outT)                           (+ z residual, optional, bf16 double rounding as stock)
Operand precision policy (the same observable numerics as the stock op's gated GEMM): the activation operand is cast to the WEIGHT dtype before the dot;
weight dtype bf16/fp16 -> native MMA; fp32 -> tf32 (RN-converted operands) if torch.backends.cuda.matmul.allow_tf32 else tf32x3.
Contraction operands are a,b in x.dtype (bf16 -> bf16 MMA; fp32 -> tf32 MMA if allow_tf32 else tf32x3), matching the stock cuBLAS class.

All tiles are fixed (no autotune at run time), no atomics, no split-K: run-to-run bit-exact and batch-invariant by construction.
Builds on triton 3.3.1 and 3.7.0 images: no TMA descriptors / 3.4+-only features.
"""
import torch, triton, triton.language as tl

# ----------------------------------------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _cvt_tf32_rn(x):
    # fp32 -> tf32 with round-to-nearest (PTX cvt.rna.tf32.f32): the operand rounding the stock op's TF32 GEMMs apply
    return tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r, r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _dot(a, b, acc, PREC: tl.constexpr):
    # PREC: 0 = native (bf16/fp16 operands), 1 = tf32 (RN converted), 2 = tf32x3, 3 = ieee
    if PREC == 0:
        return tl.dot(a, b, acc)
    elif PREC == 1:
        return tl.dot(_cvt_tf32_rn(a), _cvt_tf32_rn(b), acc, input_precision="tf32")
    elif PREC == 2:
        return tl.dot(a, b, acc, input_precision="tf32x3")
    else:
        return tl.dot(a, b, acc, input_precision="ieee")


# ----------------------------------------------------------------------------------------------------------------------
# K_A: LN_in prologue + dual gated GEMM + mask, channel-major padded store
#   z      : [M, C] row-major (M = N*N, C = K of the GEMM)
#   wp, wg : [2C, C] row-major (cat[a_p, b_p], cat[a_g, b_g])
#   ab     : [2C, Np, Np] (persistent, zero tails) ; element (n, i, k) at n*Np*Np + i*Np + k
#   mask   : [M] (same dtype as z) or none
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _k_proj(z_ptr, wp_ptr, wg_ptr, lnw_ptr, lnb_ptr, mask_ptr, ab_ptr,
            N, Np, C, NKT, eps,
            HAS_MASK: tl.constexpr, PREC: tl.constexpr, WDT: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr):
    # grid: (2C/BN, N*NKT): n-tiles adjacent in launch order so the z tile is re-read from L2, not HBM.
    # program -> row i of the pair tensor, k-tile kt : rows m = i*N + k, k in [kt*BM, kt*BM+BM)
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NKT
    kt = pid - i * NKT
    offs_k = kt * BM + tl.arange(0, BM)
    k_ok = offs_k < N
    offs_m = i.to(tl.int64) * N + offs_k          # flat z rows (row-major [M, C]: natural loads, c contiguous)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, BK)
    # ---- LN_in statistics: two-pass fp32 (mean, then centred variance) over the C channels of each row
    s1 = tl.zeros((BM,), dtype=tl.float32)
    for kk in tl.static_range(NK):
        zt = tl.load(z_ptr + offs_m[:, None] * C + (kk * BK + offs_c)[None, :], mask=k_ok[:, None], other=0.0).to(tl.float32)
        s1 += tl.sum(zt, 1)
    mean = s1 / C
    s2 = tl.zeros((BM,), dtype=tl.float32)
    for kk in tl.static_range(NK):
        zt = tl.load(z_ptr + offs_m[:, None] * C + (kk * BK + offs_c)[None, :], mask=k_ok[:, None], other=0.0).to(tl.float32)
        d = tl.where(k_ok[:, None], zt - mean[:, None], 0.0)
        s2 += tl.sum(d * d, 1)
    rstd = tl.rsqrt(s2 / C + eps)
    # ---- transposed dual GEMM: accT[n, k] = sum_c W[n, c] * x[k, c]   (W natural [BN, BK], x natural [BM, BK] -> trans)
    acc_p = tl.zeros((BN, BM), dtype=tl.float32)
    acc_g = tl.zeros((BN, BM), dtype=tl.float32)
    for kk in tl.static_range(NK):
        c_idx = kk * BK + offs_c
        zt = tl.load(z_ptr + offs_m[:, None] * C + c_idx[None, :], mask=k_ok[:, None], other=0.0)
        w = tl.load(lnw_ptr + c_idx).to(tl.float32)
        b = tl.load(lnb_ptr + c_idx).to(tl.float32)
        x = (zt.to(tl.float32) - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]
        x = x.to(zt.dtype).to(WDT)                                          # LN output rounded to z.dtype (stock LN), then the cuEq cast to the weight dtype
        xT = tl.trans(x)                                                    # [BK, BM]
        wpt = tl.load(wp_ptr + offs_n[:, None] * C + c_idx[None, :])        # [BN, BK] natural
        wgt = tl.load(wg_ptr + offs_n[:, None] * C + c_idx[None, :])
        acc_p = _dot(wpt, xT, acc_p, PREC)
        acc_g = _dot(wgt, xT, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = g * acc_p
    if HAS_MASK:
        mk = tl.load(mask_ptr + offs_m, mask=k_ok, other=0.0).to(tl.float32)
        o = o * mk[None, :]
    # ---- channel-major padded store ab[n, i, k]: k contiguous along dim 1 of the accumulator -> natural coalesced store
    optrs = ab_ptr + offs_n[:, None].to(tl.int64) * (Np * Np) + i.to(tl.int64) * Np + offs_k[None, :]
    tl.store(optrs, o.to(ab_ptr.dtype.element_ty), mask=k_ok[None, :])


# ----------------------------------------------------------------------------------------------------------------------
# K_B: batched contraction over d. a, b: [C, Np, Np]; x: [C, Np, Np]
#   OUTGOING: x[d,i,j] = sum_k a[d,i,k] b[d,j,k]   (A row-major NT)
#   INCOMING: x[d,i,j] = sum_k a[d,k,i] b[d,k,j]   (A col-major TN)
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _k_contract(a_ptr, b_ptr, x_ptr, Np, OUTGOING: tl.constexpr, PREC: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DMAJOR: tl.constexpr):
    # OUTGOING (NT): x[d,i,j] = sum_k a[d,i,k] b[d,j,k]   A natural [BM,BK], B natural [BN,BK] -> dot(A, trans(B))
    # INCOMING (TN): x[d,i,j] = sum_k a[d,k,i] b[d,k,j]   A natural [BK,BM], B natural [BK,BN] -> dot(trans(A), B)
    if DMAJOR:
        pid_d = tl.program_id(2); pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    else:
        pid_d = tl.program_id(0); pid_m = tl.program_id(1); pid_n = tl.program_id(2)
    base = pid_d.to(tl.int64) * (Np * Np)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(Np, BK)):
        k_idx = kk * BK + offs_k
        if OUTGOING:
            at = tl.load(a_ptr + base + offs_m[:, None] * Np + k_idx[None, :])     # [BM, BK]
            bt = tl.load(b_ptr + base + offs_n[:, None] * Np + k_idx[None, :])     # [BN, BK]
            acc = _dot(at, tl.trans(bt), acc, PREC)
        else:
            at = tl.load(a_ptr + base + k_idx[:, None] * Np + offs_m[None, :])     # [BK, BM]
            bt = tl.load(b_ptr + base + k_idx[:, None] * Np + offs_n[None, :])     # [BK, BN]
            acc = _dot(tl.trans(at), bt, acc, PREC)
    xptrs = x_ptr + base + offs_m[:, None] * Np + offs_n[None, :]
    tl.store(xptrs, acc.to(x_ptr.dtype.element_ty))


# ----------------------------------------------------------------------------------------------------------------------
# K_C: LN_out prologue on channel-major x + out-proj GEMM ; LN_in(z) recompute + gate GEMM ; sigmoid(g)*p (+ z) -> out
#   x   : [C, Np, Np] ; z, out : [M, C] row-major ; wpo, wgo : [C, C] row-major
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _k_out(x_ptr, z_ptr, wpo_ptr, wgo_ptr, lniw_ptr, lnib_ptr, lnow_ptr, lnob_ptr, out_ptr,
           N, Np, C, NJT, eps,
           RESIDUAL: tl.constexpr, PREC: tl.constexpr, WDT: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr):
    # grid: (C/BN, N*NJT) ; program -> row i, j-tile jt : pair positions (i, j), j in [jt*BM, jt*BM+BM)
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NJT
    jt = pid - i * NJT
    offs_j = jt * BM + tl.arange(0, BM)
    j_ok = offs_j < N
    offs_m = i.to(tl.int64) * N + offs_j          # flat rows of z / out
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, BK)
    NpNp = Np * Np
    xbase = x_ptr + i.to(tl.int64) * Np + offs_j[None, :]          # x[d, i, j] natural: [BK rows of d, BM contiguous j]
    # ---- LN_out stats over d for each j (reduce over dim 0 of the natural [BK, BM] tiles)
    s1 = tl.zeros((BM,), dtype=tl.float32)
    for kk in tl.static_range(NK):
        c_idx = kk * BK + offs_c
        xt = tl.load(xbase + c_idx[:, None].to(tl.int64) * NpNp, mask=j_ok[None, :], other=0.0).to(tl.float32)
        s1 += tl.sum(xt, 0)
    mean_o = s1 / C
    s2 = tl.zeros((BM,), dtype=tl.float32)
    for kk in tl.static_range(NK):
        c_idx = kk * BK + offs_c
        xt = tl.load(xbase + c_idx[:, None].to(tl.int64) * NpNp, mask=j_ok[None, :], other=0.0).to(tl.float32)
        d = tl.where(j_ok[None, :], xt - mean_o[None, :], 0.0)
        s2 += tl.sum(d * d, 0)
    rstd_o = tl.rsqrt(s2 / C + eps)
    # ---- LN_in stats over the z row (natural [BM, BK] tiles, reduce over dim 1)
    t1 = tl.zeros((BM,), dtype=tl.float32)
    for kk in tl.static_range(NK):
        zt = tl.load(z_ptr + offs_m[:, None] * C + (kk * BK + offs_c)[None, :], mask=j_ok[:, None], other=0.0).to(tl.float32)
        t1 += tl.sum(zt, 1)
    mean_i = t1 / C
    t2 = tl.zeros((BM,), dtype=tl.float32)
    for kk in tl.static_range(NK):
        zt = tl.load(z_ptr + offs_m[:, None] * C + (kk * BK + offs_c)[None, :], mask=j_ok[:, None], other=0.0).to(tl.float32)
        d = tl.where(j_ok[:, None], zt - mean_i[:, None], 0.0)
        t2 += tl.sum(d * d, 1)
    rstd_i = tl.rsqrt(t2 / C + eps)
    # ---- dual GEMM: p[j, n] = sum_d LN_out(x)[d, j] Wpo[n, d] ; g[j, n] = sum_c LN_in(z)[j, c] Wgo[n, c]
    acc_p = tl.zeros((BM, BN), dtype=tl.float32)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in tl.static_range(NK):
        c_idx = kk * BK + offs_c
        xt = tl.load(xbase + c_idx[:, None].to(tl.int64) * NpNp, mask=j_ok[None, :], other=0.0)          # [BK, BM]
        wo = tl.load(lnow_ptr + c_idx).to(tl.float32)
        bo = tl.load(lnob_ptr + c_idx).to(tl.float32)
        xo = (xt.to(tl.float32) - mean_o[None, :]) * rstd_o[None, :] * wo[:, None] + bo[:, None]
        xo = xo.to(xt.dtype).to(WDT)                                                                     # [BK, BM]
        zt = tl.load(z_ptr + offs_m[:, None] * C + c_idx[None, :], mask=j_ok[:, None], other=0.0)        # [BM, BK]
        wi = tl.load(lniw_ptr + c_idx).to(tl.float32)
        bi = tl.load(lnib_ptr + c_idx).to(tl.float32)
        xi = (zt.to(tl.float32) - mean_i[:, None]) * rstd_i[:, None] * wi[None, :] + bi[None, :]
        xi = xi.to(zt.dtype).to(WDT)
        wpt = tl.load(wpo_ptr + offs_n[None, :] * C + c_idx[:, None])                                    # [BK, BN] (W^T view, c contiguous along dim 0)
        wgt = tl.load(wgo_ptr + offs_n[None, :] * C + c_idx[:, None])
        acc_p = _dot(tl.trans(xo), wpt, acc_p, PREC)
        acc_g = _dot(xi, wgt, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = (g * acc_p).to(x_ptr.dtype.element_ty)            # stock: cuEq output rounded to the compute dtype (= x's dtype)
    if RESIDUAL:
        zr = tl.load(z_ptr + offs_m[:, None] * C + offs_n[None, :], mask=j_ok[:, None], other=0.0)
        o = (o.to(tl.float32) + zr.to(tl.float32)).to(out_ptr.dtype.element_ty)   # stock: `z + z_in` (promoted add, one rounding to z's dtype)
    tl.store(out_ptr + offs_m[:, None] * C + offs_n[None, :], o.to(out_ptr.dtype.element_ty), mask=j_ok[:, None])


# ----------------------------------------------------------------------------------------------------------------------
# v3 register-resident variants (single K=C dot, z/x tile loaded once, pre-transposed weights -> no tl.trans in a loop)
#   K_A3: grid (2C/BN, N*NKT); z tile [BM, C] once; LN in registers; dot(x[BM,C], WpT[C,BN]) ; channel-major store (needs trans of acc)
#   K_C3: grid (C/BN, N*NJT); x block [C, BM] once (j contiguous), stats over dim 0; z tile [BM, C] once; dots with K=C
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _k_proj3(z_ptr, wpT_ptr, wgT_ptr, lnw_ptr, lnb_ptr, mask_ptr, ab_ptr,
             N, Np, C, NKT, eps,
             HAS_MASK: tl.constexpr, PREC: tl.constexpr, WDT: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, CK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NKT
    kt = pid - i * NKT
    offs_k = kt * BM + tl.arange(0, BM)
    k_ok = offs_k < N
    offs_m = i.to(tl.int64) * N + offs_k
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, CK)
    zt = tl.load(z_ptr + offs_m[:, None] * C + offs_c[None, :], mask=k_ok[:, None], other=0.0)      # [BM, C] once
    zf = zt.to(tl.float32)
    mean = tl.sum(zf, 1) / C
    d = tl.where(k_ok[:, None], zf - mean[:, None], 0.0)
    var = tl.sum(d * d, 1) / C
    rstd = tl.rsqrt(var + eps)
    w = tl.load(lnw_ptr + offs_c).to(tl.float32)
    b = tl.load(lnb_ptr + offs_c).to(tl.float32)
    x = ((zf - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]).to(zt.dtype).to(WDT)          # [BM, C]
    wp = tl.load(wpT_ptr + offs_c[:, None] * (2 * C) + offs_n[None, :])                                # [C, BN] natural (W^T cached [C, 2C])
    wg = tl.load(wgT_ptr + offs_c[:, None] * (2 * C) + offs_n[None, :])
    acc_p = _dot(x, wp, tl.zeros((BM, BN), dtype=tl.float32), PREC)
    acc_g = _dot(x, wg, tl.zeros((BM, BN), dtype=tl.float32), PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = g * acc_p
    if HAS_MASK:
        mk = tl.load(mask_ptr + offs_m, mask=k_ok, other=0.0).to(tl.float32)
        o = o * mk[:, None]
    oT = tl.trans(o)                                                                                     # [BN, BM] -> k contiguous store
    optrs = ab_ptr + offs_n[:, None].to(tl.int64) * (Np * Np) + i.to(tl.int64) * Np + offs_k[None, :]
    tl.store(optrs, oT.to(ab_ptr.dtype.element_ty), mask=k_ok[None, :])


@triton.jit
def _k_out3(x_ptr, z_ptr, wpoT_ptr, wgoT_ptr, lniw_ptr, lnib_ptr, lnow_ptr, lnob_ptr, out_ptr,
            N, Np, C, NJT, eps,
            RESIDUAL: tl.constexpr, PREC: tl.constexpr, WDT: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, CK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NJT
    jt = pid - i * NJT
    offs_j = jt * BM + tl.arange(0, BM)
    j_ok = offs_j < N
    offs_m = i.to(tl.int64) * N + offs_j
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, CK)
    NpNp = Np * Np
    xt = tl.load(x_ptr + offs_c[:, None].to(tl.int64) * NpNp + i.to(tl.int64) * Np + offs_j[None, :], mask=j_ok[None, :], other=0.0)   # [C, BM] once
    xf = xt.to(tl.float32)
    mean_o = tl.sum(xf, 0) / C
    d = tl.where(j_ok[None, :], xf - mean_o[None, :], 0.0)
    rstd_o = tl.rsqrt(tl.sum(d * d, 0) / C + eps)
    wo = tl.load(lnow_ptr + offs_c).to(tl.float32)
    bo = tl.load(lnob_ptr + offs_c).to(tl.float32)
    xo = ((xf - mean_o[None, :]) * rstd_o[None, :] * wo[:, None] + bo[:, None]).to(xt.dtype).to(WDT)     # [C, BM]
    zt = tl.load(z_ptr + offs_m[:, None] * C + offs_c[None, :], mask=j_ok[:, None], other=0.0)            # [BM, C] once
    zf = zt.to(tl.float32)
    mean_i = tl.sum(zf, 1) / C
    d2 = tl.where(j_ok[:, None], zf - mean_i[:, None], 0.0)
    rstd_i = tl.rsqrt(tl.sum(d2 * d2, 1) / C + eps)
    wi = tl.load(lniw_ptr + offs_c).to(tl.float32)
    bi = tl.load(lnib_ptr + offs_c).to(tl.float32)
    xi = ((zf - mean_i[:, None]) * rstd_i[:, None] * wi[None, :] + bi[None, :]).to(zt.dtype).to(WDT)      # [BM, C]
    wpo = tl.load(wpoT_ptr + offs_c[:, None] * C + offs_n[None, :])                                        # [C, BN] natural (W^T cached)
    wgo = tl.load(wgoT_ptr + offs_c[:, None] * C + offs_n[None, :])
    acc_p = _dot(tl.trans(xo), wpo, tl.zeros((BM, BN), dtype=tl.float32), PREC)
    acc_g = _dot(xi, wgo, tl.zeros((BM, BN), dtype=tl.float32), PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = (g * acc_p).to(x_ptr.dtype.element_ty)
    if RESIDUAL:
        zr = tl.load(z_ptr + offs_m[:, None] * C + offs_n[None, :], mask=j_ok[:, None], other=0.0)
        o = (o.to(tl.float32) + zr.to(tl.float32)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_m[:, None] * C + offs_n[None, :], o.to(out_ptr.dtype.element_ty), mask=j_ok[:, None])


# ----------------------------------------------------------------------------------------------------------------------
# v4: LN statistics computed ONCE per row in tiny pre-passes; K_A4 / K_C4 are plain K=C GEMMs with a scale/shift on the A-operand load.
#   _k_rowstats   : mean/rstd over C for each row of a row-major [M, C] tensor (z)            -> mean[M], rstd[M] fp32
#   _k_colstats   : mean/rstd over d for each (i,j) of the channel-major x [C, Np, Np]        -> mean[M], rstd[M] fp32 (M = N*N, j contiguous)
#   Two-pass (mean, then centred variance) in fp32, rsqrt(var+eps): the same arithmetic as the cuEq LayerNorm kernel.
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _k_rowstats(z_ptr, mean_ptr, rstd_ptr, M, C, eps, BM: tl.constexpr, CK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid.to(tl.int64) * BM + tl.arange(0, BM)
    m_ok = offs_m < M
    offs_c = tl.arange(0, CK)
    zt = tl.load(z_ptr + offs_m[:, None] * C + offs_c[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(zt, 1) / C
    d = tl.where(m_ok[:, None], zt - mean[:, None], 0.0)
    var = tl.sum(d * d, 1) / C
    tl.store(mean_ptr + offs_m, mean, mask=m_ok)
    tl.store(rstd_ptr + offs_m, tl.rsqrt(var + eps), mask=m_ok)


@triton.jit
def _k_colstats(x_ptr, mean_ptr, rstd_ptr, N, Np, C, NJT, eps, BM: tl.constexpr, CK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid // NJT
    jt = pid - i * NJT
    offs_j = jt * BM + tl.arange(0, BM)
    j_ok = offs_j < N
    offs_c = tl.arange(0, CK)
    xt = tl.load(x_ptr + offs_c[:, None].to(tl.int64) * (Np * Np) + i.to(tl.int64) * Np + offs_j[None, :], mask=j_ok[None, :], other=0.0).to(tl.float32)
    mean = tl.sum(xt, 0) / C
    d = tl.where(j_ok[None, :], xt - mean[None, :], 0.0)
    var = tl.sum(d * d, 0) / C
    offs_m = i * N + offs_j
    tl.store(mean_ptr + offs_m, mean, mask=j_ok)
    tl.store(rstd_ptr + offs_m, tl.rsqrt(var + eps), mask=j_ok)


@triton.jit
def _k_proj4(z_ptr, mean_ptr, rstd_ptr, wpT_ptr, wgT_ptr, lnw_ptr, lnb_ptr, mask_ptr, ab_ptr,
             N, Np, C, NKT, eps,
             HAS_MASK: tl.constexpr, PREC: tl.constexpr, WDT: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NKT
    kt = pid - i * NKT
    offs_k = kt * BM + tl.arange(0, BM)
    k_ok = offs_k < N
    offs_m = i.to(tl.int64) * N + offs_k
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, BK)
    mean = tl.load(mean_ptr + offs_m, mask=k_ok, other=0.0)
    rstd = tl.load(rstd_ptr + offs_m, mask=k_ok, other=0.0)
    acc_p = tl.zeros((BM, BN), dtype=tl.float32)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in tl.static_range(NK):
        c_idx = kk * BK + offs_c
        zt = tl.load(z_ptr + offs_m[:, None] * C + c_idx[None, :], mask=k_ok[:, None], other=0.0)
        w = tl.load(lnw_ptr + c_idx).to(tl.float32)
        b = tl.load(lnb_ptr + c_idx).to(tl.float32)
        x = ((zt.to(tl.float32) - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]).to(zt.dtype).to(WDT)   # [BM, BK]
        wp = tl.load(wpT_ptr + c_idx[:, None] * (2 * C) + offs_n[None, :])                                           # [BK, BN] natural
        wg = tl.load(wgT_ptr + c_idx[:, None] * (2 * C) + offs_n[None, :])
        acc_p = _dot(x, wp, acc_p, PREC)
        acc_g = _dot(x, wg, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = g * acc_p
    if HAS_MASK:
        mk = tl.load(mask_ptr + offs_m, mask=k_ok, other=0.0).to(tl.float32)
        o = o * mk[:, None]
    oT = tl.trans(o)
    optrs = ab_ptr + offs_n[:, None].to(tl.int64) * (Np * Np) + i.to(tl.int64) * Np + offs_k[None, :]
    tl.store(optrs, oT.to(ab_ptr.dtype.element_ty), mask=k_ok[None, :])


@triton.jit
def _k_out4(x_ptr, z_ptr, mo_ptr, ro_ptr, mi_ptr, ri_ptr, wpoT_ptr, wgoT_ptr, lniw_ptr, lnib_ptr, lnow_ptr, lnob_ptr, out_ptr,
            N, Np, C, NJT, eps,
            RESIDUAL: tl.constexpr, PREC: tl.constexpr, WDT: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NJT
    jt = pid - i * NJT
    offs_j = jt * BM + tl.arange(0, BM)
    j_ok = offs_j < N
    offs_m = i.to(tl.int64) * N + offs_j
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, BK)
    NpNp = Np * Np
    xbase = x_ptr + i.to(tl.int64) * Np + offs_j[None, :]
    mean_o = tl.load(mo_ptr + offs_m, mask=j_ok, other=0.0)
    rstd_o = tl.load(ro_ptr + offs_m, mask=j_ok, other=0.0)
    mean_i = tl.load(mi_ptr + offs_m, mask=j_ok, other=0.0)
    rstd_i = tl.load(ri_ptr + offs_m, mask=j_ok, other=0.0)
    acc_p = tl.zeros((BM, BN), dtype=tl.float32)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in tl.static_range(NK):
        c_idx = kk * BK + offs_c
        xt = tl.load(xbase + c_idx[:, None].to(tl.int64) * NpNp, mask=j_ok[None, :], other=0.0)                       # [BK, BM] natural (j contiguous)
        wo = tl.load(lnow_ptr + c_idx).to(tl.float32)
        bo = tl.load(lnob_ptr + c_idx).to(tl.float32)
        xo = ((xt.to(tl.float32) - mean_o[None, :]) * rstd_o[None, :] * wo[:, None] + bo[:, None]).to(xt.dtype).to(WDT)
        zt = tl.load(z_ptr + offs_m[:, None] * C + c_idx[None, :], mask=j_ok[:, None], other=0.0)                     # [BM, BK] natural
        wi = tl.load(lniw_ptr + c_idx).to(tl.float32)
        bi = tl.load(lnib_ptr + c_idx).to(tl.float32)
        xi = ((zt.to(tl.float32) - mean_i[:, None]) * rstd_i[:, None] * wi[None, :] + bi[None, :]).to(zt.dtype).to(WDT)
        wpo = tl.load(wpoT_ptr + c_idx[:, None] * C + offs_n[None, :])                                                 # [BK, BN] natural
        wgo = tl.load(wgoT_ptr + c_idx[:, None] * C + offs_n[None, :])
        acc_p = _dot(tl.trans(xo), wpo, acc_p, PREC)
        acc_g = _dot(xi, wgo, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = (g * acc_p).to(x_ptr.dtype.element_ty)
    if RESIDUAL:
        zr = tl.load(z_ptr + offs_m[:, None] * C + offs_n[None, :], mask=j_ok[:, None], other=0.0)
        o = (o.to(tl.float32) + zr.to(tl.float32)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_m[:, None] * C + offs_n[None, :], o.to(out_ptr.dtype.element_ty), mask=j_ok[:, None])


# ----------------------------------------------------------------------------------------------------------------------
# v5: WEIGHT-STATIONARY persistent GEMMs.  Diagnosis (@705): K_A4 moves ~8.7 GB of weight tiles through L2 (each of the
# 67,680 programs re-loads its [C, BN] weight slices) + 2 GB of z re-reads = 10.7 GB in 1.94 ms = 5.5 TB/s = the L2 bandwidth: the
# kernel is L2-bound on weight re-reads, not on MMA.  v5 loads the two [C, BN] weight slices ONCE per program and streams the
# M tiles through a persistent loop (grid = (C_out/BN, SPLIT)); weight traffic drops to (C_out/BN)*SPLIT slices (~34 MB) and the
# z re-read factor is C_out/BN.  SPLIT is a pinned table constant (never the runtime SM count, so a co-tenant cannot change it); the output of
# every tile is independent of which program computes it, so the result is bit-exact independent of SPLIT by construction.
# Stats (mean/rstd per row) come from the v4 pre-passes.  Same arithmetic as v4 (single K=C dot per tile, fp32 accumulate).
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _k_proj5(z_ptr, mean_ptr, rstd_ptr, wpT_ptr, wgT_ptr, lnw_ptr, lnb_ptr, mask_ptr, ab_ptr,
             N, Np, C, NKT, NT, eps,
             HAS_MASK: tl.constexpr, PREC: tl.constexpr, WDT: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, CK: tl.constexpr, SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, CK)
    wp = tl.load(wpT_ptr + offs_c[:, None] * (2 * C) + offs_n[None, :])        # [CK, BN] stationary
    wg = tl.load(wgT_ptr + offs_c[:, None] * (2 * C) + offs_n[None, :])
    w = tl.load(lnw_ptr + offs_c).to(tl.float32)
    b = tl.load(lnb_ptr + offs_c).to(tl.float32)
    NpNp = Np * Np
    for t in range(pid_s, NT, SPLIT):
        i = t // NKT
        kt = t - i * NKT
        offs_k = kt * BM + tl.arange(0, BM)
        k_ok = offs_k < N
        offs_m = i.to(tl.int64) * N + offs_k
        zt = tl.load(z_ptr + offs_m[:, None] * C + offs_c[None, :], mask=k_ok[:, None], other=0.0)     # [BM, CK]
        mean = tl.load(mean_ptr + offs_m, mask=k_ok, other=0.0)
        rstd = tl.load(rstd_ptr + offs_m, mask=k_ok, other=0.0)
        x = ((zt.to(tl.float32) - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]).to(zt.dtype).to(WDT)
        acc_p = _dot(x, wp, tl.zeros((BM, BN), dtype=tl.float32), PREC)
        acc_g = _dot(x, wg, tl.zeros((BM, BN), dtype=tl.float32), PREC)
        g = 1.0 / (1.0 + tl.exp(-acc_g))
        o = g * acc_p
        if HAS_MASK:
            mk = tl.load(mask_ptr + offs_m, mask=k_ok, other=0.0).to(tl.float32)
            o = o * mk[:, None]
        oT = tl.trans(o)
        optrs = ab_ptr + offs_n[:, None].to(tl.int64) * NpNp + i.to(tl.int64) * Np + offs_k[None, :]
        tl.store(optrs, oT.to(ab_ptr.dtype.element_ty), mask=k_ok[None, :])


@triton.jit
def _k_out5(x_ptr, z_ptr, mo_ptr, ro_ptr, mi_ptr, ri_ptr, wpoT_ptr, wgoT_ptr, lniw_ptr, lnib_ptr, lnow_ptr, lnob_ptr, out_ptr,
            N, Np, C, NJT, NT, eps,
            RESIDUAL: tl.constexpr, PREC: tl.constexpr, WDT: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, CK: tl.constexpr, SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, CK)
    wpo = tl.load(wpoT_ptr + offs_c[:, None] * C + offs_n[None, :])             # [CK, BN] stationary
    wgo = tl.load(wgoT_ptr + offs_c[:, None] * C + offs_n[None, :])
    wi = tl.load(lniw_ptr + offs_c).to(tl.float32)
    bi = tl.load(lnib_ptr + offs_c).to(tl.float32)
    wo = tl.load(lnow_ptr + offs_c).to(tl.float32)
    bo = tl.load(lnob_ptr + offs_c).to(tl.float32)
    NpNp = Np * Np
    for t in range(pid_s, NT, SPLIT):
        i = t // NJT
        jt = t - i * NJT
        offs_j = jt * BM + tl.arange(0, BM)
        j_ok = offs_j < N
        offs_m = i.to(tl.int64) * N + offs_j
        xt = tl.load(x_ptr + offs_c[:, None].to(tl.int64) * NpNp + i.to(tl.int64) * Np + offs_j[None, :], mask=j_ok[None, :], other=0.0)   # [CK, BM]
        mean_o = tl.load(mo_ptr + offs_m, mask=j_ok, other=0.0)
        rstd_o = tl.load(ro_ptr + offs_m, mask=j_ok, other=0.0)
        xo = ((xt.to(tl.float32) - mean_o[None, :]) * rstd_o[None, :] * wo[:, None] + bo[:, None]).to(xt.dtype).to(WDT)
        zt = tl.load(z_ptr + offs_m[:, None] * C + offs_c[None, :], mask=j_ok[:, None], other=0.0)                                         # [BM, CK]
        mean_i = tl.load(mi_ptr + offs_m, mask=j_ok, other=0.0)
        rstd_i = tl.load(ri_ptr + offs_m, mask=j_ok, other=0.0)
        xi = ((zt.to(tl.float32) - mean_i[:, None]) * rstd_i[:, None] * wi[None, :] + bi[None, :]).to(zt.dtype).to(WDT)
        acc_p = _dot(tl.trans(xo), wpo, tl.zeros((BM, BN), dtype=tl.float32), PREC)
        acc_g = _dot(xi, wgo, tl.zeros((BM, BN), dtype=tl.float32), PREC)
        g = 1.0 / (1.0 + tl.exp(-acc_g))
        o = (g * acc_p).to(x_ptr.dtype.element_ty)
        if RESIDUAL:
            zr = tl.load(z_ptr + offs_m[:, None] * C + offs_n[None, :], mask=j_ok[:, None], other=0.0)
            o = (o.to(tl.float32) + zr.to(tl.float32)).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs_m[:, None] * C + offs_n[None, :], o.to(out_ptr.dtype.element_ty), mask=j_ok[:, None])


# ----------------------------------------------------------------------------------------------------------------------
# v6: diagnostic finding (@705): an LN prologue computed in registers costs 0.25 ms in K_A and 0.8 ms in K_C (the LN'd
# operand must round-trip registers -> shared memory and loses the async load pipeline; tl.trans of a COMPUTED tile is the worst
# case), while the SAME kernels with pure global->smem operand loads run at 1.06 / 0.75 ms.  v6 therefore materialises the two LN
# outputs exactly as stock cuEq does (x_in = LN_in(z) -> [M, C]; x_out = LN_out(x) transposed -> [M, C], both in the compute dtype,
# identical roundings) in two fused stats+normalise passes that read their input ONCE, and runs K_A / K_C as pure dual gated GEMMs
# whose operands are plain loads (tl.trans on a LOADED tile is free: the MMA reads shared memory with the other descriptor).
# Numerics: bit-identical to v4 (same fp32 two-pass stats, same rounding points, same dot order) -> checked in debug_stages.
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _k_ln_rows(z_ptr, lnw_ptr, lnb_ptr, xin_ptr, M, C, eps, BM: tl.constexpr, CK: tl.constexpr):
    # x_in[m, :] = LN(z[m, :]) rounded to z.dtype then to x_in's dtype (cuEq: LN out in x.dtype, then cast to the GEMM dtype)
    # CK = next power of two >= C (C = 128 / 256 / 384 ...); channels >= C are masked out of the statistics.
    pid = tl.program_id(0)
    offs_m = pid.to(tl.int64) * BM + tl.arange(0, BM)
    m_ok = offs_m < M
    offs_c = tl.arange(0, CK)
    c_ok = offs_c < C
    ok = m_ok[:, None] & c_ok[None, :]
    zt = tl.load(z_ptr + offs_m[:, None] * C + offs_c[None, :], mask=ok, other=0.0)
    zf = zt.to(tl.float32)
    mean = tl.sum(zf, 1) / C
    d = tl.where(ok, zf - mean[:, None], 0.0)
    rstd = tl.rsqrt(tl.sum(d * d, 1) / C + eps)
    w = tl.load(lnw_ptr + offs_c, mask=c_ok, other=0.0).to(tl.float32)
    b = tl.load(lnb_ptr + offs_c, mask=c_ok, other=0.0).to(tl.float32)
    x = ((zf - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]).to(zt.dtype).to(xin_ptr.dtype.element_ty)
    tl.store(xin_ptr + offs_m[:, None] * C + offs_c[None, :], x, mask=ok)


@triton.jit
def _k_ln_cols_T(x_ptr, lnw_ptr, lnb_ptr, xo_ptr, N, Np, C, NJT, eps, BM: tl.constexpr, CK: tl.constexpr):
    # x_out[i*N + j, d] = LN_d(x[d, i, j]) : stats over d per (i, j); channel-major in, row-major (d contiguous) out = cuEq layer_norm_transpose dbij->bijd
    pid = tl.program_id(0)
    i = pid // NJT
    jt = pid - i * NJT
    offs_j = jt * BM + tl.arange(0, BM)
    j_ok = offs_j < N
    offs_c = tl.arange(0, CK)
    c_ok = offs_c < C
    ok = c_ok[:, None] & j_ok[None, :]
    xt = tl.load(x_ptr + offs_c[:, None].to(tl.int64) * (Np * Np) + i.to(tl.int64) * Np + offs_j[None, :], mask=ok, other=0.0)   # [CK, BM]
    xf = xt.to(tl.float32)
    mean = tl.sum(xf, 0) / C
    d = tl.where(ok, xf - mean[None, :], 0.0)
    rstd = tl.rsqrt(tl.sum(d * d, 0) / C + eps)
    w = tl.load(lnw_ptr + offs_c, mask=c_ok, other=0.0).to(tl.float32)
    b = tl.load(lnb_ptr + offs_c, mask=c_ok, other=0.0).to(tl.float32)
    xo = ((xf - mean[None, :]) * rstd[None, :] * w[:, None] + b[:, None]).to(xt.dtype).to(xo_ptr.dtype.element_ty)   # [CK, BM]
    offs_m = i.to(tl.int64) * N + offs_j
    tl.store(xo_ptr + offs_m[None, :] * C + offs_c[:, None], xo, mask=ok)


@triton.jit
def _k_proj6(xin_ptr, wp_ptr, wg_ptr, mask_ptr, ab_ptr, N, Np, C, NKT,
             HAS_MASK: tl.constexpr, PREC: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr):
    # accT[n, k] = sum_c W[n, c] * x_in[i*N + k, c]  (W natural [BN, BK] from [2C, C]; x natural [BM, BK] -> trans, free) ; store ab[n, i, k]
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NKT
    kt = pid - i * NKT
    offs_k = kt * BM + tl.arange(0, BM)
    k_ok = offs_k < N
    offs_m = i.to(tl.int64) * N + offs_k
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, BK)
    acc_p = tl.zeros((BN, BM), dtype=tl.float32)
    acc_g = tl.zeros((BN, BM), dtype=tl.float32)
    for kk in tl.static_range(NK):
        c_idx = kk * BK + offs_c
        xt = tl.load(xin_ptr + offs_m[:, None] * C + c_idx[None, :], mask=k_ok[:, None], other=0.0)   # [BM, BK]
        wp = tl.load(wp_ptr + offs_n[:, None] * C + c_idx[None, :])                                  # [BN, BK]
        wg = tl.load(wg_ptr + offs_n[:, None] * C + c_idx[None, :])
        xT = tl.trans(xt)
        acc_p = _dot(wp, xT, acc_p, PREC)
        acc_g = _dot(wg, xT, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = g * acc_p
    if HAS_MASK:
        mk = tl.load(mask_ptr + offs_m, mask=k_ok, other=0.0).to(tl.float32)
        o = o * mk[None, :]
    optrs = ab_ptr + offs_n[:, None].to(tl.int64) * (Np * Np) + i.to(tl.int64) * Np + offs_k[None, :]
    tl.store(optrs, o.to(ab_ptr.dtype.element_ty), mask=k_ok[None, :])


@triton.jit
def _k_out6(xo_ptr, xin_ptr, z_ptr, wpoT_ptr, wgoT_ptr, out_ptr, M, C,
            RESIDUAL: tl.constexpr, PREC: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr):
    # out[m, n] = sigmoid(sum_c x_in[m, c] Wgo[n, c]) * (sum_c x_out[m, c] Wpo[n, c])  (+ z[m, n])
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m.to(tl.int64) * BM + tl.arange(0, BM)
    m_ok = offs_m < M
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, BK)
    acc_p = tl.zeros((BM, BN), dtype=tl.float32)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in tl.static_range(NK):
        c_idx = kk * BK + offs_c
        xo = tl.load(xo_ptr + offs_m[:, None] * C + c_idx[None, :], mask=m_ok[:, None], other=0.0)     # [BM, BK]
        xi = tl.load(xin_ptr + offs_m[:, None] * C + c_idx[None, :], mask=m_ok[:, None], other=0.0)
        wpo = tl.load(wpoT_ptr + c_idx[:, None] * C + offs_n[None, :])                                  # [BK, BN]
        wgo = tl.load(wgoT_ptr + c_idx[:, None] * C + offs_n[None, :])
        acc_p = _dot(xo, wpo, acc_p, PREC)
        acc_g = _dot(xi, wgo, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = (g * acc_p).to(xo_ptr.dtype.element_ty)
    if RESIDUAL:
        zr = tl.load(z_ptr + offs_m[:, None] * C + offs_n[None, :], mask=m_ok[:, None], other=0.0)
        o = (o.to(tl.float32) + zr.to(tl.float32)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_m[:, None] * C + offs_n[None, :], o.to(out_ptr.dtype.element_ty), mask=m_ok[:, None])


# ----------------------------------------------------------------------------------------------------------------------
# v8 A' variants: (a) SWAP form = the reference A' kernel's orientation: x tile [BM(k), BK] is the A operand (natural, no trans), W^T [BK, BN] natural
#     (pre-transposed cache), acc [BM, BN], store ab[n, i, k] through tl.trans(acc) (k contiguous).  Same k-sequential fp32 dot.
#     (b) plane STRIDE padding: ab planes may have a row stride XS and plane stride PLANE passed explicitly (PLANE = XS*XS + 64 avoids
#     DRAM partition camping at N=813); cuBLAS bmm accepts the strided view.
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["XS", "PLANE"])
def _k_proj8(xin_ptr, wpT_ptr, wgT_ptr, mask_ptr, ab_ptr, N, XS, PLANE, C, NKT,
             HAS_MASK: tl.constexpr, PREC: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NKT
    kt = pid - i * NKT
    offs_k = kt * BM + tl.arange(0, BM)
    k_ok = offs_k < N
    offs_m = i.to(tl.int64) * N + offs_k
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, BK)
    acc_p = tl.zeros((BM, BN), dtype=tl.float32)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in tl.static_range(NK):
        c_idx = kk * BK + offs_c
        xt = tl.load(xin_ptr + offs_m[:, None] * C + c_idx[None, :], mask=k_ok[:, None], other=0.0)   # [BM, BK] natural
        wp = tl.load(wpT_ptr + c_idx[:, None] * (2 * C) + offs_n[None, :])                          # [BK, BN] natural (W^T cache)
        wg = tl.load(wgT_ptr + c_idx[:, None] * (2 * C) + offs_n[None, :])
        acc_p = _dot(xt, wp, acc_p, PREC)
        acc_g = _dot(xt, wg, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = g * acc_p
    if HAS_MASK:
        mk = tl.load(mask_ptr + offs_m, mask=k_ok, other=0.0).to(tl.float32)
        o = o * mk[:, None]
    oT = tl.trans(o)                                                                                  # [BN, BM]
    optrs = ab_ptr + offs_n[:, None].to(tl.int64) * PLANE + i.to(tl.int64) * XS + offs_k[None, :]
    tl.store(optrs, oT.to(ab_ptr.dtype.element_ty), mask=k_ok[None, :])


@triton.jit(do_not_specialize=["XS", "PLANE"])
def _k_proj6s(xin_ptr, wp_ptr, wg_ptr, mask_ptr, ab_ptr, N, XS, PLANE, C, NKT,
              HAS_MASK: tl.constexpr, PREC: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr):
    # v6 orientation (W natural [BN,BK], trans(x)) with explicit plane/row strides
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NKT
    kt = pid - i * NKT
    offs_k = kt * BM + tl.arange(0, BM)
    k_ok = offs_k < N
    offs_m = i.to(tl.int64) * N + offs_k
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_c = tl.arange(0, BK)
    acc_p = tl.zeros((BN, BM), dtype=tl.float32)
    acc_g = tl.zeros((BN, BM), dtype=tl.float32)
    for kk in tl.static_range(NK):
        c_idx = kk * BK + offs_c
        xt = tl.load(xin_ptr + offs_m[:, None] * C + c_idx[None, :], mask=k_ok[:, None], other=0.0)
        wp = tl.load(wp_ptr + offs_n[:, None] * C + c_idx[None, :])
        wg = tl.load(wg_ptr + offs_n[:, None] * C + c_idx[None, :])
        xT = tl.trans(xt)
        acc_p = _dot(wp, xT, acc_p, PREC)
        acc_g = _dot(wg, xT, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = g * acc_p
    if HAS_MASK:
        mk = tl.load(mask_ptr + offs_m, mask=k_ok, other=0.0).to(tl.float32)
        o = o * mk[None, :]
    optrs = ab_ptr + offs_n[:, None].to(tl.int64) * PLANE + i.to(tl.int64) * XS + offs_k[None, :]
    tl.store(optrs, o.to(ab_ptr.dtype.element_ty), mask=k_ok[None, :])


# ----------------------------------------------------------------------------------------------------------------------
# v9: same arithmetic as v6 (k-sequential fp32 dual dot, sigmoid, one bf16 rounding) with two codegen changes taken from a reference A' kernel
# (which reaches 261 TF/s with an LN prologue inside): (1) the K loop is a RUNTIME `for k0 in range(0, C, BK)` (scf.for) so Triton's
# software pipeliner can multi-buffer the operand loads with num_stages — `tl.static_range` unrolls at the AST level and leaves no loop to
# pipeline; (2) C / NOUT are constexpr (compile-time strides) and the weight tile is read with transposed addressing [BK, BN] straight from
# the row-major [NOUT, C] weight (no tl.trans, no pre-transposed cache).  x is the natural A operand, acc [BM, BN].
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["XS", "PLANE"])
def _k_proj9(xin_ptr, wp_ptr, wg_ptr, mask_ptr, ab_ptr, N, XS, PLANE, NKT,
             C: tl.constexpr, HAS_MASK: tl.constexpr, PREC: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NKT
    kt = pid - i * NKT
    offs_k = kt * BM + tl.arange(0, BM)
    k_ok = offs_k < N
    rows = (i * N + offs_k).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc_p = tl.zeros((BM, BN), dtype=tl.float32)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, C, BK):
        offs_c = k0 + tl.arange(0, BK)
        xt = tl.load(xin_ptr + rows[:, None] * C + offs_c[None, :], mask=k_ok[:, None], other=0.0)     # [BM, BK]
        wp = tl.load(wp_ptr + offs_n[None, :] * C + offs_c[:, None])                                 # [BK, BN] (transposed addressing)
        wg = tl.load(wg_ptr + offs_n[None, :] * C + offs_c[:, None])
        acc_p = _dot(xt, wp, acc_p, PREC)
        acc_g = _dot(xt, wg, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = g * acc_p
    if HAS_MASK:
        mk = tl.load(mask_ptr + rows, mask=k_ok, other=0.0).to(tl.float32)
        o = o * mk[:, None]
    dst = ab_ptr + offs_n[None, :].to(tl.int64) * PLANE + i.to(tl.int64) * XS + offs_k[:, None].to(tl.int64)
    tl.store(dst, o.to(ab_ptr.dtype.element_ty), mask=k_ok[:, None])


@triton.jit
def _k_out9(xo_ptr, xin_ptr, z_ptr, wpo_ptr, wgo_ptr, out_ptr, M,
            C: tl.constexpr, RESIDUAL: tl.constexpr, PREC: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    m_ok = offs_m < M
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc_p = tl.zeros((BM, BN), dtype=tl.float32)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, C, BK):
        offs_c = k0 + tl.arange(0, BK)
        xo = tl.load(xo_ptr + offs_m[:, None] * C + offs_c[None, :], mask=m_ok[:, None], other=0.0)
        xi = tl.load(xin_ptr + offs_m[:, None] * C + offs_c[None, :], mask=m_ok[:, None], other=0.0)
        wpo = tl.load(wpo_ptr + offs_n[None, :] * C + offs_c[:, None])                                # [BK, BN] from row-major [C, C]
        wgo = tl.load(wgo_ptr + offs_n[None, :] * C + offs_c[:, None])
        acc_p = _dot(xo, wpo, acc_p, PREC)
        acc_g = _dot(xi, wgo, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = (g * acc_p).to(xo_ptr.dtype.element_ty)
    if RESIDUAL:
        zr = tl.load(z_ptr + offs_m[:, None] * C + offs_n[None, :], mask=m_ok[:, None], other=0.0)
        # the residual add as an explicit round-to-nearest PTX add (never contracted into an FMA with the g*p product): the launch keeps fp fusion ON
        o = tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [o.to(tl.float32), zr.to(tl.float32)], dtype=tl.float32, is_pure=True, pack=1).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_m[:, None] * C + offs_n[None, :], o.to(out_ptr.dtype.element_ty), mask=m_ok[:, None])


def plane_xs(N, pad):
    """Row stride of the channel-major planes.  pad=1: XS=N (EXACT layout, the stock einsum's lda).  pad>1: plain ceil to `pad`.
    No '+8 when ceil8 % 16 == 0' rule: the 1.5-1.7x slower A' store once seen on 16-aligned strides
    is Triton's integer-argument specialisation (a 'divisible-by-16' variant of the kernel with a slower
    vectorised channel-major store), not the hardware: with do_not_specialize=["XS", "PLANE"] on the A' kernels the store runs 0.99-1.00 ms @705 for
    every XS in {712..768}, and cuBLAS + S_out PREFER 16-aligned rows (B 0.353 vs 0.446 ms @705; S_out 0.316 vs 0.533).  FAST/cuBLAS therefore uses pad 16."""
    if pad <= 1:
        return N
    return ((N + pad - 1) // pad) * pad


# ----------------------------------------------------------------------------------------------------------------------
# v10 A' with the LN_in prologue fused (FAST mode only): per-row mean/rstd from the v4 _k_rowstats pre-pass (fp32, two-pass), then inside
# the pipelined K loop x_hat = ((z - mean) * rstd * w + b) rounded to the compute dtype = exactly the x_in that _k_ln_rows would have stored
# (same fp32 expression, same single rounding) -> bit-exact to v9-on-materialised-x_in by construction; removes the x_in write + read (2 units).
# The reference A' kernel (in-kernel LN, runtime loop) runs 1.05 ms @705 = our v9 1.02 with a materialised x_in: this fuses the 0.20 ms S_in pass.
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["XS", "PLANE"])
def _k_proj10(z_ptr, mean_ptr, rstd_ptr, lnw_ptr, lnb_ptr, wp_ptr, wg_ptr, mask_ptr, ab_ptr, N, XS, PLANE, NKT,
              C: tl.constexpr, HAS_MASK: tl.constexpr, PREC: tl.constexpr, CDT: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NKT
    kt = pid - i * NKT
    offs_k = kt * BM + tl.arange(0, BM)
    k_ok = offs_k < N
    rows = (i * N + offs_k).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mean = tl.load(mean_ptr + rows, mask=k_ok, other=0.0)
    rstd = tl.load(rstd_ptr + rows, mask=k_ok, other=0.0)
    acc_p = tl.zeros((BM, BN), dtype=tl.float32)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, C, BK):
        offs_c = k0 + tl.arange(0, BK)
        zt = tl.load(z_ptr + rows[:, None] * C + offs_c[None, :], mask=k_ok[:, None], other=0.0).to(tl.float32)
        w = tl.load(lnw_ptr + offs_c).to(tl.float32)
        b = tl.load(lnb_ptr + offs_c).to(tl.float32)
        xt = ((zt - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]).to(z_ptr.dtype.element_ty).to(CDT)   # same two roundings as _k_ln_rows
        wp = tl.load(wp_ptr + offs_n[None, :] * C + offs_c[:, None])
        wg = tl.load(wg_ptr + offs_n[None, :] * C + offs_c[:, None])
        acc_p = _dot(xt, wp, acc_p, PREC)
        acc_g = _dot(xt, wg, acc_g, PREC)
    g = 1.0 / (1.0 + tl.exp(-acc_g))
    o = g * acc_p
    if HAS_MASK:
        mk = tl.load(mask_ptr + rows, mask=k_ok, other=0.0).to(tl.float32)
        o = o * mk[:, None]
    dst = ab_ptr + offs_n[None, :].to(tl.int64) * PLANE + i.to(tl.int64) * XS + offs_k[:, None].to(tl.int64)
    tl.store(dst, o.to(ab_ptr.dtype.element_ty), mask=k_ok[:, None])


# ----------------------------------------------------------------------------------------------------------------------
# v14 A': interleaved single-accumulator
# projection — ONE tl.dot per K step over a row-interleaved [4C, C] weight (row 2n = p_in[n], row 2n+1 = g_in[n]); the (p, g) pair of each
# output channel lands in adjacent accumulator columns (tl.reshape + tl.split).  Same arithmetic as _k_proj9 (every output column is the same
# K-ordered fp32 dot, fp32 sigmoid, one rounding): bit-exact to _k_proj9 in 525/525 measured cells.  A' 0.84-0.89 ms @705 (309/294 TF/s on
# FAST/EXACT planes) vs v9 1.04-1.05.  XSA is passed as XS//8 on aligned planes (do_not_specialize keeps the store out of the slow specialised class).
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["XSA"])
def _k_proj14(xin_ptr, wpg_ptr, mask_ptr, ab_ptr, N, XSA, NKT,
              C: tl.constexpr, HAS_MASK: tl.constexpr, PREC: tl.constexpr, ALIGNED: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Interleaved single-accumulator A' (0.86 ms GEMM @705 = 302 TF/s): the weight is the row-interleaved [4C, C] matrix
    # (row 2n = p_in[n], row 2n+1 = g_in[n]; a pure layout copy), so ONE tl.dot per K step with N = BN covers BN/2 output channels and the
    # (p, g) pair of each channel lands in adjacent accumulator columns (tl.reshape + tl.split).  Same arithmetic as _k_proj9: every output
    # column is the same K-ordered dot product, fp32 accumulate, fp32 sigmoid, one rounding to the plane dtype.
    # ALIGNED (FAST planes, XS % 8 == 0): XSA = XS // 8, transposed 16-byte-aligned store; else (EXACT planes, XS = N): XSA = XS, _k_proj9 store.
    # grid: (4C/BN, N*NKT)
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    i = pid // NKT
    kt = pid - i * NKT
    if ALIGNED:
        XS = XSA * 8
    else:
        XS = XSA
    PLANE = XS * XS
    offs_k = kt * BM + tl.arange(0, BM)
    k_ok = offs_k < N
    rows = (i * N + offs_k).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, C, BK):
        offs_c = k0 + tl.arange(0, BK)
        xt = tl.load(xin_ptr + rows[:, None] * C + offs_c[None, :], mask=k_ok[:, None], other=0.0)     # [BM, BK]
        wt = tl.load(wpg_ptr + offs_n[None, :] * C + offs_c[:, None])                                # [BK, BN] interleaved (p, g) columns
        acc = _dot(xt, wt, acc, PREC)
    pg = tl.reshape(acc, (BM, BN // 2, 2))
    p, gg = tl.split(pg)
    g = 1.0 / (1.0 + tl.exp(-gg))
    o = g * p
    if HAS_MASK:
        mk = tl.load(mask_ptr + rows, mask=k_ok, other=0.0).to(tl.float32)
        o = o * mk[:, None]
    offs_no = pid_n * (BN // 2) + tl.arange(0, BN // 2)
    if ALIGNED:
        oT = tl.trans(o.to(ab_ptr.dtype.element_ty))                                                 # [BN/2, BM], k contiguous
        dst = ab_ptr + offs_no[:, None].to(tl.int64) * PLANE + i.to(tl.int64) * XS + offs_k[None, :]
        if (kt + 1) * BM <= N:
            tl.store(dst, oT)
        else:
            tl.store(dst, oT, mask=k_ok[None, :])
    else:
        dst = ab_ptr + offs_no[None, :].to(tl.int64) * PLANE + i.to(tl.int64) * XS + offs_k[:, None].to(tl.int64)
        tl.store(dst, o.to(ab_ptr.dtype.element_ty), mask=k_ok[:, None])


def _wPG(w):
    """Row-interleaved [4C, C] projection weight (row 2n = p_in[n], row 2n+1 = g_in[n]) cached in the weight dict (pure layout copy)."""
    if "pg_in_I" not in w:
        w["pg_in_I"] = torch.stack([w["p_in"], w["g_in"]], dim=1).reshape(-1, w["p_in"].shape[1]).contiguous()
    return w["pg_in_I"]


# ----------------------------------------------------------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------------------------------------------------------
_TL_DT = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}
DEFAULT_CFG = dict(
    A=dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=3),
    B=dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=3, DMAJOR=True),
    C=dict(BM=64, BN=256, BK=64, num_warps=8, num_stages=3),
    pad=64,
)


def fit_cfg(cfg, C):
    """Clamp a config to the channel width C (tiles must divide C / 2C); pad = largest contraction tile so every K_B tile is full."""
    import copy
    c = copy.deepcopy(cfg)
    c["A"]["BN"] = min(c["A"]["BN"], 4 * C if c["A"].get("v", 2) == 14 else 2 * C); c["A"]["BK"] = min(c["A"].get("BK", 64), C)
    c["C"]["BN"] = min(c["C"]["BN"], C); c["C"]["BK"] = min(c["C"].get("BK", 64), C)
    if c.get("contract", "triton") == "triton":
        c["pad"] = max(c.get("pad", 64), c["B"]["BM"], c["B"]["BN"], c["B"]["BK"])
    else:
        c["pad"] = c.get("pad", 16)           # cuBLAS contraction: any alignment works; 16 is the fast class for B and S_out once A' is despecialised; pad 1 = EXACT layout
    return c


def default_cfg(C):
    return fit_cfg(DEFAULT_CFG, C)
_BUF = {}   # diagnostic retention only: trimul_forward(key=<label>) keeps that call's planes under (key, C, Np, dtype, device)


def _prec_for(w_dtype: torch.dtype, act_dtype: torch.dtype):
    """cuEq rule: the dot runs in the weight dtype; fp32 -> tf32 (allow_tf32) or tf32x3."""
    if w_dtype in (torch.bfloat16, torch.float16):
        return 0
    return 1 if torch.backends.cuda.matmul.allow_tf32 else 2


def _contract_prec(ab_dtype: torch.dtype):
    if ab_dtype in (torch.bfloat16, torch.float16):
        return 0
    return 1 if torch.backends.cuda.matmul.allow_tf32 else 2


def _planes(C, Np, N, dtype, device, key, stats, ln):
    """The planes of one call: ab (a, b) and x (the contraction) in the compute dtype, st (v4/v5 row statistics: mean_i, rstd_i, mean_o,
    rstd_o) and xin / xo (v6+: materialised LN_in(z) and LN_out(x)^T) only when a selected kernel variant reads them.
    key None (the served paths): allocated from the caching allocator and released with the call — nothing is held between calls, so
    the process footprint outside a TriMul call is the stock's.  key given (diagnostics, timing runs): kept in _BUF and reused by the next
    call of the same key and shape.  ab's padded tails must be exact zeros (K_A writes [:, :N, :N] only): zeroed when Np > N; with
    Np == N every element is written."""
    k = None if key is None else (key, C, Np, dtype, str(device))
    ent = _BUF.get(k) if k is not None else None
    if ent is None:
        ab = torch.empty(2 * C, Np, Np, dtype=dtype, device=device)
        if Np != N:
            ab[:, N:, :].zero_(); ab[:, :, N:].zero_()
        ent = [ab, torch.empty(C, Np, Np, dtype=dtype, device=device), N, None, None, None]
        if k is not None:
            _BUF[k] = ent
    elif ent[2] != N:
        ent[0].zero_(); ent[2] = N
    if stats and ent[3] is None:
        ent[3] = torch.empty(4, Np * Np, dtype=torch.float32, device=device)
    if ln and ent[4] is None:
        ent[4] = torch.empty(Np * Np, C, dtype=dtype, device=device); ent[5] = torch.empty(Np * Np, C, dtype=dtype, device=device)
    return ent[0], ent[1], ent[3], ent[4], ent[5]


CUDA_GRID_Y_MAX = 65535          # CUDA's limit on grid axis 1 (and 2); axis 0 allows 2**31 - 1


def stage_grid_y(stage, N, tile):
    """The CUDA grid axis-1 extent trimul_forward launches for stage ``'A'`` / ``'C'`` at N tokens with the stage's tile dict — the ONE statement of
    the token-row tiling both the launch gate (:func:`launch_grid_y`) and the tile-height schedule (``trimul._select_cfg``) read: stage A tiles each
    of the N token rows in ceil(N / BM) programs (``grid_a``, every projection kernel), stage C's ``_k_out6`` / ``_k_out9`` tile the N*N flattened
    rows in ceil(N*N / BM) programs and the older epilogues (``grid_c``) N * ceil(N / BM); the split variants (v5) launch SPLIT programs there.
    Pure arithmetic; no device needed."""
    v, BM = tile.get("v", 2), tile["BM"]
    if v == 5:
        return tile.get("SPLIT", 128)
    if stage == "C" and v in (6, 7, 9):
        return -(-(N * N) // BM)
    return N * -(-N // BM)


def launch_grid_y(N, C, cfg):
    """The largest CUDA grid axis-1 extent trimul_forward launches for (N, C, cfg) (:func:`stage_grid_y` of stages A and C; the contraction's grid
    and the LayerNorm kernels' 1-D grids never bind first).  A call is launchable iff launch_grid_y(N, C, cfg) <= CUDA_GRID_Y_MAX — with a
    64-row stage-A tile N <= 2047, with 128 rows N <= 2849, 256 rows N <= 4095, 512 rows N <= 5632 (``trimul._select_cfg`` schedules the tile
    height by N so the served cells stay launchable through N = 5632).  Pure arithmetic; no device needed."""
    cfg = fit_cfg(dict(cfg), C)
    return max(stage_grid_y("A", N, cfg["A"]), stage_grid_y("C", N, cfg["C"]))


def trimul_forward(z, direction, mask, w, eps=1e-5, residual=False, out=None, cfg=None, contract=None, key=None, cdt=None):
    """z: [N, N, C] (single batch) ; mask: [N, N] or None ; w: dict of packed weights (see pack_weights) ; returns [N, N, C].
    contract: 'triton' (K_B) | 'cublas' (torch.matmul on the same padded buffers, control arm).
    key: None = the planes live for this call only (served paths); a label = kept in _BUF for post-call inspection (see _planes)."""
    N, N2, C = z.shape
    cfg = dict(cfg) if cfg is not None else dict(DEFAULT_CFG)
    if contract is None:
        contract = cfg.get("contract", "triton")
    cfg["contract"] = contract
    cfg = fit_cfg(cfg, C)
    assert N == N2 and z.is_contiguous()
    pad = cfg.get("pad", 64)
    Np = plane_xs(N, pad)
    M = N * N
    dev = z.device
    cdt = cdt or w["p_in"].dtype            # compute dtype = the dtype cuEq's gated GEMMs run in (autocast dtype under autocast, else the weight dtype)
    assert w["p_in"].dtype == cdt and w["p_out"].dtype == cdt, "GEMM weights must be packed in the compute dtype (LN weights stay native)"
    wdt = cdt
    prec = _prec_for(wdt, z.dtype)
    WDT = _TL_DT[wdt]
    cA, cB, cC = cfg["A"], cfg["B"], cfg["C"]
    vA, vC = cA.get("v", 2), cC.get("v", 2)
    ab, x, st, xin, xo = _planes(C, Np, N, cdt, dev, key, stats=(vA in (4, 5) or vC in (4, 5)),
                                 ln=(vA in (6, 7, 9, 14) or vC in (6, 7, 9)) and cfg.get("ln_mode", "fpf") != "stock")
    assert C % cA["BK"] == 0 and ((4 if cA.get("v", 2) == 14 else 2) * C) % cA["BN"] == 0 and C % cC["BK"] == 0 and C % cC["BN"] == 0
    mask_flat = None
    if mask is not None:
        mask_flat = mask.reshape(-1)
        if not mask_flat.is_floating_point():
            mask_flat = mask_flat.to(torch.float32)
        mask_flat = mask_flat.contiguous()
        assert mask_flat.numel() == M
    # ---- K_A
    nkt = triton.cdiv(N, cA["BM"])
    grid_a = ((2 * C) // cA["BN"], N * nkt)
    if (cA.get("v", 2) in (3, 4, 5) or cC.get("v", 2) in (3, 4, 5)) and (C & (C - 1)) != 0:
        raise ValueError("v3/v4/v5 kernels need a power-of-two C (engineering variants); use v2 or v6 for C=%d" % C)
    if cA.get("v", 2) in (6, 7, 9, 14) or cC.get("v", 2) in (6, 7, 9):
        if cfg.get("ln_mode", "fpf") == "stock":
            # EXACT mode: the stock library's own LayerNorm op is called (ln_mode 'stock'); the package's LN kernels sum in another order and are not bitwise equal to it.
            # Stock sequence: LN output in z's dtype (fp32 z under autocast -> fp32 LN out), then cuEq's gated GEMM casts x to the compute dtype
            # (maybe_to(autocast_dtype) under autocast; else x.to(w.element_ty) inside the kernel) -> the same single rounding our _k_ln_rows applies.
            from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose as _lnt
            xin = _lnt(z[None], w["ln_in_w"], w["ln_in_b"], eps=eps, layout="bijd->bijd")[0].reshape(M, C).to(cdt)
        else:
            lnc = cfg.get("LN", dict(BM=64, num_warps=4))
            _k_ln_rows[(triton.cdiv(M, lnc["BM"]),)](z, w["ln_in_w"], w["ln_in_b"], xin, M, C, eps, BM=lnc["BM"], CK=triton.next_power_of_2(C), num_warps=lnc["num_warps"])
    if cA.get("v", 2) == 14:
        aligned = (Np % 8 == 0)
        assert (4 * C) % cA["BN"] == 0
        _k_proj14[((4 * C) // cA["BN"], N * nkt)](xin, _wPG(w), mask_flat if mask_flat is not None else z, ab, N, Np // 8 if aligned else Np, nkt,
                          C=C, HAS_MASK=mask_flat is not None, PREC=prec, ALIGNED=aligned, BM=cA["BM"], BN=cA["BN"], BK=cA["BK"], num_warps=cA["num_warps"], num_stages=cA["num_stages"])
    elif cA.get("v", 2) == 9:
        _k_proj9[grid_a](xin, w["p_in"], w["g_in"], mask_flat if mask_flat is not None else z, ab, N, Np, Np * Np, nkt,
                         C=C, HAS_MASK=mask_flat is not None, PREC=prec, BM=cA["BM"], BN=cA["BN"], BK=cA["BK"], num_warps=cA["num_warps"], num_stages=cA["num_stages"])
    elif cA.get("v", 2) in (6, 7):
        _k_proj6[grid_a](xin, w["p_in"], w["g_in"], mask_flat if mask_flat is not None else z, ab, N, Np, C, nkt,
                         HAS_MASK=mask_flat is not None, PREC=prec, BM=cA["BM"], BN=cA["BN"], BK=cA["BK"], NK=C // cA["BK"], num_warps=cA["num_warps"], num_stages=cA["num_stages"])
    elif cA.get("v", 2) == 5:
        _k_rowstats[(triton.cdiv(M, 64),)](z, st[0], st[1], M, C, eps, BM=64, CK=C, num_warps=4)
        _k_proj5[((2 * C) // cA["BN"], cA.get("SPLIT", 128))](z, st[0], st[1], _wT(w, "p_in"), _wT(w, "g_in"), w["ln_in_w"], w["ln_in_b"], mask_flat if mask_flat is not None else z, ab,
                         N, Np, C, nkt, N * nkt, eps, HAS_MASK=mask_flat is not None, PREC=prec, WDT=WDT,
                         BM=cA["BM"], BN=cA["BN"], CK=C, SPLIT=cA.get("SPLIT", 128), num_warps=cA["num_warps"], num_stages=cA["num_stages"])
    elif cA.get("v", 2) == 4:
        _k_rowstats[(triton.cdiv(M, 64),)](z, st[0], st[1], M, C, eps, BM=64, CK=C, num_warps=4)
        _k_proj4[grid_a](z, st[0], st[1], _wT(w, "p_in"), _wT(w, "g_in"), w["ln_in_w"], w["ln_in_b"], mask_flat if mask_flat is not None else z, ab,
                         N, Np, C, nkt, eps, HAS_MASK=mask_flat is not None, PREC=prec, WDT=WDT,
                         BM=cA["BM"], BN=cA["BN"], BK=cA["BK"], NK=C // cA["BK"], num_warps=cA["num_warps"], num_stages=cA["num_stages"])
    elif cA.get("v", 2) == 3:
        _k_proj3[grid_a](z, _wT(w, "p_in"), _wT(w, "g_in"), w["ln_in_w"], w["ln_in_b"], mask_flat if mask_flat is not None else z, ab,
                         N, Np, C, nkt, eps, HAS_MASK=mask_flat is not None, PREC=prec, WDT=WDT,
                         BM=cA["BM"], BN=cA["BN"], CK=C, num_warps=cA["num_warps"], num_stages=cA["num_stages"])
    else:
        _k_proj[grid_a](z, w["p_in"], w["g_in"], w["ln_in_w"], w["ln_in_b"], mask_flat if mask_flat is not None else z, ab,
                        N, Np, C, nkt, eps, HAS_MASK=mask_flat is not None, PREC=prec, WDT=WDT,
                        BM=cA["BM"], BN=cA["BN"], BK=cA["BK"], NK=C // cA["BK"], num_warps=cA["num_warps"], num_stages=cA["num_stages"])
    a = ab[:C]; b = ab[C:]
    outgoing = direction == "outgoing"
    # ---- K_B
    if contract == "triton":
        cprec = _contract_prec(ab.dtype)
        if cB.get("DMAJOR", True):
            grid_b = (Np // cB["BM"], Np // cB["BN"], C)
        else:
            grid_b = (C, Np // cB["BM"], Np // cB["BN"])
        assert Np % cB["BM"] == 0 and Np % cB["BN"] == 0 and Np % cB["BK"] == 0, "pad must cover the contraction tiles"
        _k_contract[grid_b](a, b, x, Np, OUTGOING=outgoing, PREC=cprec, BM=cB["BM"], BN=cB["BN"], BK=cB["BK"],
                            DMAJOR=cB.get("DMAJOR", True), num_warps=cB["num_warps"], num_stages=cB["num_stages"])
    elif contract == "cublas":
        if outgoing:
            torch.matmul(a, b.transpose(-1, -2), out=x)
        else:
            torch.matmul(a.transpose(-1, -2), b, out=x)
    else:
        raise ValueError(contract)
    # ---- K_C
    if out is None:
        out = torch.empty(z.shape, dtype=(z.dtype if residual else cdt), device=dev)   # stock: kernel output in the compute dtype; z + z_in in z's dtype
    njt = triton.cdiv(N, cC["BM"])
    grid_c = (C // cC["BN"], N * njt)
    if cC.get("v", 2) in (6, 7, 9):
        if cfg.get("ln_mode", "fpf") == "stock":
            assert Np == N, "EXACT mode needs pad=1 (XS = N planes)"
            from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose as _lnt
            xo = _lnt(x[:, None], w["ln_out_w"], w["ln_out_b"], eps=eps, layout="dbij->bijd")[0].reshape(M, C).to(cdt)   # x is already in cdt: no-op cast
        else:
            lnc = cfg.get("LNO", dict(BM=64, num_warps=4))
            _k_ln_cols_T[(N * triton.cdiv(N, lnc["BM"]),)](x, w["ln_out_w"], w["ln_out_b"], xo, N, Np, C, triton.cdiv(N, lnc["BM"]), eps, BM=lnc["BM"], CK=triton.next_power_of_2(C), num_warps=lnc["num_warps"])
        if cC.get("v", 2) == 9:
            # fp32 class: with RESIDUAL and an fp32 output the epilogue `g*p + z` would be contracted into an FMA by the PTX backend under fp
            # fusion (no rounding between the product and the add), which is NOT stock's round(g*p) then + z_in.  The kernel writes that add as an
            # explicit `add.rn.f32` (never contracted), so the launch keeps fp fusion on for the rest of the epilogue: bit-identical to the former
            # fusion-off launch on cc 9.0 and 8.0 (fp32 / fp32-under-autocast / tf32 / bf16 residual cells), and the bf16 kernels are untouched
            # (their outputs round to the compute dtype between the two ops).
            _k_out9[(C // cC["BN"], triton.cdiv(M, cC["BM"]))](xo, xin, z, w["p_out"], w["g_out"], out, M,
                            C=C, RESIDUAL=residual, PREC=prec, BM=cC["BM"], BN=cC["BN"], BK=cC["BK"], num_warps=cC["num_warps"], num_stages=cC["num_stages"])
        else:
            _k_out6[(C // cC["BN"], triton.cdiv(M, cC["BM"]))](xo, xin, z, _wT(w, "p_out"), _wT(w, "g_out"), out, M, C,
                            RESIDUAL=residual, PREC=prec, BM=cC["BM"], BN=cC["BN"], BK=cC["BK"], NK=C // cC["BK"], num_warps=cC["num_warps"], num_stages=cC["num_stages"])
    elif cC.get("v", 2) == 5:
        if cA.get("v", 2) < 4:
            _k_rowstats[(triton.cdiv(M, 64),)](z, st[0], st[1], M, C, eps, BM=64, CK=C, num_warps=4)
        _k_colstats[(N * triton.cdiv(N, 64),)](x, st[2], st[3], N, Np, C, triton.cdiv(N, 64), eps, BM=64, CK=C, num_warps=4)
        _k_out5[(C // cC["BN"], cC.get("SPLIT", 128))](x, z, st[2], st[3], st[0], st[1], _wT(w, "p_out"), _wT(w, "g_out"), w["ln_in_w"], w["ln_in_b"], w["ln_out_w"], w["ln_out_b"], out,
                        N, Np, C, njt, N * njt, eps, RESIDUAL=residual, PREC=prec, WDT=WDT,
                        BM=cC["BM"], BN=cC["BN"], CK=C, SPLIT=cC.get("SPLIT", 128), num_warps=cC["num_warps"], num_stages=cC["num_stages"])
    elif cC.get("v", 2) == 4:
        if cA.get("v", 2) < 4:
            _k_rowstats[(triton.cdiv(M, 64),)](z, st[0], st[1], M, C, eps, BM=64, CK=C, num_warps=4)
        _k_colstats[(N * triton.cdiv(N, 64),)](x, st[2], st[3], N, Np, C, triton.cdiv(N, 64), eps, BM=64, CK=C, num_warps=4)
        _k_out4[grid_c](x, z, st[2], st[3], st[0], st[1], _wT(w, "p_out"), _wT(w, "g_out"), w["ln_in_w"], w["ln_in_b"], w["ln_out_w"], w["ln_out_b"], out,
                        N, Np, C, njt, eps, RESIDUAL=residual, PREC=prec, WDT=WDT,
                        BM=cC["BM"], BN=cC["BN"], BK=cC["BK"], NK=C // cC["BK"], num_warps=cC["num_warps"], num_stages=cC["num_stages"])
    elif cC.get("v", 2) == 3:
        _k_out3[grid_c](x, z, _wT(w, "p_out"), _wT(w, "g_out"), w["ln_in_w"], w["ln_in_b"], w["ln_out_w"], w["ln_out_b"], out,
                        N, Np, C, njt, eps, RESIDUAL=residual, PREC=prec, WDT=WDT,
                        BM=cC["BM"], BN=cC["BN"], CK=C, num_warps=cC["num_warps"], num_stages=cC["num_stages"])
    else:
        _k_out[grid_c](x, z, w["p_out"], w["g_out"], w["ln_in_w"], w["ln_in_b"], w["ln_out_w"], w["ln_out_b"], out,
                       N, Np, C, njt, eps, RESIDUAL=residual, PREC=prec, WDT=WDT,
                       BM=cC["BM"], BN=cC["BN"], BK=cC["BK"], NK=C // cC["BK"], num_warps=cC["num_warps"], num_stages=cC["num_stages"])
    return out


def _wT(w, name):
    """Pre-transposed GEMM weight W.t().contiguous() ([K, N] row-major) cached in the weight dict (a pure layout copy of the same bf16/fp32 values)."""
    k = name + "_T"
    if k not in w:
        w[k] = w[name].t().contiguous()
    return w[k]


def pack_weights(ln_in_w, ln_in_b, p_in, g_in, ln_out_w, ln_out_b, p_out, g_out):
    """All tensors exactly as handed to cuEq by the engine (same dtype, no cast): p_in/g_in [2C, C] = cat[a, b] rows."""
    return dict(ln_in_w=ln_in_w.contiguous(), ln_in_b=ln_in_b.contiguous(), p_in=p_in.contiguous(), g_in=g_in.contiguous(),
                ln_out_w=ln_out_w.contiguous(), ln_out_b=ln_out_b.contiguous(), p_out=p_out.contiguous(), g_out=g_out.contiguous())
