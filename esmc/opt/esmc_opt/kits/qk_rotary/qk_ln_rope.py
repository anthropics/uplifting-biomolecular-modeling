"""qk_ln_rope — ONE Triton kernel = aten vectorized_layer_norm_kernel<float,float,false> (gamma-only, the NT-128 float4
Welford order of aten/src/ATen/native/cuda/layer_norm_kernel.cu @ v2.11.0, sha256 6bc37b0cbeb94b534668c6c34b91067acf085ec7e4276a2776dec9351466e066)
on the exact bf16->fp32 up-cast of a q|k row, then the RNE fp32->bf16 down-cast (the stock's `.to(q.dtype)`), then the
flash-attn Triton rotary_kernel's math (non-interleaved, fp32 math on bf16 cos/sin, RNE store) — per token row, in-place capable.

Reference chain (the ESM C SDK route, esm/models/esmc/layers.py): q = chunk(qkv) [T, d] bf16 (row stride 3d) -> autocast up-cast
copy (fp32, contiguous) -> aten layer_norm (fp32 in/out; NT=128 threads=(32,4); vec_size=4; per-thread online Welford over
vec i = thrx, thrx+128, ... while i < d/4 (5 vecs x 4 elems for d=2560; d=960: 240 vecs, threads 0..111 read 2, the rest 1 — the ragged
last pass, RAGGED); warp shfl_down tree offsets 16..1 with cuWelfordCombine(own, other);
inter-warp smem tree offsets 2,1; sigma2/N (IEEE div); rstd = rsqrtf(sigma2+eps); y = gamma*(rstd*(x-mean))) -> .to(bf16) RNE
-> stack to [T,3,H,D] -> rotary_kernel(q, cos, sin, cu_seqlens, max_seqlen, inplace=True): o0 = x0*cos - x1*sin, o1 = x0*sin + x1*cos
(fp32; LLVM fp-contract) -> bf16 RNE.

nvcc (-fmad=true, -prec-div=true, no fast-math) contraction sites mirrored with EXPLICIT fma:
  online:  new_mean = fma(delta, 1/new_count, mean);  new_sigma2 = fma(delta, val - new_mean, sigma2)
  combine: mean = fma(nA, meanA, nB*meanB) [MEAN_FMA=0] | fma(nB, meanB, nA*meanA) [MEAN_FMA=1];  sigma2 = fma((delta*delta)*countA, nB, sigmaA+sigmaB)
  1/count and sigma2/N: IEEE div.rn (tl.math.div_rn);  rsqrt: rsqrt.approx (tl.math.rsqrt)
Variants are constexpr so each can be compared on the GPU against the stock chain; the shipped selection is CERT_DEFAULTS
(after the kernels, below).
"""
import torch
import triton
import triton.language as tl

NT = 128          # aten: threads = (warp_size, num_threads()/warp_size) = (32, 4)
VEC = 4           # aten vec_size (float4 for T=float)


@triton.jit
def _welford_combine(mB, sB, cB, mA, sA, cA, MEAN_FMA: tl.constexpr):
    # aten cuWelfordCombine(dataB = own, dataA = other)
    delta = mB - mA
    count = cA + cB
    one = tl.full(count.shape, 1.0, tl.float32)
    coef = tl.math.div_rn(one, count)          # 1.f/count  (IEEE)
    nA = cA * coef
    nB = cB * coef
    if MEAN_FMA == 0:
        mean = tl.math.fma(nA, mA, nB * mB)     # nA*meanA + nB*meanB
    else:
        mean = tl.math.fma(nB, mB, nA * mA)
    t = (delta * delta) * cA
    sigma2 = tl.math.fma(t, nB, sA + sB)       # sigmaA + sigmaB + delta*delta*countA*nB
    return mean, sigma2, count


@triton.jit
def _halve_last(x, W: tl.constexpr, L: tl.constexpr):
    # x: [W, L] -> (x[:, :L//2], x[:, L//2:])  == (own lane l, lane l + L//2) of the shfl_down tree
    x3 = tl.reshape(x, [W, 2, L // 2])
    x3 = tl.permute(x3, (0, 2, 1))
    lo, hi = tl.split(x3)
    return lo, hi


@triton.jit
def qk_ln_rope_kernel(X, Y, GAMMA, COS, SIN, CU_SEQLENS,
                      stride_xm, stride_ym, sstride_x, sstride_y, sstride_g, n_seqs, seqlen_ro, eps,
                      N: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr, NVEC_T: tl.constexpr, BLOCK_B: tl.constexpr,
                      MEAN_FMA: tl.constexpr, ROT_FMA: tl.constexpr, ROTATE: tl.constexpr, H_P2: tl.constexpr, REST_P2: tl.constexpr):
    NV: tl.constexpr = N // 4                                       # aten n_vec_to_read = N / vec_size
    RAGGED: tl.constexpr = NV % 128 != 0                            # the last pass i = thrx + (NVEC_T-1)*128 exists for thrx < NV % 128 only (aten: `for (i = thrx; i < n_vec_to_read; i += numx)`)
    row = tl.program_id(0)
    slot = tl.program_id(1)          # 0 = q, 1 = k when launched over the packed qkv; grid (T, 1) for a single tensor
    # ---- position of this token inside its sequence (varlen, cu_seqlens[0] == 0)
    bidx = tl.arange(0, BLOCK_B)
    cu = tl.load(CU_SEQLENS + bidx, mask=bidx < n_seqs + 1, other=2147483647)
    start = tl.max(tl.where(cu <= row, cu, 0), axis=0)
    pos = row - start
    # ---- per-"thread" online Welford in aten's element order: vec i = thrx + s*NT, elems ii = 0..3
    t = tl.arange(0, 128)
    xrow = X + row.to(tl.int64) * stride_xm + slot * sstride_x
    GAMMA = GAMMA + slot * sstride_g
    mean = tl.zeros([128], tl.float32)
    m2 = tl.zeros([128], tl.float32)
    cnt = tl.zeros([128], tl.float32)
    one = tl.full([128], 1.0, tl.float32)
    for s in tl.static_range(NVEC_T):
        if RAGGED and s == NVEC_T - 1:                     # the ragged last pass: threads whose vec index is past NV do not iterate (their Welford state stands)
            live = (t + s * 128) < NV
            for ii in tl.static_range(4):
                v = tl.load(xrow + (t + s * 128) * 4 + ii, mask=live, other=0.0).to(tl.float32)
                delta = v - mean
                cnt1 = cnt + 1.0
                rcp = tl.math.div_rn(one, cnt1)
                mean1 = tl.math.fma(delta, rcp, mean)
                m21 = tl.math.fma(delta, v - mean1, m2)
                cnt = tl.where(live, cnt1, cnt)
                mean = tl.where(live, mean1, mean)
                m2 = tl.where(live, m21, m2)
        else:
            for ii in tl.static_range(4):
                v = tl.load(xrow + (t + s * 128) * 4 + ii).to(tl.float32)
                delta = v - mean
                cnt = cnt + 1.0
                rcp = tl.math.div_rn(one, cnt)                 # 1.f/new_count (IEEE)
                mean = tl.math.fma(delta, rcp, mean)           # mean + delta*(1/new_count)
                m2 = tl.math.fma(delta, v - mean, m2)          # sigma2 + delta*(val - new_mean)
    # ---- intra-warp shfl_down tree: offsets 16, 8, 4, 2, 1 ; own = lane l (dataB), other = lane l+offset (dataA)
    mean = tl.reshape(mean, [4, 32])
    m2 = tl.reshape(m2, [4, 32])
    cnt = tl.reshape(cnt, [4, 32])
    mB, mA = _halve_last(mean, 4, 32)
    sB, sA = _halve_last(m2, 4, 32)
    cB, cA = _halve_last(cnt, 4, 32)
    mean, m2, cnt = _welford_combine(mB, sB, cB, mA, sA, cA, MEAN_FMA)
    mB, mA = _halve_last(mean, 4, 16)
    sB, sA = _halve_last(m2, 4, 16)
    cB, cA = _halve_last(cnt, 4, 16)
    mean, m2, cnt = _welford_combine(mB, sB, cB, mA, sA, cA, MEAN_FMA)
    mB, mA = _halve_last(mean, 4, 8)
    sB, sA = _halve_last(m2, 4, 8)
    cB, cA = _halve_last(cnt, 4, 8)
    mean, m2, cnt = _welford_combine(mB, sB, cB, mA, sA, cA, MEAN_FMA)
    mB, mA = _halve_last(mean, 4, 4)
    sB, sA = _halve_last(m2, 4, 4)
    cB, cA = _halve_last(cnt, 4, 4)
    mean, m2, cnt = _welford_combine(mB, sB, cB, mA, sA, cA, MEAN_FMA)
    mB, mA = _halve_last(mean, 4, 2)
    sB, sA = _halve_last(m2, 4, 2)
    cB, cA = _halve_last(cnt, 4, 2)
    mean, m2, cnt = _welford_combine(mB, sB, cB, mA, sA, cA, MEAN_FMA)     # [4, 1] : lane 0 of each warp
    # ---- inter-warp smem tree: offsets 2, 1 ; own = warp y (dataB), other = warp y+offset (dataA)
    mean = tl.reshape(mean, [1, 4])
    m2 = tl.reshape(m2, [1, 4])
    cnt = tl.reshape(cnt, [1, 4])
    mB, mA = _halve_last(mean, 1, 4)
    sB, sA = _halve_last(m2, 1, 4)
    cB, cA = _halve_last(cnt, 1, 4)
    mean, m2, cnt = _welford_combine(mB, sB, cB, mA, sA, cA, MEAN_FMA)     # [1, 2]
    mB, mA = _halve_last(mean, 1, 2)
    sB, sA = _halve_last(m2, 1, 2)
    cB, cA = _halve_last(cnt, 1, 2)
    mean, m2, cnt = _welford_combine(mB, sB, cB, mA, sA, cA, MEAN_FMA)     # [1, 1]
    mean_s = tl.sum(tl.reshape(mean, [1]))       # one element: exact
    m2_1 = tl.reshape(m2, [1])
    nn = tl.full([1], N, tl.float32)
    sig = tl.math.div_rn(m2_1, nn)               # wd.sigma2 / float(N)  (IEEE)
    sig_s = tl.sum(sig)
    rstd = tl.math.rsqrt(sig_s + eps)            # rsqrtf(sigma2 + eps)
    # ---- epilogue: y = gamma * (rstd * (x - mean)) -> RNE bf16 -> fp32 -> rotary on dims [0, ROT) of each head (non-interleaved,
    #      pairs d <-> d + ROT/2); dims [ROT, D) carry the LN output unrotated (the stock's in-place rotary leaves them untouched)
    H: tl.constexpr = N // D
    HALF: tl.constexpr = ROT // 2
    REST: tl.constexpr = D - ROT
    h = tl.arange(0, H_P2)
    d = tl.arange(0, HALF)
    hm = (h < H)[:, None] & (d < HALF)[None, :]
    off0 = h[:, None] * D + d[None, :]
    off1 = off0 + HALF
    x0 = tl.load(xrow + off0, mask=hm, other=0.0).to(tl.float32)
    x1 = tl.load(xrow + off1, mask=hm, other=0.0).to(tl.float32)
    g0 = tl.load(GAMMA + off0, mask=hm, other=0.0).to(tl.float32)
    g1 = tl.load(GAMMA + off1, mask=hm, other=0.0).to(tl.float32)
    y0 = g0 * (rstd * (x0 - mean_s))
    y1 = g1 * (rstd * (x1 - mean_s))
    y0 = y0.to(tl.bfloat16).to(tl.float32)      # the stock's .to(bf16) (RNE) then the rotary kernel's .to(float32)
    y1 = y1.to(tl.bfloat16).to(tl.float32)
    yrow = Y + row.to(tl.int64) * stride_ym + slot * sstride_y
    if REST > 0:
        dr = tl.arange(0, REST_P2)
        rm = (h < H)[:, None] & (dr < REST)[None, :]
        offr = h[:, None] * D + ROT + dr[None, :]
        xr = tl.load(xrow + offr, mask=rm, other=0.0).to(tl.float32)
        gr = tl.load(GAMMA + offr, mask=rm, other=0.0).to(tl.float32)
        yr = gr * (rstd * (xr - mean_s))
        tl.store(yrow + offr, yr.to(tl.bfloat16), mask=rm)
    if ROTATE:
        cmask = (d < HALF) & (pos < seqlen_ro)
        cos = tl.load(COS + pos * HALF + d, mask=cmask, other=1.0).to(tl.float32)[None, :]
        sin = tl.load(SIN + pos * HALF + d, mask=cmask, other=0.0).to(tl.float32)[None, :]
        if ROT_FMA == 0:
            o0 = y0 * cos - y1 * sin              # verbatim rotary_kernel expressions (LLVM contraction as the stock)
            o1 = y0 * sin + y1 * cos
        elif ROT_FMA == 1:
            o0 = tl.math.fma(y0, cos, -(y1 * sin))
            o1 = tl.math.fma(y0, sin, y1 * cos)
        elif ROT_FMA == 2:
            o0 = tl.math.fma(-y1, sin, y0 * cos)
            o1 = tl.math.fma(y1, cos, y0 * sin)
        else:
            o0 = (y0 * cos) - (y1 * sin)
            o1 = (y0 * sin) + (y1 * cos)
        tl.store(yrow + off0, o0.to(tl.bfloat16), mask=hm)
        tl.store(yrow + off1, o1.to(tl.bfloat16), mask=hm)
    else:
        tl.store(yrow + off0, y0.to(tl.bfloat16), mask=hm)
        tl.store(yrow + off1, y1.to(tl.bfloat16), mask=hm)


CERT_DEFAULTS = {"MEAN_FMA": 0, "ROT_FMA": 0, "fp_fusion": True}   # the shipped variant selection


def _variant(mean_fma, rot_fma, fp_fusion):
    mf = CERT_DEFAULTS["MEAN_FMA"] if mean_fma is None else mean_fma
    rf = CERT_DEFAULTS["ROT_FMA"] if rot_fma is None else rot_fma
    ff = CERT_DEFAULTS["fp_fusion"] if fp_fusion is None else fp_fusion
    return mf, rf, ff


def qk_ln_rope(x, gamma, cos, sin, cu_seqlens, out=None, head_dim=None, rotate=True, eps=1e-5,
               mean_fma=None, rot_fma=None, fp_fusion=None, num_warps=4):
    """ONE tensor. x: [T, N] bf16 (any row stride, unit column stride); gamma: [N] (bf16 or fp32 — exact up-cast either way);
    cos/sin: [seqlen_ro, Dh/2] (the module's cache); cu_seqlens: int32 [n_seqs+1] on device; out: [T, N] bf16 (may be x).
    head_dim defaults to 2*cos.shape[1] (rotary_dim == head_dim on this route)."""
    T, N = x.shape
    rot_dim = 2 * cos.shape[1]
    head_dim = rot_dim if head_dim is None else head_dim
    assert x.stride(1) == 1 and N % VEC == 0 and N % head_dim == 0 and head_dim % 2 == 0 and rot_dim <= head_dim   # aten's vectorized path: N % vec_size == 0 (960 / 1152 / 2560)
    if out is None:
        out = torch.empty_like(x)
    assert out.stride(1) == 1 and out.shape == x.shape
    n_seqs = cu_seqlens.numel() - 1
    BLOCK_B = max(2, triton.next_power_of_2(n_seqs + 1))
    mf, rf, ff = _variant(mean_fma, rot_fma, fp_fusion)
    qk_ln_rope_kernel[(T, 1)](x, out, gamma, cos, sin, cu_seqlens, x.stride(0), out.stride(0), 0, 0, 0, n_seqs, cos.shape[0], float(eps),
                              N=N, D=head_dim, ROT=rot_dim, NVEC_T=triton.cdiv(N // VEC, NT), BLOCK_B=BLOCK_B,
                              MEAN_FMA=mf, ROT_FMA=rf, ROTATE=rotate, H_P2=triton.next_power_of_2(N // head_dim),
                              REST_P2=max(1, triton.next_power_of_2(head_dim - rot_dim)), num_warps=num_warps, enable_fp_fusion=ff)
    return out


def qk_ln_rope_packed(qkv_packed, gamma2, cos, sin, cu_seqlens, eps=1e-5, mean_fma=None, rot_fma=None, fp_fusion=None, num_warps=4):
    """THE SEAM FORM, ONE launch: qkv_packed = qkv.view(T, 3, H, Dh) bf16 contiguous; gamma2 = [2, N] (q_ln.weight, k_ln.weight stacked);
    LN(q), LN(k) RNE-bf16 + rotation written IN PLACE into slots 0 and 1; slot 2 (v) untouched. Returns None."""
    T, three, H, Dh = qkv_packed.shape
    rot_dim = 2 * cos.shape[1]
    assert three == 3 and qkv_packed.is_contiguous() and rot_dim <= Dh
    N = H * Dh
    assert N % VEC == 0                                                            # aten's vectorized path (every ESM C width)
    x = qkv_packed.view(T, 3 * N)
    assert gamma2.shape == (2, N) and gamma2.is_contiguous()
    n_seqs = cu_seqlens.numel() - 1
    BLOCK_B = max(2, triton.next_power_of_2(n_seqs + 1))
    mf, rf, ff = _variant(mean_fma, rot_fma, fp_fusion)
    qk_ln_rope_kernel[(T, 2)](x, x, gamma2, cos, sin, cu_seqlens, x.stride(0), x.stride(0), N, N, N, n_seqs, cos.shape[0], float(eps),
                              N=N, D=Dh, ROT=rot_dim, NVEC_T=triton.cdiv(N // VEC, NT), BLOCK_B=BLOCK_B,
                              MEAN_FMA=mf, ROT_FMA=rf, ROTATE=True, H_P2=triton.next_power_of_2(H),
                              REST_P2=max(1, triton.next_power_of_2(Dh - rot_dim)), num_warps=num_warps, enable_fp_fusion=ff)


