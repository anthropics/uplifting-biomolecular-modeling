"""opt_core.ops.msa_opm — fused MSA OuterProductMean: stock LN + a/b projections (the stock modules' own ops = bitwise prologue) + ONE Triton kernel that
computes the outer-product GEMM over the MSA dimension and applies the c_hidden^2 -> c_z output projection (+bias, /norm) in the epilogue,
so the [N, N, c_hidden^2] outer-product tensor never exists in HBM (stock at N=705: GEMM write 1 GB + permute copy 2 GB + GEMM read 1 GB).

STOCK FORMS reproduced (two attribute schemas of the same op; the adapter names the form):
  * form 'scalar_norm' — OuterProductMean(c_m, c_z, c_hidden=32) with attributes layer_norm / linear_1 / linear_2 / linear_out, forward(m, mask=None, chunk_size=None):
        ln = LN(m) (bf16); a = linear_1(ln), b = linear_2(ln) (bf16); outer[i,j,c,e] = sum_s a[s,i,c] b[s,j,e] (autocast einsum: bf16 GEMM,
        fp32 acc, bf16 out); out = linear_out(outer.reshape(N,N,1024)) (bf16 GEMM K=1024 + bf16-rounded bias); norm = einsum(ones,ones)+eps
        = bf16(S)+1e-3 (bf16); out /= norm (bf16 IEEE division).
  * form 'mask_norm' — OuterProductMean(c_in, 32, c_out) with attributes norm / proj_a / proj_b / proj_o, forward(m, mask, chunk_size):
        mask -> bf16; a,b = proj(LN(m)) * mask; UNCHUNKED (N<=384): z = einsum(a.float(), b.float()) (fp32 GEMM, TF32 OFF); z /= num_mask
        (num_mask = bf16 sum over S of mask_i*mask_j, clamp>=1); proj_o(z.to(bf16)) (bf16 GEMM + bias).  CHUNKED (N>384, chunk_size=4):
        8 hidden-chunks, num_mask accumulated in bf16 over 64-row S-chunks, projection partials ACCUMULATED IN bf16 across the 8 chunks.
CANDIDATE numerics: bf16 operands, fp32 accumulation over S (tile order differs from cuBLAS); the outer tile is rounded to bf16 exactly where
stock rounds (scalar_norm: GEMM output; mask_norm: after /num_mask, before the projection); projection fp32 acc (K=1024 as 32 batched K=32 dots
summed in fp32); bias added in fp32 after bf16-rounding the bias (autocast casts it); /norm with IEEE division (tl.div_rn).
=> TIER 2 by construction.  No atomics, fixed pinned tiles, no split-K -> r2r bitwise, co-tenancy invariant; batch-invariant (B folds into
the grid).  Triton 3.3.1-compatible (no TMA, no permute, no 3.4+ features; reshapes are dim merges/splits only).

Kernel: grouped-ordering GEMM  C[(c,bi), (bj,e)] = A[(c,bi), s] @ B[s, (bj,e)]  with B stored [S, N*CH] (s-major); the A tile rows are
gathered in (c, bi) order so the epilogue re-layout is a pure reshape: acc -> (CH, BI*BJ, CH) = P3[c, pair, e]; then for z-chunks:
o[pair, z] = sum_c dot(P3[c], Wout3[c]) with Wout3[c, e, z] = Wout[c*CH+e, z]; + bias; (/norm); bf16 store.
A LAYOUTS (cfg keys AT / TMA; AT=True: A2[(iblk, c, bi), s] = a[s, iblk*BI + bi, c], so each program's tile is a contiguous [BM, BK] row block):
  AT=False (v1, original): A stored [S, N*CH]; the (c, bi) row gather has stride CH between consecutive tile rows -> scalar 2-byte loads
           (the load is a permutation of a contiguous 128-element span, which the vectorizer cannot see) and a tl.trans before the dot.
  AT=True  (t*): A is re-laid out ONCE per call (pure data movement, exact) to AT[c, i, s] = a[s, i, c], zero-padded to NA = cdiv(N,BI)*BI
           rows and SK = cdiv(S,BK)*BK columns; B is zero-padded to [SK, NB*CH].  The A tile is then a plain row-major [BM, BK] block
           (16-byte vector loads, no trans), the K loop is mask-free (zero padding adds exact zeros to the fp32 accumulator), and the
           tile/pipeline config is the only remaining knob.  Numerics identical to AT=False up to the MMA's internal summation order.
PROJ_IN_KERNEL=False fallback: store the outer tile to an HBM workspace in the stock's [i,j,c,e] layout (no permute copy) + cuBLAS projection.
"""
import torch, triton, triton.language as tl

@triton.jit
def _opm_epilogue(acc, WOUT, BIAS, NORM, OUT, WS, N, i0, j0, norm_scalar,
                  CH: tl.constexpr, CZ: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, BZ: tl.constexpr,
                  EPI: tl.constexpr, HAS_BIAS: tl.constexpr, PROJ_IN_KERNEL: tl.constexpr, PROJ_MODE: tl.constexpr):
    # acc: [BM, BN] fp32 with rows (c, bi) and columns (bj, e).  PROJ_MODE 0: CH batched K=32 dots + fp32 sum over c (3.3-compatible);
    # PROJ_MODE 1: one K=1024 dot per z-chunk on the permuted [NP, CH*CH] tile (Triton >= 3.4).
    BM: tl.constexpr = BI * CH
    BN: tl.constexpr = BJ * CH
    NP: tl.constexpr = BI * BJ
    i0 = i0.to(tl.int64)                                 # int64 tile origins (the callers pass them so; restated here): r_i / p_i / bi4 / bj4 and
    j0 = j0.to(tl.int64)                                 # every NORM / OUT / WS element offset below are formed in int64
    r = tl.arange(0, BM)
    r_i = i0 + (r % BI)
    rmask = r_i < N
    acc = acc.to(tl.bfloat16).to(tl.float32)            # stock: the outer-product GEMM output is bf16 (autocast einsum) in both forms
    c_j = j0 + tl.arange(0, BN) // CH
    if EPI != 0:
        nm = tl.load(NORM + r_i[:, None] * N + c_j[None, :], mask=rmask[:, None] & (c_j < N)[None, :], other=1.0).to(tl.float32)   # [BM, BN] bf16 num_mask
        acc = tl.div_rn(acc, nm)
        acc = acc.to(tl.bfloat16).to(tl.float32)         # stock: bf16 / bf16 -> bf16, then .to(m) (fp32) is value-preserving
    if PROJ_IN_KERNEL:
        P3 = tl.reshape(acc, (CH, NP, CH)).to(tl.bfloat16)                 # [c, pair=(bi,bj), e]
        pr = tl.arange(0, NP)
        p_i = i0 + pr // BJ
        p_j = j0 + pr % BJ
        pmask = (p_i < N) & (p_j < N)
        ce = tl.arange(0, CH)
        cc = tl.arange(0, CH)
        if PROJ_MODE == 1:
            P2 = tl.reshape(tl.permute(P3, (1, 0, 2)), (NP, CH * CH))       # [pair, k=(c,e)]  (Triton >= 3.4: tl.permute)
            kk = tl.arange(0, CH * CH)
        for z0 in range(0, CZ, BZ):
            zz = z0 + tl.arange(0, BZ)
            if PROJ_MODE == 1:
                w2 = tl.load(WOUT + kk[:, None] * CZ + zz[None, :])                                         # [CH*CH, BZ] bf16
                o = tl.dot(P2, w2)                                                                          # [NP, BZ] fp32, K = 1024 in one MMA chain
            else:
                w3 = tl.load(WOUT + (cc[:, None, None] * CH + ce[None, :, None]) * CZ + zz[None, None, :])     # [CH, CH, BZ] bf16
                o3 = tl.dot(P3, w3)                                                                             # [CH, NP, BZ] fp32
                o = tl.sum(o3, axis=0)                                                                          # [NP, BZ]
            if EPI == 2:
                o = o.to(tl.bfloat16).to(tl.float32)      # stock chunked: bf16 GEMM partials (accumulated in bf16) ...
                if HAS_BIAS:
                    o = o + tl.load(BIAS + CZ + zz).to(tl.float32)[None, :]     # ... + fp32 bias -> fp32 output (BIAS buffer: [bf16-rounded | exact] x CZ)
            else:
                if HAS_BIAS:
                    o += tl.load(BIAS + zz).to(tl.float32)[None, :]            # cuBLASLt epilogue: fp32 acc + bf16-cast bias, one rounding
                if EPI == 0:
                    o16 = o.to(tl.bfloat16).to(tl.float32)
                    o = tl.div_rn(o16, norm_scalar)                            # stock: outer /= norm in bf16
            tl.store(OUT + (p_i * N + p_j)[:, None] * CZ + zz[None, :], o.to(OUT.dtype.element_ty), mask=pmask[:, None])
    else:
        # workspace in the stock [i, j, c, e] layout: acc[(c, bi), (bj, e)] -> 4-D tile (c, bi, bj, e); contiguous along e -> vector stores
        acc4 = tl.reshape(acc, (CH, BI, BJ, CH)).to(tl.bfloat16)
        c4 = tl.arange(0, CH); bi4 = i0 + tl.arange(0, BI); bj4 = j0 + tl.arange(0, BJ); e4 = tl.arange(0, CH)
        ptr = WS + (bi4[None, :, None, None] * N + bj4[None, None, :, None]) * (CH * CH) + c4[:, None, None, None] * CH + e4[None, None, None, :]
        tl.store(ptr, acc4, mask=(bi4 < N)[None, :, None, None] & (bj4 < N)[None, None, :, None])


@triton.jit
def _grouped_pid(pid, N, num_pid_n, BI: tl.constexpr, GROUP_M: tl.constexpr):
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    num_pid_m = tl.cdiv(N, BI)
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(do_not_specialize=['S'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _opm_kernel(A, B, WOUT, BIAS, NORM, OUT, WS,
                S, SK, N, NB, num_pid_n, norm_scalar,
                CH: tl.constexpr, CZ: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr, BZ: tl.constexpr, GROUP_M: tl.constexpr,
                EPI: tl.constexpr, HAS_BIAS: tl.constexpr, PROJ_IN_KERNEL: tl.constexpr, A_T: tl.constexpr, PROJ_MODE: tl.constexpr):
    # EPI: 0 = scalar_norm form   (outer->bf16; proj fp32 acc + bf16 bias -> bf16; /= norm scalar -> bf16)
    #      1 = mask_norm unchunked  (outer->bf16; / num_mask[i,j] (bf16) -> bf16; proj fp32 acc + bf16 bias -> bf16)
    #      2 = mask_norm chunked    (same, but output = fp32( bf16(proj acc) + fp32 bias ) as the stock chunked path returns fp32)
    # A_T: A is the tile-contiguous layout A2[(iblk, c, bi), s] = a[s, iblk*BI + bi, c] ([NA*CH, SK] row-major, zero padded);
    #      B is [SK, NB] (NB = padded N * CH); SK % BK == 0 -> mask-free K loop.
    BM: tl.constexpr = BI * CH
    BN: tl.constexpr = BJ * CH
    pid = tl.program_id(0)
    pid_m, pid_n = _grouped_pid(pid, N, num_pid_n, BI, GROUP_M)
    pid_m = pid_m.to(tl.int64)                 # int64 tile indices: every A / B / NORM / OUT / WS element offset below (and in _opm_epilogue) is formed in int64
    pid_n = pid_n.to(tl.int64)
    i0 = pid_m * BI
    j0 = pid_n * BJ
    r = tl.arange(0, BM)                       # tile row r = c*BI + bi
    r_c = r // BI
    r_i = i0 + (r % BI)
    rmask = r_i < N
    cols = j0 * CH + tl.arange(0, BN)          # B columns (j, e) contiguous
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if A_T:
        a_ptrs = A + (pid_m * BM + r)[:, None] * SK + ks[None, :]          # [BM, BK] contiguous row block of A2
        b_ptrs = B + ks[:, None] * NB + cols[None, :]                      # [BK, BN]
        for s0 in range(0, SK, BK):
            a_t = tl.load(a_ptrs)
            b_t = tl.load(b_ptrs)
            acc = tl.dot(a_t, b_t, acc)
            a_ptrs += BK
            b_ptrs += BK * NB
    else:
        a_cols = r_i * CH + r_c                    # column of A for (i, c)
        cmask = cols < N * CH
        for s0 in range(0, S, BK):
            srow = s0 + ks
            smask = srow < S
            srow64 = srow.to(tl.int64)         # int64 MSA-row index: srow * (N * CH) is formed in int64
            a_t = tl.load(A + srow64[:, None] * (N * CH) + a_cols[None, :], mask=smask[:, None] & rmask[None, :], other=0.0)   # [BK, BM] bf16
            b_t = tl.load(B + srow64[:, None] * (N * CH) + cols[None, :], mask=smask[:, None] & cmask[None, :], other=0.0)     # [BK, BN] bf16
            acc += tl.dot(tl.trans(a_t), b_t)
    _opm_epilogue(acc, WOUT, BIAS, NORM, OUT, WS, N, i0, j0, norm_scalar, CH, CZ, BI, BJ, BZ, EPI, HAS_BIAS, PROJ_IN_KERNEL, PROJ_MODE)


@triton.jit
def _opm_kernel_tma(a_desc, b_desc, WOUT, BIAS, NORM, OUT, WS,
                    SK, N, num_pid_n, norm_scalar,
                    CH: tl.constexpr, CZ: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, BK: tl.constexpr, BZ: tl.constexpr, GROUP_M: tl.constexpr,
                    EPI: tl.constexpr, HAS_BIAS: tl.constexpr, PROJ_IN_KERNEL: tl.constexpr, PROJ_MODE: tl.constexpr):
    # Same math as _opm_kernel(A_T=True) with TMA tensor-descriptor loads (Triton >= 3.4; Hopper): a_desc over A2 [NA*CH, SK] with box [BM, BK],
    # b_desc over BT[(j, e), s] = b[s, j, e] ([NB, SK], K-major) with box [BN, BK]; the dot consumes the transposed smem tile (wgmma TN form).
    # Mask-free (zero padding), fixed tiles, no atomics.
    BM: tl.constexpr = BI * CH
    BN: tl.constexpr = BJ * CH
    pid = tl.program_id(0)
    pid_m, pid_n = _grouped_pid(pid, N, num_pid_n, BI, GROUP_M)
    i0 = pid_m.to(tl.int64) * BI               # int64 tile origins: the NORM / OUT / WS element offsets in _opm_epilogue are formed in int64
    j0 = pid_n.to(tl.int64) * BJ
    row0 = pid_m * BM                          # tensor-descriptor coordinates stay int32 (A2 / BT rows = 32 * N, columns = SK)
    col0 = pid_n * BN
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for s0 in range(0, SK, BK):
        a_t = a_desc.load([row0, s0])            # [BM, BK]
        b_t = b_desc.load([col0, s0])            # [BN, BK]
        acc = tl.dot(a_t, b_t.T, acc)
    _opm_epilogue(acc, WOUT, BIAS, NORM, OUT, WS, N, i0, j0, norm_scalar, CH, CZ, BI, BJ, BZ, EPI, HAS_BIAS, PROJ_IN_KERNEL, PROJ_MODE)


@triton.jit(do_not_specialize=['S'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _opm_prologue_kernel(M, MASK, LNW, LNB, WA, WB, BA, BB, A2, BT,
                         S, N, SK, NA, NBj, stride_ms, stride_mi, eps,
                         CM: tl.constexpr, CH: tl.constexpr, BI: tl.constexpr, BS: tl.constexpr, BIP: tl.constexpr,
                         HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr, LN_AFFINE: tl.constexpr, MASK_I64: tl.constexpr):
    # Fused stock prologue: y = LN(m[s, i, :]) in fp32 (torch.nn.LayerNorm math: biased var, rsqrt(var + eps), affine), y -> bf16 (autocast),
    # a = y16 @ Wa^T, b = y16 @ Wb^T with fp32 accumulation (+ bf16-rounded bias in fp32, one rounding: the cuBLASLt epilogue), -> bf16,
    # * mask (0/1: exact).  Written directly in the GEMM layouts: A2[(iblk, c, bi), s] (iblk = i // BI) and BT[(i, e), s], zero-padded to
    # (NA | NBj) x SK.  Tile = BIP tokens x BS MSA rows; rows ordered (i, s) so that the stores are contiguous along s.
    pid_s = tl.program_id(0).to(tl.int64)     # int64 program indices: every m / mask / A2 / BT element offset below is formed in int64
    pid_i = tl.program_id(1).to(tl.int64)
    s0 = pid_s * BS
    i0 = pid_i * BIP
    R: tl.constexpr = BIP * BS
    rr = tl.arange(0, R)
    r_i = i0 + rr // BS
    r_s = s0 + rr % BS
    valid = (r_s < S) & (r_i < N)
    cols = tl.arange(0, CM)
    x = tl.load(M + r_s[:, None] * stride_ms + r_i[:, None] * stride_mi + cols[None, :], mask=valid[:, None], other=0.0).to(tl.float32)   # [R, CM]
    mean = tl.sum(x, axis=1) / CM
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / CM
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xc * rstd[:, None]
    if LN_AFFINE:
        y = y * tl.load(LNW + cols).to(tl.float32)[None, :] + tl.load(LNB + cols).to(tl.float32)[None, :]
    y16 = y.to(tl.bfloat16)
    ch = tl.arange(0, CH)
    wa = tl.load(WA + cols[:, None] * CH + ch[None, :])          # [CM, CH] bf16 = Wa^T
    wb = tl.load(WB + cols[:, None] * CH + ch[None, :])
    a = tl.dot(y16, wa)                                           # [R, CH] fp32
    b = tl.dot(y16, wb)
    if HAS_BIAS:
        a += tl.load(BA + ch).to(tl.float32)[None, :]
        b += tl.load(BB + ch).to(tl.float32)[None, :]
    a16 = a.to(tl.bfloat16)
    b16 = b.to(tl.bfloat16)
    if HAS_MASK:
        if MASK_I64:
            mk = tl.load(MASK + r_s * N + r_i, mask=valid, other=0).to(tl.float32)
        else:
            mk = tl.load(MASK + r_s * N + r_i, mask=valid, other=0.0).to(tl.float32)
        a16 = (a16.to(tl.float32) * mk[:, None]).to(tl.bfloat16)
        b16 = (b16.to(tl.float32) * mk[:, None]).to(tl.bfloat16)
    a16 = tl.where(valid[:, None], a16, a16 * 0.0)                # zero padding rows (s >= S or i >= N)
    b16 = tl.where(valid[:, None], b16, b16 * 0.0)
    a_rows = ((r_i // BI) * CH)[:, None] + ch[None, :]
    a_ptr = A2 + (a_rows * BI + (r_i % BI)[:, None]) * SK + r_s[:, None]
    tl.store(a_ptr, a16, mask=((r_i < NA) & (r_s < SK))[:, None])
    b_ptr = BT + (r_i[:, None] * CH + ch[None, :]) * SK + r_s[:, None]
    tl.store(b_ptr, b16, mask=((r_i < NBj) & (r_s < SK))[:, None])


@triton.jit
def _num_mask_kernel(MT, NM, N, SK, CHUNKED: tl.constexpr, BT_: tl.constexpr):
    # num_mask replica (mask_norm form): MT[i, s] bf16 0/1 (zero padded [>=N, SK]); count[i, j] = sum_s MT[i, s] * MT[j, s] as exact integers (bf16 MMA,
    # fp32 acc).  CHUNKED: stock (N > 384) accumulates bf16(count of each 64-row S-chunk) with a running bf16 sum (fp32 add + round = torch
    # bf16 add) in S order; unchunked: one rounding of the exact count.  clamp(min=1) last.  BT_ = 64 = the stock chunk size.
    pid_i = tl.program_id(0).to(tl.int64)     # int64 program indices: the MT [N, SK] and NM [N, N] element offsets below are formed in int64
    pid_j = tl.program_id(1).to(tl.int64)
    ri = pid_i * BT_ + tl.arange(0, BT_)
    rj = pid_j * BT_ + tl.arange(0, BT_)
    ks = tl.arange(0, BT_)
    run = tl.zeros((BT_, BT_), dtype=tl.float32)
    for s0 in range(0, SK, BT_):
        mi = tl.load(MT + ri[:, None] * SK + (s0 + ks)[None, :])
        mj = tl.load(MT + rj[:, None] * SK + (s0 + ks)[None, :])
        cnt = tl.dot(mi, tl.trans(mj))
        if CHUNKED:
            run = (run + cnt.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        else:
            run += cnt
    if not CHUNKED:
        run = run.to(tl.bfloat16).to(tl.float32)
    run = tl.maximum(run, 1.0)
    tl.store(NM + ri[:, None] * N + rj[None, :], run.to(tl.bfloat16), mask=(ri < N)[:, None] & (rj < N)[None, :])


def fused_prologue(m, mask, ln, wa_t, wb_t, ba, bb, BI, BJ, BK, CH, num_warps=8, BS=32, BIP=8):
    """m: [S, N, CM] (fp32 or bf16), mask: [S, N] (int64 / fp32 / bf16) or None.  Returns (A2 [NA*CH, SK], BT [NBj*CH, SK], NA, NBj, SK) bf16."""
    S, N, CM = m.shape
    NA = triton.cdiv(N, BI) * BI; NBj = triton.cdiv(N, BJ) * BJ; SK = triton.cdiv(S, BK) * BK
    A2 = torch.empty(NA * CH, SK, device=m.device, dtype=torch.bfloat16)
    BT = torch.empty(NBj * CH, SK, device=m.device, dtype=torch.bfloat16)
    assert SK % BS == 0 and m.stride(2) == 1, (SK, BS, m.stride())
    grid = (SK // BS, triton.cdiv(max(NA, NBj), BIP))
    affine = ln.weight is not None
    mask_c = None if mask is None else mask.contiguous()
    _opm_prologue_kernel[grid](m, mask_c if mask_c is not None else m, ln.weight if affine else wa_t, ln.bias if affine else wa_t,
                               wa_t, wb_t, ba if ba is not None else wa_t, bb if bb is not None else wa_t, A2, BT,
                               S, N, SK, NA, NBj, m.stride(0), m.stride(1), float(ln.eps),
                               CM=CM, CH=CH, BI=BI, BS=BS, BIP=BIP, HAS_MASK=mask_c is not None, HAS_BIAS=ba is not None, LN_AFFINE=affine,
                               MASK_I64=(mask_c is not None and not mask_c.is_floating_point()), num_warps=num_warps, num_stages=1)
    return A2, BT, NA, NBj, SK


def num_mask_triton(mask, N, S, chunked):
    """mask: [S, N] (any 0/1 dtype) -> [N, N] bf16 replica of num_mask_replica (bitwise: same chunk boundaries, same bf16 running sum)."""
    NA64 = triton.cdiv(N, 64) * 64; SK64 = triton.cdiv(S, 64) * 64
    MT = torch.zeros(NA64, SK64, device=mask.device, dtype=torch.bfloat16)
    MT[:N, :S].copy_(mask.t() if mask.dtype == torch.bfloat16 else mask.to(torch.bfloat16).t())
    NM = torch.empty(N, N, device=mask.device, dtype=torch.bfloat16)
    _num_mask_kernel[(NA64 // 64, NA64 // 64)](MT, NM, N, SK64, CHUNKED=chunked, BT_=64, num_warps=4, num_stages=2)
    return NM


def _pack(module, form):
    """The packed bf16 weights of ``module`` for ``form`` ('scalar_norm': layer_norm / linear_1 / linear_2 / linear_out; 'mask_norm': norm / proj_a /
    proj_b / proj_o), cached on the module (module._fpf_cache), re-used while the form matches."""
    cache = getattr(module, "_fpf_cache", None)
    if cache is not None and cache.get("form") == form and cache.get("kind") == "opm":
        return cache
    if form == "scalar_norm":
        w1, wo, bo = module.linear_1.weight, module.linear_out.weight, module.linear_out.bias
    else:
        w1, wo, bo = module.proj_a.weight, module.proj_o.weight, module.proj_o.bias
    CH = w1.shape[0]; CZ = wo.shape[0]
    if form == "scalar_norm":
        w2 = module.linear_2.weight; b1 = getattr(module.linear_1, "bias", None); b2 = getattr(module.linear_2, "bias", None); ln = module.layer_norm
    else:
        w2 = module.proj_b.weight; b1 = getattr(module.proj_a, "bias", None); b2 = getattr(module.proj_b, "bias", None); ln = module.norm
    cache = {"form": form, "kind": "opm", "CH": CH, "CZ": CZ, "ln": ln,
             "wa_t": w1.detach().to(torch.bfloat16).t().contiguous(), "wb_t": w2.detach().to(torch.bfloat16).t().contiguous(),      # [CM, CH] bf16 (autocast weight cast)
             "ba16": None if b1 is None else b1.detach().to(torch.bfloat16).contiguous(), "bb16": None if b2 is None else b2.detach().to(torch.bfloat16).contiguous(),
             "wout_t": wo.detach().t().contiguous().to(torch.bfloat16),                                   # [CH*CH, CZ]  (= Wout3[c, e, z])
             "wout": wo.detach().to(torch.bfloat16).contiguous(),                                          # [CZ, CH*CH] for the cuBLAS fallback
             "bias16": None if bo is None else bo.detach().to(torch.bfloat16),                             # autocast rounds the bias to bf16
             "bias32": None if bo is None else torch.cat([bo.detach().to(torch.bfloat16).to(torch.float32), bo.detach().to(torch.float32)]).contiguous()}   # [rounded | exact]
    module._fpf_cache = cache
    return cache

# pinned tile table keyed (form, CZ); BZ=16 keeps the [CH, CH, BZ] weight slab at 32 KB and the whole program under the 227 KB smem limit
_CFG = {("scalar_norm", 256): dict(BI=4, BJ=8, BK=32, BZ=16, GROUP_M=8, num_warps=8, num_stages=4, AT=True),                              # a4 (Triton 3.3-compatible)
        ("mask_norm", 128): dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1, FP=True)}                  # f1 (Triton >= 3.4)
_CFG_NO_TMA = {("mask_norm", 128): dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, AT=True, PM=1)}                 # p8 (Triton 3.3-compatible fallback; 20.6 ms @705 = 2.25x in the sweep)
# per compute capability (additive; a card absent here takes the two rows above): the f1 / p8 rows need 198,656 B of shared memory per program — over sm_80's
# 166,912 B opt-in limit (A100) — and TMA is a sm_90 instruction. The 8.0 row is the pointer-load layout a2: BK=64, BZ=16, 3 stages = 3 x (BM*BK + BK*BN) x 2 B
# = 147,456 B (A100-SXM4-80GB sweep of every CFG_VARIANTS entry vs the stock module under bf16 autocast, torch 2.12.0+cu130 / triton 3.7.0: same numerics
# class — unchunked ratio_rms 1.000 / max 1.000, bitwise fraction 0.9995; chunked (N > 384) ratio_rms 0.61 — run-to-run bitwise; x1.21 vs stock at N=400|705,
# S=1024, parity (x0.98) at N=199 where stock's own unchunked path is fast: the engine adapter serves the kernel above 384 tokens on this card)
_CFG_BY_CC = {(8, 0): {("mask_norm", 128): dict(BI=4, BJ=8, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=3, AT=True)}}                 # a2
CFG_VARIANTS = {"v1": dict(BI=4, BJ=4, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=2),
                "v2": dict(BI=4, BJ=4, BK=32, BZ=16, GROUP_M=8, num_warps=4, num_stages=3),
                "v3": dict(BI=4, BJ=4, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=2),
                "v4": dict(BI=8, BJ=4, BK=32, BZ=16, GROUP_M=8, num_warps=8, num_stages=2),
                # a*: tile-contiguous A2 layout (AT=True), pointer loads, mask-free K loop (Triton 3.3-compatible)
                "a1": dict(BI=4, BJ=4, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=3, AT=True),      # 128 x 128 x 64
                "a2": dict(BI=4, BJ=8, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=3, AT=True),      # 128 x 256 x 64 (144 KB smem)
                "a3": dict(BI=8, BJ=4, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=3, AT=True),      # 256 x 128 x 64
                "a4": dict(BI=4, BJ=8, BK=32, BZ=16, GROUP_M=8, num_warps=8, num_stages=4, AT=True),      # 128 x 256 x 32, 4 stages (96 KB)
                "a5": dict(BI=4, BJ=4, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=4, AT=True),      # 128 x 128 x 64, 4 stages (128 KB)
                "a6": dict(BI=4, BJ=8, BK=64, BZ=16, GROUP_M=16, num_warps=8, num_stages=2, AT=True),     # 128 x 256 x 64, 2 stages (96 KB), wide group
                # m*: TMA descriptor loads (Triton >= 3.4 only)
                "m1": dict(BI=4, BJ=4, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=3, TMA=True),     # 128 x 128 x 64
                "m2": dict(BI=4, BJ=8, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=3, TMA=True),     # 128 x 256 x 64 (144 KB)
                "m3": dict(BI=4, BJ=8, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=4, TMA=True),     # 128 x 256 x 64, 4 stages (192 KB)
                "m4": dict(BI=8, BJ=4, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=3, TMA=True),     # 256 x 128 x 64
                "m5": dict(BI=4, BJ=8, BK=128, BZ=16, GROUP_M=8, num_warps=8, num_stages=2, TMA=True),    # 128 x 256 x 128, 2 stages (192 KB)
                "m6": dict(BI=4, BJ=8, BK=64, BZ=16, GROUP_M=16, num_warps=8, num_stages=3, TMA=True),    # 128 x 256 x 64, group 16
                "m7": dict(BI=4, BJ=4, BK=128, BZ=16, GROUP_M=8, num_warps=8, num_stages=3, TMA=True),    # 128 x 128 x 128, 3 stages (192 KB)
                "m8": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, TMA=True),     # 128 x 256 x 64, BZ=32 epilogue
                # p*: PROJ_MODE=1 (single K=1024 dot per z-chunk on the permuted tile; Triton >= 3.4)
                "p1": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1),
                "p2": dict(BI=4, BJ=8, BK=64, BZ=64, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1),
                "p3": dict(BI=4, BJ=8, BK=64, BZ=128, GROUP_M=8, num_warps=8, num_stages=2, TMA=True, PM=1),
                "p4": dict(BI=4, BJ=4, BK=64, BZ=64, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1),
                "p5": dict(BI=4, BJ=4, BK=64, BZ=128, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1),
                "p6": dict(BI=8, BJ=4, BK=64, BZ=64, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1),
                "p7": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=16, num_warps=8, num_stages=3, TMA=True, PM=1),
                "p8": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, AT=True, PM=1),    # pointer loads + PM=1
                "p9": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=16, num_stages=3, TMA=True, PM=1),  # 16 warps (128-reg cap, no spills?)
                # f*: fused Triton prologue (LN + proj + mask -> A2/BT layouts) + Triton num_mask + the p-kernel
                "f1": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1, FP=True),
                "f2": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=16, num_warps=8, num_stages=3, TMA=True, PM=1, FP=True),
                "f3": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=16, num_stages=3, TMA=True, PM=1, FP=True),
                "f4": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1, FP=True, PRO_BS=64, PRO_BIP=4),
                "f5": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1, FP=True, PRO_BS=16, PRO_BIP=16),
                "f6": dict(BI=8, BJ=4, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1, FP=True),
                "f7": dict(BI=4, BJ=8, BK=64, BZ=16, GROUP_M=8, num_warps=8, num_stages=4, TMA=True, PM=1, FP=True),
                "f8": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=4, num_warps=8, num_stages=3, TMA=True, PM=1, FP=True),
                "f9": dict(BI=4, BJ=8, BK=128, BZ=32, GROUP_M=8, num_warps=8, num_stages=2, TMA=True, PM=1, FP=True),
                "f10": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=32, num_warps=8, num_stages=3, TMA=True, PM=1, FP=True),
                "f11": dict(BI=4, BJ=8, BK=64, BZ=32, GROUP_M=8, num_warps=8, num_stages=3, TMA=True, PM=1, FP=True, PRO_BS=32, PRO_BIP=4, PRO_WARPS=4)}

def _tma_available():
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401
        return True
    except Exception:
        return False

_CC = {}
def device_cc(device=None):
    """(major, minor) of the CUDA device (cached per device); None without CUDA."""
    key = str(device)
    if key not in _CC:
        try:
            _CC[key] = tuple(torch.cuda.get_device_capability(device)) if torch.cuda.is_available() else None
        except Exception:
            _CC[key] = None
    return _CC[key]

def cfg_for(form, CZ, device=None):
    """The launch row for (form, CZ) on this device, ONE decision: FPF_OPM_CFG override -> the card's row (_CFG_BY_CC) -> the TMA row (_CFG) when the tensor
    descriptor API is importable -> the pointer row (_CFG_NO_TMA). Returns (cfg dict, source word: override:<name> | cc<M><m> | default | no_tma)."""
    over = _cfg_override()
    if over is not None:
        return over, "override:" + _os.environ.get("FPF_OPM_CFG", "").strip()
    row = (_CFG_BY_CC.get(device_cc(device)) or {}).get((form, CZ))
    if row is not None:
        return dict(row), "cc%d%d" % device_cc(device)
    cfg = _CFG[(form, CZ)]
    if cfg.get("TMA", False) and not _tma_available():
        return dict(_CFG_NO_TMA[(form, CZ)]), "no_tma"
    return dict(cfg), "default"

def cfg_name(cfg):
    """The CFG_VARIANTS name of a row (the tables' own names f1 / p8 / a2 ...), else None."""
    for name, row in CFG_VARIANTS.items():
        if dict(row) == dict(cfg):
            return name
    return None

import os as _os
_KSTATS = {}   # per-config compiled-kernel stats (regs / spills / smem), printed once
def _cfg_override():
    name = _os.environ.get("FPF_OPM_CFG", "").strip()
    return CFG_VARIANTS.get(name) if name else None

def opm_core(a, b, cache, form, norm_tensor=None, norm_scalar=1.0, proj_in_kernel=True, cfg=None, chunked=False, prebuilt=None):
    """a, b: [S, N, CH] bf16 (already masked / projected); form 'scalar_norm' (EPI 0: / norm_scalar) | 'mask_norm' (EPI 1|2: / norm_tensor[i, j]).
    Returns [N, N, CZ]: bf16 (scalar_norm / mask_norm unchunked) or fp32 (mask_norm chunked stock path).
    prebuilt = (A2, BT, NA, NBj, SK, S, N) from fused_prologue (then a, b are ignored; TMA path only)."""
    if prebuilt is not None:
        A2, BTp, NA, NBj, SK, S, N = prebuilt; CH = cache["CH"]; dev = A2.device
    else:
        S, N, CH = a.shape; dev = a.device
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, (a.dtype, b.dtype)
    CZ = cache["CZ"]
    is_mask_norm = form == "mask_norm"
    epi = 0 if not is_mask_norm else (2 if chunked else 1)
    out = torch.empty(N, N, CZ, device=dev, dtype=(torch.float32 if epi == 2 else torch.bfloat16))
    cfg = cfg or cfg_for(form, CZ, dev)[0]
    BI, BJ, BK = cfg["BI"], cfg["BJ"], cfg["BK"]
    BM, BN = BI * CH, BJ * CH
    use_tma = bool(cfg.get("TMA", False))
    a_t = bool(cfg.get("AT", False)) or use_tma
    num_pid_m = triton.cdiv(N, BI); num_pid_n = triton.cdiv(N, BJ)
    grid = (num_pid_m * num_pid_n,)
    ws = torch.empty(N, N, CH * CH, device=dev, dtype=torch.bfloat16) if not proj_in_kernel else out
    wout, bias = cache["wout_t"], (cache["bias32"] if cache["bias32"] is not None else cache["wout_t"])
    norm_t = norm_tensor if norm_tensor is not None else wout
    common = dict(CH=CH, CZ=CZ, BI=BI, BJ=BJ, BK=BK, BZ=cfg["BZ"], GROUP_M=cfg["GROUP_M"], EPI=epi, HAS_BIAS=cache["bias32"] is not None,
                  PROJ_IN_KERNEL=proj_in_kernel, PROJ_MODE=int(cfg.get("PM", 0)), num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    if prebuilt is not None:
        assert use_tma and NA == num_pid_m * BI and NBj == num_pid_n * BJ and SK % BK == 0, (NA, NBj, SK, cfg)
        NB = NBj * CH; A = A2; B = None
    elif a_t:
        NA = num_pid_m * BI; NBj = num_pid_n * BJ; SK = triton.cdiv(S, BK) * BK; NB = NBj * CH
        ap = a if NA == N else torch.cat([a, a.new_zeros(S, NA - N, CH)], 1)                      # zero-pad i (exact)
        A = torch.zeros(num_pid_m, CH, BI, SK, device=a.device, dtype=a.dtype)
        A[..., :S].copy_(ap.view(S, num_pid_m, BI, CH).permute(1, 3, 2, 0))                      # A2[(iblk, c, bi), s] = a[s, i, c] (pure re-layout)
        A = A.view(NA * CH, SK)
        if use_tma:
            B = b
        else:
            B = torch.zeros(SK, NBj, CH, device=b.device, dtype=b.dtype)
            B[:S, :N].copy_(b)
            B = B.view(SK, NB)
    else:
        NA = N; SK = S; NB = N * CH
        A = a.reshape(S, N * CH).contiguous(); B = b.reshape(S, N * CH).contiguous()
    if use_tma:
        from triton.tools.tensor_descriptor import TensorDescriptor
        if prebuilt is not None:
            BT = BTp
        else:
            BT = torch.zeros(NBj, CH, SK, device=b.device, dtype=b.dtype)
            BT[:N, :, :S].copy_(b.permute(1, 2, 0))                                              # BT[(j, e), s] = b[s, j, e] (pure re-layout)
            BT = BT.view(NB, SK)
        a_desc = TensorDescriptor.from_tensor(A, [BM, BK])
        b_desc = TensorDescriptor.from_tensor(BT, [BN, BK])
        h = _opm_kernel_tma[grid](a_desc, b_desc, wout, bias, norm_t, out, ws, SK, N, num_pid_n, float(norm_scalar), **common)
    else:
        h = _opm_kernel[grid](A, B, wout, bias, norm_t, out, ws, S, SK, N, NB, num_pid_n, float(norm_scalar), A_T=a_t, **common)
    key = (form, epi, proj_in_kernel, tuple(sorted(cfg.items())))
    if key not in _KSTATS:
        md = getattr(h, "metadata", None)
        _KSTATS[key] = dict(n_regs=getattr(h, "n_regs", None), n_spills=getattr(h, "n_spills", None), shared=getattr(md, "shared", None) if md is not None else None,
                            grid=grid[0], NA=NA, SK=SK, NB=NB, fused_prologue=prebuilt is not None)
        print(f"[opt_core.ops.msa_opm] cfg={_os.environ.get('FPF_OPM_CFG', 'default')} {dict(cfg)} epi={epi} -> {_KSTATS[key]}", flush=True)
    if proj_in_kernel:
        return out
    # fallback epilogue with the stock's own torch ops (cuBLAS projection): same rounding points as stock
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if epi == 0:
            o = torch.nn.functional.linear(ws.view(N * N, CH * CH), cache["wout"], cache["bias16"]).view(N, N, CZ)
            o = o / torch.tensor(norm_scalar, device=o.device, dtype=torch.bfloat16)
        elif epi == 1:
            z = ws.view(N, N, CH * CH)                        # the kernel already applied / num_mask (bf16) before the store
            o = torch.nn.functional.linear(z, cache["wout"], cache["bias16"])
        else:
            z = ws.view(N, N, CH * CH)
            o = torch.nn.functional.linear(z, cache["wout"]) + cache["bias32"][CZ:]
    return o

# ---- registry-signature entry points -------------------------------------------------------------------------
from .._fallback import FPFFallback

pack = _pack                     # public name of the weight packer (adapters with their own attribute names hand it a namespace of the form's schema)

def _fallback_cls():
    return FPFFallback

def scalar_norm_replica(module, m):
    """stock: norm = einsum('...abc,...adc->...bdc', mask, mask) + eps, mask = ones [S,N,1] under bf16 autocast -> every element = the bf16 GEMM
    sum over S of 1*1 (exact up to 256, then bf16-rounded by cuBLAS' fp32 accumulate -> bf16 output) + eps -> bf16."""
    S = m.shape[-3]
    ones = m.new_ones(S, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        norm = torch.einsum("...abc,...adc->...bdc", ones, ones)
    return (norm + module.eps).flatten()[0]

def forward_scalar_norm(module, m, mask=None, chunk_size=None, inplace_safe=False, proj_in_kernel=True):
    """form 'scalar_norm': OuterProductMean.forward(m[S,N,c_m], mask=None, chunk_size, inplace_safe) -> [N,N,c_z] (attributes layer_norm / linear_1 / linear_2 / linear_out / eps)."""
    F = _fallback_cls()
    if m.dim() != 3: raise F("batched_m_not_supported")
    if mask is not None and not bool(torch.all(mask != 0)): raise F("msa_mask_not_all_ones")
    cache = _pack(module, "scalar_norm")
    ln = module.layer_norm(m)
    a = module.linear_1(ln); b = module.linear_2(ln)                       # stock ops -> bitwise prologue (bf16 under autocast)
    norm = scalar_norm_replica(module, m)
    return opm_core(a.contiguous().to(torch.bfloat16), b.contiguous().to(torch.bfloat16), cache, "scalar_norm", norm_scalar=float(norm), proj_in_kernel=proj_in_kernel)

def forward_scalar_norm_ws(module, m, mask=None, chunk_size=None, inplace_safe=False):
    return forward_scalar_norm(module, m, mask, chunk_size, inplace_safe, proj_in_kernel=False)

def num_mask_replica(mask_b, chunked):
    """Replicates stock's num_mask numerics. mask_b: [S, N] bf16 (0/1). unchunked: bf16(exact count over S); chunked (N>384): running bf16 sum of
    exact per-64-row-chunk counts (stock: num_mask += (...).sum(1) in bf16)."""
    S = mask_b.shape[0]
    mf = mask_b.to(torch.float32)
    with torch.autocast("cuda", enabled=False):
        if not chunked:
            return (mf.t() @ mf).to(torch.bfloat16).clamp(min=1)
        acc = None
        for s0 in range(0, S, 64):
            c = (mf[s0:s0 + 64].t() @ mf[s0:s0 + 64]).to(torch.bfloat16)
            acc = c if acc is None else (acc + c)
    return acc.clamp(min=1)

def forward_mask_norm(module, m, mask, chunk_size=None, proj_in_kernel=True):
    """form 'mask_norm': OuterProductMean.forward(m[B,S,N,c_in], mask[B,S,N], chunk_size) -> [B,N,N,c_out] (attributes norm / proj_a / proj_b / proj_o)."""
    F = _fallback_cls()
    if m.dim() != 4: raise F("rank")
    cache = _pack(module, "mask_norm")
    Bn, S, N, _ = m.shape
    chunked = chunk_size is not None
    cfg = cfg_for("mask_norm", cache["CZ"], m.device)[0]          # the card's row, else f1 (TMA) / p8 (triton < 3.4: pointer-load A2 layout, same math, TIER2 same class — program rule: no 3.4+-only feature without a fallback)
    if cfg.get("FP", False) and proj_in_kernel:
        # fused Triton prologue (LN + proj_a/proj_b + mask, written in the GEMM layouts) + Triton num_mask replica + fused OPM kernel
        outs = []
        for bi in range(Bn):
            pre = fused_prologue(m[bi], mask[bi], cache["ln"], cache["wa_t"], cache["wb_t"], cache["ba16"], cache["bb16"], cfg["BI"], cfg["BJ"], cfg["BK"], cache["CH"],
                                 num_warps=cfg.get("PRO_WARPS", 8), BS=cfg.get("PRO_BS", 32), BIP=cfg.get("PRO_BIP", 8))
            nm = num_mask_triton(mask[bi], N, S, chunked)
            outs.append(opm_core(None, None, cache, "mask_norm", norm_tensor=nm, proj_in_kernel=True, chunked=chunked, cfg=cfg, prebuilt=pre + (S, N)))
        return outs[0].unsqueeze(0) if Bn == 1 else torch.stack(outs, 0)
    mask_b = mask.unsqueeze(-1).to(m)
    ln = module.norm(m)
    # stock: a = proj_a(ln) * mask -> under bf16 autocast proj_a outputs bf16, and * fp32 mask type-promotes to fp32; every element is a bf16 value
    # times 0/1, i.e. exactly bf16-representable -> the bf16 cast is value-preserving (bitwise prologue) and halves operand bytes / enables bf16 MMA
    mask16 = mask_b.to(torch.bfloat16)
    a = module.proj_a(ln).mul_(mask16); b = module.proj_b(ln).mul_(mask16)
    if a.dtype != torch.bfloat16: a = a.to(torch.bfloat16); b = b.to(torch.bfloat16)
    outs = []
    for bi in range(Bn):
        nm = num_mask_replica(mask_b[bi, :, :, 0], chunked).contiguous()
        outs.append(opm_core(a[bi].contiguous(), b[bi].contiguous(), cache, "mask_norm", norm_tensor=nm, proj_in_kernel=proj_in_kernel, chunked=chunked, cfg=cfg))
    return torch.stack(outs, 0)

def forward_mask_norm_ws(module, m, mask, chunk_size=None):
    return forward_mask_norm(module, m, mask, chunk_size, proj_in_kernel=False)



# ---- integration declaration (Integration format) ----------------------------------------------------------------------------------------
FPF_META = {
    "outer_product_mean": {"ln_mode": "in-kernel", "ln_note": "fused prologue: fp32 two-pass LN stats in registers (torch order NOT reproduced) -> TIER2; bf16 at the stock autocast points",
                           "stages": "LN+proj_a+proj_b+mask:K_opm_prologue|num_mask:K_num_mask(bitwise replica of stock)|outer GEMM over S:K_opm(TMA, bf16 operands, fp32 acc -> bf16 as stock einsum)|/num_mask:K_opm|proj_out(K=1024)+bias:K_opm",
                           "z_passes_removed": {"mask_norm_chunked@705": "8-chunk loop (8 x [N,N,128] write/read/div/cast/accumulate) + fp32 LN out write/read + casts -> single kernel, output written once"},
                           "tensor": "z_update[N,N,c_z]", "default": "f1 (TMA, triton >= 3.4); fallback p8 pointer-load layout on triton 3.3.1; memory: prologue layouts A2/BT bf16 ~2 x [N*32, S] (0.43 GB @705/S=4724) + out, no [N,N,1024] workspace",
                           "stock_class": {"mask_norm": "OuterProductMean(64,32,128) norm/proj_a/proj_b/proj_o, chunked(N>384)/unchunked", "scalar_norm": "OuterProductMean(128,256,32) layer_norm/linear_1/linear_2/linear_out — WITHDRAWN at c_z 256 on triton 3.3.1 (no TMA; pointer-load variants slower than the unchunked cuBLAS stock)"},
                           "mma_operands": "bf16 (RN from fp32 at the stock autocast points); fp32 accumulate; no TF32 path (v1's fp32-operand TF32 path was the withdrawn slow variant)",
                           "sweep": "38 configs (the OPM tile sweep table of record)"},
}
