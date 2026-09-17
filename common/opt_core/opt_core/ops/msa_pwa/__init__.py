"""opt_core.ops.msa_pwa — MSA pair-weighted averaging with the [S, N, N] attention never materialised per row beyond what the GEMM needs, and the
LN/projection prologue + gate/out-projection epilogue fused.

STOCK FORMS reproduced (two attribute schemas of the same op; the adapter names the form by the entry point it calls):
  * form 'unmasked' — MSAPairWeightedAveraging(c_m=128, c=8, c_z=256, n_heads=8).forward(m, z), attributes layernorm_m / linear_no_bias_mv / linear_no_bias_mg /
    layernorm_z / linear_no_bias_z / softmax_w / linear_no_bias_out:
        m = LN_m(m); v = linear_mv(m) -> [S,N,H,c]; b = linear_z(LN_z(z)) -> [N,N,H]; g = sigmoid(linear_mg(m)) -> [S,N,H,c]
        w = softmax(b, dim=-2)  (over j; fp32? under autocast softmax runs in fp32 and returns... torch autocast casts softmax to fp32 -> w fp32 [N,N,H])
        o = einsum('...ijh,...sjhc->...sihc', w, v)  -> bf16 GEMM per head (w cast to bf16 by autocast's einsum? einsum is in the autocast bf16 list ->
        operands cast to bf16, fp32 accumulate, bf16 out); o = g * o ; out = linear_out(o.reshape(S,N,H*c)) [S,N,c_m]
  * form 'masked' — PairWeightedAveraging(c_m=64, c_z=128, c_h=32, num_heads=8).forward(m, z, mask, chunk_heads), attributes norm_m / proj_m / proj_g / norm_z /
    proj_z / proj_o / inf:
        m = norm_m(m); z = norm_z(z); b = proj_z(z) [B,N,N,H]; b = b + (1-mask)*-inf-like bias; w = softmax(b over j) (fp32 via autocast);
        UNCHUNKED: v = proj_m(m) -> [B,S,N,H,c_h]; g = sigmoid(proj_g(m)); o = einsum('bmnh,bsnhd->bsmhd', w, v); o = g*o; out = proj_o(o)
        CHUNKED (N>384): per head h: v_h, g_h from sliced weights; o_h = einsum(w_h, v_h) ; g_h*o_h ; out += o_h @ proj_o[:, h-slice]  (bf16 accumulate!)
CANDIDATE: one Triton kernel per (head-group, row-tile i, MSA-tile s): out_tile[s-rows, i-rows, H*c] = sum_j w[i,j,h] v[s,j,h,:], i.e. a batched GEMM
        over s with M = S-tile, N = c, K = N(j)... Reformulated per head h as  O_h[s, i, c] = sum_j V_h[s, j, c] * W_h[i, j]  -> for a fixed s-tile:
        O_h[s-tile, i, :] = W_h[i, :] @ V_h[s-tile, :, :]  = GEMM with M = BS (MSA rows), N = c, K = N(j) (the contraction), batched over i.
        Efficient form: for each (h, i-tile of BI tokens, s-tile of BS rows):  acc[BS, BI*c]; the contraction runs over j for every (s, i) pair:
        acc[s, i, c] = sum_j V[s, j, c] W[i, j]
        = for fixed c: (V[:, :, c] @ W^T)[s, i].  So per head and per channel c it is a plain GEMM [S, N] x [N, N] -> [S, N]; stacking the c
        channels gives a batched GEMM with batch = H*c = 64 (c=8 form) / 256 (c=32 form), each [S, N] x [N, N]: FLOPs = 2 S N^2 H c (the stock
        einsum does exactly this via bmm with batch H and N = c, which is why it runs at <10 % of peak: N=c=8 tiles).
        This kernel: program = (h, s-tile BS, i-tile BI); loop over j in BJ tiles: w_t = W_h[i-tile, j-tile] (BI x BJ, bf16 as stock),
        v_t = V_h[s-tile, j-tile, :] as [BS, BJ*c]... contraction over j with c as a batch dim means a tl.dot with N-dim = c = 8 (too small for MMA
        (min 16)).  Instead transpose roles: contract over j with M = BS*c: acc[(s,c), i] = sum_j V'[(s,c), j] W[i, j]:  A = V_h[s-tile] viewed as
        [(s, c), j] — that is V stored [S, c, N] (j innermost) — then it is ONE GEMM per head: [S*c, N] x [N, N]^T -> [S*c, N] with K = N = 705:
        tile (BM = BS*c rows, BN = BI cols, BK = BJ) and full MMA efficiency.  The operand V must be laid out [S, H, c, N] (j innermost): produced
        by the prologue GEMM  v = LN(m) @ Wv  as [S, N, H*c] -> stored transposed via a strided store in the prologue kernel (or a
        torch .transpose().contiguous() = one extra pass of S*N*H*c*2 B = 185 MB at S=2048,N=705,Hc=64 — acceptable for v1).
        Epilogue: acc[(s,c), i] (fp32) -> bf16 (stock rounds the einsum output) ; * sigmoid(g)[s, i, h, c] (gate, bf16 as stock) -> store o[s, i, h, c]
        then out = linear_out(o) (cuBLAS, bitwise-class with stock).  The fully fused gate+out-projection epilogue (K = H*c = 64) is v2.
        w = softmax(b) is computed ONCE per call with the stock torch ops (fp32 softmax -> bf16 cast exactly as autocast does).
TIER: 2 by construction (K-tiling over j differs from cuBLAS bmm).  r2r bitwise; batch/row invariant (s tiles independent); co-tenancy invariant.
"""
import torch, triton, triton.language as tl
import torch.nn.functional as F

@triton.jit(do_not_specialize=['S'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _pwa_core_kernel(V, W, G, O, S, N,
                     H: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
    # V, G, O: [S, N, H*C] bf16 (natural layouts of proj_m / sigmoid(proj_g) / the gated output);  W: [H, N, N] bf16 (w[h, i, j], softmax over j)
    # program = (s-tile of BS MSA rows, i-tile of BI tokens); loops over heads h (L2 reuse of the V lines) and over j-tiles (K of the GEMM):
    #   acc[i, (s, c)] = sum_j W[h, i, j] * V[s, j, h*C + c]     -> tl.dot(w_t[BI, BJ], v_t[BJ, BS*C])
    pid_s = tl.program_id(0).to(tl.int64)                    # int64 program indices: every V / W / G / O element offset below is formed in int64
    pid_i = tl.program_id(1).to(tl.int64)
    s0 = pid_s * BS
    i0 = pid_i * BI
    ss = s0 + tl.arange(0, BS)
    ii = i0 + tl.arange(0, BI)
    cc = tl.arange(0, C)
    smask = ss < S
    imask = ii < N
    HC: tl.constexpr = H * C
    for h in range(0, H):
        acc = tl.zeros((BI, BS * C), dtype=tl.float32)
        for j0 in range(0, N, BJ):
            jj = j0 + tl.arange(0, BJ)
            jmask = jj < N
            w_t = tl.load(W + (h * N + ii)[:, None] * N + jj[None, :], mask=imask[:, None] & jmask[None, :], other=0.0)                       # [BI, BJ]
            v3 = tl.load(V + (ss[None, :, None] * N + jj[:, None, None]) * HC + h * C + cc[None, None, :],
                         mask=jmask[:, None, None] & smask[None, :, None], other=0.0)                                                         # [BJ, BS, C]
            acc += tl.dot(w_t, tl.reshape(v3, (BJ, BS * C)))
        wv16 = acc.to(tl.bfloat16)                                                                                                        # stock: einsum output bf16
        g3 = tl.load(G + (ss[None, :, None] * N + ii[:, None, None]) * HC + h * C + cc[None, None, :],
                     mask=imask[:, None, None] & smask[None, :, None], other=0.0)                                                             # [BI, BS, C]
        g2 = tl.reshape(g3, (BI, BS * C))
        o = (g2.to(tl.float32) * wv16.to(tl.float32)).to(tl.bfloat16)                                                                     # stock: g * wv in bf16 (exact product, one rounding)
        o3 = tl.reshape(o, (BI, BS, C))
        tl.store(O + (ss[None, :, None] * N + ii[:, None, None]) * HC + h * C + cc[None, None, :], o3, mask=imask[:, None, None] & smask[None, :, None])

# ---------------------------------------------------------------------------------------------------------------------------------
# v2 kernel.  Three changes vs the kernel above (variant "v0"):
#   (1) GRID ORDER: program_id(0) = i-tile (fastest).  The N/BI programs that share one s-tile run back-to-back, so the V rows of that
#       s-tile (BS*N*H*C*2 B = 1.4 MB @705/BS=4) are fetched from HBM once and re-hit in L2 by the other i-tiles.  The v0 kernel
#       has the s-tile fastest: every i-tile re-streams ALL of V from HBM (N/BI = 12 x 1.7 GB @705, S=4724 -> >= 6 ms of HBM time alone).
#   (2) PADDED W: w_h is stored [H, NPI, NPJ] with NPI/NPJ = multiples of BI/BJ (zero padded).  Row stride N=705 elements (1410 B) is
#       not 16 B aligned, so the v0 W tile load cannot vectorise (and Triton has no divisibility hint for N at runtime); with the
#       padded stride every W row is 128 B aligned and the W tile needs no mask at all (padded entries are exactly 0; the V operand is
#       masked on j, so 0 * masked-0 contributes nothing, and the i-padding rows are never stored).
#   (3) 2D V OPERAND: the B operand [BJ, BS*C] is loaded as a 2D tensor with a precomputed column offset (s_local*N*H*C + c) instead of a
#       3D gather + tl.reshape, so the load feeds tl.dot directly and Triton's software pipeliner (cp.async -> smem -> wgmma) applies.
#   Optional FUSE_SIG: the gate is read as the RAW proj_g output and sigmoid is computed in fp32 in the epilogue, rounded to bf16 like
#   torch's bf16 sigmoid (fp32 internal -> bf16 out), then g*wv as stock.  Saves one 1.7 GB read + 1.7 GB write (@705, S=4724).
#   Numerics: identical contraction structure to the shipped kernel (fp32 accumulate over j in BJ-tiles; one bf16 rounding of wv as stock;
#   g*wv in fp32 rounded once to bf16 as stock).  No atomics, no split-K, fixed tiles (no autotune).
# ---------------------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=['S'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _pwa_core_kernel_v2(V, W, G, O, S, N, NPI, NPJ,
                        H: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, FUSE_SIG: tl.constexpr):
    # V, G, O: [S, N, H*C] bf16 (natural layouts);  W: [H, NPI, NPJ] bf16 zero-padded (w[h, i, j], softmax over j)
    # program = (i-tile of BI tokens [fastest], s-tile of BS MSA rows); loops over heads h and j-tiles (K of the GEMM):
    #   acc[i, (s, c)] = sum_j W[h, i, j] * V[s, j, h*C + c]     -> tl.dot(w_t[BI, BJ], v_t[BJ, BS*C])
    pid_i = tl.program_id(0).to(tl.int64)                    # int64 program indices: every V / W / G / O / OUT element offset below is formed in int64
    pid_s = tl.program_id(1).to(tl.int64)
    s0 = pid_s * BS
    i0 = pid_i * BI
    BSC: tl.constexpr = BS * C
    HC: tl.constexpr = H * C
    ii = i0 + tl.arange(0, BI)
    imask = ii < N
    kk = tl.arange(0, BSC)                      # accumulator column = (s_local, c)
    ks = kk // C
    kc = kk % C
    ss = s0 + ks
    smask = ss < S
    vcol = ss * (N * HC) + kc                   # [BSC] column offsets into V (row s, channel c of head h added below)
    for h in range(0, H):
        acc = tl.zeros((BI, BSC), dtype=tl.float32)
        wrow = W + (h * NPI + ii) * NPJ         # [BI] (no mask: padded rows/cols are zero and in-bounds)
        vbase = V + vcol + h * C                # [BSC]
        for j0 in range(0, N, BJ):
            jj = j0 + tl.arange(0, BJ)
            jmask = jj < N
            w_t = tl.load(wrow[:, None] + jj[None, :])                                                              # [BI, BJ]
            v_t = tl.load(vbase[None, :] + jj[:, None] * HC, mask=jmask[:, None] & smask[None, :], other=0.0)      # [BJ, BSC]
            acc = tl.dot(w_t, v_t, acc)
        wv16 = acc.to(tl.bfloat16)                                                                                  # stock: einsum output bf16
        goff = (ss[None, :] * N + ii[:, None]) * HC + h * C + kc[None, :]                                           # [BI, BSC]
        gmask = imask[:, None] & smask[None, :]
        g = tl.load(G + goff, mask=gmask, other=0.0)
        if FUSE_SIG:
            g = tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16)                                                        # torch bf16 sigmoid: fp32 internal -> bf16
        o = (g.to(tl.float32) * wv16.to(tl.float32)).to(tl.bfloat16)                                                # stock: g * wv in bf16 (one rounding)
        tl.store(O + goff, o, mask=gmask)

# ---------------------------------------------------------------------------------------------------------------------------------
# v3 "hp" kernel: TWO heads per head-loop iteration (two accumulators).  In the natural [S, N, H*C] layout one 128 B line of V holds
# two heads' C=32 channels of one (s, j); the single-head loop consumes each line in two halves one full head-iteration apart
# (~50 MB of streaming in between = the whole L2), so the second half is likely an HBM re-fetch.  Loading both halves back-to-back
# makes every V line fully consumed on first touch.  Same contraction/rounding as v2.
# ---------------------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=['S'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _pwa_core_kernel_hp(V, W, G, O, S, N, NPI, NPJ,
                        H: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, FUSE_SIG: tl.constexpr):
    pid_i = tl.program_id(0).to(tl.int64)                    # int64 program indices: every V / W / G / O / OUT element offset below is formed in int64
    pid_s = tl.program_id(1).to(tl.int64)
    s0 = pid_s * BS
    i0 = pid_i * BI
    BSC: tl.constexpr = BS * C
    HC: tl.constexpr = H * C
    ii = i0 + tl.arange(0, BI)
    imask = ii < N
    kk = tl.arange(0, BSC)
    ks = kk // C
    kc = kk % C
    ss = s0 + ks
    smask = ss < S
    vcol = ss * (N * HC) + kc
    for hp in range(0, H // 2):
        h0 = 2 * hp
        acc0 = tl.zeros((BI, BSC), dtype=tl.float32)
        acc1 = tl.zeros((BI, BSC), dtype=tl.float32)
        wrow0 = W + (h0 * NPI + ii) * NPJ
        wrow1 = wrow0 + NPI.to(tl.int64) * NPJ                  # the next head's W plane: NPI * NPJ (~N^2) formed in int64
        vbase0 = V + vcol + h0 * C
        vbase1 = vbase0 + C
        for j0 in range(0, N, BJ):
            jj = j0 + tl.arange(0, BJ)
            jmask = jj < N
            vm = jmask[:, None] & smask[None, :]
            w0 = tl.load(wrow0[:, None] + jj[None, :])
            w1 = tl.load(wrow1[:, None] + jj[None, :])
            v0 = tl.load(vbase0[None, :] + jj[:, None] * HC, mask=vm, other=0.0)
            v1 = tl.load(vbase1[None, :] + jj[:, None] * HC, mask=vm, other=0.0)
            acc0 = tl.dot(w0, v0, acc0)
            acc1 = tl.dot(w1, v1, acc1)
        gmask = imask[:, None] & smask[None, :]
        goff0 = (ss[None, :] * N + ii[:, None]) * HC + h0 * C + kc[None, :]
        g0 = tl.load(G + goff0, mask=gmask, other=0.0)
        g1 = tl.load(G + goff0 + C, mask=gmask, other=0.0)
        if FUSE_SIG:
            g0 = tl.sigmoid(g0.to(tl.float32)).to(tl.bfloat16)
            g1 = tl.sigmoid(g1.to(tl.float32)).to(tl.bfloat16)
        o0 = (g0.to(tl.float32) * acc0.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        o1 = (g1.to(tl.float32) * acc1.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(O + goff0, o0, mask=gmask)
        tl.store(O + goff0 + C, o1, mask=gmask)

# ---------------------------------------------------------------------------------------------------------------------------------
# v3 "hm" kernel: V in HEAD-MAJOR layout [S, H, N, C] (j rows of one (s, h) contiguous: the [BJ, C] block is one contiguous 4 KB run, every
# 128 B line fully consumed by the head that touches it).  The layout is produced in the prologue (permute+contiguous pass, or a bmm with
# strided out=, selected by cfg['vhm']).  G and O stay in the natural layout.  Same contraction/rounding as v2.
# ---------------------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=['S'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _pwa_core_kernel_hm(V, W, G, O, S, N, NPI, NPJ,
                        H: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, FUSE_SIG: tl.constexpr):
    pid_i = tl.program_id(0).to(tl.int64)                    # int64 program indices: every V / W / G / O / OUT element offset below is formed in int64
    pid_s = tl.program_id(1).to(tl.int64)
    s0 = pid_s * BS
    i0 = pid_i * BI
    BSC: tl.constexpr = BS * C
    HC: tl.constexpr = H * C
    ii = i0 + tl.arange(0, BI)
    imask = ii < N
    kk = tl.arange(0, BSC)
    ks = kk // C
    kc = kk % C
    ss = s0 + ks
    smask = ss < S
    vcol = ss * (H * N * C) + kc                # head-major: offset(s, h, j, c) = ((s*H + h)*N + j)*C + c
    for h in range(0, H):
        acc = tl.zeros((BI, BSC), dtype=tl.float32)
        wrow = W + (h * NPI + ii) * NPJ
        vbase = V + vcol + h * (N * C)
        for j0 in range(0, N, BJ):
            jj = j0 + tl.arange(0, BJ)
            jmask = jj < N
            w_t = tl.load(wrow[:, None] + jj[None, :])
            v_t = tl.load(vbase[None, :] + jj[:, None] * C, mask=jmask[:, None] & smask[None, :], other=0.0)
            acc = tl.dot(w_t, v_t, acc)
        wv16 = acc.to(tl.bfloat16)
        goff = (ss[None, :] * N + ii[:, None]) * HC + h * C + kc[None, :]
        gmask = imask[:, None] & smask[None, :]
        g = tl.load(G + goff, mask=gmask, other=0.0)
        if FUSE_SIG:
            g = tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16)
        o = (g.to(tl.float32) * wv16.to(tl.float32)).to(tl.bfloat16)
        tl.store(O + goff, o, mask=gmask)

# ---------------------------------------------------------------------------------------------------------------------------------
# Fused PROLOGUE: LN_m -> [proj_m | proj_g -> sigmoid] in one pass (reads m once, writes v and g once).  Replaces the stock op chain
#   norm_m (autocast: bf16 -> fp32 cast, F.layer_norm fp32, fp32 out) -> proj_m / proj_g (autocast: fp32 -> bf16 cast, bf16 GEMM K=64 fp32-acc,
#   bf16 out) -> sigmoid (bf16 in, fp32 internal, bf16 out), which costs 5.5 + 1.24 + 1.24 + 1.13 ms @705/S=4724 (measured, pwatiles2 profile).
# Numerics: LN statistics in fp32 (two-pass mean / centred variance; torch uses Welford in fp32 -> ulp-level fp32 differences, then the SAME
#   bf16 rounding of the normalised row before the GEMM), y = w * (rstd * (x - mean)) + b in torch's op order, rstd = rsqrt(var + eps);
#   GEMM K=64 fp32 accumulate -> bf16 (as stock); g: bf16 round of the proj_g output THEN sigmoid in fp32 -> bf16 (stock op order).
# ---------------------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=['M'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _ln_vg_kernel(X, LNW, LNB, WVT, WGT, V, G, M, eps,
                  D: tl.constexpr, HC: tl.constexpr, BM: tl.constexpr, SIG: tl.constexpr, WITH_G: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)                      # int64 row indices: rows * D and rows * HC are formed in int64
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    dd = tl.arange(0, D)
    cc = tl.arange(0, HC)
    x = tl.load(X + rows[:, None] * D + dd[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)       # [BM, D]
    mean = tl.sum(x, 1) / D
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, 1) / D
    rstd = tl.rsqrt(var + eps)
    w = tl.load(LNW + dd).to(tl.float32)
    b = tl.load(LNB + dd).to(tl.float32)
    y = (w[None, :] * (rstd[:, None] * xc) + b[None, :]).to(tl.bfloat16)                                   # autocast: fp32 LN out -> bf16 GEMM operand
    wv = tl.load(WVT + dd[:, None] * HC + cc[None, :])                                                      # [D, HC] bf16 (pre-transposed weight)
    v = tl.dot(y, wv).to(tl.bfloat16)                                                                       # [BM, HC]
    tl.store(V + rows[:, None] * HC + cc[None, :], v, mask=rmask[:, None])
    if WITH_G:
        wg = tl.load(WGT + dd[:, None] * HC + cc[None, :])
        g = tl.dot(y, wg).to(tl.bfloat16)                                                                   # stock: proj_g output bf16
        if SIG:
            g = tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16)                                                # stock: sigmoid(bf16) = fp32 internal -> bf16
        tl.store(G + rows[:, None] * HC + cc[None, :], g, mask=rmask[:, None])

def ln_proj_vg(m, ln, w_v, w_g, sig=True, BM=64, num_warps=8, num_stages=2, with_g=True):
    """m: [..., D] bf16; ln: nn.LayerNorm(D) (fp32 params); w_v, w_g: nn.Linear weights [HC, D] -> (v, g) [..., HC] bf16 (g = sigmoid(proj_g) if sig)."""
    D = m.shape[-1]; HC = w_v.shape[0]
    x = m.reshape(-1, D)
    # the kernel loads x in ITS OWN dtype and converts to fp32 for the statistics (fp32 m -> fp32 LN exactly as stock; bf16 m -> same as stock)
    if not x.is_contiguous(): x = x.contiguous()
    M = x.shape[0]
    wvt = w_v.detach().to(torch.bfloat16).t().contiguous()                                                 # autocast casts the fp32 weight to bf16
    wgt = w_g.detach().to(torch.bfloat16).t().contiguous()
    lw = ln.weight.detach().float().contiguous(); lb = ln.bias.detach().float().contiguous()
    v = torch.empty((M, HC), dtype=torch.bfloat16, device=m.device); g = torch.empty_like(v) if with_g else v
    _ln_vg_kernel[(triton.cdiv(M, BM),)](x, lw, lb, wvt, wgt, v, g, M, float(ln.eps), D=D, HC=HC, BM=BM, SIG=bool(sig), WITH_G=bool(with_g), num_warps=num_warps, num_stages=num_stages)
    if not with_g:
        return v.view(*m.shape[:-1], HC), None
    return v.view(*m.shape[:-1], HC), g.view(*m.shape[:-1], HC)

# ---------------------------------------------------------------------------------------------------------------------------------
# "fo" kernels: the out-projection proj_o is FUSED into the epilogue.  Per head h: o_h = bf16(g_h * bf16(wv_h)) exactly as before, then
#   out_acc[(s,i), cm] += o_h[(s,i), c] @ WoT[h*C + c, cm]   (fp32 accumulate over c within the head and across heads)  -> bf16 store of
#   [S, N, CM] at the end.  Stock UNCHUNKED: one cuBLAS GEMM K=H*C with fp32 accumulation -> bf16 (same class, different K order);
#   stock CHUNKED: bf16 accumulation of the 8 per-head partials (less accurate than this).  Removes the 1.7 GB O write + 1.7 GB proj_o read.
#   fo = one head per iteration; fohp = two heads per iteration (full 128 B V-line use for C=32).
# ---------------------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=['S'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _pwa_fo_kernel(V, W, G, WOT, OUT, S, N, NPI, NPJ,
                   H: tl.constexpr, C: tl.constexpr, CM: tl.constexpr, BS: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, FUSE_SIG: tl.constexpr):
    pid_i = tl.program_id(0).to(tl.int64)                    # int64 program indices: every V / W / G / O / OUT element offset below is formed in int64
    pid_s = tl.program_id(1).to(tl.int64)
    s0 = pid_s * BS
    i0 = pid_i * BI
    BSC: tl.constexpr = BS * C
    HC: tl.constexpr = H * C
    ii = i0 + tl.arange(0, BI)
    imask = ii < N
    kk = tl.arange(0, BSC)
    ks = kk // C
    kc = kk % C
    ss = s0 + ks
    smask = ss < S
    vcol = ss * (N * HC) + kc
    cc = tl.arange(0, C)
    cm = tl.arange(0, CM)
    out_acc = tl.zeros((BS * BI, CM), dtype=tl.float32)
    for h in range(0, H):
        acc = tl.zeros((BI, BSC), dtype=tl.float32)
        wrow = W + (h * NPI + ii) * NPJ
        vbase = V + vcol + h * C
        for j0 in range(0, N, BJ):
            jj = j0 + tl.arange(0, BJ)
            jmask = jj < N
            w_t = tl.load(wrow[:, None] + jj[None, :])
            v_t = tl.load(vbase[None, :] + jj[:, None] * HC, mask=jmask[:, None] & smask[None, :], other=0.0)
            acc = tl.dot(w_t, v_t, acc)
        wv16 = acc.to(tl.bfloat16)
        goff = (ss[None, :] * N + ii[:, None]) * HC + h * C + kc[None, :]
        gmask = imask[:, None] & smask[None, :]
        g = tl.load(G + goff, mask=gmask, other=0.0)
        if FUSE_SIG:
            g = tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16)
        o = (g.to(tl.float32) * wv16.to(tl.float32)).to(tl.bfloat16)                   # [BI, BSC] = o[i, (s, c)]
        o3 = tl.permute(tl.reshape(o, (BI, BS, C)), (1, 0, 2))                          # [BS, BI, C]
        o2 = tl.reshape(o3, (BS * BI, C))                                               # rows (s, i)
        wo = tl.load(WOT + (h * C + cc)[:, None] * CM + cm[None, :])                    # [C, CM] bf16
        out_acc = tl.dot(o2, wo, out_acc)
    rr = tl.arange(0, BS * BI)
    so = s0 + rr // BI
    io = i0 + rr % BI
    omask = (so < S) & (io < N)
    tl.store(OUT + (so[:, None] * N + io[:, None]) * CM + cm[None, :], out_acc.to(tl.bfloat16), mask=omask[:, None])

@triton.jit(do_not_specialize=['S'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _pwa_fohp_kernel(V, W, G, WOT, OUT, S, N, NPI, NPJ,
                     H: tl.constexpr, C: tl.constexpr, CM: tl.constexpr, BS: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, FUSE_SIG: tl.constexpr):
    pid_i = tl.program_id(0).to(tl.int64)                    # int64 program indices: every V / W / G / O / OUT element offset below is formed in int64
    pid_s = tl.program_id(1).to(tl.int64)
    s0 = pid_s * BS
    i0 = pid_i * BI
    BSC: tl.constexpr = BS * C
    HC: tl.constexpr = H * C
    ii = i0 + tl.arange(0, BI)
    imask = ii < N
    kk = tl.arange(0, BSC)
    ks = kk // C
    kc = kk % C
    ss = s0 + ks
    smask = ss < S
    vcol = ss * (N * HC) + kc
    cc = tl.arange(0, C)
    cm = tl.arange(0, CM)
    out_acc = tl.zeros((BS * BI, CM), dtype=tl.float32)
    for hp in range(0, H // 2):
        h0 = 2 * hp
        acc0 = tl.zeros((BI, BSC), dtype=tl.float32)
        acc1 = tl.zeros((BI, BSC), dtype=tl.float32)
        wrow0 = W + (h0 * NPI + ii) * NPJ
        wrow1 = wrow0 + NPI.to(tl.int64) * NPJ                  # the next head's W plane: NPI * NPJ (~N^2) formed in int64
        vbase0 = V + vcol + h0 * C
        vbase1 = vbase0 + C
        for j0 in range(0, N, BJ):
            jj = j0 + tl.arange(0, BJ)
            jmask = jj < N
            vm = jmask[:, None] & smask[None, :]
            w0 = tl.load(wrow0[:, None] + jj[None, :])
            w1 = tl.load(wrow1[:, None] + jj[None, :])
            v0 = tl.load(vbase0[None, :] + jj[:, None] * HC, mask=vm, other=0.0)
            v1 = tl.load(vbase1[None, :] + jj[:, None] * HC, mask=vm, other=0.0)
            acc0 = tl.dot(w0, v0, acc0)
            acc1 = tl.dot(w1, v1, acc1)
        gmask = imask[:, None] & smask[None, :]
        goff0 = (ss[None, :] * N + ii[:, None]) * HC + h0 * C + kc[None, :]
        g0 = tl.load(G + goff0, mask=gmask, other=0.0)
        g1 = tl.load(G + goff0 + C, mask=gmask, other=0.0)
        if FUSE_SIG:
            g0 = tl.sigmoid(g0.to(tl.float32)).to(tl.bfloat16)
            g1 = tl.sigmoid(g1.to(tl.float32)).to(tl.bfloat16)
        o0 = (g0.to(tl.float32) * acc0.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        o1 = (g1.to(tl.float32) * acc1.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        o0 = tl.reshape(tl.permute(tl.reshape(o0, (BI, BS, C)), (1, 0, 2)), (BS * BI, C))
        o1 = tl.reshape(tl.permute(tl.reshape(o1, (BI, BS, C)), (1, 0, 2)), (BS * BI, C))
        wo0 = tl.load(WOT + (h0 * C + cc)[:, None] * CM + cm[None, :])
        wo1 = tl.load(WOT + (h0 * C + C + cc)[:, None] * CM + cm[None, :])
        out_acc = tl.dot(o0, wo0, out_acc)
        out_acc = tl.dot(o1, wo1, out_acc)
    rr = tl.arange(0, BS * BI)
    so = s0 + rr // BI
    io = i0 + rr % BI
    omask = (so < S) & (io < N)
    tl.store(OUT + (so[:, None] * N + io[:, None]) * CM + cm[None, :], out_acc.to(tl.bfloat16), mask=omask[:, None])

# ---------------------------------------------------------------------------------------------------------------------------------
# "fg" kernel: fo + the GATE computed in-kernel from the (s-tile, i-tile) block of m: y = LN_m(m[s, i, :]) (fp32 stats, bf16 out as
#   autocast), g_h = sigmoid(bf16(y @ Wg_h)) (bf16 GEMM K=64 fp32-acc -> bf16 -> fp32 sigmoid -> bf16, the stock op chain), o_h = bf16(g_h * wv_h),
#   out += o_h @ Wo_h (fp32).  The g tensor never exists (saves its 1.7 GB write + 1.7 GB read @705); the prologue only writes v.
# ---------------------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=['S'])   # the MSA depth (row count) enters as a loop bound / mask only: one compiled class per config, no re-JIT when S mod 16 flips between inputs (MSA track)
def _pwa_fg_kernel(V, W, Mx, LNW, LNB, WGT, WOT, OUT, S, N, NPI, NPJ, eps,
                   H: tl.constexpr, C: tl.constexpr, CM: tl.constexpr, D: tl.constexpr, BS: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
    pid_i = tl.program_id(0).to(tl.int64)                    # int64 program indices: every V / W / G / O / OUT element offset below is formed in int64
    pid_s = tl.program_id(1).to(tl.int64)
    s0 = pid_s * BS
    i0 = pid_i * BI
    BSC: tl.constexpr = BS * C
    HC: tl.constexpr = H * C
    ii = i0 + tl.arange(0, BI)
    imask = ii < N
    kk = tl.arange(0, BSC)
    ks = kk // C
    kc = kk % C
    ss = s0 + ks
    smask = ss < S
    vcol = ss * (N * HC) + kc
    cc = tl.arange(0, C)
    cm = tl.arange(0, CM)
    dd = tl.arange(0, D)
    rr = tl.arange(0, BS * BI)
    so = s0 + rr // BI
    io = i0 + rr % BI
    omask = (so < S) & (io < N)
    x = tl.load(Mx + (so[:, None] * N + io[:, None]) * D + dd[None, :], mask=omask[:, None], other=0.0).to(tl.float32)   # [BS*BI, D]
    mean = tl.sum(x, 1) / D
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, 1) / D
    rstd = tl.rsqrt(var + eps)
    lw = tl.load(LNW + dd).to(tl.float32)
    lb = tl.load(LNB + dd).to(tl.float32)
    y = (lw[None, :] * (rstd[:, None] * xc) + lb[None, :]).to(tl.bfloat16)                                                  # [BS*BI, D] bf16
    out_acc = tl.zeros((BS * BI, CM), dtype=tl.float32)
    for h in range(0, H):
        acc = tl.zeros((BI, BSC), dtype=tl.float32)
        wrow = W + (h * NPI + ii) * NPJ
        vbase = V + vcol + h * C
        for j0 in range(0, N, BJ):
            jj = j0 + tl.arange(0, BJ)
            jmask = jj < N
            w_t = tl.load(wrow[:, None] + jj[None, :])
            v_t = tl.load(vbase[None, :] + jj[:, None] * HC, mask=jmask[:, None] & smask[None, :], other=0.0)
            acc = tl.dot(w_t, v_t, acc)
        wv16 = acc.to(tl.bfloat16)
        wv2 = tl.reshape(tl.permute(tl.reshape(wv16, (BI, BS, C)), (1, 0, 2)), (BS * BI, C))                               # rows (s, i)
        wg = tl.load(WGT + dd[:, None] * HC + (h * C + cc)[None, :])                                                        # [D, C] bf16
        g = tl.dot(y, wg).to(tl.bfloat16)                                                                                   # stock: proj_g out bf16
        g = tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16)                                                                    # stock: sigmoid -> bf16
        o2 = (g.to(tl.float32) * wv2.to(tl.float32)).to(tl.bfloat16)                                                        # stock: g * wv -> bf16
        wo = tl.load(WOT + (h * C + cc)[:, None] * CM + cm[None, :])                                                        # [C, CM] bf16
        out_acc = tl.dot(o2, wo, out_acc)
    tl.store(OUT + (so[:, None] * N + io[:, None]) * CM + cm[None, :], out_acc.to(tl.bfloat16), mask=omask[:, None])

def pwa_core_fg(v, w_h, m, ln_w, ln_b, eps, wgt, wot, out, H, C, cfg):
    """v [S,N,H*C] bf16; w_h [H,N,N] bf16; m [S,N,D] (bf16/fp32, LN input); ln_w/ln_b fp32 [D]; wgt [D, H*C] bf16 (proj_g.weight.T); wot [H*C, CM] bf16; out [S,N,CM]."""
    S, N, HC = v.shape; CM = wot.shape[1]; D = m.shape[-1]
    wp, NPI, NPJ = _pad_w(w_h, cfg["BI"], cfg["BJ"])
    grid = (triton.cdiv(N, cfg["BI"]), triton.cdiv(S, cfg["BS"]))
    _pwa_fg_kernel[grid](v, wp, m, ln_w, ln_b, wgt, wot, out, S, N, NPI, NPJ, float(eps), H=H, C=C, CM=CM, D=D, BS=cfg["BS"], BI=cfg["BI"], BJ=cfg["BJ"],
                         num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return out

def pwa_core_fo(v, w_h, g, wot, out, H, C, cfg):
    """Fused PWA + gate + proj_o: v, g [S, N, H*C] bf16; w_h [H, N, N] bf16; wot [H*C, CM] bf16 (proj_o.weight.T); out [S, N, CM] bf16 (written)."""
    S, N, HC = v.shape; CM = wot.shape[1]
    wp, NPI, NPJ = _pad_w(w_h, cfg["BI"], cfg["BJ"])
    grid = (triton.cdiv(N, cfg["BI"]), triton.cdiv(S, cfg["BS"]))
    kw = dict(H=H, C=C, CM=CM, BS=cfg["BS"], BI=cfg["BI"], BJ=cfg["BJ"], FUSE_SIG=bool(cfg.get("fuse_sig", False)), num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    if cfg["kernel"] == "fo":
        _pwa_fo_kernel[grid](v, wp, g, wot, out, S, N, NPI, NPJ, **kw)
    elif cfg["kernel"] == "fohp":
        _pwa_fohp_kernel[grid](v, wp, g, wot, out, S, N, NPI, NPJ, **kw)
    else:
        raise KeyError(cfg["kernel"])
    return out

def v_head_major(v, H, C, how="permute"):
    """[S, N, H*C] -> [S, H, N, C] contiguous. how='permute': one extra pass (read+write S*N*H*C*2 B each)."""
    S, N, HC = v.shape
    return v.view(S, N, H, C).permute(0, 2, 1, 3).contiguous()

# pinned tile table keyed by C (BS*C = 128 columns of the accumulator)
_CFG = {8: dict(BS=16, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hp", pro="fused", PBM=64),       # = CFG_VARIANTS['fp_r2']
        32: dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=4, kernel="fo", pro="fused", PBM=64, pwarps=4)}   # = CFG_VARIANTS['g_fo4p']

# kernel="v0" = the first kernel (s-tile fastest, 3D gather); kernel="v2" = the second.  fuse_sig only for v2.
CFG_VARIANTS = {"v1": dict(BS=4, BI=64, BJ=64, num_warps=4, num_stages=2), "v2": dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=2),
                "v3": dict(BS=8, BI=32, BJ=64, num_warps=4, num_stages=2), "v4": dict(BS=4, BI=64, BJ=128, num_warps=4, num_stages=3),
                "v5": dict(BS=2, BI=128, BJ=64, num_warps=4, num_stages=2), "c8v1": dict(BS=16, BI=64, BJ=64, num_warps=4, num_stages=2),
                "c8v2": dict(BS=8, BI=128, BJ=64, num_warps=4, num_stages=2), "c8v3": dict(BS=32, BI=32, BJ=64, num_warps=4, num_stages=2),
                # ---- pwa-tiles variants (C=32, masked form) ----
                "v0":   dict(BS=4, BI=64, BJ=64, num_warps=4, num_stages=2, kernel="v0"),                    # shipped kernel, as-is baseline
                "a1":   dict(BS=4, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="v2"),                    # same tile as v0/v1, v2 kernel
                "a2":   dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="v2"),
                "a3":   dict(BS=8, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="v2"),                    # acc 64 x 256
                "a4":   dict(BS=8, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="v2"),                   # acc 128 x 256 (max intensity)
                "a5":   dict(BS=8, BI=128, BJ=32, num_warps=8, num_stages=4, kernel="v2"),                   # smaller K step: 4 % j-waste @705
                "a6":   dict(BS=8, BI=64, BJ=128, num_warps=8, num_stages=2, kernel="v2"),
                "a7":   dict(BS=4, BI=128, BJ=128, num_warps=8, num_stages=3, kernel="v2"),
                "a4s":  dict(BS=8, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="v2", fuse_sig=True),    # a4 + fused sigmoid gate
                "a3s":  dict(BS=8, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="v2", fuse_sig=True),
                "a2s":  dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="v2", fuse_sig=True),
                # ---- pwa-tiles variants (C=8, unmasked form) ----
                "p1":   dict(BS=16, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="v2"),
                "p2":   dict(BS=32, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="v2"),                   # acc 64 x 256
                "p3":   dict(BS=32, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="v2"),                  # acc 128 x 256
                "p4":   dict(BS=16, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="v2"),
                "p5":   dict(BS=32, BI=128, BJ=32, num_warps=8, num_stages=4, kernel="v2"),
                "p3s":  dict(BS=32, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="v2", fuse_sig=True),
                # ---- C=32: head-pair accumulators (hp) and head-major V layout (hm) ----
                "hp1":  dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hp"),                   # 2 x (128 x 128) acc
                "hp2":  dict(BS=8, BI=64, BJ=64, num_warps=8, num_stages=3, kernel="hp"),                    # 2 x (64 x 256) acc
                "hp3":  dict(BS=4, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="hp"),
                "hp4":  dict(BS=8, BI=128, BJ=32, num_warps=8, num_stages=3, kernel="hp"),                   # 2 x (128 x 256) acc: 256 regs/thread
                "hm1":  dict(BS=8, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hm"),
                "hm2":  dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hm"),
                "hm3":  dict(BS=8, BI=128, BJ=64, num_warps=8, num_stages=4, kernel="hm"),
                "hm4":  dict(BS=16, BI=64, BJ=64, num_warps=8, num_stages=3, kernel="hm"),                   # acc 64 x 512
                "hm1s": dict(BS=8, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hm", fuse_sig=True),
                # ---- C=32: stage-count variants of the v2 leaders ----
                "a4x":  dict(BS=8, BI=128, BJ=64, num_warps=8, num_stages=4, kernel="v2"),
                "a7x":  dict(BS=4, BI=128, BJ=128, num_warps=8, num_stages=2, kernel="v2"),
                # ---- C=32: fused LN->proj_m|proj_g->sigmoid prologue (pro='fused') on the leaders above ----
                "f_hp1":  dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hp", pro="fused"),
                "f_a4":   dict(BS=8, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="v2", pro="fused"),
                "f_a7":   dict(BS=4, BI=128, BJ=128, num_warps=8, num_stages=3, kernel="v2", pro="fused"),
                "f_a1":   dict(BS=4, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="v2", pro="fused"),
                "f_hp1b": dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hp", pro="fused", PBM=128, pwarps=8),
                "f_hp1c": dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hp", pro="fused", PBM=32, pwarps=4),
                "f_hp1s": dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hp", pro="fused", fuse_sig=True),
                "f_hp5":  dict(BS=4, BI=128, BJ=32, num_warps=8, num_stages=4, kernel="hp", pro="fused"),
                "f_hp6":  dict(BS=2, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hp", pro="fused"),
                # ---- C=32: fused proj_o epilogue (fo = 1 head/iter, fohp = 2 heads/iter) on top of the fused prologue ----
                "g_fo1":   dict(BS=4, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="fo", pro="fused", PBM=128),
                "g_fo2":   dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="fo", pro="fused", PBM=128),
                "g_fo3":   dict(BS=8, BI=64, BJ=64, num_warps=8, num_stages=3, kernel="fo", pro="fused", PBM=128),
                "g_fohp1": dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="fohp", pro="fused", PBM=128),
                "g_fohp2": dict(BS=4, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="fohp", pro="fused", PBM=128),
                "g_fohp3": dict(BS=2, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="fohp", pro="fused", PBM=128),
                "g_fohp4": dict(BS=4, BI=128, BJ=32, num_warps=8, num_stages=4, kernel="fohp", pro="fused", PBM=128),
                "f_hp1b2": dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hp", pro="fused", PBM=128, pwarps=8),
                # ---- C=8, H=8, HC=64 (unmasked form): natural-layout v2 (p*), head-major hm (q*), head-pair hp (r*), fused prologue (fp_*) ----
                "q1":    dict(BS=16, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="hm"),
                "q2":    dict(BS=32, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="hm"),
                "q3":    dict(BS=32, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hm"),
                "q4":    dict(BS=16, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hm"),
                "r1":    dict(BS=16, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="hp"),
                "r2":    dict(BS=16, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hp"),
                "r3":    dict(BS=32, BI=64, BJ=64, num_warps=8, num_stages=3, kernel="hp"),
                "fp_p1": dict(BS=16, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="v2", pro="fused", PBM=64),
                "fp_p3": dict(BS=32, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="v2", pro="fused", PBM=64),
                "fp_q1": dict(BS=16, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="hm", pro="fused", PBM=64),
                "fp_q3": dict(BS=32, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hm", pro="fused", PBM=64),
                "fp_r2": dict(BS=16, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="hp", pro="fused", PBM=64),
                # ---- C=32: neighbours of g_fo2 ----
                "g_fo4":  dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=4, kernel="fo", pro="fused", PBM=128),
                "g_fo5":  dict(BS=4, BI=128, BJ=128, num_warps=8, num_stages=3, kernel="fo", pro="fused", PBM=128),
                "g_fo6":  dict(BS=2, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="fo", pro="fused", PBM=128),
                "g_fo7":  dict(BS=4, BI=128, BJ=64, num_warps=4, num_stages=3, kernel="fo", pro="fused", PBM=128),
                "g_fo2p": dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="fo", pro="fused", PBM=64, pwarps=4),
                "g_fo9":  dict(BS=4, BI=128, BJ=32, num_warps=8, num_stages=4, kernel="fo", pro="fused", PBM=128),
                "g_fo10": dict(BS=4, BI=64, BJ=128, num_warps=4, num_stages=3, kernel="fo", pro="fused", PBM=128),
                # ---- C=32: in-kernel gate (fg: no g tensor) + prologue tile variants of g_fo4 ----
                "h_fg1":  dict(BS=4, BI=64, BJ=64, num_warps=8, num_stages=3, kernel="fg", PBM=64, pwarps=4),
                "h_fg2":  dict(BS=2, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="fg", PBM=64, pwarps=4),
                "h_fg3":  dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="fg", PBM=64, pwarps=4),
                "h_fg4":  dict(BS=4, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="fg", PBM=64, pwarps=4),
                "h_fg5":  dict(BS=2, BI=128, BJ=32, num_warps=8, num_stages=4, kernel="fg", PBM=64, pwarps=4),
                "h_fg6":  dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=2, kernel="fg", PBM=64, pwarps=4),
                "g_fo4p": dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=4, kernel="fo", pro="fused", PBM=64, pwarps=4),
                "g_fo4q": dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=4, kernel="fo", pro="fused", PBM=32, pwarps=4),
                "g_fo4r": dict(BS=4, BI=128, BJ=64, num_warps=8, num_stages=4, kernel="fo", pro="fused", PBM=64, pwarps=2),
                # ---- C=8 (unmasked form): fused prologue + fused proj_o epilogue ----
                "fp_fo1": dict(BS=16, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="fo", pro="fused", PBM=64, pwarps=4),
                "fp_fo2": dict(BS=16, BI=64, BJ=64, num_warps=4, num_stages=3, kernel="fo", pro="fused", PBM=64, pwarps=4),
                "fp_fo3": dict(BS=32, BI=64, BJ=64, num_warps=8, num_stages=3, kernel="fo", pro="fused", PBM=64, pwarps=4),
                "fp_fo4": dict(BS=8, BI=128, BJ=64, num_warps=8, num_stages=3, kernel="fo", pro="fused", PBM=64, pwarps=4)}
import os as _os
def _cfg_override():
    name = _os.environ.get("FPF_PWA_CFG", "").strip()
    return CFG_VARIANTS.get(name) if name else None

def _pad_w(w_h, BI, BJ):
    """w_h: [H, N, N] bf16 -> zero-padded [H, NPI, NPJ] (NPI/NPJ multiples of BI/BJ). 8 MB @705 -> negligible vs the 1.7 GB operands."""
    Hh, N, _ = w_h.shape
    NPI = triton.cdiv(N, BI) * BI; NPJ = triton.cdiv(N, BJ) * BJ
    if NPI == N and NPJ == N:
        return w_h.contiguous(), NPI, NPJ
    wp = torch.zeros((Hh, NPI, NPJ), dtype=w_h.dtype, device=w_h.device)
    wp[:, :N, :N] = w_h
    return wp, NPI, NPJ

def pwa_core(v, w_h, g, H, C, cfg=None):
    """v: [S, N, H*C] bf16 (proj_m output, natural layout); w_h: [H, N, N] bf16 (softmaxed, w[h,i,j]); g: [S, N, H*C] bf16 -> gated o [S, N, H*C] bf16.
    With cfg['fuse_sig'] the g argument is the RAW proj_g output (pre-sigmoid) and the sigmoid is applied in the kernel epilogue."""
    S, N, HC = v.shape
    o = torch.empty_like(v)
    cfg = cfg or _cfg_override() or _CFG[C]
    kern = cfg.get("kernel", "v0")
    if kern == "v0":
        assert not cfg.get("fuse_sig", False)
        grid = (triton.cdiv(S, cfg["BS"]), triton.cdiv(N, cfg["BI"]))
        _pwa_core_kernel[grid](v, w_h, g, o, S, N, H=H, C=C, BS=cfg["BS"], BI=cfg["BI"], BJ=cfg["BJ"], num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        return o
    wp, NPI, NPJ = _pad_w(w_h, cfg["BI"], cfg["BJ"])
    grid = (triton.cdiv(N, cfg["BI"]), triton.cdiv(S, cfg["BS"]))
    kw = dict(H=H, C=C, BS=cfg["BS"], BI=cfg["BI"], BJ=cfg["BJ"], FUSE_SIG=bool(cfg.get("fuse_sig", False)), num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    if kern == "v2":
        _pwa_core_kernel_v2[grid](v, wp, g, o, S, N, NPI, NPJ, **kw)
    elif kern == "hp":
        assert H % 2 == 0
        _pwa_core_kernel_hp[grid](v, wp, g, o, S, N, NPI, NPJ, **kw)
    elif kern == "hm":
        vhm = v_head_major(v, H, C)                                      # [S, H, N, C]; the extra pass is inside the timed call
        _pwa_core_kernel_hm[grid](vhm, wp, g, o, S, N, NPI, NPJ, **kw)
    else:
        raise KeyError(kern)
    return o

# ---- registry-signature entry points -------------------------------------------------------------------------
from .._fallback import FPFFallback
def _fallback_cls():
    return FPFFallback

def forward_unmasked(module, m, z):
    """form 'unmasked': MSAPairWeightedAveraging.forward(m[S,N,c_m], z[N,N,c_z]) -> m_update [S,N,c_m]  (c=8, H=8; attributes layernorm_m / linear_no_bias_mv /
    linear_no_bias_mg / layernorm_z / linear_no_bias_z / softmax_w / linear_no_bias_out). Stock op order kept for the prologue."""
    Fb = _fallback_cls()
    if m.dim() != 3: raise Fb("batched")
    H, C = module.n_heads, module.c
    S, N, _ = m.shape
    cfg = _cfg_override() or _CFG[C]
    b = module.linear_no_bias_z(module.layernorm_z(z))                    # [N,N,H]   (stock op)
    if cfg.get("pro") == "fused":
        # this form's fused LayerNorm runs in the INPUT dtype with weight/bias cast to it (weight.to(d)); fp32 statistics inside the CUDA ext.
        ln = module.layernorm_m
        D = m.shape[-1]
        lw = (ln.weight if getattr(ln, "weight", None) is not None else torch.ones(D, device=m.device)).detach().to(m.dtype).float()
        lb = (ln.bias if getattr(ln, "bias", None) is not None else torch.zeros(D, device=m.device)).detach().to(m.dtype).float()
        class _LNp: pass
        lnp = _LNp(); lnp.weight = lw; lnp.bias = lb; lnp.eps = float(ln.eps)
        v, g = ln_proj_vg(m, lnp, module.linear_no_bias_mv.weight, module.linear_no_bias_mg.weight, sig=not cfg.get("fuse_sig", False),
                          BM=cfg.get("PBM", 64), num_warps=cfg.get("pwarps", 8))
    else:
        mn = module.layernorm_m(m)
        v = module.linear_no_bias_mv(mn)                                  # [S,N,H*C] (stock op)
        g = module.linear_no_bias_mg(mn)                                  # [S,N,H*C] raw gate (stock op)
        if not cfg.get("fuse_sig", False):
            g = torch.sigmoid(g)                                          # stock op; autocast: sigmoid in fp32 -> bf16 out (else fused in-kernel, same math)
    w = module.softmax_w(b)                                               # stock: nn.Softmax(dim=-2) -> under autocast fp32 softmax, fp32 out
    w16 = w.to(v.dtype)                                                   # the stock einsum (autocast bf16) casts w to bf16 before the bmm
    w_h = w16.permute(2, 0, 1).contiguous()                               # [H,N,N] (w[h,i,j]); N*N*H*2 B = 8 MB @705
    o = pwa_core(v.contiguous(), w_h, g.contiguous(), H, C, cfg)          # [S,N,H*C] = g * (w @ v), bf16 (V read in its natural layout)
    return module.linear_no_bias_out(o)

def forward_masked(module, m, z, mask, chunk_heads=False):
    """form 'masked': PairWeightedAveraging.forward(m[B,S,N,c_m], z[B,N,N,c_z], mask[B,N,N], chunk_heads) -> m_update [B,S,N,c_m]  (c_h=32, H=8; attributes
    norm_m / proj_m / proj_g / norm_z / proj_z / proj_o / inf).
    Reproduces the UNCHUNKED stock branch's op order (the chunk_heads branch is the same function with bf16 partial-sum accumulation of proj_o)."""
    Fb = _fallback_cls()
    if m.dim() != 4: raise Fb("rank")
    H, C = module.num_heads, module.c_h
    Bn, S, N, _ = m.shape
    cfg = _cfg_override() or _CFG[C]
    zn = module.norm_z(z)
    b = module.proj_z(zn).permute(0, 3, 1, 2)                             # [B,H,N,N]
    b = b + (1 - mask[:, None]) * -module.inf
    w = torch.softmax(b, dim=-1)                                          # fp32 under autocast
    if cfg.get("kernel") == "fg":
        # v from the fused LN->proj_m prologue (no g tensor at all); gate + proj_o computed inside the PWA kernel from the m tile
        v, _ = ln_proj_vg(m, module.norm_m, module.proj_m.weight, module.proj_g.weight, sig=False, BM=cfg.get("PBM", 64), num_warps=cfg.get("pwarps", 4), with_g=False)
        CM = module.proj_o.out_features
        wot = module.proj_o.weight.detach().to(v.dtype).t().contiguous()
        wgt = module.proj_g.weight.detach().to(v.dtype).t().contiguous()                                                   # [D, H*C]
        lw = module.norm_m.weight.detach().float().contiguous(); lb = module.norm_m.bias.detach().float().contiguous()
        out = torch.empty((Bn, S, N, CM), dtype=v.dtype, device=v.device)
        mc = m.contiguous()
        g = None
    elif cfg.get("pro") == "fused":
        # fused LN_m -> proj_m | proj_g -> sigmoid (one read of m, one write of v and g); see _ln_vg_kernel for the numerics statement
        v, g = ln_proj_vg(m, module.norm_m, module.proj_m.weight, module.proj_g.weight, sig=not cfg.get("fuse_sig", False),
                          BM=cfg.get("PBM", 64), num_warps=cfg.get("pwarps", 8))
    else:
        mn = module.norm_m(m)
        v = module.proj_m(mn)                                             # [B,S,N,H*C]
        g = module.proj_g(mn)
        if not cfg.get("fuse_sig", False):
            g = g.sigmoid()                                               # stock op (else fused in-kernel: fp32 sigmoid -> bf16, same as torch)
    w16 = w.to(v.dtype).contiguous()                                      # [B,H,N,N] w[h,i,j] (einsum casts to bf16)
    v = v.contiguous()
    if cfg.get("kernel") == "fg":
        for bi in range(Bn):
            pwa_core_fg(v[bi], w16[bi], mc[bi], lw, lb, module.norm_m.eps, wgt, wot, out[bi], H, C, cfg)
        return out
    g = g.contiguous()
    if cfg.get("kernel") in ("fo", "fohp"):
        CM = module.proj_o.out_features
        wot = module.proj_o.weight.detach().to(v.dtype).t().contiguous()  # [H*C, CM] (autocast casts the fp32 weight to bf16)
        out = torch.empty((Bn, S, N, CM), dtype=v.dtype, device=v.device)
        for bi in range(Bn):
            pwa_core_fo(v[bi], w16[bi], g[bi], wot, out[bi], H, C, cfg)
        return out
    if Bn == 1:
        o = pwa_core(v[0], w16[0], g[0], H, C, cfg)[None]                 # view, no copy (the stack copy cost ~1 ms @705)
    else:
        o = torch.stack([pwa_core(v[bi], w16[bi], g[bi], H, C, cfg) for bi in range(Bn)], 0)
    return module.proj_o(o)

# ---- integration declaration (Integration format) ----------------------------------------------------------------------------------------
FPF_META = {
    "msa_pair_weighted_avg": {"ln_mode": "in-kernel", "ln_note": "fused prologue _ln_vg_kernel: fp32 two-pass LN stats in registers (torch order NOT reproduced; v differs from the stock torch chain in 3.0e-5 of elements by 1 bf16 ulp, sigmoid(g) in 5.2e-6) -> TIER2",
                              "stages": "LN_m+proj_m+proj_g+sigmoid:K_ln_vg|LN_z+proj_z+maskbias+softmax:stock|contract(w@v)+gate:K_pwa_fo|proj_o:K_pwa_fo(epilogue, fp32 accumulation over 8 heads)",
                              "z_passes_removed": {"masked_chunk_heads@705": "8 x (proj_m/proj_g slice GEMMs + bf16 copies + sigmoid + gate mul + proj_o partial + bf16 accumulate) + fp32 LN out write/read -> 2 kernels; 338 launches -> ~12"},
                              "tensor": "m_update[S,N,c_m]", "default": "g_fo4p (C=32 masked form): BS=4 x BI=128 x BJ=64, 8 warps, 4 stages; C=8 unmasked form: fp_r2 head-pair kernel",
                              "stock_class": {"masked": "PairWeightedAveraging(64,128,32,8) norm_m/proj_m/proj_g/norm_z/proj_z/proj_o, chunk_heads(N>384)/unchunked", "unmasked": "MSAPairWeightedAveraging(128,8,256,8) layernorm_m/linear_no_bias_mv/linear_no_bias_mg/layernorm_z/linear_no_bias_z/linear_no_bias_out"},
                              "mma_operands": "bf16 (RN from fp32 at the stock autocast points: LN out, v, g, w, o); fp32 accumulate; no TF32 path",
                              "sweep": "45 configs (the PWA tile sweep table of record)"},
}
