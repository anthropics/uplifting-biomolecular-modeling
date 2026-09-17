"""fpf_mkpf.kernels — cross-statement fusions (LayerNorm in registers + the statement that consumes it) over the FlashPairformer Pairformer z-path kernels.

F1  prologue_ln      : tri-attention prologue that reads z ONCE in its native layout (start node: contiguous rows; ending node: z^T by strides — address math only),
                       LayerNorm in registers (fp32 two-pass; LN_ARITH=2 = fpf_triatt_pro's 'welford' emulation of fast_layernorm, imported from fpf_triatt_pro if present),
                       then the glue prologue_v4 projection structure (BN-chunked q|k|v|g weight loop, software-pipelined, one full-K MMA chain per output chunk => projection
                       arithmetic/rounding IDENTICAL to the tested prologue given the same x_ln bits).  Replaces: stock fast_layernorm kernel (+ its [N,N,C] .contiguous() copy of
                       the transposed view on the ending node) + prologue.  Unpadded and PADDED (PAD8 buffers) output addressing via runtime pitches (same as fpf_glue_v2).
F3  transition_ln    : pair transition with LayerNorm in registers: reads z rows once (LN -> a|b dual GEMM -> SwiGLU -> out GEMM -> + residual FROM THE SAME z REGISTERS -> store),
                       glue transition_v3 structure (+warp_specialize on sm_100).  Replaces: stock layernorm1 kernel + y round trip + the residual re-read.
Numerics class: with LN_ARITH=1 (two-pass fp32, IEEE sqrt/div) these are TIER-2 vs stock (fast_layernorm's Welford/butterfly/rsqrt.approx bits are not reproduced; measured 2.4e-4 of x_ln
elements differ by 1 bf16 ulp, err-vs-fp64 ratio 1.00); EXACT only if LN_ARITH=2 is bit-exact (probe-gated).  Deterministic (no atomics, no split-K), capture-safe (no host sync).
Credits: projection/epilogue/transition kernel mathematics = fpf_triatt_pro (prologue v3) and fpf_transition (v2.2), welford emulation = fpf_triatt_pro
(ln_mode='welford'), restructured loops = fpf_glue_v2 — by those kernels' authors; this file adds the in-register LayerNorm and the fusions."""
import torch, triton, triton.language as tl

try:
    from triton.language.extra import libdevice as _ld     # exp for SwiGLU identical to fpf_transition (libdevice when available)
except Exception:
    try:
        from triton.language.extra.cuda import libdevice as _ld
    except Exception:
        _ld = None
_HAS_LD = tl.constexpr(_ld is not None)

def _range_supports(kw):
    try:
        import inspect
        return kw in inspect.signature(tl.range.__init__).parameters
    except Exception:
        return False
RANGE_HAS_WS = _range_supports("warp_specialize")

# optional: fpf_triatt_pro's fast_layernorm Welford emulation helpers: used only under LN_ARITH == 2
_WELF = None
try:
    from fpf_triatt_pro import prologue as _P
    if all(hasattr(_P, n) for n in ("_welford_lane_chunk8", "_welford_butterfly32", "_mul_rn", "_add_rn", "_sub_rn", "_fma_rn", "_rcp_fast", "_rsqrt_fast")):
        _WELF = _P
except Exception:
    _WELF = None
HAS_WELFORD = _WELF is not None
if HAS_WELFORD:
    _w_chunk8, _w_bfly, _mul_rn, _add_rn, _sub_rn, _fma_rn, _rcp_fast, _rsqrt_fast = (_WELF._welford_lane_chunk8, _WELF._welford_butterfly32, _WELF._mul_rn, _WELF._add_rn,
                                                                                     _WELF._sub_rn, _WELF._fma_rn, _WELF._rcp_fast, _WELF._rsqrt_fast)
else:   # placeholders so the kernel source parses; LN_ARITH==2 is refused on the host side when HAS_WELFORD is False
    @triton.jit
    def _w_chunk8(xl, BM: tl.constexpr, LANES: tl.constexpr, FMA_M2: tl.constexpr):
        m = tl.sum(xl, axis=2) / 8
        return m, tl.sum((xl - m[:, :, None]) * (xl - m[:, :, None]), axis=2)
    @triton.jit
    def _w_bfly(mean, m2, BM: tl.constexpr, FMA_MERGE: tl.constexpr):
        return mean, m2
    @triton.jit
    def _mul_rn(a, b): return a * b
    @triton.jit
    def _add_rn(a, b): return a + b
    @triton.jit
    def _sub_rn(a, b): return a - b
    @triton.jit
    def _fma_rn(a, b, c): return a * b + c
    @triton.jit
    def _rcp_fast(a): return 1.0 / a
    @triton.jit
    def _rsqrt_fast(a): return 1.0 / tl.sqrt(a)


@triton.jit
def _ln_rows(zt, wln, bln, rmask, eps, C: tl.constexpr, BM: tl.constexpr, LN_ARITH: tl.constexpr, FMA_M2: tl.constexpr, FMA_MERGE: tl.constexpr, FMA_AFF: tl.constexpr):
    """zt bf16 [BM, C] -> LayerNorm(z)*w+b as bf16 [BM, C] (rows outside rmask -> 0).  LN_ARITH 1 = two-pass fp32 (== fpf_triatt_pro ln_mode='fused' and fpf_transition ln_mode=1
    arithmetic: mean = sum/C, var = sum((x-mean)^2)/C, inv = 1/sqrt_rn(var+eps), y = xc*inv*w + b);  2 = fast_layernorm Welford emulation (fpf_triatt_pro ln_mode='welford'), C == 256 only."""
    xf = zt.to(tl.float32)
    if LN_ARITH == 1:
        mean = tl.sum(xf, axis=1) / C
        xc = xf - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / C
        inv = 1.0 / tl.sqrt_rn(var + eps)
        y = xc * inv[:, None] * wln[None, :] + bln[None, :]
    else:
        LANES: tl.constexpr = 32
        xl = tl.reshape(xf, (BM, LANES, 8))
        t_mean, t_m2 = _w_chunk8(xl, BM, LANES, FMA_M2)
        mean_l, m2_l = _w_bfly(t_mean, t_m2, BM, FMA_MERGE)
        var_l = _mul_rn(_rcp_fast(tl.zeros_like(m2_l) + C), m2_l)
        var_l = tl.maximum(var_l, 0.0)
        inv_l = _rsqrt_fast(_add_rn(var_l, tl.zeros_like(var_l) + eps))
        mean_e = tl.reshape(tl.broadcast_to(mean_l[:, :, None], (BM, LANES, 8)), (BM, C))
        inv_e = tl.reshape(tl.broadcast_to(inv_l[:, :, None], (BM, LANES, 8)), (BM, C))
        nrm = _mul_rn(inv_e, _sub_rn(xf, mean_e))
        wb2 = tl.broadcast_to(wln[None, :], (BM, C))
        bb2 = tl.broadcast_to(bln[None, :], (BM, C))
        if FMA_AFF:
            y = _fma_rn(nrm, wb2, bb2)
        else:
            y = _add_rn(_mul_rn(nrm, wb2), bb2)
    return tl.where(rmask[:, None], y, 0.0).to(tl.bfloat16)


# =============================================================================================== F1: prologue with in-register LN, native-layout z read
@triton.jit
def _f1_qkv_chunk(x, WQKVG, Q, K, V, cidx, rn, rc, dd64, rowpart, rmask, NJP, C: tl.constexpr, D: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr):
    n0 = cidx * BN
    wT = tl.load(WQKVG + (n0 + rn)[None, :].to(tl.int64) * C + rc[:, None])             # [C, BN] = W[n0:n0+BN, :]^T
    acc = tl.dot(x, wT)                                                                  # fp32 acc over the FULL K=C in one chain (== tested prologue)
    o16 = acc.to(tl.bfloat16)
    which = n0 // HD
    col0 = n0 - which * HD
    hh64 = ((col0 + rn) // D).to(tl.int64)
    off = rowpart + hh64[None, :] * NJP * D + dd64[None, :]
    if which == 0:
        tl.store(Q + off, o16, mask=rmask[:, None])
    elif which == 1:
        tl.store(K + off, o16, mask=rmask[:, None])
    else:
        tl.store(V + off, o16, mask=rmask[:, None])


@triton.jit
def _f1_prologue_ln_kernel(Z, LNW, LNB, WQKVG, WB, Q, K, V, G, BIAS, XOUT,
                           NI, NJ, s_zi, s_zj, eps,
                           NJP, NIP,                          # output pitches (unpadded: NJ, NI; padded: P, P)
                           C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HB: tl.constexpr,
                           BI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr,
                           LN_ARITH: tl.constexpr, FMA_M2: tl.constexpr, FMA_MERGE: tl.constexpr, FMA_AFF: tl.constexpr,
                           WRITE_X: tl.constexpr, GROUP_I: tl.constexpr,
                           NUM_STAGES_W: tl.constexpr, WS: tl.constexpr):
    HD: tl.constexpr = H * D
    BM: tl.constexpr = BI * BJ
    NQKV: tl.constexpr = (3 * HD) // BN
    NG: tl.constexpr = HD // BN
    # ---- tile id -> (pid_i, pid_j); GROUP_I > 0 = grouped ordering (bands of GROUP_I i-tiles, j fastest inside) so the z rows a band touches stay L2-resident for the transposed read
    if GROUP_I > 0:
        pid = tl.program_id(0)
        n_i = tl.cdiv(NI, BI)
        n_j = tl.cdiv(NJ, BJ)
        width = GROUP_I * n_j
        gid = pid // width
        first_i = gid * GROUP_I
        gsz = tl.minimum(n_i - first_i, GROUP_I)
        pid_i = first_i + (pid % width) % gsz
        pid_j = (pid % width) // gsz
    else:
        pid_i = tl.program_id(0)
        pid_j = tl.program_id(1)
    r = tl.arange(0, BM)
    ii = pid_i * BI + r // BJ
    jj = pid_j * BJ + r % BJ
    rmask = (ii < NI) & (jj < NJ)
    rc = tl.arange(0, C)
    ii64 = ii.to(tl.int64)
    jj64 = jj.to(tl.int64)
    # ---- read z rows (x-frame row (i,j) = z[i,j,:] (start) or z[j,i,:] (ending): s_zi/s_zj carry the frame) and LayerNorm in registers
    zoff = ii64 * s_zi + jj64 * s_zj
    zt = tl.load(Z + zoff[:, None] + rc[None, :], mask=rmask[:, None], other=0.0)
    wln = tl.load(LNW + rc).to(tl.float32)
    bln = tl.load(LNB + rc).to(tl.float32)
    x = _ln_rows(zt, wln, bln, rmask, eps, C, BM, LN_ARITH, FMA_M2, FMA_MERGE, FMA_AFF)
    if WRITE_X:
        tl.store(XOUT + ((ii64 * NJ + jj64) * C)[:, None] + rc[None, :], x, mask=rmask[:, None])
    # ---- projections (identical structure/rounding to glue prologue_v4)
    rn = tl.arange(0, BN)
    dd64 = (rn % D).to(tl.int64)
    NJP64 = NJP.to(tl.int64)
    rowpart = (ii64[:, None] * H) * NJP64 * D + jj64[:, None] * D
    if WS:
        for cidx in tl.range(0, NQKV, 1, num_stages=NUM_STAGES_W, warp_specialize=True):
            _f1_qkv_chunk(x, WQKVG, Q, K, V, cidx, rn, rc, dd64, rowpart, rmask, NJP64, C, D, HD, BN)
    else:
        for cidx in tl.range(0, NQKV, 1, num_stages=NUM_STAGES_W):
            _f1_qkv_chunk(x, WQKVG, Q, K, V, cidx, rn, rc, dd64, rowpart, rmask, NJP64, C, D, HD, BN)
    goff = (ii64 * NJP64 + jj64) * HD
    for cg in tl.range(0, NG, 1, num_stages=NUM_STAGES_W):
        n0g = 3 * HD + cg * BN
        wTg = tl.load(WQKVG + (n0g + rn)[None, :].to(tl.int64) * C + rc[:, None])
        accg = tl.dot(x, wTg)
        tl.store(G + goff[:, None] + (cg * BN + rn)[None, :], accg.to(tl.bfloat16), mask=rmask[:, None])
    rb = tl.arange(0, HB)
    wbT = tl.load(WB + rb[None, :].to(tl.int64) * C + rc[:, None])
    accb = tl.dot(x, wbT)
    b32 = accb.to(tl.bfloat16).to(tl.float32)
    boff = (rb[None, :].to(tl.int64) * NIP + ii64[:, None]) * NJP64 + jj64[:, None]
    tl.store(BIAS + boff, b32, mask=rmask[:, None] & (rb[None, :] < H))


_LN_ARITH = {"fused": 1, "twopass": 1, "welford": 2}

def prologue_ln(module, z, cch, cfg, ending: bool, ln_arith="fused", fma_flags=(True, True, True), out: dict = None, write_x=False):
    """F1. z: [N0, N1, C] bf16 with stride(-1)==1 (the block's z, native layout; NOT transposed/copied).  ending=False: x = z; ending=True: x = z^T (address math only).
    cch = fpf_triatt_pro.prologue.get_cache(module, dev) (weights: lnw/lnb/wqkvg/wb/eps).  cfg: dict(BI, BJ, num_warps, num_stages, BN, NUM_STAGES_W, WS, GROUP_I).
    out=None -> fresh unpadded outputs (q,k,v [I,H,J,D] bf16; g [I,J,HD] bf16; bias [H,I,J] fp32) == fpf_triatt_pro.triatt_prologue layouts;
    out=dict(q,k,v [P,H,P,D], g [P,P,HD], bias [1,H,P,P]) -> PADDED store (valid region only) == triatt_prologue_padded layouts; returns (q,k,v,g[:I,:J],bias)."""
    assert z.dim() == 3 and z.dtype == torch.bfloat16 and z.stride(-1) == 1, (tuple(z.shape), z.dtype, z.stride())
    C, H, D, HB = cch["C"], cch["H"], cch["D"], cch["HB"]
    assert z.shape[-1] == C
    if ending: NI, NJ, s_zi, s_zj = int(z.shape[1]), int(z.shape[0]), z.stride(1), z.stride(0)
    else:      NI, NJ, s_zi, s_zj = int(z.shape[0]), int(z.shape[1]), z.stride(0), z.stride(1)
    HD = H * D
    BN = int(cfg.get("BN", 64)); assert HD % BN == 0 and (3 * HD) % BN == 0 and BN % D == 0, (HD, BN, D)
    la = _LN_ARITH[ln_arith]
    if la == 2:
        assert HAS_WELFORD, "ln_arith='welford' needs fpf_triatt_pro v0.5 welford helpers on the path"
        assert C == 256
    dev = z.device
    if out is None:
        qkv = torch.empty((3, NI, H, NJ, D), dtype=torch.bfloat16, device=dev); q, k, v = qkv[0], qkv[1], qkv[2]
        g = torch.empty((NI, NJ, HD), dtype=torch.bfloat16, device=dev)
        bias = torch.empty((H, NI, NJ), dtype=torch.float32, device=dev)
        NJP, NIP, gret, bret = NJ, NI, g, bias
    else:
        q, k, v, bias = out["q"], out["k"], out["v"], out["bias"]
        P = int(q.shape[0])
        assert tuple(q.shape) == (P, H, P, D) and k.shape == q.shape and v.shape == q.shape and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        b4 = bias if bias.dim() == 4 else bias.unsqueeze(0)
        assert tuple(b4.shape) == (1, H, P, P) and b4.is_contiguous() and b4.dtype == torch.float32
        assert NI <= P and NJ <= P
        g = out.get("g")
        if g is None or tuple(g.shape) != (P, P, HD) or g.dtype != torch.bfloat16 or g.device != dev:
            g = torch.zeros((P, P, HD), dtype=torch.bfloat16, device=dev); out["g"] = g
        assert g.is_contiguous()
        NJP, NIP, gret, bret = P, P, g[:NI, :NJ], bias
        bias = b4
    xout = torch.empty((NI, NJ, C), dtype=torch.bfloat16, device=dev) if write_x else g
    BI, BJ = int(cfg["BI"]), int(cfg["BJ"])
    GROUP_I = int(cfg.get("GROUP_I", 0) or 0)
    n_i, n_j = triton.cdiv(NI, BI), triton.cdiv(NJ, BJ)
    grid = (n_i * n_j,) if GROUP_I > 0 else (n_i, n_j)
    kw = dict(num_warps=int(cfg["num_warps"]), num_stages=int(cfg.get("num_stages", 2)))
    if cfg.get("maxnreg"): kw["maxnreg"] = int(cfg["maxnreg"])
    _f1_prologue_ln_kernel[grid](z, cch["lnw"], cch["lnb"], cch["wqkvg"], cch["wb"], q, k, v, g, bias, xout,
                                 NI, NJ, s_zi, s_zj, float(cch["eps"]), NJP, NIP,
                                 C=C, H=H, D=D, HB=HB, BI=BI, BJ=BJ, BN=BN,
                                 LN_ARITH=la, FMA_M2=bool(fma_flags[0]), FMA_MERGE=bool(fma_flags[1]), FMA_AFF=bool(fma_flags[2]),
                                 WRITE_X=bool(write_x), GROUP_I=GROUP_I,
                                 NUM_STAGES_W=int(cfg.get("NUM_STAGES_W", 2)), WS=bool(cfg.get("WS", 0)) and RANGE_HAS_WS, **kw)
    outs = (q, k, v, gret, bret)
    return outs + (xout,) if write_x else outs


# =============================================================================================== F3: transition with in-register LN + residual from registers
@triton.jit
def _silu_mul_bf16(a16, b16):
    af = a16.to(tl.float32)
    if _HAS_LD:
        sg = 1.0 / (1.0 + _ld.exp(-af))
    else:
        sg = 1.0 / (1.0 + tl.exp(-af))
    s16 = (af * sg).to(tl.bfloat16)                       # F.silu output rounded to bf16 (stock: silu(a) is a bf16 tensor)
    return (s16.to(tl.float32) * b16.to(tl.float32)).to(tl.bfloat16)   # bf16 * bf16 -> bf16 (stock elementwise mul)


@triton.jit
def _f3_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C: tl.constexpr, NH: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, IL: tl.constexpr):
    rh = j + tl.arange(0, BH)
    if IL == 1:
        r2 = 2 * j + tl.arange(0, 2 * BH)
        wT = tl.load(WAB + r2[None, :] * s_wab + rc[:, None])                             # [C, 2BH] interleaved (a_j, b_j, ...)
        ab = tl.dot(y, wT)
        a32, b32 = tl.split(tl.reshape(ab, (BM, BH, 2)))
        a16 = a32.to(tl.bfloat16)
        b16 = b32.to(tl.bfloat16)
    else:
        waT = tl.load(WAB + rh[None, :] * s_wab + rc[:, None])
        wbT = tl.load(WAB + (rh + NH)[None, :] * s_wab + rc[:, None])
        a16 = tl.dot(y, waT).to(tl.bfloat16)
        b16 = tl.dot(y, wbT).to(tl.bfloat16)
    h = _silu_mul_bf16(a16, b16)
    woT = tl.load(WO + rc[None, :] * s_wo + rh[:, None])                                  # [BH, C]
    return tl.dot(h, woT, acc)                                                           # ascending hidden chunks, one fp32 chain (== tested kernel)


@triton.jit
def _f3_transition_ln_kernel(Z, LNW, LNB, WAB, WO, OUT, M, s_zm, s_wab, s_wo, s_om, eps,
                             C: tl.constexpr, NH: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, IL: tl.constexpr,
                             LN_ARITH: tl.constexpr, FMA_M2: tl.constexpr, FMA_MERGE: tl.constexpr, FMA_AFF: tl.constexpr,
                             NUM_STAGES_H: tl.constexpr, WS: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rc = tl.arange(0, C)
    mmask = rm < M
    rm64 = rm.to(tl.int64)
    zt = tl.load(Z + rm64[:, None] * s_zm + rc[None, :], mask=mmask[:, None], other=0.0)     # bf16 [BM, C]: the ONLY read of z (LN input AND residual)
    wln = tl.load(LNW + rc).to(tl.float32)
    bln = tl.load(LNB + rc).to(tl.float32)
    y = _ln_rows(zt, wln, bln, mmask, eps, C, BM, LN_ARITH, FMA_M2, FMA_MERGE, FMA_AFF)
    acc = tl.zeros((BM, C), dtype=tl.float32)
    if WS:
        for j in tl.range(0, NH, BH, num_stages=NUM_STAGES_H, warp_specialize=True):
            acc = _f3_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C, NH, BM, BH, IL)
    else:
        for j in tl.range(0, NH, BH, num_stages=NUM_STAGES_H):
            acc = _f3_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C, NH, BM, BH, IL)
    o16 = acc.to(tl.bfloat16)                                                            # transition(z) rounded to bf16 (stock: module output is bf16)
    outv = (zt.to(tl.float32) + o16.to(tl.float32)).to(tl.bfloat16)                     # z += update : bf16(fp32(z) + fp32(bf16(update))) == stock in-place add
    tl.store(OUT + rm64[:, None] * s_om + rc[None, :], outv, mask=mmask[:, None])


def transition_ln(z2d, cache, out2d, cfg, ln_arith="fused", fma_flags=(True, True, True)):
    """F3. z2d: [M, C] bf16 rows (unit column stride); cache = fpf_transition.transition._weights(module, dev) (wab16 / wabI16 / wo16 / lnw16 / lnb16 / eps);
    writes out2d = z + transition(LN(z)) (bf16).  out2d may alias z2d: row tiles are owned exclusively by one program, which loads its rows fully before storing -> in-place SAFE.
    cfg: BM, BH, num_warps, num_stages, IL, WS."""
    M, C = z2d.shape
    assert C == 256 and z2d.stride(1) == 1 and out2d.stride(1) == 1
    wo16 = cache["wo16"]; NH = wo16.shape[1]
    BM, BH, IL = int(cfg["BM"]), int(cfg["BH"]), int(cfg.get("IL", 1))
    assert NH % BH == 0
    wab = cache["wabI16"] if IL == 1 else cache["wab16"]
    la = _LN_ARITH[ln_arith]
    if la == 2:
        assert HAS_WELFORD and C == 256
    if M == 0:
        return out2d
    grid = (triton.cdiv(M, BM),)
    kw = dict(num_warps=int(cfg["num_warps"]), num_stages=int(cfg.get("num_stages", 2)))
    if cfg.get("maxnreg"): kw["maxnreg"] = int(cfg["maxnreg"])
    _f3_transition_ln_kernel[grid](z2d, cache["lnw16"], cache["lnb16"], wab, wo16, out2d, M, z2d.stride(0), wab.stride(0), wo16.stride(0), out2d.stride(0), float(cache["eps"]),
                                   C=C, NH=NH, BM=BM, BH=BH, IL=IL, LN_ARITH=la, FMA_M2=bool(fma_flags[0]), FMA_MERGE=bool(fma_flags[1]), FMA_AFF=bool(fma_flags[2]),
                                   NUM_STAGES_H=int(cfg.get("num_stages", 2)), WS=bool(cfg.get("WS", 0)) and RANGE_HAS_WS, **kw)
    return out2d


# =============================================================================================== F3 v2: LN'd tile handed to the GEMM loop as bf16 only; residual re-read (L2-hot)
@triton.jit
def _f3v2_transition_ln_kernel(Z, LNW, LNB, WAB, WO, OUT, M, s_zm, s_wab, s_wo, s_om, eps,
                               C: tl.constexpr, NH: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, IL: tl.constexpr,
                               LN_ARITH: tl.constexpr, FMA_M2: tl.constexpr, FMA_MERGE: tl.constexpr, FMA_AFF: tl.constexpr,
                               RES_MODE: tl.constexpr,          # 0 = keep z tile live (v1 behaviour) ; 1 = re-load z rows for the residual after the GEMM loop (L2-hot; frees registers)
                               NUM_STAGES_H: tl.constexpr, WS: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rc = tl.arange(0, C)
    mmask = rm < M
    rm64 = rm.to(tl.int64)
    zptr = Z + rm64[:, None] * s_zm + rc[None, :]
    zt = tl.load(zptr, mask=mmask[:, None], other=0.0)
    wln = tl.load(LNW + rc).to(tl.float32)
    bln = tl.load(LNB + rc).to(tl.float32)
    y = _ln_rows(zt, wln, bln, mmask, eps, C, BM, LN_ARITH, FMA_M2, FMA_MERGE, FMA_AFF)      # bf16 [BM, C]; zt/xf dead after this when RES_MODE == 1
    acc = tl.zeros((BM, C), dtype=tl.float32)
    if WS:
        for j in tl.range(0, NH, BH, num_stages=NUM_STAGES_H, warp_specialize=True):
            acc = _f3_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C, NH, BM, BH, IL)
    else:
        for j in tl.range(0, NH, BH, num_stages=NUM_STAGES_H):
            acc = _f3_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C, NH, BM, BH, IL)
    o16 = acc.to(tl.bfloat16)
    if RES_MODE == 1:
        zr = tl.load(zptr, mask=mmask[:, None], other=0.0)
    else:
        zr = zt
    outv = (zr.to(tl.float32) + o16.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + rm64[:, None] * s_om + rc[None, :], outv, mask=mmask[:, None])


def transition_ln_v2(z2d, cache, out2d, cfg, ln_arith="fused", fma_flags=(True, True, True)):
    """F3 v2 (same numerics as transition_ln; RES_MODE=1 re-reads the z rows for the residual so the z tile is not live across the GEMM loop).
    NOTE in-place (out2d aliasing z2d) is STILL safe with RES_MODE=1: a program re-reads only ITS OWN rows, before it stores them."""
    M, C = z2d.shape
    assert C == 256 and z2d.stride(1) == 1 and out2d.stride(1) == 1
    wo16 = cache["wo16"]; NH = wo16.shape[1]
    BM, BH, IL = int(cfg["BM"]), int(cfg["BH"]), int(cfg.get("IL", 1))
    assert NH % BH == 0
    wab = cache["wabI16"] if IL == 1 else cache["wab16"]
    la = _LN_ARITH[ln_arith]
    if la == 2:
        assert HAS_WELFORD and C == 256
    if M == 0:
        return out2d
    grid = (triton.cdiv(M, BM),)
    kw = dict(num_warps=int(cfg["num_warps"]), num_stages=int(cfg.get("num_stages", 2)))
    if cfg.get("maxnreg"): kw["maxnreg"] = int(cfg["maxnreg"])
    _f3v2_transition_ln_kernel[grid](z2d, cache["lnw16"], cache["lnb16"], wab, wo16, out2d, M, z2d.stride(0), wab.stride(0), wo16.stride(0), out2d.stride(0), float(cache["eps"]),
                                     C=C, NH=NH, BM=BM, BH=BH, IL=IL, LN_ARITH=la, FMA_M2=bool(fma_flags[0]), FMA_MERGE=bool(fma_flags[1]), FMA_AFF=bool(fma_flags[2]),
                                     RES_MODE=int(cfg.get("RES_MODE", 1)), NUM_STAGES_H=int(cfg.get("NUM_STAGES_H", cfg.get("num_stages", 2))), WS=bool(cfg.get("WS", 0)) and RANGE_HAS_WS, **kw)
    return out2d


# =============================================================================================== F3 v3: LN in registers -> y staged through a global scratch (L2-resident per CTA) so the
# 32-iteration dual-GEMM loop takes its A operand from SHARED memory exactly like the tested kernel (A-in-registers + fp32 acc[BM,256] spills on sm_90: v1/v2 finding).
# HBM traffic vs tested (LN kernel R z W y; kernel R y R z W out = 5 passes): R z, W y-scratch (write-back), R y (L2 hit), R z residual (L2 hit), W out = ~3 passes.
@triton.jit
def _f3v3_transition_ln_kernel(Z, LNW, LNB, WAB, WO, OUT, YS, M, s_zm, s_wab, s_wo, s_om, eps,
                               C: tl.constexpr, NH: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, IL: tl.constexpr,
                               LN_ARITH: tl.constexpr, FMA_M2: tl.constexpr, FMA_MERGE: tl.constexpr, FMA_AFF: tl.constexpr,
                               NUM_STAGES_H: tl.constexpr, WS: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rc = tl.arange(0, C)
    mmask = rm < M
    rm64 = rm.to(tl.int64)
    zptr = Z + rm64[:, None] * s_zm + rc[None, :]
    zt = tl.load(zptr, mask=mmask[:, None], other=0.0)
    wln = tl.load(LNW + rc).to(tl.float32)
    bln = tl.load(LNB + rc).to(tl.float32)
    y0 = _ln_rows(zt, wln, bln, mmask, eps, C, BM, LN_ARITH, FMA_M2, FMA_MERGE, FMA_AFF)
    yptr = YS + rm64[:, None] * C + rc[None, :]
    tl.store(yptr, y0, mask=mmask[:, None])
    tl.debug_barrier()                                                                    # CTA-wide: y rows of this tile visible to all warps before the operand load
    y = tl.load(yptr, mask=mmask[:, None], other=0.0)                                     # -> smem-resident A operand (same code path as the tested kernel's y load)
    acc = tl.zeros((BM, C), dtype=tl.float32)
    if WS:
        for j in tl.range(0, NH, BH, num_stages=NUM_STAGES_H, warp_specialize=True):
            acc = _f3_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C, NH, BM, BH, IL)
    else:
        for j in tl.range(0, NH, BH, num_stages=NUM_STAGES_H):
            acc = _f3_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C, NH, BM, BH, IL)
    o16 = acc.to(tl.bfloat16)
    zr = tl.load(zptr, mask=mmask[:, None], other=0.0)                                   # residual re-read (L2-hot: this CTA read these rows at entry)
    outv = (zr.to(tl.float32) + o16.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + rm64[:, None] * s_om + rc[None, :], outv, mask=mmask[:, None])


_SCRATCH = {}
def _scratch(M, C, device):
    """per-device grow-only bf16 scratch for y (capture-safe after warm-up: no allocation when large enough)."""
    key = (device.type, device.index)
    t = _SCRATCH.get(key)
    if t is None or t.numel() < M * C:
        t = torch.empty(M * C, dtype=torch.bfloat16, device=device); _SCRATCH[key] = t
    return t[: M * C].view(M, C)


def transition_ln_v3(z2d, cache, out2d, cfg, ln_arith="fused", fma_flags=(True, True, True)):
    """F3 v3 (numerics identical to v1/v2; y staged through an L2 scratch). In-place (out2d aliasing z2d) safe: per-program row ownership, rows loaded before stored."""
    M, C = z2d.shape
    assert C == 256 and z2d.stride(1) == 1 and out2d.stride(1) == 1
    wo16 = cache["wo16"]; NH = wo16.shape[1]
    BM, BH, IL = int(cfg["BM"]), int(cfg["BH"]), int(cfg.get("IL", 1))
    assert NH % BH == 0
    wab = cache["wabI16"] if IL == 1 else cache["wab16"]
    la = _LN_ARITH[ln_arith]
    if la == 2:
        assert HAS_WELFORD and C == 256
    if M == 0:
        return out2d
    ys = _scratch(M, C, z2d.device)
    grid = (triton.cdiv(M, BM),)
    kw = dict(num_warps=int(cfg["num_warps"]), num_stages=int(cfg.get("num_stages", 2)))
    if cfg.get("maxnreg"): kw["maxnreg"] = int(cfg["maxnreg"])
    _f3v3_transition_ln_kernel[grid](z2d, cache["lnw16"], cache["lnb16"], wab, wo16, out2d, ys, M, z2d.stride(0), wab.stride(0), wo16.stride(0), out2d.stride(0), float(cache["eps"]),
                                     C=C, NH=NH, BM=BM, BH=BH, IL=IL, LN_ARITH=la, FMA_M2=bool(fma_flags[0]), FMA_MERGE=bool(fma_flags[1]), FMA_AFF=bool(fma_flags[2]),
                                     NUM_STAGES_H=int(cfg.get("NUM_STAGES_H", cfg.get("num_stages", 2))), WS=bool(cfg.get("WS", 0)) and RANGE_HAS_WS, **kw)
    return out2d
