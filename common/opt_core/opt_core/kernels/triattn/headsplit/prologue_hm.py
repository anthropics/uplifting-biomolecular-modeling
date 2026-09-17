"""fpf_triatt_headsplit.prologue_hm — the MK-PF F1 tri-attention prologue (opt_core fpf_mkpf.kernels._f1_prologue_ln_kernel: z read once in its
native layout, LayerNorm in registers, q|k|v|g projections + fp32 pair bias) with ONE change: q, k, v are stored HEAD-MAJOR ([H, I, J, D] per
tensor, i.e. each head's [I, J, D] slab contiguous) instead of [I, H, J, D].  Nothing else moves: the LayerNorm rows (`_ln_rows`) and the
projection chunks (`_f1_qkv_chunk`: one full-K fp32 MMA chain per BN-wide output chunk, bf16 rounding) are the opt_core jit helpers
THEMSELVES (imported, not copied), launched with the F1 cell's tile configuration, so every stored value is the value F1 stores — only its
address differs.  Class EXACT-BY-CONSTRUCTION; proven torch.equal(q_hm.permute(1,0,2,3), q_f1) (k, v, g, bias likewise) on real z.

Why head-major: the exact tier's attention core (cuEquivariance triangle_attention -> cuDNN sm80 fmha on sm_90) re-reads the fp32 pair bias
[H, N, N] per row tile; above ~1024 tokens the 8-head bias (8*N^2*4 B >= 34 MB) falls out of H100's L2 and the kernel drops from ~90 to ~63
TF/s.  Calling it once per head keeps a 1-head bias slice (N^2*4 B) L2-resident with every CTA's arithmetic unchanged (bitwise-equal
outputs), and cuequivariance's public wrapper takes q/k/v/bias WITHOUT a copy only when they are contiguous — a head slice of [I,H,J,D] is
not, a head slab of [H,I,J,D] is.  (Lever `triatt_headsplit_exact`.)

opt_core is read-only here: this file imports fpf_mkpf.kernels' helpers and re-states only the ~40-line kernel frame around them.
"""
import torch, triton, triton.language as tl

from fpf_mkpf.kernels import _ln_rows, _f1_qkv_chunk, _LN_ARITH, HAS_WELFORD, RANGE_HAS_WS      # opt_core (imported, never edited)


@triton.jit
def _f1hm_prologue_ln_kernel(Z, LNW, LNB, WQKVG, WB, Q, K, V, G, BIAS,
                             NI, NJ, s_zi, s_zj, eps,
                             NJP, NIP,                        # g / bias pitches (unpadded: NJ, NI; padded: P, P) == F1
                             s_qi, HPD,                       # q/k/v: row pitch in elements (= NJP*D) and HEAD pitch / D (= rows_alloc * NJP): off = i*s_qi + h*HPD*D + j*D + d
                             C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HB: tl.constexpr,
                             BI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr,
                             LN_ARITH: tl.constexpr, FMA_M2: tl.constexpr, FMA_MERGE: tl.constexpr, FMA_AFF: tl.constexpr,
                             GROUP_I: tl.constexpr, NUM_STAGES_W: tl.constexpr, WS: tl.constexpr):
    HD: tl.constexpr = H * D
    BM: tl.constexpr = BI * BJ
    NQKV: tl.constexpr = (3 * HD) // BN
    NG: tl.constexpr = HD // BN
    # ---- tile id -> (pid_i, pid_j)  (== F1)
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
    # ---- read z rows in the x-frame and LayerNorm in registers (== F1: the opt_core helper)
    zoff = ii64 * s_zi + jj64 * s_zj
    zt = tl.load(Z + zoff[:, None] + rc[None, :], mask=rmask[:, None], other=0.0)
    wln = tl.load(LNW + rc).to(tl.float32)
    bln = tl.load(LNB + rc).to(tl.float32)
    x = _ln_rows(zt, wln, bln, rmask, eps, C, BM, LN_ARITH, FMA_M2, FMA_MERGE, FMA_AFF)
    # ---- projections (== F1: the opt_core chunk helper; ONLY the address frame differs: head-major)
    rn = tl.arange(0, BN)
    dd64 = (rn % D).to(tl.int64)
    NJP64 = NJP.to(tl.int64)
    HPD64 = HPD.to(tl.int64)
    rowpart = ii64[:, None] * s_qi + jj64[:, None] * D                                   # F1: (ii*H)*NJP*D + jj*D ; here i*s_qi + j*D, head term added by the helper as hh*HPD*D
    if WS:
        for cidx in tl.range(0, NQKV, 1, num_stages=NUM_STAGES_W, warp_specialize=True):
            _f1_qkv_chunk(x, WQKVG, Q, K, V, cidx, rn, rc, dd64, rowpart, rmask, HPD64, C, D, HD, BN)
    else:
        for cidx in tl.range(0, NQKV, 1, num_stages=NUM_STAGES_W):
            _f1_qkv_chunk(x, WQKVG, Q, K, V, cidx, rn, rc, dd64, rowpart, rmask, HPD64, C, D, HD, BN)
    # ---- gate g [I, J, HD] and fp32 bias [H, I, J]  (== F1 verbatim)
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


LAYOUT_KEY = "_qkv_layout"        # tag on a padded buffer set (ptx_trunk2_levers._padded_bufs dict): 'hm' after a head-major store, 'rm' after a row-major one


def mark_padded_layout(out: dict, layout: str):
    """The padded q/k/v buffers are allocated zero-filled and shared per (P, H, D, HD, device); the pad region (rows/cols >= N) must stay ZERO
    (Numerics S13).  A head-major store and a row-major store put the valid region at different addresses of the same storage, so the first
    use of a set in a different layout re-zeroes q/k/v once (the kit's own N-change rule re-zeroes on top of this)."""
    if out.get(LAYOUT_KEY) != layout:
        if out.get(LAYOUT_KEY) is not None or layout == "hm":
            for key in ("q", "k", "v"):
                out[key].zero_()
        out[LAYOUT_KEY] = layout


def prologue_ln_hm(module, z, cch, cfg, ending: bool, ln_arith="fused", fma_flags=(True, True, True), out: dict = None):
    """F1 with head-major q/k/v.  Same arguments as fpf_mkpf.kernels.prologue_ln (no write_x).  Returns (q, k, v, g, bias) where q, k, v are
    [I, H, J, D]-SHAPED VIEWS (permute(1,0,2,3)) of head-major [H, I, J, D] storage (unpadded: one fresh [3,H,I,J,D] allocation; padded:
    the caller's [P,H,P,D] buffers re-interpreted as [H,P,P,D]), so q[i0:i1][:, h] / q[:, h] are contiguous and the engine's row slicing,
    unsqueeze and shape asserts work unchanged.  g [I,J,HD] bf16 and bias [H,I,J] fp32 ([1,H,P,P] padded) exactly as F1."""
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
        store = torch.empty((3, H, NI, NJ, D), dtype=torch.bfloat16, device=dev)
        q_hm, k_hm, v_hm = store[0], store[1], store[2]
        g = torch.empty((NI, NJ, HD), dtype=torch.bfloat16, device=dev)
        bias = torch.empty((H, NI, NJ), dtype=torch.float32, device=dev)
        NJP, NIP, rows_alloc, gret, bret = NJ, NI, NI, g, bias
    else:
        qb, kb, vb, bias = out["q"], out["k"], out["v"], out["bias"]
        P = int(qb.shape[0])
        assert tuple(qb.shape) == (P, H, P, D) and kb.shape == qb.shape and vb.shape == qb.shape and qb.is_contiguous() and kb.is_contiguous() and vb.is_contiguous()
        b4 = bias if bias.dim() == 4 else bias.unsqueeze(0)
        assert tuple(b4.shape) == (1, H, P, P) and b4.is_contiguous() and b4.dtype == torch.float32
        assert NI <= P and NJ <= P
        mark_padded_layout(out, "hm")
        q_hm, k_hm, v_hm = qb.view(H, P, P, D), kb.view(H, P, P, D), vb.view(H, P, P, D)      # same storage, head-major interpretation
        g = out.get("g")
        if g is None or tuple(g.shape) != (P, P, HD) or g.dtype != torch.bfloat16 or g.device != dev:
            g = torch.zeros((P, P, HD), dtype=torch.bfloat16, device=dev); out["g"] = g
        assert g.is_contiguous()
        NJP, NIP, rows_alloc, gret, bret = P, P, P, g[:NI, :NJ], bias
        bias = b4
    BI, BJ = int(cfg["BI"]), int(cfg["BJ"])
    GROUP_I = int(cfg.get("GROUP_I", 0) or 0)
    n_i, n_j = triton.cdiv(NI, BI), triton.cdiv(NJ, BJ)
    grid = (n_i * n_j,) if GROUP_I > 0 else (n_i, n_j)
    kw = dict(num_warps=int(cfg["num_warps"]), num_stages=int(cfg.get("num_stages", 2)))
    if cfg.get("maxnreg"): kw["maxnreg"] = int(cfg["maxnreg"])
    _f1hm_prologue_ln_kernel[grid](z, cch["lnw"], cch["lnb"], cch["wqkvg"], cch["wb"], q_hm, k_hm, v_hm, g, bias,
                                   NI, NJ, s_zi, s_zj, float(cch["eps"]), NJP, NIP,
                                   NJP * D, rows_alloc * NJP,
                                   C=C, H=H, D=D, HB=HB, BI=BI, BJ=BJ, BN=BN,
                                   LN_ARITH=la, FMA_M2=bool(fma_flags[0]), FMA_MERGE=bool(fma_flags[1]), FMA_AFF=bool(fma_flags[2]),
                                   GROUP_I=GROUP_I, NUM_STAGES_W=int(cfg.get("NUM_STAGES_W", 2)), WS=bool(cfg.get("WS", 0)) and RANGE_HAS_WS, **kw)
    return (q_hm.permute(1, 0, 2, 3), k_hm.permute(1, 0, 2, 3), v_hm.permute(1, 0, 2, 3), gret, bret)
