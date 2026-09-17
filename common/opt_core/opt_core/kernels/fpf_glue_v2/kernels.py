"""fpf_glue_v2.kernels — restructured-loop kernels for the FPF BLK2 glue path (a cell is served only where the output is measured torch.equal to the shipped kernels on real dumps).

Design goal (diagnosis on B200 / triton 3.7.1): the shipped kernels compile to tcgen05+TMEM but run at ~26% of HBM peak because
 (a) the prologue unrolls its 9 projection chunks with tl.static_range -> no software pipelining of the W-chunk loads (0 cp.async in PTX), 255 regs + 64 spills, 1 CTA/SM;
 (b) the epilogue v2 loads the full gathered [BM, H*D] o and g tiles into registers before one K=256 dot -> 255 regs + 14..330 spills;
 (c) the transition runs N=32-wide MMAs (IL=0) -> tensor-core issue bound.
The math (operands, rounding points, K-accumulation chains) is IDENTICAL to the tested kernels; only loop structure / tiling changes:
 * prologue v4  : ln_mode='stock' only (EXACT arm input = stock fast_layernorm output). Runtime loop over the q|k|v column chunks (pipelined W loads, num_stages), q/k/v written into
                  ONE [3, I, H, J, D] buffer (views returned; each view is contiguous exactly like the separate tensors), then g chunks, then the bias dot. Per output element: one tl.dot over
                  the full K=C, fp32 acc, one rounding -> same bits as prologue v3 by construction (checked bit-exact on dumps before any cell is proposed).
 * epilogue v3  : KVER=3 = runtime loop over K chunks of KC columns (KC in {32,64,128}; gathered o/g chunk -> sigmoid-gate in registers -> tl.dot accumulate into ONE fp32 chain, K ascending)
                  == v1 (per-head, KC=D) / v2 (KC=K) accumulation order -> same bits expected (v1==v2 was tested on H100 and B200).
 * transition   : the tested kernel body with two extra constexpr knobs: WS (tl.range warp_specialize on the hidden-chunk loop, Blackwell) and FLAT (tl.range flatten); IL as shipped.
Credit: the kernel mathematics is that of fpf_triatt_pro (prologue v3), fpf_triatt_epi (epilogue v4) and fpf_transition (v2.2), by those kernels' authors; this file only re-structures loops/tiles.
"""
from __future__ import annotations
import os, math
import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice as _ld
except Exception:  # pragma: no cover
    try:
        from triton.language.extra.cuda import libdevice as _ld
    except Exception:
        _ld = None
_HAS_LD = tl.constexpr(_ld is not None)

# tl.range keyword support differs across triton versions; probe once (host side) so kernels can be launched with WS/FLAT only when accepted.
def _range_supports(kw):
    try:
        import inspect
        return kw in inspect.signature(tl.range.__init__).parameters
    except Exception:
        return False
RANGE_HAS_WS = _range_supports("warp_specialize")
RANGE_HAS_FLATTEN = _range_supports("flatten")


# =====================================================================================================================================  PROLOGUE v4 (ln_mode='stock')

@triton.jit
def _pro_qkv_chunk(x, WQKVG, QKV, cidx, rn, rc, dd64, plane, rowpart, rmask, NJP, C: tl.constexpr, D: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr):
    n0 = cidx * BN
    wT = tl.load(WQKVG + (n0 + rn)[None, :].to(tl.int64) * C + rc[:, None])             # [C, BN] = W[n0:n0+BN, :]^T
    acc = tl.dot(x, wT)                                                                  # fp32 acc over the FULL K=C in one chain (same as v3)
    o16 = acc.to(tl.bfloat16)
    which = n0 // HD
    col0 = n0 - which * HD
    hh64 = ((col0 + rn) // D).to(tl.int64)
    off = which * plane + rowpart + hh64[None, :] * NJP * D + dd64[None, :]              # q/k/v [.., H, NJP, D]: output row pitch NJP (== NJ unpadded; == P padded)
    tl.store(QKV + off, o16, mask=rmask[:, None])


@triton.jit
def _pro_v4_kernel(XLN, WQKVG, WB, QKV, G, BIAS,
                   NI, NJ,
                   NJP, NIP, plane,                  # OUTPUT pitches: q/k/v rows [.., H, NJP, D], g rows (i*NJP + j), bias [H, NIP, NJP]; plane = element distance between the
                                                     # q, k and v output tensors (one [3,I,H,J,D] buffer unpadded: I*H*J*D; padded: k.data_ptr - q.data_ptr in elements). Inputs unchanged.
                   C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HB: tl.constexpr,
                   BI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr,
                   NUM_STAGES_W: tl.constexpr, WS: tl.constexpr, GLOOP: tl.constexpr):
    HD: tl.constexpr = H * D
    BM: tl.constexpr = BI * BJ
    NQKV: tl.constexpr = (3 * HD) // BN
    NG: tl.constexpr = HD // BN
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    r = tl.arange(0, BM)
    ii = pid_i * BI + r // BJ
    jj = pid_j * BJ + r % BJ
    rmask = (ii < NI) & (jj < NJ)
    rc = tl.arange(0, C)
    ii64 = ii.to(tl.int64)
    jj64 = jj.to(tl.int64)
    xrow = (ii64 * NJ + jj64) * C
    x = tl.load(XLN + xrow[:, None] + rc[None, :], mask=rmask[:, None], other=0.0)          # bf16 [BM, C]  (stock LN output, contiguous x-frame)
    rn = tl.arange(0, BN)
    dd64 = (rn % D).to(tl.int64)
    plane = plane.to(tl.int64)
    NJP64 = NJP.to(tl.int64)
    rowpart = (ii64[:, None] * H) * NJP64 * D + jj64[:, None] * D                            # [BM,1] part of the [I,H,NJP,D] offset that does not depend on the column
    # ---- q | k | v : runtime loop (software-pipelined W chunk loads)
    if WS:
        for cidx in tl.range(0, NQKV, 1, num_stages=NUM_STAGES_W, warp_specialize=True):
            _pro_qkv_chunk(x, WQKVG, QKV, cidx, rn, rc, dd64, plane, rowpart, rmask, NJP64, C, D, HD, BN)
    else:
        for cidx in tl.range(0, NQKV, 1, num_stages=NUM_STAGES_W):
            _pro_qkv_chunk(x, WQKVG, QKV, cidx, rn, rc, dd64, plane, rowpart, rmask, NJP64, C, D, HD, BN)
    # ---- g : [I, J, HD] with row pitch NJP (contiguous when NJP == NJ), pre-sigmoid
    goff = (ii64 * NJP64 + jj64) * HD
    if GLOOP:
        for cg in tl.range(0, NG, 1, num_stages=NUM_STAGES_W):
            n0g = 3 * HD + cg * BN
            wTg = tl.load(WQKVG + (n0g + rn)[None, :].to(tl.int64) * C + rc[:, None])
            accg = tl.dot(x, wTg)
            tl.store(G + goff[:, None] + (cg * BN + rn)[None, :], accg.to(tl.bfloat16), mask=rmask[:, None])
    else:
        for cg in tl.static_range(0, NG):
            n0g = 3 * HD + cg * BN
            wTg = tl.load(WQKVG + (n0g + rn)[None, :].to(tl.int64) * C + rc[:, None])
            accg = tl.dot(x, wTg)
            tl.store(G + goff[:, None] + (cg * BN + rn)[None, :], accg.to(tl.bfloat16), mask=rmask[:, None])
    # ---- triangle bias: WB [HB, C] (rows >= H zero), out fp32 [H, I, J] = float(bf16(acc))
    rb = tl.arange(0, HB)
    wbT = tl.load(WB + rb[None, :].to(tl.int64) * C + rc[:, None])                          # [C, HB]
    accb = tl.dot(x, wbT)
    b32 = accb.to(tl.bfloat16).to(tl.float32)
    boff = (rb[None, :].to(tl.int64) * NIP + ii64[:, None]) * NJP64 + jj64[:, None]        # bias [H, NIP, NJP] fp32
    tl.store(BIAS + boff, b32, mask=rmask[:, None] & (rb[None, :] < H))


@triton.jit
def _pro_qkv_chunk3(x, WQKVG, Q, K, V, cidx, rn, rc, dd64, rowpart, rmask, NJP, C: tl.constexpr, D: tl.constexpr, HD: tl.constexpr, BN: tl.constexpr):
    n0 = cidx * BN
    wT = tl.load(WQKVG + (n0 + rn)[None, :].to(tl.int64) * C + rc[:, None])
    acc = tl.dot(x, wT)
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
def _pro_v4_kernel_1(XLN, WQKVG, WB, Q, K, V, G, BIAS,
                     NI, NJ, NJP, NIP,
                     C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HB: tl.constexpr,
                     BI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr,
                     NUM_STAGES_W: tl.constexpr, WS: tl.constexpr, GLOOP: tl.constexpr):
    """prologue_v4 with three independent q/k/v base pointers (padded buffers that are separate allocations). Same math; WS not used here (runtime branch on the store target)."""
    HD: tl.constexpr = H * D
    BM: tl.constexpr = BI * BJ
    NQKV: tl.constexpr = (3 * HD) // BN
    NG: tl.constexpr = HD // BN
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    r = tl.arange(0, BM)
    ii = pid_i * BI + r // BJ
    jj = pid_j * BJ + r % BJ
    rmask = (ii < NI) & (jj < NJ)
    rc = tl.arange(0, C)
    ii64 = ii.to(tl.int64)
    jj64 = jj.to(tl.int64)
    xrow = (ii64 * NJ + jj64) * C
    x = tl.load(XLN + xrow[:, None] + rc[None, :], mask=rmask[:, None], other=0.0)
    rn = tl.arange(0, BN)
    dd64 = (rn % D).to(tl.int64)
    NJP64 = NJP.to(tl.int64)
    rowpart = (ii64[:, None] * H) * NJP64 * D + jj64[:, None] * D
    for cidx in tl.range(0, NQKV, 1, num_stages=NUM_STAGES_W):
        _pro_qkv_chunk3(x, WQKVG, Q, K, V, cidx, rn, rc, dd64, rowpart, rmask, NJP64, C, D, HD, BN)
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


def prologue_v4(module, x_ln, cch, cfg):
    """x_ln: [NI, NJ, C] bf16 contiguous (stock LN output in the x-frame). cch = fpf_triatt_pro.prologue.get_cache(module, dev).
    cfg: dict(BI, BJ, num_warps, num_stages, BN=128, NUM_STAGES_W=2, WS=0, GLOOP=1, maxnreg=None). Returns (q, k, v, g, bias) with the v3 layouts."""
    C, H, D, HB = cch["C"], cch["H"], cch["D"], cch["HB"]
    NI, NJ = x_ln.shape[0], x_ln.shape[1]
    assert x_ln.is_contiguous() and x_ln.dtype == torch.bfloat16 and x_ln.shape[2] == C
    HD = H * D
    BN = int(cfg.get("BN", 128)); assert HD % BN == 0 and (3 * HD) % BN == 0 and BN % D == 0, (HD, BN, D)
    dev = x_ln.device
    qkv = torch.empty((3, NI, H, NJ, D), dtype=torch.bfloat16, device=dev)
    g = torch.empty((NI, NJ, HD), dtype=torch.bfloat16, device=dev)
    bias = torch.empty((H, NI, NJ), dtype=torch.float32, device=dev)
    BI, BJ = int(cfg["BI"]), int(cfg["BJ"])
    grid = (triton.cdiv(NI, BI), triton.cdiv(NJ, BJ))
    kw = dict(num_warps=int(cfg["num_warps"]), num_stages=int(cfg.get("num_stages", 2)))
    if cfg.get("maxnreg"): kw["maxnreg"] = int(cfg["maxnreg"])
    _pro_v4_kernel[grid](x_ln, cch["wqkvg"], cch["wb"], qkv, g, bias, NI, NJ, NJ, NI, NI * H * NJ * D,
                         C=C, H=H, D=D, HB=HB, BI=BI, BJ=BJ, BN=BN,
                         NUM_STAGES_W=int(cfg.get("NUM_STAGES_W", 2)), WS=bool(cfg.get("WS", 0)) and RANGE_HAS_WS, GLOOP=bool(cfg.get("GLOOP", 1)), **kw)
    return qkv[0], qkv[1], qkv[2], g, bias


def prologue_v4_padded(module, x_ln, cch, cfg, out: dict):
    """PADDED-LAYOUT producer == fpf_triatt_pro.prologue.triatt_prologue_padded for ln_mode='stock': same arithmetic per element as prologue_v4 (and as the
    shipped kernels) — only the OUTPUT addressing differs: q/k/v are written into the caller-owned zero-initialised buffers out['q'|'k'|'v'] [P, H, P, D] bf16 contiguous,
    g into out['g'] [P, P, H*D] (row pitch P; the [:NI, :NJ] view is returned), bias into out['bias'] [1, H, P, P] fp32 (pad columns pre-filled by the caller).
    Only the valid region i < NI, j < NJ is written.  x_ln: [NI, NJ, C] bf16 contiguous (x-frame; for the ending node the caller passes LN(z^T view), contiguous in that frame).
    Returns (q, k, v, g[:NI, :NJ], bias) exactly like triatt_prologue_padded."""
    C, H, D, HB = cch["C"], cch["H"], cch["D"], cch["HB"]
    NI, NJ = int(x_ln.shape[0]), int(x_ln.shape[1])
    assert x_ln.dim() == 3 and x_ln.is_contiguous() and x_ln.dtype == torch.bfloat16 and x_ln.shape[2] == C
    HD = H * D
    BN = int(cfg.get("BN", 128)); assert HD % BN == 0 and (3 * HD) % BN == 0 and BN % D == 0, (HD, BN, D)
    q, k, v, bias = out["q"], out["k"], out["v"], out["bias"]
    P = int(q.shape[0])
    assert tuple(q.shape) == (P, H, P, D) and k.shape == q.shape and v.shape == q.shape and q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), (tuple(q.shape), H, P, D)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16 and q.device == x_ln.device
    b4 = bias if bias.dim() == 4 else bias.unsqueeze(0)
    assert tuple(b4.shape) == (1, H, P, P) and b4.is_contiguous() and b4.dtype == torch.float32, (tuple(bias.shape), (1, H, P, P), bias.dtype)
    assert NI <= P and NJ <= P, (NI, NJ, P)
    g = out.get("g")
    if g is None or tuple(g.shape) != (P, P, HD) or g.dtype != torch.bfloat16 or not g.is_contiguous():
        g = torch.empty((P, P, HD), dtype=torch.bfloat16, device=x_ln.device); out["g"] = g
    esz = q.element_size()
    dk, dv = k.data_ptr() - q.data_ptr(), v.data_ptr() - q.data_ptr()
    if dk % esz == 0 and dv == 2 * dk and dk > 0:
        launches = [(q, 3, dk // esz)]                       # k and v reachable from q at a common element stride -> ONE launch writes q|k|v (the kernel's 'plane' mechanism)
    else:
        launches = None                                      # arbitrary separate allocations -> fall back: 3-plane temp is NOT allowed (caller owns the buffers) -> per-tensor launches below
    BI, BJ = int(cfg["BI"]), int(cfg["BJ"])
    grid = (triton.cdiv(NI, BI), triton.cdiv(NJ, BJ))
    kw = dict(num_warps=int(cfg["num_warps"]), num_stages=int(cfg.get("num_stages", 2)))
    if cfg.get("maxnreg"): kw["maxnreg"] = int(cfg["maxnreg"])
    common = dict(C=C, H=H, D=D, HB=HB, BI=BI, BJ=BJ, BN=BN, NUM_STAGES_W=int(cfg.get("NUM_STAGES_W", 2)), WS=bool(cfg.get("WS", 0)) and RANGE_HAS_WS, GLOOP=bool(cfg.get("GLOOP", 1)))
    if launches is not None:
        _pro_v4_kernel[grid](x_ln, cch["wqkvg"], cch["wb"], q, g, b4, NI, NJ, P, P, launches[0][2], **common, **kw)
    else:
        # general case: q, k, v are unrelated allocations. Write q|k|v through a [3, P, H, P, D] staging view is impossible without a copy, so run the kernel once with
        # plane = 0 into a temp [3,P,H,P,D]?  No — keep it exact and allocation-free: launch 3x with QKV base = q/k/v and a column window of one tensor each (NQKV chunks of that tensor).
        _pro_v4_kernel_1[grid](x_ln, cch["wqkvg"], cch["wb"], q, k, v, g, b4, NI, NJ, P, P, **common, **kw)
    return q, k, v, g[:NI, :NJ], bias


# =====================================================================================================================================  EPILOGUE v3 (KVER=3: K-chunk loop)

@triton.jit
def _epi_k_chunk(acc, o_row, g_row, WT, kc, kk, n, rmask, so_h, swt_k, D: tl.constexpr, KC: tl.constexpr, USE_LD: tl.constexpr):
    k = kc * KC + kk                                                                     # [KC] gathered column index (k = h*D + d)
    hh = k // D; dd = k - hh * D
    o_t = tl.load(o_row + (hh * so_h + dd)[None, :], mask=rmask[:, None], other=0.0)      # [BM, KC] bf16
    g_t = tl.load(g_row + k[None, :], mask=rmask[:, None], other=0.0)                    # [BM, KC] bf16
    gf = g_t.to(tl.float32)
    if USE_LD:
        e = _ld.exp(-gf)
    else:
        e = tl.exp(-gf)
    sg = (1.0 / (1.0 + e)).to(tl.bfloat16).to(tl.float32)                               # ATen sigmoid on bf16 (opmath fp32) -> bf16
    gated = (o_t.to(tl.float32) * sg).to(tl.bfloat16)                                    # o * g -> bf16
    w = tl.load(WT + k[:, None].to(tl.int64) * swt_k + n[None, :])                       # [KC, C] rows of WoT (coalesced)
    return tl.dot(gated, w, acc)                                                         # ONE fp32 chain, K ascending (== v1/v2 order)


@triton.jit
def _epi_v3_kernel(O, G, WT, Z, OUT,
                   I, J,
                   so_b, so_i, so_h, so_j,
                   sg_b, sg_i, sg_j,
                   swt_k,
                   sz_b, sz_r, sz_c,
                   su_b, su_i, su_j,
                   H: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                   BI: tl.constexpr, BJ: tl.constexpr, KC: tl.constexpr,
                   NUM_STAGES_K: tl.constexpr, WS: tl.constexpr,
                   ENDING: tl.constexpr, RESIDUAL: tl.constexpr, USE_LD: tl.constexpr):
    BM: tl.constexpr = BI * BJ
    K: tl.constexpr = H * D
    NKC: tl.constexpr = K // KC
    pid_i = tl.program_id(0); pid_j = tl.program_id(1); pid_b = tl.program_id(2).to(tl.int64)
    r = tl.arange(0, BM)
    ii = pid_i * BI + r // BJ
    jj = pid_j * BJ + r % BJ
    rmask = (ii < I) & (jj < J)
    ii64 = ii.to(tl.int64); jj64 = jj.to(tl.int64)
    kk = tl.arange(0, KC)
    n = tl.arange(0, C)
    o_row = O + pid_b * so_b + ii64[:, None] * so_i + jj64[:, None] * so_j              # [BM,1] -> O[b, i, 0, j, 0]
    g_row = G + pid_b * sg_b + ii64[:, None] * sg_i + jj64[:, None] * sg_j              # [BM,1] -> G[b, i, j, 0]
    acc = tl.zeros((BM, C), dtype=tl.float32)
    if WS:
        for kc in tl.range(0, NKC, 1, num_stages=NUM_STAGES_K, warp_specialize=True):
            acc = _epi_k_chunk(acc, o_row, g_row, WT, kc, kk, n, rmask, so_h, swt_k, D, KC, USE_LD)
    else:
        for kc in tl.range(0, NKC, 1, num_stages=NUM_STAGES_K):
            acc = _epi_k_chunk(acc, o_row, g_row, WT, kc, kk, n, rmask, so_h, swt_k, D, KC, USE_LD)
    u16 = acc.to(tl.bfloat16)
    if RESIDUAL:
        if ENDING:
            z_ptr = Z + pid_b * sz_b + jj64[:, None] * sz_r + ii64[:, None] * sz_c + n[None, :]
        else:
            z_ptr = Z + pid_b * sz_b + ii64[:, None] * sz_r + jj64[:, None] * sz_c + n[None, :]
        zt = tl.load(z_ptr, mask=rmask[:, None], other=0.0)
        znew = (zt.to(tl.float32) + u16.to(tl.float32)).to(tl.bfloat16)
        tl.store(z_ptr, znew, mask=rmask[:, None])
    else:
        u_ptr = OUT + pid_b * su_b + ii64[:, None] * su_i + jj64[:, None] * su_j + n[None, :]
        tl.store(u_ptr, u16, mask=rmask[:, None])


def epilogue_v3(o, g, wo16, z=None, *, ending=False, residual=False, out=None, cfg=None, woT16=None):
    """Same contract as fpf_triatt_epi.epilogue.triatt_epilogue (o [.., I, H, J, D] cuEq layout only), KVER=3 kernel. cfg: dict(BI, BJ, num_warps, num_stages, KC=64, NUM_STAGES_K=2, WS=0, maxnreg=None)."""
    from fpf_triatt_epi.epilogue import _as5, _woT
    o5 = _as5(o, "ihjd")
    Bo, I, H, J, D = o5.shape
    assert o5.stride(-1) == 1
    HD = H * D
    g4 = g.reshape((-1,) + tuple(g.shape[-3:])) if g.dim() != 4 else g
    if g4.dim() == 3: g4 = g4.unsqueeze(0)
    B = g4.shape[0]
    assert Bo in (B, 1) and g4.shape == (B, I, J, HD) and g4.stride(-1) == 1
    so_b = o5.stride(0) if Bo == B else 0
    c_out, K = wo16.shape
    assert K == HD and wo16.stride(1) == 1 and wo16.dtype == torch.bfloat16
    KC = int(cfg.get("KC", 64)); assert K % KC == 0 and KC % 16 == 0
    BI, BJ = int(cfg["BI"]), int(cfg["BJ"])
    use_ld = _ld is not None
    wt = _woT(wo16, woT16)
    grid = (triton.cdiv(I, BI), triton.cdiv(J, BJ), B)
    kw = dict(num_warps=int(cfg["num_warps"]), num_stages=int(cfg.get("num_stages", 1)))
    if cfg.get("maxnreg"): kw["maxnreg"] = int(cfg["maxnreg"])
    common = dict(H=H, D=D, C=c_out, BI=BI, BJ=BJ, KC=KC, NUM_STAGES_K=int(cfg.get("NUM_STAGES_K", 2)), WS=bool(cfg.get("WS", 0)) and RANGE_HAS_WS, USE_LD=use_ld)
    if residual:
        assert z is not None and z.dtype == torch.bfloat16 and z.stride(-1) == 1
        z4 = z if z.dim() == 4 else (z.unsqueeze(0) if z.dim() == 3 else z.reshape((-1,) + tuple(z.shape[-3:])))
        assert z4.shape == ((B, J, I, c_out) if ending else (B, I, J, c_out)) and z4.data_ptr() == z.data_ptr()
        _epi_v3_kernel[grid](o5, g4, wt, z4, z4, I, J, so_b, o5.stride(1), o5.stride(2), o5.stride(3), g4.stride(0), g4.stride(1), g4.stride(2), wt.stride(0),
                             z4.stride(0), z4.stride(1), z4.stride(2), 0, 0, 0, ENDING=bool(ending), RESIDUAL=True, **common, **kw)
        return None
    if out is None:
        out = torch.empty((B, I, J, c_out), dtype=torch.bfloat16, device=o5.device)
    out4 = out if out.dim() == 4 else out.view((B, I, J, c_out))
    _epi_v3_kernel[grid](o5, g4, wt, out4, out4, I, J, so_b, o5.stride(1), o5.stride(2), o5.stride(3), g4.stride(0), g4.stride(1), g4.stride(2), wt.stride(0),
                         0, 0, 0, out4.stride(0), out4.stride(1), out4.stride(2), ENDING=False, RESIDUAL=False, **common, **kw)
    return out4.reshape(tuple(g.shape[:-1]) + (c_out,))


# =====================================================================================================================================  TRANSITION (tested body + WS/FLAT knobs)
@triton.jit
def _silu_mul_bf16(a16, b16):
    af = a16.to(tl.float32)
    if _HAS_LD:
        e = _ld.exp(-af)
    else:
        e = tl.exp(-af)
    sv = (af / (1.0 + e)).to(tl.bfloat16).to(tl.float32)
    return (b16.to(tl.float32) * sv).to(tl.bfloat16)



@triton.jit
def _transition_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C: tl.constexpr, NH: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, IL: tl.constexpr):
    rh = j + tl.arange(0, BH)
    if IL == 1:
        r2 = 2 * j + tl.arange(0, 2 * BH)
        wT = tl.load(WAB + r2[None, :] * s_wab + rc[:, None])                             # [C, 2BH] columns (a_j, b_j, ...)
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
    return tl.dot(h, woT, acc)                                                           # ascending hidden chunks, one fp32 chain


@triton.jit
def _transition_v3_kernel(Y, WAB, WO, OUT, RES, M, s_ym, s_wab, s_wo, s_om, s_rm,
                          C: tl.constexpr, NH: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr,
                          IL: tl.constexpr, HAS_RES: tl.constexpr, NUM_STAGES_H: tl.constexpr, WS: tl.constexpr, FLAT: tl.constexpr):
    """c_in=256 class only (CP == C).  Y [M, C] bf16 = stock LN output; WAB [2NH, C] (IL=0: Wa rows then Wb rows; IL=1: interleaved); WO [C, NH]; OUT/RES [M, C]."""
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    mrow = rm < M
    rc = tl.arange(0, C)
    rm64 = rm.to(tl.int64)
    y = tl.load(Y + rm64[:, None] * s_ym + rc[None, :], mask=mrow[:, None], other=0.0)
    acc = tl.zeros((BM, C), dtype=tl.float32)
    if WS:
        for j in tl.range(0, NH, BH, num_stages=NUM_STAGES_H, warp_specialize=True):
            acc = _transition_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C, NH, BM, BH, IL)
    elif FLAT:
        for j in tl.range(0, NH, BH, num_stages=NUM_STAGES_H, flatten=True):
            acc = _transition_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C, NH, BM, BH, IL)
    else:
        for j in tl.range(0, NH, BH, num_stages=NUM_STAGES_H):
            acc = _transition_chunk(y, WAB, WO, acc, j, s_wab, s_wo, rc, C, NH, BM, BH, IL)
    o16 = acc.to(tl.bfloat16)
    if HAS_RES:
        rr = tl.load(RES + rm64[:, None] * s_rm + rc[None, :], mask=mrow[:, None], other=0.0)
        o16 = (rr.to(tl.float32) + o16.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + rm64[:, None] * s_om + rc[None, :], o16, mask=mrow[:, None])


def transition_v3(y2d, cache, out2d, res2d=None, cfg=None):
    """y2d [M, 256] bf16 (stock LN output), cache = fpf_transition.transition._weights(module, dev); writes out2d (= res + update if res2d given). cfg: BM,BH,num_warps,num_stages,IL,WS,FLAT,maxnreg."""
    M, C = y2d.shape
    assert C == 256 and y2d.stride(1) == 1 and out2d.stride(1) == 1
    wo16 = cache["wo16"]; NH = wo16.shape[1]
    BM, BH, IL = int(cfg["BM"]), int(cfg["BH"]), int(cfg.get("IL", 0))
    wab = cache["wabI16"] if IL == 1 else cache["wab16"]
    res = res2d if res2d is not None else out2d
    kw = dict(num_warps=int(cfg["num_warps"]), num_stages=int(cfg["num_stages"]))
    if cfg.get("maxnreg"): kw["maxnreg"] = int(cfg["maxnreg"])
    _transition_v3_kernel[(triton.cdiv(M, BM),)](y2d, wab, wo16, out2d, res, M, y2d.stride(0), wab.stride(0), wo16.stride(0), out2d.stride(0), res.stride(0),
                                                  C=C, NH=NH, BM=BM, BH=BH, IL=IL, HAS_RES=(res2d is not None), NUM_STAGES_H=int(cfg["num_stages"]),
                                                  WS=bool(cfg.get("WS", 0)) and RANGE_HAS_WS, FLAT=bool(cfg.get("FLAT", 0)) and RANGE_HAS_FLATTEN, **kw)
    return out2d
