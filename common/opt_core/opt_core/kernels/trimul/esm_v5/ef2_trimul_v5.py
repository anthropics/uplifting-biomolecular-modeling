"""ef2_trimul_v5 — TriangleMultiplication fast-tier kernels for ESMFold2 on sm_90 (H100), track k-trimul.

Structure = FlashPairformer fpf_trimul_v4 (t9; credited — K1/K3 below are derived from its `_k1t` / `_k3c`, same math and, with every
v5 lever off, the SAME floating-point operations: planes and outputs are then bitwise identical to t9's):

    K1  LN_in(z) + [a|b] projections + sigmoid gates + mask  -> zero-padded channel-major bf16 planes ab[2, D, Np, Np]
    B   cuBLAS strided-batched bmm over the D channels (bf16 in/out, fp32 accumulate)  -> x[D, Np, Np]
    K3  LN_out(x^T) + out-projection + recomputed LN_in(z) output gate + residual        -> pair + TriMul(pair)   (token-major bf16)

v5 levers (each independently switchable through install(...) / LEVERS; the integration wave decides the default composition):

    incnt    [engineering]  incoming direction: K1 runs on pair-COLUMN token tiles and writes the TRANSPOSED planes a'_d = a_d^T,
                            b'_d = b_d^T, so the contraction x_d = a_d^T b_d runs as a'_d b'_d^T — the same NT strided-batched GEMM the
                            outgoing direction uses.  cuBLAS serves the TN form ('a^T b' on channel-major planes) with a kernel that is up
                            to 46 % slower at some extents (Np = 1408: 2.80 vs 1.92 ms on H100); with incnt both directions cost the same.
                            Numerics: fp32-accumulate bf16 GEMM either way; the k-summation order is the library's (not bitwise vs t9).
    formtab  [engineering]  per padded extent Np, run the contraction in the GEMM form (NT 'a b^T' or TN 'a^T b') that cuBLAS serves faster
                            on this device class (FORM_TN_SM90; NT everywhere else): cuBLAS's heuristic picks visibly different kernels for
                            the two forms at some extents (e.g. Np = 1600: TN 3.54 vs NT 4.00 ms; Np = 1408: NT 1.92 vs TN 2.75 ms).  K1 writes
                            the planes in the orientation the chosen form needs (row tiles -> a, b; column tiles -> a^T, b^T).  Requires incnt
                            for the incoming direction's NT form; numerics as incnt.
    stagger  [engineering]  K1/K3 stream their weight blocks starting at a per-CTA offset ((block + tile ids) mod n_blocks) instead of all
                            CTAs reading block 0 first: spreads the L2 slices serving the broadcast weight stream.  Every block's arithmetic
                            is unchanged -> bitwise identical to stagger off.
    sigmoid  [numerics]     gates use sigmoid(x) = 0.5 * tanh.approx.f32(0.5 x) + 0.5 (1 MUFU op) instead of 1 / (1 + 2^(-x log2 e))
                            (ex2 + rcp = 2 MUFU ops): both gate epilogues are MUFU-bound on H100 (537 M + 268 M gate values per call at
                            N = 1024).  Measured gate-level error vs float64 sigmoid: max 4.0e-6 absolute, mean 2.3e-7 (x in [-30, 30]); block output
                            differs from the exact-gate kernel by <= 1 bf16 ulp (tables in the k_trimul report).  Tier-2 only.
    k3cute   [engineering]  K3 (LN_out + out-projection + gate + residual) runs as the CUDA C++/CuTe kernel of ef2_trimul_v6 (warp-specialized
                            TMA/wgmma kernel for sm_90a, shipped with the kit as an audited cubin under driver/prebuilt and loaded through
                            ef2_nvjit; see that module).  install() loads the kernel eagerly and runs its install canary, raising by name if it
                            cannot, so a mode naming k3cute refuses instead of degrading.  One object ships (FASTSIG=1, LNFOLD=1): k3cute
                            engages together with lnfold and sigmoid or not at all.
    lnfold   [numerics]     (k3cute only) LayerNorm affine folded into the projections on the host: W' = bf16(W * gamma), c = W @ beta (fp32)
                            added in the epilogue; the kernel only normalises.  One bf16 rounding of W*gamma replaces t9's bf16 rounding of the
                            LN output's affine: vs the fp32 module rel-rms 5.0e-3 (t9 4.5e-3); this is what makes the CuTe K3 1.7x faster.

Nothing here is read from the environment; nothing falls back silently: `trimul_v5_forward` raises if a kernel cannot launch, and
install() refuses (raises) when ef2_w4's T9 route is not loaded (v5 drives T9's weight pack, cell and eligibility unchanged).

Entry points
    install(levers=None)      hook: ef2_w4's T9 forward (`_T9["mod"]`) is pointed at this module (same `trimul_v4_forward` signature);
                              returns the active lever dict.  uninstall() restores fpf_trimul_v4.
    trimul_v5_forward(z, outgoing, mask, w, cfg, eps=1e-5, residual=True, stock_round=False, out=None, levers=None)
    describe()                 lever state + kernel cfg, for the ACTIVE/LEVER lines the integration wave prints.
"""
import os

import torch
import triton
import triton.language as tl

__all__ = ["install", "uninstall", "trimul_v5_forward", "trimul_v4_forward", "describe", "stats", "LEVERS", "DEFAULT_LEVERS"]

DEFAULT_LEVERS = dict(incnt=True, formtab=True, sigmoid=True, stagger=False, k3cute=False, lnfold=True)
# k3cute: K3 = the CUDA/CuTe kernel of ef2_trimul_v6 (sm_90a; the audited cubin shipped under driver/prebuilt).  lnfold: its LN-affine folding (numerics
# lever; k3cute's weight form — the one shipped object is the folded, tanh.approx-gate build: k3cute, lnfold and sigmoid engage together).
LEVERS = dict(DEFAULT_LEVERS)

# Contraction form table for sm_90 (H100 80GB HBM3, cuBLAS 13.1.1.3 of the lock): the padded extents Np at which cuBLAS's TN strided-batched bf16
# GEMM ('a_d^T b_d' on channel-major [D, Np, Np] planes) beats its NT one ('a_d b_d^T') by >= 4 % (min of 2 rounds x 15-rep medians),
# measured on H100 over Np = 192..2304 step 16 (torch 2.13 / cuBLAS 13).  Every other extent runs NT.  With `incnt` the incoming
# direction can take either form too (transposed planes), so `formtab` serves both directions.  Other devices: NT (no table) — named in describe().
FORM_TN_SM90 = frozenset({432, 624, 656, 880, 1472})          # TN faster by >= 4 %: 0.875, 0.950, 0.821, 0.923, 0.958 of NT

_STATE = dict(installed=False, prev_mod=None, alloc=False)
STATS = dict(calls=0, trans_calls=0, tn_calls=0)


# =====================================================================================================================
# device helpers
# =====================================================================================================================
@triton.jit
def _sigmoid_tanh(x):
    # sigmoid(x) = 0.5 * tanh(x / 2) + 0.5 ; tanh.approx.f32: 1 MUFU op, |rel err| <= 2^-10.7
    t = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x * 0.5], dtype=tl.float32, is_pure=True, pack=1)
    return 0.5 * t + 0.5


@triton.jit
def _gate(x, FASTSIG: tl.constexpr):
    if FASTSIG:
        return _sigmoid_tanh(x)
    else:
        return tl.sigmoid(x)


# =====================================================================================================================
# K1: LN_in + gated dual projection -> channel-major zero-padded planes.   program = one token tile of pair row-block.
#   TRANS = 0: tile = BM consecutive columns j of row i (t9's tiling); planes[n, i, j]           (grid: cdiv(Np, BM) x Np)
#   TRANS = 1: tile = BM consecutive rows i of column j;              planes[n, j, i] (a^T, b^T) (grid: cdiv(Np, BM) x Np)
# =====================================================================================================================
@triton.jit
def _k1x(z_ptr, lnw_ptr, lnb_ptr, wgT_ptr, wpT_ptr, mask_ptr, ab_ptr,
         N, Np, NN, NC, plane_stride, eps,
         C: tl.constexpr, D2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr,
         HAS_MASK: tl.constexpr, TRANS: tl.constexpr, FASTSIG: tl.constexpr, STAGGER: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    cols = tl.arange(0, C)
    r = tl.arange(0, BM)
    wg_desc = tl.make_tensor_descriptor(wgT_ptr, shape=[C, D2], strides=[D2, 1], block_shape=[C, BN])
    wp_desc = tl.make_tensor_descriptor(wpT_ptr, shape=[C, D2], strides=[D2, 1], block_shape=[C, BN])
    if TRANS:
        # column tile: BM consecutive pair ROWS i of pair column j (TMA box [BM, C] on the [N, N*C] view of z); the tile is written to the
        # transposed planes ab[n, j, i] with i contiguous -> the same 2*BM-byte store runs as the row tiling.
        j = pid1
        i0 = pid0 * BM
        ri = i0 + r
        lmask = (ri < N) & (j < N)
        zT_desc = tl.make_tensor_descriptor(z_ptr, shape=[N, NC], strides=[NC, 1], block_shape=[BM, C])
        zt = zT_desc.load([i0, tl.minimum(j, N - 1) * C])
        j64 = j.to(tl.int64)
        if HAS_MASK:
            mv = tl.load(mask_ptr + ri.to(tl.int64) * N + j64, mask=lmask, other=0.0).to(tl.float32)
        obase = j64 * Np + ri.to(tl.int64)
        smask = ri < Np
    else:
        i0 = pid1
        j0 = pid0 * BM
        rj = j0 + r
        lmask = (rj < N) & (i0 < N)
        z_desc = tl.make_tensor_descriptor(z_ptr, shape=[NN, C], strides=[C, 1], block_shape=[BM, C])
        zt = z_desc.load([tl.minimum(i0, N - 1) * N + j0, 0])
        i64 = i0.to(tl.int64)
        if HAS_MASK:
            mv = tl.load(mask_ptr + i64 * N + rj.to(tl.int64), mask=lmask, other=0.0).to(tl.float32)
        obase = i64 * Np + rj.to(tl.int64)
        smask = rj < Np
    zf = zt.to(tl.float32)
    mean = tl.sum(zf, axis=1) / C
    zc = zf - mean[:, None]
    var = tl.sum(zc * zc, axis=1) / C
    rstd = tl.rsqrt(var + eps)
    lnw = tl.load(lnw_ptr + cols)
    lnb = tl.load(lnb_ptr + cols)
    x = (zc * rstd[:, None] * lnw[None, :] + lnb[None, :]).to(tl.bfloat16)
    ncols = tl.arange(0, BN)
    NB: tl.constexpr = D2 // BN
    for nb in tl.range(0, NB, num_stages=STAGES):
        if STAGGER:
            n0 = ((nb + pid0 + pid1) % NB) * BN
        else:
            n0 = nb * BN
        wg = wg_desc.load([0, n0])
        wp = wp_desc.load([0, n0])
        acc_g = tl.dot(x, wg)
        acc_p = tl.dot(x, wp)
        val = _gate(acc_g, FASTSIG) * acc_p
        if HAS_MASK:
            val = val * mv[:, None]
        val = tl.where(lmask[:, None], val, 0.0)
        nch = (n0 + ncols).to(tl.int64) * plane_stride
        tl.store(ab_ptr + obase[:, None] + nch[None, :], val.to(tl.bfloat16), mask=smask[:, None])


# =====================================================================================================================
# K3: channel-major x -> LN_out -> out-projection; LN_in(z) recomputed -> output gate; residual; token-major store.
# program = BM consecutive tokens of pair row i; acc[BN out-channels, BM tokens] = W[BN, K] @ act[K, BM] (both operands natural layouts).
# =====================================================================================================================
@triton.jit
def _k3x(x_ptr, z_ptr, lnow_ptr, lnob_ptr, lniw_ptr, lnib_ptr, wz_ptr, wg_ptr, out_ptr,
         N, Npx, x_plane_stride, eps,
         C: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         RESIDUAL: tl.constexpr, STOCK_ROUND: tl.constexpr, FASTSIG: tl.constexpr, STAGGER: tl.constexpr):
    pid_j = tl.program_id(0)
    i = tl.program_id(1).to(tl.int64)
    rows = pid_j * BM + tl.arange(0, BM)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    ch = tl.arange(0, CH)
    cz = tl.arange(0, C)
    xt = tl.load(x_ptr + (ch.to(tl.int64) * x_plane_stride)[:, None] + (i * Npx + rows64)[None, :], mask=rmask[None, :], other=0.0)   # [CH, BM]
    xf = xt.to(tl.float32)
    mean = tl.sum(xf, axis=0) / CH
    xc = xf - mean[None, :]
    var = tl.sum(xc * xc, axis=0) / CH
    rstd = tl.rsqrt(var + eps)
    low = tl.load(lnow_ptr + ch)
    lob = tl.load(lnob_ptr + ch)
    xln = (xc * rstd[None, :] * low[:, None] + lob[:, None]).to(tl.bfloat16)     # [CH, BM]
    ztok = (i * N + rows64) * C
    zt = tl.load(z_ptr + cz.to(tl.int64)[:, None] + ztok[None, :], mask=rmask[None, :], other=0.0)   # [C, BM]
    zf = zt.to(tl.float32)
    mean2 = tl.sum(zf, axis=0) / C
    zc2 = zf - mean2[None, :]
    var2 = tl.sum(zc2 * zc2, axis=0) / C
    rstd2 = tl.rsqrt(var2 + eps)
    liw = tl.load(lniw_ptr + cz)
    lib = tl.load(lnib_ptr + cz)
    zln = (zc2 * rstd2[None, :] * liw[:, None] + lib[:, None]).to(tl.bfloat16)  # [C, BM]
    nr = tl.arange(0, BN)
    NB: tl.constexpr = C // BN
    for nb in range(0, NB):
        if STAGGER:
            n0 = ((nb + pid_j + i) % NB).to(tl.int32) * BN
        else:
            n0 = nb * BN
        n64 = (n0 + nr).to(tl.int64)
        wz = tl.load(wz_ptr + n64[:, None] * CH + ch[None, :])                    # [BN, CH]   (load -> dot -> load -> dot keeps one weight
        acc_p = tl.dot(wz, xln)                                                   #  block live at a time: 96 KB smem -> 2 CTAs / SM)
        wg = tl.load(wg_ptr + n64[:, None] * C + cz[None, :])                     # [BN, C]
        acc_g = tl.dot(wg, zln)
        if STOCK_ROUND:
            o = (acc_p.to(tl.bfloat16).to(tl.float32) * _gate(acc_g, FASTSIG)).to(tl.bfloat16).to(tl.float32)
        else:
            o = _gate(acc_g, FASTSIG) * acc_p
        optrs = n64[:, None] + ztok[None, :]
        if RESIDUAL:
            o = o + tl.load(z_ptr + optrs, mask=rmask[None, :], other=0.0).to(tl.float32)
        tl.store(out_ptr + optrs, o.to(tl.bfloat16), mask=rmask[None, :])


# =====================================================================================================================
# host side
# =====================================================================================================================
def _ceil_to(n, m):
    return ((n + m - 1) // m) * m


def _ensure_allocator():
    if not _STATE["alloc"]:
        triton.set_allocator(lambda size, align, stream: torch.empty(int(size), dtype=torch.int8, device="cuda"))
        _STATE["alloc"] = True


def _resolve_cfg(cfg):
    """t9 cell -> (k1, k3) launch dicts.  Only t9's 'tma' K1 implementation has a v5 counterpart (the H100 cells are tma)."""
    k1 = dict(cfg["k1"]); k3 = dict(cfg["k3"])
    if k1.get("impl", "tma") != "tma":
        raise RuntimeError(f"ef2_trimul_v5: K1 cell impl {k1.get('impl')!r} has no v5 kernel (v5 serves the sm_90 'tma' cells)")
    return k1, k3


def launch_k1(z3, w, mask2, ab, N, Np, k1, eps, trans, levers):
    """z3 [N, N, C] bf16 contiguous; ab [2*D, Np, Np] bf16 (written fully, zero pad); mask2 [N, N] float or None."""
    _ensure_allocator()
    C = z3.shape[-1]; D2 = ab.shape[0]
    BM, BN, ST, WARPS = k1["BM"], k1["BN"], k1["num_stages"], k1["num_warps"]
    grid = (triton.cdiv(Np, BM), Np)            # (token tiles along the tile's fast axis, the other pair axis incl. pad)
    _k1x[grid](z3, w["ln_in_w"], w["ln_in_b"], w["wgT_in"], w["wpT_in"], mask2 if mask2 is not None else z3, ab,
               N, Np, N * N, N * C, Np * Np, eps,
               C=C, D2=D2, BM=BM, BN=BN, STAGES=ST, HAS_MASK=mask2 is not None, TRANS=bool(trans),
               FASTSIG=bool(levers["sigmoid"]), STAGGER=bool(levers["stagger"]), num_warps=WARPS)


def _form(Np, outgoing, lv, device):
    """GEMM form for this call.  t9: outgoing NT, incoming TN.  incnt: incoming NT too.  formtab: TN where the device table says TN is faster
    (incoming only together with incnt — without it incoming stays TN as in t9)."""
    if outgoing:
        return "TN" if (lv.get("formtab") and Np in _form_table(device)) else "NT"
    if not lv.get("incnt"):
        return "TN"
    return "TN" if (lv.get("formtab") and Np in _form_table(device)) else "NT"


def _form_table(device):
    cc = torch.cuda.get_device_capability(device)
    return FORM_TN_SM90 if cc == (9, 0) else frozenset()


def launch_bmm(ab, x, form):
    """x_d = P0_d P1_d^T (NT) | P0_d^T P1_d (TN) over the D channel planes; cuBLAS strided-batched bf16 GEMM, fp32 accumulate."""
    D = x.shape[0]
    a, b = ab[:D], ab[D:]
    if form == "NT":
        torch.bmm(a, b.transpose(1, 2), out=x)
    else:
        torch.bmm(a.transpose(1, 2), b, out=x)


def launch_k3(x, z3, w, out, N, Np, k3, residual, stock_round, eps, levers):
    if levers.get("k3cute"):
        import ef2_trimul_v6
        return ef2_trimul_v6.launch_k3(x, z3, w, out, N, Np, eps, fastsig=bool(levers["sigmoid"]), lnfold=bool(levers.get("lnfold", True)), residual=residual, stock_round=stock_round)
    C = z3.shape[-1]; CH = x.shape[0]
    BM, BN, WARPS, ST = k3["BM"], k3["BN"], k3["num_warps"], k3["num_stages"]
    grid = (triton.cdiv(N, BM), N)
    _k3x[grid](x, z3, w["ln_out_w"], w["ln_out_b"], w["ln_in_w"], w["ln_in_b"], w["wz"], w["wg_out"], out,
               N, Np, Np * Np, eps, C=C, CH=CH, BM=BM, BN=BN, RESIDUAL=bool(residual), STOCK_ROUND=bool(stock_round),
               FASTSIG=bool(levers["sigmoid"]), STAGGER=bool(levers["stagger"]), num_warps=WARPS, num_stages=ST)


def trimul_v5_forward(z, outgoing, mask, w, cfg, eps=1e-5, residual=True, stock_round=False, out=None, levers=None, pad=16):
    """z [N, N, C] bf16 CUDA (one batch element, as ef2_w4._t9_forward calls it); mask [N, N] float/bool or None; w = ef2_w4._t9_weights pack;
    cfg = the t9 cell.  Returns out = z + TriMul(z) (residual=True).  Same signature/semantics as fpf_trimul_v4.kernels.trimul_v4_forward."""
    lv = LEVERS if levers is None else levers
    if z.dim() != 3 or z.dtype != torch.bfloat16 or not z.is_cuda:
        raise RuntimeError(f"ef2_trimul_v5: expects a CUDA bf16 [N, N, C] pair, got {tuple(z.shape)} {z.dtype} {z.device}")
    k1, k3 = _resolve_cfg(cfg)
    z3 = z.contiguous()
    N, N2, C = z3.shape
    assert N == N2, "pair must be square"
    D = int(w["wz"].shape[1])                      # latent channels per operand (== CH)
    Np = _ceil_to(N, pad)
    m2 = None
    if mask is not None:
        m2 = mask
        if m2.dtype == torch.bool:
            m2 = m2.to(torch.float32)
        m2 = m2.reshape(N, N).contiguous()
    form = _form(Np, outgoing, lv, z3.device)           # "NT": x_d = P0_d P1_d^T | "TN": x_d = P0_d^T P1_d on the planes K1 writes
    trans = (form == "NT") != bool(outgoing)             # planes transposed iff (outgoing, TN) or (incoming, NT)
    ab = torch.empty((2 * D, Np, Np), dtype=torch.bfloat16, device=z3.device)
    x = torch.empty((D, Np, Np), dtype=torch.bfloat16, device=z3.device)
    launch_k1(z3, w, m2, ab, N, Np, k1, eps, trans, lv)
    launch_bmm(ab, x, form)
    if out is None:
        out = torch.empty_like(z3)
    launch_k3(x, z3, w, out, N, Np, k3, residual, stock_round, eps, lv)
    STATS["calls"] += 1
    STATS["trans_calls"] += int(trans)
    STATS["tn_calls"] += int(form == "TN")
    return out


# name-compatible alias so this module can stand in for fpf_trimul_v4.kernels inside ef2_w4._t9_forward
trimul_v4_forward = trimul_v5_forward


def parse_levers(spec):
    """'all' | 'none' | comma/plus list of lever names (optionally name=0/1) -> lever dict over DEFAULT_LEVERS (unnamed levers off)."""
    spec = (spec or "").strip()
    if spec in ("all", "default", "1"):
        return dict(DEFAULT_LEVERS)
    lv = {k: False for k in DEFAULT_LEVERS}
    if spec in ("none", "off", "0", ""):
        return lv
    for tok in spec.replace("+", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, val = tok.partition("=")
        if name not in lv:
            raise ValueError(f"ef2_trimul_v5.parse_levers: unknown lever {name!r} (known: {sorted(lv)})")
        lv[name] = (val.strip() not in ("0", "false", "off")) if val else True
    return lv


def install(levers=None, **kw):
    """Point ef2_w4's T9 route at the v5 kernels.  `levers`: dict / keywords over {incnt, stagger, sigmoid}.  Raises if T9 is not loaded."""
    import sys
    ef2_w4 = sys.modules.get("ef2_w4")
    if ef2_w4 is None:
        import ef2_w4  # noqa: F811  (driver dir is on sys.path once ef2_server configured the kit)
    lv = dict(DEFAULT_LEVERS)
    lv.update(levers or {})
    lv.update(kw)
    unknown = set(lv) - set(DEFAULT_LEVERS)
    if unknown:
        raise ValueError(f"ef2_trimul_v5.install: unknown levers {sorted(unknown)} (known: {sorted(DEFAULT_LEVERS)})")
    if ef2_w4._T9.get("mod") is None or ef2_w4._T9.get("cfg") is None:
        if ef2_w4._t9_load() is None or ef2_w4._T9.get("cfg") is None:
            raise RuntimeError("ef2_trimul_v5.install: ef2_w4's T9 (fpf_trimul_v4) route is not loaded — v5 drives T9's pack/cell; refusing")
    _resolve_cfg(ef2_w4._T9["cfg"])
    if bool(lv.get("k3cute")) != bool(lv.get("lnfold")) or (lv.get("k3cute") and not lv.get("sigmoid")):   # one K3 object ships (FASTSIG=1, LNFOLD=1)
        raise RuntimeError(f"ef2_trimul_v5.install: k3cute={int(bool(lv.get('k3cute')))} lnfold={int(bool(lv.get('lnfold')))} sigmoid={int(bool(lv.get('sigmoid')))}: the CuTe K3 kernel "
                           "runs as ONE shipped object (LayerNorm-folded, tanh.approx gate) — k3cute, lnfold and sigmoid engage together or not at all; refusing by name")
    if lv.get("k3cute"):
        import ef2_trimul_v6                      # load the shipped cubin now (named failure -> install fails -> the mode refuses), never lazily mid-fold
        ef2_trimul_v6.build(fastsig=bool(lv["sigmoid"]), lnfold=True)
    LEVERS.clear(); LEVERS.update(lv)
    if not _STATE["installed"]:
        _STATE["prev_mod"] = ef2_w4._T9["mod"]
    ef2_w4._T9["mod"] = sys.modules[__name__]
    _STATE["installed"] = True
    return dict(LEVERS)


def uninstall():
    import sys
    ef2_w4 = sys.modules.get("ef2_w4")
    if _STATE["installed"] and ef2_w4 is not None:
        ef2_w4._T9["mod"] = _STATE["prev_mod"]
    _STATE["installed"] = False


def stats():
    """The per-process call counters (the kit's EXIT tally reads them): calls, trans_calls (planes written transposed), tn_calls (TN form)."""
    return dict(STATS)


def describe():
    dev = torch.cuda.current_device() if torch.cuda.is_available() else None
    tab = ("sm90:%d TN extents" % len(FORM_TN_SM90)) if (dev is not None and torch.cuda.get_device_capability(dev) == (9, 0)) else "none for this device (NT)"
    return dict(installed=_STATE["installed"], levers=dict(LEVERS), form_table=tab, stats=dict(STATS))
