"""ln_proj — fused LayerNorm -> projection Triton kernels for an AF3-family bf16 trunk (CUDA, inference only; pure functions, no module patching).

Three entry points, one LayerNorm prologue (fp32 statistics on the register tile, two-pass mean/variance, eps inside the sqrt, affine in
fp32, the normalised row rounded to bf16 exactly where the stock graph rounds it — the LN output tensor feeding a bf16 GEMM):

  pair_bias(z, packed)            z [B, I, J, c] -> LayerNorm(c) -> Linear(c -> H, no bias) -> bias planes [B, H, I, J] ("bhij") or [B, I, J, H]
                                  ("bijh"); the pairformer's AttentionPairBias pair-bias producer. One kernel, one pass over z: a program owns
                                  BJ consecutive j of one (b, i) row, so reads are whole 2*c-byte rows and every head plane is written in
                                  j-contiguous runs. z may be any view with unit channel stride (sliced channels, transposed i/j, expanded b).
  ln_linear(x2d, packed)          y = act(LayerNorm(x) @ W^T + b) * rowmask for [M, c] rows, nout <= 1024: the same kernel when the projection
                                  is skinny (nout <= MAX_SKINNY_NOUT: one MMA over the whole row), a row-tile x nout-tile kernel with the LN
                                  as a K-chunked prologue when it is wide.
  layernorm_rows(x2d, w, b, eps)  the prologue alone as a row LayerNorm (bf16 or fp32 in, out in the input dtype or into `out`).

Numerics: LN statistics and affine in fp32; projection = bf16 x bf16 MMA with fp32 accumulation (packed with dot_fp32=True: the fp32
normalised row and fp32 weights go through an IEEE fp32 dot instead); bias / activation / row mask applied in fp32 on the accumulator; the
output is rounded once. No atomics: results are deterministic run to run. Every index product that can pass 2^31 is formed in int64.

Inputs outside the served domain raise Unsupported(reason) (reason = one word); served_*() answer the same question without raising.
"""
from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl

__all__ = ["Unsupported", "pack_pair_bias_weights", "pair_bias", "served_pair_bias", "pack_ln_linear_weights", "ln_linear",
           "served_ln_linear", "layernorm_rows", "served_layernorm_rows", "describe", "ld_of"]

SERVED_C_PAIR = (64, 128)                  # pair_bias channel widths
ROWS_SERVED_C_PAIR = SERVED_C_PAIR + (256,)  # the xP ROWS producer's widths (opt_core.mem.rowpair.diffusion.pair_bias_rows_into / DitBias via
                                           # pack_pair_bias_weights_rows): the x1 set + 256 (the skinny kernel is width-generic: BC 256 = ln_linear's
                                           # c 256 tile); every x1 caller packs through pack_pair_bias_weights and keeps SERVED_C_PAIR, byte for byte
SERVED_C = (64, 128, 256, 384)             # ln_linear / layernorm_rows channel widths
MAX_H = 64                                 # pair_bias heads
MAX_NOUT = 1024                            # ln_linear output width
MAX_SKINNY_NOUT = 64                       # nout up to this: the one-MMA row kernel; above: the row-tile x nout-tile kernel
ACTS = {"none": 0, "sigmoid": 1, "silu": 2}
IN_DTYPES = (torch.bfloat16, torch.float32)
OUT_DTYPES = (torch.bfloat16, torch.float32)
LAYOUTS = ("bhij", "bijh")
LD_ALIGN = 8                               # "bhij": each [I, J] head plane is allocated with row length ld = J rounded up to LD_ALIGN elements


def ld_of(J):
    """Row length (elements) of the head planes pair_bias allocates for out_layout="bhij": J rounded up to a multiple of LD_ALIGN (16-byte bf16 rows)."""
    return -(-int(J) // LD_ALIGN) * LD_ALIGN


class Unsupported(ValueError):
    """The input is outside the served domain; ``reason`` is one word the caller can count (see served_*)."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__("ln_proj: unsupported input (%s)%s" % (reason, (": " + detail) if detail else ""))


# ----------------------------------------------------------------------------------------------------------------------------- kernels
@triton.jit
def _ln_normalise(x, cmask, C: tl.constexpr, EVEN_C: tl.constexpr, eps):
    """x [R, BC] fp32 with lanes >= C zero -> (x - mean) * rstd, lanes >= C zero. Two-pass statistics in fp32, biased variance, eps inside the sqrt."""
    mean = tl.sum(x, 1) / C
    if EVEN_C:
        xc = x - mean[:, None]
    else:
        xc = tl.where(cmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, 1) / C
    rstd = 1.0 / tl.sqrt_rn(var + eps)
    return xc * rstd[:, None]


@triton.jit
def _epilogue(acc, PB, RM, hs, r64, rmask, HAS_PB: tl.constexpr, ACT: tl.constexpr, HAS_RM: tl.constexpr):
    """acc [R, HP] fp32 -> (+ bias[h]) -> activation -> (* rowmask[row]), all in fp32."""
    if HAS_PB:
        acc = acc + tl.load(PB + hs)[None, :]
    if ACT == 1:
        acc = 1.0 / (1.0 + tl.exp(-acc))
    elif ACT == 2:
        acc = acc / (1.0 + tl.exp(-acc))
    if HAS_RM:
        rm = tl.load(RM + r64, mask=rmask, other=0.0).to(tl.float32)
        acc = acc * rm[:, None]
    return acc


@triton.jit
def _ln_proj_tile(t, X, OUT, RM, wt, lnw, lnb, pbv, I, J, JS, NJB, sxb, sxi, sxj, sob, soi, soj, soh, eps,
                  C: tl.constexpr, BC: tl.constexpr, EVEN_C: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, BJ: tl.constexpr,
                  HAS_LNW: tl.constexpr, HAS_LNB: tl.constexpr, HAS_PB: tl.constexpr, ACT: tl.constexpr, HAS_RM: tl.constexpr,
                  DOT_FP32: tl.constexpr, PADDED: tl.constexpr):
    """One tile t = (b, i, j-block): BJ consecutive j of one (b, i) row. Rows at X + b*sxb + i*sxi + j*sxj (unit channel stride); OUT element
    (b, i, j, h) at OUT + b*sob + i*soi + j*soj + h*soh — covers [B,H,I,ld]-strided planes, [B,I,J,H] and any preallocated 4-stride view (for
    head-major planes each tile stores H runs of BJ consecutive elements; 2-byte stores at consecutive j run at the memory system's rate here,
    an MMA-layout -> j-vectorised conversion costs more than it saves). When PADDED, columns J <= j < JS (a plane's row padding) are written as
    zeros. wt = the projection weight transposed and zero-padded, [BC, HP] registers; lnw / lnb [BC] and pbv [HP] fp32 registers (unused
    unless HAS_*); RM a per-row mask over flattened (b, i, j) rows."""
    cols = tl.arange(0, BC)
    cmask = cols < C
    hs = tl.arange(0, HP)
    jb = t % NJB
    bi = t // NJB
    i64 = (bi % I).to(tl.int64)
    b64 = (bi // I).to(tl.int64)
    js = jb * BJ + tl.arange(0, BJ)
    jmask = js < J
    j64 = js.to(tl.int64)
    xptr = X + b64 * sxb + i64 * sxi + j64[:, None] * sxj + cols[None, :]
    if EVEN_C:
        xmask = jmask[:, None]
    else:
        xmask = jmask[:, None] & cmask[None, :]
    x = tl.load(xptr, mask=xmask, other=0.0).to(tl.float32)
    y = _ln_normalise(x, cmask, C, EVEN_C, eps)
    if HAS_LNW:
        y = y * lnw[None, :]
    if HAS_LNB:
        y = y + lnb[None, :]
    if DOT_FP32:
        acc = tl.dot(y, wt, input_precision="ieee")
    else:
        acc = tl.dot(y.to(tl.bfloat16), wt)
    if HAS_PB:
        acc = acc + pbv[None, :]
    if ACT == 1:
        acc = 1.0 / (1.0 + tl.exp(-acc))
    elif ACT == 2:
        acc = acc / (1.0 + tl.exp(-acc))
    if HAS_RM:
        rm = tl.load(RM + (b64 * I + i64) * J + j64, mask=jmask, other=0.0).to(tl.float32)
        acc = acc * rm[:, None]
    if PADDED:
        acc = tl.where(jmask[:, None], acc, 0.0)
        smask = js < JS
    else:
        smask = jmask
    optr = OUT + b64 * sob + i64 * soi + j64[:, None] * soj + hs.to(tl.int64)[None, :] * soh
    if HP == H:
        tl.store(optr, acc.to(OUT.dtype.element_ty), mask=smask[:, None])
    else:
        tl.store(optr, acc.to(OUT.dtype.element_ty), mask=smask[:, None] & (hs[None, :] < H))


@triton.jit
def _ln_proj_kernel(X, OUT, LNW, LNB, WT, PB, RM, I, J, JS, NJB, NT, sxb, sxi, sxj, sob, soi, soj, soh, eps,
                    C: tl.constexpr, BC: tl.constexpr, EVEN_C: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, BJ: tl.constexpr,
                    HAS_LNW: tl.constexpr, HAS_LNB: tl.constexpr, HAS_PB: tl.constexpr, ACT: tl.constexpr, HAS_RM: tl.constexpr,
                    DOT_FP32: tl.constexpr, PADDED: tl.constexpr, PERSISTENT: tl.constexpr, STAGES: tl.constexpr):
    """NT tiles; one program per tile (PERSISTENT false: the grid is NT) or a persistent grid whose program p serves tiles p, p + nprog, ... in a
    STAGES-deep software-pipelined loop. The projection weight WT [BC, HP], the LN affine LNW / LNB [C] and the projection bias PB [HP] are
    read once per program."""
    pid = tl.program_id(0)
    cols = tl.arange(0, BC)
    cmask = cols < C
    hs = tl.arange(0, HP)
    wt = tl.load(WT + cols[:, None] * HP + hs[None, :])
    if HAS_LNW:
        lnw = tl.load(LNW + cols, mask=cmask, other=0.0).to(tl.float32)
    else:
        lnw = tl.zeros([BC], dtype=tl.float32)
    if HAS_LNB:
        lnb = tl.load(LNB + cols, mask=cmask, other=0.0).to(tl.float32)
    else:
        lnb = tl.zeros([BC], dtype=tl.float32)
    if HAS_PB:
        pbv = tl.load(PB + hs).to(tl.float32)
    else:
        pbv = tl.zeros([HP], dtype=tl.float32)
    if PERSISTENT:
        for t in tl.range(pid, NT, tl.num_programs(0), num_stages=STAGES):
            _ln_proj_tile(t, X, OUT, RM, wt, lnw, lnb, pbv, I, J, JS, NJB, sxb, sxi, sxj, sob, soi, soj, soh, eps,
                          C, BC, EVEN_C, H, HP, BJ, HAS_LNW, HAS_LNB, HAS_PB, ACT, HAS_RM, DOT_FP32, PADDED)
    else:
        _ln_proj_tile(pid, X, OUT, RM, wt, lnw, lnb, pbv, I, J, JS, NJB, sxb, sxi, sxj, sob, soi, soj, soh, eps,
                      C, BC, EVEN_C, H, HP, BJ, HAS_LNW, HAS_LNB, HAS_PB, ACT, HAS_RM, DOT_FP32, PADDED)


@triton.jit
def _ln_linear_wide_kernel(X, OUT, LNW, LNB, WT, PB, RM, M, sx, so, eps,
                           C: tl.constexpr, NOUT: tl.constexpr, NP: tl.constexpr, EVEN_N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           HAS_LNW: tl.constexpr, HAS_LNB: tl.constexpr, HAS_PB: tl.constexpr, ACT: tl.constexpr, HAS_RM: tl.constexpr,
                           DOT_FP32: tl.constexpr):
    """[M, C] rows (row stride sx) -> OUT [M, NOUT] (row stride so). Program (pid_m, pid_n) = BM rows x BN outputs. Pass 1: per-row mean and
    sum of squared deviations over BK-wide chunks merged with Chan's parallel update (each chunk two-pass on registers); pass 2: the chunks
    re-read (L1/L2 resident), normalised, affine, rounded to bf16 and accumulated into the [BM, BN] fp32 tile against WT [C, NP] chunks."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    r64 = rows.to(tl.int64)
    ks = tl.arange(0, BK)
    NKC: tl.constexpr = C // BK
    mean = tl.zeros([BM], dtype=tl.float32)
    m2 = tl.zeros([BM], dtype=tl.float32)
    for kc in tl.static_range(0, NKC):
        xk = tl.load(X + r64[:, None] * sx + (kc * BK + ks)[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        mk = tl.sum(xk, 1) / BK
        dk = xk - mk[:, None]
        m2k = tl.sum(dk * dk, 1)
        delta = mk - mean
        mean = mean + delta / (kc + 1)
        m2 = m2 + m2k + delta * delta * (BK * kc / (kc + 1))
    rstd = 1.0 / tl.sqrt_rn(m2 / C + eps)
    ns = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kc in tl.static_range(0, NKC):
        kk = kc * BK + ks
        xk = tl.load(X + r64[:, None] * sx + kk[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        yk = (xk - mean[:, None]) * rstd[:, None]
        if HAS_LNW:
            yk = yk * tl.load(LNW + kk).to(tl.float32)[None, :]
        if HAS_LNB:
            yk = yk + tl.load(LNB + kk).to(tl.float32)[None, :]
        wk = tl.load(WT + kk[:, None] * NP + ns[None, :])
        if DOT_FP32:
            acc = tl.dot(yk, wk, acc, input_precision="ieee")
        else:
            acc = tl.dot(yk.to(tl.bfloat16), wk, acc)
    acc = _epilogue(acc, PB, RM, ns, r64, rmask, HAS_PB, ACT, HAS_RM)
    optr = OUT + r64[:, None] * so + ns[None, :]
    if EVEN_N:
        tl.store(optr, acc.to(OUT.dtype.element_ty), mask=rmask[:, None])
    else:
        tl.store(optr, acc.to(OUT.dtype.element_ty), mask=rmask[:, None] & (ns[None, :] < NOUT))


@triton.jit
def _ln_rows_kernel(X, Y, LNW, LNB, M, sx, sy, eps, C: tl.constexpr, BC: tl.constexpr, EVEN_C: tl.constexpr, BR: tl.constexpr,
                    HAS_LNW: tl.constexpr, HAS_LNB: tl.constexpr):
    """[M, C] rows at X + r*sx -> Y + r*sy (Y's dtype), BR rows per program."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < M
    r64 = rows.to(tl.int64)
    cols = tl.arange(0, BC)
    cmask = cols < C
    if EVEN_C:
        x = tl.load(X + r64[:, None] * sx + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    else:
        x = tl.load(X + r64[:, None] * sx + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    y = _ln_normalise(x, cmask, C, EVEN_C, eps)
    if HAS_LNW:
        y = y * tl.load(LNW + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
    if HAS_LNB:
        y = y + tl.load(LNB + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
    if EVEN_C:
        tl.store(Y + r64[:, None] * sy + cols[None, :], y.to(Y.dtype.element_ty), mask=rmask[:, None])
    else:
        tl.store(Y + r64[:, None] * sy + cols[None, :], y.to(Y.dtype.element_ty), mask=rmask[:, None] & cmask[None, :])


# ----------------------------------------------------------------------------------------------------------------------------- tiles
# Launch settings for H100 SXM (sm_90; torch 2.10 / triton 3.6); BC = next_pow2(c).
_SKINNY_BJ = {64: 128, 128: 64, 256: 32, 512: 16}    # channel block BC -> rows per tile BJ (a 16 KB bf16 tile up to BC 128)
_SKINNY_PER_SM, _SKINNY_STAGES = 4, 3               # persistent grid: programs per SM, pipeline depth of the tile loop


def _skinny_tile(BC, HP):
    """(BJ, num_warps, programs per SM (0 = one program per tile, no loop), pipeline stages of the tile loop) for the row kernel: 2 warps
    while the register-resident W^T [BC, HP] is small (HP <= 32, BC <= 128), else 4."""
    return (_SKINNY_BJ[BC], 2 if (HP <= 32 and BC <= 128) else 4, _SKINNY_PER_SM, _SKINNY_STAGES)


_WIDE_TILE = {64: (64, 256, 64, 4, 2), 128: (64, 256, 128, 8, 3), 256: (64, 128, 128, 4, 1), 384: (64, 128, 128, 4, 1)}  # C -> (BM, BN, BK, warps, stages)
WIDE_NPAD = 256                            # the wide pack pads W^T's columns to a multiple of this (>= any BN), so nout tiles load without a column mask
_ROWS_TILE = {64: (32, 2), 128: (16, 4), 256: (8, 2), 512: (4, 2)}               # BC -> (rows per program BR, num_warps)


def _pad_hp(n):
    """Register width of the skinny kernel's output tile: nout rounded up to a power of two, at least 16 (tl.dot's minimum N)."""
    return max(16, _next_pow2(n))


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


# ----------------------------------------------------------------------------------------------------------------------------- packing
def _pack_common(ln_weight, ln_bias, w, b, eps, device, dot_fp32, nout_pad):
    w = w.detach().to(device=device, dtype=torch.float32)
    nout, C = w.shape
    BC = _next_pow2(C)
    wt32 = torch.zeros((BC, nout_pad), device=device, dtype=torch.float32)
    wt32[:C, :nout] = w.t()
    packed = {"C": int(C), "BC": int(BC), "nout": int(nout), "NP": int(nout_pad), "eps": float(eps), "dot_fp32": bool(dot_fp32),
              "device": wt32.device,             # resolved (indexed) device of the packed tensors
              # projection weight, transposed + zero padded: bf16 (the stock GEMM operand) unless dot_fp32
              "wt": wt32 if dot_fp32 else wt32.to(torch.bfloat16),
              "has_lnw": ln_weight is not None, "has_lnb": ln_bias is not None, "has_pb": b is not None}
    # LN affine and projection bias, fp32 tensors in two roundings: "raw" (the fp32 parameters: what stock applies to fp32 rows, and what the
    # dot_fp32 variant applies to every row) and "bf16r" (the parameters rounded to bf16 first: what stock's LayerNorm / Linear apply to bf16
    # rows). The launch picks by the input dtype.
    for key, p, n in (("lnw", ln_weight, C), ("lnb", ln_bias, C), ("pb", b, nout_pad)):
        if p is None:
            packed[key + "_raw"] = packed[key + "_bf16r"] = None
            continue
        p32 = torch.zeros(n, device=device, dtype=torch.float32)
        p32[: p.numel()] = p.detach().to(device=device, dtype=torch.float32).reshape(-1)
        packed[key + "_raw"] = p32
        packed[key + "_bf16r"] = p32 if dot_fp32 else p32.to(torch.bfloat16).to(torch.float32)
    return packed


def pack_pair_bias_weights(ln_weight, ln_bias, w, eps, device, dot_fp32=False):
    """Pack an AttentionPairBias pair-bias producer once: ln_weight / ln_bias [c] (either may be None), w = linear_z.weight [H, c] (no bias),
    eps = the LayerNorm's. dot_fp32=True keeps the normalised row and w in fp32 through an IEEE dot (higher precision, not the stock rounding)."""
    if w.dim() != 2:
        raise Unsupported("weight_rank", "w must be [H, c], got %s" % (tuple(w.shape),))
    H, C = w.shape
    if C not in SERVED_C_PAIR:
        raise Unsupported("width", "c=%d not in %s" % (C, SERVED_C_PAIR))
    if not (1 <= H <= MAX_H):
        raise Unsupported("heads", "H=%d outside [1, %d]" % (H, MAX_H))
    for name, p in (("ln_weight", ln_weight), ("ln_bias", ln_bias)):
        if p is not None and p.numel() != C:
            raise Unsupported("affine_shape", "%s has %d elements, c=%d" % (name, p.numel(), C))
    packed = _pack_common(ln_weight, ln_bias, w, None, eps, device, dot_fp32, _pad_hp(H))
    packed["kind"] = "pair_bias"
    return packed


def pack_pair_bias_weights_rows(ln_weight, ln_bias, w, eps, device, dot_fp32=False):
    """The xP ROWS producer's packer (opt_core.mem.rowpair.diffusion.DitBias): pack_pair_bias_weights' contract, refusals and packed layout with the
    width set ROWS_SERVED_C_PAIR = SERVED_C_PAIR + (256,) instead of SERVED_C_PAIR — c = 256 packs BC 256 for the SAME skinny kernel pair_bias
    launches (its C / BC are constexprs; _SKINNY_BJ[256] = 32 rows per tile, the tile ln_linear already runs at c 256). x1 callers keep
    pack_pair_bias_weights (SERVED_C_PAIR unchanged); nothing else consults this set."""
    if w.dim() != 2:
        raise Unsupported("weight_rank", "w must be [H, c], got %s" % (tuple(w.shape),))
    H, C = w.shape
    if C not in ROWS_SERVED_C_PAIR:
        raise Unsupported("width", "c=%d not in %s" % (C, ROWS_SERVED_C_PAIR))
    if not (1 <= H <= MAX_H):
        raise Unsupported("heads", "H=%d outside [1, %d]" % (H, MAX_H))
    for name, p in (("ln_weight", ln_weight), ("ln_bias", ln_bias)):
        if p is not None and p.numel() != C:
            raise Unsupported("affine_shape", "%s has %d elements, c=%d" % (name, p.numel(), C))
    packed = _pack_common(ln_weight, ln_bias, w, None, eps, device, dot_fp32, _pad_hp(H))
    packed["kind"] = "pair_bias"
    return packed


def pack_ln_linear_weights(ln_weight, ln_bias, w, b, eps, device, dot_fp32=False):
    """Pack a LayerNorm(c; weight/bias optional) -> Linear(c -> nout; bias optional) site once: w [nout, c], b [nout] or None."""
    if w.dim() != 2:
        raise Unsupported("weight_rank", "w must be [nout, c], got %s" % (tuple(w.shape),))
    nout, C = w.shape
    if C not in SERVED_C:
        raise Unsupported("width", "c=%d not in %s" % (C, SERVED_C))
    if not (1 <= nout <= MAX_NOUT):
        raise Unsupported("nout", "nout=%d outside [1, %d]" % (nout, MAX_NOUT))
    for name, p, n in (("ln_weight", ln_weight, C), ("ln_bias", ln_bias, C), ("bias", b, nout)):
        if p is not None and p.numel() != n:
            raise Unsupported("affine_shape", "%s has %d elements, expected %d" % (name, p.numel(), n))
    skinny = nout <= MAX_SKINNY_NOUT
    NP = _pad_hp(nout) if skinny else -(-nout // WIDE_NPAD) * WIDE_NPAD
    packed = _pack_common(ln_weight, ln_bias, w, b, eps, device, dot_fp32, NP)
    packed["kind"] = "ln_linear"
    packed["skinny"] = skinny
    return packed


def _params_for(packed, dtype):
    sfx = "_bf16r" if dtype is torch.bfloat16 else "_raw"
    return packed["lnw" + sfx], packed["lnb" + sfx], packed["pb" + sfx]


# ----------------------------------------------------------------------------------------------------------------------------- pair_bias
def _out_shape(layout, B, H, I, J):
    return (B, H, I, J) if layout == "bhij" else (B, I, J, H)


def served_pair_bias(z, packed, out_layout="bhij", out_dtype=torch.bfloat16, out=None):
    """(True, "") when pair_bias serves these arguments, else (False, reason)."""
    if packed.get("kind") != "pair_bias":
        return False, "packed_kind"
    if out_layout not in LAYOUTS:
        return False, "layout"
    if out_dtype not in OUT_DTYPES:
        return False, "out_dtype"
    if not isinstance(z, torch.Tensor) or not z.is_cuda:
        return False, "not_cuda"
    if z.device != packed["device"]:
        return False, "device"
    if z.dtype not in IN_DTYPES:
        return False, "dtype"
    if z.dim() not in (3, 4):
        return False, "rank"
    if z.shape[-1] != packed["C"]:
        return False, "width"
    if z.numel() > 0 and z.stride(-1) != 1:
        return False, "channel_stride"
    if torch.is_grad_enabled() and z.requires_grad:
        return False, "grad"
    z4 = z if z.dim() == 4 else z.unsqueeze(0)
    B, I, J, _ = z4.shape
    if B * I * (-(-J // 16)) >= 2 ** 31:
        return False, "grid"
    if out is not None:
        want = _out_shape(out_layout, B, packed["nout"], I, J)
        if not isinstance(out, torch.Tensor) or out.device != z.device:
            return False, "out_device"
        if out.dtype != out_dtype:
            return False, "out_dtype"
        if tuple(out.shape) != (want if z.dim() == 4 else want[1:]):
            return False, "out_shape"
        if _self_overlapping(out):
            return False, "out_overlap"
    return True, ""


def _self_overlapping(t):
    """True when two indices of t can address one element (a zero stride on a non-singleton dim, e.g. an expanded tensor): programs would
    race on it."""
    return t.numel() > 1 and any(st == 0 and n > 1 for st, n in zip(t.stride(), t.shape))


def pair_bias(z, packed, *, out_layout="bhij", out_dtype=torch.bfloat16, out=None):
    """z [B, I, J, c] or [I, J, c] (CUDA, bf16 or fp32, any strides with unit channel stride; never copied) -> the attention pair bias
    Linear(LayerNorm(z)) permuted head-major; 3-d z gives a 3-d result.
      out_layout "bhij": shape [B, H, I, J] returned as the [..., :J] view of a freshly allocated, 16-byte aligned [B, H, I, ld] buffer,
                         ld = ld_of(J) = J rounded up to 8 (strides (H*I*ld, I*ld, ld, 1)); the pad columns hold zeros.
      out_layout "bijh": shape [B, I, J, H] contiguous (the projection's natural layout).
    `out`, when given, is written in place and returned as is: any tensor of the result's shape IN out_layout's index order and of
    out_dtype, any non-overlapping strides (e.g. a view into a larger cache); it must not overlap z. LN affine parameters are applied as
    stock applies them to this dtype (rounded to bf16 for
    bf16 z, fp32 for fp32 z); the normalised row is rounded to bf16 before the projection unless the pack has dot_fp32.
    Raises Unsupported(reason) outside the served domain."""
    ok, reason = served_pair_bias(z, packed, out_layout, out_dtype, out)
    if not ok:
        raise Unsupported(reason)
    z4 = z if z.dim() == 4 else z.unsqueeze(0)
    B, I, J, C = z4.shape
    H = packed["nout"]
    JS = J
    if out is not None:
        out4 = out if out.dim() == 4 else out.unsqueeze(0)
    elif out_layout == "bhij":
        JS = ld_of(J)
        out4 = torch.empty((B, H, I, JS), device=z.device, dtype=out_dtype)[..., :J]
    else:
        out4 = torch.empty((B, I, J, H), device=z.device, dtype=out_dtype)
    if out4.numel() > 0:
        if out_layout == "bhij":
            sob, soh, soi, soj = out4.stride()
        else:
            sob, soi, soj, soh = out4.stride()
        lnw, lnb, _ = _params_for(packed, z.dtype)
        _launch_skinny(z4, z4.stride(0), z4.stride(1), z4.stride(2), B, I, J, JS, out4, sob, soi, soj, soh, packed, lnw, lnb, None, 0, None)
    if out is not None:
        return out
    return out4 if z.dim() == 4 else out4[0]


def _launch_skinny(x, sxb, sxi, sxj, B, I, J, JS, out, sob, soi, soj, soh, packed, lnw, lnb, pb, act, rowmask, _tile=None):
    BC = packed["BC"]
    BJ, warps, per_sm, stages = _tile or _skinny_tile(BC, packed["NP"])
    BJ = max(16, min(BJ, _next_pow2(J)))
    NJB = triton.cdiv(J, BJ)
    NT = B * I * NJB
    nprog = NT if per_sm <= 0 else min(NT, per_sm * _sm_count(x.device))
    dummy = packed["wt"]
    with _device_guard(x.device):
        _ln_proj_kernel[(nprog,)](x, out, lnw if lnw is not None else dummy, lnb if lnb is not None else dummy, packed["wt"],
                                  pb if pb is not None else dummy, rowmask if rowmask is not None else dummy,
                                  I, J, JS, NJB, NT, sxb, sxi, sxj, sob, soi, soj, soh, packed["eps"],
                                  C=packed["C"], BC=BC, EVEN_C=(BC == packed["C"]), H=packed["nout"], HP=packed["NP"], BJ=BJ,
                                  HAS_LNW=lnw is not None, HAS_LNB=lnb is not None, HAS_PB=pb is not None, ACT=act, HAS_RM=rowmask is not None,
                                  DOT_FP32=packed["dot_fp32"], PADDED=JS != J, PERSISTENT=per_sm > 0, STAGES=max(1, stages), num_warps=warps)


_SM_COUNT = {}
_NULLCTX = contextlib.nullcontext()


def _device_guard(device):
    """Triton launches on the current CUDA device: switch to the input's device only when it differs (the switch costs microseconds of host time)."""
    if device.index is None or device.index == torch.cuda.current_device():
        return _NULLCTX
    return torch.cuda.device(device.index)


def _sm_count(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


# ----------------------------------------------------------------------------------------------------------------------------- ln_linear
def served_ln_linear(x2d, packed, act="none", rowmask=None, out=None):
    """(True, "") when ln_linear serves these arguments, else (False, reason)."""
    if packed.get("kind") != "ln_linear":
        return False, "packed_kind"
    if act not in ACTS:
        return False, "act"
    if not isinstance(x2d, torch.Tensor) or not x2d.is_cuda:
        return False, "not_cuda"
    if x2d.device != packed["device"]:
        return False, "device"
    if x2d.dtype not in IN_DTYPES:
        return False, "dtype"
    if x2d.dim() != 2:
        return False, "rank"
    M, C = x2d.shape
    if C != packed["C"]:
        return False, "width"
    if x2d.numel() > 0 and x2d.stride(1) != 1:
        return False, "channel_stride"
    if torch.is_grad_enabled() and x2d.requires_grad:
        return False, "grad"
    if M >= 2 ** 31 - 2 ** 12:
        return False, "rows"
    if rowmask is not None:
        if not isinstance(rowmask, torch.Tensor) or rowmask.device != x2d.device:
            return False, "rowmask_device"
        if rowmask.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            return False, "rowmask_dtype"
        if rowmask.numel() != M or (rowmask.numel() > 1 and rowmask.reshape(-1).stride(0) != 1):
            return False, "rowmask_shape"
    if out is not None:
        if not isinstance(out, torch.Tensor) or out.device != x2d.device:
            return False, "out_device"
        if out.dtype not in OUT_DTYPES:
            return False, "out_dtype"
        if tuple(out.shape) != (M, packed["nout"]):
            return False, "out_shape"
        if out.numel() > 0 and out.stride(1) != 1 and packed["nout"] > 1 and not packed["skinny"]:
            return False, "out_stride"
        if _self_overlapping(out):
            return False, "out_overlap"
    return True, ""


def ln_linear(x2d, packed, *, act="none", rowmask=None, out=None):
    """y = act(LayerNorm(x2d) @ W^T + b) * rowmask[:, None] for x2d [M, c] (CUDA, bf16 or fp32, row-strided with unit channel stride) ->
    [M, nout] bf16 (or `out`'s dtype when `out` [M, nout] is given: bf16 or fp32). act in {"none", "sigmoid", "silu"}; rowmask [M] or None.
    The accumulator is fp32 and bias / activation / mask are applied to it before the one output rounding. Raises Unsupported(reason)."""
    ok, reason = served_ln_linear(x2d, packed, act, rowmask, out)
    if not ok:
        raise Unsupported(reason)
    M, C = x2d.shape
    nout = packed["nout"]
    if out is None:
        out = torch.empty((M, nout), device=x2d.device, dtype=torch.bfloat16)
    if M == 0:
        return out
    lnw, lnb, pb = _params_for(packed, x2d.dtype)
    rm = rowmask.reshape(-1) if rowmask is not None else None
    if packed["skinny"]:
        _launch_skinny(x2d, 0, 0, x2d.stride(0), 1, 1, M, M, out, 0, 0, out.stride(0), out.stride(1), packed, lnw, lnb, pb, ACTS[act], rm)
    else:
        _launch_wide(x2d, out, packed, lnw, lnb, pb, ACTS[act], rm)
    return out


def _launch_wide(x2d, out, packed, lnw, lnb, pb, act, rm, _tile=None):
    M, C = x2d.shape
    BM, BN, BK, warps, stages = _tile or _WIDE_TILE[C]
    NP = packed["NP"]
    nout = packed["nout"]
    assert WIDE_NPAD % BN == 0 and C % BK == 0, (BN, C, BK)
    dummy = packed["wt"]
    grid = (triton.cdiv(M, BM), triton.cdiv(nout, BN))
    with _device_guard(x2d.device):
        _ln_linear_wide_kernel[grid](x2d, out, lnw if lnw is not None else dummy, lnb if lnb is not None else dummy, packed["wt"],
                                     pb if pb is not None else dummy, rm if rm is not None else dummy, M, x2d.stride(0), out.stride(0), packed["eps"],
                                     C=C, NOUT=nout, NP=NP, EVEN_N=(nout % BN == 0), BM=BM, BN=BN, BK=BK,
                                     HAS_LNW=lnw is not None, HAS_LNB=lnb is not None, HAS_PB=pb is not None, ACT=act, HAS_RM=rm is not None,
                                     DOT_FP32=packed["dot_fp32"], num_warps=warps, num_stages=stages)


# ----------------------------------------------------------------------------------------------------------------------------- layernorm_rows
def served_layernorm_rows(x2d, weight=None, bias=None, out=None):
    if not isinstance(x2d, torch.Tensor) or not x2d.is_cuda:
        return False, "not_cuda"
    if x2d.dtype not in IN_DTYPES:
        return False, "dtype"
    if x2d.dim() != 2:
        return False, "rank"
    M, C = x2d.shape
    if C not in SERVED_C:
        return False, "width"
    if x2d.numel() > 0 and x2d.stride(1) != 1:
        return False, "channel_stride"
    if torch.is_grad_enabled() and x2d.requires_grad:
        return False, "grad"
    if M >= 2 ** 31 - 2 ** 12:
        return False, "rows"
    for name, p in (("weight", weight), ("bias", bias)):
        if p is not None and (not isinstance(p, torch.Tensor) or p.device != x2d.device or p.numel() != C):
            return False, name
    if out is not None:
        if not isinstance(out, torch.Tensor) or out.device != x2d.device:
            return False, "out_device"
        if out.dtype not in OUT_DTYPES:
            return False, "out_dtype"
        if tuple(out.shape) != (M, C) or (out.numel() > 0 and out.stride(1) != 1):
            return False, "out_shape"
        if _self_overlapping(out):
            return False, "out_overlap"
    return True, ""


def layernorm_rows(x2d, weight, bias, eps, out=None):
    """Row LayerNorm of x2d [M, c] (CUDA, bf16 or fp32, row-strided with unit channel stride; c in SERVED_C): fp32 statistics, affine in
    fp32 with `weight` / `bias` AS GIVEN (pass bf16-rounded parameters to mirror stock's bf16 path), output in x2d's dtype or into `out`
    ([M, c], bf16 or fp32; may be x2d itself — each program reads its rows before writing them). Raises Unsupported(reason)."""
    ok, reason = served_layernorm_rows(x2d, weight, bias, out)
    if not ok:
        raise Unsupported(reason)
    M, C = x2d.shape
    if out is None:
        out = torch.empty((M, C), device=x2d.device, dtype=x2d.dtype)
    if M == 0:
        return out
    _launch_rows(x2d, out, weight, bias, eps)
    return out


def _launch_rows(x2d, out, weight, bias, eps, _tile=None):
    M, C = x2d.shape
    BC = _next_pow2(C)
    BR, warps = _tile or _ROWS_TILE[BC]
    grid = (triton.cdiv(M, BR),)
    with _device_guard(x2d.device):
        _ln_rows_kernel[grid](x2d, out, weight if weight is not None else out, bias if bias is not None else out, M, x2d.stride(0), out.stride(0),
                              float(eps), C=C, BC=BC, EVEN_C=(BC == C), BR=BR, HAS_LNW=weight is not None, HAS_LNB=bias is not None, num_warps=warps)


def describe():
    """The served domain and launch settings, as data."""
    return {"pair_bias": {"c": SERVED_C_PAIR, "H": [1, MAX_H], "in_dtypes": [str(d) for d in IN_DTYPES], "out_dtypes": [str(d) for d in OUT_DTYPES],
                          "layouts": {"bhij": "[B,H,I,J] view of a [B,H,I,ld] buffer, ld = J rounded up to %d, pad columns zero" % LD_ALIGN, "bijh": "[B,I,J,H] contiguous"},
                          "z": "3-d or 4-d CUDA tensor, any strides with unit channel stride (never copied)",
                          "refusals": ["packed_kind", "layout", "out_dtype", "not_cuda", "device", "dtype", "rank", "width", "channel_stride", "grad",
                                       "grid", "out_device", "out_shape", "out_overlap"]},
            "ln_linear": {"c": SERVED_C, "nout": [1, MAX_NOUT], "skinny_nout_max": MAX_SKINNY_NOUT, "acts": list(ACTS),
                          "refusals": ["packed_kind", "act", "not_cuda", "device", "dtype", "rank", "width", "channel_stride", "grad", "rows",
                                       "rowmask_device", "rowmask_dtype", "rowmask_shape", "out_device", "out_dtype", "out_shape", "out_stride", "out_overlap"]},
            "layernorm_rows": {"c": SERVED_C, "refusals": ["not_cuda", "dtype", "rank", "width", "channel_stride", "grad", "rows", "weight", "bias",
                                                             "out_device", "out_dtype", "out_shape", "out_overlap"]},
            "tiles": {"skinny": {"BJ_by_BC": _SKINNY_BJ, "programs_per_sm": _SKINNY_PER_SM, "stages": _SKINNY_STAGES, "num_warps": "2 if HP<=32 and BC<=128 else 4"},
                      "wide": _WIDE_TILE, "rows": _ROWS_TILE},
            "numerics": "LN stats fp32 two-pass (wide kernel: chunked, Chan-merged), eps inside sqrt, affine fp32, row -> bf16 before the bf16 MMA "
                        "(fp32 IEEE dot when packed dot_fp32), fp32 accumulate, epilogue fp32, one output rounding; no atomics",
            "triton": triton.__version__, "torch": torch.__version__}
