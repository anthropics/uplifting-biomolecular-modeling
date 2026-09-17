"""exactln.triton_ln — ATen's CUDA layer_norm forward arithmetic (torch 2.12.0 `vectorized_layer_norm_kernel<float, float>`, the sequence
`exactln_fwd.cu` documents) written as Triton block functions, so a Triton cell can normalise a [BM, C] tile of fp32 rows IN ITS PROLOGUE with
the bits torch.nn.functional.layer_norm produces — the LayerNorm output never written to memory (E5b).

The per-row arithmetic is data-parallel across ATen's virtual lanes, so it maps onto tensor expressions: a row of C = 128 is the [32 lanes, 4]
tile ATen's warp 0 holds; the online Welford pass runs over the 4 elements of every lane at once (elementwise over [BM, 32]); the shuffle-down
tree (offsets 16, 8, 4, 2, 1; lane i keeps B = itself and takes A = lane i + offset) is five halving combines of the lower half (B) with the
upper half (A) of the lane axis; the shared-memory tree over the (32, 4) CTA's four warps is, for C <= 128, two combines with EMPTY partials
(executed literally: they are identities except in inf/NaN/-0.0 corners); then var = sigma2 / C (IEEE division), rstd = rsqrt(var + eps)
(approximate MUFU.RSQ, as ATen's rsqrtf), y = fma(rstd * (x - mean), gamma, beta).  Every product/sum that ATen's binary keeps separate is a
separate op and every FFMA is an explicit fma — all written as inline-PTX round-to-nearest instructions (add/sub/mul/fma.rn.f32, div.rn.f32,
rsqrt.approx.f32), so the functions keep their bits under any contraction / fast-math setting of the host kernel.  Count reciprocals are
compile-time fp32 constants (rcp_rn(n) of a known n is that same value).

Public: `ln_rows_128(x, g, b, eps)` / `ln_rows_64(x, g, b, eps)` (x: [BM, C] fp32 block, g/b: [C] fp32) -> [BM, C] fp32; `mean_rstd_128/64`.
`layer_norm_triton(x2d, weight, bias, eps, out_dtype)` is the standalone kernel used to bit-compare the functions against torch.
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

# fp32 constants exactly as ATen's arithmetic produces them (python floats holding fp32-representable values)
_F = np.float32
RCP3 = float(_F(1.0) / _F(3.0))          # rcp_rn(3.0f) == div_rn(1, 3)


# fp32 round-to-nearest primitives as inline PTX: what the functions below are made of.  Written this way the arithmetic is immune to the
# compiler's contraction setting (enable_fp_fusion) and to fast-math flags of the host kernel — a mul and an add stay a mul and an add, an fma
# is one fma — so the row functions can be dropped into any Triton cell.
@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sub(a, b):
    return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _fma(a, b, c):
    return tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _div(a, b):
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rsqrt(a):
    # ATen's rsqrtf compiled without -ftz: rsqrt.approx.f32 (MUFU.RSQ with the denormal-preserving pre/post scaling)
    return tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=f,f", [a], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _const_like(x, v: tl.constexpr):
    return tl.zeros_like(x) + v


@triton.jit
def _online4(x0, x1, x2, x3, RCP3: tl.constexpr):
    """cuWelfordOnlineSum over the 4 elements of a lane's float4, from the empty state: returns (mean, sigma2) tensors (count == 4)."""
    zero = tl.zeros_like(x0)
    # element 0: count 0 -> 1, coef = 1: delta = x0 - 0; mean = fma(delta, 1, 0); sigma2 = fma(delta, x0 - mean, 0)
    d = _sub(x0, zero)
    m = _fma(d, _const_like(x0, 1.0), zero)
    s = _fma(d, _sub(x0, m), zero)
    # element 1: coef = rcp(2) = 0.5
    d = _sub(x1, m)
    m = _fma(d, _const_like(x0, 0.5), m)
    s = _fma(d, _sub(x1, m), s)
    # element 2: coef = rcp(3)
    d = _sub(x2, m)
    m = _fma(d, _const_like(x0, RCP3), m)
    s = _fma(d, _sub(x2, m), s)
    # element 3: coef = rcp(4) = 0.25
    d = _sub(x3, m)
    m = _fma(d, _const_like(x0, 0.25), m)
    s = _fma(d, _sub(x3, m), s)
    return m, s


@triton.jit
def _combine(mb, sb, ma, sa, CB: tl.constexpr, CA: tl.constexpr):
    """cuWelfordCombine(dataB = (mb, sb, CB), dataA = (ma, sa, CA)) with compile-time counts (CA + CB > 0 here):
    delta = mb - ma; coef = rcp(count); nB = coef*CB; nA = CA*coef; mean = fma(ma, nA, nB*mb); sigma2 = fma(nB, CA*(delta*delta), sa + sb)."""
    COEF: tl.constexpr = 1.0 / (CA + CB)          # every count met here is a power of two times 1 or 3 -> see callers; exact for powers of two
    NB: tl.constexpr = COEF * CB
    NA: tl.constexpr = CA * COEF
    delta = _sub(mb, ma)
    na = _const_like(ma, NA)
    nb = _const_like(mb, NB)
    ca = _const_like(ma, CA)
    mean = _fma(ma, na, _mul(nb, mb))                # FMUL nB*B.mean, then FFMA
    dd = _mul(delta, delta)                          # FMUL
    t = _mul(ca, dd)                                 # FMUL A.count * delta^2
    sig = _fma(nb, t, _add(sa, sb))                  # FADD A.sigma2 + B.sigma2, then FFMA
    return mean, sig


@triton.jit
def _combine_empty_A(mb, sb, CB: tl.constexpr):
    """combine(B = data, A = EMPTY(0,0,0)): count = CB; delta = mb - 0; coef = rcp(CB); nB = coef*CB (= 1.0 for a power of two); nA = 0*coef = 0;
    mean = fma(0, 0, nB*mb); sigma2 = fma(nB, 0*(delta*delta), 0 + sb) — literally (NaN/inf/-0.0 behave as in ATen)."""
    COEF: tl.constexpr = 1.0 / CB
    NB: tl.constexpr = COEF * CB
    zero = tl.zeros_like(mb)
    delta = _sub(mb, zero)
    mean = _fma(zero, zero, _mul(_const_like(mb, NB), mb))
    dd = _mul(delta, delta)
    t = _mul(zero, dd)                               # A.count (= 0) * delta^2  -> 0, or NaN when delta is inf/NaN
    sig = _fma(_const_like(mb, NB), t, _add(zero, sb))   # (A.sigma2 = 0) + B.sigma2 first, then the fma
    return mean, sig


@triton.jit
def _halve(m, s, W: tl.constexpr, CNT: tl.constexpr):
    """One shuffle-down level on a [BM, W] lane tensor whose lanes all hold count CNT: lane i (< W/2) := combine(B = lane i, A = lane i + W/2).
    Returns [BM, W/2] tensors (count 2*CNT)."""
    m3 = tl.reshape(m, (m.shape[0], 2, W // 2))
    s3 = tl.reshape(s, (s.shape[0], 2, W // 2))
    m3 = tl.permute(m3, (0, 2, 1))
    s3 = tl.permute(s3, (0, 2, 1))
    mb, ma = tl.split(m3)
    sb, sa = tl.split(s3)
    return _combine(mb, sb, ma, sa, CNT, CNT)


@triton.jit
def _lanes4(x, LANES: tl.constexpr):
    """[BM, 4*LANES] -> the four [BM, LANES] element planes (x0..x3 = element j of every lane's float4)."""
    x4 = tl.reshape(x, (x.shape[0], LANES, 2, 2))
    xe, xo = tl.split(x4)            # j even (0, 2) / j odd (1, 3), each [BM, LANES, 2] indexed by j // 2
    x0, x2 = tl.split(xe)
    x1, x3 = tl.split(xo)
    return x0, x1, x2, x3


@triton.jit
def mean_rstd_128(x, eps, RCP3: tl.constexpr):
    """x: [BM, 128] fp32 -> (mean [BM], rstd [BM]) with ATen's bits (eps: fp32 scalar = float(1e-5))."""
    x0, x1, x2, x3 = _lanes4(x, 32)
    m, s = _online4(x0, x1, x2, x3, RCP3)            # [BM, 32], count 4 per lane
    m, s = _halve(m, s, 32, 4.0)                     # offset 16 -> [BM, 16], count 8
    m, s = _halve(m, s, 16, 8.0)                     # offset 8  -> [BM, 8],  count 16
    m, s = _halve(m, s, 8, 16.0)                     # offset 4  -> [BM, 4],  count 32
    m, s = _halve(m, s, 4, 32.0)                     # offset 2  -> [BM, 2],  count 64
    m, s = _halve(m, s, 2, 64.0)                     # offset 1  -> [BM, 1],  count 128
    m = tl.reshape(m, (x.shape[0],))
    s = tl.reshape(s, (x.shape[0],))
    # block tree of the (32, 4) CTA: W0 = C(W0, W2 = empty); W1 = C(empty, empty) = empty; W0 = C(W0, W1 = empty)
    m, s = _combine_empty_A(m, s, 128.0)
    m, s = _combine_empty_A(m, s, 128.0)
    var = _div(s, _const_like(s, 128.0))
    rstd = _rsqrt(_add(var, tl.zeros_like(s) + eps))
    return m, rstd


@triton.jit
def mean_rstd_64(x, eps, RCP3: tl.constexpr):
    """x: [BM, 64] fp32 -> (mean, rstd): ATen's lanes 0-15 hold the data, lanes 16-31 are empty; the offset-16 level combines each data lane
    with an EMPTY lane, then offsets 8..1 among the 16 lanes, then the two block-tree combines with empty warps."""
    x0, x1, x2, x3 = _lanes4(x, 16)
    m, s = _online4(x0, x1, x2, x3, RCP3)            # [BM, 16], count 4
    m, s = _combine_empty_A(m, s, 4.0)               # offset 16: partner lanes 16..31 are empty -> count 4
    m, s = _halve(m, s, 16, 4.0)                     # offset 8 -> count 8
    m, s = _halve(m, s, 8, 8.0)                      # offset 4 -> 16
    m, s = _halve(m, s, 4, 16.0)                     # offset 2 -> 32
    m, s = _halve(m, s, 2, 32.0)                     # offset 1 -> 64
    m = tl.reshape(m, (x.shape[0],))
    s = tl.reshape(s, (x.shape[0],))
    m, s = _combine_empty_A(m, s, 64.0)
    m, s = _combine_empty_A(m, s, 64.0)
    var = _div(s, _const_like(s, 64.0))
    rstd = _rsqrt(_add(var, tl.zeros_like(s) + eps))
    return m, rstd


@triton.jit
def affine_rows(x, mean, rstd, g, b, AFFINE: tl.constexpr):
    """y = fma(rstd * (x - mean), gamma, beta) (AFFINE=3) | (rstd*(x-mean))*gamma (1) | rstd*(x-mean) (0); x [BM, C], mean/rstd [BM], g/b [C]."""
    u = _mul(tl.broadcast_to(rstd[:, None], x.shape), _sub(x, tl.broadcast_to(mean[:, None], x.shape)))     # FADD, FMUL (rstd * (x - mean))
    if AFFINE == 3:
        y = _fma(u, tl.broadcast_to(g[None, :], x.shape), tl.broadcast_to(b[None, :], x.shape))    # FFMA (operands broadcast as they are: a -0.0 beta stays -0.0)
    elif AFFINE == 1:
        y = _mul(u, tl.broadcast_to(g[None, :], x.shape))
    else:
        y = u
    return y


@triton.jit
def ln_rows_128(x, g, b, eps, AFFINE: tl.constexpr, RCP3: tl.constexpr):
    mean, rstd = mean_rstd_128(x, eps, RCP3)
    return affine_rows(x, mean, rstd, g, b, AFFINE)


@triton.jit
def ln_rows_64(x, g, b, eps, AFFINE: tl.constexpr, RCP3: tl.constexpr):
    mean, rstd = mean_rstd_64(x, eps, RCP3)
    return affine_rows(x, mean, rstd, g, b, AFFINE)


# ----------------------------------------------------------------------------------------------------------------- standalone kernel (bit-compare / dev)
@triton.jit
def _ln_kernel(X, G, Bt, Y, rows, stride_row, eps, C: tl.constexpr, BM: tl.constexpr, AFFINE: tl.constexpr, OUT_BF16: tl.constexpr, RCP3: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    c = tl.arange(0, C)
    msk = r < rows
    x = tl.load(X + r[:, None].to(tl.int64) * stride_row + c[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
    if AFFINE & 1:
        g = tl.load(G + c).to(tl.float32)
    else:
        g = tl.zeros((C,), dtype=tl.float32) + 1.0
    if AFFINE & 2:
        b = tl.load(Bt + c).to(tl.float32)
    else:
        b = tl.zeros((C,), dtype=tl.float32)
    if C == 128:
        y = ln_rows_128(x, g, b, eps, AFFINE, RCP3)
    else:
        y = ln_rows_64(x, g, b, eps, AFFINE, RCP3)
    if OUT_BF16:
        y = y.to(tl.bfloat16)
    tl.store(Y + r[:, None].to(tl.int64) * C + c[None, :], y, mask=msk[:, None])


def layer_norm_triton(x2d: torch.Tensor, weight, bias, eps: float = 1e-5, out_dtype=torch.float32, BM: int = 16, num_warps: int = 4) -> torch.Tensor:
    """The standalone Triton form on a [rows, C] tensor with unit column stride (rows may be strided): torch.nn.functional.layer_norm bits."""
    rows, C = int(x2d.shape[0]), int(x2d.shape[1])
    assert C in (64, 128) and x2d.stride(1) == 1
    aff = (1 if weight is not None else 0) | (2 if bias is not None else 0)
    assert aff in (0, 1, 3)
    y = torch.empty((rows, C), dtype=out_dtype, device=x2d.device)
    g = weight if weight is not None else x2d
    b = bias if bias is not None else x2d
    grid = (triton.cdiv(rows, BM),)
    _ln_kernel[grid](x2d, g, b, y, rows, x2d.stride(0), float(np.float32(eps)), C=C, BM=BM, AFFINE=aff, OUT_BF16=(out_dtype == torch.bfloat16),
                     RCP3=RCP3, num_warps=num_warps)          # default contraction settings on purpose: the row functions are immune
    return y
