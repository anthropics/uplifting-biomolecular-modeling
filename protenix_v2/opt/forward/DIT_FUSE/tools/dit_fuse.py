"""dit_fuse.py — DiT-fuse lever F: byte-EXACT fused elementwise kernels for the Protenix v2 diffusion transformer blocks.

Stock (fp32, autocast disabled in the sampler) launches, per token DiffusionTransformerBlock, ~17 tiny elementwise kernels between the LayerNorms /
GEMMs / SDPA. This module replaces four chains by ONE Triton kernel each, keeping every stock rounding point:
  F-ada    AdaptiveLayerNorm tail : out = sigmoid(x1) * a_n + x2                 stock: sigmoid | mul | add          (3 -> 1)   [a_n = LN_a(a), x1 = linear_s(LN_s(s)), x2 = linear_nobias_s(LN_s(s))]
  F-gate   Attention gating       : out[n, h*c] = o[h, n, c] * sigmoid(g[n, h*c])  stock: sigmoid | mul | (transpose+reshape copy)   (3 -> 1)
  F-res    output gate + residual : out = sigmoid(gl) * x + res                  stock: sigmoid | mul | add          (3 -> 1)   [APB: a = sigmoid(linear_a_last(s)) * a ; block: attn_out + a]
  F-swiglu transition             : out = silu(x) * y                            stock: silu | mul                  (2 -> 1)
Exactness recipe: fp32 loads/stores; each stock kernel boundary is an explicit fp32 rounding (t = rn(sig * a); out = rn(t + b)); launched with
enable_fp_fusion=False so mul+add is never contracted to FMA; sigmoid/silu use libdevice exp (accurate expf) and IEEE division (div_rn), which is what
ATen's CUDA kernels compute for float (sigmoid: 1/(1+exp(-x)); silu: x/(1+exp(-x))). FORMULA is selectable per kernel (env DIT_FUSE_SIGMOID = 'rcp'|'div',
DIT_FUSE_EXP = 'libdevice'|'tl'); the defaults are the variants whose results equal ATen's bitwise.
LayerNorms and GEMMs stay stock (fast_layernorm / cuBLAS): re-implementing them changes reduction order (not bitwise).
Env: DIT_FUSE=ada,gate,res,swiglu (subset; default all when installed).
"""
from __future__ import annotations
import os, sys
import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice as _ld          # triton >= 3.0
except Exception:                                                # pragma: no cover
    try:
        from triton.language.extra.cuda import libdevice as _ld
    except Exception:
        from triton.language import math as _ld                  # triton 2.x

STATS = {"calls": {"ada": 0, "gate": 0, "res": 0, "swiglu": 0}, "fallback": 0, "cfg": {}}
_SIG = os.environ.get("DIT_FUSE_SIGMOID", "rcp")      # 'rcp': 1/(1+exp(-x)) as one IEEE division of 1 by (1+e) ; 'div' identical math, kept for A/B naming
_EXP = os.environ.get("DIT_FUSE_EXP", "libdevice")    # 'libdevice' (accurate expf) | 'tl' (tl.exp -> ex2.approx based; expected NOT bitwise)
STATS["cfg"] = {"sigmoid": _SIG, "exp": _EXP}
_USE_LD = 1 if _EXP == "libdevice" else 0


@triton.jit
def _expf(x, USE_LD: tl.constexpr):
    if USE_LD:
        return _ld.exp(x)
    else:
        return tl.exp(x)


@triton.jit
def _sigmoidf(x, USE_LD: tl.constexpr):
    # ATen (UnaryOpsKernel.cu, float path): one / (one + std::exp(-a))  -> expf, add, IEEE div
    e = _expf(-x, USE_LD)
    return _ld.div_rn(1.0, 1.0 + e)


@triton.jit
def _ada_tail_kernel(AN, X1, X2, OUT, n_elem, BLOCK: tl.constexpr, USE_LD: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)          # int64 program index: element offsets stay exact past 2**31 elements per launch
    offs = pid * BLOCK + tl.arange(0, BLOCK); m = offs < n_elem
    an = tl.load(AN + offs, mask=m, other=0.0)
    x1 = tl.load(X1 + offs, mask=m, other=0.0)
    x2 = tl.load(X2 + offs, mask=m, other=0.0)
    sg = _sigmoidf(x1, USE_LD)             # kernel 1 (sigmoid) -> fp32
    t = sg * an                            # kernel 2 (mul)     -> fp32 rounding (no FMA: enable_fp_fusion=False)
    out = t + x2                           # kernel 3 (add)
    tl.store(OUT + offs, out, mask=m)


@triton.jit
def _res_gate_kernel(GL, X, RES, OUT, n_elem, BLOCK: tl.constexpr, USE_LD: tl.constexpr, HAS_RES: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)          # int64 program index: element offsets stay exact past 2**31 elements per launch
    offs = pid * BLOCK + tl.arange(0, BLOCK); m = offs < n_elem
    gl = tl.load(GL + offs, mask=m, other=0.0)
    x = tl.load(X + offs, mask=m, other=0.0)
    sg = _sigmoidf(gl, USE_LD)
    t = sg * x                             # stock: a = sigmoid(linear_a_last(s)) * a   (or a *= sigmoid(..) in-place: same rounding)
    if HAS_RES:
        r = tl.load(RES + offs, mask=m, other=0.0)
        t = t + r                          # stock: attn_out = attn_out + a  (block residual)
    tl.store(OUT + offs, t, mask=m)


@triton.jit
def _swiglu_kernel(X, Y, OUT, n_elem, BLOCK: tl.constexpr, USE_LD: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)          # int64 program index: element offsets stay exact past 2**31 elements per launch
    offs = pid * BLOCK + tl.arange(0, BLOCK); m = offs < n_elem
    x = tl.load(X + offs, mask=m, other=0.0)
    y = tl.load(Y + offs, mask=m, other=0.0)
    # ATen silu (ActivationSiluKernel.cu, float): x / (one + ::exp(-x))
    e = _expf(-x, USE_LD)
    s = _ld.div_rn(x, 1.0 + e)             # kernel 1 (silu) -> fp32
    out = s * y                            # kernel 2 (mul)
    tl.store(OUT + offs, out, mask=m)


@triton.jit
def _attn_gate_kernel(O, G, OUT, N, HC, C, s_oh, s_on, BLOCK: tl.constexpr, USE_LD: tl.constexpr):
    # out[n, j] = o[h, n, c] * sigmoid(g[n, j]),  j = h*C + c ; o is [H, N, C] (SDPA output, contiguous per head), g/out are [N, H*C] contiguous
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK); m = offs < N * HC
    n = offs // HC; j = offs % HC; h = j // C; c = j % C
    g = tl.load(G + offs, mask=m, other=0.0)
    o = tl.load(O + h * s_oh + n * s_on + c, mask=m, other=0.0)
    sg = _sigmoidf(g, USE_LD)
    out = o * sg                           # stock: o = o * g  (g = sigmoid(linear_g(q_x)) viewed [N,H,C]; o transposed to [N,H,C])
    tl.store(OUT + offs, out, mask=m)


_BLOCK = 1024


def _flat(*ts):
    for t in ts:
        assert t.dtype == torch.float32 and t.is_contiguous(), (t.dtype, t.shape, t.stride())


def ada_tail(a_n: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """sigmoid(x1) * a_n + x2 (all same shape, fp32 contiguous)."""
    _flat(a_n, x1, x2); assert a_n.shape == x1.shape == x2.shape
    out = torch.empty_like(a_n); n = a_n.numel()
    _ada_tail_kernel[(triton.cdiv(n, _BLOCK),)](a_n, x1, x2, out, n, BLOCK=_BLOCK, USE_LD=_USE_LD, num_warps=4, enable_fp_fusion=False)
    STATS["calls"]["ada"] += 1
    return out


def res_gate(gl: torch.Tensor, x: torch.Tensor, res: torch.Tensor | None, _kind: str = "res") -> torch.Tensor:
    """sigmoid(gl) * x (+ res).  (_kind only selects the STATS counter: 'res' or 'gate')"""
    _flat(gl, x); assert gl.shape == x.shape
    if res is not None:
        _flat(res); assert res.shape == x.shape
    out = torch.empty_like(x); n = x.numel()
    _res_gate_kernel[(triton.cdiv(n, _BLOCK),)](gl, x, res if res is not None else x, out, n, BLOCK=_BLOCK, USE_LD=_USE_LD, HAS_RES=res is not None, num_warps=4, enable_fp_fusion=False)
    STATS["calls"][_kind] += 1
    return out


def swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _flat(x, y); assert x.shape == y.shape
    out = torch.empty_like(x); n = x.numel()
    _swiglu_kernel[(triton.cdiv(n, _BLOCK),)](x, y, out, n, BLOCK=_BLOCK, USE_LD=_USE_LD, num_warps=4, enable_fp_fusion=False)
    STATS["calls"]["swiglu"] += 1
    return out


def attn_gate(o: torch.Tensor, g_logits: torch.Tensor) -> torch.Tensor:
    """o: [..., H, N, C] SDPA output (last dim contiguous); g_logits: [..., N, H*C] contiguous -> [..., N, H*C] = transpose(o)*sigmoid(g) flattened."""
    H, N, C = o.shape[-3], o.shape[-2], o.shape[-1]
    lead = o.shape[:-3]
    assert g_logits.shape[-1] == H * C and g_logits.shape[-2] == N and g_logits.dtype == torch.float32 and o.dtype == torch.float32
    o3 = o.reshape(-1, H, N, C); g2 = g_logits.reshape(-1, N, H * C)
    assert o3.shape[0] == g2.shape[0]
    if not g2.is_contiguous(): g2 = g2.contiguous()
    out = torch.empty_like(g2); n = N * H * C
    for b in range(o3.shape[0]):        # leading batch (N_sample) is 1 in the sampler as pinned; loop keeps the kernel 1-D simple
        ob = o3[b]
        assert ob.stride(-1) == 1
        _attn_gate_kernel[(triton.cdiv(n, _BLOCK),)](ob, g2[b], out[b], N, H * C, C, ob.stride(0), ob.stride(1), BLOCK=_BLOCK, USE_LD=_USE_LD, num_warps=4, enable_fp_fusion=False)
    STATS["calls"]["gate"] += 1
    return out.reshape(*lead, N, H * C) if lead else out.reshape(N, H * C) if o.dim() == 3 else out
