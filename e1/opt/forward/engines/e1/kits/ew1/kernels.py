"""Fused elementwise kernels for Profluent-E1 that reproduce the stock's bf16 rounding sequence bitwise.

Every kernel rounds to bf16 at exactly the points the stock rounds (torch bf16 elementwise = fp32 opmath, one
round-to-nearest-even per op; fp32 -> bf16 is cvt.rn.bf16.f32 on sm_80+):

  clamp_rope_qkv : clamp(-c, c) + RoPE on q and k (q*cos + rotate_half(q)*sin: mul -> bf16, mul -> bf16, add -> bf16),
                   clamp on v; the cos/sin tables are the stock's own fp32 caches cast once to bf16 (the stock casts
                   per call); optional (B, H, L, hd) output layout for the flex (global) layers (data movement only)
  silu_mul       : silu(a) * b with silu = a / (1 + expf(-a)) in fp32 — libdevice __nv_expf and an IEEE round-to-nearest
                   division (inline PTX div.rn.f32, no flush-to-zero: Triton's `/` is div.full and tl.div_rn flushes
                   denormals), i.e. the arithmetic of torch's silu_kernel — rounded to bf16, then one bf16 mul
  add_rmsnorm    : y = rmsnorm(rn_bf16(residual + x)) with the hub kernel's reduction code (same BLOCK_N, same
                   num_warps, same tl.sum / sqrt / division), the sum rounded to bf16 BEFORE the reduction (the stock
                   adds in bf16 then normalises); also returns the bf16 sum (the next residual)
  embed_add      : token embedding + sequence-id embedding (fp32 add) rounded once to bf16

Arithmetic variants other than the default (VARIANT constexpr) are not used by the kit.
Launchers below the kernels take and return torch tensors; each kernel's rounding points are stated at its definition.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

MAX_FUSED_SIZE_BYTES = 65536          # hub layer_norm.py: MAX_FUSED_SIZE = 65536 // x.element_size()
ROPE_RB = 32                          # (token, head) rows per program in the rope kernel
ROPE_WARPS = 8
EW_BLOCK = 4096
EW_WARPS = 8


# ----------------------------------------------------------------------------------------------------- clamp + RoPE
@triton.jit
def _rope_rows(X, OUT, POS, COS, SIN, r0, R, H, L, clip, n_cache,
               RB: tl.constexpr, HALF: tl.constexpr, TRANSPOSED: tl.constexpr, ROPE: tl.constexpr):
    # rows r = flattened (token, head) of an (B*L, H, 2*HALF) tensor; one row = one head of one token
    r = r0 + tl.arange(0, RB)
    rm = r < R
    i = tl.arange(0, HALF)
    m = rm[:, None] & (i[None, :] < HALF)
    offs = r[:, None] * (2 * HALF) + i[None, :]
    x1 = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(X + offs + HALF, mask=m, other=0.0).to(tl.float32)
    # torch clamp (TensorCompare.cu launch_clamp_scalar): min(max(v, lo), hi) in opmath; NaN propagates
    x1 = tl.minimum(tl.maximum(x1, -clip, propagate_nan=tl.PropagateNan.ALL), clip, propagate_nan=tl.PropagateNan.ALL)
    x2 = tl.minimum(tl.maximum(x2, -clip, propagate_nan=tl.PropagateNan.ALL), clip, propagate_nan=tl.PropagateNan.ALL)
    if ROPE:
        tok = r // H
        p = tl.load(POS + tok, mask=rm, other=0)
        p = tl.where(p < 0, p + n_cache, p)                    # the stock gathers cos_cached[idxs]: -1 -> last row
        toffs = p[:, None] * (2 * HALF) + i[None, :]
        c1 = tl.load(COS + toffs, mask=m, other=0.0).to(tl.float32)
        c2 = tl.load(COS + toffs + HALF, mask=m, other=0.0).to(tl.float32)
        s1 = tl.load(SIN + toffs, mask=m, other=0.0).to(tl.float32)
        s2 = tl.load(SIN + toffs + HALF, mask=m, other=0.0).to(tl.float32)
        # stock: (q * cos) + (rotate_half(q) * sin), rotate_half = cat(-x2, x1); three bf16 roundings per element
        a1 = (x1 * c1).to(tl.bfloat16).to(tl.float32)
        b1 = ((-x2) * s1).to(tl.bfloat16).to(tl.float32)
        o1 = (a1 + b1).to(tl.bfloat16)
        a2 = (x2 * c2).to(tl.bfloat16).to(tl.float32)
        b2 = (x1 * s2).to(tl.bfloat16).to(tl.float32)
        o2 = (a2 + b2).to(tl.bfloat16)
    else:
        o1 = x1.to(tl.bfloat16)
        o2 = x2.to(tl.bfloat16)
    if TRANSPOSED:
        h = r % H
        tok2 = r // H
        b = tok2 // L
        l = tok2 % L
        orow = (b * H + h) * L + l
    else:
        orow = r
    ooffs = orow[:, None] * (2 * HALF) + i[None, :]
    tl.store(OUT + ooffs, o1, mask=m)
    tl.store(OUT + ooffs + HALF, o2, mask=m)


@triton.jit
def _clamp_rope_qkv_kernel(Q, K, V, OQ, OK, OV, POS, COS, SIN, R, H, L, clip, n_cache,
                           RB: tl.constexpr, HALF: tl.constexpr, TRANSPOSED: tl.constexpr):
    r0 = tl.program_id(0) * RB
    which = tl.program_id(1)
    if which == 0:
        _rope_rows(Q, OQ, POS, COS, SIN, r0, R, H, L, clip, n_cache, RB, HALF, TRANSPOSED, True)
    elif which == 1:
        _rope_rows(K, OK, POS, COS, SIN, r0, R, H, L, clip, n_cache, RB, HALF, TRANSPOSED, True)
    else:
        _rope_rows(V, OV, POS, COS, SIN, r0, R, H, L, clip, n_cache, RB, HALF, TRANSPOSED, False)


def clamp_rope_qkv(q, k, v, position_ids, cos_tab, sin_tab, clip, transposed=False):
    """q, k, v: (B, L, H, hd) bf16 contiguous (the q/k/v GEMM outputs viewed per head); position_ids: (B, L) int64;
    cos_tab/sin_tab: (n_cache, hd) bf16 = the rotary module's fp32 caches cast to bf16 (the stock's per-call cast).
    Returns (q, k, v) with logical shape (B, L, H, hd); with transposed=True the storage is (B, H, L, hd) contiguous
    (the flex route's own transpose(1, 2).contiguous() is then a no-op)."""
    B, L, H, hd = q.shape
    assert k.shape == q.shape and v.shape == q.shape, (q.shape, k.shape, v.shape)
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert hd % 2 == 0 and (hd // 2) & (hd // 2 - 1) == 0, hd
    assert position_ids.shape == (B, L) and position_ids.dtype == torch.int64
    assert cos_tab.dtype == torch.bfloat16 and sin_tab.dtype == torch.bfloat16
    assert cos_tab.shape == sin_tab.shape and cos_tab.shape[1] == hd and cos_tab.is_contiguous() and sin_tab.is_contiguous()
    pos = position_ids if position_ids.is_contiguous() else position_ids.contiguous()
    shape = (B, H, L, hd) if transposed else (B, L, H, hd)
    oq, ok, ov = (torch.empty(shape, dtype=q.dtype, device=q.device) for _ in range(3))
    R = B * L * H
    grid = (triton.cdiv(R, ROPE_RB), 3)
    _clamp_rope_qkv_kernel[grid](q, k, v, oq, ok, ov, pos, cos_tab, sin_tab, R, H, L, float(clip), int(cos_tab.shape[0]),
                                 RB=ROPE_RB, HALF=hd // 2, TRANSPOSED=transposed, num_warps=ROPE_WARPS, enable_fp_fusion=False)
    if transposed:
        return oq.transpose(1, 2), ok.transpose(1, 2), ov.transpose(1, 2)
    return oq, ok, ov


# ------------------------------------------------------------------------------------------------------- silu * mul
@triton.jit
def _div_rn_ieee(x, y):
    """fp32 IEEE-754 round-to-nearest-even division with denormals preserved (PTX div.rn.f32): the division of torch's
    CUDA kernels (nvcc default -prec-div=true, no -ftz)."""
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [x, y], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _silu_mul_kernel(A, Bp, OUT, n, BLOCK: tl.constexpr, VARIANT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(A + offs, mask=m, other=0.0).to(tl.float32)
    b = tl.load(Bp + offs, mask=m, other=0.0).to(tl.float32)
    if VARIANT == 0:
        # torch ActivationSiluKernel.cu: x_acc / (opmath_t(1) + ::exp(-x_acc)) -> expf + IEEE RN division (no FTZ)
        s = _div_rn_ieee(a, 1.0 + libdevice.exp(-a))
    elif VARIANT == 1:
        s = tl.div_rn(a, 1.0 + libdevice.exp(-a))  # decode: libdevice division under Triton's FTZ (flushes denormals)
    elif VARIANT == 2:
        s = a / (1.0 + tl.exp(-a))                  # decode: ex2.approx exp + div.full
    else:
        s = a / (1.0 + libdevice.exp(-a))           # decode: expf + div.full
    s = s.to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + offs, (s * b).to(tl.bfloat16), mask=m)


def silu_mul(a, b, variant=0):
    assert a.shape == b.shape and a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert a.is_contiguous() and b.is_contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    _silu_mul_kernel[(triton.cdiv(n, EW_BLOCK),)](a, b, out, n, BLOCK=EW_BLOCK, VARIANT=variant, num_warps=EW_WARPS,
                                                   enable_fp_fusion=False)
    return out


@triton.jit
def _silu_kernel(A, OUT, n, BLOCK: tl.constexpr, VARIANT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(A + offs, mask=m, other=0.0).to(tl.float32)
    if VARIANT == 0:
        s = _div_rn_ieee(a, 1.0 + libdevice.exp(-a))
    elif VARIANT == 1:
        s = tl.div_rn(a, 1.0 + libdevice.exp(-a))
    elif VARIANT == 2:
        s = a / (1.0 + tl.exp(-a))
    else:
        s = a / (1.0 + libdevice.exp(-a))
    tl.store(OUT + offs, s.to(tl.bfloat16), mask=m)


def silu_bf16(a, variant=0):
    """silu alone: bf16 in, bf16 out (the launcher the kit's tests drive over every bf16 value)."""
    out = torch.empty_like(a)
    n = a.numel()
    _silu_kernel[(triton.cdiv(n, EW_BLOCK),)](a, out, n, BLOCK=EW_BLOCK, VARIANT=variant, num_warps=EW_WARPS, enable_fp_fusion=False)
    return out


@triton.jit
def _gelu_kernel(A, OUT, n, BLOCK: tl.constexpr, VARIANT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(A + offs, mask=m, other=0.0).to(tl.float32)
    # torch ActivationGeluKernel.cu (approximate='none'): x * 0.5 * (1 + erf(x * M_SQRT1_2))
    if VARIANT == 0:
        y = x * 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
    else:
        y = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(OUT + offs, y.to(tl.bfloat16), mask=m)


def gelu_bf16(a, variant=0):
    out = torch.empty_like(a)
    n = a.numel()
    _gelu_kernel[(triton.cdiv(n, EW_BLOCK),)](a, out, n, BLOCK=EW_BLOCK, VARIANT=variant, num_warps=EW_WARPS, enable_fp_fusion=False)
    return out


# ------------------------------------------------------------------------------------------------ residual + RMSNorm
@triton.jit
def _add_rmsnorm_kernel(
    X,  # pointer to the input (the block output, bf16)
    RESIDUAL,  # pointer to the residual (bf16)
    Y,  # pointer to the output (bf16)
    RESIDUAL_OUT,  # pointer to the bf16 sum (the next residual)
    W,  # pointer to the weights (fp32)
    Rstd,  # pointer to the 1/std
    stride_x_row,
    stride_res_row,
    stride_y_row,
    stride_res_out_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    ROUND_SUM: tl.constexpr,
):
    # hub kernels-community/triton-layer-norm layer_norm.py _layer_norm_fwd_1pass_kernel with IS_RMS_NORM, HAS_RESIDUAL,
    # no bias / dropout / rowscale / x1 / w1, and the residual sum rounded to bf16 before the reduction (ROUND_SUM)
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
    x += residual
    if ROUND_SUM:
        x = x.to(tl.bfloat16).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, y, mask=mask)


def add_rmsnorm(residual, x, weight, eps, num_warps, store_sum=True, round_sum=True):
    """(y, s): s = rn_bf16(residual + x) [None if not store_sum], y = rmsnorm(s) * weight in bf16 — the stock's
    `residual + hidden_states` (bf16 add) followed by the hub Triton rms_norm (fp32 weight), as ONE kernel launched
    with the hub kernel's BLOCK_N and the pinned num_warps (the stock's autotune choice)."""
    assert x.shape == residual.shape and x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16, (x.shape, residual.shape, x.dtype, residual.dtype)
    shp = x.shape
    N = shp[-1]
    x2 = x.reshape(-1, N)
    r2 = residual.reshape(-1, N)
    assert x2.stride(-1) == 1 and r2.stride(-1) == 1
    assert weight.shape == (N,) and weight.stride(-1) == 1
    M = x2.shape[0]
    BLOCK_N = min(MAX_FUSED_SIZE_BYTES // x.element_size(), triton.next_power_of_2(N))
    assert N <= BLOCK_N
    y = torch.empty_like(x2)
    s = torch.empty_like(x2) if store_sum else y
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device.index):
        _add_rmsnorm_kernel[(M,)](x2, r2, y, s, weight, rstd, x2.stride(0), r2.stride(0), y.stride(0), s.stride(0), N, eps,
                                  BLOCK_N=BLOCK_N, STORE_RESIDUAL_OUT=store_sum, ROUND_SUM=round_sum, num_warps=int(num_warps))
    return y.reshape(shp), (s.reshape(shp) if store_sum else None)


# --------------------------------------------------------------------------------------------------------- embeddings
@triton.jit
def _embed_add_kernel(IDS, SIDS, TOK_W, SEQ_W, OUT, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    tok = tl.load(IDS + row)
    sid = tl.load(SIDS + row)
    sid = tl.maximum(sid, 0)                                   # sequence_ids.clamp(min=0)
    cols = tl.arange(0, BLOCK_N)
    m = cols < N
    a = tl.load(TOK_W + tok * N + cols, mask=m, other=0.0)
    b = tl.load(SEQ_W + sid * N + cols, mask=m, other=0.0)
    tl.store(OUT + row * N + cols, (a + b).to(tl.bfloat16), mask=m)


def embed_add(input_ids, sequence_ids, tok_weight, seq_weight):
    """bf16( embed_tokens(input_ids) + embed_seq_id(sequence_ids.clamp(min=0)) ) with fp32 weights: fp32 add, one rounding."""
    assert tok_weight.dtype == torch.float32 and seq_weight.dtype == torch.float32
    assert tok_weight.is_contiguous() and seq_weight.is_contiguous() and tok_weight.shape[1] == seq_weight.shape[1]
    B, L = input_ids.shape
    N = tok_weight.shape[1]
    ids = input_ids.reshape(-1).contiguous()
    sids = sequence_ids.reshape(-1).contiguous()
    out = torch.empty((B * L, N), dtype=torch.bfloat16, device=tok_weight.device)
    _embed_add_kernel[(B * L,)](ids, sids, tok_weight, seq_weight, out, N, BLOCK_N=triton.next_power_of_2(N), num_warps=4,
                                enable_fp_fusion=False)
    return out.reshape(B, L, N)
