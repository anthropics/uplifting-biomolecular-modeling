"""vortex RMSNorm.forward in one Triton launch, bit-identical to the stock, with the L2-norm reduction computed inside the kernel in ATen's order
(no separate `x.norm(2, dim=-1, keepdim=True)` launch), and a variant that emits the following FP8 projection's e4m3 input directly.

Stock (vortex/model/layers.py RMSNorm.forward):  y = x / (x.norm(2, dim=-1, keepdim=True) * H ** (-1/2) + eps);  return scale * y
with x, scale bf16 — every op is a bf16 tensor op: fp32 arithmetic rounded to bf16 once per op.

The reduction reproduces torch 2.7.1's CUDA reduce for a bf16 tensor whose reduced (last) dimension is contiguous
(aten/src/ATen/native/cuda/Reduce.cuh + ReduceNormKernel.cu: gpu_reduce_kernel<BFloat16, BFloat16 out>, NormTwoOps<BFloat16, float, BFloat16>,
vt0 = ReduceConfig::input_vec_size = 4):
  * setReduceConfig: reduction on the fastest-striding dim, vectorize_input (dim0 = H/4 > 128); block_width bw = min(32, .) then widened to
    512 / block_height, block_height bh = min(last_pow2(rows), 16); input_mult[BLOCK_X] = split_input(bw); the row is additionally split
    across the bh warps (input_mult[BLOCK_Y]) when values_per_thread = ceil(H / bw) >= min(16 bh, 256).  Either way a row is reduced by
    T threads (T = bw, or bw*bh when split; T = 32 for H = 4096 with rows >= 16, T = 512 for H = 8192 at any rows) with thread t owning the
    aligned 4-vectors t + T i, i ascending — a contiguous 4T-element chunk per i;
  * input_vectorized_thread_reduce_impl: four accumulators per thread, acc_j <- acc_j + x*x over that thread's vectors in order (fp32;
    nvcc contracts it to one FFMA; x*x is exact for bf16 x, so fused and unfused agree on every normal value and only the fused form keeps
    subnormal squares — the kernel uses fma.rn.f32);  then value = ((acc_0 + acc_1) + acc_2) + acc_3;
  * block_y_reduce then block_x_reduce's shared-memory phase: halves-first, P[t] += P[t + off] for off = T/2, T/4, ..., 32;  then the warp
    shfl_down phase with offsets 1, 2, 4, 8, 16 — lane 0 ends with the adjacent-pair halving tree ((v0+v1)+(v2+v3)) + ((v4+v5)+(v6+v7)) ...;
    project: sqrtf (IEEE round-to-nearest) -> BFloat16 (round-to-nearest-even).
The kernel pins that ORDER by explicit dataflow (a [T, 4] accumulator block per row, one load per i, tl.reshape/tl.split pair sums naming
both operands of every add); which physical thread executes which add is irrelevant to the bits.  Every fp32 operation is inline PTX with
round-to-nearest and WITHOUT flush-to-zero (fma.rn.f32, add.rn.f32, mul.rn.f32, div.rn.f32, sqrt.rn.f32), as in ATen's build; Triton's own
lowering of *, /, tl.math.* flushes subnormals.  The tail is the stock op sequence: d = bf16(bf16(n * H**-0.5) + eps); y = bf16(x / d);
out = bf16(scale * y).  The e4m3 variant then applies Transformer Engine's cast to `out`: e4m3(fp32(out) * scale_fwd) with saturation
(cvt.rn.satfinite.e4m3x2.f32), scale_fwd = the projection's fp8_meta["scaling_fwd"].scale[0] (static while grad is disabled).

Predicate (served here; anything else takes the two-launch path — ATen reduce + tail kernel, base._rms_fused_forward names it once), `served()` / `rmsnorm_rule.aten_partials()`:  x on CUDA,
bf16, contiguous, 8-byte aligned; scale bf16 with H elements; H % 4 == 0 and H/4 > 128; T from the rule above with H % (4T) == 0; and not the
case where ATen may add a second, grid-wide pass (row split across warps AND ceil(H / T) >= 256, i.e. H >= 131072).  evo2_7b (H 4096): T = 32
for rows >= 16, 64 / 128 / 256 / 512 for rows 8-15 / 4-7 / 2-3 / 1;  evo2_40b (H 8192): T = 512 for every row count.

Interfaces (each requires its predicate; the callers test it and take their two-launch or stock path otherwise):
  * served(x, scale) / rmsnorm_forward(x, scale, eps): RMSNorm.forward(x), bf16 in, bf16 out (E11, base._rms_fused_forward);
  * rmsnorm_forward_e4m3(x, scale, eps, q_out, q_scale[, bf16_out]): the same output cast to e4m3 the way Transformer Engine's quantizer
    casts it, written into the following FP8 projection's input buffer (E63, fp8emit.proj_norm), optionally the bf16 output as well;
  * served_add(a, u, scale[, bias]) / add_rmsnorm(a, bias, u, scale, eps) -> (s, y): the preceding residual add folded in front of the
    norm — s = bf16(bf16(a + bias) + u) with torch's bf16 adds (bias None: bf16(a + u)), y = RMSNorm(s) — in one launch, s stored for the
    block's own residual path (E56: bias + residual + post_norm of a Hyena block; E65: the attention block's residual + post_norm; E66: a
    block's closing add carried into the next block's pre_norm);
  * add_rmsnorm_e4m3(a, u, scale, eps, q_out, q_scale) -> s: the same with the e4m3 output (E66 in front of an FP8 projection, fp8emit).

Why the fp32 total is reproduced and not only the rounded norm: bf16 output equality alone hides most summation-order differences; the
fp32 sum of squares is the order-discriminating quantity, so the kernel follows ATen's partial-sum ownership and tree exactly.  With
subnormal squares only the fused multiply-add matches ATen (a `sum` of separately rounded products is a different computation there) —
hence fma.rn.f32 and the .rn PTX without flush-to-zero in place of Triton's default lowering.

Geometry: one row per program, NUM_WARPS warps, the tail in CHUNK-element chunks (rmsnorm_rule.CHUNK; served widths are multiples of it) —
launch geometry only; the arithmetic and its order are fixed by the dataflow above.

Anything outside the predicate (other widths, dtypes, strides or alignments, and row counts whose ATen scheme is grid-dependent) is not
approximated here: base._rms_fused_forward names the width once on stderr and runs ATen's own norm followed by the E11 tail kernel; the
add variants' callers run torch's add and the module's own norm.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from evo2_opt.kit.rmsnorm_rule import CHUNK, aten_partials, row_rule


@triton.jit
def _fma_rn(a, b, c):
    return tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _div_rn(a, b):
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sqrt_rn(a):
    return tl.inline_asm_elementwise("sqrt.rn.f32 $0, $1;", "=f,f", [a], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _pair_sum(p, N: tl.constexpr):
    """[2N] -> [N]: out[m] = p[2m] + p[2m+1] — one shfl_down level of the warp tree as it reaches lane 0."""
    a, b = tl.split(tl.reshape(p, (N, 2)))
    return _add_rn(a, b)


@triton.jit
def _halve(p, N: tl.constexpr):
    """[2N] -> [N]: out[t] = p[t] + p[t + N] — one halves-first level (block_y_reduce / block_x_reduce's shared-memory phase)."""
    a, b = tl.split(tl.permute(tl.reshape(p, (2, N)), (1, 0)))
    return _add_rn(a, b)


@triton.jit
def _row_sumsq(base, H: tl.constexpr, T: tl.constexpr):
    """fp32 sum of squares of the H contiguous bf16 elements at `base`, in ATen's reduce order with T partial threads. Returns a [1] tensor."""
    t = tl.arange(0, T)
    slot = tl.arange(0, 4)
    tile = t[:, None] * 4 + slot[None, :]
    acc = tl.zeros((T, 4), dtype=tl.float32)
    for i in tl.static_range(H // (4 * T)):
        x = tl.load(base + tile + 4 * T * i).to(tl.float32)
        acc = _fma_rn(x, x, acc)
    lo, hi = tl.split(tl.permute(tl.reshape(acc, (T, 2, 2)), (0, 2, 1)))
    a0, a1 = tl.split(lo)
    a2, a3 = tl.split(hi)
    p = _add_rn(_add_rn(_add_rn(a0, a1), a2), a3)
    if T >= 512:
        p = _halve(p, 256)
    if T >= 256:
        p = _halve(p, 128)
    if T >= 128:
        p = _halve(p, 64)
    if T >= 64:
        p = _halve(p, 32)
    p = _pair_sum(p, 16)
    p = _pair_sum(p, 8)
    p = _pair_sum(p, 4)
    p = _pair_sum(p, 2)
    return _pair_sum(p, 1)


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, o_ptr, q_ptr, qscale_ptr, inv_sqrt_h, eps, H: tl.constexpr, T: tl.constexpr, CH: tl.constexpr, EMIT: tl.constexpr):
    """One program per row.  EMIT 0: bf16 out | 1: e4m3 out | 2: both."""
    row = tl.program_id(0).to(tl.int64)
    base = x_ptr + row * H
    ss = _row_sumsq(base, H, T)
    nb = _sqrt_rn(ss).to(tl.bfloat16)                                                   # x.norm(2, dim=-1, keepdim=True)
    d = _mul_rn(nb.to(tl.float32), inv_sqrt_h).to(tl.bfloat16).to(tl.float32)         # bf16(norm * H**-0.5)
    d = _add_rn(d, eps).to(tl.bfloat16).to(tl.float32)                               # bf16(. + eps)
    dd = tl.broadcast_to(d, (CH,))
    if EMIT != 0:
        qs = tl.load(qscale_ptr)
    offs = tl.arange(0, CH)
    for c in tl.static_range(H // CH):
        x = tl.load(base + c * CH + offs).to(tl.float32)
        y = _div_rn(x, dd).to(tl.bfloat16).to(tl.float32)                             # bf16(x / d)
        w = tl.load(w_ptr + c * CH + offs).to(tl.float32)
        o = _mul_rn(w, y).to(tl.bfloat16)                                            # bf16(scale * y)
        if EMIT != 1:
            tl.store(o_ptr + row * H + c * CH + offs, o)
        if EMIT != 0:
            tl.store(q_ptr + row * H + c * CH + offs, _mul_rn(o.to(tl.float32), qs).to(tl.float8e4nv))   # e4m3 satfinite(fp32(out) * scale_fwd)


NUM_WARPS = 4          # launch geometry (one row per program)


def served(x: torch.Tensor, scale: torch.Tensor) -> bool:
    """The predicate: inputs this module reproduces bit for bit; everything else belongs to the two-launch path."""
    H = x.shape[-1]
    return (x.is_cuda and x.dtype == torch.bfloat16 and scale.dtype == torch.bfloat16 and scale.numel() == H and scale.is_contiguous()
            and x.is_contiguous() and x.data_ptr() % 8 == 0 and H > 0 and row_rule(x.numel() // H, H) is not None)


def rmsnorm_forward(x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """== RMSNorm.forward(x) of vortex (scale * (x / (x.norm(2, -1, keepdim=True) * H**-0.5 + eps))), bf16 in, bf16 out; requires served(x, scale)."""
    H = x.shape[-1]
    rows = x.numel() // H
    out = torch.empty_like(x)
    _rmsnorm_kernel[(rows,)](x, scale, out, out, scale, float(H ** (-1.0 / 2)), float(eps), H=H, T=aten_partials(rows, H), CH=CHUNK, EMIT=0, num_warps=NUM_WARPS)
    return out


def rmsnorm_forward_e4m3(x: torch.Tensor, scale: torch.Tensor, eps: float, q_out: torch.Tensor, q_scale: torch.Tensor, bf16_out: torch.Tensor | None = None):
    """Writes e4m3(fp32(RMSNorm.forward(x)) * q_scale) with saturation into q_out ((rows, H) float8_e4m3fn or uint8 storage of it) — the bytes
    Transformer Engine's quantizer produces from the bf16 RMSNorm output with scale = q_scale (a 1-element fp32 tensor).  With bf16_out given,
    the bf16 output is written as well.  Requires served(x, scale)."""
    H = x.shape[-1]
    rows = x.numel() // H
    q = q_out.view(torch.float8_e4m3fn) if q_out.dtype == torch.uint8 else q_out
    o = bf16_out if bf16_out is not None else x
    _rmsnorm_kernel[(rows,)](x, scale, o, q, q_scale, float(H ** (-1.0 / 2)), float(eps), H=H, T=aten_partials(rows, H), CH=CHUNK, EMIT=2 if bf16_out is not None else 1,
                             num_warps=NUM_WARPS)
    return q_out


# ---- the residual add folded in front of the norm: s = [bf16(a + bias)] + u, y = RMSNorm(s), one launch
@triton.jit
def _add_rmsnorm_kernel(a_ptr, bias_ptr, u_ptr, s_ptr, w_ptr, y_ptr, q_ptr, qscale_ptr, inv_sqrt_h, eps, H: tl.constexpr, T: tl.constexpr, CH: tl.constexpr,
                        HAS_BIAS: tl.constexpr, EMIT: tl.constexpr):
    """One program per row: s = bf16([bf16(a + bias)] + u) written in CH-element chunks; bar.sync (the program's writes become visible to all
    of its threads); the reduction re-reads s in ATen's tile order (_row_sumsq); the tail re-reads s chunk-wise and writes y = RMSNorm(s)
    (EMIT 0), its e4m3 cast e4m3(fp32(y) * qscale) (EMIT 1), or both (EMIT 2)."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, CH)
    for c in tl.static_range(H // CH):
        a = tl.load(a_ptr + row * H + c * CH + offs).to(tl.float32)
        r = tl.load(u_ptr + row * H + c * CH + offs).to(tl.float32)
        if HAS_BIAS:
            a = _add_rn(a, tl.load(bias_ptr + c * CH + offs).to(tl.float32)).to(tl.bfloat16).to(tl.float32)     # bf16(zo + bias)
        tl.store(s_ptr + row * H + c * CH + offs, _add_rn(a, r).to(tl.bfloat16))                              # bf16(. + u)
    tl.debug_barrier()
    ss = _row_sumsq(s_ptr + row * H, H, T)
    nb = _sqrt_rn(ss).to(tl.bfloat16)
    d = _mul_rn(nb.to(tl.float32), inv_sqrt_h).to(tl.bfloat16).to(tl.float32)
    d = _add_rn(d, eps).to(tl.bfloat16).to(tl.float32)
    dd = tl.broadcast_to(d, (CH,))
    if EMIT != 0:
        qs = tl.load(qscale_ptr)
    for c in tl.static_range(H // CH):
        sf = tl.load(s_ptr + row * H + c * CH + offs).to(tl.float32)
        yv = _div_rn(sf, dd).to(tl.bfloat16).to(tl.float32)                                                     # bf16(s / d)
        w = tl.load(w_ptr + c * CH + offs).to(tl.float32)
        o = _mul_rn(w, yv).to(tl.bfloat16)                                                                      # bf16(scale * .)
        if EMIT != 1:
            tl.store(y_ptr + row * H + c * CH + offs, o)
        if EMIT != 0:
            tl.store(q_ptr + row * H + c * CH + offs, _mul_rn(o.to(tl.float32), qs).to(tl.float8e4nv))           # e4m3 satfinite(fp32(y) * scale_fwd)


def served_add(a: torch.Tensor, u: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None) -> bool:
    return (a.shape == u.shape and a.dtype == torch.bfloat16 and a.is_contiguous() and a.data_ptr() % 8 == 0 and served(u, scale)
            and (bias is None or (bias.dtype == torch.bfloat16 and bias.is_contiguous() and bias.numel() == u.shape[-1])))


def add_rmsnorm(a: torch.Tensor, bias: torch.Tensor | None, u: torch.Tensor, scale: torch.Tensor, eps: float):
    """-> (s, y):  s = (a + bias) + u  [bias None: a + u]  as torch's bf16 adds;  y = RMSNorm(s).  Requires served_add(a, u, scale, bias)."""
    H = u.shape[-1]
    rows = u.numel() // H
    s = torch.empty_like(u)
    y = torch.empty_like(u)
    _add_rmsnorm_kernel[(rows,)](a, bias if bias is not None else scale, u, s, scale, y, y, scale, float(H ** (-1.0 / 2)), float(eps), H=H, T=aten_partials(rows, H), CH=CHUNK,
                                 HAS_BIAS=bias is not None, EMIT=0, num_warps=NUM_WARPS)
    return s, y


def add_rmsnorm_e4m3(a: torch.Tensor, u: torch.Tensor, scale: torch.Tensor, eps: float, q_out: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
    """-> s = a + u as torch's bf16 add, and writes e4m3(fp32(RMSNorm(s)) * q_scale) with saturation into q_out ((rows, H) float8_e4m3fn or
    uint8 storage of it): the bytes rmsnorm_forward_e4m3 writes for s.  Requires served_add(a, u, scale)."""
    H = u.shape[-1]
    rows = u.numel() // H
    s = torch.empty_like(u)
    q = q_out.view(torch.float8_e4m3fn) if q_out.dtype == torch.uint8 else q_out
    _add_rmsnorm_kernel[(rows,)](a, scale, u, s, scale, s, q, q_scale, float(H ** (-1.0 / 2)), float(eps), H=H, T=aten_partials(rows, H), CH=CHUNK,
                                 HAS_BIAS=False, EMIT=1, num_warps=NUM_WARPS)
    return s
