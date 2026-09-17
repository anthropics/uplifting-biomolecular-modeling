// edm_f32.cuh — the float32 arithmetic vocabulary of the edm ops (device code only; included by the .cu.cc files).
//
// Every operation the kernels perform on a value that reaches an output is ONE named PTX instruction spelled here as inline asm, so the
// arithmetic is fixed by this text: the compiler cannot contract a multiply and an add into an fma, pick a division algorithm, or choose
// the flush mode. Suffix meanings (PTX ISA): .rn = round to nearest even (an explicit rounding modifier also forbids ptxas from fusing
// the instruction into an fma); .ftz = subnormal inputs and results are treated as sign-preserving zero. TensorFlow 2.17.1's GPU float32
// kernels run in that flush mode: its hand-written CUDA kernels are compiled with -fcuda-flush-denormals-to-zero
// (tensorflow/tensorflow.bzl:1906-1909) and its MLIR-generated elementwise kernels with enable_ftz for f32
// (tensorflow/core/kernels/mlir_generated/build_defs.bzl:164,396 -> xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc:313-319).
//
// exp: TensorFlow's kernels call libdevice's __nv_expf (CUDA 12.3) — the MLIR kernels through the libdevice found at run time, the CUDA
// kernels through the one linked at build time. `expf_libdevice` below is the same routine as this toolkit links it (nvcc --ftz=true selects
// its flush-to-zero path, the path TensorFlow's kernels take); `expf_steps` is that routine written out instruction by instruction.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace edm {
namespace f32 {

__device__ __forceinline__ float mul(float a, float b) { float r; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float add(float a, float b) { float r; asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float sub(float a, float b) { float r; asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fma(float a, float b, float c) { float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float fma_rm(float a, float b, float c) { float r; asm("fma.rm.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }   // round toward -inf
__device__ __forceinline__ float div(float a, float b) { float r; asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }             // IEEE division, correctly rounded
__device__ __forceinline__ float div_full(float a, float b) { float r; asm("div.full.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }      // approximate division (2 ulp), LLVM's choice under -nvptx-prec-divf32=1
__device__ __forceinline__ float rcp_approx(float a) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }                    // special-function-unit reciprocal (1 ulp), LLVM's 1/x under -nvptx-prec-divf32=0
__device__ __forceinline__ float rcp(float a) { float r; asm("rcp.rn.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }                               // IEEE reciprocal, correctly rounded: LLVM's 1/x under -nvptx-prec-divf32=1 and =2
__device__ __forceinline__ float ex2_approx(float a) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }                    // one special-function-unit 2^x
__device__ __forceinline__ float rsqrt_approx(float a) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }                // one special-function-unit 1/sqrt(x): libdevice __nv_rsqrtf in flush-to-zero mode, the routine TensorFlow's Rsqrt kernel calls
__device__ __forceinline__ float sat(float a) { float r; asm("cvt.ftz.sat.f32.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }                           // clamp to [0, 1], NaN -> +0
__device__ __forceinline__ float add_nonflushing(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }     // subnormals kept: CUB's warp-sum step (cub/warp/specializations/warp_reduce_shfl.cuh:188-210, "add.f32")
__device__ __forceinline__ float neg(float a) { return __int_as_float(__float_as_int(a) ^ 0x80000000); }                                                // sign flip, exact for every input including NaN

// div(e, s) for the terms of a softmax normalisation — e = expf(l - max) in {+0} U [2^-126, 1], s >= 1 a finite sum of such terms (NaN in either
// when the logits hold one) — with the same result bits as `div` for every such (e, s), reached without the division's slow path: ptxas
// expands div.rn.f32 into a reciprocal-and-fma sequence behind an operand check that sends a zero or subnormal numerator, and one within a
// few tens of binades of the subnormal range, to a long divergent subroutine — and a softmax's losing terms are exactly that (Enformer's
// pooling pairs: e is +0 for over half of them and below 2^-100 for several percent more). Case by case:
//   s NaN                       the instruction on (e, s).
//   e ±0 or subnormal           ±0 with e's sign: the instruction flushes a subnormal e to that zero, and ±0 / s = ±0 for a number s >= 1.
//   0 < e < 2^-64               q = RN(e·2^64 / s): e·2^64 is exact (a power-of-two scaling within the normal range) and clear of the check.
//                               If q > 2^-62, q·2^-64 is a normal number formed exactly and equals RN(e / s) — rounding to nearest commutes
//                               with a power-of-two scaling while neither side is subnormal. If q <= 2^-62 the quotient is at or below
//                               2^-126, where rounding meets the flush to zero: the instruction itself is issued on (e, s).
//   e >= 2^-64, e < 0, e NaN    the instruction on (e, s).
// `volatile` keeps the rarely needed instruction under its branch (issued for the lanes that take it, not speculated for all).
__device__ __forceinline__ float div_issued(float a, float b) { float r; asm volatile("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float div_softmax(float e, float s) {
  const bool s_num = (s == s);
  const bool e_zero = ((__float_as_uint(e) & 0x7f800000u) == 0u) & s_num;
  const bool e_tiny = (e > 0.0f) & (e < 0x1p-64f) & !e_zero & s_num;
  const float q = div(e_tiny ? mul(e, 0x1p64f) : (e_zero ? 1.0f : e), s);                    // a zero lane divides 1 by s and discards the quotient
  float w = q;
  if (e_tiny) w = (q <= 0x1p-62f) ? div_issued(e, s) : mul(q, 0x1p-64f);
  return e_zero ? __uint_as_float(__float_as_uint(e) & 0x80000000u) : w;
}

// libdevice __nv_expf as this toolkit links it (CUDA's expf; --ftz=true selects the flush-to-zero path).
__device__ __forceinline__ float expf_libdevice(float a) { return ::expf(a); }

// libdevice 12.3 __nv_expf, flush-to-zero path, instruction by instruction (constants are the routine's own bit patterns):
//   t = fma.rn(a, K, 0.5) with K = log2(e)/252 ; t = saturate(t) ; j = fma.rm(t, 252, 12582913) ; n = j - 12583039 (an integer in [-126, 126])
//   r = fma.rn(a, LOG2E_HI, -n) ; r = fma.rn(a, LOG2E_LO, r) ; p = ex2.approx(r) ; result = p * 2^n, where 2^n is built from j's low mantissa bits.
#ifndef EDM_EXPF_K_BITS
#define EDM_EXPF_K_BITS 0x3BBB989D
#endif
__device__ __forceinline__ float expf_steps_scale(float a, float* two_n) {
  const float K = __int_as_float(EDM_EXPF_K_BITS);            // log2(e) / 252
  const float LOG2E_HI = __int_as_float(0x3FB8AA3B);          // 1.44269502f
  const float LOG2E_LO = __int_as_float(0x32A57060);          // 1.92596299e-8f
  float t = sat(fma(a, K, 0.5f));
  float j = fma_rm(t, 252.0f, __int_as_float(0x4B400001));   // 12582913.0f = 1.5 * 2^23 + 1
  float n = sub(j, __int_as_float(0x4B40007F));               // 12583039.0f = 12582913 + 126
  float r = fma(a, LOG2E_HI, neg(n));
  r = fma(a, LOG2E_LO, r);
  *two_n = __int_as_float(__float_as_int(j) << 23);           // 2^n: j's low 9 mantissa bits are n + 127
  return ex2_approx(r);                                       // p; exp(a) = p * 2^n
}
__device__ __forceinline__ float expf_steps(float a) { float s; float p = expf_steps_scale(a, &s); return mul(p, s); }

}  // namespace f32
}  // namespace edm
