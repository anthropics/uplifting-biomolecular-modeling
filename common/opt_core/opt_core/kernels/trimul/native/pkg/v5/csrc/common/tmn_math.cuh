// SPDX-License-Identifier: Apache-2.0
// tmn_math.cuh — the NUMERICS STATEMENT of the triangle-multiplication kernels, shared by every architecture member (sm_90a: tmn_kernels.cuh;
// sm_80: csrc/sm80).  Per-element and per-row-statistic helpers only: nothing here depends on a fragment layout, an instruction set beyond
// sm_80, or a tile shape.  Every floating-point operation that is part of the statement is an explicit round-to-nearest fp32 instruction
// (__fadd_rn / __fsub_rn / __fmul_rn / __fmaf_rn) or a named approximate instruction, so the compiler can neither contract nor reorder it.
//
// The statement (class REF = the rounding sequence of cuequivariance_torch 0.11.1 triangle_multiplicative_update, which fpf_trimul_v4 shares
// up to reduction order):
//   LN(x)     fp32 statistics per row: mean = sum(x) * (1/C); var = sum((x - mean)^2) * (1/C) (centred, biased); rstd = rsqrt.approx.ftz(var + eps);
//             y = fma((x - mean) * rstd, gamma, beta) in fp32, rounded ONCE to bf16 (RN-even).  A bf16 input enters exactly (bf16 -> fp32 is exact),
//             an fp32 input enters natively (never pre-rounded).  Reductions: sums of small groups of neighbouring elements, a balanced pairwise
//             tree over the groups a thread holds (tree_sum), then a butterfly across the lanes sharing the row.  (The reference library's own
//             trees, used by the bitwise variant, live with the fragment layout in the architecture files.)
//   GEMMs     bf16 x bf16 -> fp32 accumulate; accumulators are never rounded before the gate.
//   gate      o = sigmoid(g) * p (* mask) on the fp32 accumulators, sigmoid(g) = rcp.approx.ftz(1 + ex2.approx.ftz(-g * log2 e)), then ONE bf16
//             rounding (the a / b planes; the K3 update).  (sigmoid_div: the div.full / ex2.approx form some Triton-compiled kernels lower to; same class.)
//   residual  bf16 z: out = bf16(z + o) (the framework's bf16 add: exact fp32 sum of two bf16 values, rounded once); fp32 z: out = z + o in fp32.
// Class TX (kept for comparison with trimul_tx 1.2, whose bytes it reproduces at c_z = c_hidden = 256): the same statistics with rsqrtf, and the
// affine refactored as y = fma(x, rstd * gamma, fma(-mean * rstd, gamma, beta)) — one more rounding and a cancellation-prone constant term; it is
// measurably looser on rows whose mean is large against their spread and is never the default.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

#ifndef TMN_MATH_DEVI
#define TMN_MATH_DEVI __device__ __forceinline__
#endif

namespace tmn {
namespace math {

enum Cls : int { REF = 0, TX = 1 };

// ---------------------------------------------------------------- bf16 <-> fp32 (RN-even packing; the low half is the first / even element)
TMN_MATH_DEVI uint32_t pack_bf16_rn(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
TMN_MATH_DEVI float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
TMN_MATH_DEVI float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
TMN_MATH_DEVI float round_bf16(float v) { return bf16_lo(pack_bf16_rn(v, 0.f)); }

// ---------------------------------------------------------------- approximate instructions named by the statement
TMN_MATH_DEVI float ex2_approx_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
TMN_MATH_DEVI float rcp_approx_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
TMN_MATH_DEVI float rsqrt_approx_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
TMN_MATH_DEVI float ex2_approx(float x) { float y; asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
TMN_MATH_DEVI float div_full(float a, float b) { float y; asm("div.full.f32 %0, %1, %2;\n" : "=f"(y) : "f"(a), "f"(b)); return y; }

// ---------------------------------------------------------------- sums
// Balanced pairwise sum of n values held by one thread (the statement's in-thread reduction shape: group sums of neighbouring elements first, then
// this tree, then the caller's cross-lane butterfly).  Depth log2(n) instead of n - 1 sequential roundings.
template <int LO, int LEN> struct TreeSum {
  template <int N_> static TMN_MATH_DEVI float run(const float (&v)[N_]) { return __fadd_rn(TreeSum<LO, LEN / 2>::run(v), TreeSum<LO + LEN / 2, LEN - LEN / 2>::run(v)); }
};
template <int LO> struct TreeSum<LO, 1> {
  template <int N_> static TMN_MATH_DEVI float run(const float (&v)[N_]) { return v[LO]; }
};
template <int N_>
TMN_MATH_DEVI float tree_sum(const float (&v)[N_]) { return TreeSum<0, N_>::run(v); }   // compile-time indices only: the values stay in registers

// ---------------------------------------------------------------- LayerNorm: row statistics from the caller's sums, and the per-element affine
// mean of a row from its fp32 sum; rstd from the CENTRED sum of squares (sum over the row of (x - mean)^2) — both sums in the caller's fixed order.
TMN_MATH_DEVI float ln_mean(float row_sum, float inv_c) { return __fmul_rn(row_sum, inv_c); }
template <int CLS = REF>
TMN_MATH_DEVI float ln_rstd(float centred_sq_sum, float inv_c, float eps) {
  const float v = __fadd_rn(__fmul_rn(centred_sq_sum, inv_c), eps);
  return CLS == TX ? rsqrtf(v) : rsqrt_approx_ftz(v);
}
// one term of the centred sum of squares: acc + (x - mean)^2 as sub, then fma (the first term of a leaf may use ln_sq)
TMN_MATH_DEVI float ln_sq(float x, float mean) { const float d = __fsub_rn(x, mean); return __fmul_rn(d, d); }
TMN_MATH_DEVI float ln_sq_acc(float acc, float x, float mean) { const float d = __fsub_rn(x, mean); return __fmaf_rn(d, d, acc); }
// y = LN(x) element in fp32 (round it once with pack_bf16_rn / round_bf16): class REF
TMN_MATH_DEVI float ln_affine(float x, float mean, float rstd, float gamma, float beta) {
  return __fmaf_rn(__fmul_rn(__fsub_rn(x, mean), rstd), gamma, beta);
}
// class TX: y = fma(x, rstd * gamma, fma(-(mean * rstd), gamma, beta))
TMN_MATH_DEVI float ln_affine_tx(float x, float rstd, float mean_rstd, float gamma, float beta) {
  return fmaf(x, rstd * gamma, fmaf(-mean_rstd, gamma, beta));
}

// ---------------------------------------------------------------- gate
TMN_MATH_DEVI float sigmoid(float g) { return rcp_approx_ftz(__fadd_rn(1.f, ex2_approx_ftz(__fmul_rn(-1.4426950408889634f, g)))); }
TMN_MATH_DEVI float sigmoid_div(float g) { return div_full(1.f, __fadd_rn(1.f, ex2_approx(__fmul_rn(-g, 1.4426950408889634f)))); }
// sigmoid(g) * p (* m) on fp32 accumulators; the caller rounds the result once to bf16 (pack_bf16_rn of two neighbours)
TMN_MATH_DEVI float gate(float g, float p) { return __fmul_rn(sigmoid(g), p); }
TMN_MATH_DEVI float gate(float g, float p, float m) { return __fmul_rn(__fmul_rn(sigmoid(g), p), m); }

// ---------------------------------------------------------------- residual
// bf16 z (two packed neighbours) + the bf16-rounded update pair -> bf16 pair: the framework's bf16 add
TMN_MATH_DEVI uint32_t residual_bf16x2(uint32_t z2, uint32_t o2) {
  return pack_bf16_rn(__fadd_rn(bf16_lo(z2), bf16_lo(o2)), __fadd_rn(bf16_hi(z2), bf16_hi(o2)));
}
// fp32 z + the bf16-rounded update, in fp32
TMN_MATH_DEVI float residual_f32(float z, float o_rounded) { return __fadd_rn(z, o_rounded); }

}  // namespace math
}  // namespace tmn
