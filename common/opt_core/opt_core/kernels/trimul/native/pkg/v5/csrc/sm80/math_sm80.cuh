// math_sm80.cuh -- device math of the sm_80 (Ampere) member of the triangle-multiplication kernels, written so the pieces that do not
// depend on the instruction set (LayerNorm statement, gate epilogue, rounding points, plane addressing) can be hoisted into a shared
// header for every architecture member.  No CUTLASS, no libtorch; PTX for cp.async / ldmatrix / mma.sync only.
//
// Numerics statement (the "same-class" contract = the rounding sequence of the reference Triton kernels fpf_trimul_v4 4.x):
//   LN(x)      : fp32 statistics per row over K columns (mean, then centred biased variance) with the reference kernels' reduction
//                trees (sequential leaves of 8 | 4 contiguous values, butterfly largest-offset-first), rstd = rsqrt.approx(var + eps),
//                y = ((x - mean) * rstd) * gamma + beta in fp32 (one fma for the affine), rounded ONCE to bf16 (round-to-nearest-even);
//                a bf16 input enters the statistics exactly (bf16 -> fp32 is exact), an fp32 input enters natively (no pre-rounding).
//   GEMMs      : bf16 x bf16 -> fp32 accumulate (mma.sync m16n8k16), K in 16-steps, fp32 accumulators never rounded before the gate.
//   gate       : o = sigmoid(g) * p (* mask) on the fp32 accumulators, sigmoid(g) = 1 / (1 + exp(-g)) with ex2.approx / rcp.approx class
//                arithmetic, THEN one bf16 rounding (planes, and the K3 update before the residual add).
//   residual   : out = bf16(o) + z in fp32, stored in z's dtype (bf16 z -> bf16 sum = the framework's bf16 add; fp32 z -> fp32 sum).
// Bit-exact variant (template EXACT / LNX; bf16 z; c a multiple of 64): the same bodies with the stock library's LayerNorm summation orders
// (stock_tree0 / stock_tree_frag below); everything else in the statement above is already the stock arithmetic.
// The per-element / per-statistic arithmetic (bf16 rounding, sigmoid, affine, residual adds, approximate instructions) is csrc/common/tmn_math.cuh's;
// this file adds the sm_80 fragment / quad-row layouts, the loaders and the reduction trees that live with those layouts.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>
#include "common/tmn_math.cuh"          // the numerics statement shared by every architecture member (per-element / per-statistic helpers)

#define TM_DEVI __device__ __forceinline__

namespace tm80 {
namespace math = tmn::math;

// ------------------------------------------------------------------------------------------------------------------ primitives
TM_DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
TM_DEVI void cp_async16(uint32_t saddr, const void* g, bool pred) {
  const int n = pred ? 16 : 0;                                   // src-size 0: the 16 destination bytes are zero-filled, nothing is read
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(saddr), "l"(g), "r"(n) : "memory");
}
TM_DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> TM_DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
TM_DEVI void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
TM_DEVI void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// D[16x8 f32] += A[16x16 bf16, row-major fragments] * B[16x8 bf16, col-major fragments]
TM_DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// transpose of an 8x8 b16 matrix held one row-pair word per lane (lane 4*r + q: row r, columns 2q, 2q+1 -- the mma accumulator pair layout):
// afterwards lane 4*r + q holds row r of the TRANSPOSE, i.e. original column r, original rows 2q, 2q+1 (low half = row 2q).
TM_DEVI uint32_t movm_trans(uint32_t a) { uint32_t d; asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a)); return d; }
// bf16 packing / unpacking and the approximate instructions are the shared statement's (aliases keep the member's call sites short)
TM_DEVI uint32_t pack_bf16(float lo, float hi) { return math::pack_bf16_rn(lo, hi); }
TM_DEVI float bf16lo(uint32_t v) { return math::bf16_lo(v); }
TM_DEVI float bf16hi(uint32_t v) { return math::bf16_hi(v); }
// Bulk global 16-B store with a per-kernel cache policy: CS = true -> st.global.cs (streaming / evict-first), else the default (.wb).
template <bool CS> TM_DEVI void st_global_16(void* ptr, const uint4& v) {
  if constexpr (CS) asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n" :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
  else *reinterpret_cast<uint4*>(ptr) = v;
}
template <bool CS> TM_DEVI void st_global_16(void* ptr, const float4& v) {
  if constexpr (CS) asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
  else *reinterpret_cast<float4*>(ptr) = v;
}
TM_DEVI float round_bf16(float v) { return math::round_bf16(v); }
// sigmoid / rsqrt / division forms: the shared statement's (sigmoid(g) = rcp.approx.ftz(1 + ex2.approx.ftz(-g log2 e)), the stock library's
// arithmetic; div_full / ex2_approx_nf for the reference Triton kernels' statistics division).
TM_DEVI float ex2_approx(float t) { return math::ex2_approx_ftz(t); }
TM_DEVI float rcp_approx(float v) { return math::rcp_approx_ftz(v); }
TM_DEVI float div_full(float a, float b) { return math::div_full(a, b); }
TM_DEVI float rsqrt_approx(float v) { return math::rsqrt_approx_ftz(v); }
TM_DEVI float sigmoidf_(float g) { return math::sigmoid(g); }
// A cheaper gate OUTSIDE the acceptance class, kept as a constexpr option for measurement: sigmoid(g) = 0.5 tanh(g/2) + 0.5 with tanh.approx.f32
// (one special-function op instead of two; ~2^-11 relative error = the 'tanh-approx class').  SIG = 0: the stock form above; SIG = 1: this.
TM_DEVI float tanh_approx(float v) { float r; asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(v)); return r; }
template <int SIG> TM_DEVI float sigmoid_t(float g) {
  if constexpr (SIG == 0) return sigmoidf_(g); else return fmaf(0.5f, tanh_approx(__fmul_rn(0.5f, g)), 0.5f);
}
// The reference reduction tree over a row of K contiguous values ("leaves" = sequential sums of 8 (bf16 rows) or 4 (fp32 rows) contiguous
// values, then a butterfly over the leaf index with offsets K/16 (K/8) ... 2, 1, largest first).  A quad-loaded row (lane q = columns
// 32kk + 8q .. +7) holds leaves 4kk + q (bf16) or 8kk + 2q + e (fp32): the large offsets are in-lane over kk, offsets 2 | 1 of the bf16 tree
// (4 | 2 of the fp32 tree) are lane xor 2 | xor 1, and the fp32 tree's offset 1 is the in-lane pair e, applied last.
template <int NL> TM_DEVI float lane_tree(float (&v)[NL]) {           // in-lane butterfly over NL (padded to a power of two with zeros) leaves
  constexpr int P = NL <= 1 ? 1 : NL <= 2 ? 2 : NL <= 4 ? 4 : NL <= 8 ? 8 : 16;
  float t[P];
#pragma unroll
  for (int i = 0; i < P; ++i) t[i] = i < NL ? v[i] : 0.f;
#pragma unroll
  for (int off = P / 2; off >= 1; off >>= 1)
#pragma unroll
    for (int i = 0; i < off; ++i) t[i] = __fadd_rn(t[i], t[i + off]);
  return t[0];
}
TM_DEVI float quad_tree(float v) { v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2)); v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1)); return v; }
// byte offset of 16-B granule g of row r in a tile of ROWB-byte rows (ROWB a multiple of 128): the granule slot is xor-ed with (r & 7) inside
// its aligned group of 8, which makes ldmatrix row reads, cp.async row writes and 16-B fragment row writes shared-memory-bank-conflict free.
template <int ROWB> TM_DEVI uint32_t swz(int r, int g) { return (uint32_t)(r * ROWB + ((g ^ (r & 7)) << 4)); }

// ------------------------------------------------------------------------------------------------------------------ LayerNorm pieces
// Every floating-point operation below is an explicit round-to-nearest fp32 instruction (__fadd_rn / __fsub_rn / __fmul_rn / fmaf), so the
// compiler cannot contract or reorder them: the ORDER is the statement (it reproduces the reference kernels' reduction trees bit for bit).
//
// LayerNorm of ONE row of K elements read straight from global memory by a quad: lane q holds columns [32kk + 8q, 32kk + 8q + 8) for every
// kk < K/32 (16 contiguous bytes of bf16 / 32 of fp32 per lane per 32 columns; a quad reads 64 / 128 contiguous bytes).  Output: K/8 packed
// bf16x2 registers in column order, out[4kk + e] = columns 32kk + 8q + {2e, 2e+1}.  valid == false: the row reads as zeros (LN -> beta).
// bf16 rows: leaves = the K/8 sequential sums of 8 contiguous values (leaf 4kk + q lives in lane q), butterfly offsets K/16 .. 4 in-lane over
// kk, then lane xor 2, lane xor 1.  Variance leaves: d0*d0 then fma(d, d, acc) for the next 7.
// ---- the stock library's LayerNorm summation orders (cuequivariance_torch 0.11.1 triangle_multiplicative_update, bf16 input) for the bit-exact
// variant, any K that is a multiple of 64 (statement transcribed from csrc/tmn_kernels.cuh stock_row_sum / ln_stock, where it is bit-identical
// to the library at c 64 / 128 / 256 / 384 on sm_90a; the K = 256 orders were first established by trimul_tx 1.2):
//   the K channels are K/64 chunks of 64 accumulated elementwise first, A[c'] = ((x[c'] + x[c'+64]) + x[c'+128]) + ... (c' = channel mod 64; for the
//   centred squares: acc = d0*d0, then fma(d, d, acc) for the later chunks);
//   TREE 0 (token-major kernel = LN_in, any N): t_w = ((A[8w] + A[8w+1]) + ...) + A[8w+7] (w = 0..7), then a butterfly over w with xor 4, 2, 1;
//   TREE 1 (transposing kernel = LN_out over channel-major planes, plane row length N % 4 == 0): t[c'] = A[c'] + A[c'+32] (c' < 32), then a
//   butterfly over c' with xor 2, 1, 16, 8, 4;  TREE 2 (transposing kernel, N % 4 != 0): t_v = ((A[v] + A[v+4]) + A[v+8]) + ... + A[v+60]
//   (v = 0..3, stride 4), then S = (t0 + t2) + (t1 + t3);   mean = S * rn(1/K); rstd = rsqrt.approx.ftz(fma(S2, rn(1/K), eps)) (ONE rounding: the
//   library's divide-by-constant is a multiply by the rounded reciprocal contracted with the eps add; identical to the two-step form at power-of-two
//   K, the only bit-identical form at K = 384); y = fma((x - mean) * rstd, gamma, beta) -> bf16 (rn).
// TREE 0 in the quad row layout (lane q holds columns 32kk + 8q .. +7): kk = 2 chunk + h, c' in [32h + 8q, 32h + 8q + 8), so the chunk accumulation
// and the sequential run w = 4h + q are lane-local; xor 4 is lane-local (h), xor 2 | 1 are lane shuffles.
template <int K, int PASS>                // PASS 0: sum of x; 1: sum of (x - mean)^2 with the contracted chunk accumulation
TM_DEVI float stock_tree0(const uint32_t (&raw)[K / 8], float mean) {
  static_assert(K % 64 == 0, "the stock-order trees are written for widths that are multiples of 64");
  constexpr int NCH = K / 64;
  float t[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float A[8];
#pragma unroll
    for (int sl = 0; sl < 8; ++sl) {
      const int r = sl >> 1, e = sl & 1;
      auto xv = [&](int c) { const uint32_t w = raw[4 * (2 * c + h) + r]; return e ? bf16hi(w) : bf16lo(w); };
      if (PASS == 0) {
        float acc = xv(0);
#pragma unroll
        for (int c = 1; c < NCH; ++c) acc = __fadd_rn(acc, xv(c));
        A[sl] = acc;
      } else {
        float d = __fsub_rn(xv(0), mean), acc = __fmul_rn(d, d);
#pragma unroll
        for (int c = 1; c < NCH; ++c) { d = __fsub_rn(xv(c), mean); acc = __fmaf_rn(d, d, acc); }
        A[sl] = acc;
      }
    }
    float a = __fadd_rn(A[0], A[1]);
#pragma unroll
    for (int sl = 2; sl < 8; ++sl) a = __fadd_rn(a, A[sl]);
    t[h] = a;
  }
  float sum = __fadd_rn(t[0], t[1]);                                              // w xor 4
  sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));                    // w xor 2
  sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));                    // w xor 1
  return sum;
}
// TREE 1 / TREE 2 on mma A fragments f[K/16][4] of one m16 tile (register m of k-step ks = channel 16 ks + {2q, 2q+1, 8+2q, 9+2q}[m]; ROW 0 = regs 0/2,
// ROW 1 = regs 1/3): c' mod 16 is fixed per register slot, so the chunk accumulation is thread-local (k-steps ks4 + 4 c, c < K/64).
template <int ROW, int KS> TM_DEVI float frag_val(const uint32_t (&f)[KS][4], int ks, int m) {
  const uint32_t r = f[ks][ROW + (m >> 1) * 2];
  return (m & 1) ? bf16hi(r) : bf16lo(r);
}
template <int TREE, int ROW, int PASS, int KS>
TM_DEVI float stock_tree_frag(const uint32_t (&f)[KS][4], float mean, int lane) {
  static_assert(TREE == 1 || TREE == 2, "fragment trees: 1 (row length % 4 == 0) or 2");
  static_assert(KS % 4 == 0, "the stock-order trees are written for widths that are multiples of 64");
  constexpr int NCH = KS / 4;
  float A[4][4];
#pragma unroll
  for (int ks4 = 0; ks4 < 4; ++ks4) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      if (PASS == 0) {
        float acc = frag_val<ROW>(f, ks4, m);
#pragma unroll
        for (int c = 1; c < NCH; ++c) acc = __fadd_rn(acc, frag_val<ROW>(f, ks4 + 4 * c, m));
        A[ks4][m] = acc;
      } else {
        float d = __fsub_rn(frag_val<ROW>(f, ks4, m), mean), acc = __fmul_rn(d, d);
#pragma unroll
        for (int c = 1; c < NCH; ++c) { d = __fsub_rn(frag_val<ROW>(f, ks4 + 4 * c, m), mean); acc = __fmaf_rn(d, d, acc); }
        A[ks4][m] = acc;
      }
    }
  }
  const int q = lane & 3;
  if (TREE == 1) {
    float u[2][4];
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2) {
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float t = __fadd_rn(A[k2][m], A[k2 + 2][m]);                               // c' + 32: k-steps ks4 and ks4 + 2
        u[k2][m] = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, 1));                    // c' xor 2 lives in lane ^ 1
      }
    }
    const float v00 = __fadd_rn(u[0][0], u[0][1]), v01 = __fadd_rn(u[0][2], u[0][3]);   // c' xor 1: lo + hi
    const float v10 = __fadd_rn(u[1][0], u[1][1]), v11 = __fadd_rn(u[1][2], u[1][3]);
    const float w0 = __fadd_rn(v00, v10), w1 = __fadd_rn(v01, v11);                     // c' xor 16: ks4 0 + 1
    const float zz = __fadd_rn(w0, w1);                                                   // c' xor 8: register pair 0/1 + 2/3
    (void)q;
    return __fadd_rn(zz, __shfl_xor_sync(0xffffffffu, zz, 2));                           // c' xor 4 lives in lane ^ 2
  } else {
    // run v = c' & 3 visits c' = v + 4 j, j = 0..15: term j lives in lane-half (q >> 1) == (j & 1) (register m = 0 / 2 for j % 4 in {0,1} / {2,3} of
    // k-step ks4 = j >> 2); this lane carries two runs: v = 2 (q & 1) (registers m = 0, 2) and v + 1 (registers m = 1, 3)
    const int half = q >> 1;
    float sa = A[0][0], sb = A[0][1];
#pragma unroll
    for (int j = 1; j < 16; ++j) {
      const int ks4 = j >> 2, m0 = (j & 2) ? 2 : 0;
      const float va = __shfl_xor_sync(0xffffffffu, sa, 2), vb = __shfl_xor_sync(0xffffffffu, sb, 2);
      const float na = __fadd_rn(va, A[ks4][m0]), nb = __fadd_rn(vb, A[ks4][m0 + 1]);
      const bool mine = half == (j & 1);
      sa = mine ? na : sa; sb = mine ? nb : sb;
    }
    const float ua = __fadd_rn(sa, __shfl_xor_sync(0xffffffffu, sa, 1)), ub = __fadd_rn(sb, __shfl_xor_sync(0xffffffffu, sb, 1));   // v xor 2 pairs: (t0 + t2), (t1 + t3)
    const float S = __fadd_rn(ua, ub);                                                    // (t0 + t2) + (t1 + t3), complete in lane-half 1
    return __shfl_sync(0xffffffffu, S, lane | 2);
  }
}

template <int K> struct RowRaw { uint32_t w[K / 8]; };                 // a bf16 row's 16-B pieces held by one lane (columns 32kk + 8q .. +7)
template <int K> struct RowRawF { float w[K / 4]; };                   // an fp32 row's pieces
template <int K>
TM_DEVI void row_fetch(const __nv_bfloat16* __restrict__ row, bool valid, int q, RowRaw<K>& r) {
#pragma unroll
  for (int kk = 0; kk < K / 32; ++kk) {
    uint4 v = valid ? __ldg(reinterpret_cast<const uint4*>(row + 32 * kk + 8 * q)) : make_uint4(0u, 0u, 0u, 0u);
    r.w[4 * kk] = v.x; r.w[4 * kk + 1] = v.y; r.w[4 * kk + 2] = v.z; r.w[4 * kk + 3] = v.w;
  }
}
template <int K>
TM_DEVI void row_fetch(const float* __restrict__ row, bool valid, int q, RowRawF<K>& r) {
#pragma unroll
  for (int kk = 0; kk < K / 32; ++kk) {
    const float4 a = valid ? __ldg(reinterpret_cast<const float4*>(row + 32 * kk + 8 * q)) : make_float4(0.f, 0.f, 0.f, 0.f);
    const float4 b = valid ? __ldg(reinterpret_cast<const float4*>(row + 32 * kk + 8 * q + 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
    r.w[8 * kk + 0] = a.x; r.w[8 * kk + 1] = a.y; r.w[8 * kk + 2] = a.z; r.w[8 * kk + 3] = a.w;
    r.w[8 * kk + 4] = b.x; r.w[8 * kk + 5] = b.y; r.w[8 * kk + 6] = b.z; r.w[8 * kk + 7] = b.w;
  }
}
template <typename ZT, int K> struct RowRawT { using type = RowRaw<K>; };
template <int K> struct RowRawT<float, K> { using type = RowRawF<K>; };

template <int K, bool EXACT = false>
TM_DEVI void row_ln(const RowRaw<K>& rr, const float* __restrict__ gamma, const float* __restrict__ beta, float eps, int q, uint32_t (&out)[K / 8]) {
  constexpr int KK = K / 32;
  const uint32_t (&raw)[K / 8] = rr.w;
  if constexpr (EXACT) {                                                            // the stock library's order (TREE 0), K a multiple of 64
    const float mean = math::ln_mean(stock_tree0<K, 0>(rr.w, 0.f), 1.f / K);
    const float rstd = rsqrt_approx(__fmaf_rn(stock_tree0<K, 1>(rr.w, mean), 1.f / K, eps));
    auto affx = [&](float v, float g, float bb) { return math::ln_affine(v, mean, rstd, g, bb); };
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      const int c0 = 32 * kk + 8 * q;
      const float4 g0 = __ldg(reinterpret_cast<const float4*>(gamma + c0)), g1 = __ldg(reinterpret_cast<const float4*>(gamma + c0 + 4));
      const float4 b0 = __ldg(reinterpret_cast<const float4*>(beta + c0)), b1 = __ldg(reinterpret_cast<const float4*>(beta + c0 + 4));
      out[4 * kk + 0] = pack_bf16(affx(bf16lo(raw[4 * kk + 0]), g0.x, b0.x), affx(bf16hi(raw[4 * kk + 0]), g0.y, b0.y));
      out[4 * kk + 1] = pack_bf16(affx(bf16lo(raw[4 * kk + 1]), g0.z, b0.z), affx(bf16hi(raw[4 * kk + 1]), g0.w, b0.w));
      out[4 * kk + 2] = pack_bf16(affx(bf16lo(raw[4 * kk + 2]), g1.x, b1.x), affx(bf16hi(raw[4 * kk + 2]), g1.y, b1.y));
      out[4 * kk + 3] = pack_bf16(affx(bf16lo(raw[4 * kk + 3]), g1.z, b1.z), affx(bf16hi(raw[4 * kk + 3]), g1.w, b1.w));
    }
    return;
  }
  float leaf[KK];
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
    float a = __fadd_rn(bf16lo(raw[4 * kk]), bf16hi(raw[4 * kk]));
#pragma unroll
    for (int r = 1; r < 4; ++r) { a = __fadd_rn(a, bf16lo(raw[4 * kk + r])); a = __fadd_rn(a, bf16hi(raw[4 * kk + r])); }
    leaf[kk] = a;
  }
  const float mean = div_full(quad_tree(lane_tree<KK>(leaf)), (float)K);
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
    float d = __fsub_rn(bf16lo(raw[4 * kk]), mean), a = __fmul_rn(d, d);
    d = __fsub_rn(bf16hi(raw[4 * kk]), mean); a = fmaf(d, d, a);
#pragma unroll
    for (int r = 1; r < 4; ++r) { d = __fsub_rn(bf16lo(raw[4 * kk + r]), mean); a = fmaf(d, d, a); d = __fsub_rn(bf16hi(raw[4 * kk + r]), mean); a = fmaf(d, d, a); }
    leaf[kk] = a;
  }
  const float rstd = rsqrt_approx(__fadd_rn(div_full(quad_tree(lane_tree<KK>(leaf)), (float)K), eps));
  auto aff = [&](float v, float g, float bb) { return math::ln_affine(v, mean, rstd, g, bb); };
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
    const int c0 = 32 * kk + 8 * q;
    const float4 g0 = __ldg(reinterpret_cast<const float4*>(gamma + c0)), g1 = __ldg(reinterpret_cast<const float4*>(gamma + c0 + 4));
    const float4 b0 = __ldg(reinterpret_cast<const float4*>(beta + c0)), b1 = __ldg(reinterpret_cast<const float4*>(beta + c0 + 4));
    out[4 * kk + 0] = pack_bf16(aff(bf16lo(raw[4 * kk + 0]), g0.x, b0.x), aff(bf16hi(raw[4 * kk + 0]), g0.y, b0.y));
    out[4 * kk + 1] = pack_bf16(aff(bf16lo(raw[4 * kk + 1]), g0.z, b0.z), aff(bf16hi(raw[4 * kk + 1]), g0.w, b0.w));
    out[4 * kk + 2] = pack_bf16(aff(bf16lo(raw[4 * kk + 2]), g1.x, b1.x), aff(bf16hi(raw[4 * kk + 2]), g1.y, b1.y));
    out[4 * kk + 3] = pack_bf16(aff(bf16lo(raw[4 * kk + 3]), g1.z, b1.z), aff(bf16hi(raw[4 * kk + 3]), g1.w, b1.w));
  }
}
// fp32 rows (the fp32-resident pair tensor read natively): leaves = sequential sums of 4 contiguous values (leaf 8kk + 2q + e in lane q) for
// K <= 128 (K/4 <= 32 leaves); for K = 256 a leaf t < 32 continues sequentially with the 4 values of column group t + 32 (the second register
// repetition of a 32-lane row); butterfly offsets: in-lane over kk, lane xor 2, lane xor 1, and the pair e last.
template <int K, bool EXACT = false>
TM_DEVI void row_ln(const RowRawF<K>& rr, const float* __restrict__ gamma, const float* __restrict__ beta, float eps, int q, uint32_t (&out)[K / 8]) {
  static_assert(!EXACT, "the bit-exact statement reads a bf16 pair tensor (the stock op casts an fp32 one first)");
  constexpr int KK = K / 32;
  constexpr int NLK = (K <= 128 || K % 256 != 0) ? KK : KK / 2;   // in-lane leaf slots per e after folding repetitions (K = 256: 4)
  const float (&raw)[K / 4] = rr.w;                              // the fp32 row enters the statistics and the affine natively
  auto seq4 = [&](int i0) { return __fadd_rn(__fadd_rn(__fadd_rn(raw[i0], raw[i0 + 1]), raw[i0 + 2]), raw[i0 + 3]); };
  auto seq4c = [&](float a, int i0) { return __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(a, raw[i0]), raw[i0 + 1]), raw[i0 + 2]), raw[i0 + 3]); };
  float l0[NLK], l1[NLK];
#pragma unroll
  for (int kk = 0; kk < NLK; ++kk) {
    l0[kk] = seq4(8 * kk); l1[kk] = seq4(8 * kk + 4);
#pragma unroll
    for (int rep = 1; rep < KK / NLK; ++rep) { l0[kk] = seq4c(l0[kk], 8 * (kk + rep * NLK)); l1[kk] = seq4c(l1[kk], 8 * (kk + rep * NLK) + 4); }
  }
  const float mean = div_full(__fadd_rn(quad_tree(lane_tree<NLK>(l0)), quad_tree(lane_tree<NLK>(l1))), (float)K);
  auto sq4 = [&](int i0) { float d = __fsub_rn(raw[i0], mean), a = __fmul_rn(d, d);
                           d = __fsub_rn(raw[i0 + 1], mean); a = fmaf(d, d, a); d = __fsub_rn(raw[i0 + 2], mean); a = fmaf(d, d, a); d = __fsub_rn(raw[i0 + 3], mean); return fmaf(d, d, a); };
  auto sq4c = [&](float a, int i0) { float d = __fsub_rn(raw[i0], mean); a = fmaf(d, d, a); d = __fsub_rn(raw[i0 + 1], mean); a = fmaf(d, d, a);
                                     d = __fsub_rn(raw[i0 + 2], mean); a = fmaf(d, d, a); d = __fsub_rn(raw[i0 + 3], mean); return fmaf(d, d, a); };
#pragma unroll
  for (int kk = 0; kk < NLK; ++kk) {
    l0[kk] = sq4(8 * kk); l1[kk] = sq4(8 * kk + 4);
#pragma unroll
    for (int rep = 1; rep < KK / NLK; ++rep) { l0[kk] = sq4c(l0[kk], 8 * (kk + rep * NLK)); l1[kk] = sq4c(l1[kk], 8 * (kk + rep * NLK) + 4); }
  }
  const float rstd = rsqrt_approx(__fadd_rn(div_full(__fadd_rn(quad_tree(lane_tree<NLK>(l0)), quad_tree(lane_tree<NLK>(l1))), (float)K), eps));
  auto aff = [&](float v, float g, float bb) { return math::ln_affine(v, mean, rstd, g, bb); };
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
    const int c0 = 32 * kk + 8 * q;
    const float4 g0 = __ldg(reinterpret_cast<const float4*>(gamma + c0)), g1 = __ldg(reinterpret_cast<const float4*>(gamma + c0 + 4));
    const float4 b0 = __ldg(reinterpret_cast<const float4*>(beta + c0)), b1 = __ldg(reinterpret_cast<const float4*>(beta + c0 + 4));
    out[4 * kk + 0] = pack_bf16(aff(raw[8 * kk + 0], g0.x, b0.x), aff(raw[8 * kk + 1], g0.y, b0.y));
    out[4 * kk + 1] = pack_bf16(aff(raw[8 * kk + 2], g0.z, b0.z), aff(raw[8 * kk + 3], g0.w, b0.w));
    out[4 * kk + 2] = pack_bf16(aff(raw[8 * kk + 4], g1.x, b1.x), aff(raw[8 * kk + 5], g1.y, b1.y));
    out[4 * kk + 3] = pack_bf16(aff(raw[8 * kk + 6], g1.z, b1.z), aff(raw[8 * kk + 7], g1.w, b1.w));
  }
}

template <typename ZT, int K, bool EXACT = false>
TM_DEVI void ln_row_load(const ZT* __restrict__ row, bool valid, const float* __restrict__ gamma, const float* __restrict__ beta, float eps, int q,
                         uint32_t (&out)[K / 8]) {
  typename RowRawT<ZT, K>::type rr;
  row_fetch<K>(row, valid, q, rr);
  row_ln<K, EXACT>(rr, gamma, beta, eps, q, out);
}

// Scatter a quad-loaded LN'd row (column order, ln_row_load: lane q holds columns 32kk+8q+{0..7} as 4 words) into the mma A-fragment slots
// of row-half h (0: row gid, 1: row gid + 8) of an m16 tile in the STANDARD k order (k-step ks covers columns 16ks..16ks+15; reg h = columns
// 16ks+2q,+1, reg 2+h = 16ks+8+2q,+1).  Standard order wants lane q to hold word q of each of the quad's 4 lanes = a 4x4 transpose of
// (lane, word) inside the quad: two shuffle rounds (xor 2, xor 1), 4 shuffles per 32 columns.  Keeping the tensor-core k grouping identical
// to a natural-order GEMM keeps the fp32 accumulation sequence identical to the reference kernels'.
TM_DEVI void quad_transpose4(uint32_t (&w)[4], int q) {
  const bool hiq = (q & 2) != 0, odd = (q & 1) != 0;
  uint32_t x = __shfl_xor_sync(0xffffffffu, hiq ? w[0] : w[2], 2), y = __shfl_xor_sync(0xffffffffu, hiq ? w[1] : w[3], 2);
  if (hiq) { w[0] = x; w[1] = y; } else { w[2] = x; w[3] = y; }
  x = __shfl_xor_sync(0xffffffffu, odd ? w[0] : w[1], 1); y = __shfl_xor_sync(0xffffffffu, odd ? w[2] : w[3], 1);
  if (odd) { w[0] = x; w[2] = y; } else { w[1] = x; w[3] = y; }
}
template <int K>
TM_DEVI void scatter_row_frags(const uint32_t (&row)[K / 8], uint32_t (&f)[K / 16][4], int h, int q) {
#pragma unroll
  for (int kk = 0; kk < K / 32; ++kk) {
    uint32_t w[4] = {row[4 * kk + 0], row[4 * kk + 1], row[4 * kk + 2], row[4 * kk + 3]};
    quad_transpose4(w, q);
    f[2 * kk][h] = w[0]; f[2 * kk][2 + h] = w[1]; f[2 * kk + 1][h] = w[2]; f[2 * kk + 1][2 + h] = w[3];
  }
}
// Store a quad-loaded LN'd row into a [rows][K] bf16 shared tile (row pitch K*2 bytes, swizzled): 16 B per lane per 32 columns, standard order.
template <int K>
TM_DEVI void store_row_smem(const uint32_t (&row)[K / 8], uint8_t* tile, int r, int q) {
#pragma unroll
  for (int kk = 0; kk < K / 32; ++kk)
    *reinterpret_cast<uint4*>(tile + swz<K * 2>(r, 4 * kk + q)) = make_uint4(row[4 * kk], row[4 * kk + 1], row[4 * kk + 2], row[4 * kk + 3]);
}

// LayerNorm in place on mma A fragments of one m16 tile in the STANDARD k order (reg 0: row gid, cols 16ks+2q,+1 | reg 1: row gid+8 |
// reg 2: row gid, cols 16ks+8+2q,+1 | reg 3: row gid+8, +8) -- the layout ldmatrix / ldmatrix.trans deliver; used for LN_out over the K
// channels of the X tile.  Reference tree (a [K][tokens] tile spread over 16 thread-rows tr, each summing channels tr + 16*rep sequentially):
// leaves = the four channel residues this thread holds (2q, 2q+1, 2q+8, 2q+9 mod 16) summed over ks; then tr^2 (lane xor 1), tr^1 (the pair),
// tr^8 (the +8 pair), tr^4 (lane xor 2).
template <int K, int LNX = 0>
TM_DEVI void ln_frags(uint32_t (&f)[K / 16][4], const float* __restrict__ gamma, const float* __restrict__ beta, int q, float eps, int lane = 0) {
  constexpr int KS = K / 16;
  if constexpr (LNX != 0) {                                                         // the stock library's order (TREE 1 | 2), K a multiple of 64
    const float mA = math::ln_mean(stock_tree_frag<LNX, 0, 0>(f, 0.f, lane), 1.f / K), mB = math::ln_mean(stock_tree_frag<LNX, 1, 0>(f, 0.f, lane), 1.f / K);
    const float rA = rsqrt_approx(__fmaf_rn(stock_tree_frag<LNX, 0, 1>(f, mA, lane), 1.f / K, eps));
    const float rB = rsqrt_approx(__fmaf_rn(stock_tree_frag<LNX, 1, 1>(f, mB, lane), 1.f / K, eps));
    auto y = [&](float v, float mean, float r, float g, float bb) { return math::ln_affine(v, mean, r, g, bb); };
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int k0 = 16 * ks + 2 * q;
      const float2 g0 = __ldg(reinterpret_cast<const float2*>(gamma + k0)), b0 = __ldg(reinterpret_cast<const float2*>(beta + k0));
      const float2 g1 = __ldg(reinterpret_cast<const float2*>(gamma + k0 + 8)), b1 = __ldg(reinterpret_cast<const float2*>(beta + k0 + 8));
      f[ks][0] = pack_bf16(y(bf16lo(f[ks][0]), mA, rA, g0.x, b0.x), y(bf16hi(f[ks][0]), mA, rA, g0.y, b0.y));
      f[ks][1] = pack_bf16(y(bf16lo(f[ks][1]), mB, rB, g0.x, b0.x), y(bf16hi(f[ks][1]), mB, rB, g0.y, b0.y));
      f[ks][2] = pack_bf16(y(bf16lo(f[ks][2]), mA, rA, g1.x, b1.x), y(bf16hi(f[ks][2]), mA, rA, g1.y, b1.y));
      f[ks][3] = pack_bf16(y(bf16lo(f[ks][3]), mB, rB, g1.x, b1.x), y(bf16hi(f[ks][3]), mB, rB, g1.y, b1.y));
    }
    return;
  }
  auto tree16 = [&](float e0, float e1, float e2, float e3) {
    e0 = __fadd_rn(e0, __shfl_xor_sync(0xffffffffu, e0, 1)); e1 = __fadd_rn(e1, __shfl_xor_sync(0xffffffffu, e1, 1));
    e2 = __fadd_rn(e2, __shfl_xor_sync(0xffffffffu, e2, 1)); e3 = __fadd_rn(e3, __shfl_xor_sync(0xffffffffu, e3, 1));
    const float m = __fadd_rn(__fadd_rn(e0, e1), __fadd_rn(e2, e3));
    return __fadd_rn(m, __shfl_xor_sync(0xffffffffu, m, 2));
  };
  float a0 = bf16lo(f[0][0]), a1 = bf16hi(f[0][0]), a2 = bf16lo(f[0][2]), a3 = bf16hi(f[0][2]);
  float b0 = bf16lo(f[0][1]), b1 = bf16hi(f[0][1]), b2 = bf16lo(f[0][3]), b3 = bf16hi(f[0][3]);
#pragma unroll
  for (int ks = 1; ks < KS; ++ks) {
    a0 = __fadd_rn(a0, bf16lo(f[ks][0])); a1 = __fadd_rn(a1, bf16hi(f[ks][0])); a2 = __fadd_rn(a2, bf16lo(f[ks][2])); a3 = __fadd_rn(a3, bf16hi(f[ks][2]));
    b0 = __fadd_rn(b0, bf16lo(f[ks][1])); b1 = __fadd_rn(b1, bf16hi(f[ks][1])); b2 = __fadd_rn(b2, bf16lo(f[ks][3])); b3 = __fadd_rn(b3, bf16hi(f[ks][3]));
  }
  const float mA = div_full(tree16(a0, a1, a2, a3), (float)K), mB = div_full(tree16(b0, b1, b2, b3), (float)K);
  {
    float d;
    d = __fsub_rn(bf16lo(f[0][0]), mA); a0 = __fmul_rn(d, d); d = __fsub_rn(bf16hi(f[0][0]), mA); a1 = __fmul_rn(d, d);
    d = __fsub_rn(bf16lo(f[0][2]), mA); a2 = __fmul_rn(d, d); d = __fsub_rn(bf16hi(f[0][2]), mA); a3 = __fmul_rn(d, d);
    d = __fsub_rn(bf16lo(f[0][1]), mB); b0 = __fmul_rn(d, d); d = __fsub_rn(bf16hi(f[0][1]), mB); b1 = __fmul_rn(d, d);
    d = __fsub_rn(bf16lo(f[0][3]), mB); b2 = __fmul_rn(d, d); d = __fsub_rn(bf16hi(f[0][3]), mB); b3 = __fmul_rn(d, d);
#pragma unroll
    for (int ks = 1; ks < KS; ++ks) {
      d = __fsub_rn(bf16lo(f[ks][0]), mA); a0 = fmaf(d, d, a0); d = __fsub_rn(bf16hi(f[ks][0]), mA); a1 = fmaf(d, d, a1);
      d = __fsub_rn(bf16lo(f[ks][2]), mA); a2 = fmaf(d, d, a2); d = __fsub_rn(bf16hi(f[ks][2]), mA); a3 = fmaf(d, d, a3);
      d = __fsub_rn(bf16lo(f[ks][1]), mB); b0 = fmaf(d, d, b0); d = __fsub_rn(bf16hi(f[ks][1]), mB); b1 = fmaf(d, d, b1);
      d = __fsub_rn(bf16lo(f[ks][3]), mB); b2 = fmaf(d, d, b2); d = __fsub_rn(bf16hi(f[ks][3]), mB); b3 = fmaf(d, d, b3);
    }
  }
  const float rA = rsqrt_approx(__fadd_rn(div_full(tree16(a0, a1, a2, a3), (float)K), eps));
  const float rB = rsqrt_approx(__fadd_rn(div_full(tree16(b0, b1, b2, b3), (float)K), eps));
  auto affA = [&](float v, float g, float bb) { return math::ln_affine(v, mA, rA, g, bb); };
  auto affB = [&](float v, float g, float bb) { return math::ln_affine(v, mB, rB, g, bb); };
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const int k0 = 16 * ks + 2 * q;
    const float2 g0 = __ldg(reinterpret_cast<const float2*>(gamma + k0)), b0v = __ldg(reinterpret_cast<const float2*>(beta + k0));
    const float2 g1 = __ldg(reinterpret_cast<const float2*>(gamma + k0 + 8)), b1v = __ldg(reinterpret_cast<const float2*>(beta + k0 + 8));
    f[ks][0] = pack_bf16(affA(bf16lo(f[ks][0]), g0.x, b0v.x), affA(bf16hi(f[ks][0]), g0.y, b0v.y));
    f[ks][1] = pack_bf16(affB(bf16lo(f[ks][1]), g0.x, b0v.x), affB(bf16hi(f[ks][1]), g0.y, b0v.y));
    f[ks][2] = pack_bf16(affA(bf16lo(f[ks][2]), g1.x, b1v.x), affA(bf16hi(f[ks][2]), g1.y, b1v.y));
    f[ks][3] = pack_bf16(affB(bf16lo(f[ks][3]), g1.x, b1v.x), affB(bf16hi(f[ks][3]), g1.y, b1v.y));
  }
}

// ------------------------------------------------------------------------------------------------------------------ tile loaders (cp.async)
// rows [0, nrows) of a row-major bf16 matrix (row pitch `ld` elements, K columns copied) -> swizzled smem tile with K*2-byte rows.
template <int K, int NT>
TM_DEVI void cp_rows(uint32_t tile, const __nv_bfloat16* __restrict__ src, size_t ld, int nrows, int tid) {
  constexpr int GPR = K / 8;
  for (int i = tid; i < nrows * GPR; i += NT) {
    const int r = i / GPR, g = i % GPR;
    cp_async16(tile + swz<K * 2>(r, g), src + (size_t)r * ld + g * 8, true);
  }
}

// B fragments of two adjacent n8 tiles (rows n0..n0+15 of a [n][K] weight tile with ROWB-byte rows, k-step ks): {b[0],b[1]} feed n-tile
// n0/8, {b[2],b[3]} n-tile n0/8 + 1.
template <int ROWB>
TM_DEVI void load_b16(uint32_t (&b)[4], uint32_t tile, int n0, int ks, int lane) {
  const int n = n0 + (lane & 7) + ((lane >> 4) << 3), g = 2 * ks + ((lane >> 3) & 1);
  ldsm_x4(b, tile + swz<ROWB>(n, g));
}
// A fragments (standard order) of the m16 tile at rows r0..r0+15 of a [rows][K] tile with ROWB-byte rows, k-step ks.
template <int ROWB>
TM_DEVI void load_a16(uint32_t (&a)[4], uint32_t tile, int r0, int ks, int lane) {
  const int r = r0 + (lane & 7) + ((lane >> 3) & 1) * 8, g = 2 * ks + (lane >> 4);
  ldsm_x4(a, tile + swz<ROWB>(r, g));
}
// A fragments (standard order; rows = tokens, k = channels) of the m16 tile at tokens m0..m0+15 (m0 % 8 == 0), channels 16ks..16ks+15 of a
// CHANNEL-major [channels][tokens] bf16 tile (row pitch ROWB bytes = tokens*2), via ldmatrix.trans.
template <int ROWB>
TM_DEVI void load_a16_trans(uint32_t (&a)[4], uint32_t tile, int m0, int ks, int lane) {
  const int mtx = lane >> 3;
  const int ch = 16 * ks + (lane & 7) + 8 * (mtx >> 1), tg = (m0 >> 3) + (mtx & 1);
  ldsm_x4_t(a, tile + swz<ROWB>(ch, tg));
}

}  // namespace tm80
