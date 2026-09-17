// trimul_tx.cu — Protenix-v2 TriangleMultiplication pro/epilogue kernels for H100 (sm_90a), c_z = c_hidden = 256, bf16 activations.
//   K1 (prologue):  ab[ch, i, j] = bf16( sigmoid(LN_in(z)[t] . Wg[ch]) * (LN_in(z)[t] . Wp[ch]) * mask[t] ),  t = (i, j), ch in [0, 512) (a | b planes),
//                   written channel-major into zero-padded planes [512][Np][Np] that one strided-batched cuBLAS GEMM contracts.
//   K3 (epilogue):  out[t, :] = z[t, :] + bf16( sigmoid(LN_in(z)[t] . Wgo^T) * (LN_out(x[:, t]) . Wz^T) )   (x = the contraction result planes).
// Structure (both): persistent CTAs of 3 warpgroups — WG0 producer (TMA bulk-tensor loads into 128B-swizzled smem, mbarrier rings), WG1/WG2 consumers
// (ldmatrix -> LayerNorm in registers -> wgmma m64n64k16 bf16->fp32 with A from registers and the weight block from smem -> fused epilogue, software-
// pipelined one weight block ahead so the epilogue of block b overlaps the MMAs of block b+1).
// Numerics = the cuEq/v4 rounding points: LN output -> bf16; gate*proj(*mask) on fp32 accumulators -> bf16; fp32 LN statistics (two-pass, centred).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "tx_ptx.h"
using namespace tx;

namespace {

constexpr int C = 256;                 // pair channels = hidden channels
constexpr int NTHREADS = 384;          // 1 producer + 2 consumer warpgroups
constexpr int BM = 128;                // tokens per CTA tile (2 consumer warpgroups x 64 rows)
#ifndef TX_NSLOT
#define TX_NSLOT 8
#endif
constexpr int K1_NSLOT = TX_NSLOT;     // weight ring, half-blocks of 16 KB ([64 n][128 k] bf16 as 2 swizzled [64][64] chunks)
constexpr int K1_SMEM_A = 4 * 16384;   // z tile: 4 k-chunks x [128 tok][64 ch] bf16, 128B-swizzled rows
constexpr int K1_SMEM_W = K1_NSLOT * 16384;
constexpr int K1_SMEM_STAGE = 16 * 1024; // per consumer warpgroup: 2 x [32 ch][64 tok] bf16 epilogue transpose buffers (128-B channel rows)
constexpr int K1_SMEM_GB = 2 * 1024;    // gamma, beta fp32
constexpr int K1_SMEM_BAR = 256;
constexpr int K1_SMEM = K1_SMEM_A + K1_SMEM_W + K1_SMEM_STAGE + K1_SMEM_GB + K1_SMEM_BAR;   // 215,296 B with 8 slots

#ifndef TX_LB_THREADS
#define TX_LB_THREADS NTHREADS
#endif

struct K1Params {
  CUtensorMap tm_z;      // 3D [C (ch)][N (j)][N (i)] bf16, box [64][BJ][BI], SW128, OOB -> 0
  CUtensorMap tm_w;      // 2D [C (k)][1024 (n)] bf16 (Wgp block-interleaved), box [64][64], SW128
  const float* mask;     // [N][N] fp32 or nullptr
  const float* gamma;    // LN_in weight [C]
  const float* beta;     // LN_in bias [C]
  __nv_bfloat16* ab;     // planes [2C][Np][Np] (Np = N allowed: unpadded planes; Np % 8 != 0 takes 2-byte stores)
  int N, Np, tiles_j, num_tiles, vec;
  float eps;
};

// byte offset inside a tile of 128-byte rows written by TMA with 128B swizzle (16-byte granule index ^= row % 8)
TX_DEVI uint32_t swz128(uint32_t row, uint32_t col_byte) {
  return row * 128u + ((((col_byte >> 4) ^ (row & 7u)) << 4) | (col_byte & 15u));
}
TX_DEVI float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
TX_DEVI float sigmoidf_(float g) { return rcpf(1.f + ex2f(-1.4426950408889634f * g)); }

// LayerNorm over the 256 values of two rows held as an m64k256 A fragment (fa[ks][0|2] = row A, fa[ks][1|3] = row B; k = 16 ks + 2 (lane%4) + {0,1} (+8)),
// statistics fp32 (mean, then centred variance), affine, repacked to bf16 in place.
TX_DEVI void ln_fragment(uint32_t (&fa)[16][4], const float* sGamma, const float* sBeta, int lane, float eps) {
  float sA_ = 0.f, sB_ = 0.f;
#pragma unroll
  for (int ks = 0; ks < 16; ++ks) {
    sA_ += bf16lo(fa[ks][0]) + bf16hi(fa[ks][0]) + bf16lo(fa[ks][2]) + bf16hi(fa[ks][2]);
    sB_ += bf16lo(fa[ks][1]) + bf16hi(fa[ks][1]) + bf16lo(fa[ks][3]) + bf16hi(fa[ks][3]);
  }
  const float meanA = quad_sum(sA_) * (1.f / C), meanB = quad_sum(sB_) * (1.f / C);
#pragma unroll
  for (int ks = 0; ks < 16; ++ks) fence_regs(fa[ks]);   // re-derive the fp32 values in each pass (2 ALU ops) instead of keeping 128 floats live
  float vA = 0.f, vB = 0.f;
#pragma unroll
  for (int ks = 0; ks < 16; ++ks) {
    float d;
    d = bf16lo(fa[ks][0]) - meanA; vA += d * d; d = bf16hi(fa[ks][0]) - meanA; vA += d * d;
    d = bf16lo(fa[ks][2]) - meanA; vA += d * d; d = bf16hi(fa[ks][2]) - meanA; vA += d * d;
    d = bf16lo(fa[ks][1]) - meanB; vB += d * d; d = bf16hi(fa[ks][1]) - meanB; vB += d * d;
    d = bf16lo(fa[ks][3]) - meanB; vB += d * d; d = bf16hi(fa[ks][3]) - meanB; vB += d * d;
  }
  const float rA = rsqrtf(quad_sum(vA) * (1.f / C) + eps), rB = rsqrtf(quad_sum(vB) * (1.f / C) + eps);
  const float mrA = meanA * rA, mrB = meanB * rB;
#pragma unroll
  for (int ks = 0; ks < 16; ++ks) fence_regs(fa[ks]);
#pragma unroll
  for (int ks = 0; ks < 16; ++ks) {
    const int k0 = 16 * ks + 2 * (lane & 3);
    const float2 g0 = *reinterpret_cast<const float2*>(sGamma + k0), b0 = *reinterpret_cast<const float2*>(sBeta + k0);
    const float2 g1 = *reinterpret_cast<const float2*>(sGamma + k0 + 8), b1 = *reinterpret_cast<const float2*>(sBeta + k0 + 8);
    // y = (x - mean) * r * g + b = x * (r g) + (b - mean r g)
    fa[ks][0] = pack_bf16(fmaf(bf16lo(fa[ks][0]), rA * g0.x, fmaf(-mrA, g0.x, b0.x)), fmaf(bf16hi(fa[ks][0]), rA * g0.y, fmaf(-mrA, g0.y, b0.y)));
    fa[ks][1] = pack_bf16(fmaf(bf16lo(fa[ks][1]), rB * g0.x, fmaf(-mrB, g0.x, b0.x)), fmaf(bf16hi(fa[ks][1]), rB * g0.y, fmaf(-mrB, g0.y, b0.y)));
    fa[ks][2] = pack_bf16(fmaf(bf16lo(fa[ks][2]), rA * g1.x, fmaf(-mrA, g1.x, b1.x)), fmaf(bf16hi(fa[ks][2]), rA * g1.y, fmaf(-mrA, g1.y, b1.y)));
    fa[ks][3] = pack_bf16(fmaf(bf16lo(fa[ks][3]), rB * g1.x, fmaf(-mrB, g1.x, b1.x)), fmaf(bf16hi(fa[ks][3]), rB * g1.y, fmaf(-mrB, g1.y, b1.y)));
  }
}


// ---------------------------------------------------------------------------------------------------------------------------------------------
// LayerNorm in a summation order under which the bf16 output is bit-identical, for the supported shapes, to the LayerNorm outputs of the stock op
// (cuEquivariance 0.11, bf16 input) -- a numerics reference: the order was found by matching that op's observable bf16 outputs (per-row statistics
// included) and is stated below on its own terms; nothing here describes that library's implementation:
//   * the 256 channels are 4 chunks of 64; chunks are accumulated elementwise first: A[c'] = ((x[c'] + x[c'+64]) + x[c'+128]) + x[c'+192];
//     for the centred squares the accumulation is contracted: acc = d0*d0, then acc = fma(d_k, d_k, acc) for k = 1, 2, 3;
//   * TREE 0 (LN_in, any N): t_w = ((A[8w] + A[8w+1]) + ... ) + A[8w+7] sequentially (w = 0..7), then a butterfly over
//     w with xor offsets 4, 2, 1;  TREE 1 (LN_out, plane row length N % 4 == 0): t[c'] = A[c'] + A[c'+32] (c' < 32), then
//     a butterfly over c' with xor offsets 2, 1, 16, 8, 4;  TREE 2 (LN_out when N % 4 != 0): t_v = ((A[v] + A[v+4]) + A[v+8]) + ...
//     + A[v+60] sequentially (v = 0..3, 16 terms, stride 4), then S = (t0 + t2) + (t1 + t3);
//   * mean = S / 256; var = S2 / 256 (centred, two passes); rstd = rsqrt.approx.ftz(var + eps); y = fma((x - mean) * rstd, gamma, beta) -> bf16 (rn).
// Every fp32 operation below is an explicit round-to-nearest intrinsic so that the compiler cannot re-associate or contract differently.
// Fragment ownership (q = lane % 4): register m of k-step ks holds channel 16 ks + {2q, 2q+1, 8+2q, 9+2q}[m] for row A (fa[ks][0] lo/hi, fa[ks][2] lo/hi)
// and row B (fa[ks][1], fa[ks][3]); hence c' mod 16 is fixed per thread and every chunk accumulation is thread-local; TREE 0's sequential runs cross
// the quad (thread 0 -> 1 -> 2 -> 3, a 3-step shuffle chain per run), TREE 1's butterfly needs lane^1 (channel xor 2) and lane^2 (channel xor 4) only,
// TREE 2's stride-4 runs alternate between lanes q and q^2 at every term (15 shuffle steps per run, two runs per lane in flight).
TX_DEVI float rsqrt_approx_ftz_(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
template <int ROW>   // ROW 0: registers [ks][0],[ks][2]; ROW 1: [ks][1],[ks][3]
TX_DEVI float frag_val(const uint32_t (&fa)[16][4], int ks, int m) {
  const uint32_t r = fa[ks][ROW + (m >> 1) * 2];
  return (m & 1) ? bf16hi(r) : bf16lo(r);
}
// S over one row: PASS 0 = plain sum of x, PASS 1 = sum of (x - mean)^2 with the contracted chunk accumulation
template <int TREE, int ROW, int PASS>
TX_DEVI float cueq_row_sum(const uint32_t (&fa)[16][4], float mean, int lane) {
  float A[4][4];
#pragma unroll
  for (int ks4 = 0; ks4 < 4; ++ks4) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      if (PASS == 0) {
        float acc = frag_val<ROW>(fa, ks4, m);
        acc = __fadd_rn(acc, frag_val<ROW>(fa, ks4 + 4, m)); acc = __fadd_rn(acc, frag_val<ROW>(fa, ks4 + 8, m)); acc = __fadd_rn(acc, frag_val<ROW>(fa, ks4 + 12, m));
        A[ks4][m] = acc;
      } else {
        float d = __fsub_rn(frag_val<ROW>(fa, ks4, m), mean); float acc = __fmul_rn(d, d);
        d = __fsub_rn(frag_val<ROW>(fa, ks4 + 4, m), mean); acc = __fmaf_rn(d, d, acc);
        d = __fsub_rn(frag_val<ROW>(fa, ks4 + 8, m), mean); acc = __fmaf_rn(d, d, acc);
        d = __fsub_rn(frag_val<ROW>(fa, ks4 + 12, m), mean); acc = __fmaf_rn(d, d, acc);
        A[ks4][m] = acc;
      }
    }
  }
  const int q = lane & 3;
  if (TREE == 0) {
    // 8 runs (ks4, h): this thread's pair = (A[ks4][2h], A[ks4][2h+1]) at run position q; 3 shuffle steps carry the partial from lane q-1
    float sr[4][2];
#pragma unroll
    for (int ks4 = 0; ks4 < 4; ++ks4) {
#pragma unroll
      for (int h = 0; h < 2; ++h) sr[ks4][h] = __fadd_rn(A[ks4][2 * h], A[ks4][2 * h + 1]);      // the run's start (used when q == 0)
    }
#pragma unroll
    for (int step = 1; step < 4; ++step) {
#pragma unroll
      for (int ks4 = 0; ks4 < 4; ++ks4) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float in = __shfl_sync(0xffffffffu, sr[ks4][h], (lane - 1) & 31);
          const float nxt = __fadd_rn(__fadd_rn(in, A[ks4][2 * h]), A[ks4][2 * h + 1]);
          sr[ks4][h] = (q == step) ? nxt : sr[ks4][h];
        }
      }
    }
    // lane q == 3 now holds t_w, w = 2 ks4 + h; butterfly over w (offsets 4, 2, 1): ((t0+t4)+(t2+t6)) + ((t1+t5)+(t3+t7))
    const float S = __fadd_rn(__fadd_rn(__fadd_rn(sr[0][0], sr[2][0]), __fadd_rn(sr[1][0], sr[3][0])),
                              __fadd_rn(__fadd_rn(sr[0][1], sr[2][1]), __fadd_rn(sr[1][1], sr[3][1])));
    return __shfl_sync(0xffffffffu, S, lane | 3);
  } else if (TREE == 1) {
    // t[c'] = A[c'] + A[c'+32]: c' < 32 <-> ks4 in {0, 1}; c' + 32 <-> ks4 + 2
    float u[2][4];
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2) {
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float t = __fadd_rn(A[k2][m], A[k2 + 2][m]);
        u[k2][m] = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, 1));      // channel xor 2 lives in lane ^ 1, same register slot
      }
    }
    const float v00 = __fadd_rn(u[0][0], u[0][1]), v01 = __fadd_rn(u[0][2], u[0][3]);   // channel xor 1: lo + hi
    const float v10 = __fadd_rn(u[1][0], u[1][1]), v11 = __fadd_rn(u[1][2], u[1][3]);
    const float w0 = __fadd_rn(v00, v10), w1 = __fadd_rn(v01, v11);                     // channel xor 16: ks4 0 + 1
    const float zz = __fadd_rn(w0, w1);                                                   // channel xor 8: register pair 0/1 + 2/3
    return __fadd_rn(zz, __shfl_xor_sync(0xffffffffu, zz, 2));                           // channel xor 4 lives in lane ^ 2
  } else {
    // run v = c' & 3 visits c' = v + 4 j, j = 0..15: term j lives in lane-half (q >> 1) == (j & 1) (register m = 0 / 2 for j % 4 in {0,1} / {2,3} of
    // k-step ks4 = j >> 2); this lane carries two runs: v = 2 (q & 1) (registers m = 0, 2) and v + 1 (registers m = 1, 3)
    const int half = q >> 1;
    float sa = A[0][0], sb = A[0][1];                     // term j = 0 (owned by half 0)
#pragma unroll
    for (int j = 1; j < 16; ++j) {
      const int ks4 = j >> 2, m0 = (j & 2) ? 2 : 0;
      const float va = __shfl_xor_sync(0xffffffffu, sa, 2), vb = __shfl_xor_sync(0xffffffffu, sb, 2);
      const float na = __fadd_rn(va, A[ks4][m0]), nb = __fadd_rn(vb, A[ks4][m0 + 1]);
      const bool mine = half == (j & 1);
      sa = mine ? na : sa; sb = mine ? nb : sb;
    }
    // half 1 lanes now hold t_v (sa) and t_{v+1} (sb) with v = 2 (q & 1): q = 2 -> t0, t1; q = 3 -> t2, t3.  S = (t0 + t2) + (t1 + t3).
    const float ua = __fadd_rn(sa, __shfl_xor_sync(0xffffffffu, sa, 1)), ub = __fadd_rn(sb, __shfl_xor_sync(0xffffffffu, sb, 1));
    const float S = __fadd_rn(ua, ub);
    return __shfl_sync(0xffffffffu, S, lane | 2);        // from a half-1 lane of this quad (lane | 2 has q in {2, 3}; both hold the same S)
  }
}
template <int TREE>
TX_DEVI void ln_cueq(uint32_t (&fa)[16][4], const float* sGamma, const float* sBeta, int lane, float eps) {
  const float meanA = __fmul_rn(cueq_row_sum<TREE, 0, 0>(fa, 0.f, lane), 1.f / C);
  const float meanB = __fmul_rn(cueq_row_sum<TREE, 1, 0>(fa, 0.f, lane), 1.f / C);
#pragma unroll
  for (int ks = 0; ks < 16; ++ks) fence_regs(fa[ks]);
  const float rA = rsqrt_approx_ftz_(__fadd_rn(__fmul_rn(cueq_row_sum<TREE, 0, 1>(fa, meanA, lane), 1.f / C), eps));
  const float rB = rsqrt_approx_ftz_(__fadd_rn(__fmul_rn(cueq_row_sum<TREE, 1, 1>(fa, meanB, lane), 1.f / C), eps));
#pragma unroll
  for (int ks = 0; ks < 16; ++ks) fence_regs(fa[ks]);
#pragma unroll
  for (int ks = 0; ks < 16; ++ks) {
    const int k0 = 16 * ks + 2 * (lane & 3);
    const float2 g0 = *reinterpret_cast<const float2*>(sGamma + k0), b0 = *reinterpret_cast<const float2*>(sBeta + k0);
    const float2 g1 = *reinterpret_cast<const float2*>(sGamma + k0 + 8), b1 = *reinterpret_cast<const float2*>(sBeta + k0 + 8);
#define TX_LNY(x, mean, r, g, b) __fmaf_rn(__fmul_rn(__fsub_rn((x), (mean)), (r)), (g), (b))
    fa[ks][0] = pack_bf16(TX_LNY(bf16lo(fa[ks][0]), meanA, rA, g0.x, b0.x), TX_LNY(bf16hi(fa[ks][0]), meanA, rA, g0.y, b0.y));
    fa[ks][1] = pack_bf16(TX_LNY(bf16lo(fa[ks][1]), meanB, rB, g0.x, b0.x), TX_LNY(bf16hi(fa[ks][1]), meanB, rB, g0.y, b0.y));
    fa[ks][2] = pack_bf16(TX_LNY(bf16lo(fa[ks][2]), meanA, rA, g1.x, b1.x), TX_LNY(bf16hi(fa[ks][2]), meanA, rA, g1.y, b1.y));
    fa[ks][3] = pack_bf16(TX_LNY(bf16lo(fa[ks][3]), meanB, rB, g1.x, b1.x), TX_LNY(bf16hi(fa[ks][3]), meanB, rB, g1.y, b1.y));
#undef TX_LNY
  }
}

// ============================================================================================================ K1
// 8 bf16 (one 16-byte granule in registers) -> global, first n elements only, as predicated 2-byte stores in ONE asm block (no lane-divergent branches):
// unpadded planes whose row length is not a multiple of 8.
TX_DEVI void stg_ragged(__nv_bfloat16* p, uint4 v, int n) {
  asm volatile(
    "{\n"
    ".reg .pred q<8>;\n"
    ".reg .b16 h<8>;\n"
    "mov.b32 {h0, h1}, %1;\n mov.b32 {h2, h3}, %2;\n mov.b32 {h4, h5}, %3;\n mov.b32 {h6, h7}, %4;\n"
    "setp.gt.s32 q0, %5, 0;\n setp.gt.s32 q1, %5, 1;\n setp.gt.s32 q2, %5, 2;\n setp.gt.s32 q3, %5, 3;\n"
    "setp.gt.s32 q4, %5, 4;\n setp.gt.s32 q5, %5, 5;\n setp.gt.s32 q6, %5, 6;\n setp.gt.s32 q7, %5, 7;\n"
    "@q0 st.global.b16 [%0], h0;\n @q1 st.global.b16 [%0+2], h1;\n @q2 st.global.b16 [%0+4], h2;\n @q3 st.global.b16 [%0+6], h3;\n"
    "@q4 st.global.b16 [%0+8], h4;\n @q5 st.global.b16 [%0+10], h5;\n @q6 st.global.b16 [%0+12], h6;\n @q7 st.global.b16 [%0+14], h7;\n"
    "}\n" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(n) : "memory");
}

template <int BI, int BJ, bool HAS_MASK, int CL, int LNM>
__global__ void __launch_bounds__(TX_LB_THREADS, 1) k1_kernel(const __grid_constant__ K1Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sW = smem + K1_SMEM_A;
  uint8_t* sStage = sW + K1_SMEM_W;
  float* sGamma = reinterpret_cast<float*>(sStage + K1_SMEM_STAGE);
  float* sBeta = sGamma + C;
  uint64_t* bars = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sGamma) + K1_SMEM_GB);
  uint64_t* barA_full = bars + 0;
  uint64_t* barA_empty = bars + 1;
  uint64_t* barW_full = bars + 2;               // [K1_NSLOT]
  uint64_t* barW_empty = bars + 2 + K1_NSLOT;   // [K1_NSLOT]

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wg = warp >> 2;
  // tile schedule: CTA (cluster c, rank r) serves tiles first + r + it * gridDim.x, it < n_iter; both CTAs of a cluster run the same n_iter
  // (they consume one multicast weight stream); a rank-1 iteration past the last tile is a phantom (loads clamped, nothing stored).
  const uint32_t rank = (CL > 1) ? cluster_ctarank() : 0u;
  const int first = (int)blockIdx.x - (int)rank;
  const int n_iter = (p.num_tiles - first + (int)gridDim.x - 1) / (int)gridDim.x;
  for (int i = tid; i < C; i += NTHREADS) { sGamma[i] = p.gamma[i]; sBeta[i] = p.beta[i]; }
  if (tid == 0) {
    mbar_init(barA_full, 1); mbar_init(barA_empty, 8);
#if defined(TX_CL_NOMC) || defined(TX_MC_LOCALGATE)
    for (int s = 0; s < K1_NSLOT; ++s) { mbar_init(barW_full + s, 1); mbar_init(barW_empty + s, 8); }
#else
    for (int s = 0; s < K1_NSLOT; ++s) { mbar_init(barW_full + s, 1); mbar_init(barW_empty + s, 8 * CL); }
#endif
    fence_barrier_init();
    tma_prefetch_desc(&p.tm_z); tma_prefetch_desc(&p.tm_w);
  }
  if (CL > 1) cluster_sync(); else __syncthreads();

  if (wg == 0) {
    // ------------------------------------------------------------------ producer warpgroup
    setmaxnreg_dec<40>();
    if (warp == 0 && lane == 0) {                       // z tiles: one 64 KB load per tile, issued as soon as the consumers have pulled the previous tile into registers
      for (int it = 0; it < n_iter; ++it) {
        int tile = first + (int)rank + it * (int)gridDim.x; if (tile >= p.num_tiles) tile = p.num_tiles - 1;
        if (it > 0) mbar_wait(barA_empty, (it - 1) & 1);
        const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
        mbar_arrive_expect_tx(barA_full, K1_SMEM_A);
#pragma unroll
        for (int kc = 0; kc < 4; ++kc) tma_load_3d(sA + kc * 16384, &p.tm_z, barA_full, kc * 64, j0, i0);
      }
    } else if (warp == 1 && lane == 0) {                // weight ring: 32 half-blocks per tile, identical sequence every tile (L2-resident after the first)
      const uint32_t n_w = (uint32_t)n_iter * 32u;
#if defined(TX_CL_NOMC) || defined(TX_GATE_ONLY)
      constexpr bool MCP = false;
#else
      constexpr bool MCP = (CL > 1);
#endif
      if (rank == 0 || !MCP) {                          // the issuing CTA: waits until every consumer warp of the cluster released the slot, registers the bytes on
        uint32_t full_remote[CL > 1 ? CL : 1];          // every CTA's full barrier (remote arrive.expect_tx), then multicasts the half-block into all CTAs at once
        for (int r = 1; r < CL; ++r) full_remote[r] = mapa_shared(smem_u32(barW_full), (uint32_t)r);
        for (uint32_t w_iter = 0; w_iter < n_w; ++w_iter) {
          const int s = w_iter % K1_NSLOT; const uint32_t u = w_iter / K1_NSLOT;
          if (u > 0) mbar_wait(barW_empty + s, (u - 1) & 1);
#ifdef TX_FAKE_W
          if (w_iter >= (uint32_t)K1_NSLOT) { mbar_arrive(barW_full + s); continue; }   // diagnostic only
#endif
          mbar_arrive_expect_tx(barW_full + s, 16384);
          const int hb = w_iter & 31, b = hb >> 1, h = hb & 1;
          if (MCP) {
            for (int r = 1; r < CL; ++r) mbar_arrive_expect_tx_cluster(full_remote[r] + 8 * s, 16384);
            tma_load_2d_mc(sW + s * 16384, &p.tm_w, barW_full + s, (2 * h) * 64, 64 * b, (uint16_t)((1u << CL) - 1));
            tma_load_2d_mc(sW + s * 16384 + 8192, &p.tm_w, barW_full + s, (2 * h + 1) * 64, 64 * b, (uint16_t)((1u << CL) - 1));
          } else {
            tma_load_2d(sW + s * 16384, &p.tm_w, barW_full + s, (2 * h) * 64, 64 * b);
            tma_load_2d(sW + s * 16384 + 8192, &p.tm_w, barW_full + s, (2 * h + 1) * 64, 64 * b);
          }
        }
      }
    }
    __syncwarp();
    if (CL > 1) cluster_sync();                         // keep this CTA's smem/barriers alive until the partner is done with them
    return;
  }

  // ------------------------------------------------------------------ consumer warpgroups
  setmaxnreg_inc<232>();
#ifdef TX_CL_NOMC
  constexpr bool MC = false;
#else
  constexpr bool MC = (CL > 1);
#endif
  const uint32_t w_empty_base = MC ? mapa_shared(smem_u32(barW_empty), 0) : smem_u32(barW_empty);   // slot releases go to the issuing CTA
  const int cw = wg - 1;                 // 0 / 1 : token rows [64 cw, 64 cw + 64) of the tile
  const int wiw = warp & 3;              // warp within warpgroup: rows 16*wiw.. of the m64 tile
  const int mat = lane >> 3, r8 = lane & 7;
  const uint32_t sA_u = smem_u32(sA), sW_u = smem_u32(sW);
  const uint32_t stage_u = smem_u32(sStage) + (uint32_t)(cw * 8192);   // this warpgroup's 2 x 4 KB staging buffers [32 ch][64 tok]
  const size_t plane = (size_t)p.Np * (size_t)p.Np;
  const int rho0 = 64 * cw + 16 * wiw;                  // first of this warp's 16 token rows (tile-relative)
  const int rowA = rho0 + (lane >> 2), rowB = rowA + 8;  // this thread's accumulator rows
  // stmatrix role: lane -> (matrix idx, row rr): channel ch = 16 hq + 8 (idx>>1) + rr; token granule tg = 2 wiw + (idx&1) (8 tokens each), stored at tg ^ (ch & 7)
  const int idx = lane >> 3, rr = lane & 7;
  const int chq = 8 * (idx >> 1) + rr;                  // channel within the block for hq = 0 (+16 for hq = 1); (chq+16)&7 == chq&7
  const uint32_t sts_off0 = (uint32_t)chq * 128 + (uint32_t)(((2 * wiw + (idx & 1)) ^ (chq & 7)) * 16);
  const uint32_t sts_off1 = sts_off0 + 16 * 128;
  // store role: warp wiw stores channels 8 wiw .. 8 wiw + 7 of the block; instruction e (0,1): lanes 8k..8k+7 -> channel 8 wiw + 4 e + k, granule lane%8
  const int st_ch0 = 8 * wiw + (lane >> 3), st_g = lane & 7;            // channel (e = 0), token granule (8 tokens = 16 B)
  const uint32_t ld_off0 = (uint32_t)st_ch0 * 128 + (uint32_t)((st_g ^ (st_ch0 & 7)) * 16);
  const uint32_t ld_off1 = (uint32_t)(st_ch0 + 4) * 128 + (uint32_t)((st_g ^ ((st_ch0 + 4) & 7)) * 16);
  const int bar_id = 1 + cw;                            // named barrier of this warpgroup
  uint32_t w_iter = 0;

  for (int t_local = 0; t_local < n_iter; ++t_local) {
    int tile = first + (int)rank + t_local * (int)gridDim.x;
    const bool phantom = tile >= p.num_tiles; if (phantom) tile = p.num_tiles - 1;
    const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
    // ---- this thread's two rows: validity + mask; this lane's two store runs: pointers + predicates
    const int iA = i0 + rowA / BJ, jA = j0 + rowA % BJ, iB = i0 + rowB / BJ, jB = j0 + rowB % BJ;
    const bool vA = (iA < p.N) && (jA < p.N), vB = (iB < p.N) && (jB < p.N);
    float mA = vA ? 1.f : 0.f, mB = vB ? 1.f : 0.f;
    if (HAS_MASK) { if (vA) mA = __ldg(p.mask + (size_t)iA * p.N + jA); if (vB) mB = __ldg(p.mask + (size_t)iB * p.N + jB); }
    const int rho_g = 64 * cw + 8 * st_g;                                  // first token of this lane's store granule
    const int is_ = i0 + rho_g / BJ, js_ = j0 + rho_g % BJ;
    const bool st_ok = (is_ < p.Np) && (js_ < p.Np) && !phantom;
    const int st_n = min(8, p.Np - js_);                                                 // valid tokens of this lane's 8-token granule (ragged planes)
    __nv_bfloat16* gp0 = p.ab + (size_t)st_ch0 * plane + (size_t)is_ * p.Np + js_;   // + 32 b * plane per block; + 4 * plane for e = 1
    const size_t blk_stride = 32 * plane, e_stride = 4 * plane;

    // ---- A fragments: ldmatrix from the swizzled z tile, then LayerNorm in registers
    mbar_wait(barA_full, t_local & 1);
    uint32_t fa[16][4];
    {
      const int row = rho0 + r8 + ((mat & 1) ? 8 : 0);
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        const int kc = ks >> 2, kin = (ks & 3) * 16 + ((mat & 2) ? 8 : 0);
        ldsm_x4(fa[ks], sA_u + kc * 16384 + swz128(row, kin * 2));
      }
    }
    // WAR across proxies: the ldmatrix reads above go through the generic proxy, the producer's refill of sA is an async-proxy (TMA) write.  The
    // mbarrier release alone does not order the two; fence.proxy.async does (without it the refill can overtake in-flight reads: observed as
    // run-to-run differences whenever the consumer path between this point and the first MMA is short).
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) mbar_arrive(barA_empty);              // the producer may refill sA with the next tile now
    if (LNM == 1) ln_fragment(fa, sGamma, sBeta, lane, p.eps);        // LNM 0: the tile already holds LayerNorm output
    else if (LNM == 2) ln_cueq<0>(fa, sGamma, sBeta, lane, p.eps);   // the stock-matching summation order (bitwise class)

    // ---- 16 weight blocks of 32 channels (n64 = gate | proj); MMAs of block b+1 are issued before the epilogue of block b
    float acc0[32], acc1[32];
    auto issue_block = [&](float (&ac)[32], uint32_t wi) {
      const int s0 = wi % K1_NSLOT, s1 = (wi + 1) % K1_NSLOT;
      mbar_wait(barW_full + s0, (wi / K1_NSLOT) & 1);
      mbar_wait(barW_full + s1, ((wi + 1) / K1_NSLOT) & 1);
      const uint64_t d0 = smem_desc(sW_u + s0 * 16384, 16, 1024, 1), d1 = smem_desc(sW_u + s1 * 16384, 16, 1024, 1);
      const uint32_t d0lo = (uint32_t)d0, d0hi = (uint32_t)(d0 >> 32), d1lo = (uint32_t)d1, d1hi = (uint32_t)(d1 >> 32);
#pragma unroll
      for (int i = 0; i < 32; ++i) ac[i] = 0.f;
      fence_regs(ac);
      wgmma_fence();
      wgmma_m64n64k16_rs_off<0 * 32>(ac, fa[0], d0lo, d0hi, 0);
      wgmma_m64n64k16_rs_off<1 * 32>(ac, fa[1], d0lo, d0hi, 1);
      wgmma_m64n64k16_rs_off<2 * 32>(ac, fa[2], d0lo, d0hi, 1);
      wgmma_m64n64k16_rs_off<3 * 32>(ac, fa[3], d0lo, d0hi, 1);
      wgmma_m64n64k16_rs_off<8192 + 0 * 32>(ac, fa[4], d0lo, d0hi, 1);
      wgmma_m64n64k16_rs_off<8192 + 1 * 32>(ac, fa[5], d0lo, d0hi, 1);
      wgmma_m64n64k16_rs_off<8192 + 2 * 32>(ac, fa[6], d0lo, d0hi, 1);
      wgmma_m64n64k16_rs_off<8192 + 3 * 32>(ac, fa[7], d0lo, d0hi, 1);
      wgmma_m64n64k16_rs_off<0 * 32>(ac, fa[8], d1lo, d1hi, 1);
      wgmma_m64n64k16_rs_off<1 * 32>(ac, fa[9], d1lo, d1hi, 1);
      wgmma_m64n64k16_rs_off<2 * 32>(ac, fa[10], d1lo, d1hi, 1);
      wgmma_m64n64k16_rs_off<3 * 32>(ac, fa[11], d1lo, d1hi, 1);
      wgmma_m64n64k16_rs_off<8192 + 0 * 32>(ac, fa[12], d1lo, d1hi, 1);
      wgmma_m64n64k16_rs_off<8192 + 1 * 32>(ac, fa[13], d1lo, d1hi, 1);
      wgmma_m64n64k16_rs_off<8192 + 2 * 32>(ac, fa[14], d1lo, d1hi, 1);
      wgmma_m64n64k16_rs_off<8192 + 3 * 32>(ac, fa[15], d1lo, d1hi, 1);
      wgmma_commit();
    };
    uint32_t keep = 0; (void)keep;
    auto epilogue = [&](float (&ac)[32], int b, uint32_t wi) {
      fence_regs(ac);
      __syncwarp();
      if (lane == 0) {
#if defined(TX_MC_LOCALGATE)
        if (rank == 0) { mbar_arrive(barW_empty + (wi % K1_NSLOT)); mbar_arrive(barW_empty + ((wi + 1) % K1_NSLOT)); }
#elif defined(TX_GATE_ONLY)
        for (uint32_t r = 0; r < (uint32_t)CL; ++r) { const uint32_t eb = mapa_shared(smem_u32(barW_empty), r); mbar_arrive_cluster(eb + 8 * (wi % K1_NSLOT)); mbar_arrive_cluster(eb + 8 * ((wi + 1) % K1_NSLOT)); }
#else
        if (MC) { mbar_arrive_cluster(w_empty_base + 8 * (wi % K1_NSLOT)); mbar_arrive_cluster(w_empty_base + 8 * ((wi + 1) % K1_NSLOT)); }
        else { mbar_arrive(barW_empty + (wi % K1_NSLOT)); mbar_arrive(barW_empty + ((wi + 1) % K1_NSLOT)); }
#endif
      }
#ifdef TX_SKIP_EPI
      if (ac[0] == 12345.f && lane == 99) stg32(gp0, 0);   // keep the accumulators alive
      return;
#endif
      uint32_t pk[4][2];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
#ifdef TX_EPI_NOSIG
        float vA0 = (ac[4 * q + 0]) * ac[4 * (q + 4) + 0] * mA, vA1 = (ac[4 * q + 1]) * ac[4 * (q + 4) + 1] * mA;
        float vB0 = (ac[4 * q + 2]) * ac[4 * (q + 4) + 2] * mB, vB1 = (ac[4 * q + 3]) * ac[4 * (q + 4) + 3] * mB;
#else
        float vA0 = sigmoidf_(ac[4 * q + 0]) * ac[4 * (q + 4) + 0] * mA, vA1 = sigmoidf_(ac[4 * q + 1]) * ac[4 * (q + 4) + 1] * mA;
        float vB0 = sigmoidf_(ac[4 * q + 2]) * ac[4 * (q + 4) + 2] * mB, vB1 = sigmoidf_(ac[4 * q + 3]) * ac[4 * (q + 4) + 3] * mB;
#endif
        if (!vA) { vA0 = 0.f; vA1 = 0.f; }
        if (!vB) { vB0 = 0.f; vB1 = 0.f; }
        pk[q][0] = pack_bf16(vA0, vA1); pk[q][1] = pack_bf16(vB0, vB1);
      }
#ifdef TX_EPI_NOSTAGE
      { uint4 v0 = make_uint4(pk[0][0], pk[0][1], pk[1][0], pk[1][1]), v1 = make_uint4(pk[2][0], pk[2][1], pk[3][0], pk[3][1]);
        const size_t bo = (size_t)b * blk_stride;
        if (st_ok && lane == 99) { stg128(gp0 + bo, v0); stg128(gp0 + bo + e_stride, v1); } }
      return;
#endif
      const uint32_t sbuf = stage_u + (uint32_t)((b & 1) * 4096);            // double-buffered: one warpgroup barrier per block
#if defined(TX_EPI_KEEPALIVE)
      keep ^= pk[0][0] ^ pk[0][1] ^ pk[1][0] ^ pk[1][1] ^ pk[2][0] ^ pk[2][1] ^ pk[3][0] ^ pk[3][1];
      if (b == 15) sts32(sbuf + lane * 4 + wiw * 128, keep);
#elif defined(TX_EPI_STS32)
      // plain 32-bit shared stores of the same fragment (token-major rows: [64 tok][32 ch] bf16, row = 64 B) - layout differs, timing only
      sts32(sbuf + rowA * 64 + (2 * (lane & 3)) * 2 + 0 * 16, pk[0][0]); sts32(sbuf + rowB * 64 + (2 * (lane & 3)) * 2 + 0 * 16, pk[0][1]);
      sts32(sbuf + rowA * 64 + (2 * (lane & 3)) * 2 + 1 * 16, pk[1][0]); sts32(sbuf + rowB * 64 + (2 * (lane & 3)) * 2 + 1 * 16, pk[1][1]);
      sts32(sbuf + rowA * 64 + (2 * (lane & 3)) * 2 + 2 * 16, pk[2][0]); sts32(sbuf + rowB * 64 + (2 * (lane & 3)) * 2 + 2 * 16, pk[2][1]);
      sts32(sbuf + rowA * 64 + (2 * (lane & 3)) * 2 + 3 * 16, pk[3][0]); sts32(sbuf + rowB * 64 + (2 * (lane & 3)) * 2 + 3 * 16, pk[3][1]);
#elif !defined(TX_EPI_NOSTSM)
      stsm_x4_t(sbuf + sts_off0, pk[0][0], pk[0][1], pk[1][0], pk[1][1]);      // channels 0..15 of the block
      stsm_x4_t(sbuf + sts_off1, pk[2][0], pk[2][1], pk[3][0], pk[3][1]);      // channels 16..31
#endif
#ifndef TX_EPI_NOBAR
      named_bar_sync(bar_id, 128);
#endif
#ifdef TX_EPI_NOLDS
      const uint4 v0 = make_uint4(pk[0][0], pk[0][1], pk[1][0], pk[1][1]), v1 = make_uint4(pk[2][0], pk[2][1], pk[3][0], pk[3][1]);
#else
      const uint4 v0 = lds128(sbuf + ld_off0), v1 = lds128(sbuf + ld_off1);
#endif
      const size_t bo = (size_t)b * blk_stride;
#ifdef TX_EPI_NOSTORE
      if (st_ok && lane == 99) { stg128(gp0 + bo, v0); stg128(gp0 + bo + e_stride, v1); }
#else
      if (st_ok) {
        if (p.vec) { stg128(gp0 + bo, v0); stg128(gp0 + bo + e_stride, v1); }
        else { stg_ragged(gp0 + bo, v0, st_n); stg_ragged(gp0 + bo + e_stride, v1, st_n); }   // unpadded planes with N % 8 != 0: element stores, j < N
      }
#endif
    };

    issue_block(acc0, w_iter);
#pragma unroll 1
    for (int b = 0; b < 16; b += 2, w_iter += 4) {
      issue_block(acc1, w_iter + 2);
      wgmma_wait<1>();
      epilogue(acc0, b, w_iter);
      if (b + 2 < 16) { issue_block(acc0, w_iter + 4); wgmma_wait<1>(); }
      else { wgmma_wait<0>(); }
      epilogue(acc1, b + 1, w_iter + 2);
    }
    // the A fragments are read ASYNCHRONOUSLY by every block's wgmma: keep their registers allocated (unused by epilogue temporaries) until the last
    // group of this tile has retired (wgmma_wait<0> above) — an explicit use here pins the live range for the register allocator.
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) fence_regs(fa[ks]);
  }
  if (CL > 1) cluster_sync();
}

template <int BI, int BJ, int CL, int LNM = 1>
void k1_launch(const K1Params& p, bool has_mask, int grid, cudaStream_t st) {
  auto kern = has_mask ? k1_kernel<BI, BJ, true, CL, LNM> : k1_kernel<BI, BJ, false, CL, LNM>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, K1_SMEM));
  if (CL == 1) {
    kern<<<grid, NTHREADS, K1_SMEM, st>>>(p);
  } else {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid, 1, 1); cfg.blockDim = dim3(NTHREADS, 1, 1); cfg.dynamicSmemBytes = K1_SMEM; cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension; attr[0].val.clusterDim.x = CL; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ============================================================================================================ K3
// out[t, :] = z[t, :] + bf16( sigmoid(LN_in(z)[t] . Wgo^T) * (LN_out(x[:, t]) . Wz^T) ), t over the N x N tokens, 256 output channels.
// Tile = BI x BJ = 128 tokens (WG cw serves 64 consecutive tokens of row i0 + (64 cw)/BJ).  Per tile: the x tile (2 x 32 KB boxes [64 tok][256 ch],
// MN-major A operand via ldmatrix.trans) is pulled into registers first and released at once (the next tile's x load overlaps everything else);
// the z tile arrives as 4 channel chunks of 16 KB with their own barriers: all 4 feed the gate operand (ldmatrix), and chunk b is released after
// output block b has taken its residual from it — so the producer refills chunk b with the next tile's z while blocks b+1.. still run.
// Weights: ring of 16 KB half-blocks in the order (Wgo_b h0, h1, Wz_b h0, h1) for b = 0..3 (64 output channels each).
#ifndef TX_K3_VEC_UNROLL
#define TX_K3_VEC_UNROLL 4
#endif
constexpr int kK3VecUnroll = TX_K3_VEC_UNROLL;
constexpr int K3_NSLOT = 4;
constexpr int K3_SMEM_Z = 4 * 16384;
constexpr int K3_SMEM_X = 2 * 32768;
constexpr int K3_SMEM_W = K3_NSLOT * 16384;
constexpr int K3_SMEM_STAGE = 16 * 1024;      // per WG one [64 tok][64 ch] bf16 buffer (8 KB, 128-B rows, swizzled)
constexpr int K3_SMEM_GB = 4 * 1024;          // LN_in gamma/beta, LN_out gamma/beta
constexpr int K3_SMEM_BAR = 256;
constexpr int K3_SMEM = K3_SMEM_Z + K3_SMEM_X + K3_SMEM_W + K3_SMEM_STAGE + K3_SMEM_GB + K3_SMEM_BAR;   // 217,344 B

struct K3Params {
  CUtensorMap tm_z;      // 3D [C][N (j)][N (i)] bf16, box [64][BJ][BI]
  CUtensorMap tm_x;      // 3D [Np (j)][Np (i)][C (ch)] bf16 (the contraction planes), box [64][1][256]
  CUtensorMap tm_w;      // 2D [C (k)][512 (n)] bf16: rows 0..255 = Wgo (linear_g), 256..511 = Wz (linear_z); box [64][64]
  const float* gamma_in; const float* beta_in; const float* gamma_out; const float* beta_out;
  const __nv_bfloat16* zres;   // XIO: residual source z [N][N][C] (read from global); unused otherwise
  __nv_bfloat16* out;          // [N][N][C]
  int N, Np, tiles_j, num_tiles, residual;
  float eps;
};

// XIO = true (exact-tier path): both A operands arrive as LayerNorm OUTPUT tiles, K-major [tok][256]: tm_z carries the gate source (LN_in output) into sZ,
// tm_x (built like tm_z) carries the projection source (LN_out output, token-major) into sX as 4 channel chunks; no LayerNorm inside; the residual z
// granule is read from global (p.zres); the sZ chunks are released as soon as their fragments are in registers.
template <int BI, int BJ, bool XIO, int LNM>
__global__ void __launch_bounds__(TX_LB_THREADS, 1) k3_kernel(const __grid_constant__ K3Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sZ = smem;
  uint8_t* sX = sZ + K3_SMEM_Z;
  uint8_t* sW = sX + K3_SMEM_X;
  uint8_t* sStage = sW + K3_SMEM_W;
  float* sGin = reinterpret_cast<float*>(sStage + K3_SMEM_STAGE);
  float* sBin = sGin + C; float* sGout = sBin + C; float* sBout = sGout + C;
  uint64_t* bars = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sGin) + K3_SMEM_GB);
  uint64_t* barZ_full = bars + 0;             // [4] per channel chunk
  uint64_t* barZ_empty = bars + 4;            // [4]
  uint64_t* barX_full = bars + 8; uint64_t* barX_empty = bars + 9;
  uint64_t* barW_full = bars + 10; uint64_t* barW_empty = bars + 10 + K3_NSLOT;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wg = warp >> 2;
  if (!XIO) for (int i = tid; i < C; i += NTHREADS) { sGin[i] = p.gamma_in[i]; sBin[i] = p.beta_in[i]; sGout[i] = p.gamma_out[i]; sBout[i] = p.beta_out[i]; }
  if (tid == 0) {
    for (int kc = 0; kc < 4; ++kc) { mbar_init(barZ_full + kc, 1); mbar_init(barZ_empty + kc, 8); }
    mbar_init(barX_full, 1); mbar_init(barX_empty, 8);
    for (int s = 0; s < K3_NSLOT; ++s) { mbar_init(barW_full + s, 1); mbar_init(barW_empty + s, 8); }
    fence_barrier_init();
    tma_prefetch_desc(&p.tm_z); tma_prefetch_desc(&p.tm_x); tma_prefetch_desc(&p.tm_w);
  }
  __syncthreads();

  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (warp == 0 && lane == 0) {
      int t_local = 0;
      for (int tile = blockIdx.x; tile < p.num_tiles; tile += gridDim.x, ++t_local) {
        const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
        if (t_local > 0) mbar_wait(barX_empty, (t_local - 1) & 1);
        mbar_arrive_expect_tx(barX_full, K3_SMEM_X);
        if (XIO) {
#pragma unroll
          for (int kc = 0; kc < 4; ++kc) tma_load_3d(sX + kc * 16384, &p.tm_x, barX_full, kc * 64, j0, i0);   // token-major LN_out output, 4 channel chunks
        } else {
#pragma unroll
          for (int h = 0; h < 2; ++h) {                  // sub-tile h = tokens 64h..64h+63 of the tile
            const int ih = i0 + (64 * h) / BJ, jh = j0 + (64 * h) % BJ;
            tma_load_3d(sX + h * 32768, &p.tm_x, barX_full, jh, ih, 0);
          }
        }
        for (int kc = 0; kc < 4; ++kc) {
          if (t_local > 0) mbar_wait(barZ_empty + kc, (t_local - 1) & 1);
          mbar_arrive_expect_tx(barZ_full + kc, 16384);
          tma_load_3d(sZ + kc * 16384, &p.tm_z, barZ_full + kc, kc * 64, j0, i0);
        }
      }
    } else if (warp == 1 && lane == 0) {
      uint32_t w_iter = 0;
      for (int tile = blockIdx.x; tile < p.num_tiles; tile += gridDim.x) {
        for (int hb = 0; hb < 16; ++hb, ++w_iter) {       // (b, which, h): rows 64 b + 256 which, k chunks 2h, 2h+1
          const int s = w_iter % K3_NSLOT; const uint32_t u = w_iter / K3_NSLOT;
          if (u > 0) mbar_wait(barW_empty + s, (u - 1) & 1);
          mbar_arrive_expect_tx(barW_full + s, 16384);
          const int b = hb >> 2, which = (hb >> 1) & 1, h = hb & 1;
          tma_load_2d(sW + s * 16384, &p.tm_w, barW_full + s, (2 * h) * 64, 64 * b + 256 * which);
          tma_load_2d(sW + s * 16384 + 8192, &p.tm_w, barW_full + s, (2 * h + 1) * 64, 64 * b + 256 * which);
        }
      }
    }
    return;
  }

  setmaxnreg_inc<232>();
  const int cw = wg - 1, wiw = warp & 3, mat = lane >> 3, r8 = lane & 7;
  const uint32_t sZ_u = smem_u32(sZ), sX_u = smem_u32(sX) + (uint32_t)(cw * 32768), sXk_u = smem_u32(sX), sW_u = smem_u32(sW);
  const uint32_t stage_u = smem_u32(sStage) + (uint32_t)(cw * 8192);
  const int rho0 = 64 * cw + 16 * wiw;                   // tile-relative first row of this warp
  // stmatrix (non-trans) role: WG-relative token row = 16 wiw + 8 (idx & 1) + rr, channel granule = 2 qp + (idx >> 1)
  const int idx = lane >> 3, rr = lane & 7;
  const int st_row = 16 * wiw + 8 * (idx & 1) + rr;
  // vector pass role: row group of 4 rows per instruction: row = 16 wiw + 4 it + lane/8, granule lane%8
  const int vg = lane & 7, vr = lane >> 3;
  const int bar_id = 1 + cw;
  uint32_t w_iter = 0;
  int t_local = 0;

  for (int tile = blockIdx.x; tile < p.num_tiles; tile += gridDim.x, ++t_local) {
    const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
    const int iw = i0 + (64 * cw) / BJ, jw = j0 + (64 * cw) % BJ;     // this WG's 64 tokens: row iw, columns jw .. jw+63

    // ---- projection operand first (its buffer was refilled during the previous tile): LN_out(x), x tile [256 ch rows][64 tok] -> ldmatrix.trans
    uint32_t fx[16][4];
    mbar_wait(barX_full, t_local & 1);
    if (XIO) {                                           // K-major tile (LN_out output, token rows): same fragment walk as the gate operand
      const int row = rho0 + r8 + ((mat & 1) ? 8 : 0);
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) {
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4) {
          const int ks = 4 * kc + k4, kin = k4 * 16 + ((mat & 2) ? 8 : 0);
          ldsm_x4(fx[ks], sXk_u + kc * 16384 + swz128(row, kin * 2));
        }
      }
    } else {
      const int tokc = 16 * wiw + ((mat & 1) ? 8 : 0);
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        const int krow = 16 * ks + r8 + ((mat & 2) ? 8 : 0);
        ldsm_x4_t(fx[ks], sX_u + swz128(krow, tokc * 2));
      }
    }
    fence_proxy_async();                                  // generic-proxy reads of sX before the async-proxy refill (see K1)
    __syncwarp();
    if (lane == 0) mbar_arrive(barX_empty);
    if (!XIO) { if (LNM == 2) ln_cueq<1>(fx, sGout, sBout, lane, p.eps); else if (LNM == 3) ln_cueq<2>(fx, sGout, sBout, lane, p.eps); else ln_fragment(fx, sGout, sBout, lane, p.eps); }
    // ---- gate operand: LN_in(z), chunk by chunk as they land
    uint32_t fz[16][4];
    {
      const int row = rho0 + r8 + ((mat & 1) ? 8 : 0);
      for (int kc = 0; kc < 4; ++kc) mbar_wait(barZ_full + kc, t_local & 1);
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) {
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4) {
          const int ks = 4 * kc + k4, kin = k4 * 16 + ((mat & 2) ? 8 : 0);
          ldsm_x4(fz[ks], sZ_u + kc * 16384 + swz128(row, kin * 2));
        }
      }
    }
    if (XIO) {                                           // the gate tile is fully in registers: release all four chunks now (the producer refills them with the next tile)
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) { for (int kc = 0; kc < 4; ++kc) mbar_arrive(barZ_empty + kc); }
    } else {
      if (LNM >= 2) ln_cueq<0>(fz, sGin, sBin, lane, p.eps); else ln_fragment(fz, sGin, sBin, lane, p.eps);
    }

#pragma unroll 1
    for (int b = 0; b < 4; ++b, w_iter += 4) {
      float accg[32], accp[32];
      const int s0 = w_iter % K3_NSLOT, s1 = (w_iter + 1) % K3_NSLOT, s2 = (w_iter + 2) % K3_NSLOT, s3 = (w_iter + 3) % K3_NSLOT;
      mbar_wait(barW_full + s0, (w_iter / K3_NSLOT) & 1);
      mbar_wait(barW_full + s1, ((w_iter + 1) / K3_NSLOT) & 1);
      {
        const uint64_t d0 = smem_desc(sW_u + s0 * 16384, 16, 1024, 1), d1 = smem_desc(sW_u + s1 * 16384, 16, 1024, 1);
        const uint32_t d0lo = (uint32_t)d0, d0hi = (uint32_t)(d0 >> 32), d1lo = (uint32_t)d1, d1hi = (uint32_t)(d1 >> 32);
        fence_regs(accg);
        wgmma_fence();
        wgmma_m64n64k16_rs_off<0>(accg, fz[0], d0lo, d0hi, 0);
        wgmma_m64n64k16_rs_off<32>(accg, fz[1], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<64>(accg, fz[2], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<96>(accg, fz[3], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<8192>(accg, fz[4], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<8224>(accg, fz[5], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<8256>(accg, fz[6], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<8288>(accg, fz[7], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<0>(accg, fz[8], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<32>(accg, fz[9], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<64>(accg, fz[10], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<96>(accg, fz[11], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<8192>(accg, fz[12], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<8224>(accg, fz[13], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<8256>(accg, fz[14], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<8288>(accg, fz[15], d1lo, d1hi, 1);
        wgmma_commit();
      }
      mbar_wait(barW_full + s2, ((w_iter + 2) / K3_NSLOT) & 1);
      mbar_wait(barW_full + s3, ((w_iter + 3) / K3_NSLOT) & 1);
      {
        const uint64_t d0 = smem_desc(sW_u + s2 * 16384, 16, 1024, 1), d1 = smem_desc(sW_u + s3 * 16384, 16, 1024, 1);
        const uint32_t d0lo = (uint32_t)d0, d0hi = (uint32_t)(d0 >> 32), d1lo = (uint32_t)d1, d1hi = (uint32_t)(d1 >> 32);
        fence_regs(accp);
        wgmma_fence();
        wgmma_m64n64k16_rs_off<0>(accp, fx[0], d0lo, d0hi, 0);
        wgmma_m64n64k16_rs_off<32>(accp, fx[1], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<64>(accp, fx[2], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<96>(accp, fx[3], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<8192>(accp, fx[4], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<8224>(accp, fx[5], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<8256>(accp, fx[6], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<8288>(accp, fx[7], d0lo, d0hi, 1);
        wgmma_m64n64k16_rs_off<0>(accp, fx[8], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<32>(accp, fx[9], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<64>(accp, fx[10], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<96>(accp, fx[11], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<8192>(accp, fx[12], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<8224>(accp, fx[13], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<8256>(accp, fx[14], d1lo, d1hi, 1);
        wgmma_m64n64k16_rs_off<8288>(accp, fx[15], d1lo, d1hi, 1);
        wgmma_commit();
      }
      wgmma_wait<1>();
      fence_regs(accg);
      __syncwarp();
      if (lane == 0) { mbar_arrive(barW_empty + s0); mbar_arrive(barW_empty + s1); }
      wgmma_wait<0>();
      fence_regs(accp);
      __syncwarp();
      if (lane == 0) { mbar_arrive(barW_empty + s2); mbar_arrive(barW_empty + s3); }
      // ---- epilogue: o = bf16(sigmoid(g) * p) staged as [64 tok][64 ch] (128-B rows, granule ^= row & 7); then out = bf16(z + o) per 16-B granule,
      //      residual granule from the z tile's chunk b in smem; 128-B coalesced global rows.
      if (b > 0) named_bar_sync(bar_id, 128);           // the previous block's vector pass has drained the staging buffer
#pragma unroll
      for (int qp = 0; qp < 4; ++qp) {
        uint32_t r[4];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int q = 2 * qp + e;
          const float oA0 = sigmoidf_(accg[4 * q + 0]) * accp[4 * q + 0], oA1 = sigmoidf_(accg[4 * q + 1]) * accp[4 * q + 1];
          const float oB0 = sigmoidf_(accg[4 * q + 2]) * accp[4 * q + 2], oB1 = sigmoidf_(accg[4 * q + 3]) * accp[4 * q + 3];
          r[2 * e] = pack_bf16(oA0, oA1); r[2 * e + 1] = pack_bf16(oB0, oB1);
        }
        const int gran = 2 * qp + (idx >> 1);
        stsm_x4(stage_u + st_row * 128 + ((gran ^ (st_row & 7)) * 16), r[0], r[1], r[2], r[3]);
      }
      named_bar_sync(bar_id, 128);
#pragma unroll (kK3VecUnroll)
      for (int it = 0; it < 4; ++it) {
        const int row = 16 * wiw + 4 * it + vr;          // WG-relative token row 0..63
        const int trow = 64 * cw + row;                  // tile-relative row (z tile row)
        const int j = jw + row;
        const uint4 ov = lds128(stage_u + row * 128 + ((vg ^ (row & 7)) * 16));
        uint4 zv = make_uint4(0u, 0u, 0u, 0u);
        if (!XIO && p.residual) zv = lds128(sZ_u + b * 16384 + swz128(trow, vg * 16));
        if (iw < p.N && j < p.N) {
          const size_t off = ((size_t)iw * p.N + j) * C + 64 * b + 8 * vg;
          if (XIO && p.residual) zv = *reinterpret_cast<const uint4*>(p.zres + off);   // L2-hot: K1's input
          uint4 w4;
          w4.x = pack_bf16(bf16lo(zv.x) + bf16lo(ov.x), bf16hi(zv.x) + bf16hi(ov.x));
          w4.y = pack_bf16(bf16lo(zv.y) + bf16lo(ov.y), bf16hi(zv.y) + bf16hi(ov.y));
          w4.z = pack_bf16(bf16lo(zv.z) + bf16lo(ov.z), bf16hi(zv.z) + bf16hi(ov.z));
          w4.w = pack_bf16(bf16lo(zv.w) + bf16lo(ov.w), bf16hi(zv.w) + bf16hi(ov.w));
#ifdef TX_K3_NOSTORE
          if (lane == 99)
#endif
          stg128(p.out + off, w4);
        }
      }
      if (!XIO) {
        fence_proxy_async();                             // the residual granules were read from chunk b through the generic proxy
        __syncwarp();
        if (lane == 0) mbar_arrive(barZ_empty + b);     // this warp is done with z chunk b (gate operand pulled earlier, residual read now)
      }
    }
  }
}

template <int BI, int BJ, bool XIO = false, int LNM = 1>
void k3_launch(const K3Params& p, int grid, cudaStream_t st) {
  auto kern = k3_kernel<BI, BJ, XIO, LNM>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, K3_SMEM));
  kern<<<grid, NTHREADS, K3_SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// z: [N, N, 256] bf16 contiguous; mask: [N, N] fp32 or empty; gamma/beta fp32 [256]; wgp: [1024, 256] bf16 (block-interleaved gate|proj rows);
// ab: [512, Np, Np] bf16 (written entirely, pads = 0). tile (BI, BJ): 0 -> (2,64), 1 -> (1,128), 2 -> (4,32); BJ >= 32 keeps each lane's 8-token store granule inside one row
void k1_forward(torch::Tensor z, c10::optional<torch::Tensor> mask, torch::Tensor gamma, torch::Tensor beta, torch::Tensor wgp, torch::Tensor ab,
                double eps, int64_t tile, int64_t grid_limit, int64_t cluster, int64_t do_ln) {
  TORCH_CHECK(z.is_cuda() && z.dtype() == torch::kBFloat16 && z.dim() == 3 && z.size(2) == C && z.size(0) == z.size(1) && z.is_contiguous(), "z must be [N,N,256] bf16 contiguous");
  const int N = z.size(0);
  TORCH_CHECK(ab.dtype() == torch::kBFloat16 && ab.dim() == 3 && ab.size(0) == 2 * C && ab.size(1) == ab.size(2) && ab.is_contiguous(), "ab must be [512,Np,Np] bf16");
  const int Np = ab.size(1);
  TORCH_CHECK(Np >= N && (Np % 16 == 0 || Np == N), "Np must be >= N and a multiple of 16, or exactly N (unpadded planes)");
  TORCH_CHECK(do_ln == 1 || (tile == 0 && cluster != 2), "the no-LayerNorm / stock-order paths are built for tile 0, no cluster");
  TORCH_CHECK(wgp.dtype() == torch::kBFloat16 && wgp.dim() == 2 && wgp.size(0) == 4 * C && wgp.size(1) == C && wgp.is_contiguous(), "wgp must be [1024,256] bf16");
  TORCH_CHECK(gamma.dtype() == torch::kFloat32 && beta.dtype() == torch::kFloat32 && gamma.numel() == C && beta.numel() == C);
  bool has_mask = mask.has_value() && mask->defined() && mask->numel() > 0;
  if (has_mask) TORCH_CHECK(mask->dtype() == torch::kFloat32 && mask->numel() == (int64_t)N * N && mask->is_contiguous(), "mask must be [N,N] fp32 contiguous");
  c10::cuda::CUDAGuard guard(z.device());
  K1Params p;
  int BI, BJ;
  if (tile == 0) { BI = 2; BJ = 64; } else if (tile == 1) { BI = 1; BJ = 128; } else { BI = 4; BJ = 32; }
  { uint64_t dims[3] = {(uint64_t)C, (uint64_t)N, (uint64_t)N}; uint64_t str[2] = {(uint64_t)C * 2, (uint64_t)N * C * 2}; uint32_t box[3] = {64, (uint32_t)BJ, (uint32_t)BI};
    p.tm_z = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, z.data_ptr(), dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B); }
  { uint64_t dims[2] = {(uint64_t)C, (uint64_t)(4 * C)}; uint64_t str[1] = {(uint64_t)C * 2}; uint32_t box[2] = {64, 64};
    p.tm_w = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, wgp.data_ptr(), dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B); }
  p.mask = has_mask ? mask->data_ptr<float>() : nullptr;
  p.gamma = gamma.data_ptr<float>(); p.beta = beta.data_ptr<float>();
  p.ab = reinterpret_cast<__nv_bfloat16*>(ab.data_ptr());
  p.N = N; p.Np = Np; p.eps = (float)eps; p.vec = (Np % 8 == 0 && do_ln != 2) ? 1 : 0;   // do_ln == 2: diagnostic, element stores on any shape
  const int tiles_i = (Np + BI - 1) / BI; p.tiles_j = (Np + BJ - 1) / BJ; p.num_tiles = tiles_i * p.tiles_j;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int cap = sms; if (grid_limit > 0) cap = std::min(cap, (int)grid_limit);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (cluster == 2) {
    const int pairs = (p.num_tiles + 1) / 2;
    int grid = 2 * std::min(pairs, cap / 2);
    if (tile == 0) k1_launch<2, 64, 2>(p, has_mask, grid, st);
    else if (tile == 1) k1_launch<1, 128, 2>(p, has_mask, grid, st);
    else k1_launch<4, 32, 2>(p, has_mask, grid, st);
  } else if (do_ln == 3) {
    int grid = std::min(p.num_tiles, cap);
    k1_launch<2, 64, 1, 2>(p, has_mask, grid, st);
  } else if (do_ln != 1) {
    int grid = std::min(p.num_tiles, cap);
    k1_launch<2, 64, 1, 0>(p, has_mask, grid, st);
  } else {
    int grid = std::min(p.num_tiles, cap);
    if (tile == 0) k1_launch<2, 64, 1>(p, has_mask, grid, st);
    else if (tile == 1) k1_launch<1, 128, 1>(p, has_mask, grid, st);
    else k1_launch<4, 32, 1>(p, has_mask, grid, st);
  }
}

// x: [256, Np, Np] bf16 (contraction result planes); z: [N, N, 256] bf16; wgz: [512, 256] bf16 = cat(linear_g.weight, linear_z.weight); out: [N, N, 256] bf16.
void k3_forward(torch::Tensor x, torch::Tensor z, torch::Tensor gamma_out, torch::Tensor beta_out, torch::Tensor gamma_in, torch::Tensor beta_in,
                torch::Tensor wgz, torch::Tensor out, double eps, int64_t residual, int64_t tile, int64_t grid_limit, int64_t lnmode) {
  TORCH_CHECK(z.is_cuda() && z.dtype() == torch::kBFloat16 && z.dim() == 3 && z.size(2) == C && z.size(0) == z.size(1) && z.is_contiguous(), "z must be [N,N,256] bf16 contiguous");
  const int N = z.size(0);
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 3 && x.size(0) == C && x.size(1) == x.size(2) && x.is_contiguous(), "x must be [256,Np,Np] bf16");
  const int Np = x.size(1);
  TORCH_CHECK(Np >= N && Np % 8 == 0, "Np must be >= N and a multiple of 8 (16-byte plane rows)");
  TORCH_CHECK(lnmode == 1 || ((lnmode == 2 || lnmode == 3) && tile == 0), "lnmode must be 1 (own order), 2 (stock order, N % 4 == 0) or 3 (stock order, N % 4 != 0); 2/3 with tile 0");
  TORCH_CHECK(wgz.dtype() == torch::kBFloat16 && wgz.dim() == 2 && wgz.size(0) == 2 * C && wgz.size(1) == C && wgz.is_contiguous(), "wgz must be [512,256] bf16");
  TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.sizes() == z.sizes() && out.is_contiguous(), "out must be like z");
  TORCH_CHECK(out.data_ptr() != z.data_ptr(), "out must not alias z");
  c10::cuda::CUDAGuard guard(z.device());
  K3Params p;
  int BI, BJ;
  if (tile == 0) { BI = 2; BJ = 64; } else { BI = 1; BJ = 128; }
  { uint64_t dims[3] = {(uint64_t)C, (uint64_t)N, (uint64_t)N}; uint64_t str[2] = {(uint64_t)C * 2, (uint64_t)N * C * 2}; uint32_t box[3] = {64, (uint32_t)BJ, (uint32_t)BI};
    p.tm_z = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, z.data_ptr(), dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B); }
  { uint64_t dims[3] = {(uint64_t)Np, (uint64_t)Np, (uint64_t)C}; uint64_t str[2] = {(uint64_t)Np * 2, (uint64_t)Np * Np * 2}; uint32_t box[3] = {64, 1, (uint32_t)C};
    p.tm_x = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, x.data_ptr(), dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B); }
  { uint64_t dims[2] = {(uint64_t)C, (uint64_t)(2 * C)}; uint64_t str[1] = {(uint64_t)C * 2}; uint32_t box[2] = {64, 64};
    p.tm_w = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, wgz.data_ptr(), dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B); }
  p.gamma_in = gamma_in.data_ptr<float>(); p.beta_in = beta_in.data_ptr<float>(); p.gamma_out = gamma_out.data_ptr<float>(); p.beta_out = beta_out.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.zres = nullptr;
  p.N = N; p.Np = Np; p.eps = (float)eps; p.residual = (int)residual;
  const int tiles_i = (N + BI - 1) / BI; p.tiles_j = (N + BJ - 1) / BJ; p.num_tiles = tiles_i * p.tiles_j;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int grid = std::min(p.num_tiles, sms); if (grid_limit > 0) grid = std::min(grid, (int)grid_limit);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (lnmode == 2) k3_launch<2, 64, false, 2>(p, grid, st);
  else if (lnmode == 3) k3_launch<2, 64, false, 3>(p, grid, st);
  else if (tile == 0) k3_launch<2, 64>(p, grid, st); else k3_launch<1, 128>(p, grid, st);
}


// Exact-tier epilogue: xg [N,N,256] bf16 = LayerNorm_in output (gate source), xp [N,N,256] bf16 = LayerNorm_out output in token-major layout (projection
// source), zres [N,N,256] bf16 = the module input (residual), wgz [512,256] bf16; out [N,N,256] = zres + bf16(sigmoid(xg Wg^T) * (xp Wz^T)) (residual) or the update.
void k3x_forward(torch::Tensor xg, torch::Tensor xp, torch::Tensor zres, torch::Tensor wgz, torch::Tensor out, int64_t residual, int64_t tile, int64_t grid_limit) {
  auto chk = [&](const torch::Tensor& t, const char* nm) {
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kBFloat16 && t.dim() == 3 && t.size(2) == C && t.size(0) == t.size(1) && t.is_contiguous(), nm, " must be [N,N,256] bf16 contiguous"); };
  chk(xg, "xg"); chk(xp, "xp"); chk(zres, "zres"); chk(out, "out");
  const int N = xg.size(0);
  TORCH_CHECK(xp.size(0) == N && zres.size(0) == N && out.size(0) == N, "xg/xp/zres/out must share N");
  TORCH_CHECK(wgz.dtype() == torch::kBFloat16 && wgz.dim() == 2 && wgz.size(0) == 2 * C && wgz.size(1) == C && wgz.is_contiguous(), "wgz must be [512,256] bf16");
  c10::cuda::CUDAGuard guard(xg.device());
  K3Params p;
  int BI, BJ;
  if (tile == 0) { BI = 2; BJ = 64; } else { BI = 1; BJ = 128; }
  { uint64_t dims[3] = {(uint64_t)C, (uint64_t)N, (uint64_t)N}; uint64_t str[2] = {(uint64_t)C * 2, (uint64_t)N * C * 2}; uint32_t box[3] = {64, (uint32_t)BJ, (uint32_t)BI};
    p.tm_z = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, xg.data_ptr(), dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B);
    p.tm_x = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, xp.data_ptr(), dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B); }
  { uint64_t dims[2] = {(uint64_t)C, (uint64_t)(2 * C)}; uint64_t str[1] = {(uint64_t)C * 2}; uint32_t box[2] = {64, 64};
    p.tm_w = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, wgz.data_ptr(), dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B); }
  p.gamma_in = p.beta_in = p.gamma_out = p.beta_out = nullptr;
  p.zres = reinterpret_cast<const __nv_bfloat16*>(zres.data_ptr()); p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.N = N; p.Np = N; p.eps = 0.f; p.residual = (int)residual;
  const int tiles_i = (N + BI - 1) / BI; p.tiles_j = (N + BJ - 1) / BJ; p.num_tiles = tiles_i * p.tiles_j;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int grid = std::min(p.num_tiles, sms); if (grid_limit > 0) grid = std::min(grid, (int)grid_limit);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (tile == 0) k3_launch<2, 64, true>(p, grid, st); else k3_launch<1, 128, true>(p, grid, st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k1_forward", &k1_forward, "TriMul prologue (LN_in + gated dual projection -> channel-major planes)");
  m.def("k3_forward", &k3_forward, "TriMul epilogue (LN_out + output projection * sigmoid gate + residual)");
  m.def("k3x_forward", &k3x_forward, "TriMul epilogue on LayerNorm outputs (output projection * sigmoid gate + residual), exact-tier path");
}
