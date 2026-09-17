// edm_softmax.cu.cc — EdmRelShiftSoftmax: the attention weights of one Enformer attention block in one kernel (built by build.sh with
// nvcc; no TensorFlow headers here).
//
//   probs[b,h,i,j] = softmax over j of ( content[b,h,i,j] + rel[b,h,i,(L-1)+j-i] ),   L = 1536 columns per row
//
// = attention_module.relative_shift + the logits AddV2 (the shift expressed as an index: out[b,h,i,j] = content[b,h,i,j] + rel[b,h,i,(L-1)+j-i]) followed by TensorFlow 2.17.1's
// GPU "Softmax" kernel sequence (tensorflow/core/kernels/softmax_op_gpu.cu.cc) on rows of L = 1536 columns, operation by operation:
//
//   1. l_j = content_j + rel_(L-1+j-i)                        add.rn.ftz                      (AddV2)
//   2. m   = row maximum of l, cub::Max = (b > a) ? b : a     in CUB's segmented-reduce order (below), init -FLT_MAX
//                                                             (DoRowReduction<gpuprim::Max>: rows of >= 1024 columns take
//                                                              cub::DeviceSegmentedReduce, reduction_gpu_kernels.cu.h LaunchRowReduction)
//   3. e_j = expf(l_j - m)                                    sub.rn.ftz, libdevice expf      (SubtractAndExpFunctor)
//   4. s   = row sum of e, cub::Sum                           in the same CUB order, init 0   (DoRowReduction<gpuprim::Sum>)
//   5. probs_j = expf(l_j - m) / s                            div.rn.ftz (IEEE)               (GenerateNormalizedProb<float, float, 4>,
//                                                              in_log_space = false; it recomputes e_j: the same value)
//
// The CUB order (CUB 2.2.0, the version TensorFlow 2.17.1 is built with; DeviceSegmentedReduceKernel, Policy600: one block of 256 threads
// per row, 16 items per thread, so a 1536-column row is a single partial tile — agent_reduce.cuh ConsumeRange / ConsumeTile<IS_FIRST_TILE>):
//   a. thread t (0..255) owns columns t, t+256, t+512, t+768, t+1024, t+1280 and folds them left to right:
//      a_t = op(op(op(op(op(x[t], x[t+256]), x[t+512]), x[t+768]), x[t+1024]), x[t+1280])          op(agg, item)
//   b. BlockReduceWarpReductions: in each warp, shuffle-down steps of offset 1, 2, 4, 8, 16, lane l taking op(own, lane l+offset) while
//      l + offset <= 31 — lane 0 ends with the balanced tree over its 32 lanes. For float cub::Sum the step is CUB's inline
//      `add.f32 peer, own` (warp_reduce_shfl.cuh, no flush modifier); for cub::Max the generic step op(own, peer).
//   c. thread 0 folds the 8 warp aggregates left to right: g = op(...op(op(w0, w1), w2)..., w7)   (ApplyWarpAggregates)
//   d. result = op(init, g): 0 + g for the sum, (g > -FLT_MAX) ? g : -FLT_MAX for the max      (finalize_and_store_aggregate)
// cub::Sum's and cub::Max's operator() are plain C++ `a + b` / `(b > a) ? b : a` compiled, like all of TensorFlow's CUDA kernels, with
// flush-to-zero: add.rn.ftz / setp.gt.ftz. Every operation is written with the named instructions of edm_f32.cuh, so which thread computes
// what is the only freedom taken: here every thread folds the 8 warp aggregates itself (step c) instead of reading thread 0's result —
// the same operations in the same order.
#include <cuda_runtime.h>
#include <cfloat>
#include <stdint.h>

#include "edm_f32.cuh"
#include "edm_ops.h"

namespace edm {
namespace {

using namespace ::edm::f32;

__device__ __forceinline__ float cub_max(float a, float b) {         // cub::Max()(a, b) = CUB_MAX(a, b) = (b > a) ? b : a: greater-than (setp.gt.ftz, as TensorFlow
                                                                       // compiles it) then select — an opaque predicate the compiler cannot turn into max.f32
  int p; asm("{ .reg .pred q; setp.gt.ftz.f32 q, %1, %2; selp.s32 %0, 1, 0, q; }" : "=r"(p) : "f"(b), "f"(a));
  return p ? b : a;
}

constexpr int kSoftmaxThreads = 256;                                   // CUB Policy600 BLOCK_THREADS
constexpr int kSoftmaxWarps = kSoftmaxThreads / 32;

// One block per row (b, h, i); K = L / 256 items per thread (L = 1536: K = 6).
template <int K>
__global__ void __launch_bounds__(kSoftmaxThreads) RelShiftSoftmaxKernel(const float* content, const float* __restrict__ rel, float* out) {
  // out may alias content (the op forwards its first input): thread t loads columns t + 256 k of its row into registers (step 1) and later
  // stores exactly those columns (step 5); no thread reads an address another thread writes.
  constexpr int L = K * kSoftmaxThreads;
  constexpr int W = 2 * L - 1;
  __shared__ float wmax[kSoftmaxWarps];
  __shared__ float wsum[kSoftmaxWarps];
  const long long row = blockIdx.x;
  const int i = (int)(row % L);
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const float* c = content + row * (long long)L;
  const float* r = rel + row * (long long)W + (L - 1 - i);
  float* o = out + row * (long long)L;

  // 1. the logits this thread owns: columns t + 256 k
  float l[K];
#pragma unroll
  for (int k = 0; k < K; ++k) l[k] = add(c[t + k * kSoftmaxThreads], r[t + k * kSoftmaxThreads]);

  // 2. row max in CUB's order: a. per thread, b. warp tree, c. across warps, d. op(init, agg)
  float a = l[0];
#pragma unroll
  for (int k = 1; k < K; ++k) a = cub_max(a, l[k]);
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const float peer = __shfl_down_sync(0xffffffffu, a, off);
    if (lane + off <= 31) a = cub_max(a, peer);
  }
  if (lane == 0) wmax[warp] = a;
  __syncthreads();
  float m = wmax[0];
#pragma unroll
  for (int w = 1; w < kSoftmaxWarps; ++w) m = cub_max(m, wmax[w]);
  m = cub_max(-FLT_MAX, m);                                            // op(init, agg), init = Eigen::NumTraits<float>::lowest() = -FLT_MAX

  // 3. e_j = expf(l_j - m); 4. row sum in CUB's order
  float e[K];
#pragma unroll
  for (int k = 0; k < K; ++k) e[k] = expf_libdevice(sub(l[k], m));
  a = e[0];
#pragma unroll
  for (int k = 1; k < K; ++k) a = add(a, e[k]);                         // reduction_op(agg, item) = agg + item
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const float peer = __shfl_down_sync(0xffffffffu, a, off);
    if (lane + off <= 31) a = add_nonflushing(peer, a);                // CUB's float-sum step: add.f32 r0(peer), own
  }
  if (lane == 0) wsum[warp] = a;
  __syncthreads();
  float s = wsum[0];
#pragma unroll
  for (int w = 1; w < kSoftmaxWarps; ++w) s = add(s, wsum[w]);
  s = add(0.0f, s);                                                    // op(init, agg), init = 0

  // 5. probs_j = e_j / s
#pragma unroll
  for (int k = 0; k < K; ++k) o[t + k * kSoftmaxThreads] = div(e[k], s);
}

}  // namespace

static_assert(kRelShiftSoftmaxCols % kSoftmaxThreads == 0 && kRelShiftSoftmaxCols >= 1024 && kRelShiftSoftmaxCols < 16 * kSoftmaxThreads,
              "the order above is CUB's for rows of 1024..4095 columns (TensorFlow's segmented-reduce path, one partial tile)");

cudaError_t LaunchRelShiftSoftmax(cudaStream_t stream, const float* content, const float* rel, float* out, int64_t rows, int L) {
  if (rows <= 0) return cudaSuccess;
  if (rows >= (1LL << 31) || L != kRelShiftSoftmaxCols) return cudaErrorInvalidValue;
  RelShiftSoftmaxKernel<kRelShiftSoftmaxCols / kSoftmaxThreads><<<dim3((unsigned)rows), dim3(kSoftmaxThreads), 0, stream>>>(content, rel, out);
  return cudaGetLastError();
}

}  // namespace edm
