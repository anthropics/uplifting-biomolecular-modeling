// edm_layernorm.cu.cc — EdmLayerNorm: Sonnet's LayerNorm over the last axis (tf.nn.moments + tf.nn.batch_normalization) for rows of
// 1024 <= N < 4096 columns, one 256-thread block per row, with TensorFlow 2.17.1's own GPU float32 arithmetic (README.md §EdmLayerNorm):
//
//   mean = Mean(x, -1)                       TensorFlow's row reduction for N >= 1024: cub::DeviceSegmentedReduce (CUDA 12.3's CUB 2.2.0),
//   var  = Mean(SquaredDifference(x, mean))  one 256-thread block per row, whose summation ORDER is reproduced below step by step:
//            thread t:  a_t = ((x[t] + x[t+256]) + x[t+512]) + ...            items t, t+256, ... < N, left to right   (agent_reduce.cuh, partial tile)
//            warp w:    lanes l = 0..31 hold a_{32w+l}; five shuffle-down steps, offsets 1, 2, 4, 8, 16: v = v_own + v_peer
//                       (warp_reduce_shfl.cuh ReduceStep: lane 0 ends with the balanced tree of the 32 values)
//            block:     S = ((((((W0 + W1) + W2) + W3) + W4) + W5) + W6) + W7                              (ApplyWarpAggregates)
//            store:     out = init + S = 0 + S                                                          (finalize_and_store_aggregate)
//            Mean:      mean = out / N, IEEE division                                                   (DividesBy<float>: x / divisor)
//          every + is TensorFlow's own Sum<float> functor, `a + b` compiled in TensorFlow's flush-to-zero build: add.rn.ftz.f32 (CUB's generic
//          shuffle step calls the functor; CUB's non-flushing add.f32 shuffle specialisation is for cub::Sum only and does not apply here).
//   SquaredDifference(x, m) = (x - m) * (x - m)          sub.rn.ftz, mul.rn.ftz   (MLIR kernel: chlo.broadcast_subtract, chlo.broadcast_multiply)
//   a    = var + epsilon                                 add.rn.ftz               (AddV2)
//   rs   = Rsqrt(a)                                      rsqrt.approx.ftz.f32     (MLIR kernel: mhlo.rsqrt -> __nv_rsqrtf)
//   inv  = rs * gamma[c]                                 mul.rn.ftz               (Mul, broadcast to the full row)
//   y    = x * inv + (beta[c] - mean * inv)             mul, mul, sub, add — four roundings (Mul, Mul, Sub, AddV2)
#include <cuda_runtime.h>
#include <stdint.h>

#include "edm_f32.cuh"
#include "edm_ops.h"

namespace edm {
namespace {

using namespace ::edm::f32;

constexpr int kLnThreads = 256;                     // CUB 2.2.0 DeviceReducePolicy::Policy600: 256 threads, 16 items per thread (tile 4096)
constexpr int kLnWarps = kLnThreads / 32;
constexpr int kLnMaxItems = 16;                     // N < 4096: at most 16 items per thread, the single partial tile of the segmented reduce

// The block-wide sum of one value per thread in the order CUB's BlockReduce<float, 256, BLOCK_REDUCE_WARP_REDUCTIONS>::Reduce computes it
// for a full block (all 256 lanes valid) with TensorFlow's Sum<float>: valid in thread 0 only. `warp_smem` holds kLnWarps floats.
__device__ __forceinline__ float block_sum_tf(float v, float* warp_smem) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {                   // WarpReduceShfl::ReduceStep, STEP 0..4: peer = shfl.down(v, offset)
    const float peer = __shfl_down_sync(0xffffffffu, v, offset);
    if (offset + lane <= 31) v = add(v, peer);                        // output = reduction_op(input, peer) where the peer lane exists
  }
  if (lane == 0) warp_smem[warp] = v;
  __syncthreads();
  float s = v;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int w = 1; w < kLnWarps; ++w) s = add(s, warp_smem[w]);      // ApplyWarpAggregates: warp 0's aggregate, then + W1 ... + W7 in turn
  }
  return s;
}

// ITEMS = the items each thread holds. EXACT: N == ITEMS * kLnThreads — every item present, no per-item bounds test (the launcher picks this
// form for Enformer's N = 1536: ITEMS = 6); any other N takes ITEMS = kLnMaxItems with item k present when t + 256k < N. The same
// operations in the same order either way: the template only removes the tests and the registers of absent items.
template <int ITEMS, bool EXACT>
__global__ void __launch_bounds__(kLnThreads) LayerNormKernel(const float* x, const float* __restrict__ gamma, const float* __restrict__ beta,
                                                           float* y, int N, float n_f, float epsilon) {   // y may be x: a row's items are all read before any is written
  __shared__ float warp_smem[kLnWarps];
  __shared__ float row_stat[2];                                       // mean, rsqrt(var + epsilon)
  const long long row = blockIdx.x;
  const float* xr = x + row * (long long)N;
  float* yr = y + row * (long long)N;
  const int t = threadIdx.x;
  auto present = [&](int k) { return EXACT || t + k * kLnThreads < N; };

  float v[ITEMS];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) v[k] = present(k) ? xr[t + k * kLnThreads] : 0.0f;

  // mean: thread-sequential partial over the thread's items, then the block tree; 0 + S; S / N
  float a = v[0];                                                     // N >= 1024 > 256: every thread holds at least one item
#pragma unroll
  for (int k = 1; k < ITEMS; ++k) if (present(k)) a = add(a, v[k]);
  float s = block_sum_tf(a, warp_smem);
  if (t == 0) row_stat[0] = div(add(0.0f, s), n_f);
  __syncthreads();
  const float mean = row_stat[0];

  // variance: the same reduction over SquaredDifference(x, mean) = (x - mean) * (x - mean)
  float d = sub(v[0], mean);
  a = mul(d, d);
#pragma unroll
  for (int k = 1; k < ITEMS; ++k) if (present(k)) { d = sub(v[k], mean); a = add(a, mul(d, d)); }
  s = block_sum_tf(a, warp_smem);                                     // warp_smem reuse is safe: every thread passed the barrier above after thread 0 read it
  if (t == 0) {
    const float var = div(add(0.0f, s), n_f);
    row_stat[1] = rsqrt_approx(add(var, epsilon));
  }
  __syncthreads();
  const float rs = row_stat[1];

  // y = x * inv + (beta - mean * inv), inv = rs * gamma[c]
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    const int c = t + k * kLnThreads;
    if (present(k)) {
      const float inv = mul(rs, gamma[c]);
      yr[c] = add(mul(v[k], inv), sub(beta[c], mul(mean, inv)));
    }
  }
}

template <int ITEMS>
void launch_exact(cudaStream_t stream, const float* x, const float* gamma, const float* beta, float* y, unsigned rows, int N, float epsilon) {
  LayerNormKernel<ITEMS, true><<<dim3(rows), dim3(kLnThreads), 0, stream>>>(x, gamma, beta, y, N, (float)N, epsilon);
}

}  // namespace

cudaError_t LaunchLayerNorm(cudaStream_t stream, const float* x, const float* gamma, const float* beta, float* y, int64_t rows, int N,
                            float epsilon) {
  if (rows <= 0) return cudaSuccess;
  if (rows >= (1LL << 31) || N < 1024 || N >= kLnThreads * kLnMaxItems) return cudaErrorInvalidValue;
  const unsigned r = (unsigned)rows;
  if (N == 6 * kLnThreads) launch_exact<6>(stream, x, gamma, beta, y, r, N, epsilon);   // Enformer's N = 1536: every item present, no bounds tests
  else LayerNormKernel<kLnMaxItems, false><<<dim3(r), dim3(kLnThreads), 0, stream>>>(x, gamma, beta, y, N, (float)N, epsilon);
  return cudaGetLastError();
}

}  // namespace edm
