// edm_biasact.cu.cc — CUDA kernels of EdmBiasAct and EdmQScaleBias and their launchers (built by build.sh with nvcc, the flags of edm_ops.cu.cc).
// The arithmetic is TensorFlow 2.17.1's GPU float32 arithmetic for the replaced subgraphs, operation by operation, in the vocabulary of
// edm_f32.cuh.
#include <cuda_runtime.h>
#include <stdint.h>

#include "edm_f32.cuh"
#include "edm_ops.h"

namespace edm {
namespace {

// Relu — the GPU float32 kernel of tf.nn.relu ("Relu", T=float) is TensorFlow's MLIR-generated kernel (tensorflow/core/kernels/mlir_generated/
// gpu_op_relu.cc; relu_op.cc:108-132 compiles the Eigen registration out): tf.Relu legalizes to chlo.broadcast_maximum(0, x)
// (tensorflow/compiler/mlir/tf2xla/transforms/legalize_tf_patterns.td, "Relu op patterns"), i.e. arith.maximumf -> llvm.maximum.f32 — the
// NaN-propagating maximum that orders -0 below +0 — which the NVPTX backend emits for sm_80 and newer as one instruction, max.NaN.f32
// (flush-to-zero like every float32 operation of those kernels).
__device__ __forceinline__ float relu(float v) { float r; asm("max.NaN.ftz.f32 %0, %1, 0f00000000;" : "=f"(r) : "f"(v)); return r; }

// Softplus — the GPU float32 kernel of tf.nn.softplus ("Softplus", T=float) is TensorFlow's MLIR-generated kernel (mlir_generated/
// gpu_op_softplus.cc; softplus_op.cc:110-130 compiles the Eigen registration out). tf.Softplus legalizes (tensorflow/compiler/mlir/tf2xla/
// transforms/legalize_tf.cc, ConvertSoftplusOp) to: t = log(eps) + 2, eps = 2^-23 (float32 epsilon); e = exp(x);
// y = x > -t ? x : (x < t ? e : log1p(e)) — exp, log, log1p being libdevice's __nv_expf, __nv_logf, __nv_log1pf (the GPUToNVVM lowering of
// math.exp / math.log / math.log1p), the routines nvcc links here, under the same flush-to-zero libdevice configuration. __nv_logf and
// __nv_log1pf (CUDA 12.3) are sequences of correctly rounded fma / add / mul and integer operations with no division and no multiply-add pair
// a compiler may contract, so both compilers produce the same instruction sequence.
__device__ __forceinline__ float softplus_threshold() { return f32::add(logf(__int_as_float(0x34000000)), 2.0f); }   // log(2^-23) + 2
__device__ __forceinline__ float softplus(float v, float t) {
  const float e = f32::expf_libdevice(v);
  return v > f32::neg(t) ? v : v < t ? e : log1pf(e);
}

// EdmBiasAct — per element, c = the channel (last, contiguous axis):
//   v = x + bias[c]           Add (Sonnet Linear: tf.add(matmul, b), sonnet/src/linear.py:85-87 -> the "Add"/"AddV2" GPU kernel: one add.rn.ftz; a
//                             channels-last "BiasAdd", bias_op_gpu.cu.cc:60-67, is the same add)
//   y = act(v)                relu: Relu (above); softplus: Softplus (above)
template <int ACT>
__global__ void __launch_bounds__(256) BiasActKernel(const float* x, const float* __restrict__ bias, float* y, int64_t rows, int C) {
  const float t = ACT == kActSoftplus ? softplus_threshold() : 0.0f;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const float* xr = x + row * C;
    float* yr = y + row * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
      const float v = f32::add(xr[c], bias[c]);
      yr[c] = ACT == kActRelu ? relu(v) : softplus(v, t);
    }
  }
}

// EdmQScaleBias — the attention query after its head transpose, q: (B, H, T, K) contiguous; per element with head h and key index k:
//   m  = q * scale            Mul    (attention_module.py MultiheadAttention.__call__: q *= self._key_size**-0.5; scale is the graph's float32 constant)
//   qw = m + bias_w[h, k]     AddV2  (q + self._r_w_bias, feeding the content logits)
//   qr = m + bias_r[h, k]     AddV2  (q + self._r_r_bias, feeding the relative-position logits)
// One (b, h) slab of T*K floats per blockIdx.y; the blocks along x cover the slab, one element per thread.
__global__ void __launch_bounds__(256) QScaleBiasKernel(const float* __restrict__ q, float scale, const float* __restrict__ bias_w, const float* __restrict__ bias_r,
                                                        float* __restrict__ qw, float* __restrict__ qr, int H, int TK, int K) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= TK) return;
  const int h = blockIdx.y % H, k = (int)(i % K);
  const long long e = (long long)blockIdx.y * TK + i;
  const float m = f32::mul(q[e], scale);
  qw[e] = f32::add(m, bias_w[h * K + k]);
  qr[e] = f32::add(m, bias_r[h * K + k]);
}

constexpr int kThreads = 256;
inline unsigned grid_for(int64_t units) { return (unsigned)(units < 1 ? 1 : (units > 2147483647LL ? 2147483647LL : units)); }

}  // namespace

cudaError_t LaunchBiasAct(cudaStream_t stream, const float* x, const float* bias, float* y, int64_t rows, int C, int act) {
  if (rows <= 0 || C <= 0) return cudaSuccess;
  dim3 grid(grid_for(rows)), block(kThreads);
  switch (act) {
    case kActRelu: BiasActKernel<kActRelu><<<grid, block, 0, stream>>>(x, bias, y, rows, C); break;
    case kActSoftplus: BiasActKernel<kActSoftplus><<<grid, block, 0, stream>>>(x, bias, y, rows, C); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t LaunchQScaleBias(cudaStream_t stream, const float* q, float scale, const float* bias_w, const float* bias_r, float* qw, float* qr,
                             int64_t B, int H, int64_t T, int K) {
  if (B <= 0 || H <= 0 || T <= 0 || K <= 0) return cudaSuccess;
  if (T * K > 2147483647LL || B * H > 65535) return cudaErrorInvalidValue;
  const int64_t TK = T * K;
  dim3 grid((unsigned)((TK + kThreads - 1) / kThreads), (unsigned)(B * H)), block(kThreads);
  QScaleBiasKernel<<<grid, block, 0, stream>>>(q, scale, bias_w, bias_r, qw, qr, H, (int)TK, K);
  return cudaGetLastError();
}

}  // namespace edm
