// edm_ops.cu.cc — CUDA kernels of the edm ops and their launchers (built by build.sh with nvcc; no TensorFlow headers here).
//
// Each device function below is TensorFlow 2.17.1's own GPU float32 arithmetic for one Enformer subgraph, operation by operation and in
// the same order, spelled with the named instructions of edm_f32.cuh. Where TensorFlow's source admits more than one reading, every
// reading is a template variant; variant 0 is the reading its kernels execute (README.md lists the sequences).
#include <cuda_runtime.h>
#include <stdint.h>

#include "edm_f32.cuh"
#include "edm_ops.h"

namespace edm {
namespace {

using namespace ::edm::f32;

// ---------------------------------------------------------------------------------------------------------------------------------
// Sigmoid — the GPU float32 kernel of tf.nn.sigmoid ("Sigmoid", T=float) is TensorFlow's MLIR-generated kernel
// (tensorflow/core/kernels/mlir_generated/gpu_op_sigmoid.cc:22; the Eigen registration is compiled out for float,
// tensorflow/core/kernels/cwise_op_sigmoid.cc:22-28): mhlo.logistic (op_definitions/sigmoid.mlir.tmpl:6) lowered as
// 1 / (1 + exp(-x)) = negate, exponential, add 1.0, divide 1.0 by the sum (xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h:1079-1101),
// compiled for the device with flush-to-zero, fma contraction allowed (kernel_gen/transforms/gpu_kernel_to_blob_pass.cc:159-168) and
// -nvptx-prec-divf32=1 (xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc:540), under which LLVM's NVPTX backend emits a general a / b as the
// approximate div.full.ftz.f32 but the reciprocal 1.0f / d as rcp.rn.ftz.f32 — the correctly rounded reciprocal.
//   variant 0: e = expf(-x); d = e + 1; y = rcp.rn(d)
//   variant 1: the same with exp written out (expf_steps) and its final multiply contracted into the add: d = fma(p, 2^n, 1); y = rcp.rn(d)
//              (equal to variant 0 for every x: scaling ex2's result by 2^n is exact wherever the sum rounds differently from 1)
//   variant 2: as variant 0 with the special-function-unit reciprocal rcp.approx.ftz (LLVM's choice under -nvptx-prec-divf32=0)
template <int V>
__device__ __forceinline__ float edm_sigmoid(float x) {
  if (V == 1) {
    float two_n;
    float p = expf_steps_scale(neg(x), &two_n);
    return rcp(fma(p, two_n, 1.0f));
  }
  float e = expf_libdevice(neg(x));
  float d = add(e, 1.0f);
  return V == 2 ? rcp_approx(d) : rcp(d);
}

// Enformer's GELU (enformer.py:310 gelu: tf.nn.sigmoid(1.702 * x) * x) — three TensorFlow kernels, per element:
//   a = 1.702 * v             Mul    (1.702 -> float32 0x3FD9DB23)
//   s = sigmoid(a)            Sigmoid
//   y = s * v                 Mul
template <int V>
__device__ __forceinline__ float gelu(float v) {
  float a = mul(__int_as_float(0x3FD9DB23), v);
  float s = edm_sigmoid<V>(a);
  return mul(s, v);
}

// EdmScaleShiftGelu — per element, c = the channel (last, contiguous axis):
//   v = x * scale[c]          Mul    (tf.nn.batch_normalization, tensorflow/python/ops/nn_impl.py:1487: x * inv ...)
//   v = v + shift[c]          AddV2  (... + (offset - mean * inv), nn_impl.py:1487-1488; Sonnet BatchNorm eval path, batch_norm.py:186-192)
//   y = gelu(v)               Mul, Sigmoid, Mul (above)
template <int V>
__device__ __forceinline__ float scale_shift_gelu(float x, float scale, float shift) {
  return gelu<V>(add(mul(x, scale), shift));
}

template <int V>
__global__ void __launch_bounds__(256) ScaleShiftGeluKernel(const float* x, const float* __restrict__ scale,
                                                            const float* __restrict__ shift, float* y, int64_t rows, int C) {
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const float* xr = x + row * C;
    float* yr = y + row * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) yr[c] = scale_shift_gelu<V>(xr[c], scale[c], shift[c]);
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// EdmSoftmaxPool2 — enformer.py SoftmaxPooling1D (:276-284) with pool_size 2, given the logits its Linear produced:
//   tf.nn.softmax(logits, axis=-2) transposes the pool axis last (nn_ops.py _wrap_2d_function) and runs the "Softmax" GPU kernel
//   (tensorflow/core/kernels/softmax_op_gpu.cu.cc) on rows of 2 columns, one warp per row (reduction_gpu_kernels.cu.h:203-239, :719-731;
//   lane 0 holds column 0, its shuffle peer is column 1):
//     m  = row max by cub::Max: CUB_MAX(own, peer) = (l1 > l0) ? l1 : l0        (cub/util_macro.cuh:77; warp_reduce_shfl.cuh:346-361)
//     e_i = exp(l_i - m)                                                       (SubtractAndExpFunctor, softmax_op_gpu.cu.cc:150-155)
//     s  = row sum by cub::Sum, whose float warp step is `add.f32 peer, own`    (warp_reduce_shfl.cuh:188-210; no flush modifier)
//     w_i = exp(l_i - m) / s, IEEE division                                    (GenerateNormalizedProb, softmax_op_gpu.cu.cc:82)
//   then  p_i = x_i * w_i     Mul                                              (enformer.py:283: inputs * softmax(...))
//   and   y = p_0 + p_1       Sum over the pool axis: op(in[row 0], in[row 1]) (ColumnReduceSimpleKernel, reduction_gpu_kernels.cu.h:447-448, :1089-1097)
//   variant 0: exp = libdevice expf, division = div.rn ; variant 1: exp written out (expf_steps) ; variant 2: division = div.full (the
//   approximate division an MLIR-generated kernel would use).
template <int V>
__device__ __forceinline__ float softmax_pool2(float x0, float x1, float l0, float l1) {
  float m = (l1 > l0) ? l1 : l0;
  float e0 = V == 1 ? expf_steps(sub(l0, m)) : expf_libdevice(sub(l0, m));
  float e1 = V == 1 ? expf_steps(sub(l1, m)) : expf_libdevice(sub(l1, m));
  float s = add_nonflushing(e1, e0);
  float w0 = V == 2 ? div_full(e0, s) : div_softmax(e0, s);   // GenerateNormalizedProb recomputes exp(l_i - m): the same value as e_i; div_softmax = div's
  float w1 = V == 2 ? div_full(e1, s) : div_softmax(e1, s);   // result (edm_f32.cuh) without the division's slow path on the pair's losing side (e_i +0 or tiny)
  float p0 = mul(x0, w0);
  float p1 = mul(x1, w1);
  return add(p0, p1);
}

template <int V>
__global__ void __launch_bounds__(256) SoftmaxPool2Kernel(const float* __restrict__ x, const float* __restrict__ logits,
                                                          float* __restrict__ y, int64_t P, int C) {
  for (int64_t p = blockIdx.x; p < P; p += gridDim.x) {
    const float* x0 = x + 2 * p * C;
    const float* l0 = logits + 2 * p * C;
    float* yp = y + p * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) yp[c] = softmax_pool2<V>(x0[c], x0[C + c], l0[c], l0[C + c]);
  }
}

constexpr int kThreads = 256;
inline unsigned grid_for(int64_t units) { return (unsigned)(units < 1 ? 1 : (units > 2147483647LL ? 2147483647LL : units)); }


// ---- EdmBiasResidual: ASSOC 0: out = res + (y + bias)  — a BiasAdd (or an Add behind a Reshape) then the residual AddV2, as written;
//                       ASSOC 1: out = (res + y) + bias  — an Add feeding the residual AddV2 directly: TensorFlow's graph optimizer regroups that
//                       add tree by shape (the two activations first, the broadcast bias last), and this is the order the stock graph executes.
template <int ASSOC>
__device__ __forceinline__ float bias_residual(float y, float b, float res) { return ASSOC == 0 ? f32::add(res, f32::add(y, b)) : f32::add(f32::add(res, y), b); }
template <int ASSOC>
__global__ void __launch_bounds__(256) BiasResidualKernel(const float* __restrict__ y, const float* __restrict__ bias, const float* __restrict__ res,
                                                          float* __restrict__ out, long long rows, int C) {
  for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
    const float* yr = y + row * C; const float* rr = res + row * C; float* orow = out + row * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) orow[c] = bias_residual<ASSOC>(yr[c], bias[c], rr[c]);
  }
}

// ---- The same arithmetic with a neighbouring TensorFlow kernel taken into the pass (variant 0 throughout: the orderings TensorFlow executes).
// Each fused op (and EdmBiasResidual) has two kernels with the same per-element operation sequence: a row kernel (any shape / alignment, the layout
// of the kernels above) and a float4 kernel used when C % 4 == 0 and every pointer is 16-byte aligned — one thread per 4 consecutive channels of one row,
// 128-bit loads and stores, the four elements' operation chains independent (the launchers choose; `c4` = the thread's channel group, kept
// without a division per step: it advances by the grid stride modulo C/4 — a step only taken when the tensor has more float4 groups than the
// grid has threads, i.e. above 2^31-1 blocks of 256).
struct Channel4 { int c4, step, C4; __device__ __forceinline__ void next() { c4 += step; if (c4 >= C4) c4 -= C4; } };
__device__ __forceinline__ Channel4 channel4(int64_t i, int64_t stride, int C4) { return Channel4{(int)(i % C4), (int)(stride % C4), C4}; }
__device__ __forceinline__ float4 ld4(const float* __restrict__ v, int c4) { return reinterpret_cast<const float4*>(v)[c4]; }

// EdmBiasScaleShiftGelu: y = scale_shift_gelu(h) with h = x + bias[c] — a convolution's BiasAdd (bias_op_gpu.cu.cc BiasNHWCKernel: one
// add.rn.ftz per element) computed in registers in front of the BatchNorm -> GELU chain; h itself is never stored.
__device__ __forceinline__ float bias_scale_shift_gelu(float x, float b, float scale, float shift) { return scale_shift_gelu<0>(add(x, b), scale, shift); }
__global__ void __launch_bounds__(256) BiasScaleShiftGeluKernel(const float* x, const float* __restrict__ bias, const float* __restrict__ scale,
                                                                const float* __restrict__ shift, float* y, int64_t rows, int C) {
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const float* xr = x + row * C;
    float* yr = y + row * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) yr[c] = bias_scale_shift_gelu(xr[c], bias[c], scale[c], shift[c]);
  }
}
__global__ void __launch_bounds__(256) BiasScaleShiftGelu4Kernel(const float4* x, const float* __restrict__ bias, const float* __restrict__ scale,
                                                                 const float* __restrict__ shift, float4* y, int64_t n4, int C4) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  for (Channel4 ch = channel4(i, stride, C4); i < n4; i += stride, ch.next()) {
    const float4 v = x[i], b = ld4(bias, ch.c4), sc = ld4(scale, ch.c4), sh = ld4(shift, ch.c4);
    float4 o;
    o.x = bias_scale_shift_gelu(v.x, b.x, sc.x, sh.x); o.y = bias_scale_shift_gelu(v.y, b.y, sc.y, sh.y);
    o.z = bias_scale_shift_gelu(v.z, b.z, sc.z, sh.z); o.w = bias_scale_shift_gelu(v.w, b.w, sc.w, sh.w);
    y[i] = o;
  }
}
// EdmBiasGelu: y = gelu(x + bias[c]) — a BiasAdd followed by the GELU alone (no BatchNorm in between: the final pointwise block's output).
__global__ void __launch_bounds__(256) BiasGeluKernel(const float* x, const float* __restrict__ bias, float* y, int64_t rows, int C) {
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const float* xr = x + row * C;
    float* yr = y + row * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) yr[c] = gelu<0>(add(xr[c], bias[c]));
  }
}
// EdmSoftmaxPool2Gelu: y = scale_shift_gelu(p) with p = softmax_pool2(...) — the pooled value goes straight into the next block's
// BatchNorm -> GELU chain in registers; p itself is never stored. Output element (p, c) reads input rows 2p and 2p+1 at channel c.
__device__ __forceinline__ float softmax_pool2_gelu(float x0, float x1, float l0, float l1, float scale, float shift) {
  return scale_shift_gelu<0>(softmax_pool2<0>(x0, x1, l0, l1), scale, shift);
}
__global__ void __launch_bounds__(256) SoftmaxPool2GeluKernel(const float* __restrict__ x, const float* __restrict__ logits, const float* __restrict__ scale,
                                                              const float* __restrict__ shift, float* __restrict__ y, int64_t P, int C) {
  for (int64_t p = blockIdx.x; p < P; p += gridDim.x) {
    const float* x0 = x + 2 * p * C;
    const float* l0 = logits + 2 * p * C;
    float* yp = y + p * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) yp[c] = softmax_pool2_gelu(x0[c], x0[(int64_t)C + c], l0[c], l0[(int64_t)C + c], scale[c], shift[c]);
  }
}
__global__ void __launch_bounds__(256) SoftmaxPool2Gelu4Kernel(const float4* __restrict__ x, const float4* __restrict__ logits, const float* __restrict__ scale,
                                                               const float* __restrict__ shift, float4* __restrict__ y, int64_t n4, int C4) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  for (Channel4 ch = channel4(i, stride, C4); i < n4; i += stride, ch.next()) {
    const int64_t j = 2 * i - ch.c4;                                   // float4 index of (row 2p, channel group c4): 2*(p*C4) + c4 with i = p*C4 + c4
    const float4 x0 = x[j], x1 = x[j + C4], l0 = logits[j], l1 = logits[j + C4], sc = ld4(scale, ch.c4), sh = ld4(shift, ch.c4);
    float4 o;
    o.x = softmax_pool2_gelu(x0.x, x1.x, l0.x, l1.x, sc.x, sh.x); o.y = softmax_pool2_gelu(x0.y, x1.y, l0.y, l1.y, sc.y, sh.y);
    o.z = softmax_pool2_gelu(x0.z, x1.z, l0.z, l1.z, sc.z, sh.z); o.w = softmax_pool2_gelu(x0.w, x1.w, l0.w, l1.w, sc.w, sh.w);
    y[i] = o;
  }
}
// EdmBias2Residual: out = (res + res_bias) + (y + bias) — EdmBiasResidual's assoc-0 order with the residual input's own BiasAdd (the block's
// first convolution's) recomputed in registers instead of read back from a stored tensor.
__device__ __forceinline__ float bias2_residual(float y, float b, float res, float rb) { return f32::add(f32::add(res, rb), f32::add(y, b)); }
__global__ void __launch_bounds__(256) Bias2ResidualKernel(const float* y, const float* __restrict__ bias, const float* res,
                                                           const float* __restrict__ res_bias, float* out, int64_t rows, int C) {   // out may alias y or res
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const float* yr = y + row * C; const float* rr = res + row * C; float* orow = out + row * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) orow[c] = bias2_residual(yr[c], bias[c], rr[c], res_bias[c]);
  }
}
__global__ void __launch_bounds__(256) Bias2Residual4Kernel(const float4* y, const float* __restrict__ bias, const float4* res, const float* __restrict__ res_bias,
                                                            float4* out, int64_t n4, int C4) {                                        // out may alias y or res
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  for (Channel4 ch = channel4(i, stride, C4); i < n4; i += stride, ch.next()) {
    const float4 v = y[i], r = res[i], b = ld4(bias, ch.c4), rb = ld4(res_bias, ch.c4);
    float4 o;
    o.x = bias2_residual(v.x, b.x, r.x, rb.x); o.y = bias2_residual(v.y, b.y, r.y, rb.y);
    o.z = bias2_residual(v.z, b.z, r.z, rb.z); o.w = bias2_residual(v.w, b.w, r.w, rb.w);
    out[i] = o;
  }
}
// EdmBiasResidual's float4 kernel (the row kernel and its two association orders are above).
template <int ASSOC>
__global__ void __launch_bounds__(256) BiasResidual4Kernel(const float4* __restrict__ y, const float* __restrict__ bias, const float4* __restrict__ res,
                                                           float4* __restrict__ out, int64_t n4, int C4) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  for (Channel4 ch = channel4(i, stride, C4); i < n4; i += stride, ch.next()) {
    const float4 v = y[i], r = res[i], b = ld4(bias, ch.c4);
    float4 o;
    o.x = bias_residual<ASSOC>(v.x, b.x, r.x); o.y = bias_residual<ASSOC>(v.y, b.y, r.y);
    o.z = bias_residual<ASSOC>(v.z, b.z, r.z); o.w = bias_residual<ASSOC>(v.w, b.w, r.w);
    out[i] = o;
  }
}

// The float4 kernels serve C % 4 == 0 with every pointer 16-byte aligned (TensorFlow's GPU allocations are; a sliced input may not be).
template <typename... P>
inline bool by4(int C, P... ptrs) { return C % 4 == 0 && (((reinterpret_cast<uintptr_t>(ptrs) & 15) == 0) && ...); }
inline unsigned grid_for4(int64_t n4) { return grid_for((n4 + kThreads - 1) / kThreads); }
inline const float4* f4(const float* p) { return reinterpret_cast<const float4*>(p); }
inline float4* f4(float* p) { return reinterpret_cast<float4*>(p); }

}  // namespace

cudaError_t LaunchScaleShiftGelu(cudaStream_t stream, const float* x, const float* scale, const float* shift, float* y,
                                 int64_t rows, int C, int variant) {
  if (rows <= 0 || C <= 0) return cudaSuccess;
  dim3 grid(grid_for(rows)), block(kThreads);
  switch (variant) {
    case 0: ScaleShiftGeluKernel<0><<<grid, block, 0, stream>>>(x, scale, shift, y, rows, C); break;
    case 1: ScaleShiftGeluKernel<1><<<grid, block, 0, stream>>>(x, scale, shift, y, rows, C); break;
    case 2: ScaleShiftGeluKernel<2><<<grid, block, 0, stream>>>(x, scale, shift, y, rows, C); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t LaunchSoftmaxPool2(cudaStream_t stream, const float* x, const float* logits, float* y, int64_t P, int C, int variant) {
  if (P <= 0 || C <= 0) return cudaSuccess;
  dim3 grid(grid_for(P)), block(kThreads);
  switch (variant) {
    case 0: SoftmaxPool2Kernel<0><<<grid, block, 0, stream>>>(x, logits, y, P, C); break;
    case 1: SoftmaxPool2Kernel<1><<<grid, block, 0, stream>>>(x, logits, y, P, C); break;
    case 2: SoftmaxPool2Kernel<2><<<grid, block, 0, stream>>>(x, logits, y, P, C); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t LaunchBiasScaleShiftGelu(cudaStream_t stream, const float* x, const float* bias, const float* scale, const float* shift, float* y,
                                     int64_t rows, int C) {
  if (rows <= 0 || C <= 0) return cudaSuccess;
  if (by4(C, x, bias, scale, shift, y)) {
    const int64_t n4 = rows * (C / 4);
    BiasScaleShiftGelu4Kernel<<<dim3(grid_for4(n4)), dim3(kThreads), 0, stream>>>(f4(x), bias, scale, shift, f4(y), n4, C / 4);
  } else {
    BiasScaleShiftGeluKernel<<<dim3(grid_for(rows)), dim3(kThreads), 0, stream>>>(x, bias, scale, shift, y, rows, C);
  }
  return cudaGetLastError();
}

cudaError_t LaunchBiasGelu(cudaStream_t stream, const float* x, const float* bias, float* y, int64_t rows, int C) {
  if (rows <= 0 || C <= 0) return cudaSuccess;
  BiasGeluKernel<<<dim3(grid_for(rows)), dim3(kThreads), 0, stream>>>(x, bias, y, rows, C);
  return cudaGetLastError();
}

cudaError_t LaunchSoftmaxPool2Gelu(cudaStream_t stream, const float* x, const float* logits, const float* scale, const float* shift, float* y,
                                   int64_t P, int C) {
  if (P <= 0 || C <= 0) return cudaSuccess;
  if (by4(C, x, logits, scale, shift, y)) {
    const int64_t n4 = P * (C / 4);
    SoftmaxPool2Gelu4Kernel<<<dim3(grid_for4(n4)), dim3(kThreads), 0, stream>>>(f4(x), f4(logits), scale, shift, f4(y), n4, C / 4);
  } else {
    SoftmaxPool2GeluKernel<<<dim3(grid_for(P)), dim3(kThreads), 0, stream>>>(x, logits, scale, shift, y, P, C);
  }
  return cudaGetLastError();
}

cudaError_t LaunchBias2Residual(cudaStream_t stream, const float* y, const float* bias, const float* res, const float* res_bias, float* out, int64_t rows, int C) {
  if (rows <= 0 || C <= 0) return cudaSuccess;
  if (by4(C, y, bias, res, res_bias, out)) {
    const int64_t n4 = rows * (C / 4);
    Bias2Residual4Kernel<<<dim3(grid_for4(n4)), dim3(kThreads), 0, stream>>>(f4(y), bias, f4(res), res_bias, f4(out), n4, C / 4);
  } else {
    Bias2ResidualKernel<<<dim3(grid_for(rows)), dim3(kThreads), 0, stream>>>(y, bias, res, res_bias, out, rows, C);
  }
  return cudaGetLastError();
}

cudaError_t LaunchBiasResidual(cudaStream_t stream, const float* y, const float* bias, const float* res, float* out, int64_t rows, int C, int assoc) {
  if (rows <= 0 || C <= 0) return cudaSuccess;
  if (assoc != 0 && assoc != 1) return cudaErrorInvalidValue;
  if (by4(C, y, bias, res, out)) {
    const int64_t n4 = rows * (C / 4);
    dim3 grid(grid_for4(n4)), block(kThreads);
    if (assoc == 0) BiasResidual4Kernel<0><<<grid, block, 0, stream>>>(f4(y), bias, f4(res), f4(out), n4, C / 4);
    else BiasResidual4Kernel<1><<<grid, block, 0, stream>>>(f4(y), bias, f4(res), f4(out), n4, C / 4);
  } else {
    dim3 grid(grid_for(rows)), block(kThreads);
    if (assoc == 0) BiasResidualKernel<0><<<grid, block, 0, stream>>>(y, bias, res, out, rows, C);
    else BiasResidualKernel<1><<<grid, block, 0, stream>>>(y, bias, res, out, rows, C);
  }
  return cudaGetLastError();
}

}  // namespace edm
