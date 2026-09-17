// edm_ops.h — the boundary between the TensorFlow side (edm_ops.cc, edm_softmax.cc, edm_biasact.cc: op registration, shape functions, GPU OpKernels) and the CUDA side
// (edm_ops.cu.cc, edm_pool_logits.cu.cc, edm_softmax.cu.cc, edm_layernorm.cu.cc, edm_biasact.cu.cc: kernels and their launchers). The launchers take raw device pointers and the op's CUDA stream; they enqueue one kernel
// and return the launch status. `variant` selects one of the arithmetic orderings written in edm_ops.cu.cc (0 = the ordering that
// reproduces TensorFlow 2.17.1's kernels; the others are the documented alternatives, kept selectable without a rebuild).
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace edm {

constexpr int kScaleShiftGeluVariants = 3;   // edm_ops.cu.cc: edm_sigmoid<0|1|2>
constexpr int kSoftmaxPool2Variants = 3;     // edm_ops.cu.cc: softmax_pool2<0|1|2>

// y[r, c] = gelu(x[r, c] * scale[c] + shift[c]) for r < rows, c < C (x, y: rows*C contiguous floats, channels last; y may alias x).
cudaError_t LaunchScaleShiftGelu(cudaStream_t stream, const float* x, const float* scale, const float* shift, float* y,
                                 int64_t rows, int C, int variant);

// y[p, c] = softmax-weighted sum of the pair (x[2p, c], x[2p+1, c]) with weights softmax(logits[2p, c], logits[2p+1, c]) for p < P, c < C
// (x, logits: 2*P*C contiguous floats; y: P*C).
cudaError_t LaunchSoftmaxPool2(cudaStream_t stream, const float* x, const float* logits, float* y, int64_t P, int C, int variant);

// Y[m, n] = sum_k X[m, k] * W[k, n] for m < M - T (float32, the stock pooling-logits kernel's accumulation order: ops/README.md); rows
// M-T .. M-1 of Y are copied from tail (T*N floats). X: M*K, W: K*N, Y: M*N contiguous floats. K = N in {768 .. 1536}, multiples of 128.
cudaError_t LaunchPoolLogits(cudaStream_t stream, const float* X, const float* W, float* Y, int M, int K, int N, const float* tail, int T);

// out = content + shift(rel): content rows*L floats laid out [rows/L blocks of L rows][L], rel [rows][2L-1]; out[row, j] = content[row, j] +
// rel[row, (L-1) + j - (row % L)].
// assoc 0: out[r, c] = res[r, c] + (y[r, c] + bias[c]); assoc 1: out[r, c] = (res[r, c] + y[r, c]) + bias[c]; over rows x C.
cudaError_t LaunchBiasResidual(cudaStream_t stream, const float* y, const float* bias, const float* res, float* out, int64_t rows, int C, int assoc);
// y[r, :] = LayerNorm over the N columns of row r of x (rows x N contiguous floats; gamma, beta: N floats; 1024 <= N < 4096): TensorFlow's
// Mean / SquaredDifference / Mean / batch_normalization arithmetic in its GPU kernels' order (edm_layernorm.cu.cc, ops/README.md).
cudaError_t LaunchLayerNorm(cudaStream_t stream, const float* x, const float* gamma, const float* beta, float* y, int64_t rows, int N, float epsilon);

// probs[row, j] = softmax over j of (content[row, j] + rel[row, (L-1) + j - (row % L)]) for rows = B*H*L rows of L = kRelShiftSoftmaxCols
// columns (content, probs: rows*L floats, probs may alias content; rel: rows*(2L-1)), TensorFlow's Softmax arithmetic (edm_softmax.cu.cc).
constexpr int kRelShiftSoftmaxCols = 1536;
cudaError_t LaunchRelShiftSoftmax(cudaStream_t stream, const float* content, const float* rel, float* out, int64_t rows, int L);

// The same passes with a neighbouring bias add / BatchNorm -> GELU chain taken in (variant 0 arithmetic; ops/README.md):
// y[r, c] = gelu((x[r, c] + bias[c]) * scale[c] + shift[c])
cudaError_t LaunchBiasScaleShiftGelu(cudaStream_t stream, const float* x, const float* bias, const float* scale, const float* shift, float* y,
                                     int64_t rows, int C);
// y[r, c] = gelu(x[r, c] + bias[c])
cudaError_t LaunchBiasGelu(cudaStream_t stream, const float* x, const float* bias, float* y, int64_t rows, int C);
// y[p, c] = gelu(softmax_pool2(pair p, channel c) * scale[c] + shift[c])
cudaError_t LaunchSoftmaxPool2Gelu(cudaStream_t stream, const float* x, const float* logits, const float* scale, const float* shift, float* y,
                                   int64_t P, int C);
// out[r, c] = (res[r, c] + res_bias[c]) + (y[r, c] + bias[c])
cudaError_t LaunchBias2Residual(cudaStream_t stream, const float* y, const float* bias, const float* res, const float* res_bias, float* out, int64_t rows, int C);

// y[r, c] = act(x[r, c] + bias[c]) over rows x C, act one of the Act codes: TensorFlow's Add then its Relu / Softplus (edm_biasact.cu.cc, ops/README.md).
enum Act : int { kActRelu = 0, kActSoftplus = 1 };
cudaError_t LaunchBiasAct(cudaStream_t stream, const float* x, const float* bias, float* y, int64_t rows, int C, int act);

// q: (B, H, T, K) contiguous, T*K <= 2^31-1, B*H <= 65535; bias_w, bias_r: H*K floats; qw = q * scale + bias_w[h, k], qr = q * scale + bias_r[h, k]
// (edm_biasact.cu.cc).
cudaError_t LaunchQScaleBias(cudaStream_t stream, const float* q, float scale, const float* bias_w, const float* bias_r, float* qw, float* qr,
                             int64_t B, int H, int64_t T, int K);

}  // namespace edm
