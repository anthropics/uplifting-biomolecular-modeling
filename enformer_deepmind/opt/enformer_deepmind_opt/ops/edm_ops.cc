// edm_ops.cc — TensorFlow side of the edm ops: op registration, shape functions and the GPU OpKernels (built by build.sh with g++ against
// the TensorFlow headers of the installed wheel). The ops have GPU kernels only: placing one on the CPU fails with TensorFlow's own
// "no registered kernel" error. All tensors are float32, channels-last and contiguous; the kernels run on the op's own CUDA stream.
//
//   EdmScaleShiftGelu(x, scale, shift) -> y      x: (..., C), scale: (C), shift: (C); y = gelu(x * scale + shift), Enformer's gelu
//   EdmSoftmaxPool2(x, logits) -> y              x, logits: (B, L, C) with L even -> y: (B, L/2, C); or (B, P, 2, C) -> (B, P, C)
//   EdmPoolLogits(x, w, tail) -> logits          x: (M, K), w: (K, N) -> (M, N): the pooling logits GEMM (edm_pool_logits.cu.cc); tail: the rows the
//                                                stock library computes apart, passed through
//   EdmBiasResidual(y, bias, res; assoc) -> out  res + (y + bias) (assoc 0) or (res + y) + bias (assoc 1)
//   EdmLayerNorm(x, gamma, beta; epsilon) -> y  x: (..., N), gamma, beta: (N); Sonnet LayerNorm over the last axis, 1024 <= N < 4096 (edm_layernorm.cu.cc)
//   EdmBiasScaleShiftGelu(x, bias, scale, shift) -> y   gelu((x + bias) * scale + shift): a convolution's bias add taken into the pass
//   EdmBiasGelu(x, bias) -> y                           gelu(x + bias)
//   EdmSoftmaxPool2Gelu(x, logits, scale, shift) -> y   gelu(EdmSoftmaxPool2(x, logits) * scale + shift), the pooled tensor never stored
//   EdmBias2Residual(y, bias, res, res_bias) -> out      (res + res_bias) + (y + bias)
//   EdmRelShiftSoftmax lives in edm_softmax.cc / edm_softmax.cu.cc.
//   EdmScaleShiftGelu, EdmSoftmaxPool2: attr arith_variant (int, default 0): which arithmetic ordering of edm_ops.cu.cc runs; 0 reproduces
//   TensorFlow 2.17.1's kernels; the fused ops compute ordering 0 only (ops/README.md).
#define EIGEN_USE_GPU
#include <cuda_runtime.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "edm_ops.h"

namespace edm {
namespace {

using ::tensorflow::DEVICE_GPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;
namespace errors = ::tensorflow::errors;

Status ScaleShiftGeluShape(InferenceContext* c) {
  ShapeHandle x;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &x));
  ShapeHandle scale, shift;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scale));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shift));
  DimensionHandle ch;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(scale, 0), c->Dim(shift, 0), &ch));
  if (!c->RankKnown(x)) {
    c->set_output(0, c->UnknownShape());
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(x, -1), ch, &ch));
  ShapeHandle y;
  TF_RETURN_IF_ERROR(c->ReplaceDim(x, c->Rank(x) - 1, ch, &y));
  c->set_output(0, y);
  return absl::OkStatus();
}

Status SoftmaxPool2Shape(InferenceContext* c) {
  ShapeHandle x;
  TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &x));
  if (!c->RankKnown(x)) {
    c->set_output(0, c->UnknownShape());
    return absl::OkStatus();
  }
  const int rank = c->Rank(x);
  if (rank == 3) {          // (B, L, C) -> (B, L/2, C)
    DimensionHandle half;
    TF_RETURN_IF_ERROR(c->Divide(c->Dim(x, 1), 2, /*evenly_divisible=*/true, &half));
    c->set_output(0, c->MakeShape({c->Dim(x, 0), half, c->Dim(x, 2)}));
    return absl::OkStatus();
  }
  if (rank == 4) {          // (B, P, 2, C) -> (B, P, C)
    DimensionHandle two;
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(x, 2), 2, &two));
    c->set_output(0, c->MakeShape({c->Dim(x, 0), c->Dim(x, 1), c->Dim(x, 3)}));
    return absl::OkStatus();
  }
  return errors::InvalidArgument("EdmSoftmaxPool2: x must be (B, L, C) or (B, P, 2, C); got rank ", rank);
}

// x (..., C) with `n` per-channel vectors (C) as inputs first .. first+n-1 -> x's shape (the fused elementwise ops).
Status ChannelwiseShape(InferenceContext* c, int first, int n) {
  ShapeHandle x;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &x));
  DimensionHandle ch = c->RankKnown(x) ? c->Dim(x, -1) : c->UnknownDim();
  for (int i = first; i < first + n; ++i) {
    ShapeHandle v;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &v));
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(v, 0), ch, &ch));
  }
  if (!c->RankKnown(x)) {
    c->set_output(0, c->UnknownShape());
    return absl::OkStatus();
  }
  ShapeHandle y;
  TF_RETURN_IF_ERROR(c->ReplaceDim(x, c->Rank(x) - 1, ch, &y));
  c->set_output(0, y);
  return absl::OkStatus();
}

// The checks every per-channel vector input gets in Compute (name = the op's, what = the input's).
#define EDM_REQUIRE_VECTOR(ctx, t, C, name, what)                                                                                  \
  OP_REQUIRES(ctx, (t).dims() == 1 && (t).dim_size(0) == (C),                                                                      \
              errors::InvalidArgument(name ": " what " must be (C) with C = the activation's last dimension ", (C), "; got ", (t).shape().DebugString()))

class EdmScaleShiftGeluOp : public OpKernel {
 public:
  explicit EdmScaleShiftGeluOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("arith_variant", &variant_));
    OP_REQUIRES(c, variant_ >= 0 && variant_ < kScaleShiftGeluVariants,
                errors::InvalidArgument("EdmScaleShiftGelu: arith_variant must be in [0, ", kScaleShiftGeluVariants, "); got ", variant_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& scale = ctx->input(1);
    const Tensor& shift = ctx->input(2);
    OP_REQUIRES(ctx, x.dims() >= 1, errors::InvalidArgument("EdmScaleShiftGelu: x must have rank >= 1"));
    const int64_t C = x.dim_size(x.dims() - 1);
    OP_REQUIRES(ctx, scale.dims() == 1 && scale.dim_size(0) == C,
                errors::InvalidArgument("EdmScaleShiftGelu: scale must be (C) with C = x's last dimension ", C, "; got ",
                                        scale.shape().DebugString()));
    OP_REQUIRES(ctx, shift.dims() == 1 && shift.dim_size(0) == C,
                errors::InvalidArgument("EdmScaleShiftGelu: shift must be (C) with C = x's last dimension ", C, "; got ",
                                        shift.shape().DebugString()));
    OP_REQUIRES(ctx, C <= 2147483647LL, errors::InvalidArgument("EdmScaleShiftGelu: C too large: ", C));
    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, x.shape(), &y));
    if (x.NumElements() == 0) return;
    const int64_t rows = x.NumElements() / C;
    const cudaError_t err = LaunchScaleShiftGelu(ctx->eigen_device<Eigen::GpuDevice>().stream(), x.flat<float>().data(),
                                                 scale.flat<float>().data(), shift.flat<float>().data(), y->flat<float>().data(),
                                                 rows, static_cast<int>(C), variant_);
    OP_REQUIRES(ctx, err == cudaSuccess, errors::Internal("EdmScaleShiftGelu: kernel launch failed: ", cudaGetErrorString(err)));
  }

 private:
  int variant_ = 0;
};

class EdmSoftmaxPool2Op : public OpKernel {
 public:
  explicit EdmSoftmaxPool2Op(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("arith_variant", &variant_));
    OP_REQUIRES(c, variant_ >= 0 && variant_ < kSoftmaxPool2Variants,
                errors::InvalidArgument("EdmSoftmaxPool2: arith_variant must be in [0, ", kSoftmaxPool2Variants, "); got ", variant_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& logits = ctx->input(1);
    OP_REQUIRES(ctx, x.shape() == logits.shape(),
                errors::InvalidArgument("EdmSoftmaxPool2: x and logits must have the same shape; got ", x.shape().DebugString(),
                                        " and ", logits.shape().DebugString()));
    OP_REQUIRES(ctx, x.dims() == 3 || x.dims() == 4,
                errors::InvalidArgument("EdmSoftmaxPool2: x must be (B, L, C) or (B, P, 2, C); got ", x.shape().DebugString()));
    TensorShape out;
    int64_t P, C;
    if (x.dims() == 3) {
      OP_REQUIRES(ctx, x.dim_size(1) % 2 == 0,
                  errors::InvalidArgument("EdmSoftmaxPool2: L must be even for pool size 2; got ", x.shape().DebugString()));
      P = x.dim_size(0) * (x.dim_size(1) / 2);
      C = x.dim_size(2);
      out = TensorShape({x.dim_size(0), x.dim_size(1) / 2, C});
    } else {
      OP_REQUIRES(ctx, x.dim_size(2) == 2,
                  errors::InvalidArgument("EdmSoftmaxPool2: the pool axis (axis 2) must have size 2; got ", x.shape().DebugString()));
      P = x.dim_size(0) * x.dim_size(1);
      C = x.dim_size(3);
      out = TensorShape({x.dim_size(0), x.dim_size(1), C});
    }
    OP_REQUIRES(ctx, C <= 2147483647LL, errors::InvalidArgument("EdmSoftmaxPool2: C too large: ", C));
    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out, &y));
    if (y->NumElements() == 0) return;
    const cudaError_t err = LaunchSoftmaxPool2(ctx->eigen_device<Eigen::GpuDevice>().stream(), x.flat<float>().data(),
                                               logits.flat<float>().data(), y->flat<float>().data(), P, static_cast<int>(C), variant_);
    OP_REQUIRES(ctx, err == cudaSuccess, errors::Internal("EdmSoftmaxPool2: kernel launch failed: ", cudaGetErrorString(err)));
  }

 private:
  int variant_ = 0;
};

}  // namespace


class EdmPoolLogitsOp : public OpKernel {
 public:
  explicit EdmPoolLogitsOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0); const Tensor& w = ctx->input(1); const Tensor& t = ctx->input(2);
    OP_REQUIRES(ctx, x.dims() == 2 && w.dims() == 2 && t.dims() == 2 && x.dim_size(1) == w.dim_size(0) && t.dim_size(1) == w.dim_size(1),
                errors::InvalidArgument("EdmPoolLogits: x [M,K], w [K,N], tail [T,N] expected"));
    const int64_t M = x.dim_size(0), K = x.dim_size(1), N = w.dim_size(1), T = t.dim_size(0);
    OP_REQUIRES(ctx, M < (1LL << 31) && K % 128 == 0 && K >= 512 && N % 128 == 0 && T <= M,
                errors::InvalidArgument("EdmPoolLogits: unsupported shape (M < 2^31, K and N multiples of 128, K >= 512, T <= M)"));
    Tensor* y = nullptr; OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({M, N}), &y));
    if (M == 0) return;
    const Eigen::GpuDevice& d = ctx->eigen_device<Eigen::GpuDevice>();
    cudaError_t e = LaunchPoolLogits(d.stream(), x.flat<float>().data(), w.flat<float>().data(), y->flat<float>().data(), (int)M, (int)K, (int)N,
                                         T > 0 ? t.flat<float>().data() : nullptr, (int)T);
    OP_REQUIRES(ctx, e == cudaSuccess, errors::Internal("EdmPoolLogits launch: ", cudaGetErrorString(e)));
  }
};


class EdmBiasResidualOp : public OpKernel {
 public:
  explicit EdmBiasResidualOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("assoc", &assoc_));
    OP_REQUIRES(c, assoc_ == 0 || assoc_ == 1, errors::InvalidArgument("EdmBiasResidual: assoc must be 0 or 1; got ", assoc_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& y = ctx->input(0); const Tensor& bias = ctx->input(1); const Tensor& res = ctx->input(2);
    OP_REQUIRES(ctx, y.shape() == res.shape() && bias.dims() == 1 && y.dims() >= 1 && y.dim_size(y.dims() - 1) == bias.dim_size(0),
                errors::InvalidArgument("EdmBiasResidual: y and res of one shape [..., C] and bias [C] expected; got ", y.shape().DebugString(), ", ", bias.shape().DebugString(), ", ", res.shape().DebugString()));
    Tensor* out = nullptr; OP_REQUIRES_OK(ctx, ctx->allocate_output(0, y.shape(), &out));
    const int64_t C = bias.dim_size(0), rows = C ? y.NumElements() / C : 0;
    if (rows == 0) return;
    const cudaError_t e = LaunchBiasResidual(ctx->eigen_device<Eigen::GpuDevice>().stream(), y.flat<float>().data(), bias.flat<float>().data(), res.flat<float>().data(), out->flat<float>().data(), rows, static_cast<int>(C), assoc_);
    OP_REQUIRES(ctx, e == cudaSuccess, errors::Internal("EdmBiasResidual launch: ", cudaGetErrorString(e)));
  }
 private:
  int assoc_ = 0;
};

class EdmLayerNormOp : public OpKernel {
 public:
  explicit EdmLayerNormOp(OpKernelConstruction* c) : OpKernel(c) { OP_REQUIRES_OK(c, c->GetAttr("epsilon", &epsilon_)); }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0); const Tensor& gamma = ctx->input(1); const Tensor& beta = ctx->input(2);
    OP_REQUIRES(ctx, x.dims() >= 1, errors::InvalidArgument("EdmLayerNorm: x must have rank >= 1"));
    const int64_t N = x.dim_size(x.dims() - 1);
    OP_REQUIRES(ctx, gamma.dims() == 1 && beta.dims() == 1 && gamma.dim_size(0) == N && beta.dim_size(0) == N,
                errors::InvalidArgument("EdmLayerNorm: gamma and beta must be (N) with N = x's last dimension ", N, "; got ", gamma.shape().DebugString(), " and ", beta.shape().DebugString()));
    OP_REQUIRES(ctx, N >= 1024 && N < 4096, errors::InvalidArgument("EdmLayerNorm: the normalized axis must have 1024 <= N < 4096 columns (the rows TensorFlow reduces with one 256-thread block each); got ", N));
    Tensor* y = nullptr; OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, x.shape(), &y));
    const int64_t rows = x.NumElements() / N;
    if (rows == 0) return;
    OP_REQUIRES(ctx, rows < (1LL << 31), errors::InvalidArgument("EdmLayerNorm: too many rows: ", rows));
    const cudaError_t e = LaunchLayerNorm(ctx->eigen_device<Eigen::GpuDevice>().stream(), x.flat<float>().data(), gamma.flat<float>().data(), beta.flat<float>().data(),
                                          y->flat<float>().data(), rows, (int)N, epsilon_);
    OP_REQUIRES(ctx, e == cudaSuccess, errors::Internal("EdmLayerNorm launch: ", cudaGetErrorString(e)));
  }
 private:
  float epsilon_ = 0.f;
};

// ---- the fused elementwise ops (ordering 0 only) ------------------------------------------------------------------------------------
class EdmBiasScaleShiftGeluOp : public OpKernel {
 public:
  explicit EdmBiasScaleShiftGeluOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0); const Tensor& bias = ctx->input(1); const Tensor& scale = ctx->input(2); const Tensor& shift = ctx->input(3);
    OP_REQUIRES(ctx, x.dims() >= 1, errors::InvalidArgument("EdmBiasScaleShiftGelu: x must have rank >= 1"));
    const int64_t C = x.dim_size(x.dims() - 1);
    EDM_REQUIRE_VECTOR(ctx, bias, C, "EdmBiasScaleShiftGelu", "bias"); EDM_REQUIRE_VECTOR(ctx, scale, C, "EdmBiasScaleShiftGelu", "scale");
    EDM_REQUIRE_VECTOR(ctx, shift, C, "EdmBiasScaleShiftGelu", "shift");
    OP_REQUIRES(ctx, C <= 2147483647LL, errors::InvalidArgument("EdmBiasScaleShiftGelu: C too large: ", C));
    Tensor* y = nullptr; OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, x.shape(), &y));
    if (x.NumElements() == 0) return;
    const cudaError_t e = LaunchBiasScaleShiftGelu(ctx->eigen_device<Eigen::GpuDevice>().stream(), x.flat<float>().data(), bias.flat<float>().data(),
                                                   scale.flat<float>().data(), shift.flat<float>().data(), y->flat<float>().data(), x.NumElements() / C, static_cast<int>(C));
    OP_REQUIRES(ctx, e == cudaSuccess, errors::Internal("EdmBiasScaleShiftGelu launch: ", cudaGetErrorString(e)));
  }
};

class EdmBiasGeluOp : public OpKernel {
 public:
  explicit EdmBiasGeluOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0); const Tensor& bias = ctx->input(1);
    OP_REQUIRES(ctx, x.dims() >= 1, errors::InvalidArgument("EdmBiasGelu: x must have rank >= 1"));
    const int64_t C = x.dim_size(x.dims() - 1);
    EDM_REQUIRE_VECTOR(ctx, bias, C, "EdmBiasGelu", "bias");
    OP_REQUIRES(ctx, C <= 2147483647LL, errors::InvalidArgument("EdmBiasGelu: C too large: ", C));
    Tensor* y = nullptr; OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, x.shape(), &y));
    if (x.NumElements() == 0) return;
    const cudaError_t e = LaunchBiasGelu(ctx->eigen_device<Eigen::GpuDevice>().stream(), x.flat<float>().data(), bias.flat<float>().data(), y->flat<float>().data(),
                                         x.NumElements() / C, static_cast<int>(C));
    OP_REQUIRES(ctx, e == cudaSuccess, errors::Internal("EdmBiasGelu launch: ", cudaGetErrorString(e)));
  }
};

class EdmSoftmaxPool2GeluOp : public OpKernel {
 public:
  explicit EdmSoftmaxPool2GeluOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0); const Tensor& logits = ctx->input(1); const Tensor& scale = ctx->input(2); const Tensor& shift = ctx->input(3);
    OP_REQUIRES(ctx, x.shape() == logits.shape(),
                errors::InvalidArgument("EdmSoftmaxPool2Gelu: x and logits must have the same shape; got ", x.shape().DebugString(), " and ", logits.shape().DebugString()));
    OP_REQUIRES(ctx, (x.dims() == 3 && x.dim_size(1) % 2 == 0) || (x.dims() == 4 && x.dim_size(2) == 2),
                errors::InvalidArgument("EdmSoftmaxPool2Gelu: x must be (B, L, C) with L even, or (B, P, 2, C); got ", x.shape().DebugString()));
    const int64_t C = x.dim_size(x.dims() - 1);
    const int64_t P = x.dims() == 3 ? x.dim_size(0) * (x.dim_size(1) / 2) : x.dim_size(0) * x.dim_size(1);
    const TensorShape out = x.dims() == 3 ? TensorShape({x.dim_size(0), x.dim_size(1) / 2, C}) : TensorShape({x.dim_size(0), x.dim_size(1), C});
    EDM_REQUIRE_VECTOR(ctx, scale, C, "EdmSoftmaxPool2Gelu", "scale"); EDM_REQUIRE_VECTOR(ctx, shift, C, "EdmSoftmaxPool2Gelu", "shift");
    OP_REQUIRES(ctx, C <= 2147483647LL, errors::InvalidArgument("EdmSoftmaxPool2Gelu: C too large: ", C));
    Tensor* y = nullptr; OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out, &y));
    if (y->NumElements() == 0) return;
    const cudaError_t e = LaunchSoftmaxPool2Gelu(ctx->eigen_device<Eigen::GpuDevice>().stream(), x.flat<float>().data(), logits.flat<float>().data(),
                                                 scale.flat<float>().data(), shift.flat<float>().data(), y->flat<float>().data(), P, static_cast<int>(C));
    OP_REQUIRES(ctx, e == cudaSuccess, errors::Internal("EdmSoftmaxPool2Gelu launch: ", cudaGetErrorString(e)));
  }
};

class EdmBias2ResidualOp : public OpKernel {
 public:
  explicit EdmBias2ResidualOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& y = ctx->input(0); const Tensor& bias = ctx->input(1); const Tensor& res = ctx->input(2); const Tensor& res_bias = ctx->input(3);
    OP_REQUIRES(ctx, y.shape() == res.shape() && y.dims() >= 1,
                errors::InvalidArgument("EdmBias2Residual: y and res of one shape [..., C] expected; got ", y.shape().DebugString(), " and ", res.shape().DebugString()));
    const int64_t C = y.dim_size(y.dims() - 1);
    EDM_REQUIRE_VECTOR(ctx, bias, C, "EdmBias2Residual", "bias"); EDM_REQUIRE_VECTOR(ctx, res_bias, C, "EdmBias2Residual", "res_bias");
    OP_REQUIRES(ctx, C <= 2147483647LL, errors::InvalidArgument("EdmBias2Residual: C too large: ", C));
    Tensor* out = nullptr; OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0, 2}, 0, y.shape(), &out));
    if (y.NumElements() == 0) return;
    const cudaError_t e = LaunchBias2Residual(ctx->eigen_device<Eigen::GpuDevice>().stream(), y.flat<float>().data(), bias.flat<float>().data(), res.flat<float>().data(),
                                              res_bias.flat<float>().data(), out->flat<float>().data(), y.NumElements() / C, static_cast<int>(C));
    OP_REQUIRES(ctx, e == cudaSuccess, errors::Internal("EdmBias2Residual launch: ", cudaGetErrorString(e)));
  }
};

REGISTER_OP("EdmScaleShiftGelu")
    .Input("x: float")
    .Input("scale: float")
    .Input("shift: float")
    .Output("y: float")
    .Attr("arith_variant: int = 0")
    .SetShapeFn(ScaleShiftGeluShape)
    .Doc(R"doc(
gelu(x * scale + shift) with Enformer's gelu (sigmoid(1.702 v) v), computed with TensorFlow 2.17.1's own GPU float32 arithmetic:
Mul, AddV2, Mul, Sigmoid, Mul in that order, each rounded once. x: (..., C) channels last; scale, shift: (C). GPU only.
)doc");

REGISTER_OP("EdmSoftmaxPool2")
    .Input("x: float")
    .Input("logits: float")
    .Output("y: float")
    .Attr("arith_variant: int = 0")
    .SetShapeFn(SoftmaxPool2Shape)
    .Doc(R"doc(
Enformer's SoftmaxPooling1D with pool size 2 given its logits: y[b, j, c] = sum over the pair i in {2j, 2j+1} of
x[b, i, c] * softmax over that pair of logits[b, i, c], computed with TensorFlow 2.17.1's own GPU float32 arithmetic
(Softmax kernel on rows of 2, Mul, Sum). x, logits: (B, L, C) with L even, or (B, P, 2, C). GPU only.
)doc");

REGISTER_OP("EdmPoolLogits")
    .Input("x: float")      // [M, K]
    .Input("w: float")      // [K, N]
    .Input("tail: float")   // [T, N]: the last T rows of the result, computed by the caller and copied into place; T may be 0
    .Output("y: float")     // [M, N]: rows [0, M-T) = X[0:M-T].W in the stock kernel's accumulation order, rows [M-T, M) = tail
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle x, w, t;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x)); TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &w)); TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &t));
        c->set_output(0, c->Matrix(c->Dim(x, 0), c->Dim(w, 1))); return ::tensorflow::OkStatus(); })
    .Doc("Y = X.W in float32 with the accumulation order of the stock pooling-logits kernel; the last rows are taken from tail (ops/README.md).");
REGISTER_OP("EdmBiasResidual")
    .Input("y: float")         // [..., C]
    .Input("bias: float")      // [C]
    .Input("res: float")       // [..., C]
    .Output("out: float")      // assoc 0: res + (y + bias); assoc 1: (res + y) + bias
    .Attr("assoc: int = 0")
    .SetShapeFn([](InferenceContext* c) { c->set_output(0, c->input(0)); return ::tensorflow::OkStatus(); })
    .Doc("A bias add and the residual add in one pass, TensorFlow's float32 adds in the stock graph's executed association (ops/README.md).");

REGISTER_OP("EdmLayerNorm")
    .Input("x: float")         // [..., N]
    .Input("gamma: float")     // [N]
    .Input("beta: float")      // [N]
    .Output("y: float")        // [..., N]: (x - mean) normalized by rsqrt(var + epsilon), scaled by gamma, shifted by beta, in TensorFlow's LayerNorm arithmetic
    .Attr("epsilon: float")
    .SetShapeFn([](InferenceContext* c) { c->set_output(0, c->input(0)); return ::tensorflow::OkStatus(); })
    .Doc("Sonnet LayerNorm over the last axis (tf.nn.moments + tf.nn.batch_normalization) with TensorFlow 2.17.1's GPU float32 arithmetic, its row reductions in their kernel's summation order (ops/README.md).");

REGISTER_KERNEL_BUILDER(Name("EdmScaleShiftGelu").Device(DEVICE_GPU), EdmScaleShiftGeluOp);
REGISTER_KERNEL_BUILDER(Name("EdmSoftmaxPool2").Device(DEVICE_GPU), EdmSoftmaxPool2Op);

REGISTER_KERNEL_BUILDER(Name("EdmPoolLogits").Device(DEVICE_GPU), EdmPoolLogitsOp);
REGISTER_KERNEL_BUILDER(Name("EdmBiasResidual").Device(DEVICE_GPU), EdmBiasResidualOp);
REGISTER_KERNEL_BUILDER(Name("EdmLayerNorm").Device(DEVICE_GPU), EdmLayerNormOp);

REGISTER_OP("EdmBiasScaleShiftGelu")
    .Input("x: float")         // (..., C): a convolution's output before its bias
    .Input("bias: float")      // (C)
    .Input("scale: float")     // (C)
    .Input("shift: float")     // (C)
    .Output("y: float")        // gelu((x + bias) * scale + shift)
    .SetShapeFn([](InferenceContext* c) { return ChannelwiseShape(c, 1, 3); })
    .Doc("BiasAdd then EdmScaleShiftGelu in one pass: TensorFlow's float32 add, then the BatchNorm -> GELU chain's operations (ops/README.md). GPU only.");

REGISTER_OP("EdmBiasGelu")
    .Input("x: float")         // (..., C)
    .Input("bias: float")      // (C)
    .Output("y: float")        // gelu(x + bias)
    .SetShapeFn([](InferenceContext* c) { return ChannelwiseShape(c, 1, 1); })
    .Doc("BiasAdd then Enformer's gelu (sigmoid(1.702 h) h) in one pass, TensorFlow's float32 operations in order (ops/README.md). GPU only.");

REGISTER_OP("EdmSoftmaxPool2Gelu")
    .Input("x: float")         // (B, L, C) with L even, or (B, P, 2, C)
    .Input("logits: float")    // x's shape
    .Input("scale: float")     // (C)
    .Input("shift: float")     // (C)
    .Output("y: float")        // (B, L/2, C) / (B, P, C): gelu(EdmSoftmaxPool2(x, logits) * scale + shift)
    .SetShapeFn([](InferenceContext* c) {
        TF_RETURN_IF_ERROR(SoftmaxPool2Shape(c));
        ShapeHandle y = c->output(0), v; DimensionHandle ch = c->RankKnown(y) ? c->Dim(y, -1) : c->UnknownDim();
        for (int i = 2; i < 4; ++i) { TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &v)); TF_RETURN_IF_ERROR(c->Merge(c->Dim(v, 0), ch, &ch)); }
        if (c->RankKnown(y)) { TF_RETURN_IF_ERROR(c->ReplaceDim(y, c->Rank(y) - 1, ch, &y)); c->set_output(0, y); }
        return absl::OkStatus(); })
    .Doc("EdmSoftmaxPool2 followed by EdmScaleShiftGelu in one pass; the pooled tensor is not materialised (ops/README.md). GPU only.");

REGISTER_OP("EdmBias2Residual")
    .Input("y: float")         // [..., C]
    .Input("bias: float")      // [C]
    .Input("res: float")       // [..., C]
    .Input("res_bias: float")  // [C]
    .Output("out: float")      // (res + res_bias) + (y + bias)
    .SetShapeFn([](InferenceContext* c) { c->set_output(0, c->input(0)); return ::tensorflow::OkStatus(); })
    .Doc("Two bias adds and the residual add in one pass: (res + res_bias) + (y + bias), each TensorFlow's float32 add (ops/README.md). GPU only.");

#undef EDM_REQUIRE_VECTOR

REGISTER_KERNEL_BUILDER(Name("EdmBiasScaleShiftGelu").Device(DEVICE_GPU), EdmBiasScaleShiftGeluOp);
REGISTER_KERNEL_BUILDER(Name("EdmBiasGelu").Device(DEVICE_GPU), EdmBiasGeluOp);
REGISTER_KERNEL_BUILDER(Name("EdmSoftmaxPool2Gelu").Device(DEVICE_GPU), EdmSoftmaxPool2GeluOp);
REGISTER_KERNEL_BUILDER(Name("EdmBias2Residual").Device(DEVICE_GPU), EdmBias2ResidualOp);
}  // namespace edm
