// edm_biasact.cc — TensorFlow side of EdmBiasAct and EdmQScaleBias: op registration, shape functions and the GPU OpKernels (built by build.sh
// with g++ against the TensorFlow headers of the installed wheel; the kernels and their launchers are in edm_biasact.cu.cc).
//
//   EdmBiasAct(x, bias) -> y                     act(x + bias), attr act in {relu, softplus}; x: (..., C), bias: (C); float32, GPU only
//   EdmQScaleBias(q, bias_w, bias_r) -> qw, qr   q: (B, H, T, K); q * scale + bias_w and q * scale + bias_r, biases (1, H, 1, K); attr scale (float)
#define EIGEN_USE_GPU
#include <cuda_runtime.h>

#include <string>

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
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;
namespace errors = ::tensorflow::errors;

// x: (..., C), bias: (C) -> y: x's shape.
Status BiasActShape(InferenceContext* c) {
  ShapeHandle x, bias;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &x));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &bias));
  if (!c->RankKnown(x)) {
    c->set_output(0, c->UnknownShape());
    return absl::OkStatus();
  }
  DimensionHandle ch;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(x, -1), c->Dim(bias, 0), &ch));
  ShapeHandle y;
  TF_RETURN_IF_ERROR(c->ReplaceDim(x, c->Rank(x) - 1, ch, &y));
  c->set_output(0, y);
  return absl::OkStatus();
}

class EdmBiasActOp : public OpKernel {
 public:
  explicit EdmBiasActOp(OpKernelConstruction* c) : OpKernel(c) {
    std::string act;
    OP_REQUIRES_OK(c, c->GetAttr("act", &act));
    act_ = act == "relu" ? kActRelu : act == "softplus" ? kActSoftplus : -1;
    OP_REQUIRES(c, act_ >= 0, errors::InvalidArgument("EdmBiasAct: act must be relu or softplus; got ", act));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0); const Tensor& bias = ctx->input(1);
    OP_REQUIRES(ctx, x.dims() >= 1 && bias.dims() == 1 && x.dim_size(x.dims() - 1) == bias.dim_size(0),
                errors::InvalidArgument("EdmBiasAct: x [..., C] and bias [C] expected; got ", x.shape().DebugString(), ", ", bias.shape().DebugString()));
    OP_REQUIRES(ctx, bias.dim_size(0) <= 2147483647LL, errors::InvalidArgument("EdmBiasAct: C too large: ", bias.dim_size(0)));
    Tensor* y = nullptr; OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, x.shape(), &y));   // each thread reads its element before writing it
    const int64_t C = bias.dim_size(0), rows = C ? x.NumElements() / C : 0;
    if (rows == 0) return;
    const cudaError_t e = LaunchBiasAct(ctx->eigen_device<Eigen::GpuDevice>().stream(), x.flat<float>().data(), bias.flat<float>().data(), y->flat<float>().data(),
                                        rows, (int)C, act_);
    OP_REQUIRES(ctx, e == cudaSuccess, errors::Internal("EdmBiasAct launch: ", cudaGetErrorString(e)));
  }
 private:
  int act_ = 0;
};

class EdmQScaleBiasOp : public OpKernel {
 public:
  explicit EdmQScaleBiasOp(OpKernelConstruction* c) : OpKernel(c) { OP_REQUIRES_OK(c, c->GetAttr("scale", &scale_)); }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& q = ctx->input(0); const Tensor& bw = ctx->input(1); const Tensor& br = ctx->input(2);
    OP_REQUIRES(ctx, q.dims() == 4, errors::InvalidArgument("EdmQScaleBias: q must be (B, H, T, K); got ", q.shape().DebugString()));
    const int64_t B = q.dim_size(0), H = q.dim_size(1), T = q.dim_size(2), K = q.dim_size(3);
    for (const Tensor* b : {&bw, &br})
      OP_REQUIRES(ctx, b->dims() == 4 && b->dim_size(0) == 1 && b->dim_size(1) == H && b->dim_size(2) == 1 && b->dim_size(3) == K,
                  errors::InvalidArgument("EdmQScaleBias: the biases must be (1, H, 1, K) for q ", q.shape().DebugString(), "; got ", b->shape().DebugString()));
    OP_REQUIRES(ctx, T * K <= 2147483647LL && B * H <= 65535,
                errors::InvalidArgument("EdmQScaleBias: T*K <= 2^31-1 and B*H <= 65535 required; got q ", q.shape().DebugString()));
    Tensor* qw = nullptr; Tensor* qr = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, q.shape(), &qw));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, q.shape(), &qr));
    if (q.NumElements() == 0) return;
    const cudaError_t e = LaunchQScaleBias(ctx->eigen_device<Eigen::GpuDevice>().stream(), q.flat<float>().data(), scale_, bw.flat<float>().data(), br.flat<float>().data(),
                                           qw->flat<float>().data(), qr->flat<float>().data(), B, (int)H, T, (int)K);
    OP_REQUIRES(ctx, e == cudaSuccess, errors::Internal("EdmQScaleBias launch: ", cudaGetErrorString(e)));
  }
 private:
  float scale_ = 1.f;
};

}  // namespace

REGISTER_OP("EdmBiasAct")
    .Input("x: float")         // [..., C]
    .Input("bias: float")      // [C]
    .Output("y: float")        // act(x + bias)
    .Attr("act: {'relu', 'softplus'}")
    .SetShapeFn(BiasActShape)
    .Doc("act(x + bias) in one pass with TensorFlow 2.17.1's GPU float32 arithmetic: the bias add, then Relu or Softplus (ops/README.md).");

REGISTER_OP("EdmQScaleBias")
    .Input("q: float")         // [B, H, T, K]
    .Input("bias_w: float")    // [1, H, 1, K]
    .Input("bias_r: float")    // [1, H, 1, K]
    .Output("qw: float")       // q * scale + bias_w
    .Output("qr: float")       // q * scale + bias_r
    .Attr("scale: float")
    .SetShapeFn([](InferenceContext* c) { ShapeHandle s; TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &s)); c->set_output(0, s); c->set_output(1, s); return ::tensorflow::OkStatus(); })
    .Doc("The attention query's scaling and its two bias adds (r_w_bias, r_r_bias) in one pass, TensorFlow's float32 Mul and AddV2 arithmetic (ops/README.md).");

REGISTER_KERNEL_BUILDER(Name("EdmBiasAct").Device(DEVICE_GPU), EdmBiasActOp);
REGISTER_KERNEL_BUILDER(Name("EdmQScaleBias").Device(DEVICE_GPU), EdmQScaleBiasOp);
}  // namespace edm
