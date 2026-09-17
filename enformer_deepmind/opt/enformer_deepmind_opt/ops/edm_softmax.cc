// edm_softmax.cc — TensorFlow side of EdmRelShiftSoftmax: op registration, shape function and the GPU OpKernel (built by build.sh with g++
// against the TensorFlow headers of the installed wheel; the kernel and its launcher are in edm_softmax.cu.cc).
//
//   EdmRelShiftSoftmax(content, rel) -> probs     content: (B, H, L, L), rel: (B, H, L, 2L-1), L = 1536; float32, contiguous, GPU only
//   probs[b,h,i,:] = softmax(content[b,h,i,:] + relative_shift(rel)[b,h,i,:]) with TensorFlow 2.17.1's own float32 arithmetic (README.md)
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
using ::tensorflow::Tensor;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;
namespace errors = ::tensorflow::errors;

class EdmRelShiftSoftmaxOp : public OpKernel {
 public:
  explicit EdmRelShiftSoftmaxOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& content = ctx->input(0); const Tensor& rel = ctx->input(1);
    OP_REQUIRES(ctx, content.dims() == 4 && rel.dims() == 4 && content.dim_size(0) == rel.dim_size(0) && content.dim_size(1) == rel.dim_size(1) &&
                     content.dim_size(2) == rel.dim_size(2) && content.dim_size(3) == content.dim_size(2) && rel.dim_size(3) == 2 * content.dim_size(2) - 1,
                errors::InvalidArgument("EdmRelShiftSoftmax: content [B,H,L,L] and rel [B,H,L,2L-1] expected; got ", content.shape().DebugString(), " and ",
                                        rel.shape().DebugString()));
    const int64_t L = content.dim_size(2), rows = content.dim_size(0) * content.dim_size(1) * L;
    OP_REQUIRES(ctx, L == kRelShiftSoftmaxCols, errors::InvalidArgument("EdmRelShiftSoftmax: L = ", kRelShiftSoftmaxCols, " columns expected; got ", L));
    OP_REQUIRES(ctx, rows < (1LL << 31), errors::InvalidArgument("EdmRelShiftSoftmax: too many rows: ", rows));
    Tensor* out = nullptr; OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, content.shape(), &out));   // a block reads its row before writing it
    if (rows == 0) return;
    const cudaError_t e = LaunchRelShiftSoftmax(ctx->eigen_device<Eigen::GpuDevice>().stream(), content.flat<float>().data(), rel.flat<float>().data(),
                                                out->flat<float>().data(), rows, (int)L);
    OP_REQUIRES(ctx, e == cudaSuccess, errors::Internal("EdmRelShiftSoftmax launch: ", cudaGetErrorString(e)));
  }
};

REGISTER_OP("EdmRelShiftSoftmax")
    .Input("content: float")   // [B, H, L, L]
    .Input("rel: float")       // [B, H, L, 2L-1]
    .Output("probs: float")    // [B, H, L, L] = softmax(content + relative_shift(rel), axis=-1)
    .SetShapeFn([](InferenceContext* c) { ShapeHandle s; TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &s)); c->set_output(0, s); return ::tensorflow::OkStatus(); })
    .Doc("softmax over the last axis of content + attention_module.relative_shift(rel), rows of 1536 columns, with TensorFlow 2.17.1's own GPU float32 "
         "arithmetic: AddV2, then the Softmax kernel's row max, sum of exp and normalization in their order (ops/README.md). GPU only.");

REGISTER_KERNEL_BUILDER(Name("EdmRelShiftSoftmax").Device(DEVICE_GPU), EdmRelShiftSoftmaxOp);

}  // namespace
}  // namespace edm
