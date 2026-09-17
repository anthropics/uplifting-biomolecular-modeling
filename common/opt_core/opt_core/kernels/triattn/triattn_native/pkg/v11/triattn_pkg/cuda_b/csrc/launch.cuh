// Host side for triattn_b_sm90.cuh: TMA descriptor construction and the (cluster) launch of one Traits instantiation over a
// range of q-tiles. Included by the per-configuration translation units (one explicit instantiation each, compiled in parallel).
#pragma once
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/cluster_launch.hpp>

#include "triattn_b_sm90.cuh"

namespace triattn_b {

struct FwdArgs {
    torch::Tensor const& q; torch::Tensor const& k; torch::Tensor const& v; torch::Tensor const& bias;   // bias: staged [B*H, nq, nk, 128*BN] fp32 (stage_bias)
    c10::optional<torch::Tensor> const& mask;                                                            // [B,N,S] uint8 or none
    double scale;
    torch::Tensor& out;
    int qtile_base; int n_qtiles_launch;
    unsigned long long* trace = nullptr;                                                                 // q-tiles [base, base+n)
};

using RunFn = void (*)(FwdArgs const&);

template <class T>
void launch_cfg(FwdArgs const& a) {
    using namespace cute;
    using Element = typename T::Element;
    int const B = a.q.size(0), N = a.q.size(1), H = a.q.size(2), S = a.q.size(3), D = a.q.size(4);
    TORCH_CHECK(D == T::kHeadDim, "head dim mismatch");
    TORCH_CHECK(S <= T::kMaxS, "S exceeds kMaxS");
    typename T::ShapeQK shape_qk = make_shape(S, D, H, N, B);
    auto stride_of = [](torch::Tensor const& t) {
        return typename T::StrideQK{t.stride(3), _1{}, t.stride(2), t.stride(1), t.stride(0)};
    };
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.q.data_ptr())), shape_qk, stride_of(a.q));
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.k.data_ptr())), shape_qk, stride_of(a.k));
    auto shape_v = make_shape(D, S, H, N, B);
    typename T::StrideV stride_v{_1{}, a.v.stride(3), a.v.stride(2), a.v.stride(1), a.v.stride(0)};
    Tensor mVt = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.v.data_ptr())), shape_v, stride_v);
    int const n_qtiles = (S + T::kBlockM - 1) / T::kBlockM;
    int const n_ktiles = (S + T::kBlockN - 1) / T::kBlockN;
    auto mB = [&]() {
        if constexpr (T::kBiasMMA) {   // the caller's bf16 bias [B,1,H,S,>=S] (key stride 1, other strides 16-byte multiples)
            TORCH_CHECK(a.bias.dim() == 5 && a.bias.scalar_type() == torch::kBFloat16 && a.bias.stride(4) == 1 && a.bias.size(3) == S, "bias-MMA mode takes the bf16 [B,1,H,S,S] bias");
            typename T::ShapeB shape_b = make_shape(S, S, H, B);
            typename T::StrideB stride_b{a.bias.stride(3), _1{}, a.bias.stride(2), a.bias.stride(0)};
            return make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.bias.data_ptr())), shape_b, stride_b);
        } else {                       // staged fp32 [B*H, nq, nk, 128*BN] (stage_bias)
            TORCH_CHECK(a.bias.dim() == 4 && a.bias.size(0) == B * H && a.bias.size(1) == n_qtiles && a.bias.size(2) == n_ktiles && a.bias.size(3) == T::kBiasTileElems
                        && a.bias.is_contiguous() && a.bias.scalar_type() == torch::kFloat32, "staged bias has the wrong shape for this tile configuration");
            typename T::ShapeB shape_b = make_shape(256, T::kBlockN / 2, n_ktiles, n_qtiles, B * H);
            typename T::StrideB stride_b{_1{}, _256{}, int64_t(T::kBiasTileElems), int64_t(T::kBiasTileElems) * n_ktiles, int64_t(T::kBiasTileElems) * n_ktiles * n_qtiles};
            return make_tensor(make_gmem_ptr(a.bias.data_ptr<float>()), shape_b, stride_b);
        }
    }();
    auto shape_b = mB.shape();
    // bias-MMA: beta = the bf16-representable value nearest 1/scale multiplies the identity operand (bias term scaled by beta*scale ~ 1 - 1e-4 at D=32)
    float const beta = float(cutlass::bfloat16_t(float(1.0 / a.scale)));

    typename T::TMA_Q tma_q = make_tma_copy(SM90_TMA_LOAD{}, mQ, take<0, 2>(typename T::SmemLayoutQ{}), make_shape(Int<T::kBlockM>{}, Int<T::kHeadDim>{}), _1{});
    typename T::TMA_K tma_k = make_tma_copy(typename T::GmemTiledCopyKV{}, mK, take<0, 2>(typename T::SmemLayoutK{}), make_shape(Int<T::kBlockN>{}, Int<T::kHeadDim>{}), Int<T::CQ>{});
    typename T::TMA_V tma_v = make_tma_copy(typename T::GmemTiledCopyKV{}, mVt, typename T::SmemLayoutVload1{}, make_shape(Int<T::kHeadDim>{}, Int<T::kBlockN>{}), Int<T::CQ>{});
    auto tma_b = [&]() {
        if constexpr (T::kBiasMMA) { return make_tma_copy(typename T::GmemTiledCopyB{}, mB, typename T::SmemLayoutBiasH{}, make_shape(Int<T::kBlockM>{}, Int<T::kBlockN>{}), Int<T::CR>{}); }
        else                       { return make_tma_copy(typename T::GmemTiledCopyB{}, mB, typename T::SmemLayoutBias1{}, make_shape(_256{}, Int<T::kBlockN / 2>{}), Int<T::CR>{}); }
    }();

    TORCH_CHECK(a.n_qtiles_launch % T::CQ == 0, "q-tile count of a launch must be a multiple of the cluster x extent");
    typename T::Params p{tma_q, tma_k, tma_v, tma_b, shape_qk, shape_b,
        reinterpret_cast<Element*>(a.out.data_ptr()), a.out.stride(0), a.out.stride(1), a.out.stride(2), a.out.stride(3),
        a.mask.has_value() ? a.mask->data_ptr<uint8_t>() : nullptr,
        a.mask.has_value() ? a.mask->stride(0) : 0, a.mask.has_value() ? a.mask->stride(1) : 0, a.mask.has_value() ? a.mask->stride(2) : 0,
        S, N, H, n_qtiles, n_ktiles, a.qtile_base, float(a.scale), beta, a.trace, int(a.q.size(3) >> 40)};   // zero: 0 for any real tensor, opaque to the compiler

    int smem = sizeof(typename T::SharedStorage);
    static bool configured = false;   // per instantiation
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute((void const*)triattn_fwd_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        configured = true;
    }
    int const n_rgroups = (N + T::R - 1) / T::R;
    int const n_rgroups_pad = (n_rgroups + T::CR - 1) / T::CR * T::CR;
    dim3 grid(a.n_qtiles_launch, n_rgroups_pad, B * H), block(T::kNumThreads), cluster(T::CQ, T::CR, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if constexpr (T::CQ * T::CR > 1) {
        cutlass::ClusterLaunchParams lp{grid, block, cluster, smem, stream};
        cutlass::Status st = cutlass::launch_kernel_on_cluster(lp, (void const*)triattn_fwd_kernel<T>, p);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "cluster launch failed");
    } else {
        triattn_fwd_kernel<T><<<grid, block, smem, stream>>>(p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T>
int64_t smem_bytes_cfg() { return sizeof(typename T::SharedStorage); }

}  // namespace triattn_b
