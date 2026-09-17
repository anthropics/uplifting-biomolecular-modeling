// PyTorch binding + host-side TMA descriptor construction and cluster launch for triattn_sm90.cuh.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/cluster_launch.hpp>

#include "triattn_sm90.cuh"

namespace {

using namespace cute;

template <class T>
void launch(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& bias,
            c10::optional<torch::Tensor> const& mask, double scale, torch::Tensor& out, c10::optional<torch::Tensor> const& trace,
            c10::optional<torch::Tensor> const& counters, c10::optional<torch::Tensor> const& bias_frag) {
    using Element = typename T::Element;
    int const B = q.size(0), N = q.size(1), H = q.size(2), S = q.size(3), D = q.size(4);
    TORCH_CHECK(D == T::kHeadDim, "head dim mismatch");
    TORCH_CHECK(S <= T::kMaxS, "S exceeds kMaxS");
    typename T::ShapeQK shape_qk = make_shape(S, D, H, N, B);
    auto stride_of = [](torch::Tensor const& t) {
        return typename T::StrideQK{t.stride(3), _1{}, t.stride(2), t.stride(1), t.stride(0)};
    };
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(q.data_ptr())), shape_qk, stride_of(q));
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(k.data_ptr())), shape_qk, stride_of(k));
    auto shape_v = make_shape(D, S, H, N, B);
    typename T::StrideV stride_v{_1{}, v.stride(3), v.stride(2), v.stride(1), v.stride(0)};
    Tensor mVt = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(v.data_ptr())), shape_v, stride_v);
    // bias: [B, H, S, S_pad] bf16 (any q-row stride multiple of 8, key stride 1)
    typename T::ShapeB shape_b = make_shape(S, S, H, B);
    typename T::StrideB stride_b{bias.stride(2), _1{}, bias.stride(1), bias.stride(0)};
    Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(bias.data_ptr())), shape_b, stride_b);

    typename T::TMA_Q tma_q = make_tma_copy(SM90_TMA_LOAD{}, mQ, take<0, 2>(typename T::SmemLayoutQ{}), make_shape(Int<T::kBlockM>{}, Int<T::kHeadDim>{}), _1{});
    typename T::TMA_K tma_k = make_tma_copy(typename T::GmemTiledCopyKV{}, mK, take<0, 2>(typename T::SmemLayoutK{}), make_shape(Int<T::kBlockN>{}, Int<T::kHeadDim>{}), Int<T::ClusterQ>{});
    typename T::TMA_V tma_v = make_tma_copy(typename T::GmemTiledCopyKV{}, mVt, typename T::SmemLayoutVload1{}, make_shape(Int<T::kHeadDim>{}, Int<T::kBlockN>{}), Int<T::ClusterQ>{});
    typename T::TMA_B tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, take<0, 2>(typename T::SmemLayoutBias{}), make_shape(Int<T::kBlockM>{}, Int<T::kBlockN>{}), _1{});

    int const n_qtiles = (S + T::kBlockM - 1) / T::kBlockM;
    int const n_ktiles = (S + T::kBlockN - 1) / T::kBlockN;
    int const n_qtiles_pad = (n_qtiles + T::ClusterQ - 1) / T::ClusterQ * T::ClusterQ;
    typename T::Params p{tma_q, tma_k, tma_v, tma_b, shape_qk, shape_b,
        reinterpret_cast<Element*>(out.data_ptr()), out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        mask.has_value() ? mask->data_ptr<uint8_t>() : nullptr,
        mask.has_value() ? mask->stride(0) : 0, mask.has_value() ? mask->stride(1) : 0, mask.has_value() ? mask->stride(2) : 0,
        S, N, H, n_qtiles, n_ktiles, float(scale * 1.4426950408889634),
        trace.has_value() ? reinterpret_cast<unsigned long long*>(trace->data_ptr<int64_t>()) : nullptr,
        counters.has_value() ? reinterpret_cast<unsigned long long*>(counters->data_ptr<int64_t>()) : nullptr,
        bias_frag.has_value() ? reinterpret_cast<char const*>(bias_frag->data_ptr()) : nullptr, 0};

    int smem = sizeof(typename T::SharedStorage);
    static bool configured = false;   // per instantiation
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute((void const*)triattn::triattn_fwd_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        configured = true;
    }
    dim3 grid(n_qtiles_pad, (N + T::R - 1) / T::R, B * H), block(T::kNumThreads), cluster(T::ClusterQ, 1, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if constexpr (T::ClusterQ > 1) {
        cutlass::ClusterLaunchParams lp{grid, block, cluster, smem, stream};
        cutlass::Status st = cutlass::launch_kernel_on_cluster(lp, (void const*)triattn::triattn_fwd_kernel<T>, p);
        TORCH_CHECK(st == cutlass::Status::kSuccess, "cluster launch failed");
    } else {
        triattn::triattn_fwd_kernel<T><<<grid, block, smem, stream>>>(p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// The instantiated kernel variants: X(block_n, rows, cluster_q, stages, variant, ping_pong, poly_every, debug_bits, consumer_warpgroups, rows_per_step)
#define TRIATTN_VARIANTS(X) \
    X(64, 4, 1, 12, 1, true, 0, 0, 2, 2)

int64_t smem_bytes(int64_t block_n, int64_t rows, int64_t cluster_q, int64_t stages, int64_t variant) {
    using namespace triattn;
    using E = cutlass::bfloat16_t;
#define SMEM_CASE(BN, RR, CQ, ST, VAR, PP, POLY, DBG, WG, RPS) \
    if (block_n == BN && rows == RR && cluster_q == CQ && stages == ST && variant == VAR) return sizeof(Traits<32, BN, RR, CQ, ST, E, PP, POLY, DBG, WG, RPS>::SharedStorage);
    TRIATTN_VARIANTS(SMEM_CASE)
#undef SMEM_CASE
    return -1;
}

int64_t bias_f32(int64_t block_n, int64_t rows, int64_t cluster_q, int64_t stages, int64_t variant) {
    using namespace triattn;
    using E = cutlass::bfloat16_t;
#define F32_CASE(BN, RR, CQ, ST, VAR, PP, POLY, DBG, WG, RPS) \
    if (block_n == BN && rows == RR && cluster_q == CQ && stages == ST && variant == VAR) return Traits<32, BN, RR, CQ, ST, E, PP, POLY, DBG, WG, RPS>::kBiasF32 ? 1 : 0;
    TRIATTN_VARIANTS(F32_CASE)
#undef F32_CASE
    return -1;
}

int64_t block_m(int64_t block_n, int64_t rows, int64_t cluster_q, int64_t stages, int64_t variant) {
    using namespace triattn;
    using E = cutlass::bfloat16_t;
#define BM_CASE(BN, RR, CQ, ST, VAR, PP, POLY, DBG, WG, RPS) \
    if (block_n == BN && rows == RR && cluster_q == CQ && stages == ST && variant == VAR) return Traits<32, BN, RR, CQ, ST, E, PP, POLY, DBG, WG, RPS>::kBlockM;
    TRIATTN_VARIANTS(BM_CASE)
#undef BM_CASE
    return -1;
}

void fwd(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& bias,
         c10::optional<torch::Tensor> const& mask, double scale, torch::Tensor& out,
         int64_t block_n, int64_t rows, int64_t cluster_q, int64_t stages, int64_t variant, c10::optional<torch::Tensor> const& trace,
         c10::optional<torch::Tensor> const& counters, c10::optional<torch::Tensor> const& bias_frag) {
    using namespace triattn;
    c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && bias.scalar_type() == torch::kBFloat16, "bf16 only");
    using E = cutlass::bfloat16_t;
#define DISPATCH(BN, RR, CQ, ST, VAR, PP, POLY, DBG, WG, RPS) \
    if (block_n == BN && rows == RR && cluster_q == CQ && stages == ST && variant == VAR) { launch<Traits<32, BN, RR, CQ, ST, E, PP, POLY, DBG, WG, RPS>>(q, k, v, bias, mask, scale, out, trace, counters, bias_frag); return; }
    TRIATTN_VARIANTS(DISPATCH)
#undef DISPATCH
    TORCH_CHECK(false, "no kernel instantiated for block_n=", block_n, " rows=", rows, " cluster_q=", cluster_q, " stages=", stages, " variant=", variant);
}


#ifdef TRIATTN_MICROBENCH
std::vector<double> wgmma_bench(int64_t kind, int64_t iters, int64_t blocks) {
    using namespace triattn;
    using T = Traits<32, 64, 2, 1, 8, cutlass::bfloat16_t, false, 0, 0>;
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor out = torch::zeros({blocks * 2}, opts);
    int smem = sizeof(typename T::SharedStorage);
    auto launch_k = [&](auto* kern) {
        C10_CUDA_CHECK(cudaFuncSetAttribute((void const*)kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        kern<<<(unsigned)blocks, 256, smem, at::cuda::getCurrentCUDAStream()>>>((int)iters, reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()));
    };
    switch (kind) {
        case 0: launch_k(&triattn::wgmma_bench_kernel<T, 0, 1>); break;
        case 1: launch_k(&triattn::wgmma_bench_kernel<T, 1, 1>); break;
        case 2: launch_k(&triattn::wgmma_bench_kernel<T, 2, 1>); break;
        case 3: launch_k(&triattn::wgmma_bench_kernel<T, 3, 1>); break;
        case 10: launch_k(&triattn::wgmma_bench_kernel<T, 0, 2>); break;
        case 11: launch_k(&triattn::wgmma_bench_kernel<T, 1, 2>); break;
        case 12: launch_k(&triattn::wgmma_bench_kernel<T, 2, 2>); break;
        case 13: launch_k(&triattn::wgmma_bench_kernel<T, 3, 2>); break;
        default: TORCH_CHECK(false, "kind");
    }
    C10_CUDA_CHECK(cudaGetLastError());
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    auto h = out.cpu();
    std::vector<double> r(blocks * 2);
    for (int i = 0; i < blocks * 2; ++i) r[i] = (double)h.data_ptr<int64_t>()[i];
    return r;
}

#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef TRIATTN_MICROBENCH
    m.def("wgmma_bench", &wgmma_bench, "F4 microkernel: wgmma throughput of the kernel shapes");
#endif
    m.def("fwd", &fwd, "triangle attention forward (sm_90a)");
    m.def("bias_f32", &bias_f32);
    m.def("block_m", &block_m, "query rows per CTA of a variant");
    m.def("smem_bytes", &smem_bytes);
}
