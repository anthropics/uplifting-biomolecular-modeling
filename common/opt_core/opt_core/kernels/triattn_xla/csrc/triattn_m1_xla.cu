// libtriattn_m1_xla.so: the "M1" sm_90a triangle-attention forward of kernels/triattn/triattn_native (package generation 11, cuda_b kernel directory) for the
// XLA-FFI launcher.  The kernel header (triattn_m1_sm90.cuh + fa3_utils.h) is compiled UNMODIFIED from the carried payload directory; this
// translation unit restates that kernel's torch host code (csrc/m1/launch_m1.cuh: TMA descriptors + launch; csrc/m1/m1_binding.cu: bias / key-mask
// staging kernels, the hot -> SAFE pass pair, the fully-masked-row epilogue) over raw device pointers, explicit strides and the caller's
// stream instead of torch tensors -- same kernels, same launch geometry, same staging arithmetic, so the output bytes are the torch launch's.
// Instantiations: Traits<0> (the routed hot kernel, flags 0: no cluster launch) and Traits<1024> (its SAFE fix-list partner).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <climits>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include "triattn_m1_sm90.cuh"          // carried: kernels/triattn/triattn_native/pkg/v11/triattn_pkg/cuda_b/csrc/m1/ (includes ../fa3_utils.h)
#include "triattn_m1_abi.h"

#define TXLA_EXPORT __attribute__((visibility("default")))

// The CUTLASS headers pull in <iostream>; a GCC >= 13 host compiler then imports std::ios_base_library_init (GLIBCXX_3.4.32) and, at -O3,
// std::__throw_bad_array_new_length (GLIBCXX_3.4.29) from libstdc++.  Both are defined here (internal: the version script exports only the C
// entry points) so the library asks libstdc++ for nothing newer than the sm_90a kernel library beside it does (GLIBCXX_3.4.21).
#include <ios>
#include <cstdlib>
namespace std {
void ios_base_library_init() { static std::ios_base::Init keep; (void)keep; }
void __throw_bad_array_new_length() { std::abort(); }
}  // namespace std

namespace triattn_m1 {

struct XArgs {
    void const* q; void const* k; void const* v; int64_t sq[4], sk[4], sv[4];   // [B,N,H,S,32] bf16, element strides of (B,N,H,S)
    float const* bias_staged;                                    // [B*H, nq, W4, 4096] fp32 (stage_bias)
    float scale;
    void* out; int64_t so[4];
    int B, N, H, S;
    int* fix; int* fix_total;
    uint32_t const* maskw; uint8_t const* rowkind; int const* kcend; int const* kcstart; int const* rowkc0; int const* rowkc1;
    int force_fix;
};

static char g_err[512];
#define XCHECK(cond, ...) do { if (!(cond)) { std::snprintf(g_err, sizeof(g_err), __VA_ARGS__); return false; } } while (0)
#define XCUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::snprintf(g_err, sizeof(g_err), "%s: %s", #call, cudaGetErrorString(e_)); return false; } } while (0)

// csrc/m1/launch_m1.cuh::launch_m1<T> restated over XArgs (same descriptors, Params, grid, shared memory; flags 0 / 1024 never launch clustered)
template <class T>
bool launch_m1x(XArgs const& a, cudaStream_t stream) {
    using namespace cute;
    using Element = typename T::Element;
    int const B = a.B, N = a.N, H = a.H, S = a.S, D = 32;
    static_assert(T::kHeadDim == 32, "head dim must be 32");
    static_assert(!(T::CR > 1 && !T::kList), "clustered instantiations are not built here");
    typename T::ShapeQK shape_qk = make_shape(S, D, H, N, B);
    auto stride_of = [](int64_t const* s) { return typename T::StrideQK{s[3], _1{}, s[2], s[1], s[0]}; };
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.q)), shape_qk, stride_of(a.sq));
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.k)), shape_qk, stride_of(a.sk));
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(a.v)), shape_qk, stride_of(a.sv));
    int const n_qtiles = (S + T::kBlockM - 1) / T::kBlockM;
    int const n_ktiles = (S + T::kBlockN - 1) / T::kBlockN;
    int const n_kcol = 4 * (n_ktiles + 1);
    typename T::ShapeB shape_b = make_shape(256, 8, 2 * n_kcol, n_qtiles, B * H);
    typename T::StrideB stride_b{_1{}, _256{}, int64_t(T::kHalfElems), int64_t(T::kSlotElems) * n_kcol, int64_t(T::kSlotElems) * n_kcol * n_qtiles};
    Tensor mB = make_tensor(make_gmem_ptr(const_cast<float*>(a.bias_staged)), shape_b, stride_b);

    typename T::TMA_Q tma_q = make_tma_copy(SM90_TMA_LOAD{}, mQ, take<0, 2>(typename T::SmemLayoutQ{}), make_shape(Int<T::kBlockM>{}, Int<T::kHeadDim>{}), _1{});
    typename T::TMA_K tma_k = make_tma_copy(SM90_TMA_LOAD{}, mK, take<0, 2>(typename T::SmemLayoutK{}), make_shape(Int<T::kBlockN>{}, Int<T::kHeadDim>{}), _1{});
    typename T::TMA_V tma_v = make_tma_copy(SM90_TMA_LOAD{}, mV, take<0, 2>(typename T::SmemLayoutV{}), make_shape(Int<T::kBlockN>{}, Int<T::kHeadDim>{}), _1{});
    typename T::TMA_B tma_b = make_tma_copy(typename T::GmemTiledCopyB{}, mB, typename T::SmemLayoutBiasHalf{}, make_shape(_256{}, _8{}), Int<T::CR>{});
    typename T::TMA_BT tma_bt = make_tma_copy(SM90_TMA_LOAD{}, mB, typename T::SmemLayoutBiasThin{}, make_shape(_256{}, _1{}), _1{});

    typename T::Params p{tma_q, tma_k, tma_v, tma_b, tma_bt, shape_qk, shape_b,
        reinterpret_cast<Element*>(a.out), a.so[0], a.so[1], a.so[2], a.so[3],
        S, N, H, n_qtiles, n_ktiles, a.scale, nullptr, int(int64_t(S) >> 40), a.fix, a.fix_total, a.force_fix, a.maskw, a.rowkind, a.kcend, a.kcstart, n_kcol, a.rowkc0, a.rowkc1};

    int smem = sizeof(typename T::SharedStorage);
    static bool configured[64] = {};                             // per instantiation, per device
    int dev = 0; XCUDA(cudaGetDevice(&dev));
    if (dev >= 0 && dev < 64 && !configured[dev]) {
        XCUDA(cudaFuncSetAttribute((void const*)triattn_m1_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        configured[dev] = true;
    }
    int n_rg = (N + T::R - 1) / T::R;
    n_rg = (n_rg + T::CR - 1) / T::CR * T::CR;
    dim3 grid(n_qtiles, n_rg, B * H), block(T::kNumThreads);
    int64_t const n_ctas = int64_t(n_qtiles) * n_rg * B * H;
    if constexpr (T::kSafe) { grid = dim3(unsigned(std::min<int64_t>(n_ctas, 66)), 1, 1); }
    triattn_m1_kernel<T><<<grid, block, smem, stream>>>(p);
    XCUDA(cudaGetLastError());
    return true;
}

// ---- staging kernels: csrc/m1/m1_binding.cu, device code verbatim ------------------------------------------------------------------
__device__ __forceinline__ float to_f32(float x) { return x; }
__device__ __forceinline__ float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f32(__half x) { return __half2float(x); }

template <typename SrcT>
__global__ void stage_bias_m1_kernel(SrcT const* __restrict__ src, int64_t sb, int64_t sh, int64_t sq, int64_t sk, int H, int S,
                                     float inv_scale, uint32_t const* __restrict__ keyany, int W4, float* __restrict__ dst, int* __restrict__ fix) {
    int const kt = blockIdx.x >> 2, c = blockIdx.x & 3, qt = blockIdx.y, bh = blockIdx.z;
    if (fix != nullptr && blockIdx.x == 0 && qt == 0 && bh == 0 && threadIdx.x == 0) { fix[0] = 0; }
    int const b = bh / H, h = bh % H;
    int const nkc = gridDim.x, nq = gridDim.y;
    float* out = dst + ((int64_t(bh) * nq + qt) * nkc + blockIdx.x) * 4096;
    SrcT const* base = src + b * sb + h * sh;
    for (int idx = threadIdx.x; idx < 4096; idx += blockDim.x) {
        int const e = idx & 3, t = (idx >> 2) & 127, hu = idx >> 9;
        int const u = hu & 3, hh = hu >> 2;
        int const m = 64 * hh + 16 * (t >> 5) + ((t & 31) >> 2) + 8 * (e >> 1);
        int const n = 32 * c + 8 * u + 2 * (t & 3) + (e & 1);
        int const q = qt * 128 + m, key = kt * 128 + n;
        bool live = key < S;
        if (live && keyany != nullptr) { live = (keyany[int64_t(b) * W4 + (key >> 5)] >> (key & 31)) & 1u; }
        float val = live ? 0.f : -INFINITY;
        if (q < S && live) { val = to_f32(base[q * sq + key * sk]) * inv_scale; }
        out[idx] = val;
    }
}

__global__ void mask_words_kernel(bool const* __restrict__ mask, int64_t sb, int64_t sn, int64_t sk, int N, int S, int W4,
                                  uint32_t* __restrict__ words, uint32_t* __restrict__ keyany) {
    int const b = blockIdx.z, i = blockIdx.y;
    int const wi = blockIdx.x * 8 + int(threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (wi >= W4) { return; }
    int const key = wi * 32 + lane;
    bool const bit = key < S && mask[int64_t(b) * sb + int64_t(i) * sn + int64_t(key) * sk];
    uint32_t const word = __ballot_sync(0xffffffffu, bit);
    if (lane == 0) {
        words[(int64_t(b) * N + i) * W4 + wi] = word;
        if (word != 0u) { atomicOr(reinterpret_cast<unsigned int*>(keyany) + int64_t(b) * W4 + wi, word); }
    }
}
__global__ void mask_rows_kernel(uint32_t const* __restrict__ words, uint32_t const* __restrict__ keyany, int N, int W4,
                                 uint8_t* __restrict__ rowkind, int* __restrict__ kcend, int* __restrict__ kcstart, int* __restrict__ counts,
                                 int* __restrict__ rowkc0, int* __restrict__ rowkc1) {
    int const b = blockIdx.y, i = blockIdx.x, lane = threadIdx.x;
    uint32_t const* w = words + (int64_t(b) * N + i) * W4;
    uint32_t const* ka = keyany + int64_t(b) * W4;
    bool any = false, irr = false; int last = -1, first = INT_MAX, rlast = -1, rfirst = INT_MAX, cnt = 0;
    for (int wi = lane; wi < W4; wi += 32) {
        any |= (w[wi] != 0u); irr |= (w[wi] != ka[wi]);
        if (ka[wi] != 0u) { last = wi * 32 + 31 - __clz(ka[wi]); first = min(first, wi * 32 + __ffs(ka[wi]) - 1); }
        if (w[wi] != 0u) { rlast = wi * 32 + 31 - __clz(w[wi]); rfirst = min(rfirst, wi * 32 + __ffs(w[wi]) - 1); cnt += __popc(w[wi]); }
    }
    any = __any_sync(0xffffffffu, any); irr = __any_sync(0xffffffffu, irr);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        last = max(last, __shfl_xor_sync(0xffffffffu, last, o)); first = min(first, __shfl_xor_sync(0xffffffffu, first, o));
        rlast = max(rlast, __shfl_xor_sync(0xffffffffu, rlast, o)); rfirst = min(rfirst, __shfl_xor_sync(0xffffffffu, rfirst, o)); cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
    }
    if (lane == 0) {
        bool const interval = any && (cnt == rlast - rfirst + 1);
        int const kind = !any ? 2 : (irr ? (interval ? 1 : 3) : 0);
        rowkind[int64_t(b) * N + i] = uint8_t(kind);
        if (kind != 0) { atomicAdd(counts + (kind == 2 ? 1 : 0), 1); }
        rowkc0[int64_t(b) * N + i] = any ? rfirst / 32 : 0;
        rowkc1[int64_t(b) * N + i] = any ? rlast / 32 + 1 : 0;
        if (i == 0) { kcend[b] = (last + 32) / 32; kcstart[b] = (first == INT_MAX) ? 0 : first / 32; }
    }
}

__global__ void uniform_rows_kernel(__nv_bfloat16 const* __restrict__ v, int64_t vb, int64_t vn, int64_t vh, int64_t vs,
                                    __nv_bfloat16* __restrict__ out, int64_t ob, int64_t on, int64_t oh, int64_t os_,
                                    uint8_t const* __restrict__ rowkind, int N, int H, int S) {
    int const i = blockIdx.x, bh = blockIdx.y, b = bh / H, h = bh % H;
    if (rowkind[int64_t(b) * N + i] != 2) { return; }
    __shared__ float part[4][32];
    int const d = threadIdx.x & 31, g = threadIdx.x >> 5;
    __nv_bfloat16 const* vrow = v + b * vb + int64_t(i) * vn + h * vh;
    float s = 0.f;
    for (int key = g; key < S; key += 4) { s += __bfloat162float(vrow[int64_t(key) * vs + d]); }
    part[g][d] = s;
    __syncthreads();
    float const mean = (part[0][d] + part[1][d] + part[2][d] + part[3][d]) / float(S);
    __nv_bfloat16 const mv = __float2bfloat16_rn(mean);
    __nv_bfloat16* orow = out + b * ob + int64_t(i) * on + h * oh;
    for (int q = g; q < S; q += 4) { orow[int64_t(q) * os_ + d] = mv; }
}

// ---- the call: stage_mask (mask) -> stage_bias -> hot (Traits<0>) -> SAFE (Traits<1024>) -> uniform rows (mask); m1_binding.cu::fwd + triattn_m1.py
static bool run(TriattnM1Call* c) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(c->stream);
    int const B = c->B, N = c->N, H = c->H, S = c->S;
    XCHECK(B > 0 && N > 0 && H > 0 && S > 0, "bad dims B=%d N=%d H=%d S=%d", B, N, H, S);
    int const nq = (S + 127) / 128, nk = nq, W4 = 4 * (nk + 1);
    uint32_t* keyany = nullptr;
    if (c->has_mask) {
        XCHECK(c->mask && c->words && c->keyany && c->rowkind && c->kcend && c->kcstart && c->rowkc0 && c->rowkc1 && c->counts, "mask scratch buffers missing");
        keyany = reinterpret_cast<uint32_t*>(c->keyany);
        XCUDA(cudaMemsetAsync(c->keyany, 0, sizeof(int) * size_t(B) * W4, stream));
        XCUDA(cudaMemsetAsync(c->counts, 0, sizeof(int) * 2, stream));
        mask_words_kernel<<<dim3((W4 + 7) / 8, N, B), 256, 0, stream>>>(reinterpret_cast<bool const*>(c->mask), int64_t(N) * S, int64_t(S), 1, N, S, W4,
            reinterpret_cast<uint32_t*>(c->words), keyany);
        XCUDA(cudaGetLastError());
        mask_rows_kernel<<<dim3(N, B), 32, 0, stream>>>(reinterpret_cast<uint32_t const*>(c->words), keyany, N, W4,
            reinterpret_cast<uint8_t*>(c->rowkind), reinterpret_cast<int*>(c->kcend), reinterpret_cast<int*>(c->kcstart), reinterpret_cast<int*>(c->counts),
            reinterpret_cast<int*>(c->rowkc0), reinterpret_cast<int*>(c->rowkc1));
        XCUDA(cudaGetLastError());
    }
    XCUDA(cudaMemsetAsync(c->fix_total, 0, sizeof(int) * 2, stream));
    {   // stage_bias_m1: bias [B,H,S,S] -> [B*H, nq, W4, 4096] fp32 = bias / scale in fragment order (fix[0] zeroed here)
        dim3 grid(W4, nq, B * H), block(256);
        float const inv_scale = float(1.0 / double(c->scale));
        int64_t const sb = int64_t(H) * S * S, sh = int64_t(S) * S, sq = S, sk = 1;
        if (c->bias_dtype == 0) {
            stage_bias_m1_kernel<float><<<grid, block, 0, stream>>>(reinterpret_cast<float const*>(c->bias), sb, sh, sq, sk, H, S, inv_scale, keyany, W4, reinterpret_cast<float*>(c->bias_staged), reinterpret_cast<int*>(c->fix));
        } else if (c->bias_dtype == 1) {
            stage_bias_m1_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(c->bias), sb, sh, sq, sk, H, S, inv_scale, keyany, W4, reinterpret_cast<float*>(c->bias_staged), reinterpret_cast<int*>(c->fix));
        } else { XCHECK(false, "bias dtype code %d (0 = fp32, 1 = bf16)", c->bias_dtype); }
        XCUDA(cudaGetLastError());
    }
    XArgs a{};
    a.q = c->q; a.k = c->k; a.v = c->v;
    for (int i = 0; i < 4; ++i) { a.sq[i] = c->sq[i]; a.sk[i] = c->sk[i]; a.sv[i] = c->sv[i]; a.so[i] = c->so[i]; }
    a.bias_staged = reinterpret_cast<float const*>(c->bias_staged); a.scale = c->scale; a.out = c->out;
    a.B = B; a.N = N; a.H = H; a.S = S;
    a.fix = reinterpret_cast<int*>(c->fix); a.fix_total = reinterpret_cast<int*>(c->fix_total);
    a.maskw = c->has_mask ? reinterpret_cast<uint32_t const*>(c->words) : nullptr;
    a.rowkind = c->has_mask ? reinterpret_cast<uint8_t const*>(c->rowkind) : nullptr;
    a.kcend = c->has_mask ? reinterpret_cast<int const*>(c->kcend) : nullptr; a.kcstart = c->has_mask ? reinterpret_cast<int const*>(c->kcstart) : nullptr;
    a.rowkc0 = c->has_mask ? reinterpret_cast<int const*>(c->rowkc0) : nullptr; a.rowkc1 = c->has_mask ? reinterpret_cast<int const*>(c->rowkc1) : nullptr;
    a.force_fix = 0;
    for (int i = 0; i < 4; ++i) {                                // the torch host code's _tma_ok: unit d stride (by construction), other strides multiples of 8 elements, 16-byte aligned bases
        XCHECK(a.sq[i] % 8 == 0 && a.sk[i] % 8 == 0 && a.sv[i] % 8 == 0, "q/k/v strides must be multiples of 8 elements (TMA)");
    }
    XCHECK((reinterpret_cast<uintptr_t>(a.q) | reinterpret_cast<uintptr_t>(a.k) | reinterpret_cast<uintptr_t>(a.v)) % 16 == 0, "q/k/v must be 16-byte aligned (TMA)");
    if (!launch_m1x<Traits<0>>(a, stream)) return false;         // hot pass
    if (!launch_m1x<Traits<1024>>(a, stream)) return false;      // SAFE pass over the fix list the hot pass wrote
    if (c->has_mask) {
        uniform_rows_kernel<<<dim3(N, B * H), 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 const*>(c->v), a.sv[0], a.sv[1], a.sv[2], a.sv[3],
            reinterpret_cast<__nv_bfloat16*>(c->out), a.so[0], a.so[1], a.so[2], a.so[3],
            reinterpret_cast<uint8_t const*>(c->rowkind), N, H, S);
        XCUDA(cudaGetLastError());
    }
    return true;
}

}  // namespace triattn_m1

extern "C" {
TXLA_EXPORT int triattn_m1_xla_fwd(TriattnM1Call* c) {
    if (c == nullptr) return 2;
    if (c->abi_version != TRIATTN_M1_ABI_VERSION) { std::snprintf(c->err, sizeof(c->err), "ABI version %d != %d", c->abi_version, TRIATTN_M1_ABI_VERSION); return 3; }
    triattn_m1::g_err[0] = 0;
    if (!triattn_m1::run(c)) { std::memcpy(c->err, triattn_m1::g_err, sizeof(c->err)); c->err[sizeof(c->err) - 1] = 0; return 1; }
    return 0;
}
TXLA_EXPORT const char* triattn_m1_xla_describe(void) {
    static char buf[160];
    std::snprintf(buf, sizeof(buf), "triattn_m1 sm_90a D=32 BM=%d BN=%d R=%d threads=%d smem=%lld/%lld (hot/safe) abi=%d", triattn_m1::Traits<0>::kBlockM, triattn_m1::Traits<0>::kBlockN,
                  triattn_m1::Traits<0>::R, triattn_m1::Traits<0>::kNumThreads, (long long)sizeof(triattn_m1::Traits<0>::SharedStorage), (long long)sizeof(triattn_m1::Traits<1024>::SharedStorage), TRIATTN_M1_ABI_VERSION);
    return buf;
}
TXLA_EXPORT int64_t triattn_m1_xla_smem_bytes(int flags) { return flags & 1024 ? sizeof(triattn_m1::Traits<1024>::SharedStorage) : sizeof(triattn_m1::Traits<0>::SharedStorage); }
}
