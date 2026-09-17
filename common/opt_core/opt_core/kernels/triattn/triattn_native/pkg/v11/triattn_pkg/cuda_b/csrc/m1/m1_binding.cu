// PyTorch binding for the M1 kernel: bias staging (fragment order), flags dispatch. The per-flags kernels are explicit
// instantiations in generated translation units (see triattn_m1.py, which writes inst_m1/m1_*.cu and inst_m1/table_m1.inc).
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <map>
#include <vector>
#include <climits>
#include <cstdlib>

#include "launch_m1.cuh"

namespace triattn_m1 {

// ---- bias staging: [B,1,H,S,S] (fp32 | bf16 | fp16, any strides) -> [B*H, nq, 4*nk, 4096] fp32 = bias / scale in the M1 fragment
//      order: block (q-tile qt, k-tile kt, chunk column c) = [h(2)][u(4)][thread t(128)][e(4)] with
//      q = 128 qt + 64 h + 16 (t/32) + (t%32)/4 + 8 (e/2), key = 128 kt + 32 c + 8 u + 2 (t%4) + (e%2); out-of-range entries 0.
__device__ __forceinline__ float to_f32(float x) { return x; }
__device__ __forceinline__ float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f32(__half x) { return __half2float(x); }

// keyany: [B, W4] words of the batch-OR key mask (bit = some row attends the key) or null: folded keys become -inf columns
template <typename SrcT>
__global__ void stage_bias_m1_kernel(SrcT const* __restrict__ src, int64_t sb, int64_t sh, int64_t sq, int64_t sk, int H, int S,
                                     float inv_scale, uint32_t const* __restrict__ keyany, int W4, float* __restrict__ dst, int* __restrict__ fix) {
    int const kt = blockIdx.x >> 2, c = blockIdx.x & 3, qt = blockIdx.y, bh = blockIdx.z;   // one CTA per (k-tile, chunk column c)
    if (fix != nullptr && blockIdx.x == 0 && qt == 0 && bh == 0 && threadIdx.x == 0) { fix[0] = 0; }   // the attention kernel's fix-list counter starts at 0
    int const b = bh / H, h = bh % H;
    int const nkc = gridDim.x, nq = gridDim.y;                   // nkc = 4 * (nk + 1) key columns of 32 (the last 4 are all -inf: padding for column-shifted tile streams)
    float* out = dst + ((int64_t(bh) * nq + qt) * nkc + blockIdx.x) * 4096;
    SrcT const* base = src + b * sb + h * sh;
    for (int idx = threadIdx.x; idx < 4096; idx += blockDim.x) {
        int const e = idx & 3, t = (idx >> 2) & 127, hu = idx >> 9;
        int const u = hu & 3, hh = hu >> 2;
        int const m = 64 * hh + 16 * (t >> 5) + ((t & 31) >> 2) + 8 * (e >> 1);
        int const n = 32 * c + 8 * u + 2 * (t & 3) + (e & 1);
        int const q = qt * 128 + m, key = kt * 128 + n;
        bool live = key < S;                                         // keys beyond S or attended by no row of the batch: -inf logits (K/V rows beyond S are TMA zero-filled)
        if (live && keyany != nullptr) { live = (keyany[int64_t(b) * W4 + (key >> 5)] >> (key & 31)) & 1u; }
        float val = live ? 0.f : -INFINITY;                          // (q rows beyond S: never stored)
        if (q < S && live) { val = to_f32(base[q * sq + key * sk]) * inv_scale; }
        out[idx] = val;
    }
}

torch::Tensor stage_bias_m1(torch::Tensor const& bias, double scale, c10::optional<torch::Tensor> const& keyany, torch::Tensor& fix) {
    TORCH_CHECK(bias.dim() == 5 && bias.size(1) == 1 && bias.size(3) == bias.size(4), "bias must be [B,1,H,S,S]");
    c10::cuda::CUDAGuard guard(bias.device());
    int const B = bias.size(0), H = bias.size(2), S = bias.size(3);
    int const nq = (S + 127) / 128, nk = (S + 127) / 128, W4 = 4 * (nk + 1);
    uint32_t const* ka = nullptr;
    if (keyany.has_value()) {
        TORCH_CHECK(keyany->scalar_type() == torch::kInt32 && keyany->is_contiguous() && keyany->size(0) == B && keyany->size(1) == W4, "keyany words: [B, 4*(nk+1)] int32");
        ka = reinterpret_cast<uint32_t const*>(keyany->data_ptr<int>());
    }
    TORCH_CHECK(fix.scalar_type() == torch::kInt32 && fix.is_contiguous(), "fix list: [1 + 3*n_ctas] int32");
    auto out = torch::empty({int64_t(B) * H, nq, W4, 4096}, bias.options().dtype(torch::kFloat32));
    dim3 grid(W4, nq, B * H), block(256);
    float const inv_scale = float(1.0 / scale);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (bias.scalar_type()) {
        case torch::kFloat32: stage_bias_m1_kernel<float><<<grid, block, 0, stream>>>(bias.data_ptr<float>(), bias.stride(0), bias.stride(2), bias.stride(3), bias.stride(4), H, S, inv_scale, ka, W4, out.data_ptr<float>(), fix.data_ptr<int>()); break;
        case torch::kBFloat16: stage_bias_m1_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(bias.data_ptr()), bias.stride(0), bias.stride(2), bias.stride(3), bias.stride(4), H, S, inv_scale, ka, W4, out.data_ptr<float>(), fix.data_ptr<int>()); break;
        case torch::kFloat16: stage_bias_m1_kernel<__half><<<grid, block, 0, stream>>>(reinterpret_cast<__half const*>(bias.data_ptr()), bias.stride(0), bias.stride(2), bias.stride(3), bias.stride(4), H, S, inv_scale, ka, W4, out.data_ptr<float>(), fix.data_ptr<int>()); break;
        default: TORCH_CHECK(false, "bias dtype must be fp32, bf16 or fp16");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---- key-mask staging: mask [B, N, 1, 1, S] bool (any strides) ->
//      words  [B, N, W4] u32: bit key%32 of word key/32 = row attends key (W4 = 4 * nk words, zero-padded)
//      keyany [B, W4]    u32: OR over the rows (must be zeroed by the caller)
//      rowkind[B, N]     u8 : 0 = row words == keyany (served by the -inf columns folded into the staged bias), 1 = irregular INTERVAL (the row
//                             attends exactly the keys of one contiguous run: only the tiles holding its two ends need the mask words), 3 = irregular
//                             RAGGED (mask words on every tile of its range), 2 = fully masked
//      rowkc0 [B, N]     i32: the row's first 32-key column with an attended key; rowkc1 [B, N] i32: one past its last such column (0, 0 when none)
//      kcend  [B]        i32: 32-key columns up to the batch's last attended key (0 when none); kcstart [B] i32: first column with an attended key
//      counts [2]        i32: += irregular rows (both kinds), += fully-masked rows (census)
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
    int const b = blockIdx.y, i = blockIdx.x, lane = threadIdx.x;   // one warp per row
    uint32_t const* w = words + (int64_t(b) * N + i) * W4;
    uint32_t const* ka = keyany + int64_t(b) * W4;
    bool any = false, irr = false; int last = -1, first = INT_MAX, rlast = -1, rfirst = INT_MAX, cnt = 0;   // batch-OR extent; this row's extent and attended-key count
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
        bool const interval = any && (cnt == rlast - rfirst + 1);   // one contiguous run of attended keys
        int const kind = !any ? 2 : (irr ? (interval ? 1 : 3) : 0);
        rowkind[int64_t(b) * N + i] = uint8_t(kind);
        if (kind != 0) { atomicAdd(counts + (kind == 2 ? 1 : 0), 1); }
        rowkc0[int64_t(b) * N + i] = any ? rfirst / 32 : 0;
        rowkc1[int64_t(b) * N + i] = any ? rlast / 32 + 1 : 0;
        if (i == 0) { kcend[b] = (last + 32) / 32; kcstart[b] = (first == INT_MAX) ? 0 : first / 32; }   // no attended key: 0 columns
    }
}

// fully-masked rows (rowkind == 2): out[b,i,h,q,:] = mean over the S keys of v[b,i,h,:,:] for every q (cuEquivariance's convention),
// overwriting whatever the attention kernel stored for them (a CTA tile whose rows are all fully masked computes and stores nothing).
// grid (N, B*H), 128 threads; other rows exit immediately.
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

// returns {words [B,N,W4] i32, keyany [B,W4] i32, rowkind [B,N] u8, kcend [B] i32, kcstart [B] i32, rowkc0 [B,N] i32, rowkc1 [B,N] i32}; counts [2] i32 is accumulated in place
std::vector<torch::Tensor> stage_mask_m1(torch::Tensor const& mask, torch::Tensor& counts) {
    TORCH_CHECK(mask.dim() == 5 && mask.size(2) == 1 && mask.size(3) == 1 && mask.scalar_type() == torch::kBool, "mask must be [B,N,1,1,S] bool");
    c10::cuda::CUDAGuard guard(mask.device());
    int const B = mask.size(0), N = mask.size(1), S = mask.size(4);
    int const nk = (S + 127) / 128, W4 = 4 * (nk + 1);
    auto opts = mask.options();
    auto words = torch::empty({B, N, W4}, opts.dtype(torch::kInt32));
    auto keyany = torch::zeros({B, W4}, opts.dtype(torch::kInt32));
    auto rowkind = torch::empty({B, N}, opts.dtype(torch::kUInt8));
    auto kcend = torch::empty({B}, opts.dtype(torch::kInt32));
    auto kcstart = torch::empty({B}, opts.dtype(torch::kInt32));
    auto rowkc0 = torch::empty({B, N}, opts.dtype(torch::kInt32));
    auto rowkc1 = torch::empty({B, N}, opts.dtype(torch::kInt32));
    auto stream = at::cuda::getCurrentCUDAStream();
    mask_words_kernel<<<dim3((W4 + 7) / 8, N, B), 256, 0, stream>>>(mask.data_ptr<bool>(), mask.stride(0), mask.stride(1), mask.stride(4), N, S, W4,
        reinterpret_cast<uint32_t*>(words.data_ptr<int>()), reinterpret_cast<uint32_t*>(keyany.data_ptr<int>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    mask_rows_kernel<<<dim3(N, B), 32, 0, stream>>>(reinterpret_cast<uint32_t const*>(words.data_ptr<int>()), reinterpret_cast<uint32_t const*>(keyany.data_ptr<int>()), N, W4,
        rowkind.data_ptr<uint8_t>(), kcend.data_ptr<int>(), kcstart.data_ptr<int>(), counts.data_ptr<int>(), rowkc0.data_ptr<int>(), rowkc1.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {words, keyany, rowkind, kcend, kcstart, rowkc0, rowkc1};
}

struct Entry { RunFn run; int64_t smem; };
#include "table_m1.inc"   // generated: run_m1_<flags> declarations and `static std::map<int, Entry> make_table()`
static std::map<int, Entry> const& table() { static std::map<int, Entry> t = make_table(); return t; }

int64_t smem_bytes(int64_t flags) {
    auto it = table().find(int(flags)); TORCH_CHECK(it != table().end(), "no M1 kernel instantiated for flags=", flags); return it->second.smem;
}

// hot pass (flags) then the SAFE pass (flags | 1024) over the fix list the hot pass wrote into `fix` ([1 + 3 * n_ctas] int32, fix[0] zeroed by stage_bias)
void fwd(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& bias_staged, double scale, torch::Tensor& out,
         torch::Tensor& fix, torch::Tensor& fix_total, c10::optional<torch::Tensor> const& maskw, c10::optional<torch::Tensor> const& rowkind, c10::optional<torch::Tensor> const& kcend,
         c10::optional<torch::Tensor> const& kcstart, c10::optional<torch::Tensor> const& rowkc0, c10::optional<torch::Tensor> const& rowkc1,
         int64_t flags, c10::optional<torch::Tensor> const& trace) {
    c10::cuda::CUDAGuard guard(q.device());
    auto hot = table().find(int(flags)), safe = table().find((int(flags) & ~256) | 1024);   // the SAFE pass never runs clustered
    TORCH_CHECK(hot != table().end() && safe != table().end(), "no M1 kernel pair instantiated for flags=", flags);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16, "q/k/v must be bf16");
    TORCH_CHECK(fix.scalar_type() == torch::kInt32 && fix.is_contiguous(), "fix list must be contiguous int32");
    unsigned long long* tr = trace.has_value() ? reinterpret_cast<unsigned long long*>(trace->data_ptr<int64_t>()) : nullptr;
    Args a{q, k, v, bias_staged, scale, out, fix.data_ptr<int>(), fix_total.data_ptr<int>(),
           maskw.has_value() ? reinterpret_cast<uint32_t const*>(maskw->data_ptr<int>()) : nullptr,
           rowkind.has_value() ? rowkind->data_ptr<uint8_t>() : nullptr, kcend.has_value() ? kcend->data_ptr<int>() : nullptr,
           kcstart.has_value() ? kcstart->data_ptr<int>() : nullptr,
           rowkc0.has_value() ? rowkc0->data_ptr<int>() : nullptr, rowkc1.has_value() ? rowkc1->data_ptr<int>() : nullptr, tr};
    { char const* ff = getenv("TRIATTN_M1_FORCE_SAFE"); a.force_fix = (ff != nullptr && ff[0] == '1') ? 1 : 0; }   // debug: exact pass for every tile
    hot->second.run(a);
    safe->second.run(a);
    if (rowkind.has_value()) {                                   // fully-masked rows: mean of v
        int const B = q.size(0), N = q.size(1), H = q.size(2), S = q.size(3);
        uniform_rows_kernel<<<dim3(N, B * H), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<__nv_bfloat16 const*>(v.data_ptr()), v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            rowkind->data_ptr<uint8_t>(), N, H, S);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace triattn_m1

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &triattn_m1::fwd, "triangle attention forward, M1 consumer (sm_90a)");
    m.def("smem_bytes", &triattn_m1::smem_bytes);
    m.def("stage_bias", &triattn_m1::stage_bias_m1, "pair bias -> M1 fragment-order fp32 staging (bias / scale), batch-OR-masked keys folded to -inf");
    m.def("stage_mask", &triattn_m1::stage_mask_m1, "key mask -> per-row words, batch OR words, row kinds, batch and per-row attended key-column extents");
}
