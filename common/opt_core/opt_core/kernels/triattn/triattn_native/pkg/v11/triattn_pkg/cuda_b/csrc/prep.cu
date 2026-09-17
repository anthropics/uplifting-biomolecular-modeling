// Bias staging for the cuda_b kernels: [B,1,H,S,S] pair bias (fp32 | bf16 | fp16, any strides) -> [B*H, nq, nk, 128*BN] fp32 =
// bias / scale in MMA-fragment order per (q-tile of 128, k-tile of BN): block order [column chunk c (2)][consumer warpgroup (2)]
// [warp (4)][load v (BN/16)][lane (32)][4 floats], where load v = mi*(BN/32) + u holds, for lane l, rows m = 64*wg + 16*warp + l/4 + 8*mi
// and columns n0, n0+1, n0+8, n0+9 with n0 = c*BN/2 + 16*u + 2*(l%4). Out-of-range (q >= S or k >= S) entries are 0.
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace triattn_b {

__device__ __forceinline__ float to_f32(float x) { return x; }
__device__ __forceinline__ float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float to_f32(__half x) { return __half2float(x); }

template <typename SrcT>
__global__ void stage_bias_kernel(SrcT const* __restrict__ src, int64_t sb, int64_t sh, int64_t sq, int64_t sk, int H, int S, int BN,
                                  float inv_scale, float* __restrict__ dst) {
    int const kt = blockIdx.x, qt = blockIdx.y, bh = blockIdx.z;
    int const b = bh / H, h = bh % H;
    int const CW = BN / 2, NV = CW / 8;
    int const block_elems = 128 * BN;
    float* out = dst + ((int64_t(bh) * gridDim.y + qt) * gridDim.x + kt) * block_elems;
    SrcT const* base = src + b * sb + h * sh;
    for (int pos = threadIdx.x; pos < block_elems; pos += blockDim.x) {
        int const e = pos & 3, lane = (pos >> 2) & 31;
        int rest = pos >> 7;
        int const v = rest % NV; rest /= NV;
        int const w = rest & 3; rest >>= 2;
        int const wg = rest & 1; int const c = rest >> 1;
        int const mi = v / (CW / 16), u = v % (CW / 16);
        int const m = 64 * wg + 16 * w + (lane >> 2) + 8 * mi;
        int const n = c * CW + 16 * u + 2 * (lane & 3) + (e < 2 ? e : 6 + e);
        int const q = qt * 128 + m, k = kt * BN + n;
        float val = 0.f;
        if (q < S && k < S) { val = to_f32(base[q * sq + k * sk]) * inv_scale; }
        out[pos] = val;
    }
}

torch::Tensor stage_bias(torch::Tensor const& bias, double scale, int64_t block_n) {
    TORCH_CHECK(bias.dim() == 5 && bias.size(1) == 1 && bias.size(3) == bias.size(4), "bias must be [B,1,H,S,S]");
    c10::cuda::CUDAGuard guard(bias.device());
    int const B = bias.size(0), H = bias.size(2), S = bias.size(3), BN = int(block_n);
    int const nq = (S + 127) / 128, nk = (S + BN - 1) / BN;
    auto out = torch::empty({int64_t(B) * H, nq, nk, int64_t(128) * BN}, bias.options().dtype(torch::kFloat32));
    dim3 grid(nk, nq, B * H), block(256);
    float const inv_scale = float(1.0 / scale);
    auto stream = at::cuda::getCurrentCUDAStream();
    switch (bias.scalar_type()) {
        case torch::kFloat32: stage_bias_kernel<float><<<grid, block, 0, stream>>>(bias.data_ptr<float>(), bias.stride(0), bias.stride(2), bias.stride(3), bias.stride(4), H, S, BN, inv_scale, out.data_ptr<float>()); break;
        case torch::kBFloat16: stage_bias_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(bias.data_ptr()), bias.stride(0), bias.stride(2), bias.stride(3), bias.stride(4), H, S, BN, inv_scale, out.data_ptr<float>()); break;
        case torch::kFloat16: stage_bias_kernel<__half><<<grid, block, 0, stream>>>(reinterpret_cast<__half const*>(bias.data_ptr()), bias.stride(0), bias.stride(2), bias.stride(3), bias.stride(4), H, S, BN, inv_scale, out.data_ptr<float>()); break;
        default: TORCH_CHECK(false, "bias dtype must be fp32, bf16 or fp16");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace triattn_b
