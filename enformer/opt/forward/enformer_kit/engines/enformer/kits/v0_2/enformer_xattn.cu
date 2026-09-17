#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>
// Exact (bitwise) pieces of enformer-pytorch's Attention (the relative band GEMM and the fused add + softmax as one kernel pair):
// (1) rel_band_gemm: positional logits only on the n x n band the stock keeps after relative_shift: rel[i, j] = sum_d qp[i, d] * relk[n-1-i+j, d],
//     sequential FMA over d = 0..63 (the cuBLAS K order found on real activations; probe first);  (2) add2_softmax_exact: logits = content + rel,
//     softmax in cunn_SoftMaxForwardSmem order for dim 1536 (block 1024 threads x float4; threads >= 384 hold identities), libdevice expf, true division.
__device__ __forceinline__ float warp_sum_tree(float v) { for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off)); return v; }
__device__ float block_sum_exact(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum_tree(v); __syncthreads(); if (lane == 0) sh[wid] = v; __syncthreads();
    float r = 0.f; if (wid == 0) { r = (lane < (blockDim.x >> 5)) ? sh[lane] : 0.f; r = warp_sum_tree(r); if (lane == 0) sh[0] = r; }
    __syncthreads(); r = sh[0]; __syncthreads(); return r;
}
__device__ float block_max(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_max(v); __syncthreads(); if (lane == 0) sh[wid] = v; __syncthreads();
    float r = -INFINITY; if (wid == 0) { r = (lane < (blockDim.x >> 5)) ? sh[lane] : -INFINITY; r = warp_max(r); if (lane == 0) sh[0] = r; }
    __syncthreads(); r = sh[0]; __syncthreads(); return r;
}
template <int DIM>
__global__ void add2_softmax_exact_k(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y) {
    __shared__ float sh[32];
    const size_t row = blockIdx.x;
    const bool active = (threadIdx.x * 4) < DIM;
    float4 l = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
    if (active) {
        const float4 ca = reinterpret_cast<const float4*>(a + row * DIM)[threadIdx.x];
        const float4 cb = reinterpret_cast<const float4*>(b + row * DIM)[threadIdx.x];
        l.x = ca.x + cb.x; l.y = ca.y + cb.y; l.z = ca.z + cb.z; l.w = ca.w + cb.w;
    }
    float m = active ? fmaxf(fmaxf(l.x, l.y), fmaxf(l.z, l.w)) : -INFINITY; m = block_max(m, sh);
    float4 e = make_float4(0.f, 0.f, 0.f, 0.f); float s = 0.f;
    if (active) { e.x = expf(l.x - m); e.y = expf(l.y - m); e.z = expf(l.z - m); e.w = expf(l.w - m); s = e.x; s += e.y; s += e.z; s += e.w; }
    s = block_sum_exact(s, sh);
    if (active) { float4 o; o.x = e.x / s; o.y = e.y / s; o.z = e.z / s; o.w = e.w / s; reinterpret_cast<float4*>(y + row * DIM)[threadIdx.x] = o; }
}
#define TK 64
#define TILE 64
#define PAD 65
// ORDER 0: sequential FMA d = 0..63; 1: reverse d = 63..0; 2: two chains (0..31) + (32..63); 3: four chains of 16, ((c0+c1)+(c2+c3))
template <int ORDER>
__global__ void rel_band_gemm_64(const float* __restrict__ qp, const float* __restrict__ relk, float* __restrict__ out, int n, int h) {
    extern __shared__ float smem[];                       // dynamic: (64 + 127) x 65 floats = 49,660 B (> the 48 KB static limit; opt-in below)
    float* As = smem; float* Bs = smem + TILE * PAD;
    const int bh = blockIdx.z, hh = bh % h;
    const int i0 = blockIdx.y * TILE, j0 = blockIdx.x * TILE;
    const float* A = qp + (size_t)bh * n * TK + (size_t)i0 * TK;
    const int m0 = n - 1 + j0 - i0 - (TILE - 1);
    const float* B = relk + (size_t)hh * (2 * n - 1) * TK + (size_t)m0 * TK;
    const int t = threadIdx.x;
    for (int idx = t; idx < TILE * TK; idx += 256) { int r = idx / TK, d = idx % TK; As[r * PAD + d] = A[r * TK + d]; }
    for (int idx = t; idx < (2 * TILE - 1) * TK; idx += 256) { int r = idx / TK, d = idx % TK; Bs[r * PAD + d] = B[r * TK + d]; }
    __syncthreads();
    const int tx = t % 16, ty = t / 16;
    float acc[4][4], acc2[4][4], acc3[4][4], acc4[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int c = 0; c < 4; ++c) { acc[r][c] = 0.f; acc2[r][c] = 0.f; acc3[r][c] = 0.f; acc4[r][c] = 0.f; }
    const int wb = tx * 4 - ty * 4 + 63 - 3;
    #pragma unroll 8
    for (int dd = 0; dd < TK; ++dd) {
        const int d = (ORDER == 1) ? (TK - 1 - dd) : dd;
        float a[4], b[7];
        #pragma unroll
        for (int r = 0; r < 4; ++r) a[r] = As[(ty * 4 + r) * PAD + d];
        #pragma unroll
        for (int k = 0; k < 7; ++k) b[k] = Bs[(wb + k) * PAD + d];
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float bv = b[c - r + 3];
                if (ORDER <= 1) acc[r][c] = fmaf(a[r], bv, acc[r][c]);
                else if (ORDER == 2) { if (d < 32) acc[r][c] = fmaf(a[r], bv, acc[r][c]); else acc2[r][c] = fmaf(a[r], bv, acc2[r][c]); }
                else { if (d < 16) acc[r][c] = fmaf(a[r], bv, acc[r][c]); else if (d < 32) acc2[r][c] = fmaf(a[r], bv, acc2[r][c]);
                       else if (d < 48) acc3[r][c] = fmaf(a[r], bv, acc3[r][c]); else acc4[r][c] = fmaf(a[r], bv, acc4[r][c]); }
            }
    }
    float* O = out + (size_t)bh * n * n;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = i0 + ty * 4 + r;
        float v[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (ORDER <= 1) v[c] = acc[r][c];
            else if (ORDER == 2) v[c] = acc[r][c] + acc2[r][c];
            else v[c] = (acc[r][c] + acc2[r][c]) + (acc3[r][c] + acc4[r][c]);
        }
        float4 o4; o4.x = v[0]; o4.y = v[1]; o4.z = v[2]; o4.w = v[3];
        *reinterpret_cast<float4*>(O + (size_t)i * n + j0 + tx * 4) = o4;
    }
}
template <int DIM>
__global__ void add2_softmax_warp_k(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y) {
    constexpr int NP2 = (DIM <= 2048) ? 2048 : 4096; constexpr int IT = NP2 / 32;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const size_t row = warp;
    float e[IT];
    #pragma unroll
    for (int it = 0; it < IT; ++it) { const int j = lane + it * 32; e[it] = (j < DIM) ? (a[row * DIM + j] + b[row * DIM + j]) : -INFINITY; }
    float m = e[0];
    #pragma unroll
    for (int it = 1; it < IT; ++it) m = (m > e[it]) ? m : e[it];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) { float o = __shfl_xor_sync(0xffffffff, m, off); m = (m > o) ? m : o; }
    float s = 0.f;
    #pragma unroll
    for (int it = 0; it < IT; ++it) { e[it] = expf(e[it] - m); s += e[it]; }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    #pragma unroll
    for (int it = 0; it < IT; ++it) { const int j = lane + it * 32; if (j < DIM) y[row * DIM + j] = e[it] / s; }
}
torch::Tensor add2_softmax_warp(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && b.is_contiguous() && a.sizes() == b.sizes() && a.scalar_type() == torch::kFloat);
    const int dim = a.size(-1); auto y = torch::empty_like(a); const int rows = a.numel() / dim;
    TORCH_CHECK(dim == 1536, "unsupported dim");
    add2_softmax_warp_k<1536><<<(rows * 32 + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>());
    return y;
}
torch::Tensor add2_softmax_exact(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && b.is_contiguous() && a.sizes() == b.sizes() && a.scalar_type() == torch::kFloat);
    const int dim = a.size(-1); auto y = torch::empty_like(a); const int rows = a.numel() / dim;
    if (dim == 1536) add2_softmax_exact_k<1536><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>());
    else if (dim == 4096) add2_softmax_exact_k<4096><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>());
    else TORCH_CHECK(false, "unsupported softmax dim");
    return y;
}
torch::Tensor rel_band_gemm(torch::Tensor qp, torch::Tensor relk, int64_t order) {
    TORCH_CHECK(qp.is_cuda() && qp.is_contiguous() && relk.is_contiguous() && qp.size(-1) == 64 && relk.size(-1) == 64);
    int b = qp.size(0), h = qp.size(1), n = qp.size(2); TORCH_CHECK(relk.size(0) == h && relk.size(1) == 2 * n - 1 && n % 64 == 0);
    auto out = torch::empty({b, h, n, n}, qp.options());
    dim3 grid(n / 64, n / 64, b * h);
    const int smem = (TILE + 2 * TILE - 1) * PAD * sizeof(float);
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(rel_band_gemm_64<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(rel_band_gemm_64<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(rel_band_gemm_64<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(rel_band_gemm_64<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    switch (order) {
        case 0: rel_band_gemm_64<0><<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(qp.data_ptr<float>(), relk.data_ptr<float>(), out.data_ptr<float>(), n, h); break;
        case 1: rel_band_gemm_64<1><<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(qp.data_ptr<float>(), relk.data_ptr<float>(), out.data_ptr<float>(), n, h); break;
        case 2: rel_band_gemm_64<2><<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(qp.data_ptr<float>(), relk.data_ptr<float>(), out.data_ptr<float>(), n, h); break;
        default: rel_band_gemm_64<3><<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(qp.data_ptr<float>(), relk.data_ptr<float>(), out.data_ptr<float>(), n, h); break;
    }
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add2_softmax_exact", &add2_softmax_exact); m.def("add2_softmax_warp", &add2_softmax_warp); m.def("rel_band_gemm", &rel_band_gemm); }
