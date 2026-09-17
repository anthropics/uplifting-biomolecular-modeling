// kit v3_sdkfused_r1 — ports of v2_fusedln_r3's residual / residual+LN fusions to the ESMC SDK route (bf16 activations,
// packed [tokens, hidden] layout, TE LayerNormLinear's LN = transformer_engine ln_fwd_general_kernel).
//   residual_bf16 : r = bf16_rn(float(x) + float(bf16_rn(float(y) * inv_b)))  — the SDK stock's aten::div(scalar) + aten::add pair
//   residual_ln_te: the residual above (written to R) then TE's ln_fwd_general_kernel numerics on r, mirrored for cols <= 8192
//                   (the registration general<8192, WARPS_M 1, WARPS_N 4, BYTES_PER_LDG 16> that lower_bound(cols) selects for
//                   cols in (2048, 8192]; ctas_per_row 1): 128 threads per row, thread gidn = warp*32 + lane reads 8 elements at
//                   col = gidn*8 + it*1024 while col < N; mean = (sequential per-thread sum -> shfl_down 16..1 -> lane0 smem[warp]
//                   -> 0 + s0 + s1 + s2 + s3) * (1.f/N); var likewise over diff*diff (fmaf: nvcc's default contraction of
//                   `sqsigma += diff * diff`); rs = rsqrtf(var + eps); y = rs * (x - mu); z = fmaf(g, y, b); bf16 via __float2bfloat16_rn.
//   mode 0: no contraction; mode 1: fmaf on diff*diff and g*y+b; mode 2: mode 1 + rsqrtf(fmaf(sum_sq, rn, eps)) — the contraction
//                   set nvcc's default makes on TE's `sqsigma * rn + eps`; the test record names the bitwise mode (box 2: mode 1 differed on
//                   ~5e-6 of the elements = the rs contraction; mode 0 far off).
//   residual_ln_te_w1: the same numerics for 512 < cols <= 2048 — the registrations general<1024|2048, WARPS_M 4, WARPS_N 1, 16> that
//                   lower_bound(cols) selects there (960 -> <1024,...>: 4 loads per thread; 1152 -> <2048,...>: 8): ONE warp per row, 4 rows
//                   per CTA, thread lane reads 8 elements at col = lane*8 + it*256 while col < N; the row sums are Reducer<T,1,WARPS_M,1>::
//                   allreduce_ = the butterfly `for (it = 1; it < 32; it *= 2) x += shfl_xor(x, it)` (no shared memory, no smem order);
//                   everything after the reduction is the same source lines as above (same contraction set, MODE).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <vector>

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float v) { return __float2bfloat16_rn(v); }

struct __align__(16) bf16x8 { __nv_bfloat16 v[8]; };
__global__ void residual_bf16_k(const bf16x8* __restrict__ x, const bf16x8* __restrict__ y, bf16x8* __restrict__ o, long long n8, float inv_b) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;      // one 16-byte vector (8 bf16) per thread, as ATen's vectorized_elementwise_kernel<8>
  if (i >= n8) return;
  bf16x8 xv = x[i], yv = y[i], ov;
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    float t = bf2f(f2bf(__fmul_rn(bf2f(yv.v[j]), inv_b)));    // aten::div by the scalar: bf16_rn(float(y) * inv_b)
    ov.v[j] = f2bf(__fadd_rn(bf2f(xv.v[j]), t));               // aten::add: bf16_rn(float(x) + float(t))
  }
  o[i] = ov;
}

template <int MODE>
__global__ void __launch_bounds__(128) residual_ln_te_k(const int N, const float eps, const float inv_b,
                                                        const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ Y,
                                                        const __nv_bfloat16* __restrict__ gamma, const __nv_bfloat16* __restrict__ beta,
                                                        __nv_bfloat16* __restrict__ R, __nv_bfloat16* __restrict__ Z) {
  constexpr int NUM_ELTS = 8, THREADS_PER_ROW = 128, MAX_LDGS = 8;   // general<8192, 1, 4, 16>: 8192 / (128 * 8) = 8 loads per thread
  __shared__ float smem[2 * 4];
  const int tidx = threadIdx.x, lane = tidx % 32, warp_n = tidx / 32;
  const int gidn = warp_n * 32 + lane;                       // ctas_per_row == 1, bidn == 0: gidn = lane + warp_n * 32
  const long long row = blockIdx.x;
  const float rn = 1.f / (float)N;
  float xr[MAX_LDGS][NUM_ELTS];
  // residual (the v1_fused bits with the bf16 double rounding), one 16-byte vector (8 bf16 = TE's Ivec at BYTES_PER_LDG 16) per (it);
  // held in registers in TE's (it, jt) order; R written once
  #pragma unroll
  for (int it = 0; it < MAX_LDGS; it++) {
    const int col = gidn * NUM_ELTS + it * (THREADS_PER_ROW * NUM_ELTS);
    if (col < N) {
      const long long idx = row * N + col;
      const bf16x8 xv = *reinterpret_cast<const bf16x8*>(X + idx), yv = *reinterpret_cast<const bf16x8*>(Y + idx); bf16x8 rv;
      #pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        float t = bf2f(f2bf(__fmul_rn(bf2f(yv.v[jt]), inv_b)));
        rv.v[jt] = f2bf(__fadd_rn(bf2f(xv.v[jt]), t)); xr[it][jt] = bf2f(rv.v[jt]);   // TE loads the bf16 row and converts to fp32 (Ivec::to)
      }
      *reinterpret_cast<bf16x8*>(R + idx) = rv;
    }
  }
  // mean: per-thread sequential sum in (it, jt) order, then Reducer<float,1,1,4>::allreduce
  float mu = 0.f;
  #pragma unroll
  for (int it = 0; it < MAX_LDGS; it++) {
    const int col = gidn * NUM_ELTS + it * (THREADS_PER_ROW * NUM_ELTS);
    if (col < N) {
      #pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) mu = __fadd_rn(mu, xr[it][jt]);
    }
  }
  #pragma unroll
  for (int off = 16; off > 0; off /= 2) mu = __fadd_rn(mu, __shfl_down_sync(0xffffffffu, mu, off));   // Base::reduce (lane 0 holds)
  if (lane == 0) smem[warp_n] = mu;
  __syncthreads();
  { float out = 0.f;
    #pragma unroll
    for (int it = 0; it < 4; it++) out = __fadd_rn(out, smem[it]);
    mu = __fmul_rn(out, rn); }
  // variance
  float sq = 0.f;
  #pragma unroll
  for (int it = 0; it < MAX_LDGS; it++) {
    const int col = gidn * NUM_ELTS + it * (THREADS_PER_ROW * NUM_ELTS);
    if (col < N) {
      #pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        const float diff = __fsub_rn(xr[it][jt], mu);
        sq = (MODE >= 1) ? __fmaf_rn(diff, diff, sq) : __fadd_rn(sq, __fmul_rn(diff, diff));
      }
    }
  }
  #pragma unroll
  for (int off = 16; off > 0; off /= 2) sq = __fadd_rn(sq, __shfl_down_sync(0xffffffffu, sq, off));
  if (lane == 0) smem[4 + warp_n] = sq;                     // the reducer flips to its second smem buffer (use0_)
  __syncthreads();
  { float out = 0.f;
    #pragma unroll
    for (int it = 0; it < 4; it++) out = __fadd_rn(out, smem[4 + it]);
    sq = (MODE >= 2) ? __fmaf_rn(out, rn, eps) : __fadd_rn(__fmul_rn(out, rn), eps); }   // TE: rsqrtf(sqsigma * rn + eps) -> nvcc contracts the mul+add (MODE 2)
  const float rs = rsqrtf(sq);
  // output: y = rs * (x - mu); z = g * y + b -> bf16 RN (one 16-byte vector store per (it))
  #pragma unroll
  for (int it = 0; it < MAX_LDGS; it++) {
    const int col = gidn * NUM_ELTS + it * (THREADS_PER_ROW * NUM_ELTS);
    if (col < N) {
      const bf16x8 gv = *reinterpret_cast<const bf16x8*>(gamma + col), bv = *reinterpret_cast<const bf16x8*>(beta + col); bf16x8 zv;
      #pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        const float yv = __fmul_rn(rs, __fsub_rn(xr[it][jt], mu));
        const float g = bf2f(gv.v[jt]), b = bf2f(bv.v[jt]);
        const float z = (MODE >= 1) ? __fmaf_rn(g, yv, b) : __fadd_rn(__fmul_rn(g, yv), b);
        zv.v[jt] = f2bf(z);
      }
      *reinterpret_cast<bf16x8*>(Z + row * N + col) = zv;
    }
  }
}

template <int MODE, int MAX_LDGS>
__global__ void __launch_bounds__(128) residual_ln_te_w1_k(const int N, const long long M, const float eps, const float inv_b,
                                                           const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ Y,
                                                           const __nv_bfloat16* __restrict__ gamma, const __nv_bfloat16* __restrict__ beta,
                                                           __nv_bfloat16* __restrict__ R, __nv_bfloat16* __restrict__ Z) {
  constexpr int NUM_ELTS = 8, THREADS_PER_ROW = 32, WARPS_M = 4;      // general<1024|2048, 4, 1, 16>: LDGS = HIDDEN / (32 * 8) = 4 | 8
  const int tidx = threadIdx.x, lane = tidx % 32, warp_m = tidx / 32;  // WARPS_N == 1: warp_n = 0, gidn = lane
  const long long row = (long long)blockIdx.x * WARPS_M + warp_m;      // ctas_per_col covers the rows once (no grid-stride revisit changes arithmetic)
  if (row >= M) return;                                                // no barrier in this kernel: a whole warp exits together
  const float rn = 1.f / (float)N;
  float xr[MAX_LDGS][NUM_ELTS];
  #pragma unroll
  for (int it = 0; it < MAX_LDGS; it++) {
    const int col = lane * NUM_ELTS + it * (THREADS_PER_ROW * NUM_ELTS);
    if (col < N) {
      const long long idx = row * N + col;
      const bf16x8 xv = *reinterpret_cast<const bf16x8*>(X + idx), yv = *reinterpret_cast<const bf16x8*>(Y + idx); bf16x8 rv;
      #pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        float t = bf2f(f2bf(__fmul_rn(bf2f(yv.v[jt]), inv_b)));
        rv.v[jt] = f2bf(__fadd_rn(bf2f(xv.v[jt]), t)); xr[it][jt] = bf2f(rv.v[jt]);
      }
      *reinterpret_cast<bf16x8*>(R + idx) = rv;
    }
  }
  // mean: per-thread sequential sum in (it, jt) order, then the warp butterfly allreduce (offsets 1, 2, 4, 8, 16)
  float mu = 0.f;
  #pragma unroll
  for (int it = 0; it < MAX_LDGS; it++) {
    const int col = lane * NUM_ELTS + it * (THREADS_PER_ROW * NUM_ELTS);
    if (col < N) {
      #pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) mu = __fadd_rn(mu, xr[it][jt]);
    }
  }
  #pragma unroll
  for (int off = 1; off < 32; off *= 2) mu = __fadd_rn(mu, __shfl_xor_sync(0xffffffffu, mu, off));
  mu = __fmul_rn(mu, rn);
  // variance
  float sq = 0.f;
  #pragma unroll
  for (int it = 0; it < MAX_LDGS; it++) {
    const int col = lane * NUM_ELTS + it * (THREADS_PER_ROW * NUM_ELTS);
    if (col < N) {
      #pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        const float diff = __fsub_rn(xr[it][jt], mu);
        sq = (MODE >= 1) ? __fmaf_rn(diff, diff, sq) : __fadd_rn(sq, __fmul_rn(diff, diff));
      }
    }
  }
  #pragma unroll
  for (int off = 1; off < 32; off *= 2) sq = __fadd_rn(sq, __shfl_xor_sync(0xffffffffu, sq, off));
  sq = (MODE >= 2) ? __fmaf_rn(sq, rn, eps) : __fadd_rn(__fmul_rn(sq, rn), eps);
  const float rs = rsqrtf(sq);
  #pragma unroll
  for (int it = 0; it < MAX_LDGS; it++) {
    const int col = lane * NUM_ELTS + it * (THREADS_PER_ROW * NUM_ELTS);
    if (col < N) {
      const bf16x8 gv = *reinterpret_cast<const bf16x8*>(gamma + col), bv = *reinterpret_cast<const bf16x8*>(beta + col); bf16x8 zv;
      #pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        const float yv = __fmul_rn(rs, __fsub_rn(xr[it][jt], mu));
        const float g = bf2f(gv.v[jt]), b = bf2f(bv.v[jt]);
        const float z = (MODE >= 1) ? __fmaf_rn(g, yv, b) : __fadd_rn(__fmul_rn(g, yv), b);
        zv.v[jt] = f2bf(z);
      }
      *reinterpret_cast<bf16x8*>(Z + row * N + col) = zv;
    }
  }
}


static inline void chk_bf16(const torch::Tensor& t, const char* n) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous(), n, ": contiguous bf16 CUDA tensor"); }

torch::Tensor residual_bf16(torch::Tensor x, torch::Tensor y, double inv_b) {
  chk_bf16(x, "x"); chk_bf16(y, "y"); TORCH_CHECK(x.sizes() == y.sizes(), "x/y one shape");
  const long long n = x.numel(); TORCH_CHECK(n % 8 == 0, "numel % 8 == 0 (16-byte vectors)");
  auto o = torch::empty_like(x);
  TORCH_CHECK(((uintptr_t)x.data_ptr() & 15) == 0 && ((uintptr_t)y.data_ptr() & 15) == 0 && ((uintptr_t)o.data_ptr() & 15) == 0, "16-byte aligned");
  const long long n8 = n / 8;
  residual_bf16_k<<<(unsigned)((n8 + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16x8*>(x.data_ptr()), reinterpret_cast<const bf16x8*>(y.data_ptr()), reinterpret_cast<bf16x8*>(o.data_ptr()), n8, (float)inv_b);
  return o;
}

std::vector<torch::Tensor> residual_ln_te(torch::Tensor x, torch::Tensor y, double inv_b, torch::Tensor gamma, torch::Tensor beta, double eps, int64_t mode) {
  chk_bf16(x, "x"); chk_bf16(y, "y"); chk_bf16(gamma, "gamma"); chk_bf16(beta, "beta");
  const int N = (int)x.size(-1); const long long M = x.numel() / N;
  TORCH_CHECK(x.sizes() == y.sizes() && gamma.numel() == N && beta.numel() == N, "shapes");
  TORCH_CHECK(N % 8 == 0 && N > 512 && N <= 8192, "residual_ln_te mirrors TE's general<1024|2048,4,1,16> and <8192,1,4,16> configs: 512 < N <= 8192, N % 8 == 0");
  auto R = torch::empty_like(x); auto Z = torch::empty_like(x);
  TORCH_CHECK(((uintptr_t)x.data_ptr() & 15) == 0 && ((uintptr_t)y.data_ptr() & 15) == 0 && ((uintptr_t)gamma.data_ptr() & 15) == 0 && ((uintptr_t)beta.data_ptr() & 15) == 0, "16-byte aligned");
  auto st = at::cuda::getCurrentCUDAStream();
#define LAUNCH_LN(M) residual_ln_te_k<M><<<(unsigned)M_, 128, 0, st>>>(N, (float)eps, (float)inv_b, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), \
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()), reinterpret_cast<__nv_bfloat16*>(R.data_ptr()), reinterpret_cast<__nv_bfloat16*>(Z.data_ptr()))
  const long long M_ = M;
  TORCH_CHECK(mode >= 0 && mode <= 2, "mode 0/1/2");
  if (N > 2048) {                                            // general<8192, 1, 4, 16>: 128 threads per row, one row per CTA
    if (mode == 0) LAUNCH_LN(0); else if (mode == 1) LAUNCH_LN(1); else LAUNCH_LN(2);
    return {R, Z};
  }
#define LAUNCH_W1(MD, LD) residual_ln_te_w1_k<MD, LD><<<(unsigned)((M_ + 3) / 4), 128, 0, st>>>(N, M_, (float)eps, (float)inv_b, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), \
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()), reinterpret_cast<__nv_bfloat16*>(R.data_ptr()), reinterpret_cast<__nv_bfloat16*>(Z.data_ptr()))
  if (N <= 1024) {                                           // general<1024, 4, 1, 16>: 4 loads per lane
    if (mode == 0) LAUNCH_W1(0, 4); else if (mode == 1) LAUNCH_W1(1, 4); else LAUNCH_W1(2, 4);
  } else {                                                   // general<2048, 4, 1, 16>: 8 loads per lane
    if (mode == 0) LAUNCH_W1(0, 8); else if (mode == 1) LAUNCH_W1(1, 8); else LAUNCH_W1(2, 8);
  }
  return {R, Z};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("residual_bf16", &residual_bf16, "residual_bf16(x, y, inv_b) -> bf16");
  m.def("residual_ln_te", &residual_ln_te, "residual_ln_te(x, y, inv_b, gamma, beta, eps, mode) -> [R, Z]; 512 < N <= 8192; mode 2 = TE ln_fwd_general contraction set");
}

