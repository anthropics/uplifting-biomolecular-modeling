// v0_ew: exact fused elementwise kernels for the ProGen2 block (stock: salesforce/progen models/progen/modeling_progen.py,
// transformers 4.16.2 activations.gelu_new, torch 2.8.0+cu128 ATen CUDA kernels). Every arithmetic op of a stock chain is
// reproduced with an explicit round-to-nearest intrinsic (__fmul_rn / __fadd_rn / __fsub_rn / __fdiv_rn / __fmaf_rn) at the
// point the stock rounds; fp16 tensors round through cvt.rn.f16.f32 exactly where the stock materialises an fp16 tensor.
// Library transcendentals (tanhf, rsqrtf) are the CUDA math library's, compiled with nvcc's defaults (no fast-math, fmad on
// for the library code only; our chains never contain a contractable expression). C ABI: every launcher takes raw device
// pointers, sizes/strides in elements and the CUDA stream, launches on that stream and returns cudaGetLastError().
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float h2f(__half h) { return __half2float(h); }
__device__ __forceinline__ __half f2h(float f) { return __float2half_rn(f); }
template <typename T> __device__ __forceinline__ float tof(T v);
template <> __device__ __forceinline__ float tof<float>(float v) { return v; }
template <> __device__ __forceinline__ float tof<__half>(__half v) { return h2f(v); }
template <typename T> __device__ __forceinline__ T fromf(float v);
template <> __device__ __forceinline__ float fromf<float>(float v) { return v; }
template <> __device__ __forceinline__ __half fromf<__half>(float v) { return f2h(v); }
// round an fp32 opmath value to T and back (the stock's materialisation of a T tensor between two kernels)
template <typename T> __device__ __forceinline__ float rt(float v) { return tof<T>(fromf<T>(v)); }

// ------------------------------------------------------------------------------------------------------------ gelu_new
// transformers 4.16.2 activations.py gelu_new(x) = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
// evaluated by torch as 8 kernels, in Python's evaluation order:
//   t1 = 0.5 * x            (AUnaryFunctor mul, opmath fp32, rounds to T)
//   p  = pow(x, 3.0)        (PowKernel d_exp == 3 fast path: base * base * base in T's own arithmetic -> for Half two fp16 roundings)
//   t3 = c0 * p             (c0 = 0.044715 as the kernel's opmath scalar; the host passes it)
//   t4 = x + t3             (CUDAFunctor_add: a + 1*b)
//   t5 = c1 * t4            (c1 = sqrt(2/pi) as the kernel's opmath scalar)
//   t6 = tanh(t5)           (tanh_kernel_cuda: ::tanh in opmath fp32 -> tanhf)
//   t7 = t6 + 1.0           (CUDAFunctorOnSelf_add)
//   y  = t1 * t7            (BinaryFunctor mul)
template <typename T>
__global__ void gelu_new_kernel(const T* __restrict__ x, T* __restrict__ y, int64_t n, float c0, float c1) {
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += stride) {
    float xf = tof<T>(x[i]);
    float t1 = rt<T>(__fmul_rn(0.5f, xf));
    float p1 = rt<T>(__fmul_rn(xf, xf));
    float p  = rt<T>(__fmul_rn(p1, xf));
    float t3 = rt<T>(__fmul_rn(c0, p));
    float t4 = rt<T>(__fadd_rn(xf, t3));
    float t5 = rt<T>(__fmul_rn(c1, t4));
    float t6 = rt<T>(tanhf(t5));
    float t7 = rt<T>(__fadd_rn(t6, 1.0f));
    y[i] = fromf<T>(__fmul_rn(t1, t7));
  }
}

// Under torch.cuda.amp.autocast (the likelihood route) the SAME Python line runs a different chain on an fp16 x: torch.pow is on
// autocast's fp32 list, so p = (x*x)*x in fp32 (two RN roundings, no fp16 narrowing), and every op downstream of it promotes to
// fp32 (t3, t4 = float(x) + t3, t5, tanh, t7); only 0.5 * x stays an fp16 op (t1 rounded to fp16, then promoted); the output
// y = RN32(float(t1) * t7) is fp32, which the fc_out Linear's autocast cast then rounds to fp16 (cvt.rn.f16.f32). OUT_HALF folds
// that cast (exact by construction: the same RN of the same fp32 value); OUT_HALF = 0 emits the stock's fp32 tensor.
template <int OUT_HALF>
__global__ void gelu_new_autocast_kernel(const __half* __restrict__ x, void* __restrict__ y, int64_t n, float c0, float c1) {
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += stride) {
    float xf = h2f(x[i]);
    float t1 = h2f(f2h(__fmul_rn(0.5f, xf)));
    float p1 = __fmul_rn(xf, xf);
    float p  = __fmul_rn(p1, xf);
    float t3 = __fmul_rn(c0, p);
    float t4 = __fadd_rn(xf, t3);
    float t5 = __fmul_rn(c1, t4);
    float t6 = tanhf(t5);
    float t7 = __fadd_rn(t6, 1.0f);
    float r = __fmul_rn(t1, t7);
    if (OUT_HALF) ((__half*)y)[i] = f2h(r); else ((float*)y)[i] = r;
  }
}

// ------------------------------------------------------------------------------------------------- rotary + head split
// ProGenAttention.forward lines 157-197 folded: qkv (B, L, 3E) -> reshape (B, L, 8, 3E/8) -> split (query, value, key) of E/8 ->
// _split_heads -> (B, L, H, hd) with head h = m * (H/8) + g at column m*(3E/8) + which*(E/8) + g*hd (which: 0 query, 1 value, 2 key);
// rotary on the first rd columns of each head: out[2j] = RN(RN(x[2j]*cos[p,j]) + RN((-x[2j+1])*sin[p,j])),
// out[2j+1] = RN(RN(x[2j+1]*cos[p,j]) + RN(x[2j]*sin[p,j])) in fp32 (x promoted from fp16 exactly when the qkv is fp16; the
// stock's repeat_interleave(2) makes both columns of pair j read table column j); the pass-through columns are cast to fp32
// (the torch.cat([k_rot, k_pass]) promotion); value is a pure copy in the qkv dtype. Output element strides are arguments so
// the same kernel writes the stock's (B, L, H, hd) buffers or a cache slot (B, H, pos, hd) directly.
template <typename T>
__global__ void rotary_split_qkv_kernel(const T* __restrict__ qkv, int64_t qkv_sb, int64_t qkv_sl,
                                        int B, int L, int H, int hd, int rd, int E, int offset,
                                        const float* __restrict__ costab, const float* __restrict__ sintab,
                                        float* __restrict__ q_out, int64_t q_sb, int64_t q_sl, int64_t q_sh,
                                        float* __restrict__ k_out, int64_t k_sb, int64_t k_sl, int64_t k_sh,
                                        T* __restrict__ v_out, int64_t v_sb, int64_t v_sl, int64_t v_sh) {
  const int hp = hd >> 1;                       // pairs per head
  const int Hg = H >> 3;                        // heads per mp group (mp_num = 8)
  const int Eg = E >> 3;                        // local_dim = E / 8
  const int64_t total = (int64_t)B * L * H * hp;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total; idx += stride) {
    int j = (int)(idx % hp);
    int64_t r = idx / hp;
    int h = (int)(r % H); r /= H;
    int l = (int)(r % L);
    int b = (int)(r / L);
    int m = h / Hg, g = h - m * Hg;
    int64_t base = (int64_t)b * qkv_sb + (int64_t)l * qkv_sl + (int64_t)m * (3 * Eg) + (int64_t)g * hd + 2 * j;
    T q0 = qkv[base], q1 = qkv[base + 1];
    T v0 = qkv[base + Eg], v1 = qkv[base + Eg + 1];
    T k0 = qkv[base + 2 * Eg], k1 = qkv[base + 2 * Eg + 1];
    float xq0 = tof<T>(q0), xq1 = tof<T>(q1), xk0 = tof<T>(k0), xk1 = tof<T>(k1);
    float qo0, qo1, ko0, ko1;
    if (2 * j < rd) {
      int64_t t = (int64_t)(offset + l) * (rd >> 1) + j;
      float c = costab[t], s = sintab[t];
      qo0 = __fadd_rn(__fmul_rn(xq0, c), __fmul_rn(-xq1, s));
      qo1 = __fadd_rn(__fmul_rn(xq1, c), __fmul_rn(xq0, s));
      ko0 = __fadd_rn(__fmul_rn(xk0, c), __fmul_rn(-xk1, s));
      ko1 = __fadd_rn(__fmul_rn(xk1, c), __fmul_rn(xk0, s));
    } else {
      qo0 = xq0; qo1 = xq1; ko0 = xk0; ko1 = xk1;
    }
    int64_t qo = (int64_t)b * q_sb + (int64_t)l * q_sl + (int64_t)h * q_sh + 2 * j;
    int64_t ko = (int64_t)b * k_sb + (int64_t)l * k_sl + (int64_t)h * k_sh + 2 * j;
    int64_t vo = (int64_t)b * v_sb + (int64_t)l * v_sl + (int64_t)h * v_sh + 2 * j;
    q_out[qo] = qo0; q_out[qo + 1] = qo1;
    k_out[ko] = ko0; k_out[ko + 1] = ko1;
    v_out[vo] = v0;  v_out[vo + 1] = v1;
  }
}

// ------------------------------------------------------------------------------------------------------ residual adds
// ProGenBlock.forward line 275: hidden_states = attn_output + feed_forward_hidden_states + residual
// = RN_Tab(a + f) materialised in the dtype of a/f, then RN_Tout(s + r) in the promoted dtype of (s, r).
template <typename Tab, typename Tr, typename Tout>
__global__ void residual_add2_kernel(const Tab* __restrict__ a, const Tab* __restrict__ f, const Tr* __restrict__ r,
                                     Tout* __restrict__ out, int64_t n) {
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += stride) {
    float s = rt<Tab>(__fadd_rn(tof<Tab>(a[i]), tof<Tab>(f[i])));
    out[i] = fromf<Tout>(__fadd_rn(s, tof<Tr>(r[i])));
  }
}

// ----------------------------------------------------------------------------------------------------- attention glue
// ProGenAttention._attn lines 128-133: w / scale_attn (torch's CPU-scalar true division = multiply by the fp32 reciprocal the host
// computes as opmath(1)/opmath(scale)), torch.where(causal_mask, ., masked_bias.to(dtype)) with causal_mask = tril[key_len - q_len + i, jj],
// then + attention_mask (B, 1, 1, key_len) promoted to the weights' dtype. In T (fp32 in generate; fp16 under the likelihood autocast).
template <typename T, typename TM>
__global__ void attn_glue_kernel(const T* __restrict__ w, T* __restrict__ out, int B, int H, int Lq, int Lk, int key_length,
                                 float inv_scale, float masked_value, const TM* __restrict__ amask, int64_t am_sb, int64_t am_sk) {
  const int64_t total = (int64_t)B * H * Lq * Lk;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total; idx += stride) {
    int jj = (int)(idx % Lk);
    int64_t r = idx / Lk;
    int i = (int)(r % Lq); r /= Lq;
    int b = (int)(r / H);
    bool causal = jj <= (key_length - Lq + i);
    float v = causal ? rt<T>(__fmul_rn(tof<T>(w[idx]), inv_scale)) : masked_value;
    if (amask != nullptr) v = __fadd_rn(v, tof<TM>(amask[(int64_t)b * am_sb + (int64_t)jj * am_sk]));
    out[idx] = fromf<T>(v);
  }
}

// ------------------------------------------------------------------------------------------------------- layer norm
// A replica of ATen's vectorized_layer_norm_kernel<T, float> (aten/src/ATen/native/cuda/layer_norm_kernel.cu, torch 2.8.0), the
// arithmetic of that source file as compiled for sm_90 in the pinned torch build: block (32, 4), FOUR elements
// per vector for both dtypes (n_vec = N / 4: LDG.64 for Half, LDG.128 for float), thread thrx = tx + 32*ty consumes vectors thrx,
// thrx+128, ... of the row through Welford's online update (count+1 -> rcp.rn -> mean = fma(delta, rcp, mean) -> sigma2 =
// fma(delta, val - new_mean, sigma2)), the warp reduces by shfl_down (16, 8, 4, 2, 1) with cuWelfordCombine(self, other)
// (coef = rcp.rn(count); nA = other.count*coef; nB = self.count*coef; mean = fma(nA, other.mean, nB*self.mean);
// sigma2 = fma(nB, (delta*delta)*other.count, other.sigma2 + self.sigma2)), the four warps by the shared-memory halving
// (offsets 2, 1), sigma2 / N (div.rn), rstd = rsqrtf(var + eps) (MUFU.RSQ with the denormal rescale), y = fma(gamma, rstd * (x - mean), beta).
// The VARIANT bits keep the alternative contraction candidates selectable (BUILD.json ln_variant names the one whose output equals the stock's bit for bit):
//   bit 0: online mean = fma(delta, 1/n, mean) [0] or mean + RN(delta / n) [1]
//   bit 1: combine mean = fma(nA, meanA, RN(nB*meanB)) [0] or fma(nB, meanB, RN(nA*meanA)) [1]
//   bit 2: online sigma2 = fma(delta, val - new_mean, sigma2) [0] or RN(sigma2 + RN(delta * (val - new_mean))) [1]
struct WD { float mean, sigma2, count; };

template <int VARIANT>
__device__ __forceinline__ WD welford_online(float val, WD c) {
  float delta = __fsub_rn(val, c.mean);
  float nc = __fadd_rn(c.count, 1.0f);
  float nm;
  if (VARIANT & 1) nm = __fadd_rn(c.mean, __fdiv_rn(delta, nc));
  else             nm = __fmaf_rn(delta, __fdiv_rn(1.0f, nc), c.mean);
  float d2 = __fsub_rn(val, nm);
  float s2;
  if (VARIANT & 4) s2 = __fadd_rn(c.sigma2, __fmul_rn(delta, d2));
  else             s2 = __fmaf_rn(delta, d2, c.sigma2);
  WD r; r.mean = nm; r.sigma2 = s2; r.count = nc; return r;
}

template <int VARIANT>
__device__ __forceinline__ WD welford_combine(WD dB, WD dA) {   // cuWelfordCombine(dataB = self, dataA = other)
  float delta = __fsub_rn(dB.mean, dA.mean);
  float count = __fadd_rn(dA.count, dB.count);
  WD r; r.count = count; r.mean = 0.0f; r.sigma2 = 0.0f;
  if (count > 0.0f) {
    float coef = __fdiv_rn(1.0f, count);
    float nA = __fmul_rn(dA.count, coef);
    float nB = __fmul_rn(dB.count, coef);
    if (VARIANT & 2) r.mean = __fmaf_rn(nB, dB.mean, __fmul_rn(nA, dA.mean));
    else             r.mean = __fmaf_rn(nA, dA.mean, __fmul_rn(nB, dB.mean));
    float dd = __fmul_rn(__fmul_rn(delta, delta), dA.count);
    r.sigma2 = __fmaf_rn(dd, nB, __fadd_rn(dA.sigma2, dB.sigma2));
  }
  return r;
}

template <typename T, int VARIANT>
__global__ void __launch_bounds__(128) layer_norm_kernel(const T* __restrict__ X, const T* __restrict__ gamma, const T* __restrict__ beta,
                                                         T* __restrict__ Y, float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                                         int N, float eps) {
  __shared__ float s_data[8];                               // meansigmabuf[0..3] (2 per upper warp), countbuf[4..5]
  const int64_t row = blockIdx.x;
  const T* x = X + row * (int64_t)N;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int numx = blockDim.x * blockDim.y;
  const int thrx = tx + ty * blockDim.x;
  const int n_vec = N / 4;
  WD wd; wd.mean = 0.0f; wd.sigma2 = 0.0f; wd.count = 0.0f;
  for (int i = thrx; i < n_vec; i += numx) {
    #pragma unroll
    for (int ii = 0; ii < 4; ++ii) wd = welford_online<VARIANT>(tof<T>(x[(int64_t)i * 4 + ii]), wd);
  }
  for (int off = 16; off > 0; off >>= 1) {
    WD o; o.mean = __shfl_down_sync(0xffffffffu, wd.mean, off); o.sigma2 = __shfl_down_sync(0xffffffffu, wd.sigma2, off);
    o.count = __shfl_down_sync(0xffffffffu, wd.count, off);
    wd = welford_combine<VARIANT>(wd, o);
  }
  float* meansigmabuf = s_data;
  float* countbuf = s_data + blockDim.y;
  for (int off = blockDim.y / 2; off > 0; off /= 2) {
    if (tx == 0 && ty >= off && ty < 2 * off) {
      int wy = ty - off;
      meansigmabuf[2 * wy] = wd.mean; meansigmabuf[2 * wy + 1] = wd.sigma2; countbuf[wy] = wd.count;
    }
    __syncthreads();
    if (tx == 0 && ty < off) {
      WD o; o.mean = meansigmabuf[2 * ty]; o.sigma2 = meansigmabuf[2 * ty + 1]; o.count = countbuf[ty];
      wd = welford_combine<VARIANT>(wd, o);
    }
    __syncthreads();
  }
  if (tx == 0 && ty == 0) { meansigmabuf[0] = wd.mean; meansigmabuf[1] = __fdiv_rn(wd.sigma2, (float)N); }
  __syncthreads();
  const float mean = meansigmabuf[0];
  const float var = meansigmabuf[1];
  const float rstd = rsqrtf(__fadd_rn(var, eps));
  T* y = Y + row * (int64_t)N;
  for (int i = thrx; i < n_vec; i += numx) {
    #pragma unroll
    for (int ii = 0; ii < 4; ++ii) {
      int64_t e = (int64_t)i * 4 + ii;
      float xv = tof<T>(x[e]);
      float v = __fmaf_rn(tof<T>(gamma[e]), __fmul_rn(rstd, __fsub_rn(xv, mean)), tof<T>(beta[e]));
      y[e] = fromf<T>(v);
    }
  }
  if (thrx == 0) { if (mean_out) mean_out[row] = mean; if (rstd_out) rstd_out[row] = rstd; }
}

inline int grid_for(int64_t n, int block) {
  int64_t g = (n + block - 1) / block;
  if (g > 65535) g = 65535;
  if (g < 1) g = 1;
  return (int)g;
}

}  // namespace

extern "C" {

int ew_gelu_new(const void* x, void* y, int64_t n, int is_half, float c0, float c1, cudaStream_t stream) {
  const int block = 256;
  if (is_half) gelu_new_kernel<__half><<<grid_for(n, block), block, 0, stream>>>((const __half*)x, (__half*)y, n, c0, c1);
  else         gelu_new_kernel<float><<<grid_for(n, block), block, 0, stream>>>((const float*)x, (float*)y, n, c0, c1);
  return (int)cudaGetLastError();
}

// mode 2: autocast chain, fp32 output; mode 3: autocast chain with the fc_out cast folded (fp16 output)
int ew_gelu_new_autocast(const void* x, void* y, int64_t n, int out_half, float c0, float c1, cudaStream_t stream) {
  const int block = 256;
  if (out_half) gelu_new_autocast_kernel<1><<<grid_for(n, block), block, 0, stream>>>((const __half*)x, y, n, c0, c1);
  else          gelu_new_autocast_kernel<0><<<grid_for(n, block), block, 0, stream>>>((const __half*)x, y, n, c0, c1);
  return (int)cudaGetLastError();
}

int ew_rotary_split_qkv(const void* qkv, int64_t qkv_sb, int64_t qkv_sl, int is_half, int B, int L, int H, int hd, int rd, int E, int offset,
                        const float* costab, const float* sintab,
                        float* q_out, int64_t q_sb, int64_t q_sl, int64_t q_sh,
                        float* k_out, int64_t k_sb, int64_t k_sl, int64_t k_sh,
                        void* v_out, int64_t v_sb, int64_t v_sl, int64_t v_sh, cudaStream_t stream) {
  const int block = 256;
  int64_t total = (int64_t)B * L * H * (hd / 2);
  if (is_half)
    rotary_split_qkv_kernel<__half><<<grid_for(total, block), block, 0, stream>>>((const __half*)qkv, qkv_sb, qkv_sl, B, L, H, hd, rd, E, offset,
        costab, sintab, q_out, q_sb, q_sl, q_sh, k_out, k_sb, k_sl, k_sh, (__half*)v_out, v_sb, v_sl, v_sh);
  else
    rotary_split_qkv_kernel<float><<<grid_for(total, block), block, 0, stream>>>((const float*)qkv, qkv_sb, qkv_sl, B, L, H, hd, rd, E, offset,
        costab, sintab, q_out, q_sb, q_sl, q_sh, k_out, k_sb, k_sl, k_sh, (float*)v_out, v_sb, v_sl, v_sh);
  return (int)cudaGetLastError();
}

// dtype codes: 0 = fp32, 1 = fp16; (ab, r, out) in {(0,0,0), (1,1,1), (1,0,0)}
int ew_residual_add2(const void* a, const void* f, const void* r, void* out, int64_t n, int ab_half, int r_half, int out_half, cudaStream_t stream) {
  const int block = 256;
  int g = grid_for(n, block);
  if (!ab_half && !r_half && !out_half)
    residual_add2_kernel<float, float, float><<<g, block, 0, stream>>>((const float*)a, (const float*)f, (const float*)r, (float*)out, n);
  else if (ab_half && r_half && out_half)
    residual_add2_kernel<__half, __half, __half><<<g, block, 0, stream>>>((const __half*)a, (const __half*)f, (const __half*)r, (__half*)out, n);
  else if (ab_half && !r_half && !out_half)
    residual_add2_kernel<__half, float, float><<<g, block, 0, stream>>>((const __half*)a, (const __half*)f, (const float*)r, (float*)out, n);
  else
    return -1;
  return (int)cudaGetLastError();
}

int ew_attn_glue(const void* w, void* out, int is_half, int B, int H, int Lq, int Lk, int key_length, float inv_scale, float masked_value,
                 const void* amask, int amask_half, int64_t am_sb, int64_t am_sk, cudaStream_t stream) {
  const int block = 256;
  int64_t total = (int64_t)B * H * Lq * Lk;
  int g = grid_for(total, block);
  if (is_half) {
    if (amask_half) attn_glue_kernel<__half, __half><<<g, block, 0, stream>>>((const __half*)w, (__half*)out, B, H, Lq, Lk, key_length, inv_scale, masked_value, (const __half*)amask, am_sb, am_sk);
    else            attn_glue_kernel<__half, float><<<g, block, 0, stream>>>((const __half*)w, (__half*)out, B, H, Lq, Lk, key_length, inv_scale, masked_value, (const float*)amask, am_sb, am_sk);
  } else {
    if (amask_half) attn_glue_kernel<float, __half><<<g, block, 0, stream>>>((const float*)w, (float*)out, B, H, Lq, Lk, key_length, inv_scale, masked_value, (const __half*)amask, am_sb, am_sk);
    else            attn_glue_kernel<float, float><<<g, block, 0, stream>>>((const float*)w, (float*)out, B, H, Lq, Lk, key_length, inv_scale, masked_value, (const float*)amask, am_sb, am_sk);
  }
  return (int)cudaGetLastError();
}

int ew_layer_norm(const void* X, const void* gamma, const void* beta, void* Y, float* mean_out, float* rstd_out, int64_t M, int N, float eps,
                  int is_half, int variant, cudaStream_t stream) {
  if (N % 8 != 0 || N > (1 << 24) || M < 1) return -2;
  dim3 block(32, 4, 1);
  dim3 grid((unsigned)M);
#define LN_LAUNCH(T, V) layer_norm_kernel<T, V><<<grid, block, 0, stream>>>((const T*)X, (const T*)gamma, (const T*)beta, (T*)Y, mean_out, rstd_out, N, eps)
#define LN_DISPATCH(T) switch (variant) { case 0: LN_LAUNCH(T, 0); break; case 1: LN_LAUNCH(T, 1); break; case 2: LN_LAUNCH(T, 2); break; case 3: LN_LAUNCH(T, 3); break; \
                                          case 4: LN_LAUNCH(T, 4); break; case 5: LN_LAUNCH(T, 5); break; case 6: LN_LAUNCH(T, 6); break; case 7: LN_LAUNCH(T, 7); break; default: return -3; }
  if (is_half) { LN_DISPATCH(__half) } else { LN_DISPATCH(float) }
#undef LN_LAUNCH
#undef LN_DISPATCH
  return (int)cudaGetLastError();
}

int ew_version(void) { return 1; }

}  // extern "C"
