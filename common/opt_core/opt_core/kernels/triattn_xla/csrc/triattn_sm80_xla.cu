// libtriattn_sm80_xla.so: the sm_80 triangle-attention forward of kernels/triattn/triattn_native (package generation 11, kernel directory
// cuda_80: csrc/triattn_sm80.cuh device code, csrc/launch_sm80.cuh launch template, csrc/inst_sm80.h + inst_d16/32/64.cu instantiations --
// all compiled UNMODIFIED; their digests are recorded beside the binary) behind a plain C entry point (triattn_sm80_abi.h) for the
// XLA-FFI launcher.  This file restates the host side of the package's torch binding (csrc/sm80_binding.cu: stage_bias, stage_mask, fwd)
// over raw device pointers, caller-provided scratch buffers and the caller's stream: no torch, no allocation, no module state (the fix
// buffer and the mask census the binding keeps per device are per-call scratch here; their census fields are therefore per call).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <climits>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include "launch_sm80.cuh"          // carried: kernels/triattn/triattn_native/pkg/v11/triattn_pkg/cuda_80/csrc/ (includes triattn_sm80.cuh)
#include "inst_sm80.h"
#include "triattn_sm80_abi.h"

#define TXLA_EXPORT __attribute__((visibility("default")))

// A GCC >= 13 host compiler may import std::ios_base_library_init (GLIBCXX_3.4.32) / std::__throw_bad_array_new_length (GLIBCXX_3.4.29)
// from libstdc++ at -O3; both are defined here (internal: the version script exports only the C entry points) so the library asks libstdc++
// for nothing newer than the launcher beside it does (GLIBCXX_3.4.21).
#include <ios>
#include <cstdlib>
namespace std {
void ios_base_library_init() { static std::ios_base::Init keep; (void)keep; }
void __throw_bad_array_new_length() { std::abort(); }
}  // namespace std

namespace {
using namespace triattn_sm80;

const Inst* inst_for(int D) {
#ifdef TS_HAVE_D16
  if (D == 16) return &inst_d16;
#endif
#ifdef TS_HAVE_D32
  if (D == 32) return &inst_d32;
#endif
#ifdef TS_HAVE_D64
  if (D == 64) return &inst_d64;
#endif
  return nullptr;
}

long long cta_tiles(const GeoInfo& g, long long B, long long N, long long H, long long S) {
  const long long BM = 32 * g.QG;
  return ((S + BM - 1) / BM) * ((N + g.R - 1) / g.R) * B * H;
}

int fail(TriattnSm80Call* c, int rc, const char* what, cudaError_t e = cudaSuccess) {
  if (e != cudaSuccess) std::snprintf(c->err, sizeof(c->err), "%s: %s", what, cudaGetErrorString(e));
  else std::snprintf(c->err, sizeof(c->err), "%s", what);
  c->err[sizeof(c->err) - 1] = 0;
  return rc;
}

// == sm80_binding.cu stage_bias: bias4 [B,H,S,S] (contiguous here) fp32 | bf16 -> fp32 [B,H,S128,S64] = bias / scale in accumulator-fragment order,
// -inf on keys >= S and on keys no row attends (keyany or none).
cudaError_t stage_bias(const TriattnSm80Call* c, const uint8_t* keyany, cudaStream_t st) {
  const int B = c->B, H = c->H, S = c->S, S64 = (S + 63) / 64 * 64, S128 = (S + 127) / 128 * 128;
  const long long n = (long long)B * H * S128 * S64;
  const int threads = 256;
  const int blocks = (int)std::min<long long>((n + threads - 1) / threads, (long long)triattn_sm80::detail::num_sms() * 32);
  const float inv_scale = (float)(1.0 / (double)c->scale);
  const long long sB = (long long)H * S * S, sH = (long long)S * S, sQ = S, sK = 1;
  float* out = reinterpret_cast<float*>(c->bias_staged);
  if (c->bias_dtype == 0)
    stage_bias_kernel<float><<<blocks, threads, 0, st>>>(reinterpret_cast<const float*>(c->bias), sB, sH, sQ, sK, keyany, out, B, H, S, S128, S64, inv_scale);
  else
    stage_bias_kernel<__nv_bfloat16><<<blocks, threads, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(c->bias), sB, sH, sQ, sK, keyany, out, B, H, S, S128, S64, inv_scale);
  return cudaGetLastError();
}

// == sm80_binding.cu stage_mask: mask [B,N,S] uint8 (contiguous) -> keyany [B,S64] uint8, rows [B*N,4] int32, maskw [B*N*nkt*2] int32;
// census int32[4] and rgflag int32[B*YG] are zeroed here (per call).
cudaError_t stage_mask(const TriattnSm80Call* c, int R, cudaStream_t st) {
  const int B = c->B, N = c->N, S = c->S, S64 = (S + 63) / 64 * 64, nkt = (S + BN - 1) / BN;
  const int YG = (N + R - 1) / R;
  const uint8_t* mask = reinterpret_cast<const uint8_t*>(c->mask);
  uint8_t* keyany = reinterpret_cast<uint8_t*>(c->keyany);
  int* census = reinterpret_cast<int*>(c->census); int* rgflag = reinterpret_cast<int*>(c->rgflag);
  cudaError_t e;
  if ((e = cudaMemsetAsync(census, 0, 4 * sizeof(int), st)) != cudaSuccess) return e;
  if ((e = cudaMemsetAsync(rgflag, 0, (size_t)B * YG * sizeof(int), st)) != cudaSuccess) return e;
  const long long sB = (long long)N * S, sN = S;
  mask_or_kernel<><<<dim3((S64 + 127) / 128, B), 1024, 0, st>>>(mask, sB, sN, keyany, N, S, S64);
  if ((e = cudaGetLastError()) != cudaSuccess) return e;
  const long long nrows = (long long)B * N; const int wpb = 8;
  mask_rows_kernel<><<<(unsigned)((nrows + wpb - 1) / wpb), wpb * 32, 0, st>>>(mask, sB, sN, keyany, reinterpret_cast<int4*>(c->rows), reinterpret_cast<uint32_t*>(c->maskw),
      census, rgflag, B, N, S, S64, nkt, R);
  if ((e = cudaGetLastError()) != cudaSuccess) return e;
  const int nflags = B * YG;
  count_flags_kernel<><<<std::max(1, std::min((nflags + 255) / 256, 64)), 256, 0, st>>>(rgflag, nflags, census);
  return cudaGetLastError();
}

int fwd_impl(TriattnSm80Call* c) {
  const int B = c->B, N = c->N, H = c->H, S = c->S, D = c->D;
  const Inst* inp = inst_for(D);
  if (inp == nullptr) return fail(c, 4, "head dim not built into this library");
  const Inst& in = *inp;
  if (B < 1 || N < 1 || H < 1 || S < 1) return fail(c, 4, "B, N, H, S must be >= 1");
  if (c->bias_dtype != 0 && c->bias_dtype != 1) return fail(c, 4, "bias dtype must be fp32 (0) or bf16 (1)");
  for (int i = 0; i < 4; ++i) {
    if (c->sq[i] % 8 || c->sk[i] % 8 || c->sv[i] % 8 || c->sq[i] < 0 || c->sk[i] < 0 || c->sv[i] < 0) return fail(c, 4, "q/k/v strides must be non-negative multiples of 8 elements");
    if (c->so[i] % 2 || c->so[i] < 0) return fail(c, 4, "out strides must be non-negative and even");
  }
  if ((reinterpret_cast<uintptr_t>(c->q) | reinterpret_cast<uintptr_t>(c->k) | reinterpret_cast<uintptr_t>(c->v)) % 16) return fail(c, 4, "q/k/v must be 16-byte aligned");
  cudaStream_t st = reinterpret_cast<cudaStream_t>(c->stream);
  // geometry (== sm80_binding.cu fwd): small-S tiles when S <= small_max, or S <= small_max_rows for densely strided rows
  const bool dense_rows = c->sq[3] <= 512 && c->sk[3] <= 512 && c->sv[3] <= 512;
  const bool small = (in.small_max > 0 && S <= in.small_max) || (dense_rows && in.small_max_rows > 0 && S <= in.small_max_rows);
  const GeoInfo& g = small ? in.small : in.big;
  c->small_used = small ? 1 : 0;
  const long long nct = cta_tiles(g, B, N, H, S);
  if (c->fix_elems < FIX_LIST + 3 * nct) { std::snprintf(c->err, sizeof(c->err), "fix buffer: %lld int32 < %lld needed", (long long)c->fix_elems, (long long)(FIX_LIST + 3 * nct)); return 4; }
  cudaError_t e;
  if ((e = cudaMemsetAsync(c->fix, 0, FIX_LIST * sizeof(int), st)) != cudaSuccess) return fail(c, 5, "memset(fix header)", e);   // per-call census (the binding keeps it per device)
  const uint8_t* keyany = nullptr;
  if (c->has_mask) {
    if ((e = stage_mask(c, g.R, st)) != cudaSuccess) return fail(c, 5, "stage_mask", e);
    keyany = reinterpret_cast<const uint8_t*>(c->keyany);
  }
  if ((e = stage_bias(c, keyany, st)) != cudaSuccess) return fail(c, 5, "stage_bias", e);
  Args a{};
  a.q = reinterpret_cast<const __nv_bfloat16*>(c->q); a.q_sB = c->sq[0]; a.q_sN = c->sq[1]; a.q_sH = c->sq[2]; a.q_sS = c->sq[3];
  a.k = reinterpret_cast<const __nv_bfloat16*>(c->k); a.k_sB = c->sk[0]; a.k_sN = c->sk[1]; a.k_sH = c->sk[2]; a.k_sS = c->sk[3];
  a.v = reinterpret_cast<const __nv_bfloat16*>(c->v); a.v_sB = c->sv[0]; a.v_sN = c->sv[1]; a.v_sH = c->sv[2]; a.v_sS = c->sv[3];
  a.out = reinterpret_cast<__nv_bfloat16*>(c->out); a.o_sB = c->so[0]; a.o_sN = c->so[1]; a.o_sH = c->so[2]; a.o_sS = c->so[3];
  a.bias = reinterpret_cast<const float*>(c->bias_staged);
  a.rows = c->has_mask ? reinterpret_cast<const int4*>(c->rows) : nullptr;
  a.maskw = c->has_mask ? reinterpret_cast<const uint32_t*>(c->maskw) : nullptr;
  a.lse = c->want_lse ? reinterpret_cast<float*>(c->lse) : nullptr;
  a.fix = reinterpret_cast<int*>(c->fix);
  a.B = B; a.N = N; a.H = H; a.S_q = S; a.S_kv = S;
  a.qb = 0;                                                        // default q-tile block of the CTA order
  a.c1 = (float)((double)c->scale * 1.4426950408889634);
  a.dbg = 0;
  const int variant = c->want_lse ? 2 : (c->has_mask ? 1 : 0);
  const char* stage = "";
  e = in.run(a, variant, small, st, &stage);
  if (e != cudaSuccess) {
    std::snprintf(c->err, sizeof(c->err), "triattn_sm80 %s: %s (grid %dx%dx%d, block %d, smem %d/%d%s)", stage, cudaGetErrorString(e), (S + 32 * g.QG - 1) / (32 * g.QG), (N + g.R - 1) / g.R, B * H,
                  32 * (g.R / g.RW) * g.QG, g.smem_hot, g.smem_safe, small ? ", small-S geometry" : "");
    return 6;
  }
  return 0;
}
}  // namespace

extern "C" {
TXLA_EXPORT int triattn_sm80_xla_fwd(TriattnSm80Call* c) {
  if (c == nullptr) return 2;
  c->err[0] = 0;
  if (c->abi_version != TRIATTN_SM80_ABI_VERSION) { std::snprintf(c->err, sizeof(c->err), "ABI version %d != %d", c->abi_version, TRIATTN_SM80_ABI_VERSION); return 3; }
  return fwd_impl(c);
}
TXLA_EXPORT int64_t triattn_sm80_xla_fix_elems(int B, int N, int H, int S, int D) {
  const Inst* in = inst_for(D);
  if (in == nullptr) return -1;
  return FIX_LIST + 3 * std::max(cta_tiles(in->big, B, N, H, S), cta_tiles(in->small, B, N, H, S));
}
TXLA_EXPORT int triattn_sm80_xla_dims(int* out3) {
  int n = 0;
#ifdef TS_HAVE_D16
  out3[n++] = 16;
#endif
#ifdef TS_HAVE_D32
  out3[n++] = 32;
#endif
#ifdef TS_HAVE_D64
  out3[n++] = 64;
#endif
  return n;
}
TXLA_EXPORT int triattn_sm80_xla_geometry(int D, int small, int* out10) {
  const Inst* inp = inst_for(D);
  if (inp == nullptr) return 1;
  const GeoInfo& g = small ? inp->small : inp->big;
  const int vals[10] = {g.R, g.QG, g.STAGES, g.R / g.RW * g.QG, 32 * g.QG, g.smem_hot, g.smem_safe, g.RW, inp->small_max, inp->small_max_rows};
  for (int i = 0; i < 10; ++i) out10[i] = vals[i];
  return 0;
}
TXLA_EXPORT const char* triattn_sm80_xla_describe(void) {
  static char buf[200];
  int d[3]; const int n = triattn_sm80_xla_dims(d);
  char dims[32] = ""; int off = 0;
  for (int i = 0; i < n && off < 28; ++i) off += std::snprintf(dims + off, sizeof(dims) - off, i ? "/%d" : "%d", d[i]);
  const Inst* in = inst_for(32);
  std::snprintf(buf, sizeof(buf), "triattn_sm80 sm_80 D=%s; D32: R=%d BM=%d stages=%d smem=%d/%d (large-S), R=%d BM=%d smem=%d/%d (small-S, S<=%d | dense rows S<=%d) abi=%d",
                dims, in ? in->big.R : 0, in ? 32 * in->big.QG : 0, in ? in->big.STAGES : 0, in ? in->big.smem_hot : 0, in ? in->big.smem_safe : 0,
                in ? in->small.R : 0, in ? 32 * in->small.QG : 0, in ? in->small.smem_hot : 0, in ? in->small.smem_safe : 0, in ? in->small_max : 0, in ? in->small_max_rows : 0, TRIATTN_SM80_ABI_VERSION);
  return buf;
}
}
