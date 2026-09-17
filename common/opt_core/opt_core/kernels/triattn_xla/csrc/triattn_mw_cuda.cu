// triattn_xla CUDA library: the sm_90a triangle-attention forward of kernels/triattn/cuda_sm90a/csrc/ (triattn_mw.cu: device code + a torch host binding)
// behind a plain C entry point (triattn_cuda_abi.h) for the XLA-FFI launcher.  Compiled twice: as is (libtriattn_mw_cuda.so) and with
// -DTXLA_LSE against the device section + csrc/cuda_lse.patch (libtriattn_mw_cuda_lse.so, entry triattn_mw_cuda_fwd_lse, which also
// writes each query row's log2-sum-exp to c->lse).  The DEVICE code is included unmodified: the build
// (build.py cuda) writes mw_device.inc = every line of that triattn_mw.cu before its torch host section (the line
// `#include <torch/types.h>`), records the sha256 of both the whole file and that prefix in ../manifest.json, and compiles this
// translation unit with nvcc for sm_90a with a static CUDA runtime.  The HOST code below restates that file's torch binding
// (stage_mask, stage_bias, fwd) over raw device pointers: same kernels, same launch geometry, same staging order, same scratch
// layout; buffers come from the caller (XLA) instead of the torch allocator, and the zero-initialised ones are cleared here with
// cudaMemsetAsync on the caller's stream.  No torch, no Python, no XLA header.
#include "mw_device.inc"
#include "triattn_cuda_abi.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

struct Fail : std::runtime_error { using std::runtime_error::runtime_error; };

#define TX_CHECK(cond, msg) do { if (!(cond)) throw Fail(std::string(msg)); } while (0)
#define TX_CUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) throw Fail(std::string(#call) + ": " + cudaGetErrorString(e_)); } while (0)
#define TX_LAUNCH_CHECK(what) do { cudaError_t e_ = cudaGetLastError(); if (e_ != cudaSuccess) throw Fail(std::string(what) + ": " + cudaGetErrorString(e_)); } while (0)

struct Strides { long long sB, sN, sH, sS; };            // element strides of a [B,N,H,S,D]-indexed view (d stride 1)

Strides strides_of(int layout, long long N, long long H, long long S, long long D) {
  Strides s;
  if (layout == 0) { s.sS = D; s.sH = S * D; s.sN = H * S * D; s.sB = N * H * S * D; }          // [B,N,H,S,D]
  else             { s.sH = D; s.sS = H * D; s.sN = S * H * D; s.sB = N * S * H * D; }          // [B,N,S,H,D]
  return s;
}

CUtensorMap map_qkv(const void* base, const Strides& st, long long B, long long N, long long H, long long S, int rows) {   // box = R pair rows x rows x D
  const uint64_t dims[5] = {(uint64_t)mw::D, (uint64_t)S, (uint64_t)H, (uint64_t)N, (uint64_t)B};
  const uint64_t str[4] = {(uint64_t)st.sS * 2, (uint64_t)st.sH * 2, (uint64_t)st.sN * 2, (uint64_t)st.sB * 2};
  const uint32_t box[5] = {(uint32_t)mw::D, (uint32_t)rows, 1, (uint32_t)mw::R, 1};
  return mw::make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, const_cast<void*>(base), dims, str, box, CU_TENSOR_MAP_SWIZZLE_64B);
}

const int kMaxDev = 64;
bool g_configured[kMaxDev] = {false};
int g_num_sms[kMaxDev] = {0};

void configure_device(int dev, int smem) {
  if (dev < 0 || dev >= kMaxDev) throw Fail("device ordinal out of range");
  if (g_configured[dev]) return;
  for (const void* fn : {(const void*)mw::triattn_mw_fwd<false, false>, (const void*)mw::triattn_mw_fwd<false, true>, (const void*)mw::triattn_mw_list<false>, (const void*)mw::triattn_mw_list<true>})
    TX_CUDA(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  TX_CUDA(cudaDeviceGetAttribute(&g_num_sms[dev], cudaDevAttrMultiProcessorCount, dev));
  g_configured[dev] = true;
}

void run(TriattnCudaCall* c) {
#ifdef TXLA_LSE
  TX_CHECK(c->abi_version == TRIATTN_CUDA_ABI_LSE_VERSION, "ABI version mismatch between launcher and CUDA library (the lse library takes revision 2)");
  TX_CHECK(c->lse != nullptr, "lse output pointer is null");
#else
  TX_CHECK(c->abi_version == TRIATTN_CUDA_ABI_VERSION, "ABI version mismatch between launcher and CUDA library");
#endif
  const long long B = c->B, N = c->N, H = c->H, S = c->S;
  TX_CHECK(c->D == mw::D, "D must be 32");
  TX_CHECK(S >= 1 && S <= mw::MAX_S, "S out of range (1 <= S <= 65536)");
  TX_CHECK(B >= 1 && N >= 1 && H >= 1, "empty operand");
  TX_CHECK(B * H <= 65535 && (N + mw::R - 1) / mw::R <= 65535, "grid limits: B*H <= 65535, N <= 196605");
  TX_CHECK(c->bias_B == B, "bias batch must equal B");
  TX_CHECK(c->layout == 0 || c->layout == 1, "layout code");
  for (const void* p : {c->q, c->k, c->v, c->bias, (const void*)c->out, (const void*)c->bias_staged, (const void*)c->fix})
    TX_CHECK(p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0, "operand pointers must be non-null and 16-byte aligned");
  int dev = 0; TX_CUDA(cudaGetDevice(&dev));
  cudaStream_t st = reinterpret_cast<cudaStream_t>(c->stream);
  const Strides sq = strides_of(c->layout, N, H, S, mw::D);
  TX_CHECK(sq.sS % 8 == 0 && sq.sH % 8 == 0 && sq.sN % 8 == 0 && sq.sB % 8 == 0, "q/k/v strides: multiples of 8 elements");
  const int S64 = (int)((S + 63) / 64 * 64), S128 = (int)((S + 127) / 128 * 128);
  const int QT = (int)((S + mw::BM - 1) / mw::BM), YG = (int)((N + mw::R - 1) / mw::R), nkt = (int)((S + mw::BN - 1) / mw::BN);
  const long long nct = (long long)QT * YG * B * H;
  TX_CHECK(c->fix_elems >= 3 + 3 * nct, "fix buffer smaller than 3 + 3 * #CTA tiles");

  // ---- 1. mask tables (stage_mask): keyany [B,S64], rowkind [B,N], irr [1+B*YG], rgflag [B*YG], ktend [B], mtab [B,N,nkt,2], ctab [B,N,nkt], ktendr [B,N]
  const uint8_t* mask = reinterpret_cast<const uint8_t*>(c->mask);
  if (mask) {
    for (void* p : {c->keyany, c->rowkind, c->irr, c->rgflag, c->ktend, c->mtab, c->ctab, c->ktendr}) TX_CHECK(p != nullptr, "mask scratch buffers");
    TX_CUDA(cudaMemsetAsync(c->irr, 0, sizeof(int) * (size_t)(1 + B * YG), st));
    TX_CUDA(cudaMemsetAsync(c->rgflag, 0, sizeof(int) * (size_t)(B * YG), st));
    TX_CUDA(cudaMemsetAsync(c->ktend, 0, sizeof(int) * (size_t)B, st));
    TX_CUDA(cudaMemsetAsync(c->ktendr, 0, sizeof(int) * (size_t)(B * N), st));
    const long long m_sB = N * S, m_sN = S;
    mw::mask_or_kernel<<<dim3((S64 + 127) / 128, (unsigned)B), 1024, 0, st>>>(mask, m_sB, m_sN, (uint8_t*)c->keyany, (int*)c->ktend, (int)N, (int)S, S64);
    TX_LAUNCH_CHECK("mask_or_kernel");
    const long long rows = B * N; const int wpb = 8;
    mw::mask_rows_kernel<<<(unsigned)((rows + wpb - 1) / wpb), wpb * 32, 0, st>>>(mask, m_sB, m_sN, (const uint8_t*)c->keyany,
        (uint8_t*)c->rowkind, (int*)c->rgflag, (int*)c->irr, (int)B, (int)N, (int)S, S64);
    TX_LAUNCH_CHECK("mask_rows_kernel");
    { const long long nw = B * N * nkt; const int wpb2 = 8;
      mw::mask_tables_kernel<<<(unsigned)((nw + wpb2 - 1) / wpb2), wpb2 * 32, 0, st>>>(mask, m_sB, m_sN,
          (uint32_t*)c->mtab, (uint8_t*)c->ctab, (int*)c->ktendr, (int)B, (int)N, (int)S, nkt);
      TX_LAUNCH_CHECK("mask_tables_kernel"); }
  }

  // ---- 2. stage the bias: [B,H,S,S] (f32 | bf16 | f16, dense) -> f32 [B,H,S128,S64] / scale in C-fragment order, -inf on keys no row attends
  const float scale = c->scale > 0.f ? c->scale : (float)(1.0 / std::sqrt((double)mw::D));
  {
    const long long sB = H * S * S, sH = S * S, sQ = S, sK = 1;
    const long long n = B * H * (long long)S128 * S64;
    const int threads = 256; const int blocks = (int)std::min<long long>((n + threads - 1) / threads, 132 * 32);
    const float inv_scale = (float)(1.0 / scale);
    const uint8_t* ka = mask ? (const uint8_t*)c->keyany : nullptr;
    float* outb = (float*)c->bias_staged;
    if (c->bias_dtype == 0)      mw::stage_bias_kernel<float><<<blocks, threads, 0, st>>>((const float*)c->bias, sB, sH, sQ, sK, ka, outb, (int)B, (int)H, (int)S, S128, S64, inv_scale);
    else if (c->bias_dtype == 1) mw::stage_bias_kernel<__nv_bfloat16><<<blocks, threads, 0, st>>>((const __nv_bfloat16*)c->bias, sB, sH, sQ, sK, ka, outb, (int)B, (int)H, (int)S, S128, S64, inv_scale);
    else if (c->bias_dtype == 2) mw::stage_bias_kernel<__half><<<blocks, threads, 0, st>>>((const __half*)c->bias, sB, sH, sQ, sK, ka, outb, (int)B, (int)H, (int)S, S128, S64, inv_scale);
    else throw Fail("bias dtype code");
    TX_LAUNCH_CHECK("stage_bias_kernel");
  }

  // ---- 3. the forward (fwd): main mask-free pass, then the irregular-row list pass (mask only), then the fix pass
  mw::Params p; std::memset(&p, 0, sizeof(p));
#ifdef TXLA_LSE
  p.lse = reinterpret_cast<float*>(c->lse);                     // the patched device section stores each query row's log2-sum-exp here (every pass)
#endif
  p.tmK = map_qkv(c->k, sq, B, N, H, S, mw::BN); p.tmV = map_qkv(c->v, sq, B, N, H, S, mw::BN);
  p.q = reinterpret_cast<const __nv_bfloat16*>(c->q); p.q_sB = sq.sB; p.q_sN = sq.sN; p.q_sH = sq.sH; p.q_sS = sq.sS;
  {
    const uint64_t dims[4] = {(uint64_t)S64, (uint64_t)S128, (uint64_t)H, (uint64_t)B};
    const uint64_t str[3] = {(uint64_t)S64 * 4, (uint64_t)S128 * S64 * 4, (uint64_t)H * S128 * S64 * 4};
    const uint32_t box[4] = {32, (uint32_t)mw::BM, 1, 1};
    p.tmB = mw::make_map(CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, c->bias_staged, dims, str, box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  p.out = reinterpret_cast<__nv_bfloat16*>(c->out); p.o_sB = sq.sB; p.o_sN = sq.sN; p.o_sH = sq.sH; p.o_sS = sq.sS;
  p.mask = nullptr; p.m_sB = p.m_sN = 0; p.rowkind = nullptr; p.list = nullptr; p.list_mul = 1; p.ktend = nullptr; p.mtab = nullptr; p.ctab = nullptr; p.ktendr = nullptr;
  if (mask) {
    p.ktend = (const int*)c->ktend;
    p.mask = mask; p.m_sB = N * S; p.m_sN = S;
    p.rowkind = (const uint8_t*)c->rowkind;
    p.mtab = (const uint32_t*)c->mtab; p.ctab = (const uint8_t*)c->ctab; p.ktendr = (const int*)c->ktendr;
  }
  p.B = (int)B; p.N = (int)N; p.H = (int)H; p.S = (int)S; p.n_ktiles = nkt;
  p.c1 = (float)(scale * 1.4426950408889634);
  {
    const long long per_qtile = (long long)mw::BM * S64 * 4;
    p.qb = (int)std::max<long long>(1, std::min<long long>(QT, (16ll << 20) / per_qtile));
    if (c->dbg >> 12) p.qb = (int)std::min<long long>(QT, c->dbg >> 12);
  }
  p.dbg = c->dbg & ~64;           // no timeline builds here
  p.tl = nullptr;
  const int smem = (int)sizeof(mw::Smem) + 1024;
  configure_device(dev, smem);
  dim3 grid((unsigned)QT, (unsigned)YG, (unsigned)(B * H)), block(mw::THREADS);
  p.fix = (int*)c->fix;
  TX_CUDA(cudaMemsetAsync(p.fix, 0, 3 * sizeof(int), st));      // census words + this call's fix count (a per-call buffer here)
  const int npers = (int)std::min<long long>(nct, (long long)g_num_sms[dev]);
  if (p.dbg & 512) mw::triattn_mw_fwd<false, true><<<grid, block, smem, st>>>(p);
  else mw::triattn_mw_fwd<false, false><<<grid, block, smem, st>>>(p);
  TX_LAUNCH_CHECK("triattn_mw_fwd");
  if (p.mask) {
    mw::Params pl = p; pl.list = (const int*)c->irr; pl.list_mul = QT * (int)H;
    mw::triattn_mw_list<false><<<npers, block, smem, st>>>(pl);
    TX_LAUNCH_CHECK("triattn_mw_list<general>");
  }
  {
    mw::Params pf = p; pf.list = p.fix + 2; pf.list_mul = 1;
    mw::triattn_mw_list<true><<<npers, block, smem, st>>>(pf);
    TX_LAUNCH_CHECK("triattn_mw_list<fix>");
  }
}

}  // namespace

#define TXLA_EXPORT __attribute__((visibility("default")))

extern "C" {

#ifdef TXLA_LSE
TXLA_EXPORT int triattn_mw_cuda_fwd_lse(TriattnCudaCall* c) {
#else
TXLA_EXPORT int triattn_mw_cuda_fwd(TriattnCudaCall* c) {
#endif
  if (c == nullptr) return 3;
  c->err[0] = 0;
  try { run(c); return 0; }
  catch (const std::exception& e) { std::snprintf(c->err, sizeof(c->err), "%s", e.what()); return 1; }
  catch (...) { std::snprintf(c->err, sizeof(c->err), "unknown C++ exception"); return 2; }
}

TXLA_EXPORT long long triattn_mw_cuda_fix_elems(long long B, long long N, long long H, long long S) {
  return 3 + 3 * ((S + mw::BM - 1) / mw::BM) * ((N + mw::R - 1) / mw::R) * B * H;
}

TXLA_EXPORT long long triattn_mw_cuda_smem_bytes(void) { return (long long)sizeof(mw::Smem) + 1024; }

TXLA_EXPORT const char* triattn_mw_cuda_describe(void) {
  static char buf[160];
#ifdef TXLA_LSE
  std::snprintf(buf, sizeof(buf), "triattn_mw sm_90a D=%d BM=%d BN=%d R=%d ST=%d THREADS=%d smem=%lld abi=%d +lse", mw::D, mw::BM, mw::BN, mw::R, mw::ST, mw::THREADS,
                (long long)sizeof(mw::Smem) + 1024, TRIATTN_CUDA_ABI_LSE_VERSION);
#else
  std::snprintf(buf, sizeof(buf), "triattn_mw sm_90a D=%d BM=%d BN=%d R=%d ST=%d THREADS=%d smem=%lld abi=%d", mw::D, mw::BM, mw::BN, mw::R, mw::ST, mw::THREADS,
                (long long)sizeof(mw::Smem) + 1024, TRIATTN_CUDA_ABI_VERSION);
#endif
  return buf;
}

}  // extern "C"
