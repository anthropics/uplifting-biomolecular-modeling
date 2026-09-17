"""[esmfold2_trimul v6.1 — this copy carries the k3v6 RELEASE-ORDERING FIX (kernels_pkg/esmfold2_trimul/v6.1/src/PATCHES/k3v6_release_ordering_fix.diff):
 the empty_stage and empty_res mbarrier releases are issued only after every ldmatrix instruction that read the slot has returned its data
 (XOR of one destination register per ldmatrix -> __reduce_or_sync -> & a run-time zero -> the arrive address) and after fence.proxy.async;
 K3 arithmetic unchanged.  Everything else is the k-trimul module @ 74556e72 verbatim.]

ef2_trimul_v6 — TriangleMultiplication consumer kernel (K3) in CUDA C++ / CuTe for sm_90a, built at first use with NVRTC.  Track k-trimul.

What it computes (per TriMul call, after K1 and the batched GEMM of ef2_trimul_v5 / fpf_trimul_v4 have produced the channel-major planes
x[D, Np, Np]):   out[i, j, :] = z[i, j, :] + sigmoid(LN_in(z[i, j]) Wg^T + cg) * (LN_out(x[:, i, j]) Wo^T + cp)      (bf16 out, fp32 math)
with the LayerNorm affine transforms folded into the projection weights on the host (lever `lnfold`, the only numerics difference vs t9's K3:
W' = bf16(W * gamma), c = W @ beta in fp32; t9 applies gamma/beta in fp32 and rounds the LN output to bf16 before the GEMM — both are one bf16
rounding away from the fp32 module; error tables in the k_trimul report).  With lnfold off (build(lnfold=False)) the kernel applies gamma/beta
in registers exactly at t9's rounding points (outputs within 1 bf16 ulp of t9) at ~t9 speed.  The fold runs with autocast DISABLED (the model
forward is under bf16 autocast; the fp32 bias must stay fp32 — the kernel reads `const float*`).

Kernel structure (one persistent CTA per SM, 384 threads, ~221 KB dynamic smem):
  producer warpgroup   lane 0 of warp 0: TMA loads of the x^T tile [256 ch x 64 tok] x 2 and the z tile [128 tok x 256] (SW128, 3-D tensor maps
                       so partial tiles at a pair-row end are hardware-clipped), then the Wo/Wg [32 x 256] weight blocks through a 4-slot mbarrier
                       ring; lane 0 of warp 1: the residual z[i, j-tile, 32-channel block] tiles through a 2-slot ring per consumer warpgroup.
  consumer warpgroups  (2 x 128 threads, 64 tokens each, 232 registers/thread via setmaxnreg): ldmatrix(.trans) the staged operands straight into
                       wgmma A-fragments, LayerNorm statistics by quad shuffles and normalisation in registers (the LN'd operands never touch shared
                       memory), per 32-channel out-block two async wgmma m64n32k16 (RS: A registers, B = weight slot descriptor) into one of two
                       accumulator sets while the previous block's epilogue runs; epilogue = residual by ldmatrix from the residual ring (the m64k32
                       A-fragment thread distribution equals the m64n32 accumulator distribution), gate * proj + residual in fp32, one bf16 rounding,
                       stmatrix into a swizzled staging slice, one TMA store per warp; one warpgroup named barrier per out-block (EPIBAR) keeps
                       the four warps convergent around the warpgroup-collective wgmma/wait instructions.
Build: headers from the pip wheels pinned in environment/requirements.lock (nvidia-cutlass -> cutlass_library/source/include; nvidia-cuda-cccl;
the CUDA headers of the torch wheel set under nvidia/cu13/include); NVRTC from cuda.bindings; cubin cached under the kit JIT root keyed by
sha256(source, options, nvrtc version); loaded with the CUDA driver API and launched on torch's current stream (graph-capturable: descriptors are
kernel parameters by value, all buffers come from torch's allocator).  sm_90a only: anything else raises by name.  Nothing falls back silently.
"""
import os, sys, ctypes, hashlib, tempfile, time

import torch

__all__ = ["build", "launch_k3", "fold_weights", "describe", "V6Unavailable"]


class V6Unavailable(RuntimeError):
    """The CuTe K3 kernel cannot be built or run on this stack (named reason)."""


# =====================================================================================================================
# kernel source (CUDA C++ / CuTe).  Compile-time switches kept from development: FASTSIG (tanh.approx gate; follows the v5 `sigmoid` lever),
# LNFOLD (1 = production), NSLOT/NRES/NOUT/NACC ring depths, PROF/EPI/DO_MMA/DO_LN ablations (never set in the kit).
# =====================================================================================================================
K3_SRC = r'''
#include <cute/tensor.hpp>
using namespace cute;

struct __align__(64) TmaDesc { uint64_t w[16]; };

// ------------------------------------------------------------------------------------------------ raw PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void tma_load_3d(const TmaDesc* tm, uint64_t* bar, void* dst, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tm)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void tma_load_2d(const TmaDesc* tm, uint64_t* bar, void* dst, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tm)), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma_store_3d(const TmaDesc* tm, const void* src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
               :: "l"(reinterpret_cast<uint64_t>(tm)), "r"(smem_u32(src)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void tma_store_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr) : "memory");
}
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__ uint32_t sw64(uint32_t off) { return off ^ (((off >> 7) & 3u) << 4); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory"); }
__device__ __forceinline__ uint32_t zero_dep(uint32_t dep) {      // == 0 at run time for any dep; opaque to ptxas (dynamic smem size is a launch parameter)
  uint32_t ds; asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(ds)); return dep & (ds >> 20); }
__device__ __forceinline__ void mbar_arrive_dep(uint64_t* bar, uint32_t dep_zero) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar) + dep_zero) : "memory"); }
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) { while (!mbar_try_wait(bar, phase)) {} }
__device__ __forceinline__ void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void named_bar_sync(int id, int nthreads) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(nthreads) : "memory"); }
template <int N> __device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N)); }
template <int N> __device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N)); }
#ifndef SIGMODE
#define SIGMODE 1      // 1: ex2.approx + rcp.approx (Triton tl.sigmoid class)  3: identity (profiling only)
#endif
__device__ __forceinline__ float ex2_approx(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float rcp_approx(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float sigmoid_exact(float x) {
#if SIGMODE == 1
  return rcp_approx(1.f + ex2_approx(-1.4426950408889634f * x));
#else
  return x;
#endif
}
__device__ __forceinline__ float sigmoid_tanh(float x) { float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x)); return 0.5f * t + 0.5f; }
__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) { __nv_bfloat162 v = __floats2bfloat162_rn(a, b); return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ float2 unpack_bf16x2(uint32_t u) { __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u); return __bfloat1622float2(v); }

// ------------------------------------------------------------------------------------------------ configuration
#ifndef NSLOT
#define NSLOT 4
#endif
#ifndef NRES
#define NRES 2         // residual ring slots per consumer WG
#endif
#ifndef NOUT
#define NOUT 1         // output staging tiles per consumer WG
#endif
#ifndef EPIBAR
#define EPIBAR 1       // 1: warpgroup-wide named barrier before the weight-slot release and before each TMA store (keeps the 4 warps of a consumer
#endif                 //    warpgroup convergent around the warpgroup-collective wgmma instructions); 0 = per-warp epilogue (dev experiment)
#ifndef NACC
#define NACC 1         // accumulator sets: 2 = block b+1's wgmma overlaps block b's epilogue (costs 32 registers), 1 = sequential
#endif
#ifndef PROD_REGS
#define PROD_REGS 40
#endif
#ifndef CONS_REGS
#define CONS_REGS 232
#endif
#ifndef FASTSIG
#define FASTSIG 0
#endif
#ifndef LNFOLD
#define LNFOLD 1
#endif
#ifndef EPI
#define EPI 1
#endif
#ifndef DO_MMA
#define DO_MMA 1
#endif
#ifndef DO_LN
#define DO_LN 1
#endif
#ifndef PROF
#define PROF 0
#endif
#if PROF
#define PCLK(v) long long v = clock64()
#define PACC(slot, t0, t1) do { if (prof_on) prof_acc[slot] += (unsigned long long)((t1) - (t0)); } while (0)
#else
#define PCLK(v) long long v = 0
#define PACC(slot, t0, t1) do {} while (0)
#endif

constexpr int BT   = 128;   // tokens per CTA tile (2 consumer WGs x 64)
constexpr int CH   = 256;   // channels of x (K of the out-projection)
constexpr int CZ   = 256;   // pair channels (K of the gate projection; N of both projections)
constexpr int BNO  = 32;    // out-channel block
constexpr int NB   = CZ / BNO;
constexpr int SLOT_BYTES = BNO * 256 * 2;        // weight block [32 x 256] bf16 = 16 KB
constexpr int XT_BYTES   = CH * 64 * 2;          // one WG's x^T tile [256 ch x 64 tok] = 32 KB
constexpr int Z_BYTES    = BT * CZ * 2;          // z tile [128 tok x 256 ch] = 64 KB
constexpr int RB_BYTES   = 64 * BNO * 2;         // residual / output block [64 tok x 32 ch] bf16 = 4 KB
constexpr int OFF_XT   = 0;
constexpr int OFF_Z    = OFF_XT + 2 * XT_BYTES;
constexpr int OFF_RING = OFF_Z + Z_BYTES;
constexpr int OFF_RES  = OFF_RING + NSLOT * SLOT_BYTES;          // [cw][NRES] residual tiles
constexpr int OFF_OUT  = OFF_RES + 2 * NRES * RB_BYTES;          // [cw][NOUT] output staging tiles
constexpr int OFF_LN   = OFF_OUT + 2 * NOUT * RB_BYTES;          // 4 x 256 fp32 (LN vectors, or bias_p | bias_g when LNFOLD)
constexpr int OFF_BAR  = OFF_LN + 4 * 256 * 4;
// barriers: full_stage, empty_stage, full_w[NSLOT], empty_w[NSLOT], full_res[2][NRES], empty_res[2][NRES]
constexpr int NBARS    = 2 + 2 * NSLOT + 4 * NRES;
constexpr int SMEM_BYTES = OFF_BAR + NBARS * 8 + 1024;
static_assert(SMEM_BYTES <= 232448, "smem budget (227 KB) exceeded");

using MmaRS  = decltype(make_tiled_mma(SM90_64x32x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>{}));
using SwK128 = GMMA::Layout_K_SW128_Atom<bfloat16_t>;
using SwMN128 = GMMA::Layout_MN_SW128_Atom<bfloat16_t>;
using SwK64  = GMMA::Layout_K_SW64_Atom<bfloat16_t>;
using SLayZ  = decltype(tile_to_shape(SwK128{},  Shape<Int<BT>, Int<CZ>>{}));   // (128 tok, 256 ch): 4 slabs [128 x 64]
using SLayXT = decltype(tile_to_shape(SwMN128{}, Shape<_64, Int<CH>>{}));       // (64 tok, 256 ch) tok-contiguous, per WG
using SLayW  = decltype(tile_to_shape(SwK128{},  Shape<Int<BNO>, _256>{}));     // (32 out, 256 in): 4 slabs [32 x 64]
using SLayRB = decltype(tile_to_shape(SwK64{},   Shape<_64, Int<BNO>>{}));      // (64 tok, 32 ch) ch-contiguous 64-B rows, SW64

extern "C" __global__ void __launch_bounds__(384, 1)
k3v6(const __grid_constant__ TmaDesc tm_x, const __grid_constant__ TmaDesc tm_z, const __grid_constant__ TmaDesc tm_wo, const __grid_constant__ TmaDesc tm_wg,
     const __grid_constant__ TmaDesc tm_res, const __grid_constant__ TmaDesc tm_out,
     const float* __restrict__ ln_out_w, const float* __restrict__ ln_out_b, const float* __restrict__ ln_in_w, const float* __restrict__ ln_in_b,
     const float* __restrict__ bias_p, const float* __restrict__ bias_g,
     int N, int Np, int nJ, int n_tiles, float eps, unsigned long long* __restrict__ prof)
{
  extern __shared__ __align__(16) unsigned char smem_raw[];
  unsigned char* smem = reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + OFF_BAR);
  uint64_t* full_stage = bars + 0; uint64_t* empty_stage = bars + 1; uint64_t* full_w = bars + 2; uint64_t* empty_w = bars + 2 + NSLOT;
  uint64_t* full_res = bars + 2 + 2 * NSLOT;            // [cw * NRES + slot]
  uint64_t* empty_res = full_res + 2 * NRES;
  float* s_ln = reinterpret_cast<float*>(smem + OFF_LN);
  const int tid = threadIdx.x;
  const int wg = __shfl_sync(0xffffffff, tid / 128, 0);
#if LNFOLD
  for (int q = tid; q < 256; q += blockDim.x) { s_ln[q] = bias_p[q]; s_ln[256 + q] = bias_g[q]; }
#else
  for (int q = tid; q < 256; q += blockDim.x) { s_ln[q] = ln_out_w[q]; s_ln[256 + q] = ln_out_b[q]; s_ln[512 + q] = ln_in_w[q]; s_ln[768 + q] = ln_in_b[q]; }
#endif
  if (tid == 0) {
    mbar_init(full_stage, 1); mbar_init(empty_stage, 8);                                        // 8 consumer warps release the staging tiles
    for (int s = 0; s < NSLOT; ++s) { mbar_init(full_w + s, 1); mbar_init(empty_w + s, 2); }    // 1 elected thread per consumer WG (after wgmma.wait_group)
    for (int s = 0; s < 2 * NRES; ++s) { mbar_init(full_res + s, 1); mbar_init(empty_res + s, 4); }   // 4 warps of the WG
    fence_barrier_init();
  }
  __syncthreads();

  if (wg == 0) {
    // =========================================================================== producer
    setmaxnreg_dec<PROD_REGS>();
#ifdef PRODSTUB
    if (false) {
#else
    if (tid == 0) {                                                  // warp 0: operand staging + weight ring
#endif
      uint32_t seq = 0, it = 0;
      for (int t = blockIdx.x; t < n_tiles; t += gridDim.x, ++it) {
        const int i = t / nJ; const int j0 = (t % nJ) * BT;
        if (it > 0) mbar_wait(empty_stage, (it - 1) & 1);
        mbar_expect_tx(full_stage, 2 * XT_BYTES + Z_BYTES);
        tma_load_3d(&tm_x, full_stage, smem + OFF_XT, j0, i, 0);                       // x^T: dims {Np j, Np i, CH}; box {64, 1, 256}
        tma_load_3d(&tm_x, full_stage, smem + OFF_XT + XT_BYTES, j0 + 64, i, 0);
        for (int s4 = 0; s4 < 4; ++s4) tma_load_3d(&tm_z, full_stage, smem + OFF_Z + s4 * (BT * 64 * 2), s4 * 64, j0, i);   // z: {CZ, N j, N i}; box {64, 128, 1}
        for (int b = 0; b < NB; ++b) {
          for (int m = 0; m < 2; ++m, ++seq) {                       // Wo_b, Wg_b: [32 rows, 256 k] as 4 boxes {64 k, 32 rows}
            const int slot = seq % NSLOT; const uint32_t use = seq / NSLOT;
            if (use > 0) mbar_wait(empty_w + slot, (use - 1) & 1);
            mbar_expect_tx(full_w + slot, SLOT_BYTES);
            const TmaDesc* tm = (m == 0) ? &tm_wo : &tm_wg;
            unsigned char* dst = smem + OFF_RING + slot * SLOT_BYTES;
            for (int s4 = 0; s4 < 4; ++s4) tma_load_2d(tm, full_w + slot, dst + s4 * (BNO * 64 * 2), s4 * 64, b * BNO);
          }
        }
      }
    }
#if EPI
    else if (tid == 32) {                                            // warp 1: residual tiles z[i, j0+64cw : +64, 32b : +32]
      uint32_t rseq = 0;
      for (int t = blockIdx.x; t < n_tiles; t += gridDim.x) {
        const int i = t / nJ; const int j0 = (t % nJ) * BT;
        for (int b = 0; b < NB; ++b, ++rseq) {
          const int rs = rseq % NRES; const uint32_t use = rseq / NRES;
          for (int cw = 0; cw < 2; ++cw) {
            uint64_t* fb = full_res + cw * NRES + rs; uint64_t* eb = empty_res + cw * NRES + rs;
            if (use > 0) mbar_wait(eb, (use - 1) & 1);
            mbar_expect_tx(fb, RB_BYTES);
            tma_load_3d(&tm_res, fb, smem + OFF_RES + (cw * NRES + rs) * RB_BYTES, b * BNO, j0 + 64 * cw, i);
          }
        }
      }
    }
#endif
  } else {
    // =========================================================================== consumers
    setmaxnreg_inc<CONS_REGS>();
    const int cw = wg - 1;
    const int ltid = tid - 128 * wg;
    unsigned long long prof_acc[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const bool prof_on = (PROF != 0) && (prof != nullptr) && (ltid == 0);
    MmaRS mma;
    auto thr = mma.get_thread_slice(ltid);
    Tensor sXT = make_tensor(make_smem_ptr(reinterpret_cast<bfloat16_t*>(smem + OFF_XT + cw * XT_BYTES)), SLayXT{});
    Tensor sZfull = make_tensor(make_smem_ptr(reinterpret_cast<bfloat16_t*>(smem + OFF_Z)), SLayZ{});
    Tensor sZ = local_tile(sZfull, Shape<_64, Int<CZ>>{}, make_coord(cw, 0));
    Tensor rX = thr.partition_fragment_A(sZ);
    Tensor rZ = thr.partition_fragment_A(sZ);
    auto cpX = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, mma);
    auto cpZ = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>{}, mma);
    auto thrX = cpX.get_thread_slice(ltid); auto thrZ = cpZ.get_thread_slice(ltid);
    Tensor tXs = thrX.partition_S(sXT); Tensor tXr = thrX.retile_D(rX);
    Tensor tZs = thrZ.partition_S(sZ);  Tensor tZr = thrZ.retile_D(rZ);
    // residual / output [64 tok x 32 ch] bf16 tiles (64-B rows, SW64): per-lane ldmatrix/stmatrix row addresses, precomputed.
    //   x4 matrices: m0 rows 0-7 cols 0-7 | m1 rows 8-15 cols 0-7 | m2 rows 0-7 cols 8-15 | m3 rows 8-15 cols 8-15 of the warp's 16x16 (kb) region
    //   -> registers q = 0..3 hold C-fragment elements e = 8 kb + 2 q (+0/+1): (row r0 + 8 (q&1), cols nq + 8 (q>>1) + 16 kb)
    uint32_t res_addr0, res_addr1, out_addr0, out_addr1;      // slot 0 addresses for kb = 0 / 1; slot s = + s * RB_BYTES (4 KB-aligned tiles: same swizzle phase)
    {
      const int lane = ltid & 31, wq = ltid >> 5, mi = lane >> 3, rr = lane & 7;
      const int trow = 16 * wq + rr + 8 * (mi & 1);
      const uint32_t off0 = sw64((uint32_t)(trow * 64 + (8 * (mi >> 1)) * 2)), off1 = sw64((uint32_t)(trow * 64 + (16 + 8 * (mi >> 1)) * 2));
      res_addr0 = smem_u32(smem + OFF_RES + (cw * NRES) * RB_BYTES) + off0; res_addr1 = smem_u32(smem + OFF_RES + (cw * NRES) * RB_BYTES) + off1;
      out_addr0 = smem_u32(smem + OFF_OUT + (cw * NOUT) * RB_BYTES) + off0; out_addr1 = smem_u32(smem + OFF_OUT + (cw * NOUT) * RB_BYTES) + off1;
    }
    const int lane = ltid & 31, wq = ltid >> 5;
    const int r0 = (lane >> 2) + 16 * wq;
    const int nq = 2 * (lane & 3);
    auto sW = [&](int slot) { return make_tensor(make_smem_ptr(reinterpret_cast<bfloat16_t*>(smem + OFF_RING + slot * SLOT_BYTES)), SLayW{}); };
    auto fragB = [&](int slot) { return thr.make_fragment_B(thr.partition_B(sW(slot))); };
    uint32_t seq = 0, rseq = 0, it = 0, oseq = 0;
    for (int t = blockIdx.x; t < n_tiles; t += gridDim.x, ++it) {
      const int i = t / nJ; const int j0 = (t % nJ) * BT;
      const int tok0 = j0 + 64 * cw;
      PCLK(c0);
      mbar_wait(full_stage, it & 1);
      PCLK(c1);
      copy(cpX, tXs, tXr);
      copy(cpZ, tZs, tZr);
      {   // release the x^T / z stage only once every ldmatrix instruction of both operands has returned its data (one register per x4 atom)
        uint32_t dep = 0;
        { Tensor rXu = recast<uint32_t>(rX); CUTE_UNROLL for (int e = 0; e < size(rXu); e += 4) dep ^= rXu(e); }
        { Tensor rZu = recast<uint32_t>(rZ); CUTE_UNROLL for (int e = 0; e < size(rZu); e += 4) dep ^= rZu(e); }
        dep = zero_dep(__reduce_or_sync(0xffffffffu, dep));
        fence_async_smem();
        if (lane == 0) mbar_arrive_dep(empty_stage, dep);
      }
      PCLK(c2); PACC(0, c0, c1); PACC(1, c1, c2);
#if DO_LN
      {
        const int kq = nq;
        auto layer_norm = [&](auto& rA, const float* lw, const float* lb, const float inv_c) {
          float s0 = 0.f, s1 = 0.f;
          CUTE_UNROLL
          for (int e = 0; e < size(rA); ++e) { const float v = float(rA(e)); if (((e >> 1) & 1) == 0) s0 += v; else s1 += v; }
          s0 += __shfl_xor_sync(0xffffffff, s0, 1); s0 += __shfl_xor_sync(0xffffffff, s0, 2);
          s1 += __shfl_xor_sync(0xffffffff, s1, 1); s1 += __shfl_xor_sync(0xffffffff, s1, 2);
          const float mu0 = s0 * inv_c, mu1 = s1 * inv_c;
          float q0 = 0.f, q1 = 0.f;
          CUTE_UNROLL
          for (int e = 0; e < size(rA); ++e) { const float v = float(rA(e)); if (((e >> 1) & 1) == 0) { const float d = v - mu0; q0 += d * d; } else { const float d = v - mu1; q1 += d * d; } }
          q0 += __shfl_xor_sync(0xffffffff, q0, 1); q0 += __shfl_xor_sync(0xffffffff, q0, 2);
          q1 += __shfl_xor_sync(0xffffffff, q1, 1); q1 += __shfl_xor_sync(0xffffffff, q1, 2);
          const float is0 = rsqrtf(q0 * inv_c + eps), is1 = rsqrtf(q1 * inv_c + eps);
#if LNFOLD
          const float a0 = -mu0 * is0, a1 = -mu1 * is1;
          CUTE_UNROLL
          for (int e = 0; e < size(rA); e += 2) {
            const bool r1s = ((e >> 1) & 1) != 0;
            const float is = r1s ? is1 : is0, a = r1s ? a1 : a0;
            const uint32_t packed = pack_bf16x2(fmaf(float(rA(e)), is, a), fmaf(float(rA(e + 1)), is, a));
            rA(e) = reinterpret_cast<const bfloat16_t*>(&packed)[0]; rA(e + 1) = reinterpret_cast<const bfloat16_t*>(&packed)[1];
          }
#else
          CUTE_UNROLL
          for (int kb = 0; kb < 16; ++kb) {
            CUTE_UNROLL
            for (int hi = 0; hi < 2; ++hi) {
              const int k = 16 * kb + kq + 8 * hi;
              const float2 w2 = *reinterpret_cast<const float2*>(lw + k);
              const float2 b2 = *reinterpret_cast<const float2*>(lb + k);
              CUTE_UNROLL
              for (int rs = 0; rs < 2; ++rs) {
                const int e = 8 * kb + 4 * hi + 2 * rs;
                const float mu = rs ? mu1 : mu0, is = rs ? is1 : is0;
                const uint32_t packed = pack_bf16x2((float(rA(e)) - mu) * is * w2.x + b2.x, (float(rA(e + 1)) - mu) * is * w2.y + b2.y);
                rA(e) = reinterpret_cast<const bfloat16_t*>(&packed)[0]; rA(e + 1) = reinterpret_cast<const bfloat16_t*>(&packed)[1];
              }
            }
          }
#endif
        };
        layer_norm(rX, s_ln, s_ln + 256, 1.f / CH);
        layer_norm(rZ, s_ln + 512, s_ln + 768, 1.f / CZ);
      }
#endif
      PCLK(c3); PACC(2, c2, c3); if (prof_on) prof_acc[7] += 1;
      Tensor accP0 = partition_fragment_C(mma, Shape<_64, Int<BNO>>{}); Tensor accG0 = partition_fragment_C(mma, Shape<_64, Int<BNO>>{});
#if NACC == 2
      Tensor accP1 = partition_fragment_C(mma, Shape<_64, Int<BNO>>{}); Tensor accG1 = partition_fragment_C(mma, Shape<_64, Int<BNO>>{});
#endif
      auto issue = [&](auto& accP, auto& accG) {
        const int slotP = seq % NSLOT; const uint32_t useP = seq / NSLOT; ++seq;
        const int slotG = seq % NSLOT; const uint32_t useG = seq / NSLOT; ++seq;
        clear(accP); clear(accG);
        Tensor rBP = fragB(slotP); Tensor rBG = fragB(slotG);
        PCLK(w0);
        mbar_wait(full_w + slotP, useP & 1);
        PCLK(w1);
        warpgroup_fence_operand(rX); warpgroup_fence_operand(accP);
        warpgroup_arrive();
#if DO_MMA
        gemm(mma, rX, rBP, accP);
#endif
        warpgroup_commit_batch();
        PCLK(w2);
        mbar_wait(full_w + slotG, useG & 1);
        PCLK(w3);
        warpgroup_fence_operand(rZ); warpgroup_fence_operand(accG);
        warpgroup_arrive();
#if DO_MMA
        gemm(mma, rZ, rBG, accG);
#endif
        warpgroup_commit_batch();
        PCLK(w4); PACC(3, w0, w1); PACC(3, w2, w3); PACC(4, w1, w2); PACC(4, w3, w4);
      };
      auto epilogue = [&](int b, auto& accP, auto& accG, int slotP, int slotG) {
        // called right after wgmma.wait_group for block b (warpgroup-collective): the weight slots are free
        warpgroup_fence_operand(accP); warpgroup_fence_operand(accG);
#if EPIBAR
        named_bar_sync(1 + cw, 128);                       // all 4 warps of this WG are past wgmma.wait_group for block b
#endif
        if (ltid == 0) { mbar_arrive(empty_w + slotP); mbar_arrive(empty_w + slotG); }
        PCLK(eb0);
#if EPI
        const int os = oseq % NOUT;
        // my warp's previous TMA store out of staging slice `os` must have finished reading smem before the slice is overwritten
        if (lane == 0 && oseq >= NOUT) tma_store_wait_read<NOUT - 1>();
        // residual fragment (ldmatrix from the residual ring), per-warp release
        const int rs = rseq % NRES; const uint32_t ruse = rseq / NRES; ++rseq;
        mbar_wait(full_res + cw * NRES + rs, ruse & 1);
        uint32_t fr[2][4];
        ldsm_x4(res_addr0 + rs * RB_BYTES, fr[0][0], fr[0][1], fr[0][2], fr[0][3]);
        ldsm_x4(res_addr1 + rs * RB_BYTES, fr[1][0], fr[1][1], fr[1][2], fr[1][3]);
        {   // release the residual slot only once both ldmatrix instructions have returned their data (one destination register of each carries its
            // scoreboard) — WAR vs the producer's next TMA fill of this slot; the warp reduction makes lane 0's arrive depend on every lane; the proxy
            // fence orders these generic-proxy reads before the async-proxy (TMA) write that the release permits
          const uint32_t dep = zero_dep(__reduce_or_sync(0xffffffffu, fr[0][0] ^ fr[1][0]));
          fence_async_smem();
          if (lane == 0) mbar_arrive_dep(empty_res + cw * NRES + rs, dep);
        }
        const int nbase = b * BNO;
        CUTE_UNROLL
        for (int v2 = 0; v2 < BNO / 8; ++v2) {
#if LNFOLD
          const float2 cp = *reinterpret_cast<const float2*>(s_ln + nbase + nq + 8 * v2);
          const float2 cg = *reinterpret_cast<const float2*>(s_ln + 256 + nbase + nq + 8 * v2);
#endif
          CUTE_UNROLL
          for (int v1 = 0; v1 < 2; ++v1) {
            const int e = 2 * v1 + 4 * v2;
            const int kb = v2 >> 1, q = 2 * (v2 & 1) + v1;
            float g0 = accG(e), g1 = accG(e + 1), p0 = accP(e), p1 = accP(e + 1);
#if LNFOLD
            g0 += cg.x; g1 += cg.y; p0 += cp.x; p1 += cp.y;
#endif
            const float2 zr = unpack_bf16x2(fr[kb][q]);
#if FASTSIG
            const float o0 = sigmoid_tanh(g0) * p0 + zr.x; const float o1 = sigmoid_tanh(g1) * p1 + zr.y;
#else
            const float o0 = sigmoid_exact(g0) * p0 + zr.x; const float o1 = sigmoid_exact(g1) * p1 + zr.y;
#endif
            fr[kb][q] = pack_bf16x2(o0, o1);
          }
        }
        // my warp's 16-row slice -> staging (stmatrix), then my lane 0 stores it with one TMA (box {32 ch, 16 j, 1 i}; j >= N clipped)
        stsm_x4(out_addr0 + os * RB_BYTES, fr[0][0], fr[0][1], fr[0][2], fr[0][3]);
        stsm_x4(out_addr1 + os * RB_BYTES, fr[1][0], fr[1][1], fr[1][2], fr[1][3]);
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          tma_store_3d(&tm_out, smem + OFF_OUT + (cw * NOUT + os) * RB_BYTES + wq * (16 * BNO * 2), nbase, tok0 + 16 * wq, i);
          tma_store_commit();
        }
        ++oseq;
#else
        { float sink = 0.f;
          CUTE_UNROLL
          for (int e = 0; e < size(accP); ++e) sink += accP(e) * 1e-3f + accG(e);
          if (sink == 123.456f && prof != nullptr) prof[3] = 1; }
#endif
        PCLK(eb1); PACC(8, eb0, eb1);
      };
#if NACC == 2
      // software pipeline over the NB blocks, 2-way manual unroll (accumulator set alternation must be static), outer loop kept rolled
      issue(accP0, accG0);
      #pragma unroll 1
      for (int bb = 0; bb < NB; bb += 2) {
        {
          const int sP = (seq - 2) % NSLOT, sG = (seq - 1) % NSLOT;
          issue(accP1, accG1);
          PCLK(m0); warpgroup_wait<2>(); PCLK(m1); PACC(5, m0, m1);
          PCLK(e0); epilogue(bb, accP0, accG0, sP, sG); PCLK(e1); PACC(6, e0, e1);
        }
        {
          const int sP = (seq - 2) % NSLOT, sG = (seq - 1) % NSLOT;
          if (bb + 2 < NB) { issue(accP0, accG0); PCLK(m0); warpgroup_wait<2>(); PCLK(m1); PACC(5, m0, m1); }
          else             { PCLK(m0); warpgroup_wait<0>(); PCLK(m1); PACC(5, m0, m1); }
          PCLK(e0); epilogue(bb + 1, accP1, accG1, sP, sG); PCLK(e1); PACC(6, e0, e1);
        }
      }
#else
      #pragma unroll 1
      for (int b = 0; b < NB; ++b) {
        issue(accP0, accG0);
        const int sP = (seq - 2) % NSLOT, sG = (seq - 1) % NSLOT;
        PCLK(m0); warpgroup_wait<0>(); PCLK(m1); PACC(5, m0, m1);
        PCLK(e0);
        epilogue(b, accP0, accG0, sP, sG);
        PCLK(e1); PACC(6, e0, e1);
      }
#endif
    }
#ifndef X2
    if (lane == 0) { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
#endif
#if PROF
    if (prof_on) { for (int q = 0; q < 10; ++q) prof[(blockIdx.x * 2 + cw) * 10 + q] = prof_acc[q]; }
#endif
  }
}
'''


# =====================================================================================================================
# build (NVRTC -> cubin, cached) and launch (CUDA driver API)
# =====================================================================================================================
_CFG = dict(NSLOT=4, NRES=2, NOUT=1, NACC=2, BT=128, BNO=32)
_DEV_OPTS = []          # extra NVRTC options (development experiments only; part of the cache key)
_STATE = dict(kernels={}, compiles=[], device_checked=False)


def _smem_bytes(c=_CFG):
    off_bar = 2 * 32768 + 65536 + c["NSLOT"] * 16384 + 2 * c["NRES"] * 4096 + 2 * c["NOUT"] * 4096 + 4096
    return off_bar + (2 + 2 * c["NSLOT"] + 4 * c["NRES"]) * 8 + 1024


def _jit_dir():
    """<MODEL_OPT_JIT_ROOT>/<stack key>/nvrtc when the kit's JIT root is configured (configs/h100.env puts the Triton cache at <root>/<key>/triton),
    else next to TRITON_CACHE_DIR, else under the system temp dir.  The directory may be read-only (a pre-warmed root): see _compile."""
    root = (os.environ.get("MODEL_OPT_JIT_ROOT") or "").rstrip("/")
    if root:
        key = os.environ.get("MODEL_OPT_STACK_KEY") or ""
        if not key:
            try:                                                    # this carried copy serves prebuilt cubins and never compiles: the stack
                raise LookupError("MODEL_OPT_STACK_KEY unset")     # key comes from the environment only (no engine package is imported here)
            except Exception:
                key = "unknown-stack"
        return os.path.join(root, key, "nvrtc")
    t = (os.environ.get("TRITON_CACHE_DIR") or "").rstrip("/")
    return os.path.join(os.path.dirname(t), "nvrtc") if t else os.path.join(tempfile.gettempdir(), "esmfold2_opt_nvrtc")


def _include_dirs():
    try:
        import cutlass_library
    except ImportError as e:
        raise V6Unavailable("ef2_trimul_v6: the `nvidia-cutlass` wheel (CuTe headers) is not installed on this stack") from e
    sp = os.path.dirname(os.path.dirname(torch.__file__))
    d_cutlass = os.path.join(os.path.dirname(cutlass_library.__file__), "source", "include")
    d_cccl = os.path.join(sp, "nvidia", "cu13", "include", "cccl")
    d_cuda = os.path.join(sp, "nvidia", "cu13", "include")
    for d, probe, wheel in ((d_cutlass, "cute/tensor.hpp", "nvidia-cutlass"), (d_cccl, "cuda/std/cstdint", "nvidia-cuda-cccl"), (d_cuda, "cuda_bf16.h", "nvidia-cuda-runtime (torch cu13 wheels)")):
        if not os.path.exists(os.path.join(d, probe)):
            raise V6Unavailable(f"ef2_trimul_v6: header {probe} not found under {d} (wheel {wheel})")
    return [d_cutlass, d_cccl, d_cuda]


def _drv():
    try:
        from cuda.bindings import driver, nvrtc
    except ImportError as e:
        raise V6Unavailable("ef2_trimul_v6: cuda.bindings (cuda-bindings wheel) not importable") from e
    return driver, nvrtc


def _chk(res):
    driver, nvrtc = _drv()
    err, vals = (res[0], res[1:]) if isinstance(res, tuple) else (res, ())
    if isinstance(err, driver.CUresult):
        if err != driver.CUresult.CUDA_SUCCESS:
            _, name = driver.cuGetErrorName(err)
            raise V6Unavailable(f"ef2_trimul_v6: CUDA driver error {name.decode() if isinstance(name, bytes) else name}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise V6Unavailable(f"ef2_trimul_v6: NVRTC error {err}")
    return vals[0] if len(vals) == 1 else vals


def _check_device():
    if _STATE["device_checked"]:
        return
    if not torch.cuda.is_available():
        raise V6Unavailable("ef2_trimul_v6: no CUDA device")
    cc = torch.cuda.get_device_capability()
    if cc != (9, 0):
        raise V6Unavailable(f"ef2_trimul_v6: the CuTe K3 kernel is built for sm_90a only; this device is sm_{cc[0]}{cc[1]}")
    _STATE["device_checked"] = True


def _compile(fastsig, lnfold):
    """NVRTC compile of K3_SRC for sm_90a (or cache hit).  Prints one line naming the compile."""
    driver, nvrtc = _drv()
    opts = ["--gpu-architecture=sm_90a", "-std=c++17", "-default-device", "-diag-suppress=47",
            f"-DFASTSIG={int(bool(fastsig))}", f"-DLNFOLD={int(bool(lnfold))}", *_DEV_OPTS, f"-DNSLOT={_CFG['NSLOT']}", f"-DNRES={_CFG['NRES']}", f"-DNOUT={_CFG['NOUT']}", f"-DNACC={_CFG['NACC']}"]
    opts += [f"-I{d}" for d in _include_dirs()]
    ver = _chk(nvrtc.nvrtcVersion())
    key = hashlib.sha256((K3_SRC + "\n".join(opts) + repr(ver)).encode()).hexdigest()[:32]
    path = os.path.join(_jit_dir(), f"k3v6_{key}.cubin")
    if os.path.exists(path):
        sys.stderr.write(f"[esmfold2-opt] ef2_trimul_v6: K3 CuTe kernel (fastsig={int(bool(fastsig))} lnfold={int(bool(lnfold))}) cubin from cache {path}\n")
        return open(path, "rb").read()
    t0 = time.time()
    sys.stderr.write(f"[esmfold2-opt] ef2_trimul_v6: compiling the K3 CuTe kernel with NVRTC {ver[0]}.{ver[1]} for sm_90a (fastsig={int(bool(fastsig))} lnfold={int(bool(lnfold))}) ...\n"); sys.stderr.flush()
    prog = _chk(nvrtc.nvrtcCreateProgram(K3_SRC.encode(), b"ef2_trimul_v6_k3.cu", 0, [], []))
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), [o.encode() for o in opts])
    err = err[0] if isinstance(err, tuple) else err
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        n = _chk(nvrtc.nvrtcGetProgramLogSize(prog)); log = b" " * n; nvrtc.nvrtcGetProgramLog(prog, log)
        text = log.decode(errors="replace"); lines = text.split("\n")
        hits = [k for k, l in enumerate(lines) if " error" in l or l.startswith("error")]
        excerpt = "\n".join("\n".join(lines[max(0, k - 2):k + 4]) for k in hits[:6]) or text[-3000:]
        raise V6Unavailable(f"ef2_trimul_v6: NVRTC compilation failed ({err}):\n{excerpt}")
    n = _chk(nvrtc.nvrtcGetCUBINSize(prog)); cubin = b" " * n; _chk(nvrtc.nvrtcGetCUBIN(prog, cubin))
    try:                                                        # persist for the next process; a read-only / absent cache root is not an error (the cubin
        os.makedirs(os.path.dirname(path), exist_ok=True)       # stays in memory for this process, exactly like Triton's behaviour on such a root)
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(cubin)
        os.replace(tmp, path)
        where = path
    except OSError as e:
        where = f"memory only (JIT cache {os.path.dirname(path)} not writable: {e.strerror})"
    sys.stderr.write(f"[esmfold2-opt] ef2_trimul_v6: compiled in {time.time() - t0:.1f}s -> {where}\n")
    return cubin


class _Kernel:
    def __init__(self, cubin, name, smem_bytes):
        driver, _ = _drv()
        self.driver = driver
        self.module = _chk(driver.cuModuleLoadData(cubin))
        self.func = _chk(driver.cuModuleGetFunction(self.module, name.encode()))
        _chk(driver.cuFuncSetAttribute(self.func, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes))
        self.smem = smem_bytes

    def __call__(self, grid, block, args, stream):
        holders, ptrs = [], []
        for a in args:
            if isinstance(a, torch.Tensor):
                h = ctypes.c_uint64(a.data_ptr())
            elif isinstance(a, _TensorMap):
                h = a.cbuf
            elif isinstance(a, float):
                h = ctypes.c_float(a)
            elif isinstance(a, int):
                h = ctypes.c_int32(a)
            elif isinstance(a, ctypes.c_uint64):
                h = a
            else:
                raise TypeError(f"unsupported kernel argument {type(a)}")
            holders.append(h); ptrs.append(ctypes.addressof(h))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        _chk(self.driver.cuLaunchKernel(self.func, grid[0], grid[1], grid[2], block[0], block[1], block[2], self.smem, stream, arr, 0))
        return holders


class _TensorMap:
    """CUtensorMap over a bf16 tensor; dims/box innermost-first; strides in bytes for dims 1..; swizzle in {0, 32, 64, 128}."""
    def __init__(self, tensor, dims, strides_bytes, box, swizzle):
        driver, _ = _drv()
        sw = {0: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE, 32: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
              64: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B, 128: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B}[swizzle]
        rank = len(dims)
        self.tm = _chk(driver.cuTensorMapEncodeTiled(driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, tensor.data_ptr(),
                                                       [driver.cuuint64_t(d) for d in dims], [driver.cuuint64_t(s) for s in strides_bytes],
                                                       [driver.cuuint32_t(b) for b in box], [driver.cuuint32_t(1)] * rank,
                                                       driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                                                       driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                                       driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))
        buf = (ctypes.c_uint64 * 16)()
        vals = [int(v) for v in self.tm.opaque]
        if len(vals) != 16:
            raise V6Unavailable("ef2_trimul_v6: unexpected CUtensorMap layout in cuda.bindings")
        for k, v in enumerate(vals):
            buf[k] = v
        self.cbuf = buf
        self.keep = tensor


def build(fastsig=False, lnfold=True):
    """Compile (or load from cache) and load the K3 kernel variant (gate form x LN folding).  Raises V6Unavailable by name on any failure."""
    key = (int(bool(fastsig)), int(bool(lnfold)))
    if key in _STATE["kernels"]:
        return _STATE["kernels"][key]
    _check_device()
    cubin = _compile(fastsig, lnfold)
    kern = _Kernel(cubin, "k3v6", _smem_bytes())
    _STATE["kernels"][key] = kern
    _STATE["compiles"].append(dict(fastsig=key[0], lnfold=key[1], smem=kern.smem))
    return kern


def fold_weights(w):
    """LN affine folded into the projection weights (cached in the T9 weight pack): W' = bf16(W * gamma[k]), bias = W @ beta (fp32).
    Runs with autocast disabled: the model's forward is under bf16 autocast, which would otherwise turn the fp32 bias GEMV into a bf16
    tensor that the kernel (reading `const float*`) would misinterpret."""
    if "v6_wo" not in w:
        with torch.autocast("cuda", enabled=False):
            wz, wg = w["wz"], w["wg_out"]                              # [C_out, CH] / [C_out, C]  bf16, row-major (K contiguous)
            w["v6_wo"] = (wz.float() * w["ln_out_w"].float()[None, :]).to(torch.bfloat16).contiguous()
            w["v6_wg"] = (wg.float() * w["ln_in_w"].float()[None, :]).to(torch.bfloat16).contiguous()
            w["v6_bp"] = (wz.float() @ w["ln_out_b"].float()).float().contiguous()
            w["v6_bg"] = (wg.float() @ w["ln_in_b"].float()).float().contiguous()
    for k, dt in (("v6_wo", torch.bfloat16), ("v6_wg", torch.bfloat16), ("v6_bp", torch.float32), ("v6_bg", torch.float32)):
        t = w[k]
        if t.dtype != dt or not t.is_contiguous() or not t.is_cuda:
            raise V6Unavailable(f"ef2_trimul_v6: folded weight {k} is {t.dtype} contiguous={t.is_contiguous()} (expected {dt}, contiguous, CUDA)")
    return w


def launch_k3(x, z3, w, out, N, Np, eps, fastsig, lnfold=True, residual=True, stock_round=False):
    """x [CH, Np, Np] bf16 channel-major planes; z3 [N, N, C] bf16 contiguous; out [N, N, C] bf16 (written).  C == CH == 256."""
    if not residual or stock_round:
        raise V6Unavailable("ef2_trimul_v6: only the residual=True / stock_round=False epilogue (the one ef2_w4's T9 route uses) is implemented")
    C = z3.shape[-1]; CH = x.shape[0]
    if C != 256 or CH != 256 or x.shape[1] != Np or x.shape[2] != Np or Np % 8:
        raise V6Unavailable(f"ef2_trimul_v6: unsupported shapes x {tuple(x.shape)} z {tuple(z3.shape)} Np {Np}")
    if not (x.is_contiguous() and z3.is_contiguous() and out.is_contiguous()):
        raise V6Unavailable("ef2_trimul_v6: x, z, out must be contiguous")
    if x.dtype != torch.bfloat16 or z3.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise V6Unavailable(f"ef2_trimul_v6: x/z/out must be bf16 (got {x.dtype}, {z3.dtype}, {out.dtype})")
    for k in ("ln_out_w", "ln_out_b", "ln_in_w", "ln_in_b"):
        if w[k].dtype != torch.float32 or not w[k].is_contiguous():
            raise V6Unavailable(f"ef2_trimul_v6: weight pack entry {k} must be contiguous fp32 (got {w[k].dtype})")
    for k in ("wz", "wg_out"):
        if w[k].dtype != torch.bfloat16 or not w[k].is_contiguous() or tuple(w[k].shape) != (256, 256):
            raise V6Unavailable(f"ef2_trimul_v6: weight pack entry {k} must be contiguous bf16 [256, 256] (got {w[k].dtype} {tuple(w[k].shape)})")
    kern = build(fastsig, lnfold)
    fold_weights(w)
    wo, wg = (w["v6_wo"], w["v6_wg"]) if lnfold else (w["wz"], w["wg_out"])
    BT, BNO = _CFG["BT"], _CFG["BNO"]
    tm_x = _TensorMap(x, dims=[Np, Np, CH], strides_bytes=[Np * 2, Np * Np * 2], box=[64, 1, CH], swizzle=128)
    tm_z = _TensorMap(z3, dims=[C, N, N], strides_bytes=[C * 2, N * C * 2], box=[64, BT, 1], swizzle=128)
    tm_wo = _TensorMap(wo, dims=[CH, C], strides_bytes=[CH * 2], box=[64, BNO], swizzle=128)
    tm_wg = _TensorMap(wg, dims=[C, C], strides_bytes=[C * 2], box=[64, BNO], swizzle=128)
    tm_res = _TensorMap(z3, dims=[C, N, N], strides_bytes=[C * 2, N * C * 2], box=[BNO, 64, 1], swizzle=64)
    tm_out = _TensorMap(out, dims=[C, N, N], strides_bytes=[C * 2, N * C * 2], box=[BNO, 16, 1], swizzle=64)
    nJ = (N + BT - 1) // BT; n_tiles = N * nJ
    sms = torch.cuda.get_device_properties(z3.device).multi_processor_count
    grid = (min(sms, n_tiles), 1, 1)
    stream = torch.cuda.current_stream(z3.device).cuda_stream
    kern(grid, (384, 1, 1), [tm_x, tm_z, tm_wo, tm_wg, tm_res, tm_out, w["ln_out_w"], w["ln_out_b"], w["ln_in_w"], w["ln_in_b"], w["v6_bp"], w["v6_bg"],
                             int(N), int(Np), int(nJ), int(n_tiles), float(eps), ctypes.c_uint64(0)], stream)
    return out


def describe():
    return dict(cfg=dict(_CFG), compiled=list(_STATE["compiles"]), smem_bytes=_smem_bytes(), jit_dir=_jit_dir())
