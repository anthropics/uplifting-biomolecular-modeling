// tx_ptx.h — device-side PTX wrappers for the trimul_tx kernels (sm_90a): mbarrier, TMA bulk-tensor loads, ldmatrix/stmatrix,
// wgmma (bf16 -> fp32, A from registers, B from a shared-memory descriptor), setmaxnreg, named barriers; plus the host-side
// tensor-map encoder (driver entry point fetched at run time: no -lcuda).  No CUTLASS dependency.
// mbarrier / TMA / tensor-map encoder wrappers in the style of the earlier hand-written sm_90 kernels in this code base.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define TX_DEVI __device__ __forceinline__

namespace tx {

TX_DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---------------------------------------------------------------- ldmatrix / stmatrix / vector memory ------------------------------------------------
TX_DEVI void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
TX_DEVI void ldsm_x4_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
TX_DEVI void stsm_x4_t(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n" :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
TX_DEVI void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
TX_DEVI uint4 lds128(uint32_t addr) {
  uint4 v; asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr)); return v;
}
TX_DEVI void sts32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(addr), "r"(v) : "memory"); }
TX_DEVI void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
TX_DEVI void stg32(void* p, uint32_t v) { asm volatile("st.global.b32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory"); }
TX_DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
TX_DEVI float rcpf(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
// bf16x2 <-> f32 (low half = first / even element in memory order)
TX_DEVI uint32_t pack_bf16(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
TX_DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
TX_DEVI uint32_t pack_f16(float lo, float hi) { uint32_t r; asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
TX_DEVI float f16lo(uint32_t v) { float f; asm("{\n .reg .f16 h;\n mov.b32 {h, _}, %1;\n cvt.f32.f16 %0, h;\n}\n" : "=f"(f) : "r"(v)); return f; }
TX_DEVI float f16hi(uint32_t v) { float f; asm("{\n .reg .f16 h;\n mov.b32 {_, h}, %1;\n cvt.f32.f16 %0, h;\n}\n" : "=f"(f) : "r"(v)); return f; }
TX_DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// ---------------------------------------------------------------- mbarrier ----------------------------------------------------------------
TX_DEVI void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}
TX_DEVI void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
TX_DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
TX_DEVI void mbar_arrive(uint64_t* bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(smem_u32(bar)) : "memory");
}
TX_DEVI void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
TX_DEVI bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  return ok != 0;
}
TX_DEVI void mbar_wait(uint64_t* bar, uint32_t phase) { while (!mbar_try_wait(bar, phase)) { } }

// ---------------------------------------------------------------- TMA loads ----------------------------------------------------------------
TX_DEVI void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
TX_DEVI void tma_load_3d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
TX_DEVI void tma_prefetch_desc(const CUtensorMap* map) { asm volatile("prefetch.tensormap [%0];\n" :: "l"(map) : "memory"); }


// ---------------------------------------------------------------- clusters ----------------------------------------------------------------
TX_DEVI uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
TX_DEVI void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// shared::cluster address of the same smem offset in CTA `rank` of this cluster
TX_DEVI uint32_t mapa_shared(uint32_t saddr, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(saddr), "r"(rank)); return r;
}
TX_DEVI void mbar_arrive_cluster(uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" :: "r"(cluster_addr) : "memory");
}
TX_DEVI void mbar_arrive_expect_tx_cluster(uint32_t cluster_addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n" :: "r"(cluster_addr), "r"(bytes) : "memory");
}
// TMA 2D load multicast to every CTA in cta_mask (same smem offset and same mbarrier offset in each destination CTA)
TX_DEVI void tma_load_2d_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, uint16_t cta_mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;\n"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(cta_mask) : "memory");
}
// ---------------------------------------------------------------- warpgroup / registers ----------------------------------------------------------------
template <int N> TX_DEVI void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N)); }
template <int N> TX_DEVI void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N)); }
// bar.sync is the .aligned barrier form: every thread of a warp must reach it convergently.  Lane-dependent branches right before a barrier (edge-tile
// store predicates, ragged element stores, a guarded global load) can leave a warp diverged under independent thread scheduling, so reconverge first.
TX_DEVI void named_bar_sync(int id, int nthreads) { __syncwarp(); asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(nthreads) : "memory"); }

// ---------------------------------------------------------------- wgmma ----------------------------------------------------------------
// Shared-memory matrix descriptor (PTX ISA, "matrix descriptor"): start address >> 4 in bits [0,14), leading-dim byte offset >> 4 in [16,30),
// stride-dim byte offset >> 4 in [32,46), base offset [49,52) = 0, swizzle mode in [62,64): 0 none, 1 128B, 2 64B, 3 32B.
TX_DEVI uint64_t smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t layout) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3fffu);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3fffu) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3fffu) << 32;
  d |= (uint64_t)(layout & 0x3u) << 62;
  return d;
}
TX_DEVI void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
TX_DEVI void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> TX_DEVI void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
// keep the compiler from moving accumulator / operand register uses across the asynchronous MMA (CUTLASS warpgroup_fence_operand)
TX_DEVI void fence_reg(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
TX_DEVI void fence_reg(uint32_t& r) { asm volatile("" : "+r"(r) :: "memory"); }
template <int N> TX_DEVI void fence_regs(float (&a)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) fence_reg(a[i]);
}
template <int N> TX_DEVI void fence_regs(uint32_t (&a)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) fence_reg(a[i]);
}

// D[64x64 f32] = A[64x16 bf16, registers] * B[16x64 bf16 via smem desc, K-major (tnspB = 0)] + (scale_d ? D : 0)
TX_DEVI void wgmma_m64n64k16_rs(float (&d)[32], const uint32_t (&a)[4], uint64_t desc_b, int scale_d) {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %37, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31},"
    "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\n"
    "}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d));
}


// same, with the descriptor given as (lo, hi) 32-bit halves plus an immediate byte offset added to the start address (offset >> 4 into lo)
template <int OFF_BYTES>
TX_DEVI void wgmma_m64n64k16_rs_off(float (&d)[32], const uint32_t (&a)[4], uint32_t desc_lo, uint32_t desc_hi, int scale_d) {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    ".reg .b32 lo;\n"
    ".reg .b64 dsc;\n"
    "setp.ne.b32 p, %38, 0;\n"
    "add.u32 lo, %36, %39;\n"
    "mov.b64 dsc, {lo, %37};\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31},"
    "{%32, %33, %34, %35}, dsc, p, 1, 1, 0;\n"
    "}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(desc_lo), "r"(desc_hi), "r"(scale_d), "n"(OFF_BYTES >> 4));
}

}  // namespace tx

// ---------------------------------------------------------------- host: tensor maps ----------------------------------------------------------------
#include <stdexcept>
#include <string>
namespace tx {
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
inline EncodeTiledFn encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    void* p = nullptr; cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) != cudaSuccess || !p)
      throw std::runtime_error("cuTensorMapEncodeTiled entry point not found");
    fn = (EncodeTiledFn)p;
  }
  return fn;
}
// rank-R tiled map; dims/boxes fastest-first; strides in BYTES for dims 1..R-1 (dim 0 contiguous); OOB reads fill zeros
inline CUtensorMap make_map(CUtensorMapDataType dt, int rank, void* base, const uint64_t* dims, const uint64_t* strides_bytes,
                           const uint32_t* box, CUtensorMapSwizzle swz, CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B) {
  CUtensorMap m; cuuint32_t estr[5] = {1, 1, 1, 1, 1};
  cuuint64_t d64[5], s64[5]; cuuint32_t b32[5];
  for (int i = 0; i < rank; ++i) { d64[i] = dims[i]; b32[i] = box[i]; }
  for (int i = 0; i < rank - 1; ++i) s64[i] = strides_bytes[i];
  CUresult r = encode_fn()(&m, dt, (cuuint32_t)rank, base, d64, s64, b32, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, swz, l2, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::string s = "cuTensorMapEncodeTiled failed (" + std::to_string((int)r) + ") rank=" + std::to_string(rank) + " dims=";
    for (int i = 0; i < rank; ++i) s += std::to_string(dims[i]) + ",";
    s += " strides="; for (int i = 0; i < rank - 1; ++i) s += std::to_string(strides_bytes[i]) + ",";
    s += " box="; for (int i = 0; i < rank; ++i) s += std::to_string(box[i]) + ",";
    throw std::runtime_error(s);
  }
  return m;
}
}  // namespace tx
