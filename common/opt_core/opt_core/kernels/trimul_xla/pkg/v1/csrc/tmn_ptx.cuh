// SPDX-License-Identifier: Apache-2.0
// tmn_ptx.cuh — device-side PTX wrappers for the trimul_native kernels: mbarrier, TMA bulk-tensor loads, ldmatrix/stmatrix, vector shared/global
// memory, wgmma (bf16 -> fp32, A from registers, B from a shared-memory descriptor), setmaxnreg, named barriers.  Device code only: no host
// symbols, no CUDA runtime, no framework headers (the host side encodes tensor maps and launches through the CUDA driver from Python).
// Derived from trimul_tx 1.2 (tx_ptx.h).  The sm_90a-specific pieces (TMA, wgmma, setmaxnreg, mbarrier tx-count) are used only by the sm_90a
// mainloop/staging layer of tmn_kernels.cuh; the math helpers (bf16 packing, LayerNorm fragments) are architecture-neutral.
#pragma once
#include <cuda.h>          // CUtensorMap (opaque 128-byte descriptor type; passed by value inside the kernel parameter struct)
#include <cuda_bf16.h>
#include <cstdint>
#include <utility>

#define TMN_DEVI __device__ __forceinline__

namespace tmn {

TMN_DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
TMN_DEVI uint32_t dyn_smem_size() { uint32_t r; asm volatile("mov.u32 %0, %%dynamic_smem_size;\n" : "=r"(r)); return r; }

// ---------------------------------------------------------------- ldmatrix / stmatrix / vector memory
TMN_DEVI void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
TMN_DEVI void ldsm_x4_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
TMN_DEVI void stsm_x4_t(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};\n" :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
TMN_DEVI void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
TMN_DEVI uint4 lds128(uint32_t addr) {
  uint4 v; asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr)); return v;
}
TMN_DEVI float2 lds64f(uint32_t addr) {
  float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(addr)); return v;
}
TMN_DEVI void sts64f(uint32_t addr, float a, float b) { asm volatile("st.shared.v2.f32 [%0], {%1,%2};\n" :: "r"(addr), "f"(a), "f"(b) : "memory"); }
TMN_DEVI void sts32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(addr), "r"(v) : "memory"); }
TMN_DEVI void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};\n" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
TMN_DEVI void stg64f(void* p, float a, float b) { asm volatile("st.global.v2.f32 [%0], {%1,%2};\n" :: "l"(p), "f"(a), "f"(b) : "memory"); }
TMN_DEVI uint4 ldg128(const void* p) { uint4 v; asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
TMN_DEVI float2 ldg64f(const void* p) { float2 v; asm volatile("ld.global.nc.v2.f32 {%0,%1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "l"(p)); return v; }
TMN_DEVI uint32_t ldg32(const void* p) { uint32_t v; asm volatile("ld.global.nc.b32 %0, [%1];\n" : "=r"(v) : "l"(p)); return v; }
TMN_DEVI void stg32(void* p, uint32_t v) { asm volatile("st.global.b32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory"); }
TMN_DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
TMN_DEVI float rcpf(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
TMN_DEVI float rsqrt_approx_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
// bf16x2 <-> f32 (low half = first / even element in memory order)
TMN_DEVI uint32_t pack_bf16(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
TMN_DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
TMN_DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// 8 bf16 (one 16-byte granule in registers) -> global, first n elements only, as predicated 2-byte stores in ONE asm block (no lane-divergent
// branches): unpadded planes whose row length is not a multiple of 8.
TMN_DEVI void stg_ragged(__nv_bfloat16* p, uint4 v, int n) {
  asm volatile(
    "{\n"
    ".reg .pred q<8>;\n"
    ".reg .b16 h<8>;\n"
    "mov.b32 {h0, h1}, %1;\n mov.b32 {h2, h3}, %2;\n mov.b32 {h4, h5}, %3;\n mov.b32 {h6, h7}, %4;\n"
    "setp.gt.s32 q0, %5, 0;\n setp.gt.s32 q1, %5, 1;\n setp.gt.s32 q2, %5, 2;\n setp.gt.s32 q3, %5, 3;\n"
    "setp.gt.s32 q4, %5, 4;\n setp.gt.s32 q5, %5, 5;\n setp.gt.s32 q6, %5, 6;\n setp.gt.s32 q7, %5, 7;\n"
    "@q0 st.global.b16 [%0], h0;\n @q1 st.global.b16 [%0+2], h1;\n @q2 st.global.b16 [%0+4], h2;\n @q3 st.global.b16 [%0+6], h3;\n"
    "@q4 st.global.b16 [%0+8], h4;\n @q5 st.global.b16 [%0+10], h5;\n @q6 st.global.b16 [%0+12], h6;\n @q7 st.global.b16 [%0+14], h7;\n"
    "}\n" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(n) : "memory");
}

// ---------------------------------------------------------------- mbarrier
TMN_DEVI void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}
TMN_DEVI void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
TMN_DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
TMN_DEVI void mbar_arrive(uint64_t* bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(smem_u32(bar)) : "memory");
}
TMN_DEVI void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
TMN_DEVI bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  return ok != 0;
}
TMN_DEVI void mbar_wait(uint64_t* bar, uint32_t phase) { while (!mbar_try_wait(bar, phase)) { } }

// ---------------------------------------------------------------- TMA loads (sm_90)
TMN_DEVI void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI void tma_load_3d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
TMN_DEVI void tma_prefetch_desc(const CUtensorMap* map) { asm volatile("prefetch.tensormap [%0];\n" :: "l"(map) : "memory"); }


// SWIZZLE_64B pattern of a [rows][64 B] tile (16-byte granule ^= bits 7..8 of the offset); base 512-byte aligned
TMN_DEVI uint32_t sw64(uint32_t off) { return off ^ (((off >> 7) & 3u) << 4); }
// == 0 at run time for any dep, opaque to ptxas (the dynamic smem size is a launch parameter < 1 MB): makes an mbarrier arrive data-dependent on
// registers an asynchronous-proxy consumer must not overtake (ldmatrix destinations), i.e. the release is issued only after the reads returned.
TMN_DEVI uint32_t zero_dep(uint32_t dep) { return dep & (dyn_smem_size() >> 20); }
TMN_DEVI void mbar_arrive_dep(uint64_t* bar, uint32_t dep_zero) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_u32(bar) + dep_zero) : "memory"); }
// TMA bulk-tensor STORE smem -> global (elements outside the tensor bounds are not written) + bulk-group tracking
TMN_DEVI void tma_store_3d(const CUtensorMap* map, const void* src, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];\n"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
TMN_DEVI void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int N> TMN_DEVI void tma_store_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory"); }
TMN_DEVI void tma_store_wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }

// ---------------------------------------------------------------- warpgroup / registers (sm_90)
template <int N> TMN_DEVI void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N)); }
template <int N> TMN_DEVI void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N)); }
// bar.sync is the .aligned barrier form: every thread of a warp must reach it convergently.  Lane-dependent branches right before a barrier
// (edge-tile store predicates, ragged element stores, a guarded global load) can leave a warp diverged under independent thread scheduling,
// so reconverge first.
TMN_DEVI void named_bar_sync(int id, int nthreads) { __syncwarp(); asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(nthreads) : "memory"); }

// ---------------------------------------------------------------- wgmma (sm_90a)
// Shared-memory matrix descriptor (PTX ISA, "matrix descriptor"): start address >> 4 in bits [0,14), leading-dim byte offset >> 4 in [16,30),
// stride-dim byte offset >> 4 in [32,46), base offset [49,52) = 0, swizzle mode in [62,64): 0 none, 1 128B, 2 64B, 3 32B.
TMN_DEVI uint64_t smem_desc(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t layout) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3fffu);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3fffu) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3fffu) << 32;
  d |= (uint64_t)(layout & 0x3u) << 62;
  return d;
}
TMN_DEVI void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
TMN_DEVI void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> TMN_DEVI void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
// keep the compiler from moving accumulator / operand register uses across the asynchronous MMA (CUTLASS warpgroup_fence_operand)
TMN_DEVI void fence_reg(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
TMN_DEVI void fence_reg(uint32_t& r) { asm volatile("" : "+r"(r) :: "memory"); }
template <int N> TMN_DEVI void fence_regs(float (&a)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) fence_reg(a[i]);
}
template <int N> TMN_DEVI void fence_regs(uint32_t (&a)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) fence_reg(a[i]);
}

// D[64x64 f32] = A[64x16 bf16, registers] * B[16x64 bf16 via smem desc, K-major] + (scale_d ? D : 0).  The descriptor is given as (lo, hi)
// 32-bit halves plus an IMMEDIATE byte offset added to the start-address field inside the asm block: a chain of k-steps over one smem slot then
// keeps two registers live for the descriptor instead of one 64-bit descriptor per k-step (register pressure: the K3 consumer holds two full
// operand tiles + two accumulators).
template <int OFF_BYTES>
TMN_DEVI void wgmma_m64n64k16_rs_off(float (&d)[32], const uint32_t (&a)[4], uint32_t desc_lo, uint32_t desc_hi, int scale_d) {
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

// A full K chain (KS k-steps of 16) of one m64n64 product: k-step ks reads the 64-wide weight k-chunk ks/4, which lives in ring slot (ks/4)/SKCH at byte
// offset ((ks/4) % SKCH) * 8192, k sub-step (ks%4) * 32 bytes into the 128-byte rows; dlo/dhi hold one descriptor per slot.
template <int SKCH, int KS, int NS, int... Is>
TMN_DEVI void mma_chain_(float (&acc)[32], const uint32_t (&f)[KS][4], const uint32_t (&dlo)[NS], const uint32_t (&dhi)[NS], std::integer_sequence<int, Is...>) {
  (wgmma_m64n64k16_rs_off<(((Is >> 2) % SKCH) * 8192 + (Is & 3) * 32)>(acc, f[Is], dlo[(Is >> 2) / SKCH], dhi[(Is >> 2) / SKCH], Is > 0 ? 1 : 0), ...);
}
template <int SKCH, int KS, int NS>
TMN_DEVI void mma_chain(float (&acc)[32], const uint32_t (&f)[KS][4], const uint32_t (&dlo)[NS], const uint32_t (&dhi)[NS]) {
  static_assert(NS * SKCH * 4 == KS, "one descriptor per slot, SKCH chunks of 4 k-steps per slot");
  mma_chain_<SKCH>(acc, f, dlo, dhi, std::make_integer_sequence<int, KS>{});
}


// D[64x32 f32] (16 registers) = A[64x16 bf16, registers] * B[16x32 via smem desc] (+ D): the K3 output-block shape
template <int OFF_BYTES>
TMN_DEVI void wgmma_m64n32k16_rs_off(float (&d)[16], const uint32_t (&a)[4], uint32_t desc_lo, uint32_t desc_hi, int scale_d) {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    ".reg .b32 lo;\n"
    ".reg .b64 dsc;\n"
    "setp.ne.b32 p, %22, 0;\n"
    "add.u32 lo, %20, %23;\n"
    "mov.b64 dsc, {lo, %21};\n"
    "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},"
    "{%16, %17, %18, %19}, dsc, p, 1, 1, 0;\n"
    "}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(desc_lo), "r"(desc_hi), "r"(scale_d), "n"(OFF_BYTES >> 4));
}
// full K chain of one m64n32 product over a weight block resident in ONE slot as k-chunks of [32 n][64 k] (4 KB, 128-B swizzled rows)
template <int KS, int... Is>
TMN_DEVI void mma_chain32_(float (&acc)[16], const uint32_t (&f)[KS][4], uint32_t dlo, uint32_t dhi, std::integer_sequence<int, Is...>) {
  (wgmma_m64n32k16_rs_off<((Is >> 2) * 4096 + (Is & 3) * 32)>(acc, f[Is], dlo, dhi, Is > 0 ? 1 : 0), ...);
}
template <int KS>
TMN_DEVI void mma_chain32(float (&acc)[16], const uint32_t (&f)[KS][4], uint32_t dlo, uint32_t dhi) {
  mma_chain32_(acc, f, dlo, dhi, std::make_integer_sequence<int, KS>{});
}

}  // namespace tmn
