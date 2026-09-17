// Device-side PTX wrappers (mma.sync, ldmatrix, mbarrier, TMA bulk-tensor loads) and the host-side tensor-map encoder used by
// triattn_mw.cu.  sm_90a.  No CUTLASS dependency.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define MW_DEVI __device__ __forceinline__

namespace mw {

MW_DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

MW_DEVI void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
MW_DEVI void ldsm_x4_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
MW_DEVI float4 lds128f(uint32_t addr) {
  float4 v; asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr)); return v;
}
MW_DEVI void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
MW_DEVI void mma16816_z(float* d, const uint32_t* a, const uint32_t* b) {   // D = A B (zero C)
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f));
}
MW_DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
MW_DEVI uint32_t pack_bf16(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r; }

// ---- mbarrier ----
MW_DEVI void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}
MW_DEVI void mbar_inval(uint64_t* bar) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(smem_u32(bar)) : "memory"); }
MW_DEVI uint32_t atom_add_shared(uint32_t* p, uint32_t v) {   // atom.shared (a generic-pointer atomicAdd compiles to a global-scope ATOM)
  uint32_t old;
  asm volatile("atom.shared::cta.add.u32 %0, [%1], %2;\n" : "=r"(old) : "r"(smem_u32(p)), "r"(v) : "memory");
  return old;
}
MW_DEVI uint32_t atom_add_shared_lane0(uint32_t* p, uint32_t v, int lane) {   // predicated on lane == 0, no branch; other lanes return 0xffffffff
  uint32_t old = 0xffffffffu;
  asm volatile("{\n .reg .pred q;\n setp.eq.s32 q, %3, 0;\n @q atom.shared::cta.add.u32 %0, [%1], %2;\n}\n" : "+r"(old) : "r"(smem_u32(p)), "r"(v), "r"(lane) : "memory");
  return old;
}
MW_DEVI void st_shared_u32_if(uint32_t* p, uint32_t v, bool c) {            // predicated store, no branch
  asm volatile("{\n .reg .pred q;\n setp.ne.s32 q, %2, 0;\n @q st.shared::cta.u32 [%0], %1;\n}\n" :: "r"(smem_u32(p)), "r"(v), "r"((int)c) : "memory");
}
MW_DEVI void st_shared_u32(uint32_t* p, uint32_t v) { asm volatile("st.shared::cta.u32 [%0], %1;\n" :: "r"(smem_u32(p)), "r"(v) : "memory"); }
MW_DEVI void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p)); }
// Warp release of a ring slot on an 'empty' barrier initialised with count WARPS + 1: lane 0 arrives without completing and reads
// the arrivals still pending before its own (WARPS + 1 for the first warp to release, 2 for the last); the last one adds the
// completing arrival.  Returns that pending count on lane 0 (0 on other lanes) -- no atomics, no branches.
MW_DEVI uint32_t mbar_release_elect(uint64_t* bar, int lane) {
  uint32_t n = 0;
  asm volatile("{\n .reg .pred p, q;\n .reg .b64 st;\n"
               " setp.eq.s32 p, %2, 0;\n"
               " @p mbarrier.arrive.noComplete.shared::cta.b64 st, [%1], 1;\n"
               " @p mbarrier.pending_count.b64 %0, st;\n"
               " setp.eq.and.u32 q, %0, 2, p;\n"
               " @q mbarrier.arrive.shared::cta.b64 st, [%1];\n"
               "}\n" : "+r"(n) : "r"(smem_u32(bar)), "r"(lane) : "memory");
  return n;
}
MW_DEVI uint32_t atom_ticket_lane0(uint32_t* p, int lane, uint32_t zero_per_lane) {   // lane 0: old value of (*p)++, others: ~0
  uint32_t old = 0xffffffffu;
  const uint32_t addr = smem_u32(p) + zero_per_lane;             // zero_per_lane == 0, but not provably (defeats atomic aggregation)
  asm volatile("{\n .reg .pred q;\n setp.eq.s32 q, %2, 0;\n @q atom.shared::cta.add.u32 %0, [%1], 1;\n}\n" : "+r"(old) : "r"(addr), "r"(lane) : "memory");
  return old;
}
template <typename T> MW_DEVI T opaque(T x) { asm volatile("" : "+r"(x)); return x; }   // value the compiler must keep (no rematerialisation)
MW_DEVI void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
MW_DEVI void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
MW_DEVI void mbar_arrive(uint64_t* bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(smem_u32(bar)) : "memory");
}
MW_DEVI void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
MW_DEVI bool mbar_try_wait(uint64_t* bar, uint32_t phase) {   // with a suspendTimeHint: a not-yet-ready wait parks the warp instead of spinning
  uint32_t ok;
#ifdef MW_SUSPEND_HINT
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase), "r"(0x989680u) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
#endif
  return ok != 0;
}
MW_DEVI bool mbar_test(uint64_t* bar, uint32_t phase) {       // non-blocking probe (acquire on success)
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  return ok != 0;
}
MW_DEVI void mbar_wait(uint64_t* bar, uint32_t phase) { while (!mbar_try_wait(bar, phase)) { } }

// ---- TMA bulk tensor loads (global -> shared::cta, completion on an mbarrier) ----
MW_DEVI void tma_load_4d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
MW_DEVI void tma_load_5d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, int c2, int c3, int c4) {
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];\n"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
}
MW_DEVI void tma_prefetch_desc(const CUtensorMap* map) { asm volatile("prefetch.tensormap [%0];\n" :: "l"(map) : "memory"); }

}  // namespace mw

// ---------------------------------------------------------------- host ----------------------------------------------------------------
#include <stdexcept>
#include <string>
namespace mw {
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
// rank-R tiled map; dims/strides fastest-first; strides in BYTES for dims 1..R-1 (dim 0 is contiguous)
inline CUtensorMap make_map(CUtensorMapDataType dt, int rank, void* base, const uint64_t* dims, const uint64_t* strides_bytes,
                           const uint32_t* box, CUtensorMapSwizzle swz) {
  CUtensorMap m; uint32_t estr[5] = {1, 1, 1, 1, 1};
  CUresult r = encode_fn()(&m, dt, rank, base, dims, strides_bytes, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    std::string s = "cuTensorMapEncodeTiled failed (" + std::to_string((int)r) + ") rank=" + std::to_string(rank) + " dims=";
    for (int i = 0; i < rank; ++i) s += std::to_string(dims[i]) + ",";
    s += " strides="; for (int i = 0; i < rank - 1; ++i) s += std::to_string(strides_bytes[i]) + ",";
    s += " box="; for (int i = 0; i < rank; ++i) s += std::to_string(box[i]) + ",";
    throw std::runtime_error(s);
  }
  return m;
}
}  // namespace mw
