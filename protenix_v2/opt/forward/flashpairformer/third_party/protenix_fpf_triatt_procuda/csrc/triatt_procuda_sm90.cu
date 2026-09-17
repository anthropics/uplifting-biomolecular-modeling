// triatt_procuda_sm90.cu — sm_90a tri-attention prologue kernel (lever `triatt_prologue_cuda`):
//   z [NI, NJ, 256] bf16 (any row strides; ending node = transposed view) -> LayerNorm (the F1 cell's 'welford' emulation of upstream fast_layernorm, op for op)
//   -> q|k|v|g = x_ln @ W^T (bf16 out, fp32 accumulate, one K=256 chain per 64-col chunk = 16 x wgmma.m64n64k16, k ascending) + pair bias = fp32(bf16(x_ln @ Wb^T)).
// Persistent + warp-specialized (416 threads): WG2 = 4 producer warps stream raw z rows through a per-warp cp.async ring and normalise them into the 128B-swizzled
// A tile; warp 12 streams the 64-col weight chunks through a 3-slot TMA ring; WG0/WG1 = consumers: each copies its 64 A rows once per tile into registers
// (ldmatrix), then per chunk: 16 x wgmma (A from registers) -> wait -> epilogue (bf16 pack -> stmatrix into swizzled staging -> TMA scatter-store into q/k/v
// ([I,H,J,D] or head-major, padded or not — all in the tensor maps) / g [I,J,256]); pair bias by bounds-checked global stores.  The two consumer warpgroups
// alternate on the tensor cores.  No fast-math compilation: every LayerNorm op is explicit PTX (rn / ftz / approx as in the emulated arithmetic).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace f1v2 {


constexpr int BI = 8, BJ = 16, BM = BI * BJ;          // 128 rows per tile = (8 i) x (16 j)
constexpr int C = 256, H = 8, D = 32, HD = H * D, HB = 16;
constexpr int NCH_QKVG = 16;                          // 16 chunks of 64 output columns (q0..q3,k0..k3,v0..v3,g0..g3) + 1 bias chunk (16 cols, 8 valid)
constexpr int NCH = NCH_QKVG + 1;
constexpr int THREADS = 416;                          // WG0, WG1 = consumers (wgmma), WG2 = producers (LN), warp 12 = W-chunk TMA issuer
constexpr int WSLOTS = 3;                             // weight-chunk ring
constexpr int NSTG = 2;                               // staging buffers per consumer WG
constexpr int CP_ROWS = 16;                           // per-producer-warp cp.async ring of raw z rows (prefetch depth CP_ROWS-1 rows)
constexpr int A_BYTES = BM * C * 2;                   // 65536: [4 kc][128 rows][128 B] swizzle-128B (single buffer; consumers copy it to registers at tile start)
constexpr int WSLOT_BYTES = 64 * C * 2;               // 32768 per W slot: [4 kc][64 n][128 B] swizzle-128B (bias chunk: [4][16][128 B] = 8 KB inside)
constexpr int STG_BYTES = 8192;                       // per (wg, buffer): q/k/v box [2 h][4 i][16 j][32 d] bf16 (64-B rows, swizzle-64B) or g box [4 i][16 j][64 c] (128-B rows, swizzle-128B)
constexpr int CP_BYTES = CP_ROWS * C * 2;             // 8192 per producer warp: [CP_ROWS][512 B] raw rows (lane l owns bytes 16l..16l+15 of a row)
constexpr int OFF_A = 0, OFF_W = A_BYTES, OFF_STG = OFF_W + WSLOTS * WSLOT_BYTES, OFF_CP = OFF_STG + 2 * NSTG * STG_BYTES, OFF_BAR = OFF_CP + 4 * CP_BYTES;
constexpr int BAR_AFULL = 0, BAR_AEMPTY = 8, BAR_WFULL = 16, BAR_WEMPTY = 16 + 8 * WSLOTS;     // byte offsets inside the barrier block
constexpr int SMEM_BYTES = OFF_BAR + 16 + 16 * WSLOTS + 64;   // 229504

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// ------------------------------------------------------------------ mbarrier / fences
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count)); }
__device__ __forceinline__ void mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile("{\n .reg .pred P1;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n @P1 bra.uni DONE_%=;\n bra.uni WAIT_%=;\n DONE_%=:\n}"
                 :: "r"(bar), "r"(parity), "r"(0x989680) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }

// ------------------------------------------------------------------ TMA
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, uint32_t src_bytes) {   // 16-B global->shared async copy, zero-fills when src_bytes == 0
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, int c2, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_store_4d(const CUtensorMap* tm, uint32_t src, int c0, int c1, int c2, int c3) {
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
                 :: "l"(reinterpret_cast<uint64_t>(tm)), "r"(src), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void tma_store_3d(const CUtensorMap* tm, uint32_t src, int c0, int c1, int c2) {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
                 :: "l"(reinterpret_cast<uint64_t>(tm)), "r"(src), "r"(c0), "r"(c1), "r"(c2) : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }

// ------------------------------------------------------------------ wgmma helpers
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
    // K-major operand, 128-B swizzle: start addr >>4 | LBO 16 B (ignored for swizzled K-major) | SBO = 1024 B (8 rows x 128 B) | swizzle mode 1 (128B)
    return static_cast<uint64_t>((saddr & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
template <int N> __device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
// keeps the compiler from moving accumulator registers across the async completion point (cf. CUTLASS warpgroup_fence_operand)
__device__ __forceinline__ void fence_acc(float (&d)[32]) {
    #pragma unroll
    for (int i = 0; i < 32; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 p2 = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&p2);
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

// chunk MMA, A from registers (RS): wgmma.fence; 16 x wgmma m64n64k16 (K ascending, first overwrites D); commit — one opaque block; the 16 B descriptors
// are warp-uniform expressions of dB (ptxas keeps them in uniform registers)
__device__ __forceinline__ void mma_chunk_n64(float (&d)[32], const uint32_t (&a)[16][4], uint64_t dB) {
    asm volatile(
        "{\n .reg .pred pf, pt;\n"
        " setp.ne.b32 pf, %112, 0;\n setp.eq.b32 pt, %112, 0;\n"
        " wgmma.fence.sync.aligned;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %96, pf, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%36,%37,%38,%39}, %97, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%40,%41,%42,%43}, %98, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%44,%45,%46,%47}, %99, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%48,%49,%50,%51}, %100, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%52,%53,%54,%55}, %101, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%56,%57,%58,%59}, %102, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%60,%61,%62,%63}, %103, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%64,%65,%66,%67}, %104, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%68,%69,%70,%71}, %105, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%72,%73,%74,%75}, %106, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%76,%77,%78,%79}, %107, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%80,%81,%82,%83}, %108, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%84,%85,%86,%87}, %109, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%88,%89,%90,%91}, %110, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%92,%93,%94,%95}, %111, pt, 1, 1, 0;\n"
        " wgmma.commit_group.sync.aligned;\n}"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "r"(a[4][0]), "r"(a[4][1]), "r"(a[4][2]), "r"(a[4][3]), "r"(a[5][0]), "r"(a[5][1]), "r"(a[5][2]), "r"(a[5][3]), "r"(a[6][0]), "r"(a[6][1]), "r"(a[6][2]), "r"(a[6][3]), "r"(a[7][0]), "r"(a[7][1]), "r"(a[7][2]), "r"(a[7][3]), "r"(a[8][0]), "r"(a[8][1]), "r"(a[8][2]), "r"(a[8][3]), "r"(a[9][0]), "r"(a[9][1]), "r"(a[9][2]), "r"(a[9][3]), "r"(a[10][0]), "r"(a[10][1]), "r"(a[10][2]), "r"(a[10][3]), "r"(a[11][0]), "r"(a[11][1]), "r"(a[11][2]), "r"(a[11][3]), "r"(a[12][0]), "r"(a[12][1]), "r"(a[12][2]), "r"(a[12][3]), "r"(a[13][0]), "r"(a[13][1]), "r"(a[13][2]), "r"(a[13][3]), "r"(a[14][0]), "r"(a[14][1]), "r"(a[14][2]), "r"(a[14][3]), "r"(a[15][0]), "r"(a[15][1]), "r"(a[15][2]), "r"(a[15][3]),
          "l"(dB + (uint64_t)0), "l"(dB + (uint64_t)2), "l"(dB + (uint64_t)4), "l"(dB + (uint64_t)6), "l"(dB + (uint64_t)512), "l"(dB + (uint64_t)514), "l"(dB + (uint64_t)516), "l"(dB + (uint64_t)518), "l"(dB + (uint64_t)1024), "l"(dB + (uint64_t)1026), "l"(dB + (uint64_t)1028), "l"(dB + (uint64_t)1030), "l"(dB + (uint64_t)1536), "l"(dB + (uint64_t)1538), "l"(dB + (uint64_t)1540), "l"(dB + (uint64_t)1542), "r"(0));
}
// chunk MMA, A from registers (RS): wgmma.fence; 16 x wgmma m64n16k16 (K ascending, first overwrites D); commit — one opaque block; the 16 B descriptors
// are warp-uniform expressions of dB (ptxas keeps them in uniform registers)
__device__ __forceinline__ void mma_chunk_n16(float (&d)[32], const uint32_t (&a)[16][4], uint64_t dB) {
    asm volatile(
        "{\n .reg .pred pf, pt;\n"
        " setp.ne.b32 pf, %88, 0;\n setp.eq.b32 pt, %88, 0;\n"
        " wgmma.fence.sync.aligned;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %72, pf, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%12,%13,%14,%15}, %73, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17,%18,%19}, %74, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%20,%21,%22,%23}, %75, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%24,%25,%26,%27}, %76, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%28,%29,%30,%31}, %77, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%32,%33,%34,%35}, %78, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%36,%37,%38,%39}, %79, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%40,%41,%42,%43}, %80, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%44,%45,%46,%47}, %81, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%48,%49,%50,%51}, %82, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%52,%53,%54,%55}, %83, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%56,%57,%58,%59}, %84, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%60,%61,%62,%63}, %85, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%64,%65,%66,%67}, %86, pt, 1, 1, 0;\n"
        " wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%68,%69,%70,%71}, %87, pt, 1, 1, 0;\n"
        " wgmma.commit_group.sync.aligned;\n}"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a[0][0]), "r"(a[0][1]), "r"(a[0][2]), "r"(a[0][3]), "r"(a[1][0]), "r"(a[1][1]), "r"(a[1][2]), "r"(a[1][3]), "r"(a[2][0]), "r"(a[2][1]), "r"(a[2][2]), "r"(a[2][3]), "r"(a[3][0]), "r"(a[3][1]), "r"(a[3][2]), "r"(a[3][3]), "r"(a[4][0]), "r"(a[4][1]), "r"(a[4][2]), "r"(a[4][3]), "r"(a[5][0]), "r"(a[5][1]), "r"(a[5][2]), "r"(a[5][3]), "r"(a[6][0]), "r"(a[6][1]), "r"(a[6][2]), "r"(a[6][3]), "r"(a[7][0]), "r"(a[7][1]), "r"(a[7][2]), "r"(a[7][3]), "r"(a[8][0]), "r"(a[8][1]), "r"(a[8][2]), "r"(a[8][3]), "r"(a[9][0]), "r"(a[9][1]), "r"(a[9][2]), "r"(a[9][3]), "r"(a[10][0]), "r"(a[10][1]), "r"(a[10][2]), "r"(a[10][3]), "r"(a[11][0]), "r"(a[11][1]), "r"(a[11][2]), "r"(a[11][3]), "r"(a[12][0]), "r"(a[12][1]), "r"(a[12][2]), "r"(a[12][3]), "r"(a[13][0]), "r"(a[13][1]), "r"(a[13][2]), "r"(a[13][3]), "r"(a[14][0]), "r"(a[14][1]), "r"(a[14][2]), "r"(a[14][3]), "r"(a[15][0]), "r"(a[15][1]), "r"(a[15][2]), "r"(a[15][3]),
          "l"(dB + (uint64_t)0), "l"(dB + (uint64_t)2), "l"(dB + (uint64_t)4), "l"(dB + (uint64_t)6), "l"(dB + (uint64_t)128), "l"(dB + (uint64_t)130), "l"(dB + (uint64_t)132), "l"(dB + (uint64_t)134), "l"(dB + (uint64_t)256), "l"(dB + (uint64_t)258), "l"(dB + (uint64_t)260), "l"(dB + (uint64_t)262), "l"(dB + (uint64_t)384), "l"(dB + (uint64_t)386), "l"(dB + (uint64_t)388), "l"(dB + (uint64_t)390), "r"(0));
}

// ------------------------------------------------------------------ LN arithmetic: op-for-op transcription of the F1 cell's emulation of LayerNormForwardV2<bf16,float4> (-O3 --use_fast_math SASS)
__device__ __forceinline__ float rcp_fast(float a) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float rsqrt_fast(float a) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float mul_rn(float a, float b) { float r; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float add_rn(float a, float b) { float r; asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float sub_rn(float a, float b) { float r; asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fma_rn(float a, float b, float c) { float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float max_f(float a, float b) { float r; asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }

__device__ __forceinline__ void welford_merge(float& mean, float& m2, float ca, float r_n, int mask) {
    float b_mean = __shfl_xor_sync(0xffffffffu, mean, mask);
    float b_m2 = __shfl_xor_sync(0xffffffffu, m2, mask);
    float delta = sub_rn(b_mean, mean);
    float t = mul_rn(delta, delta);
    t = mul_rn(ca, t);
    float nb_n = mul_rn(ca, r_n);
    float s = fma_rn(nb_n, t, b_m2);
    mean = fma_rn(nb_n, delta, mean);
    m2 = add_rn(m2, s);
}

// ------------------------------------------------------------------ kernel params
struct Params {
    const __nv_bfloat16* Z; long long s_zi, s_zj;       // element strides of the row frame (i, j); channel stride 1
    int NI, NJ;
    const __nv_bfloat16* lnw; const __nv_bfloat16* lnb; float eps;
    float* bias; long long NIP, NJP;                    // bias[(h*NIP + i)*NJP + j]
    int n_tiles, n_tj;
};

// epilogue of chunk c from accumulator set `a` (runs while the next chunk's MMAs execute)
__device__ __forceinline__ void epilogue_chunk(float (&a)[32], int c, int wg, int w4, int frow, int fcol2, bool elected, uint32_t stg_base,
                                               const CUtensorMap* tmQ, const CUtensorMap* tmK, const CUtensorMap* tmV, const CUtensorMap* tmG,
                                               int i0, int j0, const Params& P) {
    fence_acc(a);
    const int lane = frow * 4 + fcol2;
    if (c < NCH_QKVG) {
        const uint32_t stg = stg_base + (c % NSTG) * STG_BYTES; // free: the store that last read it (chunk c-2) completed before chunk c-1's barrier (wait_read<0> below)
        // accumulator fragment (per warp 16 j-rows x 64 cols) -> bf16 -> 4 x stmatrix.x4: matrices m = {(rows 0-7, cb), (rows 8-15, cb), (rows 0-7, cb+1), (rows 8-15, cb+1)},
        // cb = 8-col block; lane l addresses row (l&7) of matrix (l>>3).  Staging layouts (TMA boxes): q/k/v [hloc][i=w4][j][32 d] 64-B rows SWIZZLE_64B
        // (16-B unit ^= (R>>1)&3); g [i=w4][j][64 c] 128-B rows SWIZZLE_128B (unit ^= R&7).
        const int mrow = lane & 7, msel = lane >> 3;                      // this lane's address role: row within matrix, matrix index 0..3
        const int jr = (msel & 1) * 8 + mrow;                             // j row of the addressed matrix
        #pragma unroll
        for (int gq = 0; gq < 4; ++gq) {                                  // col blocks cb = 2*gq, 2*gq+1
            const int cb = 2 * gq + (msel >> 1);                          // col block this lane addresses
            uint32_t addr;
            if (c < 12) { const int R = ((cb >> 2) * 4 + w4) * 16 + jr; addr = stg + R * 64 + (((cb & 3) ^ ((R >> 1) & 3)) << 4); }
            else        { const int R = w4 * 16 + jr;                    addr = stg + R * 128 + ((cb ^ (R & 7)) << 4); }
            stsm_x4(addr, pack_bf16(a[8 * gq + 0], a[8 * gq + 1]), pack_bf16(a[8 * gq + 2], a[8 * gq + 3]),
                          pack_bf16(a[8 * gq + 4], a[8 * gq + 5]), pack_bf16(a[8 * gq + 6], a[8 * gq + 7]));
        }
        fence_proxy_async();                              // this thread's staging writes -> visible to the TMA (async proxy)
        if (elected) bulk_wait_read<0>();                 // every earlier store has finished reading its staging buffer (chunk c+1 reuses chunk c-1's)
        named_bar_sync(1 + wg, 128);
        if (elected) {
            if (c < 12) {
                const int which = c >> 2, h0 = (c & 3) * 2;
                const CUtensorMap* tm = which == 0 ? tmQ : (which == 1 ? tmK : tmV);
                tma_store_4d(tm, stg, 0, j0, i0, h0);          // dims (d, j, i, h)
            } else {
                tma_store_3d(tmG, stg, (c - 12) * 64, j0, i0);  // dims (col, j, i)
            }
            bulk_commit();
        }
    } else {
        // bias chunk: n16 accumulator, fragment group 0 = heads 2*fcol2 + {0,1}; value = fp32(bf16(acc)); bias[(h*NIP + i)*NJP + j]
        const int i = i0 + w4;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int j = j0 + frow + 8 * half;
            if (i < P.NI && j < P.NJ) {
                #pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int h = 2 * fcol2 + e;
                    const float v = __bfloat162float(__float2bfloat16_rn(a[2 * half + e]));
                    P.bias[((long long)h * P.NIP + i) * P.NJP + j] = v;
                }
            }
        }
    }
}

__global__ void __launch_bounds__(THREADS, 1)
f1v2_kernel(const __grid_constant__ CUtensorMap tmW, const __grid_constant__ CUtensorMap tmWB,
            const __grid_constant__ CUtensorMap tmQ, const __grid_constant__ CUtensorMap tmK, const __grid_constant__ CUtensorMap tmV,
            const __grid_constant__ CUtensorMap tmG, const __grid_constant__ Params P)
{
    extern __shared__ __align__(1024) uint8_t smem[];
    const uint32_t sbase = smem_u32(smem);
    const uint32_t sA = sbase + OFF_A, sW = sbase + OFF_W, sSTG = sbase + OFF_STG, sCP = sbase + OFF_CP, sBAR = sbase + OFF_BAR;
    const int tid = threadIdx.x, lane = tid & 31;
    const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);          // provably warp-uniform
    const int wg = warp >> 2;                            // 0,1 consumers; 2 producers; warp 12 = W issuer
    const int my_tiles = ((int)blockIdx.x < P.n_tiles) ? (P.n_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;

    if (tid == 0) {
        mbar_init(sBAR + BAR_AFULL, 4);                  // A_full: 4 producer warps arrive
        mbar_init(sBAR + BAR_AEMPTY, 8);                 // A_empty: 8 consumer warps arrive (after copying their rows to registers)
        #pragma unroll
        for (int s = 0; s < WSLOTS; ++s) { mbar_init(sBAR + BAR_WFULL + 8 * s, 1); mbar_init(sBAR + BAR_WEMPTY + 8 * s, 2); }   // W_full: issuer expect_tx; W_empty: 2 WGs
        fence_barrier_init();
    }
    __syncthreads();

    if (warp == 12) {
        // =========================================================== W ISSUER: stream every chunk's weights through the 4-slot ring, as far ahead as W_empty allows
        if (lane == 0) {
            const uint32_t total_chunks = (uint32_t)my_tiles * NCH;
            for (uint32_t g = 0; g < total_chunks; ++g) {
                const uint32_t slot = g % WSLOTS, u = g / WSLOTS;
                mbar_wait(sBAR + BAR_WEMPTY + 8 * slot, (u & 1) ^ 1);   // both WGs released the slot's previous chunk
                const uint32_t dst = sW + slot * WSLOT_BYTES, fb = sBAR + BAR_WFULL + 8 * slot;
                const int cn = (int)(g % NCH);
                if (cn < NCH_QKVG) {
                    mbar_arrive_expect_tx(fb, 64 * C * 2);
                    #pragma unroll
                    for (int kc = 0; kc < 4; ++kc) tma_load_2d(dst + kc * 8192, &tmW, kc * 64, cn * 64, fb);
                } else {
                    mbar_arrive_expect_tx(fb, HB * C * 2);
                    #pragma unroll
                    for (int kc = 0; kc < 4; ++kc) tma_load_2d(dst + kc * 2048, &tmWB, kc * 64, 0, fb);
                }
            }
        }
        return;
    }
    if (wg == 2) {
        // =========================================================== PRODUCERS: each warp streams its rows (32 per tile, flattened across tiles) through a CP_ROWS-deep cp.async ring
        // (lane l copies bytes 16l..16l+15 of a row = channels 8l..8l+7; rows outside [NI, NJ) are zero-filled), normalises them (LayerNormForwardV2 arithmetic) and writes
        // the bf16 result into the swizzled A tile
        const int pw = warp - 8;                         // 0..3 -> tile rows [32*pw, 32*pw+32)
        const float r1 = rcp_fast(1.f), r2 = rcp_fast(2.f), r3 = rcp_fast(3.f), r4 = rcp_fast(4.f), r5 = rcp_fast(5.f), r6 = rcp_fast(6.f), r7 = rcp_fast(7.f), r8 = rcp_fast(8.f);
        const float rr[8] = {r1, r2, r3, r4, r5, r6, r7, r8};
        const float rns[5] = {rcp_fast(16.f), rcp_fast(32.f), rcp_fast(64.f), rcp_fast(128.f), rcp_fast(256.f)};
        const float rC = rcp_fast(256.f);
        float wv[8], bv[8];
        {
            uint4 wq = *reinterpret_cast<const uint4*>(P.lnw + 8 * lane), bq = *reinterpret_cast<const uint4*>(P.lnb + 8 * lane);
            const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&wq); const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bq);
            #pragma unroll
            for (int e = 0; e < 8; ++e) { wv[e] = __bfloat162float(wp[e]); bv[e] = __bfloat162float(bp[e]); }
        }
        const int kc = lane >> 3, unit = lane & 7;
        const uint32_t abase = sA + kc * 16384;
        const uint32_t cpbase = sCP + pw * CP_BYTES + lane * 16;
        constexpr int PR = 2;                      // rows normalised together (independent dependency chains interleaved statement by statement)
        static_assert(32 % PR == 0 && CP_ROWS % PR == 0 && CP_ROWS > PR, "ring/PR");
        const int nrows = my_tiles * 32;                 // flattened row sequence R -> tile lt = R/32, tile row r = pw*32 + R%32 (i-row r>>4, j = r&15)
        // per-tile source bases: this warp's 32 rows are 2 i-rows x 16 j; row R's 16 B for this lane = base(i-row) + j*s_zj*2 + 16*lane
        const long long sj2 = P.s_zj * 2;
        auto row_src = [&](int R, uint32_t& nbytes) -> const char* {
            const int lt = R >> 5, r = pw * 32 + (R & 31);
            const int tile = (int)blockIdx.x + lt * (int)gridDim.x;
            const int ti = tile / P.n_tj, tj = tile - ti * P.n_tj;
            const int i = ti * BI + (r >> 4), j = tj * BJ + (r & 15);
            const bool valid = (R < nrows) && (i < P.NI) && (j < P.NJ);
            nbytes = valid ? 16u : 0u;
            return valid ? (reinterpret_cast<const char*>(P.Z) + ((long long)i * P.s_zi) * 2 + (long long)j * sj2 + 16 * lane) : reinterpret_cast<const char*>(P.Z);
        };
        auto issue_rows = [&](int R0) {                 // cp.async rows R0..R0+PR-1 (one commit group); rows past the end copy 0 bytes (zero-fill) so group counting stays uniform
            #pragma unroll
            for (int u = 0; u < PR; ++u) {
                uint32_t nb; const char* src = row_src(R0 + u, nb);
                cp_async_16(cpbase + ((R0 + u) % CP_ROWS) * (C * 2), src, nb);
            }
            cp_async_commit();
        };
        constexpr int AHEAD = CP_ROWS / PR - 1;          // groups in flight ahead of the one being consumed
        #pragma unroll
        for (int g = 0; g < AHEAD; ++g) issue_rows(g * PR);
        #pragma unroll 1
        for (int R0 = 0; R0 < nrows; R0 += PR) {
            const int lt = R0 >> 5, q0 = R0 & 31;
            issue_rows(R0 + AHEAD * PR);
 
            cp_async_wait<AHEAD>();                      // the group holding rows R0..R0+PR-1 has landed (this lane's own bytes)
            uint4 cur[PR];
            #pragma unroll
            for (int u = 0; u < PR; ++u)
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(cur[u].x), "=r"(cur[u].y), "=r"(cur[u].z), "=r"(cur[u].w) : "r"(cpbase + ((R0 + u) % CP_ROWS) * (C * 2)) : "memory");
 
            if (q0 == 0) mbar_wait(sBAR + BAR_AEMPTY, (lt & 1) ^ 1);   // consumers have copied the previous tile out of A
 
            float x[PR][8], mean[PR], m2[PR];
            #pragma unroll
            for (int u = 0; u < PR; ++u) {
                const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&cur[u]);
                #pragma unroll
                for (int e = 0; e < 8; ++e) x[u][e] = __bfloat162float(xp[e]);
                mean[u] = 0.f; m2[u] = 0.f;
            }
            #pragma unroll
            for (int e = 0; e < 8; ++e) {                                // per-lane sequential Welford, counts 1..8 (PR rows interleaved)
                #pragma unroll
                for (int u = 0; u < PR; ++u) {
                    float d1 = sub_rn(x[u][e], mean[u]);
                    mean[u] = fma_rn(d1, rr[e], mean[u]);
                    float d2 = sub_rn(x[u][e], mean[u]);
                    m2[u] = fma_rn(d1, d2, m2[u]);
                }
            }
            #pragma unroll
            for (int st = 0; st < 5; ++st) {                             // xor-butterfly all-reduce (masks 16,8,4,2,1; counts 8..128)
                const int mask = 16 >> st;
                const float ca = (float)(8 << st), r_n = rns[st];
                float bm[PR], b2[PR];
                #pragma unroll
                for (int u = 0; u < PR; ++u) { bm[u] = __shfl_xor_sync(0xffffffffu, mean[u], mask); b2[u] = __shfl_xor_sync(0xffffffffu, m2[u], mask); }
                #pragma unroll
                for (int u = 0; u < PR; ++u) {
                    float delta = sub_rn(bm[u], mean[u]);
                    float t = mul_rn(delta, delta);
                    t = mul_rn(ca, t);
                    float nb_n = mul_rn(ca, r_n);
                    float sm = fma_rn(nb_n, t, b2[u]);
                    mean[u] = fma_rn(nb_n, delta, mean[u]);
                    m2[u] = add_rn(m2[u], sm);
                }
            }
            #pragma unroll
            for (int u = 0; u < PR; ++u) {
                const int r = pw * 32 + q0 + u;
                float var = mul_rn(rC, m2[u]);
                var = max_f(var, 0.f);
                const float inv = rsqrt_fast(add_rn(var, P.eps));
                uint32_t packed[4];
                #pragma unroll
                for (int e = 0; e < 8; e += 2) {
                    const float y0 = fma_rn(mul_rn(inv, sub_rn(x[u][e], mean[u])), wv[e], bv[e]);
                    const float y1 = fma_rn(mul_rn(inv, sub_rn(x[u][e + 1], mean[u])), wv[e + 1], bv[e + 1]);
                    packed[e >> 1] = pack_bf16(y0, y1);
                }
                asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(abase + r * 128 + ((unit ^ (r & 7)) << 4)), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
            }
 
            if (q0 + PR == 32) { __syncwarp(); if (lane == 0) mbar_arrive(sBAR + BAR_AFULL); }   // A_full (4 arrivals; release orders the st.shared above)
        }
        cp_async_wait<0>();
        return;
    }

    // =============================================================== CONSUMERS (2 warpgroups)
    const int wtid = tid & 127;
    const int w4 = warp & 3;                             // warp within WG -> accumulator rows 16*w4.. == tile i-row offset within the WG's 4 rows
    const int frow = lane >> 2, fcol2 = lane & 3;        // fragment row (j = frow, frow+8) and column-pair index
    const bool elected = (wtid == 0);                    // per-WG TMA-store issuer / barrier arriver
    const uint32_t stg_base = sSTG + wg * NSTG * STG_BYTES;
    // ldmatrix source addresses for this thread's A fragments: row within the tile, 16-B unit (before swizzle) = 2*(ks&3) + lane>>4, block kc = ks>>2
    const int arow = wg * 64 + w4 * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
    const uint32_t a_rowbase = sA + arow * 128;
    const int a_uhi = lane >> 4, a_sw = arow & 7;

    auto waitB = [&](uint32_t g) {                       // wait W_full for global chunk g, return its B descriptor
        const uint32_t slot = g % WSLOTS, u = g / WSLOTS;
        mbar_wait(sBAR + BAR_WFULL + 8 * slot, u & 1);
        return make_desc(sW + slot * WSLOT_BYTES);
    };
    // retire global chunk g (its MMAs are complete): release its W slot (the issuer warp refills it)
    auto retire = [&](uint32_t g) {
        if (elected) mbar_arrive(sBAR + BAR_WEMPTY + 8 * (g % WSLOTS));
    };
    uint32_t afr[16][4];                                 // this thread's A fragments for the whole tile (16 k-steps x 4 regs)
    float acc0[32], accB[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) { acc0[i] = 0.f; accB[i] = 0.f; }
    uint32_t gc = 0;                                     // global chunk counter (chunk 0 of the current tile)
    int lt = 0;
    for (int tile = blockIdx.x; tile < P.n_tiles; tile += gridDim.x, ++lt, gc += NCH) {
        const int ti = tile / P.n_tj, tj = tile - ti * P.n_tj;
        const int i0 = ti * BI + wg * 4, j0 = tj * BJ;                   // this WG's output box origin (4 i-rows x 16 j)
        mbar_wait(sBAR + BAR_AFULL, lt & 1);   // producers finished this tile's A
 
        #pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            const int u = 2 * (ks & 3) + a_uhi;
            ldsm_x4(afr[ks], a_rowbase + (ks >> 2) * 16384 + ((u ^ a_sw) << 4));
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(sBAR + BAR_AEMPTY);                    // this warp's rows are in registers: producers may overwrite A
 

        // per-WG serial pipeline (chain -> wait 0 -> epilogue); tensor-core overlap comes from the two WGs alternating (ping-pong), not from reading an older
        // accumulator while a newer chain is in flight (ptxas 13.0 serializes every HGMMA in that pattern)
        #pragma unroll 1
        for (int c = 0; c < NCH_QKVG; ++c) {
 const uint64_t dB = waitB(gc + c); 
 mma_chunk_n64(acc0, afr, dB); 
 wgmma_wait<0>(); fence_acc(acc0); retire(gc + c); 
 epilogue_chunk(acc0, c, wg, w4, frow, fcol2, elected, stg_base, &tmQ, &tmK, &tmV, &tmG, i0, j0, P); 
        }
        {   const uint64_t dB = waitB(gc + 16);
            mma_chunk_n16(accB, afr, dB);
            wgmma_wait<0>(); fence_acc(accB); retire(gc + 16);
            epilogue_chunk(accB, 16, wg, w4, frow, fcol2, elected, stg_base, &tmQ, &tmK, &tmV, &tmG, i0, j0, P);
        }
    }
    if (elected) bulk_wait<0>();                                          // all TMA stores complete before smem is released
}

}  // namespace f1v2

// ====================================================================================================================== host
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
                                  CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn g_encode = nullptr;
static int get_encode() {
    if (g_encode) return 0;
    cudaDriverEntryPointQueryResult qr;
    void* fn = nullptr;
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &qr);
    if (e != cudaSuccess || fn == nullptr || qr != cudaDriverEntryPointSuccess) return -1;
    g_encode = reinterpret_cast<EncodeTiledFn>(fn);
    return 0;
}
static int encode(CUtensorMap* tm, CUtensorMapDataType dt, int rank, void* base, const cuuint64_t* dims, const cuuint64_t* strides_bytes,
                  const cuuint32_t* box, CUtensorMapSwizzle sw) {
    cuuint32_t estr[5] = {1, 1, 1, 1, 1};
    CUresult r = g_encode(tm, dt, rank, base, dims, strides_bytes, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return r == CUDA_SUCCESS ? 0 : (int)r;
}

extern "C" {

// Returns 0 on success; 1000*k + CUresult for tensor-map k encoding failures; negative = CUDA runtime error.
int f1v2_launch(const void* Z, long long s_zi, long long s_zj, int NI, int NJ,
                const void* lnw, const void* lnb, float eps,
                const void* Wqkvg, const void* Wb,
                void* Q, void* K, void* V, long long q_si, long long q_sh,      // element strides of q/k/v along i and h (j stride = D, d stride = 1); valid extents NI/NJ
                void* G, long long g_si,                                        // g element (i, j, c) at i*g_si + j*HD + c
                float* bias, long long NIP, long long NJP,
                void* stream_ptr, int grid_ctas)
{
    using namespace f1v2;
    if (get_encode() != 0) return -1;
    static bool attr_done = false;
    if (!attr_done) {
        if (cudaFuncSetAttribute(f1v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES) != cudaSuccess) return -2;
        attr_done = true;
    }
    CUtensorMap tmW, tmWB, tmQ, tmK, tmV, tmG;
    {   // W [1024 n][256 k] bf16, K inner; box {64 k, 64 n}, swizzle 128B
        cuuint64_t dims[2] = {(cuuint64_t)C, (cuuint64_t)(4 * HD)}; cuuint64_t str[1] = {(cuuint64_t)C * 2}; cuuint32_t box[2] = {64, 64};
        int r = encode(&tmW, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(Wqkvg), dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B); if (r) return 1000 + r;
    }
    {   // Wb [16 n][256 k]
        cuuint64_t dims[2] = {(cuuint64_t)C, (cuuint64_t)HB}; cuuint64_t str[1] = {(cuuint64_t)C * 2}; cuuint32_t box[2] = {64, (cuuint32_t)HB};
        int r = encode(&tmWB, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(Wb), dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B); if (r) return 2000 + r;
    }
    {   // q/k/v: dims (d=32, j=NJ, i=NI, h=H), element strides (1, D, q_si, q_sh); box {32, 16, 4, 2}; swizzle 64B (64-B inner rows)
        cuuint64_t dims[4] = {(cuuint64_t)D, (cuuint64_t)NJ, (cuuint64_t)NI, (cuuint64_t)H};
        cuuint64_t str[3] = {(cuuint64_t)D * 2, (cuuint64_t)q_si * 2, (cuuint64_t)q_sh * 2};
        cuuint32_t box[4] = {(cuuint32_t)D, (cuuint32_t)BJ, 4, 2};
        int r;
        r = encode(&tmQ, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, Q, dims, str, box, CU_TENSOR_MAP_SWIZZLE_64B); if (r) return 3000 + r;
        r = encode(&tmK, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, K, dims, str, box, CU_TENSOR_MAP_SWIZZLE_64B); if (r) return 4000 + r;
        r = encode(&tmV, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, V, dims, str, box, CU_TENSOR_MAP_SWIZZLE_64B); if (r) return 5000 + r;
    }
    {   // g: dims (c=256, j=NJ, i=NI), strides (1, HD, g_si); box {64, 16, 4}; swizzle 128B
        cuuint64_t dims[3] = {(cuuint64_t)HD, (cuuint64_t)NJ, (cuuint64_t)NI};
        cuuint64_t str[2] = {(cuuint64_t)HD * 2, (cuuint64_t)g_si * 2};
        cuuint32_t box[3] = {64, (cuuint32_t)BJ, 4};
        int r = encode(&tmG, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, G, dims, str, box, CU_TENSOR_MAP_SWIZZLE_128B); if (r) return 6000 + r;
    }
    Params P;
    P.Z = reinterpret_cast<const __nv_bfloat16*>(Z); P.s_zi = s_zi; P.s_zj = s_zj; P.NI = NI; P.NJ = NJ;
    P.lnw = reinterpret_cast<const __nv_bfloat16*>(lnw); P.lnb = reinterpret_cast<const __nv_bfloat16*>(lnb); P.eps = eps;
    P.bias = bias; P.NIP = NIP; P.NJP = NJP;    const int n_ti = (NI + BI - 1) / BI, n_tj = (NJ + BJ - 1) / BJ;
    P.n_tiles = n_ti * n_tj; P.n_tj = n_tj;
    int grid = grid_ctas < P.n_tiles ? grid_ctas : P.n_tiles;
    if (grid < 1) grid = 1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    f1v2_kernel<<<grid, THREADS, SMEM_BYTES, stream>>>(tmW, tmWB, tmQ, tmK, tmV, tmG, P);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : -(int)e;
}

int f1v2_smem_bytes() { return f1v2::SMEM_BYTES; }

}  // extern "C"
