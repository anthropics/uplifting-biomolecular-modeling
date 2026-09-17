// triattn_v3.cu — triangle attention forward, speed build v3. sm_80 code path (mma.sync/ldmatrix/cp.async), runs on sm_80/sm_90.
// The arithmetic is fixed (bit-identical output contract); only the schedule differs between builds/variants.
// from the exploration build triattn_fwd.cu:
//   * CTA = NWARPS x 16 query rows of one (b, h) and a GROUP of G consecutive pair-rows n (n0..n0+G-1): the bias tile
//     [rows x 64 keys] is staged in shared memory ONCE per key tile and reused for all G n (bias L2 traffic / G);
//   * K/V tiles of the G n's + the bias tile stream through a 2-stage cp.async pipeline;
//   * masks arrive as 64-bit words per (n, tile) via ballots; key tiles that are entirely masked are skipped exactly
//     (P would be +0, alpha 1) once the running max is above the mask logit;
//   * per element: fma.ftz(dot,s,b) -> mul.ftz(log2e) -> max -> sub.ftz -> ex2.approx.ftz -> add.ftz (row sum) -> cvt bf16x2.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace {

constexpr float LOG2E_F = 1.4426950408889634f;
constexpr float MASK_X2 = -1442695040.0f;          // fl32(-1e9 * LOG2E_F): masked / out-of-range key logit in the log2 domain
constexpr float NEG_INF = -__builtin_huge_valf();

struct PV3 {
    const __nv_bfloat16* q; const __nv_bfloat16* k; const __nv_bfloat16* v;
    const void* bias; const uint8_t* mask; __nv_bfloat16* out;
    int B, N, H, S, n_groups, bias_fast;
    int q0, q_rows;                                               // query rows served by this launch: [q0, q0 + q_rows) (tail split)
    const uint8_t* vbad;                                          // optional [B*N*H*ntiles] bytes from vscan_kernel: 1 = all-masked tile with NaN/Inf in V
    long long sqB, sqN, sqH, sqS, skB, skN, skH, skS, svB, svN, svH, svS, sbB, sbH, sbQ, sbK, smB, smN;
    float scale;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async_16(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N_> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_) : "memory"); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float add_ftz(float a, float b) { float y; asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float sub_ftz(float a, float b) { float y; asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float mul_ftz(float a, float b) { float y; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fma_ftz(float a, float b, float c) { float y; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c)); return y; }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&t);
}
// K/V smem tile: 64 rows (keys) x 64 bytes, 16B chunks XOR-swizzled
__device__ __forceinline__ int swz_off(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }


// v2 schedule: CTA = NWR row-warps (16 query rows each) x NSPLIT n-groups; every warp serves GW pair-rows; G = NSPLIT*GW pair-rows
// share the CTA's bias tile in shared memory. <=128 registers/thread so that two 256-thread CTAs (16 warps) fit per SM.
// ---- V finiteness pre-pass: out[((b*N+n)*H+h)*ntiles + tile] = 1 iff key tile `tile` is fully masked for pair-row (b,n) AND its V tile
//      (keys < S) holds a NaN/Inf.  Such tiles must take the normal (non-skipped) path so that 0*NaN = NaN propagates exactly as in
//      the library.  One read of the masked part of V per call instead of one per q-block CTA.
__global__ void __launch_bounds__(128) vscan_kernel(const __nv_bfloat16* __restrict__ v, long long svB, long long svN, long long svH, long long svS,
                                                    const uint8_t* __restrict__ mask, long long smB, long long smN, int N, int H, int S,
                                                    uint8_t* __restrict__ out) {
    const int bn = blockIdx.x, b = bn / N, n = bn % N, h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ntiles = (S + 63) >> 6;
    const uint8_t* mr = mask + b * smB + (long long)n * smN;
    const __nv_bfloat16* vt = v + b * svB + (long long)n * svN + h * svH;
    for (int tile = warp; tile < ntiles; tile += 4) {
        const int k_lo = tile * 64 + lane, k_hi = k_lo + 32;
        const bool val = ((k_lo < S) && mr[k_lo < S ? k_lo : 0] != 0) || ((k_hi < S) && mr[k_hi < S ? k_hi : 0] != 0);
        uint8_t res = 0;
        if (__any_sync(0xffffffffu, val) == 0) {
            uint32_t acc = 0u;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int key = half ? k_hi : k_lo;
                if (key < S) {
                    const uint4* src = reinterpret_cast<const uint4*>(vt + (long long)key * svS);
#pragma unroll
                    for (int c4 = 0; c4 < 4; ++c4) {
                        const uint4 w = __ldg(src + c4);
                        acc |= ((w.x & 0x7F807F80u) + 0x00800080u) & 0x80008000u;
                        acc |= ((w.y & 0x7F807F80u) + 0x00800080u) & 0x80008000u;
                        acc |= ((w.z & 0x7F807F80u) + 0x00800080u) & 0x80008000u;
                        acc |= ((w.w & 0x7F807F80u) + 0x00800080u) & 0x80008000u;
                    }
                }
            }
            res = __any_sync(0xffffffffu, acc != 0u) ? 1 : 0;
        }
        if (lane == 0) out[((long long)bn * H + h) * ntiles + tile] = res;
    }
}

#ifndef TRIATTN_QSMEM
#define TRIATTN_QSMEM 0           // 1: keep the CTA's Q tile in shared memory and ldmatrix the A-fragments per key tile (frees 16 registers)
#endif
#ifndef TRIATTN_PERF
#define TRIATTN_PERF 0            // perf-attribution ONLY (results wrong): 1 no bias, 2 no ex2, 4 no row-sum, 8 no PV mma, 16 no QK mma, 32 no max-reduce, 64 no O rescale
#endif
#ifndef TRIATTN_MINB
#define TRIATTN_MINB(NW) (((NW) >= 16) ? 1 : (((NW) >= 8) ? 2 : 3))
#endif
#ifndef TRIATTN_SKIP
#define TRIATTN_SKIP 1            // 1: skip fully-masked key tiles (exactly); 0: always compute
#endif
#ifndef TRIATTN_LOADSKIP
#define TRIATTN_LOADSKIP 1        // 1: also skip the K/V loads of such tiles (CTA-wide decision)
#endif
template <int NWR, int NSPLIT, int GW, int NST, bool HAS_MASK, bool BIAS_BF16>
__global__ void __launch_bounds__(NWR * NSPLIT * 32, TRIATTN_MINB(NWR * NSPLIT)) triattn_v3_kernel(const PV3 p) {
    constexpr int G = NSPLIT * GW;
    constexpr int NT = NWR * NSPLIT * 32;
    constexpr int BM = NWR * 16;
    constexpr int BSTR = 72;
    constexpr int BES = BIAS_BF16 ? 2 : 4;
    constexpr int KVB = 64 * 64;
    constexpr int STAGE = G * 2 * KVB + BM * BSTR * BES;
    extern __shared__ __align__(128) uint8_t smem[];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wrow = warp % NWR, wn = warp / NWR;               // row group, n sub-group
    const int qblk = blockIdx.x, h = blockIdx.y;
    const int b = blockIdx.z / p.n_groups, ng = blockIdx.z % p.n_groups;
    const int n0 = ng * G;                                       // first pair-row of the CTA
    const int Gact = min(G, p.N - n0);
    const int S = p.S;
    const int qb0 = p.q0 + qblk * BM;
    const int lr0 = wrow * 16 + g, lr1 = lr0 + 8;               // local rows inside the CTA tile
    const int r0 = qb0 + lr0, r1 = qb0 + lr1;
    const float scale = p.scale;

    // ---- Q: either register fragments (default) or a swizzled smem tile [G][BM rows][64 B] read with ldmatrix per key tile ----
    constexpr int QSM = TRIATTN_QSMEM ? G * BM * 64 : 0;
    uint32_t qa[GW][2][4];
    if (!TRIATTN_QSMEM) {
#pragma unroll
        for (int i = 0; i < GW; ++i) {
            const int nn = wn * GW + i;
            const bool act = nn < Gact;
            const __nv_bfloat16* qb = p.q + b * p.sqB + (long long)(n0 + (act ? nn : 0)) * p.sqN + h * p.sqH;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                int d0 = ks * 16 + 2 * t;
                qa[i][ks][0] = (r0 < S && act) ? *reinterpret_cast<const uint32_t*>(qb + (long long)r0 * p.sqS + d0) : 0u;
                qa[i][ks][1] = (r1 < S && act) ? *reinterpret_cast<const uint32_t*>(qb + (long long)r1 * p.sqS + d0) : 0u;
                qa[i][ks][2] = (r0 < S && act) ? *reinterpret_cast<const uint32_t*>(qb + (long long)r0 * p.sqS + d0 + 8) : 0u;
                qa[i][ks][3] = (r1 < S && act) ? *reinterpret_cast<const uint32_t*>(qb + (long long)r1 * p.sqS + d0 + 8) : 0u;
            }
        }
    } else {
        // cooperative load: G*BM rows x 4 chunks of 16 B, XOR-swizzled like K (so ldmatrix rows are conflict-free)
        uint8_t* qs = smem;                                                            // Q tile lives at the front of dynamic smem
        for (int c = tid; c < G * BM * 4; c += NWR * NSPLIT * 32) {
            const int nn = c / (BM * 4), rem = c % (BM * 4), row = rem >> 2, ch = rem & 3;
            const int qrow = qb0 + row;
            const bool ok = (nn < Gact) && (qrow < S);
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (ok) val = *reinterpret_cast<const uint4*>(p.q + b * p.sqB + (long long)(n0 + nn) * p.sqN + h * p.sqH + (long long)qrow * p.sqS + ch * 8);
            *reinterpret_cast<uint4*>(qs + nn * BM * 64 + swz_off(row, ch)) = val;
        }
    }
    const long long bias_bh = b * p.sbB + h * p.sbH;
    const int ntiles = (S + 63) >> 6;

    // lane-constant ldmatrix offsets (the XOR swizzle term ((row>>1)&3) reduces to ((lane&7)>>1)&3 for our access pattern)
    const int lr = lane & 7, lmi = lane >> 3, swz = (lr >> 1) & 3;
    const int koff = lr * 64 + ((lmi ^ swz) << 4);                                   // + j*512
    const int voff0 = (lmi & 1) * 512 + lr * 64 + (((0 + (lmi >> 1)) ^ swz) << 4);   // + ks*1024  (d chunks 0,1)
    const int voff2 = (lmi & 1) * 512 + lr * 64 + (((2 + (lmi >> 1)) ^ swz) << 4);   // + ks*1024  (d chunks 2,3)
    const int boff0 = (lr0 * BSTR + 2 * t) * BES, boff1 = (lr1 * BSTR + 2 * t) * BES;

    // ---- loader addressing, hoisted: chunk i of this thread is c = tid + i*NT -> (nn, which, row, ch) ----
    constexpr int KVC = (G * 512) / NT;                               // 16-byte K/V chunks per thread per tile
    static_assert((G * 512) % NT == 0, "chunk mapping");
    const __nv_bfloat16* kthr = p.k + b * p.skB + h * p.skH;          // + nn*skN + key*skS + ch*8 added per chunk
    const __nv_bfloat16* vthr = p.v + b * p.svB + h * p.svH;
    constexpr int EPC = 16 / BES, CPR = 64 / EPC;                     // bias fast path: elements per 16B chunk, chunks per row
    constexpr int BCH = (BM * CPR + NT - 1) / NT;                     // bias chunks per thread per tile
    const uint8_t* bthr = reinterpret_cast<const uint8_t*>(p.bias) + bias_bh * BES;

    auto load_stage = [&](int tile, int s, uint32_t nskip) {      // nskip bit nn set -> K/V of pair-row nn not needed for this tile
        uint8_t* base = smem + QSM + s * STAGE;
        const int key0 = tile * 64;
#pragma unroll
        for (int i = 0; i < KVC; ++i) {
            const int c = tid + i * NT;
            const int nn = c >> 9, which = (c >> 8) & 1, row = (c & 255) >> 2, ch = c & 3;
            const int key = key0 + row;
            const bool want = !((nskip >> nn) & 1u);
            const bool valid = want && (key < S);                     // nn >= Gact is always inside nskip
            const int keyc = valid ? key : 0;
            const __nv_bfloat16* src = (which == 0) ? (kthr + (long long)(n0 + nn) * p.skN + (long long)keyc * p.skS + ch * 8)
                                                    : (vthr + (long long)(n0 + nn) * p.svN + (long long)keyc * p.svS + ch * 8);
            if (want) cp_async_16(base + nn * 2 * KVB + which * KVB + swz_off(row, ch), (nn < Gact) ? (const void*)src : (const void*)p.k, valid ? 16 : 0);
        }
        uint8_t* bsm = base + G * 2 * KVB;
        if (nskip == (1u << G) - 1u) return;                          // nothing to compute in this tile for the whole CTA
        if (p.bias_fast) {
#pragma unroll
            for (int i = 0; i < BCH; ++i) {
                const int c = tid + i * NT;
                if ((BM * CPR) % NT != 0 && c >= BM * CPR) break;
                const int row = c / CPR, ch = c % CPR;
                const int qrow = qb0 + row, key = key0 + ch * EPC;
                int nbytes = 0;
                if (qrow < S && key < S) { int rem_el = S - key; nbytes = rem_el >= EPC ? 16 : rem_el * BES; }
                const uint8_t* src = bthr + ((long long)(qrow < S ? qrow : 0) * p.sbQ + (key < S ? key : 0)) * BES;
                cp_async_16(bsm + (row * BSTR + ch * EPC) * BES, src, nbytes);
            }
        } else {
            // generic in-place path (bias rows not 16B-aligned, e.g. odd S, or key-strided views such as the ENGINE permuted bias):
            // each thread copies RUN consecutive keys of one row with plain loads (no per-element index math).
            constexpr int RUN = (BM * 64) / NT;                           // 16 or 32 keys per thread
            static_assert(RUN >= 2 && (64 % RUN) == 0, "run mapping");
            const int row = tid / (64 / RUN), kseg = tid % (64 / RUN);
            const int qrow = qb0 + row, kfirst = key0 + kseg * RUN;
            int nval = S - kfirst; nval = nval < 0 ? 0 : (nval > RUN ? RUN : nval);
            if (qrow >= S) nval = 0;
            const long long sbK = p.sbK;
            const uint8_t* sp = bthr + ((long long)(qrow < S ? qrow : 0) * p.sbQ + (long long)(nval > 0 ? kfirst : 0) * sbK) * BES;
            uint8_t* dp = bsm + (row * BSTR + kseg * RUN) * BES;
            if (BIAS_BF16) {
                const unsigned short* s16 = reinterpret_cast<const unsigned short*>(sp);
                uint32_t w[RUN / 2];
                if (sbK == 1) {
#pragma unroll
                    for (int i = 0; i < RUN; i += 2) {
                        uint32_t lo = (i < nval) ? (uint32_t)s16[i] : 0u, hi = (i + 1 < nval) ? (uint32_t)s16[i + 1] : 0u;
                        w[i / 2] = lo | (hi << 16);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < RUN; i += 2) {
                        uint32_t lo = (i < nval) ? (uint32_t)s16[(long long)i * sbK] : 0u, hi = (i + 1 < nval) ? (uint32_t)s16[(long long)(i + 1) * sbK] : 0u;
                        w[i / 2] = lo | (hi << 16);
                    }
                }
#pragma unroll
                for (int i = 0; i < RUN / 2; i += 2) *reinterpret_cast<uint2*>(dp + i * 4) = make_uint2(w[i], w[i + 1]);
            } else {
                const float* s32 = reinterpret_cast<const float*>(sp);
                float w[RUN];
                if (sbK == 1) {
#pragma unroll
                    for (int i = 0; i < RUN; ++i) w[i] = (i < nval) ? s32[i] : 0.f;
                } else {
#pragma unroll
                    for (int i = 0; i < RUN; ++i) w[i] = (i < nval) ? s32[(long long)i * sbK] : 0.f;
                }
#pragma unroll
                for (int i = 0; i < RUN; i += 2) *reinterpret_cast<float2*>(dp + i * 4) = make_float2(w[i], w[i + 1]);
            }
        }
    };

    float m_run[GW][2], l_run[GW][2], o[GW][4][4];
#pragma unroll
    for (int i = 0; i < GW; ++i) {
        m_run[i][0] = m_run[i][1] = NEG_INF; l_run[i][0] = l_run[i][1] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j][0] = o[i][j][1] = o[i][j][2] = o[i][j][3] = 0.f;
    }
    // okw[n][w] = 1 once every row of row-warp w has a running max >= -1e9 for pair-row n (monotone). A key tile whose keys are
    // all masked for pair-row n contributes exactly nothing once that holds for all row-warps -> its K/V need not be loaded.
    __shared__ uint32_t okw[G];                                       // bit w: row-warp w has real running maxes for pair-row nn
    if (HAS_MASK) { if (tid < G) okw[tid] = 0u; }
    uint32_t okall = 0u;                                               // register cache of "all row-warps ok" per nn (monotone)
    uint32_t okmine = 0u;                                              // this warp's own published bits
    auto n_all_ok = [&](int nn) -> bool {
        if ((okall >> nn) & 1u) return true;
        const bool ok = okw[nn] == ((1u << NWR) - 1u);
        if (ok) okall |= 1u << nn;
        return ok;
    };
    // ---- mask words: one 64-bit word per (pair-row of this CTA, key tile), built once per CTA in shared memory ----
    uint32_t* mw = reinterpret_cast<uint32_t*>(smem + QSM + NST * STAGE);   // [G][ntiles][2]
    uint32_t* amt = mw + G * ntiles * 2;                               // [ntiles]: bit nn = tile has no valid key for pair-row nn AND may be skipped
    uint32_t* vbd = amt + ntiles;                                      // [ntiles]: bit nn = all-masked tile whose V tile holds a NaN/Inf -> never skip
    if (HAS_MASK) {
        for (int t_ = tid; t_ < ntiles; t_ += NT) { amt[t_] = 0u; vbd[t_] = 0u; }
        __syncthreads();
        for (int pi = warp; pi < Gact * ntiles; pi += NWR * NSPLIT) {
            const int nn = pi / ntiles, tile = pi % ntiles;
            const uint8_t* mr = p.mask + b * p.smB + (long long)(n0 + nn) * p.smN;
            const int k_lo = tile * 64 + lane, k_hi = k_lo + 32;
            const bool v_lo = (k_lo < S) && (mr[k_lo < S ? k_lo : 0] != 0);
            const bool v_hi = (k_hi < S) && (mr[k_hi < S ? k_hi : 0] != 0);
            const uint32_t wlo_ = __ballot_sync(0xffffffffu, v_lo), whi_ = __ballot_sync(0xffffffffu, v_hi);
            bool vbad = false;
            if (TRIATTN_SKIP && (wlo_ | whi_) == 0u && p.vbad != nullptr) {
                vbad = p.vbad[(((long long)b * p.N + (n0 + nn)) * p.H + h) * ntiles + tile] != 0;   // precomputed once per (b,n,h,tile)
            } else if (TRIATTN_SKIP && (wlo_ | whi_) == 0u) {
                // Exactness guard: skipping this tile is exact only if 0 * V == +0 for every V element, i.e. V holds no NaN/Inf.
                // Scan the tile's V rows (64 B each, 16-B aligned): lane covers keys k_lo and k_hi.
                const __nv_bfloat16* vt = p.v + b * p.svB + (long long)(n0 + nn) * p.svN + h * p.svH;
                uint32_t acc_and = 0u;
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int key = half ? k_hi : k_lo;
                    if (key < S) {
                        const uint4* src = reinterpret_cast<const uint4*>(vt + (long long)key * p.svS);
#pragma unroll
                        for (int c4 = 0; c4 < 4; ++c4) {
                            const uint4 w = __ldg(src + c4);
                            // a bf16 is NaN/Inf iff its exponent bits (0x7F80) are all set; test both halves of each 32-bit word
                            acc_and |= ((w.x & 0x7F807F80u) + 0x00800080u) & 0x80008000u;   // per half: carry into bit15/31 iff exponent == 0xFF
                            acc_and |= ((w.y & 0x7F807F80u) + 0x00800080u) & 0x80008000u;
                            acc_and |= ((w.z & 0x7F807F80u) + 0x00800080u) & 0x80008000u;
                            acc_and |= ((w.w & 0x7F807F80u) + 0x00800080u) & 0x80008000u;
                        }
                    }
                }
                vbad = __any_sync(0xffffffffu, acc_and != 0u) != 0;
            }
            if (lane == 0) {
                mw[(nn * ntiles + tile) * 2 + 0] = wlo_; mw[(nn * ntiles + tile) * 2 + 1] = whi_;
                if ((wlo_ | whi_) == 0u) { if (vbad) atomicOr(&vbd[tile], 1u << nn); else atomicOr(&amt[tile], 1u << nn); }
            }
        }
    }
    const uint32_t gact_skip = (G == 32) ? 0u : (~0u << Gact) & ((1u << G) - 1u);   // bits of inactive pair-rows
    auto skipmask_for = [&](int tile) -> uint32_t {                  // pair-rows whose K/V need not be loaded for `tile`
        uint32_t m_ = gact_skip;
        if (HAS_MASK && TRIATTN_SKIP && TRIATTN_LOADSKIP) {
            uint32_t am = amt[tile] & ~m_;
            while (am) {                                              // usually empty
                const int nn = __ffs(am) - 1; am &= am - 1;
                if (n_all_ok(nn)) m_ |= 1u << nn;
            }
        }
        return m_;
    };
    __syncthreads();                                                  // okw init visible
    // prologue: loads for tiles 0..NST-2
#pragma unroll
    for (int st = 0; st < NST - 1; ++st) { if (st < ntiles) load_stage(st, st, skipmask_for(st)); cp_async_commit(); }

#pragma unroll 1
    for (int it = 0; it < ntiles; ++it) {
        const int s = it % NST;
        cp_async_wait<NST - 2>();
        __syncthreads();                                              // tile `it` landed for everyone; everyone finished tile it-1
        if (it + NST - 1 < ntiles) load_stage(it + NST - 1, (it + NST - 1) % NST, skipmask_for(it + NST - 1));
        cp_async_commit();
        const uint8_t* base = smem + QSM + s * STAGE;
        const uint8_t* bsm = base + G * 2 * KVB;
        const int key0 = it * 64;
        const int kval = S - key0;

#pragma unroll
        for (int i = 0; i < GW; ++i) {
            const int nn = wn * GW + i;                        // pair-row index inside the CTA group
            if (nn < Gact) {
                uint32_t mlo = 0xffffffffu, mhi = 0xffffffffu;
                if (HAS_MASK) { mlo = mw[(nn * ntiles + it) * 2 + 0]; mhi = mw[(nn * ntiles + it) * 2 + 1]; }
                bool skip = false;
                if (HAS_MASK && TRIATTN_SKIP) {
                    if ((mlo | mhi) == 0u && !((vbd[it] >> nn) & 1u)) {          // all keys masked and V tile finite (else exact normal path)
                        if ((okmine >> nn) & 1u) skip = true;
                        else {
                            bool ok = (m_run[i][0] >= -1.0e9f) && (m_run[i][1] >= -1.0e9f);
                            skip = __all_sync(0xffffffffu, ok);
                        }
                    }
                }
                if (skip) {
                    // exact emulation of a tile whose P are all +0: O = O*1 (mul.ftz flushes denormals) then the MMA adds zero
                    // products, which leaves O unchanged except that a -0.0 accumulator becomes +0.0 (own microtest mma_zero_probe)
#pragma unroll
                    for (int jd = 0; jd < 4; ++jd) {
                        o[i][jd][0] = add_ftz(mul_ftz(o[i][jd][0], 1.f), 0.f); o[i][jd][1] = add_ftz(mul_ftz(o[i][jd][1], 1.f), 0.f);
                        o[i][jd][2] = add_ftz(mul_ftz(o[i][jd][2], 1.f), 0.f); o[i][jd][3] = add_ftz(mul_ftz(o[i][jd][3], 1.f), 0.f);
                    }
                } else {
                    const uint8_t* sK = base + nn * 2 * KVB; const uint8_t* sV = sK + KVB;
                    if (TRIATTN_QSMEM) {           // A fragments: matrices (rows lr0-blk, k lo), (rows +8, k lo), (rows, k hi), (rows+8, k hi)
                        const uint8_t* qs = smem + nn * BM * 64;
                        const int qrow_l = wrow * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
                        for (int ks = 0; ks < 2; ++ks) ldmatrix_x4(qa[i][ks], qs + swz_off(qrow_l, ks * 2 + (lane >> 4)));
                    }
                    float x[8][4];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        x[j][0] = x[j][1] = x[j][2] = x[j][3] = 0.f;
                        uint32_t kb[4];
                        if (!(TRIATTN_PERF & 16)) {
                        ldmatrix_x4(kb, sK + koff + j * 512);
                        mma16816(x[j], qa[i][0], &kb[0]);
                        mma16816(x[j], qa[i][1], &kb[2]);
                        }
                    }
                    const bool allvalid = HAS_MASK ? ((mlo & mhi) == 0xffffffffu) : (kval >= 64);
                    const uint32_t wlo = mlo >> (2 * t), whi = mhi >> (2 * t);      // this thread's columns start at bit 2t
                    float jm0[8], jm1[8];
                    auto logits = [&](auto AV) {
                        constexpr bool ALLV = decltype(AV)::value;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            float b0, b1, b2, b3;
                            if (TRIATTN_PERF & 1) { b0 = b1 = b2 = b3 = 0.f; } else
                            if (BIAS_BF16) {
                                uint32_t u0 = *reinterpret_cast<const uint32_t*>(bsm + boff0 + j * 16);
                                uint32_t u1 = *reinterpret_cast<const uint32_t*>(bsm + boff1 + j * 16);
                                b0 = __uint_as_float(u0 << 16); b1 = __uint_as_float(u0 & 0xffff0000u);
                                b2 = __uint_as_float(u1 << 16); b3 = __uint_as_float(u1 & 0xffff0000u);
                            } else {
                                float2 f0 = *reinterpret_cast<const float2*>(bsm + boff0 + j * 32);
                                float2 f1 = *reinterpret_cast<const float2*>(bsm + boff1 + j * 32);
                                b0 = f0.x; b1 = f0.y; b2 = f1.x; b3 = f1.y;
                            }
                            float y0 = mul_ftz(fma_ftz(x[j][0], scale, b0), LOG2E_F);
                            float y1 = mul_ftz(fma_ftz(x[j][1], scale, b1), LOG2E_F);
                            float y2 = mul_ftz(fma_ftz(x[j][2], scale, b2), LOG2E_F);
                            float y3 = mul_ftz(fma_ftz(x[j][3], scale, b3), LOG2E_F);
                            if (!ALLV) {
                                bool v0, v1;
                                if (HAS_MASK) {
                                    uint32_t w = (j < 4) ? (wlo >> (8 * j)) : (whi >> (8 * (j - 4)));
                                    v0 = w & 1u; v1 = (w >> 1) & 1u;
                                } else {
                                    int col = 8 * j + 2 * t; v0 = col < kval; v1 = (col + 1) < kval;
                                }
                                y0 = v0 ? y0 : MASK_X2; y1 = v1 ? y1 : MASK_X2; y2 = v0 ? y2 : MASK_X2; y3 = v1 ? y3 : MASK_X2;
                            }
                            x[j][0] = y0; x[j][1] = y1; x[j][2] = y2; x[j][3] = y3;
                            jm0[j] = fmaxf(y0, y1); jm1[j] = fmaxf(y2, y3);
                        }
                    };
                    if (allvalid) logits(std::true_type{}); else logits(std::false_type{});
                    // max is order-independent -> balanced tree (short dependency chains)
                    if (!(TRIATTN_PERF & 32)) {
#pragma unroll
                    for (int w_ = 4; w_ >= 1; w_ >>= 1) {
#pragma unroll
                        for (int j = 0; j < w_; ++j) { jm0[j] = fmaxf(jm0[j], jm0[j + w_]); jm1[j] = fmaxf(jm1[j], jm1[j + w_]); }
                    }
                    }
                    float tmax0 = jm0[0], tmax1 = jm1[0];
                    if (!(TRIATTN_PERF & 32)) {
                    tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 2)); tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 1));
                    tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 2)); tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 1));
                    }
                    const float mn0 = fmaxf(m_run[i][0], tmax0), mn1 = fmaxf(m_run[i][1], tmax1);
                    const float ms0 = (mn0 == NEG_INF) ? 0.f : mn0, ms1 = (mn1 == NEG_INF) ? 0.f : mn1;
                    const float al0 = ex2_ftz(sub_ftz(m_run[i][0], ms0)), al1 = ex2_ftz(sub_ftz(m_run[i][1], ms1));
                    m_run[i][0] = mn0; m_run[i][1] = mn1;
                    if (!(TRIATTN_PERF & 64)) {
#pragma unroll
                    for (int jd = 0; jd < 4; ++jd) {
                        o[i][jd][0] = mul_ftz(o[i][jd][0], al0); o[i][jd][1] = mul_ftz(o[i][jd][1], al0);
                        o[i][jd][2] = mul_ftz(o[i][jd][2], al1); o[i][jd][3] = mul_ftz(o[i][jd][3], al1);
                    }
                    }
                    float s0 = 0.f, s1 = 0.f;
#pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        uint32_t pa[4];
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            const int j = 2 * ks + hh;
                            float p0, p1, p2, p3;
                            if (TRIATTN_PERF & 2) { p0 = sub_ftz(x[j][0], ms0); p1 = sub_ftz(x[j][1], ms0); p2 = sub_ftz(x[j][2], ms1); p3 = sub_ftz(x[j][3], ms1); }
                            else { p0 = ex2_ftz(sub_ftz(x[j][0], ms0)); p1 = ex2_ftz(sub_ftz(x[j][1], ms0)); p2 = ex2_ftz(sub_ftz(x[j][2], ms1)); p3 = ex2_ftz(sub_ftz(x[j][3], ms1)); }
                            if (!(TRIATTN_PERF & 4)) {
                            if (j == 0) { s0 = add_ftz(p0, p1); s1 = add_ftz(p2, p3); }
                            else { s0 = add_ftz(add_ftz(s0, p0), p1); s1 = add_ftz(add_ftz(s1, p2), p3); }
                            }
                            pa[hh * 2 + 0] = pack_bf16x2(p0, p1);
                            pa[hh * 2 + 1] = pack_bf16x2(p2, p3);
                        }
                        uint32_t vb[4];
                        if (TRIATTN_PERF & 8) {      // keep a data dependence on pa so the P work is not dead-code-eliminated
                            o[i][0][0] += __uint_as_float(pa[0] ^ pa[1] ^ pa[2] ^ pa[3]) * 0.f;
                        } else {
                        ldmatrix_x4_trans(vb, sV + voff0 + ks * 1024);
                        mma16816(o[i][0], pa, &vb[0]);
                        mma16816(o[i][1], pa, &vb[2]);
                        ldmatrix_x4_trans(vb, sV + voff2 + ks * 1024);
                        mma16816(o[i][2], pa, &vb[0]);
                        mma16816(o[i][3], pa, &vb[2]);
                        }
                    }
                    l_run[i][0] = fma_ftz(l_run[i][0], al0, s0); l_run[i][1] = fma_ftz(l_run[i][1], al1, s1);
                    if (HAS_MASK && TRIATTN_SKIP && !((okmine >> nn) & 1u)) {
                        bool ok = (m_run[i][0] >= -1.0e9f) && (m_run[i][1] >= -1.0e9f);
                        ok = __all_sync(0xffffffffu, ok);
                        if (ok) { okmine |= 1u << nn; if (lane == 0) atomicOr(&okw[nn], 1u << wrow); }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < GW; ++i) {
        const int nn = wn * GW + i;
        if (nn < Gact) {
            float l0 = l_run[i][0], l1 = l_run[i][1];
            l0 = l0 + __shfl_xor_sync(0xffffffffu, l0, 1); l0 = l0 + __shfl_xor_sync(0xffffffffu, l0, 2);
            l1 = l1 + __shfl_xor_sync(0xffffffffu, l1, 1); l1 = l1 + __shfl_xor_sync(0xffffffffu, l1, 2);
            const bool dg0 = (l0 == 0.f) || (l0 != l0), dg1 = (l1 == 0.f) || (l1 != l1);
            const float rr0 = rcp_ftz(l0), rr1 = rcp_ftz(l1);
            __nv_bfloat16* ob = p.out + ((((long long)b * p.N + (n0 + nn)) * p.H + h) * (long long)S) * 32;
#pragma unroll
            for (int jd = 0; jd < 4; ++jd) {
                int col = jd * 8 + 2 * t;
                float y0 = dg0 ? o[i][jd][0] : mul_ftz(o[i][jd][0], rr0), y1 = dg0 ? o[i][jd][1] : mul_ftz(o[i][jd][1], rr0);
                float y2 = dg1 ? o[i][jd][2] : mul_ftz(o[i][jd][2], rr1), y3 = dg1 ? o[i][jd][3] : mul_ftz(o[i][jd][3], rr1);
                if (r0 < S) *reinterpret_cast<uint32_t*>(ob + (long long)r0 * 32 + col) = pack_bf16x2(y0, y1);
                if (r1 < S) *reinterpret_cast<uint32_t*>(ob + (long long)r1 * 32 + col) = pack_bf16x2(y2, y3);
            }
        }
    }
}

template <int NWR, int NSPLIT, int GW, int NST, bool MASK, bool BB>
int launch_inst(const PV3& p, cudaStream_t st) {
    constexpr int BM = NWR * 16; constexpr int G = NSPLIT * GW;
    constexpr int STAGE = G * 2 * 4096 + BM * 72 * (BB ? 2 : 4);
    constexpr int QSM = TRIATTN_QSMEM ? G * BM * 64 : 0;
    const int ntiles = (p.S + 63) / 64;
    const size_t smem = (size_t)QSM + (size_t)NST * STAGE + (MASK ? (size_t)G * ntiles * 8 + (size_t)ntiles * 8 : 0);
    static int attr_smem = 0;
    if ((int)smem > attr_smem) {
        cudaError_t e = cudaFuncSetAttribute(triattn_v3_kernel<NWR, NSPLIT, GW, NST, MASK, BB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) return (int)e;
        attr_smem = (int)smem;
    }
    dim3 grid((p.q_rows + BM - 1) / BM, p.H, p.B * p.n_groups);
    triattn_v3_kernel<NWR, NSPLIT, GW, NST, MASK, BB><<<grid, NWR * NSPLIT * 32, smem, st>>>(p);
    return (int)cudaGetLastError();
}
template <int NWR, int NSPLIT, int GW, int NST>
int launch_cfg(const PV3& p, bool mask, bool bb, cudaStream_t st) {
    if (mask) return bb ? launch_inst<NWR, NSPLIT, GW, NST, true, true>(p, st) : launch_inst<NWR, NSPLIT, GW, NST, true, false>(p, st);
    return bb ? launch_inst<NWR, NSPLIT, GW, NST, false, true>(p, st) : launch_inst<NWR, NSPLIT, GW, NST, false, false>(p, st);
}

}  // namespace

extern "C" {
struct LaunchArgsV3 {
    const void *q, *k, *v, *bias; int bias_bf16; const void* mask; void* out;
    int B, N, H, S;
    long long sqB, sqN, sqH, sqS, skB, skN, skH, skS, svB, svN, svH, svS, sbB, sbH, sbQ, sbK, smB, smN;
    float scale;
    int G, nwarps, bias_fast;
    void* stream;
    int cfg, pad_;
    void* vbad;                                    // optional vscan_kernel output (see triattn_v3_vscan); null -> in-kernel scan
};
static void fill_params(PV3& p, const LaunchArgsV3* a, int G) {
    p.vbad = (const uint8_t*)a->vbad;
    p.q = (const __nv_bfloat16*)a->q; p.k = (const __nv_bfloat16*)a->k; p.v = (const __nv_bfloat16*)a->v;
    p.bias = a->bias; p.mask = (const uint8_t*)a->mask; p.out = (__nv_bfloat16*)a->out;
    p.B = a->B; p.N = a->N; p.H = a->H; p.S = a->S; p.bias_fast = a->bias_fast;
    p.sqB = a->sqB; p.sqN = a->sqN; p.sqH = a->sqH; p.sqS = a->sqS; p.skB = a->skB; p.skN = a->skN; p.skH = a->skH; p.skS = a->skS;
    p.svB = a->svB; p.svN = a->svN; p.svH = a->svH; p.svS = a->svS; p.sbB = a->sbB; p.sbH = a->sbH; p.sbQ = a->sbQ; p.sbK = a->sbK;
    p.smB = a->smB; p.smN = a->smN; p.scale = a->scale;
    p.n_groups = (p.N + G - 1) / G;
    // a->pad_ encodes the query-row range of this launch: 0 -> all rows; >0 -> MAIN part rows [0, S - pad_); <0 -> TAIL rows [S + pad_, S)
    if (a->pad_ > 0) { p.q0 = 0; p.q_rows = p.S - a->pad_; }
    else if (a->pad_ < 0) { p.q0 = p.S + a->pad_; p.q_rows = -a->pad_; }
    else { p.q0 = 0; p.q_rows = p.S; }
}
#ifdef TRIATTN_VARIANT
int triattn_v3var_vscan(const LaunchArgsV3* a, void* out) {
    if (a->mask == nullptr || out == nullptr) return -2;
    dim3 grid(a->B * a->N, a->H);
    vscan_kernel<<<grid, 128, 0, (cudaStream_t)a->stream>>>((const __nv_bfloat16*)a->v, a->svB, a->svN, a->svH, a->svS,
                                                             (const uint8_t*)a->mask, a->smB, a->smN, a->N, a->H, a->S, (uint8_t*)out);
    return (int)cudaGetLastError();
}
// single-configuration build for config sweeps: -DTRIATTN_VARIANT -DVAR_NWR=.. -DVAR_NSPLIT=.. -DVAR_GW=.. -DVAR_NST=..
int triattn_v3var_launch(const LaunchArgsV3* a) {
    PV3 p; fill_params(p, a, VAR_NSPLIT * VAR_GW);
    return launch_cfg<VAR_NWR, VAR_NSPLIT, VAR_GW, VAR_NST>(p, a->mask != nullptr, a->bias_bf16 != 0, (cudaStream_t)a->stream);
}
int triattn_v3var_G() { return VAR_NSPLIT * VAR_GW; }
int triattn_v3var_BM() { return VAR_NWR * 16; }
int triattn_v3var_threads() { return VAR_NWR * VAR_NSPLIT * 32; }
int triattn_v3var_sizeof_launchargs() { return (int)sizeof(LaunchArgsV3); }
#else
// cfg ids -> <row-warps, n-splits, n per warp, stages>; G = NSPLIT*GW, BM = 16*NWR
//  0: <4,2,2,2> G4 BM64 8 warps   1: <4,2,1,2> G2 BM64 8w   2: <4,1,2,2> G2 BM64 4w   3: <4,1,1,2> G1 BM64 4w
//  4: <8,1,2,2> G2 BM128 8w       5: <8,1,2,3> G2 BM128 8w 3-stage   6: <8,2,2,2> G4 BM128 16w   7: <8,2,2,3> G4 BM128 16w 3-stage
//  8: <4,2,2,3> G4 BM64 8w 3-stage
static const int kCfgG[11] = {4, 2, 2, 1, 2, 2, 4, 4, 4, 4, 4};
static const int kCfgBM[11] = {64, 64, 64, 64, 128, 128, 128, 128, 64, 16, 32};
int triattn_v3_vscan(const LaunchArgsV3* a, void* out) {       // fills out[B*N*H*ntiles]; requires a->mask != null
    if (a->mask == nullptr || out == nullptr) return -2;
    dim3 grid(a->B * a->N, a->H);
    vscan_kernel<<<grid, 128, 0, (cudaStream_t)a->stream>>>((const __nv_bfloat16*)a->v, a->svB, a->svN, a->svH, a->svS,
                                                             (const uint8_t*)a->mask, a->smB, a->smN, a->N, a->H, a->S, (uint8_t*)out);
    return (int)cudaGetLastError();
}
int triattn_v3_launch(const LaunchArgsV3* a) {
    const int cfg = a->cfg;
    if (cfg < 0 || cfg > 10) return -1;
    PV3 p; fill_params(p, a, kCfgG[cfg]);
    cudaStream_t st = (cudaStream_t)a->stream;
    const bool mask = a->mask != nullptr, bb = a->bias_bf16 != 0;
    switch (cfg) {
        case 0: return launch_cfg<4, 2, 2, 2>(p, mask, bb, st);
        case 1: return launch_cfg<4, 2, 1, 2>(p, mask, bb, st);
        case 2: return launch_cfg<4, 1, 2, 2>(p, mask, bb, st);
        case 3: return launch_cfg<4, 1, 1, 2>(p, mask, bb, st);
        case 4: return launch_cfg<8, 1, 2, 2>(p, mask, bb, st);
        case 5: return launch_cfg<8, 1, 2, 3>(p, mask, bb, st);
        case 6: return launch_cfg<8, 2, 2, 2>(p, mask, bb, st);
        case 7: return launch_cfg<8, 2, 2, 3>(p, mask, bb, st);
        case 8: return launch_cfg<4, 2, 2, 3>(p, mask, bb, st);
        case 9: return launch_cfg<1, 4, 1, 2>(p, mask, bb, st);       // BM=16, G=4 (tail rows)
        case 10: return launch_cfg<2, 4, 1, 2>(p, mask, bb, st);      // BM=32, G=4 (tail rows)
    }
    return -1;
}
int triattn_v3_cfg_G(int cfg) { return (cfg >= 0 && cfg <= 10) ? kCfgG[cfg] : -1; }
int triattn_v3_cfg_BM(int cfg) { return (cfg >= 0 && cfg <= 10) ? kCfgBM[cfg] : -1; }
int triattn_v3_sizeof_launchargs() { return (int)sizeof(LaunchArgsV3); }
#endif
}
