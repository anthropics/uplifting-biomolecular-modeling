// EdmPoolLogits: Y[M,N] = X[M,K] . W[K,N] in float32 with the accumulation order of the cuBLAS kernel the stock graph runs for the
// pooling logits (strided-batched GEMM, two rows per batch): per output element, K is split in two halves; each half is accumulated in
// consecutive chunks of 256 (a fresh accumulator per chunk starting at +0, one fused multiply-add per k in ascending order), the chunk sums
// added left to right; the result is half0 + half1. Same float32 operations in the same order as stock => the same bits.
//
// Execution structure (it decides which thread computes an element and when; the per-element operation sequence above is fixed):
// * A 128-thread block computes a 64 x 128 tile of Y; its 4 warps are a 2 x 2 grid of 32 x 64 warp tiles; lane (i = lane/8, j = lane%8)
//   of a warp owns rows {4i..4i+3, 16+4i..16+4i+3} and columns {4j..4j+3, 32+4j..32+4j+3} of the warp tile: an 8 x 8 register tile
//   updated by one rank-1 step per k. The grid is one-dimensional with the N/128 column tiles of a row block adjacent, so the blocks that
//   read the same rows of X run together. Up to three blocks share a multiprocessor (__launch_bounds__(128, 3): at most 168 registers a thread).
// * K runs in tiles of 16 through a 3-deep shared-memory ring: W's 16 x 128 tile is copied with 16-byte cp.async; X's 64 x 16 tile is
//   loaded as float4 along k into registers at the top of a K step and stored transposed (k-major, row index XOR-swizzled with 8*(k/4 % 4):
//   conflict-free stores and conflict-free float4 fragment reads) near the end of the step, one __syncthreads per K tile.
// * Per k, a thread reads its 8 x-values and 8 w-values as four float4 shared loads into registers that are double-buffered across k
//   (the loads for k+1 are issued after the 64 multiply-adds of k), and walks its 8 x 8 tile in serpentine row order so that consecutive
//   multiply-adds share an operand (the register file delivers about two operands per clock; the third comes from the operand cache).
// * The chunk accumulator lives in registers. A finished chunk sum that is not the last of its half is added into the thread's private
//   running-sum slots in shared memory (64 floats per thread); the finished first-half sum h0 is stored to Y and read back once when h1 is
//   complete (y = h0 + h1), so the second half needs no extra registers.
#include <cuda_runtime.h>
#include <cstdint>
#include "edm_ops.h"

namespace {
constexpr int THREADS = 128, BM = 64, BN = 128, BK = 16, STAGES = 3;
constexpr int A_STAGE = BK * BM;                          // floats per stage: As[k][m ^ 8*(k/4 % 4)]
constexpr int B_STAGE = BK * BN;                          // floats per stage: Bs[k][n]
constexpr int SLOT = 68;                                  // running-sum slots per thread: 64 floats + 4 of padding (16-byte aligned rows)
constexpr size_t SMEM_BYTES = (size_t)(STAGES * (A_STAGE + B_STAGE) + THREADS * SLOT) * sizeof(float);   // 71,680 bytes

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async_16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int PENDING> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING) : "memory"); }
}  // namespace

__global__ void __launch_bounds__(THREADS, 3)
pool_logits_kernel(const float* __restrict__ X, const float* __restrict__ W, float* __restrict__ Y, int M, int K, int N, int half, int chunk, int n_tiles) {
    extern __shared__ __align__(16) float smem[];
    float* const As = smem;                                   // [STAGES][A_STAGE]
    float* const Bs = smem + STAGES * A_STAGE;                // [STAGES][B_STAGE]
    float* const slots = smem + STAGES * (A_STAGE + B_STAGE); // [THREADS][SLOT]
    const unsigned bs_u32 = smem_u32(Bs);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n0 = (int)(blockIdx.x % (unsigned)n_tiles) * BN;
    const int m0 = (int)(blockIdx.x / (unsigned)n_tiles) * BM;

    // ---- producer. X tile 64 x 16: lane (r = lane/4, q = lane%4) of warp w loads rows 16w + r and 16w + 8 + r, k = 4q..4q+3 (two float4)
    //      and stores them k-major at As[4q + e][row ^ 8q]. W tile 16 x 128: thread copies k rows warp + 4u (u = 0..3), columns 4*lane..+3.
    const int q = lane & 3;
    const float* xrow[2]; bool xok[2]; int adst[2];
#pragma unroll
    for (int o = 0; o < 2; ++o) {
        const int row = 16 * warp + 8 * o + (lane >> 2), grow = m0 + row;
        xok[o] = grow < M;
        xrow[o] = X + (size_t)(xok[o] ? grow : 0) * K + 4 * q;
        adst[o] = (4 * q) * BM + (row ^ (8 * q));
    }
    const float* const wsrc = W + (size_t)warp * N + n0 + 4 * lane;
    const unsigned wdst = (unsigned)((warp * BN + 4 * lane) * 4);
    float4 xreg[2];
    auto load_x = [&](int kt) {                               // global -> registers (tile kt)
#pragma unroll
        for (int o = 0; o < 2; ++o) xreg[o] = xok[o] ? __ldg(reinterpret_cast<const float4*>(xrow[o] + (size_t)kt * BK)) : make_float4(0.f, 0.f, 0.f, 0.f);
    };
    auto store_x = [&](int stage) {                           // registers -> shared (tile's stage)
#pragma unroll
        for (int o = 0; o < 2; ++o) {
            float* d = As + stage * A_STAGE + adst[o]; const float4 v = xreg[o];
            d[0] = v.x; d[BM] = v.y; d[2 * BM] = v.z; d[3 * BM] = v.w;
        }
    };
    auto copy_w = [&](int kt, int stage) {                    // global -> shared, asynchronous (tile kt)
        const unsigned dst = bs_u32 + (unsigned)(stage * B_STAGE * 4) + wdst;
#pragma unroll
        for (int u = 0; u < 4; ++u) cp_async_16(dst + u * (4 * BN * 4), wsrc + (size_t)(kt * BK + 4 * u) * N);
    };

    // ---- consumer: warp tile origin (wm0, wn0); this lane's rows am + {0..3}, am + 16 + {0..3}, columns bn + {0..3}, bn + 32 + {0..3}
    const int wm0 = (warp >> 1) * 32, wn0 = (warp & 1) * 64;
    const int am = wm0 + 4 * (lane >> 3), bn = wn0 + 4 * (lane & 7);
    float p[8][8], a[2][8], b[2][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) p[i][j] = 0.0f;
    auto load_frag = [&](int stage, int kk, int buf) {       // the x- and w-values of k step kk into register buffer buf
        const int sw = 8 * ((kk >> 2) & 3);
        const float* as = As + stage * A_STAGE + kk * BM;
        const float* bs = Bs + stage * B_STAGE + kk * BN;
        const float4 a0 = *reinterpret_cast<const float4*>(as + (am ^ sw)), a1 = *reinterpret_cast<const float4*>(as + ((am + 16) ^ sw));
        const float4 b0 = *reinterpret_cast<const float4*>(bs + bn), b1 = *reinterpret_cast<const float4*>(bs + bn + 32);
        a[buf][0] = a0.x; a[buf][1] = a0.y; a[buf][2] = a0.z; a[buf][3] = a0.w; a[buf][4] = a1.x; a[buf][5] = a1.y; a[buf][6] = a1.z; a[buf][7] = a1.w;
        b[buf][0] = b0.x; b[buf][1] = b0.y; b[buf][2] = b0.z; b[buf][3] = b0.w; b[buf][4] = b1.x; b[buf][5] = b1.y; b[buf][6] = b1.z; b[buf][7] = b1.w;
    };
    float* const myslots = slots + tid * SLOT;                // running sum of the current half, meaningful once its first chunk has closed
#pragma unroll
    for (int s = 0; s < 16; ++s) reinterpret_cast<float4*>(myslots)[s] = make_float4(0.f, 0.f, 0.f, 0.f);   // defined contents before the first read
    const size_t yrow0 = (size_t)(m0 + am) * N + n0 + bn;     // this thread's element (i, j): Y[yrow0 + (i < 4 ? i : 12 + i) * N + j + (j >= 4) * 28]
    auto y_at = [&](int i, int jj) { return Y + yrow0 + (size_t)(i < 4 ? i : 12 + i) * N + 32 * jj; };
    auto row_ok = [&](int i) { return m0 + am + (i < 4 ? i : 12 + i) < M; };

    const int KT = K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {                    // prologue: tiles 0 .. STAGES-2
        if (s < KT) { load_x(s); store_x(s); copy_w(s, s); }
        cp_async_commit();
    }
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    load_frag(0, 0, 0);
    int rstage = 0, chunks_closed = 0, hstart = 0;            // stage of tile kt; chunks of the current half already in the slots; k where the half began
    for (int kt = 0; kt < KT; ++kt) {
        const int wstage = rstage == 0 ? STAGES - 1 : rstage - 1;   // stage of tile kt+STAGES-1: every thread finished reading it before the last barrier
        const int nstage = rstage == STAGES - 1 ? 0 : rstage + 1;
        const bool more = kt + STAGES - 1 < KT;
        if (more) { load_x(kt + STAGES - 1); copy_w(kt + STAGES - 1, wstage); }
        cp_async_commit();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const int cur = kk & 1, nxt = cur ^ 1;
            if (kk == BK - 2 && more) store_x(wstage);
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) { const int j = (i & 1) ? 7 - jj : jj; p[i][j] = __fmaf_rn(a[cur][i], b[cur][j], p[i][j]); }
            if (kk == BK - 1) {                               // tile kt+1: this thread's W copies have landed; the barrier publishes everyone's tile
                cp_async_wait<STAGES - 2>();
                __syncthreads();
                load_frag(nstage, 0, nxt);
            } else {
                load_frag(rstage, kk + 1, nxt);
            }
        }
        rstage = nstage;
        // ---- accumulation boundaries (each a multiple of BK): the end of half 0, or the end of a 256-chunk inside a half
        const int kend = (kt + 1) * BK;
        if (kend == half) {                                   // h0 = c0 (+ c1 (+ c2)) -> Y
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    float4 v; float* vp = reinterpret_cast<float*>(&v);
                    const float4 r4 = *reinterpret_cast<const float4*>(myslots + (2 * i + jj) * 4); const float* rp = reinterpret_cast<const float*>(&r4);
#pragma unroll
                    for (int e = 0; e < 4; ++e) { const int j = 4 * jj + e; vp[e] = chunks_closed == 0 ? p[i][j] : __fadd_rn(rp[e], p[i][j]); p[i][j] = 0.0f; }
                    if (row_ok(i)) *reinterpret_cast<float4*>(y_at(i, jj)) = v;
                }
            chunks_closed = 0; hstart = half;
        } else if (kend < K && (kend - hstart) % chunk == 0) { // running sum of this half += chunk
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    float4 r4 = *reinterpret_cast<const float4*>(myslots + (2 * i + jj) * 4); float* rp = reinterpret_cast<float*>(&r4);
#pragma unroll
                    for (int e = 0; e < 4; ++e) { const int j = 4 * jj + e; rp[e] = chunks_closed == 0 ? p[i][j] : __fadd_rn(rp[e], p[i][j]); p[i][j] = 0.0f; }
                    *reinterpret_cast<float4*>(myslots + (2 * i + jj) * 4) = r4;
                }
            ++chunks_closed;
        }
    }
    cp_async_wait<0>();
    // ---- y = h0 + h1, h1 = running sum + last chunk
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (!row_ok(i)) continue;
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            float* yp = y_at(i, jj);
            const float4 h0 = *reinterpret_cast<const float4*>(yp); const float* h0p = reinterpret_cast<const float*>(&h0);
            const float4 r4 = *reinterpret_cast<const float4*>(myslots + (2 * i + jj) * 4); const float* rp = reinterpret_cast<const float*>(&r4);
            float4 o; float* op_ = reinterpret_cast<float*>(&o);
#pragma unroll
            for (int e = 0; e < 4; ++e) { const int j = 4 * jj + e; const float h1 = chunks_closed == 0 ? p[i][j] : __fadd_rn(rp[e], p[i][j]); op_[e] = __fadd_rn(h0p[e], h1); }
            *reinterpret_cast<float4*>(yp) = o;
        }
    }
}

namespace edm {
cudaError_t LaunchPoolLogits(cudaStream_t stream, const float* X, const float* W, float* Y, int M, int K, int N, const float* tail, int T) {
    // rows [0, M-T) of Y computed from X; rows [M-T, M) copied from `tail` ([T, N], computed by the caller with the stock op)
    if (K % 128 != 0 || K < 512 || N % 128 != 0 || T < 0 || T > M) return cudaErrorInvalidValue;   // K multiple of 128: halves and 256-chunks fall on K-tile boundaries
    const int Mm = M - T;
    if (Mm > 0) {
        cudaError_t e = cudaFuncSetAttribute(pool_logits_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
        if (e != cudaSuccess) return e;
        const int n_tiles = N / BN, m_tiles = (Mm - 1) / BM + 1;                                      // Mm >= 1 here; no int overflow for any Mm < 2^31
        pool_logits_kernel<<<(unsigned)n_tiles * (unsigned)m_tiles, THREADS, SMEM_BYTES, stream>>>(X, W, Y, Mm, K, N, K / 2, 256, n_tiles);
        e = cudaGetLastError(); if (e != cudaSuccess) return e;
    }
    if (T > 0) return cudaMemcpyAsync(Y + (size_t)Mm * N, tail, (size_t)T * N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    return cudaSuccess;
}
}  // namespace edm
