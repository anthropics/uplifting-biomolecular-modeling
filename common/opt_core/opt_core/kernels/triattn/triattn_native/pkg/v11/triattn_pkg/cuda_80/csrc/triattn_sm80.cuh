// Triangle-attention forward for sm_80 (A100, cc 8.x): mma.sync.m16n8k16 (bf16 -> fp32), ldmatrix, cp.async multistage ring.
//
//   out[b,i,h,q,:] = softmax_k( scale * q[b,i,h,q,:].k[b,i,h,k,:] + bias[b,h,q,k]  (-inf where mask[b,i,k] == 0) ) @ v[b,i,h,k,:]
//
// Device code only (header-only, no torch): a host wrapper fills `Args` with raw pointers / element strides and calls
// launch<Traits>() (launch_sm80.cuh).  The algorithm is the family's many-warp formulation (cuda_c/csrc/triattn_mw.cu, whose numerics
// contract cuda_c/PORTING.md this member keeps mechanism by mechanism):
//   * one CTA = R pair rows x BM = 32 QG queries of one (b, h), as WR x QG warps each owning 32 queries of RW = R / WR consecutive pair rows;
//     the R rows share one staged pair-bias tile (BM x 64 keys, fp32, pre-divided by scale, laid out in m16n8 accumulator-fragment order so
//     one 16-byte shared load = one C fragment, loaded ONCE per warp and used as the initial accumulator of all its RW rows);
//   * per 32-key half tile every warp computes S = bias/scale + Q K^T (mma.sync, C operand = staged bias), p = ex2(S*c1 - m) with a
//     MAX-FREE streaming softmax (m = per-row integer offset kept SHIFT log2 units above the running log2-sum; exact power-of-two
//     renormalisations when the running sum leaves [2^-84, 2^-44]; rows seeded once from the first half holding a finite logit), O += P V
//     with P the bf16 A operand and the row sums accumulated from the SAME bf16 P by a ones-column mma;
//   * CTA tiles whose rows end non-finite or with a zero sum are appended to a fix list and recomputed by the SAFE instantiation of this
//     same tile routine (exact running max), launched unconditionally after the hot pass (census counter);
//   * masks: keys masked in every row of a batch element and keys >= S are -inf COLUMNS of the staged bias (exact); every pair row carries
//     a live key interval / kind (stage_mask): regular and interval rows stream only their own live key tiles and apply per-key masking
//     only on a tile holding an interval end; ragged rows (holes) apply 32-bit mask words per tile; fully-masked rows return the uniform
//     mean of v over the S keys (computed directly in the epilogue, outside the tile stream).  All of it in the one hot pass (no list pass):
//     a CTA streams the union of its rows' tile ranges.
// What differs from the sm_90a member: the TMA + mbarrier ring is a cp.async.cg 16-byte multistage ring filled cooperatively by all
// threads (one 32-key half tile {bias box | K x R | V x R} per ring slot, STAGES slots filled STAGES-2 halves ahead, <= 163 KB), K/V smem tiles are [32 keys][D] rows
// XOR-swizzled at 16-byte granularity (conflict-free for the cp.async writes and the ldmatrix reads), addresses are per-thread int64
// arithmetic from the native strides (transposed views included, no copies), the per-slot protocol is cp.async.wait_group + one
// __syncthreads per half tile, and the hot path is software-pipelined across halves (phaseA / phaseB) so that exponentials and tensor work
// alternate in program order.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#ifndef TS_DEVI
#define TS_DEVI __device__ __forceinline__
#endif

namespace triattn_sm80 {

// ------------------------------------------------------------------------------------------------------------------ PTX wrappers
namespace ptx {
TS_DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
TS_DEVI void ldsm_x4(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
TS_DEVI void ldsm_x2(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
TS_DEVI void ldsm_x4_t(uint32_t* r, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
TS_DEVI float4 lds128f(uint32_t addr) {
  float4 v; asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr)); return v;
}
TS_DEVI void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
TS_DEVI void mma16816c(float* d, const uint32_t* a, const uint32_t* b, const float* c) {   // d = a.b + c (c in other registers)
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
TS_DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
TS_DEVI uint32_t pack_bf16(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
template <typename U> TS_DEVI U opaque(U x) { asm volatile("" : "+r"(x)); return x; }   // value the compiler must keep (no rematerialisation)
// 16-byte global -> shared asynchronous copy (L2 only); full == false zero-fills the 16 destination bytes and reads nothing.
TS_DEVI void cp_async16(uint32_t dst, const void* src, bool full) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(full ? 16 : 0) : "memory");
}
TS_DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
// CTA-wide barrier WITHOUT the .aligned assumption of __syncthreads() (bar.sync): PTX `barrier.sync` lets the warps of a block reach the
// barrier from different code locations -- the general routine's per-warp steady / event bodies below do exactly that.
TS_DEVI void cta_sync_any() { asm volatile("barrier.sync 0;\n" ::: "memory"); }
template <int N> TS_DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
}  // namespace ptx
using namespace ptx;

// ------------------------------------------------------------------------------------------------------------------ constants
constexpr int BN = 64;             // key tile of the mask tables / fix census granularity
constexpr int BH = 32;             // keys per ring slot (half tile): the unit the stream advances by
// Max-free softmax operating point (PORTING.md 1): m sits SHIFT log2 units above the row's running log2-sum; renormalise when the
// running sum leaves [2^-(SHIFT+20), 2^-(SHIFT-20)] -- one unsigned compare on the float bits.
constexpr int SHIFT = 64;
constexpr uint32_t RN_LO = (uint32_t)(127 - SHIFT - 20) << 23, RN_WIDTH = 40u << 23;
enum : int { CLS_SKIP = 0, CLS_MIXED = 1, CLS_FULL = 2 };                    // per (row, tile) class: no work / per-key mask word / all 64 keys live
enum : int { ROW_INTERVAL = 0, ROW_RAGGED = 1, ROW_UNIFORM = 2 };           // per-row kind (stage_mask); no table = every row INTERVAL [0, S)
enum : int { FIX_TILES = 0, FIX_RESERVED = 1, FIX_COUNT = 2, FIX_LIST = 3 }; // int32 fix buffer: census of fixed tiles, reserved, this call's count, triples

// Plain-C launch arguments (element strides; d stride == 1 for q/k/v/out).
struct Args {
  const __nv_bfloat16* q; long long q_sB, q_sN, q_sH, q_sS;
  const __nv_bfloat16* k; long long k_sB, k_sN, k_sH, k_sS;
  const __nv_bfloat16* v; long long v_sB, v_sN, v_sH, v_sS;
  __nv_bfloat16* out;     long long o_sB, o_sN, o_sH, o_sS;
  const float* bias;       // staged fp32 [B,H,S128,S64] = bias/scale in fragment order, -inf on excluded key columns (stage_bias)
  const int4* rows;        // [B*N] per pair row {a, e, kind, 0}: live keys [a, e) / kind (see mask_rows_kernel); nullptr = no mask
  const uint32_t* maskw;   // [B*N*n_ktiles*2] 32-key mask words (bit = attend and key < S), read for RAGGED rows only; nullptr = no mask
  float* lse;              // [B,N,H,S_q] fp32: per query row log2(sum_k 2^(x_k)), x = log2-domain logit incl. bias (kLse instantiations); or nullptr
  int* fix;                // fix[FIX_TILES] += tiles recomputed by the SAFE pass (census, all calls); fix[FIX_COUNT] = this call's list length; fix[FIX_LIST..] = (qt, yg, bz) triples
  int B, N, H, S_q, S_kv;  // pair rows N; queries S_q; keys S_kv (== S_q for triangle attention; kept separate in the addressing)
  int n_ktiles, S128, S64; // key tiles; staged-bias row / column padding
  int qb;                  // CTA order: q tiles walked in blocks of qb (q tile fastest inside a block, then row groups)
  float c1;                // scale * log2(e)
  int dbg;                 // experiments only (results invalid unless 0 / 256): 32 = no renormalisation, 64 = no ring refills, 256 = every tile through the SAFE pass, 1024 = no per-tile barriers
};

// Compile-time geometry.  kHeadDim in {16, 32, 64}; kWR x kQG warps per CTA, each warp owning 32 queries of kRW (1 or 2) consecutive pair
// rows (R = kWR * kRW pair rows per CTA, BM = 32 kQG queries); kStages = ring slots of one 32-key half tile each; kSafe = exact
// running-max instantiation (fix pass); kGeneral = per-row kinds / mask words / tile ranges (a mask table may be present); kLse = per-row
// log2-sum-exp output; kMinBlocks = CTAs per SM the kernel is compiled for.
template <int kHeadDim, int kWR, int kRW, int kQG, int kStages, bool kSafe, bool kGeneral, bool kLse, int kMinBlocks = 1>
struct Traits {
  static constexpr int D = kHeadDim, WR = kWR, RW = kRW, R = kWR * kRW, QG = kQG, STAGES = kStages, MINB = kMinBlocks;
  static constexpr bool SAFE = kSafe, GENERAL = kGeneral, LSE = kLse;
  static constexpr int WARPS = WR * QG, THREADS = 32 * WARPS, BM = 32 * QG;
  static constexpr int KK = D / 16;                      // k16 steps of Q K^T
  static constexpr int DN = D / 8;                       // n8 column tiles of O
  static constexpr int CPR = D / 8;                      // 16-byte chunks per K/V row
  static constexpr int RP = 64 / D;                      // K/V rows per 128 bytes (swizzle period)
  static constexpr int ROWB = 2 * D;                     // bytes per K/V smem row
  static constexpr int KV_HALF = BH * ROWB;              // one K or V half tile (32 keys) of one pair row
  static constexpr int BIAS_BOX = BM * BH * 4;           // [BM q x 32 keys] fp32
  static constexpr int KOFF = BIAS_BOX, VOFF = KOFF + R * KV_HALF, SLOT_BYTES = VOFF + R * KV_HALF;   // ring slot = {bias box | K x R | V x R} of one half
  static constexpr int LEAD = STAGES - 2;                // halves in flight ahead of the one being consumed
  static constexpr int SMEM_BYTES = STAGES * SLOT_BYTES + 384;   // ring + the per-CTA row table / flags (SmemT) + 128-byte alignment slack
  static_assert(D == 16 || D == 32 || D == 64, "head dim");
  static_assert(RW == 1 || RW == 2, "rows per warp");
  static_assert(STAGES >= 3 && STAGES <= 8, "ring slots (>= 3: one landing while one is consumed and one computed ahead)");
  static_assert(SMEM_BYTES <= 163 * 1024, "shared memory per block (sm_80 opt-in max 163 KB)");
  static_assert(QG >= 1 && QG <= 4 && WR >= 1 && R <= 8, "geometry");
  static_assert(BH * CPR <= THREADS || (BH * CPR) % THREADS == 0, "K/V chunk mapping (a thread owns whole chunks of a half)");
};

template <class T>
struct __align__(128) SmemT {
  uint8_t slot[T::STAGES][T::SLOT_BYTES];
  int4 rowtab[8];          // per pair row r of the CTA tile: {a, e, kind (-1: row absent), 0} -- live keys [a, e) (e clamped to S_kv), see mask_rows_kernel
  int flagged;
  int kt_lo, kt_hi;
};
template <class T> constexpr bool smem_fits() { return sizeof(SmemT<T>) + 127 <= (size_t)T::SMEM_BYTES; }

typedef float CFrag[2][4][4];      // S (then p) of one 32-key half, C-fragment order [mt][j][e]
// Per-thread state of one warp's 32 queries of one pair row: o accumulators [mt][dn][4]; l4 = running row sums as the ones-column mma
// accumulator ([mt][0]=[mt][1] row g, [mt][2]=[mt][3] row g+8); m = row offsets; qa = Q A-fragments [mt][kk]; seeded: bit 2mt+hh.
template <class T>
struct FragT { float o[2][T::DN][4]; float l4[2][4]; float m[2][2]; uint32_t qa[2][T::KK][4]; uint32_t seeded; };
struct Lanes { uint32_t bA, bB, kLane[2], vLane[4]; };   // per-lane smem address parts: bias column j at (j odd ? bB : bA) + 64 j; K chunk groups; V d-pairs (the warp's row 0; row w adds w * KV_HALF)

// ------------------------------------------------------------------------------------------------------------------ math blocks
// K B-fragments of the 8-key column at smem row base `rowb`, the warp's row w: kb[2kk], kb[2kk+1] = d 16kk..16kk+15.
template <class T>
TS_DEVI void load_kb(uint32_t* kb, uint32_t rowb, const Lanes& L, int w) {
  if constexpr (T::D == 16) { ldsm_x2(kb, rowb + L.kLane[0] + w * T::KV_HALF); }
  else {
#pragma unroll
    for (int gq = 0; gq < T::CPR / 4; ++gq) ldsm_x4(kb + 4 * gq, rowb + L.kLane[gq] + w * T::KV_HALF);
  }
}

// s_column: S of 8-key column jj of the half in ring slot sS, for the warp's rows named in `act` (bit w): the staged-bias fragment is
// loaded from shared memory once and is the C operand of every active row's first Q K^T mma (the rows share the pair bias).  Keys
// whose bit in the row's 32-bit mask word mw[w] (bit = attend) is clear become -inf afterwards (MASK = false: words not looked at).
template <class T, bool MASK>
TS_DEVI void s_column(CFrag* c, const FragT<T>* f, const Lanes& L, uint32_t sS, int jj, unsigned act, const uint32_t* mw, int t) {
  const uint32_t a = sS + ((jj & 1) ? L.bB : L.bA) + jj * 64;
  const float4 v0 = lds128f(a), v1 = lds128f(a + 2048);
  const float cb0[4] = {v0.x, v0.y, v0.z, v0.w}, cb1[4] = {v1.x, v1.y, v1.z, v1.w};
  const uint32_t rowb = sS + T::KOFF + jj * 8 * T::ROWB;
#pragma unroll
  for (int w = 0; w < T::RW; ++w) {
    if (T::RW == 1 || (act & (1u << w))) {
      uint32_t kb[2 * T::KK];
      load_kb<T>(kb, rowb, L, w);
      mma16816c(c[w][0][jj], f[w].qa[0][0], kb, cb0); mma16816c(c[w][1][jj], f[w].qa[1][0], kb, cb1);
#pragma unroll
      for (int kk = 1; kk < T::KK; ++kk) { mma16816(c[w][0][jj], f[w].qa[0][kk], kb + 2 * kk); mma16816(c[w][1][jj], f[w].qa[1][kk], kb + 2 * kk); }
      if (MASK && mw[w] != ~0u) {
        const int bit = 8 * jj + 2 * t;
        const bool k0 = (mw[w] >> bit) & 1, k1 = (mw[w] >> (bit + 1)) & 1;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          if (!k0) { c[w][mt][jj][0] = -INFINITY; c[w][mt][jj][2] = -INFINITY; }
          if (!k1) { c[w][mt][jj][1] = -INFINITY; c[w][mt][jj][3] = -INFINITY; }
        }
      }
    }
  }
}

// front: S = bias/scale + Q K^T of all four 8-key columns of the half in slot sS, rows `act` (masks applied afterwards by apply_mask).
template <class T>
TS_DEVI void front(CFrag* c, const FragT<T>* f, const Lanes& L, uint32_t sS, unsigned act, const uint32_t* mw, int t) {
#pragma unroll
  for (int jj = 0; jj < 4; ++jj) s_column<T, false>(c, f, L, sS, jj, act, mw, t);
}

// Per-key masking of one half: keys whose bit in the 32-bit word mw (bit = attend) is clear become -inf (their p is then an exact 0).
TS_DEVI void apply_mask(CFrag& c, uint32_t mw, int t) {
#pragma unroll
  for (int jl = 0; jl < 4; ++jl) {
    const int bit = 8 * jl + 2 * t;
    const bool k0 = (mw >> bit) & 1, k1 = (mw >> (bit + 1)) & 1;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      if (!k0) { c[mt][jl][0] = -INFINITY; c[mt][jl][2] = -INFINITY; }
      if (!k1) { c[mt][jl][1] = -INFINITY; c[mt][jl][3] = -INFINITY; }
    }
  }
}

// exponentials of 8-key column j of one row: p = ex2(S*c1 - m)
template <class T>
TS_DEVI void exp_column(CFrag& c, const FragT<T>& f, int j, float c1) {
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    c[mt][j][0] = ex2f(fmaf(c[mt][j][0], c1, -f.m[mt][0])); c[mt][j][1] = ex2f(fmaf(c[mt][j][1], c1, -f.m[mt][0]));
    c[mt][j][2] = ex2f(fmaf(c[mt][j][2], c1, -f.m[mt][1])); c[mt][j][3] = ex2f(fmaf(c[mt][j][3], c1, -f.m[mt][1]));
  }
}

// columns 2kl, 2kl+1 of p -> bf16 A fragments of the 16-key slice kl, and the row sums of that same bf16 P (ones-column mma into l4)
template <class T>
TS_DEVI void pack_slice(uint32_t (&pa)[2][4], const CFrag& c, FragT<T>& f, int kl, const uint32_t* ones) {
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    pa[mt][0] = pack_bf16(c[mt][2 * kl][0], c[mt][2 * kl][1]);         pa[mt][1] = pack_bf16(c[mt][2 * kl][2], c[mt][2 * kl][3]);
    pa[mt][2] = pack_bf16(c[mt][2 * kl + 1][0], c[mt][2 * kl + 1][1]); pa[mt][3] = pack_bf16(c[mt][2 * kl + 1][2], c[mt][2 * kl + 1][3]);
    mma16816(f.l4[mt], pa[mt], ones);
  }
}

// O += P V for the 16-key slice kl (0, 1) of the half in slot sS, the warp's row w, P = packed bf16 A fragments pa.
template <class T>
TS_DEVI void pv_slice(FragT<T>& f, const Lanes& L, uint32_t sS, int kl, const uint32_t (&pa)[2][4], int w) {
  const uint32_t sV = sS + T::VOFF + w * T::KV_HALF + kl * 16 * T::ROWB;
#pragma unroll
  for (int pp = 0; pp < T::DN / 2; ++pp) {
    uint32_t vb[4]; ldsm_x4_t(vb, sV + L.vLane[pp]);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) { mma16816(f.o[mt][2 * pp], pa[mt], vb); mma16816(f.o[mt][2 * pp + 1], pa[mt], vb + 2); }
  }
}

// expoback: one row, one whole half, not interleaved (partial halves, masked keys, unseeded rows, the SAFE pass): p = ex2(c*c1 - m),
// pack, row sums, O += P V.  SAFE instantiation: first raise m to the running max per row (rescaling O, l by factors <= 1).
template <class T>
TS_DEVI void expoback(CFrag& c, FragT<T>& f, const Lanes& L, uint32_t sS, float c1, const uint32_t* ones, int w) {
  if (T::SAFE) {
    float mx[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      mx[mt][0] = fmaxf(fmaxf(fmaxf(c[mt][0][0], c[mt][0][1]), fmaxf(c[mt][1][0], c[mt][1][1])), fmaxf(fmaxf(c[mt][2][0], c[mt][2][1]), fmaxf(c[mt][3][0], c[mt][3][1])));
      mx[mt][1] = fmaxf(fmaxf(fmaxf(c[mt][0][2], c[mt][0][3]), fmaxf(c[mt][1][2], c[mt][1][3])), fmaxf(fmaxf(c[mt][2][2], c[mt][2][3]), fmaxf(c[mt][3][2], c[mt][3][3])));
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        mx[mt][hh] = fmaxf(mx[mt][hh], __shfl_xor_sync(~0u, mx[mt][hh], 1));
        mx[mt][hh] = fmaxf(mx[mt][hh], __shfl_xor_sync(~0u, mx[mt][hh], 2));
        mx[mt][hh] = fmaxf(f.m[mt][hh], mx[mt][hh] * c1);                    // candidate new offset = the running max itself (its p is exactly 1); -inf halves keep m
      }
    }
    bool up = false;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) { up |= mx[mt][0] > f.m[mt][0]; up |= mx[mt][1] > f.m[mt][1]; }
    if (__any_sync(~0u, up)) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const float corr = ex2f(fmaxf(f.m[mt][hh] - mx[mt][hh], -130.f));  // <= 1 (0 from the initial -1e30)
          f.m[mt][hh] = mx[mt][hh];
          f.l4[mt][2 * hh] *= corr; f.l4[mt][2 * hh + 1] *= corr;
#pragma unroll
          for (int dn = 0; dn < T::DN; ++dn) { f.o[mt][dn][2 * hh] *= corr; f.o[mt][dn][2 * hh + 1] *= corr; }
        }
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) exp_column<T>(c, f, j, c1);
#pragma unroll
  for (int kl = 0; kl < 2; ++kl) {
    uint32_t pa[2][4];
    pack_slice<T>(pa, c, f, kl, ones);
    pv_slice<T>(f, L, sS, kl, pa, w);
  }
}

// The hot path is software-pipelined across halves so that every stretch of exponentials (MUFU) has independent tensor work next to it
// in program order (taken when every row of the warp is fully live on the halves involved and seeded).  State of a warp in pipelined
// form at the start of half u:  columns 0,1 of c = S(u), columns 2,3 = p(u-1) (exponentials taken, not yet packed), and the tail of half
// u-1 {pack + row sums of columns 2,3, O += P V of its keys 16-31 (V of slot u-1)} pending.
//   phaseA<DEF>: exps of columns 0,1 of u  ||  [DEF: that pending tail of u-1 (slot sP)] + S of columns 2,3 of u (slot sS);
//                DEF = false when entering pipelined form from the plain form (S(u) complete, nothing pending): exps of columns 0,1 only.
//   phaseB:      pack + row sums of columns 0,1, [NEXT: S of columns 0,1 of u+1 (slot sN)], O += P V of keys 0-15 of u  ||  exps of columns 2,3.
//   flush:       the tail of u and [NEXT: S of columns 2,3 of u+1]: back to the plain form (S(u+1) complete).
template <class T, bool DEF>
TS_DEVI void phaseA(CFrag* c, FragT<T>* f, const Lanes& L, uint32_t sP, uint32_t sS, float c1, const uint32_t* ones, int t) {
  constexpr unsigned ALL = (1u << T::RW) - 1u;
  uint32_t pa[T::RW][2][4];
  if (DEF) {
#pragma unroll
    for (int w = 0; w < T::RW; ++w) pack_slice<T>(pa[w], c[w], f[w], 1, ones);
  }
#pragma unroll
  for (int w = 0; w < T::RW; ++w) {
    exp_column<T>(c[w], f[w], 0, c1);
    if (DEF) pv_slice<T>(f[w], L, sP, 1, pa[w], w);
  }
#pragma unroll
  for (int w = 0; w < T::RW; ++w) {
    exp_column<T>(c[w], f[w], 1, c1);
    if (DEF) s_column<T, false>(c, f, L, sS, T::RW == 1 ? 2 : 2 + w, ALL, nullptr, t);   // RW=2: column 2 next to row 0's exps, column 3 next to row 1's
  }
  if (DEF && T::RW == 1) s_column<T, false>(c, f, L, sS, 3, ALL, nullptr, t);
}
template <class T, bool NEXT>
TS_DEVI void phaseB(CFrag* c, FragT<T>* f, const Lanes& L, uint32_t sS, uint32_t sN, float c1, const uint32_t* ones, int t) {
  constexpr unsigned ALL = (1u << T::RW) - 1u;
  uint32_t pa[T::RW][2][4];
#pragma unroll
  for (int w = 0; w < T::RW; ++w) pack_slice<T>(pa[w], c[w], f[w], 0, ones);
#pragma unroll
  for (int w = 0; w < T::RW; ++w) {
    exp_column<T>(c[w], f[w], 2, c1);
    if (NEXT) s_column<T, false>(c, f, L, sN, T::RW == 1 ? 0 : w, ALL, nullptr, t);
  }
  if (NEXT && T::RW == 1) s_column<T, false>(c, f, L, sN, 1, ALL, nullptr, t);
#pragma unroll
  for (int w = 0; w < T::RW; ++w) {
    exp_column<T>(c[w], f[w], 3, c1);
    pv_slice<T>(f[w], L, sS, 0, pa[w], w);
  }
}
template <class T, bool NEXT>
TS_DEVI void flush(CFrag* c, FragT<T>* f, const Lanes& L, uint32_t sS, uint32_t sN, const uint32_t* ones, int t) {
  constexpr unsigned ALL = (1u << T::RW) - 1u;
  uint32_t pa[T::RW][2][4];
#pragma unroll
  for (int w = 0; w < T::RW; ++w) pack_slice<T>(pa[w], c[w], f[w], 1, ones);
#pragma unroll
  for (int w = 0; w < T::RW; ++w) {
    pv_slice<T>(f[w], L, sS, 1, pa[w], w);
    if (NEXT) s_column<T, false>(c, f, L, sN, T::RW == 1 ? 2 : 2 + w, ALL, nullptr, t);
  }
  if (NEXT && T::RW == 1) s_column<T, false>(c, f, L, sN, 3, ALL, nullptr, t);
}

// Seeding (PORTING.md 2): a row takes its offset ONCE, from the first half tile holding a finite logit for it: m = max * c1 + SHIFT
// (its largest term is then exactly 2^-SHIFT); never re-seeded, never seeded from a floor.
template <class T>
TS_DEVI void rebase(const CFrag& c, FragT<T>& f, float c1) {
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    float mx[2];
    mx[0] = fmaxf(fmaxf(fmaxf(c[mt][0][0], c[mt][0][1]), fmaxf(c[mt][1][0], c[mt][1][1])), fmaxf(fmaxf(c[mt][2][0], c[mt][2][1]), fmaxf(c[mt][3][0], c[mt][3][1])));
    mx[1] = fmaxf(fmaxf(fmaxf(c[mt][0][2], c[mt][0][3]), fmaxf(c[mt][1][2], c[mt][1][3])), fmaxf(fmaxf(c[mt][2][2], c[mt][2][3]), fmaxf(c[mt][3][2], c[mt][3][3])));
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(~0u, mx[hh], 1));
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(~0u, mx[hh], 2));
      const uint32_t bit = 1u << (2 * mt + hh);
      if (!(f.seeded & bit) && mx[hh] > -INFINITY && mx[hh] < INFINITY) { f.m[mt][hh] = mx[hh] * c1 + (float)SHIFT; f.seeded |= bit; }
    }
  }
}

// Exact power-of-two renormalisation of the rows whose running sum left the window (rare): m += k, O *= 2^-k, l *= 2^-k -- and, in
// pipelined form (PEND), the exponentials of columns 2,3 still held in c (taken with the old m, not yet summed / multiplied into O).
template <class T>
TS_DEVI void renormalise(FragT<T>& f, CFrag& c, bool pend) {
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const float l = f.l4[mt][2 * hh];
      if (l > 0.f && (__float_as_uint(l) - RN_LO) > RN_WIDTH) {   // outside the window (inf included: k clamps, the row fails validation)
        const int k = min(max((int)((__float_as_uint(l) >> 23) & 0xff) - 127 + SHIFT, -126), 126);   // 2^-k a normal power of two
        const float corr = __uint_as_float((uint32_t)(127 - k) << 23);
        f.m[mt][hh] += (float)k;
        f.l4[mt][2 * hh] *= corr; f.l4[mt][2 * hh + 1] *= corr;
#pragma unroll
        for (int dn = 0; dn < T::DN; ++dn) { f.o[mt][dn][2 * hh] *= corr; f.o[mt][dn][2 * hh + 1] *= corr; }
        if (pend) { c[mt][2][2 * hh] *= corr; c[mt][2][2 * hh + 1] *= corr; c[mt][3][2 * hh] *= corr; c[mt][3][2 * hh + 1] *= corr; }
      }
    }
}

// Fully-masked pair row (uniform attention over the S keys): out[q, :] = mean_k v[k, :] -- the same D values for every query q of the
// row -- and lse = log2(S).  The value is q-independent, so only the CTA of q tile 0 produces the row: within its row set, row wr of a
// warp's pair is served by the QG/RW warps wq == wr (mod RW), each computing the mean once (the per-lane key split, the sequential fp32
// sums and the butterfly below ARE the arithmetic of record -- unchanged) and writing every (QG/RW)-th 32-query slice of the whole row;
// the CTAs of q tiles >= 1 have nothing to do for such a row.  A dead row costs one pass over v per (row, head, serving warp).
template <class T>
TS_DEVI void uniform_rows_out(const Args& p, int b, int i, int h, int qt, int wq, int wr, int lane) {
  constexpr int D2 = T::D / 2, LP = 32 / D2;                      // d pairs; lanes per d pair (keys split LP ways)
  constexpr int WPR = T::QG / T::RW;                              // warps serving one uniform row (its 32-query slices interleaved WPR ways)
  static_assert(T::QG % T::RW == 0, "uniform rows: the q-group warps of a row set split evenly over the warp's rows");
  if (qt != 0 || (wq % T::RW) != wr) return;
  const int dp = lane % D2, part = lane / D2;
  const __nv_bfloat16* vb = p.v + (long long)b * p.v_sB + (long long)i * p.v_sN + (long long)h * p.v_sH + 2 * dp;
  float a0 = 0.f, a1 = 0.f;
#pragma unroll 8
  for (int k = part; k < p.S_kv; k += LP) {                       // (unrolled for loads in flight; the sums stay sequential in k)
    const uint32_t u = __ldg(reinterpret_cast<const uint32_t*>(vb + (long long)k * p.v_sS));
    a0 += __uint_as_float(u << 16); a1 += __uint_as_float(u & 0xffff0000u);
  }
#pragma unroll
  for (int o = D2; o < 32; o <<= 1) { a0 += __shfl_xor_sync(~0u, a0, o); a1 += __shfl_xor_sync(~0u, a1, o); }
  const float inv = 1.f / (float)p.S_kv;
  const uint32_t w = pack_bf16(a0 * inv, a1 * inv);
  __nv_bfloat16* ob = p.out + (long long)b * p.o_sB + (long long)i * p.o_sN + (long long)h * p.o_sH + 2 * dp;
  for (int q0 = (wq / T::RW) * 32; q0 < p.S_q; q0 += WPR * 32) { // this warp's 32-query slices of the row
#pragma unroll
    for (int r = part; r < 32; r += LP) {                         // LP query rows per pass: every lane stores its d pair
      const int q = q0 + r;
      if (q < p.S_q) *reinterpret_cast<uint32_t*>(ob + (long long)q * p.o_sS) = w;
    }
    if (T::LSE) { const int q = q0 + lane; if (p.lse != nullptr && q < p.S_q) p.lse[(((long long)b * p.N + i) * p.H + h) * (long long)p.S_q + q] = log2f((float)p.S_kv); }
  }
}

// 32-key mask word of keys [kbase, kbase+32) for a live interval [a, e): bit j set iff a <= kbase + j < e.
TS_DEVI uint32_t interval_word(int a, int e, int kbase) {
  const int lo = min(max(a - kbase, 0), 32), hi = min(max(e - kbase, 0), 32);
  const uint32_t mhi = hi >= 32 ? ~0u : ((1u << hi) - 1u), mlo = lo >= 32 ? ~0u : ((1u << lo) - 1u);
  return hi > lo ? (mhi & ~mlo) : 0u;
}

// ------------------------------------------------------------------------------------------------------------------ half-tile loader
// Cooperative cp.async fill of one ring slot with half tile u = keys [32 u, 32 u + 32): the [BM x 32-key] fp32 bias box and, per pair
// row of the CTA, the [32 x D] K and V tiles (rows XOR-swizzled at 16 B: chunk c of key n lands at chunk c ^ kv_swz(n)).  Every thread
// owns a fixed set of 16-byte chunks whose global and shared addresses are affine in u, so the per-half work is a pointer advance plus
// one predicate per chunk: bias chunk (row tid/8 + it*THREADS/8, 16-byte column tid%8), and K/V chunk (key n0, 16-byte column kc) of
// pair rows r0 + it*RPI.  Halves are issued in increasing consecutive order.  Out-of-range pieces (bias rows beyond the staged tensor,
// keys >= S_kv, absent pair rows) are zero-filled (src-size 0: nothing is read).
template <class T>
TS_DEVI uint32_t kv_swz(int n) { return (uint32_t)((n / T::RP) & (T::CPR - 1)); }

template <class T>
struct Loader {
  static constexpr int RSTEP = T::THREADS / 8;                            // staged-bias rows between a thread's successive bias chunks
  static constexpr int NBI = (T::BM + RSTEP - 1) / RSTEP;                 // bias chunks per thread
  static constexpr int CH = BH * T::CPR;                                  // K (or V) chunks per pair row per half
  static constexpr int CPT = (CH + T::THREADS - 1) / T::THREADS;          // chunks per thread per pair row (> 1 only when the CTA has fewer threads than a half has chunks: 4-warp CTAs at D=64)
  static constexpr int RPI = CPT > 1 ? 1 : T::THREADS / CH;               // pair rows covered per pass of the CTA
  static constexpr int NKI = (T::R + RPI - 1) / RPI;                      // row passes per thread per half (K; same for V)
  static constexpr int KSTEP = T::THREADS / T::CPR;                       // keys between a thread's successive chunks of one row (CPT > 1)
  // A geometry whose bias rows per pass (RSTEP) do not divide BM, or whose K/V rows per pass (RPI) do not divide R, has threads whose last
  // chunk lies beyond the bias box / beyond the CTA's R rows: such chunks do not exist (no cp.async at all -- a zero-fill there would land on
  // the next region of the slot).  okbits 16+it / 24+it = "bias / K-V chunk `it` of this thread exists"; the test folds away (BALL / KALL) for
  // every geometry that tiles exactly (all of them but (WR, RW, QG) = (2, 1, 4) at D16, whose RPI = 4 > R = 2).
  static constexpr bool BALL = (T::BM % RSTEP == 0), KALL = (T::R % RPI == 0);
  static_assert(NBI <= 8 && NKI <= 8, "chunk predicate bits");
  static_assert(CPT == 1 || (CH % T::THREADS == 0 && KSTEP % (T::RP * T::CPR) == 0 && CPT * KSTEP == BH), "multi-chunk K/V mapping (same swizzle phase, whole half)");
  const char* bsrc; const char* ksrc; const char* vsrc;                   // this thread's chunk of the NEXT half to issue (bytes)
  uint32_t bdst, kdst;                                                    // this thread's smem byte offsets within a slot (bias; K, V add KOFF/VOFF)
  uint32_t okbits;                                                        // bit it: bias row of chunk it is inside the staged tensor; bit 8+it: pair row of K/V chunk it present
  int key;                                                                // key index (absolute) of this thread's K/V chunk of the next half

  TS_DEVI void init(const Args& p, int b, int h, int i0, int q_base, int u0, int tid) {
    const int br = tid >> 3, bc = tid & 7;
    const int rows_ok = min(T::BM, p.S128 - q_base);
    okbits = 0u;
#pragma unroll
    for (int it = 0; it < NBI; ++it) okbits |= ((uint32_t)(br + RSTEP * it < rows_ok) << it) | ((uint32_t)(br + RSTEP * it < T::BM) << (16 + it));
    bsrc = reinterpret_cast<const char*>(p.bias + (((long long)b * p.H + h) * p.S128 + q_base + br) * (long long)p.S64 + (long long)u0 * BH + bc * 4);
    bdst = (uint32_t)(br * 128 + bc * 16);
    const int r0 = tid / CH, rem = tid % CH, n0 = rem / T::CPR, kc = rem % T::CPR, nrows = min(T::R, p.N - i0);
#pragma unroll
    for (int it = 0; it < NKI; ++it) okbits |= ((uint32_t)(r0 + it * RPI < nrows) << (8 + it)) | ((uint32_t)(r0 + it * RPI < T::R) << (24 + it));
    key = u0 * BH + n0;
    kdst = (uint32_t)(r0 * T::KV_HALF + n0 * T::ROWB) + ((((uint32_t)kc) ^ kv_swz<T>(n0)) << 4);
    ksrc = reinterpret_cast<const char*>(p.k + (long long)b * p.k_sB + (long long)(i0 + r0) * p.k_sN + (long long)h * p.k_sH + (long long)key * p.k_sS + 8 * kc);
    vsrc = reinterpret_cast<const char*>(p.v + (long long)b * p.v_sB + (long long)(i0 + r0) * p.v_sN + (long long)h * p.v_sH + (long long)key * p.v_sS + 8 * kc);
  }
  // issue this thread's chunks of the next half into the slot at shared address s0, then advance to the half after it
  TS_DEVI void load(const Args& p, uint32_t s0) {
    const long long brow = (long long)RSTEP * p.S64 * 4;                  // bytes between this thread's bias chunks (RSTEP staged rows)
#pragma unroll
    for (int it = 0; it < NBI; ++it)
      if (BALL || ((okbits >> (16 + it)) & 1u)) cp_async16(s0 + bdst + it * (RSTEP * 128), bsrc + it * brow, (okbits >> it) & 1u);   // (chunks beyond the box do not exist)
    const long long krow = (long long)RPI * p.k_sN * 2, vrow = (long long)RPI * p.v_sN * 2;   // bytes between this thread's pair rows
    if constexpr (CPT == 1) {
      const bool kok = key < p.S_kv;
#pragma unroll
      for (int it = 0; it < NKI; ++it) {
        if (!KALL && !((okbits >> (24 + it)) & 1u)) continue;                                  // (chunks beyond the CTA's R rows do not exist)
        const bool ok = kok && ((okbits >> (8 + it)) & 1u);
        const uint32_t so = s0 + kdst + it * RPI * T::KV_HALF;
        cp_async16(so + T::KOFF, ksrc + it * krow, ok);
        cp_async16(so + T::VOFF, vsrc + it * vrow, ok);
      }
    } else {                                                              // a thread owns CPT chunks of every row, KSTEP keys apart (same swizzle phase)
      const long long kkey = (long long)KSTEP * p.k_sS * 2, vkey = (long long)KSTEP * p.v_sS * 2;
#pragma unroll
      for (int it = 0; it < NKI; ++it) {
        if (!KALL && !((okbits >> (24 + it)) & 1u)) continue;
        const bool rok = (okbits >> (8 + it)) & 1u;
#pragma unroll
        for (int j = 0; j < CPT; ++j) {
          const bool ok = rok && (key + j * KSTEP < p.S_kv);
          const uint32_t so = s0 + kdst + it * RPI * T::KV_HALF + j * (KSTEP * T::ROWB);
          cp_async16(so + T::KOFF, ksrc + it * krow + j * kkey, ok);
          cp_async16(so + T::VOFF, vsrc + it * vrow + j * vkey, ok);
        }
      }
    }
    bsrc += BH * 4; ksrc += (long long)BH * p.k_sS * 2; vsrc += (long long)BH * p.v_sS * 2; key += BH;
  }
};

// ------------------------------------------------------------------------------------------------------------------ CTA tile
// One CTA tile: R pair rows (yg*R ..) x BM queries (qt) of (b, h) = bz; warp = (row set rs, query group wq) owns queries
// q_base + 32 wq .. +31 of pair rows i0 + rs*RW + w, w < RW.  The key axis is streamed in 32-key halves through a ring of STAGES slots
// filled LEAD = STAGES-2 halves ahead: per half one cp.async.wait_group + one __syncthreads (all threads' copies of half u+1 visible; every
// warp done with half u-1, whose slot takes half u+LEAD+1).  Returns (per thread) whether all its rows came out finite with a positive
// sum (absent and fully-masked rows count as fine).  Every thread of the CTA must call it (barriers inside).
template <class T>
TS_DEVI bool cta_tile(const Args& p, SmemT<T>& sm, int qt, int yg, int bz) {
  constexpr int R = T::R, RW = T::RW, QG = T::QG, ST = T::STAGES;
  static_assert(smem_fits<T>(), "SmemT must fit the dynamic shared memory the launch asks for");
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rs = warp / QG, wq = warp % QG, rr0 = rs * RW;      // this warp's first row within the CTA tile
  const int i0 = yg * R, b = bz / p.H, h = bz % p.H;
  const int q_base = qt * T::BM, S = p.S_kv, nh = (S + BH - 1) / BH, nkt = p.n_ktiles;
  const bool renorm_on = !(p.dbg & 32);

  // ---- per-row live half ranges; CTA stream = union over the rows that stream --------------------------------------------------
  // GENERAL = false routine: present rows attend [0, S) -- or, with a table of plain rows (rows_plain), one 64-aligned span [a, e) -- every
  // half of the stream FULL, absent rows (i >= N) nothing; per-row registers, the classes never change inside the stream.  The general
  // routine: sm.rowtab[r] = {a, e, kind, 0} for pair row i0 + r: live keys [a, e)
  // (e clamped to S), kind INTERVAL / RAGGED / UNIFORM (mask_rows_kernel), kind -1 for an absent row; filled by threads r < R.  Either way
  // the CTA's half range [U0, U1) is the union of the streaming rows' ranges (uniform / absent / empty rows stream nothing).
  int ra[RW], re[RW], ta[RW], te[RW]; bool present[RW];         // (the GENERAL = false routine's row state; the general routine reads sm.rowtab)
#pragma unroll
  for (int w = 0; w < RW; ++w) {
    const int i = i0 + rr0 + w;
    present[w] = i < p.N;
    ra[w] = 0; re[w] = S;                                        // no mask table: every row attends [0, S)
    if (!T::GENERAL && p.rows != nullptr && present[w]) {         // (GENERAL = false with a table: the caller checked the rows are plain, see rows_plain)
      const int4 ri = p.rows[(long long)b * p.N + i]; ra[w] = ri.x; re[w] = min(ri.y, S);
    }
    const bool streams = present[w] && re[w] > ra[w];
    ta[w] = streams ? (ra[w] / BH) : 0; te[w] = streams ? ((re[w] + BH - 1) / BH) : 0;   // this row's half range [ta, te)
  }
  __syncthreads();                                               // (persistent CTAs: the previous tile is done with sm)
  if (tid == 0) { sm.flagged = 0; sm.kt_lo = nh; sm.kt_hi = 0; }
  __syncthreads();
  if constexpr (T::GENERAL) {
    if (tid < R) {
      const int i = i0 + tid;
      int4 ri = make_int4(0, S, i < p.N ? ROW_INTERVAL : -1, 0);
      if (p.rows != nullptr && i < p.N) { const int4 g4 = p.rows[(long long)b * p.N + i]; ri.x = g4.x; ri.y = min(g4.y, S); ri.z = g4.z; }
      sm.rowtab[tid] = ri;
      if (ri.z >= 0 && ri.z != ROW_UNIFORM && ri.y > ri.x) { atomicMin(&sm.kt_lo, ri.x / BH); atomicMax(&sm.kt_hi, (ri.y + BH - 1) / BH); }
    }
  } else if (lane == 0) {
#pragma unroll
    for (int w = 0; w < RW; ++w) if (te[w] > ta[w]) { atomicMin(&sm.kt_lo, ta[w]); atomicMax(&sm.kt_hi, te[w]); }
  }
  __syncthreads();
  const int U0 = sm.kt_lo, U1 = sm.kt_hi;                        // CTA stream of halves [U0, U1) (empty when no row of the CTA has a live key)
  const uint32_t s00 = smem_u32(sm.slot[0]);
  // Class + 32-key mask word (bit = attend) of the warp's row w on half v (warp-uniform), and `next` = the first half after v at which that
  // class can change.  SKIP (word 0) outside the row's live half range [ta, te) and for absent / uniform rows; an INTERVAL row is FULL (word ~0)
  // on every half of its range except one holding an interval end strictly inside it (MIXED with the interval's word; an end at S needs no
  // word: keys >= S are -inf bias columns), so its class changes only at ta, ta+1, te-1, te; a RAGGED row takes its staged word on every half
  // (SKIP if 0, FULL if ~0, else MIXED).  A GENERAL stream re-classifies a warp's rows only at the warp's next change point `nc`: in between
  // every row is FULL or SKIP, the classes carry over and no row state is looked at (the row table lives in shared memory, not registers).
  auto classify = [&](int w, int v, uint32_t& word, int& next) -> int {
    next = 0x7fffffff;
    if constexpr (!T::GENERAL) {
      word = 0u;
      if (v < ta[w] || v >= te[w]) return CLS_SKIP;
      word = ~0u;
      return CLS_FULL;
    } else {
      const int4 ri = sm.rowtab[rr0 + w];
      word = 0u;
      if (ri.z < 0 || ri.z == ROW_UNIFORM || ri.y <= ri.x) return CLS_SKIP;
      const int tlo = ri.x / BH, thi = (ri.y + BH - 1) / BH;
      if (v < tlo) { next = tlo; return CLS_SKIP; }
      if (v >= thi) return CLS_SKIP;
      if (ri.z == ROW_RAGGED) {
        next = v + 1;
        word = __ldg(p.maskw + ((long long)b * p.N + i0 + rr0 + w) * (2ll * nkt) + v);
        return word == 0u ? CLS_SKIP : (word == ~0u ? CLS_FULL : CLS_MIXED);
      }
      const bool bs = (ri.x & (BH - 1)) != 0, be = (ri.y & (BH - 1)) != 0 && ri.y < S;   // interval start / end inside a half
      next = (be && v < thi - 1) ? thi - 1 : thi;
      if (bs && v == tlo) next = min(next, tlo + 1);
      word = ~0u;
      if (!((bs && v == tlo) || (be && v == thi - 1))) return CLS_FULL;
      word = interval_word(ri.x, ri.y, v * BH);
      return CLS_MIXED;
    }
  };

  // ---- loader (this thread's chunk addresses, int64; advanced half by half) ----------------------------------------------------
  Loader<T> ld;
  ld.init(p, b, h, i0, q_base, U0, tid);
  const bool exp_nofill = p.dbg & 64, exp_nobar = p.dbg & 1024;      // timing experiments only: no ring refills (stale tiles) / no barriers
  auto slot = [&](int u) { return s00 + (uint32_t)(((u - U0) % ST) * T::SLOT_BYTES); };
  auto issue = [&](int u) { if (u < U1 && !(exp_nofill && u >= U0 + ST)) ld.load(p, slot(u)); cp_commit(); };   // one commit group per call (empty beyond the stream)

  // ---- per-warp constants ---------------------------------------------------------------------------------------------------------
  Lanes L;
  {
    const int l7 = lane & 7, mi = lane >> 3;
    const uint32_t swz = kv_swz<T>(l7);
#pragma unroll
    for (int gq = 0; gq < 2; ++gq) L.kLane[gq] = rr0 * T::KV_HALF + l7 * T::ROWB + ((((uint32_t)(T::D == 16 ? (mi & 1) : (4 * gq + mi))) ^ swz) << 4);
#pragma unroll
    for (int pp = 0; pp < 4; ++pp) L.vLane[pp] = rr0 * T::KV_HALF + ((mi & 1) * 8 + l7) * T::ROWB + ((((uint32_t)(2 * pp + (mi >> 1))) ^ swz) << 4);
    L.bA = wq * 4096 + g * 256 + t * 16 + (g & 1) * 64; L.bB = L.bA - (g & 1) * 128;   // = ... + ((jl ^ (g & 1)) * 64) - 64 jl
    L.bA = opaque(L.bA); L.bB = opaque(L.bB); L.kLane[0] = opaque(L.kLane[0]); L.kLane[1] = opaque(L.kLane[1]);
#pragma unroll
    for (int pp = 0; pp < 4; ++pp) L.vLane[pp] = opaque(L.vLane[pp]);
  }
  const float c1 = p.c1;
  const uint32_t ones[2] = {0x3f803f80u, 0x3f803f80u};
  FragT<T> f[RW]; CFrag cb[RW];
#pragma unroll
  for (int w = 0; w < RW; ++w) {
    f[w].seeded = 0u;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      f[w].m[mt][0] = f[w].m[mt][1] = T::SAFE ? -1e30f : 0.f;
      f[w].l4[mt][0] = f[w].l4[mt][1] = f[w].l4[mt][2] = f[w].l4[mt][3] = 0.f;
#pragma unroll
      for (int dn = 0; dn < T::DN; ++dn) f[w].o[mt][dn][0] = f[w].o[mt][dn][1] = f[w].o[mt][dn][2] = f[w].o[mt][dn][3] = 0.f;
#pragma unroll
      for (int jl = 0; jl < 4; ++jl) cb[w][mt][jl][0] = cb[w][mt][jl][1] = cb[w][mt][jl][2] = cb[w][mt][jl][3] = -INFINITY;
    }
    // Q of this warp's 32 queries of row w -> mma A fragments, straight from global (overlaps the ring fill)
    const bool deadw = T::GENERAL && sm.rowtab[rr0 + w].z == ROW_UNIFORM;   // (row table published above)
    const __nv_bfloat16* qbp = p.q + (long long)b * p.q_sB + (long long)(present[w] ? i0 + rr0 + w : 0) * p.q_sN + (long long)h * p.q_sH;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int qrow = q_base + wq * 32 + mt * 16 + hh * 8 + g;
        const bool okq = present[w] && !deadw && qrow < p.S_q;       // (a fully-masked row runs no mma: no Q)
        const uint32_t* src = reinterpret_cast<const uint32_t*>(qbp + (long long)(okq ? qrow : 0) * p.q_sS + 2 * t);
#pragma unroll
        for (int kk = 0; kk < T::KK; ++kk) { f[w].qa[mt][kk][hh] = okq ? __ldg(src + 8 * kk) : 0u; f[w].qa[mt][kk][2 + hh] = okq ? __ldg(src + 8 * kk + 4) : 0u; }
      }
  }

  if (U1 > U0) {
  // ---- prologue: LEAD+1 halves in flight, wait for the first, S of it (plain form) -----------------------------------------------
  for (int st = 0; st <= T::LEAD; ++st) issue(U0 + st);
  cp_wait<T::LEAD>();
  __syncthreads();
  int clsC[RW]; uint32_t mwC[RW]; unsigned actC = 0; bool allC = true; int nc = 0x7fffffff;   // classes on half U0; nc = this warp's next change point
#pragma unroll
  for (int w = 0; w < RW; ++w) { int nx; clsC[w] = classify(w, U0, mwC[w], nx); nc = min(nc, nx); actC |= (unsigned)(clsC[w] != CLS_SKIP) << w; allC &= clsC[w] == CLS_FULL; }
  front<T>(cb, f, L, slot(U0), actC, mwC, t);
#pragma unroll
  for (int w = 0; w < RW; ++w) if (clsC[w] == CLS_MIXED) apply_mask(cb[w], mwC[w], t);
  bool unseeded = !T::SAFE;                                      // some row of this warp saw no finite logit yet (the SAFE pass keeps a true max instead)
  bool pipe = false;                                             // warp state is in pipelined form (see phaseA/phaseB)

  // ---- the stream --------------------------------------------------------------------------------------------------------------------
  // body<KIND>(u) consumes half u and starts half u+1.  KIND 0 = steady: the classes of this warp's rows carry over to u+1 (FULL / SKIP) and
  // no row state is touched; 1 = event: u+1 == nc, the rows are re-classified on u+1 (row table, mask words) and nc advances; 2 = last half
  // of the CTA stream.  Without a mask table there are no change points inside a stream (present rows attend [0, S), absent rows nothing).
  // A GENERAL stream in the 256-thread geometry runs steady bodies between change points and an event body at each (SEG: every warp makes
  // U1 - U0 body calls, one NON-ALIGNED barrier each, in its own steady / event rhythm); in the 128-thread geometry (two CTAs per SM, the tighter
  // register budget) it keeps one body that re-classifies under a rarely-taken branch instead of carrying a third copy of the loop.  The
  // mask-free stream classifies every half from its per-row registers (two compares; FULL / SKIP only).
  constexpr bool SEG = T::GENERAL && T::THREADS >= 256;
  auto body = [&](int u, auto kind_tag) {
    constexpr int KIND = decltype(kind_tag)::value;
    constexpr bool LAST = KIND == 2, EVENT = KIND == 1;
    const uint32_t sP = slot(u - 1), sS = slot(u), sN = slot(u + 1);
    if (!pipe && unseeded) {                                     // plain form holds all of S(u): seed rows that see their first finite logit
      unsigned un = 0;
#pragma unroll
      for (int w = 0; w < RW; ++w) { if (clsC[w] != CLS_SKIP) rebase<T>(cb[w], f[w], c1); un |= f[w].seeded != 15u; }
      unseeded = __any_sync(~0u, un != 0);
    }
    if (!T::SAFE) {                                              // (r) renormalisation: never for the SAFE pass (true max, l in [1, S])
      bool rn = false;
#pragma unroll
      for (int w = 0; w < RW; ++w)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) rn |= (__float_as_uint(f[w].l4[mt][2 * hh]) - RN_LO) > RN_WIDTH;
      if (__any_sync(~0u, rn) && renorm_on) {
#pragma unroll
        for (int w = 0; w < RW; ++w) renormalise<T>(f[w], cb[w], pipe);
      }
    }
    const bool fast = !T::SAFE && !unseeded && allC;             // every row of the warp fully live on half u and seeded: pipelined form
    // A: first part of half u (reads V of slot u-1 when in pipelined form -- before the barrier that frees that slot)
    if (pipe) phaseA<T, true>(cb, f, L, sP, sS, c1, ones, t);
    else if (fast) phaseA<T, false>(cb, f, L, sP, sS, c1, ones, t);
    else {
#pragma unroll
      for (int w = 0; w < RW; ++w) if (clsC[w] != CLS_SKIP) expoback<T>(cb[w], f[w], L, sS, c1, ones, w);
    }
    // half u+1 landed (this thread's copies; LEAD-1 younger groups may pend), the barrier publishes it and frees slot u-1; refill it
    if (!LAST) cp_wait<T::LEAD - 1>();
    if (!exp_nobar) { if constexpr (SEG) cta_sync_any(); else __syncthreads(); }   // (SEG: warps arrive here from different body<KIND> instantiations)
    issue(u + T::LEAD + 1);
    int clsN[RW]; uint32_t mwN[RW]; unsigned actN = actC; bool allN = allC;   // classes on half u+1
    if (LAST) {
#pragma unroll
      for (int w = 0; w < RW; ++w) { clsN[w] = CLS_SKIP; mwN[w] = 0u; }
      actN = 0; allN = false;
    } else if constexpr (!T::GENERAL) {                          // mask-free: FULL / SKIP by the row's half range
      actN = 0; allN = true;
#pragma unroll
      for (int w = 0; w < RW; ++w) { int nx; clsN[w] = classify(w, u + 1, mwN[w], nx); actN |= (unsigned)(clsN[w] != CLS_SKIP) << w; allN &= clsN[w] == CLS_FULL; }
    } else {                                                     // steady: the classes -- FULL or SKIP -- carry over ...
#pragma unroll
      for (int w = 0; w < RW; ++w) { const bool on = (actC >> w) & 1u; clsN[w] = on ? CLS_FULL : CLS_SKIP; mwN[w] = on ? ~0u : 0u; }
      if (EVENT || (!SEG && u + 1 >= nc)) {                       // ... except at a change point: re-classify on u+1, advance nc
        int ncn = 0x7fffffff; actN = 0; allN = true;
#pragma unroll
        for (int w = 0; w < RW; ++w) { int nx; clsN[w] = classify(w, u + 1, mwN[w], nx); ncn = min(ncn, nx); actN |= (unsigned)(clsN[w] != CLS_SKIP) << w; allN &= clsN[w] == CLS_FULL; }
        nc = ncn;
      }
    }
    const bool fastN = !T::SAFE && !unseeded && allN;            // half u+1 will be taken in pipelined form (rows stay seeded)
    // B: rest of half u (+ S of half u+1)
    if (pipe || fast) {
      phaseB<T, !LAST>(cb, f, L, sS, sN, c1, ones, t);
      if (fastN) pipe = true;
      else {                                                     // leave the pipelined form: finish half u, complete S(u+1), mask it
        flush<T, !LAST>(cb, f, L, sS, sN, ones, t); pipe = false;
#pragma unroll
        for (int w = 0; w < RW; ++w) if (clsN[w] == CLS_MIXED) apply_mask(cb[w], mwN[w], t);
      }
    } else {
      if (!LAST) {
        front<T>(cb, f, L, sN, actN, mwN, t);
#pragma unroll
        for (int w = 0; w < RW; ++w) if (clsN[w] == CLS_MIXED) apply_mask(cb[w], mwN[w], t);
      }
      pipe = false;
    }
#pragma unroll
    for (int w = 0; w < RW; ++w) { clsC[w] = clsN[w]; mwC[w] = mwN[w]; }
    actC = actN; allC = allN;
  };
  if constexpr (SEG) {
    int u = U0;
    for (;;) {
      const int ulim = min(nc, U1) - 1;                          // steady while u+1 < nc (and u+1 < U1)
      for (; u < ulim; ++u) body(u, std::integral_constant<int, 0>{});
      if (u >= U1 - 1) break;
      body(u, std::integral_constant<int, 1>{}); ++u;             // u+1 == nc
    }
  } else {
    for (int u = U0; u < U1 - 1; ++u) body(u, std::integral_constant<int, 0>{});
  }
  body(U1 - 1, std::integral_constant<int, 2>{});
  }   // U1 > U0

  // ---- output: O / l -> bf16, straight from registers (4-byte stores); validity; lse; uniform rows ------------------------------
  bool ok = true;
#pragma unroll
  for (int w = 0; w < RW; ++w) {
    const int i = i0 + rr0 + w;
    if (T::GENERAL && sm.rowtab[rr0 + w].z == ROW_UNIFORM) { uniform_rows_out<T>(p, b, i, h, qt, wq, w, lane); continue; }   // fully-masked row: written once, by the serving warps of the q-tile-0 CTA (see uniform_rows_out)
    __nv_bfloat16* ob = p.out + (long long)b * p.o_sB + (long long)(present[w] ? i : 0) * p.o_sN + (long long)h * p.o_sH;
    bool okw = true;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float l = f[w].l4[mt][2 * hh];
        const float inv = l > 0.f ? 1.f / l : 0.f;
        const int qrow = q_base + wq * 32 + mt * 16 + hh * 8 + g;
        float chk = l;
        uint32_t ow[T::DN];
#pragma unroll
        for (int dn = 0; dn < T::DN; ++dn) {
          chk += (f[w].o[mt][dn][2 * hh] + f[w].o[mt][dn][2 * hh + 1]) * 0.f;
          ow[dn] = pack_bf16(f[w].o[mt][dn][2 * hh] * inv, f[w].o[mt][dn][2 * hh + 1] * inv);
        }
        okw &= (chk > 0.f) && (chk < INFINITY);                    // false for l == 0, inf, or any nan/inf in O
        if (present[w] && qrow < p.S_q) {
          uint32_t* dst = reinterpret_cast<uint32_t*>(ob + (long long)qrow * p.o_sS + 2 * t);
#pragma unroll
          for (int dn = 0; dn < T::DN; ++dn) dst[4 * dn] = ow[dn];
          if (T::LSE) {                                            // log2-sum-exp in the kernel's log2 units: sum_k 2^(x_k) = l * 2^m
            if (p.lse != nullptr && t == 0) p.lse[(((long long)b * p.N + i) * p.H + h) * (long long)p.S_q + qrow] = l > 0.f ? f[w].m[mt][hh] + log2f(l) : -INFINITY;
          }
        }
      }
    ok &= okw || !present[w];
  }
  return ok;
}

// ------------------------------------------------------------------------------------------------------------------ kernels
// Hot pass: one CTA per (q tile, row group, b*H+h).  CTA order (hardware launches x fastest): within each (b, h) the q tiles are walked
// in blocks of qb -- q tile fastest inside the block (the CTAs of one row group are adjacent and share K/V), all row groups, then the
// next block -- so the staged-bias tiles the CTAs in flight share stay L2-resident at any N.
// The rows of CTA row group yg of batch element b are "plain" when every present one is an interval row over one and the same 64-aligned
// key-tile span with no interval end inside a tile (regular rows, rows equal to the batch's live-key set, e.g. one padding length per
// batch element): such a CTA tile is computed by the GENERAL = false tile routine over that span (identical arithmetic, none of the
// per-half row bookkeeping), inside the general kernel.  Uniform-only / ragged / true-interval groups take the general routine.
template <class T>
TS_DEVI bool rows_plain(const Args& p, int yg, int bz) {
  if (p.rows == nullptr) return true;
  const int b = bz / p.H, i0 = yg * T::R;
  bool ok = true; int x0 = -1, y0 = -1;
#pragma unroll
  for (int r = 0; r < T::R; ++r) {
    const int i = i0 + r;
    if (i < p.N) {
      const int4 ri = p.rows[(long long)b * p.N + i];
      const bool span_ok = ri.z == ROW_INTERVAL && (ri.x & (BN - 1)) == 0 && ((ri.y & (BN - 1)) == 0 || ri.y >= p.S_kv) && ri.y > ri.x;
      if (x0 < 0) { x0 = ri.x; y0 = ri.y; }
      ok = ok && span_ok && ri.x == x0 && ri.y == y0;
    }
  }
  return ok;
}

template <class T>
__global__ void __launch_bounds__(T::THREADS, T::MINB) fwd_kernel(const Args p) {
  extern __shared__ uint8_t smem_raw[];
  SmemT<T>& sm = *reinterpret_cast<SmemT<T>*>((reinterpret_cast<uintptr_t>(smem_raw) + 127) & ~uintptr_t(127));
  const int QT = gridDim.x, YG = gridDim.y;
  const int lin = blockIdx.x + QT * blockIdx.y, bz = blockIdx.z;
  const int blk = lin / (p.qb * YG), r2 = lin - blk * (p.qb * YG);
  const int qbn = min(p.qb, QT - blk * p.qb);                    // q tiles in this block (the last one may be short)
  const int yg = r2 / qbn, qt = blk * p.qb + (r2 - yg * qbn);
  bool ok;
  if constexpr (T::GENERAL) {
    using PT = Traits<T::D, T::WR, T::RW, T::QG, T::STAGES, T::SAFE, false, T::LSE, T::MINB>;   // same geometry and smem layout, plain tile routine
    static_assert(sizeof(SmemT<PT>) == sizeof(SmemT<T>), "smem layout");
    if (rows_plain<T>(p, yg, bz) && !(p.dbg & 8)) ok = cta_tile<PT>(p, *reinterpret_cast<SmemT<PT>*>(&sm), qt, yg, bz);   // (dbg 8: always the general routine)
    else ok = cta_tile<T>(p, sm, qt, yg, bz);
  } else {
    ok = cta_tile<T>(p, sm, qt, yg, bz);
  }
  ok = ok && !(p.dbg & 256);                                     // (dbg 256: everything through the SAFE pass)
  if (!__all_sync(~0u, ok)) {                                    // some row overflowed / vanished: queue this CTA tile for the SAFE pass
    if ((threadIdx.x & 31) == 0 && atomicExch(&sm.flagged, 1) == 0) {
      const int idx = atomicAdd(p.fix + FIX_COUNT, 1);
      p.fix[FIX_LIST + 3 * idx] = qt; p.fix[FIX_LIST + 3 * idx + 1] = yg; p.fix[FIX_LIST + 3 * idx + 2] = bz;
    }
  }
}

// SAFE pass (T::SAFE instantiations): persistent CTAs over the fix list written by the hot pass (count at fix[FIX_COUNT]); each listed
// CTA tile is recomputed whole with a true running max and written once (deterministic whatever the list order).  Normally the list
// is empty and the launch exits at once; it is launched unconditionally (graph-capturable).
template <class T>
__global__ void __launch_bounds__(T::THREADS, 1) fix_kernel(const Args p) {
  extern __shared__ uint8_t smem_raw[];
  SmemT<T>& sm = *reinterpret_cast<SmemT<T>*>((reinterpret_cast<uintptr_t>(smem_raw) + 127) & ~uintptr_t(127));
  const int n = p.fix[FIX_COUNT];
  for (int w = blockIdx.x; w < n; w += gridDim.x) {
    const int x = p.fix[FIX_LIST + 3 * w], y = p.fix[FIX_LIST + 3 * w + 1], z = p.fix[FIX_LIST + 3 * w + 2];
    cta_tile<T>(p, sm, x, y, z);
    if (threadIdx.x == 0) atomicAdd(p.fix + FIX_TILES, 1);      // census
  }
}

// ------------------------------------------------------------------------------------------------------------------ staging kernels
// (templates only so that the header can be included in several translation units)
// 1. keyany[b][k] = OR over rows i of mask[b,i,k] (k < S64; 0 beyond S).
template <int kUnused = 0>
__global__ void mask_or_kernel(const uint8_t* __restrict__ mask, long long sB, long long sN, uint8_t* __restrict__ keyany, int N, int S, int S64) {
  __shared__ uint8_t acc[8][128];
  const int b = blockIdx.y, k = blockIdx.x * 128 + (threadIdx.x & 127), part = threadIdx.x >> 7;   // 1024 threads = 8 row slices x 128 keys
  uint8_t any = 0;
  if (k < S) for (int i = part; i < N; i += 8) any |= mask[(long long)b * sB + (long long)i * sN + k];
  acc[part][threadIdx.x & 127] = any != 0;
  __syncthreads();
  if (part == 0 && k < S64) {
    uint8_t a = 0;
    for (int pp = 0; pp < 8; ++pp) a |= acc[pp][threadIdx.x & 127];
    keyany[(long long)b * S64 + k] = a;
  }
}
// 2. One warp per pair row (b, i): mask words (bit = attend, key < S), first / last attended key, count, whether the row's mask equals
//    the batch OR (keyany).  rows[b*N+i] = {a, e, kind, 0}:
//      count == 0                -> ROW_UNIFORM  (a = 0, e = S): fully-masked row, uniform average of v;
//      mask == keyany (regular)  -> ROW_INTERVAL over the batch's live tile span, no in-tile boundary (the -inf bias columns are exact):
//                                   a = 64 * (first live tile of keyany), e = 64 * (last live tile + 1);
//      one contiguous run [a, e) -> ROW_INTERVAL (a, e): its own tiles only, per-key masking on the tiles holding a or e - 1 (a run
//                                   ending at S needs no trailing boundary: keys >= S are -inf columns);
//      anything else             -> ROW_RAGGED, a / e = 64-aligned span of its attended keys, mask words on every tile.
//    census[1] += ragged rows, census[2] += uniform rows; rgflag[b*YG + i/R] = 1 for a row group holding a ragged row.
template <int kUnused = 0>
__global__ void mask_rows_kernel(const uint8_t* __restrict__ mask, long long sB, long long sN, const uint8_t* __restrict__ keyany,
                                 int4* __restrict__ rows, uint32_t* __restrict__ maskw, int* __restrict__ census, int* __restrict__ rgflag,
                                 int B, int N, int S, int S64, int nkt, int R) {
  const int lane = threadIdx.x & 31;
  const long long row = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= (long long)B * N) return;
  const int b = (int)(row / N), i = (int)(row - (long long)b * N);
  const uint8_t* m = mask + (long long)b * sB + (long long)i * sN;
  const uint8_t* ka = keyany + (long long)b * S64;
  int first = 0x7fffffff, last = -1, cnt = 0, afirst = 0x7fffffff, alast = -1; bool diff = false;
  for (int w = 0; w < 2 * nkt; ++w) {
    const int k = 32 * w + lane;
    const bool keep = k < S && m[k] != 0, anyk = k < S && ka[k] != 0;
    const uint32_t word = __ballot_sync(~0u, keep), aword = __ballot_sync(~0u, anyk);
    if (lane == 0) maskw[row * (2ll * nkt) + w] = word;
    if (word) { first = min(first, 32 * w + __ffs(word) - 1); last = max(last, 32 * w + 31 - __clz(word)); cnt += __popc(word); }
    if (aword) { afirst = min(afirst, 32 * w + __ffs(aword) - 1); alast = max(alast, 32 * w + 31 - __clz(aword)); }
    diff |= (word != aword);
  }
  if (lane == 0) {
    int4 r;
    if (cnt == 0) { r = make_int4(0, S, ROW_UNIFORM, 0); atomicAdd(census + 2, 1); }
    else if (!diff) { r = make_int4((afirst / BN) * BN, (alast / BN + 1) * BN, ROW_INTERVAL, 0); }
    else if (cnt == last - first + 1) { r = make_int4(first, (last + 1 == S) ? (last / BN + 1) * BN : last + 1, ROW_INTERVAL, 0); }
    else { r = make_int4((first / BN) * BN, (last / BN + 1) * BN, ROW_RAGGED, 0); atomicAdd(census + 1, 1); rgflag[(long long)b * ((N + R - 1) / R) + i / R] = 1; }
    rows[row] = r;
  }
}
// 3. row-group census: census[3] += number of set flags.
template <int kUnused = 0>
__global__ void count_flags_kernel(const int* __restrict__ rgflag, int n, int* __restrict__ census) {
  int c = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) c += rgflag[i] != 0;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(~0u, c, o);
  if ((threadIdx.x & 31) == 0 && c) atomicAdd(census + 3, c);
}

TS_DEVI float to_f32(float x) { return x; }
TS_DEVI float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
TS_DEVI float to_f32(__half x) { return __half2float(x); }

// bias [B,H,S,S] (any float dtype, element strides) -> fp32 staged [B,H,S128,S64] / scale, permuted so that a [128 rows x 32 keys] box
// starting at (q tile, 32-key block) lands in smem with every 16-byte chunk = one mma C fragment of one thread:
// chunk ((m8*8 + gi)*16 + pk) of a box holds bias[q0][k0], [q0][k0+1], [q0+8][k0], [q0+8][k0+1] with q0 = 16*m8 + gi, k0 = 2*kp,
// kp = pk ^ 4*(gi&1) (the XOR spreads the two rows of a quarter-warp over different banks).  Entries with q >= S or k >= S are 0.
// Keys k >= S, and keys no row attends (keyany[b][k] == 0, when a mask was staged), become -inf columns: ex2 of them is an exact 0,
// so the mask-free stream serves partial tiles and padding masks.  (Padding rows q >= S keep finite entries.)  The permutation is local
// to 16-row blocks, so any 16-aligned BM-row window of the staged tensor is a valid CTA bias tile.  Same tensor format as the sm_90a members.
template <typename U>
__global__ void stage_bias_kernel(const U* __restrict__ bias, long long sB, long long sH, long long sQ, long long sK,
                                  const uint8_t* __restrict__ keyany, float* __restrict__ out, int B, int H, int S, int S128, int S64, float inv_scale) {
  const long long n = (long long)B * H * S128 * S64;
  for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (long long)gridDim.x * blockDim.x) {
    const int cc = (int)(idx % S64); long long rest = idx / S64;
    const int rr = (int)(rest % S128); rest /= S128; const int hh = (int)(rest % H); const int bb = (int)(rest / H);
    const int qt = rr >> 7, r = rr & 127, kb32 = cc >> 5, c = cc & 31;
    const int m8 = r >> 4, gi = (r >> 1) & 7, pk = ((r & 1) << 3) + (c >> 2), e = c & 3;
    const int kp = pk ^ ((gi & 1) << 2);
    const int q = (qt << 7) + 16 * m8 + gi + 8 * (e >> 1), k = (kb32 << 5) + 2 * kp + (e & 1);
    float v = 0.f;
    const bool keyok = k < S && (!keyany || keyany[(long long)bb * S64 + k] != 0);
    if (q < S) v = keyok ? to_f32(bias[bb * sB + hh * sH + (long long)q * sQ + (long long)k * sK]) * inv_scale : -INFINITY;
    else if (k >= S) v = -INFINITY;
    out[idx] = v;
  }
}

}  // namespace triattn_sm80
