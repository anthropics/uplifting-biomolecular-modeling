// Triangle-attention forward, "many-warp mma.sync" formulation for H100 (sm_90a), D = 32, bf16 in / fp32 accumulate / bf16 out.
//
//   out[b,i,h,q,:] = softmax_k( scale * q[b,i,h,q,:].k[b,i,h,k,:] + bias[b,h,q,k]  (-inf where mask[b,i,k] == 0) ) @ v[b,i,h,k,:]
//
// One CTA = 12 warps = R(3) pair rows x 4 warps x 32 queries: a 128-query tile of 3 consecutive pair rows of one (b, h).  Per 32-key
// half tile every warp computes, in registers, S = Q K^T + bias/scale (mma.sync.m16n8k16; the pre-scaled fp32 bias tile is the mma
// C operand), p = ex2(S*scale*log2e - m), O += P V (P is the bf16 A operand; row sums come from an all-ones B column).  The softmax
// is MAX-FREE: m is a per-row integer offset that only moves, by exact powers of two applied to O and l, when a row's running sum
// leaves [2^-20, 2^20]; nothing per logit tracks a maximum.  Rows for which this ever produces inf/nan or a zero sum (logit swings
// beyond ~2^100 -- never seen on model data) are detected at the end and recomputed by a second, max-tracking instantiation of the
// same kernel (the "safe" pass) from a per-call fix-up list, so results are exact for any input.  The work is software-pipelined
// over half tiles -- one straight-line block issues the exponentials + P V of one half next to the S mma of the next -- so a warp
// always has independent tensor-pipe and MUFU work in flight.  The bias arrives as [128 q x 32 k] fp32 boxes laid out so that
// one LDS.128 is exactly one mma accumulator fragment (see stage_bias_kernel); ring A stages hold {bias keys 0-31, K tiles of the
// 3 rows} of a key tile, ring B stages {bias keys 32-63, V tiles}, both 4 deep; Q lives in registers (mma A fragments); after the
// prologue each ring stage is refilled by whichever warp releases it last (no dedicated producer warp).
// Masks.  Keys no pair row of the batch attends (the OR over rows of the mask -- with the usual padding masks that is exactly each
// row's mask) and keys >= S are folded into the staged bias as -inf columns (exact), so rows whose mask equals that OR ("regular"
// rows: all of them for padding masks) run the mask-free kernel.  Rows with their own pattern ("irregular": interior holes, shorter
// prefixes, fully-masked rows) are listed by stage_mask and recomputed by the general instantiation in a second pass over that
// list (per (row, 64-key tile) class {skip, mixed, full, uniform} + keep bits from stage_mask's per-row tables in global memory;
// mixed tiles select -inf per key; a pair
// row with no attendable key attends uniformly to all S keys = mean of v, as cuEquivariance does).
// Grid = (q tiles, row groups, B*H), q-tile fastest so the CTAs sharing K/V rows are co-resident (L2 reuse).
#include "mw_ptx.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace mw {

#ifndef MW_R
#define MW_R 3            // pair rows per CTA tile (4 warps each); MW_R=4 -> 16 warps at <= 128 registers
#endif
#ifndef MW_ST
#define MW_ST 4           // ring depth
#endif
constexpr int D = 32, BM = 128, BN = 64, R = MW_R, WARPS = 4 * R, THREADS = 32 * WARPS;
constexpr int ST = MW_ST;                                // ring depth (key tiles in flight per CTA)
constexpr int BIAS_BOX = 128 * BM;                       // bytes of one [BM q x 32 k] fp32 box (16 KB)
constexpr int KV_TILE = BN * D * 2;                      // 4 KB
constexpr int KOFF = 2 * BIAS_BOX, VOFF = KOFF + R * KV_TILE;   // stage layout: bias keys 0-31 | bias keys 32-63 | K x R rows | V x R rows
constexpr int SBYTES = VOFF + R * KV_TILE;               // 56 KB per stage
// Operating point of the max-free softmax: a row's offset m is kept SHIFT log2 units ABOVE its running log2-sum, i.e. P of the
// dominant key is ~2^-SHIFT rather than ~1: a logit may then exceed everything the row has seen so far by SHIFT + 108 log2 units
// before ex2 overflows (real pair-bias rows swing by up to ~80), while keys more than 126 - SHIFT log2 units below the running
// maximum flush to zero (weight < 2^-62: invisible even in fp32).  The renormalisation window is [2^-(SHIFT+20), 2^-(SHIFT-20)].
constexpr int SHIFT = 64;
constexpr uint32_t RN_LO = (uint32_t)(127 - SHIFT - 20) << 23, RN_WIDTH = 40u << 23;   // float bits of 2^-(SHIFT+20); 40 binades
constexpr int MAX_S = 1 << 16;                           // S <= 65536 (sanity cap; the per-row mask tables live in global memory, see stage_mask)

struct Params {
  CUtensorMap tmK, tmV, tmB;
  const __nv_bfloat16* q; long long q_sB, q_sN, q_sH, q_sS;     // Q: plain 4-byte loads straight into mma A fragments (d stride 1, other strides even)
  __nv_bfloat16* out; long long o_sB, o_sN, o_sH, o_sS;          // element strides (d stride 1)
  const uint8_t* mask; long long m_sB, m_sN;                     // [B,N,S] bytes, or nullptr (general instantiation only)
  const uint8_t* rowkind;                                        // [B,N] 1 = irregular row (main pass skips CTAs whose rows all are), or nullptr
  const uint32_t* mtab; const uint8_t* ctab; const int* ktendr;   // per-row mask tables from stage_mask (general passes; nullptr without a mask): mtab [B,N,nkt,2] keep
                                                                 // bits of the two 32-key halves of tile kt, ctab [B,N,nkt] tile class, ktendr [B,N] tiles up to the row's last kept key (0: none)
  int* fix;                                                      // fix[0], fix[1] = running census of fix-pass tiles / list-pass row groups (all calls), fix[2] = count, fix[3..] = (x, y, z) CTA tiles to redo
  const int* list; int list_mul;                                 // list passes: list[0] = count, entries from list[1]; entry e covers CTA tiles e*list_mul .. +list_mul-1 (see triattn_mw_list)
  int B, N, H, S, n_ktiles;
  const int* ktend;                                              // [B] key tiles holding an attended key (mask-free main pass; nullptr: n_ktiles)
  int qb;                                                        // CTA order: q tiles walked in blocks of qb (q tile fastest inside a block, then row groups)
  float c1;                                                      // scale * log2(e)
  int dbg;                                                       // experiments only: 32 = no renormalisation, 64 = timeline (MW_TIMELINE builds), 256 = all tiles through the safe pass
  long long* tl;                                                 // [TL_CTAS][WARPS+1][TL_T][4] clock64 stamps (dbg & 64)
};
constexpr int TL_CTAS = 16, TL_T = 64;

struct __align__(1024) Smem {
  uint8_t stage[ST][SBYTES];
  uint64_t full[ST], empty[ST];
  uint32_t tick[ST];                                             // per-slot release tickets
  int flagged;
  int kt_endr[R], uniform[R], kt_end;
};

static_assert(sizeof(Smem) + 1024 <= 227 * 1024, "shared memory budget");

enum : int { CLS_SKIP = 0, CLS_MIXED = 1, CLS_FULL = 2, CLS_UNIFORM = 3 };

typedef float CFrag[2][4][4];      // S (then p) of one 32-key half tile, mma C-fragment order [mt][j][e]
// Per-thread state of one warp's 32 queries: o accumulators [mt][dn][4]; l4 = running row sums as the ones-column mma accumulator
// ([mt][0]=[mt][1] row g, [mt][2]=[mt][3] row g+8); m = row offsets (integers).
struct Frag { float o[2][4][4]; float l4[2][4]; float m[2][2]; uint32_t qa[2][2][4]; uint32_t seeded; };   // qa = Q A-fragments [mt][kk]; seeded: bit 2mt+hh = row has its offset
struct Lanes { uint32_t bA, bB, kLane, vLane[2]; };          // per-lane smem address parts; bias column j at (j odd ? bB : bA) + 64 j

// front<CLS,HF>: c = bias/scale (fp32 C operand) + Q K^T for keys 32*HF..+31 of the tile whose A stage is at sA; MIXED: masked keys
// -> -inf; UNIFORM: c = 1 on real keys (no logits); SKIP: nothing.  mw = the half's 32 mask bits.
template <int CLS, int HF>
MW_DEVI void front(CFrag& c, const Frag& f, const Lanes& L, uint32_t sBias, uint32_t sA, uint32_t mw, int t) {   // sBias: this half's bias box; sA: ring-A stage (K)
  if (CLS == CLS_SKIP) return;
  if (CLS == CLS_UNIFORM) {
#pragma unroll
    for (int jl = 0; jl < 4; ++jl) {
      const int bit = 8 * jl + 2 * t;
      const float p0 = ((mw >> bit) & 1) ? 1.f : 0.f, p1 = ((mw >> (bit + 1)) & 1) ? 1.f : 0.f;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) { c[mt][jl][0] = c[mt][jl][2] = p0; c[mt][jl][1] = c[mt][jl][3] = p1; }
    }
    return;
  }
#pragma unroll
  for (int jl = 0; jl < 4; ++jl) {
    const uint32_t a = sBias + ((jl & 1) ? L.bB : L.bA);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const float4 v = lds128f(a + jl * 64 + mt * 2048);
      c[mt][jl][0] = v.x; c[mt][jl][1] = v.y; c[mt][jl][2] = v.z; c[mt][jl][3] = v.w;
    }
  }
  const uint32_t ka = sA + L.kLane;
#pragma unroll
  for (int jl = 0; jl < 4; ++jl) {
    uint32_t kb[4]; ldsm_x4(kb, ka + (4 * HF + jl) * 512);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) { mma16816(c[mt][jl], f.qa[mt][0], kb); mma16816(c[mt][jl], f.qa[mt][1], kb + 2); }
  }
  if (CLS == CLS_MIXED) {
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
}

// expoback<CLS,HF,SAFE>: p = ex2(c*c1 - m) (UNIFORM: c already holds p; SKIP: nothing at all), packed to bf16 A fragments, row sums
// accumulated into l4 by the ones-column mma, O += P V from the V tile (this row) at sV.  SAFE: first raise m to floor(max c*c1) per
// row (rescaling O, l4 exactly) so no exponential can overflow -- the max-tracking variant used by the fix-up pass.
template <int CLS, int HF, bool SAFE>
MW_DEVI void expoback(CFrag& c, Frag& f, const Lanes& L, uint32_t sV, float c1, const uint32_t* ones) {
  if (CLS == CLS_SKIP) return;
  if (CLS != CLS_UNIFORM) {
    if (SAFE) {
      float mx[2][2];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        mx[mt][0] = fmaxf(fmaxf(fmaxf(c[mt][0][0], c[mt][0][1]), fmaxf(c[mt][1][0], c[mt][1][1])), fmaxf(fmaxf(c[mt][2][0], c[mt][2][1]), fmaxf(c[mt][3][0], c[mt][3][1])));
        mx[mt][1] = fmaxf(fmaxf(fmaxf(c[mt][0][2], c[mt][0][3]), fmaxf(c[mt][1][2], c[mt][1][3])), fmaxf(fmaxf(c[mt][2][2], c[mt][2][3]), fmaxf(c[mt][3][2], c[mt][3][3])));
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          mx[mt][hh] = fmaxf(mx[mt][hh], __shfl_xor_sync(~0u, mx[mt][hh], 1));
          mx[mt][hh] = fmaxf(mx[mt][hh], __shfl_xor_sync(~0u, mx[mt][hh], 2));
          mx[mt][hh] = fmaxf(f.m[mt][hh], mx[mt][hh] * c1);                    // candidate new offset = the running max itself (its p is exactly 1); -inf rows keep m
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
            for (int dn = 0; dn < 4; ++dn) { f.o[mt][dn][2 * hh] *= corr; f.o[mt][dn][2 * hh + 1] *= corr; }
          }
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int jl = 0; jl < 4; ++jl) {
        c[mt][jl][0] = ex2f(fmaf(c[mt][jl][0], c1, -f.m[mt][0])); c[mt][jl][1] = ex2f(fmaf(c[mt][jl][1], c1, -f.m[mt][0]));
        c[mt][jl][2] = ex2f(fmaf(c[mt][jl][2], c1, -f.m[mt][1])); c[mt][jl][3] = ex2f(fmaf(c[mt][jl][3], c1, -f.m[mt][1]));
      }
  }
#pragma unroll
  for (int kl = 0; kl < 2; ++kl) {
    uint32_t pa[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      pa[mt][0] = pack_bf16(c[mt][2 * kl][0], c[mt][2 * kl][1]);         pa[mt][1] = pack_bf16(c[mt][2 * kl][2], c[mt][2 * kl][3]);
      pa[mt][2] = pack_bf16(c[mt][2 * kl + 1][0], c[mt][2 * kl + 1][1]); pa[mt][3] = pack_bf16(c[mt][2 * kl + 1][2], c[mt][2 * kl + 1][3]);
      mma16816(f.l4[mt], pa[mt], ones);
    }
    const int kk2 = 2 * HF + kl;
#pragma unroll
    for (int pp = 0; pp < 2; ++pp) {
      uint32_t vb[4]; ldsm_x4_t(vb, sV + L.vLane[pp] + kk2 * 1024);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) { mma16816(f.o[mt][2 * pp], pa[mt], vb); mma16816(f.o[mt][2 * pp + 1], pa[mt], vb + 2); }
    }
  }
}


// step<HF,NEXT>: the mask-free hot block, interleaved at 8-key granularity so the tensor pipe, the FMA pipe and MUFU all have
// work throughout: for each 8-key column j of the CURRENT half (c, keys 32*HF..) -- exponentials of column j; on odd j the two
// columns are packed to bf16, summed into l4 (ones-column mma) and multiplied into O (P V slice); and, when NEXT, the bias load +
// Q K^T mma of column j of the NEXT half (cn: bias box at sBiasN, K in the ring-A stage at sAN, keys 32*HFN..).
template <int HF, bool NEXT, int HFN>
MW_DEVI void step(CFrag& c, Frag& f, const Lanes& L, uint32_t sV, uint32_t sBiasN, uint32_t sAN, float c1, const uint32_t* ones) {
  // One S-fragment buffer: column j of the current half is exponentiated in place; once a column pair is packed (its fragments
  // dead), the NEXT half's S mma of that pair is issued into the same registers, overlapping the exponentials of the following pair.
  const uint32_t ka = sAN + L.kLane;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {                               // current half, column j: exponentials
      c[mt][j][0] = ex2f(fmaf(c[mt][j][0], c1, -f.m[mt][0])); c[mt][j][1] = ex2f(fmaf(c[mt][j][1], c1, -f.m[mt][0]));
      c[mt][j][2] = ex2f(fmaf(c[mt][j][2], c1, -f.m[mt][1])); c[mt][j][3] = ex2f(fmaf(c[mt][j][3], c1, -f.m[mt][1]));
    }
    if (j & 1) {                                                   // columns j-1, j: pack, row sums, P V (16-key slice kl)
      const int kl = j >> 1, kk2 = 2 * HF + kl;
      uint32_t pa[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        pa[mt][0] = pack_bf16(c[mt][j - 1][0], c[mt][j - 1][1]); pa[mt][1] = pack_bf16(c[mt][j - 1][2], c[mt][j - 1][3]);
        pa[mt][2] = pack_bf16(c[mt][j][0], c[mt][j][1]);         pa[mt][3] = pack_bf16(c[mt][j][2], c[mt][j][3]);
        mma16816(f.l4[mt], pa[mt], ones);
      }
      if (NEXT) {                                                  // next half, columns j-1 and j: C = bias fragment, += Q K^T
#pragma unroll
        for (int jj = j - 1; jj <= j; ++jj) {
          const uint32_t a = sBiasN + ((jj & 1) ? L.bB : L.bA);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) { const float4 v = lds128f(a + jj * 64 + mt * 2048); c[mt][jj][0] = v.x; c[mt][jj][1] = v.y; c[mt][jj][2] = v.z; c[mt][jj][3] = v.w; }
          uint32_t kb[4]; ldsm_x4(kb, ka + (4 * HFN + jj) * 512);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) { mma16816(c[mt][jj], f.qa[mt][0], kb); mma16816(c[mt][jj], f.qa[mt][1], kb + 2); }
        }
      }
#pragma unroll
      for (int pp = 0; pp < 2; ++pp) {
        uint32_t vb[4]; ldsm_x4_t(vb, sV + L.vLane[pp] + kk2 * 1024);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) { mma16816(f.o[mt][2 * pp], pa[mt], vb); mma16816(f.o[mt][2 * pp + 1], pa[mt], vb + 2); }
      }
    }
  }
}
#define MW_DISPATCH(cls, CALL) switch (cls) { case CLS_FULL: { constexpr int K_ = CLS_FULL; CALL; } break; case CLS_MIXED: { constexpr int K_ = CLS_MIXED; CALL; } break; \
                                              case CLS_UNIFORM: { constexpr int K_ = CLS_UNIFORM; CALL; } break; default: { constexpr int K_ = CLS_SKIP; CALL; } break; }

// One CTA tile: 3 pair rows (yg*R ..) x 128 queries (qt) of (b,h) = bz.  GEN = false: no mask and S % 64 == 0 (every tile full).
// SAFE = the max-tracking fix-up variant.  Returns (per thread) whether all its rows came out finite with a positive sum.
// Rows not yet seeded (at the start, or after leading tiles that were all masked) take their offset from the
// half tile about to be exponentiated: m = max(c) * c1, so its largest term is 2^0 -- nothing that matters can be flushed by
// ex2.approx.ftz, whatever the absolute logit level.  (Steady state stays max-free: m then only moves by powers of two.)
MW_DEVI void rebase(const CFrag& c, Frag& f, float c1) {
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    float mx[2];
    mx[0] = fmaxf(fmaxf(fmaxf(c[mt][0][0], c[mt][0][1]), fmaxf(c[mt][1][0], c[mt][1][1])), fmaxf(fmaxf(c[mt][2][0], c[mt][2][1]), fmaxf(c[mt][3][0], c[mt][3][1])));
    mx[1] = fmaxf(fmaxf(fmaxf(c[mt][0][2], c[mt][0][3]), fmaxf(c[mt][1][2], c[mt][1][3])), fmaxf(fmaxf(c[mt][2][2], c[mt][2][3]), fmaxf(c[mt][3][2], c[mt][3][3])));
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(~0u, mx[hh], 1));
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(~0u, mx[hh], 2));
      const uint32_t bit = 1u << (2 * mt + hh);                 // a row is seeded once, from the first half tile holding a finite
      if (!(f.seeded & bit) && mx[hh] > -INFINITY && mx[hh] < INFINITY) { f.m[mt][hh] = mx[hh] * c1 + (float)SHIFT; f.seeded |= bit; }   // logit for it; never re-seeded
    }
  }
}

// Per-row mask tables of the general passes: class and keep bits of (row rr of this CTA, key tile kt).  With a mask they come from
// stage_mask's global tables (a few bytes per tile per warp, L2-resident); without one (the fix pass of a mask-free call) they are
// arithmetic: keys < S attend.  Rows past N and tiles past nkt are CLS_SKIP; a row that keeps no key is CLS_UNIFORM on every tile.
MW_DEVI void keep_bits_lt_S(int kt, int S, uint32_t& w0, uint32_t& w1) {
  const int n0 = S - kt * BN, n1 = n0 - 32;
  w0 = n0 >= 32 ? ~0u : (n0 <= 0 ? 0u : ((1u << n0) - 1u)); w1 = n1 >= 32 ? ~0u : (n1 <= 0 ? 0u : ((1u << n1) - 1u));
}
MW_DEVI int tab_cls(const Params& p, const Smem& sm, int b, int i_row, bool row_present, int rr, int kt) {
  if (!row_present || kt >= p.n_ktiles) return CLS_SKIP;
  if (sm.uniform[rr]) return CLS_UNIFORM;
  if (p.mask) return (int)__ldg(p.ctab + ((long long)b * p.N + i_row) * p.n_ktiles + kt);
  return (kt * BN + BN <= p.S) ? CLS_FULL : CLS_MIXED;
}
MW_DEVI void tab_bits(const Params& p, const Smem& sm, int b, int i_row, int rr, int kt, uint32_t& w0, uint32_t& w1) {
  if (p.mask && !sm.uniform[rr]) { const uint32_t* t = p.mtab + (((long long)b * p.N + i_row) * p.n_ktiles + kt) * 2; w0 = __ldg(t); w1 = __ldg(t + 1); }
  else keep_bits_lt_S(kt, p.S, w0, w1);
}

template <bool GEN, bool SAFE>
MW_DEVI bool cta_tile(const Params& p, Smem& sm, int qt, int yg, int bz) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rr = warp >> 2, wq = warp & 3;
  const int i0 = yg * R, b = bz / p.H, h = bz % p.H;
  const int q_base = qt * BM, S = p.S, nkt = p.n_ktiles;
  const bool renorm_on = !(p.dbg & 32);
  const bool fused = !(p.dbg & 1024);                            // experiments: 1024 = the unfused expoback+front path instead of step<> (mask-free pass)

  // ---- barriers, Q tiles, tile classes ------------------------------------------------------------------------------------------
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(&sm.full[s], 1); mbar_init(&sm.empty[s], THREADS); sm.tick[s] = 0; }
    sm.flagged = 0;
    fence_barrier_init();
  }
  if (tid < R) { sm.kt_endr[tid] = 0; sm.uniform[tid] = 0; }
  __syncthreads();
  int kt_end = (!GEN && p.ktend) ? p.ktend[b] : nkt;             // mask-free pass: tiles past the batch's last attended key are dead
  if (GEN) {                                                     // per-row extent from the tables (or: every tile, without a mask)
    if (tid < R) {
      const int i = i0 + tid; int e = 0, u = 0;
      if (i < p.N) {
        e = p.mask ? p.ktendr[(long long)b * p.N + i] : nkt;
        if (e == 0) { u = 1; e = nkt; }                          // no kept key: uniform average over all S keys (cuEquivariance semantics)
      }
      sm.kt_endr[tid] = e; sm.uniform[tid] = u;
    }
    __syncthreads();
    if (tid == 0) { int e = 0; for (int r_ = 0; r_ < R; ++r_) e = max(e, sm.kt_endr[r_]); sm.kt_end = e; }
    __syncthreads();
    kt_end = sm.kt_end;
  }
  const int T = kt_end;                                          // (T == 0 only in the mask-free pass of a batch with no attended key:
                                                                 //  nothing is computed, l stays 0 and the tile goes to the fix pass)
  const int cta_flat = (bz * gridDim.y + yg) * gridDim.x + qt;
  const bool tl_on = (p.dbg & 64) && cta_flat < TL_CTAS;
  auto stamp = [&](int w, int kt, int ev) {                       // w = warp, or WARPS for issue events; compiled in with -DMW_TIMELINE
#ifdef MW_TIMELINE
    if (tl_on && kt < TL_T) p.tl[(((long long)cta_flat * (WARPS + 1) + w) * TL_T + kt) * 4 + ev] = clock64();
#else
    (void)w; (void)kt; (void)ev; (void)tl_on;
#endif
  };

  // ---- TMA issue of key tile kt into ring slot kt % ST: {bias keys 0-31, bias keys 32-63, K x R rows, V x R rows} -------------
  auto issue = [&](int kt) {
    const int s = kt % ST; uint8_t* st = sm.stage[s];
    stamp(WARPS, kt, 0);
    fence_proxy_async();                                         // v1.2: order every generic-proxy read of this slot (ldmatrix / ld.shared, released
                                                                 // on 'empty' and acquired above, or the previous tile's reads before the CTA barrier)
                                                                 // before the async-proxy (TMA) writes that refill it
    mbar_arrive_expect_tx(&sm.full[s], SBYTES);
    tma_load_4d(st, &p.tmB, &sm.full[s], kt * BN, q_base, h, b);
    tma_load_4d(st + BIAS_BOX, &p.tmB, &sm.full[s], kt * BN + 32, q_base, h, b);
    tma_load_5d(st + KOFF, &p.tmK, &sm.full[s], 0, kt * BN, h, i0, b);          // R consecutive pair rows in one box (rows >= N zero-filled)
    tma_load_5d(st + VOFF, &p.tmV, &sm.full[s], 0, kt * BN, h, i0, b);
  };
  if (tid == 0) { for (int kt = 0; kt < ST && kt < T; ++kt) issue(kt); }        // fill the ring; later tiles are issued from inside the loop
  // ---- per-warp constants --------------------------------------------------------------------------------------------------------
  const int i_row = i0 + rr;
  const bool row_present = i_row < p.N;
  const bool unif = GEN && sm.uniform[rr] != 0;                   // fully-masked row: uniform average of V (H8); p == 1, no offsets
  Lanes L;
  {
    const int l7 = lane & 7, mi = lane >> 3;
    L.kLane = KOFF + rr * KV_TILE + l7 * 64 + ((mi ^ (l7 >> 1)) << 4);
#pragma unroll
    for (int pp = 0; pp < 2; ++pp) L.vLane[pp] = VOFF + rr * KV_TILE + ((mi & 1) * 8 + l7) * 64 + (((2 * pp + (mi >> 1)) ^ (l7 >> 1)) << 4);
#pragma unroll
    L.bA = wq * 4096 + g * 256 + t * 16 + (g & 1) * 64; L.bB = L.bA - (g & 1) * 128;   // = ... + ((jl ^ (g & 1)) * 64) - 64 jl
    L.bA = opaque(L.bA); L.bB = opaque(L.bB); L.kLane = opaque(L.kLane); L.vLane[0] = opaque(L.vLane[0]); L.vLane[1] = opaque(L.vLane[1]);   // keep in registers
  }
  const float c1 = p.c1;
  const uint32_t ones[2] = {0x3f803f80u, 0x3f803f80u};
  Frag f; CFrag cb;                                              // S fragments of the half in flight (see step)
  f.seeded = 0u;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    f.m[mt][0] = f.m[mt][1] = SAFE ? -1e30f : 0.f;
    f.l4[mt][0] = f.l4[mt][1] = f.l4[mt][2] = f.l4[mt][3] = 0.f;
#pragma unroll
    for (int dn = 0; dn < 4; ++dn) f.o[mt][dn][0] = f.o[mt][dn][1] = f.o[mt][dn][2] = f.o[mt][dn][3] = 0.f;
  }
  const uint32_t s00 = smem_u32(sm.stage[0]);

  {                                                              // Q of this warp's 32 queries -> mma A fragments, straight from global (overlaps the ring fills)
    const __nv_bfloat16* qb = p.q + (long long)b * p.q_sB + (long long)i_row * p.q_sN + (long long)h * p.q_sH;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int qrow = q_base + wq * 32 + mt * 16 + hh * 8 + g;
        const bool okq = row_present && qrow < S;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(qb + (long long)qrow * p.q_sS + 2 * t);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) { f.qa[mt][kk][hh] = okq ? __ldg(src + 8 * kk) : 0u; f.qa[mt][kk][2 + hh] = okq ? __ldg(src + 8 * kk + 4) : 0u; }
      }
  }

  // ---- pipeline.  Iteration t (ring slot s = t % ST holds tile t):  (r) renormalise rows whose sum left [2^-20, 2^20]
  //      B1 { exps + P V of (t, keys 0-31) : S mma of (t, keys 32-63) }   (w) wait tile t+1   B2 { exps + P V of (t, 32-63) : S mma
  //      of (t+1, 0-31) }   (e) release slot s; the warp that drew the first ticket of tile t-1 refills t-1's slot with tile t+3.
  //      Before the loop: wait tile 0; S mma of (0, keys 0-31).  The last tile skips (w) and B2's S mma.
  int clsC = GEN ? tab_cls(p, sm, b, i_row, row_present, rr, 0) : CLS_FULL;
  uint32_t mwC0 = 0, mwC1 = 0;
  if (GEN && (clsC == CLS_MIXED || clsC == CLS_UNIFORM)) tab_bits(p, sm, b, i_row, rr, 0, mwC0, mwC1);

  if (T > 0) {                                                   // (see T above)
  mbar_wait(&sm.full[0], 0);
  if (!GEN) front<CLS_FULL, 0>(cb, f, L, s00, s00, 0, t);
  else MW_DISPATCH(clsC, (front<K_, 0>(cb, f, L, s00, s00, mwC0, t)));
  if (!SAFE && (!GEN || clsC == CLS_FULL || clsC == CLS_MIXED)) rebase(cb, f, c1);   // m from the first half tile
  bool unseeded = !SAFE && __any_sync(~0u, f.seeded != 15u);   // some row of this warp saw no finite logit yet (leading keys all excluded)

  // Slot refill.  Every warp takes a ticket for the slot early in the iteration that reads it (lane 0; the atomic's latency hides
  // under the block) and releases the slot at the end of the iteration (all lanes arrive on 'empty').  The warp that drew the FIRST
  // ticket -- the warp furthest ahead -- refills the slot one iteration later, after the empty barrier confirms every release: the
  // producer work lands on the warp with the most slack, and a leader more than about one iteration ahead of the slowest warp waits
  // there, which keeps the twelve warps loosely in step (an unchecked leader otherwise runs tiles ahead and then idles on the ring
  // while the slowest warp sets the pace).  The warp with the last ticket resets the counter.
  bool lead = false;                                             // this warp drew ticket 0 of tile t-1
  const uint32_t lane_zero = (uint32_t)lane & ((uint32_t)p.dbg >> 31);   // == 0 (dbg >= 0); opaque to the compiler
  auto body = [&](int kt, auto last_tag) {
    constexpr bool LAST = decltype(last_tag)::value;
    const int sl = kt % ST, sl1 = (kt + 1) % ST;
    const uint32_t sS = s00 + sl * SBYTES, sS1 = s00 + sl1 * SBYTES;
    if (lane == 0) stamp(warp, kt, 0);
    const uint32_t tk = atom_ticket_lane0(&sm.tick[sl], lane, lane_zero);     // lane 0's ticket for slot s (other lanes: ~0)
    // Seeding: a row takes its offset from the first half tile holding a finite logit for it, before that half is exponentiated
    // -- keys 0-31 here, keys 32-63 right after B1 fronts them (whatever the tile; 'unseeded' is warp-uniform and turns false for
    // good once every row of the warp is seeded, so the steady state pays one predicated branch per check).
    if (unseeded) {
      if (!SAFE && (!GEN || clsC == CLS_FULL || clsC == CLS_MIXED)) rebase(cb, f, c1);
      unseeded = __any_sync(~0u, f.seeded != 15u);
    }
    // (r) exact power-of-two renormalisation (rare): m += k, O *= 2^-k, l *= 2^-k with k = floor(log2 l)
    {
      bool rn = false;                                           // some row's sum left the window (l == 0 rows are skipped below).  Never for
      if (!SAFE && !unif)                                        // the safe pass (true max, l in [1, S]) or uniform rows (p == 1)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) rn |= (__float_as_uint(f.l4[mt][2 * hh]) - RN_LO) > RN_WIDTH;
      if (__any_sync(~0u, rn) && renorm_on) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const float l = f.l4[mt][2 * hh];
            if (l > 0.f && (__float_as_uint(l) - RN_LO) > RN_WIDTH) {   // outside the window (inf included: k clamps, the row fails validation)
              const int k = min(max((int)((__float_as_uint(l) >> 23) & 0xff) - 127 + SHIFT, -126), 126);   // l -> ~2^-SHIFT; 2^-k a normal power of two
              const float corr = __uint_as_float((uint32_t)(127 - k) << 23);
              f.m[mt][hh] += (float)k;
              f.l4[mt][2 * hh] *= corr; f.l4[mt][2 * hh + 1] *= corr;
#pragma unroll
              for (int dn = 0; dn < 4; ++dn) { f.o[mt][dn][2 * hh] *= corr; f.o[mt][dn][2 * hh + 1] *= corr; }
            }
          }
      }
    }
    // B1: current = (t, keys 0-31) [V of slot s], next = (t, keys 32-63) [bias box 1 + K of slot s]
    if (!GEN && !SAFE && fused) step<0, true, 1>(cb, f, L, sS, sS + BIAS_BOX, sS, c1, ones);
    else if (!GEN) { expoback<CLS_FULL, 0, SAFE>(cb, f, L, sS, c1, ones); front<CLS_FULL, 1>(cb, f, L, sS + BIAS_BOX, sS, 0, t); }
    else MW_DISPATCH(clsC, (expoback<K_, 0, SAFE>(cb, f, L, sS, c1, ones), front<K_, 1>(cb, f, L, sS + BIAS_BOX, sS, mwC1, t)));
    if (unseeded) {                                              // rows whose first finite logit is in keys 32-63 of this tile seed from them
      if (!GEN || clsC == CLS_FULL || clsC == CLS_MIXED) rebase(cb, f, c1);
      unseeded = __any_sync(~0u, f.seeded != 15u);
    }
    // (m) last iteration's leader refills tile t-1's slot with tile t+3 (every warp released it at the end of iteration t-1)
    if (lane == 0 && lead && kt + ST - 1 < T) { mbar_wait(&sm.empty[(kt + ST - 1) % ST], ((kt - 1) / ST) & 1); issue(kt + ST - 1); }
    // (w) tile t+1 + its class
    int clsN = CLS_SKIP; uint32_t mwN0 = 0, mwN1 = 0;
    if (!LAST) {
      mbar_wait(&sm.full[sl1], ((kt + 1) / ST) & 1);
      if (lane == 0) stamp(warp, kt, 2);
      if (GEN) {
        clsN = tab_cls(p, sm, b, i_row, row_present, rr, kt + 1);
        if (clsN == CLS_MIXED || clsN == CLS_UNIFORM) tab_bits(p, sm, b, i_row, rr, kt + 1, mwN0, mwN1);
      }
    }
    // B2: current = (t, keys 32-63) [V of slot s], next = (t+1, keys 0-31) [bias box 0 + K of slot s+1]
    if (!GEN && !SAFE && fused) { if (!LAST) step<1, true, 0>(cb, f, L, sS, sS1, sS1, c1, ones); else step<1, false, 0>(cb, f, L, sS, sS1, sS1, c1, ones); }
    else if (!GEN) { expoback<CLS_FULL, 1, SAFE>(cb, f, L, sS, c1, ones); if (!LAST) front<CLS_FULL, 0>(cb, f, L, sS1, sS1, 0, t); }
    else {
      MW_DISPATCH(clsC, (expoback<K_, 1, SAFE>(cb, f, L, sS, c1, ones)));
      if (!LAST) MW_DISPATCH(clsN, (front<K_, 0>(cb, f, L, sS1, sS1, mwN0, t)));
    }
    // (e) release slot s (this warp is done with tile t); rotate.  v1.2: generic->async proxy fence by every reader BEFORE the release, so
    //     the leader's TMA refill of this slot (async proxy) is ordered after this warp's ldmatrix / ld.shared reads of it (WAR across proxies)
    fence_proxy_async();
    mbar_arrive(&sm.empty[sl]);
    st_shared_u32_if(&sm.tick[sl], 0u, tk == WARPS - 1);                        // last ticket resets the counter
    lead = (tk == 0);
    if (GEN) { clsC = clsN; mwC0 = mwN0; mwC1 = mwN1; }        // (mask-free: every tile is CLS_FULL)
    if (lane == 0) stamp(warp, kt, 3);
  };
  for (int kt = 0; kt < T - 1; ++kt) body(kt, std::integral_constant<bool, false>{});
  body(T - 1, std::integral_constant<bool, true>{});
  }   // T > 0

  // ---- output: O / l -> bf16, straight from registers (4-byte stores); validity --------------------------------------------------
  bool ok = true;
  {
    __nv_bfloat16* ob = p.out + (long long)b * p.o_sB + (long long)i_row * p.o_sN + (long long)h * p.o_sH;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float l = f.l4[mt][2 * hh];
        const float inv = l > 0.f ? 1.f / l : 0.f;
        const int qrow = q_base + wq * 32 + mt * 16 + hh * 8 + g;
        float chk = l;
        uint32_t w[4];
#pragma unroll
        for (int dn = 0; dn < 4; ++dn) {
          chk += (f.o[mt][dn][2 * hh] + f.o[mt][dn][2 * hh + 1]) * 0.f;
          w[dn] = pack_bf16(f.o[mt][dn][2 * hh] * inv, f.o[mt][dn][2 * hh + 1] * inv);
        }
        ok &= (chk > 0.f) && (chk < INFINITY);                     // false for l == 0, inf, or any nan/inf in O
        if (row_present && qrow < S) {
          uint32_t* dst = reinterpret_cast<uint32_t*>(ob + (long long)qrow * p.o_sS + 2 * t);
#pragma unroll
          for (int dn = 0; dn < 4; ++dn) dst[4 * dn] = w[dn];
        }
      }
  }
  return ok || !row_present;
}

template <bool GEN, bool SAFE>
__global__ void __launch_bounds__(THREADS, 1) triattn_mw_fwd(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  // CTA order (hardware launches x fastest): within each (b, h) the q tiles are walked in blocks of qb -- q tile fastest inside the
  // block (the CTAs of one row group are adjacent and share its K/V tiles in flight), all row groups, then the next block -- so the
  // bias tiles the CTAs in flight share (qb q tiles' worth) stay within L2 at any N.
  const int QT = gridDim.x, YG = gridDim.y;
  const int lin = blockIdx.x + QT * blockIdx.y, bz = blockIdx.z;
  const int blk = lin / (p.qb * YG), r2 = lin - blk * (p.qb * YG);
  const int qbn = min(p.qb, QT - blk * p.qb);                    // q tiles in this block (the last one may be short)
  const int yg = r2 / qbn, qt = blk * p.qb + (r2 - yg * qbn);
  if (p.rowkind) {                                               // all three pair rows irregular (or absent): the list pass computes this tile
    const int b = bz / p.H, i0 = yg * R;
    bool any_regular = false;
    for (int r_ = 0; r_ < R; ++r_) any_regular |= (i0 + r_ < p.N) && p.rowkind[(long long)b * p.N + i0 + r_] == 0;
    if (!any_regular) return;
  }
  const bool ok = cta_tile<GEN, SAFE>(p, sm, qt, yg, bz) && !(p.dbg & 256);    // (dbg 256: everything through the safe pass)
  if (!__all_sync(~0u, ok)) {                                    // some row overflowed / vanished: queue this CTA tile for the safe pass
    if ((threadIdx.x & 31) == 0 && atomicExch(&sm.flagged, 1) == 0) {
      const int idx = atomicAdd(p.fix + 2, 1);
      p.fix[3 + 3 * idx] = qt; p.fix[4 + 3 * idx] = yg; p.fix[5 + 3 * idx] = bz;
    }
  }
}

// List passes, both with the general (masked) instantiation, over CTA tiles named by a device-side list (count in list[0]):
//   SAFE = false: the irregular-row pass -- entry = b * YG + yg (a batch row group holding a row whose mask differs from the batch
//                 OR), expanded to its QT x H CTA tiles (list_mul = QT * H); rows that overflow here join the fix list;
//   SAFE = true:  the fix pass (max-tracking) -- entry = one (x, y, z) CTA tile flagged by an earlier pass (list_mul = 1).
// Normally both lists are empty and the launches exit at once.
template <bool SAFE>
__global__ void __launch_bounds__(THREADS, 1) triattn_mw_list(const __grid_constant__ Params p) {
  extern __shared__ uint8_t smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int QT = (p.S + BM - 1) / BM, YG = (p.N + R - 1) / R;
  const long long n = (long long)p.list[0] * p.list_mul;
  for (long long w = blockIdx.x; w < n; w += gridDim.x) {
    int x, y, z;
    if (SAFE) { x = p.list[1 + 3 * w]; y = p.list[2 + 3 * w]; z = p.list[3 + 3 * w]; }
    else {
      const int e = p.list[1 + w / p.list_mul], sub = (int)(w % p.list_mul);   // e = b * YG + yg; sub = h * QT + qt
      const int b = e / YG; y = e % YG; x = sub % QT; z = b * p.H + sub / QT;
    }
    const bool ok = cta_tile<true, SAFE>(p, sm, x, y, z);
    if (threadIdx.x == 0) { if (SAFE) atomicAdd(p.fix, 1); else if (w % p.list_mul == 0) atomicAdd(p.fix + 1, 1); }   // census: fix tiles / list row groups
    if (!SAFE && !__all_sync(~0u, ok)) {
      if ((threadIdx.x & 31) == 0 && atomicExch(&sm.flagged, 1) == 0) {
        const int idx = atomicAdd(p.fix + 2, 1);
        p.fix[3 + 3 * idx] = x; p.fix[4 + 3 * idx] = y; p.fix[5 + 3 * idx] = z;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int s = 0; s < ST; ++s) { mbar_inval(&sm.full[s]); mbar_inval(&sm.empty[s]); }
    }
    __syncthreads();
  }
}

// ---- mask staging (only when a mask is given): 1. keyany[b][k] = OR over rows i of mask[b,i,k] (k < S64; 0 beyond S); ktend[b] =
//      number of 64-key tiles up to the last attended key of batch b (the mask-free main pass stops there)
__global__ void mask_or_kernel(const uint8_t* __restrict__ mask, long long sB, long long sN, uint8_t* __restrict__ keyany, int* __restrict__ ktend, int N, int S, int S64) {
  __shared__ uint8_t acc[8][128];
  const int b = blockIdx.y, k = blockIdx.x * 128 + (threadIdx.x & 127), part = threadIdx.x >> 7;   // 1024 threads = 8 row slices x 128 keys
  uint8_t any = 0;
  if (k < S) for (int i = part; i < N; i += 8) any |= mask[(long long)b * sB + (long long)i * sN + k];
  acc[part][threadIdx.x & 127] = any != 0;
  __syncthreads();
  uint8_t a = 0;
  if (part == 0 && k < S64) {
    for (int pp = 0; pp < 8; ++pp) a |= acc[pp][threadIdx.x & 127];
    keyany[(long long)b * S64 + k] = a;
  }
  if (part == 0 && __any_sync(~0u, a != 0) && (threadIdx.x & 31) == 0) atomicMax(&ktend[b], k / 64 + 1);   // (a warp's 32 keys lie in one tile)
}
// 2. rowkind[b][i] = 1 if row i's mask differs from keyany[b] on some key < S (one warp per row); such rows' (b, row group) ids are
//    appended once to irr (irr[0] = count) through the rgflag dedupe array.
__global__ void mask_rows_kernel(const uint8_t* __restrict__ mask, long long sB, long long sN, const uint8_t* __restrict__ keyany,
                                 uint8_t* __restrict__ rowkind, int* __restrict__ rgflag, int* __restrict__ irr, int B, int N, int S, int S64) {
  const int lane = threadIdx.x & 31;
  const long long row = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= (long long)B * N) return;
  const int b = (int)(row / N), i = (int)(row - (long long)b * N);
  const uint8_t* m = mask + (long long)b * sB + (long long)i * sN;
  bool diff = false;
  for (int k = lane; k < S; k += 32) diff |= (m[k] != 0) != (keyany[(long long)b * S64 + k] != 0);
  diff = __any_sync(~0u, diff);
  if (lane == 0) {
    rowkind[row] = diff ? 1 : 0;
    if (diff) {
      const int YG = (N + R - 1) / R, e = b * YG + i / R;
      if (atomicExch(&rgflag[e], 1) == 0) irr[1 + atomicAdd(irr, 1)] = e;
    }
  }
}

// 3. per-row tables for the general passes, one warp per (b, i, key tile): keep bits of both 32-key halves by ballot, the tile class,
//    and ktendr[b][i] = tiles up to the row's last kept key (0 when the row keeps nothing).
__global__ void mask_tables_kernel(const uint8_t* __restrict__ mask, long long sB, long long sN, uint32_t* __restrict__ mtab, uint8_t* __restrict__ ctab,
                                   int* __restrict__ ktendr, int B, int N, int S, int nkt) {
  const int lane = threadIdx.x & 31;
  const long long w = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), nw = (long long)B * N * nkt;
  if (w >= nw) return;
  const int kt = (int)(w % nkt); const long long row = w / nkt; const int b = (int)(row / N), i = (int)(row - (long long)b * N);
  const uint8_t* m = mask + (long long)b * sB + (long long)i * sN;
  const int ka = kt * BN + lane, kb = ka + 32;
  const uint32_t w0 = __ballot_sync(~0u, ka < S && m[ka] != 0), w1 = __ballot_sync(~0u, kb < S && m[kb] != 0);
  if (lane == 0) {
    const int cnt = __popc(w0) + __popc(w1);
    mtab[2 * w] = w0; mtab[2 * w + 1] = w1;
    ctab[w] = (uint8_t)(cnt == 0 ? CLS_SKIP : (cnt == BN ? CLS_FULL : CLS_MIXED));
    if (cnt) atomicMax(&ktendr[row], kt + 1);
  }
}

MW_DEVI float to_f32(float x) { return x; }
MW_DEVI float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
MW_DEVI float to_f32(__half x) { return __half2float(x); }

// bias [B,H,S,S] (any float dtype, element strides) -> fp32 staged [B,H,S128,S64] / scale, permuted so that the TMA box [128 rows x 32
// keys] starting at (q tile, 32-key block) lands in smem with every 16-byte chunk = one mma C fragment of one thread:
// chunk ((m8*8 + gi)*16 + pk) of a box holds bias[q0][k0], [q0][k0+1], [q0+8][k0], [q0+8][k0+1] with q0 = 16*m8 + gi, k0 = 2*kp,
// kp = pk ^ 4*(gi&1) (the XOR spreads the two rows of a quarter-warp over different banks).  Entries with q >= S or k >= S are 0.
// Keys k >= S, and keys no row attends (keyany[b][k] == 0, when a mask was staged), become -inf columns: exp2 of them is an
// exact 0, so the mask-free kernel serves partial tiles and padding masks.  (Padding rows q >= S keep finite entries.)
template <typename T>
__global__ void stage_bias_kernel(const T* __restrict__ bias, long long sB, long long sH, long long sQ, long long sK,
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

}  // namespace mw

// ---------------------------------------------------------------- torch binding --------------------------------------------------
#include <torch/types.h>
#include <vector>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

torch::Tensor stage_bias(torch::Tensor bias4, double scale, c10::optional<torch::Tensor> keyany) {   // bias4: [B,H,S,S] fp32 | bf16 | fp16, any strides; keyany: [B,S64] uint8 from stage_mask, or none
  TORCH_CHECK(bias4.dim() == 4 && bias4.size(2) == bias4.size(3), "bias must be [B,H,S,S]");
  c10::cuda::CUDAGuard guard(bias4.device());
  const int B = bias4.size(0), H = bias4.size(1), S = bias4.size(2), S64 = (S + 63) / 64 * 64, S128 = (S + 127) / 128 * 128;
  auto out = torch::empty({B, H, S128, S64}, bias4.options().dtype(torch::kFloat32));
  const long long n = (long long)B * H * S128 * S64;
  const int threads = 256; const int blocks = (int)std::min<long long>((n + threads - 1) / threads, 132 * 32);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const float inv_scale = (float)(1.0 / scale);
  const uint8_t* ka = nullptr;
  if (keyany.has_value()) { TORCH_CHECK(keyany->scalar_type() == torch::kUInt8 && keyany->is_contiguous() && keyany->numel() == (long long)B * S64, "keyany [B,S64] uint8"); ka = keyany->data_ptr<uint8_t>(); }
  if (bias4.scalar_type() == torch::kFloat32)
    mw::stage_bias_kernel<float><<<blocks, threads, 0, st>>>(bias4.data_ptr<float>(), bias4.stride(0), bias4.stride(1), bias4.stride(2), bias4.stride(3), ka, out.data_ptr<float>(), B, H, S, S128, S64, inv_scale);
  else if (bias4.scalar_type() == torch::kBFloat16)
    mw::stage_bias_kernel<__nv_bfloat16><<<blocks, threads, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(bias4.data_ptr()), bias4.stride(0), bias4.stride(1), bias4.stride(2), bias4.stride(3), ka, out.data_ptr<float>(), B, H, S, S128, S64, inv_scale);
  else if (bias4.scalar_type() == torch::kFloat16)
    mw::stage_bias_kernel<__half><<<blocks, threads, 0, st>>>(reinterpret_cast<const __half*>(bias4.data_ptr()), bias4.stride(0), bias4.stride(1), bias4.stride(2), bias4.stride(3), ka, out.data_ptr<float>(), B, H, S, S128, S64, inv_scale);
  else TORCH_CHECK(false, "bias dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

CUtensorMap map_qkv(const torch::Tensor& x, int rows) {          // [B,N,H,S,D] bf16, d-stride 1, other strides multiples of 8; box = R pair rows x rows x D
  const uint64_t dims[5] = {(uint64_t)x.size(4), (uint64_t)x.size(3), (uint64_t)x.size(2), (uint64_t)x.size(1), (uint64_t)x.size(0)};
  const uint64_t str[4] = {(uint64_t)x.stride(3) * 2, (uint64_t)x.stride(2) * 2, (uint64_t)x.stride(1) * 2, (uint64_t)x.stride(0) * 2};
  const uint32_t box[5] = {(uint32_t)mw::D, (uint32_t)rows, 1, (uint32_t)mw::R, 1};
  return mw::make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, x.data_ptr(), dims, str, box, CU_TENSOR_MAP_SWIZZLE_64B);
}

// mask [B,N,S] uint8 -> {keyany [B,S64] uint8 (key attended by some row), rowkind [B,N] uint8 (1 = row differs from keyany), ktend [B] int32,
// irr int32 [1 + B*YG] (irr[0] = count, then the (b*YG + yg) row groups holding an irregular row), mtab int32 [B,N,nkt,2] + ctab uint8 [B,N,nkt]
// + ktendr int32 [B,N] (per-row keep bits / tile class / extent for the general passes)}
std::vector<torch::Tensor> stage_mask(torch::Tensor mask_u8) {
  TORCH_CHECK(mask_u8.scalar_type() == torch::kUInt8 && mask_u8.dim() == 3 && mask_u8.stride(2) == 1, "mask [B,N,S] uint8, key stride 1");
  c10::cuda::CUDAGuard guard(mask_u8.device());
  const int B = mask_u8.size(0), N = mask_u8.size(1), S = mask_u8.size(2), S64 = (S + 63) / 64 * 64, YG = (N + mw::R - 1) / mw::R;
  auto opt8 = mask_u8.options(); auto opt32 = mask_u8.options().dtype(torch::kInt32);
  auto keyany = torch::empty({B, S64}, opt8), rowkind = torch::empty({B, N}, opt8);
  auto irr = torch::zeros({1 + B * YG}, opt32), rgflag = torch::zeros({B * YG}, opt32), ktend = torch::zeros({B}, opt32);
  const int nkt = (S + mw::BN - 1) / mw::BN;
  auto mtab = torch::empty({B, N, nkt, 2}, opt32), ktendr = torch::zeros({B, N}, opt32); auto ctab = torch::empty({B, N, nkt}, opt8);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  mw::mask_or_kernel<<<dim3((S64 + 127) / 128, B), 1024, 0, st>>>(mask_u8.data_ptr<uint8_t>(), mask_u8.stride(0), mask_u8.stride(1), keyany.data_ptr<uint8_t>(), ktend.data_ptr<int>(), N, S, S64);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const long long rows = (long long)B * N; const int wpb = 8;
  mw::mask_rows_kernel<<<(unsigned)((rows + wpb - 1) / wpb), wpb * 32, 0, st>>>(mask_u8.data_ptr<uint8_t>(), mask_u8.stride(0), mask_u8.stride(1), keyany.data_ptr<uint8_t>(),
      rowkind.data_ptr<uint8_t>(), rgflag.data_ptr<int>(), irr.data_ptr<int>(), B, N, S, S64);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  { const long long nw = (long long)B * N * nkt; const int wpb2 = 8;
    mw::mask_tables_kernel<<<(unsigned)((nw + wpb2 - 1) / wpb2), wpb2 * 32, 0, st>>>(mask_u8.data_ptr<uint8_t>(), mask_u8.stride(0), mask_u8.stride(1),
        reinterpret_cast<uint32_t*>(mtab.data_ptr<int>()), ctab.data_ptr<uint8_t>(), ktendr.data_ptr<int>(), B, N, S, nkt);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); }
  return {keyany, rowkind, irr, ktend, mtab, ctab, ktendr};
}

void fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias_staged, c10::optional<torch::Tensor> mask_u8,
         c10::optional<torch::Tensor> rowkind, c10::optional<torch::Tensor> irr, c10::optional<torch::Tensor> ktend,
         c10::optional<torch::Tensor> mtab, c10::optional<torch::Tensor> ctab, c10::optional<torch::Tensor> ktendr,
         double scale, torch::Tensor out, torch::Tensor fix, int64_t dbg, c10::optional<torch::Tensor> tl) {
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.dim() == 5 && q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16, "q/k/v: [B,N,H,S,D] bf16");
  const int B = q.size(0), N = q.size(1), H = q.size(2), S = q.size(3), Dd = q.size(4);
  TORCH_CHECK(Dd == mw::D, "D must be 32"); TORCH_CHECK(S >= 1 && S <= mw::MAX_S, "S out of range (1 <= S <= 65536)");
  TORCH_CHECK((long long)B * H <= 65535 && (N + mw::R - 1) / mw::R <= 65535, "grid limits: B*H <= 65535, N <= 196605");
  for (auto* t : {&q, &k, &v}) {
    const int mult = (t == &q) ? 2 : 8, align = (t == &q) ? 4 : 16;   // Q: 4-byte loads; K, V: TMA
    TORCH_CHECK(t->stride(4) == 1, "d stride must be 1");
    for (int d = 0; d < 4; ++d) TORCH_CHECK(t->stride(d) % mult == 0 && t->stride(d) >= 0, "q/k/v strides: non-negative multiples of ", mult, " elements");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % align == 0, align, "-byte alignment");
  }
  const int S64 = (S + 63) / 64 * 64, S128 = (S + 127) / 128 * 128;
  TORCH_CHECK(bias_staged.scalar_type() == torch::kFloat32 && bias_staged.is_contiguous() && bias_staged.dim() == 4 &&
              bias_staged.size(0) == B && bias_staged.size(1) == H && bias_staged.size(2) == S128 && bias_staged.size(3) == S64, "staged bias shape [B,H,S128,S64]");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.stride(4) == 1 && out.stride(3) % 8 == 0, "out layout");
  mw::Params p;
  p.tmK = map_qkv(k, mw::BN); p.tmV = map_qkv(v, mw::BN);
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); p.q_sB = q.stride(0); p.q_sN = q.stride(1); p.q_sH = q.stride(2); p.q_sS = q.stride(3);
  {
    const uint64_t dims[4] = {(uint64_t)S64, (uint64_t)S128, (uint64_t)H, (uint64_t)B};
    const uint64_t str[3] = {(uint64_t)S64 * 4, (uint64_t)S128 * S64 * 4, (uint64_t)H * S128 * S64 * 4};
    const uint32_t box[4] = {32, (uint32_t)mw::BM, 1, 1};
    p.tmB = mw::make_map(CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, bias_staged.data_ptr(), dims, str, box, CU_TENSOR_MAP_SWIZZLE_NONE);
  }
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); p.o_sB = out.stride(0); p.o_sN = out.stride(1); p.o_sH = out.stride(2); p.o_sS = out.stride(3);
  p.mask = nullptr; p.m_sB = p.m_sN = 0; p.rowkind = nullptr; p.list = nullptr; p.list_mul = 1; p.ktend = nullptr; p.mtab = nullptr; p.ctab = nullptr; p.ktendr = nullptr;
  const int QT = (S + mw::BM - 1) / mw::BM, YG = (N + mw::R - 1) / mw::R;
  if (mask_u8.has_value()) {                                     // staged by stage_mask: the bias carries the batch-OR key mask
    const torch::Tensor& m = *mask_u8;
    TORCH_CHECK(m.scalar_type() == torch::kUInt8 && m.dim() == 3 && m.size(0) == B && m.size(1) == N && m.size(2) == S && m.stride(2) == 1, "mask [B,N,S] uint8");
    TORCH_CHECK(rowkind.has_value() && irr.has_value() && ktend.has_value() && rowkind->numel() == (long long)B * N && irr->numel() >= 1 + (long long)B * YG && ktend->numel() == B && ktend->scalar_type() == torch::kInt32, "mask meta (stage_mask)");
    const int nkt = (S + mw::BN - 1) / mw::BN;
    TORCH_CHECK(mtab.has_value() && ctab.has_value() && ktendr.has_value() && mtab->scalar_type() == torch::kInt32 && mtab->is_contiguous() && mtab->numel() == (long long)B * N * nkt * 2 &&
                ctab->scalar_type() == torch::kUInt8 && ctab->is_contiguous() && ctab->numel() == (long long)B * N * nkt && ktendr->scalar_type() == torch::kInt32 && ktendr->numel() == (long long)B * N,
                "mask tables (stage_mask): mtab int32 [B,N,nkt,2], ctab uint8 [B,N,nkt], ktendr int32 [B,N]");
    p.ktend = ktend->data_ptr<int>();
    p.mask = m.data_ptr<uint8_t>(); p.m_sB = m.stride(0); p.m_sN = m.stride(1);
    p.rowkind = rowkind->data_ptr<uint8_t>();
    p.mtab = reinterpret_cast<const uint32_t*>(mtab->data_ptr<int>()); p.ctab = ctab->data_ptr<uint8_t>(); p.ktendr = ktendr->data_ptr<int>();
  }
  p.B = B; p.N = N; p.H = H; p.S = S; p.n_ktiles = (S + mw::BN - 1) / mw::BN;
  p.c1 = (float)(scale * 1.4426950408889634);
  {                                                              // q-tile block: its staged bias rows (qb x 128 x S64 x 4 B) within ~16 MB of L2
    const long long per_qtile = (long long)mw::BM * S64 * 4;
    p.qb = (int)std::max<long long>(1, std::min<long long>(QT, (16ll << 20) / per_qtile));
    if (dbg >> 12) p.qb = (int)std::min<long long>(QT, dbg >> 12);            // experiments: qb override
  }
  p.dbg = (int)dbg;
  p.tl = nullptr;
  if (dbg & 64) { TORCH_CHECK(tl.has_value() && tl->scalar_type() == torch::kInt64 && tl->numel() >= (long long)mw::TL_CTAS * (mw::WARPS + 1) * mw::TL_T * 4, "timeline buffer"); p.tl = (long long*)tl->data_ptr<int64_t>(); }
  const int smem = (int)sizeof(mw::Smem) + 1024;
  static bool configured = false;
  static int num_sms = 132;
  if (!configured) {
    for (const void* fn : {(const void*)mw::triattn_mw_fwd<false, false>, (const void*)mw::triattn_mw_fwd<false, true>, (const void*)mw::triattn_mw_list<false>, (const void*)mw::triattn_mw_list<true>})
      C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int dev = 0; C10_CUDA_CHECK(cudaGetDevice(&dev)); C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
    configured = true;
  }
  dim3 grid(QT, YG, B * H), block(mw::THREADS);
  const long long nct = (long long)grid.x * grid.y * grid.z;
  TORCH_CHECK(fix.scalar_type() == torch::kInt32 && fix.is_contiguous() && fix.numel() >= 2 + 3 * nct, "fix buffer: int32 [>= 2 + 3 * #CTAs]");
  p.fix = fix.data_ptr<int>();
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(p.fix + 2, 0, sizeof(int), st));   // this call's fix count; fix[0], fix[1] (census) are owned by the caller
  const int npers = (int)std::min<long long>(nct, num_sms);
  // 1. mask-free pass over every CTA tile (masked-everywhere keys are -inf bias columns; CTAs of irregular rows only skip)
  if (dbg & 512) mw::triattn_mw_fwd<false, true><<<grid, block, smem, st>>>(p);    // experiments: max-tracking (FA2-style offsets) main pass
  else mw::triattn_mw_fwd<false, false><<<grid, block, smem, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // 2. irregular rows (own mask pattern): general instantiation over the row groups listed by stage_mask (normally none)
  if (p.mask) {
    mw::Params pl = p; pl.list = irr->data_ptr<int>(); pl.list_mul = QT * H;
    mw::triattn_mw_list<false><<<npers, block, smem, st>>>(pl);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  // 3. fix pass: CTA tiles whose max-free softmax overflowed, recomputed with max tracking (normally none)
  {
    mw::Params pf = p; pf.list = p.fix + 2; pf.list_mul = 1;
    mw::triattn_mw_list<true><<<npers, block, smem, st>>>(pf);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

int64_t fix_elems(int64_t B, int64_t N, int64_t H, int64_t S) {
  return 3 + 3 * ((S + mw::BM - 1) / mw::BM) * ((N + mw::R - 1) / mw::R) * B * H;
}

int64_t smem_bytes() { return (int64_t)sizeof(mw::Smem) + 1024; }

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd, "triangle attention forward, many-warp mma.sync (sm_90a)");
  m.def("stage_bias", &stage_bias, "bias [B,H,S,S] (+ keyany) -> fp32 [B,H,S128,S64] / scale in C-fragment order, -inf on excluded keys");
  m.def("stage_mask", &stage_mask, "mask [B,N,S] uint8 -> (keyany [B,S64], rowkind [B,N], irregular row-group list, ktend [B], mtab, ctab, ktendr)");
  m.def("smem_bytes", &smem_bytes);
  m.def("fix_elems", &fix_elems, "size of the int32 fix-up buffer fwd() needs");
}
