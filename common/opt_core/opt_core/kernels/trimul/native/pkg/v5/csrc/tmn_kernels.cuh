// SPDX-License-Identifier: Apache-2.0
// tmn_kernels.cuh — triangle-multiplication prologue (K1) and epilogue (K3) kernels, one family parameterised over
//   C_Z (pair channels) x C_H (hidden channels) in {64,128,256,384}^2, z dtype (bf16 | fp32-resident z under bf16 compute), tile shape,
//   weight-ring depth, LayerNorm summation mode, mask on/off, dormant save-intermediates.
// Derived from trimul_tx 1.2 (trimul_tx.cu: c_z = c_hidden = 256, bf16); the structure is kept and every width-specific constant became a
// template/constexpr parameter of a config struct (tuning data), so there is one code path for all widths.
//
//   K1:  ab[ch, i, j] = bf16( sigmoid(LN_in(z)[t] . Wg[ch]) * (LN_in(z)[t] . Wp[ch]) * mask[t] ),  t = (i, j), ch in [0, 2 C_H) (a | b planes),
//        written channel-major into planes [2 C_H][Np][Np] (zero pad) that one strided-batched GEMM contracts.
//   K3:  out[t, :] = (z[t, :] +) cvt( sigmoid(LN_in(z)[t] . Wog^T) * (LN_out(X[:, t]) . Wo^T) )   (X = the contraction result planes [C_H][Np][Np]).
// Structure (both, sm_90a member): persistent CTAs of 3 warpgroups — WG0 producer (TMA bulk-tensor loads into 128B-swizzled smem, mbarrier rings),
// WG1/WG2 consumers (ldmatrix -> LayerNorm in registers -> wgmma m64n64k16 bf16->fp32 with A from registers and the weight chunk from smem ->
// fused epilogue, software-pipelined one weight block ahead).  Numerics: LN output -> bf16; gate*proj(*mask) on fp32 accumulators -> bf16;
// fp32 LN statistics (two-pass, centred; for fp32 z the statistics and the normalisation read the fp32 values).
// Layering for other architectures: everything inside `namespace sm90` is the mainloop/staging policy (TMA + mbarrier + wgmma); the LayerNorm
// fragment math, the epilogue arithmetic and the parameter blocks are architecture-neutral and shared.
#pragma once
// ---- compile-time structure switches (defaults = the measured configuration; `python -m trimul_native.build --define K=V` builds the alternative
//      for A/B comparison; every alternative computes the same values in the same order: identical bytes)
#ifndef TMN_K1_START_OFFSET
#define TMN_K1_START_OFFSET 2   // K1: consumer warpgroup 1 starts its first tile when warpgroup 0 has retired the MMAs of its first TMN_K1_START_OFFSET weight
#endif                          //   blocks, so the two warpgroups' MMA bursts and SFU / store epilogues interleave instead of coinciding (clamped to the ring depth)
#ifndef TMN_K1_BULK_STORE
#define TMN_K1_BULK_STORE 1     // K1 plane store: 1 = one cp.async.bulk.tensor store per weight block per warpgroup from the [32 ch][64 tok] staging buffer
#endif                          //   (tiles with BJ % 64 == 0 and 16-byte plane rows); 0 = lds128 + st.global per lane (always used by the other tiles)
#ifndef TMN_K3_BULK_STORE
#define TMN_K3_BULK_STORE 1     // K3 output, bf16 pair tensor in / bf16 out: 1 = residual added in registers (the pair's z re-read from the operand tile by
#endif                          //   ldmatrix in the accumulator layout), stmatrix staging, one cp.async.bulk.tensor store per warp slice; 0 = the staged vector
                                //   pass (lds128 (+ residual) + st.global), which the fp32-z / fp32-out / pre-normalised operand modes always use
#ifndef TMN_K3_REGS_24_240
#define TMN_K3_REGS_24_240 1    // K3 producer / consumer register split: 1 = 24 / 240, 0 = 40 / 232 (K1's split); both balance 384 threads x 168 registers
#endif
#include "tmn_ptx.cuh"
#include "common/tmn_math.cuh"   // the numerics statement shared with the other architecture members

namespace tmn {

constexpr int NTHREADS = 384;          // K3: 1 producer + 2 consumer warpgroups (K1: K1Cfg::NTHR, from the tile)
constexpr int BM = 128;                // K3: pair rows (tokens) per CTA tile step (2 consumer warpgroups x 64 rows); K1: K1Cfg::BMT
constexpr int SMEM_LIMIT = 232448;     // sm_90 max dynamic shared memory per block (227 KB)

// ============================================================================================================ parameter blocks (POD)
// Host packing contract (kernel.py): tensor maps first (each 128 B, 64-byte aligned), then 8-byte pointers, then 4-byte scalars; sizeof is a
// multiple of 64.  tmn_info() (tmn_sm90.cu) reports sizeof/offsetof so the Python packer verifies the layout at load time.
struct K1Params {
  CUtensorMap tm_z;      // 3D z tile source: dims [C_Z][N (fast token axis)][N (slow token axis)] elements of z's dtype, box [128/esz][BJ][BI], SW128, OOB -> 0
  CUtensorMap tm_w;      // 2D [C_Z (k)][4 C_H (n)] bf16 (gate|proj block-interleaved rows), box [64][64], SW128
  CUtensorMap tm_ab;     // 3D planes store target: dims [Np (j)][Np (i)][2 C_H (ch)] bf16, box [64][1][32], SW128 (used when vec: Np % 8 == 0)
  const float* mask;     // fp32 mask or nullptr; element (i, j) at mask[i * ms_i + j * ms_j]
  const float* gamma;    // LN_in weight [C_Z] fp32
  const float* beta;     // LN_in bias [C_Z] fp32
  __nv_bfloat16* ab;     // planes [2 C_H][Np][Np] bf16 (Np = N allowed: unpadded planes; Np % 8 != 0 takes 2-byte stores)
  float* stats;          // nullable; SAVE instantiations write (mean, rstd) of LN_in per pair row: stats[2 (i N + j) + {0, 1}]
  __nv_bfloat16* xz;     // nullable; EMITX instantiations write the LayerNorm OUTPUT bf16(LN_in(z)) per pair row: xz[(i xs_i + j xs_j) + c] (the K3
                         // gate operand of the fp32-resident form, so K3 never re-normalises and reads a bf16 tile)
  int N, Np, tiles_j, num_tiles, vec, ms_i, ms_j;
  float eps;
  int xs_i, xs_j, pad0, pad1;   // xz element strides of the tile's slow (i) and fast (j) token axes
};

struct K3Params {
  CUtensorMap tm_z;      // 3D z [C_Z][N (j)][N (i)] (z dtype), box [128/esz][BJ][BI], SW128: the gate operand tile
  CUtensorMap tm_x;      // 3D X planes [Np (j)][Np (i)][C_H (ch)] bf16, box [64][1][64], SW128
  CUtensorMap tm_wg;     // 2D W_og [C_Z (k)][C_Z (n)] bf16, box [64][32], SW128
  CUtensorMap tm_wo;     // 2D W_o  [C_H (k)][C_Z (n)] bf16, box [64][32], SW128
  CUtensorMap tm_out;    // 3D output store target (bf16 z): dims [C_Z (c)][N (j)][N (i)] bf16, box [64][16][1], SW128
  const float* gamma_in; const float* beta_in; const float* gamma_out; const float* beta_out;   // fp32 [C_Z], [C_Z], [C_H], [C_H]
  const void* zres;      // z [N][N][C_Z] (z dtype): residual source (read from global / L2)
  void* out;             // [N][N][C_Z], z's dtype
  unsigned long long* prof;   // dev instrumentation (TMN_DEV_PROF builds): per-phase cycle sums, else unused (may be null)
  int N, Np, tiles_j, num_tiles, residual, pad0;
  float eps; int pad1;
};

// ============================================================================================================ configs (tuning data)
// SCHED_ (K1 weight-ring / MMA schedule; tuning data like the ring shape): 0 = two accumulator sets, block b+1's MMAs in flight during block b's
// epilogue, all slots of a block waited for before its first MMA and released together after it retired (trimul_tx 1.2's schedule); 1 = one block
// at a time: each slot is waited for right before ITS k-steps (the MMAs start when the first slot has landed), the block retires, all its slots are
// released BEFORE the epilogue (the producer refills them during it), one accumulator set.  -1 = by the ring: schedule 0 needs room for a third
// block (two in flight + one landing) — a resident ring or NSLOT >= 3 SPB — else schedule 1.  Same arithmetic in the same order: identical bytes.
template <int CZ_, int CH_, bool ZF32_, int BI_, int BJ_, int NSLOT_, int SKCH_, int SCHED_ = -1>
struct K1Cfg {
  static constexpr int CZ = CZ_, CH = CH_, BI = BI_, BJ = BJ_, NSLOT = NSLOT_, SKCH = SKCH_;
  static constexpr bool ZF32 = ZF32_;
  static constexpr int ESZ = ZF32 ? 4 : 2;
  // CTA shape follows the tile: BI x BJ = 64 NCWG tokens -> NCWG consumer warpgroups (one m64 MMA tile of 64 token rows each) + 1 producer
  // warpgroup; a 64-token tile (one consumer warpgroup, 256 threads) runs two CTAs per SM.  Register split: the producer keeps PROD_REGS, the
  // consumers share the rest of the CTA's launch allocation (65536 / (NTHR * MINB) per thread in 8-register granules), at most 232 each.
  static constexpr int BMT = BI * BJ;                   // tokens per CTA tile step
  static constexpr int NCWG = BMT / 64;                 // consumer warpgroups
  static constexpr int NTHR = 128 * (NCWG + 1);         // threads per CTA (== the unit's launch bound)
  static constexpr int MINB = NCWG == 1 ? 2 : 1;        // CTAs per SM (== the unit's launch bound)
  static constexpr int PROD_REGS = 40;
  static constexpr int LAUNCH_REGS = (65536 / (NTHR * MINB)) / 8 * 8;
  static constexpr int CONS_REGS_ = ((LAUNCH_REGS * NTHR - 128 * PROD_REGS) / (128 * NCWG)) / 8 * 8;
  static constexpr int CONS_REGS = CONS_REGS_ > 232 ? 232 : CONS_REGS_;
  static constexpr int CHUNK_CH = 128 / ESZ;           // channels per z chunk: a chunk is [BMT tok][128 B], 128B-swizzled rows (16 KB at 128 tokens)
  static constexpr int CHUNK_BYTES = BMT * 128;
  static constexpr int NKCA = CZ / CHUNK_CH;            // z chunks per tile
  static constexpr int KS = CZ / 16;                    // k-steps of the projection MMAs
  static constexpr int NKC = CZ / 64;                   // 64-wide k chunks of a weight block ([64 n][64 k] bf16 = 8 KB each)
  static constexpr int SLOT_BYTES = SKCH * 8192;        // ring slot = SKCH consecutive k chunks of one block
  static constexpr int SPB = NKC / SKCH;                // ring slots per weight block
  static constexpr int NBLK = CH / 16;                  // n64 weight blocks = 4 C_H / 64 (each: 32 gate rows | 32 proj rows -> 32 plane channels)
  static constexpr bool W_RESIDENT = NSLOT >= NBLK * SPB;   // the whole weight stream fits the ring: loaded once per CTA, never released
  static constexpr int SCHED = SCHED_ >= 0 ? SCHED_ : ((W_RESIDENT || NSLOT >= 3 * SPB) ? 0 : 1);
  static_assert(SCHED == 0 || SCHED == 1, "K1 schedules: 0 (two blocks in flight) | 1 (one block, per-slot waits)");
  static constexpr int SMEM_A = NKCA * CHUNK_BYTES;
  static constexpr int SMEM_W = NSLOT * SLOT_BYTES;
  static constexpr int SMEM_STAGE = NCWG * 8192;        // per consumer warpgroup: 2 x [32 ch][64 tok] bf16 epilogue transpose buffers
  static constexpr int SMEM_GB = 2 * CZ * 4;            // gamma, beta fp32
  static constexpr int NBAR = 2 + 2 * NSLOT;
  static constexpr int SMEM_BAR = ((NBAR * 8 + 127) / 128) * 128;
  static constexpr int SMEM = SMEM_A + SMEM_W + SMEM_STAGE + SMEM_GB + SMEM_BAR;
  static_assert(BMT % 64 == 0 && NCWG >= 1 && NCWG <= 4, "tile = 64, 128, 192 or 256 tokens (1..4 consumer warpgroups)");
  static_assert(BI <= 256 && BJ <= 256, "TMA box extents");
  static_assert(CONS_REGS >= 96, "consumer register share");
  static_assert(BJ >= 8 && BJ % 8 == 0, "an 8-token store granule must stay inside one pair row");
  static_assert(CZ % 64 == 0 && CH % 32 == 0, "C_Z multiple of 64, C_H multiple of 32");
  static_assert(NKC % SKCH == 0, "SKCH must divide C_Z / 64");
  static_assert(NBLK % 2 == 0, "two weight blocks in flight");
  static_assert(W_RESIDENT || NSLOT >= 2 * SPB, "ring must hold two weight blocks");
  static_assert(SMEM * MINB <= SMEM_LIMIT, "shared memory over the sm_90 limit (MINB CTAs per SM)");
};

// MODE: 0 = bf16 z (operand tile = z, LN_in in K3, residual from the resident tile, bf16 out); 1 = fp32 z tile (LN_in from shared memory in fp32,
// fp32 residual from the tile, fp32 out; twice the tile bytes: split-N at C_Z = 256); 2 = pre-normalised bf16 operand (K1 wrote bf16(LN_in(z)); no
// LN_in here), fp32 residual read from global, fp32 out — the fp32-resident form at widths whose fp32 tile does not fit.
// MODE 3 = bf16 operand tile holding a bf16 CAST of an fp32 z (LN_in applied here), fp32 residual read from the fp32 z in global, fp32 out.
// MODE 4 = fp32 operand tile (LN_in on fp32 values, as mode 1), the UPDATE written in bf16 (the compute dtype), no residual.
template <int CZ_, int CH_, int MODE_, int BI_, int BJ_, int NSLOT_, int NACC_ = 1>
struct K3Cfg {
  static constexpr int CZ = CZ_, CH = CH_, BI = BI_, BJ = BJ_, NSLOT = NSLOT_, NACC = NACC_, MODE = MODE_;
  static constexpr bool ZF32 = MODE_ == 1 || MODE_ == 4;   // fp32 operand tile
  static constexpr bool OUT32 = MODE_ >= 1 && MODE_ <= 3;  // fp32 residual + output
  static constexpr bool STG32 = MODE_ == 1;             // fp32 staging of z + o (mode 1); every other mode stages the bf16 update
  static constexpr bool PRENORM = MODE_ == 2;           // operand already normalised
  static constexpr bool RESG = MODE_ == 2 || MODE_ == 3;   // residual from the fp32 z in global memory (the tile holds normalised / cast rows)
  static constexpr bool NORES = MODE_ == 4;             // mode 4 ('g'): fp32 z tile, bf16 UPDATE out, never a residual (the fp32 sum is mode 1's job)
  static_assert(MODE_ >= 0 && MODE_ <= 4, "MODE");
  static constexpr int BMT = BI * BJ;                   // tokens per CTA tile step: 128 = the two consumer warpgroups split the tokens (64 each, all output
  static constexpr bool SPLITN = BMT == 64;             // blocks); 64 = both warpgroups serve the same 64 tokens and alternate output blocks (halves the tile smem)
  static constexpr int NSUB = BMT / 64;                 // 64-token X sub-tiles per tile
  static constexpr int ESZ = ZF32 ? 4 : 2;
  static constexpr int CHUNK_CH = 128 / ESZ;
  static constexpr int CHUNK_BYTES = BMT * 128;         // z chunk = [BMT tok][128 B], 128B-swizzled rows
  static constexpr int NKCZ = CZ / CHUNK_CH;            // z chunks per tile
  static constexpr int KSG = CZ / 16, KSP = CH / 16;    // k-steps: gate (K = C_Z), projection (K = C_H)
  static constexpr int BN = 32;                         // output block width (channels): m64n32 products, 16 + 16 accumulator registers per set
  static constexpr int NB = CZ / BN;                    // output blocks per token row
  static constexpr int NBW = SPLITN ? NB / 2 : NB;      // blocks served by one warpgroup per tile
  static constexpr int NKG = CZ / 64, NKP = CH / 64;    // 4 KB k-chunks ([32 n][64 k]) per gate | projection weight block
  static constexpr int SLOTG = CZ * 64, SLOTP = CH * 64;
  static constexpr int SLOT_BYTES = SLOTG > SLOTP ? SLOTG : SLOTP;   // one ring slot holds one output block's gate OR projection weight rows
  static constexpr bool W_RESIDENT = NSLOT >= 2 * NB;   // the whole W_og | W_o fits the ring: loaded once per CTA, never released
  static constexpr int W_CONSUMERS = 8;                 // every consumer warp arrives on every slot use (split-N: the warpgroup that does not multiply a
                                                        // block still waits its arrival and releases it, so each warp observes every phase of a slot in order)
  static constexpr int NKCX = CH / 64;                  // X chunks per 64-token sub-tile ([64 ch][64 tok] bf16 = 8 KB)
  static constexpr int OB = 16 * 2 * BN * ESZ;          // per-warp output staging slice [16 tok][64 ch = a block pair] in z's dtype (2 KB bf16 | 4 KB fp32)
  static constexpr int SMEM_X = NSUB * CH * 128;
  static constexpr int SMEM_Z = NKCZ * CHUNK_BYTES;
  static constexpr int SMEM_W = NSLOT * SLOT_BYTES;
  static constexpr int SMEM_OUT = 8 * OB;
  static constexpr int SMEM_GB = (2 * CZ + 2 * CH) * 4;
  static constexpr int NBAR = 2 * NKCZ + 2 + 2 * NSLOT;
  static constexpr int SMEM_BAR = ((NBAR * 8 + 127) / 128) * 128;
  static constexpr int SMEM = SMEM_X + SMEM_Z + SMEM_W + SMEM_OUT + SMEM_GB + SMEM_BAR;
  static_assert((BMT == 128 || BMT == 64) && BJ >= 64 && BJ % 64 == 0, "a consumer warpgroup serves 64 consecutive tokens of one pair row");
  static_assert(CZ % 64 == 0 && CH % 64 == 0, "C_Z, C_H multiples of 64");
  static_assert(NACC == 1 || (NACC == 2 && !SPLITN), "one or two accumulator sets (two: full-width tiles only)");
  static_assert(NBW % 2 == 0, "output blocks are finished in pairs (128-byte row segments)");
  static_assert(W_RESIDENT || NSLOT >= 4, "streamed ring: two blocks (gate + projection each) in flight");
  static_assert(SMEM <= SMEM_LIMIT, "shared memory over the sm_90 limit");
};

// ============================================================================================================ shared math (arch-neutral)
// byte offset inside a tile of 128-byte rows written by TMA with 128B swizzle (16-byte granule index ^= row % 8)
TMN_DEVI uint32_t swz128(uint32_t row, uint32_t col_byte) {
  return row * 128u + ((((col_byte >> 4) ^ (row & 7u)) << 4) | (col_byte & 15u));
}
TMN_DEVI float quad_sum(float v) {                     // butterfly across the 4 lanes holding one fragment row: xor 1, then xor 2
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}
TMN_DEVI float sigmoidf_(float g) { return math::sigmoid(g); }   // rcp.approx.ftz(1 + ex2.approx.ftz(-g log2 e)): the statement's sigmoid

// LN statistics of the two rows a thread holds (row A = the accumulator's first row, row B = row A + 8): {meanA, rstdA, meanB, rstdB}
struct LnStats { float mA, rA, mB, rB; };

// LayerNorm over the C = 16 KS values of two rows held as an m64kC A fragment (fa[ks][0|2] = row A, fa[ks][1|3] = row B; k = 16 ks + 2 (lane%4) +
// {0,1} (+8)), statistics fp32 (mean, then centred variance), affine, repacked to bf16 in place.  (Own summation order: the fast class.)
// SERIAL: each k-step's affine loads carry an opaque data dependency on the previous step's packed result, so the normalisation runs one
// k-step at a time (the scheduler otherwise hoists every step's loads + FMAs and defers the bf16 packing, doubling the live set: with a second
// operand's fragments resident that spills).  Costs one exposed shared-load latency per step; used where two fragment sets are live (K3).
// CLS: math::REF = the shared statement (rstd = rsqrt.approx.ftz(var + eps), y = fma((x - mean) rstd, gamma, beta), one bf16 rounding) = the default
// tolerance class; math::TX = trimul_tx 1.2's arithmetic (rsqrtf, y = fma(x, rstd gamma, fma(-mean rstd, gamma, beta))), kept for byte comparison only.
template <int KS, bool SERIAL = false, int CLS = math::REF>
TMN_DEVI LnStats ln_fragment(uint32_t (&fa)[KS][4], const float* sGamma, const float* sBeta, int lane, float eps) {
  constexpr float invC = 1.f / (16 * KS);
  float meanA, meanB;
  if (CLS == math::TX) {
    float sA_ = 0.f, sB_ = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      sA_ += bf16lo(fa[ks][0]) + bf16hi(fa[ks][0]) + bf16lo(fa[ks][2]) + bf16hi(fa[ks][2]);
      sB_ += bf16lo(fa[ks][1]) + bf16hi(fa[ks][1]) + bf16lo(fa[ks][3]) + bf16hi(fa[ks][3]);
    }
    meanA = quad_sum(sA_) * invC; meanB = quad_sum(sB_) * invC;
  } else {                                              // the statement: group sums of the 4 values a k-step holds per row, balanced tree over k-steps, lane butterfly
    float gA[KS], gB[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      gA[ks] = __fadd_rn(__fadd_rn(bf16lo(fa[ks][0]), bf16hi(fa[ks][0])), __fadd_rn(bf16lo(fa[ks][2]), bf16hi(fa[ks][2])));
      gB[ks] = __fadd_rn(__fadd_rn(bf16lo(fa[ks][1]), bf16hi(fa[ks][1])), __fadd_rn(bf16lo(fa[ks][3]), bf16hi(fa[ks][3])));
    }
    meanA = math::ln_mean(quad_sum(math::tree_sum(gA)), invC); meanB = math::ln_mean(quad_sum(math::tree_sum(gB)), invC);
  }
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);   // re-derive the fp32 values in each pass (2 ALU ops) instead of keeping them live
  float rA, rB;
  if (CLS == math::TX) {
    float vA = 0.f, vB = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      float d;
      d = bf16lo(fa[ks][0]) - meanA; vA += d * d; d = bf16hi(fa[ks][0]) - meanA; vA += d * d;
      d = bf16lo(fa[ks][2]) - meanA; vA += d * d; d = bf16hi(fa[ks][2]) - meanA; vA += d * d;
      d = bf16lo(fa[ks][1]) - meanB; vB += d * d; d = bf16hi(fa[ks][1]) - meanB; vB += d * d;
      d = bf16lo(fa[ks][3]) - meanB; vB += d * d; d = bf16hi(fa[ks][3]) - meanB; vB += d * d;
    }
    rA = rsqrtf(quad_sum(vA) * invC + eps); rB = rsqrtf(quad_sum(vB) * invC + eps);
  } else {
    float gA[KS], gB[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      gA[ks] = __fadd_rn(math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][0]), meanA), bf16hi(fa[ks][0]), meanA), math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][2]), meanA), bf16hi(fa[ks][2]), meanA));
      gB[ks] = __fadd_rn(math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][1]), meanB), bf16hi(fa[ks][1]), meanB), math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][3]), meanB), bf16hi(fa[ks][3]), meanB));
    }
    rA = math::ln_rstd(quad_sum(math::tree_sum(gA)), invC, eps); rB = math::ln_rstd(quad_sum(math::tree_sum(gB)), invC, eps);
  }
  const float mrA = meanA * rA, mrB = meanB * rB; (void)mrA; (void)mrB;
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);
  uint32_t chain[2] = {0u, 0u};                        // two interleaved chains: step ks waits for step ks-2 (two steps' loads in flight)
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const uint32_t k0 = (uint32_t)(16 * ks + 2 * (lane & 3)) + (SERIAL ? zero_dep(chain[ks & 1]) : 0u);
    const float2 g0 = lds64f(smem_u32(sGamma) + 4u * k0), b0 = lds64f(smem_u32(sBeta) + 4u * k0);
    const float2 g1 = lds64f(smem_u32(sGamma) + 4u * k0 + 32u), b1 = lds64f(smem_u32(sBeta) + 4u * k0 + 32u);
    if (CLS == math::TX) {                             // y = x * (r g) + (b - mean r g)
      fa[ks][0] = pack_bf16(fmaf(bf16lo(fa[ks][0]), rA * g0.x, fmaf(-mrA, g0.x, b0.x)), fmaf(bf16hi(fa[ks][0]), rA * g0.y, fmaf(-mrA, g0.y, b0.y)));
      fa[ks][1] = pack_bf16(fmaf(bf16lo(fa[ks][1]), rB * g0.x, fmaf(-mrB, g0.x, b0.x)), fmaf(bf16hi(fa[ks][1]), rB * g0.y, fmaf(-mrB, g0.y, b0.y)));
      fa[ks][2] = pack_bf16(fmaf(bf16lo(fa[ks][2]), rA * g1.x, fmaf(-mrA, g1.x, b1.x)), fmaf(bf16hi(fa[ks][2]), rA * g1.y, fmaf(-mrA, g1.y, b1.y)));
      fa[ks][3] = pack_bf16(fmaf(bf16lo(fa[ks][3]), rB * g1.x, fmaf(-mrB, g1.x, b1.x)), fmaf(bf16hi(fa[ks][3]), rB * g1.y, fmaf(-mrB, g1.y, b1.y)));
    } else {                                             // y = fma((x - mean) r, g, b): the statement
      fa[ks][0] = pack_bf16(math::ln_affine(bf16lo(fa[ks][0]), meanA, rA, g0.x, b0.x), math::ln_affine(bf16hi(fa[ks][0]), meanA, rA, g0.y, b0.y));
      fa[ks][1] = pack_bf16(math::ln_affine(bf16lo(fa[ks][1]), meanB, rB, g0.x, b0.x), math::ln_affine(bf16hi(fa[ks][1]), meanB, rB, g0.y, b0.y));
      fa[ks][2] = pack_bf16(math::ln_affine(bf16lo(fa[ks][2]), meanA, rA, g1.x, b1.x), math::ln_affine(bf16hi(fa[ks][2]), meanA, rA, g1.y, b1.y));
      fa[ks][3] = pack_bf16(math::ln_affine(bf16lo(fa[ks][3]), meanB, rB, g1.x, b1.x), math::ln_affine(bf16hi(fa[ks][3]), meanB, rB, g1.y, b1.y));
    }
    chain[ks & 1] = fa[ks][0] ^ fa[ks][3];
  }
  return LnStats{meanA, rA, meanB, rB};
}

// The same LayerNorm for an fp32-resident tile in shared memory (chunks of [BM rows][32 ch] fp32, 128-B swizzled rows, 16 KB apart): statistics and
// normalisation read the fp32 values (three passes over smem), only the LayerNorm OUTPUT is rounded to bf16 into the A fragment.
template <int KS, int CHUNK_BYTES = 16384>
TMN_DEVI LnStats ln_rows_f32(uint32_t (&fa)[KS][4], uint32_t sA_u, int rowA, int rowB, const float* sGamma, const float* sBeta, int lane, float eps) {
  constexpr float invC = 1.f / (16 * KS);
  const int q = lane & 3;
  auto addr = [&](int row, int ks, int hi8) -> uint32_t {          // fp32 pair (16 ks + 2 q + 8 hi8, +1) of `row`: chunk ks/2, byte col ((ks&1)*16 + 2q + 8 hi8) * 4
    return sA_u + (uint32_t)(ks >> 1) * (uint32_t)CHUNK_BYTES + swz128((uint32_t)row, (uint32_t)(((ks & 1) * 16 + 2 * q + 8 * hi8) * 4));
  };
  float gA[KS], gB[KS];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const float2 a0 = lds64f(addr(rowA, ks, 0)), a1 = lds64f(addr(rowA, ks, 1)), b0 = lds64f(addr(rowB, ks, 0)), b1 = lds64f(addr(rowB, ks, 1));
    gA[ks] = __fadd_rn(__fadd_rn(a0.x, a0.y), __fadd_rn(a1.x, a1.y)); gB[ks] = __fadd_rn(__fadd_rn(b0.x, b0.y), __fadd_rn(b1.x, b1.y));
  }
  const float meanA = math::ln_mean(quad_sum(math::tree_sum(gA)), invC), meanB = math::ln_mean(quad_sum(math::tree_sum(gB)), invC);
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const float2 a0 = lds64f(addr(rowA, ks, 0)), a1 = lds64f(addr(rowA, ks, 1)), b0 = lds64f(addr(rowB, ks, 0)), b1 = lds64f(addr(rowB, ks, 1));
    gA[ks] = __fadd_rn(math::ln_sq_acc(math::ln_sq(a0.x, meanA), a0.y, meanA), math::ln_sq_acc(math::ln_sq(a1.x, meanA), a1.y, meanA));
    gB[ks] = __fadd_rn(math::ln_sq_acc(math::ln_sq(b0.x, meanB), b0.y, meanB), math::ln_sq_acc(math::ln_sq(b1.x, meanB), b1.y, meanB));
  }
  const float rA = math::ln_rstd(quad_sum(math::tree_sum(gA)), invC, eps), rB = math::ln_rstd(quad_sum(math::tree_sum(gB)), invC, eps);
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const int k0 = 16 * ks + 2 * q;
    const float2 g0 = lds64f(smem_u32(sGamma) + 4u * k0), c0 = lds64f(smem_u32(sBeta) + 4u * k0);
    const float2 g1 = lds64f(smem_u32(sGamma) + 4u * k0 + 32u), c1 = lds64f(smem_u32(sBeta) + 4u * k0 + 32u);
    const float2 a0 = lds64f(addr(rowA, ks, 0)), a1 = lds64f(addr(rowA, ks, 1)), b0 = lds64f(addr(rowB, ks, 0)), b1 = lds64f(addr(rowB, ks, 1));
    fa[ks][0] = pack_bf16(math::ln_affine(a0.x, meanA, rA, g0.x, c0.x), math::ln_affine(a0.y, meanA, rA, g0.y, c0.y));
    fa[ks][1] = pack_bf16(math::ln_affine(b0.x, meanB, rB, g0.x, c0.x), math::ln_affine(b0.y, meanB, rB, g0.y, c0.y));
    fa[ks][2] = pack_bf16(math::ln_affine(a1.x, meanA, rA, g1.x, c1.x), math::ln_affine(a1.y, meanA, rA, g1.y, c1.y));
    fa[ks][3] = pack_bf16(math::ln_affine(b1.x, meanB, rB, g1.x, c1.x), math::ln_affine(b1.y, meanB, rB, g1.y, c1.y));
  }
  return LnStats{meanA, rA, meanB, rB};
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// LayerNorm with the summation order of the reference library's bf16-input LayerNorm kernels (cuequivariance_torch 0.11.1 triangle_multiplicative_update),
// so that the bf16 output is bit-identical to theirs at C = 256 (order established from the installed function's behaviour by trimul_tx 1.2; other
// widths use the same construction with C/64 chunks — bitwise identity there is a measured property, not assumed):
//   * the C channels are C/64 chunks of 64; chunks are accumulated elementwise first: A[c'] = ((x[c'] + x[c'+64]) + x[c'+128]) + ...;
//     for the centred squares the accumulation is contracted: acc = d0*d0, then acc = fma(d_k, d_k, acc);
//   * TREE 0 (token-major kernel, LN_in): t_w = ((A[8w] + A[8w+1]) + ... ) + A[8w+7] sequentially (w = 0..7), then a butterfly over w with xor
//     offsets 4, 2, 1;  TREE 1 (transposing kernel, LN_out, plane row length % 4 == 0): t[c'] = A[c'] + A[c'+32] (c' < 32), then a butterfly over c'
//     with xor offsets 2, 1, 16, 8, 4;  TREE 2 (the transposing kernel when the row length % 4 != 0): t_v = ((A[v] + A[v+4]) + A[v+8]) + ... + A[v+60]
//     sequentially (v = 0..3, 16 terms, stride 4), then S = (t0 + t2) + (t1 + t3);
//   * mean = S * rn(1/C) (the reference's division by the constant C is a multiply by the rounded reciprocal in its machine code); centred sum of
//     squares S2 (two passes); rstd = rsqrt.approx.ftz(fma(S2, rn(1/C), eps)) -- ONE rounding: the reference's code generator contracts the
//     divide's multiply with the eps add (at power-of-two C the scaling is exact and this equals rsqrt(rn(S2 / C) + eps); at C = 384 only the
//     fused form is bit-identical); y = fma((x - mean) * rstd, gamma, beta) -> bf16 (rn).
// Every fp32 operation below is an explicit round-to-nearest intrinsic so that the compiler cannot re-associate or contract differently.
template <int ROW, int KS>   // ROW 0: registers [ks][0],[ks][2]; ROW 1: [ks][1],[ks][3]
TMN_DEVI float frag_val(const uint32_t (&fa)[KS][4], int ks, int m) {
  const uint32_t r = fa[ks][ROW + (m >> 1) * 2];
  return (m & 1) ? bf16hi(r) : bf16lo(r);
}
template <int TREE, int ROW, int PASS, int KS>
TMN_DEVI float stock_row_sum(const uint32_t (&fa)[KS][4], float mean, int lane) {
  constexpr int NCH = KS / 4;                            // chunks of 64 channels
  float A[4][4];
#pragma unroll
  for (int ks4 = 0; ks4 < 4; ++ks4) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      if (PASS == 0) {
        float acc = frag_val<ROW>(fa, ks4, m);
#pragma unroll
        for (int c = 1; c < NCH; ++c) acc = __fadd_rn(acc, frag_val<ROW>(fa, ks4 + 4 * c, m));
        A[ks4][m] = acc;
      } else {
        float d = __fsub_rn(frag_val<ROW>(fa, ks4, m), mean); float acc = __fmul_rn(d, d);
#pragma unroll
        for (int c = 1; c < NCH; ++c) { d = __fsub_rn(frag_val<ROW>(fa, ks4 + 4 * c, m), mean); acc = __fmaf_rn(d, d, acc); }
        A[ks4][m] = acc;
      }
    }
  }
  const int q = lane & 3;
  if (TREE == 0) {
    float sr[4][2];
#pragma unroll
    for (int ks4 = 0; ks4 < 4; ++ks4) {
#pragma unroll
      for (int h = 0; h < 2; ++h) sr[ks4][h] = __fadd_rn(A[ks4][2 * h], A[ks4][2 * h + 1]);
    }
#pragma unroll
    for (int step = 1; step < 4; ++step) {
#pragma unroll
      for (int ks4 = 0; ks4 < 4; ++ks4) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float in = __shfl_sync(0xffffffffu, sr[ks4][h], (lane - 1) & 31);
          const float nxt = __fadd_rn(__fadd_rn(in, A[ks4][2 * h]), A[ks4][2 * h + 1]);
          sr[ks4][h] = (q == step) ? nxt : sr[ks4][h];
        }
      }
    }
    const float S = __fadd_rn(__fadd_rn(__fadd_rn(sr[0][0], sr[2][0]), __fadd_rn(sr[1][0], sr[3][0])),
                              __fadd_rn(__fadd_rn(sr[0][1], sr[2][1]), __fadd_rn(sr[1][1], sr[3][1])));
    return __shfl_sync(0xffffffffu, S, lane | 3);
  } else if (TREE == 1) {
    float u[2][4];
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2) {
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float t = __fadd_rn(A[k2][m], A[k2 + 2][m]);
        u[k2][m] = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, 1));
      }
    }
    const float v00 = __fadd_rn(u[0][0], u[0][1]), v01 = __fadd_rn(u[0][2], u[0][3]);
    const float v10 = __fadd_rn(u[1][0], u[1][1]), v11 = __fadd_rn(u[1][2], u[1][3]);
    const float w0 = __fadd_rn(v00, v10), w1 = __fadd_rn(v01, v11);
    const float zz = __fadd_rn(w0, w1);
    return __fadd_rn(zz, __shfl_xor_sync(0xffffffffu, zz, 2));
  } else {
    const int half = q >> 1;
    float sa = A[0][0], sb = A[0][1];
#pragma unroll
    for (int j = 1; j < 16; ++j) {
      const int ks4 = j >> 2, m0 = (j & 2) ? 2 : 0;
      const float va = __shfl_xor_sync(0xffffffffu, sa, 2), vb = __shfl_xor_sync(0xffffffffu, sb, 2);
      const float na = __fadd_rn(va, A[ks4][m0]), nb = __fadd_rn(vb, A[ks4][m0 + 1]);
      const bool mine = half == (j & 1);
      sa = mine ? na : sa; sb = mine ? nb : sb;
    }
    const float ua = __fadd_rn(sa, __shfl_xor_sync(0xffffffffu, sa, 1)), ub = __fadd_rn(sb, __shfl_xor_sync(0xffffffffu, sb, 1));
    const float S = __fadd_rn(ua, ub);
    return __shfl_sync(0xffffffffu, S, lane | 2);
  }
}
template <int TREE, bool SERIAL = false, int KS>   // SERIAL: the affine pass one k-step at a time (an opaque chain on the load address, as ln_fragment does; same values)
TMN_DEVI LnStats ln_stock(uint32_t (&fa)[KS][4], const float* sGamma, const float* sBeta, int lane, float eps) {
  constexpr float invC = 1.f / (16 * KS);
  const float meanA = __fmul_rn(stock_row_sum<TREE, 0, 0>(fa, 0.f, lane), invC);
  const float meanB = __fmul_rn(stock_row_sum<TREE, 1, 0>(fa, 0.f, lane), invC);
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);
  // rstd: the reference scales the centred sum of squares by 1/C and adds eps in ONE fused multiply-add (its divide-by-constant lowers to a
  // multiply by the rounded reciprocal, which the code generator contracts with the eps add); identical to the two-step form at power-of-two C.
  const float rA = rsqrt_approx_ftz(__fmaf_rn(stock_row_sum<TREE, 0, 1>(fa, meanA, lane), invC, eps));
  const float rB = rsqrt_approx_ftz(__fmaf_rn(stock_row_sum<TREE, 1, 1>(fa, meanB, lane), invC, eps));
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);
  uint32_t chain[2] = {0u, 0u};                        // SERIAL: step ks waits for step ks-2's packed result (two steps' loads in flight)
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const uint32_t k0 = (uint32_t)(16 * ks + 2 * (lane & 3)) + (SERIAL ? zero_dep(chain[ks & 1]) : 0u);
    const float2 g0 = lds64f(smem_u32(sGamma) + 4u * k0), b0 = lds64f(smem_u32(sBeta) + 4u * k0);          // volatile shared loads: issued here, per
    const float2 g1 = lds64f(smem_u32(sGamma) + 4u * k0 + 32u), b1 = lds64f(smem_u32(sBeta) + 4u * k0 + 32u);  // k-step, never hoisted en bloc (register budget)
#define TMN_LNY(x, mean, r, g, b) __fmaf_rn(__fmul_rn(__fsub_rn((x), (mean)), (r)), (g), (b))
    fa[ks][0] = pack_bf16(TMN_LNY(bf16lo(fa[ks][0]), meanA, rA, g0.x, b0.x), TMN_LNY(bf16hi(fa[ks][0]), meanA, rA, g0.y, b0.y));
    fa[ks][1] = pack_bf16(TMN_LNY(bf16lo(fa[ks][1]), meanB, rB, g0.x, b0.x), TMN_LNY(bf16hi(fa[ks][1]), meanB, rB, g0.y, b0.y));
    fa[ks][2] = pack_bf16(TMN_LNY(bf16lo(fa[ks][2]), meanA, rA, g1.x, b1.x), TMN_LNY(bf16hi(fa[ks][2]), meanA, rA, g1.y, b1.y));
    fa[ks][3] = pack_bf16(TMN_LNY(bf16lo(fa[ks][3]), meanB, rB, g1.x, b1.x), TMN_LNY(bf16hi(fa[ks][3]), meanB, rB, g1.y, b1.y));
#undef TMN_LNY
    if (SERIAL) chain[ks & 1] = fa[ks][0] ^ fa[ks][3];
  }
  return LnStats{meanA, rA, meanB, rB};
}

// A fragment (K = 16 KS channels of this warp's 16 rows) from a K-major bf16 tile of 16 KB chunks [BM rows][64 ch] (128-B swizzled rows)
template <int KS, int CHUNK_BYTES = 16384>
TMN_DEVI void load_frag_bf16(uint32_t (&f)[KS][4], uint32_t base_u, int rho0, int lane) {
  const int mat = lane >> 3, r8 = lane & 7;
  const int row = rho0 + r8 + ((mat & 1) ? 8 : 0);
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const int kc = ks >> 2, kin = (ks & 3) * 16 + ((mat & 2) ? 8 : 0);
    ldsm_x4(f[ks], base_u + kc * CHUNK_BYTES + swz128(row, kin * 2));
  }
}

namespace sm90 {

// ============================================================================================================ K1
// LNM: 1 = the shared statement, own summation order (default class); 2 = the reference library's summation order (bitwise class; bf16 z only);
//      4 = trimul_tx 1.2's arithmetic (comparison only).
// SAVE: the dormant save-intermediates path (LN_in statistics per pair row); when false no instruction of it exists.
template <class G, bool HAS_MASK, int LNM, bool SAVE, bool EMITX = false>
TMN_DEVI void k1_body(const K1Params& p) {
  constexpr int CZ = G::CZ, KS = G::KS, NSLOT = G::NSLOT, SPB = G::SPB, NBLK = G::NBLK, SKCH = G::SKCH, SLOT_BYTES = G::SLOT_BYTES, BI = G::BI, BJ = G::BJ;
  static_assert(!(G::ZF32 && LNM == 2), "the reference-order LayerNorm is defined on bf16 inputs");
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sW = smem + G::SMEM_A;
  uint8_t* sStage = sW + G::SMEM_W;
  float* sGamma = reinterpret_cast<float*>(sStage + G::SMEM_STAGE);
  float* sBeta = sGamma + CZ;
  uint64_t* bars = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sGamma) + G::SMEM_GB);
  uint64_t* barA_full = bars + 0;
  uint64_t* barA_empty = bars + 1;
  uint64_t* barW_full = bars + 2;               // [NSLOT]
  uint64_t* barW_empty = bars + 2 + NSLOT;       // [NSLOT]
  uint64_t* barGo = bars + G::NBAR;              // one-shot: warpgroup 0 -> warpgroup 1 start offset
  static_assert((G::NBAR + 1) * 8 <= G::SMEM_BAR, "barrier area");
  // start offset in weight blocks, clamped to what the ring can hold: with a streamed ring warpgroup 0 alone can retire at most NSLOT/SPB - 1 blocks
  // before it needs a slot that only both warpgroups together release (a larger offset would deadlock by construction)
  constexpr int K1OFF = TMN_K1_START_OFFSET <= 0 ? 0 : (G::W_RESIDENT || TMN_K1_START_OFFSET <= G::NSLOT / G::SPB - 1) ? TMN_K1_START_OFFSET : (G::NSLOT / G::SPB - 1);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wg = __shfl_sync(0xffffffffu, tid >> 7, 0);   // warp-uniform by construction: the register reallocation below (setmaxnreg) and the role split
                                                            // are per warp, and the allocator budgets each side of the branch by its setmaxnreg value
  if (tid == 0 && dyn_smem_size() < (uint32_t)G::SMEM) __trap();
  const int n_iter = (p.num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;     // tiles blockIdx.x + it * gridDim.x
  for (int i = tid; i < CZ; i += G::NTHR) { sGamma[i] = p.gamma[i]; sBeta[i] = p.beta[i]; }
  if (tid == 0) {
    mbar_init(barA_full, 1); mbar_init(barA_empty, 4 * G::NCWG);           // one arrival per consumer warp
    for (int s = 0; s < NSLOT; ++s) { mbar_init(barW_full + s, 1); mbar_init(barW_empty + s, 4 * G::NCWG); }
    if (K1OFF > 0) mbar_init(barGo, 1);
    fence_barrier_init();
    tma_prefetch_desc(&p.tm_z); tma_prefetch_desc(&p.tm_w);
    if (TMN_K1_BULK_STORE) tma_prefetch_desc(&p.tm_ab);
  }
  __syncthreads();

  if (wg == 0) {
    // ------------------------------------------------------------------ producer warpgroup
    setmaxnreg_dec<G::PROD_REGS>();
    if (warp == 0 && lane == 0) {                       // z tiles: one load per tile, issued as soon as the consumers have pulled the previous tile into registers
      for (int it = 0; it < n_iter; ++it) {
        const int tile = (int)blockIdx.x + it * (int)gridDim.x;
        if (it > 0) mbar_wait(barA_empty, (it - 1) & 1);
        const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
        mbar_arrive_expect_tx(barA_full, G::SMEM_A);
#pragma unroll
        for (int kc = 0; kc < G::NKCA; ++kc) tma_load_3d(sA + kc * G::CHUNK_BYTES, &p.tm_z, barA_full, kc * G::CHUNK_CH, j0, i0);
      }
    } else if (warp == 1 && lane == 0) {                // weight ring: NBLK * SPB slot loads per tile, identical sequence every tile (L2-resident after the first)
      const uint32_t per_tile = (uint32_t)(NBLK * SPB);
      const uint32_t n_w = G::W_RESIDENT ? (n_iter > 0 ? per_tile : 0u) : (uint32_t)n_iter * per_tile;
      for (uint32_t w_iter = 0; w_iter < n_w; ++w_iter) {
        const int s = w_iter % NSLOT; const uint32_t u = w_iter / NSLOT;
        if (u > 0) mbar_wait(barW_empty + s, (u - 1) & 1);
        mbar_arrive_expect_tx(barW_full + s, SLOT_BYTES);
        const int hb = w_iter % per_tile, b = hb / SPB, h = hb % SPB;
#pragma unroll
        for (int kk = 0; kk < SKCH; ++kk) tma_load_2d(sW + s * SLOT_BYTES + kk * 8192, &p.tm_w, barW_full + s, (h * SKCH + kk) * 64, 64 * b);
      }
    }
    __syncwarp();
    return;
  }

  // ------------------------------------------------------------------ consumer warpgroups
  setmaxnreg_inc<G::CONS_REGS>();
  const int cw = wg - 1;                 // 0 .. NCWG-1 : token rows [64 cw, 64 cw + 64) of the tile
  const int wiw = warp & 3;              // warp within warpgroup: rows 16*wiw.. of the m64 tile
  const uint32_t sA_u = smem_u32(sA), sW_u = smem_u32(sW);
  const uint32_t stage_u = smem_u32(sStage) + (uint32_t)(cw * 8192);   // this warpgroup's 2 x 4 KB staging buffers [32 ch][64 tok]
  const size_t plane = (size_t)p.Np * (size_t)p.Np;
  const int rho0 = 64 * cw + 16 * wiw;                  // first of this warp's 16 token rows (tile-relative)
  const int rowA = rho0 + (lane >> 2), rowB = rowA + 8;  // this thread's accumulator rows
  // stmatrix role: lane -> (matrix idx, row rr): channel ch = 16 hq + 8 (idx>>1) + rr; token granule tg = 2 wiw + (idx&1) (8 tokens each), stored at tg ^ (ch & 7)
  const int idx = lane >> 3, rr = lane & 7;
  const int chq = 8 * (idx >> 1) + rr;
  const uint32_t sts_off0 = (uint32_t)chq * 128 + (uint32_t)(((2 * wiw + (idx & 1)) ^ (chq & 7)) * 16);
  const uint32_t sts_off1 = sts_off0 + 16 * 128;
  // store role: warp wiw stores channels 8 wiw .. 8 wiw + 7 of the block; instruction e (0,1): lanes 8k..8k+7 -> channel 8 wiw + 4 e + k, granule lane%8
  const int st_ch0 = 8 * wiw + (lane >> 3), st_g = lane & 7;
  const uint32_t ld_off0 = (uint32_t)st_ch0 * 128 + (uint32_t)((st_g ^ (st_ch0 & 7)) * 16);
  const uint32_t ld_off1 = (uint32_t)(st_ch0 + 4) * 128 + (uint32_t)((st_g ^ ((st_ch0 + 4) & 7)) * 16);
  const int bar_id = 1 + cw;                            // named barrier of this warpgroup
  uint32_t w_iter = 0;
  // TMA plane store (BJ % 64 == 0: this warpgroup's 64 tokens are one plane row segment): the elected thread stores the whole [32 ch][64 tok] staging
  // buffer of a block with one bulk-tensor store; the buffer written two blocks later is the same one, so the elected thread drains the previous
  // store's smem READ before each block's warpgroup barrier (issued a block earlier: normally complete already).
  constexpr bool TMAST = (TMN_K1_BULK_STORE != 0) && (BJ % 64 == 0);
  const bool st_elect = (wiw == 0) && (lane == 0);

  if (K1OFF > 0 && cw == 1 && n_iter > 0) mbar_wait(barGo, 0);   // start offset: warpgroup 0's first K1OFF weight blocks retire before warpgroup 1 starts
  for (int t_local = 0; t_local < n_iter; ++t_local) {
    const int tile = (int)blockIdx.x + t_local * (int)gridDim.x;
    const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
    // ---- this thread's two rows: validity + mask; this lane's two store runs: pointers + predicates
    const int iA = i0 + rowA / BJ, jA = j0 + rowA % BJ, iB = i0 + rowB / BJ, jB = j0 + rowB % BJ;
    const bool vA = (iA < p.N) && (jA < p.N), vB = (iB < p.N) && (jB < p.N);
    float mA = vA ? 1.f : 0.f, mB = vB ? 1.f : 0.f;
    if (HAS_MASK) {
      if (vA) mA = __ldg(p.mask + (size_t)iA * p.ms_i + (size_t)jA * p.ms_j);
      if (vB) mB = __ldg(p.mask + (size_t)iB * p.ms_i + (size_t)jB * p.ms_j);
    }
    const int rho_g = 64 * cw + 8 * st_g;                                  // first token of this lane's store granule
    const int is_ = i0 + rho_g / BJ, js_ = j0 + rho_g % BJ;
    const bool st_ok = (is_ < p.Np) && (js_ < p.Np);
    const int st_n = min(8, p.Np - js_);                                   // valid tokens of this lane's 8-token granule (ragged planes)
    __nv_bfloat16* gp0 = p.ab + (size_t)st_ch0 * plane + (size_t)is_ * p.Np + js_;   // + 32 b * plane per block; + 4 * plane for e = 1
    const size_t blk_stride = 32 * plane, e_stride = 4 * plane;
    const int iw = i0 + (64 * cw) / BJ, jw = j0 + (64 * cw) % BJ;              // this warpgroup's plane row / first token (TMA store coordinates)

    // ---- A fragments: from the swizzled z tile, LayerNorm applied (bf16: in registers; fp32: from smem, output bf16)
    mbar_wait(barA_full, t_local & 1);
    uint32_t fa[KS][4];
    LnStats st;
    if (G::ZF32) {
      st = ln_rows_f32<KS, G::CHUNK_BYTES>(fa, sA_u, rowA, rowB, sGamma, sBeta, lane, p.eps);
    } else {
      load_frag_bf16<KS, G::CHUNK_BYTES>(fa, sA_u, rho0, lane);
    }
    // WAR across proxies: the reads above go through the generic proxy, the producer's refill of sA is an async-proxy (TMA) write.  The mbarrier
    // release alone does not order the two; fence.proxy.async does.
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) mbar_arrive(barA_empty);              // the producer may refill sA with the next tile now
    if (!G::ZF32) {
      constexpr bool LNSER = KS >= 24;                   // the affine pass serialised at 24+ k-steps (96 live fragment registers): schedule only, same values
      if (LNM == 1) st = ln_fragment<KS, LNSER>(fa, sGamma, sBeta, lane, p.eps);
      else if (LNM == 4) st = ln_fragment<KS, false, math::TX>(fa, sGamma, sBeta, lane, p.eps);
      else if (LNM == 2) st = ln_stock<0, LNSER>(fa, sGamma, sBeta, lane, p.eps);
      else st = LnStats{0.f, 1.f, 0.f, 1.f};
    }
    if (EMITX) {                                          // the normalised rows as bf16 for K3's gate operand: 4-byte stores, 4 lanes cover 32 contiguous bytes
      const int q = lane & 3;
      __nv_bfloat16* xa = p.xz + ((size_t)iA * (size_t)p.xs_i + (size_t)jA * (size_t)p.xs_j) + 2 * q;
      __nv_bfloat16* xb = p.xz + ((size_t)iB * (size_t)p.xs_i + (size_t)jB * (size_t)p.xs_j) + 2 * q;
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        if (vA) { stg32(xa + 16 * ks, fa[ks][0]); stg32(xa + 16 * ks + 8, fa[ks][2]); }
        if (vB) { stg32(xb + 16 * ks, fa[ks][1]); stg32(xb + 16 * ks + 8, fa[ks][3]); }
      }
    }
    if (SAVE) {                                           // dormant save-intermediates: LN_in statistics per pair row (row owner lane of each quad)
      if (p.stats != nullptr && (lane & 3) == 0) {
        if (vA) { p.stats[2 * ((size_t)iA * p.N + jA)] = st.mA; p.stats[2 * ((size_t)iA * p.N + jA) + 1] = st.rA; }
        if (vB) { p.stats[2 * ((size_t)iB * p.N + jB)] = st.mB; p.stats[2 * ((size_t)iB * p.N + jB) + 1] = st.rB; }
      }
    }

    // ---- NBLK weight blocks of 32 channels (n64 = gate | proj); MMAs of block b+1 are issued before the epilogue of block b
    float acc0[32], acc1[32];
    auto slot_of = [&](uint32_t wi) -> int { return G::W_RESIDENT ? (int)(wi % (uint32_t)(NBLK * SPB)) : (int)(wi % NSLOT); };
    auto phase_of = [&](uint32_t wi) -> uint32_t { return G::W_RESIDENT ? 0u : ((wi / NSLOT) & 1u); };
    auto issue_block = [&](float (&ac)[32], uint32_t wi) {
      uint32_t dlo[SPB], dhi[SPB];
#pragma unroll
      for (int j = 0; j < SPB; ++j) {
        const int s = slot_of(wi + j);
        mbar_wait(barW_full + s, phase_of(wi + j));
        const uint64_t d = smem_desc(sW_u + s * SLOT_BYTES, 16, 1024, 1);
        dlo[j] = (uint32_t)d; dhi[j] = (uint32_t)(d >> 32);
      }
#pragma unroll
      for (int i = 0; i < 32; ++i) ac[i] = 0.f;
      fence_regs(ac);
      wgmma_fence();
      mma_chain<SKCH>(ac, fa, dlo, dhi);
      wgmma_commit();
    };
    auto epilogue = [&](float (&ac)[32], int b, uint32_t wi, bool release) {   // release: free the block's ring slots here (schedule 0) or not (already freed)
      fence_regs(ac);
      __syncwarp();
      if (K1OFF > 0 && cw == 0 && t_local == 0 && b == K1OFF - 1 && wiw == 0 && lane == 0) mbar_arrive(barGo);   // warpgroup 1 may start
      if (release && !G::W_RESIDENT) {
        if (lane == 0) {
#pragma unroll
          for (int j = 0; j < SPB; ++j) mbar_arrive(barW_empty + slot_of(wi + j));
        }
      }
      uint32_t pk[4][2];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float vA0 = math::gate(ac[4 * q + 0], ac[4 * (q + 4) + 0], mA), vA1 = math::gate(ac[4 * q + 1], ac[4 * (q + 4) + 1], mA);   // sigmoid(g) p m, fp32
        float vB0 = math::gate(ac[4 * q + 2], ac[4 * (q + 4) + 2], mB), vB1 = math::gate(ac[4 * q + 3], ac[4 * (q + 4) + 3], mB);
        if (!vA) { vA0 = 0.f; vA1 = 0.f; }
        if (!vB) { vB0 = 0.f; vB1 = 0.f; }
        pk[q][0] = pack_bf16(vA0, vA1); pk[q][1] = pack_bf16(vB0, vB1);
      }
      const uint32_t sbuf = stage_u + (uint32_t)((b & 1) * 4096);            // double-buffered: one warpgroup barrier per block
      stsm_x4_t(sbuf + sts_off0, pk[0][0], pk[0][1], pk[1][0], pk[1][1]);      // channels 0..15 of the block
      stsm_x4_t(sbuf + sts_off1, pk[2][0], pk[2][1], pk[3][0], pk[3][1]);      // channels 16..31
      if (TMAST && p.vec) {
        fence_proxy_async();                                                 // the stmatrix writes -> visible to the async proxy that reads the buffer
        if (st_elect) tma_store_wait_read<0>();                              // the store issued from the other buffer's twin a block ago has read it
        named_bar_sync(bar_id, 128);
        if (st_elect) { tma_store_3d(&p.tm_ab, sStage + cw * 8192 + (b & 1) * 4096, jw, iw, 32 * b); tma_store_commit(); }
      } else {
        named_bar_sync(bar_id, 128);
        const uint4 v0 = lds128(sbuf + ld_off0), v1 = lds128(sbuf + ld_off1);
        const size_t bo = (size_t)b * blk_stride;
        if (st_ok) {
          if (p.vec) { stg128(gp0 + bo, v0); stg128(gp0 + bo + e_stride, v1); }
          else { stg_ragged(gp0 + bo, v0, st_n); stg_ragged(gp0 + bo + e_stride, v1, st_n); }   // unpadded planes with Np % 8 != 0: element stores, j < Np
        }
      }
    };

    if constexpr (G::SCHED == 1) {
      // ---- schedule 1: one block at a time — per-slot waits under the earlier slots' MMAs, full retire, every slot released before the epilogue
      (void)acc1; (void)issue_block;
#pragma unroll 1
      for (int b = 0; b < NBLK; ++b, w_iter += SPB) {
        static_for<SPB>([&](auto Jc) {
          constexpr int J = decltype(Jc)::value;
          const int s = slot_of(w_iter + J);
          mbar_wait(barW_full + s, phase_of(w_iter + J));
          const uint64_t d = smem_desc(sW_u + s * SLOT_BYTES, 16, 1024, 1);
          if (J == 0) {
#pragma unroll
            for (int i = 0; i < 32; ++i) acc0[i] = 0.f;
            fence_regs(acc0);
          }
          wgmma_fence();
          mma_group<SKCH, KS, J>(acc0, fa, (uint32_t)d, (uint32_t)(d >> 32));
          wgmma_commit();                                // a group per slot: the wait for the next slot never sits inside an open (uncommitted) MMA group
        });
        wgmma_wait<0>();
        if (!G::W_RESIDENT) {                            // every slot of the block back to the producer before any epilogue work (measured: earlier than
#pragma unroll                                           // inside the epilogue after its register fence is worth ~6 % of K1 at c_z 384)
          for (int j = 0; j < SPB; ++j) { __syncwarp(); if (lane == 0) mbar_arrive(barW_empty + slot_of(w_iter + j)); }
        }
        epilogue(acc0, b, w_iter, false);
      }
    } else {
    issue_block(acc0, w_iter);
#pragma unroll 1
    for (int b = 0; b < NBLK; b += 2, w_iter += 2 * SPB) {
      issue_block(acc1, w_iter + SPB);
      wgmma_wait<1>();
      epilogue(acc0, b, w_iter, true);
      if (b + 2 < NBLK) { issue_block(acc0, w_iter + 2 * SPB); wgmma_wait<1>(); }
      else { wgmma_wait<0>(); }
      epilogue(acc1, b + 1, w_iter + SPB, true);
    }
    }
    // the A fragments are read ASYNCHRONOUSLY by every block's wgmma: keep their registers allocated until the last group of this tile has retired
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);
  }
  if (TMAST && st_elect) tma_store_wait_all();           // every bulk store of this CTA complete before its shared memory is released
}


// ============================================================================================================ K3
// Tile = BI x BJ = 128 tokens (WG cw serves 64 consecutive tokens of row i0 + (64 cw)/BJ).  Per tile: the X tile (2 sub-tiles [C_H ch][64 tok] bf16,
// MN-major A operand via ldmatrix.trans) is pulled into registers first and released at once (the next tile's X load overlaps everything else);
// the z tile arrives as chunks with their own barriers: all feed the gate operand, and (bf16 z) chunk group b is released after output block b
// has taken its residual from it, so the producer refills it with the next tile's z while later blocks still run; (fp32 z) the chunks are released
// as soon as the gate operand is built, residual and output go through registers (fp32, 8-byte vectors, 32-B contiguous per quad).
// LNM: 1 = the shared statement, own order; 2 = reference-library order with TREE 1 for LN_out (N % 4 == 0); 3 = with TREE 2 (N % 4 != 0); 4 = trimul_tx 1.2's arithmetic.
template <class G, int LNM, bool UPD = false>
TMN_DEVI void k3_body(const K3Params& p) {
  constexpr int CZ = G::CZ, CH = G::CH, KSG = G::KSG, KSP = G::KSP, NKG = G::NKG, NKP = G::NKP, NB = G::NB, NBW = G::NBW, BN = G::BN, NSLOT = G::NSLOT;
  constexpr int NKCZ = G::NKCZ, BI = G::BI, BJ = G::BJ, OB = G::OB, SLOT_BYTES = G::SLOT_BYTES, CHB = G::CHUNK_BYTES, ESZ = G::ESZ;
  constexpr bool SPLITN = G::SPLITN, ZF32 = G::ZF32, OUT32 = G::OUT32, STG32 = G::STG32, PRENORM = G::PRENORM, RESG = G::RESG, NORES = G::NORES;
  constexpr int OSZ = OUT32 ? 4 : 2;                     // output / residual element size
  static_assert(!(ZF32 && (LNM == 2 || LNM == 3)), "the reference-order LayerNorm is defined on bf16 inputs");
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sX = smem;
  uint8_t* sZ = sX + G::SMEM_X;
  uint8_t* sW = sZ + G::SMEM_Z;
  uint8_t* sOut = sW + G::SMEM_W;
  float* sGin = reinterpret_cast<float*>(sOut + G::SMEM_OUT);
  float* sBin = sGin + CZ; float* sGout = sBin + CZ; float* sBout = sGout + CH;
  uint64_t* bars = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sGin) + G::SMEM_GB);
  uint64_t* barZ_full = bars;                 // [NKCZ]
  uint64_t* barX_full = bars + NKCZ;
  uint64_t* barX_empty = barX_full + 1;       // X released (8 consumer warps) as soon as the fragments sit in registers
  uint64_t* barZ_empty = barX_empty + 1;      // [NKCZ]: z chunk c released by each consumer warp after its last read (fragment load, or the residual
  uint64_t* barW_full = barZ_empty + NKCZ;    //         read of the block pair living in that chunk) -> refilled with the next tile's chunk meanwhile
  uint64_t* barW_empty = barW_full + NSLOT;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wg = __shfl_sync(0xffffffffu, tid >> 7, 0);   // warp-uniform by construction: the register reallocation below (setmaxnreg) and the role split
                                                            // are per warp, and the allocator budgets each side of the branch by its setmaxnreg value
  if (tid == 0 && dyn_smem_size() < (uint32_t)G::SMEM) __trap();
  const int n_iter = (p.num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  for (int i = tid; i < CZ; i += NTHREADS) { sGin[i] = p.gamma_in[i]; sBin[i] = p.beta_in[i]; }
  for (int i = tid; i < CH; i += NTHREADS) { sGout[i] = p.gamma_out[i]; sBout[i] = p.beta_out[i]; }
  // register split: 24 / 240 for the bf16-tile kernels (bulk-store epilogue); the fp32-tile / cast / pre-normalised modes keep 40 / 232 (their live set is larger)
  constexpr bool R240 = (TMN_K3_REGS_24_240 != 0) && !G::ZF32 && !G::OUT32 && !G::RESG && !G::PRENORM;
  if (tid == 0) {
    for (int kc = 0; kc < NKCZ; ++kc) mbar_init(barZ_full + kc, 1);
    mbar_init(barX_full, 1); mbar_init(barX_empty, 8);
    for (int kc = 0; kc < NKCZ; ++kc) mbar_init(barZ_empty + kc, 8);
    for (int s = 0; s < NSLOT; ++s) { mbar_init(barW_full + s, 1); mbar_init(barW_empty + s, G::W_CONSUMERS); }
    fence_barrier_init();
    tma_prefetch_desc(&p.tm_z); tma_prefetch_desc(&p.tm_x); tma_prefetch_desc(&p.tm_wg); tma_prefetch_desc(&p.tm_wo);
    if ((TMN_K3_BULK_STORE != 0) && !G::ZF32 && !G::OUT32 && !G::RESG && !G::PRENORM) tma_prefetch_desc(&p.tm_out);
  }
  __syncthreads();

  if (wg == 0) {
    // ================================================================== producers: warp 0 operand stage, warp 1 weight ring
    setmaxnreg_dec<(R240 ? 24 : 40)>();
    if (warp == 0 && lane == 0) {
      for (int t_local = 0; t_local < n_iter; ++t_local) {
        const int tile = (int)blockIdx.x + t_local * (int)gridDim.x;
        const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
        if (t_local > 0) mbar_wait(barX_empty, (t_local - 1) & 1);
        mbar_arrive_expect_tx(barX_full, G::SMEM_X);
#pragma unroll
        for (int h = 0; h < G::NSUB; ++h) {              // sub-tile h = tokens 64h..64h+63 of the tile: [C_H ch][64 tok], 64-channel chunks of 8 KB
          const int ih = i0 + (64 * h) / BJ, jh = j0 + (64 * h) % BJ;
#pragma unroll
          for (int kx = 0; kx < G::NKCX; ++kx) tma_load_3d(sX + h * (CH * 128) + kx * 8192, &p.tm_x, barX_full, jh, ih, kx * 64);
        }
#pragma unroll 1
        for (int kc = 0; kc < NKCZ; ++kc) {
          if (t_local > 0) mbar_wait(barZ_empty + kc, (t_local - 1) & 1);
          mbar_arrive_expect_tx(barZ_full + kc, CHB);
          tma_load_3d(sZ + kc * CHB, &p.tm_z, barZ_full + kc, kc * G::CHUNK_CH, j0, i0);
        }
      }
    } else if (warp == 1 && lane == 0) {
      const uint32_t per_tile = 2u * NB;
      const uint32_t n_w = G::W_RESIDENT ? (n_iter > 0 ? per_tile : 0u) : (uint32_t)n_iter * per_tile;
      for (uint32_t seq = 0; seq < n_w; ++seq) {         // per output block b: the projection rows (W_o[32 b .., :]) then the gate rows (W_og[32 b .., :])
        const int s = seq % NSLOT; const uint32_t u = seq / NSLOT;
        if (u > 0) mbar_wait(barW_empty + s, (u - 1) & 1);
        const int b = (int)((seq % per_tile) >> 1), which = (int)(seq & 1u);
        if (which == 0) {
          mbar_arrive_expect_tx(barW_full + s, G::SLOTP);
#pragma unroll
          for (int kc = 0; kc < NKP; ++kc) tma_load_2d(sW + s * SLOT_BYTES + kc * 4096, &p.tm_wo, barW_full + s, kc * 64, BN * b);
        } else {
          mbar_arrive_expect_tx(barW_full + s, G::SLOTG);
#pragma unroll
          for (int kc = 0; kc < NKG; ++kc) tma_load_2d(sW + s * SLOT_BYTES + kc * 4096, &p.tm_wg, barW_full + s, kc * 64, BN * b);
        }
      }
    }
    __syncwarp();
    return;
  }

  // ================================================================== consumers
  setmaxnreg_inc<(R240 ? 240 : 232)>();
#ifdef TMN_DEV_PROF
  unsigned long long pf[12] = {0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull};
  long long pt0 = clock64();
#define PF(i) do { long long t1_ = clock64(); pf[i] += (unsigned long long)(t1_ - pt0); pt0 = t1_; } while (0)
#else
#define PF(i) do { } while (0)
#endif
  const int cw = wg - 1, wiw = warp & 3, mat = lane >> 3, r8 = lane & 7;
  const int tok0 = SPLITN ? 0 : 64 * cw;                 // this WG's first token (tile-relative)
  const uint32_t sZ_u = smem_u32(sZ), sX_u = smem_u32(sX) + (uint32_t)((tok0 / 64) * (CH * 128)), sW_u = smem_u32(sW);
  const uint32_t stg_u = smem_u32(sOut) + (uint32_t)((4 * cw + wiw) * OB);   // this warp's staging slice [16 tok][32 ch]
  const int rho0 = tok0 + 16 * wiw;                      // tile-relative first row of this warp
  const int rowA = rho0 + (lane >> 2), rowB = rowA + 8;   // this thread's accumulator rows (tile-relative)
  const int gq = lane >> 2, q2 = 2 * (lane & 3);         // accumulator row within the warp's 16 (rows gq, gq + 8), column pair base within an 8-column group
  // Staging slice per warp = [16 tok][64 ch] (a pair of output blocks) in z's dtype.  bf16: 128-B rows, granule ^= row & 7 (swz128); stmatrix x4 matrix
  // mi = lane/8 covers rows 8 (mi&1) + r8 and the 8 channels 32 h + 16 kb + 8 (mi>>1) (h = block parity in the pair); register q of the x4 holds the
  // C-fragment pair (row gq + 8 (q&1), cols 16 kb + 8 (q>>1) + q2).  fp32: 256-B rows, granule ^= row & 7.  The vector pass moves 16-byte granules
  // g = lane + 32 it: bf16 row g/8, channels 8 (g%8) ..; fp32 row g/16, channels 4 (g%16) ..  -> 128 / 256 contiguous bytes per token row.
  const int lrow = r8 + 8 * (mat & 1);
  constexpr int NGR = STG32 ? 8 : 4;                     // 16-byte granules per lane per block pair in the vector pass
  // TMA output store (bf16 z, bf16 out): the residual is added in registers (the pair's z fragments re-read with ldmatrix in the accumulator layout, which
  // for a 16-column slab is the A-fragment layout) before staging, and each warp's staged [16 tok][64 ch] slice leaves with one bulk-tensor store;
  // the slice is reused by the next pair once that store has READ it (waited for at the next pair's start, behind two blocks of MMAs).
  constexpr bool K3ST = (TMN_K3_BULK_STORE != 0) && !ZF32 && !OUT32 && !RESG && !PRENORM;
  uint32_t zdep = 0;
  for (int t_local = 0; t_local < n_iter; ++t_local) {
    const int tile = (int)blockIdx.x + t_local * (int)gridDim.x;
    const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
    const int iw = i0 + tok0 / BJ, jw = j0 + tok0 % BJ;               // this WG's 64 tokens: row iw, columns jw .. jw+63

    // ---- projection operand: X sub-tile [C_H ch rows][64 tok] -> ldmatrix.trans -> release the X stage -> LN_out (before z is touched:
    //      one raw fragment set live at a time keeps the normalisation inside the register budget)
    uint32_t fx[KSP][4];
    PF(0);                                               // 0: loop head / previous tile tail
    mbar_wait(barX_full, t_local & 1);
    PF(1);                                               // 1: X wait
    {
      const int tokc = 16 * wiw + ((mat & 1) ? 8 : 0);
#pragma unroll
      for (int ks = 0; ks < KSP; ++ks) {
        const int krow = 16 * ks + r8 + ((mat & 2) ? 8 : 0);
        ldsm_x4_t(fx[ks], sX_u + swz128(krow, tokc * 2));
      }
      uint32_t dep = 0;                                  // release once every read has returned (one destination register per instruction feeds the
#pragma unroll                                           // dependency); the proxy fence orders these generic-proxy reads before the async-proxy refill
      for (int ks = 0; ks < KSP; ++ks) dep ^= fx[ks][0];
      dep = zero_dep(__reduce_or_sync(0xffffffffu, dep));
      fence_proxy_async();
      if (lane == 0) mbar_arrive_dep(barX_empty, dep);
    }
    PF(2);                                               // 2: X ldmatrix + release
#ifndef TMN_DEV_NOLN
    if (LNM == 2) ln_stock<1>(fx, sGout, sBout, lane, p.eps); else if (LNM == 3) ln_stock<2>(fx, sGout, sBout, lane, p.eps);
    else if (LNM == 4) ln_fragment<KSP, true, math::TX>(fx, sGout, sBout, lane, p.eps); else ln_fragment<KSP, true>(fx, sGout, sBout, lane, p.eps);
#endif
    PF(3);                                               // 3: LN_out
    // ---- gate operand: z rows -> release the z stage -> LN_in
    uint32_t fz[KSG][4];
    for (int kc = 0; kc < NKCZ; ++kc) mbar_wait(barZ_full + kc, t_local & 1);
    PF(4);                                               // 4: z wait
    if (ZF32) ln_rows_f32<KSG, CHB>(fz, sZ_u, rowA, rowB, sGin, sBin, lane, p.eps);   // fp32 statistics + normalisation from smem -> bf16 fragments
    else load_frag_bf16<KSG, CHB>(fz, sZ_u, rho0, lane);
    {   // release the z chunks this warp will not touch again (all of them without residual; else those whose block pairs another warpgroup finishes)
      uint32_t dep = 0;
#pragma unroll
      for (int ks = 0; ks < KSG; ++ks) dep ^= fz[ks][0] ^ fz[ks][3];
      dep = zero_dep(__reduce_or_sync(0xffffffffu, dep));
      fence_proxy_async();
      if (lane == 0) {
#pragma unroll
        for (int kc = 0; kc < NKCZ; ++kc) {
          const int pair_b0 = (kc * G::CHUNK_CH) / (2 * BN) * 2;                        // first block of the pair whose channels live in chunk kc
          const bool mine = SPLITN ? (((pair_b0 >> 1) & 1) == cw) : true;             // split-N: pairs alternate between the warpgroups
          const bool now = !(p.residual && mine && !RESG && !NORES);                                // residual word 0: every chunk at once
          if (now) mbar_arrive_dep(barZ_empty + kc, dep);   // RESG: the residual is the fp32 z in global, the tile is free now
        }
      }
    }
    PF(5);                                               // 5: z load + release
#ifndef TMN_DEV_NOLN
    if (!ZF32 && !PRENORM) { if (LNM == 2 || LNM == 3) ln_stock<0>(fz, sGin, sBin, lane, p.eps); else if (LNM == 4) ln_fragment<KSG, true, math::TX>(fz, sGin, sBin, lane, p.eps); else ln_fragment<KSG, true>(fz, sGin, sBin, lane, p.eps); }
#endif
    PF(6);                                               // 6: LN_in

    // ---- output blocks of 32 channels: block q+1's MMAs are in flight while block q's epilogue runs (two accumulator sets)
    float accP0[16], accG0[16], accP1[16], accG1[16];
    auto blk_of = [&](int qb) -> int { return SPLITN ? 4 * (qb >> 1) + 2 * cw + (qb & 1) : qb; };   // qb-th block of this warpgroup (split-N: pairs alternate)
    auto slot_of = [&](uint32_t seq) -> int { return G::W_RESIDENT ? (int)(seq % (2u * NB)) : (int)(seq % NSLOT); };
    auto phase_of = [&](uint32_t seq) -> uint32_t { return G::W_RESIDENT ? 0u : ((seq / NSLOT) & 1u); };
    auto issue = [&](float (&accP)[16], float (&accG)[16], int qb) {
      const uint32_t seqP = ((uint32_t)t_local * NB + (uint32_t)blk_of(qb)) * 2u, seqG = seqP + 1u;
      {
        const int s = slot_of(seqP); mbar_wait(barW_full + s, phase_of(seqP));
        PF(7);                                           // 7: W(P) wait
        const uint64_t d = smem_desc(sW_u + s * SLOT_BYTES, 16, 1024, 1);
#pragma unroll
        for (int i = 0; i < 16; ++i) accP[i] = 0.f;
        fence_regs(accP);
        wgmma_fence();
        mma_chain32<KSP>(accP, fx, (uint32_t)d, (uint32_t)(d >> 32));
        wgmma_commit();
      }
      {
        PF(8);                                           // 8: P issue
        const int s = slot_of(seqG); mbar_wait(barW_full + s, phase_of(seqG));
        PF(7);                                           // 7: W(G) wait
        const uint64_t d = smem_desc(sW_u + s * SLOT_BYTES, 16, 1024, 1);
#pragma unroll
        for (int i = 0; i < 16; ++i) accG[i] = 0.f;
        fence_regs(accG);
        wgmma_fence();
        mma_chain32<KSG>(accG, fz, (uint32_t)d, (uint32_t)(d >> 32));
        wgmma_commit();
        PF(8);                                           // 8: G issue
      }
    };
    // vector-pass geometry of this lane for the tile: granule `it` covers token row jl + RSTEP it, 16 bytes at channel byte 16 cgl of a block pair
    // (a staging granule = 16 B of the staged slice: 8 bf16 update values (modes 0, 2) or 4 fp32 sums (mode 1); in mode 2 it expands to 32 output bytes)
    constexpr int RSTEP = STG32 ? 2 : 4;                 // token rows between a lane's consecutive granules (16 | 8 lanes per row)
    constexpr int OGB = RESG ? 32 : 16;                  // output bytes per staging granule
    const int rl = 16 * wiw + (STG32 ? (lane >> 4) : (lane >> 3)), cgl = STG32 ? (lane & 15) : (lane & 7);   // WG-relative row, granule
    const int jl = jw + rl;
    const size_t orow_off = (((size_t)iw * p.N + jl) * CZ) * OSZ + (size_t)OGB * cgl;
    uint8_t* orow = reinterpret_cast<uint8_t*>(p.out) + orow_off;
    const uint8_t* zrow = reinterpret_cast<const uint8_t*>(p.zres) + orow_off;          // mode 2: the fp32 z rows (same geometry as the fp32 output)
    const int nvalid = iw < p.N ? (p.N - jl + RSTEP - 1) / RSTEP : 0;                     // granules it < nvalid are inside the ragged edge
    constexpr size_t GSTRIDE = (size_t)RSTEP * CZ * OSZ;                                 // output bytes between a lane's consecutive granule rows
    auto release = [&](int qb) {                         // this warp's MMAs of block qb have retired (wgmma wait + syncwarp): free its two weight slots
      __syncwarp();
      if (!G::W_RESIDENT && lane == 0) {
        const uint32_t seqP = ((uint32_t)t_local * NB + (uint32_t)blk_of(qb)) * 2u;
        mbar_arrive(barW_empty + slot_of(seqP)); mbar_arrive(barW_empty + slot_of(seqP + 1u));
      }
    };
    // o = bf16(sigmoid(g) * p) in the accumulator layout -> this warp's staging slice, half h of the pair
    auto stage = [&](float (&accP)[16], float (&accG)[16], int h, int b0) {
      fence_regs(accP); fence_regs(accG);
      if (!STG32) {
        uint32_t fr[2][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {                    // n8 group j: columns 8 j + q2 (+1); rows gq -> fr[j>>1][2(j&1)], gq+8 -> fr[j>>1][2(j&1)+1]
          fr[j >> 1][2 * (j & 1)] = pack_bf16(math::gate(accG[4 * j + 0], accP[4 * j + 0]), math::gate(accG[4 * j + 1], accP[4 * j + 1]));
          fr[j >> 1][2 * (j & 1) + 1] = pack_bf16(math::gate(accG[4 * j + 2], accP[4 * j + 2]), math::gate(accG[4 * j + 3], accP[4 * j + 3]));
        }
        if (K3ST && !UPD && p.residual) {                // out = bf16(z + o): z pairs of block b0 + h in the C-fragment layout = A fragments of chunk b0/2, k-steps 2h, 2h+1
#pragma unroll
          for (int kb = 0; kb < 2; ++kb) {
            uint32_t rz[4];
            ldsm_x4(rz, sZ_u + (uint32_t)(b0 >> 1) * (uint32_t)CHB + swz128((uint32_t)(rho0 + lrow), (uint32_t)(((2 * h + kb) * 16 + ((mat & 2) ? 8 : 0)) * 2)));
            zdep ^= rz[0] ^ rz[3];
#pragma unroll
            for (int r = 0; r < 4; ++r) fr[kb][r] = math::residual_bf16x2(rz[r], fr[kb][r]);
          }
        }
        stsm_x4(stg_u + swz128((uint32_t)lrow, (uint32_t)((4 * h + (mat >> 1)) * 16)), fr[0][0], fr[0][1], fr[0][2], fr[0][3]);
        stsm_x4(stg_u + swz128((uint32_t)lrow, (uint32_t)((4 * h + 2 + (mat >> 1)) * 16)), fr[1][0], fr[1][1], fr[1][2], fr[1][3]);
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float oA0 = math::round_bf16(math::gate(accG[4 * j + 0], accP[4 * j + 0])), oA1 = math::round_bf16(math::gate(accG[4 * j + 1], accP[4 * j + 1]));
          const float oB0 = math::round_bf16(math::gate(accG[4 * j + 2], accP[4 * j + 2])), oB1 = math::round_bf16(math::gate(accG[4 * j + 3], accP[4 * j + 3]));
          const uint32_t gi = (uint32_t)(8 * h + 2 * j + (q2 >> 2)), cb = (uint32_t)((q2 & 3) * 4);
          sts64f(stg_u + (uint32_t)gq * 256u + ((gi ^ ((uint32_t)gq & 7u)) * 16u) + cb, oA0, oA1);
          sts64f(stg_u + (uint32_t)(gq + 8) * 256u + ((gi ^ ((uint32_t)(gq + 8) & 7u)) * 16u) + cb, oB0, oB1);
        }
      }
    };
    // vector pass over the staged pair (blocks b0, b0+1 = 64 contiguous channels): 16-byte granules staging -> (+ residual) -> global, ragged-predicated
    // staged pair (blocks b0, b0+1 = 64 contiguous channels) (+ residual from the z tile still resident in shared memory) -> global, 16-B granules
    auto vector_pass = [&](int b0) {
      __syncwarp();
#ifdef TMN_DEV_NOVEC
      if (lane == 99 && b0 == 12345) sts32(stg_u, 0u);
      return;
#endif
      uint32_t dep = 0;
#pragma unroll
      for (int it = 0; it < NGR; ++it) {
        const int row = (STG32 ? (lane >> 4) : (lane >> 3)) + RSTEP * it, cg = cgl;   // slice-relative row of granule it
        const uint4 ov = STG32 ? lds128(stg_u + (uint32_t)row * 256u + (((uint32_t)cg ^ ((uint32_t)row & 7u)) * 16u)) : lds128(stg_u + swz128((uint32_t)row, (uint32_t)(16 * cg)));
        if (RESG) {                                      // modes 2, 3: out (fp32, 8 values = 32 B) = z (fp32, global) + o (8 bf16 of the granule)
          if (it < nvalid) {
            const size_t boff = it * GSTRIDE + (size_t)(BN * b0 * OSZ);
            uint4 z0 = make_uint4(0u, 0u, 0u, 0u), z1 = z0;
            if (!UPD && p.residual) { z0 = ldg128(zrow + boff); z1 = ldg128(zrow + boff + 16); }
            uint4 w0, w1;
            w0.x = __float_as_uint(math::residual_f32(__uint_as_float(z0.x), bf16lo(ov.x))); w0.y = __float_as_uint(math::residual_f32(__uint_as_float(z0.y), bf16hi(ov.x)));
            w0.z = __float_as_uint(math::residual_f32(__uint_as_float(z0.z), bf16lo(ov.y))); w0.w = __float_as_uint(math::residual_f32(__uint_as_float(z0.w), bf16hi(ov.y)));
            w1.x = __float_as_uint(math::residual_f32(__uint_as_float(z1.x), bf16lo(ov.z))); w1.y = __float_as_uint(math::residual_f32(__uint_as_float(z1.y), bf16hi(ov.z)));
            w1.z = __float_as_uint(math::residual_f32(__uint_as_float(z1.z), bf16lo(ov.w))); w1.w = __float_as_uint(math::residual_f32(__uint_as_float(z1.w), bf16hi(ov.w)));
            stg128(orow + boff, w0); stg128(orow + boff + 16, w1);
          }
          continue;
        }
        uint4 zr = make_uint4(0u, 0u, 0u, 0u);
        if (!UPD && !NORES && p.residual) {                        // z chunk rows are [BMT tok][128 B] 128B-swizzled; bf16: chunk b0/2, granule cg; fp32: chunk b0 + cg/8
          const uint32_t trow = (uint32_t)(tok0 + rl + RSTEP * it);
          zr = ZF32 ? lds128(sZ_u + (uint32_t)(b0 + (cg >> 3)) * (uint32_t)CHB + swz128(trow, (uint32_t)(16 * (cg & 7))))
                    : lds128(sZ_u + (uint32_t)(b0 >> 1) * (uint32_t)CHB + swz128(trow, (uint32_t)(16 * cg)));
          dep ^= zr.x ^ zr.w;
        }
        if (it < nvalid) {
          uint4 w4;
          if (!STG32) {                                   // out = bf16(z + o) (o already bf16; z = 0 without residual): the framework's bf16 residual add
            w4.x = math::residual_bf16x2(zr.x, ov.x); w4.y = math::residual_bf16x2(zr.y, ov.y);
            w4.z = math::residual_bf16x2(zr.z, ov.z); w4.w = math::residual_bf16x2(zr.w, ov.w);
          } else {                                        // out = fp32(z) + o
            w4.x = __float_as_uint(math::residual_f32(__uint_as_float(zr.x), __uint_as_float(ov.x)));
            w4.y = __float_as_uint(math::residual_f32(__uint_as_float(zr.y), __uint_as_float(ov.y)));
            w4.z = __float_as_uint(math::residual_f32(__uint_as_float(zr.z), __uint_as_float(ov.z)));
            w4.w = __float_as_uint(math::residual_f32(__uint_as_float(zr.w), __uint_as_float(ov.w)));
          }
          stg128(orow + it * GSTRIDE + (size_t)(BN * b0 * OSZ), w4);
        }
      }
      if (!NORES && p.residual && !RESG) {                         // this warp's last use of the pair's z chunk(s): release for the next tile's refill
        dep = zero_dep(__reduce_or_sync(0xffffffffu, dep));
        fence_proxy_async();
        if (lane == 0) {
          if (ZF32) { mbar_arrive_dep(barZ_empty + b0, dep); mbar_arrive_dep(barZ_empty + b0 + 1, dep); }
          else mbar_arrive_dep(barZ_empty + (b0 >> 1), dep);
        }
      }
      __syncwarp();                                      // the slice is free for the next pair's staging
    };
    // TMA store of the staged pair: every lane fences its stmatrix writes towards the async proxy, lane 0 issues one bulk-tensor store of the slice
    // (box [64 ch][16 tok][1] at channel BN b0, this warp's token rows; the tensor bounds clip the ragged edge) and releases the pair's z chunk.
    auto store_pass = [&](int b0) {
      const uint32_t d = zero_dep(__reduce_or_sync(0xffffffffu, zdep));
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) {
        tma_store_3d(&p.tm_out, sOut + (4 * cw + wiw) * OB, BN * b0, jw + 16 * wiw, iw);
        tma_store_commit();
        if (p.residual) mbar_arrive_dep(barZ_empty + (b0 >> 1), d);
      }
      zdep = 0;
    };
    auto out_pass = [&](int b0) { if (K3ST) store_pass(b0); else vector_pass(b0); };
    auto slice_ready = [&]() { if (K3ST) { if (lane == 0) tma_store_wait_read<0>(); __syncwarp(); } };   // the previous store has read the slice
    if (G::NACC == 1) {
#pragma unroll 1
      for (int P = 0; P < NB / 2; ++P) {                 // block pairs in the producer's order; split-N: the pairs alternate between the warpgroups and a
        const int pq = SPLITN ? (P >> 1) * 2 : 2 * P;    // warpgroup passes over the other one's pair (waits + releases its slots without reading them)
        if (SPLITN && (P & 1) != cw) {
          if (!G::W_RESIDENT) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {                // gate, projection slots of blocks 2P, 2P+1
              const uint32_t sq = ((uint32_t)t_local * NB + (uint32_t)(2 * P + (u >> 1))) * 2u + (uint32_t)(u & 1);
              mbar_wait(barW_full + slot_of(sq), phase_of(sq));
            }
            __syncwarp();
            if (lane == 0) {
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                const uint32_t sq = ((uint32_t)t_local * NB + (uint32_t)(2 * P + (u >> 1))) * 2u + (uint32_t)(u & 1);
                mbar_arrive(barW_empty + slot_of(sq));
              }
            }
          }
          continue;
        }
        PF(9);
        const int b0c = blk_of(pq);
        issue(accP0, accG0, pq); wgmma_wait<0>(); PF(10); release(pq); slice_ready(); stage(accP0, accG0, 0, b0c); PF(11);   // 10: MMA wait, 11: stage+release
        issue(accP0, accG0, pq + 1); wgmma_wait<0>(); PF(10); release(pq + 1); stage(accP0, accG0, 1, b0c); PF(11);
        out_pass(b0c); PF(9);                          // 9: vector pass / TMA store
      }
      (void)accP1; (void)accG1;
    } else {                                             // one block's MMAs stay in flight during every staging / vector step
      issue(accP0, accG0, 0); PF(9);
#pragma unroll 1
      for (int pq = 0; pq < NBW; pq += 2) {
        const int b0c = blk_of(pq);
        issue(accP1, accG1, pq + 1);
        wgmma_wait<2>(); PF(10); release(pq); slice_ready(); stage(accP0, accG0, 0, b0c); PF(11);
        if (pq + 2 < NBW) { issue(accP0, accG0, pq + 2); wgmma_wait<2>(); } else { wgmma_wait<0>(); }
        PF(10); release(pq + 1); stage(accP1, accG1, 1, b0c); PF(11);
        out_pass(b0c); PF(9);
      }
    }
  }
  if (K3ST && lane == 0) tma_store_wait_all();          // every bulk store of this warp complete before the CTA's shared memory is released
#ifdef TMN_DEV_PROF
  if (lane == 0 && p.prof) { for (int i = 0; i < 12; ++i) atomicAdd(p.prof + i, pf[i]); atomicAdd(p.prof + 12, 1ull); }
#endif
}

}  // namespace sm90
}  // namespace tmn
