// SPDX-License-Identifier: Apache-2.0
// tmn_k3_wide.cuh — the wide-K epilogue member of the trimul_native kernel family (sm_90a): K3 for (c_z, c_hidden) pairs whose two MMA
// operands cannot both live in registers (c_z/16 + c_hidden/16 > 24 k-steps: every pair containing 384, and (256,256) without spills).
//
//   out[t, :] = (z[t, :] +) cvt( sigmoid(LN_in(z)[t] . Wog^T) * (LN_out(X[:, t]) . Wo^T) )      (same statement, parameter block and numerics as
//                                                                                                 tmn_kernels.cuh k3_body; only the operand staging differs)
// Operand staging ("gate RS / projection SS"):
//   * projection operand: the X sub-tile [C_H ch][64 tok] (TMA, token-contiguous) is pulled into registers ONCE per tile (ldmatrix.trans),
//     LayerNorm'd in registers exactly as in k3_body (same rounding point: LN output -> bf16), and WRITTEN BACK into the same shared-memory
//     tile token-major ([64 tok][C_H k], 128-byte rows per 64-channel chunk, 128B swizzle = the canonical K-major wgmma operand layout).  The
//     projection MMAs then read A from shared memory through a matrix descriptor (wgmma SS), so no X fragment is register-resident during the
//     block loop.  The X stage is released to the producer after the tile's last projection MMA retired.
//   * gate operand: LN_in(z) fragments in registers (RS) as in k3_body; z tile, residual-from-the-resident-z-tile, per-chunk release, output
//     staging, ragged edges, LayerNorm summation modes (1 own order | 2, 3 reference-library trees) and the fp32-resident-z form are unchanged.
//   * weight ring: slots alternate projection rows [BN n][C_H k] and gate rows [BN n][C_Z k]; slot sizes follow the operand (even slots hold
//     projection blocks, odd slots gate blocks — sequence parity == kind parity), so mixed widths do not pay max(C_Z, C_H) per slot.
// Register budget per consumer thread in the block loop: 4 C_Z/16 (gate fragments) + 32 NACC (accumulators) + addressing — 128..160 at C_Z 384.
#pragma once
#include "tmn_kernels.cuh"

// LayerNorm scheduling: k3_body serialises the per-k-step LN dependency chain (ln_fragment<KS, SERIAL=true>) because both fragment sets are
// live there; here only one set is live during each LayerNorm, so the interleaved schedule is affordable (same arithmetic, same bytes).
#ifndef TMN_K3W_LN_SERIAL       // 1 = one LayerNorm row chain at a time (the register-resident K3's instantiation: identical bytes); 0 = interleaved
#define TMN_K3W_LN_SERIAL 1
#endif
// Split-N LayerNorm de-duplication (bf16 z): warpgroup 0 LayerNorms the X sub-tile and writes it back, warpgroup 1 LayerNorms the z tile IN
// PLACE (token-major already; each warp its own rows) at the same time; both then take their gate fragments from the LN'd z tile with plain
// ldmatrix, and the residual is read from global memory (z is L2-resident: TMA fetched the tile moments earlier) instead of the — now
// normalised — resident tile.  Without it both warpgroups LayerNorm both operands of the same 64 tokens.  Same arithmetic, same bytes.
#ifndef TMN_K3W_DEDUP
#define TMN_K3W_DEDUP 1
#endif

namespace tmn {

// ============================================================================================================ config (tuning data)
template <int CZ_, int CH_, bool ZF32_, int BI_, int BJ_, int NSLOT_, int NACC_ = 1>
struct K3WCfg {
  static constexpr int CZ = CZ_, CH = CH_, BI = BI_, BJ = BJ_, NSLOT = NSLOT_, NACC = NACC_;
  static constexpr bool ZF32 = ZF32_;
  static constexpr int BMT = BI * BJ;                   // tokens per CTA tile step: 128 = the two consumer warpgroups split the tokens (64 each, all output
  static constexpr bool SPLITN = BMT == 64;             // blocks); 64 = both warpgroups serve the same 64 tokens and alternate output block pairs
  static constexpr int NSUB = BMT / 64;                 // 64-token X sub-tiles per tile
  static constexpr int ESZ = ZF32 ? 4 : 2;
  static constexpr int CHUNK_CH = 128 / ESZ;
  static constexpr int CHUNK_BYTES = BMT * 128;         // z chunk = [BMT tok][128 B], 128B-swizzled rows
  static constexpr int NKCZ = CZ / CHUNK_CH;            // z chunks per tile
  static constexpr int KSG = CZ / 16, KSP = CH / 16;    // k-steps: gate (K = C_Z), projection (K = C_H)
  static constexpr int BN = 32;                         // output block width (channels): m64n32 products
  static constexpr int NB = CZ / BN;                    // output blocks per token row
  static constexpr int NBW = SPLITN ? NB / 2 : NB;      // blocks served by one warpgroup per tile
  static constexpr int NKG = CZ / 64, NKP = CH / 64;    // 4 KB k-chunks ([32 n][64 k]) per gate | projection weight block
  static constexpr int SLOTG = CZ * 64, SLOTP = CH * 64;   // gate slot [BN n][C_Z k] bf16, projection slot [BN n][C_H k] bf16
  static constexpr int SLOT_PAIR = SLOTP + SLOTG;       // even slot s (projection) at (s/2) SLOT_PAIR, odd slot (gate) at (s/2) SLOT_PAIR + SLOTP
  static constexpr bool W_RESIDENT = NSLOT >= 2 * NB;   // the whole W_o | W_og fits the ring: loaded once per CTA, never released
  static constexpr int NSLOT_EFF = W_RESIDENT ? 2 * NB : NSLOT;
  static constexpr int W_CONSUMERS = 8;                 // every consumer warp arrives on every slot use (split-N pass-over included)
  static constexpr int NKCX = CH / 64;                  // X chunks per 64-token sub-tile ([64 ch][64 tok] bf16 = 8 KB raw; [64 tok][64 k] after LN write-back)
  static constexpr int XSUB_BYTES = CH * 128;           // one X sub-tile
  static constexpr int OB = 16 * 2 * BN * ESZ;          // per-warp output staging slice [16 tok][64 ch = a block pair] in z's dtype
  static constexpr int SMEM_X = NSUB * XSUB_BYTES;
  static constexpr int SMEM_Z = NKCZ * CHUNK_BYTES;
  static constexpr int SMEM_W = (NSLOT_EFF / 2) * SLOT_PAIR;
  static constexpr int SMEM_OUT = 8 * OB;
  static constexpr int SMEM_GB = (2 * CZ + 2 * CH) * 4;
  static constexpr int NBAR = 2 * NKCZ + 2 + 2 * NSLOT_EFF;
  static constexpr int SMEM_BAR = ((NBAR * 8 + 127) / 128) * 128;
  static constexpr int SMEM = SMEM_X + SMEM_Z + SMEM_W + SMEM_OUT + SMEM_GB + SMEM_BAR;
  static constexpr uint32_t slot_off(int s) { return (uint32_t)(s >> 1) * (uint32_t)SLOT_PAIR + ((s & 1) ? (uint32_t)SLOTP : 0u); }
  static_assert((BMT == 128 || BMT == 64) && BJ >= 64 && BJ % 64 == 0, "a consumer warpgroup serves 64 consecutive tokens of one pair row");
  static_assert(CZ % 64 == 0 && CH % 64 == 0, "C_Z, C_H multiples of 64");
  static_assert(NACC == 1 || NACC == 2, "one or two accumulator sets");
  static_assert(NACC == 1 || !SPLITN || W_RESIDENT || NSLOT >= 8, "split-N with two blocks of one warpgroup in flight needs a ring of two block pairs");
  static_assert(NB % 2 == 0, "output blocks are produced in pairs");
  static_assert(NSLOT % 2 == 0, "even ring: even slots hold projection blocks, odd slots gate blocks");
  static_assert(W_RESIDENT || NSLOT >= 4, "streamed ring: two blocks (gate + projection each) in flight");
  static_assert(SLOTP % 1024 == 0 && SLOTG % 1024 == 0 && XSUB_BYTES % 1024 == 0, "128B-swizzle atoms need 1024-byte aligned operand bases");
  static_assert(SMEM <= SMEM_LIMIT, "shared memory over the sm_90 limit");
};

// ============================================================================================================ PTX additions (sm_90a)
// D[64x32 f32] (+)= A[64x16 bf16 via smem desc, K-major] * B[16x32 bf16 via smem desc, K-major]: both descriptors as (lo, hi) halves plus IMMEDIATE
// byte offsets added to the start-address fields (one register pair per operand tile for a whole K chain).
template <int OFFA, int OFFB>
TMN_DEVI void wgmma_m64n32k16_ss_off(float (&d)[16], uint32_t a_lo, uint32_t a_hi, uint32_t b_lo, uint32_t b_hi, int scale_d) {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    ".reg .b32 la, lb;\n"
    ".reg .b64 da, db;\n"
    "setp.ne.b32 p, %20, 0;\n"
    "add.u32 la, %16, %21;\n"
    "add.u32 lb, %18, %22;\n"
    "mov.b64 da, {la, %17};\n"
    "mov.b64 db, {lb, %19};\n"
    "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},"
    " da, db, p, 1, 1, 0, 0;\n"
    "}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a_lo), "r"(a_hi), "r"(b_lo), "r"(b_hi), "r"(scale_d), "n"(OFFA >> 4), "n"(OFFB >> 4));
}
// full K chain of one m64n32 product: A = the LN'd token-major sub-tile ([64 tok][64 k] chunks of 8 KB), B = one weight block in ONE slot ([32 n][64 k] chunks of 4 KB)
template <int... Is>
TMN_DEVI void mma_chain32_ss_(float (&acc)[16], uint32_t alo, uint32_t ahi, uint32_t blo, uint32_t bhi, std::integer_sequence<int, Is...>) {
  (wgmma_m64n32k16_ss_off<((Is >> 2) * 8192 + (Is & 3) * 32), ((Is >> 2) * 4096 + (Is & 3) * 32)>(acc, alo, ahi, blo, bhi, Is > 0 ? 1 : 0), ...);
}
template <int KS>
TMN_DEVI void mma_chain32_ss(float (&acc)[16], uint32_t alo, uint32_t ahi, uint32_t blo, uint32_t bhi) {
  mma_chain32_ss_(acc, alo, ahi, blo, bhi, std::make_integer_sequence<int, KS>{});
}

// Inverse of load_frag_bf16: the warp's A fragments (16 rows x 16 KS k) -> a K-major bf16 tile of chunks [rows][64 ch] (128-B swizzled rows,
// CHUNK_BYTES apart); chunk c is written iff (c % wsplit) == wsel (several warps holding identical fragments share the stores).
template <int KS, int CHUNK_BYTES = 16384>
TMN_DEVI void store_frag_bf16(const uint32_t (&f)[KS][4], uint32_t base_u, int rho0, int lane, int wsplit = 1, int wsel = 0) {
  const int mat = lane >> 3, r8 = lane & 7;
  const int row = rho0 + r8 + ((mat & 1) ? 8 : 0);
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    const int kc = ks >> 2, kin = (ks & 3) * 16 + ((mat & 2) ? 8 : 0);
    if ((kc % wsplit) == wsel) stsm_x4(base_u + kc * CHUNK_BYTES + swz128(row, kin * 2), f[ks][0], f[ks][1], f[ks][2], f[ks][3]);
  }
}

namespace sm90 {

// ============================================================================================================ K3 (wide)
template <class G, int LNM>
TMN_DEVI void k3w_body(const K3Params& p) {
  constexpr int CZ = G::CZ, CH = G::CH, KSG = G::KSG, KSP = G::KSP, NKG = G::NKG, NKP = G::NKP, NB = G::NB, BN = G::BN;
  constexpr int NKCZ = G::NKCZ, BI = G::BI, BJ = G::BJ, OB = G::OB, CHB = G::CHUNK_BYTES, ESZ = G::ESZ, NSLOT_EFF = G::NSLOT_EFF;
  constexpr bool SPLITN = G::SPLITN, ZF32 = G::ZF32;
  constexpr bool DEDUP = SPLITN && !ZF32 && (TMN_K3W_DEDUP) != 0;   // split-N LayerNorm de-duplication (see top of file)
  static_assert(!(ZF32 && LNM >= 2), "the reference-order LayerNorm is defined on bf16 inputs");
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
  uint64_t* barX_empty = barX_full + 1;       // X released (8 consumer warps) after the tile's last projection MMA retired
  uint64_t* barZ_empty = barX_empty + 1;      // [NKCZ]
  uint64_t* barW_full = barZ_empty + NKCZ;    // [NSLOT_EFF]
  uint64_t* barW_empty = barW_full + NSLOT_EFF;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wg = __shfl_sync(0xffffffffu, tid >> 7, 0);   // warp-uniform by construction (setmaxnreg + role split per warp)
  if (tid == 0 && dyn_smem_size() < (uint32_t)G::SMEM) __trap();
  const int n_iter = (p.num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  for (int i = tid; i < CZ; i += NTHREADS) { sGin[i] = p.gamma_in[i]; sBin[i] = p.beta_in[i]; }
  for (int i = tid; i < CH; i += NTHREADS) { sGout[i] = p.gamma_out[i]; sBout[i] = p.beta_out[i]; }
  if (tid == 0) {
    for (int kc = 0; kc < NKCZ; ++kc) mbar_init(barZ_full + kc, 1);
    mbar_init(barX_full, 1); mbar_init(barX_empty, 8);
    for (int kc = 0; kc < NKCZ; ++kc) mbar_init(barZ_empty + kc, 8);
    for (int s = 0; s < NSLOT_EFF; ++s) { mbar_init(barW_full + s, 1); mbar_init(barW_empty + s, G::W_CONSUMERS); }
    fence_barrier_init();
    tma_prefetch_desc(&p.tm_z); tma_prefetch_desc(&p.tm_x); tma_prefetch_desc(&p.tm_wg); tma_prefetch_desc(&p.tm_wo);
  }
  __syncthreads();

  if (wg == 0) {
    // ================================================================== producers: warp 0 operand stage, warp 1 weight ring
    setmaxnreg_dec<40>();
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
          for (int kx = 0; kx < G::NKCX; ++kx) tma_load_3d(sX + h * G::XSUB_BYTES + kx * 8192, &p.tm_x, barX_full, jh, ih, kx * 64);
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
        const int s = (int)(seq % (uint32_t)NSLOT_EFF); const uint32_t u = seq / (uint32_t)NSLOT_EFF;
        if (u > 0) mbar_wait(barW_empty + s, (u - 1) & 1);
        const int b = (int)((seq % per_tile) >> 1), which = (int)(seq & 1u);
        uint8_t* dst = sW + G::slot_off(s);
        if (which == 0) {
          mbar_arrive_expect_tx(barW_full + s, G::SLOTP);
#pragma unroll
          for (int kc = 0; kc < NKP; ++kc) tma_load_2d(dst + kc * 4096, &p.tm_wo, barW_full + s, kc * 64, BN * b);
        } else {
          mbar_arrive_expect_tx(barW_full + s, G::SLOTG);
#pragma unroll
          for (int kc = 0; kc < NKG; ++kc) tma_load_2d(dst + kc * 4096, &p.tm_wg, barW_full + s, kc * 64, BN * b);
        }
      }
    }
    __syncwarp();
    return;
  }

  // ================================================================== consumers
  setmaxnreg_inc<232>();
#ifdef TMN_DEV_PROF
  unsigned long long pf[12] = {0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull};
  long long pt0 = clock64();
#define PFW(i) do { long long t1_ = clock64(); pf[i] += (unsigned long long)(t1_ - pt0); pt0 = t1_; } while (0)
#else
#define PFW(i) do { } while (0)
#endif
  const int cw = wg - 1, wiw = warp & 3, mat = lane >> 3, r8 = lane & 7;
  const int tok0 = SPLITN ? 0 : 64 * cw;                 // this WG's first token (tile-relative)
  const uint32_t sZ_u = smem_u32(sZ), sX_u = smem_u32(sX) + (uint32_t)((tok0 / 64) * G::XSUB_BYTES), sW_u = smem_u32(sW);
  const uint32_t stg_u = smem_u32(sOut) + (uint32_t)((4 * cw + wiw) * OB);   // this warp's staging slice [16 tok][64 ch]
  const int rho0 = tok0 + 16 * wiw;                      // tile-relative first row of this warp
  const int rowA = rho0 + (lane >> 2), rowB = rowA + 8;   // this thread's accumulator rows (tile-relative)
  const int gq = lane >> 2, q2 = 2 * (lane & 3);
  const int lrow = r8 + 8 * (mat & 1);
  // X sub-tile synchronisation: every warp that READS the raw sub-tile must be done before any warp overwrites it with LN'd rows, and every
  // writer must be done (+ async-proxy fence) before any MMA reads it.  Split-N: both warpgroups share sub-tile 0 (256 threads, barrier 3) and split
  // the write-back by chunk parity; otherwise each warpgroup owns its sub-tile (128 threads, barrier 1 + cw).
  const int xbar_id = SPLITN ? 3 : 1 + cw, xbar_n = SPLITN ? 256 : 128;
  const int wsplit = SPLITN ? 2 : 1, wsel = SPLITN ? cw : 0;
  const uint64_t descA = smem_desc(sX_u, 16, 1024, 1);   // the LN'd token-major sub-tile as the projection A operand (K-major, SW128, 8-row groups 1 KB apart)
  const uint32_t dA_lo = (uint32_t)descA, dA_hi = (uint32_t)(descA >> 32);
  for (int t_local = 0; t_local < n_iter; ++t_local) {
    const int tile = (int)blockIdx.x + t_local * (int)gridDim.x;
    const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
    const int iw = i0 + tok0 / BJ, jw = j0 + tok0 % BJ;               // this WG's 64 tokens: row iw, columns jw .. jw+63

    PFW(0);                                              // 0: loop head / previous tile tail
    uint32_t fz[KSG][4];
    if constexpr (DEDUP) {
      // ---- split-N, bf16 z: warpgroup 0 = projection operand (X sub-tile -> LN_out -> token-major write-back = the SS A operand),
      //      warpgroup 1 = gate operand (z tile -> LN_in -> written back in place); then both take LN'd z fragments from the tile.
      if (cw == 0) {
        mbar_wait(barX_full, t_local & 1);
        PFW(1);                                            // 1: X wait
        uint32_t fx[KSP][4];
        const int tokc = 16 * wiw + ((mat & 1) ? 8 : 0);
#pragma unroll
        for (int ks = 0; ks < KSP; ++ks) {
          const int krow = 16 * ks + r8 + ((mat & 2) ? 8 : 0);
          ldsm_x4_t(fx[ks], sX_u + swz128(krow, tokc * 2));
        }
        PFW(2);                                            // 2: X ldmatrix (+ write-back below)
#ifndef TMN_DEV_NOLN
        if (LNM == 2) ln_stock<1>(fx, sGout, sBout, lane, p.eps); else if (LNM == 3) ln_stock<2>(fx, sGout, sBout, lane, p.eps); else ln_fragment<KSP, (TMN_K3W_LN_SERIAL) != 0>(fx, sGout, sBout, lane, p.eps);
#endif
        PFW(3);                                            // 3: LN_out
        named_bar_sync(1, 128);                            // the four warps of this warpgroup hold all raw X fragments (their reads interleave)
        store_frag_bf16<KSP, 8192>(fx, sX_u, 16 * wiw, lane);
        fence_proxy_async();                               // generic-proxy writes -> visible to the async proxy (wgmma operand reads)
        PFW(2);
      } else {
        for (int kc = 0; kc < NKCZ; ++kc) mbar_wait(barZ_full + kc, t_local & 1);
        PFW(4);                                            // 4: z wait
        uint32_t fzw[KSG][4];                              // scoped: dies at the write-back, so no fragment set is live across the join below
        load_frag_bf16<KSG, CHB>(fzw, sZ_u, rho0, lane);
        PFW(5);                                            // 5: z load
#ifndef TMN_DEV_NOLN
        if (LNM >= 2) ln_stock<0>(fzw, sGin, sBin, lane, p.eps); else ln_fragment<KSG, (TMN_K3W_LN_SERIAL) != 0>(fzw, sGin, sBin, lane, p.eps);
#endif
        PFW(6);                                            // 6: LN_in
        __syncwarp();
        store_frag_bf16<KSG, CHB>(fzw, sZ_u, rho0, lane);  // in place: this warp's own 16 rows (row-disjoint from the other warps)
        PFW(5);
      }
      named_bar_sync(3, 256);                              // LN'd X sub-tile (warpgroup 0) and LN'd z tile (warpgroup 1) complete
      if (cw == 0) { for (int kc = 0; kc < NKCZ; ++kc) mbar_wait(barZ_full + kc, t_local & 1); }   // long complete; keeps the phase bookkeeping uniform
      load_frag_bf16<KSG, CHB>(fz, sZ_u, rho0, lane);      // both warpgroups: LN'd gate fragments straight from the tile (one definition point of fz)
      {   // every warp releases every z chunk now: the residual (if any) is read from global memory in this mode
        uint32_t dep = 0;
#pragma unroll
        for (int ks = 0; ks < KSG; ++ks) dep ^= fz[ks][0] ^ fz[ks][3];
        dep = zero_dep(__reduce_or_sync(0xffffffffu, dep));
        fence_proxy_async();
        if (lane == 0) {
#pragma unroll
          for (int kc = 0; kc < NKCZ; ++kc) mbar_arrive_dep(barZ_empty + kc, dep);
        }
      }
      PFW(5);
    } else {
      // ---- projection operand: raw X sub-tile [C_H ch][64 tok] -> registers (ldmatrix.trans) -> LN_out -> written back token-major (the SS A operand)
      mbar_wait(barX_full, t_local & 1);
      PFW(1);                                              // 1: X wait
      {
        uint32_t fx[KSP][4];
        const int tokc = 16 * wiw + ((mat & 1) ? 8 : 0);
  #pragma unroll
        for (int ks = 0; ks < KSP; ++ks) {
          const int krow = 16 * ks + r8 + ((mat & 2) ? 8 : 0);
          ldsm_x4_t(fx[ks], sX_u + swz128(krow, tokc * 2));
        }
        PFW(2);                                            // 2: X ldmatrix
  #ifndef TMN_DEV_NOLN
        if (LNM == 2) ln_stock<1>(fx, sGout, sBout, lane, p.eps); else if (LNM == 3) ln_stock<2>(fx, sGout, sBout, lane, p.eps); else ln_fragment<KSP, (TMN_K3W_LN_SERIAL) != 0>(fx, sGout, sBout, lane, p.eps);
  #endif
        PFW(3);                                            // 3: LN_out
        named_bar_sync(xbar_id, xbar_n);                   // every reader of the raw sub-tile holds its fragments
        store_frag_bf16<KSP, 8192>(fx, sX_u, 16 * wiw, lane, wsplit, wsel);
        fence_proxy_async();                               // generic-proxy writes -> visible to the async proxy (wgmma operand reads)
        named_bar_sync(xbar_id, xbar_n);                   // the LN'd sub-tile is complete
      }
      PFW(2);                                              // 2: (+ write-back)
      // ---- gate operand: z rows -> release the z stage -> LN_in
      for (int kc = 0; kc < NKCZ; ++kc) mbar_wait(barZ_full + kc, t_local & 1);
      PFW(4);                                              // 4: z wait
      if (ZF32) ln_rows_f32<KSG, CHB>(fz, sZ_u, rowA, rowB, sGin, sBin, lane, p.eps);
      else load_frag_bf16<KSG, CHB>(fz, sZ_u, rho0, lane);
      {   // release the z chunks this warp will not touch again: all of them unless the residual is read from the resident tile later
        uint32_t dep = 0;
  #pragma unroll
        for (int ks = 0; ks < KSG; ++ks) dep ^= fz[ks][0] ^ fz[ks][3];
        dep = zero_dep(__reduce_or_sync(0xffffffffu, dep));
        fence_proxy_async();
        if (lane == 0 && !p.residual) {
  #pragma unroll
          for (int kc = 0; kc < NKCZ; ++kc) mbar_arrive_dep(barZ_empty + kc, dep);
        }
      }
      PFW(5);                                              // 5: z load + release
  #ifndef TMN_DEV_NOLN
      if (!ZF32) { if (LNM >= 2) ln_stock<0>(fz, sGin, sBin, lane, p.eps); else ln_fragment<KSG, (TMN_K3W_LN_SERIAL) != 0>(fz, sGin, sBin, lane, p.eps); }
  #endif
      PFW(6);                                              // 6: LN_in
    }

    // ---- output blocks of BN = 32 channels, produced by the weight ring in pairs (2P, 2P+1).  Full tile (BMT 128): this warpgroup computes both
    //      blocks of every pair for its own 64 tokens and writes 128-byte row segments.  Split-N (BMT 64): both warpgroups serve the same 64 tokens;
    //      warpgroup cw computes block 2P + cw of every pair (64-byte row segments) and passes over the other block's two slot uses (waits for
    //      them, releases them unread) so that every consumer warp observes every slot phase in sequence order.
    constexpr int NBV = SPLITN ? 1 : 2;                  // blocks per vector pass (= per staging slice use)
    float accP0[16], accG0[16], accP1[16], accG1[16];
    auto slot_of = [&](uint32_t seq) -> int { return (int)(seq % (uint32_t)NSLOT_EFF); };
    auto phase_of = [&](uint32_t seq) -> uint32_t { return G::W_RESIDENT ? 0u : ((seq / (uint32_t)NSLOT_EFF) & 1u); };
    auto seq_of = [&](int b) -> uint32_t { return ((uint32_t)t_local * NB + (uint32_t)b) * 2u; };   // sequence number of block b's projection slot use (gate = +1)
    auto pass_over = [&](int b) {                        // the other warpgroup's block: observe both slot uses, release them unread
      if (G::W_RESIDENT) return;
      const uint32_t sq = seq_of(b);
      mbar_wait(barW_full + slot_of(sq), phase_of(sq)); mbar_wait(barW_full + slot_of(sq + 1u), phase_of(sq + 1u));
      __syncwarp();
      if (lane == 0) { mbar_arrive(barW_empty + slot_of(sq)); mbar_arrive(barW_empty + slot_of(sq + 1u)); }
    };
    auto issue = [&](float (&accP)[16], float (&accG)[16], int b) {
      const uint32_t seqP = seq_of(b), seqG = seqP + 1u;
      {
        const int s = slot_of(seqP); mbar_wait(barW_full + s, phase_of(seqP));
        PFW(7);                                          // 7: W wait
        const uint64_t d = smem_desc(sW_u + G::slot_off(s), 16, 1024, 1);
#pragma unroll
        for (int i = 0; i < 16; ++i) accP[i] = 0.f;
        fence_regs(accP);
        wgmma_fence();
        mma_chain32_ss<KSP>(accP, dA_lo, dA_hi, (uint32_t)d, (uint32_t)(d >> 32));
        wgmma_commit();
      }
      {
        PFW(8);                                          // 8: issue
        const int s = slot_of(seqG); mbar_wait(barW_full + s, phase_of(seqG));
        PFW(7);
        const uint64_t d = smem_desc(sW_u + G::slot_off(s), 16, 1024, 1);
#pragma unroll
        for (int i = 0; i < 16; ++i) accG[i] = 0.f;
        fence_regs(accG);
        wgmma_fence();
        mma_chain32<KSG>(accG, fz, (uint32_t)d, (uint32_t)(d >> 32));
        wgmma_commit();
        PFW(8);
      }
    };
    // issue block 2P + cw (split-N) with the pass-overs the sequence order demands: block 2P before ours (cw 1), block 2P+1 after ours (cw 0)
    auto issue_split = [&](float (&accP)[16], float (&accG)[16], int P) {
      if (cw == 1) pass_over(2 * P);
      issue(accP, accG, 2 * P + cw);
      if (cw == 0) pass_over(2 * P + 1);
    };
    auto release = [&](int b) {                          // this warp's MMAs of block b have retired: free its two weight slots
      __syncwarp();
      if (!G::W_RESIDENT && lane == 0) {
        const uint32_t seqP = seq_of(b);
        mbar_arrive(barW_empty + slot_of(seqP)); mbar_arrive(barW_empty + slot_of(seqP + 1u));
      }
    };
    // vector-pass geometry: the staging slice [16 tok][NBV blocks x 32 ch] in z's dtype is moved as 16-byte granules, GPR per token row; lane l owns
    // granule column l % GPR of rows l / GPR + RSTEP it
    constexpr int GPR = NBV * 2 * ESZ;                   // granules per row: bf16 4 | 8, fp32 8 | 16
    constexpr int RSTEP = 32 / GPR, NGR = 16 / RSTEP;    // rows between a lane's granules; granules per lane
    const int rl = 16 * wiw + lane / GPR, cgl = lane % GPR;
    const int jl = jw + rl;
    uint8_t* orow = reinterpret_cast<uint8_t*>(p.out) + (((size_t)iw * p.N + jl) * CZ) * ESZ + 16 * cgl;
    const uint8_t* zrow = reinterpret_cast<const uint8_t*>(p.zres) + (((size_t)iw * p.N + jl) * CZ) * ESZ + 16 * cgl;   // residual source (DEDUP: global)
    const int nvalid = iw < p.N ? (p.N - jl + RSTEP - 1) / RSTEP : 0;                     // granules it < nvalid are inside the ragged edge
    constexpr size_t GSTRIDE = (size_t)RSTEP * CZ * ESZ;
    // o = bf16(sigmoid(g) * p) in the accumulator layout -> this warp's staging slice, block position h (0 | 1) of the slice
    auto stage = [&](float (&accP)[16], float (&accG)[16], int h) {
      fence_regs(accP); fence_regs(accG);
      if (!ZF32) {
        uint32_t fr[2][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          fr[j >> 1][2 * (j & 1)] = pack_bf16(sigmoidf_(accG[4 * j + 0]) * accP[4 * j + 0], sigmoidf_(accG[4 * j + 1]) * accP[4 * j + 1]);
          fr[j >> 1][2 * (j & 1) + 1] = pack_bf16(sigmoidf_(accG[4 * j + 2]) * accP[4 * j + 2], sigmoidf_(accG[4 * j + 3]) * accP[4 * j + 3]);
        }
        stsm_x4(stg_u + swz128((uint32_t)lrow, (uint32_t)((4 * h + (mat >> 1)) * 16)), fr[0][0], fr[0][1], fr[0][2], fr[0][3]);
        stsm_x4(stg_u + swz128((uint32_t)lrow, (uint32_t)((4 * h + 2 + (mat >> 1)) * 16)), fr[1][0], fr[1][1], fr[1][2], fr[1][3]);
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float oA0 = bf16lo(pack_bf16(sigmoidf_(accG[4 * j + 0]) * accP[4 * j + 0], 0.f)), oA1 = bf16lo(pack_bf16(sigmoidf_(accG[4 * j + 1]) * accP[4 * j + 1], 0.f));
          const float oB0 = bf16lo(pack_bf16(sigmoidf_(accG[4 * j + 2]) * accP[4 * j + 2], 0.f)), oB1 = bf16lo(pack_bf16(sigmoidf_(accG[4 * j + 3]) * accP[4 * j + 3], 0.f));
          const uint32_t gi = (uint32_t)(8 * h + 2 * j + (q2 >> 2)), cb = (uint32_t)((q2 & 3) * 4);
          sts64f(stg_u + (uint32_t)gq * 256u + ((gi ^ ((uint32_t)gq & 7u)) * 16u) + cb, oA0, oA1);
          sts64f(stg_u + (uint32_t)(gq + 8) * 256u + ((gi ^ ((uint32_t)(gq + 8) & 7u)) * 16u) + cb, oB0, oB1);
        }
      }
    };
    // vector pass over the staged blocks b0 .. b0+NBV-1 (+ residual from the z tile still resident in shared memory) -> global, ragged-predicated
    auto vector_pass = [&](int b0) {
      __syncwarp();
      uint32_t dep = 0;
      const uint32_t cbase = (uint32_t)((b0 & 1) * (BN * ESZ / 16));   // granule offset of block b0 inside its 64-channel bf16 z chunk row (split-N odd blocks)
      uint4 zg[DEDUP ? NGR : 1];                         // DEDUP: the resident tile holds LN_in(z), so the residual rows come from global memory (L2-resident);
      if (DEDUP && p.residual) {                         // all granules issued up front so one memory latency covers the slice
#pragma unroll
        for (int it = 0; it < (DEDUP ? NGR : 1); ++it) zg[it] = it < nvalid ? ldg128(zrow + it * GSTRIDE + (size_t)(BN * b0 * ESZ)) : make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int it = 0; it < NGR; ++it) {
        const int row = lane / GPR + RSTEP * it, cg = cgl;               // slice-relative row, granule of this lane
        const uint4 ov = ZF32 ? lds128(stg_u + (uint32_t)row * 256u + (((uint32_t)cg ^ ((uint32_t)row & 7u)) * 16u)) : lds128(stg_u + swz128((uint32_t)row, (uint32_t)(16 * cg)));
        uint4 zr = make_uint4(0u, 0u, 0u, 0u);
        if (DEDUP && p.residual) {
          zr = zg[DEDUP ? it : 0];
        } else if (p.residual) {                         // z chunk rows are [BMT tok][128 B] 128B-swizzled; bf16: chunk b0/2 (64 ch); fp32: chunk b0 + cg/8 (32 ch each)
          const uint32_t trow = (uint32_t)(tok0 + rl + RSTEP * it);
          zr = ZF32 ? lds128(sZ_u + (uint32_t)(b0 + (cg >> 3)) * (uint32_t)CHB + swz128(trow, (uint32_t)(16 * (cg & 7))))
                    : lds128(sZ_u + (uint32_t)(b0 >> 1) * (uint32_t)CHB + swz128(trow, (uint32_t)(16 * (cbase + cg))));
          dep ^= zr.x ^ zr.w;
        }
        if (it < nvalid) {
          uint4 w4;
          if (!ZF32) {                                    // out = bf16(z + o) (o already bf16): the module's residual add in z's dtype
            w4.x = pack_bf16(bf16lo(zr.x) + bf16lo(ov.x), bf16hi(zr.x) + bf16hi(ov.x));
            w4.y = pack_bf16(bf16lo(zr.y) + bf16lo(ov.y), bf16hi(zr.y) + bf16hi(ov.y));
            w4.z = pack_bf16(bf16lo(zr.z) + bf16lo(ov.z), bf16hi(zr.z) + bf16hi(ov.z));
            w4.w = pack_bf16(bf16lo(zr.w) + bf16lo(ov.w), bf16hi(zr.w) + bf16hi(ov.w));
          } else {                                        // out = fp32(z) + o
            w4.x = __float_as_uint(__uint_as_float(zr.x) + __uint_as_float(ov.x));
            w4.y = __float_as_uint(__uint_as_float(zr.y) + __uint_as_float(ov.y));
            w4.z = __float_as_uint(__uint_as_float(zr.z) + __uint_as_float(ov.z));
            w4.w = __float_as_uint(__uint_as_float(zr.w) + __uint_as_float(ov.w));
          }
          stg128(orow + it * GSTRIDE + (size_t)(BN * b0 * ESZ), w4);
        }
      }
      if (p.residual && !DEDUP) {                        // this warp's last read of the pair's z chunk(s): release for the next tile's refill (every warp
        dep = zero_dep(__reduce_or_sync(0xffffffffu, dep));   // arrives on every chunk of the pair, read or not: the barrier counts all 8 consumer warps)
        fence_proxy_async();
        if (lane == 0) {
          const int pb = b0 & ~1;
          if (ZF32) { mbar_arrive_dep(barZ_empty + pb, dep); mbar_arrive_dep(barZ_empty + pb + 1, dep); }
          else mbar_arrive_dep(barZ_empty + (pb >> 1), dep);
        }
      }
      __syncwarp();                                      // the slice is free for the next staging
    };
    constexpr int NP = NB / 2;                           // block pairs per tile
    if (SPLITN) {
      if (G::NACC == 1) {
#pragma unroll 1
        for (int P = 0; P < NP; ++P) {
          PFW(9);
          issue_split(accP0, accG0, P); wgmma_wait<0>(); PFW(10); release(2 * P + cw); stage(accP0, accG0, 0); PFW(11);
          vector_pass(2 * P + cw); PFW(9);               // 9: vector pass, 10: MMA wait, 11: stage + release
        }
        (void)accP1; (void)accG1;
      } else {                                           // the next pair's block is in flight during this block's staging / vector pass
        issue_split(accP0, accG0, 0); PFW(9);
#pragma unroll 1
        for (int P = 0; P < NP; P += 2) {
          if (P + 1 < NP) { issue_split(accP1, accG1, P + 1); wgmma_wait<2>(); } else { wgmma_wait<0>(); }
          PFW(10); release(2 * P + cw); stage(accP0, accG0, 0); PFW(11); vector_pass(2 * P + cw); PFW(9);
          if (P + 1 < NP) {
            if (P + 2 < NP) { issue_split(accP0, accG0, P + 2); wgmma_wait<2>(); } else { wgmma_wait<0>(); }
            PFW(10); release(2 * (P + 1) + cw); stage(accP1, accG1, 0); PFW(11); vector_pass(2 * (P + 1) + cw); PFW(9);
          }
        }
      }
    } else if (G::NACC == 1) {
#pragma unroll 1
      for (int P = 0; P < NP; ++P) {
        PFW(9);
        issue(accP0, accG0, 2 * P); wgmma_wait<0>(); PFW(10); release(2 * P); stage(accP0, accG0, 0); PFW(11);
        issue(accP0, accG0, 2 * P + 1); wgmma_wait<0>(); PFW(10); release(2 * P + 1); stage(accP0, accG0, 1); PFW(11);
        vector_pass(2 * P); PFW(9);
      }
      (void)accP1; (void)accG1;
    } else {                                             // one block's MMAs stay in flight during every staging / vector step
      issue(accP0, accG0, 0); PFW(9);
#pragma unroll 1
      for (int pq = 0; pq < NB; pq += 2) {
        issue(accP1, accG1, pq + 1);
        wgmma_wait<2>(); PFW(10); release(pq); stage(accP0, accG0, 0); PFW(11);
        if (pq + 2 < NB) { issue(accP0, accG0, pq + 2); wgmma_wait<2>(); } else { wgmma_wait<0>(); }
        PFW(10); release(pq + 1); stage(accP1, accG1, 1); PFW(11);
        vector_pass(pq); PFW(9);
      }
    }
    // the tile's projection MMAs have all retired (every block waited): hand the X stage back to the producer
    __syncwarp();
    if (lane == 0) mbar_arrive(barX_empty);
  }
#ifdef TMN_DEV_PROF
  if (lane == 0 && p.prof) { for (int i = 0; i < 12; ++i) atomicAdd(p.prof + i, pf[i]); atomicAdd(p.prof + 12, 1ull); }
#endif
#undef PFW
}

}  // namespace sm90
}  // namespace tmn
