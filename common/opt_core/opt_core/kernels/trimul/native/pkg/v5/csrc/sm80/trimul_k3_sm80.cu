// build: archs=sm_80
// trimul_k3_sm80.cu -- K3 (epilogue) of the triangle multiplication, sm_80 member:
//   for every output token t = (i, j) of batch b:
//     xln = LN_out(X[b, :, i, j])  (X = the batched GEMM output, channel-major planes [B*c_hidden][Np][Np] bf16; fp32 statistics)
//     zln = LN_in(z[b, i, j, :])   (the output gate's input, recomputed from z: bf16 or fp32 z read natively)
//     o   = bf16( sigmoid(zln . Wog[n]) * (xln . Wo[n]) )      n in [0, c_z)
//     out[b, i, j, n] = o                (residual == 0)   |   o + z[b, i, j, n] summed in fp32   (residual == 1),  stored in z's dtype.
// One templated body; extern "C" instantiations = tuning-table points.  CTA = BM tokens j of one row i (grid: x = j tiles over N, y = i < N,
// z = batch), NW = BM/WM warps.  The X tile [c_hidden][BM] is cp.async-staged and read with ldmatrix.trans into register-resident A fragments
// (LN_out applied on the fragments); the zln rows go through the quad loader into a swizzled smem tile that REUSES the X tile's bytes and is
// re-read per k-step; weight chunks of BN output channels ([BN][c_hidden] rows of Wo | [BN][c_z] rows of Wog) stream through a cp.async
// ring; the epilogue stages each chunk's [tokens][BN] block per warp and writes BN-channel token segments with 16-byte stores.
#include "math_sm80.cuh"

using namespace tm80;

struct K3Params {
  const __nv_bfloat16* x;        // [B*D][Np][Np] bf16 (plane b*D + ch)
  const void* z;                 // [B, N, N, C] bf16 or fp32
  void* out;                     // [B, N, N, C]: bf16, or fp32 when residual is summed onto an fp32 z
  const __nv_bfloat16* wo;       // [C][D] bf16  (w_o, natural [out, in] layout)
  const __nv_bfloat16* wog;      // [C][C] bf16  (w_og)
  const float* g_out;            // [D] LN_out weight
  const float* b_out;            // [D] LN_out bias
  const float* g_in;             // [C] LN_in weight
  const float* b_in;             // [C] LN_in bias
  const __nv_bfloat16* zln;      // [B, N, N, C] bf16 LayerNorm_in output rows (PRELN instantiations read these instead of normalising z)
  long long z_bstride;           // N*N*C
  long long plane_elems;         // Np*Np
  int N, Np, B, D_rt;            // D_rt: c_hidden (checked against the instantiation by the host)
  int residual;
  int stash;                     // 1: the host sized dynamic smem for a raw bf16 z tile [BM][C] behind the staging area; the residual add reads it (no z re-read)                  // 0 | 1
  float eps;
};

// The residual's z chunk (residual == 1 without the raw-z tile): chunk c's BN channels of z (EB bytes per element) for one warp's WM rows are
// cp.async-ed into that warp's output staging (row pitch PITCH bytes) as 16-B pieces in the store phase's own (row, piece) order, rows >= N
// zero-filled; one commit group.  The epilogue forms the sum in place once the group has landed.
template <int C, int BN, int WM, int PITCH, int EB>
TM_DEVI void k3_issue_zchunk(const uint8_t* __restrict__ zrow, uint8_t* sO, int jw, int lane, int N, int c) {
  constexpr int LPR = BN * EB / 16;
  const uint8_t* const zc = zrow + (size_t)c * BN * EB;
#pragma unroll
  for (int sidx = lane; sidx < WM * LPR; sidx += 32) {
    const int r = sidx / LPR, part = sidx % LPR, j = jw + r;
    const bool ok = j < N;
    cp_async16(smem_u32(sO + r * PITCH + part * 16), zc + (size_t)(ok ? j : 0) * (C * EB) + part * 16, ok);
  }
  cp_commit();
}

// A bf16 z row's 16-B pieces (as row_fetch) loaded with the L2 evict-last priority: the rows this CTA normalises are re-read by the
// residual's z-chunk staging a few microseconds later, so they are kept resident in L2 in preference to the streamed planes.
template <int K>
TM_DEVI void row_fetch_keep(const __nv_bfloat16* __restrict__ row, bool valid, int q, RowRaw<K>& r) {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
#pragma unroll
  for (int kk = 0; kk < K / 32; ++kk) {
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (valid) asm volatile("ld.global.nc.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(row + 32 * kk + 8 * q), "l"(pol));
    r.w[4 * kk] = v.x; r.w[4 * kk + 1] = v.y; r.w[4 * kk + 2] = v.z; r.w[4 * kk + 3] = v.w;
  }
}

template <typename ZT, int C, int D, int BM, int BN, int WM, int STAGES, int LNX = 0, bool PRELN = false, int SIG = 0, int OPTS = -1>
TM_DEVI void k3_body(const K3Params& p) {
  constexpr int NW = BM / WM, NT = NW * 32, MI = WM / 16, DK = D / 16, CK = C / 16;
  constexpr int NCH = C / BN, NTL = BN / 8;
  constexpr int XROWB = BM * 2, ZROWB = C * 2, OROWB = D * 2, GROWB = C * 2;
  constexpr int A_BYTES = (D * BM > BM * C ? D * BM : BM * C) * 2;               // X tile [D][BM], then the zln tile [BM][C], same bytes
  constexpr int WCH_BYTES = BN * (D + C) * 2;                                     // Wo chunk [BN][D] then Wog chunk [BN][C]
  constexpr int OPITCH2 = (BN + 8) * 2;                                            // per-warp output staging row pitch (bf16 o; a bf16 residual is added in the store phase)
  constexpr int OPITCHF = (BN + (BN % 32 == 0 ? 8 : 4)) * 4;                       // per-warp fp32 staging row pitch of the fp32 residual sum (z chunk + o formed in place; 16-B rows)
  constexpr int WAITN = STAGES >= 2 ? STAGES - 2 : 0;
  using ZRaw = typename RowRawT<ZT, C>::type;
  // the warp's raw z rows are fetched into registers BEFORE the X-tile wait when they fit a 128-B-per-lane budget (MI*2 rows), so the two
  // global round trips of the prologue (X tile, z rows) overlap instead of queueing; else they are read after the X fragments as before.
  // OPTS (development override): bit 0 forces the early fetch on (1) / off (0); -1 = the budget rule.
  constexpr bool ZEARLY = !PRELN && (OPTS >= 0 ? (OPTS & 1) != 0 : (MI * 2 * sizeof(ZRaw) <= 128));
  static_assert(WM % 16 == 0 && BM % WM == 0 && BM >= 64 && C % 32 == 0 && D % 32 == 0 && BN % 16 == 0 && C % BN == 0 && STAGES >= 1, "tile parameters");
  static_assert(OPITCHF % 16 == 0 && OPITCHF >= OPITCH2, "fp32 staging rows");
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, q = lane & 3, gid = lane >> 2;
  const uint32_t sA_u = smem_u32(smem), sW_u = smem_u32(smem + A_BYTES);

  const int b = blockIdx.z, i = blockIdx.y, j0 = blockIdx.x * BM;
  const int N = p.N, Np = p.Np;
  const ZT* zrow = reinterpret_cast<const ZT*>(p.z) + (size_t)b * (size_t)p.z_bstride + ((size_t)i * N) * C;     // token (i, 0)
  const bool of32 = (sizeof(ZT) == 4) && p.residual;                             // output dtype: fp32 only for the residual sum onto an fp32 z
  const bool stash = (sizeof(ZT) == 2) && p.residual && p.stash;                  // raw bf16 z tile kept in smem for the residual (host-sized region)
  const bool zpre = (sizeof(ZT) == 2) && p.residual && !p.stash;                  // bf16 residual without the raw-z tile: z chunk staged per chunk
  // per-warp output staging: [WM tok][BN ch] bf16 (OPITCH2 rows) or, for the fp32 residual sum, [WM tok][BN ch] fp32 (OPITCHF rows).  A residual
  // launch without the raw-z tile cp.asyncs each chunk's z channels INTO this staging at the top of the chunk (in flight during the GEMMs) and
  // the epilogue forms the sum in place (bf16 z: in the bf16 staging, no extra bytes; fp32 z: in the fp32 staging, which the host sizes for
  // that launch only -- other launches keep their residency).
  uint8_t* sO_;
  if constexpr (sizeof(ZT) == 4) sO_ = smem + A_BYTES + STAGES * WCH_BYTES + warp * WM * (of32 ? OPITCHF : OPITCH2);
  else                           sO_ = smem + A_BYTES + STAGES * WCH_BYTES + warp * WM * OPITCH2;
  uint8_t* const sO = sO_;                                                        // this warp's output staging
  uint8_t* const sZ = smem + A_BYTES + STAGES * WCH_BYTES + NW * WM * OPITCH2;     // [BM][C] raw bf16 z rows (swizzled), present only when p.stash (bf16 z)
  const __nv_bfloat16* xb = p.x + ((size_t)b * D) * (size_t)p.plane_elems + (size_t)i * Np + j0;                  // channel 0, token j0

  auto load_w = [&](int chunk, int slot) {
    const uint32_t base = sW_u + (uint32_t)(slot * WCH_BYTES);
    cp_rows<D, NT>(base, p.wo + (size_t)chunk * BN * D, D, BN, tid);
    cp_rows<C, NT>(base + BN * OROWB, p.wog + (size_t)chunk * BN * C, C, BN, tid);
  };
  // ---- X tile [D ch][BM tok] (granules past Np zero-filled), then the weight prologue
  {
    constexpr int GPR = BM / 8;
    for (int s = tid; s < D * GPR; s += NT) {
      const int ch = s / GPR, g = s % GPR;
      const bool ok = j0 + 8 * g < Np;
      cp_async16(sA_u + swz<XROWB>(ch, g), xb + (size_t)ch * (size_t)p.plane_elems + (ok ? 8 * g : 0), ok);
    }
    cp_commit();
  }
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) { if (s < NCH) load_w(s, s); cp_commit(); }
  bool jok[MI][2];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int h = 0; h < 2; ++h) jok[mi][h] = j0 + warp * WM + mi * 16 + gid + 8 * h < N;
  auto fetch_row = [&](const ZT* src, bool ok, ZRaw& rr) {                        // a bf16 row that the residual re-reads per chunk is kept in L2
    if constexpr (sizeof(ZT) == 2) { if (zpre) row_fetch_keep<C>(src, ok, q, rr); else row_fetch<C>(src, ok, q, rr); }
    else row_fetch<C>(src, ok, q, rr);
  };
  ZRaw zraw[ZEARLY ? MI : 1][2];
  if constexpr (ZEARLY) {
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int j = j0 + warp * WM + mi * 16 + gid + 8 * h;
        fetch_row(zrow + (size_t)(jok[mi][h] ? j : 0) * C, jok[mi][h], zraw[mi][h]);
      }
  }
  if (STAGES >= 2) cp_wait<STAGES - 1>(); else cp_wait<0>();                     // the X tile group has landed (weight groups may still fly)
  __syncthreads();

  // ---- xln: A fragments (rows = tokens, k = channels) via ldmatrix.trans, LN_out on the fragments, register-resident
  uint32_t fx[MI][DK][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
    for (int ks = 0; ks < DK; ++ks) load_a16_trans<XROWB>(fx[mi][ks], sA_u, warp * WM + mi * 16, ks, lane);
    ln_frags<D, LNX>(fx[mi], p.g_out, p.b_out, q, p.eps, lane);
  }
  __syncthreads();                                                              // every warp holds its X fragments: the tile becomes the zln tile

  // ---- zln rows -> swizzled [BM][C] smem tile: normalised here from z (warp-private rows), or copied from p.zln (PRELN: the prologue kernel wrote them)
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = warp * WM + mi * 16 + gid + 8 * h, j = j0 + r;
      const bool ok = jok[mi][h];
      if constexpr (!PRELN) {
        uint32_t row[C / 8];
        ZRaw rr;
        if constexpr (ZEARLY) rr = zraw[mi][h];
        else fetch_row(zrow + (size_t)(ok ? j : 0) * C, ok, rr);
        if constexpr (sizeof(ZT) == 2) { if (stash) store_row_smem<C>(rr.w, sZ, r, q); }   // raw z row kept for the residual add
        row_ln<C, (LNX != 0)>(rr, p.g_in, p.b_in, p.eps, q, row);
        store_row_smem<C>(row, smem, r, q);
      }
    }
  }
  if constexpr (PRELN) {
    constexpr int GPRZ = C / 8;
    const __nv_bfloat16* zl = p.zln + (size_t)b * (size_t)p.z_bstride + ((size_t)i * N + j0) * C;                // token (i, j0) row
    for (int s = tid; s < BM * GPRZ; s += NT) {
      const int r = s / GPRZ, g = s % GPRZ;
      const bool ok = j0 + r < N;
      cp_async16(sA_u + swz<ZROWB>(r, g), zl + (ok ? (size_t)r * C + 8 * g : 0), ok);
    }
    cp_commit(); cp_wait<0>(); __syncthreads();
  } else {
    __syncwarp();
  }

  // ---- output-channel chunks
#pragma unroll 1
  for (int c = 0; c < NCH; ++c) {
    cp_wait<WAITN>();
    __syncthreads();
    if (STAGES >= 2) {                                                              // ring: the z group is OLDER than the next weight chunk's group (wait_group 1 completes it)
      if constexpr (sizeof(ZT) == 4) { if (of32) k3_issue_zchunk<C, BN, WM, OPITCHF, 4>(reinterpret_cast<const uint8_t*>(zrow), sO, j0 + warp * WM, lane, N, c); }
      else                           { if (zpre) k3_issue_zchunk<C, BN, WM, OPITCH2, 2>(reinterpret_cast<const uint8_t*>(zrow), sO, j0 + warp * WM, lane, N, c); }
    }
    { const int cn = c + STAGES - 1; if (cn < NCH) load_w(cn, cn % STAGES); cp_commit(); }
    if (STAGES == 1) {                                                              // no ring: this chunk's weights first (needed now), then the z chunk flies during the GEMMs
      cp_wait<0>(); __syncthreads();
      if constexpr (sizeof(ZT) == 4) { if (of32) k3_issue_zchunk<C, BN, WM, OPITCHF, 4>(reinterpret_cast<const uint8_t*>(zrow), sO, j0 + warp * WM, lane, N, c); }
      else                           { if (zpre) k3_issue_zchunk<C, BN, WM, OPITCH2, 2>(reinterpret_cast<const uint8_t*>(zrow), sO, j0 + warp * WM, lane, N, c); }
    }
    const uint32_t wb = sW_u + (uint32_t)((c % STAGES) * WCH_BYTES);
    float accp[MI][NTL][4], accg[MI][NTL][4];
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int nt = 0; nt < NTL; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) { accp[mi][nt][e] = 0.f; accg[mi][nt][e] = 0.f; }
    // projection: xln[BM x D] . Wo_chunk^T
#pragma unroll
    for (int ks = 0; ks < DK; ++ks) {
#pragma unroll
      for (int np = 0; np < NTL / 2; ++np) {
        uint32_t bw[4];
        load_b16<OROWB>(bw, wb, np * 16, ks, lane);
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) { mma16816(accp[mi][2 * np], fx[mi][ks], bw[0], bw[1]); mma16816(accp[mi][2 * np + 1], fx[mi][ks], bw[2], bw[3]); }
      }
    }
    // gate: zln[BM x C] . Wog_chunk^T  (A re-read per k-step from the warp's own smem rows)
#pragma unroll
    for (int ks = 0; ks < CK; ++ks) {
      uint32_t fa[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) load_a16<ZROWB>(fa[mi], sA_u, warp * WM + mi * 16, ks, lane);
#pragma unroll
      for (int np = 0; np < NTL / 2; ++np) {
        uint32_t bw[4];
        load_b16<GROWB>(bw, wb + BN * OROWB, np * 16, ks, lane);
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) { mma16816(accg[mi][2 * np], fa[mi], bw[0], bw[1]); mma16816(accg[mi][2 * np + 1], fa[mi], bw[2], bw[3]); }
      }
    }
    // ---- epilogue: o = bf16(sigmoid(g) * p) -> per-warp staging [WM][BN] bf16 -> coalesced store phase: (+ a bf16 z, read in 16-B pieces there) ->
    //      out[b, i, j, chunk] as bf16 (the update, or the bf16 sum onto a bf16 z), 16-B stores.  fp32 residual: the z chunk has landed in the
    //      fp32 staging (cp.async above); the sum bf16(o) + z is formed there in fp32 and the store phase writes it as 16-B pieces.
    if (of32 || zpre) { if (STAGES >= 2) cp_wait<1>(); else cp_wait<0>(); __syncwarp(); }   // every lane's z pieces complete and visible to the warp
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = mi * 16 + gid + 8 * h;
#pragma unroll
        for (int nt = 0; nt < NTL; ++nt) {
          const int nl = nt * 8 + 2 * q;
          const float o0 = SIG == 0 ? math::gate(accg[mi][nt][2 * h], accp[mi][nt][2 * h]) : __fmul_rn(sigmoid_t<SIG>(accg[mi][nt][2 * h]), accp[mi][nt][2 * h]);
          const float o1 = SIG == 0 ? math::gate(accg[mi][nt][2 * h + 1], accp[mi][nt][2 * h + 1]) : __fmul_rn(sigmoid_t<SIG>(accg[mi][nt][2 * h + 1]), accp[mi][nt][2 * h + 1]);
          if constexpr (sizeof(ZT) == 4) {
            if (of32) {                                                             // fp32 sum = z + bf16(o), in the fp32 staging
              float2* const zo = reinterpret_cast<float2*>(sO + r * OPITCHF + nl * 4);
              const float2 zz = *zo;
              *zo = make_float2(math::residual_f32(zz.x, round_bf16(o0)), math::residual_f32(zz.y, round_bf16(o1)));
            } else {
              *reinterpret_cast<uint32_t*>(sO + r * OPITCH2 + nl * 2) = pack_bf16(o0, o1);
            }
          } else {
            uint32_t* const so = reinterpret_cast<uint32_t*>(sO + r * OPITCH2 + nl * 2);
            if (zpre) *so = math::residual_bf16x2(*so, pack_bf16(o0, o1));           // bf16 sum = bf16(z + bf16(o)), over the staged z pair
            else      *so = pack_bf16(o0, o1);
          }
        }
      }
    }
    __syncwarp();
    if (!of32) {                                                                    // bf16 out: 8 channels per 16-B lane
      constexpr int LPR = BN / 8;
      __nv_bfloat16* const ob = reinterpret_cast<__nv_bfloat16*>(p.out) + (size_t)b * (size_t)p.z_bstride + ((size_t)i * N) * C + (size_t)c * BN;
#pragma unroll
      for (int sidx = lane; sidx < WM * LPR; sidx += 32) {
        const int r = sidx / LPR, part = sidx % LPR, j = j0 + warp * WM + r;
        if (j < N) {
          uint4 o = *reinterpret_cast<const uint4*>(sO + r * OPITCH2 + part * 16);
          if (stash) {                                                              // + z from the raw-z tile (bf16 z; the staged residual paths carry the sum already)
            const uint4 zz = *reinterpret_cast<const uint4*>(sZ + swz<ZROWB>(warp * WM + r, (c * BN) / 8 + part));
            o.x = math::residual_bf16x2(zz.x, o.x); o.y = math::residual_bf16x2(zz.y, o.y);      // the framework's bf16 add of z and the rounded update
            o.z = math::residual_bf16x2(zz.z, o.z); o.w = math::residual_bf16x2(zz.w, o.w);
          }
          st_global_16<true>(ob + (size_t)j * C + part * 8, o);                 // streaming store: the output is not re-read by this op
        }
      }
    } else if constexpr (sizeof(ZT) == 4) {                                          // fp32 out = the staged fp32 sums: 4 channels per 16-B lane
      constexpr int LPR = BN / 4;
      float* const ob = reinterpret_cast<float*>(p.out) + (size_t)b * (size_t)p.z_bstride + ((size_t)i * N) * C + (size_t)c * BN;
#pragma unroll
      for (int sidx = lane; sidx < WM * LPR; sidx += 32) {
        const int r = sidx / LPR, part = sidx % LPR, j = j0 + warp * WM + r;
        if (j < N) st_global_16<true>(ob + (size_t)j * C + part * 4, *reinterpret_cast<const float4*>(sO + r * OPITCHF + part * 16));
      }
    }
    __syncwarp();                                                                   // the staging is re-filled (next chunk's z pieces / o) only after every lane's stores read it
  }
  cp_wait<0>();
}

#define K3_SM80(NAME, ZT, C_, D_, BM_, BN_, WM_, ST_, MB_)                                                             \
  extern "C" __global__ void __launch_bounds__((BM_ / WM_) * 32, MB_) NAME(const __grid_constant__ K3Params p) {     \
    k3_body<ZT, C_, D_, BM_, BN_, WM_, ST_>(p);                                                                       \
  }
// bit-exact variants (stock-order LayerNorms; c_z = c_hidden = 256, bf16 pair tensor): k3x1_ = plane row length N % 4 == 0, k3x2_ = N % 4 != 0
#define K3X_SM80(NAME, LNX_, C_, D_, BM_, BN_, WM_, ST_, MB_)                                                          \
  extern "C" __global__ void __launch_bounds__((BM_ / WM_) * 32, MB_) NAME(const __grid_constant__ K3Params p) {     \
    k3_body<__nv_bfloat16, C_, D_, BM_, BN_, WM_, ST_, LNX_>(p);                                                      \
  }
// tanh-approx gate rows (outside the acceptance class; measurement only): prefix k3t_
#define K3T_SM80(NAME, ZT, C_, D_, BM_, BN_, WM_, ST_, MB_)                                                            \
  extern "C" __global__ void __launch_bounds__((BM_ / WM_) * 32, MB_) NAME(const __grid_constant__ K3Params p) {     \
    k3_body<ZT, C_, D_, BM_, BN_, WM_, ST_, 0, false, 1>(p);                                                          \
  }
// variants reading the prologue kernel's LayerNorm_in output rows (p.zln, bf16) instead of normalising z again; z (ZT) is read only for the
// residual sum: prefix k3p_
#define K3P_SM80(NAME, ZT, C_, D_, BM_, BN_, WM_, ST_, MB_)                                                            \
  extern "C" __global__ void __launch_bounds__((BM_ / WM_) * 32, MB_) NAME(const __grid_constant__ K3Params p) {     \
    k3_body<ZT, C_, D_, BM_, BN_, WM_, ST_, 0, true>(p);                                                              \
  }
// name grammar: k3_<bf16|f32>_c<C>_h<c_hidden>_bm<BM>_bn<BN>_wm<WM>_s<STAGES>_mb<minblocks>
// dynamic smem = max(D*BM, BM*C)*2 + STAGES*BN*(D+C)*2 + BM*(BN+8)*2 bytes (+ BM*C*2 for the optional raw-z tile of a bf16 residual);
// a residual launch onto an fp32 z sizes the staging as BM*(BN + (BN%32 ? 4 : 8))*4 instead (the host computes the same formulas)
K3_SM80(k3_bf16_c128_h128_bm128_bn16_wm32_s2_mb3, __nv_bfloat16, 128, 128, 128, 16, 32, 2, 3)
K3_SM80(k3_bf16_c128_h128_bm128_bn32_wm32_s2_mb2, __nv_bfloat16, 128, 128, 128, 32, 32, 2, 2)
K3_SM80(k3_bf16_c128_h128_bm64_bn16_wm16_s2_mb4,  __nv_bfloat16, 128, 128, 64, 16, 16, 2, 4)
K3_SM80(k3_bf16_c128_h128_bm64_bn32_wm32_s2_mb3,  __nv_bfloat16, 128, 128, 64, 32, 32, 2, 3)
K3_SM80(k3_bf16_c128_h128_bm128_bn32_wm32_s1_mb2, __nv_bfloat16, 128, 128, 128, 32, 32, 1, 2)
K3_SM80(k3_bf16_c128_h128_bm64_bn32_wm32_s1_mb4,  __nv_bfloat16, 128, 128, 64, 32, 32, 1, 4)
K3_SM80(k3_f32_c128_h128_bm128_bn16_wm32_s2_mb2,  float,         128, 128, 128, 16, 32, 2, 2)
K3_SM80(k3_f32_c128_h128_bm64_bn16_wm16_s2_mb4,   float,         128, 128, 64, 16, 16, 2, 4)
K3_SM80(k3_f32_c128_h128_bm128_bn32_wm32_s1_mb2,  float,         128, 128, 128, 32, 32, 1, 2)
K3_SM80(k3_f32_c128_h128_bm64_bn32_wm32_s1_mb4,   float,         128, 128, 64, 32, 32, 1, 4)
K3_SM80(k3_f32_c128_h128_bm64_bn16_wm32_s2_mb4,   float,         128, 128, 64, 16, 32, 2, 4)
K3_SM80(k3_bf16_c256_h256_bm64_bn32_wm32_s1_mb2,  __nv_bfloat16, 256, 256, 64, 32, 32, 1, 2)
K3_SM80(k3_bf16_c256_h256_bm64_bn16_wm32_s2_mb2,  __nv_bfloat16, 256, 256, 64, 16, 32, 2, 2)
K3_SM80(k3_bf16_c256_h256_bm64_bn16_wm32_s1_mb4,  __nv_bfloat16, 256, 256, 64, 16, 32, 1, 4)
K3_SM80(k3_f32_c256_h256_bm64_bn16_wm16_s2_mb2,   float,         256, 256, 64, 16, 16, 2, 2)
K3_SM80(k3_f32_c256_h256_bm64_bn16_wm32_s1_mb4,   float,         256, 256, 64, 16, 32, 1, 4)
K3_SM80(k3_f32_c256_h256_bm64_bn32_wm32_s1_mb2,   float,         256, 256, 64, 32, 32, 1, 2)
K3_SM80(k3_bf16_c64_h64_bm128_bn16_wm32_s2_mb4,   __nv_bfloat16, 64, 64, 128, 16, 32, 2, 4)
K3_SM80(k3_bf16_c64_h64_bm64_bn64_wm16_s1_mb4,   __nv_bfloat16, 64, 64, 64, 64, 16, 1, 4)
K3_SM80(k3_f32_c64_h64_bm128_bn16_wm32_s2_mb4,    float,         64, 64, 128, 16, 32, 2, 4)
K3_SM80(k3_bf16_c64_h64_bm128_bn32_wm32_s2_mb4,   __nv_bfloat16, 64, 64, 128, 32, 32, 2, 4)
K3_SM80(k3_bf16_c64_h64_bm64_bn32_wm32_s2_mb6,    __nv_bfloat16, 64, 64, 64, 32, 32, 2, 6)
K3_SM80(k3_bf16_c64_h128_bm128_bn16_wm32_s2_mb4,  __nv_bfloat16, 64, 128, 128, 16, 32, 2, 4)
K3_SM80(k3_f32_c64_h128_bm128_bn16_wm32_s2_mb4,   float,         64, 128, 128, 16, 32, 2, 4)
K3_SM80(k3_bf16_c64_h128_bm128_bn32_wm32_s2_mb3,  __nv_bfloat16, 64, 128, 128, 32, 32, 2, 3)
K3_SM80(k3_bf16_c64_h128_bm64_bn32_wm32_s2_mb4,   __nv_bfloat16, 64, 128, 64, 32, 32, 2, 4)
K3_SM80(k3_bf16_c64_h128_bm64_bn64_wm16_s1_mb3,   __nv_bfloat16, 64, 128, 64, 64, 16, 1, 3)   // BN = c_z: one full-row output block, W_o/W_og resident
K3_SM80(k3_bf16_c384_h384_bm64_bn16_wm16_s1_mb1,  __nv_bfloat16, 384, 384, 64, 16, 16, 1, 1)
K3_SM80(k3_f32_c384_h384_bm64_bn16_wm16_s1_mb1,   float,         384, 384, 64, 16, 16, 1, 1)
// bit-exact rows (k3x1_: plane row length N % 4 == 0, k3x2_: N % 4 != 0; one pair per diagonal width)
K3X_SM80(k3x1_bf16_c256_h256_bm64_bn16_wm32_s1_mb4, 1, 256, 256, 64, 16, 32, 1, 4)
K3X_SM80(k3x2_bf16_c256_h256_bm64_bn16_wm32_s1_mb4, 2, 256, 256, 64, 16, 32, 1, 4)
K3X_SM80(k3x1_bf16_c64_h64_bm64_bn64_wm16_s1_mb4, 1, 64, 64, 64, 64, 16, 1, 4)
K3X_SM80(k3x2_bf16_c64_h64_bm64_bn64_wm16_s1_mb4, 2, 64, 64, 64, 64, 16, 1, 4)
K3X_SM80(k3x1_bf16_c128_h128_bm128_bn32_wm32_s2_mb2, 1, 128, 128, 128, 32, 32, 2, 2)
K3X_SM80(k3x2_bf16_c128_h128_bm128_bn32_wm32_s2_mb2, 2, 128, 128, 128, 32, 32, 2, 2)
K3X_SM80(k3x1_bf16_c384_h384_bm64_bn32_wm16_s1_mb1, 1, 384, 384, 64, 32, 16, 1, 1)
K3X_SM80(k3x2_bf16_c384_h384_bm64_bn32_wm16_s1_mb1, 2, 384, 384, 64, 32, 16, 1, 1)
K3X_SM80(k3x1_bf16_c256_h256_bm64_bn32_wm16_s1_mb2, 1, 256, 256, 64, 32, 16, 1, 2)
K3X_SM80(k3x2_bf16_c256_h256_bm64_bn32_wm16_s1_mb2, 2, 256, 256, 64, 32, 16, 1, 2)
// the remaining (c_z, c_hidden) pairs of {64, 128, 256, 384}^2
K3_SM80(k3_bf16_c64_h256_bm64_bn16_wm32_s1_mb4, __nv_bfloat16, 64, 256, 64, 16, 32, 1, 4)
K3_SM80(k3_f32_c64_h256_bm64_bn16_wm32_s1_mb4, float, 64, 256, 64, 16, 32, 1, 4)
K3_SM80(k3_bf16_c64_h384_bm64_bn16_wm16_s1_mb2, __nv_bfloat16, 64, 384, 64, 16, 16, 1, 2)
K3_SM80(k3_f32_c64_h384_bm64_bn16_wm16_s1_mb2, float, 64, 384, 64, 16, 16, 1, 2)
K3_SM80(k3_bf16_c128_h64_bm128_bn32_wm32_s2_mb2, __nv_bfloat16, 128, 64, 128, 32, 32, 2, 2)
K3_SM80(k3_f32_c128_h64_bm64_bn32_wm32_s1_mb4, float, 128, 64, 64, 32, 32, 1, 4)
K3_SM80(k3_bf16_c128_h256_bm64_bn16_wm32_s1_mb4, __nv_bfloat16, 128, 256, 64, 16, 32, 1, 4)
K3_SM80(k3_f32_c128_h256_bm64_bn16_wm32_s1_mb4, float, 128, 256, 64, 16, 32, 1, 4)
K3_SM80(k3_bf16_c128_h384_bm64_bn16_wm16_s1_mb2, __nv_bfloat16, 128, 384, 64, 16, 16, 1, 2)
K3_SM80(k3_f32_c128_h384_bm64_bn16_wm16_s1_mb2, float, 128, 384, 64, 16, 16, 1, 2)
K3_SM80(k3_bf16_c256_h64_bm64_bn16_wm32_s1_mb4, __nv_bfloat16, 256, 64, 64, 16, 32, 1, 4)
K3_SM80(k3_f32_c256_h64_bm64_bn16_wm32_s1_mb4, float, 256, 64, 64, 16, 32, 1, 4)
K3_SM80(k3_bf16_c256_h128_bm64_bn16_wm32_s1_mb4, __nv_bfloat16, 256, 128, 64, 16, 32, 1, 4)
K3_SM80(k3_f32_c256_h128_bm64_bn16_wm32_s1_mb4, float, 256, 128, 64, 16, 32, 1, 4)
K3_SM80(k3_bf16_c256_h384_bm64_bn16_wm16_s1_mb2, __nv_bfloat16, 256, 384, 64, 16, 16, 1, 2)
K3_SM80(k3_f32_c256_h384_bm64_bn16_wm16_s1_mb2, float, 256, 384, 64, 16, 16, 1, 2)
K3_SM80(k3_bf16_c384_h64_bm64_bn16_wm16_s1_mb2, __nv_bfloat16, 384, 64, 64, 16, 16, 1, 2)
K3_SM80(k3_f32_c384_h64_bm64_bn16_wm16_s1_mb2, float, 384, 64, 64, 16, 16, 1, 2)
K3_SM80(k3_bf16_c384_h128_bm64_bn16_wm16_s1_mb2, __nv_bfloat16, 384, 128, 64, 16, 16, 1, 2)
K3_SM80(k3_f32_c384_h128_bm64_bn16_wm16_s1_mb2, float, 384, 128, 64, 16, 16, 1, 2)
K3_SM80(k3_bf16_c384_h256_bm64_bn16_wm16_s1_mb2, __nv_bfloat16, 384, 256, 64, 16, 16, 1, 2)
K3_SM80(k3_f32_c384_h256_bm64_bn16_wm16_s1_mb2, float, 384, 256, 64, 16, 16, 1, 2)
// pre-normalised-rows-in rows (fp32 pair tensor; same tiles as the fp32 table rows)
K3P_SM80(k3p_f32_c128_h128_bm128_bn16_wm32_s2_mb2, float, 128, 128, 128, 16, 32, 2, 2)
K3P_SM80(k3p_f32_c128_h128_bm64_bn16_wm16_s2_mb4, float, 128, 128, 64, 16, 16, 2, 4)
K3P_SM80(k3p_f32_c128_h128_bm128_bn32_wm32_s1_mb2, float, 128, 128, 128, 32, 32, 1, 2)
K3P_SM80(k3p_f32_c128_h128_bm64_bn32_wm32_s1_mb4, float, 128, 128, 64, 32, 32, 1, 4)
K3P_SM80(k3p_f32_c128_h128_bm64_bn16_wm32_s2_mb4, float, 128, 128, 64, 16, 32, 2, 4)
K3P_SM80(k3p_f32_c256_h256_bm64_bn16_wm16_s2_mb2, float, 256, 256, 64, 16, 16, 2, 2)
K3P_SM80(k3p_f32_c256_h256_bm64_bn32_wm16_s1_mb2, float, 256, 256, 64, 32, 16, 1, 2)
K3P_SM80(k3p_f32_c256_h256_bm64_bn16_wm32_s1_mb4, float, 256, 256, 64, 16, 32, 1, 4)
K3P_SM80(k3p_f32_c256_h256_bm64_bn32_wm32_s1_mb2, float, 256, 256, 64, 32, 32, 1, 2)
K3P_SM80(k3p_f32_c64_h64_bm128_bn16_wm32_s2_mb4, float, 64, 64, 128, 16, 32, 2, 4)
K3P_SM80(k3p_f32_c64_h128_bm128_bn16_wm32_s2_mb4, float, 64, 128, 128, 16, 32, 2, 4)
K3P_SM80(k3p_f32_c384_h384_bm64_bn16_wm16_s1_mb1, float, 384, 384, 64, 16, 16, 1, 1)
K3P_SM80(k3p_f32_c64_h256_bm64_bn16_wm32_s1_mb4, float, 64, 256, 64, 16, 32, 1, 4)
K3P_SM80(k3p_f32_c64_h384_bm64_bn16_wm16_s1_mb2, float, 64, 384, 64, 16, 16, 1, 2)
K3P_SM80(k3p_f32_c128_h64_bm64_bn32_wm32_s1_mb4, float, 128, 64, 64, 32, 32, 1, 4)
K3P_SM80(k3p_f32_c128_h256_bm64_bn16_wm32_s1_mb4, float, 128, 256, 64, 16, 32, 1, 4)
K3P_SM80(k3p_f32_c128_h384_bm64_bn16_wm16_s1_mb2, float, 128, 384, 64, 16, 16, 1, 2)
K3P_SM80(k3p_f32_c256_h64_bm64_bn16_wm32_s1_mb4, float, 256, 64, 64, 16, 32, 1, 4)
K3P_SM80(k3p_f32_c256_h128_bm64_bn16_wm32_s1_mb4, float, 256, 128, 64, 16, 32, 1, 4)
K3P_SM80(k3p_f32_c256_h384_bm64_bn16_wm16_s1_mb2, float, 256, 384, 64, 16, 16, 1, 2)
K3P_SM80(k3p_f32_c384_h64_bm64_bn16_wm16_s1_mb2, float, 384, 64, 64, 16, 16, 1, 2)
K3P_SM80(k3p_f32_c384_h128_bm64_bn16_wm16_s1_mb2, float, 384, 128, 64, 16, 16, 1, 2)
K3P_SM80(k3p_f32_c384_h256_bm64_bn16_wm16_s1_mb2, float, 384, 256, 64, 16, 16, 1, 2)
// 16-row-warp / 32-channel-chunk rows of the other widths and of the bf16 pair tensor (measurement)
K3P_SM80(k3p_f32_c128_h128_bm64_bn32_wm16_s1_mb2, float, 128, 128, 64, 32, 16, 1, 2)
K3P_SM80(k3p_f32_c128_h128_bm64_bn32_wm16_s2_mb2, float, 128, 128, 64, 32, 16, 2, 2)
K3_SM80(k3_bf16_c256_h256_bm64_bn32_wm16_s1_mb2, __nv_bfloat16, 256, 256, 64, 32, 16, 1, 2)
K3_SM80(k3_bf16_c256_h256_bm64_bn16_wm16_s2_mb2, __nv_bfloat16, 256, 256, 64, 16, 16, 2, 2)
K3_SM80(k3_bf16_c128_h128_bm64_bn32_wm16_s2_mb2, __nv_bfloat16, 128, 128, 64, 32, 16, 2, 2)
K3_SM80(k3_bf16_c128_h128_bm64_bn32_wm16_s2_mb3, __nv_bfloat16, 128, 128, 64, 32, 16, 2, 3)
// tanh-approx gate rows (measurement)
K3T_SM80(k3t_bf16_c128_h128_bm128_bn32_wm32_s2_mb2, __nv_bfloat16, 128, 128, 128, 32, 32, 2, 2)
K3T_SM80(k3t_bf16_c64_h64_bm128_bn32_wm32_s2_mb4, __nv_bfloat16, 64, 64, 128, 32, 32, 2, 4)
K3T_SM80(k3t_bf16_c64_h128_bm64_bn32_wm32_s2_mb4, __nv_bfloat16, 64, 128, 64, 32, 32, 2, 4)
K3T_SM80(k3t_bf16_c256_h256_bm64_bn16_wm32_s1_mb4, __nv_bfloat16, 256, 256, 64, 16, 32, 1, 4)
// c384 tuning points
K3_SM80(k3_bf16_c384_h384_bm64_bn16_wm16_s2_mb1, __nv_bfloat16, 384, 384, 64, 16, 16, 2, 1)
K3_SM80(k3_bf16_c384_h384_bm64_bn16_wm16_s1_mb2, __nv_bfloat16, 384, 384, 64, 16, 16, 1, 2)
K3_SM80(k3_bf16_c384_h384_bm64_bn32_wm16_s1_mb1, __nv_bfloat16, 384, 384, 64, 32, 16, 1, 1)
K3P_SM80(k3p_f32_c384_h384_bm64_bn32_wm16_s1_mb1, float, 384, 384, 64, 32, 16, 1, 1)
