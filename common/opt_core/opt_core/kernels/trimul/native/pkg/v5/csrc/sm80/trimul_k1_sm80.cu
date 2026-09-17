// build: archs=sm_80
// trimul_k1_sm80.cu -- K1 (prologue) of the triangle multiplication, sm_80 member:
//   for every pair token t = (i, k) of batch b:  x = LN_in(z[b, t, :])  (fp32 statistics; bf16 or fp32 z read natively)
//   ab[plane(n, b), i, k] = bf16( sigmoid(x . Wg[n]) * (x . Wp[n]) * mask[b, t] )   n in [0, 2*c_hidden): a-planes then b-planes,
//   written channel-major into zero-padded planes [(2*)B*c_hidden][Np][Np] (Np = ceil16(N)); the incoming direction reads the token (k, i)
//   instead, i.e. writes the TRANSPOSED planes, so both directions contract with the same NT batched GEMM.  Rows i >= N and columns k >= N
//   of every plane are written as zeros by this kernel (no memset anywhere).
// One templated body; the extern "C" instantiations at the end are tuning-table points (tile shape per width), not code paths.
// CTA = BM tokens of one plane row i (grid: x = k tiles over Np, y = i in [0, Np), z = batch); NW = BM/WM warps, each owning WM token rows;
// A operand (the LayerNormed rows) register-resident in mma fragment order straight from global (16 B per lane per 32 columns, then a
// quad transpose into the standard k order); weight chunks of BN output channels (gate rows | proj rows) stream through a
// cp.async ring of STAGES slots; CTA-wide epilogue staging [BN][BM+8] (two buffers by chunk parity, the plane stores of chunk j deferred past
// the ring barrier of chunk j+1) -> BM*2-byte channel-major row segments as 16-byte stores.
#include "math_sm80.cuh"

using namespace tm80;

struct K1Params {
  const void* z;                 // [B, N, N, C] bf16 or fp32, contiguous
  const float* mask;             // [B, N, N] or [N, N] (mask_bstride 0) fp32 0/1, or nullptr
  __nv_bfloat16* ab;             // planes: a-plane of (b, n<D) at index b*D + n, b-plane of (b, n>=D) at B*D + b*D + (n-D); each [Np][Np]
  const __nv_bfloat16* wg;       // [2D][C] bf16 gate weights (w_ag rows then w_bg rows), natural [out, in] layout
  const __nv_bfloat16* wp;       // [2D][C] bf16 projection weights (w_ap | w_bp)
  const float* gamma;            // [C] LN_in weight
  const float* beta;             // [C] LN_in bias
  __nv_bfloat16* zln;            // [B, N, N, C] bf16 LayerNorm output rows, written when the instantiation has ZOUT (else unused, may be null)
  long long z_bstride;           // N*N*C
  long long mask_bstride;        // N*N, or 0 for one shared mask
  long long plane_elems;         // Np*Np
  int N, Np, B, D;
  int outgoing;                  // 1: token (i, k) = z[b, i, k]; 0: token (i, k) = z[b, k, i] (transposed planes)
  float eps;
};

template <typename ZT, int C, int DD, int BM, int BN, int WM, int STAGES, bool EXACT = false, bool ZOUT = false, int SIG = 0, int OPTS = -1>
TM_DEVI void k1_body(const K1Params& p) {
  constexpr int NW = BM / WM, NT = NW * 32, MI = WM / 16, KS = C / 16, ROWB = C * 2;
  constexpr int NCH = DD / BN, NTL = BN / 8, SP = BM + 8;                        // SP: staging pitch (tokens + 8) of one channel row
  constexpr int WCH_BYTES = 2 * BN * ROWB, STG_ELEMS = BN * SP;                    // one CTA-wide staging buffer [BN ch][SP]; two, by chunk parity
  constexpr int WAITN = STAGES >= 2 ? STAGES - 2 : 0;
  using ZRaw = typename RowRawT<ZT, C>::type;
  // the warp's MI*2 raw token rows are all requested from global memory before the first LayerNorm when they fit a 128-B-per-lane budget
  // (one memory round trip for the warp's rows instead of one per row); else one row at a time as before.  OPTS bit 0 (development
  // override) forces it on / off; -1 = the budget rule.
  constexpr bool AEARLY = OPTS >= 0 ? (OPTS & 1) != 0 : (MI * 2 * sizeof(ZRaw) <= 128);
  // NPO: the chunk's GEMM runs one 16-channel column pair (np) at a time with that pair's gate epilogue right behind it, so the special-function
  // epilogue of pair np and the tensor-core work of pair np+1 are independent instruction streams the scheduler interleaves (accumulators live
  // per pair: MI*16 fp32 instead of MI*BN).  PKST: the epilogue packs (ch, ch+1) pairs to bf16x2, transposes 8x8 (token x channel) blocks across
  // the warp with movmatrix and writes 4-B token pairs into the channel-major staging (half the shared-memory store instructions and
  // conversions of the scalar path).  OPTS bits 1 / 2 force them; -1 = on (NPO only for BN >= 32).
  constexpr bool NPO = (OPTS >= 0 ? (OPTS & 2) != 0 : true) && NTL >= 4;        // needs >= 2 column pairs to interleave (BN >= 32); with one pair (BN 16) the split loop measured 2.3x slower
  constexpr bool PKST = OPTS >= 0 ? (OPTS & 4) != 0 : true;
  static_assert(WM % 16 == 0 && BM % WM == 0 && C % 32 == 0 && BN % 16 == 0 && DD % BN == 0 && STAGES >= 1 && (SP * 2) % 16 == 0, "tile parameters");
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, q = lane & 3, gid = lane >> 2;
  const uint32_t sW_u = smem_u32(smem);
  __nv_bfloat16* const sStage = reinterpret_cast<__nv_bfloat16*>(smem + STAGES * WCH_BYTES);

  const int b = blockIdx.z, i = blockIdx.y, k0 = blockIdx.x * BM;
  const int N = p.N, Np = p.Np, D = p.D;
  const bool row_ok = i < N;
  const ZT* zb = reinterpret_cast<const ZT*>(p.z) + (size_t)b * (size_t)p.z_bstride;
  const size_t tok_base = p.outgoing ? (size_t)i * N * C : (size_t)i * C;            // element offset of token (i, k=0)
  const size_t tok_stride = p.outgoing ? (size_t)C : (size_t)N * C;
  const float* mb = p.mask ? p.mask + (size_t)b * (size_t)p.mask_bstride : nullptr;
  const size_t m_base = p.outgoing ? (size_t)i * N : (size_t)i, m_stride = p.outgoing ? (size_t)1 : (size_t)N;
  __nv_bfloat16* pa = p.ab + ((size_t)b * D) * (size_t)p.plane_elems + (size_t)i * Np;            // a-plane 0 of this b, row i
  __nv_bfloat16* pb = p.ab + ((size_t)p.B * D + (size_t)b * D) * (size_t)p.plane_elems + (size_t)i * Np;

  bool use_m = p.mask != nullptr;                                               // no mask: the reference applies no multiply (pad rows force it: x 0)
  auto gate = [&](float g, float pr, float m) {                                  // sigmoid(g) * p (* mask): the shared statement's gate
    if constexpr (SIG == 0) { return use_m ? math::gate(g, pr, m) : math::gate(g, pr); }
    else { const float o = __fmul_rn(sigmoid_t<SIG>(g), pr); return use_m ? __fmul_rn(o, m) : o; }
  };
  auto load_w = [&](int chunk, int slot) {
    const uint32_t base = sW_u + (uint32_t)(slot * WCH_BYTES);
    cp_rows<C, NT>(base, p.wg + (size_t)chunk * BN * C, C, BN, tid);
    cp_rows<C, NT>(base + BN * ROWB, p.wp + (size_t)chunk * BN * C, C, BN, tid);
  };
  // the staged [BN][BM] tile of chunk jc -> its BN channel-major plane rows: BM*2-byte row segments as 16-B stores by the whole CTA
  // (columns >= Np dropped; pad tokens were staged as zeros).  Called after the CTA barrier that follows the staging writes.
  auto store_tile = [&](int jc) {
    constexpr int GR = BM / 8;
    const __nv_bfloat16* sb = sStage + (jc & 1) * STG_ELEMS;
#pragma unroll
    for (int sidx = tid; sidx < BN * GR; sidx += NT) {
      const int ch = sidx / GR, g = sidx % GR, k = k0 + g * 8;
      if (k < Np) {
        const int n = jc * BN + ch;
        __nv_bfloat16* dst = (n < D ? pa + (size_t)n * (size_t)p.plane_elems : pb + (size_t)(n - D) * (size_t)p.plane_elems) + k;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(sb + ch * SP + g * 8);
      }
    }
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) { if (s < NCH) load_w(s, s); cp_commit(); }

  // ---- A fragments: this warp's WM token rows, LayerNormed, register-resident (one row at a time through the quad loader)
  uint32_t fa[MI][KS][4];
  float mrow[MI][2];
  bool anybad = false;                                                          // some row of this warp is a pad token: its outputs must be zeros
  ZRaw zraw[AEARLY ? MI : 1][2];
  if constexpr (AEARLY) {
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int k = k0 + warp * WM + mi * 16 + gid + 8 * h;
        const bool ok = row_ok && k < N;
        row_fetch<C>(zb + tok_base + (size_t)(ok ? k : 0) * tok_stride, ok, q, zraw[mi][h]);
        mrow[mi][h] = ok ? (mb ? __ldg(mb + m_base + (size_t)k * m_stride) : 1.f) : 0.f;
      }
  }
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int k = k0 + warp * WM + mi * 16 + gid + 8 * h;
      const bool ok = row_ok && k < N;
      uint32_t row[C / 8];
      if constexpr (AEARLY) {
        row_ln<C, EXACT>(zraw[mi][h], p.gamma, p.beta, p.eps, q, row);
      } else {
        ln_row_load<ZT, C, EXACT>(zb + tok_base + (size_t)(ok ? k : 0) * tok_stride, ok, p.gamma, p.beta, p.eps, q, row);
      }
      if constexpr (ZOUT) {                                                     // the LayerNorm output row leaves once, for the epilogue kernel (16 B per lane per 32 columns)
        if (ok) {
          __nv_bfloat16* zl = p.zln + (size_t)b * (size_t)p.z_bstride + tok_base + (size_t)k * tok_stride + 8 * q;
#pragma unroll
          for (int kk = 0; kk < C / 32; ++kk) *reinterpret_cast<uint4*>(zl + 32 * kk) = make_uint4(row[4 * kk], row[4 * kk + 1], row[4 * kk + 2], row[4 * kk + 3]);
        }
      }
      scatter_row_frags<C>(row, fa[mi], h, q);
      if constexpr (!AEARLY) mrow[mi][h] = ok ? (mb ? __ldg(mb + m_base + (size_t)k * m_stride) : 1.f) : 0.f;
      anybad |= !ok;
    }
  }
  use_m = use_m || anybad;
  // ---- weight chunks
#pragma unroll 1
  for (int j = 0; j < NCH; ++j) {
    cp_wait<WAITN>();
    __syncthreads();                                                            // chunk j landed; every warp is done with chunk j-1 (its slot AND its staging writes)
    { const int jn = j + STAGES - 1; if (jn < NCH) load_w(jn, jn % STAGES); cp_commit(); }
    if (j > 0) store_tile(j - 1);                                               // deferred plane stores of the previous chunk (other parity buffer)
    if (STAGES == 1) { cp_wait<0>(); __syncthreads(); }
    const uint32_t wb = sW_u + (uint32_t)((j % STAGES) * WCH_BYTES);
    // ---- gate epilogue of one (m16 tile, n8 tile) accumulator pair -> CTA-wide staging [BN ch][BM tok] (parity j & 1); the plane stores happen
    //      after the next barrier
    __nv_bfloat16* sOut = sStage + (j & 1) * STG_ELEMS;
    auto epi = [&](int mi, int nt, const float (&ag)[4], const float (&ap)[4]) {
      const float o0 = gate(ag[0], ap[0], mrow[mi][0]), o1 = gate(ag[1], ap[1], mrow[mi][0]);         // token rA: channels ch, ch+1
      const float o2 = gate(ag[2], ap[2], mrow[mi][1]), o3 = gate(ag[3], ap[3], mrow[mi][1]);         // token rB = rA + 8
      if constexpr (PKST) {
        const uint32_t tA = movm_trans(pack_bf16(o0, o1)), tB = movm_trans(pack_bf16(o2, o3));       // -> channel nt*8+gid, tokens 2q, 2q+1 (| +8)
        __nv_bfloat16* d = sOut + (nt * 8 + gid) * SP + warp * WM + mi * 16 + 2 * q;
        *reinterpret_cast<uint32_t*>(d) = tA; *reinterpret_cast<uint32_t*>(d + 8) = tB;
      } else {
        const int rA = warp * WM + mi * 16 + gid, rB = rA + 8, ch = nt * 8 + 2 * q;
        sOut[ch * SP + rA] = __float2bfloat16_rn(o0); sOut[(ch + 1) * SP + rA] = __float2bfloat16_rn(o1);
        sOut[ch * SP + rB] = __float2bfloat16_rn(o2); sOut[(ch + 1) * SP + rB] = __float2bfloat16_rn(o3);
      }
    };
    if constexpr (NPO) {
#pragma unroll
      for (int np = 0; np < NTL / 2; ++np) {
        float accg[MI][2][4], accp[MI][2][4];
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
          for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) { accg[mi][t][e] = 0.f; accp[mi][t][e] = 0.f; }
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
          uint32_t bg[4], bp[4];
          load_b16<ROWB>(bg, wb, np * 16, ks, lane);
          load_b16<ROWB>(bp, wb + BN * ROWB, np * 16, ks, lane);
#pragma unroll
          for (int mi = 0; mi < MI; ++mi) {
            mma16816(accg[mi][0], fa[mi][ks], bg[0], bg[1]); mma16816(accg[mi][1], fa[mi][ks], bg[2], bg[3]);
            mma16816(accp[mi][0], fa[mi][ks], bp[0], bp[1]); mma16816(accp[mi][1], fa[mi][ks], bp[2], bp[3]);
          }
        }
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) { epi(mi, 2 * np, accg[mi][0], accp[mi][0]); epi(mi, 2 * np + 1, accg[mi][1], accp[mi][1]); }
      }
    } else {
      float accg[MI][NTL][4], accp[MI][NTL][4];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int nt = 0; nt < NTL; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) { accg[mi][nt][e] = 0.f; accp[mi][nt][e] = 0.f; }
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
#pragma unroll
        for (int np = 0; np < NTL / 2; ++np) {
          uint32_t bg[4], bp[4];
          load_b16<ROWB>(bg, wb, np * 16, ks, lane);
          load_b16<ROWB>(bp, wb + BN * ROWB, np * 16, ks, lane);
#pragma unroll
          for (int mi = 0; mi < MI; ++mi) {
            mma16816(accg[mi][2 * np], fa[mi][ks], bg[0], bg[1]); mma16816(accg[mi][2 * np + 1], fa[mi][ks], bg[2], bg[3]);
            mma16816(accp[mi][2 * np], fa[mi][ks], bp[0], bp[1]); mma16816(accp[mi][2 * np + 1], fa[mi][ks], bp[2], bp[3]);
          }
        }
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int nt = 0; nt < NTL; ++nt) epi(mi, nt, accg[mi][nt], accp[mi][nt]);
    }
  }
  cp_wait<0>();
  __syncthreads();
  store_tile(NCH - 1);
}

#define K1_SM80(NAME, ZT, C_, DD_, BM_, BN_, WM_, ST_, MB_)                                                            \
  extern "C" __global__ void __launch_bounds__((BM_ / WM_) * 32, MB_) NAME(const __grid_constant__ K1Params p) {     \
    k1_body<ZT, C_, DD_, BM_, BN_, WM_, ST_>(p);                                                                      \
  }
// bit-exact variant (stock-order LayerNorm; c_z = c_hidden = 256, bf16 pair tensor): name prefix k1x_
#define K1X_SM80(NAME, C_, DD_, BM_, BN_, WM_, ST_, MB_)                                                               \
  extern "C" __global__ void __launch_bounds__((BM_ / WM_) * 32, MB_) NAME(const __grid_constant__ K1Params p) {     \
    k1_body<__nv_bfloat16, C_, DD_, BM_, BN_, WM_, ST_, true>(p);                                                     \
  }
// tanh-approx gate rows (outside the acceptance class; measurement only): name prefix k1t_
#define K1T_SM80(NAME, ZT, C_, DD_, BM_, BN_, WM_, ST_, MB_)                                                           \
  extern "C" __global__ void __launch_bounds__((BM_ / WM_) * 32, MB_) NAME(const __grid_constant__ K1Params p) {     \
    k1_body<ZT, C_, DD_, BM_, BN_, WM_, ST_, false, false, 1>(p);                                                     \
  }
// variant that also writes the LayerNorm output rows (p.zln) for an epilogue that does not re-read the pair tensor: name prefix k1z_
#define K1Z_SM80(NAME, ZT, C_, DD_, BM_, BN_, WM_, ST_, MB_)                                                           \
  extern "C" __global__ void __launch_bounds__((BM_ / WM_) * 32, MB_) NAME(const __grid_constant__ K1Params p) {     \
    k1_body<ZT, C_, DD_, BM_, BN_, WM_, ST_, false, true>(p);                                                         \
  }
// name grammar (parsed by the host): k1_<bf16|f32>_c<C>_h<c_hidden>_bm<BM>_bn<BN>_wm<WM>_s<STAGES>_mb<minblocks>
// dynamic smem = STAGES*2*BN*C*2 + 2*BN*(BM+8)*2 bytes (the host computes the same formula and opts in above 48 KiB)
K1_SM80(k1_bf16_c128_h128_bm128_bn16_wm32_s2_mb3, __nv_bfloat16, 128, 256, 128, 16, 32, 2, 3)
K1_SM80(k1_bf16_c128_h128_bm128_bn16_wm32_s2_mb2, __nv_bfloat16, 128, 256, 128, 16, 32, 2, 2)
K1_SM80(k1_bf16_c128_h128_bm64_bn16_wm16_s2_mb4,  __nv_bfloat16, 128, 256, 64, 16, 16, 2, 4)
K1_SM80(k1_bf16_c128_h128_bm128_bn32_wm32_s2_mb2, __nv_bfloat16, 128, 256, 128, 32, 32, 2, 2)
K1_SM80(k1_bf16_c128_h128_bm128_bn32_wm32_s2_mb3, __nv_bfloat16, 128, 256, 128, 32, 32, 2, 3)
K1_SM80(k1_bf16_c128_h128_bm256_bn16_wm32_s2_mb1, __nv_bfloat16, 128, 256, 256, 16, 32, 2, 1)
K1_SM80(k1_f32_c128_h128_bm128_bn16_wm32_s2_mb2,  float,         128, 256, 128, 16, 32, 2, 2)
K1_SM80(k1_f32_c128_h128_bm64_bn16_wm16_s2_mb4,   float,         128, 256, 64, 16, 16, 2, 4)
K1_SM80(k1_f32_c128_h128_bm128_bn16_wm32_s2_mb3,  float,         128, 256, 128, 16, 32, 2, 3)
K1_SM80(k1_f32_c128_h128_bm128_bn32_wm32_s2_mb3,  float,         128, 256, 128, 32, 32, 2, 3)
K1_SM80(k1_bf16_c256_h256_bm128_bn32_wm32_s2_mb2, __nv_bfloat16, 256, 512, 128, 32, 32, 2, 2)
K1_SM80(k1_bf16_c256_h256_bm128_bn16_wm32_s2_mb2, __nv_bfloat16, 256, 512, 128, 16, 32, 2, 2)
K1_SM80(k1_bf16_c256_h256_bm64_bn32_wm16_s2_mb3,  __nv_bfloat16, 256, 512, 64, 32, 16, 2, 3)
K1_SM80(k1_f32_c256_h256_bm128_bn16_wm32_s2_mb2,  float,         256, 512, 128, 16, 32, 2, 2)
K1_SM80(k1_f32_c256_h256_bm64_bn32_wm16_s2_mb3,   float,         256, 512, 64, 32, 16, 2, 3)
K1_SM80(k1_f32_c256_h256_bm128_bn32_wm32_s2_mb2,  float,         256, 512, 128, 32, 32, 2, 2)
K1_SM80(k1_bf16_c64_h64_bm128_bn16_wm32_s2_mb4,   __nv_bfloat16, 64, 128, 128, 16, 32, 2, 4)
K1_SM80(k1_f32_c64_h64_bm128_bn16_wm32_s2_mb4,    float,         64, 128, 128, 16, 32, 2, 4)
K1_SM80(k1_bf16_c64_h64_bm128_bn32_wm32_s2_mb4,   __nv_bfloat16, 64, 128, 128, 32, 32, 2, 4)
K1_SM80(k1_bf16_c64_h64_bm128_bn64_wm32_s2_mb3,   __nv_bfloat16, 64, 128, 128, 64, 32, 2, 3)
K1_SM80(k1_bf16_c64_h128_bm128_bn16_wm32_s2_mb4,  __nv_bfloat16, 64, 256, 128, 16, 32, 2, 4)
K1_SM80(k1_f32_c64_h128_bm128_bn16_wm32_s2_mb4,   float,         64, 256, 128, 16, 32, 2, 4)
K1_SM80(k1_bf16_c64_h128_bm128_bn32_wm32_s2_mb4,  __nv_bfloat16, 64, 256, 128, 32, 32, 2, 4)
K1_SM80(k1_bf16_c64_h128_bm128_bn64_wm32_s2_mb3,  __nv_bfloat16, 64, 256, 128, 64, 32, 2, 3)
K1_SM80(k1_bf16_c384_h384_bm64_bn16_wm16_s2_mb2,  __nv_bfloat16, 384, 768, 64, 16, 16, 2, 2)
K1_SM80(k1_f32_c384_h384_bm64_bn16_wm16_s2_mb2,   float,         384, 768, 64, 16, 16, 2, 2)
// bit-exact rows (one per diagonal width; the stock library serves c_hidden == c_z only)
K1X_SM80(k1x_bf16_c256_h256_bm128_bn32_wm32_s2_mb2, 256, 512, 128, 32, 32, 2, 2)
K1X_SM80(k1x_bf16_c64_h64_bm128_bn32_wm32_s2_mb4, 64, 128, 128, 32, 32, 2, 4)
K1X_SM80(k1x_bf16_c128_h128_bm128_bn32_wm32_s2_mb3, 128, 256, 128, 32, 32, 2, 3)
K1X_SM80(k1x_bf16_c384_h384_bm64_bn16_wm16_s2_mb2, 384, 768, 64, 16, 16, 2, 2)
K1X_SM80(k1x_bf16_c384_h384_bm64_bn16_wm16_s2_mb3, 384, 768, 64, 16, 16, 2, 3)
K1X_SM80(k1x_bf16_c256_h256_bm128_bn16_wm32_s2_mb2, 256, 512, 128, 16, 32, 2, 2)
// the remaining (c_z, c_hidden) pairs of {64, 128, 256, 384}^2: table rows by the c_z rule above
K1_SM80(k1_bf16_c64_h256_bm128_bn32_wm32_s2_mb4, __nv_bfloat16, 64, 512, 128, 32, 32, 2, 4)
K1_SM80(k1_f32_c64_h256_bm128_bn16_wm32_s2_mb4, float, 64, 512, 128, 16, 32, 2, 4)
K1_SM80(k1_bf16_c64_h384_bm128_bn32_wm32_s2_mb4, __nv_bfloat16, 64, 768, 128, 32, 32, 2, 4)
K1_SM80(k1_f32_c64_h384_bm128_bn16_wm32_s2_mb4, float, 64, 768, 128, 16, 32, 2, 4)
K1_SM80(k1_bf16_c128_h64_bm128_bn32_wm32_s2_mb3, __nv_bfloat16, 128, 128, 128, 32, 32, 2, 3)
K1_SM80(k1_f32_c128_h64_bm128_bn32_wm32_s2_mb3, float, 128, 128, 128, 32, 32, 2, 3)
K1_SM80(k1_bf16_c128_h256_bm128_bn32_wm32_s2_mb3, __nv_bfloat16, 128, 512, 128, 32, 32, 2, 3)
K1_SM80(k1_f32_c128_h256_bm128_bn32_wm32_s2_mb3, float, 128, 512, 128, 32, 32, 2, 3)
K1_SM80(k1_bf16_c128_h384_bm128_bn32_wm32_s2_mb3, __nv_bfloat16, 128, 768, 128, 32, 32, 2, 3)
K1_SM80(k1_f32_c128_h384_bm128_bn32_wm32_s2_mb3, float, 128, 768, 128, 32, 32, 2, 3)
K1_SM80(k1_bf16_c256_h64_bm128_bn32_wm32_s2_mb2, __nv_bfloat16, 256, 128, 128, 32, 32, 2, 2)
K1_SM80(k1_f32_c256_h64_bm128_bn32_wm32_s2_mb2, float, 256, 128, 128, 32, 32, 2, 2)
K1_SM80(k1_bf16_c256_h128_bm128_bn32_wm32_s2_mb2, __nv_bfloat16, 256, 256, 128, 32, 32, 2, 2)
K1_SM80(k1_f32_c256_h128_bm128_bn32_wm32_s2_mb2, float, 256, 256, 128, 32, 32, 2, 2)
K1_SM80(k1_bf16_c256_h384_bm128_bn32_wm32_s2_mb2, __nv_bfloat16, 256, 768, 128, 32, 32, 2, 2)
K1_SM80(k1_f32_c256_h384_bm128_bn32_wm32_s2_mb2, float, 256, 768, 128, 32, 32, 2, 2)
K1_SM80(k1_bf16_c384_h64_bm64_bn16_wm16_s2_mb2, __nv_bfloat16, 384, 128, 64, 16, 16, 2, 2)
K1_SM80(k1_f32_c384_h64_bm64_bn16_wm16_s2_mb2, float, 384, 128, 64, 16, 16, 2, 2)
K1_SM80(k1_bf16_c384_h128_bm64_bn16_wm16_s2_mb2, __nv_bfloat16, 384, 256, 64, 16, 16, 2, 2)
K1_SM80(k1_f32_c384_h128_bm64_bn16_wm16_s2_mb2, float, 384, 256, 64, 16, 16, 2, 2)
K1_SM80(k1_bf16_c384_h256_bm64_bn16_wm16_s2_mb2, __nv_bfloat16, 384, 512, 64, 16, 16, 2, 2)
K1_SM80(k1_f32_c384_h256_bm64_bn16_wm16_s2_mb2, float, 384, 512, 64, 16, 16, 2, 2)
// LayerNorm-rows-out rows (fp32 pair tensor: the epilogue then reads bf16 rows instead of fp32 z)
K1Z_SM80(k1z_f32_c128_h128_bm128_bn16_wm32_s2_mb2, float, 128, 256, 128, 16, 32, 2, 2)
K1Z_SM80(k1z_f32_c128_h128_bm64_bn16_wm16_s2_mb4, float, 128, 256, 64, 16, 16, 2, 4)
K1Z_SM80(k1z_f32_c128_h128_bm128_bn16_wm32_s2_mb3, float, 128, 256, 128, 16, 32, 2, 3)
K1Z_SM80(k1z_f32_c128_h128_bm128_bn32_wm32_s2_mb3, float, 128, 256, 128, 32, 32, 2, 3)
K1Z_SM80(k1z_f32_c256_h256_bm128_bn16_wm32_s2_mb2, float, 256, 512, 128, 16, 32, 2, 2)
K1Z_SM80(k1z_f32_c256_h256_bm64_bn32_wm16_s2_mb3, float, 256, 512, 64, 32, 16, 2, 3)
K1Z_SM80(k1z_f32_c256_h256_bm128_bn32_wm32_s2_mb2, float, 256, 512, 128, 32, 32, 2, 2)
K1Z_SM80(k1z_f32_c64_h64_bm128_bn16_wm32_s2_mb4, float, 64, 128, 128, 16, 32, 2, 4)
K1Z_SM80(k1z_f32_c64_h128_bm128_bn16_wm32_s2_mb4, float, 64, 256, 128, 16, 32, 2, 4)
K1Z_SM80(k1z_f32_c384_h384_bm64_bn16_wm16_s2_mb2, float, 384, 768, 64, 16, 16, 2, 2)
K1Z_SM80(k1z_f32_c64_h256_bm128_bn16_wm32_s2_mb4, float, 64, 512, 128, 16, 32, 2, 4)
K1Z_SM80(k1z_f32_c64_h384_bm128_bn16_wm32_s2_mb4, float, 64, 768, 128, 16, 32, 2, 4)
K1Z_SM80(k1z_f32_c128_h64_bm128_bn32_wm32_s2_mb3, float, 128, 128, 128, 32, 32, 2, 3)
K1Z_SM80(k1z_f32_c128_h256_bm128_bn32_wm32_s2_mb3, float, 128, 512, 128, 32, 32, 2, 3)
K1Z_SM80(k1z_f32_c128_h384_bm128_bn32_wm32_s2_mb3, float, 128, 768, 128, 32, 32, 2, 3)
K1Z_SM80(k1z_f32_c256_h64_bm128_bn32_wm32_s2_mb2, float, 256, 128, 128, 32, 32, 2, 2)
K1Z_SM80(k1z_f32_c256_h128_bm128_bn32_wm32_s2_mb2, float, 256, 256, 128, 32, 32, 2, 2)
K1Z_SM80(k1z_f32_c256_h384_bm128_bn32_wm32_s2_mb2, float, 256, 768, 128, 32, 32, 2, 2)
K1Z_SM80(k1z_f32_c384_h64_bm64_bn16_wm16_s2_mb2, float, 384, 128, 64, 16, 16, 2, 2)
K1Z_SM80(k1z_f32_c384_h128_bm64_bn16_wm16_s2_mb2, float, 384, 256, 64, 16, 16, 2, 2)
K1Z_SM80(k1z_f32_c384_h256_bm64_bn16_wm16_s2_mb2, float, 384, 512, 64, 16, 16, 2, 2)
// tanh-approx gate rows (measurement)
K1T_SM80(k1t_bf16_c128_h128_bm128_bn32_wm32_s2_mb3, __nv_bfloat16, 128, 256, 128, 32, 32, 2, 3)
K1T_SM80(k1t_bf16_c64_h64_bm128_bn32_wm32_s2_mb4, __nv_bfloat16, 64, 128, 128, 32, 32, 2, 4)
K1T_SM80(k1t_bf16_c64_h128_bm128_bn32_wm32_s2_mb4, __nv_bfloat16, 64, 256, 128, 32, 32, 2, 4)
K1T_SM80(k1t_bf16_c256_h256_bm128_bn32_wm32_s2_mb2, __nv_bfloat16, 256, 512, 128, 32, 32, 2, 2)
// c384 tuning points
K1_SM80(k1_bf16_c384_h384_bm64_bn32_wm16_s2_mb2, __nv_bfloat16, 384, 768, 64, 32, 16, 2, 2)
K1_SM80(k1_bf16_c384_h384_bm64_bn16_wm16_s3_mb2, __nv_bfloat16, 384, 768, 64, 16, 16, 3, 2)
K1_SM80(k1_bf16_c384_h384_bm128_bn16_wm16_s2_mb1, __nv_bfloat16, 384, 768, 128, 16, 16, 2, 1)
K1_SM80(k1_bf16_c384_h384_bm64_bn16_wm16_s2_mb3, __nv_bfloat16, 384, 768, 64, 16, 16, 2, 3)
