// flash_transition_sm90.cu — the Protenix-v2 pair Transition statement  z + Linear_o( SiLU(Linear_a(LN(z))) * Linear_b(LN(z)) )  as ONE sm_90a kernel
// (variant LN=0: the LayerNorm runs first as the stock kernel and this kernel consumes its output y; variant LN=1: the kernel consumes the raw rows and
// computes the LayerNorm in its prologue with the stock fast_layernorm arithmetic — see ln_row_lane).  Per 128-row tile: y stays in shared memory, the hidden dimension
// (1024) is processed in chunks of 32: GEMM1 (wgmma SS, m64n64k16 chain over K=256) -> SwiGLU in registers -> GEMM2 (wgmma RS, m64n256k16, the h chunk
// is the register A operand) accumulating out[128,256] in fp32 registers; weight chunks stream through TMA rings ([Wa_j;Wb_j] 64x256 x3 slots, Wo chunk
// 256x32 x4 slots) multicast to a 2-CTA cluster; 1 producer warpgroup + 2 consumer warpgroups (64 rows each); residual add + bf16 stores through an
// smem-staged, 16-byte coalesced epilogue.
// Numerics = the stock chain under bf16 autocast, bit for bit: bf16 x bf16 products accumulated in fp32 in ascending K (== the cuBLAS result on the
// shapes used here, checked with torch.equal on model tensors), a = bf16(acc), silu(a) -> bf16, b*silu(a) -> bf16, u = bf16(acc2), out = bf16(fp32(z)+fp32(u)).
// The SiLU on a bf16 value is checked EXHAUSTIVELY (all 65536 bf16 inputs) against torch's F.silu at load time (silu_table).
// Structure credits: NVIDIA CUTLASS/CuTe (BSD-3-Clause) sm90 TMA/wgmma tutorial and warp-specialized mainloop conventions (headers are a build-time
// dependency); the accumulator->A-operand register relayout `convert_layout_acc_Aregs` follows FlashAttention-3 hopper/utils.h (BSD-3-Clause, Dao AI Lab).
// See LICENSE_NOTE.md next to this file.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>

using namespace cute;

template <typename Layout0>
CUTE_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {      // FA3 (BSD-3): ((2,2,V),MMA_M,MMA_N) -> ((2,2,2),MMA_M,(V/2,MMA_N))
  auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});
  return make_layout(make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)), get<1>(acc_layout), coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
}

namespace flash_transition {
using T = cutlass::bfloat16_t;
constexpr int kBM = 128, kC = 256, kCluster = 2, kConsumerWG = 2, kThreads = 128 * (1 + kConsumerWG);

template <int BH_, int S1_, int S2_>
struct Cfg {
  static constexpr int BH = BH_, S1 = S1_, S2 = S2_;
  using AtomW1 = GMMA::Layout_K_SW128_Atom<T>;
  using AtomW2 = std::conditional_t<(BH % 64 == 0), GMMA::Layout_K_SW128_Atom<T>, GMMA::Layout_K_SW64_Atom<T>>;
  using SmemLayoutY  = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<T>{}, make_shape(Int<kBM>{}, Int<kC>{})));          // (128, 256)
  using SmemLayoutW1 = decltype(tile_to_shape(AtomW1{}, make_shape(Int<2 * BH>{}, Int<kC>{}, Int<S1>{})));                    // (2BH, 256, S1)
  using SmemLayoutW2 = decltype(tile_to_shape(AtomW2{}, make_shape(Int<kC>{}, Int<BH>{}, Int<S2>{})));                        // (256, BH, S2)
  using Mma1 = decltype(make_tiled_mma(SM90::GMMA::ss_op_selector<T, T, float, Shape<Int<64>, Int<2 * BH>, Int<kC>>>()));     // m64 n(2BH) SS
  using MmaH = decltype(make_tiled_mma(SM90::GMMA::ss_op_selector<T, T, float, Shape<Int<64>, Int<BH>, Int<kC>>>()));         // layout donor
  using Mma2 = decltype(make_tiled_mma(SM90::GMMA::rs_op_selector<T, T, float, Shape<Int<64>, Int<kC>, Int<BH>>>()));         // m64 n256 RS
  struct SharedStorage {
    alignas(1024) cute::ArrayEngine<T, cosize_v<SmemLayoutY>>  Y;
    alignas(1024) cute::ArrayEngine<T, cosize_v<SmemLayoutW1>> W1;
    alignas(1024) cute::ArrayEngine<T, cosize_v<SmemLayoutW2>> W2;
    alignas(8) uint64_t y_full;
    uint64_t w1_full[S1], w1_empty[S1], w2_full[S2], w2_empty[S2];
  };
};

// SiLU of a bf16-valued float: silu(a) = a * rcp(1 + 2^(-a*log2 e)) with ex2.approx / rcp.approx.  The stock op (F.silu on a bf16 tensor, fp32 math,
// round-to-nearest-even to bf16) is a function of 65536 inputs; silu_table() below reproduces this function for every input so the loader can compare
// it with torch's own output bit for bit before the kernel is used.
__device__ __forceinline__ float silu_f(float af) {
  float e, r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(-af * 1.4426950408889634f));
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(1.0f + e));
  return af * r;
}
// Pairwise SwiGLU: every bf16 rounding through cvt.rn.bf16x2.f32 (one instruction per PAIR of values):
// rne(a0),rne(b0) | rne(a1),rne(b1) | rne(sv0),rne(sv1) | rne(h0),rne(h1) -> the packed h register (low half = element i, high half = element i+1).
__device__ __forceinline__ float bf16lo_as_float(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi_as_float(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) { __nv_bfloat162 t = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&t); }
__device__ __forceinline__ uint32_t swiglu_pair(float a0, float a1, float b0, float b1) {
  uint32_t ab0 = pack_bf16x2(a0, b0), ab1 = pack_bf16x2(a1, b1);
  float a0r = bf16lo_as_float(ab0), b0r = bf16hi_as_float(ab0), a1r = bf16lo_as_float(ab1), b1r = bf16hi_as_float(ab1);
  uint32_t svp = pack_bf16x2(silu_f(a0r), silu_f(a1r));
  return pack_bf16x2(b0r * bf16lo_as_float(svp), b1r * bf16hi_as_float(svp));   // bf16 x bf16 products are exact in fp32; one rounding like the stock `b * silu(a)`
}
__global__ void silu_table_kernel(__nv_bfloat16* out) {           // out[i] = bf16(silu(bf16 with bit pattern i))
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 65536) { __nv_bfloat16_raw r; r.x = (unsigned short)i; float a = __bfloat162float(__nv_bfloat16(r)); out[i] = __float2bfloat16(silu_f(a)); }
}

template <int STAGES> struct Ring {
  int idx = 0; uint32_t phase;
  __device__ Ring(uint32_t p0) : phase(p0) {}
  __device__ void advance() { if (++idx == STAGES) { idx = 0; phase ^= 1u; } }
};


// ---- LayerNorm in the prologue (variant LN=1).  Arithmetic = the stock Protenix fast_layernorm forward (LayerNormForwardV2<bf16, float4>, built with
// --use_fast_math: every fp32 op flush-to-zero, division -> rcp.approx + multiply/fma, rsqrt -> rsqrt.approx) reproduced instruction for instruction per
// lane: one warp per row, lane l owns columns [8l, 8l+8): sequential single-element Welford over its 8 values, xor-butterfly (16,8,4,2,1) Welford merge,
// var = m2 * rcp(count), inv = rsqrt(max(var,0) + eps), out = bf16_rn(((x - mean) * inv) * gamma + beta) with gamma/beta the bf16-cast parameters.
// Inline PTX pins each rounding/flush/contraction so the result does not depend on this file's compiler flags.
__device__ __forceinline__ float ftz_add(float a, float b) { float r; asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float ftz_sub(float a, float b) { float r; asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float ftz_mul(float a, float b) { float r; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float ftz_fma(float a, float b, float c) { float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float ftz_rcp(float a) { float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float ftz_rsqrt(float a) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float ftz_max(float a, float b) { float r; asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float bf16lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf16hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
// one row: v = this lane's 8 raw bf16 (as 4 x b32, element 2k = low half), g/b = gamma/beta for the same 8 columns; returns the 8 normalized bf16 packed.
__device__ __forceinline__ uint4 ln_row_lane(uint4 v, uint4 g, uint4 b, float eps) {
  float x[8] = {bf16lo(v.x), bf16hi(v.x), bf16lo(v.y), bf16hi(v.y), bf16lo(v.z), bf16hi(v.z), bf16lo(v.w), bf16hi(v.w)};
  float mean = 0.f, m2 = 0.f, count = 0.f;
  CUTE_UNROLL
  for (int i = 0; i < 8; ++i) {
    count = ftz_add(count, 1.f); float rc = ftz_rcp(count);
    float d1 = ftz_sub(x[i], mean); mean = ftz_fma(d1, rc, mean); float d2 = ftz_sub(x[i], mean); m2 = ftz_fma(d1, d2, m2);
  }
  CUTE_UNROLL
  for (int mask = 16; mask >= 1; mask >>= 1) {
    float bm = __shfl_xor_sync(0xffffffffu, mean, mask), bm2 = __shfl_xor_sync(0xffffffffu, m2, mask), bc = __shfl_xor_sync(0xffffffffu, count, mask);
    if (bc != 0.f) {
      float nc = ftz_add(count, bc); float rc = ftz_rcp(nc); float delta = ftz_sub(bm, mean);
      float dd = ftz_mul(delta, delta); float nb = ftz_mul(bc, rc); float ddc = ftz_mul(count, dd);
      mean = ftz_fma(nb, delta, mean); float t = ftz_fma(nb, ddc, bm2); m2 = ftz_add(m2, t); count = nc;
    }
  }
  float var = ftz_mul(ftz_rcp(count), m2); var = ftz_max(var, 0.f); float inv = ftz_rsqrt(ftz_add(var, eps));
  float gg[8] = {bf16lo(g.x), bf16hi(g.x), bf16lo(g.y), bf16hi(g.y), bf16lo(g.z), bf16hi(g.z), bf16lo(g.w), bf16hi(g.w)};
  float bb[8] = {bf16lo(b.x), bf16hi(b.x), bf16lo(b.y), bf16hi(b.y), bf16lo(b.z), bf16hi(b.z), bf16lo(b.w), bf16hi(b.w)};
  float o[8];
  CUTE_UNROLL
  for (int i = 0; i < 8; ++i) { float t = ftz_mul(inv, ftz_sub(x[i], mean)); o[i] = ftz_fma(t, gg[i], bb[i]); }
  uint4 r; r.x = pack_bf16x2(o[0], o[1]); r.y = pack_bf16x2(o[2], o[3]); r.z = pack_bf16x2(o[4], o[5]); r.w = pack_bf16x2(o[6], o[7]);
  return r;
}

template <int R>
__device__ __forceinline__ void ln_rows_lane_multi(uint4 (&v)[R], uint4 g, uint4 b, float eps, uint4 (&out)[R]) {
  float x[R][8], mean[R], m2[R], count[R];
  CUTE_UNROLL
  for (int rr = 0; rr < R; ++rr) {
    x[rr][0] = bf16lo(v[rr].x); x[rr][1] = bf16hi(v[rr].x); x[rr][2] = bf16lo(v[rr].y); x[rr][3] = bf16hi(v[rr].y);
    x[rr][4] = bf16lo(v[rr].z); x[rr][5] = bf16hi(v[rr].z); x[rr][6] = bf16lo(v[rr].w); x[rr][7] = bf16hi(v[rr].w);
    mean[rr] = 0.f; m2[rr] = 0.f; count[rr] = 0.f;
  }
  CUTE_UNROLL
  for (int i = 0; i < 8; ++i) {
    CUTE_UNROLL
    for (int rr = 0; rr < R; ++rr) {
      count[rr] = ftz_add(count[rr], 1.f); float rc = ftz_rcp(count[rr]);
      float d1 = ftz_sub(x[rr][i], mean[rr]); mean[rr] = ftz_fma(d1, rc, mean[rr]); float d2 = ftz_sub(x[rr][i], mean[rr]); m2[rr] = ftz_fma(d1, d2, m2[rr]);
    }
  }
  CUTE_UNROLL
  for (int mask = 16; mask >= 1; mask >>= 1) {
    float bm[R], bm2[R], bc[R];
    CUTE_UNROLL
    for (int rr = 0; rr < R; ++rr) { bm[rr] = __shfl_xor_sync(0xffffffffu, mean[rr], mask); bm2[rr] = __shfl_xor_sync(0xffffffffu, m2[rr], mask); bc[rr] = __shfl_xor_sync(0xffffffffu, count[rr], mask); }
    CUTE_UNROLL
    for (int rr = 0; rr < R; ++rr) {
      if (bc[rr] != 0.f) {
        float nc = ftz_add(count[rr], bc[rr]); float rc = ftz_rcp(nc); float delta = ftz_sub(bm[rr], mean[rr]);
        float dd = ftz_mul(delta, delta); float nb = ftz_mul(bc[rr], rc); float ddc = ftz_mul(count[rr], dd);
        mean[rr] = ftz_fma(nb, delta, mean[rr]); float t = ftz_fma(nb, ddc, bm2[rr]); m2[rr] = ftz_add(m2[rr], t); count[rr] = nc;
      }
    }
  }
  float gg[8] = {bf16lo(g.x), bf16hi(g.x), bf16lo(g.y), bf16hi(g.y), bf16lo(g.z), bf16hi(g.z), bf16lo(g.w), bf16hi(g.w)};
  float bb[8] = {bf16lo(b.x), bf16hi(b.x), bf16lo(b.y), bf16hi(b.y), bf16lo(b.z), bf16hi(b.z), bf16lo(b.w), bf16hi(b.w)};
  CUTE_UNROLL
  for (int rr = 0; rr < R; ++rr) {
    float var = ftz_mul(ftz_rcp(count[rr]), m2[rr]); var = ftz_max(var, 0.f); float inv = ftz_rsqrt(ftz_add(var, eps));
    float o[8];
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) { float t = ftz_mul(inv, ftz_sub(x[rr][i], mean[rr])); o[i] = ftz_fma(t, gg[i], bb[i]); }
    out[rr].x = pack_bf16x2(o[0], o[1]); out[rr].y = pack_bf16x2(o[2], o[3]); out[rr].z = pack_bf16x2(o[4], o[5]); out[rr].w = pack_bf16x2(o[6], o[7]);
  }
}
// standalone form (load-time check of the LN arithmetic against the LayerNorm module on real rows): one warp per row, global in / global out.
__global__ void ln_rows_kernel(T const* __restrict__ x, int ld_x, T const* __restrict__ g, T const* __restrict__ b, T* __restrict__ y, int ld_y, int M, float eps) {
  int lane = threadIdx.x & 31; long row = (long(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  if (row >= M) return;
  uint4 v = *reinterpret_cast<uint4 const*>(x + row * ld_x + 8 * lane);
  uint4 gv = *reinterpret_cast<uint4 const*>(g + 8 * lane), bv = *reinterpret_cast<uint4 const*>(b + 8 * lane);
  uint4 o = ln_row_lane(v, gv, bv, eps);
  *reinterpret_cast<uint4*>(y + row * ld_y + 8 * lane) = o;
}

template <class CFG, bool LN, int LNR, class TmaY, class TmaW1, class TmaW2>
__global__ static __launch_bounds__(kThreads, 1)
void kernel(int M, int NH, __grid_constant__ TmaY const tma_y, __grid_constant__ TmaW1 const tma_w1, __grid_constant__ TmaW2 const tma_w2,
            T const* __restrict__ res, T* __restrict__ out, int ld_out, int has_res, T const* __restrict__ lnw, T const* __restrict__ lnb, float eps)
{
  constexpr int BH = CFG::BH, S1 = CFG::S1, S2 = CFG::S2;
  using SharedStorage = typename CFG::SharedStorage;
  extern __shared__ char shared_memory[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  using FullBar = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBar = cutlass::arch::ClusterBarrier;
  const int wg = __shfl_sync(0xffffffff, int(threadIdx.x) / 128, 0);   // PROVABLY warp-uniform (CUTLASS canonical_warp_group_idx): smem descriptors derived from it live in uniform registers
  const int tid = threadIdx.x % 128;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_predicate = cute::elect_one_sync();
  const uint32_t cta_rank = cute::block_rank_in_cluster();
  const int nchunk = NH / BH;

  if (warp_idx == 0 && lane_predicate) {
    FullBar::init(&smem.y_full, 1);
    for (int s = 0; s < S1; ++s) { FullBar::init(&smem.w1_full[s], 1); EmptyBar::init(&smem.w1_empty[s], 4 * kConsumerWG * kCluster); }
    for (int s = 0; s < S2; ++s) { FullBar::init(&smem.w2_full[s], 1); EmptyBar::init(&smem.w2_empty[s], 4 * kConsumerWG * kCluster); }
  }
  cutlass::arch::fence_barrier_init();
  cute::cluster_sync();

  Tensor sY  = make_tensor(make_smem_ptr(smem.Y.begin()),  typename CFG::SmemLayoutY{});
  Tensor sW1 = make_tensor(make_smem_ptr(smem.W1.begin()), typename CFG::SmemLayoutW1{});
  Tensor sW2 = make_tensor(make_smem_ptr(smem.W2.begin()), typename CFG::SmemLayoutW2{});
  const uint16_t mask = uint16_t((1u << kCluster) - 1u);

  if (wg == 0) {
    // =============================== producer warpgroup: warp 0 -> y + W1 ring, warp 1 -> W2 ring ===============================
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_idx == 0 && lane_predicate) {
      Tensor mY  = tma_y.get_tma_tensor(make_shape(M, Int<kC>{}));
      Tensor gY  = local_tile(mY, make_shape(Int<kBM>{}, Int<kC>{}), make_coord(int(blockIdx.x), 0));
      auto [tYg, tYs] = tma_partition(tma_y, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sY), group_modes<0,2>(gY));
      FullBar::arrive_and_expect_tx(&smem.y_full, int(sizeof(T)) * kBM * kC);
      copy(tma_y.with(smem.y_full), tYg, tYs);
      Tensor mW1 = tma_w1.get_tma_tensor(make_shape(2 * NH, Int<kC>{}));
      Tensor gW1 = local_tile(mW1, make_shape(Int<2 * BH>{}, Int<kC>{}), make_coord(_, 0));                              // (2BH, 256, nchunk)
      auto [tW1g, tW1s] = tma_partition(tma_w1, cta_rank, Layout<Int<kCluster>>{}, group_modes<0,2>(sW1), group_modes<0,2>(gW1));
      constexpr int kBytesW1 = int(sizeof(T)) * 2 * BH * kC;
      Ring<S1> p1(1);
      for (int c = 0; c < nchunk; ++c) {
        EmptyBar::wait(&smem.w1_empty[p1.idx], p1.phase);
        FullBar::arrive_and_expect_tx(&smem.w1_full[p1.idx], kBytesW1);
        copy(tma_w1.with(smem.w1_full[p1.idx], mask), tW1g(_, c), tW1s(_, p1.idx));
        p1.advance();
      }
    } else if (warp_idx == 1 && lane_predicate) {
      Tensor mW2 = tma_w2.get_tma_tensor(make_shape(Int<kC>{}, NH));
      Tensor gW2 = local_tile(mW2, make_shape(Int<kC>{}, Int<BH>{}), make_coord(0, _));                                    // (256, BH, nchunk)
      auto [tW2g, tW2s] = tma_partition(tma_w2, cta_rank, Layout<Int<kCluster>>{}, group_modes<0,2>(sW2), group_modes<0,2>(gW2));
      constexpr int kBytesW2 = int(sizeof(T)) * kC * BH;
      Ring<S2> p2(1);
      for (int c = 0; c < nchunk; ++c) {
        EmptyBar::wait(&smem.w2_empty[p2.idx], p2.phase);
        FullBar::arrive_and_expect_tx(&smem.w2_full[p2.idx], kBytesW2);
        copy(tma_w2.with(smem.w2_full[p2.idx], mask), tW2g(_, c), tW2s(_, p2.idx));
        p2.advance();
      }
    }
  } else {
    // =============================== consumer warpgroups ===============================
    cutlass::arch::warpgroup_reg_alloc<240>();
    const int cw = wg - 1;
    typename CFG::Mma1 mma1; typename CFG::Mma2 mma2; typename CFG::MmaH mmah;
    // smem-descriptor operands are warpgroup-collective: slice with a CONSTANT thread index (CUTLASS: tiled_mma.get_slice(warp_group_thread_layout(wg)))
    // so the descriptors are uniform values (UR registers); per-thread register operands (accumulators, h, epilogue coords) use the real thread slice.
    auto wgs1 = mma1.get_slice(Int<0>{});
    auto wgs2 = mma2.get_slice(Int<0>{});
    auto thr2 = mma2.get_thread_slice(tid);
    Tensor sYw = local_tile(sY, make_shape(Int<64>{}, Int<kC>{}), make_coord(cw, 0));           // (64, 256): this warpgroup's rows (cw is warp-uniform)
    Tensor tCrY  = wgs1.make_fragment_A(wgs1.partition_A(sYw));                                  // (MMA, 1, 16)
    Tensor tCrW1 = wgs1.make_fragment_B(wgs1.partition_B(sW1));                                  // (MMA, 1, 16, S1)
    Tensor tCrW2 = wgs2.make_fragment_B(wgs2.partition_B(sW2));                                  // (MMA, 1, BH/16, S2)
    constexpr int KB1 = decltype(size<2>(tCrW1))::value, KB2 = decltype(size<2>(tCrW2))::value;
    static_assert(KB1 == kC / 16 && KB2 == BH / 16 && decltype(size<2>(tCrY))::value == KB1, "k blocks");
    Tensor acc1  = partition_fragment_C(mma1, make_shape(Int<64>{}, Int<2 * BH>{}));              // GEMM1 accumulator (a | b halves)
    Tensor acc2  = partition_fragment_C(mma2, make_shape(Int<64>{}, Int<kC>{}));                   // 128 fp32 / thread
    Tensor accH  = partition_fragment_C(mmah, make_shape(Int<64>{}, Int<BH>{}));                  // layout donor
    Tensor rHa = make_tensor<T>(convert_layout_acc_Aregs(accH.layout()));                          // ((2,2,2), 1, BH/16) bf16, double-buffered:
    Tensor rHb = make_tensor<T>(convert_layout_acc_Aregs(accH.layout()));                          //   h(c) is written while GEMM2(c-1) still reads the other buffer
    Tensor rHa32 = recast<uint32_t>(rHa); Tensor rHb32 = recast<uint32_t>(rHb);
    constexpr int NV_H = decltype(size(rHa))::value;
    static_assert(NV_H % 2 == 0 && decltype(size(rHa32))::value * 2 == NV_H, "uint32 view pairs consecutive bf16 elements");
    static_assert(NV_H * 2 == decltype(size(acc1))::value && decltype(size<2>(rHa))::value == KB2, "a|b halves / k blocks");
    const bool leader = (tid % 32) == 0;
    // Ping-pong (FlashAttention-3 style): the two consumer warpgroups take turns ISSUING their wgmma batches (named barriers 1 and 2), so the tensor
    // core executes WG0's batch, then WG1's, ... and each warpgroup's SwiGLU runs while the other warpgroup's GEMMs execute (instead of both
    // warpgroups waiting on the same TMA barrier, issuing together, and doing their ALU phases together with the tensor core idle).
    auto tc_acquire = [&]() { cutlass::arch::NamedBarrier::sync(256, 1 + cw); };
    auto tc_handoff = [&]() { cutlass::arch::NamedBarrier::arrive(256, 2 - cw); };
    if (cw == 1) cutlass::arch::NamedBarrier::arrive(256, 1);                                    // WG0 issues first
    const int row0 = int(blockIdx.x) * kBM + 64 * cw;
    constexpr int VEC_PER_ROW = kC * int(sizeof(T)) / 16;                                       // 32 x 16-byte vectors per row
    constexpr int NVEC = (64 * VEC_PER_ROW) / 128;                                               // 16 vectors per thread (a warp covers one row per step)
    uint4 resv[NVEC];
    auto prefetch_residual = [&]() {
      if (has_res) {
        CUTE_UNROLL
        for (int j = 0; j < NVEC; ++j) {
          int v = tid + 128 * j, r = v / VEC_PER_ROW, c8 = (v % VEC_PER_ROW) * 8;
          if (row0 + r < M) resv[j] = *reinterpret_cast<uint4 const*>(res + size_t(row0 + r) * ld_out + c8);
        }
      }
    };
    auto release = [&](uint64_t* bar) { if (leader) { for (uint32_t r = 0; r < uint32_t(kCluster); ++r) { EmptyBar::arrive(bar, r, 1u); } } };
    auto issue_gemm1 = [&](int slot) {                                 // acc1 = y . W1(slot)^T, one ascending k16 chain, left in flight
      warpgroup_fence_operand(acc1);
      warpgroup_arrive();
      mma1.accumulate_ = GMMA::ScaleOut::Zero;
      CUTE_UNROLL
      for (int k = 0; k < KB1; ++k) { gemm(mma1, tCrY(_, _, k), tCrW1(_, _, k, slot), acc1); mma1.accumulate_ = GMMA::ScaleOut::One; }
      warpgroup_commit_batch();
    };
    auto issue_gemm2 = [&](int slot, auto& rH, auto& rH32) {           // acc2 += h . W2(slot)^T (h = rH registers), left in flight
      warpgroup_fence_operand(acc2); warpgroup_fence_operand(rH32);
      warpgroup_arrive();
      CUTE_UNROLL
      for (int k = 0; k < KB2; ++k) { gemm(mma2, rH(_, _, k), tCrW2(_, _, k, slot), acc2); }
      warpgroup_commit_batch();
    };
    auto swiglu_to = [&](auto& rH32) {                                 // h(c) <- SwiGLU of the retired GEMM1 accumulator (reads acc1 only: no in-place accumulator
      CUTE_UNROLL                                                      // writes, so no register renaming is needed while GEMM2(c-1) is still pending)
      for (int i = 0; i < NV_H; i += 2) { rH32(i / 2) = swiglu_pair(acc1(i), acc1(i + 1), acc1(i + NV_H), acc1(i + 1 + NV_H)); }
    };
    FullBar::wait(&smem.y_full, 0);
    if constexpr (LN) {
      // sY holds the RAW rows here; normalize this warpgroup's 64 rows in place (warp w: rows 16w..16w+15, one full warp per row = the stock lane mapping),
      // then make the generic-proxy smem writes visible to the async proxy (wgmma reads sY through descriptors) and sync the warpgroup.
      const int lane = tid & 31, wq = tid >> 5;
      const uint4 gv = *reinterpret_cast<uint4 const*>(lnw + 8 * lane), bv = *reinterpret_cast<uint4 const*>(lnb + 8 * lane);
      CUTE_UNROLL
      for (int j0 = 0; j0 < 16; j0 += LNR) {                          // LNR independent rows in flight per warp (the per-row chains are latency-bound)
        uint4* p[LNR]; uint4 v[LNR], o[LNR];
        CUTE_UNROLL
        for (int rr = 0; rr < LNR; ++rr) { p[rr] = reinterpret_cast<uint4*>(&sYw(16 * wq + j0 + rr, 8 * lane)); v[rr] = *p[rr]; }
        ln_rows_lane_multi<LNR>(v, gv, bv, eps, o);
        CUTE_UNROLL
        for (int rr = 0; rr < LNR; ++rr) { if (row0 + 16 * wq + j0 + rr < M) *p[rr] = o[rr]; }   // rows past M keep the TMA zero-fill (never stored)
      }
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(128, 3 + cw);
    }
    clear(acc2);
    Ring<S1> r1(0); Ring<S2> r2(0);
    // Schedule (per consumer warpgroup; wgmma group count is 0 or 1 at every loop boundary, bodies are branch-free):
    //   prologue : G1(0); wait<0>; SwiGLU(0) -> hA
    //   step(c)  : issue G1(c); issue G2(c-1)[h_prev]; wait<1> (=> G1(c) and G2(c-2) retired); release W1(c), W2(c-2); SwiGLU(c): acc1 -> h_cur
    //   final    : G2(last)[h_last]; wait<0>; release
    int prev_s2 = -1;
    {
      int s1 = r1.idx; uint32_t p1 = r1.phase; r1.advance();
      FullBar::wait(&smem.w1_full[s1], p1);
      tc_acquire(); issue_gemm1(s1); tc_handoff();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc1);
      release(&smem.w1_empty[s1]);
      swiglu_to(rHa32);
    }
    auto step = [&](auto& h_prev, auto& h_prev32, auto& h_cur32) {
      int s1 = r1.idx; uint32_t p1 = r1.phase; r1.advance();
      int s2 = r2.idx; uint32_t p2 = r2.phase; r2.advance();          // W2 slot of chunk c-1
      FullBar::wait(&smem.w1_full[s1], p1);
      FullBar::wait(&smem.w2_full[s2], p2);
      tc_acquire();
      issue_gemm1(s1);                                                 // G1(c)
      issue_gemm2(s2, h_prev, h_prev32);                               // G2(c-1)
      tc_handoff();
      warpgroup_wait<1>();                                             // G1(c) retired (and G2(c-2)); G2(c-1) may run on
      warpgroup_fence_operand(acc1);
      release(&smem.w1_empty[s1]);
      if (prev_s2 >= 0) release(&smem.w2_empty[prev_s2]);
      prev_s2 = s2;
      swiglu_to(h_cur32);                                              // overlaps G2(c-1) on the tensor core
    };
    // chunks 1..nchunk-1 (nchunk even -> odd count): pairs (hA->hB, hB->hA) then one last step writing hB
    for (int c = 1; c + 1 < nchunk; c += 2) { step(rHa, rHa32, rHb32); step(rHb, rHb32, rHa32); }
    step(rHa, rHa32, rHb32);
    {
      int s2 = r2.idx; uint32_t p2 = r2.phase; r2.advance();
      FullBar::wait(&smem.w2_full[s2], p2);
      tc_acquire(); issue_gemm2(s2, rHb, rHb32); tc_handoff();         // G2(last)
      prefetch_residual();                                             // residual rows stream in under the last GEMM2
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc2);
      release(&smem.w2_empty[prev_s2]); release(&smem.w2_empty[s2]);
    }
    // ---- epilogue.  Stage this warpgroup's [64, 256] output tile (bf16-rounded = the stock rounding of the fp32 GEMM result) through ITS OWN rows of
    // the y tile in SMEM (dead once this warpgroup's last GEMM1 retired; the other warpgroup never reads them), then stream out coalesced 16-byte
    // vectors with the residual add (bf16(fp32(x) + fp32(u)) = the stock `z += u` rounding); the residual vectors were prefetched under the last GEMM2.
    {
      Tensor cO = make_identity_tensor(make_shape(Int<64>{}, Int<kC>{}));
      Tensor tCcO = thr2.partition_C(cO);                                        // (row, col) of each accumulator value; (i, i+1) = adjacent columns
      CUTE_UNROLL
      for (int i = 0; i < size(acc2); i += 2) {
        int r = get<0>(tCcO(i)), col = get<1>(tCcO(i));
        *reinterpret_cast<uint32_t*>(&sYw(r, col)) = pack_bf16x2(acc2(i), acc2(i + 1));   // swizzle keeps 16-byte chunks intact -> (col, col+1) adjacent
      }
      cutlass::arch::NamedBarrier::sync(128, 3 + cw);                             // this warpgroup only
      CUTE_UNROLL
      for (int j = 0; j < NVEC; ++j) {
        int v = tid + 128 * j, r = v / VEC_PER_ROW, c8 = (v % VEC_PER_ROW) * 8;
        if (row0 + r < M) {
          uint4 u4 = *reinterpret_cast<uint4 const*>(&sYw(r, c8));
          if (has_res) {
            uint32_t* up = reinterpret_cast<uint32_t*>(&u4); uint32_t const* rp = reinterpret_cast<uint32_t const*>(&resv[j]);
            CUTE_UNROLL
            for (int q = 0; q < 4; ++q) {
              up[q] = pack_bf16x2(bf16lo_as_float(rp[q]) + bf16lo_as_float(up[q]), bf16hi_as_float(rp[q]) + bf16hi_as_float(up[q]));
            }
          }
          *reinterpret_cast<uint4*>(out + size_t(row0 + r) * ld_out + c8) = u4;
        }
      }
    }
  }
  cute::cluster_sync();
}

template <int BH, int S1, int S2, bool LN, int LNR>
void launch(torch::Tensor y, torch::Tensor w1p, torch::Tensor wo, c10::optional<torch::Tensor> res, torch::Tensor out, T const* lnw, T const* lnb, float eps) {
  using CFG = Cfg<BH, S1, S2>;
  int M = y.size(0); int NH = wo.size(1);
  TORCH_CHECK(y.size(1) == kC && wo.size(0) == kC && w1p.size(0) == 2 * NH && w1p.size(1) == kC && NH % (2 * BH) == 0, "shapes");
  TORCH_CHECK(y.stride(1) == 1 && (y.stride(0) * 2) % 16 == 0 && w1p.is_contiguous() && wo.is_contiguous() && out.is_contiguous(), "layouts");
  TORCH_CHECK(!res.has_value() || (res->stride(1) == 1 && res->stride(0) == out.stride(0)), "res layout");
  auto mY  = make_tensor(make_gmem_ptr(reinterpret_cast<T const*>(y.data_ptr())),   make_shape(M, Int<kC>{}),      make_stride(int(y.stride(0)), Int<1>{}));
  auto mW1 = make_tensor(make_gmem_ptr(reinterpret_cast<T const*>(w1p.data_ptr())), make_shape(2 * NH, Int<kC>{}), make_stride(Int<kC>{}, Int<1>{}));
  auto mW2 = make_tensor(make_gmem_ptr(reinterpret_cast<T const*>(wo.data_ptr())),  make_shape(Int<kC>{}, NH),     make_stride(NH, Int<1>{}));
  typename CFG::SmemLayoutY sY; typename CFG::SmemLayoutW1 sW1; typename CFG::SmemLayoutW2 sW2;
  auto tmaY  = make_tma_atom(SM90_TMA_LOAD{},           mY,  sY,           make_shape(Int<kBM>{}, Int<kC>{}));
  auto tmaW1 = make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mW1, sW1(_, _, 0), make_shape(Int<2 * BH>{}, Int<kC>{}), Int<kCluster>{});
  auto tmaW2 = make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mW2, sW2(_, _, 0), make_shape(Int<kC>{}, Int<BH>{}),     Int<kCluster>{});
  int smem_bytes = int(sizeof(typename CFG::SharedStorage));
  auto* kptr = &kernel<CFG, LN, LNR, decltype(tmaY), decltype(tmaW1), decltype(tmaW2)>;
  static bool attr_done = false;
  if (!attr_done) { C10_CUDA_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); attr_done = true; }
  int n_cta = (M + kBM - 1) / kBM; n_cta = (n_cta + kCluster - 1) / kCluster * kCluster;
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(n_cta); cfg.blockDim = dim3(kThreads); cfg.dynamicSmemBytes = smem_bytes; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1]; attr[0].id = cudaLaunchAttributeClusterDimension; attr[0].val.clusterDim.x = kCluster; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  T const* resp = res.has_value() ? reinterpret_cast<T const*>(res->data_ptr()) : nullptr;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kptr, M, NH, tmaY, tmaW1, tmaW2, resp, reinterpret_cast<T*>(out.data_ptr()), int(out.stride(0)), int(res.has_value()), lnw, lnb, eps));
}

template <int BH, int S1, int S2> int64_t smem_of() { return int64_t(sizeof(typename Cfg<BH, S1, S2>::SharedStorage)); }
inline void silu_table_launch(torch::Tensor out) { silu_table_kernel<<<256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<__nv_bfloat16*>(out.data_ptr())); C10_CUDA_CHECK(cudaGetLastError()); }
}  // namespace flash_transition

// ---- host entry points -------------------------------------------------------------------------------------------------------------------------
// flash_transition(y, w1p, wo, res, out):  y [M,256] bf16 (LayerNorm output, unit column stride, 16-byte aligned rows);  w1p [2*NH,256] bf16 = per hidden
// chunk j of 32: rows [64j,64j+32) = Wa[32j:32j+32], rows [64j+32,64j+64) = Wb[32j:32j+32];  wo [256,NH] bf16 (= linear_no_bias.weight);  res [M,256] bf16 or
// None;  out [M,256] bf16 (written; must not alias y).  NH % 64 == 0.  One instantiation: 32-wide hidden chunks, rings of 3 (W1) and 4 (Wo) slots.
constexpr int kBH = 32, kS1 = 3, kS2 = 4;
void flash_transition_fwd(torch::Tensor y, torch::Tensor w1p, torch::Tensor wo, c10::optional<torch::Tensor> res, torch::Tensor out) {
  const at::cuda::OptionalCUDAGuard guard(y.device());
  TORCH_CHECK(y.scalar_type() == torch::kBFloat16 && w1p.scalar_type() == torch::kBFloat16 && wo.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16, "flash_transition: dtypes must be bf16");
  TORCH_CHECK(!res.has_value() || res->scalar_type() == torch::kBFloat16, "flash_transition: residual dtype must be bf16");
  TORCH_CHECK(y.dim() == 2 && out.dim() == 2 && y.size(0) == out.size(0) && y.size(0) > 0, "flash_transition: y/out rows");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(y.data_ptr()) % 16) == 0 && (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16) == 0 && (!res.has_value() || (reinterpret_cast<uintptr_t>(res->data_ptr()) % 16) == 0), "flash_transition: 16-byte alignment");
  flash_transition::launch<kBH, kS1, kS2, false, 1>(y, w1p, wo, res, out, nullptr, nullptr, 0.f);
}
// flash_transition_ln(x, w1p, wo, res, out, ln_w, ln_b, eps): same statement with the LayerNorm computed in-kernel from the RAW rows x (bf16 [M,256]);
// ln_w / ln_b = the LayerNorm weight / bias cast to bf16 ([256], contiguous); eps = its epsilon.  out must not alias x.
constexpr int kLNR = 4;      // rows in flight per warp in the LayerNorm prologue
void flash_transition_ln_fwd(torch::Tensor x, torch::Tensor w1p, torch::Tensor wo, c10::optional<torch::Tensor> res, torch::Tensor out, torch::Tensor ln_w, torch::Tensor ln_b, double eps) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w1p.scalar_type() == torch::kBFloat16 && wo.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16, "flash_transition_ln: dtypes must be bf16");
  TORCH_CHECK(ln_w.scalar_type() == torch::kBFloat16 && ln_b.scalar_type() == torch::kBFloat16 && ln_w.numel() == flash_transition::kC && ln_b.numel() == flash_transition::kC && ln_w.is_contiguous() && ln_b.is_contiguous(), "flash_transition_ln: ln_w/ln_b must be contiguous bf16 [256]");
  TORCH_CHECK(!res.has_value() || res->scalar_type() == torch::kBFloat16, "flash_transition_ln: residual dtype must be bf16");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && x.size(0) == out.size(0) && x.size(0) > 0, "flash_transition_ln: x/out rows");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) == 0 && (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16) == 0 && (!res.has_value() || (reinterpret_cast<uintptr_t>(res->data_ptr()) % 16) == 0)
              && (reinterpret_cast<uintptr_t>(ln_w.data_ptr()) % 16) == 0 && (reinterpret_cast<uintptr_t>(ln_b.data_ptr()) % 16) == 0, "flash_transition_ln: 16-byte alignment");
  auto lw = reinterpret_cast<flash_transition::T const*>(ln_w.data_ptr()); auto lb = reinterpret_cast<flash_transition::T const*>(ln_b.data_ptr());
  flash_transition::launch<kBH, kS1, kS2, true, kLNR>(x, w1p, wo, res, out, lw, lb, float(eps));
}
// ln_rows(x, ln_w, ln_b, y, eps): y = LayerNorm(x) rows with the in-kernel arithmetic (load-time check against the LayerNorm module on real rows).
void ln_rows(torch::Tensor x, torch::Tensor ln_w, torch::Tensor ln_b, torch::Tensor y, double eps) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && y.scalar_type() == torch::kBFloat16 && ln_w.scalar_type() == torch::kBFloat16 && ln_b.scalar_type() == torch::kBFloat16, "ln_rows: dtypes");
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && x.size(1) == flash_transition::kC && y.size(1) == flash_transition::kC && x.size(0) == y.size(0) && x.stride(1) == 1 && y.stride(1) == 1 && x.stride(0) % 8 == 0 && y.stride(0) % 8 == 0, "ln_rows: layouts");
  TORCH_CHECK(ln_w.numel() == flash_transition::kC && ln_b.numel() == flash_transition::kC && ln_w.is_contiguous() && ln_b.is_contiguous(), "ln_rows: ln_w/ln_b");
  int M = x.size(0); const int threads = 256; long blocks = (long(M) * 32 + threads - 1) / threads;
  flash_transition::ln_rows_kernel<<<unsigned(blocks), threads, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<flash_transition::T const*>(x.data_ptr()), int(x.stride(0)),
      reinterpret_cast<flash_transition::T const*>(ln_w.data_ptr()), reinterpret_cast<flash_transition::T const*>(ln_b.data_ptr()), reinterpret_cast<flash_transition::T*>(y.data_ptr()), int(y.stride(0)), M, float(eps));
  C10_CUDA_CHECK(cudaGetLastError());
}
int64_t smem_bytes() { return flash_transition::smem_of<kBH, kS1, kS2>(); }
int64_t hidden_chunk() { return kBH; }
void silu_table(torch::Tensor out) {
  const at::cuda::OptionalCUDAGuard guard(out.device());
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.numel() == 65536 && out.is_contiguous(), "silu_table: out must be a contiguous bf16 tensor of 65536 elements");
  flash_transition::silu_table_launch(out);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_transition", &flash_transition_fwd, "out = [res +] Linear_o(SiLU(y Wa^T) * (y Wb^T)) for y [M,256] bf16 (see source header for layouts)");
  m.def("flash_transition_ln", &flash_transition_ln_fwd, "out = [res +] Linear_o(SiLU(LN(x) Wa^T) * (LN(x) Wb^T)) with the LayerNorm computed in-kernel from the raw rows x (ln_w, ln_b = bf16 weight/bias [256], eps)");
  m.def("ln_rows", &ln_rows, "y = LayerNorm(x) rows with the in-kernel arithmetic (load-time check)");
  m.def("silu_table", &silu_table, "bf16 SiLU for all 65536 bf16 bit patterns (load-time exactness check against torch F.silu)");
  m.def("smem_bytes", &smem_bytes);
  m.def("hidden_chunk", &hidden_chunk);
}
