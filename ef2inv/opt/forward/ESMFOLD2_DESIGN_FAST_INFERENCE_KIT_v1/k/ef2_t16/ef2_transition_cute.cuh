// ef2_transition_cute.cuh — ESMFold2 pair Transition  out = x + W3( silu(W1 LN(x)) * (W2 LN(x)) )  as ONE persistent, warp-specialised
// sm_90a kernel (CUDA C++ / CuTe, compiled at first use through NVRTC by ef2_transition_cute.py).
//
//   CTA = 384 threads = 3 warpgroups, one CTA per SM (230.5 KB shared memory), persistent over 128-row tiles.
//   WG0  producer : warp 0 lane 0 streams the weight ring (W1_j, W2_j, W3_j per 64-unit hidden chunk j: 32 KB TMA boxes into a 5-slot ring with
//                   full/empty mbarriers); warps 1/2 lane 0 load the activation half-tiles (64 rows x 256, TMA, SW128) of WG1 / WG2.
//   WG1/2 consumers: 64 rows each (240 registers/thread; the producer keeps 24).  LayerNorm in registers (fp32 statistics, two-pass; one bf16
//                   rounding of x_hat) written back IN PLACE over the x half-tile (generic st.shared + fence.proxy.async), then per hidden chunk j:
//                   a = x_hat W1_j^T, b = x_hat W2_j^T (wgmma SS m64n64k16, fp32 accumulators, two commit groups so silu(a) — the MUFU work —
//                   overlaps the b chain), h = bf16(silu(a) * b) built in registers directly in the wgmma A-fragment layout (the m64n64 C fragment
//                   IS the m64k64 A fragment), acc += h W3_j^T (wgmma RS m64n256k16, left in flight into the next chunk).  Epilogue: out =
//                   bf16(fp32(x) + acc) (flags bit 0; else bf16(acc)), residual re-read from L2, direct 32-bit stores.
//   Numerics class = the kit's T10 / k-pair T15 (fast tier): fp32 LN statistics, bf16 operands, fp32 accumulation, h rounded once to bf16,
//   output rounded once; sigmoid = 1/(1 + 2^(-a log2 e)) with ex2.approx + rcp.approx (Triton's tl.sigmoid class).  No TF32, no reduced precision.
//   A development variant with a 2/4-CTA cluster + TMA-multicast weight ring, other sigmoid forms and clock64 phase profiling was measured
//   slower or equal at these shapes and is not part of the kit.
#include <cute/tensor.hpp>
using namespace cute;

constexpr int PROD_REGS = 24, CONS_REGS = 240;     // setmaxnreg split: 128 x 24 + 256 x 240 = 64512 <= 65536 registers

struct __align__(64) TmaDesc { uint64_t w[16]; };

// ------------------------------------------------------------------------------------------------------------------ raw PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void tma_load_2d(const TmaDesc* tm, uint64_t* bar, void* dst, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(tm)), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory"); }
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) { while (!mbar_try_wait(bar, phase)) {} }
__device__ __forceinline__ void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void named_bar_sync(int id, int nthreads) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(nthreads) : "memory"); }
template <int N> __device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N)); }
template <int N> __device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N)); }
__device__ __forceinline__ float ex2_approx(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float rcp_approx(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float silu_f(float a) { return a * rcp_approx(1.f + ex2_approx(-1.4426950408889634f * a)); }   // a * sigmoid(a), sigmoid = 1/(1+2^(-a log2 e))
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) { __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }
__device__ __forceinline__ float2 unpack_bf16x2(uint32_t u) { __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u); return __bfloat1622float2(v); }
__device__ __forceinline__ uint32_t ldg_u32(const void* p) { uint32_t v; asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ void stg_u32(void* p, uint32_t v) { asm volatile("st.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ float warp_allsum(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 16); v += __shfl_xor_sync(0xffffffff, v, 8); v += __shfl_xor_sync(0xffffffff, v, 4);
  v += __shfl_xor_sync(0xffffffff, v, 2);  v += __shfl_xor_sync(0xffffffff, v, 1);  return v;
}

// ------------------------------------------------------------------------------------------------------------------ configuration
constexpr int C_ = 256;                 // pair channels (K of GEMM1, N of GEMM2)
constexpr int H_ = 1024;                // hidden features
constexpr int BM = 128;                 // rows per CTA tile
constexpr int WGM = 64;                 // rows per consumer warpgroup
constexpr int BH = 64;                  // hidden units per chunk
constexpr int NSLOT = 5;                // weight-ring slots (32 KB each)
constexpr int NCHUNK = H_ / BH;         // hidden chunks per tile
constexpr int SLOT_BYTES = BH * C_ * 2; // W1_j [BH x 256] bf16 == W3_j [256 x BH] bf16
constexpr int X_BYTES  = BM * C_ * 2;   // 64 KB activation tile (4 K-slabs of [128 x 64])
constexpr int XH_BYTES = WGM * C_ * 2;  // one WG's half
constexpr int XSLAB    = BM * 64 * 2;   // 16 KB per K-slab
constexpr int XHALF    = WGM * 128;     // byte offset of half c within a slab (64 rows x 128 B)
constexpr int W1SLAB   = BH * 128;      // byte offset per K-slab inside a W1/W2 slot ([BH rows x 128 B])
constexpr int OFF_X    = 0;
constexpr int OFF_RING = OFF_X + X_BYTES;
constexpr int OFF_BAR  = OFF_RING + NSLOT * SLOT_BYTES;
constexpr int NBARS    = 4 + 2 * NSLOT;             // x_full[2], x_empty[2], w_full[NSLOT], w_empty[NSLOT]
constexpr int SMEM_BYTES = OFF_BAR + NBARS * 8 + 1024;
static_assert(SMEM_BYTES <= 232448, "shared memory budget");
constexpr int NSLOTS_PER_TILE = 3 * NCHUNK;
static_assert(H_ % BH == 0, "BH");

using MmaG1 = decltype(make_tiled_mma(GMMA::ss_op_selector<bfloat16_t, bfloat16_t, float, Shape<Int<WGM>, Int<BH>, _16>, GMMA::Major::K, GMMA::Major::K>()));
using MmaG2 = decltype(make_tiled_mma(GMMA::rs_op_selector<bfloat16_t, bfloat16_t, float, Shape<Int<WGM>, Int<C_>, _16>, GMMA::Major::K, GMMA::Major::K>()));
using SwK128 = GMMA::Layout_K_SW128_Atom<bfloat16_t>;
using SLayX  = decltype(tile_to_shape(SwK128{}, Shape<Int<BM>, Int<C_>>{}));      // (128 rows, 256 k): 4 K-slabs of 16 KB, rows at 128 B
using SLayW1 = decltype(tile_to_shape(SwK128{}, Shape<Int<BH>, Int<C_>>{}));      // (BH n, 256 k): 4 K-slabs of BH*128 B
using SLayW3 = decltype(tile_to_shape(SwK128{}, Shape<Int<C_>, Int<BH>>{}));      // (256 n, 64 k): rows at 128 B
using LayHf  = Layout<Shape<Int<WGM>, Int<BH>>, Stride<Int<BH>, _1>>;            // shape carrier for the h register fragment ([64 x BH] K-major)

// ------------------------------------------------------------------------------------------------------------------ the kernel
extern "C" __global__ void __launch_bounds__(384, 1)
ef2_transition(const __grid_constant__ TmaDesc tmX, const __grid_constant__ TmaDesc tmW12, const __grid_constant__ TmaDesc tmW3,
               const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Out,
               const float* __restrict__ ln_w, const float* __restrict__ ln_b,
               int M, int n_tiles, float eps, int flags)                     // flags bit 0: add the residual x (C.Transition); 0 -> T(x) only (PairTransition)
{
  extern __shared__ __align__(16) unsigned char smem_raw[];
  unsigned char* smem = reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + OFF_BAR);
  uint64_t* x_full = bars + 0; uint64_t* x_empty = bars + 2; uint64_t* w_full = bars + 4; uint64_t* w_empty = bars + 4 + NSLOT;
  const int tid = threadIdx.x;
  const int wg = __shfl_sync(0xffffffff, tid / 128, 0);
  const int lane = tid & 31;
  const int bid = blockIdx.x, nblk = gridDim.x;                                         // host launches nblk <= n_tiles
  const int n_iters = (bid < n_tiles) ? ((n_tiles - 1 - bid) / nblk + 1) : 0;           // tiles bid, bid + nblk, ...
  auto tile_of = [&](int i) { return bid + i * nblk; };

  if (tid == 0) {
    mbar_init(x_full + 0, 1); mbar_init(x_full + 1, 1); mbar_init(x_empty + 0, 4); mbar_init(x_empty + 1, 4);
    for (int s = 0; s < NSLOT; ++s) { mbar_init(w_full + s, 1); mbar_init(w_empty + s, 8); }          // empty: one arrive per consumer warp
    fence_barrier_init();
  }
  __syncthreads();

  if (wg == 0) {
    // ===================================================================================================== producer warpgroup
    setmaxnreg_dec<PROD_REGS>();
    const int pw = tid >> 5;
    if (pw == 0 && lane == 0) {
      // ---- weight ring: slot sequence (W1_j, W2_j, W3_j) x NCHUNK per tile, identical in every CTA
      uint32_t seq = 0;
      const uint32_t total = (uint32_t)n_iters * NSLOTS_PER_TILE;
      for (; seq < total; ++seq) {
        const int s = seq % NSLOT; const uint32_t u = seq / NSLOT;
        const int part = seq % 3; const int j = (seq / 3) % NCHUNK;
        if (u > 0) mbar_wait(w_empty + s, (u - 1) & 1);                   // all 8 consumer warps are done with the previous content of slot s
        mbar_expect_tx(w_full + s, SLOT_BYTES);
        unsigned char* dst = smem + OFF_RING + s * SLOT_BYTES;
        if (part < 2) {
          const int row0 = (part == 0 ? 0 : H_) + j * BH;                 // W12 rows: [0,H) = x1 (silu branch), [H,2H) = x2
          for (int ks = 0; ks < 4; ++ks) tma_load_2d(&tmW12, w_full + s, dst + ks * W1SLAB, ks * 64, row0);
        } else {
          tma_load_2d(&tmW3, w_full + s, dst, j * BH, 0);
        }
      }
    } else if ((pw == 1 || pw == 2) && lane == 0) {
      // ---- activation half-tiles for consumer WG (pw-1)
      const int c = pw - 1;
      for (int i = 0; i < n_iters; ++i) {
        if (i > 0) mbar_wait(x_empty + c, (i - 1) & 1);
        mbar_expect_tx(x_full + c, XH_BYTES);
        const int grow = tile_of(i) * BM + WGM * c;                       // rows >= M are zero-filled by TMA (never stored)
        for (int ks = 0; ks < 4; ++ks) tma_load_2d(&tmX, x_full + c, smem + OFF_X + ks * XSLAB + c * XHALF, ks * 64, grow);
      }
    }
  } else {
    // ===================================================================================================== consumer warpgroups
    setmaxnreg_inc<CONS_REGS>();
    const int c = wg - 1;
    const int ltid = tid - 128 * wg;
    const int wq = ltid >> 5;                 // warp within WG
    MmaG1 mma1; MmaG2 mma2;
    auto thr1 = mma1.get_thread_slice(ltid); auto thr2 = mma2.get_thread_slice(ltid);
    bfloat16_t* xs = reinterpret_cast<bfloat16_t*>(smem + OFF_X);
    Tensor sX  = make_tensor(make_smem_ptr(xs), SLayX{});
    Tensor sXc = local_tile(sX, Shape<Int<WGM>, Int<C_>>{}, make_coord(c, 0));           // this WG's 64 rows (x_hat after LN)
    Tensor tCrA = thr1.make_fragment_A(thr1.partition_A(sXc));                            // smem descriptors (SS A operand)
    auto sW1 = [&](int s) { return make_tensor(make_smem_ptr(reinterpret_cast<bfloat16_t*>(smem + OFF_RING + s * SLOT_BYTES)), SLayW1{}); };
    auto sW3 = [&](int s) { return make_tensor(make_smem_ptr(reinterpret_cast<bfloat16_t*>(smem + OFF_RING + s * SLOT_BYTES)), SLayW3{}); };
    Tensor sHf = make_tensor(make_smem_ptr(xs), LayHf{});
    Tensor hfrag = thr2.partition_fragment_A(sHf);                                         // ((2,2,2),1,BH/16) bf16 registers
    Tensor h32 = recast<uint32_t>(hfrag);
    Tensor acca = partition_fragment_C(mma1, Shape<Int<WGM>, Int<BH>>{});                 // BH/2 fp32
    Tensor accb = partition_fragment_C(mma1, Shape<Int<WGM>, Int<BH>>{});
    static_assert(decltype(size(hfrag))::value == decltype(size(acca))::value, "h fragment / accumulator size mismatch");
    static_assert(decltype(size(acca))::value == BH / 2, "accumulator size");
    // LN mapping: lane = column group g (cols 8g..8g+7 -> slab g/8, 16-B chunk g%8), rows r = wq + 4 i (i < 16) of this WG's half
    const int slab = lane >> 3, cidx = lane & 7;
    unsigned char* xh = smem + OFF_X + slab * XSLAB + c * XHALF;
    // epilogue mapping (m64n256 C fragment): row = 16 wq + lane/4 + 8 v1, col = 8 j + 2 (lane%4) + v0
    const int erow = 16 * wq + (lane >> 2), ecol = 2 * (lane & 3);
    auto release_slot = [&](int s) { if (lane == 0) mbar_arrive(w_empty + s); };            // per warp (count 8 per CTA), local only
    uint32_t wseq = 0;
    for (int i = 0; i < n_iters; ++i) {
      const long long row_base = (long long)tile_of(i) * BM + WGM * c;
      // ---------------------------------------------------------------- LayerNorm (in place, this WG's half)
      float w8[8], b8[8];
      { const float4* wp = reinterpret_cast<const float4*>(ln_w + 8 * lane); const float4* bp = reinterpret_cast<const float4*>(ln_b + 8 * lane);
        float4 t = __ldg(wp); w8[0] = t.x; w8[1] = t.y; w8[2] = t.z; w8[3] = t.w; t = __ldg(wp + 1); w8[4] = t.x; w8[5] = t.y; w8[6] = t.z; w8[7] = t.w;
        t = __ldg(bp); b8[0] = t.x; b8[1] = t.y; b8[2] = t.z; b8[3] = t.w; t = __ldg(bp + 1); b8[4] = t.x; b8[5] = t.y; b8[6] = t.z; b8[7] = t.w; }
      mbar_wait(x_full + c, i & 1);
      {
        uint4 xv[16];
        CUTE_UNROLL
        for (int ii = 0; ii < 16; ++ii) { const int r = wq + 4 * ii; xv[ii] = *reinterpret_cast<const uint4*>(xh + r * 128 + ((cidx ^ (r & 7)) << 4)); }
        float mean[16], rstd[16];
        CUTE_UNROLL
        for (int ii = 0; ii < 16; ++ii) {
          const float2 p0 = unpack_bf16x2(xv[ii].x), p1 = unpack_bf16x2(xv[ii].y), p2 = unpack_bf16x2(xv[ii].z), p3 = unpack_bf16x2(xv[ii].w);
          float s = ((p0.x + p0.y) + (p1.x + p1.y)) + ((p2.x + p2.y) + (p3.x + p3.y));
          s = warp_allsum(s);
          mean[ii] = s * (1.f / C_);
        }
        CUTE_UNROLL
        for (int ii = 0; ii < 16; ++ii) {
          const float mu = mean[ii];
          const float2 p0 = unpack_bf16x2(xv[ii].x), p1 = unpack_bf16x2(xv[ii].y), p2 = unpack_bf16x2(xv[ii].z), p3 = unpack_bf16x2(xv[ii].w);
          float d, q = 0.f;
          d = p0.x - mu; q += d * d; d = p0.y - mu; q += d * d; d = p1.x - mu; q += d * d; d = p1.y - mu; q += d * d;
          d = p2.x - mu; q += d * d; d = p2.y - mu; q += d * d; d = p3.x - mu; q += d * d; d = p3.y - mu; q += d * d;
          q = warp_allsum(q);
          rstd[ii] = rsqrtf(q * (1.f / C_) + eps);
        }
        CUTE_UNROLL
        for (int ii = 0; ii < 16; ++ii) {
          const int r = wq + 4 * ii; const float mu = mean[ii], rs = rstd[ii];
          const float2 p0 = unpack_bf16x2(xv[ii].x), p1 = unpack_bf16x2(xv[ii].y), p2 = unpack_bf16x2(xv[ii].z), p3 = unpack_bf16x2(xv[ii].w);
          uint4 o;
          o.x = pack_bf16x2((p0.x - mu) * rs * w8[0] + b8[0], (p0.y - mu) * rs * w8[1] + b8[1]);
          o.y = pack_bf16x2((p1.x - mu) * rs * w8[2] + b8[2], (p1.y - mu) * rs * w8[3] + b8[3]);
          o.z = pack_bf16x2((p2.x - mu) * rs * w8[4] + b8[4], (p2.y - mu) * rs * w8[5] + b8[5]);
          o.w = pack_bf16x2((p3.x - mu) * rs * w8[6] + b8[6], (p3.y - mu) * rs * w8[7] + b8[7]);
          *reinterpret_cast<uint4*>(xh + r * 128 + ((cidx ^ (r & 7)) << 4)) = o;
        }
      }
      fence_async_smem();                       // generic-proxy writes of x_hat -> visible to the async proxy (wgmma operand reads)
      named_bar_sync(1 + c, 128);
      // ---------------------------------------------------------------- hidden-chunk loop
      Tensor acc2 = partition_fragment_C(mma2, Shape<Int<WGM>, Int<C_>>{});              // 128 fp32
      clear(acc2);
      int s3_prev = -1;
      for (int j = 0; j < NCHUNK; ++j) {
        const uint32_t q0 = wseq; wseq += 3;
        const int sa = q0 % NSLOT, sb = (q0 + 1) % NSLOT, s3 = (q0 + 2) % NSLOT;
        mbar_wait(w_full + sa, (q0 / NSLOT) & 1);
        mbar_wait(w_full + sb, ((q0 + 1) / NSLOT) & 1);
        clear(acca); clear(accb);
        {
          Tensor tCrBa = thr1.make_fragment_B(thr1.partition_B(sW1(sa)));
          Tensor tCrBb = thr1.make_fragment_B(thr1.partition_B(sW1(sb)));
          warpgroup_fence_operand(acca); warpgroup_fence_operand(accb); warpgroup_fence_operand(acc2);
          warpgroup_arrive();
          gemm(mma1, tCrA, tCrBa, acca);
          warpgroup_commit_batch();
          gemm(mma1, tCrA, tCrBb, accb);
          warpgroup_commit_batch();
        }
        warpgroup_wait<1>();                    // chain a (and the previous chunk's W3 GEMM) retired; chain b still running
        warpgroup_fence_operand(acca); warpgroup_fence_operand(acc2);
        __syncwarp();
        if (s3_prev >= 0) release_slot(s3_prev);
        CUTE_UNROLL
        for (int e = 0; e < size(acca); ++e) acca(e) = silu_f(acca(e));
        warpgroup_wait<0>();                    // chain b retired
        warpgroup_fence_operand(accb);
        __syncwarp();
        release_slot(sa); release_slot(sb);
        if (j == NCHUNK - 1) { if (lane == 0) mbar_arrive(x_empty + c); }                 // every GEMM1 of this tile has retired: x half free
        CUTE_UNROLL
        for (int e2 = 0; e2 < size(h32); ++e2) h32(e2) = pack_bf16x2(acca(2 * e2) * accb(2 * e2), acca(2 * e2 + 1) * accb(2 * e2 + 1));
        mbar_wait(w_full + s3, ((q0 + 2) / NSLOT) & 1);
        {
          Tensor tCrB3 = thr2.make_fragment_B(thr2.partition_B(sW3(s3)));
          warpgroup_fence_operand(hfrag); warpgroup_fence_operand(acc2);
          warpgroup_arrive();
          gemm(mma2, hfrag, tCrB3, acc2);
          warpgroup_commit_batch();
        }
        s3_prev = s3;
      }
      // ---------------------------------------------------------------- epilogue: out = bf16(x + acc2)
      uint32_t res[2][32];
      const bool residual = (flags & 1) != 0;
      {
        CUTE_UNROLL
        for (int v1 = 0; v1 < 2; ++v1) {
          const long long grow = row_base + erow + 8 * v1;
          const bool ok = residual && grow < (long long)M;
          const __nv_bfloat16* xp = X + grow * C_ + ecol;
          CUTE_UNROLL
          for (int jj = 0; jj < 32; ++jj) res[v1][jj] = ok ? ldg_u32(xp + 8 * jj) : 0u;
        }
      }
      warpgroup_wait<0>();                      // the last W3 GEMM retired
      warpgroup_fence_operand(acc2);
      __syncwarp();
      release_slot(s3_prev);
      {
        CUTE_UNROLL
        for (int v1 = 0; v1 < 2; ++v1) {
          const long long grow = row_base + erow + 8 * v1;
          if (grow < (long long)M) {
            __nv_bfloat16* op = Out + grow * C_ + ecol;
            CUTE_UNROLL
            for (int jj = 0; jj < 32; ++jj) {
              const float2 r2 = unpack_bf16x2(res[v1][jj]);
              stg_u32(op + 8 * jj, pack_bf16x2(acc2(4 * jj + 2 * v1) + r2.x, acc2(4 * jj + 2 * v1 + 1) + r2.y));
            }
          }
        }
      }
    }
  }
}
