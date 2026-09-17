// Host-side launch of the sm_80 triangle-attention forward (triattn_sm80.cuh): grid order, shared-memory opt-in, the hot pass and the
// unconditional SAFE (fix-list) pass.  Torch-free: plain CUDA runtime; errors are returned as cudaError_t.
//
//   Args a = ...;                       // pointers, element strides, B/N/H/S, staged bias, mask tables (or nullptr), fix buffer, lse (or nullptr), c1, dbg
//   cudaError_t e = triattn_sm80::launch<triattn_sm80::Hot<32, /*general=*/true, /*lse=*/false>>(a, stream, &stage_name);
//
// The caller owns: the staged bias (stage_bias_kernel), the mask tables (mask_or_kernel + mask_rows_kernel) when a mask is given, an
// int32 fix buffer of fix_elems<T>(a) entries whose [FIX_TILES] census it keeps across calls, and the output / lse buffers.
#pragma once
#include "triattn_sm80.cuh"
#include <algorithm>

namespace triattn_sm80 {

// Default (large-S) geometry per head dim: WR x QG warps, each owning 32 queries of RW consecutive pair rows (R = WR*RW rows per CTA,
// BM = 32*QG queries), ST ring slots of one 32-key half tile each ({bias box BM x 32 fp32 | K x R | V x R}), filled ST-2 halves ahead:
//   D=32: 2 x 4 warps x 2 rows (R=4, 128 q), 3 slots x 32 KB = 96 KB;   D=16: 2 x 4 warps x 1 row (R=2, 128 q), 4 slots x 20 KB = 80 KB, two CTAs
//   per SM (16 warps at <= 128 registers; serves transposed views at S > SMAX16 -- measured against the two-row form);
//   D=64: 1 x 4 warps x 1 row (R=1, 128 q), 3 slots x 24 KB = 72 KB, TWO CTAs per SM -- used only for transposed (long position stride)
//         operands at S > TS_SMAX64; everything else runs GeoS<64> below (2 x 2 warps, R=2 x 64 q, also two 4-warp CTAs per SM: measured
//         8-10 % faster than one 8-warp CTA of either tile shape at every N on A100-SXM4-80GB; see the maintainers' notes).
// Override at build time with -DTS_WR<D>= / -DTS_RW<D>= / -DTS_QG<D>= / -DTS_ST<D>= / -DTS_MINB<D>= (development A/B only).
#ifndef TS_WR32
#define TS_WR32 2
#endif
#ifndef TS_RW32
#define TS_RW32 2
#endif
#ifndef TS_QG32
#define TS_QG32 4
#endif
#ifndef TS_ST32
#define TS_ST32 3
#endif
#ifndef TS_MINB32
#define TS_MINB32 1
#endif
#ifndef TS_WR16
#define TS_WR16 2
#endif
#ifndef TS_RW16
#define TS_RW16 1
#endif
#ifndef TS_QG16
#define TS_QG16 4
#endif
#ifndef TS_ST16
#define TS_ST16 4
#endif
#ifndef TS_MINB16
#define TS_MINB16 2
#endif
#ifndef TS_WR64
#define TS_WR64 1
#endif
#ifndef TS_RW64
#define TS_RW64 1
#endif
#ifndef TS_QG64
#define TS_QG64 4
#endif
#ifndef TS_ST64
#define TS_ST64 3
#endif
#ifndef TS_MINB64
#define TS_MINB64 2
#endif

template <int D> struct Geo;
template <> struct Geo<32> { static constexpr int WR = TS_WR32, RW = TS_RW32, QG = TS_QG32, ST = TS_ST32, MINB = TS_MINB32; };
template <> struct Geo<16> { static constexpr int WR = TS_WR16, RW = TS_RW16, QG = TS_QG16, ST = TS_ST16, MINB = TS_MINB16; };
template <> struct Geo<64> { static constexpr int WR = TS_WR64, RW = TS_RW64, QG = TS_QG64, ST = TS_ST64, MINB = TS_MINB64; };

// Small-S geometry: 64-query CTA tiles (QG = 2) cut the query padding of short sequences (S = 400: 448 instead of 512 rows computed) and
// give more, smaller CTAs (D=32/16: 4-warp CTAs, two per SM).  fwd uses it when S_q <= SMAX for any operand layout, and up to SMAX_ROWS
// when q/k/v rows are densely strided (position stride <= 512 elements: contiguous tensors and projection views; transposed views at large
// S keep the 128-query tiles, whose half as many CTAs walk the long position stride).  Measured crossovers; 0 disables.
#ifndef TS_SMAX32
#define TS_SMAX32 832
#endif
#ifndef TS_SMAXROWS32
#define TS_SMAXROWS32 1280
#endif
#ifndef TS_SMAX16
#define TS_SMAX16 1664
#endif
#ifndef TS_SMAXROWS16
#define TS_SMAXROWS16 (1 << 30)
#endif
#ifndef TS_SMAX64
#define TS_SMAX64 1408
#endif
#ifndef TS_SMAXROWS64
#define TS_SMAXROWS64 (1 << 30)
#endif
template <int D> struct GeoS;
template <> struct GeoS<32> { static constexpr int WR = 2, RW = 2, QG = 2, ST = 3, MINB = 2, SMAX = TS_SMAX32, SMAX_ROWS = TS_SMAXROWS32; };
template <> struct GeoS<16> { static constexpr int WR = 2, RW = 2, QG = 2, ST = 4, MINB = 2, SMAX = TS_SMAX16, SMAX_ROWS = TS_SMAXROWS16; };
// D=64 "small" geometry = the general one: 2 x 2 warps x 1 row (R=2 rows x 64 q), two CTAs per SM; fwd takes it for any layout up to
// S = TS_SMAX64 and for densely strided rows at every S (TS_SMAXROWS64).  Overridable for A/B work (-DTS_S_WR64= / -DTS_S_QG64= /
// -DTS_S_ST64= / -DTS_S_MINB64=).
#ifndef TS_S_WR64
#define TS_S_WR64 2
#endif
#ifndef TS_S_QG64
#define TS_S_QG64 2
#endif
#ifndef TS_S_ST64
#define TS_S_ST64 3
#endif
#ifndef TS_S_MINB64
#define TS_S_MINB64 2
#endif
template <> struct GeoS<64> { static constexpr int WR = TS_S_WR64, RW = 1, QG = TS_S_QG64, ST = TS_S_ST64, MINB = TS_S_MINB64, SMAX = TS_SMAX64, SMAX_ROWS = TS_SMAXROWS64; };

// The hot instantiations for head dim D (large-S and small-S geometry) and their SAFE partners (same geometry, general row kinds, same lse flag).
template <int D, bool GENERAL, bool LSE> using Hot   = Traits<D, Geo<D>::WR, Geo<D>::RW, Geo<D>::QG, Geo<D>::ST, false, GENERAL, LSE, Geo<D>::MINB>;
template <int D, bool GENERAL, bool LSE> using HotS  = Traits<D, GeoS<D>::WR, GeoS<D>::RW, GeoS<D>::QG, GeoS<D>::ST, false, GENERAL, LSE, GeoS<D>::MINB>;
template <int D, bool LSE>               using Safe  = Traits<D, Geo<D>::WR, Geo<D>::RW, Geo<D>::QG, Geo<D>::ST, true, true, LSE, 1>;
template <class T> using SafeOf = Traits<T::D, T::WR, T::RW, T::QG, T::STAGES, true, true, T::LSE, 1>;

template <class T> inline int q_tiles(const Args& a) { return (a.S_q + T::BM - 1) / T::BM; }
template <class T> inline int row_groups(const Args& a) { return (a.N + T::R - 1) / T::R; }
template <class T> inline long long n_cta_tiles(const Args& a) { return (long long)q_tiles<T>(a) * row_groups<T>(a) * a.B * a.H; }
template <class T> inline long long fix_elems(const Args& a) { return FIX_LIST + 3 * n_cta_tiles<T>(a); }   // int32 entries the fix buffer must hold
template <class T> constexpr int smem_bytes() { return T::SMEM_BYTES; }

// q-tile block of the CTA order: the staged-bias rows of qb q tiles (qb x BM x S64 x 4 B) stay within ~16 MB of L2 (PORTING.md 7).
#ifndef TS_QB_MB
#define TS_QB_MB 16          // staged-bias bytes per q-tile block of the CTA order (MB); development A/B only
#endif
template <class T> inline int qtile_block(const Args& a) {
  const long long per_qtile = (long long)T::BM * a.S64 * 4;
  return (int)std::max<long long>(1, std::min<long long>(q_tiles<T>(a), ((long long)TS_QB_MB << 20) / per_qtile));
}

namespace detail {
template <class T, bool kFix> inline cudaError_t opt_in_smem() {   // once per (instantiation, device): dynamic shared memory above 48 KB
  static bool done[64] = {};
  int dev = 0; cudaError_t e = cudaGetDevice(&dev); if (e != cudaSuccess) return e;
  if (dev >= 0 && dev < 64 && done[dev]) return cudaSuccess;
  const void* fn;
  if constexpr (kFix) fn = reinterpret_cast<const void*>(&fix_kernel<T>); else fn = reinterpret_cast<const void*>(&fwd_kernel<T>);   // (only the launched kernels get instantiated)
  e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM_BYTES);
  if (e == cudaSuccess && dev >= 0 && dev < 64) done[dev] = true;
  return e;
}
inline int num_sms() {
  static int n[64] = {};
  int dev = 0; if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return 108;
  if (n[dev] == 0 && cudaDeviceGetAttribute(&n[dev], cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) n[dev] = 108;
  return n[dev];
}
}  // namespace detail

// launch<T>: fills the derived fields of a copy of `a` (n_ktiles, S128, S64, qb), clears this call's fix count, launches the hot pass
// over every CTA tile and then, unconditionally, the SAFE pass over the fix list (empty list = immediate exit; graph-capturable).
template <class T>
inline cudaError_t launch(const Args& a_in, cudaStream_t st, const char** stage = nullptr) {
  static_assert(!T::SAFE, "launch<> takes the hot instantiation; its SAFE partner is derived");
  using S = SafeOf<T>;
  Args a = a_in;
  a.n_ktiles = (a.S_kv + BN - 1) / BN;
  a.S64 = (a.S_kv + 63) / 64 * 64; a.S128 = (a.S_q + 127) / 128 * 128;
  if (a.qb <= 0) a.qb = qtile_block<T>(a);
  cudaError_t e;
  auto fail = [&](const char* w, cudaError_t err) { if (stage) *stage = w; return err; };
  if ((e = detail::opt_in_smem<T, false>()) != cudaSuccess) return fail("cudaFuncSetAttribute(hot)", e);
  if ((e = detail::opt_in_smem<S, true>()) != cudaSuccess) return fail("cudaFuncSetAttribute(safe)", e);
  if ((e = cudaMemsetAsync(a.fix + FIX_COUNT, 0, sizeof(int), st)) != cudaSuccess) return fail("memset(fix count)", e);   // this call's list; fix[FIX_TILES] (census) is the caller's
  const dim3 grid(q_tiles<T>(a), row_groups<T>(a), a.B * a.H), block(T::THREADS);
  fwd_kernel<T><<<grid, block, T::SMEM_BYTES, st>>>(a);
  if ((e = cudaGetLastError()) != cudaSuccess) return fail("hot pass launch", e);
  const long long nct = n_cta_tiles<T>(a);
  const int npers = (int)std::min<long long>(nct, detail::num_sms());
  fix_kernel<S><<<npers, dim3(S::THREADS), S::SMEM_BYTES, st>>>(a);
  if ((e = cudaGetLastError()) != cudaSuccess) return fail("safe pass launch", e);
  return cudaSuccess;
}

}  // namespace triattn_sm80
