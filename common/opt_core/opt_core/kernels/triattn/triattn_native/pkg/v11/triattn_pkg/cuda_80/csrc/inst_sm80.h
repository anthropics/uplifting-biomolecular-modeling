// Per-head-dim instantiation table: inst_d<D>.cu each define `inst_d<D>` (compiled as separate translation units, in parallel).
#pragma once
#include "launch_sm80.cuh"

namespace triattn_sm80 {
struct GeoInfo { int R, RW, QG, STAGES, smem_hot, smem_safe; };
struct Inst {
  int D;
  GeoInfo big, small;      // large-S geometry; small-S geometry (64-query tiles), see small_max
  int small_max, small_max_rows;   // small-S geometry when S_q <= small_max (any layout) or S_q <= small_max_rows (densely strided rows)
  // variant: 0 = mask-free hot instantiation, 1 = general (row kinds / mask words), 2 = general + lse output; each followed by its SAFE pass
  cudaError_t (*run)(const Args&, int variant, bool small, cudaStream_t, const char** stage);
};
#define TS_GEOINFO(HOT) GeoInfo{HOT<DIM_, true, false>::R, HOT<DIM_, true, false>::RW, HOT<DIM_, true, false>::QG, HOT<DIM_, true, false>::STAGES, \
                                HOT<DIM_, true, false>::SMEM_BYTES, SafeOf<HOT<DIM_, true, false>>::SMEM_BYTES}
#define TS_DEFINE_INST(DIM)                                                                                              \
  static cudaError_t run_d##DIM(const Args& a, int variant, bool small, cudaStream_t st, const char** stage) {          \
    if (small) switch (variant) {                                                                                        \
      case 0: return launch<HotS<DIM, false, false>>(a, st, stage);                                                     \
      case 1: return launch<HotS<DIM, true, false>>(a, st, stage);                                                      \
      default: return launch<HotS<DIM, true, true>>(a, st, stage);                                                      \
    }                                                                                                                    \
    switch (variant) {                                                                                                   \
      case 0: return launch<Hot<DIM, false, false>>(a, st, stage);                                                      \
      case 1: return launch<Hot<DIM, true, false>>(a, st, stage);                                                       \
      default: return launch<Hot<DIM, true, true>>(a, st, stage);                                                       \
    }                                                                                                                    \
  }                                                                                                                      \
  namespace { constexpr int DIM_ = DIM; }                                                                                \
  const Inst inst_d##DIM = {DIM, TS_GEOINFO(Hot), TS_GEOINFO(HotS), GeoS<DIM>::SMAX, GeoS<DIM>::SMAX_ROWS, &run_d##DIM};
extern const Inst inst_d16, inst_d32, inst_d64;
}  // namespace triattn_sm80
