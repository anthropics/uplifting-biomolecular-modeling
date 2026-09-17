// SPDX-License-Identifier: Apache-2.0
// build: archs=sm_90a
// build: flags=-Xptxas --warn-on-spills
// build: variant=tmn90w_z384_h384 defines=TMN_CZ=384,TMN_CH=384
// build: variant=tmn90w_z256_h256 defines=TMN_CZ=256,TMN_CH=256
// build: variant=tmn90w_z128_h384 defines=TMN_CZ=128,TMN_CH=384
// build: variant=tmn90w_z384_h128 defines=TMN_CZ=384,TMN_CH=128
// build: variant=tmn90w_z256_h384 defines=TMN_CZ=256,TMN_CH=384
// build: variant=tmn90w_z384_h256 defines=TMN_CZ=384,TMN_CH=256
// build: variant=tmn90w_z64_h384 defines=TMN_CZ=64,TMN_CH=384
// build: variant=tmn90w_z384_h64 defines=TMN_CZ=384,TMN_CH=64
// tmn_sm90_wide.cu — sm_90a units of the wide-K member (tmn_k3_wide.cuh): one cubin per (C_Z, C_H) pair (build/sm_90a/tmn90w_z<C_Z>_h<C_H>.cubin)
// holding that pair's K1 instantiations (the shared K1 template of tmn_kernels.cuh) and its wide K3 instantiations:
//   tmn_k1_z<C_Z>_h<C_H>_<b|f>_t<BI>x<BJ>_s<NSLOT>k<SKCH>_m<mask 0|1>_l<LNM>_v<save 0|1>     (same names as the per-width units)
//   tmn_k3w_z<C_Z>_h<C_H>_<b|f>_t<BI>x<BJ>_s<NSLOT>a<NACC>_l<LNM>                          (LNM 1 own order | 2, 3 reference-library trees)
#include "tmn_k3_wide.cuh"

#define TMN_K1(CZ, CH, ZF, ZC, BI, BJ, NS, SK, M, L, V)                                                                              \
  extern "C" __global__ void __launch_bounds__(tmn::NTHREADS, 1)                                                                 \
  tmn_k1_z##CZ##_h##CH##_##ZC##_t##BI##x##BJ##_s##NS##k##SK##_m##M##_l##L##_v##V(const __grid_constant__ tmn::K1Params p) {     \
    tmn::sm90::k1_body<tmn::K1Cfg<CZ, CH, ZF, BI, BJ, NS, SK>, (M) != 0, L, (V) != 0>(p);                                        \
  }
#define TMN_K3W(CZ, CH, ZF, ZC, BI, BJ, NS, NA, L)                                                                                   \
  extern "C" __global__ void __launch_bounds__(tmn::NTHREADS, 1)                                                                 \
  tmn_k3w_z##CZ##_h##CH##_##ZC##_t##BI##x##BJ##_s##NS##a##NA##_l##L(const __grid_constant__ tmn::K3Params p) {                   \
    tmn::sm90::k3w_body<tmn::K3WCfg<CZ, CH, ZF, BI, BJ, NS, NA>, L>(p);                                                            \
  }
// K1 set: both mask variants, own-order LN; K3 wide set: own-order LN, one and two accumulator schedules
#define TMN_K1_PAIR(CZ, CH, BI, BJ, NS, SK)  TMN_K1(CZ, CH, false, b, BI, BJ, NS, SK, 0, 1, 0) TMN_K1(CZ, CH, false, b, BI, BJ, NS, SK, 1, 1, 0)
#define TMN_K1_PAIR_L2(CZ, CH, BI, BJ, NS, SK)  TMN_K1(CZ, CH, false, b, BI, BJ, NS, SK, 0, 2, 0) TMN_K1(CZ, CH, false, b, BI, BJ, NS, SK, 1, 2, 0)
#define TMN_K3W_BF16(CZ, CH, BI, BJ, NS, NA)  TMN_K3W(CZ, CH, false, b, BI, BJ, NS, NA, 1)
#define TMN_K3W_EXACT(CZ, CH, BI, BJ, NS, NA)  TMN_K3W(CZ, CH, false, b, BI, BJ, NS, NA, 2) TMN_K3W(CZ, CH, false, b, BI, BJ, NS, NA, 3)
#define TMN_K3W_F32(CZ, CH, BI, BJ, NS, NA)  TMN_K3W(CZ, CH, true, f, BI, BJ, NS, NA, 1)

#if TMN_CZ == 384 && TMN_CH == 384
TMN_K1_PAIR(384, 384, 2, 64, 6, 2)              // 96 KB z tile + 6 x 16 KB ring (two 3-slot blocks in flight)
TMN_K1_PAIR_L2(384, 384, 2, 64, 6, 2)           // reference-order LN_in (bitwise class)
TMN_K1(384, 384, false, b, 1, 128, 6, 2, 1, 1, 0)   // 1 x 128 tile (transposed / incoming reads at large N)
TMN_K1(384, 384, false, b, 2, 64, 4, 3, 1, 1, 0)    // 4 x 24 KB slots = the same two blocks in flight, coarser ring steps
TMN_K3W_BF16(384, 384, 1, 64, 4, 1)             // split-N: X 48 KB + z 48 KB + 4 x 24 KB slots + staging = 214 KB
TMN_K3W_EXACT(384, 384, 1, 64, 4, 1)
#elif TMN_CZ == 256 && TMN_CH == 256
TMN_K3W_BF16(256, 256, 2, 64, 4, 1)             // full 128-token tile, projection operand from shared memory (vs the register-resident K3 of tmn_kernels.cuh)
TMN_K3W_BF16(256, 256, 2, 64, 4, 2)
TMN_K3W_BF16(256, 256, 1, 128, 4, 1)
TMN_K3W_EXACT(256, 256, 2, 64, 4, 1)
TMN_K3W_F32(256, 256, 1, 64, 4, 1)              // fp32-resident z: split-N (z tile 64 KB fp32)
#elif TMN_CZ == 128 && TMN_CH == 384
TMN_K1_PAIR(128, 384, 2, 64, 8, 2)              // 24 weight blocks of 16 KB through an 8-slot ring
TMN_K1_PAIR_L2(128, 384, 2, 64, 8, 2)
TMN_K3W_BF16(128, 384, 2, 64, 4, 1)             // X 96 KB + z 32 KB + 2 x (24 + 8) KB
TMN_K3W_BF16(128, 384, 2, 64, 4, 2)
#elif TMN_CZ == 384 && TMN_CH == 128
TMN_K1_PAIR(384, 128, 2, 64, 6, 2)
TMN_K1_PAIR_L2(384, 128, 2, 64, 6, 2)
TMN_K3W_BF16(384, 128, 2, 64, 4, 1)             // X 32 KB + z 96 KB + 2 x (8 + 24) KB
TMN_K3W_BF16(384, 128, 2, 64, 4, 2)
#elif TMN_CZ == 256 && TMN_CH == 384
TMN_K1_PAIR(256, 384, 2, 64, 8, 2)
TMN_K1_PAIR_L2(256, 384, 2, 64, 8, 2)
TMN_K3W_BF16(256, 384, 1, 64, 4, 1)             // split-N: X 48 KB + z 32 KB + 2 x (24 + 16) KB
#elif TMN_CZ == 384 && TMN_CH == 256
TMN_K1_PAIR(384, 256, 2, 64, 6, 2)
TMN_K1_PAIR_L2(384, 256, 2, 64, 6, 2)
TMN_K3W_BF16(384, 256, 1, 64, 4, 1)             // split-N: X 32 KB + z 48 KB + 2 x (16 + 24) KB
#elif TMN_CZ == 64 && TMN_CH == 384
TMN_K1_PAIR(64, 384, 2, 64, 8, 1)               // 24 blocks x 8 KB through an 8-slot ring
TMN_K1_PAIR_L2(64, 384, 2, 64, 8, 1)
TMN_K3W_BF16(64, 384, 2, 64, 4, 1)              // resident weights: 2 blocks x (24 + 4) KB; X 96 KB + z 16 KB
TMN_K3W_BF16(64, 384, 2, 64, 4, 2)
#elif TMN_CZ == 384 && TMN_CH == 64
TMN_K1_PAIR(384, 64, 2, 64, 6, 2)
TMN_K1_PAIR_L2(384, 64, 2, 64, 6, 2)
TMN_K3W_BF16(384, 64, 2, 64, 4, 1)              // X 16 KB + z 96 KB + 2 x (4 + 24) KB
TMN_K3W_BF16(384, 64, 2, 64, 4, 2)
#else
#error "no instantiation list for this (TMN_CZ, TMN_CH)"
#endif

// layout probe for the host-side packer: sizeof / offsetof of the parameter blocks (same as the per-width units)
extern "C" __global__ void tmn_info(int* out) {
  if (threadIdx.x != 0) return;
  out[0] = (int)sizeof(tmn::K1Params); out[1] = (int)offsetof(tmn::K1Params, tm_w); out[2] = (int)offsetof(tmn::K1Params, mask); out[3] = (int)offsetof(tmn::K1Params, N); out[4] = (int)offsetof(tmn::K1Params, eps);
  out[5] = (int)sizeof(tmn::K3Params); out[6] = (int)offsetof(tmn::K3Params, zres); out[7] = (int)offsetof(tmn::K3Params, gamma_in); out[8] = (int)offsetof(tmn::K3Params, N); out[9] = (int)offsetof(tmn::K3Params, eps);
}
