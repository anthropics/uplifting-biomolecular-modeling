// SPDX-License-Identifier: Apache-2.0
// tmn_sm90_unit.cuh — the sm_90a instantiation list of the trimul_native kernel family (tmn_kernels.cuh).  One compilation unit per
// (C_Z, C_H) pair includes this file with TMN_CZ / TMN_CH defined (csrc/tmn90_z<C_Z>_h<C_H>.cu -> build/sm_90a/tmn90_z<C_Z>_h<C_H>.cubin);
// every kernel is an extern "C" symbol whose name encodes its configuration (kernel.py selects by name = the tile table's compiled support):
//   tmn_k1_z<C_Z>_h<C_H>_<b|f>_t<BI>x<BJ>_s<NSLOT>k<SKCH>_m<mask 0|1>_l<LNM>_v<save 0|1>
//   tmn_k3_z<C_Z>_h<C_H>_<b|f>_t<BI>x<BJ>_s<NSLOT>a<NACC>_l<LNM>
// (b = bf16 z, f = fp32-resident z; LNM 1 = the shared numerics statement (csrc/common/tmn_math.cuh), 2/3 = the reference library's summation trees
// (bitwise class), 4 = trimul_tx 1.2's arithmetic (comparison only, c256)).
#pragma once
#include "tmn_kernels.cuh"

#define TMN_K1(CZ, CH, ZF, ZC, BI, BJ, NS, SK, M, L, V)                                                                              \
  extern "C" __global__ void __launch_bounds__(tmn::NTHREADS, 1)                                                                 \
  tmn_k1_z##CZ##_h##CH##_##ZC##_t##BI##x##BJ##_s##NS##k##SK##_m##M##_l##L##_v##V(const __grid_constant__ tmn::K1Params p) {     \
    tmn::sm90::k1_body<tmn::K1Cfg<CZ, CH, ZF, BI, BJ, NS, SK>, (M) != 0, L, (V) != 0>(p);                                        \
  }
// fp32 z, also emitting bf16(LN_in(z)) rows for a pre-normalised K3 (name field x1)
#define TMN_K1X(CZ, CH, BI, BJ, NS, SK, M)                                                                                           \
  extern "C" __global__ void __launch_bounds__(tmn::NTHREADS, 1)                                                                 \
  tmn_k1_z##CZ##_h##CH##_f_t##BI##x##BJ##_s##NS##k##SK##_m##M##_l1_v0_x1(const __grid_constant__ tmn::K1Params p) {            \
    tmn::sm90::k1_body<tmn::K1Cfg<CZ, CH, true, BI, BJ, NS, SK>, (M) != 0, 1, false, true>(p);                                    \
  }
#define TMN_MODE_b 0
#define TMN_MODE_f 1
#define TMN_MODE_p 2
#define TMN_MODE_c 3
#define TMN_K3(CZ, CH, ZF, ZC, BI, BJ, NS, NA, L)                                                                                    \
  extern "C" __global__ void __launch_bounds__(tmn::NTHREADS, 1)                                                                 \
  tmn_k3_z##CZ##_h##CH##_##ZC##_t##BI##x##BJ##_s##NS##a##NA##_l##L(const __grid_constant__ tmn::K3Params p) {                    \
    tmn::sm90::k3_body<tmn::K3Cfg<CZ, CH, TMN_MODE_##ZC, BI, BJ, NS, NA>, L>(p);                                                   \
  }

// K1: both mask variants, fast LN (l1) always; reference-order LN (l2) for bf16; save-stats variant (v1) for the fast bf16 kernel.
#define TMN_K1_SET_BF16(CZ, CH, BI, BJ, NS, SK)  \
  TMN_K1(CZ, CH, false, b, BI, BJ, NS, SK, 0, 1, 0) TMN_K1(CZ, CH, false, b, BI, BJ, NS, SK, 1, 1, 0) \
  TMN_K1(CZ, CH, false, b, BI, BJ, NS, SK, 0, 2, 0) TMN_K1(CZ, CH, false, b, BI, BJ, NS, SK, 1, 2, 0) \
  TMN_K1(CZ, CH, false, b, BI, BJ, NS, SK, 1, 1, 1)
#define TMN_K1_SET_F32(CZ, CH, BI, BJ, NS, SK)  \
  TMN_K1(CZ, CH, true, f, BI, BJ, NS, SK, 0, 1, 0) TMN_K1(CZ, CH, true, f, BI, BJ, NS, SK, 1, 1, 0)
// K3: fast LN (l1) + the two reference-order trees (l2: N % 4 == 0 planes, l3: other N) for bf16; l1 for fp32 z.
#define TMN_K3_SET_BF16(CZ, CH, BI, BJ, NS, NA)  \
  TMN_K3(CZ, CH, false, b, BI, BJ, NS, NA, 1) TMN_K3(CZ, CH, false, b, BI, BJ, NS, NA, 2) TMN_K3(CZ, CH, false, b, BI, BJ, NS, NA, 3)
#define TMN_K3_SET_F32(CZ, CH, BI, BJ, NS, NA)  \
  TMN_K3(CZ, CH, true, f, BI, BJ, NS, NA, 1)

#define TMN_K3_X(CZ, CH, ZF, ZC, BI, BJ, NS, NA, L) TMN_K3(CZ, CH, ZF, ZC, BI, BJ, NS, NA, L)   // expands TMN_CZ / TMN_CH before pasting
#if defined(TMN_DEV_K3ONLY)
TMN_K3_X(TMN_CZ, TMN_CH, false, b, 2, 64, 4, 1, 1) // dev: K3 (2,64) only, both accumulator schedules, 4- and 5-slot rings
TMN_K3_X(TMN_CZ, TMN_CH, false, b, 2, 64, 4, 2, 1)
#elif TMN_CZ == 256 && TMN_CH == 256
TMN_K1_SET_BF16(256, 256, 2, 64, 8, 2)          // 64 KB z tile + 8 x 16 KB weight ring (= trimul_tx 1.2's configuration)
TMN_K1(256, 256, false, b, 2, 64, 8, 2, 0, 4, 0) TMN_K1(256, 256, false, b, 2, 64, 8, 2, 1, 4, 0)   // l4: trimul_tx 1.2's arithmetic (byte comparison only)
TMN_K1_SET_BF16(256, 256, 1, 128, 8, 2)
TMN_K1_SET_BF16(256, 256, 2, 64, 4, 4)          // 4 x 32 KB ring slots (4 k-chunks each): the tuned default (tile-table data, large-N sweep)
TMN_K1_SET_F32(256, 256, 2, 64, 4, 2)           // 128 KB fp32 z tile + 4 x 16 KB ring
TMN_K1X(256, 256, 2, 64, 4, 2, 0) TMN_K1X(256, 256, 2, 64, 4, 2, 1)   // + bf16(LN_in(z)) rows for the pre-normalised K3 (mode p)
TMN_K3_SET_BF16(256, 256, 2, 64, 4, 1)          // X 64 KB + z 64 KB + 4 x 16 KB weight ring + 8 KB staging
TMN_K3(256, 256, false, b, 2, 64, 4, 1, 4)         // l4: trimul_tx 1.2's arithmetic
TMN_K3_SET_BF16(256, 256, 1, 128, 4, 1)
TMN_K3_SET_BF16(256, 256, 1, 64, 4, 1)          // split-N: 64-token tile, warpgroups alternate output blocks
TMN_K3_SET_F32(256, 256, 1, 64, 4, 1)           // fp32 z: split-N (the 128-token fp32 tile does not fit)
TMN_K3(256, 256, false, p, 2, 64, 4, 1, 1)         // mode p: pre-normalised bf16 operand tile, fp32 residual from global, fp32 out (the fp32-resident form at 256)
TMN_K3(256, 256, false, c, 2, 64, 4, 1, 1) TMN_K3(256, 256, false, c, 2, 64, 4, 1, 2)   // mode c: bf16 cast tile (LN_in here), fp32 residual from global, fp32 out; l1 and reference-order l2
#elif TMN_CZ == 128 && TMN_CH == 128
TMN_K1_SET_BF16(128, 128, 2, 64, 8, 2)          // 8 x 16 KB = the whole [512 x 128] weight stream: resident (loaded once per CTA)
TMN_K1_SET_BF16(128, 128, 1, 128, 8, 2)
TMN_K1_SET_BF16(128, 128, 2, 64, 4, 2)          // streamed ring variant (tuning comparison)
TMN_K1_SET_F32(128, 128, 2, 64, 8, 2)
TMN_K3_SET_BF16(128, 128, 2, 64, 8, 1)          // 8 x 8 KB = both weight matrices resident
TMN_K3_SET_BF16(128, 128, 2, 64, 8, 2)          // two accumulator sets (tile-table data: N <= 1536)
TMN_K3_SET_BF16(128, 128, 1, 128, 8, 1)
TMN_K3_SET_BF16(128, 128, 2, 64, 4, 1)          // streamed ring variant
TMN_K3_SET_BF16(128, 128, 1, 64, 8, 1)          // split-N
TMN_K3_SET_F32(128, 128, 2, 64, 8, 1)
TMN_K3_SET_F32(128, 128, 1, 64, 8, 1)
#elif TMN_CZ == 64 && TMN_CH == 64
TMN_K1_SET_BF16(64, 64, 2, 64, 4, 1)            // 4 blocks x 1 slot of 8 KB: resident
TMN_K1_SET_BF16(64, 64, 1, 128, 4, 1)
TMN_K1_SET_F32(64, 64, 2, 64, 4, 1)
TMN_K3_SET_BF16(64, 64, 2, 64, 4, 1)            // resident (2 blocks x (gate, projection) x 4 KB)
TMN_K3_SET_BF16(64, 64, 1, 128, 4, 1)
TMN_K3_SET_F32(64, 64, 2, 64, 4, 1)
#elif TMN_CZ == 64 && TMN_CH == 128
TMN_K1_SET_BF16(64, 128, 2, 64, 8, 1)           // 8 blocks x 1 slot: resident
TMN_K1_SET_BF16(64, 128, 1, 128, 8, 1)
TMN_K1_SET_F32(64, 128, 2, 64, 8, 1)
TMN_K3_SET_BF16(64, 128, 2, 64, 4, 1)           // resident (slot = 8 KB projection rows | 4 KB gate rows)
TMN_K3_SET_BF16(64, 128, 1, 128, 4, 1)
TMN_K3_SET_F32(64, 128, 2, 64, 4, 1)
#elif TMN_CZ == 384 && TMN_CH == 384
TMN_K1(384, 384, false, b, 2, 64, 6, 2, 1, 1, 0) // 96 KB z tile + 6 x 16 KB ring (two 3-slot blocks in flight); K3 at 384 needs the smem-operand (SS) mainloop: not instantiated here
TMN_K1(384, 384, false, b, 2, 64, 6, 2, 0, 1, 0)
TMN_K1(384, 384, false, b, 2, 64, 6, 2, 0, 2, 0) TMN_K1(384, 384, false, b, 2, 64, 6, 2, 1, 2, 0)   // l2: reference-order LN_in (default numerics)
#else
#error "no instantiation list for this (TMN_CZ, TMN_CH)"
#endif

// layout probe for the host-side packer: sizeof / offsetof of the parameter blocks
extern "C" __global__ void tmn_info(int* out) {
  if (threadIdx.x != 0) return;
  out[0] = (int)sizeof(tmn::K1Params); out[1] = (int)offsetof(tmn::K1Params, tm_w); out[2] = (int)offsetof(tmn::K1Params, mask); out[3] = (int)offsetof(tmn::K1Params, N); out[4] = (int)offsetof(tmn::K1Params, eps);
  out[5] = (int)sizeof(tmn::K3Params); out[6] = (int)offsetof(tmn::K3Params, zres); out[7] = (int)offsetof(tmn::K3Params, gamma_in); out[8] = (int)offsetof(tmn::K3Params, N); out[9] = (int)offsetof(tmn::K3Params, eps);
}
