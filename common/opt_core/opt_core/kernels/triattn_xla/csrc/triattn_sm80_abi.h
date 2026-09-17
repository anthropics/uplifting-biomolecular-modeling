/* C ABI between the XLA-FFI launcher and libtriattn_sm80_xla.so (the sm_80 member of kernels/triattn/triattn_native's kernel family, package
 * generation 11, its cuda_80 kernel directory: bf16, head_dim 16 / 32 / 64, pair bias fp32 | bf16, optional key mask, optional per-row
 * log2-sum-exp output).  Same conventions as triattn_m1_abi.h: plain C, device pointers on the stream's device, every scratch buffer
 * allocated by the caller (XLA) with the sizes stated below, the caller's stream, error text returned in `err`. */
#ifndef TRIATTN_SM80_ABI_H_
#define TRIATTN_SM80_ABI_H_
#include <stdint.h>
#define TRIATTN_SM80_ABI_VERSION 1
#ifdef __cplusplus
extern "C" {
#endif
typedef struct TriattnSm80Call {
  int32_t abi_version;             /* = TRIATTN_SM80_ABI_VERSION */
  int32_t B, N, H, S, D;           /* q/k/v/out logical [B, N, H, S, D], D in {16, 32, 64} */
  int32_t bias_dtype;              /* 0 = fp32, 1 = bf16 : bias [B, H, S, S] contiguous */
  int32_t has_mask;                /* mask u8 [B, N, S] contiguous (nonzero = attend) or absent */
  int32_t want_lse;                /* lse fp32 [B, N, H, S] written when nonzero */
  float scale;
  int64_t sq[4], sk[4], sv[4], so[4];   /* element strides of q, k, v, out for dims (B, N, H, S); dim D has stride 1; multiples of 8 */
  const void *q, *k, *v, *bias, *mask;
  void* out;
  void* lse;
  /* scratch (device): */
  void* bias_staged;               /* fp32 [B, H, S128, S64], S128 = ceil(S/128)*128, S64 = ceil(S/64)*64 */
  void* fix;                       /* int32 [fix_elems], fix_elems >= triattn_sm80_xla_fix_elems(B, N, H, S, D) */
  int64_t fix_elems;
  void* census;                    /* int32 [4] */
  void* keyany;                    /* uint8 [B, S64]            (mask only) */
  void* rows;                      /* int32 [B*N, 4]           (mask only) */
  void* maskw;                     /* int32 [B*N*ceil(S/64)*2] (mask only) */
  void* rgflag;                    /* int32 [B*N]              (mask only; >= B*ceil(N/R) for every geometry) */
  void* stream;                    /* cudaStream_t */
  int32_t small_used;              /* out: 1 when the small-S geometry served the call */
  char err[512];                   /* out: error text when the call returns non-zero */
} TriattnSm80Call;
/* returns 0 on success; nonzero with err filled otherwise */
int triattn_sm80_xla_fwd(TriattnSm80Call* c);
typedef int (*triattn_sm80_fwd_fn)(TriattnSm80Call*);
int64_t triattn_sm80_xla_fix_elems(int B, int N, int H, int S, int D);   /* -1 when D is not built */
int triattn_sm80_xla_dims(int* out3);                                    /* writes the built head dims, returns their count */
int triattn_sm80_xla_geometry(int D, int small, int* out10);             /* (R, QG, STAGES, WARPS, BM, smem_hot, smem_safe, RW, small_max, small_max_rows); 0 on success */
const char* triattn_sm80_xla_describe(void);
#ifdef __cplusplus
}
#endif
#endif
