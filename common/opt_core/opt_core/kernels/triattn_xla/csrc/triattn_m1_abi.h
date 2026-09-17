/* C ABI between the XLA-FFI launcher and libtriattn_m1_xla.so (the sm_90a "M1" triangle-attention forward of kernels/triattn/triattn_native,
 * generation 11 (the C entry is unchanged since generation 10), cuda_b kernel: bf16, head_dim 32, pair bias + optional key mask).  Plain C so the launcher (g++) and the library (nvcc) agree
 * without sharing C++ headers.  All pointers are device pointers on the stream's device; scratch buffers are allocated by XLA (sizes below). */
#ifndef TRIATTN_M1_ABI_H_
#define TRIATTN_M1_ABI_H_
#include <stdint.h>
#define TRIATTN_M1_ABI_VERSION 1
#ifdef __cplusplus
extern "C" {
#endif
typedef struct TriattnM1Call {
  int32_t abi_version;             /* = TRIATTN_M1_ABI_VERSION */
  int32_t B, N, H, S;              /* q/k/v/out logical [B, N, H, S, 32] */
  int32_t bias_dtype;              /* 0 = fp32, 1 = bf16 : bias [B, H, S, S] contiguous */
  int32_t has_mask;                /* mask u8 [B, N, S] contiguous (nonzero = attend) or absent */
  float scale;
  int64_t sq[4], sk[4], sv[4], so[4];   /* element strides of q, k, v, out for dims (B, N, H, S); dim D has stride 1 */
  const void *q, *k, *v, *bias, *mask;
  void* out;
  /* scratch (device): */
  void* bias_staged;               /* fp32 [B*H, nq, W4, 4096], nq = ceil(S/128), W4 = 4*(nq+1) */
  void* fix;                       /* int32 [1 + 3*n_ctas], n_ctas = nq * ((N+2)/3 + 1) * B*H */
  void* fix_total;                 /* int32 [2] */
  void* words;                     /* int32 [B, N, W4]      (mask only) */
  void* keyany;                    /* int32 [B, W4]         (mask only) */
  void* rowkind;                   /* uint8 [B, N]          (mask only) */
  void* kcend;                     /* int32 [B]             (mask only) */
  void* kcstart;                   /* int32 [B]             (mask only) */
  void* rowkc0;                    /* int32 [B, N]          (mask only) */
  void* rowkc1;                    /* int32 [B, N]          (mask only) */
  void* counts;                    /* int32 [2]             (mask only) */
  void* stream;                    /* cudaStream_t */
  char err[512];                   /* out: error text when the call returns non-zero */
} TriattnM1Call;
/* returns 0 on success; nonzero with err filled otherwise */
int triattn_m1_xla_fwd(TriattnM1Call* c);
typedef int (*triattn_m1_fwd_fn)(TriattnM1Call*);
const char* triattn_m1_xla_describe(void);
int64_t triattn_m1_xla_smem_bytes(int flags);
#ifdef __cplusplus
}
#endif
#endif
