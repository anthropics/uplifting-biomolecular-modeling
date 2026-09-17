/* triattn_xla: the plain-C boundary between the XLA-FFI launcher (cubin_launch.cc, g++, one build per XLA-FFI API version) and the
 * CUDA library (triattn_mw_cuda.cu, nvcc sm_90a, one build for every jaxlib line).  No XLA, Python or torch type crosses it. */
#ifndef TRIATTN_CUDA_ABI_H_
#define TRIATTN_CUDA_ABI_H_

#define TRIATTN_CUDA_ABI_VERSION 1       /* the call struct as libtriattn_mw_cuda.so reads it */
#define TRIATTN_CUDA_ABI_LSE_VERSION 2   /* revision 2 = the same struct with `lse` appended and set: what libtriattn_mw_cuda_lse.so requires */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TriattnCudaCall {
  int abi_version;                 /* = TRIATTN_CUDA_ABI_VERSION */
  void* stream;                    /* cudaStream_t / CUstream the work is enqueued on (XLA's compute stream) */
  /* inputs (device pointers) */
  const void* q; const void* k; const void* v;   /* bf16 [B,N,H,S,D] (layout 0) or [B,N,S,H,D] (layout 1), dense row-major */
  const void* bias;                /* [bias_B,H,S,S] dense, dtype bias_dtype: 0 f32, 1 bf16, 2 f16 */
  const void* mask;                /* u8 [B,N,S] dense (1 = attend) or NULL */
  int bias_dtype; int layout;
  long long B, N, H, S, D, bias_B;
  /* outputs / scratch (device pointers; sizes as documented in cubin_launch.cc CudaFwdImpl) */
  void* out;                       /* bf16, q's shape and layout */
  void* bias_staged; void* keyany; void* rowkind; void* irr; void* rgflag; void* ktend; void* mtab; void* ctab; void* ktendr; void* fix;
  long long fix_elems;             /* capacity of fix (int32 elements) */
  float scale;                     /* softmax scale (D ** -0.5 when the caller passes <= 0) */
  int dbg;                         /* 0 in service */
  char err[512];                   /* out: error text when the call returns non-zero */
  void* lse;                       /* revision 2 only (abi_version == 2): fp32 [B,N,H,S] dense, per query row log2-sum-exp; appended last so a
                                      revision-1 callee's view of the struct is unchanged */
} TriattnCudaCall;

typedef int (*triattn_cuda_fwd_fn)(TriattnCudaCall*);

/* exported by the CUDA library */
int triattn_mw_cuda_fwd(TriattnCudaCall* c);                                   /* 0 = launched; else c->err says why (nothing else was substituted) */
int triattn_mw_cuda_fwd_lse(TriattnCudaCall* c);                               /* the lse library's entry (libtriattn_mw_cuda_lse.so): as above + writes c->lse; abi_version 2 */
long long triattn_mw_cuda_fix_elems(long long B, long long N, long long H, long long S);
long long triattn_mw_cuda_smem_bytes(void);
const char* triattn_mw_cuda_describe(void);                                    /* "triattn_mw sm_90a D=32 R=3 ST=4 ..." */

#ifdef __cplusplus
}
#endif
#endif  /* TRIATTN_CUDA_ABI_H_ */
