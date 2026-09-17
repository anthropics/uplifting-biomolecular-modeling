// dit_attn_exact.cu — Protenix v2 DiffusionTransformer pair-bias attention, fp32:  out[B,H,N,48] = softmax(q k^T + bias) v  (scale 1.0,
// bias [1|B,H,N,N] broadcast over B), producing bit-for-bit the output of torch.nn.functional.scaled_dot_product_attention's memory-efficient
// CUDA kernel for float32 (AttentionKernel<float, Sm80, aligned, 64, 64, 64>) on the same inputs: the per-element arithmetic — 3xTF32 operand
// split, tf32 tensor-core k-group chains with the same staged-accumulation grouping, fp32 bias add, FMUL by log2(e), FADD + ex2.approx (no FTZ),
// the online-softmax rescale and row-sum orders, rcp.rn epilogue — follows that kernel; the work decomposition is this file's own (kernel v7):
// Hopper wgmma for both GEMMs (wgmma m64nNk8 tf32 produces the same bits as mma.m16n8k8 tf32 on identical operands — established on the device
// and re-checked bitwise against the replaced op at load time), the three stock accumulation chains issued as one asynchronous batch into three
// register tiles, register-resident P as the A operand of P.V (k-permutation inside every 8-key group, V rows stored to match), K / V^T operands
// split to tf32 pairs once per CTA into the canonical core-matrix shared-memory layout, one producer warpgroup (cp.async + operand split + bias
// tiles) feeding two consumer warpgroups through a two-stage named-barrier pipeline, setmaxnreg register rebalancing, persistent CTAs.
// Arithmetic order derived from PyTorch aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h (BSD-3-Clause; see LICENSE_NOTE.md).
// Envelope: sm_90a; q,k,v fp32 [B,H,N,48] last-dim contiguous, row strides multiple of 4, 16-byte aligned; bias fp32 last-dim contiguous.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace dit_attn_exact {

constexpr int D = 48, BK = 64, NCWARPS = 8, BQ = 16 * NCWARPS, NT = 32 * NCWARPS;   // NT = consumer threads (256); BQ = 128 query rows per CTA
constexpr int NPROD = 128, NT_ALL = NT + NPROD;                                      // + producer warpgroup = 384 threads
constexpr int RP = 52;                                   // raw K / V row pitch (floats)
constexpr int BP = 72;                                   // bias pitch (floats)
constexpr int RAW_BYTES = 2 * BK * RP * 4;               // 26,624 (producer-private)
constexpr int K_LBO = 1040, K_SBO = 128, K_TILE = 12 * K_LBO;      // 12,480 B
constexpr int V_LBO = 880, V_SBO = 144, V_TILE = 16 * V_LBO;       // 14,080 B
constexpr int KSTAGE = 2 * K_TILE, VSTAGE = 2 * V_TILE, STAGE_BYTES = KSTAGE + VSTAGE;   // 53,120 per stage
constexpr int BIAS_BYTES = BQ * BP * 4;                  // 36,864 per stage
constexpr size_t SMEM_BYTES = RAW_BYTES + 2 * STAGE_BYTES + 2 * BIAS_BYTES;   // 206,592
constexpr int REG_PRODUCER = 56, REG_CONSUMER = 224;   // setmaxnreg budgets (128 x 56 + 256 x 224 <= 64K registers)
constexpr int BAR_FULL0 = 1;    // ids 1..4: FULL[stage][cwg] = 1 + 2*stage + cwg  (producer 128 arrive + consumer wg 128 sync = 256)
constexpr int BAR_EMPTY0 = 5;   // ids 5..6: EMPTY[stage]                          (consumers 256 arrive + producer 128 sync = 384)

__device__ __forceinline__ void bar_sync(int id, int count) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int count) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory"); }
template <int R> __device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(R)); }
template <int R> __device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(R)); }
__device__ __forceinline__ uint32_t f2u(float f) { return __float_as_uint(f); }
__device__ __forceinline__ float u2f(uint32_t u) { return __uint_as_float(u); }
__device__ __forceinline__ void split_tf32(float x, uint32_t& big, uint32_t& sml) {
  uint32_t b = f2u(x) & 0xffffe000u;
  float r = __fsub_rn(x, u2f(b));
  uint32_t rb = f2u(r);
  if (isfinite(r)) rb += 0x1000u;
  big = b; sml = rb;
}
__device__ __forceinline__ float ex2_full(float x) { float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }       // non-FTZ (stock)
__device__ __forceinline__ float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }    // == ex2_full for x >= -126
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ float4 lds128f(uint32_t a) { float4 v; asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ float2 lds64f(uint32_t a) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a)); return v; }
__device__ __forceinline__ void sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w)); }
__device__ __forceinline__ void sts32(uint32_t a, uint32_t x) { asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(x)); }
__device__ __forceinline__ void cp_async16(uint32_t s, const void* gmem, int src_bytes) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes)); }
__device__ __forceinline__ void cp_async16_full(uint32_t s, const void* gmem) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem)); }
__device__ __forceinline__ void cp_async4(uint32_t s, const void* gmem, int src_bytes) { asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes)); }
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
// 64-bit shared-memory matrix descriptor (no swizzle): start address, leading-dim byte offset (K direction), stride-dim byte offset (M/N direction)
__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((smem_addr & 0x3FFFFu) >> 4) | ((uint64_t)((lbo & 0x3FFFFu) >> 4) << 16) | ((uint64_t)((sbo & 0x3FFFFu) >> 4) << 32);
}
__device__ __forceinline__ void wg_m64n64k8_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t bdesc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(bdesc), "r"(1));
}
__device__ __forceinline__ void wg_m64n48k8_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t bdesc) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k8.f32.tf32.tf32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, %28, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(bdesc), "r"(1));
}
__device__ __forceinline__ void fence_acc32(float* d) { asm volatile("" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) :: "memory"); }
__device__ __forceinline__ void fence_acc24(float* d) { asm volatile("" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]) :: "memory"); }

struct Params {
  const float* q; const float* k; const float* v; const float* bias; float* out;
  int64_t q_sB, q_sH, q_sM, k_sB, k_sH, k_sM, v_sB, v_sH, v_sM, b_sB, b_sH, b_sM, o_sB, o_sH, o_sM;
  int N, B, H, nqb, total, bias_al16;
};

// ---------------- producer side ----------------
// chunk i = ptid + 128 j (j < 6): key row = i / 12, dims 4*(i % 12)..+3 — for K and V alike (12 consecutive threads read one 192-B row: coalesced)
struct PDesc {
  int row[6], dc[6];
  uint32_t kraw[6], kdst[6], vdst[6];      // raw V slot = kraw + BK*RP*4
};
template <bool kFull>
__device__ __forceinline__ void p_load_kv_raw(const PDesc& d, const float* kp_blk, const float* vp_blk, int64_t k_sM, int64_t v_sM, int nk) {
#pragma unroll
  for (int j = 0; j < 6; ++j) {
    const float* ks = kp_blk + (int64_t)d.row[j] * k_sM + 4 * d.dc[j];
    const float* vs = vp_blk + (int64_t)d.row[j] * v_sM + 4 * d.dc[j];
    if (kFull) { cp_async16_full(d.kraw[j], ks); cp_async16_full(d.kraw[j] + BK * RP * 4, vs); }
    else { int ok = d.row[j] < nk; cp_async16(d.kraw[j], ok ? ks : kp_blk, ok ? 16 : 0); cp_async16(d.kraw[j] + BK * RP * 4, ok ? vs : vp_blk, ok ? 16 : 0); }
  }
}
__device__ __forceinline__ void p_split(const PDesc& d, uint32_t stage_off) {
#pragma unroll
  for (int j = 0; j < 6; ++j) {
    float4 x = lds128f(d.kraw[j]);
    uint32_t b0, s0, b1, s1, b2, s2, b3, s3;
    split_tf32(x.x, b0, s0); split_tf32(x.y, b1, s1); split_tf32(x.z, b2, s2); split_tf32(x.w, b3, s3);
    sts128(d.kdst[j] + stage_off, b0, b1, b2, b3);
    sts128(d.kdst[j] + stage_off + K_TILE, s0, s1, s2, s3);
    float4 y = lds128f(d.kraw[j] + BK * RP * 4);
    const uint32_t vb = d.vdst[j] + stage_off, vsm = vb + V_TILE;
    split_tf32(y.x, b0, s0); split_tf32(y.y, b1, s1); split_tf32(y.z, b2, s2); split_tf32(y.w, b3, s3);
    sts32(vb, b0); sts32(vb + 16, b1); sts32(vb + 32, b2); sts32(vb + 48, b3);
    sts32(vsm, s0); sts32(vsm + 16, s1); sts32(vsm + 32, s2); sts32(vsm + 48, s3);
  }
}
// bias tile [BQ=128][64] -> stage (pitch BP): producer thread owns column chunk (ptid & 15) of rows (ptid >> 4) + 8 j, j = 0..15
template <bool kFull>
__device__ __forceinline__ void p_load_bias16(uint32_t bs_stage, const float* brow0, int64_t b_sM8, int row0, int col0, int mvalid, int nk, const float* safe) {
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    uint32_t dst = bs_stage + ((row0 + 8 * j) * BP + col0) * 4;
    const float* src = brow0 + (int64_t)j * b_sM8;
    if (kFull) cp_async16_full(dst, src);
    else {
      int nb = (row0 + 8 * j < mvalid) ? 4 * (nk - col0) : 0;
      nb = nb < 0 ? 0 : (nb > 16 ? 16 : nb);
      cp_async16(dst, nb > 0 ? src : safe, nb);
    }
  }
}
__device__ __forceinline__ void p_load_bias4(uint32_t bs_stage, const float* bp, int64_t b_sM, int q0, int mvalid, int kb0, int nk, int ptid) {
  for (int i = ptid; i < BQ * BK; i += NPROD) {
    int row = i >> 6, col = i & 63;
    int ok = (row < mvalid) && (col < nk);
    const float* src = bp + (int64_t)(q0 + (ok ? row : 0)) * b_sM + kb0 + (ok ? col : 0);
    cp_async4(bs_stage + (row * BP + col) * 4, src, ok ? 4 : 0);
  }
}
__device__ __forceinline__ void add64(float (&acc)[8][4], const float (&tmp)[8][4]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) { acc[i][0] = __fadd_rn(acc[i][0], tmp[i][0]); acc[i][1] = __fadd_rn(acc[i][1], tmp[i][1]); acc[i][2] = __fadd_rn(acc[i][2], tmp[i][2]); acc[i][3] = __fadd_rn(acc[i][3], tmp[i][3]); }
}
__device__ __forceinline__ void add48(float (&acc)[6][4], const float (&tmp)[6][4]) {
#pragma unroll
  for (int i = 0; i < 6; ++i) { acc[i][0] = __fadd_rn(acc[i][0], tmp[i][0]); acc[i][1] = __fadd_rn(acc[i][1], tmp[i][1]); acc[i][2] = __fadd_rn(acc[i][2], tmp[i][2]); acc[i][3] = __fadd_rn(acc[i][3], tmp[i][3]); }
}

template <bool kFull>
__device__ __forceinline__ void softmax_block(float (&acc)[8][4], float (&acc_o)[6][4], uint32_t bs_lane /* bs stage + (rl*BP+2t)*4 */, int nk, bool first,
                                              float& mi_lo, float& mi_hi, float& s_lo, float& s_hi, int t) {
  const float kLog2e = u2f(0x3fb8aa3bu);
  float mx_lo = -INFINITY, mx_hi = -INFINITY;
#pragma unroll
  for (int ni = 0; ni < 8; ++ni) {
    const int c = 8 * ni + 2 * t;
    float2 blo = lds64f(bs_lane + ni * 32), bhi = lds64f(bs_lane + ni * 32 + 8 * BP * 4);
    float x0 = __fmul_rn(__fadd_rn(acc[ni][0], blo.x), kLog2e), x1 = __fmul_rn(__fadd_rn(acc[ni][1], blo.y), kLog2e);
    float x2 = __fmul_rn(__fadd_rn(acc[ni][2], bhi.x), kLog2e), x3 = __fmul_rn(__fadd_rn(acc[ni][3], bhi.y), kLog2e);
    acc[ni][0] = x0; acc[ni][1] = x1; acc[ni][2] = x2; acc[ni][3] = x3;
    if (kFull) { mx_lo = fmaxf(mx_lo, fmaxf(x0, x1)); mx_hi = fmaxf(mx_hi, fmaxf(x2, x3)); }
    else {
      if (c < nk) { mx_lo = fmaxf(mx_lo, x0); mx_hi = fmaxf(mx_hi, x2); }
      if (c + 1 < nk) { mx_lo = fmaxf(mx_lo, x1); mx_hi = fmaxf(mx_hi, x3); }
    }
  }
  mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffff, mx_lo, 1)); mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffff, mx_lo, 2));
  mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffff, mx_hi, 1)); mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffff, mx_hi, 2));
  const float mnew_lo = fmaxf(mi_lo, mx_lo), mnew_hi = fmaxf(mi_hi, mx_hi);
  float r_lo = 1.f, r_hi = 1.f;
  if (mi_lo < mnew_lo) { r_lo = ex2_full(__fsub_rn(mi_lo, mnew_lo)); s_lo = __fmul_rn(s_lo, r_lo); }
  if (mi_hi < mnew_hi) { r_hi = ex2_full(__fsub_rn(mi_hi, mnew_hi)); s_hi = __fmul_rn(s_hi, r_hi); }
  mi_lo = mnew_lo; mi_hi = mnew_hi;
  const float nsub_lo = (mnew_lo == -INFINITY) ? -0.f : -mnew_lo, nsub_hi = (mnew_hi == -INFINITY) ? -0.f : -mnew_hi;   // x + (-mi) == x - mi bitwise
  if (!first) {
#pragma unroll
    for (int ni = 0; ni < 6; ++ni) {
      acc_o[ni][0] = __fmul_rn(acc_o[ni][0], r_lo); acc_o[ni][1] = __fmul_rn(acc_o[ni][1], r_lo);
      acc_o[ni][2] = __fmul_rn(acc_o[ni][2], r_hi); acc_o[ni][3] = __fmul_rn(acc_o[ni][3], r_hi);
    }
  }
  // arguments a = X - mi (in place) and their minimum (decides the MUFU-only fast path; result bits are identical on both paths)
  float amin = 0.f;
#pragma unroll
  for (int ni = 0; ni < 8; ++ni) {
    float a0 = __fadd_rn(acc[ni][0], nsub_lo), a1 = __fadd_rn(acc[ni][1], nsub_lo), a2 = __fadd_rn(acc[ni][2], nsub_hi), a3 = __fadd_rn(acc[ni][3], nsub_hi);
    acc[ni][0] = a0; acc[ni][1] = a1; acc[ni][2] = a2; acc[ni][3] = a3;
    amin = fminf(amin, fminf(fminf(a0, a1), fminf(a2, a3)));
  }
  float h0_lo = 0.f, h0_hi = 0.f, h1_lo = 0.f, h1_hi = 0.f;
  if (amin >= -126.f) {          // NaN amin compares false -> exact path
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
      const int c = 8 * ni + 2 * t;
      float p0 = ex2_ftz(acc[ni][0]), p1 = ex2_ftz(acc[ni][1]), p2 = ex2_ftz(acc[ni][2]), p3 = ex2_ftz(acc[ni][3]);
      if (!kFull) { const bool c0ok = c < nk, c1ok = (c + 1) < nk; p0 = c0ok ? p0 : 0.f; p1 = c1ok ? p1 : 0.f; p2 = c0ok ? p2 : 0.f; p3 = c1ok ? p3 : 0.f; }
      acc[ni][0] = p0; acc[ni][1] = p1; acc[ni][2] = p2; acc[ni][3] = p3;
      if (ni < 4) { h0_lo = __fadd_rn(__fadd_rn(h0_lo, p0), p1); h0_hi = __fadd_rn(__fadd_rn(h0_hi, p2), p3); }
      else        { h1_lo = __fadd_rn(__fadd_rn(h1_lo, p0), p1); h1_hi = __fadd_rn(__fadd_rn(h1_hi, p2), p3); }
    }
  } else {
#pragma unroll
    for (int ni = 0; ni < 8; ++ni) {
      const int c = 8 * ni + 2 * t;
      float p0 = ex2_full(acc[ni][0]), p1 = ex2_full(acc[ni][1]), p2 = ex2_full(acc[ni][2]), p3 = ex2_full(acc[ni][3]);
      if (!kFull) { const bool c0ok = c < nk, c1ok = (c + 1) < nk; p0 = c0ok ? p0 : 0.f; p1 = c1ok ? p1 : 0.f; p2 = c0ok ? p2 : 0.f; p3 = c1ok ? p3 : 0.f; }
      acc[ni][0] = p0; acc[ni][1] = p1; acc[ni][2] = p2; acc[ni][3] = p3;
      if (ni < 4) { h0_lo = __fadd_rn(__fadd_rn(h0_lo, p0), p1); h0_hi = __fadd_rn(__fadd_rn(h0_hi, p2), p3); }
      else        { h1_lo = __fadd_rn(__fadd_rn(h1_lo, p0), p1); h1_hi = __fadd_rn(__fadd_rn(h1_hi, p2), p3); }
    }
  }
  h0_lo = __fadd_rn(h0_lo, __shfl_xor_sync(0xffffffff, h0_lo, 1)); h0_lo = __fadd_rn(h0_lo, __shfl_xor_sync(0xffffffff, h0_lo, 2));
  h0_hi = __fadd_rn(h0_hi, __shfl_xor_sync(0xffffffff, h0_hi, 1)); h0_hi = __fadd_rn(h0_hi, __shfl_xor_sync(0xffffffff, h0_hi, 2));
  h1_lo = __fadd_rn(h1_lo, __shfl_xor_sync(0xffffffff, h1_lo, 1)); h1_lo = __fadd_rn(h1_lo, __shfl_xor_sync(0xffffffff, h1_lo, 2));
  h1_hi = __fadd_rn(h1_hi, __shfl_xor_sync(0xffffffff, h1_hi, 1)); h1_hi = __fadd_rn(h1_hi, __shfl_xor_sync(0xffffffff, h1_hi, 2));
  s_lo = __fadd_rn(__fadd_rn(s_lo, h0_lo), h1_lo);
  s_hi = __fadd_rn(__fadd_rn(s_hi, h0_hi), h1_hi);
}


__device__ __forceinline__ void zero32(float* a) {
#pragma unroll
  for (int i = 0; i < 32; ++i) a[i] = 0.f;
}
__device__ __forceinline__ void zero24(float* a) {
#pragma unroll
  for (int i = 0; i < 24; ++i) a[i] = 0.f;
}

// =====================================================================================================================================
__global__ void __launch_bounds__(NT_ALL, 1) dit_attn_exact_kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t smem0 = smem_u32(smem_raw);
  const uint32_t raw_u32 = smem0, st_u32 = smem0 + RAW_BYTES, bias_u32 = smem0 + RAW_BYTES + 2 * STAGE_BYTES;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int N = p.N, nkb = (N + BK - 1) / BK;

  if (warp >= NCWARPS) {
    // =============================== PRODUCER warpgroup ===============================
    setmaxnreg_dec<REG_PRODUCER>();
    const int ptid = tid - NT;
    PDesc d;
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      int i = ptid + NPROD * j; int row = i / 12, dc = i % 12;
      d.row[j] = row; d.dc[j] = dc;
      d.kraw[j] = raw_u32 + (row * RP + 4 * dc) * 4;
      d.kdst[j] = st_u32 + dc * K_LBO + (row >> 3) * K_SBO + (row & 7) * 16;
      const int chunk = 2 * (row >> 3) + (row & 1), slot = (row & 7) >> 1, dim0 = 4 * dc;
      d.vdst[j] = st_u32 + KSTAGE + chunk * V_LBO + (dim0 >> 3) * V_SBO + (dim0 & 7) * 16 + slot * 4;
    }
    const int brow0 = ptid >> 4, bcol0 = 4 * (ptid & 15);
    auto issue_raw = [&](int item, int kb) {
      const int b = item % p.B, hq = item / p.B, h = hq % p.H;
      const float* kp = p.k + b * p.k_sB + h * p.k_sH; const float* vp = p.v + b * p.v_sB + h * p.v_sH;
      const int kb0 = kb * BK, nk = min(BK, N - kb0);
      const float* kb_ptr = kp + (int64_t)kb0 * p.k_sM; const float* vb_ptr = vp + (int64_t)kb0 * p.v_sM;
      if (nk == BK) p_load_kv_raw<true>(d, kb_ptr, vb_ptr, p.k_sM, p.v_sM, nk); else p_load_kv_raw<false>(d, kb_ptr, vb_ptr, p.k_sM, p.v_sM, nk);
    };
    int gkb = 0;                                               // global block counter (same sequence as the consumers)
    if ((int)blockIdx.x < p.total) { issue_raw(blockIdx.x, 0); }
    cp_async_commit();                                         // G raw(first)
    for (int item = blockIdx.x; item < p.total; item += gridDim.x) {
      const int b = item % p.B, hq = item / p.B, h = hq % p.H, qb = hq / p.H;
      const int q0 = qb * BQ, mvalid = min(BQ, N - q0);
      const float* bp = p.bias + b * p.b_sB + h * p.b_sH;
      const float* bthr = bp + (int64_t)(q0 + brow0) * p.b_sM + bcol0;
      const bool rows_full = (mvalid == BQ);
      for (int kb = 0; kb < nkb; ++kb, ++gkb) {
        const int s = gkb & 1, kb0 = kb * BK, nk = min(BK, N - kb0);
        if (gkb >= 2) bar_sync(BAR_EMPTY0 + s, NT_ALL);        // consumers done with the block that last used stage s
        // bias(kb) -> bias stage s
        {
          const uint32_t stage = bias_u32 + s * BIAS_BYTES;
          if (!p.bias_al16) p_load_bias4(stage, bp, p.b_sM, q0, mvalid, kb0, nk, ptid);
          else if (rows_full && nk == BK) p_load_bias16<true>(stage, bthr + kb0, 8 * p.b_sM, brow0, bcol0, mvalid, nk, bp);
          else                            p_load_bias16<false>(stage, bthr + kb0, 8 * p.b_sM, brow0, bcol0, mvalid, nk, bp);
        }
        cp_async_commit();                                     // G bias(kb)        pending: raw(kb), bias(kb)
        cp_async_wait<1>();                                    // raw(kb) landed (own chunks)
        p_split(d, s * STAGE_BYTES);                           // -> K / V^T tiles of stage s
        // prefetch the next block's raw K/V (this item or the next one) into the private raw buffer (own slots only: no hazard)
        if (kb + 1 < nkb) issue_raw(item, kb + 1);
        else if (item + (int)gridDim.x < p.total) issue_raw(item + gridDim.x, 0);
        cp_async_commit();                                     // G raw(next)       pending: bias(kb), raw(next)
        cp_async_wait<1>();                                    // bias(kb) landed
        fence_proxy_async();                                   // split stores -> async proxy (wgmma reads)
        __threadfence_block();
        bar_arrive(BAR_FULL0 + 2 * s + 0, NPROD + 128);        // stage s (K, V^T, bias) ready for consumer wg 0
        bar_arrive(BAR_FULL0 + 2 * s + 1, NPROD + 128);        // ... and consumer wg 1
      }
    }
    cp_async_wait<0>();
    return;
  }

  // =============================== CONSUMER warpgroups ===============================
  setmaxnreg_inc<REG_CONSUMER>();
  const int g = lane >> 2, t = lane & 3, cwg = warp >> 2;
  const uint64_t dK0 = make_desc(st_u32, K_LBO, K_SBO), dKs0 = make_desc(st_u32 + K_TILE, K_LBO, K_SBO);
  const uint64_t dV0 = make_desc(st_u32 + KSTAGE, V_LBO, V_SBO), dVs0 = make_desc(st_u32 + KSTAGE + V_TILE, V_LBO, V_SBO);
  const int rl = 16 * warp + g, rh = rl + 8;
  const uint32_t bs_lane = bias_u32 + (rl * BP + 2 * t) * 4;
  int gkb = 0;

  for (int item = blockIdx.x; item < p.total; item += gridDim.x) {
    const int b = item % p.B, hq = item / p.B, h = hq % p.H, qb = hq / p.H;
    const int q0 = qb * BQ, mvalid = min(BQ, N - q0);
    const float* qp = p.q + b * p.q_sB + h * p.q_sH;
    float* op = p.out + b * p.o_sB + h * p.o_sH;

    uint32_t qbig[6][4], qsml[6][4];
    {
      const int rlo = q0 + rl, rhi = q0 + rh;
      const float* qlo = qp + (int64_t)rlo * p.q_sM + t; const float* qhi = qp + (int64_t)rhi * p.q_sM + t;
#pragma unroll
      for (int kg = 0; kg < 6; ++kg) {
        float x0 = rlo < N ? __ldg(qlo + 8 * kg) : 0.f, x1 = rhi < N ? __ldg(qhi + 8 * kg) : 0.f;
        float x2 = rlo < N ? __ldg(qlo + 8 * kg + 4) : 0.f, x3 = rhi < N ? __ldg(qhi + 8 * kg + 4) : 0.f;
        split_tf32(x0, qbig[kg][0], qsml[kg][0]); split_tf32(x1, qbig[kg][1], qsml[kg][1]);
        split_tf32(x2, qbig[kg][2], qsml[kg][2]); split_tf32(x3, qbig[kg][3], qsml[kg][3]);
      }
    }
    float acc_o[6][4]; zero24(&acc_o[0][0]);
    float mi_lo = -INFINITY, mi_hi = -INFINITY, s_lo = 0.f, s_hi = 0.f;

    for (int kb = 0; kb < nkb; ++kb, ++gkb) {
      const int s = gkb & 1, kb0 = kb * BK, nk = min(BK, N - kb0);
      const uint32_t soff16 = (uint32_t)(s * STAGE_BYTES) >> 4;
      bar_sync(BAR_FULL0 + 2 * s + cwg, NPROD + 128);        // stage s ready (K, V^T split tiles + bias)

      // ===== MM0: S = Q K^T.  Stock staged accumulation = three chains T_A{g0} | T_B{g1,g2} | T_C{g3,g4,g5}, S = ((0 + T_A) + T_B) + T_C.
      // The chains accumulate into three separate register tiles, issued as ONE async batch (interleaved so adjacent wgmmas are independent),
      // one wait; per-accumulator issue order = chain order, so every element sees exactly the stock operation sequence.
      float acc[8][4], tB[8][4], tC[8][4];     // acc doubles as T_A
      const uint64_t dK = dK0 + soff16, dKs = dKs0 + soff16;
      constexpr uint64_t KG16 = (2 * K_LBO) >> 4;
      zero32(&acc[0][0]); zero32(&tB[0][0]); zero32(&tC[0][0]);
      wg_fence();
      // pass 1 (a_small . b_big) of g0 | g1 | g3, pass 2, pass 3, then the remaining groups of chains B and C
      wg_m64n64k8_rs(&acc[0][0], qsml[0][0], qsml[0][1], qsml[0][2], qsml[0][3], dK);
      wg_m64n64k8_rs(&tB[0][0],  qsml[1][0], qsml[1][1], qsml[1][2], qsml[1][3], dK + 1 * KG16);
      wg_m64n64k8_rs(&tC[0][0],  qsml[3][0], qsml[3][1], qsml[3][2], qsml[3][3], dK + 3 * KG16);
      wg_m64n64k8_rs(&acc[0][0], qbig[0][0], qbig[0][1], qbig[0][2], qbig[0][3], dKs);
      wg_m64n64k8_rs(&tB[0][0],  qbig[1][0], qbig[1][1], qbig[1][2], qbig[1][3], dKs + 1 * KG16);
      wg_m64n64k8_rs(&tC[0][0],  qbig[3][0], qbig[3][1], qbig[3][2], qbig[3][3], dKs + 3 * KG16);
      wg_m64n64k8_rs(&acc[0][0], qbig[0][0], qbig[0][1], qbig[0][2], qbig[0][3], dK);
      wg_m64n64k8_rs(&tB[0][0],  qbig[1][0], qbig[1][1], qbig[1][2], qbig[1][3], dK + 1 * KG16);
      wg_m64n64k8_rs(&tC[0][0],  qbig[3][0], qbig[3][1], qbig[3][2], qbig[3][3], dK + 3 * KG16);
      // chain B group g2 and chain C group g4 interleaved, then chain C group g5
      wg_m64n64k8_rs(&tB[0][0],  qsml[2][0], qsml[2][1], qsml[2][2], qsml[2][3], dK + 2 * KG16);
      wg_m64n64k8_rs(&tC[0][0],  qsml[4][0], qsml[4][1], qsml[4][2], qsml[4][3], dK + 4 * KG16);
      wg_m64n64k8_rs(&tB[0][0],  qbig[2][0], qbig[2][1], qbig[2][2], qbig[2][3], dKs + 2 * KG16);
      wg_m64n64k8_rs(&tC[0][0],  qbig[4][0], qbig[4][1], qbig[4][2], qbig[4][3], dKs + 4 * KG16);
      wg_m64n64k8_rs(&tB[0][0],  qbig[2][0], qbig[2][1], qbig[2][2], qbig[2][3], dK + 2 * KG16);
      wg_m64n64k8_rs(&tC[0][0],  qbig[4][0], qbig[4][1], qbig[4][2], qbig[4][3], dK + 4 * KG16);
      wg_m64n64k8_rs(&tC[0][0],  qsml[5][0], qsml[5][1], qsml[5][2], qsml[5][3], dK + 5 * KG16);
      wg_m64n64k8_rs(&tC[0][0],  qbig[5][0], qbig[5][1], qbig[5][2], qbig[5][3], dKs + 5 * KG16);
      wg_m64n64k8_rs(&tC[0][0],  qbig[5][0], qbig[5][1], qbig[5][2], qbig[5][3], dK + 5 * KG16);
      wg_commit(); wg_wait<0>(); fence_acc32(&acc[0][0]); fence_acc32(&tB[0][0]); fence_acc32(&tC[0][0]);
#pragma unroll
      for (int i = 0; i < 8; ++i) { acc[i][0] = __fadd_rn(0.f, acc[i][0]); acc[i][1] = __fadd_rn(0.f, acc[i][1]); acc[i][2] = __fadd_rn(0.f, acc[i][2]); acc[i][3] = __fadd_rn(0.f, acc[i][3]); }
      add64(acc, tB);
      add64(acc, tC);

      // ===== softmax (registers; identical to v4)
      const uint32_t bsl = bs_lane + s * BIAS_BYTES;
      if (nk == BK) softmax_block<true>(acc, acc_o, bsl, nk, kb == 0, mi_lo, mi_hi, s_lo, s_hi, t);
      else          softmax_block<false>(acc, acc_o, bsl, nk, kb == 0, mi_lo, mi_hi, s_lo, s_hi, t);

      // ===== MM1: O += P V.  Stock chains U_A{j0} | U_B{j1..j4} | U_C{j5..j7} (tail block nk<=32: U_A{j0} | U_B{j1..j3}), O = ((O + U_A) + U_B) + U_C.
      // Batch 1 issues U_A and U_B into two register tiles (one wait), batch 2 issues U_C.
      const uint64_t dV = dV0 + soff16, dVs = dVs0 + soff16;
      constexpr uint64_t JG16 = (2 * V_LBO) >> 4;
      {
        float uA[6][4], uB[6][4];
        uint32_t pa[5][8];                     // split A fragments of groups j0..j4: (P[j][0], P[j][2], P[j][1], P[j][3]) -> big[0..3], small[4..7]
#pragma unroll
        for (int j = 0; j < 5; ++j) {
          split_tf32(acc[j][0], pa[j][0], pa[j][4]); split_tf32(acc[j][2], pa[j][1], pa[j][5]); split_tf32(acc[j][1], pa[j][2], pa[j][6]); split_tf32(acc[j][3], pa[j][3], pa[j][7]);
        }
        zero24(&uA[0][0]); zero24(&uB[0][0]);
        wg_fence();
        wg_m64n48k8_rs(&uA[0][0], pa[0][4], pa[0][5], pa[0][6], pa[0][7], dV);                 // j0 pass 1
        wg_m64n48k8_rs(&uB[0][0], pa[1][4], pa[1][5], pa[1][6], pa[1][7], dV + 1 * JG16);      // j1 pass 1
        wg_m64n48k8_rs(&uA[0][0], pa[0][0], pa[0][1], pa[0][2], pa[0][3], dVs);                // j0 pass 2
        wg_m64n48k8_rs(&uB[0][0], pa[1][0], pa[1][1], pa[1][2], pa[1][3], dVs + 1 * JG16);     // j1 pass 2
        wg_m64n48k8_rs(&uA[0][0], pa[0][0], pa[0][1], pa[0][2], pa[0][3], dV);                 // j0 pass 3
        wg_m64n48k8_rs(&uB[0][0], pa[1][0], pa[1][1], pa[1][2], pa[1][3], dV + 1 * JG16);      // j1 pass 3
#pragma unroll
        for (int j = 2; j < 4; ++j) {
          wg_m64n48k8_rs(&uB[0][0], pa[j][4], pa[j][5], pa[j][6], pa[j][7], dV + j * JG16);
          wg_m64n48k8_rs(&uB[0][0], pa[j][0], pa[j][1], pa[j][2], pa[j][3], dVs + j * JG16);
          wg_m64n48k8_rs(&uB[0][0], pa[j][0], pa[j][1], pa[j][2], pa[j][3], dV + j * JG16);
        }
        if (nk > 32) {
          wg_m64n48k8_rs(&uB[0][0], pa[4][4], pa[4][5], pa[4][6], pa[4][7], dV + 4 * JG16);
          wg_m64n48k8_rs(&uB[0][0], pa[4][0], pa[4][1], pa[4][2], pa[4][3], dVs + 4 * JG16);
          wg_m64n48k8_rs(&uB[0][0], pa[4][0], pa[4][1], pa[4][2], pa[4][3], dV + 4 * JG16);
        }
        wg_commit(); wg_wait<0>(); fence_acc24(&uA[0][0]); fence_acc24(&uB[0][0]);
        add48(acc_o, uA);
        add48(acc_o, uB);
      }
      if (nk > 32) {
        float uC[6][4]; uint32_t pc[3][8];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          split_tf32(acc[5 + j][0], pc[j][0], pc[j][4]); split_tf32(acc[5 + j][2], pc[j][1], pc[j][5]); split_tf32(acc[5 + j][1], pc[j][2], pc[j][6]); split_tf32(acc[5 + j][3], pc[j][3], pc[j][7]);
        }
        zero24(&uC[0][0]);
        wg_fence();
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          wg_m64n48k8_rs(&uC[0][0], pc[j][4], pc[j][5], pc[j][6], pc[j][7], dV + (5 + j) * JG16);
          wg_m64n48k8_rs(&uC[0][0], pc[j][0], pc[j][1], pc[j][2], pc[j][3], dVs + (5 + j) * JG16);
          wg_m64n48k8_rs(&uC[0][0], pc[j][0], pc[j][1], pc[j][2], pc[j][3], dV + (5 + j) * JG16);
        }
        wg_commit(); wg_wait<0>(); fence_acc24(&uC[0][0]);
        add48(acc_o, uC);
      }
          bar_arrive(BAR_EMPTY0 + s, NT_ALL);                     // this warp is done reading stage s
    }
    {
      const float a_lo = __frcp_rn(s_lo == 0.f ? 1.f : s_lo), a_hi = __frcp_rn(s_hi == 0.f ? 1.f : s_hi);
      float* olo = op + (int64_t)(q0 + rl) * p.o_sM + 2 * t; float* ohi = op + (int64_t)(q0 + rh) * p.o_sM + 2 * t;
#pragma unroll
      for (int ni = 0; ni < 6; ++ni) {
        if (rl < mvalid) *reinterpret_cast<float2*>(olo + 8 * ni) = make_float2(__fmul_rn(a_lo, acc_o[ni][0]), __fmul_rn(a_lo, acc_o[ni][1]));
        if (rh < mvalid) *reinterpret_cast<float2*>(ohi + 8 * ni) = make_float2(__fmul_rn(a_hi, acc_o[ni][2]), __fmul_rn(a_hi, acc_o[ni][3]));
      }
    }
  }
}

static bool g_attr_set = false;
static int g_num_sms = 0;
static void ensure_init() {
  if (!g_attr_set) { C10_CUDA_CHECK(cudaFuncSetAttribute(dit_attn_exact_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES)); g_attr_set = true; }
  if (!g_num_sms) g_num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}
void init() { ensure_init(); }

static void check_input(const at::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat, name, " must be a CUDA float32 tensor");
  TORCH_CHECK(x.dim() == 4, name, " must be 4-D [B,H,N,D]");
  TORCH_CHECK(x.stride(3) == 1, name, " last dim must be contiguous");
}

at::Tensor forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& bias, int64_t ctas_per_sm) {
  check_input(q, "q"); check_input(k, "k"); check_input(v, "v"); check_input(bias, "bias");
  const int64_t B = q.size(0), H = q.size(1), N = q.size(2);
  TORCH_CHECK(q.size(3) == D && k.size(3) == D && v.size(3) == D, "head dim must be 48");
  TORCH_CHECK(k.size(2) == N && v.size(2) == N && k.size(0) == B && v.size(0) == B && k.size(1) == H && v.size(1) == H, "shape mismatch");
  TORCH_CHECK((bias.size(0) == 1 || bias.size(0) == B) && bias.size(1) == H && bias.size(2) == N && bias.size(3) == N, "bias shape");
  TORCH_CHECK(q.stride(2) % 4 == 0 && k.stride(2) % 4 == 0 && v.stride(2) % 4 == 0, "q/k/v row strides must be multiples of 4 floats");
  TORCH_CHECK(((uintptr_t)q.data_ptr() % 16 == 0) && ((uintptr_t)k.data_ptr() % 16 == 0) && ((uintptr_t)v.data_ptr() % 16 == 0), "q/k/v must be 16B aligned");
  TORCH_CHECK(q.stride(0) % 2 == 0 && q.stride(1) % 2 == 0 && k.stride(0) % 4 == 0 && k.stride(1) % 4 == 0 && v.stride(0) % 4 == 0 && v.stride(1) % 4 == 0, "batch/head strides alignment");
  TORCH_CHECK(N >= 1 && B * H * ((N + BQ - 1) / BQ) < (1 << 30), "size");
  auto out_bnhd = at::empty({B, N, H, D}, q.options());
  auto out = out_bnhd.transpose(1, 2);
  Params p;
  p.q = q.data_ptr<float>(); p.k = k.data_ptr<float>(); p.v = v.data_ptr<float>(); p.bias = bias.data_ptr<float>(); p.out = out.data_ptr<float>();
  p.q_sB = q.stride(0); p.q_sH = q.stride(1); p.q_sM = q.stride(2);
  p.k_sB = k.stride(0); p.k_sH = k.stride(1); p.k_sM = k.stride(2);
  p.v_sB = v.stride(0); p.v_sH = v.stride(1); p.v_sM = v.stride(2);
  p.b_sB = bias.size(0) == 1 ? 0 : bias.stride(0); p.b_sH = bias.stride(1); p.b_sM = bias.stride(2);
  p.o_sB = out.stride(0); p.o_sH = out.stride(1); p.o_sM = out.stride(2);
  p.N = (int)N; p.B = (int)B; p.H = (int)H; p.nqb = (int)((N + BQ - 1) / BQ); p.total = p.nqb * (int)H * (int)B;
  p.bias_al16 = ((uintptr_t)bias.data_ptr() % 16 == 0) && (p.b_sM % 4 == 0) && (p.b_sH % 4 == 0) && (p.b_sB % 4 == 0);
  ensure_init();
  int cps = ctas_per_sm > 0 ? (int)ctas_per_sm : 1;
  int grid = std::min(p.total, g_num_sms * cps);
  dit_attn_exact_kernel<<<grid, NT_ALL, SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace dit_attn_exact

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dit_attn_exact::forward, "out = softmax(q k^T + bias) v, fp32, bit-identical to the memory-efficient SDPA kernel (q,k,v,bias, ctas_per_sm=1)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("bias"), py::arg("ctas_per_sm") = 1);
  m.def("init", &dit_attn_exact::init, "set kernel attributes / query the device once (call outside CUDA-graph capture)");
  m.attr("version") = "7";
  m.attr("head_dim") = 48;
}
