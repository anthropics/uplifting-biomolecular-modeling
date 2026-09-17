// Torch extension `triattn_sm80_ext`: staging kernels (bias -> fp32 fragment order, mask -> key/row tables), the forward call, and the
// geometry / buffer-size queries the Python entry (triattn_sm80.py) uses.  The kernels themselves are instantiated per head dim in
// inst_d16.cu / inst_d32.cu / inst_d64.cu (separate translation units so they compile in parallel); which of them are part of this
// build is named by -DTS_HAVE_D16 / -DTS_HAVE_D32 / -DTS_HAVE_D64.
#include "launch_sm80.cuh"
#include "inst_sm80.h"

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>
#include <tuple>
#include <vector>

namespace {
using namespace triattn_sm80;

const Inst* inst_for(int D) {
#ifdef TS_HAVE_D16
  if (D == 16) return &inst_d16;
#endif
#ifdef TS_HAVE_D32
  if (D == 32) return &inst_d32;
#endif
#ifdef TS_HAVE_D64
  if (D == 64) return &inst_d64;
#endif
  return nullptr;
}

std::vector<int64_t> dims_built() {
  std::vector<int64_t> d;
#ifdef TS_HAVE_D16
  d.push_back(16);
#endif
#ifdef TS_HAVE_D32
  d.push_back(32);
#endif
#ifdef TS_HAVE_D64
  d.push_back(64);
#endif
  return d;
}

const Inst& need_inst(int64_t D) {
  const Inst* in = inst_for((int)D);
  TORCH_CHECK(in != nullptr, "head dim ", D, " is not part of this build");
  return *in;
}

// (R, QG, STAGES, WARPS, BM, smem bytes of the hot kernel, smem bytes of the SAFE kernel, RW rows per warp, small_max, small_max_rows) for head
// dim D: the large-S geometry (small = false) or the small-S one (small = true; fwd uses it when S <= small_max, or S <= small_max_rows and
// the q/k/v position strides are <= 512 elements)
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> geometry(int64_t D, bool small) {
  const Inst& in = need_inst(D);
  const GeoInfo& g = small ? in.small : in.big;
  return std::make_tuple((int64_t)g.R, (int64_t)g.QG, (int64_t)g.STAGES, (int64_t)(g.R / g.RW * g.QG), (int64_t)(32 * g.QG), (int64_t)g.smem_hot, (int64_t)g.smem_safe, (int64_t)g.RW, (int64_t)in.small_max, (int64_t)in.small_max_rows);
}

long long cta_tiles(const GeoInfo& g, int64_t B, int64_t N, int64_t H, int64_t S) {
  const int64_t BM = 32 * g.QG;
  return ((S + BM - 1) / BM) * ((N + g.R - 1) / g.R) * B * H;
}

int64_t fix_buffer_elems(int64_t B, int64_t N, int64_t H, int64_t S, int64_t D) {   // int32 entries of the fix buffer fwd() needs (either geometry)
  const Inst& in = need_inst(D);
  return FIX_LIST + 3 * std::max(cta_tiles(in.big, B, N, H, S), cta_tiles(in.small, B, N, H, S));
}

// bias4 [B,H,S,S] fp32 | bf16 | fp16, any strides (transposed views included) -> fp32 [B,H,S128,S64] = bias / scale in accumulator-fragment
// order, -inf on keys >= S and on keys no row attends (keyany [B,S64] uint8 from stage_mask, or none).
torch::Tensor stage_bias(torch::Tensor bias4, double scale, c10::optional<torch::Tensor> keyany) {
  TORCH_CHECK(bias4.dim() == 4 && bias4.size(2) == bias4.size(3), "bias must be [B,H,S,S]");
  TORCH_CHECK(bias4.is_cuda(), "bias must be a CUDA tensor");
  c10::cuda::CUDAGuard guard(bias4.device());
  const int B = bias4.size(0), H = bias4.size(1), S = bias4.size(2), S64 = (S + 63) / 64 * 64, S128 = (S + 127) / 128 * 128;
  auto out = torch::empty({B, H, S128, S64}, bias4.options().dtype(torch::kFloat32));
  const long long n = (long long)B * H * S128 * S64;
  const int threads = 256;
  const int blocks = (int)std::min<long long>((n + threads - 1) / threads, (long long)triattn_sm80::detail::num_sms() * 32);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const float inv_scale = (float)(1.0 / scale);
  const uint8_t* ka = nullptr;
  if (keyany.has_value()) {
    TORCH_CHECK(keyany->scalar_type() == torch::kUInt8 && keyany->is_contiguous() && keyany->numel() == (long long)B * S64, "keyany [B,S64] uint8");
    ka = keyany->data_ptr<uint8_t>();
  }
  if (bias4.scalar_type() == torch::kFloat32)
    stage_bias_kernel<float><<<blocks, threads, 0, st>>>(bias4.data_ptr<float>(), bias4.stride(0), bias4.stride(1), bias4.stride(2), bias4.stride(3), ka, out.data_ptr<float>(), B, H, S, S128, S64, inv_scale);
  else if (bias4.scalar_type() == torch::kBFloat16)
    stage_bias_kernel<__nv_bfloat16><<<blocks, threads, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(bias4.data_ptr()), bias4.stride(0), bias4.stride(1), bias4.stride(2), bias4.stride(3), ka, out.data_ptr<float>(), B, H, S, S128, S64, inv_scale);
  else if (bias4.scalar_type() == torch::kFloat16)
    stage_bias_kernel<__half><<<blocks, threads, 0, st>>>(reinterpret_cast<const __half*>(bias4.data_ptr()), bias4.stride(0), bias4.stride(1), bias4.stride(2), bias4.stride(3), ka, out.data_ptr<float>(), B, H, S, S128, S64, inv_scale);
  else TORCH_CHECK(false, "bias dtype must be fp32, bf16 or fp16");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// mask [B,N,S] uint8 (key stride 1) -> {keyany [B,S64] uint8, rows [B*N,4] int32 ({a, e, kind, 0} per pair row), maskw [B*N*nkt*2] int32
// (32-key mask words)}.  census int32[>=4] accumulates: [1] += ragged rows, [2] += fully-masked rows, [3] += row groups (of R rows)
// holding a ragged row.
std::vector<torch::Tensor> stage_mask(torch::Tensor mask_u8, torch::Tensor census, int64_t R) {
  TORCH_CHECK(mask_u8.scalar_type() == torch::kUInt8 && mask_u8.dim() == 3 && mask_u8.stride(2) == 1 && mask_u8.is_cuda(), "mask [B,N,S] uint8, key stride 1, CUDA");
  TORCH_CHECK(census.scalar_type() == torch::kInt32 && census.is_contiguous() && census.numel() >= 4 && census.is_cuda(), "census int32[>=4] CUDA");
  c10::cuda::CUDAGuard guard(mask_u8.device());
  const int B = mask_u8.size(0), N = mask_u8.size(1), S = mask_u8.size(2), S64 = (S + 63) / 64 * 64, nkt = (S + BN - 1) / BN;
  const int YG = (N + (int)R - 1) / (int)R;
  auto opt8 = mask_u8.options(); auto opt32 = mask_u8.options().dtype(torch::kInt32);
  auto keyany = torch::empty({B, S64}, opt8);
  auto rows = torch::empty({(long long)B * N, 4}, opt32);
  auto maskw = torch::empty({(long long)B * N * nkt * 2}, opt32);
  auto rgflag = torch::zeros({(long long)B * YG}, opt32);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  mask_or_kernel<><<<dim3((S64 + 127) / 128, B), 1024, 0, st>>>(mask_u8.data_ptr<uint8_t>(), mask_u8.stride(0), mask_u8.stride(1), keyany.data_ptr<uint8_t>(), N, S, S64);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const long long nrows = (long long)B * N; const int wpb = 8;
  mask_rows_kernel<><<<(unsigned)((nrows + wpb - 1) / wpb), wpb * 32, 0, st>>>(mask_u8.data_ptr<uint8_t>(), mask_u8.stride(0), mask_u8.stride(1), keyany.data_ptr<uint8_t>(),
      reinterpret_cast<int4*>(rows.data_ptr<int>()), reinterpret_cast<uint32_t*>(maskw.data_ptr<int>()), census.data_ptr<int>(), rgflag.data_ptr<int>(), B, N, S, S64, nkt, (int)R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int nflags = B * YG;
  count_flags_kernel<><<<std::max(1, std::min((nflags + 255) / 256, 64)), 256, 0, st>>>(rgflag.data_ptr<int>(), nflags, census.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {keyany, rows, maskw};
}

void fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias_staged, c10::optional<torch::Tensor> rows, c10::optional<torch::Tensor> maskw,
         double scale, torch::Tensor out, torch::Tensor fix, c10::optional<torch::Tensor> lse, int64_t dbg) {
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.dim() == 5 && q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16 && v.scalar_type() == torch::kBFloat16, "q/k/v: [B,N,H,S,D] bf16");
  const int B = q.size(0), N = q.size(1), H = q.size(2), S = q.size(3), D = q.size(4);
  const Inst& in = need_inst(D);
  TORCH_CHECK(k.dim() == 5 && v.dim() == 5 && k.size(0) == B && k.size(1) == N && k.size(2) == H && k.size(4) == D && v.sizes() == k.sizes(), "k/v shape");
  TORCH_CHECK(k.size(3) == S, "S_q != S_kv");
  TORCH_CHECK(S >= 1, "S >= 1");
  for (auto* t : {&q, &k, &v}) {
    TORCH_CHECK(t->stride(4) == 1, "d stride must be 1");
    for (int d = 0; d < 4; ++d) TORCH_CHECK(t->stride(d) % 8 == 0 && t->stride(d) >= 0, "q/k/v strides: non-negative multiples of 8 elements");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "q/k/v 16-byte alignment");
  }
  const int S64 = (S + 63) / 64 * 64, S128 = (S + 127) / 128 * 128;
  TORCH_CHECK(bias_staged.scalar_type() == torch::kFloat32 && bias_staged.is_contiguous() && bias_staged.dim() == 4 &&
              bias_staged.size(0) == B && bias_staged.size(1) == H && bias_staged.size(2) == S128 && bias_staged.size(3) == S64, "staged bias shape [B,H,S128,S64] fp32 contiguous");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 5 && out.size(0) == B && out.size(1) == N && out.size(2) == H && out.size(3) == S && out.size(4) == D &&
              out.stride(4) == 1 && out.stride(3) % 2 == 0 && out.stride(2) % 2 == 0 && out.stride(1) % 2 == 0 && out.stride(0) % 2 == 0, "out layout [B,N,H,S,D] bf16, d stride 1, even strides");
  Args a{};
  a.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); a.q_sB = q.stride(0); a.q_sN = q.stride(1); a.q_sH = q.stride(2); a.q_sS = q.stride(3);
  a.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()); a.k_sB = k.stride(0); a.k_sN = k.stride(1); a.k_sH = k.stride(2); a.k_sS = k.stride(3);
  a.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()); a.v_sB = v.stride(0); a.v_sN = v.stride(1); a.v_sH = v.stride(2); a.v_sS = v.stride(3);
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); a.o_sB = out.stride(0); a.o_sN = out.stride(1); a.o_sH = out.stride(2); a.o_sS = out.stride(3);
  a.bias = bias_staged.data_ptr<float>();
  a.rows = nullptr; a.maskw = nullptr; a.lse = nullptr;
  const int nkt = (S + BN - 1) / BN;
  if (rows.has_value()) {
    TORCH_CHECK(maskw.has_value(), "rows without maskw");
    TORCH_CHECK(rows->scalar_type() == torch::kInt32 && rows->is_contiguous() && rows->numel() == (long long)B * N * 4, "rows [B*N,4] int32");
    TORCH_CHECK(maskw->scalar_type() == torch::kInt32 && maskw->is_contiguous() && maskw->numel() == (long long)B * N * nkt * 2, "maskw [B*N*nkt*2] int32");
    a.rows = reinterpret_cast<const int4*>(rows->data_ptr<int>()); a.maskw = reinterpret_cast<const uint32_t*>(maskw->data_ptr<int>());
  }
  if (lse.has_value()) {
    TORCH_CHECK(lse->scalar_type() == torch::kFloat32 && lse->is_contiguous() && lse->numel() == (long long)B * N * H * S, "lse [B,N,H,S] fp32 contiguous");
    a.lse = lse->data_ptr<float>();
  }
  const bool dense_rows = q.stride(3) <= 512 && k.stride(3) <= 512 && v.stride(3) <= 512;
  bool small = (in.small_max > 0 && S <= in.small_max) || (dense_rows && in.small_max_rows > 0 && S <= in.small_max_rows);
  if (dbg & 2048) small = false; else if (dbg & 4096) small = true;   // experiments: force the large-S / small-S geometry
  const GeoInfo& g = small ? in.small : in.big;
  const long long nct = cta_tiles(g, B, N, H, S);
  TORCH_CHECK(fix.scalar_type() == torch::kInt32 && fix.is_contiguous() && fix.numel() >= FIX_LIST + 3 * nct, "fix buffer: int32 [>= ", FIX_LIST + 3 * nct, "]");
  a.fix = fix.data_ptr<int>();
  a.B = B; a.N = N; a.H = H; a.S_q = S; a.S_kv = S;
  a.qb = (int)(dbg >> 16);                                       // 0: default block size; experiments: qb override
  a.c1 = (float)(scale * 1.4426950408889634);
  a.dbg = (int)(dbg & 2047);
  const int variant = lse.has_value() ? 2 : (rows.has_value() ? 1 : ((dbg & 512) ? 1 : 0));   // (dbg 512: general instantiation on an unmasked call)
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const char* stage = "";
  const cudaError_t e = in.run(a, variant, small, st, &stage);
  TORCH_CHECK(e == cudaSuccess, "triattn_sm80 ", stage, ": ", cudaGetErrorString(e), " (grid ", (S + 32 * g.QG - 1) / (32 * g.QG), "x", (N + g.R - 1) / g.R, "x", B * H,
              ", block ", 32 * (g.R / g.RW) * g.QG, ", smem ", g.smem_hot, "/", g.smem_safe, small ? ", small-S geometry)" : ")");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd, "triangle attention forward (sm_80): q,k,v,bias_staged,rows?,maskw?,scale,out,fix,lse?,dbg");
  m.def("stage_bias", &stage_bias, "bias [B,H,S,S] (+ keyany) -> fp32 [B,H,S128,S64] / scale in C-fragment order, -inf on excluded keys");
  m.def("stage_mask", &stage_mask, "mask [B,N,S] uint8 -> (keyany [B,S64], rows [B*N,4] int32, maskw [B*N*nkt*2] int32); census += (ragged rows, uniform rows, ragged row groups)");
  m.def("geometry", &geometry, "(R, QG, STAGES, WARPS, BM, smem_hot, smem_safe, RW, small_max, small_max_rows) for head dim D (small: the small-S geometry)", pybind11::arg("D"), pybind11::arg("small") = false);
  m.def("fix_elems", &fix_buffer_elems, "int32 entries of the fix buffer fwd() needs for (B, N, H, S, D)");
  m.def("dims", &dims_built, "head dims compiled into this build");
}
