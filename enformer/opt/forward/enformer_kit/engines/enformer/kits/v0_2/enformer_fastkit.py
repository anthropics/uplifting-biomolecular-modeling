"""enformer_fastkit — the module levers of the Enformer exact kit (upstream enformer-pytorch 0.8.12), each reproducing the stock's
arithmetic so the outputs are byte-identical:

  poscache  the relative-position basis + its projection (``to_rel_k``) computed once per (layer, length) and kept on the device — the stock
            recomputes the basis, copies the TF-gamma table host->device and re-projects it inside every Attention.forward
  xattn     the relative-position attention chain of each layer as one exact kernel sequence (enformer_xattn.py / enformer_xattn.cu)
  fused     the trunk's elementwise passes as five exact kernels (_CUDA_SRC below, prebuilt as enformer_fastkit_fused.so): BatchNorm+GELU;
            conv bias + BatchNorm + GELU with the biased tensor kept for the residual; conv bias + residual add; the AttentionPool tail; the
            AttentionPool tail + the next stage's BatchNorm+GELU. Convolutions stay the stock cuDNN calls (issued without their bias, which
            ATen adds in a separate pass anyway); every fused kernel performs the same fp32 operations in the same order as the modules it
            replaces, only without the round trips through memory
  (graph)   the trunk CUDA graph is kits/v0's KitV0 / TrunkGraph, applied on top of these patches

``FastKit(model)`` applies the patches (class-level on Attention / AttentionPool, module swaps inside the ConvBlocks, instance forwards on
``stem`` / ``conv_tower``), refcounted per model and fully reversible (``close``)."""
from __future__ import annotations

import hashlib
import os

import torch
from einops import rearrange

from enformer_pytorch.modeling_enformer import Attention, AttentionPool, GELU, Residual, get_positional_embed, relative_shift

HEADS = ("human", "mouse")
SEQ_LEN = 196_608

# ----------------------------------------------------------------------------------------- lever: poscache
_STOCK_ATTENTION_FORWARD = Attention.forward


def _attention_forward_cached(self, x):
    n, h, device = x.shape[-2], self.heads, x.device
    q = self.to_q(x); k = self.to_k(x); v = self.to_v(x)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
    q = q * self.scale
    content_logits = torch.einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)
    key = (n, str(device), str(self.to_rel_k.weight.dtype))
    cache = getattr(self, "_fastkit_relk", None)
    if cache is None or cache[0] != key:
        positions = get_positional_embed(n, self.num_rel_pos_features, device, use_tf_gamma=self.use_tf_gamma,
                                         dtype=self.to_rel_k.weight.dtype)
        positions = self.pos_dropout(positions)          # byte-equality in eval; kept so the semantics are the stock's
        rel_k = rearrange(self.to_rel_k(positions), 'n (h d) -> h n d', h=h)
        self._fastkit_relk = (key, rel_k)
    rel_logits = torch.einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, self._fastkit_relk[1])
    rel_logits = relative_shift(rel_logits)
    attn = self.attn_dropout((content_logits + rel_logits).softmax(dim=-1))
    out = rearrange(torch.einsum('b h i j, b h j d -> b h i d', attn, v), 'b h n d -> b n (h d)')
    return self.to_out(out)


def apply_poscache(model, enable: bool = True):
    """Switch the Attention forward (class-level; the model's 11 blocks share it)."""
    if not model.training and enable:
        Attention.forward = _attention_forward_cached
    else:
        Attention.forward = _STOCK_ATTENTION_FORWARD
        for m in model.modules():
            if isinstance(m, Attention) and hasattr(m, "_fastkit_relk"):
                del m._fastkit_relk


# ----------------------------------------------------------------------------------------- lever: fused kernels
_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
// Every kernel below reproduces, element by element, the fp32 operation sequence of the stock modules it replaces (ATen / cuDNN):
//   conv bias:      x1 = y + b[c]                                   (ATen add: one fp32 add, round-to-nearest)
//   BatchNorm eval: v  = fma(g[c] * (x1 - m[c]), inv[c], beta[c])   (cuDNN bn_fw_inf arithmetic; inv = rsqrt(var + eps) precomputed as torch does)
//   GELU (upstream): 1.702*x -> sigmoid -> *x                       (ATen: each op rounded once; sigmoid = 1/(1+expf(-t)))
//   Residual:       x2 = r + x1 with r = y1 + b1[c]                 (two fp32 adds in the stock's order)
//   AttentionPool:  softmax over the pair of logits, x*p summed      (ATen softmax on 2 elements, mul, sum(dim=-1) of 2 = one add)
// Fusing changes only WHERE intermediate values live (registers instead of a round trip through memory), never an operation or its order,
// so outputs are byte-identical to the unfused stock sequence at any shape. float4 paths require L % 4 == 0 (rows 16-byte aligned); the
// launchers fall back to the scalar kernels otherwise.
__device__ __forceinline__ float gelu_chain(float x) {
    float t = __fmul_rn(1.702f, x);
    float e = expf(-t);
    float s = __fdiv_rn(1.0f, __fadd_rn(1.0f, e));
    return __fmul_rn(s, x);
}
__device__ __forceinline__ float bn_gelu_one(float x, float inv, float m, float beta, float g) {   // BatchNorm(eval) then GELU
    return gelu_chain(__fmaf_rn(__fmul_rn(g, __fsub_rn(x, m)), inv, beta));
}
__device__ __forceinline__ float pool_one(float x0, float x1, float l0, float l1) {
    float mx = fmaxf(l0, l1);
    float e0 = expf(__fsub_rn(l0, mx)), e1 = expf(__fsub_rn(l1, mx));
    float sum = __fadd_rn(e0, e1);
    float p0 = __fdiv_rn(e0, sum), p1 = __fdiv_rn(e1, sum);
    return __fadd_rn(__fmul_rn(x0, p0), __fmul_rn(x1, p1));
}
// ---------------------------------------------------------------- scalar (any shape) kernels
__global__ void bn_gelu_kernel(const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ inv,
                               const float* __restrict__ m, const float* __restrict__ beta, const float* __restrict__ g,
                               long long C, long long L, long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < N; i += stride) { long long c = (i / L) % C; y[i] = bn_gelu_one(x[i], inv[c], m[c], beta[c], g[c]); }
}
__global__ void pool_kernel(const float* __restrict__ x, const float* __restrict__ lg, float* __restrict__ y, long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < N; i += stride) y[i] = pool_one(x[2*i], x[2*i+1], lg[2*i], lg[2*i+1]);
}
__global__ void bias_bn_gelu_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ x1, float* __restrict__ y,
                                    const float* __restrict__ inv, const float* __restrict__ m, const float* __restrict__ beta, const float* __restrict__ g,
                                    long long C, long long L, long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < N; i += stride) {
        long long c = (i / L) % C;
        float v = __fadd_rn(x[i], bias[c]); x1[i] = v;
        y[i] = bn_gelu_one(v, inv[c], m[c], beta[c], g[c]);
    }
}
__global__ void bias_residual_kernel(const float* __restrict__ x1, const float* __restrict__ y1, const float* __restrict__ bias, float* __restrict__ out,
                                     long long C, long long L, long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < N; i += stride) { long long c = (i / L) % C; out[i] = __fadd_rn(__fadd_rn(y1[i], bias[c]), x1[i]); }
}
__global__ void pool_bn_gelu_kernel(const float* __restrict__ x, const float* __restrict__ lg, float* __restrict__ y,
                                    const float* __restrict__ inv, const float* __restrict__ m, const float* __restrict__ beta, const float* __restrict__ g,
                                    long long C, long long Lout, long long N) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < N; i += stride) {
        long long c = (i / Lout) % C;
        y[i] = bn_gelu_one(pool_one(x[2*i], x[2*i+1], lg[2*i], lg[2*i+1]), inv[c], m[c], beta[c], g[c]);
    }
}
// ---------------------------------------------------------------- row-vectorized kernels: grid = (rows = B*C, column blocks); one channel per row
template <bool BIAS>
__global__ void bn_gelu_rows(const float4* __restrict__ x, const float* __restrict__ bias, float4* __restrict__ x1, float4* __restrict__ y,
                             const float* __restrict__ inv, const float* __restrict__ m, const float* __restrict__ beta, const float* __restrict__ g,
                             int C, long long L4) {
    long long row = blockIdx.x; int c = (int)(row % C);
    const float ci = inv[c], cm = m[c], cb = beta[c], cg = g[c]; const float bb = BIAS ? bias[c] : 0.f;
    const float4* xr = x + row * L4; float4* yr = y + row * L4; float4* x1r = BIAS ? x1 + row * L4 : nullptr;
    for (long long j = (long long)blockIdx.y * blockDim.x + threadIdx.x; j < L4; j += (long long)gridDim.y * blockDim.x) {
        float4 v = xr[j];
        if (BIAS) { v.x = __fadd_rn(v.x, bb); v.y = __fadd_rn(v.y, bb); v.z = __fadd_rn(v.z, bb); v.w = __fadd_rn(v.w, bb); x1r[j] = v; }
        float4 o;
        o.x = bn_gelu_one(v.x, ci, cm, cb, cg); o.y = bn_gelu_one(v.y, ci, cm, cb, cg); o.z = bn_gelu_one(v.z, ci, cm, cb, cg); o.w = bn_gelu_one(v.w, ci, cm, cb, cg);
        yr[j] = o;
    }
}
__global__ void bias_residual_rows(const float4* __restrict__ x1, const float4* __restrict__ y1, const float* __restrict__ bias, float4* __restrict__ out, int C, long long L4) {
    long long row = blockIdx.x; int c = (int)(row % C); const float bb = bias[c];
    const float4* a = x1 + row * L4; const float4* b = y1 + row * L4; float4* o = out + row * L4;
    for (long long j = (long long)blockIdx.y * blockDim.x + threadIdx.x; j < L4; j += (long long)gridDim.y * blockDim.x) {
        float4 u = a[j], w = b[j], r;
        r.x = __fadd_rn(__fadd_rn(w.x, bb), u.x); r.y = __fadd_rn(__fadd_rn(w.y, bb), u.y); r.z = __fadd_rn(__fadd_rn(w.z, bb), u.z); r.w = __fadd_rn(__fadd_rn(w.w, bb), u.w);
        o[j] = r;
    }
}
// pool over pairs: input row = 2*Lout floats (Lout pairs), logits alike; four pairs (two float4 of x, two of lg) -> four outputs (one float4)
template <bool BNGELU>
__global__ void pool_rows(const float4* __restrict__ x, const float4* __restrict__ lg, float4* __restrict__ y,
                          const float* __restrict__ inv, const float* __restrict__ m, const float* __restrict__ beta, const float* __restrict__ g,
                          int C, long long L4) {   // L4 = Lout / 4 = float4 outputs per row; the input row holds 2 * L4 float4
    long long row = blockIdx.x; int c = (int)(row % C);
    float ci = 0.f, cm = 0.f, cb = 0.f, cg = 0.f; if (BNGELU) { ci = inv[c]; cm = m[c]; cb = beta[c]; cg = g[c]; }
    const float4* xr = x + row * 2 * L4; const float4* lr = lg + row * 2 * L4; float4* yr = y + row * L4;
    for (long long j = (long long)blockIdx.y * blockDim.x + threadIdx.x; j < L4; j += (long long)gridDim.y * blockDim.x) {
        float4 v0 = xr[2 * j], v1 = xr[2 * j + 1], l0 = lr[2 * j], l1 = lr[2 * j + 1]; float4 o;
        o.x = pool_one(v0.x, v0.y, l0.x, l0.y); o.y = pool_one(v0.z, v0.w, l0.z, l0.w);
        o.z = pool_one(v1.x, v1.y, l1.x, l1.y); o.w = pool_one(v1.z, v1.w, l1.z, l1.w);
        if (BNGELU) { o.x = bn_gelu_one(o.x, ci, cm, cb, cg); o.y = bn_gelu_one(o.y, ci, cm, cb, cg); o.z = bn_gelu_one(o.z, ci, cm, cb, cg); o.w = bn_gelu_one(o.w, ci, cm, cb, cg); }
        yr[j] = o;
    }
}
static inline int nblocks(long long N) { long long b = (N + 255) / 256; return (int)(b > 65535 * 8 ? 65535 * 8 : b); }
static inline dim3 rows_grid(long long rows, long long per_row_vec) { long long cb = (per_row_vec + 255) / 256; if (cb > 65535) cb = 65535; if (cb < 1) cb = 1; return dim3((unsigned)rows, (unsigned)cb); }
static inline bool vec_ok(const torch::Tensor& t, long long L) { return (L % 4 == 0) && (reinterpret_cast<uintptr_t>(t.data_ptr<float>()) % 16 == 0); }
#define F4(t) reinterpret_cast<float4*>((t).data_ptr<float>())
#define CF4(t) reinterpret_cast<const float4*>((t).data_ptr<float>())
static void check3(const torch::Tensor& x) { TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.is_cuda() && x.dim() == 3, "expected a (B, C, L) float32 CUDA tensor"); }

torch::Tensor bn_gelu(torch::Tensor x, torch::Tensor inv, torch::Tensor m, torch::Tensor beta, torch::Tensor g) {
    check3(x); x = x.contiguous(); auto y = torch::empty_like(x);
    long long B = x.size(0), C = x.size(1), L = x.size(2), N = B * C * L; auto st = at::cuda::getCurrentCUDAStream();
    if (vec_ok(x, L) && vec_ok(y, L) && B * C < 2147483647LL)
        bn_gelu_rows<false><<<rows_grid(B * C, L / 4), 256, 0, st>>>(CF4(x), nullptr, nullptr, F4(y), inv.data_ptr<float>(), m.data_ptr<float>(), beta.data_ptr<float>(), g.data_ptr<float>(), (int)C, L / 4);
    else bn_gelu_kernel<<<nblocks(N), 256, 0, st>>>(x.data_ptr<float>(), y.data_ptr<float>(), inv.data_ptr<float>(), m.data_ptr<float>(), beta.data_ptr<float>(), g.data_ptr<float>(), C, L, N);
    return y;
}
// (x1, act) = (y + bias[c], gelu(bn(y + bias[c]))) — the conv's bias add and the next ConvBlock's BatchNorm+GELU, the biased tensor kept for the residual
std::vector<torch::Tensor> bias_bn_gelu(torch::Tensor y, torch::Tensor bias, torch::Tensor inv, torch::Tensor m, torch::Tensor beta, torch::Tensor g) {
    check3(y); y = y.contiguous(); auto x1 = torch::empty_like(y); auto act = torch::empty_like(y);
    long long B = y.size(0), C = y.size(1), L = y.size(2), N = B * C * L; auto st = at::cuda::getCurrentCUDAStream();
    if (vec_ok(y, L) && vec_ok(x1, L) && vec_ok(act, L) && B * C < 2147483647LL)
        bn_gelu_rows<true><<<rows_grid(B * C, L / 4), 256, 0, st>>>(CF4(y), bias.data_ptr<float>(), F4(x1), F4(act), inv.data_ptr<float>(), m.data_ptr<float>(), beta.data_ptr<float>(), g.data_ptr<float>(), (int)C, L / 4);
    else bias_bn_gelu_kernel<<<nblocks(N), 256, 0, st>>>(y.data_ptr<float>(), bias.data_ptr<float>(), x1.data_ptr<float>(), act.data_ptr<float>(), inv.data_ptr<float>(), m.data_ptr<float>(), beta.data_ptr<float>(), g.data_ptr<float>(), C, L, N);
    return {x1, act};
}
// x2 = (y1 + bias[c]) + x1 — the pointwise conv's bias add and the Residual add
torch::Tensor bias_residual(torch::Tensor x1, torch::Tensor y1, torch::Tensor bias) {
    check3(x1); check3(y1); TORCH_CHECK(x1.sizes() == y1.sizes()); x1 = x1.contiguous(); y1 = y1.contiguous(); auto out = torch::empty_like(x1);
    long long B = x1.size(0), C = x1.size(1), L = x1.size(2), N = B * C * L; auto st = at::cuda::getCurrentCUDAStream();
    if (vec_ok(x1, L) && vec_ok(y1, L) && vec_ok(out, L) && B * C < 2147483647LL)
        bias_residual_rows<<<rows_grid(B * C, L / 4), 256, 0, st>>>(CF4(x1), CF4(y1), bias.data_ptr<float>(), F4(out), (int)C, L / 4);
    else bias_residual_kernel<<<nblocks(N), 256, 0, st>>>(x1.data_ptr<float>(), y1.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), C, L, N);
    return out;
}
// AttentionPool tail: x, lg = (B, C, Lout, 2) -> (B, C, Lout); with BN params also applies the NEXT ConvBlock's BatchNorm+GELU to the pooled value
torch::Tensor pool_tail(torch::Tensor x, torch::Tensor lg) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.is_cuda() && x.size(-1) == 2);
    x = x.contiguous(); lg = lg.contiguous();
    auto sizes = x.sizes().vec(); sizes.pop_back(); auto y = torch::empty(sizes, x.options()); long long N = y.numel(); auto st = at::cuda::getCurrentCUDAStream();
    long long Lout = sizes.back(); long long rows = N / (Lout > 0 ? Lout : 1);
    if (x.dim() == 4 && Lout % 4 == 0 && vec_ok(x, 4) && vec_ok(lg, 4) && vec_ok(y, 4) && rows < 2147483647LL)
        pool_rows<false><<<rows_grid(rows, Lout / 4), 256, 0, st>>>(CF4(x), CF4(lg), F4(y), nullptr, nullptr, nullptr, nullptr, (int)x.size(1), Lout / 4);
    else pool_kernel<<<nblocks(N), 256, 0, st>>>(x.data_ptr<float>(), lg.data_ptr<float>(), y.data_ptr<float>(), N);
    return y;
}
torch::Tensor pool_bn_gelu(torch::Tensor x, torch::Tensor lg, torch::Tensor inv, torch::Tensor m, torch::Tensor beta, torch::Tensor g) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.is_cuda() && x.dim() == 4 && x.size(-1) == 2);
    x = x.contiguous(); lg = lg.contiguous();
    auto sizes = x.sizes().vec(); sizes.pop_back(); auto y = torch::empty(sizes, x.options()); long long N = y.numel(); auto st = at::cuda::getCurrentCUDAStream();
    long long C = x.size(1), Lout = x.size(2), rows = x.size(0) * C;
    if (Lout % 4 == 0 && vec_ok(x, 4) && vec_ok(lg, 4) && vec_ok(y, 4) && rows < 2147483647LL)
        pool_rows<true><<<rows_grid(rows, Lout / 4), 256, 0, st>>>(CF4(x), CF4(lg), F4(y), inv.data_ptr<float>(), m.data_ptr<float>(), beta.data_ptr<float>(), g.data_ptr<float>(), (int)C, Lout / 4);
    else pool_bn_gelu_kernel<<<nblocks(N), 256, 0, st>>>(x.data_ptr<float>(), lg.data_ptr<float>(), y.data_ptr<float>(), inv.data_ptr<float>(), m.data_ptr<float>(), beta.data_ptr<float>(), g.data_ptr<float>(), C, Lout, N);
    return y;
}
'''
_CPP_SRC = r'''
#include <torch/extension.h>
#include <vector>
torch::Tensor bn_gelu(torch::Tensor x, torch::Tensor inv, torch::Tensor m, torch::Tensor beta, torch::Tensor g);
torch::Tensor pool_tail(torch::Tensor x, torch::Tensor lg);
std::vector<torch::Tensor> bias_bn_gelu(torch::Tensor y, torch::Tensor bias, torch::Tensor inv, torch::Tensor m, torch::Tensor beta, torch::Tensor g);
torch::Tensor bias_residual(torch::Tensor x1, torch::Tensor y1, torch::Tensor bias);
torch::Tensor pool_bn_gelu(torch::Tensor x, torch::Tensor lg, torch::Tensor inv, torch::Tensor m, torch::Tensor beta, torch::Tensor g);
'''
FUSED_FUNCTIONS = ("bn_gelu", "pool_tail", "bias_bn_gelu", "bias_residual", "pool_bn_gelu")   # the extension's entry points (kits.v0_2.check_files requires them of a prebuilt object)
_EXT = None
FUSED_SO = None                     # the prebuilt fused-kernel object to load (kits.v0_2.attach sets it); None = build from the sources above with nvcc


def fused_ext():
    """The fused-kernel extension: the prebuilt object ``FUSED_SO`` (set by kits.v0_2.attach: the kit's own sm_90 object or a class
    build), else built here by nvcc from the sources above with PyTorch's default flags (no -use_fast_math, -ftz=false, -prec-div=true,
    -prec-sqrt=true)."""
    global _EXT
    if _EXT is None:
        so = FUSED_SO
        if so:
            import importlib.util
            spec = importlib.util.spec_from_file_location("enformer_fastkit_fused", so)
            _EXT = importlib.util.module_from_spec(spec); spec.loader.exec_module(_EXT)
        else:
            from torch.utils.cpp_extension import load_inline
            _EXT = load_inline(name="enformer_fastkit_fused", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                               functions=list(FUSED_FUNCTIONS), extra_cuda_cflags=["-O3"], verbose=False)
    return _EXT




class FusedBNGELU(torch.nn.Module):
    """BatchNorm1d (eval, cuDNN arithmetic) + GELU in one kernel; keeps the stock BatchNorm1d as `bn` for restore."""

    def __init__(self, bn: torch.nn.BatchNorm1d):
        super().__init__()
        assert not bn.training and bn.track_running_stats and bn.affine
        self.bn = bn
        with torch.no_grad():
            self.inv = torch.rsqrt(bn.running_var + bn.eps).contiguous()   # fp32, the rsqrtf cuDNN uses
            self.m, self.beta, self.g = (t.detach().contiguous() for t in (bn.running_mean, bn.bias, bn.weight))

    def forward(self, x):
        return fused_ext().bn_gelu(x, self.inv, self.m, self.beta, self.g)


_STOCK_POOL_FORWARD = AttentionPool.forward


def _pool_forward_fused(self, x):
    b, d, n = x.shape
    if n % self.pool_size:                      # the stock pads; never the case at 196,608 bp — fall back to stock
        return _STOCK_POOL_FORWARD(self, x)
    xr = x.contiguous().reshape(b, d, n // self.pool_size, self.pool_size)
    return fused_ext().pool_tail(xr, self.to_attn_logits(xr))


def _conv_nobias(conv, x):
    """The stock nn.Conv1d call minus its bias: ATen runs cudnn_convolution WITHOUT the bias and adds it in a separate elementwise pass
    (output.add_(bias)); the kernels above take over that add, so the convolution itself is the identical cuDNN call."""
    assert conv.padding_mode == "zeros"
    return torch.nn.functional.conv1d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)


def _bn_params(fbn):
    return fbn.inv, fbn.m, fbn.beta, fbn.g


def _is_stage(m) -> bool:
    """A trunk stage: Sequential(<Conv1d | ConvBlock>, Residual(ConvBlock k=1), AttentionPool) — the stem and each conv-tower layer."""
    return (isinstance(m, torch.nn.Sequential) and len(m) == 3 and isinstance(m[1], Residual) and isinstance(m[2], AttentionPool)
            and isinstance(m[1].fn, torch.nn.Sequential) and len(m[1].fn) == 3 and isinstance(m[1].fn[0], FusedBNGELU) and isinstance(m[1].fn[2], torch.nn.Conv1d)
            and (isinstance(m[0], torch.nn.Conv1d) or (isinstance(m[0], torch.nn.Sequential) and len(m[0]) == 3 and isinstance(m[0][0], FusedBNGELU) and isinstance(m[0][2], torch.nn.Conv1d))))


def _stage_forward(stage, x, pre_activated: bool, next_bn):
    """One trunk stage with the elementwise passes fused: conv (cuDNN, no bias) -> [bias + BN + GELU, the biased tensor kept] -> pointwise
    conv (cuDNN, no bias) -> [bias + residual add] -> attention-pool logits conv (cuDNN) -> [pool tail (+ the next stage's BN + GELU)].
    ``pre_activated``: x is already gelu(bn(input)) of this stage's first ConvBlock (emitted by the previous stage's pool); ``next_bn``: the
    next stage's first FusedBNGELU (its BN + GELU is applied to the pooled value), or None (the raw pooled tensor is returned)."""
    E = fused_ext()
    first, res, pool = stage[0], stage[1], stage[2]
    if isinstance(first, torch.nn.Conv1d):                              # the stem: a bare conv on the one-hot input
        conv = first
        y = _conv_nobias(conv, x)
    else:                                                                # ConvBlock: BN + GELU (unless the previous pool emitted it) -> conv
        conv = first[2]
        y = _conv_nobias(conv, x if pre_activated else first[0](x))
    del x                                                                # each intermediate is dropped as soon as it is consumed, as the stock Sequential drops it (same peak live set)
    cb1 = res.fn                                                         # Residual(ConvBlock k=1): BN + GELU -> pointwise conv, + input
    if conv.bias is not None:
        x1, act = E.bias_bn_gelu(y, conv.bias, *_bn_params(cb1[0]))
    else:
        x1, act = y, cb1[0](y)
    del y
    y1 = _conv_nobias(cb1[2], act)
    del act
    if cb1[2].bias is not None:
        x2 = E.bias_residual(x1, y1, cb1[2].bias)
    else:
        x2 = y1 + x1
    del x1, y1
    b, d, n = x2.shape
    if n % pool.pool_size:                                               # the stock pads an odd length; never the case at 196,608 bp
        out = _STOCK_POOL_FORWARD(pool, x2)
        return next_bn(out) if next_bn is not None else out
    xr = x2.reshape(b, d, n // pool.pool_size, pool.pool_size)
    del x2
    lg = pool.to_attn_logits(xr)                                         # the pool's 1x1 logits conv (cuDNN, no bias in the stock module)
    return E.pool_bn_gelu(xr, lg, *_bn_params(next_bn)) if next_bn is not None else E.pool_tail(xr, lg)


def _stem_forward_fused(self, x):
    nxt = _TOWER_OF.get(id(self))
    return _stage_forward(self, x, False, nxt[0][0][0] if nxt is not None else None)


def _tower_forward_fused(self, x):
    n = len(self)
    for k in range(n):
        x = _stage_forward(self[k], x, True, self[k + 1][0][0] if k + 1 < n else None)
    return x


_TOWER_OF = {}                                  # id(stem Sequential) -> the conv tower whose first BN + GELU the stem's pool emits


def apply_fused(model, enable: bool = True):
    """Swap every ConvBlock (BatchNorm1d, GELU) pair for the fused BN+GELU kernel, AttentionPool.forward for the fused tail, and — when the model
    has the stock stem / conv-tower structure — run each trunk stage through _stage_forward (the conv bias adds, the residual add and the pool
    fused into the neighbouring kernels; instance-level forwards on `stem` and `conv_tower`). enable=False restores the stock modules and forwards."""
    for m in model.modules():
        if isinstance(m, torch.nn.Sequential) and len(m) == 3 \
                and isinstance(m[0], (torch.nn.BatchNorm1d, FusedBNGELU)) and isinstance(m[1], (GELU, torch.nn.Identity)):
            stock_bn = m[0].bn if isinstance(m[0], FusedBNGELU) else m[0]
            if enable:
                m[0], m[1] = FusedBNGELU(stock_bn), torch.nn.Identity()
            else:
                m[0], m[1] = stock_bn, GELU()
    AttentionPool.forward = _pool_forward_fused if enable else _STOCK_POOL_FORWARD
    stem, tower = getattr(model, "stem", None), getattr(model, "conv_tower", None)
    staged = (enable and stem is not None and tower is not None and _is_stage(stem)
              and isinstance(tower, torch.nn.Sequential) and len(tower) > 0 and all(_is_stage(t) and not isinstance(t[0], torch.nn.Conv1d) for t in tower))
    for mod in (stem, tower):
        if mod is not None and "forward" in mod.__dict__:
            del mod.__dict__["forward"]
    if stem is not None:
        _TOWER_OF.pop(id(stem), None)
    if staged:
        import types
        _TOWER_OF[id(stem)] = tower
        stem.forward = types.MethodType(_stem_forward_fused, stem)
        tower.forward = types.MethodType(_tower_forward_fused, tower)


_STOCK_POOL_FORWARD = AttentionPool.forward




# ----------------------------------------------------------------------------------------- lever: exact attention (xattn)

def _xattn_module():
    import enformer_xattn as XA
    XA.ORDER["order"] = 0; XA.SOFTMAX["fn"] = "add2_softmax_warp"      # the orders named on the image (jobs 43/46)
    return XA

def apply_xattn(model, enable: bool = True):
    XA = _xattn_module() if enable else None
    for m in model.modules():
        if isinstance(m, Attention):
            if enable: m.forward = (lambda self: (lambda x: XA.exact_forward(self, x)))(m)
            elif "forward" in m.__dict__: del m.__dict__["forward"]









_ACTIVE, _ACTIVE_LEVERS = {}, {}     # id(model) -> number of live kits / the patch set they share


LEVER_CLASS = {"poscache": "bitwise by construction (the same tensors, computed once)",
               "fused": "bitwise by reproduced arithmetic (BN+GELU prologue, AttentionPool tail; H100 and A100)",
               "xattn": "bitwise by reproduced reduction order (rel band GEMM + softmax; H100 and A100)"}


class FastKit:
    """The module patches of ``levers`` applied to an eval-mode ``model`` (poscache: class-level Attention.forward; fused: the ConvBlock
    (BatchNorm1d, GELU) pairs and AttentionPool.forward; xattn: an instance forward on every Attention). Several handles may share one
    model: the patches are applied once and refcounted; ``close()`` restores the stock modules when the last handle closes."""

    def __init__(self, model, levers=("poscache", "fused", "xattn"), device="cuda"):
        assert not model.training, "eval mode only (BatchNorm running stats, dropout byte-equality)"
        self.model, self.levers, self.device = model, set(levers), torch.device(device)
        unknown = self.levers - set(LEVER_CLASS)
        if unknown:
            raise ValueError(f"enformer_fastkit: unknown levers {sorted(unknown)}")
        n = _ACTIVE.get(id(model), 0)
        if n == 0:
            apply_poscache(model, "poscache" in self.levers)
            apply_fused(model, "fused" in self.levers)
            apply_xattn(model, "xattn" in self.levers)                                   # instance patch: supersedes poscache's class patch
            _ACTIVE_LEVERS[id(model)] = frozenset(self.levers)
        elif _ACTIVE_LEVERS[id(model)] != frozenset(self.levers):
            raise ValueError(f"enformer_fastkit: model already patched with {sorted(_ACTIVE_LEVERS[id(model)])}; close() those kits first")
        _ACTIVE[id(model)] = n + 1

    def forward_device(self, x):
        """The patched forward, eager: device outputs {head: (b, 896, n)} for a (b, SEQ_LEN, 4) float32 input."""
        if x.ndim != 3 or tuple(x.shape[1:]) != (SEQ_LEN, 4) or x.dtype != torch.float32:
            raise ValueError(f"enformer_fastkit: input must be (b, {SEQ_LEN}, 4) float32, got {tuple(x.shape)} {x.dtype}")
        with torch.no_grad():
            return self.model(x)

    def stamp(self) -> dict:
        """The record of this handle: levers, module and extension bytes, stack."""
        st = {"kit": "enformer_fastkit", "levers": sorted(self.levers), "lever_class": {l: LEVER_CLASS[l] for l in sorted(self.levers)},
              "module_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "gpu": torch.cuda.get_device_name(self.device), "sm": ".".join(map(str, torch.cuda.get_device_capability(self.device)))}
        if "fused" in self.levers:
            st["fused_so"] = FUSED_SO; st["fused_so_sha256"] = hashlib.sha256(open(FUSED_SO, "rb").read()).hexdigest() if FUSED_SO and os.path.exists(FUSED_SO) else None
        if "xattn" in self.levers:
            import enformer_xattn as XA
            st["xattn_so"] = XA.SO_PATH; st["xattn_so_sha256"] = hashlib.sha256(open(XA.SO_PATH, "rb").read()).hexdigest() if XA.SO_PATH and os.path.exists(XA.SO_PATH) else None
        return st

    def close(self):
        n = _ACTIVE.get(id(self.model), 0)
        if n <= 1:
            apply_poscache(self.model, False); apply_fused(self.model, False); apply_xattn(self.model, False)
            _ACTIVE.pop(id(self.model), None); _ACTIVE_LEVERS.pop(id(self.model), None)
        else:
            _ACTIVE[id(self.model)] = n - 1
