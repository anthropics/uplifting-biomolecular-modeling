"""P5 `fusedattn`: gather-fused column cross-attention -- the attention reads the REDUCED clade key/value rows through the gather
index; the full-size (B, L, C, Ah) key and value tensors are never materialised.

What stock computes.  GPNStarColCrossAttention runs F.scaled_dot_product_attention on the MATH backend with a (B, L, A, 1, D)
query against (B, L, A, C, D) keys/values (A column heads, D = head size, C clades; Lq = 1 per token).  ATen's math path does,
in this order: q*sqrt(scale); k^T*sqrt(scale) (a full-size elementwise pass), then at::matmul -> reshape copies of k^T and v to
(B*L*A, D, C) / (B*L*A, C, D) and two cuBLAS batched GEMMs with n = 1, i.e. batched GEMVs; += mask; _safe_softmax; second GEMV.
On the pinned stack (torch 2.13 + cu130) the batched GEMV of these shapes is `gemv2N_kernel<..., 128, 1, 2, 4, 1>`: every output
is ONE sequential fp32 FMA chain over the reduction index in increasing order, starting from +0.0 -- measured by exact probes
(the summation tree decoded from absorbed-unit patterns) and confirmed bit for bit on the real operands.  cuBLAS issues that
kernel in chunks of 65535 batch matrices; the remainder (nb mod 65535 matrices at the END of the batch) goes through its
small-batch heuristic, which picks other kernels with other (fixed, decoded) summation trees: two contiguous halves
(`gemvx`), 4-wide lanes folded by a shuffle butterfly (`gemvNSP`), two strided partials, four contiguous quarters, ...

What this lever does.  Two small CUDA kernels (compiled at first use with NVRTC through torch's own driver bindings; no nvcc,
no extension build) compute scores[b,l,a,c] = sum_d q_s[b,l,a,d] * k_s[idx[b,l,c], a, d] and context[b,l,a,d] = sum_c p[b,l,a,c]
* v[idx[b,l,c], a, d] with exactly those FMA chains, reading the reduced rows k_s = key(X)*sqrt(scale) and v = value(X) (X = the
unified / de-duplicated clade rows of lever `unifiedkv`/`dedup`) through the index; the remainder matrices are computed by a
third kernel under a named summation tree.  Mask add and _safe_softmax stay the stock ATen ops on the (small) score tensor; the
context is written directly in the (B, L, 1, Ah) layout stock produces by transpose + contiguous.

Guard (declared, per batch shape, before first use): the first forward of a batch geometry (B, L) that reaches this route
computes, at the first layer that gets there, the MATERIALISED reference of that layer on the real data (gather -> the stock SDPA
math ops) and accepts the fused route only if (i) the fused scores equal at::matmul's bit for bit on the whole tensor, the
remainder tree being chosen among the decoded candidates by that same equality, (ii) likewise the fused context given the
reference probabilities, and (iii) the fused forward's context equals F.scaled_dot_product_attention's output (torch.equal).
Otherwise the shape runs the materialised route of `unifiedkv` (exact, by name, warned once).  No tolerance anywhere.
"""
from __future__ import annotations

import ctypes
import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .patches import ExactState, _sdpa_math, _state, core_model

CHUNK = 65535  # cuBLAS issues the batched GEMV kernel over at most this many matrices per launch; the remainder is a separate call

# candidate summation trees of the remainder call, (name, mode, P): tried in this order, accepted only on whole-tensor equality
TAIL_CANDIDATES = (
    ("seq", 0, 1),          # gemv2N: one chain (== no special tail)
    ("half_seq", 1, 2),     # gemvx: two contiguous halves of ceil(k/2), added first+second
    ("quadfly", 3, 4),      # gemvNSP: lane t sums elements 4t..4t+3, shuffle-down butterfly 16,8,4,2,1
    ("stride2", 4, 2),      # gemvx (about 1024 matrices): two strided partials (even/odd), added
    ("chunk4_seq", 1, 4),   # gemv2N (about 512 matrices): four contiguous quarters, added in order
    ("chunk8_fly", 2, 8),   # gemvx (about 64 matrices): eight contiguous chunks, butterfly
    ("chunk4_fly", 2, 4), ("chunk3_seq", 1, 3), ("chunk2_fly", 2, 2), ("stride4", 4, 4), ("stride8", 4, 8), ("fly_vec2", 3, 2),
)

_SRC = r"""
// ---- main kernels: every dot product = one sequential fp32 FMA chain from +0.0 (cuBLAS gemv2N<...,128,1,2,4,1>) --------------
extern "C" __global__ void fq_qk_simple(const float* __restrict__ q, const float* __restrict__ ku, const long long* __restrict__ idx,
                                      const float* __restrict__ mask, float* __restrict__ out, int C, int A, int HD, int L, int tail_start, double s_d)
{   // (fallback geometry) one block per (b,l), one thread per (head, clade) chain reading its key row segment itself; q (nBL*A, HD) raw query; ku (rows, A*HD) raw keys; idx (nBL, C); mask (B, A, C) or null; out (nBL*A, C)
    // scores = sum_d (k*s)(q*s) as one FMA chain from +0 (cuBLAS gemv2N), then + mask (SDPA-math's attn.add_): the same fp32 roundings
    extern __shared__ long long idx_s[];
    const int bl = blockIdx.x;
    const float s = (float)s_d;                       // at::mul casts the double scalar to float the same way
    for (int c = threadIdx.x; c < C; c += blockDim.x) idx_s[c] = idx[(long long)bl * C + c];
    __syncthreads();
    const int AH = A * HD, nv = HD >> 2, b = (L > 0) ? bl / L : 0;
    for (int w = threadIdx.x; w < A * C; w += blockDim.x) {
        const int c = w / A, a = w - c * A;
        const long long ib = (long long)bl * A + a;
        if (ib >= (long long)tail_start) continue;
        const float4* kp = reinterpret_cast<const float4*>(ku + idx_s[c] * AH + a * HD);
        const float4* qp = reinterpret_cast<const float4*>(q + ib * HD);
        float acc = 0.f;
        for (int i = 0; i < nv; ++i) {
            const float4 kv = __ldg(kp + i); const float4 qv = __ldg(qp + i);
            acc = fmaf(__fmul_rn(kv.x, s), __fmul_rn(qv.x, s), acc); acc = fmaf(__fmul_rn(kv.y, s), __fmul_rn(qv.y, s), acc);
            acc = fmaf(__fmul_rn(kv.z, s), __fmul_rn(qv.z, s), acc); acc = fmaf(__fmul_rn(kv.w, s), __fmul_rn(qv.w, s), acc);
        }
        out[ib * C + c] = (L >= 0) ? __fadd_rn(acc, __ldg(mask + ((long long)b * A + a) * C + c)) : acc;
    }
}
extern "C" __global__ void fq_qk_main(const float* __restrict__ q, const float* __restrict__ ku, const long long* __restrict__ idx,
                                      const float* __restrict__ mask, float* __restrict__ out, int C, int A, int HD, int L, int tail_start, double s_d)
{   // one block per (b,l): the clade key rows are staged through shared memory RC = blockDim/A rows at a time with coalesced 16-byte loads
    // (a row is A*HD contiguous floats), then thread (a = t % A, cc = t / A) runs the chain of (head a, clade c0+cc) out of shared memory:
    // scores = sum_d (k*s)(q*s), one FMA chain from +0 in increasing d (cuBLAS gemv2N), then + mask (SDPA-math's attn.add_).  Rows are
    // stored with a per-head pad of 4 floats so the A heads of one row sit in different banks.
    extern __shared__ float smf[];
    const int T = blockDim.x, RC = T / A, HP = HD + 4, rowf = A * HP, AH = A * HD, nv = HD >> 2;
    float* qsm = smf + RC * rowf;                                        // [A][HP]: this token's query, all heads
    long long* idx_s = reinterpret_cast<long long*>(smf + ((RC * rowf + A * HP + 1) & ~1));
    const int bl = blockIdx.x, b = (L > 0) ? bl / L : 0;
    const float s = (float)s_d;                                          // at::mul casts the double scalar to float the same way
    for (int c = threadIdx.x; c < C; c += T) idx_s[c] = idx[(long long)bl * C + c];
    for (int f = threadIdx.x; f < A * nv; f += T) {                      // q row of this token (A*HD contiguous floats), scaled: q*s
        const int a2 = f / nv, j = f - a2 * nv;
        float4 v = __ldg(reinterpret_cast<const float4*>(q + (long long)bl * AH) + f);
        v.x = __fmul_rn(v.x, s); v.y = __fmul_rn(v.y, s); v.z = __fmul_rn(v.z, s); v.w = __fmul_rn(v.w, s);
        *reinterpret_cast<float4*>(qsm + a2 * HP + 4 * j) = v;
    }
    const int a = threadIdx.x % A, cc = threadIdx.x / A;
    __syncthreads();
    for (int c0 = 0; c0 < C; c0 += RC) {
        const int nr = min(RC, C - c0);
        for (int f = threadIdx.x; f < nr * A * nv; f += T) {             // coalesced: consecutive threads take consecutive float4 of a row
            const int r = f / (A * nv), rem = f - r * (A * nv), a2 = rem / nv, j = rem - a2 * nv;
            float4 v = __ldg(reinterpret_cast<const float4*>(ku + idx_s[c0 + r] * AH) + rem);
            v.x = __fmul_rn(v.x, s); v.y = __fmul_rn(v.y, s); v.z = __fmul_rn(v.z, s); v.w = __fmul_rn(v.w, s);   // k*s, staged
            *reinterpret_cast<float4*>(smf + r * rowf + a2 * HP + 4 * j) = v;
        }
        __syncthreads();
        if (cc < nr) {
            const int c = c0 + cc;
            const long long ib = (long long)bl * A + a;
            if (ib < (long long)tail_start) {
                const float4* kp = reinterpret_cast<const float4*>(smf + cc * rowf + a * HP);
                const float4* qp = reinterpret_cast<const float4*>(qsm + a * HP);
                float acc = 0.f;
                for (int i = 0; i < nv; ++i) {
                    const float4 kv = kp[i]; const float4 qv = qp[i];               // both pre-scaled in shared memory
                    acc = fmaf(kv.x, qv.x, acc); acc = fmaf(kv.y, qv.y, acc); acc = fmaf(kv.z, qv.z, acc); acc = fmaf(kv.w, qv.w, acc);
                }
                out[ib * C + c] = (L >= 0) ? __fadd_rn(acc, __ldg(mask + ((long long)b * A + a) * C + c)) : acc;
            }
        }
        __syncthreads();
    }
}
extern "C" __global__ void fq_pv_main(const float* __restrict__ p, const float* __restrict__ vu, const long long* __restrict__ idx,
                                      float* __restrict__ out, int C, int A, int HD, int tail_start)
{   // one block per (b,l); p (nBL*A, C); vu (rows, A*HD); idx (nBL, C); out (nBL, A*HD) == stock's (B, L, 1, Ah) context
    extern __shared__ float sm[];
    float* p_s = sm;
    long long* idx_s = reinterpret_cast<long long*>(sm + ((A * C + 1) & ~1));
    const int bl = blockIdx.x, AH = A * HD, nv = HD >> 2;
    for (int c = threadIdx.x; c < C; c += blockDim.x) idx_s[c] = idx[(long long)bl * C + c];
    for (int w = threadIdx.x; w < A * C; w += blockDim.x) p_s[w] = p[(long long)bl * A * C + w];
    __syncthreads();
    for (int w = threadIdx.x; w < A * nv; w += blockDim.x) {
        const int a = w / nv, d4 = w - a * nv;
        const long long ib = (long long)bl * A + a;
        if (ib >= (long long)tail_start) continue;
        const float* vb = vu + a * HD + 4 * d4;
        const float* pa = p_s + a * C;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int c = 0; c < C; ++c) {
            const float4 vv = __ldg(reinterpret_cast<const float4*>(vb + idx_s[c] * AH)); const float pc = pa[c];
            acc.x = fmaf(vv.x, pc, acc.x); acc.y = fmaf(vv.y, pc, acc.y); acc.z = fmaf(vv.z, pc, acc.z); acc.w = fmaf(vv.w, pc, acc.w);
        }
        reinterpret_cast<float4*>(out + (long long)bl * AH + a * HD)[d4] = acc;
    }
}
// ---- remainder kernels: one thread per output, the summation tree named by (mode, P) ------------------------------------------
struct FQK { const float* kb; const float* qb; float s;
             __device__ float operator()(int j, float acc) const { return fmaf(__fmul_rn(__ldg(kb + j), s), __fmul_rn(__ldg(qb + j), s), acc); } };
struct FPV { const float* vb; const long long* ib; const float* pb; long long AH;
             __device__ float operator()(int j, float acc) const { return fmaf(__ldg(vb + ib[j] * AH), __ldg(pb + j), acc); } };
template <class Fn> __device__ float dot_tree(const Fn& f, int k, int mode, int P)
{
    float part[64];
    if (mode == 0) { float acc = 0.f; for (int j = 0; j < k; ++j) acc = f(j, acc); return acc; }
    if (mode == 1 || mode == 2) {                       // P contiguous chunks of ceil(k/P); 1: added in order, 2: butterfly down
        const int cs = (k + P - 1) / P;
        for (int q = 0; q < P; ++q) { float s = 0.f; const int j0 = q * cs, j1 = min(k, j0 + cs); for (int j = j0; j < j1; ++j) s = f(j, s); part[q] = s; }
        if (mode == 1) { float acc = part[0]; for (int q = 1; q < P; ++q) acc = acc + part[q]; return acc; }
        int n = 1; while (n < P) n <<= 1; for (int q = P; q < n; ++q) part[q] = 0.f;
        for (int off = n >> 1; off > 0; off >>= 1) for (int q = 0; q < off; ++q) part[q] = part[q] + part[q + off];
        return part[0];
    }
    if (mode == 3) {                                    // 32 lanes, lane t sums elements P*t .. P*t+P-1, butterfly down 16..1
        for (int l = 0; l < 32; ++l) { float s = 0.f; const int j0 = l * P, j1 = min(k, j0 + P); for (int j = j0; j < j1; ++j) s = f(j, s); part[l] = s; }
        for (int off = 16; off > 0; off >>= 1) for (int l = 0; l < off; ++l) part[l] = part[l] + part[l + off];
        return part[0];
    }
    if (mode == 4) {                                    // P strided partials (j = q, q+P, ...), butterfly down
        for (int q = 0; q < P; ++q) { float s = 0.f; for (int j = q; j < k; j += P) s = f(j, s); part[q] = s; }
        int n = 1; while (n < P) n <<= 1; for (int q = P; q < n; ++q) part[q] = 0.f;
        for (int off = n >> 1; off > 0; off >>= 1) for (int q = 0; q < off; ++q) part[q] = part[q] + part[q + off];
        return part[0];
    }
    return __int_as_float(0x7fc00000);                  // unknown mode: NaN (never equal -> rejected)
}
extern "C" __global__ void fq_qk_tail(const float* __restrict__ q, const float* __restrict__ ku, const long long* __restrict__ idx,
                                      const float* __restrict__ mask, float* __restrict__ out, int C, int A, int HD, int L, int tail_start, int nb,
                                      int mode, int P, double s_d)
{   // outputs (ib, c) for ib in [tail_start, nb): the remainder call's summation tree, then + mask
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long total = (long long)(nb - tail_start) * C;
    if (t >= total) return;
    const long long ib = tail_start + t / C; const int c = (int)(t % C);
    const long long bl = ib / A; const int a = (int)(ib - bl * A); const int AH = A * HD; const int b = (L > 0) ? (int)(bl / L) : 0;
    FQK f; f.kb = ku + idx[bl * C + c] * AH + a * HD; f.qb = q + ib * HD; f.s = (float)s_d;
    const float acc = dot_tree(f, HD, mode, P);
    out[ib * C + c] = (L >= 0) ? __fadd_rn(acc, __ldg(mask + ((long long)b * A + a) * C + c)) : acc;
}
extern "C" __global__ void fq_pv_tail(const float* __restrict__ p, const float* __restrict__ vu, const long long* __restrict__ idx,
                                      float* __restrict__ out, int C, int A, int HD, int tail_start, int nb, int mode, int P)
{   // outputs (ib, d) for ib in [tail_start, nb)
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long total = (long long)(nb - tail_start) * HD;
    if (t >= total) return;
    const long long ib = tail_start + t / HD; const int d = (int)(t % HD);
    const long long bl = ib / A; const int a = (int)(ib - bl * A); const int AH = A * HD;
    FPV f; f.vb = vu + a * HD + d; f.ib = idx + bl * C; f.pb = p + ib * C; f.AH = AH;
    out[bl * AH + a * HD + d] = dot_tree(f, C, mode, P);
}
"""
_KERNEL_NAMES = ("fq_qk_main", "fq_qk_simple", "fq_pv_main", "fq_qk_tail", "fq_pv_tail")
SMEM_DEFAULT_MAX = 48 * 1024  # dynamic shared memory a kernel gets without an opt-in attribute
_KERNELS: dict = {}  # device index -> {name: kernel}


def _compile(device_index: int) -> dict:
    """NVRTC-compile the source for the device's architecture and load it through torch's driver bindings (no nvcc, no CUDA_HOME)."""
    from torch.cuda import _utils as cu
    lib = cu._get_gpu_rtc_library() if hasattr(cu, "_get_gpu_rtc_library") else cu._get_nvrtc_library()
    props = torch.cuda.get_device_properties(device_index)
    opts = [f"--gpu-architecture=sm_{props.major}{props.minor}".encode(), b"--std=c++17", b"--fmad=true", b"--ftz=false", b"--prec-div=true"]
    prog = ctypes.c_void_p()
    if lib.nvrtcCreateProgram(ctypes.byref(prog), _SRC.encode(), b"gpnstar_fused_col.cu", 0, None, None) != 0:
        raise RuntimeError("nvrtcCreateProgram failed")
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    if rc != 0:
        n = ctypes.c_size_t(); lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n)); buf = ctypes.create_string_buffer(n.value); lib.nvrtcGetProgramLog(prog, buf)
        raise RuntimeError("gpnstar fused column attention: NVRTC compilation failed:\n" + buf.value.decode(errors="replace"))
    n = ctypes.c_size_t()
    if lib.nvrtcGetCUBINSize(prog, ctypes.byref(n)) != 0:
        raise RuntimeError("nvrtcGetCUBINSize failed")
    cubin = ctypes.create_string_buffer(n.value)
    if lib.nvrtcGetCUBIN(prog, cubin) != 0:
        raise RuntimeError("nvrtcGetCUBIN failed")
    with torch.cuda.device(device_index):
        return cu._cuda_load_module(cubin.raw, list(_KERNEL_NAMES))


def _kernels(dev: torch.device) -> dict:
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    k = _KERNELS.get(i)
    if k is None:
        k = _KERNELS[i] = _compile(i)
    return k


def tail_start_of(nb: int) -> int:
    """First batch matrix of the remainder call (== nb when nb is a multiple of CHUNK)."""
    return CHUNK * (nb // CHUNK)


def fused_scores(q: Tensor, k_u: Tensor, idx: Tensor, mask, s: float, C: int, A: int, HD: int, tail: tuple) -> Tensor:
    """scores (B, L, A, 1, C) = (q*s) . (k*s) [+ mask]: q (B, L, 1, A*HD) raw query projection (contiguous); k_u (rows, A*HD) raw reduced
    keys; idx (B, L, C) int64; mask (B, 1, A, 1, C) contiguous or None; s = sqrt(scale) as the Python float SDPA-math uses;
    tail = (start, mode, P): matrices >= start are the remainder call's, summed under tree (mode, P)."""
    B, L = idx.shape[0], idx.shape[1]
    nBL, nb = B * L, B * L * A
    out = torch.empty((B, L, A, 1, C), dtype=torch.float32, device=q.device)
    ks = _kernels(q.device)
    ts, mode, P = tail
    use_mask = mask is not None
    margs = [mask if use_mask else out]  # a valid pointer either way; L < 0 tells the kernel there is no mask
    Lm = L if use_mask else -1
    if ts > 0:
        T, smem = _qk_geometry(C, A, HD)
        if T:
            ks["fq_qk_main"](grid=(nBL, 1, 1), block=(T, 1, 1), args=[q, k_u, idx] + margs + [out, C, A, HD, Lm, ts, float(s)], shared_mem=smem)
        else:
            ks["fq_qk_simple"](grid=(nBL, 1, 1), block=(128, 1, 1), args=[q, k_u, idx] + margs + [out, C, A, HD, Lm, ts, float(s)], shared_mem=8 * C)
    if ts < nb:
        total = (nb - ts) * C
        ks["fq_qk_tail"](grid=((total + 127) // 128, 1, 1), block=(128, 1, 1), args=[q, k_u, idx] + margs + [out, C, A, HD, Lm, ts, nb, mode, P, float(s)])
    return out


def _qk_geometry(C: int, A: int, HD: int):
    """(threads, dynamic smem bytes) for the staged score kernel, or (0, 0) when the geometry does not fit it (-> the simple kernel)."""
    for T in (128, 64, 256, 32):
        if T % A or T // A < 1:
            continue
        RC, HP = T // A, HD + 4
        nfloat = (RC * A * HP + A * HP + 1) & ~1
        smem = 4 * nfloat + 8 * C
        if smem <= SMEM_DEFAULT_MAX:
            return T, smem
    return 0, 0


def fused_context(p: Tensor, v_u: Tensor, idx: Tensor, C: int, A: int, HD: int, tail: tuple) -> Tensor:
    """context (B, L, 1, A*HD): p (B, L, A, 1, C) contiguous probabilities; v_u (rows, A*HD); idx (B, L, C) int64."""
    B, L = idx.shape[0], idx.shape[1]
    nBL, nb = B * L, B * L * A
    out = torch.empty((B, L, 1, A * HD), dtype=torch.float32, device=p.device)
    ks = _kernels(p.device)
    ts, mode, P = tail
    if ts > 0:
        nthreads = min(256, max(32, A * (HD // 4)))
        ks["fq_pv_main"](grid=(nBL, 1, 1), block=(nthreads, 1, 1), args=[p, v_u, idx, out, C, A, HD, ts], shared_mem=4 * ((A * C + 1) & ~1) + 8 * C)
    if ts < nb:
        total = (nb - ts) * HD
        ks["fq_pv_tail"](grid=((total + 127) // 128, 1, 1), block=(128, 1, 1), args=[p, v_u, idx, out, C, A, HD, ts, nb, mode, P])
    return out


def _geometry_ok(self, X: Tensor, idx: Tensor) -> bool:
    HD = int(self.attention_head_size)
    return (X.dtype == torch.float32 and X.is_cuda and idx.dtype == torch.long and HD % 4 == 0 and HD <= 2048 and idx.shape[-1] <= 2048
            and X.is_contiguous() and idx.is_contiguous())


def _unique_kv(self, st: ExactState, X: Tensor):
    """The reduced key/value projections of the rows of X (the same GEMMs the unifiedkv/dedup self-check validated for this row count)."""
    if bool(st.fuse_kv_unified):
        if not hasattr(self, "_w_kv"):
            self._w_kv = torch.cat([self.key.weight, self.value.weight], 0).contiguous()
            self._b_kv = torch.cat([self.key.bias, self.value.bias], 0).contiguous()
        kv = F.linear(X, self._w_kv, self._b_kv)
        Ah = self.all_head_size
        return kv[:, :Ah].contiguous(), kv[:, Ah:].contiguous()
    return self.key(X), self.value(X)


def _selfcheck(self, st: ExactState, q: Tensor, q_l: Tensor, k_u: Tensor, v_u: Tensor, idx: Tensor, mask: Tensor, s: float, scale: float):
    """Pick the remainder trees by whole-tensor equality against the by-construction operands of the `colattn` route on THIS data (the
    contiguous scaled K^T and V that ATen's SDPA-math path clones, built one at a time from the distinct rows, and the same cuBLAS calls on them),
    staged so that the check holds at most ONE K-sized tensor at a time -- the peak stays below stock's; return ((tail_qk, tail_pv), out) or (None, why)."""
    from .colattn import _softmax_as_sdpa, gather_keys_transposed, gather_values
    B, L, C = idx.shape
    A, HD = int(self.num_attention_heads), int(self.attention_head_size)
    nb = B * L * A
    ts = tail_start_of(nb)
    kT = gather_keys_transposed(k_u * s, idx, A, HD)       # (B, L, A, HD, C) contiguous == ATen's contiguous copy of key.transpose(-2,-1) * s
    ref_scores = torch.matmul(q_l * s, kT)                 # == the first matmul inside SDPA-math, bit for bit (pre-mask): the same cuBLAS call on the same operand
    del kT
    tail_qk = None
    for name, mode, P in TAIL_CANDIDATES:
        if ts == nb and name != "seq":
            break
        eff = (nb if mode == 0 else ts, mode, P)  # a sequential remainder is what the main kernel computes: no remainder launch
        if torch.equal(fused_scores(q, k_u, idx, None, s, C, A, HD, eff), ref_scores):
            tail_qk = (name,) + eff
            break
    if tail_qk is None:
        return None, "scores"
    ref_probs = _softmax_as_sdpa(ref_scores.add_(mask))    # ATen: attn.add_(mask); at::_safe_softmax(attn, -1)
    del ref_scores
    V = gather_values(v_u, idx, A, HD)                     # (B, L, A, C, HD) contiguous == ATen's contiguous copy of value
    ref_ctx = torch.matmul(ref_probs, V).transpose(-2, -3).contiguous().view(B, L, 1, A * HD)   # second matmul + the stock epilogue
    del V
    tail_pv = None
    for name, mode, P in TAIL_CANDIDATES:
        if ts == nb and name != "seq":
            break
        eff = (nb if mode == 0 else ts, mode, P)
        if torch.equal(fused_context(ref_probs, v_u, idx, C, A, HD, eff), ref_ctx):
            tail_pv = (name,) + eff
            break
    del ref_probs
    if tail_pv is None:
        return None, "context"
    out = _fused_attention(q, k_u, v_u, idx, mask, s, C, A, HD, tail_qk[1:], tail_pv[1:])
    if not torch.equal(out, ref_ctx):
        return None, "attention"
    return (tail_qk, tail_pv), out


def _fused_attention(q, k_u, v_u, idx, mask, s, C, A, HD, tail_qk, tail_pv) -> Tensor:
    scores = fused_scores(q, k_u, idx, mask, s, C, A, HD, tail_qk)   # (q*s).(k*s) + mask: SDPA-math's scaling, first matmul, attn.add_(mask)
    probs = torch._safe_softmax(scores, -1)                            # SDPA-math's softmax (the stock ATen op)
    return fused_context(probs, v_u, idx, C, A, HD, tail_pv)           # second matmul, written in the (B, L, 1, Ah) layout of transpose+contiguous


def _col_forward_fused(self, hidden_states, source_embeddings, attention_mask=None, evol_time_bias=None, output_attentions=False):
    st: ExactState = self._exact_state
    X, idx = st.kv_input, st.kv_index
    if (output_attentions or self.training or st.kv_fallback_full or X is None or idx is None or not getattr(st, "fused_enabled", True)
            or hidden_states.shape[0] * hidden_states.shape[1] < int(getattr(st, "colattn_min_tokens", 2048))   # launch-bound sizes (and every CUDA-graphed shape) keep the previous forward
            or (st.validate and st.kv_validation.get((int(X.shape[0]), int(st.kv_m_stock))) is not True) or not _geometry_ok(self, X, idx)):
        return self._unified_forward(hidden_states, source_embeddings, attention_mask, evol_time_bias, output_attentions)
    if attention_mask is None or evol_time_bias is None:
        raise ValueError("attention_mask and evol_time_bias are required")
    B, L, C = idx.shape
    A, HD = int(self.num_attention_heads), int(self.attention_head_size)
    key = (int(B), int(L), int(C))
    status = st.fused_status.get(key)
    if status is False:  # rejected geometry: the materialised route, by name
        return self._unified_forward(hidden_states, source_embeddings, attention_mask, evol_time_bias, output_attentions)
    try:
        k_u, v_u = _unique_kv(self, st, X)
    except torch.cuda.OutOfMemoryError:
        return self._unified_forward(hidden_states, source_embeddings, attention_mask, evol_time_bias, output_attentions)
    scale = 1 / math.sqrt(self.attention_head_size)
    s = math.sqrt(scale)                    # SDPA-math scales q and k^T by sqrt(scale) each (double -> float inside at::mul)
    q = self.query(hidden_states)           # (B, L, 1, Ah): the memory transpose_for_scores views as (B, L, A, 1, HD)
    if not q.is_contiguous():
        q = q.contiguous()
    mask = attention_mask + evol_time_bias.to(q.dtype)              # (B, 1, A, 1, C), as stock forms it
    if tuple(mask.shape) != (B, 1, A, 1, C):
        mask = mask.expand(B, 1, A, 1, C)
    mask = mask.contiguous()
    if status is None:  # first forward of this geometry on the fused route: decide it on this layer's real data
        try:
            modes, info = _selfcheck(self, st, q, self.transpose_for_scores(q), k_u, v_u, idx, mask, s, scale)
        except torch.cuda.OutOfMemoryError:
            modes, info = None, "out of memory during the self-check"
        except Exception as e:  # noqa: BLE001 - no runtime compiler / driver refusal: the lever is off for the process, the route stays exact
            modes, info = None, f"{type(e).__name__}: {e}"
            st.fused_enabled = False
        nb = B * L * A
        rec = {"shape": key, "nb": nb, "tail_start": tail_start_of(nb), "ok": modes is not None}
        if modes is None:
            st.fused_status[key] = False
            rec["rejected_at"] = info
            st.fused_log.append(rec)
            warnings.warn(f"gpnstar_exact: the fused column attention did not reproduce the materialised attention bit for bit at batch shape "
                          f"B={B} x L={L} ({info}) -> this shape runs the materialised unified K/V route (exact).", RuntimeWarning, stacklevel=2)
            del k_u, v_u
            return self._unified_forward(hidden_states, source_embeddings, attention_mask, evol_time_bias, output_attentions)
        (tqk, tpv) = modes
        st.fused_status[key] = (tqk[1:], tpv[1:])  # (effective remainder start, mode, P) per GEMV
        rec["tail_qk"], rec["tail_pv"] = tqk[0], tpv[0]
        st.fused_log.append(rec)
        return (info,)                      # the checked fused output of this layer
    tail_qk, tail_pv = status
    return (_fused_attention(q, k_u, v_u, idx, mask, s, C, A, HD, tail_qk, tail_pv),)


def patch_fused_col_attention(model: nn.Module, enabled: bool = True) -> None:
    """Install P5 on top of patch_unified_kv (which must already be applied): the column cross-attention modules get the fused
    forward; everything it cannot serve goes to the unified forward it wraps."""
    core = core_model(model)
    st = _state(model)
    if getattr(st, "fused_status", None) is None:
        st.fused_status = {}
        st.fused_log = []
    st.fused_enabled = bool(enabled)
    for layer in core.encoder.layer:
        ca = layer.attention.col_attention.self
        if not hasattr(ca, "_exact_state"):
            raise RuntimeError("patch_fused_col_attention: patch_unified_kv must be applied first")
        if not hasattr(ca, "_unified_forward"):
            ca._unified_forward = ca.forward
            ca.forward = _col_forward_fused.__get__(ca, type(ca))


def fused_report(model: nn.Module) -> dict:
    st = _state(model)
    return {"enabled": bool(getattr(st, "fused_enabled", False)), "status": {f"{k[0]}x{k[1]}": (v if v is False else {"tail_qk": v[0], "tail_pv": v[1]})
            for k, v in (getattr(st, "fused_status", None) or {}).items()}, "log": list(getattr(st, "fused_log", None) or [])}
