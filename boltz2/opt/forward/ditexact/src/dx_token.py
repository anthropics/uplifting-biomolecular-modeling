"""dx_token.py — DITEXACT: a bitwise-identical RE-SCHEDULING of the Boltz-2 diffusion TOKEN transformer
(boltz 2.2.1 `boltz.model.modules.transformersv2.DiffusionTransformer` as built by `DiffusionModule.token_transformer`: 24 x DiffusionTransformerLayer,
dim = dim_single_cond = 2*token_s = 768, 16 heads, pair_bias_attn=True, to_keys=None, post_lnorm=Identity; inference only).

What the exact row otherwise runs per denoiser step (autocast OFF, float32_matmul_precision 'highest'): per layer 14 fp32 cuBLAS SIMT GEMM calls with
M = B*multiplicity*N_tokens rows (10 x 768->768, 768->3072, 768->1536, 1536->768) + 2 batched fp32 matmuls + ~30 elementwise/LayerNorm/softmax kernels,
all on ONE stream, i.e. a serial chain in the captured step graph.  At the token counts this kit
targets one such GEMM launches far fewer CTAs than the H100's 132 SMs
can hold, so most of the machine idles.

Levers in this file (each its own word of the row switch the kit adapter reads; each individually off-able; `off` = the previous forward, untouched):

  par         parallel graph branches.  The SAME torch ops as stock, on the SAME tensors/views/dtypes (hence the same ATen kernels and the same cuBLAS
              heuristic pick per call: identical problem descriptors, identical per-(handle, stream) workspace size), issued on forked CUDA streams so
              that independent work co-runs: (i) the whole s-conditioned path of a layer — AdaLN `s_norm(s) -> sigmoid(s_scale(.)), s_bias(.)` for the
              attention AdaLN and the transition AdaLN, and the two `sigmoid(output_projection(s))` gates: 2 LayerNorms + 6 GEMMs + 4 sigmoids that read
              ONLY s (fixed for the whole step) — is issued `prefetch` layers ahead on `n_spath` side streams; (ii) k / v / g projections run beside q;
              (iii) a_to_b runs beside swish_gate.  Joins are CUDA events (graph dependency edges under capture; no extra kernel nodes).  Cross-stream
              buffers are allocated on the CONSUMING (main) stream before the fork and written by the producer with `out=` (mm/addmm/sigmoid_), so the
              caching allocator's per-stream reuse ordering is the main stream's own — no record_stream, no extra pool growth under capture.
              Arithmetic: none changed (every element is produced by the same kernel from the same operands; only the launch
              stream differs).  Checked equal by torch.equal per call class AND by output-file sha256 vs `--mode off`.
  mask        the additive token mask `attn + (1 - mask)*-inf` (attentionv2.py:103; the kit's dit_hoist caches the term) is SKIPPED when the term is
              provably -0.0 in every element: AttentionPairBias.inf = 1e6 is FINITE, so with token_pad_mask all ones the cached term is
              (1-1)*-1e6 = 0.0*-1e6 = -0.0 exactly, and x + (-0.0) == x bit for bit for every fp32 x.  The all-ones predicate is evaluated ONCE
              per cached mask tensor OUTSIDE capture (host sync at cache build / eager step 0) and read as a Python bool inside; a mask with any zero
              keeps the stock add (counted `mask_applied`).  One [B*m,16,N,N] fp32 read+write pass per layer-step less.

  sba         `attn / head_dim**0.5 + bias [+ maskterm]` (attentionv2.py:102-103: two / three elementwise kernels, each a full pass over the fp32
              [B*m,16,N,N] scores) as ONE Triton kernel writing in place into the scores, with the SAME roundings: ATen's `tensor / python_scalar` is
              `x * fp32(1/s)` (BinaryDivTrueKernel.cu: the CPU scalar's reciprocal in accscalar_t = float), then `+ bias` and
              `+ maskterm` are IEEE adds — the kernel evaluates `t = x * r; t = t + b; [t = t + m]` with FMA contraction OFF (enable_fp_fusion=False),
              r = numpy.float32(1) / numpy.float32(s) computed on the host exactly as ATen
              does.  One (two) [B*m,16,N,N] read+write passes and one (two)
              allocations per layer-step less.  Refuses by name (the torch statements serve, counted `sba_torch_by`) off-CUDA, on a dtype/shape it does
              not serve, or when Triton is not importable.

  smx         `softmax(attn / head_dim**0.5 + bias [+ maskterm], dim=-1)` as ONE in-place CUDA kernel (NVRTC via torch.cuda._compile_kernel): the
              sba arithmetic above followed by a line-by-line replica of ATen's `softmax_warp_forward<float,float,float,log2_ceil(N),false,false,32>`
              (PersistentSoftmax.cuh — the kernel `Tensor.softmax(-1)` dispatches for a contiguous fp32 last dim of <= 2048 elements): one warp per
              row, lane l holds elements l + 32*it, sequential max then xor-butterfly Max (`a < b ? b : a`), `e = expf(x - max)` stored, sequential
              `sum += e` then xor-butterfly Add, `out = e / sum` (quiet NaN when sum == 0).  Serves 128 < N <= 2048 (WARP_BATCH 1, WARP_SIZE 32);
              declines by name otherwise (the sba / torch statements serve).  RUNTIME BIT-COMPARISON: at the first eager encounter of
              each (N, masked) class the torch statements and the kernel run on the same real scores, torch.equal -> the class is served for the
              process, else refused by name (`smx_bitcmp`), never inside capture.  Two [B*m,16,N,N] passes and one allocation per layer-step less
              than sba (the scores are read once and the probabilities written in place).
  glue        the layer's main-chain elementwise glue as three NVRTC CUDA kernels with ATen's per-element op sequence and roundings:
              `affine`  y = sc*x + sb (AdaLN tail: mul then add, two roundings, no FMA) — 2 kernels -> 1, twice per layer;
              `gres`    a + og*o (gated residual: mul then add) — 2 -> 1, twice per layer;
              `swiglu`  (silu(gates) * x) * ab with silu(g) = g / (1 + expf(-g)) (ATen's silu functor) — 3 -> 1, once per layer.
              Same toolchain family as ATen's own kernels (CUDA C++, expf / IEEE div, no fast-math); each kernel is BIT-COMPARED AT RUN TIME per
              shape class at its first eager use (torch statements vs kernel, torch.equal) and declines by name for that class otherwise.  −6 graph
              nodes per layer on the critical path.
  Compute-capability gate: smx / glue replicate ATen math sequences and serve only on the capabilities listed in PROVEN_CC (checked bitwise there);
  on any other card they are refused by name at install (`cc_refused`: unproven_cc:sm_XY) and the torch statements (and sba) serve — par / mask
  run the stock kernels and sba is plain IEEE mul/add: they serve everywhere.
  The 1/sqrt(head_dim) reciprocal both kernels multiply by is PROBED, not assumed: at the first eager call `x / s` (ATen) is
  compared bitwise with `x * r` for r in {fp32(1)/fp32(s), fp32(1/s in double)}; the matching r serves (both coincide for head_dim 48: 0x3e13cd3a),
  none matching -> the kernels decline by name (`recip_probe`) and the torch statements serve.

Composition.  The kit adapter (boltz2_opt/ditexact.py) installs `forward` as an INSTANCE attribute of `DiffusionModule.token_transformer` (so it wins
over the class-level patches of boltz_dit_hoist, whichever order they were applied in) and supplies `terms(dt, bias, mask, multiplicity)` -> the 24
expanded per-layer pair biases [B*m,16,N,N] fp32 and the additive mask term the hoist (level 2) already caches at fixed addresses — this module never
recomputes or copies them.  The sampler roll-out's capture contract holds: no host sync, no
data-dependent shape, no per-step Python state inside the forward; the
streams/events are created at the first (eager) call; per-call scalars are Python constants of the module (48 ** 0.5).
"""
from __future__ import annotations

import sys
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F

WORDS = ("par", "mask", "sba", "smx", "glue")   # the lever words this module implements (the adapter maps the row switch onto them)
PROVEN_CC = {"smx": {(9, 0), (8, 0)}, "glue": {(9, 0), (8, 0)}}   # replicas of ATen math sequences (expf / division / softmax order) serve only on the compute capabilities they were proven bitwise on (sm_90; sm_80: the per-class bit-compare served N400 / N800 / N1200 and the three glue classes on A100-80GB, every output file of --mode exact byte-identical); elsewhere refused by name at install (the other words serve)
STATS = {"calls": 0, "calls_par": 0, "calls_prev": 0, "layers": 0, "mask_skipped": 0, "mask_applied": 0, "prev_by": {}, "capturing_calls": 0,
         "sba_fused": 0, "sba_torch": 0, "sba_torch_by": {}, "recip": {}, "smx_fused": 0, "smx_declined": 0, "smx_declined_by": {}, "smx_bitcmp": {},
         "glue_fused": 0, "glue_declined": 0, "glue_declined_by": {}, "glue_bitcmp": {}, "cc_refused": {}, "nvrtc_compile_s": {},
         "n_spath": None, "prefetch": None, "last": {}}
_DEBUG = [False]


def _log(*a):
    if _DEBUG[0]:
        print("[dx_token]", *a, file=sys.stderr, flush=True)


def _prev(reason):
    STATS["calls_prev"] += 1
    STATS["prev_by"][reason] = STATS["prev_by"].get(reason, 0) + 1


class _Res:
    """Per-(instance, device) CUDA resources: side streams and an event pool.  Built at the first eager call (never during capture: creating a
    stream/event allocates nothing on the device, but we keep object creation out of the captured region on principle)."""

    def __init__(self, device, n_spath: int, n_aside: int = 3):
        self.device = device
        self.sstreams = [torch.cuda.Stream(device=device) for _ in range(max(1, n_spath))]
        self.astreams = [torch.cuda.Stream(device=device) for _ in range(n_aside)]
        self._events: List[torch.cuda.Event] = []
        self._next = 0

    def event(self) -> torch.cuda.Event:
        if self._next == len(self._events):
            self._events.append(torch.cuda.Event())          # enable_timing=False: capturable, cheapest
        e = self._events[self._next]
        self._next += 1
        return e

    def reset(self):
        self._next = 0


def qualifies(dt) -> Optional[str]:
    """None when `dt` is a DiffusionTransformer this schedule serves (the token transformer's structure), else the refusal word."""
    try:
        layers = list(dt.layers)
    except Exception:
        return "no_layers"
    if not layers:
        return "no_layers"
    if not getattr(dt, "pair_bias_attn", False):
        return "no_pair_bias"
    for ly in layers:
        apb = getattr(ly, "pair_bias_attn", None)
        if apb is None or getattr(apb, "compute_pair_bias", True):
            return "apb_computes_pair_bias"
        if not isinstance(getattr(ly, "post_lnorm", None), torch.nn.Identity):
            return "post_lnorm"
        tr = ly.transition
        if tr.swish_gate[0].weight.shape[0] != 2 * tr.a_to_b.weight.shape[0]:
            return "swish_shape"
        if ly.adaln.a_norm.weight is not None or ly.adaln.s_norm.bias is not None:
            return "adaln_affine"
        if apb.proj_q.bias is None or apb.proj_k.bias is not None or apb.proj_v.bias is not None or apb.proj_g.bias is not None or apb.proj_o.bias is not None:
            return "apb_bias_layout"
        if ly.adaln.s_scale.bias is None or ly.adaln.s_bias.bias is not None or ly.output_projection[0].bias is None or tr.output_projection[0].bias is None:
            return "adaln_bias_layout"
        if tr.swish_gate[0].bias is not None or tr.a_to_b.bias is not None or tr.b_to_a.bias is not None:
            return "transition_bias_layout"
    return None


# ----------------------------------------------------------------------------------------------------------------- the s-conditioned path of ONE layer
def _spath_adaln(ad, s, sc, sb):
    """AdaLN's s branch (transformersv2.py:27-29): s_norm(s) -> sigmoid(s_scale(.)) into `sc`, s_bias(.) into `sb`.  Same ATen calls as stock:
    nn.LayerNorm.forward == F.layer_norm(...); nn.Linear on a contiguous 3-D input with bias == addmm(bias, x2d, W.t()) (aten Linear.cpp), without bias
    == mm(x2d, W.t()) (matmul's fold); torch.sigmoid in place == out of place per element."""
    n = ad.s_norm
    sn = F.layer_norm(s, n.normalized_shape, n.weight, n.bias, n.eps)
    sn2 = sn.view(-1, sn.shape[-1])
    torch.addmm(ad.s_scale.bias, sn2, ad.s_scale.weight.t(), out=sc.view(-1, sc.shape[-1]))
    sc.sigmoid_()
    torch.mm(sn2, ad.s_bias.weight.t(), out=sb.view(-1, sb.shape[-1]))


def _spath_gate(lin, s2, og):
    """`nn.Sequential(Linear(s), Sigmoid())` (transformersv2.py:54, 165): addmm + sigmoid into `og`."""
    torch.addmm(lin.bias, s2, lin.weight.t(), out=og.view(-1, og.shape[-1]))
    og.sigmoid_()


# ----------------------------------------------------------------------------------------------------------------- `sba`: fused scale+bias(+mask)
_SBA = {"kernel": None, "error": None}


def _sba_kernel():
    """Build (once) the Triton kernel; None when Triton is not importable (the reason lands in _SBA['error'])."""
    if _SBA["kernel"] is not None or _SBA["error"] is not None:
        return _SBA["kernel"]
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _sba(X, B, MT, n_elem, hnn, n_tok, r, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            ok = offs < n_elem
            x = tl.load(X + offs, mask=ok, other=0.0)
            b = tl.load(B + offs, mask=ok, other=0.0)
            t = x * r                                  # == aten div by a CPU scalar: x * fp32(1/s)   (fp fusion disabled at launch: no FMA contraction)
            t = t + b                                  # == aten add
            if HAS_MASK:
                bm = offs // hnn                       # maskterm is [B*m, 1, 1, N]: element (bm, 0, 0, j), j = column
                j = offs % n_tok
                mt = tl.load(MT + bm * n_tok + j, mask=ok, other=0.0)
                t = t + mt                             # == aten add of the broadcast mask term
            tl.store(X + offs, t, mask=ok)

        _SBA["kernel"] = _sba
    except Exception as e:                             # noqa: BLE001  Triton missing / unusable: the torch statements serve, by name
        _SBA["error"] = f"{type(e).__name__}: {e}"
    return _SBA["kernel"]


def _sba_reason(attn, bias_i, maskterm):
    """None when the fused kernel serves this call, else the refusal word."""
    if _sba_kernel() is None:
        return "no_triton"
    if not attn.is_cuda:
        return "cpu"
    if attn.dtype != torch.float32 or bias_i.dtype != torch.float32 or (maskterm is not None and maskterm.dtype != torch.float32):
        return "dtype"
    if attn.dim() != 4 or bias_i.shape != attn.shape or not attn.is_contiguous() or not bias_i.is_contiguous():
        return "layout"
    if maskterm is not None and (tuple(maskterm.shape) != (attn.shape[0], 1, 1, attn.shape[3]) or not maskterm.is_contiguous()):
        return "mask_layout"
    return None


_RECIP = {}


def recip(scale: float, sample) -> Optional[float]:
    """The fp32 r with `x * r == x / scale` bitwise under ATen (probed once per scale on a real tensor `sample`, outside capture); None = no candidate
    matched (the kernels decline by name) or not yet probed while capturing."""
    if scale in _RECIP:
        return _RECIP[scale]
    if torch.cuda.is_current_stream_capturing():
        return None
    import numpy as np
    cands = {"one_over_fp32": float(np.float32(1.0) / np.float32(scale)), "fp32_of_double_recip": float(np.float32(1.0 / float(scale)))}
    x = sample.detach()[..., : min(sample.shape[-2], 8), :].contiguous()
    ref = x / scale
    ok = {k: r for k, r in cands.items() if torch.equal(x * r, ref)}
    r = next(iter(ok.values())) if ok else None
    _RECIP[scale] = r
    STATS["recip"][repr(scale)] = {"served": ("0x%08x" % int(np.float32(r).view(np.uint32))) if r is not None else None, "matched": sorted(ok), "candidates": {k: "0x%08x" % int(np.float32(v).view(np.uint32)) for k, v in cands.items()}}
    return r


# ----------------------------------------------------------------------------------------------------------------- `smx`: fused scale+bias(+mask)+softmax
_SMX_SRC = r"""
#define NEG_INF __int_as_float(0xff800000)
template <int LOG2, bool HAS_MASK>
__device__ __forceinline__ void smx_row(float* __restrict__ S, const float* __restrict__ B, const float* __restrict__ MT, int rows, int n, int hnn, float r)
{
    constexpr int NP2 = 1 << LOG2;             // next_power_of_two (LOG2 >= 8 here: WARP_SIZE 32, WARP_BATCH 1, 4 warps per 128-thread block)
    constexpr int WI = NP2 / 32;               // WARP_ITERATIONS
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;                   // whole warp exits together (row is warp-uniform)
    int lane = threadIdx.x;
    float* src = S + row * (long)n;
    const float* bsrc = B + row * (long)n;
    const float* msrc = HAS_MASK ? (MT + (row / hnn) * (long)n) : nullptr;
    float e[WI];
    #pragma unroll
    for (int it = 0; it < WI; ++it) {
        int j = lane + it * 32;
        if (j < n) {
            float t = __fmul_rn(src[j], r);    // aten: x / s == x * fp32(1/s)   (r probed on the host)
            t = __fadd_rn(t, bsrc[j]);         // aten: + bias
            if (HAS_MASK) t = __fadd_rn(t, msrc[j]);   // aten: + (1 - mask) * -inf term  (compile-time branch: a runtime one halves the bandwidth)
            e[it] = t;
        } else {
            e[it] = NEG_INF;
        }
    }
    // ---- softmax_warp_forward<float,float,float,LOG2,false,false,32> from here on (ATen PersistentSoftmax.cuh), WARP_BATCH == 1
    float max_value = e[0];
    #pragma unroll
    for (int it = 0; it < WI; ++it) max_value = max_value > e[it] ? max_value : e[it];
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) { float b = __shfl_xor_sync(0xffffffff, max_value, offset, 32); max_value = max_value < b ? b : max_value; }
    float sum = 0.0f;
    #pragma unroll
    for (int it = 0; it < WI; ++it) { e[it] = expf(e[it] - max_value); sum += e[it]; }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) { float b = __shfl_xor_sync(0xffffffff, sum, offset, 32); sum = sum + b; }
    #pragma unroll
    for (int it = 0; it < WI; ++it) {
        int j = lane + it * 32;
        if (j < n) {
            if (sum == 0) src[j] = __int_as_float(0x7fc00000); else src[j] = e[it] / sum;
        } else break;
    }
}
// launcher ABI (torch.cuda._compile_kernel): tensors -> pointers, Python int -> 32-bit int, Python float -> double.  One entry per (LOG2, mask).
#define SMX_ENTRY(NAME, L2, HM, LB) extern "C" __global__ void LB NAME(float* S, const float* B, const float* MT, int rows, int n, int hnn, double r) \
    { smx_row<L2, HM>(S, B, MT, rows, n, hnn, (float)r); }
SMX_ENTRY(dx_smx8_0, 8, false, )   SMX_ENTRY(dx_smx8_1, 8, true, )   SMX_ENTRY(dx_smx9_0, 9, false, )   SMX_ENTRY(dx_smx9_1, 9, true, )
SMX_ENTRY(dx_smx10_0, 10, false, ) SMX_ENTRY(dx_smx10_1, 10, true, )
SMX_ENTRY(dx_smx11_0, 11, false, __launch_bounds__(128, 6)) SMX_ENTRY(dx_smx11_1, 11, true, __launch_bounds__(128, 6))
"""
_SMX = {"kernels": None, "error": None, "ok": {}}       # ok: (N, masked) -> True (served) | False (refused: bit-comparison mismatch)


def _log2_ceil(v: int) -> int:
    l2 = 0
    while (1 << l2) < v:
        l2 += 1
    return l2


def _smx_kernels():
    """Compile (once, NVRTC) the fused softmax kernels; None + _SMX['error'] when the toolchain is not available."""
    if _SMX["kernels"] is not None or _SMX["error"] is not None:
        return _SMX["kernels"]
    import time
    t0 = time.time()
    try:
        _nvrtc_home()
        ks = {}
        for l2 in (8, 9, 10, 11):
            for hm in (0, 1):
                ks[(l2, hm)] = torch.cuda._compile_kernel(_SMX_SRC, f"dx_smx{l2}_{hm}")
        _SMX["kernels"] = ks
    except Exception as e:                             # noqa: BLE001
        _SMX["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    STATS["nvrtc_compile_s"]["smx"] = round(time.time() - t0, 2)           # first-use compile cost (once per process, before capture)
    return _SMX["kernels"]


def _smx_launch(attn, bias_i, maskterm, r):
    Bm, H, N, _ = attn.shape
    l2 = _log2_ceil(N)
    rows = Bm * H * N
    warps_per_block = 4
    blocks = (rows + warps_per_block - 1) // warps_per_block
    hm = 1 if maskterm is not None else 0
    _SMX["kernels"][(l2, hm)](grid=(blocks, 1, 1), block=(32, warps_per_block, 1),
                              args=[attn, bias_i, maskterm if hm else attn, int(rows), int(N), int(H * N), float(r)])


def _smx_decline(why):
    STATS["smx_declined"] += 1; STATS["smx_declined_by"][why] = STATS["smx_declined_by"].get(why, 0) + 1


def softmax_scale_bias_mask(attn, scale: float, bias_i, maskterm, words) -> "torch.Tensor":
    """`softmax(attn / scale + bias_i [+ maskterm], -1)`: the fused smx kernel when it serves (bit-compared once per (N, masked) class), else
    scale_bias_mask (sba or torch) followed by torch's softmax."""
    if "smx" in words:
        why = None
        Bm, H, N, N2 = attn.shape if attn.dim() == 4 else (0, 0, 0, -1)
        if _smx_kernels() is None:
            why = "no_nvrtc"
        elif not attn.is_cuda or attn.dtype != torch.float32 or bias_i.dtype != torch.float32:
            why = "dtype"
        elif attn.dim() != 4 or N != N2 or bias_i.shape != attn.shape or not attn.is_contiguous() or not bias_i.is_contiguous():
            why = "layout"
        elif not (128 < N <= 2048):
            why = "n_range"
        elif maskterm is not None and (tuple(maskterm.shape) != (Bm, 1, 1, N) or maskterm.dtype != torch.float32 or not maskterm.is_contiguous()):
            why = "mask_layout"
        r = recip(scale, attn) if why is None else None
        if why is None and r is None:
            why = "recip_probe"
        if why is None:
            key = (N, maskterm is not None)
            ok = _SMX["ok"].get(key)
            if ok is None:                                     # first encounter of this class: bit-compare on these very scores (eager only)
                if torch.cuda.is_current_stream_capturing():
                    why = "unproven_in_capture"
                else:
                    ref = attn / scale + bias_i
                    if maskterm is not None:
                        ref = ref + maskterm
                    ref = ref.softmax(dim=-1)
                    trial = attn.clone()
                    _smx_launch(trial, bias_i, maskterm, r)
                    ok = bool(torch.equal(trial, ref))
                    _SMX["ok"][key] = ok
                    STATS["smx_bitcmp"][f"N{N}{'m' if key[1] else ''}"] = "served" if ok else "refused"
                    del ref, trial
            if ok is False:
                why = "smx_bitcmp"
        if why is None:
            _smx_launch(attn, bias_i, maskterm, r)
            STATS["smx_fused"] += 1
            return attn
        _smx_decline(why)
    attn = scale_bias_mask(attn, scale, bias_i, maskterm, "sba" in words)
    return attn.softmax(dim=-1)


def scale_bias_mask(attn, scale: float, bias_i, maskterm, fused: bool):
    """`attn / scale + bias_i [+ maskterm]` — stock's three statements, or (fused) one in-place Triton pass with the same roundings."""
    why = _sba_reason(attn, bias_i, maskterm) if fused else "off"
    r = None
    if why is None:
        r = recip(scale, attn)
        if r is None:
            why = "recip_probe"
    if why is not None:
        if fused:
            STATS["sba_torch"] += 1; STATS["sba_torch_by"][why] = STATS["sba_torch_by"].get(why, 0) + 1
        attn = attn / scale + bias_i
        if maskterm is not None:
            attn = attn + maskterm
        return attn
    n = attn.numel(); Bm, H, N, N2 = attn.shape
    BLOCK = 2048
    grid = ((n + BLOCK - 1) // BLOCK,)
    _SBA["kernel"][grid](attn, bias_i, maskterm if maskterm is not None else attn, n, H * N * N2, N2, r,
                         HAS_MASK=maskterm is not None, BLOCK=BLOCK, num_warps=8, enable_fp_fusion=False)
    STATS["sba_fused"] += 1
    return attn


# ----------------------------------------------------------------------------------------------------------------- `glue`: fused elementwise replicas
_GLUE_SRC = r"""
// launcher ABI (torch.cuda._compile_kernel): tensors -> pointers, Python int -> int32.  n4 = number of float4 groups (numel / 4).
extern "C" __global__ void dx_affine4(const float4* __restrict__ sc, const float4* __restrict__ x, const float4* __restrict__ sb, float4* __restrict__ y, int n4)
{   // y = sc * x + sb : aten mul (RN) then aten add (RN); explicit _rn intrinsics are never contracted into an FMA
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n4) return;
    float4 a = sc[i], b = x[i], c = sb[i], o;
    o.x = __fadd_rn(__fmul_rn(a.x, b.x), c.x); o.y = __fadd_rn(__fmul_rn(a.y, b.y), c.y); o.z = __fadd_rn(__fmul_rn(a.z, b.z), c.z); o.w = __fadd_rn(__fmul_rn(a.w, b.w), c.w);
    y[i] = o;
}
extern "C" __global__ void dx_gres4(const float4* __restrict__ a, const float4* __restrict__ g, const float4* __restrict__ o, float4* __restrict__ y, int n4)
{   // y = a + g * o : aten mul then aten add
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n4) return;
    float4 av = a[i], gv = g[i], ov = o[i], r;
    r.x = __fadd_rn(av.x, __fmul_rn(gv.x, ov.x)); r.y = __fadd_rn(av.y, __fmul_rn(gv.y, ov.y)); r.z = __fadd_rn(av.z, __fmul_rn(gv.z, ov.z)); r.w = __fadd_rn(av.w, __fmul_rn(gv.w, ov.w));
    y[i] = r;
}
__device__ __forceinline__ float dx_silu(float g) { return g / (1.0f + expf(-g)); }          // aten silu functor: x / (1 + exp(-x)), IEEE division
extern "C" __global__ void dx_swiglu4(const float* __restrict__ h, const float4* __restrict__ ab, float4* __restrict__ y, int n4, int inner)
{   // h = [M, 2*inner] (x = h[:, :inner], gates = h[:, inner:]); y = (silu(gates) * x) * ab, aten order: silu kernel, mul, mul
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n4) return;
    int q = inner / 4; int m = i / q, c4 = i - m * q;
    const float4 xv = *reinterpret_cast<const float4*>(h + (long)m * 2 * inner + 4 * c4);
    const float4 gv = *reinterpret_cast<const float4*>(h + (long)m * 2 * inner + inner + 4 * c4);
    const float4 bv = ab[i]; float4 r;
    r.x = __fmul_rn(__fmul_rn(dx_silu(gv.x), xv.x), bv.x); r.y = __fmul_rn(__fmul_rn(dx_silu(gv.y), xv.y), bv.y);
    r.z = __fmul_rn(__fmul_rn(dx_silu(gv.z), xv.z), bv.z); r.w = __fmul_rn(__fmul_rn(dx_silu(gv.w), xv.w), bv.w);
    y[i] = r;
}
"""
_GLUE = {"kernels": None, "error": None, "ok": {}}       # ok: (kernel, shape) -> True | False


def _nvrtc_home():
    """torch's NVRTC helper insists on a CUDA include root: point it at the cu13 runtime wheel when CUDA_HOME is unset."""
    import os, site
    import torch.utils.cpp_extension as cx
    if not getattr(cx, "CUDA_HOME", None):
        for sp in site.getsitepackages():
            cand = os.path.join(sp, "nvidia", "cu13")
            if os.path.isfile(os.path.join(cand, "include", "cuda_runtime.h")):
                cx.CUDA_HOME = cand
                break


def _glue_kernels():
    if _GLUE["kernels"] is not None or _GLUE["error"] is not None:
        return _GLUE["kernels"]
    import time
    t0 = time.time()
    try:
        _nvrtc_home()
        _GLUE["kernels"] = {n: torch.cuda._compile_kernel(_GLUE_SRC, n) for n in ("dx_affine4", "dx_gres4", "dx_swiglu4")}
    except Exception as e:                             # noqa: BLE001
        _GLUE["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    STATS["nvrtc_compile_s"]["glue"] = round(time.time() - t0, 2)          # first-use compile cost (once per process, before capture)
    return _GLUE["kernels"]


def _glue_decline(why):
    STATS["glue_declined"] += 1; STATS["glue_declined_by"][why] = STATS["glue_declined_by"].get(why, 0) + 1


def _glue_ok(tensors, out_numel):
    """Common admissibility: CUDA fp32 contiguous, same numel, float4-able, kernels compiled; returns the decline word or None."""
    if _glue_kernels() is None:
        return "no_nvrtc"
    for t in tensors:
        if not (t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() and t.numel() == out_numel and t.data_ptr() % 16 == 0):
            return "layout"
    if out_numel % 4 or out_numel >= 2 ** 31:
        return "layout"
    return None


def _glue_serve(name, key, ref_fn, run_fn):
    """Bit-compare once per (kernel, shape class) on the real operands (eager only), then serve or decline by name."""
    ok = _GLUE["ok"].get(key)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return "unproven_in_capture"
        ref = ref_fn(); got = run_fn()
        ok = bool(torch.equal(got, ref))
        _GLUE["ok"][key] = ok
        STATS["glue_bitcmp"][f"{name}:{'x'.join(map(str, key[1]))}"] = "served" if ok else "refused"
        del ref, got
    return None if ok else "glue_bitcmp"


def glue_affine(sc, x, sb, words):
    """AdaLN tail `sigmoid(s_scale(s)) * a_norm(a) + s_bias(s)` with the sigmoid already applied to sc: y = sc*x + sb."""
    if "glue" in words:
        why = _glue_ok((sc, x, sb), x.numel())
        if why is None:
            n4 = x.numel() // 4
            run = lambda: (lambda y: (_GLUE["kernels"]["dx_affine4"](grid=((n4 + 255) // 256, 1, 1), block=(256, 1, 1), args=[sc, x, sb, y, n4]), y)[1])(torch.empty_like(x))   # noqa: E731
            why = _glue_serve("affine", ("affine", tuple(x.shape)), lambda: sc * x + sb, run)
            if why is None:
                STATS["glue_fused"] += 1
                return run()
        _glue_decline(why)
    return sc * x + sb


def glue_gres(a, g, o, words):
    """Gated residual `a + g*o` (stock: o = gate * o; a = a + o)."""
    if "glue" in words:
        why = _glue_ok((a, g, o), a.numel())
        if why is None:
            n4 = a.numel() // 4
            run = lambda: (lambda y: (_GLUE["kernels"]["dx_gres4"](grid=((n4 + 255) // 256, 1, 1), block=(256, 1, 1), args=[a, g, o, y, n4]), y)[1])(torch.empty_like(a))   # noqa: E731
            why = _glue_serve("gres", ("gres", tuple(a.shape)), lambda: a + g * o, run)
            if why is None:
                STATS["glue_fused"] += 1
                return run()
        _glue_decline(why)
    return a + g * o


def glue_swiglu(h, ab, words):
    """`SwiGLU(h) * ab` = (silu(gates) * x) * ab with x, gates = h.chunk(2, -1) (stock: swish_gate(t) * a_to_b(t))."""
    if "glue" in words:
        inner = h.shape[-1] // 2
        why = None
        if _glue_kernels() is None:
            why = "no_nvrtc"
        elif not (h.is_cuda and h.dtype == torch.float32 and ab.dtype == torch.float32 and h.is_contiguous() and ab.is_contiguous()
                  and h.shape[-1] == 2 * inner and inner % 4 == 0 and ab.shape[-1] == inner and ab.numel() * 2 == h.numel() and ab.numel() < 2 ** 31
                  and h.data_ptr() % 16 == 0 and ab.data_ptr() % 16 == 0):
            why = "layout"
        if why is None:
            n4 = ab.numel() // 4
            run = lambda: (lambda y: (_GLUE["kernels"]["dx_swiglu4"](grid=((n4 + 255) // 256, 1, 1), block=(256, 1, 1), args=[h, ab, y, n4, int(inner)]), y)[1])(torch.empty_like(ab))   # noqa: E731

            def ref():
                x_, gates = h.chunk(2, dim=-1)
                return (F.silu(gates) * x_) * ab
            why = _glue_serve("swiglu", ("swiglu", tuple(ab.shape)), ref, run)
            if why is None:
                STATS["glue_fused"] += 1
                return run()
        _glue_decline(why)
    x_, gates = h.chunk(2, dim=-1)
    return (F.silu(gates) * x_) * ab


class Schedule:
    """The parallel-branch schedule of one DiffusionTransformer instance.  `terms(dt, bias, mask, multiplicity)` must return
    `(bias_layers, maskterm, mask_all_ones)`: bias_layers[i] the expanded fp32 [B*m, H, N, N] pair bias of layer i (what stock computes as
    `proj_z(bias_l).repeat_interleave(m, 0)`, cached by the hoist), maskterm the fp32 additive mask term `(1 - mask[:, None, None]) * -inf` (cached by
    the hoist) and mask_all_ones a Python bool (None = unknown -> the add is kept); or None when the terms are not available for this call (the previous
    forward then serves the call, counted by name)."""

    def __init__(self, dt, terms: Callable, words: Sequence[str] = ("par", "mask"), n_spath: int = 1, prefetch: int = 1, prev_forward: Optional[Callable] = None):
        self.dt = dt
        self.terms = terms
        self.words = tuple(w for w in words if w in WORDS)
        self.par = "par" in self.words
        self.mask = "mask" in self.words
        if torch.cuda.is_available():                      # compute-capability gate for the arithmetic replicas (PROVEN_CC)
            cc = tuple(torch.cuda.get_device_capability())
            kept = []
            for w in self.words:
                if w in PROVEN_CC and cc not in PROVEN_CC[w]:
                    STATS["cc_refused"][w] = "unproven_cc:sm_%d%d" % cc
                else:
                    kept.append(w)
            self.words = tuple(kept)
        self.sba = "sba" in self.words
        self.smx = "smx" in self.words
        self.glue = "glue" in self.words
        if self.glue:
            _glue_kernels()
        if self.sba:
            _sba_kernel()                                  # import/JIT-declare Triton now (compilation happens at the first eager call, outside capture)
        if self.smx:
            _smx_kernels()                                 # NVRTC-compile the fused softmax kernels now (outside capture)
        self.n_spath = int(n_spath)
        self.prefetch = int(prefetch)
        self.prev_forward = prev_forward          # the forward this instance would run without us (bound: prev_forward(a, s, bias, mask, to_keys, multiplicity))
        self._res = {}
        STATS["n_spath"], STATS["prefetch"] = self.n_spath, self.prefetch

    # -- resources
    def _resources(self, device) -> _Res:
        r = self._res.get(device)
        if r is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("dx_token: first call of this DiffusionTransformer happens during CUDA-graph capture — the schedule's streams must be built by an eager call first")
            r = _Res(device, self.n_spath)
            self._res[device] = r
        return r

    # -- the forward (signature of DiffusionTransformer.forward)
    def __call__(self, a, s, bias=None, mask=None, to_keys=None, multiplicity=1):
        STATS["calls"] += 1
        dt = self.dt
        if to_keys is not None or dt.training or (not a.is_cuda) or torch.is_grad_enabled() and (a.requires_grad or s.requires_grad):
            _prev("to_keys" if to_keys is not None else "training" if dt.training else "cpu" if not a.is_cuda else "grad")
            return self.prev_forward(a, s, bias, mask, to_keys, multiplicity)
        got = self.terms(dt, bias, mask, multiplicity)
        if got is None:
            _prev("no_terms")
            return self.prev_forward(a, s, bias, mask, to_keys, multiplicity)
        bias_layers, maskterm, all_ones = got
        if self.mask and all_ones is True:
            maskterm_eff = None
            STATS["mask_skipped"] += 1
        else:
            STATS["mask_applied"] += 1
            if maskterm is None:                                   # terms gave no cached term: the stock expression on the stock mask
                maskterm = (1 - mask[:, None, None].float()) * -dt.layers[0].pair_bias_attn.inf
            maskterm_eff = maskterm
        if torch.cuda.is_current_stream_capturing():
            STATS["capturing_calls"] += 1
        if not self.par:
            return self._forward_serial(a, s, bias_layers, maskterm_eff)
        STATS["calls_par"] += 1
        return self._forward_par(a, s, bias_layers, maskterm_eff)

    # -- stock op sequence, one stream (used when only `mask` is on): literally DiffusionTransformerLayer/AdaLN/AttentionPairBias/CTB forwards inlined
    def _forward_serial(self, a, s, bias_layers, maskterm):
        for i, layer in enumerate(self.dt.layers):
            a = _layer_serial(layer, a, s, bias_layers[i], maskterm, self.words)
            STATS["layers"] += 1
        return a

    # -- par: the same ops on forked streams
    def _forward_par(self, a, s, bias_layers, maskterm):
        dt = self.dt
        layers = dt.layers
        L = len(layers)
        res = self._resources(a.device)
        res.reset()
        M = torch.cuda.current_stream(a.device)
        Bm, N, D = a.shape
        s2 = s.reshape(-1, s.shape[-1])                       # a view for the stock-contiguous s (reshape copies only if s is strided; values unchanged)
        s3 = s2.view(s.shape[0], s.shape[1], s.shape[2]) if s2.data_ptr() != s.data_ptr() else s
        apb0 = layers[0].pair_bias_attn
        H, Dh, c_s = apb0.num_heads, apb0.head_dim, apb0.c_s
        dim_inner = layers[0].transition.a_to_b.weight.shape[0]
        S = {}

        def issue(i):
            layer = layers[i]
            evf = res.event(); evf.record(M)
            new = lambda: torch.empty((Bm, N, D), dtype=s.dtype, device=s.device)   # noqa: E731  allocated on M = the consumer's stream
            sc, sb, og, tsc, tsb, tg = new(), new(), new(), new(), new(), new()
            jobs = ((lambda: _spath_adaln(layer.adaln, s3, sc, sb)),
                    (lambda: (_spath_gate(layer.output_projection[0], s2, og), _spath_gate(layer.transition.output_projection[0], s2, tg))),
                    (lambda: _spath_adaln(layer.transition.adaln, s3, tsc, tsb)))
            evs = []
            for j, job in enumerate(jobs):
                st = res.sstreams[(3 * i + j) % len(res.sstreams)]
                st.wait_event(evf)
                with torch.cuda.stream(st):
                    job()
                e = res.event(); e.record(st); evs.append(e)
            S[i] = (sc, sb, og, tsc, tsb, tg, evs)

        for i in range(min(L, self.prefetch + 1)):
            issue(i)
        sk, sv, sg = res.astreams
        for i, layer in enumerate(layers):
            nxt = i + self.prefetch + 1
            if nxt < L and nxt not in S:
                issue(nxt)
            sc, sb, og, tsc, tsb, tg, evs = S.pop(i)
            ad, apb, tr = layer.adaln, layer.pair_bias_attn, layer.transition
            # ---- AdaLN(a, s)                                             transformersv2.py:27-30
            b = F.layer_norm(a, ad.a_norm.normalized_shape, ad.a_norm.weight, ad.a_norm.bias, ad.a_norm.eps)
            M.wait_event(evs[0])
            b = glue_affine(sc, b, sb, self.words)
            # ---- AttentionPairBias(s=b, z=<cached>, mask, k_in=b)          attentionv2.py:87-111
            b2 = b.view(-1, D)
            kbuf = torch.empty((Bm, N, c_s), dtype=b.dtype, device=b.device)
            vbuf = torch.empty((Bm, N, c_s), dtype=b.dtype, device=b.device)
            gbuf = torch.empty((Bm, N, c_s), dtype=b.dtype, device=b.device)
            evb = res.event(); evb.record(M)
            sk.wait_event(evb)
            with torch.cuda.stream(sk):
                torch.mm(b2, apb.proj_k.weight.t(), out=kbuf.view(-1, c_s))
            ek = res.event(); ek.record(sk)
            sv.wait_event(evb)
            with torch.cuda.stream(sv):
                torch.mm(b2, apb.proj_v.weight.t(), out=vbuf.view(-1, c_s))
            ev_ = res.event(); ev_.record(sv)
            sg.wait_event(evb)
            with torch.cuda.stream(sg):
                torch.mm(b2, apb.proj_g.weight.t(), out=gbuf.view(-1, c_s))
                gbuf.sigmoid_()
            eg = res.event(); eg.record(sg)
            q = F.linear(b, apb.proj_q.weight, apb.proj_q.bias).view(Bm, -1, H, Dh)
            k = kbuf.view(Bm, -1, H, Dh)
            v = vbuf.view(Bm, -1, H, Dh)
            M.wait_event(ek)
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = softmax_scale_bias_mask(attn, Dh ** 0.5, bias_layers[i].float(), maskterm, self.words)
            M.wait_event(ev_)
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
            del attn
            o = o.reshape(Bm, -1, c_s)
            M.wait_event(eg)
            o = F.linear(gbuf * o, apb.proj_o.weight)
            # ---- gate, residual                                          transformersv2.py:202-204
            M.wait_event(evs[1])
            a = glue_gres(a, og, o, self.words)
            del o, b, b2, q, k, v, kbuf, vbuf, gbuf
            # ---- ConditionedTransitionBlock(a, s)                        transformersv2.py:61-65
            tad = tr.adaln
            t = F.layer_norm(a, tad.a_norm.normalized_shape, tad.a_norm.weight, tad.a_norm.bias, tad.a_norm.eps)
            M.wait_event(evs[2])
            t = glue_affine(tsc, t, tsb, self.words)
            t2 = t.view(-1, D)
            abuf = torch.empty((Bm, N, dim_inner), dtype=t.dtype, device=t.device)
            evt = res.event(); evt.record(M)
            sk.wait_event(evt)
            with torch.cuda.stream(sk):
                torch.mm(t2, tr.a_to_b.weight.t(), out=abuf.view(-1, dim_inner))
            ea = res.event(); ea.record(sk)
            h = F.linear(t, tr.swish_gate[0].weight)
            M.wait_event(ea)
            bb = glue_swiglu(h, abuf, self.words)                          # SwiGLU.forward (utils.py:34-35) * a_to_b(a)
            out = F.linear(bb, tr.b_to_a.weight)
            a = glue_gres(a, tg, out, self.words)                          # a + output_projection(s) * b_to_a(b)
            del t, t2, abuf, h, bb, out, sc, sb, og, tsc, tsb, tg
            STATS["layers"] += 1
        # every side stream's last event has been waited on by M above (each produced tensor is consumed on M after a wait), so all forks are joined.
        return a


def _layer_serial(layer, a, s, bias_i, maskterm, words=()):
    """One stream, stock op sequence with the cached bias / mask term (== boltz_dit_hoist's level-2 token path); `maskterm None` = the mask lever's skip;
    words: `sba` = the fused scale+bias(+mask) pass, `smx` = the fused softmax kernel."""
    ad, apb, tr = layer.adaln, layer.pair_bias_attn, layer.transition
    b = ad.a_norm(a)
    sn = ad.s_norm(s)
    b = glue_affine(torch.sigmoid(ad.s_scale(sn)), b, ad.s_bias(sn), words)
    B = b.shape[0]
    q = apb.proj_q(b).view(B, -1, apb.num_heads, apb.head_dim)
    k = apb.proj_k(b).view(B, -1, apb.num_heads, apb.head_dim)
    v = apb.proj_v(b).view(B, -1, apb.num_heads, apb.head_dim)
    g = apb.proj_g(b).sigmoid()
    with torch.autocast("cuda", enabled=False):
        attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        attn = softmax_scale_bias_mask(attn, apb.head_dim ** 0.5, bias_i.float(), maskterm, words)
        o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    o = o.reshape(B, -1, apb.c_s)
    o = apb.proj_o(g * o)
    a = glue_gres(a, layer.output_projection(s), o, words)
    tad = tr.adaln
    t = tad.a_norm(a)
    sn2 = tad.s_norm(s)
    t = glue_affine(torch.sigmoid(tad.s_scale(sn2)), t, tad.s_bias(sn2), words)
    bb = glue_swiglu(tr.swish_gate[0](t), tr.a_to_b(t), words) if "glue" in words else tr.swish_gate(t) * tr.a_to_b(t)
    return glue_gres(a, tr.output_projection(s), tr.b_to_a(bb), words)


def install(dt, terms: Callable, words=("par", "mask"), n_spath: int = 1, prefetch: int = 1) -> Schedule:
    """Bind the schedule as the INSTANCE forward of `dt` (wins over class-level patches); returns the Schedule.  `uninstall(dt)` restores."""
    why = qualifies(dt)
    if why is not None:
        raise ValueError(f"dx_token: this DiffusionTransformer is not the token transformer's structure ({why})")
    cls_forward = type(dt).forward                                   # whatever the class runs now (stock or boltz_dit_hoist's) — resolved at CALL time below
    prev = lambda *a, **k: type(dt).forward(dt, *a, **k)             # noqa: E731  late-bound: a later class patch (hoist applied after us) is honoured
    sch = Schedule(dt, terms, words=words, n_spath=n_spath, prefetch=prefetch, prev_forward=prev)
    dt.__dict__["forward"] = sch
    dt.__dict__["_dx_schedule"] = sch
    _log("installed on", type(dt).__name__, "words", sch.words, "class forward at install:", getattr(cls_forward, "__name__", cls_forward))
    return sch


def uninstall(dt) -> bool:
    if "_dx_schedule" in dt.__dict__:
        dt.__dict__.pop("forward", None)
        dt.__dict__.pop("_dx_schedule", None)
        return True
    return False
