# fl_levers.py — the `fast` tier's levers for BoltzGen 0.3.2 (environment-driven; every switch unset = nothing patched). Design step only:
# the levers act inside AtomDiffusion.preconditioned_network_forward(training=False) — the 500-step sampler's denoiser call — and only
# when the trunk batch is 1 (upstream's predict batch) and the step's sigma is ONE value for the whole diffusion batch, where every
# conditioning tensor the sampler repeats over the B designs of a diffusion batch (`multiplicity`) is B identical rows by construction of the
# pinned modules (stock/PINS.json `touched_modules`):
#   diffusion.py  DiffusionModule.forward l.176-186: single_conditioner(times, s_trunk.repeat_interleave(B, 0), s_inputs.repeat_interleave(B, 0));
#                 AtomDiffusion.sample l.572-574: t_hat is a Python float, and preconditioned_network_forward l.389-391 fills sigma =
#                 torch.full((batch,), t_hat) from it -> times = c_noise(sigma) is one value per step for the whole batch; the graph
#                 sampler (bg_graph_patch.py `_StepGraph.body`) fills its static (batch,) sigma buffer from one 0-d t_hat and this
#                 module marks that buffer while the body runs (`_graph_body_marked`). Upstream's signature also admits a per-row (B,) sigma:
#                 such a call is NOT trusted — `context_by.sigma_per_row`, every site runs the stock statement (counted, never served).
#   attention.py  AttentionPairBias.forward l.113-119: bias.repeat_interleave(B, 0) of the trunk's (1, H, N, N) pair bias and the (1, N) pad
#                 mask -> the (B, H, N, N) fp32 attn_mask is B identical slices (xa_hoist builds ONE (1, H, N, N) slice per design,
#                 which reaches this module's SDPA wrapper as a batch-1 mask: the `attn_mask` site counts it `single` — nothing to de-duplicate).
#
#   FL_COND_DEDUP=1 : `cond_dedup` — the multiplicity-invariant conditioning path runs on ONE row and is broadcast over the B designs instead of
#                     being recomputed B times per step: SingleConditioning (2 transitions), and in each of the 24 token DiffusionTransformer
#                     layers the AdaLN s-path (s_norm, s_scale, s_bias; twice per layer) and the two sigmoid output gates on s. Same statements,
#                     same math; only the GEMM row count changes (M = N instead of B*N), so cuBLAS may pick another kernel / summation order:
#                     tolerance class (last-ulp), never a bitwise claim. The narrow / broadcast primitive is the shared core's
#                     opt_core.capture.hoist.RowDedup — one RowDedup object per statement site below; this module
#                     is only the BoltzGen glue (which statements, the structural precondition, the census). A site serves a call only when its
#                     input is uniform BY CONSTRUCTION (the sampler context above for SingleConditioning; a stride-0 expanded view produced by an
#                     upstream RowDedup site for the layer sites); every other call runs the stock statement inline and is COUNTED by reason
#                     (`inline_by`), never silently. The atom transformers' layers (same classes, conditioning `c` a real (B*NW, W, 128) copy)
#                     take the stock statements (`inline_by.rows_real`). B == 1 makes every site `single`.
#   FL_ATTN_BF16=1  : `attn_bf16` — the token transformer's pair-biased self-attention (24 layers x 500 steps, (B, 16, N, N) logits) runs as
#                     [and the six atom-transformer layers' windowed attention (32 queries x 128 keys per window, (B*NW, 4, 32, 128)
#                     logits: the mask arrives per call in stock's layout — xa_hoist builds it once per design — and is cast to bf16 whole,
#                     no row de-duplication: its rows differ per window)]
#                     ONE fused call through the shared core's opt_core.attn.sdpa_bias (compute_dtype bf16, backend pinned `efficient`: torch
#                     SDPA's memory-efficient kernel on tensor cores, fp32 accumulation) instead of stock's fp32
#                     call, every eligible call accounted by an opt_core.attn.size_gate.SizeGate (`served` / `fallback:<the core's refusal
#                     word>`; unbounded — the admitted range is every sampler shape). Kit glue only: the eligibility rules, and the fp32
#                     (B, H, N, N) attn_mask converted once per call on ONE slice (RowDedup site `attn_mask`, trusted by the construction above)
#                     into a 16-element-aligned bf16 buffer handed to the core as the additive bias, broadcast (stride 0) over the batch; the
#                     output comes back fp32 where stock's fp32 output was consumed. The trunk's attention, the atom transformers' windowed
#                     attention (queries 32 x keys 128) and every non-sampling call pass through unchanged, counted by reason (`passthrough_by`);
#                     the core's own inner SDPA call reaches torch through the same wrapper and is passed through (`passthrough_by.reentrant`).
#                     Tolerance class (a bf16 island inside the fp32 sampler; GEMMs stay TF32 as upstream ships them).
#   FL_ATTN_BACKEND=<pin> : the SDPA backend pin attn_bf16 hands the core ('efficient' when unset; the `fast` mode exports `cudnn`: torch's cuDNN
#                     fused attention, which reads the stride-0 bf16 bias like the memory-efficient kernel and is faster at every sampler shape;
#                     same numerics class, run-to-run bitwise). A pin the running torch cannot honour or a kernel that rejects the inputs is the
#                     core's refusal by name (`backend_unavailable:<pin>` / `no_kernel:<pin>`): booked as the gate's fallback, the lever DISABLED
#                     by name, the stock fp32 call serves (partial run, exit 3) — a pin never degrades to another kernel unannounced.
#   FL_DIT_FUSED=1  : `dit_fused` — in the 24 token DiffusionTransformer layers, the fp32 elementwise statements around the GEMMs run as fused
#                     Triton row kernels of the shared core (opt_core kernels `dtk_kernels`, routed by name and sha-held before import):
#                     AdaLN's a-path (LayerNorm without affine, times the conditioning's sigmoid scale, plus its shift:
#                     `ln_modulate`), the attention block's sigmoid output gate with the layer's residual add (`gate_residual`), the transition's
#                     SwiGLU times its value GEMM (`swiglu` with a third factor, its two input GEMMs run as ONE on weights concatenated once per
#                     module) and its output gate with the layer's second residual add (`gate_residual`). The conditioning operands are the
#                     cond_dedup sites' ONE-row blocks handed to the kernels as periodic rows (never materialised over the batch), so the lever
#                     REQUIRES cond_dedup (FL_DIT_FUSED=1 without FL_COND_DEDUP=1 refuses at install by name) and serves exactly the calls
#                     cond_dedup serves (the sampler context, a stride-0 conditioning view); the atom transformers' layers (real per-row
#                     conditioning, key windows) and every other call run the cond_dedup statements, counted by reason (`dit_passthrough_by`).
#                     fp32 in, fp32 out, fp32 statistics; no atomics (run-to-run bitwise at fixed seed). Tolerance class (LayerNorm / sigmoid / SiLU
#                     re-associated; one concatenated GEMM instead of two: cuBLAS kernel choice changes the TF32 summation order).
#
# STATS / census (fail-loud): `<lever>_enabled` flips True the instant install() patches the classes (registry.py "stats" probe);
# `[fl_levers] <lever> installed ...` is the line the caller's census reads (boltzgen_opt stack.RUNNER_LINES); at interpreter exit `report_lines()` prints
# one `[fl_levers] LEVER name=<site> ... class=tolerance ...` evidence line per RowDedup site and one for attn_bf16 (opt_core's lever_line grammar, the size gate's
# census as its evidence), `[fl_levers] <lever> stats: {json}` (stack.RUNNER_STATS_LINE) and, when a lever that was switched on provably did
# not act where it had to — a RowDedup site with counted fallbacks, no served call although rows > 1 were seen, an attention lever that never
# ran although eligible calls were seen, any size-gate fallback — `[fl_levers] <lever> GATE-FAIL <problems>` (stack.RUNNER_DISABLED_LINES:
# the run is partial, exit 3). A call the core refuses by name (`opt_core.attn.sdpa_bias.Refused`: no kernel for the pin, an older torch)
# books `fallback:<word>` on the gate and disables attn_bf16 BY NAME (`[fl_levers] attn_bf16 DISABLED reason=...`, the stock call serves from
# then on) — the same partial exit. An out-of-memory error is always re-raised (opt_core.oom.is_oom), never converted.
import atexit, json, os, sys, threading
import torch
import torch.nn.functional as F
from torch import sigmoid
from opt_core import report as core_report
from opt_core.attn.sdpa_bias import Refused as SdpaRefused, sdpa_bias
from opt_core.attn.size_gate import SizeGate
from opt_core.capture.hoist import RowDedup
from opt_core.oom import is_oom

TAG = "fl_levers"
ATTN_STRATEGY = "F5.flash_attn_dense"      # opt_core strategy id (STRATEGIES.json): attention with an additive pair bias in one fused SDPA call
ATTN_BACKEND = os.environ.get("FL_ATTN_BACKEND") or "efficient"   # the backend pin handed to the core: `efficient` unless the mode exports another (fast: `cudnn`); a pin never degrades to another kernel unannounced
DIT_STRATEGY = "LOCAL.dit_fused_kernels"   # opt_core strategy id (STRATEGIES.json): the diffusion-transformer fused row kernels (core kernels `dtk_kernels`)
DIT_KERNEL = "dtk_kernels"                 # the core kernel name routed (opt_core.kernels.route) and sha-held (route_check) before import
DIT_MAX_WIDTH = 8192                       # the row kernels hold one row per program: wider activations are refused by name (`width`)
STATS = {
    "dedup_enabled": False, "dedup_sites": None, "sampling_calls": 0, "max_rows": 0, "graph_integration": False,
    "context_by": {"sampling": 0, "training_path": 0, "trunk_batch_gt1": 0, "sigma_per_row": 0},
    "sigma_by": {"float": 0, "scalar": 0, "expanded": 0, "graph_static": 0},
    "inline_by": {"not_sampling": 0, "rows_real": 0},
    "attn_enabled": False, "attn_site": None, "attn_calls": 0, "attn_windowed_calls": 0, "attn_eligible": 0, "attn_backend": ATTN_BACKEND, "attn_event": None,
    "passthrough_by": {"not_sampling": 0, "no_float_mask": 0, "other": 0, "reentrant": 0}, "attn_disabled_reason": None,
    "dit_enabled": False, "dit_active": False, "dit_site": None, "dit_layers_served": 0, "dit_eligible": 0, "dit_kernel_file": None, "dit_wcat_bytes": 0,
    "dit_passthrough_by": {"not_sampling": 0, "rows_real": 0, "to_keys": 0, "dtype": 0, "layout": 0, "width": 0, "device": 0}, "dit_disabled_reason": None,
    "gate_problems": [],
}
_INSTALLED = False
_CTX = threading.local()          # .sampling: True inside the sampler's denoiser call with a batch-1 trunk and a batch-uniform sigma (the precondition of every site);
                                   # .rows: that call's design count; .graph_sigma: bg_graph_patch's static sigma buffer while its step body runs; .in_core: inside the core's fused call
SITE_NAMES = ("cond", "adaln", "layer_gate", "transition_gate", "attn_mask")
MASK_SITES = ("attn_mask",)         # the attention lever's site (reported with that lever, not with cond_dedup's)
SITES = {}                          # name -> RowDedup (one object per deduplicated statement site; created at install so a process that never installs owns none)
ATTN_GATE = None                    # opt_core.attn.size_gate.SizeGate of attn_bf16 (created at install when the lever is on): every eligible call is served, refused-by-name (fallback:<word>) or, never here, gated
DIT_GATE = None                     # opt_core.attn.size_gate.SizeGate of dit_fused (created at install when the lever is on): one decision per served token-transformer LAYER call
K = None                            # the routed core kernel module `dtk_kernels` (import at install, behind route_check + route)
MASK_ALIGN = 16                     # elements: the memory-efficient SDPA kernel reads a bias whose row strides are multiples of its alignment without a padding copy


def _log(msg):
    print(f"[{TAG}] {msg}", file=sys.stderr, flush=True)


def _require(cls, attr, where):
    if not hasattr(cls, attr):
        raise AttributeError(f"{TAG}: {where} has no {attr!r} — version/shape mismatch, refusing a silent no-op patch")


def _sampling() -> bool:
    return getattr(_CTX, "sampling", False)


def _dedupable(s) -> bool:
    """The layer sites' structural precondition: inside the sampler context and `s` is an expanded (stride-0) view over its row dim —
    i.e. produced by an upstream RowDedup site, uniform by construction. Anything else takes the stock statement (counted)."""
    if not _sampling():
        STATS["inline_by"]["not_sampling"] += 1; return False
    if s.dim() < 2 or s.shape[0] < 2 or s.stride(0) != 0:
        STATS["inline_by"]["rows_real"] += 1; return False
    return True


def _apply(site, fn, x, dim=0):
    """One served statement, trusted by construction (the sampler context's precondition; RowDedup counts every call)."""
    return SITES[site].apply(fn, x, dim, trusted=True)


def _sigma_class(sigma):
    """How the denoiser call's sigma is ONE value for the whole batch, judged by type and provenance (never by reading device values, so it
    holds inside a CUDA-graph capture): a Python number (stock sample(), l.572-574); a one-element tensor; a stride-0 expanded view of one
    value; bg_graph_patch's static (batch,) buffer while its step body runs (filled from one 0-d t_hat by `_StepGraph.body`, marked by
    `_graph_body_marked`). None = a per-row sigma this module cannot vouch for: the call is not trusted."""
    if isinstance(sigma, (float, int)) and not isinstance(sigma, bool):
        return "float"
    if torch.is_tensor(sigma):
        if sigma.numel() == 1:
            return "scalar"
        if sigma.dim() == 1 and sigma.stride(0) == 0:
            return "expanded"
        if sigma is getattr(_CTX, "graph_sigma", None):
            return "graph_static"
    return None


# ----------------------------------------------------------------------------------------------------------------- cond_dedup bodies
def _pnf_wrapper(_stock_pnf):
    def preconditioned_network_forward(self, noised_atom_coords, sigma, network_condition_kwargs, training=True):
        nck = network_condition_kwargs
        st = nck.get("s_trunk") if isinstance(nck, dict) else None
        sig = None
        if training or self.training:
            reason = "training_path"
        elif not (torch.is_tensor(st) and st.shape[0] == 1):
            reason = "trunk_batch_gt1"
        else:
            sig = _sigma_class(sigma)
            reason = "sampling" if sig is not None else "sigma_per_row"
        STATS["context_by"][reason] += 1
        prev, prev_rows = getattr(_CTX, "sampling", False), getattr(_CTX, "rows", None)
        _CTX.sampling = reason == "sampling"
        if _CTX.sampling:
            STATS["sampling_calls"] += 1
            STATS["sigma_by"][sig] += 1
            _CTX.rows = int(nck.get("multiplicity", 1) or 1)              # the design count of this call (max_rows: the gate judges the sites only once a call with more than one row was seen)
            STATS["max_rows"] = max(STATS["max_rows"], _CTX.rows)
        try:
            return _stock_pnf(self, noised_atom_coords, sigma, network_condition_kwargs, training=training)
        finally:
            _CTX.sampling, _CTX.rows = prev, prev_rows
    preconditioned_network_forward._fl_levers = True
    return preconditioned_network_forward


def _graph_body_marked(_body):
    """Over bg_graph_patch._StepGraph.body: marks the graph's static (batch,) sigma buffer as batch-uniform for the denoiser call the body
    makes (the body fills it from ONE 0-d t_hat with `sigma.copy_`). Python-level only: nothing is added to the captured work."""
    def body(self):
        prev = getattr(_CTX, "graph_sigma", None)
        _CTX.graph_sigma = self.static.get("sigma") if isinstance(getattr(self, "static", None), dict) else None
        try:
            return _body(self)
        finally:
            _CTX.graph_sigma = prev
    body._fl_levers = True
    return body


def _sc_wrapper(_stock_sc):
    def single_conditioning_forward(self, times, s_trunk, s_inputs):
        if not _sampling():
            STATS["inline_by"]["not_sampling"] += 1
            return _stock_sc(self, times, s_trunk, s_inputs)
        fn = lambda xs: _stock_sc(self, *xs)                          # the stock forward on (times, s_trunk, s_inputs), row-independent along dim 0 (returns (s, normed_fourier))
        return _apply("cond", fn, (times, s_trunk, s_inputs), 0)
    single_conditioning_forward._fl_levers = True
    return single_conditioning_forward


def _adaln_scale_bias(m, s):
    s = m.s_norm(s)                                               # stock AdaLN.forward l.30-31, the s-path: s_norm, sigmoid(s_scale), s_bias
    return torch.cat((sigmoid(m.s_scale(s)), m.s_bias(s)), dim=-1)


def _dit_eligible(a, s, to_keys=None) -> bool:
    """dit_fused's structural precondition for one layer / AdaLN call, judged by type and layout (never by reading device values): the lever
    is on and active, the call is one cond_dedup serves (sampler context, stride-0 conditioning view), the token transformer's form (no key
    windows), fp32 CUDA activations laid out as (rows, C) with unit column stride and C within the row kernels' width. Anything else takes
    the cond_dedup statements, counted by reason."""
    if not (STATS["dit_enabled"] and STATS["dit_active"]):
        return False
    by = STATS["dit_passthrough_by"]
    if not _sampling():
        by["not_sampling"] += 1; return False
    if s.dim() < 2 or s.shape[0] < 2 or s.stride(0) != 0:
        by["rows_real"] += 1; return False
    if to_keys is not None:
        by["to_keys"] += 1; return False
    if a.dtype != torch.float32:
        by["dtype"] += 1; return False
    if not a.is_cuda:
        by["device"] += 1; return False
    if a.dim() != 3 or s.dim() != 3 or a.shape[:2] != s.shape[:2] or not a.is_contiguous():
        by["layout"] += 1; return False
    if a.shape[-1] > DIT_MAX_WIDTH:
        by["width"] += 1; return False
    return True


def _block(x):
    """The ONE-row block of a RowDedup-served conditioning view (B, N, C') with stride 0 over B: its (N, C') base, handed to the row kernels
    as a periodic operand (period N)."""
    return x[0]


def _adaln_fused(m, a, s):
    """AdaLN.forward with the a-path fused: y = sigmoid(s_scale(s_norm(s))) * LayerNorm(a) + s_bias(s_norm(s)), the s-path on ONE row through
    the `adaln` RowDedup site (exactly cond_dedup's statement), LayerNorm + modulate in one `ln_modulate` pass (fp32 statistics)."""
    B, N, C = a.shape
    sb = _block(_apply("adaln", lambda x: _adaln_scale_bias(m, x), s, 0))      # (N, 2C): sigmoid(scale) | shift
    y = K.ln_modulate(a.view(B * N, C), scale=sb[:, :C], shift=sb[:, C:], eps=m.a_norm.eps, sigmoid_scale=False, mod_period=N)
    return y.view(B, N, C)


def adaln_forward(self, a, s):
    if _dit_eligible(a, s):
        return _adaln_fused(self, a, s)
    a = self.a_norm(a)
    if _dedupable(s):
        sb = _apply("adaln", lambda x: _adaln_scale_bias(self, x), s, 0)
        d = sb.shape[-1] // 2
        return sb[..., :d] * a + sb[..., d:]
    s = self.s_norm(s)
    a = sigmoid(self.s_scale(s)) * a + self.s_bias(s)
    return a
adaln_forward._fl_levers = True


def _wcat(t):
    """The transition's swish_gate | a_to_b weights concatenated ONCE per module ((2*inner + inner, dim), both bias-free upstream) so the two
    input GEMMs run as one; cached on the module, refreshed if the weights were replaced (checkpoint reload assigns new Parameters)."""
    w1, w2 = t.swish_gate[0].weight, t.a_to_b.weight
    c = getattr(t, "_fl_wcat", None)
    if c is None or c[0] is not w1 or c[1] is not w2 or c[2].device != w1.device:
        w = torch.cat((w1.detach(), w2.detach()), 0).contiguous()
        t._fl_wcat = c = (w1, w2, w)
        STATS["dit_wcat_bytes"] += w.numel() * w.element_size()
    return c[2]


def _layer_fused(self, a, s, bias, mask, multiplicity):
    """DiffusionTransformerLayer.forward (token transformer form: no key windows) with the elementwise statements fused around unchanged
    GEMMs and the unchanged attention call: adaln -> pair_bias_attn -> [a + sigmoid-gate * b] -> transition [adaln -> ONE GEMM ->
    swiglu*value -> b_to_a GEMM] -> [a + sigmoid-gate * t] -> post_lnorm. The gates are the cond_dedup sites' post-sigmoid ONE-row blocks."""
    B, N, C = a.shape
    b = _adaln_fused(self.adaln, a, s)
    b = self.pair_bias_attn(s=b, z=bias, mask=mask, multiplicity=multiplicity, k_in=b)
    g = _block(_apply("layer_gate", self.output_projection, s, 0))               # (N, C) post-sigmoid
    a1 = K.gate_residual(b.reshape(B * N, C), gate=g, res=a.view(B * N, C), gate_period=N, sigmoid_gate=False)      # a + g * b
    t = self.transition
    x = _adaln_fused(t.adaln, a1.view(B, N, C), s).view(B * N, C)
    h = F.linear(x, _wcat(t))                                                   # (B*N, 3*inner): swish_gate's 2*inner | a_to_b's inner
    inner = t.a_to_b.weight.shape[0]
    bb = K.swiglu(h[:, :2 * inner], a_first_silu=False, c=h[:, 2 * inner:])    # utils.SwiGLU: silu(second half) * first half; times a_to_b(x)
    tg = _block(_apply("transition_gate", t.output_projection, s, 0))          # (N, C) post-sigmoid
    a2 = K.gate_residual(F.linear(bb, t.b_to_a.weight), gate=tg, res=a1, gate_period=N, sigmoid_gate=False)          # a1 + tg * b_to_a(bb)
    return self.post_lnorm(a2.view(B, N, C))


def _dit_disable(word, detail):
    """dit_fused off BY NAME for the rest of the process (the cond_dedup statements serve): the DISABLED line the caller's census reads."""
    STATS["dit_active"] = False
    STATS["dit_disabled_reason"] = f"{word}: {str(detail).splitlines()[0][:200]}" if detail else word
    _log(f"dit_fused DISABLED reason={STATS['dit_disabled_reason']}")


def transition_forward(self, a, s):
    a = self.adaln(a, s)
    b = self.swish_gate(a) * self.a_to_b(a)
    g = _apply("transition_gate", self.output_projection, s, 0) if _dedupable(s) else self.output_projection(s)
    a = g * self.b_to_a(b)
    return a
transition_forward._fl_levers = True


def layer_forward(self, a, s, bias=None, mask=None, to_keys=None, multiplicity=1):
    if _dit_eligible(a, s, to_keys):
        STATS["dit_eligible"] += 1
        d = DIT_GATE.decide(int(a.shape[0]) * int(a.shape[1]))                  # counted: served (the admitted range is every sampler shape) — or re-booked as fallback:<word> below
        if d.served:
            try:
                out = _layer_fused(self, a, s, bias, mask, multiplicity)
                STATS["dit_layers_served"] += 1
                return out
            except Exception as e:                                            # OOM propagates; anything else: booked + disabled by name, the cond_dedup statements serve (partial run)
                if is_oom(e):
                    raise
                DIT_GATE.fallback("error:" + type(e).__name__)
                _dit_disable(type(e).__name__, e)
    b = self.adaln(a, s)
    k_in = b
    if to_keys is not None:
        k_in = to_keys(b)
        mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
    b = self.pair_bias_attn(s=b, z=bias, mask=mask, multiplicity=multiplicity, k_in=k_in)
    g = _apply("layer_gate", self.output_projection, s, 0) if _dedupable(s) else self.output_projection(s)
    b = g * b
    a = a + b
    a = a + self.transition(a, s)
    a = self.post_lnorm(a)
    return a
layer_forward._fl_levers = True


# ----------------------------------------------------------------------------------------------------------------- attn_bf16 body
def _mask_bf16(m):
    """One (1, H, N, N) fp32 slice -> bf16 in a buffer whose last dimension is padded to MASK_ALIGN (returned as the (1, H, N, N) view):
    every row stride is then a multiple of the kernel's alignment, so SDPA broadcasts it without a padding copy."""
    n = m.shape[-1]
    npad = -(-n // MASK_ALIGN) * MASK_ALIGN
    buf = torch.empty(m.shape[:-1] + (npad,), dtype=torch.bfloat16, device=m.device)
    out = buf[..., :n]
    out.copy_(m)
    return out


def _attn_disable(word, detail):
    """attn_bf16 off BY NAME for the rest of the process (the stock fp32 call serves): the DISABLED line the caller's census reads."""
    STATS["attn_enabled"] = False
    STATS["attn_disabled_reason"] = f"{word}: {str(detail).splitlines()[0][:200]}" if detail else word
    _log(f"attn_bf16 DISABLED reason={STATS['attn_disabled_reason']}")


def _sdpa_wrapper(_stock_sdpa):
    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *args, **kwargs):
        if getattr(_CTX, "in_core", False):                            # the core's own fused call (opt_core.attn.sdpa_bias) reaching torch through this wrapper
            STATS["passthrough_by"]["reentrant"] += 1
        elif not _sampling():
            STATS["passthrough_by"]["not_sampling"] += 1
        elif attn_mask is None or not torch.is_tensor(attn_mask) or attn_mask.dtype != torch.float32 or attn_mask.dim() != 4 or is_causal or dropout_p:
            STATS["passthrough_by"]["no_float_mask"] += 1
        elif query.dtype != torch.float32 or query.dim() != 4 or args or kwargs:
            STATS["passthrough_by"]["other"] += 1
        else:
            STATS["attn_eligible"] += 1
            if STATS["attn_enabled"]:
                d = ATTN_GATE.decide(int(query.shape[-2]))            # counted: served (the admitted range is every sampler shape) — or re-booked as fallback:<word> below
                if d.served:
                    windowed = query.shape[-2] != key.shape[-2]        # an atom-transformer layer (32-query x 128-key windows): the mask's rows differ per window — cast whole, no row site
                    try:
                        m = attn_mask.to(torch.bfloat16) if windowed else _apply("attn_mask", _mask_bf16, attn_mask, 0)
                        _CTX.in_core = True
                        try:
                            o, ev = sdpa_bias(query, key, value, bias=m, layout="bhld", bias_layout="bhqk", compute_dtype="bf16",
                                              out_dtype=query.dtype, backend=STATS["attn_backend"])
                        finally:
                            _CTX.in_core = False
                        STATS["attn_calls"] += 1; STATS["attn_windowed_calls"] += int(windowed); STATS["attn_event"] = ev
                        return o
                    except SdpaRefused as e:                          # the core refused this call BY NAME: booked on the gate, the lever off by name, the stock call serves (partial run)
                        ATTN_GATE.fallback(e.event)
                        _attn_disable(e.event, e.reason)
                    except Exception as e:                            # anything else the fused path raised: OOM propagates; otherwise booked + disabled by name, the stock call serves
                        if is_oom(e):
                            raise
                        ATTN_GATE.fallback("error:" + type(e).__name__)
                        _attn_disable(type(e).__name__, e)
        return _stock_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, *args, **kwargs)
    scaled_dot_product_attention._fl_levers = True
    return scaled_dot_product_attention


# ----------------------------------------------------------------------------------------------------------------- install / report
def install():
    """Idempotent. Patches, by class, exactly: AtomDiffusion.preconditioned_network_forward (the sampler context; wrapped, body unchanged),
    bg_graph_patch._StepGraph.body when that module is importable (wrapped: marks its static sigma buffer), SingleConditioning.forward
    (wrapped), AdaLN.forward / ConditionedTransitionBlock.forward / DiffusionTransformerLayer.forward (stock bodies with the s-statements
    served by RowDedup sites; with FL_DIT_FUSED=1 the token-layer bodies take the fused branch) for cond_dedup / dit_fused;
    torch.nn.functional.scaled_dot_product_attention (wrapped) for attn_bf16. Each target is checked present first (a missing one raises
    by name); dit_fused's core kernel is sha-held and routed by name before import (a refusal disables the lever by name)."""
    global _INSTALLED, ATTN_GATE, DIT_GATE, K
    if _INSTALLED:
        return
    dedup = os.environ.get("FL_COND_DEDUP", "0") == "1"
    attn = os.environ.get("FL_ATTN_BF16", "0") == "1"
    dit = os.environ.get("FL_DIT_FUSED", "0") == "1"
    if dit and not dedup:
        raise RuntimeError(f"{TAG}: dit_fused REFUSED reason=requires_cond_dedup (FL_DIT_FUSED=1 needs FL_COND_DEDUP=1: the fused kernels take the cond_dedup sites' one-row conditioning blocks)")
    if not (dedup or attn):
        _INSTALLED = True; return
    import boltzgen.model.modules.diffusion as D
    _require(D.AtomDiffusion, "preconditioned_network_forward", "boltzgen.model.modules.diffusion.AtomDiffusion")
    if not getattr(D.AtomDiffusion.preconditioned_network_forward, "_fl_levers", False):
        D.AtomDiffusion.preconditioned_network_forward = _pnf_wrapper(D.AtomDiffusion.preconditioned_network_forward)
    try:
        import bg_graph_patch as G                                   # the graph sampler (opt/forward/fast_inference/src, on the path in every kit mode); absent: the stock float path is the only sampler
    except ImportError:
        G = None
    if G is not None:
        _require(G, "_StepGraph", "bg_graph_patch"); _require(G._StepGraph, "body", "bg_graph_patch._StepGraph")
        if not getattr(G._StepGraph.body, "_fl_levers", False):
            G._StepGraph.body = _graph_body_marked(G._StepGraph.body)
        STATS["graph_integration"] = True
    for name in SITE_NAMES:
        SITES.setdefault(name, RowDedup(name, klass="tolerance"))      # tolerance class by the mode table: row-count-dependent kernels, never a bitwise claim (the evidence line says class=tolerance)
    if dedup:
        import boltzgen.model.modules.encoders as E
        import boltzgen.model.modules.transformers as T
        _require(E.SingleConditioning, "forward", "boltzgen.model.modules.encoders.SingleConditioning")
        for cls in (T.AdaLN, T.ConditionedTransitionBlock, T.DiffusionTransformerLayer):
            _require(cls, "forward", f"boltzgen.model.modules.transformers.{cls.__name__}")
        for cls, attrs in ((T.AdaLN, ("a_norm", "s_norm", "s_scale", "s_bias")), (T.ConditionedTransitionBlock, ("adaln", "swish_gate", "a_to_b", "b_to_a", "output_projection")),
                           (T.DiffusionTransformerLayer, ("adaln", "pair_bias_attn", "output_projection", "transition", "post_lnorm"))):
            probe = cls.__init__.__code__.co_names + cls.forward.__code__.co_names
            missing = [x for x in attrs if x not in probe]
            if missing:
                raise AttributeError(f"{TAG}: {cls.__name__} no longer names {missing} — version mismatch, refusing to patch")
        E.SingleConditioning.forward = _sc_wrapper(E.SingleConditioning.forward)
        T.AdaLN.forward = adaln_forward
        T.ConditionedTransitionBlock.forward = transition_forward
        T.DiffusionTransformerLayer.forward = layer_forward
        STATS["dedup_enabled"] = True
        STATS["dedup_sites"] = "encoders.SingleConditioning.forward transformers.AdaLN.forward transformers.ConditionedTransitionBlock.forward transformers.DiffusionTransformerLayer.forward"
        print(f"[{TAG}] cond_dedup installed sites={','.join(n for n in SITE_NAMES if n not in MASK_SITES)} strategy=F7.row_dedup class=tolerance graph_integration={int(STATS['graph_integration'])}", flush=True)
        if dit:
            for cls, attrs in ((T.ConditionedTransitionBlock, ("swish_gate", "a_to_b", "b_to_a", "output_projection", "adaln")), (T.AdaLN, ("a_norm",))):
                missing = [x for x in attrs if x not in cls.__init__.__code__.co_names]
                if missing:
                    raise AttributeError(f"{TAG}: dit_fused: {cls.__name__} no longer names {missing} — version mismatch, refusing to patch")
            from opt_core.kernels import route, route_check
            DIT_GATE = SizeGate(name="dit_fused", source="kit:every-sampler-shape")
            STATS["dit_enabled"] = True
            STATS["dit_site"] = "transformers.DiffusionTransformerLayer.forward + transformers.AdaLN.forward (token layers of the sampler, cond_dedup-served calls)"
            route(DIT_KERNEL)                                             # the core copy by name (a meta-path finder), then the core's gate: the routed copy is the one that resolves and its sums hold
            g = route_check(DIT_KERNEL)
            if not g.ok:                                                  # the core's refusal by name (sums mismatch, kernel not resolvable): the lever off by name before any call, the cond_dedup statements serve (partial run)
                DIT_GATE.fallback("route_refused", rebook=False)
                _dit_disable("route", g.reason)
            else:
                try:
                    import dtk_kernels as _K
                    K = _K; STATS["dit_kernel_file"] = getattr(_K, "__file__", None); STATS["dit_active"] = True
                except Exception as e:                                    # no triton / no CUDA toolchain: off by name
                    if is_oom(e):
                        raise
                    DIT_GATE.fallback("import:" + type(e).__name__, rebook=False)
                    _dit_disable("import:" + type(e).__name__, e)
            print(f"[{TAG}] dit_fused installed core=opt_core.kernels:{DIT_KERNEL} strategy={DIT_STRATEGY} class=tolerance state={'on' if STATS['dit_active'] else 'disabled'} sites=adaln,layer_gate_residual,transition gate={DIT_GATE.bounds_word()}", flush=True)
    if attn:
        F = torch.nn.functional
        _require(F, "scaled_dot_product_attention", "torch.nn.functional")
        ATTN_GATE = SizeGate(name="attn_bf16", source="kit:every-sampler-shape")
        if not getattr(F.scaled_dot_product_attention, "_fl_levers", False):
            F.scaled_dot_product_attention = _sdpa_wrapper(F.scaled_dot_product_attention)
        STATS["attn_enabled"] = True
        STATS["attn_site"] = "torch.nn.functional.scaled_dot_product_attention (AttentionPairBias.forward's call, token and atom layers of the sampler)"
        print(f"[{TAG}] attn_bf16 installed core=opt_core.attn.sdpa_bias strategy={ATTN_STRATEGY} compute=bf16 backend={ATTN_BACKEND} mask_align={MASK_ALIGN} gate={ATTN_GATE.bounds_word()}", flush=True)
    atexit.register(report_lines)
    _INSTALLED = True


def gate() -> list:
    """[problem sentences] — empty = every switched-on lever acted where it had to. cond_dedup: each RowDedup site's own gate (no counted
    fallback; at least one served call) once any sampler call with more than one row was seen; attn_bf16: the size gate's problems (any
    fallback; at least one served call once an eligible call was seen), a disabled lever, the mask site's gate."""
    problems = []
    if STATS["dedup_enabled"] and STATS["max_rows"] > 1:
        for name in SITE_NAMES:
            if name in MASK_SITES:
                continue
            problems += [f"{name}: {p}" for p in SITES[name].gate(require_served=True)]
            st = SITES[name].stats()
            if st["bypassed"]:                                     # a trusted site is served inside a capture too: any bypassed call means the captured graph carries the stock statement
                problems.append(f"{name}: bypassed={st['bypassed']} calls ran the stock statement inside a CUDA-graph capture (dedupe absent from the replayed graph)")
    if STATS["dedup_enabled"] and STATS["context_by"]["sigma_per_row"] and STATS["sampling_calls"] == 0:
        problems.append(f"cond_dedup: {STATS['context_by']['sigma_per_row']} denoiser call(s) carried a per-row sigma and none a batch-uniform one: no call was trusted")
    if STATS["dit_site"] is not None:
        if STATS["dit_disabled_reason"]:
            problems.append(f"dit_fused disabled: {STATS['dit_disabled_reason']}")
        if DIT_GATE is not None:
            problems += [f"dit_fused: {p}" for p in DIT_GATE.problems(expect_served_min=1 if STATS["dit_eligible"] else None, allow_fallback=False)]
        if STATS["max_rows"] > 1 and STATS["dit_eligible"] == 0 and not STATS["dit_disabled_reason"]:
            problems.append("dit_fused: sampler calls with more than one row were seen but no token-transformer layer call was eligible (dit_passthrough_by names why)")
    if STATS["attn_site"] is not None:
        if STATS["attn_disabled_reason"]:
            problems.append(f"attn_bf16 disabled: {STATS['attn_disabled_reason']}")
        if ATTN_GATE is not None:
            problems += [f"attn_bf16: {p}" for p in ATTN_GATE.problems(expect_served_min=1 if STATS["attn_eligible"] else None, allow_fallback=False)]
        if STATS["max_rows"] > 1 and "attn_mask" in SITES:
            stm = SITES["attn_mask"].stats()
            problems += [f"attn_mask: {p}" for p in SITES["attn_mask"].gate(require_served=STATS["attn_calls"] > 0 and stm["calls"] > stm["single"])]   # a mask that arrives as ONE row (xa_hoist's shared slice) is de-duplicated already: `single`, nothing to serve
            st = SITES["attn_mask"].stats()
            if st["bypassed"]:
                problems.append(f"attn_mask: bypassed={st['bypassed']} mask conversions ran on every row inside a CUDA-graph capture")
    return problems


def report() -> dict:
    out = dict(STATS)
    out["sites"] = {n: SITES[n].stats() for n in SITES}
    out["attn_gate"] = ATTN_GATE.census() if ATTN_GATE is not None else None
    out["dit_gate"] = DIT_GATE.census() if DIT_GATE is not None else None
    return out


def dit_evidence_line() -> str:
    """`[fl_levers] LEVER name=fl_levers.dit_fused state=<on|skipped> [reason=…] impl=opt_core.kernels:dtk_kernels origin=core strategy=LOCAL.dit_fused_kernels
    sites=adaln,layer_gate_residual,transition served=… fallback=… gated=… layers=… eligible=…` (opt_core's lever_line grammar; the size gate's census as evidence)."""
    pairs = [("sites", "adaln,layer_gate_residual,transition")] + list(DIT_GATE.lever_evidence()) + [("layers", STATS["dit_layers_served"]), ("eligible", STATS["dit_eligible"])]
    if STATS["dit_disabled_reason"]:
        return core_report.lever_line(TAG, f"{TAG}.dit_fused", "skipped", *pairs, reason=STATS["dit_disabled_reason"].split(":")[0].replace(" ", "_")[:120],
                                      impl=f"opt_core.kernels:{DIT_KERNEL}", origin="core", strategy=DIT_STRATEGY)
    return core_report.lever_line(TAG, f"{TAG}.dit_fused", "on", *pairs, impl=f"opt_core.kernels:{DIT_KERNEL}", origin="core", strategy=DIT_STRATEGY)


def attn_evidence_line() -> str:
    """`[fl_levers] LEVER name=fl_levers.attn_bf16 state=<on|skipped> [reason=…] impl=opt_core.attn.sdpa_bias origin=core strategy=F5.flash_attn_dense
    compute=bf16 backend=efficient event=<last served event|none> served=… fallback=… [fallback_by=…] gated=… calls=… eligible=…` (opt_core's lever_line grammar)."""
    pairs = [("compute", "bf16"), ("backend", STATS["attn_backend"]), ("event", STATS["attn_event"] or "none")] + list(ATTN_GATE.lever_evidence()) + [("eligible", STATS["attn_eligible"])]
    if STATS["attn_disabled_reason"]:
        return core_report.lever_line(TAG, f"{TAG}.attn_bf16", "skipped", *pairs, reason=STATS["attn_disabled_reason"].split(":")[0].replace(" ", "_")[:120],
                                      impl="opt_core.attn.sdpa_bias", origin="core", strategy=ATTN_STRATEGY)
    return core_report.lever_line(TAG, f"{TAG}.attn_bf16", "on", *pairs, impl="opt_core.attn.sdpa_bias", origin="core", strategy=ATTN_STRATEGY)


def _census_word(name, gate_obj, disabled_reason):
    """The caller's census kind for a lever with a counted stock fallback: `engaged:<name>[served=n]` only when no eligible call took the
    stock path; `partial:<name>(stock=m,stock_by=<reasons>)[served=n]` when some did (the run is also GATE-FAIL, exit 3); `off:<name>(<reason>)`
    when the lever was refused by name before serving."""
    ev = dict(gate_obj.lever_evidence()) if gate_obj is not None else {}
    served, stock = int(ev.get("served", 0) or 0), int(ev.get("fallback", 0) or 0)
    if disabled_reason and not served:
        return f"off:{name}({str(disabled_reason).split(':')[0].replace(' ', '_')[:60]})"
    if stock or disabled_reason:
        by = str(ev.get("fallback_by", disabled_reason or "?")).replace(" ", "")[:120]
        return f"partial:{name}(stock={stock},stock_by={by})[served={served}]"
    return f"engaged:{name}[served={served}]"


def report_lines():
    """At interpreter exit: the per-site evidence lines, the attention lever's evidence line, the two stats lines the caller's census keeps,
    and the gate verdict lines."""
    try:
        for n in SITES:
            print(SITES[n].evidence_line(TAG), flush=True)
        if ATTN_GATE is not None:
            print(attn_evidence_line(), flush=True)
        if DIT_GATE is not None:
            print(dit_evidence_line(), flush=True)
        words = []
        if STATS["dit_site"] is not None:
            words.append(_census_word("dit_fused", DIT_GATE, STATS["dit_disabled_reason"]))
        if STATS["attn_site"] is not None:
            words.append(_census_word("attn_bf16", ATTN_GATE, STATS["attn_disabled_reason"]) + f":{STATS['attn_backend']}")
        if words:
            print(f"[{TAG}] CENSUS {' '.join(words)}", flush=True)
        rep = report()
        probs = gate(); STATS["gate_problems"] = probs
        if STATS["dedup_sites"] is not None:
            d = {k: rep[k] for k in ("sampling_calls", "max_rows", "context_by", "sigma_by", "inline_by", "graph_integration")}
            d["sites"] = {n: rep["sites"][n] for n in SITE_NAMES if n not in MASK_SITES and n in rep["sites"]}
            print(f"[{TAG}] cond_dedup stats: {json.dumps(d, sort_keys=True)}", flush=True)
            bad = [p for p in probs if not p.startswith(("attn", "dit"))]
            if bad:
                print(f"[{TAG}] cond_dedup GATE-FAIL {' | '.join(bad)}", flush=True)
        if STATS["dit_site"] is not None:
            d = {k: rep[k] for k in ("dit_layers_served", "dit_eligible", "dit_passthrough_by", "dit_disabled_reason", "dit_kernel_file", "dit_wcat_bytes", "max_rows", "dit_gate")}
            print(f"[{TAG}] dit_fused stats: {json.dumps(d, sort_keys=True)}", flush=True)
            bad = [p for p in probs if p.startswith("dit")]
            if bad:
                print(f"[{TAG}] dit_fused GATE-FAIL {' | '.join(bad)}", flush=True)
        if STATS["attn_site"] is not None:
            d = {k: rep[k] for k in ("attn_calls", "attn_eligible", "attn_event", "passthrough_by", "attn_disabled_reason", "attn_backend", "max_rows", "attn_gate")}
            d["sites"] = {"attn_mask": rep["sites"].get("attn_mask")}
            print(f"[{TAG}] attn_bf16 stats: {json.dumps(d, sort_keys=True)}", flush=True)
            bad = [p for p in probs if p.startswith("attn")]
            if bad:
                print(f"[{TAG}] attn_bf16 GATE-FAIL {' | '.join(bad)}", flush=True)
    except Exception as e:                                            # never nothing: a census that cannot format itself says so
        print(f"[{TAG}] stats: census-failed:{type(e).__name__}:{e}", flush=True)


# installs on import (boltzgen_opt's activation imports this module after bg_hook and the xa_* lever modules, so the wrappers here sit
# over xa_hoist's patches where both touch a class — xa_hoist's DiffusionModule.forward is left in place).
install()
