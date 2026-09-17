"""boltz2_precision — the fast tier's precision units for Boltz-2 inference (boltz 2.2.1).

Tier 2, every unit: the model's arithmetic is unchanged (same statements, same weights, same step count / recycles / samples), the
PRECISION some statements run at is lowered from what the stock process runs — declared here by name, one unit per site, each its own
registry lever, all read from ONE row word (``BOLTZ_PRECISION=<unit>[,<unit>...]``; the kit adapter boltz2_opt/precision.py owns the word,
this module owns the mechanics). A unit that is not named installs nothing: the stock statement runs untouched (``off`` = stock).

What stock runs (boltz 2.2.1 `boltz predict`, pytorch_lightning bf16-mixed autocast around predict_step, ``torch.set_float32_matmul_precision
('highest')`` in main.py:1096, cuEquivariance kernels on):
  * the trunk under bf16 autocast: nn.Linear / matmul / einsum operands bf16 (fp32 accumulate), LayerNorm / softmax in fp32; the pair
    representation z is fp32 from the first Pairformer residual on (``z + dropout_mask_fp32 * update_bf16`` promotes), the sequence stack
    (pairformer.py:105-111) is an explicit fp32 island (autocast(enabled=False) + .float()): its GEMMs are IEEE fp32;
  * the diffusion sampler is an explicit fp32 island (boltz2.py:532 autocast(enabled=False), diffusionv2.py .float() casts): every GEMM of
    the 24-layer token transformer, the atom encoder/decoder transformers and the conditioner transitions is an IEEE fp32 cuBLAS GEMM
    (no TF32: 'highest'), attention logits / softmax / P.V in fp32, coordinates fp32;
  * the confidence module under bf16 autocast like the trunk; the affinity module fp32.

Units (name -> site -> what changes; each unit's numerics were measured per denoiser step and per structure against stock's seed-to-seed band):

  dit_bf16       diffusionv2.DiffusionModule (the score model: SingleConditioning transitions, AtomAttentionEncoder/Decoder incl. their
                 3+3-layer windowed atom transformers, s_to_a, the 24-layer token DiffusionTransformer, a_norm) runs under
                 ``torch.autocast('cuda', torch.bfloat16, cache_enabled=False)`` instead of stock's fp32 island: nn.Linear / matmul take
                 bf16 operands with fp32 accumulation (cuBLAS bf16 tensor-core GEMMs); LayerNorm statistics and outputs stay fp32
                 (autocast policy), every residual stream stays fp32 (a fp32 + bf16 add promotes), sigmoid gates / SwiGLU products run on
                 bf16 GEMM outputs, the attention core (q.k logits, +bias, +mask, softmax, P.V) stays fp32 — stock's inner island
                 (attentionv2.py:99) is left alone (see dit_attn_bf16). Pinned to fp32 BY this unit (autocast disabled + fp32 operands):
                 the noisy-coordinate embedding ``AtomAttentionEncoder.r_to_q_trans`` (encodersv2.py:466: the network's only read of the
                 coordinates), the position-update head ``AtomAttentionDecoder.atom_feat_to_atom_pos_update`` (encodersv2.py:564: its only
                 write) and the noise-level ``FourierEmbedding`` (encodersv2.py:17-33: cos(2*pi*Linear(1,256)(c_noise)) — bf16 there rounds the
                 phase) — coordinates and the noise level enter and leave the network at fp32 resolution; the sampler's own arithmetic (noise
                 scaling, c_skip / c_out combination, SVD alignment, Euler update) is outside the score model and untouched (fp32).
                 cache_enabled=False: every weight is used once per call, so autocast's cast cache buys nothing and can never alias across
                 a CUDA-graph capture boundary (the kit's graph sampler captures one score-model call).
  dit_tf32       inside the score-model call only: ``torch.backends.cuda.matmul.allow_tf32 = True`` on entry, the previous value restored on
                 exit and ASSERTED back to stock's (False, matmul precision 'highest') after every call (counted ``flag_restored``; a violation raises) — the GEMMs that are still fp32 there (all of them without dit_bf16; with it: the attention einsums, the atom<->token
                 aggregation bmm, the pinned coordinate projections excepted: they run with the flag restored) take TF32 tensor-core products
                 (10-bit mantissa operands, fp32 accumulate). The process policy outside the call ('highest': trunk, confidence, SVD) is
                 untouched; cuEquivariance kernels never run inside the score model.
  dit_attn_bf16  the token transformer's 24 pair-biased attention cores (AttentionPairBias with a precomputed bias, attentionv2.py:86-110) as
                 one fused bf16 attention: ``F.scaled_dot_product_attention(q, k, v, attn_mask=bias + mask_term)`` with q/k/v bf16, the additive
                 pair bias + key-padding term materialised once per call in bf16 ([B*m, 16, N, N]: one pass), softmax statistics and the
                 P.V accumulation fp32 inside the kernel, no [16, N, N] fp32 logits tensor. Projections (q/k/v/g/o) follow the ambient
                 policy (bf16 under dit_bf16, else fp32). The windowed atom-transformer attentions (4 heads, 32x128 windows) keep stock's core.
  seq_bf16       the Pairformer sequence attention (pairformer.py:105-109, trunk + confidence pairformers; AttentionPairBias with its own
                 LayerNorm + 128->16 pair-bias projection of z): q/k/v/g/o and the bias projection as bf16 GEMMs (LayerNorm statistics fp32),
                 the core as the same fused bf16 SDPA (bias + key-padding term in bf16, fp32 softmax statistics / P.V accumulation) instead
                 of the fp32 island's IEEE GEMMs and materialised [16, N, N] fp32 logits; the s residual stays fp32 (the update is returned fp32
                 to stock's ``s.float() + ...``). The sequence Transition (pairformer.py:110) stays fp32: the kit's fused-transition adapter owns
                 Transition.forward and counts that call ``autocast_off``.
  (no TriMul unit: cuEquivariance's triangle_multiplicative_update / triangle_attention cast their operands to the autocast dtype themselves
  (cuequivariance_ops_torch gated_gemm_torch.py:755, triangle_attention.py:675), so under boltz's bf16 autocast the stock TriMul and triangle
  attention already run bf16 tensor-core kernels; their fp32-input TF32/TF32x3 path (triangle.py:206) is never taken in inference — measured:
  no tf32 kernel in the stock process's kernel census.)

Composition: every unit installs at CLASS level on ``__call__`` (DiffusionModule, AttentionPairBias, FourierEmbedding,
SingleConditioning, DiffusionTransformerLayer), never on ``forward`` — the kit's DiT hoist / graph sampler / trunk levers replace ``forward`` and
``AtomDiffusion.sample`` at class level after this module has run, and ``__call__`` dispatches to whatever ``forward`` is installed at call
time, so the units act on top of them in either install order. The instance-level coordinate pins are installed at the first score-model call
and decide per call (unit on -> fp32 statement; unit off -> the module's own forward). Under the kit's CUDA-graph sampler the score model is
entered from Python once at step 0 and once at capture per item; the replays carry the captured bf16 / TF32 kernels — the census counts entries.
``DEV`` holds the per-op policy inside dit_bf16 (what stays fp32); its shipped values ARE the lever, ``set_dev`` is a development hook, not a lever.

API: ``enable(units) -> report dict``; ``disable()`` restores every class attribute (stock statements); ``report()`` = the census
(per unit: state, calls served, dtype facts) the adapter prints on its LEVER lines. ``UNITS`` = the known names; an unknown name raises
``ValueError`` (the adapter refuses by name). No environment is read here.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn

UNITS = ("dit_bf16", "dit_tf32", "dit_attn_bf16", "seq_bf16")
SITES = {   # unit -> (stock file:line at the boltz 2.2.1 pin, one-line what)
    "dit_bf16": ("boltz/model/models/boltz2.py:532 + boltz/model/modules/diffusionv2.py:129-205", "score model under bf16 autocast (coords io pinned fp32)"),
    "dit_tf32": ("boltz/model/modules/diffusionv2.py:129-205 (every nn.Linear / einsum inside)", "TF32 products for the score model's fp32 GEMMs"),
    "dit_attn_bf16": ("boltz/model/layers/attentionv2.py:99-107 (token transformer instances)", "fused bf16 SDPA core, fp32 softmax stats"),
    "seq_bf16": ("boltz/model/layers/pairformer.py:105-109 (AttentionPairBias, compute_pair_bias=True)", "sequence attention: bf16 GEMMs + fused bf16 SDPA core"),
}

_S: Dict[str, Any] = {"on": set(), "orig": {}, "census": {}, "errors": []}

# The per-op policy INSIDE dit_bf16 — the shipped values are the lever; the other values exist for the per-op numerics matrix only (an external
# measuring script flips them through set_dev(); no kit row, switch or environment reaches them).
DEV_DEFAULTS = {
    "pin_rq": True,          # AtomAttentionEncoder.r_to_q_trans (the coordinate READ, K=3 GEMM) in fp32 — False: bf16 operands like every other Linear
    "pin_pos": True,         # AtomAttentionDecoder.atom_feat_to_atom_pos_update (LayerNorm + 128->3, the coordinate WRITE) in fp32 — False: bf16 GEMM, bf16 r_update
    "pin_fourier": True,     # FourierEmbedding (encodersv2.py:17-33: Linear(1,256) on the noise level c_noise=ln(sigma/sigma_data)/4, weights ~N(0,1), then cos(2*pi*.))
                             # in fp32 — False: autocast rounds c_noise, the weights AND the projection to bf16 before the cos (phase error ~2^-7 * 2pi)
    "cond_fp32": False,      # the whole SingleConditioning (Fourier -> LayerNorm -> Linear(256,768), LayerNorm+Linear on [s_trunk|s_inputs], 2 SwiGLU transitions on
                             # [N,768]) kept fp32 — the shipped policy runs it bf16 except the Fourier embedding
    "resid_bf16": False,     # the token transformer's residual stream `a` rounded to bf16 after every layer (what a bf16-module re-implementation carries) — shipped: fp32 stream
    "atom_resid_bf16": False,  # the same for the atom encoder/decoder transformer layers
    "atom_attn_bf16": False,   # the windowed atom-transformer attention cores (4 heads, 32x128) as bf16 SDPA too — shipped: stock's fp32 core there
}
DEV: Dict[str, bool] = dict(DEV_DEFAULTS)


def set_dev(**flags) -> Dict[str, bool]:
    """Evidence-matrix switch (probe use only): override per-op policy flags; unknown names raise. Returns the policy now in force."""
    for k, v in flags.items():
        if k not in DEV_DEFAULTS:
            raise ValueError(f"unknown dev flag {k!r}; known: {sorted(DEV_DEFAULTS)}")
        DEV[k] = bool(v)
    return dict(DEV)


def _c(unit: str, key: str, n: int = 1) -> None:
    d = _S["census"].setdefault(unit, {})
    d[key] = d.get(key, 0) + n


def _fact(unit: str, key: str, val) -> None:
    _S["census"].setdefault(unit, {}).setdefault("facts", {})[key] = val


def active(unit: str) -> bool:
    return unit in _S["on"]


# =============================================================================================== score model: dit_bf16 / dit_tf32
def _prepare(dm) -> None:
    """Once per DiffusionModule instance: role tags on the attention modules (token / atom) and the instance-level coordinate pins. The pins
    decide at CALL time (unit on + flag) and otherwise run the module's own forward, so an inactive unit is the stock statement."""
    if getattr(dm, "_prec_prepared", False):
        return
    for m in dm.modules():                                  # SingleConditioning.transitions: stock statements at instance level while dit_bf16 is active (see _sc_transition_forward)
        if type(m).__name__ == "SingleConditioning":
            for t in getattr(m, "transitions", []):
                if "forward" not in t.__dict__:
                    t.forward = _sc_transition_forward(t); _S.setdefault("inst", []).append((t, "forward"))
    _S.setdefault("inst", []).append((dm, "_prec_prepared"))
    enc, dec = dm.atom_attention_encoder, dm.atom_attention_decoder
    for layer in dm.token_transformer.layers:
        layer._prec_role = layer.pair_bias_attn._prec_role = "token"
    for tr in (enc.atom_encoder, dec.atom_decoder):
        for layer in tr.diffusion_transformer.layers:
            layer._prec_role = layer.pair_bias_attn._prec_role = "atom"
    lin = enc.r_to_q_trans                                   # LinearNoBias(3, atom_s): the network's only read of the noisy coordinates
    lin_fwd = lin.forward

    def r_to_q(r, _lin=lin, _orig=lin_fwd):
        if active("dit_bf16") and DEV["pin_rq"]:
            _c("dit_bf16", "pin_r_to_q")
            with torch.autocast("cuda", enabled=False):
                return F.linear(r.float(), _lin.weight.float(), None if _lin.bias is None else _lin.bias.float())
        return _orig(r)
    lin.forward = r_to_q; _S["inst"].append((lin, "forward"))
    head = dec.atom_feat_to_atom_pos_update                  # Sequential(LayerNorm(atom_s), LinearNoBias(atom_s, 3)): its only write
    head_fwd = head.forward

    def pos_update(q, _head=head, _orig=head_fwd):
        if active("dit_bf16") and DEV["pin_pos"]:
            _c("dit_bf16", "pin_pos_update")
            with torch.autocast("cuda", enabled=False):
                x = q.float()
                for m in _head:
                    x = m(x)
                return x
        return _orig(q)
    head.forward = pos_update; _S["inst"].append((head, "forward"))
    dm._prec_prepared = True


def _dm_call(self, *args, **kwargs):
    bf16, tf32 = active("dit_bf16"), active("dit_tf32")
    _prepare(self)
    if not (bf16 or tf32):
        return nn.Module.__call__(self, *args, **kwargs)
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            _c("dit_tf32", "calls")
        if bf16:
            _c("dit_bf16", "calls")
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                out = nn.Module.__call__(self, *args, **kwargs)
        else:
            out = nn.Module.__call__(self, *args, **kwargs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    if torch.backends.cuda.matmul.allow_tf32 is not False or torch.get_float32_matmul_precision() != "highest":   # the process policy outside the call is stock's ('highest', no TF32): asserted after EVERY call, counted; a violation raises (never a silent TF32 process)
        _S["errors"].append("tf32_flag_not_restored")
        raise RuntimeError("boltz2_precision: torch.backends.cuda.matmul.allow_tf32 / float32_matmul_precision not back to stock's (False / 'highest') after the score-model call")
    if tf32:
        _c("dit_tf32", "flag_restored")
    if isinstance(out, torch.Tensor) and out.dtype != torch.float32:   # the pinned head returns fp32; pin_pos off (evidence matrix) or a foreign forward might not
        _c("dit_bf16", "out_upcast"); out = out.float()
    return out


def _sc_transition_forward(t):
    """Instance-level forward for SingleConditioning's two Transition(768, 1536) modules while dit_bf16 is installed: the STOCK statements
    (transition.py:63-67: norm -> silu(fc1 x) * fc2 x -> fc3), which under this unit's autocast run as bf16 cuBLAS GEMMs. Why an instance forward:
    the kit's fused-transition adapter owns ``Transition.forward`` at class level and, once autocast is on here, would serve these calls — an
    UNPINNED (768,1536) cell, which engages opt_core's process-wide SAFE transition settings and slows every later trunk transition (~+80 %/call,
    trunk +7.5 %). The instance attribute wins over the class patch (nn.Module.__call__ resolves self.forward on the instance first), so the adapter
    never sees the cell; removed by disable(). (A pinned (768,1536,sm90) transition cell in the core's table would let the fused kernel serve them instead.)"""
    def forward(x, chunk_size=None):
        if not active("dit_bf16"):                           # unit off: whatever owns Transition.forward at class level (stock, or the kit's adapter: autocast is off there)
            return type(t).forward(t, x, chunk_size)
        _c("dit_bf16", "sc_transition_stock")
        x = t.norm(x)
        x = t.silu(t.fc1(x)) * t.fc2(x)
        x = t.fc3(x)
        return x
    return forward


def _fe_call(self, times):
    """FourierEmbedding.__call__: the shipped pin — the noise-level embedding in fp32 inside dit_bf16 (a [B,1]x[1,256] product + cos: free)."""
    if active("dit_bf16") and DEV["pin_fourier"] and torch.is_autocast_enabled():
        _c("dit_bf16", "pin_fourier")
        with torch.autocast("cuda", enabled=False):
            return nn.Module.__call__(self, times.float())
    return nn.Module.__call__(self, times)


def _sc_call(self, *args, **kwargs):
    """SingleConditioning.__call__: evidence-matrix flag cond_fp32 keeps the noise-level conditioning path fp32 inside dit_bf16."""
    if active("dit_bf16") and DEV["cond_fp32"] and torch.is_autocast_enabled():
        _c("dit_bf16", "cond_fp32_calls")
        with torch.autocast("cuda", enabled=False):
            return nn.Module.__call__(self, *[a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args],
                                      **{k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in kwargs.items()})
    return nn.Module.__call__(self, *args, **kwargs)


def _dtl_call(self, *args, **kwargs):
    """DiffusionTransformerLayer.__call__: evidence-matrix flags resid_bf16 / atom_resid_bf16 round the layer's output (the residual stream) to bf16."""
    out = nn.Module.__call__(self, *args, **kwargs)
    if active("dit_bf16") and isinstance(out, torch.Tensor) and out.dtype == torch.float32:
        role = getattr(self, "_prec_role", None)
        if (role == "token" and DEV["resid_bf16"]) or (role == "atom" and DEV["atom_resid_bf16"]):
            _c("dit_bf16", f"resid_bf16_{role}")
            return out.to(torch.bfloat16).to(torch.float32)     # the storage precision of a bf16 residual stream; fp32 dtype kept for stock's inner fp32 islands (encodersv2.py:481)
    return out


# =============================================================================================== attention: dit_attn_bf16 / seq_bf16
def _apb_sdpa_bf16(self, s, z, mask, k_in, multiplicity=1):
    """AttentionPairBias.forward with a precomputed bias (compute_pair_bias=False: the diffusion transformers): projections as the ambient policy
    runs them, the core as one bf16 SDPA with the additive bias (+ key-padding term) materialised once per call in bf16."""
    B = s.shape[0]
    H, D = self.num_heads, self.head_dim
    q = self.proj_q(s).view(B, -1, H, D).transpose(1, 2)
    k = self.proj_k(k_in).view(B, -1, H, D).transpose(1, 2)
    v = self.proj_v(k_in).view(B, -1, H, D).transpose(1, 2)
    zb = self.proj_z(z)                                      # Rearrange 'b ... h -> b h ...': a [B0, H, N, M] view of the precomputed bias (fp32 or bf16)
    if multiplicity > 1 and zb.shape[0] != B:
        zb = zb.repeat_interleave(multiplicity, 0)
    N, M = zb.shape[-2], zb.shape[-1]
    mterm = ((1 - mask.float()) * -self.inf).to(torch.bfloat16)[:, None, None, :]      # [B, 1, 1, M] (stock adds it to fp32 logits; -1e6 is held to 0.06 % in bf16 and saturates the softmax either way)
    bias = torch.empty((B, H, N, M), dtype=torch.bfloat16, device=s.device)
    torch.add(zb, mterm, out=bias)                          # ONE pass: strided read of the bias slice (+ convert) + broadcast mask term -> contiguous bf16
    with torch.autocast("cuda", enabled=False):
        o = F.scaled_dot_product_attention(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), attn_mask=bias,
                                           dropout_p=0.0, scale=1.0 / math.sqrt(D))
    o = o.transpose(1, 2).reshape(B, -1, self.c_s).to(v.dtype)
    g = self.proj_g(s).sigmoid()
    return self.proj_o(g * o)


def _apb_call(self, *args, **kwargs):
    if not self.compute_pair_bias:                           # the diffusion transformers (token + atom): a precomputed bias
        role = getattr(self, "_prec_role", None)
        if (role == "token" and active("dit_attn_bf16")) or (role == "atom" and active("dit_bf16") and DEV["atom_attn_bf16"]):
            _c("dit_attn_bf16" if role == "token" else "dit_bf16", "calls" if role == "token" else "atom_attn_bf16_calls")
            return _apb_sdpa_bf16(self, *args, **kwargs)
        return nn.Module.__call__(self, *args, **kwargs)
    if active("seq_bf16"):                                   # the Pairformer sequence attention (trunk / confidence pairformers)
        _c("seq_bf16", "attn_calls")
        with torch.autocast("cuda", dtype=torch.bfloat16):    # re-enabled inside stock's fp32 island: q/k/v/g/o and the LayerNorm(fp32 stats)+128->16 bias projection take bf16 operands
            out = _apb_sdpa_bf16(self, *args, **kwargs)      # proj_z = Sequential(LayerNorm, Linear, Rearrange) here: the same SDPA core on the projected bias
        return out.float()                                   # stock: s = s.float() + attention(...) — hand back fp32 as the island did
    return nn.Module.__call__(self, *args, **kwargs)


# =============================================================================================== install / remove / report
_CLASSES = (("dm_call", "boltz.model.modules.diffusionv2", "DiffusionModule", "_dm_call"),
            ("apb_call", "boltz.model.layers.attentionv2", "AttentionPairBias", "_apb_call"),
            ("fe_call", "boltz.model.modules.encodersv2", "FourierEmbedding", "_fe_call"),
            ("sc_call", "boltz.model.modules.encodersv2", "SingleConditioning", "_sc_call"),
            ("dtl_call", "boltz.model.modules.transformersv2", "DiffusionTransformerLayer", "_dtl_call"))


def enable(units: Iterable[str]) -> Dict[str, Any]:
    """Install the named units (idempotent; the set REPLACES the current one). Returns report()."""
    import importlib
    units = [u.strip() for u in units if u and u.strip()]
    bad = [u for u in units if u not in UNITS]
    if bad:
        raise ValueError(f"unknown precision unit(s) {bad}; known: {list(UNITS)}")
    O = _S["orig"]
    for key, mod, cls, fn in _CLASSES:                       # every unit installs the same four __call__ hooks: each decides per call by the unit set, so
        if key not in O:                                     # enable() may switch units between calls (the evidence probe does) with nothing re-installed
            C = getattr(importlib.import_module(mod), cls)
            O[key] = C.__dict__.get("__call__")              # None: inherited from nn.Module
            C.__call__ = globals()[fn]
    _S["on"] = set(units)
    for u in units:
        _S["census"].setdefault(u, {})
    return report()


def disable() -> None:
    """Remove every install: the classes' own attributes back (stock statements). Instance pins stay but run the module's own forward."""
    for obj, attr in _S.get("inst", []):                    # instance-level installs (conditioner transitions) — the pins are removed with their modules' attributes below
        try:
            del obj.__dict__[attr]
        except KeyError:
            pass
    _S["inst"] = []
    import importlib
    O = _S["orig"]
    for key, mod, cls, fn in _CLASSES:
        if key in O:
            C = getattr(importlib.import_module(mod), cls)
            if O[key] is None:
                try:
                    del C.__call__
                except AttributeError:
                    pass
            else:
                C.__call__ = O[key]
            del O[key]
    _S["on"] = set()


def report() -> Dict[str, Any]:
    dev = {k: v for k, v in DEV.items() if v != DEV_DEFAULTS[k]}
    return {"units_on": sorted(_S["on"]), "known": list(UNITS), "census": {k: dict(v) for k, v in _S["census"].items()},
            "sites": {u: SITES[u][0] for u in _S["on"]}, "errors": list(_S["errors"]), "dev_overrides": dev,
            "tf32_flag_now": bool(torch.backends.cuda.matmul.allow_tf32), "matmul_precision": torch.get_float32_matmul_precision(),
            "process_state": process_state()}


def process_state() -> Dict[str, Any]:
    """Process-wide numerics / dispatch state as read NOW (the adapter reads it at exit, on the main thread): what a unit could leak."""
    st = {"allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
          "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "matmul_precision": torch.get_float32_matmul_precision(),
          "bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
          "autocast_enabled": bool(torch.is_autocast_enabled()), "autocast_cache_enabled": bool(torch.is_autocast_cache_enabled()),
          "grad_enabled": bool(torch.is_grad_enabled()), "inference_mode": bool(torch.is_inference_mode_enabled())}
    try:
        st["autocast_gpu_dtype"] = str(torch.get_autocast_dtype("cuda"))
    except Exception:
        pass
    try:
        st["tls_include"] = str(torch._C._dispatch_tls_local_include_set()); st["tls_exclude"] = str(torch._C._dispatch_tls_local_exclude_set())
    except Exception:
        pass
    try:                                                     # caching-allocator health: cudaMalloc/cudaFree counts and cache-flush retries over the process
        ms = torch.cuda.memory_stats()
        st["alloc"] = {k: int(ms.get(k, -1)) for k in ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms", "num_sync_all_streams")}
        st["alloc"]["reserved_gib"] = round(torch.cuda.memory_reserved() / 2**30, 2)
    except Exception:
        pass
    return st
