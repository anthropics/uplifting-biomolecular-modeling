"""The `ew1` kit's patches over the stock Profluent-E1 model (E1 @ bfd2620a): each lever replaces one stock code path with
the fused kernel of kernels.py and counts every call (CTR). Bound per instance on the model's own modules (never on
package globals); the q/k/v and w1/w3 projections receive the SAME hidden_states tensor object the stock passes them.

Levers (each independently switchable; the kit applies the exact set):
  rope     : Attention.prepare_qkv -> q/k/v GEMMs, then ONE clamp+RoPE kernel (the stock's 3 clamps, 2 cache casts,
             2 gathers, 2x(mul, neg, cat, mul, add)); flex (global) layers get the (B, H, L, hd) layout directly
  glu      : GLUMLP.forward -> w1, w3 GEMMs, ONE silu*mul kernel, w2 GEMM
  addnorm  : DecoderLayer.forward + E1Model.forward -> every `residual + x` followed by an RMSNorm becomes ONE fused
             add+rmsnorm kernel (the layer returns the (residual, delta) pair; the next norm consumes it; the final norm
             consumes the last pair); the first layer's input norm (the embeddings) stays the stock hub kernel
  embed    : E1Model.forward -> ONE gather+add+cast kernel for the two embedding lookups
The hub RMSNorm kernel's autotune pin (num_warps, engines.e1.kits.pins.apply_autotune_pin) is applied by the kit
surface before the first forward; the fused add+RMSNorm kernel launches at that num_warps.
"""
from __future__ import annotations

import collections
import hashlib
import inspect
import types
import weakref

import torch

CTR = collections.Counter()
_state = {"applied": [], "W": None, "rope_transposed": True, "originals": {}, "tables": weakref.WeakKeyDictionary(), "owns_model_forward": False,
          "n_cache_within": None, "n_cache_global": None,
          "model": None}
EXACT_LEVERS = ("rope", "glu", "addnorm", "embed")


class KitRefused(RuntimeError):
    """A pin / precondition of the kit failed: the process fails closed."""


def _modeling():
    import E1.modeling as M
    return M


def _K():
    """The Triton kernels module, imported at first use (the kit surface imports on a CPU host without Triton)."""
    from . import kernels
    return kernels


# ------------------------------------------------------------------------------------------------------- source pins
def source_sha256(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def stock_source_pins() -> dict:
    """sha256 of the stock methods this kit replaces (read from the installed package): a drift refuses apply."""
    M = _modeling()
    from E1.model import attention as A
    from E1.model import ffn as F_
    return {"E1Model.forward": source_sha256(M.E1Model.forward), "DecoderLayer.forward": source_sha256(M.DecoderLayer.forward),
            "NormAttentionNorm.forward": source_sha256(M.NormAttentionNorm.forward), "RMSNorm.forward": source_sha256(M.RMSNorm.forward),
            "Attention.prepare_qkv": source_sha256(A.Attention.prepare_qkv), "RotaryPositionalEmbedding.forward": source_sha256(A.RotaryPositionalEmbedding.forward),
            "GLUMLP.forward": source_sha256(F_.GLUMLP.forward)}


# ------------------------------------------------------------------------------------------------------------- rope
def _rope_tables(rot, device):
    """(cos_bf16, sin_bf16) of THIS rotary module: its own fp32 caches cast once to bf16 (the stock casts per call) — one table
    pair per module, held in a WeakKeyDictionary; no cross-module comparison, no device-to-host copy on any forward."""
    ent = _state["tables"].get(rot)
    if ent is None or ent["n"] != int(rot.max_seq_len_cached) or ent["device"] != str(device):
        ent = {"cos": rot.cos_cached.to(device=device, dtype=torch.bfloat16).contiguous(),
               "sin": rot.sin_cached.to(device=device, dtype=torch.bfloat16).contiguous(), "n": int(rot.max_seq_len_cached), "device": str(device)}
        _state["tables"][rot] = ent
        CTR["rope_table_built"] += 1
    return ent["cos"], ent["sin"]


def _prepare_qkv_ew(self, hidden_states, position_ids, past_key_value=None, use_cache=False):
    from E1.model.attention import AttentionLayerType
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    val_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
    val_states = val_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
    if self.clip_qkv is None:
        raise KitRefused("kit ew: clip_qkv is None (the kit's rope kernel is the clamp+rope of E1's configs)")
    rot = self.rotary_emb
    if not _state["owns_model_forward"]:
        # without the kit's E1Model.forward (its once-per-forward device-side range check) every layer checks its own ids
        torch._assert_async(((position_ids < rot.max_seq_len_cached) & (position_ids >= -1)).all(),
                            f"position ids must be in [-1, {rot.max_seq_len_cached})")
        CTR["range_check_async_layer"] += 1
    cos_t, sin_t = _rope_tables(rot, query_states.device)
    transposed = bool(_state["rope_transposed"]) and self.layer_type == AttentionLayerType.GLOBAL and not (use_cache and past_key_value is not None)
    query_states, key_states, val_states = _K().clamp_rope_qkv(query_states, key_states, val_states, position_ids, cos_t, sin_t,
                                                            self.clip_qkv, transposed)
    CTR["clamp_rope"] += 1
    if transposed:
        CTR["clamp_rope_transposed"] += 1
    if use_cache and past_key_value is not None:
        key_states, val_states = past_key_value.update(key_states, val_states, self.layer_idx)
    input_dtype = query_states.dtype
    if torch.is_autocast_enabled():
        target_dtype = torch.get_autocast_gpu_dtype()
    else:
        target_dtype = self.q_proj.weight.dtype
    if input_dtype != target_dtype:
        CTR["rope_dtype_cast"] += 1
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        val_states = val_states.to(target_dtype)
    return query_states, key_states, val_states


# -------------------------------------------------------------------------------------------------------------- glu
def _glu_forward_ew(self, hidden_states):
    a = self.w1(hidden_states)
    b = self.w3(hidden_states)
    CTR["silu_mul"] += 1
    return self.w2(_K().silu_mul(a, b))


# ---------------------------------------------------------------------------------------------------------- addnorm
def _add_rmsnorm(residual, x, norm, store_sum=True):
    W = _state["W"]
    if W is None:
        raise KitRefused("kit ew: the hub autotune pin (num_warps) is not set")
    y, s = _K().add_rmsnorm(residual, x, norm.weight, norm.variance_epsilon, W, store_sum=store_sum, round_sum=True)
    CTR["add_rmsnorm"] += 1
    return y, s


def _decoder_layer_forward_ew(self, hidden_states, within_seq_position_ids, global_position_ids, sequence_ids, attention_args=None,
                              past_key_value=None, output_attentions=False, use_cache=False):
    nan = self.norm_attn_norm
    if isinstance(hidden_states, tuple):
        residual_in, delta = hidden_states
        n1, h = _add_rmsnorm(residual_in, delta, nan.input_layernorm)
    else:
        h = hidden_states
        n1 = nan.input_layernorm(h)
        CTR["stock_rmsnorm"] += 1
    a, self_attn_weights, present_key_value = nan.self_attn(
        hidden_states=n1, within_seq_position_ids=within_seq_position_ids, global_position_ids=global_position_ids,
        sequence_ids=sequence_ids, attention_args=attention_args, past_key_value=past_key_value,
        output_attentions=output_attentions, use_cache=use_cache)
    n2, h2 = _add_rmsnorm(h, a, nan.post_attention_layernorm)
    f = self.ffn(n2)
    return (h2, f), self_attn_weights, present_key_value


# --------------------------------------------------------------------------------------- E1Model.forward (embed, addnorm)
def _e1model_forward_ew(self, input_ids, within_seq_position_ids, global_position_ids, sequence_ids, past_key_values=None,
                        use_cache=False, output_attentions=False, output_hidden_states=False):
    M = _modeling()
    if output_hidden_states:
        raise KitRefused("kit ew: output_hidden_states=True is outside the kit's outputs (logits + embeddings)")
    if self.training:
        raise KitRefused("kit ew: training mode")
    batch_size, seq_length = input_ids.shape
    if use_cache and past_key_values is None:
        past_key_values = M.DynamicCache()
    elif not use_cache:
        past_key_values = None
    global_position_ids = global_position_ids.view(-1, seq_length).long()
    within_seq_position_ids = within_seq_position_ids.view(-1, seq_length).long()
    sequence_ids = sequence_ids.view(-1, seq_length).long()
    # the stock's host-side range check (torch.max(...).item()) as DEVICE-side asserts: no sync, capture-safe, a trap on failure;
    # both position-id tensors are checked against the rotary caches the kit's rope kernel gathers from (never extended in-kernel)
    torch._assert_async(((within_seq_position_ids < _state["n_cache_within"]) & (within_seq_position_ids >= -1)).all(),
                        f"within-seq position ids must be in [-1, {_state['n_cache_within']})")
    torch._assert_async(((global_position_ids < _state["n_cache_global"]) & (global_position_ids >= -1)).all(),
                        f"global position ids must be in [-1, {_state['n_cache_global']})")
    CTR["range_check_async"] += 1
    if torch.is_autocast_enabled():
        target_dtype = torch.get_autocast_gpu_dtype()
    else:
        target_dtype = self.layers[0].norm_attn_norm.self_attn.q_proj.weight.dtype
    if "embed" in _state["applied"]:
        if target_dtype != torch.bfloat16:
            raise KitRefused(f"kit ew embed: target dtype {target_dtype} (the kernel's output is bf16 — the stock's autocast bf16 route)")
        hidden_states = _K().embed_add(input_ids, sequence_ids, self.embed_tokens.weight, self.embed_seq_id.weight)
        CTR["embed_add"] += 1
    else:
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds + self.embed_seq_id(sequence_ids.clamp(min=0))
        hidden_states = inputs_embeds.to(target_dtype)
    past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    attention_args = None
    if past_key_values_length == 0:
        block_mask = M.create_block_causal_mask_optimized(sequence_ids)
        flex_attention_args = M.FlexAttentionArgs(block_mask=block_mask)
        attention_args = M.AttentionArgs(flex_attention_args=flex_attention_args)
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    for decoder_layer in self.layers:
        layer_outputs = decoder_layer(
            hidden_states, within_seq_position_ids=within_seq_position_ids, global_position_ids=global_position_ids,
            sequence_ids=sequence_ids, attention_args=attention_args, past_key_value=past_key_values,
            output_attentions=output_attentions, use_cache=use_cache)
        hidden_states, self_attn_weights, present_key_value = layer_outputs
        if use_cache:
            next_decoder_cache = past_key_values = present_key_value
        if output_attentions:
            all_self_attns += (self_attn_weights,)
    if isinstance(hidden_states, tuple):
        hidden_states, _ = _add_rmsnorm(hidden_states[0], hidden_states[1], self.norm, store_sum=False)
    else:
        hidden_states = self.norm(hidden_states)
    next_cache = next_decoder_cache if use_cache else None
    return M.E1ModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=None, attentions=all_self_attns)


# ------------------------------------------------------------------------------------------------------ apply / undo
def _bind(obj, name, fn):
    key = (id(obj), name)
    if key not in _state["originals"]:
        _state["originals"][key] = (obj, name, obj.__dict__.get(name))
    setattr(obj, name, types.MethodType(fn, obj))


def apply(model, levers=EXACT_LEVERS, W=None, rope_transposed=True):
    """Patch a loaded E1ForMaskedLM in place. `W` = the num_warps the hub RMSNorm kernel is pinned at in this process (the
    fused add+RMSNorm kernel launches at the same value). Returns the apply record."""
    from E1.model.attention import Attention
    from E1.model.ffn import GLUMLP
    M = _modeling()
    if not isinstance(model, M.E1ForMaskedLM):
        raise KitRefused(f"kit ew: expected E1ForMaskedLM, got {type(model)}")
    if model.training:
        raise KitRefused("kit ew: model.training is True")
    if W is None:
        raise KitRefused("kit ew: no hub autotune pin (W)")
    _state["W"] = int(W)
    levers = tuple(levers)
    unknown = set(levers) - set(EXACT_LEVERS)
    if unknown:
        raise KitRefused(f"unknown levers {unknown}")
    cfg = model.config
    _state["rope_transposed"] = bool(rope_transposed)
    _state["model"] = model
    n_attn = n_glu = 0
    if "rope" in levers:
        if cfg.clip_qkv is None:
            raise KitRefused("clip_qkv is None")
        for mod in model.modules():
            if isinstance(mod, Attention):
                if mod.head_dim != 64:
                    raise KitRefused(f"head_dim {mod.head_dim} (kernel tested at 64)")
                if mod.num_kv_heads != mod.num_heads:
                    raise KitRefused("num_kv_heads != num_heads (not a shipped E1 config)")
                _bind(mod, "prepare_qkv", _prepare_qkv_ew)
                n_attn += 1
    if "glu" in levers:
        if not cfg.gated_mlp or cfg.hidden_act != "silu":
            raise KitRefused(f"gated_mlp={cfg.gated_mlp} hidden_act={cfg.hidden_act}: the silu*mul kernel is for E1's GLU-silu MLP")
        for mod in model.modules():
            if isinstance(mod, GLUMLP):
                if type(mod.act_fn).__name__ not in ("SiLU", "SiLUActivation"):
                    raise KitRefused(f"act_fn is {type(mod.act_fn).__name__}")
                _bind(mod, "forward", _glu_forward_ew)
                n_glu += 1
    if "addnorm" in levers:
        for mod in model.modules():
            if isinstance(mod, M.DecoderLayer):
                _bind(mod, "forward", _decoder_layer_forward_ew)
        for mod in model.modules():
            if isinstance(mod, M.RMSNorm) and mod.weight.dtype != torch.float32:
                raise KitRefused(f"RMSNorm weight dtype {mod.weight.dtype} (the pinned hub kernel loads fp32 weights)")
    from E1.model.attention import AttentionLayerType
    n_w = {int(m.rotary_emb.max_seq_len_cached) for m in model.modules() if isinstance(m, Attention) and m.layer_type != AttentionLayerType.GLOBAL}
    n_g = {int(m.rotary_emb.max_seq_len_cached) for m in model.modules() if isinstance(m, Attention) and m.layer_type == AttentionLayerType.GLOBAL}
    if len(n_w) != 1 or len(n_g) != 1:
        raise KitRefused(f"rotary cache lengths differ across layers: within {sorted(n_w)} global {sorted(n_g)}")
    _state["n_cache_within"], _state["n_cache_global"] = n_w.pop(), n_g.pop()
    _state["owns_model_forward"] = ("addnorm" in levers or "embed" in levers)
    if _state["owns_model_forward"]:
        _bind(model.model, "forward", _e1model_forward_ew)
    _state["applied"] = list(levers)
    CTR.clear()
    return {"levers": list(levers), "W": _state["W"], "rope_transposed": bool(rope_transposed),
            "n_attention_patched": n_attn, "n_glu_patched": n_glu, "num_hidden_layers": int(cfg.num_hidden_layers)}


def unapply():
    """Restore every patched bound method (levers can be toggled in one process)."""
    for (obj, name, orig) in list(_state["originals"].values()):
        if orig is None:
            obj.__dict__.pop(name, None)
        else:
            setattr(obj, name, orig)
    _state["originals"].clear()
    _state["applied"] = []
    _state["owns_model_forward"] = False
    CTR.clear()


def expected_counts(model, levers=None) -> dict:
    """Counter expectations for ONE forward under `levers` (one kernel call per patched site; `model` = the patched model or
    None for the one the kit holds)."""
    levers = _state["applied"] if levers is None else list(levers)
    n = int((model if model is not None else _state["model"]).config.num_hidden_layers)
    exp = {}
    if "rope" in levers:
        exp["clamp_rope"] = n
    if "glu" in levers:
        exp["silu_mul"] = n
    if "addnorm" in levers:
        exp["add_rmsnorm"] = 2 * n
        exp["stock_rmsnorm"] = 1
    if "embed" in levers:
        exp["embed_add"] = 1
    if "addnorm" in levers or "embed" in levers:
        exp["range_check_async"] = 1
    elif "rope" in levers:
        exp["range_check_async_layer"] = n
    return exp


def counters_snapshot() -> dict:
    return {k: int(v) for k, v in CTR.items()}


def assert_counts(delta: dict, expected: dict, forwards: int = 1):
    """Every expected path exactly forwards x its count; no fallback path (a dtype cast or a rotary-cache extension) taken."""
    bad = {k: (delta.get(k, 0), v * forwards) for k, v in expected.items() if delta.get(k, 0) != v * forwards}
    fb = {k: v for k, v in delta.items() if k in ("rope_dtype_cast", "rope_cache_extended") and v}
    if bad or fb:
        raise KitRefused(f"kit ew path counts off: bad={bad} unexpected={fb} delta={delta} expected x{forwards}={expected}")
