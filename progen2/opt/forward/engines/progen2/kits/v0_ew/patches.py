"""The v0_ew kit's patches over the stock ProGen2 model (models.progen.modeling_progen of salesforce/progen @ c27a419c; transformers
4.16.2): each lever replaces ONE stock elementwise chain with ONE kernel of kernels.cu and counts every call (CTR).

Levers (EXACT_LEVERS; each independently switchable):
  gelu     : ProGenMLP.act (the instance attribute the stock sets to transformers.activations.gelu_new) -> ONE gelu_new kernel
             (the stock's 8 kernels: 3 scalar muls, pow, add, tanh, add, mul)
  rotary   : ProGenAttention.forward's qkv -> (query, key, value) section (reshape, split, 3 _split_heads copies, the per-call CPU
             sin/cos tables + H2D, 2 x (repeat_interleave x2, neg, stack, mul, mul, add), 2 cats with fp32 promotion) -> the stock
             function's OWN tables hoisted once per (device, rotary_dim) for n_positions rows + ONE rotary_split_qkv kernel; the
             rest of the stock forward verbatim (the cache cats, _attn, _merge_heads, out_proj)
  residual : ProGenBlock.forward's `attn_output + feed_forward_hidden_states + residual` (2 adds) -> ONE kernel
  glue     : ProGenAttention._attn's `/ scale_attn`, torch.where(causal, ., masked_bias), `+ attention_mask` (3 kernels) -> ONE
             kernel; QK^T, softmax, the value-dtype cast and AV stay the stock's
  ln       : nn.LayerNorm.forward bound per instance on every ln_1 and ln_f -> the vectorized layer-norm replica (the VARIANT recorded
             in BUILD.json ln_variant); under autocast the fp32 casts autocast performs (input, weight, bias) are done here,
             the fp32 weight/bias copies cached per instance (a cast is a bijection: exact by construction)
Every patch is a class-attribute / instance-attribute patch of the STOCK objects, undone by unapply().
SHADOW mode (off unless apply(shadow=True)) makes every patched site ALSO run the stock chain and count bit-pattern mismatches.
"""
from __future__ import annotations

import collections
import hashlib
import inspect

import torch

from . import ext
from .ext import KitRefused

CTR = collections.Counter()
EXACT_LEVERS = ("gelu", "rotary", "residual", "glue", "ln")
LN_VARIANT_DEFAULT = 0
_state = {"applied": [], "originals": [], "tables": {}, "model": None, "shadow": False, "ln_variant": LN_VARIANT_DEFAULT,
          "scalars": {}, "stock": None}


def _mp():
    from models.progen import modeling_progen as mp
    return mp


def _act():
    from transformers import activations
    return activations


def source_sha256(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def stock_source_pins() -> dict:
    """sha256 of the stock source this kit replaces / relies on (read from the installed packages): a drift refuses apply."""
    mp = _mp()
    A = _act()
    return {"ProGenAttention.forward": source_sha256(mp.ProGenAttention.forward), "ProGenAttention._attn": source_sha256(mp.ProGenAttention._attn),
            "ProGenBlock.forward": source_sha256(mp.ProGenBlock.forward), "ProGenMLP.forward": source_sha256(mp.ProGenMLP.forward),
            "fixed_pos_embedding": source_sha256(mp.fixed_pos_embedding), "apply_rotary_pos_emb": source_sha256(mp.apply_rotary_pos_emb),
            "rotate_every_two": source_sha256(mp.rotate_every_two), "gelu_new": source_sha256(A.gelu_new),
            "nn.LayerNorm.forward": source_sha256(torch.nn.LayerNorm.forward)}


def _stock():
    """The stock function objects, captured at first use (apply() asserts they are the package's own by source sha)."""
    if _state["stock"] is None:
        mp = _mp()
        _state["stock"] = {"fixed_pos_embedding": mp.fixed_pos_embedding, "apply_rotary_pos_emb": mp.apply_rotary_pos_emb,
                           "rotate_every_two": mp.rotate_every_two, "attn_forward": mp.ProGenAttention.forward, "_attn": mp.ProGenAttention._attn,
                           "block_forward": mp.ProGenBlock.forward, "gelu_new": _act().gelu_new, "ln_forward": torch.nn.LayerNorm.forward}
    return _state["stock"]


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(torch.equal(a.contiguous().view(torch.int16 if a.dtype == torch.float16 else torch.int32),
                            b.contiguous().view(torch.int16 if b.dtype == torch.float16 else torch.int32)))


def _shadow(site: str, kit_out: torch.Tensor, stock_out: torch.Tensor):
    CTR[f"shadow_{site}"] += 1
    if not _bits_equal(kit_out, stock_out):
        CTR[f"shadow_mismatch_{site}"] += 1
        d = (kit_out.float() - stock_out.float()).abs()
        CTR[f"shadow_mismatch_elems_{site}"] += int((kit_out.contiguous().view(torch.int16 if kit_out.dtype == torch.float16 else torch.int32) !=
                                                     stock_out.contiguous().view(torch.int16 if stock_out.dtype == torch.float16 else torch.int32)).sum().item())
        _state.setdefault("shadow_maxabs", {})[site] = max(_state.get("shadow_maxabs", {}).get(site, 0.0), float(d.max().item()))


# ------------------------------------------------------------------------------------------------------------ scalars
def gelu_scalars(dtype: torch.dtype) -> tuple[float, float]:
    """The chain's two non-trivial scalars as the mul kernels see them: the wrapped Python double read straight into the opmath
    float (TensorIterator keeps a CPU scalar operand as is; the kernel's scalar_value<float> converts double -> float) — the SAME
    fp32 constants for fp16 and fp32 tensors. Pinned by an exhaustive sweep: the fp16-rounded alternative differs on 156 of
    65,536 patterns, the fp32 constants match all 65,536 and all 2^32 fp32 patterns."""
    import math
    key = ("gelu", dtype)
    if key not in _state["scalars"]:
        c0 = torch.tensor(0.044715, dtype=torch.float64).to(torch.float32).item()
        c1 = torch.tensor(math.sqrt(2.0 / math.pi), dtype=torch.float64).to(torch.float32).item()
        _state["scalars"][key] = (c0, c1)
    return _state["scalars"][key]


def glue_scalars(attn, dtype: torch.dtype) -> tuple[float, float]:
    """inv_scale = opmath(1) / opmath(scale_attn) (torch's CPU-scalar true division), masked = masked_bias.to(dtype) as a float."""
    cache = attn.__dict__.setdefault("_v0_ew_glue", {})
    if dtype not in cache:
        scale = attn.scale_attn
        if scale.dim() != 0 or scale.device.type != "cpu":
            raise KitRefused(f"kit v0_ew: scale_attn is not the stock's 0-dim CPU tensor ({scale.shape}, {scale.device})")
        inv = (torch.tensor(1.0, dtype=torch.float32) / scale.to(torch.float32)).item()
        masked = attn.masked_bias.detach().cpu().to(dtype).to(torch.float32).item()
        cache[dtype] = (inv, masked)
    return cache[dtype]


# --------------------------------------------------------------------------------------------------------------- gelu
def _gelu_ew(x: torch.Tensor) -> torch.Tensor:
    c0, c1 = gelu_scalars(x.dtype)
    if torch.is_autocast_enabled() and x.is_cuda:
        # the likelihood route: torch.pow autocasts to fp32, the chain's tail is fp32, the fc_out Linear's autocast cast rounds
        # the fp32 result to fp16 -> folded here (exact by construction; shadow mode emits the stock's fp32 tensor to compare)
        if x.dtype != torch.float16:
            raise KitRefused(f"kit v0_ew: gelu under autocast expects the fc_in output in fp16 (got {x.dtype})")
        if _state["shadow"]:
            y32 = ext.gelu_new_autocast(x, c0, c1, out_half=False)
            _shadow("gelu_new_autocast", y32, _stock()["gelu_new"](x))
            CTR["gelu_new_autocast"] += 1
            return y32
        y = ext.gelu_new_autocast(x, c0, c1, out_half=True)
        CTR["gelu_new_autocast"] += 1
        return y
    y = ext.gelu_new(x, c0, c1)
    CTR["gelu_new"] += 1
    if _state["shadow"]:
        _shadow("gelu_new", y, _stock()["gelu_new"](x))
    return y


# ------------------------------------------------------------------------------------------------------------- rotary
def _tables(attn, device, rd: int, need: int):
    """(cos, sin) (n_pos, rd/2) fp32 = the stock fixed_pos_embedding computed ONCE on this device for n_pos = the model's
    n_positions (stock: sin, cos = fixed_pos_embedding(k_rot, 1, seq_len)); refuses a request beyond the table."""
    key = (str(device), rd)
    ent = _state["tables"].get(key)
    if ent is None:
        n_pos = int(_state["n_positions"])
        probe = torch.empty((1, 1, 1, rd), dtype=torch.float32, device=device)
        sin, cos = _stock()["fixed_pos_embedding"](probe, 1, seq_len=n_pos)
        ent = (cos.contiguous(), sin.contiguous())
        assert ent[0].shape == (n_pos, rd // 2) and ent[0].dtype == torch.float32, (ent[0].shape, ent[0].dtype)
        _state["tables"][key] = ent
        CTR["rotary_table_built"] += 1
    if need > ent[0].shape[0]:
        raise KitRefused(f"kit v0_ew: position {need} beyond the hoisted rotary table ({ent[0].shape[0]} = n_positions)")
    return ent


def table_sha() -> dict:
    """sha256 of the hoisted (cos, sin) tables per (device, rotary_dim) — the host-vendor witness of the resident-table lever."""
    import hashlib
    out = {}
    for (dev, rd), (cos, sin) in _state["tables"].items():
        out[f"{dev}/rd{rd}"] = {"cos_sha256": hashlib.sha256(cos.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                                "sin_sha256": hashlib.sha256(sin.detach().cpu().contiguous().numpy().tobytes()).hexdigest(), "rows": int(cos.shape[0])}
    return out


def _stock_qkv_section(self, qkv, layer_past):
    """The stock lines 158-197 (reshape/split/_split_heads/rotary/cat/permute) on a given qkv: the shadow reference."""
    mp = _mp()
    S = _stock()
    mp_num = 8
    qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))
    local_dim = self.head_dim * self.num_attention_heads // mp_num
    query, value, key = torch.split(qkv_split, local_dim, dim=-1)
    query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
    key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)
    value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
    value = value.permute(0, 2, 1, 3)
    seq_len = key.shape[1]
    offset = 0
    if layer_past is not None:
        offset = layer_past[0].shape[-2]
        seq_len += offset
    k_rot = key[:, :, :, : self.rotary_dim]
    k_pass = key[:, :, :, self.rotary_dim:]
    q_rot = query[:, :, :, : self.rotary_dim]
    q_pass = query[:, :, :, self.rotary_dim:]
    sincos = S["fixed_pos_embedding"](k_rot, 1, seq_len=seq_len)
    k_rot = S["apply_rotary_pos_emb"](k_rot, sincos, offset=offset)
    q_rot = S["apply_rotary_pos_emb"](q_rot, sincos, offset=offset)
    key = torch.cat([k_rot, k_pass], dim=-1)
    query = torch.cat([q_rot, q_pass], dim=-1)
    key = key.permute(0, 2, 1, 3)
    query = query.permute(0, 2, 1, 3)
    return query, key, value


def qkv_heads_rotary(self, qkv: torch.Tensor, layer_past=None, q_out=None, k_out=None, v_out=None):
    """The kit's qkv -> (query, key, value) in the stock's layouts: (B, H, L, hd) permuted views of contiguous (B, L, H, hd)
    buffers (query, key fp32; value in the qkv dtype). Optional out views (B, L, H, hd) with unit stride on hd (e.g. cache-slot
    views) replace the fresh buffers for key / value."""
    B, L = qkv.shape[0], qkv.shape[1]
    H, hd, rd = self.num_attention_heads, self.head_dim, self.rotary_dim
    if rd is None:
        raise KitRefused("kit v0_ew: rotary_dim is None (no shipped ProGen2 config takes the rotary-over-the-full-head branch)")
    offset = 0 if layer_past is None else int(layer_past[0].shape[-2])
    costab, sintab = _tables(self, qkv.device, rd, offset + L)
    query = q_out if q_out is not None else torch.empty((B, L, H, hd), dtype=torch.float32, device=qkv.device)
    key = k_out if k_out is not None else torch.empty((B, L, H, hd), dtype=torch.float32, device=qkv.device)
    value = v_out if v_out is not None else torch.empty((B, L, H, hd), dtype=qkv.dtype, device=qkv.device)
    ext.rotary_split_qkv(qkv, H, hd, rd, offset, costab, sintab, query, key, value)
    CTR["rotary_split_qkv"] += 1
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    if _state["shadow"]:
        sq, sk, sv = _stock_qkv_section(self, qkv, layer_past)
        _shadow("rotary_query", query, sq)
        _shadow("rotary_key", key, sk)
        _shadow("rotary_value", value, sv)
    return query, key, value


def _attn_forward_ew(self, hidden_states, attention_mask=None, layer_past=None, head_mask=None, use_cache=False, output_attentions=False):
    qkv = self.qkv_proj(hidden_states)
    query, key, value = qkv_heads_rotary(self, qkv, layer_past)
    if layer_past is not None:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
    if use_cache is True:
        present = (key, value)
    else:
        present = None
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)
    return outputs


# --------------------------------------------------------------------------------------------------------------- glue
def _attn_ew(self, query, key, value, attention_mask=None, head_mask=None):
    query_length, key_length = query.size(-2), key.size(-2)
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    inv_scale, masked_value = glue_scalars(self, attn_weights.dtype)
    if attention_mask is not None and attention_mask.dim() != 4:
        raise KitRefused(f"kit v0_ew: attention_mask rank {attention_mask.dim()} (the stock passes (B, 1, 1, key_len))")
    w = ext.attn_glue(attn_weights.contiguous(), key_length, inv_scale, masked_value, attention_mask)
    CTR["attn_glue"] += 1
    if _state["shadow"]:
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
        s = attn_weights / self.scale_attn
        s = torch.where(causal_mask, s, self.masked_bias.to(s.dtype))
        if attention_mask is not None:
            s = s + attention_mask
        _shadow("attn_glue", w, s)
    attn_weights = torch.nn.Softmax(dim=-1)(w)
    attn_weights = attn_weights.to(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


# ----------------------------------------------------------------------------------------------------------- residual
def _block_forward_ew(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask,
                             use_cache=use_cache, output_attentions=output_attentions)
    attn_output = attn_outputs[0]
    outputs = attn_outputs[1:]
    feed_forward_hidden_states = self.mlp(hidden_states)
    if attn_output.shape != residual.shape or attn_output.dtype != feed_forward_hidden_states.dtype:
        raise KitRefused(f"kit v0_ew: residual shapes/dtypes {attn_output.shape}/{attn_output.dtype} vs {residual.shape}/{feed_forward_hidden_states.dtype}")
    h = ext.residual_add2(attn_output, feed_forward_hidden_states, residual)
    CTR["residual_add2"] += 1
    if _state["shadow"]:
        _shadow("residual_add2", h, attn_output + feed_forward_hidden_states + residual)
    hidden_states = h
    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]
    return outputs


# ------------------------------------------------------------------------------------------------------------- layer norm
def _ln_fp32(self):
    ent = self.__dict__.get("_v0_ew_fp32")
    if ent is None or ent[0] is not self.weight or ent[1] is not self.bias:
        ent = (self.weight, self.bias, self.weight.detach().to(torch.float32).contiguous(), self.bias.detach().to(torch.float32).contiguous())
        self.__dict__["_v0_ew_fp32"] = ent
    return ent[2], ent[3]


def _ln_forward_ew(self, input):
    if torch.is_autocast_enabled() and input.device.type == "cuda":
        x = input.to(torch.float32)
        w, b = _ln_fp32(self)
        CTR["ln_autocast_fp32"] += 1
    else:
        x, w, b = input, self.weight, self.bias
    if x.dtype != w.dtype or x.dtype != b.dtype:
        raise KitRefused(f"kit v0_ew: LayerNorm dtypes input {x.dtype} weight {w.dtype} bias {b.dtype}")
    if tuple(self.normalized_shape) != (x.shape[-1],):
        raise KitRefused(f"kit v0_ew: LayerNorm normalized_shape {self.normalized_shape} vs {x.shape}")
    y = ext.layer_norm(x, w, b, self.eps, _state["ln_variant"])
    CTR["layer_norm"] += 1
    if _state["shadow"]:
        _shadow("layer_norm", y, _stock()["ln_forward"](self, input))
    return y


# ------------------------------------------------------------------------------------------------------------ apply
def _bind(obj, name, fn):
    orig = obj.__dict__.get(name)
    _state["originals"].append((obj, name, orig))
    if fn is None:
        obj.__dict__.pop(name, None)
    else:
        import types
        setattr(obj, name, types.MethodType(fn, obj))


def _set_attr(obj, name, value):
    _state["originals"].append((obj, name, obj.__dict__.get(name)))
    setattr(obj, name, value)


def _set_class_attr(cls, name, fn):
    _state["originals"].append((cls, name, cls.__dict__.get(name)))
    setattr(cls, name, fn)


def apply(model, levers=EXACT_LEVERS, shadow: bool = False, ln_variant: int = LN_VARIANT_DEFAULT) -> dict:
    """Patch a loaded ProGenForCausalLM in place; returns the apply record."""
    mp = _mp()
    if not isinstance(model, mp.ProGenForCausalLM):
        raise KitRefused(f"kit v0_ew: expected ProGenForCausalLM, got {type(model)}")
    if model.training:
        raise KitRefused("kit v0_ew: model.training is True")
    if _state["applied"]:
        raise KitRefused("kit v0_ew: already applied in this process")
    levers = tuple(levers)
    unknown = set(levers) - set(EXACT_LEVERS)
    if unknown:
        raise KitRefused(f"kit v0_ew: unknown levers {unknown}")
    S = _stock()
    if S["attn_forward"] is not mp.ProGenAttention.__dict__.get("forward") or S["_attn"] is not mp.ProGenAttention.__dict__.get("_attn") \
            or S["block_forward"] is not mp.ProGenBlock.__dict__.get("forward"):
        raise KitRefused("kit v0_ew: the stock classes were patched before this kit captured them (apply this kit FIRST)")
    cfg = model.config
    _state.update(model=model, shadow=bool(shadow), ln_variant=int(ln_variant), n_positions=int(cfg.n_positions))
    n_layer = int(cfg.n_layer)
    n_mlp = n_attn = n_block = n_ln = 0
    if "gelu" in levers:
        if cfg.activation_function != "gelu_new":
            raise KitRefused(f"kit v0_ew: activation_function {cfg.activation_function!r} (the kernel is gelu_new)")
        for mod in model.modules():
            if isinstance(mod, mp.ProGenMLP):
                if mod.act is not S["gelu_new"]:
                    raise KitRefused("kit v0_ew: ProGenMLP.act is not transformers.activations.gelu_new")
                _set_attr(mod, "act", _gelu_ew)
                n_mlp += 1
    if "rotary" in levers:
        for mod in model.modules():
            if isinstance(mod, mp.ProGenAttention):
                if mod.rotary_dim is None or mod.rotary_dim % 2 or mod.rotary_dim > mod.head_dim or mod.num_attention_heads % 8:
                    raise KitRefused(f"kit v0_ew: rotary_dim {mod.rotary_dim} head_dim {mod.head_dim} heads {mod.num_attention_heads}")
                n_attn += 1
        _set_class_attr(mp.ProGenAttention, "forward", _attn_forward_ew)
    if "glue" in levers:
        _set_class_attr(mp.ProGenAttention, "_attn", _attn_ew)
    if "residual" in levers:
        _set_class_attr(mp.ProGenBlock, "forward", _block_forward_ew)
        n_block = sum(isinstance(m, mp.ProGenBlock) for m in model.modules())
    if "ln" in levers:
        lns = [blk.ln_1 for blk in model.transformer.h] + [model.transformer.ln_f]
        for ln in lns:
            if not isinstance(ln, torch.nn.LayerNorm) or ln.weight is None or ln.bias is None or ln.normalized_shape[-1] % 8:
                raise KitRefused(f"kit v0_ew: LayerNorm {ln} is not the stock's affine LayerNorm over a multiple of 8")
            _bind(ln, "forward", _ln_forward_ew)
            n_ln += 1
    # everything the hot path would otherwise compute on first use is computed now, so no forward does it: the glue scalars of every
    # attention module for both weight dtypes (the masked_bias buffer read ONCE here, on the host), the rotary tables on the
    # model's device, the fp32 LayerNorm weight/bias copies for the autocast route
    dev = model.transformer.wte.weight.device
    for mod in model.modules():
        if isinstance(mod, mp.ProGenAttention):
            for dt in (torch.float32, torch.float16):
                glue_scalars(mod, dt)
    if "rotary" in levers and dev.type == "cuda":
        _tables(None, dev, int(cfg.rotary_dim), 1)
    if "ln" in levers:
        for ln in [blk.ln_1 for blk in model.transformer.h] + [model.transformer.ln_f]:
            _ln_fp32(ln)
    for dt in (torch.float32, torch.float16):
        gelu_scalars(dt)
    _state["applied"] = list(levers)
    CTR.clear()
    return {"levers": list(levers), "n_layer": n_layer, "n_mlp_patched": n_mlp, "n_attention": n_attn, "n_block_patched": n_block,
            "n_ln_patched": n_ln, "shadow": bool(shadow), "ln_variant": int(ln_variant), "n_positions": int(cfg.n_positions),
            "head_dim": int(cfg.n_embd // cfg.n_head), "rotary_dim": int(cfg.rotary_dim)}


def unapply():
    for (obj, name, orig) in reversed(_state["originals"]):
        if isinstance(obj, type):
            if orig is None:
                delattr(obj, name)
            else:
                setattr(obj, name, orig)
        else:
            if orig is None:
                obj.__dict__.pop(name, None)
            else:
                setattr(obj, name, orig)
    m = _state.get("model")
    if m is not None:
        for mod in m.modules():
            mod.__dict__.pop("_v0_ew_glue", None)
            mod.__dict__.pop("_v0_ew_fp32", None)
    _state["originals"].clear()
    _state["applied"] = []
    _state["shadow"] = False
    _state["tables"].clear()
    _state["scalars"].clear()
    _state["model"] = None
    CTR.clear()


def expected_counts(model, levers=None, autocast: bool = False) -> dict:
    """Counter expectations for ONE forward (any shape) under `levers`; autocast = the likelihood route's chain."""
    levers = _state["applied"] if levers is None else list(levers)
    n = int((model if model is not None else _state["model"]).config.n_layer)
    exp = {}
    if "gelu" in levers:
        exp["gelu_new" if not autocast else "gelu_new_autocast"] = n
    if "rotary" in levers:
        exp["rotary_split_qkv"] = n
    if "residual" in levers:
        exp["residual_add2"] = n
    if "glue" in levers:
        exp["attn_glue"] = n
    if "ln" in levers:
        exp["layer_norm"] = n + 1
    return exp


def counters_snapshot() -> dict:
    return {k: int(v) for k, v in CTR.items()}


def assert_counts(delta: dict, expected: dict, forwards: int = 1):
    bad = {k: (delta.get(k, 0), v * forwards) for k, v in expected.items() if delta.get(k, 0) != v * forwards}
    mism = {k: v for k, v in delta.items() if k.startswith("shadow_mismatch_") and v}
    if bad or mism:
        raise KitRefused(f"kit v0_ew path counts off: bad={bad} shadow_mismatches={mism} delta={delta} expected x{forwards}={expected}")
