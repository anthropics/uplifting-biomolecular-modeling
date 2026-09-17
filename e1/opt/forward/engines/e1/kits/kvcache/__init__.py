"""Kit ``kvcache`` for Profluent-E1 — the prefilled-cache forward (retrieval-augmented scoring: upstream's `KVCache` reuses a
context's key/values for every later batch of the same context) without the cache copies.

What upstream does on a batch whose `past_key_values` is prefilled with a context of C tokens (E1 @ bfd2620a):
  KVCache.before_forward   `DynamicCache.batch_repeat_interleave(B)`: every layer's (1, C, nkv, hd) K and V -> a (B, C, nkv, hd) COPY
  Attention.prepare_qkv    `DynamicCache.update`: torch.cat([cache (B, C), new (B, Lq)]) -> a (B, C+Lq) COPY, in EVERY layer —
                           the within-sequence layers then use only `[:, -Lq:]` of it (= the new K/V), the global layers gather it once
                           more (`_unpad_input`: index_first_axis) into flash_attn_varlen_func's packed layout
  KVCache.after_forward    `crop(C)` + `batch_select_indices([0])`: back to (1, C)
i.e. three passes over B x C x layers of K/V per batch (plus ~4 host synchronisations per global layer in `_get_unpad_data`) to feed a
kernel that reads each byte once.

LEVERS (each exact by construction — the same kernel, the same call, the same argument values; only the data movement changes):
    views            `batch_repeat_interleave(B)` on a batch-1 cache = an expand VIEW (all rows alias the one context row);
                     `batch_select_indices(idx)` on such a view = a view again (its rows are identical by construction). A cache whose
                     batch is not 1, or a tensor that is not an expanded view, takes the stock call (the prefill's select stays a copy,
                     so the cache never pins the prefill batch's storage).
    noappend_within  within-sequence layers of a prefilled forward use the new K/V as they are: bit-for-bit the stock's
                     `cat(cache, new)[:, -Lq:]`. Nothing is appended to the cache entry — upstream discards that appendix in
                     `after_forward` (`crop(C)`), so the cache object ends every batch in the stock's state.
    pack_global      global layers of a prefilled forward: the packed [context ; own tokens] K/V of every row is WRITTEN ONCE into
                     flash_attn_varlen_func's layout (no repeat_interleave, no cat, no gather), cu_seqlens / max_seqlen come from the
                     row layout computed once per forward (one device-to-host read of the row lengths instead of the stock's per-layer
                     nonzero / unique_consecutive / item / equal synchronisations), and the kernel is called exactly as
                     `E1.model.flash_attention.flash_attention_func` calls it. Rows with padding are packed row by row with the stock's
                     `index_first_axis` / `pad_input` on the query side.
Where the mechanism does not apply the stock path runs untouched: no cache / an empty cache (every single-sequence and every prefill
batch), `use_cache=False`, flash-attn absent (upstream's varlen-flex fallback), `output_attentions`, a cache batch other than 1 or B.
The decision is taken once per forward by a pre-hook on the inner `E1Model`; the two patches that must act together (`update` not
appending, `_flash_attn` packing) are interlocked — if either is not the attribute in force the forward runs the stock's cache protocol.
Under the eager kit the packing function sits BELOW the attn adapter (in its fallback slot): the adapter keeps routing the dense
within-sequence layers of cached batches (their K/V are the query's own now, so its A1 route applies) and hands the global layers down.

Counters (`counters()`): forwards, forwards_prefilled, forwards_stock:<reason>, views_expand, views_select, within_noappend,
global_stashed, global_pack, global_pack_padded, layout_computed, stock_update, stock_repeat_interleave, stock_select.
"""
from __future__ import annotations

import collections

import torch

KIT = "kvcache"
LEVERS = ("views", "noappend_within", "pack_global")
CTR = collections.Counter()
_R: dict = {"installed": False, "orig": {}, "hook": None, "model": None, "inner": None, "fwd": None, "layer_types": {}, "sig": None, "slot": None}


class KvcacheRefused(RuntimeError):
    """A precondition failed at install: nothing is left in force."""


class _Fwd:
    """Per-forward state of a prefilled-cache forward (set by the pre-hook, cleared by the post-hook)."""
    __slots__ = ("cache", "ctx_len", "active", "stash", "layout")

    def __init__(self, cache, ctx_len, active):
        self.cache, self.ctx_len, self.active = cache, ctx_len, active
        self.stash = {}
        self.layout = None


class _Layout:
    """The query-side row layout of one forward: computed once from `sequence_ids` (B, Lq), reused by every global layer."""
    __slots__ = ("B", "Lq", "ident", "qlens", "max_q", "cu_q", "cu_k", "idx_q", "total_q", "ctx_len")

    def __init__(self, sequence_ids: torch.Tensor, ctx_len: int):
        B, Lq = sequence_ids.shape
        valid = sequence_ids != -1
        qlens = [int(x) for x in valid.sum(dim=1).tolist()]            # ONE device-to-host read per forward
        self.B, self.Lq, self.ctx_len = int(B), int(Lq), int(ctx_len)
        self.qlens = qlens
        self.ident = all(n == Lq for n in qlens)
        self.max_q = max(qlens) if qlens else 0
        self.total_q = sum(qlens)
        dev = sequence_ids.device
        cu_q = [0]
        for n in qlens:
            cu_q.append(cu_q[-1] + n)
        cu_k = [cu_q[b] + b * self.ctx_len for b in range(B + 1)]
        self.cu_q = torch.tensor(cu_q, dtype=torch.int32, device=dev)
        self.cu_k = torch.tensor(cu_k, dtype=torch.int32, device=dev)
        # the stock's indices_q (nonzero of the flattened validity mask), needed only when a row is padded
        self.idx_q = None if self.ident else torch.nonzero(valid.flatten(), as_tuple=False).flatten()


# ----------------------------------------------------------------------------------------------------------- DynamicCache patches
def _batch_repeat_interleave(self, repeats: int) -> None:
    for i in range(len(self.key_cache)):
        k = self.key_cache[i]
        if not k.numel():
            continue
        v = self.value_cache[i]
        if k.shape[0] == 1 and v.shape[0] == 1:
            self.key_cache[i] = k.expand(int(repeats), *k.shape[1:])          # every row IS the context row: a view, no copy
            self.value_cache[i] = v.expand(int(repeats), *v.shape[1:])
            CTR["views_expand"] += 1
        else:
            self.key_cache[i] = k.repeat_interleave(repeats, dim=0)            # the stock call
            self.value_cache[i] = v.repeat_interleave(repeats, dim=0)
            CTR["stock_repeat_interleave"] += 1


def _is_expanded(t: torch.Tensor) -> bool:
    """True iff selecting rows of `t` can be a view of row 0 without pinning more memory than the stock's copy would hold: every
    batch row is the same memory (an expanded view of one row), or the batch is one row whose storage is (about) the row itself."""
    if t.dim() < 1 or t.shape[0] < 1:
        return False
    if t.shape[0] > 1:
        return t.stride(0) == 0
    row_bytes = t[0].numel() * t.element_size()
    return t.untyped_storage().nbytes() <= row_bytes + row_bytes // 4


def _batch_select_indices(self, indices) -> None:
    for i in range(len(self.key_cache)):
        k = self.key_cache[i]
        if not k.numel():
            continue
        v = self.value_cache[i]
        n = int(indices.numel()) if torch.is_tensor(indices) else len(indices)
        if n >= 1 and _is_expanded(k) and _is_expanded(v):
            self.key_cache[i] = k[:1].expand(n, *k.shape[1:]) if n != 1 else k[:1]      # rows identical by construction: any selection = row 0, n times
            self.value_cache[i] = v[:1].expand(n, *v.shape[1:]) if n != 1 else v[:1]
            CTR["views_select"] += 1
        else:
            self.key_cache[i] = k[indices, ...]                                 # the stock call (a copy: the prefill batch's storage is released)
            self.value_cache[i] = v[indices, ...]
            CTR["stock_select"] += 1


def _update(self, key_states, value_states, layer_idx: int):
    f = _R["fwd"]
    if (f is None or not f.active or f.cache is not self or layer_idx >= len(self.key_cache)
            or not self.key_cache[layer_idx].numel()):
        CTR["stock_update"] += 1
        return _R["orig"]["update"](self, key_states, value_states, layer_idx)
    if _R["layer_types"].get(layer_idx) == "global":
        f.stash[layer_idx] = (self.key_cache[layer_idx], self.value_cache[layer_idx])
        CTR["global_stashed"] += 1
    else:
        CTR["within_noappend"] += 1
    # nothing appended: the entry stays the context (upstream's after_forward crop(C) is then a no-op, its select a view)
    return key_states, value_states


# --------------------------------------------------------------------------------------------------------- Attention._flash_attn
def _flash_attn_kv(self, query_states, key_states, val_states, sequence_ids, attention_args=None, output_attentions=False):
    f = _R["fwd"]
    ent = f.stash.pop(self.layer_idx, None) if (f is not None and f.active) else None
    if ent is None:
        return _R["orig"]["_flash_attn"](self, query_states, key_states, val_states, sequence_ids, attention_args, output_attentions)
    from flash_attn import flash_attn_varlen_func
    from E1.model import flash_attention_utils as U
    assert not output_attentions, "Flash attention doesn't support returning attention masks"
    ck, cv = ent                                                   # (1 | B, C, nkv, hd): the context's keys / values (a view)
    B, Lq, nh, hd = query_states.shape
    nkv = key_states.shape[2]
    C = int(ck.shape[1])
    lay = f.layout
    if lay is None or lay.B != B or lay.Lq != Lq or lay.ctx_len != C:
        lay = _Layout(sequence_ids, C)
        f.layout = lay
        CTR["layout_computed"] += 1
    total_k = B * C + lay.total_q
    Kp = torch.empty((total_k, nkv, hd), dtype=key_states.dtype, device=key_states.device)
    Vp = torch.empty((total_k, nkv, hd), dtype=val_states.dtype, device=val_states.device)
    if lay.ident:
        Kv, Vv = Kp.view(B, C + Lq, nkv, hd), Vp.view(B, C + Lq, nkv, hd)
        Kv[:, :C].copy_(ck.expand(B, C, nkv, hd) if ck.shape[0] == 1 else ck)
        Vv[:, :C].copy_(cv.expand(B, C, nkv, hd) if cv.shape[0] == 1 else cv)
        Kv[:, C:].copy_(key_states)
        Vv[:, C:].copy_(val_states)
        q = query_states.reshape(B * Lq, nh, hd)
        CTR["global_pack"] += 1
    else:
        s = 0
        for b in range(B):
            n = lay.qlens[b]
            r = b if ck.shape[0] == B else 0
            Kp[s:s + C].copy_(ck[r])
            Vp[s:s + C].copy_(cv[r])
            if n:
                Kp[s + C:s + C + n].copy_(key_states[b, :n])
                Vp[s + C:s + C + n].copy_(val_states[b, :n])
            s += C + n
        q = U.index_first_axis(query_states.reshape(B * Lq, nh, hd), lay.idx_q)
        CTR["global_pack_padded"] += 1
    out = flash_attn_varlen_func(q, Kp, Vp, cu_seqlens_q=lay.cu_q, cu_seqlens_k=lay.cu_k, max_seqlen_q=lay.max_q, max_seqlen_k=C + lay.max_q,
                                 causal=False)
    if lay.ident:
        out = out.view(B, Lq, nh, hd)
    else:
        out = U.pad_input(out, lay.idx_q, B, Lq)
    return out.reshape(B, Lq, self.hidden_size).contiguous(), None


# ------------------------------------------------------------------------------------------------------------------ forward hooks
def _slot():
    """Where this kit's `_flash_attn` sits: BELOW the attn adapter when the adapter is in force (its fallback slot — the adapter keeps
    routing dense within-sequence layers and its own introspection stays valid), else the class attribute itself.
    Returns (get, set, name)."""
    from E1.model import attention as A
    try:
        from ..attn import adapter as AD
    except Exception:                                   # pragma: no cover - the adapter module is part of the tree
        AD = None
    if AD is not None and AD._S.get("cfg") is not None and A.Attention.__dict__.get("_flash_attn") is AD._flash_attn_routed:
        return (lambda: AD._S["orig"].get("_flash_attn"), lambda fn: AD._S["orig"].__setitem__("_flash_attn", fn), "attn_adapter_fallback")
    return (lambda: A.Attention.__dict__.get("_flash_attn"), lambda fn: setattr(A.Attention, "_flash_attn", fn), "Attention._flash_attn")


def _patches_in_force() -> bool:
    from E1 import dynamic_cache as DC
    slot = _R.get("slot")
    return (slot is not None and slot[0]() is _flash_attn_kv and DC.DynamicCache.update is _update
            and DC.DynamicCache.batch_repeat_interleave is _batch_repeat_interleave and DC.DynamicCache.batch_select_indices is _batch_select_indices)


def _pre_hook(module, args, kwargs):
    from E1.dynamic_cache import DynamicCache
    from E1.model.flash_attention import is_flash_attention_available
    bound = _R["sig"].bind_partial(*args, **kwargs)
    pkv = bound.arguments.get("past_key_values")
    use_cache = bool(bound.arguments.get("use_cache", False))
    CTR["forwards"] += 1
    if pkv is None or not use_cache or not isinstance(pkv, DynamicCache) or pkv.get_seq_length() <= 0:
        _R["fwd"] = None                                             # no prefilled cache: nothing of this kit is on the path
        return
    CTR["forwards_prefilled"] += 1
    reason = None
    if not _patches_in_force():
        reason = "patches_not_in_force"
    elif not is_flash_attention_available():
        reason = "flash_attn_unavailable"
    elif bool(bound.arguments.get("output_attentions", False)):
        reason = "output_attentions"
    if reason is not None:
        CTR[f"forwards_stock:{reason}"] += 1
        _R["fwd"] = _Fwd(pkv, int(pkv.get_seq_length()), active=False)
        return
    _R["fwd"] = _Fwd(pkv, int(pkv.get_seq_length()), active=True)


def _post_hook(module, args, kwargs, output):
    f = _R["fwd"]
    _R["fwd"] = None
    if f is not None and f.stash:
        # a stashed global layer that never reached _flash_attn would be a silent change: refuse loudly instead
        raise RuntimeError(f"kit kvcache: {len(f.stash)} global layer(s) bypassed the packed attention path")


# ------------------------------------------------------------------------------------------------------------------- surface
def _inner_model(model):
    import E1.modeling as M
    if isinstance(model, M.E1Model):
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, M.E1Model):
        return inner
    raise KvcacheRefused(f"kit {KIT}: no E1Model inside {type(model).__name__}")


def install(model) -> dict:
    """Patch DynamicCache (views / update) and Attention._flash_attn (over whatever is in force — the attn adapter's routed function
    under the eager kit) and hook the inner E1Model. Returns the install record. A second install raises."""
    import inspect
    import E1.modeling as M
    from E1 import dynamic_cache as DC
    from E1.model.attention import AttentionLayerType
    if _R["installed"]:
        raise KvcacheRefused(f"kit {KIT}: already installed in this process")
    inner = _inner_model(model)
    types = {}
    for layer in inner.layers:
        att = layer.norm_attn_norm.self_attn
        types[int(att.layer_idx)] = "global" if att.layer_type == AttentionLayerType.GLOBAL else "within"
    slot = _slot()
    below = slot[0]()
    if below is None:
        raise KvcacheRefused(f"kit {KIT}: no _flash_attn to install over at {slot[2]}")
    _R["orig"] = {"update": DC.DynamicCache.update, "batch_repeat_interleave": DC.DynamicCache.batch_repeat_interleave,
                  "batch_select_indices": DC.DynamicCache.batch_select_indices, "_flash_attn": below}
    _R["slot"] = slot
    DC.DynamicCache.update = _update
    DC.DynamicCache.batch_repeat_interleave = _batch_repeat_interleave
    DC.DynamicCache.batch_select_indices = _batch_select_indices
    slot[1](_flash_attn_kv)
    _R["sig"] = inspect.signature(M.E1Model.forward)
    _R["hook"] = (inner.register_forward_pre_hook(_pre_hook, with_kwargs=True), inner.register_forward_hook(_post_hook, with_kwargs=True))
    _R.update(installed=True, model=model, inner=inner, layer_types=types, fwd=None)
    CTR.clear()
    return {"kit": KIT, "levers": list(LEVERS), "n_global": sum(1 for t in types.values() if t == "global"),
            "n_within": sum(1 for t in types.values() if t == "within"), "slot": slot[2],
            "over": getattr(below, "__name__", str(below))}


def uninstall() -> None:
    if not _R["installed"]:
        return
    from E1 import dynamic_cache as DC
    from E1.model import attention as A
    o = _R["orig"]
    DC.DynamicCache.update = o["update"]
    DC.DynamicCache.batch_repeat_interleave = o["batch_repeat_interleave"]
    DC.DynamicCache.batch_select_indices = o["batch_select_indices"]
    slot = _R.get("slot")
    if slot is not None and slot[0]() is _flash_attn_kv:
        slot[1](o["_flash_attn"])
    elif A.Attention.__dict__.get("_flash_attn") is _flash_attn_kv:
        A.Attention._flash_attn = o["_flash_attn"]
    for h in (_R["hook"] or ()):
        try:
            h.remove()
        except Exception:
            pass
    _R.update(installed=False, orig={}, hook=None, model=None, inner=None, fwd=None, layer_types={}, sig=None, slot=None)


def in_force() -> bool:
    return bool(_R["installed"]) and _patches_in_force()


def counters() -> dict:
    return {k: int(v) for k, v in CTR.items()}


def is_pristine() -> dict:
    """True per patch point iff this kit's attribute is NOT in place (all True = nothing of this kit is installed)."""
    from E1 import dynamic_cache as DC
    from E1.model import attention as A
    try:
        from ..attn import adapter as AD
        below_adapter = AD._S.get("orig", {}).get("_flash_attn") is _flash_attn_kv
    except Exception:
        below_adapter = False
    return {"update": DC.DynamicCache.update is not _update, "batch_repeat_interleave": DC.DynamicCache.batch_repeat_interleave is not _batch_repeat_interleave,
            "batch_select_indices": DC.DynamicCache.batch_select_indices is not _batch_select_indices,
            "_flash_attn": A.Attention.__dict__.get("_flash_attn") is not _flash_attn_kv and not below_adapter}
