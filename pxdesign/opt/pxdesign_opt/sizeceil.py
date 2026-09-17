"""The package's size levers — the `registry.PACKAGE_LEVERS`, applied by the package-lever hook (`stack.apply_to`, after the kit's
`install(model)`) and nowhere else (`APPLIERS`: the hook's one dispatch table). The two in this file remove `N_atom²`-sized tensors that the
sampler never needs at that size; neither changes a value the model computes. The third, `rowpipe` (`rowpipe.py`, dispatched from here),
removes the `N_tok²`-sized fp32 conditioning plane from the hoist kit's prepare. The two tolerance-class levers (`fast` and `big`) are
dispatched from here as well: `tf32` (`precision.py`: the numerics policy) and `sdedup` (`sdedup.py`: the single-conditioning row dedup).

  featdiet  `InferenceRunner.predict` drops `input_feature_dict["bond_mask"]` before upstream's `to_device` (`runner/inference.py:129`):
            the `[N_atom, N_atom]` int64 bond-loss mask the featurizer emits for the training loss (`protenix/data/featurizer.py`), read by
            no inference code path (a reader would raise KeyError — the failure is loud, never numeric). 8 B × N_atom² never reaches the GPU.
  padmask   the atom transformer's windowed padding mask / padding bias (`n_queries` 32, `n_keys` 128) built from indices as
            `[n_trunks, 32, 128]` instead of sliced out of a dense `[N_atom(+pad), N_atom(+pad)]` ones / zeros tensor (4 B × N_atom² per
            construction, plus its `optimized_concat_split` copy): `primitives.rearrange_qk_to_dense_trunk(compute_mask=True)` (the
            encoder's geometry masks, `broadcast_token_to_local_atom_pair`), `primitives.rearrange_to_dense_trunk(attn_bias=None)`
            (`_local_attention` outside the hoist's cache), and the hoist kit's h5 cache (`hoist._MASK_CACHE`, seeded here so its dense
            construction — whose `unfold` view would keep the dense tensor resident per shape — never runs). Element for element the
            stock values (`window_valid` states the index algebra; `tests/test_sizeceil.py` checks torch.equal against the stock
            functions loaded from `stock/src`); tier 1.

`apply(names)` patches process-wide (class / module attributes, like the kit's own `install()`), is idempotent, and returns the hook's
record: `applied`, `fallback` (a lever whose patch points are not all in place after the call), `fields` (what was rebound), `reasons`.
"""
import functools
import math
import sys
from typing import Dict, Iterable

from . import precision, rowpipe, sdedup

TAG = "sizeceil"
N_QUERIES, N_KEYS = 32, 128                                  # AtomAttentionEncoder/Decoder window (protenix configs: atom_encoder n_queries / n_keys)
DIET_KEYS = ("bond_mask",)                                   # featurizer outputs no inference path reads
MASK_CACHE_CAP = 64                                          # hoist.py:284 — the kit's own cap on _MASK_CACHE entries
_STATE: Dict[str, object] = {"featdiet_dropped_bytes": 0, "calls": {"qk_mask": 0, "dense_bias": 0, "h5_seed": 0}}


# --------------------------------------------------------------------------------------------------------------- padmask algebra
def window_valid(n: int, n_queries: int, n_keys: int, device):
    """valid[b, i, j]: query row b·n_queries+i < n and key index b·n_queries + j − pad_left ∈ [0, n), pad_left = (n_keys − n_queries)//2 —
    the value stock obtains by zero-padding a dense [n, n] ones tensor to [n+q_pad, pad_left+n+pad_right], `optimized_concat_split`, and
    `unfold(-1, n_keys, step=W+n_queries)` (`primitives.py:292-407`): window b of query block b starts at padded column b·n_queries."""
    import torch
    n_trunks = int(math.ceil(n / n_queries))
    pad_left = (n_keys - n_queries) // 2
    t = torch.arange(n_trunks, device=device)[:, None, None] * n_queries
    i = torch.arange(n_queries, device=device)[None, :, None]
    j = torch.arange(n_keys, device=device)[None, None, :]
    g = t + j - pad_left
    return ((t + i) < n) & (g >= 0) & (g < n)                # [n_trunks, n_queries, n_keys] bool


def analytic_bias(n: int, n_queries: int, n_keys: int, inf: float, dtype, device, n_lead: int):
    """== `rearrange_to_dense_trunk(attn_bias=None)`'s `attn_bias_trunked`: zeros of q.dtype, −inf at padded positions,
    shape [(1,)·n_lead, n_trunks, n_queries, n_keys] (stock returns a strided view of the dense tensor holding these values)."""
    import torch
    valid = window_valid(n, n_queries, n_keys, device)
    bias = torch.zeros(valid.shape, dtype=dtype, device=device).masked_fill_(~valid, -inf)
    return bias.reshape(*(1,) * n_lead, *valid.shape)


def _wrap_qk(stock_fn):
    @functools.wraps(stock_fn)
    def rearrange_qk_to_dense_trunk(q, k, dim_q, dim_k, n_queries: int = 32, n_keys: int = 128, compute_mask: bool = True):
        q_tr, k_tr, info = stock_fn(q, k, dim_q, dim_k, n_queries=n_queries, n_keys=n_keys, compute_mask=False)
        if compute_mask:
            q0 = q[0] if isinstance(q, list) else q
            dq = dim_q[0] if isinstance(dim_q, list) else dim_q
            m = window_valid(q0.size(dq), n_queries, n_keys, q0.device)
            info["mask_trunked"] = m.reshape(*(1,) * len(q0.shape[:-2]), *m.shape)     # stock: (1,)·len(q[0].shape[:-2]) lead dims, .bool()
            _STATE["calls"]["qk_mask"] += 1
        return q_tr, k_tr, info
    rearrange_qk_to_dense_trunk._pxdesign_opt_lever = "padmask"
    return rearrange_qk_to_dense_trunk


def _wrap_dense(stock_fn, stock_qk_fn):
    @functools.wraps(stock_fn)
    def rearrange_to_dense_trunk(q, k, v, n_queries: int, n_keys: int, attn_bias=None, inf: float = 1e10):
        if attn_bias is not None:                            # a given bias: the stock pad-and-window path (no dense construction there)
            return stock_fn(q, k, v, n_queries=n_queries, n_keys=n_keys, attn_bias=attn_bias, inf=inf)
        q_tr, kv_tr, info = stock_qk_fn(q=q, k=[k, v], dim_q=-2, dim_k=[-2, -2], n_queries=n_queries, n_keys=n_keys, compute_mask=False)
        bias = analytic_bias(q.shape[-2], n_queries, n_keys, inf, q.dtype, q.device, len(q.shape[:-2]))
        _STATE["calls"]["dense_bias"] += 1
        return q_tr, kv_tr[0], kv_tr[1], bias, info["q_pad"]
    rearrange_to_dense_trunk._pxdesign_opt_lever = "padmask"
    return rearrange_to_dense_trunk


def _wrap_h5(hoist_mod):
    """The kit's h5 (`_local_attention_hoisted`) with `_MASK_CACHE` seeded analytically under the kit's own key (hoist.py:276)."""
    hoisted = hoist_mod._local_attention_hoisted

    @functools.wraps(hoisted)
    def _local_attention(q, k, v, n_queries, n_keys, attn_bias=None, trunked_attn_bias=None, inf=1e10, use_efficient_implementation=False,
                         inplace_safe=False, chunk_size=None):
        if attn_bias is None and hoist_mod._STATE.get("cache") is not None:
            n = int(q.shape[-2])
            key = (n, int(n_queries), int(n_keys), float(inf), q.dtype, q.device, len(q.shape[:-2]))
            cache = hoist_mod._MASK_CACHE
            if key not in cache:
                cache[key] = analytic_bias(n, n_queries, n_keys, inf, q.dtype, q.device, len(q.shape[:-2]))
                _STATE["calls"]["h5_seed"] += 1
                while len(cache) > MASK_CACHE_CAP:
                    cache.pop(next(iter(cache)))
        return hoisted(q, k, v, n_queries, n_keys, attn_bias=attn_bias, trunked_attn_bias=trunked_attn_bias, inf=inf,
                       use_efficient_implementation=use_efficient_implementation, inplace_safe=inplace_safe, chunk_size=chunk_size)
    _local_attention._pxdesign_opt_lever = "padmask"
    return _local_attention


def apply_padmask(hoist_mod=None) -> dict:
    import protenix.model.modules.primitives as PR
    import protenix.model.modules.transformer as TR
    fields = {}
    if getattr(PR.rearrange_qk_to_dense_trunk, "_pxdesign_opt_lever", None) != "padmask":
        stock_qk = PR.rearrange_qk_to_dense_trunk
        PR._pxdesign_opt_stock_rearrange_qk_to_dense_trunk = stock_qk
        PR.rearrange_qk_to_dense_trunk = _wrap_qk(stock_qk)
        if getattr(TR, "rearrange_qk_to_dense_trunk", None) is stock_qk:          # transformer.py imports the name (AtomAttentionEncoder)
            TR.rearrange_qk_to_dense_trunk = PR.rearrange_qk_to_dense_trunk
        PR._pxdesign_opt_stock_rearrange_to_dense_trunk = PR.rearrange_to_dense_trunk
        PR.rearrange_to_dense_trunk = _wrap_dense(PR.rearrange_to_dense_trunk, stock_qk)
    hoist_mod = hoist_mod or sys.modules.get("pxd_xattempt.hoist")
    if hoist_mod is not None and PR._local_attention is getattr(hoist_mod, "_local_attention_hoisted", None):
        PR._local_attention = _wrap_h5(hoist_mod)
    fields["rearrange_qk_to_dense_trunk"] = getattr(PR.rearrange_qk_to_dense_trunk, "_pxdesign_opt_lever", None) == "padmask"
    fields["rearrange_to_dense_trunk"] = getattr(PR.rearrange_to_dense_trunk, "_pxdesign_opt_lever", None) == "padmask"
    fields["transformer_alias"] = getattr(getattr(TR, "rearrange_qk_to_dense_trunk", None), "_pxdesign_opt_lever", None) == "padmask"
    fields["h5_seeded"] = getattr(PR._local_attention, "_pxdesign_opt_lever", None) == "padmask" if hoist_mod is not None else None
    return fields


# ---------------------------------------------------------------------------------------------------------------------- featdiet
def featdiet(data) -> int:
    """Drop DIET_KEYS from a featurized item in place; returns the bytes that will not be copied to the device."""
    import torch
    f = data.get("input_feature_dict") if isinstance(data, dict) else None
    if f is None:
        return 0
    freed = 0
    for k in DIET_KEYS:
        t = f.pop(k, None)
        if torch.is_tensor(t):
            freed += t.numel() * t.element_size()
    _STATE["featdiet_dropped_bytes"] = int(_STATE["featdiet_dropped_bytes"]) + freed
    return freed


def apply_featdiet() -> dict:
    from pxdesign.runner import inference as I
    cls = I.InferenceRunner
    if getattr(cls.predict, "_pxdesign_opt_lever", None) != "featdiet":
        inner = cls.predict

        @functools.wraps(inner)
        def predict(self, data, *a, **kw):
            featdiet(data)
            return inner(self, data, *a, **kw)
        predict._pxdesign_opt_lever = "featdiet"
        cls.predict = predict
    return {"predict_wrapped": getattr(cls.predict, "_pxdesign_opt_lever", None) == "featdiet"}


APPLIERS = {"featdiet": lambda hoist_mod=None: apply_featdiet(), "padmask": lambda hoist_mod=None: apply_padmask(hoist_mod),
            "rowpipe": lambda hoist_mod=None: rowpipe.apply_rowpipe(hoist_mod), "tf32": lambda hoist_mod=None: precision.apply_tf32(hoist_mod),
            "sdedup": lambda hoist_mod=None: sdedup.apply(hoist_mod)}
REQUIRED_FIELDS = {"featdiet": ("predict_wrapped",), "padmask": ("rearrange_qk_to_dense_trunk", "rearrange_to_dense_trunk", "h5_seeded"),
                   "rowpipe": ("prepare_cache_rebound", "resolved_by_global"), "tf32": ("applied",), "sdedup": ("hook",)}
RUNTIME_GATES = {"sdedup": sdedup.gate}   # levers whose activation evidence completes at run time (counts): name -> () -> [problem sentences]; the design verb folds them into its exit (stack.runtime_gate)
RUNTIME_CENSUS = {"sdedup": sdedup.stats}  # name -> () -> dict: the run-time counters the manifest and the EXIT census carry


def apply(names: Iterable[str], hoist_mod=None) -> dict:
    """Apply the named package levers; returns the hook's record (stack.classify's `package` block): planned / applied / fallback /
    skipped, `fields` = one flat evidence field per patch point (`<lever>_<point>=True|False|None`, the PACKAGE line's k=v cells),
    `reasons` for every lever not applied. A lever whose patch points are not all in place after the call is a fallback, by name."""
    names = list(names)
    rec = {"planned": names, "applied": [], "fallback": [], "skipped": [], "fields": {}, "reasons": {}}
    for name in names:
        if name not in APPLIERS:
            rec["fallback"].append(name); rec["reasons"][name] = f"unknown package lever {name!r} (registry.PACKAGE_LEVERS: {sorted(APPLIERS)})"; continue
        try:
            fields = APPLIERS[name](hoist_mod)
        except (ImportError, AttributeError) as e:            # a module or patch point that is not there: named, never silent — the activation is partial (fallbacks=<name>) and takes its NOT ACTIVE route; anything else propagates
            rec["fallback"].append(name); rec["reasons"][name] = f"{type(e).__name__}: {e}"; continue
        rec["fields"].update({f"{name}_{k}": v for k, v in fields.items()})
        missing = [f for f in REQUIRED_FIELDS[name] if fields.get(f) is False]
        if missing:
            rec["fallback"].append(name); rec["reasons"][name] = "patch points not in place after apply(): " + ",".join(missing)
        else:
            rec["applied"].append(name)
    return rec


def stats() -> dict:
    return {"featdiet_dropped_bytes": _STATE["featdiet_dropped_bytes"], "calls": dict(_STATE["calls"])}
