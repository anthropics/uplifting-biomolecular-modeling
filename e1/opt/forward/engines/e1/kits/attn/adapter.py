"""Attention-kernel adapter for Profluent-E1: one routing decision per forward, the stock code path for everything
the decision does not cover.

Dense mode = every row of the batch is ONE sequence with no padding (every masked-marginal batch the scorer builds).
In dense mode the WITHIN_SEQ layers run ``flash_attn_varlen_func`` on the flat (B*L) views with shape-derived
cu_seqlens (no unpad gathers, no pad scatter, no host syncs) and the GLOBAL layers run ``flex_attention`` compiled
with the pinned kernel options under the index-only all-true BlockMask of the stock's block structure (the stock's
block-causal document mask is all-true per row there). Any other batch (padding, multi-sequence rows, a prefilled
cache) takes the stock code path verbatim, with the unpad index data computed once per forward.

Every patched path counts its calls (``counters()``); the per-forward expectation is ``expected_per_forward``.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import inspect

import torch

ROUTES = ("stock", "fa2_varlen", "flash_dense", "fa2_kvcache_ns1", "flex", "sdpa_cudnn", "sdpa_flash", "sdpa_efficient", "sdpa_math", "fa3")
MASK_MODES = ("cheap", "none", "cached", "stock")


@dataclasses.dataclass(frozen=True)
class Config:
    within: str = "flash_dense"
    global_: str = "flex"
    flex_kernel_options: tuple = ()      # sorted (key, value) pairs -> kernel_options=dict(...); () = torch's default config
    flex_block_mask: str = "cheap"       # dense mode: "cheap" (index-only all-true mask_mod, the stock's block structure, cached per shape),
                                         # "none" (no mask object), "cached" (stock mask cached per content), "stock" (built per forward)
    flex_dynamic: bool = True
    unpad_once: bool = True              # fallback path: unpad index data once per forward (+ pass-through fast path)

    def __post_init__(self):
        assert self.within in ROUTES and self.global_ in ROUTES, (self.within, self.global_)
        assert self.flex_block_mask in MASK_MODES, self.flex_block_mask
        assert isinstance(self.flex_kernel_options, tuple) and self.flex_kernel_options == tuple(sorted(self.flex_kernel_options))

    def kernel_options(self) -> dict | None:
        return dict(self.flex_kernel_options) if self.flex_kernel_options else None

    def as_dict(self) -> dict:
        return {"within": self.within, "global": self.global_, "flex_kernel_options": dict(self.flex_kernel_options),
                "flex_block_mask": self.flex_block_mask, "flex_dynamic": self.flex_dynamic, "unpad_once": self.unpad_once}


class _Forward:
    __slots__ = ("dense", "unpad", "block_mask", "mask_key")

    def __init__(self, dense: bool):
        self.dense = dense
        self.unpad = {}
        self.block_mask = None
        self.mask_key = None


_S: dict = {"cfg": None, "fwd": None, "model": None, "hook": None, "sig": None, "orig": {}, "flex_fns": {},
            "mask_cache": {}, "cheap_masks": {}, "cu_seqlens": {}, "dense_by_key": {}, "fa3": None, "ctr": collections.Counter()}


def all_true_mask_mod(b, h, q_idx, kv_idx):
    """Dense mode: the stock document mask `sequence_ids[b,q] >= sequence_ids[b,kv] & both != -1` is true for every
    in-range (q, kv) pair of a single-sequence row, so create_block_mask classifies blocks by the sequence-length
    boundary alone; this index-only mask_mod yields the same BlockMask structure without the per-score gathers."""
    return (q_idx >= 0) & (kv_idx >= 0)


def cheap_block_mask(B: int, L: int, device):
    """The all-true BlockMask of the stock's structure for (B, L), cached per shape (content-free). Built on the CPU
    (integer block bookkeeping only) and moved to ``device``: the kit's dense path then launches no block-mask
    construction kernel at all."""
    from torch.nn.attention.flex_attention import create_block_mask
    key = (int(B), int(L), str(device))
    bm = _S["cheap_masks"].get(key)
    if bm is None:
        bm = create_block_mask(all_true_mask_mod, int(B), 1, int(L), int(L), device="cpu")
        if str(device) != "cpu":
            bm = bm.to(device)
        _S["cheap_masks"][key] = bm
        _S["ctr"]["cheap_mask_built"] += 1
    return bm


def dense_cu_seqlens(B: int, L: int, device):
    """cu_seqlens of a dense batch of B rows of length L (what _get_unpad_data returns for it), cached per shape."""
    key = (int(B), int(L), str(device))
    cu = _S["cu_seqlens"].get(key)
    if cu is None:
        cu = torch.arange(0, (int(B) + 1) * int(L), int(L), dtype=torch.int32, device=device)
        _S["cu_seqlens"][key] = cu
    return cu


# ------------------------------------------------------------------------------------------------------------ kernels
def _flex_fn(cfg: Config):
    """The compiled flex_attention for ``cfg`` (one distinct code object per kernel-options set: dynamo caches per code
    object, and every set is its own compile)."""
    key = (cfg.flex_kernel_options, cfg.flex_dynamic)
    fn = _S["flex_fns"].get(key)
    if fn is None:
        from torch.nn.attention.flex_attention import flex_attention as _eager
        tag = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
        src = (f"def flex_{tag}(q, k, v, block_mask=None):\n"
               f"    return _eager(q, k, v, block_mask=block_mask, kernel_options=_KO)\n")
        ns = {"_eager": _eager, "_KO": cfg.kernel_options()}
        exec(src, ns)
        fn = ns[f"flex_{tag}"]
        if torch.cuda.is_available():                                    # the stock compiles flex only on CUDA (flex_attention.py L8-9)
            fn = torch.compile(fn, dynamic=cfg.flex_dynamic)
        _S["flex_fns"][key] = fn
    return fn


def flex_fn(kernel_options: dict | None = None, dynamic: bool = True):
    return _flex_fn(Config(flex_kernel_options=tuple(sorted((kernel_options or {}).items())), flex_dynamic=dynamic))


def _fa3():
    if _S["fa3"] is None:
        from kernels import get_kernel
        _S["fa3"] = get_kernel("kernels-community/flash-attn3")
    return _S["fa3"]


def run_kernel(route: str, q, k, v, block_mask=None, cfg: Config | None = None):
    """Bidirectional attention over each row of (B, L, nh, hd) q/k/v -> (B, L, nh, hd)."""
    if route == "flash_dense":
        from flash_attn import flash_attn_func
        return flash_attn_func(q, k, v, causal=False)
    if route == "fa2_varlen":
        from flash_attn import flash_attn_varlen_func
        B, L, nh, hd = q.shape
        cu = dense_cu_seqlens(B, L, q.device)
        out = flash_attn_varlen_func(q.reshape(B * L, nh, hd), k.reshape(B * L, k.shape[2], hd), v.reshape(B * L, v.shape[2], hd), cu, cu, L, L, causal=False)
        return out.view(B, L, nh, hd)
    if route == "fa2_kvcache_ns1":
        from flash_attn import flash_attn_with_kvcache
        return flash_attn_with_kvcache(q, k, v, causal=False, num_splits=1)
    if route == "flex":
        fn = _flex_fn(cfg) if cfg is not None else flex_fn()
        out = fn(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), block_mask=block_mask)
        return out.transpose(1, 2)
    if route.startswith("sdpa_"):
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        be = {"sdpa_cudnn": SDPBackend.CUDNN_ATTENTION, "sdpa_flash": SDPBackend.FLASH_ATTENTION,
              "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION, "sdpa_math": SDPBackend.MATH}[route]
        with sdpa_kernel([be]):
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False)
        return out.transpose(1, 2)
    if route == "fa3":
        m = _fa3()
        f = getattr(m, "flash_attn_func", None) or getattr(getattr(m, "flash_attn_interface", None), "flash_attn_func")
        out = f(q, k, v, causal=False)
        return out[0] if isinstance(out, (tuple, list)) else out
    raise ValueError(route)


# ------------------------------------------------------------------------------------------------------ routing state
def _dense(sequence_ids: torch.Tensor) -> bool:
    """The dense decision from the device tensor: one host sync per forward (counted as ``dense_check_sync``)."""
    return bool(((sequence_ids != -1) & (sequence_ids == sequence_ids[:, :1])).all().item())


def _graph_layout():
    """The host-side layout of the forward being run from an optional ``engines.e1.kits.graph`` module ({key: (B, T,
    blake2b(sequence_ids bytes)), ...}, computed before the H2D copy) — None when that module is absent or not in a forward."""
    g = _S.get("graph_mod")
    if g is None:
        try:
            from engines.e1.kits import graph as g
        except Exception:
            g = False
        _S["graph_mod"] = g
    if not g:
        return None
    try:
        return g.current_layout()
    except Exception:
        return None


def _pre_hook(module, args, kwargs):
    bound = _S["sig"].bind_partial(*args, **kwargs)
    sid = bound.arguments.get("sequence_ids")
    if sid is None:
        _S["fwd"] = None
        return
    sid = sid.view(-1, sid.shape[-1]).long()
    # capture safety: the dense decision is a host sync on the device tensor; when a host-side layout key is available it is
    # taken ONCE per content key (B, T, blake2b(sequence_ids)) and reused sync-free
    layout = _graph_layout()
    key = layout.get("key") if isinstance(layout, dict) else None
    if key is not None and key in _S["dense_by_key"]:
        dense = _S["dense_by_key"][key]
        _S["ctr"]["dense_from_layout_key"] += 1
    else:
        dense = _dense(sid)
        _S["ctr"]["dense_check_sync"] += 1
        if key is not None:
            _S["dense_by_key"][key] = dense
    _S["fwd"] = _Forward(dense)
    _S["ctr"]["forwards"] += 1
    _S["ctr"]["forwards_dense" if dense else "forwards_fallback"] += 1


def _unpad_data_cached(st: _Forward, ids: torch.Tensor):
    from E1.model import flash_attention_utils as U
    key = (ids.data_ptr(), tuple(ids.shape))
    d = st.unpad.get(key)
    if d is None:
        idx, cu, mx = U._get_unpad_data(ids)
        B, L = ids.shape
        ident = (idx.numel() == B * L) and bool((idx == torch.arange(B * L, device=idx.device)).all())
        d = (idx, cu, mx, ident)
        st.unpad[key] = d
        _S["ctr"]["unpad_computed"] += 1
    else:
        _S["ctr"]["unpad_reused"] += 1
    return d


def _unpad_input_once(q, k, v, q_ids, k_ids):
    """Fallback path: the stock _unpad_input with the index data computed once per forward; no-op gathers skipped."""
    from E1.model import flash_attention_utils as U
    st = _S["fwd"]
    cfg = _S["cfg"]
    if st is None or cfg is None or not cfg.unpad_once:
        return _S["orig"]["_unpad_input"](q, k, v, q_ids, k_ids)
    idx_k, cu_k, mx_k, ident_k = _unpad_data_cached(st, k_ids)
    if q_ids is k_ids or torch.equal(q_ids, k_ids):
        idx_q, cu_q, mx_q, ident_q = idx_k, cu_k, mx_k, ident_k
    else:
        idx_q, cu_q, mx_q, ident_q = _unpad_data_cached(st, q_ids)
    B, Lk, nh, hd = k.shape
    Lq, nq = q.shape[1], q.shape[2]
    if ident_k and ident_q:
        _S["ctr"]["unpad_identity"] += 1
        return (q.reshape(B * Lq, nq, hd), k.reshape(B * Lk, nh, hd), v.reshape(B * Lk, nh, hd), idx_q, (cu_q, cu_k), (mx_q, mx_k))
    k2 = U.index_first_axis(k.reshape(B * Lk, nh, hd), idx_k)
    v2 = U.index_first_axis(v.reshape(B * Lk, nh, hd), idx_k)
    q2 = U.index_first_axis(q.reshape(B * Lq, nq, hd), idx_q)
    return q2, k2, v2, idx_q, (cu_q, cu_k), (mx_q, mx_k)


def _pad_input_once(h, indices, batch, seqlen):
    from E1.model import flash_attention_utils as U
    if h.shape[0] == batch * seqlen and _S["fwd"] is not None and _S["cfg"] is not None and _S["cfg"].unpad_once:
        _S["ctr"]["pad_identity"] += 1
        return h.reshape(batch, seqlen, *h.shape[1:])
    return U.pad_input(h, indices, batch, seqlen)


def _block_mask(sequence_ids: torch.Tensor):
    """modeling.create_block_causal_mask_optimized replacement. Dense mode: None under flex_block_mask="none", the shape-cached
    all-true mask under "cheap", the stock call under "stock"; "cached" and every non-dense (fallback) forward: the stock mask
    cached per content."""
    from E1.model import flex_attention as FX
    st, cfg = _S["fwd"], _S["cfg"]
    if st is None or cfg is None:
        return FX.create_block_causal_mask_optimized(sequence_ids)
    if st.dense and cfg.global_ != "stock" and cfg.flex_block_mask == "none":
        _S["ctr"]["mask_none"] += 1
        return None
    if st.dense and cfg.global_ == "flex" and cfg.flex_block_mask == "cheap":
        _S["ctr"]["mask_cheap"] += 1
        bm = cheap_block_mask(sequence_ids.shape[0], sequence_ids.shape[1], sequence_ids.device)
        st.block_mask = bm
        return bm
    if cfg.flex_block_mask == "stock" and st.dense:
        _S["ctr"]["mask_stock"] += 1
        bm = FX.create_block_causal_mask_optimized(sequence_ids)
        st.block_mask = bm
        return bm
    key = (tuple(sequence_ids.shape), hashlib.sha256(sequence_ids.cpu().numpy().tobytes()).hexdigest())
    bm = _S["mask_cache"].get(key)
    if bm is None:
        _S["mask_cache"].clear()
        bm = FX.create_block_causal_mask_optimized(sequence_ids)
        _S["mask_cache"][key] = bm
        _S["ctr"]["mask_cached_miss"] += 1
    else:
        _S["ctr"]["mask_cached_hit"] += 1
    st.block_mask = bm
    st.mask_key = key
    return bm


def _routable(self, query_states, key_states, sequence_ids) -> bool:
    st = _S["fwd"]
    return (st is not None and st.dense and query_states.shape[1] == key_states.shape[1]
            and tuple(sequence_ids.shape) == tuple(query_states.shape[:2]))


def _flash_attn_routed(self, query_states, key_states, val_states, sequence_ids, attention_args=None, output_attentions=False):
    cfg = _S["cfg"]
    route = cfg.within if cfg is not None else "stock"
    from E1.model.attention import AttentionLayerType
    if route != "stock" and self.layer_type == AttentionLayerType.WITHIN_SEQ and _routable(self, query_states, key_states, sequence_ids):
        assert not output_attentions
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        out = run_kernel(route, query_states, key_states, val_states, block_mask=None, cfg=cfg)
        _S["ctr"][f"within:{route}"] += 1
        return out.reshape(bsz, q_len, self.hidden_size).contiguous(), None
    _S["ctr"]["within:stock" if self.layer_type == AttentionLayerType.WITHIN_SEQ else "global_prefill:stock"] += 1
    return _S["orig"]["_flash_attn"](self, query_states, key_states, val_states, sequence_ids, attention_args, output_attentions)


def _flex_attn_routed(self, query_states, key_states, val_states, sequence_ids, attention_args=None, output_attentions=False):
    cfg = _S["cfg"]
    route = cfg.global_ if cfg is not None else "stock"
    if route != "stock" and _routable(self, query_states, key_states, sequence_ids):
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        fa = attention_args.get("flex_attention_args") if attention_args else None
        bm = fa.get("block_mask") if fa else None
        if route == "flex" and cfg.flex_block_mask == "none":
            bm = None
        out = run_kernel(route, query_states, key_states, val_states, block_mask=bm, cfg=cfg)
        _S["ctr"][f"global:{route}"] += 1
        return out.reshape(bsz, q_len, self.hidden_size).contiguous(), None
    _S["ctr"]["global:stock"] += 1
    return _S["orig"]["_flex_attn"](self, query_states, key_states, val_states, sequence_ids, attention_args, output_attentions)


# -------------------------------------------------------------------------------------------------------- apply / undo
def layer_counts(model) -> dict:
    cfg = model.config
    n = int(cfg.num_hidden_layers)
    g = int(cfg.global_attention_every_n_layers)
    n_global = sum(1 for i in range(n) if g > 0 and (i + 1) % g == 0)
    return {"layers": n, "global": n_global, "within": n - n_global}


def expected_per_forward(model, cfg: Config) -> dict:
    """Counter deltas ONE dense-mode forward must produce under ``cfg``."""
    lc = layer_counts(model)
    exp = {"forwards": 1, "forwards_dense": 1, f"within:{cfg.within}": lc["within"], f"global:{cfg.global_}": lc["global"]}
    if cfg.global_ != "stock" and cfg.flex_block_mask == "none":
        exp["mask_none"] = 1
    if cfg.global_ == "flex" and cfg.flex_block_mask == "cheap":
        exp["mask_cheap"] = 1
    return exp


def counters() -> dict:
    return {k: int(v) for k, v in _S["ctr"].items()}


def reset_counters():
    _S["ctr"].clear()


def counters_delta(prev: dict) -> dict:
    now = counters()
    return {k: now.get(k, 0) - prev.get(k, 0) for k in set(now) | set(prev)}


def inner_model(model):
    m = getattr(model, "model", None)
    return m if m is not None and hasattr(m, "layers") else model


def apply(model, cfg: Config) -> dict:
    import E1.modeling as M
    from E1.model import attention as A
    from E1.model import flash_attention as FA
    if _S["cfg"] is not None:
        raise RuntimeError("attention adapter already applied in this process (unapply first)")
    if not _S["orig"]:
        _S["orig"] = {"_flash_attn": A.Attention.__dict__["_flash_attn"], "_flex_attn": A.Attention.__dict__["_flex_attn"],
                      "_unpad_input": FA._unpad_input, "pad_input": FA.pad_input,
                      "create_block_causal_mask_optimized": M.create_block_causal_mask_optimized}
    im = inner_model(model)
    _S["sig"] = inspect.signature(im.forward)
    _S["hook"] = im.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    _S["model"] = im
    _S["cfg"] = cfg
    _S["fwd"] = None
    A.Attention._flash_attn = _flash_attn_routed
    A.Attention._flex_attn = _flex_attn_routed
    FA._unpad_input = _unpad_input_once
    FA.pad_input = _pad_input_once
    M.create_block_causal_mask_optimized = _block_mask
    if cfg.global_ == "flex":
        _flex_fn(cfg)
    return {"config": cfg.as_dict(), "layer_counts": layer_counts(model), "expected_per_forward": expected_per_forward(model, cfg)}


def unapply() -> None:
    import E1.modeling as M
    from E1.model import attention as A
    from E1.model import flash_attention as FA
    if _S["cfg"] is None:
        return
    o = _S["orig"]
    A.Attention._flash_attn = o["_flash_attn"]
    A.Attention._flex_attn = o["_flex_attn"]
    FA._unpad_input = o["_unpad_input"]
    FA.pad_input = o["pad_input"]
    M.create_block_causal_mask_optimized = o["create_block_causal_mask_optimized"]
    if _S["hook"] is not None:
        _S["hook"].remove()
    _S["mask_cache"].clear()
    _S.update(cfg=None, fwd=None, model=None, hook=None)


def is_pristine() -> dict:
    """True per patch point iff the stock attribute is in place (all True = nothing of the adapter is installed)."""
    import E1.modeling as M
    from E1.model import attention as A
    from E1.model import flash_attention as FA
    from E1.model import flash_attention_utils as U
    from E1.model import flex_attention as FX
    return {"_flash_attn": A.Attention.__dict__["_flash_attn"] is not _flash_attn_routed,
            "_flex_attn": A.Attention.__dict__["_flex_attn"] is not _flex_attn_routed,
            "_unpad_input": FA._unpad_input is U._unpad_input, "pad_input": FA.pad_input is U.pad_input,
            "create_block_causal_mask_optimized": M.create_block_causal_mask_optimized is FX.create_block_causal_mask_optimized}
