"""guard.py — the ew1 index-bound guard: a silent non-stock output above a size bound is refused or routed to the stock, COUNTED, and
named on the guard's report line (`line()`).

THE DEFECT it guards: ew1's Triton kernels index with INT32 offsets — engines/e1/kits/ew1/kernels.py: the elementwise kernels
`offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)` (_silu_mul_kernel, _silu_kernel, _gelu_kernel: max offset = n - 1 with
n = a.numel() = tokens x intermediate_size), the row kernels `X += row * stride_x_row` (_add_rmsnorm_kernel: max = rows x N with
N = hidden_size) and `OUT + row * N + cols` (_embed_add_kernel), the rope kernel `offs = r[:, None] * (2 * HALF) + i` with
r < R = B * L * H (_rope_rows: max = tokens x num_heads x head_dim = tokens x hidden_size). Triton's int32 arithmetic wraps past 2^31 - 1:
above the bound the kernels read and write the WRONG elements silently.
The tightest bound is the GLU intermediate (intermediate_size > hidden_size in every E1 config).

THE GUARD (exact by construction): ew1 binds its patched methods on the module INSTANCES (ew1/patches.py `_bind`: obj.__dict__[name]
= MethodType(fn, obj)); the class-level methods stay the STOCK's. The guard wraps every ew1-bound method: the OUTER one (E1Model.forward, the
whole forward) computes tokens = input_ids.numel() and, if tokens x max(intermediate_size, hidden_size) >= 2^31, sets a process flag and runs
the CLASS-level (stock) E1Model.forward; the INNER wrapped sites (DecoderLayer.forward, GLUMLP.forward, Attention.prepare_qkv) run their
class-level (stock) method whenever the flag is set — the whole forward is then the stock's own code path (the stock's torch ops, int64-safe;
P1's precast bf16 weights are the same bytes autocast would produce). Below the bound nothing changes: the ew1-bound method runs. Every routed
call is COUNTED (CTR) and the bound per size is reported once by `line()`: 'ew1: stock path above 2^31 elements of the GLU intermediate:
tokens >= <bound>'. Never a wrong bit, never a crash on this wrapped call. With ew1_i64 composed (kits/eager does)
the bound is the int64 one (never met) and calls above the int32 bound are only COUNTED. Above that scale the stock's own hub RMSNorm Triton kernel indexes with int32 the same way:
a stock ceiling far beyond any batch E1Predictor forms (max_batch_tokens), not something this guard fixes.
"""
from __future__ import annotations

import types
from collections import Counter

INT32_BOUND = 2 ** 31                       # numel of the GLU intermediate >= 2^31 -> the stock path
INT64_BOUND = 2 ** 63                       # with ew1_i64 composed: the index type's own bound (never met on any E1 input); the int32 bound is then COUNTED, not routed
CTR = Counter()
_G = {"installed": False, "stock": False, "bound_tokens": None, "bound": None, "int32_bound_tokens": None, "sizes": None, "wrapped": [], "model_id": None}


class GuardRefused(RuntimeError):
    pass


def bound_tokens(hidden_size: int, intermediate_size: int, bound: int = INT32_BOUND) -> int:
    """The smallest token count at which any ew1 kernel's int32 index space overflows: ceil(bound / max(d_ff, d_model)) — the GLU
    intermediate (tokens x d_ff elements) is the tightest site; the row / rope kernels (tokens x d_model) follow at a larger count."""
    width = max(int(hidden_size), int(intermediate_size))
    if width <= 0:
        raise GuardRefused(f"guard: non-positive width {width}")
    return -(-bound // width)


def route_to_stock(tokens: int, hidden_size: int, intermediate_size: int, bound: int = INT32_BOUND) -> bool:
    """True iff the call's GLU intermediate (or any row/rope site) reaches the int32 bound: tokens x max(d_ff, d_model) >= bound."""
    return int(tokens) * max(int(hidden_size), int(intermediate_size)) >= bound


def _wrap_outer(obj, name, bound, sizes):
    ew1_bound = obj.__dict__[name]                        # the ew1-bound method (MethodType)
    stock_fn = getattr(type(obj), name)                   # the class-level function = the stock's

    def guarded(self, input_ids, *args, **kwargs):
        tokens = int(input_ids.numel()) if hasattr(input_ids, "numel") else int(input_ids.shape[0] * input_ids.shape[1])
        CTR["forwards"] += 1
        if tokens > CTR.get("max_tokens_seen", 0):
            CTR["max_tokens_seen"] = tokens
        if route_to_stock(tokens, sizes["hidden_size"], sizes["intermediate_size"], INT32_BOUND):
            CTR["above_int32_bound_forwards"] += 1        # the belt: counted whatever the routing bound
        if route_to_stock(tokens, sizes["hidden_size"], sizes["intermediate_size"], bound):
            CTR["stock_path_forwards"] += 1
            CTR["stock_path_tokens_last"] = tokens
            _G["stock"] = True
            try:
                return stock_fn(self, input_ids, *args, **kwargs)
            finally:
                _G["stock"] = False
        return ew1_bound(input_ids, *args, **kwargs)
    guarded._guard_outer = True
    guarded._guard_inner_fn = ew1_bound
    obj.__dict__[name] = types.MethodType(guarded, obj)
    return (obj, name, ew1_bound)


def _wrap_inner(obj, name):
    ew1_bound = obj.__dict__[name]
    stock_fn = getattr(type(obj), name)

    def guarded(self, *args, **kwargs):
        if _G["stock"]:
            CTR["stock_path_sites"] += 1
            return stock_fn(self, *args, **kwargs)
        return ew1_bound(*args, **kwargs)
    guarded._guard_inner = True
    obj.__dict__[name] = types.MethodType(guarded, obj)
    return (obj, name, ew1_bound)


def install(model, ew1_state: dict, hidden_size: int, intermediate_size: int, bound: int = INT32_BOUND) -> dict:
    """Wrap every ew1-bound site of `model` (ew1_state['originals'] = {(id(obj), name): (obj, name, prior)} — ew1's own record of what it
    bound). The OUTER site is model.model.forward (ew1's E1Model.forward); every other bound site is an inner one."""
    if _G["installed"]:
        raise GuardRefused("guard: already installed (uninstall first)")
    sizes = {"hidden_size": int(hidden_size), "intermediate_size": int(intermediate_size)}
    bound_tok = bound_tokens(hidden_size, intermediate_size, bound)
    int32_tok = bound_tokens(hidden_size, intermediate_size, INT32_BOUND)
    inner_model = getattr(model, "model", None)
    if inner_model is None or "forward" not in inner_model.__dict__:
        raise GuardRefused("guard: ew1's E1Model.forward binding is not on model.model (the guard needs the outer site)")
    wrapped = []
    outer_key = (id(inner_model), "forward")
    for key, (obj, name, _prior) in list(ew1_state["originals"].items()):
        if name not in obj.__dict__:
            continue
        if key == outer_key:
            wrapped.append(_wrap_outer(obj, name, int(bound), sizes))
        else:
            wrapped.append(_wrap_inner(obj, name))
    if not any(w[0] is inner_model and w[1] == "forward" for w in wrapped):
        raise GuardRefused("guard: the outer site (model.model.forward) is not among ew1's bindings")
    _G.update({"installed": True, "bound_tokens": bound_tok, "bound": int(bound), "int32_bound_tokens": int32_tok, "sizes": sizes, "wrapped": wrapped, "model_id": id(model)})
    CTR.clear()
    return {"installed": True, "bound": int(bound), "bound_tokens": bound_tok, "int32_bound_tokens": int32_tok, "sites_wrapped": len(wrapped), "outer": "E1Model.forward", **sizes,
            "line": f"ew1: stock path above 2^31 elements of the GLU intermediate: tokens >= {int32_tok} (d_ff {sizes['intermediate_size']}, d_model {sizes['hidden_size']})"}


def uninstall() -> None:
    for (obj, name, ew1_bound) in _G["wrapped"]:
        if name in obj.__dict__:
            obj.__dict__[name] = ew1_bound            # ew1's own binding back (ew1.unapply restores the stock below it)
    _G.update({"installed": False, "stock": False, "bound_tokens": None, "bound": None, "int32_bound_tokens": None, "sizes": None, "wrapped": [], "model_id": None})


def counters() -> dict:
    return {f"guard:{k}": int(v) for k, v in CTR.items()}


def line() -> str | None:
    if not _G["installed"]:
        return None
    return (f"ew1: stock path above {'2^31' if _G['bound'] == INT32_BOUND else '2^63 (ew1_i64 composed)'} elements of the GLU intermediate: tokens >= {_G['bound_tokens']}; "
            f"routed {CTR.get('stock_path_forwards', 0)} of {CTR.get('forwards', 0)} forwards; above the int32 bound {CTR.get('above_int32_bound_forwards', 0)}")
