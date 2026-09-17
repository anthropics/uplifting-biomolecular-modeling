"""anylen.py — the any-length flex route of the attention lever (the GLOBAL layers' flex_attention call).

The attn adapter compiles ONE dynamic flex_attention callable per (kernel options, dynamic) and looks it up on every global-layer
call (attn/adapter.py `_flex_fn`: `_S['flex_fns'][(kernel_options, dynamic)]`). Left alone, a second distinct sequence length at batch 1
re-enters torch's flex-DECODING lowering (torch/_inductor/kernel/flex_attention.py `_use_flex_decoding`: B=1 static, seq_len_q < 128 by
hint), whose `guard_leq(seq_len_q * G, kernel_options["BLOCK_M"])` asserts against A3's pinned BLOCK_M — a crash on short inputs. This
module installs, in that same slot, a callable that serves EVERY length:

  hybrid (the form in force):  >= FLEX_SHORT_TOKENS query tokens -> the kit's dynamic compile with A3's tiling + FORCE_USE_FLEX_ATTENTION=True
                               (the decoding branch unreachable: one kernel for every length, bitwise with the stock's main flex kernel —
                               equal BLOCK_N = the same per-row reduction order);
                               <  FLEX_SHORT_TOKENS -> the stock's OWN compiled object (E1.model.flex_attention.flex_attention,
                               torch.compile(flex_attention, dynamic=True)) called as the stock calls it — the stock's own short-length
                               decision, bitwise trivially.
The adapter's cheap block mask (A2) and its counters are untouched; A3's tiling rides in the kernel options of the long branch (with A3 off —
classes a100 / b200 / l40s — the options are torch's defaults, as the stock's).
"""
from __future__ import annotations

import collections

FLEX_ROUTE_FORMS = ("hybrid", "force_flex", "stock_object")
FLEX_SHORT_TOKENS = 128            # _use_flex_decoding's own threshold (torch/nn/attention/flex_attention.py: Lt(seq_len_q, 128))
CTR = collections.Counter()
_R = {"flex": None}


class AnylenRefused(RuntimeError):
    """A precondition failed: the process fails closed (never a silent fallback to another numerics path)."""


def _stock_flex_object():
    from E1.model import flex_attention as FX
    fn = FX.flex_attention
    if fn is None or not hasattr(fn, "_torchdynamo_orig_callable"):
        raise AnylenRefused("E1.model.flex_attention.flex_attention is not the stock's compiled object in this process "
                            "(upstream compiles it only when CUDA is available)")
    return fn


def flex_route_fns(form: str, kernel_options: dict | None):
    """The callables of a form, keyed as the adapter's run_kernel calls them: fn(q, k, v, block_mask=) on (B, nh, L, hd) contiguous
    q/k/v -> (B, nh, L, hd)."""
    import torch
    if form not in FLEX_ROUTE_FORMS:
        raise AnylenRefused(f"flex route form {form!r} not in {FLEX_ROUTE_FORMS}")
    fns = {}
    if form in ("stock_object", "hybrid"):
        stock = _stock_flex_object()

        def flex_stock_object(q, k, v, block_mask=None, _f=stock):
            CTR["flex:stock_object"] += 1
            return _f(q, k, v, block_mask=block_mask, score_mod=None, enable_gqa=q.shape[1] != k.shape[1])     # as E1.model.flex_attention.flex_attention_func calls it
        fns["stock_object"] = flex_stock_object
    if form in ("force_flex", "hybrid"):
        from torch.nn.attention.flex_attention import flex_attention as _eager
        KO = dict(kernel_options or {})
        KO["FORCE_USE_FLEX_ATTENTION"] = True

        def flex_force(q, k, v, block_mask=None, _KO=KO):
            return _eager(q, k, v, block_mask=block_mask, kernel_options=_KO)
        compiled = torch.compile(flex_force, dynamic=True) if torch.cuda.is_available() else flex_force

        def flex_force_counted(q, k, v, block_mask=None, _c=compiled):
            CTR["flex:force_flex"] += 1
            return _c(q, k, v, block_mask=block_mask)
        fns["force_flex"] = flex_force_counted
        fns["_kernel_options"] = KO
    if form == "hybrid":
        short, long_ = fns["stock_object"], fns["force_flex"]

        def flex_hybrid(q, k, v, block_mask=None, _s=short, _l=long_):
            return _s(q, k, v, block_mask=block_mask) if q.shape[-2] < FLEX_SHORT_TOKENS else _l(q, k, v, block_mask=block_mask)
        fns["hybrid"] = flex_hybrid
    return fns


def install(form: str = "hybrid") -> dict:
    """Replace the adapter's compiled flex callable for the config IN FORCE by the form's callable. Requires kits/attn applied."""
    from ..attn import adapter as A
    cfg = A._S.get("cfg")
    if cfg is None:
        raise AnylenRefused("the attn adapter is not applied (kits/v1 first)")
    key = (cfg.flex_kernel_options, cfg.flex_dynamic)
    fns = flex_route_fns(form, cfg.kernel_options())
    prev = A._S["flex_fns"].get(key)
    A._S["flex_fns"][key] = fns[form]
    CTR.clear()
    rec = {"form": form, "key": [dict(key[0]), key[1]], "kernel_options_force": fns.get("_kernel_options"), "short_tokens": FLEX_SHORT_TOKENS, "replaced": prev is not None}
    _R["flex"] = {"key": key, "prev": prev, "rec": rec}
    return rec


def uninstall() -> None:
    f = _R.get("flex")
    if not f:
        return
    try:
        from ..attn import adapter as A
        if f["prev"] is not None:
            A._S["flex_fns"][f["key"]] = f["prev"]
        else:
            A._S["flex_fns"].pop(f["key"], None)
    except Exception:
        pass
    _R["flex"] = None


def in_force() -> bool:
    return _R.get("flex") is not None


def counters() -> dict:
    return dict(CTR)
