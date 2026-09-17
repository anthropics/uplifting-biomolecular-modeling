"""The `apb_hoist` lever (exact class): `AttentionPairBias._prep_bias` (core/model/layers/attention_pair_bias.py) rebuilds the key-mask bias
`(self.inf * (mask - 1))[..., None, None, :]` on EVERY call — 48 pairformer blocks per recycle pass and the confidence head's blocks per call
share one mask, so the term is constant for the stack call. This lever memoises it (one slot per module CLASS, keyed on the mask tensor's
(data_ptr, version, shape, dtype, device) + the batch dims + `inf`; the memo holds the mask so its storage cannot be recycled under the key;
under torch.inference_mode the mask is an inference tensor without a version counter, so the guard against an IN-PLACE edit of a live mask
rests on the engine never editing a mask in place between the blocks of a stack call — true for the stock trees this lever is bound to)
and computes the pair-bias term exactly as the engine does (`layer_norm_z` -> `linear_z` -> `permute_final_dims`). Same kernels on the same
operands: bitwise. A call without a mask, or a call under CUDA-graph capture (a memo hit would bake a foreign tensor address into the graph),
runs the engine's own statement BY NAME (counted: `fallback:no_mask|capturing`). The served body is verified against the engine's source once
at install; a tree whose `_prep_bias` differs is refused by name and left unpatched.

Composition: a kit lever that REPLACES the pair-bias producer for some instances (e.g. of3_sampler.apb_trunk, installed after this one and
outermost) serves those calls itself and passes the rest to this function — this lever then serves what reaches it, counted.

Switch (the kit adapter's, `configure(ENV=…)`): <KIT>_APB_HOIST=1. Exit line `<PREFIX> LEVER name=apb_hoist state=on hit=<n> fill=<n> fallback=<..>`.
"""
from __future__ import annotations

import atexit
import inspect
import os
import sys
from typing import Any, Dict, Optional

PREFIX = "[opt_core/of3_trunk.apb_hoist]"
ENV: Optional[str] = None                                # <KIT>_APB_HOIST=1
VALUES = ("1",)
M_APB: Optional[str] = None                              # ….core.model.layers.attention_pair_bias (class AttentionPairBias, permute_final_dims)
CONFIGURABLE = ("PREFIX", "ENV", "M_APB")
SOURCE = ("mask.expand((*batch_dims, -1))", "(self.inf * (mask - 1))[..., None, None, :]", "self.layer_norm_z(z)", "self.linear_z(z)",
          "permute_final_dims(z, [2, 0, 1])")


def configure(**kw) -> None:
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "hit": 0, "fill": 0, "fallback": {}, "patched": []}
_MEMO = {"key": None, "mask": None, "bias": None}


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if not ENV:
        return False
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def serving() -> bool:
    return STATE["state"] == "on"


def _tkey(t) -> tuple:
    try:
        v = None if t.is_inference() else t._version
    except Exception as e:  # noqa: BLE001
        from ..oom import is_oom
        if is_oom(e):
            raise
        v = None
    return (t.data_ptr(), v, tuple(t.shape), t.dtype, str(t.device))


def release() -> None:
    """Drop the memoised mask bias (an item boundary release; the next call refills)."""
    _MEMO.update(key=None, mask=None, bias=None)


def _make_prep_bias(orig, A):
    import torch
    permute_final_dims = A.permute_final_dims

    def _prep_bias(self, a, z, mask=None):
        if mask is None or (mask.is_cuda and torch.cuda.is_current_stream_capturing()):
            _count(STATE["fallback"], "no_mask" if mask is None else "capturing")
            return orig(self, a=a, z=z, mask=mask)          # by keyword, as the engine's caller passes them
        batch_dims = a.shape[:-2]
        key = (_tkey(mask), tuple(batch_dims), float(self.inf))
        if _MEMO["key"] == key and _MEMO["mask"] is mask:
            mask_bias = _MEMO["bias"]; STATE["hit"] += 1
        else:
            m = mask.expand((*batch_dims, -1))
            mask_bias = (self.inf * (m - 1))[..., None, None, :]
            _MEMO["key"], _MEMO["mask"], _MEMO["bias"] = key, mask, mask_bias; STATE["fill"] += 1
        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        z = permute_final_dims(z, [2, 0, 1])
        return [mask_bias, z]
    _prep_bias.__wrapped__ = orig; _prep_bias._of3opt_apb_hoist = True
    return _prep_bias


def census_line() -> str:
    return (f"{PREFIX} LEVER name=apb_hoist state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" hit={STATE['hit']} fill={STATE['fill']} fallback={','.join('%s:%d' % kv for kv in sorted(STATE['fallback'].items())) or 'none'}")


def install(environ=None) -> dict:
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not M_APB:
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_APB=) before install")
    import importlib
    A = importlib.import_module(M_APB)
    orig = A.AttentionPairBias._prep_bias
    try:
        src = inspect.getsource(orig)
        missing = [s for s in SOURCE if s not in src]
    except Exception as e:  # noqa: BLE001
        from ..oom import is_oom
        if is_oom(e):
            raise
        missing = [f"source unreadable: {type(e).__name__}"]
    if missing:
        STATE.update(installed=True, state="refused", reason="prep_bias_source_differs")
        _log(f"REFUSED: AttentionPairBias._prep_bias differs from the statement this lever serves (missing {missing}) — the engine's runs as it is")
    else:
        if not getattr(orig, "_of3opt_apb_hoist", False):
            A.AttentionPairBias._prep_bias = _make_prep_bias(orig, A); STATE["patched"].append("AttentionPairBias._prep_bias")
        STATE.update(installed=True, state="on")
        _log("installed: AttentionPairBias._prep_bias serves the key-mask bias from a memo keyed on the mask's storage + version (pair bias computed as the engine does)")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
