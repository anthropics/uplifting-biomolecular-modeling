"""The `castcache` lever (exact class): the engine's `Linear` / `LayerNorm` primitives (core/model/primitives/{linear,normalization}.py) cast their
fp32 weight and bias to the activation dtype on EVERY bf16 call — `self.weight.to(dtype=d)`, `self.bias.to(dtype=d)`: two copy kernels per call,
~21 launches per pairformer block on a bf16 line, ~4 k of the ~17 k trunk launches per item at 400 tokens, ~10 k on the fp32 lines' bf16 regions.
The weights are constant at inference, so the bf16 copies are memoised per module (in the module's own `__dict__`, key = the fp32 tensors'
(data_ptr, version, shape, dtype, device): reloaded / moved / in-place-edited weights refresh the memo) and handed to the SAME
`nn.functional.linear` / `nn.functional.layer_norm` call under the same autocast-disabled region → the same kernel on the same operands: bitwise
identical. Only the bf16 branch of the two statements is served (Linear: `precision is None`, bf16 input, DeepSpeed comm not initialised;
LayerNorm: bf16 input, DeepSpeed comm not initialised); every other call runs the engine's own statement (not a fallback: the same code path as
stock). Each served branch is checked against the engine's source once at install (the statement this lever restates must be the one in the
tree; a primitive whose statement differs runs as it is, named on the census; neither matching = refused by name, nothing patched). Memory: one bf16 copy of every weight that runs in bf16 (~336 MiB for the OF3 code family's AF3-size
trunk on the fast line; census `mib=`).

Switch (the kit adapter's, `configure(ENV=…)`): <KIT>_CASTCACHE=1. Evidence `<PREFIX> installed …` and one exit line
`<PREFIX> LEVER name=castcache state=on modules=<n> mib=<x> refresh=<n> served=<n>`. Engines: the OF3 code family; the kits' `cells/castcache.py`
are the adapters (M_LINEAR / M_NORM).
"""
from __future__ import annotations

import atexit
import inspect
import os
import sys
from typing import Any, Dict, Optional

PREFIX = "[opt_core/of3_trunk.castcache]"
ENV: Optional[str] = None                                # <KIT>_CASTCACHE=1
VALUES = ("1",)
M_LINEAR: Optional[str] = None                           # ….core.model.primitives.linear (class Linear)
M_NORM: Optional[str] = None                             # ….core.model.primitives.normalization (class LayerNorm)
CONFIGURABLE = ("PREFIX", "ENV", "M_LINEAR", "M_NORM")
ATTR = "_of3opt_cast16"                                  # the per-module memo: (key, weight16, bias16, nbytes)

# the engine statements this lever restates (verified at install; a tree whose primitives differ is refused by name)
LINEAR_SOURCE = ("if d is torch.bfloat16 and not deepspeed_is_initialized:", "bias = self.bias.to(dtype=d) if self.bias is not None else None",
                 "return nn.functional.linear(input, self.weight.to(dtype=d), bias)", "if self.precision is not None:")
NORM_SOURCE = ("if d is torch.bfloat16 and not deepspeed_is_initialized:", "weight = self.weight.to(dtype=d) if self.weight is not None else None",
               "bias = self.bias.to(dtype=d) if self.bias is not None else None", "normalized_shape=self.c_in,", "eps=self.eps,")


def configure(**kw) -> None:
    """The kit adapter's binding: reassigns this module's engine words before install; unknown names raise."""
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "modules": 0, "bytes": 0, "refresh": 0, "served": 0, "patched": [], "skipped": []}


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


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
    """identity + content version of a tensor: (data_ptr, version, shape, dtype, device); inference tensors carry no version counter (None) —
    the memo then keys on storage identity and holds the fp32 tensor's module, whose parameter it is, so the storage is not recycled under it"""
    try:
        v = None if t.is_inference() else t._version
    except Exception as e:  # noqa: BLE001
        from ..oom import is_oom
        if is_oom(e):
            raise
        v = None
    return (t.data_ptr(), v, tuple(t.shape), t.dtype, str(t.device))


def cached16(m, dtype):
    """(weight16, bias16) of module m in `dtype`, memoised on the fp32 tensors' identity + version (the same `.to(dtype=…)` copy stock makes per call)."""
    w, b = getattr(m, "weight", None), getattr(m, "bias", None)
    key = (dtype, None if w is None else _tkey(w), None if b is None else _tkey(b))
    ent = m.__dict__.get(ATTR)
    if ent is not None and ent[0] == key:
        return ent[1], ent[2]
    w16 = None if w is None else w.to(dtype=dtype)
    b16 = None if b is None else b.to(dtype=dtype)
    nb = sum(t.numel() * t.element_size() for t in (w16, b16) if t is not None)
    if ent is not None:
        STATE["bytes"] -= ent[3]; STATE["refresh"] += 1
    else:
        STATE["modules"] += 1
    STATE["bytes"] += nb
    m.__dict__[ATTR] = (key, w16, b16, nb)
    return w16, b16


def _verify(fn, want, what) -> Optional[str]:
    try:
        src = inspect.getsource(fn)
    except Exception as e:  # noqa: BLE001
        from ..oom import is_oom
        if is_oom(e):
            raise
        return f"{what}: source unreadable ({type(e).__name__})"
    missing = [w for w in want if w not in src]
    return f"{what} differs from the statement this lever serves (missing {missing})" if missing else None


def _make_linear(orig, L):
    import torch
    import torch.nn.functional as F

    def forward(self, input):
        if self.precision is None and input.dtype is torch.bfloat16 and not (L.deepspeed_is_installed and L.deepspeed.comm.comm.is_initialized()):
            w16, b16 = cached16(self, torch.bfloat16)
            STATE["served"] += 1
            with torch.amp.autocast("cuda", enabled=False):
                return F.linear(input, w16, b16)
        return orig(self, input)                           # fp32 / explicit-precision / DeepSpeed-initialised calls: the engine's own statement
    forward.__wrapped__ = orig; forward._of3opt_castcache = True
    return forward


def _make_layernorm(orig, N):
    import torch
    import torch.nn.functional as F

    def forward(self, x):
        if x.dtype is torch.bfloat16 and not (N.deepspeed_is_installed and N.deepspeed.comm.comm.is_initialized()):
            w16, b16 = cached16(self, torch.bfloat16)
            STATE["served"] += 1
            with torch.amp.autocast("cuda", enabled=False):
                return F.layer_norm(input=x, normalized_shape=self.c_in, weight=w16, bias=b16, eps=self.eps)
        return orig(self, x)
    forward.__wrapped__ = orig; forward._of3opt_castcache = True
    return forward


def census_line() -> str:
    skipped = ",".join(STATE["skipped"]) or "none"
    return (f"{PREFIX} LEVER name=castcache state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" primitives={','.join(p.split('.')[0].lower() for p in STATE['patched']) or 'none'} skipped={skipped}"
            f" modules={STATE['modules']} mib={STATE['bytes'] / 2 ** 20:.1f} refresh={STATE['refresh']} served={STATE['served']}")


def install(environ=None) -> dict:
    """Check the engine's two statements against its source and rebind `Linear.forward` / `LayerNorm.forward` — each on its own: a primitive whose statement is the
    one this lever restates is served, a primitive whose statement differs (an engine whose LayerNorm upcasts to fp32 instead of casting its
    weight to bf16 has no per-call cast to memoise) runs as it is, named (`skipped=layernorm:<why>` on the census). Idempotent; a tree where
    neither statement matches is refused by name (state=refused, nothing patched)."""
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not (M_LINEAR and M_NORM):
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_LINEAR=, M_NORM=) before install")
    import importlib
    L = importlib.import_module(M_LINEAR); N = importlib.import_module(M_NORM)
    lin, ln = L.Linear.forward, N.LayerNorm.forward
    why_lin, why_ln = _verify(lin, LINEAR_SOURCE, "Linear.forward"), _verify(ln, NORM_SOURCE, "LayerNorm.forward")
    if why_lin and why_ln:
        STATE.update(installed=True, state="refused", reason="statements_differ")
        _log(f"REFUSED: {why_lin}; {why_ln} — the engine's primitives run as they are")
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
        return STATE
    if why_lin:
        STATE["skipped"].append("linear:statement_differs"); _log(f"Linear NOT served: {why_lin} — it runs as it is")
    elif not getattr(lin, "_of3opt_castcache", False):
        L.Linear.forward = _make_linear(lin, L); STATE["patched"].append("Linear.forward")
    if why_ln:
        STATE["skipped"].append("layernorm:statement_differs"); _log(f"LayerNorm NOT served: {why_ln} — it runs as it is")
    elif not getattr(ln, "_of3opt_castcache", False):
        N.LayerNorm.forward = _make_layernorm(ln, N); STATE["patched"].append("LayerNorm.forward")
    STATE.update(installed=True, state="on")
    _log("installed: %s serve their bf16 branch from memoised bf16 weight copies (same F.linear / F.layer_norm call; memo keyed on the fp32 tensors' "
         "storage + version)" % " / ".join(STATE["patched"]))
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
