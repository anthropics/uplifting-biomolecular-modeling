"""ln_core — the pair-row LayerNorms of the fast / big lines served by the core's LayerNorm provider (``opt_core.kernels.ln``) by TIER word.

Upstream's ``FusedLayerNorm`` (``opendde/model/layer_norm/layer_norm.py:233-290``, bound by ``LAYERNORM_TYPE=fast_layernorm`` — the stock base
of this tree) runs the compiled extension ``fast_layer_norm_cuda_v2`` (fp32 Welford statistics, parameters cast to the input's dtype per call,
output in the input's dtype).  This lever wraps ``FusedLayerNorm.forward`` (the core's per-site patch at the module's import): a call of the
PAIR form — width ``C`` in ``SERVED_WIDTHS`` (c_z = c_s = 384 on the pin), weight AND bias present, a CUDA tensor, no autograd, NOT under
CUDA-graph capture — asks the provider's ``select()`` ONCE per call class (C, dtype, rows, alignment) for the row its measured cell table names
for this line's word (``ODDE_LN=fast`` on LSTAR2A, ``=big`` on the big lines) and serves it through ``opt_core.kernels.ln.layer_norm``
(Triton rows ``fastln`` / ``fastln:lp`` / ``ln_rows`` at pair rows: x1.4-1.65 the extension per call on H100, x1.2-1.5 on A100 at
N = 400..1200, C = 384).  Everything else is the extension's BY NAME, counted: a cell whose winner is a stock row
(``aten`` | ``fast_layernorm`` — the provider's decision at small rows), a width outside ``SERVED_WIDTHS`` (the MSA c=64 / atom c=128 / DiT
c=768 / c=16 forms: on this stack the tier's Triton row is x0.4-0.96 the extension at <= 160000 rows, so they stay on the
extension), a weight-only / offset-free module, a CPU tensor, autograd on, a call under capture or inside the sampler step graph's plan-for-capture
window under the big word (aside word stepgraph_window; the sampler step graph replays the
extension's current-stream build: lever lnstream), a provider refusal (its word), or an error of a served row (its type; that class then
stays on the extension for the process).  Tolerance class: the Triton rows are not bitwise the extension (max |d| 1-2 bf16 ulp = 2^-6..2^-5
in bf16, ~1.4e-6 in fp32 vs the extension at every measured cell; rel-RMS vs fp64 of the same order as the extension's); the exact line does
not carry the lever (no provider row is bitwise the extension on this stack: exact = the extension by name, lever lnstream).

Census: ``COUNTS`` (word, calls served per row / cell, stock_calls by reason, refusals, errors, classes) -> the LEVER line
(``report.lever_evidence``) and ``ran.COUNTERS`` (calls served); the provider records every select() in ``opt_core.cell_census`` itself.
Ablation: ``MODEL_OPT_LEVERS_OFF=ln_core`` (modes.LEVER_SWITCHES: the word absent = the wrapper is never installed).
"""
from __future__ import annotations

import os
import sys
import threading

TAG = "opendde-opt"
LEVER = "ln_core"
ENV = "ODDE_LN"                                          # the tier word: fast | big (| exact | a provider row name for a maintainer's A/B — opt_core.kernels.ln TIER_WORDS / ROW_NAMES)
TARGET = "opendde.model.layer_norm.layer_norm"           # upstream's module: the FusedLayerNorm class lives here
ATTR = "FusedLayerNorm.forward"
SERVED_WIDTHS = (384,)                                   # c_z = c_s on the pin (config/model_base.py:19-20): the pair-row / single-row form the cells favour a carried row for;
                                                         # other widths stay on the extension by name (form:c<C>) until the core's cells carry this stack for them
STATS = {"installed": False, "armed": False, "patch": None, "word": None, "calls": 0}
STEPGRAPH_MODULE = "opendde_opt.stepgraph"                               # the sampler step graph's plan-for-capture window, read through sys.modules (no import cycle)


WINDOW_WORDS = ("big",)                                # the tier word(s) under which this lever honours the step graph's plan-for-capture window: big only —
                                                         # under fast the captured step holds on the literal capture state at every measured size, so fast's eager steps keep the provider row


def _stepgraph_window() -> bool:
    """True inside stepgraph.planning() — an admitted sampler call runs its denoiser steps — under a word of WINDOW_WORDS (big; under
    fast / a row word the lever follows the literal capture state alone)."""
    if COUNTS.get("word") not in WINDOW_WORDS:
        return False
    sg = sys.modules.get(STEPGRAPH_MODULE)
    try:
        return bool(sg is not None and sg.planning())
    except Exception:                                                     # pragma: no cover
        return False


COUNTS: dict = {"active": False, "word": None, "calls": 0, "stock_calls": 0, "rows": {}, "cells": {}, "asides": {}, "refusals": {}, "errors": {},
                "classes": 0, "dtypes": {}, "stack": None}
_LOCK = threading.Lock()
_CACHE: dict = {}                                        # (C, dtype, rows, aligned) -> ("serve", selection, arm_word) | ("stock", reason)
_ST = {"PL": None, "cc": None, "stack": None, "has_triton": None, "stock_rows": ()}


class ActivationError(RuntimeError):
    pass


def word(environ=None) -> str | None:
    """The tier word from the environment (None = unset / empty / off: the lever inert)."""
    if environ is None:
        w = (os.environ.get("ODDE_LN") or "").strip()
    else:
        w = (environ.get("ODDE_LN") or "").strip()
    return None if w in ("", "off", "0") else w


def _bump(section: str, key: str, n: int = 1) -> None:
    d = COUNTS[section]
    d[key] = int(d.get(key) or 0) + n


def _provider():
    """opt_core.kernels.ln imported once (the face; its rows import torch / triton lazily when selected)."""
    if _ST["PL"] is None:
        from opt_core.kernels import ln as PL
        _ST["PL"] = PL
        _ST["stock_rows"] = tuple(getattr(PL, "STOCK_ROWS", ()))
        try:
            import triton  # noqa: F401
            _ST["has_triton"] = True
        except ImportError:
            _ST["has_triton"] = False
    return _ST["PL"]


def _decide(x, C: int, rows: int, aligned: bool):
    """The provider's decision for this call class, cached: ('serve', Selection, arm word) or ('stock', reason word)."""
    key = (C, x.dtype, rows, aligned)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    import torch
    PL = _provider()
    if _ST["cc"] is None:
        _ST["cc"] = tuple(torch.cuda.get_device_capability(x.device))
        try:
            _ST["stack"] = PL.stack_word(x.device)
        except Exception:  # noqa: BLE001
            _ST["stack"] = None
        COUNTS["stack"] = _ST["stack"]
    n_tok = int(round(rows ** 0.5))
    cw = PL.cell_for_rows(rows, n_tok, C)
    if cw is None:                                                        # no cell family lists this width: the extension by name (no select(), nothing for the census to alert on)
        dec = ("stock", "no_cell_family:c%d" % C)
    else:
        try:
            sel = PL.select(_ST["cc"], x.dtype, cw, n_tok, word=STATS["word"], widen=False, out=None, C=C, has_triton=_ST["has_triton"],
                            aligned=aligned, stack=_ST["stack"], rows=rows, capture=False)
        except PL.Refusal as r:
            _bump("refusals", "%s:%s" % (getattr(r, "row", None) or STATS["word"], getattr(r, "kind", "refused")))
            dec = ("stock", "refused:%s" % (getattr(r, "kind", "refused"),))
        else:
            arm = PL.arm_word(sel.row, sel.variant)
            if sel.row in _ST["stock_rows"]:                              # the cell's winner for this word is a stock row: the extension serves (the provider's decision, counted)
                dec = ("stock", "stock_cell:%s" % arm)
            else:
                dec = ("serve", sel, arm)
                _bump("cells", str(sel.cell), 0)
    _CACHE[key] = dec
    COUNTS["classes"] = len(_CACHE)
    return dec


def make_forward(orig):
    """FusedLayerNorm.forward with the pair form routed through the provider; every other call is ``orig`` (the extension) by name."""
    import torch

    def _aside_reason(self, input):
        """None when the call is the served form; else the word naming why the extension serves it."""
        ns = self.normalized_shape
        C = int(ns[-1]) if len(ns) == 1 else None
        if C is None or C not in SERVED_WIDTHS:
            return "form:c%s" % ("x".join(str(int(v)) for v in ns) if C is None else C)
        if self.weight is None or self.bias is None:
            return "form:affine_%s" % ("none" if (self.weight is None and self.bias is None) else "weight_only" if self.bias is None else "bias_only")
        if not input.is_cuda:
            return "cpu_tensor"
        if torch.is_grad_enabled() and (input.requires_grad or self.weight.requires_grad):
            return "autograd"
        if input.dtype not in (torch.bfloat16, torch.float32):
            return "dtype:%s" % str(input.dtype).replace("torch.", "")
        if torch.cuda.is_current_stream_capturing():
            return "capturing"                                            # the sampler step graph: the extension's current-stream build (lnstream) is what it records
        if _stepgraph_window():
            return "stepgraph_window"                                     # … and, under the big word, so do the EAGER steps of a sampler call the graph is planned for: the capture's
                                                                          # eager reference / the held replay's re-run serve what the graph records, not this lever's row
        if input.numel() == 0:
            return "empty"
        return None

    def forward(self, input):                                            # noqa: A002  (upstream's parameter name)
        if STATS["word"] is None:
            return orig(self, input)
        y = None
        C = rows = aligned = None
        try:
            why = _aside_reason(self, input)
            if why is None:
                C = int(self.normalized_shape[-1])
                x = input.contiguous()                                    # upstream's own first statement (FusedLayerNormAffineFunction.forward)
                rows = x.numel() // C
                aligned = (x.data_ptr() % 16 == 0) and ((C * x.element_size()) % 16 == 0)
                dec = _decide(x, C, rows, aligned)
                if dec[0] == "stock":
                    why = dec[1]
                else:
                    sel, arm = dec[1], dec[2]
                    d = x.dtype
                    y, _ = _ST["PL"].layer_norm(x, (C,), self.weight.to(d), self.bias.to(d), self.eps, selection=sel)
                    if y.dtype != d or y.shape != x.shape:
                        yd, ys, y = y.dtype, tuple(y.shape), None
                        raise TypeError("served row %s returned %s%s for %s%s" % (arm, str(yd).replace("torch.", ""), ys, str(d).replace("torch.", ""), tuple(x.shape)))
        except Exception as e:  # noqa: BLE001  the provider's / a served row's failure: that class back on the extension for the process, named; this call served by the extension
            y = None
            why = "error:%s" % type(e).__name__
            _bump("errors", type(e).__name__)
            if C and rows:
                _CACHE[(C, input.dtype, rows, aligned)] = ("stock", why)
            if int(sum(COUNTS["errors"].values())) <= 3:
                print(f"[{TAG}] NOTE ln_core: {type(e).__name__}: {str(e)[:160]} — this LayerNorm class continues on the extension (fast_layer_norm_cuda_v2)", file=sys.stderr, flush=True)
        if y is None:                                                     # the extension serves (upstream's statement, outside the try: its own errors are the caller's, never swallowed here)
            COUNTS["stock_calls"] += 1; _bump("asides", why)
            return orig(self, input)
        COUNTS["calls"] += 1; STATS["calls"] = COUNTS["calls"]
        _bump("rows", arm); _bump("cells", str(sel.cell)); _bump("dtypes", str(d).replace("torch.", ""))
        return y.view(input.shape) if y.shape != input.shape else y

    forward.__wrapped__ = orig
    forward._odde_ln_core = True
    return forward


def _sync() -> None:
    p = STATS.get("patch")
    if p is not None:
        STATS["installed"], STATS["armed"] = p.state == "installed", p.state == "armed"


def install(environ=None) -> None:
    """Read the word and patch ``FusedLayerNorm.forward`` now when upstream's module is imported, else at its import (the core's per-site
    patch). Idempotent. No word = nothing installed (the lever inert by name)."""
    from opt_core import autoload
    w = word(environ)
    STATS["word"] = w
    COUNTS["word"] = w
    COUNTS["active"] = w is not None
    if w is None:
        return
    _sync()
    if STATS["installed"] or STATS["armed"]:
        return
    try:
        STATS["patch"] = autoload.patch_attr_at_import(TARGET, ATTR, make_forward, tag=TAG, name=LEVER)
    except autoload.PatchError as e:
        raise ActivationError(str(e)) from None
    _sync()


def bound() -> bool:
    """True when upstream's class method is this lever's wrapper."""
    lm = sys.modules.get(TARGET)
    cls = getattr(lm, "FusedLayerNorm", None) if lm is not None else None
    f = getattr(cls, "forward", None) if cls is not None else None
    return bool(getattr(f, "_odde_ln_core", False))


def aside_word() -> str | None:
    """The lever's own account when it reached its site and served NO call through a provider row: 'aside:<reason:n,...>'; None otherwise
    (registry.ENGAGEMENT stepped_aside: inert by design, complete)."""
    c = COUNTS
    if int(c.get("calls") or 0) > 0 or int(c.get("stock_calls") or 0) == 0 or c.get("errors"):
        return None
    return "aside:" + ",".join(f"{k}:{v}" for k, v in sorted((c.get("asides") or {}).items()))


def fallbacks(planned) -> list:
    """Named events at exit: the word live but upstream's module imported without the wrapper bound, or served-row errors."""
    out = []
    if LEVER not in (planned or []) or STATS["word"] is None:
        return out
    _sync()
    if sys.modules.get(TARGET) is not None and not bound():
        out.append(f"{LEVER}:FusedLayerNorm.forward not bound (patch state {getattr(STATS.get('patch'), 'state', None)})")
    if COUNTS.get("errors"):
        out.append(f"{LEVER}:served_row_errors=" + ",".join(f"{k}:{v}" for k, v in sorted(COUNTS["errors"].items())))
    return out


def describe() -> dict:
    return {"word": COUNTS.get("word"), "calls": int(COUNTS.get("calls") or 0), "stock_calls": int(COUNTS.get("stock_calls") or 0),
            "rows": dict(COUNTS.get("rows") or {}), "cells": {k: v for k, v in (COUNTS.get("cells") or {}).items() if v},
            "asides": dict(COUNTS.get("asides") or {}), "refusals": dict(COUNTS.get("refusals") or {}), "errors": dict(COUNTS.get("errors") or {}),
            "classes": int(COUNTS.get("classes") or 0), "dtypes": dict(COUNTS.get("dtypes") or {}), "stack": COUNTS.get("stack"),
            "widths": ",".join(str(c) for c in SERVED_WIDTHS), "bound": bound()}


def kit_stats() -> dict:
    return describe()


def reset_counts() -> None:
    for k in ("calls", "stock_calls", "classes"):
        COUNTS[k] = 0
    for k in ("rows", "cells", "asides", "refusals", "errors", "dtypes"):
        COUNTS[k] = {}
    _CACHE.clear()
    STATS["calls"] = 0


def _reset() -> None:
    """Test hook: forget the patch and the word (the autoload registry keeps its own AttrPatch per site)."""
    reset_counts()
    STATS.update({"installed": False, "armed": False, "patch": None, "word": None})
    COUNTS.update({"active": False, "word": None, "stack": None})
    _ST.update({"PL": None, "cc": None, "stack": None, "has_triton": None, "stock_rows": ()})
