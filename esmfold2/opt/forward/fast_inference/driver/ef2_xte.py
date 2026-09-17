"""ef2_xte — lever ``xte`` (EXACT set): the pair trunk's fused inference Transition (every d=256 / hidden=1024 C.Transition: folding trunk,
lm_encoder, parcae coda, confidence trunk; ``x + T(x)``) through the shared core's transition provider BY TIER WORD — the mode's own word
(``exact`` in exact) on the provider's ESM-family fused-statement cells (family esmpair, form esmfused).  The provider's cell for this card,
stack and size decides: where the exact word names a row bitwise-vouched on this stack (compute capability 9.0 from the vouch floor up: the
one-kernel fused statement esm_fused_exact) that row serves; where it names the statement itself (compute capability 8.0 today; sizes below
the vouch floor) or refuses BY NAME, the module's own fused statement runs — counted, named once on stderr, never silent.  This module holds NO
kernel and NO row window: instance bindings, eligibility, weight packs (through ef2_pair_v2.serve) and the engagement record only.

    install(model)   binds the eligible C.Transition instances (d = 256, LayerNorm weight + bias, eps = the fused kernel's literal, the fused
                     backend installed on the module); packs built now (eager, before any graph capture).  ef2_server.configure calls it in the pair slot.
    EF2_PAIR_WORD    the A/B override of the bound word (shared with ef2_pair_v2).
"""
import collections
import sys
import time
import types

import torch
import transformers.models.esmfold2.modeling_esmfold2_common as C

import ef2_pair_v2 as P

VERSION = "xte.2.0"
LEVERS = ("xte",)
XTE_C, XTE_HIDDEN = 256, 1024                 # the provider's ESM-family cell: d = 256, hidden 1024
FUSED_EPS = 1e-5                             # upstream's fused kernel evaluates LayerNorm with this literal; a module whose norm.eps differs is not eligible (stays on its statement)
UPSTREAM_SOURCES = (("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"),)   # the upstream function this lever re-issues (ef2_srcguard)
STATS = collections.Counter()                # xte_calls / xte_fallthrough — per call, this process
_STATE = {"on": False, "refusal": None, "modules": 0, "patched": [], "install_s": None}
_BF16 = torch.bfloat16


class XteUnavailable(RuntimeError):
    """install(strict=True) on a model with no eligible module."""


def _slug(s):
    return str(s).replace(" ", "_").replace("=", ":")[:96]


def _autocast_bf16():
    return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == _BF16


def eligible_modules(model):
    """``[(qualified name, module)]``: the C.Transition instances this lever binds (d = 256, hidden 1024, LayerNorm weight + bias, eps = the
    fused kernel's literal, no linear bias, the fused backend installed on the module)."""
    out = []
    for name, m in model.named_modules():
        if not isinstance(m, C.Transition):
            continue
        norm, ffn = getattr(m, "norm", None), getattr(m, "ffn", None)
        if norm is None or ffn is None or tuple(norm.normalized_shape) != (XTE_C,) or norm.weight is None or norm.bias is None:
            continue
        if abs(float(norm.eps) - FUSED_EPS) > 0.0 or int(getattr(ffn, "hidden_features", 0)) != XTE_HIDDEN or getattr(m, "_fused_swiglu", None) is None:
            continue
        if tuple(ffn.w12.weight.shape) != (2 * XTE_HIDDEN, XTE_C) or tuple(ffn.w3.weight.shape) != (XTE_C, XTE_HIDDEN) or ffn.w12.bias is not None or ffn.w3.bias is not None:
            continue
        out.append((name, m))
    return out


def _xte_forward(self, x):
    """C.Transition.forward (instance binding, lever xte): x + Transition(x) from the provider's row for the tier word when the call is the fused
    inference statement's; every other call (and every call the word leaves on the statement) runs the class forward of this moment, counted."""
    if not (_STATE["on"] and x.is_cuda and x.dtype == _BF16 and x.dim() >= 2 and x.shape[-1] == XTE_C and x.numel() > 0 and not torch.is_grad_enabled()
            and self._can_use_fused_path(x) and (self._chunk_size is None or x.shape[1] <= self._chunk_size) and _autocast_bf16()):
        STATS["xte_fallthrough"] += 1
        return C.Transition.forward(self, x)
    y = P.serve(self, x, residual=True, kind="xte")
    if y is None:
        STATS["xte_fallthrough"] += 1
        return C.Transition.forward(self, x)
    STATS["xte_calls"] += 1
    return y


def install(model, xte=True, strict=False):
    """Bind the eligible C.Transition instances of ``model`` to the provider by the tier word (packs built now).  A model with no eligible module
    leaves the lever off BY NAME (refusal() names it; strict=True raises XteUnavailable).  Idempotent."""
    import ef2_srcguard
    ef2_srcguard.check("xte", keys=UPSTREAM_SOURCES)
    t0 = time.time()
    if not xte:
        return describe()
    if _STATE["on"]:
        return describe()
    P._face(); P.bound_word()
    mods = [(n, m) for n, m in eligible_modules(model) if m.ffn.w12.weight.is_cuda]
    if not mods:
        _STATE["on"] = False; _STATE["refusal"] = "no_eligible_module"
        print(f"[ef2_xte] xte steps aside by name: {_STATE['refusal']} (the trunk Transition keeps its own statement)", file=sys.stderr, flush=True)
        if strict:
            raise XteUnavailable(_STATE["refusal"])
        return describe()
    with torch.no_grad():
        for _name, m in mods:                                # packs built now (eager, before any graph capture); instance forwards bound
            P._pack(m)
            m.forward = types.MethodType(_xte_forward, m)
            _STATE["patched"].append(m)
    _STATE["modules"] = len(mods)
    _STATE["on"] = True
    _STATE["refusal"] = None
    _STATE["install_s"] = round(time.time() - t0, 2)
    return describe()


def uninstall(model=None):
    """Drop every instance binding (the class forward of the moment serves again); keeps the packs cached on the modules."""
    for m in _STATE["patched"]:
        try:
            del m.forward
        except AttributeError:
            pass
    _STATE["patched"] = []
    _STATE["on"] = False
    _STATE["modules"] = 0
    return describe()


def levers_on():
    return {"xte": bool(_STATE["on"])}


def refusal():
    return _STATE["refusal"]


def stats():
    """Flat per-process counters for the package's EXIT tally: xte_calls / xte_fallthrough / xte_modules (+ the provider rows served / left on the statement)."""
    out = {k: int(STATS.get(k, 0)) for k in ("xte_calls", "xte_fallthrough")}
    out["xte_modules"] = int(_STATE.get("modules") or 0)
    return out


def evidence():
    """Blank-free words for the LEVER line: word=<bound word> core=<version> modules=<n> provider=opt_core.kernels.transition (+ rows= / refused= once calls ran)."""
    ev = dict(P.bind_words())
    ev["modules"] = int(_STATE.get("modules") or 0)
    return ev


def describe():
    return {"version": VERSION, "levers": levers_on(), "refusal": _STATE["refusal"], "word": P.bound_word() if P._STATE.get("word") else None,
            "core": P._STATE.get("core"), "modules": int(_STATE.get("modules") or 0), "install_s": _STATE["install_s"], "stats": stats()}
