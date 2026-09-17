"""chai1_exactln.serve — the eager trunk's LayerNorm STATEMENTS bound to the shared core's LayerNorm provider (``opt_core.kernels.ln``) BY TIER WORD.

The kit mode names the word (``exact`` -> ``exact``, ``fast`` -> ``fast``, ``big`` -> ``big``: ``TIER_WORDS``) and every statement CLASS the hook
sees — (statement kind, C, affine, rows, layout, eps) — asks the provider ONCE with its own key: ``select(cc, dtype, cell_for_rows(rows, n, C), n,
word=<tier>, widen/out for the bf16 kind, C, rows, stack)``.  The row the provider's cell names serves the class for the rest of the process:

  * a CARRIED row — ``exactln[:widen]`` (the ATen replica: the exact word's row wherever it is bitwise AND at or above the statement's speed on this
    stack), ``fastln[:lp]`` / ``rfd`` / ``dtk_ln`` / ``ln_rows`` (the Triton rows of the fast / big words) — served through the provider's
    own serving face (``exactln`` through the replica's plan cache, the others through ``opt_core.kernels.ln.layer_norm(selection=...)``);
  * a STOCK row (``aten`` / ``aten_autocast``: the statement is what the cell names — the single-track rows —, or the exact word's floor on a
    stack the replica is not listed bitwise on, or a width no cell family lists) — ``NotImplemented``: the caller's own statement runs, counted
    ``statement:<row>`` (a named, expected census word; nothing is substituted).

Two statement kinds cover the Chai-1 trunk (chai1_eager/trunk.py ``ln_bf16`` / ``ln_f32``; the transpiled confidence head's ``torch.layer_norm``
through ``bind_flat``):
  ``bf16``  ``F.layer_norm(x.to(float32), (C,), w, b, eps).to(bfloat16)`` — the provider's ``bf16o`` form (bf16 x, fp32 parameters and statistics,
            bf16 OUT: ``widen=True, out='bf16'``);
  ``f32``   ``F.layer_norm(x, (C,), w, b, eps)`` — the ``fp32`` form.

The exact tier's run-time rule for arithmetic replicas: the FIRST call of every class an exact-class row (``exactln``) serves runs torch's
statement too and is compared bit for bit (NaN-strict); a differing class takes the statement for the rest of the process BY NAME
(``bitcmp_mismatch:<class>``) and the fail-closed gate refuses the run at exit.  Tolerance-class rows (the fast / big words' Triton rows) are not
bit-compared.  Other census words: ``cpu`` / ``grad`` / ``dtype:*`` (outside
the hook's envelope), ``refused:<row>:<reason>`` (a row refused at serve time, named on stderr once: its class stays on the statement).
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Tuple

SURFACE = ("package", "select_by_word", "install", "gate", "line", "ln", "bind_flat", "cast_cached", "conf_fields", "reset")   # names pairtrack / the tests reach (chai1_opt tests/test_lever_surfaces.py)
TAG = "chai1-opt"
NAME = "exactln"                                  # the kit's lever name (registry / modes / the ACTIVE line's word): stable
PROVIDER = "opt_core.kernels.ln"                  # the LayerNorm family's ONE provider (pairtrack.SERVE_MODULE)
TIER_WORDS = {"exact": "exact", "fast": "fast", "big": "big"}   # kit mode -> the provider's tier word (big never shares fast's word)
LINE_CLASS = ("bf16", 256, 512)                   # the class the LEVER line's provider= fact is resolved for at install: the pairformer's pair LayerNorm (bf16 kind, c 256) at crop 512
WCAST_WEIGHTS_NUMEL = 1 << 22                     # the constant-cast cache's element cap (the confidence head's weights only; memory tier)
WCAST_MAX_NUMEL = WCAST_WEIGHTS_NUMEL             # live activations are not served, only the component's constant weights: an activation's cast is never asked twice
EXPECTED = ("statement", "refused")               # census words a healthy run may show: the provider named the statement for a class; a row refused by name at serve time
MAX_CLASSES = 4096
_S: Dict[str, Any] = {"LN": None, "E": None, "installed": False, "mode": None, "word": None, "stack": None, "cc": None, "has_triton": None, "has_nvrtc": None,
                      "classes": {}, "hot": {}, "served": {}, "rows_served": {}, "statement": {}, "fallback": {}, "errors": {}, "class_ok": {}, "selftest": {},
                      "refused": {}, "facts": {}, "first_error": None}


def _cnt(d: Dict[str, int], k: str, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def package():
    """The shared core's LayerNorm provider module (``opt_core.kernels.ln``: rows, cells, ``select`` and the serving face), imported once."""
    if _S["LN"] is None:
        import importlib
        _S["LN"] = importlib.import_module(PROVIDER)
    return _S["LN"]


def select_by_word(cc: str, word: Optional[str] = None, stack: Optional[str] = None, kind: str = LINE_CLASS[0], c: int = LINE_CLASS[1], crop: int = LINE_CLASS[2]) -> dict:
    """The provider's Selection for a tier word on this card at one trunk class (default: the pair LayerNorm, bf16 kind, c 256, crop 512 —
    the LEVER line's fact); a ``Refusal`` -> RuntimeError by name."""
    LN = package()
    word = word or _S["word"] or "fast"
    widen = kind == "bf16"
    rows = crop * crop if c != 384 else crop
    try:
        sel = LN.select(cc, "bf16" if widen else "fp32", LN.cell_for_rows(rows, crop, c), crop, word=word, widen=widen, out=("bf16" if widen else None), C=c, rows=rows,
                        stack=stack, has_triton=_S["has_triton"], has_nvrtc=_S["has_nvrtc"])          # the stack word for EVERY tier word: the provider's table is keyed by stack (exact: its bitwise listing too)
    except Exception as e:                                     # LN.Refusal or a table error: named
        raise RuntimeError(f"{PROVIDER} refused word {word!r} on cc {cc}: {getattr(e, 'kind', '')} {e}")
    return {"row": LN.arm_word(sel.row, sel.variant), "variant": sel.variant, "cls": sel.cls, "cell": sel.cell, "word": word, "provider": PROVIDER, "reason": sel.reason,
            "statement": sel.row in LN.STOCK_ROWS}


def install(mode: str = "fast") -> dict:
    """Bind the tier word of ``mode``; import the provider and the replica's bindings (their absence is a fact the provider's exact rows refuse by
    name on, not an install failure); record the stack word the exact vouch is keyed by.  Returns facts for the LEVER line."""
    import torch
    if mode not in TIER_WORDS:
        raise RuntimeError(f"mode:{mode}: the LayerNorm lever binds the provider's tier words for {sorted(TIER_WORDS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("no_cuda: the LayerNorm lever serves CUDA layer_norm calls")
    LN = package()
    import opt_core
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    cc = f"{p.major}.{p.minor}"
    try:
        import triton  # noqa: F401
        has_triton = True
    except ImportError:
        has_triton = False
    E, has_nvrtc, why_e = None, False, None
    try:
        E = LN.carried_module("exactln")                        # the replica's package (NVRTC through cuda.bindings | ctypes; prebuilt CUBINs for sm_90 / sm_80)
        has_nvrtc = True
    except Exception as e:                                     # LN.Refusal('import:...'): the exact rows refuse by name in select(); the floor serves
        why_e = f"{getattr(e, 'kind', type(e).__name__)}"
    word = TIER_WORDS[mode]
    _S.update(E=E, mode=mode, word=word, stack=LN.stack_word(), cc=cc, has_triton=has_triton, has_nvrtc=has_nvrtc, installed=True)
    sel = select_by_word(cc, word, _S["stack"])
    _S["facts"] = {"torch": torch.__version__, "cc": f"{p.major}{p.minor}", "gpu": p.name, "word": word, "stack": _S["stack"], "core": getattr(opt_core, "__version__", "?"),
                   "provider": f"{PROVIDER}:{word}->{sel['row']}@{sel['cell'] or 'no_cell'}", "has_triton": has_triton, "has_nvrtc": has_nvrtc, "exactln_import": why_e}
    return dict(_S["facts"])


def _bits_equal(a, b) -> bool:
    import torch
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    it = {torch.float32: torch.int32, torch.bfloat16: torch.int16}.get(a.dtype)
    if it is None:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.contiguous().view(it), b.contiguous().view(it)))


def stock(kind: str, x, c: int, w, b, eps: float):
    """The trunk's own statements (chai1_eager/trunk.py), verbatim."""
    import torch
    import torch.nn.functional as F
    if kind == "bf16":
        return F.layer_norm(x.to(torch.float32), (c,), w, b, eps).to(torch.bfloat16)
    return F.layer_norm(x, (c,), w, b, eps)


def _aligned(LN, x, C) -> bool:
    f = getattr(LN, "_rows_aligned", None)
    try:
        return bool(f(x, C)) if f is not None else True
    except Exception:                                          # a 1-d tensor / exotic view: the provider's own serving face re-derives it
        return True


def _decide(kind: str, x, C: int, w, b, rows: int, aligned: bool) -> tuple:
    """Ask the provider once for this class: -> (arm | None, Selection | None, census word for the statement path | None, exact_class)."""
    LN = _S["LN"]
    widen = kind == "bf16"
    n = max(int(round(rows ** 0.5)), 1)                        # informational: the provider buckets the cell by ROWS
    cell = LN.cell_for_rows(rows, n, C)
    try:
        sel = LN.select(_S["cc"], x.dtype, cell, n, word=_S["word"], widen=widen, out=("bf16" if widen else None), C=C, rows=rows,
                        stack=_S["stack"], has_triton=_S["has_triton"], has_nvrtc=_S["has_nvrtc"], aligned=aligned)
    except Exception as e:                                     # LN.Refusal (a tier word raises only under prefer=) or a table error: the statement, by name
        return None, None, f"refused:{getattr(e, 'row', None) or '-'}:{getattr(e, 'kind', type(e).__name__)}", False
    arm = LN.arm_word(sel.row, sel.variant)
    if sel.row in LN.STOCK_ROWS:
        return None, sel, f"statement:{arm}", False
    return arm, sel, None, sel.row in LN.EXACT_ROWS


def _refuse_class(key, arm: str, why: str) -> None:
    """A carried row refused / mis-served a class at serve time: the class takes the statement from now on, named on stderr once."""
    _S["classes"][key] = (None, None, f"refused:{arm}:{why}", False)
    tag = f"{arm}:{why}"
    if tag not in _S["refused"]:
        sys.stderr.write(f"{TAG} LEVER {NAME}: row {arm} refused class {key[0]}:C{key[1]}:rows{key[4]} by name ({why}); the statement serves it\n")
        sys.stderr.flush()
    _cnt(_S["refused"], tag)


def ln(kind: str, x, c: int, w=None, b=None, eps: float = 1e-5):
    """Serve one LayerNorm statement of ``kind`` ('bf16' | 'f32') through the row the provider names for its class, or return NotImplemented (the
    caller runs its own statement), counted either way."""
    import torch
    if not _S["installed"]:
        return NotImplemented
    fb = _S["fallback"]
    if not x.is_cuda:
        _cnt(fb, "cpu"); return NotImplemented
    if torch.is_grad_enabled() and (x.requires_grad or (w is not None and w.requires_grad)):
        _cnt(fb, "grad"); return NotImplemented
    widen = kind == "bf16"
    if x.dtype != (torch.bfloat16 if widen else torch.float32):
        _cnt(fb, f"dtype:{str(x.dtype).replace('torch.', '')}"); return NotImplemented
    C = int(c)
    rows = (x.numel() // C) if C > 0 else 0
    if rows == 0 or int(x.shape[-1]) != C:
        _cnt(fb, "statement:shape"); return NotImplemented
    LN = _S["LN"]
    aligned = _aligned(LN, x, C)
    key = (kind, C, w is not None, b is not None, rows, ("c" if x.is_contiguous() else "s") + ("" if aligned else "u"))
    d = _S["classes"].get(key)
    if d is None:
        d = _decide(kind, x, C, w, b, rows, aligned)
        if len(_S["classes"]) < MAX_CLASSES:
            _S["classes"][key] = d
    arm, sel, word_fb, exact_class = d
    if arm is None:
        _cnt(fb, word_fb); _cnt(_S["statement"], word_fb.split(":", 1)[1] if word_fb.startswith("statement:") else word_fb); return NotImplemented
    out_dtype = torch.bfloat16 if widen else None
    label = f"{kind}:C{C}:a{int(w is not None)}{int(b is not None)}:{arm}:{key[5]}:eps{float(eps):g}"
    ok = _S["class_ok"].get(label)
    if ok is False:
        _cnt(fb, f"bitcmp_mismatch:{label}"); return NotImplemented
    try:
        if sel.row == "exactln":                               # the replica through its plan cache (one plan per operand signature)
            E = _S["E"]
            sig = E.call_signature(x, (C,), w, b, widen, out_dtype) + (float(eps), kind)
            pl = _S["hot"].get(sig)
            if pl is None:
                try:
                    pl = E.plan(x, (C,), w, b, widen=widen)
                except E.Unsupported as u:                     # a layout / width the replica does not serve: this call takes the statement, by name
                    _cnt(fb, f"refused:{arm}:{u.reason}"); return NotImplemented
                if len(_S["hot"]) < MAX_CLASSES:
                    _S["hot"][sig] = pl
            y = E.layer_norm(x, (C,), w, b, eps, widen=widen, out_dtype=out_dtype, _plan=pl)
        else:                                                  # a Triton row through the provider's serving face
            try:
                y, _sel = LN.layer_norm(x, (C,), w, b, eps, selection=sel, out_dtype=out_dtype)
            except LN.Refusal as r:                            # a library absent / an envelope edge: the class takes the statement, by name
                _refuse_class(key, arm, getattr(r, "kind", str(r))[:80].replace(" ", "_"))
                _cnt(fb, f"refused:{arm}:{getattr(r, 'kind', 'refusal')[:40]}"); return NotImplemented
    except Exception as ex:  # noqa: BLE001 — a kernel / driver error: torch's statement serves this call, the gate refuses at exit
        _cnt(_S["errors"], f"{type(ex).__name__}")
        if _S["first_error"] is None:
            _S["first_error"] = f"{label}: {ex!r}"
        return NotImplemented
    if ok is None:                                             # first call of the class
        want = torch.bfloat16 if widen else torch.float32
        if y.dtype != want or tuple(y.shape) != tuple(x.shape):   # a row whose serving form differs from the statement's: refused by name, the statement serves
            _refuse_class(key, arm, f"form:{str(y.dtype).replace('torch.', '')}")
            _cnt(fb, f"refused:{arm}:form"); return NotImplemented
        if exact_class:                                        # an arithmetic replica is served only where it is bit-identical to the statement on these operands
            ref = stock(kind, x, C, w, b, eps)
            same = _bits_equal(y, ref)
            _S["class_ok"][label] = same
            _S["selftest"][label] = "pass" if same else "MISMATCH"
            if not same:
                _cnt(fb, f"bitcmp_mismatch:{label}")
                sys.stderr.write(f"{TAG} LEVER {NAME} BIT-COMPARE MISMATCH class={label}: the class takes torch's statement for this process (refused by name at exit)\n")
                sys.stderr.flush()
                return ref
        else:
            _S["class_ok"][label] = True
    _cnt(_S["served"], label); _cnt(_S["rows_served"], arm)
    return y


def shim_layer_norm(orig):
    """``torch.layer_norm`` for a transpiled component's namespace (ts2eager TorchShim): fp32 CUDA calls over one trailing dimension ask the provider
    per class like the trunk's ``f32`` statements (the single-track widths resolve to the statement by the cells); everything else ``orig``."""
    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True):
        _CONF["ln_calls"] += 1
        ns = tuple(int(v) for v in (normalized_shape if isinstance(normalized_shape, (list, tuple)) else (normalized_shape,)))
        if len(ns) == 1 and input.is_cuda and input.dim() >= 1 and int(input.shape[-1]) == ns[0]:
            import torch
            if input.dtype == torch.float32:
                r = ln("f32", input, ns[0], weight, bias, float(eps))
                if r is not NotImplemented:
                    return r
            else:
                _cnt(_S["fallback"], f"dtype:{str(input.dtype).replace('torch.', '')}")
        return orig(input, normalized_shape, weight, bias, eps, cudnn_enable)
    layer_norm.chai1_opt_lever = NAME
    return layer_norm


# ----------------------------------------------------------------------------------------------------------------- evidence
def census() -> dict:
    served = sum(_S["served"].values()); fb = dict(sorted(_S["fallback"].items()))
    return {"served": served, "served_by": dict(sorted(_S["served"].items())), "rows_served": dict(sorted(_S["rows_served"].items())),
            "statement": dict(sorted(_S["statement"].items())), "fallback": sum(fb.values()), "fallback_by": fb, "errors": dict(_S["errors"]),
            "first_error": _S["first_error"], "selftest": dict(_S["selftest"]), "refused": dict(_S["refused"]), "classes": len(_S["classes"]),
            "word": _S["word"], "facts": dict(_S["facts"])}


def unexpected() -> list:
    return [k for k in _S["fallback"] if not any(k == e or k.startswith(e + ":") for e in EXPECTED)]


def gate() -> Tuple[bool, str]:
    """Fail-closed: installed, the hook saw the trunk's calls, no kernel error, no mismatched class, no unexpected census word."""
    if not _S["installed"]:
        return False, "not_installed"
    bad = []
    if _S["errors"]:
        bad.append(f"kernel_errors={dict(_S['errors'])} first={_S['first_error']}")
    mism = [k for k, v in _S["selftest"].items() if v != "pass"]
    if mism:
        bad.append("bitcmp_mismatch=" + ",".join(mism))
    ux = unexpected()
    if ux:
        bad.append("unexpected_fallback=" + ",".join(f"{k}:{_S['fallback'][k]}" for k in ux))
    if not _S["classes"]:
        bad.append("no_calls")
    return (not bad), "; ".join(bad)


def _class_word() -> str:
    if _S["word"] == "exact":
        return "exact(bitwise: the provider's exact-class row per cell where vouched on this stack, first-call bit-compare per class; the statement by name elsewhere)"
    return f"tolerance(word={_S['word']}: the provider's measured row per cell; inside the engine's identity band)"


def line(tag: str = TAG) -> str:
    c = census(); f = c["facts"]
    ok, why = gate()
    rows = ",".join(f"{k}:{v}" for k, v in c["rows_served"].items()) or "-"
    stm = ",".join(f"{k}:{v}" for k, v in c["statement"].items()) or "-"
    st = c["selftest"]
    return (f"{tag} LEVER {NAME} pairtrack word={c['word']} provider={PROVIDER} class={_class_word()} served={c['served']} rows={rows} statement={stm} "
            f"fallback={c['fallback']} fallback_by={','.join(f'{k}:{v}' for k, v in c['fallback_by'].items()) or '-'} classes={c['classes']} "
            f"selftest={'all_pass' if st and all(v == 'pass' for v in st.values()) else ','.join(f'{k}={v}' for k, v in st.items()) or '-'} "
            f"refused={','.join(f'{k}:{v}' for k, v in c['refused'].items()) or '-'} stack={f.get('stack')} cc=sm_{f.get('cc')} core={f.get('core')} "
            f"line_class={f.get('provider')} conf_bound={_CONF['bound']} conf_wcast_hit={_CONF['wcast_hit']} conf_wcast_miss={_CONF['wcast_miss']} "
            f"conf_wcast_entries={_CONF['wcast_entries']} gate={'ok' if ok else 'REFUSED:' + why}")


# ----------------------------------------------------------------------------------------------------------- confidence head (flat)
_CONF: Dict[str, Any] = {"bound": False, "wcast_hit": 0, "wcast_miss": 0, "wcast_entries": 0, "ln_calls": 0}
_WCAST: Dict[int, Any] = {}                          # id(src) -> (weakref(src), version, dtype code, cast result): constants only survive (a dead source misses)


def bind_flat(flat, name: str = "confidence_head") -> str:
    """Serve the transpiled (ts2eager) component ``flat``: its methods' ``torch`` is a per-component TorchShim instance; two names are bound on
    THAT instance (nothing else in the process sees them): ``layer_norm`` -> the provider per class (``shim_layer_norm``: the tier word's row,
    bit-compared per class when it is an exact-class row), and ``to`` -> ``cast_cached``: the dtype cast of a CONSTANT (a tensor that is the same
    live object with the same version at a later call: the component's fp32 weights, cast to bf16 at every use by the export) computed once and
    reused — a cast is a pure function of its operand, so the reused result is the statement's own bits.  Returns the binding word for the LEVER line."""
    import torch
    g = None
    rt = getattr(flat, "_rt", None)                                     # chai1_eager.ts2eager.EagerTSModule: the component's Runtime holds the globals every
    if isinstance(getattr(rt, "globals", None), dict) and "torch" in rt.globals:   # method is compiled against ({"torch": TorchShim(), ...}) — present before any method
        g = rt.globals                                                  # compiles (methods compile lazily at their first call)
    else:
        for n in [n for n in dir(flat) if n.startswith("forward_")]:
            gg = getattr(getattr(getattr(flat, n), "__func__", None), "__globals__", None)
            if isinstance(gg, dict) and "torch" in gg:
                g = gg
                break
    if g is None:
        _cnt(_S["fallback"], f"{name}_shim_absent")
        return f"{name}:shim_absent"
    shim = g["torch"]
    orig_to = shim.to
    shim.layer_norm = shim_layer_norm(torch.layer_norm)
    shim.to = cast_cached(orig_to)
    _CONF["bound"] = True
    return f"{name}:layer_norm+to"


def cast_cached(orig_to):
    """TorchShim.to with the constant-cast cache: ``torch.to(x, <dtype int>[, non_blocking, copy])`` on a small (<= 2**22 elements) floating
    CUDA tensor that does not require grad is looked up by identity; a hit is valid only while the SAME object is alive with the same version
    counter (weakref + ``_version``) — activations die between calls and miss, the component's weights hit from their second use on."""
    import weakref
    import torch

    def to(x, *a, **k):
        if (len(a) >= 1 and isinstance(a[0], int) and not k and (len(a) < 3 or not a[2]) and torch.is_tensor(x) and x.is_cuda
                and x.is_floating_point() and not x.requires_grad and x.numel() <= WCAST_MAX_NUMEL):
            key = id(x); e = _WCAST.get(key)
            if e is not None:
                src = e[0]()
                if src is x and e[1] == x._version and e[2] == a[0]:
                    _CONF["wcast_hit"] += 1
                    return e[3]
                _WCAST.pop(key, None)
            r = orig_to(x, *a, **k)
            if r is not x:                                            # a real cast (same-dtype .to returns self: nothing to keep)
                try:
                    _WCAST[key] = (weakref.ref(x, lambda _w, _k=key: _WCAST.pop(_k, None)), x._version, a[0], r)   # the entry leaves with its source:
                    _CONF["wcast_entries"] = max(_CONF["wcast_entries"], len(_WCAST))                               # an activation's cast is never retained past it
                except TypeError:
                    pass
            _CONF["wcast_miss"] += 1
            return r
        return orig_to(x, *a, **k)
    to.chai1_opt_lever = NAME
    return to


def conf_fields() -> dict:
    return dict(_CONF)


def reset() -> None:
    _S.update(installed=False, mode=None, word=None, stack=None, cc=None, classes={}, hot={}, served={}, rows_served={}, statement={}, fallback={}, errors={},
              class_ok={}, selftest={}, refused={}, facts={}, first_error=None)
    _CONF.update(bound=False, wcast_hit=0, wcast_miss=0, wcast_entries=0, ln_calls=0); _WCAST.clear()
