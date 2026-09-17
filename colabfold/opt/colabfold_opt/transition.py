"""TRANSITION — AlphaFold's ``Transition`` module (``alphafold/model/modules.py``: LayerNorm -> Linear(C -> f*C) -> ReLU -> Linear(f*C -> C), run by
stock through ``mapping.inference_subbatch`` in ``global_config.subbatch_size``-row chunks) at EVERY call site — the Evoformer's ``msa_transition``
(act [S, N, 256], f 4) and ``pair_transition`` ([N, N, 128], f 4), the extra-MSA stack's ([S_extra, N, 64] and [N, N, 128]), the template pair
stack's ``pair_transition`` ([N, N, 64], f 2) — bound to the shared core's JAX-family provider FACE
``opt_core.kernels.pallas.serve.transition(x, params, activation="relu", form="af2", word=<tier word>, direction="fwd", n_tokens=N)`` by the
mode's TIER WORD (``fast`` under `fast`, ``big`` under `big`; modes.TABLE). This module owns no kernel and no cell, tile or size table: the
provider resolves the word per call class — (jax line, compute capability, dtype, family ``af2_relu_c<C>_x<f>``, N bucket, fwd) — against
its cell table (``opt_core/kernels/pallas/PALLAS_CELLS.json``) and serves the row the cell names fastest (``mlp_transition`` /
``cd_transition``: one fused LayerNorm -> GEMM -> ReLU -> GEMM Pallas kernel per call; or ``xla``: the stock statement). When the word resolves
to the stock statement for a call class (its cell names ``xla`` fastest, or the table has no cell for the class on this stack) the module's
OWN body runs — stock's chunked statement, byte for byte — and the call is counted under ``fallback_by=cell_stock:<n>`` / ``no_cell:<n>``
(registry.STEP_ASIDE_RULES: by design, never a partial activation); a row the provider refuses at the call (its ``Refusal``, by kind) also
runs stock's body, counted under the refusal's kind. `exact` does not carry the lever: the provider's cells name no transition row bitwise
equal to the stock statement (their ``exact`` arm is ``xla`` at every AF2 transition cell), so `exact` keeps the stock module by name.

Numerics: tolerance class (the provider's ``tol`` rows: bf16 operands, f32 accumulation, a different summation order than XLA's GEMMs) — inside
`fast`'s identity form; f32 activations resolve to the stock statement on both supported parts (the cells' word). Under `big` at
``--n_gpu`` P > 1 the row-sharded pair stack runs the stock block bodies on each device's row block; this one-device lever steps
aside there by name (``reason=n_gpu>1``, modes.ONE_DEVICE_LEVERS).

Ablation: ``MODEL_OPT_LEVERS_OFF=TRANSITION`` removes the lever (ablation.py); ``MODEL_OPT_LEVERS_OFF=pallas:<row>`` (e.g. ``pallas:mlp_transition``)
is the provider's own word — that row steps aside by name inside the tier word and the next row of the order (or the stock statement) serves.

Census (the LEVER line at exit, report.lever_line over ``_STATE``): ``calls`` (served by a provider row) ``fallbacks`` ``fallback_by=<word>:<n>,…``
``word=<tier word>`` ``rows=<site>:<row>,…`` (the row the provider SERVED per call class; site = ``<unit>_c<C>x<f>``, unit pair | msa by the
operand's leading axes) ``served=<n>`` ``cells=<site>:<cell key>,…`` (the provider's cell each class resolved in; ``none`` = no cell)
``shapes=<site>@N<n>:<count>,…`` ``precision`` ``cc`` ``core`` ``jax``.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Optional, Tuple

NAME = "TRANSITION"
STRATEGY = "LOCAL.fused_transition"
MODULES = "alphafold.model.modules"
CLASS = "Transition"
MARKER = "_transition_served"
PROVIDER = "opt_core.kernels.pallas"               # the shared core's JAX-family provider (Refusal, family, TIER_WORDS)
SERVE = PROVIDER + ".serve"                        # its call faces: resolve(op, fam, dtype, n_tokens, word=…) / transition(x, params, …, word=…) (tests point both at a stub)
IMPL = SERVE + ":transition"                       # the LEVER line's impl= token
OP, FORM, ACTIVATION, DIRECTION = "transition", "af2", "relu", "fwd"
STOCK_ROW = "xla"                                  # the provider's word for the program's own statement
WORDS = {"fast": "fast", "big": "big"}         # kit mode -> the provider's tier word (modes.TABLE carries the lever in these modes only)
CELL_STOCK = "cell_stock"                          # fallback words (blank-free tokens of fallback_by=): the cell names the stock statement fastest — stock's body, by design
NO_CELL = "no_cell"                                #   no cell for this call class on this stack — the stock statement by name, by design
SHAPE = "shape"                                    #   an operand that is not [rows…, N, C] rank 3
DTYPE = "dtype"                                    #   activations neither bf16 nor f32
N_GPU_REASON = "n_gpu>1"                           # the LEVER line's reason under big at P > 1 (modes.ONE_DEVICE_LEVERS)
PARAM_SCOPES = ("input_layer_norm", "transition1", "transition2")   # the stock sub-module scopes (modules.py Transition)

_STATE: Dict[str, Any] = {"enabled": False, "calls": 0, "fallbacks": 0, "fallback_by": "none", "word": None, "rows": "none", "served": 0, "cells": "none",
                          "shapes": "none", "precision": "none", "cc": None, "core": None, "jax": None}
_RESOLVED: Dict[Tuple, Tuple[Any, Optional[str]]] = {}   # (cc, dtype word, C, factor, N) -> (Selection | None, fallback word | None) — the provider asked once per call class
_COUNTS: Dict[str, Any] = {"fallback_by": {}, "rows": {}, "cells": {}, "shapes": {}, "precision": set()}
_ORIG: Dict[str, Any] = {"cls": None, "modules": None}


class Refusal(RuntimeError):
    """The lever cannot run here (no jax GPU backend, a shared core without the provider face): the mode refuses by name, never degrades."""

    def __init__(self, kind: str, detail: str = ""):
        self.kind, self.detail = kind, detail
        super().__init__(f"{NAME}: {kind}" + (f" — {detail}" if detail else ""))


def _render(d: Dict) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))) or "none"


def compute_capability() -> Optional[str]:
    """The first jax GPU device's compute capability ("9.0", "8.0", …), None when jax or a GPU device is not there."""
    try:
        import jax
        for d in jax.devices():
            cc = getattr(d, "compute_capability", None)
            if cc:
                return str(cc)
    except Exception:  # noqa: BLE001 - a label; the caller names the absence
        return None
    return None


def word_for(mode: Optional[str]) -> str:
    """The provider's tier word for a kit mode (``fast`` -> fast, ``big`` -> big); a mode outside WORDS raises Refusal ``no_word``."""
    w = WORDS.get(str(mode or "").strip().lower())
    if w is None:
        raise Refusal("no_word", f"mode {mode!r} carries no tier word for this lever ({'|'.join(sorted(WORDS))})")
    return w


def require() -> Dict[str, Any]:
    """The lever's floor: jax importable with the GPU backend, the shared core's provider and its ``serve`` face with ``transition`` /
    ``resolve``. Raises ``Refusal`` (kind = no_jax | not_gpu_backend | core_face_missing) otherwise. A part / dtype / size the rows do not
    serve is never a refusal here: the provider's word names the stock statement for it and stock's body runs, counted."""
    try:
        import jax
    except Exception as e:  # noqa: BLE001
        raise Refusal("no_jax", f"jax is not importable here ({type(e).__name__}: {e})") from None
    if jax.default_backend() != "gpu":
        raise Refusal("not_gpu_backend", f"the default jax backend is {jax.default_backend()!r}, the provider's rows need the GPU backend")
    try:
        S = importlib.import_module(SERVE)
        importlib.import_module(PROVIDER)
    except Exception as e:  # noqa: BLE001
        raise Refusal("core_face_missing", f"{SERVE} is not importable ({type(e).__name__}: {e})") from None
    if not (callable(getattr(S, "transition", None)) and callable(getattr(S, "resolve", None))):
        raise Refusal("core_face_missing", f"{SERVE} carries no transition / resolve face (an older shared core)")
    core = getattr(importlib.import_module("opt_core"), "__version__", None)
    return {"jax": jax.__version__, "backend": "gpu", "cc": compute_capability(), "core": core}


def _dtype_name(dt) -> str:
    return str(getattr(dt, "name", dt))


def _dtype_word(dt) -> str:
    return {"bfloat16": "bf16", "float32": "f32"}.get(_dtype_name(dt), _dtype_name(dt))


def _family(P, c: int, factor: int) -> str:
    """The provider's transition family word for AF2's module at (C, factor): ``af2_relu_c<C>_x<factor>``."""
    try:
        return P.family(OP, form=FORM, c=int(c), factor=int(factor), activation=ACTIVATION)
    except Exception:  # noqa: BLE001 — an older face keyword set: the literal family word (PALLAS_CELLS 'families' transition)
        return f"{FORM}_{ACTIVATION}_c{int(c)}_x{int(factor)}"


def site_word(shape: Tuple[int, ...], factor: int) -> str:
    """``<unit>_c<C>x<f>``: unit ``pair`` for a square [N, N, C] operand (pair / template pair stack), ``msa`` otherwise (MSA / extra-MSA rows)."""
    unit = "pair" if len(shape) == 3 and shape[0] == shape[1] else "msa"
    return f"{unit}_c{int(shape[-1])}x{int(factor)}"


def _eligible(act) -> Optional[str]:
    """None when the face can take this operand, else the fallback word."""
    shape = tuple(int(x) for x in act.shape)
    if len(shape) != 3:
        return SHAPE
    if _dtype_name(act.dtype) not in ("bfloat16", "float32"):
        return DTYPE
    return None


def _resolve(S, P, cc: Optional[str], dt_word: str, c: int, factor: int, n: int):
    """(Selection, None) when the tier word resolves to a provider ROW for this call class; (Selection | None, word) when it resolves to the
    stock statement (``cell_stock`` / ``no_cell``) or the provider refuses the class (the refusal's kind) — asked once per (cc, dtype, C, f, N)."""
    key = (cc, dt_word, int(c), int(factor), int(n))
    if key in _RESOLVED:
        return _RESOLVED[key]
    sel, why = None, None
    try:
        sel = S.resolve(OP, _family(P, c, factor), dt_word, int(n), word=_STATE["word"], direction=DIRECTION, jax_line=_STATE.get("jax"), cc=cc)
    except getattr(P, "Refusal", Exception) as e:
        why = str(getattr(e, "kind", None) or "refused")
    if sel is not None and str(getattr(sel, "row", STOCK_ROW)) == STOCK_ROW:
        why = CELL_STOCK if getattr(sel, "cell", None) else NO_CELL
    _RESOLVED[key] = (sel, why)
    return sel, why


def _count_fallback(reason: str) -> None:
    _STATE["fallbacks"] += 1
    fb = _COUNTS["fallback_by"]
    fb[reason] = fb.get(reason, 0) + 1
    _STATE["fallback_by"] = _render(fb)


def _count_class(site: str, sel) -> None:
    """Record the row the provider resolved for a call class and the cell it resolved in (served or stock alike: the census names both)."""
    _COUNTS["rows"][site] = str(getattr(sel, "row", STOCK_ROW)) if sel is not None else STOCK_ROW
    _COUNTS["cells"][site] = str(getattr(sel, "cell_key", None) or "none") if sel is not None else "none"
    _STATE["rows"], _STATE["cells"] = _render(_COUNTS["rows"]), _render(_COUNTS["cells"])


def _count_served(site: str, n: int, dtype_name: str) -> None:
    _STATE["calls"] += 1
    _STATE["served"] += 1
    key = f"{site}@N{int(n)}"
    sh = _COUNTS["shapes"]
    sh[key] = sh.get(key, 0) + 1
    _STATE["shapes"] = _render(sh)
    _COUNTS["precision"].add("bf16" if dtype_name == "bfloat16" else ("f32" if dtype_name == "float32" else dtype_name))
    _STATE["precision"] = "+".join(sorted(_COUNTS["precision"]))


def _build(stock_cls, S, P):
    import haiku as hk

    def zeros(shape, dtype):
        import jax.numpy as jnp
        return jnp.zeros(shape, dtype)

    def ones(shape, dtype):
        import jax.numpy as jnp
        return jnp.ones(shape, dtype)

    class _Params(hk.Module):
        """A reader for one stock sub-module scope: ``_Params(name='transition1')('weights', shape, dtype, init)`` is the parameter
        ``<module>/transition1/weights`` — the stock Linear's / LayerNorm's own name, shape and dtype."""

        def __call__(self, pname, shape, dtype, init):
            return hk.get_parameter(pname, shape, dtype, init=init)

    class Transition(stock_cls):
        def __call__(self, act, mask, is_training=True):
            reason = _eligible(act)
            sel = None
            if reason is None:
                C = int(act.shape[-1])
                factor = int(self.config.num_intermediate_factor)
                N = int(act.shape[-2])
                site = site_word(tuple(int(x) for x in act.shape), factor)
                sel, reason = _resolve(S, P, _STATE["cc"], _dtype_word(act.dtype), C, factor, N)
                _count_class(site, sel)
            if reason is not None:                                          # the stock statement by the cell's word, or an operand the face does not take: stock's body, counted
                _count_fallback(reason)
                return super().__call__(act, mask, is_training=is_training)
            import jax.numpy as jnp
            f32, dt = jnp.float32, act.dtype
            F = int(C * factor)
            ln, t1, t2 = (_Params(name=s) for s in PARAM_SCOPES)
            params = {"ln_scale": ln("scale", (C,), f32, ones), "ln_offset": ln("offset", (C,), f32, zeros),
                      "w1": t1("weights", (C, F), dt, zeros), "b1": t1("bias", (F,), dt, zeros),
                      "w2": t2("weights", (F, C), dt, zeros), "b2": t2("bias", (C,), dt, zeros)}
            try:
                out = S.transition(act, params, activation=ACTIVATION, form=FORM, word=_STATE["word"], direction=DIRECTION, n_tokens=N,
                                   jax_line=_STATE.get("jax"), cc=_STATE["cc"], strict=True, selection=sel)
            except getattr(P, "Refusal", ()) as e:                          # the row refused THIS call (by kind): stock's body serves it, counted under the kind
                _count_fallback(str(getattr(e, "kind", None) or "refused"))
                return super().__call__(act, mask, is_training=is_training)
            _count_served(site, N, _dtype_name(dt))
            return out.astype(dt)

    setattr(Transition, MARKER, True)
    Transition.__qualname__ = Transition.__name__ = CLASS
    return Transition


def enable(mode: Optional[str] = None, modules=None) -> Dict[str, Any]:
    """Require the floor (``require``: a ``Refusal`` by name propagates — the activation's NOT ACTIVE), take the mode's tier word, import the
    provider and its face, rebind ``modules.Transition`` to the served subclass (idempotent). ``modules``: the target module (tests pass a stub
    with a ``Transition`` class); None imports ``alphafold.model.modules``."""
    word = word_for(mode)
    facts = require()
    P = importlib.import_module(PROVIDER)
    S = importlib.import_module(SERVE)
    m = modules if modules is not None else importlib.import_module(MODULES)
    cur = getattr(m, CLASS)
    _STATE["word"] = word
    _STATE["cc"] = facts.get("cc") if isinstance(facts, dict) else None
    _STATE["core"] = facts.get("core") if isinstance(facts, dict) else None
    _STATE["jax"] = facts.get("jax") if isinstance(facts, dict) else None
    if getattr(cur, MARKER, False):
        _STATE["enabled"] = True
        return dict(_STATE)
    setattr(m, CLASS, _build(cur, S, P))
    _ORIG.update(cls=cur, modules=m)
    _STATE["enabled"] = True
    return dict(_STATE)


def disable() -> None:
    m, cls = _ORIG["modules"], _ORIG["cls"]
    if m is not None and cls is not None:
        setattr(m, CLASS, cls)
    _STATE["enabled"] = False


def marker_present(modules=None) -> Optional[bool]:
    m = modules if modules is not None else sys.modules.get(MODULES)
    if m is None:
        return None
    return bool(getattr(getattr(m, CLASS, None), MARKER, False))


def reset_for_tests() -> None:
    disable()
    _STATE.update(enabled=False, calls=0, fallbacks=0, fallback_by="none", word=None, rows="none", served=0, cells="none", shapes="none", precision="none",
                  cc=None, core=None, jax=None)
    _COUNTS.update(fallback_by={}, rows={}, cells={}, shapes={}, precision=set())
    _ORIG.update(cls=None, modules=None)
    _RESOLVED.clear()
