"""TRIMUL_PALLAS — AlphaFold-Multimer's fused-projection ``TriangleMultiplication`` (outgoing AND incoming, every call site: Evoformer, extra-MSA
stack, template pair stack) served through the shared core's JAX-family provider ``opt_core.kernels.pallas`` by its TIER WORD: the call face
``opt_core.kernels.pallas.serve.triangle_multiplication(act, mask, params, equation=…, form="af2", unit=…, word=<tier>, direction="fwd")``
resolves each call's cell (jax line x compute capability x dtype x family x N bucket, ``PALLAS_CELLS.json``) to the row its cell table names
fastest there (``word=fast``), lowest-peak (``word=big``) or bitwise the stock op (``word=exact`` — by the provider's exact rule
the stock statement by name wherever no arm is vouched bitwise, which is why the exact mode does not carry this lever), and serves it. This module
is the AlphaFold-multimer / colabfold adapter around that face and owns no kernel, no row name and no cell or launch table (strategy F2.trimul):
the rows (``cd_trimul`` — fused Pallas kernels the core carries; ``fpf_trimul``; ``native_xla``; …), their launch
settings and the per-card / per-size choices are the provider's, read from its cell table at each call.

What stock does: ``modules.TriangleMultiplication._fused_triangle_multiplication`` (``alphafold/model/modules.py:1471-1512``, the form
``fuse_projection_weights`` selects — every configuration colabfold builds: ``colabfold/alphafold/models.py:119-124`` sets it and
``alphafold/model/utils.flat_params_to_haiku(fuse=True)`` fuses an older checkpoint's projections at load, the alphafold2_ptm ones included):
``x = LN_in(act)``; ``proj = mask·(x@W[Cz,2C] + b)·σ(x@G[Cz,2C] + g)``; left | right halves; ``t = einsum(c.equation, left, right)``;
``out = (LN_c(t)@Wo + bo)·σ(x@Wgl + bgl)`` — ~15 XLA kernels per call (the einsum's operands and result transposed through HBM around cuBLAS).

What the lever does, without editing stock: ``enable(mode=…)`` rebinds ``alphafold.model.modules.TriangleMultiplication`` to the subclass built
below before any model is traced (every call site resolves the class through ``modules`` at trace time). Its ``__call__`` (a) checks the call
against the face's envelope (``_eligible``: fused layout, rank-3 square activations, channel counts powers of two, bf16 / f32, one of the two
equations, an ``[N, N]`` mask), (b) asks the provider ONCE per call class (``_selection``: ``serve.resolve('trimul', family, dtype, N,
word=<tier>)`` memoised on (cc, dtype, C_z, C, equation, N)) which row the tier word names for the cell — a refusal is BY NAME (its ``kind``),
and a cell whose tier word names the provider's STOCK statement (no cell for the class on this card / jax line, or the cell names the stock op)
keeps the STOCK body here, counted by name (``stock_by_name``), (c) reads the module's own parameters under their STOCK names and dtypes
through reader sub-modules reproducing the stock scopes (``left_norm_input / projection / gate / center_norm / output_projection /
gating_linear``; zero new parameter names: a checkpoint is unchanged) into the face's math layout (``serve.TRIMUL_KEYS`` + ``TRIMUL_BIAS_KEYS``:
left = columns ``[:C]`` of projection / gate, right = ``[C:]`` — the stock split) and (d) calls the face with the resolved selection at the
model's own N. A call outside the envelope or refused runs the STOCK body and is counted under its reason, never silently.

Numerics (the provider's class for the served row, this kit's Tier 2): every product keeps stock's
operand class (bf16 operands / f32 accumulation under the multimer bf16 getter — ``precision=bf16`` on the LEVER line; an f32 pair stack would
print ``tf32``), LayerNorm statistics in f32, sigmoids in f32 through tanh.approx; fewer rounding points than stock — never bitwise vs stock,
deterministic run to run (no atomics).

Requirement (``require``): jax with the Pallas Triton lowering and its compiler-params API (jax >= 0.5; the kit's pinned 0.5.3 has it) on the
GPU backend, a compute capability >= 8.0 part (bf16 tensor-core MMA), and the provider's call face importable and
answering the tier word for the model's main cell — otherwise ``Refusal`` by name, which the activation turns into NOT ACTIVE (exit 3), never
a stock-bodied `fast`. The core's version floor is the kit's pin (``pyproject.toml`` ``[tool.opt_core] version``, gated before any import:
``_core_gate``).

Switches: ``MODEL_OPT_LEVERS_OFF=TRIMUL_PALLAS`` removes the lever from the mode (ablation.py: nothing rebound, the stock class runs);
``MODEL_OPT_LEVERS_OFF=pallas:<row>`` switches one provider ROW off by name inside the face (``levers_off``: the tier's next arm of
the cell serves) and ``MODEL_OPT_LEVERS_OFF=pallas`` every non-stock row (every class keeps the stock body: ``stock_by_name``).

Evidence: ``_STATE`` — ``calls`` (call sites a provider row served, counted as haiku traces them), ``fallbacks``, ``fallback_by``
(``reason:count,…``), ``word`` (the tier word asked), ``rows`` (``<class>:<arm>:<count>,…`` — the provider arm that SERVED each shape class,
``C<cz>E<0|1>``: E0 = outgoing ``ikc,jkc->ijc``, E1 = incoming; ``stock`` where the class kept the stock body), ``cells`` (``<class>:<N
bucket of the deciding cell | none>,…``), ``shapes`` (``N<n>xC<cz>xE<0|1>:count,…``), ``precision``, ``cc``, ``core`` (the opt_core version
that served), ``jax`` — blank-free, the LEVER line's grammar; printed at exit, recorded in the manifest (``lever_states_exit.TRIMUL_PALLAS``).
The provider's own census (``[opt_core] CELLS …`` and its ``CELL_HIT`` / ``UNCOVERED_CELL`` / ``NAMED_FALLBACK`` tokens) names every decided
cell in the same stream. Marker: ``modules.TriangleMultiplication._trimul_pallas``. Under ``big`` at ``--n_gpu`` P > 1 the row-sharded pair
stack owns the class: the activation drops this lever from the process's set there and its LEVER line says ``state=off reason=n_gpu>1``.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Optional, Tuple

NAME = "TRIMUL_PALLAS"
STRATEGY = "F2.trimul"
MODULES = "alphafold.model.modules"
CLASS = "TriangleMultiplication"
MARKER = "_trimul_pallas"
PROVIDER = "opt_core.kernels.pallas"              # the shared core's JAX-family provider: Refusal / STOCK_ROWS / TIER_WORDS / family
SERVE = PROVIDER + ".serve"                       # its call face: resolve(op, family, dtype, n_tokens, word=, direction=) / triangle_multiplication(act, mask, params, equation=, form=, unit=,
                                                  # word=, direction=, selection=) / COUNTS (tests point SERVE at a stub)
OP = "trimul"
FORM = "af2"
FACE = "triangle_multiplication"
DIRECTION = "fwd"                                 # inference: the module is never differentiated here
IMPL = SERVE + ":" + FACE                         # the LEVER line's impl= token
WORD_BY_MODE = {"fast": "fast", "big": "big", "exact": "exact"}   # the provider's TIER word for each mode (opt_core.kernels.pallas TIER_WORDS); the exact table does not name this lever
                                                  # (the provider's exact rule: no triangle-multiplication arm is vouched bitwise on this stack — the stock op by name), the word is here for completeness
DEFAULT_WORD = "fast"
EQ_OUT, EQ_IN = "ikc,jkc->ijc", "kjc,kic->ijc"    # the two equations the face serves (serve.EQUATIONS, repeated here so this module imports no jax)
UNFUSED_LAYOUT = "unfused_layout"                 # fallback names (blank-free tokens of fallback_by=)
SHAPE = "shape"
CHANNELS = "channels"
DTYPE = "dtype"
EQUATION = "equation"
MASK_SHAPE = "mask_shape"
STOCK_BY_NAME = "stock_by_name"                   # the tier word resolved to the provider's STOCK statement for the class (no cell on this card / jax line, or the cell names the stock op):
                                                  # the stock body runs here, by name (registry.STEP_ASIDE_RULES: by design, not a partial activation)
STOCK_ARM = "stock"                               # the rows= token of such a class
N_GPU_REASON = "n_gpu>1"                          # the LEVER line's reason when the row-sharded pair stack owns the class (big, P > 1)
MIN_CC = (8, 0)                                   # bf16 tensor-core MMA + tanh.approx.f32: an sm_80-class part or newer
MAIN_CELL = ("bf16", 128, 128, EQ_OUT, 512)       # (dtype, C_z, C, equation, N) of the model's main call class: require() builds its family word on the face
PARAM_SCOPES = ("left_norm_input", "projection", "gate", "center_norm", "output_projection", "gating_linear")   # the stock sub-module scopes (modules.py:1471-1512)

_STATE: Dict[str, Any] = {"enabled": False, "calls": 0, "fallbacks": 0, "fallback_by": "none", "word": DEFAULT_WORD, "rows": "none", "cells": "none", "shapes": "none",
                          "precision": "none", "cc": None, "core": None, "jax": None}
_SELECT: Dict[Tuple, Tuple[Any, Optional[str]]] = {}   # (cc, dtype, C_z, C, equation, N) -> (Selection | None, fallback reason | None) — the provider asked once per call class
_COUNTS: Dict[str, Any] = {"fallback_by": {}, "shapes": {}, "rows": {}, "cells": {}, "precision": set()}   # the dict-typed bookkeeping behind _STATE's blank-free renderings
_ORIG: Dict[str, Any] = {"cls": None, "modules": None}


class Refusal(RuntimeError):
    """The lever cannot run here (no Pallas-Triton lowering on this jax, no GPU backend, a part below compute capability 8.0, the provider's face not importable or
    refusing the word): the mode refuses by name, never degrades."""

    def __init__(self, kind: str, detail: str = ""):
        self.kind, self.detail = kind, detail
        super().__init__(f"{NAME}: {kind}" + (f" — {detail}" if detail else ""))


def _render(d: Dict) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))) or "none"


def _cc_tuple(cc) -> Optional[tuple]:
    try:
        major, _, minor = str(cc).partition(".")
        return int(major), int(minor or 0)
    except (TypeError, ValueError):
        return None


def compute_capability() -> Optional[str]:
    """The first jax GPU device's compute capability ("9.0", "8.0", …), None when jax or a GPU device is not there."""
    try:
        import jax
        for d in jax.devices():
            cc = getattr(d, "compute_capability", None)
            if cc:
                return str(cc)
    except Exception:  # noqa: BLE001 - a label / gate input; the caller names the absence
        return None
    return None


def word_for(mode: Optional[str]) -> str:
    """The provider tier word a kit mode asks (``fast`` -> fast, ``big`` -> big, ``exact`` -> exact; anything else -> fast)."""
    return WORD_BY_MODE.get(str(mode), DEFAULT_WORD)


def class_word(cz: int, equation: str) -> str:
    """``C<cz>E<0|1>``: the shape-class word of the rows= / cells= census (E0 outgoing, E1 incoming)."""
    return f"C{int(cz)}E{0 if equation == EQ_OUT else 1}"


def _unit(cz: int) -> str:
    """The provider's family unit word for the site: the template pair stack runs at c_z 64 (``tmpl``), the Evoformer / extra-MSA stacks at 128 (``pair``)."""
    return "tmpl" if int(cz) <= 64 else "pair"


def _bucket(sel) -> str:
    """The N bucket of the cell that decided a selection (``N<=800``), ``none`` without a cell."""
    key = getattr(sel, "cell_key", None)
    if not key:
        return "none"
    try:
        return str(key).split("|")[5]
    except IndexError:
        return "cell"


def require(word: str = DEFAULT_WORD) -> Dict[str, Any]:
    """The floor: jax with the Pallas Triton lowering and its compiler-params API, the GPU backend, a compute-capability >= 8.0 device, the provider's call face importable
    and ``word`` one of its tier words. Raises ``Refusal`` (kind = no_pallas_triton | no_compiler_params | not_gpu_backend | cc_below_8_0 | no_provider_face | unknown_word)
    otherwise."""
    try:
        import jax
        from jax.experimental import pallas as pl  # noqa: F401
        from jax.experimental.pallas import triton as plgpu
    except Exception as e:  # noqa: BLE001
        raise Refusal("no_pallas_triton", f"jax Pallas/Triton is not importable here ({type(e).__name__}: {e})") from None
    if not (hasattr(plgpu, "TritonCompilerParams") or hasattr(plgpu, "CompilerParams")):
        raise Refusal("no_compiler_params", "this jax has no Pallas Triton compiler-params API (the Pallas rows need jax >= 0.5)")
    if jax.default_backend() != "gpu":
        raise Refusal("not_gpu_backend", f"the default jax backend is {jax.default_backend()!r}, the kernels need the GPU backend")
    cc = compute_capability()
    t = _cc_tuple(cc)
    if t is not None and t < MIN_CC:
        raise Refusal("cc_below_8_0", f"compute capability {cc}: the bf16 Pallas rows need an sm_80-class part or newer")
    try:
        P = importlib.import_module(PROVIDER)
        S = importlib.import_module(SERVE)
        getattr(S, FACE), getattr(S, "resolve")
    except Exception as e:  # noqa: BLE001
        raise Refusal("no_provider_face", f"{SERVE} is not importable ({type(e).__name__}: {e})") from None
    if word not in tuple(getattr(P, "TIER_WORDS", (word,))):               # the face is asked a TIER word (fast | exact | big), never a row: an unknown word is refused by name before any model is traced
        raise Refusal("unknown_word", f"{word!r} is not a provider tier word ({', '.join(getattr(P, 'TIER_WORDS', ()))})")
    _, cz, c, eq, _n = MAIN_CELL
    _family(P, cz, c, eq)                                                 # the family word of the model's main cell builds on this face
    core = getattr(importlib.import_module("opt_core"), "__version__", None)
    return {"jax": jax.__version__, "backend": "gpu", "cc": cc, "core": core}


def _family(P, cz: int, c: int, equation: str) -> str:
    """The provider's trimul family word for AF2's module at (C_z, C, equation): ``af2_<pair|tmpl>_c<Cz>_ch<C>_<outgoing|incoming>``."""
    try:
        return P.family(OP, form=FORM, unit=_unit(cz), c=int(cz), c_hidden=int(c), equation=equation)
    except Exception:  # noqa: BLE001 — an older face keyword set: the literal family word (PALLAS_CELLS 'families' trimul)
        return f"{FORM}_{_unit(cz)}_c{int(cz)}_ch{int(c)}_{'outgoing' if equation == EQ_OUT else 'incoming'}"


def _selection(S, P, cc: Optional[str], dtype, cz: int, c: int, equation: str, n: int) -> Tuple[Any, Optional[str]]:
    """(Selection, None) when the tier word names a provider row for this call's cell; (Selection | None, reason) when the class keeps the stock body: the provider's
    refusal kind, or ``stock_by_name`` (the word resolved to the stock statement). Asked once per (cc, dtype, C_z, C, equation, N)."""
    key = (cc, _dtype_word(dtype), int(cz), int(c), equation, int(n))
    if key in _SELECT:
        return _SELECT[key]
    sel, why = None, None
    try:
        sel = S.resolve(OP, _family(P, cz, c, equation), _dtype_word(dtype), int(n), word=_STATE["word"], direction=DIRECTION, cc=cc)
        if getattr(sel, "row", None) in tuple(getattr(P, "STOCK_ROWS", ("xla",))) + ("xla",):
            why = STOCK_BY_NAME
    except getattr(P, "Refusal", ()) as e:
        why = str(getattr(e, "kind", None) or "refused")
    _SELECT[key] = (sel, why)
    cls = class_word(cz, equation)
    _COUNTS["cells"][cls] = _bucket(sel)
    _STATE["cells"] = _render(_COUNTS["cells"])
    return sel, why


def _count_fallback(reason: str, cz: Optional[int] = None, equation: Optional[str] = None) -> None:
    _STATE["fallbacks"] += 1
    fb = _COUNTS["fallback_by"]
    fb[reason] = fb.get(reason, 0) + 1
    _STATE["fallback_by"] = _render(fb)
    if reason == STOCK_BY_NAME and cz is not None and equation is not None:
        _count_row(class_word(cz, equation), STOCK_ARM)


def _count_row(cls: str, arm: str) -> None:
    rows = _COUNTS["rows"]
    k = f"{cls}:{arm}"
    rows[k] = rows.get(k, 0) + 1
    _STATE["rows"] = _render(rows)


def _count_served(n: int, cz: int, equation: str, dtype_name: str, arm: str) -> None:
    _STATE["calls"] += 1
    key = f"N{int(n)}xC{int(cz)}xE{0 if equation == EQ_OUT else 1}"
    sh = _COUNTS["shapes"]
    sh[key] = sh.get(key, 0) + 1
    _STATE["shapes"] = _render(sh)
    _COUNTS["precision"].add("bf16" if dtype_name == "bfloat16" else ("tf32" if dtype_name == "float32" else dtype_name))
    _STATE["precision"] = "+".join(sorted(_COUNTS["precision"]))
    _count_row(class_word(cz, equation), arm)


def _dtype_name(dt) -> str:
    return str(getattr(dt, "name", dt))


def _dtype_word(dt) -> str:
    return {"bfloat16": "bf16", "float32": "f32"}.get(_dtype_name(dt), _dtype_name(dt))


def _eligible(config, left_act, left_mask) -> Optional[str]:
    """None when the face's rows serve this call, else the fallback reason word (layout, rank / square, channels, dtype, equation, the mask's shape)."""
    if not getattr(config, "fuse_projection_weights", False):
        return UNFUSED_LAYOUT
    shape = tuple(int(x) for x in left_act.shape)
    if len(shape) != 3 or shape[0] != shape[1]:
        return SHAPE
    cz, ci = shape[-1], int(config.num_intermediate_channel)
    pow2 = lambda v: v >= 16 and (v & (v - 1)) == 0  # noqa: E731
    if not (pow2(cz) and pow2(ci)):
        return CHANNELS
    if _dtype_name(left_act.dtype) not in ("bfloat16", "float32"):
        return DTYPE
    if str(config.equation) not in (EQ_OUT, EQ_IN):
        return EQUATION
    if tuple(int(x) for x in left_mask.shape) != shape[:2]:
        return MASK_SHAPE
    return None


def _served_arm(S, before: Dict[str, int], sel) -> str:
    """The provider arm that served the call just made: the face's ``served:<face>:<arm>`` counter that moved, else the selection's own arm."""
    counts = getattr(S, "COUNTS", None) or {}
    pre = f"served:{FACE}:"
    moved = [k[len(pre):] for k, v in counts.items() if k.startswith(pre) and v > before.get(k, 0)]
    return moved[0] if moved else str(getattr(sel, "arm", None) or getattr(sel, "row", None) or "served")


def _build(stock_cls, S, P):
    import haiku as hk

    def zeros(shape, dtype):
        import jax.numpy as jnp
        return jnp.zeros(shape, dtype)

    def ones(shape, dtype):
        import jax.numpy as jnp
        return jnp.ones(shape, dtype)

    class _Params(hk.Module):
        """A reader for one stock sub-module scope: ``_Params(name='projection')('weights', shape, dtype, init)`` is the parameter
        ``<module>/projection/weights`` — the stock Linear's / LayerNorm's own name, shape and dtype."""

        def __call__(self, pname, shape, dtype, init):
            return hk.get_parameter(pname, shape, dtype, init=init)

    class TriangleMultiplication(stock_cls):
        def __call__(self, left_act, left_mask, is_training=True):
            c = self.config
            reason = _eligible(c, left_act, left_mask)
            sel = None
            if reason is None:                                            # the provider's decision for this call class (cc, dtype, C_z, C, equation, N): a row, the stock statement by name, or a refusal kind
                sel, reason = _selection(S, P, _STATE["cc"], left_act.dtype, int(left_act.shape[-1]), int(c.num_intermediate_channel), str(c.equation), int(left_act.shape[0]))
            if reason is not None:
                _count_fallback(reason, int(left_act.shape[-1]), str(c.equation))
                return super().__call__(left_act, left_mask, is_training=is_training)
            import jax.numpy as jnp
            f32, dt = jnp.float32, left_act.dtype
            CZ, C = int(left_act.shape[-1]), int(c.num_intermediate_channel)
            ln, pj, gt, cn, op, gl = (_Params(name=s) for s in PARAM_SCOPES)
            w_proj, b_proj = pj("weights", (CZ, 2 * C), dt, zeros), pj("bias", (2 * C,), dt, zeros)
            w_gate, b_gate = gt("weights", (CZ, 2 * C), dt, zeros), gt("bias", (2 * C,), dt, ones)
            params = {                                                    # the face's math layout (serve.TRIMUL_KEYS + TRIMUL_BIAS_KEYS): left = columns [:C], right = [C:] of the stock fused projection / gate
                "ln_in_scale": ln("scale", (CZ,), f32, ones), "ln_in_offset": ln("offset", (CZ,), f32, zeros),
                "left_w": w_proj[:, :C], "right_w": w_proj[:, C:], "left_gate_w": w_gate[:, :C], "right_gate_w": w_gate[:, C:],
                "ln_c_scale": cn("scale", (C,), f32, ones), "ln_c_offset": cn("offset", (C,), f32, zeros),
                "out_w": op("weights", (C, CZ), dt, zeros), "gate_w": gl("weights", (CZ, CZ), dt, zeros),
                "left_b": b_proj[:C], "right_b": b_proj[C:], "left_gate_b": b_gate[:C], "right_gate_b": b_gate[C:],
                "out_b": op("bias", (CZ,), dt, zeros), "gate_b": gl("bias", (CZ,), dt, ones),
            }
            before = {k: v for k, v in (getattr(S, "COUNTS", None) or {}).items() if k.startswith(f"served:{FACE}:")}
            try:
                out = S.triangle_multiplication(left_act, left_mask, params, equation=str(c.equation), form=FORM, unit=_unit(CZ), word=_STATE["word"], direction=DIRECTION,
                                                cc=_STATE["cc"], selection=sel)
            except getattr(P, "Refusal", ()) as e:                        # every candidate of the cell refused with the arrays in hand: the stock body, by the last refusal's name
                _count_fallback(str(getattr(e, "kind", None) or "refused"))
                return super().__call__(left_act, left_mask, is_training=is_training)
            _count_served(int(left_act.shape[0]), CZ, str(c.equation), _dtype_name(dt), _served_arm(S, before, sel))
            return out.astype(dt)

    setattr(TriangleMultiplication, MARKER, True)
    TriangleMultiplication.__qualname__ = TriangleMultiplication.__name__ = CLASS
    return TriangleMultiplication


def enable(modules=None, mode: Optional[str] = None, word: Optional[str] = None) -> Dict[str, Any]:
    """Require the floor (``require``: a ``Refusal`` by name propagates — the activation's NOT ACTIVE), import the provider's face, rebind
    ``modules.TriangleMultiplication`` to the served subclass (idempotent). ``mode``: the kit mode whose tier word the face is asked
    (``word_for``); ``word`` overrides it. ``modules``: the target module (tests pass a stub with a ``TriangleMultiplication`` class); None imports
    ``alphafold.model.modules``."""
    _STATE["word"] = str(word) if word else word_for(mode)
    facts = require(_STATE["word"])
    P = importlib.import_module(PROVIDER)
    S = importlib.import_module(SERVE)
    m = modules if modules is not None else importlib.import_module(MODULES)
    cur = getattr(m, CLASS)
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
    _STATE.update(enabled=False, calls=0, fallbacks=0, fallback_by="none", word=DEFAULT_WORD, rows="none", cells="none", shapes="none", precision="none", cc=None,
                  core=None, jax=None)
    _COUNTS.update(fallback_by={}, shapes={}, rows={}, cells={}, precision=set())
    _ORIG.update(cls=None, modules=None)
    _SELECT.clear()
