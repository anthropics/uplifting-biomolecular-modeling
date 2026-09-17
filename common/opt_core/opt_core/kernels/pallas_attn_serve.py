"""Pallas flash attention with per-head pair bias and key mask, JAX (family F1, lever ``F1.pallas_attn``).

What it is. The carried kernel ``pallas_attn`` (package ``opt_core/kernels/pallas_attn/``: ``af2_flash_pallas.py`` + NOTICE,
matching its kit byte-for-byte, described by ``META/pallas_attn.json``; this file is the generic layer beside it, like
``flash_triattn_serve`` beside ``flash_triattn``) is a Pallas (Triton-lowering) implementation of
softmax(q·kᵀ·scale + key-mask + pair-bias)·v that never materialises the ``[B, H, S_q, S_k]`` logits, FORWARD and BACKWARD
(``jax.custom_vjp``: separate dQ and dK/dV kernels, no atomics — run-to-run bit-exact, gradients included). Layout heads-major: q, k, v
``[B, H, S, D]`` (bf16 / f16 / f32; D >= 16, the same for q and v), pair bias ``[H, S_q, S_k]`` shared over B (the stock-form
``nonbatched_bias``), key mask ``[B, S_k]`` bool. bf16 products with fp32 accumulation and an exact online softmax: re-associated
relative to a materialised softmax → class ``fast`` (Tier 2) unless the engine's equality tests prove otherwise. This
module is generic core code around the carried bytes — its import surface is standard library plus ``opt_core.report``; jax and haiku
are imported inside the functions that need them and their absence, or a jax below the floor, is a NAMED :class:`Refusal`.

Public API (nothing here names an engine; a kit adapter is ≤ 10 lines: its module object(s), its size rule, its tag):

* :func:`probe` / :func:`require` — ``{ok, kind, detail, jax, jaxlib, backend, tested}``; kinds ``jax_missing`` · ``jax_older_than_floor``
  (< 0.5.0: no Pallas Triton lowering of this form) · ``pallas_missing`` · ``backend_not_gpu`` · ``kernel_import_failed`` ·
  ``twin_loaded`` (a loose ``af2_flash_pallas`` module from another directory is already imported — an adopting kit does not import
  its own copy). ``tested`` is True on the jax versions of the kit's test record (:data:`JAX_TESTED`); an untested newer jax is
  NOT refused — it compiles or raises at the first call, loudly either way.
* :func:`kernel_module` — the carried module ``pallas_attn.af2_flash_pallas`` of this process (the routed top-level ``pallas_attn`` when
  the kit called ``opt_core.kernels.route("pallas_attn")``, else ``opt_core.kernels.pallas_attn``): ``make_flash_attention``,
  ``flash_attention``, ``attention_af2``, ``reference_attention``; :func:`kernel_origin` → ``core`` | ``kit``; :func:`kernel_impl` →
  ``pallas_attn@<sha8>``.
* :func:`make_flash_attention` / :func:`flash_attention` / :func:`attention_core` / :func:`reference_attention` — thin pass-throughs;
  ``attention_core(q, k, v, mask_bias, nonbatched_bias, scale)`` is the drop-in for the einsum+softmax core of a stock-form
  attention module (``mask_bias`` additive ``[B, 1, 1, S_k]`` with -1e9 = masked; ``nonbatched_bias`` ``[H, S_q, S_k]`` or None).
* :func:`ineligible` — the per-call eligibility rule as ONE function returning a reason name or None (the probe's kind when the kernel
  cannot serve at all · ``backend_not_gpu`` · ``no_pair_bias`` · ``key_dim_ne_value_dim`` · ``head_dim_lt_16`` · ``bias_form`` ·
  ``below_size_rule``).
* :func:`enable` / :func:`disable` — rebind the haiku ``Attention`` class of the module objects a kit names (a stock-form
  ``Attention(config, global_config, output_dim)`` whose ``__call__(q_data, m_data, bias, nonbatched_bias=None)`` computes
  softmax(q·kᵀ/√d + bias + nonbatched_bias)·v with optional gating): eligible calls go through the Pallas op, the rest through the stock
  ``__call__`` with a NAMED fallback in the ledger. Projections, gating and output projection are the stock parameters and einsums; the
  replacement class keeps the class name so haiku's parameter tree is unchanged. Call BEFORE the model function is traced / jitted.
* :func:`ledger` — this lever's ``opt_core.counters.Ledger`` (served / fallback-by-reason counters, shapes, fail-closed ``gate()``) and
  :func:`emit_line` — the ONE per-lever activation-evidence line (``[<tag>] LEVER name=F1.pallas_attn state=<on|skipped|off> [reason=…]
  impl=pallas_attn@… origin=<core|kit> served=… fallback=… fallback_by=…``, ``opt_core.report.lever_line``).

Kit usage (JAX engine; this IS the whole adapter)::

    from opt_core.kernels import pallas_attn_serve as F1
    from <the engine's package>.model import modules            # the module object that defines the haiku `Attention` class
    F1.require()                                                # named Refusal on an old jax / CPU backend (→ the kit's NOT ACTIVE line)
    LEDGER = F1.ledger(min_tokens=KIT_MIN_TOKENS)               # the kit's size rule, cited in its mode table
    F1.enable([modules], ledger=LEDGER, all_calls=False)        # before jit; pair-bias calls only (triangle + MSA-row-with-pair-bias)
    ... run ...;  F1.emit_line(LEDGER, tag=KIT_TAG);  LEDGER.gate()     # once per arm per process; the gate feeds the kit's exit rule

Backward knobs of the carried kernel (environment, read when an op is built): ``AF_PALLAS_ATTN_PRECISE_BWD``, ``AF_PALLAS_ATTN_DBIAS_F32``,
``AF_PALLAS_ATTN_BWD_BATCH_CHUNK`` — see ``make_flash_attention`` in the carried file; forward numerics do not depend on them; pass an
``op=make_flash_attention(...)`` to :func:`enable` to choose them explicitly.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..counters import Ledger
from ..report import emit

NAME = "pallas_attn"                                 # the carried-kernel name (META/pallas_attn.json) = the LEVER line's impl=
FAMILY = "F1"
LEVER = "F1.pallas_attn"                             # the LEVER line's name= (the strategy id of this lever)
CLASS_OF_RECORD = "fast"                             # Tier 2 wherever measured (bf16 re-association); never 'exact' without a bit-exact proof
KERNEL_FILE = "af2_flash_pallas"                     # the carried module inside the pallas_attn package
JAX_FLOOR: Tuple[int, int, int] = (0, 5, 0)          # below: no Pallas-Triton lowering of the form the kernel uses → Refusal jax_older_than_floor
JAX_TESTED: Tuple[str, ...] = ("0.5.3", "0.6.0")     # the jax versions of the add-on's test record
MIN_HEAD_DIM = 16                                    # Triton dot operands need >= 16; a smaller head dim is refused (head_dim_lt_16) or zero-padded to it (pad_head_dim)
MASKED_BIAS_THRESHOLD = -1e8                         # stock mask bias = 1e9 * (mask - 1): key j is masked where bias < -1e8

# fallback reason names (Ledger keys)
BACKEND_NOT_GPU = "backend_not_gpu"
NO_PAIR_BIAS = "no_pair_bias"
KEY_DIM_NE_VALUE_DIM = "key_dim_ne_value_dim"
HEAD_DIM_LT_16 = "head_dim_lt_16"
PALLAS_API_REMOVED = "pallas_api_removed"             # the running jax lacks the Pallas functions the carried kernel calls
PALLAS_API = ("pallas_call", "BlockSpec")           # jax.experimental.pallas names the carried kernel uses
PALLAS_IO_API = ("load", "store")                    # the masked load/store the kernel uses: jax.experimental.pallas (jax 0.5-0.7) OR jax.experimental.pallas.triton on a ref view (newer)
BELOW_KEYS_RULE = "below_keys_rule"                   # a kit's per-call floor on N_keys (all_calls serving of key-poor calls left to XLA)
PADDED_HEAD_DIM_SUFFIX = "p16"                        # census-key suffix of a call whose head dim was zero-padded to MIN_HEAD_DIM
BIAS_FORM = "bias_form"
BELOW_SIZE_RULE = "below_size_rule"
KERNEL_UNAVAILABLE = "kernel_unavailable"


class Refusal(RuntimeError):
    """A named refusal (``kind``, ``detail``): jax_missing · jax_older_than_floor · pallas_missing · backend_not_gpu · kernel_import_failed ·
    twin_loaded · haiku_missing · not_an_attention_module."""

    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{NAME}: {kind}" + (f" — {detail}" if detail else ""))


def _vtuple(v: str) -> Tuple[int, ...]:
    out = []
    for part in str(v).split("+", 1)[0].split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out) or (0,)


# ----------------------------------------------------------------------------------------------------------------- availability
_PROBE: Optional[Dict[str, Any]] = None
_KMOD = None
_LOCK = threading.Lock()


def probe(refresh: bool = False, require_gpu: bool = True) -> Dict[str, Any]:
    """``{ok, kind, detail, jax, jaxlib, backend, tested}`` — never raises. ``require_gpu=False`` lets a CPU process load the kernel
    module (e.g. for its reference function); the Pallas op itself only lowers on the GPU backend."""
    global _PROBE
    if _PROBE is not None and not refresh and _PROBE.get("require_gpu") == require_gpu:
        return dict(_PROBE)
    out: Dict[str, Any] = {"ok": False, "kind": None, "detail": "", "jax": None, "jaxlib": None, "backend": None, "tested": False,
                           "require_gpu": require_gpu}

    def done() -> Dict[str, Any]:
        global _PROBE
        _PROBE = out
        return dict(out)

    try:
        import jax  # noqa: WPS433 (lazy by contract)
        out["jax"] = getattr(jax, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        out.update(kind="jax_missing", detail=repr(e)[:200])
        return done()
    try:
        import jaxlib  # noqa: WPS433
        out["jaxlib"] = getattr(jaxlib, "__version__", None) or getattr(getattr(jaxlib, "version", None), "__version__", "?")
    except Exception:  # noqa: BLE001
        out["jaxlib"] = None
    if _vtuple(out["jax"]) < JAX_FLOOR:
        out.update(kind="jax_older_than_floor", detail=f"jax {out['jax']} < {'.'.join(map(str, JAX_FLOOR))} (Pallas Triton lowering of this form absent)")
        return done()
    out["tested"] = out["jax"] in JAX_TESTED
    try:
        from jax.experimental import pallas as _pl  # noqa: F401
        from jax.experimental.pallas import triton as _plgpu  # noqa: F401
    except Exception as e:  # noqa: BLE001
        out.update(kind="pallas_missing", detail=repr(e)[:200])
        return done()
    missing = [n for n in PALLAS_API if not hasattr(_pl, n)] + [n for n in PALLAS_IO_API if not (hasattr(_pl, n) or hasattr(_plgpu, n))]
    if missing:                                                        # a jax whose Pallas lacks a function the kernel calls under either home
        out.update(kind=PALLAS_API_REMOVED, detail=f"jax {out['jax']}: jax.experimental.pallas lacks {missing} (the carried kernel's API; tested on {sorted(JAX_TESTED)})")
        return done()
    try:
        out["backend"] = jax.default_backend()
    except Exception as e:  # noqa: BLE001
        out["backend"] = f"unknown:{e!r}"[:60]
    if require_gpu and out["backend"] != "gpu":
        out.update(kind=BACKEND_NOT_GPU, detail=f"jax.default_backend() = {out['backend']!r}; the Pallas op lowers on gpu only")
        return done()
    try:
        _load_kernel()
    except Refusal as r:
        out.update(kind=r.kind, detail=r.detail)
        return done()
    out["ok"] = True
    return done()


def require(require_gpu: bool = True) -> Dict[str, Any]:
    """The probe, or a :class:`Refusal` naming why the kernel cannot serve in this process."""
    p = probe(require_gpu=require_gpu)
    if not p["ok"]:
        raise Refusal(p["kind"] or "kernel_import_failed", p["detail"])
    return p


def _core_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), NAME)


def _routed_names() -> List[str]:
    from . import routed                                                            # opt_core.kernels.routed (stdlib-only)
    return list(routed())


def _load_kernel():
    """The carried kernel module: ``pallas_attn.af2_flash_pallas`` when the kit routed / imported the top-level name, else the core
    copy. A loose top-level ``af2_flash_pallas`` already imported from ANOTHER directory is refused by name (``twin_loaded``): one
    kernel per process — an adopting kit does not import its own copy."""
    global _KMOD
    if _KMOD is not None:
        return _KMOD
    with _LOCK:
        if _KMOD is not None:
            return _KMOD
        loose = sys.modules.get(KERNEL_FILE)
        if loose is not None:
            where = os.path.dirname(os.path.abspath(getattr(loose, "__file__", "") or ""))
            if where != os.path.abspath(_core_dir()):
                raise Refusal("twin_loaded", f"a module named {KERNEL_FILE} is already imported from {where}; route/import {NAME} only")
        try:
            pkg_name = NAME if (NAME in sys.modules or NAME in _routed_names()) else __package__ + "." + NAME
            mod = importlib.import_module(pkg_name + "." + KERNEL_FILE)
        except Refusal:
            raise
        except Exception as e:  # noqa: BLE001
            raise Refusal("kernel_import_failed", repr(e)[:300])
        for fn in ("make_flash_attention", "flash_attention", "attention_af2", "reference_attention"):
            if not hasattr(mod, fn):
                raise Refusal("kernel_import_failed", f"{getattr(mod, '__file__', '?')} lacks {fn}")
        _KMOD = mod
        return mod


def kernel_module(require_gpu: bool = False):
    """The carried kernel module (imports jax). ``Refusal`` when jax is missing / too old / pallas missing / a twin is loaded (and, with
    ``require_gpu=True``, when the backend is not gpu)."""
    require(require_gpu=require_gpu)
    return _load_kernel()


def kernel_origin() -> str:
    """``core`` when the kernel module in use is the core copy (routed by name or imported as opt_core.kernels.pallas_attn), ``kit``
    for a kit's own copy, ``unavailable:<kind>`` when it cannot be imported."""
    try:
        mod = kernel_module(require_gpu=False)
    except Refusal as r:
        return f"unavailable:{r.kind}"
    here = os.path.dirname(os.path.abspath(getattr(mod, "__file__", "") or ""))
    return "core" if here == os.path.abspath(_core_dir()) else "kit"


def kernel_impl() -> str:
    """``pallas_attn@<sha256[:8] of the kernel file in use>`` — the LEVER line's ``impl=``."""
    try:
        with open(getattr(kernel_module(require_gpu=False), "__file__"), "rb") as fh:
            return f"{NAME}@{hashlib.sha256(fh.read()).hexdigest()[:8]}"
    except Exception:  # noqa: BLE001
        return f"{NAME}@unavailable"


# --------------------------------------------------------------------------------------------------------------- the operations
def dq_modes() -> tuple:
    """How the carried kernel's backward may form dQ (`make_flash_attention(dq=)`; its own DQ_MODES), default first: ('kernel', 'from_ds')."""
    K = kernel_module(require_gpu=False)
    d = K.DQ_MODE_DEFAULT
    return (d,) + tuple(w for w in K.DQ_MODES if w != d)


def dbias_modes() -> tuple:
    """How the carried kernel's backward may form d(pair bias) (`make_flash_attention(dbias=)`; its own DBIAS_MODES), default first:
    ('xla', 'kernel') — per-batch dS partials reduced by XLA, or summed over the batch inside a third kernel (no partials buffer)."""
    K = kernel_module(require_gpu=False)
    d = K.DBIAS_MODE_DEFAULT
    return (d,) + tuple(w for w in K.DBIAS_MODES if w != d)


def bwd_f32_precisions() -> tuple:
    """The backward float32 product classes the carried kernel offers (`make_flash_attention(bwd_f32_precision=)`; its own
    BWD_F32_PRECISIONS), default first: ('ieee', 'tf32'). A kit that opts into tf32 records bwd_precision=<word> on its LEVER line."""
    K = kernel_module(require_gpu=False)
    d = K.F32_PRECISION_DEFAULT
    return (d,) + tuple(w for w in K.BWD_F32_PRECISIONS if w != d)


def f32_precisions() -> tuple:
    """The forward float32 product classes the carried kernel offers (`make_flash_attention(f32_precision=)`; its own F32_PRECISIONS) and its
    default first: e.g. ('ieee', 'tf32', 'bf16').  A kit records precision=<word> on its LEVER line; `emit_line` stamps it itself."""
    K = kernel_module(require_gpu=False)
    d = K.F32_PRECISION_DEFAULT
    return (d,) + tuple(p for p in K.F32_PRECISIONS if p != d)


def __getattr__(name):                                   # `pallas_attn_serve.F32_PRECISIONS` = the carried kernel's tuple (one source of truth)
    if name == "F32_PRECISIONS":
        return kernel_module(require_gpu=False).F32_PRECISIONS
    raise AttributeError(name)


def describe_op(op) -> dict:
    """What `make_flash_attention` built for `op` (f32_precision, bwd_f32_precision, dq, dbias, bq, bk, num_warps, num_stages, bq_bwd, bk_bwd, batch_chunk)
    — the kit's LEVER-line evidence."""
    return kernel_module(require_gpu=False).describe_op(op)


def make_flash_attention(**knobs):
    """Build the differentiable op ``op(q, k, v, pair_bias, key_mask, scale) -> [B, H, S_q, D]`` with the carried kernel's knobs
    (``bq``, ``bk``, ``num_warps``, ``num_stages``, ``dbias_dtype``, ``bq_bwd``, ``bk_bwd``, ``batch_chunk``, ``precise_bwd``)."""
    return kernel_module().make_flash_attention(**knobs)


def flash_attention(q, k, v, pair_bias, key_mask, scale):
    """The default op: q, k, v ``[B, H, S, D]``; pair_bias ``[H, S_q, S_k]`` in q's dtype; key_mask ``[B, S_k]`` bool; scale float."""
    return kernel_module().flash_attention(q, k, v, pair_bias, key_mask, float(scale))


def pad_head_dim(q, k, v):
    """``(q, k, v, d_v)``: q/k/v zero-padded on the last (head) dim up to ``MIN_HEAD_DIM`` when smaller (a no-op otherwise) and the original
    value head dim to slice the output back to. Exact: zero columns add nothing to q·k, and the padded output columns are dropped; the
    softmax scale is always passed explicitly from the ORIGINAL key dim."""
    import jax.numpy as jnp  # noqa: WPS433
    dq, dv = int(q.shape[-1]), int(v.shape[-1])
    if dq >= MIN_HEAD_DIM and dv >= MIN_HEAD_DIM:
        return q, k, v, dv

    def pad(x):
        p = MIN_HEAD_DIM - int(x.shape[-1])
        return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, p)]) if p > 0 else x
    return pad(q), pad(k), pad(v), dv


def attention_core(q, k, v, mask_bias, nonbatched_bias, scale, impl=None, pad_head_dim_below_min: bool = False):
    """Drop-in for the einsum+softmax core of a stock-form attention: q, k, v ``[B, H, S, D]`` heads-major; ``mask_bias`` additive
    ``[B, 1, 1, S_k]`` (-1e9 = masked); ``nonbatched_bias`` ``[H, S_q, S_k]`` or None → ``[B, H, S_q, D]``. ``pad_head_dim_below_min``:
    serve ``D < MIN_HEAD_DIM`` by zero-padding (``pad_head_dim``) and slicing the output back."""
    mod = kernel_module()
    dv = int(v.shape[-1])
    if pad_head_dim_below_min:
        q, k, v, dv = pad_head_dim(q, k, v)
    out = mod.attention_af2(q, k, v, mask_bias, nonbatched_bias, scale, impl=impl if impl is not None else mod.flash_attention)
    return out if int(out.shape[-1]) == dv else out[..., :dv]


def reference_attention(q, k, v, pair_bias, key_mask, scale, precision=None):
    """The materialised XLA reference in the inputs' dtype (the stock math) — for adapters' fallbacks and for parity probes."""
    return kernel_module(require_gpu=False).reference_attention(q, k, v, pair_bias, key_mask, scale, precision=precision)


def ineligible(head_dim_q: int, head_dim_v: int, has_pair_bias: bool, mask_bias_shape: Optional[Sequence[int]] = None,
               all_calls: bool = True, n_queries: Optional[int] = None, min_tokens: int = 0, backend: Optional[str] = None,
               n_keys: Optional[int] = None, min_keys: int = 0, pad_head_dim: bool = False) -> Optional[str]:
    """The per-call eligibility rule as ONE function: None when the call may be served, else the reason name. When the kernel cannot
    serve in this process at all the reason is the probe's kind (``jax_older_than_floor``, ``jax_missing``, …), not a shape reason.
    ``backend`` defaults to the probed jax backend; ``all_calls=False`` restricts serving to calls that carry a pair bias;
    ``n_queries``/``min_tokens`` apply a kit's per-call size rule (``below_size_rule``); ``n_keys``/``min_keys`` a floor on the number of
    keys (``below_keys_rule`` — e.g. an MSA column attention over few sequences stays on XLA while the same module over many sequences is
    served); ``pad_head_dim`` lets a head dim below ``MIN_HEAD_DIM`` be served by zero-padding q/k/v to it (``head_dim_lt_16`` otherwise)."""
    p = probe()
    if not p["ok"] and p["kind"] not in (None, BACKEND_NOT_GPU):
        return str(p["kind"])
    if backend is None:
        backend = p.get("backend")
    if backend != "gpu":
        return BACKEND_NOT_GPU
    if not has_pair_bias and not all_calls:
        return NO_PAIR_BIAS
    if int(head_dim_q) != int(head_dim_v):
        return KEY_DIM_NE_VALUE_DIM
    if int(head_dim_q) < MIN_HEAD_DIM and not pad_head_dim:
        return HEAD_DIM_LT_16
    if min_keys and n_keys is not None and int(n_keys) < int(min_keys):
        return BELOW_KEYS_RULE
    if mask_bias_shape is not None:
        s = tuple(int(x) for x in mask_bias_shape)
        if len(s) != 4 or s[1] != 1 or s[2] != 1:
            return BIAS_FORM
    if min_tokens and n_queries is not None and int(n_queries) < int(min_tokens):
        return BELOW_SIZE_RULE
    return None


def shape_key(q) -> str:
    """``B<b>xH<h>xS<s>xD<d>`` from a heads-major ``[B, H, S, D]`` array — the served-shape census key."""
    s = tuple(int(x) for x in getattr(q, "shape", ()))
    s = (1,) * (4 - len(s)) + s if len(s) < 4 else s[-4:]
    return f"B{s[0]}xH{s[1]}xS{s[2]}xD{s[3]}"


def ledger(min_tokens: int = 0, *, expected: Tuple[str, ...] = (BELOW_SIZE_RULE, NO_PAIR_BIAS), impl: Optional[str] = None,
           origin: Optional[str] = None, **kw) -> Ledger:
    """The per-process ``opt_core.counters.Ledger`` of this lever: ``name=F1.pallas_attn``, ``impl`` (default the kernel name; :func:`kernel_impl`
    gives ``pallas_attn@<sha8>``), ``origin`` (:func:`kernel_origin`), the kit's size rule ``min_tokens`` and the fallback reasons its mode
    DECLARES (``expected``; e.g. add ``head_dim_lt_16`` when ``all_calls=True`` serves template attention)."""
    return Ledger(LEVER, impl=impl if impl is not None else NAME, origin=origin, min_tokens=int(min_tokens), expected=expected, **kw)


def emit_line(ledger: Ledger, tag: str, **evidence) -> str:
    """Print this lever's ONE activation-evidence line (``report.emit(ledger.line(tag, ...))``) with ``impl``/``origin`` resolved from the
    kernel in use when the ledger does not carry them yet. Returns the text."""
    if ledger.origin is None:
        ledger.origin = kernel_origin()
    if ledger.impl in (None, NAME):
        ledger.impl = kernel_impl()
    for key, field in (("precision", "f32_precision"), ("bwd_precision", "bwd_f32_precision"), ("dq", "dq"), ("dbias", "dbias")):
        if key not in evidence:                             # the op's own words (forward / backward float32 product class, dQ and dbias modes) — stamped here, one grammar
            evidence[key] = served_word(field, ledger)
    return emit(ledger.line(tag, **evidence))


def served_word(field: str, ledger: Optional[Ledger] = None, op: Optional[Callable] = None) -> str:
    """The `describe_op` word `field` ('f32_precision' | 'bwd_f32_precision' | 'dq' | 'dbias') of `op` (or of every op enabled with `ledger`,
    '+'-joined if they differ; the module default op when `enable(op=None)`), 'unavailable:<kind>' when the kernel cannot be imported."""
    try:
        K = kernel_module(require_gpu=False)
    except Refusal as r:
        return f"unavailable:{r.kind}"
    defaults = {"f32_precision": K.F32_PRECISION_DEFAULT, "bwd_f32_precision": K.F32_PRECISION_DEFAULT, "dq": K.DQ_MODE_DEFAULT, "dbias": K.DBIAS_MODE_DEFAULT}
    ops = [op] if op is not None else [e.get("op") for e in _ENABLED.values() if ledger is None or e.get("ledger") is ledger]
    words = []
    for o in (ops or [None]):
        o = K.flash_attention if o is None else o
        words.append(str(K.describe_op(o).get(field, defaults[field])))
    return "+".join(sorted(set(words))) if words else str(defaults[field])


def served_precision(ledger: Optional[Ledger] = None, op: Optional[Callable] = None) -> str:
    """The forward `f32_precision` word (see `served_word`)."""
    return served_word("f32_precision", ledger, op)


# ------------------------------------------------------------------------------------------- the haiku Attention-class rebinding
_ENABLED: Dict[int, Dict[str, Any]] = {}             # id(module object) -> {"module", "stock_cls", "ledger"}


def served_attention_call(self, q_data, m_data, bias, nonbatched_bias=None, *, all_calls: bool = True, ledger: Optional[Ledger] = None,
                          min_tokens: int = 0, op: Optional[Callable] = None, min_keys: int = 0, pad_head_dim_below_min: bool = False):
    """Functional form of the served path for a haiku ``Attention`` module instance ``self`` (call it from inside the module's
    ``__call__``, i.e. within its haiku name scope). Returns None — after counting a NAMED fallback in ``ledger`` — when the call is
    ineligible (:func:`ineligible`), so the caller runs the stock math; else the attention output ``[batch, N_queries, output_dim]``
    computed with the stock parameters (``query_w``, ``key_w``, ``value_w``, ``gating_w``/``gating_b``, ``output_w``, ``output_b``)."""
    import jax  # noqa: WPS433 (lazy by contract)
    import jax.numpy as jnp  # noqa: WPS433
    try:
        import haiku as hk  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        raise Refusal("haiku_missing", repr(e)[:200])
    num_head = self.config.num_head
    key_dim = self.config.get("key_dim", int(q_data.shape[-1])) // num_head
    value_dim = self.config.get("value_dim", int(m_data.shape[-1])) // num_head
    why = ineligible(key_dim, value_dim, nonbatched_bias is not None, getattr(bias, "shape", None), all_calls=all_calls,
                     n_queries=int(q_data.shape[-2]), min_tokens=min_tokens, n_keys=int(m_data.shape[-2]), min_keys=min_keys,
                     pad_head_dim=pad_head_dim_below_min)
    if why is not None:
        if ledger is not None:
            ledger.fallback(why)
        return None
    mod = kernel_module()
    flash = op if op is not None else mod.flash_attention
    glorot_uniform = lambda: hk.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")  # noqa: E731
    q_weights = hk.get_parameter("query_w", shape=(q_data.shape[-1], num_head, key_dim), dtype=q_data.dtype, init=glorot_uniform())
    k_weights = hk.get_parameter("key_w", shape=(m_data.shape[-1], num_head, key_dim), dtype=q_data.dtype, init=glorot_uniform())
    v_weights = hk.get_parameter("value_w", shape=(m_data.shape[-1], num_head, value_dim), dtype=q_data.dtype, init=glorot_uniform())
    # heads-major projections: the stock contraction 'bqa,ahc->bqhc' with a different output layout; 1/sqrt(d) is applied inside the kernel in fp32
    q = jnp.einsum("bqa,ahc->bhqc", q_data, q_weights)
    k = jnp.einsum("bka,ahc->bhkc", m_data, k_weights)
    v = jnp.einsum("bka,ahc->bhkc", m_data, v_weights)
    kmask = bias[:, 0, 0, :] > MASKED_BIAS_THRESHOLD
    sq, sk = q.shape[2], k.shape[2]
    nb = jnp.zeros((num_head, sq, sk), q.dtype) if nonbatched_bias is None else nonbatched_bias.astype(q.dtype)
    census = shape_key(q)
    if key_dim < MIN_HEAD_DIM:                                                         # pad_head_dim_below_min (ineligible() let it through)
        q, k, v, _ = pad_head_dim(q, k, v)
        census += PADDED_HEAD_DIM_SUFFIX
    weighted_avg = flash(q, k, v, nb, kmask, float(key_dim ** (-0.5)))                # [b, h, q, c]; the scale is the ORIGINAL key dim's
    if int(weighted_avg.shape[-1]) != value_dim:
        weighted_avg = weighted_avg[..., :value_dim]
    weighted_avg = jnp.swapaxes(weighted_avg, 1, 2)                                    # [b, q, h, c]
    init = hk.initializers.Constant(0.0) if self.global_config.zero_init else glorot_uniform()
    if self.config.gating:
        gating_weights = hk.get_parameter("gating_w", shape=(q_data.shape[-1], num_head, value_dim), dtype=q_data.dtype,
                                          init=hk.initializers.Constant(0.0))
        gating_bias = hk.get_parameter("gating_b", shape=(num_head, value_dim), dtype=q_data.dtype, init=hk.initializers.Constant(1.0))
        gate_values = jnp.einsum("bqc, chv->bqhv", q_data, gating_weights) + gating_bias
        weighted_avg *= jax.nn.sigmoid(gate_values)
    o_weights = hk.get_parameter("output_w", shape=(num_head, value_dim, self.output_dim), dtype=q_data.dtype, init=init)
    o_bias = hk.get_parameter("output_b", shape=(self.output_dim,), dtype=q_data.dtype, init=hk.initializers.Constant(0.0))
    if ledger is not None:
        ledger.serve(census)
    return jnp.einsum("bqhc,hco->bqo", weighted_avg, o_weights) + o_bias


def enable(modules_list: Sequence[Any], *, ledger: Optional[Ledger] = None, all_calls: bool = False, min_tokens: Optional[int] = None,
           op: Optional[Callable] = None, min_keys: int = 0, pad_head_dim_below_min: bool = False) -> List[Any]:
    """Rebind ``<module>.Attention`` in every module object of ``modules_list`` (the kit names them — nothing is imported by name here)
    so that eligible calls are served by the Pallas op and the rest run the stock ``__call__`` with a NAMED fallback counted in
    ``ledger``. ``all_calls=False`` serves only calls carrying a pair bias (triangle attention, MSA row attention with pair bias);
    ``min_tokens`` (default ``ledger.min_tokens``) is a per-call size rule on N_queries; ``min_keys`` a floor on N_keys (``below_keys_rule``:
    with ``all_calls=True`` an MSA column attention over many sequences is served while key-poor calls stay on XLA); ``pad_head_dim_below_min``
    serves head dims below ``MIN_HEAD_DIM`` by zero-padding (census key suffix ``p16``); ``op`` overrides the kernel op
    (``make_flash_attention(...)``). haiku detail: the replacement class is created through haiku's metaclass INSIDE a class body with
    the SAME class name, so the auto-derived module name and hence the parameter tree are unchanged, and ``hk.get_parameter`` runs in the
    module's own scope. Call BEFORE the model function is traced / jitted; :func:`require` first. Returns the patched module objects."""
    require()
    gate = int(min_tokens if min_tokens is not None else ((ledger.min_tokens or 0) if ledger is not None else 0))
    out = []
    for m in list(modules_list):
        A = getattr(m, "Attention", None)
        if A is None or not isinstance(A, type):
            raise Refusal("not_an_attention_module", f"{getattr(m, '__name__', m)!r} has no Attention class")
        if not getattr(A, "_opt_core_pallas_attn", False):
            m.Attention = _rebound_class(A, all_calls=all_calls, ledger=ledger, min_tokens=gate, op=op, min_keys=int(min_keys),
                                         pad_head_dim_below_min=bool(pad_head_dim_below_min))
            _ENABLED[id(m)] = {"module": m, "stock_cls": A, "ledger": ledger, "op": op}
        out.append(m)
    return out


def _rebound_class(A: type, *, all_calls: bool, ledger: Optional[Ledger], min_tokens: int, op: Optional[Callable], min_keys: int = 0,
                   pad_head_dim_below_min: bool = False) -> type:
    """A subclass of the stock haiku ``Attention`` class ``A`` with the SAME name whose ``__call__`` serves eligible calls (one class per
    module object; every closure variable is bound here, not in the caller's loop)."""
    stock_call = A.__call__

    def _served(self, q_data, m_data, bias, nonbatched_bias=None):
        res = served_attention_call(self, q_data, m_data, bias, nonbatched_bias, all_calls=all_calls, ledger=ledger,
                                    min_tokens=min_tokens, op=op, min_keys=min_keys, pad_head_dim_below_min=pad_head_dim_below_min)
        return stock_call(self, q_data, m_data, bias, nonbatched_bias) if res is None else res

    class Attention(A):                              # noqa: D101 — defined in a class body so haiku's metaclass wraps __call__
        _opt_core_pallas_attn = True
        _stock_cls = A

        def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
            return _served(self, q_data, m_data, bias, nonbatched_bias)

    Attention.__qualname__ = "Attention"
    Attention.__module__ = A.__module__
    return Attention


def disable(modules_list: Optional[Sequence[Any]] = None) -> List[Any]:
    """Restore the stock ``Attention`` class on the given (default: every enabled) module objects; re-trace / re-jit afterwards."""
    mods = list(modules_list) if modules_list is not None else [rec["module"] for rec in _ENABLED.values()]
    out = []
    for m in mods:
        A = getattr(m, "Attention", None)
        if A is not None and getattr(A, "_opt_core_pallas_attn", False):
            m.Attention = A._stock_cls
            out.append(m)
        _ENABLED.pop(id(m), None)
    return out


def enabled() -> List[Any]:
    """The module objects currently rebound by :func:`enable`."""
    return [rec["module"] for rec in _ENABLED.values()]
