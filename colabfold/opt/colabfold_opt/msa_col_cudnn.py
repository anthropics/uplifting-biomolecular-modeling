"""MSA_COL_CUDNN — MSA column attention (the bias-free ``Attention`` call class: 8 heads × 32 channels, keys = the cluster MSA depth) through
the shared core's JAX-family provider by TIER WORD with EVERY row admitted (``opt_core.kernels.pallas``:
``serve.attention(…, bias=None, key_mask, word=<mode's tier word>, kind="msacol", n_seq=<keys>)``).  The provider's cell for the call's
(jax line, compute capability, dtype, family, bucket) names what serves: on cc 9.0 / jax 0.5 its ``cudnn`` row — the library's fused
attention in the key-LENGTHS form (``serve.cudnn_lengths_attention``: per-row attended-key counts, no dense mask operand; rows whose
attended keys are not a prefix permuted attended-first in graph; a row with no attended key = the stock statement's uniform answer) —
and on other cells the first row of the provider's order there (``served=`` names it).
No kernel, no row name, no size floor and no probe lives here: the provider refuses by name what a row cannot serve and walks its
order; the ``exact`` mode never applies this lever.

``modules.Attention`` is rebound OVER the class bound before it (PALLAS_MSA's; modes.SUPERSEDES: PALLAS_MSA left with ``calls=0`` under this
lever is ``superseded_by``, not partial).  A call is this lever's iff it is bias-free with equal q / v head widths and stock's ``[b,1,1,k]``
mask bias (``msa_attn.classify`` / ``key_mask_of``); pair-biased calls pass through UNCOUNTED to the class below; a bias-free call the
provider has no family for (template point attention) or whose every candidate it refuses proceeds to the wrapped class,
counted by name (``no_cell_family`` / ``mask_form`` / ``face_refused:<kind>``).  Ablation: ``MODEL_OPT_LEVERS_OFF=MSA_COL_CUDNN`` leaves
the column site to PALLAS_MSA, which asks the tier word for every row but ``cudnn`` — a run whose column attention is not on cuDNN; the
provider's own words (``pallas:<row>``) switch single rows off underneath either lever.  At ``--n_gpu`` > 1 the lever stays on (the
MSA stack is replicated per GPU inside the recipe's regions: the site's shapes are the one-GPU shapes).  Floor at ``enable()`` (``require``): jax importable with the GPU backend
and the provider face importable — each a ``Refusal`` by kind = the activation's NOT ACTIVE.

Census (``_STATE``, the LEVER line): ``word= served= cells= calls= fallbacks= fallback_by= shapes= precision= cc=`` (msa_attn's words).
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Optional

from . import msa_attn as _msa

NAME = "MSA_COL_CUDNN"
MODULES = "alphafold.model.modules"
CLASS = "Attention"
MARKER = "_msa_col_cudnn"
MIN_CC = (8, 0)                                    # cuDNN's fused attention needs an sm_80-class part; not read on the call path (the provider's cells decide per card)
N_GPU_REASON = "n_gpu>1"
_STATE: Dict[str, Any] = {"enabled": False, "word": "none", "served": "none", "cells": "none", "calls": 0, "fallbacks": 0, "fallback_by": "none", "shapes": "none",
                          "precision": "none", "cc": None}
_COUNTS: Dict[str, Any] = {"served": {}, "cells": {}, "fallback_by": {}, "shapes": {}, "precision": set()}
_ORIG: Dict[str, Any] = {"cls": None, "modules": None}


class Refusal(RuntimeError):
    """The lever cannot engage on this stack; ``kind`` is the word the activation prints."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{NAME}: {kind} — {detail}" if detail else f"{NAME}: {kind}")
        self.kind = kind
        self.detail = detail


def _cc_tuple(cc) -> Optional[tuple]:
    try:
        major, _, minor = str(cc).partition(".")
        return (int(major), int(minor or 0))
    except (TypeError, ValueError):
        return None


def compute_capability() -> Optional[str]:
    return _msa.compute_capability()


def _dtype_word(dt) -> str:
    s = str(getattr(dt, "name", dt))
    return {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}.get(s, s)


def require() -> Dict[str, Any]:
    """The floor: jax with the GPU backend, and the provider face importable — else Refusal by kind."""
    try:
        import jax  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        raise Refusal("no_jax", repr(e)) from None
    if jax.default_backend() != "gpu":
        raise Refusal("not_gpu_backend", f"the default jax backend is {jax.default_backend()!r}; the provider's attention rows need the GPU backend")
    try:
        _msa.face(); _msa.provider()
    except Exception as e:  # noqa: BLE001
        raise Refusal("no_provider_face", f"{_msa.FACE} is not importable from this opt_core ({type(e).__name__}: {e})") from None
    return {"jax": jax.__version__, "backend": "gpu", "cc": compute_capability()}


def _build(wrapped_cls):
    class Attention(wrapped_cls):  # noqa: D401 - the stock name, so haiku parameter scopes are unchanged
        def __call__(self, q_data, m_data, *args, **kwargs):
            nonbatched_bias = kwargs.get("nonbatched_bias", args[1] if len(args) > 1 else None)
            bias = args[0] if args else kwargs.get("bias", kwargs.get("mask"))
            num_head = self.config.num_head
            key_dim = self.config.get("key_dim", int(q_data.shape[-1])) // num_head
            value_dim = self.config.get("value_dim", int(m_data.shape[-1])) // num_head
            if _STATE["enabled"] and _msa.classify(int(key_dim), int(value_dim), nonbatched_bias is not None) == _msa.KIND_COLUMN:
                out = _msa.serve_call(self, q_data, m_data, bias, nonbatched_bias, kind=_msa.KIND_COLUMN, prefer=None, state=_STATE, counts=_COUNTS)
                if out is not None:
                    _COUNTS["precision"].add(_dtype_word(q_data.dtype))
                    _STATE["precision"] = ",".join(sorted(_COUNTS["precision"]))
                    return out
            return wrapped_cls.__call__(self, q_data, m_data, *args, **kwargs)
    setattr(Attention, MARKER, True)
    Attention.__qualname__ = Attention.__name__ = CLASS
    return Attention


def enable(modules=None) -> Dict[str, Any]:
    """Rebind ``alphafold.model.modules.Attention`` over the class bound there now (PALLAS_MSA's), after the floor (``require``)."""
    facts = require()
    m = modules if modules is not None else importlib.import_module(MODULES)
    A = getattr(m, CLASS)
    if getattr(A, MARKER, False):
        _STATE["enabled"] = True
        return facts
    _ORIG.update(cls=A, modules=m)
    setattr(m, CLASS, _build(A))
    _STATE.update(enabled=True, word=_msa.tier_word(), cc=facts.get("cc"))
    return facts


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
    _STATE.update(enabled=False, word="none", served="none", cells="none", calls=0, fallbacks=0, fallback_by="none", shapes="none", precision="none", cc=None)
    for k, d in _COUNTS.items():
        d.clear()
    _ORIG.update(cls=None, modules=None)
