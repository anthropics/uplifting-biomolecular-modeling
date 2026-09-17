"""PALLAS_MSA — the two bias-free / narrow-head attention sites of AlphaFold-Multimer through the shared core's JAX-family provider by TIER
WORD (``opt_core.kernels.pallas``: ``serve.attention(q, k, v, bias, key_mask, scale, word=<mode's tier word>, kind=…, n_seq=…)``).  The
provider's cell for the call's (jax line, compute capability, dtype, family, token bucket) names the row that serves — its Pallas
flash rows (head dims below 16 zero-padded to 16 inside the face), cuDNN's fused attention, or the XLA statement — and this module owns no
kernel, no row name, no size floor and no launch setting: the tier word applies the provider's tuned launch settings itself.

Sites (``alphafold/model/modules.py:635-718``, the ``Attention`` body): (1) extra-MSA row attention (8 heads × 8 channels + the pair bias
``[h, q, k]``, keys = residues; provider kind ``extramsa_slab512``, depth = the call's MSA rows) and (2) every BIAS-FREE call whose family
the provider's cell table carries — MSA column attention (8 heads × 32 channels, keys = the cluster MSA depth; kind ``msacol``, depth = keys).  At
the column site this lever asks the tier word for every row BUT ``cudnn`` (``prefer=``): the cuDNN row at that site is the lever
``MSA_COL_CUDNN`` (msa_col_cudnn.py, bound over this class; modes.SUPERSEDES) — so each of the two stays individually switchable
(``MODEL_OPT_LEVERS_OFF``).  A call of neither class (pair-biased 32-channel heads: AF_PALLAS_ATTN's / TRIATTN_XLA's; a bias-free call whose
family has no cell, e.g. template point attention over 4 templates: ``no_cell_family``; a mask bias that is not stock's
``[b,1,1,k]`` key mask: ``mask_form``) runs the class this one wraps, counted by name.  Served body: stock's q / k / v projections (same
einsums, same parameters), the face, stock's gating and output projection.  Numerics: the provider's tolerance class for its Pallas / cuDNN
rows (bf16 operands, f32 accumulation, exact softmax) = tier 2, inside stock's own bf16 re-association band; never in ``exact``.

Census (``_STATE``, the LEVER line): ``word=<tier word asked> served=<arm>:<n>,…|none cells=<cell key>:<n>,…|none calls=<n> fallbacks=<n>
fallback_by=<word>:<n>,…|none shapes=b<b>xs<k>xh<h>xd<d>[p16]:<n>,…|none cc=<cc>`` — ``served`` / ``cells`` are the provider's own words for
what it served per call class (an arm it stepped aside from at call time is in its own census, ``opt_core.cell_census``).
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

MODULES = "alphafold.model.modules"                  # the module object whose ``Attention`` is rebound (the multimer code resolves it there at call time)
CLASS = "Attention"
MARKER = "_pallas_msa"                               # set on the rebound class (registry probe: ("class_attr", MODULES, CLASS, MARKER))
PROVIDER = "opt_core.kernels.pallas"                 # the JAX-family provider (select / family / families / Refusal / ROW_NAMES)
FACE = "opt_core.kernels.pallas.serve"               # its call faces (attention, report)
SERVE = "opt_core.kernels.pallas_attn_serve"         # the flash row's serve layer: its mask-bias threshold and census shape words only (no call goes through it)
KIND_COLUMN = "msacol"                               # provider kind of a bias-free call (MSA column attention; depth = keys)
KIND_NARROW = "extramsa_slab512"                     # provider kind of a pair-biased call with heads narrower than the flash rows' operand floor (extra-MSA rows; depth = MSA rows in the call)
NARROW_BELOW = 16                                    # per-head channels below which a PAIR-BIASED call is this lever's (the provider pads to 16: PAD_HEAD_DIM_TO); at / above: AF_PALLAS_ATTN's / TRIATTN_XLA's
COLUMN_ROW_ELSEWHERE = "cudnn"                       # the provider row this lever leaves to MSA_COL_CUDNN at the column site (prefer= every other row)
NO_CELL_FAMILY = "no_cell_family"                    # fallback word: a bias-free call whose family has no provider cell (template point attention) -> the wrapped class
MASK_FORM = "mask_form"                              # fallback word: the call's mask bias is not stock's [b,1,1,k] key mask -> the wrapped class
FACE_REFUSED = "face_refused"                        # fallback word: the provider refused every candidate by name at this call -> the wrapped class
TIER_WORDS = ("exact", "fast", "big")
_STATE: Dict[str, Any] = {"enabled": False, "word": "none", "served": "none", "cells": "none", "calls": 0, "fallbacks": 0, "fallback_by": "none", "shapes": "none", "cc": None}
_COUNTS: Dict[str, Dict[str, int]] = {"served": {}, "cells": {}, "fallback_by": {}, "shapes": {}}
_ORIG: Dict[str, Any] = {"cls": None, "modules": None}


def provider():
    return importlib.import_module(PROVIDER)


def face():
    return importlib.import_module(FACE)


def tier_word(environ=None) -> str:
    """The tier word this process asks the provider: the kit mode's own word (COLABFOLD_OPT=fast -> 'fast', big -> 'big')."""
    from . import modes as _modes
    m = _modes.from_env(environ)
    return m if m in TIER_WORDS else "fast"


def _bump(state: Dict[str, Any], counts: Dict[str, Dict[str, int]], field: str, key: str) -> None:
    d = counts[field]
    d[key] = d.get(key, 0) + 1
    state[field] = ",".join("%s:%d" % kv for kv in sorted(d.items()))


def key_mask_of(bias):
    """Stock's mask bias ``1e9 * (mask - 1)`` as ``[b,1,1,k]`` -> the key mask ``[b,k]`` (True = attend); None when the bias is not that form."""
    shape = getattr(bias, "shape", None)
    if shape is None or len(shape) != 4 or int(shape[1]) != 1 or int(shape[2]) != 1:
        return None
    try:
        thr = float(getattr(importlib.import_module(SERVE), "MASKED_BIAS_THRESHOLD"))
    except Exception:  # noqa: BLE001 - the serve layer absent: stock's masked value is -1e9, attended 0
        thr = -1e4
    return bias[:, 0, 0, :] > thr


def classify(key_dim: int, value_dim: int, has_pair_bias: bool) -> Optional[str]:
    """The provider kind of a call this lever routes, or None (not this lever's call: the wrapped class, uncounted)."""
    if has_pair_bias:
        return KIND_NARROW if (key_dim == value_dim and key_dim < NARROW_BELOW) else None
    return KIND_COLUMN if key_dim == value_dim else None


def serve_call(self, q_data, m_data, bias, nonbatched_bias, *, kind: str, prefer: Optional[Sequence[str]], state: Dict[str, Any],
               counts: Dict[str, Dict[str, int]], aside_word: Optional[str] = None):
    """The served body for a haiku ``Attention`` instance (call it inside the module's ``__call__``): stock's projections, the provider face
    by tier word, stock's gating / output.  Returns None after counting a NAMED fallback when the call is not served here (the caller then
    runs the wrapped class): ``no_cell_family`` (the provider has no family for this call class), ``mask_form``, ``face_refused``."""
    import jax  # noqa: WPS433 (lazy by contract)
    import jax.numpy as jnp  # noqa: WPS433
    import haiku as hk  # noqa: WPS433
    P, F = provider(), face()
    num_head = self.config.num_head
    key_dim = self.config.get("key_dim", int(q_data.shape[-1])) // num_head
    value_dim = self.config.get("value_dim", int(m_data.shape[-1])) // num_head
    kmask = key_mask_of(bias)
    if kmask is None:
        state["fallbacks"] += 1
        _bump(state, counts, "fallback_by", MASK_FORM)
        return None
    n_keys, n_rows = int(m_data.shape[-2]), int(q_data.shape[0])
    depth = n_keys if kind == KIND_COLUMN else n_rows                             # msacol: keys = the MSA depth; extra-MSA rows: the rows in this (sub-batched) call
    fam = P.family("attn", kind=kind, heads=int(num_head), head_dim=int(key_dim), n_seq=depth)
    if fam not in P.families("attn"):                                            # no cell family for this call class: the stock statement is the wrapped class, by name
        state["fallbacks"] += 1
        _bump(state, counts, "fallback_by", NO_CELL_FAMILY)
        return None
    glorot_uniform = lambda: hk.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")  # noqa: E731
    q_weights = hk.get_parameter("query_w", shape=(q_data.shape[-1], num_head, key_dim), dtype=q_data.dtype, init=glorot_uniform())
    k_weights = hk.get_parameter("key_w", shape=(m_data.shape[-1], num_head, key_dim), dtype=q_data.dtype, init=glorot_uniform())
    v_weights = hk.get_parameter("value_w", shape=(m_data.shape[-1], num_head, value_dim), dtype=q_data.dtype, init=glorot_uniform())
    q = jnp.einsum("bqa,ahc->bqhc", q_data, q_weights)                            # stock's contractions and layout [b, q, h, c]; 1/sqrt(c) is the face's `scale`
    k = jnp.einsum("bka,ahc->bkhc", m_data, k_weights)
    v = jnp.einsum("bka,ahc->bkhc", m_data, v_weights)
    nb = None if nonbatched_bias is None else nonbatched_bias.astype(q.dtype)
    before = dict(F.report().get("counts", {}))
    try:
        weighted_avg = F.attention(q, k, v, nb, kmask, float(key_dim) ** -0.5, word=state["word"], kind=kind, n_seq=depth, layout="BSHD",
                                   prefer=(list(prefer) if prefer else None))
    except P.Refusal as e:                                                       # every candidate refused by name at this call: the wrapped class, counted
        state["fallbacks"] += 1
        _bump(state, counts, "fallback_by", "%s:%s" % (FACE_REFUSED, getattr(e, "kind", "refused")))
        return None
    after = F.report().get("counts", {})
    arm = next((k_[len("served:attention:"):] for k_, n in after.items() if k_.startswith("served:attention:") and n > before.get(k_, 0)), "served")
    state["calls"] += 1
    _bump(state, counts, "served", arm + ("(%s)" % aside_word if aside_word else ""))
    try:
        sel = F.resolve("attn", fam, q.dtype, int(q.shape[1]), word=state["word"], prefer=(list(prefer) if prefer else None), key_masked=True)
        if getattr(sel, "cell_key", None):
            _bump(state, counts, "cells", str(sel.cell_key).replace("|", "/"))
    except Exception:  # noqa: BLE001 - the census is observation; the call was served above
        pass
    shape = "b%dxs%dxh%dxd%d%s" % (n_rows, n_keys, int(num_head), int(key_dim), "p16" if key_dim < NARROW_BELOW else "")
    _bump(state, counts, "shapes", shape)
    if int(weighted_avg.shape[-1]) != value_dim:
        weighted_avg = weighted_avg[..., :value_dim]
    init = hk.initializers.Constant(0.0) if self.global_config.zero_init else glorot_uniform()
    if self.config.gating:
        gating_weights = hk.get_parameter("gating_w", shape=(q_data.shape[-1], num_head, value_dim), dtype=q_data.dtype, init=hk.initializers.Constant(0.0))
        gating_bias = hk.get_parameter("gating_b", shape=(num_head, value_dim), dtype=q_data.dtype, init=hk.initializers.Constant(1.0))
        gate_values = jnp.einsum("bqc, chv->bqhv", q_data, gating_weights) + gating_bias
        weighted_avg *= jax.nn.sigmoid(gate_values)
    o_weights = hk.get_parameter("output_w", shape=(num_head, value_dim, self.output_dim), dtype=q_data.dtype, init=init)
    o_bias = hk.get_parameter("output_b", shape=(self.output_dim,), dtype=q_data.dtype, init=hk.initializers.Constant(0.0))
    return jnp.einsum("bqhc,hco->bqo", weighted_avg, o_weights) + o_bias


def column_prefer() -> Tuple[str, ...]:
    """Every provider row but the one MSA_COL_CUDNN binds at the column site."""
    try:
        rows = tuple(provider().ROW_NAMES)
    except Exception:  # noqa: BLE001
        rows = ()
    return tuple(r for r in rows if r != COLUMN_ROW_ELSEWHERE)


def _make_class(A):
    class Attention(A):  # noqa: D401 - the stock name, so haiku parameter scopes are unchanged
        def __call__(self, q_data, m_data, *args, **kwargs):
            nonbatched_bias = kwargs.get("nonbatched_bias", args[1] if len(args) > 1 else None)
            bias = args[0] if args else kwargs.get("bias", kwargs.get("mask"))
            num_head = self.config.num_head
            key_dim = self.config.get("key_dim", int(q_data.shape[-1])) // num_head
            value_dim = self.config.get("value_dim", int(m_data.shape[-1])) // num_head
            kind = classify(int(key_dim), int(value_dim), nonbatched_bias is not None)
            if kind is not None and _STATE["enabled"]:
                out = serve_call(self, q_data, m_data, bias, nonbatched_bias, kind=kind, prefer=(column_prefer() if kind == KIND_COLUMN else None),
                                 state=_STATE, counts=_COUNTS, aside_word=("%s_by=MSA_COL_CUDNN" % COLUMN_ROW_ELSEWHERE if kind == KIND_COLUMN else None))
                if out is not None:
                    return out
            return A.__call__(self, q_data, m_data, *args, **kwargs)
    setattr(Attention, MARKER, True)
    Attention.__qualname__ = Attention.__name__ = CLASS
    return Attention


def compute_capability() -> Optional[str]:
    try:
        import jax  # noqa: WPS433
        for d in jax.devices():
            cc = getattr(d, "compute_capability", None)
            if cc:
                return str(cc)
    except Exception:  # noqa: BLE001
        return None
    return None


def require() -> None:
    """The floor, by name: the flash row's serve layer (``SERVE``: a shared core without the Pallas attention kernels is ``pallas_missing``)
    and the provider face (``FACE``) importable from this opt_core."""
    for mod, word in ((SERVE, "pallas_missing"), (PROVIDER, "provider_missing"), (FACE, "provider_missing")):
        try:
            got = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("%s: %s is not importable from this opt_core (%s: %s)" % (word, mod, type(e).__name__, e)) from None
        req = getattr(got, "require", None) if mod == SERVE else None
        if callable(req):
            req(require_gpu=False)                                                # the serve layer's own floor (jax line; raises by name)


def enable(modules=None) -> None:
    """Rebind ``alphafold.model.modules.Attention`` over the class bound there now (AF_PALLAS_ATTN's when it is on, else stock's)."""
    require()
    m = modules if modules is not None else importlib.import_module(MODULES)
    A = getattr(m, CLASS)
    if getattr(A, MARKER, False):
        _STATE["enabled"] = True
        return
    _ORIG.update(cls=A, modules=m)
    setattr(m, CLASS, _make_class(A))
    _STATE.update(enabled=True, word=tier_word(), cc=compute_capability())


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
    _STATE.update(enabled=False, word="none", served="none", cells="none", calls=0, fallbacks=0, fallback_by="none", shapes="none", cc=None)
    for d in _COUNTS.values():
        d.clear()
    _ORIG.update(cls=None, modules=None)
