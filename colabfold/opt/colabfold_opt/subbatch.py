"""SUBBATCH — the sub-batch lever (strategy F7.jax_subbatch; the decision primitive is the shared core's
``opt_core.jax_design.subbatch_policy``, this module is the AlphaFold-multimer adapter around it).

What stock does (alphafold-colabfold 2.3.13): every model is built with ``global_config.subbatch_size = 4`` (``alphafold/model/config.py``
``CONFIG`` / ``CONFIG_MULTIMER`` ``global_config``), and ``mapping.inference_subbatch`` runs every Attention call site — triangle attention
starting / ending node (``modules.py:1049-1054``), MSA row attention (``:875-880``), MSA column and global column attention (``:930-935``,
``:790``) — and every ``Transition`` (``:609-614``) as a sequential scan over chunks of 4 rows: N/4 launches per site of GEMMs and softmaxes
sized ``[4·N, c]``, a setting sized for 16-GB cards. On a card with several times that memory those chunks are launch-bound: the same arithmetic in
fewer, larger launches is what the lever buys.

What the lever does, without editing stock: ``enable(tokens, device_bytes)`` decides ONE value and wraps ``alphafold.model.config.model_config``
so that every model configuration colabfold builds in this process (``colabfold/alphafold/models.py:109``) carries
``global_config.subbatch_size = <value>`` before any model is traced; nothing else in the configuration is touched. The decision
(``decide``): the core policy ``subbatch_policy.choose(tokens=<the run's largest input>, stock_value=4, requested='auto',
device_bytes=<the visible GPU's memory>, peak_estimator=PEAK)`` decides whether the chunk-128 executable fits the device —
``auto:fits`` → 128 (``CHUNK``); ``auto:exceeds`` → stock's 4, named; ``auto:no_device`` → stock's 4, named; no token count (the Python
route without queries) → 128 with ``source=requested:no_tokens``, named. The raised value is a chunk and not "unchunked" because the
unchunked extra-MSA row attention materialises ``[2048, 8, N, N]`` logits (48.8 GiB at 1000 tokens; out of memory at 1400 on 80 GB).
``PEAK`` is the quadratic fit (``subbatch_policy.fit_quadratic``, +10 % margin) through ``PEAK_POINTS``: (tokens, GiB) device peaks of the
chunk-128 program (one multimer model, 3 recycles, an 80 GB card). Under the row-sharded pair stack (``big --n_gpu P``, P > 1;
``decide(..., n_gpu=P)``) the value is 128 with ``source=auto:rowpair``, named: there every chunked site runs on this device's row block
``[N/P, N]`` through the flash kernels the recipe binds (no ``[rows, H, N, N]`` logits are materialised), so the chunk adds nothing to the
per-device allocator peak over stock's 4 and gives fewer, larger launches per site. One GPU (``n_gpu=1``, the default): the
decision above, unchanged.

Numerics: the arithmetic per element is unchanged (the same contractions over the same reduction lengths; only the number of rows per
launch changes), but XLA compiles a different program (GEMM tilings are chosen per shape), so byte equality with stock is not
constructed (Tier 2, inside stock's seed band) for each mode that carries the lever (``modes.TABLE``).
Memory: the chunked sites hold 128 rows of logits instead of 4 (``[128, H, N, N]`` per attention site), which is what ``PEAK`` accounts for.

Evidence: ``_STATE`` — ``calls`` (model configurations rewritten), ``value``, ``source``, ``tokens``, ``stock_value``, ``est_gib`` /
``device_gib`` (the policy's estimate and the device size when it ran); the LEVER line prints them at exit and the manifest records them
(``lever_states_exit.SUBBATCH``). Marker: ``model_config._colabfold_opt_subbatch`` on the wrapper.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from opt_core.jax_design import subbatch_policy as _policy

NAME = "SUBBATCH"
STRATEGY = "F7.jax_subbatch"
TARGET_MODULE = "alphafold.model.config"
TARGET_FUNCTION = "model_config"
MARKER = "_colabfold_opt_subbatch"
STOCK_VALUE = 4                                   # alphafold/model/config.py global_config.subbatch_size (monomer and multimer configurations)
CHUNK = 128                                       # the raised sub-batch: rows per inference_subbatch launch when the chunk-128 program fits the device
PEAK_POINTS = ((400, 2.24), (1000, 6.62), (1400, 13.79))     # (tokens, peak GiB in use) of the chunk-128 program, H100 80 GB, one model, 3 recycles
PEAK = _policy.fit_quadratic(PEAK_POINTS, unit=float(2 ** 30), margin=0.10)
NO_TOKENS_SOURCE = "requested:no_tokens"
ROWPAIR_SOURCE = "auto:rowpair"                    # the value's source under big --n_gpu P > 1: the row-sharded program takes the chunk (decide, n_gpu > 1)

_STATE: Dict[str, Any] = {"enabled": False, "calls": 0, "value": None, "source": None, "tokens": None, "stock_value": STOCK_VALUE,
                          "est_gib": None, "device_gib": None}
_ORIG: Dict[str, Any] = {"model_config": None, "module": None}


def decide(tokens: Optional[int], device_bytes: Optional[int], n_gpu: int = 1) -> Dict[str, Any]:
    """The value for a run whose largest input has ``tokens`` tokens on a device of ``device_bytes`` (None = unknown): the core policy's
    decision with the kit's chunk substituted for 'unchunked' when the chunk-128 program fits (fields: value, source, tokens, stock_value,
    est_gib, device_gib). ``n_gpu`` > 1 (the row-sharded pair stack of ``big --n_gpu P``): the chunk, ``source=auto:rowpair`` (module
    docstring); ``n_gpu`` = 1: the one-GPU decision, byte for byte."""
    if int(n_gpu) > 1:
        return {"value": CHUNK, "source": ROWPAIR_SOURCE, "tokens": int(tokens) if tokens is not None else None, "stock_value": STOCK_VALUE,
                "est_gib": None, "device_gib": round(device_bytes / 2 ** 30, 2) if device_bytes else None}
    if tokens is None:
        return {"value": CHUNK, "source": NO_TOKENS_SOURCE, "tokens": None, "stock_value": STOCK_VALUE, "est_gib": None,
                "device_gib": round(device_bytes / 2 ** 30, 2) if device_bytes else None}
    d = _policy.choose(tokens=int(tokens), stock_value=STOCK_VALUE, requested="auto", device_bytes=device_bytes, peak_estimator=PEAK,
                       when_device_unknown="stock")
    value = CHUNK if d.source == "auto:fits" else d.value
    return {"value": value, "source": d.source, "tokens": int(tokens), "stock_value": STOCK_VALUE,
            "est_gib": round(d.estimated_bytes / 2 ** 30, 2) if d.estimated_bytes is not None else None,
            "device_gib": round(d.device_bytes / 2 ** 30, 2) if d.device_bytes else None}


def _wrap(orig, value):
    def model_config(*args, **kwargs):
        cfg = orig(*args, **kwargs)
        cfg.model.global_config.subbatch_size = value
        _STATE["calls"] += 1
        return cfg
    setattr(model_config, MARKER, True)
    model_config.__wrapped__ = orig
    model_config.__doc__ = orig.__doc__
    return model_config


def enable(tokens: Optional[int] = None, device_bytes: Optional[int] = None, module=None, n_gpu: int = 1) -> Dict[str, Any]:
    """Decide the value and wrap ``alphafold.model.config.model_config`` (idempotent; a second call re-decides and re-wraps the stock
    function). ``module``: the target module (tests pass a stub); None imports ``alphafold.model.config``. ``n_gpu``: the resource axis of
    big (decide). Returns the decision."""
    m = module if module is not None else importlib.import_module(TARGET_MODULE)
    cur = getattr(m, TARGET_FUNCTION)
    orig = getattr(cur, "__wrapped__", None) if getattr(cur, MARKER, False) else cur
    dec = decide(tokens, device_bytes, n_gpu=int(n_gpu))
    setattr(m, TARGET_FUNCTION, _wrap(orig, dec["value"]))
    _ORIG.update(model_config=orig, module=m)
    _STATE.update(dec)
    _STATE["enabled"] = True
    return dict(dec)


def disable() -> None:
    """Restore the stock ``model_config`` (models built afterwards carry stock's value)."""
    m, orig = _ORIG["module"], _ORIG["model_config"]
    if m is not None and orig is not None:
        setattr(m, TARGET_FUNCTION, orig)
    _STATE["enabled"] = False


def marker_present(module=None) -> Optional[bool]:
    import sys
    m = module if module is not None else sys.modules.get(TARGET_MODULE)
    if m is None:
        return None
    return bool(getattr(getattr(m, TARGET_FUNCTION, None), MARKER, False))


def reset_for_tests() -> None:
    disable()
    _STATE.update(enabled=False, calls=0, value=None, source=None, tokens=None, stock_value=STOCK_VALUE, est_gib=None, device_gib=None)
    _ORIG.update(model_config=None, module=None)
