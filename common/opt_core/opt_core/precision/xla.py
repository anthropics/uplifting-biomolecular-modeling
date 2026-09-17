"""The XLA form of a deterministic recipe, rendered into the core's shape (``opt_core.det.Recipe``).

:func:`xla_recipe` — level 0 = ``opt_core.det.PRODUCTION`` (production numerics, nothing set); level >= 1 = XLA flag words prepended to
the caller's own ``XLA_FLAGS`` (``--xla_gpu_autotune_level=0``, ``--xla_gpu_deterministic_ops=true``), plus plain variables (a
compilation-cache directory). A recipe applies identically to BOTH arms of an equality row (it is the row's definition, never a lever);
``opt_core.det.describe(recipe)`` is its evidence. Standard library at import.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence

from .. import det as _det

__all__ = ["XLA_FLAGS", "xla_flags_value", "xla_recipe"]

XLA_FLAGS = "XLA_FLAGS"


def xla_flags_value(words: Sequence[str], current: Optional[str]) -> str:
    """``words`` prepended to the caller's own XLA_FLAGS string (kept verbatim after them); a word already present is not repeated."""
    have = (current or "").split()
    new = [w for w in words if w not in have]
    return " ".join(new + have).strip()


def xla_recipe(lv: int, flags: Sequence[str] = ("--xla_gpu_autotune_level=0",), extra_env: Optional[Mapping[str, str]] = None,
               environ: Optional[Mapping[str, str]] = None, note: str = "XLA deterministic flags") -> _det.Recipe:
    """The JAX-engine recipe of a level: level 0 -> ``det.PRODUCTION``; level >= 1 -> ``XLA_FLAGS`` = ``flags`` prepended to the value
    found in ``environ`` (default ``os.environ``) at build time, plus ``extra_env`` (e.g. a fixed compilation-cache directory)."""
    if int(lv) == 0:
        return _det.PRODUCTION
    environ = os.environ if environ is None else environ
    env = {XLA_FLAGS: xla_flags_value(tuple(flags), environ.get(XLA_FLAGS))}
    if extra_env:
        env.update({str(k): str(v) for k, v in dict(extra_env).items()})
    return _det.Recipe(level=int(lv), env=env, note=note)
