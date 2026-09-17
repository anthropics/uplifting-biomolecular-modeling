"""The deterministic recipe (``--det``), applied the same way on every arm — a stock-side switch, never a mode.

At default XLA autotuning, XLA may pick different GEMM/conv kernels in different PROCESSES on some software stacks, so two processes
of the same driver on the same input need not write the same bytes. The recipe exports one XLA flag to every process, stock and kit
alike: ``XLA_FLAGS=--xla_gpu_autotune_level=0`` prepended to whatever ``XLA_FLAGS`` the caller had — rendered by the tree's one JAX
recipe form, ``opt_core.precision.xla.xla_recipe`` (:func:`recipe` is this kit's level table over it; a flag word the caller
already carries is not repeated). ``env(1)`` is that variable for a child process. The driver records the variable it saw in its own
``proc_start`` timers line (``predict_pdb.py:705-708``, ``env.XLA_FLAGS``) — the tool's own evidence of the recipe. Cost: +11-12 %
steady model time, -35 % compile time. Without ``--det`` nothing is set, on either arm ("default numerics").
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

from opt_core import det as _core_det
from opt_core.precision import xla as _core_xla

ENV_XLA = _core_xla.XLA_FLAGS                                # "XLA_FLAGS"
FLAG = "--xla_gpu_autotune_level=0"
LEVELS = {1: "XLA_FLAGS=--xla_gpu_autotune_level=0 on every process"}
PREFIX = "[det]"


def recipe(level: int, environ: Optional[dict] = None) -> "_core_det.Recipe":
    """The level as the tree's recipe shape (``opt_core.det.Recipe``): this kit's flag word prepended to the caller's XLA_FLAGS."""
    environ = os.environ if environ is None else environ
    if int(level) not in LEVELS:
        raise ValueError(f"--det takes {sorted(LEVELS)} (got {level!r})")
    return _core_xla.xla_recipe(int(level), flags=(FLAG,), environ=environ, note=LEVELS[int(level)])


def env(level: int, environ: Optional[dict] = None) -> Dict[str, str]:
    """The recipe's one variable for a child process: the flag prepended to the caller's XLA_FLAGS."""
    return dict(recipe(level, environ).env)


def describe(level: Optional[int], environ: Optional[dict] = None) -> dict:
    """The recipe block of the manifest: level, the variable as set, whether the flag is present."""
    environ = os.environ if environ is None else environ
    if level is None:
        return {"on": False, "level": None, "env": {}, "flag_present": FLAG in (environ.get(ENV_XLA) or "")}
    return {"on": True, "level": int(level), "what": LEVELS[int(level)], "env": {ENV_XLA: environ.get(ENV_XLA)}, "flag_present": FLAG in (environ.get(ENV_XLA) or "")}


def line(level: int, environ: dict) -> str:
    return f"{PREFIX} applied level={int(level)} {ENV_XLA}={environ.get(ENV_XLA)!r}"


def print_line(level: int, environ: dict, stream=None) -> str:
    s = line(level, environ)
    print(s, file=stream or sys.stderr, flush=True)
    return s
