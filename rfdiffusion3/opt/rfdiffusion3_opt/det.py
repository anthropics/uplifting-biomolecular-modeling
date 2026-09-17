"""The deterministic recipe (``design --det 0|1``), applied identically on every route — ``off`` (the stock child applies it itself,
before upstream is imported), ``exact`` / ``fast`` (the package applies it in-process after the activation, before upstream is imported)
and the ``.pth`` route (``RFDIFFUSION3_OPT=<mode> RFDIFFUSION3_OPT_DET=1``). A stock-side switch, never a mode; level 0 (the default)
leaves the process at upstream's default numerics.

Level 1 = the four statements under which stock RFdiffusion3 reproduces its own bits run to run at a fixed ``seed=``:
  (1) ``CUBLAS_WORKSPACE_CONFIG=:4096:8``                     exported before ``import torch`` (cuBLAS reads it when its handle is created)
  (2) ``torch.use_deterministic_algorithms(True, warn_only=True)``; ``torch.backends.cudnn.deterministic = True``; ``benchmark = False``
  (3) ``FOUNDRY_DET_SCATTER=1`` and ``foundry.utils.torch.scatter_mean`` served by the fixed-order segment reduction of ``det_scatter``
      (upstream's ``index_reduce(..., "mean")`` accumulates with CUDA atomics in an undefined order: the one op of the network that is not
      repeatable run to run). Numerics class of (3): the same mean, a different (fixed) summation order, fp32 accumulation — not bitwise
      equal to the atomic op, bitwise repeatable. ``exact`` under the recipe produces the designs ``off`` under the recipe produces.
The two variable names of (1) and (3) are listed in stock/PINS.json ``stock_environment.must_be_absent``: a caller's inherited copies never
reach the stock child (stack.strip_env); at level 1 the child sets them itself after its environment proof.
"""
from __future__ import annotations

import os
import sys
from typing import Dict

LEVELS = (0, 1)
DEFAULT_LEVEL = 0
ENV_LEVEL = "RFDIFFUSION3_OPT_DET"                                   # the .pth route's form of --det (a package switch: _autoload.ENV_NAMES; stripped from the stock child)
CUBLAS_ENV = ("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
SCATTER_ENV = ("FOUNDRY_DET_SCATTER", "1")
RECIPE_ENV = (CUBLAS_ENV[0], SCATTER_ENV[0])                         # the two variables a level-1 process carries
SCATTER_TARGET = ("foundry.utils.torch", "scatter_mean")            # upstream's scatter-mean (foundry/utils/torch.py), bound by name at rfd3's call sites
_MARK = "__rfdiffusion3_opt_det__"


def level(lv) -> int:
    lv = int(lv)
    if lv not in LEVELS:
        raise ValueError(f"--det {lv}: levels are {LEVELS}")
    return lv


def level_from_env(environ=None) -> int:
    """``RFDIFFUSION3_OPT_DET`` unset or ``0`` -> 0, ``1`` -> 1; anything else raises ValueError by name."""
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV_LEVEL) or "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"{ENV_LEVEL}={v!r}: levels are 0|1")
    return int(v)


def env_before_torch(lv: int) -> Dict[str, str]:
    """The variables a level-1 process exports before ``import torch`` (statements (1) and (3)); {} at level 0."""
    return {CUBLAS_ENV[0]: CUBLAS_ENV[1], SCATTER_ENV[0]: SCATTER_ENV[1]} if level(lv) else {}


def install_scatter() -> dict:
    """Serve ``foundry.utils.torch.scatter_mean`` by det_scatter.foundry_scatter_mean (statement (3)); modules that already bound the
    name get the same function. Idempotent. Returns what was bound."""
    import importlib
    from . import det_scatter
    mod = importlib.import_module(SCATTER_TARGET[0])
    orig = getattr(mod, SCATTER_TARGET[1])
    if getattr(orig, _MARK, False):
        return {"target": ".".join(SCATTER_TARGET), "installed": "already", "rebound": []}

    def scatter_mean(zeros, dim, index, source):
        if zeros.device.type == "mps":                               # upstream's own MPS branch stays upstream's
            return orig(zeros, dim, index, source)
        return det_scatter.foundry_scatter_mean(zeros, dim, index, source)

    setattr(scatter_mean, _MARK, True)
    scatter_mean.__wrapped__ = orig
    setattr(mod, SCATTER_TARGET[1], scatter_mean)
    rebound = []
    for name, m in list(sys.modules.items()):                        # a module imported earlier that bound the name (`from foundry.utils.torch import scatter_mean`)
        if m is None or m is mod:
            continue
        try:
            if getattr(m, SCATTER_TARGET[1], None) is orig:
                setattr(m, SCATTER_TARGET[1], scatter_mean)
                rebound.append(name)
        except Exception:
            continue
    return {"target": ".".join(SCATTER_TARGET), "installed": "yes", "rebound": sorted(rebound)}


def apply(lv: int, environ=None) -> dict:
    """Apply the recipe in THIS process: the variables (before torch when it is not imported yet), torch's switches, the scatter.
    Call before upstream (``rfd3``) is imported. Returns what was set (``{"level": 0}`` at level 0)."""
    lv = level(lv)
    if not lv:
        return {"level": 0}
    environ = os.environ if environ is None else environ
    torch_was_imported = "torch" in sys.modules
    for k, v in env_before_torch(lv).items():
        environ[k] = v
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    scatter = install_scatter()
    return {"level": 1, CUBLAS_ENV[0]: environ.get(CUBLAS_ENV[0]), SCATTER_ENV[0]: environ.get(SCATTER_ENV[0]),
            "deterministic_algorithms": True, "warn_only": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
            "torch_imported_before": torch_was_imported, "scatter": scatter}


def word(lv: int) -> str:
    """The activation / STOCK line's word: ``det=0`` | ``det=1``."""
    return f"det={level(lv)}"
