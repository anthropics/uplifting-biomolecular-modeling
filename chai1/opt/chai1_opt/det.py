"""The deterministic recipe (``--det 0|1``), applied identically in every arm — the package's one documented copy of kit code.

Level 1 is the kit's own ``CHAI_DETERMINISTIC=1``: the four statements of ``apply_deterministic_mode()`` in ``kit/chai_proto.py``,
which the kit driver executes itself when that variable is set (``chai_worker.py``, after ``import torch``, before the first CUDA call):
  (1) ``torch._C._jit_set_profiling_mode(False)``   unless CHAI_JIT_PROFILING_OFF=0 — every call of the traced ESM2-3B fp16 module runs
                                                    the same (first-call) graph (kit KNOWN_ISSUES.md §2)
  (2) ``CUBLAS_WORKSPACE_CONFIG=:4096:8``           ``os.environ.setdefault``
  (3) ``torch.backends.cudnn.deterministic = True``; ``torch.backends.cudnn.benchmark = False``
  (4) ``torch.use_deterministic_algorithms(True, warn_only=False)``   (the kit's ``CHAI_DETERMINISTIC=warn`` -> warn_only=True)

Kit route (``pred --mode exact|fast|big --det 1``): the package sets ``CHAI_DETERMINISTIC=1`` in the driver's environment and the kit's own
function runs — nothing here is executed. Stock route and in-process route (``enable(mode, det=1)``, ``CHAI1_OPT=<mode>`` with
``CHAI_DETERMINISTIC=1``): those processes have no kit module on their path, so ``apply()`` below restates the same four statements
(the lock test ``tests/test_locks.py`` ``TestDetRecipePair`` parses the kit function and compares). One difference of placement, none
of effect: the stock caller exports (2) BEFORE ``import torch`` (``env_before_torch``), the kit sets it after the import and before
the first CUDA call; cuBLAS reads the variable when its handle is created, at the first CUDA call in both cases.

These are stock-side switches, never a mode; level 0 leaves the process at default numerics.
"""
from __future__ import annotations

import os
from typing import Dict

LEVELS = (0, 1)
DEFAULT_LEVEL = 0
ENV_KIT = "CHAI_DETERMINISTIC"                   # the kit's switch (read by apply_deterministic_mode)
ENV_JIT = "CHAI_JIT_PROFILING_OFF"               # the kit's opt-out of statement (1)
CUBLAS_ENV = ("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def level(lv) -> int:
    lv = int(lv)
    if lv not in LEVELS:
        raise ValueError(f"--det {lv}: levels are {LEVELS}")
    return lv


def level_from_env(environ=None) -> int:
    """The level the kit's own switch asks for: ``CHAI_DETERMINISTIC`` unset or ``0`` -> 0, ``1`` -> 1; any other value (the kit's
    ``warn`` variant included) is not a level of this package and raises ValueError."""
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV_KIT) or "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"{ENV_KIT}={v!r}: the package's levels are 0|1 (the kit's 'warn' form is not offered)")
    return int(v)


def env_before_torch(lv: int) -> Dict[str, str]:
    """Environment the stock caller exports before ``import torch`` at this level (statement (2) only)."""
    return {CUBLAS_ENV[0]: CUBLAS_ENV[1]} if level(lv) else {}


def env_for_driver(lv: int) -> Dict[str, str]:
    """Environment the package sets for the kit driver at this level: the kit's own switch, so that the kit's function applies the
    recipe itself. Level 0 sets nothing (the driver's ``apply_deterministic_mode`` returns "off")."""
    return {ENV_KIT: "1"} if level(lv) else {}


def apply(lv: int, torch=None, environ=None) -> dict:
    """Apply the recipe in THIS process (stock caller, after ``import torch``, before the first CUDA call). Returns what was set."""
    lv = level(lv)
    environ = os.environ if environ is None else environ
    if not lv:
        return {"level": 0}
    if torch is None:
        import torch  # noqa: F811
    profiling_off = environ.get(ENV_JIT, "1") != "0"
    if profiling_off:
        torch._C._jit_set_profiling_mode(False)                                                    # (1)
    environ.setdefault(*CUBLAS_ENV)                                                                  # (2)
    torch.backends.cudnn.deterministic = True                                                       # (3)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)                                       # (4)
    return {"level": 1, "jit_profiling_mode_off": profiling_off, CUBLAS_ENV[0]: environ.get(CUBLAS_ENV[0]),
            "cudnn_deterministic": True, "cudnn_benchmark": False, "deterministic_algorithms": True, "warn_only": False}
