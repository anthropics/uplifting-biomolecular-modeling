"""The deterministic recipe (`--det 0|1`, env E1_OPT_DET), applied identically in both modes — off and exact alike. The recipe's
VALUES live in the kit's own pins module (`pins.DET_RECIPE`) and are read from there; this module only applies them. The switches are
stock-side settings, never a mode.

  environment   every `DET_RECIPE["env"]` entry, set before torch is imported (the wrapper command exports them into the item's
                subprocess environment, so they precede the CUDA context; `apply_env` also sets them in-process for a standalone run)
  seeds         `random.seed`, `numpy.random.seed` (when numpy is present), `torch.manual_seed` (+ `cuda.manual_seed_all`) = DET_RECIPE["seed"]
  torch         `torch.use_deterministic_algorithms(DET_RECIPE["deterministic_algorithms"])`, `backends.cuda.matmul.allow_tf32` and
                `backends.cudnn.allow_tf32` = DET_RECIPE["tf32"], `backends.cudnn.benchmark` = DET_RECIPE["cudnn_benchmark"]
  autotune pin  `pins.apply_autotune_pin(size)` — the kit's own function — before the first forward: the hub RMSNorm kernel's Triton
                autotuner picks `num_warps` by a timing run at the first call, so two cold processes can differ; the pin selects
                one of the stock's own outcomes (the kit's pin table), it adds no numerics. Under exact the kit applies it itself; under
                off the stock runner applies it at E1Scorer construction (stock_score.py), the same point.

`--det 0` (the default) touches nothing: the tool runs at its own defaults in both modes.
"""
from __future__ import annotations

import os
import random
import sys
from typing import Optional

LEVELS = (0, 1)


def check_level(det) -> int:
    try:
        lv = int(det)
    except (TypeError, ValueError):
        raise ValueError(f"--det must be one of {LEVELS}, got {det!r}")
    if lv not in LEVELS:
        raise ValueError(f"--det must be one of {LEVELS}, got {det!r}")
    return lv


def recipe(pins) -> dict:
    """The recipe as the pins module states it (a copy of the fields this module applies)."""
    r = pins.DET_RECIPE
    return {"seed": r.get("seed"), "deterministic_algorithms": r.get("deterministic_algorithms"), "tf32": r.get("tf32"),
            "cudnn_benchmark": r.get("cudnn_benchmark"), "env": dict(r.get("env") or {}), "framework": r.get("framework"),
            "triton_autotune_pin": "pins.apply_autotune_pin(size)"}


def env_for(pins, level: int) -> dict:
    """The environment entries the recipe names (empty at level 0) — what the wrapper command exports into the item's subprocess."""
    if level == 0:
        return {}
    return {k: str(v) for k, v in (pins.DET_RECIPE.get("env") or {}).items()}


def env_missing(pins, level: int, environ=None) -> list:
    """[(key, present value or None, wanted value)] for every recipe environment entry not in force (empty at level 0)."""
    environ = os.environ if environ is None else environ
    return [(k, environ.get(k), v) for k, v in env_for(pins, level).items() if environ.get(k) != v]


def apply_env(pins, level: int, environ=None) -> dict:
    """Set the recipe's environment entries (before torch is imported). Returns what was set."""
    environ = os.environ if environ is None else environ
    out = env_for(pins, level)
    for k, v in out.items():
        environ[k] = v
    return out


def apply_torch(pins, level: int = 1) -> dict:
    """Seeds and torch switches of the recipe (imports torch). Returns the applied record; at level 0 nothing is touched."""
    applied = {"level": level, "applied": {}}
    if level == 0:
        return applied
    r = pins.DET_RECIPE
    seed = int(r["seed"])
    random.seed(seed)
    applied["applied"]["random.seed"] = seed
    try:
        import numpy as np
        np.random.seed(seed)
        applied["applied"]["numpy.random.seed"] = seed
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if r["deterministic_algorithms"]:
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = bool(r["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(r["tf32"])
    torch.backends.cudnn.benchmark = bool(r["cudnn_benchmark"])
    applied["applied"].update({"torch.manual_seed": seed, "torch.use_deterministic_algorithms": bool(r["deterministic_algorithms"]),
                               "torch.tf32": bool(r["tf32"]), "torch.cudnn.benchmark": bool(r["cudnn_benchmark"])})
    return applied


def describe(pins, level: int) -> dict:
    """What the level does (values from the pins module): the record `describe` callers print or test against."""
    if level == 0:
        return {"level": 0, "what": "nothing: the tool's own defaults in every arm"}
    return {"level": 1, "what": "the recipe: env + seeds + torch deterministic switches + the kit's autotune pin, in every arm", "recipe": recipe(pins)}


def torch_state() -> Optional[dict]:
    """The torch switches as they are now (only when torch is loaded)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return {"deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}
    except Exception:  # noqa: BLE001
        return None
