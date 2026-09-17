"""The deterministic recipes (``--det 0|1|2``), applied identically in every arm — the package's one documented deviation from
"wrap, never transcribe". These are stock-side switches, never a mode.

Level 1 — the kit's recipe. The fast-inference kit applies it inside its own drivers, not through the served driver, so a prediction
made through the package cannot inherit it: this module states the same settings so that ``pred --det 1`` (stock subprocess and kit
modes alike) runs under them. The recipe:
  environment      CUBLAS_WORKSPACE_CONFIG=:4096:8 and ESMFOLD2_DETERMINISTIC_SCATTER=1 (set before torch is imported)
  torch            torch.use_deterministic_algorithms(True, warn_only=True)
  model module     modeling_esmfold2_common.set_deterministic_scatter(True) when that function exists
  fold kwargs      lm_dropout=0.0, msa_column_mask_rate=0.0
Two of its switches are inert on the pinned build (``ESMFOLD2_DETERMINISTIC_SCATTER`` has no reader; the ``lm_dropout=0`` keyword is a
no-op in upstream's ``_lm_dropout_context`` — ``None``/``0`` leaves the checkpoint's per-loop LM dropout active); they are carried as
written. Effective content: the cuBLAS workspace, deterministic algorithms, column mask 0, the seed.

Level 2 — level 1 plus the config-level switch the keyword cannot reach: ``lm_encoder.lm_dropout = 0.0`` set on
the model config BEFORE the weights load (``MODEL_CONFIG_OVERRIDES``, ``apply_config``), so the per-loop LM dropout is off. A
stock-side setting, never a mode; ``pred`` and the stock caller take it (the kit's server loads its own
models).

Stock run-to-run reproducibility under either recipe is empirical, not a torch guarantee.
"""
import os
import re
from typing import Dict, Optional

LEVELS = (0, 1, 2)
LEVEL_WHAT = {0: "off", 1: "the kit's recipe (cuBLAS workspace + deterministic algorithms + column mask 0 + seed)",
              2: "level 1 + config-level lm_encoder.lm_dropout=0 before the weights load"}
ENV: Dict[str, str] = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "ESMFOLD2_DETERMINISTIC_SCATTER": "1"}
FOLD_OVERRIDES: Dict[str, float] = {"lm_dropout": 0.0, "msa_column_mask_rate": 0.0}
TORCH_DETERMINISTIC = {"mode": True, "warn_only": True}
MODEL_CONFIG_OVERRIDES: Dict[str, float] = {"lm_encoder.lm_dropout": 0.0}           # level 2 only: dotted config attribute -> value
DET_SWITCHES: Dict[str, str] = {"ESMFOLD2_DETERMINISTIC_SCATTER": "1"}                 # the kit's own switch inside the recipe (the model module's deterministic scatter)


def recipe(lv: int = 1):
    """The recipe of a level in the release tree's shape (``opt_core.det.Recipe`` via ``opt_core.precision.recipe.torch_recipe``): level 0 =
    production numerics; levels 1 and 2 = the cuBLAS workspace + the kit's scatter switch (``ENV``, the same words — the package test
    holds ``recipe(1).env == ENV``). Level 2's config override and the fold keywords are model-side (``apply_config``, ``fold_overrides``),
    not environment. The in-process statements stay ``apply_torch`` (``warn_only=True`` + the scatter switch: the kit's own words)."""
    from opt_core.precision.recipe import torch_recipe
    return torch_recipe(level(lv), switches=DET_SWITCHES, note=LEVEL_WHAT[min(level(lv), 1)] if level(lv) else "off")


def level(value) -> int:
    lv = int(value or 0)
    if lv not in LEVELS:
        raise ValueError(f"--det must be one of {LEVELS} (got {value!r}): 0 = off, 1 = the kit's recipe, 2 = level 1 + config-level lm_encoder.lm_dropout=0")
    return lv


def apply_config(config, lv: int) -> dict:
    """Level 2: the config-level overrides on an upstream model config object, BEFORE ``from_pretrained(..., config=config)``; returns
    what was set (``{}`` below level 2)."""
    applied = {}
    if level(lv) < 2:
        return applied
    for dotted, value in MODEL_CONFIG_OVERRIDES.items():
        obj, parts = config, dotted.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        applied[dotted] = value
    return applied


def apply_env(environ: Optional[dict] = None) -> dict:
    """Set the recipe's environment variables (``setdefault``: a caller's value is kept) and return what was added."""
    environ = os.environ if environ is None else environ
    added = {}
    for k, v in ENV.items():
        if k not in environ:
            environ[k] = v; added[k] = v
    return added


def apply_torch() -> dict:
    """Deterministic algorithms + the scatter switch (when the installed model module has it). Returns what was applied."""
    import torch
    torch.use_deterministic_algorithms(TORCH_DETERMINISTIC["mode"], warn_only=TORCH_DETERMINISTIC["warn_only"])
    out = {"use_deterministic_algorithms": dict(TORCH_DETERMINISTIC), "set_deterministic_scatter": False}
    try:
        import transformers.models.esmfold2.modeling_esmfold2_common as CMN
        if hasattr(CMN, "set_deterministic_scatter"):
            CMN.set_deterministic_scatter(True); out["set_deterministic_scatter"] = True
    except Exception:
        pass
    return out


def fold_overrides(lv: int = 1) -> Dict[str, float]:
    """The recipe's fold keywords (levels 1 and 2 alike; ``{}`` at level 0)."""
    return dict(FOLD_OVERRIDES) if level(lv) else {}
