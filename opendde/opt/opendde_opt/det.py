"""The deterministic recipe — stock-side switches applied to every arm, stock included; never a mode.

Level 1 (`--det 1`): `--deterministic true` on the stock CLI (upstream `runner/batch_inference.py:778-783` -> `configs.deterministic`;
the runner applies the process-wide switches at construction, before CUDA initialises (`runner/inference.py:91-107,1001`
`_apply_determinism_runtime`: `torch.use_deterministic_algorithms(True)`, cudnn.deterministic on, cudnn.benchmark off,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`), re-applies them per seed (`runner/inference.py:1644,1683` `seed_everything(seed, deterministic=True)`;
`opendde/utils/seed.py:10-23`) and restores the previous process state only at `Runner.close()` (`runner/inference.py:1184-1195`) — so the
algorithms are deterministic for the whole of inference — plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before python (the one upstream name
the recipe sets: stock/PINS.json "allowed_under_det"). Under deterministic algorithms upstream's single-GPU MSA module takes its
deterministic outer-product-mean path (`opendde/model/modules/pairformer.py:1702-1714`: a fixed 2x2 tile schedule whose per-tile transient
is (N/2)^2 x c_hidden^2 elements, larger than the default chunked path's) — a recipe run can reach its memory ceiling at a smaller N than a
default run.
Without the recipe the trunk is bitwise across processes and the sampler's scatter_add_ atomics are the run-to-run floor; equality
comparisons use level 1 on both sides.
"""
from __future__ import annotations

from opt_core import det as _core_det
from opt_core.precision import recipe as _recipe

LEVELS = (0, 1)
CUBLAS = _recipe.CUBLAS_WORKSPACE                            # ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"): the core's one spelling
NOTE = "upstream --deterministic true (torch deterministic algorithms, cuDNN deterministic) + cuBLAS workspace"


def check_level(level) -> int:
    lv = int(level or 0)
    if lv not in LEVELS:
        raise ValueError(f"unknown --det {level!r}; expected one of {LEVELS}")
    return lv


def cli_args(level: int) -> list[str]:
    return ["--deterministic", "true"] if check_level(level) else []


def recipe(level: int) -> _core_det.Recipe:
    """The recipe of a level in the core's shape (``opt_core.det.Recipe`` built by ``opt_core.precision.recipe.torch_recipe``):
    level 0 = production numerics; level 1 = the cuBLAS workspace. The in-process statements (deterministic algorithms, cuDNN) are upstream's own under
    ``--deterministic true`` (``cli_args``), applied identically on both arms."""
    lv = check_level(level)
    return _recipe.torch_recipe(lv, switches={}, note=NOTE)


def env(level: int) -> dict:
    """The exports of the recipe: CUBLAS."""
    return dict(recipe(level).env)


def describe(level: int) -> dict:
    lv = check_level(level)
    r = recipe(lv)
    return {"level": lv, "cli_args": cli_args(lv), "env": dict(r.env), "recipe": _core_det.describe(r),
            "note": ("stock-side switches applied to every arm under the recipe" if lv else "no deterministic recipe")}
