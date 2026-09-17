"""The deterministic recipe (`--det 1`), applied identically in every mode, `off` (stock) included.

`CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before the first cuBLAS call, then Protenix's `seed_everything(seed, deterministic=True)` (`protenix/utils/seed.py:22-34`:
random / numpy / torch / torch.cuda seeds, `cudnn.benchmark=False`, `cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG` re-set). `--det 0` is upstream's own call, `seed_everything(seed, deterministic=False)` (`runner/inference.py:236`).
A stock-side switch, never a mode: under `--det 1` stock and the kit modes run the in-process route and differ in the mode flag alone;
`--det 0` keeps upstream's own seeding in every mode.

The recipe's shape is the core's (`opt_core.det.Recipe`, `LEVELS`): `recipe(det)` is the recipe record a caller describes or carves out of the
stock proof (`stock_exception`: exactly the recipe's variables, nothing beyond — `opt_core.stock_proof`). The two functions a stock
process calls — `apply_env` (export the recipe's variables) and `seed` (upstream's own seeding call) — import nothing of the core: a stock
process holds no core module beyond the proof machinery (`opt_core.stock_proof.CORE_ALLOWED_IN_STOCK`).
"""
import os
from typing import Dict, Optional

RECIPE_ENV: Dict[str, str] = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}    # exported before the first cuBLAS call; seed_everything(deterministic=True) re-sets it (protenix/utils/seed.py:22-34)
LEVELS = (0, 1)                                                           # --det values: 0 = production numerics (upstream's seeding), 1 = the kit's recipe
NOTE = "the kit's recipe: CUBLAS_WORKSPACE_CONFIG + seed_everything(seed, deterministic=True) on every arm"


def recipe(det: int):
    """The `opt_core.det.Recipe` of a --det level: level 1 = `RECIPE_ENV`; level 0 = production numerics (an empty recipe)."""
    from opt_core.det import PRODUCTION, Recipe
    if not det:
        return PRODUCTION
    return Recipe(level=1, env=dict(RECIPE_ENV), note=NOTE)


def stock_exception(det: int) -> Optional[dict]:
    """The stock proof's carve-out under the recipe (`opt_core.det.stock_exception`): None at --det 0."""
    from opt_core.det import stock_exception as _stock_exception
    return _stock_exception(recipe(det))


def apply_env(det: int, environ=None) -> Dict[str, str]:
    """Export the recipe's environment (before torch's first cuBLAS call). Returns what was set. Core-free: a stock process calls it."""
    environ = os.environ if environ is None else environ
    if not det:
        return {}
    for k, v in RECIPE_ENV.items():
        environ[k] = v
    return dict(RECIPE_ENV)


def seed(seed_value: int, det: int) -> dict:
    """Seed exactly as upstream does, deterministic only under the recipe (the same stock function on every arm)."""
    from protenix.utils.seed import seed_everything
    seed_everything(seed=int(seed_value), deterministic=bool(det))
    import torch
    return {"seed": int(seed_value), "deterministic": bool(det), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
