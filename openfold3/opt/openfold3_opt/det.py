"""The deterministic recipe (``--det 0|1``), applied identically in every arm — stock-side switches, never a mode.

Level 1 is the fast-inference kit's reference configuration, which every add-on honours by the same spelling:
  environment   OF3_DETERMINISTIC=1 (the hook then sets CUBLAS_WORKSPACE_CONFIG=:4096:8 and torch.use_deterministic_algorithms(True);
                registry.AIDS) plus the explicit export CUBLAS_WORKSPACE_CONFIG=:4096:8 (fast_inference/README.md),
                both set BEFORE torch is imported;
  graphed arms  OF3_GRAPHS_STRICT=1 (capture failure raises; every graphed line of the mode table carries it already — the recipe
                re-asserts it on any graphed arm so a caller's override cannot drop it);
  every arm     --runner-yaml <the mode's configuration> (always, det or not: cli.row_yaml — the stock configuration unless the
                caller names a yaml, modes.STOCK_YAML), one query per call, upstream's default
                DataLoader workers (num_workers: 0 changes the featurisation RNG stream).
The stock arm under the recipe carries the kit's ``of3_levers`` on sys.path with no lever switch set and executes the kit hook — exactly
the kit's own deterministic stock arm — so the same code sets the same torch state; the hook installs no
finder without a lever switch (sitecustomize.py). Stock run-to-run equality under the recipe is the add-ons' own finding (bitwise on
one GPU model + stack), not a torch guarantee; the recipe costs run time.
"""
import os
from typing import Dict, Optional

from . import modes

LEVELS = (0, 1)
LEVEL_WHAT = {0: "off", 1: "the kit's deterministic reference: OF3_DETERMINISTIC=1 + CUBLAS_WORKSPACE_CONFIG=:4096:8 (+ OF3_GRAPHS_STRICT=1 on graphed arms)"}
KIT_SWITCH: Dict[str, str] = {"OF3_DETERMINISTIC": "1"}                # the fast-inference kit's own switch
ENV: Dict[str, str] = {"OF3_DETERMINISTIC": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}   # the level-1 words; == recipe(1).env (the core's torch recipe of the kit switch — tests lock the pair)
GRAPHED_EXTRA: Dict[str, str] = {"OF3_GRAPHS_STRICT": "1"}
KIT_HOOK_RELPATH = os.path.join(modes.HOOK_DIR["fast_inference"], modes.HOOK_FILE)


def level(value) -> int:
    lv = int(value or 0)
    if lv not in LEVELS:
        raise ValueError(f"--det must be one of {LEVELS} (got {value!r}): 0 = off, 1 = the kit's deterministic reference")
    return lv


def recipe(lv: int):
    """The level as the tree's shared recipe shape (opt_core.precision.recipe.torch_recipe → opt_core.det.Recipe: level, env, unset,
    pythonpath, note). Imported here, not at module level: the stock caller imports this module for `level` and the carrier and must hold
    no core module beyond its proof (env.env_proof)."""
    from opt_core.precision import recipe as _recipe
    return _recipe.torch_recipe(level(lv), switches=KIT_SWITCH, note=LEVEL_WHAT[1])


def describe(lv: int) -> str:
    """The level's recipe in the core's words (opt_core.det.describe: `det=<level> env=… unset=… pythonpath=… (<note>)`)."""
    from opt_core.det import describe as _describe
    return _describe(recipe(lv))


def apply_env(lv: int, graphed: bool = False, environ: Optional[dict] = None) -> dict:
    """Set the recipe's variables (``setdefault``: a caller's value is kept) and return what was added; nothing below level 1."""
    environ = os.environ if environ is None else environ
    added = {}
    if level(lv) < 1:
        return added
    items = dict(ENV)
    if graphed:
        items.update(GRAPHED_EXTRA)
    for k, v in items.items():
        if environ.get(k) is None:
            environ[k] = v
            added[k] = v
    return added


def carrier(home: str) -> str:
    """The kit hook file the stock arm executes under the recipe (its OF3_DETERMINISTIC block; no lever switch set -> no finder)."""
    return os.path.join(home, modes.KITS["fast_inference"], KIT_HOOK_RELPATH)


def stock_arm_carrier(home: str) -> dict:
    """For the stock caller under --det 1: put the kit's of3_levers on sys.path and execute its hook (the kit's deterministic stock arm)."""
    import runpy
    import sys
    d = os.path.dirname(carrier(home))
    if d not in sys.path:
        sys.path.insert(0, d)
    runpy.run_path(carrier(home), run_name="openfold3_opt_det_carrier")
    finders = [type(f).__name__ for f in sys.meta_path if type(f).__name__ == "_LeverFinder"]
    return {"carrier": carrier(home), "lever_finder_installed": bool(finders)}
