"""The deterministic recipe (``--det 0|1``), applied identically on every arm — stock-side switches, never a mode.

Level 1 (``LEVEL_WHAT[1]``) is, in the tree's shared recipe shape (opt_core.precision.recipe.torch_recipe -> opt_core.det.Recipe):
  environment   ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (cuBLAS's reproducible workspace, exported BEFORE the interpreter touches CUDA) and
                ``OF3_DETERMINISTIC=1`` (``SWITCH``: the name the fast-inference kit's hook keys its own deterministic block on, so a kit arm
                and the stock arm read one switch);
  statements    ``torch.use_deterministic_algorithms(True)``, cuDNN deterministic on / benchmark off — opt_core.precision.recipe.apply_torch,
                the one implementation: a process this package drives applies them in-process (``apply_inprocess``); the stock child, which the
                package does not drive past its proof, gets them from the det site (``DET_SITE``: ``det_site/sitecustomize.py`` first on its
                PYTHONPATH, run by the interpreter at start-up — the stock proof's one named carve-out, opt_core.stock_proof ``det_exception``);
  graphed arms  ``OF3_GRAPHS_STRICT=1`` re-asserted (a capture failure raises; every graphed line carries it already).
Which runner yaml an arm runs under the recipe is the mode table's (modes.STOCK_DET_YAML for ``off`` / ``exact``: cli.row_yaml), not this
module's. Level 0 is production numerics: nothing is exported or applied.
"""
import os
from typing import Dict, Optional

LEVELS = (0, 1)
LEVEL_WHAT = {0: "off", 1: "deterministic algorithms + cuBLAS workspace :4096:8 + OF3_DETERMINISTIC=1 (+ OF3_GRAPHS_STRICT=1 on graphed arms)"}
SWITCH: Dict[str, str] = {"OF3_DETERMINISTIC": "1"}                    # the recipe's switch (the fast-inference kit's name for its deterministic block; the det site keys on it too)
CUBLAS: Dict[str, str] = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
ENV: Dict[str, str] = {**SWITCH, **CUBLAS}                             # the level-1 words; == recipe(1).env (tests lock the pair against the core's torch_recipe)
GRAPHED_EXTRA: Dict[str, str] = {"OF3_GRAPHS_STRICT": "1"}
DET_SITE = os.path.join("opt", "openfold3_ob0_opt", "det_site")        # relative to the openfold3_ob0/ tree: the directory whose sitecustomize.py applies the statements at interpreter start


def level(value) -> int:
    lv = int(value or 0)
    if lv not in LEVELS:
        raise ValueError(f"--det must be one of {LEVELS} (got {value!r}): 0 = off, 1 = {LEVEL_WHAT[1]}")
    return lv


def det_site(home: str) -> str:
    """The det site directory of the tree at `home` (absolute)."""
    return os.path.join(home, DET_SITE)


def recipe(lv: int, home: Optional[str] = None):
    """The level as the tree's shared recipe shape (opt_core.det.Recipe: level, env, unset, pythonpath, note); the det site is the recipe's one
    PYTHONPATH entry when `home` is given (the stock child's form), none without it (the in-process form)."""
    from opt_core.precision import recipe as _recipe
    return _recipe.torch_recipe(level(lv), switches=SWITCH, det_site=det_site(home) if (home and level(lv) >= 1) else None, note=LEVEL_WHAT[1])


def stock_exception(lv: int, home: str) -> Optional[dict]:
    """The stock proof's carve-out under the recipe (opt_core.det.stock_exception: ``{"env": ENV, "pythonpath": [det site]}``), None at level 0."""
    from opt_core.det import stock_exception as _stock_exception
    return _stock_exception(recipe(lv, home))


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


def apply_inprocess(lv: int, graphed: bool = False, environ: Optional[dict] = None) -> dict:
    """The recipe in a process this package drives (`pred` under exact / fast / big): the variables (apply_env) then the statements
    (opt_core.precision.recipe.apply_torch, which imports torch and refuses by name when CUDA was initialised before the cuBLAS variable was
    exported). Returns ``{"det", "env_added", ...the core's apply_torch record}``; ``{"det": 0}`` at level 0."""
    lv = level(lv)
    if lv < 1:
        return {"det": 0}
    added = apply_env(lv, graphed=graphed, environ=environ)
    from opt_core.precision import recipe as _recipe
    rec = _recipe.apply_torch(recipe(lv), environ=os.environ if environ is None else environ)
    rec["env_added"] = added
    return rec
