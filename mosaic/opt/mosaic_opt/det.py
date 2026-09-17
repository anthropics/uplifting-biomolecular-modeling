"""The deterministic recipe: the settings the kit applies to every arm, stock included, so that two runs of one shape and seed see the same
inputs and the same process environment.

The recipe (the owner of every value is named beside it):
  (1) frozen features — `--features-in NPZ --features-sha SHA`, the npz written once by the stock arm A (`--features-out`) and loaded by
      every other arm of the same shape: Boltz-2 featurization re-draws `ref_pos` in every process (`frozen.py:1-4`); as a lever this is
      P3, as recipe it is "identical inputs" (`inputs.py` names the shape's file);
  (2) explicit PRNG keys — `x0 = softmax(0.5·gumbel(key(seed), (L, 20)))`, stage keys `fold_in(key(seed), 1|2)`, refold `key(0)` — the
      recipe's own, on every arm (`tools/recipe.py` `x0`, `stage_key`, `REFOLD_KEY`, called by the driver; not a switch);
  (3) `recycling_steps=1, sampling_steps=25, deterministic=True` asserted by the driver (`tools/recipe.py` `BOLTZ2_LOSS`, asserted in `build_loss`; not a switch);
  (4) the global numeric state snapshotted before and after and asserted unchanged (`numstate.py`; the driver's `numstate.snapshot()` before the model load and `numstate.diff` after the refold; not a switch);
  (5) the stdio setting exported for every arm under ``--det 1`` — `PYTHONUNBUFFERED=1` (``ENV`` below, applied by every route of the
      package including the stock subprocess; ``--det 0``, the default, exports nothing). The JAX allocator is the library's default on
      every arm at every level — its preallocated device pool, exactly as stock runs: growth mode (`XLA_PYTHON_CLIENT_PREALLOCATE=false`)
      fragments the pool and ends a design that fits under preallocation with RESOURCE_EXHAUSTED, so the kit never sets it;
  (6) `JAX_ENABLE_X64` / `JAX_DEFAULT_MATMUL_PRECISION` unset (the kit never changes precision) — refused on every arm.
Under this recipe a process reproduces its own designs run to run; two fresh STOCK processes are NOT bitwise by XLA
design (every fresh compile re-autotunes) — that is what P1 (the persistent compilation cache + autotune pin) adds.
"""
import os
from typing import Dict, Optional, Tuple

from opt_core.det import Recipe, describe as _describe

LEVELS: Tuple[int, ...] = (0, 1)
DEFAULT_LEVEL = 0                                                                             # off unless asked for: `--det 1` applies the recipe's process settings
ENV: Dict[str, str] = {"PYTHONUNBUFFERED": "1"}                                               # every arm's stdio setting under --det 1; no allocator variable at any level
MUST_BE_UNSET: Tuple[str, ...] = ("JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION")           # refused on every arm: the kit never changes precision
RECIPES: Dict[int, Recipe] = {                                                                # the levels in the shared core's recipe shape (opt_core.det): what each SETS;
    0: Recipe(0, note="the library's allocator and stdio defaults (nothing exported)"),       # MUST_BE_UNSET are refusals on every arm (precision_refusal), never
    1: Recipe(1, env=ENV, note="the kit's stdio setting for every arm; a value the caller set stands"),   # silent unsets, so no
}                                                                                             # level lists them under `unset`


def describe(lv: int) -> str:
    """One line naming what `--det <lv>` sets (the core's wording over RECIPES)."""
    return _describe(RECIPES[level(lv)])


def level(value) -> int:
    try:
        lv = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"--det expects one of {LEVELS}, got {value!r}")
    if lv not in LEVELS:
        raise ValueError(f"--det expects one of {LEVELS}, got {value!r}")
    return lv


def apply_env(env: dict, lv: int) -> dict:
    """The recipe's environment at `lv` on top of `env` (a copy): the level's ``RECIPES[lv].env`` where the caller has not set a value (the
    caller's value stands: a set value is kept); level 0 sets nothing. Returns the new environment."""
    out = dict(env)
    for k, v in RECIPES[level(lv)].env.items():
        out.setdefault(k, v)
    return out


def precision_refusal(env: Optional[dict] = None) -> Optional[str]:
    """The kit's refusal for every arm: a set JAX_ENABLE_X64 / JAX_DEFAULT_MATMUL_PRECISION."""
    env = os.environ if env is None else env
    bad = [k for k in MUST_BE_UNSET if env.get(k)]
    if bad:
        return (f"{' / '.join(bad)} must be unset (the kit never changes precision): "
                f"unset {' '.join(bad)} and re-run")
    return None
