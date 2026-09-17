"""The deterministic-recipe shape: what a ``--det <level>`` means, applied and described the same way everywhere in the kit.

Contract. A kit's recipe table is its own (``det.LEVELS`` in the kit: level -> :class:`Recipe`); the core carries the shape and two
rules. :func:`apply_env` exports the recipe's variables (and unsets its ``unset`` names) and returns the previous values;
:func:`stock_exception` is the ONE named carve-out the stock proof allows under the recipe — exactly the recipe's variables with their
values and exactly its PYTHONPATH entries, nothing beyond (stock_proof.env_proof ``det_exception``). Level 0 is production numerics:
an empty recipe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Optional

import os


@dataclass(frozen=True)
class Recipe:
    level: int
    env: Mapping[str, str] = field(default_factory=dict)
    unset: tuple = ()
    pythonpath: tuple = ()
    note: str = ""

    def __post_init__(self):
        object.__setattr__(self, "env", {str(k): str(v) for k, v in dict(self.env).items()})
        object.__setattr__(self, "unset", tuple(self.unset))
        object.__setattr__(self, "pythonpath", tuple(self.pythonpath))


PRODUCTION = Recipe(level=0, note="production numerics")


def apply_env(recipe: Recipe, environ: Optional[MutableMapping[str, str]] = None) -> dict:
    """Export the recipe (module contract). Returns ``{name: previous value or None}`` for every name touched."""
    environ = os.environ if environ is None else environ
    before: dict = {}
    for name in recipe.unset:
        before[name] = environ.get(name)
        environ.pop(name, None)
    for name, value in recipe.env.items():
        before.setdefault(name, environ.get(name))
        environ[name] = value
    if recipe.pythonpath:
        before.setdefault("PYTHONPATH", environ.get("PYTHONPATH"))
        kept = [e for e in (environ.get("PYTHONPATH") or "").split(os.pathsep) if e and e not in recipe.pythonpath]
        environ["PYTHONPATH"] = os.pathsep.join(list(recipe.pythonpath) + kept)
    return before


def describe(recipe: Recipe) -> str:
    """One line: ``det=<level> env=<names> unset=<names> pythonpath=<n entries>`` (+ the note)."""
    names = ",".join(recipe.env) or "none"
    unset = ",".join(recipe.unset) or "none"
    line = f"det={recipe.level} env={names} unset={unset} pythonpath={len(recipe.pythonpath)}"
    return line + (f" ({recipe.note})" if recipe.note else "")


def stock_exception(recipe: Recipe) -> Optional[dict]:
    """``{"env": {...}, "pythonpath": [...]}`` for the stock proof, or None at level 0 / an empty recipe."""
    if not recipe.env and not recipe.pythonpath:
        return None
    return {"env": dict(recipe.env), "pythonpath": list(recipe.pythonpath)}
