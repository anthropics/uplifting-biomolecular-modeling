"""The torch side of a kit's deterministic recipe (``--det 1``) under the entry the sequence kits call: a FAÇADE over
``opt_core.precision.recipe`` (the one in-process applier).

``apply(recipe, *, seed=None, seed_scope=..., deterministic_algorithms=None, warn_only=False, cudnn_deterministic=None,
cudnn_benchmark=None, tf32=None, env_accept=None, environ=None)`` applies exactly the fields a kit names and nothing implied (``None`` =
untouched) — seeds per ``seed_scope`` (a seed that cannot be applied is NAMED: ``skipped:numpy-unavailable``,
``skipped:cuda-unavailable``), deterministic algorithms (+ ``warn_only``), the two cuDNN switches, ``tf32`` (one value, both the cuBLAS
matmul and the cuDNN switch) — with the whole ``opt_core.det.Recipe`` held: every ``env`` name carries its value (or an ``env_accept``
alternative), every ``unset`` name is absent, every ``pythonpath`` entry is on ``PYTHONPATH``; a name not in force is put in force when
CUDA is not yet initialised (``env_late``) and is a :data:`DetRecipeError` (= ``opt_core.precision.recipe.RecipeOrderError``) naming it
once CUDA is initialised; a missing ``pythonpath`` entry always is. Returns the applied record a manifest carries: ``{"applied": {<torch
call>: value}, "env", "env_late", "torch_imported_before", "readback": opt_core.seq.numerics.readback(), "applier"}`` — ``applier``
names the module whose statements ran (``opt_core.precision.recipe`` from opt_core 0.3.1, whose ``apply_switches`` this calls;
``opt_core.seq.det_torch`` on an opt_core without it, where the same statements run from this file). Level 0 (production numerics) is
the caller not calling this. torch is imported inside :func:`apply` (``opt_core.seq.numerics.TorchUnavailable`` when it cannot be).
"""
from __future__ import annotations

import os
import random
import sys
from typing import Mapping, MutableMapping, Optional, Sequence

from ..det import Recipe, apply_env
from ..precision import recipe as _recipe
from ..precision.recipe import RecipeOrderError as DetRecipeError
from . import numerics as _numerics

_ROOT_APPLY = getattr(_recipe, "apply_switches", None)          # opt_core >= 0.3.1
SEED_SCOPES = getattr(_recipe, "SEED_SCOPES", ("random", "numpy", "torch"))
SKIPPED_NUMPY = getattr(_recipe, "SKIPPED_NUMPY", "skipped:numpy-unavailable")
SKIPPED_CUDA = getattr(_recipe, "SKIPPED_CUDA", "skipped:cuda-unavailable")
RECORD_KEYS = ("applied", "env", "env_late", "torch_imported_before", "readback", "applier")


def env_missing(recipe: Recipe, environ: Optional[Mapping[str, str]] = None,
                env_accept: Optional[Mapping[str, Sequence[str]]] = None) -> dict:
    """``{name: (present value or None, wanted value or None)}`` for every recipe variable not in force: an ``env`` name without its value
    (an ``env_accept[name]`` value counts as in force) and an ``unset`` name that is present (wanted None)."""
    if hasattr(_recipe, "env_missing"):
        return _recipe.env_missing(recipe, environ, env_accept)
    environ = os.environ if environ is None else environ
    accept = env_accept or {}
    out = {}
    for name, want in recipe.env.items():
        have = environ.get(name)
        if have == want or (have is not None and have in tuple(accept.get(name, ()))):
            continue
        out[name] = (have, want)
    for name in recipe.unset:
        if name in environ:
            out[name] = (environ.get(name), None)
    return out


def pythonpath_missing(recipe: Recipe, environ: Optional[Mapping[str, str]] = None) -> list:
    """The ``recipe.pythonpath`` entries absent from ``PYTHONPATH`` (in the recipe's order)."""
    if hasattr(_recipe, "pythonpath_missing"):
        return _recipe.pythonpath_missing(recipe, environ)
    environ = os.environ if environ is None else environ
    have = [e for e in (environ.get("PYTHONPATH") or "").split(os.pathsep) if e]
    return [e for e in recipe.pythonpath if e not in have]


def apply(recipe: Optional[Recipe] = None, *, seed: Optional[int] = None, seed_scope: Sequence[str] = SEED_SCOPES,
          deterministic_algorithms: Optional[bool] = None, warn_only: bool = False, cudnn_deterministic: Optional[bool] = None,
          cudnn_benchmark: Optional[bool] = None, tf32: Optional[bool] = None,
          env_accept: Optional[Mapping[str, Sequence[str]]] = None, environ: Optional[MutableMapping[str, str]] = None,
          torch=None) -> dict:
    """Apply exactly the given fields (module contract); return the applied record (:data:`RECORD_KEYS`)."""
    environ = os.environ if environ is None else environ
    imported_before = "torch" in sys.modules
    if _ROOT_APPLY is not None:
        full = _ROOT_APPLY(recipe, torch=torch, seed=seed, seed_scope=tuple(seed_scope), deterministic_algorithms=deterministic_algorithms,
                           warn_only=warn_only, cudnn_deterministic=cudnn_deterministic, cudnn_benchmark=cudnn_benchmark, tf32=tf32,
                           env_accept=env_accept, export_env=True, check_order=True, environ=environ)
        rec = {"applied": dict(full["applied"]), "env": dict(full["env"]), "env_late": list(full["env_late"]),
               "torch_imported_before": imported_before, "applier": "opt_core.precision.recipe"}
    else:
        rec = _apply_here(recipe, torch=torch, seed=seed, seed_scope=tuple(seed_scope), deterministic_algorithms=deterministic_algorithms,
                          warn_only=warn_only, cudnn_deterministic=cudnn_deterministic, cudnn_benchmark=cudnn_benchmark, tf32=tf32,
                          env_accept=env_accept, environ=environ)
        rec["torch_imported_before"] = imported_before
        rec["applier"] = "opt_core.seq.det_torch"
    rec["readback"] = _numerics.readback(environ, torch=torch)
    return {k: rec[k] for k in RECORD_KEYS}


def _apply_here(recipe, *, torch, seed, seed_scope, deterministic_algorithms, warn_only, cudnn_deterministic, cudnn_benchmark, tf32,
                env_accept, environ) -> dict:
    """The statements of :func:`apply` for an opt_core whose ``precision.recipe`` has no ``apply_switches`` (same table, same order)."""
    bad_scope = [s for s in seed_scope if s not in SEED_SCOPES]
    if bad_scope:
        raise ValueError("seed_scope %r: names outside %r" % (bad_scope, SEED_SCOPES))
    rec = {"applied": {}, "env": {}, "env_late": []}
    torch = _numerics.require_torch(torch)
    lv = int(recipe.level) if recipe is not None else None
    if recipe is not None:
        no_path = pythonpath_missing(recipe, environ)
        if no_path:
            raise DetRecipeError("recipe PYTHONPATH entries absent %s — an interpreter-start path cannot be added to a running process; "
                                 "export it before the interpreter starts (the kit's command line does)" % (no_path,), det=lv, pythonpath=no_path)
        missing = env_missing(recipe, environ, env_accept)
        if missing:
            if _cuda_initialised(torch):
                raise DetRecipeError("recipe environment not in force after CUDA initialised {name: (present, wanted)} = %s — export it "
                                     "before the process touches CUDA (the kit's command line does), never after" % (missing,), det=lv)
            apply_env(Recipe(level=recipe.level, env={k: w for k, (_h, w) in missing.items() if w is not None},
                             unset=tuple(k for k, (_h, w) in missing.items() if w is None)), environ)
            rec["env_late"] = sorted(missing)
        rec["env"] = {k: environ.get(k) for k in list(recipe.env) + list(recipe.unset)}
    if seed is not None:
        s = int(seed)
        if "random" in seed_scope:
            random.seed(s)
            rec["applied"]["random.seed"] = s
        if "numpy" in seed_scope:
            try:
                import numpy as np  # noqa: PLC0415 — optional, lazy
            except ImportError:
                rec["applied"]["numpy.random.seed"] = SKIPPED_NUMPY          # named, never silent: numpy is not importable here
            else:
                np.random.seed(s)
                rec["applied"]["numpy.random.seed"] = s
        if "torch" in seed_scope:
            torch.manual_seed(s)
            rec["applied"]["torch.manual_seed"] = s
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
                rec["applied"]["torch.cuda.manual_seed_all"] = s
            else:
                rec["applied"]["torch.cuda.manual_seed_all"] = SKIPPED_CUDA  # named: no CUDA device in this process
    if deterministic_algorithms is not None:
        torch.use_deterministic_algorithms(bool(deterministic_algorithms), warn_only=bool(warn_only))
        rec["applied"]["torch.use_deterministic_algorithms"] = {"mode": bool(deterministic_algorithms), "warn_only": bool(warn_only)}
    if cudnn_deterministic is not None:
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
        rec["applied"]["torch.backends.cudnn.deterministic"] = bool(cudnn_deterministic)
    if cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        rec["applied"]["torch.backends.cudnn.benchmark"] = bool(cudnn_benchmark)
    if tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        rec["applied"]["torch.backends.cuda.matmul.allow_tf32"] = bool(tf32)
        rec["applied"]["torch.backends.cudnn.allow_tf32"] = bool(tf32)
    return rec


def _cuda_initialised(torch) -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001
        return False


__all__ = ["DetRecipeError", "RECORD_KEYS", "SEED_SCOPES", "SKIPPED_CUDA", "SKIPPED_NUMPY", "apply", "env_missing", "pythonpath_missing"]
