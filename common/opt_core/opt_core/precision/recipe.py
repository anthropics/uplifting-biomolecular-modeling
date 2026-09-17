"""Deterministic recipes rendered into the core's shape (``opt_core.det.Recipe``), and the in-process statements of a level.

Contract. A kit's ``--det <level>`` table stays the kit's (``det.LEVELS``); this module builds its entries from two shared forms
so the words cannot drift between the kit driver, the stock caller and ``run.sh``:

* :func:`torch_recipe` — level 0 = ``opt_core.det.PRODUCTION`` (production numerics, nothing set); level >= 1 = the cuBLAS workspace
  variable (``CUBLAS_WORKSPACE_CONFIG=:4096:8``, which must be in the environment BEFORE the process initialises cuBLAS), the kit's own
  switches (``{"<KIT>_DETERMINISTIC": "1"}``, ``{"<KIT>_DET_SCATTER": "1"}`` — names are the kit's, passed in), and an optional det site
  directory first on ``PYTHONPATH`` (a ``sitecustomize.py`` that applies the statements in a stock interpreter the kit does not drive).
* :func:`apply_switches` — THE in-process torch applier (the only one): exactly the switches a kit names and nothing implied
  (``None`` = untouched) — ``seed`` (random / numpy / torch(+cuda) per ``seed_scope``; a seed that cannot be applied is NAMED in the
  record, ``skipped:numpy-unavailable`` / ``skipped:cuda-unavailable``, never dropped), ``deterministic_algorithms`` (+ ``warn_only``),
  ``cudnn_deterministic``, ``cudnn_benchmark``, ``tf32`` (one value, BOTH the cuBLAS matmul and the cuDNN switch) — with the whole
  ``opt_core.det.Recipe`` held, not assumed: every ``env`` name carries its value (or an ``env_accept`` alternative), every ``unset`` name
  is absent, every ``pythonpath`` entry is on ``PYTHONPATH``. A name not in force is put in force when CUDA is not yet initialised and
  ``export_env`` (recorded under ``env_late``), only recorded (``env_missing``) when not ``export_env``, and a :class:`RecipeOrderError`
  naming it once CUDA is initialised (such settings are read at context creation — a late export would be a label, not a recipe;
  ``order_sensitive`` scopes that refusal to the names read at CUDA initialisation); a missing ``pythonpath`` entry is a
  :class:`RecipeOrderError` (an interpreter-start path cannot be retrofitted) unless ``hold_pythonpath=False`` records it instead (a driver
  whose det site is for the child it launches). Returns the applied record: ``{"applied": {<torch call>: value}, "env", "env_late",
  "env_missing", "pythonpath_missing", "torch_imported_before", "before", "readback", "changed"}``
  — ``before``/``readback`` are ``dict(policy.numerics_signature())`` + ``cublas_workspace`` around the calls, ``changed`` the fields that moved.
* :func:`apply_torch` — what a ``--det`` LEVEL means, over :func:`apply_switches`: level >= 1 = ``deterministic_algorithms=True`` (+
  ``warn_only``) and, when ``cudnn``, cuDNN deterministic on / timing run off, with the order check scoped to the cuBLAS workspace variable
  (``check_order``) and the det site recorded, not enforced; level 0 sets nothing and returns the record of what it found. Its record is the level's: ``{"det", "deterministic_algorithms", "warn_only",
  "cudnn_deterministic", "cudnn_benchmark", "cublas_workspace", "changed"}``.

A recipe applies identically to BOTH arms of an equality row (it is the row's definition, never a lever); its statements change
kernels' reduction orders, so numbers under a recipe are the recipe's, and its cost belongs in the kit's CHANGES (it is never a
production default). Activation evidence: :func:`apply_torch` returns ``{"det", "deterministic_algorithms", "warn_only",
"cudnn_deterministic", "cudnn_benchmark", "cublas_workspace", "changed"}``; ``opt_core.det.describe(recipe)`` is the env side.
"""
from __future__ import annotations

import os
import random
import sys
from typing import Mapping, MutableMapping, Optional, Sequence

from .. import det as _det
from . import PrecisionError, require_torch

CUBLAS_WORKSPACE = ("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # the value every deterministic torch recipe in the tree uses
SEED_SCOPES = ("random", "numpy", "torch")
SKIPPED_NUMPY = "skipped:numpy-unavailable"
SKIPPED_CUDA = "skipped:cuda-unavailable"


class RecipeOrderError(PrecisionError):
    """Level >= 1 applied in a process whose CUDA context already exists without the cuBLAS workspace variable exported."""

    event = "det_recipe_order"


class RecipeLevelError(PrecisionError, ValueError):
    """A level outside the kit's table."""

    event = "det_level_refused"


def level(lv, levels: Sequence[int] = (0, 1)) -> int:
    """``int(lv)`` when it is one of ``levels``, else :class:`RecipeLevelError` naming the table (the kit's ``--det`` argument check)."""
    try:
        v = int(lv)
    except (TypeError, ValueError):
        raise RecipeLevelError("--det %r: levels are %s" % (lv, tuple(levels)), levels=tuple(levels))
    if v not in tuple(levels):
        raise RecipeLevelError("--det %r: levels are %s" % (lv, tuple(levels)), levels=tuple(levels))
    return v


def torch_recipe(lv: int, switches: Optional[Mapping[str, str]] = None, det_site: Optional[str] = None,
                 extra_env: Optional[Mapping[str, str]] = None, cublas: bool = True, unset: Sequence[str] = (),
                 note: str = "deterministic algorithms + cuBLAS workspace") -> _det.Recipe:
    """The torch-engine recipe of a level in the core's shape. Level 0 -> ``det.PRODUCTION``. Level >= 1 -> env = cuBLAS workspace (when
    ``cublas``) + ``switches`` (the kit's own variable names and values) + ``extra_env``; ``det_site`` first on PYTHONPATH when given."""
    if int(lv) == 0:
        return _det.PRODUCTION
    env: dict = {}
    if cublas:
        env[CUBLAS_WORKSPACE[0]] = CUBLAS_WORKSPACE[1]
    for src in (switches, extra_env):
        if src:
            env.update({str(k): str(v) for k, v in dict(src).items()})
    return _det.Recipe(level=int(lv), env=env, unset=tuple(unset), pythonpath=((det_site,) if det_site else ()), note=note)


def apply_env(recipe: _det.Recipe, environ: Optional[MutableMapping[str, str]] = None) -> dict:
    """``opt_core.det.apply_env`` (re-exported so an adapter imports one module): export the recipe, return the previous values."""
    return _det.apply_env(recipe, environ)


def _level_of(recipe_or_level) -> int:
    return int(recipe_or_level.level) if isinstance(recipe_or_level, _det.Recipe) else int(recipe_or_level)


def env_missing(recipe: _det.Recipe, environ: Optional[Mapping[str, str]] = None,
                env_accept: Optional[Mapping[str, Sequence[str]]] = None) -> dict:
    """``{name: (present value or None, wanted value or None)}`` for every recipe variable not in force: an ``env`` name without its value
    (an ``env_accept[name]`` value counts as in force) and an ``unset`` name that is present (wanted None)."""
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


def pythonpath_missing(recipe: _det.Recipe, environ: Optional[Mapping[str, str]] = None) -> list:
    """The ``recipe.pythonpath`` entries absent from ``PYTHONPATH`` (in the recipe's order)."""
    environ = os.environ if environ is None else environ
    have = [e for e in (environ.get("PYTHONPATH") or "").split(os.pathsep) if e]
    return [e for e in recipe.pythonpath if e not in have]


def _state(torch, environ) -> dict:
    from . import policy                                  # noqa: PLC0415 — sibling, light; kept out of this module's import line
    out = dict(policy.numerics_signature(torch))
    out["cublas_workspace"] = environ.get(CUBLAS_WORKSPACE[0])
    return out


def apply_switches(recipe: Optional[_det.Recipe] = None, *, torch=None, seed: Optional[int] = None,
                   seed_scope: Sequence[str] = SEED_SCOPES, deterministic_algorithms: Optional[bool] = None, warn_only: bool = False,
                   cudnn_deterministic: Optional[bool] = None, cudnn_benchmark: Optional[bool] = None, tf32: Optional[bool] = None,
                   env_accept: Optional[Mapping[str, Sequence[str]]] = None, export_env: bool = True, check_order: bool = True,
                   hold_pythonpath: bool = True, order_sensitive: Optional[Sequence[str]] = None,
                   environ: Optional[MutableMapping[str, str]] = None) -> dict:
    """THE in-process torch applier (module contract): exactly the switches named, the recipe held, the applied record returned.
    ``hold_pythonpath``: a missing ``recipe.pythonpath`` entry is a :class:`RecipeOrderError` (default — a stock interpreter the recipe must
    hold) or, with ``False``, only recorded under ``pythonpath_missing`` (a kit driver whose det site is for the stock CHILD it launches).
    ``order_sensitive``: the recipe variables that are read at CUDA initialisation and therefore refused once CUDA is initialised — ``None`` =
    every recipe variable (default); a tuple (e.g. ``(CUBLAS_WORKSPACE[0],)``) scopes the refusal to those names and records the other
    missing names under ``env_missing`` / ``env_late``."""
    environ = os.environ if environ is None else environ
    bad_scope = [s for s in seed_scope if s not in SEED_SCOPES]
    if bad_scope:
        raise RecipeLevelError("seed_scope %r: names outside %r" % (bad_scope, SEED_SCOPES), seed_scope=list(bad_scope))
    rec = {"applied": {}, "env": {}, "env_late": [], "env_missing": [], "pythonpath_missing": [], "torch_imported_before": "torch" in sys.modules}
    torch = require_torch(torch)
    lv = int(recipe.level) if recipe is not None else None
    if recipe is not None:
        no_path = pythonpath_missing(recipe, environ)
        if no_path and hold_pythonpath:
            raise RecipeOrderError("recipe PYTHONPATH entries absent %s — an interpreter-start path cannot be added to a running process; "
                                   "export it before the interpreter starts (the kit's command line does)" % (no_path,), det=lv, pythonpath=no_path)
        rec["pythonpath_missing"] = list(no_path)
        missing = env_missing(recipe, environ, env_accept)
        if missing:
            sensitive = dict(missing) if order_sensitive is None else {k: v for k, v in missing.items() if k in tuple(order_sensitive)}
            if sensitive and check_order and _cuda_initialised(torch):
                raise RecipeOrderError("recipe environment not in force after CUDA initialised {name: (present, wanted)} = %s — export it "
                                       "before the process touches CUDA (the kit's command line does), never after" % (sensitive,),
                                       det=lv, cublas_workspace=str(environ.get(CUBLAS_WORKSPACE[0])))
            if export_env:
                _det.apply_env(_det.Recipe(level=recipe.level, env={k: w for k, (_h, w) in missing.items() if w is not None},
                                           unset=tuple(k for k, (_h, w) in missing.items() if w is None)), environ)
                rec["env_late"] = sorted(missing)
            else:
                rec["env_missing"] = sorted(missing)
        rec["env"] = {k: environ.get(k) for k in list(recipe.env) + list(recipe.unset)}
    rec["before"] = _state(torch, environ)
    if seed is not None:
        s = int(seed)
        if "random" in seed_scope:
            random.seed(s)
            rec["applied"]["random.seed"] = s
        if "numpy" in seed_scope:
            try:
                import numpy as np                        # noqa: PLC0415 — optional, lazy
            except ImportError:
                rec["applied"]["numpy.random.seed"] = SKIPPED_NUMPY
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
                rec["applied"]["torch.cuda.manual_seed_all"] = SKIPPED_CUDA
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
    rec["readback"] = _state(torch, environ)
    rec["changed"] = [k for k in sorted(rec["readback"]) if rec["readback"][k] != rec["before"].get(k)]
    return rec


def apply_torch(recipe_or_level, torch=None, warn_only: bool = False, cudnn: bool = True, check_order: bool = True,
                environ: Optional[Mapping[str, str]] = None) -> dict:
    """What a ``--det`` LEVEL means in a torch process the kit drives, over :func:`apply_switches` (module contract). Takes an
    ``opt_core.det.Recipe`` or a bare level (a bare level >= 1 stands for the cuBLAS-workspace recipe, :func:`torch_recipe`). It does NOT
    export environment variables (they belong in the environment before the interpreter touches CUDA: ``opt_core.det.apply_env`` in the
    launcher, or :func:`apply`). The order check is the cuBLAS workspace variable's: at level >= 1 it must already be exported when CUDA
    is initialised, else :class:`RecipeOrderError` (``check_order=False`` records instead of raising); the kit's own switches and the det
    site on ``PYTHONPATH`` are the stock child's and are recorded (``apply_switches`` ``env_missing`` / ``pythonpath_missing``), never
    enforced against the driver's own process."""
    environ = os.environ if environ is None else environ
    lv = _level_of(recipe_or_level)
    rcp = recipe_or_level if isinstance(recipe_or_level, _det.Recipe) else (torch_recipe(lv) if lv >= 1 else None)
    on = lv >= 1
    full = apply_switches(rcp, torch=torch, deterministic_algorithms=(True if on else None), warn_only=warn_only,
                          cudnn_deterministic=(True if (on and cudnn) else None), cudnn_benchmark=(False if (on and cudnn) else None),
                          export_env=False, check_order=check_order, hold_pythonpath=False, order_sensitive=(CUBLAS_WORKSPACE[0],),
                          environ=environ)
    rb = full["readback"]
    ws = rb.get("cublas_workspace")
    rec = {"det": lv, "deterministic_algorithms": bool(rb["deterministic_algorithms"]), "warn_only": bool(rb["warn_only"]),
           "cudnn_deterministic": bool(rb["cudnn_deterministic"]), "cudnn_benchmark": bool(rb["cudnn_benchmark"]),
           "cublas_workspace": ws if ws is not None else "unset"}
    rec["changed"] = [k for k in full["changed"] if k in ("deterministic_algorithms", "warn_only", "cudnn_deterministic", "cudnn_benchmark")]
    return rec


def apply(recipe: _det.Recipe, torch=None, warn_only: bool = False, cudnn: bool = True,
          environ: Optional[MutableMapping[str, str]] = None) -> dict:
    """Environment AND statements in one call, for a process the kit itself starts early enough: exports the recipe
    (``opt_core.det.apply_env``) — refused with :class:`RecipeOrderError` when CUDA is already initialised and the export would change the
    cuBLAS workspace variable — then :func:`apply_torch`. Returns the apply_torch record + ``env=<names exported>``."""
    torch = require_torch(torch)
    environ = os.environ if environ is None else environ
    lv = _level_of(recipe)
    if lv >= 1 and CUBLAS_WORKSPACE[0] in recipe.env and environ.get(CUBLAS_WORKSPACE[0]) != recipe.env[CUBLAS_WORKSPACE[0]] \
            and _cuda_initialised(torch):
        raise RecipeOrderError("--det %d: CUDA is initialised before the recipe's %s could be exported" % (lv, CUBLAS_WORKSPACE[0]),
                               det=lv, cublas_workspace=str(environ.get(CUBLAS_WORKSPACE[0])))
    _det.apply_env(recipe, environ)
    rec = apply_torch(recipe, torch=torch, warn_only=warn_only, cudnn=cudnn, check_order=True, environ=environ)
    rec["env"] = sorted(recipe.env) or None
    return rec


def _cuda_initialised(torch) -> bool:
    try:
        cuda = torch.cuda
    except AttributeError:                               # a build without the cuda module answers False; anything else propagates
        return False
    return bool(cuda.is_available() and cuda.is_initialized())
