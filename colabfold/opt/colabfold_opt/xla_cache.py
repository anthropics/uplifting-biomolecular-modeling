"""XLA_CACHE — the compile-cache deployment lever (strategy F3.jit_cache_keyed): JAX's persistent compilation cache under a directory
keyed by the stack AND by the numerics recipe (``<root>/<stack key>/<recipe>/jax``, stack key = stack.stack_key, e.g.
``jax0.5.3-cu12.9-sm90``; recipe = :func:`recipe`: ``det`` when the caller's ``XLA_FLAGS`` carry XLA's deterministic flags
(``--xla_gpu_deterministic_ops[=true]`` or ``--xla_gpu_autotune_level=0``), else ``default``), so a process finds the executables an
earlier process on the same stack and recipe compiled instead of compiling them again (colabfold_batch compiles every model for every new
input shape).

Why the recipe is part of the key. jax >= 0.4.36 couples XLA's per-fusion AUTOTUNE results cache to the compilation-cache
directory (config ``jax_persistent_cache_enable_xla_caches``, default ``xla_gpu_per_fusion_autotune_cache_dir``: the results live in
``<cache dir>/xla_gpu_per_fusion_autotune_cache_dir``), and XLA keys those results by fusion and device, NOT by XLA flags: a process
compiling under ``--xla_gpu_autotune_level=0`` LOADS the tuned kernel choices a default-numerics process stored there and no longer produces
the bytes of a process that compiled alone (``exact`` under the deterministic flags on a directory default-numerics runs populated is not
byte-equal to a cache-free deterministic run; on a fresh directory it is; stock itself under those flags on such a directory moves the
same way — the hazard is jax's, for any process). Hence, for the directory THE KIT places, under recipe ``det``:
(1) the directory is the recipe's (``…/det/jax``: nothing a ``default`` process wrote is under it), and (2) the fence — the process
neither loads nor stores XLA's autotune results there: ``jax_persistent_cache_enable_xla_caches`` = ``none`` in-process (``jax.config``)
and ``JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`` exported for child processes. The executables themselves stay cached under either
recipe (jax keys them by program, flags, jaxlib and device). Under recipe ``default`` nothing but the directory name changes
(``…/default/jax``; jax's coupling stays — a tuned result reloaded is the point of the cache there).

Placement, by the environment the activation sees:
  * ``JAX_COMPILATION_CACHE_DIR`` already set (by the caller): KEPT — the directory is the caller's as given, under either recipe, and
    the kit sets nothing beside it (the line says ``source=kept recipe=<word> xla_caches=<the caller's JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES
    | jax_default>``): ``--mode off`` = stock reads that directory exactly as the kit's process does, so ``exact`` = stock holds on it
    whatever it holds (on a contaminated kept directory both move alike; fencing the kit's side alone would break that equality). For bytes equal to a cache-free deterministic run, leave it unset (the keyed root) or
    keep a directory only deterministic runs use (README Notes).
  * else ``COLABFOLD_OPT_JIT_ROOT`` set (configs/h100.env gives it a default): KEYED — ``opt_core.capture.xla_cache.persistent_cache_env``
    names ``<root>/<stack key>/<recipe>/jax`` and the thresholds that make every compilation eligible; they are exported for child processes
    and applied to this process's jax through ``xla_cache.enable_persistent_cache`` (``jax.config``); the line says ``source=keyed``.
  * else: SKIPPED by name (``reason=no_cache_root``) — the run compiles as stock does.
A core without ``opt_core.capture`` is a skip by name too (``reason=opt_core.capture_unavailable``), never a silent local substitute.
Numerics: none — an executable loaded from the cache is the one that was compiled, and under ``det`` no autotune result crosses processes.
Evidence: the LEVER line at exit (``name=XLA_CACHE state=on source=kept|keyed recipe=det|default root=<root|caller> dir=<dir>
xla_caches=none|jax_default|<caller's word> entries=<files under dir at exit>``) and the activation report's ``xla_cache``.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

from opt_core.oom import is_oom

from . import modes as _modes

NAME = "XLA_CACHE"
ROOT_ENV = _modes.ENV + "_JIT_ROOT"                 # COLABFOLD_OPT_JIT_ROOT
PRESET_ENV = "JAX_COMPILATION_CACHE_DIR"
SUBDIR = "jax"
XLA_FLAGS_ENV = "XLA_FLAGS"
RECIPES = ("default", "det")                        # the directory words under <stack key>/
DET_OPS_FLAG, DET_AUTOTUNE_FLAG = "--xla_gpu_deterministic_ops", "--xla_gpu_autotune_level"   # XLA's deterministic recipe (README Notes): either word makes the recipe `det`
XLA_CACHES_ENV, XLA_CACHES_OPTION = "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "jax_persistent_cache_enable_xla_caches"
XLA_CACHES_FENCE, XLA_CACHES_DEFAULT = "none", "jax_default"    # the LEVER line's xla_caches= words: fenced (the keyed det directory) | jax's own default (coupled) — or the caller's own JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES value, reported as set
ROOT_KEPT = "caller"                                # the LEVER line's root= word when the caller's JAX_COMPILATION_CACHE_DIR is kept (no kit root is used)

_STATE: Dict[str, object] = {"enabled": False, "state": "off", "reason": None, "detail": None, "source": None, "dir": None,
                             "root": None, "recipe": None, "xla_caches": None}   # reason: a blank-free token (the LEVER line's); detail: its text


def recipe(xla_flags: Optional[str]) -> str:
    """The numerics recipe word from the caller's ``XLA_FLAGS`` text: ``det`` when it carries ``--xla_gpu_deterministic_ops`` (bare, or
    ``=true`` / ``=1``) or ``--xla_gpu_autotune_level=0``; ``default`` otherwise (unset, empty, ``--xla_gpu_deterministic_ops=false``,
    another autotune level, unrelated flags)."""
    for word in (xla_flags or "").split():
        name, eq, value = word.partition("=")
        value = value.strip().lower()
        if name == DET_OPS_FLAG and (not eq or value in ("true", "1")):
            return "det"
        if name == DET_AUTOTUNE_FLAG and eq and value == "0":
            return "det"
    return "default"


def _fence(rec: str) -> Dict[str, str]:
    """The child-process environment of the fence: under `det` XLA's caches coupled to the jax cache directory are off; nothing under `default`."""
    return {XLA_CACHES_ENV: XLA_CACHES_FENCE} if rec == "det" else {}


def _fence_in_process(rec: str) -> None:
    """Apply the fence to this process's jax for the KEYED det directory (jax imported already: it read its environment at import, so the
    switch is jax.config). A jax without the option (< 0.4.36) has no coupling to fence — nothing to do, by name on the line (xla_caches=uncoupled)."""
    if rec != "det" or "jax" not in sys.modules:
        return
    import jax  # noqa: PLC0415 — in sys.modules already
    try:
        jax.config.update(XLA_CACHES_OPTION, XLA_CACHES_FENCE)
    except AttributeError:                                             # no such option: this jax couples no XLA cache to the directory
        _STATE["xla_caches"] = "uncoupled"


def apply(stack_key: str, environ=None) -> dict:
    """Place the cache for this process and its children (module docstring); returns the lever's state. Idempotent."""
    env = os.environ if environ is None else environ
    preset, root = env.get(PRESET_ENV), env.get(ROOT_ENV)
    rec = recipe(env.get(XLA_FLAGS_ENV))
    _STATE.update(recipe=rec, xla_caches=env.get(XLA_CACHES_ENV) or XLA_CACHES_DEFAULT, root=None, dir=None, source=None)
    if preset:                                                           # the caller's directory, as given, under either recipe: nothing set beside it (stock reads it the same way)
        _STATE.update(enabled=True, state="on", reason=None, source="kept", dir=preset, root=ROOT_KEPT)
        return dict(_STATE)
    if not root:
        _STATE.update(enabled=False, state="skipped", reason="no_cache_root")
        return dict(_STATE)
    try:
        from opt_core.capture import xla_cache as _core_xla_cache
    except ImportError:
        _STATE.update(enabled=False, state="skipped", reason="opt_core.capture_unavailable")
        return dict(_STATE)
    cache_env = _core_xla_cache.persistent_cache_env(root, stack_key, os.path.join(rec, SUBDIR))   # <root>/<stack key>/<recipe>/jax
    directory = cache_env[PRESET_ENV]
    cache_env.update(_fence(rec))
    if "jax" in sys.modules:                                             # jax read its environment at import: the in-process switch is jax.config
        try:
            _core_xla_cache.enable_persistent_cache(directory)
        except Exception as e:  # noqa: BLE001 — the core names the missing option (XlaCacheUnavailable); any other error is a skip by name too; out-of-memory propagates
            if is_oom(e):
                raise
            _STATE.update(enabled=False, state="skipped", reason="jax_config_refused", detail=str(e))
            return dict(_STATE)
    else:
        os.makedirs(directory, exist_ok=True)
    if rec == "det":
        _STATE["xla_caches"] = XLA_CACHES_FENCE
        _fence_in_process(rec)                                           # (may downgrade the word to `uncoupled` on a jax without the option)
    env.update(cache_env)                                                # child processes (pred's model process, colabfold's workers) inherit it
    _STATE.update(enabled=True, state="on", reason=None, source="keyed", dir=directory, root=root)
    return dict(_STATE)


def entries() -> Optional[int]:
    """Files under the cache directory now (the LEVER line's `entries` at exit); None when no directory is placed."""
    d = _STATE.get("dir")
    if not d or not os.path.isdir(str(d)):
        return None
    return sum(len(files) for _, _, files in os.walk(str(d)))


def exit_evidence() -> dict:
    """The LEVER line's fields at exit: source, recipe, root, dir, xla_caches, entries."""
    return {"source": _STATE["source"], "recipe": _STATE["recipe"], "root": _STATE["root"], "dir": _STATE["dir"],
            "xla_caches": _STATE["xla_caches"], "entries": entries()}


def reset_for_tests() -> None:
    _STATE.update(enabled=False, state="off", reason=None, detail=None, source=None, dir=None, root=None, recipe=None, xla_caches=None)
