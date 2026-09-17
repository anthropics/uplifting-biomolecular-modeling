"""Executed by ``af3_jax_opt_autoload.pth`` at interpreter start (a .pth-installed lazy hook, one per package). On this model the hook applies nothing:
the levers are a script plus a flag row and a launcher on the command line (modes.py), never an in-process patch, so no import of the model family
could activate anything. It exists only to refuse: if ``AF3_JAX_OPT`` names a mode other than ``off`` in an interpreter that then imports
``alphafold3`` directly (bypassing the wrapper), or an unknown ``AF3_JAX_*`` name is set, it prints the NOT ACTIVE line and exits 3 rather than
silently ignoring the variable — the wrapper itself strips these variables from its model process, so the normal wrapper chain never trips it."""
from __future__ import annotations

import os
import sys

ENV_MODE = "AF3_JAX_OPT"
ENV_PREFIX = "AF3_JAX_"                                                    # stack.ENV_PREFIX / stack.DECLARED_ENV, spelled here so nothing of the kit is imported at start (tests hold the two equal)
DECLARED_ENV = ("AF3_JAX_OPT", "AF3_JAX_VARIANT", "AF3_JAX_OPT_HOME", "AF3_JAX_REPO", "AF3_JAX_PY", "AF3_JAX_PARAMS_ROOT", "AF3_JAX_CACHE_ROOT", "AF3_JAX_N_GPU",
                "AF3_JAX_DATTN", "AF3_JAX_TTR", "AF3_JAX_TRIATT_XLA", "AF3_JAX_SAMPLER_BF16", "AF3_JAX_ATOM_ATTN", "AF3_JAX_TRIMUL_CD", "AF3_JAX_LNP", "AF3_JAX_HOIST_LOGITS", "AF3_JAX_COND_SHARE", "AF3_JAX_ATOM_COND_HOIST")           # the last two: the tree levers' model-process switches (stack.LEVER_SWITCH_ENV) — present in a model process the wrapper launched, so accepted here
TARGET = "alphafold3"
TAG = "af3-jax-opt"
PREFIX = f"[{TAG}]"                                                        # report.PREFIX, spelled here (stdlib only: this module must load without the core; tests hold them equal)
EXIT_NOT_ACTIVE = 3                                                       # opt_core.report.EXIT_NOT_ACTIVE, likewise
CORE_PACKAGE = "opt_core"
FINDER = None                                                             # the finder this module put first on sys.meta_path, or None (unset / off / declared names only)
_STATE = {"armed": False, "mode": None, "undeclared": [], "refusal": None, "core_missing": None}


def core_missing(exc: BaseException) -> str:
    """The missing module's name when ``exc`` is the shared core failing to import (``core_missing:<module>``), else ''."""
    name = getattr(exc, "name", None) or ""
    if isinstance(exc, ImportError) and (name.split(".")[0] == CORE_PACKAGE or (not name and CORE_PACKAGE in str(exc))):
        return name or CORE_PACKAGE
    return ""


def not_active_core_missing(module: str, mode=None) -> int:
    """Print the kit's NOT ACTIVE line for a missing core and return exit code 3 — the one form of this event for the start-up hook, the
    command line (__main__) and the package import: never a traceback, never a stock run under a kit variable."""
    sys.stderr.write(f"{PREFIX} NOT ACTIVE mode={mode or '?'} reason=core_missing:{module} (the shared core is not importable beside this kit: "
                     "pip install -e common/opt_core -e af3_jax/opt); exit 3\n")
    sys.stderr.flush()
    return EXIT_NOT_ACTIVE


def refusal_line(mode, undeclared, modes) -> str:
    """Why this process cannot activate: undeclared ``AF3_JAX_*`` names, a value outside the mode table, or a known mode (the wrapper's)."""
    if undeclared:                                                        # a mistyped variable name under the package prefix is never ignored
        why = f"undeclared variable(s) {', '.join(undeclared)} (declared: {', '.join(DECLARED_ENV)})"
    elif mode not in modes:
        why = f"{ENV_MODE}={mode!r} is not a mode ({'|'.join(modes)})"
    else:                                                                 # a known mode cannot be applied in this process: the levers are the wrapper's command line
        why = f"{ENV_MODE}={mode} is honoured by the wrapper command only (af3-jax-opt pred / run.sh pred); this process would run stock"
    return f"{PREFIX} NOT ACTIVE: {why}; exit 3"


class _RefusedFinder:
    """A refusal line held for the model family's first look-up (stdlib only): the pin gate's NOT ACTIVE line when the core is absent or not
    the pinned one — this interpreter may be doing anything; only the model family's import is refused, by name, exit 3."""

    def __init__(self, line):
        self.line = line

    def find_spec(self, name, path=None, target=None):  # noqa: D401 - importlib protocol (duck-typed)
        if name == TARGET or name.startswith(TARGET + "."):
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            sys.stderr.write(self.line.rstrip("\n") + "\n"); sys.stderr.flush()
            raise SystemExit(EXIT_NOT_ACTIVE)
        return None


class _CoreMissingFinder:
    """The refusal this module prints itself: the core absent under a set variable, at the model family's first look-up (stdlib only)."""

    def __init__(self, module, mode):
        self.module, self.mode = module, mode

    def find_spec(self, name, path=None, target=None):  # noqa: D401 - importlib protocol (duck-typed)
        if name == TARGET or name.startswith(TARGET + "."):
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            raise SystemExit(not_active_core_missing(self.module, self.mode))
        return None


_mode = os.environ.get(ENV_MODE, "").strip()
_undeclared = sorted(k for k in os.environ if k.startswith(ENV_PREFIX) and k not in DECLARED_ENV)
if (_mode and _mode.lower() != "off") or _undeclared:
    _STATE.update(armed=True, mode=_mode or None, undeclared=_undeclared)
    import io as _io
    from ._core_gate import CoreGateRefused, gate as _gate               # THE pin gate (kit_template copy, stdlib only) BEFORE any opt_core import
    try:
        _STATE["core"] = _gate(__file__, tag=TAG, stream=_io.StringIO())
    except CoreGateRefused as _r:                                         # the core absent / not the pinned one: its line is held for the trigger import (never printed into an unrelated process)
        _STATE["core_refused"] = _r.line
        FINDER = _RefusedFinder(_r.line)
    else:
        try:
            from opt_core.autoload import AutoloadSpec, Finder            # the shared core's finder; the kit supplies its spec and its reason
            from .modes import MODES
        except ImportError as _e:                                         # the pinned core lacks a module this package imports: named at the trigger import, exit 3
            _STATE["core_missing"] = core_missing(_e) or getattr(_e, "name", None) or "?"
            FINDER = _CoreMissingFinder(_STATE["core_missing"], _mode or None)
        else:
            _STATE["refusal"] = refusal_line(_mode, _undeclared, MODES)
            FINDER = Finder(AutoloadSpec(env=ENV_MODE, package="af3_jax_opt", tag=TAG, triggers=(TARGET,), modes=MODES, exit_not_active=EXIT_NOT_ACTIVE),
                            {"mode": _mode or None, "variant": None}, refuse=_STATE["refusal"])
    sys.meta_path.insert(0, FINDER)
