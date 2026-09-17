"""The kit's autoload hook, run at interpreter start by `opendde_opt_autoload.pth` (`import opendde_opt._autoload`).

Import-free until OPENDDE_OPT names a mode: the .pth runs in every interpreter of
the environment, the stock arm's processes included, and those load nothing of the core. A variable under the package prefix the package does
not read (OPENDDE_OPT_MODE, a mistyped name) is refused here, in every process, with the NOT ACTIVE line and exit 3 — never ignored
silently; a selection outside the mode table (and no line) is refused the same way. Otherwise the core's finder (`opt_core.autoload`)
is installed from this kit's AutoloadSpec: it waits for the first import of a trigger package and, right after that package's own body
has executed, calls `opendde_opt.enable(mode, strict=True, trigger=<name>)`; when the mode cannot be activated (a line name modes.py
does not know, the tree, pins, no GPU) the package prints its
NOT ACTIVE line and the process exits 3: stock never runs silently under OPENDDE_OPT. Until then nothing else is imported (no torch, no
opendde); with OPENDDE_OPT unset or "off", no finder is installed. A bare `importlib.util.find_spec(<trigger>)` probe leaves
the finder armed — it disarms only when the trigger's body has run.

Triggers are the top of the OpenDDE model family — `runner` (the stock CLI `runner.batch_inference` and the kit worker enter
through it) and `opendde.model` (library use). Firing at `runner` precedes `runner.batch_inference`, so the served-levers hook
armed by the shim wraps `get_default_runner` at that module's import, exactly as on the kit's own PYTHONPATH route. `runner` is a
generic name: `accept` takes it only when an `opendde` package sits beside it. The finder fires once and removes itself.
"""
import os
import sys

from . import MODES
from . import ENV_MODE as ENV

TAG = "opendde-opt"
TRIGGERS = ("runner", "opendde.model")
EXIT_NOT_ACTIVE = 3                                # cli.EXIT_NOT_ACTIVE, restated: nothing else is imported at interpreter start
DECLARED = (ENV,)                                 # every variable the package reads under its prefix (a test locks it)


def _refuse(reason):
    """The NOT ACTIVE line and the exit code itself: this runs from the .pth at interpreter start, where SystemExit is a site error (rc 1).
    The sentence always carries ``reason=`` (fail-loud: a refusal names itself in one greppable field)."""
    reason = str(reason)
    if not (reason.startswith("reason=") or " reason=" in reason):
        reason = "reason=" + reason
    sys.stderr.write(f"[{TAG}] NOT ACTIVE: {reason}\n")
    sys.stderr.flush()
    os._exit(EXIT_NOT_ACTIVE)


def _opendde_family(fullname, spec):
    """`runner` is a generic name: accept it only when an `opendde` package sits beside it."""
    if fullname != "runner":
        return True
    origin = getattr(spec, "origin", None) or ""
    return os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(origin)), "opendde", "__init__.py"))


def selection(environ=None):
    """The mode from OPENDDE_OPT (stripped, case-folded)."""
    environ = os.environ if environ is None else environ
    return (environ.get(ENV) or "").strip().lower()


def spec():
    """This kit's AutoloadSpec for the core's finder (imports the core: called under a set variable, or from the package)."""
    from opt_core.autoload import AutoloadSpec
    return AutoloadSpec(env=ENV, package=__package__, tag=TAG, triggers=TRIGGERS, modes=tuple(MODES), accept=_opendde_family,
                        exit_not_active=EXIT_NOT_ACTIVE, on_unknown="exit_now", fold_mode=True)


def install(environ=None):
    """The kit's gate (idempotent): an undeclared name or an unknown selection is refused at once with the package's line and exit 3;
    nothing is installed unless a selection is named; else the core's finder from ``spec()``. Returns the finder, or None when nothing
    is to be done."""
    environ = os.environ if environ is None else environ
    undeclared = sorted(k for k in environ if k.startswith(ENV) and k not in DECLARED)
    if undeclared:                                     # a mistyped variable under the package prefix is never silently ignored
        _refuse(f"undeclared variable(s) {', '.join(undeclared)} (the package reads {ENV})")
    mode = selection(environ)
    if not mode or mode == "off":
        return None
    if mode not in MODES:
        _refuse(f"unknown {ENV}={mode!r} (expected {'|'.join(MODES)})")   # never print-and-return: stock would run silently under the variable
    from ._core_gate import CoreGateRefused, gate
    try:                                               # THE pin gate: an absent / mismatched opt_core under a kit selection is its NOT ACTIVE line and exit 3
        gate(__file__, TAG)                            # (os._exit: SystemExit from a .pth line is a site error, rc 1)
    except CoreGateRefused:
        sys.stderr.flush()
        os._exit(EXIT_NOT_ACTIVE)
    from ._producers import refusal as _producers_refusal
    r = _producers_refusal()                            # then the finer words: a producer module this package imports is absent (never a stock run under the variable)
    if r:
        _refuse(r.split(" NOT ACTIVE ", 1)[1])
    try:
        from opt_core.autoload import install as core_install
    except ImportError as e:                           # the shared core absent under a kit selection: named, exit 3 (never a stock run under the variable)
        _refuse(f"reason=core_missing:{getattr(e, 'name', None) or e} (the shared core is not importable; pip install -e common/opt_core -e opendde/opt)")
    return core_install(spec(), {**environ, ENV: mode})


def disarm() -> bool:
    """Remove the finder (called by the core when enable() runs first, so the finder never fires a second activation). True when one was
    armed. A finder exists only when the core's finder module was imported, so nothing of the core is imported otherwise."""
    core = sys.modules.get("opt_core.autoload")
    return core is not None and core.disarm(spec())


FINDER = install()
