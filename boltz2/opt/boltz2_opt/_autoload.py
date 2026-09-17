"""The kit's autoload hook, run at interpreter start by `boltz2_opt_autoload.pth` (`import boltz2_opt._autoload`).

Import-free until BOLTZ2_OPT names a selection: the .pth runs in every interpreter on the box, the stock arm's processes included, and
those load nothing of the core. A BOLTZ2_OPT* name the package does not read (`BOLTZ2_OPT_MODE`, `BOLTZ2_OPTS`, …: a mistyped selection)
is refused here, in every process, with the NOT ACTIVE line and exit 3 — never ignored silently (and stripped from the stock subprocess).
With BOLTZ2_OPT=<a mode other than off> the core's finder (`opt_core.autoload`) is installed from this kit's AutoloadSpec: it waits for the
first import of a trigger package and, right after that package's own body has executed, calls `boltz2_opt.enable(mode, strict=True,
trigger=<name>)`. For Boltz-2 that call refuses in a stock-CLI process by name (no kit hook composes the mode's levers in one process:
modes.CLI_ROUTE) — the package prints its `[boltz2-opt] NOT ACTIVE: <reason>` line and the process exits 3, so `BOLTZ2_OPT=exact boltz
predict ...` never runs stock silently; the route is `boltz2-opt pred`. Until the trigger nothing else is imported (no torch, no
boltz); with BOLTZ2_OPT unset or "off" no finder is installed at all; a value outside the mode table prints the NOT ACTIVE line and exits
3 at once (the selection is stripped and case-folded onto a declared mode). A bare `importlib.util.find_spec(<trigger>)` probe leaves the
finder armed — it disarms only when the trigger's body has run. The package's own worker and stock subprocesses are launched without
BOLTZ2_OPT (stack.child_env, stock_pred), so the finder never fires in them.

Triggers are the upstream model family's packages — `boltz.model` (the model) and `boltz.main` (the CLI) — whichever a program imports
first. The finder fires once and removes itself.
"""
import os
import sys

ENV = "BOLTZ2_OPT"
TAG = "boltz2-opt"
EXIT_NOT_ACTIVE = 3                              # report.EXIT_NOT_ACTIVE, restated: nothing is imported at interpreter start
TRIGGERS = ("boltz.model", "boltz.main")
MODES = ("exact", "fast", "big", "off")


BIG_WORD = "BOLTZ2_BIG_"                                          # the memory line's per-lever switch / setting / opt-out words: none is read; a caller's is refused by name


def undeclared_names(environ=None):
    """Environment names under the package prefix that the package does not read (`BOLTZ2_OPT_MODE`, `BOLTZ2_OPTS`, …) and any
    `BOLTZ2_BIG_*` word (the memory mode's levers and settings are the mode's row): a mistyped selection would otherwise be ignored
    silently (and stripped from the stock subprocess)."""
    environ = os.environ if environ is None else environ
    return sorted(k for k in environ if (k.startswith(ENV) and k != ENV) or k.startswith(BIG_WORD))


def spec():
    """This kit's AutoloadSpec for the core's finder (imports the core: called under a set variable, or from the package). `on_unknown="exit"`:
    an unknown selection raises SystemExit(EXIT_NOT_ACTIVE) after the line — in-process semantics; the .pth line turns it into the exit code
    itself below."""
    from opt_core.autoload import AutoloadSpec
    return AutoloadSpec(env=ENV, package=__package__, tag=TAG, triggers=TRIGGERS, modes=MODES, exit_not_active=EXIT_NOT_ACTIVE,
                        on_unknown="exit", fold_mode=True)


REQUIRED_CORE_MODULES = ("opt_core", "opt_core.autoload", "opt_core.gates", "opt_core.report", "opt_core.stock_proof", "opt_core.mem.ngpu")   # what cli / stack / report / tp import on every verb


def require_core() -> None:
    """Statement two of every entry route, after the kit's core pin gate (``_core_gate.gate``: pinned == installed) — ``python -m boltz2_opt`` /
    ``boltz2-opt``, ``boltz2_opt.enable``, ``configs/*.env``: the pinned core must carry the modules this package imports — else the NOT ACTIVE line naming
    ``core_missing:<module>`` and SystemExit(EXIT_NOT_ACTIVE), before anything of the core is imported by the route (never a traceback,
    never a stock run). The .pth route (``install``) applies the same rule to ``opt_core.autoload``."""
    import importlib
    for mod in REQUIRED_CORE_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing = getattr(e, "name", None) or mod
            sys.stderr.write(f"[{TAG}] NOT ACTIVE: reason=core_missing:{missing} — this package imports the shared core opt_core at the release pinned in "
                             f"opt/pyproject.toml [tool.opt_core] (with the n_gpu axis, opt_core.mem.ngpu); it is absent or older ({type(e).__name__}: {e}; "
                             f"pip install -e common/opt_core -e boltz2/opt); exit {EXIT_NOT_ACTIVE}\n")
            raise SystemExit(EXIT_NOT_ACTIVE)


def install(environ=None):
    """The kit's gate (idempotent): an undeclared BOLTZ2_OPT* name raises SystemExit(EXIT_NOT_ACTIVE) after the NOT ACTIVE line; nothing is
    installed unless the variable names a selection; else the core's finder from ``spec()`` (an unknown selection: the line, then
    SystemExit(EXIT_NOT_ACTIVE)). Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    bad = undeclared_names(environ)
    if bad:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: undeclared {', '.join(bad)} in the environment (the one variable is {ENV}={'|'.join(MODES)})\n")
        raise SystemExit(EXIT_NOT_ACTIVE)
    sel = (environ.get(ENV) or "").strip().lower()
    if sel in ("", "off"):
        return None
    from ._core_gate import gate                          # the kit's core pin gate: the importable opt_core is the pinned one, or the NOT ACTIVE line + exit 3 here
    gate(__file__, TAG)
    try:
        from opt_core.autoload import install as core_install
    except ImportError as e:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: reason=core_missing:{getattr(e, 'name', None) or 'opt_core'} — the shared core is not importable ({e}; pip install -e common/opt_core -e boltz2/opt)\n")
        raise SystemExit(EXIT_NOT_ACTIVE)
    return core_install(spec(), environ)


try:
    FINDER = install()
except SystemExit as e:                                # at .pth time site.py cannot carry SystemExit cleanly: the line, then the exit code itself
    sys.stderr.flush(); os._exit(int(e.code or EXIT_NOT_ACTIVE))
