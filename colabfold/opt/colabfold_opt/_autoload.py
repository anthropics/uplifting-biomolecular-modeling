"""Lazy autoload, installed at interpreter start by `colabfold_opt_autoload.pth` (`import colabfold_opt._autoload`).

With COLABFOLD_OPT set to anything but `off` (or empty), a meta-path finder waits for the first import of the trigger module `colabfold.batch` and,
right after that module's own body has executed, installs the package's hook on `colabfold.batch.run` (stack.hook_run): the activation
then happens at the `run` call — with the run's queries (their token count) and its data_dir (the parameters gate) — and the stock `run`
follows. When the mode cannot be activated (kit, pin, jax range, GPU, parameters, late activation) the core
prints its NOT ACTIVE line and the process exits 3 — stock never runs silently under COLABFOLD_OPT: an unknown value (a typo)
is refused the same way at the trigger import (modes.resolve, in the colabfold process only — other interpreters carrying the .pth are
untouched: nothing fires without the trigger). Until the trigger
nothing else is imported (no jax, no colabfold, no other module of this package); with COLABFOLD_OPT unset or "off" no finder is
installed at all. The finder fires once and removes itself.

`ENV` and `TRIGGERS` are spelled again here (modes.ENV, stack.TRIGGER_MODULE) because this module must
import nothing at interpreter start; the pairs are locked by the package tests.
"""
import os
import sys

ENV = "COLABFOLD_OPT"
ENV_NAMES = (ENV, ENV + "_DATA_DIR", ENV + "_HOME", ENV + "_KIT", ENV + "_LAUNCH_ID", ENV + "_WORK_DIR", ENV + "_JIT_ROOT",
             ENV + "_N_GPU")   # every variable this kit reads under its prefix (…_N_GPU: modes.ENV_N_GPU, big's axis in the model process)
LEVERS_OFF_ENV = "MODEL_OPT_LEVERS_OFF"                   # ablation.ENV, spelled again (locked by the tests): the ablation switch — a mode minus the named levers; under off / unset there is nothing to ablate and the trigger import refuses it by name
PREFIX = "[colabfold-opt]"                                # report.PREFIX, spelled again: the core-missing refusal below must print without importing report (which imports the core)
TAG = PREFIX[1:-1]                                        # the tag the core pin gate prints its line under
EXIT_NOT_ACTIVE = 3                                       # report.EXIT_NOT_ACTIVE, likewise
TRIGGERS = ("colabfold.batch",)


def gate_core() -> dict:
    """THE core pin gate of every entry (the package's byte-identical copy of the shared core's kit_template/_core_gate.py): the opt_core this
    interpreter would import — located without importing it — is the one opt/pyproject.toml [tool.opt_core] pins, else ONE line
    (`NOT ACTIVE: reason=core_missing:opt_core …` / `reason=core_mismatch: opt_core pinned … installed …` / `reason=core_pin_unreadable …`)
    and SystemExit(3). Returns the gate's facts on a match. Statement one of __main__.main, Finder._fire, the in-process enable/check/status
    and run.sh / configs/*.env; core_missing() below is statement two (a module of the pinned core this package imports is missing)."""
    from ._core_gate import gate
    return gate(os.path.abspath(__file__), tag=TAG)


def core_missing(e: BaseException) -> int:
    """Print the NOT ACTIVE line for a package import that failed because a module of the shared core is missing and return the exit
    code — the one text for the trigger import (Finder._fire), the command line (__main__.main) and the in-process route."""
    sys.stderr.write(f"{PREFIX} NOT ACTIVE: core_missing:{getattr(e, 'name', None) or e} "
                     f"(pip install -e <release>/common/opt_core -e <release>/colabfold/opt)\n")
    sys.stderr.flush()
    return EXIT_NOT_ACTIVE


def undeclared(environ=None):
    """The variables set under the kit's prefix that no module of the kit reads (a mistyped name, e.g. COLABFOLD_OPT_MODE): refused by
    name at the trigger import (Finder) and at the command line (cli.main) — never ignored, never stripped silently."""
    environ = os.environ if environ is None else environ
    return sorted(k for k in environ if k.startswith(ENV) and k not in ENV_NAMES)


class Finder:
    def __init__(self, mode, undeclared_names=(), levers_off=""):
        self.mode = mode
        self.undeclared_names = tuple(undeclared_names)
        self.levers_off = levers_off                      # MODEL_OPT_LEVERS_OFF's text when COLABFOLD_OPT is off / unset (nothing to ablate: refused at the trigger); "" otherwise — a kit mode's names are the activation's to validate
        self.fired = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TRIGGERS or self.fired:
            return None
        try:
            sys.meta_path.remove(self)                    # let the real finders resolve the spec
            import importlib.util
            spec = importlib.util.find_spec(fullname)
        finally:
            if self not in sys.meta_path and not self.fired:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            self._fire(module)
        loader.exec_module = exec_module
        return spec

    def _fire(self, module):
        self.fired = module.__name__
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        gate_core()                                       # statement one: the pinned core is what this interpreter imports, or NOT ACTIVE and exit 3 — before any opt_core import
        try:
            from colabfold_opt import modes as _modes, report as _report, stack as _stack
        except ImportError as e:                          # a module of the core this package imports is missing: refused by name, never a traceback or a stock run
            sys.exit(core_missing(e))
        if self.undeclared_names:                         # a variable under the prefix this kit does not read (a mistyped name): NOT ACTIVE and exit 3 at the trigger import
            _report.emit(_report.NOT_ACTIVE_FMT.format(prefix=_report.PREFIX, reason=f"undeclared variable(s) {', '.join(self.undeclared_names)} (the names this kit reads: {', '.join(ENV_NAMES)})"))
            sys.exit(_report.EXIT_NOT_ACTIVE)
        if self.levers_off:                               # the ablation switch with no kit mode selected: stock applies no lever — refused by name, never a silent stock run under an ablation's name
            _report.emit(_report.NOT_ACTIVE_FMT.format(prefix=_report.PREFIX, reason=f"{LEVERS_OFF_ENV}={self.levers_off}: {ENV} is {self.mode or 'unset'} — mode off applies no lever, there is nothing to ablate (unset {LEVERS_OFF_ENV} for the stock route)"))
            sys.exit(_report.EXIT_NOT_ACTIVE)
        try:
            _modes.resolve(self.mode)                     # a selection outside the mode table: NOT ACTIVE and exit 3 here, at the trigger import
        except _modes.UnsupportedMode as e:
            _report.emit(_report.NOT_ACTIVE_FMT.format(prefix=_report.PREFIX, reason=e))
            sys.exit(_report.EXIT_NOT_ACTIVE)
        _stack.hook_run(self.mode, strict=True, trigger=module.__name__, module=module)


def install(environ=None):
    """Install the finder for COLABFOLD_OPT (idempotent). Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    mode = (environ.get(ENV) or "").strip().lower()
    bad = undeclared(environ)
    off = not mode or mode == "off"
    levers_off = (environ.get(LEVERS_OFF_ENV) or "").replace(" ", "").strip(",") if off else ""
    if off and not bad and not levers_off:
        return None
    for f in sys.meta_path:
        if isinstance(f, Finder):
            return f
    f = Finder(mode, bad, levers_off)                     # an undeclared name (or the ablation switch under off) installs the finder too: the trigger import is refused, whatever COLABFOLD_OPT says
    sys.meta_path.insert(0, f)
    return f


FINDER = install()
