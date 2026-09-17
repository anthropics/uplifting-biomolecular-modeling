"""Lazy autoload, installed at interpreter start by `colabdesign_opt_autoload.pth` (its one guarded line imports this module: the interpreter
holds exactly `colabdesign_opt` and `colabdesign_opt._autoload` after start-up, with or without COLABDESIGN_OPT — no other module of the
package, nothing of the core, of jax or of colabdesign; `names` is read at install / trigger time only).

With COLABDESIGN_OPT=fast in the environment, a meta-path finder waits for the first import of `colabdesign` and, right
after that package's own body has executed, installs the mode's levers in this process via `stack.activate(mode, route="env", strict=True)`
(levers.install: the mode's lever set in registry order, each rebinding its ColabDesign class or method). `import colabdesign`
builds no model (colabdesign/__init__.py imports the classes), so the hook is in time: the levers are in place before any
`mk_afdesign_model(...)`. When the mode cannot be activated — configuration: colabdesign off its pinned commit, the core pin, an unknown mode, a
late activation — the process prints `NOT ACTIVE reason=...` and exits 3: stock never runs silently under COLABDESIGN_OPT. A LEVER that cannot
engage here is not that (the kit rule): it steps aside BY NAME — its `LEVER … state=skipped reason=cannot_run|no_attention_kernel` line, and
`skipped=<levers>` on the ACTIVE line — and the mode runs the rest of its set, every engaged lever named on `levers=`.
A card other than the tested one (or below the kernel's compute-capability floor) and a stack distribution off its pin are NAMED on
the ACTIVE line (`gpu=`, `stack_drift=`), never refused: the levers engage. Until the trigger nothing else is imported;
with COLABDESIGN_OPT unset or `off` no finder is installed; an unknown name installs a finder that prints the refusal and exits 3 at the
trigger. The finder fires once and removes itself. This route cannot set the host process's exit code at the END of the run: a lever installed
and then not applied (a PARTIAL activation — e.g. a kernel that served no call) is recorded only by the levers'
own LEVER lines in that process; the gated form is `colabdesign-opt design` (exit 3 on a partial arm: an installed lever that did not engage), which reads
those same lines from the arm's log.

The package's arms are never levered by the hook (`driver_process_refusal`): `kit_launch.py` installs its mode's levers itself and
`stock_launch.py` must stay stock; `stock_design.py` is the stock arm's own script. In those processes the hook prints NOT ACTIVE and exits 3 rather than levering silently — unset COLABDESIGN_OPT to run them by hand.
"""
import os
import sys

ENV = "COLABDESIGN_OPT"                              # == names.ENV / modes.ENV, restated: this module imports nothing of the package at interpreter start
TRIGGERS = ("colabdesign",)


class Finder:
    """Duck-typed meta-path finder (no importlib.abc import at interpreter start)."""

    def __init__(self, mode, refusal=None):
        self.mode = mode
        self.refusal = refusal                         # set: the finder refuses at the trigger (exit 3) instead of activating
        self.armed = True
        self.fired = None

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname not in TRIGGERS:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        self.armed = False
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)                              # the trigger package's own body first, then the activation
            self._fire(module.__name__)
        spec.loader.exec_module = exec_module
        return spec

    def _fire(self, trigger):
        self.fired = trigger
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        why = self.refusal or driver_process_refusal()
        if why:
            sys.stderr.write(f"[colabdesign-opt] NOT ACTIVE reason={why}\n"); sys.stderr.flush()
            sys.exit(3)
        from ._core_gate import gate
        gate(__file__, tag="colabdesign-opt")          # the core pin, before any opt_core import: [colabdesign-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … -> exit 3
        try:
            from . import report, stack                # the activation path imports the shared core (opt_core); the finder chain above imports nothing of it
        except ImportError as e:                       # a pinned core lacking a sub-module this package imports: named, exit 3 — never a silent run of stock under the variable
            sys.stderr.write(f"[colabdesign-opt] NOT ACTIVE: core_missing:{getattr(e, 'name', None) or e} ({e}); install the tree's core at the kit's pin: pip install -e <tree>/common/opt_core -e <tree>/colabdesign/opt\n"); sys.stderr.flush()
            sys.exit(3)
        try:
            rep = stack.activate(self.mode, route="env", strict=True, trigger=trigger)
        except stack.ActivationError as e:
            report.log(f"NOT ACTIVE reason={e}")
            sys.exit(3)
        report.log(report.active_line(rep))


def driver_process_refusal(argv=None):
    """The reason the hook refuses in this process, or None: the process is one of the package's arms."""
    from .names import NEVER_LEVERED                       # trigger time
    argv = sys.argv if argv is None else argv
    a0 = str(argv[0]) if argv else ""
    base = os.path.basename(a0)
    if base in NEVER_LEVERED:
        return (f"{ENV} is set but this process is {base}: the package's arms install their levers themselves (kit_launch.py --mode) and are "
                f"composed by `colabdesign-opt design` — unset {ENV} to run it by hand")
    return None


def install(environ=None):
    """Install the finder for COLABDESIGN_OPT (idempotent). Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    mode = (environ.get(ENV) or "").strip().lower()
    if not mode or mode == "off":
        return None
    from .names import MODE_NAMES as MODES, split_mode_word   # only when a mode is set: `os` + literals, nothing else (names.py is import-free)
    refusal = None                                     # an unknown name: the process exits 3 at the trigger — stock never runs silently under the variable
    if split_mode_word(mode)[0] not in MODES:          # the word's base (the word itself unless it is the subtractive form <mode>-no-<lever>, which the trigger's resolve validates in full)
        refusal = f"unknown {ENV}={mode!r} (expected {'|'.join(MODES)}, or <mode>-no-<lever>[-no-<lever>...])"
    for f in sys.meta_path:
        if isinstance(f, Finder):
            return f
    f = Finder(mode, refusal)
    sys.meta_path.insert(0, f)
    return f


FINDER = install()
