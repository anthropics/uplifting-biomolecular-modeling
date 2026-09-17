"""The CALIBY_OPT environment route, installed at interpreter start by `caliby_opt_autoload.pth` — the build backend's generated guard
line around `import caliby_opt._autoload` (`python opt/_build_backend.py caliby_opt CALIBY_OPT caliby-opt`): under a set CALIBY_OPT an
unimportable package is a `[caliby-opt] NOT ACTIVE` line and exit 3 rather than site.py's swallowed ImportError, and a kit refusal
raised while the .pth line runs keeps its exit code.

With CALIBY_OPT=fast (single) or CALIBY_OPT=exact and CALIBY_VARIANT=ensemble32 in the environment — the variant's one kit mode; the
other kit mode on a variant is refused by name at the trigger — a meta-path finder waits for the first import of the upstream package
`caliby` and, BEFORE that package's body executes (its body imports `caliby.api`, one of the files the kits replace), calls
`caliby_opt.enable(mode, variant)` with the variant from CALIBY_VARIANT. `enable()` proves the installed tree is the pinned
upstream, proves the kits' files against their manifests and installs the import hook that loads them under the upstream module
names (overlay.py), exports the row's switches, prints the activation line and registers the exit tally; when it cannot (a patched
tree, pins, no GPU, no compiler, weights) it prints `[caliby-opt] NOT ACTIVE: <reason>` and the process exits 3 — stock never runs
silently under CALIBY_OPT. At the trigger the core pin gate runs first (`stack.core_gate`): an absent, older, newer or edited shared
core is its one `NOT ACTIVE: reason=core_...` line and exit 3 before anything of the core is imported. Until the trigger nothing else
is imported (no torch, no upstream, no other module of this package, nothing of the shared core); with CALIBY_OPT unset or "off" no
finder is installed at all and the process stays core-free. The finder fires once, BEFORE the trigger's body (that body imports an
overlaid module), and removes itself — the one difference from the shared core's finder (`opt_core.autoload`, which activates after
the trigger's body), and the reason this kit keeps its own. A run that was not the mode's run ends the process NOT ACTIVE by name on
this route by the same two rules the CLI verb (`caliby-opt design`) applies (`report._exit_gate`, at interpreter exit, after the exit
tally line `[caliby-opt] EXIT ...`): a lever of the row that fell back to its stock line at call time (a PARTIAL activation,
`activate.completion`) — `[caliby-opt] NOT ACTIVE: <mode> ran short of its lever set — ...; exit 3; CALIBY_OPT=off runs stock`; lever
modules that were not the mode's files (`activate.modules_wrong`, `tree=mixed`) — `[caliby-opt] NOT ACTIVE: the lever modules loaded in
this process were not the <tree> arm's files (tree=mixed: ...); ...; exit 3; CALIBY_OPT=off runs stock`; then exit code 3 whatever the
program was about to exit with (exit hooks the program registered before the trigger do not run then). A mode is all of its levers on
its own files, never a subset under its name; this route has no opt-out — `CALIBY_OPT=off` (or unset) runs stock.
"""
import os
import sys

ENV, ENV_VARIANT = "CALIBY_OPT", "CALIBY_VARIANT"
EXIT_NOT_ACTIVE = 3                                                    # report.EXIT_NOT_ACTIVE (the shared core's table), restated import-free: nothing else is imported at start
TRIGGERS = ("caliby",)
MODES = ("exact", "fast", "off")                                        # modes.MODES, restated import-free (locked equal by the tests)


class Finder:
    """Duck-typed meta-path finder (no importlib.abc import at interpreter start)."""

    def __init__(self, mode):
        self.mode = mode
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
            self._fire(module.__name__)                    # activation first (the tree proof, the row), then the package body
            _orig(module)
        spec.loader.exec_module = exec_module
        return spec

    def _fire(self, trigger):
        self.fired = trigger
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        from .stack import TAG, core_gate                              # trigger time: the package's facts module (standard library only; nothing of the core)
        if self.mode not in MODES:                                     # an unknown selection is a refusal at the model's import, with the kit's code
            sys.stderr.write(f"[{TAG}] NOT ACTIVE: unknown {ENV}={self.mode!r} (expected {'|'.join(MODES)}); exit {EXIT_NOT_ACTIVE}\n")
            _leave(EXIT_NOT_ACTIVE)
        try:                                                           # a kit mode is named: the core pin gate first — an absent / older / newer / edited core is ITS line, exit 3, before anything of the core is imported
            core_gate()
        except SystemExit as x:
            _leave(x.code)
        import caliby_opt
        variant = (os.environ.get(ENV_VARIANT) or "").strip().lower() or None
        try:
            caliby_opt.enable(self.mode, variant, strict=True, trigger=trigger)
        except caliby_opt.ActivationError:
            _leave(EXIT_NOT_ACTIVE)                                    # the package has printed `[caliby-opt] NOT ACTIVE: <reason>`; the process stops here rather than running stock silently


def _leave(code) -> None:
    """The one way a refusal at the trigger ends the process: both streams flushed, then ``os._exit`` with the kit's code. Not ``SystemExit``:
    the trigger runs inside the host's ``import caliby``, where a host's ``except BaseException`` (or site.py, when the import happens while a
    .pth line runs) would swallow it, the finder is already gone, and the host's next ``import caliby`` would run stock silently under CALIBY_OPT."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code if isinstance(code, int) else EXIT_NOT_ACTIVE)


def install(environ=None):
    """Install the finder for CALIBY_OPT (idempotent). Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    mode = (environ.get(ENV) or "").strip().lower()
    if not mode or mode == "off":
        return None
    # an unknown mode is refused at the trigger (Finder._fire: the NOT ACTIVE line with exit 3), never noted-and-continued here: a process that
    # never imports the model is untouched, and a line that says NOT ACTIVE always comes with the kit's exit code
    for f in sys.meta_path:
        if isinstance(f, Finder):
            return f
    f = Finder(mode)
    sys.meta_path.insert(0, f)
    return f


FINDER = install()
