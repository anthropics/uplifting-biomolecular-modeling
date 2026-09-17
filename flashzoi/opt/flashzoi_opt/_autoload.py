"""Lazy autoload, installed at interpreter start by `flashzoi_opt_autoload.pth` (the generated guard line imports `flashzoi_opt._autoload`).

With FLASHZOI_OPT=exact in the environment, a meta-path finder waits for the first import of the trigger package and, right
after that package's own body has executed, calls `flashzoi_opt.enable(mode, strict=True)`; when the mode cannot be activated
(pins, no GPU, kit missing) the core prints its NOT ACTIVE line and the process exits 3 — stock never runs silently under
FLASHZOI_OPT. Until then nothing else is imported (no torch, no upstream); with FLASHZOI_OPT unset or "off" no finder is installed
at all; an unknown FLASHZOI_OPT installs a refusing finder: the NOT ACTIVE line at start and exit 3 at the trigger import (the
CLI's exit code), so stock never runs under a misspelt mode.

The trigger is the upstream package `borzoi_pytorch` (its own body imports the model module, so the model class exists when the
finder fires). enable() arms the application on the model class's forward: the kit's own apply line runs on each model instance at
its first forward on a CUDA device (stack.py), which is the only point where a loaded model exists. The finder fires once and
removes itself.
"""
import os
import sys

ENV = "FLASHZOI_OPT"
TRIGGERS = ("borzoi_pytorch",)
MODES = ("off", "exact")


class Finder:
    """Duck-typed meta-path finder (no importlib.abc import at interpreter start)."""

    def __init__(self, mode, refuse=None):
        self.mode = mode
        self.refuse = refuse                           # an unknown FLASHZOI_OPT: the finder still waits for the trigger and then exits 3 (never stock under a bad name)
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
            _orig(module)                              # the trigger package's own body first, then the arming
            self._fire(module.__name__)
        spec.loader.exec_module = exec_module
        return spec

    def _fire(self, trigger):
        self.fired = trigger
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        if self.refuse:
            sys.stderr.write(f"[flashzoi-opt] NOT ACTIVE: {self.refuse}\n")   # the unknown mode named again at the trigger; exit 3 like the CLI
            sys.stdout.flush(); sys.stderr.flush(); os._exit(3)     # the interpreter ends here, nothing of the importer's runs after a refusal (a SystemExit could be swallowed by the importer)
        import flashzoi_opt
        try:
            flashzoi_opt.enable(self.mode, strict=True, trigger=trigger)
        except flashzoi_opt.ActivationError:
            # the core has printed `[flashzoi-opt] NOT ACTIVE: <reason>`; the process stops here (exit 3) rather than running stock silently
            sys.stdout.flush(); sys.stderr.flush(); os._exit(3)     # the interpreter ends here, nothing of the importer's runs after a refusal (a SystemExit could be swallowed by the importer)


def install(environ=None):
    """Install the finder for FLASHZOI_OPT (idempotent). Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    mode = (environ.get(ENV) or "").strip().lower()
    if not mode or mode == "off":
        return None
    refuse = None
    if mode not in MODES:
        refuse = f"unknown {ENV}={mode!r} (expected {'|'.join(MODES)})"
        sys.stderr.write(f"[flashzoi-opt] NOT ACTIVE: {refuse}\n")           # printed at interpreter start; the trigger import then exits 3
    for f in sys.meta_path:
        if isinstance(f, Finder):
            return f
    f = Finder(mode, refuse=refuse)
    sys.meta_path.insert(0, f)
    return f


FINDER = install()
