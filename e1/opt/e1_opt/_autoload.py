"""Lazy autoload, installed at interpreter start by `e1_opt_autoload.pth` (a guarded `import e1_opt._autoload`: opt/_build_backend.py pth_text).

With E1_OPT=exact in the environment, a meta-path finder waits for the first import of a trigger module of the upstream model family
and, right after that module's own body has executed, calls `e1_opt.enable(mode, variant, strict=True)` with the variant from
E1_VARIANT; when the mode cannot be activated (no variant named, pins, stack, card, kit missing) the core prints its NOT ACTIVE line
and the process exits 3 — stock never runs silently under E1_OPT. Until then nothing else is imported (no torch, no upstream); with
E1_OPT unset or "off" no finder is installed at all; an unknown name is refused at the trigger (exit 3).

Triggers: `E1.predictor` (the class the kit applies at: E1Predictor — E1Scorer builds one, so this fires on the documented command line
too) and `E1.scorer` (whichever a program imports first; E1's package __init__ imports nothing, so `import E1` alone is not a trigger).
enable() arms the constructor wrap; the kit's own apply() runs on the model at the first E1Predictor construction (stack.py). The
finder fires once and removes itself.
"""
import os
import sys

from ._names import ENV, ENV_VARIANT, ENV_DET   # the one names module (stdlib-only, nothing else imported at interpreter start)
TRIGGERS = ("E1.predictor", "E1.scorer")


class Finder:
    """Duck-typed meta-path finder (no importlib.abc import at interpreter start). `on_fire(trigger)`: what runs right after the trigger
    module's own body (default: the activation below); the KERNELS proof (accel.install) arms its constructor hook through the same
    finder so that nothing upstream is imported ahead of the tool's own import order."""

    def __init__(self, mode, on_fire=None, triggers=TRIGGERS):
        self.mode = mode
        self.on_fire = on_fire
        self.triggers = tuple(triggers)
        self.armed = True
        self.fired = None

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname not in self.triggers:
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
            _orig(module)                              # the trigger module's own body first, then the arming
            self._fire(module.__name__)
        spec.loader.exec_module = exec_module
        return spec

    def _fire(self, trigger):
        self.fired = trigger
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        if self.on_fire is not None:
            self.on_fire(trigger)
            return
        import e1_opt
        variant = (os.environ.get(ENV_VARIANT) or "").strip().lower() or None
        try:
            e1_opt.enable(self.mode, variant, strict=True, trigger=trigger)
        except e1_opt.ActivationError:
            # the core has printed `[e1-opt] NOT ACTIVE: <reason>`; the process stops here (exit 3) rather than running stock silently
            sys.exit(3)


def install(environ=None):
    """Install the finder for E1_OPT (idempotent). Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    mode = (environ.get(ENV) or "").strip().lower()
    if not mode or mode == "off":
        return None
    # any other value (exact, or a misspelling) arms the finder: the core refuses unknown names at the trigger (exit 3)
    for f in sys.meta_path:
        if isinstance(f, Finder) and f.on_fire is None:      # the activation finder (a KERNELS-proof finder carries on_fire)
            return f
    f = Finder(mode)
    sys.meta_path.insert(0, f)
    _export_recipe_env(environ)
    return f


def _die(code: int) -> None:
    """Exit the interpreter with `code` from inside the .pth import: no user code has run, nothing to unwind (sys.exit here would be reported
    as 'Failed to import the site module', rc 1)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _export_recipe_env(environ) -> None:
    """E1_OPT_DET=1 on the env route: the recipe's environment entries (the kit's pins DET_RECIPE["env"]: CUBLAS_WORKSPACE_CONFIG, the numpy /
    OpenBLAS CPU-feature entries) exported here, at interpreter start, before numpy / torch are imported — the CLI's wrapper exports the same
    entries into its item subprocess (the recipe's values, as there). The pins module is torch-free; nothing else is imported."""
    if not (environ.get(ENV_DET) or "").strip():
        return
    from . import stack
    try:
        level = stack.det_level(environ)                 # the one reader of E1_OPT_DET (0 | 1; anything else is refused by name, here too)
    except stack.ActivationError as e:
        print(f"[e1-opt] NOT ACTIVE: {e}", flush=True)
        _die(3)                                          # at interpreter start (a .pth import) a SystemExit is a fatal init error, rc 1: exit directly
    if level != 1:
        return
    try:
        from . import det
        det.apply_env(stack.load_pins(), 1, environ)
    except Exception:  # noqa: BLE001 — the trigger's own gates report a missing / drifted kit tree by name; nothing is refused at import
        return


FINDER = install()
