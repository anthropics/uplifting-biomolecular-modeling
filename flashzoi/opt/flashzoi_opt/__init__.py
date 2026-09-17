"""flashzoi_opt — explicit interface to the Flashzoi (borzoi-pytorch) inference optimizations.

    import flashzoi_opt
    report = flashzoi_opt.enable("exact")        # or "off"; the default without a name is "exact"; idempotent, once per process
    flashzoi_opt.status()                        # the last activation report

or, without code changes, `FLASHZOI_OPT=<mode> python my_script.py`: the package's .pth installs a lazy import hook that activates
the mode the first time the upstream package (`borzoi_pytorch`) is imported. Nothing is imported at interpreter start beyond this
module; torch and the upstream package load only when a mode is activated.

Flashzoi has no command line upstream: the model is a Python API (`Borzoi.from_pretrained` + the model's forward, or the package's
`pytorch_borzoi_helpers.predict_tracks` function), and the kit is a Python-API kit that attaches to a LOADED model
(`kit.KitRunner(model)`). Activation is therefore in two steps: `enable()` resolves and gates the mode and arms a class-level hook on
the stock model's forward; the kit's own apply line runs on each model instance at its first forward on a CUDA device (stack.py) —
eagerly by `flashzoi-opt pred`, or lazily under the hook. An instance on a CPU or other non-CUDA device is refused by name at that
point (never a silent stock forward under an active mode).

Modes (`modes.MODES`): "exact" = `KitRunner(model)` at the kit's defaults — every component in the kit's own LEVERS with
numerics='tf32' (the stock's TF32 class set inside each call and restored); the kit's claim: byte-identical to stock at torch's default
numerics — the package DEFAULT (`modes.DEFAULT_MODE`), also selected with --mode exact / FLASHZOI_OPT=exact / enable('exact'); "off" = stock
(the tree's stock caller in a clean subprocess). `registry.components()` describes each component from the kit's own table; the kit's own code
applies them.

The contract: `enable(mode)` returns the activation report (a dict; `active` says whether the mode is armed, `reason` says why not),
`status()` returns the last report. Late activation: `enable()` may run before or after the upstream package is imported and after
model instances exist, but is refused by name once a kit runner exists in the process under other knobs; repeated calls return the
first report, and a different mode in the same process is refused. `check` (CLI) is the same resolution as a dry run: it gates the
mode on this machine and applies nothing.
"""
__version__ = "0.1.1"
__all__ = ["enable", "status", "MODES", "ActivationError", "registry", "modes", "stack", "manifest", "settings"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (mode not shipped, pins, no GPU, kit missing, a runner under other knobs, or the
    kit's apply line failed), or a hooked forward reached an instance the mode cannot serve (a non-CUDA device)."""


def enable(mode: str = None, *, strict: bool = False, trigger: str = None) -> dict:
    """Activate `mode` ("exact" | "off"; default: FLASHZOI_OPT from the environment, else the package default "exact") in this
    process; returns the activation report {"active", "mode", "components_applied", "components_fallback", "components_unavailable", "partial",
    "numerics", "route_class", "class_label", "gpu": {"name", "cc", ...}, "borzoi_pytorch_version", "package_version", "reason"?, ...}.
    Idempotent: repeated calls return the first report; a different mode in the same process is refused. `strict=True` raises
    ActivationError instead of returning an inactive report."""
    from . import stack
    return stack.activate(mode, strict=strict, trigger=trigger)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    from . import stack
    return stack.status()


def __getattr__(name):                                   # PEP 562: keep `import flashzoi_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in ("registry", "modes", "stack", "manifest", "settings", "inputs", "outputs", "report", "weights", "det",
                "cli", "warm", "_autoload", "loop", "multi"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
