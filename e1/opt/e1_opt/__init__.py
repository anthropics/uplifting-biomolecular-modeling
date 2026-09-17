"""e1_opt — explicit interface to the E1 inference optimizations.

    import e1_opt
    report = e1_opt.enable("exact", variant="300m")   # or "off"; idempotent, once per process
    e1_opt.status()                                    # the last activation report
    e1_opt.apply(model)                                # a bare E1ForMaskedLM already on the GPU (E1Predictor / E1Scorer need nothing: enable() arms them)
    e1_opt.check("exact", variant="300m")              # dry run: resolves and gates the mode on this box, applies nothing

or, without code changes, `E1_OPT=exact E1_VARIANT=300m python -m E1.tools.score ...`: the package's .pth installs a lazy import hook
that activates the mode the first time the upstream model family (`E1`) is imported — a plain `python -c pass` pays nothing and prints
nothing. The kit (opt/forward/engines/e1/kits/eager, mode word `eager`) applies its ONE lever set to the model at its construction:
`enable()` gates the box and wraps `E1Predictor.__init__` / `E1Scorer.__init__`; the first construction in the process runs the kit's
`apply(model, size=variant)` (its KIT / LEVER lines) and the KERNELS proof. Everything the model then computes is the stock's, bit for bit
(against the pinned stock under the deterministic recipe, `--det 1`); a lever of the set that cannot run on the host is the
mode's refusal by name (NOT ACTIVE, exit 3 — the mode is all of its levers). Anything about the host that is off the kit's pins (a
dependency off its pin, a GPU outside the tested classes) is named on the ACTIVE line (`notes=`), never refused. The mode set: "off" = stock (no component applied; the wrapper command additionally
runs the upstream CLI in a subprocess proven free of kit variables), "exact" = the kit (the package DEFAULT). Variants
(`registry.variants()`): 150m | 300m | 600m — the three pinned checkpoints. `enable()` may run before or after `import E1` but is refused
by name once the kit reports itself applied or an `E1Predictor` / `E1Scorer` instance exists in the process (a model served before the
switch would not be the mode's); a second `enable()` with a different mode or variant in the same process is refused. `check` (CLI) is the
same resolution as a dry run: it gates the mode on this box and applies nothing. The wrapper command `e1-opt score` (cli.py) is the same
resolution around the upstream CLI in a subprocess.
"""
__version__ = "0.2.1"
__all__ = ["enable", "apply", "status", "check", "MODES", "ActivationError", "registry", "modes", "stack"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (pins, stack, card, kit missing, variant conflict, or the kit refused)."""


def enable(mode: str, variant: str = None, *, strict: bool = False, trigger: str = None) -> dict:
    """Activate `mode` ("exact" | "off") for `variant` (default: the E1_VARIANT environment variable) in this
    process; returns the activation report {"active", "mode", "variant", "kit", "kit_mode", "gpu": {"name", "mib", "cc", "sm", "class"},
    "stack", "package_version", "reason"?, ...}. Idempotent: repeated calls return the first report; a different mode or variant in
    the same process is refused. `strict=True` raises ActivationError instead of returning an inactive report."""
    from . import stack
    return stack.activate(mode, variant, strict=strict, trigger=trigger, arm=True)


def apply(model, variant: str = None, det: bool = None) -> dict:
    """Apply the kit's lever set to a bare `E1ForMaskedLM` ALREADY ON THE GPU (`model.to("cuda")` first: the levers build their tables on
    the model's device) — for code that drives `model.forward` itself rather than E1Predictor / E1Scorer (whose construction applies the kit
    on its own after `enable()`). Activates mode exact when the process has not (`enable("exact", variant=...)`), then applies once; returns
    the kit's record (its KIT / LEVER lines printed). A refusal by name raises ActivationError / the kit's own KitRefused."""
    from . import stack
    rep = enable("exact", variant=variant, strict=True)
    rec = stack.apply_kit_once(model, rep["variant"], rep, det=det)
    return rec if rec is not None else (stack.status() or {}).get("applied")


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    from . import stack
    return stack.status()


def check(mode: str, variant: str = None) -> dict:
    """Dry run: resolve and gate `mode` for `variant` on this box (pins, stack, card, kit tree) and apply nothing; the same report
    `e1-opt check` prints (report["dry_run"] is True; report["would_refuse"] names the gate activation would fail)."""
    from . import stack
    return stack.activate(mode, variant, dry_run=True)


def __getattr__(name):                                   # PEP 562: keep `import e1_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in ("registry", "modes", "stack", "outputs", "report", "det", "stock_score", "kit_score", "warm", "cli", "_autoload"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
