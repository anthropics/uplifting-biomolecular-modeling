"""protenix_v1_opt — explicit interface to the Protenix v1 (protenix==1.1.0) inference optimizations.

    import protenix_v1_opt
    report = protenix_v1_opt.enable("fast")       # the default; or "exact" / "off"; idempotent, once per process
    protenix_v1_opt.status()                      # the last activation report

or, without code changes, `PROTENIX_V1_OPT=fast protenix pred ...`: the package's .pth installs a lazy import hook that activates
the mode right after the stock CLI's `runner` package is imported (before any model code loads). Nothing is imported at interpreter
start beyond this module; torch and protenix load only when a mode is activated.

Modes (modes.KIT_MODES, the one place that says what a mode means and carries each mode's arm): "exact" = Tier 1, equal to stock bit for
bit under the deterministic recipe; "fast" = Tier 2, same error class as stock; "big" = the memory mode (fast-class numerics, big.py;
`--n_gpu P` its resource axis, ngpu.py); "off" = stock (`pred --mode off` runs the stock CLI in a clean, proven subprocess: stock_pred.py).
The kit's own code applies the levers: `levers_ptx1.bind_model(runner.model)` + `levers_ptx1.apply(<arm>)` on the runner the stock CLI
builds (stack.py wraps `runner.inference.InferenceRunner.__init__`). `enable()` is refused by name once a model instance exists.
"""
__version__ = "0.2.49"
__all__ = ["enable", "status", "MODES", "ActivationError", "modes", "kit", "stack", "det"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (version gate, no GPU, kit missing, or the kit refused an arm)."""


def enable(mode: str, *, strict: bool = False, trigger: str = None, det: bool = False, allow_partial: bool = None, n_gpu: int = None) -> dict:
    """Activate `mode` ("exact" | "fast" | "big" | "off") in this process; returns the activation report
    {"active", "mode", "arm", "levers", "partial", "partial_reason", "allow_partial", "n_gpu", "sharding", "gpu": {"name", "sm"},
    "protenix_version", "package_version", "det", "reason"?, ...}. `n_gpu` = devices for one prediction (None reads
    PROTENIX_V1_OPT_N_GPU, else 1; P>1 only under big — ngpu.py, refused by name otherwise).
    Idempotent: repeated calls return the first report. `strict=True` raises ActivationError instead of returning an inactive report.
    `det=True` installs the kit's deterministic recipe (det.py) before the model code loads. `allow_partial=True` admits a PARTIAL
    activation (a lever of the mode the kit did not apply) with the levers recorded by name; the default refuses it:
    `NOT ACTIVE: partial activation — ...` (report.py, the exit rule)."""
    _producers_gate()
    from . import stack
    return stack.activate(mode, strict=strict, trigger=trigger, det=det, allow_partial=allow_partial, n_gpu=n_gpu)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    _producers_gate()
    from . import stack
    return stack.status()


def _producers_gate():
    """Statement one of the in-process route too: the core pin gate (_core_gate: the importable opt_core is the one this package pins, else
    `NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: …`, exit 3), then the producers gate (_producers: `reason=producer_missing:
    <modules>`, exit 3) — no report can be built without the pinned core, so the process ends by name, as on the CLI and the .pth routes."""
    import sys
    from ._core_gate import gate
    gate(__file__, tag="protenix-v1-opt")
    from ._producers import refuse_if_missing
    refuse_if_missing(exit=sys.exit)


def __getattr__(name):                                   # PEP 562: keep `import protenix_v1_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in ("modes", "kit", "stack", "det", "report", "_autoload"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
