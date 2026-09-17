"""esmfold2_opt — explicit interface to the ESMFold2 inference optimizations.

    import esmfold2_opt
    report = esmfold2_opt.enable("fast", variant="full_msa")   # or "exact" / "big" / "off"; idempotent, once per process
    esmfold2_opt.status()                                        # the last activation report

or, without code changes, `ESMFOLD2_OPT=<mode> ESMFOLD2_VARIANT=<variant> python my_script.py`: the package's .pth installs a lazy
import hook that activates the mode the first time the upstream model family (`transformers.models.esmfold2`,
`esm.models.esmfold2`) is imported. Nothing is imported at interpreter start beyond this module; torch and the upstream
packages load only when a mode is activated.

ESMFold2 has no command line upstream: the model is a Python API (`ESMFold2Model.from_pretrained` + `ESMFold2InputBuilder.fold`),
and the kit is a driver whose `configure()` installs in-process patches on a loaded model. Activation is therefore in
two steps: `enable()` resolves and gates the mode (and arms the hook), and the levers are applied to each model instance by the
kit's own `configure()` — eagerly by `esmfold2-opt pred`, or at a model's first `fold()` under the lazy hook.

Modes (`modes.KIT_MODES`): "fast" = the kit server's own default mode (tier 2) — the package default (modes.DEFAULT_MODE); "exact" (bitwise-identical to the library's fused backend under the det recipe) = the
kit's bit-exact lever set, server mode `opt7x`, the same line on every variant (selected by name: `--mode exact` / `ESMFOLD2_OPT=exact`);
"off" = stock (the upstream API in a clean subprocess). Variants (`modes.VARIANTS`): "fast", "full_msa", "full_nomsa" — one
variant per process. `registry.LEVERS` describes each lever; the kit's own code applies them (see stack.py).

The contract: `enable(mode, variant=...)` returns the activation report (a dict; `active` says whether the levers are on, `reason`
says why not), `status()` returns the last report. Late activation: `enable()` may run any time after the upstream packages are
imported, but is refused by name once an `ESMFold2Model` instance exists in the process (a model built before activation would fold
unconfigured; instances are counted by a constructor wrap on the model class from activation on, `stack.instance_check`) or once the
kit's own lever modules report a lever applied (`stack.kit_levers_applied`); repeated calls return the first
report, and a different mode or variant in the same process is refused. `check` (CLI) is the same resolution as a dry run: it gates the
mode on this box and applies nothing. A mode is all of its levers on a GPU class: a partial application — a lever of the mode's set the kit's
own records show not applied on this device (`status()["partial"]`, the APPLIED line's fallbacks, the EXIT tally) — refuses `esmfold2-opt pred`
by name before its first fold (exit 3); on this environment route it is recorded in `status()` for the host script to read.
"""
__version__ = "0.3.27"
__all__ = ["enable", "status", "MODES", "VARIANTS", "ActivationError", "registry", "modes", "stack", "settings"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (pins, no GPU, kit missing, variant conflict, or a lever failed to apply)."""


def enable(mode: str, variant: str = None, *, strict: bool = False, trigger: str = None, line: str = None) -> dict:
    """Activate `mode` ("exact" | "fast" | "big" | "off") for `variant` ("fast" | "full_msa" | "full_nomsa"; default: the
    ESMFOLD2_VARIANT environment variable) in this process; returns the activation report
    {"active", "mode", "variant", "server_mode", "levers_applied", "levers_fallback", "gpu": {"name", "sm"}, "esm_version",
    "transformers_version", "package_version", "reason"?, ...}. Idempotent: repeated calls return the first report; a different
    mode or variant in the same process is refused (levers patch process-wide); refused once an ESMFold2Model instance exists
    (allowed any time after import, before a model is built). `strict=True` raises ActivationError instead of returning an inactive
    report. `line` = big's internal composition key (None = the mode)."""
    from . import stack
    return stack.activate(mode, variant, strict=strict, trigger=trigger, line=line)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    from . import stack
    return stack.status()


def __getattr__(name):                                   # PEP 562: keep `import esmfold2_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name == "VARIANTS":
        from .modes import VARIANTS
        return VARIANTS
    if name in ("registry", "modes", "stack", "settings", "inputs", "outputs", "report", "stock_fold", "_autoload"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
