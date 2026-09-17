"""chai1_opt — explicit interface to the Chai-1 inference optimizations.

    import chai1_opt
    report = chai1_opt.enable("exact")        # or "fast" / "off"; idempotent, once per process
    chai1_opt.status()                        # the last activation report

or, without code changes, `CHAI1_OPT=<mode> python my_script.py`: the package's .pth installs a lazy import hook that activates the
mode the first time the upstream model module (`chai_lab.chai1`) is imported. Nothing is imported at interpreter start beyond this
module; torch and the upstream package load only when a mode is activated.

Chai-1 upstream is a Python API (`chai_lab.chai1.run_inference`; the `chai-lab fold` command is that function under typer) and the
driver kit is a DRIVER script (`kit/chai_worker.py`) whose levers are its own `--levels` list: three module attributes
patched at its import (W1, W5) and feature-context reuse in its loop (W2). Two more kits
ride on it: the eager stack (`chai1_eager`, the exported model re-expressed as eager PyTorch, installed through its own loader) and the
DSTEP add-on (`chai1_fastln`, the eager denoiser step's levers `hoist2` / `compiled` / `dit_attn`, built through its own `build_lever_parts`). The route of
record is `chai1-opt pred --mode <mode> ...`, which runs the driver as shipped and installs the mode's eager line in the driver process;
`enable()` serves a program that calls `run_inference` itself, by executing the kit's own W1/W5 statements in that program and then
installing the same line (stack.py) — W2 is then not applicable.

Modes (`modes.KIT_MODES`): exact = `--levels W1,W2,W5` + the eager stack's `tier1` line + `templ_empty` (byte-identical to stock
under the deterministic recipe on one and the same torch / CUDA stack — the pinned stack, `modes.PINNED_STACK`;
on another stack the run carries the STACK record), fast = exact's composition + the add-on's `compiled` and `dit_attn` + the shared core's fused TriMul `v4trimul` and triangle attention `triattn` + `msa_rows` + the implied opt-in `tf32`
(tier 2; runs at any fold settings; its Triton kernels build on the pinned stack), big = fast + the memory line (big.py), off = stock
(the upstream API in a clean process, `stock_fold.py`). An out-of-memory error propagates out of every mode: no lever is disabled,
nothing is rerouted to stock or retried on OOM (`opt_core.oom.is_oom` re-raised first at every served-path handler).
Default mode `modes.DEFAULT_MODE` = fast; refused by name where its kernels cannot build (README.md, Modes). Fold settings: stock's own run_inference
flags with stock's defaults, passed through on every mode (settings.py); the deterministic recipe: det.py (`--det 0|1`; `enable(mode, det=1)` applies it
in-process, the .pth route reads the kit's own `CHAI_DETERMINISTIC`); observability: report.py; outputs: outputs.py.
"""
__version__ = "0.4.33"

__all__ = ["enable", "status", "ActivationError", "__version__"]


class ActivationError(RuntimeError):
    """A mode could not be activated (strict mode); the reason is the message and the NOT ACTIVE line has been printed."""


def enable(mode: str, *, strict: bool = False, trigger: str = None, det: int = 0, allow_partial: bool = False, n_gpu=None) -> dict:
    """Activate `mode` in this process. Returns the activation report {"active": bool, "mode", "levels", "levers_applied",
    "levers_not_applicable", "route", "chai_lab_version", "torch_version", "gpu", "reason"?, ...}. Idempotent: repeated calls
    return the first report; a different mode in the same process is refused (levers patch process-wide); refused once the traced
    ESM is loaded or `chai1.load_exported` is no longer upstream's (allowed any time after `chai_lab.chai1` is imported, before the
    first fold). `det=1` applies the deterministic recipe in this process right after the kit's lever statements (det.apply; the
    report's "det_applied"). The row's implied levers (modes.KitMode.implied_optin) ride with the mode. `n_gpu` = GPUs per fold (None = 1; P ∈ {1} on chai1: a larger P is refused
    by name, ngpu.py). A PARTIAL activation (a lever of the mode the kits' state does not show applied — the run's own switches turned
    it off) is refused by name unless `allow_partial=True` (pred takes --allow-partial; the CHAI1_OPT environment route reads CHAI1_OPT_ALLOW_PARTIAL=1,
    that route only):
    then it proceeds, named on a NOTE line and recorded in the report (partial, levers_off, partial_note, allow_partial). `strict=True`
    raises ActivationError instead of returning an inactive report."""
    from . import stack
    return stack.activate(mode, strict=strict, trigger=trigger, det=det, allow_partial=allow_partial, n_gpu=n_gpu)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    from . import stack
    return stack.status()


def __getattr__(name):                                   # PEP 562: keep `import chai1_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in ("registry", "modes", "stack", "settings", "inputs", "outputs", "report", "det", "precision", "alloc", "hoist", "stock_fold", "driver",
                "warm", "cli", "ngpu", "big", "_autoload"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
