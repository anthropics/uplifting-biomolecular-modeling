"""protenix_opt — explicit interface to the Protenix v2 inference optimizations.

    import protenix_opt
    report = protenix_opt.enable("exact")      # or "fast" / "big"; idempotent, once per process ("off" = stock: pred --mode off)
    protenix_opt.status()                      # the last activation report

or, without code changes, `PROTENIX_OPT=exact protenix pred ...`: the package's .pth installs a lazy import hook that activates
the mode the first time the protenix model family (`runner`, `protenix.model`) is imported. Nothing is imported at interpreter
start beyond this module; torch and protenix load only when a mode is activated.

Modes: "exact" = E* — the kit README's exact composition for this box's (cc | triton) row (ARM=E env.sh + the row's switches) +
DEADSKIP + lazy init (byte-identical to stock under the deterministic recipe), "fast" = T* — the README's Tier-2 row
(the row's pre switches, ARM=T env.sh + the row's switches: the block core's Tier-2 tri-attention, small-N routing, the shared core's TriMul by tier word), "big" = fast's levers without
the graphs plus the memory levers (big.py; `--n_gpu P` > 1: the row-sharded multi-GPU line, tp.py), "off" = stock (`pred --mode off`
runs the stock CLI in a clean, proven subprocess: stock_pred.py). `MODES` lists the levers per mode; `registry.LEVERS` describes each lever;
the kit's own code applies them (see stack.py). The tree carries three units under opt/forward/ (the FlashPairformer kit with its levers add-on, whose env.sh and
sitecustomize are the route; the row-sharded multi-GPU unit PTX_TP; the sampler add-on DIT_FUSE); `kits` names them and checks their files are present. `enable()` is allowed after `import protenix.model` and refused by name once a model
instance exists (counted by a constructor wrap on the model class from activation on), the kit reports a lever applied, or the
pairformer module is imported.
"""
__version__ = "0.3.63"
__all__ = ["enable", "status", "MODES", "ActivationError", "registry", "modes", "stack", "manifest", "kits", "det"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (version gate, no GPU, kit missing, or a lever failed to apply)."""


def _refuse_without_core() -> None:
    """Statement one of the in-process route (enable / status / the lazy module attributes), before any module of the shared core is
    imported: the pinned copy placed on sys.path when no core is installed, the kit's pre-import core pin gate (``_core_gate``: an absent or
    mismatched core or an unreadable pin prints ``[protenix-opt] NOT ACTIVE: reason=…`` and raises SystemExit(3)), then the module-granular
    producer words. Never a traceback, never a run without the lever."""
    import sys
    from . import _core
    _core.gate()
    why = _core.producer_refusal()
    if why:
        sys.stderr.write(f"[protenix-opt] NOT ACTIVE: {why}\n"); sys.stderr.flush()
        raise SystemExit(3)


def enable(mode: str, *, strict: bool = False, trigger: str = None) -> dict:
    """Activate `mode` ("exact" | "fast" | "off") in this process; returns the activation report
    {"active", "mode", "levers_applied", "levers_fallback", "gpu": {"name", "sm"}, "protenix_version", "package_version", "reason"?, ...}.
    Idempotent: repeated calls return the first report. `strict=True` raises ActivationError instead of returning an inactive report."""
    _refuse_without_core()
    from . import stack
    return stack.activate(mode, strict=strict, trigger=trigger)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    _refuse_without_core()
    from . import stack
    return stack.status()


def __getattr__(name):                                   # PEP 562: keep `import protenix_opt` (the .pth) free of any other import
    if name == "MODES":
        _refuse_without_core()
        from .modes import MODES
        return MODES
    if name in ("registry", "modes", "stack", "manifest", "kits", "det", "_autoload"):
        import importlib
        if name != "_autoload":
            _refuse_without_core()
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
