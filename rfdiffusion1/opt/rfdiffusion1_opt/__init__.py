"""rfdiffusion1_opt — explicit interface to the RFdiffusion-1 inference optimizations.

    import rfdiffusion1_opt
    report = rfdiffusion1_opt.enable("exact")                 # or "off" | "fast"; idempotent, once per process
    rfdiffusion1_opt.status()                                  # the last activation report
    rfdiffusion1_opt.run_design("exact", ["inference.input_pdb=t.pdb", "contigmap.contigs=[A1-150/0 70-100]", "inference.output_prefix=out/des"])   # the designs, the same as `rfdiffusion1-opt design`

or, without code changes, ``RFDIFFUSION1_OPT=<mode> rfdiffusion1-opt design inference.input_pdb=… 'contigmap.contigs=[…]' inference.output_prefix=…`` (upstream's own overrides).

RFdiffusion-1's optimizations live in the control flow of the kit's resident driver (opt/forward/fast_inference/drivers/rfd_bench.py):
a resident process, the memoised constants, the CUDA-graph wrapping of the instantiated model. None of that can attach to the stock command line's
one-process-per-invocation shape, so a kit mode is a *driver line*: `enable()` resolves and gates the mode on this box and arms it
for `design()`, which runs the line in its own process and reads the kits' own evidence lines back. `RFDIFFUSION1_OPT=exact` on the
stock command line (`scripts/run_inference.py`) is therefore refused by name at the first import of `rfdiffusion` (the .pth hook,
_autoload.py) instead of running stock silently under a kit mode's name; `RFDIFFUSION1_OPT=off` leaves the stock line untouched.

Modes (modes.MODES, the one mode table): "off" = stock (the upstream command line in a clean subprocess); "exact" = the base kit's
production line K on its resident driver; "fast" = K + the SE(3) add-on's Triton line (lever T2, RFD_SE3FAST=t2 in the mode's
environment row, armed in the driver process by driver_run.arm_se3fast), the Triton row LayerNorm (K2) and TF32 GEMMs (TF32) — tier 2,
run-to-run deterministic, not byte-equal to stock. The default mode (no `mode` on the package's CLI) is `modes.default_mode()`: fast
(`exact` by name for byte-equal equality; RFDIFFUSION1_OPT unset leaves the env route inert).
The served line (`rfdiffusion1-opt design --pack K`, serve.py): the mode's line in K resident workers under uncapped CUDA MPS through
common/mps_packing/mps_workers.sh, on the same driver (modes.resolve(served=True): each worker's process history the stock command
line's); K is an axis of that line, never a mode.
registry.LEVERS describes each lever; det.py holds the deterministic recipe; the kits' own code applies everything. The package stands on
the shared core (common/opt_core, pinned in opt/pyproject.toml [tool.opt_core]): the autoload finder, the gate probes and sums rule, the
JIT-cache-key grammar, the exit-tally hook and the recipe shape are the core's; every line, mode name and lever of this package is its own.
The core pin gate (`_core_gate.py`, the core's kit template byte for byte; `gate(__file__)` is statement one of every entry — `python -m
rfdiffusion1_opt` and the console script (`__main__.py`), `run.sh` and `configs/h100.env` (their first probes), the `.pth` route
(`_autoload.install` under a set RFDIFFUSION1_OPT) and the in-process `enable()` / `status()` / `run_design()` below) refuses by name with
exit 3 — `[rfdiffusion1-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …` — on an absent, older /
newer or edited core before anything of the core is imported; `import rfdiffusion1_opt` itself stays inert and imports nothing of the core.
`__all__` is that API; the submodules import by name, and the ones whose import reaches the core (`stack`, `warm`) pass the
same gate when reached as attributes of the package (`rfdiffusion1_opt.stack`).

The contract: `enable(mode)` returns the activation report (a dict; `active` says whether the mode is armed, `attach`
= driver | stock-cli, `reason` says why not), `status()` returns the last report. Late activation: `enable()` may run any time
after import and is refused by name once an upstream model instance exists in this process or a kit lever module reports itself
applied here (stack.instance_check / kit_levers_applied); repeated calls return the first report; a different mode in the
same process is refused. `check` (CLI) is the same resolution as a dry run (exit 3 with the NOT ACTIVE line where the box would refuse).
The exit rule (report.verdict; the codes' one home is report.py): a pass whose levers left no evidence line, or printed a forbidden
line, is `partial` — `[rfdiffusion1-opt] NOT ACTIVE: partial activation — <levers>: <reason>; exit 3`, the outputs kept and the levers
named in the manifest (`partial`, `partial_reason`).
"""
__version__ = "0.7.3"
__all__ = ["enable", "status", "run_design", "MODES", "ActivationError", "__version__"]   # the API; submodules import by name (`rfdiffusion1_opt.<module>`)
_SUBMODULES = ("registry", "modes", "stack", "det", "manifest", "outputs", "report", "stock_cli", "warm", "_autoload", "cli")   # resolved on attribute access (__getattr__)
_CORE_BOUND = ("stack", "warm")                           # of those, the ones whose import reaches opt_core at module level: attribute access passes the core pin gate first


class ActivationError(RuntimeError):
    """A requested mode could not be activated (unknown, no GPU, no checkout / weights, kit missing, tools missing, late activation, mode conflict)."""


def enable(mode: str = None, overrides=None, *, strict: bool = False, trigger: str = None) -> dict:
    """Resolve, gate and arm `mode` ("off" | "exact" | "fast", modes.MODE_NAMES — any other name is refused by name; default: RFDIFFUSION1_OPT,
    else modes.default_mode()) in this process; returns the activation report {"active", "mode", "tier", "attach",
    "levers_planned", "levers_applied", "levers_fallback", "levers_unavailable", "partial", "env", "line", "gpu": {"name", "cc", "mem_gib",
    "key", "class"}, "stack", "tools", "rfdiffusion_version", "package_version", "reason"?, ...}. The applied / unavailable /
    partial keys are filled after each driver pass from the kits' own evidence lines (stack.record_pass; the levers are applied by the
    kit's driver in its own process); levers_fallback stays empty (the driver refuses instead of falling back). Idempotent per process.
    `strict=True` raises ActivationError instead of returning an inactive report."""
    from ._core_gate import gate
    from .report import TAG
    gate(__file__, tag=TAG)                               # the core pin gate: statement one of the entry, before any opt_core import (exit 3 by name)
    from . import stack
    return stack.activate(mode, overrides, strict=strict, trigger=trigger)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    from ._core_gate import gate
    from .report import TAG
    gate(__file__, tag=TAG)                               # the core pin gate: statement one of the entry, before any opt_core import (exit 3 by name)
    from . import stack
    return stack.status()


def run_design(mode: str = None, overrides=None, **kw) -> dict:
    """The `design` verb as a call: the manifest dict (its "status" is ok | partial | failed | incomplete | refused | dry-run; its
    "exit_code" the exit rule's code). Input as the `design` verb: upstream's hydra overrides (`overrides`, one target — scripts/run_inference.py's
    own KEY=VALUE tokens); keyword arguments as design.run (timeout, dry_run, det_flag)."""
    from ._core_gate import gate
    from .report import TAG
    gate(__file__, tag=TAG)                               # the core pin gate: statement one of the entry, before any opt_core import (exit 3 by name)
    from . import design as _design
    _rc, man = _design.run(mode, overrides, **kw)
    return man


def __getattr__(name):                                   # PEP 562: keep `import rfdiffusion1_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in _SUBMODULES:
        if name in _CORE_BOUND:
            from ._core_gate import gate
            from .report import TAG
            gate(__file__, tag=TAG)                           # `rfdiffusion1_opt.stack` (et al.) as an attribute imports the core: past the gate, never a raw ModuleNotFoundError
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
