"""rfdiffusion3_opt — explicit interface to the RFdiffusion3 inference optimizations.

    import rfdiffusion3_opt
    report = rfdiffusion3_opt.enable("exact")    # or "fast" | "off"; idempotent, once per process, BEFORE rfd3.model.hoist is imported
    rfdiffusion3_opt.status()                    # the last activation report

or, without code changes, `RFDIFFUSION3_OPT=exact rfd3 design ...`: the package's .pth arms the shared core's autoload finder
(`opt_core.autoload`, this kit's spec in `_autoload.py`), which activates the mode when the upstream package `rfd3` is first imported
(after that package's own body, which imports nothing of `rfd3.model`). Nothing of the package or the core is imported at interpreter
start unless RFDIFFUSION3_OPT names a selection; torch and the upstream package load only when the program imports them. The package
stands on `opt_core`, pinned in opt/pyproject.toml [tool.opt_core]: the core pin gate (`_core_gate.gate` through `_autoload.core_gate()`,
statement one of every entry — `python -m rfdiffusion3_opt` / the console script, `run.sh`, `configs/h100.env`, the `.pth` route under a
set RFDIFFUSION3_OPT, `enable()`, `status()`, the public names that load a core-importing module — `MODES`, `VARIANTS`, `modes`,
`stack`, `report`, `cli`, `warm` — and the `install` / `stock_design` child modules) refuses by name, exit 3 (`[rfdiffusion3-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … |
core_pin_unreadable: …`), on a core that is absent or older than the pinned floor, before anything of the core is imported; importing
the package itself stays inert.

The kit is a file overlay on the installed `rfd3` package (ten files under site-packages/rfd3/model, put there by the
kit's own `install.sh`), and its levers are environment switches read once at import: `RFD3_HOIST` when `rfd3.model.hoist` is imported
(hoist.py:18), `RFD3_FZT` when `rfd3.model.layers.layer_utils` is imported (the fused SwiGLU transition served by `opt_core`),
`RFD3_CUDAGRAPH` at the sampler's construction (the full list per mode: `modes.py`). Activation is therefore an environment delta exported at the process boundary: `enable("exact")` gates the mode
(the kit's files present in the tree and unmodified (`git diff` against HEAD, not a stored digest), this interpreter's rfd3 tree in the kit's `patched` state, upstream at its
pin, a visible GPU) and exports the mode's delta into os.environ; the kit's module reads it at its import. `enable()` is refused by
name once `rfd3.model.hoist` has been imported in the process (the switch was already read: late activation is impossible by
construction), once the kit's module reports the lever in another state, and for a second mode in the same process.

Modes (`modes.MODES`, the one table): "exact" = `RFD3_HOIST=1 RFD3_FZT=1 RFD3_CUDAGRAPH=1 RFD3_INIT_CHUNK=1` (all five hoist parts, the fused
transition served by the shared core's cell table, the graph replay of the eager step, the row-blocked initialiser embedding; byte-identical
designs to stock under the kit's deterministic recipe, `rfdiffusion3/CHANGES.md`), "fast" = the exact delta plus `RFD3_COMPILE=1 RFD3_TOKEN_SDPA=1
RFD3_GATHER_ATTN=1` (upstream's four compile targets compiled by the kit inside the graph replay, the token attention on dense SDPA, the atom
attention through the shared core's gather kernel; small numeric differences within stock's seed-to-seed spread; the DEFAULT mode when none
is named), "off" = stock (the upstream command line in the pristine interpreter, `stock_design.py`). One variant, "design" (`modes.VARIANTS`).

The contract: `enable(mode)` returns the activation report (a dict; `active` says whether the delta is exported, `reason` says why
not), `status()` returns the last report; `check` (CLI) is the same resolution as a dry run: it gates the mode on this box and
exports nothing.
"""
__version__ = "0.6.2"
__all__ = ["enable", "status", "MODES", "VARIANTS", "ActivationError", "modes", "registry", "stack", "manifest", "settings"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (kit not installed in this interpreter, pins, no GPU, late activation, second mode)."""


def enable(mode: str, *, strict: bool = False, trigger: str = None, hook: bool = True) -> dict:
    """Activate `mode` ("exact" | "fast" | "off") in this process: gate it and export its environment delta; returns the activation report
    {"active", "mode", "env", "interpreter", "kit", "gpu", "foundry_version", "package_version", "reason"?, ...}. Idempotent: repeated
    calls return the first report; a different mode in the same process is refused; refused once `rfd3.model.hoist` is imported
    (the switch is read at that import). `strict=True` raises ActivationError instead of returning an inactive report. `hook=False`
    leaves the APPLIED line and the exit rule to the caller (warm's child: the warm verb reads the child's line)."""
    from ._autoload import core_gate
    core_gate()                                          # the core pin gate, before the first opt_core import (stack imports the core): NOT ACTIVE + exit 3 by name
    from . import stack
    return stack.activate(mode, strict=strict, trigger=trigger, hook=hook)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    from ._autoload import core_gate
    core_gate()                                          # the core pin gate, before the first opt_core import
    from . import stack
    return stack.status()


_LAZY = ("modes", "registry", "stack", "manifest", "settings", "install", "det", "report", "stock_design", "warm", "_autoload", "cli")
_CORE_NAMES = ("MODES", "VARIANTS", "cli", "modes", "report", "stack", "warm")   # the lazy names whose import reaches opt_core at module level: gated below (the set is locked by test_core_gate_routes against the modules' import closure)


def __getattr__(name):                                   # PEP 562: keep `import rfdiffusion3_opt` (the .pth) free of any other import
    if name in _CORE_NAMES:
        from ._autoload import core_gate
        core_gate()                                      # the core pin gate, before the import below reaches opt_core: NOT ACTIVE + exit 3 by name
    if name in ("MODES", "VARIANTS"):
        from . import modes as _modes
        return getattr(_modes, name)
    if name in _LAZY:
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
