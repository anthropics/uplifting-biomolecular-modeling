"""pxdesign_opt — explicit interface to the PXDesign inference optimizations.

    import pxdesign_opt
    report = pxdesign_opt.enable("exact")     # or "off"; idempotent, once per process
    pxdesign_opt.status()                     # the last activation report

or, without code changes, `PXDESIGN_OPT=exact pxdesign infer ...`: the package's .pth installs a lazy import hook that activates
the mode the first time the upstream package `pxdesign` is imported. Nothing is imported at interpreter start beyond this module;
torch, the upstream packages and the kit load only when a mode is activated and applied.

The lever (`opt/forward/hoist/pxd_xattempt/hoist.py`) patches Protenix module classes process-wide and acts on a loaded
model: `hoist.install(model)` tags the model's diffusion module and is the kit's own activation point — after the checkpoint is
loaded (`InferenceRunner.load_checkpoint`). Activation is
therefore in two steps: `enable()` resolves and gates the mode, exports the mode's switches and arms the hook on
`pxdesign.runner.inference.InferenceRunner.load_checkpoint`; the lever is applied to the runner's model by the kit's own `install()`
when that method returns (`stack.apply_to(runner)` does the same eagerly).

Modes (`modes.KIT_MODES`): "exact" = the kit's tier-1 lever set (`PXD_HOIST=1 PXD_HOIST_MODE=shape PXD_HOIST_MASK=1`; selected by name;
byte-identical to stock), "fast" = the package default (`modes.DEFAULT_MODE`): speed inside the engine's tier-2 band
(exact's levers with `PXD_HOIST_MODE=rows`, the size levers featdiet + padmask, the numerics policy `tf32` and the single-conditioning row
dedup `sdedup`; `precision.py`, `sdedup.py`),
"big" = the memory-reach line, never the default (exact's levers with `PXD_HOIST_MODE=rows` plus the package's size levers featdiet +
padmask, `sizeceil.py`, and rowpipe, `rowpipe.py`; 2788 (`exact`) → 5988 tokens (`big`) on one 80 GB H100 at N_sample 5; fast-class numerics, tier 2),
"off" = stock (the upstream CLI, or the in-process upstream route under the
deterministic recipe, in a clean subprocess). No `fast` mode ships for this model and there is no variant axis; `PXDESIGN_OPT` unset = off;
any other name is refused by name (`modes.unknown_message`).
The package runs on the shared core `opt_core` (`common/opt_core`, pinned in `opt/pyproject.toml` `[tool.opt_core]`): the autoload finder,
the mode table's check, the gates, the stock proof, the line primitives and the manifest writer are the core's; every value in them is this
kit's. The core pin gate (`core_gate()` below: `_core_gate.gate`, the core's `kit_template/_core_gate.py` carried byte-for-byte inside this
package) is statement one of every entry — `python -m pxdesign_opt` / `pxdesign-opt` (`__main__.py`), `run.sh` and `configs/h100.env` (their
first `python` probes), the `.pth` route (`_autoload.py`, when `PXDESIGN_OPT` names a kit mode), the stock caller's child (`stock_infer.py`),
`cli.py` run directly, `enable()` / `status()` here and the lazy attributes that import a core-reaching module (`MODES`, `DEFAULT_MODE`,
`modes`, `stack`, `manifest`, `report`, `stock_infer`; `CORE_ATTRS`): it locates the `opt_core` this interpreter would import without importing it, compares it with the pin, and
refuses by name — one `[pxdesign-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …` line on stderr,
exit status 3 — on an absent, older, newer or edited core, before anything of the core is imported. It is the one producer of the 'pinned X,
installed Y' fact (the activation report's `core_pin` block is its return value); there is no override.
`registry.LEVERS` describes each lever; the kit's own code applies them (see stack.py).

The contract: `enable(mode)` returns the activation report (a dict; `active` says whether the levers are on, `reason` says why not),
`status()` returns the last report. Late activation: `enable()` may run any time after the upstream packages are imported, but is
refused by name once a `ProtenixDesign` model instance exists in the process (a model built before activation would sample without
the lever; instances are counted by a constructor wrap on the model class from activation on, `stack.instance_check`) or once the
kit's own module reports the lever installed (`stack.kit_levers_applied`); repeated calls return the first report, and a different mode
in the same process is refused. `check` (CLI) is the same resolution as a dry run: it gates the mode on this box and applies nothing.
"""
__version__ = "0.7.2"
__all__ = ["enable", "status", "core_gate", "TAG", "MODES", "DEFAULT_MODE", "ActivationError", "registry", "modes", "stack", "manifest", "options"]
TAG = "pxdesign-opt"                                     # the kit's tag on every line it prints (`[pxdesign-opt] …`): the one spelling — report.py, _autoload.py and the core pin gate's `tag=` read it


class ActivationError(RuntimeError):
    """A requested mode could not be activated (no GPU, kit missing, a model instance already built, or the lever failed to apply)."""


def core_gate() -> dict:
    """The core pin gate, statement one of every entry: `_core_gate.gate(__file__, tag=TAG)`. Refuses by name — one
    `[pxdesign-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …` line on stderr, `SystemExit(3)` —
    unless the `opt_core` this interpreter would import (located, not imported) is at least the version `opt/pyproject.toml` `[tool.opt_core]`
    pins; returns the facts `{"pinned": {path, version, pyproject}, "installed": {package_dir, root, version}, "tag"}` on a pass. Imports nothing of the core, touches no `sys.path`; cheap and idempotent (the
    activation report's `core_pin` block is a second call of the same producer, not a second gate)."""
    from ._core_gate import gate
    return gate(__file__, tag=TAG)


def enable(mode: str, *, strict: bool = False, trigger: str = None) -> dict:
    """Activate `mode` ("exact" | "fast" | "big" | "off") in this process; returns the activation report {"active", "mode", "levers_applied",
    "levers_planned", "env", "gpu": {"name", "sm", "cc", "mem_gib"}, "upstream": {...}, "package_version", "core_pin", "reason"?, ...}.
    Idempotent: repeated calls return the first report; a different mode in the same process is refused (the lever patches
    process-wide); refused once a ProtenixDesign instance exists (allowed any time after import, before a model is built).
    `strict=True` raises ActivationError instead of returning an inactive report. The core pin gate runs first (exit 3 by name on an
    absent or mismatched core)."""
    core_gate()                                          # [pxdesign-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …  -> exit 3
    from . import stack
    return stack.activate(mode, strict=strict, trigger=trigger)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run. The core pin gate runs first."""
    core_gate()
    from . import stack
    return stack.status()


CORE_FREE_ATTRS = ("registry", "options", "outputs", "infer_loop", "det", "warm", "_autoload", "stamps")   # submodules that import nothing of the core at module level (the stock caller's and the .pth's)
CORE_ATTRS = ("MODES", "DEFAULT_MODE", "modes", "stack", "manifest", "report", "stock_infer")                # names whose module imports the core at module level: the gate runs first


def __getattr__(name):                                   # PEP 562: keep `import pxdesign_opt` (the .pth) free of any other import; every lazy name that reaches the core is gated
    if name in CORE_ATTRS:
        core_gate()                                      # [pxdesign-opt] NOT ACTIVE: reason=…  -> exit 3, before the core-importing module below
        if name in ("MODES", "DEFAULT_MODE"):
            from . import modes as _modes
            return getattr(_modes, name)
    if name in CORE_ATTRS or name in CORE_FREE_ATTRS:
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
