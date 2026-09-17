"""caliby_opt — explicit interface to the Caliby / SolubleCaliby sequence-design inference optimizations.

    import caliby_opt
    report = caliby_opt.enable("fast", variant="single")    # single: "fast" | "off"; ensemble32: "exact" | "off"; once per process, BEFORE `import caliby`
    caliby_opt.status()                                     # the last activation report

or, without code changes, `caliby-opt design --mode fast --variant single --input <structures> --out_dir <out>` (`run.sh design`),
or `CALIBY_OPT=fast` in the environment of that command (the same flag by another route).

Caliby has no command line upstream: the model is a Python API (`caliby.clean_pdbs` -> `caliby.load_model` -> `model.sample`), and
the kits are files that replace six upstream modules (plus one helper and two Protpardelle-1c modules). `enable(mode,
variant)`, in the process that will import caliby and before it does, proves the installed tree is the pinned upstream (both modes),
and for `exact` proves the kits' files present and loads them under the upstream module names by import hook
(`overlay.py`) — site-packages is never written — then exports the mode's row of kit switches, prints one activation line and
registers the exit tally. It is refused by name once any module the kits replace is imported (`import caliby` alone imports
`caliby.api`). The shared core this package runs on (`opt_core`, pinned in `opt/pyproject.toml` `[tool.opt_core]`) is imported at
activation, never at interpreter start, and only behind the core pin gate (`stack.core_gate` -> `_core_gate.py`, the core's kit
template): statement one of every entry — `python -m caliby_opt`, the console script, `run.sh`, `configs/h100.env`, the `.pth` route's
trigger, the design children, `enable()` / `status()` / `check()` here — refuses an absent, older/newer or edited core by name
(`[caliby-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …`, exit 3) before anything of the
core is imported. Importing this package stays inert and core-free.

Modes (`modes.MODES`: off | exact | fast): each variant has one kit ROW, stated once in the mode table (`modes.ROWS`) — "single"'s
is tier 2 ("fast": the kit's own activation rows, deterministic, the forward-pass band of stock), "ensemble32"'s is tier 1 ("exact":
stock's outputs; "fast" on ensemble32 runs that same row); "off" = stock (the pinned upstream tree, no switch set, proven by environment
and tree digest). "exact" on single (no bit-identical row there) and an unknown name are refused by name
(`modes.mode_refusal`). Without `--mode` / CALIBY_OPT the command line runs "fast" on both variants (`modes.default_mode`; never exact); the
`.pth` route of a script of your own activates under CALIBY_OPT=<mode> only — unset, that script runs stock. Variants (`modes.VARIANTS`):
"single" (structure-conditioned design) and "ensemble32" (design conditioned on 32 Protpardelle-1c conformers per structure) — one variant
per process. `registry.LEVERS` describes each switch; the kits' own code applies them.
"""
__version__ = "0.3.2"
__all__ = ["enable", "status", "check", "MODES", "VARIANTS", "ActivationError", "registry", "modes", "stack", "tree", "overlay", "manifest", "settings"]

# Nothing below imports a submodule at package import: the autoload .pth imports this package at every interpreter start, and the
# submodules (which read the tree, and torch on activation) load on first use.
import importlib as _importlib


class ActivationError(RuntimeError):
    """A requested mode could not be activated (pins, tree state, no GPU, kit missing, late activation, variant conflict)."""


def enable(mode: str, variant: str = None, *, clean_workers: int = None,
           strict: bool = False, trigger: str = None, ckpt: str = None) -> dict:
    """Activate `mode` (single: "fast" | "off"; ensemble32: "exact" | "fast" | "off" — "exact" on single raises ActivationError with
    the mode table's words) for `variant` (default single) in this process — see activate.py. `clean_workers` selects the
    single row (modes.resolve); `ckpt` names the run's sequence-design checkpoint (any name or path upstream's load_model takes; the
    weights gate requires files only for the names stock/PINS.json lists). Statement one is the core pin gate."""
    from .stack import core_gate
    core_gate()
    from . import activate
    return activate.enable(mode, variant, clean_workers=clean_workers, strict=strict, trigger=trigger, ckpt=ckpt)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} before any activation. Statement one is the core pin gate
    (activate.py imports the core's report module)."""
    from .stack import core_gate
    core_gate()
    from . import activate
    return activate.status()


def check(mode: str, variant: str = None, clean_workers: int = None, need_gpu: bool = True, *,
          env: dict = None, ckpt: str = None) -> dict:
    """Dry run of enable(): resolve and gate the mode on this box, apply nothing (`env`: gate that environment instead of this process's).
    Statement one is the core pin gate."""
    from .stack import core_gate
    core_gate()
    from . import activate
    return activate.check(mode, variant, clean_workers, need_gpu=need_gpu, env=env, ckpt=ckpt)


def __getattr__(name):
    if name in ("MODES", "VARIANTS", "KIT_MODE_OF", "DEFAULT_VARIANT"):
        return getattr(_importlib.import_module(".modes", __name__), name)
    if name in ("registry", "modes", "stack", "tree", "overlay", "manifest", "settings", "activate", "report", "cli"):
        return _importlib.import_module("." + name, __name__)
    raise AttributeError(name)
