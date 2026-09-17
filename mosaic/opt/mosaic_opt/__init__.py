"""mosaic_opt — explicit interface to the mosaic (Boltz-2/joltz binder hallucination) inference optimizations.

    import mosaic_opt
    report = mosaic_opt.enable("exact")      # or "off"; idempotent, once per process
    mosaic_opt.status()                      # the last activation report

or, without code changes, `MOSAIC_OPT=exact python my_script.py`: the package's .pth installs a lazy import hook that activates the
mode the first time the upstream package (`mosaic`) is imported. Nothing is imported at interpreter start beyond this module; jax
and the upstream packages load only when a mode is activated.

mosaic has no command line upstream: the model is a Python API (`Boltz2()`, `binder_features`, `build_loss`, `simplex_APGM`,
`predict`), and the kit is a set of add-only functions plus one driver script that composes them
(`opt/mosaic_opt/tools/public_design_run.py`) through the kit's recipe module `tools/recipe.py` — the notebook recipe's
literals and calls, once; `mosaic_opt.recipe` imports that module for callers that build a design step through it. The kit's levers
attach in two different ways:

* P1 (persistent compilation cache + autotune pin) attaches at the process boundary — the environment row exported before the driver
  starts (`modes.ROWS`) — or in-process by the kit's own `repro_cache.enable(DIR)` before the first jax
  computation. `enable()` uses the second form; `mosaic-opt design` uses the first.
* P2 (fast weight load) and P3 (frozen features) replace two calls at the call site — `Boltz2()` by
  `mosaic.fast.fastload.load_stock_fast_init()`, `model.binder_features(...)` by the driver's inline load of the frozen npz after its sha256
  check (the driver's `--features-in` branch through `tools/recipe.py` `load_frozen_features`, the in-line equivalent of `frozen.load_features`, which no row imports) — which the
  driver selects with its own flags (`--weights fastinit`, `--features-in NPZ --features-sha SHA`). No import hook can apply them to a
  program that makes the stock calls, so on the in-process route (`enable()`, `MOSAIC_OPT=<mode>`) they STEP ASIDE BY NAME: the ACTIVE
  line's row token names them (`-aside[P2:driver_only]`, exact: `-aside[P2:driver_only,P3:driver_only]`), the report lists them under
  `levers_driver_only`, and every other lever of the row applies — not a partial activation (`partial` stays false, `unavailable=none`), no
  exit. `partial` means a lever the route should apply and did not: the driver route's gate (`mosaic-opt design`: a lever of the row the
  driver's manifest does not show applied exits 3 unless `--allow-partial`, recorded as `allow_partial`); in-process the activation refuses
  by name instead (NOT ACTIVE), and MOSAIC_OPT_ALLOW_PARTIAL=1 stays that route's opt-out should a report come back partial.

Modes (`modes.KIT_MODES`; `modes.MODES` = the words served): "exact" = P1 + P2 + P3, the warm row `C_p1warm_p2`, "off" = stock (the driver
with every lever off in a clean subprocess, `stock_design.py`), "fast" (the fast-class per-step levers, the package default) and "big" (the
memory tier) — the kit's vocabulary (`modes.WORDS`); a tier word is served — present in `MODES`, composed by its row `T_fast` /
`U_big` — once a per-step lever is wired into it, and refused by name otherwise (`modes.NOT_WIRED`; `mosaic/CHANGES.md` "What is not wired"). Per-step levers have ONE installer, `mosaic_opt.levers.install(mode)`
(in-process: `enable()`, the `.pth` hook; in the driver: its `--levers` flag); `registry.LEVERS` describes each lever; the kit's own code applies them.

The contract: `enable(mode)` returns the activation report (a dict; `active` says whether the levers are on, `reason` says why not),
`status()` returns the last report. Late activation: `enable()` may run any time after `mosaic` / `jax` are imported, but is refused by
name once the JAX backend is initialised in the process (XLA_FLAGS — the autotune dump/load — are read at backend initialisation), once a
`Boltz2` instance exists (its executables compiled without the cache; instances are counted by a constructor wrap from activation on,
`stack.instance_check`), or once the kit's own state shows P1 already on (`stack.kit_levers_applied`); repeated calls return the first
report, and a different mode in the same process is refused. `check` (CLI) is the same resolution as a dry run: it gates the mode on
this box and applies nothing.

The core pin gate (`core_gate()` = `_core_gate.gate`, the shared core's kit template carried byte for byte as `mosaic_opt/_core_gate.py`) is
statement one of every entry — `python -m mosaic_opt` / `mosaic-opt`, `enable()` / `status()`, the package's core-backed attributes
(`mosaic_opt.stack` / `.report` / `.det` / `.warm` / `.levers`, `CORE_BACKED`), the `.pth` hook under a set `MOSAIC_OPT`, `run.sh`,
`configs/h100.env`: an absent or older `opt_core` (against `opt/pyproject.toml` `[tool.opt_core]`, a path + minimum-version floor) is one
`[mosaic-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: …` line and exit 3 before anything of the
core is imported (a newer core passes). Importing the package itself, `MODES` and the `CORE_FREE` attributes stay inert and core-free.
"""
__version__ = "0.3.24"
__all__ = ["enable", "status", "core_gate", "MODES", "TAG", "ActivationError", "registry", "modes", "levers", "stack", "settings", "inputs", "recipe"]
TAG = "mosaic-opt"                                       # the kit's line tag: every printed line is `[mosaic-opt] ...` (report.PREFIX, the .pth hook's AutoloadSpec, the core pin gate, the generated .pth)


class ActivationError(RuntimeError):
    """A requested mode could not be activated (pins, no GPU, kit missing, cache root unset, or the process is past the point of activation)."""


def core_gate() -> dict:
    """THE core pin gate (`_core_gate.gate`, the core's kit template), statement one of every entry: compares `opt/pyproject.toml`
    `[tool.opt_core]` (path, version — a minimum-version floor) with the `opt_core` this interpreter would import — located, not imported; its
    `__version__` read from disk — and refuses by name on an absent or older core: one `[mosaic-opt] NOT ACTIVE: reason=...` line
    on stderr and `SystemExit(3)` (`_core_gate.CoreGateRefused`), never a traceback, never a silent stock run. Returns the facts on a match
    ({"pinned": {path, version, pyproject}, "installed": {package_dir, root, version}, "tag"});
    cheap and idempotent — activation fills its report's `core` block from the same producer. Imports nothing of the core."""
    from ._core_gate import gate
    return gate(__file__, tag=TAG)


def enable(mode: str, *, cache_dir: str = None, strict: bool = False, trigger: str = None) -> dict:
    """Activate `mode` (a word of `MODES`: "fast" | "exact" | "off"; "big", the memory tier) in this process; returns the activation report {"active", "mode", "route", "levers_applied",
    "levers_unavailable", "partial", "levers_driver_only" (the row's call-site levers, stepping aside by name on this route), "levers_aside",
    "p1": {"cache_dir", "autotune_file", "autotune": "dump"|"load"}, "gpu": {"name", "cc", "sm"},
    "upstream": {...versions}, "package_version", "reason"?, ...}. `cache_dir` names the P1 directory (default: MOSAIC_OPT_CACHE_DIR,
    else <MOSAIC_OPT_CACHE_ROOT>/campaign). Idempotent: repeated calls return the first report; a different mode in the same process is
    refused; refused once the JAX backend is initialised or a Boltz2 instance exists. `strict=True` raises ActivationError instead of
    returning an inactive report. Statement one is the core pin gate (`core_gate`): a core not at the pin is its NOT ACTIVE line and SystemExit(3)."""
    core_gate()
    from . import stack
    return stack.activate(mode, cache_dir=cache_dir, strict=strict, trigger=trigger)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run (after the core pin gate: `stack` imports the core)."""
    core_gate()
    from . import stack
    return stack.status()


CORE_BACKED = ("stack", "report", "det", "warm", "levers")   # the lazy attributes whose modules import the core at module level: served after the core pin gate (levers: the ONE per-step lever installer, LEVER lines through opt_core.report)
CORE_FREE = ("registry", "modes", "settings", "inputs", "outputs", "stock_design", "recipe", "_autoload")   # served as they are: nothing of the core loads with them (recipe: the accessor of the kit's tools/recipe.py)


def __getattr__(name):                                   # PEP 562: keep `import mosaic_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in CORE_BACKED or name in CORE_FREE:
        if name in CORE_BACKED:
            core_gate()                                  # `mosaic_opt.stack`, `from mosaic_opt import report`, `from mosaic_opt import *`: one NOT ACTIVE line and SystemExit(3) on a core not at the pin, never a bare ModuleNotFoundError
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
