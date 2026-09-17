"""af2ig_opt — explicit interface to the AF2 initial-guess (dl_binder_design) inference optimizations.

The tool is a command line: the kit's PyRosetta-free driver `af2_initial_guess/predict_pdb.py` (the kit's patch series on upstream
dl_binder_design), whose levers are its own opt-in flags. A mode (`MODES`: off | exact | fast | big) is a flag set composed from the
kit's statement of it (`modes.py` over MANIFEST.json `levers`) and run as the driver's process by `af2ig-opt pred --mode M` (`cli.py`);
the stock line (`off`) is the same driver with no lever flag, run by the stock caller `stock_cli.py` in a clean subprocess that proves
its environment. Nothing attaches to a running process: a lever of this kit exists only as a flag parsed by the driver at its start, so
the package has no `enable()` and no import hook — `AF2IG_OPT=<mode>` is read by the command line (and `run.sh`) as the mode when `--mode`
is absent. `check` is the dry run (would the mode activate here, and why not); a gate that fails names itself (`ActivationError`, exit
3; `stack.py`); a lever of the mode without the driver's evidence of it after a run is a partial activation (exit 3 unless
`--allow-partial`; `stack.applied`).
"""
__version__ = "0.8.2"
TAG = "af2ig-opt"                                   # the tag of every line this package prints ([af2ig-opt] ACTIVE / NOT ACTIVE / LEVER / EXIT …)
__all__ = ["MODES", "ActivationError", "registry", "modes", "stack", "manifest", "det"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (pins, checkout, weights, kit missing)."""


def __getattr__(name):                                   # PEP 562: keep `import af2ig_opt` free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in ("registry", "modes", "stack", "manifest", "det", "report", "stock_cli", "warm", "cli", "ccache", "programs"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
