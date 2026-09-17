"""chrombpnet_opt — explicit interface to the chrombpnet `pred_bw` optimizations (the kit at `opt/kit_ho`, its torch side carried at `opt/kit`).

    import chrombpnet_opt
    report = chrombpnet_opt.enable("fast")       # or "exact" | "off"; idempotent, once per process; exact composes TensorFlow's determinism settings by its mode
    chrombpnet_opt.status()                        # the last report

or, without code changes, `CHROMBPNET_OPT=fast chrombpnet pred_bw ...`: the package's .pth installs a lazy finder that fires at the first
import of the top-level `chrombpnet` package (the stock console script `chrombpnet = chrombpnet.CHROMBPNET:main`, stock setup.py:27) and,
for the `pred_bw` subcommand, replaces the process with the kit's documented line (`python <kit>/tf/pred_bw_fast.py <the same arguments>`). Nothing is imported at interpreter start beyond this module (PEP 562: the submodules load on first use).

The kit's form is a replacement entry script, not an in-process patch: `enable()` resolves the mode on this machine and records the report
but has nothing to apply inside a running process; the mode engages on the process boundary — `chrombpnet-opt pred_bw --mode fast`
(cli.py) or the environment route (_autoload.py). Modes (modes.MODES): "fast" = the kit's documented line with no extra flags (the kit
decides the route per (GPU class, mode) from its own tables), the package DEFAULT; "off" = the stock CLI in a clean subprocess.
`--mode exact` composes TensorFlow's determinism settings for the kit's process; `--mode off --det 1` / CHROMBPNET_OPT_DET=1 puts the same block on stock
(det.py) on either arm; `chrombpnet-opt pred_bw --items <items.tsv>` serves N regions files in one process (cli.py). Both routes run the
kit's line as a child and take cli.run_fast's path after it: the kit's run record (handed back through the file the package names, folded
into `opt_manifest.json`) is judged by the package (stack.applied) — a lever of the class's set that cannot start makes the kit refuse the
mode by name (exit 3, `NOT ACTIVE mode=fast reason=the kit refused by name: …`); a partial activation by that record (a forward other than
the route the kit's tables give) prints `[chrombpnet-opt] NOT ACTIVE: partial activation — <detail>; exit 3 (--mode off runs stock)`.
"""
__version__ = "0.2.0"
__all__ = ["enable", "status", "MODES", "ActivationError", "registry", "modes", "stack", "manifest", "det", "report"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (kit missing, a route-critical kit file absent, an unknown mode, a second mode)."""


def enable(mode: str, det: bool = False, *, strict: bool = False, trigger: str = None) -> dict:
    """Resolve `mode` ("fast" | "exact" | "off") on this machine and record the report {"active", "mode", "route", "kit_version", "gpu", "det",
    "line", "reason", "package_version", ...}. Idempotent: repeated calls return the first report; a different mode in the same process
    is refused by name, as is an unknown mode. `strict=True` raises ActivationError instead of returning an inactive report."""
    from . import stack
    return stack.activate(mode, det=det, strict=strict, trigger=trigger)


def status() -> dict:
    """The last report, or {"active": False, "reason": ...} when enable() has not run."""
    from . import stack
    return stack.status()


def __getattr__(name):                                   # PEP 562: keep `import chrombpnet_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in ("registry", "modes", "stack", "manifest", "det", "report", "warm", "cli", "_autoload"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
