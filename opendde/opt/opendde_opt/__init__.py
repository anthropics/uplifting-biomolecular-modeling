"""opendde_opt — explicit interface to the OpenDDE inference optimizations.

    import opendde_opt
    report = opendde_opt.enable("exact")    # or "fast" / "off" / "big", or a kit line by name; idempotent, once per process
    opendde_opt.status()                       # the last activation report

or, without code changes, `OPENDDE_OPT=exact opendde pred ...`: the package's .pth installs a lazy import hook that activates the
mode the first time the OpenDDE model family (`runner`, `opendde.model`) is imported. Nothing is imported at interpreter start
beyond this module; torch and opendde load only when a mode is activated.

OpenDDE is a command line upstream (`opendde pred`, `runner.batch_inference:opendde_cli`) whose runner is built by
`runner.batch_inference.get_default_runner`. The kits are runtime add-ons loaded by one shim (`sitecustomize.py`) first on
`PYTHONPATH`; `enable()` exports the line's switches, puts the kit directories on `sys.path` in the line's order and executes that
shim — the levers themselves install when the kit's hook sees the runner (see stack.py). A runner built any other way than through
`get_default_runner` gets no lever: that is the hook's scope.

Modes (`modes.MODES`): "exact" = line S1 (the kit's base line S — the served-levers hook loading dit_hoist, dit_align and ARM Z
after the runner exists, and the tuned tile cache — plus FPF TriMul EXACT enabled in-process by the kit's own
`fpf_engines.enable_from_env()`; in-process only), "fast" = line LSTAR2A (S + ARM U + the bf16 TriMul cell +
bf16 DiT attention), "big" = line BIG_F, "off" = stock (the upstream CLI in a clean subprocess). `registry.LEVERS` describes each lever; the kit's own code applies them.

The contract: `enable(mode)` returns the activation report (a dict; `active` says whether the line is armed, `reason` says why not),
`status()` returns the last report. Late activation: `enable()` may run any time after the upstream packages are imported and is
refused by name once a kit shim is active or a runner instance exists.
"""
__version__ = "0.2.72"
__all__ = ["enable", "status", "MODES", "ActivationError", "registry", "modes", "stack", "manifest"]
ENV_MODE = "OPENDDE_OPT"                          # the env route: OPENDDE_OPT=<mode> (read by _autoload.py, run.sh and the CLI)
MODES = ("off", "exact", "fast", "big")      # the house names (README "What the modes promise"), each named by its guarantee; big = memory


class ActivationError(RuntimeError):
    """A requested mode could not be activated (refused by name, version gate, kit missing, late activation, or the shim failed)."""


def enable(mode: str, *, strict: bool = False, trigger: str = None) -> dict:
    """Activate `mode` ("exact" | "fast" | "big" | "off") or a kit line by its own name in this process; returns the activation report
    {"active", "mode", "line", "levers_applied", "levers_unavailable", "levers_fallback", "partial", "gpu", "opendde_version",
    "package_version", "reason"?, ...}. Idempotent: repeated calls return the first report. `strict=True` raises ActivationError
    instead of returning an inactive report."""
    from ._core_gate import gate
    gate(__file__, "opendde-opt")                            # THE pin gate: an absent / mismatched opt_core is its NOT ACTIVE line and SystemExit(3), before any core import
    from ._producers import refusal                          # then the finer words: a producer module this package imports is absent
    r = refusal()
    if r:
        import sys
        sys.stderr.write(r + "\n"); sys.stderr.flush()
        if strict:
            raise ActivationError(r)
        return {"active": False, "mode": mode, "reason": r.split(" NOT ACTIVE ", 1)[1], "exit_code": 3}
    from . import nostdin, stack
    nostdin.detach()                                         # every activation route (the autoload finder, library use) runs off a non-tty stdin before any lever installs (nostdin: kalign under --use_template)
    return stack.activate(mode, strict=strict, trigger=trigger)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    from . import stack
    return stack.status()


def __getattr__(name):                                   # PEP 562: keep `import opendde_opt` (the .pth) free of any other import
    if name in ("registry", "modes", "stack", "manifest", "_autoload", "settings", "det", "inputs", "outputs", "warm", "cli", "report", "stock_pred"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
