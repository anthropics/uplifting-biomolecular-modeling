"""af3_jax_opt — explicit interface to the inference optimizations of the AlphaFold 3 open code running the ported OF3-p2 checkpoint. Activation
composes a command and launches the fork's own `run_alphafold.py`; the package itself never imports `alphafold3` or `jax`, and nothing of
the model runs at interpreter start. There is no import-time activation: `AF3_JAX_OPT` in the environment is honoured by the wrapper
command only — importing `alphafold3` directly under a kit mode is refused rather than silently running stock. `enable(mode, variant=...)`
may run any time until a model process has launched, after which a different (mode, variant) raises `ActivationError`.

Nothing is imported at interpreter start beyond this module and `_autoload` (the .pth hook): the names below resolve on first use (PEP 562
`__getattr__`), so an interpreter with the mode unset or `off` holds no other module of the kit or of the shared core."""
__version__ = "0.3.39"
__all__ = ["enable", "activate", "check", "status", "ActivationError", "MODES", "DEFAULT_MODE", "VARIANTS", "CORE_MISSING", "__version__"]

_NAMES = {"enable": ("stack", "activate"), "activate": ("stack", "activate"), "check": ("stack", "check"), "status": ("stack", "status"),
          "ActivationError": ("stack", "ActivationError"), "MODES": ("modes", "MODES"), "DEFAULT_MODE": ("modes", "DEFAULT_MODE"), "VARIANTS": ("variants", "VARIANTS")}
_RESOLVED = {}


def _core_missing_interface(missing):
    """The package surface when the shared core is not importable: every entry answers NOT ACTIVE reason=core_missing:<module> (the command line
    prints it and exits 3; enable()/check() raise) — never a traceback at import, never a stock run under a kit variable."""
    class ActivationError(RuntimeError):
        """Raised by enable()/check() when the shared core is missing (the command line prints NOT ACTIVE and exits 3 instead)."""

    def enable(*args, **kwargs):
        from ._autoload import TAG as _TAG
        from ._core_gate import gate as _gate
        _gate(__file__, tag=_TAG)                                         # THE pin gate: the core absent or not the pinned one → its NOT ACTIVE line, SystemExit 3 (the same first statement as every entry)
        raise ActivationError(f"NOT ACTIVE reason=core_missing:{missing} (the pinned core is importable but lacks this module: pip install -e common/opt_core -e af3_jax/opt)")

    def status():
        return None
    return {"enable": enable, "activate": enable, "check": enable, "status": status, "ActivationError": ActivationError,
            "MODES": (), "DEFAULT_MODE": None, "VARIANTS": (), "CORE_MISSING": missing}


def _resolve():
    if _RESOLVED:
        return _RESOLVED
    import importlib
    try:
        vals = {name: getattr(importlib.import_module(f"{__name__}.{mod}"), attr) for name, (mod, attr) in _NAMES.items()}
        vals["CORE_MISSING"] = None                                       # the shared core imported
    except ImportError as _e:                                             # the shared core is not importable: the package still imports (the start-up hook and the command
        from ._autoload import core_missing as _core_missing              # line depend on it) and every entry point answers NOT ACTIVE reason=core_missing:<module>, exit 3
        missing = _core_missing(_e)
        if not missing:
            raise
        vals = _core_missing_interface(missing)
    _RESOLVED.update(vals)
    return _RESOLVED


def __getattr__(name: str):
    """PEP 562: the interface resolves on first use (stack / modes / variants and the shared core are not imported at interpreter start)."""
    if name in _NAMES or name == "CORE_MISSING":
        return _resolve()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
