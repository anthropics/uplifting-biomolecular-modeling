"""rosettafold3_opt — explicit interface to the RoseTTAFold3 (foundry ``rf3``) inference optimizations.

    import rosettafold3_opt
    report = rosettafold3_opt.enable("exact")     # idempotent, once per process, before the first import of rf3.graph_flags
    rosettafold3_opt.status()                     # the last activation report

or, without code changes, ``ROSETTAFOLD3_OPT=exact rf3 fold ...`` on the patched interpreter: the package's .pth installs a lazy
import hook that activates the mode the first time the upstream package ``rf3`` is imported. Nothing is imported at interpreter
start beyond this module.

The add-on installs its bytes into site-packages (five ``rf3`` files, ``opt/forward/rf3_xattempt_addon/install.sh``) and
reads its switches from the environment when ``rf3.graph_flags`` is imported. Activation is therefore a tree state plus an
environment row: ``enable()`` proves by sha256 that this interpreter carries the patched bytes (tree.py), gates the box, and
exports the row (``modes.KIT_MODES``); the kit's own module reads it. The stock arm is a second, pristine interpreter
(``stock/venv``) that this package never touches: ``off`` is ``pred --mode off`` (stock_fold.py), not an uninstall.

Modes (``modes.KIT_MODES``): "exact" = ``RF3_CUDAGRAPH=1 RF3_HOIST=1`` plus the add-on's graph arm over stock kernels and the exact-tier
provider rows applied in-process (``modes.FPF_ARM_TG_SAPB``; bitwise equal to stock under the deterministic recipe), "fast" = the same
switches plus the FPF add-on's fast arm (``modes.FPF_ARM``) and the package's fast levers, "big" = the memory mode (the ``fast`` row by
reference + the memory levers of big.py / levers.py, fast-class numerics; its resource axis ``--n_gpu P`` row-shards the pair stack,
rowpair.py), "off" = stock. ``registry.LEVERS`` describes each lever; the kits' own code applies them.

The contract: ``enable(mode)`` returns the activation report (``active``, ``mode``, ``row``/``switches``, ``tree_state``, ``levers``,
``gpu``, ``upstream``, ``package_version``, ``reason`` when inactive), ``status()`` the last report. Late activation: allowed any
time before ``rf3.graph_flags`` is imported, refused by name (``stack.LateActivationError``) once it is; repeated calls return the
first report; a different mode in the same process is refused. ``check`` (CLI) is the same resolution as a dry run.
"""
__version__ = "0.2.12.0"
__all__ = ["enable", "status", "MODES", "ActivationError", "registry", "modes", "stack", "tree", "manifest", "settings"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (tree state, pins, no GPU, kit missing, contradicting environment, late)."""


def enable(mode: str, *, strict: bool = False, trigger: str = None) -> dict:
    """Activate ``mode`` ("fast" | "exact" | "big" | "off") in this process; returns the activation report. ``strict=True`` raises ActivationError
    instead of returning an inactive report."""
    from . import _core                                 # exposes the pinned checkout on sys.path when no opt_core is installed
    from ._core_gate import gate
    gate(__file__)                                      # THE pin gate, before any opt_core import: core absent / not the pinned one → the NOT ACTIVE line, SystemExit(3)
    _core.require_or_exit()                             # a core lacking a module this package loads (or whose MANIFEST names the pin while its tree does not carry it) → the NOT ACTIVE line, SystemExit(3), before anything resolves
    from . import stack
    return stack.activate(mode, strict=strict, trigger=trigger)


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    from . import stack
    return stack.status()


def __getattr__(name):                                   # PEP 562: keep `import rosettafold3_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in ("registry", "modes", "stack", "tree", "manifest", "settings", "inputs", "report", "stock_fold", "install",
                "warm", "fold", "_autoload"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
