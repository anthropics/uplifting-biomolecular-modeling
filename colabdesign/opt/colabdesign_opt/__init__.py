"""colabdesign_opt — explicit interface to the ColabDesign AF-Multimer design-step optimizations.

    import colabdesign_opt
    report = colabdesign_opt.enable("fast")                  # or "exact" | "off" | "<mode>-no-<lever>"; idempotent, once per process
    from colabdesign import mk_afdesign_model                 # build models AFTER enable(): the levers rebind ColabDesign's classes and methods first

or with no code change: `COLABDESIGN_OPT=fast python your_pipeline.py` (the `.pth` installs `_autoload`, which activates at the first
`import colabdesign`), or the CLI `colabdesign-opt design ...` (the arm runs in a subprocess with a proven-clean environment; `--mode off`
is stock in a subprocess too).

Every route installs the mode's levers through the package's ONE installer (levers.install), in registry order (registry.py: per lever
what it changes, its numerics class and the module that applies it). What each lever did is printed by the lever itself
(`[colabdesign-opt] LEVER ...`) — the CLI's EVIDENCE line reads those lines (evidence.py).

Modes (modes.py, the one table, named by guarantee; lever sets derived from registry.py): `off` — stock; `exact` — the exact-class levers
(compilecache + parcompile + hoist_prev: stock's bytes from the kit's process); `fast` — every lever (exact's + nosub/nosub_fn + the kit's Pallas
kernels trimul/triatt + opm_fold/ln/proj/transition + txla: never bitwise, held per step to the accuracy band against stock's reference design states).
The kit rule: a lever that cannot engage here steps aside BY NAME (`LEVER … state=skipped reason=…`) and the mode runs the rest of its set;
only configuration refuses. Late activation is refused once a `mk_af_model` instance exists or a lever's marker is already on the classes
(stack.py).
"""
__version__ = "0.12.0"


TAG = "colabdesign-opt"                             # the tag of every line this package prints ([colabdesign-opt] ACTIVE / NOT ACTIVE / LEVER / EXIT …); == names.TAG


class ActivationError(RuntimeError):
    """`enable(strict=True)` when the mode cannot be activated: configuration — colabdesign off its pinned commit, the core pin, an unknown mode, a
    late activation. Never a lever: a lever of the mode that cannot engage here steps aside by name (report `skipped`) and the rest engage. A
    card or a stack other than the tested one is never this either: it is named on the report (`card`, `stack_drift`)."""


def enable(mode: str, *, strict: bool = False, trigger: str = None, route: str = "api") -> dict:
    """Activate `mode` ("off" | "exact" | "fast" | a subtractive word `<mode>-no-<lever>`) in this process; returns the activation report
    (stack.activate: {"active", "mode", "route", "tier", "levers", "skipped", "numerics_class", "arm", "line", "gpu", "upstream",
    "package_version", ..., "reason" when not active}).
    Idempotent: repeated calls return the first report; a different mode in the same process is refused."""
    from ._core_gate import gate
    gate(__file__, tag=TAG)                          # the core pin, before any opt_core import: [colabdesign-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: … -> exit 3
    from . import stack
    try:
        return stack.activate(mode, route=route, strict=strict, trigger=trigger)
    except stack.ActivationError as e:
        raise ActivationError(str(e)) from e


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} when enable() has not run."""
    from ._core_gate import gate
    gate(__file__, tag=TAG)
    from . import stack
    return stack.status()


def __getattr__(name):                                   # PEP 562: keep `import colabdesign_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in ("modes", "registry", "stack", "report", "evidence", "settings", "inputs", "outputs", "driver", "units",
                "envproof", "launch", "levers", "nosub", "pallas", "cli", "_autoload"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
