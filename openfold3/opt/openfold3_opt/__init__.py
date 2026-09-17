"""openfold3_opt — explicit interface to the OpenFold3 inference optimizations.

    import openfold3_opt
    report = openfold3_opt.enable("fast")        # or "exact" / "big" / "off"; idempotent, once per process
    openfold3_opt.status()                       # the last activation report

or, without code changes, `OPENFOLD3_OPT=fast run_openfold predict ...`: the package's .pth installs a lazy import hook that
activates the mode the first time the `openfold3` package is imported (right after its own body has run). Nothing is imported at
interpreter start beyond this module; torch and openfold3 load only when a mode is activated.

The add-ons as shipped are import hooks: each ships a `sitecustomize.py` that, on `PYTHONPATH`, installs a meta-path finder and applies its
levers when the OpenFold3 modules they target are first imported, switched on by environment variables. `enable()` reproduces that route
inside one process: it exports the mode's environment line (modes.py, the one mode table), puts the mode's hook directories on
`sys.path` in the add-ons' documented order and executes the first hook file, which chains the next ones the way the add-ons' own READMEs
say (stack.py). No lever is transcribed or re-implemented here.

Modes (`modes.MODES`; the package default is `fast`): "exact" = stock's own cuEquivariance triangle kernels
(the stock configuration) plus the levers that are bitwise on top of them, eager: fast init and the trunk-kernels add-on's two exact levers
(`modes.EXACT_LINES` = ("cueq",));
byte-identical to stock under the det recipe (both arms with the DS4Sci attention off). "fast" = the kit line plus the trunk-kernels add-on's default set (its two
exact levers and its two Tier-2 kernel swaps). "off" = stock (`pred --mode off` runs the stock CLI in a clean, proven subprocess:
stock_pred.py).
`registry.LEVERS` describes each lever; the add-ons' own code applies them.

Late activation follows one rule: `enable()` is allowed after `import openfold3` and refused by name once any module the add-ons' hooks
target has been imported (`openfold3.core.model.primitives.linear`, `openfold3.core.model.structure.diffusion_module`,
`openfold3.projects.of3_all_atom.model`), once an `OpenFold3` instance exists (counted by a constructor wrap from activation on), or
when a kit hook is already installed in the process through its own `PYTHONPATH` route. Repeated calls return the first report; a
different mode in the same process is refused. `check` (CLI) is the same resolution as a dry run: it gates the mode on this box and
applies nothing.
"""
__version__ = "0.8.20.1"
__all__ = ["enable", "status", "MODES", "ActivationError", "registry", "modes", "stack", "manifest"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (pins, a conflicting switch, late activation, or a hook failed to install)."""


def _entry_gate(mode, strict):
    """None when the installed core carries this package's producers; else the NOT ACTIVE report (or ActivationError with `strict`) —
    `_autoload.core_refusal`, the one probe of every entry route (core_missing / producer_missing), run before `.stack` is imported."""
    from ._autoload import TAG, core_refusal
    from ._core_gate import gate
    from ._core_gate import CoreGateRefused
    import sys
    try:
        gate(__file__, tag=TAG)                          # statement one: the core pin gate (kit_template/_core_gate.py) prints the NOT ACTIVE line itself on absent / mismatched core
        reason = core_refusal()                          # statement two: the module-granular producer probe
        if reason is None:
            return None
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: reason={reason}\n")
    except CoreGateRefused as e:
        reason = e.reason + (e.line.split("reason=" + e.reason, 1)[1].rstrip() if ("reason=" + e.reason) in e.line else "")
    if strict:
        raise ActivationError(reason)
    return {"active": False, "mode": mode, "reason": reason, "package_version": __version__}


def enable(mode: str, *, strict: bool = False, trigger: str = None, n_tokens: int = None, n_gpu=None) -> dict:
    """Activate `mode` ("exact" | "fast" | "big" | "off") in this process; returns the activation report
    {"active", "mode", "line", "levers_requested", "levers_applied", "levers_unavailable", "levers_pending", "partial", "arm_complete",
    "hooks", "gpu": {"name", "cc"}, "openfold3_version", "package_version", "reason"?, ...}. On the hook route the add-ons apply the
    levers when the target modules are imported, after this call returns: the report lists the line's levers as requested and pending;
    `status()` reads the add-ons' own records as of the call (applied / unavailable, `partial`, `arm_complete`), as does the exit tally.
    The mode's line is never a caller's choice: `exact` runs its one composition (modes.DEFAULT_EXACT_LINE); under `big` the GPU count
    selects it (`n_gpu` 1 / unset = the resident line, `n_gpu` 2|4|8 = the tp line); the tp line activates only inside a rank process the CLI spawned
    (`pred --mode big --n_gpu P`, P > 1, tp.py) — anywhere else it is refused with the NOT ACTIVE line.
    `n_tokens` (the query's polymer token count, `pred` passes it) applies the CUDA-graph size gate (modes.graphs_gate: above
    modes.GRAPHS_MAX_TOKENS the graph levers are not requested; the report's `size_gate` says which way).
    Idempotent: repeated calls return the first report. `strict=True` raises ActivationError instead of returning an inactive report."""
    refusal = _entry_gate(mode, strict)                # an absent or older core: refused by name before anything of the core beyond opt_core is imported
    if refusal is not None:
        return refusal
    if mode != "off":
        warm_core_imports()                             # start-up: the core's import warm-up, once, before any lever configures (exact class: imports only, no bytes)
    from . import stack
    return stack.activate(mode, strict=strict, trigger=trigger, n_tokens=n_tokens, n_gpu=n_gpu)


WARM_LIBRARIES = ("torch", "cuequivariance_ops_torch", "cuequivariance_torch")   # torch FIRST: the cuEquivariance ops library links libnvrtc / libcudart, which
                                                                                    # the image resolves through torch's own loading; the ops library imported
                                                                                    # before torch fails on cu13 wheels and cuequivariance_torch then records its
                                                                                    # ops unavailable for the rest of the process


def warm_core_imports(core=None):
    """The core's import warm-up (`opt_core.warm_imports(libraries=WARM_LIBRARIES)`: torch and then the cuEquivariance
    libraries the model imports anyway, imported once at activation so their import-time cost is paid before the first item; free when already
    warm) -- called once by `enable()` in every mode except `off`.  The order is load-bearing (torch before the ops library, WARM_LIBRARIES);
    torch is imported here first in any case.  A core without the attribute: nothing happens (no pin change); a core whose `warm_imports` takes
    no `libraries` is called plain, after torch.  Returns the core's report (None on an older core); never raises on a missing library."""
    if core is None:
        try:
            import opt_core as core
        except ImportError:
            return None
    fn = getattr(core, "warm_imports", None)
    if fn is None:
        return None
    try:
        import torch  # noqa: F401 -- first, before any cuEquivariance library (WARM_LIBRARIES)
    except ImportError:
        pass
    import inspect
    try:
        takes_libraries = "libraries" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        takes_libraries = False
    return fn(libraries=WARM_LIBRARIES) if takes_libraries else fn()


def status() -> dict:
    """The last activation report with the lever fields as the add-ons' records say now, or {"active": False, "reason": ...} when enable()
    has not run."""
    refusal = _entry_gate(None, False)                 # an absent or older core: the NOT ACTIVE report, never a traceback
    if refusal is not None:
        return refusal
    from . import stack
    return stack.status()


UPSTREAM_ISSUES_DIR = "upstream_issues"                    # <tree>/upstream_issues/<ID>_<slug>.py: one file per confirmed upstream issue, fixed on request only (--upstream-fix <ID>)
UPSTREAM_ISSUES_MODULES = "openfold3_opt_upstream_issues"     # sys.modules name of a loaded issue file: f"{UPSTREAM_ISSUES_MODULES}.{ID}" (loaded by path; the directory is never on sys.path)


def upstream_fix_registry(home: str):
    """``--upstream-fix``: this tree's confirmed upstream issues registered with the core's registry (opt_core.upstream_fix — the flag's
    grammar, its refusals by name and its census line live there). One file ``<home>/upstream_issues/<ID>_<slug>.py`` per issue (the ID is
    the file name up to its first ``_``), each a file-backed entry: nothing under upstream_issues/ is loaded until a run names its ID. Every
    process that resolves or applies the flag builds it — the CLI before any launch, a kit arm after activation, the stock caller after its
    proof, every tp rank — so it imports nothing but the core module (standard library only)."""
    import glob
    import os
    from opt_core import upstream_fix
    reg = upstream_fix.Registry(where=f" — one file per issue under {UPSTREAM_ISSUES_DIR}/ (README 'Known upstream issues (not fixed by default)')")
    for path in sorted(glob.glob(os.path.join(home, UPSTREAM_ISSUES_DIR, "*_*.py"))):
        name = os.path.basename(path)
        fid = name.split("_", 1)[0]
        reg.register_file(fid, path, module_name=f"{UPSTREAM_ISSUES_MODULES}.{fid}", file=f"{UPSTREAM_ISSUES_DIR}/{name}", summary=name[len(fid) + 1:-3],
                          extra=lambda mod: {"consumes_templates": bool(getattr(mod, "CONSUMES_TEMPLATES", False))})   # the template guard words its census by it (cli.apply_upstream_fix -> templ_guard.consumed_by)
    return reg


def __getattr__(name):                                   # PEP 562: keep `import openfold3_opt` (the .pth) free of any other import
    if name == "MODES":
        from .modes import MODES
        return MODES
    if name in ("registry", "modes", "stack", "manifest", "report", "env", "hooks", "det", "inputs", "stock_pred", "warm", "_autoload"):
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
