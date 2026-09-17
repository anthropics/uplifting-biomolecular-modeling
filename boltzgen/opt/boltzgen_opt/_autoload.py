"""Lazy autoload, installed at interpreter start by `boltzgen_opt_autoload.pth` (`import boltzgen_opt._autoload`).

With BOLTZGEN_OPT=<mode> in the environment, a meta-path finder waits for the first import of the trigger package `boltzgen` and,
right after that package's own body has executed, runs the core pin gate (`boltzgen_opt.core_gate`: the shared core the package pins
against the installed one, nothing of the core imported before it — an absent or mismatched core is its one NOT ACTIVE line and
exit 3) and then calls `boltzgen_opt.enable(mode, strict=True)`; when the mode cannot be activated (a lever directory missing, pins,
no GPU, a model instance or a lever module already present) the core prints its NOT ACTIVE line and the process exits 3 — stock never
runs silently under BOLTZGEN_OPT. Until then nothing else is imported (no torch, no upstream, nothing of the core); with BOLTZGEN_OPT
unset, empty or "off" no finder is installed at all. A value that is not a mode (`faster`: no such mode exists) is refused the same
way — at the first `import boltzgen`, exit 3 — never at interpreter start (the interpreter stays unchanged until the model package is
imported; the .pth line itself is the build backend's generated guard, `opt/_build_backend.py`, which turns a refusal raised while
`site` processes it into exit 3 instead of a swallowed error).

Upstream's own `boltzgen run` starts one interpreter per pipeline step with `BOLTZGEN_PIPELINE_STEP` set and hands it the environment
of the parent, which an activated parent has completed with the mode's line: the lever directories' `src` on PYTHONPATH, the mode's
switches and BOLTZGEN_OPT itself (stack.activate). Such a step process starts on `fast_inference`'s route before any code of the step
runs — `site` imports `fast_inference/src/sitecustomize.py` (`import bg_hook`: the seed hook, the graph sampler under BG_GRAPH), and
`bg_hook` itself imports the trigger. The finder therefore lets the trigger pass while `sitecustomize` executes and, once it has
finished, calls the same `enable()`: the core recognises the hook already imported under the mode's own line and completes it the way
the runner `xa_run.py` does after its own sitecustomize (the imports of the `xa_*` lever modules). A refusal found there is printed
and ends the process at once (`os._exit(3)`: `site` swallows exceptions raised inside sitecustomize and an exit raised there is
fatal). The two CPU steps (`analysis`, `filtering`) are left as the runner leaves them (stock subprocesses, with `sitecustomize.py` on
the path as in the runner's own environment): a NOTE line naming the step, nothing applied; the pool workers of `analysis`
(multiprocessing 'spawn') and its resource tracker print no line.
The finder fires once and removes itself.

A PARTIAL activation (a lever of the mode could not run in the process: `partial=<levers>` on the ACTIVE line) is a refusal by name: a
mode is all of its levers, so the NOT ACTIVE line names the levers and the process exits 3 before upstream's step runs.
"""
import os
import sys

from . import TAG                          # the package's one tag; the package module is already loaded when this one is imported and imports nothing itself

ENV = "BOLTZGEN_OPT"
TRIGGER = "boltzgen"
SITECUSTOMIZE = "sitecustomize"
MODES = ("off", "exact", "fast", "big")   # the import-free copy of modes.MODES (held equal by a test; anything else is refused at the trigger)
CPU_STEPS = ("analysis", "filtering")
STEP_ENV = "BOLTZGEN_PIPELINE_STEP"
EXIT_NOT_ACTIVE = 3                        # the import-free copy of codes.EXIT_NOT_ACTIVE (held equal by a test)
CORE_PACKAGE = "opt_core"                  # the shared core the package imports at activation (opt/pyproject.toml [tool.opt_core]); this module imports none of it
MP_CHILD_FLAG = "--multiprocessing-fork"    # on the command line of every multiprocessing 'spawn' child (multiprocessing.spawn.get_command_line)


def _real_spec(finder, fullname, path, target):
    for other in sys.meta_path:
        if other is finder:
            continue
        try:
            spec = other.find_spec(fullname, path, target)
        except Exception:
            spec = None
        if spec is not None:
            return spec
    return None


class Finder:
    """Duck-typed meta-path finder (no importlib.abc import at interpreter start)."""

    def __init__(self, mode):
        self.mode = mode
        self.armed = True
        self.fired = None                   # the module whose import fired the activation: the trigger, or sitecustomize
        self.in_sitecustomize = False       # fast_inference's sitecustomize (the `import bg_hook` line) is executing: the trigger passes, the activation completes afterwards

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed:
            return None
        if fullname == SITECUSTOMIZE and not self.in_sitecustomize:
            spec = _real_spec(self, fullname, path, target)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _SiteLoader(spec.loader, self)
            return spec
        if fullname != TRIGGER or self.in_sitecustomize:
            return None
        spec = _real_spec(self, fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        self.armed = False
        spec.loader = _Loader(spec.loader, self)
        return spec

    def retire(self):
        self.armed = False
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


class _Loader:
    """Wraps the real loader of the trigger: exec the package body, then activate."""

    def __init__(self, inner, finder):
        self.inner = inner
        self.finder = finder

    def create_module(self, spec):
        return self.inner.create_module(spec) if hasattr(self.inner, "create_module") else None

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.finder.fired = module.__name__
        self.finder.retire()
        _activate(self.finder.mode)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _SiteLoader:
    """Wraps the real loader of ``sitecustomize``: while it executes the trigger passes (``bg_hook``, imported there, imports it);
    afterwards, when the trigger was imported in there, the same activation completes the line."""

    def __init__(self, inner, finder):
        self.inner = inner
        self.finder = finder

    def create_module(self, spec):
        return self.inner.create_module(spec) if hasattr(self.inner, "create_module") else None

    def exec_module(self, module):
        self.finder.in_sitecustomize = True
        try:
            self.inner.exec_module(module)
        finally:
            self.finder.in_sitecustomize = False
        if self.finder.armed and TRIGGER in sys.modules:
            self.finder.fired = SITECUSTOMIZE
            self.finder.retire()
            _activate(self.finder.mode, in_site=True)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _multiprocessing_child() -> bool:
    """A process that multiprocessing itself started — a 'spawn' pool worker or the resource tracker: its command line is
    multiprocessing's own (spawn.get_command_line: ``-c 'from multiprocessing.spawn import …' --multiprocessing-fork``;
    resource_tracker.ensure_running: ``-c 'from multiprocessing.resource_tracker import …'``)."""
    argv = tuple(getattr(sys, "orig_argv", ()))
    return MP_CHILD_FLAG in argv or any(a.startswith("from multiprocessing.") for a in argv)


def _say(msg):
    sys.stderr.write(f"[{TAG}] NOT ACTIVE: {msg}\n")
    sys.stderr.flush()


def _note(msg):
    """An informational line (the package's ``NOTE`` grammar, stack.gpu_gate's; import-free): nothing was asked of this process, nothing is refused."""
    sys.stderr.write(f"[{TAG}] NOTE {msg}\n")
    sys.stderr.flush()


def _refuse(msg, in_site):
    _say(msg)
    if in_site:                             # inside `site`: an exception is swallowed (site.execsitecustomize) or fatal; the line is printed, the process ends
        os._exit(EXIT_NOT_ACTIVE)
    sys.exit(EXIT_NOT_ACTIVE)


def _run_core_gate(in_site):
    """The core pin gate, first at the trigger under a set BOLTZGEN_OPT (``boltzgen_opt.core_gate``; the package itself imports nothing of
    the core). A refusal has printed its line: inside `site` the process ends at once (an exit raised there is fatal or swallowed),
    elsewhere the gate's ``SystemExit(3)`` propagates like every other refusal of this module."""
    import boltzgen_opt
    try:
        boltzgen_opt.core_gate()
    except SystemExit as x:
        if in_site:
            sys.stderr.flush()
            os._exit(x.code if isinstance(x.code, int) else EXIT_NOT_ACTIVE)
        raise


def _activate(mode, in_site=False):
    _run_core_gate(in_site)
    if mode not in MODES:
        _refuse(f"{ENV}={mode!r} is not a mode ({'|'.join(MODES)})", in_site)
    step = os.environ.get(STEP_ENV)
    if step in CPU_STEPS:
        if not _multiprocessing_child():                                     # the step's own process says it, not its pool workers / resource tracker
            _note(f"CPU step {step!r}: upstream's own code, no lever of the mode acts here (the kits run analysis/filtering as stock subprocesses too, xa_run.py l.62)")
        return
    import boltzgen_opt
    try:
        boltzgen_opt.enable(mode, strict=True)
    except ImportError as e:                 # first: the package's own imports — the pinned core absent is a named refusal, never a traceback into stock or a silent stock run (evaluating `boltzgen_opt.ActivationError` below would itself import the core)
        if (getattr(e, "name", None) or "").split(".")[0] != CORE_PACKAGE and not in_site:
            raise
        _refuse(f"{e.name or CORE_PACKAGE} is not importable ({e}); this kit stands on the core it pins: "
                f"bash boltzgen/run.sh install (= pip install -e common/opt_core -e boltzgen/opt)", in_site)
    except (boltzgen_opt.ActivationError, ValueError) as e:
        _refuse(str(e), in_site)
    except Exception as e:  # noqa: BLE001 — inside `site` an error would be swallowed and the step would run half-activated
        if not in_site:
            raise
        _refuse(f"activation failed after the partner's sitecustomize: {type(e).__name__}: {e}", in_site)   # str(e): an OSError names its path there, repr() does not


ENV_KERNELS = "BOLTZGEN_OPT_KERNELS"       # census.ENV_KERNELS (held equal by tests/test_kernels_census.py): arms the accelerator census in a kit child


def arm_census(in_site=False):
    """With ``BOLTZGEN_OPT_KERNELS=<payload>`` (design.kit_env exports it into every kit child; stack.activate into an activated process's
    children) arm the accelerator census in this process when it is a model process: not one of upstream's CPU steps (``STEP_ENV`` in
    ``CPU_STEPS``: they never touch the library and print no census line) and not a multiprocessing child (DataLoader / pool workers
    re-executing site). Arming is EARLY (census.arm early=True): nothing is imported during start-up; at the first import of upstream's
    own ``boltzgen`` package — before any step can run — the census imports torch and the library and wraps it, and an accelerator the
    mode routes through that this process cannot provide means the mode cannot activate here: the NOT ACTIVE line naming the escape, the
    census line, exit ``EXIT_NOT_ACTIVE`` (``os._exit``: the refusal fires inside upstream's import, where an exception would surface as a
    traceback of the step instead)."""
    text = os.environ.get(ENV_KERNELS)
    if not text or _multiprocessing_child() or os.environ.get(STEP_ENV) in CPU_STEPS:
        return None
    from . import census, print_fresh

    def refuse_early():                        # a KIT mode's process only (BOLTZGEN_OPT_KERNELS rides the kit children's environment; a stock process, stock_design, carries no census and imports no policy)
        from . import kernels
        refusal = kernels.refuse_if_absent()
        if refusal:
            print_fresh(refusal)
            census.print_line()                  # the census line itself (absent words), then the exit: nothing of the step ran
            os._exit(EXIT_NOT_ACTIVE)

    try:
        return census.arm(text, early=True, on_early=refuse_early)
    except ValueError as e:                    # a malformed payload is named on stderr, never silently ignored; the caller's account (kernels.verdict) then names the missing line
        print_fresh(f"[boltzgen-opt] {ENV_KERNELS}: {e}")
        return None


def install():
    arm_census(in_site=True)
    mode = os.environ.get(ENV)
    if not mode or mode == "off":
        return None
    if any(isinstance(f, Finder) for f in sys.meta_path):
        return None
    f = Finder(mode)
    sys.meta_path.insert(0, f)
    return f


install()
