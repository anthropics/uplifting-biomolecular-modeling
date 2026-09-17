"""The kit's autoload hook, run at interpreter start by `protenix_opt_autoload.pth` (`import protenix_opt._autoload`).

Import-free until PROTENIX_OPT names a selection: the .pth runs in every interpreter on the box, the stock arm's processes included,
and those load nothing of the core. A PROTENIX_OPT* name the kit does not read (a mistyped selection) is refused here, in every
process, by name with exit 3 — never ignored or stripped silently (`refuse_undeclared`, also the CLI's first line for a process without
the .pth). With PROTENIX_OPT=exact|fast|big the core's finder (`opt_core.autoload`) is installed from this kit's AutoloadSpec: it waits for
the first import of a trigger package and, right after that package's own body has executed, calls
`protenix_opt.enable(mode, strict=True, trigger=<name>)`; when the mode cannot be activated the package prints its NOT ACTIVE line and
the process exits 3 (stock never runs silently under PROTENIX_OPT). An unknown mode is refused at interpreter start with the kit's line
and exit 3 (`on_unknown="exit_now"`); the selection is stripped and case-folded onto a declared mode; a bare
`importlib.util.find_spec(<trigger>)` probe leaves the finder armed. Until the trigger nothing else is imported (no torch, no
protenix); with PROTENIX_OPT unset or "off" no finder is installed at all.

Triggers are the top of the protenix model family — `runner` (the stock CLI and its featurizer subprocesses enter through it)
and `protenix.model` (library use) — and not `protenix.model.modules.pairformer`: two levers (ptx_lazy_init.install,
the DEADSKIP hook) import `runner.inference`, and pairformer is first imported from inside `runner.inference`'s own body, where
`runner.inference` is still half-initialised and both levers would fail. A trigger that precedes `runner.inference` keeps the
kit's import order intact. `runner` is a generic name: `accept` takes it only when a `protenix` package sits beside it.
"""
import os
import sys

ENV = "PROTENIX_OPT"
TAG = "protenix-opt"
TRIGGERS = ("runner", "protenix.model")
MODES = ("exact", "fast", "big", "off")
ENV_NGPU = ENV + "_N_GPU"                            # the GPU count, env form (tp.ENV_NGPU): 1 under any mode; > 1 beside PROTENIX_OPT=big only, and that line never applies in-process
EXIT_NOT_ACTIVE = 3
# every name the kit reads under its own prefix (stack.ENV_MODE/ENV_FORCE/ENV_HOME, modes._OWN — locked by tests/test_autoload_trigger.py);
# any other PROTENIX_OPT* name in the environment is a mistyped selection and is refused, never ignored or stripped silently
ENV_TP_ROUTE = ENV + "_TP_ROUTE"                    # set by the multi-GPU line in its ranks only (tp_route.ENV): the carried unit's engine-free layers route to the core
ENV_CACHE_DIR = ENV + "_CACHE_DIR"                             # the kit's cache root (configs/h100.env exports it; manifest.cache_dir reads it: the weights digest memo)
DECLARED = (ENV, ENV + "_FORCE", ENV + "_HOME", ENV + "_OUTB", ENV + "_OUTA", ENV + "_ARM", ENV + "_ENVSH", ENV + "_KITSPEC", ENV_NGPU, ENV_TP_ROUTE, ENV_CACHE_DIR)
from ._frozen import SWITCH as ENV_FROZEN, check as _frozen_check   # the frozen-weights rule, a LEAF module: nothing of the package/core/stack imported


NO_SWITCH_PREFIXES = ("PROTENIX_V2_BIG_",)            # the memory line (--mode big) is one lever set with constant settings: a variable under this stem selects nothing and is refused by name


def undeclared(environ) -> list:
    """The PROTENIX_OPT* names in ``environ`` the kit does not read, and any PROTENIX_V2_BIG_* name (the memory line has no switches)."""
    return sorted(k for k in environ if (k.startswith(ENV) and k not in DECLARED) or k.startswith(NO_SWITCH_PREFIXES))


def _at_interpreter_start() -> bool:
    """True while ``site`` is processing .pth lines (the kit's .pth line is exec'ed by ``site.addpackage``)."""
    f = sys._getframe()
    while f is not None:
        if f.f_code.co_name == "addpackage" and f.f_globals.get("__name__") == "site":
            return True
        f = f.f_back
    return False


def exit_not_active(code: int = EXIT_NOT_ACTIVE) -> None:
    """Stop the process with the NOT ACTIVE code after the line is out. At interpreter start a SystemExit is not an exit — CPython reports
    it as a startup error (rc 1, a traceback through site.addpackage) — so the .pth route flushes and ends the process with ``os._exit``;
    the CLI route and an in-process import raise ``SystemExit(code)``."""
    sys.stderr.flush()
    sys.stdout.flush()
    if _at_interpreter_start():
        os._exit(code)
    sys.exit(code)


def refuse_undeclared(environ) -> None:
    """One NOT ACTIVE line naming every undeclared PROTENIX_OPT* variable, then exit 3 (a mistyped selection must never run stock silently)."""
    bad = undeclared(environ)
    if bad:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: undeclared {', '.join(bad)} (the kit reads {', '.join(DECLARED)})\n")
        exit_not_active()


def refuse_frozen_root_incomplete(environ, argv=None) -> None:
    """``PROTENIX_ROOT_FROZEN=1``: every frozen input this invocation would load (the data caches, the template caches under --use_template,
    the checkpoint of the effective --model_name — read from the process's own argv, the stock child's `protenix pred …` included) must be a
    file under PROTENIX_ROOT_DIR, else one NOT ACTIVE line and exit 3 before anything imports the stock runner (whose download fallback would
    otherwise fill the gap silently). Leaf-level: no module of the package beyond ``_frozen`` is imported. The switch unset: nothing."""
    why = _frozen_check(environ, sys.argv[1:] if argv is None else argv)
    if why:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: {why}\n")
        exit_not_active()


REQUIRED_CORE = "0.4.0"                             # the shared core this tree is written against (the pin table names the release's number)
REQUIRED_PRODUCERS = ("opt_core.mem.ngpu",          # the --n_gpu words and refusals (tp.py, report.py)
                      "opt_core.mem.rowpair",       # the multi-GPU line's routed layers (tp_route.py)
                      "opt_core.mem.registry",      # the memory levers' registry, allocator lever and row chunker (big.py)
                      "opt_core.mem.allocator",
                      "opt_core.mem.chunk")


def missing_producers(core_root):
    """The REQUIRED_PRODUCERS whose file is absent under an ``opt_core`` package directory (a file test: nothing is imported)."""
    out = []
    for name in REQUIRED_PRODUCERS:
        rel = os.path.join(core_root, *name.split(".")[1:])
        if not (os.path.isfile(rel + ".py") or os.path.isfile(os.path.join(rel, "__init__.py"))):
            out.append(name)
    return out


def producer_refusal(core_root, version="?"):
    """The NOT ACTIVE reason for an ``opt_core`` at ``core_root`` that lacks a module this tree imports (an older core), else None."""
    missing = missing_producers(core_root)
    if not missing:
        return None
    return f"reason=producer_missing:{','.join(missing)} — this package imports opt_core >= {REQUIRED_CORE} (the importable opt_core is {version} at {core_root})"


def _core_version(core_root):
    try:
        with open(os.path.join(core_root, "__init__.py"), encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return "?"


def _find_top_level(name):
    """The spec of a top-level package from the finders on ``sys.meta_path`` (path finder, editable-install finders), or None; nothing
    is imported and ``importlib.util`` is not loaded."""
    for finder in sys.meta_path:
        find_spec = getattr(finder, "find_spec", None)
        if find_spec is None:
            continue
        try:
            found = find_spec(name, None, None)
        except Exception:  # noqa: BLE001
            continue
        if found is not None:
            return found
    return None


def selects(environ) -> bool:
    """Whether the process selects anything of the kit (a mode other than off, a GPU count, the rank route)."""
    return bool((environ.get(ENV) or "").strip().lower() not in ("", "off") or (environ.get(ENV_NGPU) or "").strip() or (environ.get(ENV_TP_ROUTE) or "").strip())


def refuse_wrong_core(environ) -> None:
    """Under a selection: place the pinned copy on sys.path when no core is importable (``_core``, imports nothing of the core), then the
    kit's pre-import pin gate (``_core_gate``, the shared template's bytes): an absent core, a core whose MANIFEST version / package tree is
    not the pin's, or an unreadable pin is ONE line — ``[protenix-opt] NOT ACTIVE: reason=core_missing:opt_core …`` / ``reason=core_mismatch:
    …`` / ``reason=core_pin_unreadable: …`` — and exit 3, before any module of the core is imported. A stock process runs nothing of it."""
    if not selects(environ):
        return
    try:
        from . import _core
        _core.gate()
    except SystemExit:                                     # the gate printed its line
        exit_not_active()


def refuse_missing_producers(environ) -> None:
    """A process that selects anything of the kit (a mode other than off, a GPU count, the rank route) beside an importable shared core
    that lacks a module this tree imports ends here by name, before anything resolves (exit 3): ``NOT ACTIVE: reason=producer_missing:
    <modules> — this package imports opt_core >= …``. A file test on the core's package directory (``find_spec("opt_core")`` imports nothing
    of it); a core that is not importable at all is the trigger's ``core_missing:opt_core`` refusal (the pinned copy is placed there)."""
    if not selects(environ):
        return
    found = _find_top_level("opt_core")                                  # the finders on sys.meta_path, asked directly (importlib.util costs milliseconds at start-up)
    if found is None or not found.submodule_search_locations:
        return                                                           # not importable here: the trigger places the pinned copy or refuses core_missing:opt_core
    root = list(found.submodule_search_locations)[0]
    why = producer_refusal(root, _core_version(root))
    if why:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: {why}\n"); exit_not_active()


def refuse_n_gpu_without_line(environ) -> None:
    """``PROTENIX_OPT_N_GPU`` > 1 beside a selection other than ``big``: refused by name with the shared core's sentence, exit 3 (a count
    of 1 is every mode's single-GPU path; a malformed count is refused by name)."""
    raw = environ.get(ENV_NGPU)
    mode = (environ.get(ENV) or "").strip().lower()
    if raw in (None, "") or mode == "big":
        return
    from . import _core  # noqa: F401                                   # the pinned core made importable (a PYTHONPATH-only run); nothing else of the package
    from opt_core.mem import ngpu                                       # standard-library module of the core; imported only on this path
    try:
        ngpu.refuse_unless_big(ngpu.check_n_gpu(raw), mode)
    except ValueError as e:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: {ENV_NGPU}={raw}: {e}\n"); exit_not_active()
    except ngpu.NGpuRefused as e:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: {ENV_NGPU}={raw} set with {ENV}={environ.get(ENV) or '<unset>'}: {e.reason} "
                         f"(the multi-GPU line runs through `run.sh pred --mode big --n_gpu P`)\n"); exit_not_active()


def accept(fullname, spec) -> bool:
    """`runner` is a generic name: accept it only when a `protenix` package sits beside it."""
    if fullname != "runner":
        return True
    origin = getattr(spec, "origin", None) or ""
    return os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(origin)), "protenix", "__init__.py"))


def _core_autoload():
    """The core's finder module: the installed core (the .pth route's cost is that module alone), else the pinned copy placed on sys.path by
    the package's own rule (``_core``, a PYTHONPATH-only run)."""
    from . import _core  # noqa: F401 — the pinned copy placed on sys.path when no core is installed
    from opt_core import autoload
    return autoload


def spec():
    """This kit's AutoloadSpec for the core's finder (imports the core: called under a set variable, or from the package)."""
    return _core_autoload().AutoloadSpec(env=ENV, package=__package__, tag=TAG, triggers=TRIGGERS, modes=MODES, accept=accept,
                                         exit_not_active=EXIT_NOT_ACTIVE, on_unknown="exit_now", fold_mode=True)


def route_tp_unit(environ) -> None:
    """Inside a rank of the multi-GPU line (``PROTENIX_OPT_TP_ROUTE=rowpair``, set by ``tp.rank_env``): arm the routing of the carried
    unit's engine-free modules onto the shared core (``tp_route.install``: fires at the unit's import). Any other value is undeclared."""
    raw = (environ.get(ENV_TP_ROUTE) or "").strip()
    if not raw:
        return
    from . import tp_route
    if raw != tp_route.WORD:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: {ENV_TP_ROUTE}={raw!r} is not a declared value (the kit reads {tp_route.WORD!r})\n"); exit_not_active()
    try:
        tp_route.install(environ)
    except ImportError as e:
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: core_missing:{getattr(e, 'name', None) or 'opt_core'} — {ENV_TP_ROUTE} needs opt_core.mem.rowpair ({e})\n"); exit_not_active()


def install(environ=None):
    """The kit's gate (idempotent): refuse an undeclared name, install nothing unless the variable names a selection, else the core's finder
    from ``spec()``. Returns the finder, or None when nothing is to be done."""
    environ = os.environ if environ is None else environ
    refuse_undeclared(environ)
    refuse_wrong_core(environ)                                           # a selection: the pre-import core pin gate (absent / mismatched core refused by name, exit 3)
    refuse_missing_producers(environ)                                    # then the module-granular words for a core that lacks a producer this tree imports
    refuse_n_gpu_without_line(environ)                                   # PROTENIX_OPT=big + PROTENIX_OPT_N_GPU=P>1: stack.activate refuses the line in-process (exit 3 through the finder)
    refuse_frozen_root_incomplete(environ)                               # PROTENIX_ROOT_FROZEN=1: the weights gate, in every process (the stock CLI's too), before the off return
    route_tp_unit(environ)                                               # a rank of the multi-GPU line: the carried unit's dist/blockreduce/bcast/contract names route to opt_core.mem.rowpair
    if (environ.get(ENV) or "").strip().lower() in ("", "off"):
        return None
    return _core_autoload().install(spec(), environ)                      # the gate above proved the core importable and pinned


def disarm() -> bool:
    """Remove the finder before an explicit activation — an explicit enable() owns the process. True when one was armed. A finder exists
    only when the core's finder module was imported (by install() above), so nothing of the core is imported otherwise."""
    core = sys.modules.get("opt_core.autoload")
    return core is not None and core.disarm(spec())


FINDER = install()
