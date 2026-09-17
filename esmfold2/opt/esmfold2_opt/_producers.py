"""The release-tree modules this package needs BEFORE anything resolves (the core and its producers), read WITHOUT importing them, and
the ONE refusal sentence every entry route prints when one is absent — the CLI (`python -m esmfold2_opt <verb>`), the shell entry points (run.sh, configs/h100.env: `python -m esmfold2_opt._producers`), the `.pth` hook
at interpreter start (`_autoload.py`) and `enable()` from Python (`stack.activate`): the kit's NOT ACTIVE line, exit 3, never a traceback.

* a wholly absent core → ``core_missing:opt_core (...)``
* a core that lacks a producer this package imports (an older core) → ``producer_missing:<module>,<module> — this package imports
  opt_core >= MIN_CORE (...)``

Stdlib only, import-light (os / sys; importlib only when the core is not imported yet; no core module body runs): `_autoload.py` calls it in every interpreter that names a mode.
"""
from __future__ import annotations

import os                                            # import-light on purpose (os, sys): the .pth hook calls this at interpreter start under a named mode
import sys

MIN_CORE = "0.5.18.1"                                # the first core release that carries every producer below (rankdata's data forms and feature broadcast)
REQUIRED_PRODUCERS = (                               # module -> the file that must exist under the core package (checked on disk: nothing is imported)
    ("opt_core.autoload", "autoload.py"), ("opt_core.gates", "gates.py"), ("opt_core.home", "home.py"), ("opt_core.modes", "modes.py"),
    ("opt_core.report", "report.py"), ("opt_core.jit_cache", "jit_cache.py"), ("opt_core.kernels", os.path.join("kernels", "__init__.py")),
    ("opt_core.mem", os.path.join("mem", "__init__.py")), ("opt_core.mem.torch_alloc", os.path.join("mem", "torch_alloc.py")),
    ("opt_core.mem.graph_gate", os.path.join("mem", "graph_gate.py")), ("opt_core.mem.ngpu", os.path.join("mem", "ngpu.py")),
    ("opt_core.mem.patchset", os.path.join("mem", "patchset.py")), ("opt_core.mem.rowpair", os.path.join("mem", "rowpair", "__init__.py")),
    ("opt_core.mem.rowpair.launch", os.path.join("mem", "rowpair", "launch.py")), ("opt_core.mem.rowpair.evidence", os.path.join("mem", "rowpair", "evidence.py")),
    ("opt_core.mem.rowpair.rankdata", os.path.join("mem", "rowpair", "rankdata.py")),
    ("opt_core.arch", "arch.py"),
)
INSTALL_HINT = "pip install -e common/opt_core -e esmfold2/opt"
ENV_HOME = "ESMFOLD2_OPT_HOME"                       # esmfold2/opt when the package does not run from its source tree (stack.opt_home's variable)


def kit_anchor(default):
    """The file the core gate reads the kit's pin beside: ``$ESMFOLD2_OPT_HOME/pyproject.toml`` when that variable names the kit's opt
    directory (a package installed away from its tree), else ``default`` (the caller's ``__file__``: an editable install finds
    ``opt/pyproject.toml`` above the package)."""
    h = os.environ.get(ENV_HOME)
    return os.path.join(h, "pyproject.toml") if h else default


def core_dir():
    """The opt_core package directory — the imported package's when it is already in sys.modules (the .pth hook imports the core's
    autoload first: no finder walk at interpreter start), else its spec's location without importing it — or None when no core PACKAGE
    is importable."""
    mod = sys.modules.get("opt_core")
    if mod is not None:
        path = list(getattr(mod, "__path__", None) or [])
        return path[0] if path else None                 # a plain module named opt_core is not the core
    import importlib.util                            # only when the core is not imported yet (the CLI / enable() routes; never the .pth route)
    try:
        spec = importlib.util.find_spec("opt_core")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def missing_producers():
    """Names of REQUIRED_PRODUCERS absent from the importable core (``["opt_core"]`` when there is no core at all)."""
    d = core_dir()
    if d is None:
        return ["opt_core"]
    return [name for name, rel in REQUIRED_PRODUCERS if not os.path.isfile(os.path.join(d, rel))]


def refusal():
    """The NOT ACTIVE reason when the core or one of its producers is absent, else None."""
    miss = missing_producers()
    if not miss:
        return None
    if miss == ["opt_core"]:
        return f"core_missing:opt_core (opt_core not importable; the kit runs on the release tree's core: {INSTALL_HINT})"
    return (f"producer_missing:{','.join(miss)} — this package imports opt_core >= {MIN_CORE} (the release tree's common/opt_core at the pin in "
            f"opt/pyproject.toml [tool.opt_core]; found {core_dir()}): {INSTALL_HINT}")


def main() -> int:
    """``python -m esmfold2_opt._producers``: exit 0 when the core and every producer are there, else the kit's NOT ACTIVE line on stderr and
    exit 3 — the shell entry points' gate (run.sh, configs/h100.env) BEFORE they resolve anything."""
    why = refusal()
    if why:
        sys.stderr.write(f"[esmfold2-opt] NOT ACTIVE: {why}\n")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
