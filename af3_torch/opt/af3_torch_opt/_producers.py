"""The shared-core producers this package imports, checked BEFORE anything of the package resolves (standard library only). Every entry
route — ``python -m af3_torch_opt <verb>`` / the ``af3-torch-opt`` console script (``__main__.main``), ``run.sh <verb>`` and
``configs/<gpu>.env`` (both go through ``python -m af3_torch_opt``), the ``.pth`` hook (``_autoload``, when ``AF3_TORCH_OPT`` arms it) —
calls :func:`refuse_if_missing` first. With the shared core absent the process prints
``[af3-torch-opt] NOT ACTIVE mode=<m> reason=core_missing:opt_core — this package imports opt_core >= MIN_CORE (...)`` and exits 3; with an
OLDER core that lacks a producer it prints ``... reason=producer_missing:<module,...> — ...`` and exits 3 — never a traceback, never a run
that resolves a mode without its producers. The model-process scripts (forward.py, postprocess.py, rowpair_xfold.py) import the same core
from the directory the wrapper passes them (``--opt-core``), so this list covers them too."""
import importlib
import importlib.util
import os
import sys

PREFIX = "[af3-torch-opt]"       # == report.PREFIX (tests/test_producers: the documented pair; this module imports nothing of the package)
EXIT_NOT_ACTIVE = 3               # == opt_core.report.EXIT_NOT_ACTIVE


def _pinned_core_version() -> str:
    """The core version this package pins (opt/pyproject.toml [tool.opt_core] version — the ONE place it is written; read as text)."""
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml"), encoding="utf-8") as f:
            text = f.read()
        block = text.split("[tool.opt_core]", 1)[1]
        for line in block.splitlines():
            if line.strip().startswith("version") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except (OSError, IndexError):
        pass
    return "?"


MIN_CORE = _pinned_core_version()  # the opt_core line whose producers are listed below (opt/pyproject.toml [tool.opt_core] version)
CORE = "opt_core"
ENV_MODE = "AF3_TORCH_OPT"
REQUIRED_PRODUCERS = (            # every opt_core module a module of this package imports (tests/test_producers locks the list against the sources)
    "opt_core", "opt_core.arch", "opt_core.cli", "opt_core.gates", "opt_core.home", "opt_core.kernels",
    "opt_core.modes", "opt_core.oom", "opt_core.process", "opt_core.report", "opt_core.shape_policy", "opt_core.stock_proof",
    "opt_core.mem", "opt_core.mem.ngpu", "opt_core.mem.torch_alloc",
    "opt_core.mem.rowpair", "opt_core.mem.rowpair.bcast", "opt_core.mem.rowpair.census", "opt_core.mem.rowpair.confidence",
    "opt_core.mem.rowpair.diffusion", "opt_core.mem.rowpair.dist", "opt_core.mem.rowpair.evidence", "opt_core.mem.rowpair.heads",
    "opt_core.mem.rowpair.launch", "opt_core.mem.rowpair.msa", "opt_core.mem.rowpair.pairstack", "opt_core.mem.rowpair.shard",
    "opt_core.mem.rowpair.template", "opt_core.mem.rowpair.transition", "opt_core.mem.rowpair.triatt", "opt_core.mem.rowpair.trimul", "opt_core.mem.rowpair.trimul_fused",
    "opt_core.mem.rowpair.trunk",
)


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _core_present() -> bool:
    """The core package itself imports (its __init__ is import-light by contract); a spec that exists but cannot execute counts as absent."""
    try:
        importlib.import_module(CORE)
        return True
    except Exception:  # noqa: BLE001
        return False


def missing_producers(names=REQUIRED_PRODUCERS) -> list:
    """The names among ``names`` this interpreter cannot resolve (an absent core lists as the one entry 'opt_core')."""
    if not _importable(CORE) or not _core_present():
        return [CORE]
    return [n for n in names if n != CORE and not _importable(n)]


def mode_word(argv=None, environ=None) -> str:
    """The mode the refusal line names: ``--mode`` on the command line when given, else ``$AF3_TORCH_OPT``, else ``none``."""
    argv = sys.argv if argv is None else list(argv)
    environ = os.environ if environ is None else environ
    mode = (environ.get(ENV_MODE) or "").strip() or None
    for i, a in enumerate(argv):
        if a == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1]
        elif isinstance(a, str) and a.startswith("--mode="):
            mode = a.split("=", 1)[1]
    return mode if mode is not None else "none"


def refusal_line(missing, mode: str = "none") -> str:
    kind = "core_missing" if list(missing) == [CORE] else "producer_missing"
    return (f"{PREFIX} NOT ACTIVE mode={mode} reason={kind}:{','.join(missing)} — this package imports {CORE} >= {MIN_CORE} "
            f"(opt/pyproject.toml [tool.{CORE}]; install the release tree's common/{CORE}: pip install -e common/{CORE} -e opt); exit {EXIT_NOT_ACTIVE}")


def refuse_if_missing(stream=None, exit=sys.exit, argv=None, environ=None):
    """Print the refusal line and end the process with EXIT_NOT_ACTIVE when a producer is missing; return None otherwise."""
    missing = missing_producers()
    if missing:
        print(refusal_line(missing, mode_word(argv, environ)), file=stream or sys.stderr, flush=True)
        try:
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        exit(EXIT_NOT_ACTIVE)
    return None
