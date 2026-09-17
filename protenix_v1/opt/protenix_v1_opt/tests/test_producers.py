"""The producers gate (_producers.py): the shared core absent, or older than this package imports (a producer module missing), refuses
EVERY entry route by name with exit 3 before anything of the package resolves — the CLI (`python -m protenix_v1_opt` == the console
script == run.sh's verbs) and the `.pth` hook — never a traceback; and the producer list is locked against the package's own imports."""
import ast
import os
import subprocess
import sys
import textwrap

from protenix_v1_opt import _producers as P
from protenix_v1_opt import report as R

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # .../opt (the package's parent)
PKG = os.path.join(OPT, "protenix_v1_opt")


def _env(pythonpath, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_V1_OPT") and k != "PYTHONPATH"}
    env.update(PYTHONPATH=os.pathsep.join(pythonpath), PYTHONDONTWRITEBYTECODE="1", **extra)
    return env


def _shadow_core(tmp_path, with_mem=False):
    """A core OLDER than this package imports: opt_core importable (its top modules stubbed), opt_core.mem.ngpu / rowpair absent."""
    root = tmp_path / "oldcore"; core = root / "opt_core"; core.mkdir(parents=True)
    (core / "__init__.py").write_text('__version__ = "0.3.2"\n')
    for m in ("cli", "gates", "home", "report", "stock_proof"):
        (core / f"{m}.py").write_text("")
    (core / "mem").mkdir(); (core / "mem" / "__init__.py").write_text(""); (core / "mem" / "torch_alloc.py").write_text("")
    return str(root)


def _absent_core(tmp_path):
    """No core at all: a directory FIRST on the path whose opt_core cannot import (shadows any installed core)."""
    root = tmp_path / "nocore"; core = root / "opt_core"; core.mkdir(parents=True)
    (core / "__init__.py").write_text('raise ImportError("opt_core is absent on this interpreter (test shadow)")\n')
    return str(root)


def test_documented_pairs():
    assert P.PREFIX == R.PREFIX and P.EXIT_NOT_ACTIVE == R.EXIT_NOT_ACTIVE == 3


def test_the_producer_list_covers_every_opt_core_import_of_the_package():
    imported = set()
    for fn in os.listdir(PKG):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(PKG, fn)).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "opt_core":
                imported.add(node.module)
                for al in node.names:                                # `from opt_core.mem import ngpu` imports the module opt_core.mem.ngpu when it is one
                    imported.add(f"{node.module}.{al.name}")
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names if a.name.split(".")[0] == "opt_core")
    required = set(P.REQUIRED_PRODUCERS)
    modules = {n for n in imported if n in required or _is_module(n)}
    missing = sorted(n for n in modules if n not in required)
    assert missing == [], f"opt_core modules imported by the package but absent from _producers.REQUIRED_PRODUCERS: {missing}"


def _is_module(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def test_nothing_missing_on_this_interpreter():
    assert P.missing_producers() == []


def test_missing_producers_names_the_modules_an_older_core_lacks(tmp_path, monkeypatch):
    """The producers gate's own words (statement two, after the pin gate): the modules this package imports that the core lacks."""
    shadow = _shadow_core(tmp_path)
    code = ("import sys; sys.path.insert(0, %r); from protenix_v1_opt import _producers as P; m = P.missing_producers(); "
            "print(','.join(m)); print(P.refusal_line(m))" % shadow)
    p = subprocess.run([sys.executable, "-S", "-c", code], env=_env([OPT]), capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-400:]
    mods, line = p.stdout.splitlines()[:2]
    assert "opt_core.mem.ngpu" in mods.split(",") and "opt_core.kernels" in mods.split(",")
    assert line.startswith("[protenix-v1-opt] NOT ACTIVE: reason=producer_missing:") and line.endswith("exit 3")
    absent = _absent_core(tmp_path)
    p = subprocess.run([sys.executable, "-S", "-c", "import sys; sys.path.insert(0, %r); from protenix_v1_opt import _producers as P; print(P.missing_producers())" % absent], env=_env([OPT]), capture_output=True, text=True, timeout=60)
    assert p.stdout.strip() == "['opt_core']", (p.stdout, p.stderr[-300:])
