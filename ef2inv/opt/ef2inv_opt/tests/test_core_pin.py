"""The shared core this kit runs on: opt/pyproject.toml [tool.opt_core] pins the core the process imports (path, minimum version); the
pin is checked against the core actually importable on the path, refusing by name when it's missing or too old; opt/pyproject.toml
declares the custom build backend; run.sh parses and names the core in its install line. No torch, no GPU."""
import os
import shutil
import subprocess
import sys

import pytest

from ._paths import ROOT as MODEL_OPT

OPT = os.path.join(MODEL_OPT, "opt")
CORE = os.path.normpath(os.path.join(OPT, "..", "..", "common", "opt_core"))


def _pin_table():
    """[tool.opt_core] of the kit's opt/pyproject.toml, read at test time (tomllib) — the tests carry no copy of the pin's values."""
    import tomllib
    with open(os.path.join(OPT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)["tool"]["opt_core"]


def test_pin_names_the_imported_core():
    """[tool.opt_core] names a path + a MINIMUM version; the core actually on the path (its own __version__, its own package_dir) is
    what the pin is checked against — never a stored tree digest (core_pin returns {path, version, abs_path} only)."""
    opt_core = pytest.importorskip("opt_core")
    from opt_core.gates import core_pin, core_pin_check
    pin = core_pin(os.path.join(OPT, "pyproject.toml"))
    assert {"path", "version", "abs_path"} <= set(pin)      # core_pin's own contract keys; the kit's [tool.opt_core] table carries more (forward_numerics)
    from opt_core.gates import version_tuple
    assert version_tuple(opt_core.__version__) >= version_tuple(pin["version"])   # the pin is a FLOOR (`>=`): a newer core satisfies it; bump the floor only when the kit adopts a newer core feature
    assert os.path.normpath(pin["abs_path"]) == CORE
    g = core_pin_check(os.path.join(OPT, "pyproject.toml"))
    assert g.ok and g.details["imported"]["version"] and g.details["imported"]["package_dir"]   # the core's live import meets the floor


def test_core_gate_refuses_by_name_without_the_core(tmp_path):
    """`python -m ef2inv_opt check` with no opt_core importable: one NOT ACTIVE line naming core_missing, exit 3 — never a traceback."""
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": OPT, "MODEL_OPT": MODEL_OPT, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    r = subprocess.run([sys.executable, "-S", "-m", "ef2inv_opt", "check", "--mode", "exact"], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 3 and "NOT ACTIVE: reason=core_missing:opt_core" in r.stderr and "Traceback" not in r.stderr, r.stderr[-400:]
def test_pyproject_declares_the_custom_backend_and_ships_no_pth():
    text = open(os.path.join(OPT, "pyproject.toml"), encoding="utf-8").read()
    assert 'build-backend = "_build_backend"' in text and 'backend-path = ["."]' in text
    assert not [f for f in os.listdir(OPT) if f.endswith("_autoload.pth")]      # argv-only kit: no .pth ships (the backend then builds plain wheels)


def test_run_sh_parses_and_names_the_core():
    run_sh = os.path.join(MODEL_OPT, "run.sh")
    if shutil.which("bash"):
        r = subprocess.run(["bash", "-n", run_sh], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        r = subprocess.run(["bash", "-n", os.path.join(MODEL_OPT, "configs", "h100.env")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    text = open(run_sh, encoding="utf-8").read()
    assert 'python -c "import opt_core"' in text and "common/opt_core -e" in text
    r = subprocess.run(["bash", run_sh], capture_output=True, text=True)             # no verb: usage, rc 2 (proves the file executes past parse)
    assert r.returncode == 2 and "run.sh design" in r.stderr


def _arm_import_graph():
    """The package modules an arm process imports: stock_design.py and, transitively, every `from . import …` / `from .x import …` inside
    ef2inv_opt (AST; the standard library, torch and the upstream are outside the tree)."""
    import ast
    pkg = os.path.join(OPT, "ef2inv_opt")
    seen, todo = set(), ["stock_design"]
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        tree = ast.parse(open(os.path.join(pkg, mod + ".py"), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                names = [node.module] if node.module else [a.name for a in node.names]
                for n in names:
                    if os.path.isfile(os.path.join(pkg, n + ".py")):
                        todo.append(n)
    seen.add("__init__")
    return sorted(seen)


def _forward_numerics_files():
    import glob as G
    try:
        import tomllib
    except ModuleNotFoundError:
        pytest.skip("tomllib (python >= 3.11) reads the array")
    globs = tomllib.load(open(os.path.join(OPT, "pyproject.toml"), "rb"))["tool"]["opt_core"]["forward_numerics"]
    files = set()
    for g in globs:
        hits = [p for p in G.glob(os.path.join(MODEL_OPT, g), recursive=True) if os.path.isfile(p)]
        assert hits, f"forward_numerics glob matches nothing: {g}"
        files.update(os.path.relpath(p, MODEL_OPT) for p in hits)
    return globs, files


def test_forward_numerics_cover_the_arm_import_graph():
    """[tool.opt_core] forward_numerics names every package module the arm process imports (an uncovered module = a file whose change
    would move the arithmetic unrecorded), every runtime module of the carried design kit, its cookbook file, the cloud-SDK stand-in, the stock
    file and the pins; and nothing the arm never imports from the package (cli, manifest)."""
    globs, files = _forward_numerics_files()
    for mod in _arm_import_graph():
        assert f"opt/ef2inv_opt/{mod}.py" in files, f"arm-process module not declared in forward_numerics: {mod}.py"
    for mod in ("cli", "manifest", "__main__"):
        assert f"opt/ef2inv_opt/{mod}.py" not in files and mod not in _arm_import_graph()
    dk = "opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1"
    k_files = {os.path.relpath(p, MODEL_OPT) for p in __import__("glob").glob(os.path.join(MODEL_OPT, dk, "k", "*.py"))}
    assert k_files and k_files <= files
    for must in ("opt/ef2inv_opt/fastkit.py", "opt/ef2inv_opt/_absent_sdk_stub.py", "stock/src/cookbook/tutorials/binder_design.py", "stock/PINS.json"):
        assert must in files, must
    assert not [f for f in files if "/results/" in f or "/expected/" in f or f.endswith(".md")]       # evidence and prose are not numerics
