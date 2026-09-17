"""The core pin gate (_core_gate.py, byte-identical to common/opt_core/kit_template/_core_gate.py): statement one of every
documented entry — `python -m protenix_v1_opt <verb>` / the console script, `bash run.sh <verb>` and `source configs/h100.env` (through
`python -m protenix_v1_opt._producers`), the `.pth` hook, in-process `enable()` / `status()`. Against an ABSENT core every route prints
`[protenix-v1-opt] NOT ACTIVE: reason=core_missing:opt_core …` and exits 3; against a STALE core (present, older than the pinned minimum
version) `reason=core_mismatch: opt_core pinned >= v<want> at <path>, installed v<have> at <root>` exit 3; never a traceback. A core at or
above the pinned version always passes regardless of anything else about it — the pin is a version floor, not a byte comparison. Against a core built to MATCH the pin
the gate passes and statement two (the producers gate) names the modules that core lacks (`reason=producer_missing:…`, exit 3). The
build backend is the core's template byte for byte too."""
import os
import subprocess
import sys
import textwrap

from opt_core import gates as G

from .conftest import OPT

KITDIR = os.path.dirname(OPT)
PYPROJECT = os.path.join(OPT, "pyproject.toml")
SIBLING = os.path.normpath(os.path.join(OPT, "..", "..", "common", "opt_core"))
LINE = "[protenix-v1-opt] NOT ACTIVE: reason="


def _env(pythonpath, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_V1_OPT") and k not in ("PYTHONPATH", "MODEL_OPT")}
    env.update(PYTHONPATH=os.pathsep.join(pythonpath), PYTHONDONTWRITEBYTECODE="1", **extra)
    return env


def _python_S(tmp_path):
    """A `python` first on PATH that runs the test interpreter with -S (no site-packages: only PYTHONPATH decides what opt_core is)."""
    b = tmp_path / "bin"; b.mkdir(exist_ok=True)
    p = b / "python"
    p.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n'); p.chmod(0o755)
    return str(b)


def _core(tmp_path, name, version, modules=()):
    """A core on disk: <root>/opt_core/__init__.py (+ empty modules). No manifest file: installed_core() reads the __version__ literal of
    __init__.py only and never looks for one."""
    root = tmp_path / name; pkg = root / "opt_core"; pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n')
    for m in modules:
        d = pkg / os.path.dirname(m.replace(".", "/")) if "." in m else pkg
        d.mkdir(parents=True, exist_ok=True)
        if "." in m:
            (pkg / m.split(".")[0] / "__init__.py").write_text("")
        (pkg / (m.replace(".", "/") + ".py")).write_text("")
    return str(root)


def _pin():
    from protenix_v1_opt import _core_gate as CG
    return CG.read_table(PYPROJECT, "tool.opt_core")


ROUTES = ("module", "run_sh", "source_env", "pth", "enable")


def _run(route, core_path, tmp_path, extra_env=None):
    path = ([core_path] if core_path else []) + [OPT]
    env = _env(path, **(extra_env or {}))
    env["PATH"] = _python_S(tmp_path) + os.pathsep + env.get("PATH", "")
    if route == "module":
        cmd = ["python", "-m", "protenix_v1_opt", "check", "--mode", "fast"]
    elif route == "run_sh":
        cmd = ["bash", os.path.join(KITDIR, "run.sh"), "check", "--mode", "fast"]
    elif route == "source_env":
        cmd = ["bash", "-c", f"source {os.path.join(KITDIR, 'configs', 'h100.env')}; echo SOURCED-rc=$?"]
    elif route == "pth":
        env["PROTENIX_V1_OPT"] = "fast"
        stock = tmp_path / "stock_runner"; (stock / "runner").mkdir(parents=True, exist_ok=True); (stock / "runner" / "__init__.py").write_text("print('STOCK RAN')\n")
        (stock / "protenix").mkdir(exist_ok=True); (stock / "protenix" / "__init__.py").write_text("")            # `runner` counts as the stock's only beside a `protenix` package (_autoload._protenix_family)
        env["PYTHONPATH"] = os.pathsep.join(path + [str(stock)])
        cmd = ["python", "-c", "import protenix_v1_opt._autoload; import runner; print('RETURNED')"]
    elif route == "enable":
        cmd = ["python", "-c", "import protenix_v1_opt; protenix_v1_opt.enable('fast'); print('RETURNED')"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90, cwd=KITDIR)
    rc = int(p.stdout.split("SOURCED-rc=")[1].split()[0]) if route == "source_env" else p.returncode
    return rc, p.stdout, p.stderr


def test_the_gate_is_the_core_template_byte_for_byte():
    assert G.sha256_file(os.path.join(OPT, "protenix_v1_opt", "_core_gate.py")) == G.sha256_file(os.path.join(SIBLING, "kit_template", "_core_gate.py"))
    assert G.sha256_file(os.path.join(OPT, "_build_backend.py")) == G.sha256_file(os.path.join(SIBLING, "kit_template", "_build_backend.py"))


def test_every_route_refuses_an_absent_core_by_name(tmp_path):
    for route in ROUTES:
        rc, out, err = _run(route, None, tmp_path)
        assert rc == 3 and (LINE + "core_missing:opt_core") in err and "Traceback" not in err and "RETURNED" not in out, (route, rc, err[-500:])


def test_every_route_refuses_a_stale_core_by_name(tmp_path):
    """A core OLDER than the pinned minimum version refuses by name (the pin is a version floor: same version, or newer, always passes —
    there are no bytes for the gate to compare, only `version_tuple(installed) < version_tuple(pinned)`)."""
    stale = _core(tmp_path, "stale", "0.3.2", modules=("cli", "gates", "home", "report", "stock_proof"))
    want = _pin()
    for route in ROUTES:
        rc, out, err = _run(route, stale, tmp_path)
        expect = LINE + "core_mismatch: opt_core pinned >= v%s at %s, installed v0.3.2 at %s" % (want["version"], want["path"], stale)
        assert rc == 3 and expect in err and "Traceback" not in err and "RETURNED" not in out, (route, rc, err[-500:])


def test_a_matching_core_passes_the_gate_and_the_producers_gate_names_what_it_lacks(tmp_path):
    want = _pin()
    fake = _core(tmp_path, "match", want["version"], modules=("cli", "gates"))
    rc, out, err = _run("module", fake, tmp_path)
    assert rc == 3 and (LINE + "producer_missing:") in err and "opt_core.mem.ngpu" in err and "core_mismatch" not in err and "Traceback" not in err, err[-500:]


def test_the_gate_precedes_every_core_and_package_import_at_each_entry():
    """In __main__.main, __init__._producers_gate and _autoload.Finder._fire the gate call comes before the producers gate and before any
    import of the package's core-importing modules (nothing of opt_core is imported ahead of it)."""
    import ast
    pkg = os.path.join(OPT, "protenix_v1_opt")
    for fn, func in (("__main__.py", "main"), ("__init__.py", "_producers_gate"), ("_autoload.py", "_fire")):
        text = open(os.path.join(pkg, fn), encoding="utf-8").read()
        node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == func)
        body = node.body[1:] if isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) else node.body   # skip the docstring
        src = "\n".join(ast.get_source_segment(text, stmt) for stmt in body)
        g = src.index("gate(__file__")
        for later in ("refuse_if_missing", "import protenix_v1_opt", "from .cli import", "from . import stack", "opt_core"):
            j = src.replace("._core_gate", "._core_GATE").find(later)
            assert j < 0 or g < j, (fn, func, later)


def test_pin_names_the_core_line_this_package_imports():
    """Un-stubbed, facts from the tree: the pin's version in opt/pyproject.toml [tool.opt_core] is the FLOOR of the opt_core this package
    imports (the importable core here is at or above it: the gate's own rule, _core_gate.gate — a core below the pin is core_mismatch by
    name), equals _producers.MIN_CORE (the line the refusal names), and every module of
    _producers.REQUIRED_PRODUCERS imports from that core (tp.py binds the pinned core's row-shard modules: an older core is core_mismatch /
    producer_missing by name, never a late ImportError). No version literal: a re-pin moves the pin, MIN_CORE and the sibling core together."""
    import importlib
    import opt_core
    from protenix_v1_opt import _core_gate as CG, _producers as P
    pin = CG.read_table(PYPROJECT, "tool.opt_core")
    assert P.MIN_CORE is not None and P.MIN_CORE == P.pinned_core_line() == pin["version"], (pin, P.MIN_CORE)   # one fact, read from the tree by both the gate's reader and the producers module — never a literal
    assert CG.version_tuple(opt_core.__version__) >= CG.version_tuple(pin["version"]), (opt_core.__version__, pin["version"])   # a floor (>=), as the gate judges it: the importable core at or above the kit's pin
    missing = [m for m in P.REQUIRED_PRODUCERS if importlib.util.find_spec(m) is None]
    assert missing == [], missing
    for m in P.REQUIRED_PRODUCERS:
        importlib.import_module(m)
