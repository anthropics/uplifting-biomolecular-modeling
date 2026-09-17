"""The core pin gate: its facts (the installed core IS the pin, byte-equal template copies, the generated .pth) and its refusal at every
documented entry (kit_template ``_core_gate.py``; ``gate(__file__)`` is statement one, before any ``opt_core``
import): with the core ABSENT, STALE (an older version first on the path — the pin is a floor: a newer core passes, an older one refuses) or
UNVERSIONED (a core present with no ``__version__`` literal at all), every route ends with ONE ``[rfdiffusion1-opt] NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: …`` line and exit 3 — no
traceback, no second line, nothing of the route's own work reached, never the importability probe's ``not importable`` text — and with
RFDIFFUSION1_OPT unset / ``off`` the ``.pth`` stays inert. The routes, by the names the tree documents them under: ``python -m rfdiffusion1_opt`` ·
the ``rfdiffusion1-opt`` console script (setuptools' wrapper: ``from rfdiffusion1_opt.__main__ import main``) · in-process ``enable()`` /
``status()`` / ``run_design()`` · ``run.sh <verb>`` (a kit mode by flag and by RFDIFFUSION1_OPT in the environment, the stock route ``--mode
off``, ``--config h100``) · ``source configs/h100.env`` (plain and under the variable) · the ``.pth`` under a kit mode (the stock
command line's ``import rfdiffusion``, and any interpreter of the install). The install the cases run on: a stdlib venv whose site-packages
carries the REAL regenerated ``rfdiffusion1_opt_autoload.pth`` and a path ``.pth`` onto a copy of ``opt/`` (package, pyproject.toml,
_build_backend.py) — what ``pip install -e opt`` lays down — with ``run.sh`` and ``configs/`` beside that copy and NO ``common/opt_core``
anywhere; its ``bin/python`` is first on PATH for the bash routes; the stale / pre-manifest cores are shadow trees first on PYTHONPATH. The
gate closure (the modules an entry imports before its gate statement) imports nothing of the core at module level; ``run.sh`` and every
``configs/*.env`` run the two gate probes as their first ``python``; ``__all__`` is the API and the core-bound submodules pass the gate when
reached as attributes."""
import ast
import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys

import pytest

import opt_core
import rfdiffusion1_opt
from rfdiffusion1_opt import _autoload, _core_gate, report, stack

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../rfdiffusion1/opt/rfdiffusion1_opt
OPT_DIR = os.path.dirname(PKG_DIR)                                              # .../rfdiffusion1/opt
KIT_DIR = os.path.dirname(OPT_DIR)                                              # .../rfdiffusion1
PTH = "rfdiffusion1_opt_autoload.pth"
TAG_PREFIX = "[%s] NOT ACTIVE: reason=" % report.TAG
STALE = {"version": "0.2.5"}                                       # older than the pin (0.5.17.4): the floor refuses it, never a sha comparison
GATE_CLOSURE = ("_core_gate.py", "report.py", "__init__.py", "__main__.py", "_autoload.py")   # imported before (or as) the gate statement of an entry
STOCK_CLOSURE = ("stock_cli.py", "det.py", "__init__.py", "_autoload.py", "__main__.py")       # the stock arm's process: no core on its path
PROBE_IMPORT = 'env -u RFDIFFUSION1_OPT python -c "import rfdiffusion1_opt" 2>/dev/null || {'   # the importability probe, the variable unset on it alone
PROBE_GATE = 'python -c "import rfdiffusion1_opt as k; from rfdiffusion1_opt.report import TAG; from rfdiffusion1_opt._core_gate import gate; gate(k.__file__, tag=TAG)" >/dev/null || '
API = {"enable", "status", "run_design", "MODES", "ActivationError", "__version__"}


# ----------------------------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """<tmp>/model-opt-release/rfdiffusion1/{opt/{pyproject.toml, _build_backend.py, rfdiffusion1_opt/, <pth>}, run.sh, configs/} with NO
    common/opt_core; <tmp>/venv = a stdlib venv (no pip, no system site-packages) whose site-packages holds the regenerated <pth> and a path .pth
    onto that opt/ (the editable install's two files); two shadow cores (stale: opt_core/__init__.py declares v0.2.5, older than the pin;
    unversioned: opt_core/__init__.py present with no __version__ literal at all); a stub `rfdiffusion` (the .pth trigger)."""
    tmp = tmp_path_factory.mktemp("core_gate_routes")
    kit = tmp / "model-opt-release" / "rfdiffusion1"
    opt = kit / "opt"
    opt.mkdir(parents=True)
    for name in ("pyproject.toml", "_build_backend.py", PTH):
        shutil.copy(os.path.join(OPT_DIR, name), str(opt / name))
    shutil.copytree(PKG_DIR, str(opt / "rfdiffusion1_opt"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
    shutil.copy(os.path.join(KIT_DIR, "run.sh"), str(kit / "run.sh"))
    shutil.copytree(os.path.join(KIT_DIR, "configs"), str(kit / "configs"))
    assert not (tmp / "model-opt-release" / "common").exists()
    vdir = tmp / "venv"
    base = os.path.realpath(sys.executable)                                       # the real interpreter binary: the venv's `home` names ITS directory (a venv made from inside a venv otherwise records the outer venv's bin/)
    subprocess.run([base, "-m", "venv", "--without-pip", str(vdir)], check=True, capture_output=True, text=True, timeout=300,
                   env={k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "RFDIFFUSION1", "VIRTUAL_ENV"))})
    vpython = str(vdir / "bin" / "python")
    site = subprocess.run([vpython, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True, check=True,
                          env={k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "RFDIFFUSION1"))}).stdout.strip()
    assert site.startswith(str(vdir)) and os.path.isdir(site), site
    with open(os.path.join(site, "__editable__.rfdiffusion1_opt.pth"), "w") as f:      # the path half of an editable install (sorts before the autoload .pth, as pip's does)
        f.write(str(opt) + "\n")
    shutil.copy(str(opt / PTH), os.path.join(site, PTH))
    shadows = {}
    for name, versioned in (("stale", True), ("unversioned", False)):
        root = tmp / ("shadow_" + name)
        (root / "opt_core").mkdir(parents=True)
        if versioned:
            (root / "opt_core" / "__init__.py").write_text('__version__ = "%s"\n' % STALE["version"])
        else:
            (root / "opt_core" / "__init__.py").write_text("# no __version__ literal here\n")
        shadows[name] = str(root)
    stub = tmp / "stub"
    (stub / "rfdiffusion").mkdir(parents=True)
    (stub / "rfdiffusion" / "__init__.py").write_text("VERSION = 'stub'\n")
    pin = _core_gate.read_table(str(opt / "pyproject.toml"), "tool.opt_core")
    assert all(pin.get(k) for k in _core_gate.PIN_KEYS), pin
    w = {"tmp": str(tmp), "kit": str(kit), "opt": str(opt), "python": vpython, "bin": str(vdir / "bin"), "site": site, "shadow": shadows, "stub": str(stub), "pin": pin}
    pre = subprocess.run([vpython, "-c", "import importlib.util as u, rfdiffusion1_opt; print(u.find_spec('opt_core'), rfdiffusion1_opt.__file__)"],
                         cwd=str(tmp), env=_env(w, "absent"), capture_output=True, text=True, timeout=120)
    assert pre.returncode == 0 and pre.stdout.split()[0] == "None" and pre.stdout.split()[1].startswith(str(opt)), (pre.stdout, pre.stderr)   # the premise: no core, the copy is what imports
    return w


def _env(world, fixture, *, mode=None, extra=()):
    """A clean child environment: none of the caller's RFD_* / RFDIFFUSION1* / PYTHON* / MODEL_OPT* / VIRTUAL_ENV; the venv's bin first on PATH
    (`python` there processes the installed .pth files at start); PYTHONPATH = [the shadow core of the fixture] + extra."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RFD_", "RFDIFFUSION1", "PYTHON", "MODEL_OPT", "VIRTUAL_ENV"))}
    path = []
    if fixture in ("stale", "unversioned"):
        path.append(world["shadow"][fixture])
    else:
        assert fixture == "absent", fixture
    path += list(extra)
    env["PYTHONPATH"] = os.pathsep.join(path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PATH"] = world["bin"] + os.pathsep + env.get("PATH", os.defpath)
    if mode is not None:
        env["RFDIFFUSION1_OPT"] = mode
    return env


def _expected_reason(world, fixture):
    pin = world["pin"]
    if fixture == "absent":
        return "core_missing:opt_core (pinned >= v%s at %s; nothing importable as opt_core on sys.path)" % (pin["version"], pin["path"])
    have = STALE["version"] if fixture == "stale" else "?"                            # "?" is installed_core()'s word for no __version__ literal
    return "core_mismatch: opt_core pinned >= v%s at %s, installed v%s at %s" % (pin["version"], pin["path"], have, world["shadow"][fixture])


def _assert_refused(p, world, fixture, marker):
    lines = [ln for ln in p.stderr.splitlines() if "NOT ACTIVE" in ln]
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert len(lines) == 1 and p.stderr.count("NOT ACTIVE") == 1, p.stderr                # exactly one refusal line: the gate's
    assert lines[0] == TAG_PREFIX + _expected_reason(world, fixture), (lines[0], _expected_reason(world, fixture))
    assert "Traceback" not in p.stderr and "Error processing line" not in p.stderr, p.stderr   # no raw exception; site.py never saw one from the .pth line
    assert "not importable" not in p.stderr, p.stderr                             # the importability probe never speaks for the core
    assert marker not in p.stdout, p.stdout                                       # nothing past the gate ran


def _run(world, argv, fixture, *, cwd="tmp", mode=None, extra=()):
    argv = [world["python"] if a == "python" else a for a in argv]
    return subprocess.run(argv, cwd=world[cwd], env=_env(world, fixture, mode=mode, extra=extra), capture_output=True, text=True, timeout=120)


FIXTURES = ("absent", "stale", "unversioned")
PY_ROUTES = {                                                                     # id -> (argv, RFDIFFUSION1_OPT, stdout marker that must NOT appear)
    "python-m":                 (["python", "-m", "rfdiffusion1_opt", "check", "--mode", "exact"], None, "PLAN"),
    "python-m ENV=exact":       (["python", "-m", "rfdiffusion1_opt", "check"], "exact", "PLAN"),
    "console-script":           (["python", "-c", "import sys; from rfdiffusion1_opt.__main__ import main; sys.exit(main())", "check", "--mode", "exact"], None, "PLAN"),
    "console-script ENV=exact": (["python", "-c", "import sys; from rfdiffusion1_opt.__main__ import main; sys.exit(main())", "check"], "exact", "PLAN"),
    "enable":                   (["python", "-c", "import rfdiffusion1_opt as k; k.enable('exact'); print('REACHED')"], None, "REACHED"),
    "status":                   (["python", "-c", "import rfdiffusion1_opt as k; k.status(); print('REACHED')"], None, "REACHED"),
    "run_design":               (["python", "-c", "import rfdiffusion1_opt as k; k.run_design('exact', ['inference.input_pdb=t.pdb', 'contigmap.contigs=[10-10]', 'inference.output_prefix=out/des']); print('REACHED')"], None, "REACHED"),
    "attribute k.stack":        (["python", "-c", "import rfdiffusion1_opt as k; k.stack; print('REACHED')"], None, "REACHED"),
    "pth: any interpreter ENV=exact": (["python", "-c", "print('REACHED')"], "exact", "REACHED"),
}
BASH_ROUTES = {                                                                   # id -> (argv, RFDIFFUSION1_OPT, stdout marker)
    "run.sh check --mode exact":             (["bash", "run.sh", "check", "--mode", "exact"], None, "PLAN"),
    "run.sh design --mode off (stock)":      (["bash", "run.sh", "design", "--mode", "off", "inference.input_pdb=t.pdb", "contigmap.contigs=[10-10]", "inference.output_prefix=out/des"], None, "PLAN"),
    "run.sh check --config h100":            (["bash", "run.sh", "check", "--config", "h100", "--mode", "exact"], None, "PLAN"),
    "source configs/h100.env":               (["bash", "-c", "source configs/h100.env || exit $?; echo REACHED"], None, "REACHED"),
    "ENV=exact run.sh check":                (["bash", "run.sh", "check"], "exact", "PLAN"),
    "ENV=exact run.sh design --config h100": (["bash", "run.sh", "design", "--config", "h100", "inference.input_pdb=t.pdb", "contigmap.contigs=[10-10]", "inference.output_prefix=out/des", "--dry-run"], "exact", "PLAN"),
    "ENV=exact run.sh check --config h100":  (["bash", "run.sh", "check", "--config", "h100"], "exact", "PLAN"),
    "ENV=exact source configs/h100.env":     (["bash", "-c", "source configs/h100.env || exit $?; echo REACHED"], "exact", "REACHED"),
    "ENV=off run.sh design (stock)":         (["bash", "run.sh", "design", "inference.input_pdb=t.pdb", "contigmap.contigs=[10-10]", "inference.output_prefix=out/des"], "off", "PLAN"),
}


# ------------------------------------------------------------------------------------------------------------------ routes
@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("route", sorted(PY_ROUTES))
def test_python_route_refuses_by_name(world, route, fixture):
    argv, mode, marker = PY_ROUTES[route]
    _assert_refused(_run(world, argv, fixture, mode=mode), world, fixture, marker)


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("route", sorted(BASH_ROUTES))
def test_bash_route_refuses_by_name(world, route, fixture):
    argv, mode, marker = BASH_ROUTES[route]
    _assert_refused(_run(world, argv, fixture, cwd="kit", mode=mode), world, fixture, marker)


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("mode", ["exact", "EXACT", "fast", "turbo"])
def test_pth_refuses_by_name_at_the_stock_command_line(world, fixture, mode):
    """The installed .pth under a set variable: the gate refuses at interpreter start (os._exit 3) before the trigger `rfdiffusion` is ever
    imported — for a kit mode (`exact`, `fast`) and a value that is no mode alike (the gate precedes the mode check)."""
    p = _run(world, ["python", "-c", "import rfdiffusion; print('stock would run here')"], fixture, mode=mode, extra=[world["stub"]])
    _assert_refused(p, world, fixture, "stock would run here")


@pytest.mark.parametrize("mode", [None, "", "  ", "off", "OFF"])
def test_pth_is_inert_without_a_kit_mode(world, mode):
    """Unset / empty / blank / off: the .pth imports rfdiffusion1_opt._autoload, which installs nothing, gates nothing and imports nothing of the
    core — the ABSENT install runs stock (rc 0, the marker printed, no NOT ACTIVE line, no .pth error)."""
    p = _run(world, ["python", "-c", "import rfdiffusion; import sys; assert 'rfdiffusion1_opt._autoload' in sys.modules and 'opt_core' not in sys.modules; "
                    "print('stock would run here')"], "absent", mode=mode, extra=[world["stub"]])
    assert p.returncode == 0 and "stock would run here" in p.stdout and "NOT ACTIVE" not in p.stderr and "Error processing line" not in p.stderr, (p.returncode, p.stdout, p.stderr)


def test_importing_the_package_is_inert_on_the_absent_install(world):
    """`import rfdiffusion1_opt`, `from rfdiffusion1_opt import *` (the API) and the core-free submodules gate nothing and import nothing of the core."""
    body = ("import rfdiffusion1_opt, sys; from rfdiffusion1_opt import *; "
            "import rfdiffusion1_opt.modes, rfdiffusion1_opt.registry, rfdiffusion1_opt.det, rfdiffusion1_opt.report, rfdiffusion1_opt.outputs, rfdiffusion1_opt.manifest, rfdiffusion1_opt.cli; "
            "assert 'opt_core' not in sys.modules, sorted(m for m in sys.modules if 'opt' in m); print('inert', rfdiffusion1_opt.__version__, sorted(MODES))")
    p = _run(world, ["python", "-c", body], "absent")
    assert p.returncode == 0 and p.stdout.startswith("inert ") and p.stderr == "", (p.returncode, p.stdout, p.stderr)


def test_hand_started_children_are_core_free_on_the_absent_install(world):
    """The package's own child processes started by module name — `python -s -m rfdiffusion1_opt.stock_cli` (the stock arm) and `python -m
    rfdiffusion1_opt.driver_run` (the driver wrapper) — import nothing of the core, so they are not entries and carry no gate: on the ABSENT
    install both modules import (rc 0) with `opt_core` absent from sys.modules. (The package never passes RFDIFFUSION1_OPT to a child:
    stack.child_environment.)"""
    body = ("import rfdiffusion1_opt.stock_cli, rfdiffusion1_opt.driver_run, rfdiffusion1_opt._autoload, sys; "
            "assert 'opt_core' not in sys.modules, sorted(m for m in sys.modules if 'opt' in m); print('core-free')")
    for mode in (None, "off"):
        p = _run(world, ["python", "-c", body], "absent", mode=mode)
        assert p.returncode == 0 and p.stdout.strip() == "core-free" and "NOT ACTIVE" not in p.stderr, (mode, p.returncode, p.stdout, p.stderr)
    p = _run(world, ["python", "-m", "rfdiffusion1_opt.driver_run", "--help"], "absent")
    assert p.returncode == 0 and "--fixed" in p.stdout and "NOT ACTIVE" not in p.stderr, (p.returncode, p.stdout, p.stderr)


def test_matching_core_passes_the_gate_and_reaches_the_kits_own_refusal(world):
    """The positive control on the same install: with the core (the one this test process imports) first on the path the gate returns
    its facts, and the .pth under RFDIFFUSION1_OPT=exact reaches the kit's OWN sentence at the trigger (one NOT ACTIVE line, exit 3 —
    nothing past the import runs) — the gate changed nothing past itself."""
    facts = _core_gate.gate(PKG_DIR, tag=report.TAG)
    core_root = facts["installed"]["root"]
    body = ("import rfdiffusion1_opt as k; from rfdiffusion1_opt._core_gate import gate; f = gate(k.__file__, tag='t'); "
            "print(f['installed']['version'], f['pinned']['version'])")
    p = _run(world, ["python", "-c", body], "absent", extra=[core_root])
    assert p.returncode == 0 and p.stdout.split() == [facts["pinned"]["version"], facts["pinned"]["version"]], (p.returncode, p.stdout, p.stderr)
    p = _run(world, ["python", "-c", "import rfdiffusion; print('stock would run here')"], "absent", mode="exact", extra=[core_root, world["stub"]])
    assert p.returncode == 3 and p.stdout.strip() == "", (p.returncode, p.stdout, p.stderr)                          # the kit names its fact and the process ends there: stock never runs under the mode's name
    assert "NOT ACTIVE: mode=exact cannot serve upstream's own command line" in p.stderr and "reason=core_" not in p.stderr and p.stderr.count("NOT ACTIVE") == 1, p.stderr
    assert "Traceback" not in p.stderr


# ------------------------------------------------------------------------------------------------------------- placement
def _module_level_imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                names += ["." + a.name for a in node.names]                          # `from . import x, y`
            else:
                names.append(("." * node.level) + (node.module or ""))
    return names


def _reaches_the_core_at_module_level(module, seen=None):
    """Does importing rfdiffusion1_opt.<module> import opt_core at module level (directly or through a package-relative module-level import)?"""
    seen = seen if seen is not None else set()
    seen.add(module)
    for name in _module_level_imports(os.path.join(PKG_DIR, module + ".py")):
        top = name.split(".")[0] if not name.startswith(".") else None
        if top == "opt_core":
            return True
        if name.startswith(".") and name[1:]:
            dep = name[1:].split(".")[0]
            if os.path.isfile(os.path.join(PKG_DIR, dep + ".py")) and dep not in seen and _reaches_the_core_at_module_level(dep, seen):
                return True
    return False


def test_gate_closure_imports_nothing_of_the_core_at_module_level():
    """Everything an entry imports before its gate statement (the gate itself, report.py for TAG, the package marker, __main__ up to the gate,
    _autoload) is standard library + package-relative, core-free at module level — else the gate could not run on an absent core."""
    for name in GATE_CLOSURE:
        mods = _module_level_imports(os.path.join(PKG_DIR, name))
        assert not any(m.split(".")[0] == "opt_core" for m in mods), (name, mods)
    assert set(_module_level_imports(os.path.join(PKG_DIR, "_core_gate.py"))) <= {"importlib.util", "json", "os", "re", "sys"}
    assert [m for m in _module_level_imports(os.path.join(PKG_DIR, "report.py")) if m.startswith(".")] == []      # report.TAG costs no package import either
    assert not _reaches_the_core_at_module_level("__main__") and not _reaches_the_core_at_module_level("cli")       # __main__'s only package import past the gate is cli, itself core-free at module level


def test_all_is_the_api_and_core_bound_submodules_pass_the_gate_as_attributes():
    """`__all__` names the API only (no submodule); `_CORE_BOUND` is exactly the set of `_SUBMODULES` whose import reaches opt_core at module
    level (the package's `__getattr__` runs the gate before importing those)."""
    assert set(rfdiffusion1_opt.__all__) == API
    assert all(os.path.isfile(os.path.join(PKG_DIR, m + ".py")) for m in rfdiffusion1_opt._SUBMODULES)
    assert set(rfdiffusion1_opt._CORE_BOUND) == {m for m in rfdiffusion1_opt._SUBMODULES if _reaches_the_core_at_module_level(m)}
    assert not API & set(rfdiffusion1_opt._SUBMODULES)


def test_gate_is_statement_one_of_main_and_of_the_in_process_entries():
    """__main__.py: the gate call precedes `from .cli import main`; __init__.py: enable / status / run_design open with the gate before their
    first package import; _autoload.install gates before its opt_core import and exits by os._exit."""
    main = ast.parse(open(os.path.join(PKG_DIR, "__main__.py"), encoding="utf-8").read())
    body = [n for n in main.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]   # drop the docstring
    assert [type(n).__name__ for n in body[:3]] == ["ImportFrom", "ImportFrom", "Expr"] and [n.module for n in body[:2]] == ["_core_gate", "report"]
    assert isinstance(body[2].value, ast.Call) and body[2].value.func.id == "gate"
    cli_at = next(i for i, n in enumerate(body) if isinstance(n, ast.ImportFrom) and n.module == "cli")
    assert cli_at > 2
    init = ast.parse(open(os.path.join(PKG_DIR, "__init__.py"), encoding="utf-8").read())
    for fn in (n for n in init.body if isinstance(n, ast.FunctionDef) and n.name in ("enable", "status", "run_design")):
        stmts = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
        assert [getattr(n, "module", None) for n in stmts[:2]] == ["_core_gate", "report"], fn.name
        assert isinstance(stmts[2], ast.Expr) and isinstance(stmts[2].value, ast.Call) and stmts[2].value.func.id == "gate", fn.name
    src = open(os.path.join(PKG_DIR, "_autoload.py"), encoding="utf-8").read()
    assert src.index("gate(__file__, tag=TAG)") < src.index("from opt_core.autoload import install")
    assert re.search(r"except SystemExit as x:\s*#[^\n]*\n(\s*sys\.stderr\.flush\(\)\n)?\s*os\._exit\(", src), "the .pth-time refusal ends with os._exit"


def test_run_sh_and_every_config_gate_first():
    """run.sh and every configs/*.env: their first two `python` invocations are the importability probe (RFDIFFUSION1_OPT unset on it alone, its
    stderr silenced) and the gate probe (the variable and stderr untouched), in that order; a card's file that runs no python of its own sets its
    target word and sources configs/h100.env last (the probes then run first there)."""
    files = [os.path.join(KIT_DIR, "run.sh")] + sorted(os.path.join(KIT_DIR, "configs", f) for f in os.listdir(os.path.join(KIT_DIR, "configs")) if f.endswith(".env"))
    assert len(files) >= 2
    for path in files:
        code = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip() and not ln.lstrip().startswith("#")]
        py = [ln for ln in code if re.search(r"(^|[\s;|&(=])python\d*(\s|$)", ln)]
        if not py and path.endswith(".env"):                                  # a card's file that runs no python of its own: exports, then configs/h100.env (whose probes are then its own) as its LAST statement
            assert code and code[-1].split("#")[0].strip() == '. "$(dirname "${BASH_SOURCE[0]}")/h100.env"' and os.path.basename(path) != "h100.env", (os.path.relpath(path, KIT_DIR), code[-1:])
            assert all(ln.startswith("export MODEL_OPT_TARGET_GPU=") for ln in code[:-1]), (os.path.relpath(path, KIT_DIR), code[:-1])
            continue
        assert len(py) >= 2 and py[0].startswith(PROBE_IMPORT) and py[1].startswith(PROBE_GATE), (os.path.relpath(path, KIT_DIR), py[:2])
        tail = "exit 3; }" if path.endswith("run.sh") else "return 3 2>/dev/null || exit 3; }"
        assert py[0].split("#")[0].rstrip().endswith(tail), (os.path.relpath(path, KIT_DIR), py[0])
        assert "2>" not in py[1].split("||")[0] and "RFDIFFUSION1_OPT" not in py[1].split("python")[0], (os.path.relpath(path, KIT_DIR), py[1])


# ----------------------------------------------------------------------------------------------------------------- the pin's facts, and the stock closure
def _facts():
    return _core_gate.gate(rfdiffusion1_opt.__file__, tag=report.TAG)


def test_pin_matches_the_installed_core():
    """The gate's facts: the installed core's version meets the pin (a floor — installed >= pinned, never required to be equal: a core
    bumped past the pin without a pin update still passes), the installed version IS the one this process imports, and the located
    core IS the one this process imports."""
    from opt_core.gates import version_tuple
    f = _facts()
    assert f["tag"] == report.TAG == "rfdiffusion1-opt"
    assert f["installed"]["version"] == opt_core.__version__, f                            # the gate's installed fact is this process's own core
    assert version_tuple(f["installed"]["version"]) >= version_tuple(f["pinned"]["version"]), f   # the floor: installed >= pinned, not ==
    assert os.path.samefile(f["installed"]["package_dir"], os.path.dirname(os.path.abspath(opt_core.__file__)))
    assert os.path.samefile(f["pinned"]["pyproject"], os.path.join(OPT_DIR, "pyproject.toml"))
    rep_core = dict(f["installed"], ok=True, pinned=f["pinned"])                          # the report's core block is these facts (stack.activate)
    assert rep_core["ok"] is True and version_tuple(rep_core["version"]) >= version_tuple(rep_core["pinned"]["version"])


def test_template_copies_are_the_installed_cores_bytes():
    """opt/_build_backend.py and rfdiffusion1_opt/_core_gate.py are the installed core's kit_template files byte for byte; the gate has exactly one
    copy, inside the package."""
    template = os.path.join(_facts()["installed"]["root"], "kit_template")
    for copy, name in ((os.path.join(OPT_DIR, "_build_backend.py"), "_build_backend.py"), (os.path.join(PKG_DIR, "_core_gate.py"), "_core_gate.py")):
        assert open(copy, "rb").read() == open(os.path.join(template, name), "rb").read(), copy
    assert [os.path.relpath(p, OPT_DIR) for p in glob.glob(os.path.join(OPT_DIR, "*", "_core_gate.py"))] == [os.path.join("rfdiffusion1_opt", "_core_gate.py")]   # the release check's own glob: one copy, in the package


def test_pth_is_the_backends_generated_text():
    """opt/rfdiffusion1_opt_autoload.pth == pth_text(package, ENV, TAG) of the kit's own opt/_build_backend.py: the header naming the three and the
    one guarded `import rfdiffusion1_opt._autoload` line; the only *_autoload.pth beside the backend."""
    spec = importlib.util.spec_from_file_location("_kit_build_backend", os.path.join(OPT_DIR, "_build_backend.py"))
    bb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bb)
    text = open(os.path.join(OPT_DIR, "rfdiffusion1_opt_autoload.pth"), encoding="utf-8").read()
    assert text == bb.pth_text("rfdiffusion1_opt", _autoload.ENV, report.TAG)
    assert bb.pth_fields(text) == ("rfdiffusion1_opt", "RFDIFFUSION1_OPT", "rfdiffusion1-opt", _autoload.EXIT_NOT_ACTIVE) and bb.PTH == "rfdiffusion1_opt_autoload.pth"
    assert "import rfdiffusion1_opt._autoload" in text.splitlines()[1] and len(text.splitlines()) == 2


def test_a_core_other_than_the_pin_is_refused_by_name_in_process(monkeypatch, capsys):
    """enable() / status() with a stale core located (an older version — the pin is a floor), an unversioned core (no __version__ literal
    at all), or none: SystemExit 3 and the one line — the in-process route (direct calls, a monkeypatched installed_core), complementing
    the subprocess routes above."""
    real = _core_gate.installed_core()
    stack.reset_for_tests()
    cases = ((lambda: dict(real, version="0.2.5"), "core_mismatch: opt_core pinned >= v%s at ../../common/opt_core, installed v0.2.5 at " % real["version"]),
             (lambda: dict(real, version=None), "core_mismatch: opt_core pinned >= v%s at ../../common/opt_core, installed v? at " % real["version"]),
             (lambda: None, "core_missing:opt_core (pinned >= v%s at ../../common/opt_core; nothing importable as opt_core on sys.path)" % real["version"]))
    for fake, reason in cases:
        monkeypatch.setattr(_core_gate, "installed_core", fake)
        for entry in (lambda: rfdiffusion1_opt.enable("exact"), rfdiffusion1_opt.status):
            with pytest.raises(SystemExit) as ex:
                entry()
            assert ex.value.code == 3 and isinstance(ex.value, _core_gate.CoreGateRefused) and ex.value.reason == reason.split(":")[0], (reason, ex.value)
            err = capsys.readouterr().err
            assert err.count("NOT ACTIVE") == 1 and err.startswith("[rfdiffusion1-opt] NOT ACTIVE: reason=" + reason) and err.endswith("\n"), err
    monkeypatch.undo()
    assert stack.status().get("active") is not True                                       # nothing was armed by a refused entry


def test_an_unreadable_pin_is_refused_by_name(tmp_path, capsys):
    """A pyproject.toml whose [tool.opt_core] lacks a key (PIN_KEYS: path, version), or no pyproject at all above the anchor:
    core_pin_unreadable, exit 3."""
    (tmp_path / "opt" / "pkg").mkdir(parents=True)
    (tmp_path / "opt" / "pyproject.toml").write_text('[project]\nname = "x_opt"\n[tool.opt_core]\npath = "../../common/opt_core"\n')   # lacks version
    with pytest.raises(SystemExit) as ex:
        _core_gate.gate(str(tmp_path / "opt" / "pkg" / "x.py"), tag=report.TAG)
    err = capsys.readouterr().err
    assert ex.value.code == 3 and "reason=core_pin_unreadable: " in err and "lacks version" in err and ex.value.reason == "core_pin_unreadable"
    (tmp_path / "opt" / "pyproject.toml").unlink()
    with pytest.raises(SystemExit) as ex:
        _core_gate.gate(str(tmp_path / "opt" / "pkg" / "x.py"), tag=report.TAG)
    assert ex.value.code == 3 and "no pyproject.toml with [tool.opt_core]" in capsys.readouterr().err


def test_stock_closure_imports_no_core_at_module_level():
    """The modules the stock arm's process can load import nothing of the core at module level — the stock arm runs with no core on its path."""
    for name in STOCK_CLOSURE:
        mods = _module_level_imports(os.path.join(PKG_DIR, name))
        assert not any(m.split(".")[0] == "opt_core" for m in mods), (name, mods)
    assert not any(m for m in _module_level_imports(os.path.join(PKG_DIR, "det.py")) if m.startswith(".")), "det.py imports nothing of the package either"


def test_stock_closure_runs_without_the_core_on_the_path():
    """`python -S` with only rfdiffusion1/opt on the path: the stock caller and the recipe import (rc 0); the core is provably absent (rc 1)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RFD_", "RFDIFFUSION1", "PYTHON"))}
    env["PYTHONPATH"] = OPT_DIR
    p = subprocess.run([sys.executable, "-S", "-c", "import opt_core"], env=env, capture_output=True, text=True)
    assert p.returncode == 1 and "ModuleNotFoundError" in p.stderr                    # the premise: no core on this path
    p = subprocess.run([sys.executable, "-S", "-c", "import rfdiffusion1_opt.stock_cli, rfdiffusion1_opt.det, rfdiffusion1_opt._autoload; print('ok')"],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip() == "ok", p.stderr


def test_pin_table_reads_through_the_no_tomllib_readers(monkeypatch):
    """opt/pyproject.toml's [tool.opt_core] table must come back whole through the no-tomllib line readers (a python < 3.11 stack, the image of
    record's case) of both the gate (rfdiffusion1_opt/_core_gate.read_table, the reader at every entry) and the core (opt_core.gates,
    the release tools' reader): the header line is exactly `[tool.opt_core]` and the three pin lines carry no comment; both agree with tomllib."""
    import builtins
    from opt_core import gates
    pyproject = os.path.join(OPT_DIR, "pyproject.toml")
    with_tomllib = _core_gate.read_table(pyproject, "tool.opt_core")
    real_import = builtins.__import__
    def no_tomllib(name, *a, **k):
        if name == "tomllib":
            raise ModuleNotFoundError("masked for the test")
        return real_import(name, *a, **k)
    monkeypatch.delitem(sys.modules, "tomllib", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_tomllib)
    gate_fallback = _core_gate.read_table(pyproject, "tool.opt_core")
    core_fallback = gates.read_pin_table(pyproject)
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert set(_core_gate.PIN_KEYS) == set(gates.PIN_KEYS) <= set(gate_fallback) and all(gate_fallback[k] for k in _core_gate.PIN_KEYS), gate_fallback
    assert gate_fallback == with_tomllib == {k: v for k, v in core_fallback.items() if k in gate_fallback}, (gate_fallback, with_tomllib, core_fallback)
    lines = open(pyproject, encoding="utf-8").read().splitlines()
    assert [ln for ln in lines if ln.startswith("[tool.opt_core]")] == ["[tool.opt_core]"]
    at = lines.index("[tool.opt_core]")
    assert [ln.split(" = ")[0] for ln in lines[at + 1:at + 4]] == list(_core_gate.PIN_KEYS) and not any("#" in ln for ln in lines[at + 1:at + 4])
