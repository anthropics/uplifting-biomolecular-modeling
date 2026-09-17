"""The core pin gate at every documented entry of the kit, proven in child interpreters that cannot see the test interpreter's own core.

Fixtures: a copy of the kit's `opt/` (package without its tests, `pyproject.toml`, `_build_backend.py`, the generated `.pth`) plus `run.sh` and
`configs/` in a temporary tree that has NO `common/opt_core` (ABSENT); the same with a shadow core first on `PYTHONPATH` whose `opt_core/__init__.py`
carries an older `__version__` than the pin (STALE — the floor: a newer core passes, an older one refuses); a shadow core whose `__init__.py`
carries no `__version__` literal at all (UNVERSIONED — installed_core() returns version=None, displayed as `v?`). Children run
`python -S` (no site-packages: an installed core is out of reach; `PYTHONPATH` is honoured), the bash routes through a `python` wrapper first
on `PATH` that adds `-S`, the `.pth` route through the REAL generated `.pth` in a site directory (`site.addsitedir` in a `-S` child, and a bare
`venv` interpreter's own start-up).

Every refusing case: exit status 3, exactly one `NOT ACTIVE` line on stderr — `[pxdesign-opt] NOT ACTIVE: reason=core_missing:opt_core …`
(ABSENT) or `reason=core_mismatch: …` naming the pinned and installed versions and the installed root (STALE / UNVERSIONED) — no traceback, no
`Error processing line`, and the marker the route would print past the gate absent. The inert cases (`PXDESIGN_OPT` unset or `off`: bare
package import, the `.pth` line) exit 0 with nothing of the core imported. Test ids name the route.
"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                  # pxdesign/
PKG, ENV, TAG = "pxdesign_opt", "PXDESIGN_OPT", "pxdesign-opt"
PTH = PKG + "_autoload.pth"
STALE_VERSION = "0.2.5"                                                       # older than the pin (floor semantics): the pin's version currently exceeds this
MARKER = "REACHED-PAST-THE-GATE"


def _gate_module():
    """The kit's own copy of the gate, loaded by path (its table reader gives the pin the lines must name)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_pxdesign_core_gate_under_test", os.path.join(TREE, "opt", PKG, "_core_gate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("core_gate_routes")
    eng = tmp / "pxdesign"
    opt = eng / "opt"
    shutil.copytree(os.path.join(TREE, "opt", PKG), str(opt / PKG), ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "_build_backend.py", PTH):
        shutil.copy(os.path.join(TREE, "opt", name), str(opt / name))
    shutil.copy(os.path.join(TREE, "run.sh"), str(eng / "run.sh"))
    shutil.copytree(os.path.join(TREE, "configs"), str(eng / "configs"))
    assert not (tmp / "common").exists() and not (eng.parent / "common" / "opt_core").exists()
    shadow = tmp / "shadow"                                                  # STALE: a core older than the pin (floor semantics)
    (shadow / "opt_core").mkdir(parents=True)
    (shadow / "opt_core" / "__init__.py").write_text('__version__ = "%s"\n' % STALE_VERSION)
    unversioned = tmp / "unversioned"                                        # UNVERSIONED: a core whose __init__.py has no __version__ literal
    (unversioned / "opt_core").mkdir(parents=True)
    (unversioned / "opt_core" / "__init__.py").write_text("# no __version__ literal\n")
    bindir = tmp / "bin"                                                     # `python` first on PATH for the bash routes: this interpreter, -S
    bindir.mkdir()
    wrapper = bindir / "python"
    wrapper.write_text('#!/bin/sh\nexec "%s" -S "$@"\n' % sys.executable)
    wrapper.chmod(0o755)
    stub = tmp / "stub"                                                       # a stand-in for the upstream trigger package `pxdesign` (the .pth route's inert case imports it)
    (stub / "pxdesign").mkdir(parents=True)
    (stub / "pxdesign" / "__init__.py").write_text("STOCK = True\n")
    site_dir = tmp / "site"                                                   # a site directory holding the kit's generated .pth, as an editable install lays it down
    site_dir.mkdir()
    shutil.copy(str(opt / PTH), str(site_dir / PTH))
    pin = _gate_module().read_table(str(opt / "pyproject.toml"), "tool.opt_core")
    assert set(pin) >= {"path", "version"}, pin
    return {"tmp": tmp, "eng": eng, "opt": opt, "shadow": shadow, "unversioned": unversioned, "bin": bindir, "stub": stub, "site": site_dir, "pin": pin}


def _env(kit, fixture, mode=None, extra_path=()):
    """The child's environment: nothing inherited that names python paths or the kit; PYTHONPATH = [shadow core] + the kit's opt/ (+ extras);
    the -S `python` wrapper first on PATH."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "PXD", "MODEL_OPT", "VIRTUAL_ENV"))}
    paths = {"absent": [], "stale": [str(kit["shadow"])], "unversioned": [str(kit["unversioned"])]}[fixture] + [str(kit["opt"])] + [str(p) for p in extra_path]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PATH"] = str(kit["bin"]) + os.pathsep + os.environ.get("PATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if mode is not None:
        env[ENV] = mode
    return env


def _run(argv, env, cwd=None):
    return subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=120)


def _assert_refused(r, kit, fixture, marker=MARKER):
    """rc 3; exactly one NOT ACTIVE line, the gate's, naming the reason of the fixture; no traceback; the marker never printed."""
    lines = [ln for ln in r.stderr.splitlines() if "NOT ACTIVE" in ln]
    assert r.returncode == 3 and r.stderr.count("NOT ACTIVE") == 1 and len(lines) == 1, (r.returncode, r.stdout[-400:], r.stderr[-800:])
    line = lines[0]
    pin = kit["pin"]
    if fixture == "absent":
        assert line == ("[%s] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v%s at %s; nothing importable as opt_core on sys.path)"
                        % (TAG, pin["version"], pin["path"])), line
    elif fixture == "stale":
        assert line == ("[%s] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v%s at %s, installed v%s at %s"
                        % (TAG, pin["version"], pin["path"], STALE_VERSION, kit["shadow"])), line
    else:
        assert line == ("[%s] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v%s at %s, installed v? at %s"
                        % (TAG, pin["version"], pin["path"], kit["unversioned"])), line
    for word in ("Traceback", "Error processing line", "Fatal Python error"):
        assert word not in r.stderr, (word, r.stderr[-800:])
    if marker:
        assert marker not in r.stdout, r.stdout[-400:]
    return line


FIXTURES = ["absent", "stale", "unversioned"]


# --------------------------------------------------------------------------------------------- python -m / console script / in-process

@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("verb", [["check", "--mode", "exact"], ["check", "--mode", "rows"], ["design", "--mode", "off", "--tasks", "t.json", "--out_dir", "o"], ["warm"]],
                         ids=["check-exact", "check-rows(refused-name:gate-first)", "design-off(stock)", "warm(default-mode)"])
def test_python_m_pxdesign_opt(kit, fixture, verb):
    """`python -m pxdesign_opt <verb>`: the gate is statement one of __main__.py (cli imports the core at module level)."""
    r = _run([sys.executable, "-S", "-m", PKG, *verb], _env(kit, fixture), cwd=str(kit["tmp"]))
    _assert_refused(r, kit, fixture)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_python_m_pxdesign_opt_cli(kit, fixture):
    """`python -m pxdesign_opt.cli …` runs cli.py directly (its __main__ block): the gate is that module's statement one too."""
    r = _run(["python", "-m", PKG + ".cli", "check", "--mode", "exact"], _env(kit, fixture))
    _assert_refused(r, kit, fixture)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_console_script_pxdesign_opt(kit, fixture):
    """`pxdesign-opt check …` exactly as setuptools writes the console script: `from pxdesign_opt.__main__ import main; sys.exit(main())`."""
    code = "import sys; from %s.__main__ import main; sys.exit(main())" % PKG
    r = _run([sys.executable, "-S", "-c", code, "check", "--mode", "exact"], _env(kit, fixture))
    _assert_refused(r, kit, fixture)


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("call", ["k.enable('exact')", "k.enable('rows')", "k.enable('off')", "k.enable('exact', strict=True)", "k.status()", "k.core_gate()",
                                  "k.MODES", "k.DEFAULT_MODE", "k.modes", "k.stack", "k.manifest", "k.report", "__import__('pxdesign_opt.cli')"],
                         ids=["enable(exact)", "enable(rows:refused-name,gate-first)", "enable(off)", "enable(exact,strict)", "status()", "core_gate()",
                              "MODES", "DEFAULT_MODE", "modes", "stack", "manifest", "report", "import-pxdesign_opt.cli"])
def test_in_process_entries(kit, fixture, call):
    """The Python route: `import pxdesign_opt; pxdesign_opt.enable(...)` / `status()` / the lazy names that import a core-reaching module
    (`CORE_ATTRS`) / `import pxdesign_opt.cli` gate before anything of the core is imported."""
    r = _run([sys.executable, "-S", "-c", "import %s as k; %s; print(%r)" % (PKG, call, MARKER)], _env(kit, fixture))
    _assert_refused(r, kit, fixture)


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("mode", [None, "exact"], ids=["env-unset", "PXDESIGN_OPT=exact"])
def test_stock_caller_child_module(kit, fixture, mode):
    """`python -s -m pxdesign_opt.stock_infer …` (the process `design --mode off` starts): gated before its `opt_core.stock_proof` import."""
    r = _run([sys.executable, "-S", "-s", "-m", PKG + ".stock_infer", "--tree", str(kit["eng"]), "--tasks", "t.json", "--out_dir", "o"], _env(kit, fixture, mode))
    _assert_refused(r, kit, fixture)


# ------------------------------------------------------------------------------------------------------------------ bash routes

@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("args", [["check", "--mode", "exact"], ["check"], ["design", "--mode", "off", "--tasks", "t.json", "--out_dir", "o"],
                                  ["warm", "--config", "h100"], ["warm", "--mode=exact"]],
                         ids=["check-exact", "check(no-mode)", "design-off(stock)", "warm-h100", "warm-exact"])
def test_run_sh(kit, fixture, args):
    """`run.sh <verb> …`: the two probes before the configuration is sourced — the package importable, then the gate (rc 3, its line)."""
    r = _run(["bash", str(kit["eng"] / "run.sh"), *args], _env(kit, fixture), cwd=str(kit["tmp"]))
    _assert_refused(r, kit, fixture)
    assert "stock pins" not in r.stderr and "no configuration" not in r.stderr        # refused at the gate, before the config and the pins check


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("mode", [None, "exact", "off"], ids=["env-unset", "PXDESIGN_OPT=exact", "PXDESIGN_OPT=off"])
def test_source_configs_h100_env(kit, fixture, mode):
    """`source configs/h100.env` (the env route's first step): its first python lines are the two probes; a refusal returns 3 from the sourced file."""
    r = _run(["bash", "-c", 'source "%s" || exit $?; echo %s' % (kit["eng"] / "configs" / "h100.env", MARKER)], _env(kit, fixture, mode), cwd=str(kit["tmp"]))
    _assert_refused(r, kit, fixture)
    assert "MODEL_OPT_STACK_KEY" not in r.stderr                                      # the gate precedes the stack-key probe


# -------------------------------------------------------------------------------------------------------------- the .pth route

@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("mode", ["exact", "rows", "Exact "], ids=["PXDESIGN_OPT=exact", "PXDESIGN_OPT=rows(refused-name:gate-first)", "PXDESIGN_OPT=Exact(folded)"])
def test_pth_line_via_addsitedir(kit, fixture, mode):
    """The REAL generated .pth in a site directory, executed by `site.addsitedir` in a `-S` child with a kit mode set: the gate runs inside
    `pxdesign_opt._autoload` at .pth time and ends the interpreter with status 3 (os._exit) — the upstream import after it never runs."""
    code = "import site; site.addsitedir(%r); import pxdesign; print(%r)" % (str(kit["site"]), MARKER)
    r = _run([sys.executable, "-S", "-c", code], _env(kit, fixture, mode, extra_path=[kit["stub"]]))
    _assert_refused(r, kit, fixture)


def _bare_venv(kit):
    """A `venv --without-pip` of this interpreter with the kit's generated .pth in its site-packages: the .pth line at REAL interpreter start.
    None (skip by name) when the venv's own interpreter can already import an opt_core without PYTHONPATH."""
    venv_dir = kit["tmp"] / "venv"
    if not venv_dir.exists():
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv_dir)], check=True, capture_output=True, text=True, timeout=300)
    py = venv_dir / "bin" / "python"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "PXD", "MODEL_OPT", "VIRTUAL_ENV"))}
    probe = subprocess.run([str(py), "-c", "import site, sys; sys.stdout.write(site.getsitepackages()[0]); import opt_core"], env=env, capture_output=True, text=True, timeout=120)
    if probe.returncode == 0:
        return None, None
    sp = probe.stdout.strip()
    assert sp and os.path.isdir(sp), (probe.returncode, probe.stdout, probe.stderr[-400:])
    if not os.path.isfile(os.path.join(sp, PTH)):
        shutil.copy(str(kit["opt"] / PTH), os.path.join(sp, PTH))                    # the kit's generated .pth, as the install lays it down
        with open(os.path.join(sp, "__editable__.pxdesign_opt.pth"), "w") as fh:       # and the editable install's path entry: the package importable with no PYTHONPATH
            fh.write(str(kit["opt"]) + "\n")
    return py, sp


@pytest.mark.parametrize("fixture", ["absent", "stale"])
@pytest.mark.parametrize("route", ["run.sh check", "source configs/h100.env"])
def test_env_variable_form_through_the_installed_pth(kit, fixture, route):
    """`PXDESIGN_OPT=exact ./run.sh check` and `PXDESIGN_OPT=exact source configs/h100.env` with the kit INSTALLED (a bare venv whose
    site-packages hold the generated .pth and the editable path entry; its python first on PATH, site enabled): the .pth line runs at the start
    of every python the script spawns, so the scripts neutralise the variable on their stderr-silenced importability probe and the refusal is the
    gate's one line on the next probe — never a silent rc 3, never the 'not importable' diagnosis."""
    py, _sp = _bare_venv(kit)
    if py is None:
        pytest.skip("the bare venv's interpreter imports an opt_core without PYTHONPATH")
    env = _env(kit, fixture, "exact")
    env["PYTHONPATH"] = os.pathsep.join(p for p in env["PYTHONPATH"].split(os.pathsep) if p != str(kit["opt"]))   # the package comes from site-packages; STALE keeps its shadow core here
    if not env["PYTHONPATH"]:
        del env["PYTHONPATH"]
    env["PATH"] = str(py.parent) + os.pathsep + os.environ.get("PATH", "")          # the venv's python first: no -S wrapper
    if route == "run.sh check":
        r = _run(["bash", str(kit["eng"] / "run.sh"), "check"], env, cwd=str(kit["eng"]))
    else:
        r = _run(["bash", "-c", 'source "$1" || exit $?; echo %s' % MARKER, "_", str(kit["eng"] / "configs" / "h100.env")], env, cwd=str(kit["eng"]))
    _assert_refused(r, kit, fixture)
    assert "not importable" not in r.stderr and "MODEL_OPT_STACK_KEY" not in r.stderr, r.stderr[-600:]


@pytest.mark.parametrize("fixture", FIXTURES)
def test_pth_line_at_interpreter_start(kit, fixture):
    """The .pth executed by site.py while the interpreter initialises (a bare venv holding the kit's .pth; PXDESIGN_OPT=exact; the package —
    and for STALE the shadow core — on PYTHONPATH): the gate's line and exit status 3, never `Fatal Python error` (what a SystemExit at that
    point would be) and never the program text after start-up."""
    py, _sp = _bare_venv(kit)
    if py is None:
        pytest.skip("the bare venv's interpreter imports an opt_core without PYTHONPATH (a core on the base interpreter's path): the addsitedir cases cover the .pth line")
    env = _env(kit, fixture, "exact")
    env.pop("PATH", None); env["PATH"] = os.environ.get("PATH", "")                 # no wrapper: the venv's own python, site enabled
    r = _run([str(py), "-c", "print(%r)" % MARKER], env)
    _assert_refused(r, kit, fixture)


# ------------------------------------------------------------------------------------------------------------ inert routes (core-free)

@pytest.mark.parametrize("mode", [None, "", "off", "OFF"], ids=["env-unset", "PXDESIGN_OPT=''", "PXDESIGN_OPT=off", "PXDESIGN_OPT=OFF"])
def test_package_and_autoload_import_are_core_free(kit, mode):
    """`import pxdesign_opt, pxdesign_opt._autoload` with the variable unset or off, no core anywhere: rc 0, nothing of the core imported, no
    finder, no line — the .pth import is inert in every interpreter that does not select a kit mode."""
    code = ("import sys, %s as k, %s._autoload as a; assert 'opt_core' not in sys.modules, sorted(m for m in sys.modules if 'opt_core' in m); "
            "assert a.FINDER is None; assert k.TAG == a.TAG == %r; print(%r)") % (PKG, PKG, TAG, MARKER)
    r = _run([sys.executable, "-S", "-c", code], _env(kit, "absent", mode))
    assert r.returncode == 0 and MARKER in r.stdout and "NOT ACTIVE" not in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stdout, r.stderr[-600:])


def test_core_free_lazy_attributes_stay_core_free(kit):
    """The package's core-free lazy names (`CORE_FREE_ATTRS`: the stock caller's and the .pth's modules) import under an ABSENT core with no
    gate line and nothing of the core; every other lazy name is in `CORE_ATTRS` (gated: test_in_process_entries)."""
    code = ("import sys, %s as k; mods = [getattr(k, n).__name__ for n in k.CORE_FREE_ATTRS]; "
            "assert 'opt_core' not in sys.modules, sorted(m for m in sys.modules if 'opt_core' in m); "
            "assert set(k.__all__) - {'enable', 'status', 'core_gate', 'TAG', 'ActivationError'} <= set(k.CORE_ATTRS) | set(k.CORE_FREE_ATTRS), k.__all__; print(%r, mods)") % (PKG, MARKER)
    r = _run([sys.executable, "-S", "-c", code], _env(kit, "absent"))
    assert r.returncode == 0 and MARKER in r.stdout and "NOT ACTIVE" not in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stdout, r.stderr[-600:])


@pytest.mark.parametrize("fixture", ["absent", "stale"])
@pytest.mark.parametrize("mode", [None, "off"], ids=["env-unset", "PXDESIGN_OPT=off"])
def test_pth_line_is_inert_without_a_kit_mode(kit, fixture, mode):
    """The generated .pth with the variable unset or off: the upstream import runs (stock), rc 0, stderr carries no NOT ACTIVE, no core imported."""
    code = ("import site, sys; site.addsitedir(%r); import pxdesign; assert pxdesign.STOCK; "
            "assert 'opt_core' not in sys.modules and '%s._autoload' in sys.modules; print(%r)") % (str(kit["site"]), PKG, MARKER)
    r = _run([sys.executable, "-S", "-c", code], _env(kit, fixture, mode, extra_path=[kit["stub"]]))
    assert r.returncode == 0 and MARKER in r.stdout and "NOT ACTIVE" not in r.stderr and "Error processing line" not in r.stderr, (r.returncode, r.stdout, r.stderr[-600:])


def test_pth_line_is_inert_at_interpreter_start_without_a_kit_mode(kit):
    py, _sp = _bare_venv(kit)
    if py is None:
        pytest.skip("the bare venv's interpreter imports an opt_core without PYTHONPATH")
    env = _env(kit, "absent", "off", extra_path=[kit["stub"]])
    env["PATH"] = os.environ.get("PATH", "")
    r = _run([str(py), "-c", "import sys, pxdesign; assert 'opt_core' not in sys.modules; print(%r)" % MARKER], env)
    assert r.returncode == 0 and MARKER in r.stdout and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stdout, r.stderr[-600:])


def test_run_sh_usage_errors_precede_the_gate(kit):
    """run.sh's own usage refusals (rc 2, pure bash) come before any python: no gate line under ABSENT."""
    r = _run(["bash", str(kit["eng"] / "run.sh"), "bogus"], _env(kit, "absent"))
    assert r.returncode == 2 and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stderr)
    r = _run(["bash", str(kit["eng"] / "run.sh"), "serve", "--mode", "exact"], _env(kit, "absent"))                 # not a verb: usage, before the gate
    assert r.returncode == 2 and "usage:" in r.stderr and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stderr)
    r = _run(["bash", str(kit["eng"] / "run.sh"), "check", "--mode", "rows"], _env(kit, "absent"))            # not a mode: refused by name in bash, before the gate
    assert r.returncode == 2 and r.stderr.strip() == "[pxdesign-opt] unknown --mode rows (modes: off|exact|fast|big)", (r.returncode, r.stderr)


def test_the_pth_of_the_fixture_is_the_generated_text(kit):
    """The .pth the route cases execute is the kit's shipped file, and that file is `pth_text(PKG, ENV, TAG)` of the kit's own backend."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_pxdesign_build_backend_under_test", os.path.join(TREE, "opt", "_build_backend.py"))
    bb = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(bb)
    finally:
        os.chdir(cwd)
    shipped = open(os.path.join(TREE, "opt", PTH), encoding="utf-8").read()
    assert shipped == (kit["site"] / PTH).read_text() == bb.pth_text(PKG, ENV, TAG)
    assert bb.pth_fields(shipped) == (PKG, ENV, TAG, 3)
