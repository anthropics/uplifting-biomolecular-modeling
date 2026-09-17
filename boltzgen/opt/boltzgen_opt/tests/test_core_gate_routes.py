"""The core pin gate through every documented entry of the package. Two broken cores, each through every route: NO shared core anywhere
on the path (a copy of the kit in a temporary tree without common/opt_core, `python -S` children so an installed core is invisible), an
OLDER core first on the path (a shadow `opt_core` 0.2.5), and an UNVERSIONED core (importable, but no `__version__` line — `installed v?`
in the refusal). The gate never reads a MANIFEST.json (there is none in these shadows). Every route is exit 3 with ONE
`[boltzgen-opt] NOT ACTIVE: reason=core_missing:opt_core` / `reason=core_mismatch: …` line — never a traceback, never the route's body,
never a silent stock run — because `boltzgen_opt.core_gate` (→ `_core_gate.gate`, the core's kit template) is statement one of each:
`python -m boltzgen_opt` and the console script's body, `enable()` / `status()` / `ActivationError`, `run.sh` (a kit mode, the stock route
`--mode off`, `check` without a mode (the default), the mode from the environment), `source configs/h100.env` (also with the mode in the
environment), and the real
`opt/boltzgen_opt_autoload.pth` in a temporary site dir at its trigger (`import boltzgen` under BOLTZGEN_OPT=<value>; `off` / unset stay
inert). Importing the package, `MODES`, `DEFAULT_MODE` and the `.pth` module itself touch nothing of the core and never refuse.
"""
import os
import shutil
import subprocess
import sys

import pytest

from boltzgen_opt import _core_gate, report                                       # both import-free: nothing of the core

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                       # boltzgen/
PTH = "boltzgen_opt_autoload.pth"
PIN = _core_gate.read_table(os.path.join(KIT, "opt", "pyproject.toml"), "tool." + _core_gate.PIN_TABLE)   # the kit's pin as the gate reads it: every refusal names it
PINNED_VERSION, PINNED_PATH = PIN["version"], PIN["path"]
OLDER_VERSION = "0.2.5"                                                           # older than PINNED_VERSION: FLOOR semantics refuse it
TAG_LINE = f"[{report.TAG}] NOT ACTIVE"


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """<tmp>/boltzgen/{opt/{pyproject.toml,_build_backend.py,boltzgen_opt/,boltzgen_opt_autoload.pth},run.sh,configs/} and NO
    <tmp>/common/opt_core; a shadow older core and a shadow unversioned core (no MANIFEST.json in either — the gate never reads one); a
    `python` wrapper that runs this interpreter with -S; a fake upstream `boltzgen`; a site dir holding the real .pth."""
    tmp = str(tmp_path_factory.mktemp("gate"))
    kit = os.path.join(tmp, "boltzgen")
    os.makedirs(os.path.join(kit, "opt"))
    shutil.copytree(os.path.join(KIT, "opt", "boltzgen_opt"), os.path.join(kit, "opt", "boltzgen_opt"), ignore=shutil.ignore_patterns("tests", "__pycache__"))
    for rel in ("opt/pyproject.toml", "opt/_build_backend.py", "opt/" + PTH, "run.sh"):
        shutil.copy(os.path.join(KIT, *rel.split("/")), os.path.join(kit, *rel.split("/")))
    shutil.copytree(os.path.join(KIT, "configs"), os.path.join(kit, "configs"))
    shadows = {}
    for name, init_text in (("older", f'__version__ = "{OLDER_VERSION}"\n'), ("unversioned", "# no __version__ literal here\n")):
        shadow = os.path.join(tmp, "shadow_" + name)
        os.makedirs(os.path.join(shadow, "opt_core"))
        open(os.path.join(shadow, "opt_core", "__init__.py"), "w").write(init_text)
        shadows[name] = shadow
    bindir = os.path.join(tmp, "bin")
    os.makedirs(bindir)
    wrapper = os.path.join(bindir, "python")
    open(wrapper, "w").write(f'#!/bin/sh\nexec {sys.executable} -S "$@"\n')
    os.chmod(wrapper, 0o755)
    fake = os.path.join(tmp, "fake")
    os.makedirs(os.path.join(fake, "boltzgen"))
    open(os.path.join(fake, "boltzgen", "__init__.py"), "w").write('__version__ = "0.3.2"\n')
    site_dir = os.path.join(tmp, "site")
    os.makedirs(site_dir)
    shutil.copy(os.path.join(KIT, "opt", PTH), site_dir)
    assert not os.path.exists(os.path.join(tmp, "common"))
    probe = subprocess.run([wrapper, "-c", "import opt_core"], env={"PYTHONPATH": os.path.join(kit, "opt"), "PATH": os.environ.get("PATH", "")}, capture_output=True, text=True)
    assert probe.returncode != 0, "precondition: no opt_core is importable in a -S child of the temporary tree"
    return {"tmp": tmp, "kit": kit, "opt": os.path.join(kit, "opt"), "shadows": shadows, "bin": bindir, "fake": fake, "site": site_dir}


def _paths(tree, core):
    return ([] if core == "absent" else [tree["shadows"][core]]) + [tree["opt"]]


def _env(tree, core, extra=None):
    """core: 'absent' | 'older' | 'unversioned'. PYTHONPATH carries the kit (and the shadow core first unless absent); PATH starts with the -S wrapper."""
    env = {"PATH": tree["bin"] + os.pathsep + os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join(_paths(tree, core)), "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": tree["tmp"], "LANG": "C.UTF-8"}
    env.update(extra or {})
    return env


def _assert_refused(r, core, tree, marker=None):
    assert r.returncode == 3, (r.returncode, r.stderr[-800:], r.stdout[-300:])
    assert r.stderr.count("NOT ACTIVE") == 1 and r.stderr.count(TAG_LINE) == 1, r.stderr[-800:]
    assert f"pinned >= v{PINNED_VERSION} at {PINNED_PATH}" in r.stderr, r.stderr[-800:]              # the pinned side is always named this way
    if core == "absent":
        assert "reason=core_missing:opt_core (" in r.stderr and "nothing importable as opt_core on sys.path)" in r.stderr, r.stderr[-800:]
    else:
        have = OLDER_VERSION if core == "older" else "?"
        assert f"reason=core_mismatch: opt_core pinned >= v{PINNED_VERSION} at {PINNED_PATH}, installed v{have} at {tree['shadows'][core]}" in r.stderr, r.stderr[-800:]
    assert "Traceback" not in r.stderr and "Error processing line" not in r.stderr, r.stderr[-800:]
    if marker:
        assert marker not in r.stdout, r.stdout[-300:]


CORES = ["absent", "older", "unversioned"]


@pytest.mark.parametrize("core", CORES)
def test_python_dash_m(tree, core):
    r = subprocess.run(["python", "-m", "boltzgen_opt", "check", "--mode", "exact"], env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, tree)


@pytest.mark.parametrize("core", CORES)
def test_console_script_body(tree, core):
    """`boltzgen-opt` = the setuptools-written script: `from boltzgen_opt.__main__ import main; sys.exit(main())`."""
    r = subprocess.run(["python", "-c", "import re, sys; from boltzgen_opt.__main__ import main; sys.exit(main())", "check", "--mode", "exact"],
                       env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, tree)


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("call", ["k.enable('exact')", "k.enable('off')", "k.enable('big', dry_run=True)", "k.status()", "k.ActivationError"])
def test_in_process_entries(tree, core, call):
    r = subprocess.run(["python", "-c", f"import boltzgen_opt as k; {call}; print('BODY RAN')"], env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, tree, marker="BODY RAN")


@pytest.mark.parametrize("core", CORES)
def test_package_import_modes_and_the_pth_module_stay_core_free(tree, core):
    """Importing the package, `MODES` / `DEFAULT_MODE` / `__version__` and the .pth module import nothing of the core and never refuse: the gate is at the entries."""
    r = subprocess.run(["python", "-c", "import boltzgen_opt as k, boltzgen_opt._autoload, sys; k.MODES; k.DEFAULT_MODE; k.__version__; "
                        "assert 'opt_core' not in sys.modules, sorted(sys.modules); print('INERT')"],
                       env=_env(tree, core, {"BOLTZGEN_OPT": "exact"}), capture_output=True, text=True, cwd=tree["tmp"])
    assert r.returncode == 0 and "INERT" in r.stdout and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stderr[-600:])


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("argv", [["check", "--mode", "exact"], ["design", "--mode", "off", "x.yaml", "--output", "o"], ["warm", "--mode", "big"], ["check"]],
                         ids=["check-exact", "design-off(stock route)", "warm-big", "check(no mode: the default)"])
def test_run_sh(tree, core, argv):
    r = subprocess.run(["bash", os.path.join(tree["kit"], "run.sh"), *argv], env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, tree)


@pytest.mark.parametrize("core", CORES)
def test_source_h100_env(tree, core):
    r = subprocess.run(["bash", "-c", f"source {os.path.join(tree['kit'], 'configs', 'h100.env')} || exit $?; echo REACHED"],
                       env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, tree, marker="REACHED")


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("mode", ["exact", "big"])
def test_run_sh_and_the_config_with_the_mode_in_the_environment(tree, core, mode):
    """`BOLTZGEN_OPT=<mode> run.sh check` / `BOLTZGEN_OPT=<mode> source configs/h100.env`: the gate's own line (probe 2), never probe 1's
    'not importable' diagnosis — the importability probe runs with the variable removed and the hook is a lazy trigger anyway."""
    env = _env(tree, core, {"BOLTZGEN_OPT": mode})
    r = subprocess.run(["bash", os.path.join(tree["kit"], "run.sh"), "check"], env=env, capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, tree)
    assert "not importable" not in r.stderr, r.stderr[-600:]
    r = subprocess.run(["bash", "-c", f"source {os.path.join(tree['kit'], 'configs', 'h100.env')} || exit $?; echo REACHED"], env=env, capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, tree, marker="REACHED")
    assert "not importable" not in r.stderr, r.stderr[-600:]


def _pth_child(tree, core, mode_env):
    """The REAL generated .pth processed by site.addsitedir in a -S child: the hook installs the finder at start-up; `import boltzgen` is the trigger."""
    paths = _paths(tree, core) + [tree["fake"]]
    prog = f"import sys, site; sys.path[:0] = {paths!r}; site.addsitedir({tree['site']!r}); import boltzgen; print('stock ran')"
    env = {"PATH": os.environ.get("PATH", ""), "HOME": tree["tmp"], "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(mode_env)
    return subprocess.run([sys.executable, "-S", "-c", prog], env=env, capture_output=True, text=True, cwd=tree["tmp"])


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("mode", ["exact", "big", "fast", "faster"], ids=["exact", "big", "fast", "faster(not a mode: the gate still comes first)"])
def test_real_pth_trigger_under_a_set_variable(tree, core, mode):
    _assert_refused(_pth_child(tree, core, {"BOLTZGEN_OPT": mode}), core, tree, marker="stock ran")


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("mode", ["exact", "big", "fast"])
def test_real_pth_at_start_without_the_trigger_is_inert_under_a_set_variable(tree, core, mode):
    """The hook is a lazy trigger: with the variable naming a mode and a broken core, an interpreter that processes the .pth and imports the
    package (never the upstream trigger) starts and ends clean — what run.sh's / the config's importability probe relies on."""
    paths = _paths(tree, core)
    prog = f"import sys, site; sys.path[:0] = {paths!r}; site.addsitedir({tree['site']!r}); import boltzgen_opt; assert 'opt_core' not in sys.modules; print('IMPORT OK')"
    env = {"PATH": os.environ.get("PATH", ""), "HOME": tree["tmp"], "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "BOLTZGEN_OPT": mode}
    r = subprocess.run([sys.executable, "-S", "-c", prog], env=env, capture_output=True, text=True, cwd=tree["tmp"])
    assert r.returncode == 0 and "IMPORT OK" in r.stdout and "NOT ACTIVE" not in r.stderr and "Error processing line" not in r.stderr, (r.returncode, r.stderr[-600:])


@pytest.mark.parametrize("core", CORES)
def test_real_pth_off_and_unset_stay_inert_whatever_the_core(tree, core):
    for mode_env in ({"BOLTZGEN_OPT": "off"}, {"BOLTZGEN_OPT": ""}, {}):
        r = _pth_child(tree, core, mode_env)
        assert r.returncode == 0 and "stock ran" in r.stdout and "NOT ACTIVE" not in r.stderr and "Error processing line" not in r.stderr, (mode_env, r.returncode, r.stderr[-600:])


def test_a_missing_package_under_a_set_variable_is_the_guards_line(tree):
    """The generated .pth guard itself: the .pth in a site dir but the package NOT importable — under a kit mode one NOT ACTIVE line and exit 3
    (never site.py's swallowed 'Error processing line'); unset / off: nothing."""
    prog = f"import site; site.addsitedir({tree['site']!r}); print('stock ran')"
    base = {"PATH": os.environ.get("PATH", ""), "HOME": tree["tmp"], "LANG": "C.UTF-8"}
    r = subprocess.run([sys.executable, "-S", "-c", prog], env=dict(base, BOLTZGEN_OPT="exact"), capture_output=True, text=True, cwd=tree["tmp"])
    assert r.returncode == 3 and r.stderr.count(TAG_LINE) == 1 and "boltzgen_opt._autoload is not importable" in r.stderr and "stock ran" not in r.stdout, (r.returncode, r.stderr[-600:])
    assert "Error processing line" not in r.stderr and "Traceback" not in r.stderr, r.stderr[-600:]
    r = subprocess.run([sys.executable, "-S", "-c", prog], env=base, capture_output=True, text=True, cwd=tree["tmp"])
    assert (r.returncode, r.stderr, r.stdout.strip()) == (0, "", "stock ran")
