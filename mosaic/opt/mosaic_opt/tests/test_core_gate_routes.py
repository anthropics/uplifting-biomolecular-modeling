"""The core pin gate through every documented entry of the package. Two broken cores, each through every route: NO shared core anywhere
on the path (a copy of the kit in a temporary tree without common/opt_core, `python -S` children so an installed core is invisible), and an
OLDER core first on the path (a shadow `opt_core` 0.2.5 — below the pin's floor). Every route is exit 3 with ONE
`[mosaic-opt] NOT ACTIVE: reason=core_missing:opt_core` / `reason=core_mismatch: …` line —
never a traceback, never the route's body, never a silent stock run — because `mosaic_opt.core_gate` (→ `_core_gate.gate`, the core's kit
template) is statement one of each: `python -m mosaic_opt` and the console script's body, `enable()` / `status()` and the package's
core-backed attributes (`mosaic_opt.stack` …, `from mosaic_opt import *`), `run.sh` (a kit mode, the stock route, a verb without a mode),
`source configs/h100.env`, the same two under the variable form (`MOSAIC_OPT=exact run.sh check` / `source configs/h100.env` on a venv
whose site-packages carries the generated .pth: the gate speaks once, never the importability probe's line), and the real
`opt/mosaic_opt_autoload.pth` in a temporary site dir at interpreter start under MOSAIC_OPT=<mode> (`off` / unset stay inert and core-free
whatever the core; on the pinned core the hook arms and the program runs). Importing the package, `mosaic_opt._autoload`, the mode
table and the core-free attributes stay core-free and never refuse."""
import glob
import os
import shutil
import subprocess
import sys
import venv

import pytest

from . import core_src
from mosaic_opt import _core_gate                                                  # import-free: nothing of the core

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                       # mosaic/
PIN = _core_gate.read_table(os.path.join(KIT, "opt", "pyproject.toml"), "tool." + _core_gate.PIN_TABLE)   # the kit's pin as the gate reads it: every refusal names it (test_core_adoption locks the table to the tree)
PINNED_VERSION = PIN["version"]


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """<tmp>/mosaic/{opt/{pyproject.toml,_build_backend.py,mosaic_opt/,mosaic_opt_autoload.pth},run.sh,configs/} and NO <tmp>/common/opt_core;
    a shadow older core; a `python` wrapper that runs this interpreter with -S; a fake upstream `mosaic`
    (the hook's trigger package); a site dir holding the real .pth."""
    tmp = str(tmp_path_factory.mktemp("gate"))
    kit = os.path.join(tmp, "mosaic")
    os.makedirs(os.path.join(kit, "opt"))
    shutil.copytree(os.path.join(KIT, "opt", "mosaic_opt"), os.path.join(kit, "opt", "mosaic_opt"), ignore=shutil.ignore_patterns("tests", "__pycache__"))
    for rel in ("opt/pyproject.toml", "opt/_build_backend.py", "opt/mosaic_opt_autoload.pth", "run.sh"):
        shutil.copy(os.path.join(KIT, *rel.split("/")), os.path.join(kit, *rel.split("/")))
    shutil.copytree(os.path.join(KIT, "configs"), os.path.join(kit, "configs"))
    shadow = os.path.join(tmp, "shadow")                                            # an older core: version literal below the pin's floor
    os.makedirs(os.path.join(shadow, "opt_core"))
    open(os.path.join(shadow, "opt_core", "__init__.py"), "w").write('__version__ = "0.2.5"\n')
    bindir = os.path.join(tmp, "bin")
    os.makedirs(bindir)
    wrapper = os.path.join(bindir, "python")
    open(wrapper, "w").write(f'#!/bin/sh\nexec {sys.executable} -S "$@"\n')
    os.chmod(wrapper, 0o755)
    fake = os.path.join(tmp, "fake")
    os.makedirs(os.path.join(fake, "mosaic"))
    open(os.path.join(fake, "mosaic", "__init__.py"), "w").write("")                # upstream's __init__ is empty too
    site_dir = os.path.join(tmp, "site")
    os.makedirs(site_dir)
    shutil.copy(os.path.join(KIT, "opt", "mosaic_opt_autoload.pth"), site_dir)
    vdir = os.path.join(tmp, "venv")                                               # an install as pip leaves it: the generated .pth and the package on a venv's site-packages
    venv.create(vdir, with_pip=False, symlinks=True)
    vsite = glob.glob(os.path.join(vdir, "lib", "python*", "site-packages"))[0]
    shutil.copy(os.path.join(KIT, "opt", "mosaic_opt_autoload.pth"), vsite)
    open(os.path.join(vsite, "_kit_tree.pth"), "w").write(os.path.join(kit, "opt") + "\n")   # the editable install's path line
    assert not os.path.exists(os.path.join(tmp, "common"))
    return {"tmp": tmp, "kit": kit, "opt": os.path.join(kit, "opt"), "shadow": shadow, "bin": bindir, "fake": fake, "site": site_dir,
            "vbin": os.path.join(vdir, "bin")}


CORES = ["absent", "stale"]


def _paths(tree, core):
    """The import path of a child: the broken core first (none for 'absent'), then the kit's opt/."""
    return {"absent": [], "stale": [tree["shadow"]], "pinned": [core_src()]}[core] + [tree["opt"]]


def _env(tree, core, extra=None):
    """PYTHONPATH carries the kit (and the broken core first); PATH starts with the -S wrapper; no MOSAIC_OPT unless `extra` sets it."""
    env = {"PATH": tree["bin"] + os.pathsep + os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join(_paths(tree, core)), "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": tree["tmp"], "LANG": "C.UTF-8"}
    env.update(extra or {})
    return env


def _assert_refused(r, core, marker=None):
    want = "NOT ACTIVE: reason=core_missing:opt_core" if core == "absent" else "NOT ACTIVE: reason=core_mismatch:"
    assert r.returncode == 3, (r.returncode, r.stderr[-800:], r.stdout[-300:])
    assert r.stderr.count("NOT ACTIVE") == 1 and want in r.stderr and r.stderr.count("[mosaic-opt] NOT ACTIVE") == 1, r.stderr[-800:]
    assert "v" + PINNED_VERSION in r.stderr, r.stderr[-800:]                                         # the pin's floor version is named on every refusal
    if core == "stale":
        assert "v0.2.5" in r.stderr, r.stderr[-800:]                                                 # both sides: pinned version and installed version
    assert "Traceback" not in r.stderr and "Error processing line" not in r.stderr and "Fatal Python error" not in r.stderr, r.stderr[-800:]
    if marker:
        assert marker not in r.stdout, r.stdout[-300:]


@pytest.mark.parametrize("core", CORES)
def test_python_dash_m(tree, core):
    r = subprocess.run(["python", "-m", "mosaic_opt", "check", "--mode", "exact"], env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core)


@pytest.mark.parametrize("core", CORES)
def test_console_script_body(tree, core):
    """`mosaic-opt` = the setuptools-written script: `from mosaic_opt.__main__ import main; sys.exit(main())`."""
    r = subprocess.run(["python", "-c", "import re, sys; from mosaic_opt.__main__ import main; sys.exit(main())", "check", "--mode", "exact"],
                       env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core)


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("call", ["k.enable('exact')", "k.enable('off')", "k.status()", "k.stack", "k.report", "k.det", "k.warm"])
def test_in_process_entries(tree, core, call):
    r = subprocess.run(["python", "-c", f"import mosaic_opt as k; {call}; print('BODY RAN')"], env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, marker="BODY RAN")


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("stmt", ["from mosaic_opt import stack", "from mosaic_opt import *"], ids=["from-import-stack", "star-import"])
def test_import_statements_of_core_backed_names(tree, core, stmt):
    r = subprocess.run(["python", "-c", f"{stmt}; print('BODY RAN')"], env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, marker="BODY RAN")


@pytest.mark.parametrize("core", CORES)
def test_in_process_entry_under_the_variable(tree, core):
    """MOSAIC_OPT=exact exported and no .pth in play (a tree on PYTHONPATH): `enable()` is still one line and status 3."""
    r = subprocess.run(["python", "-c", "import mosaic_opt as k; k.enable('exact'); print('BODY RAN')"], env=_env(tree, core, {"MOSAIC_OPT": "exact"}),
                       capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, marker="BODY RAN")


@pytest.mark.parametrize("core", CORES)
def test_package_import_autoload_and_mode_table_stay_core_free(tree, core):
    """Importing the package, the .pth module (variable unset) and the mode table locate and import nothing of the core and never refuse:
    the gate is at the entries."""
    prog = ("import mosaic_opt as k, mosaic_opt._autoload, sys; assert k.MODES == ('fast', 'exact', 'big', 'off'); "
            "mods = [getattr(k, n).__name__ for n in k.CORE_FREE]; assert len(mods) == len(k.CORE_FREE), mods; "
            "assert 'opt_core' not in sys.modules and 'mosaic_opt._core_gate' not in sys.modules, sorted(sys.modules); print('INERT')")
    r = subprocess.run(["python", "-c", prog], env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    assert r.returncode == 0 and "INERT" in r.stdout and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stderr[-600:])


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("argv", [["check", "--mode", "exact"], ["check", "--mode", "off"], ["design", "--mode", "off", "--out", "o"], ["warm"]],
                         ids=["check-exact", "check-off", "design-off", "warm-nomode"])
def test_run_sh(tree, core, argv):
    r = subprocess.run(["bash", os.path.join(tree["kit"], "run.sh"), *argv], env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core)


def _venv_env(tree, core):
    """The venv's python first on PATH (its site-packages carries the generated .pth and the kit's path line), MOSAIC_OPT=exact exported,
    the broken core (if any) on PYTHONPATH."""
    env = {"PATH": tree["vbin"] + os.pathsep + os.environ.get("PATH", ""), "MOSAIC_OPT": "exact", "PYTHONDONTWRITEBYTECODE": "1", "HOME": tree["tmp"], "LANG": "C.UTF-8"}
    broken = _paths(tree, core)[:-1]
    if broken:
        env["PYTHONPATH"] = os.pathsep.join(broken)
    return env


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("cmd", [["bash", "run.sh", "check"], ["bash", "-c", "source configs/h100.env || exit $?; echo REACHED"]], ids=["run.sh-check", "source-h100.env"])
def test_variable_form_on_an_installed_pth(tree, core, cmd):
    """`MOSAIC_OPT=exact run.sh check` / `MOSAIC_OPT=exact; source configs/h100.env` with the .pth installed: every python the scripts start
    gates at interpreter start, so the importability probe runs with the variable neutralised and the gate probe speaks — once, by name;
    never the probe's own 'not importable' line."""
    r = subprocess.run(cmd, env=_venv_env(tree, core), capture_output=True, text=True, cwd=tree["kit"])
    _assert_refused(r, core, marker="REACHED")
    assert "not importable" not in r.stderr, r.stderr[-800:]


def test_variable_form_probe_is_inert_on_the_venv(tree):
    """The fixture itself: the venv imports the temp kit's package through its path line, and the neutralised probe is silent and 0 even
    with no core anywhere (importability is not activation)."""
    env = dict(_venv_env(tree, "absent"), MOSAIC_OPT="")
    r = subprocess.run(["python", "-c", "import mosaic_opt, sys; print(mosaic_opt.__file__); assert 'opt_core' not in sys.modules"], env=env, capture_output=True, text=True, cwd=tree["tmp"])
    assert r.returncode == 0 and r.stdout.startswith(os.path.join(tree["opt"], "mosaic_opt")) and r.stderr == "", (r.returncode, r.stdout, r.stderr[-600:])


@pytest.mark.parametrize("core", CORES)
def test_source_h100_env(tree, core):
    r = subprocess.run(["bash", "-c", f"source {os.path.join(tree['kit'], 'configs', 'h100.env')} || exit $?; echo REACHED"],
                       env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, marker="REACHED")


def _pth_child(tree, core, mode_env, prog_tail="import mosaic; print('stock ran')"):
    """The REAL generated .pth processed by site.addsitedir in a -S child; under a set variable the hook gates at start-up, before `import mosaic`."""
    paths = _paths(tree, core) + [tree["fake"]]
    prog = f"import sys, site; sys.path[:0] = {paths!r}; site.addsitedir({tree['site']!r}); {prog_tail}"
    env = {"PATH": os.environ.get("PATH", ""), "HOME": tree["tmp"], "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(mode_env)
    return subprocess.run([sys.executable, "-S", "-c", prog], env=env, capture_output=True, text=True, cwd=tree["tmp"])


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_real_pth_under_a_set_variable(tree, core, mode):
    """A kit mode and an unknown mode alike: the gate is the hook's first statement under any value but off / unset."""
    _assert_refused(_pth_child(tree, core, {"MOSAIC_OPT": mode}), core, marker="stock ran")


@pytest.mark.parametrize("core", CORES)
def test_real_pth_off_and_unset_stay_inert_whatever_the_core(tree, core):
    for mode_env in ({"MOSAIC_OPT": "off"}, {"MOSAIC_OPT": ""}, {}):
        r = _pth_child(tree, core, mode_env, prog_tail="import mosaic, sys; assert 'opt_core' not in sys.modules; print('stock ran')")
        assert r.returncode == 0 and "stock ran" in r.stdout and "NOT ACTIVE" not in r.stderr and "Error processing line" not in r.stderr, (mode_env, r.returncode, r.stderr[-600:])


def test_real_pth_on_the_pinned_core_arms_the_hook_and_the_program_runs(tree):
    """The pinned core (this tree's common/opt_core) beside the kit: the same .pth passes the gate, installs the finder and the program runs
    (nothing imports the trigger here, so nothing activates)."""
    r = _pth_child(tree, "pinned", {"MOSAIC_OPT": "exact"},
                   prog_tail="import sys; print('finder armed' if any(type(f).__name__ == 'Finder' for f in sys.meta_path) else 'no finder')")
    assert r.returncode == 0 and "finder armed" in r.stdout and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stdout[-300:], r.stderr[-600:])
