"""The core pin gate through every documented entry of the package. Three broken cores, each through every route: NO shared core anywhere
on the path (a copy of the kit in a temporary tree without common/opt_core, `python -S` children so an installed core is invisible), an
OLDER core first on the path (a shadow `opt_core` 0.2.5), and a core with no `__version__` literal at all (unversioned). Every route is
exit 3 with ONE `[proteinmpnn-opt] NOT ACTIVE: reason=core_missing:opt_core` / `reason=core_mismatch: …` line — never a traceback, never the
route's body, never a silent stock run — because `proteinmpnn_opt.core_gate` (→ `_core_gate.gate`, the core's kit template) is statement one
of each: `python -m proteinmpnn_opt` and the console script's body, `enable()` / `status()`, a by-hand `stage` import,
`run.sh` (a kit mode, the stock route, a verb without a mode, `--config h100`) and `source configs/h100.env`. Importing the package itself
stays core-free. This kit ships no autoload `.pth` (argv-only: test_cli_only_no_autoload below), so there is no `.pth` route."""
import glob
import json
import os
import shutil
import site
import subprocess
import sys
import sysconfig

import pytest

from proteinmpnn_opt import TAG, _core_gate                                     # both import-free: nothing of the core

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                       # proteinmpnn/
TAG_LINE = "[%s] NOT ACTIVE" % TAG
PIN = _core_gate.read_table(os.path.join(KIT, "opt", "pyproject.toml"), "tool." + _core_gate.PIN_TABLE)   # the kit's pin as the gate reads it: every refusal names it
PINNED_VERSION = PIN["version"]
STALE_VERSION = "0.2.5"                      # an older core: the floor semantics refuse it (never "same version, different bytes")
CORES = ["absent", "stale", "unversioned"]


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """<tmp>/proteinmpnn/{opt/{pyproject.toml,_build_backend.py,proteinmpnn_opt/},run.sh,configs/} and NO <tmp>/common/opt_core; a shadow
    older core and a shadow core with no `__version__` literal at all; a `python` wrapper that runs this interpreter with -S."""
    tmp = str(tmp_path_factory.mktemp("gate"))
    kit = os.path.join(tmp, "proteinmpnn")
    os.makedirs(os.path.join(kit, "opt"))
    shutil.copytree(os.path.join(KIT, "opt", "proteinmpnn_opt"), os.path.join(kit, "opt", "proteinmpnn_opt"), ignore=shutil.ignore_patterns("tests", "__pycache__"))
    for rel in ("opt/pyproject.toml", "opt/_build_backend.py", "run.sh"):
        shutil.copy(os.path.join(KIT, *rel.split("/")), os.path.join(kit, *rel.split("/")))
    shutil.copytree(os.path.join(KIT, "configs"), os.path.join(kit, "configs"))
    stale = os.path.join(tmp, "stale")
    os.makedirs(os.path.join(stale, "opt_core"))
    open(os.path.join(stale, "opt_core", "__init__.py"), "w").write(f'__version__ = "{STALE_VERSION}"\n')
    unversioned = os.path.join(tmp, "unversioned")
    os.makedirs(os.path.join(unversioned, "opt_core"))
    open(os.path.join(unversioned, "opt_core", "__init__.py"), "w").write("# no __version__ literal\n")
    bindir = os.path.join(tmp, "bin")
    os.makedirs(bindir)
    wrapper = os.path.join(bindir, "python")
    open(wrapper, "w").write(f'#!/bin/sh\nexec {sys.executable} -S "$@"\n')
    os.chmod(wrapper, 0o755)
    assert not os.path.exists(os.path.join(tmp, "common"))
    return {"tmp": tmp, "kit": kit, "opt": os.path.join(kit, "opt"), "stale": stale, "unversioned": unversioned, "bin": bindir}


def _env(tree, core, extra=None):
    """core: 'absent' | 'stale' | 'unversioned'. PYTHONPATH carries the kit (and the shadow core first for the two shadow cases); PATH starts
    with the -S wrapper; nothing of this session's package variables leaks in."""
    pp = [tree["opt"]] if core == "absent" else [tree[core], tree["opt"]]
    env = {"PATH": tree["bin"] + os.pathsep + os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join(pp), "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": tree["tmp"], "LANG": "C.UTF-8"}
    env.update(extra or {})
    return env


def _assert_refused(r, core, marker=None):
    want = "NOT ACTIVE: reason=core_missing:opt_core" if core == "absent" else "NOT ACTIVE: reason=core_mismatch:"
    assert r.returncode == 3, (r.returncode, r.stderr[-800:], r.stdout[-300:])
    assert r.stderr.count("NOT ACTIVE") == 1 and want in r.stderr and r.stderr.count(TAG_LINE) == 1, r.stderr[-800:]
    assert "v" + PINNED_VERSION in r.stderr, r.stderr[-800:]      # every refusal names the pin's minimum version
    if core == "stale":
        assert "installed v%s at" % STALE_VERSION in r.stderr, r.stderr[-800:]
    if core == "unversioned":
        assert "installed v? at" in r.stderr, r.stderr[-800:]    # no __version__ literal: the gate formats it as "?"
    assert "Traceback" not in r.stderr and "ModuleNotFoundError" not in r.stderr and "Error processing line" not in r.stderr, r.stderr[-800:]
    if marker:
        assert marker not in r.stdout, r.stdout[-300:]


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("argv", [["check", "--mode", "exact"], ["design", "--mode", "off", "--input", "x", "--out", "o"], ["help"]], ids=["check-exact", "design-off", "help"])
def test_python_dash_m(tree, core, argv):
    r = subprocess.run(["python", "-m", "proteinmpnn_opt", *argv], env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core)


@pytest.mark.parametrize("core", CORES)
def test_console_script_body(tree, core):
    """`proteinmpnn-opt` = the setuptools-written script: `from proteinmpnn_opt.__main__ import main; sys.exit(main())` (opt/pyproject.toml [project.scripts])."""
    r = subprocess.run(["python", "-c", "import re, sys; from proteinmpnn_opt.__main__ import main; sys.exit(main())", "check", "--mode", "exact"],
                       env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core)


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("call", ["k.enable('exact', 'soluble')", "k.enable()", "k.status()", "import proteinmpnn_opt.stage"], ids=["enable-exact", "enable-env", "status", "stage-import"])
def test_in_process_entries(tree, core, call):
    r = subprocess.run(["python", "-c", f"import proteinmpnn_opt as k; {call}; print('BODY RAN')"], env=_env(tree, core, {"PROTEINMPNN_OPT": "exact"}),
                       capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, marker="BODY RAN")


@pytest.mark.parametrize("core", CORES)
def test_package_import_stays_core_free(tree, core):
    """Importing the package imports nothing of the core and never refuses, whatever core is or is not on the path: the gate is at the entries."""
    r = subprocess.run(["python", "-c", "import proteinmpnn_opt as k, sys; assert 'opt_core' not in sys.modules, sorted(sys.modules); "
                        "assert [m for m in sys.modules if m.startswith('proteinmpnn_opt')] == ['proteinmpnn_opt'], sorted(sys.modules); "
                        f"assert k.TAG == {TAG!r}; print('INERT')"],
                       env=_env(tree, core, {"PROTEINMPNN_OPT": "exact"}), capture_output=True, text=True, cwd=tree["tmp"])
    assert r.returncode == 0 and "INERT" in r.stdout and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stderr[-600:])


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("argv", [["check", "--mode", "exact"], ["design", "--mode", "off", "--input", "x", "--out", "o"], ["check"], ["check", "--config", "h100", "--mode", "exact"]],
                         ids=["check-exact", "design-off", "check-default-mode", "check-config-h100"])
def test_run_sh(tree, core, argv):
    r = subprocess.run(["bash", os.path.join(tree["kit"], "run.sh"), *argv], env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core)


@pytest.mark.parametrize("core", CORES)
def test_source_h100_env(tree, core):
    r = subprocess.run(["bash", "-c", f"source {os.path.join(tree['kit'], 'configs', 'h100.env')} || exit $?; echo REACHED"],
                       env=_env(tree, core), capture_output=True, text=True, cwd=tree["tmp"])
    _assert_refused(r, core, marker="REACHED")


def test_run_sh_install_and_usage_are_not_gated(tree):
    """Two run.sh paths reach no gate by design: a usage error exits 2 before any probe, and `install` is the verb that installs the pinned pair."""
    r = subprocess.run(["bash", os.path.join(tree["kit"], "run.sh"), "frobnicate"], env=_env(tree, "absent"), capture_output=True, text=True, cwd=tree["tmp"])
    assert r.returncode == 2 and TAG_LINE not in r.stderr and "usage" not in r.stdout, (r.returncode, r.stderr[-400:])   # the usage text (run.sh's header) on stderr, rc 2
    src = open(os.path.join(tree["kit"], "run.sh"), encoding="utf-8").read()
    install = [l for l in src.splitlines() if l.startswith('if [ "${1:-}" = install ]')]
    assert len(install) == 1 and src.index(install[0]) < src.index("pip install -e") < src.index("core_gate()"), install   # the install block: its pip step runs before any gate


def test_cli_only_no_autoload():
    """RF-6, table B (no `.pth` ships): the env route (`PROTEINMPNN_OPT`) is not offered on a direct stock invocation — no `*.pth` /
    `_autoload.py` anywhere in the tree, none declared by `opt/pyproject.toml`, none in any site directory of the running interpreter, no
    patch file in the carried tree, `stack.py` (not an import hook) is the actual reader, and the README states the precondition."""
    assert not (glob.glob(os.path.join(KIT, "**", "*.pth"), recursive=True) + glob.glob(os.path.join(KIT, "**", "_autoload.py"), recursive=True))
    pyproject = open(os.path.join(KIT, "opt", "pyproject.toml"), encoding="utf-8").read()
    assert "autoload.pth" not in pyproject
    assert "include-package-data = false" in pyproject
    sites = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"], *site.getsitepackages(), site.getusersitepackages()}
    assert not [p for s in sites if os.path.isdir(s) for p in glob.glob(os.path.join(s, "proteinmpnn_opt*autoload*"))]
    assert not glob.glob(os.path.join(KIT, "opt", "forward", "**", "*.patch"), recursive=True)   # the tree carries no patch file: the executables are whole files, staged as copies
    pkg = os.path.join(KIT, "opt", "proteinmpnn_opt")
    readers = sorted(n for n in os.listdir(pkg) if n.endswith(".py")
                      and ('os.environ.get(ENV_MODE)' in open(os.path.join(pkg, n), encoding="utf-8").read()
                           or 'ENV_MODE = "PROTEINMPNN_OPT"' in open(os.path.join(pkg, n), encoding="utf-8").read()))
    assert "stack.py" in readers, readers                                          # the resolver, not an import hook; cli.py calls it, does not read the env itself
    for rel in (("stock", "src", "protein_mpnn_run.py"), ("opt", "forward", "mpnn_exact_worker", "addon", "mpnn_worker2.py")):
        assert "PROTEINMPNN_OPT" not in open(os.path.join(KIT, *rel), encoding="utf-8").read(), rel   # no design process reads the mode variable: stack.py, behind the CLI, is its one reader
