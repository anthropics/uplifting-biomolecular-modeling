"""The core pin gate through every documented entry of the package. Two broken cores, each through every route: NO shared core anywhere
on the path (a copy of the kit in a temporary tree without common/opt_core, `python -S` children so an installed core is invisible), and an
OLDER core first on the path (a shadow `opt_core` 0.2.5 — the pin is a floor, so only "absent" and "older" are refusals; a newer core
passes and is not exercised here). Every route is exit 3 with ONE `[caliby-opt] NOT ACTIVE: reason=core_missing:opt_core` / `reason=core_mismatch: …` line — never a traceback, never
the route's body, never a silent stock run — because `stack.core_gate` (→ `_core_gate.gate`, the core's kit template) is statement one of
each: `python -m caliby_opt` and the console script's body, `enable()` / `status()` / `check()`, the design children started by hand
(`python -m caliby_opt.design_run`, `python -s -m caliby_opt.stock_design`), `run.sh` (a kit mode, the stock route `--mode off`, no
`--mode`, every verb), `source configs/h100.env`, and the real `opt/caliby_opt_autoload.pth` in a temporary site dir at its trigger
(`import caliby` under CALIBY_OPT=fast; `off` / unset stay inert and core-free; an unknown CALIBY_OPT is refused at the trigger by name
before the gate). The package import and `caliby_opt._autoload` are core-free routes:
exit 0 and nothing of the core imported with no core on the path."""
import os
import shutil
import subprocess
import sys

import pytest

from .. import _core_gate, stack

KIT = stack.tree_home()                                                             # caliby/
OPT = stack.opt_home()                                                              # caliby/opt
PIN = _core_gate.read_table(os.path.join(OPT, "pyproject.toml"), "tool.opt_core")   # the kit's pin, read the gate's way
PINNED_V = "v" + PIN["version"]
STALE_V = "v0.2.5"


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """<tmp>/caliby/{opt/{pyproject.toml,_build_backend.py,caliby_opt/,caliby_opt_autoload.pth},run.sh,configs/} and NO <tmp>/common/opt_core;
    a shadow older core; a `python` wrapper that runs this interpreter with -S; a fake upstream `caliby` whose body prints BODY RAN; a
    site dir holding the real .pth."""
    tmp = str(tmp_path_factory.mktemp("gate"))
    kit = os.path.join(tmp, "caliby")
    os.makedirs(os.path.join(kit, "opt"))
    shutil.copytree(os.path.join(OPT, "caliby_opt"), os.path.join(kit, "opt", "caliby_opt"), ignore=shutil.ignore_patterns("tests", "__pycache__"))
    for rel in ("opt/pyproject.toml", "opt/_build_backend.py", "opt/caliby_opt_autoload.pth", "run.sh"):
        shutil.copy(os.path.join(KIT, *rel.split("/")), os.path.join(kit, *rel.split("/")))
    shutil.copytree(os.path.join(KIT, "configs"), os.path.join(kit, "configs"))
    shadow = os.path.join(tmp, "shadow")
    os.makedirs(os.path.join(shadow, "opt_core"))
    open(os.path.join(shadow, "opt_core", "__init__.py"), "w").write('__version__ = "0.2.5"\n')
    bindir = os.path.join(tmp, "bin")
    os.makedirs(bindir)
    wrapper = os.path.join(bindir, "python")
    open(wrapper, "w").write(f'#!/bin/sh\nexec {sys.executable} -S "$@"\n')
    os.chmod(wrapper, 0o755)
    fake = os.path.join(tmp, "fake")
    os.makedirs(os.path.join(fake, "caliby"))
    open(os.path.join(fake, "caliby", "__init__.py"), "w").write("print('BODY RAN')\n")
    site_dir = os.path.join(tmp, "site")
    os.makedirs(site_dir)
    shutil.copy(os.path.join(OPT, "caliby_opt_autoload.pth"), site_dir)
    work = os.path.join(tmp, "work")                                                # the children's cwd: empty, so nothing there is importable by accident
    os.makedirs(work)
    assert not os.path.exists(os.path.join(tmp, "common")) and not os.path.exists(os.path.join(kit, "opt", "caliby_opt", "tests"))
    return {"tmp": tmp, "kit": kit, "opt": os.path.join(kit, "opt"), "shadow": shadow, "bin": bindir, "fake": fake, "site": site_dir, "work": work}


CORES = ["absent", "stale"]


def _paths(tree, core):
    """The kit's opt dir, with the broken core (if any) FIRST."""
    return {"absent": [], "stale": [tree["shadow"]]}[core] + [tree["opt"]]


def _env(tree, core, extra=None):
    """PYTHONPATH carries the kit (and the broken core first); PATH starts with the -S wrapper; no CALIBY_* / MODEL_OPT* of the caller leaks in."""
    env = {"PATH": tree["bin"] + os.pathsep + os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join(_paths(tree, core)), "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": tree["tmp"], "LANG": "C.UTF-8"}
    env.update(extra or {})
    return env


def _run(argv, tree, core, extra=None):
    return subprocess.run(argv, env=_env(tree, core, extra), capture_output=True, text=True, cwd=tree["work"], timeout=120)


def _assert_refused(r, core, marker=None):
    want = "NOT ACTIVE: reason=core_missing:opt_core" if core == "absent" else "NOT ACTIVE: reason=core_mismatch:"
    assert r.returncode == 3, (r.returncode, r.stderr[-800:], r.stdout[-300:])
    assert r.stderr.count("NOT ACTIVE") == 1 and want in r.stderr and r.stderr.count(f"[{stack.TAG}] NOT ACTIVE") == 1, r.stderr[-800:]
    assert PINNED_V in r.stderr, r.stderr[-800:]                                                            # the pin's version is named on every refusal
    if core == "stale":
        assert STALE_V in r.stderr, r.stderr[-800:]                                                          # the installed (older) version is named too
    assert "Traceback" not in r.stderr and "Error processing line" not in r.stderr, r.stderr[-800:]
    assert "BODY RAN" not in r.stdout, r.stdout[-300:]                                                        # the upstream package body never ran
    if marker:
        assert marker not in r.stdout, r.stdout[-300:]


# --------------------------------------------------------------------------------------------------------- python -m / console script
@pytest.mark.parametrize("core", CORES)
def test_python_dash_m(tree, core):
    _assert_refused(_run(["python", "-m", "caliby_opt", "check", "--mode", "fast", "--no-gpu"], tree, core), core)


@pytest.mark.parametrize("core", CORES)
def test_console_script_body(tree, core):
    """`caliby-opt` = the setuptools-written script: `from caliby_opt.__main__ import main; sys.exit(main())`."""
    r = _run(["python", "-c", "import re, sys; from caliby_opt.__main__ import main; sys.exit(main())", "check", "--mode", "fast", "--no-gpu"], tree, core)
    _assert_refused(r, core)


# --------------------------------------------------------------------------------------------------------- in-process entries
@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("call", ["k.enable('fast')", "k.enable('off')", "k.status()", "k.check('fast', need_gpu=False)", "k.check('off', need_gpu=False)"])
def test_in_process_entries(tree, core, call):
    _assert_refused(_run(["python", "-c", f"import caliby_opt as k; {call}; print('BODY RAN')"], tree, core), core, marker="BODY RAN")


# --------------------------------------------------------------------------------------------------------- the design children by hand
@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("argv", [["-m", "caliby_opt.design_run", "--mode", "fast", "--variant", "single", "--out_dir", "o_child", "--", "--inputs", "x.cif"],
                                  ["-s", "-m", "caliby_opt.stock_design", "--variant", "single", "--out_dir", "o_child", "--"]],
                         ids=["design_run", "stock_design"])
def test_design_children_started_by_hand(tree, core, argv):
    _assert_refused(_run(["python", *argv], tree, core), core)
    assert not os.path.exists(os.path.join(tree["work"], "o_child")), "the child wrote before the gate"


# --------------------------------------------------------------------------------------------------------- core-free routes
@pytest.mark.parametrize("mode_env", [{}, {"CALIBY_OPT": "fast"}, {"CALIBY_OPT": "off"}], ids=["unset", "fast", "off"])
def test_package_import_and_autoload_stay_core_free(tree, mode_env):
    """Importing the package and the .pth module (the finder installed under a set CALIBY_OPT, not fired) imports nothing of the core, no other
    module of the package, and never refuses: the gate is at the entries and at the trigger."""
    prog = ("import caliby_opt, caliby_opt._autoload, sys; "
            "assert 'opt_core' not in sys.modules and sorted(m for m in sys.modules if m.startswith('caliby_opt')) == ['caliby_opt', 'caliby_opt._autoload'], sorted(sys.modules); "
            "print('INERT')")
    r = _run(["python", "-c", prog], tree, "absent", mode_env)
    assert r.returncode == 0 and "INERT" in r.stdout and "NOT ACTIVE" not in r.stderr, (mode_env, r.returncode, r.stderr[-600:])


# --------------------------------------------------------------------------------------------------------- run.sh / configs/h100.env
@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("argv", [["check", "--mode", "fast"], ["check", "--mode", "off"], ["check"], ["design", "--mode", "off", "--input", "x.cif", "--out_dir", "o"],
                                  ["warm", "--mode", "fast"]],
                         ids=["check-fast", "check-off(stock route)", "check(no --mode)", "design-off(stock route)", "warm-fast"])
def test_run_sh(tree, core, argv):
    _assert_refused(_run(["bash", os.path.join(tree["kit"], "run.sh"), *argv], tree, core), core)


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("verb", ["check", "design"])
def test_run_sh_mode_from_the_environment(tree, core, verb):
    """`CALIBY_OPT=fast bash run.sh <verb>` (the mode from the environment, run.sh's other mode source): probe (1) stays inert under the set
    variable (the hook is lazy), probe (2) is the refusal."""
    argv = [verb] if verb == "check" else [verb, "--input", "x.cif", "--out_dir", "o_env"]
    _assert_refused(_run(["bash", os.path.join(tree["kit"], "run.sh"), *argv], tree, core, {"CALIBY_OPT": "fast"}), core)
    assert not os.path.exists(os.path.join(tree["work"], "o_env"))


@pytest.mark.parametrize("core", CORES)
def test_run_sh_with_config(tree, core):
    """`--config h100` sources configs/h100.env first: the gate there is the refusal (one line, rc 3), run.sh's own probes never run."""
    _assert_refused(_run(["bash", os.path.join(tree["kit"], "run.sh"), "check", "--config", "h100", "--mode", "fast"], tree, core), core)


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("mode_env", [{}, {"CALIBY_OPT": "fast"}], ids=["no-env-mode", "CALIBY_OPT=fast"])
def test_source_h100_env(tree, core, mode_env):
    r = _run(["bash", "-c", f"source {os.path.join(tree['kit'], 'configs', 'h100.env')} || exit $?; echo REACHED"], tree, core, mode_env)
    _assert_refused(r, core, marker="REACHED")


# --------------------------------------------------------------------------------------------------------- the real .pth at its trigger
def _pth_child(tree, core, mode_env):
    """The REAL generated .pth processed by site.addsitedir in a -S child: the guard line imports caliby_opt._autoload, which installs the finder
    under a set CALIBY_OPT; `import caliby` (the fake upstream, body = BODY RAN) is the trigger."""
    paths = _paths(tree, core) + [tree["fake"]]
    prog = f"import sys, site; sys.path[:0] = {paths!r}; site.addsitedir({tree['site']!r}); import caliby; print('stock would run here')"
    env = {"PATH": os.environ.get("PATH", ""), "HOME": tree["tmp"], "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(mode_env)
    return subprocess.run([sys.executable, "-S", "-c", prog], env=env, capture_output=True, text=True, cwd=tree["work"], timeout=120)


@pytest.mark.parametrize("core", CORES)
def test_real_pth_trigger_under_the_kit_mode(tree, core):
    _assert_refused(_pth_child(tree, core, {"CALIBY_OPT": "fast"}), core, marker="stock would run here")


@pytest.mark.parametrize("core", CORES)
def test_real_pth_off_and_unset_stay_inert_whatever_the_core(tree, core):
    for mode_env in ({"CALIBY_OPT": "off"}, {"CALIBY_OPT": ""}, {}):
        r = _pth_child(tree, core, mode_env)
        assert r.returncode == 0 and "BODY RAN" in r.stdout and "stock would run here" in r.stdout, (mode_env, r.returncode, r.stderr[-600:])
        assert "NOT ACTIVE" not in r.stderr and "Error processing line" not in r.stderr, (mode_env, r.stderr[-600:])


@pytest.mark.parametrize("core", CORES)
def test_real_pth_unknown_mode_refused_at_the_trigger_before_the_gate(tree, core):
    """An unknown CALIBY_OPT is not a kit mode: refused by name at the trigger (exit 3) with the mode-table fact, whatever the core."""
    r = _pth_child(tree, core, {"CALIBY_OPT": "turbo"})
    assert r.returncode == 3 and r.stderr.count("NOT ACTIVE") == 1 and f"[{stack.TAG}] NOT ACTIVE: unknown CALIBY_OPT='turbo'" in r.stderr, r.stderr[-600:]
    assert "BODY RAN" not in r.stdout and "stock would run here" not in r.stdout and "Traceback" not in r.stderr, (r.stdout[-300:], r.stderr[-600:])
