"""The env route (OPENDDE_OPT=<mode>, the .pth finder) and the stock route (mode off) in fresh interpreters: the finder fires after the
trigger package's body and activates on the kit's own shim; a refused or unknown mode never runs stock silently (exit 3 / NOT ACTIVE);
`off` installs no finder; the stock caller's proof refuses a process with a kit variable, a kit directory on the path or a kit module,
and passes a clean one.

RF-6 (env route only): run.sh refuses a mode named by OPENDDE_OPT and not on the command line unless the autoload hook is live in the
interpreter — two venvs on the same box: A (the editable install: the .pth lands in site, the hook imports at start) passes; B (the package
importable through PYTHONPATH only) is refused with the one line, exit 3, on the env route — and NOT on the --mode route, which activates
in-process (its ACTIVE/DRY line is the evidence). Mutation-checked: with the gate removed from run.sh, B's env-route `check` proceeds into
the package and the assertion on the gate's line fails. Needs the pinned upstream importable by the base interpreter (the box condition)."""
import json
import os
import shutil
import subprocess
import sys
import venv

import pytest

from opendde_opt import cli, stock_pred
from opendde_opt.tests import _stubs


@pytest.fixture
def site(tmp_path):
    return _stubs.make_site(str(tmp_path))


def test_pth_installs_the_finder_only_under_opendde_opt(site):
    code = "import sys; print([type(f).__name__ for f in sys.meta_path if type(f).__module__ == 'opt_core.autoload'], [getattr(f, 'mode', None) for f in sys.meta_path if type(f).__name__ == 'Finder'])"
    r = _stubs.run_py(code, env={"OPENDDE_OPT": "exact"}, pythonpath=[site])
    assert r.returncode == 0 and "['Finder'] ['exact']" in r.stdout, r                   # the core's finder on this kit's spec
    r = _stubs.run_py(code, env={"OPENDDE_OPT": " Exact "}, pythonpath=[site])
    assert r.returncode == 0 and "['Finder'] ['exact']" in r.stdout, r                   # the selection stripped and case-folded
    r = _stubs.run_py(code, env={"OPENDDE_OPT_LINE": "LSTAR2A"}, pythonpath=[site])
    assert r.returncode == 3 and "undeclared variable(s) OPENDDE_OPT_LINE" in r.stderr, r   # no line selector: a variable under the prefix the package does not read
    r = _stubs.run_py(code, env={"OPENDDE_OPT": "off"}, pythonpath=[site])
    assert r.returncode == 0 and "[]" in r.stdout
    r = _stubs.run_py(code, env={}, pythonpath=[site])
    assert r.returncode == 0 and "[]" in r.stdout
    code2 = "import sys; print(sorted(m for m in sys.modules if m == 'opt_core' or m.startswith('opt_core.')))"
    for env in ({}, {"OPENDDE_OPT": "off"}, {"OPENDDE_OPT": " OFF "}):
        r = _stubs.run_py(code2, env=env, pythonpath=[site])              # nothing of the core is imported when nothing is armed
        assert r.returncode == 0 and "[]" in r.stdout, (env, r)
    r = _stubs.run_py(code2, env={"OPENDDE_OPT": "exact"}, pythonpath=[site])   # armed: the finder's module alone
    assert r.returncode == 0 and "['opt_core', 'opt_core.autoload']" in r.stdout, r
    r = subprocess.run([sys.executable, "-S", "-c", "import sys; print('reached', 'opendde_opt' in sys.modules)"], capture_output=True, text=True,
                       env={**{k: v for k, v in os.environ.items() if not k.startswith("OPENDDE_OPT")}, "OPENDDE_OPT": "faster"})
    assert r.returncode == 0 and "reached False" in r.stdout and "NOT ACTIVE" not in r.stderr, r      # `python -S` processes no .pth: untouched
    r = _stubs.run_py(code, env={"OPENDDE_OPT": "faster"}, pythonpath=[site])          # an unknown selection: NOT ACTIVE line AND exit 3 at interpreter start
    assert r.returncode == 3 and "[]" not in r.stdout, r
    assert r.stderr.strip().splitlines()[-1] == "[opendde-opt] NOT ACTIVE: reason=unknown OPENDDE_OPT='faster' (expected off|exact|fast|big)"   # reference


def test_env_route_fires_after_the_trigger_and_arms_the_kit_shim(site):
    code = _stubs.ADMIT_ALL + """
import sys
assert 'odde_served_levers' not in sys.modules and 'torch' not in sys.modules
import runner                                   # the trigger: the finder fires after runner's own body
assert runner.SHIM_LOADED_DURING_BODY is False and 'odde_served_levers' in sys.modules, runner.SHIM_LOADED_DURING_BODY
import opendde_opt
st = opendde_opt.status(); assert st['active'], st
import runner.batch_inference as BI
assert getattr(BI.get_default_runner, '_served_levers_wrapped', False)
import os; print('OK', st['line'].split('(')[0], st['trigger'], os.environ['ODDE_ADDON_LEVERS'], sys.modules['odde_served_levers'].__version__)
"""
    r = _stubs.run_py(code, env={"OPENDDE_OPT": "exact"}, pythonpath=[site])
    assert r.returncode == 0, r.stderr
    assert "OK S1 runner dit_hoist,dit_align 0.2.0" in r.stdout
    assert "[opendde-opt] ACTIVE mode=exact line=S1(" in r.stderr and "[opendde-opt] EXIT pid=" in r.stderr


def test_env_route_refusal_exits_3_never_stock_silently(site):
    from opendde_opt import smalln
    code = _stubs.ADMIT_ALL + "import runner; print('reached')"
    r = _stubs.run_py(code, env={"OPENDDE_OPT": "fast", smalln.GATE: "small"}, pythonpath=[site])   # a selection the small-input floor refuses by name (a malformed variable): refused at the trigger
    assert r.returncode == 3 and "reached" not in r.stdout
    assert f"NOT ACTIVE mode=fast reason={smalln.GATE}='small'" in r.stderr, r.stderr[-600:]
    r = _stubs.run_py(_stubs.ADMIT_ALL + "import runner, opendde_opt; st = opendde_opt.status(); print('OK', st['active'], st['mode'], st['line'].split('(')[0])", env={"OPENDDE_OPT": "fast"}, pythonpath=[site])
    assert _stubs.verdict_ok(r, "OK True fast LSTAR"), ("interpreter teardown SIGSEGV after verdict (torch+triton CPU) is accepted only with the verdict line printed", r)


def test_unknown_mode_name_exits_3_with_the_not_active_line(site):
    """A mistyped OPENDDE_OPT is refused at the trigger with the package's own line and exit 3 — stock never runs under it."""
    r = _stubs.run_py("import runner; print('reached')", env={"OPENDDE_OPT": "bogus"}, pythonpath=[site])
    assert r.returncode == 3 and "reached" not in r.stdout and "NOT ACTIVE: reason=unknown OPENDDE_OPT='bogus'" in r.stderr, r


def test_undeclared_variable_under_the_prefix_exits_3(site):
    """OPENDDE_OPT_MODE=exact (a mistyped name) is refused at interpreter start with the package's line, exit 3 — never ignored."""
    r = _stubs.run_py("print('reached')", env={"OPENDDE_OPT_MODE": "exact"}, pythonpath=[site])
    assert r.returncode == 3 and "reached" not in r.stdout, r
    assert r.stderr.strip().splitlines()[-1] == "[opendde-opt] NOT ACTIVE: reason=undeclared variable(s) OPENDDE_OPT_MODE (the package reads OPENDDE_OPT)"   # reference
    from opendde_opt import _autoload, stack
    assert set(_autoload.DECLARED) == {"OPENDDE_OPT"}                                            # the declared set restates every reader
    r = _stubs.run_py("print('reached')", env={"OPENDDE_OPT_ROUTE": "cli"}, pythonpath=[site])       # no route variable exists: refused like any undeclared name
    assert r.returncode == 3 and "undeclared variable(s) OPENDDE_OPT_ROUTE" in r.stderr, r


def test_find_spec_probe_does_not_disarm_the_finder(site):
    """importlib.util.find_spec(<trigger>) before the real import (a probe, no module executed) leaves the hook armed: the import that
    follows still fires and arms the shim."""
    code = (_stubs.ADMIT_ALL + "import importlib.util, sys; sp = importlib.util.find_spec('runner'); assert sp is not None; "
            "assert 'runner' not in sys.modules; import runner, opendde_opt; st = opendde_opt.status(); print('FIRED', st['active'], st['mode'])")
    r = _stubs.run_py(code, env={"OPENDDE_OPT": "exact"}, pythonpath=[site])
    assert r.returncode == 0 and "FIRED True exact" in r.stdout, r
    r = _stubs.run_py("import importlib.util; importlib.util.find_spec('runner'); import runner; print('reached')", env={"OPENDDE_OPT": "exact_typo"}, pythonpath=[site])
    assert r.returncode == 3 and "reached" not in r.stdout and "NOT ACTIVE" in r.stderr, r   # a refused selection is still refused after a probe


def test_generic_runner_package_without_opendde_beside_it_is_not_a_trigger(site, tmp_path):
    other = tmp_path / "other"
    (other / "runner").mkdir(parents=True)
    (other / "runner" / "__init__.py").write_text("")
    code = "import runner, opendde_opt; print(opendde_opt.status()['active'])"
    r = _stubs.run_py(code, env={"OPENDDE_OPT": "exact"}, pythonpath=[str(other)])
    assert r.returncode == 0 and "False" in r.stdout, r


def test_env_proof_refuses_kit_traces_and_passes_clean(tmp_path):
    kit_dir = os.path.join(_stubs.TREE, "opt", "forward", "fast_inference", "levers", "ACCEL")
    p = stock_pred.env_proof(["ODDE_", "OPENDDE_OPT"], [kit_dir], environ={"ODDE_ARM_Z": "1", "HOME": "/x"}, modules={}, path=["/usr/lib"], meta_path=[])
    assert not p["ok"] and p["forbidden_present"] == ["ODDE_ARM_Z"]
    p = stock_pred.env_proof(["ODDE_"], [kit_dir], environ={}, modules={}, path=[kit_dir], meta_path=[])
    assert not p["ok"] and p["kit_dirs_on_path"] == [kit_dir]
    import types
    m = types.ModuleType("odde_served_levers"); m.__file__ = os.path.join(kit_dir, "odde_served_levers.py")
    p = stock_pred.env_proof(["ODDE_"], [kit_dir], environ={}, modules={"odde_served_levers": m}, path=[], meta_path=[])
    assert not p["ok"] and p["kit_modules_loaded"] == ["odde_served_levers"]
    p = stock_pred.env_proof(["ODDE_"], [kit_dir], ["OPENDDE_ROOT_DIR"], environ={"OPENDDE_ROOT_DIR": "/models/opendde"}, modules={}, path=[], meta_path=[])
    assert p["ok"] and p["reads"] == {"OPENDDE_ROOT_DIR": "/models/opendde"} and not p["torch_loaded_before_proof"]


def test_stock_command_strips_switches_and_kit_dirs(monkeypatch):
    monkeypatch.setenv("MODEL_OPT", _stubs.TREE)
    monkeypatch.setenv("ODDE_ARM_Z", "1"); monkeypatch.setenv("OPENDDE_OPT", "exact"); monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("OPENDDE_ROOT_DIR", "/models/opendde")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([os.path.join(_stubs.TREE, "opt", "forward", "fast_inference", "levers", "ARMT"), "/keep/me"]))
    cmd, env = cli.stock_command(_stubs.TREE, ["pred", "-i", "q.json"], "/tmp/proof.json")
    assert cmd[:4] == [sys.executable, "-s", "-m", "opendde_opt.stock_pred"] and cmd[-4:] == ["--", "pred", "-i", "q.json"]
    assert "ODDE_ARM_Z" not in env and "OPENDDE_OPT" not in env and "CUBLAS_WORKSPACE_CONFIG" not in env
    assert env["OPENDDE_ROOT_DIR"] == "/models/opendde" and env["PYTHONPATH"] == "/keep/me"
    absent = cmd[cmd.index("--env-absent") + 1].split(",")
    assert "ODDE_" in absent and "CUBLAS_WORKSPACE_CONFIG" in absent and "OPENDDE_OPT" in absent
    cmd_det, _ = cli.stock_command(_stubs.TREE, ["pred"], "/tmp/proof.json", det_level=1)
    assert "CUBLAS_WORKSPACE_CONFIG" not in cmd_det[cmd_det.index("--env-absent") + 1].split(",")


def test_stock_caller_runs_the_stub_cli_and_writes_the_proof(site, tmp_path):
    proof = tmp_path / "stock_env_proof.json"
    kit_dirs = os.pathsep.join(os.path.join(_stubs.TREE, "opt", d) for d in ("forward/fast_inference/levers/ACCEL", "forward/fast_inference/levers/ARMT"))
    argv = ["-s", "-m", "opendde_opt.stock_pred", "--proof-json", str(proof), "--env-absent", "ODDE_,OPENDDE_OPT", "--env-reads", "OPENDDE_ROOT_DIR",
            "--kit-dirs", kit_dirs, "--", "pred", "-i", "x.json"]
    # the stub's opendde_cli (runner/cli.py, the console entry) is a plain function without .main: patch it in the stub site to a click-less callable recording its args
    (tmp_path / "site" / "runner" / "cli.py").write_text(
        "class _E:\n    def main(self, args=None, prog_name=None, standalone_mode=True):\n        open('stock_called.txt','w').write(' '.join(args)); raise SystemExit(0)\n"
        "opendde_cli = _E()\n")
    r = _stubs.run_py("import runpy, sys; sys.argv = ['stock_pred'] + %r; runpy.run_module('opendde_opt.stock_pred', run_name='__main__')" % argv[3:],
                      env={"OPENDDE_ROOT_DIR": "/models/opendde"}, cwd=str(tmp_path), pythonpath=[site])
    assert r.returncode == 0, r.stderr
    p = json.loads(proof.read_text())
    assert p["ok"] and p["reads"] == {"OPENDDE_ROOT_DIR": "/models/opendde"} and p["kit_modules_loaded_after"] == [] and p["opendde_version"] == _stubs.PINNED
    assert (tmp_path / "stock_called.txt").read_text() == "pred -i x.json"
    assert "[opendde-opt stock] ENV-CLEAN ok" in r.stderr
    # a kit variable in the environment: the proof fails and nothing runs (exit 3)
    (tmp_path / "stock_called.txt").unlink()
    r = _stubs.run_py("import runpy, sys; sys.argv = ['stock_pred'] + %r; runpy.run_module('opendde_opt.stock_pred', run_name='__main__')" % argv[3:],
                      env={"ODDE_ARM_Z": "1"}, cwd=str(tmp_path), pythonpath=[site])
    assert r.returncode == 3 and "NOT STOCK" in r.stderr and not (tmp_path / "stock_called.txt").exists()


HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))            # opendde/
CORE = os.path.abspath(os.path.join(KIT, "..", "common", "opt_core"))
GATE_LINE = "needs the autoload hook, which is not live in"
STALE_LINE = "would run a stale copy"
BASE_BIN = os.path.join(sys.base_prefix, "bin")   # the base interpreter (never a venv): its user site is enabled — a venv disables the user site by Python's rule


def _base_has_upstream():
    return subprocess.run([sys.executable, "-I", os.path.join(KIT, "stock", "check_pins.py"), "--quiet"], capture_output=True).returncode == 0


def _run(venv_dir, *args, env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENDDE_OPT", "ODDE_", "FPF_", "PYTHONPATH", "VIRTUAL_ENV", "PYTHONUSERBASE"))}
    env["PATH"] = (venv_dir if venv_dir == BASE_BIN else os.path.join(venv_dir, "bin")) + os.pathsep + env.get("PATH", "")
    if venv_dir != BASE_BIN:
        env["VIRTUAL_ENV"] = venv_dir
    env.update(env_extra or {})
    return subprocess.run(["bash", os.path.join(KIT, "run.sh"), *args], cwd=KIT, env=env, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def venvs(tmp_path_factory):
    if not _base_has_upstream():
        pytest.skip("the pinned upstream is not importable by this interpreter (stock/check_pins.py): the box condition")
    if subprocess.run([sys.executable, "-c", "import wheel"], capture_output=True).returncode != 0:
        pytest.skip("`wheel` is absent from the base interpreter: the editable installs of the fixture need bdist_wheel (the box form installs wheel first)")
    base_sites = subprocess.run([os.path.join(BASE_BIN, "python"), "-c", "import site; print(chr(10).join(site.getsitepackages()))"], capture_output=True, text=True).stdout.split()
    if any(os.path.isfile(os.path.join(d, "opendde_opt_autoload.pth")) for d in base_sites):
        pytest.skip("the base interpreter's site carries opendde_opt_autoload.pth (the kit is installed editable there): venv B inherits it through "
                    "system site packages, so RF-6's 'importable but no live hook' condition cannot be built on this box")
    root = tmp_path_factory.mktemp("rf6")
    a, b = str(root / "A"), str(root / "B")
    for d in (a, b):
        venv.EnvBuilder(system_site_packages=True, with_pip=False, symlinks=True).create(d)   # the system site carries the upstream + setuptools
    pip = [os.path.join(a, "bin", "python"), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "-e", CORE, "-e", os.path.join(KIT, "opt")]
    r = subprocess.run(pip, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-800:]
    site_a = subprocess.run([os.path.join(a, "bin", "python"), "-c", "import site; print(site.getsitepackages()[0])"], capture_output=True, text=True).stdout.strip()
    assert os.path.isfile(os.path.join(site_a, "opendde_opt_autoload.pth")), sorted(os.listdir(site_a))[:20]
    return a, b


def test_venv_a_with_the_pth_passes_the_env_route(venvs):
    a, _ = venvs
    r = _run(a, "check", "--config", "h100", env_extra={"OPENDDE_OPT": "exact"})
    assert GATE_LINE not in r.stderr and r.returncode != 3, (r.returncode, r.stderr[-600:])


PP = {"PYTHONPATH": os.path.join(KIT, "opt") + os.pathsep + CORE}


def test_venv_b_without_the_pth_is_refused_on_the_env_route(venvs):
    _, b = venvs
    for mode in ("exact", "fast", "big"):
        r = _run(b, "check", "--config", "h100", env_extra={**PP, "OPENDDE_OPT": mode})
        assert r.returncode == 3 and GATE_LINE in r.stderr and "pip install -e" in r.stderr and r.stderr.count("run.sh:") == 1, (mode, r.returncode, r.stderr[-600:])
        assert "present at " + os.path.join(KIT, "opt") + " but not processed" in r.stderr, r.stderr[-600:]     # the diagnostic: opt/ carries the .pth; a PYTHONPATH entry is no site dir
    r = _run(b, "pred", "-i", "q.json", "-o", "o", "--config", "h100", env_extra={**PP, "OPENDDE_OPT": "fast"})
    assert r.returncode == 3 and GATE_LINE in r.stderr, (r.returncode, r.stderr[-400:])


def test_venv_b_on_the_mode_route_is_not_gated(venvs):
    """--mode names the mode on the command line: the package activates in-process (its own DRY/ACTIVE line), no hook needed."""
    _, b = venvs
    r = _run(b, "check", "--mode", "exact", "--config", "h100", env_extra=PP)
    assert GATE_LINE not in r.stderr and r.returncode != 3 and "[opendde-opt] DRY RUN" in r.stdout + r.stderr, (r.returncode, (r.stdout + r.stderr)[-600:])
    r = _run(b, "check", "--mode", "fast", "--config", "h100", env_extra={**PP, "OPENDDE_OPT": "fast"})                 # both given and agreeing: the command line names it
    assert GATE_LINE not in r.stderr and r.returncode != 3, (r.returncode, r.stderr[-400:])


def test_the_diagnostic_names_an_absent_pth(venvs):
    _, b = venvs
    kit_copy = os.path.join(os.path.dirname(b), "kitcopy")                 # the package importable from a copy of opt/ WITHOUT its .pth: absent everywhere
    if not os.path.isdir(kit_copy):
        shutil.copytree(os.path.join(KIT, "opt", "opendde_opt"), os.path.join(kit_copy, "opendde_opt"), symlinks=True)
    r = _run(b, "check", "--config", "h100", env_extra={"PYTHONPATH": kit_copy + os.pathsep + CORE, "OPENDDE_OPT": "exact"})
    # a bare copy of the package has no opt/pyproject.toml above it: the core gate (statement one of `python -m opendde_opt`, which the config
    # runs before anything) refuses it by name before run.sh's hook gate is reached — either line is a refusal by name, exit 3, never stock
    assert r.returncode == 3 and (GATE_LINE in r.stderr or "NOT ACTIVE: reason=core_pin_unreadable" in r.stderr), (r.returncode, r.stderr[-600:])


def test_the_diagnostic_names_a_stale_copy(venvs, tmp_path):
    """venv C: the editable install made from a COPY of the tree — the hook is live, but from the copy: this tree's run.sh refuses the env
    route as 'a stale copy' (the live hook imports opendde_opt from another tree). With the copy removed the import itself fails and run.sh's
    config's package check refuses first (the real entry: 'not usable on …', rc 3)."""
    if not _base_has_upstream():
        pytest.skip("box condition")
    c = str(tmp_path / "C"); copy = tmp_path / "tree"
    shutil.copytree(os.path.join(KIT, "opt"), copy / "opt", symlinks=True); shutil.copytree(CORE, copy / "core", symlinks=True)
    venv.EnvBuilder(system_site_packages=True, with_pip=False, symlinks=True).create(c)
    r = subprocess.run([os.path.join(c, "bin", "python"), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "-e", str(copy / "core"), "-e", str(copy / "opt")], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-800:]
    r = _run(c, "check", "--config", "h100", env_extra={"OPENDDE_OPT": "exact"})
    assert r.returncode == 3 and STALE_LINE in r.stderr and str(copy / "opt") in r.stderr, (r.returncode, r.stderr[-600:])
    r = _run(c, "check", "--mode", "exact", "--config", "h100")                        # the --mode route is not gated: this tree's package runs (PATH's python has the copy's, the route imports opendde_opt from it — the DRY line is the copy's own)
    assert STALE_LINE not in r.stderr and GATE_LINE not in r.stderr and r.returncode != 3, (r.returncode, r.stderr[-400:])
    shutil.rmtree(copy)
    r = _run(c, "check", "--config", "h100", env_extra={"OPENDDE_OPT": "exact"})
    assert r.returncode == 3 and "is not usable on" in r.stderr and GATE_LINE not in r.stderr, (r.returncode, r.stderr[-400:])   # configs/h100.env's package check (the real entry `python -m opendde_opt help`, rc 3) fires before the hook gate


def test_a_user_site_install_passes_the_env_route(venvs, tmp_path):
    """The BASE interpreter (a venv disables the user site by Python's rule) with the package installed `pip install --user` under a
    PYTHONUSERBASE of this test: the route's `python` processes that .pth, the gate passes — the probe runs the route's interpreter form,
    never `python -I`. The same copy with the user site disabled (PYTHONNOUSERSITE=1) beside a PYTHONPATH copy is refused, the clause naming
    the disabled user site; without the PYTHONPATH copy the config's import check refuses first."""
    if not _base_has_upstream():
        pytest.skip("box condition")
    ub = str(tmp_path / "userbase")
    r = subprocess.run([os.path.join(BASE_BIN, "python"), "-m", "pip", "install", "--user", "--no-deps", "--no-build-isolation", "-q", "-e", CORE, "-e", os.path.join(KIT, "opt")],
                       env={**os.environ, "PYTHONUSERBASE": ub}, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-800:]
    r = _run(BASE_BIN, "check", "--config", "h100", env_extra={"OPENDDE_OPT": "exact", "PYTHONUSERBASE": ub})
    assert r.returncode == 0 and GATE_LINE not in r.stderr and STALE_LINE not in r.stderr, (r.returncode, r.stderr[-600:])
    r = _run(BASE_BIN, "check", "--config", "h100", env_extra={"OPENDDE_OPT": "exact", "PYTHONUSERBASE": ub, "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.path.join(KIT, "opt") + os.pathsep + CORE})
    assert r.returncode == 3 and GATE_LINE in r.stderr and "the user site is disabled" in r.stderr, (r.returncode, r.stderr[-600:])   # importable through PYTHONPATH (the import check passes), the only .pth in a user site this run disables: the clause names it
    r = _run(BASE_BIN, "check", "--config", "h100", env_extra={"OPENDDE_OPT": "exact", "PYTHONUSERBASE": ub, "PYTHONNOUSERSITE": "1"})
    assert r.returncode == 3 and "is not usable on" in r.stderr, (r.returncode, r.stderr[-600:])          # without the PYTHONPATH copy nothing is importable: the config's package check (the real entry, rc 3) refuses first


def test_the_stock_route_is_exempt(venvs):
    _, b = venvs
    for env in ({**PP, "OPENDDE_OPT": "off"}, PP):                                                         # the stock route by the variable or by default: no gate
        r = _run(b, "pred", "--config", "h100", env_extra=env)                                           # no -i/-o: the package's usage error, not the gate
        assert GATE_LINE not in r.stderr and r.returncode != 3, (env, r.returncode, r.stderr[-400:])
