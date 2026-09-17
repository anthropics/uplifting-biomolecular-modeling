"""RF-6 (env route only, hook-live): with no --mode and PROTENIX_OPT naming a non-off mode, run.sh refuses by name (exit 3, one line) unless
the start-up hook `protenix_opt._autoload` is live in a fresh `python` (the route's own interpreter, no -I: a user-site install is processed when the user site is enabled) — the env route (`PROTENIX_OPT=<mode>` + the stock CLI, every child
of `pred`) would run stock silently on a package that is merely importable. Two venvs: A = the kit installed (`pip install -e opt`: the .pth
at the wheel root lands in site → the hook is live); B = the package importable through PYTHONPATH only, no .pth in site. The --mode route
is not gated in either (it activates in-process); --mode off / PROTENIX_OPT=off exempt; a .pth copied beside a PYTHONPATH entry is never
processed → refused with the 'copy on PYTHONPATH' diagnostic; a broken .pth in a site → 'present but not processed'. Mutation-checked by
construction: with the gate removed, venv B's env route runs on to `python -m protenix_opt` (no gate line, another exit) and the refusal
tests fail."""
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from conftest import HAS_DURABLE_OPT_CORE

HERE = Path(__file__).resolve().parent.parent          # protenix_v2/
RUN = HERE / "run.sh"; OPT = HERE / "opt"; PTH = "protenix_opt_autoload.pth"
GATE_LINE = "NOT ACTIVE: the env route (PROTENIX_OPT="


def _venv(tmp_path, name):
    d = tmp_path / name; venv.EnvBuilder(system_site_packages=True, with_pip=False, symlinks=True).create(d)
    py = d / "bin" / "python"; site = subprocess.run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True, check=True).stdout.strip()
    return d, py, Path(site)


def _run(py_dir, cmd, extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_OPT") and k != "PYTHONPATH"}
    env["PATH"] = f"{py_dir / 'bin'}:{env.get('PATH', '')}"; env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PROTENIX_ROOT_DIR", str(py_dir / "weights"))       # configs/h100.env requires the weights root by name; nothing here reads weights
    env.update(extra_env or {})
    r = subprocess.run(["bash", str(RUN), *cmd], env=env, capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout + r.stderr


def _hook_live(py, extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_OPT") and k != "PYTHONPATH"}; env.update(extra_env or {})
    r = subprocess.run([str(py), "-c", "import sys; print('protenix_opt._autoload' in sys.modules)"], env=env, capture_output=True, text=True)
    return r.stdout.strip() == "True"


@pytest.fixture(scope="module")
def venvs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rf6")
    a, py_a, site_a = _venv(tmp, "A"); b, py_b, site_b = _venv(tmp, "B")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "--prefix", str(a), "-e", str(OPT)], capture_output=True, text=True)
    if r.returncode != 0 or not (site_a / PTH).is_file():          # a pip that cannot build the editable wheel here: the two artifacts by hand
        site_a.mkdir(parents=True, exist_ok=True); (site_a / PTH).write_text((OPT / PTH).read_text()); (site_a / "protenix_opt_tree.pth").write_text(str(OPT) + "\n")
    assert (site_a / PTH).is_file() and _hook_live(py_a), "venv A must carry the kit's .pth in site and the hook must be live"
    assert not (site_b / PTH).exists() and not _hook_live(py_b)
    return {"A": (a, py_a, site_a), "B": (b, py_b, site_b)}


def test_venv_A_env_route_passes_the_gate(venvs):
    a, py_a, site_a = venvs["A"]
    rc, out = _run(a, ["check", "--config", "h100"], {"PROTENIX_OPT": "exact"})
    assert GATE_LINE not in out, out[-800:]


def test_venv_B_env_route_refuses_by_name(venvs):
    b, py_b, site_b = venvs["B"]
    for verb in ("pred", "check", "warm"):
        rc, out = _run(b, [verb, "--config", "h100"], {"PROTENIX_OPT": "exact", "PYTHONPATH": str(OPT)})
        lines = [l for l in out.splitlines() if GATE_LINE in l]
        assert rc == 3 and len(lines) == 1, (verb, rc, out[-800:])
        assert "absent from the searched sites" in lines[0] and str(site_b) in lines[0] and "pip install -e" in lines[0]


def test_venv_B_mode_route_is_not_gated(venvs):
    b, py_b, site_b = venvs["B"]
    rc, out = _run(b, ["check", "--config", "h100", "--mode", "exact"], {"PYTHONPATH": str(OPT)})
    assert GATE_LINE not in out, out[-800:]                           # the --mode route activates in-process: never this gate
    if subprocess.run([str(py_b), "-c", "import protenix, torch"], capture_output=True).returncode == 0:   # a host with the stack: the package's own line
        assert "[protenix-opt] ACTIVE" in out or "[protenix-opt] CHECK" in out, out[-800:]


def test_off_is_exempt_in_venv_B(venvs):
    b, py_b, site_b = venvs["B"]
    rc, out = _run(b, ["check", "--config", "h100"], {"PROTENIX_OPT": "off", "PYTHONPATH": str(OPT)})
    assert GATE_LINE not in out, out[-800:]


def test_copy_on_pythonpath_is_refused_as_never_processed(venvs, tmp_path):
    """A copy of the package on PYTHONPATH with its pin beside it (as in a checkout's opt/): the RF-6 line names the never-processed .pth.
    The same copy WITHOUT its pyproject.toml is refused first by the package's core pin gate (core_pin_unreadable), also exit 3 by name."""
    b, py_b, site_b = venvs["B"]
    d = tmp_path / "pp"; d.mkdir(); (d / PTH).write_text((OPT / PTH).read_text()); (d / "protenix_opt").symlink_to(OPT / "protenix_opt")
    rc, out = _run(b, ["pred", "--config", "h100"], {"PROTENIX_OPT": "exact", "PYTHONPATH": str(d)})
    assert rc == 3 and "NOT ACTIVE: reason=core_pin_unreadable: no pyproject.toml with [tool.opt_core] at or above " + str(d / "protenix_opt") in out, out[-800:]
    if not HAS_DURABLE_OPT_CORE:
        pytest.skip("opt_core is reachable only via this process's own PYTHONPATH, not a durable site install; configs/h100.env's "
                    "own stack-key probe needs the real thing before this scenario's RF-6 line is ever reached")
    (d / "pyproject.toml").symlink_to(OPT / "pyproject.toml")
    rc, out = _run(b, ["pred", "--config", "h100"], {"PROTENIX_OPT": "exact", "PYTHONPATH": str(d)})
    lines = [l for l in out.splitlines() if GATE_LINE in l]
    assert rc == 3 and len(lines) == 1 and "a copy on PYTHONPATH at" in lines[0] and str(d / PTH) in lines[0], out[-800:]


def test_broken_pth_in_site_is_refused_as_present_but_not_processed(tmp_path):
    c, py_c, site_c = _venv(tmp_path, "C")
    site_c.mkdir(parents=True, exist_ok=True); (site_c / PTH).write_text("import protenix_opt_no_such_module_autoload\n")   # a stale copy: site.py swallows its error
    rc, out = _run(c, ["pred", "--config", "h100"], {"PROTENIX_OPT": "exact", "PYTHONPATH": str(OPT)})
    lines = [l for l in out.splitlines() if GATE_LINE in l]
    assert rc == 3 and len(lines) == 1 and "present but not processed" in lines[0] and str(site_c / PTH) in lines[0], out[-800:]


def test_user_site_install_passes_the_gate(tmp_path):
    """A `pip install --user` form: the .pth in the USER site (PYTHONUSERBASE) is processed by the route's `python` when the user site is
    enabled — the gate passes (a `python -I` probe would have over-refused it). Skipped by name on an interpreter whose user site is disabled."""
    py = Path(sys.executable)
    ub = tmp_path / "userbase"; env0 = {"PYTHONUSERBASE": str(ub), "PYTHONPATH": str(OPT)}
    r = subprocess.run([str(py), "-c", "import site, json; print(json.dumps([site.ENABLE_USER_SITE, site.getusersitepackages()]))"], env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, **env0}, capture_output=True, text=True)
    enabled, usp = json.loads(r.stdout.strip())
    if not enabled:
        pytest.skip(f"{py}: the user site is disabled for this interpreter (a venv) — the user-site form does not exist here")
    Path(usp).mkdir(parents=True, exist_ok=True); (Path(usp) / PTH).write_text((OPT / PTH).read_text())
    bindir = tmp_path / "bin"; bindir.mkdir(); (bindir / "python").symlink_to(py)          # run.sh calls `python`: this interpreter, first on PATH
    assert _hook_live(py, env0), "the user-site .pth must make the hook live in the route's own python"
    rc, out = _run(tmp_path, ["check", "--config", "h100"], {"PROTENIX_OPT": "exact", **env0})
    assert GATE_LINE not in out, out[-800:]


def test_exported_selection_does_not_kill_the_presence_probes_A13(venvs, tmp_path):
    """A13: with the hook LIVE (venv A: the installed .pth is processed at every interpreter start) and PROTENIX_OPT already exported,
    run.sh's and configs/h100.env's own presence probes (`python -c … find_spec('protenix_opt') …`) must not be killed by the start-up
    gate: against a core that is not the pinned one the route ends with the gate's by-name line (reason=core_mismatch, exit 3) from the
    real entry — never the probes' false 'protenix_opt is not installed' (exit 2) with the gate's line swallowed by their 2>/dev/null."""
    a, py_a, site_a = venvs["A"]
    shadow = tmp_path / "shadow" / "opt_core"; shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text('__version__ = "0.0.0-shadow"\n')                    # importable ahead of the installed core, not the pin
    for argv in (["pred", "--config", "h100", "--input", "x", "--out_dir", str(tmp_path / "o")], ["check", "--config", "h100"], ["check"]):
        rc, out = _run(a, argv, {"PROTENIX_OPT": "exact", "PYTHONPATH": str(tmp_path / "shadow")})
        assert "is not installed" not in out, (argv, rc, out[-600:])
        assert rc == 3 and "[protenix-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned " in out and "0.0.0-shadow" in out, (argv, rc, out[-600:])
