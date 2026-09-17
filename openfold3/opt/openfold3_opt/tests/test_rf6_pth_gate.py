"""RF6 (env route only, hook-live): the env route (OPENFOLD3_OPT=<mode>, no --mode) hooks through the kit's .pth as site.py processes
it; `run.sh` on that route asks a FRESH interpreter (`python -I`) whether `openfold3_opt._autoload` is in sys.modules and refuses by name
when it is not, naming the diagnostic; the --mode route is ungated (activation by construction). Venvs
built without system site-packages: A = the package installed (its .pth and a path file for opt/ and the core in site-packages) passes the
gate; B = importable through PYTHONPATH only (the shipped .pth sits beside that entry, which site.py never processes) is refused 'present
but not processed'; C = importable through PYTHONPATH from a copy of opt/ without the .pth is refused 'absent from the searched sites';
D = the .pth in site with no path file (site.py runs it, the import fails) is refused 'present but not processed'; venv B on the --mode
route is not refused by this gate; `off` under the variable is exempt. Removing the gate from run.sh makes the B/C/D tests fail (the
mutation check)."""
import os
import shutil
import subprocess
import sys
import tempfile
import venv

import pytest

from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()
PTH = "openfold3_opt_autoload.pth"
GATE_LINE = "the kit's hook is not live at interpreter start"


def _venv(root, with_pth: bool, with_paths: bool):
    venv.EnvBuilder(system_site_packages=False, with_pip=False, symlinks=True).create(root)
    py = os.path.join(root, "bin", "python")
    sp = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True, check=True).stdout.strip()
    os.makedirs(sp, exist_ok=True)
    if with_pth:
        shutil.copyfile(os.path.join(HOME, "opt", PTH), os.path.join(sp, PTH))
    if with_paths:
        with open(os.path.join(sp, "__editable__.openfold3_opt.pth"), "w") as fh:     # an editable install's path entries: the package and the core (sorted before the hook's .pth)
            fh.write(os.path.join(HOME, "opt") + "\n" + os.path.join(HOME, "..", "common", "opt_core") + "\n")
    return py


def _run(py, extra_env, mode="fast", route="env"):
    """`run.sh check` under the venv's python: route "env" = OPENFOLD3_OPT=<mode> with no --mode (the gated route); "arg" = --mode <mode>."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENFOLD3_OPT", "OF3", "PYTHONPATH")) and k != "PYTHONNOUSERSITE"}   # the user site as the route's python sees it
    env["PATH"] = os.path.dirname(py) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTHONUSERBASE", EMPTY_USERBASE)                                # hermetic: no stray ~/.local hook unless the case sets its own
    env.update(extra_env)
    argv = ["bash", os.path.join(HOME, "run.sh"), "check"]
    if route == "env":
        env["OPENFOLD3_OPT"] = mode
    else:
        argv += ["--mode", mode]
    return subprocess.run(argv, env=env, capture_output=True, text=True, cwd=HOME)


PYTHONPATH_OPT = os.path.join(HOME, "opt") + os.pathsep + os.path.join(HOME, "..", "common", "opt_core")
EMPTY_USERBASE = tempfile.mkdtemp(prefix="rf6-empty-userbase-")


@pytest.fixture(scope="module")
def venvs(tmp_path_factory):
    root = tmp_path_factory.mktemp("rf6")
    nopth = str(root / "opt_without_pth")                                            # a copy of opt/ without the .pth beside it (case C)
    shutil.copytree(os.path.join(HOME, "opt"), nopth, ignore=shutil.ignore_patterns("*.pth", "__pycache__"))
    return {"A": _venv(str(root / "A"), True, True), "B": _venv(str(root / "B"), False, False), "D": _venv(str(root / "D"), True, False), "nopth": nopth}


def test_venv_A_with_the_pth_passes_the_gate(venvs):
    r = _run(venvs["A"], {})
    assert GATE_LINE not in r.stderr, r.stderr[-800:]                              # past the gate: whatever the check then says is the package's


def test_venv_B_pth_beside_pythonpath_is_refused_present_but_not_processed(venvs):
    r = _run(venvs["B"], {"PYTHONPATH": PYTHONPATH_OPT})
    assert r.returncode == 3 and GATE_LINE in r.stderr and "present but not processed" in r.stderr and "sits beside a PYTHONPATH entry" in r.stderr, (r.returncode, r.stderr[-900:])
    assert r.stderr.count("[openfold3-opt] NOT ACTIVE:") == 1


def test_venv_C_no_pth_anywhere_is_refused_absent(venvs):
    r = _run(venvs["B"], {"PYTHONPATH": venvs["nopth"] + os.pathsep + os.path.join(HOME, "..", "common", "opt_core")})
    assert r.returncode == 3 and GATE_LINE in r.stderr and "absent from the searched sites" in r.stderr, (r.returncode, r.stderr[-900:])


def test_venv_D_pth_in_site_unprocessed_is_refused_present_but_not_processed(venvs):
    """The .pth is in site but its import cannot run (no path file: the package is not importable from that site) — present, not live."""
    r = _run(venvs["D"], {})
    assert r.returncode == 3 and GATE_LINE in r.stderr and "present but not processed" in r.stderr and "is in site but its import did not run" in r.stderr, (r.returncode, r.stderr[-900:])


def test_venv_B_mode_route_is_not_gated(venvs):
    """`run.sh check --mode fast` in venv B: the --mode route activates by construction (python -m openfold3_opt) — this gate never fires;
    whatever refuses afterwards is the package's or the pin check's, by its own name."""
    r = _run(venvs["B"], {"PYTHONPATH": PYTHONPATH_OPT}, route="arg")
    assert GATE_LINE not in r.stderr, r.stderr[-600:]
    assert r.returncode == 0 or "is not installed at its pin" in r.stderr, (r.returncode, r.stderr[-600:])   # the route went past this gate: the next word is the pin check's (this venv carries no upstream)


def test_venv_B_off_under_the_variable_is_exempt(venvs):
    """OPENFOLD3_OPT=off is the stock route (no hook): the gate does not fire; the stock caller's own gates decide."""
    r = _run(venvs["B"], {"PYTHONPATH": PYTHONPATH_OPT}, mode="off")
    assert GATE_LINE not in r.stderr, r.stderr[-600:]


def _user_site(root):
    """A PYTHONUSERBASE tree carrying the kit's hook (.pth) and the editable path file in its site-packages — a `pip install --user` layout."""
    v = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sp = os.path.join(root, "lib", v, "site-packages"); os.makedirs(sp, exist_ok=True)
    shutil.copyfile(os.path.join(HOME, "opt", PTH), os.path.join(sp, PTH))
    with open(os.path.join(sp, "__editable__.openfold3_opt.pth"), "w") as fh:
        fh.write(os.path.join(HOME, "opt") + "\n" + os.path.join(HOME, "..", "common", "opt_core") + "\n")
    return root


def test_venv_E_user_site_install_passes_the_gate(venvs, tmp_path):
    """RF-6 follow-on (user site): the probe runs the route's own `python` (no -I), so a --user / PYTHONUSERBASE install is processed when the
    interpreter enables its user site — venv B (no .pth of its own) with `include-system-site-packages = true` (site.py enables the user
    site for such a venv) and PYTHONUSERBASE carrying the hook passes the gate; with the user site disabled (PYTHONNOUSERSITE=1) the same
    call is refused 'absent' unless the base interpreter's own site carries the hook (then the negative control is not decidable here)."""
    ub = _user_site(str(tmp_path / "userbase"))
    cfg = os.path.join(os.path.dirname(os.path.dirname(venvs["B"])), "pyvenv.cfg")
    txt = open(cfg, encoding="utf-8").read()
    open(cfg, "w", encoding="utf-8").write(txt.replace("include-system-site-packages = false", "include-system-site-packages = true"))
    try:
        r = _run(venvs["B"], {"PYTHONUSERBASE": ub})
        assert GATE_LINE not in r.stderr, r.stderr[-800:]                          # the user site's .pth ran: the hook is live
        base_site = subprocess.run([venvs["B"], "-c", "import site, os, sys; print(any(os.path.isfile(os.path.join(d, 'openfold3_opt_autoload.pth')) for d in site.getsitepackages()))"],
                                   capture_output=True, text=True, env={"PYTHONNOUSERSITE": "1", "PATH": os.environ.get("PATH", "")}).stdout.strip() == "True"
        r2 = _run(venvs["B"], {"PYTHONUSERBASE": ub, "PYTHONNOUSERSITE": "1"})
        if base_site:
            pytest.skip("the base interpreter's own site carries the hook: the user-site-off control is not decidable on this box")
        assert r2.returncode == 3 and GATE_LINE in r2.stderr and "user site disabled" in r2.stderr, (r2.returncode, r2.stderr[-600:])
    finally:
        open(cfg, "w", encoding="utf-8").write(txt)
