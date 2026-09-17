"""run.sh's hook-live gate on the ENV ROUTE: with ROSETTAFOLD3_OPT naming a kit mode and no --mode given, `pred` / `check` / `warm`
refuse by name (exit 3, one line) unless the kit's .pth hook is LIVE on the interpreter — `rosettafold3_opt._autoload`
imported at the start of a fresh interpreter — the route's own `python`, no -I, the variable unset (the site-processed .pth's effect);
a package that is merely importable (PYTHONPATH), a .pth copied where site.py does not read it, or a stale copy would run stock silently
under the variable. The --mode route is not gated (the package's CLI activates in-process). Venvs: A carries the package on its site (a
path .pth, as an editable install writes) plus the autoload .pth (the pip-installed state) — past the gate; B has the package importable
via PYTHONPATH only, no .pth on its path — 'absent'; C = B with the .pth beside its PYTHONPATH entry (the tree's opt/, where the kit ships
it) — 'present but not processed'; D carries a stale copy in its site — 'a stale copy'; U = a USER-SITE install (PYTHONUSERBASE, the two
.pth files under site.getusersitepackages()) on an interpreter whose user site is enabled — past the gate, and 'present but not processed'
(ENABLE_USER_SITE=False) once the user site is disabled (PYTHONNOUSERSITE). `--mode off` / ROSETTAFOLD3_OPT=off need no hook. Removing the
gate fails the refusal tests (B/C/D then fail later, on another line)."""
import os
import shutil
import subprocess
import sys
import venv

import pytest

from .. import stack

PTH = "rosettafold3_opt_autoload.pth"
HEAD = "[rosettafold3-opt] NOT ACTIVE: the .pth hook is not live on "


def _write_pth(sp, pth):
    """The two .pth files of an install under site dir `sp`: the package's path line (pip's editable form; sorts first, as pip's does) and the
    autoload line — 'live' as the kit ships it, 'stale' with another line."""
    os.makedirs(sp, exist_ok=True)
    open(os.path.join(sp, "__editable__.rosettafold3_opt.pth"), "w").write(stack.opt_root() + "\n")
    if pth == "live":
        shutil.copyfile(os.path.join(stack.opt_root(), PTH), os.path.join(sp, PTH))
    else:
        open(os.path.join(sp, PTH), "w").write("import rosettafold3_opt\n")


def _venv(path, pth, system_site=False):
    """pth: None (no site .pth), 'live' (the package's path .pth + the autoload .pth, as pip writes them), 'stale' (the autoload .pth with another line)."""
    venv.EnvBuilder(with_pip=False, symlinks=True, system_site_packages=system_site).create(path)
    py = os.path.join(path, "bin", "python")
    sp = subprocess.run([py, "-c", "import site; print(site.getsitepackages()[0])"], capture_output=True, text=True, check=True).stdout.strip()
    if pth:
        _write_pth(sp, pth)
    return py


def _run(py, cmd, mode=None, env_mode=None, pythonpath=None, extra_env=None):
    """run.sh <cmd> [--mode <mode>] with ROSETTAFOLD3_OPT=<env_mode> when given (the env route: the variable names the mode, no --mode)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROSETTAFOLD3_OPT") and k not in ("PYTHONUSERBASE", "PYTHONNOUSERSITE")}
    from .. import _core
    env.update({"ROSETTAFOLD3_OPT_PYTHON": py, "PYTHONPATH": pythonpath or os.pathsep.join([stack.opt_root(), _core.pin_path()]), "PYTHONDONTWRITEBYTECODE": "1"})   # the tree's opt/ + its pinned core
    env.update(extra_env or {})
    if env_mode:
        env["ROSETTAFOLD3_OPT"] = env_mode
    argv = ["bash", os.path.join(stack.tree_root(), "run.sh"), cmd] + (["--mode", mode] if mode else [])
    return subprocess.run(argv, capture_output=True, text=True, env=env)


@pytest.fixture(scope="module")
def venvs(tmp_path_factory):
    d = tmp_path_factory.mktemp("pth_gate")
    bare = d / "pkg_only"                                        # the package importable from a directory that carries no .pth (venv B's PYTHONPATH)
    shutil.copytree(os.path.join(stack.opt_root(), "rosettafold3_opt"), bare / "rosettafold3_opt", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    from .. import _core as _c                                   # the copy carries its pin table (the pin gate reads it beside the package), the core path made absolute
    (bare / "pyproject.toml").write_text(open(_c.PYPROJECT).read().replace('path = "../../common/opt_core"', f'path = "{_c.pin_path()}"'))
    ub = d / "userbase"                                          # a user-site install: PYTHONUSERBASE, the .pth files under its site-packages
    py_u = _venv(str(d / "U"), None, system_site=True)           # --system-site-packages: the user site is enabled on this interpreter
    usp = subprocess.run([py_u, "-c", "import site; print(site.getusersitepackages())"], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONUSERBASE": str(ub)}).stdout.strip()
    _write_pth(usp, "live")
    from .. import _core
    return {"A": _venv(str(d / "A"), "live"), "B": _venv(str(d / "B"), None), "D": _venv(str(d / "D"), "stale"), "U": py_u, "userbase": str(ub),
            "bare": os.pathsep.join([str(bare), _core.pin_path()])}     # the package (no .pth anywhere) + the pinned core: the route reaches the hook gate, not the producer check


def _refused(r, py, why):
    assert r.returncode == 3, (r.returncode, r.stderr)
    lines = r.stderr.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith(HEAD + py) and why in lines[0] and "pip install -e" in lines[0], r.stderr


def test_env_route_without_the_hook_is_refused_absent(venvs):
    for cmd, env_mode in (("pred", "exact"), ("check", "fast"), ("warm", "exact")):
        _refused(_run(venvs["B"], cmd, env_mode=env_mode, pythonpath=venvs["bare"]), venvs["B"], f"{PTH} is absent from the searched sites")


def test_env_route_pth_beside_a_pythonpath_entry_is_present_but_not_processed(venvs):
    _refused(_run(venvs["B"], "check", env_mode="exact"), venvs["B"], "is present but not processed")   # the tree's opt/ on PYTHONPATH carries the file; site.py never reads it there


def test_env_route_stale_pth_in_site_is_refused(venvs):
    _refused(_run(venvs["D"], "check", env_mode="exact"), venvs["D"], "is a stale copy")


def test_env_route_with_the_live_hook_passes_the_gate(venvs):
    for cmd, env_mode in (("pred", "exact"), ("check", "fast")):
        r = _run(venvs["A"], cmd, env_mode=env_mode)              # past the gate the command goes on (the pins / the GPU decide the rest)
        assert HEAD not in r.stderr and PTH not in r.stderr, (cmd, r.stderr)


def test_env_route_user_site_install_is_live(venvs, tmp_path):
    """pip install --user / PYTHONUSERBASE: the route's own `python` processes that site (no -I in the probe) — past the gate; with the user site
    disabled the same files are 'present but not processed', the line naming ENABLE_USER_SITE=False. Runs on a non-venv interpreter whose user
    site is enabled and which does not already carry the hook in a system site; skipped BY NAME otherwise (a venv disables the user site by
    construction; an interpreter with the package pip-installed passes the gate from its system site whatever the user site says)."""
    py = os.path.join(sys.base_prefix, "bin", "python3")
    if not os.path.isfile(py):
        pytest.skip("no base interpreter at sys.base_prefix/bin/python3 (the user-site route needs a non-venv python)")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROSETTAFOLD3_OPT") and k not in ("PYTHONUSERBASE", "PYTHONNOUSERSITE", "PYTHONPATH")}
    env["PYTHONUSERBASE"] = str(tmp_path / "userbase")
    probe = subprocess.run([py, "-c", "import site, sys; print(site.getusersitepackages()); print(site.ENABLE_USER_SITE); "
                            "import importlib.util; print(any(__import__('os').path.isfile(__import__('os').path.join(p, 'rosettafold3_opt_autoload.pth')) for p in site.getsitepackages()))"],
                           capture_output=True, text=True, env=env, check=True).stdout.split()
    if probe[1] != "True":
        pytest.skip(f"the user site is disabled for {py} (ENABLE_USER_SITE={probe[1]})")
    if probe[2] == "True":
        pytest.skip(f"{py} already carries {PTH} in a system site (a pip install of this package): the user-site route cannot be told apart there")
    _write_pth(probe[0], "live")
    r = _run(py, "check", env_mode="exact", pythonpath=venvs["bare"], extra_env={"PYTHONUSERBASE": env["PYTHONUSERBASE"]})
    assert HEAD not in r.stderr and PTH not in r.stderr, r.stderr
    r = _run(py, "check", env_mode="exact", pythonpath=venvs["bare"], extra_env={"PYTHONUSERBASE": env["PYTHONUSERBASE"], "PYTHONNOUSERSITE": "1"})
    _refused(r, py, "is present but not processed")
    assert "ENABLE_USER_SITE=False" in r.stderr, r.stderr


def test_mode_route_is_not_gated(venvs):
    """`--mode <kit mode>` on the command line: the package's CLI activates in-process (its own ACTIVE line is the evidence) — never this gate,
    hook or no hook; the same with the variable set and agreeing."""
    for env_mode in (None, "exact"):
        r = _run(venvs["B"], "check", mode="exact", env_mode=env_mode, pythonpath=venvs["bare"])
        assert HEAD not in r.stderr and PTH not in r.stderr, r.stderr


def test_off_needs_no_hook(venvs):
    for mode, env_mode in (("off", None), (None, "off")):
        r = _run(venvs["B"], "check", mode=mode, env_mode=env_mode, pythonpath=venvs["bare"])
        assert HEAD not in r.stderr and PTH not in r.stderr, r.stderr


def test_an_exported_mode_never_kills_the_probes_on_a_live_hook(venvs):
    """Class A13: with the INSTALLED .pth processed at interpreter start and ROSETTAFOLD3_OPT=<mode> already exported, every probe of run.sh
    and configs/h100.env (python -m rosettafold3_opt._require, the stack-key probe, the hook-liveness probe) either passes or refuses BY
    NAME — never a silent rc with a false 'not installed' diagnosis. This tree's .pth arms a lazy finder for `rf3` (the gate runs at the
    upstream import / in enable()), so a valid mode passes the probes and the command refuses by name later (stock tree / no GPU here); a
    mistyped mode is the .pth's own one-line refusal at start, surfaced verbatim by the probe that hit it."""
    r = _run(venvs["A"], "check", env_mode="exact")
    assert r.returncode == 3 and "rosettafold3_opt is not installed on" not in r.stderr and HEAD not in r.stderr, (r.returncode, r.stderr[-600:])
    assert any(l.startswith("[rosettafold3-opt] NOT ACTIVE: ") or l.startswith("run.sh: the pinned upstream is not installed at the pin") for l in r.stderr.splitlines()), r.stderr[-600:]   # past both probes: the next gate (the pins on this bare venv, else the activation) refuses by name
    r = _run(venvs["A"], "check", env_mode="bogus")
    assert r.returncode == 3 and "rosettafold3_opt is not installed on" not in r.stderr, (r.returncode, r.stderr[-600:])
    assert any(l.startswith("[rosettafold3-opt] NOT ACTIVE: unknown ROSETTAFOLD3_OPT='bogus'") for l in r.stderr.splitlines()), r.stderr[-600:]
    from .. import _core
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROSETTAFOLD3_OPT") and k not in ("PYTHONUSERBASE", "PYTHONNOUSERSITE", "MODEL_OPT_STACK_KEY", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")}
    env.update({"ROSETTAFOLD3_OPT_PYTHON": venvs["A"], "PYTHONPATH": os.pathsep.join([stack.opt_root(), _core.pin_path()]), "PYTHONDONTWRITEBYTECODE": "1"})
    script = f"source {os.path.join(stack.tree_root(), 'configs', 'h100.env')}; rc=$?; echo rc=$rc key=${{MODEL_OPT_STACK_KEY:-unset}}; exit $rc"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**env, "ROSETTAFOLD3_OPT": "exact"})
    assert r.returncode == 0 and "key=unset" not in r.stdout and "key=\n" not in r.stdout and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stdout, r.stderr[-600:])
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**env, "ROSETTAFOLD3_OPT": "bogus"})
    assert r.returncode == 3 and "rosettafold3_opt is not installed on" not in r.stderr and "key=unset" in r.stdout, (r.returncode, r.stdout, r.stderr[-600:])   # refused by name, nothing exported
    assert r.stderr.strip().splitlines()[0].startswith("[rosettafold3-opt] NOT ACTIVE: unknown ROSETTAFOLD3_OPT='bogus'"), r.stderr[-600:]
