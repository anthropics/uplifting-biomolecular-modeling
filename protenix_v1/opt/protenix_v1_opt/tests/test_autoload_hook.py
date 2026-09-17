"""RF-6: run.sh's guard on the ENVIRONMENT ROUTE ONLY — it fires when PROTENIX_V1_OPT names a kit mode and no
--mode was given (the command-line mode read apart from the environment, before any merge), and the check is HOOK-LIVE: a fresh
interpreter (`env -u PROTENIX_V1_OPT python`) must carry protenix_v1_opt._autoload in sys.modules at start (the site-processed .pth's
effect, not its presence; no -I: the user site is processed when the interpreter enables it, as the route's `python` does); the one
refusal line carries the three-way diagnostic (absent / present but not processed / a stale copy).
The --mode route is un-gated (activation by construction: the package resolves the mode in-process). Two venvs of the running
interpreter (`python -m venv --without-pip`, no system site; the package + the core through a path file that sorts before the .pth):
A carries the .pth in its site-packages (what `run.sh install` ships) — the env route passes the guard; B carries no .pth — the env
route refuses with the one line ('absent'); B with the .pth beside a PYTHONPATH entry — 'present but not processed'; A with a stale
copy — 'a stale copy'; `PYTHONPATH=<kit> run.sh check --mode exact` in B (no install) is NOT refused — the package's own line; the
stock route (PROTENIX_V1_OPT=off, or --mode off) is exempt. Mutation-checked: with run.sh's guard removed,
test_venv_b_env_route_is_refused_absent fails."""
import os
import shutil
import subprocess
import sys
import venv

import pytest

from protenix_v1_opt import kit as K
from protenix_v1_opt import report as R

HOOK_LINE = "NOT ACTIVE: autoload: protenix_v1_opt._autoload is not in sys.modules at interpreter start"
RUN_SH = os.path.join(K.tree_home(), "run.sh")
CORE = os.path.join(K.tree_home(), "..", "common", "opt_core")


def _venv(tmp_path, name, with_pth):
    d = tmp_path / name
    venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(str(d))
    py = d / "bin" / "python"
    sp = subprocess.run([str(py), "-c", "import site; print(site.getsitepackages()[0])"], capture_output=True, text=True, check=True).stdout.strip()
    (d / "site").mkdir(exist_ok=True)
    with open(os.path.join(sp, "00_kit_paths.pth"), "w") as fh:               # the package + the core importable in the venv (sorts before the autoload .pth)
        fh.write(K.opt_home() + "\n" + os.path.abspath(CORE) + "\n")
    if with_pth:
        shutil.copy(K.autoload_pth_source(), sp)
    return d, sp


def _run(d, args, env_extra, cwd=None):
    env = {k: v for k, v in os.environ.items() if k not in ("PROTENIX_V1_OPT", "PYTHONPATH", "MODEL_OPT")}
    env["PATH"] = str(d / "bin") + os.pathsep + env.get("PATH", "")
    env.update(env_extra)
    return subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=cwd)


def _hook_lines(r):
    return [l for l in (r.stdout + r.stderr).splitlines() if HOOK_LINE in l]


def test_venv_a_env_route_passes_the_guard(tmp_path):
    d, sp = _venv(tmp_path, "A", with_pth=True)
    probe = subprocess.run([str(d / "bin" / "python"), "-c", "import sys; print('protenix_v1_opt._autoload' in sys.modules)"], capture_output=True, text=True).stdout.strip()
    assert probe == "True", probe                                          # the .pth in A's site is processed by a fresh interpreter: the hook is live
    r = _run(d, ["check"], {"PROTENIX_V1_OPT": "exact"})
    assert not _hook_lines(r), r.stderr[-800:]                             # the guard passes; what follows is the package's own check on this box
    assert '"gates"' in r.stdout or R.PREFIX in r.stderr


def test_venv_b_env_route_is_refused_absent(tmp_path):
    d, sp = _venv(tmp_path, "B", with_pth=False)
    cmd = ["check"]                                                        # the guard fires identically for every verb; one is enough
    r = _run(d, cmd, {"PROTENIX_V1_OPT": "exact"})
    lines = _hook_lines(r)
    assert r.returncode == R.EXIT_NOT_ACTIVE == 3 and len(lines) == 1 and "absent from the searched sites" in lines[0] and "run.sh install" in lines[0], (cmd, r.stderr[-600:])
    assert [l for l in r.stderr.splitlines() if l.startswith(R.PREFIX)] == lines   # exactly the one kit line
    assert "Traceback" not in r.stderr


def test_venv_b_pth_beside_pythonpath_is_present_but_not_processed(tmp_path):
    d, sp = _venv(tmp_path, "B2", with_pth=False)
    extra = tmp_path / "extra"; extra.mkdir(); shutil.copy(K.autoload_pth_source(), extra)   # a copy beside a PYTHONPATH entry: site.py never processes it
    r = _run(d, ["check"], {"PROTENIX_V1_OPT": "exact", "PYTHONPATH": str(extra)})
    lines = _hook_lines(r)
    assert r.returncode == 3 and len(lines) == 1 and "present but not processed" in lines[0] and str(extra) in lines[0], r.stderr[-600:]


def test_venv_a_stale_copy_is_refused_by_name(tmp_path):
    """A copy in site whose bytes differ from the tree's and whose line no longer imports the hook: not live, present in a site dir —
    the diagnostic reads 'a stale copy' (checked before 'present but not processed')."""
    d, sp = _venv(tmp_path, "A2", with_pth=True)
    stale = os.path.join(sp, os.path.basename(K.autoload_pth_source()))
    with open(stale, "w") as fh:
        fh.write("import protenix_v1_opt._autoload_retired\n")                # a retired hook name: site.py's import fails, the hook is not live
    probe = subprocess.run([str(d / "bin" / "python"), "-c", "import sys; print('protenix_v1_opt._autoload' in sys.modules)"], capture_output=True, text=True).stdout.strip()
    assert probe == "False", probe
    r = _run(d, ["check"], {"PROTENIX_V1_OPT": "exact"})
    lines = _hook_lines(r)
    assert r.returncode == 3 and len(lines) == 1 and "a stale copy" in lines[0] and stale in lines[0], r.stderr[-600:]


def test_venv_b_mode_route_is_not_gated_pythonpath_no_install(tmp_path):
    """The shape's passing case: `PYTHONPATH=<kit> run.sh check --mode exact` with NO install — never refused by this guard; the package's
    own line follows (activation by construction: the mode is resolved in-process)."""
    d = tmp_path / "C"; venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(str(d))   # no path file, no .pth: PYTHONPATH only
    r = _run(d, ["check", "--mode", "exact"], {"PYTHONPATH": K.opt_home() + os.pathsep + os.path.abspath(CORE)})
    assert not _hook_lines(r), r.stderr[-600:]
    assert '"gates"' in r.stdout or [l for l in r.stderr.splitlines() if l.startswith(R.PREFIX)], (r.stdout[-300:], r.stderr[-300:])
    r = _run(d, ["check", "--mode", "exact"], {"PYTHONPATH": K.opt_home() + os.pathsep + os.path.abspath(CORE), "PROTENIX_V1_OPT": "exact"})   # the variable set AND --mode given: still un-gated
    assert not _hook_lines(r), r.stderr[-600:]


def test_stock_route_is_exempt(tmp_path):
    d, sp = _venv(tmp_path, "B3", with_pth=False)
    r = _run(d, ["pred", "--input", "x.json", "--out_dir", str(tmp_path / "o")], {"PROTENIX_V1_OPT": "off"})
    assert not _hook_lines(r), r.stderr[-600:]                             # the env names off: no guard
    r = _run(d, ["check", "--mode", "off"], {"PROTENIX_V1_OPT": "exact"})
    assert not _hook_lines(r), r.stderr[-600:]                             # --mode given: no guard (run.sh then refuses check on the stock route, rc 2)
    assert r.returncode == 2


def test_guard_reads_the_cli_mode_before_the_merge():
    """run.sh: CLIMODE is captured from the parsed --mode before MODE merges with PROTENIX_V1_OPT, and the guard tests CLIMODE."""
    s = open(RUN_SH).read()
    assert s.index("CLIMODE=$MODE") < s.index("MODE=${MODE:-${PROTENIX_V1_OPT:-}}") < s.index('if [ -z "$CLIMODE" ] && [ -n "${PROTENIX_V1_OPT:-}" ] && [ "$PROTENIX_V1_OPT" != off ]')
    assert "kit.autoload_hook_main()" in s and s.count("autoload_hook_main") == 1


def _user_site_layout(userbase):
    """The user site dir a fresh interpreter derives from PYTHONUSERBASE (site.getusersitepackages())."""
    return subprocess.run([sys.executable, "-c", "import site; print(site.getusersitepackages())"], capture_output=True, text=True, env=dict(os.environ, PYTHONUSERBASE=str(userbase)), check=True).stdout.strip()


def test_probe_processes_the_user_site_when_enabled(tmp_path):
    """RF-6 follow-on: the probe runs a fresh `python` WITHOUT -I, so a `pip install --user` / PYTHONUSERBASE install of the .pth is a live
    install wherever the interpreter enables its user site (the route's `python` processes it the same way). Skipped where this
    interpreter has the user site disabled (a venv without system site: the diagnostic names that case instead)."""
    if sys.prefix != sys.base_prefix:
        pytest.skip("a venv interpreter: its user site is disabled by construction (site.ENABLE_USER_SITE False)")
    userbase = tmp_path / "userbase"; usite = _user_site_layout(userbase); os.makedirs(usite, exist_ok=True)
    shutil.copy(K.autoload_pth_source(), usite)
    env = dict(os.environ, PYTHONUSERBASE=str(userbase)); env.pop("PROTENIX_V1_OPT", None); env.pop("PYTHONNOUSERSITE", None)
    probe = subprocess.run([sys.executable, "-c", "import sys, site; print(site.ENABLE_USER_SITE, site.getusersitepackages() in sys.path, 'protenix_v1_opt._autoload' in sys.modules)"],
                           capture_output=True, text=True, env=env).stdout.split()
    assert probe == ["True", "True", "True"], probe                       # the user site enabled, on sys.path, and its .pth processed (the hook live) — a -I probe reads False/False
    pr = K.autoload_hook_probe()                                          # the gate's own probe form: the user site enabled in ITS fresh interpreter (a -I probe would read False)
    assert pr["user_enabled"] is True and pr["live"] is True, pr
    r = _run_sys(["check"], {"PROTENIX_V1_OPT": "exact", "PYTHONUSERBASE": str(userbase)})
    assert not _hook_lines(r), r.stderr[-600:]                             # the gate passes on the user-site install


def test_venv_user_site_copy_is_named_disabled(tmp_path):
    """In a venv without system site the user site is disabled: a .pth there is 'present but not processed … which this interpreter has
    disabled' (the truth for that interpreter; the route's `python` would not process it either)."""
    d, sp = _venv(tmp_path, "B4", with_pth=False)
    userbase = tmp_path / "ub"; usite = subprocess.run([str(d / "bin" / "python"), "-c", "import site; print(site.getusersitepackages())"], capture_output=True, text=True, env=dict(os.environ, PYTHONUSERBASE=str(userbase)), check=True).stdout.strip()
    os.makedirs(usite, exist_ok=True); shutil.copy(K.autoload_pth_source(), usite)
    r = _run(d, ["check"], {"PROTENIX_V1_OPT": "exact", "PYTHONUSERBASE": str(userbase)})
    lines = _hook_lines(r)
    assert r.returncode == 3 and len(lines) == 1 and "present but not processed" in lines[0] and "has disabled" in lines[0], r.stderr[-600:]


def _run_sys(args, env_extra):
    """run.sh with THIS interpreter's `python` first on PATH (the installed one), the kit variables cleared, then env_extra."""
    env = {k: v for k, v in os.environ.items() if k not in ("PROTENIX_V1_OPT", "PYTHONPATH", "MODEL_OPT")}
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env.update(env_extra)
    return subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env)


def test_a13_the_installed_pth_never_kills_run_shs_import_probe_silently(tmp_path):
    """A13 lock (venv A: the INSTALLED .pth processed at interpreter start): under PROTENIX_V1_OPT=<mode> the bare import probe passes
    (the hook fires on the stock's trigger only, not at start); under a mistyped mode / an undeclared PROTENIX_V1_OPT* name run.sh prints
    the .pth's own NOT ACTIVE line BY NAME and exits 3 — never the false 'not installed' diagnosis with the line silenced."""
    d, sp = _venv(tmp_path, "A13", with_pth=True)
    py = str(d / "bin" / "python")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_V1_OPT") and k not in ("PYTHONPATH", "MODEL_OPT")}
    p = subprocess.run([py, "-c", "import protenix_v1_opt; print('IMPORTED')"], capture_output=True, text=True, env=dict(env, PROTENIX_V1_OPT="fast"))
    assert p.returncode == 0 and "IMPORTED" in p.stdout, p.stderr[-400:]
    for extra in ({"PROTENIX_V1_OPT": "fsat"}, {"PROTENIX_V1_OPT": "fast", "PROTENIX_V1_OPT_MOOD": "1"}):
        r = _run(d, ["check"], extra)
        assert r.returncode == 3, (extra, r.returncode, r.stderr[-500:])
        assert "[protenix-v1-opt] NOT ACTIVE:" in r.stderr and "is not installed" not in r.stderr, (extra, r.stderr[-500:])


# ------------------------------------------------------------------------------------------------ the package probe (run.sh probe_package; configs/*.env call `run.sh probe`)
def test_the_config_route_keeps_the_pths_refusal_by_name(tmp_path):
    """`run.sh --config h100 check` and a shell that sources configs/h100.env (`source … || exit $?`, run.sh's form) under an undeclared
    PROTENIX_V1_OPT* name: the probe exits with python's own code (3) and the .pth's own NOT ACTIVE line — never `not installed`."""
    d, sp = _venv(tmp_path, "P1", with_pth=True)
    for extra in ({"PROTENIX_V1_OPT": "fast", "PROTENIX_V1_OPT_MOOD": "1"}, {"PROTENIX_V1_OPT": "fsat"}):
        r = _run(d, ["--config", "h100", "check"], extra)
        assert r.returncode == 3 and "[protenix-v1-opt] NOT ACTIVE:" in r.stderr and "is not installed" not in r.stderr, (extra, r.returncode, r.stderr[-500:])
        env = {k: v for k, v in os.environ.items() if k not in ("PROTENIX_V1_OPT", "PYTHONPATH", "MODEL_OPT")}
        env["PATH"] = str(d / "bin") + os.pathsep + env.get("PATH", ""); env.update(extra)
        r = subprocess.run(["bash", "-c", f'source "{os.path.join(K.tree_home(), "configs", "h100.env")}" || exit $?; echo SOURCED-PAST-THE-PROBE'], capture_output=True, text=True, env=env)
        assert r.returncode == 3 and "SOURCED-PAST-THE-PROBE" not in r.stdout and "[protenix-v1-opt] NOT ACTIVE:" in r.stderr and "is not installed" not in r.stderr, (extra, r.returncode, r.stderr[-500:])


def test_an_import_time_refusal_keeps_its_own_code_and_words(tmp_path):
    """A package whose import raises SystemExit(5) with its own line (a stand-in for any refusal raised while importing): `run.sh probe`
    exits 5 with that line, never `not installed`."""
    d, sp = _venv(tmp_path, "P2", with_pth=False)
    fake = tmp_path / "fakepkg" / "protenix_v1_opt"; fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("import sys\nsys.stderr.write('[protenix-v1-opt] NOT ACTIVE: a refusal raised while importing\\n')\nraise SystemExit(5)\n")
    with open(os.path.join(sp, "00_kit_paths.pth"), "w") as fh:                 # the stand-in package shadows the real one in this venv
        fh.write(str(tmp_path / "fakepkg") + "\n")
    r = _run(d, ["probe"], {})
    assert r.returncode == 5 and "a refusal raised while importing" in r.stderr and "is not installed" not in r.stderr, (r.returncode, r.stderr[-400:])


def test_an_absent_package_is_not_installed_rc_2(tmp_path):
    """No protenix_v1_opt importable at all (a bare venv): `run.sh probe` and the config say `not installed`, rc 2."""
    d = tmp_path / "P3"
    venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(str(d))
    r = _run(d, ["probe"], {})
    assert r.returncode == 2 and "NOT ACTIVE: protenix_v1_opt is not installed" in r.stderr, (r.returncode, r.stderr[-400:])
    r = _run(d, ["--config", "h100", "check", "--mode", "exact"], {})
    assert r.returncode == 2 and "NOT ACTIVE: protenix_v1_opt is not installed" in r.stderr, (r.returncode, r.stderr[-400:])


def test_an_importable_package_passes_the_probe(tmp_path):
    d, sp = _venv(tmp_path, "P4", with_pth=True)
    r = _run(d, ["probe"], {"PROTENIX_V1_OPT": "fast"})
    assert r.returncode == 0 and "NOT ACTIVE" not in r.stderr, (r.returncode, r.stderr[-400:])
