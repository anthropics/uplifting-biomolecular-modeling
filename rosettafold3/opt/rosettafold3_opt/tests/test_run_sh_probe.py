"""run.sh's (and configs/h100.env's) 'is the package usable?' probe is a pass-through: whatever `python -m rosettafold3_opt._require` prints is
shown verbatim and its exit code becomes the route's; the 'not installed' word is said ONLY when the package has no import location on the
interpreter (importlib's find_spec is None). A shim interpreter stands in for the box's python: it answers the two probe forms by environment."""
import os
import stat
import subprocess
import sys

import pytest

from .. import _core, stack

SHIM = """#!/bin/bash
# a stand-in interpreter: `-m rosettafold3_opt._require` prints $SHIM_TEXT and exits $SHIM_RC; the find_spec absence question exits $SHIM_PRESENT (0 present, 41 absent)
for a in "$@"; do
  case "$a" in
    rosettafold3_opt._require) [ -n "${SHIM_TEXT:-}" ] && echo "$SHIM_TEXT" >&2; exit "${SHIM_RC:-0}" ;;
    *find_spec*) exit "${SHIM_PRESENT:-0}" ;;
  esac
done
exit 0
"""


def _shim(tmp_path):
    p = tmp_path / "python"; p.write_text(SHIM); p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


def _env(tmp_path, rc, text, present):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROSETTAFOLD3_OPT") and not k.startswith("MODEL_OPT")}
    env.update({"ROSETTAFOLD3_OPT_PYTHON": _shim(tmp_path), "SHIM_RC": str(rc), "SHIM_TEXT": text, "SHIM_PRESENT": "0" if present else "41"})
    return env


def _run_sh(tmp_path, rc, text, present):
    return subprocess.run(["bash", os.path.join(stack.tree_root(), "run.sh"), "check", "--mode", "exact"], capture_output=True, text=True, env=_env(tmp_path, rc, text, present))


def test_a_refusal_at_import_passes_through_with_its_text_and_exit_code(tmp_path):
    r = _run_sh(tmp_path, 5, "[rosettafold3-opt] NOT ACTIVE: reason=refused_by_this_test", present=True)
    assert r.returncode == 5, (r.returncode, r.stderr)
    assert "reason=refused_by_this_test" in r.stderr and "not installed" not in r.stderr


def test_the_packages_own_refusal_code_stays_3_without_the_install_word(tmp_path):
    r = _run_sh(tmp_path, 3, "[rosettafold3-opt] NOT ACTIVE: reason=producer_missing:opt_core.oom", present=True)
    assert r.returncode == 3 and "producer_missing" in r.stderr and "not installed" not in r.stderr, (r.returncode, r.stderr)


def test_a_missing_package_is_the_only_not_installed(tmp_path):
    r = _run_sh(tmp_path, 1, "ModuleNotFoundError: No module named 'rosettafold3_opt'", present=False)
    assert r.returncode == 3 and "rosettafold3_opt is not installed on" in r.stderr and "No module named" in r.stderr, (r.returncode, r.stderr)


def test_the_sourced_config_passes_the_refusal_through_too(tmp_path):
    cfg = os.path.join(stack.tree_root(), "configs", "h100.env")
    script = f"source {cfg}; echo rc=$? key=${{MODEL_OPT_STACK_KEY:-unset}}"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=_env(tmp_path, 5, "[rosettafold3-opt] NOT ACTIVE: reason=refused_by_this_test", present=True))
    assert r.stdout.strip() == "rc=5 key=unset" and "reason=refused_by_this_test" in r.stderr and "not installed" not in r.stderr, (r.stdout, r.stderr)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=_env(tmp_path, 1, "ModuleNotFoundError: No module named 'rosettafold3_opt'", present=False))
    assert r.stdout.strip() == "rc=3 key=unset" and "rosettafold3_opt is not installed on" in r.stderr, (r.stdout, r.stderr)


# ---- a LIVE start-up hook: what an installed .pth does at every interpreter start, staged with sitecustomize on a real interpreter
REFUSAL = "[rosettafold3-opt] NOT ACTIVE: refused at interpreter start by this test's shadow package"


def _shadow(tmp_path, with_pkg, how="os_exit"):
    """A directory first on PYTHONPATH: sitecustomize.py imports rosettafold3_opt at start-up (guarded against a genuine absence); with_pkg adds a
    shadowing rosettafold3_opt whose import writes a refusal and ends the interpreter with 5 — ``os._exit(5)`` (how a start-up hook refuses:
    the process exits 5) or ``raise SystemExit(5)`` (which CPython's site initialisation turns into its own fatal-init exit code)."""
    d = tmp_path / "shadow"; d.mkdir()
    (d / "sitecustomize.py").write_text("try:\n    import rosettafold3_opt\nexcept ImportError:\n    pass\n")
    if with_pkg:
        end = "os._exit(5)" if how == "os_exit" else "raise SystemExit(5)"
        (d / "rosettafold3_opt").mkdir()
        (d / "rosettafold3_opt" / "__init__.py").write_text(f"import os, sys\nsys.stderr.write({REFUSAL!r} + chr(10)); sys.stderr.flush()\n{end}\n")
    return str(d)


def _live_env(tmp_path, with_pkg, importable, how="os_exit"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROSETTAFOLD3_OPT") and not k.startswith("MODEL_OPT") and k not in ("PYTHONNOUSERSITE",)}
    paths = [_shadow(tmp_path, with_pkg, how)] + ([stack.opt_root(), _core.pin_path()] if importable else [])
    env.update({"ROSETTAFOLD3_OPT_PYTHON": sys.executable, "PYTHONPATH": os.pathsep.join(paths), "PYTHONDONTWRITEBYTECODE": "1", "ROSETTAFOLD3_OPT": "exact"})
    return env


@pytest.mark.parametrize("how", ["os_exit", "systemexit"])
def test_a_package_refusing_at_interpreter_start_is_never_called_not_installed(tmp_path, how):
    env = _live_env(tmp_path, with_pkg=True, importable=True, how=how)
    probe = subprocess.run([sys.executable, "-c", "pass"], capture_output=True, text=True, env=env)
    rc = probe.returncode                                                                               # the stage itself: every start of this interpreter refuses (5 by os._exit; the
    assert rc != 0 and rc != 41 and REFUSAL in probe.stderr, (rc, probe.stderr)                        # interpreter's own fatal-init code for a SystemExit inside site initialisation)
    assert how != "os_exit" or rc == 5, rc
    r = subprocess.run(["bash", os.path.join(stack.tree_root(), "run.sh"), "check", "--mode", "exact"], capture_output=True, text=True, env=env)
    assert r.returncode == rc and REFUSAL in r.stderr and "not installed" not in r.stderr, (r.returncode, rc, r.stderr[-800:])
    script = f"source {os.path.join(stack.tree_root(), 'configs', 'h100.env')}; echo rc=$? key=${{MODEL_OPT_STACK_KEY:-unset}}"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert r.stdout.strip() == f"rc={rc} key=unset" and REFUSAL in r.stderr and "not installed" not in r.stderr, (r.stdout, r.stderr[-800:])


def test_a_genuinely_absent_package_under_a_live_startup_hook_is_not_installed(tmp_path):
    env = _live_env(tmp_path, with_pkg=False, importable=False)
    found = subprocess.run([sys.executable, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('rosettafold3_opt') is not None else 41)"], capture_output=True, text=True, env=env)
    if found.returncode != 41:
        pytest.skip("rosettafold3_opt is installed on this interpreter itself: a genuine absence cannot be staged here")
    r = subprocess.run(["bash", os.path.join(stack.tree_root(), "run.sh"), "check", "--mode", "exact"], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "rosettafold3_opt is not installed on" in r.stderr, (r.returncode, r.stderr[-800:])
    script = f"source {os.path.join(stack.tree_root(), 'configs', 'h100.env')}; echo rc=$? key=${{MODEL_OPT_STACK_KEY:-unset}}"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert r.stdout.strip() == "rc=3 key=unset" and "rosettafold3_opt is not installed on" in r.stderr, (r.stdout, r.stderr[-800:])
