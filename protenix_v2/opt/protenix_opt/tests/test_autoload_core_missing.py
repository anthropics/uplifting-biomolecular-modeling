"""PROTENIX_OPT=<mode> with the shared core not importable: one NOT ACTIVE line naming core_missing:<module> and exit 3 — never a
traceback, never a silent stock run (the .pth route's `protenix_opt._autoload`)."""
import os
import subprocess
import sys

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # protenix_v2/opt


def test_core_missing_is_not_active_exit_3():
    code = ("import sys; sys.modules['opt_core'] = None; sys.modules['opt_core.autoload'] = None; "   # the core absent from this interpreter
            "import protenix_opt._autoload")
    env = dict(os.environ, PROTENIX_OPT="exact", PYTHONPATH=OPT)
    env.pop("PROTENIX_ROOT_FROZEN", None)
    # -S: no site processing, so an installed kit's .pth cannot import _autoload (with the core present) before the test's first line;
    # PYTHONPATH still names the package, and the pinned-path fallback (_core) meets the same absent core
    r = subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3, (r.returncode, r.stderr[-600:])
    assert "[protenix-opt] NOT ACTIVE: reason=core_missing:opt_core" in r.stderr and "Traceback" not in r.stderr, r.stderr[-600:]


def test_core_present_and_mode_off_installs_nothing():
    r = subprocess.run([sys.executable, "-c", "import protenix_opt._autoload as a; print(a.FINDER)"],
                       env=dict(os.environ, PROTENIX_OPT="off", PYTHONPATH=OPT), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == "None", r.stderr[-400:]
