"""The env route with the shared core absent: `OPENFOLD3_OPT=<mode>` at interpreter start must end in the kit's NOT ACTIVE line and exit 3
(reason=core_missing:<module>), never a traceback and never a silent stock run (_autoload.install)."""
import os
import subprocess
import sys
import textwrap

PKG_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))      # opt/ — openfold3_opt importable from here without the core


def test_core_missing_is_exit_3_named(tmp_path):
    # a python that can import openfold3_opt (from opt/) but NOT opt_core: -S drops site-packages (the editable installs), sys.path carries opt/ only
    code = textwrap.dedent("""
        import sys
        sys.path[:0] = [sys.argv[1]]
        for k in [k for k in sys.modules if k == 'opt_core' or k.startswith('opt_core.')]:
            del sys.modules[k]
        import openfold3_opt._autoload          # runs install() at import, as the .pth does
        print('IMPORTED-WITHOUT-EXIT')
    """)
    env = {k: v for k, v in os.environ.items() if not (k.startswith("OPENFOLD3_OPT") or k.startswith("OF3") or k == "PYTHONPATH")}
    env.update({"OPENFOLD3_OPT": "exact", "PYTHONDONTWRITEBYTECODE": "1"})
    r = subprocess.run([sys.executable, "-S", "-c", code, PKG_PARENT], capture_output=True, text=True, env=env)
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr[-600:])
    assert "[openfold3-opt] NOT ACTIVE: reason=core_missing:opt_core" in r.stderr and "Traceback" not in r.stderr and "IMPORTED-WITHOUT-EXIT" not in r.stdout, r.stderr[-600:]


def test_core_present_env_route_still_arms():
    env = {k: v for k, v in os.environ.items() if not (k.startswith("OPENFOLD3_OPT") or k.startswith("OF3"))}
    env.update({"OPENFOLD3_OPT": "exact", "PYTHONDONTWRITEBYTECODE": "1"})
    r = subprocess.run([sys.executable, "-c", "import openfold3_opt._autoload as a; print('FINDER', type(a.FINDER).__name__)"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "FINDER" in r.stdout, (r.returncode, r.stderr[-400:])


def test_run_sh_and_python_m_refuse_by_name_when_the_core_is_absent(tmp_path):
    """`bash run.sh check` and `python -m openfold3_opt check` with opt_core NOT importable (a shadow package that raises ImportError ahead of
    the real one on sys.path): exit 3 with the NOT ACTIVE core_missing line — the real entry routes, never a raw ModuleNotFoundError."""
    import os, subprocess, sys
    kit = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))      # …/openfold3
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENFOLD3_OPT", "OF3", "PYTHONPATH"))}
    env["PYTHONPATH"] = os.path.join(kit, "opt")                                                             # the package alone: nothing importable as opt_core (-S below: no site dir either)
    r = subprocess.run([sys.executable, "-S", "-m", "openfold3_opt", "check", "--mode", "fast"], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "[openfold3-opt] NOT ACTIVE: reason=core_missing:opt_core" in r.stderr, (r.returncode, r.stderr[-500:])
    shim = tmp_path / "bin"; shim.mkdir()                                                   # run.sh calls `python`: a shim = this interpreter with -S (no site dir, so an
    (shim / "python").write_text(f"#!/bin/sh\nexec {sys.executable} -S \"$@\"\n"); (shim / "python").chmod(0o755)   # installed core cannot answer for the absent one)
    env["PATH"] = str(shim) + os.pathsep + env.get("PATH", "")
    r = subprocess.run(["bash", os.path.join(kit, "run.sh"), "check", "--config", "h100", "--mode", "fast"], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 3 and "[openfold3-opt] NOT ACTIVE: reason=core_missing" in r.stderr, (r.returncode, r.stderr[-500:], r.stdout[-300:])
