"""`import protenix_v1_opt` (the .pth line) imports neither torch nor protenix and is quick; PROTENIX_V1_OPT=exact arms a finder and nothing else."""
import importlib
import os
import subprocess
import sys

from .conftest import OPT

CORE = os.path.dirname(os.path.dirname(os.path.abspath(importlib.import_module("opt_core").__file__)))   # common/opt_core, installed beside the kit

CODE = "import sys, time; t=time.perf_counter(); import protenix_v1_opt._autoload as A; dt=time.perf_counter()-t; " \
       "print(int('torch' in sys.modules), int('protenix' in sys.modules), int(A.FINDER is not None), round(dt, 3))"


def _run(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PROTENIX_V1_OPT", "PYTHONPATH"))}
    env.update(env_extra, PYTHONPATH=os.pathsep.join([OPT, CORE]), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, check=True).stdout.split()


def test_import_is_light_and_installs_no_finder_by_default():
    torch, ptx, finder, dt = _run({})
    assert (torch, ptx, finder) == ("0", "0", "0") and float(dt) < 1.0


def test_env_mode_arms_a_finder_only():
    torch, ptx, finder, dt = _run({"PROTENIX_V1_OPT": "exact"})
    assert (torch, ptx, finder) == ("0", "0", "1")
