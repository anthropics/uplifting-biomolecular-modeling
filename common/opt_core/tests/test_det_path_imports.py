"""The deterministic recipe's import path holds no core module beyond its own closure.

A stock process that applies the recipe through a ``sitecustomize`` (``from opt_core.precision.recipe import apply_torch``) is proven clean by
its kit against an allow-list of exactly these modules; a module-level import added anywhere on the path (``gates`` importing ``report``, say)
makes that proof refuse by name. The closure is measured in a fresh interpreter with only this core on ``sys.path`` (no torch needed: the
recipe imports torch on use)."""
import json
import os
import subprocess
import sys

CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # common/opt_core (the directory holding the opt_core package)
DET_PATH = ("opt_core.det", "opt_core.gates", "opt_core.precision.policy", "opt_core.precision.recipe")
CLOSURE = {"opt_core", "opt_core.det", "opt_core.gates", "opt_core.precision", "opt_core.precision.policy", "opt_core.precision.recipe"}


def _loaded_after_importing(*mods):
    code = ("import importlib, json, sys; sys.path.insert(0, %r)\nfor m in %r: importlib.import_module(m)\n"
            "print(json.dumps({'core': sorted(k for k in sys.modules if k == 'opt_core' or k.startswith('opt_core.')), 'torch': 'torch' in sys.modules}))" % (CORE, list(mods)))
    out = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_det_path_holds_only_its_closure():
    got = _loaded_after_importing(*DET_PATH)
    assert set(got["core"]) == CLOSURE, sorted(set(got["core"]) ^ CLOSURE)
    assert got["torch"] is False


def test_each_det_path_module_alone_never_loads_report():
    for m in DET_PATH:
        got = _loaded_after_importing(m)
        assert "opt_core.report" not in got["core"], (m, got["core"])
        assert set(got["core"]) <= CLOSURE, (m, sorted(set(got["core"]) - CLOSURE))
