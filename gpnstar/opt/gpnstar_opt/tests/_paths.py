"""Where the tree is, for the tests that run run.sh / check_pins.py / read PINS.json (skipped when the package is not inside a gpnstar/ tree)."""
import os
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../opt/gpnstar_opt
OPT = os.path.dirname(PKG)                                                  # .../opt
TREE = os.path.dirname(OPT)                                                 # .../gpnstar
RUN_SH = os.path.join(TREE, "run.sh")
PINS_JSON = os.path.join(TREE, "stock", "PINS.json")
CHECK_PINS = os.path.join(TREE, "stock", "check_pins.py")
PY = sys.executable


def tree_or_skip():
    if not (os.path.isfile(RUN_SH) and os.path.isfile(PINS_JSON)):
        pytest.skip("the package is not inside a gpnstar/ tree (run.sh, stock/PINS.json)")
    return TREE


def env_clean(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPNSTAR_OPT")}
    env.update(extra)
    return env
