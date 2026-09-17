import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(TREE, "opt"))


def run_routes_text(run_sh_src: str) -> str:
    """run.sh past its install block: the design / check / warm routes (the install step is the one place the launcher calls stock/check_pins.py)."""
    return run_sh_src.split('if [ "$VERB" = install ]; then', 1)[1].split("\n  exit 0\nfi\n", 1)[1]


@pytest.fixture
def tree():
    return TREE


@pytest.fixture
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("PXD_", "PXDESIGN_", "MODEL_OPT", "CUBLAS_WORKSPACE_CONFIG")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PXDESIGN_OPT_HOME", TREE)
    return TREE


@pytest.fixture
def fresh_stack(clean_env):
    from . import _stubs
    from pxdesign_opt import stack
    stack.reset_for_tests()
    _stubs.uninstall()
    yield stack
    stack.reset_for_tests()
    _stubs.uninstall()
    for k in list(os.environ):                     # switches exported by activate() during the test
        if k.startswith(("PXD_", "PXDESIGN_HOIST")):
            del os.environ[k]
