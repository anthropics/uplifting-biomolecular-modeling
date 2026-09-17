import os
import sys

import pytest

from .. import mem as _mem
from .. import stack as _stack
from .. import stock_fold as _stock_fold
from . import _stubs

@pytest.fixture(autouse=True)
def _clean_process(monkeypatch):
    """Every test starts with no mode active, none of the kits' names in the environment (stock/PINS.json's must-be-absent table, the one
    list stock_fold strips), and no rf3 module imported."""
    _stack.STATE["report"] = None
    prefixes = _stock_fold.must_be_absent(_stack.pins())
    for k in list(os.environ):
        if k.startswith(tuple(prefixes)) or k in _mem.EXPORTS:                    # the memory policy's exports too
            monkeypatch.delenv(k, raising=False)
    for m in [m for m in sys.modules if m == "rf3" or m.startswith("rf3.")]:
        sys.modules.pop(m, None)
    for f in list(sys.meta_path):
        if type(f).__name__ == "_ImportWatch":
            sys.meta_path.remove(f)
    yield
    _stack.STATE["report"] = None
    for m in [m for m in sys.modules if m == "rf3" or m.startswith("rf3.")]:
        sys.modules.pop(m, None)


@pytest.fixture
def patched_tree(tmp_path, monkeypatch):
    """A stub tree in the patched state on sys.path with a vcs dist record and a fake GPU: the activation fixture of every test that
    enables a kit row (test_activation.py, test_mem.py, test_big.py)."""
    root = _stubs.make_tree(str(tmp_path / "sp"), "patched")
    monkeypatch.syspath_prepend(root)
    _stubs.make_dist(root, route="vcs")
    import importlib
    importlib.invalidate_caches()
    monkeypatch.setattr(_stack, "gpu_info", lambda: dict(_stubs.FAKE_GPU))
    return root
