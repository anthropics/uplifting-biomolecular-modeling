"""pytest session settings: the tests import BindCraft's modules from the vendored tree (stock/src/bindcraft/functions) and the package from
its checkout, so the interpreter writes no bytecode — the vendored tree stays upstream's bytes and nothing else; the session ends by asserting
no `__pycache__` under it."""
import os
import sys

import pytest

sys.dont_write_bytecode = True

VENDORED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "stock", "src")   # colabdesign/stock/src


@pytest.fixture(scope="session", autouse=True)
def no_bytecode_under_the_vendored_tree():
    yield
    if os.path.isdir(VENDORED):
        found = [os.path.join(r, d) for r, ds, _ in os.walk(VENDORED) for d in ds if d == "__pycache__"]
        assert not found, f"bytecode written under the vendored tree: {found}"
