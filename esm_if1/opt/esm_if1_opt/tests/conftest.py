"""The package's CPU tests run from the tree: MODEL_OPT names the esm_if1/ directory (run.sh's variable) so tree resolution never depends on cwd."""
import os

import pytest

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))     # esm_if1/


@pytest.fixture(autouse=True)
def _tree_env(monkeypatch):
    monkeypatch.setenv("MODEL_OPT", TREE)
    monkeypatch.delenv("ESM_IF1_OPT", raising=False)
    yield
