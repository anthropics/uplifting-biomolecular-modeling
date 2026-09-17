import os
import tempfile
import sys

import pytest

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="ef2inv-opt-test-xdg-")   # the design launch's weights digest memo (cli.WEIGHTS_CACHE, read at import) lands under a test tmp, never the real ~/.cache


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("EF2_", "EF2INV_", "ESMFOLD2_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF", raising=False)   # the ablation word (fastkit.LEVERS_OFF_VAR) is read from the process env by enable/launch: an ambient value in the developer's shell must not steer the suite — tests that exercise it set it themselves
    yield


def pytest_sessionfinish(session, exitstatus):
    from . import _paths as P
    for r, dirs, fns in os.walk(P.FWD):
        assert "__pycache__" not in dirs, f"bytecode written under the kit tree: {r}"
