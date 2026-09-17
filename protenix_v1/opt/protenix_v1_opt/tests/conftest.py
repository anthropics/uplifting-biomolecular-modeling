"""Shared paths for the package's CPU tests: the tree, opt/, the kit directory (opt/forward/v05_addon), stock/PINS.json. No torch and no GPU stack;
the pinned stock wheel installed with `--no-deps` (its metadata and pure-Python modules feed the stock-version and frozen-weights gates
the `pred` tests drive)."""
import json
import shutil
import tempfile
import os
import sys

import pytest

sys.dont_write_bytecode = True                                   # bytecode caches are not files of the kit (test_kit_carry walks it)
OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TREE = os.path.dirname(OPT)
KIT = os.path.join(OPT, "forward", "v05_addon")
PINS = os.path.join(TREE, "stock", "PINS.json")


def pins() -> dict:
    with open(PINS, encoding="utf-8") as fh:
        return json.load(fh)


def n_kit_files() -> int:
    """The kit directory's live file count under the package's tree rule (kit.tree_files): the number the `check` gate detail
    reports. The tree is identified by the git commit carrying it; no count is pinned."""
    from protenix_v1_opt import kit as K
    return len(K.tree_files(KIT))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    from opt_core import kernels as KR
    from protenix_v1_opt import stack
    exports = {v for name in stack.ROUTED_KERNELS for v in (KR.sums(name).get("exports") or {})}   # the routed kernels' cell-table exports (route_kernels sets them process-wide)
    for k in ("PROTENIX_V1_OPT", "PROTENIX_V1_OPT_DET", "PROTENIX_V1_OPT_KIT", "MODEL_OPT", "PYTHONPATH", *sorted(exports)):
        monkeypatch.delenv(k, raising=False)
    from protenix_v1_opt import kit as K
    memo_root = tempfile.mkdtemp(prefix="ptx1-digest-memo-")            # the weights digest memo of a test process: a fresh cache root per test (kit.digest_memo_dir)
    monkeypatch.setenv(K.DIGEST_MEMO_ENV, memo_root)
    monkeypatch.delenv(K.WEIGHTS_MEMO_ENV, raising=False)
    monkeypatch.setattr(K, "_DIGESTS", {})
    yield
    shutil.rmtree(memo_root, ignore_errors=True)


@pytest.fixture(autouse=True)
def weights_root(tmp_path_factory, monkeypatch):
    """A weights root for the weights boot gate (kit.frozen_weights_check): a small stand-in checkpoint (present = accepted; its digest is
    not the pin's, so the gate names it NOT PINNED and proceeds) and the data caches under <root>/common, PROTENIX_ROOT_DIR
    (stock.root_env) pointing at it — every test that runs `pred` or the gates passes the gate; test_frozen_weights.py sets its own roots."""
    st = pins()["stock"]; root = tmp_path_factory.mktemp("weights")
    ck = root / st["checkpoint"]; ck.parent.mkdir(parents=True, exist_ok=True)
    if not ck.exists():
        with open(ck, "wb") as fh:
            fh.truncate(4096)
    (root / "common").mkdir(exist_ok=True)
    try:
        import importlib.util
        from protenix_v1_opt import kit as _K
        p = _K.stock_url_module()
        if os.path.isfile(p):
            ms = importlib.util.spec_from_file_location("u", p); m = importlib.util.module_from_spec(ms); ms.loader.exec_module(m)
            for n in st["data_files"]:
                if n in m.URL: (root / "common" / os.path.basename(m.URL[n])).write_bytes(b"x")
    except Exception:
        pass                                                         # without the stock package the gate refuses by name (test_frozen_weights covers it)
    monkeypatch.setenv(st["root_env"], str(root))
    yield str(root)


@pytest.fixture(autouse=True)
def _entry_gate_isolated(monkeypatch):
    """The in-process tests drive cli.main / enable() against whatever opt_core the test interpreter imports; the entry pin gate
    (_core_gate.gate: is THAT core the one opt/pyproject.toml pins?) is exercised through the real routes in test_core_gate.py, with cores
    built to match or not — here it is a recorded no-op so a unit test never depends on the sibling core's pin state."""
    from protenix_v1_opt import _core_gate
    calls = []
    monkeypatch.setattr(_core_gate, "gate", lambda anchor, tag=None, stream=None: calls.append((anchor, tag)) or {"pinned": None, "installed": None, "tag": tag})
    return calls
