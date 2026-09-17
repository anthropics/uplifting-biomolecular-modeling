"""CPU tests for lever keep_pool (lib/ptx1_keep_pool.py): caller classification, the sampler-graph guard relay, and the pinned stock wheel's
empty_cache sites (stock/protenix-1.1.0-py3-none-any.whl: the census STOCK_SITES restates). No GPU, no torch."""
import glob
import os
import re
import sys
import zipfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                       # protenix_v1/
LIB = os.path.join(KIT, "opt", "forward", "v05_addon", "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)
import ptx1_keep_pool as KP  # noqa: E402


def _fn_in(filename, name="f"):
    """A function object whose code lives in `filename` and calls KP.empty_cache()."""
    g = {"EC": KP.empty_cache}
    exec(compile(f"def {name}():\n    return EC()\n", filename, "exec"), g)
    return g[name]


@pytest.fixture(autouse=True)
def _armed():
    saved = dict(KP._STATE)
    KP._STATE.update({"installed": True, "skipped": {}, "passed": {}, "file_class": {}, "errors": 0, "released": 0})
    KP._STATE["orig"] = lambda: KP._STATE.__setitem__("released", KP._STATE["released"] + 1)
    yield
    KP._STATE.clear(); KP._STATE.update(saved)


def test_forward_sites_skipped_runner_sites_pass():
    for fn in ("/usr/local/lib/python3.11/site-packages/protenix/model/modules/confidence.py", "/w/stock/src/protenix/model/modules/confidence.py"):
        _fn_in(fn)()
    assert KP._STATE["released"] == 0 and sum(KP._STATE["skipped"].values()) == 2 and not KP._STATE["passed"]
    assert all(k.startswith("protenix/model/modules/confidence.py:") for k in KP._STATE["skipped"])
    for fn in ("/usr/local/lib/python3.11/site-packages/runner/inference.py", "/w/stock/src/runner/inference.py"):
        _fn_in(fn)()
    assert KP._STATE["released"] == 2 and all(k.startswith("runner/inference.py:") for k in KP._STATE["passed"])
    r = KP.report()
    assert (r["skipped_total"], r["passed_total"], r["errors"], r["installed"]) == (2, 2, 0, True)


def test_kit_core_and_torch_callers_pass_through():
    for fn in (KIT + "/opt/forward/v05_addon/lib/kit112_src/infopt_graphs/protenix/graphed.py",      # a dir named protenix inside the kit is NOT stock
               KIT + "/opt/protenix_v1_opt/big.py", "/usr/local/lib/python3.11/site-packages/torch/cuda/graphs.py",
               "/opt/tree/common/opt_core/opt_core/mem/allocator.py", "<stdin>"):
        _fn_in(fn, name="flush_if_safe")()
    assert KP._STATE["released"] == 5 and not KP._STATE["skipped"]


def test_guard_relay_is_transparent():
    """EmptyCacheGuard.__call__ (graphed.py) relaying a stock call -> skipped; relaying a kit call -> passed."""
    guard = _fn_in("/x/infopt_graphs/protenix/graphed.py", name="__call__")
    g_stock = {"G": guard}; exec(compile("def forward():\n    return G()\n", "/sp/site-packages/protenix/model/modules/confidence.py", "exec"), g_stock)
    g_kit = {"G": guard}; exec(compile("def release():\n    return G()\n", "/kit/opt/protenix_v1_opt/big.py", "exec"), g_kit)
    g_stock["forward"](); g_kit["release"]()
    assert sum(KP._STATE["skipped"].values()) == 1 and KP._STATE["released"] == 1
    w = _fn_in("/kit/some/other_wrapper.py", name="wrapped")                    # a registered transparent wrapper is looked through the same way
    KP.register_wrapper(w)
    g2 = {"G": w}; exec(compile("def forward():\n    return G()\n", "/sp/site-packages/protenix/model/modules/confidence.py", "exec"), g2)
    g2["forward"]()
    assert sum(KP._STATE["skipped"].values()) == 2


def test_pinned_stock_wheel_sites():
    """The inference-path empty_cache() sites of the pinned stock wheel are exactly STOCK_SITES (loss.py / train.py are training-only); the
    swallowed subset is the model-forward (protenix/) sites."""
    assert KP.SKIPPED_SITES == tuple(x for x in KP.STOCK_SITES if x.startswith("protenix/")) and len(KP.SKIPPED_SITES) == 3
    wheels = glob.glob(os.path.join(KIT, "stock", "protenix-*.whl"))
    if not wheels:
        pytest.skip("stock wheel not found under stock/")
    z = zipfile.ZipFile(wheels[0])
    found = []
    for rel in ("protenix/model/modules/confidence.py", "runner/inference.py", "protenix/model/protenix.py", "protenix/model/generator.py",
                "protenix/model/modules/pairformer.py", "protenix/model/sample_confidence.py", "runner/dumper.py"):
        for i, line in enumerate(z.read(rel).decode("utf-8").splitlines(), 1):
            if re.search(r"torch\.cuda\.empty_cache\(\)", line) and not line.lstrip().startswith("#"):
                found.append(f"{rel}:{i}")
    assert sorted(found) == sorted(KP.STOCK_SITES), found
