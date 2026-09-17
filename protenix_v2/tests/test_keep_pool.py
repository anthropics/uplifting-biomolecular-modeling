"""CPU tests for lever keep_pool (module protenix_ptx_keep_pool): caller classification, guard relay, and the pinned stock tree's empty_cache sites."""
import os, re, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
# workbench layout: tail/tests/ next to tail/levers/; kit layout: protenix_v2/tests/ next to opt/forward/flashpairformer/src/
for _cand in (os.path.join(HERE, "..", "levers"), os.path.join(HERE, "..", "opt", "forward", "flashpairformer", "src")):
    if os.path.isfile(os.path.join(_cand, "protenix_ptx_keep_pool.py")):
        sys.path.insert(0, _cand); break
import protenix_ptx_keep_pool as K  # noqa: E402

KIT = os.environ.get("KIT") or (os.path.abspath(os.path.join(HERE, "..")) if os.path.isfile(os.path.join(HERE, "..", "run.sh"))
                                else os.path.abspath(os.path.join(HERE, "../../../../../model-opt-release/protenix_v2")))
STOCK_SRC = os.path.join(KIT, "stock", "src")


def _fn_in(filename, lineno_pad=0, name="f"):
    """A function object whose code lives in `filename` that calls K.empty_cache()."""
    src = "\n" * lineno_pad + f"def {name}():\n    return EC()\n"
    g = {"EC": K.empty_cache}
    exec(compile(src, filename, "exec"), g)
    return g[name]


def setup_function(_):
    K._STATE.update({"installed": True, "orig": None, "skipped": {}, "passed": {}, "file_class": {}, "errors": 0})
    K._STATE["orig"] = lambda: K._STATE.__setitem__("released", K._STATE.get("released", 0) + 1)
    K._STATE["released"] = 0


def test_forward_sites_skipped_runner_sites_pass():
    for fn in ("/usr/local/lib/python3.11/site-packages/protenix/model/modules/confidence.py", STOCK_SRC + "/protenix/model/modules/confidence.py"):
        _fn_in(fn)()
    assert K._STATE["released"] == 0 and sum(K._STATE["skipped"].values()) == 2 and not K._STATE["passed"]
    assert all(k.startswith("protenix/model/modules/confidence.py:") for k in K._STATE["skipped"])
    for fn in ("/usr/local/lib/python3.11/site-packages/runner/inference.py", STOCK_SRC + "/runner/inference.py"):
        _fn_in(fn)()
    assert K._STATE["released"] == 2 and all(k.startswith("runner/inference.py:") for k in K._STATE["passed"])


def test_kit_and_torch_callers_pass_through():
    for fn in (KIT + "/opt/forward/flashpairformer/src/fpf_stackgraph/stackgraph.py",
               KIT + "/opt/forward/flashpairformer/src/infopt_graphs/protenix/graphed.py",   # a dir named protenix inside the kit is NOT stock
               "/usr/local/lib/python3.11/site-packages/torch/cuda/graphs.py",
               "/opt/tree/common/opt_core/opt_core/mem/allocator.py", "<stdin>"):
        _fn_in(fn, name="flush_if_safe")()
    assert K._STATE["released"] == 5 and not K._STATE["skipped"]


def test_guard_relay_is_transparent():
    """EmptyCacheGuard.__call__ (graphed.py) relaying a stock call -> skipped; relaying a kit call -> passed."""
    guard = _fn_in("/x/infopt_graphs/protenix/graphed.py", name="__call__")          # def __call__(): return EC()
    g_stock = {"G": guard}; exec(compile("def forward():\n    return G()\n", "/sp/site-packages/protenix/model/modules/confidence.py", "exec"), g_stock)
    g_kit = {"G": guard}; exec(compile("def capture():\n    return G()\n", "/kit/src/fpf_stackgraph/stackgraph.py", "exec"), g_kit)
    g_stock["forward"](); g_kit["capture"]()
    assert sum(K._STATE["skipped"].values()) == 1 and K._STATE["released"] == 1


def test_switch_values():
    assert K.enabled({}) is False and K.enabled({"PTX_KEEP_POOL": "0"}) is False and K.enabled({"PTX_KEEP_POOL": "1"}) is True
    try:
        K.enabled({"PTX_KEEP_POOL": "yes"}); raise AssertionError("expected refusal")
    except ValueError as e:
        assert "PTX_KEEP_POOL" in str(e)


def test_pinned_stock_tree_sites():
    """The inference-path empty_cache() sites of the pinned stock tree are exactly STOCK_SITES (loss.py / train.py are training-only); the
    swallowed subset is the model-forward (protenix/) sites."""
    assert K.SKIPPED_SITES == tuple(x for x in K.STOCK_SITES if x.startswith("protenix/")) and len(K.SKIPPED_SITES) == 3
    if not os.path.isdir(STOCK_SRC):
        import pytest; pytest.skip(f"stock tree not found at {STOCK_SRC}")
    found = []
    for rel in ("protenix/model/modules/confidence.py", "runner/inference.py", "protenix/model/protenix.py", "protenix/model/generator.py",
                "protenix/model/modules/pairformer.py", "protenix/model/sample_confidence.py", "runner/dumper.py"):
        p = os.path.join(STOCK_SRC, rel)
        for i, line in enumerate(open(p), 1):
            if re.search(r"torch\.cuda\.empty_cache\(\)", line) and not line.lstrip().startswith("#"):
                found.append(f"{rel}:{i}")
    assert sorted(found) == sorted(K.STOCK_SITES), found
