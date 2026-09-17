"""keep_pool: torch.cuda.empty_cache under the lever — the stock model's in-forward sites become counted no-ops, the runner's and the kit's own
callers pass through to the previous callable; caller identification walks past the stock cleanup relay (opendde/utils/torch_utils.py); the
pinned sites exist in the stock tree; install / uninstall on a stand-in torch."""
import os
import sys
import types

import pytest

from opendde_opt import keeppool

TREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(TREE, "stock", "src")


def _call_from(path: str, lineno: int, fn, relay: str | None = None):
    """Execute `fn()` from a frame whose co_filename is `path` at line `lineno` — directly, or through two relay frames compiled under the
    file name `relay` (the stock chain: site -> cleanup_device_memory -> _clear_accelerator_cache -> empty_cache, both relays in torch_utils.py)."""
    target = fn
    if relay:
        ns = {}
        exec(compile("\n" * 9 + "def relay2(f):\n    return f()\n\ndef relay1(f):\n    return relay2(f)\n", relay, "exec"), ns)
        target = ns["relay1"]
    ns = {}
    exec(compile("\n" * (lineno - 2) + "def site(t, f):\n    return t(f) if t is not f else f()\n", path, "exec"), ns)
    return ns["site"](target, fn)


@pytest.fixture
def fake_torch(monkeypatch):
    calls = {"n": 0}
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    memory = types.ModuleType("torch.cuda.memory")
    def stock_empty_cache():
        calls["n"] += 1
        return "released"
    cuda.empty_cache = stock_empty_cache
    memory.empty_cache = stock_empty_cache
    cuda.memory = memory
    torch.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda)
    monkeypatch.setitem(sys.modules, "torch.cuda.memory", memory)
    monkeypatch.setitem(keeppool.STATS, "installed", False)
    monkeypatch.setitem(keeppool.STATS, "skipped", {})
    monkeypatch.setitem(keeppool.STATS, "passed", {})
    monkeypatch.setitem(keeppool.STATS, "errors", 0)
    keeppool.install()
    yield torch, calls
    keeppool.uninstall()
    assert cuda.empty_cache is stock_empty_cache and memory.empty_cache is stock_empty_cache


def test_model_sites_skipped_runner_and_kit_sites_passed(fake_torch):
    torch, calls = fake_torch
    ec = torch.cuda.empty_cache
    assert getattr(ec, "_keep_pool", False) and torch.cuda.memory.empty_cache is ec
    model = os.path.join(SRC, "opendde", "model", "opendde.py")
    conf = os.path.join(SRC, "opendde", "model", "modules", "confidence.py")
    relay = os.path.join(SRC, "opendde", "utils", "torch_utils.py")
    runner = os.path.join(SRC, "runner", "inference.py")
    assert _call_from(model, 580, ec) is None                                   # direct in-forward site -> no-op
    assert _call_from(conf, 298, ec, relay=relay) is None                       # through cleanup_device_memory -> the confidence head decides -> no-op
    assert _call_from(runner, 775, ec, relay=relay) == "released"               # the runner through the same relay -> passes
    assert _call_from(runner, 1653, ec) == "released"
    assert _call_from("/kit/opt/forward/fast_inference/levers/OFFLOAD/odde_offload.py", 40, ec) == "released"   # the kit's own units pass
    assert ec() == "released"                                                    # this test file passes
    assert calls["n"] == 4
    st = keeppool.kit_stats()
    assert st["skipped"] == 2 and st["passed"] == 4 and st["errors"] == 0
    assert set(st["skipped_sites"]) == {"opendde/model/opendde.py:580", "opendde/model/modules/confidence.py:298"}
    assert "runner/inference.py:775" in st["passed_sites"] and "runner/inference.py:1653" in st["passed_sites"]
    assert keeppool.fallbacks(["keep_pool"]) == []
    torch.cuda.empty_cache = lambda: None                                        # re-bound behind our back -> named
    assert keeppool.fallbacks(["keep_pool"]) and "re-bound" in keeppool.fallbacks(["keep_pool"])[0]
    torch.cuda.empty_cache = ec


def test_registered_wrappers_are_looked_through(fake_torch):
    torch, calls = fake_torch
    ec = torch.cuda.empty_cache
    def other_units_wrapper():
        return ec()
    keeppool.register_wrapper(other_units_wrapper)
    model = os.path.join(SRC, "opendde", "model", "opendde.py")
    assert _call_from(model, 1598, other_units_wrapper) is None
    assert keeppool.kit_stats()["skipped_sites"] == {"opendde/model/opendde.py:1598": 1}
    keeppool.WRAPPER_CODES.discard(other_units_wrapper.__code__)


def test_pinned_sites_exist_in_the_stock_tree():
    """Every SKIPPED / PASSED site names a real empty_cache / cleanup line of the pinned stock (a moved site fails here, before a GPU run)."""
    for site in keeppool.SKIPPED_SITES + keeppool.PASSED_SITES:
        rel, lineno = site.rsplit(":", 1)
        with open(os.path.join(SRC, rel)) as fh:
            lines = fh.readlines()
        i = int(lineno) - 1
        window = "".join(lines[max(0, i - 12): i + 3])
        assert ("empty_cache" in window) or ("cleanup_device_memory" in window), site
    # and the census is complete for the single-GPU route: every direct empty_cache in opendde/model/*.py outside a Fold-CP branch is pinned
    import re
    found = []
    for rel in ("opendde/model/opendde.py", "opendde/model/modules/confidence.py"):
        src = open(os.path.join(SRC, rel)).read().splitlines()
        for n, line in enumerate(src, 1):
            if re.search(r"^\s*torch\.cuda\.empty_cache\(\)", line) or re.search(r"^\s*cleanup_device_memory\($", line) or re.search(r"^\s*cleanup_device_memory\(.*collect_garbage=False\)", line):
                found.append(f"{rel}:{n}")
    pinned = set(keeppool.SKIPPED_SITES)
    foldcp_only = {"opendde/model/opendde.py:404", "opendde/model/modules/confidence.py:609", "opendde/model/modules/confidence.py:637"}   # Fold-CP mesh branches (read at the pin)
    for s in found:
        head = s if s in pinned or s in foldcp_only else None
        near = any(abs(int(s.rsplit(":", 1)[1]) - int(p.rsplit(":", 1)[1])) <= 3 and s.split(":")[0] == p.split(":")[0] for p in pinned | foldcp_only)
        assert head or near, f"unpinned empty_cache site {s}"


def test_registry_row_and_lines():
    from opendde_opt import modes, registry
    if "keep_pool" not in registry.LEVERS:
        pytest.skip("keep_pool not wired in this tree")
    lv = registry.LEVERS["keep_pool"]
    assert (lv.kit, lv.tier, lv.file) == (registry.HOUSE, "exact", "opendde_opt/keeppool.py")
    for ln in ("S1", "LSTAR2A"):
        assert "keep_pool" in modes.LINES[ln].levers
    assert registry.validate() == []
