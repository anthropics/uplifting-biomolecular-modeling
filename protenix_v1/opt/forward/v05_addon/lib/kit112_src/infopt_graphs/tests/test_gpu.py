"""GPU tests for infopt_graphs (run on any CUDA device): capture/replay parity, RNG guard, sync census, launch counter,
autocast weight-cast cache safety.  `python -m pytest infopt_graphs/tests/test_gpu.py -q` or `python test_gpu.py`."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from infopt_graphs import GraphedFunction, StaticGraph, SyncCensus, LaunchCounter, RNGGuard, rng_fingerprint, audit_capture_safety, signature_of


class Block(nn.Module):
    def __init__(self, c=64, n=4):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.lin = nn.ModuleList([nn.Linear(c, c) for _ in range(n)])

    def forward(self, x):
        for l in self.lin:
            x = x + F.gelu(l(self.ln(x)))
        return x


def test_graphed_function_parity_and_replay():
    torch.manual_seed(0)
    m = Block().cuda().eval()
    g = GraphedFunction(m.forward, name="blk", family="t")
    x = torch.randn(8, 128, 64, device="cuda")
    with torch.no_grad():
        ref = m(x)
        y1 = g(x)  # warm-up (== eager)
        y2 = g(x)  # replay
        assert torch.equal(y1, ref), "warm-up output must be the eager output"
        assert torch.equal(y2, ref), f"replay differs from eager: max|d|={(y2-ref).abs().max().item()}"
        x2 = torch.randn_like(x)
        assert torch.equal(g(x2), m(x2)), "replay with new inputs must equal eager on those inputs"
        x3 = torch.randn(8, 96, 64, device="cuda")  # new signature
        assert torch.equal(g(x3), m(x3))
    s = g.summary()
    assert s["stats"]["captures"] == 2 and s["stats"]["replay"] == 2, s["stats"]
    assert all(e.capture_s >= 0 for e in g.entries.values())
    # sync census of warm-up (pass 2) must be 0 for this pure-GPU block
    ev = [e for e in g.events if e["event"] == "captured"][0]
    assert ev["audit"]["pass2"]["syncs"]["total_syncs"] == 0, ev["audit"]
    print("parity ok", s["stats"], "pool_mb", s["memory"])


def test_autocast_cache_safety():
    """Capture under torch.autocast and replay AFTER the outer autocast context has exited (the weight-cast cache is gone)."""
    torch.manual_seed(0)
    m = Block(c=128).cuda().eval()
    g = GraphedFunction(m.forward, name="ac", family="t2")
    x = torch.randn(4, 256, 128, device="cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = m(x)
            y1 = g(x)
            y2 = g(x)
        torch.cuda.synchronize()
        # fresh autocast context (new cache) + garbage-collect: replay must still match
        junk = [torch.randn(1024, 1024, device="cuda") for _ in range(8)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y3 = g(x)
            ref2 = m(x)
    assert torch.equal(y1, ref) and torch.equal(y2, ref) and torch.equal(y3, ref2), "autocast replay mismatch"
    print("autocast cache safety ok")


def test_rng_guard_refuses_dropout():
    m = nn.Sequential(nn.Linear(32, 32), nn.Dropout(0.5)).cuda().train()
    g = GraphedFunction(m.forward, name="drop", family="t3")
    x = torch.randn(4, 32, device="cuda")
    y = g(x)
    assert g.stats.get("eager_unsupported", 0) == 1 or any(e["event"] == "warmup_failed" for e in g.events), g.events
    assert "RNGGuard" in g.events[0]["error"], g.events[0]["error"]
    print("rng guard ok:", g.events[0]["error"][:80])


def test_sync_census_counts_item():
    x = torch.randn(100, device="cuda")
    with SyncCensus(mode="warn") as c:
        for _ in range(3):
            _ = x.sum().item()
        _ = x.cpu()
    r = c.report()
    assert r["total_syncs"] >= 4, r
    assert any("test_gpu.py" in s for s, n in r["by_site"]), r["by_site"]
    print("census ok:", r["total_syncs"], r["by_site"][:2])
    # error mode raises
    raised = False
    try:
        with SyncCensus(mode="error"):
            _ = x.sum().item()
    except RuntimeError:
        raised = True
    assert raised


def test_launch_counter_graph_vs_eager():
    m = Block(c=64, n=8).cuda().eval()
    x = torch.randn(2, 64, 64, device="cuda")
    g = GraphedFunction(m.forward, name="lc", family="t4")
    with torch.no_grad():
        g(x); g(x); torch.cuda.synchronize()
        with LaunchCounter() as e:
            m(x); torch.cuda.synchronize()
        with LaunchCounter() as r:
            g(x); torch.cuda.synchronize()
    es, rs = e.summary(), r.summary()
    assert es["kernel_launches"] > 20, es
    assert rs["graph_launches"] >= 1 and rs["kernel_launches"] < es["kernel_launches"] / 4, (es, rs)
    print("launches eager", es["kernel_launches"], "graphed", rs["kernel_launches"], "graph_launches", rs["graph_launches"])


def test_static_graph_inplace_loop():
    """A diffusion-like loop: x <- x + f(x) in place, noise drawn outside the graph (stock order preserved)."""
    torch.manual_seed(1)
    m = Block(c=32, n=2).cuda().eval()
    x = torch.randn(4, 32, device="cuda")
    eps = torch.zeros_like(x)
    xs = x.clone()

    def body():
        xs.copy_(xs + 0.1 * m(xs) + eps)
        return xs

    gen = torch.Generator(device="cuda").manual_seed(7)
    with torch.no_grad():
        sg = StaticGraph(name="loop").capture(body)  # warm-up executed body once on xs (== step 0); then capture
        # reference: eager loop from the same x with the same noise sequence
        xr = x.clone()
        xr = xr + 0.1 * m(xr) + torch.zeros_like(x)  # step 0 (eps=0 at warm-up)
        for i in range(5):
            e = torch.randn(x.shape, device="cuda", generator=gen)
            eps.copy_(e); sg.replay()
            xr = xr + 0.1 * m(xr) + e
    assert torch.allclose(xs, xr, atol=0, rtol=0) or (xs - xr).abs().max().item() < 1e-5, (xs - xr).abs().max().item()
    assert sg.info["audit"]["syncs"]["total_syncs"] == 0, sg.info
    print("static graph loop ok, max|d|", (xs - xr).abs().max().item(), sg.info)


def test_pool_reuse_after_last_graph_dies():
    """Regression (bug #2 of the Protenix integration): a shared graph_pool_handle() reused for a NEW capture after the last graph
    captured into it was freed trips `CUDACachingAllocator ... use_count > 0 INTERNAL ASSERT` inside capture_begin, which leaves the
    CUDA default generator flagged 'capturing' so every later torch.randn on CUDA raises 'Offset increment outside graph capture'.
    The PoolRegistry hands out a fresh handle once the live-graph count of a family reaches 0; this test evicts every graph of a
    family (max_entries=1 -> each new signature evicts the previous graph), captures again, and checks torch.randn still works."""
    from infopt_graphs.core import POOLS
    m = Block(c=32, n=2).cuda().eval()
    g = GraphedFunction(m.forward, name="pool", family="pool_test", max_entries=1)
    with torch.no_grad():
        for n in (16, 24, 32, 40):          # 4 signatures, 1 live graph at a time -> 3 evictions, 3 re-acquisitions
            x = torch.randn(2, n, 32, device="cuda")
            assert torch.equal(g(x), m(x)); assert torch.equal(g(x), m(x))
            _ = torch.randn(8, device="cuda")   # generator must not be stuck in capture mode
    assert g.stats["captures"] == 4 and g.stats["evicted"] == 3, g.stats
    assert POOLS.live.get("pool_test", 0) == 1, POOLS.summary()
    g.clear(); assert POOLS.live.get("pool_test", 0) == 0
    _ = torch.randn(8, device="cuda")
    print("pool reuse ok", g.stats, POOLS.summary())


if __name__ == "__main__":
    for f in [test_graphed_function_parity_and_replay, test_autocast_cache_safety, test_rng_guard_refuses_dropout, test_sync_census_counts_item,
              test_launch_counter_graph_vs_eager, test_static_graph_inplace_loop, test_pool_reuse_after_last_graph_dies]:
        f()
    print("ALL GPU TESTS PASSED")
