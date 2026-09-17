"""torch-dependent tests of opt_core.host (skipped where torch is absent; the CUDA ones where no device is present): cold start
totality and equality with stock construction, mmap load, tensor digests, to_host on device trees with and without a pinned pool."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from opt_core.host import HostRefused                                  # noqa: E402
from opt_core.host.coldstart import (load_total, materialize, meta_init, mmap_state_dict, no_init,  # noqa: E402
                                     state_digest)
from opt_core.host.memo import digest                                  # noqa: E402
from opt_core.host.outputs import PinnedPool, to_host                  # noqa: E402


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)
        self.emb = nn.Embedding(5, 8)
        self.norm = nn.LayerNorm(4)
        self.register_buffer("table", torch.arange(4, dtype=torch.float32))        # computed, not an init call; persistent
        self.register_buffer("scratch", torch.zeros(2), persistent=False)          # not in the checkpoint


def test_no_init_then_total_load_equals_stock(tmp_path):
    torch.manual_seed(0)
    stock = Small()
    sd = stock.state_dict()
    with no_init() as c:
        cold = Small()
    assert c.n >= 4                                                      # Linear (2), Embedding (1), LayerNorm (2) fillers skipped
    rep = load_total(cold, sd, skipped_inits=c.n)
    assert rep.missing == [] and rep.keys == len(sd)
    assert state_digest(cold) == state_digest(stock)
    assert rep.active_line("kit").startswith("[kit] LEVER name=F6.weights_residency_init state=on impl=host.coldstart origin=core form=no_init skipped_inits=")
    partial = {k: v for k, v in sd.items() if k != "lin.bias"}
    with no_init():
        cold2 = Small()
    with pytest.raises(HostRefused) as ei:
        load_total(cold2, partial)
    assert ei.value.event.reason == "missing_keys" and ei.value.event.fields["first"] == "lin.bias"
    assert nn.init.kaiming_uniform_.__name__ == "kaiming_uniform_" and not hasattr(nn.init.kaiming_uniform_, "__wrapped_noop__")
    fresh = nn.Linear(2, 2)                                              # init restored after the context
    assert fresh.weight.abs().sum().item() > 0


def test_meta_init_materialize_refuses_uncovered_tensors():
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 1):
        pytest.skip("assign=True needs torch >= 2.1")
    stock = Small()
    sd = stock.state_dict()
    with meta_init():
        m = Small()
    assert m.lin.weight.device.type == "meta"
    with pytest.raises(HostRefused) as ei:                               # 'scratch' is non-persistent: left on meta → named refusal
        materialize(m, sd)
    assert ei.value.event.reason == "left_on_meta"

    class Covered(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 4)
    ref = Covered()
    with meta_init():
        c = Covered()
    rep = materialize(c, ref.state_dict())
    assert rep.form == "meta" and state_digest(c) == state_digest(ref)


def test_mmap_state_dict_roundtrip_and_refusal(tmp_path):
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 1):
        pytest.skip("mmap needs torch >= 2.1")
    m = Small()
    p = tmp_path / "w.pt"
    torch.save(m.state_dict(), str(p))
    sd = mmap_state_dict(str(p))
    with no_init() as c:
        cold = Small()
    load_total(cold, sd, skipped_inits=c.n)
    assert state_digest(cold) == state_digest(m)
    junk = tmp_path / "nonzip.bin"
    junk.write_bytes(b"not a zip")
    with pytest.raises(HostRefused) as ei:
        mmap_state_dict(str(junk))
    assert ei.value.event.reason == "not_zipfile"


def test_tensor_digest_by_content():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert digest(a) == digest(a.clone())
    assert digest(a) != digest(a.double())
    assert digest(a.t()) == digest(a.t().contiguous())
    b = torch.zeros(3, dtype=torch.bfloat16)
    assert len(digest(b)) == 64                                          # dtypes numpy lacks still digest (byte view)


def test_to_host_cpu_tree_is_passthrough_for_cpu_tensors():
    t = torch.ones(2)
    out = to_host({"x": t, "y": [t, 3]})
    assert out["x"] is t and out["y"][0] is t and out["y"][1] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_to_host_device_tree_with_and_without_pool():
    dev = torch.device("cuda")
    tree = {"plddt": torch.rand(3, 50, device=dev), "pae": [torch.rand(50, 50, device=dev, dtype=torch.float16)], "n": 5,
            "flag": torch.tensor([True, False], device=dev)}
    stats = {}
    out = to_host(tree, stats=stats)
    assert out["n"] == 5 and out["plddt"].device.type == "cpu" and torch.equal(out["plddt"], tree["plddt"].cpu())
    assert torch.equal(out["pae"][0], tree["pae"][0].cpu()) and out["flag"].tolist() == [True, False]
    assert stats["leaves"] == 3 and stats["unpinned"] == 3
    pool = PinnedPool(1 << 20)
    stats = {}
    out2 = to_host(tree, pool=pool, stats=stats)
    assert torch.equal(out2["pae"][0], out["pae"][0]) and stats["pinned"] == 3
    leases = []
    out3 = to_host(tree, pool=pool, leases=leases, stats={})
    assert len(leases) == 3 and all(b.is_pinned() for b in leases) and torch.equal(out3["plddt"], out["plddt"])
    for b in leases:
        pool.release(b)
    t = pool.tally()
    assert t["reuses"] >= 3 and t["unpinned"] == 0
    small = PinnedPool(1 << 12)
    to_host({"pae": tree["pae"]}, pool=small, stats=stats)               # 5000 B > 4 KiB budget → unpinned, counted, named
    assert small.tally()["unpinned"] == 1 and small.events[0].reason == "over_budget"


def test_pinned_pool_without_cuda_refuses_by_name():
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(HostRefused) as ei:
        PinnedPool(1 << 20)
    assert ei.value.event.reason == "no_cuda"
