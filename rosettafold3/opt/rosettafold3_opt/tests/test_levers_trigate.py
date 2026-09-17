"""triatt_chunk as the fused triangle attention's size gate (levers.tri_forward_chunked): a pair of at most `gate` tokens goes to the class
forward found at install (the FPF add-on's fused kernel), a larger pair steps past it BY NAME to the row-block statement on stock's parts."""
import sys
import types

from rosettafold3_opt import levers


class _Pair:
    def __init__(self, n, c=128):
        self.shape = (1, n, n, c)

    def dim(self):
        return 4


def test_gate_dispatch_marks_and_routes(monkeypatch):
    calls = []
    monkeypatch.setitem(levers.STATE, "tri_rows", 512)
    monkeypatch.setitem(levers.STATE, "tri_gate", 2048)
    monkeypatch.setitem(levers.STATE, "tri_owner", True)
    monkeypatch.setitem(levers.STATE, "orig", dict(levers.STATE.get("orig") or {}, tri_forward=lambda self, pair: calls.append(("gflash", pair.shape[1])) or "fused"))
    adapter = types.ModuleType("fpf_rf3_adapter"); adapter._ORIG = {"tri_forward": lambda self, pair: calls.append(("stock", pair.shape[1])) or "stock"}
    monkeypatch.setitem(sys.modules, "fpf_rf3_adapter", adapter)
    A = types.ModuleType("rf3.model.layers.attention"); A.SHOULD_USE_CUEQUIVARIANCE = False        # the vanilla route: above the gate the STOCK forward runs (named fallback), never the kernel
    monkeypatch.setitem(sys.modules, "rf3.model.layers.attention", A)
    marks = []
    monkeypatch.setattr(levers, "mark", lambda lever, detail=None: marks.append(detail))
    monkeypatch.setattr(levers, "fallback", lambda lever, reason: marks.append("fallback"))
    m = types.SimpleNamespace(use_cuequivariance=True, start_node=True)
    assert levers.tri_forward_chunked(m, _Pair(1200)) == "fused" and calls[-1] == ("gflash", 1200) and marks[-1] == "gflash:N<=2048"
    assert levers.tri_forward_chunked(m, _Pair(2048)) == "fused" and calls[-1] == ("gflash", 2048)
    assert levers.tri_forward_chunked(m, _Pair(3000)) == "stock" and calls[-1] == ("stock", 3000) and "gate:N=3000>2048:gflash-steps-aside" in marks
    monkeypatch.setitem(levers.STATE, "tri_gate", 0)                                               # gate 0 under a gflash arm (big today): the kernel steps aside at EVERY size, stock's parts run
    assert levers.tri_forward_chunked(m, _Pair(1200)) == "stock" and calls[-1] == ("stock", 1200) and "gate:N=1200>0:gflash-steps-aside" in marks
    monkeypatch.setitem(levers.STATE, "tri_owner", False)                                          # no gflash in the arm: the gate is moot — the class forward below rows / vanilla
    assert levers.tri_forward_chunked(m, _Pair(3000)) == "fused" and calls[-1] == ("gflash", 3000)   # (orig here stands for whatever forward the class had)


def test_the_gate_has_no_per_card_row_today():
    """big.tri_gflash_gate(rep['gpu']): no per-card row -> TRI_GFLASH_GATE (0: the row-block path at every size) on every device (the fused path's
    ~3072*N^2-byte transient sits above the memory row's peak line at every size, so no row); an unknown / absent probe also reads the default."""
    from rosettafold3_opt import big
    assert big.TRI_GFLASH_GATE == 0 and big.TRI_GFLASH_GATE_ROWS == {}
    assert big.tri_gflash_gate({"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": 81559}) == 0
    assert big.tri_gflash_gate({"name": "NVIDIA A100-SXM4-80GB", "cc": (8, 0), "memory_mib": "81920"}) == 0
    assert big.tri_gflash_gate(None) == 0 and big.tri_gflash_gate({}) == 0


def test_big_sets_the_transition_pack_residency_to_lru():
    """The memory row asks the shared core's transition provider to hold packed weights LRU (xtr binds the provider under the EXACT word, whose
    default residency is per layer): big.set_pack_residency calls pack_residency('lru'); an older core without the call is named, not fatal."""
    from rosettafold3_opt import big
    calls = []
    class TR:
        @staticmethod
        def pack_residency(word): calls.append(word)
    assert big.set_pack_residency(TR) == "lru" and calls == ["lru"] and big.PACK_RESIDENCY == "lru"
    class Old: pass
    assert big.set_pack_residency(Old).startswith("absent:")
    class Bad:
        @staticmethod
        def pack_residency(word): raise RuntimeError("x")
    assert big.set_pack_residency(Bad).startswith("error:")
