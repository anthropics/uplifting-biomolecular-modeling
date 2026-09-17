"""The trunk-graph token budget (tgbudget): the routing rule's census words (CPU; the wrap of rf3's Recycler.forward is GPU-box evidence)."""
import pytest

from .. import tgbudget as tgb


def test_budget_is_1000_tokens():
    assert tgb.MAX_I == 1000


def test_rule_routes_by_token_count_and_counts_both_paths(monkeypatch):
    import sys, types
    calls = []
    graph = lambda self, f, a, b, c, S, Z: calls.append("graph") or ("g", Z)
    eager = lambda self, f, a, b, c, S, Z: calls.append("eager") or ("e", Z)
    RS = types.SimpleNamespace(Recycler=type("Recycler", (), {"forward": graph}))
    pkg = types.ModuleType("rf3"); model = types.ModuleType("rf3.model"); mod = types.ModuleType("rf3.model.RF3_structure"); mod.Recycler = RS.Recycler
    monkeypatch.setitem(sys.modules, "rf3", pkg); monkeypatch.setitem(sys.modules, "rf3.model", model); monkeypatch.setitem(sys.modules, "rf3.model.RF3_structure", mod)
    adapter = types.SimpleNamespace(_ORIG={"rec_forward": eager})
    tgb.CENSUS.clear(); tgb.STATE.update({"on": False, "max_i": None})
    try:
        d = tgb.enable(adapter)
        assert d["on"] and d["max_i"] == 1000

        class Z:                                    # a pair tensor stand-in with .shape
            def __init__(self, n): self.shape = (1, n, n, 128)
        r = RS.Recycler()
        assert RS.Recycler.forward(r, None, None, None, None, None, Z(400))[0] == "g"
        assert RS.Recycler.forward(r, None, None, None, None, None, Z(1000))[0] == "g"      # the budget is inclusive
        assert RS.Recycler.forward(r, None, None, None, None, None, Z(1400))[0] == "e"      # above it: the pre-graph forward, a named skip
        assert tgb.census() == {"graphed": 2, "skipped": 1, "by_key": {"graphed:400": 1, "graphed:1000": 1, "skipped:1400": 1}}
        assert tgb.lever_evidence() == [("budget_max_i", 1000), ("graphed", 2), ("skipped", 1)]
        with pytest.raises(tgb.TgBudgetRefused):                                               # an arm without tg: nothing to budget
            tgb.enable(types.SimpleNamespace(_ORIG={}))
    finally:
        tgb.CENSUS.clear(); tgb.STATE.update({"on": False, "max_i": None})


def test_budget_by_arm_kernel_class(monkeypatch):
    """The fast-kernel arm's graph pays over a shorter token range: the adapter's trimul mode picks the budget."""
    import sys, types
    assert tgb.MAX_I_FAST == 300 and tgb.BUDGETS == {"stock": 1000, "fast": 300}
    assert tgb.budget_for(types.SimpleNamespace(describe=lambda: {"mode": "fast"})) == 300
    assert tgb.budget_for(types.SimpleNamespace(describe=lambda: {"mode": "stock"})) == 1000
    assert tgb.budget_for(types.SimpleNamespace(describe=lambda: {"mode": None})) == tgb.budget_for(types.SimpleNamespace()) == 1000   # unknown / no record: the stock budget
    graph = lambda self, f, a, b, c, S, Z: ("g", Z)
    eager = lambda self, f, a, b, c, S, Z: ("e", Z)
    Rec = type("Recycler", (), {"forward": graph})
    mod = types.ModuleType("rf3.model.RF3_structure"); mod.Recycler = Rec
    monkeypatch.setitem(sys.modules, "rf3", types.ModuleType("rf3")); monkeypatch.setitem(sys.modules, "rf3.model", types.ModuleType("rf3.model"))
    monkeypatch.setitem(sys.modules, "rf3.model.RF3_structure", mod)
    tgb.CENSUS.clear(); tgb.STATE.update({"on": False, "max_i": None})
    try:
        d = tgb.enable(types.SimpleNamespace(_ORIG={"rec_forward": eager}, describe=lambda: {"mode": "fast", "served": 0}))
        assert d["max_i"] == 300

        class Z:
            def __init__(self, n): self.shape = (1, n, n, 128)
        assert Rec.forward(Rec(), None, None, None, None, None, Z(200))[0] == "g"
        assert Rec.forward(Rec(), None, None, None, None, None, Z(400))[0] == "e"                 # the fast arm's trunk runs its kernels eagerly from 400 tokens
        assert tgb.enable(types.SimpleNamespace(_ORIG={"rec_forward": eager}, describe=lambda: {"mode": "fast"}), max_i=1000)["max_i"] == 1000   # an explicit budget wins
    finally:
        tgb.CENSUS.clear(); tgb.STATE.update({"on": False, "max_i": None})


def test_the_fast_arm_has_no_trunk_graph_on_compute_capability_8_0():
    """CARD_BUDGETS: on 8.0 the fast-kernel arm's budget is 0 tokens (every trunk call takes the pre-graph forward by name); the stock-kernel
    arm keeps MAX_I; 9.0 keeps both budgets; no GPU (the dry run) keeps the arm budgets."""
    class A:
        def __init__(self, mode): self.mode = mode
        def describe(self): return {"mode": self.mode}
    assert tgb.budget_for(A("fast"), cc=(8, 0)) == 0 and tgb.STATE["card"] == "cc8.0:fast:max_i=0"
    assert tgb.budget_for(A("stock"), cc=(8, 0)) == tgb.MAX_I and tgb.STATE["card"] is None
    assert tgb.budget_for(A("fast"), cc=(9, 0)) == tgb.MAX_I_FAST
    assert tgb.budget_for(A("fast"), cc=None) == tgb.MAX_I_FAST
    assert tgb.describe()["card"] is None
