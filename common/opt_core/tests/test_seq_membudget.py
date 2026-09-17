"""opt_core.seq.membudget: the tested-shape table, the linear byte model, the device budget with cached decisions, the reason counter."""
import ast
import io
import sys

import pytest

from opt_core.seq import membudget as mb

H100 = "NVIDIA H100 80GB HBM3"


def test_shape_table_admit_forms():
    t = mb.ShapeTable.compose({"H100": {(1, 8192), (4, 8192)}, "H200": {(1, 8192), (4, 8192)}},
                              {"H100": {(2, 8192)}, "B200": {(1, 8192), (4, 8192)}})
    assert t.certified["H100"] == {(1, 8192), (2, 8192), (4, 8192)} and t.certified["B200"] == {(1, 8192), (4, 8192)}
    assert t.admit((4, 8192), H100) == ("kit", None)
    r = t.admit((8, 8192), H100)
    assert r[0] == "stock" and r[1].startswith("shape|(B=8, L=8192) not certified on H100 (certified: [(1, 8192), (2, 8192), (4, 8192)])")
    assert t.admit((1, 8192), "NVIDIA A100-SXM4-80GB") == ("stock", "device_class|'NVIDIA A100-SXM4-80GB' has no certified shapes")
    assert t.admit((1, 8192), None)[1].startswith("device_class|")
    # blockers win first, in order
    b = t.admit((1, 8192), H100, blockers=[("inference_params", None, "generate"), ("padding_mask", object(), "padded forward")])
    assert b == ("stock", "padding_mask|padded forward")
    assert t.device_class("NVIDIA H200") == "H200"
    assert t.line(H100) == "shapes=H100:(B=1, L=8192);(B=2, L=8192);(B=4, L=8192)"
    assert t.line("Tesla T4").startswith("shapes=none")
    assert mb.ShapeTable({"L4": [(3, 5, 7)]}).admit([3, 5, 7], "NVIDIA L4") == ("kit", None)
    # order-deterministic: composition keeps base order then added-only classes; the longest fragment in the name wins regardless of order
    assert list(t.certified) == ["H100", "H200", "B200"]
    for order in ({"H200": [(1, 1)], "GH200": [(2, 2)]}, {"GH200": [(2, 2)], "H200": [(1, 1)]}):
        g = mb.ShapeTable(order)
        assert g.device_class("NVIDIA GH200 480GB") == "GH200" and g.device_class("NVIDIA H200") == "H200"
        assert g.admit((2, 2), "NVIDIA GH200 480GB") == ("kit", None) and g.admit((2, 2), "NVIDIA H200")[0] == "stock"
    tie = mb.ShapeTable({"H100 PCIe": [(1, 1)], "H100 80GB": [(2, 2)]})
    assert tie.device_class("NVIDIA H100 80GB HBM3") == "H100 80GB" and tie.device_class("NVIDIA H100 PCIe") == "H100 PCIe"


def test_linear_need_scaling():
    ln = mb.LinearNeed(base_B=1, base_L=8192, store_bytes=8 << 30, stock_peak_delta_bytes=2 << 30)
    need, stock, basis = ln.at(2, 16384)
    assert need == 16 << 30 and stock == 8 << 30 and "scaled by L/8192" in basis and "scaled by B·L" in basis


def test_device_budget_decisions_cached_and_lined():
    bud = mb.DeviceBudget(margin_bytes=int(2 * mb.GIB), reserve_frac=0.0, total_bytes=int(80 * mb.GIB))
    d = bud.fits((1, 32768), need_bytes=int(30 * mb.GIB), stock_need_bytes=int(8 * mb.GIB), free_bytes=int(50 * mb.GIB))
    assert d.ok and d.need_gib == 30.0 and d.margin_gib == 2.0 and "budget ok" in d.reason and "≤ free 50.0 GiB" in d.reason
    d2 = bud.fits((1, 65536), need_bytes=int(60 * mb.GIB), stock_need_bytes=int(16 * mb.GIB), free_bytes=int(50 * mb.GIB))
    assert not d2.ok and "budget FAILS" in d2.reason and d2.reason.endswith("→ the stock path for this shape; counted")
    same = bud.fits((1, 65536), need_bytes=int(60 * mb.GIB), stock_need_bytes=int(16 * mb.GIB), free_bytes=int(500 * mb.GIB))
    assert same is d2                                                                    # same key, same need: the one decision stands (free not re-measured)
    d2b = bud.fits((1, 65536), need_bytes=int(10 * mb.GIB), stock_need_bytes=int(16 * mb.GIB), free_bytes=int(50 * mb.GIB))
    assert d2b is not d2 and d2b.ok and "re-decided: need 60.0+16.0 GiB -> 10.0+16.0 GiB" in d2b.reason      # a new need is never answered by the stale verdict
    assert bud.decided[(1, 65536)] is d2b
    bud.forget((1, 65536))
    d3 = bud.fits((1, 65536), need_bytes=0, free_bytes=int(80 * mb.GIB), note="after releasing the other shapes' stores")
    assert d3.ok and "after releasing" in d3.reason
    assert bud.lines() == [d.reason, d2.reason, d2b.reason, d3.reason]
    r = mb.DeviceBudget(margin_bytes=0, reserve_frac=0.5, total_bytes=100)
    assert r.margin(None) == 50 and not r.fits("k", need_bytes=60, free_bytes=100).ok and r.fits("k2", need_bytes=50, free_bytes=100).ok


def test_free_device_bytes_named_refusal_without_torch(monkeypatch):
    numerics = pytest.importorskip("opt_core.seq.numerics")  # the seq package's single torch refusal lives there
    monkeypatch.setitem(sys.modules, "torch", None)          # import torch -> ImportError
    with pytest.raises(numerics.TorchUnavailable):
        mb.free_device_bytes()
    with pytest.raises(numerics.TorchUnavailable):
        mb.DeviceBudget(margin_bytes=0, reserve_frac=0.0).fits((1, 1), need_bytes=1)         # no free_bytes given -> measures -> refuses by name


def test_free_device_bytes_named_refusal_without_cuda(monkeypatch):
    pytest.importorskip("opt_core.seq.numerics")
    import types
    fake = types.ModuleType("torch"); fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(mb.CudaUnavailable):
        mb.free_device_bytes()


def test_budget_margins_are_required():
    with pytest.raises(TypeError):
        mb.DeviceBudget()                                    # margins are the kit's tested values: required


def test_reason_counter_counts_all_announces_once():
    s = io.StringIO()
    rc = mb.ReasonCounter("kit vX", stream=s)
    for _ in range(3):
        rc.record("shape|(B=8, L=8192) not certified on H100")
    rc.record("padding_mask|padded forward")
    assert rc.counts == {"shape": 3, "padding_mask": 1}
    out = s.getvalue().splitlines()
    assert out == ["[kit vX] STOCK PATH shape|(B=8, L=8192) not certified on H100; counted", "[kit vX] STOCK PATH padding_mask|padded forward; counted"]
    assert rc.line() == "routed=padding_mask:1,shape:3"
    assert mb.ReasonCounter("t").line() == "routed=none"
    from opt_core.mem import Ledger
    shared = Ledger()
    a = mb.ReasonCounter("t", ledger=shared, stream=s); a.record("x|1"); a.record("x|2")
    assert shared.facts() == {"routed_x": 2} and a.counts == {"x": 2}


def test_line_fields_hook():
    t = mb.ShapeTable({"H100": {(1, 8192)}})
    bud = mb.DeviceBudget(margin_bytes=0, reserve_frac=0.0, total_bytes=10)
    rc = mb.ReasonCounter("t", stream=io.StringIO())
    assert mb.line_fields(t, H100, bud, rc) == {"shapes": "H100:(B=1, L=8192)", "membudget": "none decided", "routed": "none"}
    bud.fits((1, 8192), need_bytes=1, free_bytes=5); rc.record("shape|x")
    f = mb.line_fields(t, H100, bud, rc)
    assert f["membudget"].startswith("shape (B=1, L=8192): budget ok") and f["routed"] == "shape:1"
    assert all(isinstance(v, str) for v in f.values()) and mb.line_fields() == {}


def test_source_parses_at_python_3_8_and_top_imports_are_stdlib():
    src = open(mb.__file__, encoding="utf-8").read()
    ast.parse(src, feature_version=(3, 8))
    tops = [n for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {(n.module if isinstance(n, ast.ImportFrom) else n.names[0].name) for n in tops}
    assert names <= {"__future__", "collections", "sys", "typing", "mem"}, names          # opt_core.mem is standard library only
