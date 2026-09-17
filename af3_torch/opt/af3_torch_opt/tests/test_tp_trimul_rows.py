"""rowpair_xfold's triangle-multiplication seam under P > 1: which TriMulFns the adapter hands the core's ``trimul_update_``.
With the kit's trimul lever on (big's set carries it) the core's fused row kernels (``opt_core.mem.rowpair.trimul_fused.fused_trimul_fns``)
get the ten canonical weights + the module's eager layers as plain ``TriMulFns`` callables — their named per-unit fallback (below the core's
``min_tokens`` pair size, under the core's ``ROWPAIR_TRIMUL_KERNELS=torch``, or for a shape the kernels do not serve; each counted on the core's
F2.trimul_rows line). Lever off (``--levers`` without trimul): those callables alone — no fused hooks, no census word — the streamed
contraction's hook-less path. A core without the module keeps the callables and says so by name. CPU only: real xfold
``TriangleMultiplication`` modules, no GPU, no collectives."""
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("triton")

from af3_torch_opt.tests.test_tp_xfold_cpu import KIT, _load_rpx  # noqa: E402  (one loader for the by-path module; the triton autotune stub for GPU-less boxes)
from opt_core import trimul as core_trimul  # noqa: E402
from opt_core.mem.rowpair import trimul as TM  # noqa: E402


class _Kernels:                      # the kit kernels module's lever words the adapter reads (af3_kernels._ON / _DEAD)
    def __init__(self, on):
        self._ON, self._DEAD = tuple(on), {}


@pytest.fixture()
def rpx(monkeypatch):
    m = _load_rpx()
    st = m._fresh_state(); st.update(P=2, rank=0, K=_Kernels(("trimul", "triattn")), C=m._core())
    monkeypatch.setattr(m, "STATE", st)
    return m


@pytest.fixture(scope="module")
def mods():
    if KIT not in sys.path:
        sys.path.insert(0, KIT)
    import af3_torch_api  # noqa: F401  (xfold.of3.OF3 = True before modules are built, as the model process does)
    from xfold.nn.triangle_multiplication import TriangleMultiplication
    torch.manual_seed(0)
    out, inc = TriangleMultiplication(c_pair=128, _outgoing=True).eval(), TriangleMultiplication(c_pair=128, _outgoing=False).eval()
    for mod in (out, inc):
        for prm in mod.parameters():
            with torch.no_grad():
                prm.copy_(torch.randn_like(prm) * 0.1)
    return {"outgoing": out, "incoming": inc}


def _fake_core_module(monkeypatch, record):
    """A stand-in ``opt_core.mem.rowpair.trimul_fused`` with the frozen API's two entry points."""
    import opt_core.mem.rowpair as pkg
    fake = types.ModuleType("opt_core.mem.rowpair.trimul_fused")

    def fused_trimul_fns(weights, stock_fns, eps=1e-5, cells=None, ledger=None):
        record.append(dict(weights=weights, stock_fns=stock_fns, eps=eps, cells=cells, ledger=ledger))
        return ("fused", stock_fns)

    fake.fused_trimul_fns = fused_trimul_fns
    fake.emit_line = lambda tag: f"[{tag}] LEVER name=F2.trimul_rows state=on"
    from opt_core.counters import Ledger
    fake.LEDGER = Ledger("F2.trimul_rows", impl="fpf_trimul_v4", origin="core", min_tokens=2048)
    fake.LEDGER.set("min_tokens", 2048)                                   # a ledger fact, as the core records its size gate
    fake.describe = lambda: dict(fake.LEDGER.fields())
    monkeypatch.setitem(sys.modules, "opt_core.mem.rowpair.trimul_fused", fake)
    monkeypatch.setattr(pkg, "trimul_fused", fake, raising=False)
    return fake


def test_lever_off_is_the_eager_callables_unchanged(rpx, mods):
    """``--levers`` without trimul (or no kernels module at all): the hook-less TriMulFns the streamed contraction has always run — no census
    word, nothing recorded, the core module never consulted."""
    for K in (_Kernels(("triattn",)), None):
        rpx.STATE["K"] = K
        for name, mod in mods.items():
            fns = rpx.trimul_fns(mod, outgoing=(name == "outgoing"))
            assert type(fns) is TM.TriMulFns and int(fns.C_h) == 128, (name, type(fns))
            assert not hasattr(fns, "proj_into") and not hasattr(fns, "tile_epilogue")      # the core's trimul_update_ takes its hook-less path
    assert "trimul_rows" not in rpx.STATE["stats"]["levers"]
    assert rpx.emit_trimul_rows_line("rowpair") is None and rpx.trimul_rows_record() is None


def test_lever_on_builds_the_cores_fused_provider_over_the_eager_callables(rpx, mods):
    """The default on a core that carries ``trimul_fused``: a ``FusedTriMulFns`` (the core's class) whose proj / out / gate ARE the eager
    callables, min_tokens the core's default gate, the ledger recorded for the run record."""
    RF = rpx.trimul_fused_module()
    assert RF is not None
    for name, mod in mods.items():
        fns = rpx.trimul_fns(mod, outgoing=(name == "outgoing"))
        assert isinstance(fns, RF.FusedTriMulFns) and isinstance(fns, TM.TriMulFns) and int(fns.C_h) == 128, (name, type(fns))
        assert fns.min_tokens == RF.DEFAULT_MIN_TOKENS == 2048 and hasattr(fns, "proj_into") and hasattr(fns, "tile_epilogue")
    assert rpx.STATE["stats"]["levers"]["trimul_rows"] == "fpf_v4"
    rec = rpx.trimul_rows_record()
    assert isinstance(rec, dict) and rec["name"] == "F2.trimul_rows" and (rec.get("facts") or {}).get("min_tokens") == 2048, rec


def test_a_core_without_the_module_keeps_torch_by_name(rpx, mods, monkeypatch):
    import opt_core.mem.rowpair as pkg
    monkeypatch.setitem(sys.modules, "opt_core.mem.rowpair.trimul_fused", None)             # import raises ImportError: a core that does not carry it
    monkeypatch.delattr(pkg, "trimul_fused", raising=False)
    assert rpx.trimul_fused_module() is None
    fns = rpx.trimul_fns(mods["outgoing"], outgoing=True)
    assert type(fns) is TM.TriMulFns and not hasattr(fns, "proj_into")
    assert rpx.STATE["stats"]["levers"]["trimul_rows"] == "fallback:core_without_trimul_fused"
    assert rpx.emit_trimul_rows_line("rowpair") is None and rpx.trimul_rows_record() is None


def test_the_core_module_is_handed_the_weights_and_the_torch_fallback(rpx, mods, monkeypatch):
    record = []
    _fake_core_module(monkeypatch, record)
    assert rpx.trimul_fused_module() is not None
    for name, mod in mods.items():
        outgoing = name == "outgoing"
        got = rpx.trimul_fns(mod, outgoing=outgoing)
        assert got[0] == "fused" and type(got[1]) is TM.TriMulFns, name                    # whatever the core returns is what trimul_update_ gets
        call = record[-1]
        assert set(call["weights"]) == set(core_trimul.WEIGHT_KEYS) and call["eps"] == 1e-5 and call["cells"] is None and call["stock_fns"] is got[1]
        Pw, Gw = mod.projection.weight, mod.gate.weight                                    # interleaved a|b rows; incoming swaps a and b (the generic form)
        a_p, b_p, a_g, b_g = (Pw[0::2], Pw[1::2], Gw[0::2], Gw[1::2]) if outgoing else (Pw[1::2], Pw[0::2], Gw[1::2], Gw[0::2])
        w = call["weights"]
        assert torch.equal(w["w_ap"], a_p) and torch.equal(w["w_bp"], b_p) and torch.equal(w["w_ag"], a_g) and torch.equal(w["w_bg"], b_g), name
        assert w["ln_in_w"] is not None and torch.equal(w["w_o"], mod.output_projection.weight) and torch.equal(w["w_og"], mod.gating_linear.weight)
        assert torch.equal(w["ln_out_w"], mod.center_norm.weight) and torch.equal(w["ln_in_b"], mod.left_norm_input.bias)
    assert rpx.STATE["stats"]["levers"]["trimul_rows"] == "fpf_v4"
    assert rpx.emit_trimul_rows_line("rowpair") == "[rowpair] LEVER name=F2.trimul_rows state=on"


def test_the_trimul_lever_off_never_calls_the_core(rpx, mods, monkeypatch):
    record = []
    _fake_core_module(monkeypatch, record)
    rpx.STATE["K"] = _Kernels(("triattn",))                                              # `--levers` without trimul: the fused rows honour the lever
    assert type(rpx.trimul_fns(mods["outgoing"], outgoing=True)) is TM.TriMulFns and record == [] and "trimul_rows" not in rpx.STATE["stats"]["levers"]


def test_weight_layout_is_the_p1_packs():
    """The a|b de-interleave and the incoming swap are stated once per path; both files carry the same statements (the P = 1 pack:
    af3_kernels._trimul_weights)."""
    k = open(os.path.join(os.path.dirname(KIT), "kernels", "af3_kernels.py"), encoding="utf-8").read()
    for stmt in ("P[0::2].contiguous(), P[1::2].contiguous(), Gw[0::2].contiguous(), Gw[1::2].contiguous()", "w_ap, w_bp, w_ag, w_bg = b_p, a_p, b_g, a_g"):
        assert stmt in k, stmt
    r = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rowpair_xfold.py"), encoding="utf-8").read()
    for stmt in ("Pw[0::2].contiguous(), Pw[1::2].contiguous(), Gw[0::2].contiguous(), Gw[1::2].contiguous()", "a_p, b_p, a_g, b_g = b_p, a_p, b_g, a_g"):
        assert stmt in r, stmt

