"""The dtk lever's arithmetic and preconditions on CPU: ``dtk.attn`` against the stock block for every batch / bias shape the diffusion
transformer can hand it (a torch reference stands in for the routed Triton kernel ``dtk_kernels``), the size gate below / above, the
counted fallback, and ``enable``'s named refusals."""
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from .. import dtk


def _ref_kernel_module():
    """sys.modules['dtk_kernels'] stand-in: flash_bias_attn(q, k, v [H,N,D], bias [H,N,N]) = softmax(q k^T * scale + bias) v -> [N, H*D]."""
    m = types.ModuleType("dtk_kernels")

    def flash_bias_attn(q, k, v, bias=None, out_dtype=None, scale=None, **_):
        H, N, D = q.shape
        s = torch.einsum("hid,hjd->hij", q.float(), k.float()) * scale
        if bias is not None:
            s = s + bias.float()
        o = torch.einsum("hij,hjd->hid", torch.softmax(s, dim=-1), v.float())          # [H, N, D]
        return o.permute(1, 0, 2).reshape(N, H * D).to(out_dtype or q.dtype)
    m.flash_bias_attn = flash_bias_attn
    m.__file__ = "<ref dtk_kernels>"
    return m


@pytest.fixture
def lever(monkeypatch):
    monkeypatch.setitem(sys.modules, dtk.KERNEL, _ref_kernel_module())
    monkeypatch.setattr(dtk, "GATE", dtk.make_gate(0))
    monkeypatch.setitem(dtk.STATE, "bias", "relayout")
    monkeypatch.setitem(dtk.STATE, "on", True)
    return dtk


def _qkvg(lead, I=8, H=2, D=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda: torch.randn(*lead, I, H, D, generator=g)
    return mk(), mk(), mk(), torch.sigmoid(mk())


@pytest.mark.parametrize("lead,bias_lead", [((), ()), ((5,), ()), ((5,), (1,)), ((5,), (5,)), ((1, 5), (1, 1)), ((1, 5), (1, 5)), ((1,), ())])
def test_attn_equals_the_stock_block_for_every_batch_and_bias_shape(lever, lead, bias_lead):
    I, H, D = 8, 2, 4
    Q, K, V, G = _qkvg(lead, I, H, D)
    B = torch.randn(*bias_lead, I, I, H, generator=torch.Generator().manual_seed(1))
    got = lever.attn(Q, K, V, B, G, D)
    want = dtk._stock_block(Q, K, V, B, G, D)
    assert got.shape == want.shape == (*lead, I, H * D)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got - want).abs().max()
    c = lever.GATE.census()
    assert c["served"] == 1 and c["fallback"] == 0 and c["gated"] == 0


def test_below_the_gate_runs_the_stock_block_bitwise_and_counts_gated(lever, monkeypatch):
    monkeypatch.setattr(dtk, "GATE", dtk.make_gate())                                   # the shipped gate: min 400 tokens
    Q, K, V, G = _qkvg((5,))
    B = torch.randn(8, 8, 2)
    assert torch.equal(lever.attn(Q, K, V, B, G, 4), dtk._stock_block(Q, K, V, B, G, 4))  # I=8 < 400: the stock block itself
    c = dtk.GATE.census()
    assert c == {**c, "calls": 1, "served": 0, "gated": 1, "fallback": 0} and c["gated_by"] == {"lt_min": 1}
    assert dtk.problems() == []                                                            # gated by size is the lever's design, not a failure


def test_a_bias_of_another_batch_is_a_counted_fallback_to_the_stock_block(lever):
    Q, K, V, G = _qkvg((5,))
    lever.attn(Q, K, V, torch.randn(5, 8, 8, 2), G, 4)                                    # one bias per sample: served
    c = lever.GATE.census()
    assert c["served"] == 1 and c["fallback"] == 0
    B = torch.randn(3, 8, 8, 2)                                                            # neither shared nor one per sample
    with pytest.raises(RuntimeError):                                                     # the stock block itself cannot broadcast it —
        lever.attn(Q, K, V, B, G, 4)                                                       # the fallback is counted BEFORE the stock arithmetic raises
    c = lever.GATE.census()
    assert c["fallback"] == 1 and c["fallback_by"] == {"bias_shape": 1}
    assert dtk.problems() and "fallback" in dtk.problems()[0]
    d = dtk.describe()
    assert d["ok"] is False and "bias_shape" in d["reason"] and d["census"]["calls"] == 2


def test_enable_refuses_by_name(monkeypatch):
    monkeypatch.delitem(sys.modules, dtk.ADAPTER, raising=False)
    monkeypatch.setitem(dtk.STATE, "on", False)
    with pytest.raises(dtk.DtkRefused, match="is not imported"):
        dtk.enable()
    adp = types.ModuleType(dtk.ADAPTER); adp._ORIG = {}; adp._DATTN_BLOCK = "x"; adp.DATTN = {"on": True}
    monkeypatch.setitem(sys.modules, dtk.ADAPTER, adp)
    with pytest.raises(dtk.DtkRefused, match="dattn is on"):
        dtk.enable()
    adp.DATTN = {"on": False}
    monkeypatch.setitem(sys.modules, dtk.KERNEL, None)                                     # the routed kernel not importable: named
    with pytest.raises(dtk.DtkRefused, match=dtk.KERNEL):
        dtk.enable()


def test_describe_before_enable_is_not_installed():
    if dtk.STATE["on"]:
        pytest.skip("lever installed in this process")
    d = dtk.describe()
    assert d["on"] is False and d["ok"] is False and d["reason"] == "not installed"
