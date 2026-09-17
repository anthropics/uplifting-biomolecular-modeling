"""The `apb_trunk` cell: switch parsing and the conflict with the trunk-kernels add-on's OF3T_APB route (CPU); on a GPU with openfold3
importable, the served forward of a pairformer-style AttentionPairBias (16 heads x 24, gating, pair bias from z) against the module's own
fp64 statement — within twice the stock bf16 path's error, trunk-shaped (one z) and confidence-shaped (a sample dimension on a and z) —
a registered producer is called instead of the module's LayerNorm + GEMM, and the refusal path (a vendor-kernel flag) runs the stock
forward, counted."""
import os
import sys

import pytest

try:
    import torch
except ImportError:                                                      # the switch-parsing test needs no torch
    torch = None
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from openfold3_opt.cells import apb_trunk as AT  # noqa: E402

cuda = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="no CUDA device")


def test_switch_parsing_and_conflict():
    assert AT._core.ENV == AT.ENV == "OPENFOLD3_OPT_APB_TRUNK" and AT._core.HIGH_PRECISION == "honour" and AT._core.PREFIX == "[openfold3-opt/apb_trunk]"
    assert AT.requested({}) is False and AT.requested({AT.ENV: ""}) is False
    assert AT.requested({AT.ENV: "1"}) is True and AT.requested({AT.ENV: "1", AT.ENV_MIN: "256"}) is True
    with pytest.raises(ValueError):
        AT.requested({AT.ENV: "yes"})
    with pytest.raises(ValueError):
        AT.requested({AT.ENV: "1", AT.ENV_MIN: "many"})
    with pytest.raises(ValueError):
        AT.requested({AT.ENV: "1", AT.CONFLICT_ENV: "cueq"})            # both would own AttentionPairBias.forward


def _module(heads=16, dim=24, c_s=384, c_z=128):
    apb = pytest.importorskip("openfold3.core.model.layers.attention_pair_bias")
    torch.manual_seed(0)
    m = apb.AttentionPairBias(c_q=c_s, c_k=c_s, c_v=c_s, c_s=c_s, c_z=c_z, c_hidden=dim, no_heads=heads, use_ada_layer_norm=False, gating=True, inf=1e9)
    with torch.no_grad():
        for name, p in m.named_parameters():                            # trained-like magnitudes instead of the zero final / gating inits
            p.copy_(torch.randn_like(p) * (p.shape[-1] ** -0.5 if p.dim() > 1 else 0.3) + (1.0 if name.endswith("weight") and p.dim() == 1 else 0.0))
    return apb, m.cuda().eval()


def _rel(x, ref):
    return float((x.double() - ref).norm() / ref.norm()), float((x.double() - ref).abs().max() / ref.abs().max())


@cuda
@pytest.mark.parametrize("lead_a,lead_z", [((1,), (1,)), ((1, 3), (1, 3)), ((1, 3), (1, 1))])   # trunk; confidence head batched over samples; samples sharing one z
def test_served_forward_matches_the_fp64_statement_within_the_stock_bf16_class(lead_a, lead_z):
    apb, m = _module()
    AT._core._CORE["mod"] = None
    N = 203
    torch.manual_seed(1)
    a = torch.randn(*lead_a, N, 384, device="cuda"); z = torch.randn(*lead_z, N, N, 128, device="cuda")
    mask = (torch.rand(*lead_a, N, device="cuda") > 0.1).float()
    with torch.no_grad():
        ref = m.double()(a=a.double(), z=z.double(), s=None, mask=mask.double()); m.float()
        with torch.autocast("cuda", torch.bfloat16):
            stock = m(a=a, z=z, s=None, mask=mask)
            ok, why = AT.plan(m, a, z, mask, {})
            assert ok, why
            ours = AT.served_forward(m, a, z, mask)
    assert ours.shape == stock.shape
    (n_s, m_s), (n_o, m_o) = _rel(stock, ref), _rel(ours, ref)
    assert n_o <= 2.0 * n_s + 1e-6 and m_o <= 2.5 * m_s + 1e-6, ((n_o, m_o), (n_s, m_s))


@cuda
def test_registered_producer_refusals_and_the_patched_class():
    apb, m = _module()
    AT._core._CORE["mod"] = None
    N = 96
    torch.manual_seed(2)
    a = torch.randn(1, N, 384, device="cuda"); z = torch.randn(1, N, N, 128, device="cuda"); mask = torch.ones(1, N, device="cuda")
    calls = {"n": 0}

    def producer(module, z_, dtype):
        calls["n"] += 1
        return AT.headmajor_bias(module, z_, dtype)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        base = AT.served_forward(m, a, z, mask)
        AT.use_producer("stand_in", producer)
        try:
            got = AT.served_forward(m, a, z, mask)
        finally:
            AT.use_producer("headmajor_mm", None)
    assert calls["n"] == 1 and torch.equal(got, base) and AT.STATE["producer"] == "headmajor_mm"
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        assert AT.plan(m, a, z, mask, {"use_lma": True}) == (False, "flag:use_lma")
        assert AT.plan(m, a, None, mask, {}) == (False, "no_pair")
        assert AT.plan(m, a, z, mask[:, : N - 1], {}) == (False, "shape:mask")
    with torch.no_grad():
        assert AT.plan(m, a, z, mask, {})[1] == "dtype:float32"                  # no autocast: fp32 activations are the stock path's
    # the patched class: served calls counted, a flagged call runs the stock forward counted by reason
    AT._core._patch(apb); AT.STATE["state"] = "on"
    try:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            s0 = AT.STATE["served"]
            out = m(a=a, z=z, s=None, mask=mask)
            assert AT.STATE["served"] == s0 + 1 and torch.equal(out, base)
            fb0 = dict(AT.STATE["fallback"])
            want = apb.AttentionPairBias.forward.__wrapped__(m, a=a, z=z, s=None, mask=mask, use_high_precision_attention=True)
            got = m(a=a, z=z, s=None, mask=mask, use_high_precision_attention=True)
            assert torch.equal(got, want) and AT.STATE["fallback"].get("flag:use_high_precision_attention", 0) == fb0.get("flag:use_high_precision_attention", 0) + 1
        assert "producer=headmajor_mm" in AT.census_line() and "served=" in AT.census_line()
    finally:
        apb.AttentionPairBias.forward = apb.AttentionPairBias.forward.__wrapped__; AT.STATE["state"] = "off"
