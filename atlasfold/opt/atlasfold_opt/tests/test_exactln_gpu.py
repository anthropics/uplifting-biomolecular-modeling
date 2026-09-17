"""exactln on a CUDA device: under the exact word every form EQUALS the stock statement bit for bit (served by a row the provider records bitwise on
this stack, or ATen by name — contiguous and transposed pair views, pair rows and small token rows); under the fast word a served row is inside the
LayerNorm tolerance band and a stock-row answer is the statement; AFO_EXACTLN=0 is `disabled`."""
import importlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


def _restore(cls, attr):
    fn = getattr(cls, attr)
    while getattr(fn, "__wrapped_stock__", None) is not None:
        fn = fn.__wrapped_stock__
    setattr(cls, attr, fn)


@pytest.mark.parametrize("mode", ["exact", "fast", "big"])
def test_forms_by_tier_word_against_the_stock_statement(monkeypatch, mode):
    from atlasfold_opt.hooks import exactln
    nm = importlib.import_module(exactln.TARGET)
    monkeypatch.delenv("AFO_EXACTLN", raising=False); monkeypatch.delenv("AFO_EXACTLN_WORD", raising=False)
    exactln._STATE["override"] = None
    _restore(nm.LayerNorm, "forward")
    stock = nm.LayerNorm.forward
    ins = exactln.install(mode, "atlasfold-opt", {"mode": mode})
    try:
        if ins.facts.get("provider") is None:
            pytest.skip(f"provider absent here: {ins.facts.get('refusal')}")
        assert ins.facts["word"] == mode
        g = torch.Generator(device="cuda").manual_seed(0)
        L = ins.facts["ledger"]
        for xdt, pdt in ((torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32)):
            m = nm.LayerNorm(128).cuda().to(pdt)
            with torch.no_grad():
                m.weight.normal_(); m.bias.normal_()
                x = (torch.randn(1, 320, 320, 128, device="cuda", generator=g) * 3 + 0.5).to(xdt)
                small = torch.randn(1, 64, 128, device="cuda", generator=g).to(xdt)
                for view in (x, x.transpose(1, 2), small):
                    ref = stock(m, view); out = m(view)
                    assert out.dtype == ref.dtype and out.shape == ref.shape, (mode, xdt, pdt)
                    if mode == "exact":
                        assert torch.equal(out, ref), (xdt, pdt, view.is_contiguous(), L.fields())
                    else:
                        err = float((out.float() - ref.float()).abs().max()); scale = float(ref.float().abs().max())
                        assert err <= (0.02 if xdt == torch.bfloat16 else 1e-4) * max(scale, 1.0), (mode, xdt, pdt, err, scale, L.fields())
        assert not L.errors and ins.gates[0]().ok, L.fields()
        for why in L.fallbacks:
            assert why.startswith(("stock_row:", "refused:", "no_affine:")), L.fields()
        line = ins.lines[0]()
        assert f" word={mode}" in line and " pinned=0" in line and " min_rows=" not in line, line
        monkeypatch.setenv("AFO_EXACTLN", "0")
        with torch.no_grad():
            assert torch.equal(m(x), stock(m, x))
        assert "disabled:1" in ins.lines[0]()
        assert ins.gates[0]().ok
    finally:
        _restore(nm.LayerNorm, "forward")
        exactln._STATE["override"] = None


def test_no_affine_norm_is_served_or_steps_aside_by_name(monkeypatch):
    from atlasfold_opt.hooks import exactln
    nm = importlib.import_module(exactln.TARGET)
    monkeypatch.delenv("AFO_EXACTLN", raising=False); monkeypatch.delenv("AFO_EXACTLN_WORD", raising=False)
    exactln._STATE["override"] = None
    _restore(nm.LayerNorm, "forward")
    stock = nm.LayerNorm.forward
    ins = exactln.install("exact", "atlasfold-opt", {"mode": "exact"})
    try:
        if ins.facts.get("provider") is None:
            pytest.skip(f"provider absent here: {ins.facts.get('refusal')}")
        m = torch.nn.LayerNorm(128, elementwise_affine=False).cuda()
        m.__class__ = nm.LayerNorm if issubclass(nm.LayerNorm, torch.nn.LayerNorm) else m.__class__
        if not isinstance(m, nm.LayerNorm):
            pytest.skip("stock LayerNorm is not an nn.LayerNorm subclass here")
        x = torch.randn(5, 256, 128, 96 if False else 128, device="cuda")
        with torch.no_grad():
            assert torch.equal(m(x), stock(m, x))
        L = ins.facts["ledger"]
        assert not L.errors and ins.gates[0]().ok, L.fields()
    finally:
        _restore(nm.LayerNorm, "forward")
        exactln._STATE["override"] = None
