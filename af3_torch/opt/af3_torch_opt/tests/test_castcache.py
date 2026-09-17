"""The package lever ``castcache`` restates ``nn.Linear.forward`` (torch's, pinned here) with the memoised bf16 casts standing in for the
fp32 parameters exactly where autocast casts them. Bitwise by construction; the GPU proof is the kit's canaries (``exact`` vs ``off``),
here the statement, the eligibility rule and the equality under CPU autocast."""
import inspect

import pytest

from af3_torch_opt import castcache, modes, registry


def test_lever_tables():
    assert "castcache" in registry.PACKAGE_LEVERS and "castcache" in registry.LEVERS and registry.EVIDENCE["castcache"] == "package"
    assert registry.BITWISE_EVIDENCE["castcache"] == {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True}
    assert registry.STRATEGY["castcache"] == "LOCAL.af3_torch.castcache" and registry.IMPL["castcache"] == ("castcache.install", "kit")
    assert [m for m in modes.MODES if "castcache" in modes.MODE_PACKAGE_LEVERS[m]] == ["exact"]        # fast / big: bf16w leaves autocast nothing to cast
    assert "bf16w" in modes.kit_lever_sets()[modes.MODE_SETS["fast"]] and "bf16w" in modes.kit_lever_sets()[modes.MODE_SETS["big"]] and "bf16w" not in registry.EXACT
    assert castcache.take() == {"served": 0, "stock": 0}


def test_the_stock_statement_is_torchs_linear_forward():
    torch = pytest.importorskip("torch")
    src = inspect.getsource(torch.nn.Linear.forward)
    assert "return F.linear(input, self.weight, self.bias)" in src, src                         # the statement castcache.forward restates
    mine = inspect.getsource(castcache.forward)
    assert "return F.linear(input, w, self._castcache_bias)" in mine and "return F.linear(input, self.weight, self.bias)" in mine


def test_memoised_cast_is_bit_for_bit_autocasts_and_eligibility():
    torch = pytest.importorskip("torch")
    import types
    torch.manual_seed(0)
    lin = torch.nn.Linear(48, 40, bias=True); nob = torch.nn.Linear(48, 24, bias=False)
    x = torch.randn(7, 48)
    for m in (lin, nob):
        w16 = m.weight.detach().to(torch.bfloat16); b16 = None if m.bias is None else m.bias.detach().to(torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            stock = m(x)                                                                       # autocast casts weight / bias / input per call
            memo = torch.nn.functional.linear(x, w16, b16)                                     # the memoised casts handed to the same call
        assert stock.dtype == torch.bfloat16 and torch.equal(stock, memo)
    # eligibility: fp32 CUDA nn.Linear proper only — CPU modules, bf16 weights (bf16w) and subclasses are left alone, by rule
    assert not castcache.eligible(lin, torch)                                                  # CPU
    class Sub(torch.nn.Linear):
        pass
    assert not castcache.eligible(Sub(4, 4), torch)
    model = torch.nn.Sequential(lin, nob)
    r = castcache.install(model)
    assert r["installed"] and not r["already"] and r["linears"] == 0 and r["reason"] == "no_fp32_cuda_linear(weights_not_fp32)"   # nothing on CPU: named
    assert castcache.install(model)["already"] is True
    # the bound forward outside autocast runs the stock statement and counts it
    lin._castcache_weight, lin._castcache_bias = lin.weight.detach().to(torch.bfloat16), lin.bias.detach().to(torch.bfloat16)
    lin.forward = types.MethodType(castcache.forward, lin); castcache.take()
    assert torch.equal(lin(x), torch.nn.functional.linear(x, lin.weight, lin.bias)) and castcache.take() == {"served": 0, "stock": 1}
