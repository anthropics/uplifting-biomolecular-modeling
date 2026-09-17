"""atom_sdpa on CPU: row / registry / installer membership and order (after sampler_hoist), the expected words == the module's
literals, the windowed folded statement against upstream's Attention.forward on the same weights (rank-5 atom-window operands, B in {1, 2},
mask with and without an Lq extent, with and without pair bias, a key extent that is NOT a multiple of 16), the named routes below the
wrapper (rank-4 DiT call, high precision, kv lead, AFO_ATOM_SDPA=0)."""
import importlib

import pytest
import torch

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import installers


def _restore(cls, attr):
    fn = getattr(cls, attr)
    while getattr(fn, "__wrapped_stock__", None) is not None:
        fn = fn.__wrapped_stock__
    setattr(cls, attr, fn)


def test_row_registry_installer_order():
    row = modes.MODES[modes.FAST]
    assert "atom_sdpa" in row and "atom_sdpa" not in modes.MODES[modes.EXACT]
    assert row.index("sampler_hoist") < row.index("atom_sdpa") < row.index("denoiser_graph")
    assert registry.LEVERS["atom_sdpa"]["cls"] == "fast" and "atom_sdpa" in installers()


def test_expected_words_are_the_module_literals():
    from atlasfold_opt.hooks import atom_sdpa
    assert tuple(registry.LEVERS["atom_sdpa"]["expected"]) == atom_sdpa.EXPECTED
    src = open(atom_sdpa.__file__).read()
    for word in atom_sdpa.EXPECTED:
        assert f'ledger.fallback("{word}")' in src, word
    assert 'ledger.fallback("no_fused_backend")' in src and "no_fused_backend" not in atom_sdpa.EXPECTED


def _upstream(att):
    fn = att.Attention.forward
    while getattr(fn, "__wrapped_stock__", None) is not None:
        fn = fn.__wrapped_stock__
    return fn


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("mask_lq", [True, False])
def test_windowed_statement_matches_upstream_on_cpu(B, with_bias, mask_lq, monkeypatch):
    from atlasfold_opt.hooks import atom_sdpa
    att = importlib.import_module(atom_sdpa.TARGET)
    monkeypatch.delenv("AFO_ATOM_SDPA", raising=False); atom_sdpa._STATE["override"] = None
    upstream = _upstream(att)
    ins = atom_sdpa.install("fast", "atlasfold-opt", {})
    try:
        torch.manual_seed(3)
        N, W, Lq, Lk, C, H = 2, 3, 56, 168, 96, 2
        m = att.Attention(C, H).eval()
        a_q = torch.randn(B, N, W, Lq, C); a_k = torch.randn(B, N, W, Lk, C)
        mask = torch.rand(B, 1, W, Lq, Lk) > 0.3 if mask_lq else torch.rand(B, 1, W, 1, Lk) > 0.3
        pb = torch.randn(B, 1, W, H, Lq, Lk) if with_bias else None
        with torch.no_grad():
            ref = upstream(m, a_q, a_k, mask, pb)
            out = m(a_q, a_k, mask, pb)
        assert out.shape == ref.shape == (B, N, W, Lq, C)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, ref, atol=2e-5, rtol=1e-4), float((out - ref).abs().max())
        line = ins.lines[0]()
        assert "name=LOCAL.atlasfold.atom_sdpa state=on" in line and "served=1" in line and f"W{W}xN{N}x{Lq}x{Lk}" in line, line
    finally:
        _restore(att.Attention, "forward"); atom_sdpa._STATE["override"] = None


def test_named_routes_below_the_wrapper(monkeypatch):
    from atlasfold_opt.hooks import atom_sdpa
    att = importlib.import_module(atom_sdpa.TARGET)
    monkeypatch.delenv("AFO_ATOM_SDPA", raising=False); atom_sdpa._STATE["override"] = None
    upstream = _upstream(att)
    ins = atom_sdpa.install("fast", "atlasfold-opt", {})
    try:
        torch.manual_seed(4)
        C, H = 96, 2
        m = att.Attention(C, H).eval()
        with torch.no_grad():
            x = torch.randn(1, 2, 16, C); mask4 = torch.ones(1, 1, 1, 16, dtype=torch.bool)
            assert torch.equal(m(x, x, mask4, None), upstream(m, x, x, mask4, None))              # rank 4 (DiT form): `rank`, the statement below
            a_q = torch.randn(1, 2, 3, 56, C); a_k = torch.randn(1, 1, 3, 168, C); mask = torch.ones(1, 1, 3, 1, 168, dtype=torch.bool)
            ref = upstream(m, a_q, a_k, mask, None); out = m(a_q, a_k, mask, None)                    # kv lead (B,1,W) != (B,N,W): `kv_lead`
            assert torch.equal(out, ref)
            hp = att.Attention(C, H, use_high_precision=True).eval()
            a_k2 = torch.randn(1, 2, 3, 168, C)
            assert torch.equal(hp(a_q, a_k2, mask, None), upstream(hp, a_q, a_k2, mask, None))    # `high_precision`
            monkeypatch.setenv("AFO_ATOM_SDPA", "0")
            assert torch.equal(m(a_q, a_k2, mask, None), upstream(m, a_q, a_k2, mask, None))      # `disabled`
        line = ins.lines[0]()
        for word in ("rank:1", "kv_lead:1", "high_precision:1", "disabled:1"):
            assert word in line, line
        assert "served=0" in line and ins.gates[0]().ok
    finally:
        _restore(att.Attention, "forward"); atom_sdpa._STATE["override"] = None
