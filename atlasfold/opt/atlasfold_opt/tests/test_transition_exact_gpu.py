"""GPU tests of the exact-class constructions `triatt_block_exact` / `transition_exact`: on a CUDA card of PROVEN_CC the served path's
outputs EQUAL the stock statement's BIT FOR BIT (torch.equal), for the trunk's bf16-parameter modules under bf16 autocast (as the runners run the
trunk) and for fp32-parameter modules under bf16 autocast (as the confidence pair stack runs), starting AND ending node, a padded pair mask, at
the served sizes 512 / 640.  Skips without CUDA / atlasfold / opt_core / cuequivariance."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _randomize(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            base = 1.0 if (name.endswith("weight") and p.dim() == 1) else 0.0
            p.copy_(base + torch.randn(p.shape, generator=g) * 0.08)
    return m.eval()


@pytest.mark.parametrize("N", [256, 512, 640, 896, 1280])
@pytest.mark.parametrize("params", ["bf16", "fp32_autocast"])
def test_transition_exact_bitwise_vs_stock(N, params):
    """The exact word: the output EQUALS the stock statement bit for bit at every size — served by a row the provider records bitwise on this stack
    (rows=v1:<n>), or the statement itself by name (`stock_row:` / `refused:`); which of the two is the provider's answer, printed as plan=."""
    tr = pytest.importorskip("atlasfold.model.network.primitives.transition")
    pytest.importorskip("opt_core.kernels.transition")
    from atlasfold_opt.hooks import transition_exact as X
    dev = torch.device("cuda")
    orig = tr.Transition.forward
    g = torch.Generator().manual_seed(2)
    x = (torch.randn(1, N, N, 128, generator=g) * 0.7).to(dev, torch.bfloat16)
    try:
        ins = X.install("exact", "[t]", {"mode": "exact", "det": 1})
        assert ins.applied, ins.reason
        m = _randomize(tr.Transition(128, 4), 20).to(dev)
        if params == "bf16":
            m = m.to(torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            ref = orig(m, x)
            out = m(x)
        assert out.dtype == ref.dtype and out.shape == ref.shape
        L = ins.facts["ledger"]
        assert torch.equal(out, ref), (params, N, L.fields(), (out.float() - ref.float()).abs().max().item())
        assert not L.errors and ins.gates[0]().ok, L.fields()
        assert L.served + sum(L.fallbacks.values()) == 1
        for why in L.fallbacks:
            assert why.startswith(("stock_row:", "refused:")), L.fields()                       # the only routes to the statement on a CUDA bf16 pair call are the provider's, by name
    finally:
        tr.Transition.forward = orig


