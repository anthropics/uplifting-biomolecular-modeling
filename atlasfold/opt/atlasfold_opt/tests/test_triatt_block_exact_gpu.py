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


@pytest.mark.parametrize("N", [512, 640])
@pytest.mark.parametrize("params", ["bf16", "fp32_autocast"])
def test_triatt_block_exact_bitwise_vs_stock(N, params):
    tu = pytest.importorskip("atlasfold.model.network.primitives.triangle_update")
    pytest.importorskip("cuequivariance_torch"); pytest.importorskip("opt_core.attn.pair_fused")
    from atlasfold_opt.hooks import triatt_block_exact as X
    dev = torch.device("cuda")
    origs = {c: getattr(tu, c).forward for c, _ in X.CLASSES}
    g = torch.Generator().manual_seed(1)
    z = (torch.randn(1, N, N, 128, generator=g) * 0.7).to(dev, torch.bfloat16)
    valid = torch.ones(1, N, dtype=torch.bool); valid[:, N - 37:] = False                       # a padded bucket: the last 37 tokens masked
    mk = (valid[:, :, None] & valid[:, None, :]).to(dev)
    try:
        ins = X.install("exact", "[t]", {"mode": "exact", "det": 1})
        assert ins.applied, ins.reason
        for i, (cls_name, ending) in enumerate(X.CLASSES):
            m = _randomize(getattr(tu, cls_name)(128, 4), 10 + i).to(dev)
            if params == "bf16":
                m = m.to(torch.bfloat16)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                ref = origs[cls_name](m, z, mk, kernel_backend="cuequiv")
                out = m(z, mk, kernel_backend="cuequiv")
            assert out.dtype == ref.dtype and out.shape == ref.shape
            assert torch.equal(out, ref), (cls_name, params, N, (out.float() - ref.float()).abs().max().item())
        L = ins.facts["ledger"]
        assert L.served == 2 and not L.fallbacks, (L.served, L.fallbacks)                  # one starting + one ending call, both served
        assert ins.gates[0]().ok
    finally:
        for c, f in origs.items():
            getattr(tu, c).forward = f
