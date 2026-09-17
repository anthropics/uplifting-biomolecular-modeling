"""dit_apb on a CUDA device with the real provider: the fast and big TIER WORDS serve the bf16 DiT statement inside the fast row's
tolerance of the stock statement at the same operands (B = 1 shared bias and B = 2; the 256- and 896-token eager cells), one launch per batch
element, the census tokens on the line (plan= / rows= / word=<tier>); the exact word serves nothing unvouched (fp32 calls bitwise stock,
stock_row:<arm> by name)."""
import importlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


def _restore(cls, attr):
    fn = getattr(cls, attr)
    while getattr(fn, "__wrapped_stock__", None) is not None:
        fn = fn.__wrapped_stock__
    setattr(cls, attr, fn)


def _real_provider(monkeypatch):
    import opt_core.kernels as K
    monkeypatch.setattr(K, "apb", importlib.import_module("opt_core.kernels.apb"), raising=False)   # the real provider (a test double of an earlier test may still sit on the package attribute)


@pytest.mark.parametrize("mode", ["fast", "big"])
@pytest.mark.parametrize("L", [256, 896])
def test_tier_word_serves_the_dit_statement(monkeypatch, mode, L):
    from atlasfold_opt.hooks import dit_apb
    att = importlib.import_module(dit_apb.TARGET)
    monkeypatch.delenv("AFO_DIT_APB_WORD", raising=False)
    dit_apb._STATE["override"] = None
    _restore(att.Attention, "forward")
    stock = att.Attention.forward
    _real_provider(monkeypatch)
    ins = dit_apb.install(mode, "atlasfold-opt", {})
    try:
        assert ins.facts["word"] == mode
        torch.manual_seed(0)
        C, H = 768, 16
        m = att.Attention(C, H).cuda().eval()
        for B, N in ((1, 5), (2, 2)):
            a = torch.randn(B, N, L, C, device="cuda"); mask = torch.ones(B, 1, 1, L, dtype=torch.bool, device="cuda"); mask[..., L - 7:] = False
            pb = torch.randn(B, 1, H, L, L, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ref = stock(m, a, a, mask, pb); out = m(a, a, mask, pb)
            assert out.shape == ref.shape and out.dtype == ref.dtype == torch.bfloat16
            err = (out.float() - ref.float()).abs().max().item(); scale = ref.float().abs().max().item()
            assert err <= 0.05 * scale, (B, N, err, scale)
        line = ins.lines[0]()
        if "served=0" in line:
            pytest.skip(f"the provider refused the {mode} word's rows here: {line}")
        for tok in ("served=2", f"word={mode}", f"plan={L}e:", "rows="):
            assert tok in line, (tok, line)
        assert ins.gates[0]().ok, line
        print(line)
    finally:
        _restore(att.Attention, "forward")
        dit_apb._STATE["override"] = None


def test_exact_word_serves_nothing_unvouched(monkeypatch):
    from atlasfold_opt.hooks import dit_apb
    att = importlib.import_module(dit_apb.TARGET)
    monkeypatch.delenv("AFO_DIT_APB_WORD", raising=False)
    dit_apb._STATE["override"] = None
    _restore(att.Attention, "forward")
    stock = att.Attention.forward
    _real_provider(monkeypatch)
    ins = dit_apb.install("exact", "atlasfold-opt", {})
    try:
        torch.manual_seed(0)
        C, H, L = 768, 16, 256
        m = att.Attention(C, H).cuda().eval()
        with torch.no_grad():                                                   # fp32 activations (the exact row's statement): below by name, bitwise stock
            a = torch.randn(1, 5, L, C, device="cuda"); mask = torch.ones(1, 1, 1, L, dtype=torch.bool, device="cuda"); pb = torch.randn(1, 1, H, L, L, device="cuda")
            assert torch.equal(m(a, a, mask, pb), stock(m, a, a, mask, pb))
        line = ins.lines[0]()
        assert "served=0" in line and "word=exact" in line and ("stock_row:" in line or "refused:" in line), line
        assert ins.gates[0]().ok, line
        print(line)
    finally:
        _restore(att.Attention, "forward")
        dit_apb._STATE["override"] = None
