"""Unit tests for ef2_bf16_confidence (CUDA; random-initialised small ESMFold2-Experimental model, no checkpoint).

    pytest -q k/test_ef2_bf16_confidence.py        or:   python k/test_ef2_bf16_confidence.py

Standard (FAST-class): the training path (distogram, input gradient) and the sampled coordinates stay BITWISE (the head feeds nothing
back); confidence outputs keep their stock dtypes and shapes and stay close to the fp32 head's (|Δ iptm|, |Δ ptm| small on the fixture);
disable() restores the fp32 head bit for bit."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
try:
    import pytest
    cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
except ImportError:
    cuda = lambda f: f
import _ef2_testmodel as T
import ef2_bf16_confidence as bc


@cuda
def test_bf16_confidence_head():
    model = T.small_model(confidence=True); feats = T.design_inputs()
    out0, d0, g0 = T.assert_floor(model, feats, seed=4, num_sampling_steps=3)
    assert bc.enable(model) is True; bc.enable(model)
    out1, d1, g1 = T.run_design_call(model, feats, seed=4, num_sampling_steps=3)
    assert torch.equal(d0, d1) and torch.equal(g0, g1) and torch.equal(out0["sample_atom_coords"], out1["sample_atom_coords"]), "gradient path / coords must be untouched"
    for k, v in out0.items():
        if torch.is_tensor(v) and v.is_floating_point():
            assert out1[k].dtype == v.dtype and out1[k].shape == v.shape, (k, v.dtype, out1[k].dtype)
            assert torch.isfinite(out1[k]).all(), k
    d_iptm = (out0["iptm"] - out1["iptm"]).abs().max().item(); d_ptm = (out0["ptm"] - out1["ptm"]).abs().max().item()
    print(f"  fixture |d iptm| {d_iptm:.2e} |d ptm| {d_ptm:.2e} (iptm {out0['iptm'].flatten()[0].item():.4f})")
    assert d_iptm < 2e-2 and d_ptm < 2e-2, (d_iptm, d_ptm)
    bc.disable(model)
    assert "forward" not in vars(model.confidence_head)
    out2, _, _ = T.run_design_call(model, feats, seed=4, num_sampling_steps=3)
    assert torch.equal(out2["iptm"], out0["iptm"]) and torch.equal(out2["pae_logits"], out0["pae_logits"])
    assert bc.enable(T.small_model(confidence=False)) is False


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name, flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback; fails += 1; print("FAIL", name, type(e).__name__, e); traceback.print_exc()
    print(f"RESULT {'OK' if not fails else 'FAILED'} ({fails} failed)"); sys.exit(1 if fails else 0)
