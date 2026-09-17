"""Unit tests for ef2_lazy_structure (CUDA; random-initialised small ESMFold2-Experimental model, no checkpoint).

    pytest -q k/test_ef2_lazy_structure.py        or, without pytest:   python k/test_ef2_lazy_structure.py

Standard: exact lever — every training-path tensor (distogram, input gradient) BITWISE equal to stock with the lever on; a deferred
sample that is read later equals the eagerly computed one BITWISE; a call without a seed, or with the confidence head present, is the
stock (eager) path; disable() restores the stock forward."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
try:
    import pytest
    cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
except ImportError:                                            # pytest-free images: the __main__ runner below
    cuda = lambda f: f
import _ef2_testmodel as T
import ef2_lazy_structure as els


@cuda
def test_deferred_is_bitwise_and_materializes_bitwise():
    model = T.small_model(confidence=False)                     # no confidence head: nothing inside forward reads the coordinates
    feats = T.design_inputs(); T.assert_floor(model, feats, seed=7)
    out0, d0, g0 = T.run_design_call(model, feats, seed=7)      # stock, eager
    c0 = out0["sample_atom_coords"].clone()
    els.enable(model); els.enable(model)                        # idempotent
    out1, d1, g1 = T.run_design_call(model, feats, seed=7)
    assert isinstance(out1, els.LazyOutput) and out1.lazy_state() == "deferred", type(out1)
    assert torch.equal(d0, d1), "distogram differs with the lever on"
    assert torch.equal(g0, g1), "input gradient differs with the lever on"
    junk = torch.randn(1000, device=T.dev)                      # move the ambient CUDA RNG and do unrelated work before the read
    c1 = out1["sample_atom_coords"]                             # materialise now
    assert out1.lazy_state() == "materialized" and torch.equal(c0, c1), "materialised coordinates differ from the eager ones"
    assert els.stats(model) == dict(deferred=1, materialized=1, eager=0), els.stats(model)
    # every dict read path materialises (fresh call): values(), dict(...), other.update(...)
    out2, _, _ = T.run_design_call(model, feats, seed=7)
    assert torch.equal(dict(out2)["sample_atom_coords"], c0)
    out3, _, _ = T.run_design_call(model, feats, seed=7); plain = {}; plain.update(out3)
    assert torch.is_tensor(plain["sample_atom_coords"]) and torch.equal(plain["sample_atom_coords"], c0)
    els.disable(model)
    assert not hasattr(model, "_els_orig_forward")
    out4, d4, _ = T.run_design_call(model, feats, seed=7)
    assert type(out4) is dict and torch.equal(out4["sample_atom_coords"], c0) and torch.equal(d4, d0)


@cuda
def test_eager_when_unseeded_or_confidence_runs():
    model = T.small_model(confidence=True)                      # the confidence head reads the coordinates inside forward -> eager
    feats = T.design_inputs(); T.assert_floor(model, feats, seed=3, num_sampling_steps=3)
    out0, d0, g0 = T.run_design_call(model, feats, seed=3, num_sampling_steps=3)
    els.enable(model)
    out1, d1, g1 = T.run_design_call(model, feats, seed=3, num_sampling_steps=3)
    assert type(out1) is dict, "must not defer when the confidence head runs"
    for k in ("sample_atom_coords", "iptm", "ptm", "pae_logits"):
        assert torch.equal(out0[k], out1[k]), k
    assert torch.equal(d0, d1) and torch.equal(g0, g1)
    with torch.no_grad():
        out2 = model(**{k: v for k, v in feats.items() if k != "res_type_soft"}, num_diffusion_samples=1, num_sampling_steps=2)   # no seed -> eager
    assert type(out2) is dict and torch.is_tensor(out2["sample_atom_coords"])
    assert els.stats(model)["deferred"] == 0 and els.stats(model)["eager"] == 2, els.stats(model)
    els.disable(model)


@cuda
def test_with_skip_unused_confidence_if_available():
    """The kit's exact set removes the confidence head for calculate_confidence=False calls; the lever must defer under it and stay
    eager (and bitwise) on a confidence call."""
    try:
        import ef2_autograd_kernels as agk
    except ImportError:
        return
    model = T.small_model(confidence=True); feats = T.design_inputs(); T.assert_floor(model, feats, seed=5)
    agk.enable_skip_unused_confidence(model)
    ref, rd, rg = T.run_design_call(model, feats, seed=5, num_sampling_steps=1, calculate_confidence=False)
    refc, rdc, rgc = T.run_design_call(model, feats, seed=5, num_sampling_steps=4, calculate_confidence=True)
    els.enable(model)
    out, d, g = T.run_design_call(model, feats, seed=5, num_sampling_steps=1, calculate_confidence=False)
    assert isinstance(out, els.LazyOutput) and out.lazy_state() == "deferred"
    assert torch.equal(d, rd) and torch.equal(g, rg) and torch.equal(out["sample_atom_coords"], ref["sample_atom_coords"])
    outc, dc, gc = T.run_design_call(model, feats, seed=5, num_sampling_steps=4, calculate_confidence=True)
    assert type(outc) is dict and torch.equal(outc["iptm"], refc["iptm"]) and torch.equal(outc["sample_atom_coords"], refc["sample_atom_coords"]) and torch.equal(dc, rdc)
    els.disable(model); agk.disable_skip_unused_confidence(model)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name, flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback; fails += 1; print("FAIL", name, type(e).__name__, e); traceback.print_exc()
    print(f"RESULT {'OK' if not fails else 'FAILED'} ({fails} failed)"); sys.exit(1 if fails else 0)
