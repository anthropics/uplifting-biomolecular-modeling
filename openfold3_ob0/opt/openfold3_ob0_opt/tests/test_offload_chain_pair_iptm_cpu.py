"""The chunked confidence scorer's bespoke-ipTM tail (`of3o_confidence._bespoke_from_chain_pair`: chain_has_frame per SAMPLE, the masked
per-chain mean, the ligand rules, the pair maps) against upstream 0.5.0's `compute_chain_pair_iptm` on random logits (CPU; skipped without
torch or the openfold3 wheel). The chain-pair matrix itself comes from upstream here (the port computes it with BLOCKREDUCE on the GPU);
what is checked is every statement after it — the part upstream changed at 0.5.0 (per-sample frame validity)."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openfold3", reason="the openfold3 wheel is not installed: the bespoke-ipTM tail is checked against upstream's compute_chain_pair_iptm")

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OF3O = os.path.join(HOME, "opt", "forward", "offload", "of3o")


@pytest.fixture(scope="module")
def OC():
    sys.path.insert(0, OF3O)
    try:
        import of3o_confidence
        yield of3o_confidence
    finally:
        sys.path.remove(OF3O)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bespoke_tail_equals_upstream(OC, seed):
    from openfold3.core.metrics.sample_ranking import compute_chain_pair_iptm
    gen = torch.Generator().manual_seed(seed)
    S, N, no_bins = 3, 20, 64
    asym = torch.tensor([0] * 8 + [1] * 7 + [2] * 5)
    batch = {"token_mask": torch.ones(N), "asym_id": asym, "is_ligand": (asym == 2)}
    if seed == 2:
        batch["token_mask"][3] = 0.0
    logits = torch.randn(S, N, N, no_bins, generator=gen)
    has_frame = torch.rand(S, N, generator=gen) > 0.3
    has_frame[1, asym == 1] = False                             # sample 1: chain 1 has no valid frame (per-sample validity is what 0.5.0 changed)
    has_frame[2, asym == 2] = False
    bin_kw = dict(bin_min=0, bin_max=32, no_bins=no_bins)
    ref = compute_chain_pair_iptm(batch=batch, logits=logits, has_frame=has_frame, **bin_kw)
    unique_chains = torch.unique(asym).tolist()
    C = len(unique_chains)
    chain_pair = torch.zeros(S, C, C)
    for i in range(C):
        for j in range(C):
            if i < j:
                v = ref["chain_pair_iptm"][f"({unique_chains[i]},{unique_chains[j]})"]
                chain_pair[:, i, j] = v
                chain_pair[:, j, i] = v
    out = OC._bespoke_from_chain_pair(batch, chain_pair, unique_chains, has_frame, torch.device("cpu"))
    assert set(out) == {"chain_pair_iptm", "bespoke_iptm"} and set(out["bespoke_iptm"]) == set(ref["bespoke_iptm"])
    for k in ref["bespoke_iptm"]:
        assert torch.equal(out["chain_pair_iptm"][k], ref["chain_pair_iptm"][k]), k
        assert torch.allclose(out["bespoke_iptm"][k], ref["bespoke_iptm"][k], rtol=0, atol=1e-6), (k, out["bespoke_iptm"][k], ref["bespoke_iptm"][k])
