"""`of3_offload.embed_zij_rows` (the O1 heads unit: the confidence heads' pair embedding added IN PLACE per row block on the one GPU copy,
the per-sample distance term after one materialised broadcast) against upstream 0.5.0's `PairformerEmbedding.embed_zij` statement, bound to a
stand-in module carrying exactly the members the statement reads (CPU, float64; skipped without torch or the openfold3 wheel)."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openfold3", reason="the openfold3 wheel is not installed: embed_zij rows are checked against upstream's PairformerEmbedding.embed_zij")

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OF3O = os.path.join(HOME, "opt", "forward", "offload", "of3o")


@pytest.fixture(scope="module")
def O():
    sys.path.insert(0, OF3O)
    try:
        import of3_offload
        yield of3_offload
    finally:
        sys.path.remove(OF3O)


class _PE(torch.nn.Module):
    """The members PairformerEmbedding.embed_zij reads (prediction_heads.py): linear_i, linear_j, linear_distance, min_bin, max_bin, no_bin, inf."""
    def __init__(self, c_s_input, c_z, no_bin, gen):
        super().__init__()
        self.min_bin, self.max_bin, self.no_bin, self.inf = 3.25, 50.75, no_bin, 1e8
        self.linear_i = torch.nn.Linear(c_s_input, c_z).double()
        self.linear_j = torch.nn.Linear(c_s_input, c_z).double()
        self.linear_distance = torch.nn.Linear(no_bin, c_z).double()
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype))

    from openfold3.core.model.heads.prediction_heads import PairformerEmbedding as _Up
    embed_zij = _Up.embed_zij                                                   # upstream's statement, verbatim, bound to the stand-in


@pytest.mark.parametrize("si_per_sample", [False, True])
@pytest.mark.parametrize("samples", [1, 3])
@pytest.mark.parametrize("rows", ["4", "7", "0"])
def test_embed_zij_rows_equals_upstream(O, samples, rows, si_per_sample, monkeypatch):
    monkeypatch.setenv("OF3O_ROWS", rows)
    gen = torch.Generator().manual_seed(int(rows) + samples)
    N, c_s_input, c_z, no_bin = 11, 6, 8, 15
    pe = _PE(c_s_input, c_z, no_bin, gen).eval()
    si_input = torch.randn(1, samples if si_per_sample else 1, N, c_s_input, generator=gen, dtype=torch.float64)   # the model's form: a size-1 sample axis
    zij = torch.randn(1, 1, N, N, c_z, generator=gen, dtype=torch.float64)          # below the per-sample cutoff every sample shares one zij: stock broadcasts it
    x_pred = torch.randn(1, samples, N, 3, generator=gen, dtype=torch.float64) * 12.0
    with torch.no_grad():
        ref = pe.embed_zij(si_input=si_input, zij=zij, x_pred=x_pred)
        out = O.embed_zij_rows(pe, si_input, zij.clone(), x_pred)
    assert out.shape == ref.shape == (1, samples, N, N, c_z)
    assert torch.allclose(out, ref, rtol=0, atol=1e-11), float((out - ref).abs().max())


def test_embed_zij_rows_names_a_fallback_when_the_projection_is_narrower(O, monkeypatch):
    """In-place adds cannot promote: projections narrower than zij (an autocast state narrower than the pair tensor) fall back to stock BY NAME."""
    gen = torch.Generator().manual_seed(5)
    pe = _PE(6, 8, 15, gen).eval().float()                                        # fp32 projections ...
    si_input = torch.randn(1, 1, 9, 6, generator=gen)
    zij = torch.randn(1, 1, 9, 9, 8, generator=gen, dtype=torch.float64)          # ... into an fp64 pair tensor
    x_pred = torch.randn(1, 1, 9, 3, generator=gen)
    before = dict(O.STATE["fallbacks"])
    with torch.no_grad():
        out = O.embed_zij_rows(pe, si_input, zij.clone(), x_pred)
        ref = pe.embed_zij(si_input=si_input, zij=zij, x_pred=x_pred)
    assert torch.allclose(out, ref) and O.STATE["fallbacks"].get("PairformerEmbedding.embed_zij", 0) == before.get("PairformerEmbedding.embed_zij", 0) + 1
