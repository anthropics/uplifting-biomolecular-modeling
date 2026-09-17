"""cells/dit_attn.py `_bias_planes`: the DiT pair bias reaches the core's pair-bias attention KEY-CONTIGUOUS. Upstream's bias is its
[N, N, H] projection permuted to [H, N, N] (key stride H); at S = 1 the core's fused rows read the planes directly and take unit key stride only
(opt_core kernels.apb fpf_apb.apb_views asserted on it: big/resident at 5,060 tokens with one diffusion sample, rc 1). The cell re-lays such a
view once (values unchanged) and passes an already key-contiguous bias through untouched."""
import pytest

torch = pytest.importorskip("torch", reason="needs torch")

from openfold3_ob0_opt.cells.dit_attn import _bias_planes


@pytest.mark.parametrize("nb,H,N", [(1, 16, 24), (1, 4, 7), (2, 16, 12)])
def test_upstreams_permuted_view_comes_out_key_contiguous_and_equal(nb, H, N):
    z = torch.randn(nb, N, N, H)                                   # upstream: LayerNorm+Linear on z -> [.., N, N, H]
    view = z.permute(0, 3, 1, 2)                                    # -> [.., H, N, N], key stride H (what AttentionPairBias hands on)
    assert view.stride(-1) == H != 1
    pb = _bias_planes(view, nb, H, N)
    assert pb.shape == (nb, H, N, N) and pb.stride(-1) == 1
    assert torch.equal(pb, view.reshape(nb, H, N, N))              # values unchanged (bitwise)


@pytest.mark.parametrize("nb,H,N", [(1, 16, 24), (5, 16, 8)])
def test_a_key_contiguous_bias_passes_through_untouched(nb, H, N):
    b = torch.randn(nb, H, N, N)
    pb = _bias_planes(b, nb, H, N)
    assert pb.data_ptr() == b.data_ptr() and pb.stride() == b.stride()          # no copy
    flat = torch.randn(nb * H, N, N)                                             # a reshape-able contiguous source: still no copy
    pf = _bias_planes(flat, nb, H, N)
    assert pf.data_ptr() == flat.data_ptr() and pf.stride(-1) == 1
