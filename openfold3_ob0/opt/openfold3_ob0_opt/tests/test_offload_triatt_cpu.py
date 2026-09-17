"""The offload port's ending-node triangle attention (`of3_offload.tri_att_end_lean`, the O1 `triatt_lean` lever, and the PairBlock method it
is installed as, `pairblock_tri_att_start_end`) against upstream 0.5.0's own `TriangleAttention` / `PairBlock.tri_att_start_end` on a tiny pair
tensor (CPU, float64; skipped without torch or the openfold3 wheel). Upstream 0.5.0 hands the ending node the transposed VIEW of z with
`transpose_bias=True` — the triangle bias is the un-transposed pair tensor's; a port that kept 0.4.1's orientation fails these by a wide margin."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openfold3", reason="the openfold3 wheel is not installed: the lean ending node is checked against upstream's modules")

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))      # the tree home (opt/openfold3_ob0_opt/tests -> .)
OF3O = os.path.join(HOME, "opt", "forward", "offload", "of3o")


@pytest.fixture(scope="module")
def O():
    sys.path.insert(0, OF3O)
    try:
        import of3_offload
        yield of3_offload
    finally:
        sys.path.remove(OF3O)


def _randomise(module, gen):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype))


def _tiny(N=12, c=8, seed=0):
    from openfold3.core.model.layers.triangular_attention import TriangleAttention
    gen = torch.Generator().manual_seed(seed)
    ta = TriangleAttention(c_in=c, c_hidden=4, no_heads=2, inf=1e9).double().eval()
    _randomise(ta, gen)                                        # upstream's 'final' init zeroes the output projection: random weights make the check meaningful
    z = torch.randn(1, N, N, c, generator=gen, dtype=torch.float64)
    pair_mask = (torch.rand(1, N, N, generator=gen) > 0.15).double()
    return ta, z, pair_mask


def _stock_ending(ta, z, pair_mask, chunk):
    """PairBlock.tri_att_start_end's ending-node statements (base_blocks.py, 0.5.0): the z^T view, transpose_bias=True, add in place, view back."""
    from openfold3.core.utils.tensor_utils import add
    z = z.clone()
    zt = z.transpose(-2, -3)
    zt = add(zt, ta(zt, mask=pair_mask.transpose(-1, -2), transpose_bias=True, chunk_size=chunk, inplace_safe=True), inplace=True)
    return zt.transpose(-2, -3)


@pytest.mark.parametrize("chunk", [4, None])
@pytest.mark.parametrize("ln_block", ["256", "5"])
def test_tri_att_end_lean_equals_upstream_ending_node(O, chunk, ln_block, monkeypatch):
    monkeypatch.setattr(O, "LN_BLOCK", int(ln_block))          # the column block LN(z^T) is built in: one block, or several uneven ones
    monkeypatch.delenv("OF3T_TRIATT", raising=False)
    ta, z, pair_mask = _tiny()
    ref = _stock_ending(ta, z, pair_mask, chunk)
    with torch.no_grad():
        out = O.tri_att_end_lean(ta, z.clone(), pair_mask, chunk, False, False, False, False)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, rtol=0, atol=1e-10), float((out - ref).abs().max())
    # sensitivity: 0.4.1's orientation (the bias of the TRANSPOSED tensor, transpose_bias=False) is a different function of z
    zt = z.clone().transpose(-2, -3)
    old = (zt + ta(zt, mask=pair_mask.transpose(-1, -2), transpose_bias=False, chunk_size=chunk)).transpose(-2, -3)
    assert not torch.allclose(old, ref, rtol=0, atol=1e-6)


def test_pairblock_tri_att_start_end_equals_upstream(O, monkeypatch):
    """The installed method: stock's starting node then the lean ending node == upstream PairBlock.tri_att_start_end (both nodes)."""
    import openfold3.core.config.default_linear_init_config as lin_init
    from openfold3.core.model.latent.pairformer import PairFormerBlock
    monkeypatch.delenv("OF3T_TRIATT", raising=False)
    gen = torch.Generator().manual_seed(1)
    blk = PairFormerBlock(c_s=6, c_z=8, c_hidden_pair_bias=4, no_heads_pair_bias=2, c_hidden_mul=8, c_hidden_pair_att=4, no_heads_pair=2,
                          transition_type="swiglu", transition_n=2, pair_dropout=0.0, fuse_projection_weights=False, inf=1e9,
                          linear_init_params=lin_init.pairformer_init).double().eval()
    pb = blk.pair_stack if hasattr(blk, "pair_stack") else blk
    _randomise(pb, gen)
    N = 10
    z = torch.randn(1, N, N, 8, generator=gen, dtype=torch.float64)
    pair_mask = (torch.rand(1, N, N, generator=gen) > 0.2).double()
    with torch.no_grad():
        ref = pb.tri_att_start_end(z=z.clone(), _attn_chunk_size=4, pair_mask=pair_mask, use_deepspeed_evo_attention=False,
                                   use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False, use_lma=False, inplace_safe=True)
        out = O.pairblock_tri_att_start_end(pb, z.clone(), 4, pair_mask, False, False, False, False, True)
    assert torch.allclose(out, ref, rtol=0, atol=1e-10), float((out - ref).abs().max())
