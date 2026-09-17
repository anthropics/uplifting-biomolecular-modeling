"""The ENDING-node triangle attention of the stock openfold3 (>= 0.5.0) PairBlock, on CPU, against a plain-torch re-statement parameterised by
the pair-bias frame the tree's core names (opt_core.attn.pair_fused `bias_frame`: 'x' = the bias projected from the transposed, normalised
attention-frame input; 'z' = projected from norm(z) in z's OWN frame == the x-frame bias with its two token axes swapped). Upstream's
`PairBlock.tri_att_start_end` calls `tri_att_end(zᵀ, mask=maskᵀ, transpose_bias=True)`; this test shows that call equals the 'z'-frame
re-statement and differs from the 'x'-frame one — the fact `cells/pairfused.py` (BIAS_FRAME[True] == 'z') stands on. No GPU, no kit kernels."""
import math

import pytest

torch = pytest.importorskip("torch")
try:
    from openfold3.core.model.layers.triangular_attention import TriangleAttention
    HAVE_OF3 = True
except Exception:                                                       # noqa: BLE001
    HAVE_OF3 = False

from openfold3_ob0_opt.cells import pairfused

needs_of3 = pytest.mark.skipif(not HAVE_OF3, reason="needs the openfold3 wheel (TriangleAttention)")


def _restated_node(ta, z, mask, *, ending: bool, bias_frame: str):
    """The triangle-attention update of node `ta` on pair rep z [N, N, C] with pair mask [N, N], plain torch, fp64: q/k/v/g from the attention
    frame x (= zᵀ for the ending node), the pair bias from frame `bias_frame`; returns the update in z's frame."""
    zt = z.transpose(-2, -3) if ending else z
    mt = mask.transpose(-1, -2) if ending else mask
    x = ta.layer_norm(zt).double()                                     # [I, J, C] attention frame
    H, D = ta.mha.no_heads, ta.mha.c_hidden
    W = lambda lin: lin.weight.double()                                 # noqa: E731
    q = (x @ W(ta.mha.linear_q).T).view(*x.shape[:-1], H, D).transpose(-2, -3) / math.sqrt(D)   # [I, H, J, D]
    k = (x @ W(ta.mha.linear_k).T).view(*x.shape[:-1], H, D).transpose(-2, -3)
    v = (x @ W(ta.mha.linear_v).T).view(*x.shape[:-1], H, D).transpose(-2, -3)
    bias_x = (x @ W(ta.linear_z).T).permute(2, 0, 1)                    # [H, a, b] = W_b · x[a, b]      (the 'x' frame)
    bias = bias_x.transpose(-1, -2) if (bias_frame == "z" and ending) else bias_x   # 'z' on the ending node: token axes swapped = W_b · norm(z)[a, b] in z's
    #                                                                                     own index order (pair_fused.prologue: the two frames coincide when x = z)
    logits = q @ k.transpose(-1, -2) + bias.unsqueeze(0) + (1e9 * (mt.double() - 1))[:, None, None, :]   # [I, H, J(q), J(k)]
    o = torch.softmax(logits, -1) @ v                                   # [I, H, J, D]
    o = o.transpose(-2, -3).reshape(*x.shape[:-1], H * D)
    g = torch.sigmoid(x @ W(ta.mha.linear_g).T)
    u = (o * g) @ W(ta.mha.linear_o).T + (ta.mha.linear_o.bias.double() if ta.mha.linear_o.bias is not None else 0)
    return u.transpose(-2, -3) if ending else u                         # back to z's frame


@needs_of3
@pytest.mark.parametrize("ending", [False, True])
def test_upstream_node_matches_the_frame_the_cell_passes(ending):
    torch.manual_seed(0)
    N, C, Hd, H = 7, 8, 4, 2
    ta = TriangleAttention(C, Hd, H, inf=1e9).double().eval()
    with torch.no_grad():
        for p in ta.parameters():                                       # upstream's init leaves gates/outputs at fixed points; randomise everything
            p.copy_(torch.randn_like(p) * 0.5)
    z = torch.randn(N, N, C, dtype=torch.float64)
    mask = (torch.rand(N, N) > 0.25).double()
    mask[:, 0] = 1.0                                                    # every key row keeps at least one position
    with torch.no_grad():
        zt = z.transpose(-2, -3) if ending else z                        # exactly PairBlock.tri_att_start_end's call (latent/base_blocks.py)
        mt = mask.transpose(-1, -2) if ending else mask
        stock = ta(zt, mask=mt, transpose_bias=ending)
        stock = stock.transpose(-2, -3) if ending else stock
        frame = pairfused.BIAS_FRAME[ending]
        same = _restated_node(ta, z, mask, ending=ending, bias_frame=frame)
        other = _restated_node(ta, z, mask, ending=ending, bias_frame=("x" if frame == "z" else "z"))
    assert torch.allclose(stock, same, atol=1e-9, rtol=1e-9), float((stock - same).abs().max())
    if ending:                                                          # the starting node's two frames coincide by definition; the ending node's do not
        assert not torch.allclose(stock, other, atol=1e-6), "the 0.4.x ('x') frame must NOT reproduce the 0.5.0 ending node"
