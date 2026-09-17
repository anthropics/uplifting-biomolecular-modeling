"""The ENDING triangle attention of the tp line's pair-block binding equals STOCK openfold3 0.5.0 on CPU. Upstream 0.5.0 projects the ending
node's pair bias in z's OWN frame: ``PairBlock.tri_att_start_end`` calls ``tri_att_end(z.transpose(-2, -3), mask=pair_mask.transpose(-1, -2),
transpose_bias=True)`` (``base_blocks.py:387-405``) and ``TriangleAttention.forward`` permutes ``linear_z(layer_norm(x))`` by ``(2, 1, 0)`` instead of
``(2, 0, 1)`` (``triangular_attention.py:166-169``). The binding (``tp_rowpair.pairstack.triatt_fns(tri_att_end, transpose_bias=True)``) runs
``layer_norm`` / ``linear_z`` on each rank's rows of z^T exactly as upstream runs them on ``x``, gathers every rank's bias rows (the core's
``triatt.gather_triangle_bias``) and swaps the two token dims of the gathered ``[N, N, H]`` bias inside ``attend``. Per rank thread of a P-way layout
(``opt_core.testing.run_ranks``: the threaded backend, every collective real): the rows the binding produces must equal stock
``tri_att_end(z^T, mask^T, transpose_bias=True)`` restricted to those rows to fp32 round-off (<= 1e-5), and the ``transpose_bias=False`` binding must
NOT (max |diff| > 1e-3: the orientation is model math at this size, so the test has teeth). Skipped when ``openfold3`` / ``torch`` / the core's
pair-block driver are not importable.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openfold3.core.model.latent.base_blocks")
pytest.importorskip("opt_core.mem.rowpair.pairstack")

from openfold3_ob0_opt.tests.test_tp_rowpair_pairstack_cpu import ACHUNK, CHUNK, F, N, TOL, build   # noqa: E402

P_CASES = [2, 3]


def _rank(rank: int, P: int, n: int):
    torch.set_num_threads(1)
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair.triatt import gather_triangle_bias
    from openfold3_ob0_opt.tp_rowpair import pairstack as PS
    lay = D.ctx(n, align=CHUNK)
    assert int(lay.P) == P, (lay.P, P)
    blk, _stack, z, mask, _s, _smask = build()
    ta = blk.tri_att_end
    zT, maskT = z[0].transpose(-2, -3).contiguous(), mask[0].transpose(-1, -2).contiguous()   # [N, N, C], [N, N]: what PairBlock hands tri_att_end
    out = {"rank": rank, "P": P, "rows": f"{lay.r0}:{lay.r1}", "metrics": {}}
    with torch.no_grad():
        ref = ta(zT.unsqueeze(0), mask=maskT.unsqueeze(0), transpose_bias=True, chunk_size=ACHUNK, **F)[0]   # [N, N, C]: the residual delta PairBlock adds
        for name, transpose_bias in (("ending.z_frame", True), ("ending.transposed_frame", False)):
            fns = PS.triatt_fns(ta, transpose_bias=transpose_bias)
            x = fns.ln(zT[lay.r0:lay.r1])                                # this rank's rows of z^T, LayerNorm'd (triangular_attention.py:160)
            tb_full = gather_triangle_bias(fns.bias(x), lay)              # [N, N, H]: every rank's linear_z(LN(x rows)) — COLLECTIVE
            o = fns.attend(x, maskT[lay.r0:lay.r1], tb_full, (0, int(lay.R)))   # [R, N, C]
            got, want = o, ref[lay.r0:lay.r1]
            out["metrics"][name] = {"maxabs": float((want - got).abs().max()), "equal": bool(torch.equal(want, got)), "finite": bool(torch.isfinite(got).all())}
    return out


@pytest.mark.parametrize("P", P_CASES)
def test_triatt_ending_bias_frame_vs_stock(P):
    from opt_core.testing import run_ranks
    res = run_ranks(P, _rank, N, timeout_s=600.0)
    assert len(res) == P, res
    for r in res:
        ok, wrong = r["metrics"]["ending.z_frame"], r["metrics"]["ending.transposed_frame"]
        assert ok["finite"] and ok["maxabs"] <= TOL, r                    # the kit's binding equals stock's ending node
        assert wrong["maxabs"] > 100 * TOL, r                             # the (2, 0, 1) orientation would not: the test has teeth


def test_ending_node_is_bound_with_transpose_bias():
    """``pair_block_fns`` / ``pairformer_block_fns`` bind ``tri_att_end`` with ``transpose_bias=True`` and ``tri_att_start`` without (a source pin)."""
    import inspect
    from openfold3_ob0_opt.tp_rowpair import pairstack as PS
    for fn in (PS.pair_block_fns, PS.pairformer_block_fns):
        src = inspect.getsource(fn)
        assert "triatt_end=triatt_fns(blk.tri_att_end, transpose_bias=True)" in src, fn.__name__
        assert "triatt_start=triatt_fns(blk.tri_att_start)" in src, fn.__name__
