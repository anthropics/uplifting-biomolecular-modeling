"""opt_core.mem.rowpair — the distributed transposes of :mod:`rowpair.ring` and the pair-block driver's ending-attention orientation
(``pairstack.pair_block_(transpose_inplace=True)`` = ``ring.transpose_shard_inplace_``, what a kit's ``ROWPAIR_TRANSPOSE_INPLACE=1`` selects) on
layouts that leave a ONE-ROW shard. ``Layout.auto(257, 2)`` is the grid ``B=128``: rows ``[0, 256)`` on rank 0 and ``[256, 257)`` on rank 1, so
rank 1's diagonal block is ``1 x 1`` — the transposed view of a ``1 x 1`` block is already contiguous and must still go through a real temporary.
Everything is asserted BIT-EXACT (``torch.equal``) against the dense ``z^T`` / the dense statement of the block: that geometry, a sweep of every
``N mod P`` remainder x ``P in {2, 3, 4, 8}`` through ``Layout.auto`` (the plan the kits use), one-row shards built by hand under both layout
policies (grid, aligned) including ``B=1`` (EVERY rank one row) with a leading dim of 1 and of 2, and the driver on the failing geometry. Ranks are threads of one process
(:func:`opt_core.testing.run_ranks`); CPU; seconds.

Run: ``python -m pytest tests/test_rowpair_transpose_onerow_0518.py -q -rfE``.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from opt_core.mem.rowpair import dist as D, pairstack as PS, ring as RG  # noqa: E402
from opt_core.mem.rowpair.dist import Layout  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402


def _gen(seed):
    return torch.Generator().manual_seed(int(seed))


def _layout(how, N, P, rank):
    kind = how[0]
    if kind == "auto":
        return Layout.auto(N, P, rank)
    if kind == "grid":
        return Layout(N, P, rank, B=int(how[1]))
    if kind == "align":
        return Layout(N, P, rank, B=int(how[1]), align=int(how[1]))
    raise ValueError(how)


# ================================================================================================ the geometry of the defect, stated once
def test_layout_auto_257_over_2_leaves_one_row_on_rank_1():
    assert [Layout.auto(257, 2, r).bounds[r] for r in range(2)] == [(0, 256), (256, 257)] and Layout.auto(257, 2, 1).R == 1
    assert [Layout.auto(257, 3, r).R for r in range(3)] == [128, 128, 1]                      # the same one-row last rank at P=3
    assert [Layout.auto(257, 4, r).R for r in range(4)] == [80, 80, 80, 17]                  # ... and none at P=4 (B=16)
    assert [Layout.auto(244, 2, r).R for r in range(2)] == [128, 116]                        # the 244-token protein-only query: none


# ================================================================================================ ring transposes on threaded ranks, bit-exact
def _transposes_worker(rank, P, z, how):
    N = int(z.shape[-2])
    lay = _layout(how, N, P, rank)
    assert (lay.P, lay.rank) == (P, rank) and not lay.replicated and lay.R >= 1
    zl = z[..., lay.r0:lay.r1, :, :].clone()                                                  # [L, n_loc, N, C]: this rank's rows, its own storage (as a kit's shard)
    ztr = z.transpose(-2, -3)[..., lay.r0:lay.r1, :, :]                                      # dense reference: this rank's rows of z^T
    ok = {}
    ok["transpose_shard"] = torch.equal(RG.transpose_shard(zl, lay), ztr)
    for k in (1, 2):
        ok[f"transpose_streamed_peers{k}"] = torch.equal(RG.transpose_shard_streamed(zl, lay, peers_in_flight=k), ztr)
    for k in (1, 2):
        zi = zl.clone()
        ok[f"transpose_inplace_peers{k}"] = RG.transpose_shard_inplace_(zi, lay, peers_in_flight=k) is zi and torch.equal(zi, ztr)
        ok[f"transpose_inplace_roundtrip_peers{k}"] = RG.transpose_shard_inplace_(zi, lay, peers_in_flight=k) is zi and torch.equal(zi, zl)
    z3 = zl[0].clone()                                                                       # the [n_loc, N, C] form the pair-block driver holds
    zi3 = z3.clone()
    ok["transpose_inplace_3d"] = RG.transpose_shard_inplace_(zi3, lay) is zi3 and torch.equal(zi3, ztr[0])
    ok["transpose_shards_a2a"] = torch.equal(D.transpose_shards(z3, lay, chunks=2), ztr[0])
    mask = (z[0, :, :, 0] > 0).to(z.dtype)                                                 # [N, N], not symmetric
    ok["mask_transposed"] = torch.equal(PS.mask_transposed(mask[lay.r0:lay.r1].contiguous(), lay), mask.t()[lay.r0:lay.r1])
    return ok, repr(lay)


def _failures(res):
    return [(r, k) for r, (ok, _) in enumerate(res) for k, v in ok.items() if not v]


ONE_ROW_AUTO = [(257, 2), (257, 3)]                                                          # Layout.auto plans whose last rank owns one row
SWEEP = [(base + rem, P) for P, base in ((2, 256), (3, 255), (4, 256), (8, 1024)) for rem in range(P)]   # every N mod P remainder per P


@pytest.mark.parametrize("N,P", ONE_ROW_AUTO + [c for c in SWEEP if c not in ONE_ROW_AUTO])
def test_transposes_bitwise_on_layout_auto(N, P):
    z = torch.randn(1, N, N, 2, generator=_gen(1000 * P + N))
    res = run_ranks(P, _transposes_worker, z, ("auto",))
    assert _failures(res) == [], (_failures(res), [lay for _, lay in res])


def _one_row_by_hand():
    cases = []
    for P in (2, 3, 4, 8):
        bpr = min(2, P)
        cases.append((("grid", 4), (P - 1) * bpr * 4 + 1, P))                                # grid B=4: the last rank owns exactly one row
        cases.append((("align", 4), 4 * P + 1, P))                                           # aligned k=4: the ragged last chunk is one row
        cases.append((("grid", 1), P, P))                                                    # B=1, N=P: EVERY rank owns one row
    return cases


@pytest.mark.parametrize("how,N,P", _one_row_by_hand(), ids=lambda v: f"{v[0]}{v[1]}" if isinstance(v, tuple) else None)
@pytest.mark.parametrize("L", [1, 2], ids=["lead1", "lead2"])                              # lead 1: the transposed 1x1 view IS contiguous; lead 2: it is not
def test_transposes_bitwise_on_one_row_shards(how, N, P, L):
    lays = [_layout(how, N, P, r) for r in range(P)]
    assert lays[-1].R == 1 and all(l.R >= 1 for l in lays), [l.bounds for l in lays]
    z = torch.randn(L, N, N, 3, generator=_gen(7 * P + N + 1000 * L))
    res = run_ranks(P, _transposes_worker, z, how)
    assert _failures(res) == [], (_failures(res), [lay for _, lay in res])


# ================================================================================================ the pair-block driver on the failing geometry
def _colscale(w):
    """``fn(x, mask, layout) -> x``: ``x[.., j, :] += 0.5 * x[.., j, :] * w[j]`` IN PLACE over the columns j of the shard it is handed (a stand-in
    for a pair update; NOT transpose-equivariant, so the ending update only equals the dense statement if the shard really holds rows of z^T)."""
    def fn(x, mask, layout):
        x.add_(0.5 * x * w[None, :, None].to(x.dtype))
        return x
    return fn


def _dense_block(z, w):
    z = z.clone()
    for key in ("trimul_out", "trimul_in", "triatt_start"):
        z += 0.5 * z * w[key][None, :, None]
    zT = z.transpose(0, 1).contiguous()
    zT += 0.5 * zT * w["triatt_end"][None, :, None]
    z = zT.transpose(0, 1).contiguous()
    z += 0.5 * z * w["transition"][None, :, None]
    return z


def _block_worker(rank, P, z, w, depth, inplace):
    N = int(z.shape[0])
    lay = Layout.auto(N, P, rank)
    fns = PS.PairBlockFns(*[_colscale(w[k]) for k in ("trimul_out", "trimul_in", "triatt_start", "triatt_end", "transition")])
    zl = z[lay.r0:lay.r1].clone()                                                           # this rank's rows, its own storage (as a kit's shard)
    got, _ = PS.pair_stack_([fns] * depth, zl, None, lay, transpose_inplace=inplace)
    ref = z
    for _ in range(depth):
        ref = _dense_block(ref, w)
    return {"rows": (lay.r0, lay.r1), "bitwise": bool(torch.equal(got, ref[lay.r0:lay.r1])), "same_object": got is zl}


@pytest.mark.parametrize("N,P", [(257, 2), (257, 3), (257, 4), (244, 2)])
@pytest.mark.parametrize("inplace", [True, False])
def test_pair_stack_driver_equals_dense_on_the_one_row_geometry(N, P, inplace):
    g = _gen(N * 10 + P)
    z = torch.randn(N, N, 4, generator=g)
    w = {k: torch.rand(N, generator=g) for k in ("trimul_out", "trimul_in", "triatt_start", "triatt_end", "transition")}
    res = run_ranks(P, _block_worker, z, w, 2, inplace)
    assert all(r["bitwise"] and r["same_object"] for r in res), res
    if (N, P) == (257, 2):
        assert res[1]["rows"] == (256, 257)
