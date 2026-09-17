"""opt_core.mem.rowpair 0.4.3 substrate — the comm surface (solo | threaded), the Layout ``align`` policy and its Ctx aliases, the p2p-built
transposes / ring passes of :mod:`rowpair.ring`, ``shard.produce_rows_``, ``bcast.sync_tensordict_from_rank0``, the per-cycle checkpoint writer,
and the exact-finish confidence reducer ON the threaded backend. Ranks are threads of one process (:func:`opt_core.testing.run_ranks`), P in
{1, 2, 3, 4}; data movement is asserted BIT-EXACT (``torch.equal``) against the dense tensor every rank was handed; uneven N throughout.

Run: ``python -m pytest tests/test_rowpair_shard_043.py -q -rfE`` (torch required; skipped without it).
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import bcast as BC, ckpt as CK, dist as D, ring as RG, shard as S  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402

PS = (1, 2, 3, 4)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _all_true(results):
    bad = sorted({k for ok, _ in results for k, v in ok.items() if v is not True})
    return bad


# ================================================================================================================ pure layout
def test_row_parts_and_layout_align_policy():
    for N in (1, 7, 16, 37, 128, 129, 1000, 2501):
        for P in (1, 2, 3, 4, 8):
            for align in (1, 4, 16, 128):
                parts = D.row_parts(N, P, align)
                assert len(parts) == P and parts[0][0] == 0 and parts[-1][1] == N
                assert all(parts[q][1] == parts[q + 1][0] for q in range(P - 1))
                assert all(q0 % align == 0 or q0 == N for q0, _ in parts)          # starts on the chunk grid (empty trailing ranks sit at N)
                nch = [-(-(q1 - q0) // align) for q0, q1 in parts]
                assert max(nch) - min(nch) <= 1, (N, P, align, parts)                  # balanced in chunks
                n_chunks = -(-N // align)
                if n_chunks >= P:
                    lays = [D.Layout.checked(N, P, q, align=align) for q in range(P)]
                    for q, lay in enumerate(lays):
                        assert lay.parts == parts and lay.bounds == parts and (lay.r0, lay.r1) == parts[q]
                        assert lay.n_loc == lay.R == parts[q][1] - parts[q][0] and lay.n_max == lay.Rmax == max(b - a for a, b in parts)
                        assert lay.rows_of(q) == parts[q] and lay.align == align and lay.B == align and lay.policy == "aligned"
                        assert not lay.replicated
                        for i in range(0, N, max(1, N // 13)):
                            o = lay.owner(i)
                            assert parts[o][0] <= i < parts[o][1]
                        cs = list(lay.chunks(align))
                        assert sum(c1 - c0 for c0, c1 in cs) == lay.R and all(c0 % align == 0 for c0, _ in cs)
                elif P > 1:
                    with pytest.raises(RowpairRefused):
                        D.Layout.checked(N, P, 0, align=align)


def test_layout_grid_policy_unchanged():
    """align=None is the P-invariant grid: bounds on multiples of B, replicated iff N < P*B, padded_is_global."""
    for N in (15, 128, 300, 1000):
        for P in (1, 2, 3, 4):
            for B in (16, 64, 128):
                lay = D.Layout(N, P, 0, B)
                assert lay.policy == "grid" and lay.align is None and lay.B == B
                if N < P * B:
                    assert lay.replicated and all(b == (0, N) for b in lay.bounds)
                else:
                    assert lay.bounds == [(min(N, q * lay.Rmax), min(N, (q + 1) * lay.Rmax)) for q in range(P)]
                    assert lay.padded_is_global and lay.parts is lay.bounds and lay.n_loc == lay.R and lay.n_max == lay.Rmax
    assert D.Layout.auto(1000, 4, 3).B == 128 and D.Layout.auto(1000, 4, 3, align=32).policy == "aligned"


def test_iter_row_blocks_alignment():
    lay = D.Layout.checked(37, 3, 2, align=4)                      # rows 28:37
    assert S.iter_row_blocks(lay, 8) == [(0, 8, 28, 36), (8, 9, 36, 37)]
    assert S.iter_row_blocks(lay, 7) == [(0, 4, 28, 32), (4, 8, 32, 36), (8, 9, 36, 37)]   # 7 -> rounded down to the unit 4
    assert S.iter_row_blocks(lay, 2) == [(0, 4, 28, 32), (4, 8, 32, 36), (8, 9, 36, 37)]   # below one unit -> one unit
    assert S.iter_row_blocks(lay, None) == [(0, 9, 28, 37)] == S.iter_row_blocks(lay, 9)   # the whole shard fits: one block
    assert S.iter_row_blocks(lay, 5, unit=1) == [(0, 5, 28, 33), (5, 9, 33, 37)]
    grid = D.Layout(300, 3, 1, 64)                                 # grid policy, no ROWPAIR_CHUNK_ALIGN: unit 1
    assert S.iter_row_blocks(grid, 50)[0] == (0, 50, 128, 178)
    assert S.row_block_size(1000, 128, 4, target_bytes=2 ** 20) == 2 and S.row_block_size(10, 1, 4, target_bytes=10 ** 9, cap_elems=25) == 2
    assert S.row_blocks(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert S.choose_block_rows(1000, 128, 4) == (1048, "default")
    assert S.choose_block_rows(1000, 128, 4, rows=100, align=16) == (96, "given+align")
    assert S.choose_block_rows(1000, 128, 4, budget_bytes=10 * 2 ** 20, align=4, n_max=12) == (12, "budget+n_max")
    assert S.choose_block_rows(100000, 64, 4, budget_bytes=10 ** 12) == (335, "budget+cap")
    os.environ["ROWPAIR_TEST_ROWS_MB"] = "1"
    try:
        assert S.choose_block_rows(1000, 128, 4, env="ROWPAIR_TEST_ROWS_MB") == (2, "env:ROWPAIR_TEST_ROWS_MB")
    finally:
        del os.environ["ROWPAIR_TEST_ROWS_MB"]
    assert S.choose_block_rows(bytes_per_row=4096, env=None, default_mb=1.0) == (256, "default")
    from opt_core.mem.rowpair import evidence as EV
    EV.record_schedule(produce_block_rows=8)
    assert EV.schedule()["produce_block_rows"] == 8 and EV.reset_schedule()["produce_block_rows"] == 8 and EV.schedule() == {}


# ================================================================================================================ solo (P == 1): equality, no allocation
def test_solo_comm_is_identity():
    c = D.comm()
    assert c.backend == "solo" and c.world == 1 and c.rank == 0 and not c.active and not D.is_dist() and D.world() == (1, 0)
    t = torch.arange(6.0)
    assert c.bcast_(t, 0) is t and c.allreduce_(t, "max") is t and D.allreduce_(t) is t
    assert c.allgather_obj({"a": 1}) == [{"a": 1}] and c.broadcast_obj(5) == 5 and c.split_counts(7) == [(0, 7)] and c.checksum(t, "t")
    c.p2p({}, {})
    c.p2p_start({}, {}).wait()
    with pytest.raises(RowpairRefused):
        c.p2p({1: t}, {})
    lay = D.Layout(9, 1, 0, B=4, align=4)
    z = torch.randn(1, 9, 9, 3, generator=_gen())
    assert S.shard_rows(z, lay, dim=-3) is z and S.gather_rows(z, lay, dim=-3) is z and S.gather_rows_to(z, lay, dim=-3) is z
    assert [(q, b is z) for q, b in D.ring_blocks(z, lay, dim=-3)] == [(0, True)]
    assert [(q, b is z) for q, b in RG.ring_pass(z, lay, [tuple(z.shape)])] == [(0, True)]
    assert torch.equal(RG.transpose_shard(z, lay), z.transpose(-2, -3)) and not RG.transpose_budget_exceeded(z, lay, budget_gb=0.0)
    zi = z.clone()
    assert RG.transpose_shard_inplace_(zi, lay) is zi and torch.equal(zi, z.transpose(-2, -3))
    assert BC.sync_tensordict_from_rank0({"x": z})["synced"] == 0 and BC.broadcast_tensordict({"x": z})["x"] is z
    with D.comm().local():
        assert not D.comm().active


# ================================================================================================================ threaded ranks: data movement
def _prims_worker(rank, P, z, x, align):
    N = z.shape[-2]
    lay = S.ctx(N, align=align) if align else D.layout_here(N, B=4)
    assert D.comm().backend == ("threaded") and D.is_dist() == (P > 1) and D.world() == ((P, rank) if P > 1 else (1, 0))
    assert (lay.P, lay.rank) == (P, rank)
    zl = S.shard_rows(z, lay, dim=-3).contiguous()
    ok = {}
    ok["shard_rows_view"] = torch.equal(zl, z[..., lay.r0:lay.r1, :, :])
    ok["gather_rows"] = torch.equal(S.gather_rows(zl, lay, dim=-3), z)
    ok["unshard_rows_name"] = S.unshard_rows is S.gather_rows
    g0 = S.gather_rows_to(zl, lay, dim=-3)
    ok["gather_rows_to0"] = torch.equal(g0, z) if rank == 0 else g0 is None
    ztr = z.transpose(-2, -3)[..., lay.r0:lay.r1, :, :]
    ok["transpose_shard"] = torch.equal(RG.transpose_shard(zl, lay), ztr)
    for k in (1, 2, 3):
        ok[f"transpose_streamed_peers{k}"] = torch.equal(RG.transpose_shard_streamed(zl, lay, peers_in_flight=k), ztr)
    ok["transpose_budget_dispatch"] = torch.equal(RG.transpose_shard(zl, lay, budget_gb=0.0), ztr) and (RG.transpose_budget_exceeded(zl, lay, 0.0) == (P > 1))
    out = torch.empty_like(ztr.contiguous())
    ok["transpose_out"] = RG.transpose_shard(zl, lay, out=out) is out and torch.equal(out, ztr)
    zi = zl.clone()
    ok["transpose_inplace"] = RG.transpose_shard_inplace_(zi, lay, peers_in_flight=2) is zi and torch.equal(zi, ztr)
    ok["transpose_shards_dim0"] = torch.equal(D.transpose_shards(zl[0], lay, chunks=2), ztr[0])
    ok["transpose_shards_blockrows"] = torch.equal(D.transpose_shards(zl[0], lay, block_rows=4), ztr[0])
    seen = []
    for i0, i1, blk in D.transpose_blocks(zl[0], lay, step=3):
        seen.append((i0, i1, torch.equal(blk, ztr[0][i0:i1])))
    ok["transpose_blocks"] = all(s[2] for s in seen) and sum(b - a for a, b, _ in seen) == lay.n_loc
    order, good = [], []
    for q, blk in D.ring_blocks(zl, lay, dim=-3):
        order.append(q)
        good.append(torch.equal(blk, z[..., lay.parts[q][0]:lay.parts[q][1], :, :]))
    ok["ring_blocks_order"] = order == [(rank - t) % P for t in range(P)]          # own block first, then rank-1, rank-2, ...
    ok["ring_blocks_content"] = all(good) and sorted(order) == list(range(P))
    RB, j0 = 3, 2
    sub = zl[0][j0:j0 + RB]
    got = {q: torch.equal(blk, z[0][lay.parts[q][0] + j0: min(lay.parts[q][1], lay.parts[q][0] + j0 + RB)]) for q, blk in D.ring_blocks(sub, lay, rows=(j0, RB))}
    ok["ring_blocks_subslab"] = sorted(got) == list(range(P)) and all(got.values())
    shapes = [tuple(zl.shape[:-3]) + (e - s_, N, zl.shape[-1]) for s_, e in lay.parts]
    order2, good2 = [], []
    for q, blk in RG.ring_pass(zl, lay, shapes):
        order2.append(q)
        good2.append(torch.equal(blk, z[..., lay.parts[q][0]:lay.parts[q][1], :, :]))
    ok["ring_pass"] = order2 == [(rank - t) % P for t in range(P)] and all(good2)
    absent = [None if q == P - 1 else shapes[q] for q in range(P)]               # the last rank has no block this pass
    own = None if rank == P - 1 else zl
    order3 = [q for q, _ in RG.ring_pass(own, lay, absent, dtype=zl.dtype, device=zl.device)]
    ok["ring_pass_absent"] = order3 == [q for q in [(rank - t) % P for t in range(P)] if q != P - 1]
    windows = [(1, min(3, N))] * P
    ok["transpose_band"] = torch.equal(RG.transpose_band(zl, lay, windows), z.transpose(-2, -3)[..., 1:min(3, N), :, :])
    wins2 = [((lay.parts[q][0], lay.parts[q][0] + 2) if q % 2 == 0 else (0, 0)) for q in range(P)]   # odd ranks want nothing
    band = RG.transpose_band(zl, lay, wins2)
    w0, w1 = wins2[rank]
    ok["transpose_band_ragged"] = (band is None) if w1 <= w0 else torch.equal(band, z.transpose(-2, -3)[..., w0:w1, :, :])
    ok["gather_2d_dim0"] = torch.equal(S.gather_rows(S.shard_rows(x, lay, 0).contiguous(), lay, 0), x)
    ok["gather_2d_dim1"] = torch.equal(S.gather_rows(S.shard_rows(x.t().contiguous(), lay, 1).contiguous(), lay, 1), x.t())
    cat = D.gather_cat_to_rank0(zl[0], [e - s_ for s_, e in lay.parts])
    ok["gather_cat_to_rank0"] = torch.equal(cat, z[0]) if rank == 0 else cat is None
    t = torch.tensor([float(rank + 1)])
    ok["allreduce_sum"] = D.allreduce_(t) is t and t.item() == P * (P + 1) / 2
    ok["allreduce_max"] = D.comm().allreduce_(torch.tensor([rank]), "max").item() == P - 1
    ok["comm_checksum"] = D.comm().checksum(z, "z")
    ok["allreduce_checksum"] = len(D.allreduce_checksum(z, "z")) == P
    bad = z.clone()
    if rank == P - 1 and P > 1:
        bad[0, 0, 0, 0] += 1.0
    try:
        D.allreduce_checksum(bad, "bad")
        ok["checksum_mismatch_refused"] = P == 1
    except RowpairRefused:
        ok["checksum_mismatch_refused"] = P > 1
    ok["broadcast_obj"] = D.broadcast_obj({"a": rank}) == {"a": 0}
    ok["allgather_obj"] = D.comm().allgather_obj(rank) == list(range(P))
    ok["split_counts"] = D.comm().split_counts(10) == [((r * 10) // P, ((r + 1) * 10) // P) for r in range(P)]
    y = torch.zeros(1, N, 4)
    S.bcast_small(y.copy_(torch.full_like(y, float(rank))), src=P - 1)
    ok["bcast_small"] = bool((y == float(P - 1)).all())
    return ok, repr(lay)


@pytest.mark.parametrize("P", PS)
@pytest.mark.parametrize("align", [4, 1, None])
def test_threaded_data_movement_bitwise(P, align):
    N = 37 if align else 40                                        # uneven N under the aligned policy; grid needs N >= P*B
    g = _gen(P * 10 + (align or 0))
    z = torch.randn(1, N, N, 5, generator=g)
    x = torch.randn(N, 3, generator=g)
    res = run_ranks(P, _prims_worker, z, x, align)
    assert _all_true(res) == [], (_all_true(res), res[-1][1])


def test_run_ranks_reports_rank_failure():
    def worker(rank, P):
        if rank == 1:
            raise ValueError("boom")
        D.comm().barrier()
        return rank
    with pytest.raises(RuntimeError, match="rank 1 failed"):
        run_ranks(3, worker, timeout_s=30)


def test_threaded_p2p_shape_mismatch_refused():
    def worker(rank, P):
        c = D.comm()
        peer = 1 - rank
        c.p2p({peer: torch.zeros(3)}, {peer: torch.zeros(4)})
    with pytest.raises(RuntimeError, match="expects|sends"):
        run_ranks(2, worker, timeout_s=30)


# ================================================================================================================ produce_rows_
def _produce_worker(rank, P, a, b):
    N = a.shape[0]
    lay = S.ctx(N, align=4)
    dense = torch.einsum("ic,jc->ijc", a, b).unsqueeze(0)          # [1, N, N, C]
    fn = lambda g0, g1: torch.einsum("ic,jc->ijc", a[g0:g1], b).unsqueeze(0)  # noqa: E731
    ok = {}
    out = torch.zeros(1, lay.n_loc, N, a.shape[1])
    calls = []
    S.produce_rows_(out, lay, lambda g0, g1: (calls.append((g0, g1)), fn(g0, g1))[1], block_rows=9)
    ok["set_equals_dense_rows"] = torch.equal(out, dense[:, lay.r0:lay.r1])
    ok["blocks_aligned"] = all(g0 % 4 == 0 for g0, _ in calls) and calls[0][0] == lay.r0 and calls[-1][1] == lay.r1
    ok["blocks_tile"] = all(calls[i][1] == calls[i + 1][0] for i in range(len(calls) - 1))
    S.produce_rows_(out, lay, fn, op="add", block_rows=5)
    ok["add_accumulates"] = torch.equal(out, dense[:, lay.r0:lay.r1] * 2)
    out2 = torch.zeros(lay.n_loc, N, a.shape[1])
    S.produce_rows_(out2, lay, lambda g0, g1: fn(g0, g1)[0], row_dim=0)     # default block_rows (whole shard under the default budget)
    ok["row_dim0_default_blocks"] = torch.equal(out2, dense[0, lay.r0:lay.r1])
    ok["gather_of_produced"] = torch.equal(S.gather_rows(out2, lay, dim=0), dense[0] )
    try:
        S.produce_rows_(out2, lay, lambda g0, g1: fn(g0, g1)[0][:, :3], row_dim=0)
        ok["shape_mismatch_refused"] = False
    except RowpairRefused:
        ok["shape_mismatch_refused"] = True
    return ok, repr(lay)


@pytest.mark.parametrize("P", PS)
def test_produce_rows_equals_dense(P):
    g = _gen(7)
    a, b = torch.randn(37, 6, generator=g), torch.randn(37, 6, generator=g)
    res = run_ranks(P, _produce_worker, a, b)
    assert _all_true(res) == [], _all_true(res)


# ================================================================================================================ bcast
def _bcast_worker(rank, P, ref):
    ok = {}
    mine = {"f": ref["f"].clone() if rank == 0 else torch.zeros(3 + rank, 2),          # shape differs off rank 0 -> re-allocated
            "i": ref["i"].clone() if rank == 0 else ref["i"].clone() + rank,
            "b": ref["b"].clone() if rank == 0 else ~ref["b"],
            "nested": {"lst": [ref["nested"]["lst"][0].clone() * (1 if rank == 0 else 0)], "name": "q1"},
            "host_only": torch.full((2,), float(rank))}
    cen = BC.sync_tensordict_from_rank0(mine, skip_keys=("host_only",), tag="t")
    ok["synced_values"] = all(torch.equal(mine[k], ref[k]) for k in ("f", "i", "b")) and torch.equal(mine["nested"]["lst"][0], ref["nested"]["lst"][0])
    ok["census"] = cen["synced"] == 4 and cen["checked"] == 4 and cen["skipped"] == ["/host_only"] and cen["realloc"] == (0 if rank == 0 else 1)
    ok["skipped_untouched"] = bool((mine["host_only"] == float(rank)).all())
    meta_bad = {"x": ref["f"].clone(), "name": f"r{rank}"}
    try:
        BC.sync_tensordict_from_rank0(meta_bad, tag="meta")
        ok["meta_mismatch_refused"] = P == 1
    except RowpairRefused:
        ok["meta_mismatch_refused"] = P > 1
    d = BC.broadcast_tensordict(ref if rank == 0 else None, src=0, coalesce_bytes=64)
    ok["broadcast_tensordict"] = all(torch.equal(d[k], ref[k]) for k in ("f", "i", "b")) and d["nested"]["name"] == "q1"
    ok["assert_replicated"] = set(BC.assert_replicated(d, "d")) >= {"f", "i", "b"}
    return ok, None


@pytest.mark.parametrize("P", (2, 3))
def test_sync_and_broadcast_tensordict(P):
    g = _gen(3)
    ref = {"f": torch.randn(3, 2, generator=g), "i": torch.arange(5), "b": torch.tensor([True, False, True]),
           "nested": {"lst": [torch.randn(40, generator=g)], "name": "q1"}}
    res = run_ranks(P, _bcast_worker, ref)
    assert _all_true(res) == [], _all_true(res)


# ================================================================================================================ ckpt
def _ckpt_worker(rank, P, root, z, s):
    N = z.shape[-2]
    lay = S.ctx(N, align=4)
    zl = S.shard_rows(z, lay, dim=-3).contiguous()
    ok = {}
    ck = CK.TrunkCheckpointer("q", 7, N, root=root, every=1, resume=True, keep=2, log=lambda m: None)
    ok["nothing_to_resume"] = ck.try_resume(layout=lay, device="cpu") is None
    torch.manual_seed(1234) if rank == 0 else None
    for cyc in range(3):
        ck.maybe_save_cycle(cyc, layout=lay, s=s, z_loc=zl * (cyc + 1))
        D.comm().barrier()
    ck.finalize()
    D.comm().barrier()
    tags = [t for t, _ in CK.list_complete(ck.root)]
    ok["keep_prunes"] = sorted(tags) == ["cycle_001", "cycle_002"]
    ok["latest_complete_cycle"] = CK.latest_complete_cycle(ck.root, P) == 2 and CK.latest_complete_cycle(ck.root, P + 1) is None
    tag, man = ck.find_resume(lay)
    ok["find_resume_newest_cycle"] = tag == "cycle_002" and man["format"] == CK.FORMAT and man["P"] == P and man["meta"]["parts"] == [list(p) for p in lay.parts]
    st = ck.try_resume(layout=lay, device="cpu")
    ok["resume_cycle_values"] = st["cycle"] == 2 and torch.equal(st["z_loc"], zl * 3) and torch.equal(st["s"], s) and "s_input" not in st
    sh, rep, rng = CK.load_cycle(ck.root, 2, rank, P=P)
    ok["load_cycle"] = torch.equal(sh["z_loc"], zl * 3) and torch.equal(rep["s"], s) and rng is not None and "torch_cpu" in rng
    ck.save_trunk_final(layout=lay, s_input=s + 1, s=s, z_loc=zl, cycle=2)
    D.comm().barrier()
    st2 = ck.try_resume(layout=lay, device="cpu")
    ok["trunk_final_preferred"] = st2["tag"] == "trunk_final" and torch.equal(st2["s_input"], s + 1) and torch.equal(st2["z_loc"], zl)
    other = D.Layout(N, P, rank, B=1, align=1) if P > 1 else D.Layout(N + 4, 1, 0, align=4)
    ok["incompatible_partition_skipped"] = (ck.find_resume(other)[0] is None) if other.parts != lay.parts else True
    try:
        CK.load(ck.root, "cycle_002", rank=rank, P=P + 1)
        ok["load_wrong_P_refused"] = False
    except RowpairRefused:
        ok["load_wrong_P_refused"] = True
    p = CK.save_cycle(os.path.join(root, "plain"), 5, rank, {"z_loc": zl}, {"s": s}, None)
    D.comm().barrier()
    ok["save_cycle_manifest"] = os.path.exists(p) and CK.latest_complete_cycle(os.path.join(root, "plain"), P) == 5
    return ok, None


@pytest.mark.parametrize("P", (1, 3))
def test_trunk_checkpointer_roundtrip(P):
    g = _gen(5)
    z = torch.randn(1, 37, 37, 3, generator=g)
    s = torch.randn(37, 8, generator=g)
    with tempfile.TemporaryDirectory() as root:
        res = run_ranks(P, _ckpt_worker, root, z, s)
    assert _all_true(res) == [], _all_true(res)


def _ckpt_identity_worker(rank, P, root, z, s):
    """K1-K3: per-rank feature digest + precision tag in every tag, refusals by name on resume, the adopter-facing log / census words."""
    import json as _json
    import re as _re
    from opt_core.mem.rowpair import evidence as EV
    N = z.shape[-2]
    lay = S.ctx(N, align=4)
    zl = S.shard_rows(z, lay, dim=-3).contiguous()
    from opt_core.mem.rowpair import msa_host as MH
    msa = torch.randn(40, N, 6)
    feats = {"tok": torch.arange(N), "msa_rows_on_this_rank": torch.full((3,), float(rank)), "name": "q",
             "msa": msa if rank == 0 else MH.host_placeholder(msa, 0)}                                      # per-rank features differ by design (mode rank0)
    lines = []
    mk = lambda **kw: CK.TrunkCheckpointer("q", 7, N, root=root, every=1, resume=True, keep=2, log=lines.append, **kw)   # noqa: E731
    ok = {}
    ck = mk(features=feats, precision="fp32", dtype=torch.float32)
    ok["digest_is_feature_hash"] = ck.features == CK.feature_hash(feats) and len(ck.features) == 64
    ok["nothing_to_resume"] = ck.try_resume(layout=lay, device="cpu") is None and EV.schedule().get("ckpt_resume") == "none"
    ck.maybe_save_cycle(0, layout=lay, s=s, z_loc=zl)
    D.comm().barrier()
    ck.save_trunk_final(layout=lay, s_input=s + 1, s=s, z_loc=zl * 2, cycle=0)
    ck.finalize()
    D.comm().barrier()
    ok["census_wrote"] = EV.schedule().get("ckpt_wrote") == "trunk_final" and ck.last_written == "trunk_final"
    if rank == 0:
        ok["wrote_line_vocabulary"] = any(_re.match(r"^\[ckpt\] wrote trunk_final ranks %d bytes \d+ in [0-9.]+s" % P, m) for m in lines)
        man = _json.load(open(os.path.join(ck.root, "trunk_final", "manifest.json")))
        prec = man["meta"]["precision"]
        ok["manifest_precision_tag"] = prec == {"word": "fp32", "z_dtype": "float32", "s_dtype": "float32", "s_input_dtype": "float32"}
        ok["manifest_per_rank_digest"] = all(man["files"][f"rank{r}.pt"]["features"] is not None and len(man["files"][f"rank{r}.pt"]["features"]) == 64
                                             for r in range(P)) and man["files"]["rank0.pt"]["features"] == ck.features and man["format"] == "rowpair.ckpt/v2"
        ok["per_rank_digests_differ"] = P == 1 or len({man["files"][f"rank{r}.pt"]["features"] for r in range(P)}) == P
        cyc = _json.load(open(os.path.join(ck.root, "cycle_000", "manifest.json")))
        ok["cycle_precision_tag"] = cyc["meta"]["precision"] == {"word": "fp32", "z_dtype": "float32", "s_dtype": "float32"}
    # same equality: resumes, says 'digest ok'
    lines.clear()
    st = mk(features=feats, precision="fp32", dtype=torch.float32).try_resume(layout=lay, device="cpu")
    ok["resume_ok"] = st is not None and st["tag"] == "trunk_final" and torch.equal(st["z_loc"], zl * 2)
    # the resumed CPU shard is torch-OWNED: the trunk's release / re-grow / park statements accept it and the next statements are bit-identical
    from opt_core.mem.rowpair.shard import owns_whole_storage, regrow_storage_, release_storage_, storage_resizable
    from opt_core.mem.rowpair.trunk import ShardPark, recycle_shard_
    zr = st["z_loc"]
    ok["resumed_shard_owned"] = storage_resizable(zr) and owns_whole_storage(zr) and storage_resizable(st["s"]) and storage_resizable(st["s_input"])
    park = ShardPark(zr, park=True, force_host=True, name="z_init resumed")     # a park accepts the resumed shard
    ok["resumed_shard_parks"] = park.park and torch.equal(park.block(0, int(zr.shape[-3])), zl * 2)
    if P > 1:                                                                # the recycling statement is a sharded statement (refused by name at P == 1)
        upd = lambda blk: blk * 0.5 + 1.0                                   # noqa: E731 — a recycling update statement
        cont_resumed = recycle_shard_(zr.clone(), park, upd, lay)
        cont_fresh = recycle_shard_((zl * 2).clone(), ShardPark((zl * 2).clone(), park=False), upd, lay)
        ok["trunk_continues_bitwise"] = torch.equal(cont_resumed, cont_fresh)
        from opt_core.mem.rowpair import pairstack as PS
        Wp = torch.full((zr.shape[-1], zr.shape[-1]), 0.03125)

        fns = PS.PairBlockFns(trimul_out=lambda z, m, lay_: z.add_(torch.tanh(z @ Wp)), trimul_in=lambda z, m, lay_: z.mul_(0.5),
                              triatt_start=lambda z, m, lay_: z.add_(0.25), triatt_end=lambda zT, mT, lay_: zT.add_(torch.sigmoid(zT @ Wp)),
                              transition=lambda z, m, lay_: z.sub_(0.125))                  # in place on the shard, as the trunk's updates are
        st2 = mk(features=feats, precision="fp32").try_resume(layout=lay, device="cpu")          # a fresh resume: the shard as the trunk receives it
        zs = st2["z_loc"][0] if st2["z_loc"].dim() == 4 else st2["z_loc"]                    # the block takes [R, N, C]
        ok["resumed_block_operand_owned"] = storage_resizable(zs)
        blk_resumed, _ = PS.pair_block_(fns, zs, None, lay, end_free=True, transpose_inplace=False)
        D.comm().barrier()                                                  # every rank recorded its word before any rank reads the census
        freed_word = EV.schedule().get("pairstack_end_free")
        D.comm().barrier()
        zf = (zl * 2).clone().contiguous()
        blk_fresh, _ = PS.pair_block_(fns, zf[0] if zf.dim() == 4 else zf, None, lay, end_free=True, transpose_inplace=False)
        ok["pair_blocks_continue_bitwise_end_free_applied"] = torch.equal(blk_resumed, blk_fresh) and freed_word == "applied"
    nb = release_storage_(zr)
    ok["resumed_shard_release_regrow"] = nb == zr.numel() * zr.element_size() and zr.untyped_storage().nbytes() == 0 and regrow_storage_(zr, nb) is not None \
        and zr.untyped_storage().nbytes() == nb
    ok["resume_line_vocabulary"] = any(m.startswith("[ckpt] resume from trunk_final (cycle 0) digest ok") for m in lines)
    ok["census_resume"] = EV.schedule().get("ckpt_resume") == "trunk_final"
    # a precomputed digest string is taken as is
    ok["digest_string_accepted"] = mk(features=CK.feature_hash(feats), precision="fp32").try_resume(layout=lay, device="cpu") is not None
    # other inputs: refused BY NAME before any byte loads; allowed + named under the override
    other = dict(feats, name="another query")
    try:
        mk(features=other, precision="fp32").try_resume(layout=lay, device="cpu")
        ok["features_differ_refused"] = False
    except RowpairRefused as e:
        ok["features_differ_refused"] = str(e.reason).startswith(CK.REFUSED_FEATURES) and "ROWPAIR_RESUME_ALLOW_FEATS=1" in str(e.reason)
    ok["census_refused_word"] = str(EV.schedule().get("ckpt_resume", "")).startswith(CK.REFUSED_FEATURES + ":trunk_final")
    lines.clear()
    st = mk(features=other, precision="fp32", allow_features=True).try_resume(layout=lay, device="cpu")
    ok["features_differ_allowed_named"] = st is not None and any("features DIFFER" in m and "ALLOWED" in m for m in lines) \
        and any("digest DIFFERS (allowed" in m for m in lines)
    try:
        mk(features=None, precision="fp32").try_resume(layout=lay, device="cpu")
        ok["unrecorded_side_refused"] = False
    except RowpairRefused as e:
        ok["unrecorded_side_refused"] = str(e.reason).startswith(CK.REFUSED_FEATURES) and "unrecorded" in str(e.reason)
    # another precision word / another shard dtype: refused by name (no override)
    try:
        mk(features=feats, precision="bf16").try_resume(layout=lay, device="cpu")
        ok["precision_refused"] = False
    except RowpairRefused as e:
        ok["precision_refused"] = str(e.reason).startswith(CK.REFUSED_PRECISION) and "'fp32'" in str(e.reason) and "'bf16'" in str(e.reason)
    try:
        mk(features=feats, precision="fp32", dtype=torch.bfloat16).try_resume(layout=lay, device="cpu")
        ok["dtype_refused"] = False
    except RowpairRefused as e:
        ok["dtype_refused"] = str(e.reason).startswith(CK.REFUSED_DTYPE)
    # an older-format tag is passed over, SAID in the log (never a silent fresh start)
    D.comm().barrier()
    if rank == 0:
        d = os.path.join(ck.root, "cycle_007")
        os.makedirs(d, exist_ok=True)
        _json.dump({"tag": "cycle_007", "P": P, "meta": {"N": N, "parts": [list(p) for p in lay.parts]}, "files": {}, "format": "rowpair.ckpt/v1"},
                   open(os.path.join(d, "manifest.json"), "w"))
    D.comm().barrier()
    lines.clear()
    tag, _man = mk(features=feats, precision="fp32").find_resume(lay)
    ok["old_format_passed_over_said"] = tag == "trunk_final" and any("cycle_007 passed over: format rowpair.ckpt/v1" in m for m in lines)
    return ok, None


@pytest.mark.parametrize("P", (1, 2, 3))
def test_trunk_checkpointer_identity_words(P):
    g = _gen(6)
    z = torch.randn(1, 37, 37, 3, generator=g)
    s = torch.randn(37, 8, generator=g)
    with tempfile.TemporaryDirectory() as root:
        res = run_ranks(P, _ckpt_identity_worker, root, z, s)
    assert _all_true(res) == [], _all_true(res)


class _Untouchable(object):
    """A leaf whose hashing would raise: proves a checkpointer never looks at the features object."""
    def __repr__(self):
        raise AssertionError("the features object was touched")


def test_disabled_checkpointer_never_touches_the_features(monkeypatch):
    monkeypatch.delenv("ROWPAIR_CKPT_DIR", raising=False)
    lay = S.Layout(16, 1, 0, B=1)
    feats = {"x": torch.randn(3), "trap": _Untouchable()}
    ck = CK.TrunkCheckpointer("q", 1, 16, features=feats, precision="fp32", log=lambda m: None)      # ROWPAIR_CKPT_DIR unset: disabled
    assert not ck.enabled and ck._features_obj is None                    # not even kept
    assert ck.try_resume(layout=lay, device="cpu") is None and ck.maybe_save_cycle(0, layout=lay, s=torch.zeros(2), z_loc=torch.zeros(1, 16, 16, 1)) is None
    ck.save_trunk_final(layout=lay, s_input=torch.zeros(2), s=torch.zeros(2), z_loc=torch.zeros(1, 16, 16, 1), cycle=0)
    ck.finalize()
    assert ck.features is None and ck.digest_features() is None
    with tempfile.TemporaryDirectory() as root:                            # enabled: kept unhashed at construction, hashed ONCE at trunk entry
        ok_feats = {"x": torch.randn(3), "n": 5}
        ck = CK.TrunkCheckpointer("q", 1, 16, root=root, features=ok_feats, precision="fp32", log=lambda m: None)
        assert ck.enabled and ck._features_obj is ok_feats and ck._features_digest is None
        assert ck.try_resume(layout=lay, device="cpu") is None
        assert ck._features_obj is None and ck._features_digest == CK.feature_hash(ok_feats) and ck.features == ck._features_digest
        trap = CK.TrunkCheckpointer("q", 2, 16, root=root, features={"trap": _Untouchable()}, log=lambda m: None)
        with pytest.raises(AssertionError, match="touched"):               # enabled + an object: hashed at the first try_resume (and only then)
            trap.try_resume(layout=lay, device="cpu")
        given = CK.TrunkCheckpointer("q", 3, 16, root=root, features="AB" * 32, log=lambda m: None)
        assert given.features == "ab" * 32                                 # a 64-hex digest is taken as is (lower-cased), nothing hashed


def test_feature_hash_is_chunked_and_zero_copy(monkeypatch):
    import hashlib
    g = _gen(11)
    from opt_core.mem.rowpair import msa_host as MH
    feats = {"f": torch.randn(300, 7, generator=g), "b": torch.rand(50, generator=g) > 0.5, "h": torch.randn(33, generator=g).to(torch.bfloat16),
             "i": torch.arange(40, dtype=torch.int32).reshape(5, 8).t(),      # non-contiguous
             "empty3d": torch.empty(0, 9, 32), "scalar": torch.tensor(2.5), "placeholder": MH.host_placeholder(torch.randn(40, 9, 6, generator=g), 0),
             "nest": {"l": [torch.ones(2, dtype=torch.float64), "name", 3]}}
    assert tuple(feats["placeholder"].shape) == (0, 9, 6)                   # mode rank0's zero-row stand-in on ranks > 0 hashes (header only)
    ref = hashlib.sha256()

    def rec(o, path):                                                       # the reference statement with whole-tensor tobytes()
        if isinstance(o, torch.Tensor):
            tt = o.detach().to("cpu").contiguous()
            ref.update(f"{path}|{tt.dtype}|{tuple(tt.shape)}|".encode())
            if tt.dtype == torch.bool:
                tt = tt.to(torch.uint8)
            if tt.dtype == torch.bfloat16:
                tt = tt.view(torch.int16)
            ref.update(tt.numpy().tobytes())
        elif isinstance(o, dict):
            for k in sorted(o.keys(), key=str):
                rec(o[k], f"{path}/{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                rec(v, f"{path}/{i}")
        else:
            ref.update(f"{path}|{o!r}".encode())
    rec(feats, "")
    full = CK.feature_hash(feats)
    assert full == ref.hexdigest()
    monkeypatch.setattr(CK, "HASH_CHUNK_BYTES", 7)                          # tiny slices: the chunked walk hashes the same bytes
    assert CK.feature_hash(feats) == full
    assert CK.feature_hash({"f": feats["f"].clone()}) != full and CK.feature_hash(feats) == CK.feature_hash(dict(feats))


# ================================================================================================================ confidence reducer on the threaded comm
def _reducer_worker(rank, P, pae, pde, contact, asym, has_frame):
    from opt_core.mem.rowpair.confidence import ChainIndex, RowBlockReducer
    N = pae.shape[0]
    lay = S.ctx(N, align=4)
    red = RowBlockReducer(ChainIndex(asym, has_frame), lay.r0, lay.r1, "cpu", finish="exact", allow_unsharded=(P == 1))
    for b0, b1, g0, g1 in S.iter_row_blocks(lay, 8):
        red.consume(g0, g1, pae[g0:g1], pde[g0:g1], contact[g0:g1])
    out = red.finalize(lay.bounds)
    if rank != 0:
        return out is None
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in out.items()}


def test_blockreduce_exact_finish_equals_dense_on_threaded_comm():
    g = _gen(11)
    N = 37
    pae, pde = torch.randn(N, N, 64, generator=g), torch.randn(N, N, 64, generator=g)
    contact = torch.rand(N, N, generator=g)
    asym = torch.tensor([0] * 15 + [1] * 13 + [2] * 9)
    has_frame = torch.rand(N, generator=g) > 0.3
    dense = run_ranks(1, _reducer_worker, pae, pde, contact, asym, has_frame)[0]
    report = {}
    for P in (2, 3, 4):
        res = run_ranks(P, _reducer_worker, pae, pde, contact, asym, has_frame)
        assert all(r is True for r in res[1:])
        got = res[0]
        assert set(got) == set(dense)
        for k in dense:
            if not torch.is_tensor(dense[k]):
                assert got[k] == dense[k], k
                continue
            eq = bool(torch.equal(got[k], dense[k]))
            md = float((got[k].double() - dense[k].double()).abs().nan_to_num(0.0).max()) if got[k].numel() else 0.0
            report[(P, k)] = (eq, md)
            assert md <= 1e-5, (P, k, md)
    bit = all(eq for eq, _ in report.values())
    print(f"RESULT blockreduce_exact_threaded bitwise_all={bit} max_abs={max(md for _, md in report.values()):.3e}")


# ================================================================================================================ host signals
def _signal_worker(rank, P):
    import time as _t
    ok = {}
    if rank == 0:
        _t.sleep(0.2)
        D.host_signal("heads_done", "cycle3")
        ok["signal"] = True
    else:
        t0 = _t.monotonic()
        ok["waited_value"] = D.host_wait("heads_done", timeout_s=30) == "cycle3"
        ok["waited_at_all"] = _t.monotonic() - t0 >= 0.1
    try:
        D.host_wait("never", timeout_s=0.2)
        ok["timeout_refused"] = False
    except RowpairRefused:
        ok["timeout_refused"] = True
    D.comm().barrier()
    return ok, None


def test_host_signal_threaded():
    res = run_ranks(2, _signal_worker)
    assert _all_true(res) == [], _all_true(res)
    c = D.comm()                                                    # solo: no-op signal, immediate wait
    assert c.host_wait("absent", timeout_s=5) == ""
    D.host_signal("k", "v")
    assert D.host_wait("k") == "v"


def _mp_signal_entry():
    """Runs in each of 2 gloo processes (launch.run_sharded): the c10d store of the file:// rendezvous carries the signal."""
    import time as _t
    from opt_core.mem.rowpair import dist as DD
    P, r = DD.world()
    assert DD.comm().backend == "gloo" and P == 2
    out = {"rank": r}
    if r == 1:
        _t.sleep(0.5)
        DD.host_signal("templ_done", f"from{r}")
    else:
        t0 = _t.monotonic()
        out["value"] = DD.host_wait("templ_done", timeout_s=60)
        out["waited_s"] = _t.monotonic() - t0
    try:
        DD.host_wait("never_mp", timeout_s=0.3)
        out["timeout_refused"] = False
    except RowpairRefused:
        out["timeout_refused"] = True
    DD.barrier()
    return out


def test_mp_host_signal_gloo_store():
    from opt_core.mem.rowpair import launch
    res = launch.run_sharded(2, _mp_signal_entry, cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert res["rank"] == 0 and res["value"] == "from1" and res["waited_s"] >= 0.3 and res["timeout_refused"] is True, res
