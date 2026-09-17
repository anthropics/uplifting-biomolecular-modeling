"""Scoped wire buffers (``dist.Staging``): every point-to-point / all-to-all schedule — ring_blocks, ring_pass, transpose_shard (all / streamed /
in place), transpose_band, alltoall_window — stages its sends (including a copy of its OWN block: no caller tensor goes on the wire) and receives
in buffers that live for ONE call / ONE ring pass, are reused across the steps inside it (no per-step allocation), and are freed at the end of
the scope (after a current-stream synchronize on nccl). Values compared bit-exact with the dense statement; rank-threads of one process
(``opt_core.testing.run_ranks``)."""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import dist as D, ring as RG                     # noqa: E402
from opt_core.mem.rowpair.dist import Layout                              # noqa: E402
from opt_core.mem.rowpair.evidence import reset_schedule, schedule_fields  # noqa: E402
from opt_core.testing import run_ranks                                    # noqa: E402


def _z(N, C=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, N, C, generator=g)


# ----------------------------------------------------------------------------------------------------------------- ring_blocks / ring_pass
def _ring_blocks_rank(rank, P, N, RB, passes):
    lay = Layout(N, P, rank, align=1)
    x = torch.arange(N * 5, dtype=torch.float32).reshape(N, 5)
    shard = x[lay.r0:lay.r1]
    st = D.comm().staging
    ok, own_staged, per_pass_allocs, per_pass_ptrs, live_after = True, True, [], [], []
    for p in range(passes):
        slab = shard[:RB] if shard.shape[0] >= RB else shard                # rows=(0, RB): min(R, RB) own entries (ragged ranks travel padded)
        a0, ptrs = st.allocs, set()
        for q, blk in D.ring_blocks(slab, lay, rows=(0, RB), dim=0):
            want = x[lay.bounds[q][0]:lay.bounds[q][0] + min(lay.nrows(q), RB)]
            ok = ok and torch.equal(blk, want)
            ptrs.add(blk.untyped_storage().data_ptr())
            if q == rank:
                own_staged = own_staged and blk.untyped_storage().data_ptr() != slab.untyped_storage().data_ptr()
        per_pass_allocs.append(st.allocs - a0)
        per_pass_ptrs.append(len(ptrs))
        live_after.append(st.stats()["bufs"])
    facts = dict(schedule_fields())
    return {"ok": ok, "own_staged": own_staged, "per_pass_allocs": per_pass_allocs, "per_pass_ptrs": per_pass_ptrs, "live_after": live_after,
            "peak": st.peak_bytes, "facts": sorted(k for k in facts if k.startswith("staging"))}


@pytest.mark.parametrize("P,N,RB", [(2, 37, 16), (3, 40, 8), (3, 41, 16), (4, 64, 16)])
def test_ring_blocks_stages_own_block_and_allocates_once_per_pass(P, N, RB):
    reset_schedule()
    res = run_ranks(P, _ring_blocks_rank, N, RB, 3)
    for r in res:
        assert r["ok"] and r["own_staged"], r
        assert all(a == min(P, 2) for a in r["per_pass_allocs"]), r         # own copy + recv A (+ nothing per step; B is the own copy)
        assert all(n <= 2 for n in r["per_pass_ptrs"]), r                    # every yielded block lives in one of two buffers
        assert all(n == 0 for n in r["live_after"]), r                       # freed at the end of each pass
        assert r["peak"] > 0 and r["facts"] == ["staging_allocs", "staging_peak_mb", "staging_syncs"], r


def _ring_pass_rank(rank, P, passes):
    lay = Layout(8 * P, P, rank, align=1)
    shapes = [None if q == 1 else [3 + q, 4] for q in range(P)]            # per-source shapes, one absent source
    own = None if shapes[rank] is None else torch.full(shapes[rank], float(rank))
    st = D.comm().staging
    seen, allocs, ptrs_n, live_after, own_staged = [], [], [], [], True
    for p in range(passes):
        got, ptrs, a0 = [], set(), st.allocs
        for src, blk in RG.ring_pass(own, lay, shapes, dtype=torch.float32, device="cpu"):
            got.append((src, tuple(blk.shape), float(blk.flatten()[0])))
            ptrs.add(blk.untyped_storage().data_ptr())
            if src == rank:
                own_staged = own_staged and blk.untyped_storage().data_ptr() != own.untyped_storage().data_ptr()
        seen.append(got); allocs.append(st.allocs - a0); ptrs_n.append(len(ptrs)); live_after.append(st.stats()["bufs"])
    return {"seen": seen, "allocs": allocs, "ptrs_n": ptrs_n, "live_after": live_after, "own_staged": own_staged}


@pytest.mark.parametrize("P", [2, 3, 4])
def test_ring_pass_is_double_buffered_and_scoped(P):
    res = run_ranks(P, _ring_pass_rank, 3)
    for rank, r in enumerate(res):
        want = [(q, (3 + q, 4), float(q)) for q in [(rank - s) % P for s in range(P)] if q != 1]
        assert r["seen"][0] == r["seen"][1] == r["seen"][2] == want, (rank, r["seen"][0], want)
        assert all(a <= 3 for a in r["allocs"]) and all(n <= 3 for n in r["ptrs_n"]), r   # own copy + A + B, whatever P
        assert all(n == 0 for n in r["live_after"]) and r["own_staged"], r


# ----------------------------------------------------------------------------------------------------------------- transposes / band / all-to-all
def _transpose_rank(rank, P, N, form):
    lay = Layout(N, P, rank, align=1)
    z = _z(N)
    zl = z[lay.r0:lay.r1].contiguous()
    st = D.comm().staging
    outs, allocs, live_after = [], [], []
    for rep in range(2):
        a0 = st.allocs
        if form == "all":
            o = RG.transpose_shard(zl, lay)
        elif form == "streamed":
            o = RG.transpose_shard_streamed(zl, lay, peers_in_flight=1)
        elif form == "inplace":
            o = RG.transpose_shard_inplace_(zl.clone(), lay)
        elif form == "band":
            windows = [(lay.bounds[q][0], min(lay.bounds[q][1], lay.bounds[q][0] + 5)) for q in range(P)]
            o = RG.transpose_band(zl, lay, windows)
        else:
            o = D.transpose_shards(zl, lay, chunks=2)
        outs.append(o.clone()); allocs.append(st.allocs - a0); live_after.append(st.stats()["bufs"])
    zT = z.transpose(0, 1)
    want = zT[lay.r0:min(lay.r1, lay.r0 + 5)] if form == "band" else zT[lay.r0:lay.r1]
    return {"ok": all(torch.equal(o, want) for o in outs), "allocs": allocs, "live_after": live_after}


@pytest.mark.parametrize("form", ["all", "streamed", "inplace", "band", "a2a"])
@pytest.mark.parametrize("P,N", [(2, 22), (3, 40), (4, 41)])
def test_transposes_stage_per_call_and_free_at_exit(P, N, form):
    res = run_ranks(P, _transpose_rank, N, form)
    for r in res:
        assert r["ok"], r
        assert all(n == 0 for n in r["live_after"]), r                       # nothing outlives the call
        assert r["allocs"][0] == r["allocs"][1], r                            # the same wire set every call
    if form == "all":
        assert all(r["allocs"][0] == 2 * (P - 1) for r in res), res          # (P-1) send + (P-1) recv blocks
    if form in ("streamed", "inplace"):
        assert all(r["allocs"][0] <= 2 for r in res), res                    # peers_in_flight = 1: one pair reused every step, whatever P


def test_solo_paths_stage_nothing():
    """P == 1: every schedule is the local statement; no staging scope is opened and no staging census word is recorded."""
    reset_schedule()
    lay = Layout(12, 1, 0)
    z = _z(12)
    assert torch.equal(RG.transpose_shard(z, lay), z.transpose(0, 1))
    assert [q for q, _ in D.ring_blocks(z, lay)] == [0]
    assert not any(k.startswith("staging") for k, _ in schedule_fields())
    assert D.release_staging() == 0
