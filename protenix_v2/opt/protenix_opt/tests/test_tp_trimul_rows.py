"""The sharded TriMul's tiling is ONE schedule on every rank (``ptx_tp.trimul.agreed_rows``): RA (resident A rows per pass) and RB
(streamed B slab rows) fix npass = ceil(Rmax / RA) and nslab = ceil(Rmax / RB) — how many passes, b slabs, ring rounds and all-to-all
windows a ``tp_trimul`` call issues, i.e. its collective sequence. A value the caller leaves unset is sized from a share of the rank's
own free device memory (``contract.default_rows_a`` / ``default_rows_b``), which differs between ranks (ragged last shard, rank 0's
featurization residue, allocator history), so the sized values are AGREED as the minimum over ranks before they are used; a value the
caller passes is used as given and costs no collective; at P == 1 the local values are the answer.

The rank-consistency is pinned on the CPU two ways: a simulated world (the min all-reduce played over P local results; no process
group) and, where this host can stand one up, a real ``gloo`` group of 3 ranks calling the carried ``dist.allreduce_``."""
import sys

import pytest

from protenix_opt import tp
from protenix_opt.tests.conftest import needs_gloo

UNIT = tp.unit_dir()
if UNIT not in sys.path:
    sys.path.insert(0, UNIT)

torch = pytest.importorskip("torch")
trimul = pytest.importorskip("ptx_tp.trimul")
D = pytest.importorskip("ptx_tp.dist")

N, C, ELT, P = 7084, 128, 2, 4                       # the exp-scale case: 56 blocks of 128 over 4 ranks -> R = 1792, 1792, 1792, 1708 (ragged last rank)
FREE_ROWS_B = {0: 1664, 1: 1792, 2: 1792, 3: 1792}   # what a free-memory share sizes per rank when rank 0 holds more than the others
FREE_ROWS_A = {0: 1792, 1: 1792, 2: 1792, 3: 1536}


def _simulated_world(monkeypatch, rows_a, rows_b, world=P):
    """Patch ptx_tp.trimul's sizing to per-rank tables and its all-reduce to the min over the world's local tensors; returns
    (run(rank, **kw) -> (RA, RB), calls) where calls records every all-reduce as (rank, op, list)."""
    calls = []
    state = {"rank": 0}
    monkeypatch.setattr(trimul, "is_dist", lambda: world > 1)
    monkeypatch.setattr(trimul, "default_rows_a", lambda n, c, e, rmax: min(rmax, rows_a[state["rank"]]))
    monkeypatch.setattr(trimul, "default_rows_b", lambda n, c, e, rmax: min(rmax, rows_b[state["rank"]]))

    def locals_of(q, RA, RB):
        lay = D.Layout(N, world, q)
        return [int(RA) if RA else min(lay.Rmax, rows_a[q]), int(RB) if RB else min(lay.Rmax, rows_b[q])]

    def run(rank, RA=None, RB=None):
        state["rank"] = rank

        def fake_allreduce_(t, op="sum"):
            calls.append((rank, op, t.tolist()))
            assert op == "min" and t.dtype == torch.int64 and t.numel() == 2
            allv = torch.tensor([locals_of(q, RA, RB) for q in range(world)], dtype=torch.int64)
            t.copy_(allv.min(dim=0).values)
            return t
        monkeypatch.setattr(trimul, "allreduce_", fake_allreduce_)
        return trimul.agreed_rows(N, C, ELT, D.Layout(N, world, rank), RA, RB, device=torch.device("cpu"))
    return run, calls


def _schedule(rank, ra, rb, world=P):
    lay = D.Layout(N, world, rank)
    return (-(-lay.Rmax // ra), -(-lay.Rmax // rb))    # (npass, nslab) as tp_trimul derives them


def test_memory_sized_rows_are_agreed_as_the_minimum_over_ranks(monkeypatch):
    run, calls = _simulated_world(monkeypatch, FREE_ROWS_A, FREE_ROWS_B)
    got = {q: run(q) for q in range(P)}
    assert len(set(got.values())) == 1, f"ranks disagree on (RA, RB): {got}"
    assert got[0] == (min(FREE_ROWS_A.values()), min(FREE_ROWS_B.values())) == (1536, 1664)
    assert len({_schedule(q, *got[q]) for q in range(P)}) == 1, "one (npass, nslab) on every rank"
    assert [c[:2] for c in calls] == [(q, "min") for q in range(P)], "exactly one min all-reduce per rank and call"
    assert calls[0][2] == [FREE_ROWS_A[0], FREE_ROWS_B[0]] and calls[3][2] == [FREE_ROWS_A[3], FREE_ROWS_B[3]], "each rank reduces its OWN sizing"


def test_unagreed_sizing_splits_the_schedule():
    """Unagreed per-rank sizing splits the schedule at this shape: rank 0 two b slabs, rank 3 two passes, the others one of each."""
    per_rank = {q: _schedule(q, min(D.Layout(N, P, q).Rmax, FREE_ROWS_A[q]), min(D.Layout(N, P, q).Rmax, FREE_ROWS_B[q])) for q in range(P)}
    assert per_rank[0] == (1, 2) and per_rank[1] == (1, 1) and per_rank[3] == (2, 1), per_rank


def test_given_rows_pass_through_without_a_collective(monkeypatch):
    run, calls = _simulated_world(monkeypatch, FREE_ROWS_A, FREE_ROWS_B)
    assert {q: run(q, RA=1024, RB=256) for q in range(P)} == {q: (1024, 256) for q in range(P)}
    assert calls == [], "both values given: nothing memory-derived, no all-reduce"
    got = {q: run(q, RB=256) for q in range(P)}                       # RA memory-sized -> agreed; RB given -> as given
    assert set(got.values()) == {(min(FREE_ROWS_A.values()), 256)}, got
    assert len(calls) == P


def test_single_process_uses_its_own_sizing(monkeypatch):
    run, calls = _simulated_world(monkeypatch, FREE_ROWS_A, FREE_ROWS_B, world=1)
    assert run(0) == (FREE_ROWS_A[0], FREE_ROWS_B[0]) and calls == []


def test_tp_trimul_takes_its_tiling_from_agreed_rows(monkeypatch):
    """tp_trimul on a sharded layout asks agreed_rows for (RA, RB) — with the call's N, the module's hidden width, the shard's element
    size, both values unset and the shard's device — before it sizes anything (a sentinel raised from agreed_rows ends the call there)."""
    from types import SimpleNamespace
    seen = []

    class Sentinel(Exception):
        pass

    def recorder(N_, C_, elt, layout, RA=None, RB=None, device=None):
        seen.append((N_, C_, elt, layout.P, RA, RB, str(device)))
        raise Sentinel()
    monkeypatch.setattr(trimul, "is_dist", lambda: True)
    monkeypatch.setattr(trimul, "agreed_rows", recorder)
    mod = SimpleNamespace(_outgoing=True, c_hidden=32, c_z=16)
    with pytest.raises(Sentinel):
        trimul.tp_trimul(mod, torch.zeros((128, 256, 16)), D.Layout(256, 2, 0), outgoing=True)
    assert seen == [(256, 32, 4, 2, None, None, "cpu")], seen


def test_given_rows_below_one_are_refused():
    with pytest.raises(ValueError):
        trimul.agreed_rows(N, C, ELT, D.Layout(N, P, 0), RA=-128, RB=256, device=torch.device("cpu"))


def test_the_carried_allreduce_is_the_identity_without_a_group():
    t = torch.tensor([3, 5], dtype=torch.int64)
    assert D.allreduce_(t, "min") is t and t.tolist() == [3, 5]
    with pytest.raises(ValueError):
        D.allreduce_(t, "prod")


# ---------------------------------------------------------------------------------------------- a real group: 3 gloo ranks on the CPU
def _rank_entry(unit_dir):
    import sys as _sys
    if unit_dir not in _sys.path:
        _sys.path.insert(0, unit_dir)
    import torch as _torch
    import torch.distributed as _dist
    from ptx_tp import dist as _D, trimul as _T
    world, rank = _dist.get_world_size(), _dist.get_rank()
    rows_b = 128 * (world - rank) + 1024                              # 1408 / 1280 / 1152: the "free memory" differs by rank
    _T.default_rows_b = lambda n, c, e, rmax: min(rmax, rows_b)       # noqa: E731 -- this rank's own sizing
    _T.default_rows_a = lambda n, c, e, rmax: rmax                    # noqa: E731
    lay = _D.Layout(N, world, rank)
    got = [tuple(_T.agreed_rows(N, C, ELT, lay, device=_torch.device("cpu"))) for _ in range(3)]   # repeated calls: the same answer, the group stays in step
    expect = (lay.Rmax, 128 * 1 + 1024)
    assert all(g == expect for g in got), (rank, got, expect)
    given = tuple(_T.agreed_rows(N, C, ELT, lay, RA=512, RB=384, device=_torch.device("cpu")))
    assert given == (512, 384), given
    return {"rank": rank, "world": world, "agreed": list(got[0]), "given": list(given)}


try:
    import opt_core.mem.rowpair.launch as _core_launch  # noqa: F401
    HAVE_CORE = True
except Exception:  # noqa: BLE001
    HAVE_CORE = False


@needs_gloo
@pytest.mark.skipif(not HAVE_CORE, reason="opt_core.mem.rowpair.launch required to start the CPU ranks")
def test_three_gloo_ranks_agree_on_one_schedule(monkeypatch):
    from opt_core.mem.rowpair import launch
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    res = launch.run_sharded(3, _rank_entry, UNIT, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=600)
    assert res["world"] == 3 and res["agreed"] == [D.Layout(N, 3, 0).Rmax, 1152] and res["given"] == [512, 384], res
