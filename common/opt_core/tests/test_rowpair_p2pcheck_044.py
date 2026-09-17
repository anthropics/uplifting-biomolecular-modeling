"""``ROWPAIR_P2P_CHECK=1``: every point-to-point site publishes its (peer, numel, dtype) send / receive sets before posting and REFUSES BY
NAME an unmatched transfer (the device-side-hang class) and a rank-dependent schedule argument (``transpose_band`` windows, ``ring_pass``
shapes); matched exchanges pass and the schedule census says ``p2p_check=on``; with the variable unset nothing is collected."""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused          # noqa: E402
from opt_core.mem.rowpair import dist as D, ring         # noqa: E402
from opt_core.mem.rowpair.dist import Layout, comm       # noqa: E402
from opt_core.testing import run_ranks                   # noqa: E402

N, C = 24, 4


@pytest.fixture()
def check_on(monkeypatch):
    monkeypatch.setenv("ROWPAIR_P2P_CHECK", "1")
    yield


def _matched(rank, P):
    from opt_core.mem.rowpair import evidence as EV
    lay = Layout(N, P, rank)
    z = torch.arange(N * N * C, dtype=torch.float32).reshape(N, N, C)
    out = ring.transpose_shard(z[lay.r0:lay.r1].contiguous(), lay)
    ok = bool(torch.equal(out, z.transpose(0, 1)[lay.r0:lay.r1]))
    return {"ok": ok, "census": dict(EV.schedule_fields()).get("p2p_check")}


def _unmatched(rank, P):
    cm = comm()
    t = torch.ones(8)
    sends = {(rank + 1) % P: t} if rank == 0 else {}          # rank 0 sends to rank 1; nobody posts the receive
    try:
        cm.p2p(sends, {}, "test_site")
        return "posted"
    except RowpairRefused as e:
        return str(e)


def _wrong_numel(rank, P):
    cm = comm()
    n = 8 if rank == 0 else 6                                  # rank 0 sends 8 to 1, rank 1 expects 6 from 0
    try:
        cm.p2p({1: torch.ones(n)} if rank == 0 else {}, {0: torch.empty(n)} if rank == 1 else {}, "numel_site")
        return "posted"
    except RowpairRefused as e:
        return str(e)


def _band_windows_rank_dependent(rank, P):
    lay = Layout(N, P, rank)
    z = torch.zeros(lay.R, N, C)
    windows = [(0, 4)] * P if rank == 0 else [(0, 5)] * P     # differs across ranks: the unmatched-transfer class
    try:
        ring.transpose_band(z, lay, windows)
        return "ran"
    except RowpairRefused as e:
        return str(e)


@pytest.mark.parametrize("P", [2, 3])
def test_matched_exchange_passes_and_census_names_the_check(check_on, P):
    from opt_core.mem.rowpair import evidence as EV
    EV.reset_schedule()
    res = run_ranks(P, _matched)
    assert all(r["ok"] for r in res) and all(r["census"] == "on" for r in res), res


@pytest.mark.parametrize("P", [2, 3])
def test_unmatched_send_is_refused_by_name_on_every_rank(check_on, P):
    res = run_ranks(P, _unmatched)
    assert all("ROWPAIR_P2P_CHECK" in r and "unmatched" in r and "test_site" in r for r in res), res


def test_numel_mismatch_is_refused(check_on):
    res = run_ranks(2, _wrong_numel)
    assert all("no matching" in r for r in res), res


@pytest.mark.parametrize("P", [2, 3])
def test_rank_dependent_transpose_band_windows_are_refused(check_on, P):
    res = run_ranks(P, _band_windows_rank_dependent)
    assert all("transpose_band: windows" in r and "differs across ranks" in r for r in res), res


def test_off_by_default_collects_nothing(monkeypatch):
    from opt_core.mem.rowpair import evidence as EV
    monkeypatch.delenv("ROWPAIR_P2P_CHECK", raising=False)
    EV.reset_schedule()                                        # the census words are process-global
    res = run_ranks(2, _matched)
    assert all(r["ok"] for r in res) and all(r["census"] is None for r in res), res
