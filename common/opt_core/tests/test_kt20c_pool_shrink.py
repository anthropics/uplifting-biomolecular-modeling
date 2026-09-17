"""K.T20c — pinned-pool shrink accounting (CPU): PinPool.live, host_cache_trim's named answer, pinned_shrink's words / env switch / record."""
import os, sys, types
import pytest

torch = pytest.importorskip("torch")
from opt_core.mem import torch_hostpair as TH
from opt_core.mem.rowpair import heads as HD, trimul as RM


def test_pinpool_live_counts_and_release():
    pool = TH.PinPool(None, pin=False, pageable=True, lever="test")
    a = pool.alloc((1024,), torch.uint8, "a"); b = pool.alloc((2048,), torch.uint8, "b")
    assert pool.live() == (2, 3072)
    pool.release(a)
    assert pool.live() == (1, 2048)
    pool.release(b)
    assert pool.live() == (0, 0)


def test_host_cache_trim_answers_by_name():
    r = TH.host_cache_trim()
    assert set(("call", "available", "ok", "error", "rss_before", "rss_after")) <= set(r)
    if not r["available"]:
        assert r["ok"] is False and "no host-allocator empty-cache binding" in r["error"]
    pm = TH.proc_mem()
    assert "VmRSS" in pm or pm == {}


class _FakeHost:
    n_chunks, pinned_bytes, alloc_bytes, pinned = 3, 3 * 2 ** 30, 3 * 2 ** 30, True

class _FakeParked:
    host, where = _FakeHost(), "host_pinned"

class _FakePlan:
    parked = _FakeParked()


def test_pinned_shrink_off_by_env(monkeypatch):
    monkeypatch.setenv(HD.ENV_POOL_SHRINK, "0")
    lines = []
    rec = HD.pinned_shrink(lines.append, plan=_FakePlan())
    assert rec["pool_shrink"] == "off" and rec["kept_chunks"] == 3 and abs(rec["kept_gib"] - 3.0) < 1e-6
    assert lines and lines[0].startswith("[pool] pinned shrink: off (ROWPAIR_POOL_SHRINK=0)")


def test_pinned_shrink_releases_mirror_and_names_trim(monkeypatch):
    monkeypatch.delenv(HD.ENV_POOL_SHRINK, raising=False)
    released = {"n": 0}
    monkeypatch.setattr(RM, "host_mirror_bytes", lambda: (1, 2, 5 * 2 ** 30))
    monkeypatch.setattr(RM, "release_host_mirror", lambda: released.__setitem__("n", released["n"] + 1) or 1)
    monkeypatch.setattr(TH, "host_cache_trim", lambda lever="pool": {"call": "_host_emptyCache", "available": True, "ok": True, "error": None,
                                                                    "reserved_before": 9 * 2 ** 30, "reserved_after": 3 * 2 ** 30, "rss_before": 20.0, "rss_after": 14.0})
    seq = iter([{"VmRSS": 20.0, "VmLck": 0.0}, {"VmRSS": 14.0, "VmLck": 0.0}])
    monkeypatch.setattr(TH, "proc_mem", lambda: next(seq))
    lines = []
    rec = HD.pinned_shrink(lines.append, plan=_FakePlan())
    assert released["n"] == 1
    assert rec["pool_shrink"] == "on" and rec["call"] == "_host_emptyCache"
    assert abs(rec["returned_gib"] - 6.0) < 1e-6 and abs(rec["mirror_gib"] - 5.0) < 1e-6 and rec["kept_chunks"] == 3
    line = lines[0]
    assert line.startswith("[pool] pinned shrink: kept 3.00 GiB in 3 chunks (z_trunk park host_pinned), returned 6.00 GiB (row mirror 5.00 GiB in 2 chunks + cached-free 1.00 GiB) via _host_emptyCache")
    assert "VmRSS 20.0 -> 14.0 GiB" in line


def test_pinned_shrink_trim_unavailable_is_named(monkeypatch):
    monkeypatch.delenv(HD.ENV_POOL_SHRINK, raising=False)
    monkeypatch.setattr(RM, "host_mirror_bytes", lambda: (0, 0, 0))
    monkeypatch.setattr(RM, "release_host_mirror", lambda: 0)
    monkeypatch.setattr(TH, "host_cache_trim", lambda lever="pool": {"call": None, "available": False, "ok": False, "error": "no host-allocator empty-cache binding in torch X",
                                                                    "reserved_before": None, "reserved_after": None, "rss_before": 1.0, "rss_after": 1.0})
    lines = []
    rec = HD.pinned_shrink(lines.append, plan=None)
    assert rec["pool_shrink"] == "unavailable" and "TRIM UNAVAILABLE" in lines[0] and "nothing parked" in lines[0]
