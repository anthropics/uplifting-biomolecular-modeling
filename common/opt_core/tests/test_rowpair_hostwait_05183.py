"""``mem.rowpair.dist.host_wait_refusal`` — the words of a failed store wait in ``Comm.host_wait``: a wait that ran out its timeout says
``no rank signalled it within <T>s``; a wait that failed BEFORE its deadline says the store connection was lost (the rank hosting the store
exited before signalling, under a tcp:// rendezvous). Proven on synthetic store errors (the classifier) and on a real ``torch.distributed.TCPStore`` whose master
process is killed mid-wait (the client's wait fails within seconds, not at the timeout, and is worded as a lost store).

Run: ``python -m pytest tests/test_rowpair_hostwait_05183.py -q -rfE``.
"""
from __future__ import annotations

import datetime
import os
import select
import signal
import socket
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from opt_core.mem.rowpair import dist as D  # noqa: E402


class DistStoreError(RuntimeError):          # the names torch raises (their spelling is what the classifier reads; no torch import needed here)
    pass


class DistNetworkError(RuntimeError):
    pass


@pytest.mark.parametrize("exc,timeout_s,waited_s,expect", [
    (DistStoreError("wait timeout after 10000ms, keys: /rowpair/host/feats/0"), 10.0, 10.02, "no rank signalled it within 10.0s (DistStoreError)"),
    (RuntimeError("Socket Timeout"), 1800.0, 1799.5, "no rank signalled it within 1800.0s (RuntimeError)"),
    (DistStoreError("wait timeout after 300000ms"), 1800.0, 300.0, "no rank signalled it within 1800.0s (DistStoreError)"),      # the store's own words win over the clock
    (RuntimeError("Broken pipe"), 60.0, 59.0, "no rank signalled it within 60.0s (RuntimeError)"),                                # at the deadline (within the slack): a timeout
    (DistNetworkError("Connection reset by peer"), 1800.0, 1.2,
     "the store failed after 1.2s of a 1800.0s wait, before its timeout — the store is unreachable (under a tcp:// rendezvous: the rank hosting it exited before signalling) (DistNetworkError: Connection reset by peer)"),
    (RuntimeError("failed to recv, got 0 bytes"), 600.0, 42.0,
     "the store failed after 42.0s of a 600.0s wait, before its timeout — the store is unreachable (under a tcp:// rendezvous: the rank hosting it exited before signalling) (RuntimeError: failed to recv, got 0 bytes)"),
])
def test_host_wait_refusal_words(exc, timeout_s, waited_s, expect):
    words = D.host_wait_refusal("feats/0", timeout_s, waited_s, exc)
    assert words == f"host_wait('feats/0'): {expect}", words


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_a_killed_store_master_is_worded_as_a_lost_store_not_a_timeout():
    torch = pytest.importorskip("torch")
    import torch.distributed as td
    if not td.is_available():
        pytest.skip("torch.distributed not available")
    port = _free_port()
    master = subprocess.Popen([sys.executable, "-c",
                               "import datetime, sys, time, torch.distributed as td; "
                               f"s = td.TCPStore('127.0.0.1', {port}, 2, True, datetime.timedelta(seconds=120), wait_for_workers=False); "
                               "print('up', flush=True); time.sleep(120)"],
                              stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        ready, _, _ = select.select([master.stdout], [], [], 60.0)         # the master prints 'up' once it listens; a stalled child fails the test, never hangs it
        assert ready and master.stdout.readline().strip() == "up"
        client = td.TCPStore("127.0.0.1", port, 2, False, datetime.timedelta(seconds=60))
        t0 = time.monotonic()                                              # (i) a live store, a key nobody sets: the wait runs out its timeout
        with pytest.raises(Exception) as ei:                               #     (3 s > HOST_WAIT_SLACK_S, so the store's WORDS must classify it, not the clock)
            client.wait(["rowpair/host/never"], datetime.timedelta(seconds=3.0))
        waited = time.monotonic() - t0
        words = D.host_wait_refusal("never", 3.0, 0.0, ei.value)           #     waited_s=0 handed in: only the store's words can make this a timeout
        assert "no rank signalled it within 3.0s" in words and waited >= 2.5, (words, waited, repr(ei.value))
        os.killpg(master.pid, signal.SIGKILL)                              # (ii) the master dies: the client's wait fails long before its deadline
        master.wait(10)
        t0 = time.monotonic()
        with pytest.raises(Exception) as ei:
            client.wait(["rowpair/host/feats/0"], datetime.timedelta(seconds=30.0))
        waited = time.monotonic() - t0
        words = D.host_wait_refusal("feats/0", 30.0, waited, ei.value)
        assert waited < 20.0 and "the store failed after" in words and "before its timeout" in words and "exited before signalling" in words, (words, waited)
    finally:
        if master.poll() is None:
            os.killpg(master.pid, signal.SIGKILL)
            master.wait(10)
