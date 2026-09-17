"""opt_core.seq.hostio: rows come back in submission order at any thread count, every item is a row (ok or failed-by-name), slots are
always released, depth and wait_for hold the device loop back. The pool is a CPU stand-in with the pinned/release protocol."""
import random
import threading
import time

import pytest

from opt_core.seq import hostio
from opt_core import report


class FakePool:
    """n slots; lease() hands out a free slot id and stores the 'device output'; pinned() returns it after a small wait (the copy);
    release() frees. Asserts the protocol: no read without a lease, no double release, no lease of a held slot."""
    def __init__(self, n=4, copy_s=0.0):
        self.n, self.copy_s = n, copy_s
        self.held = {}
        self.released = []
        self.lock = threading.Lock()

    def lease(self, value, wait_for=None):
        for _ in range(10000):
            with self.lock:
                free = [k for k in range(self.n) if k not in self.held]
                if free:
                    self.held[free[0]] = value
                    return free[0]
            if wait_for is not None:
                wait_for(min(range(self.n)))          # block on one slot the writer holds, then retry
            else:
                time.sleep(0.001)
        raise AssertionError("pool starved")

    def pinned(self, slot):
        time.sleep(self.copy_s)
        with self.lock:
            assert slot in self.held, "read of slot %r without a lease" % slot
            return self.held[slot]

    def release(self, slot):
        with self.lock:
            assert slot in self.held, "double release of slot %r" % slot
            del self.held[slot]
            self.released.append(slot)


def write_ok(item_id, arrays, meta):
    time.sleep(random.Random(item_id).uniform(0, 0.004))   # uneven host work so threads finish out of order
    return {"id": item_id, "value": arrays, "meta": meta, "ok": True}


@pytest.mark.parametrize("threads", [1, 3])
def test_rows_in_submission_order_and_every_slot_released(threads):
    pool = FakePool(n=4, copy_s=0.001)
    seen = []
    w = hostio.D2HWriter(pool, write_ok, threads=threads, on_row=seen.append)
    n = 25
    for i in range(n):
        slot = pool.lease(i * 10, wait_for=w.wait_for)
        w.submit("item%02d" % i, slot, meta={"i": i})
    out = w.close()
    ids = [r["id"] for r in out["rows"]]
    assert ids == ["item%02d" % i for i in range(n)] == [r["id"] for r in seen]
    assert [r["value"] for r in out["rows"]] == [i * 10 for i in range(n)]
    assert out["failed"] == [] and len(pool.released) == n and pool.held == {}
    s = out["stats"]
    assert s["submitted"] == n == s["written"] and s["failed"] == 0 and s["threads"] == threads and s["in_flight"] == 0
    assert s["max_in_flight"] <= pool.n


def test_failures_are_named_rows_never_dropped():
    pool = FakePool(n=3)

    class BadPool(FakePool):
        def pinned(self, slot):
            v = FakePool.pinned(self, slot)
            if v == 4:
                raise RuntimeError("copy event failed")
            return v

    def write(item_id, arrays, meta):
        if arrays in (2, 7):
            raise IOError("disk full at %s" % item_id)
        return {"id": item_id, "ok": True}

    bad = BadPool(n=3)
    w = hostio.D2HWriter(bad, write, threads=2)
    for i in range(10):
        w.submit("u%d" % i, bad.lease(i, wait_for=w.wait_for))
    out = w.close()
    assert [r["id"] for r in out["rows"]] == ["u%d" % i for i in range(10)]            # positions kept
    failed = {r["id"]: r["error"] for r in out["rows"] if not r["ok"]}
    assert set(failed) == {"u2", "u4", "u7"}                                              # every injected failure is a named row
    assert sorted(f["id"] for f in out["failed"]) == sorted(failed)                         # the failed list names each exactly once, none
    assert len(out["failed"]) == len(failed) == 3                                           # dropped (its order is worker-completion order)
    assert failed["u2"].startswith("OSError: disk full") and failed["u4"] == "RuntimeError: copy event failed"
    stages = {r["id"]: r["stage"] for r in out["rows"] if not r["ok"]}
    assert stages == {"u2": "write", "u4": "pinned", "u7": "write"}
    assert out["stats"]["written"] == 7 and out["stats"]["failed"] == 3 and bad.held == {} and len(bad.released) == 10
    assert out["stats"]["submitted"] == out["stats"]["written"] + out["stats"]["failed"] == 10


def test_non_dict_row_is_a_named_failure():
    pool = FakePool(n=2)
    w = hostio.D2HWriter(pool, lambda i, a, m: ["not", "a", "row"])
    w.submit("x", pool.lease(1))
    out = w.close()
    assert out["rows"][0]["ok"] is False and "not a dict row" in out["rows"][0]["error"] and out["stats"]["failed"] == 1


def test_depth_bounds_in_flight_and_wait_for_blocks():
    pool = FakePool(n=8)
    gate = threading.Event()

    def slow_write(item_id, arrays, meta):
        gate.wait(2.0)
        return {"id": item_id, "ok": True}

    w = hostio.D2HWriter(pool, slow_write, threads=1, depth=2)
    w.submit("a", pool.lease("A"))
    w.submit("b", pool.lease("B"))
    assert w.in_flight() == 2
    t0 = time.perf_counter()
    blocked = {}

    def third():
        w.submit("c", pool.lease("C"))                 # must block until the gate opens and an item completes
        blocked["s"] = time.perf_counter() - t0
    th = threading.Thread(target=third)
    th.start()
    time.sleep(0.05)
    assert th.is_alive() and w.in_flight() == 2        # still blocked at depth
    waited = {}

    def waiter():
        t = time.perf_counter(); w.wait_for(0); waited["s"] = time.perf_counter() - t     # slot 0 holds item "a"
    th2 = threading.Thread(target=waiter)
    th2.start()
    time.sleep(0.05)
    assert th2.is_alive()
    gate.set()
    th.join(2.0); th2.join(2.0)
    out = w.close()
    assert [r["id"] for r in out["rows"]] == ["a", "b", "c"] and blocked["s"] >= 0.05 and waited["s"] >= 0.04
    assert out["stats"]["max_in_flight"] == 2 and out["stats"]["blocked_s"] >= 0.05


def test_close_idempotent_submit_after_close_refused_and_line_fields():
    pool = FakePool(n=2)
    with hostio.D2HWriter(pool, write_ok, threads=1) as w:
        w.submit("only", pool.lease(3))
    out1 = w.close(); out2 = w.close()
    assert out1["rows"] == out2["rows"] and len(out1["rows"]) == 1
    with pytest.raises(RuntimeError):
        w.submit("late", 0)
    f = hostio.line_fields(w)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in f.items())
    line = report.kv(**f)
    assert line.startswith("writer=thread threads=1 submitted=1 written=1 failed=0 on_row_failed=0 release_failed=0 blocked_s=")
    assert report.kv(**hostio.line_fields(w, key="host_writer", when="active")) == "host_writer=thread threads=1 depth=none"
    with pytest.raises(ValueError):
        hostio.line_fields(w, when="later")
    with pytest.raises(ValueError):
        hostio.D2HWriter(pool, write_ok, threads=0)


def test_on_row_exception_is_recorded_not_raised():
    pool = FakePool(n=2)
    def on_row(row):
        raise KeyError("jsonl closed")
    w = hostio.D2HWriter(pool, write_ok, on_row=on_row)
    w.submit("z", pool.lease(9))
    out = w.close()
    assert out["rows"][0]["ok"] is True and out["stats"]["written"] == 1 and out["stats"]["failed"] == 0      # the item IS written
    assert out["stats"]["on_row_failed"] == 1 and out["failed"][0]["stage"] == "on_row" and out["failed"][0]["error"].startswith("on_row raised KeyError")
    assert out["stats"]["submitted"] == out["stats"]["written"] + out["stats"]["failed"]                       # counts close


def test_slot_identity_rules():
    same = hostio._same_slot
    assert same(3, 3) and not same(3, 4) and same("a", "a") and not same("3", 3)
    class IntLike:
        def __init__(self, v): self.v = v
        def __index__(self): return self.v
    assert same(IntLike(5), 5) and same(5, IntLike(5))            # numpy-integer-like handles compare by value
    buf1, buf2 = bytearray(4), bytearray(4)
    assert same(buf1, buf1) and not same(buf1, buf2)               # equal-valued buffers are different slots (equality)


def test_write_fn_runs_strictly_after_pinned_returns():
    """The documented contract: readiness is the pool's `pinned`; the writer never reads a slot before `pinned` returned. The stand-in
    pool's copy 'lands' late (a producer thread fills the slot and sets its event after a delay); write_fn must always see the landed value."""
    class LatePool(FakePool):
        def __init__(self, n):
            FakePool.__init__(self, n)
            self.events = {}
        def lease(self, value, wait_for=None):
            k = FakePool.lease(self, None, wait_for=wait_for)          # slot holds a placeholder until the 'copy' lands
            ev = threading.Event(); self.events[k] = ev
            def land():
                time.sleep(0.01)
                with self.lock:
                    self.held[k] = value
                ev.set()
            threading.Thread(target=land).start()
            return k
        def pinned(self, slot):
            assert self.events[slot].wait(2.0), "copy never landed"    # the event sync a CUDA pool does inside pinned()
            return FakePool.pinned(self, slot)

    pool = LatePool(n=3)
    seen = []
    w = hostio.D2HWriter(pool, lambda i, a, m: {"id": i, "value": a, "ok": True}, threads=2, on_row=seen.append)
    for i in range(12):
        w.submit(i, pool.lease("landed-%d" % i, wait_for=w.wait_for))
    out = w.close()
    assert [r["value"] for r in out["rows"]] == ["landed-%d" % i for i in range(12)]      # never the placeholder None
    assert out["stats"]["failed"] == 0 and out["stats"]["wait_s"] > 0


def test_release_failure_recorded_even_after_write_failure():
    class StickyPool(FakePool):
        def release(self, slot):
            FakePool.release(self, slot)
            if slot == 0:
                raise RuntimeError("pool refuses release of slot 0")
    pool = StickyPool(n=1)                                             # one slot: every item uses slot 0
    def write(item_id, arrays, meta):
        if item_id == "bad":
            raise IOError("disk full")
        return {"id": item_id, "ok": True}
    w = hostio.D2HWriter(pool, write, threads=1)
    w.submit("good", pool.lease(1, wait_for=w.wait_for))
    w.submit("bad", pool.lease(2, wait_for=w.wait_for))
    out = w.close()
    assert [r["id"] for r in out["rows"]] == ["good", "bad"] and out["rows"][0]["ok"] is True and out["rows"][1]["stage"] == "write"
    st = out["stats"]
    assert st["written"] == 1 and st["failed"] == 1 and st["release_failed"] == 2 and st["submitted"] == st["written"] + st["failed"]
    rel = [f for f in out["failed"] if f["stage"] == "release"]
    assert [f["id"] for f in rel] == ["good", "bad"]                     # the release fault after the WRITE failure is recorded too, never dropped
    assert report.kv(**hostio.line_fields(w)).split()[-2] == "release_failed=2"


def test_close_timeout_accounts_undelivered_items_by_row():
    pool = FakePool(n=4)
    gate = threading.Event()
    def stuck(item_id, arrays, meta):
        if item_id != "w0":
            gate.wait(5.0)                                             # w1.. hang until released after close
        return {"id": item_id, "ok": True}
    w = hostio.D2HWriter(pool, stuck, threads=1)
    for i in range(3):
        w.submit("w%d" % i, pool.lease(i))
    time.sleep(0.05)
    out = w.close(timeout=0.1)                                         # thread alive: w1 in write_fn, w2 queued
    ids = [(r["id"], r["ok"], r.get("stage")) for r in out["rows"]]
    assert ids == [("w0", True, None), ("w1", False, "close"), ("w2", False, "close")]
    st = out["stats"]
    assert st["submitted"] == 3 == st["written"] + st["failed"] and st["failed"] == 2 and st["threads_alive"]
    gate.set(); time.sleep(0.1)                                        # the stuck writes finish late: counted as late, rows unchanged
    assert w.stats()["late"] >= 1 and [r["id"] for r in w.rows] == ["w0", "w1", "w2"] and w.stats()["failed"] == 2


def test_on_row_runs_outside_the_writer_lock():
    """on_row may take its time or call back into the writer (stats / in_flight) and other writers keep delivering meanwhile."""
    pool = FakePool(n=6)
    inside = []
    w = None
    def on_row(row):
        inside.append((row["id"], w.in_flight(), w.stats()["written"]))  # re-entrant calls must not deadlock
        time.sleep(0.01)
    w = hostio.D2HWriter(pool, write_ok, threads=3, on_row=on_row)
    for i in range(12):
        w.submit(i, pool.lease(i, wait_for=w.wait_for))
    out = w.close()
    assert [r["id"] for r in out["rows"]] == list(range(12)) == [t[0] for t in inside]
    assert out["stats"]["on_row_failed"] == 0 and out["stats"]["failed"] == 0


def test_host_leases_protocol_over_a_byte_pool(tmp_path):
    """stage()/HostLeases without a GPU: LeasedHost handles are the slots; release returns every leased buffer to the pool."""
    class BytePool:
        def __init__(self): self.released = []
        def release(self, buf): self.released.append(buf)
    pool = BytePool()
    leases = hostio.HostLeases(pool)
    w = hostio.D2HWriter(leases, lambda i, tree, m: {"id": i, "n": len(tree["human"]), "ok": True}, threads=2)
    for i in range(5):
        h = hostio.LeasedHost({"human": [0.0] * (i + 1), "mouse": [1.0]}, leases=["buf%da" % i, "buf%db" % i], stats={"unpinned": 1 if i == 4 else 0})
        w.submit("r%d" % i, h)
    out = w.close()
    assert [r["n"] for r in out["rows"]] == [1, 2, 3, 4, 5] and out["stats"]["failed"] == 0
    assert sorted(pool.released) == sorted(["buf%d%s" % (i, ab) for i in range(5) for ab in "ab"])
    assert leases.items == 5 and leases.unpinned_leaves == 1
    with pytest.raises(TypeError):
        leases.pinned("not a handle")


def test_witness_words(tmp_path):
    good = tmp_path / "m.json"; good.write_text('{"ok": true}')
    data = tmp_path / "a.npz"; data.write_bytes(b"x" * 10)
    empty = tmp_path / "e.npz"; empty.write_bytes(b"")
    bad = tmp_path / "b.json"; bad.write_text("{nope")
    assert hostio.witness([str(good), str(data)]) == ""
    assert hostio.witness([str(good), str(tmp_path / "missing.npz")]) == "witness:" + str(tmp_path / "missing.npz")
    assert hostio.witness([str(empty)]) == "witness:" + str(empty)
    assert hostio.witness([str(bad)]) == "json:" + str(bad)
    assert hostio.witness([]) == ""


def test_stage_uses_the_core_to_host():
    outputs = pytest.importorskip("opt_core.host.outputs")
    h = hostio.stage(None, {"a": [1, 2], "b": ("x",)})            # no device leaves: the tree comes back as is, nothing leased
    assert isinstance(h, hostio.LeasedHost) and h.tree == {"a": [1, 2], "b": ("x",)} and h.leases == [] and h.stats.get("leaves", 0) == 0
