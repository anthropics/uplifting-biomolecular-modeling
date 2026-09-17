"""CPU tests of opt_core.host.workers (order, bounded look-ahead, failure delivery, accounting) and the framework-free parts of
opt_core.host.outputs / coldstart / events (writer ownership at submit in every mode, back-pressure and accounting, the exit guard,
to_host pass-through, refusals, line grammar)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

import opt_core


def _run_py(code):
    """A child interpreter with this opt_core importable (the package's parent directory first on PYTHONPATH)."""
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)

from opt_core.host import HostEvent, HostRefused
from opt_core.host.events import require
from opt_core.host.outputs import AsyncWriter, to_host
from opt_core.host.workers import Prefetcher, WorkerError


def _square(x):
    return x * x


def test_prefetch_delivers_in_input_order_with_workers():
    def build(x):
        time.sleep(0.01 * (5 - x % 5))                  # later items finish first; delivery order must not change
        return x * 10
    with Prefetcher(build, list(range(12)), workers=4, depth=4) as pf:
        got = [(i, v) for i, v in pf]
    assert got == [(i, i * 10) for i in range(12)]
    t = pf.tally()
    assert (t["submitted"], t["produced"], t["consumed"], t["failed"], t["cancelled"]) == (12, 12, 12, 0, 0)
    assert pf.active_line("kit") == "[kit] LEVER name=F6.item_ordering_prefetch state=on impl=host.workers origin=core pool=featurise workers=4 depth=4 executor=thread"
    assert pf.tally_line("kit").startswith("[kit] HOST host.workers TALLY name=featurise submitted=12 produced=12 consumed=12 failed=0 cancelled=0 discarded=0 unsubmitted=0")


def test_prefetch_look_ahead_is_bounded():
    started = []
    lock = threading.Lock()

    def build(x):
        with lock:
            started.append(x)
        return x
    pf = Prefetcher(build, list(range(100)), workers=2, depth=3)
    time.sleep(0.2)
    with lock:
        assert len(started) <= 3                        # nothing beyond the window was built before consumption began
    it = iter(pf)
    next(it)
    time.sleep(0.2)
    with lock:
        assert len(started) <= 4
    pf.close()
    t = pf.tally()
    assert t["consumed"] == 1 and t["unsubmitted"] == 100 - t["submitted"]
    assert t["submitted"] == t["consumed"] + t["failed"] + t["cancelled"] + t["discarded"]


def test_failed_build_raises_at_its_turn_with_item():
    def build(x):
        if x == 3:
            raise ValueError("bad msa")
        return x
    pf = Prefetcher(build, list(range(6)), workers=2, depth=2)
    got = []
    with pytest.raises(WorkerError) as ei:
        for item, v in pf:
            got.append(item)
    assert got == [0, 1, 2] and ei.value.index == 3 and isinstance(ei.value.__cause__, ValueError)
    t = pf.tally()
    assert t["failed"] >= 1 and t["submitted"] == t["consumed"] + t["failed"] + t["cancelled"] + t["discarded"]


def test_records_form_accounts_every_item():
    def build(x):
        if x % 4 == 0:
            raise RuntimeError(f"item {x}")
        return -x
    with Prefetcher(build, list(range(10)), workers=3, depth=3) as pf:
        recs = list(pf.records())
    assert [r.index for r in recs] == list(range(10))
    assert [r.ok for r in recs] == [x % 4 != 0 for x in range(10)]
    assert all((r.value == -r.item) for r in recs if r.ok)
    t = pf.tally()
    assert (t["consumed"], t["failed"]) == (7, 3)
    with pytest.raises(RuntimeError):
        with Prefetcher(build, [1, 2], workers=1, depth=1) as pf2:
            next(iter(pf2))
            next(pf2.records())                         # one delivery form per prefetcher


def test_process_executor_and_unpicklable_refusal():
    with Prefetcher(_square, [1, 2, 3], workers=2, depth=2, executor="process") as pf:
        assert [v for _, v in pf] == [1, 4, 9]
    with pytest.raises(HostRefused) as ei:
        Prefetcher(lambda x: x, [1], executor="process")
    assert ei.value.event.reason == "unpicklable" and ei.value.__cause__ is not None
    with Prefetcher(_square, [2, (lambda: 0), 3], workers=1, depth=2, executor="process") as pf:   # an item a worker process cannot receive
        recs = list(pf.records())
    assert [r.ok for r in recs] == [True, False, True] and isinstance(recs[1].error, HostRefused)
    assert recs[1].error.event.reason == "unpicklable" and recs[0].value == 4
    with pytest.raises(ValueError):
        Prefetcher(_square, [1], workers=0)
    p2 = Prefetcher(_square, [1, 2])
    it = iter(p2)
    with pytest.raises(RuntimeError):
        p2.records()                                    # the delivery form is chosen at the first call, before any item is driven
    p2.close()


def test_async_writer_backpressure_and_failure_accounting(tmp_path):
    written = []

    def write(item):
        name, payload = item
        if payload is None:
            raise IOError(f"cannot write {name}")
        (tmp_path / name).write_text(payload)
        written.append(name)
        return len(payload)
    w = AsyncWriter(write, workers=2, max_pending=2)
    for i in range(6):
        w.submit((f"f{i}.txt", "x" * i))
    w.submit(("bad.txt", None))
    with pytest.raises(RuntimeError) as ei:
        w.close()
    assert isinstance(ei.value.__cause__, IOError)
    assert sorted(written) == [f"f{i}.txt" for i in range(6)]
    t = w.tally()
    assert (t["submitted"], t["written"], t["failed"], t["pending"]) == (7, 6, 1, 0)
    assert w.failures[0][0] == ("bad.txt", None)
    assert w.tally_line("kit") == "[kit] HOST host.outputs TALLY part=writer name=writer mode=thread submitted=7 written=6 failed=1 pending=0"
    with pytest.raises(RuntimeError):
        w.submit(("late.txt", "z"))                     # closed


def test_to_host_passes_host_values_through():
    tree = {"plddt": [1.0, 2.0], "meta": ("a", 3), "nested": {"s": "x"}}
    stats = {}
    assert to_host(tree, stats=stats) is tree           # nothing to move: the same object back
    assert stats["leaves"] == 0
    np = pytest.importorskip("numpy")
    arr = np.zeros(3)
    out = to_host({"a": arr})
    assert out["a"] is arr


def test_event_grammar_and_require_refusals():
    ev = HostEvent("host.memo", "BYPASS", "smiles_ligand", {"name": "feat", "chains": [2, 3]})
    assert ev.line("ef2-opt") == "[ef2-opt] HOST host.memo BYPASS reason=smiles_ligand name=feat chains=2,3"
    with pytest.raises(ValueError):
        HostEvent("host.memo", "WHATEVER")
    with pytest.raises(HostRefused) as ei:
        require("host.coldstart", "no_such_framework_xyz")
    assert ei.value.event.reason == "missing:no_such_framework_xyz"
    assert str(ei.value) == "HOST host.coldstart REFUSED reason=missing:no_such_framework_xyz"
    assert ei.value.event.lever_line("kit", "F6.weights_residency_init") == "[kit] LEVER name=F6.weights_residency_init state=skipped reason=missing:no_such_framework_xyz impl=host.coldstart origin=core"
    assert ei.value.event.lever_line("kit", "local.thing", origin="kit").endswith("impl=host.coldstart origin=kit")
    with pytest.raises(ValueError):
        ei.value.event.lever_line("kit", "F6.weights_residency_init", origin="elsewhere")
    with pytest.raises(ValueError):
        HostEvent("host.memo", "TALLY").lever_line("kit", "F6.feature_cache")   # records are not activation states
    with pytest.raises(HostRefused) as ei:
        require("host.coldstart", "json", min_version="99.0")     # json.__version__ = 2.0.9
    assert ei.value.event.reason == "too_old:json" and ei.value.event.fields["need"] == "99.0"


def test_import_surface_is_stdlib_only():
    code = ("import sys; import opt_core.host, opt_core.host.memo, opt_core.host.workers, opt_core.host.outputs, opt_core.host.coldstart; "
            "bad = [m for m in ('torch', 'jax', 'numpy', 'triton') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = _run_py(code)
    assert r.returncode == 0, r.stdout + r.stderr


def test_fast_exit_runs_exit_tallies_and_syncs(tmp_path):
    out = tmp_path / "result.json"
    code = (
        "import sys\n"
        "from opt_core import report\n"
        "from opt_core.host.coldstart import fast_exit, lazy_import\n"
        "m = lazy_import('json')\n"
        f"open({str(out)!r}, 'w').write(m.dumps({{'ok': 1}}))\n"
        "report.register_exit_tally('kit', lambda: '[kit] EXIT pid=0 source=memory levers=none')\n"
        f"fast_exit(0, synced=[{str(out)!r}])\n"
        "print('never reached')\n"
    )
    r = _run_py(code)
    assert r.returncode == 0, r.stderr
    assert "[kit] EXIT pid=0 source=memory levers=none" in r.stderr
    assert "never reached" not in r.stdout
    assert out.read_text() == '{"ok": 1}'


def test_async_writer_copies_on_submit_reused_buffer_40_files(tmp_path):
    np = pytest.importorskip("numpy")
    buf = np.zeros(64, dtype=np.int64)                 # ONE producer buffer re-used for every item (the normal producer pattern)
    meta = {"name": None, "scores": buf}

    def write(item):
        time.sleep(0.005)                               # the writer lags the producer: by-reference hand-off would see later items' bytes
        np.save(str(tmp_path / (item["name"] + ".npy")), item["scores"])
        return item["name"]
    with AsyncWriter(write, workers=2, max_pending=4) as w:
        assert w.evidence()["copy_on_submit"] == "on"
        for i in range(40):
            buf[:] = i
            meta["name"] = f"item{i:02d}"
            w.submit(meta)                              # returns with an owned snapshot; the producer overwrites buf at once
    wrong = [i for i in range(40) if int(np.load(str(tmp_path / f"item{i:02d}.npy"))[0]) != i]
    assert wrong == [], wrong
    assert w.tally()["written"] == 40


def test_async_writer_by_reference_form_is_explicit():
    seen = []
    with AsyncWriter(lambda item: seen.append(item), copy_on_submit=False) as w:
        obj = {"payload": bytearray(b"x")}
        w.submit(obj)
    assert seen[0] is obj                               # handed over, not copied — the caller gave it up


def test_snapshot_owns_arrays_and_keeps_immutables():
    from opt_core.host.outputs import snapshot
    np = pytest.importorskip("numpy")
    a = np.arange(3)
    item = {"a": a, "t": ("s", 1, None), "b": bytearray(b"yz"), "l": [a]}
    snap = snapshot(item)
    assert snap["a"] is not a and (snap["a"] == a).all() and snap["l"][0] is not a
    assert snap["t"] == ("s", 1, None) and snap["b"] == b"yz" and isinstance(snap["b"], bytes)
    a[0] = 99
    assert snap["a"][0] == 0


# ------------------------------------------------------------------------------------------------ AsyncWriter process modes + exit guard


def _np_save(item):                                     # module level: a fork worker unpickles it by reference to this module
    import numpy as np
    time.sleep(0.005)
    np.save(item["path"], item["scores"])


def test_async_writer_fork_mode_owns_the_item_at_submit_reused_buffer_40_files(tmp_path):
    np = pytest.importorskip("numpy")
    mp = pytest.importorskip("multiprocessing")
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("no fork start method on this platform")
    buf = np.zeros(64, dtype=np.int64)
    meta = {"path": None, "scores": buf}
    with AsyncWriter(_np_save, workers=2, max_pending=4, mode="fork") as w:
        ev = w.evidence()
        assert (ev["mode"], ev["copy_on_submit"]) == ("fork", "on")
        assert w.active_line("kit") == "[kit] LEVER name=F6.output_overlap state=on impl=host.outputs origin=core part=writer writer=writer mode=fork workers=2 max_pending=4 copy_on_submit=on"
        for i in range(40):
            buf[:] = i
            meta["path"] = str(tmp_path / f"item{i:02d}.npy")
            w.submit(meta)                              # pickled here; the producer overwrites buf at once
    wrong = [i for i in range(40) if int(np.load(str(tmp_path / f"item{i:02d}.npy"))[0]) != i]
    assert wrong == [], wrong
    t = w.tally()
    assert (t["submitted"], t["written"], t["failed"], t["pending"], t["mode"]) == (40, 40, 0, 0, "fork")


def test_async_writer_process_mode_rules():
    with pytest.raises(ValueError, match="unknown AsyncWriter mode"):
        AsyncWriter(print, mode="processes")
    with pytest.raises(ValueError, match="thread mode only"):
        AsyncWriter(print, mode="fork", copy_on_submit=False)
    w = AsyncWriter(lambda item: None, mode="fork")     # a lambda cannot be pickled: the usage error is raised by submit itself
    with pytest.raises(TypeError, match="cannot pickle"):
        w.submit({"path": "x"})
    assert w.tally()["submitted"] == 0 and w._pool is None      # nothing was queued and no pool was forked
    w.close()
    lock_item = {"path": "y", "handle": threading.Lock()}
    wt = AsyncWriter(lambda item: None)                 # thread mode: an item deepcopy cannot own is named at submit too
    with pytest.raises(TypeError, match="cannot snapshot item #0"):
        wt.submit(lock_item)
    assert wt.tally()["submitted"] == 0
    wt.close()


def test_async_writer_prunes_finished_entries(tmp_path):
    with AsyncWriter(lambda item: None, workers=2, max_pending=3) as w:
        for i in range(50):
            w.submit({"i": i, "blob": b"x" * 1000})
            assert len(w._futures) <= 3 + 1               # done entries are dropped at submit: a run's snapshots are not retained
    assert w.tally()["written"] == 50


_GUARD_SCRIPT = """
import os, sys, time
from opt_core.host.outputs import AsyncWriter
OUT = {out!r}
def w(item):
    time.sleep({sleep})
    if item.get("fail"):
        raise IOError("disk full: " + item["path"])
    open(item["path"], "w").write(item["text"])
writer = AsyncWriter(w, workers=2, max_pending=8, mode={mode!r})
{register}
for i in range(3):
    writer.submit({{"path": os.path.join(OUT, "f%d.txt" % i), "text": "t%d" % i, "fail": {fail} and i == 1}})
{tail}
"""


def _guard_run(tmp_path, mode, *, fail=False, sleep=0.3, register="", tail="# no drain, no close: the exit guard accounts"):
    mp = pytest.importorskip("multiprocessing")
    if mode != "thread" and mode not in mp.get_all_start_methods():
        pytest.skip(f"no {mode} start method on this platform")
    out = tmp_path / mode
    out.mkdir()
    r = _run_py(_GUARD_SCRIPT.format(out=str(out), sleep=sleep, mode=mode, register=register, fail=fail, tail=tail))
    files = sorted(p.name for p in out.iterdir() if p.suffix == ".txt")
    lines = [ln for ln in r.stderr.splitlines() if "reason=undrained_at_exit" in ln]
    return r, files, lines


def test_exit_guard_drains_pending_fork_writes_and_names_them(tmp_path):
    r, files, lines = _guard_run(tmp_path, "fork")
    assert r.returncode == 0, r.stderr
    assert files == ["f0.txt", "f1.txt", "f2.txt"]         # the guard drained them: nothing lost
    assert len(lines) == 1, r.stderr
    assert lines[0].startswith("[opt_core.host] HOST host.outputs TALLY reason=undrained_at_exit part=writer name=writer mode=fork pending_at_exit=")
    assert lines[0].endswith("submitted=3 written=3 failed=0")
    assert "pending_at_exit=0" not in lines[0]              # the writes WERE pending when the interpreter began to exit


def test_exit_guard_forces_a_nonzero_exit_when_an_undrained_write_was_lost(tmp_path):
    r, files, lines = _guard_run(tmp_path, "fork", fail=True, register="writer.register_exit_tally('acme-opt')")
    assert r.returncode == 70, r.stderr                     # EXIT_UNDRAINED_LOSS
    assert files == ["f0.txt", "f2.txt"]
    assert len(lines) == 1 and lines[0].startswith("[acme-opt] HOST host.outputs TALLY reason=undrained_at_exit") and lines[0].endswith("written=2 failed=1")


def test_exit_guard_names_a_thread_mode_failure_nobody_drained(tmp_path):
    # thread workers finish their queue before atexit runs; a FAILED write nobody drained must still not leave rc 0 behind
    r, files, lines = _guard_run(tmp_path, "thread", fail=True, sleep=0.01)
    assert r.returncode == 70, r.stderr
    assert files == ["f0.txt", "f2.txt"] and len(lines) == 1 and "mode=thread" in lines[0] and lines[0].endswith("written=2 failed=1")


def test_exit_guard_is_silent_for_a_drained_writer(tmp_path):
    r, files, lines = _guard_run(tmp_path, "fork", tail="writer.close()")
    assert r.returncode == 0 and files == ["f0.txt", "f1.txt", "f2.txt"] and lines == [], r.stderr
    r, files, lines = _guard_run(tmp_path, "thread", sleep=0.01, tail="writer.close()")
    assert r.returncode == 0 and len(files) == 3 and lines == [], r.stderr


def test_async_writer_spawn_mode_smoke(tmp_path):
    mod = tmp_path / "acme_writer_mod.py"
    mod.write_text("import os\ndef w(item):\n    open(item['path'], 'w').write(item['text'])\n")
    out = tmp_path / "spawned"
    out.mkdir()
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r}); os.environ['PYTHONPATH'] = {str(tmp_path)!r} + os.pathsep + os.environ.get('PYTHONPATH', '')\n"
        "from acme_writer_mod import w\n"
        "from opt_core.host.outputs import AsyncWriter\n"
        "with AsyncWriter(w, workers=2, mode='spawn') as writer:\n"
        "    for i in range(3):\n"
        f"        writer.submit({{'path': os.path.join({str(out)!r}, 'f%d.txt' % i), 'text': 't%d' % i}})\n"
        "print(writer.tally_line('kit'))\n"
    )
    r = _run_py(code)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in out.iterdir()) == ["f0.txt", "f1.txt", "f2.txt"]
    assert "[kit] HOST host.outputs TALLY part=writer name=writer mode=spawn submitted=3 written=3 failed=0 pending=0" in r.stdout


def test_exit_guard_counts_a_writer_garbage_collected_with_a_lost_write(tmp_path):
    # R-CORE H1: the writer is gone before atexit runs; its guard line printed at collection; the exit status must still be 70
    r, files, lines = _guard_run(tmp_path, "thread", fail=True, sleep=0.01, tail="import gc, time\ntime.sleep(0.2)\ndel writer\ngc.collect()\nprint('collected')")
    assert r.returncode == 70, (r.returncode, r.stderr)
    assert "collected" in r.stdout and files == ["f0.txt", "f2.txt"]
    assert len(lines) == 1 and "reason=undrained_at_exit" in lines[0] and lines[0].endswith("written=2 failed=1")


def test_exit_guard_prints_the_kits_earlier_exit_tallies_before_the_forced_status(tmp_path):
    # R-CORE R2-L: report exit tallies registered before the writer existed would run after the guard (LIFO) — the guard re-issues them
    pre = "from opt_core import report\nreport.register_exit_tally('acme-opt:early', lambda: '[acme-opt] EXIT pid=0 early_tally=1')\n"
    script = _GUARD_SCRIPT.replace("from opt_core.host.outputs import AsyncWriter\n", "from opt_core.host.outputs import AsyncWriter\n" + pre)
    out = tmp_path / "thread"
    out.mkdir()
    r = _run_py(script.format(out=str(out), sleep=0.01, mode="thread", register="", fail=True, tail="# undrained"))
    assert r.returncode == 70, r.stderr
    assert r.stderr.count("[acme-opt] EXIT pid=0 early_tally=1") == 1, r.stderr
