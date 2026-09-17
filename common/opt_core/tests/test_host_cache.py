"""host_cache: the background writer writes the same bytes as the inline call and accounts for every write; fast_init patches and
restores exactly the named sites and refuses an incomplete load; the resident item loop puts every item in one status and journals it."""
import json
import os
import subprocess
import sys
import types

import pytest

from opt_core import report
from opt_core.host_cache import bg_writer, fast_init, resident

CORE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(fast_init.__file__))))   # common/opt_core


# ----------------------------------------------------------------------------- bg_writer
def render(i: int) -> bytes:
    return (f"MODEL {i}\n" + "".join(f"ATOM  {j:5d}  CA  GLY A{j:4d}    {i:8.3f}{j:8.3f}   0.000\n" for j in range(50)) + "ENDMDL\n").encode()


def stock_write(path, i):                       # module level: picklable for the process modes
    with open(path, "wb") as fh:
        fh.write(render(i))


@pytest.mark.parametrize("mode,workers", [("off", 1), ("thread", 3), ("fork", 2)])
def test_writer_bytes_identical_and_census(tmp_path, mode, workers):
    if mode == "fork" and sys.platform.startswith("win"):
        pytest.skip("no fork")
    ref = tmp_path / "ref"; out = tmp_path / mode; ref.mkdir(); out.mkdir()
    for i in range(20):
        stock_write(str(ref / f"d{i}.pdb"), i)
    w = bg_writer.BackgroundWriter(mode, workers, name="pdb")
    for i in range(10):
        w.submit(stock_write, str(out / f"d{i}.pdb"), i)
    for i in range(10, 20):
        w.write_bytes(str(out / f"d{i}.pdb"), render(i))
    census = w.join()
    assert census["submitted"] == 20 and census["written"] == 20 and census["failed"] == 0 and census["complete"] is True
    assert census["workers"] == (0 if mode == "off" else workers)
    for i in range(20):
        assert (out / f"d{i}.pdb").read_bytes() == (ref / f"d{i}.pdb").read_bytes()
    assert not [p for p in os.listdir(out) if ".part" in p]
    assert w.fields() == {"bg_writer": f"{mode}:{census['workers']}", "bg_written": "20/20"}
    assert report.kv(**w.fields()) == f"bg_writer={mode}:{census['workers']} bg_written=20/20"
    assert w.join() == census                                        # idempotent
    with pytest.raises(RuntimeError):
        w.submit(stock_write, str(out / "late.pdb"), 0)


def failing_write(path, i):
    if i == 3:
        raise OSError(f"disk full at {path}")
    stock_write(path, i)


@pytest.mark.parametrize("mode", ["thread", "fork"])
def test_writer_failure_is_counted_named_and_raised_after_census(tmp_path, mode):
    w = bg_writer.BackgroundWriter(mode, 2)
    for i in range(6):
        w.submit(failing_write, str(tmp_path / f"d{i}.pdb"), i)
    with pytest.raises(bg_writer.BackgroundWriterError) as e:
        w.join()
    c = e.value.census
    assert (c["submitted"], c["written"], c["failed"], c["complete"]) == (6, 5, 1, False)
    assert c["errors"] == [f"#3 {tmp_path / 'd3.pdb'}: OSError: disk full at {tmp_path / 'd3.pdb'}"]
    assert w.fields()["bg_written"] == "5/6"


def test_writer_off_mode_raises_inline_like_stock(tmp_path):
    w = bg_writer.BackgroundWriter("off")
    with pytest.raises(OSError):
        w.submit(failing_write, str(tmp_path / "d3.pdb"), 3)
    assert w.census()["failed"] == 1 and w.census()["written"] == 0


def test_writer_context_manager_joins_on_error_path(tmp_path):
    with pytest.raises(ZeroDivisionError):
        with bg_writer.BackgroundWriter("thread", 2) as w:
            for i in range(5):
                w.submit(stock_write, str(tmp_path / f"d{i}.pdb"), i)
            1 / 0
    assert w.census()["written"] == 5 and w.census()["pending"] == 0
    assert all((tmp_path / f"d{i}.pdb").exists() for i in range(5))


def test_writer_backpressure_bounds_pending(tmp_path):
    w = bg_writer.BackgroundWriter("thread", 1, max_pending=4)
    for i in range(9):
        w.submit(stock_write, str(tmp_path / f"d{i}.pdb"), i)
        assert w.census()["pending"] < 4
    assert w.join()["written"] == 9


def test_writer_unknown_mode_and_unjoined_line(tmp_path):
    with pytest.raises(ValueError):
        bg_writer.BackgroundWriter("async")
    # a process that exits with pending writes prints ONE named line (the census in the manifest is what fails the run)
    script = tmp_path / "unjoined.py"
    script.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {CORE_DIR!r})\n"
        "from opt_core.host_cache import bg_writer\n"
        "def slow(p):\n    time.sleep(0.2); open(p, 'w').write('x')\n"
        "w = bg_writer.BackgroundWriter('thread', 1, name='cif')\n"
        f"w.submit(slow, {str(tmp_path / 'late.cif')!r})\n")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr                               # drained by the guard: nothing lost → exit code untouched
    assert r.stderr.count("BG-WRITER NOT JOINED") == 1               # one line, once (atexit and __del__ do not both print)
    assert "[opt_core.host_cache] BG-WRITER NOT JOINED name=cif mode=thread pending=1 submitted=1 written=1 failed=0" in r.stderr
    assert (tmp_path / "late.cif").read_text() == "x"                # the guard did the write


def write_buffer(path, buf):
    with open(path, "wb") as fh:
        fh.write(bytes(buf))


@pytest.mark.parametrize("mode", ["thread", "fork"])
def test_writer_snapshots_arguments_at_submit(tmp_path, mode):
    """The caller reuses ONE mutable buffer for every item; each file still holds its own item's bytes."""
    buf = bytearray(64)
    w = bg_writer.BackgroundWriter(mode, 2)
    for i in range(40):
        buf[:] = bytes([i % 256]) * 64
        w.submit(write_buffer, str(tmp_path / f"d{i}.bin"), buf)
    assert w.join()["written"] == 40
    for i in range(40):
        assert (tmp_path / f"d{i}.bin").read_bytes() == bytes([i % 256]) * 64, i
    with pytest.raises(TypeError):
        bg_writer.BackgroundWriter("thread", 1).submit(write_buffer, str(tmp_path / "x"), (x for x in ()))   # unpicklable → usage error at submit


def test_writer_exit_guard_drains_fork_writes_and_fails_loud_on_loss(tmp_path):
    script = tmp_path / "unjoined_fork.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {CORE_DIR!r})\n"
        "from opt_core.host_cache import bg_writer\n"
        "def wr(p, fail):\n"
        "    if fail: raise OSError('disk full')\n"
        "    open(p, 'w').write('x')\n"
        "w = bg_writer.BackgroundWriter('fork', 2, name='pdb')\n"
        f"for i in range(3): w.submit(wr, {str(tmp_path)!r} + f'/f{{i}}.pdb', False)\n"
        "fail = len(sys.argv) > 1\n"
        f"if fail: w.submit(wr, {str(tmp_path)!r} + '/lost.pdb', True)\n")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert all((tmp_path / f"f{i}.pdb").read_text() == "x" for i in range(3))          # drained: the writes landed
    assert r.stderr.count("BG-WRITER NOT JOINED") == 1 and "written=3 failed=0" in r.stderr
    r2 = subprocess.run([sys.executable, str(script), "lose-one"], capture_output=True, text=True, timeout=120)
    assert r2.returncode == bg_writer.EXIT_UNJOINED_LOSS, r2.stderr                     # a lost write never exits 0
    assert "written=3 failed=1" in r2.stderr and r2.stderr.count("BG-WRITER NOT JOINED") == 1


# ----------------------------------------------------------------------------- fast_init
def make_engine():
    """A stand-in for an engine: module `initialize` defines the initialisers (module-global lookups, as real modules do), module
    `layers` imported one by name, construction calls both."""
    initialize = types.ModuleType("fake_engine.initialize")
    exec(
        "draws = []\n"
        "def trunc_normal_init_(weights, scale=1.0, fan='fan_in'):\n"
        "    draws.append(('trunc', scale)); weights[:] = [0.5] * len(weights); return weights\n"
        "def lecun_normal_init_(weights):\n"
        "    draws.append(('lecun',)); return trunc_normal_init_(weights, scale=1.0)\n", initialize.__dict__)
    layers = types.ModuleType("fake_engine.layers")
    layers.trunc_normal_init_ = initialize.trunc_normal_init_        # `from initialize import trunc_normal_init_`

    def build():
        w1, w2 = [0.0] * 4, [0.0] * 4
        layers.trunc_normal_init_(w1, scale=2.0)
        initialize.lecun_normal_init_(w2)
        return w1, w2
    return initialize, layers, build, initialize.draws


def test_fast_init_patches_every_listed_site_counts_and_restores():
    initialize, layers, build, draws = make_engine()
    orig = (initialize.trunc_normal_init_, layers.trunc_normal_init_, initialize.lecun_normal_init_)
    with fast_init.suppressed(initialize, layers) as rec:
        w1, w2 = build()
        assert w1 == [0.0] * 4 and w2 == [0.0] * 4                     # nothing initialised
    assert draws == [("lecun",)]                                     # lecun ran (not listed) but its inner trunc call was a no-op
    assert rec.calls == {"trunc_normal_init_": 2} and rec.skipped == 2 and rec.restored
    assert sorted(rec.sites) == ["fake_engine.initialize.trunc_normal_init_", "fake_engine.layers.trunc_normal_init_"]
    assert (initialize.trunc_normal_init_, layers.trunc_normal_init_, initialize.lecun_normal_init_) == orig
    build()
    assert draws[-3:] == [("trunc", 2.0), ("lecun",), ("trunc", 1.0)]        # restored: stock behaviour again
    assert rec.fields() == {"fast_init": "unchecked:2"}                 # no check_loaded yet: the line says so
    fast_init.check_loaded(missing_keys=[], record=rec)
    assert rec.fields() == {"fast_init": "skipped:2"} and report.kv(**rec.fields()) == "fast_init=skipped:2"
    assert fast_init.off_fields() == {"fast_init": "off"}


def test_fast_init_more_names_and_restore_on_error():
    initialize, layers, build, draws = make_engine()
    orig = initialize.lecun_normal_init_
    with pytest.raises(KeyError):
        with fast_init.suppressed(initialize, layers, names=("trunc_normal_init_", "lecun_normal_init_")) as rec:
            build()
            raise KeyError("load failed")
    assert rec.calls == {"trunc_normal_init_": 1, "lecun_normal_init_": 1} and rec.restored
    assert initialize.lecun_normal_init_ is orig and draws == []


def test_fast_init_stale_site_list_and_empty_names_refused():
    with pytest.raises(fast_init.FastInitError):
        with fast_init.suppressed(types.ModuleType("nothing_here")):
            pass
    with pytest.raises(ValueError):
        with fast_init.suppressed(types.ModuleType("m"), names=()):
            pass


def test_check_loaded_fail_closed():
    Result = types.SimpleNamespace
    rec = fast_init.check_loaded(Result(missing_keys=[], unexpected_keys=["ema.decay"]))
    assert rec.checked and rec.missing_keys == [] and rec.unexpected_keys == ["ema.decay"]
    with pytest.raises(fast_init.FastInitError) as e:
        fast_init.check_loaded(Result(missing_keys=["trunk.layers.0.w", "trunk.layers.1.w"], unexpected_keys=[]))
    assert "2 parameter(s) unrestored" in str(e.value) and "trunk.layers.0.w" in str(e.value)
    assert fast_init.check_loaded(missing_keys=["buf.idx"], ignore_missing=["buf.idx"]).missing_keys == []
    with pytest.raises(fast_init.FastInitError):
        fast_init.check_loaded(missing_keys=[], unexpected_keys=["x"], allow_unexpected=False)
    with pytest.raises(fast_init.FastInitError):
        fast_init.check_loaded(Result(missing_keys=[], unexpected_keys=[]), record=rec)      # checked twice
    with pytest.raises(fast_init.FastInitError):
        fast_init.check_loaded(None)                                                          # nothing to check: refused, not assumed
    with pytest.raises(fast_init.FastInitError):
        fast_init.check_loaded(object())


# ----------------------------------------------------------------------------- resident
def test_run_items_accounts_for_every_item_and_journals(tmp_path):
    seeds_seen, errs = [], []
    journal = str(tmp_path / "journal.jsonl")

    def reseed(i, item):
        seeds_seen.append(1000 + i)

    def run_one(item):
        if item == "b":
            raise ValueError("bad contig")
        return {"n_res": len(item) * 10}

    class Sink:
        def write(self, s): errs.append(s)
        def flush(self): pass

    c = resident.run_items(["a", "b", "c"], run_one, before_item=reseed, journal=journal, name="design", stderr=Sink())
    assert (c.requested, c.ok, c.complete) == (3, 2, False)
    assert c.failures() == [("b", "ValueError: bad contig")]
    assert c.incomplete() == "1 of 3 items not ok: b (ValueError: bad contig)"
    assert c.fields() == {"design_items": "2/3", "design_failed": ["b"]}
    assert report.kv(**c.fields()) == "design_items=2/3 design_failed=b"
    assert seeds_seen == [1000, 1001, 1002]
    assert "ValueError: bad contig" in "".join(errs)                 # the traceback reached the log
    rows = resident.read_journal(journal)
    assert [r["status"] for r in rows] == ["ok", "failed", "ok"] and rows[0]["result"] == {"n_res": 10}
    d = c.as_dict()
    assert (d["n_items"], d["n_items_complete"], d["items_complete"]) == (3, 2, False)
    assert d["item_failed"] == [{"id": "b", "status": "failed", "reason": "ValueError: bad contig"}]
    v = report.verdict(0, None, False, incomplete=c.incomplete())
    assert v["exit_code"] == report.EXIT_FAIL and v["incomplete"] == c.incomplete()


def test_run_items_complete_and_stop_on_error(tmp_path):
    c = resident.run_items([3, 1, 2], lambda x: None, item_id=lambda i, x: f"design{x}")
    assert c.complete and c.incomplete() is None and c.fields() == {"resident_items": "3/3", "resident_failed": None}
    assert report.kv(**c.fields()) == "resident_items=3/3 resident_failed=none"

    def boom(x):
        if x == 1:
            raise RuntimeError("CUDA error")
    c2 = resident.run_items([0, 1, 2, 3], boom, stop_on_error=True, stderr=open(os.devnull, "w"))
    assert [r["status"] for r in c2.rows] == ["ok", "failed", "not_run", "not_run"]
    assert c2.incomplete().startswith("3 of 4 items not ok: 1 (RuntimeError: CUDA error); 2 (not_run)")
    assert resident.Census().incomplete() == "no items ran"


def test_run_items_interrupt_is_journaled_and_propagates(tmp_path):
    journal = str(tmp_path / "j.jsonl")

    def run_one(x):
        if x == 1:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        resident.run_items([0, 1, 2], run_one, journal=journal)
    rows = resident.read_journal(journal)
    assert [r["status"] for r in rows] == ["ok", "interrupted"]
    with open(journal, "a") as fh:
        fh.write('{"i": 2, "id": "2", "sta')                            # a torn line from a killed process
    assert resident.read_journal(journal)[-1]["status"] == "interrupted"


def test_run_items_after_item_failure_is_the_items_failure(tmp_path):
    journal = str(tmp_path / "j.jsonl")

    def after(i, item, row):
        if item == 1:
            raise OSError("rename failed")
    c = resident.run_items([0, 1, 2], lambda x: {"v": x}, after_item=after, journal=journal, stderr=open(os.devnull, "w"))
    assert [r["status"] for r in c.rows] == ["ok", "failed", "ok"]
    assert c.rows[1]["reason"] == "after_item: OSError: rename failed" and "result" not in c.rows[1]
    assert [r["status"] for r in resident.read_journal(journal)] == ["ok", "failed", "ok"]     # journaled as failed, never ok-then-raise


def test_package_import_is_lazy():
    r = subprocess.run([sys.executable, "-c",
                        "import sys; import opt_core.host_cache as h; "
                        "assert not [m for m in sys.modules if m.startswith('opt_core.host_cache.')], list(sys.modules); "
                        "h.resident; assert 'opt_core.host_cache.resident' in sys.modules"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": CORE_DIR})
    assert r.returncode == 0, r.stderr


def test_host_cache_imports_nothing_heavy():
    r = subprocess.run([sys.executable, "-c",
                        "import sys; from opt_core.host_cache import bg_writer, fast_init, resident; "
                        "bad = [m for m in ('torch', 'jax', 'numpy', 'triton') if m in sys.modules]; "
                        "print(bad); sys.exit(1 if bad else 0)"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": CORE_DIR})
    assert r.returncode == 0, r.stdout + r.stderr
