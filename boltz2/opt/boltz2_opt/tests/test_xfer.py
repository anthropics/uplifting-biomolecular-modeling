"""boltz2_opt.xfer — the featurized batch's transport for large CPU storages (a disk file, not /dev/shm), and the persistent featurizer's
deadlines (boltz2_opt.prefetch: a helper / stock loader that never delivers FAILS BY NAME, the predicting process never sleeps). CPU tests,
torch required.

  * words and the ``auto`` floor parse and refuse by name; ``shm`` registers nothing (torch's transport);
  * a tensor tree through ``ForkingPickler`` under ``file``: every non-empty CPU storage is a file in the transport directory, the rebuilt
    tensors equal the sent ones dtype / shape / stride / storage offset / bytes, the files are gone once read;
  * ``auto`` splits by size: a storage under the floor takes torch's reducer, one at or above it a file (the module's own counts say which);
  * ACROSS processes, with torch's shared-memory path made to fail above a small bound the way a full /dev/shm fails (``unable to allocate
    shared memory … No space left on device``): under ``auto`` with the floor at that bound a batch carrying a storage BIGGER than the bound
    arrives whole (the big storage never touches shm; the small one still does), under ``shm`` the same batch does not arrive and the parent's
    bounded wait says so — the shared-memory exhaustion hang, reproduced on a CPU;
  * the deadlines: ``Helper.result`` on a live helper that never answers raises ``helper_timeout``; an input that fell back to the stock
    loader whose worker never delivers raises ``[boltz2-opt prefetch] FAILED by name …`` within ``BOLTZ_OPT_LOADER_TIMEOUT_S``.
"""
import multiprocessing as mp
import os
import pickle
import time

import pytest

torch = pytest.importorskip("torch")

from .. import prefetch as PF  # noqa: E402
from .. import xfer as XF      # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    XF.reset_for_tests()
    for k in (XF.ENV_WORD, XF.ENV_MIN_MIB, XF.ENV_DIR, PF.HELPER_TIMEOUT_ENV, PF.LOADER_TIMEOUT_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(XF.ENV_DIR, str(tmp_path / "xfer"))
    yield
    XF.reset_for_tests()


def _dumps(obj) -> bytes:
    from multiprocessing.reduction import ForkingPickler
    return bytes(ForkingPickler.dumps(obj))


def _tree():
    base = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    return {"f32_view": base[:, 1:, ::2],                          # non-contiguous, storage offset 4, strides (12, 4, 2)
            "i64": torch.arange(10_000, dtype=torch.int64).view(100, 100),
            "bool": torch.tensor([True, False, True, True]),
            "u8": torch.full((7,), 200, dtype=torch.uint8),
            "empty": torch.empty(0, dtype=torch.float32),
            "nested": [torch.ones(3, dtype=torch.float64) * 2.5, {"k": torch.zeros(2, 2, dtype=torch.int32)}],
            "plain": {"name": "rec", "n": 3}}


def _same(x, y, path="t"):
    if isinstance(x, dict):
        assert isinstance(y, dict) and set(x) == set(y), path
        for k in x:
            _same(x[k], y[k], f"{path}.{k}")
    elif isinstance(x, (list, tuple)):
        assert type(x) is type(y) and len(x) == len(y), path
        for i, (a, b) in enumerate(zip(x, y)):
            _same(a, b, f"{path}[{i}]")
    elif isinstance(x, torch.Tensor):
        assert isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape, path
        assert x.stride() == y.stride() and x.storage_offset() == y.storage_offset(), (path, x.stride(), y.stride(), x.storage_offset(), y.storage_offset())
        assert torch.equal(x, y), path
    else:
        assert x == y, path


def test_words_floor_and_directory(monkeypatch, tmp_path):
    assert XF.word({}) == "auto" and XF.word({XF.ENV_WORD: " FILE "}) == "file" and XF.word({XF.ENV_WORD: "shm"}) == "shm"
    with pytest.raises(XF.Refused):
        XF.word({XF.ENV_WORD: "disk"})
    assert XF.min_bytes({}) == XF.MIN_BYTES_DEFAULT == 4 << 30 and XF.min_bytes({XF.ENV_MIN_MIB: "16"}) == 16 << 20
    with pytest.raises(XF.Refused):
        XF.min_bytes({XF.ENV_MIN_MIB: "lots"})
    assert XF.xfer_dir({XF.ENV_DIR: str(tmp_path / "d")}) == str(tmp_path / "d")
    wd = tmp_path / "kit"; wd.mkdir()
    assert XF.xfer_dir({XF.WORKDIR_ENV: str(wd)}) == str(wd / XF.SUBDIR)                    # the launch's kit directory by default
    assert os.path.basename(XF.xfer_dir({})).startswith("bz2xfer_")                          # else the temp dir
    assert "unknown word" not in " ".join(PF.problems({PF.SWITCH: "persistent", XF.ENV_WORD: "disk"})) and PF.problems({PF.SWITCH: "persistent", XF.ENV_WORD: "disk"})   # prefetch refuses a bad transport word at apply, by name
    assert PF.problems({PF.SWITCH: "persistent"}) == []
    monkeypatch.setenv(XF.ENV_WORD, "shm")
    assert XF.install() is False and not XF.installed() and XF.describe()["word"] == "shm"   # shm: nothing registered — torch's reducer stands
    from multiprocessing.reduction import ForkingPickler
    from torch.multiprocessing import reductions as TR
    assert ForkingPickler._extra_reducers[torch.UntypedStorage] is TR.reduce_storage
    assert not XF.goes_by_file(1 << 40)


def test_file_word_round_trip_in_one_process(monkeypatch, tmp_path):
    monkeypatch.setenv(XF.ENV_WORD, "file")
    assert XF.install() is True and XF.describe()["word"] == "file" and XF.describe()["min_mib"] == 0
    tree = _tree()
    blob = _dumps(tree)
    d = tmp_path / "xfer"
    files = sorted(os.listdir(d))
    st = XF.stats()
    assert st["files_written"] == len(files) == 6 and st["delegated"] == 1, (st, files)      # six non-empty CPU storages -> six files; the empty one took torch's path
    want = sum(t.untyped_storage().nbytes() for t in (tree["f32_view"], tree["i64"], tree["bool"], tree["u8"], tree["nested"][0], tree["nested"][1]["k"]))
    assert st["bytes_written"] == want == PF.batch_bytes(tree) and all(f.startswith(f"st_{os.getpid()}_") for f in files), (st, want)
    back = pickle.loads(blob)
    _same(tree, back)
    assert os.listdir(d) == [] and XF.stats()["files_read"] == 6 and XF.stats()["bytes_read"] == want   # every file unlinked by its reader
    assert back["f32_view"].untyped_storage().nbytes() == tree["f32_view"].untyped_storage().nbytes() == 2 * 3 * 4 * 4   # the whole storage travelled, the view rebuilt over it
    assert XF.cleanup() is None and d.exists()                                                # a caller-named directory is left in place


def test_auto_floor_splits_by_size(monkeypatch, tmp_path):
    monkeypatch.setenv(XF.ENV_MIN_MIB, "1")
    assert XF.install() is True and XF.describe() == {"word": "auto", "min_mib": 1, "dir": str(tmp_path / "xfer"), "installed": True}
    small, big = torch.ones(1000, dtype=torch.float32), torch.ones((2 << 20) // 4 + 3, dtype=torch.float32)   # 4 KB / 2 MiB + 12 B
    assert not XF.goes_by_file(small.untyped_storage().nbytes()) and XF.goes_by_file(big.untyped_storage().nbytes()) and not XF.goes_by_file(1 << 30, "cuda")
    blob = _dumps({"small": small, "big": big})
    st = XF.stats()
    assert st["files_written"] == 1 and st["delegated"] == 1 and st["bytes_written"] == big.untyped_storage().nbytes() == (2 << 20) + 12
    back = pickle.loads(blob)
    assert torch.equal(back["big"], big) and torch.equal(back["small"], small)
    assert XF.stats()["files_read"] == 1                                                      # one storage came as a file, the other through torch's segment (delegated=1 above)
    keep = {k: PF.STATE[k] for k in ("installed", "feats_bytes_max", "helper_xfer")}
    PF.STATE.update(installed=True, feats_bytes_max=3 << 30, helper_xfer="auto")
    try:                                                                                       # the census rides its OWN line; the LEVER line's token set is unchanged (downstream tools parse it)
        xl = PF.xfer_line()
        assert xl.startswith("[boltz2-opt] XFER lever=prefetch word=auto min_mib=1 feats_gib_max=3.00 files_read=1 gib_read=") and " helper=auto helper_timeout_s=3600 loader_timeout_s=1800 dir=" in xl, xl
        assert "xfer" not in PF.line() and "feats_gib" not in PF.line(), PF.line()
        from .. import report as REP
        ev = {"prefetch_report": dict(PF.report())}
        lines = REP.xfer_lines(ev)
        assert len(lines) == 1 and lines[0].startswith("[boltz2-opt] XFER lever=prefetch word=auto min_mib=1 feats_gib_max=3.0 files_read=1 ") and "helper=auto helper_timeout_s=3600.0 loader_timeout_s=1800.0" in lines[0], lines
        assert REP.xfer_lines({"prefetch_report": {k: v for k, v in ev["prefetch_report"].items() if k != "xfer"}}) == [] == REP.xfer_lines({})   # a worker log without the census (an older worker log): no line
    finally:
        PF.STATE.update(keep)


# --------------------------------------------------------------------------------------------- across processes, torch's shm path failing above a bound
BOUND = 1 << 20          # the fake /dev/shm's room: a storage bigger than this cannot be placed in shared memory


def _tiny_shm(storage):
    """torch's reducer below BOUND; above it, what a full /dev/shm raises inside storage._share_fd_cpu_()."""
    from torch.multiprocessing import reductions as TR
    if storage.nbytes() > BOUND:
        raise RuntimeError(f"unable to allocate shared memory(shm) for file </torch_{os.getpid()}_0_0>: No space left on device (28)")
    return TR.reduce_storage(storage)


def _child_send(q, word, min_mib, xdir):
    torch.set_num_threads(1)                                              # a forked sender runs ATen single-threaded, as torch's worker loop and the kit's helper do (a parent's OpenMP pool does not survive fork)
    os.environ[XF.ENV_WORD] = word; os.environ[XF.ENV_MIN_MIB] = str(min_mib); os.environ[XF.ENV_DIR] = xdir
    XF.reset_for_tests(); XF.install()
    from multiprocessing.reduction import ForkingPickler
    if word == "shm":
        ForkingPickler.register(torch.UntypedStorage, _tiny_shm)         # torch's path IS the transport: the bound bites in the queue feeder
    else:
        XF._STATE["orig"] = _tiny_shm                                     # auto: torch's path (bounded) below the floor, a file from the floor up
    big = torch.arange((8 << 20) // 8, dtype=torch.int64)                 # 8 MiB > BOUND
    q.put({"big": big, "small": torch.arange(64, dtype=torch.int32), "tag": word})
    time.sleep(30)                                                        # a loader worker stays alive after its feeder dropped the payload — the parent must not wait on it for ever


@pytest.mark.parametrize("word", ["auto", "shm"])
def test_storage_bigger_than_the_shm_bound_across_processes(word, tmp_path, monkeypatch):
    monkeypatch.setenv(XF.ENV_MIN_MIB, "1")
    XF.install()                                                          # the receiving side (rebuild_storage_file) — as prefetch.apply installs it in the predicting process
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child_send, args=(q, word, 1, str(tmp_path / "xfer")), daemon=True)
    p.start()
    try:
        t0 = time.monotonic()
        if word == "auto":
            got = q.get(timeout=20)
            assert got["tag"] == "auto" and torch.equal(got["big"], torch.arange((8 << 20) // 8, dtype=torch.int64)) and got["small"].tolist() == list(range(64))
            assert XF.stats()["files_read"] == 1 and os.listdir(tmp_path / "xfer") == []       # the big storage came as a file (one file read, gone once read), the small one through (the fake, bounded) shm path
        else:
            import queue as _queue
            with pytest.raises(_queue.Empty):                                                 # the feeder raised 'No space left on device' and dropped the batch: nothing arrives —
                q.get(timeout=5)                                                              # a BOUNDED wait names that (prefetch: torch's DataLoader(timeout=) -> FAILED by name)
            assert p.is_alive() and time.monotonic() - t0 < 20                                # … while the sending process idles on, the large-input picture
    finally:
        p.kill(); p.join(5)


# --------------------------------------------------------------------------------------------- the deadlines, by name
def test_helper_that_never_answers_is_helper_timeout():
    a, b = mp.Pipe()
    h = PF.Helper.__new__(PF.Helper)
    h.conn, h.pid, h.alive = a, os.getpid(), (lambda: True)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match=r"^helper_timeout:1s$"):
        h.result(seq=1, poll_s=0.2, timeout_s=1.0)
    assert time.monotonic() - t0 < 5
    h.alive = lambda: False                                                                   # a dead helper is named at once, deadline or not
    with pytest.raises(RuntimeError, match=r"^helper_died:gone$"):
        h.result(seq=1, poll_s=0.1, timeout_s=60.0)
    b.close(); a.close()


def test_deadline_seconds_from_the_environment(monkeypatch):
    assert PF.helper_timeout_s() == PF.HELPER_TIMEOUT_S == 3600.0 and PF.loader_timeout_s() == PF.LOADER_TIMEOUT_S == 1800.0
    monkeypatch.setenv(PF.LOADER_TIMEOUT_ENV, "2.5"); monkeypatch.setenv(PF.HELPER_TIMEOUT_ENV, "nope")
    assert PF.loader_timeout_s() == 2.5 and PF.helper_timeout_s() == 3600.0


class _NeverDelivers(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, i):
        time.sleep(60)                                                                        # the worker is alive and delivers nothing (its feeder dropped the payload)
        return {"i": i}


class _DeadHelper:
    pid = 0

    def alive(self):
        return True

    def submit(self, kwargs, base_seed, num_workers):
        return 1

    def result(self, seq, **_):
        raise RuntimeError("helper_died:exitcode=-9")


def test_fallback_loader_that_never_delivers_fails_by_name(monkeypatch):
    monkeypatch.setenv(PF.LOADER_TIMEOUT_ENV, "2")
    keep = {k: PF.STATE[k] for k in ("helper", "refused", "fallback", "fallback_by", "base_seed_draws", "pin")}
    PF.STATE.update(helper=_DeadHelper(), refused=None, pin=False)
    try:
        loader = PF.PrefetchLoader(_NeverDelivers(), (lambda xs: xs[0]), {}, num_workers=1, pin=False)
        it = iter(loader)                                                                     # draws base_seed, submits to the (dead) helper
        t0 = time.monotonic()
        with pytest.raises(RuntimeError) as ei:
            next(it)                                                                          # helper_died -> FALLBACK by name -> the stock loader's worker never delivers -> FAILED by name
        msg = str(ei.value)
        assert msg.startswith("[boltz2-opt prefetch] FAILED by name: the stock DataLoader this input fell back to (fallback_by=helper_died:exitcode=-9) delivered no batch"), msg
        assert "BOLTZ_OPT_LOADER_TIMEOUT_S=2s" in msg and time.monotonic() - t0 < 30, (msg, time.monotonic() - t0)
        assert PF.STATE["fallback_by"].get("helper_died:exitcode=-9") == 1
    finally:
        it.fallback_iter = None
        PF.STATE.update(keep)
