"""Background output writer: per-item output files are written by worker threads or processes while the device computes the next item.

Contract. A :class:`BackgroundWriter` takes the kit's own write call — ``submit(fn, *args)`` with ``fn`` the stock writer (e.g. a PDB /
mmCIF formatter) and ``args`` a snapshot of what it writes — or already-rendered bytes (``write_bytes(path, data)``), and runs it in
``workers`` background workers of the chosen ``mode``:

    off      the call runs inline, in the caller, before ``submit`` returns (stock behaviour; the census still counts it)
    thread   a thread pool in this process (rendered bytes / GIL-releasing I/O)
    fork     a process pool forked lazily at the first submit; workers never touch the device — the parent keeps every statement but
             the write itself; ``fn`` is a module-level function
    spawn    the same with freshly spawned interpreters (for callers where fork after device initialisation is not acceptable);
             ``fn``'s module must then be importable in a fresh interpreter

Snapshot rule (every background mode): ``submit`` pickles ``(fn, args, kwargs)`` BEFORE it returns, so the worker writes the values as
they were at submit time even when the caller reuses its buffers for the next item; an argument that cannot be pickled is a usage error
raised by ``submit`` itself, never a later surprise. ``write_bytes`` copies its ``bytes`` the same way.

The bytes written are the writer's own: the same function on the same values, so files are byte-identical to the inline call. Order of
completion is free; order of submission is recorded. ``join()`` is mandatory and idempotent: it waits for every pending write, closes the
pool, and returns the census; the first worker exception is re-raised by ``join()`` AFTER the census is complete (``failed`` counts every
one, ``errors`` names them). Used as a context manager the writer joins on exit, on the error path too. A writer garbage-collected or
alive at interpreter exit with writes still pending is a kit defect the guard makes loud, not lossy: the exit guard drains the pending
writes itself (the files land), prints ONE named line to stderr (``on_unjoined``; default ``[opt_core.host_cache] BG-WRITER NOT JOINED
name=… pending=N … failed=F``), and when any of those writes failed or could not be drained it forces exit status
:data:`EXIT_UNJOINED_LOSS` — a lost write never leaves a zero exit code behind.

Evidence. ``fields()`` → ``{"bg_writer": "<mode>:<workers>", "bg_written": "<written>/<submitted>"}`` for the kit's ACTIVE / summary
line via :func:`opt_core.report.kv`; ``census()`` is the manifest block.
"""
from __future__ import annotations

import atexit
import itertools
import os
import pickle
import sys
import threading
import weakref
from typing import Any, Callable, Optional

MODES = ("off", "thread", "fork", "spawn")
UNJOINED_LINE = "[opt_core.host_cache] BG-WRITER NOT JOINED name={name} mode={mode} pending={pending} submitted={submitted} written={written} failed={failed}"
EXIT_UNJOINED_LOSS = 70                     # exit status forced by the exit guard when an unjoined writer lost a write
GUARD_TIMEOUT_S = 120.0                     # per pending write, exit guard only: a hang becomes a counted loss, never a stuck exit

_LIVE = weakref.WeakSet()          # writers with a pool; the atexit guard names any that still hold pending writes


class BackgroundWriterError(RuntimeError):
    """A background write failed; raised by ``join()`` after the census is complete (``.census`` carries it)."""

    def __init__(self, message: str, census: dict):
        super().__init__(message)
        self.census = census


def _call_pickled(payload: bytes) -> None:
    fn, args, kwargs = pickle.loads(payload)
    fn(*args, **kwargs)


_TMP_SEQ = itertools.count()


def _write_bytes(path: str, data: bytes, makedirs: bool, seq: int) -> None:
    """Worker-side body of ``write_bytes``: ``<path>.part<pid>.<seq>`` then ``os.replace`` — a reader never sees a torn file and two
    writes to one path never share a temporary name."""
    if makedirs:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.part{os.getpid()}.{seq}"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


class BackgroundWriter:
    """See the module contract. ``max_pending`` bounds memory: when that many results are outstanding the older half is waited for
    inside ``submit`` (the submitting thread does the waiting, so back-pressure is visible in the item wall, never a silent queue)."""

    def __init__(self, mode: str = "off", workers: int = 1, *, name: str = "bg_writer", max_pending: int = 512,
                 on_unjoined: Optional[Callable[[dict], None]] = None):
        if mode not in MODES:
            raise ValueError(f"unknown bg_writer mode {mode!r} (expected {'|'.join(MODES)})")
        self.mode = mode
        self.workers = max(int(workers), 1) if mode != "off" else 0
        self.name = name
        self.max_pending = max(int(max_pending), 2)
        self.on_unjoined = on_unjoined
        self._pool = None
        self._pending = []                 # (seq, path_or_label, async result | future)
        self._lock = threading.Lock()
        self._submitted = 0
        self._written = 0
        self._failed = 0
        self._errors = []                  # (seq, label, "Type: message")
        self._joined = False
        self._guarded = False               # the unjoined guard ran (exit or collection): one line, once

    # ------------------------------------------------------------------ submit side
    def submit(self, fn: Callable[..., Any], *args, label: Optional[str] = None, **kwargs) -> None:
        """Schedule ``fn(*args, **kwargs)``. ``label`` names the write in the census errors (default: the first argument, usually the path)."""
        if self._joined:
            raise RuntimeError(f"bg_writer {self.name!r}: submit after join")
        label = str(label if label is not None else (args[0] if args else getattr(fn, "__name__", "write")))
        with self._lock:
            seq = self._submitted
            self._submitted += 1
        if self.mode == "off":
            try:
                fn(*args, **kwargs)
            except Exception as e:                                     # inline: count, name, re-raise now (stock raises here too)
                self._record_failure(seq, label, e)
                raise
            with self._lock:
                self._written += 1
            return
        try:                                                            # the snapshot: values as of now, whatever the caller reuses next
            payload = pickle.dumps((fn, args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            with self._lock:
                self._submitted -= 1
            raise TypeError(f"bg_writer {self.name!r}: cannot snapshot the arguments of write #{seq} {label}: {type(e).__name__}: {e}") from e
        pool = self._ensure_pool()
        if self.mode == "thread":
            handle = pool.submit(_call_pickled, payload)
        else:
            handle = pool.apply_async(_call_pickled, (payload,))
        self._pending.append((seq, label, handle))
        if len(self._pending) >= self.max_pending:
            half = len(self._pending) // 2
            self._drain(self._pending[:half])
            self._pending = self._pending[half:]

    def write_bytes(self, path: str, data: bytes, *, makedirs: bool = False) -> None:
        """Schedule an atomic write of already-rendered ``data`` to ``path`` (the common case: the kit renders text on the host, the
        worker only writes)."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("write_bytes takes bytes; render text in the caller (data.encode())")
        self.submit(_write_bytes, str(path), bytes(data), bool(makedirs), next(_TMP_SEQ), label=str(path))

    # ------------------------------------------------------------------ join side
    def join(self) -> dict:
        """Wait for every pending write, close the pool, return the census; re-raise the first worker failure afterwards."""
        if not self._joined:
            try:
                self._drain(self._pending)
                self._pending = []
            finally:
                self._close_pool()
                self._joined = True
        census = self.census()
        if self._errors:
            seq, label, reason = self._errors[0]
            raise BackgroundWriterError(f"bg_writer {self.name!r}: {len(self._errors)} of {census['submitted']} writes failed; "
                                        f"first: #{seq} {label}: {reason}", census)
        return census

    close = join

    def census(self) -> dict:
        with self._lock:
            return {"name": self.name, "mode": self.mode, "workers": self.workers, "submitted": self._submitted,
                    "written": self._written, "failed": self._failed, "pending": len(self._pending),
                    "complete": self._joined and self._failed == 0 and self._written == self._submitted,
                    "errors": [f"#{s} {l}: {r}" for s, l, r in self._errors]}

    def fields(self) -> dict:
        """The evidence fields for the kit's line: ``bg_writer=<mode>:<workers> bg_written=<written>/<submitted>``."""
        c = self.census()
        return {"bg_writer": f"{c['mode']}:{c['workers']}", "bg_written": f"{c['written']}/{c['submitted']}"}

    def __enter__(self) -> "BackgroundWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.join()
            return False
        try:                                                           # error path: still wait for what was handed out, then let
            self.join()                                                # the caller's own exception propagate
        except BackgroundWriterError:
            pass
        return False

    def __del__(self):
        try:
            if self._pending and not self._joined and not self._guarded:
                self._guard()
        except Exception:
            pass

    # ------------------------------------------------------------------ internals
    def _ensure_pool(self):
        if self._pool is None:
            sys.stdout.flush(); sys.stderr.flush()                     # a forked child must not re-emit buffered parent output
            if self.mode == "thread":
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=self.name)
            else:
                import multiprocessing as mp
                self._pool = mp.get_context(self.mode).Pool(self.workers)
            _LIVE.add(self)
            _register_exit_guard()                                     # after the pool exists: atexit is LIFO, so the guard runs
        return self._pool                                              # before the pool machinery's own exit finalizers

    def _drain(self, entries, timeout: Optional[float] = None) -> None:
        """Wait for each entry; ``timeout`` (seconds per entry, used by the exit guard only) turns a hang into a counted failure."""
        for seq, label, handle in entries:
            try:
                handle.result(timeout=timeout) if self.mode == "thread" else handle.get(timeout=timeout)
            except Exception as e:                                     # incl. the timeout types of both pool kinds
                self._record_failure(seq, label, e)
            else:
                with self._lock:
                    self._written += 1

    def _record_failure(self, seq: int, label: str, e: BaseException) -> None:
        with self._lock:
            self._failed += 1
            self._errors.append((seq, label, f"{type(e).__name__}: {e}"))

    def _close_pool(self) -> None:
        pool, self._pool = self._pool, None
        if pool is None:
            return
        if self.mode == "thread":
            pool.shutdown(wait=True)
        else:
            pool.close(); pool.join()

    def _guard(self) -> dict:
        """The unjoined guard: drain what is pending (the writes land), close the pool, print ONE line, return the census. Runs once."""
        self._guarded = True
        c0 = self.census()                                             # pending as found, for the line
        try:
            self._drain(self._pending, timeout=GUARD_TIMEOUT_S)
            self._pending = []
            self._close_pool()
        except Exception as e:                                         # undrainable (pool gone at shutdown): every pending write is a loss
            for seq, label, _ in self._pending:
                self._record_failure(seq, label, e)
            self._pending = []
        self._joined = True
        c = dict(self.census(), pending=c0["pending"])
        if self.on_unjoined is not None:
            self.on_unjoined(c)
        else:
            print(UNJOINED_LINE.format(**c), file=sys.stderr, flush=True)
        return c


_GUARD_REGISTERED = False


def _register_exit_guard() -> None:
    global _GUARD_REGISTERED
    if not _GUARD_REGISTERED:
        atexit.register(_exit_guard)
        _GUARD_REGISTERED = True


def _exit_guard() -> None:
    lost = 0
    for w in list(_LIVE):
        if w._pending and not w._joined and not w._guarded:
            lost += w._guard()["failed"]
    if lost:
        sys.stderr.flush()
        os._exit(EXIT_UNJOINED_LOSS)
