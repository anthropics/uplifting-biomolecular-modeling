"""Outputs off the critical path: one bulk device→host transfer of a result tree, pinned staging buffers, writing in the background.

Contract. Three primitives, framework code imported inside the call (``import opt_core.host.outputs`` is standard library only):

:func:`to_host` ``(tree)`` — every device array in a nested dict/list/tuple tree copied to host in ONE pass: torch tensors through
non-blocking copies on the current stream and one synchronisation per CUDA device at the end (into pinned buffers from a
:class:`PinnedPool` when one is given; tensors on other device types are copied blocking, leaf by leaf), jax arrays through one
``jax.device_get`` over all of them, numpy and host values untouched. Containers come back as the same types (a dict subclass that
cannot be rebuilt from key-value pairs comes back as a plain ``dict``). It replaces the per-leaf lazy
transfer (a ``.cpu()`` / ``float()`` / ``np.asarray`` per confidence field inside post-processing, each a device synchronisation) with
one. The values are the same bytes: no numerics move.

:class:`PinnedPool` ``(max_bytes)`` — page-locked host buffers re-used across items, keyed by byte size, so a steady-state loop
allocates none; a request beyond ``max_bytes`` is served UNPINNED and counted (``FALLBACK reason=over_budget`` — pageable memory
copies synchronously; correct, slower, named); no CUDA device is :class:`~opt_core.host.events.HostRefused` ``reason=no_cuda``,
torch missing ``reason=missing:torch`` — the adapter logs the line and uses plain ``to_host`` without a pool as its recorded choice.

:class:`AsyncWriter` ``(write, workers, max_pending, mode)`` — ``write(item)`` runs in ``workers`` background workers of ``mode``:
``thread`` (a thread pool in this process — rendered bytes, GIL-releasing I/O), ``fork`` (a process pool forked lazily at the first
submit; ``write`` is a module-level function and the workers never touch the device) or ``spawn`` (the same with fresh interpreters, for
callers where forking after device initialisation is not acceptable; ``write``'s module must import in a fresh interpreter). OWNERSHIP
RULE: ``submit(item)`` takes an owned copy of the item HERE, in the caller's thread, before it returns — :func:`snapshot` in thread mode,
the pickle of ``(write, item)`` in the process modes — so a producer re-using one output buffer per item (the normal case) never gets item
k+1's bytes under item k's name, and an item that cannot be copied is a ``TypeError`` raised by ``submit`` itself, never a later surprise;
``copy_on_submit=False`` (thread mode only) is the by-reference hand-off for callers that give the item up. Back-pressure: at most
``max_pending`` writes outstanding, the producer blocks in ``submit`` rather than buffering a run's worth of outputs. ``drain()`` waits for
every write and raises the first failure AFTER all have finished (every failure is in ``failures`` with its item — its label in the process
modes —: the run's accounting names each unwritten file); leaving the ``with`` block drains and closes. EXIT GUARD: a writer left undrained
at interpreter exit (or garbage collection) with writes pending or failed is a kit defect made loud, not lossy — the guard drains the pending
writes itself (the files land; a write still running after :data:`GUARD_TIMEOUT_S` is counted failed), prints ONE record line
``[<tag>] HOST host.outputs TALLY reason=undrained_at_exit part=writer name=… mode=… pending_at_exit=N submitted=… written=… failed=F`` to
stderr, and when ``F > 0`` forces exit status :data:`EXIT_UNDRAINED_LOSS` (70) at interpreter exit — also for a writer garbage-collected
undrained earlier in the run: a lost write never leaves a zero exit code behind. The forced status skips atexit hooks registered before the
guard; the kit's own EXIT tallies (:func:`opt_core.report.register_exit_tally`) among them are printed by the guard first, so the run
record stays whole (other libraries' atexit hooks registered earlier do not run — the process is failing by name).

Evidence: each object has ``evidence()`` / ``active_line(tag, name)`` (``[<tag>] LEVER name=F6.output_overlap state=on impl=host.outputs
origin=core part=pool|writer ...``) and ``tally()`` / ``tally_line(tag)`` (``[<tag>] HOST host.outputs TALLY part=.. ...``); :func:`to_host` returns its census through the optional ``stats`` dict (leaves, bytes, pinned, unpinned).
"""
from __future__ import annotations

import atexit
import collections
import copy
import os
import pickle
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import report
from .events import HostEvent, HostRefused, require

LEVER = "host.outputs"
MODES = ("thread", "fork", "spawn")       # AsyncWriter worker kinds
EXIT_UNDRAINED_LOSS = 70                   # exit status the exit guard forces when a writer left undrained lost a write
GUARD_TIMEOUT_S = 120.0                    # per pending write, exit guard only: a hung write becomes a counted loss, never a stuck exit
GUARD_TAG = "opt_core.host"                # the tag of the guard's line until register_exit_tally(tag) names the kit's
_LIVE = weakref.WeakSet()                  # writers holding a pool: the exit guard visits every one left undrained


def _kind(x: Any) -> str:
    mod = type(x).__module__
    if mod == "torch" or mod.startswith("torch."):
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(x, torch.Tensor):
            return "torch"
    if mod.startswith("jax") or mod.startswith("jaxlib"):
        return "jax"
    return "other"


def _leaves(tree: Any, path: Tuple = ()) -> List[Tuple[Tuple, Any]]:
    if isinstance(tree, dict):
        out = []
        for k in tree:
            out.extend(_leaves(tree[k], path + (k,)))
        return out
    if isinstance(tree, (list, tuple)):
        out = []
        for i, v in enumerate(tree):
            out.extend(_leaves(v, path + (i,)))
        return out
    return [(path, tree)]


def _rebuild(tree: Any, repl: Dict[Tuple, Any], path: Tuple = ()) -> Any:
    if isinstance(tree, dict):
        items = [(k, _rebuild(v, repl, path + (k,))) for k, v in tree.items()]
        try:
            return type(tree)(items)
        except Exception:                                 # noqa: BLE001 — a dict subclass with another constructor: a plain dict
            return dict(items)
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):          # namedtuple
        return type(tree)(*[_rebuild(v, repl, path + (i,)) for i, v in enumerate(tree)])
    if isinstance(tree, (list, tuple)):
        seq = [_rebuild(v, repl, path + (i,)) for i, v in enumerate(tree)]
        return seq if isinstance(tree, list) else tuple(seq)
    return repl.get(path, tree)


def snapshot(item: Any) -> Any:
    """An owned copy of ``item`` taken NOW, in the caller's thread: containers rebuilt, numpy arrays ``.copy()``, torch tensors
    ``.detach().clone()``, ``bytearray``/``memoryview`` → ``bytes``, immutable scalars/str/bytes and jax arrays (immutable) as they are,
    anything else ``copy.deepcopy``. What :class:`AsyncWriter` hands to the background thread, so a producer that re-uses its buffers for
    the next item cannot change what is written for this one."""
    if isinstance(item, dict):
        items = [(k, snapshot(v)) for k, v in item.items()]
        try:
            return type(item)(items)
        except Exception:                                 # noqa: BLE001 — a dict subclass with another constructor: a plain dict
            return dict(items)
    if isinstance(item, tuple) and hasattr(item, "_fields"):
        return type(item)(*[snapshot(v) for v in item])
    if isinstance(item, (list, tuple)):
        seq = [snapshot(v) for v in item]
        return seq if isinstance(item, list) else tuple(seq)
    if item is None or isinstance(item, (str, bytes, int, float, bool)):
        return item
    if isinstance(item, (bytearray, memoryview)):
        return bytes(item)
    kind = _kind(item)
    if kind == "torch":
        return item.detach().clone()
    if kind == "jax":
        return item                                       # jax arrays are immutable
    mod = type(item).__module__
    if (mod == "numpy" or mod.startswith("numpy.")) and hasattr(item, "copy"):
        return item.copy()
    return copy.deepcopy(item)


class PinnedPool:
    """Re-usable page-locked host buffers keyed by byte size (torch). ``lease(nbytes)`` → a 1-D ``uint8`` pinned tensor of at least
    that size (views are the caller's); ``release(buf)`` returns it to the size bucket's free list. Page-locking, the byte budget and the
    pageable fallback are :class:`opt_core.mem.torch_hostpair.PinPool`'s (the package's one pinned host-buffer pool); over budget or a
    refused page-lock → an unpinned buffer, counted (FALLBACK ``over_budget`` | ``pin_failed``)."""

    def __init__(self, max_bytes: int, *, name: str = "pool") -> None:
        torch = require(LEVER, "torch")
        if not torch.cuda.is_available():
            raise HostRefused(LEVER, "no_cuda", part="pool")
        self._torch = torch
        self.name, self.max_bytes = str(name), int(max_bytes)
        from ..mem.torch_hostpair import PinPool          # noqa: PLC0415 — the package's ONE pinned host-buffer pool (budget, page-lock or refuse, accounting)
        self._pool = PinPool(self.max_bytes, pageable=True, lever=LEVER)   # over budget / page-lock refused -> a pageable buffer, counted
        self._free: Dict[int, List[Any]] = collections.defaultdict(list)
        self._lock = threading.Lock()
        self._t = {"leases": 0, "reuses": 0, "allocations": 0, "unpinned": 0}
        self._seen: set = set()
        self.events: List[HostEvent] = []               # first occurrence per fallback reason; the tally carries the counts

    @staticmethod
    def _bucket(nbytes: int) -> int:
        b = 1 << 12
        while b < nbytes:
            b <<= 1
        return b

    _REASON_OF_KIND = {"pin_budget_exceeded": "over_budget", "pin_alloc_failed": "pin_failed"}   # the pool's refusal kinds in this lever's FALLBACK words

    def lease(self, nbytes: int):
        size = self._bucket(max(1, int(nbytes)))
        with self._lock:
            self._t["leases"] += 1
            if self._free[size]:
                self._t["reuses"] += 1
                return self._free[size].pop()
        buf, denied = self._pool.alloc_counted((size,), self._torch.uint8, tag=self.name)   # budget reserved before the page-lock; refused -> pageable, kind named
        with self._lock:
            if denied is None:
                self._t["allocations"] += 1
            else:
                self._t["unpinned"] += 1
                reason = self._REASON_OF_KIND.get(denied, denied)
                if reason not in self._seen:                # the first occurrence per reason is an event record; the tally carries the counts
                    self._seen.add(reason)
                    self.events.append(HostEvent(LEVER, "FALLBACK", reason, {"part": "pool", "max_bytes": self.max_bytes}))
        return buf

    def release(self, buf) -> None:
        if buf.is_pinned():
            with self._lock:
                self._free[buf.numel()].append(buf)

    def evidence(self) -> dict:
        return {"lever": LEVER, "part": "pool", "pool": self.name, "max_mib": self.max_bytes >> 20}

    def active_line(self, tag: str, name: str = "F6.output_overlap", origin: str = "core") -> str:
        """The kit's ONE activation line for this lever: ``[<tag>] LEVER name=<name> state=on impl=host.outputs origin=core ...`` (``name`` = strategy id)."""
        ev = self.evidence()
        return HostEvent(LEVER, "ACTIVE", "", {k: v for k, v in ev.items() if k != "lever"}).lever_line(tag, name, origin)

    def tally(self) -> dict:
        with self._lock:
            out = dict(self._t)
        out.update(name=self.name, pinned_mib=self._pool.bytes_now >> 20)
        return out

    def tally_line(self, tag: str) -> str:
        t = self.tally()
        fields = collections.OrderedDict((k, t[k]) for k in ("name", "leases", "reuses", "allocations", "unpinned", "pinned_mib"))
        fields["part"] = "pool"
        fields.move_to_end("part", last=False)
        return HostEvent(LEVER, "TALLY", "", fields).line(tag)


def to_host(tree: Any, *, pool: Optional[PinnedPool] = None, leases: Optional[list] = None, stats: Optional[dict] = None) -> Any:
    """A copy of ``tree`` with every torch tensor / jax array replaced by its host copy (torch: a CPU tensor; jax: a numpy array),
    in one pass with one synchronisation per device. With a ``pool`` the copies land in leased pinned buffers: pass a ``leases`` list to
    receive the buffers (the returned tensors are views over them — release each with ``pool.release`` after use, e.g. at the end of
    the writer's ``write``), or omit it to get owned copies (the buffers are released here). ``stats`` (optional) receives ``leaves``,
    ``bytes``, ``pinned``, ``unpinned``, ``jax_leaves``."""
    leaves = _leaves(tree)
    repl: Dict[Tuple, Any] = {}
    census = {"leaves": 0, "bytes": 0, "pinned": 0, "unpinned": 0, "jax_leaves": 0}
    torch_leaves = [(p, x) for p, x in leaves if _kind(x) == "torch" and x.device.type != "cpu"]
    jax_leaves = [(p, x) for p, x in leaves if _kind(x) == "jax"]
    if torch_leaves:
        torch = sys.modules["torch"]
        pending = []
        for path, x in torch_leaves:
            src = x.detach().contiguous()
            nbytes = src.numel() * src.element_size()
            buf = pool.lease(nbytes) if (pool is not None and nbytes) else None
            if buf is not None:
                dst = buf[:nbytes].view(src.dtype).view(src.shape)
                census["pinned" if buf.is_pinned() else "unpinned"] += 1
            else:
                dst = torch.empty(src.shape, dtype=src.dtype)
                census["unpinned"] += 1
            dst.copy_(src, non_blocking=(src.device.type == "cuda"))   # CUDA: queued on the device's current stream, one sync below; other devices: blocking
            pending.append((path, dst, buf))
            census["leaves"] += 1
            census["bytes"] += nbytes
        for dev in {x.device for _, x in torch_leaves}:
            if dev.type == "cuda":
                torch.cuda.current_stream(dev).synchronize()
        for path, dst, buf in pending:
            if buf is not None and leases is None:
                repl[path] = dst.clone()                  # owned bytes; the pinned buffer goes straight back to the pool
                pool.release(buf)
            else:
                repl[path] = dst
                if buf is not None:
                    leases.append(buf)
    if jax_leaves:
        jax = require(LEVER, "jax")
        host_vals = jax.device_get([x for _, x in jax_leaves])
        for (path, _), hv in zip(jax_leaves, host_vals):
            repl[path] = hv
        census["jax_leaves"] = len(jax_leaves)
        census["leaves"] += len(jax_leaves)
    if stats is not None:
        stats.update(census)
    if not repl:
        return tree
    if not isinstance(tree, (dict, list, tuple)):
        return repl.get((), tree)
    return _rebuild(tree, repl)


def _call_pickled(blob: bytes) -> None:
    """Process-mode worker body: ``write(item)`` on the values pickled at submit. The result does not travel back (it may not pickle)."""
    write, item = pickle.loads(blob)
    write(item)


def _label(item: Any, seq: int) -> str:
    """The name of write #seq in a process-mode failure record: the item's path/name field or first string element, else its type."""
    if isinstance(item, dict):
        for k in ("path", "name", "file", "id"):
            if k in item:
                return "#%d %s" % (seq, item[k])
    if isinstance(item, (tuple, list)) and item and isinstance(item[0], (str, bytes, os.PathLike)):
        return "#%d %s" % (seq, item[0])
    return "#%d %s" % (seq, type(item).__name__)


def _done(handle: Any) -> bool:
    return handle.done() if hasattr(handle, "done") else handle.ready()


class AsyncWriter:
    """``write(item)`` in background workers (``mode`` thread | fork | spawn) with submit-time ownership, back-pressure, total accounting
    and the exit guard. See the module contract."""

    def __init__(self, write: Callable[[Any], Any], *, workers: int = 1, max_pending: int = 4, name: str = "writer",
                 copy_on_submit: bool = True, mode: str = "thread") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown AsyncWriter mode {mode!r} (known: {'|'.join(MODES)})")
        if workers < 1 or max_pending < 1:
            raise ValueError("workers and max_pending must be >= 1")
        if mode != "thread" and not copy_on_submit:
            raise ValueError("copy_on_submit=False (the by-reference hand-off) exists in thread mode only: a fork/spawn worker receives the "
                             "bytes pickled at submit")
        self.write, self.name, self.workers, self.max_pending, self.mode = write, str(name), int(workers), int(max_pending), mode
        self.copy_on_submit = bool(copy_on_submit)
        self._pool: Any = None                            # created at the first submit (a forked pool must not exist before it is needed)
        self._slots = threading.BoundedSemaphore(self.max_pending)
        self._lock = threading.Lock()
        self._futures: List[Tuple[Any, Any]] = []         # (item as kept — the snapshot (thread) or the label (fork/spawn) —, handle); done entries are pruned at submit
        self.failures: List[Tuple[Any, BaseException]] = []
        self._t = {"submitted": 0, "written": 0, "failed": 0}
        self._closed = False
        self._guarded = False
        self._tag = GUARD_TAG

    # ---------------------------------------------------------------------------------------------------------------- submit side
    def submit(self, item: Any) -> Any:
        """Queue ``write(item)`` and return its handle (a ``Future`` in thread mode, an ``AsyncResult`` in the process modes). The item is
        copied HERE (see the module's ownership rule): the caller may re-use its buffers for the next item as soon as this returns."""
        self._slots.acquire()                             # back-pressure: block the producer, never buffer without bound
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("AsyncWriter is closed")
                seq = self._t["submitted"]
            if self.mode == "thread":
                if self.copy_on_submit:
                    try:
                        payload = snapshot(item)
                    except Exception as e:                # noqa: BLE001 — named as the caller's usage error, at submit
                        raise TypeError(f"HOST {LEVER} writer {self.name!r}: cannot snapshot item #{seq} ({type(e).__name__}: {e}); submit "
                                        f"plain containers / arrays / bytes, or construct with copy_on_submit=False when the item is given up") from e
                else:
                    payload = item
                keep = payload
            else:
                try:
                    payload = pickle.dumps((self.write, item), protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:                    # noqa: BLE001
                    raise TypeError(f"HOST {LEVER} writer {self.name!r}: cannot pickle (write, item #{seq}) for a {self.mode} worker "
                                    f"({type(e).__name__}: {e}); `write` must be a module-level function and the item picklable") from e
                keep = _label(item, seq)
            pool = self._ensure_pool()
            with self._lock:
                if self._closed:
                    raise RuntimeError("AsyncWriter is closed")
                self._futures[:] = [(k, h) for k, h in self._futures if not _done(h)]
                self._t["submitted"] += 1
                if self.mode == "thread":
                    handle = pool.submit(self._run, payload)
                else:
                    handle = pool.apply_async(_call_pickled, (payload,), callback=lambda _r: self._record(None, None),
                                              error_callback=lambda exc, k=keep: self._record(k, exc))
                self._futures.append((keep, handle))
        except BaseException:
            self._slots.release()
            raise
        return handle

    def _run(self, item: Any) -> Any:                     # thread-mode worker body
        try:
            out = self.write(item)
            with self._lock:
                self._t["written"] += 1
            return out
        except BaseException as exc:                      # recorded with its item; re-raised by drain()
            with self._lock:
                self._t["failed"] += 1
                self.failures.append((item, exc))
            raise
        finally:
            self._slots.release()

    def _record(self, keep: Any, exc: Optional[BaseException]) -> None:   # process-mode completion, in the pool's result thread
        try:
            with self._lock:
                if exc is None:
                    self._t["written"] += 1
                else:
                    self._t["failed"] += 1
                    self.failures.append((keep, exc))
        finally:
            self._slots.release()

    def _ensure_pool(self) -> Any:
        global _GUARD_REGISTERED, _GUARD_AFTER_MP
        if self._pool is None:
            if self.mode == "thread":
                self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=f"opt-core-{self.name}")
            else:
                import multiprocessing                    # noqa: PLC0415 — process modes only
                sys.stdout.flush()
                sys.stderr.flush()                        # a forked child must not re-emit the parent's buffered output
                self._pool = multiprocessing.get_context(self.mode).Pool(self.workers)
            _LIVE.add(self)
            # atexit is LIFO: registered after the pool exists, the guard runs BEFORE the pool machinery's own exit finalizers (a process
            # pool is terminated by multiprocessing's exit function, registered when its util module is first imported — hence once more
            # after the first process pool of this interpreter)
            if not _GUARD_REGISTERED or (self.mode != "thread" and not _GUARD_AFTER_MP):
                atexit.register(_exit_guard)                              # exit tallies registered earlier would print AFTER the guard (LIFO): a forced exit re-issues them (report.forced_exit)
                _GUARD_REGISTERED = True
                _GUARD_AFTER_MP = _GUARD_AFTER_MP or self.mode != "thread"
        return self._pool

    # ---------------------------------------------------------------------------------------------------------------- drain side
    def pending(self) -> int:
        with self._lock:
            return sum(1 for _, h in self._futures if not _done(h))

    def _wait(self, handle: Any, timeout: Optional[float]) -> bool:
        """Wait for one write; False = still running after ``timeout`` (accounting of finished writes happened in the worker callback)."""
        if self.mode == "thread":
            try:
                handle.result(timeout=timeout)
            except FuturesTimeout:
                return handle.done()
            except Exception:                             # noqa: BLE001 — accounted by _run
                return True
            return True
        handle.wait(timeout)
        return handle.ready()

    def _wait_all(self, timeout: Optional[float] = None) -> int:
        """Wait until no write is outstanding (a submit racing with this is waited for too); a write still running after ``timeout``
        seconds (exit guard only) is counted failed with its item. Returns the number so counted."""
        given_up: List[Any] = []
        while True:
            with self._lock:
                entries = [(k, h) for k, h in self._futures if not _done(h) and not any(h is g for g in given_up)]
            if not entries:
                return len(given_up)
            for keep, h in entries:
                if not self._wait(h, timeout):
                    given_up.append(h)
                    with self._lock:
                        self._t["failed"] += 1
                        self.failures.append((keep, TimeoutError(f"write still running after {timeout}s at exit")))

    def drain(self) -> None:
        """Wait for every submitted write; then raise the first failure (all failures stay in ``failures``)."""
        self._wait_all()
        if self.failures:
            item, exc = self.failures[0]
            raise RuntimeError(f"HOST {LEVER} writer {self.name!r}: {len(self.failures)} write(s) failed; first item {item!r}") from exc

    def _shutdown_pool(self, *, terminate: bool = False) -> None:
        pool, self._pool = self._pool, None
        if pool is None:
            return
        if self.mode == "thread":
            pool.shutdown(wait=not terminate)
        elif terminate:
            pool.terminate()
            pool.join()
        else:
            pool.close()
            pool.join()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.drain()
        finally:
            self._shutdown_pool()

    def __enter__(self) -> "AsyncWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        try:                                              # the body failed: drain what was submitted, keep ITS exception (failures stay in .failures)
            self.close()
        except RuntimeError:
            pass

    def __del__(self) -> None:
        try:
            if self._undrained():
                self._guard()
        except Exception:                                 # noqa: BLE001 — interpreter teardown: nothing left to report with
            pass

    # ---------------------------------------------------------------------------------------------------------------- the exit guard
    def _undrained(self) -> bool:
        """Left undrained with something to account for: writes pending, or failures nobody raised."""
        if self._closed or self._guarded or self._pool is None:
            return False
        with self._lock:
            failed = self._t["failed"]
        return self.pending() > 0 or failed > 0

    def _guard(self) -> dict:
        """Runs once per writer: drain what is pending (the writes land), close the pool, print ONE record line to stderr, return the tally."""
        self._guarded = True
        found = self.pending()
        try:
            hung = self._wait_all(timeout=GUARD_TIMEOUT_S)
            self._shutdown_pool(terminate=hung > 0)
        except Exception as e:                            # noqa: BLE001 — the pool is unusable at shutdown: every unfinished write is a loss
            with self._lock:
                for keep, h in self._futures:
                    if not _done(h):
                        self._t["failed"] += 1
                        self.failures.append((keep, e))
            self._pool = None
        self._closed = True
        t = self.tally()
        fields = collections.OrderedDict([("part", "writer"), ("name", self.name), ("mode", self.mode), ("pending_at_exit", found),
                                          ("submitted", t["submitted"]), ("written", t["written"]), ("failed", t["failed"])])
        print(HostEvent(LEVER, "TALLY", "undrained_at_exit", fields).line(self._tag), file=sys.stderr, flush=True)
        if t["failed"]:
            global _LOST_BEFORE_EXIT
            with _LOST_LOCK:
                _LOST_BEFORE_EXIT += int(t["failed"])          # a writer collected before exit still forces the loss exit status (_exit_guard sums it)
        return t

    # ---------------------------------------------------------------------------------------------------------------- evidence
    def evidence(self) -> dict:
        return {"lever": LEVER, "part": "writer", "writer": self.name, "mode": self.mode, "workers": self.workers, "max_pending": self.max_pending,
                "copy_on_submit": "on" if self.copy_on_submit else "off"}

    def active_line(self, tag: str, name: str = "F6.output_overlap", origin: str = "core") -> str:
        """The kit's ONE activation line for this lever: ``[<tag>] LEVER name=<name> state=on impl=host.outputs origin=core ...`` (``name`` = strategy id)."""
        ev = self.evidence()
        return HostEvent(LEVER, "ACTIVE", "", {k: v for k, v in ev.items() if k != "lever"}).lever_line(tag, name, origin)

    def tally(self) -> dict:
        with self._lock:
            out = dict(self._t)
        out.update(name=self.name, mode=self.mode, pending=self.pending())
        return out

    def tally_line(self, tag: str) -> str:
        t = self.tally()
        fields = collections.OrderedDict([("part", "writer")] + [(k, t[k]) for k in ("name", "mode", "submitted", "written", "failed", "pending")])
        return HostEvent(LEVER, "TALLY", "", fields).line(tag)

    def register_exit_tally(self, tag: str) -> bool:
        """Register the kit's exit TALLY line for this writer (report's exit tally) and name the exit guard's line with the same tag."""
        self._tag = str(tag)
        return report.register_exit_tally(tag + ":" + LEVER + ":" + self.name, lambda: self.tally_line(tag))


_GUARD_REGISTERED = False
_LOST_BEFORE_EXIT = 0                                  # writes counted failed by a guard that ran BEFORE interpreter exit (a writer garbage-collected undrained)
_LOST_LOCK = threading.Lock()
_GUARD_AFTER_MP = False


def _exit_guard() -> None:
    """atexit: every writer left undrained accounts for itself (one line each); any lost write forces exit status EXIT_UNDRAINED_LOSS."""
    lost = 0
    for w in list(_LIVE):
        try:
            if w._undrained():
                w._guard()                                # counts its failed writes into _LOST_BEFORE_EXIT
        except Exception:                                 # noqa: BLE001 — a writer that cannot even be inspected is a loss
            lost += 1
    lost += _LOST_BEFORE_EXIT                             # + writers guarded earlier (garbage-collected undrained with a failed write)
    if lost:
        report.forced_exit(EXIT_UNDRAINED_LOSS)           # the kit's EXIT tallies atexit would have printed after this guard, the flush os._exit skips, then the status
