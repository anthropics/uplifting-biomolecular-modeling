"""L7 `-precompile N`: the STREAMING precompile — the first design is never gated on other lengths' compiles.

Preparing EVERY distinct input length on N threads and joining them all before the design loop would make a cold process write its first output
only after the slowest of the concurrent compiles (two lengths compiled together also slow each other on a small host). Instead the FIRST length
of the loop's order is prepared first and ALONE; the remaining lengths are submitted to the N background threads only once the
first program is ready; the loop starts at once and each design waits only on ITS OWN length (``wait(L)``) — design 1 on one compile (or one store
load), later designs usually on nothing (their compile overlapped the earlier designs' forwards). Output order is unchanged (the loop's order); a
failure inside a background prepare surfaces as that length's exception from ``wait`` (the loop records the design as failed, by name) — never a hang.
``--precompile N`` keeps its meaning as the most background prepares in flight; ``--precompile 0`` = the stock lazy jit inside each design.

    sched = Streaming(prepare, threads=6)          # prepare(item) -> result, item = (key, payload)
    sched.start([(400, a), (800, b), (1200, c)])   # loop order; returns at once
    for key, ... in loop:  sched.wait(key)         # blocks on that key only; re-raises that key's prepare exception
    sched.drain()  -> [(key, seconds, result-or-None, exc-or-None), ...] ready since the last drain (the driver emits its records from these)
    sched.finish() -> {"n": .., "threads": .., "dt_first": .., "dt_all": .., "failed": [...]}   (waits for the stragglers)
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

IMPL = "streaming_precompile"


def host_cpus() -> int:
    """The CPUs this process may use: the cgroup CPU quota when one is set (an 8-vCPU container on a larger node reports its node's CPUs through
    os.cpu_count() and even through the affinity mask), else the affinity set, else os.cpu_count()."""
    import math as _math, os as _os
    try:
        n = max(1, len(_os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        n = max(1, _os.cpu_count() or 1)
    for path, form in (("/sys/fs/cgroup/cpu.max", "v2"), ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "v1")):
        try:
            txt = open(path).read().split()
            if form == "v2" and txt and txt[0] != "max":
                n = min(n, max(1, _math.ceil(int(txt[0]) / int(txt[1]))))
            elif form == "v1" and txt and int(txt[0]) > 0:
                period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read().split()[0])
                n = min(n, max(1, _math.ceil(int(txt[0]) / period)))
            break
        except (OSError, ValueError, IndexError, ZeroDivisionError):
            continue
    return n


HOLD_TIMEOUT_S = 120.0          # the background lengths start when the loop releases them (its first design written) or this long after the first program is ready


def background_threads(requested: int, cpus: Optional[int] = None) -> int:
    """The background compile concurrency for the lengths after the first: min(requested, max(1, cpus - 3)). An XLA compile keeps a few host threads
    busy (~3 GB host RAM each) but several together still finish sooner than one after another (8 vCPUs, five lengths: 327 s five at once vs ~108 s
    each alone), so the cap only keeps a couple of CPUs for the running designs; the FIRST length is always prepared alone and the background lengths
    are held until the loop's first design is written (Streaming.release)."""
    cpus = int(cpus if cpus is not None else host_cpus())
    return max(1, min(int(requested or 1), max(1, cpus - 3)))


class Streaming:
    def __init__(self, prepare: Callable[[Tuple[Any, Any]], Any], threads: int = 1):
        self.prepare = prepare
        self.threads = max(1, int(threads or 1))
        self.t0 = None                                   # start() time
        self.order: List[Any] = []
        self.futures: Dict[Any, Future] = {}
        self.ready_at: Dict[Any, float] = {}             # key -> seconds from start() to ready (ok or failed)
        self.began_at: Dict[Any, float] = {}
        self._ready: List[tuple] = []                    # undrained (key, dt, result, exc)
        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._first_done = threading.Event()
        self._rest: List[Tuple[Any, Any]] = []
        self._feeder: Optional[threading.Thread] = None
        self._closed = False
        self._released = threading.Event()             # set by the loop once its first design is written (release()), or HOLD_TIMEOUT_S after the first program is ready
        self.hold_s = None                             # seconds the background lengths were held after the first program was ready

    # ---- submission
    def start(self, items: Sequence[Tuple[Any, Any]]) -> None:
        """``items`` = [(key, payload), ...] in the LOOP's order (duplicates of a key are ignored). Returns immediately."""
        seen, todo = set(), []
        for key, payload in items:
            if key in seen: continue
            seen.add(key); todo.append((key, payload))
        self.t0 = time.time(); self.order = [todo[0][0]] if todo else []   # add_rest() appends the later keys in loop order
        if not todo:
            self._first_done.set(); self._closed = True; return
        self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="precompile")
        first = todo[0]
        self.futures[first[0]] = self._submit(first, first_alone=True)   # the first length starts compiling NOW — before the caller has even featurized the other lengths
        self._feeder = threading.Thread(target=self._feed_rest, name="precompile-feeder", daemon=True); self._feeder.start()
        self.add_rest(todo[1:])

    def add_rest(self, items: Sequence[Tuple[Any, Any]]) -> None:
        """More (key, payload) items in loop order, after start() (the driver featurizes them while the first length compiles); they are submitted to
        the threads only once the first program is ready. close() (or finish()) says no more will come."""
        with self._lock:
            for key, payload in items:
                if key in self.futures: continue
                self.futures[key] = Future(); self.order.append(key)   # a placeholder: wait(key) blocks until the feeder submits it and it completes
                self._rest.append((key, payload))

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _submit(self, item, first_alone=False) -> Future:
        key = item[0]; self.began_at[key] = time.time() - self.t0
        fut = self._pool.submit(self._run, item)
        return fut

    def _run(self, item):
        key = item[0]; t = time.time()
        try:
            res = self.prepare(item); exc = None
        except BaseException as e:                                     # recorded and re-raised through the future (wait(key) raises it in the loop)
            res, exc = None, e
        dt = time.time() - t
        with self._lock:
            self.ready_at[key] = time.time() - self.t0; self._ready.append((key, dt, res, exc))
        if key == self.order[0]:
            self._first_done.set()
        if exc is not None:
            raise exc
        return res

    def release(self) -> None:
        """The loop's first design is written: the background lengths may start (they never compete with the first design's forward and output)."""
        self._released.set()

    def _feed_rest(self):
        self._first_done.wait()                                        # the first length is prepared ALONE: nothing else is submitted until it is ready (ok or failed)
        t_hold = time.time(); self._released.wait(HOLD_TIMEOUT_S); self.hold_s = time.time() - t_hold   # ... and until the loop has written its first design (or the hold timeout)
        fed = 0
        while True:
            with self._lock:
                pending = self._rest[fed:]; closed = self._closed
            if not pending:
                if closed: return
                time.sleep(0.05); continue
            for item in pending:
                key = item[0]; placeholder = self.futures[key]
                real = self._submit(item)
                def _relay(f, ph=placeholder):
                    try: ph.set_result(f.result())
                    except BaseException as e: ph.set_exception(e)
                real.add_done_callback(_relay)
                fed += 1

    # ---- the loop's side
    def __contains__(self, key) -> bool:
        return key in self.futures

    def wait(self, key, timeout: Optional[float] = None):
        """Block until ``key``'s program is ready; returns (result, seconds waited here). Re-raises that key's prepare exception."""
        fut = self.futures.get(key)
        if fut is None:
            return None, 0.0
        t = time.time(); res = fut.result(timeout=timeout)
        return res, time.time() - t

    def drain(self) -> List[tuple]:
        with self._lock:
            out, self._ready = self._ready, []
        return out

    def finish(self, timeout: Optional[float] = None) -> dict:
        """Wait for every submitted prepare (the exit record); never raises for a failed key (they are listed)."""
        self.close(); self.release()
        if self._feeder is not None:
            self._feeder.join(timeout)
        failed = []
        for key in self.order:
            fut = self.futures.get(key)
            try:
                if fut is not None: fut.result(timeout=timeout)
            except BaseException as e:
                failed.append((key, repr(e)))
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        ra = [self.ready_at.get(k) for k in self.order if k in self.ready_at]
        return {"impl": IMPL, "n": len(self.order), "threads": self.threads, "order": list(self.order), "first": (self.order[0] if self.order else None),
                "dt_first": (self.ready_at.get(self.order[0]) if self.order else None), "dt_all": (max(ra) if ra else None), "failed": failed,
                "began_at": {k: round(v, 3) for k, v in self.began_at.items()}, "ready_at": {k: round(v, 3) for k, v in self.ready_at.items()}, "hold_s": self.hold_s}
