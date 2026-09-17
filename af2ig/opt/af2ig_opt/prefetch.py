"""prefetch — L15 ``-prefetch N``: the INPUTS of the next N designs prepared ahead of the loop on one background worker thread. Runs INSIDE the driver.

Stock (predict_pdb.py, as predict.py) handles a directory serially per design: read the PDB file, featurise it (template / single-sequence MSA
features, chain-break detection, the TensorFlow feature pipeline — host CPU work, one core), run the model on the GPU, post-process and write.
While the GPU runs design i the host is idle, and while the host featurises design i+1 the GPU is idle. This lever moves the featurisation of
the next ``depth`` inputs onto ONE worker thread that runs ahead of the loop: the loop takes each prepared input when its turn comes
(:meth:`Prefetcher.take`), usually without waiting, so a design's wall becomes ~max(model + output, load + featurise) instead of their sum.

What does NOT change (the lever is exact-class — the stock line's bytes under ``--det 1``): every input is featurised by the same code exactly
once, ONE at a time (a single worker: no two featurisations ever run concurrently, as in stock), in the loop's own order; the loop consumes
them in that order; the model call, the post-processing, the score file, the checkpoint file and every written structure are produced by the
main thread in the stock order. Only WHEN a featurisation runs moves (earlier, overlapped with the previous design's forward). A featurisation
that raises re-raises in the loop at the design's turn (``take``), where stock's would have surfaced — the driver's per-design failure
handling (message, ``failed`` record, checkpoint) is unchanged. Inputs the precompile pass (L7) already featurised are not featurised again
(``skip``). Host memory: at most ``depth`` featurised inputs wait in the queue (bounded look-ahead; tens of MB per input at 1200 residues).
GPU memory: none (features live on the host until the model call, as in stock).
Evidence: :meth:`Prefetcher.close` -> the ``prefetch`` timer record {depth, workers, queued, taken, ready, waited, skipped, wait_s, work_s,
overlap_s, errors}; ``overlap_s`` = worker seconds that did not appear as loop waiting = the host work hidden behind the GPU.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

IMPL = "thread_lookahead"


class Prefetcher:
    """One worker thread preparing ``prepare(item)`` for the items of ``order`` at most ``depth`` ahead of the consumer, in order."""

    def __init__(self, prepare: Callable, order: Sequence, depth: int = 1, skip: Optional[Callable[[object], bool]] = None, workers: int = 1, info: Optional[dict] = None):
        if depth < 1:
            raise ValueError("prefetch depth must be >= 1")
        self.prepare, self.order, self.depth = prepare, list(order), int(depth)
        self.skip = skip or (lambda item: False)
        self.workers = max(1, int(workers))                       # 1 = stock's one-at-a-time featurisation, moved earlier; > 1 is NOT offered by the driver (kept for measurement)
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="af2ig_prefetch")
        self.futures: Dict[int, Future] = {}
        self.next_submit = 0                                      # index of the next item to hand to the worker
        self.pos = 0                                              # index of the next item the loop will take
        self.lock = threading.Lock()
        self.info = dict(info or {})                              # the driver's facts about how the worker prepares (e.g. offdevice), carried into the census
        self.stats = {"impl": IMPL, "depth": self.depth, "workers": self.workers, "queued": 0, "taken": 0, "ready": 0, "waited": 0, "skipped": 0, "unscheduled": 0,
                      "errors": 0, "wait_s": 0.0, "work_s": 0.0}
        self._fill()

    # -- worker side
    def _work(self, item):
        t = time.time()
        try:
            if self.skip(item):
                return ("skipped", None, 0.0)
            return ("ok", self.prepare(item), time.time() - t)
        except BaseException as e:                                # handed to the loop at take(): raised there, where stock's featurisation error surfaces
            return ("error", e, time.time() - t)

    def _fill(self):
        with self.lock:
            self.next_submit = max(self.next_submit, self.pos)      # never prepare an item the loop is already past
            while self.next_submit < len(self.order) and self.next_submit < self.pos + self.depth:
                i = self.next_submit
                self.futures[i] = self.pool.submit(self._work, self.order[i])
                self.next_submit += 1
                self.stats["queued"] += 1

    # -- loop side
    def take(self, item) -> Tuple[Optional[object], float]:
        """(prepared, waited_s) for ``item`` — the next item of the order. Blocks until the worker has prepared it; re-raises the worker's
        exception; (None, 0.0) when the item was skipped (prepared elsewhere) or is not the scheduled next item (the caller prepares it itself)."""
        with self.lock:
            i = self.pos
            if i >= len(self.order) or self.order[i] != item:
                try:                                              # out-of-order request: realign on the item if it is ahead in the order, else let the caller prepare it
                    i = self.order.index(item, self.pos)
                except ValueError:
                    self.stats["unscheduled"] += 1
                    return None, 0.0
            self.pos = i + 1
            fut = self.futures.pop(i, None)
        self._fill()                                              # keep the worker `depth` ahead of the new position
        if fut is None:
            with self.lock:
                self.stats["unscheduled"] += 1
            return None, 0.0
        t = time.time()
        ready = fut.done()
        status, payload, work_s = fut.result()
        waited = time.time() - t
        with self.lock:
            self.stats["work_s"] += work_s
            self.stats["wait_s"] += waited
            if status == "skipped":
                self.stats["skipped"] += 1
                return None, waited
            self.stats["taken"] += 1
            self.stats["ready" if ready else "waited"] += 1
            if status == "error":
                self.stats["errors"] += 1
        if status == "error":
            raise payload
        return payload, waited

    def census(self) -> dict:
        with self.lock:
            c = dict(self.stats)
        c["wait_s"], c["work_s"] = round(c["wait_s"], 3), round(c["work_s"], 3)
        c["overlap_s"] = round(max(0.0, c["work_s"] - c["wait_s"]), 3)   # worker seconds the loop did not wait for: host work hidden behind the GPU
        c.update(self.info)
        return c

    def close(self) -> dict:
        for f in list(self.futures.values()):
            f.cancel()
        self.pool.shutdown(wait=True)
        c = self.census(); c["state"] = "on"
        return c


class OutputWriter:
    """L16 ``-overlap_output N``: ONE writer thread runs each design's output step (the driver's process_output: confidence metrics, RMSDs,
    the PDB file, the score line; then the checkpoint line) in submission order while the loop goes on to the next design's forward; at
    most ``depth`` outputs wait behind it (the loop blocks on submit beyond that: bounded host memory). The step's code, its inputs and the
    order of every file append are the loop's own; an exception in a step is handed to ``on_error(tag, exc)`` (the driver prints and records
    it as the serial loop does) and ``always(tag)`` (the checkpoint) still runs, as the serial loop records a design done either way."""

    def __init__(self, depth: int = 1, on_error: Optional[Callable] = None):
        self.depth = max(1, int(depth))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="af2ig_writer")
        self.slots = threading.Semaphore(self.depth + 1)          # the job running + `depth` waiting
        self.on_error = on_error or (lambda tag, exc: None)
        self.futures: List[Future] = []
        self.tags: set = set()
        self.lock = threading.Lock()
        self.stats = {"impl": "thread_writer", "depth": self.depth, "jobs": 0, "errors": 0, "busy_s": 0.0, "lag_s": 0.0, "blocked_s": 0.0, "max_pending": 0}

    def submit(self, tag, fn: Callable, always: Optional[Callable] = None) -> None:
        t = time.time(); self.slots.acquire(); blocked = time.time() - t
        t_sub = time.time()
        def job():
            t0 = time.time()
            try:
                fn()
            except BaseException as e:
                with self.lock: self.stats["errors"] += 1
                try: self.on_error(tag, e)
                except Exception: pass
            finally:
                try:
                    if always is not None: always()
                finally:
                    with self.lock:
                        self.stats["busy_s"] += time.time() - t0; self.stats["lag_s"] += t0 - t_sub
                    self.slots.release()
        with self.lock:
            self.tags.add(tag); self.stats["jobs"] += 1; self.stats["blocked_s"] += blocked
            pending = sum(1 for f in self.futures if not f.done()) + 1; self.stats["max_pending"] = max(self.stats["max_pending"], pending)
            self.futures.append(self.pool.submit(job))

    def submitted(self, tag) -> bool:
        with self.lock: return tag in self.tags

    def drain(self) -> None:
        for f in list(self.futures): f.result()

    def census(self) -> dict:
        with self.lock: c = dict(self.stats)
        for k in ("busy_s", "lag_s", "blocked_s"): c[k] = round(c[k], 3)
        return c

    def close(self) -> dict:
        self.drain(); self.pool.shutdown(wait=True)
        c = self.census(); c["state"] = "on"; return c


def writer_line(census: Optional[dict], tag: str = "af2ig-opt") -> str:
    """The L16 LEVER line at exit."""
    if not census:
        return f"[{tag}] LEVER name=L16 state=off reason=not_requested flag=-overlap_output"
    c = census
    return (f"[{tag}] LEVER name=L16 state=on impl=thread_writer origin=kit flag=-overlap_output depth={c.get('depth')} jobs={c.get('jobs')} errors={c.get('errors')} "
            f"busy_s={c.get('busy_s')} lag_s={c.get('lag_s')} blocked_s={c.get('blocked_s')} max_pending={c.get('max_pending')}")


def line(census: Optional[dict], depth: int = 0, tag: str = "af2ig-opt") -> str:
    """The L15 LEVER line at exit (the driver prints it; the package's own line is built from the timer record)."""
    if not census:
        return f"[{tag}] LEVER name=L15 state=off reason=not_requested flag=-prefetch"
    c = census
    return (f"[{tag}] LEVER name=L15 state=on impl={IMPL} origin=kit flag=-prefetch depth={c.get('depth')} workers={c.get('workers')} offdevice={c.get('offdevice')} queued={c.get('queued')} taken={c.get('taken')} "
            f"ready={c.get('ready')} waited={c.get('waited')} skipped={c.get('skipped')} errors={c.get('errors')} wait_s={c.get('wait_s')} work_s={c.get('work_s')} overlap_s={c.get('overlap_s')}")
