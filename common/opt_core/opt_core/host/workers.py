"""Ordered, bounded prefetch: build item k+1 on host workers while the consumer works on item k.

Contract. ``Prefetcher(build, items, workers=W, depth=D)`` runs ``build(item)`` for the items ahead of the consumer on ``W`` threads
(``executor="process"``: worker processes — ``build`` and the items must pickle, else :class:`~opt_core.host.events.HostRefused`
``reason=unpicklable``), keeping at most ``D`` finished-or-running items beyond the one being consumed (host memory is bounded by
``D`` feature sets, not by the input list). Delivery is IN INPUT ORDER, one of two forms — the consumer picks the one its accounting
needs, never both on one prefetcher:

* ``for item, value in prefetcher:`` — a build that raised re-raises at that item's turn as :class:`WorkerError` (``.index``,
  ``.item``, ``__cause__`` = the original); nothing later is silently skipped (iteration stops there; ``close()`` cancels the rest).
* ``for rec in prefetcher.records():`` — every item yields a :class:`Record` ``(index, item, value, error, build_s)``; the consumer
  writes ``ok`` or ``failed:<reason>`` per item and the counts close (total accounting).

``close()`` (or the ``with`` block) cancels unstarted builds, waits for running ones, and joins the pool; the tally names produced,
consumed, failed, cancelled (never started) and discarded (built, never delivered) so a launcher can check
``consumed + failed + cancelled + discarded == submitted`` (``unsubmitted`` = items never handed to a worker at an early close: a count
when the input has a length, else ``unknown``). A build that never returns makes ``close()`` wait for it: a hung featuriser is a hung
run here exactly as in stock, visibly (no progress), never a skipped item.

Numerics: none moved — PROVIDED ``build`` takes its randomness from the item (a seed carried in the item), not from process-global
generators consumed in build order: with ``W > 1`` builds run concurrently and finish in any order. An adapter whose featuriser seeds a
global RNG per item keeps ``workers=1`` (overlap with the consumer, no reordering among builds) and says so in the kit's CHANGES.
Evidence: ``evidence()`` / ``active_line(tag, name)`` → ``[<tag>] LEVER name=F6.item_ordering_prefetch state=on impl=host.workers origin=core pool=.. workers=..
depth=.. executor=..``;
``tally()`` / ``tally_line(tag)`` → ``... TALLY name=.. submitted=.. produced=.. consumed=.. failed=.. cancelled=.. build_s=.. wait_s=..``
(``wait_s`` = time the consumer blocked waiting for a build: near zero means the prefetch hid the host work; near ``build_s`` means
it hid nothing).
"""
from __future__ import annotations

import collections
import pickle
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Deque, Iterable, Iterator, Optional, Tuple

from .. import report
from .events import HostEvent, HostRefused

LEVER = "host.workers"
EXECUTORS = ("thread", "process")


class WorkerError(RuntimeError):
    """``build(item)`` raised; ``index``/``item`` say which, ``__cause__`` is the original exception."""

    def __init__(self, index: int, item: Any, cause: BaseException) -> None:
        self.index, self.item = index, item
        super().__init__(f"HOST {LEVER} build failed for item #{index}: {cause!r}")


@dataclass
class Record:
    index: int
    item: Any
    value: Any = None
    error: Optional[BaseException] = None
    build_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def _timed_build(build: Callable[[Any], Any], item: Any) -> Tuple[Any, float]:
    t0 = time.perf_counter()
    value = build(item)
    return value, time.perf_counter() - t0


class Prefetcher:
    def __init__(self, build: Callable[[Any], Any], items: Iterable[Any], *, workers: int = 1, depth: int = 2,
                 executor: str = "thread", name: str = "featurise") -> None:
        if workers < 1 or depth < 1:
            raise ValueError("workers and depth must be >= 1")
        if executor not in EXECUTORS:
            raise ValueError(f"executor must be one of {EXECUTORS}")
        self.build, self.name = build, str(name)
        self.workers, self.depth, self.executor = int(workers), int(depth), executor
        self._total = len(items) if hasattr(items, "__len__") else None   # for the unsubmitted count at an early close
        self._items = iter(items)
        self._next_index = 0
        self._exhausted = False
        self._inflight: Deque[Tuple[int, Any, Future]] = collections.deque()
        self._closed = False
        self._iter_form: Optional[str] = None
        self._t = {"submitted": 0, "produced": 0, "consumed": 0, "failed": 0, "cancelled": 0, "discarded": 0, "unsubmitted": 0, "build_s": 0.0, "wait_s": 0.0}
        if executor == "process":
            try:
                pickle.dumps(build)
            except Exception as exc:                      # noqa: BLE001 — any pickling failure is the same refusal
                raise HostRefused(LEVER, "unpicklable", what="build", error=type(exc).__name__) from exc
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        else:
            self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=f"opt-core-{self.name}")
        self._fill()

    # -- scheduling
    def _fill(self) -> None:
        """Keep up to ``depth`` items in flight beyond the consumer."""
        while not self._exhausted and not self._closed and len(self._inflight) < self.depth:
            try:
                item = next(self._items)
            except StopIteration:
                self._exhausted = True
                break
            idx = self._next_index
            self._next_index += 1
            fut = None
            if self.executor == "process":
                try:
                    pickle.dumps(item)
                except Exception as exc:                  # noqa: BLE001 — an item a worker process cannot receive: failed, with its name
                    fut = Future()
                    refusal = HostRefused(LEVER, "unpicklable", what=f"item#{idx}", error=type(exc).__name__)
                    refusal.__cause__ = exc
                    fut.set_exception(refusal)
            if fut is None:
                fut = self._pool.submit(_timed_build, self.build, item)
            self._inflight.append((idx, item, fut))
            self._t["submitted"] += 1

    def _take(self) -> Optional[Record]:
        if self._closed or not self._inflight:
            return None
        idx, item, fut = self._inflight.popleft()
        t0 = time.perf_counter()
        try:
            value, build_s = fut.result()
            err = None
        except Exception as exc:                          # noqa: BLE001 — the build's own failure, delivered with its item below
            value, build_s, err = None, 0.0, exc
        self._t["wait_s"] += time.perf_counter() - t0
        if err is None:
            self._t["produced"] += 1
            self._t["build_s"] += build_s
        else:
            self._t["failed"] += 1
        self._fill()                                      # top the window up before handing the item over
        return Record(idx, item, value, err, build_s)

    # -- delivery, form 1: values in order, a failure raises at its turn
    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        self._choose("iter")
        return self._values()

    def _values(self) -> Iterator[Tuple[Any, Any]]:
        while True:
            rec = self._take()
            if rec is None:
                return
            if rec.error is not None:
                self.close()
                raise WorkerError(rec.index, rec.item, rec.error) from rec.error
            self._t["consumed"] += 1
            yield rec.item, rec.value

    # -- delivery, form 2: a record per item, the consumer accounts for failures
    def records(self) -> Iterator[Record]:
        self._choose("records")
        return self._records()

    def _records(self) -> Iterator[Record]:
        while True:
            rec = self._take()
            if rec is None:
                return
            if rec.error is None:
                self._t["consumed"] += 1
            yield rec

    def _choose(self, form: str) -> None:
        if self._iter_form not in (None, form):
            raise RuntimeError("a Prefetcher is consumed through __iter__ OR records(), not both")
        self._iter_form = form

    # -- shutdown
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _, _, fut in self._inflight:
            if fut.cancel():
                self._t["cancelled"] += 1
        for _, _, fut in self._inflight:                  # running builds finish (their exceptions are theirs; the census counts them)
            if not fut.cancelled():
                try:
                    fut.result()
                    self._t["produced"] += 1
                    self._t["discarded"] += 1             # built but never delivered (closed early)
                except Exception:                         # noqa: BLE001 — a build failure at shutdown is still counted
                    self._t["failed"] += 1
        self._inflight.clear()
        if not self._exhausted:                           # never submitted: counted when the input had a length, else named unknown
            self._t["unsubmitted"] = (self._total - self._next_index) if self._total is not None else "unknown"
            self._exhausted = True
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "Prefetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- census and evidence
    def evidence(self) -> dict:
        return {"lever": LEVER, "pool": self.name, "workers": self.workers, "depth": self.depth, "executor": self.executor}

    def active_line(self, tag: str, name: str = "F6.item_ordering_prefetch", origin: str = "core") -> str:
        """The kit's ONE activation line for this lever: ``[<tag>] LEVER name=<name> state=on impl=host.workers origin=core ...`` (``name`` = strategy id)."""
        ev = self.evidence()
        return HostEvent(LEVER, "ACTIVE", "", {k: v for k, v in ev.items() if k != "lever"}).lever_line(tag, name, origin)

    def tally(self) -> dict:
        out = dict(self._t)
        out["name"] = self.name
        out["build_s"] = round(out["build_s"], 3)
        out["wait_s"] = round(out["wait_s"], 3)
        return out

    def tally_line(self, tag: str) -> str:
        t = self.tally()
        fields = collections.OrderedDict((k, t[k]) for k in ("name", "submitted", "produced", "consumed", "failed", "cancelled", "discarded", "unsubmitted", "build_s", "wait_s"))
        return HostEvent(LEVER, "TALLY", "", fields).line(tag)

    def register_exit_tally(self, tag: str) -> bool:
        return report.register_exit_tally(tag + ":" + LEVER + ":" + self.name, lambda: self.tally_line(tag))
