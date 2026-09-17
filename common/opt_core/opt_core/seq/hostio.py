"""Host-side output path: a writer that drains device outputs staged in a pinned host pool on background thread(s), so serialisation,
hashing and file writes overlap the next forward instead of sitting between forwards.

Contract. The pool is the kit's (injected): any object with ``pinned(slot)`` — block until the asynchronous device-to-host copy into
``slot`` has landed and return the host arrays — and ``release(slot)`` — the slot may be reused. The device loop leases a slot from its
pool, starts the copy, and calls :meth:`D2HWriter.submit` with the item id and the slot; a writer thread takes the item in submission
order, waits on ``pinned``, calls the kit's ``write_fn(item_id, host_arrays, meta) -> row`` (the kit's own bytes: npz + sha256, a
``.pt`` file, a jsonl row), releases the slot, and delivers the row. Rows come back in SUBMISSION order whatever the thread count
(``on_row`` is called in that order on a writer thread; :meth:`D2HWriter.close` returns them in that order), so a kit's per-item file and
row sequence is the sequence of its plain loop. Total accounting: every submitted item yields exactly one row and
``submitted == written + failed`` at close. ``written`` counts items whose ``write_fn`` RETURNED a row — the row is the kit's verbatim,
including a kit's own ``"ok": False`` row for a unit the kit accounted itself; ``failed`` counts items for which THIS module made the row
because ``pinned`` or ``write_fn`` raised (``{"id", "ok": False, "stage": "pinned"|"write", "error": "<Type>: <message>"}``) or because
the item was still undelivered when :meth:`D2HWriter.close` gave up on a live thread (``"stage": "close"``). Faults that do not cost the
item its row are side counters with their own ``failed``-list entries: ``release_failed`` (the pool refused ``release`` — recorded
whether or not the write succeeded) and ``on_row_failed`` (the kit's sink raised; the row still comes back from ``close``). Nothing is
dropped and :meth:`D2HWriter.close` never raises for an item. Back-pressure: ``submit`` blocks while ``depth`` items are in flight, and
:meth:`D2HWriter.wait_for` blocks while an in-flight item holds a given slot (the callable a lease-pool passes as its ``wait_for``).
:func:`line_fields` gives the ``{key: value}`` fields of the kit's lines (``opt_core.report.kv(**fields)``). Standard library only:
the arrays are whatever the pool returns; this module never imports a framework.

With the core's host primitives (``opt_core.host.outputs``). ``PinnedPool`` there leases page-locked byte buffers and ``to_host(tree,
pool=..., leases=[...])`` copies a whole result tree into them in one pass with one synchronisation; :func:`stage` wraps exactly that and
returns a :class:`LeasedHost` handle, and :class:`HostLeases` is the ``pinned`` / ``release`` view of that pool this writer drains — so a
kit takes its pinned buffers from the core's ONE pool and keeps this module for what it adds: rows in submission order, one row per
item with named failures, the ``on_row`` sink, the census. ``opt_core.host.outputs.AsyncWriter`` is the other writer: fire-and-forget
``write(item)`` on a thread pool with back-pressure, unordered, ``drain()`` raising the first failure — the right tool when a kit has no
per-item row and wants an exception; :class:`D2HWriter` is the tool when the per-item row sequence (a jsonl, a rows table, per-unit ok /
failed-by-name accounting) IS the kit's output contract. :func:`witness` is the output proof a kit runs before
``opt_core.host.coldstart.fast_exit``.

Readiness precondition (the pool's side of the contract, not this module's). ``pinned(slot)`` must return only after BOTH the device
work that produced the output AND the device-to-host copy into the slot have completed. A CUDA pool meets it by making its copy stream
wait on the stream that produced the output (``copy_stream.wait_stream(producer_stream)`` or an event recorded on the producer stream)
before the ``non_blocking`` copy, recording an event after the copy, and synchronising on that event inside ``pinned``. An output
produced on a stream other than the one the pool waits on (a CUDA-graph replay or a kit kernel on a side stream) is NOT covered by
waiting on the current stream: the kit passes that stream / a ready event to its pool, or synchronises the device before the lease.
:class:`D2HWriter` never inspects streams or devices; the return of ``pinned`` is the only readiness signal it uses, and it calls
``write_fn`` strictly after it. The :func:`stage` route meets the precondition by construction for outputs produced on the CURRENT
stream (``to_host`` copies on it and synchronises it before returning); an output produced on another stream must be waited on first.
"""
from __future__ import annotations

import operator
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

_STOP = object()


def _slot_key(slot):
    """Slots are the pool's handles: integer-likes (int, numpy integer — anything with ``__index__``) and strings compare by value,
    anything else (a buffer object) by equality — never by an elementwise ``==``."""
    if isinstance(slot, str):
        return ("s", slot)
    if hasattr(slot, "__index__") and not isinstance(slot, (bytes, bytearray)):
        return ("i", operator.index(slot))
    return ("o", id(slot))


def _same_slot(a, b) -> bool:
    return _slot_key(a) == _slot_key(b)


class D2HWriter:
    """Writer thread(s) over an injected pinned pool.

    ``pool``: has ``pinned(slot) -> host_arrays`` (returns only once the producer's work and the copy are complete — the readiness
    precondition in the module docstring) and ``release(slot)``. ``write_fn(item_id, host_arrays, meta) -> dict``: the kit's
    writer; its return value is the item's row verbatim. ``threads``: writer threads (1 keeps a single writer; more overlap several items'
    host work — rows are still delivered in submission order). ``depth``: the most items in flight before ``submit`` blocks (default: no
    bound beyond the pool's own). ``on_row(row)``: called once per item, in submission order, on a writer thread (e.g. append the row to
    the run's jsonl). ``name``: thread name prefix."""

    def __init__(self, pool, write_fn: Callable[[object, object, object], dict], threads: int = 1, depth: Optional[int] = None,
                 on_row: Optional[Callable[[dict], None]] = None, name: str = "hostio-writer"):
        if int(threads) < 1:
            raise ValueError("threads must be >= 1, got {t}".format(t=threads))
        self.pool, self.write_fn, self.on_row = pool, write_fn, on_row
        self.n_threads = int(threads)
        self.depth = None if depth is None else int(depth)
        self._q = queue.Queue()                       # unbounded: back-pressure is counted in _cv on in-flight items, not queue slots
        self._cv = threading.Condition()              # guards counters, in-flight, ordering; NEVER held while kit code (on_row) runs
        self._emit_cv = threading.Condition(threading.Lock())   # serialises on_row calls in ticket order, outside _cv
        self._emit_next = 0                           # next emit ticket to run
        self._emit_tickets = 0                        # tickets handed out (under _cv)
        self._finalised = False                       # close() accounted every item; later deliveries are counted as late, not applied
        self._in_flight = {}                          # type: Dict[int, Tuple[object, object]]   seq -> (slot, item_id)
        self._done = {}                               # type: Dict[int, dict]     seq -> row, waiting for in-order delivery
        self._rows = []                               # type: List[dict]          delivered, submission order
        self._failed = []                             # type: List[dict]
        self._next_seq = 0                            # next sequence number to hand out
        self._next_emit = 0                           # next sequence number to deliver
        self._closed = False
        self._stats = {"submitted": 0, "written": 0, "failed": 0, "on_row_failed": 0, "release_failed": 0, "late": 0,
                       "blocked_s": 0.0, "wait_s": 0.0, "host_s": 0.0, "max_in_flight": 0}
        self._threads = [threading.Thread(target=self._loop, name="{n}-{i}".format(n=name, i=i), daemon=True) for i in range(self.n_threads)]
        for t in self._threads:
            t.start()

    # ---------------------------------------------------------------- device-loop side

    def submit(self, item_id, slot, meta=None) -> int:
        """Queue one item whose device-to-host copy into ``slot`` (the pool's handle: an int, a str, or an object compared by equality) has
        been started. Blocks while ``depth`` items are in flight. Returns
        the item's sequence number."""
        with self._cv:
            if self._closed:
                raise RuntimeError("submit after close")
            t0 = time.perf_counter()
            while self.depth is not None and len(self._in_flight) >= self.depth:
                self._cv.wait()
            self._stats["blocked_s"] += time.perf_counter() - t0
            seq = self._next_seq
            self._next_seq += 1
            self._in_flight[seq] = (slot, item_id)
            self._stats["submitted"] += 1
            self._stats["max_in_flight"] = max(self._stats["max_in_flight"], len(self._in_flight))
        self._q.put((seq, item_id, slot, meta))
        return seq

    def wait_for(self, slot) -> None:
        """Block while an in-flight item holds ``slot``; returns when the slot is free for the pool to lease again."""
        with self._cv:
            t0 = time.perf_counter()
            while any(_same_slot(s, slot) for s, _ in self._in_flight.values()):
                self._cv.wait()
            self._stats["blocked_s"] += time.perf_counter() - t0

    def in_flight(self) -> int:
        with self._cv:
            return len(self._in_flight)

    def close(self, timeout: Optional[float] = None) -> dict:
        """Drain every submitted item, stop the threads, and return ``{"rows": [...submission order...], "failed": [...], "stats": {...}}``.
        Idempotent."""
        with self._cv:
            already = self._closed
            self._closed = True
        if not already:
            for _ in self._threads:
                self._q.put(_STOP)
            for t in self._threads:
                t.join(timeout)
        alive = [t.name for t in self._threads if t.is_alive()]
        emit = []                                     # type: List[dict]
        with self._cv:
            if not self._finalised:
                if alive:                             # a join timed out: every item still without a row gets one, by name
                    for seq in sorted(self._in_flight):
                        if seq not in self._done and seq >= self._next_emit:
                            slot, item_id = self._in_flight[seq]
                            row = {"id": item_id, "ok": False, "stage": "close", "error": "undelivered at close: writer thread alive after timeout={t}".format(t=timeout)}
                            self._done[seq] = row
                            self._failed.append(row)
                            self._stats["failed"] += 1
                    emit = self._drain_locked()
                    self._in_flight.clear()
                self._finalised = True
            ticket = self._take_ticket_locked() if emit else None
        self._emit(ticket, emit)
        out = {"rows": self.rows, "failed": list(self._failed), "stats": self.stats()}
        if alive:
            out["stats"]["threads_alive"] = alive         # a join that timed out is said, not hidden
        return out

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ---------------------------------------------------------------- writer side

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is _STOP:
                return
            seq, item_id, slot, meta = item
            row, failed, release_fault = None, None, None
            stage = "pinned"
            try:
                t0 = time.perf_counter()
                arrays = self.pool.pinned(slot)
                t1 = time.perf_counter()
                stage = "write"
                row = self.write_fn(item_id, arrays, meta)
                t2 = time.perf_counter()
                if not isinstance(row, dict):
                    raise TypeError("write_fn returned {t}, not a dict row".format(t=type(row).__name__))
                timing = (t1 - t0, t2 - t1)
            except Exception as e:  # noqa: BLE001 — a failed item is a named row, never silence
                row = {"id": item_id, "ok": False, "stage": stage, "error": "{t}: {m}".format(t=type(e).__name__, m=e)}
                failed = row
                timing = (0.0, 0.0)
            finally:
                try:
                    self.pool.release(slot)
                except Exception as e:  # noqa: BLE001 — a pool that refuses release is recorded whether or not the write succeeded
                    release_fault = {"id": item_id, "ok": False, "stage": "release", "error": "release({s!r}) raised {t}: {m}".format(s=slot, t=type(e).__name__, m=e)}
            self._deliver(seq, row, failed, release_fault, timing)

    def _deliver(self, seq: int, row: dict, failed: Optional[dict], release_fault: Optional[dict], timing: Tuple[float, float]) -> None:
        with self._cv:
            if self._finalised:                       # close() already gave this item its row; say that a late result arrived, apply nothing
                self._stats["late"] += 1
                return
            self._stats["wait_s"] += timing[0]
            self._stats["host_s"] += timing[1]
            if failed is not None:
                self._failed.append(failed)
                self._stats["failed"] += 1
            else:
                self._stats["written"] += 1
            if release_fault is not None:
                self._failed.append(release_fault)
                self._stats["release_failed"] += 1
            self._done[seq] = row
            emit = self._drain_locked()
            self._in_flight.pop(seq, None)
            self._cv.notify_all()
            ticket = self._take_ticket_locked() if emit else None
        self._emit(ticket, emit)                      # kit code runs here, outside _cv: submit / wait_for / other writers are never held by it

    def _drain_locked(self) -> List[dict]:
        """Under ``_cv``: move every row that is next in submission order from ``_done`` to ``_rows``; return them (to emit)."""
        out = []                                      # type: List[dict]
        while self._next_emit in self._done:          # in-order delivery: a later item waits in _done for an earlier one
            r = self._done.pop(self._next_emit)
            self._rows.append(r)
            out.append(r)
            self._next_emit += 1
        return out

    def _take_ticket_locked(self) -> int:
        """Under ``_cv``: the emit ticket for a batch drained by :meth:`_drain_locked` — tickets follow drain order, so ``on_row`` order
        follows submission order even when several writer threads deliver at once."""
        t = self._emit_tickets
        self._emit_tickets += 1
        return t

    def _emit(self, ticket: Optional[int], rows: List[dict]) -> None:
        """Run ``on_row`` for ``rows`` when ``ticket`` comes up; only ``_emit_cv`` is held, so two threads never interleave the kit's appends
        and no writer-side lock is held while kit code runs."""
        if ticket is None:
            return
        with self._emit_cv:
            while ticket != self._emit_next:
                self._emit_cv.wait()
            try:
                if self.on_row is not None:
                    for r in rows:
                        try:
                            self.on_row(r)
                        except Exception as e:  # noqa: BLE001 — the row exists (returned by close); its delivery to the kit's sink failed, said by name
                            with self._cv:
                                self._failed.append({"id": r.get("id"), "ok": False, "stage": "on_row", "error": "on_row raised {t}: {m}".format(t=type(e).__name__, m=e)})
                                self._stats["on_row_failed"] += 1
            finally:
                self._emit_next += 1
                self._emit_cv.notify_all()

    # ---------------------------------------------------------------- record

    @property
    def rows(self) -> List[dict]:
        with self._cv:
            return list(self._rows)

    def stats(self) -> dict:
        """``{"writer": "thread", "threads", "depth", "submitted", "written", "failed", "on_row_failed", "release_failed", "late",
        "in_flight", "max_in_flight", "blocked_s" (device loop blocked in submit / wait_for), "wait_s" (writers waiting for copies),
        "host_s" (inside write_fn)}``; ``submitted == written + failed`` once closed (definitions in the module docstring)."""
        with self._cv:
            s = dict(self._stats)
            s.update({"writer": "thread", "threads": self.n_threads, "depth": self.depth, "in_flight": len(self._in_flight)})
            for k in ("blocked_s", "wait_s", "host_s"):
                s[k] = round(s[k], 4)
            return s


class LeasedHost:
    """One item's host copy: ``tree`` (the result tree with host tensors — views over leased pinned buffers), ``leases`` (the buffers to
    give back), ``stats`` (``to_host``'s census: leaves, bytes, pinned, unpinned). The handle IS the writer's slot (compared by equality)."""
    __slots__ = ("tree", "leases", "stats")

    def __init__(self, tree, leases, stats=None):
        self.tree, self.leases, self.stats = tree, list(leases), dict(stats or {})


class HostLeases:
    """The ``pinned`` / ``release`` protocol over a byte-buffer pool with ``release(buf)`` (``opt_core.host.outputs.PinnedPool``): ``pinned(h)``
    returns ``h.tree`` (already synchronised by :func:`stage`), ``release(h)`` returns every leased buffer to the pool. Counts unpinned
    (pageable, over-budget) leaves so the kit's line can say whether the pool's budget held."""

    def __init__(self, pool):
        self.pool = pool
        self.items = 0
        self.unpinned_leaves = 0

    def pinned(self, handle):
        if not isinstance(handle, LeasedHost):
            raise TypeError("HostLeases.pinned: expected a LeasedHost handle from stage(), got {t}".format(t=type(handle).__name__))
        self.items += 1
        self.unpinned_leaves += int(handle.stats.get("unpinned", 0))
        return handle.tree

    def release(self, handle):
        for buf in handle.leases:
            self.pool.release(buf)
        handle.leases = []


def stage(pool, tree) -> LeasedHost:
    """Copy ``tree`` (dict / list / tuple of device tensors) to host through ``opt_core.host.outputs.to_host`` into buffers leased from
    ``pool`` (``opt_core.host.outputs.PinnedPool``) — one pass, one synchronisation of the current stream — and return the
    :class:`LeasedHost` handle to pass to :meth:`D2HWriter.submit`. The host tensors are views over pinned bytes with the source's dtype and
    shape (the same bytes ``.cpu()`` would give); convert dtype on the DEVICE before staging where the kit's writer expects a conversion."""
    from opt_core.host.outputs import to_host             # framework-free import; torch is touched only for torch leaves
    leases = []                                            # type: List[object]
    stats = {}                                             # type: Dict[str, int]
    host = to_host(tree, pool=pool, leases=leases, stats=stats)
    return LeasedHost(host, leases, stats)


def witness(paths) -> str:
    """The output proof before a fast exit: ``""`` when every path is an existing non-empty file and every ``.json`` among them parses;
    else the first failure as a word — ``"witness:<path>"`` (missing / empty) or ``"json:<path>"`` (does not parse). The kit puts a
    non-empty word on its EXIT line as ``exit=teardown(<word>)`` and leaves through the interpreter; on ``""`` it prints ``exit=fast`` and
    calls ``opt_core.host.coldstart.fast_exit(0, synced=paths)``."""
    import json
    import os
    for p in [str(x) for x in paths]:
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return "witness:" + p
        if p.endswith(".json"):
            try:
                with open(p) as fh:
                    json.load(fh)
            except (OSError, ValueError):
                return "json:" + p
    return ""


def line_fields(writer, key: str = "writer", when: str = "exit") -> Dict[str, str]:
    """The line fields of a writer (or its ``stats()`` dict), every value a str. ``when="active"`` (the configuration, known when the
    lever is switched on): ``{<key>: "thread", "threads": <n>, "depth": <n|none>}``. ``when="exit"`` (the census, known after
    :meth:`D2HWriter.close`): ``{<key>: "thread", "threads", "submitted", "written", "failed", "on_row_failed", "release_failed",
    "blocked_s"}`` — ``failed`` is what a run record needs (0 = every item got the kit's row); ``blocked_s`` says whether the pool / depth
    starved the device loop. Feed to
    ``opt_core.report.kv(**fields)`` (insertion order is the field order)."""
    s = writer.stats() if hasattr(writer, "stats") else dict(writer)
    fields = {key: str(s.get("writer", "thread")), "threads": str(s["threads"])}  # type: Dict[str, str]
    if when == "active":
        fields["depth"] = "none" if s.get("depth") is None else str(s.get("depth"))
        return fields
    if when != "exit":
        raise ValueError("when must be 'active' or 'exit', got {w!r}".format(w=when))
    for k in ("submitted", "written", "failed"):
        fields[k] = str(s[k])
    fields["on_row_failed"] = str(s.get("on_row_failed", 0))
    fields["release_failed"] = str(s.get("release_failed", 0))
    fields["blocked_s"] = str(s["blocked_s"])
    return fields
