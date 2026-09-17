"""The item loop of a resident process: many designs / requests in one process, each seeded and accounted for as if it had its own.

Contract. :func:`run_items` runs ``run_one(item)`` for every item of a request in order. Around each item it calls the kit's hooks —
``before_item(index, item)`` (the place a kit re-seeds every generator exactly as a fresh process at that item's seed would, resets
per-item state, selects the output directory) and ``after_item(index, item, row)``, whose own failure marks the item failed — and records ONE row per item: ``ok`` with the wall
time and whatever JSON-serialisable summary ``run_one`` returned, or ``failed`` with ``reason = "<Type>: <message>"`` (the traceback goes
to stderr in full: a run log shows the stack). Nothing is retried and nothing is skipped silently: an item is ``ok`` or ``failed`` with a
named reason, and the census closes (``ok + failed == requested``; with ``stop_on_error`` the items never started are listed as
``not_run`` and counted). ``KeyboardInterrupt`` / ``SystemExit`` are not items' failures: the current row is written as ``interrupted``
and the exception propagates.

The journal (``journal=path``) is append-only JSON lines, one per item, flushed and fsynced as each item ends, so a process that dies
mid-request leaves the record of what it finished; :func:`read_journal` reads it back. The census (:class:`Census`) is the request's
account: ``fields()`` for the kit's summary line via :func:`opt_core.report.kv`, ``incomplete()`` = the clause for
:func:`opt_core.report.verdict` (``None`` when every item is ok), ``as_dict()`` for the manifest's pass block
(``n_items`` / ``n_items_complete`` / ``item_failed`` in :func:`opt_core.manifest.build` terms).

What stays the kit's: the meaning of an item (a design index, a job line, a seed), the reseed rule, the model held across items, the
per-item output paths, and the proof that item *i* of a resident run equals item *i* of a fresh process (the engine's equality row).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Optional

STATUSES = ("ok", "failed", "not_run", "interrupted")


class Census:
    """The account of one request: every item in exactly one status."""

    def __init__(self, name: str = "resident"):
        self.name = name
        self.rows = []

    @property
    def requested(self) -> int:
        return len(self.rows)

    def count(self, status: str) -> int:
        return sum(1 for r in self.rows if r["status"] == status)

    @property
    def ok(self) -> int:
        return self.count("ok")

    @property
    def complete(self) -> bool:
        return self.requested > 0 and self.ok == self.requested

    def failures(self) -> list:
        return [(r["id"], r["status"] if r["status"] != "failed" else r["reason"]) for r in self.rows if r["status"] != "ok"]

    def incomplete(self) -> Optional[str]:
        """The ``incomplete`` clause for :func:`opt_core.report.verdict`: ``None`` when every item is ok."""
        if self.complete:
            return None
        if not self.rows:
            return "no items ran"
        bad = self.failures()
        head = "; ".join(f"{i} ({why})" for i, why in bad[:5]) + (f"; … (+{len(bad) - 5})" if len(bad) > 5 else "")
        return f"{len(bad)} of {self.requested} items not ok: {head}"

    def fields(self) -> dict:
        bad = [i for i, _ in self.failures()]
        return {f"{self.name}_items": f"{self.ok}/{self.requested}", f"{self.name}_failed": bad or None}

    def as_dict(self) -> dict:
        return {"name": self.name, "n_items": self.requested, "n_items_complete": self.ok, "items_complete": self.complete,
                "item_failed": [{"id": r["id"], "status": r["status"], "reason": r.get("reason")} for r in self.rows if r["status"] != "ok"],
                "wall_s_total": round(sum(r.get("wall_s") or 0.0 for r in self.rows), 3)}


def _jsonable(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return repr(x)


def _append(journal: Optional[str], row: dict) -> None:
    if not journal:
        return
    with open(journal, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_items(items: Iterable[Any], run_one: Callable[[Any], Any], *, item_id: Callable[[int, Any], str] = None,
              before_item: Optional[Callable[[int, Any], None]] = None, after_item: Optional[Callable[[int, Any, dict], None]] = None,
              journal: Optional[str] = None, stop_on_error: bool = False, name: str = "resident",
              clock: Callable[[], float] = time.perf_counter, stderr=None) -> Census:
    """Run every item; return the census (module contract). ``item_id(index, item)`` names an item in the record (default: ``str(item)``
    for str/int items, else the index)."""
    err = stderr if stderr is not None else sys.stderr
    items = list(items)
    census = Census(name)

    def ident(i, item):
        if item_id is not None:
            return str(item_id(i, item))
        return str(item) if isinstance(item, (str, int)) else str(i)

    stopped_at = None
    for i, item in enumerate(items):
        row = {"i": i, "id": ident(i, item), "status": None, "reason": None, "wall_s": None, "t_end": None}
        if stopped_at is not None:
            row.update(status="not_run", reason=f"stop_on_error after item {stopped_at}")
            census.rows.append(row); _append(journal, row)
            continue
        t0 = clock()
        try:
            if before_item is not None:
                before_item(i, item)
            result = run_one(item)
        except (KeyboardInterrupt, SystemExit) as e:
            row.update(status="interrupted", reason=type(e).__name__, wall_s=round(clock() - t0, 3), t_end=time.time())
            census.rows.append(row); _append(journal, row)
            raise
        except Exception as e:
            traceback.print_exc(file=err)
            row.update(status="failed", reason=f"{type(e).__name__}: {e}", wall_s=round(clock() - t0, 3), t_end=time.time())
            if stop_on_error:
                stopped_at = i
        else:
            row.update(status="ok", wall_s=round(clock() - t0, 3), t_end=time.time())
            if result is not None:
                row["result"] = _jsonable(result)
        if after_item is not None:                                      # runs before the row is final: its failure is the item's
            try:
                after_item(i, item, row)
            except Exception as e:
                traceback.print_exc(file=err)
                if row["status"] == "ok":
                    row.update(status="failed", reason=f"after_item: {type(e).__name__}: {e}")
                    row.pop("result", None)
                    if stop_on_error:
                        stopped_at = i
        census.rows.append(row)
        _append(journal, row)
    return census


def read_journal(path: str) -> list:
    """The rows of a journal file, in order; a torn last line (a process killed mid-write) is reported as a row with status
    ``interrupted`` and reason ``torn journal line``, never dropped."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                rows.append({"i": None, "id": f"line{n}", "status": "interrupted", "reason": "torn journal line", "raw": line[:200]})
    return rows
