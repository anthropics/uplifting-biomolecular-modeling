"""Test helpers a kit's adoption tests use: recorded fixture lines and recorded fixture manifests.

Contract. Adoption of the core is accepted only when the kit's observable record is unchanged: :func:`golden_lines` matches the lines a
formatter produced against the lines (a literal, or a regex given as ``re:<pattern>``, each matched against a full line —
never a truncated copy) and returns every line left unmatched and every produced line the record forbids;
:func:`manifest_diff` compares a manifest with a new one key by key (dotted paths) and returns the additions, the removals and
the changed values — an adoption may add keys, never remove or change one.

:func:`run_ranks` is the CPU launcher of the row-sharded family (:mod:`opt_core.mem.rowpair`): P "ranks" run as threads of one process on the
``threaded`` comm backend (:class:`opt_core.mem.rowpair.dist.ThreadHub`), so a sharded statement is compared with its dense statement in one
interpreter without a process group. The global torch RNG is shared by the rank threads: build random weights / inputs in the main thread and
pass them in (or use a per-thread ``torch.Generator``); module-level monkeypatches are process-global — install them before ``run_ranks``.
"""
from __future__ import annotations

import re
import threading
import traceback
from typing import Callable, Iterable, Mapping, Sequence


def _matches(spec: str, line: str) -> bool:
    if spec.startswith("re:"):
        return re.search(spec[3:], line) is not None
    return spec == line


def golden_lines(expected: Sequence[str], produced: Iterable[str], forbid: Sequence[str] = ()) -> dict:
    """``{"unmatched": [expected specs no produced line matched], "forbidden": [produced lines a forbid spec matched], "ok": bool}``."""
    produced = [ln.rstrip("\n") for ln in produced]
    unmatched = [spec for spec in expected if not any(_matches(spec, ln) for ln in produced)]
    forbidden = [ln for ln in produced if any(_matches(spec, ln) for spec in forbid)]
    return {"unmatched": unmatched, "forbidden": forbidden, "ok": not unmatched and not forbidden}


def _flatten(doc, prefix: str = "") -> dict:
    out = {}
    if isinstance(doc, Mapping):
        for k, v in doc.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        if not doc:
            out[prefix.rstrip(".")] = {}
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            out.update(_flatten(v, f"{prefix}{i}."))
        if not doc:
            out[prefix.rstrip(".")] = []
    else:
        out[prefix.rstrip(".")] = doc
    return out


def manifest_diff(record: Mapping, new: Mapping, volatile: Sequence[str] = ()) -> dict:
    """Key-by-key comparison of two manifests. ``volatile`` names dotted paths (or path prefixes) whose values may differ (wall
    clocks, pids, temp paths). Returns ``{"added": [...], "removed": [...], "changed": {path: (record, new)}, "ok": bool}`` where
    ``ok`` = nothing removed and nothing changed."""
    a, b = _flatten(record), _flatten(new)

    def is_volatile(path):
        return any(path == v or path.startswith(v + ".") for v in volatile)

    added = sorted(k for k in b if k not in a)
    removed = sorted(k for k in a if k not in b)
    changed = {k: (a[k], b[k]) for k in sorted(set(a) & set(b)) if a[k] != b[k] and not is_volatile(k)}
    return {"added": added, "removed": removed, "changed": changed, "ok": not removed and not changed}


def run_ranks(P: int, fn: Callable, *args, timeout_s: float = 600.0, **kwargs) -> list:
    """Run ``fn(rank, P, *args, **kwargs)`` on P rank-threads sharing one :class:`opt_core.mem.rowpair.dist.ThreadHub`; returns the results
    ordered by rank. Inside ``fn``, ``opt_core.mem.rowpair.dist.comm()`` is the calling thread's rank (``backend == "threaded"``) and every
    primitive of the family works. The first rank that raises aborts the hub (peers blocked in a collective are released) and its traceback is
    re-raised here as ``RuntimeError``; a run longer than ``timeout_s`` is an error naming the ranks still alive."""
    from .mem.rowpair.dist import ThreadHub, clear_layout_cache, register_thread_comm, unregister_thread_comm
    P = int(P)
    hub = ThreadHub(P)
    out = [None] * P
    err = [None] * P

    def body(r):
        register_thread_comm(r, P, hub)
        try:
            out[r] = fn(r, P, *args, **kwargs)
        except BaseException as e:  # noqa: BLE001 — every failure is reported with its rank
            err[r] = (e, traceback.format_exc())
            try:
                hub.bar.abort()                                             # unblock peers waiting in a collective
            except Exception:  # noqa: BLE001
                pass
        finally:
            unregister_thread_comm()

    clear_layout_cache()
    threads = [threading.Thread(target=body, args=(r,), name=f"rank{r}", daemon=True) for r in range(P)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout_s)
    alive = [t.name for t in threads if t.is_alive()]
    if alive:
        try:
            hub.bar.abort()
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"run_ranks(P={P}): ranks {alive} still running after {timeout_s}s (a rank skipped a collective?)")
    first = [r for r in range(P) if err[r] is not None and not isinstance(err[r][0], threading.BrokenBarrierError)]
    rest = [r for r in range(P) if err[r] is not None]
    for r in first + rest:
        raise RuntimeError(f"rank {r} failed:\n{err[r][1]}")
    return out
