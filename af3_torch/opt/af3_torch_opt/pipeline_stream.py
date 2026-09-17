"""pipeline_stream.py — the streamed process chain of `pred` (package levers ``prefetch`` and ``write_behind``): the hand-off protocol between
the featuriser, the model process and the writers when they run AT THE SAME TIME instead of one after the other, and the wrapper's
launcher of a step in the background.

The chain's three processes exchange the same files as the sequential chain (``<work>/<name>/seed-<s>/batch.npz`` + ``batch.pkl`` →
``result.npz``); what this module adds is WHEN a file may be read. A producer closes its files, then creates an empty marker beside them;
a consumer reads only after the marker exists — so every byte a consumer reads is a byte the sequential chain would have handed it (the
outputs are the sequential chain's, byte for byte: the levers change scheduling, never data):

    <work>/<name>/seed-<s>/batch.ready     featurise.py: this seed's batch.npz + batch.pkl are complete            (prefetch)
    <work>/<name>/featurise.ok             featurise.py: every seed of the item is complete
    <work>/<name>/featurise.failed         featurise.py: the item failed (its error is in featurise.json as before) — the item is WITHDRAWN downstream
    <work>/featurise.done                  the wrapper: the featurise process has exited (whatever its rc)
    <work>/<name>/seed-<s>/result.ready    forward.py: this seed's result.npz is complete                          (write_behind)
    <work>/forward.done                    the wrapper: the model process has exited (whatever its rc)

The featuriser under ``--stream 1`` also prints ONE line ``SEEDS {"<name>": [s, ...], ...}`` (:data:`SEEDS_MARK`) once every fold input is
loaded and before the first is featurised: the wrapper launches the model process (every input x seed of that table) the moment it reads it,
so the model is built and its weights loaded while the first input featurises, item i+1 featurises while item i is on the GPU, and — under
``write_behind`` — item i is written while item i+1 is on the GPU. A consumer waiting on an item that will never come (its featurisation
failed, the producer died) sees ``featurise.failed`` / ``<step>.done`` without the ready marker and WITHDRAWS the item: it is then accounted
exactly as the sequential chain accounts it (a featurise failure named by featurise.json; no forward record, no writer record).
Standard library only: imported by featurise.py / postprocess.py (JAX venv), forward.py (torch venv) and cli.py alike.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

STREAM_LEVERS = ("prefetch", "write_behind", "feat_par")   # the package levers that stream the chain (modes.MODE_PACKAGE_LEVERS; registry.PACKAGE_LEVERS)
FEAT_WORKERS = 2                      # feat_par: the featurise step as this many processes, the inputs dealt round-robin (each announces its own SEEDS line)
SEEDS_MARK = "SEEDS "                 # featurise.py --stream 1: `SEEDS {"name": [seeds], ...}` once, before the first item featurises (the wrapper's cue to launch the model process)
BATCH_READY = "batch.ready"           # beside batch.npz / batch.pkl in <work>/<name>/seed-<s>/
RESULT_READY = "result.ready"         # beside result.npz
ITEM_OK = "featurise.ok"              # in <work>/<name>/: every seed featurised
ITEM_FAILED = "featurise.failed"      # in <work>/<name>/: the item's featurisation failed (withdrawn downstream)
STEP_DONE = "{step}.done"             # in <work>/: the wrapper's record that a step's process has exited
POLL_S = 0.02                         # a consumer's poll period while it waits (20 ms: below the resolution any wall here is read at)


def exit_with_parent(poll_s: float = 1.0) -> None:
    """A process of the streamed chain ends when the wrapper that launched it is gone (the sequential chain's watchdog kills a step's process
    group from the wrapper's own thread; a step running BESIDE the wrapper learns the wrapper's death here instead of polling a work dir nobody
    will complete): a daemon thread watches the parent pid and exits this process (os._exit(70)) once it changes."""
    parent = os.getppid()

    def _watch():
        while True:
            time.sleep(poll_s)
            if os.getppid() != parent:
                os._exit(70)

    threading.Thread(target=_watch, name="af3t-parent-watch", daemon=True).start()


def touch(path: str) -> None:
    """Create the (empty) marker file ``path`` — after the files it vouches for are closed. Its directory is made when absent."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8"):
        pass


def step_done_path(root: str, step: str) -> str:
    return os.path.join(root, STEP_DONE.format(step=step))


def item_dir_of(seed_file: str) -> str:
    """<work>/<name>/ for a <work>/<name>/seed-<s>/<file> path."""
    return os.path.dirname(os.path.dirname(os.path.abspath(seed_file)))


def await_batch(batch_path: str, root: str, poll_s: float = POLL_S, clock=time.monotonic) -> dict:
    """Block until the featuriser has handed over ``batch_path`` (its seed dir carries BATCH_READY) → ``{"ready": True, "wait_s"}``; or until it
    never will — the item's ITEM_FAILED marker, or the featurise process gone (``featurise.done`` under ``root``) or the item complete (ITEM_OK)
    without this seed's marker → ``{"ready": False, "wait_s", "reason"}`` (the item is withdrawn). The ready marker is looked for AGAIN after a
    terminal marker is seen: the producer writes ready strictly before any of them, so a ready batch is never withdrawn by a race."""
    seed_dir = os.path.dirname(os.path.abspath(batch_path)); item_dir = os.path.dirname(seed_dir)
    ready = os.path.join(seed_dir, BATCH_READY)
    terminal = ((os.path.join(item_dir, ITEM_FAILED), "featurise_failed"), (os.path.join(item_dir, ITEM_OK), "seed_not_featurised"),
                (step_done_path(root, "featurise"), "featuriser_gone"))
    t0 = clock()
    while True:
        if os.path.exists(ready):
            return {"ready": True, "wait_s": round(clock() - t0, 3)}
        for marker, reason in terminal:
            if os.path.exists(marker):
                if os.path.exists(ready):                                   # written before the terminal marker: not a withdrawal
                    return {"ready": True, "wait_s": round(clock() - t0, 3)}
                return {"ready": False, "wait_s": round(clock() - t0, 3), "reason": reason}
        time.sleep(poll_s)


def seed_dirs_present(item_dir: str) -> List[str]:
    """The seed-<s> dirs of an item's work dir that hold a batch.pkl (what postprocess.seed_dirs reads), sorted by seed."""
    out = []
    if os.path.isdir(item_dir):
        for d in os.listdir(item_dir):
            if d.startswith("seed-") and d[5:].isdigit() and os.path.isfile(os.path.join(item_dir, d, "batch.pkl")):
                out.append((int(d[5:]), os.path.join(item_dir, d)))
    return [p for _, p in sorted(out)]


def await_results(item_dir: str, root: str, poll_s: float = POLL_S, clock=time.monotonic) -> dict:
    """Block until the writers may take the item at ``item_dir``: its featurisation complete (ITEM_OK, or the featuriser gone) AND every seed dir
    carries RESULT_READY → ``{"state": "ready", "early": 1, ...}`` (the model process is still live: the write overlaps its next item); or the
    model process gone (``forward.done``) → ``"ready"`` with ``early=0`` when results are there (all or some: postprocess names the missing ones as
    it always has), ``"withdrawn"`` when the item has no result at all or its featurisation failed (the sequential chain never hands such an
    item to the writers). ``wait_s`` = how long the writer waited here."""
    t0 = clock()
    failed, ok = os.path.join(item_dir, ITEM_FAILED), os.path.join(item_dir, ITEM_OK)
    feat_done, fwd_done = step_done_path(root, "featurise"), step_done_path(root, "forward")
    while True:
        if os.path.exists(failed):
            return {"state": "withdrawn", "reason": "featurise_failed", "wait_s": round(clock() - t0, 3), "early": 0}
        featurised = os.path.exists(ok) or os.path.exists(feat_done)
        model_gone = os.path.exists(fwd_done)
        if featurised or model_gone:
            dirs = seed_dirs_present(item_dir)
            have = [d for d in dirs if os.path.exists(os.path.join(d, RESULT_READY))]
            if dirs and len(have) == len(dirs) and featurised:
                return {"state": "ready", "wait_s": round(clock() - t0, 3), "early": int(not model_gone), "seeds": len(dirs)}
            if model_gone:
                if have:
                    return {"state": "ready", "wait_s": round(clock() - t0, 3), "early": 0, "seeds": len(dirs)}
                return {"state": "withdrawn", "reason": "no_result", "wait_s": round(clock() - t0, 3), "early": 0}
        time.sleep(poll_s)


def merge_featurise_reports(reports: Sequence[dict], names: Sequence[str]) -> dict:
    """feat_par: the W featurisers' reports as ONE featurise report — the first worker's fields, every worker's rows in the pred's input order,
    ok = every row ok, workers = W (what account_featurise and the run record read, as if one process had featurised everything)."""
    reports = [r or {"items": []} for r in reports]
    rows = {row.get("name"): row for r in reports for row in (r.get("items") or [])}
    merged = dict(reports[0]); merged["items"] = [rows[n] for n in names if n in rows] + [row for k, row in rows.items() if k not in names]
    merged["ok"] = bool(merged["items"]) and all(bool(row.get("ok")) for row in merged["items"]) and all(r.get("ok", True) is not False or r.get("items") for r in reports)
    merged["workers"] = len(reports)
    return merged


def merge_step_records(records: Sequence[dict]) -> dict:
    """feat_par: the W featurise step records as one (they ran at the same time): rc = the first non-zero, wall_s = the longest, ok = all ok, workers = W."""
    recs = [r for r in records if r]
    out = dict(recs[0]); out["wall_s"] = max(float(r.get("wall_s") or 0.0) for r in recs); out["ok"] = all(bool(r.get("ok")) for r in recs)
    bad = [r for r in recs if r.get("rc")]
    if bad:
        out["rc"] = bad[0]["rc"]; out.update({k: bad[0][k] for k in ("error", "reason") if k in bad[0]})
    out["workers"] = len(recs); out["logs"] = [r.get("log") for r in recs]
    return out


def seeds_line(table: Dict[str, Sequence[int]]) -> str:
    """The featuriser's SEEDS line for {name: seeds} (printed once, flushed, before the first item featurises)."""
    return SEEDS_MARK + json.dumps({k: [int(s) for s in v] for k, v in table.items()}, sort_keys=False)


def parse_seeds_line(text: str) -> Optional[Dict[str, List[int]]]:
    """{name: [seeds]} from a transcript line when it is the SEEDS line, else None."""
    s = text.strip()
    if not s.startswith(SEEDS_MARK):
        return None
    try:
        d = json.loads(s[len(SEEDS_MARK):])
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    return {str(k): [int(x) for x in (v or [])] for k, v in d.items()}


class Job:
    """One step of the chain running in the background: ``run(argv, ..., on_line=)`` (cli.run_step's shape) on a thread; the SEEDS line, when
    the step prints one, sets ``seeds`` and the ``announced`` event; when the process exits the step's ``<step>.done`` marker is created under
    ``root`` (before ``finished`` is set), so consumers polling the work dir learn it without the wrapper's help. ``join()`` returns run's record."""

    def __init__(self, step: str, root: str, run: Callable[..., dict], argv: Sequence[str], env: dict, log_path: str, on_line: Optional[Callable[[str], None]] = None, mark_done: bool = True):
        self.step, self.root = step, root
        self.seeds: Optional[Dict[str, List[int]]] = None
        self.announced, self.finished = threading.Event(), threading.Event()
        self.record: Optional[dict] = None
        self.error: Optional[BaseException] = None
        self.t0 = time.monotonic(); self.t1: Optional[float] = None

        def _on_line(ln: str):
            tbl = parse_seeds_line(ln)
            if tbl is not None and self.seeds is None:
                self.seeds = tbl; self.announced.set()
            if on_line is not None:
                on_line(ln)

        def _body():
            try:
                self.record = run(step, list(argv), env, log_path, on_line=_on_line)
            except BaseException as e:                                       # noqa: BLE001 — re-raised by join() on the wrapper's thread
                self.error = e
            finally:
                self.t1 = time.monotonic()
                try:
                    if mark_done:                                            # feat_par: one of several featurisers — the wrapper marks the step done once ALL of them have ended
                        touch(step_done_path(root, step))
                finally:
                    self.finished.set(); self.announced.set()                 # a step that never announced releases its waiter when it ends

        self.thread = threading.Thread(target=_body, name=f"af3t-{step}", daemon=True)
        self.thread.start()

    def wait_announced(self) -> Optional[Dict[str, List[int]]]:
        """Block until the step printed its SEEDS line (→ the table) or exited without one (→ None)."""
        self.announced.wait()
        return self.seeds

    def join(self) -> dict:
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self.record

    @property
    def span(self):
        """(start, end) on the wrapper's monotonic clock; end None while running."""
        return self.t0, self.t1


def overlap_s(spans) -> float:
    """Σ step walls − the wall they cover together (the union of their [start, end] spans): the seconds the chain's steps ran at the same time."""
    spans = sorted((a, b) for a, b in spans if a is not None and b is not None)
    total = sum(b - a for a, b in spans)
    union, cur = 0.0, None
    for a, b in spans:
        if cur is None or a > cur[1]:
            if cur is not None:
                union += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur is not None:
        union += cur[1] - cur[0]
    return round(total - union, 3)
