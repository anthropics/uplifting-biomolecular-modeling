"""The `writer_overlap` lever (exact class, in BYTES): the engine's prediction writer renders
and writes item k's files while item k+1 is featurised and forwarded, instead of between the two forwards on the predicting process's main thread.

The engine's writer callback (`<engine>.core.runners.writer.OF3OutputWriter.on_predict_batch_end` -> `write_all_outputs`) renders, per
diffusion sample, one mmCIF through biotite (`write_structure_prediction`) and two indented JSON texts (`write_confidence_scores`) on the main
thread after `predict_step` returned and before the loop takes item k+1 — seconds per five-sample item that neither the GPU nor the next
item's featurisation worker has to wait for. This module re-states `on_predict_batch_end` (its text at the pin: DIGESTS, per kit adapter; a
different text refuses the lever by name — empty = unchecked) so that
  * the callback's own bookkeeping runs verbatim on the main thread (repeated items, `total_count`, failed forwards);
  * the device-to-host copies the writer would make first (`.cpu()` of the predicted coordinates and of every confidence tensor, then the
    writer's `.float()`) are made on the main thread in the writer's order, and the output sub-directories the writer creates are created
    (the engine's `PredictTimer` writes `timing.json` into them right after this callback); the item then holds no device memory and the
    engine's `del batch, outputs` frees what it freed;
  * the rendering and writing — the writer instance's OWN `write_all_outputs(batch=…, outputs=…, confidence_scores=…)` on those host copies,
    so every other patch of the writer (e.g. `fastjson`) applies unchanged — runs in ONE writer worker, jobs in item order, at most QUEUE_MAX
    items in flight (beyond that the main thread waits: bounded host memory): a forked child process (route `process`, the default: no
    interpreter lock shared with the next forward's kernel launches; the child touches no device, runs torch single-threaded and only views
    the host arrays it is sent) or a thread of the predicting process (route `thread`);
  * `on_predict_end` (the engine's summary), a full queue, a worker that died, and interpreter exit all DRAIN first: every submitted item is
    written and acknowledged, success / failure tallied into the callback's own counters as the engine tallies them (a failed item logged with
    its traceback), before the summary is composed; an item the worker could not take, or one in flight when the worker died, is written
    inline by the engine's statement — never lost, counted (`inline=<reason>:n`), and the route demoted to `sync` for the rest of the process.
Inline by design (the engine's statements on the main thread, counted, the lever still `on`): the LAST item of the predict loop
(`trainer.num_predict_batches`: nothing runs after it that its writing could hide behind — a single-query process writes exactly as the
engine does), `write_features` / `write_latent_outputs`
(torch.save of the whole batch / outputs), a distributed predict (`trainer.world_size > 1`), route `sync` (the A/B arm: installed, every
item written where the engine writes it).

Why it is exact in bytes: every file is produced by the engine's own function on inputs equal to the ones it would have read — the writer's
`x[b].cpu().float().numpy()` sees a host float32 tensor holding the device tensor's values (`.cpu()` copies, `.float()` is the cast the
writer applies after its own `.cpu()`; on the host copy both are the identity), the per-sample statements and their order are unchanged,
the child process is a fork of the process that would have run them (same modules, same patches, same encoder classes). Only WHEN the
statements run moves. `timing.json` (the engine's `PredictTimer`: forward + callback wall per item) no longer includes the writing.
Not exact: nothing. The model forward is untouched; what moves is the item's wall time after it — on the `thread` route the worker shares
the interpreter lock with the next forward's launches.

Census (exit): `<PREFIX> LEVER name=writer_overlap state=<on|off|refused> route=<process|thread|sync> items=<n> overlapped=<n>
inline=<reason:n,…|none> host_copy_s=<s> wait_s=<s the main thread waited on a full queue> drain_s=<s waited at drain> write_s=<sum of the
worker's per-item write seconds> write_exc=<items whose write raised; the engine logs them> worker=<pid|thread|->`.
Switches (the kit adapter's `configure(ENV=…, ENV_ROUTE=…)`): <KIT>_WRITER_OVERLAP=1 arms; <KIT>_WRITER_OVERLAP_ROUTE=process|thread|sync
(default process). Engines: the OF3 code family (0.4.x, 0.5.x: the same callback text); the kits' `cells/writer_overlap.py` bind it.
"""
from __future__ import annotations

import atexit
import hashlib
import inspect
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

CONFIGURABLE = ("PREFIX", "ENV", "ENV_ROUTE", "M_WRITER", "WRITER_CLASS", "QUEUE_MAX", "DIGESTS", "CHILD_CENSUS")
PREFIX = "[of3-opt/writer_overlap]"
ENV = ""                                    # the arming switch (<KIT>_WRITER_OVERLAP)
ENV_ROUTE = ""                              # the route knob (<KIT>_WRITER_OVERLAP_ROUTE)
M_WRITER = ""                               # the engine's writer module (openfold3.core.runners.writer)
WRITER_CLASS = "OF3OutputWriter"
QUEUE_MAX = 2                               # items in flight (submitted, not yet acknowledged) before the main thread waits
DIGESTS: Dict[str, Tuple[str, ...]] = {}    # "on_predict_batch_end" -> accepted sha256[:16] of the engine's method source (dedented); empty = unchecked
CHILD_CENSUS: List[Tuple[str, Any, Any]] = []   # [(name, snapshot() -> {counter: number|dict}, merge(delta) -> None)]: cells whose counters move in the worker PROCESS (the kit binding registers fastjson's)
VALUES = ("1",)
ROUTES = ("process", "thread", "sync")
MARK = "_of3opt_writer_overlap"

STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "route": "process", "patched": 0, "items": 0, "overlapped": 0, "inline": {},
                         "host_copy_s": 0.0, "wait_s": 0.0, "drain_s": 0.0, "write_s": 0.0, "failed": 0, "worker": "-", "digest": {}}
ORIG: Dict[str, Any] = {}
_WORKER: Dict[str, Any] = {"obj": None, "writer": None, "seq": 0}
_LOCK = threading.Lock()
_ATEXIT = {"registered": False}


def configure(**kw) -> None:
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if not ENV:
        return False
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def route(environ=None) -> str:
    environ = os.environ if environ is None else environ
    v = ((environ.get(ENV_ROUTE) if ENV_ROUTE else "") or "").strip() or "process"
    if v not in ROUTES:
        raise ValueError(f"{ENV_ROUTE}={v!r} is not one of {'|'.join(ROUTES)}")
    return v


def serving() -> bool:
    return bool(STATE["installed"]) and STATE["state"] == "on"


def census_line() -> str:
    inl = ",".join("%s:%d" % kv for kv in sorted(STATE["inline"].items())) or "none"
    return (f"{PREFIX} LEVER name=writer_overlap state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" route={STATE['route']} items={STATE['items']} overlapped={STATE['overlapped']} inline={inl} host_copy_s={STATE['host_copy_s']:.2f}"
            f" wait_s={STATE['wait_s']:.2f} drain_s={STATE['drain_s']:.2f} write_s={STATE['write_s']:.2f} write_exc={STATE['failed']} worker={STATE['worker']}")


def digest(fn) -> str:
    import textwrap
    try:
        return hashlib.sha256(textwrap.dedent(inspect.getsource(fn)).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return "unreadable"


# ----------------------------------------------------------------------------------------------------------------- host copies ----
class _T:
    """A leaf that was a torch tensor: travels as a host numpy array, becomes a torch tensor again (torch.from_numpy: same memory, same dtype) where
    the writer runs — so the writer's `isinstance(x, torch.Tensor)` branches take exactly the leaves they took."""
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a


def _host(t):
    """The writer's own first statements on a device tensor, made now: `.cpu()` (a copy to the host) and, for a leaf of rank >= 2 — the
    leaves the writer's `_take_batch_dim` / coordinate statements cast — `.float()`; a rank <= 1 leaf keeps its dtype (the writer hands it on
    as it is). Detached, contiguous. A dtype numpy cannot hold raises here: the caller writes that item with the engine's statements."""
    h = t.detach().cpu()
    if len(h.shape) > 1:
        h = h.float()
    return _T((h if h.is_contiguous() else h.contiguous()).numpy())


def to_host(x):
    import torch
    if isinstance(x, torch.Tensor):
        return _host(x)
    if isinstance(x, dict):
        return {k: to_host(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_host(v) for v in x)
    return x


def from_host(x):
    if isinstance(x, _T):
        import torch
        return torch.from_numpy(x.a)
    if isinstance(x, dict):
        return {k: from_host(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(from_host(v) for v in x)
    return x


BATCH_KEYS = ("atom_array", "seed", "query_id")     # what write_all_outputs reads of the batch when write_features is off


def make_job(seq: int, batch: dict, outputs: dict, confidence_scores: dict):
    b = {k: to_host(batch[k]) for k in BATCH_KEYS if k in batch}
    o = {"atom_positions_predicted": to_host(outputs["atom_positions_predicted"])}
    c = to_host(confidence_scores)
    return (seq, b, o, c)


def run_job(writer, job) -> tuple:
    """(seq, ok, query_ids, error_text, seconds): the engine's write_all_outputs on the job's host copies, timed; never raises."""
    seq, b, o, c = job
    t = time.perf_counter()
    qids = list(b.get("query_id") or [])
    try:
        writer.write_all_outputs(batch=from_host(b), outputs=from_host(o), confidence_scores=from_host(c))
        ok, err = True, None
    except Exception:  # noqa: BLE001 — the engine's own catch-all around this call (its text: logger.exception + failed bookkeeping)
        ok, err = False, traceback.format_exc()
    return (seq, ok, qids, err, time.perf_counter() - t)


def _delta(before: dict, after: dict) -> dict:
    out = {}
    for k, v in after.items():
        b = before.get(k)
        if isinstance(v, (int, float)) and isinstance(b, (int, float)):
            if v != b:
                out[k] = v - b
        elif isinstance(v, dict):
            d = {kk: vv - (b or {}).get(kk, 0) for kk, vv in v.items() if isinstance(vv, (int, float)) and vv != (b or {}).get(kk, 0)}
            if d:
                out[k] = d
    return out


# --------------------------------------------------------------------------------------------------------------------- workers ----
class WorkerDied(RuntimeError):
    pass


class ThreadWorker:
    kind = "thread"

    def __init__(self, writer):
        import queue
        self.writer = writer
        self.q = queue.Queue()
        self.results: list = []
        self.inflight = 0
        self.cv = threading.Condition()
        self.t = threading.Thread(target=self._loop, name="of3opt-writer", daemon=True)
        self.t.start()

    def _loop(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            res = run_job(self.writer, job) + ({},)
            with self.cv:
                self.results.append(res)
                self.inflight -= 1
                self.cv.notify_all()

    def submit(self, job) -> None:
        with self.cv:
            while self.inflight >= QUEUE_MAX:
                if not self.t.is_alive():
                    raise WorkerDied("writer thread is not alive")
                self.cv.wait(timeout=1.0)
            self.inflight += 1
        self.q.put(job)

    def collect(self) -> list:
        with self.cv:
            r, self.results = self.results, []
        return r

    def drain(self) -> list:
        with self.cv:
            while self.inflight > 0:
                if not self.t.is_alive():
                    raise WorkerDied("writer thread is not alive")
                self.cv.wait(timeout=1.0)
        return self.collect()

    def unacknowledged(self) -> list:
        return []                                            # a thread cannot die with the process alive and the job data lost: nothing to re-run

    def close(self) -> None:
        self.q.put(None)
        self.t.join(timeout=60)

    def ident(self) -> str:
        return "thread"


CLOSE_JOIN_S = 30.0                                             # close(): seconds to wait for the child after the end-of-work sentinel ...
TERM_JOIN_S = 10.0                                              # ... and after SIGTERM, before SIGKILL


def _child_main(jobq, resq, snapshots) -> None:                # the writer PROCESS: a fork of the predicting process; touches no device
    import signal
    for name in ("SIGTERM", "SIGHUP"):                          # the fork inherits the predicting process's handlers (Lightning's SIGTERM notifier does not exit):
        try:                                                    # the default disposition back, so the parent's terminate() ends this child whatever it inherited
            signal.signal(getattr(signal, name), signal.SIG_DFL)
        except (AttributeError, OSError, ValueError):
            pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import torch
        torch.set_num_threads(1)                             # no OpenMP team in a forked child (the DataLoader workers' own rule); the child only views host arrays
    except Exception:  # noqa: BLE001
        pass
    writer = _WORKER["writer"]
    while True:
        job = jobq.get()
        if job is None:
            break
        before = {n: f() for n, f in snapshots}
        res = run_job(writer, job)
        delta = {}
        for n, f in snapshots:
            try:
                d = _delta(before[n], f())
            except Exception:  # noqa: BLE001
                d = {}
            if d:
                delta[n] = d
        resq.put(res + (delta,))
    try:
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


class ProcessWorker:
    kind = "process"

    def __init__(self, writer):
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        _WORKER["writer"] = writer                            # read by the child through the fork, not pickled
        self.jobq = ctx.SimpleQueue()
        self.resq = ctx.SimpleQueue()
        self.inflight: Dict[int, tuple] = {}                  # seq -> job, kept until acknowledged (a dead child's items are re-run inline from these)
        self.results: list = []
        snaps = [(n, s) for (n, s, _m) in CHILD_CENSUS]
        self.p = ctx.Process(target=_child_main, args=(self.jobq, self.resq, snaps), name="of3opt-writer", daemon=True)
        self.p.start()

    def _pump(self, block: bool) -> None:
        """Take every available acknowledgement; with block, wait until at least one arrives (polling the child's life once a second)."""
        got = False
        while True:
            ready = self.resq._reader.poll(1.0 if (block and not got) else 0)
            if ready:
                res = self.resq.get()
                self.inflight.pop(res[0], None)
                self.results.append(res)
                got = True
                continue
            if not block or got:
                return
            if not self.p.is_alive():
                raise WorkerDied(f"writer process {self.p.pid} exited with {self.p.exitcode}")

    def submit(self, job) -> None:
        self._pump(block=False)
        while len(self.inflight) >= QUEUE_MAX:
            self._pump(block=True)
        if not self.p.is_alive():
            raise WorkerDied(f"writer process {self.p.pid} exited with {self.p.exitcode}")
        self.inflight[job[0]] = job
        try:
            self.jobq.put(job)                               # pickled here, on the caller's thread (host numpy arrays, the atom arrays, strings)
        except BaseException:
            self.inflight.pop(job[0], None)                  # never queued: the caller writes it inline (not an item in flight)
            raise

    def collect(self) -> list:
        self._pump(block=False)
        r, self.results = self.results, []
        return r

    def drain(self) -> list:
        while self.inflight:
            self._pump(block=True)
        return self.collect()

    def unacknowledged(self) -> list:
        jobs = [self.inflight[k] for k in sorted(self.inflight)]
        self.inflight.clear()
        return jobs

    def close(self) -> None:
        """End of work: the sentinel, then — for a child that does not leave — SIGTERM, then SIGKILL; never an unbounded join."""
        try:
            self.jobq.put(None)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.p.join(timeout=CLOSE_JOIN_S)
            if self.p.is_alive():
                self.p.terminate(); self.p.join(timeout=TERM_JOIN_S)
            if self.p.is_alive():
                self.p.kill(); self.p.join(timeout=TERM_JOIN_S)
        except Exception:  # noqa: BLE001
            pass

    def ident(self) -> str:
        return str(self.p.pid)


# ------------------------------------------------------------------------------------------------------------------ the callback ----
def _apply_results(writer, results: list) -> None:
    """The engine's bookkeeping for every acknowledged item (its on_predict_batch_end text: success_count / failed_count / failed_queries /
    the failure logged), plus the worker's census deltas merged into the cells that own them."""
    import logging
    logger = logging.getLogger(M_WRITER)
    merges = {n: m for (n, _s, m) in CHILD_CENSUS}
    for seq, ok, qids, err, secs, delta in sorted(results, key=lambda r: r[0]):
        if ok:
            writer.success_count += 1
        else:
            writer.failed_count += 1
            writer.failed_queries.extend(qids)
            STATE["failed"] += 1
            logger.error(f"Failed to write predictions for query_id(s) {', '.join(qids)}:\n{err}")
        STATE["write_s"] += secs
        for n, d in (delta or {}).items():
            try:
                merges[n](d)
            except Exception:  # noqa: BLE001
                pass


def _inline(writer, batch, outputs, confidence_scores, reason: str) -> None:
    """The engine's statements, where the engine runs them (its try / except / bookkeeping verbatim); counted by reason."""
    import logging
    logger = logging.getLogger(M_WRITER)
    _count(STATE["inline"], reason)
    t = time.perf_counter()
    try:
        writer.write_all_outputs(batch=batch, outputs=outputs, confidence_scores=confidence_scores)
        writer.success_count += 1
    except Exception as e:  # noqa: BLE001
        writer.failed_count += 1
        writer.failed_queries.extend(batch["query_id"])
        logger.exception(f"Failed to write predictions for query_id(s) {', '.join(batch['query_id'])}: {e}")
    STATE["write_s"] += time.perf_counter() - t


def _rerun_inline(writer, jobs: list, reason: str) -> None:
    for job in jobs:
        seq, b, o, c = job
        _inline(writer, from_host(b), from_host(o), from_host(c), reason)


def _demote(why: str) -> None:
    if STATE["route"] != "sync":
        _log(f"REFUSED mid-run: {why} — the remaining items are written inline (route=sync from here); items in flight were written inline")
        STATE["route"] = "sync"
        STATE["reason"] = ("demoted:" + why.split(":")[0].replace(" ", "_"))[:60]


def _worker_for(writer):
    w = _WORKER["obj"]
    if w is not None and _WORKER["writer"] is writer:
        return w
    if w is not None:                                        # another writer instance (a second predict in one process): finish the first one's items first
        drain(_WORKER["writer"])
        w.close()
    _WORKER["writer"] = writer
    _WORKER["obj"] = ProcessWorker(writer) if STATE["route"] == "process" else ThreadWorker(writer)
    STATE["worker"] = _WORKER["obj"].ident()
    atexit.register(_atexit)                                  # registered AFTER multiprocessing's own exit handler now exists (it terminates and joins daemonic
    _log(f"worker started: route={STATE['route']} worker={STATE['worker']} queue_max={QUEUE_MAX}")   # children): exit handlers run last-in-first-out, so the
    return _WORKER["obj"]                                     # drain + close below always precede it, whatever imported multiprocessing when (idempotent)


def drain(writer=None) -> None:
    """Every submitted item written and acknowledged (or, the worker dead, re-run inline); the results tallied. Idempotent; safe at exit."""
    w = _WORKER["obj"]
    writer = writer or _WORKER["writer"]
    if w is None or writer is None:
        return
    t = time.perf_counter()
    try:
        res = w.drain()
    except WorkerDied as e:
        res = w.collect()
        _apply_results(writer, res)
        res = []
        _rerun_inline(writer, w.unacknowledged(), "worker_died")
        _demote(str(e))
    _apply_results(writer, res)
    STATE["drain_s"] += time.perf_counter() - t


def _is_last_item(trainer, batch_idx, dataloader_idx=0) -> bool:
    """True when Lightning says this is the predict loop's last batch of this dataloader (trainer.num_predict_batches); unknown -> False."""
    try:
        nb = getattr(trainer, "num_predict_batches", None)
        if nb is None:
            return False
        n = nb[dataloader_idx] if isinstance(nb, (list, tuple)) else nb
        n = float(n)
        return n != float("inf") and int(batch_idx) >= int(n) - 1
    except Exception:  # noqa: BLE001
        return False


def make_on_predict_batch_end(orig):
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # -- the engine's text (bookkeeping) ---------------------------------------------------------------------------------
        if batch.get("repeated_sample"):
            return
        self.total_count += 1
        if outputs is None:
            self.failed_count += 1
            self.failed_queries.extend(batch["query_id"])
            return
        batch, outputs = outputs
        confidence_scores = outputs["confidence_scores"]
        # -- the lever ------------------------------------------------------------------------------------------------------
        STATE["items"] += 1
        w = _WORKER["obj"]
        if w is not None:
            _apply_results(self, w.collect())                  # acknowledgements of earlier items, tallied as they arrive
        reason = None
        if STATE["route"] == "sync":
            reason = "route_sync" if not STATE["reason"] else "demoted"
        elif getattr(self, "write_features", False) or getattr(self, "write_latent_outputs", False):
            reason = "features_or_latent"
        elif int(getattr(trainer, "world_size", 1) or 1) > 1:
            reason = "world_size"
        elif _is_last_item(trainer, batch_idx, dataloader_idx):   # nothing runs after the last item: its files are written here, by the engine's statements
            reason = "last_item"                               # (a worker would only add the fork's copy-on-write and a drain at the end)
        if reason is not None:
            _inline(self, batch, outputs, confidence_scores, reason)
            del batch, outputs
            return
        t0 = time.perf_counter()
        try:
            from pathlib import Path
            _WORKER["seq"] += 1
            job = make_job(_WORKER["seq"], batch, outputs, confidence_scores)
            for b in range(len(batch["atom_array"])):         # the writer's mkdir, now (PredictTimer writes timing.json into it right after this callback)
                (Path(self.output_dir) / batch["query_id"][b] / f"seed_{batch['seed'][b]}").mkdir(parents=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001 — a batch / outputs form this port does not know: the engine's statements, inline, counted
            STATE["host_copy_s"] += time.perf_counter() - t0
            _inline(self, batch, outputs, confidence_scores, "host_copy:" + type(e).__name__)
            del batch, outputs
            return
        t1 = time.perf_counter()
        STATE["host_copy_s"] += t1 - t0
        try:
            _worker_for(self).submit(job)
            STATE["overlapped"] += 1
        except WorkerDied as e:
            w = _WORKER["obj"]
            _apply_results(self, w.collect() if w is not None else [])
            _rerun_inline(self, (w.unacknowledged() if w is not None else []), "worker_died")
            _demote(str(e))
            _rerun_inline(self, [job], "worker_died")
        except Exception as e:  # noqa: BLE001 — the worker could not be started or the job not queued: this item inline, the route demoted
            _demote(f"{type(e).__name__}: {e}")
            _rerun_inline(self, [job], "submit_error")
        STATE["wait_s"] += time.perf_counter() - t1
        del batch, outputs
    on_predict_batch_end.__wrapped__ = orig
    setattr(on_predict_batch_end, MARK, True)
    return on_predict_batch_end


def _retire() -> None:
    """The predict is over: stop the worker (its items are drained by the caller first)."""
    w = _WORKER["obj"]
    if w is not None:
        try:
            w.close()
        finally:
            _WORKER.update(obj=None, writer=None)


def make_on_predict_end(orig):
    def on_predict_end(self, trainer, pl_module):
        drain(self)                                            # every item written and tallied before the engine composes its summary
        _retire()                                              # and the worker stopped here, not at interpreter exit
        return orig(self, trainer, pl_module)
    on_predict_end.__wrapped__ = orig
    setattr(on_predict_end, MARK, True)
    return on_predict_end


def _atexit() -> None:
    if _ATEXIT.get("ran"):                                     # registered twice by design (install, and again after the worker starts): runs once
        return
    _ATEXIT["ran"] = True
    try:
        drain()
        _retire()
    except Exception:  # noqa: BLE001
        pass
    sys.stderr.write(census_line() + "\n")


def patch_writer_module(mod) -> bool:
    cls = getattr(mod, WRITER_CLASS, None)
    if cls is None or not hasattr(cls, "on_predict_batch_end") or not hasattr(cls, "write_all_outputs"):
        STATE.update(state="refused", reason=f"no_writer:{M_WRITER}.{WRITER_CLASS}")
        _log(f"REFUSED: {M_WRITER}.{WRITER_CLASS}.on_predict_batch_end / write_all_outputs not found — the engine's writer runs as it is")
        return False
    if getattr(cls.on_predict_batch_end, MARK, None):
        STATE.update(patched=1, state="on", reason="")
        return True
    d = digest(cls.on_predict_batch_end)
    STATE["digest"] = {"on_predict_batch_end": d}
    want = DIGESTS.get("on_predict_batch_end") or ()
    if want and d not in want:
        STATE.update(state="refused", reason=f"digest:on_predict_batch_end={d}")
        _log(f"REFUSED: {M_WRITER}.{WRITER_CLASS}.on_predict_batch_end is not the text this lever re-states (sha256[:16] {d}, accepted {'|'.join(want)}) — the engine's writer runs as it is")
        return False
    ORIG["on_predict_batch_end"], ORIG["on_predict_end"] = cls.on_predict_batch_end, cls.on_predict_end
    cls.on_predict_batch_end = make_on_predict_batch_end(cls.on_predict_batch_end)
    cls.on_predict_end = make_on_predict_end(cls.on_predict_end)
    STATE.update(patched=1, state="on", reason="")
    _log(f"installed: {M_WRITER}.{WRITER_CLASS}.on_predict_batch_end -> the item's files rendered and written by one writer worker while the next item runs "
         f"(route={STATE['route']}, queue_max={QUEUE_MAX}; same statements, same bytes)")
    return True


class _WriterOverlapFinder:
    """Meta-path finder that patches the writer module right after its body runs; delegates the find to the other finders. Re-entrant by
    name: another finder for the same module (fastjson's) that delegates back to this one while this one is delegating gets None, so the
    two compose (each wraps the loader's exec_module once, in meta-path order) instead of recursing."""

    def __init__(self):
        self._busy = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname != M_WRITER or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self or type(finder).__name__ == type(self).__name__:
                    continue
                try:
                    spec = finder.find_spec(fullname, path, target)
                except Exception:  # noqa: BLE001
                    spec = None
                if spec is not None:
                    break
        finally:
            self._busy.discard(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            try:
                patch_writer_module(module)
            except Exception as e:  # noqa: BLE001
                STATE.update(state="refused", reason=f"patch_error:{type(e).__name__}")
                _log(f"REFUSED: patching {M_WRITER} raised {type(e).__name__}: {e} — the engine's writer runs as it is")
        spec.loader.exec_module = exec_module
        return spec


FINDER = _WriterOverlapFinder()


def install(environ=None) -> dict:
    """Idempotent. Not requested: nothing. Requested: the route knob read, the writer class patched now when its module is imported, else the
    ONE finder armed for its import; drain + census registered at exit."""
    environ = os.environ if environ is None else environ
    if STATE["installed"]:
        return STATE
    try:
        if not requested(environ):
            return STATE
        if not M_WRITER:
            raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_WRITER=) before install")
        STATE["route"] = route(environ)
    except ValueError as e:                                   # a switch word the cell does not know: the lever steps aside BY NAME, the engine's writer runs as it is
        STATE.update(installed=True, state="refused", reason="bad_word:" + str(e).split("=", 1)[0])
        _log(f"REFUSED: {e} — the engine's writer runs as it is")
        if not _ATEXIT["registered"]:
            atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
            _ATEXIT["registered"] = True
        return STATE
    STATE.update(installed=True, state="armed", reason="")
    mod = sys.modules.get(M_WRITER)
    if mod is not None:
        patch_writer_module(mod)
    else:
        sys.meta_path[:] = [f for f in sys.meta_path if f is FINDER or type(f).__name__ != type(FINDER).__name__]
        if FINDER not in sys.meta_path:
            sys.meta_path.insert(0, FINDER)
        _log(f"armed: {M_WRITER}.{WRITER_CLASS} is patched when the engine imports its writer (route={STATE['route']})")
    if not _ATEXIT["registered"]:
        atexit.register(_atexit)
        _ATEXIT["registered"] = True
    return STATE


def uninstall() -> None:
    """Drain, stop the worker, disarm the finder and put the engine's methods back (tests; a process keeps one route)."""
    try:
        drain()
        w = _WORKER["obj"]
        if w is not None:
            w.close()
    except Exception:  # noqa: BLE001
        pass
    _WORKER.update(obj=None, writer=None)
    try:
        sys.meta_path.remove(FINDER)
    except ValueError:
        pass
    mod = sys.modules.get(M_WRITER)
    cls = getattr(mod, WRITER_CLASS, None) if mod is not None else None
    if cls is not None and ORIG:
        for k, v in ORIG.items():
            setattr(cls, k, v)
    ORIG.clear()
    _ATEXIT["ran"] = False
    STATE.update(installed=False, state="off", reason="", patched=0)
