# hl_levers.py — the host-side levers of BoltzGen's design step (environment-driven; HL_ASYNC_WRITER unset = nothing patched).
#
#   HL_ASYNC_WRITER=1 : `async_writer` — upstream's DesignWriter.write_on_batch_end (task/predict/writer.py: per design of the batch, two
#                     Structure.from_feat, two mmCIF renderings, one compressed npz — pure host work after the unit's tensors exist) runs OFF
#                     the sampler's critical path: at each batch end the prediction and batch trees are copied to host memory in ONE pass
#                     (opt_core.host.outputs.to_host: the same bytes, one synchronisation) and the STOCK writer body — the very function upstream
#                     ships, called unchanged on those host tensors — is handed to the shared core's background writer
#                     (opt_core.host.outputs.AsyncWriter, `fork` mode: WORKERS = 4 worker processes forked at the
#                     first batch end, which never touch the device and render in parallel off this process's GIL (the DataLoader-worker
#                     pattern); at most PENDING = 5 batches outstanding — the producer blocks rather than buffers). The next unit's featurisation and sampling proceed while
#                     the previous unit's files are rendered and written; the worker receives the writer object, the host tensors and the ONE
#                     trainer attribute the body reads (datamodule.cfg.multiplicity) pickled at submit. Every write is accounted: the writer is
#                     drained (all files on disk, the first failure re-raised) at the predict loop's end (Callback.on_predict_end) — before the
#                     pipeline step returns and before any later step reads the directory — and the core's exit guard turns a write still pending
#                     or failed at interpreter exit into exit status 70 with one TALLY line, never a silently missing design. Exact: the file
#                     set, names and bytes are the synchronous writer's (same code, same tensors, host arithmetic on 0/1 masks only).
#
# Census (fail-loud; read by boltzgen_opt, stack.RUNNER_LINES): `[hl_levers] async_writer installed ...` at install, the core writer's own
# LEVER line (`WRITER.active_line`), and at interpreter exit the core's TALLY line (submitted / written / failed) plus
# `[hl_levers] async_writer stats: {json}`; a submitted count that does not equal the written count, or any failed write, prints
# `[hl_levers] async_writer GATE-FAIL ...` (stack.RUNNER_DISABLED_LINES: the run is partial, exit 3) on top of the core's own exit status.
import atexit, json, os, sys, threading, time
from pathlib import PurePath
from opt_core import report as core_report
from opt_core.host.outputs import AsyncWriter, to_host

TAG = "hl_levers"
STRATEGY = "F6.output_overlap"
MODE, WORKERS, PENDING = "fork", 4, 5                    # the background writer: forked worker processes, and the back-pressure bound (batches outstanding)
STATS = {"writer_enabled": False, "writer_site": None, "mode": None, "workers": None, "max_pending": None, "batches": 0, "designs": 0, "to_host_s": 0.0,
         "submit_wait_s": 0.0, "drain_s": 0.0, "drains": 0, "to_host_leaves": 0, "to_host_bytes": 0, "gate_problems": [],
         "batches_failed_upstream": 0, "view_dropped": []}
WRITER = None                                   # the core AsyncWriter (created at install when the lever is on)
_INSTALLED = False
_LOCK = threading.Lock()


def _log(msg):
    print(f"[{TAG}] {msg}", file=sys.stderr, flush=True)


class _Cfg:
    """The one trainer attribute upstream's writer body reads (`trainer.datamodule.cfg.multiplicity`), carried to a worker process by value."""
    def __init__(self, multiplicity):
        self.multiplicity = multiplicity


class _TrainerView:
    def __init__(self, multiplicity):
        self.datamodule = _Cfg.__new__(_Cfg); self.datamodule.cfg = _Cfg(multiplicity)


_WORKER_READY = False
_REPORTED = False
_STOCK_WOBE = None                                     # upstream's DesignWriter.write_on_batch_end as found at install()


_PLAIN = (str, bytes, int, float, bool, type(None), PurePath)


def _view_of(writer):
    """A plain-data snapshot of the writer object for the worker: its class (by module and name) and every attribute whose value is plain
    data (paths, strings, numbers, flags, sets/lists/tuples of those). Anything else — e.g. a wrapper another module bound onto the
    instance — is not shipped (named in STATS["view_dropped"]); the stock body reads only plain attributes (outdir, mol_dir, the format
    flags, file_suffix) and raises by name if one it needs were missing."""
    keep, dropped = {}, []
    for k, v in vars(writer).items():
        if isinstance(v, _PLAIN) or (isinstance(v, (set, frozenset, list, tuple)) and all(isinstance(x, _PLAIN) for x in v)):
            keep[k] = v
        else:
            dropped.append(k)
    cls = type(writer)
    return {"cls": (cls.__module__, cls.__qualname__), "attrs": keep}, dropped


def _from_view(view):
    if not isinstance(view, dict) or "cls" not in view:                    # the thread form hands the live writer object over as is
        return view
    import importlib
    modname, qual = view["cls"]
    cls = getattr(importlib.import_module(modname), qual)
    obj = cls.__new__(cls)
    obj.__dict__.update(view["attrs"])
    return obj


def _write_item(item):
    """The background write: upstream's own DesignWriter.write_on_batch_end, unchanged, on host tensors — in a worker process (a fork of the
    design process, or a fresh spawned interpreter) or the writer thread; one intra-op thread per worker process so W workers do not
    oversubscribe the host. The item carries data only (the writer object, the call's arguments): the stock body is this module's record of
    it (``_STOCK_WOBE``, set by install() before any worker is forked; inherited by a fork, the same process for the thread) or, in a spawned
    interpreter where install() never runs, the class's own unpatched method."""
    global _WORKER_READY
    view, args = item
    writer_self = _from_view(view)
    if not _WORKER_READY:
        _WORKER_READY = True
        import multiprocessing
        if multiprocessing.current_process().name != "MainProcess":          # a writer process (forked or spawned): background work — one intra-op thread, lowered priority, so W writers never slow the design process's host side
            try:
                os.nice(10)
            except Exception:
                pass
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:
                pass
    body = _STOCK_WOBE
    if body is None:                                                          # a spawned worker: the class here is upstream's, unpatched
        body = type(writer_self).write_on_batch_end
        if getattr(body, "_hl_levers", False):
            raise RuntimeError(f"{TAG}: async_writer worker has the patched writer class but no record of the stock body (install() ran in a worker?)")
    return body(writer_self, *args)


def install():
    """Patch boltzgen.task.predict.writer.DesignWriter: write_on_batch_end -> one to_host pass + submit of the stock body to the core's
    AsyncWriter; on_predict_end -> drain. The target is checked present first (a missing one raises by name)."""
    global _INSTALLED, WRITER
    if _INSTALLED:
        return
    if os.environ.get("HL_ASYNC_WRITER", "0") != "1":
        _INSTALLED = True; return
    global _STOCK_WOBE
    import boltzgen.task.predict.writer as W
    cls = W.DesignWriter
    stock = cls.write_on_batch_end
    if getattr(stock, "_hl_levers", False):
        _INSTALLED = True; return
    _STOCK_WOBE = stock                                                       # the record the background write calls (inherited by forked workers; a spawned one reads its own unpatched class)
    for name in ("write_on_batch_end", "init_outdir"):
        if name not in cls.__dict__:
            raise AttributeError(f"{TAG}: async_writer: DesignWriter no longer defines {name} — version mismatch, refusing to patch")
    try:
        import multiprocessing as _mp
        if _mp.parent_process() is not None:                                       # a spawned worker re-importing the runner: the lever lives in the design process only
            _INSTALLED = True; return
    except Exception:
        pass
    mode, workers, pending = MODE, WORKERS, PENDING
    here = os.path.dirname(os.path.abspath(__file__))                              # a worker unpickles `hl_levers._write_item` by module name: this directory must be importable there
    pp = os.environ.get("PYTHONPATH", "")
    if here not in pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = here + (os.pathsep + pp if pp else "")
    WRITER = AsyncWriter(_write_item, workers=workers, max_pending=pending, name="design_writer", mode=mode)
    STATS.update(writer_enabled=True, writer_site="task.predict.writer.DesignWriter.write_on_batch_end", mode=mode, workers=workers, max_pending=pending)

    def write_on_batch_end(self, trainer=None, pl_module=None, prediction=None, batch_indices=None, batch=None, batch_idx=None,
                           dataloader_idx=0, sample_id=None):
        if prediction is not None and prediction.get("exception"):       # the stock body's first statement, kept in this process so the writer's own `failed` count stays true
            self.failed += 1
            with _LOCK:
                STATS["batches_failed_upstream"] += 1
            return
        t0 = time.perf_counter()
        st = {}
        pred_h = to_host(prediction, stats=st)                    # ONE device->host pass per tree (same bytes); the stock body then runs on host tensors
        batch_h = to_host(batch, stats=st)
        t1 = time.perf_counter()
        try:
            n = int(prediction["coords"].shape[0]) if prediction is not None and "coords" in prediction else 0
        except Exception:
            n = 0
        if mode in ("fork", "spawn"):                                          # by value to a worker process: the writer object, the host trees, the one trainer attribute the body reads
            tv = _TrainerView(getattr(getattr(getattr(trainer, "datamodule", None), "cfg", None), "multiplicity", 1))
            view, dropped = _view_of(self)
            if dropped:
                with _LOCK:
                    STATS["view_dropped"] = sorted(set(STATS["view_dropped"]) | set(dropped))
            WRITER.submit((view, (tv, None, pred_h, batch_indices, batch_h, batch_idx, dataloader_idx, sample_id)))
        else:
            WRITER.submit((self, (trainer, pl_module, pred_h, batch_indices, batch_h, batch_idx, dataloader_idx, sample_id)))
        t2 = time.perf_counter()
        with _LOCK:
            STATS["batches"] += 1; STATS["designs"] += n
            STATS["to_host_s"] += t1 - t0; STATS["submit_wait_s"] += t2 - t1
            STATS["to_host_leaves"] += int(st.get("leaves", 0) or 0); STATS["to_host_bytes"] += int(st.get("bytes", 0) or 0)
    write_on_batch_end._hl_levers = True
    cls.write_on_batch_end = write_on_batch_end

    prev_end = cls.__dict__.get("on_predict_end")
    def on_predict_end(self, trainer, pl_module):
        t0 = time.perf_counter()
        try:
            WRITER.drain()                                        # every file of this predict loop on disk; the first failed write re-raised here, by name
        except Exception:
            with _LOCK:
                STATS["drain_s"] += time.perf_counter() - t0; STATS["drains"] += 1
            report_lines()                                        # the census of the failure (GATE-FAIL, TALLY, stats) printed now: the core's exit guard may end the process before atexit handlers run
            raise
        with _LOCK:
            STATS["drain_s"] += time.perf_counter() - t0; STATS["drains"] += 1
        if prev_end is not None:
            return prev_end(self, trainer, pl_module)
        up = getattr(super(cls, self), "on_predict_end", None)
        return up(trainer, pl_module) if up is not None else None
    on_predict_end._hl_levers = True
    cls.on_predict_end = on_predict_end
    _INSTALLED = True
    print(f"[{TAG}] async_writer installed core=opt_core.host.outputs strategy={STRATEGY} class=exact mode={mode} workers={workers} max_pending={pending} "
          f"site=DesignWriter.write_on_batch_end drain=on_predict_end", flush=True)
    print(WRITER.active_line(TAG), flush=True)


def report() -> dict:
    out = dict(STATS)
    out["writer"] = WRITER.tally() if WRITER is not None else None
    return out


def gate() -> list:
    problems = []
    if WRITER is not None:
        t = WRITER.tally()
        sub, wr, fl = int(t.get("submitted", 0)), int(t.get("written", 0)), int(t.get("failed", 0))
        if fl:
            problems.append(f"async_writer: {fl} write(s) failed")
        if sub != wr + fl:
            problems.append(f"async_writer: submitted={sub} written={wr} failed={fl} do not close (undrained)")
        if STATS["batches"] != sub:
            problems.append(f"async_writer: batches seen={STATS['batches']} submitted={sub}")
    return problems


def evidence_line() -> str:
    t = WRITER.tally() if WRITER is not None else {}
    pairs = [("site", "DesignWriter.write_on_batch_end"), ("mode", STATS["mode"]), ("workers", STATS["workers"]), ("batches", STATS["batches"]), ("designs", STATS["designs"]),
             ("submitted", t.get("submitted", 0)), ("written", t.get("written", 0)), ("failed", t.get("failed", 0))]
    return core_report.lever_line(TAG, f"{TAG}.async_writer", "on", *pairs, impl="opt_core.host.outputs:AsyncWriter", origin="core", strategy=STRATEGY)


def report_lines():
    global _REPORTED
    if not STATS["writer_enabled"] or _REPORTED:
        return
    _REPORTED = True
    try:
        if WRITER is not None and WRITER.pending():                   # a predict loop that ended without on_predict_end (an exception mid-loop): drain here so the files land and the tally is final
            t0 = time.perf_counter()
            try:
                WRITER.drain()
            except Exception as e:                                    # the failure is in the tally and the GATE-FAIL line; the core's guard sets the exit status
                _log(f"async_writer drain at exit raised {type(e).__name__}: {str(e).splitlines()[0][:200]}")
            STATS["drain_s"] += time.perf_counter() - t0; STATS["drains"] += 1
        print(evidence_line(), flush=True)
        if WRITER is not None:
            print(WRITER.tally_line(TAG), flush=True)
            t = WRITER.tally()
            print(f"[{TAG}] CENSUS async_writer[W={STATS['workers']},mode={STATS['mode']},submitted={t.get('submitted', 0)},written={t.get('written', 0)},failed={t.get('failed', 0)}]", flush=True)
        probs = gate(); STATS["gate_problems"] = probs
        rep = report()
        print(f"[{TAG}] async_writer stats: {json.dumps({k: rep[k] for k in ('mode', 'workers', 'batches', 'designs', 'batches_failed_upstream', 'view_dropped', 'to_host_s', 'submit_wait_s', 'drain_s', 'drains', 'to_host_leaves', 'to_host_bytes', 'max_pending', 'writer')}, sort_keys=True, default=str)}", flush=True)
        if probs:
            print(f"[{TAG}] async_writer GATE-FAIL {' | '.join(probs)}", flush=True)
    except Exception as e:                                            # the exit report must never mask the process's own status
        _log(f"report_lines failed: {type(e).__name__}: {e}")


install()
atexit.register(report_lines)                          # registered after the core writer's own exit guard (install() constructs it), so these lines print first (atexit is LIFO) even when that guard forces a loss status
