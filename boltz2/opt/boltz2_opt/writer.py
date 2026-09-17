"""boltz2_opt.writer — the output writer off the critical path (registry: ``writer_overlap``; switch ``BOLTZ_WRITER=overlap``).

What it replaces. After every prediction the pipelined worker (``bz_worker_lev*.py --pipeline 1``) calls boltz's own
``BoltzWriter.write_on_batch_end`` (boltz/data/write/writer.py) in the predicting thread: the result tensors are read back from the device one
``.cpu()`` / ``.item()`` at a time (the mmCIF formatter alone reads every residue's pLDDT with ``.item()`` twice: 2N+ synchronising
device-to-host copies per structure), the mmCIF text is built in Python, ``pae`` / ``pde`` / ``plddt`` are DEFLATE-compressed
(``np.savez_compressed``), the confidence JSON written, and the worker then moves the files under ``by_seed/<name>/s<seed>/``. All of it
runs between item k's forward and item k+1's forward: 0.24 s per item at 400 tokens … 0.86-0.89 s at 1,200 (H100), 0.5 … 1.8 s on A100 —
more than the confidence module at every size.

What it does instead. The SAME writer function on the SAME values, elsewhere:

  * in the predicting thread, right where stock calls the writer, every tensor the writer reads (``coords``, ``masks``, ``plddt``, ``pae``,
    ``pde``, the nine confidence scalars, ``pair_chains_iptm``; ``s`` / ``z`` only when embeddings are written) is copied device-to-host
    into PINNED buffers with ``non_blocking=True`` on a side stream that waits on the compute stream (``Tensor.record_stream`` keeps the
    allocator from reusing the sources early), one CUDA event recorded behind the copies — microseconds of host time, no synchronisation;
  * a light consumer thread waits for that event (outside the GIL), and hands the host copies to ONE writer process — a grandchild of the
    kit's pre-CUDA parsing zygote (``boltz2_opt.prep``, the lineage the persistent featurizer uses: no CUDA context, no hook, no page
    shared with the predicting process; reached over one unix-socket connection) — which runs boltz's ``BoltzWriter.write_on_batch_end``
    VERBATIM on those CPU tensors (``.cpu()`` is then the identity, ``.item()`` a host read, ``np.savez_compressed`` / ``to_mmcif`` /
    ``json.dumps`` the very calls, same dtypes, same order) and then performs the worker's own move statements (``glob`` + ``shutil.move``
    under ``by_seed/<name>/s<seed>/``). Item k's files are written while item k+1 predicts; the writer process holds no GIL of the
    predicting process, so the forward's launch loop is untouched (a writer THREAD would contend for it: ``BOLTZ_WRITER=overlap,thread`` is
    the diagnostic word that runs the same function in the consumer thread instead — measured, not shipped).

Bytes. Exact-class by construction: no arithmetic moves, only WHERE the host reads happen. The one statement whose result could depend on
the device it runs on is the ranking ``torch.argsort(confidence_score, descending=True)`` at ``diffusion_samples > 1`` with two samples of
bit-identical score (tie order is implementation-defined): such an item — and a NaN score — is written inline by the stock call, by name
(``inline_by=tied_scores:<n>``); with distinct scores the descending order is unique, so every implementation returns the same permutation.
A prediction boltz itself skipped (``{"exception": True}``) takes the stock call inline (it writes nothing and counts ``failed``).

Fail-closed. ``join()`` is mandatory before the worker reports: every pending write / move has landed or is FAILED BY NAME — the worker
records the (input, seed) unit under ``failed`` with the writer's reason (the parent prints ``[boltz2-opt] FAILED item=… seed=… reason=…``)
and exits non-zero; a helper that dies fails the units it held and every later item is written inline by the stock call
(``fallback_by=helper_died:<n>``); the gate refuses on any failure or fallback. An affinity input waits for its structure files before
upstream's affinity leg runs (the leg reads ``pre_affinity_<id>.npz``): ``wait_item``. ``--pipeline 0`` (Lightning's own loop) and a
process without the zygote leave the stock call in place, by name (``skipped_by_name:pipeline0`` / ``no_zygote``). At ``--n_gpu`` > 1 the
row takes the lever off by name (modes.TP_DROPS: the xP line is not measured with it).

LEVER line (stderr at exit; ``writer_report`` in the worker log)::

    [boltz2-opt] LEVER name=writer_overlap state=on impl=boltz2_opt.writer origin=kit strategy=LOCAL.boltz2.writer_overlap backend=process
                 items=<n> served=<n> inline=<n> [inline_by=<reason:n,…>] fallback=<n> [fallback_by=…] failed=<n> moves=<n> queued_max=<n>
                 join_s=<s> bytes_written=<b> stage_ms_max=<ms> wait_ms_total=<ms> write_s_total=<s> helper_pid=<pid> helper_lineage=zygote
                 forked_before_cuda=<bool> execution=executed:<n>

Contract (worker_launch): ``LEVERS``; ``apply()`` -> levers installed; ``dispositions()``; ``report()``. Worker-side calls (bz_worker_lev*.py):
``begin_item(name, seed)`` before the prediction, ``finish_item(name, seed, pred_dir, dst)`` where the worker moved the files, ``wait_item(name)``
before the affinity leg, ``collect()`` after each item, ``join()`` before the log is final; all of them are the stock statements when the
lever is not installed.
"""
from __future__ import annotations

import glob
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

TAG = "boltz2-opt"
SWITCH = "BOLTZ_WRITER"
WORD = "overlap"
THREAD = "thread"                      # diagnostic backend word: the consumer thread writes (GIL shared with the forward's launch loop)
LEVERS = ("writer_overlap",)
STRATEGY = "LOCAL.boltz2.writer_overlap"
IMPL = "boltz2_opt.writer"
WRITER_MODULE = "boltz.data.write.writer"
SPAWN = "boltz2_opt.writer:spawn"      # the zygote target that detaches the writer process (below)
MAX_PENDING = 4                        # write jobs in flight before the predicting thread waits for the oldest (back-pressure, counted: wait_ms_total)
READ_KEYS = ("coords", "masks", "confidence_score", "plddt", "pae", "pde", "ptm", "iptm", "ligand_iptm", "protein_iptm",
             "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde", "pair_chains_iptm")   # what BoltzWriter.write_on_batch_end reads (boltz 2.2.1 writer.py:57-231)
EMBED_KEYS = ("s", "z")                # read only when the writer was built with write_embeddings (writer.py:233-241)

STATE: Dict[str, Any] = {
    "installed": False, "disposition": None, "backend": None, "orig": None,
    "helper": None, "helper_pid": None, "forked_before_cuda": None, "helper_cuda_initialized": None,
    "items": 0,              # write_on_batch_end calls this module took (submitted + inline)
    "submitted": 0,          # write jobs handed to the background
    "served": 0,             # write jobs that landed
    "inline": 0, "inline_by": {},        # stock call inline BY NAME (exception / tied_scores): correct bytes, no overlap, not a failure
    "fallback": 0, "fallback_by": {},    # stock call inline because the background is gone (helper_died, stage_error): the gate refuses
    "failed": 0, "errors": [],           # background writes / moves that raised: the unit is failed by name
    "moves": 0, "moves_done": 0,
    "queued_max": 0, "join_s": 0.0, "bytes_written": 0, "stage_ms_max": 0.0, "stage_ms_total": 0.0,
    "wait_ms_total": 0.0, "write_s_total": 0.0, "move_s_total": 0.0, "d2h_wait_ms_total": 0.0,
    "dead": None,            # reason the background died (every later item inline, fallback_by)
    "current": None,         # (name, seed) the worker is predicting (begin_item)
    "per_item": [],
    "joined": False,
}
_LOCK = threading.Lock()
_COND = threading.Condition(_LOCK)
_PENDING: Dict[str, int] = {}          # name -> background jobs (write or move) outstanding
_INFLIGHT: List[dict] = []             # write jobs not yet done (back-pressure order)
_DONE: List[dict] = []                 # finished jobs the worker has not collected
_Q: "queue.Queue" = queue.Queue()
_THREAD: Optional[threading.Thread] = None
_STREAMS: Dict[Any, Any] = {}          # device -> side stream
_POOL: Dict[tuple, List[Any]] = {}     # (device-free) (dtype, shape) -> free pinned host tensors
_SEQ = [0]


def _say(*words) -> None:
    sys.stderr.write(f"[{TAG} writer] " + " ".join(str(w) for w in words) + "\n"); sys.stderr.flush()


def _env(environ=None):
    return os.environ if environ is None else environ


def words(environ=None) -> List[str]:
    return [w.strip() for w in (_env(environ).get(SWITCH) or "").split(",") if w.strip()]


def requested(environ=None) -> bool:
    return WORD in words(environ)


def levers_of(environ=None) -> List[str]:
    return list(LEVERS) if requested(environ) else []


def problems(environ=None) -> List[str]:
    """Unknown words are a refusal by name (apply raises)."""
    bad = [w for w in words(environ) if w not in (WORD, THREAD)]
    out = [f"{SWITCH}: unknown word(s) {','.join(bad)} (known: {WORD}[,{THREAD}])"] if bad else []
    if THREAD in words(environ) and WORD not in words(environ):
        out.append(f"{SWITCH}={THREAD} needs {WORD} ({SWITCH}={WORD},{THREAD})")
    return out


def backend_of(environ=None) -> str:
    return "thread" if THREAD in words(environ) else "process"


# ----------------------------------------------------------------------------------------------------------------- the write itself (helper or thread)
_WRITERS: Dict[tuple, Any] = {}


def _original():
    """boltz's own BoltzWriter.write_on_batch_end (the function this module displaced in the worker; the module attribute in the helper)."""
    if STATE.get("orig") is not None:
        return STATE["orig"]
    import importlib
    return importlib.import_module(WRITER_MODULE).BoltzWriter.write_on_batch_end


def _writer_for(cfg: dict):
    key = tuple(sorted(cfg.items()))
    w = _WRITERS.get(key)
    if w is None:
        import importlib
        W = importlib.import_module(WRITER_MODULE).BoltzWriter
        w = W(data_dir=cfg["data_dir"], output_dir=cfg["output_dir"], output_format=cfg["output_format"], boltz2=cfg["boltz2"], write_embeddings=cfg["write_embeddings"])
        _WRITERS[key] = w
    return w


def _to_numpy(x):
    """Host tensors -> numpy views (zero copy; what crosses the socket to the helper)."""
    import torch
    if isinstance(x, torch.Tensor):
        return ("t", x.detach().numpy())
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    return x


def _to_torch(x):
    import numpy as np
    import torch
    if isinstance(x, tuple) and len(x) == 2 and x[0] == "t" and isinstance(x[1], np.ndarray):
        return torch.from_numpy(x[1])
    if isinstance(x, dict):
        return {k: _to_torch(v) for k, v in x.items()}
    return x


def do_write(cfg: dict, prediction: dict, records: list) -> dict:
    """Run boltz's write_on_batch_end on host tensors (helper process or consumer thread). Returns {write_s, files: {rel: bytes}}."""
    w = _writer_for(cfg)
    t = time.perf_counter()
    _original()(w, None, None, prediction, None, {"record": records}, 0, 0)
    dt = time.perf_counter() - t
    files = {}
    for r in records:
        d = os.path.join(cfg["output_dir"], r.id)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    files[f"{r.id}/{f}"] = os.path.getsize(p)
    return {"write_s": round(dt, 4), "files": files}


def do_move(pred_dir: str, dst: str) -> dict:
    """The worker's own statements (bz_worker_lev*.py): every file under predictions/<name>/ moved under by_seed/<name>/s<seed>/."""
    t = time.perf_counter()
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(pred_dir, "*")))
    nbytes = 0
    for f in files:
        try:
            nbytes += os.path.getsize(f)
        except OSError:
            pass
        shutil.move(f, os.path.join(dst, os.path.basename(f)))
    return {"files": [os.path.basename(f) for f in files], "bytes": nbytes, "move_s": round(time.perf_counter() - t, 4)}


# ----------------------------------------------------------------------------------------------------------------- the helper process (zygote lineage)
def _helper_main(address: str, ready_fd: int) -> None:
    """Bind, accept the worker's ONE connection, serve ("write", seq, cfg, payload, records) / ("move", seq, pred_dir, dst) -> ("ok", seq, info) |
    ("err", seq, "Type: message", traceback) until None / EOF. No CUDA, no model, no hook: boltz's writer module as imported by the zygote."""
    import multiprocessing
    from multiprocessing.connection import Listener
    import torch
    key = multiprocessing.current_process().authkey
    with Listener(address, family="AF_UNIX", backlog=1, authkey=key) as listener:
        sys.stdout.write(f"ready pid={os.getpid()} cuda_initialized={torch.cuda.is_initialized()}\n"); sys.stdout.flush()
        os.close(ready_fd)
        conn = listener.accept()
    conn.send(("hello", os.getpid(), bool(torch.cuda.is_initialized()), torch.__version__))
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            break
        if job is None:
            break
        kind, seq = job[0], job[1]
        try:
            if kind == "write":
                _, _, cfg, payload, records = job
                info = do_write(cfg, _to_torch(payload), records)
            elif kind == "move":
                _, _, pred_dir, dst = job
                info = do_move(pred_dir, dst)
            else:
                raise ValueError(f"unknown job kind {kind!r}")
            conn.send(("ok", seq, info))
        except Exception as e:  # noqa: BLE001
            try:
                conn.send(("err", seq, f"{type(e).__name__}: {e}", traceback.format_exc()))
            except Exception:  # noqa: BLE001
                break
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


def spawn(address: str) -> None:
    """Zygote target (``prep.zygote().run(SPAWN, address=…)``): in a FRESH child of the zygote, detach the writer process (double fork, new session,
    stdout/stderr back on the real descriptors) and return once its listener is bound (prints ``helper pid=<pid>``)."""
    r, w = os.pipe()
    sys.stdout.flush(); sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            os.close(r)
            os.setsid()
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            _helper_main(address, w)
        except BaseException:
            code = 1
            try:
                sys.__stderr__.write(f"[{TAG} writer] helper died:\n{traceback.format_exc()}"); sys.__stderr__.flush()
            except Exception:  # noqa: BLE001
                pass
        finally:
            os._exit(code)
    os.close(w)
    with os.fdopen(r) as fh:
        fh.read()
    print(f"helper pid={pid}")


class Helper:
    """The worker's handle on the writer process: spawned through the zygote, one unix-socket connection; used by the consumer thread only."""

    def __init__(self):
        import multiprocessing
        import tempfile
        from multiprocessing.connection import Client
        from . import prep
        z = prep.zygote()
        if z is None or not hasattr(z, "run"):
            raise RuntimeError("no_zygote")
        d = z.describe() if hasattr(z, "describe") else {}
        if "zygote_pid" in d and d.get("zygote_pid") is None:
            z.start()
            d = z.describe()
        self.zygote_cuda_initialized = d.get("cuda_initialized_at_fork")
        self.dir = tempfile.mkdtemp(prefix="bz2wr_")
        self.address = os.path.join(self.dir, "s")
        res = z.run(SPAWN, address=self.address)
        if res.rc != 0 or "helper pid=" not in res.out:
            tail = (res.err or res.out).strip().splitlines()
            raise RuntimeError(f"spawn_failed:rc={res.rc}:{tail[-1][:120] if tail else '-'}")
        self.pid = int(res.out.split("helper pid=")[1].split()[0])
        self.conn = Client(self.address, family="AF_UNIX", authkey=multiprocessing.current_process().authkey)
        if not self.conn.poll(60.0):
            raise RuntimeError("spawn_failed:no_hello")
        hello = self.conn.recv()
        if not (isinstance(hello, tuple) and hello[:1] == ("hello",)):
            raise RuntimeError("spawn_failed:bad_hello")
        self.helper_cuda_initialized = bool(hello[2])

    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
        except OSError:
            return False
        return not self.conn.closed

    def call(self, msg: tuple, poll_s: float = 1.0):
        """Send one job, block for its reply (the helper is sequential); raises RuntimeError('helper_died:…') when it is gone."""
        seq = msg[1]
        try:
            self.conn.send(msg)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"helper_died:send:{type(e).__name__}") from None
        while True:
            try:
                ready = self.conn.poll(poll_s)
            except (EOFError, OSError) as e:
                raise RuntimeError(f"helper_died:{type(e).__name__}") from None
            if ready:
                try:
                    rep = self.conn.recv()
                except (EOFError, OSError) as e:
                    raise RuntimeError(f"helper_died:{type(e).__name__}") from None
                if rep[1] != seq:
                    continue
                return rep
            if not self.alive():
                raise RuntimeError("helper_died:gone")

    def close(self) -> None:
        for f in (lambda: self.conn.send(None), lambda: self.conn.close(), lambda: shutil.rmtree(self.dir, ignore_errors=True)):
            try:
                f()
            except Exception:  # noqa: BLE001
                pass


# ----------------------------------------------------------------------------------------------------------------- device -> pinned host staging
def _pool_take(shape, dtype):
    import torch
    key = (dtype, tuple(shape))
    with _LOCK:
        free = _POOL.get(key)
        if free:
            return free.pop()
    return torch.empty(tuple(shape), dtype=dtype, pin_memory=True)


def _pool_give(bufs) -> None:
    with _LOCK:
        for b in bufs:
            _POOL.setdefault((b.dtype, tuple(b.shape)), []).append(b)


def _side_stream(device):
    import torch
    s = _STREAMS.get(device)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _STREAMS[device] = s
    return s


def _stage(x, staged: list, streams: dict):
    """Nested dict of tensors -> the same structure with every CUDA tensor copied non_blocking into a pinned host buffer on the side stream of its
    device (the stream waits on the tensor's compute stream first; the source is recorded on the side stream); CPU tensors pass as they are."""
    import torch
    if isinstance(x, torch.Tensor):
        if not x.is_cuda:
            return x.detach()
        dev = x.device
        side = streams.get(dev)
        if side is None:
            side = _side_stream(dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            streams[dev] = side
        h = _pool_take(x.shape, x.dtype)
        with torch.cuda.stream(side):
            h.copy_(x.detach(), non_blocking=True)
            x.record_stream(side)
        staged.append(h)
        return h
    if isinstance(x, dict):
        return {k: _stage(v, staged, streams) for k, v in x.items()}
    return x


def _inline_reason(prediction: dict) -> Optional[str]:
    """A named reason to run the stock call inline for this prediction, else None."""
    import torch
    if prediction.get("exception"):
        return "exception"
    cs = prediction.get("confidence_score")
    if isinstance(cs, torch.Tensor) and cs.numel() > 1:            # diffusion_samples > 1: the ranking argsort must see distinct, non-NaN scores
        ok = bool((~torch.isnan(cs)).all().item()) and int(torch.unique(cs).numel()) == int(cs.numel())
        if not ok:
            return "tied_scores"
    return None


# ----------------------------------------------------------------------------------------------------------------- the displaced method + worker calls
def _write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
    orig = STATE["orig"]
    STATE["items"] += 1
    if STATE["dead"] or not STATE["installed"] or STATE["joined"]:
        why = STATE["dead"] or ("joined" if STATE["joined"] else "not_installed")
        STATE["fallback"] += 1; STATE["fallback_by"][why] = STATE["fallback_by"].get(why, 0) + 1
        return orig(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)
    why = _inline_reason(prediction)
    if why:
        STATE["inline"] += 1; STATE["inline_by"][why] = STATE["inline_by"].get(why, 0) + 1
        return orig(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)
    try:
        submit(self, prediction, batch)
    except Exception as e:  # noqa: BLE001  — staging failed: the stock call still writes this item; the gate refuses by name
        why = f"stage_error:{type(e).__name__}"
        _say(f"staging failed ({type(e).__name__}: {e}); the stock writer runs inline for this item")
        STATE["fallback"] += 1; STATE["fallback_by"][why] = STATE["fallback_by"].get(why, 0) + 1
        return orig(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)
    return None


def submit(writer, prediction: dict, batch: dict) -> dict:
    """Stage the tensors the writer reads and queue the write; returns the job. Called in the predicting thread."""
    import torch
    t0 = time.perf_counter()
    cfg = {"data_dir": os.path.abspath(str(writer.data_dir)), "output_dir": os.path.abspath(str(writer.output_dir)), "output_format": writer.output_format,
           "boltz2": bool(writer.boltz2), "write_embeddings": bool(writer.write_embeddings)}
    keys = [k for k in READ_KEYS if k in prediction] + ([k for k in EMBED_KEYS if k in prediction] if writer.write_embeddings else [])
    sub = {"exception": False, **{k: prediction[k] for k in keys}}
    records = list(batch["record"])
    staged: list = []; streams: dict = {}
    with torch.inference_mode():
        host = _stage(sub, staged, streams)
        events = []
        for dev, side in streams.items():
            ev = torch.cuda.Event()
            ev.record(side)
            torch.cuda.current_stream(dev).wait_event(ev)          # later compute-stream work (the next item, a graph replay over static outputs) is ordered after the copies
            events.append(ev)
    nbytes = sum(int(h.numel()) * int(h.element_size()) for h in staged)
    _SEQ[0] += 1
    name, seed = STATE["current"] if STATE["current"] else (records[0].id if records else "?", None)
    job = {"kind": "write", "seq": _SEQ[0], "name": name, "seed": seed, "names": [r.id for r in records], "cfg": cfg, "host": host,
           "records": records, "events": events, "staged": staged, "t_submit": time.perf_counter(), "stage_bytes": nbytes, "done": False,
           "ok": None, "error": None, "info": None}
    with _COND:
        for n in set(job["names"]) | {name}:
            _PENDING[n] = _PENDING.get(n, 0) + 1
        _INFLIGHT.append(job)
        STATE["submitted"] += 1
        STATE["queued_max"] = max(STATE["queued_max"], len(_INFLIGHT))
    _Q.put(job)
    ms = (time.perf_counter() - t0) * 1000.0
    STATE["stage_ms_max"] = max(STATE["stage_ms_max"], ms); STATE["stage_ms_total"] += ms
    if len(_INFLIGHT) > MAX_PENDING:                              # back-pressure: the predicting thread waits for the oldest write (counted)
        tw = time.perf_counter()
        oldest = _INFLIGHT[0]
        with _COND:
            while not oldest["done"]:
                _COND.wait(0.5)
        STATE["wait_ms_total"] += (time.perf_counter() - tw) * 1000.0
    return job


def begin_item(name: str, seed) -> None:
    """The worker names the (input, seed) unit it is about to predict: the write job carries it (failed-by-name accounting)."""
    STATE["current"] = (str(name), seed)


def finish_item(name: str, seed, pred_dir, dst) -> Optional[List[str]]:
    """The worker's per-item move. Inline (stock statements; the file list returned) unless a background write for `name` is pending —
    then the move is queued behind it and None is returned (the file list arrives with collect())."""
    pred_dir, dst = os.path.abspath(str(pred_dir)), os.path.abspath(str(dst))
    with _COND:
        pending = _PENDING.get(str(name), 0) > 0 and STATE["installed"] and not STATE["joined"]
        if pending:
            _SEQ[0] += 1
            job = {"kind": "move", "seq": _SEQ[0], "name": str(name), "seed": seed, "pred_dir": pred_dir, "dst": dst, "t_submit": time.perf_counter(),
                   "done": False, "ok": None, "error": None, "info": None}
            _PENDING[str(name)] = _PENDING.get(str(name), 0) + 1
            STATE["moves"] += 1
    if pending:
        _Q.put(job)
        return None
    return do_move(pred_dir, dst)["files"]


def wait_item(name: str) -> float:
    """Block until no background job names `name` (the affinity leg reads the structure files). Returns the seconds waited (counted)."""
    t = time.perf_counter()
    with _COND:
        while _PENDING.get(str(name), 0) > 0 and not STATE["dead"]:
            _COND.wait(0.5)
    dt = time.perf_counter() - t
    STATE["wait_ms_total"] += dt * 1000.0
    return dt


def collect() -> List[dict]:
    """Finished background jobs since the last call: [{kind, name, seed, ok, error, files, bytes, write_s, move_s}] (the worker logs them)."""
    with _COND:
        out = list(_DONE); _DONE.clear()
    return out


def failed_count() -> int:
    return int(STATE["failed"])


def _complete(job: dict, ok: bool, info: Optional[dict], error: Optional[str]) -> None:
    job["ok"], job["info"], job["error"] = ok, info, error
    rec = {"kind": job["kind"], "name": job["name"], "seed": job["seed"], "ok": ok, "error": error, "seq": job["seq"]}
    if job["kind"] == "write":
        rec.update(write_s=(info or {}).get("write_s"), files=sorted((info or {}).get("files") or {}), stage_bytes=job.get("stage_bytes"))
        if ok:
            STATE["served"] += 1; STATE["write_s_total"] += float((info or {}).get("write_s") or 0.0)
            STATE["bytes_written"] += int(sum(((info or {}).get("files") or {}).values()))
    else:
        rec.update(files=list((info or {}).get("files") or []), bytes=(info or {}).get("bytes"), move_s=(info or {}).get("move_s"))
        if ok:
            STATE["moves_done"] += 1; STATE["move_s_total"] += float((info or {}).get("move_s") or 0.0)
    if not ok:
        STATE["failed"] += 1; STATE["errors"].append(f"#{job['seq']} {job['kind']} {job['name']} s{job['seed']}: {error}")
        _say(f"FAILED {job['kind']} item={job['name']} seed={job['seed']} reason={error}")
    STATE["per_item"].append({k: rec[k] for k in ("kind", "name", "seed", "ok", "write_s", "move_s", "stage_bytes") if k in rec})
    with _COND:
        job["done"] = True
        names = (set(job.get("names") or []) | {job["name"]}) if job["kind"] == "write" else {job["name"]}
        for n in names:
            if _PENDING.get(n, 0) > 0:
                _PENDING[n] -= 1
        if job in _INFLIGHT:
            _INFLIGHT.remove(job)
        _DONE.append(rec)
        _COND.notify_all()


def _consumer() -> None:
    """The background: per write job wait for its copies (outside the GIL), hand it to the helper (or write here: thread backend); per move job the same."""
    helper = STATE.get("helper")
    while True:
        job = _Q.get()
        if job is None:
            break
        ok, info, error = False, None, None
        try:
            if STATE["dead"]:
                raise RuntimeError(STATE["dead"])
            if job["kind"] == "write":
                tw = time.perf_counter()
                for ev in job["events"]:
                    ev.synchronize()
                STATE["d2h_wait_ms_total"] += (time.perf_counter() - tw) * 1000.0
                if helper is not None:
                    rep = helper.call(("write", job["seq"], job["cfg"], _to_numpy(job["host"]), job["records"]))
                    if rep[0] != "ok":
                        raise RuntimeError(rep[2] + (("\n" + rep[3]) if len(rep) > 3 and rep[3] else ""))
                    info = rep[2]
                else:
                    import torch
                    with torch.inference_mode():
                        info = do_write(job["cfg"], job["host"], job["records"])
            else:
                if helper is not None:
                    rep = helper.call(("move", job["seq"], job["pred_dir"], job["dst"]))
                    if rep[0] != "ok":
                        raise RuntimeError(rep[2])
                    info = rep[2]
                else:
                    info = do_move(job["pred_dir"], job["dst"])
            ok = True
        except Exception as e:  # noqa: BLE001
            error = str(e).splitlines()[0][:400] if str(e) else type(e).__name__
            if str(e).startswith("helper_died") and not STATE["dead"]:
                STATE["dead"] = str(e).split(":")[0] + ":" + str(e).split(":")[1] if ":" in str(e) else str(e)
                _say(f"the writer process is gone ({e}); every later item is written inline by the stock call (fallback_by={STATE['dead']})")
            if not str(e).startswith("helper_died"):
                sys.stderr.write(traceback.format_exc()); sys.stderr.flush()
        finally:
            if job["kind"] == "write":
                _pool_give(job.get("staged") or [])
                job["host"] = None; job["staged"] = None; job["records"] = None
            _complete(job, ok, info, error)


def join(timeout: Optional[float] = None) -> dict:
    """Wait for every pending write / move, stop the consumer and the helper; idempotent. Returns the census. The worker calls it before its log is final."""
    global _THREAD
    if STATE["joined"]:
        return census()
    t = time.perf_counter()
    STATE["joined"] = True
    if _THREAD is not None:
        _Q.put(None)
        _THREAD.join(timeout)
        if _THREAD.is_alive():                                   # a hung background: every job still in flight is a counted loss, never a silent one
            with _COND:
                hung = [j for j in _INFLIGHT if not j["done"]]
            for j in hung:
                _complete(j, False, None, "join_timeout")
        _THREAD = None
    h = STATE.get("helper")
    if h is not None:
        h.close(); STATE["helper"] = None
    STATE["join_s"] = round(time.perf_counter() - t, 3)
    return census()


def census() -> dict:
    st = STATE
    return {"items": st["items"], "submitted": st["submitted"], "served": st["served"], "inline": st["inline"], "inline_by": dict(st["inline_by"]),
            "fallback": st["fallback"], "fallback_by": dict(st["fallback_by"]), "failed": st["failed"], "errors": list(st["errors"]),
            "moves": st["moves"], "moves_done": st["moves_done"], "queued_max": st["queued_max"], "join_s": st["join_s"],
            "bytes_written": st["bytes_written"], "complete": bool(st["joined"] and st["failed"] == 0 and st["served"] == st["submitted"] and st["moves_done"] == st["moves"])}


# ----------------------------------------------------------------------------------------------------------------- attach contract
def dispositions() -> Dict[str, str]:
    d = STATE.get("disposition")
    return {"writer_overlap": d} if d else {}


def apply(spec=None) -> List[str]:
    """Spawn the writer process (or start the thread backend) and displace BoltzWriter.write_on_batch_end. Returns the levers installed."""
    if STATE["installed"]:
        return list(LEVERS)
    if not requested():
        return []
    bad = problems()
    if bad:
        raise RuntimeError("; ".join(bad))
    from .prefetch import worker_pipeline
    pipe = worker_pipeline()
    if pipe != 1:
        STATE["disposition"] = "skipped_by_name:pipeline0"
        _say(f"not installed: the worker runs --pipeline {pipe} (Lightning's own predict loop); the stock writer call stays, by name")
        return []
    import importlib
    W = importlib.import_module(WRITER_MODULE)
    backend = backend_of()
    if backend == "process":
        try:
            h = Helper()
        except Exception as e:  # noqa: BLE001
            STATE["disposition"] = f"skipped_by_name:{str(e).split(':')[0] if str(e).startswith(('no_zygote', 'spawn_failed')) else 'helper_' + type(e).__name__}"
            _say(f"not installed: the writer process could not be spawned through the zygote ({type(e).__name__}: {e}); the stock writer call stays, by name")
            return []
        STATE.update(helper=h, helper_pid=h.pid, forked_before_cuda=(h.zygote_cuda_initialized is False and h.helper_cuda_initialized is False),
                     helper_cuda_initialized=h.helper_cuda_initialized)
    _start(backend, W.BoltzWriter.write_on_batch_end)
    W.BoltzWriter.write_on_batch_end = _write_on_batch_end
    import atexit
    atexit.register(_at_exit)
    _say((f"helper pid={STATE['helper_pid']} spawned through the zygote (forked_before_cuda={STATE['forked_before_cuda']})" if backend == "process" else
          "thread backend (diagnostic word: the consumer thread writes in this process)") + "; BoltzWriter.write_on_batch_end -> pinned D2H staging + background write/move")
    return list(LEVERS)


def _start(backend: str, orig) -> None:
    """Record the displaced function, start the consumer thread, mark installed (apply's tail; a test starts the thread backend over its own writer function here)."""
    global _THREAD
    STATE["backend"] = backend
    STATE["orig"] = orig
    _THREAD = threading.Thread(target=_consumer, name="bz2_writer", daemon=True)
    _THREAD.start()
    STATE["installed"] = True


def line() -> str:
    from opt_core.report import lever_line
    st = STATE
    if not st["installed"]:
        reason = st.get("disposition") or "not_requested"
        return lever_line(TAG, "writer_overlap", "skipped", reason=reason.replace("skipped_by_name:", ""), impl=IMPL, origin="kit", strategy=STRATEGY,
                          execution=st.get("disposition") or "not_requested")
    fields: Dict[str, Any] = dict(backend=st["backend"], items=st["items"], served=st["served"], inline=st["inline"])
    if st["inline_by"]:
        fields["inline_by"] = ",".join(f"{k}:{v}" for k, v in sorted(st["inline_by"].items()))
    fields["fallback"] = st["fallback"]
    if st["fallback_by"]:
        fields["fallback_by"] = ",".join(f"{k}:{v}" for k, v in sorted(st["fallback_by"].items()))
    fields.update(failed=st["failed"], moves=st["moves"], queued_max=st["queued_max"], join_s=st["join_s"], bytes_written=st["bytes_written"],
                  stage_ms_max=round(st["stage_ms_max"], 2), wait_ms_total=round(st["wait_ms_total"], 1), write_s_total=round(st["write_s_total"], 2),
                  helper_pid=st["helper_pid"], helper_lineage="zygote" if st["backend"] == "process" else "none", forked_before_cuda=st["forked_before_cuda"],
                  execution=(f"dead:{st['dead']}" if st["dead"] else f"executed:{st['served']}"))
    return lever_line(TAG, "writer_overlap", "on", impl=IMPL, origin="kit", strategy=STRATEGY, **fields)


def gate() -> Dict[str, Any]:
    """Fail-closed: installed, joined, no failed write / move, no fallback, every submitted write served and every queued move done,
    the writer process forked before CUDA (process backend)."""
    st = STATE
    lineage_ok = (st["forked_before_cuda"] is True) if st["backend"] == "process" else True
    ok = bool(st["installed"] and st["joined"] and st["failed"] == 0 and st["fallback"] == 0 and not st["dead"] and lineage_ok
              and st["served"] == st["submitted"] and st["moves_done"] == st["moves"])
    why = [] if ok else [w for w, c in (("not_installed", not st["installed"]), ("not_joined", not st["joined"]), ("failed_writes", st["failed"] > 0),
                                         ("fallbacks", st["fallback"] > 0), ("dead", bool(st["dead"])), ("forked_after_cuda", not lineage_ok),
                                         ("unserved", st["served"] != st["submitted"] or st["moves_done"] != st["moves"])) if c]
    return {"ok": ok, "why": why, "idle": bool(st["installed"] and st["submitted"] == 0)}


def report() -> Dict[str, Any]:
    st = STATE
    return {"installed": st["installed"], "applied": list(LEVERS) if st["installed"] else [], "disposition": st["disposition"], "switch": os.environ.get(SWITCH),
            "backend": st["backend"], "helper_pid": st["helper_pid"], "forked_before_cuda": st["forked_before_cuda"], "helper_cuda_initialized": st["helper_cuda_initialized"],
            **census(), "stage_ms_max": round(st["stage_ms_max"], 3), "stage_ms_total": round(st["stage_ms_total"], 3), "wait_ms_total": round(st["wait_ms_total"], 1),
            "d2h_wait_ms_total": round(st["d2h_wait_ms_total"], 1), "write_s_total": round(st["write_s_total"], 3), "move_s_total": round(st["move_s_total"], 3),
            "dead": st["dead"], "per_item": list(st["per_item"]), "gate": gate(), "line": line()}


def _at_exit() -> None:
    """Interpreter exit: a worker that did not join (it crashed) still gets its finished predictions' files (drained here, bounded) and ONE line."""
    try:
        if STATE["installed"] and not STATE["joined"]:
            _say("not joined by the worker: draining the pending writes at exit")
            join(timeout=120.0)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[{TAG} writer] drain at exit failed: {type(e).__name__}: {e}\n")
    try:
        sys.stderr.write(line() + "\n"); sys.stderr.flush()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[{TAG} writer] LEVER line not written: {type(e).__name__}: {e}\n")


def reset_for_tests() -> None:
    """Undo apply() in a test process (restore the stock method, stop the background, clear the counters)."""
    global _THREAD
    if STATE.get("orig") is not None:
        try:
            import importlib
            importlib.import_module(WRITER_MODULE).BoltzWriter.write_on_batch_end = STATE["orig"]
        except Exception:  # noqa: BLE001
            pass
    if _THREAD is not None and not STATE["joined"]:
        try:
            join(timeout=30.0)
        except Exception:  # noqa: BLE001
            pass
    _THREAD = None
    for k, v in (("installed", False), ("disposition", None), ("backend", None), ("orig", None), ("helper", None), ("helper_pid", None), ("forked_before_cuda", None),
                 ("helper_cuda_initialized", None), ("items", 0), ("submitted", 0), ("served", 0), ("inline", 0), ("fallback", 0), ("failed", 0), ("moves", 0),
                 ("moves_done", 0), ("queued_max", 0), ("join_s", 0.0), ("bytes_written", 0), ("stage_ms_max", 0.0), ("stage_ms_total", 0.0), ("wait_ms_total", 0.0),
                 ("write_s_total", 0.0), ("move_s_total", 0.0), ("d2h_wait_ms_total", 0.0), ("dead", None), ("current", None), ("joined", False)):
        STATE[k] = v
    for k in ("inline_by", "fallback_by"):
        STATE[k] = {}
    STATE["errors"] = []; STATE["per_item"] = []
    _PENDING.clear(); _INFLIGHT.clear(); _DONE.clear(); _WRITERS.clear()
    while not _Q.empty():
        try:
            _Q.get_nowait()
        except Exception:  # noqa: BLE001
            break
