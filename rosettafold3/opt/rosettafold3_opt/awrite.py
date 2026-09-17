"""Lever ``awrite``: the asynchronous writer — an item's output files are written by one side process while the fold process
moves on to the next item; the loop never blocks on CIF / JSON serialisation.

What it replaces. After each forward ``RF3InferenceEngine.run`` (``rf3/inference_engines/rf3.py``) writes, in the fold process and
before it touches the next item: ``dump_ranking_scores`` (a 5-row CSV), ``dump_top_ranked_outputs`` (the best sample's CIF +
summary + full confidences JSON) and ``RF3Output.dump`` per sample (CIF + two JSONs each) — 6 CIF serialisations (biotite) and 12 JSON
dumps whose full-PAE matrices go through ``dump_json_compact_arrays`` (``json.dumps(indent=2)`` then a regex collapse loop over a
string of tens of MB): seconds per item, growing with token count, all of it single-threaded Python.

What it does instead. The three writer names of that module are replaced, class- and module-level, by submitters: each call pickles
its arguments ONCE per item (``dump_ranking_scores`` carries the item's ``RF3Output`` list; the top-ranked and per-sample dumps that
follow reference that list by sample index, so the confidences travel once) to the writer side process — forked with the featurizer at
``_construct_pipeline``, before CUDA (``sidecar.py``) — which runs the UPSTREAM functions on the unpickled objects: same functions, same
values → the same bytes, in the same order (one FIFO), merely later. The parent drains the writer's acknowledgements at every submit
and at the end of ``run()`` (``AWRITE flush``: every file is on disk before ``run()`` returns, so the CLI's exit still means "written");
a call the writer failed is re-run in the fold process by the stock statement with the writer's traceback printed (``errors=<n>``,
never a lost file). ``dump_trajectories`` / in-memory results (``out_dir=None``) are untouched stock paths. A finite-value census
rides the hand-off (``finite=<ok|NONFINITE:…>`` per item: coordinates and summary confidences checked with one reduction each, outside
the forward clock; a report word, never a refusal — upstream's own ``assert_no_nans(network_output)`` in ``validation_step`` already
raises on NaN before any writer runs).

Named step-asides: ``n_gpu>1`` (declined by name at activation: LEVER ``state=off reason=conflict:n_gpu``, nothing forked), ``writer_gone`` (the side process died: stock writers from then
on), ``no_cache`` (a dump whose item the writer was not handed: written in-process).
"""
from __future__ import annotations

import sys
import time
import traceback
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import sidecar as _sc

NAME = "awrite"
ENGINE_MODULE = "rf3.inference_engines.rf3"
ENGINE_CLASS = "RF3InferenceEngine"
OUTPUT_CLASS = "RF3Output"
CONSTRUCT = "_construct_pipeline"
WRITERS = ("dump_ranking_scores", "dump_top_ranked_outputs")        # module-level writer functions of ENGINE_MODULE (+ RF3Output.dump)
PREFIX = "[rosettafold3-opt]"
FLUSH_TIMEOUT_S = 3600.0

STATE: Dict[str, Any] = {"armed": False, "installed": False, "on": False, "reason": None, "off_reason": None, "conflict": None, "n_gpu": 1,
                         "writer_pid": None, "forked_before_cuda": None, "nice": _sc.NICE}
CENSUS: Counter = Counter()
ITEMS: List[dict] = []
_ORIG: Dict[str, Any] = {}
_W: Dict[str, Any] = {"car": None, "pending": {}, "item": None, "outs": None, "n_call": 0}      # the parent's writer link, its in-flight calls, the current item's outputs


class AwriteRefused(RuntimeError):
    """The lever cannot be installed (upstream reshaped)."""


def _say(line: str) -> None:
    print(f"{PREFIX} AWRITE {line}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- the writer (child) side
def _serve(rx, tx, originals: dict) -> None:
    """Messages: ``("item", example_id, outputs)`` caches the item's RF3Output list; ``("call", call_id, fn, example_id, args, kwargs)``
    runs ``originals[fn]`` — ``dump_ranking_scores`` / ``dump_top_ranked_outputs`` get the cached list as their first argument, ``dump``
    the cached sample ``args[0]`` (an index) as ``self``; reply ``("done", call_id, fn, seconds)`` or ``("error", call_id, fn, traceback)``."""
    cache: Dict[str, list] = {}

    def handle(msg):
        kind = msg[0]
        if kind == "item":
            cache.clear()
            cache[msg[1]] = msg[2]
            return None
        if kind == "call":
            _, call_id, fn, ex, args, kwargs = msg
            t0 = time.perf_counter()
            try:
                outs = cache.get(ex)
                if outs is None:
                    raise LookupError(f"no cached outputs for item {ex!r}")
                if fn == "dump":
                    originals["dump"](outs[int(args[0])], *args[1:], **kwargs)
                else:
                    originals[fn](outs, *args, **kwargs)
                return ("done", call_id, fn, time.perf_counter() - t0)
            except Exception:                               # noqa: BLE001
                return ("error", call_id, fn, traceback.format_exc())
        return ("error", -1, "?", f"unknown message {kind!r}")

    _sc.serve_loop(rx, tx, handle, threaded=True)          # a reader thread drains the inbox while a write runs: the parent's hand-off never waits on it


# ---------------------------------------------------------------- the fold process side
def finite_word(outputs) -> str:
    """``ok`` when every sample's coordinates and summary numbers are finite; else ``NONFINITE:<what>`` (first offender)."""
    import math
    import numpy as np
    try:
        for o in outputs:
            c = getattr(getattr(o, "atom_array", None), "coord", None)
            if c is not None and not np.isfinite(np.asarray(c, dtype=np.float64)).all():
                return f"NONFINITE:coord:sample{getattr(o, 'sample_idx', '?')}"
            for k, v in (getattr(o, "summary_confidences", None) or {}).items():
                vals = v if isinstance(v, list) else [v]
                flat = []
                for x in vals:
                    flat.extend(x if isinstance(x, list) else [x])
                for x in flat:
                    if isinstance(x, float) and not math.isfinite(x):
                        return f"NONFINITE:{k}:sample{getattr(o, 'sample_idx', '?')}"
        return "ok"
    except Exception as e:                                   # noqa: BLE001
        return f"unchecked:{type(e).__name__}"


def _drain(block: bool = False, until: Optional[int] = None) -> None:
    """Read the writer's acknowledgements: every one waiting (``block=False``), or until call ``until`` is acknowledged."""
    car = _W["car"]
    while _W["pending"] and car is not None:
        if not block and not car.poll():
            return
        try:
            reply, _n = car.recv(timeout=FLUSH_TIMEOUT_S)
        except _sc.SidecarError as e:
            _writer_gone(str(e))
            return
        kind, call_id, fn = reply[0], reply[1], reply[2]
        rec = _W["pending"].pop(call_id, None)
        if rec is None:
            continue
        if kind == "done":
            CENSUS["files_done"] += rec["files"]
            CENSUS["calls_done"] += 1
            rec["itemrec"]["write_s"] = round(rec["itemrec"].get("write_s", 0.0) + float(reply[3]), 3)
            rec["itemrec"]["done"] += 1
            if rec["itemrec"]["done"] == rec["itemrec"]["calls"]:
                rec["itemrec"]["t_ack"] = time.perf_counter()
                if rec["itemrec"].get("closed"):
                    _item_line(rec["itemrec"])
        else:
            CENSUS["errors"] += 1
            _say(f"writer failed {fn} for item={rec['ex']} (the stock statement runs here):\n{reply[3]}")
            _run_here(rec)
        if until is not None and call_id == until:
            return


def _item_line(ir: dict) -> None:
    if ir.get("printed"):
        return
    ir["printed"] = True
    lag = (ir.get("t_ack") or time.perf_counter()) - ir["t0"]
    ir["lag_s"] = round(lag, 3)
    _say(f"done item={ir['item']} calls={ir['calls']} files={ir['files']} write_s={ir.get('write_s', 0.0):.3f} here_s={ir.get('here_s', 0.0):.3f} "
         f"ack_lag_s={lag:.3f} (write_s: the writer's own seconds; ack_lag_s: hand-off -> last acknowledgement read, an upper bound quantised by the parent's drain points)")


def _run_here(rec: dict) -> None:
    """Fallback: the stock statement in the fold process for one call (writer error / writer gone)."""
    fn, args, kwargs, outs = rec["fn"], rec["args"], rec["kwargs"], rec["outs"]
    t0 = time.perf_counter()
    if fn == "dump":
        _ORIG["dump"](outs[int(args[0])], *args[1:], **kwargs)
    else:
        _ORIG[fn](outs, *args, **kwargs)
    CENSUS["files_here"] += rec["files"]
    CENSUS["calls_here"] += 1
    rec["itemrec"]["done"] += 1
    rec["itemrec"]["here_s"] = round(rec["itemrec"].get("here_s", 0.0) + time.perf_counter() - t0, 3)


def _writer_gone(why: str) -> None:
    if STATE["off_reason"] is None:
        STATE["off_reason"] = f"writer_gone:{why}"
        _say(f"writer gone ({why}); pending calls and every later file written in the fold process")
    pend = list(_W["pending"].values())
    _W["pending"].clear()
    for rec in pend:
        _run_here(rec)


_FILES = {"dump_ranking_scores": 1, "dump_top_ranked_outputs": 3, "dump": 3}


def _submit(fn: str, ex: str, outs: list, args: tuple, kwargs: dict) -> None:
    car = _W["car"]
    ir = _W["item"]
    if ir is None or ir["item"] != ex:                        # first writer call of this item: hand the writer the item's outputs once
        if ir is not None:
            ir["closed"] = True
            if ir["done"] == ir["calls"]:
                _item_line(ir)
        ir = {"item": ex, "t0": time.perf_counter(), "calls": 0, "files": 0, "done": 0, "handoff_s": 0.0, "mib": 0.0,
              "finite": finite_word(outs)}
        _W["item"] = ir
        _W["outs"] = outs
        ITEMS.append(ir); del ITEMS[:-64]
        CENSUS["items"] += 1
        if ir["finite"] != "ok":
            CENSUS["nonfinite"] += 1
        t0 = time.perf_counter()
        n = 0
        if car is not None and car.alive() and not STATE["off_reason"]:
            try:
                n = car.send(("item", ex, outs))
            except _sc.SidecarError as e:
                _writer_gone(str(e))
            except Exception as e:                           # noqa: BLE001 — the item's objects do not pickle: the stock writers, here, from now on (named)
                _writer_gone(f"unpicklable:{type(e).__name__}: {e}")
        ir["handoff_s"] += time.perf_counter() - t0
        ir["mib"] += n / 2 ** 20
        _say(f"item={ex} handoff_s={ir['handoff_s']:.3f} pickled_mib={ir['mib']:.1f} finite={ir['finite']} samples={len(outs)}"
             + ("" if n else f" here={(STATE['off_reason'] or 'writer_gone').split(':')[0]}"))
    rec = {"fn": fn, "ex": ex, "args": args, "kwargs": kwargs, "outs": outs, "files": _FILES.get(fn, 1), "itemrec": ir}
    ir["calls"] += 1
    ir["files"] += rec["files"]
    CENSUS["calls"] += 1
    if STATE["off_reason"] or car is None or not car.alive():
        if STATE["off_reason"] is None:
            _writer_gone("not alive")
        CENSUS[f"here:{(STATE['off_reason'] or 'writer_gone').split(':')[0]}"] += 1
        _run_here(rec)
        return
    _W["n_call"] += 1
    cid = _W["n_call"]
    t0 = time.perf_counter()
    try:
        car.send(("call", cid, fn, ex, args, kwargs))
    except _sc.SidecarError as e:
        _writer_gone(str(e))
        _run_here(rec)
        return
    ir["handoff_s"] += time.perf_counter() - t0
    _W["pending"][cid] = rec
    _drain(block=False)


def _shim_ranking(outputs, out_dir, example_id, *args, **kwargs):
    _submit("dump_ranking_scores", str(example_id), outputs, (out_dir, example_id) + tuple(args), kwargs)


def _shim_top(outputs, out_dir, example_id, *args, **kwargs):
    _submit("dump_top_ranked_outputs", str(example_id), outputs, (out_dir, example_id) + tuple(args), kwargs)
    return max(outputs, key=lambda o: o.summary_confidences.get("ranking_score", float("-inf")))   # what upstream returns (run() ignores it)


def _shim_dump(self, *args, **kwargs):
    ir, outs = _W["item"], _W["outs"]
    idx = None
    if ir is not None and outs is not None and ir["item"] == getattr(self, "example_id", None):
        idx = next((i for i, o in enumerate(outs) if o is self), None)
    if idx is None:                                          # a dump whose object the writer was not handed: the stock statement, here, named
        CENSUS["here:no_cache"] += 1
        CENSUS["calls_here"] += 1
        CENSUS["files_here"] += _FILES["dump"]
        return _ORIG["dump"](self, *args, **kwargs)
    _submit("dump", ir["item"], outs, (idx,) + tuple(args), kwargs)


def _wrap_run(orig):
    def run(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        finally:
            flush()
    run.__wrapped__ = orig
    run.__name__ = "run_awrite"
    return run


def flush() -> None:
    """Block until every submitted call is acknowledged (end of ``run()``); prints ``AWRITE flush``."""
    if _W["car"] is None:
        return
    n = len(_W["pending"])
    t0 = time.perf_counter()
    _drain(block=True)
    ir = _W["item"]
    if ir is not None:
        ir["closed"] = True
        _item_line(ir)
    _W["outs"] = None
    dt = time.perf_counter() - t0
    CENSUS["flush_wait_ms"] += int(dt * 1000)
    _say(f"flush pending={n} wait_s={dt:.3f} calls={CENSUS.get('calls', 0)} done={CENSUS.get('calls_done', 0)} "
         f"here={CENSUS.get('calls_here', 0)} errors={CENSUS.get('errors', 0)}")


def _wrap_construct(orig):
    def _construct_pipeline(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            start_writer()
        except Exception as e:                               # noqa: BLE001
            STATE.update({"on": False, "off_reason": f"start_failed:{type(e).__name__}", "reason": f"{type(e).__name__}: {e}"})
            _say(f"writer not started ({type(e).__name__}: {e}); files written in the fold process")
        return out
    _construct_pipeline.__wrapped__ = orig
    _construct_pipeline.__name__ = "_construct_pipeline_awrite"
    return _construct_pipeline


def start_writer() -> Optional[_sc.Sidecar]:
    if _W["car"] is not None:
        return _W["car"]
    if int(STATE.get("n_gpu") or 1) > 1:
        STATE["off_reason"] = "n_gpu>1"
        _say("not forked: n_gpu>1 (the row-sharded line writes as stock)")
        return None
    try:
        car = _sc.Sidecar(NAME, _serve, (dict(_ORIG),), nice=int(STATE.get("nice") or _sc.NICE)).start()
    except Exception as e:                                  # noqa: BLE001 — the platform refuses a side process (fork / pipes): the writers stay in the loop, named
        STATE["off_reason"] = f"start_refused:{type(e).__name__}"
        _say(f"not forked: start_refused ({type(e).__name__}: {e}) — every file written in the loop as stock")
        return None
    _W["car"] = car
    STATE.update({"on": True, "writer_pid": car.pid, "forked_before_cuda": car.forked_before_cuda})
    _say(f"writer pid={car.pid} forked_before_cuda={car.forked_before_cuda} nice={car.nice}")
    return car


def enable(engine_module=None) -> dict:
    """Install on ``rf3.inference_engines.rf3``: the two module-level writers, ``RF3Output.dump``, ``RF3InferenceEngine.run`` (flush at
    its end) and ``_construct_pipeline`` (the fork point). Idempotent; a reshaped upstream is :class:`AwriteRefused`."""
    m = engine_module if engine_module is not None else __import__(ENGINE_MODULE, fromlist=["_"])
    cls = getattr(m, ENGINE_CLASS, None)
    ocls = getattr(m, OUTPUT_CLASS, None)
    missing = [w for w in WRITERS if not callable(getattr(m, w, None))]
    missing += [w for w, ok in ((f"{OUTPUT_CLASS}.dump", hasattr(ocls, "dump")), (f"{ENGINE_CLASS}.run", hasattr(cls, "run")),
                                (f"{ENGINE_CLASS}.{CONSTRUCT}", hasattr(cls, CONSTRUCT))) if not ok]
    if missing:
        STATE.update({"installed": False, "on": False, "reason": "upstream reshaped: " + ",".join(missing)})
        raise AwriteRefused(STATE["reason"])
    if not getattr(m.dump_ranking_scores, "_rosettafold3_opt_awrite", False):
        _ORIG["dump_ranking_scores"] = m.dump_ranking_scores
        _ORIG["dump_top_ranked_outputs"] = m.dump_top_ranked_outputs
        _ORIG["dump"] = ocls.dump
        for name, fn in (("dump_ranking_scores", _shim_ranking), ("dump_top_ranked_outputs", _shim_top)):
            fn._rosettafold3_opt_awrite = True
            fn.__wrapped__ = _ORIG[name]
            setattr(m, name, fn)
        _shim_dump.__wrapped__ = _ORIG["dump"]
        ocls.dump = _shim_dump
        _ORIG["run"] = cls.run
        cls.run = _wrap_run(cls.run)
        cur = getattr(cls, CONSTRUCT)
        _ORIG[CONSTRUCT] = cur
        setattr(cls, CONSTRUCT, _wrap_construct(cur))
    STATE.update({"installed": True, "reason": None})
    return describe()


def disable(engine_module=None) -> None:
    m = engine_module if engine_module is not None else sys.modules.get(ENGINE_MODULE)
    if m is None or "dump" not in _ORIG:
        return
    cls, ocls = getattr(m, ENGINE_CLASS), getattr(m, OUTPUT_CLASS)
    m.dump_ranking_scores = _ORIG.pop("dump_ranking_scores")
    m.dump_top_ranked_outputs = _ORIG.pop("dump_top_ranked_outputs")
    ocls.dump = _ORIG.pop("dump")
    cls.run = _ORIG.pop("run")
    setattr(cls, CONSTRUCT, _ORIG.pop(CONSTRUCT))
    if _W["car"] is not None:
        _W["car"].stop()
        _W["car"] = None
    STATE.update({"installed": False, "on": False})


CONFLICT_N_GPU = ("the row-sharded line (n_gpu > 1: rowpair) runs the item's writers in every rank process as stock; the lever is a single-GPU "
                  "lever and steps aside by name pending a measured n_gpu=2 run (composition #6)")


def decline(kind: str, why: str) -> dict:
    """Off BY NAME for this process (``conflict:<kind>`` on the LEVER line; the tally block carries the sentence): nothing is forked or installed."""
    STATE.update(on=False, installed=False, conflict=kind, reason=why, off_reason=kind)
    return describe()


def arm(rep: dict, install_watch: Callable) -> dict:
    """Activation-time hook: when the row names ``awrite``, install on ``rf3.inference_engines.rf3`` as soon as it executes."""
    if NAME not in (rep.get("levers") or []) or STATE.get("armed"):
        return rep
    STATE["armed"] = True
    STATE["n_gpu"] = int(rep.get("n_gpu") or 1)
    if STATE["n_gpu"] > 1:                                     # the row-sharded line: off by name in every rank, like confhoist / confln (rowpair owns the item loop there)
        rep[NAME] = decline("n_gpu", CONFLICT_N_GPU)
        return rep

    def on_engine(module):
        try:
            rep[NAME] = enable(module)
        except AwriteRefused as e:
            rep[NAME] = {"on": False, "installed": False, "reason": str(e), "census": None}
            return
        rep["levers_applied"] = list(dict.fromkeys(list(rep.get("levers_applied") or []) + [NAME]))

    if ENGINE_MODULE in sys.modules:
        on_engine(sys.modules[ENGINE_MODULE])
    else:
        install_watch(ENGINE_MODULE, on_engine, rep)
    return rep


def census() -> dict:
    c = dict(CENSUS)
    here = {k[len("here:"):]: v for k, v in c.items() if k.startswith("here:")}
    return {"items": c.get("items", 0), "calls": c.get("calls", 0), "calls_done": c.get("calls_done", 0), "calls_here": c.get("calls_here", 0),
            "files_done": c.get("files_done", 0), "files_here": c.get("files_here", 0), "errors": c.get("errors", 0),
            "nonfinite": c.get("nonfinite", 0), "here": here, "flush_wait_ms": c.get("flush_wait_ms", 0), "by_key": c}


def describe() -> dict:
    return {"on": bool(STATE["on"]), "installed": bool(STATE["installed"]), "reason": STATE["reason"], "off_reason": STATE["off_reason"], "conflict": STATE["conflict"],
            "writer_pid": STATE["writer_pid"], "forked_before_cuda": STATE["forked_before_cuda"], "nice": STATE["nice"],
            "rule": "an item's writers run in one pre-CUDA-forked side process on the item's own objects; flushed before run() returns; stock statement here by name otherwise",
            "census": census(), "items": [{k: v for k, v in ir.items() if k not in ("printed", "t0", "t_ack", "closed")} for ir in ITEMS]}


def lever_tokens(desc: Optional[dict] = None) -> List[Tuple[str, Any]]:
    """(key, value) tokens for the LEVER line: files=<written by the writer> here=<written in-process> errors nonfinite flush_wait_ms."""
    d = desc or describe()
    c = d.get("census") or census()
    out: List[Tuple[str, Any]] = [("files", c["files_done"]), ("here", c["files_here"]), ("errors", c["errors"]), ("nonfinite", c["nonfinite"]),
                                  ("flush_wait_ms", c["flush_wait_ms"])]
    if c["here"]:
        out.append(("here_by", "|".join(f"{k}:{v}" for k, v in sorted(c["here"].items()))))
    return out
