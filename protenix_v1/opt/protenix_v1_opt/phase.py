"""The per-item PHASE line: the coarse stage split of one `InferenceRunner.predict` item, timed on every `pred` route (the stock route's
child process and each activated mode) at the same three method boundaries of `protenix.model.protenix.Protenix`:

  trunk   = `get_pairformer_output` (protenix/model/protenix.py:170 — input embedder, MSA module, template embedder and every pairformer
            recycle of the item); on a kit arm the callable the mode installed under that name (big's `recycle_carry` transcription,
            tp's row-sharded trunk) — the same boundary
  sampler = `sample_diffusion` (protenix.py:306 — all N_step steps x all N_sample samples of the item)
  conf    = `run_confidence_head` (protenix.py:338)
  lm      = `-`: the engine has no protein-LM encode in the forward (the optional ESM variant reads precomputed embeddings as input features)

  `<prefix> PHASE item=<sample_name> lm_s=- trunk_s=<f> sampler_s=<f> conf_s=<f> total_s=<f>`

total_s = the whole stock `predict` body (runner/inference.py:204), `torch.cuda.synchronize()` before and after every boundary +
`time.perf_counter()`; trunk_s + sampler_s + conf_s <= total_s (the rest is the distogram head, `prepare_cache`, the summary confidences
and host work). The stock CLI's own `Model forward time` (runner/inference.py:492-494) is this total plus the CIF/confidence dump.
Timing only — no computed value changes: `install` wraps `InferenceRunner.predict` once (class level); the three method wrappers go on the
class at the start of each item, over whatever the mode put there at runner build, and come off at its end, so every patch and every
object check the kit makes outside an item is untouched. A boundary with no call in an item prints `NA` and one
`PHASE-NOTE <phase> no call at <qualname> in item=<name>` line; a boundary entered while a CUDA stream is capturing is not synchronised
(`PHASE-NOTE <phase> not separable: entered under stream capture`). After the PHASE line, the item's allocator peaks in the shared
grammar: `<prefix> PEAK item=<sample_name> alloc_gib=<f> reserved_gib=<f>` (torch.cuda max_memory_allocated / max_memory_reserved, GiB = 2^30,
device 0) over the item's own window — the peak counters are reset at item entry, and the process-wide peak the reset would lose is carried
here (`process_peak()`: the max over every item so far and the counters since, the number stack.gpu_peak and the rank line read); no CUDA, no
PEAK line. The reset is of torch's device-0 peak counters themselves: a reader outside this module that reads or resets them directly (the
shared core's opt_core.mem allocator / peak / sample_loop / rowpair-census words) sees "since this item began" in a multi-item process, and a
core reset inside an item makes that item's PEAK line read from the reset on. Imports nothing of the kit (the stock route's child loads it after its proof, beside det.py).
"""
from __future__ import annotations

import functools
import re
import sys
import time
from typing import Callable, Dict, Optional

RUNNER_MODULE, RUNNER_CLASS = "runner.inference", "InferenceRunner"
MODEL_MODULE, MODEL_CLASS = "protenix.model.protenix", "Protenix"
PHASES = (("trunk", "get_pairformer_output"), ("sampler", "sample_diffusion"), ("conf", "run_confidence_head"))
LM = "-"                                                   # no protein-LM encode phase in this engine's forward
_MISSING = object()
_STATE: Dict[str, object] = {"installed": None, "acc": {}, "notes": [], "carried": {"alloc": 0, "reserved": 0}}
GIB = float(2 ** 30)


def _cuda():
    """torch.cuda when torch is imported in this process and a device is available, else None (no import of torch from here)."""
    torch = sys.modules.get("torch")
    try:
        return torch.cuda if torch is not None and torch.cuda.is_available() else None
    except Exception:
        return None


def _peak_reset() -> None:
    """Item entry: fold the counters so far into the carried process peak, then open the item's own window (reset_peak_memory_stats)."""
    cuda = _cuda()
    if cuda is None:
        return
    try:
        c = _STATE["carried"]
        c["alloc"] = max(int(c["alloc"]), int(cuda.max_memory_allocated(0))); c["reserved"] = max(int(c["reserved"]), int(cuda.max_memory_reserved(0)))
        cuda.reset_peak_memory_stats(0)
    except Exception:
        pass


def process_peak() -> Dict[str, int]:
    """{alloc, reserved} bytes: the process-wide allocator peak (the carried max over the items' windows and the counters since the last reset);
    {} without CUDA. The per-item reset makes torch's own max_* an item number — readers of the process number use this."""
    cuda = _cuda()
    if cuda is None:
        return {}
    try:
        c = _STATE["carried"]
        return {"alloc": max(int(c["alloc"]), int(cuda.max_memory_allocated(0))), "reserved": max(int(c["reserved"]), int(cuda.max_memory_reserved(0)))}
    except Exception:
        return {}


def peak_line(prefix: str, item: str) -> Optional[str]:
    """`<prefix> PEAK item=<item> alloc_gib=<f> reserved_gib=<f>` for the item's window; None without CUDA."""
    cuda = _cuda()
    if cuda is None:
        return None
    try:
        return f"{prefix} PEAK item={item} alloc_gib={cuda.max_memory_allocated(0) / GIB:.2f} reserved_gib={cuda.max_memory_reserved(0) / GIB:.2f}"
    except Exception:
        return None


def qualnames() -> Dict[str, str]:
    """{phase: dotted name of the timed callable} (+ total = the runner's predict)."""
    q = {k: f"{MODEL_MODULE}.{MODEL_CLASS}.{m}" for k, m in PHASES}
    q["total"] = f"{RUNNER_MODULE}.{RUNNER_CLASS}.predict"
    q["lm"] = LM
    return q


def _sync() -> bool:
    """Synchronise the device; False when a stream capture is in progress (a synchronize there is illegal) or torch has no CUDA."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    torch.cuda.synchronize()
    return True


def _timed(fn: Callable, key: str) -> Callable:
    @functools.wraps(fn)
    def f(*a, **kw):
        ok = _sync()
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            ok = _sync() and ok
            if ok:
                _STATE["acc"][key] = _STATE["acc"].get(key, 0.0) + (time.perf_counter() - t)
            else:
                _STATE["notes"].append(f"PHASE-NOTE {key} not separable: entered under stream capture")
    f.__phase__ = key
    return f


def _wrap_methods():
    """Put the three timers on the model class over its current attributes; -> the restore list (empty when the model module is not loaded)."""
    mod = sys.modules.get(MODEL_MODULE)
    cls = getattr(mod, MODEL_CLASS, None) if mod is not None else None
    if cls is None:
        _STATE["notes"].append(f"PHASE-NOTE trunk,sampler,conf no model class {MODEL_MODULE}.{MODEL_CLASS} loaded in this process")
        return []
    restore = []
    for key, name in PHASES:
        own = vars(cls).get(name, _MISSING)
        cur = getattr(cls, name, None)
        if cur is None or getattr(cur, "__phase__", None) == key:
            continue
        w = _timed(cur, key)
        setattr(cls, name, w)
        restore.append((cls, name, own, w))
    return restore


def _unwrap(restore) -> None:
    for cls, name, own, w in reversed(restore):
        if vars(cls).get(name) is not w:                   # something re-patched the name inside the item: leave its object in place
            continue
        if own is _MISSING:
            delattr(cls, name)
        else:
            setattr(cls, name, own)


def item_key(data) -> str:
    name = data.get("sample_name") if isinstance(data, dict) else None
    return re.sub(r"\s+", "_", str(name if name is not None else "?"))


def line(prefix: str, item: str, total_s: float, acc: Optional[Dict[str, float]] = None, lm: str = LM) -> str:
    acc = _STATE["acc"] if acc is None else acc

    def f(k):
        return f"{acc[k]:.3f}" if k in acc else "NA"
    return f"{prefix} PHASE item={item} lm_s={lm} trunk_s={f('trunk')} sampler_s={f('sampler')} conf_s={f('conf')} total_s={total_s:.3f}"


def notes(prefix: str, item: str, acc: Optional[Dict[str, float]] = None) -> list:
    acc = _STATE["acc"] if acc is None else acc
    q = qualnames()
    out = [f"{prefix} {n}" for n in dict.fromkeys(_STATE["notes"])]
    out += [f"{prefix} PHASE-NOTE {k} no call at {q[k]} in item={item}" for k, _ in PHASES if k not in acc and not any(k in n for n in _STATE["notes"])]
    return out


def _stderr(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


def install(prefix: str, log: Optional[Callable[[str], object]] = None) -> Dict[str, str]:
    """Wrap `runner.inference.InferenceRunner.predict` (class level, once per process) so every item that returns prints its PHASE line
    (+ PHASE-NOTE lines) through `log` (default: stderr). -> qualnames()."""
    if _STATE["installed"]:
        return dict(_STATE["installed"])
    import importlib
    rcls = getattr(importlib.import_module(RUNNER_MODULE), RUNNER_CLASS)
    orig = rcls.predict
    emit = log or _stderr

    @functools.wraps(orig)
    def predict(self, data, *a, **kw):
        item = item_key(data)
        _STATE["acc"] = {}
        _STATE["notes"] = []
        restore = _wrap_methods()
        _sync()
        _peak_reset()                                       # the item's own allocator-peak window (the process peak is carried: process_peak)
        t = time.perf_counter()
        done = False
        try:
            out = orig(self, data, *a, **kw)
            done = True
            return out
        finally:
            _sync()
            total = time.perf_counter() - t
            _unwrap(restore)
            if done:
                emit(line(prefix, item, total))
                for n in notes(prefix, item):
                    emit(n)
                pk = peak_line(prefix, item)
                if pk:
                    emit(pk)

    predict.__phase__ = "total"
    rcls.predict = predict
    _STATE["installed"] = qualnames()
    return dict(_STATE["installed"])
