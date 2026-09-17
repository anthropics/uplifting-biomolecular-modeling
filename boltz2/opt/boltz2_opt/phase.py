"""Per-item PHASE timing line — the same instrument on both arms (the stock CLI process, ``stock_pred``; the kits' worker process,
``worker_launch``), at the same semantic boundaries of ``Boltz2.predict_step``; numerics unchanged (it synchronizes the device and reads a
clock, nothing else).

One line per ``predict_step`` (one record = one item; ``item`` is the record id — the YAML stem, the worker's item name, stock's output
directory name), printed to stdout when the step returns::

    PHASE item=<record id> lm_s=- trunk_s=<f> sampler_s=<f> conf_s=<f|-> total_s=<f> cond_s=<f|-> fwd_s=<f>
    [boltz2-opt] PEAK item=<record id> alloc_gib=<f> reserved_gib=<f>     (the item's torch allocator peak through the core's counters, opt_core.mem.allocator:
                                                                           reset_peak at item start, max_memory_allocated / max_memory_reserved at its end; nothing
                                                                           else on the line; a rank prints `[boltz2-opt] PEAK-NOTE item=<id> rank=<r>` beside it and
                                                                           the parent reduces the ranks: worker.peak_summary)

Boundaries (``torch.cuda.synchronize()`` then ``time.perf_counter()`` at each; measured on the model INSTANCE at call time, so every
class-level replacement the kits install — the CUDA-graph sampler / DiT hoist ``AtomDiffusion.sample``, the memory line's
``DiffusionConditioning.forward`` / ``ConfidenceModule.forward``, the flash patch's or the row-sharded ``Boltz2.forward`` — is timed at the
same boundary as the stock statement it replaces):

- ``trunk_s``  = ``Boltz2.forward`` entry → ``DiffusionConditioning.forward`` entry: input embedder, relative-position encoding, the
  recycling loop (template / MSA module / Pairformer × (recycling_steps + 1)), distogram head — the whole trunk incl. recycles.
- ``cond_s``   = ``DiffusionConditioning.forward`` (once per item; reported on its own, in neither trunk nor sampler).
- ``sampler_s`` = ``AtomDiffusion.sample`` (``self.structure_module.sample``): all sampling steps × all diffusion samples.
- ``conf_s``   = ``ConfidenceModule.forward`` (``self.confidence_module``; ``-`` when the model runs without confidence).
- ``lm_s``     = ``-``: Boltz-2 has no protein-language-model encode.
- ``fwd_s``    = ``Boltz2.forward`` (``trunk + cond + sampler + conf ≤ fwd``; a violation prints ``PHASE-WARN``).
- ``total_s``  = ``Boltz2.predict_step`` entry → return (the worker's ``model_s`` boundary: the cross-check).

Every phase is a separate Python call at this pin (no whole-model CUDA graph: the graph sampler captures one DiT step inside
``AtomDiffusion.sample``), so no field is ``NA``; a boundary that did not run for an item prints ``-``. A synchronize is skipped while a
CUDA stream is capturing. The instrument never raises into the model: a hook that cannot be placed prints ``PHASE-NOTE`` and the step runs
untimed.
"""
from __future__ import annotations

import functools
import sys
import time

MODEL_MODULE = "boltz.model.models.boltz2"          # the trigger: Boltz2 lives here (stock/src/boltz/model/models/boltz2.py:40)
BOUNDARIES = {                                        # qualnames at the pin (stock/src/boltz/…), for the record
    "total": "boltz.model.models.boltz2.Boltz2.predict_step (boltz2.py:1057)",
    "fwd": "boltz.model.models.boltz2.Boltz2.forward (boltz2.py:401)",
    "trunk": "Boltz2.forward entry (boltz2.py:414 input_embedder … :439 recycling loop … :491 distogram) → DiffusionConditioning.forward entry (boltz2.py:516)",
    "cond": "boltz.model.modules.diffusion_conditioning.DiffusionConditioning.forward (diffusion_conditioning.py:83; called boltz2.py:516)",
    "sampler": "boltz.model.modules.diffusionv2.AtomDiffusion.sample (diffusionv2.py:295; called boltz2.py:533)",
    "conf": "boltz.model.modules.confidencev2.ConfidenceModule.forward (confidencev2.py:109; called boltz2.py:587)",
    "lm": "- (Boltz-2 has no protein-LM encode)",
}
_STATE = {"installed_on": None, "orig": None, "n_lines": 0}


def _say(msg: str) -> None:
    print(msg, flush=True)


PEAK_PREFIX = "[boltz2-opt] PEAK"                                      # the per-item allocator peak line, both arms — exactly `[boltz2-opt] PEAK item=<id> alloc_gib=<f> reserved_gib=<f>`
                                                                       # (the log reader's grammar: nothing else on the line); rank / absence notes go on a `[boltz2-opt] PEAK-NOTE` line


def _alloc():
    """The core's allocator counters (opt_core.mem.allocator: reset_peak / counters over torch.cuda; never imports torch itself)."""
    from opt_core.mem import allocator
    return allocator


def _peak_reset() -> None:
    """reset_peak_memory_stats at item start (report-only; numerics unchanged)."""
    try:
        _alloc().reset_peak()
    except Exception:  # noqa: BLE001
        pass


def _peak_line(item: str) -> None:
    """The item's allocator peak since _peak_reset (max_memory_allocated / max_memory_reserved, GiB); without CUDA a PEAK-NOTE names the absence
    (never a zero); on a row-sharded rank (ROWPAIR_RANK) a PEAK-NOTE names the rank (the parent reduces the ranks: worker.peak_summary)."""
    import os
    rk = os.environ.get("ROWPAIR_RANK", "").strip()
    try:
        c = _alloc().counters()
    except Exception as e:  # noqa: BLE001
        c = {"max_allocated_gib": None, "error": f"{type(e).__name__}: {e}"[:120]}
    if c.get("max_allocated_gib") is None:
        _say(f"{PEAK_PREFIX}-NOTE item={item} cuda=absent" + (f" error={c['error']}" if c.get("error") else "") + (f" rank={rk}" if rk else "")); return
    _say(f"{PEAK_PREFIX} item={item} alloc_gib={c['max_allocated_gib']:.2f} reserved_gib={c['max_reserved_gib']:.2f}")
    if rk:
        _say(f"{PEAK_PREFIX}-NOTE item={item} rank={rk}")


def _clock():
    """perf_counter after a device synchronize (skipped without CUDA, before CUDA is initialised, or while a stream is capturing)."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized() and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 — a clock read never raises into the model
        pass
    return time.perf_counter()


def _item_id(batch) -> str:
    try:
        recs = batch["record"]
        return "+".join(str(getattr(r, "id", r)) for r in recs) or "?"
    except Exception:  # noqa: BLE001
        return "?"


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


class _Marks:
    """Start/end clock marks per boundary; durations accumulate over repeated calls, the first start is kept for the trunk boundary."""

    def __init__(self):
        self.first = {}; self.open = {}; self.dur = {}

    def start(self, name, *_a, **_k):
        t = _clock(); self.first.setdefault(name, t); self.open[name] = t

    def end(self, name, *_a, **_k):
        t = _clock(); t0 = self.open.pop(name, None)
        if t0 is not None:
            self.dur[name] = self.dur.get(name, 0.0) + (t - t0)
        self.first.setdefault(name + ":end", t)

    def get(self, name):
        return self.dur.get(name)


def _instrument(model, marks):
    """Place the per-call boundaries on this model instance; returns the undo callables."""
    undo = []
    undo.append(model.register_forward_pre_hook(lambda m, a: marks.start("fwd")).remove)
    undo.append(model.register_forward_hook(lambda m, a, o: marks.end("fwd")).remove)
    dc = getattr(model, "diffusion_conditioning", None)
    if dc is not None and hasattr(dc, "register_forward_pre_hook"):
        undo.append(dc.register_forward_pre_hook(lambda m, a: marks.start("cond")).remove)
        undo.append(dc.register_forward_hook(lambda m, a, o: marks.end("cond")).remove)
    cm = getattr(model, "confidence_module", None)
    if cm is not None and hasattr(cm, "register_forward_pre_hook"):
        undo.append(cm.register_forward_pre_hook(lambda m, a: marks.start("conf")).remove)
        undo.append(cm.register_forward_hook(lambda m, a, o: marks.end("conf")).remove)
    sm = getattr(model, "structure_module", None)
    if sm is not None and callable(getattr(type(sm), "sample", None)):
        inner = sm.__dict__.get("sample")                      # an instance-level sample already in place (the row-sharded line's probe) is timed as is

        def timed_sample(*a, **k):                             # late-bound: whatever AtomDiffusion.sample is at CALL time (graph sampler, hoist, stock) runs inside the marks
            marks.start("sampler")
            try:
                return inner(*a, **k) if inner is not None else type(sm).sample(sm, *a, **k)
            finally:
                marks.end("sampler")

        sm.__dict__["sample"] = timed_sample

        def restore():
            if inner is not None:
                sm.__dict__["sample"] = inner
            else:
                sm.__dict__.pop("sample", None)
        undo.append(restore)
    return undo


def _line(item, marks, total):
    fwd = marks.get("fwd"); cond = marks.get("cond"); samp = marks.get("sampler"); conf = marks.get("conf")
    t_fwd0 = marks.first.get("fwd"); t_edge = marks.first.get("cond", marks.first.get("sampler"))
    if t_fwd0 is not None and t_edge is not None:
        trunk = t_edge - t_fwd0
    elif fwd is not None:                                          # a forward that ran no structure module: the whole forward is trunk
        trunk = fwd
    else:
        trunk = None
    _say(f"PHASE item={item} lm_s=- trunk_s={_fmt(trunk)} sampler_s={_fmt(samp)} conf_s={_fmt(conf)} total_s={_fmt(total)} cond_s={_fmt(cond)} fwd_s={_fmt(fwd)}")
    _STATE["n_lines"] += 1
    parts = sum(x for x in (trunk, cond, samp, conf) if x is not None)
    if fwd is not None and parts > fwd * 1.001 + 1e-3:
        _say(f"PHASE-WARN item={item} phases sum {parts:.3f}s > fwd_s {fwd:.3f}s")
    if fwd is not None and total is not None and fwd > total * 1.001 + 1e-3:
        _say(f"PHASE-WARN item={item} fwd_s {fwd:.3f}s > total_s {total:.3f}s")


def _wrap_predict_step(orig):
    @functools.wraps(orig)
    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        marks = _Marks(); item = _item_id(batch)
        _peak_reset()                                                  # the PEAK line's window is the item
        try:
            undo = _instrument(self, marks)
        except Exception as e:  # noqa: BLE001 — the step runs untimed, never broken
            _say(f"PHASE-NOTE item={item} instrument not placed: {type(e).__name__}: {e}")
            try:
                return orig(self, batch, batch_idx, dataloader_idx)
            finally:
                _peak_line(item)
        t0 = _clock()
        try:
            return orig(self, batch, batch_idx, dataloader_idx)
        finally:
            total = _clock() - t0
            for u in reversed(undo):
                try:
                    u()
                except Exception:  # noqa: BLE001
                    pass
            _line(item, marks, total)
            _peak_line(item)
    predict_step._phase_timing = True
    return predict_step


def install(model_cls=None):
    """Wrap ``Boltz2.predict_step`` (class level, chained over whatever is installed, idempotent) and print the boundaries once.
    ``model_cls`` defaults to ``boltz.model.models.boltz2.Boltz2`` (imported here: call after the stock package may be imported)."""
    if model_cls is None:
        import importlib
        model_cls = importlib.import_module(MODEL_MODULE).Boltz2
    cur = model_cls.predict_step
    if getattr(cur, "_phase_timing", False):
        return False
    _STATE["orig"] = cur; _STATE["installed_on"] = f"{model_cls.__module__}.{model_cls.__qualname__}"
    model_cls.predict_step = _wrap_predict_step(cur)
    _say("PHASE-NOTE boundaries " + "; ".join(f"{k}={v}" for k, v in BOUNDARIES.items()) + " — separable=yes (every phase a separate Python call; sync+perf_counter at each edge; numerics unchanged)")
    return True


def install_after(module) -> None:
    """The after-import hook body (worker_launch): ``module`` is the executed ``boltz.model.models.boltz2``."""
    try:
        install(getattr(module, "Boltz2"))
    except Exception as e:  # noqa: BLE001 — timing never refuses a run
        sys.stderr.write(f"PHASE-NOTE instrument not installed: {type(e).__name__}: {e}\n")


def uninstall(model_cls=None) -> bool:
    if _STATE["orig"] is None:
        return False
    if model_cls is None:
        import importlib
        model_cls = importlib.import_module(MODEL_MODULE).Boltz2
    model_cls.predict_step = _STATE["orig"]; _STATE["orig"] = None
    return True
