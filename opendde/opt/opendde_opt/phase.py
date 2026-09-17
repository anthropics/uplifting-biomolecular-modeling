"""Per-item phase timing — one line per prediction on every ``pred`` route (stock ``off`` and the kit modes, rank processes included):

    PHASE item=<sample_name> lm_s=- trunk_s=<f> sampler_s=<f> conf_s=<f> total_s=<f>

Timing only: no argument, return value or tensor is touched. The boundaries are upstream's own stage callables on ``opendde.model.opendde.OpenDDE``
(``PHASES``): trunk = ``get_pairformer_output`` (input embedder, MSA module, pairformer x N_cycle recycles) + ``expand_to_structural_tokens`` (the
structural-token refiner that finalises s/z; absent from the sum when the model does not call it); sampler = ``prepare_diffusion_cache_for_sampling``
(the per-item diffusion conditioning) + ``sample_diffusion`` (all N_sample x N_step denoiser calls); conf = ``run_confidence_head``; lm = ``-``
(OpenDDE has no protein language model). total = ``runner.inference.InferenceRunner.predict`` (feature transfer + the model forward over every model
seed) — the runner's own ``Model forward time`` line for the same item additionally holds batch preparation and result dumping, so total_s <= it.
Each phase is bracketed by ``torch.cuda.synchronize()`` and ``time.perf_counter()``.

The kit modes are timed at the same class attributes: where a kit lever rebinds one (pair_offload: ``get_pairformer_output`` /
``run_confidence_head``; the row-sharded pair track: ``get_pairformer_output``), the wrap times the lever's replacement at the same boundary — the attributes are (re)wrapped at every ``predict`` call, after
every lever has installed, and a wrap never hides the callable it times (``functools.wraps`` carries its attributes). A phase attribute missing from the
class prints ``NA`` and one ``PHASE-NOTE`` line. Nothing of upstream is imported here: ``install()`` arms a meta-path entry that wraps
``InferenceRunner.predict`` once upstream itself imports ``runner.inference``; this module imports the standard library only.
"""
from __future__ import annotations

import functools
import os
import re
import sys
import time

RUNNER_MODULE = "runner.inference"
RUNNER_ATTR = ("InferenceRunner", "predict")
MODEL_MODULE = "opendde.model.opendde"
PHASES = {                                                              # field -> the OpenDDE methods whose outermost calls accrue to it
    "trunk": ("get_pairformer_output", "expand_to_structural_tokens"),
    "sampler": ("prepare_diffusion_cache_for_sampling", "sample_diffusion"),
    "conf": ("run_confidence_head",),
}
REQUIRED = {"trunk": "get_pairformer_output", "sampler": "sample_diffusion", "conf": "run_confidence_head"}   # the method a field cannot be reported without
FIELDS = ("lm", "trunk", "sampler", "conf")
MARK = "_opendde_opt_phase"

CURRENT = {"item": None}                                                    # the item `InferenceRunner.predict` is serving now (read by per-item census lines of house levers); None between items
_ACC = {k: 0.0 for k in PHASES}
_DEPTH = {k: 0 for k in PHASES}
_NA = set()                                                             # fields whose required method is absent on the model class (upstream changed shape)
_NOTED = set()
_STATE = {"installed": False, "predict_wrapped": False}
_SUBSCRIBERS: list = []                                                # callables handed the imported `runner.inference` module (lncensus: the resolved-kernel read);
                                                                        # this module's ONE meta-path hook serves them with the timing wrap


def _sync():
    torch = sys.modules.get("torch")
    if torch is None:
        return
    cuda = torch.cuda
    if cuda.is_available() and cuda.is_initialized():
        cuda.synchronize()


def _emit(line: str):
    if os.environ.get("RANK", "0") in ("0", ""):                       # one line per item: rank 0 speaks for a multi-GPU launch
        print(line, flush=True)


# ---- the per-item allocator PEAK line (every route, every rank; the driver reader's fixed grammar — NO other token on the line):
#      `[opendde-opt] PEAK item=<id> alloc_gib=<max_memory_allocated/2**30:.2f> reserved_gib=<max_memory_reserved/2**30:.2f>`
PEAK_PROCESS = {"alloc_gib": 0.0, "reserved_gib": 0.0}                 # the process high-water folded across the per-item resets (report.cuda_peak reads it)
PEAK_RE = re.compile(r"PEAK item=(\S+) alloc_gib=([0-9.]+) reserved_gib=([0-9.]+)$", re.M)


def _peak_start() -> bool:
    """Item start: fold the allocator's maxima so far into PEAK_PROCESS, then reset them (the core's allocator helpers). False without CUDA."""
    from opt_core.mem import allocator as _al
    c = _al.counters()
    if c.get("max_allocated_gib") is not None:
        PEAK_PROCESS["alloc_gib"] = max(PEAK_PROCESS["alloc_gib"], c["max_allocated_gib"]); PEAK_PROCESS["reserved_gib"] = max(PEAK_PROCESS["reserved_gib"], c["max_reserved_gib"])
    return _al.reset_peak()


def peak_line(item: str) -> str | None:
    """The item's PEAK line from the allocator's maxima since _peak_start (None without CUDA). A rank prints the same line (the reader takes the max)."""
    from opt_core.mem import allocator as _al
    from .report import PREFIX
    c = _al.counters()
    if c.get("max_allocated_gib") is None:
        return None
    return f"{PREFIX} PEAK item={item} alloc_gib={c['max_allocated_gib']:.2f} reserved_gib={c['max_reserved_gib']:.2f}"


def peak_summary(transcripts: dict) -> list:
    """The multi-GPU parent's reduce of the ranks' PEAK lines, `transcripts` = {rank (int) | None (a merged stream): text}: one
    `PEAK-RANKS item=<id> lines=<n> rank_max_alloc_gib=… rank_max_reserved_gib=… rank0_alloc_gib=… rank0_reserved_gib=…` line per item
    (rank 0's values `-` when only a merged stream is known)."""
    from .report import PREFIX
    items = {}
    for rank, text in transcripts.items():
        for m in PEAK_RE.finditer(text or ""):
            items.setdefault(m.group(1), []).append((rank, float(m.group(2)), float(m.group(3))))
    out = []
    for item, rows in items.items():
        r0 = [x for x in rows if x[0] == 0]
        out.append(f"{PREFIX} PEAK-RANKS item={item} lines={len(rows)} rank_max_alloc_gib={max(x[1] for x in rows):.2f} rank_max_reserved_gib={max(x[2] for x in rows):.2f} "
                   f"rank0_alloc_gib={(f'{r0[0][1]:.2f}' if r0 else '-')} rank0_reserved_gib={(f'{r0[0][2]:.2f}' if r0 else '-')}")
    return out


def _timed(fn, phase: str):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        outer = _DEPTH[phase] == 0
        if outer:
            _sync()
            t0 = time.perf_counter()
        _DEPTH[phase] += 1
        try:
            return fn(*args, **kwargs)
        finally:
            _DEPTH[phase] -= 1
            if outer:
                _sync()
                _ACC[phase] += time.perf_counter() - t0
    setattr(wrapper, MARK, phase)
    return wrapper


def wrap_model_class(cls) -> dict:
    """(Re)wrap the phase methods on ``cls`` that are not wrapped yet (a lever may have rebound one since the last call). Returns
    {method: 'wrapped'|'kept'|'absent'}."""
    out = {}
    for phase, names in PHASES.items():
        for name in names:
            cur = cls.__dict__.get(name) if name in cls.__dict__ else getattr(cls, name, None)
            if isinstance(cur, (staticmethod, classmethod)):                # never re-shaped into a plain function: left as bound, reported NA
                cur = None
            if cur is None:
                out[name] = "absent"
                if REQUIRED.get(phase) == name:
                    _NA.add(phase)
                    if phase not in _NOTED:
                        _NOTED.add(phase)
                        _emit(f"PHASE-NOTE {phase} not separable: {cls.__module__}.{cls.__qualname__}.{name} not found on the pinned upstream")
                continue
            if getattr(cur, MARK, None) == phase:
                out[name] = "kept"
                continue
            setattr(cls, name, _timed(cur, phase))
            out[name] = "wrapped"
    return out


def _fmt(v) -> str:
    return v if isinstance(v, str) else f"{v:.3f}"


def phase_line(item: str, total_s: float, acc: dict | None = None) -> str:
    acc = _ACC if acc is None else acc
    vals = {"lm": "-"}
    for k in PHASES:
        vals[k] = "NA" if k in _NA else acc[k]
    return "PHASE item={} {} total_s={}".format(item, " ".join(f"{k}_s={_fmt(vals[k])}" for k in FIELDS), _fmt(total_s))


def _wrap_predict(predict):
    @functools.wraps(predict)
    def wrapper(self, data, *args, **kwargs):
        model = getattr(self, "model", None)
        if model is not None:
            wrap_model_class(type(model))
        for k in _ACC:
            _ACC[k] = 0.0
            _DEPTH[k] = 0
        item = "unknown"
        try:
            item = str(data.get("sample_name", "unknown"))
        except Exception:  # noqa: BLE001
            pass
        CURRENT["item"] = item
        _sync()
        have_peak = _peak_start()                                        # the allocator's maxima reset at item START (the PEAK line below reads this item's own peak)
        t0 = time.perf_counter()
        ok = False
        try:
            out = predict(self, data, *args, **kwargs)
            ok = True
            return out
        finally:
            CURRENT["item"] = None
            _sync()
            total = time.perf_counter() - t0
            if ok:
                _emit(phase_line(item, total))
                pk = peak_line(item) if have_peak else None
                if pk:
                    print(pk, flush=True)                                # every rank prints its own PEAK line; the multi-GPU parent reduces them (peak_summary)
    setattr(wrapper, MARK, "total")
    return wrapper


def _serve_subscribers(module) -> None:
    for cb in list(_SUBSCRIBERS):
        try:
            cb(module)
        except Exception as e:  # noqa: BLE001 — a subscriber's own failure is named, never fatal to the import it rides
            _emit(f"PHASE-NOTE runner-module subscriber {getattr(cb, '__module__', '?')}.{getattr(cb, '__name__', '?')} failed: {type(e).__name__}: {e}")


def on_runner_module(callback) -> str:
    """Hand ``callback`` the ``runner.inference`` module: now when it is already imported, else at its import through this module's one hook
    (armed here if it is not yet). Idempotent per callback; returns 'called' | 'armed'."""
    if callback not in _SUBSCRIBERS:
        _SUBSCRIBERS.append(callback)
    m = sys.modules.get(RUNNER_MODULE)
    if m is not None:
        callback(m)
        return "called"
    install()
    return "armed"


def wrap_runner_module(module) -> bool:
    """Wrap ``InferenceRunner.predict`` on an imported ``runner.inference`` (idempotent) and serve the subscribers. False when the class or
    method is absent."""
    _serve_subscribers(module)
    cls = getattr(module, RUNNER_ATTR[0], None)
    fn = getattr(cls, RUNNER_ATTR[1], None) if cls is not None else None
    if fn is None:
        _emit(f"PHASE-NOTE total not separable: {RUNNER_MODULE}.{'.'.join(RUNNER_ATTR)} not found on the pinned upstream; no PHASE lines")
        return False
    if getattr(fn, MARK, None) != "total":
        setattr(cls, RUNNER_ATTR[1], _wrap_predict(fn))
    _STATE["predict_wrapped"] = True
    return True


class _Loader:
    """Runs the module body with upstream's own loader, then wraps."""

    def __init__(self, loader, hook):
        self._loader, self._hook = loader, hook

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._hook.remove()
        wrap_runner_module(sys.modules.get(module.__name__, module))


class _Hook:
    """Meta-path entry: serves nothing itself — asks the finders behind it (a kit lever's own import hook on ``runner.inference`` included,
    so that hook still serves the import and fires its patches) for the spec and wraps that spec's loader in :class:`_Loader`. Re-entrant
    calls (another hook resolving the real spec through the meta path) get None, so the module body runs exactly once."""

    armed = True

    def __init__(self):
        self._resolving = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != RUNNER_MODULE or self._resolving:
            return None
        self._resolving = True
        try:
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                try:
                    spec = find(fullname, path, target)
                except Exception:  # noqa: BLE001
                    spec = None
                if spec is not None and spec.loader is not None:
                    spec.loader = _Loader(spec.loader, self)
                    return spec
            return None
        finally:
            self._resolving = False

    def remove(self):
        self.armed = False
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


def install() -> str:
    """Arm the timing: wrap now when ``runner.inference`` is already imported, else at its import. Idempotent; returns 'wrapped'|'armed'."""
    if _STATE["installed"]:
        return "wrapped" if _STATE["predict_wrapped"] else "armed"
    _STATE["installed"] = True
    m = sys.modules.get(RUNNER_MODULE)
    if m is not None:
        wrap_runner_module(m)
        return "wrapped"
    sys.meta_path.insert(0, _Hook())
    return "armed"
