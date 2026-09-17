"""Per-fold phase timing — the ``PHASE`` line every timed pass prints beside its ``fold <id> s<seed>`` line, on the stock route and
under every kit mode alike (stock_fold.fold_items, the one prediction loop of every route, installs and reads it):

    PHASE item=<id> seed=<s> lm_s=<f|-|NA> trunk_s=<f|-|NA> sampler_s=<f|-|NA> conf_s=<f|-|NA> total_s=<f>

The four phases are timed at upstream's own callables on the loaded ``ESMFold2Model`` instance — whatever each name resolves to when
the prediction loop starts: upstream's method on the stock route, the kit's replacement of it (the same boundary) under a mode that
rebinds it. One ``PHASE-BOUNDARY`` line per pass names the callable actually wrapped for each phase.

    lm       model._compute_lm_hidden_states      the ESM-C protein-LM encode (a kit ESM-C cache hit is a ~0 s call of it)
    trunk    model._run_one_loop                  the Parcae recycle loop: every iteration of lm_encoder / msa_encoder / folding_trunk
    sampler  model.structure_head.sample          the whole diffusion sampling of the fold: all steps x all samples
    conf     model.confidence_head.forward        the confidence heads (their own pair trunk + the pLDDT / PAE / PDE heads)

``total_s`` is the fold window's own wall (stock_fold.ITEM_WALL), passed in by the loop for cross-check: the phases never sum above it —
input featurisation, the input embedder, parcae readout / coda, the distogram head and the host copies are in the total only. Each
boundary is ``torch.cuda.synchronize()`` + ``time.perf_counter()`` on entry and on exit (perf_counter alone without CUDA); nested calls
of one phase count once (the outermost). ``-`` = the model has no such callable; ``NA`` = the phase was not observable in this window
(its callable was rebound past the timer, never called, or a boundary fell inside a CUDA-graph capture where no synchronize is allowed)
— a ``PHASE-NOTE`` line names the reason; a split is never estimated. Timing only: no argument, result or setting is touched.
"""
import functools
import time
from typing import Callable, Dict, List, Optional, Tuple

PHASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (            # phase -> attribute path on the model (the last element is rebound)
    ("lm", ("_compute_lm_hidden_states",)),
    ("trunk", ("_run_one_loop",)),
    ("sampler", ("structure_head", "sample")),
    ("conf", ("confidence_head", "forward")),
)
FIELDS = {"lm": "lm_s", "trunk": "trunk_s", "sampler": "sampler_s", "conf": "conf_s"}
MARK = "_ef2_phase"                                               # the attribute a timing wrapper carries (its phase name)

_STATE: Dict[str, object] = {"model": None}


def _fresh(model) -> None:
    _STATE.clear()
    _STATE.update(model=model, bound={}, acc={p: 0.0 for p, _ in PHASES}, calls={p: 0 for p, _ in PHASES},
                  depth={p: 0 for p, _ in PHASES}, bad={}, notes=[])


def _sync() -> bool:
    """Wait for the device at a boundary. False when the boundary is not observable (inside a CUDA-graph capture)."""
    try:
        import torch
    except ImportError:
        return True
    cuda = getattr(torch, "cuda", None)
    avail = getattr(cuda, "is_available", None)
    if cuda is None or not callable(avail) or not avail():
        return True
    capturing = getattr(cuda, "is_current_stream_capturing", None)
    if callable(capturing) and capturing():
        return False
    sync = getattr(cuda, "synchronize", None)
    if callable(sync):
        sync()
    return True


def _name(fn) -> str:
    """``module:qualname`` of a callable (through bound methods and functools wrappers' ``__wrapped__`` is NOT followed: the outermost name)."""
    f = getattr(fn, "__func__", fn)
    return f"{getattr(f, '__module__', None) or '?'}:{getattr(f, '__qualname__', None) or getattr(f, '__name__', None) or type(f).__name__}"


def _wrap(phase: str, fn: Callable) -> Callable:
    @functools.wraps(fn)
    def timed(*a, **k):
        st = _STATE
        if st["depth"][phase] > 0:                                # a nested call of the same phase: the outermost call is the boundary
            return fn(*a, **k)
        ok0 = _sync()
        t0 = time.perf_counter()
        st["depth"][phase] += 1
        try:
            return fn(*a, **k)
        finally:
            st["depth"][phase] -= 1
            ok1 = _sync()
            st["acc"][phase] += time.perf_counter() - t0
            st["calls"][phase] += 1
            if not (ok0 and ok1):
                st["bad"][phase] = "a boundary fell inside a CUDA-graph capture (no synchronize possible)"
    setattr(timed, MARK, phase)
    return timed


def _resolve(model, path: Tuple[str, ...]):
    owner = model
    for p in path[:-1]:
        owner = getattr(owner, p, None)
        if owner is None:
            return None, None
    cur = getattr(owner, path[-1], None)
    return (owner, cur) if callable(cur) else (owner, None)


def _bind(phase: str, path: Tuple[str, ...]) -> Optional[str]:
    """Wrap the callable at `path` (idempotent). Returns the wrapped callable's name, or None when the model has no such callable."""
    owner, cur = _resolve(_STATE["model"], path)
    if cur is None:
        return None
    if getattr(cur, MARK, None) == phase:
        return _STATE["bound"].get(phase) or _name(getattr(cur, "__wrapped__", cur))
    try:
        setattr(owner, path[-1], _wrap(phase, cur))
    except Exception as e:  # noqa: BLE001 — an owner that takes no attribute (not an ESMFold2Model): the phase is absent, by name
        _STATE["notes"].append(f"PHASE-NOTE {phase} not separable: cannot rebind {'.'.join(path)} on {type(owner).__name__} ({type(e).__name__})")
        return None
    return _name(cur)


def install(model) -> Dict[str, Optional[str]]:
    """Wrap the four phase callables on `model` (idempotent per model; a callable rebound since the last call is wrapped again, outermost).
    Returns phase -> ``module:qualname`` of the callable wrapped (None: absent on this model)."""
    if _STATE.get("model") is not model:
        _fresh(model)
    for phase, path in PHASES:
        _STATE["bound"][phase] = _bind(phase, path)
    return dict(_STATE["bound"])


def boundary_line(bound: Dict[str, Optional[str]]) -> str:
    """``PHASE-BOUNDARY lm=<path -> callable> trunk=... sampler=... conf=...`` — which callable each phase times in this process."""
    parts = []
    for phase, path in PHASES:
        parts.append(f"{phase}=model.{'.'.join(path)}->{bound.get(phase) or '-'}")
    return "PHASE-BOUNDARY " + " ".join(parts)


def begin() -> List[str]:
    """Open a fold window: zero the accumulators and re-wrap any phase callable rebound since install. Returns PHASE-NOTE lines (usually none)."""
    notes: List[str] = list(_STATE.get("notes") or [])
    if _STATE.get("model") is None:
        return notes
    for phase, path in PHASES:
        before = _STATE["bound"].get(phase)
        owner, cur = _resolve(_STATE["model"], path)
        if before is not None and cur is not None and getattr(cur, MARK, None) != phase:
            _STATE["bound"][phase] = _bind(phase, path)
            notes.append(f"PHASE-NOTE {phase} rebound since install: model.{'.'.join(path)} now times {_STATE['bound'][phase]} (outermost)")
        _STATE["acc"][phase] = 0.0; _STATE["calls"][phase] = 0; _STATE["depth"][phase] = 0
    _STATE["bad"] = {}; _STATE["notes"] = []
    return notes


def line(item_id: str, seed: int, total_s: float) -> Tuple[str, List[str]]:
    """The window's ``PHASE`` line and its ``PHASE-NOTE`` lines (`total_s`: the fold window's wall from the loop's own timer)."""
    notes: List[str] = list(_STATE.get("notes") or [])
    fields = []; summed = 0.0
    for phase, path in PHASES:
        key = FIELDS[phase]
        if _STATE.get("model") is None or _STATE["bound"].get(phase) is None:
            fields.append(f"{key}=-"); continue
        if phase in _STATE["bad"]:
            fields.append(f"{key}=NA"); notes.append(f"PHASE-NOTE {phase} not separable: {_STATE['bad'][phase]}"); continue
        if _STATE["calls"][phase] == 0:
            fields.append(f"{key}=NA")
            notes.append(f"PHASE-NOTE {phase} not observed: model.{'.'.join(path)} was not called inside this window (rebound past the timer or skipped by the route)")
            continue
        v = float(_STATE["acc"][phase]); summed += v
        fields.append(f"{key}={v:.3f}")
    if summed > float(total_s) + 1e-3:
        notes.append(f"PHASE-NOTE phases sum {summed:.3f}s > total {float(total_s):.3f}s: overlapping boundaries — do not read this line as a split")
    return f"PHASE item={item_id} seed={'none' if seed is None else int(seed)} " + " ".join(fields) + f" total_s={float(total_s):.3f}", notes
