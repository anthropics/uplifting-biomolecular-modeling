"""Per-rollout memo of step-invariant sampler work that a CAPTURED step reads — the machinery the OF3-family `atom_hoist` and `token_agg`
levers share (not a lever itself). Unlike :mod:`opt_core.capture.hoist` (ConstMemo: bypassed inside captures), this store SERVES INSIDE the
captured graph through address-stable buffers and one eager refresh step per rollout.

The diffusion sampler (`SampleDiffusion.forward`: 200 denoiser steps over the sample batch) recomputes, every step, quantities that depend only
on the rollout's inputs (the trunk outputs and the input features), never on the noisy coordinates or the timestep. A lever that memoises
such a quantity wraps its producer with `memo_call(what, key, compute)`: inside a rollout (`inside()`), a key that was filled in the CURRENT
rollout epoch answers from the store; any other call computes the stock expression (`compute()`, in the caller's context) and stores it.

Rollout epoch. `SampleDiffusion.forward` (and `_sample_rollout` where present) is the boundary: `arm_boundary()` wraps it at install and once
more at the first model forward (MODEL_CLASS.forward; every other add-on's import-time rebinding of the sampler is in place by then; markers keep each
function wrapped once). Entering the outermost boundary advances the epoch; the sampler instance carries it as `_of3opt_rollout_epoch`
(int) for cells that publish per-rollout buffers.

Address stability. Entries are keyed by producer + operand shapes/dtypes/device. On the eager route nothing survives the rollout (the store
is dropped at the boundary's exit). Under CUDA graphs the store is FROZEN in the sense a captured graph needs: a later rollout of the same key
REFRESHES the same buffers in place (`copy_`; the entry's recorded addresses are asserted unchanged, `AddressError` by name otherwise — a
same-key value of another dtype/stride included), an entry is evicted only when no live graph can hold it (last touched two or more rollouts
ago: the graphed step keeps ONE generation, captured or replayed in the previous or the current rollout), and a key that met the byte cap
(`set_cap_gb`, the sampler-memo cap every user shares) is skipped for good (`cap_skips`, sticky: the captured graph computes that quantity
inline, so it is never stored behind the graph's back). A CUDA-graph capture met while the cooperation below is NOT armed raises by name
(`CaptureError`): serving addresses into an unknown capture would be silent staleness at replay.

CUDA graphs (the fast-inference add-on's `of3_graphs`, OF3_CUDA_GRAPHS=1 — M_GRAPHS / GRAPHS_ENV). The graphed step replays captured kernels: Python-side memo logic
does not run at replay, so the buffers a graph captured must be refreshed BEFORE the first replay of each rollout. `arm_graphs_coop()` (run
at the first sampler call) places ONE wrapper immediately around the kit's graphed call (`of3_graphs.GraphedStep.__call__` — the class
attribute when it is the kit's own function, else the closure cell of the add-on wrapper that holds it, e.g. the trunk-kernels pair
cache's): the FIRST denoiser call of every rollout runs eager through `GraphedStep.module` (every cell's hooks apply — the bf16 region, the
fp32 islands), its memo misses refresh the buffers, and calls 2..T replay. One eager step per rollout is the price. When the graphed call
cannot be located (an unknown wrapper shape, more than one live graph generation) the cooperation is REFUSED BY NAME and every store user
turns itself off (`refuse_all`), so the step runs exactly as the line ships it.
"""
from __future__ import annotations

import inspect
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

PREFIX = "[opt_core/rollout_memo]"                       # the kit adapter names itself here (configure(PREFIX=...))
M_DIFFUSION: Optional[str] = None                        # the sampler's module (….core.model.structure.diffusion_module): SAMPLER_CLASS (the rollout boundary) lives here
SAMPLER_CLASS = "SampleDiffusion"
BOUNDARY_METHODS = ("forward", "_sample_rollout")        # the sampler methods that are one rollout each (the second where present)
M_MODEL: Optional[str] = None                            # the model module (….projects.of3_all_atom.model): MODEL_CLASS's first forward re-arms the boundary once (every add-on's rebinding is in place by then)
MODEL_CLASS: Optional[str] = None
M_GRAPHS = "of3_graphs"                                   # the fast-inference add-on's graphed-step module (GraphedStep, MAX_GEN) and its switch
GRAPHS_ENV = "OF3_CUDA_GRAPHS"
CONFIGURABLE = ("PREFIX", "M_DIFFUSION", "SAMPLER_CLASS", "BOUNDARY_METHODS", "M_MODEL", "MODEL_CLASS", "M_GRAPHS", "GRAPHS_ENV")
EPOCH_ATTR = "_of3opt_rollout_epoch"                    # on the SampleDiffusion instance: the current rollout epoch (int)
BOUNDARY_MARK = "_of3opt_memo_boundary"
INNER_MARK = "_of3opt_memo_inner"

class AddressError(RuntimeError):
    """A stored entry would have to change address (or layout) while a captured graph may read it."""


class CaptureError(RuntimeError):
    """A CUDA-graph capture is under way while the graphs cooperation is not armed: the store refuses to serve into it."""


_BOUND: Dict[str, Any] = {}                              # the engine words as the ONE binding set them (a second, different binding raises)


def configure(**kw) -> None:
    """The kit's binding (one site per kit): assigns this module's engine words (log prefix, the sampler / model / graphed-step module and class
    names) before install; unknown names raise; re-binding a word to a DIFFERENT value raises (last-writer-wins would silently move every user)."""
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        if k in _BOUND and _BOUND[k] != v:
            raise ValueError(f"{__name__}.configure: {k} already bound to {_BOUND[k]!r}, refusing to rebind it to {v!r} (one engine binding per process)")
        _BOUND[k] = v; globals()[k] = v


STATE: Dict[str, Any] = {"armed": False, "rollouts": 0, "fills": 0, "hits": 0, "bytes": 0, "bytes_peak": 0, "entries": 0, "cap_gb": 1.5,
                         "cap_skips": {}, "evictions_runs": 0, "refresh_eager_steps": 0, "graphs_coop": "pending", "boundaries": [], "errors": {}, "first_fill": None}
_EPOCH = {"n": 0, "depth": 0}
_GRAPHS = {"active": False, "checked": False}
_USERS: Dict[str, dict] = {}                             # lever name -> {"on": callable -> bool, "refuse": callable(reason), "on_boundary": callable(sampler) | None, "on_drop": callable | None}
_LOCK = threading.RLock()
_ONCE = set()


def _log(msg: str, once_key=None) -> None:
    if once_key is not None:
        if once_key in _ONCE:
            return
        _ONCE.add(once_key)
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


# ----------------------------------------------------------------------------------------------------------------- users
def register_user(name: str, on: Callable[[], bool], refuse: Callable[[str], None], on_boundary: Optional[Callable] = None,
                  on_drop: Optional[Callable] = None) -> None:
    """A lever that keeps entries in the store: `on()` says whether it is serving, `refuse(reason)` turns it off by name (graphs cooperation
    refused), `on_boundary(sampler)` runs at every rollout entry (e.g. to arm instance memos), `on_drop()` when the store is dropped."""
    _USERS[name] = {"on": on, "refuse": refuse, "on_boundary": on_boundary, "on_drop": on_drop}


def users_on() -> bool:
    return any(u["on"]() for u in _USERS.values())


def refuse_all(reason: str) -> None:
    for name, u in _USERS.items():
        try:
            u["refuse"](reason)
        except Exception as e:  # noqa: BLE001
            _count(STATE["errors"], f"refuse:{name}:{type(e).__name__}")


def set_cap_gb(gb: float) -> None:
    """The ONE byte cap of the store (every user's entries count against it)."""
    STATE["cap_gb"] = float(gb)


# ----------------------------------------------------------------------------------------------------------------- epoch
def epoch() -> int:
    return _EPOCH["n"]


def inside() -> bool:
    return _EPOCH["depth"] > 0


def active() -> bool:
    """Inside a rollout, inference (no grad): the condition under which a memo may answer."""
    if not inside():
        return False
    try:
        import torch
        return not torch.is_grad_enabled()
    except Exception:  # noqa: BLE001
        return False


def _enter_rollout(sampler) -> None:
    if _EPOCH["depth"] == 0:
        _EPOCH["n"] += 1
        STATE["rollouts"] += 1
    _EPOCH["depth"] += 1
    try:
        sampler.__dict__[EPOCH_ATTR] = _EPOCH["n"]
    except Exception:  # noqa: BLE001
        pass
    for name, u in _USERS.items():
        if u["on_boundary"] is not None and u["on"]():
            try:
                u["on_boundary"](sampler)
            except Exception as e:  # noqa: BLE001
                from ..oom import is_oom
                if is_oom(e):
                    raise
                _count(STATE["errors"], f"boundary:{name}:{type(e).__name__}")


def _exit_rollout() -> None:
    _EPOCH["depth"] = max(0, _EPOCH["depth"] - 1)
    if _EPOCH["depth"] == 0 and not _GRAPHS["active"]:
        STORE.drop_all()                                     # eager route: nothing survives the rollout (memory back before the next one)


# ----------------------------------------------------------------------------------------------------------------- the store
def _same_layout(a: Tuple, b: Tuple) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device or x.stride() != y.stride():
            return False
    return True


def sig(t) -> tuple:
    """A tensor's memo signature: shape, dtype, device."""
    return (tuple(t.shape), t.dtype, str(t.device))


class _Store:
    """key -> {epoch (last fill/refresh), touched (last fill/refresh/hit), t: tuple(tensors), ptrs, nbytes, what}. get() answers only for the
    current epoch; put() keeps addresses when the key is known (under graphs: always in place, asserted)."""

    def __init__(self):
        self.ent: Dict[Any, dict] = {}
        self.skipped: Dict[Any, int] = {}                    # under graphs: key -> the epoch it first met the cap (sticky skip)

    def get(self, key):
        e = self.ent.get(key)
        if e is not None and e["epoch"] == _EPOCH["n"]:
            e["touched"] = _EPOCH["n"]
            return e["t"]
        return None

    def evictable(self):
        """Keys no live graph can hold: on the eager route every stale-epoch entry (nothing outlives a rollout there anyway); under graphs only
        entries last touched two or more rollouts ago — the graphed step keeps ONE generation, captured or replayed in the previous or the current
        rollout, and a graph reads only entries touched in the rollout it ran in."""
        n = _EPOCH["n"]
        if _GRAPHS["active"]:
            return [k for k, v in self.ent.items() if v["touched"] < n - 1]
        return [k for k, v in self.ent.items() if v["epoch"] != n]

    def put(self, key, tensors: Tuple, what: str):
        n = _EPOCH["n"]; frozen = _GRAPHS["active"]
        e = self.ent.get(key)
        if e is not None:
            if _same_layout(e["t"], tensors):
                if frozen:
                    for d, s_, p0 in zip(e["t"], tensors, e["ptrs"]):
                        if d.data_ptr() != p0:               # cannot happen by construction (the entry holds its tensors); asserted because a graph depends on it
                            raise AddressError(f"{PREFIX} {what}: a memoised buffer moved ({p0:#x} -> {d.data_ptr():#x}) while a captured graph may read it")
                        if d.data_ptr() != s_.data_ptr():
                            d.copy_(s_)                      # refresh in place: a captured graph reads these addresses
                    out = e["t"]
                else:
                    e["t"] = tuple(tensors); e["ptrs"] = tuple(t.data_ptr() for t in e["t"]); out = e["t"]
                e["epoch"] = n; e["touched"] = n
                return out
            if frozen:                                       # same key (the operand shapes are in it), another dtype / stride: not refreshable in place
                raise AddressError(f"{PREFIX} {what}: the producer's output layout changed between rollouts ({[sig(t) for t in e['t']]} -> "
                                   f"{[sig(t) for t in tensors]}) while a captured graph may read the stored buffers — refusing to re-address them")
            STATE["bytes"] -= e["nbytes"]; del self.ent[key]
        if frozen and key in self.skipped:                   # sticky: the graph captured at this shape computes the quantity inline; storing it now serves no replay
            _count(STATE["cap_skips"], what)
            return None
        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        cap = STATE["cap_gb"] * 2 ** 30
        if STATE["bytes"] + nbytes > cap:
            for k in self.evictable():
                STATE["bytes"] -= self.ent[k]["nbytes"]; del self.ent[k]
            _count(STATE, "evictions_runs")
        if STATE["bytes"] + nbytes > cap:
            if frozen:
                self.skipped[key] = n
            _count(STATE["cap_skips"], what)
            _log("cap %.2f GB reached: %s (%.0f MiB) not memoised%s — computed per step" % (STATE["cap_gb"], what, nbytes / 2 ** 20, " at this shape (sticky under graphs)" if frozen else " this rollout"),
                 once_key=("cap", what))
            STATE["entries"] = len(self.ent)
            return None
        t = tuple(tensors)
        self.ent[key] = {"epoch": n, "touched": n, "t": t, "ptrs": tuple(x.data_ptr() for x in t), "nbytes": nbytes, "what": what}
        STATE["bytes"] += nbytes; STATE["bytes_peak"] = max(STATE["bytes_peak"], STATE["bytes"]); STATE["entries"] = len(self.ent)
        return t

    def drop_all(self) -> None:
        self.ent.clear(); self.skipped.clear(); STATE["bytes"] = 0; STATE["entries"] = 0
        for name, u in _USERS.items():
            if u["on_drop"] is not None:
                try:
                    u["on_drop"]()
                except Exception as e:  # noqa: BLE001
                    _count(STATE["errors"], f"drop:{name}:{type(e).__name__}")


STORE = _Store()


def memo_call(what: str, key, compute: Callable):
    """The memo protocol: a current-epoch hit -> the stored tensors; else `compute()` (the stock expression, in the caller's context) -> stored
    (or not: the cap) -> its tensors. Returns a tuple of tensors."""
    if not _GRAPHS["active"] and _capturing():                 # a graph is being captured around us while the cooperation is not armed: never serve into it
        raise CaptureError(f"{PREFIX} {what}: a CUDA-graph capture is under way while the memo's graphs cooperation is {STATE['graphs_coop']!r} — "
                           "the store would hand this capture addresses it later drops or re-fills unseen; arm_graphs_coop() must locate the graphed call first")
    got = STORE.get(key)
    if got is not None:
        STATE["hits"] += 1
        return got
    out = compute()
    out = tuple(out) if isinstance(out, (tuple, list)) else (out,)
    kept = STORE.put(key, out, what)
    STATE["fills"] += 1
    if STATE["first_fill"] is None:
        STATE["first_fill"] = what
        _log("first fill (%s) at rollout epoch %d; graphs_coop=%s" % (what, _EPOCH["n"], STATE["graphs_coop"]))
    return out if kept is None else kept


def _capturing() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def unwrap(t: Tuple, single: bool):
    return t[0] if single else t


def memo_instance_forward(m, what: str, serving: Callable[[], bool]) -> bool:
    """Memoise `m.forward(x)` per rollout (instance attribute; the class is untouched): key = (what, id(m), sig(x)). Idempotent."""
    if "forward" in m.__dict__:
        return False
    cls_forward = type(m).forward

    def forward(x, *a, **kw):
        if a or kw or not (serving() and active()):
            return cls_forward(m, x, *a, **kw)
        out = memo_call(what, (what, id(m), sig(x)), lambda: cls_forward(m, x))
        return out[0]
    forward.__wrapped__ = cls_forward
    m.__dict__["forward"] = forward
    return True


# ----------------------------------------------------------------------------------------------------------------- boundary
def _wrap_boundary(cls, name: str) -> bool:
    fn = getattr(cls, name, None)
    if fn is None or getattr(fn, BOUNDARY_MARK, False):
        return False

    def bounded(self, *a, **kw):
        _on_first_sampler_call()
        _enter_rollout(self)
        try:
            return fn(self, *a, **kw)
        finally:
            _exit_rollout()
    setattr(bounded, BOUNDARY_MARK, True); bounded.__wrapped__ = fn; bounded.__name__ = name
    for attr in ("_of3t_boundary",):                       # the pair cache's boundary marker travels with the wrapper
        if hasattr(fn, attr):
            setattr(bounded, attr, getattr(fn, attr))
    setattr(cls, name, bounded)
    STATE["boundaries"].append(f"{cls.__name__}.{name}")
    return True


def arm_boundary() -> None:
    """The sampler's rollout methods (SAMPLER_CLASS.BOUNDARY_METHODS in M_DIFFUSION) as the rollout boundary: wrapped now, and re-checked once at
    the first MODEL_CLASS.forward (any later import-time rebinding by another add-on is then wrapped too). Idempotent."""
    with _LOCK:
        if STATE["armed"]:
            return
        STATE["armed"] = True
    if not (M_DIFFUSION and SAMPLER_CLASS):
        raise RuntimeError(f"{PREFIX} rollout_memo is not bound to an engine: the kit adapter must configure(M_DIFFUSION=, SAMPLER_CLASS=, M_MODEL=, MODEL_CLASS=) before install")
    dmod = sys.modules.get(M_DIFFUSION)
    if dmod is None:
        import importlib
        dmod = importlib.import_module(M_DIFFUSION)
    Sampler = getattr(dmod, SAMPLER_CLASS)
    for n in BOUNDARY_METHODS:
        _wrap_boundary(Sampler, n)
    mod = sys.modules.get(M_MODEL) if M_MODEL else None
    OF3 = getattr(mod, MODEL_CLASS, None) if (mod is not None and MODEL_CLASS) else None
    if OF3 is None:
        return
    orig = OF3.forward

    def forward(self, batch, *a, **k):
        if not getattr(OF3, "_of3opt_memo_boundaries_done", False):
            d2 = sys.modules.get(M_DIFFUSION)
            if d2 is not None and hasattr(d2, SAMPLER_CLASS):
                for n in BOUNDARY_METHODS:
                    _wrap_boundary(getattr(d2, SAMPLER_CLASS), n)
            OF3._of3opt_memo_boundaries_done = True
        return orig(self, batch, *a, **k)
    forward.__wrapped__ = orig
    for attr in dir(orig):                                   # keep foreign markers (e.g. the forward timer's) visible on the wrapper
        if attr.startswith("_of3") and not hasattr(forward, attr):
            try:
                setattr(forward, attr, getattr(orig, attr))
            except Exception:  # noqa: BLE001
                pass
    OF3.forward = forward


# ----------------------------------------------------------------------------------------------------------------- graphs cooperation
def _find_graphed_call():
    """-> ("class", cls, None) when GraphedStep.__call__ is the kit's own function; ("cell", holder, cell) when a wrapper's closure cell holds
    it; ("done", …) when this module's wrapper is already in place; (None, reason, None) otherwise."""
    G = sys.modules.get(M_GRAPHS)
    cls = getattr(G, "GraphedStep", None) if G is not None else None
    if cls is None:
        return None, "no_graphed_step", None
    if getattr(G, "MAX_GEN", 1) != 1:
        return None, "max_gen_%s" % getattr(G, "MAX_GEN", "?"), None

    def is_orig(f):
        return inspect.isfunction(f) and f.__module__ == M_GRAPHS and f.__qualname__ == "GraphedStep.__call__"
    top = cls.__call__
    if getattr(top, INNER_MARK, False):
        return "done", cls, None
    if is_orig(top):
        return "class", cls, None
    seen = set()

    def walk(f, depth):
        if depth > 6 or id(f) in seen:
            return None
        seen.add(id(f))
        for cell in (getattr(f, "__closure__", None) or ()):
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if getattr(v, INNER_MARK, False):
                return ("done", f, cell)
            if is_orig(v):
                return ("cell", f, cell)
            if inspect.isfunction(v):
                r = walk(v, depth + 1)
                if r is not None:
                    return r
        return None
    r = walk(top, 0)
    if r is None:
        return None, "graphed_call_not_found", None
    return r


def arm_graphs_coop() -> None:
    """Place the per-rollout eager refresh step immediately around the kit's graphed call. Idempotent; refuses by name."""
    if _GRAPHS["checked"]:
        return
    _GRAPHS["checked"] = True
    if os.environ.get(GRAPHS_ENV) != "1" or M_GRAPHS not in sys.modules:
        STATE["graphs_coop"] = "eager_route"; return
    kind, holder, cell = _find_graphed_call()
    if kind is None:
        STATE["graphs_coop"] = "refused:%s" % holder
        refuse_all("graphs_coop:%s" % holder)
        _log("REFUSED under %s=1:" % GRAPHS_ENV + " the per-rollout refresh cannot be placed inside the graphed call (%s); the memo levers are off, the step runs as the line ships it" % holder)
        return
    if kind == "done":
        STATE["graphs_coop"] = "inner"; _GRAPHS["active"] = True; return
    orig = holder.__call__ if kind == "class" else cell.cell_contents
    last = {"epoch": -1}

    def inner(gs, batch, xl_noisy, t, si_input, si_trunk, zij_trunk, **kw):
        if users_on() and inside() and last["epoch"] != _EPOCH["n"]:
            last["epoch"] = _EPOCH["n"]
            # the first denoiser call of this rollout: eager, through the module (every cell's hooks apply); its memo misses refresh the buffers
            out = gs.module(batch=batch, xl_noisy=xl_noisy, token_mask=batch["token_mask"], atom_mask=batch["atom_mask"], t=t,
                            si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, **kw)
            STATE["refresh_eager_steps"] += 1
            return out
        return orig(gs, batch, xl_noisy, t, si_input, si_trunk, zij_trunk, **kw)
    setattr(inner, INNER_MARK, True); inner.__wrapped__ = orig; inner.__qualname__ = "GraphedStep.__call__[rollout_memo]"
    if kind == "class":
        holder.__call__ = inner; STATE["graphs_coop"] = "class"
    else:
        cell.cell_contents = inner; STATE["graphs_coop"] = "cell:%s.%s" % (getattr(holder, "__module__", "?"), getattr(holder, "__qualname__", "?"))
    _GRAPHS["active"] = True
    _log("graphs cooperation armed (%s): the first denoiser call of every rollout runs eager and refreshes the memoised buffers; calls 2..T replay" % STATE["graphs_coop"])


def graphs_active() -> bool:
    return _GRAPHS["active"]


def _on_first_sampler_call() -> None:
    if _GRAPHS["checked"] or not users_on():
        return
    try:
        arm_graphs_coop()
    except Exception as e:  # noqa: BLE001
        from ..oom import is_oom
        if is_oom(e):
            raise
        STATE["graphs_coop"] = "refused:%s" % type(e).__name__
        refuse_all("graphs_coop:%s" % type(e).__name__)
        _log("REFUSED: graphs cooperation failed: %r; the memo levers are off" % (e,))


def fields() -> str:
    """The store's census words for a lever's LEVER line."""
    return ("graphs_coop=%s rollouts=%d fills=%d hits=%d refresh_eager_steps=%d bytes_peak_mib=%d entries=%d cap_gb=%.2f cap_skips=%s sticky_skips=%d memo_errors=%s" % (
        STATE["graphs_coop"], STATE["rollouts"], STATE["fills"], STATE["hits"], STATE["refresh_eager_steps"], STATE["bytes_peak"] // 2 ** 20,
        STATE["entries"], STATE["cap_gb"], ",".join("%s:%d" % kv for kv in sorted(STATE["cap_skips"].items())) or "none", len(STORE.skipped),
        ",".join("%s:%d" % kv for kv in sorted(STATE["errors"].items())) or "none"))
