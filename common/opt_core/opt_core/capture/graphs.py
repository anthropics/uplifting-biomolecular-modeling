"""CUDA-graph capture and replay of a callable at static shapes: ONE discipline for every kit that graphs a diffusion step, a trunk
stack, a recycle step or an encoder.

Contract. :class:`GraphCache` wraps one callable site (the kit adapter decides which: "the denoiser step", "the 48-block stack"). A call
is keyed by the SIGNATURE of its arguments (shape, dtype, device and contiguity of every tensor leaf; the value of every non-tensor
leaf; the adapter's ``extra_key``) — never by tensor values. WHEN a key is captured is the ``policy``'s answer
(:mod:`opt_core.capture.pool`: fixed K-th sighting, break-even, pinned table, job plan; ``min_sightings`` is sugar for a fixed K — a shape
seen once never pays a capture under K=2); sightings before that run EAGER by name; the admitting sighting ALWAYS returns the plain eager
result computed before any warm-up (eager-first: a first replay handed back as the answer is where bit-exact test records were lost), i.e. it computes this call's result eagerly, then warms up on a side stream, captures the
callable into a CUDA graph reading STATIC copies of the arguments, replays it once on the same inputs and holds the replay BIT-EXACT to
the eager result (``verify>=1``; ``verify=2`` also holds the next call, on new inputs, before the graph is armed; ``verify_every=N`` re-holds every
N-th replay); later calls copy their arguments into the static buffers (arguments wrapped in :class:`Held` are read by reference instead —
roll-out constants the adapter refreshes in place, never copied per step), replay, and return the static outputs (cloned unless ``clone_outputs=False``). Replaying the kernels
eager would have launched, in the same order on the same data, is what makes the mechanism exact where the engine's eager path is
deterministic; the engine's equality suite decides the class, this module only refuses what would make it untrue:

* RNG. ``rng="forbid"`` (default): the callable must draw nothing from the CUDA generator — the generator state is compared across the
  warm-up; a callable that drew is refused BY NAME (the state is restored first, so the stock stream is not consumed). ``rng="graph"``:
  draws inside the callable are allowed (torch registers the default CUDA generator with the graph and advances its offset per replay
  as an eager call would); warm-up draws are rolled back by restoring the generator state, and every checking replay is RNG-neutral
  (state saved and restored around it).
* Envelope, not failure — counted, printed once per key, never silent, never raised: a size/shape gate the adapter supplies (``gate``
  returns a reason string or an ``opt_core.mem.graph_gate`` GateDecision → ``eager_gate``, its reason printed as ``gate=``), autograd live (``eager_grad``), tensors off the one CUDA device (``eager_device``), first sightings
  (``eager_sighting``; ``eager_policy`` when the policy files the key as never-capture), a full cache under ``on_full="eager"`` (``eager_full``), the memory budget (``eager_budget``: a new key whose
  predicted private pool does not fit ``mem_budget_bytes`` runs eager and NOTHING captured is freed — never evict under a live neighbour).
* Failure — ``strict=True`` (every tested line) RAISES, and each exception is a cannot-run event carrying ``cannot_run = True``
  (``opt_core.gates.is_cannot_run``: the kit refuses the MODE by name, nothing continues eager under a graphed line's label): a capture that threw (:class:`CaptureFailed`, after
  :func:`recover_after_failed_capture` restored stream / allocator / generator so the exception is the only damage), an RNG draw under
  ``rng="forbid"`` or a replay that is not bit-exact the eager result (:class:`CaptureRefused`). ``strict=False`` (named development lines
  only) turns the same events into :meth:`GraphCache.disable`: every later call runs eager, counted ``eager_disabled`` with the reason
  (``on_fail="key"``: only the failed key stays eager, counted ``eager_failed``; the site keeps capturing other keys);
  :meth:`GraphCache.partial` is the reason the kit's exit verdict records (fail-closed unless the run recorded ``--allow-partial``).
* Pools. ``pool="private"`` (default): one memory pool per graph, so evicting one entry (``on_full="evict"``, LRU) frees only its own
  blocks. ``pool="generation"``: all graphs of the cache share one pool handle and are only ever dropped TOGETHER (:meth:`reset`;
  ``on_full`` defaults to ``"reset"``) — a single graph is never destroyed while a neighbour in its pool lives.
* Reset protocol (:meth:`reset`): synchronize first; drop graphs and their static buffers together; ``gc.collect()`` so the destructors
  run now; start a fresh pool; ``empty_cache()`` last. :meth:`new_item` is the adapter's item boundary: with ``module=`` given,
  parameter/buffer storage that moved since capture (a ``.cpu()``/``.cuda()`` round trip) resets by name instead of replaying stale
  addresses; ``reset_on_item=True`` makes every item its own generation. ``mode_fn`` (e.g. :func:`numerics_mode`) folds the process's
  numerics mode into the generation: a change since the last capture resets by name before the call runs.
* Evidence. :meth:`stats` is the census; :meth:`evidence_fields` / :meth:`evidence_line` are the ONE activation-evidence line the kit
  prints per arm per process under its own tag: ``LEVER name=<name> state=<on|skipped> impl=cuda_graphs graphs=<active|captured|eager|idle> captures= replays=
  eager=<kind:n,…> refused=<kind:n,…> failures= resets= keys= …`` — "wired AND active" reads ``replays>0``; every call the site received
  is in exactly one bucket (captures + replays + eager kinds + refused kinds + failures).

Everything CUDA is imported inside the functions that need it (``import opt_core.capture.graphs`` is standard library only; Python 3.8
floor); a process without torch or without a CUDA device gets :class:`CaptureUnavailable` from the first :meth:`GraphCache.run` under
``strict`` (a named refusal), or a cache disabled by name otherwise. The per-engine glue — which callable, the static argument layout,
the size-gate thresholds, where noise is drawn, the mode-table switch — lives in the kit's adapter, never here.
"""
from __future__ import annotations

import gc
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import report
from ..oom import is_oom
from . import hoist as _hoist

EAGER_KINDS = ("gate", "grad", "device", "sighting", "policy", "full", "budget", "unarmed", "failed", "disabled")
REFUSED_KINDS = ("rng", "verify", "mutates_inputs")


class CaptureUnavailable(RuntimeError):
    """torch or a CUDA device is missing (or torch lacks ``torch.cuda.CUDAGraph``): the named refusal of the whole mechanism. A cannot-run
    event (``opt_core.gates.is_cannot_run``): the kit refuses the mode by name."""

    cannot_run = True


class CaptureRefused(RuntimeError):
    """The callable cannot be graphed truthfully at this key: it drew from the CUDA generator under ``rng="forbid"``, or its replay was not
    bit-exact its eager result. ``kind`` is ``"rng"`` or ``"verify"``; ``detail`` carries the max-abs difference of a check refusal."""

    cannot_run = True

    def __init__(self, kind: str, message: str, detail: Optional[dict] = None):
        super().__init__(message)
        self.kind = kind
        self.detail = dict(detail or {})


class CaptureFailed(RuntimeError):
    """The capture raised. ``report`` is :func:`recover_after_failed_capture`'s record of what was restored; ``healthy`` False means the
    process must not continue (its later outputs cannot be trusted to equal the stock function's). A cannot-run event
    (``opt_core.gates.is_cannot_run``)."""

    cannot_run = True

    def __init__(self, message: str, recovery: Optional[dict] = None):
        super().__init__(message)
        self.report = dict(recovery or {})
        self.healthy = bool(self.report.get("healthy", False))


# ----------------------------------------------------------------------------------------------------------------- torch, lazily
def _torch():
    try:
        import torch  # noqa: PLC0415 — the one lazy import of the mechanism
    except Exception as e:  # noqa: BLE001
        raise CaptureUnavailable(f"CUDA graphs unavailable: torch import failed ({e!r})") from None
    return torch


def availability() -> Tuple[bool, Optional[str]]:
    """``(ok, reason)``: torch importable, a CUDA device present, ``torch.cuda.CUDAGraph`` / ``graph`` / ``graph_pool_handle`` present."""
    try:
        torch = _torch()
    except CaptureUnavailable as e:
        return False, str(e)
    if not torch.cuda.is_available():
        return False, "CUDA graphs unavailable: no CUDA device"
    if not (hasattr(torch.cuda, "CUDAGraph") and hasattr(torch.cuda, "graph") and hasattr(torch.cuda, "graph_pool_handle")):
        return False, f"CUDA graphs unavailable: torch {getattr(torch, '__version__', '?')} has no torch.cuda.CUDAGraph/graph/graph_pool_handle"
    return True, None


def _graph_context(torch, graph, pool, stream, capture_error_mode):
    """``torch.cuda.graph(...)`` with ``capture_error_mode`` where this torch has it (2.1+), without it otherwise (recorded by the caller)."""
    try:
        return torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode), capture_error_mode
    except TypeError:
        return torch.cuda.graph(graph, pool=pool, stream=stream), "unsupported"


# ----------------------------------------------------------------------------------------------------------------- argument trees
def _is_tensor(x) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(x, torch.Tensor)


def tensor_signature(obj) -> Any:
    """The hashable signature of an argument tree: tensors by (shape, dtype, device, contiguity) — never by value or address; lists,
    tuples and dicts structurally; other leaves by value (flags, ints, floats, strings, None; an unhashable leaf is a TypeError naming it)."""
    if isinstance(obj, Held):
        t = obj.tensor
        return ("H", tuple(t.shape), str(t.dtype), str(t.device), t.data_ptr(), tuple(t.stride()))
    if _is_tensor(obj):
        return ("T", tuple(obj.shape), str(obj.dtype), str(obj.device), bool(obj.is_contiguous()))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, tuple(tensor_signature(x) for x in obj))
    if isinstance(obj, dict):
        return ("dict", tuple((k, tensor_signature(obj[k])) for k in sorted(obj, key=str)))
    try:
        hash(obj)
    except TypeError:
        raise TypeError(f"graph key: unhashable non-tensor argument leaf of type {type(obj).__name__} — pass it through key_fn/extra_key or make it hashable") from None
    return ("v", obj)


def tree_tensors(obj, out: Optional[list] = None) -> list:
    """Every tensor leaf of an argument tree, depth first (dict keys in ``str`` order)."""
    out = [] if out is None else out
    if isinstance(obj, Held):
        out.append(obj.tensor)
    elif _is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            tree_tensors(x, out)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            tree_tensors(obj[k], out)
    return out


def tree_clone(obj):
    """A structural copy with every tensor leaf cloned (the static buffers of a capture; the returned outputs of a replay)."""
    if isinstance(obj, Held):
        return obj.tensor                                       # by reference: the graph reads the caller's buffer
    if _is_tensor(obj):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(tree_clone(x) for x in obj)
    if isinstance(obj, list):
        return [tree_clone(x) for x in obj]
    if isinstance(obj, dict):
        return {k: tree_clone(v) for k, v in obj.items()}
    return obj


def tree_copy_into(static, live) -> int:
    """Copy the tensor leaves of ``live`` into the matching static buffers in place (a leaf that IS the static buffer is skipped). Returns
    the number of copies issued."""
    n = 0
    if live is None or isinstance(live, Held):
        return 0                                                # Held: same buffer by construction (its address is in the key); None: a stripped Held slot
    if _is_tensor(static):
        if static.data_ptr() != live.data_ptr():
            static.copy_(live, non_blocking=True)
            n = 1
        return n
    if isinstance(static, (list, tuple)):
        for s, l in zip(static, live):
            n += tree_copy_into(s, l)
    elif isinstance(static, dict):
        for k in static:
            n += tree_copy_into(static[k], live[k])
    return n


def bitwise_equal(a, b) -> bool:
    """Bit-exact equality of two output trees (NaN payloads compare by bits, so equal NaNs are equal)."""
    if _is_tensor(a) or _is_tensor(b):
        if not (_is_tensor(a) and _is_tensor(b)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.numel() == 0:
            return True
        torch = _torch()
        try:
            return bool(torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8), b.detach().contiguous().reshape(-1).view(torch.uint8)))
        except (RuntimeError, TypeError):
            return bool(torch.equal(a, b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(bitwise_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(bitwise_equal(a[k], b[k]) for k in a)
    return a == b


def max_abs_diff(a, b) -> Optional[float]:
    """The largest |a-b| over the floating tensor leaves of two trees (None when there are none) — the detail of a check refusal."""
    worst = None
    for x, y in zip(tree_tensors(a), tree_tensors(b)):
        if x.shape == y.shape and x.is_floating_point() and x.numel():
            d = float((x.detach().float() - y.detach().float()).abs().max().item())
            worst = d if worst is None else max(worst, d)
    return worst


def storage_signature(module) -> tuple:
    """The addresses (+ device) of every parameter and buffer of a module: a captured graph baked them in, so a module whose storage moved
    (``.cpu()``/``.cuda()`` round trip, re-loaded weights) must not be replayed."""
    return tuple((t.data_ptr(), str(t.device)) for t in list(module.parameters()) + list(module.buffers()))


class Held:
    """An argument the graph reads (and may write) BY REFERENCE: no static copy at capture, no copy-in at replay; in-place updates the callable
    makes to it are visible to the caller exactly as in eager (it is excluded from the mutates-inputs check). Around a capture and every
    checking the module snapshots the Held buffers, starts each warm-up call, the capture and the checking replay from the contents
    THIS call saw, compares the buffers' post-replay contents with their post-eager contents (an in-place write is checked like an output),
    and leaves them exactly one call advanced — the cost is one temporary copy of each Held buffer at capture/checking time only. For roll-out constants (hoisted pair
    biases, masks) the adapter keeps in a persistent buffer and refreshes once per roll-out — the graph replays against the buffer's
    current contents. The buffer's address is part of the key (a different buffer is a different key, captured by name); its contents
    are the adapter's responsibility (that is the point). ``Held(t)`` anywhere in the argument tree."""
    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor


def _unwrap(obj):
    """The argument tree with every :class:`Held` replaced by its tensor (what ``fn`` receives)."""
    if isinstance(obj, Held):
        return obj.tensor
    if isinstance(obj, tuple):
        return tuple(_unwrap(x) for x in obj)
    if isinstance(obj, list):
        return [_unwrap(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    return obj


def _held_tensors(args, kwargs) -> list:
    """The tensors passed as :class:`Held` (by-reference buffers), depth first."""
    out: list = []

    def walk(o):
        if isinstance(o, Held):
            out.append(o.tensor)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                walk(o[k])
    walk(args)
    walk(kwargs)
    return out


def _restore(tensors, snaps) -> None:
    for t, snap in zip(tensors, snaps):
        t.copy_(snap)


def _strip_held(obj):
    """The argument tree with every :class:`Held` leaf replaced by ``None``: Held buffers are read/WRITE by reference (the callable may update
    them in place — that is visible to the caller exactly as in eager), so they are excluded from the mutates-inputs comparison."""
    if isinstance(obj, Held):
        return None
    if isinstance(obj, tuple):
        return tuple(_strip_held(x) for x in obj)
    if isinstance(obj, list):
        return [_strip_held(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip_held(v) for k, v in obj.items()}
    return obj


def modules_in_train_mode(*modules) -> list:
    """Names of sub-modules with ``training=True`` (dropout live): a graphed callable over them is refused under ``rng="forbid"``."""
    out = []
    for m in modules:
        if m is None:
            continue
        for name, sub in m.named_modules():
            if getattr(sub, "training", False):
                out.append(name or type(sub).__name__)
    return out


def _gate_reason(decision) -> Optional[str]:
    """The eager reason of a gate's answer, or None to proceed: ``None`` | a reason string | a GateDecision-like object with ``mode`` in
    {"capture", "eager"} and ``reason`` (the memory/token-gated DECISION is :func:`opt_core.mem.graph_gate.decide`'s; this layer consumes it and
    prints its reason as ``gate=`` on the LEVER line — no second wording here)."""
    if decision is None:
        return None
    if isinstance(decision, str):
        return decision or None
    mode = getattr(decision, "mode", None)
    if mode is None and isinstance(decision, dict):
        mode, reason = decision.get("mode"), decision.get("reason")
    else:
        reason = getattr(decision, "reason", None)
    if mode == "capture":
        return None
    if mode == "eager":
        return str(reason or "gate: eager (no reason given)")
    raise TypeError(f"gate returned {type(decision).__name__}: expected None, a reason string, or a GateDecision with mode in ('capture','eager')")


def numerics_mode() -> tuple:
    """The process-wide torch numerics switches a captured graph silently depends on: TF32 (matmul, cuDNN), float32 matmul precision,
    deterministic algorithms, cuDNN deterministic/benchmark, autocast state. A kit's ``mode_fn`` extends it with its own switches (kernel
    backend flags, chunk sizes)."""
    torch = _torch()
    try:
        autocast = (bool(torch.is_autocast_enabled()), str(torch.get_autocast_gpu_dtype()))
    except Exception:  # noqa: BLE001
        autocast = None
    return (bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32), str(torch.get_float32_matmul_precision()),
            bool(torch.are_deterministic_algorithms_enabled()), bool(torch.backends.cudnn.deterministic), bool(torch.backends.cudnn.benchmark), autocast)


# ----------------------------------------------------------------------------------------------------------------- failed-capture recovery
def end_dangling_capture() -> bool:
    """If the CURRENT stream is still capturing (an exception skipped ``capture_end``), end it into a throw-away graph."""
    torch = _torch()
    try:
        if torch.cuda.is_current_stream_capturing():
            g = torch.cuda.CUDAGraph()
            try:
                g.capture_end()
            except Exception:  # noqa: BLE001
                pass
            del g
            return not torch.cuda.is_current_stream_capturing()      # True only if the dangling capture actually ended
    except Exception:  # noqa: BLE001
        pass
    return False


def reset_generator_capture_state() -> bool:
    """A trivial SUCCESSFUL capture (one memset + add, no RNG op) on a fresh side stream: runs the default CUDA generator's capture epilogue,
    clearing the 'capturing' flag a failed capture leaves behind ("Offset increment outside graph capture" on every later eager draw).
    Numerics-free: the philox offset is not advanced."""
    torch = _torch()
    try:
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            torch.cuda.synchronize()
            g.capture_begin()
            x = torch.zeros(1, device="cuda")
            x.add_(1)
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        del g
        return True
    except Exception:  # noqa: BLE001
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        return False


def release_failed_capture_pool(graph=None, pool=None, device=None, capture_began: Optional[bool] = None, capture_ended: bool = False) -> dict:
    """The allocator bookkeeping ``capture_end`` would have done, for a capture that FAILED: ``capture_begin`` starts allocating to the pool
    before the stream capture begins and only a successful ``capture_end`` stops it, so after a failed capture the allocator still lists the
    capture as underway (the next ``empty_cache()`` trips ``captures_underway.empty()``) and the pool's use count is never returned. Ends the
    allocation to the pool if underway and releases the pool exactly once — ONLY for a capture that did not complete (a completed graph owns
    its bookkeeping; a second release would free a shared pool under live graphs). Uses torch's private bindings when present; reports what
    it did either way."""
    torch = _torch()
    rep: Dict[str, Any] = {"pool_id": None, "end_allocate_to_pool": None, "release_pool": None}
    if capture_ended:
        rep["skipped"] = "capture_end completed: the graph owns the pool bookkeeping"
        return rep
    pid = None
    if pool is not None:
        try:
            pid = tuple(pool)
        except TypeError:
            pid = None
    if pid is None and graph is not None:
        try:
            pid = tuple(graph.pool())
        except Exception as e:  # noqa: BLE001
            rep["pool_err"] = repr(e)[:120]
    if not pid or (pid[0] == 0 and pid[1] == 0):
        rep["skipped"] = "no pool id (capture_begin did not reach the allocator)"
        return rep
    rep["pool_id"] = pid
    dev = torch.cuda.current_device() if device is None else int(device)

    def _private(names):
        for n in names:
            f = getattr(torch._C, n, None)
            if f is not None:
                return f
        return None
    f_end = _private(("_cuda_endAllocateCurrentStreamToPool", "_cuda_endAllocateToPool"))
    f_rel = _private(("_cuda_releasePool",))
    was_underway = False
    if f_end is not None:
        try:
            f_end(dev, pid)
            was_underway = True
            rep["end_allocate_to_pool"] = "done"
        except Exception as e:  # noqa: BLE001 — "not currently recording": capture_end already ended it, or it never began
            rep["end_allocate_to_pool"] = f"not underway ({str(e)[:80]})"
    else:
        rep["end_allocate_to_pool"] = "unavailable in this torch"
    if f_rel is None:
        rep["release_pool"] = "unavailable in this torch"
    elif bool(capture_began) or was_underway:
        try:
            f_rel(dev, pid)
            rep["release_pool"] = "done"
        except Exception as e:  # noqa: BLE001
            rep["release_pool"] = f"error ({str(e)[:80]})"
    else:
        rep["release_pool"] = "skipped (capture never began allocating to the pool)"
    return rep


def allocator_health_probe() -> bool:
    """``empty_cache()`` plus a small allocation must work (this is what a failed capture left 'underway' breaks)."""
    torch = _torch()
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        x = torch.empty(1024, device="cuda")
        del x
        torch.cuda.empty_cache()
        return True
    except Exception:  # noqa: BLE001
        return False


def rng_health_probe() -> bool:
    """The default CUDA generator serves an eager draw — WITHOUT consuming the stock stream (CPU and CUDA states saved and restored)."""
    torch = _torch()
    try:
        dev = torch.cuda.current_device()
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(dev)
        try:
            torch.rand(1, device="cuda")
            torch.cuda.synchronize()
        finally:
            torch.cuda.set_rng_state(cuda_state, dev)
            torch.random.set_rng_state(cpu_state)
        return True
    except Exception:  # noqa: BLE001
        return False


def recover_after_failed_capture(graph=None, pool=None, device=None, capture_began: Optional[bool] = None, capture_ended: bool = False,
                                 raise_if_unhealthy: bool = True) -> dict:
    """From the ``except`` of ANY failed capture: (1) end a dangling stream capture; (2) the allocator epilogue of the failed capture;
    (3) take the default CUDA generator out of capture mode; (4) probe allocator and generator health (retrying 2-3 once); ``healthy`` False
    with ``raise_if_unhealthy`` raises :class:`CaptureFailed` — the process must not continue producing 'stock' outputs."""
    rep: Dict[str, Any] = {"dangling_capture_ended": end_dangling_capture()}
    rep["pool"] = release_failed_capture_pool(graph=graph, pool=pool, device=device, capture_began=capture_began, capture_ended=capture_ended)
    rep["generator_recovered"] = reset_generator_capture_state()
    rep["allocator_healthy"] = allocator_health_probe()
    rep["rng_healthy"] = rng_health_probe()
    if not (rep["rng_healthy"] and rep["allocator_healthy"]):
        rep["generator_recovered"] = reset_generator_capture_state() or rep["generator_recovered"]
        rep["allocator_healthy"] = allocator_health_probe()
        rep["rng_healthy"] = rng_health_probe()
    rep["healthy"] = bool(rep["rng_healthy"] and rep["allocator_healthy"])
    if raise_if_unhealthy and not rep["healthy"]:
        raise CaptureFailed(f"CUDA generator/allocator unusable after a failed graph capture ({rep}); refusing to continue", rep)
    return rep


# ----------------------------------------------------------------------------------------------------------------- the cache
class GraphByteBudget:
    """The memory budget shared by one or more caches: ``budget`` (an int, a callable evaluated once at the first admission, or None) and the
    running sum of measured graph-pool bytes. :class:`GraphCache` makes its own when none is passed; :class:`opt_core.capture.pool.GraphPool`
    passes one to every site so all sites draw on ONE budget."""

    def __init__(self, budget=None):
        self._spec = budget
        self._budget: Optional[int] = None
        self._resolved = budget is None
        self.bytes_total = 0

    @property
    def measuring(self) -> bool:
        return self._spec is not None

    def budget(self) -> Optional[int]:
        if not self._resolved:
            v = self._spec() if callable(self._spec) else self._spec
            self._budget = int(v) if v is not None else None
            self._resolved = True
        return self._budget

    def add(self, n: int) -> None:
        self.bytes_total += int(n or 0)

    def sub(self, n: int) -> None:
        self.bytes_total = max(0, self.bytes_total - int(n or 0))


MemoryBudget = GraphByteBudget          # additive alias (the earlier name of the byte budget)


class _Entry:
    __slots__ = ("key", "graph", "pool", "static_args", "static_kwargs", "static_out", "pool_bytes", "capture_s", "armed", "replays", "device",
                 "capture_error_mode", "_accounted", "eager_s", "replay_s")

    def __init__(self, key):
        self.key = key
        self.graph = None
        self.pool = None
        self.static_args = None
        self.static_kwargs = None
        self.static_out = None
        self.pool_bytes = 0
        self.capture_s = 0.0
        self.armed = False
        self.replays = 0
        self.device = None
        self.capture_error_mode = None
        self._accounted = False
        self.eager_s = None
        self.replay_s = None


class GraphCache:
    """Capture-or-replay of ONE callable site keyed by argument signature (module contract). Arguments:

    ``name`` the lever id the evidence line carries · ``max_entries`` live graphs · ``on_full`` ``"evict"`` (LRU) | ``"reset"`` | ``"eager"``
    (default: evict for private pools, reset for a generation pool) · ``pool`` ``"private"`` | ``"generation"`` · ``mem_budget_bytes`` an int,
    a callable evaluated at the first capture (e.g. a fraction of the memory free after model load), or None · ``estimate_bytes(key, args,
    kwargs)`` the adapter's pool-size predictor for admission (default: the largest pool measured so far) · ``warmup`` eager calls on the
    side stream before capture · ``policy`` a :class:`opt_core.capture.pool.CapturePolicy` (default: fixed K = ``min_sightings``) · ``policy_key(args,
    kwargs)`` the key the policy sees (default: the argument signature) · ``key_fn(args, kwargs)`` replaces the argument signature in the key (VALUE-keyed
    captures: a temperature, a motif mask's bytes) · ``watch()`` returns the NON-argument tensors the callable reads (attributes, closure cells, per-item
    conditioning): their addresses/shapes are part of the key, so a rebinding recaptures by name instead of replaying a stale pointer · ``mutates_copied_inputs``
    True DECLARES that the callee writes into its (copied-in) input tensors and the caller consumes the RETURNED tensors (``z = stack(z)``): the site is
    exempt from the mutates_inputs refusal, warm-up / capture / checking start from the pre-call input values, checking compares the returned
    outputs; printed as ``inputs=copied-mutable`` — the contract is
    'every per-item tensor the callable reads is an ARGUMENT (copied in), a ``Held`` argument (read by reference) or WATCHED' · ``verify_every`` N: every N-th replay of an entry is recomputed eager and held bit-exact (the eager
    result is returned on those calls) · ``generators`` extra ``torch.Generator`` objects registered with every graph (``rng="graph"`` only; the default
    CUDA generator is always registered by torch) · ``on_capture(key)`` / ``on_replay(key)`` adapter callbacks (refresh a hoist cache an entry reads) · ``verify`` 0 | 1 | 2 · ``rng`` ``"forbid"`` | ``"graph"`` ·
    ``strict`` · ``gate(*args, **kwargs) -> Optional[str]`` the adapter's envelope (a reason string runs the call eager, by name) ·
    ``extra_key()`` folded into every key (kernel-backend flags, id of the model) · ``mode_fn()`` the numerics-mode signature (a change resets)
    · ``module`` whose storage signature is held across :meth:`new_item` · ``reset_on_item`` · ``clone_outputs`` · ``capture_error_mode``
    torch's (``"thread_local"``: other threads may touch CUDA during capture) · ``log`` a callable taking one line (default: stderr) ·
    ``verbose`` also logs every capture (refusals, failures, disables and the first eager event per key and kind are always logged) · ``ledger`` a
    :class:`GraphByteBudget` shared with other caches (one memory budget for several sites) instead of ``mem_budget_bytes``."""

    def __init__(self, name: str, *, max_entries: int = 8, on_full: Optional[str] = None, pool: str = "private",
                 mem_budget_bytes=None, estimate_bytes: Optional[Callable] = None, warmup: int = 2, min_sightings: int = 1,
                 policy=None, policy_key: Optional[Callable] = None, key_fn: Optional[Callable] = None, verify: int = 1,
                 verify_every: int = 0, generators: Optional[list] = None, on_capture: Optional[Callable] = None, on_replay: Optional[Callable] = None,
                 on_fail: str = "disable", watch: Optional[Callable[[], Any]] = None, mutates_copied_inputs: bool = False, rng: str = "forbid", strict: bool = True, gate: Optional[Callable] = None,
                 extra_key: Optional[Callable[[], Any]] = None, mode_fn: Optional[Callable[[], Any]] = None, module=None,
                 reset_on_item: bool = False, clone_outputs: bool = True, capture_error_mode: str = "thread_local",
                 log: Optional[Callable[[str], None]] = None, verbose: bool = False, ledger: Optional["GraphByteBudget"] = None):
        if pool not in ("private", "generation"):
            raise ValueError(f"pool must be 'private' or 'generation', not {pool!r}")
        on_full = on_full or ("evict" if pool == "private" else "reset")
        if on_full not in ("evict", "reset", "eager"):
            raise ValueError(f"on_full must be 'evict', 'reset' or 'eager', not {on_full!r}")
        if on_full == "evict" and pool == "generation":
            raise ValueError("on_full='evict' with pool='generation' would destroy one graph under live neighbours sharing its pool; use 'reset' or 'eager'")
        if rng not in ("forbid", "graph"):
            raise ValueError(f"rng must be 'forbid' or 'graph', not {rng!r}")
        if int(verify) not in (0, 1, 2):
            raise ValueError("verify must be 0, 1 or 2")
        self.name = str(name)
        self.max_entries = max(1, int(max_entries))
        self.on_full = on_full
        self.pool_kind = pool
        if ledger is not None and mem_budget_bytes is not None:
            raise ValueError("give mem_budget_bytes OR a shared ledger, not both")
        self.ledger = ledger if ledger is not None else GraphByteBudget(mem_budget_bytes)
        self._budget_announced = False
        self.estimate_bytes = estimate_bytes
        self.warmup = max(1, int(warmup))
        self.min_sightings = max(1, int(min_sightings))
        if policy is None:
            from .pool import FixedK  # noqa: PLC0415 — pure-python sibling
            policy = FixedK(self.min_sightings)
        self.policy = policy
        self.policy_key = policy_key
        self.key_fn = key_fn
        self.verify_every = max(0, int(verify_every))
        self.generators = list(generators or [])
        if self.generators and rng != "graph":
            raise ValueError("generators= registers RNG state with the graph: it needs rng='graph'")
        self.on_capture = on_capture
        self.on_replay = on_replay
        if on_fail not in ("disable", "key"):
            raise ValueError("on_fail is 'disable' (the site runs eager from then on) or 'key' (only the failed key stays eager, counted eager_failed)")
        self.on_fail = on_fail
        self.watch = watch
        self.mutates_copied_inputs = bool(mutates_copied_inputs)
        self.verify = int(verify)
        self.rng = rng
        self.strict = bool(strict)
        self.gate = gate
        self._gate_reasons: Dict[str, int] = {}
        self.extra_key = extra_key
        self.mode_fn = mode_fn
        self.module = module
        self.reset_on_item = bool(reset_on_item)
        self.clone_outputs = bool(clone_outputs)
        self.capture_error_mode = capture_error_mode
        self._log = log
        self.verbose = bool(verbose)
        self._lock = threading.RLock()
        self.handle_stats = {"handles": 0, "captured": 0, "eager": 0, "replays": 0, "verified": 0, "stale": 0}
        self._entries: "OrderedDict[Any, _Entry]" = OrderedDict()
        self._sightings: Dict[Any, int] = {}
        self._eager_keys: Dict[Any, Tuple[str, str]] = {}      # keys that run eager for the rest of this generation: (kind, reason)
        self._said: set = set()
        self._pool = None
        self._pool_bytes_total = 0
        self._mode = None
        self._mode_seen = False
        self._storage_sig = None
        self._disabled: Optional[str] = None
        self._checked = False
        self.stats_: Dict[str, Any] = {"captures": 0, "replays": 0, "capture_s": 0.0, "verify_s": 0.0, "failures": 0, "evictions": 0, "items": 0,
                                       "verify_pass": 0, "verified_replays": 0, "drops": 0, "pool_bytes": 0}
        for k in EAGER_KINDS:
            self.stats_["eager_" + k] = 0
        for k in REFUSED_KINDS:
            self.stats_["refused_" + k] = 0
        self.resets: Dict[str, int] = {}

    # ------------------------------------------------------------------------------------------------------------- words out
    def _say(self, line: str, once_key=None) -> None:
        if once_key is not None:
            if once_key in self._said:
                return
            self._said.add(once_key)
        text = f"[opt_core.graphs:{self.name}] {line}"
        if self._log is not None:
            self._log(text)
        else:
            print(text, file=sys.stderr, flush=True)

    @staticmethod
    def _short(key) -> str:
        s = repr(key)
        return s if len(s) <= 160 else s[:157] + "..."

    # ------------------------------------------------------------------------------------------------------------- state changes
    def disable(self, reason: str) -> None:
        """Every later call runs eager, counted ``eager_disabled``; the evidence line carries the reason (a named degraded event)."""
        with self._lock:
            if self._disabled is None:
                self._disabled = str(reason)
                self._say(f"DISABLED: {reason} -> every later call of this site runs eager (counted eager_disabled)")
                try:
                    self.reset("disabled")
                except CaptureUnavailable:
                    pass

    @property
    def disabled(self) -> Optional[str]:
        return self._disabled

    def reset(self, reason: str = "reset") -> None:
        """Drop the whole generation: synchronize, graphs + static buffers together, gc, fresh pool, empty_cache (module contract)."""
        with self._lock:
            self.resets[reason] = self.resets.get(reason, 0) + 1
            self._eager_keys.clear()
            self._sightings.clear()
            if not self._entries and self._pool is None:
                return
            torch = _torch()
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            for e in list(self._entries.values()):
                e.graph = None
                e.static_args = e.static_kwargs = e.static_out = None
                e.pool = None
            self._entries.clear()
            self._pool = None
            self.ledger.sub(self._pool_bytes_total)
            self._pool_bytes_total = 0
            gc.collect()
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def new_item(self) -> None:
        """The adapter's item boundary (a new query / seed set / request): counts the item; ``reset_on_item`` resets; with ``module`` given,
        storage that moved since the last capture resets by name."""
        with self._lock:
            self.stats_["items"] += 1
            if self.reset_on_item and self._entries:
                self.reset("item")
                return
            if self.module is not None and self._entries and self._storage_sig is not None:
                if storage_signature(self.module) != self._storage_sig:
                    self._say("module parameter/buffer storage moved since capture -> whole generation dropped, next call recaptures (never replay moved storage)")
                    self.reset("storage_moved")

    def _release(self, ent: _Entry) -> None:
        torch = _torch()
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        if getattr(ent, '_accounted', False):
            self._pool_bytes_total -= int(ent.pool_bytes or 0)
            self.ledger.sub(ent.pool_bytes)
            ent._accounted = False
        ent.graph = None
        ent.static_args = ent.static_kwargs = ent.static_out = None
        ent.pool = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _evict_one(self) -> None:
        _, ent = self._entries.popitem(last=False)
        self._release(ent)
        self.stats_["evictions"] += 1

    def _drop_entry(self, ent: _Entry) -> None:
        if ent.key in self._entries:
            del self._entries[ent.key]
        if self.pool_kind == "generation" and self._entries:
            self.reset("dropped_member")                            # never free one graph of a shared pool under live neighbours
            return
        self._release(ent)

    # ------------------------------------------------------------------------------------------------------------- the call
    def _eager(self, kind: str, fn, args, kwargs, key=None, reason: str = ""):
        self.stats_["eager_" + kind] += 1
        if kind not in ("sighting", "grad") or self.verbose:
            self._say(f"eager ({kind}){': ' + reason if reason else ''}", once_key=("eager", kind, key if key is None or isinstance(key, str) else self._short(key), reason))
        return fn(*args, **kwargs)

    def _check_available(self) -> Optional[str]:
        if self._checked:
            return None
        ok, why = availability()
        if not ok:
            if self.strict:
                raise CaptureUnavailable(why)
            self.disable(str(why))
            return why
        self._checked = True
        return None

    def _budget_bytes(self) -> Optional[int]:
        b = self.ledger.budget()
        if b is not None and not self._budget_announced:
            self._budget_announced = True
            self._say(f"memory budget for graph pools: {b / 2 ** 30:.2f} GiB (new keys are captured while the measured pools fit; beyond it they run eager; nothing captured is freed)")
        return b

    def _admit(self, key, args, kwargs) -> Optional[str]:
        """None to capture, or the named reason this key runs eager (full cache under on_full='eager', memory budget)."""
        if len(self._entries) >= self.max_entries:
            if self.on_full == "eager":
                return f"full: {len(self._entries)} live graphs (max_entries={self.max_entries}); this key runs eager"
            if self.on_full == "reset":
                self._say(f"cache full ({self.max_entries}) on a new key -> whole generation dropped before capture", once_key=("fullreset",))
                self.reset("full")
            else:
                while len(self._entries) >= self.max_entries:
                    self._evict_one()
        budget = self._budget_bytes()
        if budget is not None:
            torch = _torch()
            if self.estimate_bytes is not None:
                est = int(self.estimate_bytes(key, args, kwargs))
            else:
                est = max([int(e.pool_bytes or 0) for e in self._entries.values()] + [0])
            free_now = torch.cuda.mem_get_info()[0] + (torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
            if self.ledger.bytes_total + est > budget:
                return f"budget: pools {self.ledger.bytes_total / 2 ** 30:.2f} GiB + est {est / 2 ** 30:.2f} GiB > budget {budget / 2 ** 30:.2f} GiB"
            if est > free_now:
                return f"budget: est {est / 2 ** 30:.2f} GiB > free {free_now / 2 ** 30:.2f} GiB"
        return None

    def _key_for(self, args, kwargs):
        """The cache key of a call: (extra_key, the argument signature or key_fn's, the watched non-argument tensors' signature)."""
        body = self.key_fn(args, kwargs) if self.key_fn is not None else (tensor_signature(args), tensor_signature(kwargs))
        wsig = tuple(tensor_signature(Held(t)) for t in tree_tensors(self.watch())) if self.watch is not None else None
        return (self.extra_key() if self.extra_key is not None else None, body, wsig)   # watched NON-argument tensors: their addresses are part of the key

    def run(self, fn: Callable, *args, **kwargs):
        """Capture-or-replay ``fn(*args, **kwargs)`` (module contract). ``fn`` must be a pure function of its tensor arguments and of state
        that does not change between capture and replay (module weights: give ``module=``; numerics switches: give ``mode_fn``)."""
        return self._run(fn, args, kwargs, None)

    def _run(self, fn: Callable, args, kwargs, keybox: Optional[list]):
        """:meth:`run`'s body; ``keybox`` (a list, or None) receives the cache key this call was decided under, for :meth:`handle`."""
        with self._lock:
            if self._disabled is not None:
                return self._eager("disabled", fn, _unwrap(args), _unwrap(kwargs), reason=self._disabled)
            why = self._check_available()
            if why is not None:
                return self._eager("disabled", fn, _unwrap(args), _unwrap(kwargs), reason=str(why))
            torch = _torch()
            tensors = tree_tensors(args) + tree_tensors(kwargs)
            uargs, ukwargs = _unwrap(args), _unwrap(kwargs)
            if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
                return self._eager("grad", fn, uargs, ukwargs, reason="autograd is live on an input (inference callables only)")
            devs = {str(t.device) for t in tensors}
            if not tensors or len(devs) != 1 or not next(iter(devs)).startswith("cuda"):
                return self._eager("device", fn, uargs, ukwargs, reason=f"tensor arguments on {sorted(devs) or 'no tensors'} (one CUDA device required)")
            if self.gate is not None:
                g = _gate_reason(self.gate(*uargs, **ukwargs))
                if g is not None and g not in self._gate_reasons:
                    self._gate_reasons[g] = 0
                if g is not None:
                    self._gate_reasons[g] += 1
                if g:
                    return self._eager("gate", fn, uargs, ukwargs, key=str(g), reason=str(g))
            if self.mode_fn is not None:
                mode = self.mode_fn()
                if self._mode_seen and mode != self._mode and self._entries:
                    self._say(f"numerics mode changed since the last capture ({self._mode} -> {mode}) -> whole generation dropped", once_key=("mode", repr(mode)))
                    self.reset("numerics_mode")
                self._mode, self._mode_seen = mode, True
            key = self._key_for(args, kwargs)
            if keybox is not None:
                keybox.append(key)                                  # handle(): the key THIS call was decided under (no re-derivation after fn ran)
            ent = self._entries.get(key)
            if ent is not None:
                self._entries.move_to_end(key)
                if not ent.armed:                                   # verify=2: hold the replay to eager on these NEW inputs before arming
                    return self._verify_second(ent, fn, args, kwargs)
                return self._replay(ent, args, kwargs, fn)
            named = self._eager_keys.get(key)
            if named is not None:
                self.stats_["eager_" + named[0]] += 1
                return fn(*uargs, **ukwargs)
            n = self._sightings.get(key, 0) + 1
            self._sightings[key] = n
            pkey = self.policy_key(args, kwargs) if self.policy_key is not None else key
            action, reason, final = self.policy.decide(pkey, n)
            if action != "capture":
                if final:
                    self._eager_keys[key] = ("policy", reason)
                    return self._eager("policy", fn, uargs, ukwargs, key=key, reason=reason)
                return self._eager("sighting", fn, uargs, ukwargs, key=key, reason=reason)
            refusal = self._admit(key, args, kwargs)
            if refusal is not None:
                kind = refusal.split(":", 1)[0]
                self._eager_keys[key] = (kind, refusal)
                return self._eager(kind, fn, uargs, ukwargs, key=key, reason=refusal)
            return self._capture_and_answer(key, fn, args, kwargs, pkey)

    __call__ = run

    def handle(self, fn: Callable, *args, **kwargs) -> "ReplayHandle":
        """Capture-or-replay ``fn(*args, **kwargs)`` through :meth:`run` (every rule of the normal path: policy, admission, warm-up, checking,
        RNG discipline, budget, evidence) and hand back a :class:`ReplayHandle` on the entry that call left ARMED — the hot-loop form for callers
        that replay one captured step thousands of times per item (a sampler sweep, a per-block loop) and cannot afford the per-call bookkeeping of
        :meth:`run`. When the call was answered eagerly (a policy sighting, a refusal, a disabled cache) the handle is not captured: ``h.captured`` is
        False, ``h.reason`` names why, ``h.out`` is the eager value and ``h.replay()`` raises :class:`HandleUnavailable` — the caller keeps calling
        ``handle()`` (a K-th-sighting policy captures later) or runs eager BY NAME. Counted in :attr:`handle_stats`."""
        keybox: list = []
        out = self._run(fn, args, kwargs, keybox)
        with self._lock:
            self.handle_stats["handles"] += 1
            ent, reason = None, None
            if self._disabled is not None:
                reason = "disabled:" + str(self._disabled)
            elif not keybox:
                reason = "eager:answered before a key was formed (availability, device, grad or gate refusal)"
            else:
                key = keybox[0]                                     # the key run() decided under — recorded there, not re-derived after fn ran
                ent = self._entries.get(key)
                if ent is None:
                    named = self._eager_keys.get(key)
                    reason = (f"{named[0]}:{named[1]}" if named else f"sighting:{self._sightings.get(key, 0)}")
                elif not ent.armed or ent.graph is None:
                    reason, ent = "unarmed:verification pending on the next call", None
            self.handle_stats["captured" if ent is not None else "eager"] += 1
            return ReplayHandle(self, ent, out, reason)

    def handle_fields(self) -> List[Tuple[str, Any]]:
        """``handles=… handle_captured=… handle_eager=… handle_replays=… handle_verified=… handle_stale=…`` for a kit's LEVER line."""
        h = self.handle_stats
        return [("handles", h["handles"]), ("handle_captured", h["captured"]), ("handle_eager", h["eager"]), ("handle_replays", h["replays"]),
                ("handle_verified", h["verified"]), ("handle_stale", h["stale"])]

    # ------------------------------------------------------------------------------------------------------------- replay
    def _replay(self, ent: _Entry, args, kwargs, fn=None):
        if fn is not None and self.verify_every and (ent.replays + 1) % self.verify_every == 0:
            return self._verify_periodic(ent, fn, args, kwargs)
        if self.on_replay is not None:
            self.on_replay(ent.key)
        tree_copy_into(ent.static_args, args)
        tree_copy_into(ent.static_kwargs, kwargs)
        ent.graph.replay()
        ent.replays += 1
        self.stats_["replays"] += 1
        return tree_clone(ent.static_out) if self.clone_outputs else ent.static_out

    def _verify_periodic(self, ent: _Entry, fn, args, kwargs):
        """Every ``verify_every``-th replay: eager recomputation, RNG-neutral replay, bit-exact comparison; the EAGER result is returned."""
        torch = _torch()
        t0 = time.perf_counter()
        rng_ref = self._rng_snapshot(torch, ent.device)
        held = _held_tensors(args, kwargs)
        held_before = [t.clone() for t in held]
        tree_copy_into(ent.static_args, args)
        tree_copy_into(ent.static_kwargs, kwargs)
        ref = fn(*_unwrap(args), **_unwrap(kwargs))
        torch.cuda.synchronize(ent.device)
        rng_after = self._rng_snapshot(torch, ent.device)
        if held:
            ref = tree_clone(ref)
            held_after = [t.clone() for t in held]
            _restore(held, held_before)
        if self.on_replay is not None:
            self.on_replay(ent.key)
        self._verification_replay(ent, torch, ent.device, rng_ref, rng_after)
        ent.replays += 1
        self.stats_["replays"] += 1
        self.stats_["verified_replays"] += 1
        ok = bitwise_equal(ent.static_out, ref) and (not held or bitwise_equal(held, held_after))
        self.stats_["verify_s"] += time.perf_counter() - t0
        if ok:
            return ref
        detail = {"max_abs_diff": max_abs_diff(ent.static_out, ref), "stage": "verify_every", "replay": ent.replays}
        self._drop_entry(ent)
        return self._refuse("verify", f"periodic verification: replay {ent.replays} differs from eager ({detail})", detail, answer=lambda: ref, key=ent.key)

    def _rng_snapshot(self, torch, dev):
        """The generator position(s) an eager call is about to draw from (``None`` under ``rng="forbid"``)."""
        if self.rng != "graph":
            return None
        return (torch.cuda.get_rng_state(dev), [g.get_state() for g in self.generators])

    def _rng_restore(self, snap, torch, dev) -> None:
        if snap is None:
            return
        torch.cuda.set_rng_state(snap[0], dev)
        for g, st in zip(self.generators, snap[1]):
            g.set_state(st)

    def _verification_replay(self, ent: _Entry, torch, dev, rng_ref, rng_after) -> None:
        """One replay FOR COMPARISON with an eager reference: under ``rng="graph"`` the generator is first put where the reference call STARTED
        (``rng_ref``: a replay fills its captured offset holder from the generator's current position, so it must draw the offsets the
        reference drew), and afterwards where the reference call ENDED (``rng_after``) — the stream position is exactly one eager call's worth,
        the caller having received that call's result."""
        self._rng_restore(rng_ref, torch, dev)
        try:
            ent.graph.replay()
            torch.cuda.synchronize(dev)
        finally:
            self._rng_restore(rng_after, torch, dev)

    def _verify_second(self, ent: _Entry, fn, args, kwargs):
        torch = _torch()
        dev = ent.device
        t0 = time.perf_counter()
        rng_ref = self._rng_snapshot(torch, dev)
        held = _held_tensors(args, kwargs)
        held_before = [t.clone() for t in held]
        tree_copy_into(ent.static_args, args)                       # copy-in BEFORE the eager call: a callable that mutates its inputs must not
        tree_copy_into(ent.static_kwargs, kwargs)                   #   poison the replay's inputs (mutation itself is refused at verify1)
        ref = fn(*_unwrap(args), **_unwrap(kwargs))               # this call's answer, eager (the stock function by definition)
        torch.cuda.synchronize(dev)
        rng_after = self._rng_snapshot(torch, dev)
        if held:
            ref = tree_clone(ref)
            held_after = [t.clone() for t in held]
            _restore(held, held_before)
        if self.on_replay is not None:
            self.on_replay(ent.key)
        self._verification_replay(ent, torch, dev, rng_ref, rng_after)
        ok = bitwise_equal(ent.static_out, ref) and (not held or bitwise_equal(held, held_after))
        self.stats_["verify_s"] += time.perf_counter() - t0
        if ok:
            ent.armed = True
            self.stats_["verify_pass"] += 1
            self.stats_["eager_unarmed"] += 1
            if self.verbose:
                self._say(f"armed after second verification key={self._short(ent.key)}")
            return ref
        detail = {"max_abs_diff": max_abs_diff(ent.static_out, ref), "stage": "verify2"}
        self._drop_entry(ent)
        return self._refuse("verify", f"replay of a NEW input differs from eager ({detail}) — the callable reads state outside its arguments", detail,
                            answer=lambda: ref, key=ent.key)

    def _refuse(self, kind: str, message: str, detail: Optional[dict], answer: Callable, key):
        self.stats_["refused_" + kind] += 1
        if self.strict:
            raise CaptureRefused(kind, f"{self.name}: {message}", detail)
        self.disable(f"refused ({kind}): {message}")
        return answer()

    # ------------------------------------------------------------------------------------------------------------- capture
    def _capture_and_answer(self, key, fn, args, kwargs, pkey=None):
        torch = _torch()
        tensors0 = tree_tensors(args) + tree_tensors(kwargs)
        if self.rng == "forbid" and self.module is not None:
            live = modules_in_train_mode(self.module)
            if live:
                return self._refuse("rng", f"module has sub-modules in train mode (dropout live): {live[:5]} — eval() them or declare rng='graph'", {"stage": "precheck"},
                                    answer=lambda: fn(*_unwrap(args), **_unwrap(kwargs)), key=key)
        dev0 = tensors0[0].device
        held = _held_tensors(args, kwargs)
        ref_box: Dict[str, Any] = {"rng_ref": self._rng_snapshot(torch, dev0), "held": held,
                                   "held_before": [t.clone() for t in held]}   # Held buffers are read/WRITE by reference: their pre-call contents, so warm-up,
        if self.verify >= 1 or self.mutates_copied_inputs:          #   capture and the checking replay each start from what THIS call saw
            ref_box["inputs"] = (tree_clone(_strip_held(args)), tree_clone(_strip_held(kwargs)))   # the copied-in inputs as given (Held buffers excluded)
        v = fn(*_unwrap(args), **_unwrap(kwargs))                   # EAGER FIRST: the capturing call returns the plain eager result, computed before any
        torch.cuda.synchronize(dev0)                                #   warm-up/capture (a first replay handed back is not this call's stock answer)
        ref_box["v"] = tree_clone(v) if held else v                 #   (copied when Held buffers exist: the outputs may alias a buffer the warm-up rewrites)
        ref_box["held_after"] = [t.clone() for t in held]           # the buffers' contents after exactly ONE eager call: what the caller must observe at return
        ref_box["rng_after"] = self._rng_snapshot(torch, dev0)
        if "inputs" in ref_box and not self.mutates_copied_inputs and not (bitwise_equal(ref_box["inputs"][0], _strip_held(args)) and bitwise_equal(ref_box["inputs"][1], _strip_held(kwargs))):
            ref_box.pop("inputs", None)
            return self._refuse("mutates_inputs", "the callable mutates a COPIED-IN input tensor in place (an eager call changes the caller's tensor, a replay would change "
                                "only the static copy) — declare mutates_copied_inputs=True if the caller consumes the RETURNED tensors, pass an adapter-owned "
                                "persistent buffer as graphs.Held(buffer), or make the callable functional", {"stage": "precheck"},
                                answer=lambda: ref_box["v"], key=key)
        _hoist.set_capturing(True)                                  # const memos compute inline inside warm-up/capture (never store, never serve)
        try:
            return self._capture_body(torch, key, fn, args, kwargs, pkey, ref_box)
        finally:
            _hoist.set_capturing(False)
            if held:                                                # whatever happened (captured, refused, failed): the caller's Held buffers hold the state
                try:                                                #   after exactly one eager call, as if the callable had run once
                    _restore(held, ref_box["held_after"])
                    torch.cuda.synchronize(dev0)
                except Exception:  # noqa: BLE001
                    pass

    def _capture_body(self, torch, key, fn, args, kwargs, pkey, ref_box):
        tensors = tree_tensors(args) + tree_tensors(kwargs)
        dev = tensors[0].device
        t0 = time.perf_counter()

        def answer():
            return ref_box["v"]
        ent = _Entry(key)
        ent.device = dev
        ent.static_args = tree_clone(args)
        ent.static_kwargs = tree_clone(kwargs)
        src_args, src_kwargs = (ref_box["inputs"] if self.mutates_copied_inputs else (args, kwargs))   # copy-in source: the PRE-call input values when the
        tree_copy_into(ent.static_args, src_args)                   #   callee mutates its copies (the eager-first call already advanced the caller's tensors)
        tree_copy_into(ent.static_kwargs, src_kwargs)
        rng_before = torch.cuda.get_rng_state(dev)
        gens_before = [g.get_state() for g in self.generators]
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        try:
            eager_s = None
            with torch.cuda.stream(side):
                for i in range(self.warmup):
                    _restore(ref_box["held"], ref_box["held_before"])   # a Held buffer the callable rewrites starts every warm-up call from its pre-call contents
                    if self.mutates_copied_inputs:
                        tree_copy_into(ent.static_args, src_args)   # and so do static copies the callee mutates
                        tree_copy_into(ent.static_kwargs, src_kwargs)
                    if i == self.warmup - 1:
                        torch.cuda.synchronize(dev)
                        tw = time.perf_counter()
                    fn(*ent.static_args, **ent.static_kwargs)
                torch.cuda.synchronize(dev)
                eager_s = time.perf_counter() - tw
            torch.cuda.current_stream(dev).wait_stream(side)
            torch.cuda.synchronize(dev)
        except Exception as e:  # noqa: BLE001 — an eager failure on the side stream: nothing to recover but the generator position
            if is_oom(e):                                            # an out-of-memory is the caller's to see; the eager route needs no less memory
                raise
            try:
                torch.cuda.synchronize(dev)
            except Exception:  # noqa: BLE001
                pass
            torch.cuda.set_rng_state(rng_before, dev)
            return self._fail(f"warm-up raised {e!r}", {"healthy": True, "stage": "warmup"}, answer, key)
        drew = not bool(torch.equal(rng_before, torch.cuda.get_rng_state(dev)))
        if drew or self.rng == "graph":
            torch.cuda.set_rng_state(rng_before, dev)               # warm-up draws are never part of the stock stream
            for g, st in zip(self.generators, gens_before):
                g.set_state(st)
        if drew and self.rng == "forbid":
            return self._refuse("rng", "the callable drew from the CUDA generator during warm-up (rng='forbid'): move the draws outside the graphed callable "
                                "or construct the cache with rng='graph'", {"stage": "warmup"}, answer=answer, key=key)
        tree_copy_into(ent.static_args, src_args)                   # warm-up may have written into static buffers the callable mutates in place
        tree_copy_into(ent.static_kwargs, src_kwargs)
        _restore(ref_box["held"], ref_box["held_before"])
        free0 = None
        if self.ledger.measuring or self.verbose:
            try:
                torch.cuda.empty_cache()
                free0 = torch.cuda.mem_get_info()[0]
            except Exception:  # noqa: BLE001
                free0 = None
        if self.pool_kind == "generation":
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            pool = self._pool
        else:
            pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        for g in self.generators:
            reg = getattr(graph, "register_generator_state", None)
            if reg is None:
                return self._refuse("rng", f"torch {torch.__version__} has no CUDAGraph.register_generator_state; generators= cannot be honoured", {"stage": "register"}, answer=answer, key=key)
            reg(g)
        began = False
        ended = False
        t1 = time.perf_counter()
        ctx, ent.capture_error_mode = _graph_context(torch, graph, pool, side, self.capture_error_mode)
        try:
            with ctx:
                began = True
                ent.static_out = fn(*ent.static_args, **ent.static_kwargs)
            ended = True
            torch.cuda.synchronize(dev)
        except Exception as e:  # noqa: BLE001
            rep: Dict[str, Any] = {"stage": "capture"}
            try:                                                     # the dangling capture is ended and its pool released either way (a healthy device for the caller)
                rep.update(recover_after_failed_capture(graph=graph, pool=pool, device=dev.index, capture_began=began, capture_ended=ended, raise_if_unhealthy=False))
            except Exception as e2:  # noqa: BLE001
                rep["recovery_error"] = repr(e2)[:200]
                rep["healthy"] = False
            try:
                torch.cuda.set_rng_state(rng_before, dev)
            except Exception:  # noqa: BLE001
                pass
            del graph
            ent.static_args = ent.static_kwargs = ent.static_out = None
            if is_oom(e):                                            # an out-of-memory during capture is the caller's to see, never the named eager route
                raise e
            return self._fail(f"capture raised {e!r}", rep, answer, key)
        if self.rng == "graph":
            torch.cuda.set_rng_state(rng_before, dev)               # the capture registers offsets; the stock stream position is the pre-capture one
            for g, st in zip(self.generators, gens_before):
                g.set_state(st)
        ent.graph, ent.pool = graph, pool
        ent.capture_s = time.perf_counter() - t1
        if free0 is not None:
            try:
                torch.cuda.empty_cache()
                ent.pool_bytes = max(0, int(free0 - torch.cuda.mem_get_info()[0]))
            except Exception:  # noqa: BLE001
                ent.pool_bytes = 0
        if self.verify >= 1:                                        # checking 1: replay on the capture inputs, bit-exact against this call's eager answer
            tv = time.perf_counter()
            tree_copy_into(ent.static_args, src_args)
            tree_copy_into(ent.static_kwargs, src_kwargs)
            _restore(ref_box["held"], ref_box["held_before"])       # the replay starts from the buffers THIS call saw
            torch.cuda.synchronize(dev)
            tr = time.perf_counter()
            self._verification_replay(ent, torch, dev, ref_box.get("rng_ref"), ref_box.get("rng_after"))
            replay_s = time.perf_counter() - tr
            ok = bitwise_equal(ent.static_out, ref_box["v"])
            held_ok = bitwise_equal(ref_box["held"], ref_box["held_after"])   # Held buffers after the replay == after the eager call (in-place writes checked too)
            self.stats_["verify_s"] += time.perf_counter() - tv
            if not (ok and held_ok):
                detail = {"max_abs_diff": max_abs_diff(ent.static_out, ref_box["v"]), "held_max_abs_diff": (max_abs_diff(ref_box["held"], ref_box["held_after"]) if ref_box["held"] else None), "stage": "verify1"}
                self._release(ent)
                del graph
                return self._refuse("verify", f"replay differs from eager on the capture inputs ({detail}) — a kernel in the callable is not capture-safe "
                                    "or reads memory produced outside the graph", detail, answer=answer, key=key)
            self.stats_["verify_pass"] += 1
        else:
            replay_s = None
        ent.eager_s, ent.replay_s = eager_s, replay_s
        try:
            self.policy.observe(pkey if pkey is not None else key, capture_s=ent.capture_s, eager_s=eager_s, replay_s=replay_s)
        except Exception as e:  # noqa: BLE001 — a policy's bookkeeping never fails a capture
            self._say(f"policy.observe raised {e!r} (ignored)")
        ent.armed = self.verify < 2
        self._entries[key] = ent
        self._pool_bytes_total += int(ent.pool_bytes or 0)
        self.ledger.add(ent.pool_bytes)
        ent._accounted = True
        self.stats_["pool_bytes"] = self._pool_bytes_total
        self.stats_["captures"] += 1
        self.stats_["capture_s"] += time.perf_counter() - t0
        if self.module is not None:
            self._storage_sig = storage_signature(self.module)
        if self.on_capture is not None:
            self.on_capture(key)
        if self.verbose:
            self._say(f"captured key={self._short(key)} warmup={self.warmup} capture_s={ent.capture_s:.3f} pool_mib={ent.pool_bytes / 2 ** 20:.0f} "
                      f"verify={'pass' if self.verify else 'off'} armed={ent.armed} live={len(self._entries)} rng={self.rng} capture_error_mode={ent.capture_error_mode} "
                      f"eager_ms={'%.2f' % (1e3 * eager_s) if eager_s is not None else 'na'} replay_ms={'%.2f' % (1e3 * replay_s) if replay_s is not None else 'na'}")
        return ref_box["v"]

    def _fail(self, message: str, rep: dict, answer: Callable, key):
        self.stats_["failures"] += 1
        healthy = bool(rep.get("healthy", False))
        self._say(f"capture FAILED: {message}; recovery={rep}")
        if self.strict or not healthy:
            raise CaptureFailed(f"{self.name}: {message} (recovery: {rep})", rep)
        if self.on_fail == "key" and key is not None:
            self._eager_keys[key] = ("failed", message)            # this unit stays eager for the generation, counted; the site keeps capturing others
            self._say(f"key {self._short(key)} stays EAGER after a failed capture (counted eager_failed): {message}")
        else:
            self.disable(f"capture failed: {message}")
        return answer()

    def drop(self, key) -> bool:
        """Drop ONE captured entry by key (private pools only: its blocks are its own). Under ``pool="generation"`` a single graph is never
        destroyed while neighbours in its pool live — the call is refused by name (use :meth:`reset`). Returns whether an entry was dropped."""
        with self._lock:
            ent = self._entries.get(key)
            if ent is None:
                return False
            if self.pool_kind == "generation":
                self._say(f"drop({self._short(key)}) refused: generation pool — reset() drops the generation whole")
                return False
            self._drop_entry(ent)
            self.stats_["drops"] += 1
            return True

    # ------------------------------------------------------------------------------------------------------------- census
    def stats(self) -> dict:
        """The census: counters, resets by reason, live keys, pool bytes, the disabled reason (None when live)."""
        with self._lock:
            d = dict(self.stats_)
            d.update(name=self.name, strict=self.strict, pool=self.pool_kind, rng=self.rng, verify=self.verify, keys=len(self._entries),
                     resets=dict(self.resets), disabled=self._disabled, gate_reasons=dict(self._gate_reasons), budget_bytes=self.ledger.budget() if self.ledger.measuring else None, ledger_mib=int(self.ledger.bytes_total / 2 ** 20), state=self.state(),
                     eager=sum(self.stats_["eager_" + k] for k in EAGER_KINDS), refused=sum(self.stats_["refused_" + k] for k in REFUSED_KINDS),
                     per_key=[{"key": self._short(e.key), "replays": e.replays, "capture_s": round(e.capture_s, 4), "pool_bytes": e.pool_bytes, "armed": e.armed, "eager_s": e.eager_s, "replay_s": e.replay_s}
                              for e in self._entries.values()])
            return d

    def state(self) -> str:
        """``disabled:<reason>`` · ``active`` (replays > 0) · ``captured`` (captures, no replay yet) · ``eager`` (calls, no capture) · ``idle``."""
        s = self.stats_
        if self._disabled is not None:
            return "disabled:" + self._disabled.replace(" ", "_")[:80]
        if s["replays"] > 0:
            return "active"
        if s["captures"] > 0:
            return "captured"
        if sum(s["eager_" + k] for k in EAGER_KINDS) > 0:
            return "eager"
        return "idle"

    def evidence_fields(self) -> List[Tuple[str, Any]]:
        """The ordered ``(key, value)`` pairs of the ONE activation-evidence line of this lever (the kit prefixes its own tag)."""
        s = self.stats_
        return [("lever", self.name), ("state", self.state()), ("captures", s["captures"]), ("replays", s["replays"]),
                ("eager", {k: s["eager_" + k] for k in EAGER_KINDS if s["eager_" + k]}), ("refused", {k: s["refused_" + k] for k in REFUSED_KINDS if s["refused_" + k]}),
                ("failures", s["failures"]), ("resets", dict(self.resets) or None), ("keys", len(self._entries)), ("items", s["items"]),
                ("strict", int(self.strict)), ("pool", self.pool_kind), ("rng", self.rng), ("verify", self.verify), ("inputs", "copied-mutable" if self.mutates_copied_inputs else "copied"), ("policy", self.policy.describe().replace(" ", "")),
                ("capture_s", round(s["capture_s"], 3)), ("pool_mib", int(self._pool_bytes_total / 2 ** 20))]

    def evidence_line(self, tag: str, name: Optional[str] = None) -> str:
        """The lever's ONE activation-evidence line in the core's LEVER grammar: ``[<tag>] LEVER name=<name> state=<on|skipped> reason=…
        impl=cuda_graphs graphs=<active|captured|eager|idle> captures=… replays=… eager=… refused=… failures=… …`` (``name`` defaults to the
        site name; ``state=skipped`` when the site disabled itself, with the reason)."""
        fields = self.evidence_fields()
        detail = [(k, v) for k, v in fields if k not in ("lever", "state")]
        if self._disabled is not None:
            head = [("name", name or self.name), ("state", "skipped"), ("reason", self._disabled), ("impl", "cuda_graphs"), ("origin", "core"), ("graphs", "disabled")]
        else:
            head = [("name", name or self.name), ("state", "on"), ("impl", "cuda_graphs"), ("origin", "core"), ("graphs", self.state())]
        return f"{report.prefix(tag)} LEVER {report.kv(*(head + detail))}"

    def partial(self) -> Optional[str]:
        """The reason this lever counts as a PARTIAL activation for the kit's exit verdict (disabled, a refusal or a failure happened), else None.
        Envelope eager calls (gate, budget, sightings) are not partial: they are the lever's declared shape."""
        s = self.stats_
        if self._disabled is not None:
            return f"disabled: {self._disabled}"
        if s["failures"] or any(s["refused_" + k] for k in REFUSED_KINDS):
            return "failures=%d refused=%s" % (s["failures"], {k: s["refused_" + k] for k in REFUSED_KINDS})
        return None


def graphed(cache: GraphCache, fn: Callable) -> Callable:
    """``fn`` wrapped so that calling it goes through ``cache.run`` (for patching a bound method: ``obj.step = graphed(cache, obj.step)``)."""
    def _wrapped(*args, **kwargs):
        return cache.run(fn, *args, **kwargs)
    _wrapped.__wrapped__ = fn                    # type: ignore[attr-defined]
    _wrapped.graph_cache = cache                 # type: ignore[attr-defined]
    return _wrapped


class HandleUnavailable(RuntimeError):
    """:meth:`ReplayHandle.replay` on a handle whose call was answered eagerly (``reason`` names why) — nothing was captured to replay."""


class HandleStale(RuntimeError):
    """:meth:`ReplayHandle.replay` on an entry its cache has since dropped (reset, eviction, storage move, failed checking): re-handle."""


class ReplayHandle:
    """The bare-replay view of one ARMED :class:`GraphCache` entry (:meth:`GraphCache.handle`). ``replay()`` is ``graph.replay()`` plus two counters —
    no argument walk, no signature, no copies, no clone: the caller owns the stream it replays on (``with torch.cuda.stream(s): h.replay()``), writes
    new inputs into :attr:`static_args` / :attr:`static_kwargs` itself or through :meth:`copy_in`, and reads results from :attr:`out` (the static
    output tree, by reference — valid until the next replay). :meth:`verify` is the explicit eager-recompute-and-compare (the cache's periodic
    checking, on demand: at item boundaries or every k-th sweep); a mismatch drops the entry BY NAME through the cache's refusal path and the
    handle goes stale. :attr:`alive` is False once the cache dropped the entry for any reason; ``replay()`` then raises :class:`HandleStale`.
    Generators registered with the cache (``rng='graph'``) advance per replay exactly as under :meth:`GraphCache.run`.

    OBLIGATIONS of the caller (what ``replay()`` does NOT do that :meth:`GraphCache.run` does per call): it does not re-check that module
    storage is unmoved, that the numerics mode is unchanged, or that the inputs still have the captured signature — a caller re-handles
    (``cache.handle(...)`` again) after anything that can move weights or switch modes (a ``.to()``, a mode flip, a new item when the cache
    resets per item) and calls :meth:`verify` at its own cadence (every item boundary at least); the adopting kit prints ``handle_verified`` and
    ``handle_stale`` (:meth:`GraphCache.handle_fields`) on its LEVER line and gates ``handle_stale == 0`` and ``handle_verified >= items``."""

    __slots__ = ("cache", "_ent", "first_out", "reason", "key")

    def __init__(self, cache: GraphCache, ent: Optional[_Entry], first_out, reason: Optional[str]):
        self.cache = cache
        self._ent = ent
        self.first_out = first_out                     # what the handle() call itself returned (a replay result or the eager value)
        self.reason = reason
        self.key = ent.key if ent is not None else None

    @property
    def captured(self) -> bool:
        return self._ent is not None

    @property
    def alive(self) -> bool:
        e = self._ent
        return e is not None and e.graph is not None and self.cache._entries.get(e.key) is e

    @property
    def out(self):
        """The static output tree (by reference) of a captured handle — raises :class:`HandleStale` once the cache dropped the entry (like
        ``replay``); the eager value (``first_out``) of an UNCAPTURED handle. After a failed :meth:`verify` the eager result is in ``first_out``."""
        if self._ent is None:
            return self.first_out
        return self._live_entry().static_out

    @property
    def static_args(self):
        return self._ent.static_args if self._ent is not None else None

    @property
    def static_kwargs(self):
        return self._ent.static_kwargs if self._ent is not None else None

    @property
    def replays(self) -> int:
        return int(self._ent.replays) if self._ent is not None else 0

    def _live_entry(self) -> _Entry:
        if self._ent is None:
            raise HandleUnavailable(f"{self.cache.name}: nothing captured for this call ({self.reason})")
        if not self.alive:
            with self.cache._lock:
                self.cache.handle_stats["stale"] += 1
            raise HandleStale(f"{self.cache.name}: the cache dropped this entry (resets={dict(self.cache.resets)}); call handle() again")
        return self._ent

    def replay(self):
        """``graph.replay()`` on the caller's current stream; returns :attr:`out`. No copies, no checking, no clone."""
        ent = self._live_entry()
        cache = self.cache
        if cache.on_replay is not None:
            cache.on_replay(ent.key)
        ent.graph.replay()
        with cache._lock:                                           # counters only; the replay itself is the caller's stream business
            ent.replays += 1
            cache.stats_["replays"] += 1
            cache.handle_stats["replays"] += 1
        return ent.static_out

    def copy_in(self, *args, **kwargs) -> int:
        """Copy new argument values into the static input buffers (same tree shape as the captured call; ``Held`` leaves are skipped) — for callers
        that do not write :attr:`static_args` in place themselves. Returns the number of tensors copied."""
        ent = self._live_entry()
        return tree_copy_into(ent.static_args, args) + tree_copy_into(ent.static_kwargs, kwargs)

    def verify(self, fn: Callable, *args, **kwargs) -> bool:
        """Eager ``fn`` on these arguments vs one RNG-neutral replay, bit-exact (the cache's periodic checking, on demand). True = identical (the
        replay counted, the entry stays); False = the cache refused and dropped the entry by name (the handle is stale; ``on_fail`` decides whether the
        cache disabled itself) — the eager result of this call is then in :attr:`first_out`."""
        ent = self._live_entry()
        cache = self.cache
        with cache._lock:
            ref = cache._verify_periodic(ent, fn, args, kwargs)
            cache.handle_stats["verified"] += 1
            ok = self.alive
            if not ok:
                self.first_out = ref
            return ok
