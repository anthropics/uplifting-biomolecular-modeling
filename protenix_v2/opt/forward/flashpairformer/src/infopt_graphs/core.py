"""infopt_graphs.core — engine-agnostic CUDA-graph capture with static buffers and exact-shape caching.

Design (see README.md):
  * GraphedFunction(fn, family=..., warmup=1, pool="shared"|"private", clone_outputs=True)
        A callable cache: on each call the input pytree (dicts / lists / tuples / tensors / python scalars) is reduced
        to a static SIGNATURE (per tensor: shape, dtype, device, strides; non-tensor leaves: their value).  The first call
        for a signature WARMS UP fn on a side stream, CAPTURES it into a torch.cuda.CUDAGraph with static input buffers
        (allocated with the same shape/dtype/STRIDES as the example so kernel selection matches eager), and replays it.
        Later calls with the same signature copy the new leaves into the static buffers (a D2D copy, no sync) and replay.
        Tensor leaves registered as `constant` (identity-tracked) are not copied when the same object is passed again.
  * Fallback: if capture fails (an op that synchronizes / allocates outside the caching allocator / uses the host), the
        signature is marked `unsupported`, the error (with the Python site from the sync census) is logged, and the call
        runs eagerly — the integration never changes results silently.
  * RNG: the captured region must not consume random numbers (audit.RNGGuard is applied during warm-up); random tensors
        are inputs, drawn OUTSIDE in the engine's stock order.
  * Memory: graphs of one `family` may share one memory pool (torch.cuda.graph_pool_handle()).  Rule that makes sharing
        safe: the outputs of a replay are consumed (cloned, default) before any other graph of the family is replayed, and
        graphs are never replayed concurrently.  `graph_memory()` reports the reserved bytes per entry (measured at capture
        as the private-pool growth).
  * Eviction: LRU by `max_entries`; entries release their graph and static buffers.

Functional helpers: `signature_of(tree)`, `static_like(tree)`, `copy_into(static_tree, tree)`.
"""
from __future__ import annotations

from opt_core.oom import is_oom   # a broad handler that reroutes around a lever re-raises device out-of-memory first (opt_core.oom.is_oom)
import collections
import contextlib
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from opt_core.tools.graph_audit.audit import RNGGuard, SyncCensus

log = logging.getLogger("infopt_graphs")


# ----------------------------------------------------------------------------------------------------------------------
# pytree <-> signature / static buffers
# ----------------------------------------------------------------------------------------------------------------------
def _leaf_sig(x: Any) -> Hashable:
    if isinstance(x, torch.Tensor):
        return ("T", tuple(x.shape), str(x.dtype), x.device.type, tuple(x.stride()), bool(x.requires_grad))
    if isinstance(x, (bool, int, float, str, type(None))):
        return ("S", type(x).__name__, x)
    if isinstance(x, torch.dtype) or isinstance(x, torch.device):
        return ("S", str(x))
    try:
        hash(x)
        return ("H", repr(x))
    except TypeError:
        return ("R", repr(x))


def signature_of(tree: Any) -> Tuple[Hashable, ...]:
    leaves, spec = tree_flatten(tree)
    return (repr(spec),) + tuple(_leaf_sig(l) for l in leaves)


def _is_nod(t: torch.Tensor) -> bool:
    # exact check: sort strides descending with sizes and check tiling
    sz, st = list(t.shape), list(t.stride())
    order = sorted(range(len(sz)), key=lambda i: (st[i], sz[i]))
    expect = 1
    for i in order:
        if sz[i] == 1:
            continue
        if st[i] != expect:
            return False
        expect *= sz[i]
    return True


def static_like(tree: Any, constants: Optional[set] = None) -> Any:
    """Allocate static buffers with the example's shape/dtype/device/strides for every tensor leaf (constants by id are
    kept by reference).  Non-tensor leaves are kept as-is."""
    leaves, spec = tree_flatten(tree)
    out = []
    for l in leaves:
        if isinstance(l, torch.Tensor):
            if constants and id(l) in constants:
                out.append(l)
            elif l.is_contiguous() or not _is_nod(l):
                out.append(torch.empty(l.shape, dtype=l.dtype, device=l.device))
            else:
                out.append(torch.empty_strided(l.shape, l.stride(), dtype=l.dtype, device=l.device))
        else:
            out.append(l)
    return tree_unflatten(out, spec)


def copy_into(static_tree: Any, tree: Any, skip_same_object: bool = True) -> int:
    """Copy every tensor leaf of `tree` into the corresponding static leaf (D2D, no host sync). Returns #copies."""
    sl, _ = tree_flatten(static_tree)
    tl, _ = tree_flatten(tree)
    assert len(sl) == len(tl), "pytree mismatch"
    n = 0
    for s, t in zip(sl, tl):
        if isinstance(t, torch.Tensor):
            if skip_same_object and s is t:
                continue
            s.copy_(t, non_blocking=True)
            n += 1
    return n


# ----------------------------------------------------------------------------------------------------------------------
# captured entry
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class GraphEntry:
    signature: Tuple
    graph: Optional[torch.cuda.CUDAGraph]
    static_args: Any
    static_kwargs: Any
    static_out: Any
    capture_s: float
    warmup_s: float
    pool_growth_bytes: int
    n_replays: int = 0
    last_used: float = field(default_factory=time.time)
    unsupported: Optional[str] = None


class PoolRegistry:
    """Shared graph memory pools per family.  A torch.cuda.graph_pool_handle() may only be reused for a new capture while at
    least one graph captured into it is still alive (CUDACachingAllocator::beginAllocateToPool asserts use_count > 0 for a
    known mempool id; violating it leaves the CUDA RNG generator in 'capturing' state -> every later torch.randn on CUDA fails
    with 'Offset increment outside graph capture').  acquire() returns the family handle (fresh if no graph is alive);
    release() must be called when a graph of the family is dropped."""

    def __init__(self):
        self.handles: Dict[str, Any] = {}
        self.live: Dict[str, int] = {}

    def acquire(self, family: str):
        if self.live.get(family, 0) <= 0 or family not in self.handles:
            self.handles[family] = torch.cuda.graph_pool_handle()
            self.live[family] = 0
        self.live[family] += 1
        return self.handles[family]

    def release(self, family: str):
        if family in self.live:
            self.live[family] = max(0, self.live[family] - 1)
            if self.live[family] == 0:
                self.handles.pop(family, None)

    def summary(self):
        return {f: n for f, n in self.live.items()}


POOLS = PoolRegistry()


def _recover_generator_after_failed_capture():
    """If capture_begin raised AFTER the default CUDA generator's capture_prologue (e.g. the allocator pool assert), the
    generator stays flagged 'capturing' and all later RNG calls fail.  A successful trivial capture runs capture_epilogue
    and clears the flag.  Numerics-free (the dummy graph is discarded; the generator offset is not advanced)."""
    try:
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
    except Exception:  # noqa
        return False


@contextlib.contextmanager
def capture_context(graph: torch.cuda.CUDAGraph, pool=None, stream: Optional[torch.cuda.Stream] = None, gc_collect: Optional[bool] = None):
    """Like torch.cuda.graph(g, pool, stream) but WITHOUT the gc.collect() + torch.cuda.empty_cache() that torch.cuda.graph
    runs before every capture (on a large heap those two calls dominate the capture time).  Controlled by INFOPT_GRAPHS_GC (default 0 = skip; 1 = torch's default behaviour).  The capture itself is
    identical: capture_begin/capture_end on a side stream that waits on the current stream."""
    if gc_collect is None:
        import os
        gc_collect = os.environ.get("INFOPT_GRAPHS_GC", "0") not in ("0", "false", "off")
    if gc_collect:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    s = stream or torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        torch.cuda.synchronize()
        try:
            if pool is None:
                graph.capture_begin()
            else:
                graph.capture_begin(pool=pool)
        except Exception:
            _recover_generator_after_failed_capture()
            raise
        ok = False
        try:
            yield graph
            ok = True
        finally:
            try:
                graph.capture_end()
            except Exception:
                _recover_generator_after_failed_capture()
                if ok:
                    raise
    torch.cuda.current_stream().wait_stream(s)


def _autocast_nocache():
    """Re-enter the caller's CUDA autocast state with the weight-cast cache DISABLED.
    Rationale: autocast caches fp32->bf16 weight casts for the lifetime of the outermost autocast context.  A graph captured
    while such a cache is populated would read cached casts that are freed when the context exits -> a later replay reads
    freed memory.  With cache_enabled=False every cast is (re)captured as a kernel inside the graph (cheap, weights are
    small) and the graph is self-contained."""
    try:
        enabled = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda")
    except TypeError:  # older torch
        enabled = torch.is_autocast_enabled()
        dtype = torch.get_autocast_gpu_dtype()
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True, cache_enabled=False)


class GraphedFunction:
    """Cache of captured CUDA graphs for `fn`, keyed by the static signature of the call arguments.

    First call for a signature: warm-up run on a side stream (its OUTPUTS are returned to the caller — nothing is wasted),
    then capture; later calls: copy inputs into the static buffers, replay, return (clones of) the static outputs."""


    def __init__(self, fn: Callable, name: str = "fn", family: str = "default", warmup: int = 2, pool: str = "shared",
                 clone_outputs: bool = True, max_entries: int = 16, constants: Optional[List[torch.Tensor]] = None,
                 enabled: bool = True, rng_guard: bool = True, sync_audit: bool = True, on_capture: Optional[Callable] = None,
                 accept: Optional[Callable[..., bool]] = None):
        self.fn = fn
        self.name = name
        self.family = family
        self.warmup = warmup
        self.pool_mode = pool
        self.clone_outputs = clone_outputs
        self.max_entries = max_entries
        self.constants = set(id(t) for t in (constants or []))
        self.enabled = enabled and torch.cuda.is_available()
        self.rng_guard = rng_guard
        self.sync_audit = sync_audit
        self.on_capture = on_capture
        self.accept = accept  # accept(*args, **kwargs) -> bool: False => run eagerly (e.g. training mode, unsupported flags)
        self.entries: "collections.OrderedDict[Tuple, GraphEntry]" = collections.OrderedDict()
        self.stats = collections.Counter()
        self.events: List[Dict[str, Any]] = []

    # -- pools
    def _pool(self):
        if self.pool_mode == "private":
            return None
        return POOLS.acquire(self.family)

    # -- public
    def __call__(self, *args, **kwargs):
        if not self.enabled or (self.accept is not None and not self.accept(*args, **kwargs)):
            self.stats["eager_disabled"] += 1
            return self.fn(*args, **kwargs)
        sig = signature_of((args, kwargs))
        e = self.entries.get(sig)
        if e is None:
            e, first_out = self._capture(sig, args, kwargs)
            if e.unsupported is not None:
                self.stats["eager_unsupported"] += 1
                return first_out if first_out is not None else self.fn(*args, **kwargs)
            self.entries.move_to_end(sig)
            self._evict()
            self.stats["first_call_from_warmup"] += 1
            return first_out
        if e.unsupported is not None:
            self.stats["eager_unsupported"] += 1
            return self.fn(*args, **kwargs)
        self.entries.move_to_end(sig)
        n = copy_into((e.static_args, e.static_kwargs), (args, kwargs))
        self.stats["input_copies"] += n
        e.graph.replay()
        e.n_replays += 1
        e.last_used = time.time()
        self.stats["replay"] += 1
        return self._out(e)

    def _out(self, e: GraphEntry):
        if not self.clone_outputs:
            return e.static_out
        leaves, spec = tree_flatten(e.static_out)
        return tree_unflatten([l.clone() if isinstance(l, torch.Tensor) else l for l in leaves], spec)

    def _evict(self):
        while len(self.entries) > self.max_entries:
            k, old = self.entries.popitem(last=False)
            self.stats["evicted"] += 1
            if old.graph is not None and self.pool_mode != "private":
                old.graph = None
                POOLS.release(self.family)
            del old

    def clear(self):
        """Drop all graphs (and release their pool shares)."""
        self.max_entries, m = 0, self.max_entries
        self._evict()
        self.max_entries = m

    # -- capture
    def _capture(self, sig, args, kwargs):
        """Returns (entry, output_of_the_warmup_run).  The warm-up run IS the computation for this call (numerically the
        plain eager call); the capture that follows records the same kernels for later replays."""
        t0 = time.time()
        static_args, static_kwargs = static_like((args, kwargs), self.constants)
        copy_into((static_args, static_kwargs), (args, kwargs))
        torch.cuda.synchronize()
        pool = self._pool()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        audit_rep = {}
        first_out = None
        try:
            with torch.cuda.stream(s), _autocast_nocache():
                for i in range(max(1, self.warmup)):
                    if i > 0:
                        copy_into((static_args, static_kwargs), (args, kwargs))  # fn may mutate its inputs in place
                    with RNGGuard(raise_on_use=False) as rg, SyncCensus(mode="warn") as sc:
                        first_out = self.fn(*static_args, **static_kwargs)
                    audit_rep[f"pass{i + 1}"] = {"syncs": sc.report(), "rng_consumed": rg.consumed}
                    if rg.consumed and self.rng_guard:
                        raise RuntimeError(f"RNGGuard: region consumed random numbers from {rg.consumed}")
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
        except Exception as ex:  # noqa
            if is_oom(ex): raise
            msg = f"{type(ex).__name__}: {str(ex)[:300]}"
            self.events.append({"event": "warmup_failed", "name": self.name, "error": msg, "trace": traceback.format_exc()[-2000:]})
            log.warning("[infopt_graphs] %s: warm-up/audit failed, eager fallback for this signature: %s", self.name, msg)
            torch.cuda.synchronize()
            ent = GraphEntry(sig, None, None, None, None, 0.0, time.time() - t0, 0, unsupported=msg)
            self.entries[sig] = ent
            return ent, None
        # the warm-up output may alias the static input buffers (in-place fns): detach it from them by cloning
        leaves, spec = tree_flatten(first_out)
        first_out = tree_unflatten([l.clone() if isinstance(l, torch.Tensor) else l for l in leaves], spec)
        copy_into((static_args, static_kwargs), (args, kwargs))  # restore inputs mutated in place by the warm-up
        warm_s = time.time() - t0
        t1 = time.time()
        reserved0 = torch.cuda.memory_reserved()
        g = torch.cuda.CUDAGraph()
        try:
            with _autocast_nocache(), capture_context(g, pool=pool, stream=s):
                static_out = self.fn(*static_args, **static_kwargs)
        except Exception as ex:  # noqa
            if is_oom(ex): raise
            msg = f"{type(ex).__name__}: {str(ex)[:300]}"
            self.events.append({"event": "capture_failed", "name": self.name, "error": msg, "trace": traceback.format_exc()[-2000:]})
            if pool is not None:
                POOLS.release(self.family)
            log.warning("[infopt_graphs] %s: capture failed, eager fallback for this signature: %s", self.name, msg)
            del g
            torch.cuda.synchronize()
            ent = GraphEntry(sig, None, None, None, None, time.time() - t1, warm_s, 0, unsupported=msg)
            self.entries[sig] = ent
            return ent, first_out
        torch.cuda.synchronize()
        cap_s = time.time() - t1
        growth = torch.cuda.memory_reserved() - reserved0
        e = GraphEntry(sig, g, static_args, static_kwargs, static_out, cap_s, warm_s, growth)
        self.entries[sig] = e
        self.stats["captures"] += 1
        ev = {"event": "captured", "name": self.name, "family": self.family, "warmup_s": round(warm_s, 4), "capture_s": round(cap_s, 4),
              "pool_growth_mb": round(growth / 1e6, 1), "n_entries": len(self.entries), "audit": audit_rep,
              "sig_tensors": [l for l in sig[1:] if isinstance(l, tuple) and l and l[0] == "T"][:6]}
        self.events.append(ev)
        if self.on_capture:
            self.on_capture(ev)
        return e, first_out

    # -- teacher-forced parity on identical inputs (eager twice = floor, replay once)
    def teacher_forced_check(self, *args, **kwargs) -> Dict[str, Any]:
        sig = signature_of((args, kwargs))
        e = self.entries.get(sig)
        if e is None or e.unsupported is not None:
            return {"available": False}
        copy_into((e.static_args, e.static_kwargs), (args, kwargs))
        torch.cuda.synchronize()
        with _autocast_nocache():
            o1 = self.fn(*e.static_args, **e.static_kwargs)
            l1 = [l.clone() for l in tree_flatten(o1)[0] if isinstance(l, torch.Tensor)]
            copy_into((e.static_args, e.static_kwargs), (args, kwargs))
            o2 = self.fn(*e.static_args, **e.static_kwargs)
            l2 = [l.clone() for l in tree_flatten(o2)[0] if isinstance(l, torch.Tensor)]
            copy_into((e.static_args, e.static_kwargs), (args, kwargs))
        torch.cuda.synchronize()
        e.graph.replay()
        torch.cuda.synchronize()
        lg = [l.clone() for l in tree_flatten(e.static_out)[0] if isinstance(l, torch.Tensor)]
        rec = {"available": True, "n_outputs": len(l1)}
        rec["d_graph_vs_eager_max"] = [float((a.float() - b.float()).abs().max()) for a, b in zip(lg, l1)]
        rec["d_eager_vs_eager_max"] = [float((a.float() - b.float()).abs().max()) for a, b in zip(l2, l1)]
        rec["graph_bitwise_equal_eager"] = [bool(torch.equal(a, b)) for a, b in zip(lg, l1)]
        rec["eager_bitwise_repeatable"] = [bool(torch.equal(a, b)) for a, b in zip(l2, l1)]
        rec["out_abs_max"] = [float(a.float().abs().max()) for a in l1]
        self.events.append({"event": "teacher_forced", **rec})
        return rec

    # -- introspection
    def graph_memory(self) -> Dict[str, Any]:
        return {"entries": len(self.entries), "pool_growth_mb_per_entry": [round(e.pool_growth_bytes / 1e6, 1) for e in self.entries.values()],
                "unsupported": sum(1 for e in self.entries.values() if e.unsupported)}

    def summary(self) -> Dict[str, Any]:
        return {"name": self.name, "family": self.family, "pool": self.pool_mode, "stats": dict(self.stats), "memory": self.graph_memory(),
                "capture_times_s": [round(e.capture_s, 4) for e in self.entries.values()],
                "warmup_times_s": [round(e.warmup_s, 4) for e in self.entries.values()],
                "audits": [e.get("audit") for e in self.events if e["event"] == "captured"][:8],
                "teacher_forced": [e for e in self.events if e["event"] == "teacher_forced"][-10:],
                "events": [e for e in self.events if e["event"] not in ("captured", "teacher_forced")][:10]}


# ----------------------------------------------------------------------------------------------------------------------
# explicit static-buffer graph (for loops where the graph updates its own state in place, e.g. a diffusion step)
# ----------------------------------------------------------------------------------------------------------------------
class StaticGraph:
    """Lower-level API: you own the static buffers.  `capture(fn)` warms up and captures `fn()` (a closure over your static
    buffers); `replay()` runs it.  `fn` may mutate its static inputs in place (e.g. x.copy_(x_new)) so that no output
    copy is needed between steps."""

    def __init__(self, name: str = "graph", family: str = "default", pool: str = "shared", warmup: int = 1,
                 rng_guard: bool = True, sync_audit: bool = True):
        self.name, self.family, self.pool_mode, self.warmup = name, family, pool, warmup
        self.rng_guard, self.sync_audit = rng_guard, sync_audit
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.out = None
        self.info: Dict[str, Any] = {}

    def capture(self, fn: Callable[[], Any]) -> "StaticGraph":
        t0 = time.time()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        audit_rep = None
        with torch.cuda.stream(s), _autocast_nocache():
            for i in range(max(1, self.warmup)):
                if i == 0 and (self.sync_audit or self.rng_guard):
                    with RNGGuard(raise_on_use=self.rng_guard) as rg, SyncCensus(mode="warn") as sc:
                        out = fn()
                    audit_rep = {"syncs": sc.report(), "rng_consumed": rg.consumed}
                else:
                    out = fn()
                del out
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        warm_s = time.time() - t0
        t1 = time.time()
        reserved0 = torch.cuda.memory_reserved()
        pool = None if self.pool_mode == "private" else POOLS.acquire(self.family)
        g = torch.cuda.CUDAGraph()
        try:
            with _autocast_nocache(), capture_context(g, pool=pool, stream=s):
                self.out = fn()
        except Exception:
            if pool is not None:
                POOLS.release(self.family)
            raise
        torch.cuda.synchronize()
        self.graph = g
        self.info = {"warmup_s": round(warm_s, 4), "capture_s": round(time.time() - t1, 4),
                     "pool_growth_mb": round((torch.cuda.memory_reserved() - reserved0) / 1e6, 1), "audit": audit_rep}
        return self

    def replay(self):
        self.graph.replay()
        return self.out


def graphs_enabled() -> bool:
    import os
    return os.environ.get("INFOPT_GRAPHS", "1") not in ("0", "false", "False", "off")
