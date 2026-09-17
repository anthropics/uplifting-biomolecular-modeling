"""P6 `graph`: whole-forward CUDA graphs, one per batch shape, for the exact kit.

Why: at small eval batches the GPN-Star forward is host/launch-bound -- ~1,450 kernel launches per forward whatever the batch, so at
B=8 x 128 bp the GPU is busy for about half of the wall time and at B=1 for much less.  The exact levers P0-P2 already removed every
host->device copy and host sync from the steady-state forward (the layer-0 self-check and the constant caches run on the FIRST forward
of a shape only), which is what makes the whole forward capturable; this lever records it once per batch shape into a CUDA graph and
replays the graph for every later batch of that shape: one launch, no Python between kernels.

Exactness: a replay runs the kernels the eager forward dispatched at capture time, on the same shapes, in the same order, with the
same cuBLAS/cuBLASLt algorithm choices (made at capture on the same handle / workspace size) -> the logits are the eager kit's, which
are the stock's, bit for bit.  This is checked on the GPU by the engage tests and the probes (per-window logits digests), never assumed.

How a shape is served (ShapeGraphs.__call__):
  call 1 (and any call that still validates): EAGER -- the kit's first-forward work for the shape (target-species contract, cached
          constants, the layer-0 reduced-vs-full K/V self-check with its host syncs) happens here, outside any capture;
  first call after a clean eager call: CAPTURE -- static input buffers (clones), one warm-up forward on a side stream, capture into
          the process's shared graph pool, then a replay serves this very call; printed once per shape:
          `[gpnstar-opt] GRAPH state=captured shape=<B>x<L> cost_s=<s> pool_mib=<MiB> graphs=<n>`;
  every later call: REPLAY -- copy input_ids/source_ids into the static buffers (device-to-device), check target_species against the
          captured constant, graph.replay(), return a CLONE of the static output (the caller may hold outputs across batches, as the
          transformers Trainer's prediction loop does).
The captured forward uses the static unified-K/V route (P3b) for the shape, never the dynamic de-dup (P4: its unique-row count is
data dependent -> one host sync per forward, not capturable); a shape whose reduced route the self-check moved to the stock
projections is captured on the stock projections (also sync-free).  Shapes above GRAPH_MAX_TOKENS tokens (B*L) keep the eager de-dup
route: there the forward is bandwidth-bound, launch overhead is a small fraction, and de-dup's GEMM savings win (measured crossover
between 16 and 32 windows of 128 bp on the H100, see CHANGES.md).  A capture that does not fit in free device memory by the footprint rule, or that fails, leaves the shape UNGRAPHED
(printed once, `GRAPH state=ungraphed shape=<B>x<L> reason=<why>`): the same patched modules run eagerly -- exact either way.

Not served by a graph (-> the eager kit, unchanged): grad mode on, model.train(), CPU tensors, positional arguments, labels /
output_probs / loss_weight / output_attentions / output_hidden_states / return_dict=False, a caller that is itself capturing.
"""

from __future__ import annotations

import gc
import sys
import time
from typing import Any, Callable

import torch
from torch import nn

from .patches import core_model

GRAPH_MAX_TOKENS = 2048            # B*L up to which a batch shape is captured (16 x 128 bp; 4 x 512 bp).  Measured on the H100 under TF32 with `colattn`: the graph wins by 1.30x at 16 x 128, ties the eager de-dup route at 24 x 128 and loses from 32 x 128 on; above the cutoff the eager route runs, untouched.
CAPTURE_MARGIN = 1.25              # a capture is attempted only when CAPTURE_MARGIN x (the shape's eager activation footprint) + CAPTURE_FIXED_BYTES is free
CAPTURE_FIXED_BYTES = 512 << 20
_INPUT_KEYS = ("input_ids", "source_ids", "target_species")
_FLAG_FALSE_OK = ("output_attentions", "output_hidden_states")          # accepted when falsy
_NONE_OK = ("labels", "output_probs", "loss_weight")                    # accepted when None
TAG = "[gpnstar-opt]"


def _default_notify(text: str) -> None:
    print(f"{TAG} {text}", file=sys.stderr, flush=True)


class _Entry:
    __slots__ = ("key", "shape", "calls", "eager_calls", "clean", "graph", "static_in", "rest", "out", "failed", "capture_s",
                 "pool_bytes", "need_bytes", "replays")

    def __init__(self, key, shape):
        self.key = key
        self.shape = shape              # (B, L)
        self.calls = 0
        self.eager_calls = 0
        self.clean = False              # the last eager call of this shape ran without self-check / memory-fallback work -> capturable
        self.graph = None
        self.static_in = None
        self.rest = None
        self.out = None
        self.failed = None              # reason (str) once the shape is UNGRAPHED
        self.capture_s = None
        self.pool_bytes = 0
        self.need_bytes = 0             # eager activation footprint observed for the shape (upper bound), for the capture-fits rule
        self.replays = 0


class ShapeGraphs:
    """Callable installed as `model.forward`: serves each eligible call from a per-shape CUDA graph (capturing it first), else eagerly."""

    def __init__(self, model: nn.Module, inner: Callable, *, max_tokens: int = GRAPH_MAX_TOKENS, notify: Callable[[str], None] | None = None):
        self.model = model
        self.inner = inner
        self.max_tokens = int(max_tokens)
        self.notify = notify or _default_notify
        self.entries: dict[Any, _Entry] = {}
        self.pool = None
        self.enabled = True
        self.n_eager = 0
        self.n_replay = 0

    # ------------------------------------------------------------------ dispatch
    def __call__(self, *args, **kwargs):
        e = self._entry_for(args, kwargs) if self.enabled else None
        if e is None:
            return self._eager_call(None, args, kwargs)
        e.calls += 1
        if e.graph is not None:
            return self._replay(e, kwargs)
        if e.failed is None and e.clean:
            ok = self._try_capture(e, kwargs)
            if ok:
                return self._replay(e, kwargs)
        return self._eager_call(e, args, kwargs)

    def _entry_for(self, args, kwargs) -> _Entry | None:
        if args or torch.is_grad_enabled() or self.model.training:
            return None
        try:
            ids, src, ts = kwargs["input_ids"], kwargs["source_ids"], kwargs["target_species"]
        except KeyError:
            return None
        for k, v in kwargs.items():
            if k in _INPUT_KEYS:
                continue
            if k in _FLAG_FALSE_OK and not v:
                continue
            if k in _NONE_OK and v is None:
                continue
            if k == "return_dict" and v in (None, True):
                continue
            return None
        if not (torch.is_tensor(ids) and torch.is_tensor(src) and torch.is_tensor(ts)):
            return None
        if not (ids.is_cuda and src.is_cuda and ts.is_cuda) or not (ids.device == src.device == ts.device):
            return None
        if ids.dim() != 3 or src.dim() != 3 or ts.dim() != 2 or ids.shape[:1] != ts.shape[:1] or ids.shape[:2] != src.shape[:2]:
            return None
        B, L = int(ids.shape[0]), int(ids.shape[1])
        if B * L > self.max_tokens or B == 0:
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        st = getattr(core_model(self.model), "_exact_state", None)
        if st is None or not st.clade_species:      # the graphs rest on P0-P2 (no host syncs / H2D copies in the steady-state forward)
            return None
        key = (B, L, int(ids.shape[2]), int(src.shape[2]), ids.dtype, src.dtype, ts.dtype, ids.device.index, bool(torch.is_inference_mode_enabled()),
               tuple(sorted((k, bool(v) if k in _FLAG_FALSE_OK else None) for k, v in kwargs.items() if k not in _INPUT_KEYS)))
        e = self.entries.get(key)
        if e is None:
            e = _Entry(key, (B, L))
            self.entries[key] = e
        return e

    # ------------------------------------------------------------------ eager
    def _eager_call(self, e: _Entry | None, args, kwargs):
        st = getattr(core_model(self.model), "_exact_state", None)
        n_log = n_mf = None
        if e is not None and st is not None:
            self._route_for_capture(st, e.shape)
            n_log = len(st.validation_log or [])
            n_mf = sum((st.memory_fallbacks or {}).values())
            dev = kwargs["input_ids"].device
            alloc0 = torch.cuda.memory_allocated(dev)
        try:
            out = self.inner(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            if not self._release_all("out of device memory in an eager forward: graph pools released"):
                raise
            out = None
        if out is None:  # retried once with the graph pools handed back (never fail where the eager kit runs); a second OOM is the kit's/stock's own
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            out = self.inner(*args, **kwargs)
        self.n_eager += 1
        if e is not None and st is not None:
            e.eager_calls += 1
            e.clean = (len(st.validation_log or []) == n_log) and (sum((st.memory_fallbacks or {}).values()) == n_mf)
            peak = torch.cuda.max_memory_allocated(dev)
            e.need_bytes = max(e.need_bytes, int(peak - alloc0))   # >= this call's own activation peak (an earlier, larger peak only makes the rule stricter)
        return out

    @staticmethod
    def _route_for_capture(st, shape) -> None:
        """A captured shape runs the static unified-K/V route (P3b), decided BEFORE its first (validating) eager forward so the route the
        self-check validates is the one the graph records.  A shape already moved to the stock projections stays there."""
        if st.dedup is True:
            m_stock = shape[0] * shape[1] * len(st.clade_species)
            if st.mode_override is None:
                st.mode_override = {}
            if st.mode_override.get(m_stock) is None:
                st.mode_override[m_stock] = "p3b"

    # ------------------------------------------------------------------ capture
    def _fits(self, e: _Entry, dev) -> tuple[bool, int, int]:
        free, _total = torch.cuda.mem_get_info(dev)
        free += max(0, torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev))  # cached, unoccupied blocks are returned before the capture
        need = int(CAPTURE_MARGIN * e.need_bytes) + CAPTURE_FIXED_BYTES
        return need <= free, need, int(free)

    def _try_capture(self, e: _Entry, kwargs) -> bool:
        dev = kwargs["input_ids"].device
        B, L = e.shape
        fits, need, free = self._fits(e, dev)
        if not fits:
            e.failed = f"capture_needs_{need >> 20}MiB_free_{free >> 20}MiB"
            self.notify(f"GRAPH state=ungraphed shape={B}x{L} reason={e.failed}")
            return False
        t0 = time.perf_counter()
        err = None
        try:
            self._capture(e, kwargs, dev)
        except Exception as ex:  # noqa: BLE001 -- a failed capture leaves the shape to the eager kit (exact); the reason is printed once
            err = ex
        if err is not None:
            e.graph = e.static_in = e.out = e.rest = None
            e.failed = ("capture_oom" if isinstance(err, torch.cuda.OutOfMemoryError) else "capture_error:" + type(err).__name__)
            gc.collect()
            try:
                torch.cuda.synchronize(dev)
            except Exception:  # noqa: BLE001
                pass
            torch.cuda.empty_cache()
            self.notify(f"GRAPH state=ungraphed shape={B}x{L} reason={e.failed} detail={_token(str(err))[:160]}")
            return False
        e.capture_s = time.perf_counter() - t0
        n = sum(1 for x in self.entries.values() if x.graph is not None)
        self.notify(f"GRAPH state=captured shape={B}x{L} cost_s={e.capture_s:.2f} pool_mib={max(0, e.pool_bytes) >> 20} graphs={n}")
        return True

    def _capture(self, e: _Entry, kwargs, dev) -> None:
        with torch.inference_mode(False), torch.no_grad():   # normal tensors: writable in place under no_grad AND inference_mode callers
            static_in = {k: kwargs[k].detach().clone() for k in _INPUT_KEYS}
        rest = {k: v for k, v in kwargs.items() if k not in _INPUT_KEYS}
        torch.cuda.synchronize(dev)
        gc.collect()
        torch.cuda.empty_cache()                      # blocks cached by the eager forwards belong to the caller's stream: hand them back before the side stream allocates its own
        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):                   # warm-up on the capture stream (cuBLAS handle/workspace for the stream, Triton/lazy inits)
            self.inner(**static_in, **rest)
        torch.cuda.current_stream(dev).wait_stream(s)
        torch.cuda.synchronize(dev)
        gc.collect()
        torch.cuda.empty_cache()                      # the warm-up's activations sit in the allocator's cache: hand them back before the pool takes its own
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()   # ONE pool for every shape of this model: graphs replay one at a time on one stream
        reserved0 = torch.cuda.memory_reserved(dev)
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, pool=self.pool, stream=s):
                out = self.inner(**static_in, **rest)
        except BaseException:
            g.reset()                                 # a failed capture returns its pool share at once
            raise
        torch.cuda.synchronize(dev)
        e.pool_bytes = int(torch.cuda.memory_reserved(dev) - reserved0)
        e.graph, e.static_in, e.rest, e.out = g, static_in, rest, out

    # ------------------------------------------------------------------ replay
    def _replay(self, e: _Entry, kwargs):
        sin = e.static_in
        ts = kwargs["target_species"]
        sts = sin["target_species"]
        if ts.data_ptr() != sts.data_ptr() and not torch.equal(ts, sts):   # one small host sync; the graph baked the captured species constants in
            st = getattr(core_model(self.model), "_exact_state", None)
            raise ValueError("gpnstar_exact: target_species differs from the value the exact-mode constants were built for "
                             f"(expected all == {getattr(st, 'target_index', 0)}); refusing to replay the captured forward for this batch.")
        for k in ("input_ids", "source_ids"):
            v = kwargs[k]
            if v.data_ptr() != sin[k].data_ptr():
                sin[k].copy_(v, non_blocking=True)
        e.graph.replay()
        e.replays += 1
        self.n_replay += 1
        return _clone_output(e.out)

    # ------------------------------------------------------------------ housekeeping
    def _release_all(self, reason: str) -> bool:
        """Drop every graph (pools handed back at the next empty_cache) and stop capturing; True if anything was released."""
        had = False
        for x in self.entries.values():
            if x.graph is not None:
                had = True
                try:
                    x.graph.reset()
                except Exception:  # noqa: BLE001
                    pass
            x.graph = x.static_in = x.out = x.rest = None
            if x.failed is None:
                x.failed = "released"
        if had:
            self.enabled = False
            self.notify(f"GRAPH state=released shape=all reason={_token(reason)}")
        return had

    def report(self) -> dict:
        return {"enabled": self.enabled, "max_tokens": self.max_tokens, "n_eager": self.n_eager, "n_replay": self.n_replay,
                "shapes": {f"{x.shape[0]}x{x.shape[1]}": {"calls": x.calls, "eager_calls": x.eager_calls, "replays": x.replays,
                                                        "state": "captured" if x.graph is not None else ("ungraphed:" + x.failed if x.failed else "eager"),
                                                        "capture_s": x.capture_s, "pool_mib": x.pool_bytes >> 20, "need_mib": x.need_bytes >> 20}
                           for x in self.entries.values()}}


def _token(s: str) -> str:
    return "_".join(str(s).split()) or "none"


def _clone_output(out):
    """A copy of the static output the caller may keep: tensors cloned (fresh allocator memory), everything else as is."""
    if torch.is_tensor(out):
        return out.clone()
    if isinstance(out, tuple):
        return type(out)(_clone_output(o) for o in out)
    if hasattr(out, "items") and hasattr(out, "__class__"):
        try:
            return out.__class__(**{k: (v.clone() if torch.is_tensor(v) else _clone_output(v)) for k, v in out.items()})
        except Exception:  # noqa: BLE001
            pass
    return out


def patch_graphs(model: nn.Module, *, max_tokens: int = GRAPH_MAX_TOKENS, notify: Callable[[str], None] | None = None) -> ShapeGraphs:
    """Install P6 on `model` (a GPNStarForMaskedLM or GPNStarModel with the exact levers applied, `colattn` included: its Triton gathers are
    compiled and probed at install / first eager forwards, before any capture): model.forward becomes a ShapeGraphs.
    Idempotent; returns the runner (also at model._shape_graphs)."""
    sg = getattr(model, "_shape_graphs", None)
    if isinstance(sg, ShapeGraphs):
        sg.max_tokens = int(max_tokens)
        if notify is not None:
            sg.notify = notify
        return sg
    sg = ShapeGraphs(model, model.forward, max_tokens=max_tokens, notify=notify)
    model.forward = sg           # instance attribute shadows the bound method; nn.Module.__call__ (hooks included) drives it unchanged
    model._shape_graphs = sg
    return sg


def unpatch_graphs(model: nn.Module) -> None:
    sg = getattr(model, "_shape_graphs", None)
    if isinstance(sg, ShapeGraphs):
        sg._release_all("unpatched")
        model.forward = sg.inner
        model._shape_graphs = None
        gc.collect()
        torch.cuda.empty_cache()


def graph_report(model: nn.Module) -> dict | None:
    sg = getattr(model, "_shape_graphs", None)
    return sg.report() if isinstance(sg, ShapeGraphs) else None
