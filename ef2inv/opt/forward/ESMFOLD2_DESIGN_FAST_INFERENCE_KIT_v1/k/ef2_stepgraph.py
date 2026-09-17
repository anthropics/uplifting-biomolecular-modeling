"""ef2_stepgraph — CUDA-graph replay of the grad-enabled FoldingTrunk pass (forward graph + backward graph per recycle slot) for EVERY
trunk composition the design kit runs, the stock one included.

What it adds over a plain per-pass capture (the kit's earlier K-C pool, which it supersedes)
---------------------------------------------------------------------
* The backward graph is captured WITHOUT ``retain_graph`` (the make_graphed_callables recipe).  The stock trunk's PairUpdateBlocks carry
  torch.compile'd pieces (the cookbook's ``COMPILE`` / the fork's compile fusions) whose AOTAutograd backward is built with donated
  buffers and refuses a retained graph ("This backward function was compiled with non-empty donated buffers ..."); K-C's capture retains
  and therefore cannot graph the ``exact`` mode's trunk (cuEquivariance triangle kernels + compiled blocks).  Nothing after capture needs
  the autograd graph: replay re-runs the recorded kernels on the recorded buffers.
* Every tensor argument is a static INPUT copied per call — the pair tensor and the pair mask.  A new mask at a captured shape replays
  with that mask (K-C keys its slots on shape/dtype and bakes the first mask in).
* One private memory pool per slot, never shared (interleaved fwd0, fwd1, bwd1, bwd0 replays over one pool alias each other's live
  activations).
* Named counters for the evidence layer: ``stats`` = captures / replays_fwd / replays_bwd / eager (grad-less or non-CUDA calls that fell
  through by design) / recaptures (a shape or autocast change after the first capture: correct, but a cost worth naming), and
  ``disabled_reason`` when capture failed and the trunk runs eagerly from then on (announced once with warnings.warn; never silent).

Numerics: replay launches the kernels the eager pass launches, on the same data, in the same order — bit-identical outputs AND input
gradients whenever those kernels are deterministic (they are under the kit's det recipe; k/test_ef2_stepgraph.py proves tensor equality on
random-init trunks: plain, checkpointed, torch.compile'd blocks, and the cuEquivariance backend when installed).  No RNG runs in the trunk
at inference (dropout off), so no generator registration is needed.

Cost: n_slots × (one trunk pass's saved activations + workspace) held permanently in the private pools (torch.cuda.memory_reserved grows by
that; ``pool_bytes()`` reports it), capture time ≈ (n_warmup + 1) eager passes per slot at the first design step.

Usage (instance-level, reversible):
    import ef2_stepgraph as sg
    sg.enable(model, n_slots=num_loops + 1)      # graphs model.folding_trunk under grad; the no-grad folds (critics) run eagerly
    ... design steps ...
    sg.stats(model); sg.pool_bytes(model)
    sg.disable(model)                           # restores the trunk's forward, frees the pools

Constraints: shapes / dtypes / autocast state fixed per slot (a change re-captures, counted); module patches (kernel levers, checkpoint
policy) must be installed BEFORE the first grad-enabled call and not changed afterwards (disable + enable to re-capture); one trunk pass per
slot per design step (``n_slots`` = trunk passes per step = num_loops + 1; a step that asks for more slots than installed raises by name).
"""
from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = ["GraphedGradSegment", "SegmentPool", "enable", "disable", "stats", "pool_bytes", "LEVER"]

LEVER = "trunk_graphs_v2"


def _sig(tensors: Sequence[Optional[Tensor]], autocast_dtype) -> tuple:
    return tuple((None if t is None else (tuple(t.shape), t.dtype, t.device.index)) for t in tensors) + (autocast_dtype,)


class GraphedGradSegment:
    """``fn(*inputs) -> Tensor`` captured at fixed shapes as two CUDA graphs over one private pool: forward (static inputs -> static output)
    and backward (static grad_out -> static input-grads for the inputs listed in ``grad_idx``).  Inputs are copied into the static buffers
    per call; ``forward()`` / ``backward()`` replay and return the static output / grads (clone them if they must outlive the next replay —
    ``SegmentPool`` does)."""

    def __init__(self, fn: Callable[..., Tensor], example_inputs: Sequence[Optional[Tensor]], *, grad_idx: Sequence[int] = (0,),
                 autocast_dtype=torch.bfloat16, n_warmup: int = 2, capture_error_mode: str = "thread_local"):
        import time
        t0 = time.perf_counter()
        dev = next(t.device for t in example_inputs if t is not None)
        if dev.type != "cuda":
            raise ValueError("GraphedGradSegment needs CUDA tensors")
        self.fn, self.grad_idx, self.autocast_dtype = fn, tuple(grad_idx), autocast_dtype
        self.static_in: List[Optional[Tensor]] = [None if t is None else t.detach().clone() for t in example_inputs]
        for i in self.grad_idx:
            self.static_in[i].requires_grad_(True)
        self.sig = _sig(example_inputs, autocast_dtype)
        wrt = [self.static_in[i] for i in self.grad_idx]

        def run_fwd() -> Tensor:
            with torch.enable_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                return fn(*self.static_in)

        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):                       # warm-up off the capturing path: workspaces, autotune, compiles, allocator
            for _ in range(max(1, n_warmup)):
                out = run_fwd()
                torch.autograd.grad((out,), wrt, (torch.ones_like(out),))
                del out
        torch.cuda.current_stream(dev).wait_stream(side)
        torch.cuda.synchronize(dev)
        self.pool = torch.cuda.graph_pool_handle()          # this segment's own pool (fwd + bwd of ONE slot share it, nothing else)
        self.g_fwd, self.g_bwd = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        m0 = torch.cuda.memory_reserved(dev)
        # capture_error_mode thread_local: other threads / streams may keep allocating while this thread captures (a side-stream lever, a
        # data loader); the default 'global' mode would abort this capture (or fault them) on any such allocator activity
        with torch.cuda.graph(self.g_fwd, pool=self.pool, capture_error_mode=capture_error_mode):
            self.static_out = run_fwd()
        self.static_gout = torch.zeros_like(self.static_out)
        with torch.cuda.graph(self.g_bwd, pool=self.pool, capture_error_mode=capture_error_mode):
            self.static_grads: Tuple[Tensor, ...] = tuple(torch.autograd.grad((self.static_out,), wrt, (self.static_gout,), retain_graph=False))
        torch.cuda.synchronize(dev)
        self.pool_bytes = _pool_bytes(self.pool)                          # the private pool's segments (reserved for this slot's two graphs, permanently)
        if not self.pool_bytes:
            self.pool_bytes = max(0, torch.cuda.memory_reserved(dev) - m0)
        self.capture_s = time.perf_counter() - t0                        # warm-up passes + the two captures (each capture also runs synchronize + gc.collect + empty_cache), wall

    def load(self, inputs: Sequence[Optional[Tensor]]) -> None:
        with torch.no_grad():
            for buf, t in zip(self.static_in, inputs):
                if buf is not None and t is not None and buf.data_ptr() != t.data_ptr():
                    buf.copy_(t)

    def forward(self) -> Tensor:
        self.g_fwd.replay()
        return self.static_out

    def backward(self, grad_out: Tensor) -> Tuple[Tensor, ...]:
        if grad_out.data_ptr() != self.static_gout.data_ptr():
            self.static_gout.copy_(grad_out)
        self.g_bwd.replay()
        return self.static_grads


def _pool_bytes(pool_id) -> int:
    """Bytes of allocator segments owned by the private pool ``pool_id`` (torch.cuda.memory_snapshot's segment_pool_id), 0 if unknown."""
    try:
        want = tuple(pool_id)
        return int(sum(seg.get("total_size", 0) for seg in torch.cuda.memory_snapshot() if tuple(seg.get("segment_pool_id", ())) == want))
    except Exception:  # noqa: BLE001 — snapshot format is not a stable API; the reserved-delta fallback covers it
        return 0


class _SegmentFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, seg: GraphedGradSegment, pool: "SegmentPool", *inputs):
        seg.load(inputs)
        out = seg.forward()
        ctx.seg, ctx.pool, ctx.n_in = seg, pool, len(inputs)
        pool.stats["replays_fwd"] += 1
        return out.clone() if pool.clone_outputs else out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        seg, pool = ctx.seg, ctx.pool
        grads = seg.backward(grad_out.contiguous())
        pool.stats["replays_bwd"] += 1
        out = [None] * ctx.n_in
        for i, g in zip(seg.grad_idx, grads):
            out[i] = g.clone() if pool.clone_outputs else g
        return (None, None, *out)


class SegmentPool:
    """Round-robin slots of GraphedGradSegment for one callable: ``begin_step()`` at the start of every design step, then one call per
    trunk pass.  Slot i is captured lazily at its first use for the seen signature (shapes, dtypes, device, autocast dtype); a later call
    with another signature re-captures that slot (``stats['recaptures']``)."""

    def __init__(self, fn: Callable[..., Tensor], n_slots: int, *, grad_idx: Sequence[int] = (0,), n_warmup: int = 2, clone_outputs: bool = True,
                 capture_error_mode: str = "thread_local", name: str = "trunk"):
        self.fn, self.n_slots, self.grad_idx, self.n_warmup, self.clone_outputs, self.name = fn, int(n_slots), tuple(grad_idx), n_warmup, clone_outputs, name
        self.capture_error_mode = capture_error_mode
        self.slots: List[Optional[GraphedGradSegment]] = [None] * self.n_slots
        self.cursor = 0
        self.stats = {"captures": 0, "recaptures": 0, "replays_fwd": 0, "replays_bwd": 0, "eager": 0, "steps": 0, "capture_s": 0.0}
        self.disabled_reason: Optional[str] = None

    def begin_step(self) -> None:
        self.cursor = 0
        self.stats["steps"] += 1

    def release(self) -> None:
        self.slots = [None] * self.n_slots
        import gc
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()

    def pool_bytes(self) -> int:
        return sum(s.pool_bytes for s in self.slots if s is not None)

    def __call__(self, *inputs: Optional[Tensor]) -> Tensor:
        if self.cursor >= self.n_slots:
            raise RuntimeError(f"ef2_stepgraph {self.name}: pass {self.cursor + 1} in one step but n_slots={self.n_slots} "
                               f"(install with n_slots = trunk passes per step, and call begin_step() once per step)")
        amp = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
        sig = _sig(inputs, amp)
        i = self.cursor; self.cursor += 1
        seg = self.slots[i]
        if seg is None or seg.sig != sig:
            self.stats["recaptures" if seg is not None else "captures"] += 1
            self.slots[i] = None; del seg
            seg = GraphedGradSegment(self.fn, inputs, grad_idx=self.grad_idx, autocast_dtype=amp, n_warmup=self.n_warmup, capture_error_mode=self.capture_error_mode)
            self.slots[i] = seg
            self.stats["capture_s"] = round(self.stats["capture_s"] + seg.capture_s, 3)
        return _SegmentFn.apply(seg, self, *inputs)


# ---------------------------------------------------------------------------------------------------------------- model-level install
def _trunk_of(model) -> torch.nn.Module:
    t = getattr(model, "folding_trunk", None)
    if t is None:
        raise ValueError("ef2_stepgraph.enable: model has no .folding_trunk")
    return t


def enable(model, n_slots: int = 2, *, n_warmup: int = 2, clone_outputs: bool = True, capture_error_mode: str = "thread_local") -> SegmentPool:
    """Graph ``model.folding_trunk`` under grad: every grad-enabled CUDA call ``trunk(pair, pair_attention_mask=mask)`` replays a captured
    slot (round-robin, reset by a pre-hook on ``model.forward``); grad-less calls run the trunk as it is.  Idempotent: a second enable returns
    the installed pool.  Install AFTER the kernel / checkpoint levers (they are captured as they stand at the first design step).
    ``n_slots`` = trunk passes per design step (num_loops + 1; memory = n_slots private pools, see pool_bytes()); ``n_warmup`` eager passes
    per slot before capture; ``clone_outputs`` False hands out the static output / grad buffers (one copy less per pass; the caller must not
    hold them across the next replay of that slot); ``capture_error_mode`` as torch.cuda.graph's (thread_local: concurrent streams may
    allocate during capture)."""
    trunk = _trunk_of(model)
    st = getattr(model, "_sg", None)
    if st is not None:
        if vars(trunk).get("forward") is st["wrapper"]:     # installed and still the trunk's forward: idempotent
            return st["pool"]
        disable(model)                                      # another lever re-bound trunk.forward after this lever (e.g. a checkpoint policy re-applied): re-install over it, re-capture
    inner = trunk.forward                                   # whatever forward the trunk has now (stock, or a kit policy bound on the instance)

    def trunk_pass(pair: Tensor, mask: Optional[Tensor]) -> Tensor:
        return inner(pair, pair_attention_mask=mask)

    pool = SegmentPool(trunk_pass, n_slots, grad_idx=(0,), n_warmup=n_warmup, clone_outputs=clone_outputs, capture_error_mode=capture_error_mode,
                       name=f"trunk[{type(model).__name__}]")

    def graphed_forward(pair: Tensor, pair_attention_mask: Optional[Tensor] = None) -> Tensor:
        if pool.disabled_reason is None and torch.is_grad_enabled() and pair.is_cuda and not torch.cuda.is_current_stream_capturing():
            try:
                return pool(pair, pair_attention_mask)
            except Exception as e:  # noqa: BLE001 — capture failed on this stack (incl. CUDA OOM while filling a slot's pool): named once, eager from now on
                pool.disabled_reason = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
                pool.release()
                warnings.warn(f"ef2_stepgraph {pool.name}: CUDA-graph capture failed, the trunk runs eagerly from now on — {pool.disabled_reason}")
        pool.stats["eager"] += 1
        return inner(pair, pair_attention_mask=pair_attention_mask)

    handle = model.register_forward_pre_hook(lambda m, a: pool.begin_step())
    had_instance_forward = "forward" in vars(trunk)
    trunk.forward = graphed_forward
    model._sg = {"pool": pool, "inner": inner, "hook": handle, "had_instance_forward": had_instance_forward, "wrapper": graphed_forward}
    return pool


def disable(model) -> None:
    st = getattr(model, "_sg", None)
    if st is None:
        return
    trunk = _trunk_of(model)
    st["hook"].remove()
    if vars(trunk).get("forward") is st["wrapper"]:         # this wrapper is still installed: put back what it wrapped (else leave the later lever's forward alone)
        if st["had_instance_forward"]:
            trunk.forward = st["inner"]
        else:
            del trunk.forward
    st["pool"].release()
    del model._sg


def stats(model) -> Optional[dict]:
    st = getattr(model, "_sg", None)
    if st is None:
        return None
    p = st["pool"]
    return dict(p.stats, slots=sum(s is not None for s in p.slots), n_slots=p.n_slots, disabled_reason=p.disabled_reason, pool_bytes=p.pool_bytes(),
                active=vars(_trunk_of(model)).get("forward") is st["wrapper"])


def pool_bytes(model) -> int:
    st = getattr(model, "_sg", None)
    return 0 if st is None else st["pool"].pool_bytes()
