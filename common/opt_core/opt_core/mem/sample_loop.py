"""opt_core.mem.sample_loop — a diffusion engine's roll-out AND its confidence head run one SAMPLE CHUNK at a time (``chunk`` of the
``n_samples`` structures per pass, default 1), each chunk's outputs moved to the host before the next chunk starts, the per-sample
outputs assembled along the sample dimension on the host in stock order. Device memory is then independent of the sample count: the
roll-out's activations, the confidence head's per-sample pair representation ``[S, N, N, C]`` and its logits exist for ``chunk`` samples
at a time, never for ``S``. Engine-free (the roll-out and the confidence head enter as callables), P-independent (no
:class:`~opt_core.mem.rowpair.dist.Layout`: under row sharding every rank runs the same loop on its rows; at ``P == 1`` it is a plain
memory lever), framework-lazy (torch is imported inside the calls).

The RNG discipline is :mod:`opt_core.mem.ckpt`'s ``samples_per_pass`` mechanism (ONE implementation: :class:`~opt_core.mem.ckpt.StockOrderDraws`
driven by its chunk loop ``opt_core.mem.ckpt.run_chunks``): the generator state at loop entry is captured, REWOUND before every chunk, and every
batch-shaped random draw of the roll-out is made at the STOCK (all-samples) shape and sliced to the chunk (``draws.draw(fn, chunk)``), so
sample ``s`` sees exactly the noise the batched roll-out gives it and the generator ends where ONE batched pass ends. Two gates of that
mechanism hold it (every chunk draws the same census of shapes; every chunk's end state equals the first's) and one gate here closes the
remaining hole (a roll-out that ignores ``draws`` and makes chunk-shaped draws after the rewind would REPEAT samples: two chunks whose
roll-out outputs are bit-identical are refused by name, :data:`REFUSE_IDENTICAL`).

Exactness. Per element the arithmetic is the engine's own statements on fewer samples: chunked == batched bit for bit WHEN the engine's
kernels are batch-invariant (fp32 on CPU always; on a GPU stack cuBLAS / attention kernels may select by batch size — the engine's equality
row decides the word, ``bitwise`` where ``torch.equal`` holds, else the mode's band). A confidence head that needs a CROSS-SAMPLE statistic
inside the head (not the per-sample scalars a ranking reads afterwards) cannot be chunked: the adapter declares it with
``needs_cross_sample='<statistic>'`` and :func:`plan` refuses by name (:data:`REFUSE_CROSS_SAMPLE`).

API::

    plan(n_samples, chunk=None, *, default_chunk=1, needs_cross_sample=None, draw_order="stock") -> SamplePlan
        ``chunk`` None -> ``default_chunk`` (source ``default``) else ``given``; ``chunk == n_samples`` -> ONE pass (``passes == 1``, the
        stock pass, named ``one_pass`` in the fields); refusals (:class:`SampleLoopRefused`): ``n_samples < 1``, ``chunk < 1``,
        ``chunk > n_samples`` (:data:`REFUSE_CHUNK`), ``needs_cross_sample`` set, ``draw_order`` outside :data:`~opt_core.mem.ckpt.DRAW_ORDERS`.
    run(plan, rollout, confidence=None, *, generator=None, device=None, sample_dim=1, host=True, pool=None, peak_device=None,
        reset_peaks=False, guard_identical=True, log=None) -> SampleLoopOut
        per chunk, in stock order: [rewind] -> ``x = rollout(chunk, draws)`` (the engine's roll-out over samples ``[chunk.start, chunk.stop)``;
        EVERY batch-shaped draw through ``draws.draw(lambda: <the stock draw>, chunk)``) -> ``y = confidence(x, chunk)`` (the engine's head on
        that chunk; ``None`` -> ``y = x``) -> ``y`` to the host (:func:`opt_core.host.outputs.to_host`, pinned buffers from ``pool`` when given)
        -> the chunk's device peak recorded (``torch.cuda.max_memory_allocated(peak_device)``: the allocator's RUNNING peak after the chunk;
        ``reset_peaks=True`` resets the allocator's peak counter before every chunk so the numbers are per-chunk peaks — off by default because
        the reset also clears the peak the engine's own end-of-run census reads) -> device copies dropped.
        ``out.value`` = the chunks' ``y`` concatenated along ``sample_dim`` (tensors / nested dicts / lists / tuples of tensors; a leaf whose
        shape has no sample dim must not be returned per chunk — return sample-invariant results once, outside the loop).
    SampleLoopOut.fields() -> dict          the census / LEVER-line facts: ``samples``, ``sample_chunk``, ``sample_chunk_source``,
                                            ``sample_passes``, ``draw_order``, ``sample_host`` (``pinned|pageable|device``), ``sample_peak_gb``
                                            (max over chunks; ``none`` without CUDA), ``sample_peaks_gb`` (per chunk),
                                            ``sample_peaks`` (``running`` | ``per_chunk``)
    take(t, chunk, dim)                     ``t.narrow(dim, ...)`` — the slice of an already-materialised all-samples tensor for this chunk
    LEVER = "sample_loop"                   the lever name a kit's ACTIVE / LEVER line carries; registered in :mod:`opt_core.mem.registry`
                                            (family ``chunk``, declared ``measured``: bit-exact-claimed by construction, the engine's equality
                                            row narrows it) with hooks ``n_samples``, ``rollout``, ``confidence``, ``device``, ``sample_dim``
                                            and settings ``chunk``, ``draw_order``; :func:`run_applied` runs the applied lever from a
                                            :class:`~opt_core.mem.registry.Ctx`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .ckpt import DRAW_ORDERS, StockOrderDraws, sample_chunks
from .ckpt import run_chunks as run_chunks_stock_order         # the ONE chunk loop with the stock-order draw discipline (opt_core.mem.ckpt)
from .primitives import MemLeverRefused
from .registry import Applied, Ctx, Refusal, refuse, register

__all__ = ["LEVER", "REFUSE_CHUNK", "REFUSE_CROSS_SAMPLE", "REFUSE_IDENTICAL", "SampleLoopRefused", "SamplePlan", "SampleLoopOut", "plan", "run",
           "take", "run_applied", "HOOKS", "SETTINGS"]

LEVER = "sample_loop"
REFUSE_CHUNK = "chunk_gt_samples"                      # chunk > n_samples
REFUSE_CROSS_SAMPLE = "needs_cross_sample"             # the engine's confidence head needs a cross-sample statistic inside the head
REFUSE_IDENTICAL = "identical_chunks"                  # two chunks' roll-out outputs bit-identical: chunk-shaped draws under the rewind
HOOKS = ("n_samples", "rollout", "confidence", "device", "sample_dim")
SETTINGS = ("chunk", "draw_order")


class SampleLoopRefused(MemLeverRefused):
    """A sample-loop precondition failed (``reason`` starts with one of the REFUSE_* words)."""

    def __init__(self, reason: str):
        super().__init__(LEVER, reason)


def _torch():
    import torch  # noqa: PLC0415
    return torch


@dataclass
class SamplePlan:
    """``n_samples`` structures in passes of ``chunk``; ``chunks`` = the sample slices in stock order; ``source`` = where ``chunk`` came
    from (``given`` | ``default`` | ``one_pass``); ``draw_order`` = :data:`opt_core.mem.ckpt.DRAW_ORDERS`."""
    n_samples: int
    chunk: int
    chunks: List[slice]
    source: str
    draw_order: str = "stock"

    @property
    def passes(self) -> int:
        return len(self.chunks)


def plan(n_samples: int, chunk: Optional[int] = None, *, default_chunk: int = 1, needs_cross_sample: Optional[str] = None,
         draw_order: str = "stock") -> SamplePlan:
    """The chunk plan (module docstring). Refusals by name, never a silent one-pass fallback."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise SampleLoopRefused(f"n_samples must be an int >= 1 (got {n_samples!r})")
    if needs_cross_sample:
        raise SampleLoopRefused(f"{REFUSE_CROSS_SAMPLE}: this engine's confidence head needs the cross-sample statistic {needs_cross_sample!r} inside "
                                "the head — its samples cannot be scored one chunk at a time")
    if draw_order not in DRAW_ORDERS:
        raise SampleLoopRefused(f"draw_order must be one of {DRAW_ORDERS} (got {draw_order!r})")
    if chunk is None:
        k, source = int(default_chunk), "default"
    else:
        if isinstance(chunk, bool) or not isinstance(chunk, int):
            raise SampleLoopRefused(f"chunk must be an int (got {chunk!r})")
        k, source = int(chunk), "given"
    if k < 1:
        raise SampleLoopRefused(f"chunk must be >= 1 (got {k})")
    if k > n_samples:
        raise SampleLoopRefused(f"{REFUSE_CHUNK}: chunk={k} > samples={n_samples} — ask for at most the sample count")
    if k == n_samples:
        source = "one_pass"
    return SamplePlan(n_samples=int(n_samples), chunk=k, chunks=sample_chunks(n_samples, k), source=source, draw_order=draw_order)


def take(t, chunk: slice, dim: int):
    """This chunk's slice of an all-samples tensor ``t`` along ``dim`` (a view)."""
    return t.narrow(dim, chunk.start, chunk.stop - chunk.start)


@dataclass
class SampleLoopOut:
    """``value``: the assembled outputs; ``plan``; ``peaks``: per-chunk device peak bytes (None without CUDA); ``record``: the draw
    census of the chunk loop (``chunks``, ``draw_order``, ``census``); ``host``: ``pinned`` | ``pageable`` | ``device``."""
    value: Any
    plan: SamplePlan
    peaks: List[Optional[int]] = field(default_factory=list)
    record: dict = field(default_factory=dict)
    host: str = "device"
    peaks_kind: str = "running"

    def fields(self) -> dict:
        gb = [None if p is None else round(p / 1e9, 2) for p in self.peaks]
        known = [p for p in self.peaks if p is not None]
        return {"samples": self.plan.n_samples, "sample_chunk": self.plan.chunk, "sample_chunk_source": self.plan.source,
                "sample_passes": self.plan.passes, "draw_order": self.plan.draw_order, "sample_host": self.host,
                "sample_peak_gb": (round(max(known) / 1e9, 2) if known else None), "sample_peaks_gb": gb, "sample_peaks": self.peaks_kind}


def _fingerprint(x):
    """A cheap bit-exact fingerprint of a chunk's roll-out output for the identical-chunks gate: the first tensor leaf, as bytes of a
    bounded prefix plus its float64 sum (equal tensors -> equal fingerprints; the converse holds for all practical roll-outs)."""
    torch = _torch()
    leaf = x
    while not isinstance(leaf, torch.Tensor):
        if isinstance(leaf, dict) and leaf:
            leaf = next(iter(leaf.values()))
        elif isinstance(leaf, (list, tuple)) and leaf:
            leaf = leaf[0]
        else:
            return None
    flat = leaf.detach().reshape(-1)
    head = flat[:4096].to("cpu")
    return (tuple(leaf.shape), str(leaf.dtype), head.numpy().tobytes() if head.dtype != torch.bfloat16 else head.float().numpy().tobytes(),
            float(flat.double().sum().item()))


def run(plan_: SamplePlan, rollout: Callable[[slice, StockOrderDraws], Any], confidence: Optional[Callable[[Any, slice], Any]] = None, *,
        generator=None, device=None, sample_dim: int = 1, host: bool = True, pool=None, peak_device=None, reset_peaks: bool = False,
        guard_identical: bool = True, log: Optional[Callable[[str], None]] = None) -> SampleLoopOut:
    """The loop (module docstring). ``generator`` / ``device`` name the RNG the roll-out draws from (``None`` / ``None`` = the CPU default
    generator; ``device='cuda:i'`` = that device's default generator; an explicit ``torch.Generator`` otherwise); ``peak_device`` = the CUDA
    device whose allocator peak is recorded per chunk (default: ``device`` when it is a CUDA device)."""
    torch = _torch()
    peaks: List[Optional[int]] = []
    prints: list = []
    pdev = peak_device if peak_device is not None else (device if (device is not None and str(device).startswith("cuda")) else None)
    stats_all = {"pinned": 0, "unpinned": 0}

    def body(chunk: slice, draws: StockOrderDraws):
        if pdev is not None and torch.cuda.is_available() and reset_peaks:
            torch.cuda.synchronize(pdev)
            torch.cuda.reset_peak_memory_stats(pdev)
        x = rollout(chunk, draws)
        if guard_identical:
            prints.append(_fingerprint(x))
        y = confidence(x, chunk) if confidence is not None else x
        del x
        if host:
            from ..host.outputs import to_host  # noqa: PLC0415
            st: dict = {}
            y_host = to_host(y, pool=pool, stats=st)
            stats_all["pinned"] += int(st.get("pinned", 0))
            stats_all["unpinned"] += int(st.get("unpinned", 0))
            del y
            y = y_host
        if pdev is not None and torch.cuda.is_available():
            torch.cuda.synchronize(pdev)
            peaks.append(int(torch.cuda.max_memory_allocated(pdev)))
        else:
            peaks.append(None)
        if log is not None:
            pk = peaks[-1]
            log(f"[{LEVER}] samples {chunk.start}:{chunk.stop} of {plan_.n_samples} done" + (f"; chunk peak {pk / 1e9:.2f} GB" if pk is not None else ""))
        return y

    record: dict = {}
    value = run_chunks_stock_order(plan_.n_samples, plan_.chunk, body, generator=generator, device=device, draw_order=plan_.draw_order,
                                   sample_dim=int(sample_dim), record=record)
    if guard_identical and len(prints) > 1:
        seen = {}
        for i, fp in enumerate(prints):
            if fp is None:
                continue
            if fp in seen:
                a, b = plan_.chunks[seen[fp]], plan_.chunks[i]
                raise SampleLoopRefused(f"{REFUSE_IDENTICAL}: the roll-out outputs of samples {a.start}:{a.stop} and {b.start}:{b.stop} are bitwise identical — "
                                        "the roll-out makes chunk-shaped random draws after the per-chunk rewind (route every batch-shaped draw through "
                                        "draws.draw(<stock-shaped draw>, chunk)) or its draws do not depend on the generator named to run()")
            seen[fp] = i
    host_word = "device" if not host else ("pinned" if stats_all["pinned"] and not stats_all["unpinned"] else ("pageable" if stats_all["unpinned"] or stats_all["pinned"] == 0 else "pinned"))
    return SampleLoopOut(value=value, plan=plan_, peaks=peaks, record=record, host=host_word, peaks_kind=("per_chunk" if reset_peaks else "running"))


# ------------------------------------------------------------------------------------------------------------ the registered lever
def _applies(ctx: Ctx) -> Optional[Refusal]:
    h = ctx.require(LEVER, "n_samples", "rollout")
    n = h["n_samples"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return refuse(LEVER, "hooks.n_samples", f"n_samples must be an int >= 1 (got {n!r})")
    if not callable(h["rollout"]):
        return refuse(LEVER, "hooks.rollout", "rollout must be a callable rollout(chunk, draws)")
    conf = ctx.hook(LEVER, "confidence")
    if conf is not None and not callable(conf):
        return refuse(LEVER, "hooks.confidence", "confidence must be a callable confidence(x, chunk) or None")
    chunk = ctx.setting(LEVER, "chunk", None, cast=int)
    order = ctx.setting(LEVER, "draw_order", "stock")
    try:
        plan(int(n), chunk, needs_cross_sample=ctx.hook(LEVER, "needs_cross_sample"), draw_order=order)
    except SampleLoopRefused as e:
        return refuse(LEVER, "plan", e.reason)
    return None


@register(LEVER, family="chunk", exact="measured",
          exact_reason="the roll-out and the confidence head on `chunk` of the N samples per pass, per-sample outputs assembled on the host in stock order; "
                       "draw_order=stock keeps every sample's noise bitwise (opt_core.mem.ckpt StockOrderDraws: RNG rewound per chunk, stock-shaped draws "
                       "sliced, census + end-state gates, identical-chunks gate) so chunked == batched by construction where the engine's kernels are "
                       "batch-invariant — the engine's identity row narrows (bitwise where torch.equal holds, else the mode's band) with its id",
          applies=_applies, description="diffusion roll-out + confidence head one sample chunk at a time; outputs assembled on the host",
          preconditions=("hooks.n_samples", "hooks.rollout", "hooks.confidence", "chunk", "draw_order", "plan"), settings=SETTINGS)
def sample_loop(ctx: Ctx) -> Applied:
    n = int(ctx.hook(LEVER, "n_samples"))
    chunk = ctx.setting(LEVER, "chunk", None, cast=int)
    order = ctx.setting(LEVER, "draw_order", "stock")
    p = plan(n, chunk, draw_order=order)
    return Applied(lever=LEVER, exact="measured",
                   exact_reason=f"draw_order={order}: per-sample noise kept bitwise by construction; kernel selection on {p.chunk} of {n} samples is the engine's",
                   settings={"chunk": p.chunk, "chunk_source": p.source, "draw_order": order, "passes": p.passes},
                   sites=("hooks.rollout",) + (("hooks.confidence",) if ctx.hook(LEVER, "confidence") is not None else ()))


def run_applied(ctx: Ctx, **kw) -> SampleLoopOut:
    """Run the lever as applied on ``ctx.record`` (its ``chunk`` / ``draw_order``) with the hooks' callables; ``kw`` overrides
    :func:`run`'s keyword arguments. The lever not applied -> the one-pass plan (stock), still through the loop (nothing chunked)."""
    rec = getattr(ctx, "record", None)
    applied = None
    if rec is not None:
        for a in rec.applied:
            if a.lever == LEVER:
                applied = a
    n = int(ctx.hook(LEVER, "n_samples"))
    if applied is None:
        p = plan(n, n)
    else:
        p = plan(n, int(applied.settings["chunk"]), draw_order=str(applied.settings["draw_order"]))
    kw.setdefault("device", ctx.hook(LEVER, "device"))
    kw.setdefault("sample_dim", int(ctx.hook(LEVER, "sample_dim", 1)))
    out = run(p, ctx.hook(LEVER, "rollout"), ctx.hook(LEVER, "confidence"), **kw)
    if rec is not None:
        rec.mark(LEVER, detail=f"n={n} chunk={p.chunk} passes={p.passes} draw_order={p.draw_order}")
        if applied is not None:
            applied.settings["last_pass"] = out.fields()
    return out
