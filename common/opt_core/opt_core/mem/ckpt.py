"""Contract. Boundary discipline for inference — the recycle seam, diffusion-sample chunking, the seed-batch cap, the graph-capture
budget and the hoist-cache switch — as engine-free levers of the ``big`` registry (:mod:`opt_core.mem.registry`), declared with
``@register`` at import and applied by name through :func:`opt_core.mem.apply`. A lever reads its hook points from
``ctx.hooks[<lever>]`` (:data:`HOOKS` lists the keys) and its values through ``ctx.setting`` (the kit's ``ctx.settings[<lever>]``
flags over ``ctx.settings``); a precondition that is absent is a ``Refusal`` naming it — never a silent no-op, never a silent stock
fallback; a discipline violated at run time is :class:`CkptError`. A configuration that would be stock's own (one chunk of every
sample, one seed per pass) is refused by name — never a phantom change, never a narrowed label. Every lever carries its declared
label at apply (``Applied.exact`` = the registered one; a narrower label is the engine's equality row's to write, with its record id
as ``narrowed_by``), the values in force and the sites wired. A levered path marks the record when it runs (``ctx.record.mark``): the
census of :mod:`opt_core.mem.record` sees every seam, every sample pass, every capture decision; the per-item levers
(``recycle_carry``, ``samples_per_pass``, ``seed_batch``) are unit scope — the adapter opens a unit per item
(``record.unit_begin`` / ``unit_end``) around the seams, the sample pass and the seed plan; ``graph_capture`` and ``hoist_off`` act once
per process (``scope="process"``). No lever here writes an environment
variable or names an engine: an engine adapter wires the hooks and maps the typed values onto its own kit's vocabulary. Parking is
:class:`opt_core.mem.offload.HostPark`'s (the one pinned-parking primitive of the library; imported inside the mechanism);
the seam's cache release is the ``cache_release`` lever's (:func:`opt_core.mem.allocator.release`); the peak counters are
:func:`opt_core.mem.allocator.counters`.

Levers (name · family · declared exact · settings · hooks):

* ``recycle_carry`` · ckpt · bit-exact · ``park`` (``host`` | ``device``, no default), ``pin_max_gb`` (the pinned budget; required
  under ``host``), ``flush`` (0|1, default 1) · hooks ``device``, ``carried`` — the recycle seam. :class:`RecycleCarry` holds the
  carried state the engine names in ``carried`` (the single and pair representations, the recycle features, the cycle-invariant
  initial embeddings; a name outside it is :class:`CkptError`); every other reference of a cycle is the engine's to drop before
  :meth:`RecycleCarry.seam`, which releases the caching allocator through the ``cache_release`` lever (policy ``per_stage``, applied
  earlier in the line; ``flush=1`` refuses by name without it) so the next cycle's transients are re-allocated from a clean cache.
  ``park=host`` parks the carried tensors in pinned host memory between their uses through ``HostPark`` (a cycle-invariant plane
  parked for the trunk body is one full pair plane off the device; on a measured trunk −12.5 % of ``max_memory_allocated`` at every
  size); :meth:`RecycleCarry.get` returns a fresh device copy. ``park=device`` is the seam-only diagnostic arm: the tensors stay
  where they are and the peak counters are unchanged on the same trunk (the seam release moves ``reserved`` at the seam, never the
  in-cycle peak) — its ``Applied`` says so. Copies change no byte: ``bitwise``. The in-process seam is the lever; a
  checkpoint-to-disk-and-resume form is not (a resumed process is Tier-2 by resume).
* ``samples_per_pass`` · chunk · measured · ``k`` (``< N``; ``k >= N`` is the stock pass and refuses by name), ``draw_order``
  (``stock`` | ``chunked``) · hooks ``n_samples``, ``run``, ``generator``, ``device``, ``sample_dim`` — the diffusion head on ``k`` of
  the ``N`` samples at a time (:func:`run_sample_chunks` at sampling time; outputs concatenated along ``sample_dim`` in stock order).
  Under ``draw_order=stock`` the draws are stock's by construction (:class:`StockOrderDraws`: the RNG state at sampler entry is
  captured, rewound before every chunk, and every batch-shaped draw is made at the stock shape and sliced to the chunk along
  ``sample_dim``, so every sample sees the noise stock would have given it and the stream ends where stock's ends; two gates hold it:
  the draw census — a chunk that draws a different count or shape than the first is :class:`CkptError` — and the end-state gate — the
  RNG state at the end of every chunk must equal the first chunk's, so a draw made outside ``draws.draw`` is caught; the equality
  record :data:`IDENTITY_RECORD` holds the draws ``torch.equal`` and the stream's end state at every size and ``k``); the denoiser's
  kernel selection on the smaller batch is the engine's — ``measured``, the engine's equality row narrows (``bitwise`` where
  ``torch.equal`` holds, ``band`` by the D82 seed-spread word) with its id. ``draw_order=chunked`` (chunk-shaped draws, no rewind, no
  gates) re-orders the stream: ``measured`` with the D82 reason — band-class by construction, the equality row writes it with its id.
* ``seed_batch`` · ckpt · measured · ``k`` (``>= 2``; ``k=1`` is the stock loop and refuses by name; no default) · hooks ``seeds``,
  ``capable`` — ``k`` seeds per pass in stock order (:func:`seed_batch_plan`): each seed's draws are its own by construction (K
  independent RNG streams in one batch) — bit-exact-claimed until the engine's equality row, which narrows with its id (the batched
  kernels may select differently from the per-seed ones: the equality record's batched-einsum finding); a speed lever with a memory
  cost (k× the batched stage's per-seed working set), never a default; ``capable`` = the engine's runner batches seeds on independent
  RNG streams (refuses otherwise).
* ``graph_capture`` · setting · measured (bit-exact-claimed) · process scope · ``policy`` (``off`` | ``budget``, default ``off``),
  ``budget`` (tokens ≥ 1) · hooks ``switch``, ``on_event`` — the CUDA-graph capture policy as a typed value: :class:`GraphPolicy` answers ``admit(site,
  tokens)`` (``off``: never; ``budget``: only a fold of at most ``budget`` tokens), counts every refusal per site, calls
  ``on_event(site, tokens)`` once per (site, tokens) and marks every decision on the record. The adapter's ``switch(policy)`` maps it
  onto the kit's own capture plumbing and vocabulary (a kit whose own budget variable means "0 = capture everything" maps
  ``policy=off`` to its no-capture form and ``budget:N`` to ``N``; the core's budget is never 0). The eager path launches the
  captured path's kernels — bit-exact-claimed, ``measured`` until the engine's equality row narrows it with its id. Pairs with the
  allocator lever's graph-pool rule (``ctx.graphs`` False under this lever).
* ``hoist_off`` · setting · measured (bit-exact-claimed) · process scope · no settings · hook ``disable`` — the sampler's step-invariant
  (hoist) cache off through the adapter's ``disable()`` (returns the restore callable or None): a memoization removed, the values
  recomputed per step — bit-exact-claimed, ``measured`` until the engine's equality row narrows it with its id.

Tested properties. ``recycle_carry`` carries exactly the per-cycle state set (the input single, the single, the pair, the cycle
index) and is ``torch.equal`` to the un-parked recycle loop under both parks. ``samples_per_pass`` draws in stock order — a sampler chunk
that re-orders draws is refused by construction (:class:`StockOrderDraws`) — with draws ``torch.equal`` and the RNG stream's end state
equal to stock's at every size and ``k``; the denoiser's outputs are the engine's equality row's to label. ``seed_batch`` caps the seed
branches evaluated at once (bit-exact per branch, a memory cost per extra branch). ``graph_capture``'s budget is a token count ≥ 1
(``off`` = never capture; this module's budget is never 0). :data:`IDENTITY_RECORD` names the equality record of these mechanisms (one
H100 80GB, through the registry path); no lever here narrows its declared label: ``Applied.narrowed_by`` is the engine's equality row's
to write, with its own record id.

This module imports only the standard library and the registry at module level; torch and the parking primitive are imported inside
the mechanisms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from . import allocator, graph_gate
from .registry import Applied, Ctx, Refusal, off_ref, refuse, register, setting_ref

IDENTITY_RECORD = "run-20260901T100609Z"                                 # the equality record of these mechanisms (module docstring)
PARKS = ("host", "device")
DRAW_ORDERS = ("stock", "chunked")
GRAPH_POLICIES = ("off", "budget")

HOOKS: dict = {
    "recycle_carry": ("device", "carried"),
    "samples_per_pass": ("n_samples", "run", "generator", "device", "sample_dim"),
    "seed_batch": ("seeds", "capable"),
    "graph_capture": ("switch", "on_event"),
    "hoist_off": ("disable",),
}


class CkptError(RuntimeError):
    """A boundary discipline violated at run time (the message names the lever and the violation)."""


def _torch():
    import torch  # noqa: PLC0415  (never at module level: the core imports only the standard library there)
    return torch


def _is_cuda_device(device) -> bool:
    return device is not None and str(device).startswith("cuda")


def _applied(ctx: Optional[Ctx], lever: str) -> Optional[Applied]:
    """The lever's Applied on the record attached to ``ctx``, or None (the lever is not applied: its levered path is not taken)."""
    rec = getattr(ctx, "record", None)
    if rec is None:
        return None
    for a in rec.applied:
        if a.lever == lever:
            return a
    return None


def _mark(ctx: Optional[Ctx], lever: str, detail: str) -> None:
    rec = getattr(ctx, "record", None)
    if rec is not None:
        rec.mark(lever, detail=detail)


def parse_bool(v: str) -> bool:
    """The 0|1 setting cast (``1/true/on/yes`` → True, ``0/false/off/no`` → False; anything else raises by name) — the one copy the
    lever modules share."""
    s = str(v).strip().lower()
    if s in ("1", "true", "on", "yes"):
        return True
    if s in ("0", "false", "off", "no"):
        return False
    raise ValueError("expected 0 or 1")


# ------------------------------------------------------------------------------------------------------------ recycle seam


class RecycleCarry:
    """The carried state across recycle seams (module contract). ``hold(name=tensor, ...)`` registers the carried tensors (names from
    ``carried`` only; detached): under ``park="host"`` each is parked through ``HostPark`` (a pinned buffer on a CUDA device, a plain
    host buffer on the CPU test line; a buffer the pinned budget or the host cannot give is the primitive's refusal by name — never a
    pageable copy) and the device copy is the caller's to drop; under ``park="device"`` it is kept where it is. ``get(name)``
    returns the tensor to compute on: a fresh device copy of a parked tensor (the caller drops it after use), the held tensor itself
    under ``device``. ``seam(cycle)`` is the boundary after the engine dropped every reference of the cycle (its ``del``s must leave
    no reference cycle: nothing here collects garbage): the allocator is released through the ``cache_release`` lever when
    ``flush`` (policy ``per_stage`` on the record, checked at construction), the counters before / after go on the seam record and
    the record is marked. ``release()`` closes the park and drops everything (``get`` after it is :class:`CkptError`). ``record``
    accumulates ``{"park", "flush", "seams": [...], "held": {name: {where, bytes, shape, dtype}}}`` and, under ``host``, the
    primitive's own record at release."""

    def __init__(self, ctx: Optional[Ctx], device, carried: Sequence[str], *, park: str, flush: bool = True,
                 pin_max_gb: Optional[float] = None, tag: str = "recycle_carry"):
        if park not in PARKS:
            raise CkptError(f"recycle_carry: park must be one of {PARKS} (got {park!r})")
        if flush and _cache_release_policy(ctx) != "per_stage":
            raise CkptError("recycle_carry: flush=1 needs the cache_release lever applied with policy per_stage on the record before it; "
                            "set flush=0 to park without a seam release")
        if isinstance(carried, str) or not carried:
            raise CkptError(f"recycle_carry: carried must name the carried tensors (got {carried!r})")
        self.ctx, self.device, self.park, self.flush = ctx, device, park, bool(flush)
        self.carried = tuple(str(c) for c in carried)
        self._held: dict = {}
        self._park = None
        self.record: dict = {"park": park, "flush": self.flush, "carried": list(self.carried), "seams": [], "held": {}}
        if park == "host":
            from . import offload  # noqa: PLC0415  (the one parking primitive; a lazy import keeps the module stdlib-only at load)
            if pin_max_gb is None:
                raise CkptError("recycle_carry: park=host needs pin_max_gb (the pinned budget for the carried planes, GiB)")
            dev = "cpu" if str(device) == "cpu" else str(device)
            self._park = offload.HostPark(offload.Settings(pin_max_gb=float(pin_max_gb), min_tokens=0, cols="rowloop", device=dev), tag=tag)
            self._park.check()
            self.record["pin_max_gb"] = float(pin_max_gb)

    def hold(self, **tensors) -> None:
        torch = _torch()
        for name, t in tensors.items():
            if name not in self.carried:
                raise CkptError(f"recycle_carry: {name!r} is not a carried name ({', '.join(self.carried)})")
            if not isinstance(t, torch.Tensor):
                raise CkptError(f"recycle_carry: carried {name!r} is not a tensor ({type(t).__name__})")
            t = t.detach()
            nbytes = t.numel() * t.element_size()
            if self._park is not None:
                t = t.contiguous()
                if name in self._park.tensors:
                    ht = self._park.tensors[name]
                    if tuple(t.shape) == ht.shape and t.dtype == ht.dtype:
                        st = self._park.streams
                        ht.put(t)                                            # D2H into the same parked buffer
                        st.sync(st.compute())                                # landed before the caller may drop t
                    else:
                        self._park.release(name)
                        self._park.park(name, t)
                else:
                    self._park.park(name, t)
                self._held[name] = True
                self.record["held"][name] = {"where": "host", "bytes": nbytes, "shape": list(t.shape), "dtype": str(t.dtype)}
            else:
                self._held[name] = t
                self.record["held"][name] = {"where": str(t.device), "bytes": nbytes, "shape": list(t.shape), "dtype": str(t.dtype)}

    def get(self, name: str):
        if name not in self._held:
            raise CkptError(f"recycle_carry: {name!r} was never held (held: {sorted(self._held) or 'none'})")
        if self._park is not None:
            return self._park.get(name).to_device()                          # H2D on the compute stream: ordered before the next compute
        return self._held[name]

    def names(self) -> list:
        return sorted(self._held)

    def seam(self, cycle: int) -> dict:
        """The recycle boundary: the engine dropped every reference of the cycle; the allocator is released per the ``cache_release``
        policy; the counters before / after are recorded and the record is marked."""
        cuda = _is_cuda_device(self.device)
        before = allocator.counters(self.device) if cuda else None
        released = bool(allocator.release(self.ctx, "stage")) if self.flush else False
        after = allocator.counters(self.device) if cuda else None
        entry = {"cycle": int(cycle), "released": released, "before": before, "after": after, "held": self.names()}
        self.record["seams"].append(entry)
        _mark(self.ctx, "recycle_carry", f"seam cycle={cycle} released={released}")
        return entry

    def release(self) -> None:
        if self._park is not None:
            self._park.close()
            self.record["host_park"] = self._park.record()                 # after close: every park / put / release event, closed=True
        self._held.clear()
        self.record["released"] = True


def _cache_release_policy(ctx: Optional[Ctx]) -> Optional[str]:
    a = _applied(ctx, "cache_release")
    return None if a is None else a.settings.get("policy")


def _recycle_carry_applies(ctx: Ctx) -> Optional[Refusal]:
    lever = "recycle_carry"
    h = ctx.require(lever, "device", "carried")
    carried = h["carried"]
    if isinstance(carried, str) or not isinstance(carried, (list, tuple)) or not carried or not all(isinstance(c, str) for c in carried):
        return refuse(lever, "hooks.carried", f"carried must name the carried tensors (a non-empty sequence of names; got {carried!r})")
    dev = str(h["device"])
    if not (dev == "cpu" or _is_cuda_device(dev)):
        return refuse(lever, "hooks.device", f"device must be 'cpu', 'cuda' or 'cuda:<i>' (got {h['device']!r})")
    park = ctx.setting(lever, "park", None)
    if park is None:
        return refuse(lever, "park", f"no park: set {setting_ref(lever, 'park')} = host (the memory arm) or device (the seam-only diagnostic arm)")
    if park not in PARKS:
        return refuse(lever, "park", f"park must be one of {PARKS} (got {park!r})")
    flush = ctx.setting(lever, "flush", True, cast=parse_bool)
    pin_max_gb = ctx.setting(lever, "pin_max_gb", None, cast=float)
    if park == "host":
        if pin_max_gb is None or isinstance(pin_max_gb, bool) or not float(pin_max_gb) > 0:
            return refuse(lever, "pin_max_gb", f"park=host needs {setting_ref(lever, 'pin_max_gb')} = <GiB > 0> (the pinned budget for the carried planes)")
        if _is_cuda_device(dev) and not _torch().cuda.is_available():
            return refuse(lever, "pin", "pinned host parking needs a CUDA runtime (torch.cuda.is_available() is False)")
        from . import offload  # noqa: PLC0415
        try:
            offload.HostPark(offload.Settings(pin_max_gb=float(pin_max_gb), min_tokens=0, cols="rowloop", device=dev), tag=lever).check()
        except offload.OffloadRefusal as r:
            return refuse(lever, f"host_park.{r.name}", r.reason, **{k: v for k, v in r.details.items() if isinstance(v, (str, int, float, bool))})
    if flush:
        if not _is_cuda_device(dev):
            return refuse(lever, "flush", f"the seam release needs a CUDA device (got {dev!r}); set flush=0")
        policy = _cache_release_policy(ctx)
        if policy != "per_stage":
            return refuse(lever, "flush", "the seam release needs lever cache_release applied with policy per_stage earlier in the line "
                                          f"(found: {policy or 'not applied'}); set {setting_ref('cache_release', 'policy')} = per_stage or flush=0")
    return None


@register("recycle_carry", family="ckpt", exact="bitwise",
          exact_reason="the carried tensors are copied, never recomputed; the seam release frees cached blocks only",
          applies=_recycle_carry_applies,
          description="only the carried state survives a recycle seam; park=host (HostPark, pinned) | device (seam only); release via cache_release",
          preconditions=("hooks.device", "hooks.carried", "park", "pin_max_gb", "pin", "host_park", "flush"), settings=("park", "pin_max_gb", "flush"))
def recycle_carry(ctx: Ctx) -> Applied:
    h = ctx.require("recycle_carry", "device", "carried")
    park = ctx.setting("recycle_carry", "park", None)
    flush = ctx.setting("recycle_carry", "flush", True, cast=parse_bool)
    pin_max_gb = ctx.setting("recycle_carry", "pin_max_gb", None, cast=float)
    carry = RecycleCarry(ctx, h["device"], h["carried"], park=park, flush=flush, pin_max_gb=pin_max_gb)
    ctx.hooks.setdefault("recycle_carry", {})["carry"] = carry             # the handle the adapter's trunk loop uses (hold / get / seam)
    notes = []
    if park == "device":
        notes.append("park=device is the seam-only diagnostic arm: peak counters unchanged on the proof trunk (the seam release moves "
                     "reserved at the seam, never the in-cycle peak); the memory arm is park=host")
    return Applied(lever="recycle_carry",
                   settings={"carried": list(h["carried"]), "park": park, "flush": flush, "pin_max_gb": pin_max_gb if park == "host" else None,
                             "stock": False, "seams": carry.record["seams"]},
                   sites=("hooks.carry",), notes=notes, undo=carry.release)


def carry_of(ctx: Ctx) -> RecycleCarry:
    """The :class:`RecycleCarry` the lever installed (``ctx.hooks["recycle_carry"]["carry"]``); :class:`CkptError` when the lever is
    not applied."""
    c = ctx.hook("recycle_carry", "carry")
    if not isinstance(c, RecycleCarry):
        raise CkptError("recycle_carry is not applied on this ctx (no carry installed)")
    return c


# ------------------------------------------------------------------------------------------------------------ diffusion-sample chunking


def sample_chunks(n_samples: int, k: int) -> list:
    """``[slice(0, k), slice(k, 2k), ...]`` over ``n_samples`` in stock order (the last chunk may be short)."""
    n, k = int(n_samples), int(k)
    if n < 1 or k < 1:
        raise CkptError(f"samples_per_pass: n_samples and k must be >= 1 (got {n}, {k})")
    return [slice(i, min(i + k, n)) for i in range(0, n, k)]


class StockOrderDraws:
    """The stock draw order across chunks (module contract). :meth:`rewind` restores the RNG state captured at construction and opens
    a chunk's census entry; :meth:`draw` runs the engine's stock-shaped draw and returns the chunk's slice of it along ``sample_dim``
    (a view: the full draw lives as long as the view), recording the draw's shape / dtype in the chunk's census entry; a draw before
    :meth:`rewind` / :meth:`open_chunk` is :class:`CkptError`. :meth:`close_chunk` captures the chunk's end state; :meth:`end_states_equal`
    is the end-state gate (every chunk's end state equals the first's); :meth:`census_consistent` the draw census (every chunk drew
    the same sequence of shapes and dtypes)."""

    def __init__(self, generator=None, device=None, sample_dim: int = 0):
        torch = _torch()
        self.generator = generator
        self.device = device if device is not None else (generator.device if generator is not None else torch.device("cpu"))
        self.sample_dim = int(sample_dim)
        self._state = self._get_state()
        self.census: list = []
        self.end_states: list = []
        self._current: Optional[list] = None

    def _get_state(self):
        torch = _torch()
        if self.generator is not None:
            return self.generator.get_state()
        if str(self.device).startswith("cuda"):
            return torch.cuda.get_rng_state(self.device)
        return torch.get_rng_state()

    def _set_state(self, state) -> None:
        torch = _torch()
        if self.generator is not None:
            self.generator.set_state(state)
        elif str(self.device).startswith("cuda"):
            torch.cuda.set_rng_state(state, self.device)
        else:
            torch.set_rng_state(state)

    def rewind(self) -> None:
        self._set_state(self._state)
        self.open_chunk()

    def open_chunk(self) -> None:
        self._current = []
        self.census.append(self._current)

    def close_chunk(self) -> None:
        self.end_states.append(self._get_state())

    def draw(self, fn: Callable[[], Any], chunk: slice):
        if self._current is None:
            raise CkptError("samples_per_pass: draw before rewind(): a chunk's draws start with rewind() (stock order) or open_chunk()")
        full = fn()
        self._current.append((tuple(full.shape), str(full.dtype)))
        return full.narrow(self.sample_dim, chunk.start, chunk.stop - chunk.start)

    def census_consistent(self) -> bool:
        return all(c == self.census[0] for c in self.census[1:]) if self.census else True

    def end_states_equal(self) -> bool:
        torch = _torch()
        return all(torch.equal(s, self.end_states[0]) for s in self.end_states[1:]) if self.end_states else True


def _cat_outputs(outs: Sequence, dim: int):
    torch = _torch()
    first = outs[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(list(outs), dim=dim)
    if isinstance(first, Mapping):
        keys = list(first)
        for o in outs[1:]:
            if list(o) != keys:
                raise CkptError(f"samples_per_pass: chunk outputs differ in keys: {keys} vs {list(o)}")
        return {k: _cat_outputs([o[k] for o in outs], dim) for k in keys}
    if isinstance(first, (tuple, list)):
        n = len(first)
        for o in outs[1:]:
            if len(o) != n:
                raise CkptError(f"samples_per_pass: chunk outputs differ in length: {n} vs {len(o)}")
        return type(first)(_cat_outputs([o[i] for o in outs], dim) for i in range(n))
    raise CkptError(f"samples_per_pass: cannot concatenate chunk outputs of type {type(first).__name__}")


def run_chunks(n_samples: int, k: int, run: Callable[[slice, StockOrderDraws], Any], *, generator=None, device=None,
                draw_order: str = "stock", sample_dim: int = 0, record: Optional[dict] = None):
    """The chunk loop: ``run(chunk, draws)`` per chunk in stock order (rewound under ``stock``, both gates checked after every chunk);
    outputs concatenated along ``sample_dim``; ``record`` receives ``chunks``, ``draw_order``, ``census`` (None under ``chunked``)."""
    if draw_order not in DRAW_ORDERS:
        raise CkptError(f"samples_per_pass: draw_order must be one of {DRAW_ORDERS} (got {draw_order!r})")
    chunks = sample_chunks(n_samples, k)
    draws = StockOrderDraws(generator=generator, device=device, sample_dim=sample_dim)
    outs = []
    for chunk in chunks:
        if draw_order == "stock":
            draws.rewind()
        else:
            draws.open_chunk()
        outs.append(run(chunk, draws))
        draws.close_chunk()
        if draw_order == "stock":
            if not draws.census_consistent():
                raise CkptError(f"samples_per_pass: draw census differs at chunk {chunk}: {draws.census[-1]} vs {draws.census[0]} — "
                                "every chunk must make the same stock-shaped draws in the same order")
            if not draws.end_states_equal():
                raise CkptError(f"samples_per_pass: RNG end state differs at chunk {chunk}: a draw was made outside draws.draw (the stream "
                                "diverged from stock's) — route every batch-shaped draw through draws.draw")
    if record is not None:
        census = ({"draws_per_chunk": [len(c) for c in draws.census], "consistent": draws.census_consistent(),
                   "end_states_equal": draws.end_states_equal()} if draw_order == "stock" else None)
        record.update({"chunks": [[c.start, c.stop] for c in chunks], "draw_order": draw_order, "census": census})
    return _cat_outputs(outs, sample_dim)



_run_chunks = run_chunks                                                   # private alias (same object)

def run_sample_chunks(ctx: Ctx, run: Optional[Callable[[slice, StockOrderDraws], Any]] = None, *, n_samples: Optional[int] = None,
                      generator=None, device=None, sample_dim: Optional[int] = None):
    """The adapter's call at sampling time: the diffusion head on ``k`` samples per pass with the applied lever's ``k`` and
    ``draw_order`` (the lever not applied = the stock pass of every sample, nothing recorded); ``run``, ``n_samples``, ``generator``,
    ``device``, ``sample_dim`` default to the lever's hooks. Every pass marks the record; the pass record lands on the Applied."""
    a = _applied(ctx, "samples_per_pass")
    run = ctx.hook("samples_per_pass", "run") if run is None else run
    n = ctx.hook("samples_per_pass", "n_samples") if n_samples is None else n_samples
    generator = ctx.hook("samples_per_pass", "generator") if generator is None else generator
    device = ctx.hook("samples_per_pass", "device") if device is None else device
    sample_dim = ctx.hook("samples_per_pass", "sample_dim", 0) if sample_dim is None else sample_dim
    if run is None or n is None:
        raise CkptError("samples_per_pass: run_sample_chunks needs run and n_samples (hooks or arguments)")
    if a is None:
        return run_chunks(int(n), int(n), run, generator=generator, device=device, draw_order="stock", sample_dim=int(sample_dim))
    rec: dict = {}
    k, order = int(a.settings["k"]), str(a.settings["draw_order"])
    out = run_chunks(int(n), k, run, generator=generator, device=device, draw_order=order, sample_dim=int(sample_dim), record=rec)
    a.settings["last_pass"] = rec
    _mark(ctx, "samples_per_pass", f"n={n} k={k} chunks={len(rec['chunks'])} draw_order={order}")
    return out


def _samples_per_pass_applies(ctx: Ctx) -> Optional[Refusal]:
    lever = "samples_per_pass"
    h = ctx.require(lever, "n_samples", "run")
    n = h["n_samples"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return refuse(lever, "hooks.n_samples", f"n_samples must be an int >= 1 (got {n!r})")
    if not callable(h["run"]):
        return refuse(lever, "hooks.run", "run must be a callable run(chunk, draws) — the engine's sampler over one chunk of samples")
    k = ctx.setting(lever, "k", None, cast=int)
    if k is None:
        return refuse(lever, "k", f"no k: set {setting_ref(lever, 'k')} = <samples per pass>")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        return refuse(lever, "k", f"k must be an int >= 1 (got {k!r})")
    if k >= n:
        return refuse(lever, "k", f"k={k} >= N={n} is the stock pass (one chunk of every sample): nothing to apply — turn the lever off by "
                                  f"name ({off_ref(lever)}) or set k < N")
    order = ctx.setting(lever, "draw_order", "stock")
    if order not in DRAW_ORDERS:
        return refuse(lever, "draw_order", f"draw_order must be one of {DRAW_ORDERS} (got {order!r})")
    sample_dim = ctx.hook(lever, "sample_dim", 0)
    if isinstance(sample_dim, bool) or not isinstance(sample_dim, int):
        return refuse(lever, "hooks.sample_dim", f"sample_dim must be an int (got {sample_dim!r})")
    return None


@register("samples_per_pass", family="chunk", exact="measured",
          exact_reason="draw_order=stock: the draws are stock's by construction (RNG rewound per chunk, stock-shaped draws sliced, census + end-state "
                       "gates; the identity record holds them torch.equal) and the denoiser's kernel selection on the smaller batch is the engine's; "
                       "draw_order=chunked: chunk-shaped draws re-order the stock RNG stream (band-class by the D82 seed-spread word, B2) — the "
                       "engine's identity row narrows with its id",
          applies=_samples_per_pass_applies, description="the diffusion head on k < N samples at a time; stock draw order kept (or chunked, named)",
          preconditions=("hooks.n_samples", "hooks.run", "k", "draw_order", "hooks.sample_dim"), settings=("k", "draw_order"))
def samples_per_pass(ctx: Ctx) -> Applied:
    n = int(ctx.hook("samples_per_pass", "n_samples"))
    k = int(ctx.setting("samples_per_pass", "k", None, cast=int))
    order = ctx.setting("samples_per_pass", "draw_order", "stock")
    if order == "stock":
        why = ("draw_order=stock: the draws are stock's by construction (RNG rewound per chunk, stock-shaped draws sliced, census + end-state gates); "
               "the denoiser's kernel selection on the smaller batch is the engine's — its identity row narrows (bitwise where torch.equal holds, "
               "band by the D82 word) with its id")
    else:
        why = "draw_order=chunked: chunk-shaped draws re-order the stock RNG stream — band-class by the D82 seed-spread word (B2); the identity row writes it with its id"
    notes = [f"identity record {IDENTITY_RECORD}: draws torch.equal and the RNG end state equal to stock's at every size and k; outputs within "
             "4.8e-07 on a batched-einsum denoiser"]
    return Applied(lever="samples_per_pass",
                   settings={"n_samples": n, "k": k, "draw_order": order, "sample_dim": int(ctx.hook("samples_per_pass", "sample_dim", 0)), "stock": False},
                   sites=("hooks.run",), exact_reason=why, notes=notes)


# ------------------------------------------------------------------------------------------------------------ seed batching


def seed_batches(seeds: Sequence[int], k: int) -> list:
    """``[[s1..sk], [sk+1..], ...]`` in the stock seed order (``k=1`` is the stock loop)."""
    seeds = list(seeds)
    k = int(k)
    if k < 1:
        raise CkptError(f"seed_batch: k must be >= 1 (got {k})")
    return [seeds[i:i + k] for i in range(0, len(seeds), k)]


def seed_batch_plan(ctx: Ctx, seeds: Optional[Sequence[int]] = None) -> list:
    """The adapter's call per item, inside the item's unit: the seed batches of the applied lever (``k`` from the record; the lever not
    applied = one seed per batch, nothing recorded). Marks the record (unit scope)."""
    seeds = list(ctx.hook("seed_batch", "seeds") if seeds is None else seeds)
    a = _applied(ctx, "seed_batch")
    if a is None:
        return seed_batches(seeds, 1)
    plan = seed_batches(seeds, int(a.settings["k"]))
    _mark(ctx, "seed_batch", f"seeds={len(seeds)} k={a.settings['k']} batches={len(plan)}")
    return plan


def _seed_batch_applies(ctx: Ctx) -> Optional[Refusal]:
    lever = "seed_batch"
    h = ctx.require(lever, "seeds", "capable")
    seeds = h["seeds"]
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, (list, tuple)) or not seeds:
        return refuse(lever, "hooks.seeds", f"seeds must be a non-empty sequence (got {seeds!r})")
    k = ctx.setting(lever, "k", None, cast=int)
    if k is None:
        return refuse(lever, "k", f"no k: set {setting_ref(lever, 'k')} = <seeds per pass >= 2> (never a default; k=1 is the stock loop)")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        return refuse(lever, "k", f"k must be an int >= 2 (got {k!r})")
    if k == 1:
        return refuse(lever, "k", f"k=1 is the stock loop (one seed per pass): nothing to apply — turn the lever off by name "
                                  f"({off_ref(lever)}) or set k >= 2")
    if not h["capable"]:
        return refuse(lever, "hooks.capable", "the engine's runner does not batch seeds on independent RNG streams (capable is False): "
                                              "k >= 2 needs a batched runner")
    return None


@register("seed_batch", family="ckpt", exact="measured",
          exact_reason="K seeds in one batch, each on its own RNG stream: every seed's draws are its own by construction — bitwise-claimed until the "
                       "engine's identity row narrows it with its id (the batched kernels may select differently from the per-seed ones); a speed "
                       "lever with a memory cost (k× the batched stage's per-seed working set)",
          applies=_seed_batch_applies, description="k >= 2 seeds per pass in stock order (k=1 is the stock loop and refuses); never a default",
          preconditions=("hooks.seeds", "hooks.capable", "k"), settings=("k",))
def seed_batch(ctx: Ctx) -> Applied:
    seeds = list(ctx.hook("seed_batch", "seeds"))
    k = int(ctx.setting("seed_batch", "k", None, cast=int))
    return Applied(lever="seed_batch", settings={"k": k, "n_seeds": len(seeds), "batches": seed_batches(seeds, k), "stock": False},
                   sites=("hooks.seeds",))


# ------------------------------------------------------------------------------------------------------------ graph capture policy


@dataclass
class GraphPolicy:
    """The typed capture policy (module contract): ``policy`` ``on`` (every shape captured — the stock policy, what ``undo`` installs),
    ``off`` (nothing captured), ``budget`` (a fold above ``budget`` tokens runs eagerly; ``budget`` ≥ 1). :meth:`admit` returns True
    when ``site`` may capture a graph for a fold of ``tokens`` and False when it must run eagerly; every refusal is counted in
    ``events[site]``, the first refusal per (site, tokens) calls ``on_event(site, tokens)`` — the engine prints its own line there —
    and every decision marks the record."""

    policy: str = "on"
    budget: Optional[int] = None
    on_event: Optional[Callable[[str, int], None]] = None
    ctx: Optional[Ctx] = None
    events: dict = field(default_factory=dict)
    _seen: set = field(default_factory=set)

    def __post_init__(self):
        if self.policy not in ("on",) + GRAPH_POLICIES:
            raise CkptError(f"graph_capture: policy must be one of {('on',) + GRAPH_POLICIES} (got {self.policy!r})")
        if self.policy == "budget" and (isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 1):
            raise CkptError(f"graph_capture: policy budget needs budget = an int >= 1 (tokens; got {self.budget!r})")

    @property
    def cap(self) -> Optional[int]:
        """The policy as :mod:`opt_core.mem.graph_gate`'s token cap: ``on`` -> None (no cap), ``budget`` -> the budget, ``off`` -> 0 (never)."""
        return None if self.policy == "on" else (int(self.budget) if self.policy == "budget" else 0)

    def decision(self, site: str, tokens: int) -> "graph_gate.GateDecision":
        """The ONE capture decision (:func:`opt_core.mem.graph_gate.decide`) for ``site`` at ``tokens``; the kit's capture layer prints its
        ``fragment`` / ``gate=<reason>`` words as-is. :meth:`admit` is this decision's ``capture`` plus the census event."""
        return graph_gate.decide(int(tokens), self.cap, name="graph")

    def admit(self, site: str, tokens: int) -> bool:
        ok = self.decision(site, tokens).capture
        if not ok:
            self.events[site] = self.events.get(site, 0) + 1
            if (site, int(tokens)) not in self._seen:
                self._seen.add((site, int(tokens)))
                if self.on_event is not None:
                    self.on_event(site, int(tokens))
        _mark(self.ctx, "graph_capture", f"{site}:{tokens} {'captured' if ok else 'eager'}")
        return ok

    def describe(self) -> str:
        return self.policy if self.policy != "budget" else f"budget:{self.budget}"


def _graph_capture_applies(ctx: Ctx) -> Optional[Refusal]:
    lever = "graph_capture"
    h = ctx.require(lever, "switch")
    if not callable(h["switch"]):
        return refuse(lever, "hooks.switch", "switch must be a callable switch(policy) that maps the GraphPolicy onto the kit's capture plumbing")
    policy = ctx.setting(lever, "policy", "off")
    if policy not in GRAPH_POLICIES:
        return refuse(lever, "policy", f"policy must be one of {GRAPH_POLICIES} (got {policy!r}); the stock policy is the lever off by name: "
                                       f"{off_ref(lever)}")
    budget = ctx.setting(lever, "budget", None, cast=int)
    if policy == "budget" and (budget is None or isinstance(budget, bool) or budget < 1):
        return refuse(lever, "budget", f"policy budget needs {setting_ref(lever, 'budget')} = <tokens >= 1> (got {budget!r})")
    on_event = ctx.hook(lever, "on_event")
    if on_event is not None and not callable(on_event):
        return refuse(lever, "hooks.on_event", "on_event must be a callable on_event(site, tokens) or absent")
    return None


@register("graph_capture", family="setting", exact="measured",
          exact_reason="bitwise-claimed (the eager path launches the captured path's kernels) until the engine's identity row narrows it with its id",
          applies=_graph_capture_applies, description="CUDA-graph capture policy off | budget:<tokens>; refusals counted per site; typed, mapped by the adapter",
          preconditions=("hooks.switch", "policy", "budget", "hooks.on_event"), settings=("policy", "budget"), scope="process")
def graph_capture(ctx: Ctx) -> Applied:
    policy = ctx.setting("graph_capture", "policy", "off")
    budget = ctx.setting("graph_capture", "budget", None, cast=int)
    pol = GraphPolicy(policy=policy, budget=budget if policy == "budget" else None, on_event=ctx.hook("graph_capture", "on_event"), ctx=ctx)
    switch = ctx.hook("graph_capture", "switch")
    switch(pol)
    ctx.hooks.setdefault("graph_capture", {})["policy_object"] = pol

    def undo():
        switch(GraphPolicy(policy="on", ctx=ctx))

    return Applied(lever="graph_capture", settings={"policy": policy, "budget": pol.budget, "stock": False, "graphs": False},
                   sites=("hooks.switch",), undo=undo)


# ------------------------------------------------------------------------------------------------------------ hoist cache


def _hoist_off_applies(ctx: Ctx) -> Optional[Refusal]:
    h = ctx.require("hoist_off", "disable")
    if not callable(h["disable"]):
        return refuse("hoist_off", "hooks.disable", "disable must be a callable disable() that turns the sampler's hoist cache off and returns the "
                                                    "restore callable (or None)")
    return None


@register("hoist_off", family="setting", exact="measured",
          exact_reason="bitwise-claimed (a memoization removed: the step-invariant values recomputed per step by the same kernels on the same bytes) "
                       "until the engine's identity row narrows it with its id",
          applies=_hoist_off_applies, description="the sampler's step-invariant (hoist) cache off through the adapter's disable() (an opt_core.capture.hoist.ConstMemo: its disable(reason))",
          preconditions=("hooks.disable",), settings=(), scope="process")
def hoist_off(ctx: Ctx) -> Applied:
    restore = ctx.hook("hoist_off", "disable")()
    return Applied(lever="hoist_off", settings={"hoist": "off", "stock": False}, sites=("hooks.disable",),
                   undo=restore if callable(restore) else None, notes=["hoist cache disabled"])      # process scope: the record marks it at apply
