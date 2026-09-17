"""Lever denoiser_graph (class fast) — the denoiser network issued as ONE CUDA graph per call instead of ~600 kernel launches.

Stock: DiffusionHead.sample (diffusion_head.py L283-373) runs config.num_steps (200) denoising steps; every step calls, once per sample chunk
(inference_step L431-443; SamplingConfig.chunk_size = 5), ``self.score_model(batch, r_noisy, single_cond, pair_bias)`` = DiffusionModule.forward
(L122-181: atom encoder (fp32 island) -> 12-block diffusion transformer -> atom decoder (fp32 island)). Within one sample() call the batch dict's
tensors and pair_bias are the same tensors at every step and the shapes are static by construction; only r_noisy [B, n, L, 14, 3] and single_cond
[B, 1, L, c_s] are new tensors each step. RNG (noise draws, random augmentation), the EDM pre/post-conditioning, single_conditioning and the chunk
loop are outside that call.

Here DiffusionModule.forward is captured into a torch.cuda.CUDAGraph at the first call of a roll-out — one warm-up call on a side stream, then the
capture on that stream, both under ``torch.autocast('cuda', <the current autocast dtype/state>, cache_enabled=False)`` so every weight cast is a
kernel inside the graph (no cached-cast address is baked in) — and REPLAYED at every call of the roll-out whose key matches, the capturing call
included (its own result is the first replay): r_noisy / single_cond are copy_()'d into private input buffers of the identical size/stride, the
batch dict and pair_bias are read by reference, the output is cloned out of the graph's memory pool. The identical kernels run on the identical
buffers; the launches become one. Graph lifetime = one roll-out: DiffusionHead.sample is wrapped so that its ``finally:`` drops every graph, pool
buffer and held reference of its score_model (the wrapper composes over whatever DiffusionHead.sample is at install time — in the fast row that is
diffusion_bf16's autocast wrapper, which therefore stays outermost).

Key of a call = the structure of its arguments with, for EVERY tensor reachable from them (the batch dict incl. nested containers and relpos_lazy's
LazyFeat entries with their producer closures' tensors; pair_bias), its (data_ptr, size, stride, dtype, device); for r_noisy / single_cond (copied,
not referenced) their (size, stride, dtype, device); plus the autocast state and the grad mode. The roll-out HOLDS a reference to every keyed
object for the graph's lifetime, so a keyed address is never freed and re-used under the same key. A key not yet seen in the roll-out captures one
more graph into a small dict (ragged sample chunks: num_samples 7 at chunk_size 5 -> chunk shapes 5 and 2 -> two graphs per roll-out).

Fallbacks (the eager stock statement runs; every call is counted exactly once, served or by reason):
  ``above_gate``               tokens L (= r_noisy.shape[2]) > AFO_DENOISER_GRAPH_MAX_TOKENS: every call of that roll-out, by design — decided before
                               anything is allocated (the sampler is GPU-bound there: launches are not the cost, and the pool would only add
                               memory). Expected.
  ``cpu``                      r_noisy is not a CUDA tensor. Expected.
  ``outside_sample``           DiffusionModule.forward called with no DiffusionHead.sample roll-out around it (no lifetime scope -> no graph).
  ``capturing``                the current stream is already being captured by someone else: the stock statement runs inline (its kernels join
                               that capture).
  ``warmup_failed:<Exc>``      the warm-up call (eager, on the side stream) raised <Exc> (device state intact): eager for the rest of the roll-out,
                               counted (eager=, fallback_by=warmup_failed:<Exc>:n); the gate is refused at exit (exit 3, outputs complete).
  ``capture_failed:<Exc>``     the CAPTURE itself raised <Exc>: the call is counted under this word and the exception is re-raised with the lever named: torch (2.7)
                               keeps the CUDA RNG generator and the caching allocator in capture state after a failed stream capture, so the
                               process cannot draw noise, empty its cache or capture again — the item ends there, loudly, instead of limping on.
  ``recapture_storm``          a roll-out asked for more captures than distinct call shapes + 1 (an argument the key pins changes address between
                               steps) or for more than MAX_GRAPHS graphs: eager for the rest of the roll-out.
The last four are NOT expected: any of them refuses the lever's gate (fail-loud). LEVER-line facts: ``captures`` (graphs captured = roll-outs at or
below the gate x distinct chunk shapes), ``replays`` (calls answered by a graph replay, the capturing calls included = ``served``), ``eager`` (calls
that ran the stock statement, for any reason = ``fallback``; 0 at or below the gate), ``recaptures`` (captures of a shape already captured in the
same roll-out; 0), ``rollouts`` (DiffusionHead.sample calls), ``max_tokens`` (the gate literal), ``capture_error`` (only after a failed capture).

Census of OTHER levers with this lever on: a lever whose Python body runs inside DiffusionModule.forward counts only the calls that execute eagerly
— per roll-out the warm-up call and the capture call of each graph, plus any eager fallback calls — while its kernels run at every replay. In the
fast row these are the levers patched inside the atom encoder / decoder, the diffusion transformer's attention and the LayerNorms of the
denoiser (a site in each of the 12 DiT blocks counts served = 12 x (2 x captures + eager calls) per roll-out instead of 12 x steps x chunks);
relpos_lazy's atom_rel_pos producer is replayed inside the graph but counts once per predict in
compute_rel_pos_encoding (unchanged); pair_transition_chunk's Transition sites (PairConditioning: once per sample(); SingleConditioning: per step)
and diffusion_bf16 (around sample()) are outside the captured call (unchanged). The sampler's RNG (noise draws, random augmentation) is outside
the captured call and stays eager.

Memory: a graph's private pool keeps the peak transient memory of ONE denoiser call reserved for the whole roll-out (eagerly those transients come
and go inside blocks the trunk left cached), so device memory in use during the sampler grows by about that working set (growing ~L^2 with the
token count) while torch's peak-allocated counter barely moves; torch.cuda.graph also empties the allocator cache before each
capture. Both end with the roll-out.

AFO_DENOISER_GRAPH_MAX_TOKENS (default 1024) is the one knob: the largest token count (bucketed length) at which the graph is used.

Keeping graphs beyond one roll-out is NOT this lever's business: ``RollOut`` has no-op hook points (``adopt`` / ``before_capture`` /
``after_capture`` / ``owns``) and the sample wrapper builds its roll-out through ``ROLLOUT_FACTORY`` (None = RollOut);
hooks/graph_reuse installs a subclass there that keeps the last item's graphs for the next item of the same structure.  Without that
lever nothing here changes: one roll-out = one graph lifetime."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from opt_core.counters import Ledger

from . import Installed, LeverAborted, rebind, size_gated
from ..registry import LEVERS
from .relpos import LazyFeat

NAME = "LOCAL.atlasfold.denoiser_graph"
TARGET = "atlasfold.model.network.diffusion_head"
STATE_ATTR = "_afo_graph"                                           # the per-instance roll-out state on the DiffusionModule (None outside sample())
MAX_GRAPHS = 4                                                      # graphs one roll-out may hold (distinct chunk shapes: at most 2 with the stock chunk loop)
EXPECTED = tuple(LEVERS["denoiser_graph"]["expected"])              # ("above_gate", "cpu") — one vocabulary, owned by the registry row

# fallback vocabulary (the registry row's `expected` names the first two)
ABOVE_GATE, CPU, OUTSIDE_SAMPLE, CAPTURING, WARMUP_FAILED, CAPTURE_FAILED, RECAPTURE_STORM = (
    "above_gate", "cpu", "outside_sample", "capturing", "warmup_failed", "capture_failed", "recapture_storm")


def max_tokens() -> int:
    """AFO_DENOISER_GRAPH_MAX_TOKENS (default 1024), read when the lever installs."""
    return int(os.environ.get("AFO_DENOISER_GRAPH_MAX_TOKENS", "1024"))


def gate_word(r_noisy: torch.Tensor, limit: int) -> Optional[str]:
    """The by-design fallback of a call, or None when the graph route applies: ``cpu`` (no CUDA tensor) before ``above_gate`` (L > limit)."""
    if r_noisy.device.type != "cuda":
        return CPU
    if int(r_noisy.shape[2]) > int(limit):
        return ABOVE_GATE
    return None


def autocast_state(device_type: str = "cuda") -> Tuple[bool, torch.dtype]:
    """(enabled, dtype) of autocast for ``device_type`` in this thread."""
    try:
        return bool(torch.is_autocast_enabled(device_type)), torch.get_autocast_dtype(device_type)
    except TypeError:                                               # older torch signatures
        if device_type == "cuda":
            return bool(torch.is_autocast_enabled()), torch.get_autocast_gpu_dtype()
        return bool(torch.is_autocast_cpu_enabled()), torch.get_autocast_cpu_dtype()


def _tensor_sig(t: torch.Tensor, by_reference: bool) -> tuple:
    lay = (tuple(t.size()), tuple(t.stride()), t.dtype, str(t.device))
    return (("T", t.data_ptr()) + lay) if by_reference else (("S",) + lay)


def flatten_key(obj: Any, refs: List[Any], _depth: int = 0) -> tuple:
    """The key of everything reachable from ``obj`` — tensors as (data_ptr, size, stride, dtype, device); dicts (sorted by key), lists, tuples
    and LazyFeat producers (the tensors their closure holds) by structure; scalars by value; anything else by type and id() — appending
    every keyed tensor / object to ``refs`` (the holder keeps them alive for the key's lifetime)."""
    if isinstance(obj, torch.Tensor):
        refs.append(obj)
        return _tensor_sig(obj, True)
    if _depth > 8:
        refs.append(obj)
        return ("O", type(obj).__name__, id(obj))
    if isinstance(obj, dict):
        keys = sorted(obj.keys(), key=repr)
        return ("D", tuple((repr(k), flatten_key(obj[k], refs, _depth + 1)) for k in keys))
    if isinstance(obj, (list, tuple)):
        return ("L" if isinstance(obj, list) else "U", tuple(flatten_key(v, refs, _depth + 1) for v in obj))
    if isinstance(obj, LazyFeat):                                   # relpos_lazy: the producer's inputs live in fn's closure (or partial args); its autocast replay state in .ac
        refs.append(obj)
        fn = obj.fn
        held = []
        for c in tuple(getattr(fn, "__closure__", None) or ()):
            try:
                held.append(c.cell_contents)
            except ValueError:                                      # an empty cell
                continue
        held += list(getattr(fn, "args", None) or ()) + list((getattr(fn, "keywords", None) or {}).values())   # functools.partial producers
        inner = tuple(flatten_key(v, refs, _depth + 1) for v in held if isinstance(v, (torch.Tensor, dict, list, tuple, LazyFeat)))
        return ("F", id(obj), repr(getattr(obj, "ac", None)), inner)
    if obj is None or isinstance(obj, (bool, int, float, str, torch.dtype, torch.device)):
        return ("V", repr(obj))
    refs.append(obj)
    return ("O", type(obj).__name__, id(obj))


def call_key(batch: Any, r_noisy: torch.Tensor, single_cond: torch.Tensor, pair_bias: Any, refs: List[Any]) -> tuple:
    """The replay key of one DiffusionModule.forward call (module docstring); ``refs`` receives every object the key pins by address."""
    ac = autocast_state(r_noisy.device.type if r_noisy.device.type in ("cuda", "cpu") else "cuda")
    return (flatten_key(batch, refs), _tensor_sig(r_noisy, False), _tensor_sig(single_cond, False), flatten_key(pair_bias, refs),
            ("A",) + tuple(repr(x) for x in ac), ("G", torch.is_grad_enabled(), torch.is_inference_mode_enabled()))


def static_copy(t: torch.Tensor) -> torch.Tensor:
    """A private buffer holding t's data at t's exact size/stride (the captured kernels read the layout the eager call reads); an overlapping
    (stride-0 broadcast) input — none on the sampler's route: r_noisy and single_cond are fresh dense tensors — gets a dense copy instead (same
    values; the layout the graph reads then differs from the eager call's)."""
    if any(st == 0 and sz > 1 for st, sz in zip(t.stride(), t.size())):
        return t.clone()
    out = torch.empty_strided(tuple(t.size()), tuple(t.stride()), dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out


def _fact_text(msg: str, n: int = 96) -> str:
    """A one-token rendering of a message for a LEVER-line fact (no spaces, bounded)."""
    return "_".join(str(msg).split())[:n]


class CaptureAborted(RuntimeError):
    """The capture itself (not the warm-up call) raised ``original``: the process's CUDA capture state is not restorable (module docstring)."""
    def __init__(self, original: BaseException):
        super().__init__(f"{type(original).__name__}: {original}")
        self.original = original


class CudaGraphs:
    """The torch.cuda graph calls of this lever, in one place: warm-up on the lever's side stream, capture on that stream, both under the caller's
    autocast state with the cast cache off; replay on the current stream. One side stream per device for the process (a new stream per roll-out
    would leave one cuBLAS workspace per pool stream behind)."""
    impl = "torch.cuda.graphs"
    _side: Dict[int, Any] = {}

    @classmethod
    def side_stream(cls, device) -> "torch.cuda.Stream":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if idx not in cls._side:
            cls._side[idx] = torch.cuda.Stream(device=idx)
        return cls._side[idx]

    @staticmethod
    def is_capturing() -> bool:
        return bool(torch.cuda.is_current_stream_capturing())

    @classmethod
    def warmup(cls, fn: Callable[[], torch.Tensor], device) -> torch.Tensor:
        """-> ``fn()`` run once eagerly on the side stream under the caller's autocast state with the cast cache off — the warm-up call of
        ``capture`` on its own (the current stream waits for it before returning); an exception is an ordinary eager exception."""
        enabled, dtype = autocast_state("cuda")
        side = cls.side_stream(device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.autocast("cuda", dtype=dtype, enabled=enabled, cache_enabled=False):
            with torch.cuda.stream(side):
                out = fn()
            torch.cuda.current_stream().wait_stream(side)
        return out

    @classmethod
    def capture(cls, fn: Callable[[], torch.Tensor], device):
        """-> (graph, static_output): ``fn`` run once eagerly on the side stream (allocator blocks, cuBLAS workspaces, lazy inits happen outside
        the capture; an exception here is an ordinary eager exception and propagates as raised), then captured on it (an exception here is
        re-raised as CaptureAborted carrying the ORIGINAL error, not capture_end's follow-up); ``graph.replay()`` recomputes ``static_output``
        in place from whatever the static inputs hold."""
        enabled, dtype = autocast_state("cuda")
        side = cls.side_stream(device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.autocast("cuda", dtype=dtype, enabled=enabled, cache_enabled=False):
            with torch.cuda.stream(side):
                fn()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            first: List[BaseException] = []
            try:
                with torch.cuda.graph(graph, stream=side, capture_error_mode="thread_local"):
                    try:
                        out = fn()
                    except Exception as e:  # noqa: BLE001 — remembered: torch.cuda.graph's exit raises its own error over it
                        first.append(e)
                        raise
            except Exception as e:  # noqa: BLE001
                raise CaptureAborted(first[0] if first else e) from e
        torch.cuda.current_stream().wait_stream(side)
        return graph, out


def probe() -> dict:
    """The route ``check`` prints as ``KERNEL cuda_graph``: can this device warm up, CAPTURE and replay a CUDA graph through the lever's own backend
    (CudaGraphs.capture: side stream, the caller's autocast state, thread-local capture errors)?  A [5, 16, 32] bf16-autocast Linear + SiLU is
    captured once, its static input overwritten, the graph replayed, and the replayed output compared with the eager statement on the same input
    (equal to the bit: the identical kernels on the identical buffers).  {ok True | False + reason}; ok None without a CUDA device (nothing to
    capture on; the check run on the GPU host decides).  A device that cannot capture says so here, before any item runs."""
    if not torch.cuda.is_available():
        return {"kernel": "cuda_graph", "ok": None, "routed": None, "resolved": None, "reason": "no_cuda"}
    resolved = f"{CudaGraphs.impl}@{torch.__version__.split('+')[0]}"
    try:
        dev = torch.device("cuda", torch.cuda.current_device())
        lin = torch.nn.Linear(32, 32).to(dev)
        x = torch.randn(5, 16, 32, device=dev)
        fn = lambda: torch.nn.functional.silu(lin(x))                      # noqa: E731
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            graph, static_out = CudaGraphs.capture(fn, dev)
            x.copy_(torch.randn(5, 16, 32, device=dev))                    # new input values in the captured buffer
            graph.replay(); replayed = static_out.clone()
            eager = fn()
        torch.cuda.synchronize()
        del graph
        if replayed.shape != eager.shape or not bool(torch.equal(replayed, eager)):
            return {"kernel": "cuda_graph", "ok": False, "routed": False, "resolved": resolved, "reason": "replay != eager on the probe statement"}
        return {"kernel": "cuda_graph", "ok": True, "routed": True, "resolved": resolved, "reason": None}
    except CaptureAborted as e:
        return {"kernel": "cuda_graph", "ok": False, "routed": False, "resolved": resolved, "reason": f"{CAPTURE_FAILED}:{type(e.original).__name__}: {str(e.original)[:160]}"}
    except Exception as e:  # noqa: BLE001
        return {"kernel": "cuda_graph", "ok": False, "routed": False, "resolved": resolved, "reason": f"{type(e).__name__}: {str(e)[:160]}"}


class _Entry:
    """One captured graph: the graph, its static r_noisy / single_cond input buffers and static output, the objects its key pins (held), the
    call shape; ``owner`` / ``table`` are None unless a keeper (hooks/graph_reuse) took the entry over at capture time (``owner`` = the keeper's
    set the entry belongs to, ``table`` = the keeper's record of every tensor the captured kernels read by reference)."""
    __slots__ = ("graph", "r", "sc", "out", "refs", "shape", "owner", "table")

    def __init__(self, graph, r, sc, out, refs, shape):
        self.graph, self.r, self.sc, self.out, self.refs, self.shape = graph, r, sc, out, refs, shape
        self.owner = None
        self.table = None

    def release(self) -> None:
        """Drop the graph (its private pool becomes freeable at once: CUDAGraph.reset), the static buffers and every held reference."""
        g = self.graph
        self.refs = None; self.out = None; self.r = None; self.sc = None; self.graph = None; self.table = None; self.owner = None
        if g is not None and hasattr(g, "reset"):
            try:
                g.reset()
            except Exception:  # noqa: BLE001 — a graph torch already tore down
                pass


ROLLOUT_FACTORY = None                                              # None: RollOut below (one roll-out = one graph lifetime); hooks/graph_reuse installs its keeper's subclass here
INSTALLED: Dict[str, Any] = {}                                      # {ledger, limit} once install() applied the lever (hooks/graph_reuse reads it)


class RollOut:
    """The graphs of ONE DiffusionHead.sample call for one DiffusionModule: key -> (graph, static r_noisy / single_cond / output, held refs).
    ``call`` is the forward statement (capture-or-replay, or a named eager fallback); ``close`` drops everything.  Three hook points with
    no-op defaults let a subclass keep graphs beyond the roll-out (hooks/graph_reuse): ``adopt`` (answer a key miss with a graph captured
    earlier instead of a capture), ``before_capture`` / ``after_capture`` (around every capture of this roll-out), ``owns`` (which entries
    ``close`` / ``trip`` must not release)."""

    def __init__(self, ledger: Ledger, limit: int, graphs=None):
        self.ledger, self.limit = ledger, int(limit)
        self.backend = graphs if graphs is not None else CudaGraphs
        self.graphs: Dict[tuple, _Entry] = {}
        self.shapes: set = set()
        self.captures = 0
        self.dead: Optional[str] = None                             # the reason word once this roll-out went eager for good
        self.aborted = False                                        # a capture (not a warm-up) failed: the device's capture state is unusable from here
        self.closed = False

    # ----------------------------------------------------------------------------------------------------------- hook points (no-ops here)
    def adopt(self, key, refs, shape, stock_forward, module, batch, r_noisy, single_cond, pair_bias) -> Optional[_Entry]:
        """An entry captured OUTSIDE this roll-out that answers this call — its static output already holding THIS call's result — or None
        (the default: every roll-out captures its own graphs)."""
        return None

    def before_capture(self, module, shape) -> None:
        """Runs right before a capture of this roll-out (the keeper drops what it holds for other keys here: one shape's graphs alive)."""

    def after_capture(self, module, key, ent: _Entry, batch, pair_bias) -> None:
        """Runs right after a successful capture (the keeper records the entry and the tensors its kernels read by reference here)."""

    def owns(self, ent: _Entry) -> bool:
        """True when this roll-out may release ``ent`` (always, here; a kept entry belongs to its keeper)."""
        return True

    # ----------------------------------------------------------------------------------------------------------- the statement
    def eager(self, reason: str, stock: Callable[[], torch.Tensor]) -> torch.Tensor:
        self.ledger.fallback(reason)
        self.ledger.count("eager")
        return stock()

    def call(self, stock_forward: Callable, module, batch, r_noisy, single_cond, pair_bias) -> torch.Tensor:
        stock = lambda: stock_forward(module, batch, r_noisy, single_cond, pair_bias)     # noqa: E731 — the eager statement of THIS call
        word = gate_word(r_noisy, self.limit)
        if word is not None:
            return self.eager(word, stock)
        if self.dead is not None:
            return self.eager(self.dead, stock)
        if self.backend.is_capturing():
            return self.eager(CAPTURING, stock)
        refs: List[Any] = []
        key = call_key(batch, r_noisy, single_cond, pair_bias, refs)
        ent = self.graphs.get(key)
        if ent is None:
            shape = (tuple(r_noisy.shape), tuple(single_cond.shape))
            seen_shape = shape in self.shapes
            kept = self.adopt(key, refs, shape, stock_forward, module, batch, r_noisy, single_cond, pair_bias)
            if kept is not None:                                    # a graph captured earlier answers this call: its output holds this call's replay already
                self.graphs[key] = kept
                self.shapes.add(shape)
                self.ledger.serve("L%dxN%d" % (int(r_noisy.shape[2]), int(r_noisy.shape[1])))
                self.ledger.count("replays")
                return kept.out.clone()
            if self.captures + 1 > len(self.shapes | {shape}) + 1 or len(self.graphs) >= MAX_GRAPHS:
                return self.trip(RECAPTURE_STORM, stock)
            self.before_capture(module, shape)
            r_s, sc_s = static_copy(r_noisy), static_copy(single_cond)
            try:
                graph, out = self.backend.capture(lambda: stock_forward(module, batch, r_s, sc_s, pair_bias), r_noisy.device)
            except CaptureAborted as e:                             # the capture itself failed: counted, named, and the item ends here (module docstring)
                orig = e.original
                word = f"{CAPTURE_FAILED}:{type(orig).__name__}"
                self.ledger.set("capture_error", _fact_text(f"{type(orig).__name__}:{orig}"))
                self.ledger.fallback(word)
                self.dead, self.aborted = word, True
                self._drop()
                raise LeverAborted("denoiser_graph", word,
                                   f"denoiser_graph: the CUDA-graph capture of DiffusionModule.forward failed ({type(orig).__name__}: {orig}). "
                                   f"torch {torch.__version__} cannot continue after a failed stream capture (the CUDA RNG generator and the caching "
                                   f"allocator keep their capture state), so this item is not finished eagerly. AFO_DENOISER_GRAPH_MAX_TOKENS=0 runs "
                                   f"the denoiser eagerly.", hint=f"max_tokens={self.limit}; AFO_DENOISER_GRAPH_MAX_TOKENS=0 runs the denoiser eagerly") from orig
            except Exception as e:  # noqa: BLE001 — the warm-up call raised (device state intact): a NAMED eager roll-out
                self.ledger.set("capture_error", _fact_text(f"{type(e).__name__}:{e}"))
                return self.trip(f"{WARMUP_FAILED}:{type(e).__name__}", stock)
            ent = _Entry(graph, r_s, sc_s, out, refs, shape)
            self.graphs[key] = ent
            self.shapes.add(shape)
            self.captures += 1
            self.ledger.count("captures")
            if seen_shape:
                self.ledger.count("recaptures")
            self.after_capture(module, key, ent, batch, pair_bias)
        else:
            ent.r.copy_(r_noisy)
            ent.sc.copy_(single_cond)
        ent.graph.replay()
        self.ledger.serve("L%dxN%d" % (int(r_noisy.shape[2]), int(r_noisy.shape[1])))
        self.ledger.count("replays")
        return ent.out.clone()

    def trip(self, reason: str, stock: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Go eager for the rest of this roll-out under ``reason``; the graphs held so far are dropped now."""
        self.dead = reason
        self._drop()
        return self.eager(reason, stock)

    # ----------------------------------------------------------------------------------------------------------- lifetime
    def _drop(self) -> None:
        for ent in list(self.graphs.values()):
            if self.owns(ent):
                ent.release()
        self.graphs.clear()

    def close(self) -> None:
        self._drop()
        self.shapes.clear()
        self.closed = True


def _line(ledger: Ledger, tag: str):
    def line():
        return ledger.line(tag)
    return line


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        dh = importlib.import_module(TARGET)
    except Exception as e:  # noqa: BLE001
        return Installed("denoiser_graph", False, reason=f"import:{TARGET}:{type(e).__name__}")
    if not (hasattr(torch.cuda, "CUDAGraph") and hasattr(torch.cuda, "graph") and hasattr(torch.cuda, "is_current_stream_capturing")):
        return Installed("denoiser_graph", False, reason=f"torch_cuda_graphs_missing:torch@{torch.__version__}")
    limit = max_tokens()
    ledger = Ledger(NAME, impl=f"{CudaGraphs.impl}@{torch.__version__.split('+')[0]}", origin="kit", expected=EXPECTED)
    for k in ("captures", "replays", "eager", "recaptures", "rollouts"):
        ledger.set(k, 0)
    ledger.set("max_tokens", limit)
    mod_cls, head_cls = dh.DiffusionModule, dh.DiffusionHead
    stock_forward = mod_cls.forward
    inner_sample = head_cls.sample                                  # stock, or diffusion_bf16's autocast wrapper when that lever installed first (fast row)

    def forward(self, batch, r_noisy, single_cond, pair_bias):
        ro = getattr(self, STATE_ATTR, None)
        if ro is None or ro.closed:
            ledger.fallback(OUTSIDE_SAMPLE); ledger.count("eager")
            return stock_forward(self, batch, r_noisy, single_cond, pair_bias)
        return ro.call(stock_forward, self, batch, r_noisy, single_cond, pair_bias)
    forward.__qualname__ = "DiffusionModule.forward[atlasfold_opt:denoiser_graph]"

    def sample(self, *args, **kwargs):
        sm = getattr(self, "score_model", None)
        ro = (ROLLOUT_FACTORY or RollOut)(ledger, limit)
        prev = getattr(sm, STATE_ATTR, None) if sm is not None else None
        if sm is not None:
            setattr(sm, STATE_ATTR, ro)
        ledger.count("rollouts")
        try:
            return inner_sample(self, *args, **kwargs)
        finally:                                                    # graph lifetime = one roll-out: graphs, pool buffers, held references go here
            ro.close()
            if sm is not None:
                setattr(sm, STATE_ATTR, prev)
    sample.__qualname__ = "DiffusionHead.sample[atlasfold_opt:denoiser_graph]"

    rebind(mod_cls, "forward", forward, stock_forward)
    rebind(head_cls, "sample", sample, inner_sample)
    INSTALLED.update(ledger=ledger, limit=limit)                     # read by hooks/graph_reuse (installed after this lever in the row)
    ctx["denoiser_graph"] = {"ledger": ledger, "limit": limit}
    return Installed("denoiser_graph", True, lines=[_line(ledger, tag)], gates=[size_gated(ledger)],
                     facts={"impl": ledger.impl, "max_tokens": limit, "max_graphs": MAX_GRAPHS, "ledger": ledger})
