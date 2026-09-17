"""boltz_graph_trunk.py — CUDA-graph capture / replay of the Boltz-2 TRUNK modules
(boltz 2.2.1, inference only; the installed boltz package is never edited: class-level method replacement at attach time, `remove()`
restores the stock attributes).

What it does. Boltz-2's trunk runs, per prediction, (recycling_steps + 1) = 4 passes of [template module -> MSA module -> 64-block
Pairformer], then the confidence module's own 8-block Pairformer; at small inputs those passes are launch-bound (each Pairformer layer is
75-154 kernel launches issued from Python at ~40-65 us of host time apiece; the GPU idles between them). This lever wraps the forward of

    unit `pf`       boltz.model.layers.pairformer.PairformerModule.forward       (the trunk's 64-block stack AND the confidence module's stack)
    unit `pfnoseq`  boltz.model.layers.pairformer.PairformerNoSeqModule.forward  (the template module's 2-block pair stack)
    unit `msa`      boltz.model.modules.trunkv2.MSAModule.forward                (the 4-block MSA module, pair stack included)

in a per-(module instance, argument signature) CUDA graph. Policy per new signature: the process-first call of a module instance runs its own
forward eagerly (`eager:warm`: every process-level lazy initialisation — cuBLAS handles / heuristics, cuDNN plans, cuEquivariance kernel
selection — happens outside capture, and the `feats` keys the body reads are recorded); for every LATER new signature of a `pf` instance the
lever PRIMES instead of warming (policy `prime`, the module constant POLICY): one layer of the stack runs eagerly on the static input copies
under torch.random.fork_rng(devices=[cuda]) — the Triton / cuEquivariance kernels JIT-compile and load for the new shape outside capture,
and the layer's 4 dropout-mask draws leave the CPU and CUDA generators exactly as they were (state saved and restored; checked by offset) —
then the whole stack is captured and replayed for that very call: all 4 recycles of a first-of-shape item are served by the graph. Policy
`warm` (msa, pfnoseq always; pf when POLICY=warm) runs the first call of a signature eagerly and captures on the second. Every later call and
every later item of the same shapes copies its arguments into the static buffers and replays. Outputs are returned as fresh clones (no
graph-pool memory reaches the caller). One torch.cuda.MemPool per token-count generation; KEEP generations stay resident (default 2: the item
of a third token count evicts the oldest generation's graphs and returns that pool's segments to the driver; a call outside the token range
evicts everything resident when the pools hold more than 0.5 GiB, so nothing sizeable of this lever occupies memory while a large input runs
eagerly, and an alternating small/large workload keeps its small-shape graphs).
Unit `templ` is the one statement-level prerequisite fix this lever carries: TemplateModule / TemplateV2Module.forward build the distogram bin
`boundaries` with torch.linspace ON THE CPU and copy it to the device inside the forward — a pageable host->device copy = a stream
synchronisation per call (and an illegal operation under capture). The replacement body is the pinned stock body verbatim (checked by source
digest at apply; a mismatch refuses the unit by name) with that one tensor computed by the same CPU statement ONCE per (module, device) and
kept on the device: the same fp32 values, no sync.

Why it is exact (Tier 1, bitwise). A replay re-issues the captured kernels — the very kernels the eager call launches (stock's, or whatever
lever the mode row installed underneath: this wrapper sits below boltz_trunk_levers' module scoping and above the layer-level adapters) — with
the same launch geometry on the same parameter storage and on static input buffers holding bit-copies of the call's arguments; kernel
arithmetic does not depend on the issuing mechanism. RNG: the eval-mode dropout-mask draws (torch.rand on the CUDA generator, 4 per Pairformer
layer) are captured with PyTorch's graph-safe Philox bookkeeping — at replay the kernels read the generator's current offset from its
device-side state and the host-side offset advances by the graph's total increment — so the generator OFFSET after a replayed call equals the
eager call's (measured: +1024 per 64-layer pass, +80 per MSA call, +128 per confidence stack, capture itself +0, prime +0), which is exactly
what the item's downstream draws (diffusion noise) need to be bitwise; the mask VALUES are all-ones at this pin whatever the offset
(dropout.py: `rand >= 0.0`, dropout 0 in eval). Capture runs under torch.autocast(cache_enabled=False) so no weight cast cached
OUTSIDE the graph is baked into it (the cast kernels are captured and re-run on the live fp32 parameters; a cast is deterministic). cuBLAS
workspaces: the map is cleared (the ORIGINAL torch._C._cuda_clearCublasWorkspaces) right before every capture, so each graph's GEMM nodes take
a workspace allocated inside the graph's own pool (lifetime = the pool's), and again when a generation is evicted (no map entry ever points
into a freed pool); Lightning-facing clearing is a no-op while graphs live (as the sampler graph lever does; both levers' protocol is the
same: workspaces in pools, true clears only between captures — scratch memory, no numerics). Host syncs are not capturable: the trivial-mask
flag of boltz_trunk_levers (one `.all().item()` per module call) is taken by ITS wrapper, which wraps this one (attach order), and enters this
wrapper as a plain Python value that is part of the graph key.

Never silent. Every call is counted by unit under exactly one word: `replayed` | `captured` (the capturing call, replayed once for its own
result; `primed` counts the prime steps) | `eager:warm` (process-first call of an instance, or first call of a signature under policy warm) |
`eager:below_min_tokens` / `eager:above_max_tokens` (the row's token range) | `eager:no_cuda` / `eager:training` / `eager:grad` /
`eager:signature` (not an inference CUDA call of the pinned signature) | `eager:dead:<ExcType>` (a signature whose capture raised: that call and
its successors run the module's forward; the error is counted and the gate REFUSES — a capture failure is never absorbed). `report()` carries
the census, per-capture seconds and pool growth, MemPool GiB per resident shape, generations evicted, and the fail-closed verdict.

Switches (the mode row's words; family BOLTZ_GRAPH_ is stripped from callers by the kit): BOLTZ_GRAPH_TRUNK=<units> — comma list of
pf,pfnoseq,msa,templ | all | off; BOLTZ_GRAPH_TRUNK_MIN_TOKENS / BOLTZ_GRAPH_TRUNK_MAX_TOKENS (token range in which graphs are used; outside
it the forwards run eagerly, by name); BOLTZ_GRAPH_TRUNK_KEEP=<n> shape generations kept resident (default 2; a call outside the token range releases them all only when they hold more than RESIDENT_RELEASE_BYTES = 0.5 GiB); the capture policy
(prime | warm) is the module constant POLICY, not a switch. BOLTZ_GRAPH_DEBUG=1 (the graph levers' shared verbose word) prints capture events to stderr.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

UNITS = ("pf", "pfnoseq", "msa", "templ", "sovl")
SWITCH = "BOLTZ_GRAPH_TRUNK"
_DEBUG = os.environ.get("BOLTZ_GRAPH_DEBUG", "0") == "1"     # the graph levers' shared verbose word (stripped with the lever words; never part of a row)
_LOCK = threading.Lock()
POLICY = "prime"     # per new signature of a primer-carrying unit: "prime" (JIT-prime one layer traceless, capture at that very call) | "warm" (first call eager, capture at the second) — a module constant, not a switch

_STATE: Dict[str, Any] = {"applied": False, "units": (), "orig": {}, "min_tokens": 0, "max_tokens": 1 << 30, "keep": 2, "policy": "prime", "t_apply": None,
                          "templ_patched": [], "refused": {}, "sovl": False, "body": {}}
STATS: Dict[str, Any] = {"census": {}, "captures": [], "errors": {}, "generations_evicted": 0, "pool_gib_max": 0.0, "static_gib": 0.0, "prime_rng_checks": 0}
# sha256[:16] of inspect.getsource(<stock function>) at the boltz 2.2.1 pin (stock/boltz-2.2.1-py3-none-any.whl): the statements this module restates
# (the template forwards verbatim-but-for-one-statement; the Pairformer modules' eval chunk-size statement inside the prime step) are valid for these texts
SRC_SHA = {"TemplateV2Module.forward": "3cd6d2487ecfcda8", "TemplateModule.forward": "a674f8f9e001ca71", "PairformerModule.forward": "9e97470f82992467", "PairformerNoSeqModule.forward": "be9f522b649b5c5d", "MSAModule.forward": "b73fba24ca797c2e",
           "PairformerLayer.forward": "d3e9b735f96058e1"}
# The bodies a graph unit may wrap, per patched method: source digest -> traits. `stock` = the vendored statement (SRC_SHA above). The PAIRFUSE
# layer driver's class-level replacements (forward/pairfuse/bz_pairfuse.py pfm_forward / pfnm_forward, the fast row) are capturable regions
# except for ONE host synchronisation per stack call — `pair_mask is None or bool((pair_mask == 1).all())` — so they carry the trait
# trivial_mask_as_none: this wrapper decides the pair mask's triviality OUTSIDE the captured region (the trunk levers' per-call flag, else the
# same single reduction) and calls the body with pair_mask=None when it is trivial, which is that body's own trivial branch (every core gets no
# mask); a real mask runs the body eagerly, by name (eager:mask). prime=body: a new shape is primed by the body itself on a one-layer view of
# the module (its per-shape site probe, provider JIT and proj_z installation happen outside capture; its RNG draws stay traceless under fork_rng).
# A digest outside this table refuses the unit by name at apply (refused_at_apply) — the owner of a body re-pins it here when it changes.
BODIES = {
    "PairformerModule.forward": {"9e97470f82992467": {"name": "stock", "prime": "layer0"},
                                 "ba126d772b95a471": {"name": "pairfuse.pfm_forward", "prime": "body", "trivial_mask_as_none": True, "sovl": False}},
    "PairformerNoSeqModule.forward": {"be9f522b649b5c5d": {"name": "stock", "prime": "layer0"},
                                      "2c69c761afa1df54": {"name": "pairfuse.pfnm_forward", "prime": "body", "trivial_mask_as_none": True, "sovl": False}},
}
# the layer forwards under which unit sovl's ordering assumptions hold (the sequence stack is the layer's last statement group; z is updated out of place):
# stock PairformerLayer.forward (above) and boltz_trunk_levers._pf_layer_forward (the resid lever's restatement)
SOVL_LAYER_FORWARDS = {"d3e9b735f96058e1": "stock", "ce1fbd1f0c771495": "boltz_trunk_levers._pf_layer_forward"}   # the lever's forward with the residual/LayerNorm fuser sites: z still updated out of place, read on S only inside attention.proj_z
# the AttentionPairBias forwards under which the side stream reads z ONLY inside attention.proj_z and s / k_in / mask only through its own
# projections (stock attentionv2.AttentionPairBias.forward and boltz_trunk_levers._apb_forward, the mask2 lever's restatement):
SOVL_APB_FORWARDS = {"d78391e056ba3b3d": "stock", "eb8fd7761b2764fb": "boltz_trunk_levers._apb_forward",
                     "a14b8dc723baae3a": "boltz_dit_hoist._apb_forward>_stock_forward"}   # the DiT hoist's class patch: outside a diffusion
# sampler call (always the case inside a Pairformer stack) it delegates to AttentionPairBias._stock_forward — the body it displaced, which
# must itself be one of the non-delegating bodies above (checked: name>attr = follow the class attribute `attr` and check its digest too)


def _log(*a):
    if _DEBUG:
        print("[boltz_graph_trunk]", *a, file=sys.stderr, flush=True)


def _cnt(unit: str, word: str, n: int = 1):
    c = STATS["census"].setdefault(unit, {})
    c[word] = c.get(word, 0) + n


def parse_units(spec: Optional[str]) -> Tuple[str, ...]:
    spec = (spec or "").strip().lower()
    if spec in ("", "0", "off", "none"):
        return ()
    if spec in ("1", "all", "on"):
        return UNITS
    out: List[str] = []
    for tok in spec.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in UNITS and not (tok.isidentifier() and tok.islower()):     # a word outside UNITS names a unit another lever registers (register_unit); junk is refused
            raise ValueError(f"{SWITCH}: unknown unit '{tok}' (built-in: {UNITS}; other levers register theirs by a lowercase word)")
        if tok not in out:
            out.append(tok)
    return tuple(out)


def _int_env(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    try:
        return int(v) if v else default
    except ValueError:
        raise ValueError(f"{name}: not an integer: {v!r}")


def _src_sha(fn) -> str:
    try:
        return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unreadable"


# ----------------------------------------------------------------------------------------------------------------------
# cuBLAS workspace protocol (shared with the sampler graph lever, opt/forward/dit_hoist/src/boltz_graph_patch.py)
# ----------------------------------------------------------------------------------------------------------------------
_WS = {"replaced": None, "installed": False}


def _true_clear_ws():
    """The ORIGINAL torch._C._cuda_clearCublasWorkspaces: the sampler lever's stored original when it is loaded (it may have replaced the
    attribute with its Lightning-facing no-op before or after this module did), else the attribute this module replaced, else the live one."""
    gp = sys.modules.get("boltz_graph_patch")
    f = None
    if gp is not None:
        f = (getattr(gp, "_APPLIED", {}) or {}).get("clear_ws_orig")
    if f is None:
        f = _WS["replaced"]
    if f is None:
        f = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    return f


def _clear_cublas_workspaces_now():
    f = _true_clear_ws()
    if f is not None:
        f()


def _guard_cublas_workspaces(enable: bool):
    """While graphs live, the Lightning-facing torch._C._cuda_clearCublasWorkspaces is a no-op (a captured GEMM node reads its workspace; the
    teardown's clear + empty_cache would unmap it). Installed lazily at the first capture (after the sampler lever's own guard, whose stored
    original _true_clear_ws() prefers), removed by remove()."""
    C = torch._C
    if enable and not _WS["installed"] and hasattr(C, "_cuda_clearCublasWorkspaces"):
        _WS["replaced"] = C._cuda_clearCublasWorkspaces
        C._cuda_clearCublasWorkspaces = lambda: None
        _WS["installed"] = True
    elif not enable and _WS["installed"]:
        C._cuda_clearCublasWorkspaces = _WS["replaced"]; _WS["replaced"] = None; _WS["installed"] = False


# ----------------------------------------------------------------------------------------------------------------------
# graph machinery
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# unit sovl: the Pairformer layer's fp32 sequence stack on a side stream
# ----------------------------------------------------------------------------------------------------------------------
# Within PairformerLayer l the sequence stack [pre_norm_s -> AttentionPairBias(s, bias = proj_z(z_l)) -> transition_s -> s_post_norm] is the
# LAST statement group; it reads z_l (the layer's finished pair tensor) and s_{l-1}, and nothing of layer l+1's pair track reads s. So the
# sequence stack of layer l may run on a second stream S concurrently with layer l+1's pair track on the main stream M: same kernels, same
# operands, same order within each stream => the same bits (a stream is not an operand). Ordering, by
# module hooks (no statement of the layer is restated): at pre_norm_s of layer l, S waits on M (z_l and s_{l-1} complete) and becomes the
# current stream; at the layer's return M becomes current again; M waits on S's `bias_l read` event before layer l+1's first residual update
# (tri_mul_out's return — protects a lever that would update z in place; stock and the resid lever update out of place) and on S's `s_l done`
# event before layer l+2 starts; after the last layer M waits on S and s is re-materialised on M. MEMORY (the hazard the stream events do
# not cover): the caching allocator orders a freed block's reuse by its ALLOCATING stream only, so every main-stream tensor the side stream
# reads is lent to S with Tensor.record_stream(S) at the hand-over (pre_norm_s's input — `s.float()`, a main-stream temporary whenever the
# stack's input s is bf16, i.e. layer 0 of every trunk and confidence pass — and AttentionPairBias's inputs: z_l, mask): its block is not
# reissued to main-stream work until S has passed the read (inside a capture such frees are deferred to the capture's end: no node of the
# graph can alias it). Without the lending, a 61-118-token input let a layer-1 main-stream temporary land in the freed `s.float()` block
# while S's LayerNorm still read it (bitwise-different outputs in ~1 of 3 runs); 199-1400-token inputs never did, by allocator luck only.
# Valid for the layer forwards in SOVL_LAYER_FORWARDS and the AttentionPairBias forwards in SOVL_APB_FORWARDS — z is read on S ONLY inside
# attention.proj_z (checked by digest at the first call of each module instance; anything else runs without the overlap, counted
# sovl:skipped:<why>). Composes with capture: inside a captured call the fork/join events become graph dependencies and the two branches
# replay concurrently with no launch cost.
RESIDENT_RELEASE_BYTES = 1 << 29   # 0.5 GiB: below this the resident generations stay through out-of-range (eager) calls — an alternating
# small/large workload keeps its small-shape graphs; above it (unit msa, or a MAX_TOKENS in the thousands) a large eager input gets the memory back

_SOVL_STREAMS: Dict[int, torch.cuda.Stream] = {}
_SOVL_OBJS: List["_Sovl"] = []


def _sovl_stream(dev: torch.device) -> torch.cuda.Stream:
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    st = _SOVL_STREAMS.get(i)
    if st is None:
        st = torch.cuda.Stream(device=i); _SOVL_STREAMS[i] = st
    return st


class _Sovl:
    """Per-PairformerModule-instance stream plumbing for unit sovl (hooks on its layers; inert unless begin() armed it for the call in flight)."""

    def __init__(self, module):
        import functools
        self.active = False; self.M = None; self.S = None; self.dev = None; self.done: Dict[int, Any] = {}; self.bias_done: Dict[int, Any] = {}
        self.handles = []; self.n = len(module.layers); self.ok = None; self.why = None
        for i, layer in enumerate(module.layers):
            if not all(hasattr(layer, a) for a in ("pre_norm_s", "attention", "tri_mul_out")) or not hasattr(layer.attention, "proj_z"):
                self.ok = False; self.why = "layer_structure"; break
        if self.ok is not False:
            for i, layer in enumerate(module.layers):
                self.handles.append(layer.register_forward_pre_hook(functools.partial(self._layer_pre, i)))
                self.handles.append(layer.register_forward_hook(functools.partial(self._layer_post, i)))
                self.handles.append(layer.pre_norm_s.register_forward_pre_hook(functools.partial(self._seq_pre, i)))
                self.handles.append(layer.attention.register_forward_pre_hook(functools.partial(self._attn_pre, i), with_kwargs=True))
                self.handles.append(layer.attention.proj_z.register_forward_hook(functools.partial(self._bias_post, i)))
                self.handles.append(layer.tri_mul_out.register_forward_hook(functools.partial(self._tmo_post, i)))
            self.ok = True
        _SOVL_OBJS.append(self)

    def check_forward(self, module) -> bool:
        """The live layer forward must be one whose ordering this unit knows (digest); else no overlap for this instance (counted by name)."""
        fn = type(module.layers[0]).forward
        sha = _src_sha(fn)
        if sha not in SOVL_LAYER_FORWARDS:
            self.ok = False; self.why = f"unknown_layer_forward:{getattr(fn, '__name__', '?')}:{sha}"
            return False
        acls = type(module.layers[0].attention)
        afn = acls.forward
        for _ in range(3):                        # follow at most a short chain of delegating bodies (name>attr entries)
            asha = _src_sha(afn)
            known = SOVL_APB_FORWARDS.get(asha)
            if known is None:
                self.ok = False; self.why = f"unknown_attention_forward:{getattr(afn, '__name__', '?')}:{asha}"
                return False
            if ">" not in known:
                return True
            afn = getattr(acls, known.split(">", 1)[1], None)
            if afn is None:
                self.ok = False; self.why = f"attention_delegate_missing:{known}"
                return False
        self.ok = False; self.why = "attention_delegate_chain_too_long"
        return False

    def begin(self, dev: torch.device):
        self.dev = dev; self.M = torch.cuda.current_stream(dev); self.S = _sovl_stream(dev)
        self.done.clear(); self.bias_done.clear(); self.active = True

    def end(self):
        if not self.active:
            return
        try:
            torch.cuda.set_stream(self.M)
            self.M.wait_stream(self.S)
        finally:
            self.active = False

    # hooks (positional-call modules; each returns None so inputs/outputs are untouched)
    def _layer_pre(self, i, mod, args):
        if self.active:
            if torch.cuda.current_stream(self.dev) != self.M:
                torch.cuda.set_stream(self.M)
            ev = self.done.get(i - 2)
            if ev is not None:
                self.M.wait_event(ev)

    def _tmo_post(self, i, mod, args, out):
        if self.active:
            ev = self.bias_done.get(i - 1)
            if ev is not None:
                torch.cuda.current_stream(self.dev).wait_event(ev)

    def _seq_pre(self, i, mod, args):
        if self.active:
            ev = self.M.record_event()
            torch.cuda.set_stream(self.S)
            self.S.wait_event(ev)
            self._lend(args)                 # pre_norm_s's input: `s.float()` — a main-stream TEMPORARY when the stack's input s is bf16 (layer 0)

    def _attn_pre(self, i, mod, args, kwargs):
        if self.active:
            self._lend(args); self._lend(kwargs.values())    # s_normed / k_in (side-stream tensors: no-op), z(l) and mask (main-stream tensors)

    def _lend(self, tensors):
        """Every main-stream tensor the side stream reads is LENT to it through the allocator (Tensor.record_stream): when its last reference
        drops — for an argument temporary that is immediately — its block is not handed to a later main-stream allocation until the side
        stream's pending work has passed (the allocator records an event on S at free; inside a capture the free is deferred to the capture's
        end, so no node of the graph can alias it). This is the ordering the caching allocator does NOT infer by itself: it orders block reuse
        by the ALLOCATING stream only."""
        for t in tensors:
            if torch.is_tensor(t) and t.is_cuda:
                t.record_stream(self.S)

    def _bias_post(self, i, mod, args, out):
        if self.active and torch.cuda.current_stream(self.dev) == self.S:
            self.bias_done[i] = self.S.record_event()

    def _layer_post(self, i, mod, args, out):
        if self.active:
            self.done[i] = self.S.record_event()
            torch.cuda.set_stream(self.M)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _sovl_for(module) -> Optional["_Sovl"]:
    """The instance's _Sovl (created at first use), or None when unit sovl is off or this instance cannot take it (counted once by reason)."""
    if not _STATE.get("sovl"):
        return None
    sv = module.__dict__.get("_graph_trunk_sovl")
    if sv is None:
        sv = _Sovl(module)
        if sv.ok:
            sv.check_forward(module)
        module.__dict__["_graph_trunk_sovl"] = sv
        if not sv.ok:
            _cnt("sovl", "skipped:" + str(sv.why))
    return sv if sv.ok else None


def _run(unit: str, orig, module, dev: Optional[torch.device], a: tuple, k: dict):
    """orig(module, *a, **k) — with unit sovl's fork/join around it for PairformerModule calls on CUDA (unit pf), s re-materialised on the main stream."""
    sv = _sovl_for(module) if (unit == "pf" and dev is not None and _STATE["body"].get("pf", {}).get("sovl", True)) else None
    if sv is None:
        return orig(module, *a, **k)
    sv.begin(dev)
    try:
        out = orig(module, *a, **k)
    finally:
        sv.end()
    _cnt("sovl", "calls")
    if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]):
        return (out[0].clone(), out[1])            # s was produced on the side stream: hand the caller a main-stream tensor
    return out


class _Entry:
    __slots__ = ("key", "unit", "state", "graph", "static_in", "static_feats", "src_memo", "static_out", "is_tuple", "n_replay", "capture_s", "err")

    def __init__(self, key, unit):
        self.key = key; self.unit = unit; self.state = "warm"; self.graph = None; self.static_in: Dict[str, torch.Tensor] = {}
        self.static_feats: Dict[str, Dict[str, torch.Tensor]] = {}; self.src_memo: Dict[Any, tuple] = {}; self.static_out = None
        self.is_tuple = False; self.n_replay = 0; self.capture_s = None; self.err = None

    def drop(self):
        self.graph = None; self.static_in = {}; self.static_feats = {}; self.src_memo = {}; self.static_out = None


class _Gen:
    """One shape generation: the graphs captured for one trunk token count share a torch.cuda.MemPool; evicting the generation drops every
    graph and static buffer of it, clears the cuBLAS workspace map (entries pointing into the pool) and releases the pool's segments."""
    def __init__(self, tokens: int, device: torch.device):
        self.tokens = tokens; self.device = device
        self.pool = torch.cuda.MemPool()
        self.entries: Dict[tuple, _Entry] = {}
        self.t = time.time()

    def release(self) -> int:
        n = sum(1 for e in self.entries.values() if e.graph is not None)
        try:
            torch.cuda.synchronize(self.device)
        except Exception:  # noqa: BLE001
            pass
        for e in list(self.entries.values()):
            e.drop()
        self.entries.clear()
        _clear_cublas_workspaces_now()      # no (handle, stream) workspace entry may point into the pool being freed
        self.pool = None                    # MemPool.__del__: its segments are returned to the driver (this pool's only)
        return n


class _Recorder(dict):
    """A dict that records which keys the module body reads (the warm call learns the `feats` entries to stage as static buffers)."""
    def __init__(self, d, seen: set):
        super().__init__(d); self._seen = seen
    def __getitem__(self, k):
        self._seen.add(k); return dict.__getitem__(self, k)
    def get(self, k, default=None):
        self._seen.add(k); return dict.get(self, k, default)


_GENS: List[_Gen] = []                       # resident generations, oldest first
_KNOWN_FEATS: Dict[tuple, set] = {}         # (unit, id(module), dict-arg name) -> feats keys the body was seen to read
_WARMED: set = set()                        # (unit, id(module)) that completed >= 1 eager call in this process (prime allowed from then on)
_SIDE: Dict[int, torch.cuda.Stream] = {}    # capture stream per device


def _sig(t: torch.Tensor):
    return (tuple(t.shape), str(t.dtype), t.device.index)


def _side_stream(dev: torch.device) -> torch.cuda.Stream:
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    st = _SIDE.get(i)
    if st is None:
        st = torch.cuda.Stream(device=i); _SIDE[i] = st
    return st


def release_all() -> int:
    """Drop every resident graph generation (graphs, static buffers, pools). Returns the number of graphs dropped."""
    n = 0
    while _GENS:
        g = _GENS.pop(0); n += g.release(); del g
        STATS["generations_evicted"] += 1
    return n


def _generation(tokens: int, dev: torch.device) -> _Gen:
    for g in _GENS:
        if g.tokens == tokens:
            return g
    while len(_GENS) >= max(1, _STATE["keep"]):
        old = _GENS.pop(0); n = old.release(); del old
        STATS["generations_evicted"] += 1
        _log(f"evicted a generation ({n} graphs)")
    g = _Gen(tokens, dev); _GENS.append(g)
    return g


def _driver_api(name: str):
    """An optional entry point of the PAIRFUSE driver module (mask_predicate(bool) context / primed(module, z, pair_mask) readiness), or None."""
    m = sys.modules.get("bz_pairfuse")
    return getattr(m, name, None) if m is not None else None


def _mask_flag():
    """boltz_trunk_levers' trivial-mask flag for the module call in flight (a plain value here: its host sync happened in the outer wrapper)."""
    m = sys.modules.get("boltz_trunk_levers")
    if m is None:
        return None
    return getattr(getattr(m, "_CTX", None), "mask_trivial", None)


def _bind(params: Tuple[str, ...], defaults: Dict[str, Any], self, a: tuple, k: dict) -> Optional[Dict[str, Any]]:
    """Bind (self, *a, **k) to the STOCK parameter names of the wrapped forward (the pin's signature, stated by the caller of _graphed — not
    read off `orig`, which may itself be a wrapper with an opaque (*a, **k) signature). None when the call does not fit those names."""
    if len(a) > len(params):
        return None
    args: Dict[str, Any] = {"self": self}
    args.update(zip(params, a))
    for name, val in k.items():
        if name not in params or name in args:
            return None
        args[name] = val
    for name in params:
        if name not in args:
            if name not in defaults:
                return None
            args[name] = defaults[name]
    return {n: args[n] for n in ("self",) + params}


def _key(unit, module, args, tensor_args, dict_args, ac, extra=None):
    parts = [unit, id(module), ac, _mask_flag(), extra]
    for name, val in args.items():
        if name == "self":
            continue
        if name in tensor_args:
            parts.append((name, _sig(val) if torch.is_tensor(val) else repr(val)))
        elif name in dict_args:
            known = _KNOWN_FEATS.get((unit, id(module), name), set())
            parts.append((name, tuple((kk, _sig(val[kk]) if torch.is_tensor(val[kk]) else repr(val[kk])) for kk in sorted(known) if kk in val)))
        else:
            parts.append((name, repr(val)))
    return tuple(parts)


def _copy_in(entry: _Entry, slot, dst: torch.Tensor, src: torch.Tensor):
    """dst.copy_(src): a device-to-device copy per staged argument per call (tens of MB at most: microseconds). No identity/version elision —
    the worker runs under torch.inference_mode, whose tensors carry no version counter, so 'the same unmodified tensor as last call' is not
    decidable; a bit-copy always is."""
    if dst.data_ptr() != src.data_ptr():
        dst.copy_(src)


def _stage_inputs(entry: _Entry, args, tensor_args, dict_args, allocate: bool) -> Dict[str, Any]:
    """Copy the call's tensor arguments into the entry's static buffers (allocating them on first use). Returns the kwargs to call with."""
    call: Dict[str, Any] = {}
    nbytes = 0
    for name, val in args.items():
        if name == "self":
            continue
        if name in tensor_args and torch.is_tensor(val):
            if allocate:
                entry.static_in[name] = val.detach().clone(memory_format=torch.contiguous_format); nbytes += val.numel() * val.element_size()
            else:
                _copy_in(entry, name, entry.static_in[name], val)
            call[name] = entry.static_in[name]
        elif name in dict_args and isinstance(val, dict):
            known = _KNOWN_FEATS.get((entry.unit, id(args["self"]), name), set())
            st = entry.static_feats.setdefault(name, {})
            d = {}
            for kk, vv in val.items():
                if kk in known and torch.is_tensor(vv) and vv.is_cuda:
                    if allocate:
                        st[kk] = vv.detach().clone(memory_format=torch.contiguous_format); nbytes += vv.numel() * vv.element_size()
                    else:
                        _copy_in(entry, (name, kk), st[kk], vv)
                    d[kk] = st[kk]
                elif kk in known:
                    d[kk] = vv                          # a non-tensor / CPU value the body reads: passed as is (part of the key by repr)
                # keys the body never read are not passed: a body that reads a new key under capture raises KeyError -> counted, dead, eager
            call[name] = d
        else:
            call[name] = val
    if allocate:
        STATS["static_gib"] = round(STATS["static_gib"] + nbytes / 2**30, 4)
    return call


class _OneLayerView:
    """What a stack body reads off its module (layers, training) — over the FIRST layer only: the body's own per-shape work (site probe,
    provider JIT, lazy sub-module installation) runs once, eagerly, at a fraction of the stack's cost."""
    __slots__ = ("layers", "training", "_module")

    def __init__(self, module):
        self.layers = module.layers[:1]; self.training = False; self._module = module

    def __getattr__(self, name):                 # anything else the body reads: the module's own attribute
        return getattr(self._module, name)


def _prime_pairformer(module, call: Dict[str, Any], dev: torch.device, unit: str = "pf", orig=None) -> None:
    """The built-in primer of units pf / pfnoseq: the stack's FIRST LAYER once, eagerly, on the static input copies — every per-shape lazy step
    of the kernels underneath (Triton JIT + module load, cuEquivariance selection, cuBLAS heuristics, a driver's site probe) happens here,
    outside capture. Stock body: the layer module itself; a driver body (BODIES prime=body): that body on a one-layer view of the module."""
    if _STATE["body"].get(unit, {}).get("prime") == "body" and orig is not None:
        orig(_OneLayerView(module), **call)
        ready = _driver_api("primed")               # the driver's host-only readiness statement for (module, shape, mask), when offered: ADVISORY —
        if ready is not None:                       # a module the driver does not serve (e.g. the 64-wide template stack) never becomes "primed"
            r = ready(module, call.get("z"), call.get("pair_mask"))   # although its body (the driver's own fallback statements) is capturable;
            if isinstance(r, dict) and not r.get("ok", True):          # a body that does synchronise fails the capture itself, by name
                STATS["primed_missing"] = STATS.get("primed_missing", 0) + 1
                _log(f"driver readiness for unit {unit}: missing={r.get('missing')} — capturing anyway (a synchronising body fails the capture by name)")
        return
    from boltz.data import const
    z = call["z"]
    chunk_size_tri_attn = 128 if z.shape[1] > const.chunk_size_threshold else 512     # PairformerModule / PairformerNoSeqModule.forward's eval statement (pairformer.py, digest-checked at apply)
    layer = module.layers[0]
    if "s" in call:
        layer(call["s"], call["z"], call["mask"], call["pair_mask"], chunk_size_tri_attn, call["use_kernels"])
    else:
        layer(call["z"], call["pair_mask"], chunk_size_tri_attn, call["use_kernels"])


def _prime(unit: str, module, call: Dict[str, Any], dev: torch.device, orig=None):
    """Policy prime: the unit's primer under torch.random.fork_rng(devices=[dev]) — the CPU generator and this device's CUDA generator are
    saved before and restored after, so the primer's dropout-mask draws leave no trace (checked: the CUDA generator offset after == before,
    else RuntimeError -> the capture path refuses by name)."""
    gen = torch.cuda.default_generators[dev.index if dev.index is not None else torch.cuda.current_device()]
    off0 = gen.get_offset()
    with torch.random.fork_rng(devices=[dev]):
        if _PRIMERS[unit] is _prime_pairformer:
            _prime_pairformer(module, call, dev, unit=unit, orig=orig)
        else:
            _PRIMERS[unit](module, call, dev)
    if gen.get_offset() != off0:
        raise RuntimeError(f"prime: CUDA generator offset not restored ({off0} -> {gen.get_offset()})")
    STATS["prime_rng_checks"] += 1
    _cnt(unit, "primed")


_PRIMERS: Dict[str, Any] = {"pf": _prime_pairformer, "pfnoseq": _prime_pairformer}   # unit -> prime(module, call, dev): JIT-prime a new shape outside capture (policy prime); absent = policy warm
_REGISTERED: Dict[str, Dict[str, Any]] = {}   # units registered through register_unit (other levers' capturable regions), by name


def register_unit(name: str, cls, method: str = "forward", *, tensor_args: Tuple[str, ...], dict_args: Tuple[str, ...] = (), tokens_of,
                  prime=None, src_sha256: Optional[str] = None) -> str:
    """The registration hook for OTHER levers' capturable regions (this module owns capture, pools, census and the gate; a registrant
    owns its region's capturability). Wraps `cls.<method>` in the same per-(instance, argument-signature) capture/replay machinery as the
    built-in units, under the unit word `name`:
      tensor_args  parameter names bit-copied into static buffers per call (their shape/dtype/stride are part of the graph key);
      dict_args    parameter names holding {str: Tensor} dicts (the keys the body reads are recorded at the warm call and staged likewise);
                   every other parameter is a plain value and part of the key BY VALUE (so Python control flow on it is legal in the body);
      tokens_of    f(bound_args: dict) -> int, the call's token count: the MIN/MAX_TOKENS gate and the shape generation (KEEP) it lives in;
      prime        optional f(module, bound_args, device) run under torch.random.fork_rng before the capture of a NEW signature of an already
                   warmed instance (policy prime: JIT-compile / select kernels for the new shape outside capture, no side effects that outlive
                   it); None = policy warm for this unit (first call of a signature eager, capture at the second);
      src_sha256   optional pin of inspect.getsource(cls.<method>): a mismatch refuses the unit by name (refused_at_apply in the gate).
    The body's contract (the registrant's to prove): capturable — no host synchronisation (.item / .cpu / nonzero / pageable copies), no
    tensor-VALUE-dependent shapes or control flow, RNG only through torch's CUDA generator, every tensor it reads arrives through tensor_args /
    dict_args / the module's own parameters and buffers (anything else is baked into the graph by address), outputs a Tensor or a tuple/list of
    Tensors (returned to the caller as fresh clones). The unit is switched by THIS lever's row word (its name listed in BOLTZ_GRAPH_TRUNK) —
    registered but not listed = never installed (census word absent, LEVER line units= without it); listed but never registered by the time of
    the first call = refused by name. Counters, pools, KEEP generations, eviction, the cuBLAS-workspace protocol and the fail-closed gate are
    this module's and cover the unit like the built-in ones. Returns the state: "installed" | "pending" (installed at apply) | "refused:<why>"."""
    import inspect
    fn = getattr(cls, method, None)
    if fn is None or not callable(fn):
        return "refused:no_such_method"
    if src_sha256 is not None and _src_sha(fn) != src_sha256:
        _STATE["refused"][name] = "source_mismatch"
        return "refused:source_mismatch"
    sig = inspect.signature(fn)
    params = tuple(sig.parameters)
    defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    spec = {"cls": cls, "method": method, "params": params, "defaults": defaults, "tensor_args": tuple(tensor_args), "dict_args": tuple(dict_args),
            "tokens_of": tokens_of, "key": f"{cls.__name__}.{method}"}
    if prime is not None:
        _PRIMERS[name] = prime
    _REGISTERED[name] = spec
    with _LOCK:
        if _STATE["applied"]:
            return _install_registered(name)
    return "pending"


def _install_registered(name: str) -> str:
    """Install a registered unit if the row's spec lists it (called at apply for pending registrations, or at registration after apply)."""
    spec = _REGISTERED.get(name)
    wanted = _STATE.get("wanted", ())
    if spec is None or name not in wanted:
        return "pending"
    if name in _STATE["units"]:
        return "installed"
    cls, method = spec["cls"], spec["method"]
    orig = getattr(cls, method)
    _STATE["orig"][spec["key"]] = orig
    _EXTRA_CLASSES[spec["key"]] = cls
    setattr(cls, method, _graphed(name, orig, spec["params"], spec["defaults"], spec["tensor_args"], spec["dict_args"], spec["tokens_of"]))
    _STATE["units"] = tuple(_STATE["units"]) + (name,)
    _log(f"unit {name} installed on {spec['key']} (registered)")
    return "installed"


_EXTRA_CLASSES: Dict[str, Any] = {}     # key -> class, for registered units (remove() restores them)


def _graphed(unit: str, orig, params: Tuple[str, ...], defaults: Dict[str, Any], tensor_args: Tuple[str, ...], dict_args: Tuple[str, ...], tokens_of):
    """Build the replacement forward for `orig` (the class attribute at apply time: stock's function or a lever's). params/defaults: the stock
    signature at the pin; tensor_args / dict_args: parameter names staged as static buffers; every other argument is a plain value that becomes
    part of the graph key. tokens_of(args) -> the call's token count."""

    def forward(self, *a, **k):
        args = _bind(params, defaults, self, a, k)
        if args is None:
            _cnt(unit, "eager:signature"); return orig(self, *a, **k)
        probe = next((args[n] for n in tensor_args if torch.is_tensor(args.get(n))), None)
        if probe is None or not probe.is_cuda:
            _cnt(unit, "eager:no_cuda"); return orig(self, *a, **k)
        dev = probe.device
        if self.training:
            _cnt(unit, "eager:training"); return orig(self, *a, **k)
        if torch.cuda.is_current_stream_capturing():             # called inside ANOTHER unit's capture (an enclosing registered region): the body is
            _cnt(unit, "eager:inside_capture"); return orig(self, *a, **k)   # issued into that capture as is — one graph, no capture-within-capture
        if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
            _cnt(unit, "eager:grad"); return _run(unit, orig, self, dev, a, k)
        N = int(tokens_of(args))
        if N < _STATE["min_tokens"] or N > _STATE["max_tokens"]:
            if _GENS and sum(_pool_bytes(g) for g in _GENS) > RESIDENT_RELEASE_BYTES:   # a large input runs eagerly: give its transients the pools'
                n = release_all(); _log(f"call outside the token range (N={N}): released {n} resident graphs")   # memory — unless they are small
            _cnt(unit, "eager:below_min_tokens" if N < _STATE["min_tokens"] else "eager:above_max_tokens"); return _run(unit, orig, self, dev, a, k)
        pred_ctx = contextlib.nullcontext()
        extra = None
        if _STATE["body"].get(unit, {}).get("trivial_mask_as_none") and torch.is_tensor(args.get("pair_mask")):
            flag = _mask_flag()                                      # the trunk levers' per-call verdict (its reduction ran in their outer wrapper) ...
            trivial = bool(flag) if flag is not None else bool((args["pair_mask"] == 1).all())   # ... else the body's own single reduction, here, OUTSIDE the captured region
            mp = _driver_api("mask_predicate")
            if mp is not None:                                       # the driver takes a host-stated predicate: zero syncs inside, real masks capturable too
                pred_ctx = mp(trivial); extra = ("mask_predicate", trivial)
                STATS["mask_predicate"] = STATS.get("mask_predicate", 0) + 1
            elif trivial:                                            # an older driver body: its own trivial branch is pair_mask=None (no mask reaches any core)
                args = dict(args); args["pair_mask"] = None
                a, k = (), {n: v for n, v in args.items() if n != "self"}
                STATS["mask_as_none"] = STATS.get("mask_as_none", 0) + 1
            else:
                _cnt(unit, "eager:mask"); return _run(unit, orig, self, dev, a, k)   # a real pair mask and no predicate API: the body decides it itself (one sync) — eagerly, by name
        with pred_ctx:
            return _serve(unit, orig, self, dev, args, tensor_args, dict_args, tokens_of, N, a, k, extra)

    forward.__wrapped_by_graph_trunk__ = True
    forward._graph_trunk_orig = orig
    forward.__wrapped__ = orig                   # other levers' displacement checks walk __wrapped__ chains down to their own body
    return forward


def _serve(unit, orig, self, dev, args, tensor_args, dict_args, tokens_of, N, a, k, extra):
    """The capture / replay decision for one in-range call (split out of the wrapper so a driver's predicate context can enclose it)."""
    if True:   # (kept at the wrapper's indentation)
        ac = (torch.is_autocast_enabled("cuda"), str(torch.get_autocast_dtype("cuda")))
        key = _key(unit, self, args, tensor_args, dict_args, ac, extra)
        gen = _generation(N, dev)
        entry = gen.entries.get(key)
        inst = (unit, id(self))
        if entry is None:
            if unit in _PRIMERS and _STATE["policy"] == "prime" and inst in _WARMED and not dict_args:
                entry = _Entry(key, unit); gen.entries[key] = entry
                return _capture_and_replay(unit, orig, self, args, entry, gen, dev, tensor_args, dict_args, a, k, prime=True)
            # warm: the module's own forward, eagerly, recording the feats keys it reads
            seen = {n: set() for n in dict_args}
            call = dict(args); call.pop("self")
            for n in dict_args:
                if isinstance(call.get(n), dict):
                    call[n] = _Recorder(call[n], seen[n])
            out = _run(unit, orig, self, dev, (), call)
            for n in dict_args:
                _KNOWN_FEATS.setdefault((unit, id(self), n), set()).update(seen[n])
            key = _key(unit, self, args, tensor_args, dict_args, ac, extra)     # the key now carries the feats signatures just learned
            gen.entries[key] = _Entry(key, unit)
            _WARMED.add(inst)
            _cnt(unit, "eager:warm")
            return out
        if entry.state == "dead":
            _cnt(unit, "eager:dead:" + str(entry.err)); return _run(unit, orig, self, dev, a, k)
        if entry.state == "warm":
            return _capture_and_replay(unit, orig, self, args, entry, gen, dev, tensor_args, dict_args, a, k, prime=False)
        return _replay(unit, entry, args, tensor_args, dict_args)


def _capture_and_replay(unit, orig, module, args, entry: _Entry, gen: _Gen, dev, tensor_args, dict_args, a, k, prime: bool):
    t0 = time.perf_counter()
    try:
        _guard_cublas_workspaces(True)
        call = _stage_inputs(entry, args, tensor_args, dict_args, allocate=True)
        if prime:
            _prime(unit, module, call, dev, orig=orig)
        cur = torch.cuda.current_stream(dev)
        side = _side_stream(dev)
        side.wait_stream(cur)                        # ordering only — NO host synchronisation here: the capture is host work (issuing ~80 launches per
        # layer into the capturing stream, nothing executes), so it proceeds while the GPU drains what the trunk already queued on `cur` (the MSA
        # module of this recycle, the prime layer); the workspace clear below and every allocator free are stream-ordered (a block freed on `cur` is
        # reused by `cur` work only, in order; the capture allocates from this generation's private pool), so no in-flight kernel loses memory.
        _clear_cublas_workspaces_now()               # the capture's GEMMs take a workspace allocated INSIDE this generation's pool
        free0 = torch.cuda.mem_get_info(dev)[0]
        g = torch.cuda.CUDAGraph()
        ac_enabled, ac_dtype = torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
        err = None; out = None
        with torch.cuda.stream(side):
            g.capture_begin(gen.pool.id, capture_error_mode="thread_local")
            try:
                with torch.autocast("cuda", dtype=ac_dtype, enabled=ac_enabled, cache_enabled=False):
                    out = _run(unit, orig, module, dev, (), call)
            except Exception as e:   # noqa: BLE001 — end the capture first, decide below
                err = e
            try:
                g.capture_end()
            except Exception as e2:  # noqa: BLE001
                err = err or e2
        if err is not None:
            raise err
        cur.wait_stream(side)
        entry.is_tuple = isinstance(out, tuple)
        entry.static_out = out if entry.is_tuple else (out,)
        entry.graph = g
        entry.capture_s = round(time.perf_counter() - t0, 4)   # host time of prime + staging + capture + instantiate (the GPU may still be draining `cur`)
        growth = round(max(0, free0 - torch.cuda.mem_get_info(dev)[0]) / 2**30, 3)   # driver-level free memory: no synchronisation needed
        STATS["captures"].append({"unit": unit, "tokens": gen.tokens, "capture_s": entry.capture_s, "primed": bool(prime), "pool_growth_gib": growth})
        STATS["pool_gib_max"] = max(STATS["pool_gib_max"], _pool_gib(gen))
        entry.state = "ready"
        _cnt(unit, "captured")
        _log(f"captured {unit} tokens={gen.tokens} prime={prime} in {entry.capture_s}s (+{growth} GiB from the driver; pool {_pool_gib(gen)} GiB)")
    except Exception as e:  # noqa: BLE001 — a capture failure: counted, this signature is dead (eager from now on), the gate refuses
        try:
            torch.cuda.synchronize(dev)
        except Exception:  # noqa: BLE001
            pass
        entry.drop(); entry.state = "dead"; entry.err = type(e).__name__
        ek = unit + ":" + type(e).__name__
        STATS["errors"][ek] = STATS["errors"].get(ek, 0) + 1
        print(f"[boltz_graph_trunk] capture of unit {unit} FAILED ({type(e).__name__}: {(str(e).splitlines() or [''])[0][:240]}) — this signature runs the module's own forward eagerly from now on; the gate refuses",
              file=sys.stderr, flush=True)
        try:
            from opt_core.oom import is_oom
            if is_oom(e):
                raise
        except ImportError:
            pass
        _cnt(unit, "eager:dead:" + entry.err)
        return _run(unit, orig, module, dev, a, k)
    entry.graph.replay()                              # the capturing call's own result (the static inputs hold this call's arguments)
    entry.n_replay += 1
    outs = tuple(o.clone() if torch.is_tensor(o) else o for o in entry.static_out)
    return outs if entry.is_tuple else outs[0]


def _replay(unit, entry: _Entry, args, tensor_args, dict_args):
    _stage_inputs(entry, args, tensor_args, dict_args, allocate=False)
    entry.graph.replay()
    entry.n_replay += 1
    _cnt(unit, "replayed")
    outs = tuple(o.clone() if torch.is_tensor(o) else o for o in entry.static_out)
    return outs if entry.is_tuple else outs[0]


def _pool_bytes(gen: _Gen) -> int:
    try:
        return int(sum(seg["total_size"] for seg in gen.pool.snapshot())) if gen.pool is not None else 0
    except Exception:  # noqa: BLE001
        return 0


def _pool_gib(gen: _Gen) -> float:
    try:
        segs = gen.pool.snapshot()
        return round(sum(int(sg.get("total_size", 0)) for sg in segs) / 2**30, 3)
    except Exception:  # noqa: BLE001
        return -1.0


# ----------------------------------------------------------------------------------------------------------------------
# unit templ: the template module's distogram boundaries, computed by the stock CPU statement once and kept on the device
# ----------------------------------------------------------------------------------------------------------------------
def _templ_boundaries(self, device):
    """torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1) — the stock CPU statement — evaluated once per (module, device);
    the device copy is a bit-copy of those fp32 values (what stock's per-call `.to(device)` produces)."""
    cache = self.__dict__.setdefault("_graph_trunk_boundaries", {})
    t = cache.get(str(device))
    if t is None:
        b = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1)     # CPU, exactly the stock call
        t = b.to(device)
        cache[str(device)] = t
        _cnt("templ", "boundaries_built")
    _cnt("templ", "boundaries_served")
    return t


def _templ_v2_forward(self, z, feats, pair_mask, use_kernels: bool = False):
    _cnt("templ", "served")
    # boltz 2.2.1 boltz/model/modules/trunkv2.py TemplateV2Module.forward, verbatim except `boundaries` (see _templ_boundaries)
    from torch.nn.functional import one_hot
    res_type = feats["template_restype"]
    frame_rot = feats["template_frame_rot"]
    frame_t = feats["template_frame_t"]
    frame_mask = feats["template_mask_frame"]
    cb_coords = feats["template_cb"]
    ca_coords = feats["template_ca"]
    cb_mask = feats["template_mask_cb"]
    visibility_ids = feats["visibility_ids"]
    template_mask = feats["template_mask"].any(dim=2).float()
    num_templates = template_mask.sum(dim=1)
    num_templates = num_templates.clamp(min=1)
    b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
    b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]
    b_cb_mask = b_cb_mask[..., None]
    b_frame_mask = b_frame_mask[..., None]
    B, T = res_type.shape[:2]  # noqa: N806
    tmlp_pair_mask = (visibility_ids[:, :, :, None] == visibility_ids[:, :, None, :]).float()
    with torch.autocast(device_type="cuda", enabled=False):
        cb_dists = torch.cdist(cb_coords, cb_coords)
        boundaries = _templ_boundaries(self, cb_dists.device)
        distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
        distogram = one_hot(distogram, num_classes=self.num_bins)
        frame_rot = frame_rot.unsqueeze(2).transpose(-1, -2)
        frame_t = frame_t.unsqueeze(2).unsqueeze(-1)
        ca_coords = ca_coords.unsqueeze(3).unsqueeze(-1)
        vector = torch.matmul(frame_rot, (ca_coords - frame_t))
        norm = torch.norm(vector, dim=-1, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
        unit_vector = unit_vector.squeeze(-1)
        a_tij = [distogram, b_cb_mask, unit_vector, b_frame_mask]
        a_tij = torch.cat(a_tij, dim=-1)
        a_tij = a_tij * tmlp_pair_mask.unsqueeze(-1)
        res_type_i = res_type[:, :, :, None]
        res_type_j = res_type[:, :, None, :]
        res_type_i = res_type_i.expand(-1, -1, -1, res_type.size(2), -1)
        res_type_j = res_type_j.expand(-1, -1, res_type.size(2), -1, -1)
        a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
        a_tij = self.a_proj(a_tij)
    pair_mask = pair_mask[:, None].expand(-1, T, -1, -1)
    pair_mask = pair_mask.reshape(B * T, *pair_mask.shape[2:])
    v = self.z_proj(self.z_norm(z[:, None])) + a_tij
    v = v.view(B * T, *v.shape[2:])
    v = v + self.pairformer(v, pair_mask, use_kernels=use_kernels)
    v = self.v_norm(v)
    v = v.view(B, T, *v.shape[1:])
    template_mask = template_mask[:, :, None, None, None]
    num_templates = num_templates[:, None, None, None]
    u = (v * template_mask).sum(dim=1) / num_templates.to(v)
    u = self.u_proj(self.relu(u))
    return u


def _templ_v1_forward(self, z, feats, pair_mask, use_kernels: bool = False):
    _cnt("templ", "served")
    # boltz 2.2.1 boltz/model/modules/trunkv2.py TemplateModule.forward, verbatim except `boundaries` (see _templ_boundaries)
    from torch.nn.functional import one_hot
    asym_id = feats["asym_id"]
    res_type = feats["template_restype"]
    frame_rot = feats["template_frame_rot"]
    frame_t = feats["template_frame_t"]
    frame_mask = feats["template_mask_frame"]
    cb_coords = feats["template_cb"]
    ca_coords = feats["template_ca"]
    cb_mask = feats["template_mask_cb"]
    template_mask = feats["template_mask"].any(dim=2).float()
    num_templates = template_mask.sum(dim=1)
    num_templates = num_templates.clamp(min=1)
    b_cb_mask = cb_mask[:, :, :, None] * cb_mask[:, :, None, :]
    b_frame_mask = frame_mask[:, :, :, None] * frame_mask[:, :, None, :]
    b_cb_mask = b_cb_mask[..., None]
    b_frame_mask = b_frame_mask[..., None]
    B, T = res_type.shape[:2]  # noqa: N806
    asym_mask = (asym_id[:, :, None] == asym_id[:, None, :]).float()
    asym_mask = asym_mask[:, None].expand(-1, T, -1, -1)
    with torch.autocast(device_type="cuda", enabled=False):
        cb_dists = torch.cdist(cb_coords, cb_coords)
        boundaries = _templ_boundaries(self, cb_dists.device)
        distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
        distogram = one_hot(distogram, num_classes=self.num_bins)
        frame_rot = frame_rot.unsqueeze(2).transpose(-1, -2)
        frame_t = frame_t.unsqueeze(2).unsqueeze(-1)
        ca_coords = ca_coords.unsqueeze(3).unsqueeze(-1)
        vector = torch.matmul(frame_rot, (ca_coords - frame_t))
        norm = torch.norm(vector, dim=-1, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
        unit_vector = unit_vector.squeeze(-1)
        a_tij = [distogram, b_cb_mask, unit_vector, b_frame_mask]
        a_tij = torch.cat(a_tij, dim=-1)
        a_tij = a_tij * asym_mask.unsqueeze(-1)
        res_type_i = res_type[:, :, :, None]
        res_type_j = res_type[:, :, None, :]
        res_type_i = res_type_i.expand(-1, -1, -1, res_type.size(2), -1)
        res_type_j = res_type_j.expand(-1, -1, res_type.size(2), -1, -1)
        a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
        a_tij = self.a_proj(a_tij)
    pair_mask = pair_mask[:, None].expand(-1, T, -1, -1)
    pair_mask = pair_mask.reshape(B * T, *pair_mask.shape[2:])
    v = self.z_proj(self.z_norm(z[:, None])) + a_tij
    v = v.view(B * T, *v.shape[2:])
    v = v + self.pairformer(v, pair_mask, use_kernels=use_kernels)
    v = self.v_norm(v)
    v = v.view(B, T, *v.shape[1:])
    template_mask = template_mask[:, :, None, None, None]
    num_templates = num_templates[:, None, None, None]
    u = (v * template_mask).sum(dim=1) / num_templates.to(v)
    u = self.u_proj(self.relu(u))
    return u




# ----------------------------------------------------------------------------------------------------------------------
# apply / remove / report
# ----------------------------------------------------------------------------------------------------------------------
def apply(spec: Optional[str] = None) -> Tuple[str, ...]:
    """Install the units named by `spec` (default: env BOLTZ_GRAPH_TRUNK). Idempotent. Returns the tuple of installed units."""
    with _LOCK:
        if _STATE["applied"]:
            return _STATE["units"]
        units = parse_units(os.environ.get(SWITCH) if spec is None else spec)
        _STATE["units"] = units
        if not units:
            return units
        _STATE["min_tokens"] = _int_env(SWITCH + "_MIN_TOKENS", 0)
        _STATE["max_tokens"] = _int_env(SWITCH + "_MAX_TOKENS", 1 << 30)
        _STATE["keep"] = max(1, _int_env(SWITCH + "_KEEP", 2))
        _STATE["policy"] = POLICY
        from boltz.model.layers import pairformer as PF
        from boltz.model.modules import trunkv2 as TR
        O = _STATE["orig"]
        installed: List[str] = []

        def _pinned(key, fn) -> bool:
            sha = _src_sha(fn)
            if SRC_SHA[key] == sha:
                return True
            _STATE["refused"][key] = f"source_mismatch:{sha}"
            return False

        def _body(unit, key, fn) -> bool:
            """A unit whose method may carry one of several known bodies (BODIES): the traits of the one installed, or refused by name."""
            sha = _src_sha(fn)
            traits = BODIES[key].get(sha)
            if traits is None:
                _STATE["refused"][key] = f"source_mismatch:{sha}"
                return False
            _STATE["body"][unit] = dict(traits, sha=sha)
            return True

        if "pf" in units and _body("pf", "PairformerModule.forward", PF.PairformerModule.forward):
            O["PairformerModule.forward"] = PF.PairformerModule.forward
            PF.PairformerModule.forward = _graphed("pf", O["PairformerModule.forward"], ("s", "z", "mask", "pair_mask", "use_kernels"), {"use_kernels": False},
                                                   ("s", "z", "mask", "pair_mask"), (), lambda b: b["z"].shape[1])
            installed.append("pf")
        if "pfnoseq" in units and _body("pfnoseq", "PairformerNoSeqModule.forward", PF.PairformerNoSeqModule.forward):
            O["PairformerNoSeqModule.forward"] = PF.PairformerNoSeqModule.forward
            PF.PairformerNoSeqModule.forward = _graphed("pfnoseq", O["PairformerNoSeqModule.forward"], ("z", "pair_mask", "use_kernels"), {"use_kernels": False},
                                                        ("z", "pair_mask"), (), lambda b: b["z"].shape[1])
            installed.append("pfnoseq")
        if "msa" in units and _pinned("MSAModule.forward", TR.MSAModule.forward):
            O["MSAModule.forward"] = TR.MSAModule.forward
            TR.MSAModule.forward = _graphed("msa", O["MSAModule.forward"], ("z", "emb", "feats", "use_kernels"), {"use_kernels": False},
                                            ("z", "emb"), ("feats",), lambda b: b["z"].shape[1])
            installed.append("msa")
        if "sovl" in units:
            if "pf" not in installed:
                _STATE["refused"]["sovl"] = "requires_unit_pf"
            elif _pinned("PairformerLayer.forward", PF.PairformerLayer.forward):
                _STATE["sovl"] = True; installed.append("sovl")
        if "templ" in units:
            for cls, fn, key in ((TR.TemplateV2Module, _templ_v2_forward, "TemplateV2Module.forward"), (TR.TemplateModule, _templ_v1_forward, "TemplateModule.forward")):
                if not _pinned(key, cls.forward):
                    continue
                O[key] = cls.forward
                cls.forward = fn
                _STATE["templ_patched"].append(key)
            if _STATE["templ_patched"]:
                installed.append("templ")
        _STATE["units"] = tuple(installed)
        _STATE["applied"] = True
        _STATE["wanted"] = tuple(units)
        for name in list(_REGISTERED):
            _install_registered(name)
        _STATE["t_apply"] = time.time()
    print(f"[boltz_graph_trunk] APPLIED units={','.join(_STATE['units']) or 'none'} policy={_STATE['policy']} min_tokens={_STATE['min_tokens']} max_tokens={_STATE['max_tokens']} keep={_STATE['keep']}"
          + (f" refused={_STATE['refused']}" if _STATE["refused"] else ""), file=sys.stderr, flush=True)
    return _STATE["units"]


def remove() -> None:
    with _LOCK:
        if not _STATE["applied"]:
            return
        from boltz.model.layers import pairformer as PF
        from boltz.model.modules import trunkv2 as TR
        O = _STATE["orig"]
        for key, fn in O.items():
            cls_name, meth = key.split(".")
            cls = _EXTRA_CLASSES.get(key) or getattr(PF, cls_name, None) or getattr(TR, cls_name, None)
            setattr(cls, meth, fn)
        _EXTRA_CLASSES.clear()
        O.clear()
        release_all()
        for sv in _SOVL_OBJS:
            sv.remove()
        _SOVL_OBJS.clear()
        _guard_cublas_workspaces(False)
        _STATE["applied"] = False; _STATE["units"] = (); _STATE["templ_patched"] = []; _STATE["sovl"] = False


def census() -> Dict[str, Dict[str, int]]:
    return {u: dict(sorted(c.items())) for u, c in STATS["census"].items()}


def verdict() -> Dict[str, Any]:
    """Fail-closed: refused when nothing is applied, when a unit named by the row was refused at apply (source mismatch), on any capture
    error (a dead signature), or when the graph units received calls inside the token range and none was ever replayed. Idle (ok) when every
    call took a declared eager word (token range / no cuda) — installed behind its gate, never engaged."""
    if not _STATE["applied"] or not _STATE["units"]:
        return {"ok": False, "idle": False, "reason": "not applied"}
    if _STATE["refused"]:
        return {"ok": False, "idle": False, "reason": "refused_at_apply:" + ",".join(f"{k}={v}" for k, v in sorted(_STATE["refused"].items()))}
    if STATS["errors"]:
        return {"ok": False, "idle": False, "reason": "capture_error:" + ",".join(sorted(STATS["errors"]))}
    missing = [u for u in _STATE.get("wanted", ()) if u not in UNITS and u not in _STATE["units"]]
    if missing:                                   # the row lists a unit another lever was to register and it never did: named, never silent
        return {"ok": False, "idle": False, "reason": "listed_not_registered:" + ",".join(missing)}
    graph_units = [u for u in _STATE["units"] if u not in ("templ", "sovl")]
    c = STATS["census"]
    sk = {w: n for w, n in c.get("sovl", {}).items() if w.startswith("skipped:")}
    if _STATE.get("sovl") and sk:
        return {"ok": False, "idle": False, "reason": "sovl_skipped:" + ",".join(f"{w[8:]}={n}" for w, n in sorted(sk.items()))}
    served = sum(c.get(u, {}).get("replayed", 0) + c.get(u, {}).get("captured", 0) for u in graph_units)
    in_range = sum(n for u in graph_units for w, n in c.get(u, {}).items() if w in ("eager:warm", "captured", "replayed") or w.startswith("eager:dead"))
    if graph_units and in_range > 0 and served == 0:
        calls = sum(n for u in graph_units for w, n in c.get(u, {}).items() if not w.startswith("primed"))
        if calls > len(graph_units) * 2:            # more than a warm call or two per unit and still nothing replayed: engaged, never served
            return {"ok": False, "idle": False, "reason": "engaged_never_replayed"}
    if graph_units and in_range == 0:
        return {"ok": True, "idle": True, "reason": "idle: every call took a declared eager word (token range / no cuda) — installed, never engaged"}
    return {"ok": True, "idle": False, "reason": None}


def report() -> Dict[str, Any]:
    resident = [{"tokens": g.tokens, "graphs": len([e for e in g.entries.values() if e.state == "ready"]), "pool_gib": _pool_gib(g)} for g in _GENS]
    return {"applied": list(_STATE["units"]) if _STATE["applied"] else [], "switch": SWITCH, "spec": os.environ.get(SWITCH), "policy": _STATE["policy"],
            "min_tokens": _STATE["min_tokens"], "max_tokens": _STATE["max_tokens"], "keep": _STATE["keep"], "census": census(), "captures": list(STATS["captures"]),
            "capture_s_total": round(sum(c["capture_s"] for c in STATS["captures"]), 3), "errors": dict(STATS["errors"]),
            "generations_evicted": STATS["generations_evicted"], "resident": resident, "pool_gib_max": STATS["pool_gib_max"], "static_gib": STATS["static_gib"],
            "prime_rng_checks": STATS["prime_rng_checks"], "templ_patched": list(_STATE["templ_patched"]), "refused": dict(_STATE["refused"]),
            "src_sha": dict(SRC_SHA), "bodies": {u: b.get("name") for u, b in _STATE["body"].items()}, "mask_as_none": STATS.get("mask_as_none", 0), "mask_predicate": STATS.get("mask_predicate", 0), "primed_missing": STATS.get("primed_missing", 0), "registered": sorted(_REGISTERED), "gate": verdict(), "torch": torch.__version__}
