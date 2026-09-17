"""[RFD3_HOIST] reuse of step-invariant sub-graphs inside one RFdiffusion3 sampler roll-out.

RFD3_HOIST=1  -> within one call of sample_diffusion_like_af3 (one design x diffusion batch, num_timesteps-1 denoiser calls, each with
                 n_recycle passes), tensors whose free variables are only the roll-out constants {f (static features), Q_L_init, C_L,
                 P_LL, S_I (initializer outputs)} and module parameters are computed ONCE by the unchanged upstream modules and reused:
                   * LocalAttentionPairBias.to_b(P_LL) in the 3 atom-encoder and 3 atom-decoder blocks (the dense L x L x H pair bias;
                     the same Linear on the same P_LL every call), and
                   * RFD3DiffusionModule.downcast_c(C_L, S_I) (cross-attention pooling of the initial atom features).
                 Everything that reads X_noisy_L, t, the recycled X_L / distogram, or the token pair track Z_II (rebuilt every call from
                 coordinates) is NOT touched. Default 0 = previous code path, byte for byte.
The cache is opened by the sampler at the start of a roll-out and dropped (after a device sync) at its end, so nothing can leak across
designs/batches; outside a roll-out hoist_get(key, fn) is just fn().
"""
import os

import torch

HOIST = os.environ.get("RFD3_HOIST", "0").strip().lower() in ("1", "true", "on")
# The five exact reuses RFD3_HOIST=1 enables (PARTS; each call site names its part):
#   pll      : LocalAttentionPairBias.to_b(P_LL) once per roll-out (atom encoder/decoder blocks)
#   downcast : RFD3DiffusionModule.downcast_c(C_L, S_I) once per roll-out
#   valid    : the dense (D, L, L) key mask of dense_sdpa_pairbias_attention once per denoiser call per index set (it is rebuilt from the
#              same f["attn_indices"] by every one of the 9 atom-block calls, and by the 18 token blocks per recycle)
#   where    : dense bias masking as ONE torch.where pass instead of clone-of-expand + masked_fill_ (pure data movement; identical values)
#   dedup    : the masked dense bias (bias x key-mask) of a block is identical for the n_recycle passes of one denoiser call (same hoisted bias,
#              same indices) -> built once per call per block (costs D*H*L*L*2 bytes held per atom-decoder block until the call ends)
PARTS = frozenset(("pll", "downcast", "valid", "where", "dedup"))
CALL_CACHE = {}  # per-denoiser-call cache (valid masks), opened/cleared by RFD3DiffusionModule.forward

import functools  # noqa: E402
import logging  # noqa: E402

from opt_core.capture import compile as _core_compile  # the shared core's torch.compile plumbing (boundary marker, installer, census)  # noqa: E402

logger = logging.getLogger(__name__)

# [RFD3_COMPILE] inductor fusion under the roll-out cache. RFD3_COMPILE=1 -> at the first roll-out the four diffusion submodules upstream's own
# compile switch targets (rfd3.engine.RFD3InferenceEngine._COMPILE_TARGETS: encoder, diffusion_token_encoder, diffusion_transformer, decoder)
# are wrapped in torch.compile as engine.py does (shape treatment: COMPILE_DYNAMIC below), on the module the sampler receives (the one that runs). The cache
# accessors of this module, the CUDA-graph sampler's memo helpers, the attention-path decision, the attention-index builder and the fused
# transition are dynamo boundaries (`nodynamo`: torch.compiler.disable when RFD3_COMPILE=1, the identity otherwise): they run eagerly between
# compiled segments, so dynamo never guards on cache state and never re-traces the shared core's Triton kernel. Under RFD3_CUDAGRAPH=1 the
# compiled step is what the graph captures (dynamo's guards and inductor's launches run at warm-up and capture; replay is kernels only).
# Numerics: inductor's, not eager's -> a tolerance-tier lever (mode fast), never part of exact. Default 0 = nothing below is wrapped.
COMPILE = os.environ.get("RFD3_COMPILE", "0").strip().lower() in ("1", "true", "on")
# [RFD3_TOKEN_SDPA] read here with the other lever names (layers/attention.py routes on it): the 18-block token transformer's pair-bias
# attention (upstream: full=True -> pairbias_attention_, einsum + masked_fill + softmax) runs through upstream's dense_sdpa_pairbias_attention
# (F.scaled_dot_product_attention over the same key set, -inf-masked additive bias). Tolerance tier (SDPA's summation order), mode fast.
TOKEN_SDPA = os.environ.get("RFD3_TOKEN_SDPA", "0").strip().lower() in ("1", "true", "on")

# The boundary marker and the installer are the shared core's (opt_core.capture.compile, strategy F3.torch_compile); this module binds them to
# the kit's switch and to upstream's target list. `nodynamo` = torch.compiler.disable iff RFD3_COMPILE=1, the function itself otherwise.
nodynamo = functools.partial(_core_compile.nodynamo, active=COMPILE)
# Shape treatment: torch.compile(dynamic=None), torch's automatic dynamic shapes — the first item compiles static kernels; the first item of a
# DIFFERENT shape recompiles once with the changed dims symbolic, and that compile serves every later shape (a run over one length keeps static
# kernels, a run over several lengths compiles twice instead of once per shape; upstream's own compile switch uses dynamic=False, one compile per
# distinct shape). The CUDA graph is captured per shape either way (RFD3_CUDAGRAPH keys its entries on the shape).
COMPILE_DYNAMIC = None                                                                # torch.compile's `dynamic`: None = automatic dynamic shapes (the core prints it `auto`)
COMPILE_RECORD = _core_compile.CompileRecord("RFD3_COMPILE", dynamic=COMPILE_DYNAMIC, cache_size_limit=64, log=logger.info) if COMPILE else None
# The frames dynamo attempts and does NOT convert in this model (census `frames_skipped` = frames_total - frames_ok; they run eager), named: the
# allowance the compile gate accepts, counted on the pinned stack (torch 2.13) — the same count at every length and batch under the automatic
# shape treatment. Which frames and why: `rfdiffusion3/CHANGES.md` "Levers" (RFD3_COMPILE row). A count above it (a torch / dynamo other than the
# pinned one) is NAMED by the package (`COMPILE frames_skipped=<n> above the <allowed> measured …`), never a reason to call the lever off.
COMPILE_FRAMES_SKIPPED = {"auto": 11}
# cache_size_limit=64: one attention code object serves the atom encoder, the atom decoder and the token blocks at their own static shapes, plus one
# resume function per boundary — dynamo's default budget of 8 would fall back to eager silently past it (the record's gate fails on any such hit)


def compile_install(diffusion_module):
    """[RFD3_COMPILE] wrap upstream's compile targets of `diffusion_module` (rfd3.engine.RFD3InferenceEngine._COMPILE_TARGETS, read from the
    class, never restated) in torch.compile once per process (automatic dynamic shapes), on the module the sampler receives. No-op unless RFD3_COMPILE=1."""
    if COMPILE_RECORD is None or diffusion_module is None or COMPILE_RECORD.stats_["installs"]:
        return 0
    from rfd3.engine import RFD3InferenceEngine

    return COMPILE_RECORD.install([(diffusion_module, name) for name in RFD3InferenceEngine._COMPILE_TARGETS])


def compile_describe():
    """What RFD3_COMPILE did in this process: the switch plus the core record's census (installs, wrapped/declared targets, dynamo graphs,
    frames ok/total/skipped, graph breaks, recompile-limit hits, per-code cache entries against the budget, compile seconds) and the record's
    gate sentences — zeros and an empty gate when the lever is off."""
    d = {"RFD3_COMPILE": COMPILE, "installs": 0, "wrapped": 0, "declared": 0, "graphs": 0, "graph_breaks": 0, "frames_ok": 0, "frames_total": 0,
         "frames_skipped": 0, "recompile_limit_hits": 0, "cache_entries_max": None, "cache_size_limit": None, "at_limit": [], "compile_s": 0.0,
         "dynamic": _core_compile.dynamic_word(COMPILE_DYNAMIC), "frames_skipped_allowed": None, "targets": "-", "gate": ""}
    if COMPILE_RECORD is not None:
        c = COMPILE_RECORD.census()
        d.update({k: c[k] for k in d if k in c and k not in ("RFD3_COMPILE",)})
        d["targets"] = ",".join(c["wrapped_names"]) or "-"
        # The gate: every declared target wrapped, compiled frames ran, NO recompile-limit hit, no code object at the cache budget (else the census
        # word is a named fallback). Frames dynamo attempted and did not convert are NOT gated: their count rides on the line beside the allowance
        # (frames_skipped_allowed) and the package names an excess. Graph breaks are not gated: the kit's dynamo boundaries ARE graph breaks.
        d["frames_skipped_allowed"] = COMPILE_FRAMES_SKIPPED.get(d["dynamic"])
        d["gate"] = "; ".join(COMPILE_RECORD.gate(require_frames=True, max_frames_skipped=None))
    return d


def compile_lever_line(tag):
    """The core's LEVER census line for RFD3_COMPILE (`[<tag>] LEVER name=RFD3_COMPILE state=on|skipped impl=torch_compile origin=core
    strategy=F3.torch_compile compiled=<n> declared=<n> … graphs=… frames=… graph_breaks=… recompile_limit_hits=…`), or None with the lever off."""
    return COMPILE_RECORD.line(tag, "RFD3_COMPILE") if COMPILE_RECORD is not None else None


_CONST_IDS = set()
_CONST_PINS = []


def mark_const(*tensors):
    """Called at the top of every denoiser call with the roll-out-constant arguments (Q_L_init, C_L, P_LL, S_I as received from the
    sampler). Only tensors registered here are eligible for the 'pll' hoist at the shared attention call site - the same source line also
    serves the token-level transformer, whose pair input is the per-call Z_II track and must never be cached."""
    global _CONST_IDS, _CONST_PINS
    _CONST_PINS = [t for t in tensors if torch.is_tensor(t)]  # keep alive for the duration of the call so ids cannot be recycled
    _CONST_IDS = {id(t) for t in _CONST_PINS}


@nodynamo
def is_const(t):
    return id(t) in _CONST_IDS


def part(name):
    return HOIST and name in PARTS


def call_begin():
    CALL_CACHE.clear()
    _DEDUP_ORDER.clear()


def call_end():
    CALL_CACHE.clear()
    _DEDUP_ORDER.clear()


DEDUP_CAPACITY = 3  # = number of atom-decoder blocks (the only blocks re-entered within a call)
_DEDUP_ORDER = []


@nodynamo
def call_get_fifo(key, pin, fn, part_name="dedup"):
    """Like call_get but keeps at most DEDUP_CAPACITY entries of this family (FIFO): the atom-encoder entries (never re-read) are evicted by
    the decoder's, which are then re-read on the next recycle. Bounded extra memory = DEDUP_CAPACITY masked-bias tensors."""
    if not part(part_name):
        return fn()
    pins = pin if isinstance(pin, (tuple, list)) else (pin,)
    k = (key,) + tuple(id(t) for t in pins)
    hit = CALL_CACHE.get(k)
    if hit is not None:
        return hit[1]
    v = fn()
    CALL_CACHE[k] = (pins, v)
    _DEDUP_ORDER.append(k)
    while len(_DEDUP_ORDER) > DEDUP_CAPACITY:
        CALL_CACHE.pop(_DEDUP_ORDER.pop(0), None)
    return v


@nodynamo
def call_get(key, pin, fn, part_name="valid"):
    """Per-denoiser-call memo: value of fn() for (key, identity of the `pin` tensor(s)). The pins (e.g. the indices tensor) are held in the
    entry so their ids cannot be recycled within the call. Outside RFD3_HOIST (or the part switched off) -> fn()."""
    if not part(part_name):
        return fn()
    pins = pin if isinstance(pin, (tuple, list)) else (pin,)
    k = (key,) + tuple(id(t) for t in pins)
    hit = CALL_CACHE.get(k)
    if hit is not None:
        return hit[1]
    v = fn()
    CALL_CACHE[k] = (pins, v)
    return v

HOIST_CACHE = None  # dict during a roll-out iff HOIST, else None (eager / non-graph path)
_THUNKS = {}  # key -> the fn seen at cache fill (graph path: closes over the graph entry's static input buffers, so it stays valid)
HOIST_STATS = {"rollouts": 0, "entries_last": 0, "hits_last": 0}
_HITS = 0


def hoist_begin():
    """Open a per-roll-out cache (iff RFD3_HOIST)."""
    global HOIST_CACHE, _HITS
    HOIST_CACHE = {} if HOIST else None
    _HITS = 0


def hoist_end():
    global HOIST_CACHE
    if HOIST_CACHE is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        HOIST_STATS["rollouts"] += 1
        HOIST_STATS["entries_last"] = len(HOIST_CACHE)
        HOIST_STATS["hits_last"] = _HITS
    HOIST_CACHE = None


def _graph_entry():
    """The partner CUDA-graph entry being recorded/captured (RFD3_CUDAGRAPH=1), else None."""
    try:
        from rfd3.model import cudagraph_sampler as _cg
        if _cg.graph_mode() is None:
            return None
        return _cg._STATE.get("entry")
    except Exception:
        return None


def signature_token():
    """Folded into the CUDA-graph entry signature (cudagraph_sampler._signature): an entry captured with one hoist configuration is never replayed under another."""
    return f"hoist={int(bool(HOIST))};parts={','.join(sorted(PARTS)) if HOIST else '-'};"


def hoist_refresh(entry):
    """Graph path, entry REUSED for a new roll-out (same shape signature; the graph sampler has already copy_()ed the new f / initializer outputs into
    entry.static_f / entry.static_io): recompute every roll-out-cached tensor with the stock sub-graph on the refreshed static inputs and
    copy_ it IN PLACE into the buffer the captured graph reads. The thunks stored at capture time close over the entry's static input
    tensors (the denoiser was called with **entry.static_io during warm-up/capture), so calling them again now yields the values for the
    new roll-out. Runs under the autocast state recorded at capture. Returns the number of refreshed entries (0 without a cache)."""
    h = getattr(entry, "hoist", None)
    if not h:
        return 0
    thunks = getattr(entry, "hoist_thunks", None) or {}
    n = 0
    ac_on = getattr(entry, "autocast_on", True)
    ac_dtype = getattr(entry, "autocast_dtype", None) or torch.bfloat16
    with torch.no_grad(), torch.autocast("cuda", enabled=bool(ac_on), dtype=ac_dtype, cache_enabled=False):
        for key, buf in h.items():
            fn = thunks.get(key) or _THUNKS.get(key)
            if fn is None:
                raise RuntimeError(f"[RFD3_HOIST] no thunk to refresh cached entry {key!r} on graph-entry reuse")
            fresh = fn()
            if isinstance(buf, (tuple, list)):
                for a, b in zip(buf, fresh):
                    a.copy_(b)
            else:
                buf.copy_(fresh)
            n += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return n


@nodynamo
def hoist_get(key, fn):
    """cache[key] if present else fn() (stored). No roll-out in flight -> fn().
    Graph path (RFD3_CUDAGRAPH=1 record/capture): the cache is owned by the graph entry (entry.hoist) so the captured graph's input
    buffers stay alive with the entry; see hoist_refresh for entry reuse."""
    global _HITS
    if not HOIST or (isinstance(key, tuple) and len(key) > 2 and key[2] not in PARTS):
        return fn()
    e = _graph_entry()
    if e is not None:
        h = getattr(e, "hoist", None)
        if h is None:
            h = e.hoist = {}
        v = h.get(key)
        if v is None:
            with torch.autocast("cuda", enabled=torch.is_autocast_enabled("cuda"), dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False):
                v = fn()
            h[key] = v
            _THUNKS[key] = fn
            if getattr(e, "hoist_thunks", None) is None:
                e.hoist_thunks = {}
            e.hoist_thunks[key] = fn
            return v
        _HITS += 1
        return v
    c = HOIST_CACHE
    if c is None:
        return fn()
    v = c.get(key)
    if v is None:
        v = fn()
        c[key] = v
        return v
    _HITS += 1
    return v


def describe():
    return {"RFD3_HOIST": HOIST, "parts": ",".join(sorted(PARTS)), "RFD3_COMPILE": COMPILE, "RFD3_TOKEN_SDPA": TOKEN_SDPA}

