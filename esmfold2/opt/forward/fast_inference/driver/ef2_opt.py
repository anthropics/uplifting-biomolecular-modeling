"""ESMFold2 inference-execution levers (no change to model math / inputs / seeds / steps).

install(model, trunk_graphs=True, sampler_graphs=True, fuse_msa_trimul=False, probe=False, ..., loop_static=False, recycle_graph=False)

Levers
------
G1  CUDA-graph capture+replay of every FoldingTrunk call (48/24-layer pair trunk x 11 loop iterations, lm_encoder,
    parcae coda, confidence-head trunk). One graph per (module, input shape, dtype); LRU per module. The captured
    kernels are exactly the kernels the eager path launches (same backend, same chunking), replayed with new data.
G2  Re-implementation of DiffusionStructureHead.sample with (a) the per-step host syncs removed (schedule / gamma
    values computed once as Python floats -> no .item() per step) and (b) the DiffusionModule call for steps >= 1
    captured in a CUDA graph and replayed (step 0 stays eager: it fills the module's inference cache and contains the
    data-dependent host syncs). RNG consumption order and every tensor op on the sampling path are unchanged, so with
    the same seed the noise stream is identical to the stock sampler.
M1  (optional, numerics-changing like the vendor 'fused' backend) route the MSA-encoder triangle updates through the
    vendored fused TriMul+residual Triton kernel (stock set_kernel_backend('fused') leaves the MSA encoder on the
    reference einsum path).
G4  loop_static: the kit's re-issue of ESMFold2Model._run_one_loop (the trunk recycling loop), statement by statement,
    with the graphs' static I/O used in place: the trunk input `a*z + linear(...)` is written by the same add kernel INTO
    the trunk graph's static input (no copy-in at replay); the graphed trunk / MSA encoder (and, when no MSA encoder runs
    between, the LM encoder) hand back their static outputs without the per-call clone, each consumed before any other
    graph of the shared pool replays (the trunk's `a*z` is formed right after its replay: a graph's output block is only
    intact until ANOTHER graph of the generation's pool replays); the MSA encoder's loop-invariant inputs are staged once
    per fold; upstream's `lm_z.to(dtype)` cast is written straight into the LM-encoder graph's static input. Bitwise.
G5  recycle_graph: ONE CUDA graph for the deterministic body of a recycle (LM encoder + MSA encoder + injection + folding
    trunk) per shape signature; the RNG prologue (lm dropout, MSA row subsample) runs eagerly in upstream's order and is
    staged into statics; the loop-carried pair state lives in a static the graph updates in place. Recycle 1 of a new
    signature is issued eagerly (warm-up), recycle 2 captures, the rest replay; later folds replay all recycles. For pair
    planes <= recycle_graph_max_tokens. Bitwise.
Per-site graph budgets: graph_budget_tokens_{trunk,encoder,sampler} (-1 inherit the global budget, 0 no cap, N tokens).
Capture policy: EF2_GRAPH_CAPTURE=0 = no CUDA graph at any site (every graph lever runs its eager forward, same kernels: the
memory line's word); 1 = the budgets decide.
probe=True runs the eager path next to every graph replay and records max|diff| (expect exactly 0.0).
"""
import collections, math, os, sys, time, types
import torch
import torch.nn.functional as F

STATS = collections.Counter()
VERIFY_LOG = []


class _Cfg:
    enabled = True
    lru_trunk = int(os.environ.get("EF2_GRAPH_LRU_TRUNK", "1"))      # graphs per trunk module (distinct shapes/dtypes); folding_trunk sees 1 shape per design
    lru_sampler = int(os.environ.get("EF2_GRAPH_LRU_SAMPLER", "8"))     # sampler graphs kept per generation (never evicted individually)
    warmup = 1
    probe = False
    graph_lm_encoder = False
    esmc_cache_ok = True
    lru_generic = int(os.environ.get("EF2_GRAPH_LRU_GENERIC", "3"))
    lru_pair_bias = int(os.environ.get("EF2_PAIR_BIAS_LRU", "1"))     # static bias buffers per attention block (1 = current token shape only)   # msa_encoder sees 2 shapes per design when depth < max (no: subsample keeps M fixed) ; lm_encoder 1
    # per-shape graph budget (MEM lever): a fold whose pair token count exceeds this runs the trunk / encoder / sampler EAGERLY (no capture, no
    # private pool, no static clones — the same kernels, so the numerics are the eager path's = the captured path's) and counts the event;
    # 0 = no budget (every shape captured). The graph pools hold the captured region's whole working set for the life of the generation, which
    # is what pushes the kit over the card where the stock fits at ~1000 tokens (H100 80 GB, 5 diffusion samples).
    graph_budget_tokens = int(os.environ.get("EF2_GRAPH_BUDGET_TOKENS", "0"))
    # batched over-budget regions (S > 1: the confidence head's trunk on [S, L, L, C]) run eagerly; the captured generation is KEPT beside them
    # when memory allows (_batched_region_keeps_generation) instead of being released every fold. 1 = release it unconditionally.
    batched_release = bool(int(os.environ.get("EF2_GRAPH_BATCHED_RELEASE", "0")))
    batched_keep_factor = 4.0            # eager working set of the batched pair trunk ~ factor x its [S, L, L, C] input bytes (fp32 pair 256: 4 x = the two
    batched_keep_margin_gib = 2.0        # triangle-update operand planes + the transition's 4C hidden at bf16 + slack), plus this margin
    # bias_f32 (k-diffusion, exact-safe): keep an fp32 twin of each cached [B,H,Q,K] pair-bias buffer, refreshed in place whenever the bf16
    # buffer is recomputed (the fold's eager step 0), so the SDPA of every block reads it directly instead of casting bf16 -> fp32 at every
    # step (the cast's values ARE the twin's values: `.to(float32)` of the same buffer; a static buffer, graph-address-stable like `buf`).
    bias_f32 = False
    # per-site budgets (trunk graphs G1 incl. coda / confidence trunk and the G5 recycle graph, encoder graphs G3, sampler graphs G2):
    # -1 = the site inherits graph_budget_tokens; 0 = no cap for the site; N = the site runs eagerly above N plane tokens. Lets a mode keep the
    # sampler graphs (positive at every N) while the trunk / encoder graphs (neutral-to-negative at large N, and the large pools) stop earlier.
    graph_budget_tokens_trunk = int(os.environ.get("EF2_GRAPH_BUDGET_TOKENS_TRUNK", "-1"))
    graph_budget_tokens_encoder = int(os.environ.get("EF2_GRAPH_BUDGET_TOKENS_ENCODER", "-1"))
    graph_budget_tokens_sampler = int(os.environ.get("EF2_GRAPH_BUDGET_TOKENS_SAMPLER", "-1"))
    # G4 (loop_static): the kit's re-implementation of ESMFold2Model._run_one_loop stages the trunk's input straight into the trunk graph's static
    # input buffer (torch.add(a*z, F.linear(...), out=static_in): the same add kernel, another destination) and consumes the graphs' static OUTPUT
    # buffers in place inside the loop (no clone per replay; one clone when the loop returns), and stages the MSA encoder's loop-invariant inputs
    # once per fold. Same kernels, same order: bitwise. Off unless install(loop_static=True).
    loop_static = False
    # G5 (recycle_graph): ONE graph for the deterministic body of a trunk recycle (LM encoder + MSA encoder + injection + folding trunk), the
    # per-recycle RNG prologue (lm dropout, MSA row subsample) issued eagerly in upstream's order and staged into static buffers, the
    # loop-carried pair state resident in a static buffer the graph updates in place. Host work per recycle: the prologue's few launches + one
    # replay. Engaged for pair planes up to recycle_graph_max_tokens (small N is where the loop is launch-bound); larger shapes take G4/G1.
    recycle_graph = False
    recycle_graph_max_tokens = int(os.environ.get("EF2_RECYCLE_GRAPH_MAX_TOKENS", "512"))   # G5's ceiling (pair plane tokens): above it a recycle takes G4/G1 (the loop is no longer launch-bound there)
    # the capture POLICY: "0" = no CUDA graph is captured at ANY site — the trunk / coda / confidence-trunk graphs (G1), the step-graph
    # sampler (G2), the encoder graphs (G3), the recycle graph (G5) and ef2_dit's roll-out (it asks _over_graph_budget('sampler', L)) all run
    # their EAGER forwards: the same kernels in the same order (capture vs eager is bitwise), no private pool, no static clones, whatever the
    # budgets say. The memory line exports it (esmfold2_opt.modes: big — no CUDA graphs on the memory line); "1" (the default) = the per-shape /
    # per-site budgets above decide, as under exact / fast. Printed once per site as a named event (STATS graph_capture_off_<site>).
    graph_capture = os.environ.get("EF2_GRAPH_CAPTURE", "1").strip().lower() not in ("0", "off", "no", "false", "eager")
    _body_eager = False        # internal: set while G5 issues / captures a recycle body so the per-module graph wrappers run their eager forwards


CFG = _Cfg()
_POOL = {"handle": None}
_BUDGET_SEEN = set()


def _capture_off(where):
    """The capture policy's verdict for a site: True (= run eagerly) when CFG.graph_capture is off (EF2_GRAPH_CAPTURE=0, the memory line), counted
    per site (STATS graph_capture_off_<where>) and printed once per site — a named event, never a silent fallback; False when the budgets decide."""
    if CFG.graph_capture:
        return False
    STATS["graph_capture_off_" + where] += 1
    if ("capture_off", where) not in _BUDGET_SEEN:
        _BUDGET_SEEN.add(("capture_off", where))
        print(f"[ef2_opt] graph capture: off (EF2_GRAPH_CAPTURE=0) -> {where} runs eagerly (no capture, no pool, no static clones; same kernels)", flush=True)
    return True


def plane_tokens(tokens, batch=1):
    """The token count a pair region is budgeted as: a [B, L, L, C] pair holds the plane of ONE pair of ceil(L * sqrt(B)) tokens (B * L * L
    elements), so a batched region — the confidence head's trunk over the S diffusion samples of a fold runs on [S, L, L, C] — meets the
    budget at the size its memory is, not at its L. B = 1 is L itself."""
    batch = max(1, int(batch))
    return int(tokens) if batch == 1 else int(math.ceil(int(tokens) * math.sqrt(batch)))


def _over_graph_budget(where, tokens, batch=1, plane_bytes=None):
    """True when the per-shape graph budget refuses capture for this fold (CFG.graph_budget_tokens > 0 and the region's plane tokens above
    it): counted per site (STATS graph_budget_eager_<where>) and printed once per (site, token count) with the gate's reason — a named event,
    never a silent fallback. The DECISION is the release tree's one capture gate (opt_core.mem.graph_gate.decide: budget 0 = no cap; else
    cap = budget) on plane_tokens(tokens, batch). A BATCHED region over the budget (batch > 1: the confidence head's trunk on S samples, once
    per fold at S > 1) runs eagerly beside the captured generation when the device has room for it (_batched_region_keeps_generation:
    STATS graph_budget_batched_kept_<where>; the generation's graphs replay on the next fold) and returns the generation's graph pools first
    when it has not (clear_graphs, STATS graph_clears_budget_<where>; the next fold captures afresh — an unconditional release would, at
    S > 1, re-capture every graph of the kit on every fold; EF2_GRAPH_BATCHED_RELEASE=1 selects it). The capture POLICY comes first: with
    CFG.graph_capture off (EF2_GRAPH_CAPTURE=0, the memory line) every site is eager whatever its budget or token count (_capture_off)."""
    if _capture_off(where):                                       # the policy: no CUDA graph at any site (the memory line) — eager, named once per site
        return True
    b = _site_budget(where)
    if tokens is None or not b:                                   # no budget configured (0, the default outside configs/h100.env): nothing to decide, no core import
        return False
    from opt_core.mem.graph_gate import decide
    eff = plane_tokens(tokens, batch)
    gate = decide(int(eff), int(b), name=where)
    if gate.capture:
        return False
    STATS["graph_budget_eager_" + where] += 1
    released = ""
    if int(batch) > 1 and (_POOL["handle"] is not None or _SAMPLER_POOL["handle"] is not None):   # a generation has captured beside a BATCHED over-budget region (S > 1: the confidence head's trunk on [S, L, L, C])
        keep, why = _batched_region_keeps_generation(plane_bytes, int(batch))
        if keep:                                                  # the generation STAYS (its graphs replay on the next fold: no per-fold re-capture of trunk / encoder / sampler graphs at S > 1)
            STATS["graph_budget_batched_kept_" + where] += 1
            released = f" generation=kept({why})"
        else:                                                     # not enough free memory beside the region (or EF2_GRAPH_BATCHED_RELEASE=1): the pools go back first
            clear_graphs(reason="budget_" + where)
            released = f" pools=released({why})"
    cat = released.split("(")[0] + ("" if "avail=" in released else released)   # printed once per (site, token count, decision kind): kept | released for memory | released by word / unknown size
    if (where, eff, cat) not in _BUDGET_SEEN:
        _BUDGET_SEEN.add((where, eff, cat))
        more = "" if int(batch) <= 1 else f" batch={int(batch)} pair_tokens={int(tokens)}{released}"
        print(f"[ef2_opt] graph budget: {where} tokens={eff} > EF2_GRAPH_BUDGET_TOKENS={b} -> eager (no capture; same kernels) gate={gate.reason}{more}", flush=True)   # b: the site's effective budget
    return True


def _batched_region_keeps_generation(plane_bytes, batch):
    """The memory policy of a BATCHED over-budget region: keep the captured generation beside it when the device has room for the region's
    eager working set, else release it first. The working set is estimated as CFG.batched_keep_factor x the
    batched pair tensor's bytes (the pair trunk's live temporaries are a small multiple of its [S, L, L, C] input) plus CFG.batched_keep_margin_gib;
    `free` is the driver's free memory plus the caching allocator's unused reserve. EF2_GRAPH_BATCHED_RELEASE=1 restores the unconditional
    release (ablation / the memory line's escape). Returns (keep: bool, reason word)."""
    if CFG.batched_release:
        return False, "EF2_GRAPH_BATCHED_RELEASE=1"
    if plane_bytes is None:
        return False, "plane_bytes=unknown"
    try:
        free, _total = torch.cuda.mem_get_info()
        reserve = int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated())
    except Exception:
        return False, "mem_get_info=unavailable"
    avail = int(free) + max(0, reserve)
    need = int(CFG.batched_keep_factor * int(plane_bytes) + CFG.batched_keep_margin_gib * 2 ** 30)
    word = f"avail={avail / 2 ** 30:.1f}GiB,need={need / 2 ** 30:.1f}GiB,batch={int(batch)}"
    return (avail >= need), word


def _site_budget(where):
    """The budget that governs a capture site: the site's own (trunk_* and recycle -> trunk, generic_* -> encoder, sampler* -> sampler) when set
    (>= 0), else the global graph_budget_tokens."""
    if where.startswith("trunk_") or where.startswith("recycle"):
        own = CFG.graph_budget_tokens_trunk
    elif where.startswith("generic_"):
        own = CFG.graph_budget_tokens_encoder
    elif where.startswith("sampler"):
        own = CFG.graph_budget_tokens_sampler
    else:
        own = -1
    return CFG.graph_budget_tokens if own is None or int(own) < 0 else int(own)


def _pair_tokens(args, kwargs):
    """The token count the graph budget is decided on: the L of the first pair-shaped tensor ([B, L, L, C]) among the call's arguments;
    when no argument is square in its two middle extents (a ROW BLOCK [B, R, L, C] of a pair tensor, as a row-sharded caller passes),
    the largest middle extent of any 4-D tensor argument — so such a call is still budget-gated, never captured unbudgeted; None only
    when no 4-D tensor is passed (no budget decision)."""
    tensors = [t for t in list(args) + list(kwargs.values()) if torch.is_tensor(t) and t.dim() == 4]
    for t in tensors:
        if t.shape[1] == t.shape[2]:
            return int(t.shape[1])
    if tensors:
        return int(max(max(t.shape[1], t.shape[2]) for t in tensors))
    return None


def _pool():
    if _POOL["handle"] is None:
        _POOL["handle"] = torch.cuda.graph_pool_handle()
    return _POOL["handle"]


# The step-graph sampler's SUB-GENERATION. The diffusion step graphs (G2, `sg`) capture into their OWN private pool, apart from the trunk /
# encoder / recycle graphs' pool of the generation: when the sampler's LRU (CFG.lru_sampler step graphs) is spent, ALL step graphs and their
# pool go together (clear_sampler_graphs) and the trunk graphs, the static side buffers (SWA masks, pair-bias buffers, ef2_atom's registry) and
# the exact caches STAY — a step-graph capture (~0.1-0.4 s) instead of a whole-generation re-capture (1.5-3.9 s of trunk / encoder capture per
# event), and no clear_static() in the middle of a fold (HAZARDS 29's trigger). Graphs never outlive the buffers they read (the buffers
# are dropped only by clear_graphs, which drops these graphs first); the pool invariant holds per pool: no step graph is destroyed while another step
# graph of the same sampler pool lives. A whole-generation clear resets the sub-generation with it.
_SAMPLER_POOL = {"handle": None}


def _sampler_pool():
    if _SAMPLER_POOL["handle"] is None:
        _SAMPLER_POOL["handle"] = torch.cuda.graph_pool_handle()
    return _SAMPLER_POOL["handle"]


def clear_sampler_graphs(reason="lru"):
    """Drop every diffusion STEP graph (all graphed samplers) and the sampler pool; keep everything else of the generation. Same protocol as
    clear_graphs for what it drops: synchronize first, drop all step graphs together, gc so the CUDAGraph destructors run now, fresh sampler
    pool, empty_cache last. Counted: STATS sampler_graph_clears_<reason>, sampler_generations."""
    import gc
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    for sh in list(_GRAPHED_SAMPLERS):
        sh._ef2opt_step_graphs.clear()
    gc.collect()
    _SAMPLER_POOL["handle"] = None
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    STATS["sampler_graph_clears_" + reason] += 1; STATS["sampler_generations"] += 1


def clear_graphs(model=None, reason="shape"):
    """Drop every captured graph and every static side buffer a graph may read (SWA masks, pair-bias buffers).
    Robustness protocol (the pool invariant): (1) torch.cuda.synchronize() FIRST so no replay is in flight while graph objects /
    their private-pool blocks are released; (2) drop graphs + all ptr-referenced static buffers together (never one
    without the other); (3) gc.collect() so CUDAGraph destructors run now, not at an arbitrary later allocation;
    (4) start a FRESH graph pool for the next generation of captures (no block reuse across generations);
    (5) empty_cache last."""
    import gc
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    for m in list(_GRAPHED_MODULES):
        m._ef2opt_graphs.clear()
    for sh in list(_GRAPHED_SAMPLERS):
        sh._ef2opt_step_graphs.clear()
    for blk in list(_PB_BLOCKS):
        blk._ef2opt_pb.clear()
    _SWA_STATIC.clear()
    gc.collect()
    _POOL["handle"] = None; _SAMPLER_POOL["handle"] = None
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    STATS["graph_clears_" + reason] += 1; STATS["graph_generations"] += 1


def invariant_report():
    """The pool invariant for long-running processes: no individual CUDA graph is ever destroyed while other graphs of the same pool generation live (the step
    graphs' sampler pool is its own generation: clear_sampler_graphs drops all of them together, STATS sampler_generations).
    No code path pops a single graph; this counter exists so drivers can print+assert it at job end."""
    pops = int(STATS.get("individual_graph_pops", 0)) + int(STATS.get("sampler_graph_evict", 0)) + int(STATS.get("trunk_graph_evict", 0)) + int(STATS.get("generic_graph_evict", 0)) + int(STATS.get("pair_bias_evict", 0))
    return dict(individual_graph_pops=pops, graph_generations=int(STATS.get("graph_generations", 0)),
                clears={k: int(v) for k, v in STATS.items() if k.startswith("graph_clears_")}, sampler_generations=int(STATS.get("sampler_generations", 0)),
                unexpected_sampler_miss=int(STATS.get("sampler_graph_unexpected_miss", 0)),
                reset_after_error=int(STATS.get("reset_after_error", 0)), ok=(pops == 0))


def reset_after_error():
    """call from a driver's except: block. If the CUDA context is still healthy, drop all graphs so the next fold
    re-captures from scratch; if the context is poisoned (sticky error) re-raise so the process exits non-zero and
    the shard can be resumed by a fresh process (rows already written are kept by the drivers)."""
    STATS["reset_after_error"] += 1
    try:
        torch.cuda.synchronize()
        healthy = True
    except Exception:
        healthy = False
    if healthy:
        try:
            clear_graphs(reason="error")
        except Exception:
            healthy = False
    return healthy


# --------------------------------------------------------------------------------------------------------------
# D59b numerics-mode guard: every captured graph / cached tensor is only valid for the numerics mode it was produced in
# (kernel backend, chunk size, TF32 / fp32-matmul precision, cuDNN flags, deterministic mode, autocast state, model dtype).
# The graph pools and caches are keyed on shape/content, so the MODE is folded into the pool GENERATION instead: on any
# change of the signature since the last capture, all graphs + static buffers + exact caches are dropped (whole generation)
# before the request runs, and the new generation is captured in the new mode. ef2_server.py sets the mode once per process
# and never toggles, so this never fires there; it protects daemons that embed ef2_opt and switch modes between requests.
_MODE = {"sig": None, "models": []}


def numerics_mode_signature():
    """Per-process numerics mode: the `_kernel_backend` of EVERY submodule that has one (PairUpdateBlock, TriangleMultiplication,
    AttentionPairBias, transitions: set by model.set_kernel_backend) and every `_chunk_size` (set by model.set_chunk_size), the
    param dtype, TF32 / fp32-matmul precision, cuDNN flags, deterministic mode, NVIDIA_TF32_OVERRIDE. The submodule list is
    cached at install(); reading ~300 python attributes costs ~30 us per call."""
    kb = []
    for m in _MODE["models"]:
        mods = _MODE.setdefault("scan", {}).get(id(m))
        if mods is None:
            mods = [(n, sub) for n, sub in m.named_modules() if hasattr(sub, "_kernel_backend") or hasattr(sub, "_chunk_size")]
            _MODE["scan"][id(m)] = mods
        kb.append(tuple((n, str(getattr(sub, "_kernel_backend", "-")), str(getattr(sub, "_chunk_size", "-"))) for n, sub in mods))
        try:
            kb.append(("param_dtype", str(next(m.parameters()).dtype)))
        except Exception:
            pass
    return (hash(tuple(kb)), bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32), torch.get_float32_matmul_precision(),
            bool(torch.are_deterministic_algorithms_enabled()), bool(torch.backends.cudnn.deterministic), bool(torch.backends.cudnn.benchmark),
            os.environ.get("NVIDIA_TF32_OVERRIDE"))


def numerics_mode_guard():
    """Call at the entry of every graphed forward / cache lookup. Returns True if the generation was reset."""
    if not int(os.environ.get("EF2_NUMERICS_GUARD", "1")):
        return False
    sig = numerics_mode_signature()
    if _MODE["sig"] is None:
        _MODE["sig"] = sig; return False
    if sig == _MODE["sig"]:
        return False
    STATS["numerics_mode_changes"] += 1
    try:
        diff = [(a, b) for a, b in zip(_MODE["sig"], sig) if a != b]
    except Exception:
        diff = (_MODE["sig"], sig)
    print(f"[ef2_opt] numerics mode changed since last capture -> dropping all graphs + exact caches (D59b). changed fields: {diff}", flush=True)
    try:
        clear_graphs(reason="numerics_mode")
    except Exception:
        pass
    for cache in (_FEATURE_CACHE_REF, _ESMC_CACHE_REF):
        c = cache.get("obj")
        if c is not None:
            try:
                c.clear()
            except Exception:
                pass
    _MODE["sig"] = sig
    return True


_FEATURE_CACHE_REF = {"obj": None}
_ESMC_CACHE_REF = {"obj": None}

_GRAPHED_MODULES = []
_GRAPHED_SAMPLERS = []
_PB_BLOCKS = []
_SWA_STATIC = {}          # (B, N, half_window, device) -> dict(allowed=..., valid=..., idx_sig=...)  fold-scoped static mask buffers
_FOLD = {"n": 0}          # fold epoch: bumped at every structure_head.sample() call


def _tkey(t):
    return None if t is None else (tuple(t.shape), t.dtype)


# --------------------------------------------------------------------------------------------------------------
# G1: graphed FoldingTrunk
# --------------------------------------------------------------------------------------------------------------
def _graphed_trunk_forward(self, pair, pair_attention_mask=None):
    eager = self._ef2opt_eager_forward
    if CFG._body_eager:                                      # inside a G5 recycle body (eager issue or capture): the module's own kernels, no nested graph
        return eager(pair, pair_attention_mask=pair_attention_mask)
    numerics_mode_guard()
    if (not CFG.enabled) or torch.is_grad_enabled() or (not pair.is_cuda) or len(self.blocks) == 0:
        return eager(pair, pair_attention_mask=pair_attention_mask)
    if _over_graph_budget("trunk_" + self._ef2opt_name, int(pair.shape[1]), batch=int(pair.shape[0]), plane_bytes=pair.numel() * pair.element_size()):   # a batched pair (the confidence head's trunk on S samples) is budgeted by its plane
        return eager(pair, pair_attention_mask=pair_attention_mask)
    key = (_tkey(pair), _tkey(pair_attention_mask))
    cache = self._ef2opt_graphs
    ent = cache.get(key)
    if ent is None:
        if len(cache) >= CFG.lru_trunk:
            # never destroy a single graph that shares the pool with live graphs -> reset the whole generation
            clear_graphs(reason="new_shape_" + self._ef2opt_name)
        t0 = time.perf_counter()
        static_in = pair.clone()
        static_mask = pair_attention_mask.clone() if pair_attention_mask is not None else None
        rng_before = torch.cuda.get_rng_state(pair.device)
        n_warm = CFG.warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warm):
                eager(static_in, pair_attention_mask=static_mask)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=_pool()):
            static_out = eager(static_in, pair_attention_mask=static_mask)
        torch.cuda.synchronize()
        rng_after = torch.cuda.get_rng_state(pair.device)
        if not torch.equal(rng_before, rng_after):
            # module consumed RNG during capture (dropout active) -> graphs would change the computation. disable for this module.
            STATS["trunk_graph_rng_refused_" + self._ef2opt_name] += 1
            torch.cuda.set_rng_state(rng_before, pair.device)
            self.forward = self._ef2opt_eager_forward
            del g
            return eager(pair, pair_attention_mask=pair_attention_mask)
        ent = (g, static_in, static_mask, static_out)
        cache[key] = ent
        STATS["trunk_graph_captures"] += 1
        STATS["trunk_graph_capture_s"] += time.perf_counter() - t0
    else:
        cache.move_to_end(key)
    g, static_in, static_mask, static_out = ent
    if static_in.data_ptr() != pair.data_ptr():            # G4 stages the trunk input straight into static_in: nothing to copy then
        static_in.copy_(pair)
    else:
        STATS["trunk_copy_in_elided"] += 1
    if static_mask is not None and static_mask.data_ptr() != pair_attention_mask.data_ptr():
        static_mask.copy_(pair_attention_mask)
    g.replay()
    if getattr(self, "_ef2opt_out_alias", False) and static_out.data_ptr() != static_in.data_ptr():
        out = static_out                                     # G4's loop consumes the static output before this module's next replay (its contract)
        STATS["trunk_clone_out_elided"] += 1
    else:
        out = static_out.clone()
    STATS["trunk_graph_replays"] += 1
    if CFG.probe:
        ref = eager(pair, pair_attention_mask=pair_attention_mask)
        d = (ref.float() - out.float()).abs().max().item()
        VERIFY_LOG.append(("trunk", self._ef2opt_name, tuple(pair.shape), d))
    return out


def install_trunk_graphs(model):
    names = []
    cands = [("folding_trunk", getattr(model, "folding_trunk", None)), ("lm_encoder", getattr(model, "lm_encoder", None)),
             ("parcae_coda", getattr(model, "parcae_coda", None))]
    ch = getattr(model, "confidence_head", None)
    if ch is not None and getattr(ch, "folding_trunk", None) is not None:
        cands.append(("confidence_head.folding_trunk", ch.folding_trunk))
    for name, mod in cands:
        if mod is None or getattr(mod, "_ef2opt_graphed", False):
            continue
        if name == "lm_encoder" and not CFG.graph_lm_encoder:
            # the LM encoder is run with per-loop lm_dropout (F.dropout(training=True)) as shipped:
            # it consumes the CUDA RNG -> a replayed graph would freeze the dropout mask. Never graph it unless lm_dropout == 0.
            continue
        mod._ef2opt_eager_forward = mod.forward          # bound original
        mod._ef2opt_graphs = collections.OrderedDict()
        mod._ef2opt_name = name
        mod.forward = types.MethodType(_graphed_trunk_forward, mod)
        mod._ef2opt_graphed = True
        _GRAPHED_MODULES.append(mod)
        names.append(name)
    return names


# --------------------------------------------------------------------------------------------------------------
# G2: sync-free sampler with graphed diffusion-module steps
# --------------------------------------------------------------------------------------------------------------
def sampler_fold_prologue(pair_bias_epoch):
    """The per-fold prologue of a `structure_head.sample` implementation — the ONE named place a sampler announces a new sample() call before
    its first DiffusionModule step: bumps the fold epoch (SWA static masks are recomputed at the fold's eager step 0), optionally the pair-bias
    epoch (every pb-cached attention block recomputes its bias buffer at the eager step 0; `_sample_v2` passes False because the
    `sample_with_epoch` wrapper installed around it bumps it, a sampler that REPLACES the whole chain — ef2_dit's roll-out — passes True), and
    runs the numerics guard. Returns (fold_epoch, pair_bias_epoch)."""
    _FOLD["n"] += 1
    if pair_bias_epoch:
        _PB_EPOCH["n"] += 1
    numerics_mode_guard()
    return _FOLD["n"], _PB_EPOCH["n"]


def pair_bias_cache_blocks():
    """number of attention blocks whose pair bias is served from the epoch-keyed cache (0 = lever pb not installed)."""
    return len(_PB_BLOCKS)


_DM_TENSOR_KEYS = ["x_noisy", "t_hat", "ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars", "ref_space_uid", "tok_idx",
                   "s_inputs", "s_trunk", "z_trunk", "relative_position_encoding", "asym_id", "residue_index", "entity_id", "token_index", "sym_id",
                   "token_attention_mask"]


def _tree_sig(obj):
    if torch.is_tensor(obj):
        return ("T", tuple(obj.shape), obj.dtype)
    if isinstance(obj, dict):
        return ("D", tuple((k, _tree_sig(v)) for k, v in obj.items()))
    if isinstance(obj, (tuple, list)):
        return ("L", tuple(_tree_sig(v) for v in obj))
    return ("V", obj)


def _tree_clone(obj):
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _tree_clone(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_tree_clone(v) for v in obj)
    if isinstance(obj, list):
        return [_tree_clone(v) for v in obj]
    return obj


def _tree_copy_(dst, src):
    if torch.is_tensor(dst):
        if dst.data_ptr() != src.data_ptr():               # a caller that staged into the static buffer itself (G4) passes that buffer: no copy
            dst.copy_(src)
        return
    if isinstance(dst, dict):
        for k in dst:
            _tree_copy_(dst[k], src[k])
        return
    if isinstance(dst, (tuple, list)):
        for a, b in zip(dst, src):
            _tree_copy_(a, b)
        return
    assert dst == src, f"static non-tensor value changed: {dst!r} vs {src!r}"


class _GraphedDiffusionStep:
    """one captured DiffusionModule forward (steps>=1 of a design/shape)."""

    def __init__(self, dm, kwargs, cache, num_diffusion_samples):
        self.dm = dm
        self.static_kwargs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        self.static_cache = _tree_clone(cache)
        self.nds = num_diffusion_samples
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(CFG.warmup):
                self._call()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=_sampler_pool()):                # the sampler's own pool (sub-generation): step graphs leave together, the generation stays
            out = self._call()
        torch.cuda.synchronize()
        self.static_x_denoised = out["x_denoised"]
        self.static_token_repr = out["token_repr"]

    def _call(self):
        return self.dm(**self.static_kwargs, num_diffusion_samples=self.nds, return_token_repr=True, return_atom_repr=False, inference_cache=self.static_cache)

    def run(self, kwargs, cache):
        for k, v in kwargs.items():
            sv = self.static_kwargs[k]
            if torch.is_tensor(sv):
                if sv.data_ptr() != v.data_ptr():
                    sv.copy_(v)
            else:
                assert sv == v
        _tree_copy_(self.static_cache, cache)
        self.graph.replay()
        return self.static_x_denoised.clone(), (self.static_token_repr.clone() if torch.is_tensor(self.static_token_repr) else self.static_token_repr)


def _sample_v2(self, z_trunk, s_inputs, s_trunk, relative_position_encoding, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, ref_space_uid,
               tok_idx, asym_id, residue_index, entity_id, token_index, sym_id, token_attention_mask=None, num_diffusion_samples=1, num_sampling_steps=None,
               max_inference_sigma=256.0, noise_scale=None, step_scale=None, return_atom_repr=False, use_inference_cache=True, denoising_early_exit_rmsd=None):
    # fall back to the stock sampler for options the graphed path does not cover
    sampler_fold_prologue(pair_bias_epoch=False)                 # fold epoch + numerics guard (the pb epoch is bumped by install_pair_bias_cache's outer wrapper)
    if (not CFG.enabled) or return_atom_repr or (denoising_early_exit_rmsd is not None) or (not use_inference_cache) or (not s_inputs.is_cuda) \
            or _over_graph_budget("sampler", int(z_trunk.shape[1])):
        STATS["sampler_fallback"] += 1
        return self._ef2opt_eager_sample(z_trunk=z_trunk, s_inputs=s_inputs, s_trunk=s_trunk, relative_position_encoding=relative_position_encoding, ref_pos=ref_pos,
                                         ref_charge=ref_charge, ref_mask=ref_mask, ref_element=ref_element, ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,
                                         tok_idx=tok_idx, asym_id=asym_id, residue_index=residue_index, entity_id=entity_id, token_index=token_index, sym_id=sym_id,
                                         token_attention_mask=token_attention_mask, num_diffusion_samples=num_diffusion_samples, num_sampling_steps=num_sampling_steps,
                                         max_inference_sigma=max_inference_sigma, noise_scale=noise_scale, step_scale=step_scale, return_atom_repr=return_atom_repr,
                                         use_inference_cache=use_inference_cache, denoising_early_exit_rmsd=denoising_early_exit_rmsd)
    with torch.inference_mode():
        n_atoms = tok_idx.shape[1]
        device = s_inputs.device
        target_batch = s_inputs.shape[0] * num_diffusion_samples
        inference_cache = {}
        steps = self.inference_num_steps if num_sampling_steps is None else int(num_sampling_steps)
        schedule = self.inference_noise_schedule(steps, device)              # identical tensor ops to stock
        if max_inference_sigma is not None:
            schedule = schedule[schedule <= float(max_inference_sigma)]
            schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))
        lam = self.noise_scale if noise_scale is None else float(noise_scale)
        eta = self.step_scale if step_scale is None else float(step_scale)
        x = schedule[0] * torch.randn(target_batch, n_atoms, 3, device=device, dtype=torch.float32)   # same RNG draw #1 as stock
        atom_mask = ref_mask.repeat_interleave(num_diffusion_samples, 0).float()
        gammas = torch.where(schedule > self.gamma_min, torch.full_like(schedule, self.gamma_0), torch.zeros_like(schedule))
        sched_list = schedule.tolist(); gam_list = gammas.tolist()          # ONE host sync instead of 3 per step; float() of fp32 elements == .item()
        num_steps = len(sched_list) - 1
        x_denoised_prev = None
        token_repr = None
        dm = self.diffusion_module
        base_kwargs = dict(ref_pos=ref_pos, ref_charge=ref_charge, ref_mask=ref_mask, ref_element=ref_element, ref_atom_name_chars=ref_atom_name_chars,
                           ref_space_uid=ref_space_uid, tok_idx=tok_idx, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                           relative_position_encoding=relative_position_encoding, asym_id=asym_id, residue_index=residue_index, entity_id=entity_id,
                           token_index=token_index, sym_id=sym_id, token_attention_mask=token_attention_mask)
        gstep = None
        # generation-atomic pools: decide BEFORE eager step 0 whether this fold needs a new sampler graph; if the cache already
        # holds a graph for another shape, reset the whole graph generation (all graphs + static side buffers + pool) instead of
        # destroying one graph that shares the memory pool with live graphs (partial destruction of a shared pool = the SIGSEGV pattern).
        # No shape prediction here. The sampler keeps up to CFG.lru_sampler graphs per generation WITHOUT ever evicting one
        # (capture is ~0.05 s); only when that budget is exhausted is the whole generation reset (at step 1, after a synchronize,
        # before the new capture; static side buffers are rebuilt by the eager warm-up inside the capture helper).
        for step_idx in range(num_steps):
            sigma_tm_val = float(sched_list[step_idx]); sigma_t_val = float(sched_list[step_idx + 1]); gamma_val = float(gam_list[step_idx + 1])
            x, x_denoised_prev = self._center_random_augmentation(x, atom_mask, second_coords=x_denoised_prev)     # RNG draws as stock
            t_hat_val = sigma_tm_val * (1.0 + gamma_val)
            eps_std = lam * max(t_hat_val ** 2 - sigma_tm_val ** 2, 0.0) ** 0.5
            x_noisy = x + eps_std * torch.randn_like(x)                                                          # RNG draw as stock
            t_hat = torch.full((target_batch,), t_hat_val, device=device, dtype=torch.float32)
            kwargs = dict(base_kwargs, x_noisy=x_noisy, t_hat=t_hat)
            if step_idx == 0:
                dm_out = dm(**kwargs, num_diffusion_samples=num_diffusion_samples, return_token_repr=True, return_atom_repr=False, inference_cache=inference_cache)
                x_denoised = dm_out["x_denoised"]; token_repr = dm_out["token_repr"]
                STATS["sampler_eager_steps"] += 1
            else:
                if gstep is None:
                    sig = (_tree_sig(kwargs), _tree_sig(inference_cache), num_diffusion_samples)
                    gcache = self._ef2opt_step_graphs
                    gstep = gcache.get(sig)
                    if gstep is None:
                        if len(gcache) >= CFG.lru_sampler:
                            clear_sampler_graphs(reason="lru")         # every step graph + the sampler pool (never an individual graph); the generation's trunk graphs and static buffers stay
                            gcache = self._ef2opt_step_graphs
                        if False:
                            gstep = "eager"
                        else:
                          t0 = time.perf_counter()
                          # capture must not consume the global RNG (the module is RNG-free at inference); guard anyway
                          rng_state = torch.cuda.get_rng_state(device)
                          gstep = _GraphedDiffusionStep(dm, kwargs, inference_cache, num_diffusion_samples)
                          if not torch.equal(rng_state, torch.cuda.get_rng_state(device)):
                              STATS["sampler_graph_rng_advanced"] += 1
                          torch.cuda.set_rng_state(rng_state, device)
                          gcache[sig] = gstep
                          STATS["sampler_graph_captures"] += 1; STATS["sampler_graph_capture_s"] += time.perf_counter() - t0
                    elif gstep is not None:
                        gcache.move_to_end(sig)
                if gstep == "eager":
                    dm_out = dm(**kwargs, num_diffusion_samples=num_diffusion_samples, return_token_repr=True, return_atom_repr=False, inference_cache=inference_cache)
                    x_denoised = dm_out["x_denoised"]; token_repr = dm_out["token_repr"]
                else:
                    x_denoised, token_repr = gstep.run(kwargs, inference_cache)
                    STATS["sampler_graph_replays"] += 1
                if CFG.probe and gstep != "eager":
                    _en = CFG.enabled; CFG.enabled = False          # eager reference with ALL ef2_opt forwards bypassed (stock pair-bias, stock SWA mask)
                    try:
                        ref_out = dm(**kwargs, num_diffusion_samples=num_diffusion_samples, return_token_repr=True, return_atom_repr=False, inference_cache=inference_cache)
                    finally:
                        CFG.enabled = _en
                    d = (ref_out["x_denoised"].float() - x_denoised.float()).abs().max().item()
                    VERIFY_LOG.append(("sampler_step", step_idx, tuple(x_noisy.shape), d))
            with torch.autocast(device_type="cuda", enabled=False):
                x_noisy = self._weighted_rigid_align(x_noisy.float(), x_denoised.float(), atom_mask, atom_mask)
            x_noisy = x_noisy.to(dtype=x_denoised.dtype)
            denoised_over_sigma = (x_noisy - x_denoised) / t_hat_val
            x = x_noisy + eta * (sigma_t_val - t_hat_val) * denoised_over_sigma
            x_denoised_prev = x_denoised
        result = {"sample_atom_coords": x, "diff_token_repr": token_repr}
        return result


def install_sampler_graphs(model):
    sh = model.structure_head
    if getattr(sh, "_ef2opt_graphed", False):
        return False
    sh._ef2opt_eager_sample = sh.sample
    sh._ef2opt_step_graphs = collections.OrderedDict()
    sh.sample = types.MethodType(_sample_v2, sh)
    sh._ef2opt_graphed = True
    _GRAPHED_SAMPLERS.append(sh)
    return True


# --------------------------------------------------------------------------------------------------------------
# M1: fused TriMul for the MSA encoder blocks (optional; same Triton kernel the 'fused' pair trunk uses)
# --------------------------------------------------------------------------------------------------------------
def _msa_trimul_call(C, blk, pb, direction, maskf):
    """One MSA-block fused TriMul+residual call.  When ef2_w4's weight-cast cache or its provider binding is live, the call goes through
    ef2_w4's PairUpdateBlock wrapper: the eight bf16 weight casts are cached per (block, direction) and the provider receives a STABLE weight
    identity, so its address-keyed weight packs are made once per block — the per-call casts this path used to hand down made the provider
    pack (and, on cc 8.0, retain) a fresh weight set on every call: +0.25 GiB per fold at 1400 tokens.  Values identical (same casts of the
    same parameters; same mask, eps, residual)."""
    W4 = sys.modules.get("ef2_w4")
    st = getattr(W4, "_STATE", None) if W4 is not None else None
    if st and st.get("enabled") and (st.get("weight_cache") or st.get("tx")) and hasattr(W4, "_pub_fused_trimul_w4"):
        return W4._pub_fused_trimul_w4(blk, pb, direction, maskf)
    eng = (blk.tri_mul_out if direction == "outgoing" else blk.tri_mul_in)._engine
    p_in_w, g_in_w = eng.split_kernel_weights()
    bf = lambda t: t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16)  # noqa: E731
    return C._fused_trimul_with_residual(pb, direction, residual=pb, drop_mask=None, norm_in_weight=bf(eng.norm_start.weight), norm_in_bias=bf(eng.norm_start.bias),
                                        p_in_weight=bf(p_in_w), g_in_weight=bf(g_in_w), norm_out_weight=bf(eng.norm_mix.weight), norm_out_bias=bf(eng.norm_mix.bias),
                                        p_out_weight=bf(eng.proj_emit.weight), g_out_weight=bf(eng.proj_gate.weight), mask=maskf, eps=C._EPS)


def _msa_block_forward_fused(self, m, pair, msa_attention_mask, pair_attention_mask):
    import transformers.models.esmfold2.modeling_esmfold2_common as C
    pair = pair + self.outer_product_mean(m, msa_attention_mask)
    if not self.is_final_block:
        m = m + self.msa_pair_weighted_averaging(m, pair, pair_attention_mask)
        m = m + self.msa_transition(m)
    _kb = getattr(self, '_ef2opt_backend_owner', None)
    _fused_mode = (getattr(_kb, '_kernel_backend', 'fused') == 'fused') if _kb is not None else True
    if CFG.enabled and _fused_mode and (not torch.is_grad_enabled()) and pair.is_cuda and C.TRITON_KERNELS_AVAILABLE:   # D59b: M1 follows the model's kernel backend (reference mode -> stock einsum path)
        orig_dtype = pair.dtype
        pb = pair.to(torch.bfloat16)
        maskf = pair_attention_mask.to(pb.dtype) if pair_attention_mask is not None else None
        for direction in ("outgoing", "incoming"):
            pb = _msa_trimul_call(C, self, pb, direction, maskf)           # cached casts + a stable provider weight identity (one pack per block)
        pair = pb.to(orig_dtype)
        STATS["msa_fused_trimul_calls"] += 2
    else:
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
    pair = pair + self.pair_transition(pair)
    return m, pair


def install_msa_fused_trimul(model):
    enc = getattr(model, "msa_encoder", None)
    if enc is None:
        return 0
    n = 0
    for blk in enc.blocks:
        if getattr(blk, "_ef2opt_fused", False):
            continue
        blk.forward = types.MethodType(_msa_block_forward_fused, blk)
        blk._ef2opt_fused = True
        object.__setattr__(blk, "_ef2opt_backend_owner", getattr(model, "folding_trunk", None))   # plain attribute (not a submodule) -> follows set_kernel_backend()
        n += 1
    return n



# --------------------------------------------------------------------------------------------------------------
# S1: SWA3DRoPEAttention SDPA-fallback mask precompute (exact: same boolean mask, built once per atom layout instead of
#     6x per diffusion step with an index_put that is not graph-capturable). Only active when flash-attn is absent.
# --------------------------------------------------------------------------------------------------------------
def _swa_idx_sig(attention_params):
    if len(attention_params) > 2 and torch.is_tensor(attention_params[2]):
        idx = attention_params[2]
        return ("idx", idx.data_ptr(), tuple(idx.shape), idx.dtype)
    return ("none",)


def _swa_build_mask(attention_params, B, N, half_window, device):
    """identical boolean mask to the stock SDPA fallback in SWA3DRoPEAttention.forward."""
    if len(attention_params) > 2 and torch.is_tensor(attention_params[2]):
        idx = attention_params[2]
        valid = torch.zeros(B * N, dtype=torch.bool, device=device)
        valid[idx] = True
        valid = valid.view(B, N)
    else:
        valid = torch.ones(B, N, dtype=torch.bool, device=device)
    rank = torch.cumsum(valid, dim=1) - 1
    within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= half_window
    allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed |= torch.eye(N, dtype=torch.bool, device=device)
    return allowed.unsqueeze(1), valid


def _swa_allowed_mask(attention_params, B, N, half_window, device):
    """Return (allowed, valid) as STATIC buffers per (B, N, half_window): the same tensor objects for the whole life of a
    graph generation (clear_graphs drops them together with the graphs), refreshed IN PLACE (copy_) whenever the fold
    epoch or the atom-index layout changes. So a captured sampler graph always reads a live buffer holding the CURRENT
    fold's mask. (The hazard: a ptr-keyed LRU could evict+free a mask that a cached graph still referenced ->
    replay read unmapped memory after the next capture's empty_cache -> sticky 'unspecified launch failure' exactly at a
    same-token-count item transition; and a same-shape next item could have read the previous item's mask content.)"""
    key = (B, N, half_window, str(device))
    ent = _SWA_STATIC.get(key)
    sig = (_FOLD["n"], _swa_idx_sig(attention_params))
    capturing = torch.cuda.is_current_stream_capturing()
    if ent is None:
        if capturing:
            raise RuntimeError("SWA static mask missing during graph capture (step 0 must run eagerly first)")
        allowed, valid = _swa_build_mask(attention_params, B, N, half_window, device)
        ent = dict(allowed=allowed, valid=valid, sig=sig)
        _SWA_STATIC[key] = ent
        STATS["swa_mask_builds"] += 1
        return ent["allowed"], ent["valid"]
    if ent["sig"] != sig:
        if capturing:
            # cannot rebuild during capture; the eager step 0 of this fold must have refreshed it already
            raise RuntimeError("SWA static mask stale during graph capture")
        allowed, valid = _swa_build_mask(attention_params, B, N, half_window, device)
        if CFG.probe:
            VERIFY_LOG.append(("swa_mask_refresh_changed", key[1], 0, float((allowed != ent["allowed"]).any().item())))
        ent["allowed"].copy_(allowed); ent["valid"].copy_(valid); ent["sig"] = sig
        STATS["swa_mask_refresh"] += 1
    else:
        STATS["swa_mask_hits"] += 1
    return ent["allowed"], ent["valid"]


def _swa_forward_cached(self, x, attention_params):
    C = _common()
    if C.FLASH_ATTN_AVAILABLE or not CFG.enabled:
        return self._ef2opt_eager_forward(x, attention_params)
    B, N = x.shape[:2]
    cos, sin = attention_params[0], attention_params[1]
    x_input = x
    qkv = self.Wqkv(x)
    qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
    q, k, v = qkv.unbind(0)
    q, k = C.qk_norm(q), C.qk_norm(k)
    q = C.apply_rotary_emb_3d(q, cos, sin)
    k = C.apply_rotary_emb_3d(k, cos, sin)
    input_dtype = q.dtype
    if q.dtype not in (torch.float16, torch.bfloat16):
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    allowed, valid = _swa_allowed_mask(attention_params, B, N, self.half_window, q.device)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed, scale=self.scale).transpose(1, 2)
    out = out * valid.unsqueeze(-1).unsqueeze(-1)
    out = out.to(input_dtype).reshape(B, N, -1)
    out = out * torch.sigmoid(self.gate_proj(x_input))
    return self.out_proj(out)


def _common():
    import transformers.models.esmfold2.modeling_esmfold2_common as C
    return C


def install_swa_mask_cache(model):
    C = _common()
    n = 0
    for m in model.modules():
        if isinstance(m, C.SWA3DRoPEAttention) and not getattr(m, "_ef2opt_swa", False):
            m._ef2opt_eager_forward = m.forward
            m.forward = types.MethodType(_swa_forward_cached, m)
            m._ef2opt_swa = True
            n += 1
    return n



# --------------------------------------------------------------------------------------------------------------
# G3: generic CUDA-graph wrapper for modules called with several tensor args/kwargs (lm_encoder, msa_encoder).
#     The per-loop lm_dropout (F.dropout on lm_z) and the MSA row subsampling (randperm) happen in _run_one_loop
#     BEFORE these modules are called, so the modules themselves are RNG-free; the RNG guard below checks it.
# --------------------------------------------------------------------------------------------------------------
def _sig_args(args, kwargs):
    return (tuple(_tree_sig(a) for a in args), tuple((k, _tree_sig(v)) for k, v in sorted(kwargs.items())))


def _graphed_generic_forward(self, *args, **kwargs):
    eager = self._ef2opt_eager_forward
    if CFG._body_eager:
        return eager(*args, **kwargs)
    numerics_mode_guard()
    dev_t = next((a for a in list(args) + list(kwargs.values()) if torch.is_tensor(a)), None)
    if (not CFG.enabled) or torch.is_grad_enabled() or dev_t is None or (not dev_t.is_cuda):
        return eager(*args, **kwargs)
    if _over_graph_budget("generic_" + self._ef2opt_name, _pair_tokens(args, kwargs)):
        return eager(*args, **kwargs)
    key = _sig_args(args, kwargs)
    cache = self._ef2opt_graphs
    ent = cache.get(key)
    if ent is None:
        if len(cache) >= max(1, CFG.lru_generic):
            clear_graphs(reason="new_shape_" + self._ef2opt_name)
        t0 = time.perf_counter()
        s_args = [_tree_clone(a) for a in args]; s_kwargs = {k: _tree_clone(v) for k, v in kwargs.items()}
        dev = dev_t.device
        rng_before = torch.cuda.get_rng_state(dev)
        n_warm = CFG.warmup
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warm):
                eager(*s_args, **s_kwargs)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=_pool()):
            s_out = eager(*s_args, **s_kwargs)
        torch.cuda.synchronize()
        if not torch.equal(rng_before, torch.cuda.get_rng_state(dev)):
            STATS["generic_graph_rng_refused_" + self._ef2opt_name] += 1
            torch.cuda.set_rng_state(rng_before, dev)
            self.forward = self._ef2opt_eager_forward
            del g
            return eager(*args, **kwargs)
        ent = (g, s_args, s_kwargs, s_out); cache[key] = ent
        STATS["generic_graph_captures_" + self._ef2opt_name] += 1; STATS["generic_graph_capture_s"] += time.perf_counter() - t0
    else:
        cache.move_to_end(key)
    g, s_args, s_kwargs, s_out = ent
    for d_, s_ in zip(s_args, args):
        _tree_copy_(d_, s_)
    for k in s_kwargs:
        _tree_copy_(s_kwargs[k], kwargs[k])
    g.replay()
    STATS["generic_graph_replays"] += 1
    if getattr(self, "_ef2opt_out_alias", False):
        out = s_out                                          # G4's loop consumes the static output before this module's next replay (its contract)
        STATS["generic_clone_out_elided"] += 1
    else:
        out = _tree_clone(s_out)
    if CFG.probe:
        ref = eager(*args, **kwargs)
        rt = ref if torch.is_tensor(ref) else ref[0]
        ot = out if torch.is_tensor(out) else out[0]
        VERIFY_LOG.append(("generic", self._ef2opt_name, tuple(rt.shape), (rt.float() - ot.float()).abs().max().item()))
    return out


def generic_static_entry_sig(mod, arg_sigs, kwargs):
    """As generic_static_entry, with the positional arguments given as _tree_sig signatures (("T", shape, dtype)) instead of tensors — for a
    caller that wants the static buffer BEFORE materialising the argument (G4 writes upstream's dtype cast straight into it)."""
    cache = getattr(mod, "_ef2opt_graphs", None)
    if not cache:
        return None
    return cache.get((tuple(arg_sigs), tuple((k, _tree_sig(v)) for k, v in sorted(kwargs.items()))))


def generic_static_entry(mod, args, kwargs):
    """The live graph entry (g, static_args, static_kwargs, static_out) a graphed generic module holds for a call of these arguments' signature,
    else None (not graphed, not captured yet, released). G4 stages loop-invariant inputs into the static kwargs once per fold and passes those
    buffers themselves, which the replay path recognises (pointer-equal: no copy)."""
    cache = getattr(mod, "_ef2opt_graphs", None)
    if not cache:
        return None
    return cache.get(_sig_args(args, kwargs))


def install_generic_graphs(model, names=("lm_encoder", "msa_encoder")):
    done = []
    for name in names:
        mod = getattr(model, name, None)
        if mod is None or getattr(mod, "_ef2opt_graphed", False):
            continue
        mod._ef2opt_eager_forward = mod.forward
        mod._ef2opt_graphs = collections.OrderedDict()
        mod._ef2opt_name = name
        mod.forward = types.MethodType(_graphed_generic_forward, mod)
        mod._ef2opt_graphed = True
        _GRAPHED_MODULES.append(mod)
        done.append(name)
    return done



# --------------------------------------------------------------------------------------------------------------
# G4: loop-resident static I/O for the trunk recycling loop (ESMFold2Model._run_one_loop re-issued by the kit).
#     Upstream's loop, statement by statement, with three execution changes and no arithmetic change:
#       (1) the trunk input `a * z + F.linear(injected, b_mat)` is written by the same add kernel INTO the folding-trunk graph's static
#           input buffer (torch.add(..., out=static_in)) when that graph exists -> the replay path has nothing to copy in;
#       (2) inside the loop the graphed folding_trunk / lm_encoder / msa_encoder hand back their static OUTPUT buffers (no clone per call):
#           every consumer of those tensors in the loop body runs before the producing graph replays again (next iteration), and the
#           loop returns a clone, so no static buffer escapes the loop;
#       (3) the MSA encoder's loop-invariant inputs (x_pair = z_init, x_inputs) are staged into its graph's static kwargs once per fold
#           instead of once per iteration;
#       (4) the recycle's dead pair temporaries (the per-loop lm_z draw and its cast, the encoders' outputs, z_inject_pair, injected_pair,
#           the previous trunk output, the trunk's input once it returned) are released each at its last use — reference drops only; the XL
#           add-on's x2 LOOPFREE statement, which cannot reach this frame (STATS loop_static_released per recycle).
#     RNG order (per-loop lm dropout, MSA row subsample) and every tensor op are upstream's: exact tier. Engaged by install(loop_static=True)
#     on a model whose folding_trunk carries G1; anything else (grad enabled, CPU tensors, G1 absent, CFG off) runs upstream's own method.
# --------------------------------------------------------------------------------------------------------------
def _modeling():
    import transformers.models.esmfold2.modeling_esmfold2 as MOD
    return MOD


def _run_one_loop_v2(self, z, z_init, lm_z, _msa_inputs, pair_mask, a, b_mat, tok_mask, total_steps):
    eager_loop = self._ef2opt_eager_run_one_loop
    ft = getattr(self, "folding_trunk", None)
    if (not CFG.enabled) or (not CFG.loop_static) or torch.is_grad_enabled() or (not torch.is_tensor(z)) or (not z.is_cuda) \
            or ft is None or not getattr(ft, "_ef2opt_graphed", False):
        STATS["loop_static_eager"] += 1
        return eager_loop(z=z, z_init=z_init, lm_z=lm_z, _msa_inputs=_msa_inputs, pair_mask=pair_mask, a=a, b_mat=b_mat, tok_mask=tok_mask, total_steps=total_steps)
    MOD = _modeling()
    F = MOD.F
    numerics_mode_guard()
    if CFG.recycle_graph and total_steps >= 1:
        out = _run_one_loop_g5(self, z, z_init, lm_z, _msa_inputs, pair_mask, a, b_mat, tok_mask, total_steps, MOD)
        if out is not None:
            STATS["recycle_graph_loops"] += 1
            return out
    STATS["loop_static_calls"] += 1
    lm_cfg = self.config.lm_encoder
    _per_loop_lm_dropout = (lm_z is not None and getattr(lm_cfg, "per_loop_lm_dropout", False) and getattr(lm_cfg, "lm_dropout", 0.0) > 0.0)
    _lm_dropout_p = getattr(lm_cfg, "lm_dropout", 0.0)
    lm_enc = self.lm_encoder
    msa_enc = self.msa_encoder
    msa_runs = msa_enc is not None and _msa_inputs is not None
    # Static-output aliasing under the generation's SHARED graph pool: a graph's output block stays intact only until ANOTHER graph of the pool
    # replays (a later-captured graph may hold that block as one of its intermediates). So an output is taken without its clone only where the
    # loop consumes it before any other graph replays: the trunk's (a * z is formed right after the trunk replay, below), the MSA encoder's
    # (added into z_inject_pair immediately), and the LM encoder's only when no MSA-encoder replay sits between it and its use.
    alias_mods = [m for m, ok in ((ft, True), (msa_enc, True), (lm_enc, not msa_runs)) if ok and m is not None and getattr(m, "_ef2opt_graphed", False)]
    staged = {"ent": None, "x_pair_src": None, "x_inputs_src": None}       # per fold: which msa-encoder graph entry holds this fold's invariant inputs
    az_next = None                                                         # a * z of the trunk output, formed right after the replay that produced it
    try:
        for m in alias_mods:
            m._ef2opt_out_alias = True
        for it in range(total_steps):
            if _per_loop_lm_dropout:
                assert lm_z is not None
                lm_z_i = F.dropout(lm_z, p=_lm_dropout_p, training=True)
            else:
                lm_z_i = lm_z

            refined_lm_z = None
            if lm_z_i is not None and lm_enc is not None:
                x_arg = None
                if lm_z_i.dtype != z_init.dtype and getattr(lm_enc, "_ef2opt_graphed", False):
                    # upstream's `lm_z_i.to(z_init.dtype)` is at::_to_copy = empty + copy_: the same copy kernel writes the cast straight into the
                    # LM-encoder graph's static input (cloned before capture, outside the pool) -> no transient tensor, no second copy at replay
                    ent_l = generic_static_entry_sig(lm_enc, (("T", tuple(lm_z_i.shape), z_init.dtype),), {"pair_attention_mask": pair_mask})
                    if ent_l is not None:
                        x_arg = ent_l[1][0]
                        x_arg.copy_(lm_z_i)
                        STATS["loop_static_lm_staged"] += 1
                if x_arg is None:
                    x_arg = lm_z_i.to(z_init.dtype)
                lm_z_i = None                                                         # (4) the per-loop draw (fp32 under lm dropout) is dead once cast
                refined_lm_z = lm_enc(x_arg, pair_attention_mask=pair_mask)
                x_arg = None                                                          # (4) the cast is dead once encoded (upstream's is a call temporary)
                has_lm_pair = False                                                   # the LM pair signal now lives in refined_lm_z
            else:
                has_lm_pair = lm_z_i is not None

            z_inject_pair = z_init
            if has_lm_pair and lm_enc is None:
                z_inject_pair = z_inject_pair + lm_z_i.to(z_inject_pair.dtype)
            lm_z_i = None                                                             # (4)

            if msa_runs:
                msa_i, mask_i, hd_i, dv_i = MOD.maybe_subsample_msa(
                    _msa_inputs["msa"], _msa_inputs["msa_attention_mask"], _msa_inputs["has_deletion"], _msa_inputs["deletion_value"],
                    max_depth=_msa_inputs["max_depth"], enabled=_msa_inputs["subsample_enabled"])
                B_msa, M, L_msa = msa_i.shape
                msa_oh = F.one_hot(msa_i.permute(0, 2, 1).long(), num_classes=MOD.NUM_RES_TYPES).float()
                msa_attn = (mask_i.permute(0, 2, 1).float() if mask_i is not None else tok_mask[:, :, None].expand(-1, -1, M).float())
                msa_oh = msa_oh * msa_attn.unsqueeze(-1)
                hd = (hd_i.permute(0, 2, 1).float() if hd_i is not None else torch.zeros(B_msa, L_msa, M, device=msa_i.device))
                dv = (dv_i.permute(0, 2, 1).float() if dv_i is not None else torch.zeros(B_msa, L_msa, M, device=msa_i.device))
                x_pair_arg = z_inject_pair; x_inputs_arg = _msa_inputs["x_inputs"]
                if getattr(msa_enc, "_ef2opt_graphed", False):
                    kw = dict(x_pair=z_inject_pair, x_inputs=_msa_inputs["x_inputs"], msa_oh=msa_oh, has_deletion=hd, deletion_value=dv, msa_attention_mask=msa_attn)
                    ent = generic_static_entry(msa_enc, (), kw)
                    if ent is not None and z_inject_pair is z_init:                    # x_pair is the fold's z_init on this branch: loop-invariant
                        s_kwargs = ent[2]                                              # static kwargs: cloned BEFORE capture, outside the graph pool
                        if staged["ent"] is not ent or staged["x_pair_src"] != z_init.data_ptr() or staged["x_inputs_src"] != _msa_inputs["x_inputs"].data_ptr():
                            _tree_copy_(s_kwargs["x_pair"], z_inject_pair); _tree_copy_(s_kwargs["x_inputs"], _msa_inputs["x_inputs"])
                            staged["ent"] = ent; staged["x_pair_src"] = z_init.data_ptr(); staged["x_inputs_src"] = _msa_inputs["x_inputs"].data_ptr()
                            STATS["loop_static_msa_staged"] += 1
                        else:
                            STATS["loop_static_msa_copy_elided"] += 1
                        x_pair_arg = s_kwargs["x_pair"]; x_inputs_arg = s_kwargs["x_inputs"]
                msa_pair = msa_enc(x_pair=x_pair_arg, x_inputs=x_inputs_arg, msa_oh=msa_oh, has_deletion=hd, deletion_value=dv,
                                   msa_attention_mask=msa_attn).to(z_inject_pair.dtype)
                msa_oh = msa_attn = hd = dv = x_pair_arg = x_inputs_arg = None            # (4) the draw's inputs are dead once encoded
                z_inject_pair = (msa_pair if self.config.msa_encoder_overwrite else (z_inject_pair + msa_pair))
                msa_pair = None                                                       # (4)

            if refined_lm_z is not None:
                z_inject_pair = z_inject_pair + refined_lm_z.to(z_inject_pair.dtype)
            refined_lm_z = None                                                       # (4) dead once added (a graph's static output when aliased: nothing freed then)

            injected_pair = self.parcae_input_norm(z_inject_pair)
            z_inject_pair = None                                                      # (4) dead once normalised (z_init itself is the fold's and stays)
            az = az_next if az_next is not None else a * z                            # upstream: z = a * z + F.linear(injected_pair.to(z.dtype), b_mat)
            az_next = None
            z_dtype = z.dtype
            z = None                                                                  # (4) the previous trunk output is dead once scaled (upstream frees it when the add rebinds z)
            lin = F.linear(injected_pair.to(z_dtype), b_mat)
            del injected_pair                                                         # (4) dead once projected
            ent_t = ft._ef2opt_graphs.get(((tuple(torch.broadcast_shapes(az.shape, lin.shape)), torch.result_type(az, lin)), _tkey(pair_mask)))
            if ent_t is not None:
                z_arg = ent_t[1]                                                      # the trunk graph's static input (cloned before capture): the add writes it in place
                torch.add(az, lin, out=z_arg)
                STATS["loop_static_trunk_staged"] += 1
            else:
                z_arg = az + lin
            del az, lin
            # (4) every DEAD pair temporary of the recycle is released by now, each at its last use above: the per-loop lm_z draw, its cast, the
            # LM / MSA encoders' outputs once added, z_inject_pair once normalised, injected_pair once projected, the previous trunk output once
            # scaled — only z_arg (the trunk's input) and the fold's own tensors are alive through the trunk. Reference drops only
            # (arithmetic-free, RNG order untouched): the XL add-on's x2 LOOPFREE statement performed in G4's frame (a callee cannot release
            # the caller's locals), counted per recycle for the memory line's census.
            STATS["loop_static_released"] += 1
            z = ft(z_arg, pair_attention_mask=pair_mask)
            z_arg = None                                                              # (4) the trunk's input is dead once it returned (upstream's rebinding of z frees it there)
            if it + 1 < total_steps and _is_static_out(ft, z):
                az_next = a * z                                                       # consume the aliased trunk output NOW, before another graph of the pool replays
    finally:
        for m in alias_mods:
            m._ef2opt_out_alias = False
    if _is_static_out(ft, z):
        z = z.clone()                                                                 # the loop's result never aliases a graph's static buffer
    return z


def _is_static_out(mod, t):
    """Whether `t` is (the storage of) one of the graphed module's static output buffers."""
    cache = getattr(mod, "_ef2opt_graphs", None)
    if not cache or not torch.is_tensor(t):
        return False
    return any(torch.is_tensor(e[3]) and e[3].data_ptr() == t.data_ptr() for e in cache.values())



# --------------------------------------------------------------------------------------------------------------
# G5: whole-recycle graph. The body of one recycle after its RNG prologue — lm_encoder(lm_x) [+ msa_encoder(...)] + injection
#     (LayerNorm, a*z + linear) + folding_trunk — is deterministic given (z, z_init, lm_x, the MSA draw, x_inputs, pair_mask, a, b_mat,
#     tok_mask), so it is captured ONCE per shape signature as a single CUDA graph over static copies of those tensors and replayed once
#     per recycle; the prologue (F.dropout on lm_z, maybe_subsample_msa) runs eagerly in upstream's order — the RNG stream is untouched —
#     and its outputs are staged into the statics (the lm dropout output through the same cast-copy upstream's .to() performs). The
#     loop-carried pair state lives in the static z buffer, which the graph itself updates at its end (a device copy: data movement only).
#     First recycle of a new signature: the body is issued eagerly (its warm-up); second: capture + replay; later recycles and later folds of
#     the signature: replay. One signature per generation (a new one resets the generation, as G1 does). Exact tier.
# --------------------------------------------------------------------------------------------------------------
def _recycle_body(model, S, MOD):
    """The deterministic body of one recycle on the tensors in S (live tensors when issued eagerly, the statics under capture). Returns the new z."""
    F = MOD.F
    lm_enc = model.lm_encoder; msa_enc = model.msa_encoder
    z = S["z"]; z_init = S["z_init"]; pair_mask = S["pair_mask"]
    refined_lm_z = None
    if S["lm_x"] is not None and lm_enc is not None:
        refined_lm_z = lm_enc(S["lm_x"], pair_attention_mask=pair_mask)
    z_inject_pair = z_init
    if S["lm_x"] is not None and lm_enc is None:
        z_inject_pair = z_inject_pair + S["lm_x"]                      # lm_x is already lm_z_i.to(z_inject_pair.dtype) (z_init.dtype)
    if S["msa"] is not None:
        msa_i, mask_i, hd_i, dv_i = S["msa"]
        B_msa, M, L_msa = msa_i.shape
        msa_oh = F.one_hot(msa_i.permute(0, 2, 1).long(), num_classes=MOD.NUM_RES_TYPES).float()
        msa_attn = (mask_i.permute(0, 2, 1).float() if mask_i is not None else S["tok_mask"][:, :, None].expand(-1, -1, M).float())
        msa_oh = msa_oh * msa_attn.unsqueeze(-1)
        hd = (hd_i.permute(0, 2, 1).float() if hd_i is not None else torch.zeros(B_msa, L_msa, M, device=msa_i.device))
        dv = (dv_i.permute(0, 2, 1).float() if dv_i is not None else torch.zeros(B_msa, L_msa, M, device=msa_i.device))
        msa_pair = msa_enc(x_pair=z_inject_pair, x_inputs=S["x_inputs"], msa_oh=msa_oh, has_deletion=hd, deletion_value=dv,
                           msa_attention_mask=msa_attn).to(z_inject_pair.dtype)
        z_inject_pair = (msa_pair if model.config.msa_encoder_overwrite else (z_inject_pair + msa_pair))
    if refined_lm_z is not None:
        z_inject_pair = z_inject_pair + refined_lm_z.to(z_inject_pair.dtype)
    injected_pair = model.parcae_input_norm(z_inject_pair)
    z = S["a"] * z + F.linear(injected_pair.to(z.dtype), S["b_mat"])
    return model.folding_trunk(z, pair_attention_mask=pair_mask)


class _RecycleGraph:
    def __init__(self, eager_left):
        self.graph = None; self.S = None; self.eager_left = eager_left

    def capture(self, model, live, MOD):
        """Clone every body input into a static (outside the pool), capture the body + the in-place update of the static z, then replay once:
        the capture only recorded the kernels, the replay IS this recycle. Leaves S["z"] = the new pair state."""
        t0 = time.perf_counter()
        S = {k: _tree_clone(v) for k, v in live.items()}
        dev = S["z"].device
        rng_before = torch.cuda.get_rng_state(dev)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=_pool()):
            z_new = _recycle_body(model, S, MOD)
            S["z"].copy_(z_new)                                          # the loop-carried state stays in the static buffer (data movement only)
        torch.cuda.synchronize()
        if not torch.equal(rng_before, torch.cuda.get_rng_state(dev)):
            torch.cuda.set_rng_state(rng_before, dev)
            del g
            raise RuntimeError("ef2_opt G5: the recycle body consumed RNG under capture (a dropout inside the encoders/trunk is active) - refusing whole-recycle graphs")
        self.graph = g; self.S = S
        STATS["recycle_graph_captures"] += 1; STATS["recycle_graph_capture_s"] += time.perf_counter() - t0
        g.replay()
        STATS["recycle_graph_replays"] += 1


def _run_one_loop_g5(self, z, z_init, lm_z, _msa_inputs, pair_mask, a, b_mat, tok_mask, total_steps, MOD):
    """G5 driver: upstream's recycle loop with the RNG prologue eager and the deterministic body through one CUDA graph. Returns the loop's z,
    or None — decided BEFORE any work or RNG draw — when the shape is not G5's (the caller then runs the loop the G4 way)."""
    B, N = int(z.shape[0]), int(z.shape[1])
    if plane_tokens(N, B) > CFG.recycle_graph_max_tokens or _over_graph_budget("recycle", N, batch=B, plane_bytes=z.numel() * z.element_size()):
        return None
    F = MOD.F
    lm_cfg = self.config.lm_encoder
    _per_loop_lm_dropout = (lm_z is not None and getattr(lm_cfg, "per_loop_lm_dropout", False) and getattr(lm_cfg, "lm_dropout", 0.0) > 0.0)
    _lm_dropout_p = getattr(lm_cfg, "lm_dropout", 0.0)
    msa_runs = self.msa_encoder is not None and _msa_inputs is not None
    x_inputs = _msa_inputs["x_inputs"] if msa_runs else None
    tmask = tok_mask if msa_runs else None
    cache = self._ef2opt_graphs                                          # OrderedDict on the model (registered in _GRAPHED_MODULES: generation clears drop it)
    rg = None
    CFG._body_eager = True
    try:
        for it in range(total_steps):
            # --- RNG prologue, upstream's order: lm dropout, then the MSA row subsample (eager; the graph body is RNG-free)
            if _per_loop_lm_dropout:
                lm_z_i = F.dropout(lm_z, p=_lm_dropout_p, training=True)
            else:
                lm_z_i = lm_z
            msa_draw = None
            if msa_runs:
                msa_draw = tuple(MOD.maybe_subsample_msa(_msa_inputs["msa"], _msa_inputs["msa_attention_mask"], _msa_inputs["has_deletion"], _msa_inputs["deletion_value"],
                                                        max_depth=_msa_inputs["max_depth"], enabled=_msa_inputs["subsample_enabled"]))
            if it == 0:
                lm_sig = None if lm_z_i is None else ("T", tuple(lm_z_i.shape), z_init.dtype)
                sig = (tuple(_tree_sig(t) for t in (z, z_init, msa_draw, x_inputs, pair_mask, a, b_mat, tmask)), lm_sig,
                       ("g5", bool(_per_loop_lm_dropout), bool(msa_runs), bool(self.config.msa_encoder_overwrite) if msa_runs else None, self.lm_encoder is None))
                rg = cache.get(sig)
                if rg is None:
                    if len(cache) >= 1:
                        clear_graphs(reason="new_shape_recycle")         # one recycle signature per generation (as G1: a new shape re-captures everything)
                    warm = self.__dict__.setdefault("_ef2opt_warm", set())
                    rg = _RecycleGraph(eager_left=0 if sig in warm else 1)   # one eager recycle (the body's warm-up) before the capture, once per signature and process
                    warm.add(sig); cache[sig] = rg
                else:
                    cache.move_to_end(sig)
                    if rg.graph is not None:                             # a resident graph of this signature: this fold's invariants + initial state into its statics
                        for k, v in (("z", z), ("z_init", z_init), ("x_inputs", x_inputs), ("pair_mask", pair_mask), ("a", a), ("b_mat", b_mat), ("tok_mask", tmask)):
                            if v is not None:
                                _tree_copy_(rg.S[k], v)
                        STATS["recycle_fold_staged"] += 1
            resident = rg.graph is not None
            # --- the prologue's outputs: into the statics when the graph is resident, else materialised as upstream does
            if lm_z_i is None:
                lm_x = None
            elif resident:
                if _per_loop_lm_dropout or it == 0:
                    rg.S["lm_x"].copy_(lm_z_i)                           # upstream's lm_z_i.to(z_init.dtype) = empty + copy_: the same copy kernel, into the static
                    STATS["recycle_lm_staged"] += 1
                lm_x = rg.S["lm_x"]
            else:
                lm_x = lm_z_i.to(z_init.dtype)
            if msa_draw is not None and resident:
                _tree_copy_(rg.S["msa"], msa_draw)
            live = {"z": (rg.S["z"] if resident else z), "z_init": z_init, "lm_x": lm_x, "msa": msa_draw, "x_inputs": x_inputs,
                    "pair_mask": pair_mask, "a": a, "b_mat": b_mat, "tok_mask": tmask}
            if not resident:
                if rg.eager_left > 0:
                    rg.eager_left -= 1
                    z = _recycle_body(self, live, MOD)                   # eager recycle: upstream's statements on the live tensors
                    STATS["recycle_eager"] += 1
                else:
                    rg.capture(self, live, MOD)                          # statics cloned from this recycle's inputs; capture; one replay = this recycle
                    z = rg.S["z"]
                continue
            rg.graph.replay()
            STATS["recycle_graph_replays"] += 1
            z = rg.S["z"]
    finally:
        CFG._body_eager = False
    if rg is not None and rg.graph is not None and z.data_ptr() == rg.S["z"].data_ptr():
        z = z.clone()                                                    # the loop's result never aliases the static state buffer
    return z


def install_recycle_graph(model):
    """G5: registers the model as a graph holder (its _ef2opt_graphs holds the recycle graph; generation clears drop it) and enables the G5
    route inside _run_one_loop_v2 (G4 must be installed: it owns the loop)."""
    if not getattr(model, "_ef2opt_loop_static", False):
        return False
    if not hasattr(model, "_ef2opt_graphs"):
        model._ef2opt_graphs = collections.OrderedDict()
        model._ef2opt_name = "recycle"
        _GRAPHED_MODULES.append(model)
    CFG.recycle_graph = True
    return True


def install_loop_static(model):
    """G4: route model._run_one_loop through _run_one_loop_v2 (instance attribute; upstream's bound method kept as _ef2opt_eager_run_one_loop)."""
    if getattr(model, "_ef2opt_loop_static", False):
        return True
    if not callable(getattr(model, "_run_one_loop", None)) or getattr(model, "folding_trunk", None) is None:
        return False
    model._ef2opt_eager_run_one_loop = model._run_one_loop
    model._run_one_loop = types.MethodType(_run_one_loop_v2, model)
    model._ef2opt_loop_static = True
    CFG.loop_static = True
    return True


# --------------------------------------------------------------------------------------------------------------
# Pair-bias launch bound. The upstream fused pair-bias kernel (transformers kernels/fused_attention_pair_bias.py:
# _pair_bias_kernel — LN(z) @ pair_bias_proj + key mask for AttentionPairBias, the diffusion token transformer's
# attention) forms its element offsets in 32-bit integers: z [B, Q, K, DIM_Z] is read at
# `pid_b*Q*K*DIM_Z + pid_q*K*DIM_Z + k*DIM_Z + c` and the output [B, H, Q, K] written at
# `pid_b*H*Q*K + h*Q*K + pid_q*K + k`. A pair plane with B*Q*K*max(DIM_Z, H) > 2**31 - 1 (DIM_Z = 256, one
# sample: 2897 tokens and more) wraps the row offsets negative — an illegal address, not an out-of-memory error.
# fused_pair_bias_rows() launches such a plane one sample at a time in ROW BLOCKS of z that each stay inside the
# bound (every block a contiguous view of z: nothing is copied in) and writes the blocks' [1, H, rows, K] outputs into
# one [B, H, Q, K] buffer: the same kernel with the same per-element arithmetic (its LN and projection read one (q, k)
# row of z; B and Q enter only the address and the grid), so the bytes equal the single launch's. A plane inside the
# bound is upstream's ONE launch, its call untouched. install() routes the upstream module's `_fused_pair_bias`
# through it (install_pair_bias_rows), so every caller in the process — the stock AttentionPairBias.forward (which
# resolves that module global at call time), the pair-bias cache below, ef2_mk_sampler — launches inside the bound;
# the EXIT line's ef2_opt stats count the blocked launches (pair_bias_row_launches). upstream/U5_pair_bias_int64_offsets.md.
# --------------------------------------------------------------------------------------------------------------
PAIR_BIAS_OFFSET_MAX = 2**31 - 1                                   # the kernel's offsets are int32: the largest element offset one launch may form


def pair_bias_launch_rows(B, Q, K, dim_z, num_heads, bound=PAIR_BIAS_OFFSET_MAX):
    """Rows of z per launch of the fused pair-bias kernel so every element offset it forms stays within `bound`: a launch over
    [B, rows, K, DIM_Z] reaches B*rows*K*DIM_Z - 1 in z and B*H*rows*K - 1 in its output. Returns Q when one launch covers the
    plane (the bound is not reached), else the block height (>= 1), or 0 when even one row exceeds the bound."""
    per_row = int(B) * int(K) * max(int(dim_z), int(num_heads))
    if per_row <= 0 or int(B) * int(Q) * int(K) * max(int(dim_z), int(num_heads)) <= bound:
        return int(Q)
    return min(int(Q), bound // per_row)


def fused_pair_bias_rows(fused, z, mask, w_proj_z, pair_norm_w=None, pair_norm_b=None, *, num_heads, bound=PAIR_BIAS_OFFSET_MAX, **kw):
    """upstream's `fused_pair_bias(z, mask, w_proj_z, pair_norm_w, pair_norm_b, num_heads=…)` (bias [B, H, Q, K]) launched inside the
    kernel's offset bound: one call when the plane fits (the call as given), else one call per (sample, row block) of z — z[b:b+1, s:e]
    with that sample's key mask mask[b:b+1] — the blocks' outputs written into one [B, H, Q, K] buffer (same kernel, same per-element
    arithmetic)."""
    B, Q, K, D = z.shape
    if pair_bias_launch_rows(B, Q, K, D, num_heads, bound) >= Q:
        return fused(z, mask, w_proj_z, pair_norm_w, pair_norm_b, num_heads=num_heads, **kw)
    rows = pair_bias_launch_rows(1, Q, K, D, num_heads, bound)         # per sample: the block height does not shrink with B and every block is a contiguous view
    if rows < 1:
        raise RuntimeError(f"ef2_opt: pair-bias plane [B={B}, K={K}, DIM_Z={D}, heads={num_heads}]: one row of z exceeds the fused pair-bias kernel's "
                           f"32-bit element offsets ({bound}); no launch can cover it")
    out = None
    for b in range(B):
        zb = z[b:b + 1]
        mb = None if mask is None else mask[b:b + 1]
        for s in range(0, Q, rows):
            e = min(s + rows, Q)
            blk = fused(zb[:, s:e], mb, w_proj_z, pair_norm_w, pair_norm_b, num_heads=num_heads, **kw)
            if out is None:
                out = torch.empty((B, blk.shape[1], Q, K), dtype=blk.dtype, device=blk.device)
            out[b:b + 1, :, s:e].copy_(blk)
            del blk
            STATS["pair_bias_row_launches"] += 1
    return out


def install_pair_bias_rows():
    """Route the upstream module's `_fused_pair_bias` through fused_pair_bias_rows (idempotent; the module keeps the routed function for the
    process). Returns whether the module carries the fused kernel at all (it is None where the Triton kernels do not import)."""
    C = _common()
    inner = getattr(C, "_fused_pair_bias", None)
    if inner is None:
        return False
    if getattr(inner, "_ef2opt_rows", False):
        return True

    def _fused_pair_bias_in_bound(z, mask, w_proj_z, pair_norm_w=None, pair_norm_b=None, *, num_heads, **kw):
        return fused_pair_bias_rows(inner, z, mask, w_proj_z, pair_norm_w, pair_norm_b, num_heads=num_heads, **kw)

    _fused_pair_bias_in_bound._ef2opt_rows = True
    _fused_pair_bias_in_bound._ef2opt_inner = inner
    _fused_pair_bias_in_bound.__wrapped__ = inner
    C._fused_pair_bias = _fused_pair_bias_in_bound
    return True


# --------------------------------------------------------------------------------------------------------------
# P1: per-fold pair-bias cache for the 12 AttentionPairBias blocks of the diffusion token transformer
#     z (pair conditioning) is fixed during sampling, so
#     pair_norm(z) -> pair_bias_proj (or the fused _pair_bias_kernel incl. mask) is a pure function of (z, weights, mask)
#     and is recomputed 68x per fold by the stock code. We compute it at step 0 (eager) into a per-(module, shape)
#     static buffer and reuse it for steps 1..67 (graph replays read the same buffer). Bitwise identical by construction
#     (same kernel, same inputs); probe mode asserts max|diff| == 0 against a fresh recompute.
# --------------------------------------------------------------------------------------------------------------
_PB_EPOCH = {"n": 0}


def _apb_forward_cached(self, a, s, z, beta=0.0, attention_mask=None, num_diffusion_samples=1, **kw):
    C = _common()
    eager = self._ef2opt_eager_forward
    if kw or (not CFG.enabled) or torch.is_grad_enabled() or z is None or z.dim() != 4 or not a.is_cuda:
        return eager(a, s, z, beta=beta, attention_mask=attention_mask, num_diffusion_samples=num_diffusion_samples, **kw)
    bsz, n_queries, d_model = a.shape
    if s is not None:
        x = self.adaln(a, s)
    else:
        x = self.pre_norm(a)
    n_keys = x.shape[1]
    q = self.q_proj(x).view(bsz, n_queries, self.num_heads, self.head_dim)
    kv = self.kv_proj(x)
    k, v = kv.chunk(2, dim=-1)
    k = k.view(bsz, n_keys, self.num_heads, self.head_dim)
    v = v.view(bsz, n_keys, self.num_heads, self.head_dim)
    if z.shape[0] != bsz and num_diffusion_samples > 1:
        z = z.repeat_interleave(num_diffusion_samples, dim=0)
    if attention_mask is not None and attention_mask.shape[0] != bsz and num_diffusion_samples > 1:
        attention_mask = attention_mask.repeat_interleave(num_diffusion_samples, dim=0)
    use_fused = self._can_use_fused_pair_bias(z, n_queries, beta)
    if (not use_fused) and self._can_use_cueq_pair_bias(z, n_queries, beta):
        return eager(a, s, z, beta=beta, attention_mask=attention_mask, num_diffusion_samples=num_diffusion_samples)
    key = ("fused" if use_fused else "ref", tuple(z.shape), z.dtype, None if attention_mask is None else tuple(attention_mask.shape))
    ent = self._ef2opt_pb.get(key)

    def compute_bias():
        if use_fused:
            kernel_mask = attention_mask if attention_mask is not None else torch.ones(bsz, n_queries, device=a.device, dtype=torch.bool)
            pair_norm_w = self.pair_norm.weight
            pair_norm_b = self.pair_norm.bias if self.pair_norm.bias is not None else torch.zeros_like(pair_norm_w)
            z_bf = z if z.dtype == torch.bfloat16 else z.to(torch.bfloat16)
            return C._fused_pair_bias(z_bf, kernel_mask, self.pair_bias_proj.weight, num_heads=self.num_heads, pair_norm_w=pair_norm_w, pair_norm_b=pair_norm_b)
        return self.pair_bias_proj(self.pair_norm(z))

    if ent is None or ent[1] != _PB_EPOCH["n"]:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("pair-bias cache miss during graph capture (step 0 must run eagerly first)")
        bias_new = compute_bias()
        if ent is None:
            # NEW static buffer for this (module, shape). A captured sampler graph bakes in the buffer ADDRESS it read at capture
            # time, so (a) buffers are never re-allocated for a known key (always copy_ in place) and (b) whenever a new buffer is
            # created, all captured sampler graphs are dropped so the next capture reads the live buffer. (The hazard: LRU eviction +
            # re-allocation would leave replayed graphs reading a stale buffer of an earlier design with the same token length.)
            # A new key can only require a generation reset when this block's own LRU is full. (Resetting whenever ANY
            # sampler graph existed would, with two model instances (Fast+Full) in one process, fire at every variant switch and
            # wipe the trunk graphs just captured for the new item: correct but ~2x recaptures. A same-shape sampler graph of this
            # block's model cannot exist without this key's buffer (buffers and graphs are only ever dropped together), and graphs of
            # other shapes/models never read this new buffer.)
            if len(self._ef2opt_pb) >= CFG.lru_pair_bias:
                clear_graphs(reason="pair_bias_new_key")          # whole-generation reset (also empties self._ef2opt_pb)
            buf = bias_new.clone()
            STATS["pair_bias_new_buffer"] += 1
        else:
            buf = ent[0]; buf.copy_(bias_new)
        buf32 = None
        if CFG.bias_f32 and use_fused and buf.dtype != torch.float32:
            buf32 = ent[2] if (ent is not None and len(ent) > 2 and torch.is_tensor(ent[2]) and ent[2].shape == buf.shape) else None
            if buf32 is None:
                buf32 = buf.to(torch.float32); STATS["pair_bias_f32_new_buffer"] += 1     # once per (block, shape): static like `buf`
            else:
                buf32.copy_(buf)                                                        # == buf.to(float32) elementwise
            STATS["pair_bias_f32_refresh"] += 1
        ent = (buf, _PB_EPOCH["n"], buf32); self._ef2opt_pb[key] = ent
        STATS["pair_bias_computes"] += 1
    else:
        STATS["pair_bias_cache_hits"] += 1
        if CFG.probe and not torch.cuda.is_current_stream_capturing():
            VERIFY_LOG.append(("pair_bias", "apb", tuple(ent[0].shape), (compute_bias().float() - ent[0].float()).abs().max().item()))
    bias = ent[0]
    if use_fused:
        q_bhqd = q.transpose(1, 2); k_bhqd = k.transpose(1, 2); v_bhqd = v.transpose(1, 2)
        bias32 = ent[2] if (len(ent) > 2 and ent[2] is not None and q_bhqd.dtype == torch.float32) else None
        if bias32 is not None:
            STATS["pair_bias_f32_served"] += 1
        attn_out = F.scaled_dot_product_attention(q_bhqd, k_bhqd, v_bhqd, attn_mask=(bias32 if bias32 is not None else bias.to(q_bhqd.dtype)))
        g = torch.sigmoid(self.g_proj(x)).view(bsz, n_queries, self.num_heads, self.head_dim)
        ctx = g * attn_out.transpose(1, 2)
        out = self.out_proj(ctx.reshape(bsz, n_queries, d_model))
        if s is not None:
            out = torch.sigmoid(self.out_gate(s)) * out
        return out
    g = torch.sigmoid(self.g_proj(x)).view(bsz, n_queries, self.num_heads, self.head_dim)
    logits = torch.einsum("... i h d, ... j h d -> ... i j h", q, k) * self.scale
    logits = logits + bias.to(dtype=logits.dtype)
    if attention_mask is not None:
        min_val = torch.finfo(logits.dtype).min
        mask_bias = torch.where(attention_mask.bool()[:, None, :, None], 0.0, min_val)
        logits = logits + mask_bias.to(dtype=logits.dtype)
    attn = torch.softmax(logits, dim=-2).to(dtype=v.dtype)
    ctx = torch.einsum("... i j h, ... j h d -> ... i h d", attn, v)
    ctx = g * ctx
    out = self.out_proj(ctx.reshape(bsz, n_queries, d_model))
    if s is not None:
        out = torch.sigmoid(self.out_gate(s)) * out
    return out


def install_pair_bias_cache(model):
    C = _common()
    sh = model.structure_head
    dm = sh.diffusion_module if hasattr(sh, "diffusion_module") else None
    tt = None
    for m in sh.modules():
        if isinstance(m, C.DiffusionTransformer):
            tt = m; break
    n = 0
    if tt is None:
        return 0
    for blk in tt.attn_blocks:
        if isinstance(blk, C.AttentionPairBias) and hasattr(blk, "pair_norm") and not getattr(blk, "_ef2opt_pb_installed", False):
            blk._ef2opt_eager_forward = blk.forward
            blk._ef2opt_pb = collections.OrderedDict()
            blk.forward = types.MethodType(_apb_forward_cached, blk)
            blk._ef2opt_pb_installed = True
            _PB_BLOCKS.append(blk)
            n += 1
    # epoch bump at every sample() call (wraps whatever sample is installed: stock or _sample_v2)
    if not getattr(sh, "_ef2opt_pb_epoch_wrapped", False):
        inner = sh.sample

        def sample_with_epoch(*a, **k):
            _PB_EPOCH["n"] += 1
            return inner(*a, **k)
        sh.sample = sample_with_epoch
        sh._ef2opt_pb_epoch_wrapped = True
    return n



# --------------------------------------------------------------------------------------------------------------
# C1/C2: exact caches for seed-independent per-design work, each proven identical on its second call (caching of provably identical
# intermediate results). Protocol per cache key: call 1 -> compute + store; call 2 -> compute AGAIN and compare
# bitwise with the stored value (torch.equal on every tensor); only if identical is the entry marked confirmed and
# served from call 3 on; any mismatch disables caching for that key permanently (and is counted). So every cached
# value has been proven bitwise-reproducible for that very design on that very host/process, and 2 of N seeds are
# always computed fresh. RNG: prepare_input runs inside esm's _seed_context (RNG state restored on exit) and the
# ESMC forward is checked to not advance the CUDA RNG (else the cache is disabled), so the per-seed RNG stream seen
# by dropout / MSA subsampling / diffusion noise is unchanged.
# --------------------------------------------------------------------------------------------------------------
class _VerifiedCache:
    def __init__(self, name, max_entries=4):
        self.name, self.max_entries = name, max_entries
        self.d = collections.OrderedDict()

    def clear(self):
        self.d.clear(); STATS[f"{self.name}_clears"] += 1

    def get(self, key, compute, equal):
        numerics_mode_guard()
        ent = self.d.get(key)
        if ent is not None and ent["state"] == "confirmed":
            self.d.move_to_end(key); STATS[f"{self.name}_hits"] += 1
            return ent["value"]
        if ent is not None and ent["state"] == "disabled":
            STATS[f"{self.name}_disabled_calls"] += 1
            return compute()
        val = compute()
        if ent is None:
            while len(self.d) >= self.max_entries:
                self.d.pop(next(iter(self.d)))
            self.d[key] = dict(state="new", value=val); STATS[f"{self.name}_stores"] += 1
            return val
        # second call: confirm
        try:
            same = equal(ent["value"], val)
        except Exception as e:
            from opt_core.oom import is_oom
            if is_oom(e): raise
            same = False
        if same:
            ent["state"] = "confirmed"; STATS[f"{self.name}_confirmed"] += 1
        else:
            ent["state"] = "disabled"; ent["value"] = None; STATS[f"{self.name}_MISMATCH_disabled"] += 1
        return val


def _tensors_equal_tree(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_tensors_equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_tensors_equal_tree(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b) or repr(a) == repr(b)
    except Exception:
        return repr(a) == repr(b)


_FEATURE_CACHE = _VerifiedCache("feature_cache", max_entries=int(os.environ.get("EF2_FEATURE_CACHE_N", "4")))
_ESMC_CACHE = _VerifiedCache("esmc_cache", max_entries=int(os.environ.get("EF2_ESMC_CACHE_N", "2")))
_FEATURE_CACHE_REF["obj"] = _FEATURE_CACHE; _ESMC_CACHE_REF["obj"] = _ESMC_CACHE


def ensure_cache_capacity(n_items):
    """xb mode: the exact feature/ESMC caches must hold every item of a length group that is cycled across seeds (else they thrash: store, never hit)."""
    for c in (_FEATURE_CACHE, _ESMC_CACHE):
        if c.max_entries < n_items:
            c.max_entries = int(n_items); STATS[f"{c.name}_capacity"] = int(n_items)


def _spi_content_key(input):
    """Hashable CONTENT signature of a StructurePredictionInput: per chain (class, id, sequence / smiles / ccd, modifications,
    MSA depth + sha1 of all MSA rows), plus pocket / distogram / covalent-bond conditioning reprs. Two inputs with equal keys
    are the same folding problem. Cheap: one pass over the MSA strings (~ms for 2048 x 300)."""
    import hashlib
    parts = [type(input).__name__]
    seqs = getattr(input, "sequences", None)
    if seqs is None:
        return ("repr", repr(input))
    for ch in seqs:
        h = hashlib.sha1()
        msa = getattr(ch, "msa", None)
        depth = -1
        if msa is not None:
            try:
                entries = getattr(msa, "entries", None) or getattr(msa, "sequences", None) or []
                depth = 0
                for e in entries:
                    sq = getattr(e, "sequence", e)
                    h.update(str(sq).encode()); h.update(b"|"); depth += 1
            except Exception:
                h.update(repr(msa).encode())
        parts.append((type(ch).__name__, getattr(ch, "id", None), getattr(ch, "sequence", None), getattr(ch, "smiles", None), getattr(ch, "ccd", None),
                      repr(getattr(ch, "modifications", None)), depth, h.hexdigest()))
    for extra in ("pocket", "distogram_conditioning", "covalent_bonds"):
        parts.append(repr(getattr(input, extra, None)))
    return hashlib.sha1(repr(parts).encode()).hexdigest()          # a plain string: always hashable (chain fields may be lists)


def install_feature_cache(builder):
    """cache ESMFold2InputBuilder.prepare_input(input, seed, device) per (input object, device) across seeds."""
    if getattr(builder, "_ef2opt_fc", False):
        return False
    orig = builder.prepare_input
    pins = collections.OrderedDict()          # keep the input objects alive so id() cannot be recycled while cached

    def prepare_input_cached(input, seed=None, device=None):
        if not CFG.enabled:
            return orig(input, seed=seed, device=device)
        key = (_spi_content_key(input), str(device))     # CONTENT key (all chain sequences / ligands / msa paths), never id():
        pins[key] = input                                   # id() of a freed input object can be recycled by the next design (seen in
        while len(pins) > _FEATURE_CACHE.max_entries:       # the multi-seed path: a same-shape next design was served stale features)
            pins.pop(next(iter(pins)))
        val = _FEATURE_CACHE.get(key, lambda: orig(input, seed=seed, device=device), _tensors_equal_tree)
        feats, cinfos = val
        return dict(feats), cinfos               # fresh dict object, same (read-only) tensors
    builder.prepare_input = prepare_input_cached
    builder._ef2opt_fc = True
    return True


def install_esmc_cache(model):
    """cache model._compute_lm_hidden_states(input_ids, asym_id, residue_index, mol_type, tok_mask, lm_mask_pct) when lm_mask_pct == 0."""
    if getattr(model, "_ef2opt_ec", False):
        return False
    orig = model._compute_lm_hidden_states

    def lm_cached(input_ids, asym_id, residue_index, mol_type, tok_mask, lm_mask_pct=0.0):
        if (not CFG.enabled) or (lm_mask_pct or 0.0) > 0.0 or not input_ids.is_cuda:
            return orig(input_ids, asym_id, residue_index, mol_type, tok_mask, lm_mask_pct=lm_mask_pct)
        key = (id(model._esmc), bool(getattr(model, "_esmc_fp8", False)), tuple(input_ids.shape),
               tuple(input_ids.flatten().tolist()), tuple(asym_id.flatten().tolist()), tuple(residue_index.flatten().tolist()),
               tuple(mol_type.flatten().tolist()), tuple(tok_mask.flatten().to(torch.int8).tolist()))
        dev = input_ids.device

        def compute():
            rng = torch.cuda.get_rng_state(dev)
            out = orig(input_ids, asym_id, residue_index, mol_type, tok_mask, lm_mask_pct=lm_mask_pct)
            if not torch.equal(rng, torch.cuda.get_rng_state(dev)):
                STATS["esmc_cache_rng_advanced_DISABLED"] += 1
                CFG.esmc_cache_ok = False
            return out
        if not CFG.esmc_cache_ok:
            return compute()
        return _ESMC_CACHE.get(key, compute, _tensors_equal_tree)
    model._compute_lm_hidden_states = lm_cached
    model._ef2opt_ec = True
    return True


def install(model, trunk_graphs=True, sampler_graphs=True, fuse_msa_trimul=False, probe=False, swa_mask_cache=True, encoder_graphs=False, pair_bias_cache=False, esmc_cache=False, builder=None,
            loop_static=False, recycle_graph=False):
    if model not in _MODE["models"]:
        _MODE["models"].append(model)
    _MODE["sig"] = None      # re-baseline the numerics signature at install time
    CFG.probe = bool(probe)
    out = {}
    out["pair_bias_rows"] = install_pair_bias_rows()                   # every kit mode: the fused pair-bias kernel launches inside its 32-bit offset bound at any token count
    if swa_mask_cache or sampler_graphs:
        out["swa_mask_cache_modules"] = install_swa_mask_cache(model)
    if trunk_graphs:
        out["trunk_graphs"] = install_trunk_graphs(model)
    if sampler_graphs:
        out["sampler_graphs"] = install_sampler_graphs(model)
    if fuse_msa_trimul:
        out["msa_fused_trimul_blocks"] = install_msa_fused_trimul(model)
    if encoder_graphs:
        out["encoder_graphs"] = install_generic_graphs(model)
    if loop_static and trunk_graphs:
        import ef2_srcguard
        ef2_srcguard.check("ls")                                            # G4 re-issues ESMFold2Model._run_one_loop statement by statement: refuse by name on another upstream source
        out["loop_static"] = install_loop_static(model)                    # G4: after G1/G3 so the loop finds the graphed modules
    if recycle_graph and loop_static and trunk_graphs:
        enc = getattr(model, "msa_encoder", None)                           # G5 captures the recycle BODY once and replays it: no lever with a per-call host-side decision may sit inside
        for m_, what in ((enc, "msa_encoder"), (getattr(model, "lm_encoder", None), "lm_encoder"), (getattr(model, "folding_trunk", None), "folding_trunk")):
            if m_ is not None and (getattr(m_, "_ef2msa2_inner_forward", None) is not None or getattr(m_, "_host_decision_lever", None)):
                raise RuntimeError(f"ef2_opt G5 recycle_graph: a host-decision lever ({getattr(m_, '_host_decision_lever', 'mh')}) is installed on {what}; "
                                   "its per-call decision would be frozen into the recycle graph - refusing the composition by name")
        out["recycle_graph"] = install_recycle_graph(model)                # G5: inside G4's loop, small shapes
    if pair_bias_cache:
        out["pair_bias_cache_blocks"] = install_pair_bias_cache(model)   # after sampler_graphs so the epoch wrapper is outermost
    if esmc_cache:
        out["esmc_cache"] = install_esmc_cache(model)
    if builder is not None:
        out["feature_cache"] = install_feature_cache(builder)
    return out


def stats():
    d = dict(STATS)
    if VERIFY_LOG:
        d["verify_n"] = len(VERIFY_LOG)
        d["verify_max_abs_diff_trunk"] = max([v[3] for v in VERIFY_LOG if v[0] == "trunk"] or [0.0])
        d["verify_max_abs_diff_sampler"] = max([v[3] for v in VERIFY_LOG if v[0] == "sampler_step"] or [0.0])
        d["verify_max_abs_diff_generic"] = max([v[3] for v in VERIFY_LOG if v[0] == "generic"] or [0.0])
        d["verify_max_abs_diff_pair_bias"] = max([v[3] for v in VERIFY_LOG if v[0] == "pair_bias"] or [0.0])
        d["verify_n_pair_bias"] = len([v for v in VERIFY_LOG if v[0] == "pair_bias"])
    return d
