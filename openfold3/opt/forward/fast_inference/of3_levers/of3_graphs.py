"""Lever C3: CUDA-graph replay of the OpenFold3 diffusion denoising step (OpenFold3 fast-inference kit).

Stock SampleDiffusion.forward (suppl. Alg. 18) runs `for tau in 200 steps: xl = augment(xl); noise = randn; xl_denoised = diffusion_module(xl+noise, t, ...)`.
Each diffusion_module call launches ~2,500 small kernels (24-block diffusion transformer + atom-attention encoder/decoder); at <=300 tokens
the GPU sits idle for much of the step (launch-bound).  This lever captures ONE CUDA graph of `diffusion_module.forward` per shape key
(query: n_atoms, n_tokens, batch/sample dims, dtypes) after an eager warm-up step, and replays it for the remaining steps of this seed and
for every later seed/sample of the same query (5 seeds x 200 steps = 1,000 replays per capture).

RNG / numerics: all random draws of the sampler (initial xl, per-step random rotation+translation, per-step noise) stay OUTSIDE the graph,
in the stock order, with the stock shapes/dtypes -> the Philox stream is bit-identical to stock.  diffusion_module.forward contains no RNG op
at eval (dropout off) - asserted at capture time by comparing the CUDA RNG offset before/after the warm-up call.  The graph replays exactly the
kernels that eager mode launched at capture; results equal eager up to the run-to-run nondeterminism that stock itself has (atomics).
The eager warm-up call is an EXTRA evaluation of the step whose result is discarded (no state, no RNG) -> semantics unchanged.

Pool discipline (pool rule): ONE memory pool (torch.cuda.MemPool, held by this module) serves every capture of the process; graphs are never
evicted individually - on a new shape key the old (graph, static buffers, gather tables) generation is dropped as a whole before capturing the
new one (one generation kept), its blocks returning to the pool for the next capture (no allocator flush at a generation drop: a re-capture costs
roughly half the eager time).  The first capture of the process warms the step up twice on the side stream (lazy compilation settles there), later
captures once.  A new item whose token->atom gather tables differ from the captured item's at EQUAL shapes has the memoised tables refreshed IN
PLACE (re-derived with the stock function exactly as at record time and checked to reproduce the stock broadcast) and keeps the resident graph;
a table whose shape differs drops the generation and recaptures, by name.  Keep across items (OPENFOLD3_OPT_GRAPHS_KEEP, default on): the
capture key is taken over the tensors the step READS — the step's arguments plus the feature-dict entries the eager warm-up read, recorded by
name; the static input dict the graph is captured on holds exactly those entries — so a later predict item whose read entries have the captured
shapes/dtypes (same token/atom bucket, sample count, numerics mode, kernel flags) REPLAYS the resident generation after the per-item checks
(parameter storage unmoved, gather tables refreshed in place, every static buffer at its captured address, read feature values copied into the
static buffers) instead of re-capturing it because an entry the step never reads (the MSA / template stacks) changed shape; census `kept=<n>`.
A read entry of another shape, or OPENFOLD3_OPT_GRAPHS_KEEP=0 with any differing entry, drops and re-captures as before (one generation kept).
The sampler loop iterates a HOST copy of the noise schedule for its
one host-side comparison (`c_tau > gamma_min`), so no step synchronises the device; every piece of device arithmetic keeps its operands (the
device schedule elements), bitwise.  DeepSpeed's DS4Sci attention kernel launches on the legacy default stream (it races producers queued on a
side / capture stream and cannot be captured): a rollout that requests it (`use_deepspeed_evo_attention=True`: the stock-configuration lines)
runs with the flag off inside the sampler, named once (`ds4sci_off_rollouts`); cuEquivariance and Triton attention are capture-safe.
`release()` drops every generation, the gather tables and the pool (an item-boundary release under memory pressure: the kit's post_release
lever calls it); the next capture flushes the allocator cache before its warm-up (a side-stream warm-up cannot reuse the default stream's
cached-free segments).

Enable: OF3_CUDA_GRAPHS=1 (sitecustomize hook calls enable() after openfold3.core.model.structure.diffusion_module is imported).
Invariant line printed to stderr per capture: "[of3_graphs] captured key=... warmup_ms=... capture_ms=... rng_offset_unchanged=True".
"""
import os, sys, time
import torch
from opt_core.oom import is_oom                                   # every served fallback below re-raises an out-of-memory first (README: OOM propagates; no fallback applied)

_STOCK_FORWARD = None
_STATE = {"enabled": False, "gens": [], "pool": None, "released": False,
          "stats": {"captures": 0, "replays": 0, "eager_steps": 0, "fallbacks": 0, "gen_drops": 0, "kept": 0, "keep_drops": 0, "regather_inplace": 0, "regather_tables": 0,
                    "stale_gather_recaptures": 0, "warmup_steps": 0, "host_schedule_rollouts": 0, "ds4sci_off_rollouts": 0, "releases": 0,
                    "pool": "none"}}
KEEP_ENV = "OPENFOLD3_OPT_GRAPHS_KEEP"          # the kit's knob (modes.ENV_GRAPHS_KEEP): unset / 1 = a captured generation survives across predict items whose capture key is
                                                # equal on every tensor the step reads (default); 0 = drop and re-capture at every item whose feature-dict signature differs anywhere
_KEEP_WORDS = {"": True, "1": True, "on": True, "true": True, "yes": True, "0": False, "off": False, "false": False, "no": False}
_SEEN_ONCE = set()


def keep_enabled(environ=None):
    """OPENFOLD3_OPT_GRAPHS_KEEP as a bool: unset/1/on = keep across items (default), 0/off = the drop policy; any other word is named once and read as the default."""
    raw = ((os.environ if environ is None else environ).get(KEEP_ENV) or "").strip().lower()
    if raw in _KEEP_WORDS:
        return _KEEP_WORDS[raw]
    if ("keepword", raw) not in _SEEN_ONCE:
        _SEEN_ONCE.add(("keepword", raw)); _log(f"{KEEP_ENV}={raw!r} is not one of 1|0 (on|off): read as the default (keep across items)")
    return True


def key_names_differing(key_a, key_b):
    """Names of the components that differ between two capture keys (_make_key form: ((name, sig), ...) items + extra + mode) — the census word of a
    re-capture (`differs=`) and the input of the keep decision. Order: the items by name, then `extra` / `mode`."""
    ia, ib = dict(key_a[0]), dict(key_b[0])
    names = [n for n in sorted(set(ia) | set(ib)) if ia.get(n, _MISSING) != ib.get(n, _MISSING)]
    if key_a[1] != key_b[1]: names.append("extra")
    if key_a[2:] != key_b[2:]: names.append("mode")
    return names


def keep_decision(gen_key, read_names, new_key, keep=True):
    """The item-boundary verdict for a resident generation captured on `gen_key`, whose step read the feature entries `read_names` (None = unknown:
    every entry counts as read), met by a step whose key is `new_key`: ("hit", []) identical key; ("keep", unread) the keys differ only in feature
    entries the step never read (their names returned) and keep is on -> replay the generation; ("drop", names) a read entry / an argument / the
    numerics mode / the kernel flags differ, or keep is off -> drop and re-capture (the names that differ returned, read ones first)."""
    if gen_key == new_key:
        return "hit", []
    diff = key_names_differing(gen_key, new_key)
    if not keep:
        return "drop", diff
    read = [n for n in diff if n in _ARG_NAMES or n in ("extra", "mode") or read_names is None or n in read_names]
    if read:
        return "drop", read + [n for n in diff if n not in read]
    return "keep", diff


_MISSING = object()
_ARG_NAMES = ("xl", "t", "si", "s", "z")          # the step's tensor arguments in the key (always read)


class _ReadView(dict):
    """The feature dict as the eager warm-up sees it: records which entries the step reads (by name). Any whole-dict access (iteration, keys/items/values,
    copy) marks every entry read — the keep decision then compares the full signature, as before."""
    def __init__(self, d):
        super().__init__(d); self.read = set(); self.read_all = False
    def __getitem__(self, k):
        self.read.add(k); return dict.__getitem__(self, k)
    def get(self, k, default=None):
        self.read.add(k); return dict.get(self, k, default)
    def _all(self):
        self.read_all = True
    def __iter__(self):
        self._all(); return dict.__iter__(self)
    def keys(self):
        self._all(); return dict.keys(self)
    def items(self):
        self._all(); return dict.items(self)
    def values(self):
        self._all(); return dict.values(self)
    def copy(self):
        self._all(); return dict.copy(self)
    def __reduce_ex__(self, proto):
        self._all(); return dict.__reduce_ex__(self, proto)
    def read_names(self):
        return None if self.read_all else frozenset(self.read)

# --------------------------------------------------------------------------- graph-safe broadcast_token_feat_to_atoms
# Stock broadcast_token_feat_to_atoms (atomize_utils) does two host syncs per call: torch.repeat_interleave with a tensor `repeats`
# (output length) and `max_num_atoms = torch.max(...).int()` used as a reshape size.  Both depend only on (token_mask, num_atoms_per_token,
# max_num_atoms_per_token, shapes) which are constant for a query.  Memoised version: on first sight of a (token_mask, num_atoms_per_token)
# signature (checked by VALUE, eagerly, outside capture) we run the STOCK function on an arange "feature" to obtain the gather index and the
# output size, assert that index_select reproduces the stock output bit-exactly for the real feature, and thereafter compute
# atom_feat = index_select(padded_token_feat, idx) -> identical values (pure row copy), no host sync, capturable.
_BC_CACHE = []          # list of dict(token_mask, natpt (clones), max_apt, token_dim, feat_rank -> idx, max_num_atoms)
_BC_MODE = {"phase": "eager"}   # eager: stock | record: build+check cache entries | replay: cache only (raise if missing)
_BC_STOCK = None

def _bc_lookup(token_mask, natpt, max_apt, token_dim, feat_shape):
    for e in _BC_CACHE:
        if (e["max_apt"] == max_apt and e["token_dim"] == token_dim and e["feat_shape"] == tuple(feat_shape)
                and e["token_mask"].shape == token_mask.shape and e["natpt"].shape == natpt.shape):
            if _BC_MODE["phase"] == "replay":
                return e            # values were checked at record time; tensors are the static graph inputs
            if torch.equal(e["token_mask"], token_mask) and torch.equal(e["natpt"], natpt):
                return e
    return None

def _broadcast_graphsafe(token_mask, num_atoms_per_token, token_feat, token_dim=-1, max_num_atoms_per_token=None):
    if _BC_MODE["phase"] == "eager":
        return _BC_STOCK(token_mask=token_mask, num_atoms_per_token=num_atoms_per_token, token_feat=token_feat, token_dim=token_dim, max_num_atoms_per_token=max_num_atoms_per_token)
    e = _bc_lookup(token_mask, num_atoms_per_token, max_num_atoms_per_token, token_dim, token_feat.shape)
    if e is None:
        if _BC_MODE["phase"] == "replay":
            raise RuntimeError("broadcast_token_feat_to_atoms: unseen signature during capture")
        # record: derive flat gather index by pushing a row-id feature through the STOCK function
        n_token = token_mask.shape[-1]
        batch_dims = token_mask.shape[:-1]
        feat_batch_dims = token_feat.shape[:token_dim]
        feat_dims = token_feat.shape[token_dim:][1:]
        nrows = int(torch.tensor(feat_batch_dims).prod().item()) * n_token if len(feat_batch_dims) else n_token
        rowid = torch.arange(nrows, device=token_feat.device, dtype=torch.float64).reshape(*feat_batch_dims, n_token)
        # rowid has no feat dims; stock handles feat_dims=() fine. Padding rows map to value 0 of the zero pad -> mark pads with -1 by using rowid+1
        out = _BC_STOCK(token_mask=token_mask, num_atoms_per_token=num_atoms_per_token, token_feat=rowid + 1.0, token_dim=-1, max_num_atoms_per_token=max_num_atoms_per_token)
        # NOTE token_mask multiplies the feature in stock -> masked tokens give 0 too; both pads and masked tokens become "take zero row"
        flat = out.reshape(-1).round().long() - 1          # -1 => zero row
        max_num_atoms = out.shape[len(feat_batch_dims)]
        e = dict(token_mask=token_mask.detach().clone(), natpt=num_atoms_per_token.detach().clone(), max_apt=max_num_atoms_per_token, token_dim=token_dim,
                 feat_shape=tuple(token_feat.shape), idx=flat, max_num_atoms=int(max_num_atoms), feat_batch_dims=tuple(feat_batch_dims), feat_dims=tuple(feat_dims), n_token=n_token)
        # check against stock on the real feature (bit-exact) before trusting
        ref = _BC_STOCK(token_mask=token_mask, num_atoms_per_token=num_atoms_per_token, token_feat=token_feat, token_dim=token_dim, max_num_atoms_per_token=max_num_atoms_per_token)
        got = _bc_apply(e, token_feat)
        if ref.shape != got.shape or not torch.equal(ref, got):
            raise RuntimeError(f"graph-safe broadcast mismatch vs stock (shape {tuple(ref.shape)} vs {tuple(got.shape)})")
        _BC_CACHE.append(e)
        return got
    return _bc_apply(e, token_feat)

def _bc_apply(e, token_feat):
    fb, fd, n_token = e["feat_batch_dims"], e["feat_dims"], e["n_token"]
    flat_feat = token_feat.reshape(-1, *fd) if len(fd) else token_feat.reshape(-1)
    zero = torch.zeros((1, *fd), dtype=token_feat.dtype, device=token_feat.device) if len(fd) else torch.zeros((1,), dtype=token_feat.dtype, device=token_feat.device)
    src = torch.cat([flat_feat, zero], dim=0)
    idx = torch.where(e["idx"] < 0, torch.full_like(e["idx"], src.shape[0] - 1), e["idx"])
    out = src.index_select(0, idx)
    return out.reshape(*fb, e["max_num_atoms"], *fd)



def _bc_refresh(e, tm, na):
    """Refresh gather entry `e` IN PLACE for the live item's (token_mask tm, num_atoms_per_token na): True = refreshed and checked, str = the
    reason it cannot be (the caller then drops the generation).  The index is re-derived exactly as _broadcast_graphsafe records it (the STOCK
    function on a row-id feature) and the refreshed table must reproduce the stock broadcast on a synthetic feature of the entry's own rank —
    a pure row gather exact on distinct row ids is exact on any feature."""
    if _BC_STOCK is None:
        return "no_stock_fn"
    try:
        tm_e = tm if tm.shape == e["token_mask"].shape else torch.broadcast_to(tm.reshape((1,) * (e["token_mask"].dim() - tm.dim()) + tuple(tm.shape)), e["token_mask"].shape)
        na_e = na if na.shape == e["natpt"].shape else torch.broadcast_to(na.reshape((1,) * (e["natpt"].dim() - na.dim()) + tuple(na.shape)), e["natpt"].shape)
    except RuntimeError:
        return "shape_not_broadcastable"
    if tm_e.dtype != e["token_mask"].dtype or na_e.dtype != e["natpt"].dtype:
        return "dtype"
    fb, fd, n_token = tuple(e["feat_batch_dims"]), tuple(e["feat_dims"]), e["n_token"]
    nrows = n_token
    for d in fb:
        nrows *= int(d)
    rowid = torch.arange(nrows, device=e["idx"].device, dtype=torch.float64).reshape(*fb, n_token)
    out = _BC_STOCK(token_mask=tm_e, num_atoms_per_token=na_e, token_feat=rowid + 1.0, token_dim=-1, max_num_atoms_per_token=e["max_apt"])
    if int(out.shape[len(fb)]) != e["max_num_atoms"]:
        return "max_num_atoms_%d_vs_%d" % (int(out.shape[len(fb)]), e["max_num_atoms"])
    flat = out.reshape(-1).round().long() - 1
    if flat.shape != e["idx"].shape:
        return "idx_shape"
    e["idx"].copy_(flat); e["token_mask"].copy_(tm_e); e["natpt"].copy_(na_e)
    feat = (rowid + 1.0).reshape(*fb, n_token, *((1,) * len(fd))).expand(*fb, n_token, *fd) if fd else rowid + 1.0
    got = _bc_apply(e, feat)
    ref = _BC_STOCK(token_mask=tm_e, num_atoms_per_token=na_e, token_feat=feat, token_dim=e["token_dim"], max_num_atoms_per_token=e["max_apt"])
    if got.shape != ref.shape or not torch.equal(got, ref):
        return "FATAL_mismatch_after_refresh"
    return True


def _allocator_flush():
    """torch's allocator flush whatever a caller put over torch.cuda.empty_cache (the defining module keeps the original)."""
    f = getattr(getattr(torch.cuda, "memory", None), "empty_cache", None)
    (f if callable(f) else torch.cuda.empty_cache)()


def _shared_pool():
    """The ONE pool every capture of the process goes into: a torch.cuda.MemPool held here (it outlives the graphs captured into it, so a dropped
    generation's blocks serve the next capture); its id is what torch.cuda.graph(pool=) takes. A torch without MemPool gets one private pool per
    capture (torch.cuda.graph_pool_handle), named on the census (pool=per_capture)."""
    if _STATE["pool"] is None:
        if hasattr(torch.cuda, "MemPool"):
            _STATE["pool"] = torch.cuda.MemPool(); _STATE["stats"]["pool"] = "shared"
        else:
            _STATE["stats"]["pool"] = "per_capture"
            return torch.cuda.graph_pool_handle()
    return _STATE["pool"].id


def release():
    """Drop every captured generation, the memoised gather tables and the shared pool (their blocks become cached-free; the caller flushes the
    allocator) — the item-boundary release under memory pressure (the kit's post_release lever). The next step re-captures as a first item does,
    flushing the allocator cache before its warm-up. Returns {"gens": n}."""
    n = len(_STATE["gens"])
    while _STATE["gens"]:
        g = _STATE["gens"].pop(0)
        del g.graph, g.static_out, g.static_args, g.static_batch, g.pool
        del g
    _BC_CACHE.clear()
    _STATE["pool"] = None
    _STATE["released"] = True
    _STATE["stats"]["releases"] += 1
    if n:
        _STATE["stats"]["gen_drops"] += n
    return {"gens": n}


MAX_GEN = 1                       # captured generations (graph + static buffers) kept resident
WARMUP_STEPS = 2                  # eager evaluations of the step on the side stream before the FIRST capture of the process (lazy compilation settles there)
WARMUP_STEPS_LATER = 1            # ... before every later capture (the step is compiled; one evaluation records the gather tables)


def _log(msg):
    sys.stderr.write(f"[of3_graphs] {msg}\n"); sys.stderr.flush()


class _Gen:
    """one captured generation: graph + pool + static tensors for one shape key"""
    def __init__(self, key):
        self.key = key; self.graph = None; self.pool = None
        self.static_batch = None; self.static_args = None; self.static_out = None


def _tensor_sig(t):
    return (tuple(t.shape), t.dtype, t.device.index if t.device.type == "cuda" else -1)


def _numerics_mode():
    """D59b: the numerics MODE is part of every pool key, so a request in a different mode can never replay a graph (= kernel selection) captured
    in another mode. (The memoised gather/predicate tables are integer index copies -> mode-independent by construction.)"""
    try:
        ac_on = torch.is_autocast_enabled("cuda"); ac_dt = str(torch.get_autocast_dtype("cuda")) if ac_on else None
    except TypeError:  # older torch signature
        ac_on = torch.is_autocast_enabled(); ac_dt = str(torch.get_autocast_gpu_dtype()) if ac_on else None
    return (("det", torch.are_deterministic_algorithms_enabled()), ("f32mm", torch.get_float32_matmul_precision()),
            ("tf32mm", torch.backends.cuda.matmul.allow_tf32), ("tf32cudnn", torch.backends.cudnn.allow_tf32), ("cudnn_det", torch.backends.cudnn.deterministic),
            ("cudnn_bench", torch.backends.cudnn.benchmark), ("autocast", ac_on, ac_dt), ("sdp", torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled()),
            ("cublas_ws", os.environ.get("CUBLAS_WORKSPACE_CONFIG")))


def _make_key(batch, xl_noisy, t, si_input, si_trunk, zij_trunk, extra):
    items = [("xl", _tensor_sig(xl_noisy)), ("t", _tensor_sig(t)), ("si", _tensor_sig(si_input)), ("s", _tensor_sig(si_trunk)), ("z", _tensor_sig(zij_trunk))]
    for k in sorted(batch.keys()):
        v = batch[k]
        if isinstance(v, torch.Tensor):
            items.append((k, _tensor_sig(v)))
    return ((tuple(items), extra)) + (("mode", _numerics_mode()),)


def _copy_batch_into(static_batch, batch):
    """Refresh the static feature entries (the ones the captured step reads) with the live item's values, in place; a no-op for an entry that IS the
    static tensor. An entry of another shape/dtype cannot be here (the keep decision compared exactly these signatures) — raised by name if it is."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and k in static_batch:
            d = static_batch[k]
            if d.data_ptr() != v.data_ptr():
                if d.shape != v.shape or d.dtype != v.dtype:
                    raise RuntimeError(f"static feature {k!r} {tuple(d.shape)}/{d.dtype} cannot take the item's {tuple(v.shape)}/{v.dtype} (keep decision bypassed?)")
                d.copy_(v, non_blocking=True)


def _storage_signature(module):
    """addresses (+device) of every parameter and buffer of the captured module; cheap (a few hundred tensors)."""
    return tuple((t.data_ptr(), str(t.device)) for t in list(module.parameters()) + list(module.buffers()))


def _static_ptrs(gen):
    """{name: data_ptr} of every static buffer a captured generation reads or writes: the static feature entries, the step arguments, the output."""
    out = {("feat", k): v.data_ptr() for k, v in gen.static_batch.items() if isinstance(v, torch.Tensor)}
    out.update({("arg", k): v.data_ptr() for k, v in gen.static_args.items()})
    outs = gen.static_out if isinstance(gen.static_out, (tuple, list)) else (gen.static_out,)
    out.update({("out", i): o.data_ptr() for i, o in enumerate(outs) if isinstance(o, torch.Tensor)})
    return out


def _static_moved(gen):
    """Names of static buffers whose address differs from the one recorded at capture ([] = all in place; cannot happen by construction — the
    generation holds its tensors — asserted because a replay for a new item depends on it)."""
    now = _static_ptrs(gen)
    return [f"{a}:{b}" for (a, b) in sorted(set(now) | set(gen.static_ptrs), key=str) if now.get((a, b)) != gen.static_ptrs.get((a, b))]


def _read_values_differing(gen, batch):
    """Names of NON-tensor feature entries the step read whose value differs from the captured item's (a Python value is baked into the graph as a
    constant: control flow, a size). Scalars / strings / tuples of them compare by value; any other object counts as differing unless it is the same object."""
    out = []
    for k, v in gen.static_batch.items():
        if isinstance(v, torch.Tensor):
            continue
        w = batch.get(k, _MISSING) if hasattr(batch, "get") else _MISSING
        if isinstance(v, (bool, int, float, str, type(None), tuple)) and not isinstance(w, torch.Tensor):
            try:
                same = type(v) is type(w) and v == w
            except Exception:  # noqa: BLE001
                same = False
        else:
            same = v is w
        if not same:
            out.append(k)
    return out


class GraphedStep:
    """callable replacing self.diffusion_module(...) inside SampleDiffusion.forward"""

    def __init__(self, module):
        self.module = module
        self.rollouts = 0                                             # advanced by the patched sampler at every rollout start (the per-item checks key on it, not on id(batch) alone)

    def _drop_old(self):
        while len(_STATE["gens"]) >= MAX_GEN:
            g = _STATE["gens"].pop(0)
            _STATE["stats"]["gen_drops"] += 1
            _BC_CACHE.clear()
            _log(f"dropping WHOLE generation (graph+static buffers+gather tables; its blocks return to the shared pool) key0={g.key[0][:2]}")
            del g.graph, g.static_out, g.static_args, g.static_batch, g.pool
            del g
        torch.cuda.synchronize()

    def _capture(self, key, batch, xl_noisy, t, si_input, si_trunk, zij_trunk, kw):
        if kw.get("use_deepspeed_evo_attention"):
            # DeepSpeed's DS4Sci_EvoformerAttention launches on the LEGACY DEFAULT STREAM whatever the current stream is (profiled: every other kernel
            # on the current stream, its `attention_kernel_batched_impl` on stream 7): it races producers queued on the warm-up / capture stream (NaN)
            # and a legacy-stream launch cannot be captured. The patched sampler loop turns the flag off inside the rollout; this is the backstop.
            raise RuntimeError("OF3_CUDA_GRAPHS refuses to capture a step that requests use_deepspeed_evo_attention (DS4Sci launches on the legacy "
                               "default stream: uncapturable, races side-stream producers); cuEquivariance / Triton / torch attention are capture-safe")
        self._drop_old()
        if _STATE["released"]:                                    # a release() since the last capture: the default stream's cached-free segments cannot serve the
            _STATE["released"] = False                            # side-stream warm-up below -> return them to the driver first
            torch.cuda.synchronize(); _allocator_flush()
        gen = _Gen(key)
        dev = xl_noisy.device
        # static copies of the step's tensor arguments (trunk embeddings constant within a seed; xl_noisy / t refreshed every step)
        gen.static_args = dict(xl_noisy=xl_noisy.clone(), t=t.clone(), si_input=si_input.clone(), si_trunk=si_trunk.clone(), zij_trunk=zij_trunk.clone())
        # --- warm-up (eager) on a side stream, checking that the step consumes no CUDA RNG. The FIRST warm-up evaluation reads the item's own feature
        # dict through a recording view (which entries the step reads, by name); the static feature dict the graph is captured on then holds exactly
        # the read entries (clones; features are constant within a query) — the entries the step never reads (the MSA / template stacks) are neither
        # copied nor part of the keep decision. Every later warm-up evaluation and the capture run on the static dict.
        gstate = torch.cuda.get_rng_state(dev)
        t0 = time.perf_counter()
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        _BC_MODE["phase"] = "record"
        n_warm = WARMUP_STEPS if _STATE["stats"]["captures"] == 0 else WARMUP_STEPS_LATER
        _STATE["stats"]["warmup_steps"] += n_warm
        view = _ReadView({k: v for k, v in batch.items()})
        try:
            with torch.cuda.stream(s):
                for i in range(n_warm):
                    b = view if i == 0 else gen.static_batch
                    _ = self.module(batch=b, xl_noisy=gen.static_args["xl_noisy"], token_mask=b["token_mask"],
                                    atom_mask=b["atom_mask"], t=gen.static_args["t"], si_input=gen.static_args["si_input"],
                                    si_trunk=gen.static_args["si_trunk"], zij_trunk=gen.static_args["zij_trunk"], **kw)
                    if i == 0:
                        gen.read_names = view.read_names()                 # frozenset of the entries the step read, or None (a whole-dict access: every entry counts as read)
                        gen.static_batch = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
                                            if gen.read_names is None or k in gen.read_names}
        finally:
            _BC_MODE["phase"] = "eager"
        torch.cuda.current_stream(dev).wait_stream(s)
        torch.cuda.synchronize(dev)
        warm_ms = (time.perf_counter() - t0) * 1e3
        rng_unchanged = bool(torch.equal(gstate, torch.cuda.get_rng_state(dev)))
        if not rng_unchanged:
            # restore and refuse to graph (would desynchronise the Philox stream vs stock)
            torch.cuda.set_rng_state(gstate, dev)
            raise RuntimeError("diffusion step consumed CUDA RNG; refusing to capture")
        # --- capture
        t1 = time.perf_counter()
        gen.pool = _shared_pool()
        gen.graph = torch.cuda.CUDAGraph()
        _BC_MODE["phase"] = "replay"
        try:
            with torch.cuda.graph(gen.graph, pool=gen.pool, stream=s, capture_error_mode="thread_local"):
                gen.static_out = self.module(batch=gen.static_batch, xl_noisy=gen.static_args["xl_noisy"], token_mask=gen.static_batch["token_mask"],
                                             atom_mask=gen.static_batch["atom_mask"], t=gen.static_args["t"], si_input=gen.static_args["si_input"],
                                             si_trunk=gen.static_args["si_trunk"], zij_trunk=gen.static_args["zij_trunk"], **kw)
        finally:
            _BC_MODE["phase"] = "eager"
        torch.cuda.synchronize(dev)
        cap_ms = (time.perf_counter() - t1) * 1e3
        _STATE["gens"].append(gen); _STATE["stats"]["captures"] += 1
        gen.static_ptrs = _static_ptrs(gen)                               # every static buffer's address at capture: asserted unchanged before a later item replays this generation
        n_tensors = sum(1 for v in batch.values() if isinstance(v, torch.Tensor))
        n_static = sum(1 for v in gen.static_batch.values() if isinstance(v, torch.Tensor))
        static_mib = sum(v.numel() * v.element_size() for v in gen.static_batch.values() if isinstance(v, torch.Tensor)) / 2 ** 20
        unread_mib = sum(v.numel() * v.element_size() for k, v in batch.items() if isinstance(v, torch.Tensor) and k not in gen.static_batch) / 2 ** 20
        _log(f"INVARIANT captured key(n_atom={xl_noisy.shape[-2]}, n_tok={si_trunk.shape[-2]}, lead={tuple(xl_noisy.shape[:-2])}) warmup_steps={n_warm} warmup_ms={warm_ms:.0f} capture_ms={cap_ms:.0f} rng_offset_unchanged={rng_unchanged} mode=det:{torch.are_deterministic_algorithms_enabled()}/f32mm:{torch.get_float32_matmul_precision()} static_features={n_static}/{n_tensors} read_all={gen.read_names is None} static_mib={static_mib:.0f} unread_mib={unread_mib:.0f} keep={'on' if keep_enabled() else 'off'} individual_graph_pops=0 gen_drops={_STATE['stats'].get('gen_drops',0)} kept={_STATE['stats']['kept']} gens_live={len(_STATE['gens'])} captures={_STATE['stats']['captures']} replays={_STATE['stats']['replays']}")
        if ("readnames",) not in _SEEN_ONCE:                               # once per process: the feature entries the step reads (the keep decision's domain) and the ones it never reads
            _SEEN_ONCE.add(("readnames",))
            shp = lambda k, v: f"{k}{list(v.shape)}" if isinstance(v, torch.Tensor) else f"{k}={type(v).__name__}"   # noqa: E731
            _log("step reads features [%s]; never reads [%s]" % (",".join(shp(k, v) for k, v in sorted(gen.static_batch.items())) if gen.read_names is not None else "ALL (whole-dict access)",
                                                                 ",".join(shp(k, v) for k, v in sorted(batch.items()) if k not in gen.static_batch)))
        gen.storage_sig = _storage_signature(self.module)
        return gen

    def _recheck_item(self, gen, batch):
        """New (query, seed) item hitting an existing graph generation: the graph baked in (a) the memoised token->atom gather tables and
        (b) static copies of every batch feature.  (b) is refreshed by value in __call__; (a) must serve the new item: a table recorded from
        equal (token_mask, num_atoms_per_token) values is kept, a table of EQUAL SHAPES but different values is refreshed IN PLACE (_bc_refresh:
        re-derived with the stock function as at record time, checked to reproduce the stock broadcast), a table that cannot be refreshed
        (different shape / atom count) makes the generation stale -> False (the caller drops it and recaptures; never replay a stale gather)."""
        tm, na = batch.get("token_mask"), batch.get("num_atoms_per_token")
        _STATE["stats"]["item_rechecks"] = _STATE["stats"].get("item_rechecks", 0) + 1
        if tm is None or na is None:
            return True
        stale, refreshed = [], 0
        for e in _BC_CACHE:
            try:
                same = e["token_mask"].shape == tm.shape and torch.equal(e["token_mask"], tm) and torch.equal(e["natpt"], na)
            except Exception:  # noqa: BLE001
                same = False
            if same:
                continue
            r = _bc_refresh(e, tm, na)
            if r is True:
                refreshed += 1
            else:
                stale.append(r)
                if r.startswith("FATAL"):
                    break
        if stale:
            _STATE["stats"]["stale_gather_recaptures"] += 1
            _log("gather table of the new item cannot be refreshed in place (%s) -> dropping generation and recapturing (invariant: never replay a stale gather)" % ",".join(stale))
            return False
        if refreshed:
            _STATE["stats"]["regather_inplace"] += 1; _STATE["stats"]["regather_tables"] += refreshed
            _log(f"gather tables refreshed in place for the new item ({refreshed} table(s)); resident graph kept")
        return True

    def _item_boundary(self, key, batch):
        """A step whose full key matches no resident generation: decide against the newest one (MAX_GEN = 1: the one) — KEEP it when the keys differ
        only in feature entries the captured step never read (and every read non-tensor value is the captured one and every static buffer sits at its
        captured address), else say why it re-captures. Returns the generation to replay, or None (the caller captures; _drop_old drops the old one)."""
        if not _STATE["gens"]:
            return None
        g = _STATE["gens"][-1]
        keep = keep_enabled()
        verdict, names = keep_decision(g.key, getattr(g, "read_names", None), key, keep)
        if verdict == "keep":
            vals = _read_values_differing(g, batch); moved = _static_moved(g)
            if vals:
                verdict, names = "drop", ["value:" + n for n in vals] + names
            elif moved:
                verdict, names = "drop", ["static_moved:" + n for n in moved[:5]]
        if verdict == "keep":
            g.key = key                                                  # the new item's full signature: its later steps hit directly
            g.keep_pending = list(names)                                 # counted (`kept`) and named once the per-item checks in __call__ hold
            g.last_marker = None                                         # force those checks for this step whatever the marker says
            return g
        if verdict == "drop":
            if keep:
                _STATE["stats"]["keep_drops"] += 1
            _log(f"re-capture for the new item: key differs in [{','.join(names[:12])}{',…' if len(names) > 12 else ''}] "
                 f"({'read by the step' if keep else KEEP_ENV + '=0: any difference re-captures'})")
        return None

    def __call__(self, batch, xl_noisy, t, si_input, si_trunk, zij_trunk, **kw):
        extra = tuple(sorted((k, v) for k, v in kw.items() if isinstance(v, (bool, int, float, str, type(None)))))
        key = _make_key(batch, xl_noisy, t, si_input, si_trunk, zij_trunk, extra)
        marker = (id(batch), self.rollouts)                                # the (feature dict, rollout) this step belongs to: a new one re-runs the per-item checks below
        gen = next((g for g in _STATE["gens"] if g.key == key), None)
        try:
            if gen is None:
                gen = self._item_boundary(key, batch)                      # keep across items (a generation whose read entries match) or None -> capture below
            if gen is not None and marker != getattr(gen, "last_marker", None):
                # first step of a new predict item on an existing generation: (1) the captured graph baked in raw parameter/buffer addresses ->
                # check the module's storage has not moved since capture (Lightning's Strategy.teardown() does .cpu() after every predict; a
                # resident server then re-uploads -> possibly new addresses; joint rule D59); (2) check gather tables by value.
                sig = _storage_signature(self.module)
                if sig != gen.storage_sig:
                    _STATE["stats"]["storage_moved_recaptures"] = _STATE["stats"].get("storage_moved_recaptures", 0) + 1
                    _log("module parameter/buffer storage moved since capture (e.g. .cpu()/.cuda() round trip) -> dropping generation and recapturing (invariant: never replay on moved storage)")
                    _STATE["gens"] = [g for g in _STATE["gens"] if g is not gen]; _BC_CACHE.clear(); _STATE["stats"]["gen_drops"] += 1
                    del gen; torch.cuda.synchronize(); gen = None
            if gen is not None and marker != getattr(gen, "last_marker", None):
                if not self._recheck_item(gen, batch):
                    _STATE["gens"] = [g for g in _STATE["gens"] if g is not gen]; _BC_CACHE.clear(); _STATE["stats"]["gen_drops"] += 1
                    del gen; torch.cuda.synchronize(); gen = None
            if gen is None:
                gen = self._capture(key, batch, xl_noisy, t, si_input, si_trunk, zij_trunk, kw)
            if getattr(gen, "keep_pending", None) is not None:                # a generation KEPT for this item (its checks above held): count and name it once
                _STATE["stats"]["kept"] += 1
                _log(f"KEPT generation for the new item: capture key equal on every tensor the step reads (n_atom={gen.static_args['xl_noisy'].shape[-2]}, "
                     f"n_tok={gen.static_args['si_trunk'].shape[-2]}); unread entries that differ: {','.join(gen.keep_pending) or 'none'} kept={_STATE['stats']['kept']}")
                gen.keep_pending = None
            gen.last_marker = marker
            # refresh static inputs in place (no-ops when the caller already passed the static tensors)
            _copy_batch_into(gen.static_batch, batch)
            sa = gen.static_args
            if sa["si_input"].data_ptr() != si_input.data_ptr(): sa["si_input"].copy_(si_input, non_blocking=True)
            if sa["si_trunk"].data_ptr() != si_trunk.data_ptr(): sa["si_trunk"].copy_(si_trunk, non_blocking=True)
            if sa["zij_trunk"].data_ptr() != zij_trunk.data_ptr(): sa["zij_trunk"].copy_(zij_trunk, non_blocking=True)
            sa["xl_noisy"].copy_(xl_noisy, non_blocking=True); sa["t"].copy_(t, non_blocking=True)
            gen.graph.replay(); _STATE["stats"]["replays"] += 1
            return gen.static_out
        except Exception as e:  # never silently change results: fall back to eager for this call and say so
            if is_oom(e): raise                                       # an out-of-memory is the caller's to see, never rerouted
            _STATE["stats"]["fallbacks"] += 1
            _log(f"FALLBACK to eager ({type(e).__name__}: {str(e)[:200]})" + ("  <<< EVERY step of this process will run eager; unset OF3_CUDA_GRAPHS or the kernel flags >>>" if "refuses to capture" in str(e) else ""))
            if os.environ.get("OF3_GRAPHS_STRICT") == "1":
                raise
            return self.module(batch=batch, xl_noisy=xl_noisy, token_mask=batch["token_mask"], atom_mask=batch["atom_mask"], t=t,
                               si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, **kw)



def _install_broadcast():
    import openfold3.core.utils.atomize_utils as AU
    global _BC_STOCK
    if _BC_STOCK is not None:
        return
    _BC_STOCK = AU.broadcast_token_feat_to_atoms
    AU.broadcast_token_feat_to_atoms = _broadcast_graphsafe
    n = 0
    for modname, mod in list(sys.modules.items()):
        if modname.startswith("openfold3") and mod is not None and getattr(mod, "broadcast_token_feat_to_atoms", None) is _BC_STOCK:
            setattr(mod, "broadcast_token_feat_to_atoms", _broadcast_graphsafe); n += 1
    _log(f"broadcast_token_feat_to_atoms rebound in {n} modules (memoised, checked-vs-stock gather)")


def enable():
    if _STATE["enabled"]:
        return
    from openfold3.core.model.structure import diffusion_module as DM
    SD = DM.SampleDiffusion
    stock_forward = SD.forward
    global _STOCK_FORWARD
    _STOCK_FORWARD = stock_forward

    def forward(self, batch, si_input, si_trunk, zij_trunk, noise_schedule, no_rollout_samples, use_conditioning=True, chunk_size=None,
                use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False, use_lma=False,
                use_high_precision_attention=False, _mask_trans=True):
        # identical to stock SampleDiffusion.forward (0.4.1) except that self.diffusion_module(...) is routed through GraphedStep
        if self.training or torch.is_grad_enabled():
            return stock_forward(self, batch=batch, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, noise_schedule=noise_schedule,
                                 no_rollout_samples=no_rollout_samples, use_conditioning=use_conditioning, chunk_size=chunk_size,
                                 use_deepspeed_evo_attention=use_deepspeed_evo_attention, use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                                 use_triton_triangle_kernels=use_triton_triangle_kernels, use_lma=use_lma,
                                 use_high_precision_attention=use_high_precision_attention, _mask_trans=_mask_trans)
        if not hasattr(self, "_graphed_step"):
            self._graphed_step = GraphedStep(self.diffusion_module)
        step = self._graphed_step
        step.rollouts += 1
        if use_deepspeed_evo_attention:                            # DS4Sci launches on the legacy default stream (uncapturable, races the warm-up stream): the
            use_deepspeed_evo_attention = False                    # rollout runs the engine's other attention paths (cuEquivariance where it serves, else torch)
            _STATE["stats"]["ds4sci_off_rollouts"] += 1
            if _STATE["stats"]["ds4sci_off_rollouts"] == 1:
                _log("use_deepspeed_evo_attention=True requested for the rollout: OFF inside the graphed sampler (DS4Sci launches on the legacy default stream - "
                     "uncapturable); the diffusion module's attention runs the engine's cuEquivariance / torch path [ds4sci_off_rollouts]")
        kw = dict(use_conditioning=use_conditioning, chunk_size=chunk_size, use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                  use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels, use_lma=use_lma,
                  use_high_precision_attention=use_high_precision_attention, _mask_trans=_mask_trans)
        atom_mask = batch["atom_mask"]
        batch_dim, num_atoms = atom_mask.shape[0], atom_mask.shape[-1]
        xl = noise_schedule[0] * torch.randn((batch_dim, no_rollout_samples, num_atoms, 3), device=atom_mask.device, dtype=atom_mask.dtype)
        schedule_host = noise_schedule.detach().to("cpu") if noise_schedule.is_cuda else noise_schedule   # ONE copy per rollout: the loop's `c_tau > gamma_min` runs on the host
        _STATE["stats"]["host_schedule_rollouts"] += int(noise_schedule.is_cuda)                          # (no per-step device sync); c_tau enters `dt = c_tau - t` as the 0-dim
        for tau, c_tau in enumerate(schedule_host[1:]):                                                  # host operand of the same device subtraction; t, the noise scale and
            xl = DM.centre_random_augmentation(xl=xl, atom_mask=atom_mask)                               # the division keep the DEVICE elements noise_schedule[tau] (bitwise)
            gamma = self.gamma_0 if c_tau > self.gamma_min else 0
            t = noise_schedule[tau] * (gamma + 1)
            noise = self.noise_scale * torch.sqrt(t**2 - noise_schedule[tau] ** 2) * torch.randn_like(xl)
            xl_noisy = xl + noise
            xl_denoised = step(batch, xl_noisy, t.to(xl_noisy.device), si_input, si_trunk, zij_trunk, **kw)
            delta = (xl_noisy - xl_denoised) / t
            dt = c_tau - t
            xl = xl_noisy + self.step_scale * dt * delta
        _STATE["stats"]["eager_steps"] += 0
        return xl

    SD.forward = forward
    _install_broadcast()
    _STATE["enabled"] = True
    _log("SampleDiffusion.forward patched (CUDA-graph diffusion step)")


def stats():
    return dict(_STATE["stats"], gens_live=len(_STATE["gens"]))
