"""OF3T lever D1 — per-rollout PAIR CACHE for OpenFold3's diffusion sampler (exact class).

Within one diffusion rollout (one predicted item = one query x one seed; 200 steps x n samples) the pair inputs of every step are identical:
  * DiffusionConditioning: zij = transitions(linear_z(layer_norm_z(cat[zij_trunk, relpos(batch)])))  depends on zij_trunk and constant batch
    features only (token_index, residue_index, asym_id, entity_id, sym_id, chain-level ids) — NOT on t, the noisy coordinates or the RNG.
  * DiffusionTransformer block b (24 blocks): pair bias_b = permute(linear_z_b(layer_norm_z_b(zij)))  depends on zij only.
Stock recomputes both at every one of the 200 steps.  This lever computes them at the first step of a rollout and reuses the tensors for the remaining
steps: the SAME kernels on the SAME inputs, evaluated once instead of every step -> bitwise identical results under the deterministic reference configuration
(stock itself is only reproducible there), same distribution under default numerics.
NOT cached (recomputed every step as in stock): si (depends on t through the Fourier noise embedding), the atom-attention encoder/decoder incl. its own
projection of zij (D2 candidate), everything depending on x_noisy.
Seed-invariance is NOT assumed: OpenFold3 re-seeds per predicted item and subsamples the MSA with torch RNG in the input embedder, so zij_trunk differs
between seeds of the same query; the cache is keyed on the rollout (the zij_trunk tensor object + shape/device/dtype + step counter) and is dropped when a
new rollout starts.  Memory: zij [S,N,N,128] fp32 + 24 x [S,16,N,N] fp32; printed at first fill; a 4 GB cap bounds it
(falls back to stock computation for the blocks that do not fit, never partially wrong).
Layout of the cached per-block biases: HEAD-MAJOR with 16-byte aligned rows — each bias_b [*, H, N, N] is held as the [..., :N] view of a
[*, H, N, ceil8(N)] buffer (values exactly the stock permute(linear_z_b(...)) view's; only the storage order differs), so the diffusion transformer's
attention levers (dit_attn, dit_glue) stream the bias in place instead of re-laying the stock heads-last view on every call; stock consumers add it by
value whatever its strides.
env:  OF3T_PAIRCACHE=1            enable
Composes with OF3_CUDA_GRAPHS (fast-inference kit C3): the kit captures diffusion_module.forward once per shape; with this lever active the FIRST call of a
rollout (eager, fills the cache) must happen before capture and the cached tensors become static inputs of the graph -> see of3t_levers integration:
when OF3_CUDA_GRAPHS=1 the cache tensors are allocated once per shape key as static buffers and refreshed in place at each rollout start (copy_), so a
captured graph keeps reading valid memory; graph replays per rollout are unchanged (asserted in the test record job via the kit's replay counter).
"""
import os, sys, time, weakref
import torch

ENABLED = os.environ.get("OF3T_PAIRCACHE") == "1"
MAX_GB = 4.0                                                  # memory cap (GB) of the cache: conditioning zij + per-block token pair biases; blocks over it are computed as stock
STATS = {"rollouts": 0, "outside_calls": 0, "zij_fills": 0, "zij_hits": 0, "bias_fills": 0, "bias_hits": 0, "bytes": 0, "cap_skips": 0, "static_refresh": 0}
_SEEN = set()


def _log(msg, once_key=None):
    if once_key is not None:
        if once_key in _SEEN:
            return
        _SEEN.add(once_key)
    sys.stderr.write(f"[of3t_paircache] {msg}\n"); sys.stderr.flush()


def _ceil8(n):
    return (int(n) + 7) // 8 * 8


def _head_major(pb):
    """pb [*, H, N, N] (any strides: stock hands the heads-last permute view of the [*, N, N, H] projection) -> the same values held head-major with
    16-byte aligned rows: the [..., :N] view of a fresh [*, H, N, ceil8(N)] buffer."""
    n = int(pb.shape[-1])
    buf = torch.empty(tuple(pb.shape[:-1]) + (_ceil8(n),), dtype=pb.dtype, device=pb.device)
    out = buf[..., :n]
    out.copy_(pb)
    return out


def _head_major_bytes(pb):
    """bytes of the head-major buffer `_head_major(pb)` allocates for a bias of pb's shape (rows padded to ceil8)."""
    return pb.numel() // int(pb.shape[-1]) * _ceil8(pb.shape[-1]) * pb.element_size()


class _Rollout:
    """cache for ONE rollout.  v1.2.1: a rollout is delimited by the CALL boundary (SampleDiffusion.forward / _sample_rollout, and OpenFold3._rollout as an
    idempotent outer boundary), never by tensor identity/address: under torch.inference_mode a freed zij_trunk whose Python id AND CUDA block are recycled by
    the next seed is indistinguishable by (id, data_ptr, shape) (ABA) -> v1.2 could serve the previous seed's buffers (observed 1/40 under DET in a composed run).
    Tensor identity is kept only as an EXTRA invalidation trigger inside an epoch (a different zij_trunk object mid-epoch => drop + refill), never as a reason to reuse."""
    __slots__ = ("key", "ref", "zij", "bias", "step", "nbytes", "static", "epoch")
    def __init__(self, key, ref=None, epoch=-1):
        self.key = key; self.ref = ref; self.zij = None; self.bias = {}; self.step = 0; self.nbytes = 0; self.static = False; self.epoch = epoch
    def matches(self, key, zij_trunk, epoch):
        return self.epoch == epoch and self.key == key and self.ref is not None and self.ref() is zij_trunk


_CUR = {"r": None}
_STATIC = {}                                             # graphs cooperation: shape key -> _Rollout with static zij + bias dict (by block id); empty in eager mode
_EPOCH = {"n": 0, "depth": 0, "static_done": -1}      # rollout-call epoch counter; depth>0 <=> inside a rollout call; static_done = epoch whose static buffers are fresh


def _key_of(zij_trunk, batch):
    # secondary key (inside an epoch only): address + shape + token count (+ version counter when available, i.e. not under inference_mode)
    try:
        ver = zij_trunk._version if not zij_trunk.is_inference() else -1
    except Exception:
        ver = -1
    return (zij_trunk.data_ptr(), tuple(zij_trunk.shape), zij_trunk.dtype, str(zij_trunk.device), ver, int(batch["token_mask"].shape[-1]))


def _enter_rollout():
    """called at every rollout CALL boundary (outermost wins; nested boundaries are idempotent)"""
    if _EPOCH["depth"] == 0:
        _EPOCH["n"] += 1; STATS["rollouts"] += 1
        r = _CUR["r"]
        if r is not None and not r.static:
            _CUR["r"] = None                                   # drop the eager cache of the previous rollout (frees memory before the new one is built)
    _EPOCH["depth"] += 1

def _exit_rollout():
    _EPOCH["depth"] -= 1
    if _EPOCH["depth"] == 0:
        r = _CUR["r"]
        if r is not None and not r.static:
            _CUR["r"] = None

def release():
    """Drop the current rollout's cache and the graphs cooperation's static pair buffers — the item-boundary release under memory pressure (the
    kit's post_release lever calls it between the sampler and the confidence head at large token counts). The next rollout recomputes and
    re-allocates exactly as a first rollout does (static_done reset -> the static refresh runs again before the next capture/replay). Returns the
    bytes dropped."""
    nb = sum(int(getattr(r, "nbytes", 0) or 0) for r in _STATIC.values())
    _STATIC.clear()
    r = _CUR["r"]
    if r is not None and not getattr(r, "static", False):
        nb += int(getattr(r, "nbytes", 0) or 0)
    _CUR["r"] = None
    _EPOCH["static_done"] = -1
    STATS["releases"] = STATS.get("releases", 0) + 1
    return nb


def _inside():
    return _EPOCH["depth"] > 0


def install():
    if not ENABLED:
        return
    from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
    from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias
    from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
    from openfold3.core.utils.tensor_utils import permute_final_dims

    # ---- 0. rollout CALL boundaries (v1.2.1): SampleDiffusion.forward (0.4.x loop) / SampleDiffusion._sample_rollout (0.5.x loop) and OpenFold3._rollout (outer, idempotent).
    # Installed LAST among hooks that replace these methods? No — we wrap whatever is CURRENTLY bound and re-wrap lazily at first model forward (see _late_wrap) so a kit hook
    # that rebinds SampleDiffusion.forward after us is still enclosed by our boundary via OpenFold3._rollout.
    from openfold3.core.model.structure.diffusion_module import SampleDiffusion
    def _wrap_boundary(cls, name):
        fn = getattr(cls, name, None)
        if fn is None or getattr(fn, "_of3t_boundary", False):
            return False
        def bounded(self, *a, **kw):
            _enter_rollout()
            try:
                return fn(self, *a, **kw)
            finally:
                _exit_rollout()
        bounded._of3t_boundary = True; bounded.__wrapped__ = fn; bounded.__name__ = name
        setattr(cls, name, bounded); return True
    nb = [n for n in ("forward", "_sample_rollout") if _wrap_boundary(SampleDiffusion, n)]
    try:
        from openfold3.projects.of3_all_atom.model import OpenFold3 as _OF3
        _orig_model_forward = _OF3.forward
        def _late_wrap(self, batch):
            # re-wrap at first model forward: by now every import-time hook (kit graphs/persist) has rebound SampleDiffusion methods; wrap the final bindings once.
            if not getattr(_OF3, "_of3t_boundaries_done", False):
                got = [n for n in ("forward", "_sample_rollout") if _wrap_boundary(SampleDiffusion, n)]
                _OF3._of3t_boundaries_done = True
                _log(f"rollout call boundaries: SampleDiffusion.{{forward,_sample_rollout}} wrapped (late: {got}); epoch counter drives all cache refreshes")
            return _orig_model_forward(self, batch)
        _OF3.forward = _late_wrap
    except Exception as e:
        _log(f"could not install late boundary wrap: {e}")
    _log(f"rollout call boundaries installed at import: {nb}")

    # ---- 1. DiffusionConditioning: cache the pair path (zij), recompute si every step (depends on t)
    _orig_embed = DiffusionConditioning._embed_trunk_inputs
    _orig_forward = DiffusionConditioning.forward

    def forward(self, batch, t, si_input, si_trunk, zij_trunk, use_conditioning, chunk_size=None):
        if not use_conditioning or torch.is_grad_enabled():
            return _orig_forward(self, batch=batch, t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, use_conditioning=use_conditioning, chunk_size=chunk_size)
        if not _inside():                                   # not within a seen rollout call boundary -> stock (never risk a stale cache)
            STATS["outside_calls"] = STATS.get("outside_calls", 0) + 1
            return _orig_forward(self, batch=batch, t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, use_conditioning=use_conditioning, chunk_size=chunk_size)
        key = _key_of(zij_trunk, batch)
        r = _CUR["r"]
        if r is None or not r.matches(key, zij_trunk, _EPOCH["n"]):
            # first step of this rollout epoch (or, defensively, a different zij_trunk object inside the epoch): new cache
            _CUR["r"] = r = _Rollout(key, weakref.ref(zij_trunk), _EPOCH["n"])
        r.step += 1
        if r.zij is None:
            si, zij = _orig_forward(self, batch=batch, t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, use_conditioning=use_conditioning, chunk_size=chunk_size)
            nb = zij.numel() * zij.element_size()
            if (r.nbytes + nb) / 2**30 <= MAX_GB:
                r.zij = zij; r.nbytes += nb; STATS["zij_fills"] += 1; STATS["bytes"] = max(STATS["bytes"], r.nbytes)
                _log(f"rollout cache: zij {tuple(zij.shape)} {zij.dtype} = {nb/2**20:.0f} MiB (N_token={key[-1]})", once_key=("zij", key[1], key[-1]))
            else:
                STATS["cap_skips"] += 1; _log(f"zij not cached: cap {MAX_GB} GB reached", once_key="capz")
            return si, zij
        # cached: compute si only (stock arithmetic for si: identical ops), reuse zij
        STATS["zij_hits"] += 1
        si = torch.cat([si_trunk, si_input], dim=-1)
        si = self.linear_s(self.layer_norm_s(si))
        n = 0.25 * torch.log(t / self.sigma_data)
        n = self.fourier_emb(n.unsqueeze(-1))
        si = si + self.linear_n(self.layer_norm_n(n)).unsqueeze(-2)
        token_mask = batch["token_mask"]
        if chunk_size is not None:
            # stock _chunk_forward applies transition_s / transition_z chunked; si path only here (same chunking as stock for si)
            si = _si_transitions(self, si, token_mask, chunk_size)
        else:
            for l in self.transition_s:
                si = si + l(si, mask=token_mask, chunk_size=None)
        return si, r.zij
    DiffusionConditioning.forward = forward

    def _si_transitions(self, si, token_mask, chunk_size):
        # mirrors DiffusionConditioning._chunk_forward for the si half
        for l in self.transition_s:
            si = si + l(si, mask=token_mask, chunk_size=chunk_size)
        return si

    # ---- 2. DiffusionTransformer blocks: cache per-block pair bias.  Only for AttentionPairBias instances that live inside a DiffusionTransformer
    # whose z is the cached zij (checked by identity: z is r.zij) -> atom transformers (cross attention, different z) and trunk pairformers untouched.
    _orig_prep = AttentionPairBias._prep_bias

    def _prep_bias(self, a, z, mask):
        r = _CUR["r"] if _inside() else None
        if r is None or r.zij is None or z is not r.zij or torch.is_grad_enabled():
            return _orig_prep(self, a=a, z=z, mask=mask)
        bid = id(self)
        ent = r.bias.get(bid)
        if ent is None:
            biases = _orig_prep(self, a=a, z=z, mask=mask)          # [mask_bias, pair_bias]
            pb = biases[1]
            nb = _head_major_bytes(pb)
            if (r.nbytes + nb) / 2**30 <= MAX_GB:
                ent = _head_major(pb)
                r.bias[bid] = ent; r.nbytes += nb; STATS["bias_fills"] += 1; STATS["bytes"] = max(STATS["bytes"], r.nbytes)
                if len(r.bias) == 1:
                    _log(f"rollout cache: per-block pair bias {tuple(pb.shape)} {pb.dtype} head-major = {nb/2**20:.0f} MiB x n_blocks", once_key=("pb", tuple(pb.shape)))
                return [biases[0], ent]
            STATS["cap_skips"] += 1; _log(f"pair bias not cached for some blocks: cap {MAX_GB} GB reached (total would be {(r.nbytes+nb)/2**30:.2f} GB)", once_key="capb")
            return biases
        STATS["bias_hits"] += 1
        # mask bias exactly as stock (cheap, depends on mask only)
        if mask is None:
            mask = a.new_ones(a.shape[:-1])
        batch_dims = a.shape[:-2]
        mask = mask.expand((*batch_dims, -1))
        mask_bias = (self.inf * (mask - 1))[..., None, None, :]
        return [mask_bias, ent]
    AttentionPairBias._prep_bias = _prep_bias

    # ---- 2b. later upstream layouts: the token DiffusionTransformer applies ONE shared layer_norm_z(z) per step (DiffusionTransformer.forward) and its blocks use
    # DiffusionAttentionPairBias (linear_z only).  Cache the shared LN output per rollout (z is r.zij -> LN(z) is step-invariant) and the per-block linear_z biases
    # (keyed on the cached LN output object).  On 0.4.x these classes/attributes do not exist and nothing is patched here.
    try:
        from openfold3.core.model.layers.attention_pair_bias import DiffusionAttentionPairBias as _DAPB
    except Exception:
        _DAPB = None
    if _DAPB is not None and hasattr(DiffusionTransformer, "forward") and hasattr(DiffusionTransformer, "__init__"):
        _orig_dt_forward = DiffusionTransformer.forward
        def dt_forward(self, a, s, z, *args, **kw):
            r = _CUR["r"] if _inside() else None
            ln = getattr(self, "layer_norm_z", None)
            if r is None or r.zij is None or z is not r.zij or ln is None or torch.is_grad_enabled():
                return _orig_dt_forward(self, a, s, z, *args, **kw)
            ent = r.bias.get(("lnz", id(self)))
            if ent is None:
                zn = ln(z); nb = zn.numel() * zn.element_size()
                if (r.nbytes + nb) / 2**30 <= MAX_GB:
                    r.bias[("lnz", id(self))] = zn; r.nbytes += nb; STATS["bias_fills"] += 1; STATS["bytes"] = max(STATS["bytes"], r.nbytes)
                    _log(f"rollout cache (0.5.x): shared DiT layer_norm_z(z) {tuple(zn.shape)} = {nb/2**20:.0f} MiB", once_key=("lnz", tuple(zn.shape)))
                else:
                    STATS["cap_skips"] += 1
                    return _orig_dt_forward(self, a, s, z, *args, **kw)
                ent = zn
            else:
                STATS["bias_hits"] += 1
            # call the stock forward with the LN temporarily replaced by identity on THIS instance (so the body runs unchanged: z = self.layer_norm_z(z) -> cached)
            self.__dict__["layer_norm_z"] = (lambda _zn: (lambda _z: _zn))(ent)
            try:
                return _orig_dt_forward(self, a, s, z, *args, **kw)
            finally:
                del self.__dict__["layer_norm_z"]          # restores the registered nn.Module attribute
        DiffusionTransformer.forward = dt_forward

        _orig_dprep = _DAPB._prep_bias
        def _dprep_bias(self, a, z, mask):
            r = _CUR["r"] if _inside() else None
            # z here is the (cached) shared-LN output; accept it if it is one of our cached lnz tensors
            if r is None or torch.is_grad_enabled() or not any((k[0] == "lnz" and v is z) for k, v in r.bias.items() if isinstance(k, tuple)):
                return _orig_dprep(self, a=a, z=z, mask=mask)
            bid = ("dapb", id(self)); ent = r.bias.get(bid)
            if ent is None:
                biases = _orig_dprep(self, a=a, z=z, mask=mask); pb = biases[1]
                nb = _head_major_bytes(pb)
                if (r.nbytes + nb) / 2**30 <= MAX_GB:
                    ent = _head_major(pb)
                    r.bias[bid] = ent; r.nbytes += nb; STATS["bias_fills"] += 1; STATS["bytes"] = max(STATS["bytes"], r.nbytes)
                    return [biases[0], ent]
                STATS["cap_skips"] += 1
                return biases
            STATS["bias_hits"] += 1
            if mask is None:
                mask = a.new_ones(a.shape[:-1])
            mask = mask.expand((*a.shape[:-2], -1))
            mask_bias = (self.inf * (mask - 1))[..., None, None, :]
            return [mask_bias, ent]
        _DAPB._prep_bias = _dprep_bias
        _log("openfold3 0.5.x diffusion-transformer classes detected: shared layer_norm_z + DiffusionAttentionPairBias caching enabled")

    if _DAPB is None:
        _orig_dprep = None
    # ---- 3. CUDA-graph cooperation (fast-inference kit OF3_CUDA_GRAPHS=1): static buffers per shape key, refreshed eagerly at every rollout start
    # BEFORE the kit's GraphedStep runs (capture or replay).  Inside capture the patched modules above see `z is r.zij` (the static buffer) and read the
    # static bias buffers; on replay the graph reads the same addresses, which we have just refreshed with this rollout's values (copy_).
    if os.environ.get("OF3_CUDA_GRAPHS") == "1":
        try:
            import of3_graphs as G
        except Exception as e:
            _log(f"OF3_CUDA_GRAPHS=1 but of3_graphs not importable ({e}); pair cache runs in eager mode only")
            G = None
        if G is not None and hasattr(G, "GraphedStep"):
            _orig_call = G.GraphedStep.__call__       # the static sets live in the module-level _STATIC (release() empties it)

            def _refresh_static(step_obj, batch, t, si_input, si_trunk, zij_trunk, kw):
                """eager stock computation of the step-invariant pair tensors for THIS rollout into static buffers"""
                dm = step_obj.module                       # DiffusionModule
                dc = dm.diffusion_conditioning; dt_ = dm.diffusion_transformer
                skey = (tuple(zij_trunk.shape), zij_trunk.dtype, str(zij_trunk.device), int(batch["token_mask"].shape[-1]))
                r = _STATIC.get(skey)
                with torch.no_grad():
                    _CUR["r"] = None                       # force stock path inside this eager computation
                    si_ref, zij_new = _orig_forward(dc, batch=batch, t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
                                                    use_conditioning=kw.get("use_conditioning", True), chunk_size=kw.get("chunk_size"))
                    if r is None:
                        # drop static sets of other shapes (LRU of size 1, like the kit's graph pool) before allocating
                        _STATIC.clear()
                        r = _Rollout(("static",) + skey); r.static = True; r.zij = zij_new.clone(); r.nbytes = r.zij.numel() * r.zij.element_size()
                        _STATIC[skey] = r
                        _log(f"static pair buffers for graphs: zij {tuple(r.zij.shape)} allocated", once_key=("szij", skey))
                    else:
                        r.zij.copy_(zij_new)
                    STATS["static_refresh"] += 1
                    # per-block biases of the token diffusion transformer (blocks whose attention consumes zij)
                    token_mask = batch["token_mask"]
                    ln_shared = getattr(dt_, "layer_norm_z", None) if _DAPB is not None else None      # 0.5.x: shared LN before the blocks
                    zsrc = r.zij
                    if ln_shared is not None:
                        zn = ln_shared(r.zij); k = ("lnz", id(dt_)); ent = r.bias.get(k)
                        if ent is None:
                            r.bias[k] = zn.clone(); r.nbytes += zn.numel() * zn.element_size()
                        else:
                            ent.copy_(zn)
                        zsrc = r.bias[k]
                    for blk in dt_.blocks:
                        apb = blk.attention_pair_bias
                        if _DAPB is not None and isinstance(apb, _DAPB):
                            pb = _orig_dprep(apb, a=si_ref, z=zsrc, mask=token_mask)[1]; key_ = ("dapb", id(apb))
                        else:
                            pb = _orig_prep(apb, a=si_ref, z=r.zij, mask=token_mask)[1]; key_ = id(apb)     # a only supplies shape/dtype for the mask; z -> bias
                        ent = r.bias.get(key_)
                        if ent is None:
                            ent = _head_major(pb); r.bias[key_] = ent; r.nbytes += _head_major_bytes(pb)      # static AND head-major: the graph reads these rows in place
                        else:
                            ent.copy_(pb)
                    STATS["bytes"] = max(STATS["bytes"], r.nbytes)
                _CUR["r"] = r; r.step = 0
                return r

            def __call__(self, batch, xl_noisy, t, si_input, si_trunk, zij_trunk, **kw):
                if not _inside():
                    # graphed step called outside any seen rollout boundary: cannot know whether buffers are fresh -> refresh EVERY call (exact, slow) and say so once
                    STATS["outside_calls"] = STATS.get("outside_calls", 0) + 1
                    _log("WARNING: graphed diffusion step outside a rollout call boundary; refreshing static pair buffers every step (correct but slow)", once_key="outside")
                    _refresh_static(self, batch, t, si_input, si_trunk, zij_trunk, kw)
                elif _EPOCH["static_done"] != _EPOCH["n"] or _CUR["r"] is None or not _CUR["r"].static or (_CUR["r"].ref is not None and _CUR["r"].ref() is not zij_trunk):
                    # first graphed step of this rollout EPOCH (call boundary) -> unconditional refresh; identity change mid-epoch also refreshes (extra trigger only)
                    r = _refresh_static(self, batch, t, si_input, si_trunk, zij_trunk, kw); r.ref = weakref.ref(zij_trunk); r.epoch = _EPOCH["n"]; _EPOCH["static_done"] = _EPOCH["n"]
                    _log(f"graphs mode: static pair buffers refreshed at rollout epoch {_EPOCH['n']} ({r.nbytes/2**30:.2f} GB)", once_key="gmode")
                _CUR["r"].step += 1
                return _orig_call(self, batch, xl_noisy, t, si_input, si_trunk, zij_trunk, **kw)
            G.GraphedStep.__call__ = __call__

            # in graphs mode the conditioning forward must treat the static r.zij as THE cached value for any zij_trunk (keys differ: static copy)
            def forward_static(self, batch, t, si_input, si_trunk, zij_trunk, use_conditioning, chunk_size=None):
                r = _CUR["r"]
                if r is None or not r.static or not use_conditioning or torch.is_grad_enabled():
                    return _orig_forward(self, batch=batch, t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, use_conditioning=use_conditioning, chunk_size=chunk_size)
                STATS["zij_hits"] += 1
                si = torch.cat([si_trunk, si_input], dim=-1)
                si = self.linear_s(self.layer_norm_s(si))
                n = 0.25 * torch.log(t / self.sigma_data)
                n = self.fourier_emb(n.unsqueeze(-1))
                si = si + self.linear_n(self.layer_norm_n(n)).unsqueeze(-2)
                token_mask = batch["token_mask"]
                for l in self.transition_s:
                    si = si + l(si, mask=token_mask, chunk_size=chunk_size)
                return si, r.zij
            DiffusionConditioning.forward = forward_static
            _log("CUDA-graph cooperation enabled (static pair buffers refreshed per rollout before capture/replay)" + ("; 0.5.x shared-LN path active" if _DAPB is not None else ""))

    import atexit
    def _report():
        _log(f"stats {STATS}  (peak cache {STATS['bytes']/2**30:.2f} GB)")
    atexit.register(_report)
    _log(f"installed (scope=full, cap={MAX_GB} GB)")
