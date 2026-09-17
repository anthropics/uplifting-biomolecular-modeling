"""boltz_dit_hoist.py — hoist the STEP-INVARIANT work of the Boltz-2 diffusion denoiser out of the 200-step sampling loop
(boltz 2.2.1, inference only; runtime monkeypatch in the style of boltz_graph_patch.py — no file of the boltz package is edited; default OFF).

DiffusionModule.forward is called 200x per AtomDiffusion.sample() with the SAME s_trunk, s_inputs, feats, diffusion_conditioning; only r_noisy and
times (sigma) change. Already hoisted by STOCK (once per sample(), in Boltz2.forward -> DiffusionConditioning): pairwise conditioning z, the 24
token-layer pair-bias projections (token_trans_bias), atom encoder q/c/p/to_keys, the 6 atom-layer pair-bias projections. Still recomputed per step:

  level 1 (BOLTZ_DIT_HOIST=1):
   (a) SingleConditioning invariant part s0 = single_embed(norm_single(cat(s_trunk, s_inputs)))  [Fourier(sigma) is added AFTER -> s0 invariant;
       transitions after the add are NOT]
   (b) the per-step .float() casts of q, c, atom_enc_bias, atom_dec_bias, token_trans_bias (bf16 -> fp32 copies when the trunk ran under autocast)
   (c) c.repeat_interleave(m) and, for each of the 6 atom-transformer layers (3 encoder + 3 decoder; BOTH conditioned on the step-invariant atom cond c —
       the decoder receives the encoder's expanded c as c_skip): AdaLN s_norm(c) -> sigmoid(s_scale(.)), s_bias(.) [attention AdaLN AND transition AdaLN],
       attention output gate sigmoid(Linear(c)), transition output gate sigmoid(Linear(c)).
       (Token-transformer AdaLN/gates are conditioned on s0 + Fourier(sigma) -> NOT invariant.)
  level 2 (BOLTZ_DIT_HOIST=2) adds the attention glue:
   (d) per attention layer (24 token + 6 atom): the expanded pair bias `proj_z(z).repeat_interleave(multiplicity, 0)` (a permute+contiguous COPY of the
       invariant bias every step: 24 x [B*m, H, N, N] fp32 at token level), the AtomTransformer-level `bias.repeat_interleave(m,0).view(...)`,
   (e) mask glue: token_pad_mask.repeat_interleave(m).float(), atom mask .float(), per-atom-layer to_keys(mask), and the additive mask term
       (1 - mask) * -inf of every attention layer.
Everything else (q/k/v/g projections, attention, transitions on a, Fourier path, encoder r-embedding, decoder a->q, aggregation bmm) depends on x_t or
sigma and stays per step.

EXACT by construction: each cached tensor is produced by calling the STOCK sub-module / the stock expression on the STOCK input in the STOCK
(multiplicity-expanded, window-reshaped) shape — the very call every stock step makes (same kernel, same problem size -> same cuBLAS/cuDNN algorithm ->
same bits) — and the per-step forwards read the cache instead of recomputing. With kernels off Boltz-2 is bitwise run-to-run, so 'exact' = bitwise CIF /
PAE npz equality vs stock, which is the exactness claim.

Memory: ONE cache object per DiffusionModule (conditioning-dict ids are aliases; tensors re-filled in place, re-allocated only on shape change; apply(0)
  frees it; a sample() of an input at or above boltz_graph_patch.RELEASE_MIN_TOKENS — BOLTZ_GRAPH_RELEASE_MIN_TOKENS, 1024 — releases it together
  with the step graph when it returns, release_after_sample, and the next sample() rebuilds it: the cache never outlives the sampler of a large
  input into the confidence module or the next prediction's trunk). It holds the fp32 conditioning casts + (level 2) the 24 token-layer and 6
  atom-layer expanded pair biases and mask terms:
  bytes ~= 2 x (24 layers x 16 heads x 4 B) x N_tokens^2  +  ~50 KB x N_atoms   (level 2; level 1 ~= half of the first term + the second)
  -> STATS['last']['cache_mb'] / cache_report() give the measured MB per token count. For scale: stock itself allocates the fp32 cast of
  token_trans_bias (24 x 16 x 4 B x N^2) plus one expanded per-layer bias transiently at EVERY step, so the hoist's steady-state footprint is ~2x the
  stock sampler's own per-step transient.

Flags
  BOLTZ_DIT_HOIST=0|1|2          level (0 = nothing patched, default)
  BOLTZ_DIT_HOIST_DEBUG=1        verbose (stderr)
Programmatic: `import boltz_dit_hoist as H; H.apply(2)`; `H.apply(0)` restores every stock forward.

Composition with the kit's CUDA-graph step patch (boltz_graph_patch.py): apply the graph patch FIRST, then this module — the hoist wrapper must be the
OUTERMOST AtomDiffusion.sample so the cache is filled eagerly BEFORE the inner (graphed) sampler captures/replays. Cache tensors are allocated once per
shape key and refreshed IN PLACE (.copy_) on later sample() calls, so a captured graph keeps reading the same addresses and sees the refreshed values
(the graph patch does the same with its own static conditioning buffers). If an entry ever had to be re-allocated under an unchanged key the module
invalidates the graph (forces a recapture) rather than let a replay read a stale buffer. `poison()` fills the cache with NaN: a replay after poison()
must output NaN (test that the graph really reads the cache); the next sample() refreshes the cache.
"""
from __future__ import annotations

import os
import sys
import time

import torch

STATS = {"level": 0, "samples": 0, "cache_builds": 0, "cache_refreshes": 0, "reallocs_under_same_key": 0,
         "hits": {}, "build_s": [], "last": {}, "captured_with_cache": 0, "capture_stock_fallbacks": 0, "graph_invalidations": 0, "cache_releases": 0, "headroom_gated": 0,
         "token_gated": 0, "max_tokens": None}     # the memory row's token ceiling (BOLTZ_SAMPLER_MAX_TOKENS): sample() calls of inputs above it open no cache generation — the inner
                                                    # sampler serves them (the roll-out steps aside by the same word; the stock eager loop with the fused step runs), counted token_gated
MAX_TOKENS_ENV = "BOLTZ_SAMPLER_MAX_TOKENS"


def _max_tokens() -> int:
    v = (os.environ.get(MAX_TOKENS_ENV) or "").strip()
    return int(v) if v.isdigit() else 0                # `card` (unresolved) or absent: no ceiling
_DEBUG = os.environ.get("BOLTZ_DIT_HOIST_DEBUG", "0") == "1"
_CUR = {"cache": None, "gen": None, "net_calls": 0, "atom_s": None, "atom_maskf": None, "atom_maskk": None, "atom_maskterm": None, "tok_mask": None, "tok_maskterm": None}
_GEN = [0]
import weakref
_ARMED = weakref.WeakSet()      # AtomDiffusion modules that have sampled with the hoist armed (their captured graphs may read cache buffers)


def _log(*a):
    if _DEBUG:
        print("[boltz_dit_hoist]", *a, file=sys.stderr, flush=True)


def _hit(k):
    STATS["hits"][k] = STATS["hits"].get(k, 0) + 1


class _Cache:
    def __init__(self):
        self.key = None; self.t = {}; self.dcf = None; self.m = None; self.active = False; self.net_calls = 0; self.fresh = True; self.realloc = False
        self.gen = None; self.src = None; self.s_shape = None; self.owned = {}


def _put(C, k, val):
    """first time: allocate (clone); later: copy_ in place (static address for CUDA-graph consumers)."""
    old = C.t.get(k)
    if old is None or old.shape != val.shape or old.dtype != val.dtype or old.device != val.device:
        if old is not None or not C.fresh:
            C.realloc = True
        t = val.clone(); C.t[k] = t
        C.owned[k] = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
    else:
        old.copy_(val)
    return C.t[k]


def _active():
    C = _CUR["cache"]
    return C if (C is not None and C.active) else None


def _capturing(t):
    return bool(t.is_cuda and torch.cuda.is_current_stream_capturing())


# =============================================================================== patched forwards (cache when active, else the stock forward) ======
def _adaln_forward(self, a, s):
    C = _active()
    if C is not None and s is _CUR["atom_s"]:
        ent = C.t.get(("adaln", id(self)))
        if ent is not None:
            scale, bias = ent
            a = self.a_norm(a)
            _hit("adaln")
            return scale * a + bias
    return self._stock_forward(a, s)


def _ctb_forward(self, a, s):
    C = _active()
    if C is not None and s is _CUR["atom_s"]:
        og = C.t.get(("ctb_gate", id(self)))
        if og is not None:
            a = self.adaln(a, s)
            b = self.swish_gate(a) * self.a_to_b(a)
            _hit("ctb_gate")
            return og * self.b_to_a(b)
    return self._stock_forward(a, s)


def _layer_forward(self, a, s, bias=None, mask=None, to_keys=None, multiplicity=1):
    """DiffusionTransformerLayer.forward. Atom layers (s is the cached atom cond): attention output gate from the cache and (level 2) the to_keys'd mask
    from the cache. Token layers: stock arithmetic (their s depends on sigma); level-2 bias/mask reads happen inside AttentionPairBias."""
    C = _active()
    if C is None:
        return self._stock_forward(a, s, bias, mask, to_keys, multiplicity)
    is_atom = (s is _CUR["atom_s"]) and (("layer_gate", id(self)) in C.t)
    b = self.adaln(a, s)
    k_in = b
    if to_keys is not None:
        k_in = to_keys(b)
        if STATS["level"] >= 2 and mask is _CUR["atom_maskf"] and _CUR["atom_maskk"] is not None:
            mask = _CUR["atom_maskk"]; _hit("atom_maskk")
        else:
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
    if self.pair_bias_attn:
        b = self.pair_bias_attn(s=b, z=bias, mask=mask, multiplicity=multiplicity, k_in=k_in)
    else:
        b = self.no_pair_bias_attn(s=b, mask=mask, k_in=k_in)
    if is_atom:
        b = C.t[("layer_gate", id(self))] * b; _hit("layer_gate")
    else:
        b = self.output_projection(s) * b
    a = a + b
    a = a + self.transition(a, s)
    a = self.post_lnorm(a)
    return a


def _apb_forward(self, s, z, mask, k_in, multiplicity=1):
    """AttentionPairBias.forward (compute_pair_bias=False in every diffusion layer): level 2 reads the expanded bias and the additive mask term from the cache."""
    C = _active()
    if C is None or STATS["level"] < 2:
        return self._stock_forward(s, z, mask, k_in, multiplicity)
    B = s.shape[0]
    q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
    k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
    v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
    zb = C.t.get(("apb_z", id(self)))
    if zb is not None and z is zb:
        bias = C.t[("apb_bias", id(self))]; _hit("apb_bias")
    else:
        bias = self.proj_z(z)
        bias = bias.repeat_interleave(multiplicity, 0)
    g = self.proj_g(s).sigmoid()
    if mask is _CUR["atom_maskk"] and _CUR["atom_maskterm"] is not None:
        mterm = _CUR["atom_maskterm"]; _hit("atom_maskterm")
    elif mask is _CUR["tok_mask"] and _CUR["tok_maskterm"] is not None:
        mterm = _CUR["tok_maskterm"]; _hit("tok_maskterm")
    else:
        mterm = None
    with torch.autocast("cuda", enabled=False):
        attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        attn = attn / (self.head_dim**0.5) + bias.float()
        attn = attn + ((1 - mask[:, None, None].float()) * -self.inf if mterm is None else mterm)
        attn = attn.softmax(dim=-1)
        o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    o = o.reshape(B, -1, self.c_s)
    o = self.proj_o(g * o)
    return o


def _dt_forward(self, a, s, bias=None, mask=None, to_keys=None, multiplicity=1):
    """DiffusionTransformer.forward: when `bias` is a cached tensor, hand each layer the persistent per-layer view object (so AttentionPairBias recognises it)."""
    C = _active()
    views = C.t.get(("dt_views", id(self))) if (C is not None and STATS["level"] >= 2) else None
    if views is None or bias is not C.t.get(("dt_bias_src", id(self))):
        return self._stock_forward(a, s, bias, mask, to_keys, multiplicity)
    _hit("dt_views")
    for i, layer in enumerate(self.layers):
        a = layer(a, s, views[i], mask, to_keys, multiplicity)
    return a


def _single_cond_forward(self, times, s_trunk, s_inputs):
    C = _active()
    if C is None or s_trunk is not None:
        return self._stock_forward(times, s_trunk, s_inputs)
    from einops import rearrange
    s = C.t[("s0",)]; _hit("s0")
    if not self.disable_times:
        fourier_embed = self.fourier_embed(times)
        normed_fourier = self.norm_fourier(fourier_embed)
        fourier_to_single = self.fourier_to_single(normed_fourier)
        s = rearrange(fourier_to_single, "b d -> b 1 d") + s
    for transition in self.transitions:
        s = transition(s) + s
    return s, normed_fourier if not self.disable_times else None


def _atom_transformer_forward(self, q, c, bias, to_keys, mask, multiplicity=1):
    """AtomTransformer.forward: stock, but (i) registers the windowed view of the cached c as THE atom-cond object (identity used by AdaLN/gates below),
    (ii) level 2: expanded+windowed bias and mask.float()/to_keys(mask)/mask-term from the cache."""
    C = _active()
    if C is None or c is not C.t.get(("c_m",)):
        return self._stock_forward(q, c, bias, to_keys, mask, multiplicity)
    W = self.attn_window_queries; H = self.attn_window_keys
    B, N, D = q.shape; NW = N // W
    q = q.view((B * NW, W, -1))
    c = C.t[("c_w", id(self))]                       # == c.view((B * NW, W, -1)) of the cached c_m (persistent view object)
    ent = C.t.get(("atf", id(self))) if STATS["level"] >= 2 else None
    if ent is not None and bias is ent["bias_src"] and mask.shape == ent["mask_shape"]:
        bias = ent["bias_w"]; maskf = ent["maskf"]; _hit("atf_bias_mask")
        prev = (_CUR["atom_maskf"], _CUR["atom_maskk"], _CUR["atom_maskterm"])
        _CUR["atom_maskf"], _CUR["atom_maskk"], _CUR["atom_maskterm"] = maskf, ent["maskk"], ent["maskterm"]
    else:
        mask = mask.view(B * NW, W)
        bias = bias.repeat_interleave(multiplicity, 0)
        bias = bias.view((bias.shape[0] * NW, W, H, -1))
        maskf = mask.float(); prev = None
    to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)  # noqa: E731
    prev_s = _CUR["atom_s"]; _CUR["atom_s"] = c
    try:
        q = self.diffusion_transformer(a=q, s=c, bias=bias, mask=maskf, multiplicity=1, to_keys=to_keys_new)
    finally:
        _CUR["atom_s"] = prev_s
        if prev is not None:
            _CUR["atom_maskf"], _CUR["atom_maskk"], _CUR["atom_maskterm"] = prev
    q = q.view((B, NW * W, D))
    return q


def _encoder_forward(self, feats, q, c, atom_enc_bias, to_keys, r=None, multiplicity=1):
    """AtomAttentionEncoder.forward: stock, with c.repeat_interleave(m) read from the cache (the cached object then flows on to the decoder as c_skip)."""
    C = _active()
    if C is None or c is not C.dcf.get("c") or multiplicity != C.m:
        return self._stock_forward(feats, q, c, atom_enc_bias, to_keys, r, multiplicity)
    B, N, _ = feats["ref_pos"].shape
    atom_mask = feats["atom_pad_mask"].bool()
    if self.structure_prediction:
        q = q.repeat_interleave(multiplicity, 0)
        r_to_q = self.r_to_q_trans(r)
        q = q + r_to_q
    c = C.t[("c_m",)]; _hit("enc_c")
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
    q = self.atom_encoder(q=q, mask=atom_mask, c=c, bias=atom_enc_bias, multiplicity=multiplicity, to_keys=to_keys)
    with torch.autocast("cuda", enabled=False):
        q_to_a = self.atom_to_token_trans(q).float()
        atom_to_token = feats["atom_to_token"].float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_to_token_mean = atom_to_token / (atom_to_token.sum(dim=1, keepdim=True) + 1e-6)
        a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)
    a = a.to(q)
    return a, q, c, to_keys


def _dm_forward(self, s_inputs, s_trunk, r_noisy, times, feats, diffusion_conditioning, multiplicity=1):
    """DiffusionModule.forward: stock, with the conditioning casts / s0 / token mask / atom-layer terms read from the cache. The cache is built LAZILY
    here, from exactly the objects this call receives (s_trunk, s_inputs, feats, diffusion_conditioning, multiplicity), once per sample() call and
    per distinct conditioning dict — so it is correct both for the stock sampler (which passes the dict given to sample()) and for the kit's graph
    patch (whose captured body passes its OWN static clones st["dc"], st["s_trunk"], ...; those clones are refreshed in place by the graph patch
    before every prediction and our cache entries for that dict are refreshed in place here, at the eager step-0 / first call of every sample())."""
    G = _CUR["gen"]
    if G is None or (_capturing(r_noisy) and _cache_for(self, diffusion_conditioning, create=False) is None):
        return self._stock_forward(s_inputs, s_trunk, r_noisy, times, feats, diffusion_conditioning, multiplicity)
    C = _cache_for(self, diffusion_conditioning, create=True)
    src_now = (id(s_trunk), id(s_inputs))
    if C.gen == G and C.m == multiplicity and C.src != src_now and getattr(C, "alias_src", None) == src_now:
        pass                                    # same cache, reached through the caller's dict whose values equal the graph's static clones (refreshed this sample())
    elif C.gen != G or C.m != multiplicity or C.src != src_now:
        if _capturing(r_noisy):
            # never build during capture (allocations would land in the graph pool and the build kernels would be captured): stock path, and SAY SO
            STATS["capture_stock_fallbacks"] += 1
            _log("WARNING: denoiser call during CUDA-graph capture without a valid cache -> stock forward captured (no hoist in this graph)")
            return self._stock_forward(s_inputs, s_trunk, r_noisy, times, feats, diffusion_conditioning, multiplicity)
        _build_cache(self, C, dict(s_trunk=s_trunk, s_inputs=s_inputs, feats=feats, diffusion_conditioning=diffusion_conditioning, multiplicity=multiplicity))
        C.gen = G; C.src = (id(s_trunk), id(s_inputs))
    if _capturing(r_noisy):
        STATS["captured_with_cache"] += 1
    C.net_calls += 1; _CUR["net_calls"] += 1
    prev = _CUR["cache"]; _CUR["cache"] = C; C.active = True
    try:
        dcf = C.dcf
        s, normed_fourier = self.single_conditioner(times, None, None)          # patched: s0 from cache + Fourier(times) + transitions (stock code)
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(feats=feats, q=dcf["q"], c=dcf["c"], atom_enc_bias=dcf["atom_enc_bias"],
                                                                 to_keys=diffusion_conditioning["to_keys"], r=r_noisy, multiplicity=multiplicity)
        a = a + self.s_to_a_linear(s)
        if STATS["level"] >= 2:
            maskf = C.t[("tok_maskf",)]; _hit("tok_mask")
            _CUR["tok_mask"], _CUR["tok_maskterm"] = maskf, C.t[("tok_maskterm",)]
        else:
            maskf = feats["token_pad_mask"].repeat_interleave(multiplicity, 0).float()
        try:
            a = self.token_transformer(a, mask=maskf, s=s, bias=dcf["token_trans_bias"], multiplicity=multiplicity)
        finally:
            _CUR["tok_mask"] = None; _CUR["tok_maskterm"] = None
        a = self.a_norm(a)
        r_update = self.atom_attention_decoder(a=a, q=q_skip, c=c_skip, atom_dec_bias=dcf["atom_dec_bias"], feats=feats, multiplicity=multiplicity, to_keys=to_keys)
    finally:
        C.active = False; _CUR["cache"] = prev
    return r_update


def _cache_for(diff_module, dc, create):
    """one _Cache per (DiffusionModule, conditioning dict object). Lifetime: see sample_hoisted (at most the current sample()'s dict + the
    CUDA-graph's static dict; both usually alias ONE cache)."""
    store = diff_module.__dict__.setdefault("_dit_hoist_caches", {})
    C = store.get(id(dc))
    if C is None and create:
        # ONE cache object per DiffusionModule: a new conditioning dict (the caller's dict of a new sample(), or the graph patch's static clones)
        # becomes an alias of the existing object; _dm_forward refreshes the tensors IN PLACE whenever the source objects of the call differ from the
        # ones the cache was last filled from (C.src), so the cache always holds the values of the current call's inputs, at fixed addresses.
        existing = next(iter(store.values()), None)
        C = existing if existing is not None else _Cache()
        store[id(dc)] = C
    return C


def cache_report(diff_or_net):
    """MB held per cached-tensor class: storages ALLOCATED by the hoist (views and pass-through references to the model's own tensors excluded)."""
    net = getattr(diff_or_net, "score_model", diff_or_net)
    caches = {id(C): C for C in getattr(net, "_dit_hoist_caches", {}).values()}
    rep, seen = {}, set()
    for C in caches.values():
        for k, (ptr, nb) in C.owned.items():
            if (ptr, nb) in seen: continue
            seen.add((ptr, nb)); rep[k[0]] = rep.get(k[0], 0) + nb
    out = {k: round(v / 2**20, 2) for k, v in sorted(rep.items(), key=lambda kv: -kv[1])}
    out["_n_cache_objects"] = len(caches); out["_total_mb"] = round(sum(rep.values()) / 2**20, 1)
    return out


def cache_bytes(diff_or_net):
    return int(cache_report(diff_or_net)["_total_mb"] * 2**20)


# =============================================================================== cache build: stock modules, stock inputs, stock shapes ===========
def _atom_layers(net):
    out = []
    for pref, atf in (("enc", net.atom_attention_encoder.atom_encoder), ("dec", net.atom_attention_decoder.atom_decoder)):
        for i, layer in enumerate(atf.diffusion_transformer.layers):
            out.append((pref, i, atf, layer))
    return out


def _stock(mod):
    """call the stock forward of a (possibly patched) module class."""
    f = getattr(type(mod), "_stock_forward", None)
    return (lambda *a, **k: f(mod, *a, **k)) if f is not None else mod


def _build_cache(net, C, kw):
    """(Re)compute every hoisted tensor with the STOCK sub-modules on the given inputs in the stock shapes; allocate once per key, then refresh in place."""
    t0 = time.time()
    m = int(kw.get("multiplicity", 1))
    feats = kw["feats"]; dc = kw["diffusion_conditioning"]; s_trunk = kw["s_trunk"]; s_inputs = kw["s_inputs"]
    key = (STATS["level"], tuple(s_trunk.shape), str(s_trunk.dtype), tuple(feats["atom_pad_mask"].shape), tuple(feats["token_pad_mask"].shape), m,
           tuple((k, tuple(v.shape), str(v.dtype)) for k, v in sorted(dc.items()) if torch.is_tensor(v)))
    C.fresh = (C.key != key); C.realloc = False
    if C.fresh:
        C.t = {}; C.owned = {}; C.key = key; STATS["cache_builds"] += 1
    else:
        STATS["cache_refreshes"] += 1
    C.m = m; C.s_shape = tuple(s_trunk.shape); C.net_calls = 0
    lvl = STATS["level"]
    with torch.no_grad():
        # (b) fp32 casts (stock: `.float()` inside DiffusionModule.forward every step; a no-op returning the same object when already fp32 -> keep object)
        dcf = {}
        for k in ("q", "c", "atom_enc_bias", "atom_dec_bias", "token_trans_bias"):
            v = dc[k]; fv = v.float()
            dcf[k] = _put(C, ("dcf", k), fv) if fv is not v else v
        C.dcf = dcf
        # (a) s0 in the expanded shape stock uses
        sc = net.single_conditioner
        s_cat = torch.cat((s_trunk.repeat_interleave(m, 0), s_inputs.repeat_interleave(m, 0)), dim=-1)
        _put(C, ("s0",), sc.single_embed(sc.norm_single(s_cat)))
        # (c) atom cond expanded + windowed; AdaLN / gates of the 6 atom layers
        enc_atf = net.atom_attention_encoder.atom_encoder; dec_atf = net.atom_attention_decoder.atom_decoder
        W = enc_atf.attn_window_queries; H = enc_atf.attn_window_keys
        c_m = _put(C, ("c_m",), dcf["c"].repeat_interleave(m, 0))
        Bm, N, D = c_m.shape; NW = N // W
        if C.fresh or ("c_w", id(enc_atf)) not in C.t:
            C.t[("c_w", id(enc_atf))] = c_m.view((Bm * NW, W, -1)); C.t[("c_w", id(dec_atf))] = c_m.view((Bm * NW, W, -1))
        c_w = C.t[("c_w", id(enc_atf))]
        for pref, i, atf, layer in _atom_layers(net):
            for ad in (layer.adaln, layer.transition.adaln):
                sn = ad.s_norm(c_w)
                sc_ = _put(C, ("adaln_scale", id(ad)), torch.sigmoid(ad.s_scale(sn)))
                bi_ = _put(C, ("adaln_bias", id(ad)), ad.s_bias(sn))
                C.t[("adaln", id(ad))] = (sc_, bi_)
            _put(C, ("layer_gate", id(layer)), layer.output_projection(c_w))
            _put(C, ("ctb_gate", id(layer.transition)), layer.transition.output_projection(c_w))
        if lvl >= 2:
            # (e) token mask + mask term (stock: mask = token_pad_mask.repeat_interleave(m,0); .float(); per layer (1 - mask[:,None,None].float()) * -inf)
            tok = net.token_transformer
            inf_t = tok.layers[0].pair_bias_attn.inf
            maskf = _put(C, ("tok_maskf",), feats["token_pad_mask"].repeat_interleave(m, 0).float())
            _put(C, ("tok_maskterm",), (1 - maskf[:, None, None].float()) * -inf_t)
            # (d) token layers: per-layer view objects of the cached fp32 bias + expanded bias copies (stock: proj_z(bias_l).repeat_interleave(m, 0))
            ttb = dcf["token_trans_bias"]
            Bt, Nt, Mt, Dt = ttb.shape; L = len(tok.layers)
            if C.fresh or ("dt_views", id(tok)) not in C.t or C.t.get(("dt_bias_src", id(tok))) is not ttb:
                b5 = ttb.view(Bt, Nt, Mt, L, Dt // L)
                C.t[("dt_views", id(tok))] = [b5[:, :, :, i] for i in range(L)]; C.t[("dt_bias_src", id(tok))] = ttb
            views = C.t[("dt_views", id(tok))]
            for i, layer in enumerate(tok.layers):
                apb = layer.pair_bias_attn
                C.t[("apb_z", id(apb))] = views[i]
                _put(C, ("apb_bias", id(apb)), apb.proj_z(views[i]).repeat_interleave(m, 0))
            # atom transformers (encoder uses atom_enc_bias, mask = atom_pad_mask.bool().repeat(m); decoder atom_dec_bias, mask = atom_pad_mask.repeat(m))
            for atf, bkey, mask_src in ((enc_atf, "atom_enc_bias", feats["atom_pad_mask"].bool().repeat_interleave(m, 0)),
                                         (dec_atf, "atom_dec_bias", feats["atom_pad_mask"].repeat_interleave(m, 0))):
                bsrc = dcf[bkey]
                bias_w = _put(C, ("atf_bias_w", id(atf)), bsrc.repeat_interleave(m, 0).view((bsrc.shape[0] * m * NW, W, H, -1)))
                mk = mask_src.view(Bm * NW, W)
                maskf_a = _put(C, ("atf_maskf", id(atf)), mk.float())
                to_keys = dc["to_keys"]
                to_keys_new = lambda x: to_keys(x.view(Bm, NW * W, -1)).view(Bm * NW, H, -1)  # noqa: E731
                maskk = _put(C, ("atf_maskk", id(atf)), to_keys_new(maskf_a.unsqueeze(-1)).squeeze(-1))
                inf_a = atf.diffusion_transformer.layers[0].pair_bias_attn.inf
                maskterm = _put(C, ("atf_maskterm", id(atf)), (1 - maskk[:, None, None].float()) * -inf_a)
                dtr = atf.diffusion_transformer; La = len(dtr.layers)
                if C.fresh or ("dt_views", id(dtr)) not in C.t or C.t.get(("dt_bias_src", id(dtr))) is not bias_w:
                    Bb, Nb, Mb, Db = bias_w.shape; b5 = bias_w.view(Bb, Nb, Mb, La, Db // La)
                    C.t[("dt_views", id(dtr))] = [b5[:, :, :, i] for i in range(La)]; C.t[("dt_bias_src", id(dtr))] = bias_w
                aviews = C.t[("dt_views", id(dtr))]
                for i, layer in enumerate(dtr.layers):
                    apb = layer.pair_bias_attn
                    C.t[("apb_z", id(apb))] = aviews[i]
                    _put(C, ("apb_bias", id(apb)), apb.proj_z(aviews[i]).repeat_interleave(1, 0))     # AtomTransformer passes multiplicity=1 to its layers
                C.t[("atf", id(atf))] = {"bias_src": bsrc, "bias_w": bias_w, "maskf": maskf_a, "maskk": maskk, "maskterm": maskterm, "mask_shape": tuple(mask_src.shape)}
    if torch.cuda.is_available() and s_trunk.is_cuda:
        torch.cuda.synchronize()
    if C.realloc and not C.fresh:
        STATS["reallocs_under_same_key"] += 1
        _CUR["invalidate_graph"] = True                           # sample wrapper drops the captured graph before the inner sampler can replay it
        _log("WARNING: cache entry re-allocated under an unchanged key -> graph will be invalidated")
    STATS["build_s"].append(round(time.time() - t0, 4))
    _log(("built" if C.fresh else "refreshed"), "cache in", STATS["build_s"][-1], "s; entries", len(C.t), "m", m, "level", lvl)
    return C


def poison(diff):
    """Test hook: overwrite every cached tensor (all caches of diff.score_model) with NaN. A step (eager or graph replay) that truly reads the cache
    must then output NaN. The next sample() call rebuilds/refreshes the caches (new generation)."""
    n = 0
    with torch.inference_mode():            # cache tensors may be inference tensors (built under the predict loop's inference_mode)
        for C in getattr(diff.score_model, "_dit_hoist_caches", {}).values():
            for k, v in list(C.t.items()):
                if torch.is_tensor(v) and v.is_floating_point() and k[0] not in ("c_w", "dt_bias_src", "apb_z"):
                    v.fill_(float("nan")); n += 1
            C.gen = None            # force a refresh at the next sample() even within the same generation
    return n


# =============================================================================== sample wrapper ===================================================
def release_after_sample(self, n_tokens) -> bool:
    """At the return of a sample() of an input of `n_tokens`: when the graph patch's size rule says so (boltz_graph_patch.release_due:
    BOLTZ_GRAPH_RELEASE_MIN_TOKENS), drop this module's hoist cache and its captured step graph (boltz_graph_patch.release_step_graph), so the
    confidence module and the next prediction run with the stock sampler's resident memory; the next sample() rebuilds the cache at its eager
    step 0 and the graph patch re-captures with it. False (nothing released) below the threshold or without the graph patch importable."""
    try:
        import boltz_graph_patch as BGP                     # staged beside this file (make_worker_variant); importing it applies nothing by itself
    except ImportError:
        return False
    if not BGP.release_due(n_tokens):
        return False
    self.score_model.__dict__.pop("_dit_hoist_caches", None)
    BGP.release_step_graph(self)                              # idempotent: the graphed sampler released its graph at its own return already
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    STATS["cache_releases"] += 1
    return True


def _out_features(mod):
    """out_features of the last Linear inside `mod` (a Linear, or a Sequential / block holding one); None when there is none (a pure rearrangement)."""
    last = None
    for sub in mod.modules():
        if isinstance(sub, torch.nn.Linear):
            last = sub
    return int(last.out_features) if last is not None else None


def projected_cache_bytes(net, nck, multiplicity) -> int:
    """Bytes _build_cache allocates for one sample() of these inputs at STATS["level"] — the same terms in the same order, by shape and dtype alone
    (nothing is computed or allocated): the a-priori side of the memory-headroom gate (sample_hoisted); cache_report is the measured side and both
    ride the census. Linear outputs take the autocast dtype when autocast is on (stock samples with it off: fp32)."""
    m = int(multiplicity); lvl = STATS["level"]; dc = nck["diffusion_conditioning"]; feats = nck["feats"]; s_trunk = nck["s_trunk"]
    f4 = 4
    try:
        lin = torch.get_autocast_dtype("cuda").itemsize if torch.is_autocast_enabled() else f4
    except Exception:
        lin = f4
    total = 0
    for k in ("q", "c", "atom_enc_bias", "atom_dec_bias", "token_trans_bias"):          # (b) the fp32 casts, held only for a tensor not already fp32
        if torch.is_tensor(dc.get(k)) and dc[k].dtype != torch.float32:
            total += int(dc[k].numel()) * f4
    b, n_tok = int(s_trunk.shape[0]), int(s_trunk.shape[1])
    sc = net.single_conditioner                                                          # (a) s0 [B*m, N, single_embed out]
    total += b * m * n_tok * (_out_features(sc.single_embed) or 2 * int(s_trunk.shape[-1])) * lin
    c = dc["c"]; rows = int(c.numel()) // int(c.shape[-1]) * m                          # (c) c_m: c fp32 repeated m; per atom layer AdaLN scale + bias, the two gates
    total += int(c.numel()) * m * f4
    for _pref, _i, _atf, layer in _atom_layers(net):
        for ad in (layer.adaln, layer.transition.adaln):
            total += rows * ((_out_features(ad.s_scale) or 0) + (_out_features(ad.s_bias) or 0)) * lin
        total += rows * ((_out_features(layer.output_projection) or 0) + (_out_features(layer.transition.output_projection) or 0)) * lin
    if lvl >= 2:
        tpm = feats["token_pad_mask"]; total += 2 * int(tpm.numel()) * m * f4          # (e) token mask + mask term
        tok = net.token_transformer; L = len(tok.layers); ttb = dc["token_trans_bias"]   # (d) per token layer the expanded bias [B*m, H, N, N]
        heads = _out_features(tok.layers[0].pair_bias_attn.proj_z)
        if heads is None:                                                                 # proj_z a rearrangement of the layer's Dt/L channels of the fp32 bias
            total += int(ttb.numel()) * m * f4
        else:
            total += b * m * int(ttb.shape[1]) * int(ttb.shape[2]) * heads * L * lin
        enc_atf = net.atom_attention_encoder.atom_encoder
        W = int(enc_atf.attn_window_queries); H = int(enc_atf.attn_window_keys)
        apm = feats["atom_pad_mask"]; n_mask = int(apm.numel()) * m; nw = n_mask // max(W, 1)
        for atf, bkey in ((enc_atf, "atom_enc_bias"), (net.atom_attention_decoder.atom_decoder, "atom_dec_bias")):
            bsrc = dc[bkey]; bias_w = int(bsrc.numel()) * m * f4
            total += bias_w + n_mask * f4 + 2 * nw * H * f4                               # bias_w, maskf, maskk + maskterm
            La = len(atf.diffusion_transformer.layers); ha = _out_features(atf.diffusion_transformer.layers[0].pair_bias_attn.proj_z)
            total += bias_w if ha is None else (int(bsrc.numel()) * m // max(int(bsrc.shape[-1]), 1)) * ha * La * lin   # per atom layer the expanded bias (its layers' slices of bias_w, multiplicity 1)
    return int(total)


def headroom_gate(self, nck, multiplicity):
    """The sampler levers' memory-headroom gate for this sample() (boltz_graph_patch.headroom over this module's projection): None off CUDA;
    else the decision dict — ``gated``, ``projected_gib`` (cache + static clones), ``transient_gib``, ``margin_gib``, ``need_gib``, ``free_gib`` —
    with ``projected_cache_gib`` apart. Evaluated before any lever state of this sample() exists."""
    s_trunk = nck["s_trunk"]
    if not (torch.cuda.is_available() and torch.is_tensor(s_trunk) and s_trunk.is_cuda):
        return None
    try:
        import boltz_graph_patch as BGP                                                  # staged beside this file; the headroom arithmetic lives there once
    except ImportError:
        return None
    n_atoms = int(nck["feats"]["atom_pad_mask"].shape[-1]) if torch.is_tensor(nck["feats"].get("atom_pad_mask")) else 0
    cache_b = projected_cache_bytes(self.score_model, nck, multiplicity)
    static_b = BGP.projected_static_bytes(nck, multiplicity, n_atoms) if BGP.STATS.get("mode") == "graph" else 0
    heads = getattr(self.score_model.token_transformer.layers[0].pair_bias_attn, "num_heads", 16)
    gate = BGP.headroom(cache_b + static_b, BGP.step_transient_bytes(nck, multiplicity, heads))
    gate["projected_cache_gib"] = round(cache_b / BGP.GIB, 2); gate["projected_static_gib"] = round(static_b / BGP.GIB, 2)
    return gate


def _sample_stock_by_headroom(self, gate, inner, atom_mask, num_sampling_steps, multiplicity, max_parallel_samples, steering_args, network_condition_kwargs):
    """This sample() on the stock path by the memory-headroom gate: whatever an earlier prediction left resident (a cache, a step graph) is freed
    first, the stock sampler runs (eager: the denoiser calls take _stock_forward since no generation is open), the call is counted."""
    try:
        import boltz_graph_patch as BGP
    except ImportError:
        BGP = None
    STATS["headroom_gated"] += 1
    self.score_model.__dict__.pop("_dit_hoist_caches", None)
    if BGP is not None:
        BGP.release_step_graph(self)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log("memory headroom: levers' projected working set %.1f GiB (cache %.1f) does not fit the free %.1f GiB with transients and margin -> stock sampler for this prediction"
         % (gate.get("projected_gib") or 0, gate.get("projected_cache_gib") or 0, gate.get("free_gib") or 0))
    stock = getattr(self, "_stock_sample", None)                                          # the graph patch keeps AtomDiffusion.sample's original under this name
    rolled = getattr(type(self), "_bzs_inner_sample", None)                               # the sampler roll-out (opt/forward/sampler) keeps it under this one, on the class
    fn = stock if stock is not None else ((lambda *a, **k: rolled(self, *a, **k)) if rolled is not None else (lambda *a, **k: inner(self, *a, **k)))   # the hoist alone: its inner sampler IS the stock one
    try:
        return fn(atom_mask, num_sampling_steps=num_sampling_steps, multiplicity=multiplicity, max_parallel_samples=max_parallel_samples,
                  steering_args=steering_args, **network_condition_kwargs)
    finally:
        n_tokens = int(network_condition_kwargs["s_trunk"].shape[1])
        if BGP is not None:
            BGP.note_headroom_gated(gate, n_tokens, multiplicity, steps=num_sampling_steps)
        STATS["last"] = {"hoist_level": STATS["level"], "net_calls": 0, "hits_delta": {}, "cache_builds": 0, "cache_refreshes": 0, "build_s_last": None,
                         "reallocs_under_same_key": STATS["reallocs_under_same_key"], "n_caches": 0, "captured_with_cache": STATS["captured_with_cache"],
                         "capture_stock_fallbacks": STATS["capture_stock_fallbacks"], "cache_mb": 0.0, "cache_report": {}, "graph_invalidations": STATS["graph_invalidations"],
                         "cache_released": False, "headroom_gated": True, "projected_cache_gib": gate.get("projected_cache_gib"), "free_gib": gate.get("free_gib")}


def sample_hoisted(self, atom_mask, num_sampling_steps=None, multiplicity=1, max_parallel_samples=None, steering_args=None, **network_condition_kwargs):
    """Outermost AtomDiffusion.sample: opens a new cache generation (every cache is rebuilt/refreshed at its first denoiser call inside this sample()),
    runs the inner sampler (stock, or the kit's graphed sampler), closes the generation. Inference without FK steering / physical guidance only."""
    inner = type(self)._dit_hoist_inner_sample
    ok = (STATS["level"] >= 1 and not self.training and steering_args is not None and not steering_args["fk_steering"]
          and not steering_args["physical_guidance_update"])
    if not ok:
        return inner(self, atom_mask, num_sampling_steps=num_sampling_steps, multiplicity=multiplicity, max_parallel_samples=max_parallel_samples,
                     steering_args=steering_args, **network_condition_kwargs)
    n_tokens = int(network_condition_kwargs["s_trunk"].shape[1]) if torch.is_tensor(network_condition_kwargs.get("s_trunk")) else 0
    mt = _max_tokens(); STATS["max_tokens"] = mt or None
    if mt and n_tokens > mt:                                                               # the memory row's token ceiling: no cache generation for this prediction; whatever an earlier
        STATS["token_gated"] += 1                                                          # (smaller) prediction left resident is freed first, then the inner sampler serves the call
        had = self.score_model.__dict__.pop("_dit_hoist_caches", None) is not None         # (the roll-out reads the same word and hands it to the stock eager loop, the fused step prepared)
        try:
            import boltz_graph_patch as BGP
            had = bool(BGP.release_step_graph(self)) or had
        except ImportError:
            pass
        if had and torch.cuda.is_available():                                              # the allocator's cache is emptied ONLY when levers' state was resident: an above-ceiling pass
            torch.cuda.empty_cache()                                                       # keeps its cached blocks (emptying them mid-pass raises the reserved high-water and slows the sampler)
        _log("token ceiling: %d tokens > %d -> no hoist cache for this prediction (inner sampler%s)" % (n_tokens, mt, "; resident lever state freed" if had else ""))
        try:
            return inner(self, atom_mask, num_sampling_steps=num_sampling_steps, multiplicity=multiplicity, max_parallel_samples=max_parallel_samples,
                         steering_args=steering_args, **network_condition_kwargs)
        finally:
            STATS["last"] = {"hoist_level": STATS["level"], "net_calls": 0, "hits_delta": {}, "cache_builds": 0, "cache_refreshes": 0, "build_s_last": None,
                             "reallocs_under_same_key": STATS["reallocs_under_same_key"], "n_caches": 0, "captured_with_cache": STATS["captured_with_cache"],
                             "capture_stock_fallbacks": STATS["capture_stock_fallbacks"], "cache_mb": 0.0, "cache_report": {}, "graph_invalidations": STATS["graph_invalidations"],
                             "cache_released": False, "headroom_gated": False, "token_gated": True, "max_tokens": mt, "n_tokens": n_tokens}
    gate = headroom_gate(self, network_condition_kwargs, multiplicity)                    # the levers' memory-headroom gate, before any of their state exists
    if gate is not None and gate["gated"]:
        return _sample_stock_by_headroom(self, gate, inner, atom_mask, num_sampling_steps, multiplicity, max_parallel_samples, steering_args, network_condition_kwargs)
    _GEN[0] += 1; _CUR["gen"] = _GEN[0]; _CUR["net_calls"] = 0; _CUR["invalidate_graph"] = False
    STATS["samples"] += 1; h0 = dict(STATS["hits"]); nb0 = STATS["cache_builds"]; nr0 = STATS["cache_refreshes"]
    # graph composition: the graph patch refreshes its static conditioning clones and then replays WITHOUT an eager call on the static dict when the
    # graph already exists -> refresh our cache for the static dict NOW (eagerly, before any replay) from the graph's static objects.
    _ARMED.add(self)
    G = getattr(self, "_step_graph", None)
    # cache lifetime: keep ONLY the cache of the live CUDA graph's static conditioning dict (if any); every other entry belongs to a previous
    # prediction's (dead) conditioning dict -> drop it now (bounded memory: <= 1 cache between predictions, <= 2 transiently inside the first
    # prediction of a new shape while the graph patch captures).
    store = self.score_model.__dict__.setdefault("_dit_hoist_caches", {})
    keep_id = id(G.static["dc"]) if (G is not None and G.graph is not None) else None
    Cobj = next(iter(store.values()), None)
    for k in list(store):
        if k != keep_id and k != "_obj": del store[k]        # drop aliases of dead conditioning dicts (their ids may be recycled by Python)
    if Cobj is not None: store["_obj"] = Cobj                # keep the single cache object (its tensors are re-filled in place / re-allocated on shape change)
    if G is not None and G.graph is not None and _cache_for(self.score_model, G.static["dc"], create=False) is None:
        # this graph was captured while the hoist was NOT armed (or its cache was dropped): it replays the stock step and would never read a
        # cache. Results would still be exact, but to get the hoisted graph we force the graph patch to recapture now (once).
        _log("captured graph has no hoist cache -> dropping it so that the graph patch recaptures with the hoist armed")
        self._step_graph = None; G = None
        STATS["graph_invalidations"] += 1
    if G is not None and G.graph is not None:
        st = G.static; nck = network_condition_kwargs; dcn = nck["diffusion_conditioning"]
        Cg = _cache_for(self.score_model, st["dc"], create=False)
        same = (Cg is not None and tuple(st["s_trunk"].shape) == tuple(nck["s_trunk"].shape) and tuple(st["s_inputs"].shape) == tuple(nck["s_inputs"].shape)
                and set(k for k, v in dcn.items() if torch.is_tensor(v)) == set(k for k, v in st["dc"].items() if torch.is_tensor(v))
                and all(tuple(st["dc"][k].shape) == tuple(v.shape) and st["dc"][k].dtype == v.dtype for k, v in dcn.items() if torch.is_tensor(v))
                and all((k in st["feats"]) and torch.is_tensor(st["feats"][k]) and tuple(st["feats"][k].shape) == tuple(nck["feats"][k].shape) for k in (G.feats_keys or []) if k in nck["feats"] and torch.is_tensor(nck["feats"][k])))
        if same:
            # Same shapes -> the graph patch will (if its full key matches) refresh its static clones with copy_ and replay WITHOUT any eager network call on
            # the static dict. Do the identical copy_ first (idempotent w.r.t. the graph patch's own copy) and refresh our cache entries for the static dict
            # from the static objects, eagerly, before any replay. If the graph patch's key does NOT match after all (e.g. parameter pointers changed) it
            # recaptures, and the warm-up calls of the capture rebuild our cache lazily from the new static dict -> still correct.
            st["s_trunk"].copy_(nck["s_trunk"]); st["s_inputs"].copy_(nck["s_inputs"])
            for k, v in dcn.items():
                if torch.is_tensor(v):
                    st["dc"][k].copy_(v)
                elif hasattr(v, "keywords") and "indexing_matrix" in getattr(v, "keywords", {}) and k in st["dc"] and hasattr(st["dc"][k], "keywords"):
                    if tuple(st["dc"][k].keywords["indexing_matrix"].shape) == tuple(v.keywords["indexing_matrix"].shape):
                        st["dc"][k].keywords["indexing_matrix"].copy_(v.keywords["indexing_matrix"])
                    else:
                        same = False
            for k in (G.feats_keys or []):
                if k in nck["feats"] and torch.is_tensor(nck["feats"][k]):
                    st["feats"][k].copy_(nck["feats"][k])
        if same:
            _build_cache(self.score_model, Cg, dict(s_trunk=st["s_trunk"], s_inputs=st["s_inputs"], feats=st["feats"], diffusion_conditioning=st["dc"], multiplicity=Cg.m))
            Cg.gen = _GEN[0]; Cg.src = (id(st["s_trunk"]), id(st["s_inputs"]))
            # the eager step 0 of this sample() passes the caller's dict, whose tensor VALUES now equal the static clones -> let it use the same cache
            # object (no second copy of the cached tensors); its first call refreshes the (identical) values once more from the caller's objects.
            store[id(dcn)] = Cg; Cg.alias_src = (id(nck["s_trunk"]), id(nck["s_inputs"]))
            if _CUR.get("invalidate_graph"):
                self._step_graph = None; _CUR["invalidate_graph"] = False
        elif Cg is not None:
            Cg.gen = None          # shapes changed: the graph patch will recapture; the stale cache for the old static dict must not be trusted
    try:
        out = inner(self, atom_mask, num_sampling_steps=num_sampling_steps, multiplicity=multiplicity, max_parallel_samples=max_parallel_samples,
                    steering_args=steering_args, **network_condition_kwargs)
    finally:
        ncalls = _CUR["net_calls"]
        for k in _CUR: _CUR[k] = None
        _CUR["net_calls"] = 0
        STATS["last"] = {"hoist_level": STATS["level"], "net_calls": ncalls, "hits_delta": {k: STATS["hits"].get(k, 0) - h0.get(k, 0) for k in STATS["hits"]},
                         "cache_builds": STATS["cache_builds"] - nb0, "cache_refreshes": STATS["cache_refreshes"] - nr0, "build_s_last": (STATS["build_s"] or [None])[-1],
                         "reallocs_under_same_key": STATS["reallocs_under_same_key"], "n_caches": len(getattr(self.score_model, "_dit_hoist_caches", {})),
                         "captured_with_cache": STATS["captured_with_cache"], "capture_stock_fallbacks": STATS["capture_stock_fallbacks"],
                         "cache_mb": cache_report(self)["_total_mb"], "cache_report": cache_report(self), "graph_invalidations": STATS["graph_invalidations"]}
        STATS["last"].update(headroom_gated=False, projected_cache_gib=(gate or {}).get("projected_cache_gib"), free_gib=(gate or {}).get("free_gib"))
        BGP = sys.modules.get("boltz_graph_patch")
        if BGP is not None and isinstance(BGP.STATS.get("last"), dict):                                                # the graphed sampler's census of this call: the gate's figures beside it
            BGP.STATS["last"].update({k: (gate or {}).get(k) for k in ("projected_gib", "transient_gib", "margin_gib", "need_gib", "free_gib")})
        STATS["last"]["cache_released"] = release_after_sample(self, int(network_condition_kwargs["s_trunk"].shape[1]))   # after the census: cache_mb is what this sample() held
        STATS["last"]["token_gated"] = False; STATS["last"]["max_tokens"] = mt or None
    return out


sample_hoisted._dit_hoist = True


def _classes():
    from boltz.model.modules.diffusionv2 import DiffusionModule
    from boltz.model.modules.transformersv2 import AdaLN, ConditionedTransitionBlock, DiffusionTransformerLayer, DiffusionTransformer, AtomTransformer
    from boltz.model.modules.encodersv2 import SingleConditioning, AtomAttentionEncoder
    from boltz.model.layers.attentionv2 import AttentionPairBias
    return [(AdaLN, _adaln_forward), (ConditionedTransitionBlock, _ctb_forward), (DiffusionTransformerLayer, _layer_forward), (DiffusionTransformer, _dt_forward),
            (AttentionPairBias, _apb_forward), (SingleConditioning, _single_cond_forward), (AtomTransformer, _atom_transformer_forward),
            (AtomAttentionEncoder, _encoder_forward), (DiffusionModule, _dm_forward)]


def apply(level=None):
    """Install at `level` (1|2; 0/None-env = off) or remove (0). Idempotent. Apply AFTER boltz_graph_patch.apply() when composing (outermost wrapper)."""
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    if level is None:
        level = os.environ.get("BOLTZ_DIT_HOIST", "0")
    if isinstance(level, bool):
        level = 1 if level else 0
    elif isinstance(level, str):
        level = {"0": 0, "off": 0, "": 0, "false": 0, "1": 1, "on": 1, "true": 1, "2": 2}[level.lower()]
    level = int(level)
    if level != STATS["level"]:
        n_inv = 0
        for mod in list(_ARMED):
            if getattr(mod, "_step_graph", None) is not None:
                mod._step_graph = None; n_inv += 1          # captured body depends on the hoist level -> force a recapture at the new level
        if n_inv:
            STATS["graph_invalidations"] += n_inv; _log(f"level {STATS['level']} -> {level}: invalidated {n_inv} captured step graph(s)")
        if torch.cuda.is_available() and n_inv:
            torch.cuda.synchronize()
    if level >= 1:
        for cls, fn in _classes():
            if not hasattr(cls, "_stock_forward"):
                cls._stock_forward = cls.forward
            cls.forward = fn
        cur = AtomDiffusion.sample
        if not getattr(cur, "_dit_hoist", False):
            AtomDiffusion._dit_hoist_inner_sample = cur          # stock sample, or the graph patch's sample_graphed (apply the graph patch first)
            AtomDiffusion.sample = sample_hoisted
        STATS["level"] = level
    else:
        for cls, fn in _classes():
            if hasattr(cls, "_stock_forward"):
                cls.forward = cls._stock_forward
                delattr(cls, "_stock_forward")
        if getattr(AtomDiffusion.sample, "_dit_hoist", False):
            AtomDiffusion.sample = AtomDiffusion._dit_hoist_inner_sample
            delattr(AtomDiffusion, "_dit_hoist_inner_sample")
        for mod in list(_ARMED):                # release the cache memory (any graph that read it was dropped above because the level changed)
            getattr(mod, "score_model", mod).__dict__.pop("_dit_hoist_caches", None)
        STATS["level"] = 0
    _log("apply level", STATS["level"], "inner sampler:", getattr(getattr(AtomDiffusion, "_dit_hoist_inner_sample", None), "__name__", None))
    return STATS["level"]


if os.environ.get("BOLTZ_DIT_HOIST", "0") not in ("0", "off", "", "false"):
    try:
        apply()
    except Exception as e:  # boltz not importable: stay inert
        print("[boltz_dit_hoist] not applied:", e, file=sys.stderr)
