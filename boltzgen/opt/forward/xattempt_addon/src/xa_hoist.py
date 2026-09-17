
# xa_hoist.py — `hoist` for BoltzGen 0.3.2: hoist the step-invariant attention-bias preparation of the diffusion transformers out of the
# diffusion sampling loop. Works stand-alone (stock eager sampler) and on top of the CUDA-graph sampler (bg_graph_patch).
#
# Stock, in EVERY denoiser call (default 500 steps) for EACH of the 24 token-transformer layers, AttentionPairBias.forward does
#     bias = self.proj_z(z)                                  # Rearrange('b ... h -> b h ...') view of dc['token_trans_bias'][..., layer]
#     bias = bias.repeat_interleave(multiplicity, 0)         # materialises a permuted fp32 (B*mult, 16, N, N) copy  (strided-copy kernel)
#     attn_mask = (1 - mask[:, None, None].float()) * -self.inf
#     attn_mask = attn_mask + bias.float()                   # a second full (B*mult, 16, N, N) fp32 tensor
# z and mask are trunk outputs (diffusion_conditioning['token_trans_bias'], feats['token_pad_mask']); they do not change during sampling,
# so per step this is 24 strided copies and adds over tensors of that size for values that are the same at every step.
#
# Lever: compute the 24 attn_mask tensors ONCE per design with the same tensor ops in the same order (=> identical values), keep them in
# the design's diffusion_conditioning dict (lifetime == the design; under the graph sampler == the graph's static-buffer dict, i.e. the
# graph), and let AttentionPairBias read them.
#   - One structure in the trunk batch (upstream's predict batch): ONE (1, 16, N, N) fp32 slice per layer. The B*mult rows stock
#     materialises are identical by construction (one pad mask, one pair bias, repeat_interleave over the diffusion batch), and torch's SDPA
#     broadcasts an attn_mask of batch 1 over q's batch itself (the memory-efficient kernel reads the bias through its strides; same kernel,
#     same values => same output bits) — 24 x 16 x N x N x 4 bytes per design instead of 24 x B x 16 x N x N x 4 (25 GB at N=500, B=64). The
#     slice lives in a buffer whose last dimension is padded to MASK_ALIGN elements so every row stride meets the kernel's alignment and torch
#     makes no padding copy per call. A trunk batch > 1 (several structures in one predict batch) keeps the full per-row build (rows differ
#     per structure).
#   - The SIX atom-transformer layers (atom_attention_encoder 3 + atom_attention_decoder 3; windows of W=32 queries x H=128 keys) get the
#     same treatment: their pair bias (diffusion_conditioning['atom_enc_bias' / 'atom_dec_bias']), key mask (to_keys of feats['atom_pad_mask'])
#     and per-layer proj_z are step-invariant too, and stock recomputes proj_z (LayerNorm + Linear + permute) on the multiplicity-REPEATED
#     windows at every step — at 7000 atoms x 64 designs that is ~10^9 rows of elementwise work per call. The lever builds each layer's
#     (B*NW, heads, W, H) fp32 mask once per design on the UN-repeated structures and hands SDPA a per-call broadcast copy over the diffusion
#     batch in stock's own layout (row b*mult+m of stock's tensor == row b of ours: identical values, same dtype/shape/strides for the kernel).
# No pointer-keyed caches: the entry is tied to the dict object it was computed from and carries the source tensors' in-place version
# counters, so any in-place refresh (bg_graph_patch copies a new design's conditioning into its static buffers on a graph-cache hit) is
# detected; in addition bg_graph_patch's `_get_step_graph` is wrapped so the hoisted tensors are recomputed IN PLACE (same addresses, which
# the captured graph reads) right after it refreshes its static buffers and before replay. Runtime self-check: on first eager use per entry, token layers 0 and L-1 and every atom layer recompute the stock expression from the actual call arguments and assert torch.equal with the hoisted tensor; a mismatch disables the lever for the process (falls back to stock) and is logged.
# Exactness: common-subexpression hoisting of deterministic ops on unchanged inputs; no RNG consumed; SDPA receives a tensor with identical
# dtype/shape/strides/values, or the batch-1 slice it broadcasts to the same values. Env: XA_HOIST=1 (default on when imported); the self-check is always on.
import os, sys, threading
import torch
from opt_core.oom import is_oom

STATS = {"enabled": os.environ.get("XA_HOIST", "1") == "1", "builds": 0, "refresh_in_place": 0, "uses": 0, "capture_uses": 0, "capture_stock_fallback": 0,
         "selfchecks": 0, "selfcheck_fail": 0, "shared": 0, "atom_layers": 0, "atom_uses": 0, "disabled_reason": None}   # shared: designs whose 24 token masks were ONE (1, H, N, N) slice each; atom_layers: atom-transformer layers hoisted per design (6); atom_uses: their calls served
_CTX = threading.local()
_KEY = "_xa_hoist_tt"
MASK_ALIGN = 16                                   # last-dim padding (elements) of the shared slice: a multiple of every SDPA kernel's mask alignment (fp32 and bf16)


def _log(m):
    print("[xa_hoist] " + m, file=sys.stderr, flush=True)


def _disable(reason):
    STATS["enabled"] = False; STATS["disabled_reason"] = reason; _log("DISABLED: " + reason)


def _shared(feats):
    """One structure in the trunk batch (upstream's predict batch is 1): every diffusion-batch row of the stock attn_mask is the same slice."""
    return int(feats["token_pad_mask"].shape[0]) == 1


def _build_masks(score_model, feats, dc, multiplicity, into=None):
    """The stock op sequence (diffusion.py DiffusionModule.forward -> transformers.py DiffusionTransformer.forward ->
    attention.py AttentionPairBias.forward), evaluated once. Returns list of per-layer attn_mask tensors (or copies into `into`):
    (1, H, N, N) aligned slices when the trunk batch is one structure (_shared), the full (B*mult, H, N, N) tensors otherwise."""
    tt = score_model.token_transformer
    layers = tt.layers
    shared = _shared(feats)
    mask = feats["token_pad_mask"] if shared else feats["token_pad_mask"].repeat_interleave(multiplicity, 0)      # DiffusionModule.forward (the repeat makes identical rows: skipped when shared)
    mask = mask.float()                                                     # token_transformer(mask=mask.float(), ...)
    bias = dc["token_trans_bias"].float()                                   # bias=diffusion_conditioning["token_trans_bias"].float()
    B, N, M, D = bias.shape; L = len(layers)
    bias = bias.view(B, N, M, L, D // L)                                    # DiffusionTransformer.forward
    out = []
    for i, layer in enumerate(layers):
        apb = layer.pair_bias_attn
        bias_l = bias[:, :, :, i]
        b = apb.proj_z(bias_l)                                              # AttentionPairBias.forward
        if not shared:
            b = b.repeat_interleave(multiplicity, 0)
        attn_mask = (1 - mask[:, None, None].float()) * -apb.inf
        attn_mask = attn_mask + b.float()
        if into is not None:
            into[i].copy_(attn_mask)
        elif shared:                                                        # the slice in a last-dim-aligned buffer (no per-call padding copy in SDPA); the view IS (1, H, N, N)
            npad = -(-attn_mask.shape[-1] // MASK_ALIGN) * MASK_ALIGN
            buf = torch.empty(attn_mask.shape[:-1] + (npad,), dtype=attn_mask.dtype, device=attn_mask.device)
            view = buf[..., :attn_mask.shape[-1]]
            view.copy_(attn_mask)
            out.append(view)
        else:
            out.append(attn_mask)
    return into if into is not None else out


def _atom_transformers(score_model):
    """[(AtomTransformer, conditioning key, mask as its caller hands it)] for the encoder and the decoder (encoders.py
    AtomAttentionEncoder.forward: mask=feats['atom_pad_mask'].bool(); AtomAttentionDecoder.forward: mask=feats['atom_pad_mask'])."""
    enc = getattr(getattr(score_model, "atom_attention_encoder", None), "atom_encoder", None)
    dec = getattr(getattr(score_model, "atom_attention_decoder", None), "atom_decoder", None)
    if enc is None or dec is None:
        return []
    return [(enc, "atom_enc_bias", lambda f: f["atom_pad_mask"].bool()), (dec, "atom_dec_bias", lambda f: f["atom_pad_mask"])]


def _build_atom_masks(score_model, feats, dc, into=None):
    """The stock op sequence for the atom layers (encoders.py AtomAttention{En,De}coder.forward -> transformers.py AtomTransformer.forward ->
    DiffusionTransformer.forward -> DiffusionTransformerLayer.forward -> attention.py AttentionPairBias.forward), evaluated once per design on
    the UN-repeated structures (multiplicity 1: every repeated row equals its structure's row). Returns the per-layer (B*NW, heads, W, H) fp32
    attn_mask tensors in layer order encoder 0..L-1 then decoder 0..L-1 (or copies them into `into`)."""
    to_keys = dc["to_keys"]
    out = []; k = 0
    for tfm, key, mask_of in _atom_transformers(score_model):
        W, H = tfm.attn_window_queries, tfm.attn_window_keys
        mask = mask_of(feats)                                                   # AtomAttention{En,De}coder.forward (repeat_interleave(multiplicity, 0) skipped: multiplicity 1)
        bias = dc[key].float()                                                  # DiffusionModule.forward: atom_enc_bias=... .float()
        B, N = mask.shape[0], mask.shape[1]; NW = N // W
        mask = mask.view(B * NW, W)                                             # AtomTransformer.forward
        bias = bias.repeat_interleave(1, 0)
        bias = bias.view((bias.shape[0] * NW, W, H, -1))
        to_keys_new = lambda x, B=B, NW=NW, W=W, H=H: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)
        mask = mask.float()                                                     # AtomTransformer.forward: mask=mask.float()
        layers = tfm.diffusion_transformer.layers; L = len(layers)
        Bq, Nq, M, D = bias.shape
        bias = bias.view(Bq, Nq, M, L, D // L)                                  # DiffusionTransformer.forward
        maskk = to_keys_new(mask.unsqueeze(-1)).squeeze(-1)                     # DiffusionTransformerLayer.forward
        for i, layer in enumerate(layers):
            apb = layer.pair_bias_attn
            b = apb.proj_z(bias[:, :, :, i])                                    # AttentionPairBias.forward
            b = b.repeat_interleave(1, 0)
            attn_mask = (1 - maskk[:, None, None].float()) * -apb.inf
            attn_mask = attn_mask + b.float()                                   # (B*NW, heads, W, H)
            if into is not None:
                into[k].copy_(attn_mask)
            else:
                out.append(attn_mask.contiguous())
            k += 1
    return into if into is not None else out


def _atom_versions(feats, dc):
    parts = []
    for key in ("atom_enc_bias", "atom_dec_bias"):
        t = dc.get(key); parts += [id(t), _ver(t), tuple(t.shape)] if t is not None else [None]
    am = feats.get("atom_pad_mask"); parts += [id(am), _ver(am), tuple(am.shape)] if am is not None else [None]
    parts.append(id(dc.get("to_keys")))
    return tuple(parts)


def _ver(t):
    # inference-mode tensors (Lightning's predict loop) have no version counter -> treat as immutable within a design (they are: fresh
    # tensors per design; the only in-place writer is bg_graph_patch's static-buffer refresh, which is hooked explicitly below).
    try:
        return t._version
    except Exception:
        return -1


def _versions(feats, dc, multiplicity):
    tb = dc["token_trans_bias"]; tm = feats["token_pad_mask"]
    return (id(tb), _ver(tb), tuple(tb.shape), id(tm), _ver(tm), tuple(tm.shape), int(multiplicity)) + _atom_versions(feats, dc)


def _get_entry(score_model, feats, dc, multiplicity):
    """Return the hoisted mask list for this (dc, feats, multiplicity), building or refreshing it if needed.
    Under CUDA-graph capture: an entry OWNED by this dict with matching shapes is used as-is (its values are refreshed in place from the
    graph's static sources by the graph-bind hook before every replay); nothing is ever built inside a capture."""
    capturing = torch.cuda.is_current_stream_capturing()
    ent = dc.get(_KEY)
    if ent is not None and ent.get("owner") != id(dc):
        ent = None                                            # aliased from another dict (bg_graph_patch's static copy shares non-tensor values)
        if not capturing:
            dc.pop(_KEY, None)
    if ent is not None and ent["model"] == id(score_model):
        shapes_ok = (ent["mult"] == int(multiplicity) and tuple(ent["src_shape"]) == tuple(dc["token_trans_bias"].shape)
                     and tuple(ent["mask_shape"]) == tuple(feats["token_pad_mask"].shape) and ent["atom_shapes"] == _atom_shapes(feats, dc))
        if capturing:
            if shapes_ok:
                STATS["capture_uses"] += 1
                return ent
            STATS["capture_stock_fallback"] += 1
            return None
        if ent["ver"] == _versions(feats, dc, multiplicity):
            return ent
        try:                                                  # same dict, sources refreshed in place or replaced: rebuild
            if shapes_ok:
                _build_masks(score_model, feats, dc, multiplicity, into=ent["masks"]); _build_atom_masks(score_model, feats, dc, into=ent["amasks"]); STATS["refresh_in_place"] += 1
            else:
                ent["masks"] = _build_masks(score_model, feats, dc, multiplicity); ent["amasks"] = _build_atom_masks(score_model, feats, dc); STATS["builds"] += 1
            ent["ver"] = _versions(feats, dc, multiplicity); ent["mult"] = int(multiplicity)
            ent["src_shape"] = tuple(dc["token_trans_bias"].shape); ent["mask_shape"] = tuple(feats["token_pad_mask"].shape); ent["atom_shapes"] = _atom_shapes(feats, dc); ent["checked"] = set()
            return ent
        except Exception as e:
            if is_oom(e): raise
            _disable("refresh failed: %r" % e); return None
    if capturing:
        STATS["capture_stock_fallback"] += 1                  # first sight inside a capture: stock path (exact, just not hoisted)
        return None
    masks = _build_masks(score_model, feats, dc, multiplicity); amasks = _build_atom_masks(score_model, feats, dc); STATS["builds"] += 1
    by_apb = {id(layer.pair_bias_attn): i for i, layer in enumerate(score_model.token_transformer.layers)}
    k = 0
    for tfm, _key, _m in _atom_transformers(score_model):                    # atom layers: ("a", k) in encoder-then-decoder order (== amasks' order)
        for layer in tfm.diffusion_transformer.layers:
            by_apb[id(layer.pair_bias_attn)] = ("a", k); k += 1
    STATS["atom_layers"] = len(amasks)
    ent = {"ver": _versions(feats, dc, multiplicity), "masks": masks, "amasks": amasks, "model": id(score_model), "mult": int(multiplicity),
           "src_shape": tuple(dc["token_trans_bias"].shape), "mask_shape": tuple(feats["token_pad_mask"].shape), "atom_shapes": _atom_shapes(feats, dc),
           "checked": set(), "owner": id(dc), "by_apb": by_apb}
    dc[_KEY] = ent
    return ent


def _atom_shapes(feats, dc):
    return tuple(tuple(dc[k].shape) if k in dc else None for k in ("atom_enc_bias", "atom_dec_bias")) + (tuple(feats["atom_pad_mask"].shape) if "atom_pad_mask" in feats else None,)


def install():
    import boltzgen.model.modules.diffusion as D
    from boltzgen.model.layers.attention import AttentionPairBias
    if getattr(D.DiffusionModule.forward, "_xa_hoist", False):
        return
    _stock_dm_forward = D.DiffusionModule.forward
    _stock_apb_forward = AttentionPairBias.forward

    def dm_forward(self, s_inputs, s_trunk, r_noisy, times, feats, diffusion_conditioning, multiplicity=1):
        ent = None
        if STATS["enabled"] and not self.training and not torch.is_grad_enabled():
            try:
                ent = _get_entry(self, feats, diffusion_conditioning, multiplicity)
            except Exception as e:
                if is_oom(e): raise
                _disable("build failed: %r" % e); ent = None
        prev = getattr(_CTX, "ent", None)
        _CTX.ent = ent
        try:
            return _stock_dm_forward(self, s_inputs, s_trunk, r_noisy, times, feats, diffusion_conditioning, multiplicity=multiplicity)
        finally:
            _CTX.ent = prev
    dm_forward._xa_hoist = True
    D.DiffusionModule.forward = dm_forward

    def _atom_apb_forward(self, s, z, mask, k_in, multiplicity, ent, k):
        """Stock AttentionPairBias.forward for a windowed ATOM layer with `bias = proj_z(z); attn_mask = (1 - mask)*-inf + bias` replaced by the
        design's hoisted (B*NW, heads, W, H) tensor broadcast over the diffusion batch into stock's own (B*mult*NW, heads, W, H) layout (ONE
        copy kernel; row (b*mult + m)*NW + w of stock's tensor is row b*NW + w of ours). multiplicity here is 1 (AtomTransformer folds it into
        the batch); the batch factor is the entry's."""
        am = ent["amasks"][k]
        Bw = s.shape[0]; BNW = am.shape[0]; mult = max(1, Bw // BNW)
        B_ = ent["mask_shape"][0]; NW = BNW // B_

        def stock_call():                                                   # stock's forward on stock's tensor: the un-repeated windows (at_forward) expanded to stock's rows first
            zz = z if z.shape[0] == Bw else z.view(B_, 1, NW, *z.shape[1:]).expand(B_, mult, NW, *z.shape[1:]).reshape(Bw, *z.shape[1:])
            return _stock_apb_forward(self, s, zz, mask, k_in, multiplicity=multiplicity)
        if Bw != BNW * mult or not STATS["enabled"]:
            return stock_call()                                             # a shape the entry was not built for, or the lever went off mid-design: stock, this call
        q = self.proj_q(s).view(Bw, -1, self.num_heads, self.head_dim)
        kk = self.proj_k(k_in).view(Bw, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(Bw, -1, self.num_heads, self.head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q)
            kk = self.k_norm(kk)
        q = q.transpose(1, 2)
        kk = kk.transpose(1, 2)
        v = v.transpose(1, 2)
        if mult == 1:
            attn_mask = am
        else:
            attn_mask = am.view(B_, 1, NW, *am.shape[1:]).expand(B_, mult, NW, *am.shape[1:]).reshape(Bw, *am.shape[1:])   # the broadcast over the diffusion batch, materialised once (stock's layout)
        if ("a", k) not in ent["checked"] and not torch.cuda.is_current_stream_capturing():
            STATS["selfchecks"] += 1                                        # once per design per atom layer: stock's statements on the FIRST and the LAST window row equal ours
            same = attn_mask.dtype == torch.float32 and tuple(attn_mask.shape) == (Bw, self.num_heads) + tuple(am.shape[2:])
            for r in sorted({0, Bw - 1}):
                zr = r if z.shape[0] == Bw else (r // (mult * NW)) * NW + r % NW   # stock row r = structure b, design m, window w -> its bias row: r itself when the layer received
                ref = self.proj_z(z[zr:zr + 1])                                 # stock's repeated tensor, row b*NW + w of the un-repeated windows otherwise (at_forward below)
                ref = (1 - mask[r:r + 1, None, None].float()) * -self.inf + ref.float()
                same = same and torch.equal(ref, attn_mask[r:r + 1])
            if not same:
                STATS["selfcheck_fail"] += 1
                _disable("self-check mismatch at atom layer %d (rows 0 and %d of %d vs hoisted %s)" % (k, Bw - 1, Bw, tuple(am.shape)))
                return stock_call()
            ent["checked"].add(("a", k))
        STATS["atom_uses"] += 1
        g = self.proj_g(s)
        g.sigmoid_()
        with torch.autocast("cuda", enabled=False):
            o = torch.nn.functional.scaled_dot_product_attention(q.float(), kk.float(), v.float(), attn_mask=attn_mask)
        o = o.permute(0, 2, 1, 3).reshape(Bw, -1, self.c_s)
        o = o * g
        o = self.proj_o(o)
        return o

    def apb_forward(self, s, z, mask, k_in, multiplicity=1):
        ent = getattr(_CTX, "ent", None)
        li = None if ent is None else ent["by_apb"].get(id(self))
        if li is None:
            return _stock_apb_forward(self, s, z, mask, k_in, multiplicity=multiplicity)
        if isinstance(li, tuple):                                              # an ATOM-transformer layer (windowed attention): ("a", k)
            return _atom_apb_forward(self, s, z, mask, k_in, multiplicity, ent, li[1])
        # ---- stock AttentionPairBias.forward with the two bias/attn_mask statements replaced by the hoisted tensor ----
        B = s.shape[0]
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = ent["masks"][li]                                        # (1, H, N, N) shared slice (torch's SDPA broadcasts batch 1 over q's batch) or the full stock-shaped tensor
        if li in (0, len(ent["masks"]) - 1) and li not in ent["checked"] and not torch.cuda.is_current_stream_capturing():
            STATS["selfchecks"] += 1                                        # once per design at the first and the last layer: the stock statements' tensor equals the hoisted one
            Bm = int(mask.shape[0]); shared = attn_mask.shape[0] == 1 and Bm != 1
            if shared:                                                      # a shared slice: the FIRST and the LAST of the B*mult rows stock would hand SDPA, each built by the
                bias1 = self.proj_z(z)                                      # stock statements on that row alone (row r of bias.repeat_interleave(mult, 0) is bias[r // mult]) —
                same = attn_mask.dtype == torch.float32                     # no (B*mult, H, N, N) tensor is materialised for the check
                for r in sorted({0, Bm - 1}):
                    ref = (1 - mask[r:r + 1, None, None].float()) * -self.inf
                    ref = ref + bias1[r // int(multiplicity):r // int(multiplicity) + 1].float()
                    same = same and ref.shape == attn_mask.shape and torch.equal(ref, attn_mask)
                what = "rows 0 and %d of %d" % (Bm - 1, Bm)
            else:
                bias = self.proj_z(z)                                       # the literal stock statements, whole tensor
                bias = bias.repeat_interleave(multiplicity, 0)
                ref = (1 - mask[:, None, None].float()) * -self.inf
                ref = ref + bias.float()
                same = ref.shape == attn_mask.shape and ref.dtype == attn_mask.dtype and ref.stride() == attn_mask.stride() and torch.equal(ref, attn_mask)
                what = "tensor %s" % (tuple(ref.shape),)
            if not same:
                STATS["selfcheck_fail"] += 1
                _disable("self-check mismatch at layer %d (%s vs hoisted %s)" % (li, what, tuple(attn_mask.shape)))
                bias = self.proj_z(z)                                       # this call: the stock tensor
                bias = bias.repeat_interleave(multiplicity, 0)
                attn_mask = (1 - mask[:, None, None].float()) * -self.inf
                attn_mask = attn_mask + bias.float()
            else:
                ent["checked"].add(li)
                if shared and li == 0:
                    STATS["shared"] += 1
        STATS["uses"] += 1
        g = self.proj_g(s)
        g.sigmoid_()
        with torch.autocast("cuda", enabled=False):
            o = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=attn_mask)
        o = o.permute(0, 2, 1, 3).reshape(B, -1, self.c_s)
        o = o * g
        o = self.proj_o(o)
        return o
    apb_forward._xa_hoist = True
    AttentionPairBias.forward = apb_forward

    # ---- transformers.py AtomTransformer.forward: its `bias = bias.repeat_interleave(multiplicity, 0)` materialises the atom pair bias for
    # every design of the diffusion batch at every step ((B*mult*NW, W, H, D) fp32 — gigabytes at thousands of atoms x 64 designs) only for
    # proj_z to read it; with the atom layers served from the hoisted masks nothing reads it, so the statement is skipped and the layers
    # receive the un-repeated windows (read only by the self-check, which maps a stock row to its structure's window). Every other statement
    # is stock's, in stock's order.
    import boltzgen.model.modules.transformers as T
    _stock_at_forward = T.AtomTransformer.forward

    def at_forward(self, q, c, bias, to_keys, mask, multiplicity=1):
        ent = getattr(_CTX, "ent", None)
        if ent is None or not STATS["enabled"] or not ent.get("amasks") or multiplicity == 1:
            return _stock_at_forward(self, q, c, bias, to_keys, mask, multiplicity=multiplicity)
        W = self.attn_window_queries
        H = self.attn_window_keys
        B, N, D = q.shape
        NW = N // W
        q = q.view((B * NW, W, -1))
        c = c.view((B * NW, W, -1))
        mask = mask.view(B * NW, W)
        bias = bias.view((bias.shape[0] * NW, W, H, -1))                       # NOT repeat_interleave'd over the diffusion batch
        to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)
        q = self.diffusion_transformer(a=q, s=c, bias=bias, mask=mask.float(), multiplicity=1, to_keys=to_keys_new)
        q = q.view((B, NW * W, D))
        return q
    at_forward._xa_hoist = True
    T.AtomTransformer.forward = at_forward

    # ---- graph-sampler (bg_graph_patch) integration ----
    # bg_graph_patch's _StepGraph owns a conditioning dict G.static["dc"]: tensors are CLONED from the design's dict at capture and refreshed
    # IN PLACE (.copy_) on a graph-cache hit; non-tensor values (such as our entry) are passed BY REFERENCE. The captured kernels read whatever
    # attn_mask tensors AttentionPairBias used during capture. Rule enforced here:
    #   (1) _StepGraph.body (used for its eager warm-up AND the capture) always runs with an entry OWNED by G.static["dc"]: an aliased
    #       entry inherited from the design's dict is dropped before the first warm-up body, so _get_entry builds a fresh one from
    #       G.static's own tensors; the capture then bakes THOSE addresses into the graph.
    #   (2) after every _get_step_graph (capture or cache hit) the owned entry is recomputed in place from G.static's (refreshed) sources,
    #       before the first replay.
    try:
        import bg_graph_patch as G
        if not getattr(G._get_step_graph, "_xa_hoist", False):
            _orig_body = G._StepGraph.body
            def body(self):
                sdc = self.static.get("dc")
                if STATS["enabled"] and isinstance(sdc, dict):
                    ent = sdc.get(_KEY)
                    if ent is not None and ent.get("owner") != id(sdc):
                        del sdc[_KEY]                                  # (1) aliased -> rebuild owned by this graph's dict
                return _orig_body(self)
            G._StepGraph.body = body
            _orig_gsg = G._get_step_graph
            def _get_step_graph(diff, x, R, tr, eps, atom_mask, multiplicity, nck):
                SG = _orig_gsg(diff, x, R, tr, eps, atom_mask, multiplicity, nck)
                if SG is None or not STATS["enabled"]:
                    return SG
                try:
                    st = SG.static; sdc = st["dc"]; ent = sdc.get(_KEY)
                    if ent is None or ent.get("owner") != id(sdc):
                        _disable("graph static dict has no owned hoist entry after capture (unexpected); falling back to stock for new graphs")
                        return SG
                    _build_masks(diff.score_model, st["feats"], sdc, st["multiplicity"], into=ent["masks"])   # (2)
                    _build_atom_masks(diff.score_model, st["feats"], sdc, into=ent["amasks"])
                    ent["ver"] = _versions(st["feats"], sdc, st["multiplicity"]); STATS["refresh_in_place"] += 1
                except Exception as e:
                    if is_oom(e): raise
                    _disable("refresh after graph bind failed: %r" % e)
                return SG
            _get_step_graph._xa_hoist = True
            G._get_step_graph = _get_step_graph
            _log("installed (with partner graph-sampler integration)")
        else:
            _log("installed")
    except ImportError:
        _log("installed (stand-alone; partner bg_graph_patch not on path)")


if STATS["enabled"]:
    try:
        install()
    except Exception as e:
        _disable("not installed: %r" % e)
