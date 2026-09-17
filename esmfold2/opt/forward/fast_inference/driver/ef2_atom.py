"""ef2_atom — the atom path of ESMFold2's diffusion module: ESMFold2AtomEncoder / ESMFold2AtomDecoder and the SWAAtomTransformer
blocks under them (modeling_esmfold2_common.py @ef32577f: encoder :771-925, decoder :933-1002, SWAAtomBlock :651-700, SWA3DRoPEAttention :547-635),
run once per diffusion step (134 executed steps per fold) inside the sampler graph.

    import ef2_atom
    ef2_atom.install(model, hoist_adaln=True, prefix_varlen=True, rope_tables=True, skip_copies=True)            # exact tier (bitwise)
    ef2_atom.install(model, <exact levers>, fused_block=True, seg_mean=True, gemm="bf16")                        # fast tier (the kit's composition)
    ef2_atom.install(model, <exact levers>, fused_block=True, seg_mean=True, gemm="fp32" | "tf32" | "tf32x3")   # the GEMM precision sub-choice
    ef2_atom.install(model, **ef2_atom.parse_flags("ax,af,gemm=bf16")); ef2_atom.uninstall(model)              # ax = a1..a4, af = a5+a6 (the kit's names)

Install BEFORE ef2_opt.install (graph capture last); instance patches only (the diffusion module's atom_encoder / atom_decoder and their 6 blocks;
the input embedder's atom encoder, once per fold and cache-less, keeps its stock forward). Nothing under stock/ is edited; no environment variable
is read here. Every lever is its own install flag; STATE / describe() / stats() record what engaged.

EXACT tier (bitwise: the same kernels on operands of the same values; only caching of step-invariant results and pure data movement change):
  a1 hoist_adaln    SWAAtomBlock.adaln_modulation(c_l) — SiLU + Linear(128->768) on the step-invariant per-atom conditioning c_l — and the two
                    `1 + scale` factors of its rms-adaLN, computed once per fold per block at the fold's eager step 0 into static buffers and read at
                    every later step (24 kernels/step fewer).
  a2 prefix_varlen  when the fold's valid atoms are a prefix of the flattened [B*N] axis (indices == arange(n_valid): decided once per fold at step 0
                    from the indices tensor, never assumed), index_first_axis(q|k|v, indices) is the row slice [:n_valid] (a view) and flash-attn
                    writes its output into the zero-initialised padded buffer (varlen_fwd out=) instead of pad_input's index_put; other layouts run
                    the stock gather / index_put (30 kernels/step fewer).
  a3 rope_tables    apply_rotary_emb_3d's tables built once per fold ([B,N,1,hd]: cat(cos,cos) and the signed cat(-sin,sin), so rotate_half's
                    negation rides in the table: (-x2)*sin == x2*(-sin)); the cat with the empty tail x[..., ro_dim:] (ro_dim == head_dim) dropped
                    (8 kernels/block fewer).
  a4 skip_copies    repeat_interleave(1, 0) copies of step-invariant tensors (c, q's base, atom_to_token) skipped when one diffusion sample runs.
FAST tier (tolerance class; deterministic — no atomics):
  a5 fused_block    one SWAAtomBlock = a handful of kernels instead of ~55. gemm="fp32" (default): the five projection GEMMs stay on cuBLAS fp32
                    (q|k|v|gate as ONE [512,128] GEMM — bitwise the stock Linears' values — out_proj, w_up, w_down, and the coder tails) and everything
                    between them is fused in Triton: [rms-adaLN (+ the previous block's gated residual, + the encoder's coords_linear / the decoder's
                    token->atom gather+add prologue)] [window-128 attention straight off the GEMM output: qk rms-norm + 3D RoPE + bf16 cast on load,
                    bf16 x bf16 -> fp32 softmax(QK^T)V exactly stock's flash precision class, * sigmoid(gate), fp32 out] [SwiGLU] [last residual
                    (+ the decoder's LayerNorm + 128->3 head)]: 58 launches/step, fp32 arithmetic everywhere stock is fp32.
                    gemm="bf16" | "tf32" | "tf32x3" (the GEMM PRECISION lever, same vocabulary as ef2_dit): fully fused blocks, 3 kernels each —
                    K1 [prologue + rms-adaLN + q|k|v|gate GEMM + qk-norm + RoPE], K2 [attention per (query tile, head)], K3 [out_proj + residual +
                    rms-adaLN + W_up + SwiGLU + W_down + residual (+ head)] — with the GEMMs on tensor cores: bf16 operands / fp32 accumulate, TF32,
                    or 3xTF32 split emulation (fp32-faithful, slower); 25 launches/step.
  a6 seg_mean       the encoder tail relu(atom_to_token_linear(q)) -> per-token mean as a segmented reduction over the (sorted — checked once per fold —)
                    atom->token map instead of scatter_reduce (gemm="fp32": cuBLAS GEMM + one relu/segment-mean kernel; else fused GEMM+relu+mean).
Layout decisions are made ONCE per fold at the eager step 0 and recorded (STATS): valid atoms a row prefix (else the fused path hands the fold to
the unfused levers), atom->token map sorted (else the stock scatter mean). Multi-sample folds (num_diffusion_samples > 1) run the fused path.

GRAPH SAFETY (ef2_opt's pool discipline): every hoisted tensor lives in a static buffer keyed by (module, tag, shape, dtype, device), refreshed IN
PLACE at each fold's eager step 0 (the atom encoder meets the sampler's fresh, empty inference cache there — two folds of the same padded shape
share the buffers and each refills them) and never freed while a captured graph may read it: clear_static() drops them all and is chained to
ef2_opt.clear_graphs at install. A stale buffer met during capture raises by name. Every consumer fetches its buffer through _static() at every
call — no _static() result is cached across calls (HAZARDS 29): a generation reset in the middle of a fold (after step 0 filled the
registry) is followed by an eager warm-up that re-registers what the next capture reads, and only registry buffers are refreshed in place by the
next fold; a buffer cached outside the registry across such a reset is freed under a live graph.
"""
import os, sys, types, collections, threading

import torch
import torch.nn.functional as F

VERSION = "katom0.7"
STATS = collections.Counter()
STATE = {"installed": False, "levers": {}, "modules": [], "notes": []}
LEVERS_EXACT = ("hoist_adaln", "prefix_varlen", "rope_tables", "skip_copies")
LEVERS_FAST = ("fused_block", "seg_mean")
GEMM_PRECISIONS = ("fp32", "tf32", "bf16", "tf32x3")
FLAG_NAMES = {"a1": "hoist_adaln", "a2": "prefix_varlen", "a3": "rope_tables", "a4": "skip_copies", "a5": "fused_block", "a6": "seg_mean"}
GROUP_NAMES = {"ax": ("hoist_adaln", "prefix_varlen", "rope_tables", "skip_copies"), "af": ("fused_block", "seg_mean")}   # the kit's lever names: ax = the exact hoists, af = the fused block + segmented mean

_CFG = {k: False for k in LEVERS_EXACT + LEVERS_FAST}
_CFG.update(gemm="fp32", tc_gemm=False)
_STATIC = {}                       # (id(module), tag, shape, dtype, device) -> tensor (static address for the life of a graph generation)
_FILLED = {}                       # same key -> fold epoch at which the buffer was last refreshed
_FOLD = {"n": 0}                   # fold epoch: bumped when the diffusion module's atom encoder meets an empty inference cache (step 0 of sample())
_CTX = threading.local()           # per-call context handed from the encoder/decoder forward to the blocks (None outside them)
_PATCHED = []                      # (module, attr, original) for uninstall


def _common():
    import transformers.models.esmfold2.modeling_esmfold2_common as C
    return C


# =====================================================================================================================
# static buffers
# =====================================================================================================================
def _static(key, build):
    """The static tensor for ``key``; (re)filled from ``build()`` when its epoch is not the current fold's. Refill = copy_ in place when the
    shape/dtype match (live graphs read this address), else a new buffer under the (shape-including) key. Never called to build during capture."""
    ep = _FOLD["n"]
    ent = _STATIC.get(key)
    if ent is not None and _FILLED.get(key) == ep:
        STATS["static_hits"] += 1
        return ent
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"ef2_atom: static buffer {key[1]!r} missing/stale during CUDA-graph capture (the fold's eager step 0 must run first)")
    val = build()
    if ent is not None and ent.shape == val.shape and ent.dtype == val.dtype and ent.device == val.device:
        ent.copy_(val)
        STATS["static_refresh"] += 1
    else:
        ent = val.clone() if val.is_contiguous() else val.contiguous()
        _STATIC[key] = ent
        STATS["static_allocs"] += 1
    _FILLED[key] = ep
    return ent


def clear_static():
    """Drop every static buffer (call together with ef2_opt.clear_graphs: a graph of the dropped generation must never replay afterwards)."""
    _STATIC.clear(); _FILLED.clear()
    STATS["static_clears"] += 1


def release_statics_if_eager(where="confidence"):
    """The memory mode's release point: when NO CUDA graph is captured in this process (EF2_GRAPH_CAPTURE=0 — ef2_opt.CFG.graph_capture
    False; the memory mode and its row-sharded route), the atom-path static buffers (adaLN modulation tables, RoPE tables, window layouts:
    ~1 GiB at 1400 tokens, resident for the life of a graph generation otherwise) are dead once the sampler returned and are dropped before
    ``where`` (the confidence head, the fold's last pair-sized phase); the next fold's eager step 0 refills them.  With graphs captured
    (exact / fast) nothing is released: a live graph reads these addresses.  Returns the GiB released (0.0 when nothing was)."""
    opt = sys.modules.get("ef2_opt")
    cfg = getattr(opt, "CFG", None)
    if cfg is not None:
        capture = bool(getattr(cfg, "graph_capture", True))
    else:
        capture = os.environ.get("EF2_GRAPH_CAPTURE", "1").strip().lower() not in ("0", "off", "no", "false")
    if capture or not _STATIC:
        return 0.0
    nbytes = sum(int(t.numel()) * int(t.element_size()) for t in _STATIC.values() if hasattr(t, "numel"))
    clear_static()
    STATS["static_released_eager"] += 1
    STATS["static_released_mib"] += int(nbytes // 2 ** 20)
    return nbytes / 2 ** 30


# =====================================================================================================================
# SWA3DRoPEAttention.forward — stock statements (:561-635) with levers a2 (prefix_varlen) and a3 (rope_tables)
# =====================================================================================================================
def _rope_tables(x, cos_rep, sin_signed):
    """apply_rotary_emb_3d with per-fold tables: cos_rep = cat(cos, cos), sin_signed = cat(-sin, sin) ([B,N,1,hd]). Bitwise the stock statements:
    rotate_half(x) * sin_rep == cat(x2, x1) * cat(-sin, sin) elementwise ((-a)*b == a*(-b) in IEEE arithmetic), x[..., :ro_dim] == x when
    ro_dim == head_dim, and the cat with the empty tail x[..., ro_dim:] is the identity copy (dropped): 4 kernels per call instead of 8."""
    ro_dim = cos_rep.shape[-1]
    if ro_dim != x.shape[-1]:                       # not the model's configuration (ro_dim < head_dim): the stock statements
        xr = x[..., :ro_dim]
        x1, x2 = xr.chunk(2, dim=-1)
        return torch.cat([xr * cos_rep + torch.cat((x2, x1), dim=-1) * sin_signed, x[..., ro_dim:]], dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos_rep + torch.cat((x2, x1), dim=-1) * sin_signed


def _attn_forward(self, x, attention_params):
    C = _common()
    ctx = getattr(_CTX, "cur", None)
    B, N = x.shape[:2]
    cos, sin = attention_params[0], attention_params[1]

    x_input = x
    qkv = self.Wqkv(x)
    qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
    q, k, v = qkv.unbind(0)
    q, k = C.qk_norm(q), C.qk_norm(k)

    if ctx is not None and ctx.get("rope") is not None:                        # a3
        cos_rep, sin_signed = ctx["rope"]
        q = _rope_tables(q, cos_rep, sin_signed)
        k = _rope_tables(k, cos_rep, sin_signed)
        STATS["a3_rope_table_calls"] += 1
    else:
        q = C.apply_rotary_emb_3d(q, cos, sin)
        k = C.apply_rotary_emb_3d(k, cos, sin)

    input_dtype = q.dtype
    if q.dtype not in (torch.float16, torch.bfloat16):
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    if len(attention_params) > 2 and C.FLASH_ATTN_AVAILABLE:
        indices, cu_seqlens, max_seqlen = attention_params[2], attention_params[3], attention_params[4]
        n_valid = ctx.get("prefix_n") if ctx is not None else None
        if n_valid is not None:                                                # a2: indices == arange(n_valid) (proven at this fold's step 0)
            from flash_attn.flash_attn_interface import flash_attn_gpu
            qf = q.reshape(-1, self.n_heads, self.head_dim)
            kf = k.reshape(-1, self.n_heads, self.head_dim)
            vf = v.reshape(-1, self.n_heads, self.head_dim)
            q_unpad, k_unpad, v_unpad = qf[:n_valid], kf[:n_valid], vf[:n_valid]
            if q_unpad.stride(-1) != 1: q_unpad = q_unpad.contiguous()
            if k_unpad.stride(-1) != 1: k_unpad = k_unpad.contiguous()
            if v_unpad.stride(-1) != 1: v_unpad = v_unpad.contiguous()
            if n_valid == B * N:
                out_full = torch.empty(B * N, self.n_heads, self.head_dim, dtype=q.dtype, device=q.device)
            else:
                out_full = torch.zeros(B * N, self.n_heads, self.head_dim, dtype=q.dtype, device=q.device)   # pad_input's zero rows
            flash_attn_gpu.varlen_fwd(q_unpad, k_unpad, v_unpad, out_full[:n_valid], cu_seqlens, cu_seqlens, None, None, None, None,
                                      max_seqlen, max_seqlen, 0.0, self.scale, False, False, self.half_window, self.half_window, 0.0, False, None)
            out = out_full.view(B, N, self.n_heads, self.head_dim)
            STATS["a2_prefix_calls"] += 1
        else:
            q_unpad = C.index_first_axis(q.reshape(-1, self.n_heads, self.head_dim), indices)
            k_unpad = C.index_first_axis(k.reshape(-1, self.n_heads, self.head_dim), indices)
            v_unpad = C.index_first_axis(v.reshape(-1, self.n_heads, self.head_dim), indices)
            out_unpad = C.flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                                                 softmax_scale=self.scale, window_size=(self.half_window, self.half_window))
            out = C.pad_input(out_unpad, indices, B, N)
    elif C.FLASH_ATTN_AVAILABLE:
        out = C.flash_attn_func(q, k, v, softmax_scale=self.scale, window_size=(self.half_window, self.half_window))
    else:
        return self._katom_eager_forward(x, attention_params)                  # the dense fallback: not this module's business (esmfold2_opt.atom_swa)

    out = out.to(input_dtype).reshape(B, N, -1)
    out = out * torch.sigmoid(self.gate_proj(x_input))
    return self.out_proj(out)


# =====================================================================================================================
# SWAAtomBlock.forward — stock statements (:687-700) with lever a1 (hoist_adaln)
# =====================================================================================================================
def _block_forward(self, x, c_l, attention_params):
    ctx = getattr(_CTX, "cur", None)
    if ctx is not None and ctx.get("hoist") and not torch.is_grad_enabled():
        key = (id(self), "adaln_mod", tuple(c_l.shape), c_l.dtype, str(c_l.device))
        mod = _static(key, lambda: self.adaln_modulation(c_l))               # a1: SiLU + Linear(128->768) once per fold
        STATS["a1_mod_uses"] += 1
    else:
        mod = self.adaln_modulation(c_l)
    if mod.dim() == 2:
        mod = mod.unsqueeze(1)
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.chunk(6, dim=-1)
    raw = ctx is not None and ctx.get("hoist") and not torch.is_grad_enabled() and self.__dict__.get("_katom_raw_adaln", False)
    if raw:                                                                    # a1: `1 + scale` is step-invariant too (same add kernel, once per fold)
        C = _common()
        ka = (id(self), "one_plus_scale_a", tuple(scale_a.shape), scale_a.dtype, str(scale_a.device))
        kf = (id(self), "one_plus_scale_f", tuple(scale_f.shape), scale_f.dtype, str(scale_f.device))
        ops_a = _static(ka, lambda: 1 + scale_a)
        ops_f = _static(kf, lambda: 1 + scale_f)
        attn_input = F.rms_norm(x, (x.shape[-1],)) * ops_a + shift_a            # _rms_adaln_raw's statements with the hoisted factor
    else:
        attn_input = self._rms_adaln(x, scale_a, shift_a)
    attn_out = self.attn(attn_input, attention_params)
    x = self._gated_residual(x, gate_a, attn_out)

    if raw:
        ffn_input = F.rms_norm(x, (x.shape[-1],)) * ops_f + shift_f
    else:
        ffn_input = self._rms_adaln(x, scale_f, shift_f)
    ffn_out = self.ffn(ffn_input)
    x = self._gated_residual(x, gate_f, ffn_out)
    return x


# =====================================================================================================================
# ESMFold2AtomEncoder.forward (diffusion module's, structure_prediction=True) — stock statements (:825-925) + the per-fold context
# =====================================================================================================================
def _is_prefix(indices):
    """indices (sorted, unique: torch.nonzero's output) == arange(n) iff its last element is n-1 (n distinct sorted ints in [0, n-1])."""
    n = int(indices.numel())
    if n == 0:
        return False
    return int(indices[-1].item()) == n - 1                                  # one host read at the fold's step 0 (which already synchronises three times)


def _fold_context(enc, attention_params, new_fold):
    """Build the per-call context the blocks read: hoist flag, prefix length, rope tables. Host decisions are taken at step 0 only and kept in
    STATE['fold'] (python values, not in the inference cache: ef2_opt asserts non-tensor cache values never change)."""
    fold = STATE.setdefault("fold", {})
    if new_fold:
        fold.clear()
        fold["epoch"] = _FOLD["n"]
        if _CFG["prefix_varlen"] and len(attention_params) > 2 and torch.is_tensor(attention_params[2]):
            idx = attention_params[2]
            fold["prefix_n"] = int(idx.numel()) if _is_prefix(idx) else None
            STATS["a2_prefix_folds" if fold["prefix_n"] is not None else "a2_nonprefix_folds"] += 1
        else:
            fold["prefix_n"] = None
    ctx = {"hoist": bool(_CFG["hoist_adaln"]), "prefix_n": fold.get("prefix_n")}
    if _CFG["rope_tables"]:
        cos, sin = attention_params[0], attention_params[1]
        kc = (id(enc), "cos_rep", tuple(cos.shape), cos.dtype, str(cos.device))
        ks = (id(enc), "sin_signed", tuple(sin.shape), sin.dtype, str(sin.device))
        ctx["rope"] = (_static(kc, lambda: cos.unsqueeze(2).repeat(1, 1, 1, 2)), _static(ks, lambda: torch.cat((-sin, sin), dim=-1).unsqueeze(2)))
    else:
        ctx["rope"] = None
    return ctx


def _encoder_forward(self, ref_pos, atom_attention_mask, ref_space_uid, ref_charge, ref_element, ref_atom_name_chars, atom_to_token,
                     r_l=None, pred_r1=None, s_i=None, z_ij=None, num_diffusion_samples=1, return_intermediates=False, inference_cache=None):
    C = _common()
    if inference_cache is None or torch.is_grad_enabled():                   # no per-fold cache (input embedder / training): stock forward, no context
        STATS["enc_stock_calls"] += 1
        return self._katom_eager_forward(ref_pos, atom_attention_mask, ref_space_uid, ref_charge, ref_element, ref_atom_name_chars, atom_to_token,
                                         r_l=r_l, pred_r1=pred_r1, s_i=s_i, z_ij=z_ij, num_diffusion_samples=num_diffusion_samples,
                                         return_intermediates=return_intermediates, inference_cache=inference_cache)
    B, N = ref_pos.shape[:2]
    layer_cache = inference_cache.setdefault("atomencoder", {})
    new_fold = len(layer_cache) == 0
    if new_fold:                                                              # step 0 of a sample() call: the stock cache fill (:851-885), verbatim
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ef2_atom: empty atom-encoder inference cache during CUDA-graph capture (step 0 must run eagerly)")
        _FOLD["n"] += 1; STATS["folds"] += 1
        atom_feats = torch.cat([ref_pos, ref_charge.unsqueeze(-1), atom_attention_mask.unsqueeze(-1), ref_element,
                                ref_atom_name_chars.reshape(B, N, C.MAX_CHARS * C.CHAR_VOCAB_SIZE)], dim=-1)
        c_base = self.atom_norm(self.atom_linear(atom_feats))
        cos, sin = self.atom_transformer._build_3d_rope(ref_pos, ref_space_uid)
        cos = cos.repeat_interleave(num_diffusion_samples, 0)
        sin = sin.repeat_interleave(num_diffusion_samples, 0)
        mask_exp = atom_attention_mask.repeat_interleave(num_diffusion_samples, 0)
        seqlens = mask_exp.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(mask_exp.flatten(), as_tuple=False).flatten()
        max_seqlen = int(seqlens.max().item())
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        attention_params = (cos, sin, indices, cu_seqlens, max_seqlen)
        n_tokens = int(atom_to_token.max().item()) + 1
        layer_cache["c_base"] = c_base
        layer_cache["attention_params"] = attention_params
        layer_cache["mask_exp"] = mask_exp
        layer_cache["n_tokens"] = n_tokens
        layer_cache["atom_to_token_exp"] = atom_to_token.repeat_interleave(num_diffusion_samples, 0)
    else:
        c_base = layer_cache["c_base"]
        attention_params = layer_cache["attention_params"]
        mask_exp = layer_cache["mask_exp"]
        n_tokens = layer_cache["n_tokens"]

    ctx = _fold_context(self, attention_params, new_fold)
    if _CFG["fused_block"] and not return_intermediates:
        atom_to_token_exp = layer_cache["atom_to_token_exp"]
        res = _encoder_fused(self, layer_cache, c_base, attention_params, mask_exp, n_tokens, atom_to_token_exp, r_l, pred_r1, num_diffusion_samples, new_fold)
        if res is not None:
            a, q, c, fs = res
            STATE["dec_link"] = {"c_ptr": c.data_ptr(), "c_shape": tuple(c.shape), "ctx": ctx, "atom_to_token_ptr": atom_to_token.data_ptr(),
                                 "atom_to_token_exp": atom_to_token_exp, "nds": num_diffusion_samples, "fs": fs}
            return a, q, c, attention_params, []
        STATS["a5_layout_fallbacks"] += 1
    one = (num_diffusion_samples == 1) and _CFG["skip_copies"]
    c = c_base
    q = c
    if self.structure_prediction and r_l is not None:
        q = q if one else q.repeat_interleave(num_diffusion_samples, 0)          # a4
        if pred_r1 is None:
            pred_r1 = torch.zeros_like(r_l)
        r_input = torch.cat([r_l, pred_r1], dim=-1)
        r_to_q = self.coords_linear(r_input)
        q = q + r_to_q
    c = c if one else c.repeat_interleave(num_diffusion_samples, 0)              # a4
    if one:
        STATS["a4_copies_skipped"] += 2

    _CTX.cur = ctx
    try:
        result = self.atom_transformer(q_l=q, c_l=c, attention_params=attention_params, return_intermediates=return_intermediates)
    finally:
        _CTX.cur = None
    if return_intermediates:
        q, intermediates = result
    else:
        q = result
        intermediates = []

    q_to_a = F.relu(self.atom_to_token_linear(q))
    atom_to_token_exp = layer_cache["atom_to_token_exp"] if "atom_to_token_exp" in layer_cache else atom_to_token.repeat_interleave(num_diffusion_samples, 0)
    a = C.scatter_atom_to_token(q_to_a, atom_to_token_exp, n_tokens, atom_mask=mask_exp.bool())
    # hand the decoder of this step what it needs to run the same levers (it receives no inference cache): keyed by the c tensor it will be given
    STATE["dec_link"] = {"c_ptr": c.data_ptr(), "c_shape": tuple(c.shape), "ctx": ctx, "atom_to_token_ptr": atom_to_token.data_ptr(),
                         "atom_to_token_exp": atom_to_token_exp, "nds": num_diffusion_samples}
    return a, q, c, attention_params, intermediates


# =====================================================================================================================
# ESMFold2AtomDecoder.forward — stock statements (:970-1002); levers through the context the encoder of the same step left
# =====================================================================================================================
def _decoder_forward(self, a_i, q_l, c_l, p_lm, atom_to_token, atom_attention_mask, num_diffusion_samples=1, return_intermediates=False):
    C = _common()
    link = STATE.get("dec_link")
    ok = (link is not None and not torch.is_grad_enabled() and link["c_ptr"] == c_l.data_ptr() and link["c_shape"] == tuple(c_l.shape)
          and link["nds"] == num_diffusion_samples)
    if not ok:
        STATS["dec_stock_calls"] += 1
        return self._katom_eager_forward(a_i, q_l, c_l, p_lm, atom_to_token, atom_attention_mask, num_diffusion_samples=num_diffusion_samples,
                                         return_intermediates=return_intermediates)
    if link.get("fs") is not None and not return_intermediates and link["atom_to_token_ptr"] == atom_to_token.data_ptr():
        r_l = _decoder_fused(self, a_i, q_l, c_l, link["fs"], num_diffusion_samples)     # a5: fused decoder (same fold state as the encoder of this step)
        return r_l, []
    if _CFG["skip_copies"] and link["atom_to_token_ptr"] == atom_to_token.data_ptr():
        atom_to_token_exp = link["atom_to_token_exp"]                             # a4: the encoder's cached expansion (same values)
        STATS["a4_copies_skipped"] += 1
    else:
        atom_to_token_exp = atom_to_token.repeat_interleave(num_diffusion_samples, 0)
    a_to_q = self.token_to_atom_linear(a_i)
    a_to_q = C.gather_token_to_atom(a_to_q, atom_to_token_exp)
    q_l = q_l + a_to_q

    _CTX.cur = link["ctx"]
    try:
        result = self.atom_transformer(q_l=q_l, c_l=c_l, attention_params=p_lm, return_intermediates=return_intermediates)
    finally:
        _CTX.cur = None
    if return_intermediates:
        q_l, intermediates = result
    else:
        q_l = result
        intermediates = []

    r_l = self.output_linear(self.norm(q_l))
    return r_l, intermediates


# =====================================================================================================================
# a5 fused_block / a6 seg_mean — Triton kernels (fast tier). One SWAAtomBlock = 3 kernels:
#   K1 _k1_pre_attn : [prologue: x = base + coords_linear(r) | base + gather(token->atom)] rms-adaLN(shift_a, scale_a) -> QKV GEMM -> per-head
#                     rms-norm(q,k) -> 3D RoPE -> bf16 q|k|v [M,384];  gate = sigmoid(xn @ Wg^T) fp32 [M,128]
#   K2 _k2_swa      : sliding-window (|i-j| <= 64 within a batch row's valid prefix) softmax attention for a 64-query tile x 4 heads (bf16 operands,
#                     fp32 accumulate — the precision class of stock's flash call) -> * gate -> @ Wo^T -> x + gate_a * y   (fp32 residual stream)
#   K3 _k3_ffn      : rms-adaLN(shift_f, scale_f) -> W_up GEMM -> silu(x1)*x2 -> W_down GEMM -> x + gate_f * y  [+ decoder head: LayerNorm -> 128->3]
#   K4 _k4_a2t      : encoder tail relu(q @ W_a2t^T) averaged over each token's (contiguous, sorted) valid atoms — GEMM + relu + segment mean, no atomics
# GEMM precision (sub-lever gemm=): 'ieee' fp32 FMA dots (fp32-faithful, different summation order than cuBLAS), 'tf32x3' split-emulated fp32 on tensor
# cores, 'tf32', 'bf16' (bf16 operands, fp32 accumulate). The attention products are bf16 x bf16 -> fp32 in every setting (as stock: q,k,v are cast
# to bf16 before flash-attn); the attention OUTPUT stays fp32 here (stock rounds it to bf16) — strictly more precise, tolerance class.
# =====================================================================================================================
_TRITON = {"ok": None, "why": ""}


def _triton():
    if _TRITON["ok"] is None:
        try:
            import triton, triton.language as tl  # noqa: F401
            _TRITON["ok"] = True
        except Exception as e:  # noqa: BLE001
            _TRITON["ok"] = False; _TRITON["why"] = f"{type(e).__name__}: {e}"
    return _TRITON["ok"]


if _triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _mm(a, b, PREC: tl.constexpr, LOWP: tl.constexpr):
        if LOWP:
            return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        else:
            return tl.dot(a, b, input_precision=PREC)

    @triton.jit(do_not_specialize=["M", "N", "L", "S"])
    def _k1_pre_attn(X, BASE, R, PR, WC, T2A, TOK, MOD, W1, COS, SIN, QKV, GATE,
                     M, N, L, S, mod_stride, eps,
                     PRO: tl.constexpr, HAS_PR: tl.constexpr, PREC: tl.constexpr, LOWP: tl.constexpr, BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rm = rows < M
        cols = tl.arange(0, 128)
        c16 = tl.arange(0, 16)
        j64 = tl.arange(0, 64)
        xptr = rows[:, None] * 128 + cols[None, :]
        if PRO == 0:
            x = tl.load(X + xptr, mask=rm[:, None], other=0.0)
        else:
            p = rows % N
            bq = rows // N
            if PRO == 1:
                brow = (bq // S) * N + p                                        # base c is [B, N, 128]; rows run over B*S samples
                x = tl.load(BASE + brow[:, None] * 128 + cols[None, :], mask=rm[:, None], other=0.0)
                for c in tl.static_range(3):
                    r_c = tl.load(R + rows * 3 + c, mask=rm, other=0.0)
                    w_c = tl.load(WC + cols * 6 + c)
                    x += r_c[:, None] * w_c[None, :]
                if HAS_PR:
                    for c in tl.static_range(3):
                        r_c = tl.load(PR + rows * 3 + c, mask=rm, other=0.0)
                        w_c = tl.load(WC + cols * 6 + 3 + c)
                        x += r_c[:, None] * w_c[None, :]
            else:
                x = tl.load(BASE + xptr, mask=rm[:, None], other=0.0)
                t = tl.load(TOK + rows, mask=rm, other=0)
                x += tl.load(T2A + (bq * L + t)[:, None] * 128 + cols[None, :], mask=rm[:, None], other=0.0)
            tl.store(X + xptr, x, mask=rm[:, None])
        ms = tl.sum(x * x, axis=1) / 128
        rstd = 1.0 / tl.sqrt_rn(ms + eps)
        shift = tl.load(MOD + rows[:, None] * mod_stride + cols[None, :], mask=rm[:, None], other=0.0)
        scale = tl.load(MOD + rows[:, None] * mod_stride + 128 + cols[None, :], mask=rm[:, None], other=0.0)
        xn = (x * rstd[:, None]) * (1 + scale) + shift
        cs = tl.load(COS + rows[:, None] * 16 + c16[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        sn = tl.load(SIN + rows[:, None] * 16 + c16[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        csb = tl.reshape(tl.broadcast_to(cs[:, None, :], (BM, 4, 16)), (BM, 64))
        snb = tl.reshape(tl.broadcast_to(sn[:, None, :], (BM, 4, 16)), (BM, 64))
        colo = (j64 // 16) * 32 + (j64 % 16)
        for part in tl.static_range(2):                                         # q, k: two half-GEMMs (rotary halves), rms-norm per head, rope
            w1 = tl.load(W1 + (part * 128 + j64)[:, None] * 128 + cols[None, :])
            w2 = tl.load(W1 + (part * 128 + 64 + j64)[:, None] * 128 + cols[None, :])
            a1 = _mm(xn, tl.trans(w1), PREC, LOWP)
            a2 = _mm(xn, tl.trans(w2), PREC, LOWP)
            s1 = tl.sum(tl.reshape(a1 * a1, (BM, 4, 16)), axis=2)
            s2 = tl.sum(tl.reshape(a2 * a2, (BM, 4, 16)), axis=2)
            rs = 1.0 / tl.sqrt_rn((s1 + s2) / 32 + eps)
            rsb = tl.reshape(tl.broadcast_to(rs[:, :, None], (BM, 4, 16)), (BM, 64))
            a1 = a1 * rsb
            a2 = a2 * rsb
            o1 = a1 * csb - a2 * snb
            o2 = a2 * csb + a1 * snb
            optr = rows[:, None] * 384 + part * 128 + colo[None, :]
            tl.store(QKV + optr, o1.to(tl.bfloat16), mask=rm[:, None])
            tl.store(QKV + optr + 16, o2.to(tl.bfloat16), mask=rm[:, None])
        wv = tl.load(W1 + (256 + cols)[:, None] * 128 + cols[None, :])
        v = _mm(xn, tl.trans(wv), PREC, LOWP)
        tl.store(QKV + rows[:, None] * 384 + 256 + cols[None, :], v.to(tl.bfloat16), mask=rm[:, None])
        wg = tl.load(W1 + (384 + cols)[:, None] * 128 + cols[None, :])
        g = _mm(xn, tl.trans(wg), PREC, LOWP)
        g = 1.0 / (1.0 + tl.exp(-g))
        tl.store(GATE + xptr, g, mask=rm[:, None])

    @triton.jit(do_not_specialize=["N", "n_tiles"])
    def _k2_swa(QKV, GATE, SEQLEN, O, N, n_tiles, scale,
                HW: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, OBF: tl.constexpr):
        """one (query tile, head): window-128 softmax attention off the bf16 q|k|v rows of K1 (bf16 x bf16 -> fp32, online softmax over <= 3 key
        tiles), * gate (sigmoid already applied by K1) -> O[:, h*32:(h+1)*32] (bf16 when the out_proj GEMM of K3 takes bf16 operands, else fp32);
        rows past the batch row's valid prefix write 0."""
        pid = tl.program_id(0)
        h = tl.program_id(1)
        b = pid // n_tiles
        t = pid % n_tiles
        p0 = t * BM
        prow = p0 + tl.arange(0, BM)
        inrow = prow < N
        rows = b * N + prow
        Lb = tl.load(SEQLEN + b)
        qvalid = prow < Lb
        d32 = tl.arange(0, 32)
        lo = tl.maximum(p0 - HW, 0)
        hi = tl.minimum(p0 + BM + HW, Lb)
        q = tl.load(QKV + rows[:, None] * 384 + h * 32 + d32[None, :], mask=inrow[:, None], other=0.0)
        m_i = tl.full((BM,), -1e30, dtype=tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)
        acc = tl.zeros((BM, 32), dtype=tl.float32)
        for j0 in range(lo, hi, BK):
            kpos = j0 + tl.arange(0, BK)
            kin = kpos < hi
            krows = b * N + kpos
            k = tl.load(QKV + krows[:, None] * 384 + 128 + h * 32 + d32[None, :], mask=kin[:, None], other=0.0)
            v = tl.load(QKV + krows[:, None] * 384 + 256 + h * 32 + d32[None, :], mask=kin[:, None], other=0.0)
            sc = tl.dot(q, tl.trans(k)) * scale
            diff = kpos[None, :] - prow[:, None]
            allowed = kin[None, :] & (diff <= HW) & (diff >= -HW)
            sc = tl.where(allowed, sc, -1e30)
            m_new = tl.maximum(m_i, tl.max(sc, axis=1))
            alpha = tl.exp(m_i - m_new)
            pr = tl.exp(sc - m_new[:, None])
            pr = tl.where(allowed, pr, 0.0)
            l_i = l_i * alpha + tl.sum(pr, axis=1)
            acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), v)
            m_i = m_new
        o = acc / tl.maximum(l_i, 1e-30)[:, None]
        g = tl.load(GATE + rows[:, None] * 128 + h * 32 + d32[None, :], mask=inrow[:, None], other=0.0)
        o = tl.where(qvalid[:, None], o * g, 0.0)
        if OBF:
            tl.store(O + rows[:, None] * 128 + h * 32 + d32[None, :], o.to(tl.bfloat16), mask=inrow[:, None])
        else:
            tl.store(O + rows[:, None] * 128 + h * 32 + d32[None, :], o, mask=inrow[:, None])

    @triton.jit(do_not_specialize=["M"])
    def _k3_ffn(X, O, WOT, MOD, WUP, WDNT, XOUT, M, mod_stride, eps, LNW, LNB, WH, ROUT, ln_eps,
                HEAD: tl.constexpr, PREC: tl.constexpr, LOWP: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
        """x2 = x + gate_a * (o @ Wo^T)  [attention out_proj + gated residual];  x3 = x2 + gate_f * W_down(silu*mul(W_up(rms-adaLN_f(x2))))
        [+ HEAD: r = LayerNorm(x3) @ Wh^T, the decoder's output head]."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rm = rows < M
        cols = tl.arange(0, 128)
        rc = tl.arange(0, BC)
        xptr = rows[:, None] * 128 + cols[None, :]
        x = tl.load(X + xptr, mask=rm[:, None], other=0.0)
        o = tl.load(O + xptr, mask=rm[:, None], other=0.0)
        wo = tl.load(WOT + cols[:, None] * 128 + cols[None, :])
        ga = tl.load(MOD + rows[:, None] * mod_stride + 256 + cols[None, :], mask=rm[:, None], other=0.0)
        x = x + ga * _mm(o, wo, PREC, LOWP)
        ms = tl.sum(x * x, axis=1) / 128
        rstd = 1.0 / tl.sqrt_rn(ms + eps)
        shift = tl.load(MOD + rows[:, None] * mod_stride + 384 + cols[None, :], mask=rm[:, None], other=0.0)
        scale = tl.load(MOD + rows[:, None] * mod_stride + 512 + cols[None, :], mask=rm[:, None], other=0.0)
        xn = (x * rstd[:, None]) * (1 + scale) + shift
        y = tl.zeros((BM, 128), dtype=tl.float32)
        for c in tl.static_range(0, 256, BC):
            wa = tl.load(WUP + (c + rc)[:, None] * 128 + cols[None, :])
            wb = tl.load(WUP + (256 + c + rc)[:, None] * 128 + cols[None, :])
            a = _mm(xn, tl.trans(wa), PREC, LOWP)
            bb = _mm(xn, tl.trans(wb), PREC, LOWP)
            hmid = a * (1.0 / (1.0 + tl.exp(-a))) * bb
            wd = tl.load(WDNT + (c + rc)[:, None] * 128 + cols[None, :])
            y += _mm(hmid, wd, PREC, LOWP)
        gf = tl.load(MOD + rows[:, None] * mod_stride + 640 + cols[None, :], mask=rm[:, None], other=0.0)
        xo = x + gf * y
        tl.store(XOUT + xptr, xo, mask=rm[:, None])
        if HEAD:
            mu = tl.sum(xo, axis=1) / 128
            xc = xo - mu[:, None]
            var = tl.sum(xc * xc, axis=1) / 128
            lnw = tl.load(LNW + cols)
            lnb = tl.load(LNB + cols)
            ln = xc * (1.0 / tl.sqrt_rn(var + ln_eps))[:, None] * lnw[None, :] + lnb[None, :]
            for c3 in tl.static_range(3):
                w = tl.load(WH + c3 * 128 + cols)
                r = tl.sum(ln * w[None, :], axis=1)
                tl.store(ROUT + rows * 3 + c3, r, mask=rm)

    @triton.jit(do_not_specialize=["N", "L", "n_ttiles"])
    def _k4_a2t(Q, WA, TSTART, OUT, N, L, n_ttiles,
                DT: tl.constexpr, BT: tl.constexpr, BA: tl.constexpr, BD: tl.constexpr, PREC: tl.constexpr, LOWP: tl.constexpr):
        pid = tl.program_id(0)
        pc = tl.program_id(1)
        b = pid // n_ttiles
        tt = pid % n_ttiles
        t0 = tt * BT
        toks = t0 + tl.arange(0, BT)
        tin = toks < L
        st = tl.load(TSTART + b * (L + 1) + toks, mask=tin, other=0)
        en = tl.load(TSTART + b * (L + 1) + toks + 1, mask=tin, other=0)
        a_lo = tl.load(TSTART + b * (L + 1) + t0)
        a_hi = tl.load(TSTART + b * (L + 1) + tl.minimum(t0 + BT, L))
        cols = tl.arange(0, 128)
        dcols = pc * BD + tl.arange(0, BD)
        w = tl.load(WA + dcols[:, None] * 128 + cols[None, :])                 # [BD, 128] rows of W_a2t [DT, 128]
        acc = tl.zeros((BT, BD), dtype=tl.float32)
        for a0 in range(a_lo, a_hi, BA):
            arow = a0 + tl.arange(0, BA)
            ain = arow < a_hi
            qt = tl.load(Q + (b * N + arow)[:, None] * 128 + cols[None, :], mask=ain[:, None], other=0.0)
            h = _mm(qt, tl.trans(w), PREC, LOWP)                              # [BA, BD]
            h = tl.maximum(h, 0.0)
            sel = (arow[None, :] >= st[:, None]) & (arow[None, :] < en[:, None]) & ain[None, :] & tin[:, None]
            acc += tl.dot(sel.to(tl.float32), h, input_precision="ieee")     # exact 0/1 selection: segment sums
        cnt = (en - st).to(tl.float32)
        res = tl.where(cnt[:, None] > 0, acc / tl.maximum(cnt, 1.0)[:, None], 0.0)
        tl.store(OUT + (b * L + toks)[:, None] * DT + dcols[None, :], res, mask=tin[:, None])


    # ---------------------------------------------------------------------------------------------------------------
    # hybrid path (a5 fused_ew): fp32 cuBLAS GEMMs stay (F.linear, the stock kernels); everything between them is fused
    # ---------------------------------------------------------------------------------------------------------------
    @triton.jit(do_not_specialize=["M", "N", "L", "S"])
    def _e1_adaln(X, XN, BASE, R, PR, WC, T2A, TOK, XP, YP, MODG, goff, MODS, soff, scoff,
                  M, N, L, S, mod_stride, eps,
                  PRO: tl.constexpr, HAS_PR: tl.constexpr, BM: tl.constexpr):
        """x = [X | c+coords_linear(r) | base+gather(t2a) | xp + gate*yp] ; store x (PRO != 0) ; XN = rms_norm(x)*(1+scale)+shift."""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rm = rows < M
        cols = tl.arange(0, 128)
        xptr = rows[:, None] * 128 + cols[None, :]
        if PRO == 0:
            x = tl.load(X + xptr, mask=rm[:, None], other=0.0)
        elif PRO == 1:
            p = rows % N
            bq = rows // N
            brow = (bq // S) * N + p
            x = tl.load(BASE + brow[:, None] * 128 + cols[None, :], mask=rm[:, None], other=0.0)
            acc = tl.zeros((BM, 128), dtype=tl.float32)
            for c in tl.static_range(3):
                r_c = tl.load(R + rows * 3 + c, mask=rm, other=0.0)
                w_c = tl.load(WC + cols * 6 + c)
                acc += r_c[:, None] * w_c[None, :]
            if HAS_PR:
                for c in tl.static_range(3):
                    r_c = tl.load(PR + rows * 3 + c, mask=rm, other=0.0)
                    w_c = tl.load(WC + cols * 6 + 3 + c)
                    acc += r_c[:, None] * w_c[None, :]
            x = x + acc
            tl.store(X + xptr, x, mask=rm[:, None])
        elif PRO == 2:
            bq = rows // N
            x = tl.load(BASE + xptr, mask=rm[:, None], other=0.0)
            t = tl.load(TOK + rows, mask=rm, other=0)
            x = x + tl.load(T2A + (bq * L + t)[:, None] * 128 + cols[None, :], mask=rm[:, None], other=0.0)
            tl.store(X + xptr, x, mask=rm[:, None])
        else:
            xp = tl.load(XP + xptr, mask=rm[:, None], other=0.0)
            yp = tl.load(YP + xptr, mask=rm[:, None], other=0.0)
            g = tl.load(MODG + rows[:, None] * mod_stride + goff + cols[None, :], mask=rm[:, None], other=0.0)
            x = xp + g * yp
            tl.store(X + xptr, x, mask=rm[:, None])
        ms = tl.sum(x * x, axis=1) / 128
        rstd = 1.0 / tl.sqrt_rn(ms + eps)
        shift = tl.load(MODS + rows[:, None] * mod_stride + soff + cols[None, :], mask=rm[:, None], other=0.0)
        scale = tl.load(MODS + rows[:, None] * mod_stride + scoff + cols[None, :], mask=rm[:, None], other=0.0)
        xn = (x * rstd[:, None]) * (1 + scale) + shift
        tl.store(XN + xptr, xn, mask=rm[:, None])

    @triton.jit
    def _rope_norm_half(a1, a2, cs, sn, eps):
        """per-head rms-norm over the 32 = 16 + 16 channels, then rotary on the (d, d+16) pairs; returns bf16 halves."""
        ssum = tl.sum(a1 * a1, axis=1) + tl.sum(a2 * a2, axis=1)
        rs = 1.0 / tl.sqrt_rn(ssum / 32 + eps)
        a1 = a1 * rs[:, None]
        a2 = a2 * rs[:, None]
        o1 = a1 * cs - a2 * sn
        o2 = a2 * cs + a1 * sn
        return o1.to(tl.bfloat16), o2.to(tl.bfloat16)

    @triton.jit(do_not_specialize=["N", "n_tiles"])
    def _e2_attn(QKVG, COS, SIN, SEQLEN, O, N, n_tiles, scale, eps,
                 HW: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
        """one (query tile, head) straight off the fp32 [M,512] QKV|gate GEMM output: qk rms-norm + 3D RoPE + bf16 cast on load (query tile once,
        key tiles per visit), bf16 x bf16 -> fp32 products (stock's precision class), online softmax, * sigmoid(gate) -> O fp32 [M, h*32:(h+1)*32];
        rows outside the batch row's valid prefix write 0 (pad_input's rows)."""
        pid = tl.program_id(0)
        h = tl.program_id(1)
        b = pid // n_tiles
        t = pid % n_tiles
        p0 = t * BM
        prow = p0 + tl.arange(0, BM)
        inrow = prow < N
        rows = b * N + prow
        Lb = tl.load(SEQLEN + b)
        qvalid = prow < Lb
        c16 = tl.arange(0, 16)
        d32 = tl.arange(0, 32)
        csq = tl.load(COS + rows[:, None] * 16 + c16[None, :], mask=inrow[:, None], other=0.0).to(tl.float32)
        snq = tl.load(SIN + rows[:, None] * 16 + c16[None, :], mask=inrow[:, None], other=0.0).to(tl.float32)
        lo = tl.maximum(p0 - HW, 0)
        hi = tl.minimum(p0 + BM + HW, Lb)
        qa = tl.load(QKVG + rows[:, None] * 512 + h * 32 + c16[None, :], mask=inrow[:, None], other=0.0)
        qb = tl.load(QKVG + rows[:, None] * 512 + h * 32 + 16 + c16[None, :], mask=inrow[:, None], other=0.0)
        q1, q2 = _rope_norm_half(qa, qb, csq, snq, eps)
        m_i = tl.full((BM,), -1e30, dtype=tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)
        acc = tl.zeros((BM, 32), dtype=tl.float32)
        for j0 in range(lo, hi, BK):
            kpos = j0 + tl.arange(0, BK)
            kin = kpos < hi
            krows = b * N + kpos
            ka = tl.load(QKVG + krows[:, None] * 512 + 128 + h * 32 + c16[None, :], mask=kin[:, None], other=0.0)
            kb = tl.load(QKVG + krows[:, None] * 512 + 128 + h * 32 + 16 + c16[None, :], mask=kin[:, None], other=0.0)
            csk = tl.load(COS + krows[:, None] * 16 + c16[None, :], mask=kin[:, None], other=0.0).to(tl.float32)
            snk = tl.load(SIN + krows[:, None] * 16 + c16[None, :], mask=kin[:, None], other=0.0).to(tl.float32)
            k1, k2 = _rope_norm_half(ka, kb, csk, snk, eps)
            v = tl.load(QKVG + krows[:, None] * 512 + 256 + h * 32 + d32[None, :], mask=kin[:, None], other=0.0).to(tl.bfloat16)
            sc = (tl.dot(q1, tl.trans(k1)) + tl.dot(q2, tl.trans(k2))) * scale
            diff = kpos[None, :] - prow[:, None]
            allowed = kin[None, :] & (diff <= HW) & (diff >= -HW)
            sc = tl.where(allowed, sc, -1e30)
            m_new = tl.maximum(m_i, tl.max(sc, axis=1))
            alpha = tl.exp(m_i - m_new)
            pr = tl.exp(sc - m_new[:, None])
            pr = tl.where(allowed, pr, 0.0)
            l_i = l_i * alpha + tl.sum(pr, axis=1)
            acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), v)
            m_i = m_new
        o = acc / tl.maximum(l_i, 1e-30)[:, None]
        g = tl.load(QKVG + rows[:, None] * 512 + 384 + h * 32 + d32[None, :], mask=inrow[:, None], other=0.0)
        o = o * (1.0 / (1.0 + tl.exp(-g)))
        o = tl.where(qvalid[:, None], o, 0.0)
        tl.store(O + rows[:, None] * 128 + h * 32 + d32[None, :], o, mask=inrow[:, None])

    @triton.jit(do_not_specialize=["M"])
    def _e4_swiglu(UP, H, M, BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rm = rows < M
        cols = tl.arange(0, 256)
        a = tl.load(UP + rows[:, None] * 512 + cols[None, :], mask=rm[:, None], other=0.0)
        bb = tl.load(UP + rows[:, None] * 512 + 256 + cols[None, :], mask=rm[:, None], other=0.0)
        h = a * (1.0 / (1.0 + tl.exp(-a))) * bb
        tl.store(H + rows[:, None] * 256 + cols[None, :], h, mask=rm[:, None])

    @triton.jit(do_not_specialize=["M"])
    def _e5_residual(XP, YP, MOD, goff, XOUT, M, mod_stride, LNW, LNB, WH, ROUT, ln_eps,
                     HEAD: tl.constexpr, BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rm = rows < M
        cols = tl.arange(0, 128)
        xptr = rows[:, None] * 128 + cols[None, :]
        xp = tl.load(XP + xptr, mask=rm[:, None], other=0.0)
        yp = tl.load(YP + xptr, mask=rm[:, None], other=0.0)
        g = tl.load(MOD + rows[:, None] * mod_stride + goff + cols[None, :], mask=rm[:, None], other=0.0)
        xo = xp + g * yp
        tl.store(XOUT + xptr, xo, mask=rm[:, None])
        if HEAD:
            mu = tl.sum(xo, axis=1) / 128
            xc = xo - mu[:, None]
            var = tl.sum(xc * xc, axis=1) / 128
            lnw = tl.load(LNW + cols)
            lnb = tl.load(LNB + cols)
            ln = xc * (1.0 / tl.sqrt_rn(var + ln_eps))[:, None] * lnw[None, :] + lnb[None, :]
            for c3 in tl.static_range(3):
                w = tl.load(WH + c3 * 128 + cols)
                r = tl.sum(ln * w[None, :], axis=1)
                tl.store(ROUT + rows * 3 + c3, r, mask=rm)

    @triton.jit(do_not_specialize=["N", "L"])
    def _e6_segmean(HT, TSTART, OUT, N, L, DT: tl.constexpr, BA: tl.constexpr, BD: tl.constexpr):
        """OUT[b, t, cols] = mean over the token's valid atoms a in [tstart[t], tstart[t+1]) of relu(HT[b*N + a, cols]); 0 for atom-less tokens."""
        pid = tl.program_id(0)
        pc = tl.program_id(1)
        b = pid // L
        t = pid % L
        st = tl.load(TSTART + b * (L + 1) + t)
        en = tl.load(TSTART + b * (L + 1) + t + 1)
        dcols = pc * BD + tl.arange(0, BD)
        acc = tl.zeros((BD,), dtype=tl.float32)
        for a0 in range(st, en, BA):
            arow = a0 + tl.arange(0, BA)
            ain = arow < en
            h = tl.load(HT + (b * N + arow)[:, None] * DT + dcols[None, :], mask=ain[:, None], other=0.0)
            acc += tl.sum(tl.maximum(h, 0.0), axis=0)
        cnt = (en - st).to(tl.float32)
        res = tl.where(cnt > 0, acc / tl.maximum(cnt, 1.0), 0.0)
        tl.store(OUT + (b * L + t) * DT + dcols, res)


_K = {"BM": 64, "BK": 64, "BC": 64, "BT": 16, "BA": 32, "BD": 128, "warps1": 8, "warps2": 4, "warps3": 8, "warps4": 4,
      "EBM": 16, "ewarps": 4, "ABM": 64, "ABK": 64, "awarps": 4, "SBA": 8, "SBD": 256, "SMS": 132}


def _sm_count():
    try:
        _K["SMS"] = int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    except Exception:  # noqa: BLE001
        pass


def _hy_weights(blk):
    """fp32 weight views for the hybrid path: W1 = [Wq; Wk; Wv; Wgate] (512,128) for one merged projection GEMM; the rest are the stock Parameters."""
    pk = blk.__dict__.get("_katom_hy")
    if pk is None:
        at = blk.attn
        assert at.n_heads == 4 and at.head_dim == 32, (at.n_heads, at.head_dim)
        W1 = torch.cat([at.Wqkv.weight.detach().float(), at.gate_proj.weight.detach().float()], 0).contiguous()
        pk = {"W1": W1, "Wo": at.out_proj.weight, "Wup": blk.ffn.w_up.weight, "Wdn": blk.ffn.w_down.weight}
        blk.__dict__["_katom_hy"] = pk
    return pk


def _hy_coder(blocks, fs, x0=None, prologue=None, head=None):
    """Run a stack of SWAAtomBlocks on the hybrid kernels. x0: [M,128] fp32 residual input (PRO 0) or None with a prologue (coords | gather).
    Returns (x_out [M,128] fp32, head_out [M,3] | None). Per block: E1(adaln; fused with the previous block's residual) -> GEMM(qkv|gate) ->
    E2(attention) -> GEMM(out_proj) -> E1(residual + adaln_f) -> GEMM(w_up) -> E4(swiglu) -> GEMM(w_down); then E5 (last residual [+ head])."""
    M, N = fs["M"], fs["N"]
    mods = fs["mods"]
    dev = fs["cos"].device
    eps = float(torch.finfo(torch.float32).eps)
    EBM, ew = _K["EBM"], _K["ewarps"]
    grid_e = (triton.cdiv(M, EBM),)
    ABM = 64 if fs["Bp"] * triton.cdiv(N, 64) * 4 >= 2 * _K["SMS"] else 32
    n_tiles = triton.cdiv(N, ABM)
    f32 = dict(dtype=torch.float32, device=dev)
    x = x0
    xp = yp = None; gprev = None                    # pending residual of the previous block: x = xp + mod_prev[:, 640:768] * yp
    for i, blk in enumerate(blocks):
        pk = _hy_weights(blk)
        mod = mods[i]; ms_ = mod.stride(0)
        xn = torch.empty(M, 128, **f32)
        if i == 0:
            if prologue is None:
                _e1_adaln[grid_e](x, xn, x, x, x, x, x, x, x, x, mod, 0, mod, 0, 128, M, N, 1, 1, ms_, eps, PRO=0, HAS_PR=False, BM=EBM, num_warps=ew)
            elif prologue[0] == "coords":
                _, base, r, pr, wc, S = prologue
                x = torch.empty(M, 128, **f32)
                _e1_adaln[grid_e](x, xn, base, r, pr if pr is not None else r, wc, x, x, x, x, mod, 0, mod, 0, 128, M, N, 1, S, ms_, eps,
                                  PRO=1, HAS_PR=pr is not None, BM=EBM, num_warps=ew)
            else:
                _, base, t2a, tok, L = prologue
                x = torch.empty(M, 128, **f32)
                _e1_adaln[grid_e](x, xn, base, x, x, x, t2a, tok, x, x, mod, 0, mod, 0, 128, M, N, L, 1, ms_, eps, PRO=2, HAS_PR=False, BM=EBM, num_warps=ew)
        else:
            x = torch.empty(M, 128, **f32)          # x = xp + gate_f(prev) * yp, then this block's adaln_a
            _e1_adaln[grid_e](x, xn, x, x, x, x, x, x, xp, yp, gprev, 640, mod, 0, 128, M, N, 1, 1, ms_, eps, PRO=3, HAS_PR=False, BM=EBM, num_warps=ew)
        qkvg = F.linear(xn, pk["W1"])                                            # [M, 512] fp32 (cuBLAS)
        o = torch.empty(M, 128, **f32)
        _e2_attn[(fs["Bp"] * n_tiles, 4)](qkvg, fs["cos"], fs["sin"], fs["seqlen"], o, N, n_tiles, float(blk.attn.scale), eps,
                                         HW=int(blk.attn.half_window), BM=ABM, BK=_K["ABK"], num_warps=_K["awarps"])
        y = F.linear(o, pk["Wo"])                                                # [M, 128]  (stock out_proj kernel)
        x2 = torch.empty(M, 128, **f32); xn2 = torch.empty(M, 128, **f32)
        _e1_adaln[grid_e](x2, xn2, x2, x2, x2, x2, x2, x2, x, y, mod, 256, mod, 384, 512, M, N, 1, 1, ms_, eps, PRO=3, HAS_PR=False, BM=EBM, num_warps=ew)
        up = F.linear(xn2, pk["Wup"])                                            # [M, 512]  (stock w_up kernel)
        hmid = torch.empty(M, 256, **f32)
        _e4_swiglu[grid_e](up, hmid, M, BM=EBM, num_warps=ew)
        y2 = F.linear(hmid, pk["Wdn"])                                           # [M, 128]  (stock w_down kernel)
        xp, yp, gprev = x2, y2, mod
    xout = torch.empty(M, 128, **f32)
    if head is None:
        _e5_residual[grid_e](xp, yp, gprev, 640, xout, M, gprev.stride(0), xp, xp, xp, xp, 0.0, HEAD=False, BM=EBM, num_warps=ew)
        return xout, None
    lnw, lnb, wh, ln_eps = head
    rout = torch.empty(M, 3, **f32)
    _e5_residual[grid_e](xp, yp, gprev, 640, xout, M, gprev.stride(0), lnw, lnb, wh, rout, float(ln_eps), HEAD=True, BM=EBM, num_warps=ew)
    return xout, rout


def _prec():
    """(triton input_precision, low-precision operands) of the fused tensor-core GEMMs for the configured gemm= value."""
    g = _CFG["gemm"]
    if g == "bf16":
        return "ieee", True
    if g == "tf32":
        return "tf32", False
    if g == "tf32x3":
        return "tf32x3", False
    raise ValueError(f"ef2_atom: gemm={g!r} has no fused tensor-core path (known: {GEMM_PRECISIONS}; 'fp32' is the cuBLAS hybrid)")


def _pack_block(blk, lowp):
    """Per-block weight packs for the fused kernels (built once; plain attributes, not parameters/buffers)."""
    key = ("bf16" if lowp else "fp32")
    packs = blk.__dict__.setdefault("_katom_packs", {})
    if key in packs:
        return packs[key]
    at = blk.attn
    H, hd = at.n_heads, at.head_dim
    assert H * hd == 128 and hd == 32 and H == 4, (H, hd)
    W = at.Wqkv.weight.detach().float()                                      # [384, 128] rows (3, H, hd)
    idx1 = torch.tensor([h * hd + d for h in range(H) for d in range(hd // 2)], device=W.device)
    idx2 = idx1 + hd // 2
    Wq, Wk, Wv = W[0:128], W[128:256], W[256:384]
    W1 = torch.cat([Wq[idx1], Wq[idx2], Wk[idx1], Wk[idx2], Wv, at.gate_proj.weight.detach().float()], 0).contiguous()   # [512, 128]
    WoT = at.out_proj.weight.detach().float().t().contiguous()              # [128 in, 128 out]
    Wup = blk.ffn.w_up.weight.detach().float().contiguous()                  # [512, 128]
    WdnT = blk.ffn.w_down.weight.detach().float().t().contiguous()           # [256, 128]
    assert tuple(W1.shape) == (512, 128) and tuple(Wup.shape) == (512, 128) and tuple(WdnT.shape) == (256, 128), (W1.shape, Wup.shape, WdnT.shape)
    dt = torch.bfloat16 if lowp else torch.float32
    pk = {"W1": W1.to(dt), "WoT": WoT.to(dt), "Wup": Wup.to(dt), "WdnT": WdnT.to(dt)}
    packs[key] = pk
    return pk


def _row_tile(M):
    """rows per program for the GEMM-carrying kernels: fill the SMs at small M (>= ~132 programs when possible), 64-row tiles at large M."""
    if M >= 64 * _K["SMS"]:
        return 64, 8
    if M >= 32 * _K["SMS"]:
        return 32, 4
    return 16, 4


def _fused_block(blk, x, mod, fs, prologue=None, head=None):
    """x: [M,128] fp32 contiguous (ignored values when prologue builds x); mod: [M, 768] fp32; fs: fold state. Returns (x_out [M,128], head_out|None).
    K1 (rms-adaLN -> q|k|v|gate) ; K2 per (query tile, head) ; K3 (out_proj + residual -> rms-adaLN -> FFN + residual [-> head])."""
    prec, lowp = _prec()
    pk = _pack_block(blk, lowp)
    M = fs["M"]; N = fs["N"]
    dev = mod.device
    eps = float(torch.finfo(torch.float32).eps)
    qkv = torch.empty(M, 384, dtype=torch.bfloat16, device=dev)
    gate = torch.empty(M, 128, dtype=torch.float32, device=dev)
    BM, nw = _row_tile(M)
    grid = (triton.cdiv(M, BM),)
    if prologue is None:
        xin = x
        _k1_pre_attn[grid](xin, xin, xin, xin, xin, xin, xin, mod, pk["W1"], fs["cos"], fs["sin"], qkv, gate, M, N, 1, 1, mod.stride(0), eps,
                           PRO=0, HAS_PR=False, PREC=prec, LOWP=lowp, BM=BM, num_warps=nw)
    elif prologue[0] == "coords":
        _, base, r, pr, wc, S = prologue
        xin = torch.empty(M, 128, dtype=torch.float32, device=dev)
        _k1_pre_attn[grid](xin, base, r, pr if pr is not None else r, wc, xin, xin, mod, pk["W1"], fs["cos"], fs["sin"], qkv, gate, M, N, 1, S, mod.stride(0), eps,
                           PRO=1, HAS_PR=pr is not None, PREC=prec, LOWP=lowp, BM=BM, num_warps=nw)
    else:
        _, base, t2a, tok, L = prologue
        xin = torch.empty(M, 128, dtype=torch.float32, device=dev)
        _k1_pre_attn[grid](xin, base, xin, xin, xin, t2a, tok, mod, pk["W1"], fs["cos"], fs["sin"], qkv, gate, M, N, L, 1, mod.stride(0), eps,
                           PRO=2, HAS_PR=False, PREC=prec, LOWP=lowp, BM=BM, num_warps=nw)
    o = torch.empty(M, 128, dtype=(torch.bfloat16 if lowp else torch.float32), device=dev)
    ABM = 64 if fs["Bp"] * triton.cdiv(N, 64) * 4 >= 2 * _K["SMS"] else 32
    n_tiles = triton.cdiv(N, ABM)
    _k2_swa[(fs["Bp"] * n_tiles, 4)](qkv, gate, fs["seqlen"], o, N, n_tiles, float(blk.attn.scale),
                                       HW=int(blk.attn.half_window), BM=ABM, BK=_K["BK"], OBF=lowp, num_warps=_K["warps2"])
    x3 = torch.empty(M, 128, dtype=torch.float32, device=dev)
    if head is None:
        _k3_ffn[grid](xin, o, pk["WoT"], mod, pk["Wup"], pk["WdnT"], x3, M, mod.stride(0), eps, x3, x3, x3, x3, 0.0,
                      HEAD=False, PREC=prec, LOWP=lowp, BM=BM, BC=_K["BC"], num_warps=nw)
        return x3, None
    lnw, lnb, wh, ln_eps = head
    rout = torch.empty(M, 3, dtype=torch.float32, device=dev)
    _k3_ffn[grid](xin, o, pk["WoT"], mod, pk["Wup"], pk["WdnT"], x3, M, mod.stride(0), eps, lnw, lnb, wh, rout, float(ln_eps),
                  HEAD=True, PREC=prec, LOWP=lowp, BM=BM, BC=_K["BC"], num_warps=nw)
    return x3, rout


def _block_mod(blk, c_flat, hoist):
    """[M, 768] fp32 modulation for a block: hoisted static (a1) or computed now with the stock Sequential."""
    if hoist:
        key = (id(blk), "adaln_mod_flat", tuple(c_flat.shape), c_flat.dtype, str(c_flat.device))
        return _static(key, lambda: blk.adaln_modulation(c_flat).contiguous())
    return blk.adaln_modulation(c_flat).contiguous()


def _fused_fold_state(enc, layer_cache, attention_params, mask_exp, atom_to_token_exp, n_tokens, new_fold):
    """Per-fold device tables for the fused path (static buffers) + the two layout decisions taken once at step 0:
    row_prefix (each batch row's valid atoms are its first seqlen[b] atoms) and tok_sorted (valid atoms' token ids non-decreasing)."""
    # GRAPH SAFETY (HAZARDS 29). The three per-fold device tables the fused kernels read BY ADDRESS inside a captured step graph (seqlen,
    # tstart, tok_flat) come from the static registry (_static) on EVERY call, exactly like cos / sin below; the fold state keeps only the host
    # layout facts and the step-0 SOURCE tensors, which feed _static() and are never handed to a kernel. A whole-generation reset in the MIDDLE of
    # a fold — ef2_opt.clear_graphs -> clear_static() at the step-graph sampler's LRU reset (`sampler_budget`, step 1) or a pair-bias new-key
    # reset — empties the registry AFTER this fold's step 0 filled it; the capture helper's eager warm-up then re-registers the tables from the
    # kept sources (no host read; STATS a5_static_reregistered) and the graph captured next reads registry buffers that the next fold of the
    # signature refreshes in place. Caching the _static() RESULTS here across calls is the hazard (seen once): after a mid-fold reset the graph captured next baked
    # in buffers only this dict referenced, the next fold's fold.clear() freed them, and that graph's first replay on the next same-signature item
    # read freed memory (cusolver gesvd INTERNAL_ERROR -> illegal address at 800 / 1200 tokens, `linalg.svd failed to converge` at 400: the
    # composition sg + af, reached wherever ro steps aside by name).
    fold = STATE.setdefault("fold", {})
    Bp, N = mask_exp.shape
    M = Bp * N
    cos, sin = attention_params[0], attention_params[1]
    ks = lambda tag, t: (id(enc), tag, tuple(t.shape), t.dtype, str(t.device))
    if new_fold or "fused" not in fold:
        m = mask_exp.bool()
        seqlen = m.sum(-1).to(torch.int32).contiguous()
        ar = torch.arange(N, device=m.device)
        row_prefix = bool((m == (ar[None, :] < seqlen[:, None])).all().item())          # host read at step 0 only
        tok = atom_to_token_exp.long()
        tokv = torch.where(m, tok, torch.full_like(tok, n_tokens))
        srt = bool(((tokv[:, 1:] - tokv[:, :-1]) >= 0).all().item()) and row_prefix     # sorted within the valid prefix (dump id n_tokens after it)
        cnt = torch.zeros(Bp, n_tokens + 1, dtype=torch.int64, device=m.device)
        cnt.scatter_add_(1, tokv, torch.ones_like(tokv))
        tstart = torch.cat([torch.zeros(Bp, 1, dtype=torch.int64, device=m.device), cnt[:, :n_tokens].cumsum(1)], 1).to(torch.int32).contiguous()   # [Bp, L+1]
        fold["fused"] = {"row_prefix": row_prefix, "tok_sorted": srt, "Bp": int(Bp), "N": int(N), "M": int(M), "L": int(n_tokens),
                         "_src": (seqlen, tstart, tok.reshape(-1).contiguous())}          # step-0 sources of the registry tables (never read by a graph)
        STATS["a5_rowprefix_folds" if row_prefix else "a5_nonprefix_folds"] += 1
        STATS["a6_sorted_folds" if srt else "a6_unsorted_folds"] += 1
    f = fold["fused"]
    seqlen, tstart, tokf = f["_src"]
    fs = {k: v for k, v in f.items() if k != "_src"}
    if (not new_fold) and ks("seqlen", seqlen) not in _STATIC:                             # the registry was reset since this fold's step 0: re-register (eager warm-up before a capture)
        STATS["a5_static_reregistered"] += 1
    fs["seqlen"] = _static(ks("seqlen", seqlen), lambda: seqlen)
    fs["tstart"] = _static(ks("tstart", tstart), lambda: tstart)
    fs["tok"] = _static(ks("tok_flat", tokf), lambda: tokf)
    fs["cos"] = _static(ks("cos_flat", cos.reshape(M, -1)), lambda: cos.reshape(M, -1).contiguous())
    fs["sin"] = _static(ks("sin_flat", sin.reshape(M, -1)), lambda: sin.reshape(M, -1).contiguous())
    return fs


def _encoder_fused(self, layer_cache, c_base, attention_params, mask_exp, n_tokens, atom_to_token_exp, r_l, pred_r1, num_diffusion_samples, new_fold):
    """The diffusion module's atom encoder on the fused kernels. Returns None when the fold's layout is not the supported class (caller runs unfused)."""
    C = _common()
    fs = _fused_fold_state(self, layer_cache, attention_params, mask_exp, atom_to_token_exp, n_tokens, new_fold)
    if not fs["row_prefix"]:
        return None
    Bp, N, M, L = fs["Bp"], fs["N"], fs["M"], fs["L"]
    S = int(num_diffusion_samples)
    hoist = bool(_CFG["hoist_adaln"])
    c = c_base if (S == 1 and _CFG["skip_copies"]) else c_base.repeat_interleave(S, 0)
    c_flat = c.reshape(M, 128)
    blocks = list(self.atom_transformer.blocks)
    wc = self.__dict__.get("_katom_wc")
    if wc is None and self.structure_prediction:
        wc = self.coords_linear.weight.detach().float().contiguous(); self.__dict__["_katom_wc"] = wc         # [128, 6]
    pro = None
    if self.structure_prediction and r_l is not None:
        pro = ("coords", c_base.reshape(-1, 128).contiguous(), r_l.reshape(M, 3).contiguous().float(),
               None if pred_r1 is None else pred_r1.reshape(M, 3).contiguous().float(), wc, S)   # x = c + coords_linear([r_l, pred_r1]) in the prologue
    if _CFG["tc_gemm"]:
        prec, lowp = _prec()
        x = None
        for i, blk in enumerate(blocks):
            mod = _block_mod(blk, c_flat, hoist)
            if i == 0 and pro is not None:
                x, _ = _fused_block(blk, None, mod, fs, prologue=pro)
            elif i == 0:
                x, _ = _fused_block(blk, c_flat.contiguous().clone(), mod, fs)
            else:
                x, _ = _fused_block(blk, x, mod, fs)
    else:
        prec, lowp = "ieee", False
        fs["mods"] = [_block_mod(blk, c_flat, hoist) for blk in blocks]
        x, _ = _hy_coder(blocks, fs, x0=None if pro is not None else c_flat.contiguous().clone(), prologue=pro)
    q = x.view(Bp, N, 128)
    if _CFG["seg_mean"] and fs["tok_sorted"] and _CFG["tc_gemm"]:
        wa = self.__dict__.get("_katom_wa_" + ("bf16" if lowp else "fp32"))
        if wa is None:
            wa = self.atom_to_token_linear.weight.detach().to(torch.bfloat16 if lowp else torch.float32).contiguous(); self.__dict__["_katom_wa_" + ("bf16" if lowp else "fp32")] = wa
        DT = int(wa.shape[0]); assert DT % _K["BD"] == 0, DT
        a = torch.empty(Bp, L, DT, dtype=torch.float32, device=x.device)
        n_tt = triton.cdiv(L, _K["BT"])
        _k4_a2t[(Bp * n_tt, DT // _K["BD"])](x, wa, fs["tstart"], a, N, L, n_tt, DT=DT, BT=_K["BT"], BA=_K["BA"], BD=_K["BD"], PREC=prec, LOWP=lowp, num_warps=_K["warps4"])
        STATS["a6_calls"] += 1
    elif _CFG["seg_mean"] and fs["tok_sorted"]:
        ht = F.linear(x, self.atom_to_token_linear.weight)                     # [M, DT] fp32 (stock GEMM; relu + mean fused below)
        DT = int(ht.shape[1]); assert DT % _K["SBD"] == 0, DT
        a = torch.empty(Bp, L, DT, dtype=torch.float32, device=x.device)
        _e6_segmean[(Bp * L, DT // _K["SBD"])](ht, fs["tstart"], a, N, L, DT=DT, BA=_K["SBA"], BD=_K["SBD"], num_warps=4)
        STATS["a6_calls"] += 1
    else:
        q_to_a = F.relu(self.atom_to_token_linear(q))
        a = C.scatter_atom_to_token(q_to_a, atom_to_token_exp, n_tokens, atom_mask=mask_exp.bool())
    STATS["a5_enc_calls"] += 1
    return a, q, c, fs


def _decoder_fused(self, a_i, q_l, c_l, fs, num_diffusion_samples):
    Bp, N, M, L = fs["Bp"], fs["N"], fs["M"], fs["L"]
    hoist = bool(_CFG["hoist_adaln"])
    c_flat = c_l.reshape(M, 128)
    if not c_flat.is_contiguous():
        c_flat = c_flat.contiguous()
    t2a = self.token_to_atom_linear(a_i).reshape(-1, 128).contiguous().float()            # [Bp*L, 128] (cuBLAS, stock op)
    blocks = list(self.atom_transformer.blocks)
    x = None; rout = None
    nw = self.__dict__.get("_katom_head")
    if nw is None:
        nw = (self.norm.weight.detach().float().contiguous(), self.norm.bias.detach().float().contiguous(), self.output_linear.weight.detach().float().contiguous(), float(self.norm.eps))
        self.__dict__["_katom_head"] = nw
    pro = ("gather", q_l.reshape(M, 128).contiguous(), t2a, fs["tok"], L)
    if _CFG["tc_gemm"]:
        for i, blk in enumerate(blocks):
            mod = _block_mod(blk, c_flat, hoist)
            head = nw if i == len(blocks) - 1 else None
            if i == 0:
                x, rout = _fused_block(blk, None, mod, fs, prologue=pro, head=head)
            else:
                x, rout = _fused_block(blk, x, mod, fs, head=head)
    else:
        fsd = dict(fs); fsd["mods"] = [_block_mod(blk, c_flat, hoist) for blk in blocks]
        x, rout = _hy_coder(blocks, fsd, x0=None, prologue=pro, head=nw)
    STATS["a5_dec_calls"] += 1
    return rout.view(Bp, N, 3)


# =====================================================================================================================
# install / uninstall / describe
# =====================================================================================================================
def parse_flags(text):
    """'ax,af' | 'a1,a2,a3,a4' | 'hoist_adaln+prefix_varlen' | 'exact' (= a1..a4 = ax) | 'all' -> install kwargs; 'gemm=bf16' sets the GEMM precision."""
    kw = {}
    for tok in str(text).replace("+", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1); kw[k.strip()] = v.strip(); continue
        if tok == "exact":
            kw.update({k: True for k in LEVERS_EXACT}); continue
        if tok == "all":
            kw.update({k: True for k in LEVERS_EXACT + LEVERS_FAST}); continue
        if tok in GROUP_NAMES:
            kw.update({k: True for k in GROUP_NAMES[tok]}); continue
        name = FLAG_NAMES.get(tok, tok)
        if name not in _CFG:
            raise ValueError(f"ef2_atom.parse_flags: unknown lever {tok!r} (known: {sorted(FLAG_NAMES)} / {sorted(_CFG)})")
        kw[name] = True
    return kw


def _patch(m, attr, fn):
    orig = getattr(m, attr)
    _PATCHED.append((m, attr, m.__dict__.get(attr, None)))
    setattr(m, "_katom_eager_" + attr, orig)
    setattr(m, attr, types.MethodType(fn, m))


def _chain_clear_graphs():
    """Tie clear_static() to ef2_opt.clear_graphs (graphs and the static buffers they read are dropped together, never one without the other)."""
    ef2_opt = sys.modules.get("ef2_opt")
    if ef2_opt is None:
        return False
    if getattr(ef2_opt, "_katom_chained", False):
        return True
    orig = ef2_opt.clear_graphs

    def clear_graphs(*a, **k):
        try:
            return orig(*a, **k)
        finally:
            clear_static()
    ef2_opt.clear_graphs = clear_graphs
    ef2_opt._katom_chained = True
    return True


def install(model, hoist_adaln=False, prefix_varlen=False, rope_tables=False, skip_copies=False, fused_block=False, seg_mean=False, gemm="fp32"):
    """Patch the diffusion module's atom encoder / decoder (instance forwards) with the levers asked for. Returns the description (also STATE)."""
    import ef2_srcguard                                                       # the encoder / decoder / block / attention forwards are re-issued: refuse by name on another upstream source
    ef2_srcguard.check_many({"ax": bool(hoist_adaln or prefix_varlen or rope_tables or skip_copies), "af": bool(fused_block or seg_mean)})
    C = _common()
    gemm = str(gemm)
    if gemm not in GEMM_PRECISIONS:
        raise ValueError(f"ef2_atom: gemm={gemm!r} (known: {GEMM_PRECISIONS})")
    if (fused_block or seg_mean) and not _triton():
        raise RuntimeError("ef2_atom: fused_block / seg_mean need triton: " + _TRITON["why"])
    if seg_mean and not fused_block:
        raise ValueError("ef2_atom: seg_mean rides on fused_block (it consumes the fused encoder's output)")
    if gemm != "fp32" and not fused_block:
        raise ValueError(f"ef2_atom: gemm={gemm!r} is the precision of the FUSED blocks (fused_block=True); the unfused levers keep stock's GEMMs")
    tc_gemm = bool(fused_block) and gemm != "fp32"
    _CFG.update(hoist_adaln=bool(hoist_adaln), prefix_varlen=bool(prefix_varlen), rope_tables=bool(rope_tables), skip_copies=bool(skip_copies),
                fused_block=bool(fused_block), seg_mean=bool(seg_mean), tc_gemm=tc_gemm, gemm=gemm)
    if tc_gemm:
        _prec()
    if fused_block:
        _sm_count()
    sh = getattr(model, "structure_head", None)
    dm = getattr(sh, "diffusion_module", None) if sh is not None else None
    if dm is None or not isinstance(getattr(dm, "atom_encoder", None), C.ESMFold2AtomEncoder) or not isinstance(getattr(dm, "atom_decoder", None), C.ESMFold2AtomDecoder):
        raise RuntimeError("ef2_atom.install: model.structure_head.diffusion_module.{atom_encoder, atom_decoder} not found (the atom path surface moved)")
    if STATE.get("installed"):
        uninstall(model)
    try:
        import ef2_opt  # noqa: F401  (so clear_graphs is chained even when this install runs before ef2_opt's)
    except Exception:
        pass
    mods = []
    _patch(dm.atom_encoder, "forward", _encoder_forward); mods.append("atom_encoder")
    _patch(dm.atom_decoder, "forward", _decoder_forward); mods.append("atom_decoder")
    nb = 0
    for coder in (dm.atom_encoder, dm.atom_decoder):
        for blk in coder.atom_transformer.blocks:
            assert isinstance(blk, C.SWAAtomBlock) and isinstance(blk.attn, C.SWA3DRoPEAttention)
            _patch(blk, "forward", _block_forward)
            blk.__dict__["_katom_raw_adaln"] = getattr(blk, "_rms_adaln", None) is C._rms_adaln_raw   # hoist `1 + scale` only around the raw statements
            _patch(blk.attn, "forward", _attn_forward)          # on top of ef2_opt's U1 forward if present (kept as _katom_eager_forward for the dense fallback)
            nb += 1
    chained = _chain_clear_graphs()
    STATE.update(installed=True, levers={k: _CFG[k] for k in LEVERS_EXACT + LEVERS_FAST}, gemm=_CFG["gemm"], tc_gemm=_CFG["tc_gemm"], modules=mods, blocks=nb,
                 flash_attn=bool(C.FLASH_ATTN_AVAILABLE), clear_chained=chained, version=VERSION)
    STATE.pop("fold", None); STATE.pop("dec_link", None)
    return describe()


def uninstall(model=None):
    for m, attr, orig in reversed(_PATCHED):
        if orig is None:
            m.__dict__.pop(attr, None)
        else:
            m.__dict__[attr] = orig
        m.__dict__.pop("_katom_eager_" + attr, None)
    _PATCHED.clear()
    clear_static()
    STATE.update(installed=False, levers={}, modules=[])
    return True


def groups_on():
    """The kit's lever names whose member levers are ALL on: {'ax': bool, 'af': bool} (STATE['levers'] holds the members)."""
    lev = STATE.get("levers") or {}
    return {g: bool(STATE.get("installed")) and all(lev.get(k) for k in members) for g, members in GROUP_NAMES.items()}


def describe():
    on = [k for k in LEVERS_EXACT + LEVERS_FAST if _CFG.get(k)]
    return f"{VERSION} levers={','.join(on) or 'none'} gemm={_CFG['gemm']}({'fused tensor-core blocks' if _CFG.get('tc_gemm') else ('cuBLAS fp32 + fused elementwise' if _CFG.get('fused_block') else 'stock')}) blocks={STATE.get('blocks', 0)} flash_attn={STATE.get('flash_attn')} clear_chained={STATE.get('clear_chained')}"


def stats():
    return dict(STATS)
