"""pxd_xattempt.hoist — step-invariant hoisting for the PXDesign-d denoiser (Protenix v0.5.0+pxd DiffusionModule).

WHAT IT DOES (scheduling only; same model, same weights, same inputs, same 400 steps, same RNG draws in the same order):
within one `sample_diffusion` call the conditioning (input_feature_dict, s_inputs, s_trunk, z_trunk) is constant across
all steps and chunks. Stock recomputes, at EVERY denoiser call, sub-graphs that depend only on that conditioning:
  H1  DiffusionConditioning pair path: pair_z = LN+Linear(cat(z_trunk, relpe)) + transition_z1 + transition_z2   [N_tok^2 x 128]
  H1s DiffusionConditioning single path base: Linear(LN(cat(s_trunk, s_inputs)))                                 [N_tok x 384]
  H2  per DiffusionTransformer block (16 blocks x 16 heads): pair bias = Linear(LN(z_pair)) -> [16, N_tok, N_tok]
  H3  AtomAttentionEncoder step-invariant part: c_l, p_lm (incl. small MLP), dense-trunk geometry                 [N_atom-scale]
  H4  per AtomTransformer block (4 encoder + 4 decoder blocks x 4 heads): local pair bias = Linear(LN(p_lm))
  H5  _local_attention padding-mask bias (attn_bias=None branch: new_zeros + -inf fills + concat_split/unfold), 8 atom blocks/call   [N_atom-scale]
       (PXD_HOIST_MASK=0 disables). AtomAttentionDecoder: nothing beyond H4/H5 (its q path depends on a).
This module computes H1-H4 ONCE per sample_diffusion call, BY CALLING THE STOCK SUB-MODULES on the stock inputs (identical
kernels, identical shapes -> designed to be bitwise identical per element), and replays them at every step through thin patched forwards. Per-step dynamic work
(noise embedding, transitions on single_s, atom transformer on q, token transformer on a, decoder) is the untouched stock code.

Kernel arguments of the hoisted ops (PXD_HOIST_MODE): stock runs LN/Linear on z-derived tensors expanded over the chunk's N_sample,
i.e. on [N_sample*N_tok^2, c] rows that are N_sample identical copies. `shape` (the default) evaluates the hoisted ops on those expanded
rows like stock and keeps row-block 0 alone only when every row-block is torch.equal to it (the bitwise target; _shape_exact); `rows`
evaluates them once on the un-expanded [N_tok^2, c] rows and broadcasts (cheaper prepare; cuBLAS may tile another M differently: last-ulp moves).

USAGE:  import pxd_xattempt.hoist as H; H.install(model)   # model = runner.model (ProtenixDesign); idempotent; H.uninstall(model) restores stock
ENV:    PXD_HOIST=0 disables install(); PXD_HOIST_MODE=shape|rows (default shape = bitwise target); PXD_HOIST_MASK=0 leaves H5 to stock.
"""
from __future__ import annotations
import os, time, types, functools
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F

_STATE: Dict[str, Any] = {"installed": False, "cache": None, "stats": {"prepares": 0, "hits": {}, "prepare_s": []}}


def _hit(name):
    h = _STATE["stats"]["hits"]; h[name] = h.get(name, 0) + 1


# --------------------------------------------------------------------------------------------------------------------------
# cache preparation (runs the STOCK sub-modules once)
# --------------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def _ns_slice(x, dim, tag):
    """'shape' mode evaluates hoisted ops on N_sample-expanded rows (the stock shapes). The N_sample row-blocks of such an evaluation are
    identical at most shapes but not at all of them (cuBLAS may tile a larger M differently), so row-block [0:1] is kept only if every row-block
    is torch.equal to row-block 0; otherwise the full N_sample tensor is kept (== what stock recomputes at every step). Consumers are unchanged
    (expand() is then a no-op); a kept-full tensor is counted under stats()["nonuniform_kept"] and named on the prepare line."""
    n = x.shape[dim]
    if n > 1:
        x0 = x.narrow(dim, 0, 1)
        for r in range(1, n):
            if not torch.equal(x.narrow(dim, r, 1), x0):
                st = _STATE["stats"].setdefault("nonuniform_kept", {}); st[tag] = st.get(tag, 0) + 1
                _STATE["_nonuniform_this_call"] = _STATE.get("_nonuniform_this_call", []) + [tag]
                return x.contiguous()
        return x0.contiguous()
    return x.contiguous()


def prepare_cache(dm, input_feature_dict, s_inputs, s_trunk, z_trunk, inplace_safe, use_conditioning=True, n_sample_shape=0):
    """dm = DiffusionModule. Returns dict of step-invariant tensors. Mirrors stock f_forward / DiffusionConditioning.forward /
    AtomAttentionEncoder.forward line by line for the parts that do not depend on (x_noisy, t_hat)."""
    from protenix.model.utils import expand_at_dim, broadcast_token_to_atom, permute_final_dims
    from protenix.model.modules.primitives import rearrange_qk_to_dense_trunk, broadcast_token_to_local_atom_pair
    t0 = time.time()
    cond = dm.diffusion_conditioning
    C: Dict[str, Any] = {}
    # ---- H1: pair path of DiffusionConditioning (stock lines, verbatim)
    if not use_conditioning:
        s_trunk = 0 * s_trunk; z_trunk = 0 * z_trunk
    pair_z = torch.cat(tensors=[z_trunk, cond.relpe(input_feature_dict)], dim=-1)
    pair_z = cond.linear_no_bias_z(cond.layernorm_z(pair_z))
    if inplace_safe:
        pair_z += cond.transition_z1(pair_z)
        pair_z += cond.transition_z2(pair_z)
    else:
        pair_z = pair_z + cond.transition_z1(pair_z)
        pair_z = pair_z + cond.transition_z2(pair_z)
    C["pair_z"] = pair_z                                            # [..., N_tok, N_tok, c_z]
    # ---- H1s: single path base (before the per-step noise embedding is added)
    single_s = torch.cat(tensors=[s_trunk, s_inputs], dim=-1)
    C["single_base"] = cond.linear_no_bias_s(cond.layernorm_s(single_s))   # [..., N_tok, c_s]
    # ---- what f_forward does next: z_pair = expand_at_dim(pair_z, -4, N_sample); the token transformer gets z_pair.to(fp32)
    NS = int(n_sample_shape) if n_sample_shape else 1               # shape-exact mode: evaluate on N_sample-expanded rows like stock, cache slice [0:1]
    C["n_sample_shape"] = NS
    z_e = expand_at_dim(pair_z, dim=-4, n=NS)                         # stock f_forward: z_pair = expand_at_dim(z_pair, -4, N_sample)
    z32 = z_e.to(dtype=torch.float32)                               # [..., NS, N_tok, N_tok, c_z]  (a view; LN's .contiguous() materialises like stock)
    # ---- H2: token-level DiffusionTransformer pair biases, one per block: bias = permute(Linear(LN(z)), [2,0,1]) -> [..., H, N, N]
    tb = []
    for blk in dm.diffusion_transformer.blocks:
        apb = blk.attention_pair_bias
        b = apb.linear_nobias_z(apb.layernorm_z(z32))
        tb.append(_ns_slice(permute_final_dims(b, [2, 0, 1]), -4, "tok_bias"))   # [..., 1|NS, H, N_tok, N_tok]  (v1.2.2)
    C["tok_bias"] = tb
    # ---- H3: AtomAttentionEncoder invariant part (stock lines, verbatim, with s = s_trunk expanded later)
    enc = dm.atom_attention_encoder
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    batch_shape = input_feature_dict["ref_pos"].shape[:-2]
    N_atom = input_feature_dict["ref_pos"].shape[-2]
    c_l = enc.linear_no_bias_ref_pos(input_feature_dict["ref_pos"]) + enc.linear_no_bias_ref_charge(
        torch.arcsinh(input_feature_dict["ref_charge"]).reshape(*batch_shape, N_atom, 1))
    if inplace_safe:
        c_l += enc.linear_no_bias_f(torch.cat([input_feature_dict[name].reshape(*batch_shape, N_atom, enc.input_feature[name]) for name in enc.input_feature], dim=-1).to(dtype=c_l.dtype))
        c_l *= input_feature_dict["ref_mask"].reshape(*batch_shape, N_atom, 1)
    else:
        c_l = c_l + enc.linear_no_bias_f(torch.cat([input_feature_dict[name].reshape(*batch_shape, N_atom, enc.input_feature[name]) for name in enc.input_feature], dim=-1).to(dtype=c_l.dtype))
        c_l = c_l * input_feature_dict["ref_mask"].reshape(*batch_shape, N_atom, 1)
    q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
        q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
        k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
        dim_q=[-2, -1], dim_k=[-2, -1], n_queries=enc.n_queries, n_keys=enc.n_keys, compute_mask=True)
    d_lm = q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
    v_lm = (q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()).unsqueeze(dim=-1)
    p_lm = (enc.linear_no_bias_d(d_lm) * v_lm) * pad_info["mask_trunked"].unsqueeze(dim=-1)
    if inplace_safe:
        p_lm += enc.linear_no_bias_invd(1 / (1 + (d_lm ** 2).sum(dim=-1, keepdim=True))) * v_lm
        p_lm += enc.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype))
    else:
        p_lm = p_lm + enc.linear_no_bias_invd(1 / (1 + (d_lm ** 2).sum(dim=-1, keepdim=True))) * v_lm
        p_lm = p_lm + enc.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype))
    # trunk part (r_l is not None in the sampler): stock uses s = expand_at_dim(s_trunk,-3,N_sample), z = expand_at_dim(pair_z,-4,N_sample)
    # here computed with N_sample = 1 (identical rows); broadcasting happens at use.
    s1 = expand_at_dim(s_trunk, dim=-3, n=NS)                        # stock: s = expand_at_dim(s_trunk, -3, N_sample)
    z1 = z_e                                                         # stock: z = expand_at_dim(z_pair, -4, N_sample)
    n_token = s1.size(-2)
    c_l = c_l.unsqueeze(dim=-3) + broadcast_token_to_atom(x_token=enc.linear_no_bias_s(enc.layernorm_s(s1)), atom_to_token_idx=atom_to_token_idx)  # [...,1,N_atom,c_atom]
    p_lm = (p_lm.unsqueeze(dim=-5) + broadcast_token_to_local_atom_pair(z_token=enc.linear_no_bias_z(enc.layernorm_z(z1)),
            atom_to_token_idx=atom_to_token_idx, n_queries=enc.n_queries, n_keys=enc.n_keys, compute_mask=False)[0])          # [...,1,nb,nq,nk,c_atompair]
    c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(q=c_l, k=c_l, dim_q=-2, dim_k=-2, n_queries=enc.n_queries, n_keys=enc.n_keys, compute_mask=False)
    if inplace_safe:
        p_lm += enc.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
        p_lm += enc.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
        p_lm += enc.small_mlp(p_lm)
    else:
        p_lm = (p_lm + enc.linear_no_bias_cl(F.relu(c_l_q[..., None, :])) + enc.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :])))
        p_lm = p_lm + enc.small_mlp(p_lm)
    c_l_full, p_lm_full = c_l, p_lm                                   # [..., NS, ...] (identical copies along NS by construction)
    c_l = _ns_slice(c_l, -3, "enc_c_l"); p_lm = _ns_slice(p_lm, -5, "enc_p_lm")          # (v1.2.2)
    C["enc_c_l"] = c_l            # [..., 1, N_atom, c_atom]
    C["enc_p_lm"] = p_lm          # [..., 1, n_blocks, n_q, n_k, c_atompair]
    C["n_token"] = n_token
    # ---- H4: AtomTransformer local pair biases (encoder + decoder use the SAME p_lm: decoder receives p_skip = p_lm)
    def atom_biases(atom_transformer, p):
        out = []
        for blk in atom_transformer.diffusion_transformer.blocks:
            apb = blk.attention_pair_bias
            b = apb.linear_nobias_z(apb.layernorm_z(p))                       # [..., 1, nb, nq, nk, H]
            out.append(permute_final_dims(b, [3, 0, 1, 2]).contiguous())      # [..., 1, H, nb, nq, nk]
        return out
    C["enc_bias"] = [_ns_slice(b, -5, "enc_bias") for b in atom_biases(enc.atom_transformer, p_lm_full)]          # (v1.2.2)
    C["dec_bias"] = [_ns_slice(b, -5, "dec_bias") for b in atom_biases(dm.atom_attention_decoder.atom_transformer, p_lm_full)]          # (v1.2.2)
    # AdaLN conditioning of the atom transformers on c (= c_l, step-invariant): layernorm_s(c), sigmoid(linear_s(.)), linear_nobias_s(.), and
    # the output gate sigmoid(linear_a_last(c)) and transition gate sigmoid(linear_s(c)) are ALSO step-invariant. Hoisted per block:
    def atom_adaln(atom_transformer, c):
        out = []
        for blk in atom_transformer.diffusion_transformer.blocks:
            apb = blk.attention_pair_bias; ctb = blk.conditioned_transition_block
            e = {}
            ln_a = apb.layernorm_a                 # AdaptiveLayerNorm(a, s): sigmoid(linear_s(LN_s(s))) * LN_a(a) + linear_nobias_s(LN_s(s))
            s_n = ln_a.layernorm_s(c)
            e["attn_gate_s"] = torch.sigmoid(ln_a.linear_s(s_n)); e["attn_bias_s"] = ln_a.linear_nobias_s(s_n)
            if apb.cross_attention_mode:           # atom blocks: kv = layernorm_kv(a, s) — a SECOND AdaLN with its own weights (stock transformer.py AttentionPairBias.forward)
                ln_kv = apb.layernorm_kv
                s_k = ln_kv.layernorm_s(c)
                e["kv_gate_s"] = torch.sigmoid(ln_kv.linear_s(s_k)); e["kv_bias_s"] = ln_kv.linear_nobias_s(s_k)
            e["attn_out_gate"] = torch.sigmoid(apb.linear_a_last(c))          # output projection gate (adaLN-Zero)
            s_n2 = ctb.adaln.layernorm_s(c)
            e["tr_gate_s"] = torch.sigmoid(ctb.adaln.linear_s(s_n2)); e["tr_bias_s"] = ctb.adaln.linear_nobias_s(s_n2)
            e["tr_out_gate"] = torch.sigmoid(ctb.linear_s(c))
            out.append(e)
        return out
    def _slice0(d): return {k: _ns_slice(v, -3, "adaln." + k) for k, v in d.items()}          # (v1.2.2)
    C["enc_adaln"] = [_slice0(e) for e in atom_adaln(enc.atom_transformer, c_l_full)]
    C["dec_adaln"] = [_slice0(e) for e in atom_adaln(dm.atom_attention_decoder.atom_transformer, c_l_full)]
    del c_l_full, p_lm_full, z_e, z32
    if torch.cuda.is_available(): torch.cuda.synchronize()
    _STATE["stats"]["prepares"] += 1; _STATE["stats"]["prepare_s"].append(round(time.time() - t0, 4))
    _nu = _STATE.pop("_nonuniform_this_call", []); print(f"[pxd_hoist v1.2.2] prepare_cache NS={NS}: non-uniform N_sample row-blocks kept full = {sorted(set(_nu)) or 'none'} ({len(_nu)} tensors)", flush=True)
    return C


# --------------------------------------------------------------------------------------------------------------------------
# patched forwards (bound while a cache is active; fall through to stock otherwise)
# --------------------------------------------------------------------------------------------------------------------------
def _f_forward_hoisted(self, r_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, inplace_safe=False,
                       chunk_size=None, use_conditioning=True):
    C = _STATE["cache"]
    if C is None:
        return self._pxd_stock_f_forward(r_noisy=r_noisy, t_hat_noise_level=t_hat_noise_level, input_feature_dict=input_feature_dict,
                                         s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, inplace_safe=inplace_safe,
                                         chunk_size=chunk_size, use_conditioning=use_conditioning)
    from protenix.model.utils import expand_at_dim, broadcast_token_to_atom, aggregate_atom_to_token
    _hit("f_forward")
    N_sample = r_noisy.size(-3)
    C = _resolve_cache(C, N_sample)
    _STATE["cache_active"] = C
    cond = self.diffusion_conditioning
    # ---- single conditioning: per-step part only (stock lines); optional N_sample dedup (fuse.sdedup: all t_hat of the chunk are equal in the sampler)
    s_single = None
    try:
        from pxd_xattempt import fuse as _fuse
        if _fuse.state()["sdedup"]:
            s_single = _fuse._cond_forward_single_dedup(cond, t_hat_noise_level, s_inputs, s_trunk, inplace_safe, single_base=C["single_base"])   # [...,1,N_tok,c_s] or None
            if s_single is not None: _hit("sdedup")
    except ImportError:
        pass
    if s_single is None:
        single_s = C["single_base"]
        noise_n = cond.fourier_embedding(t_hat_noise_level=torch.log(input=t_hat_noise_level / cond.sigma_data) / 4).to(single_s.dtype)
        single_s = single_s.unsqueeze(dim=-3) + cond.linear_no_bias_n(cond.layernorm_n(noise_n)).unsqueeze(dim=-2)
        if inplace_safe:
            single_s += cond.transition_s1(single_s)
            single_s += cond.transition_s2(single_s)
        else:
            single_s = single_s + cond.transition_s1(single_s)
            single_s = single_s + cond.transition_s2(single_s)
        s_single = single_s
    # ---- atom attention encoder: per-step part only
    enc = self.atom_attention_encoder
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    c_l = C["enc_c_l"]                                      # [..., 1, N_atom, c_atom]
    q_l = c_l + enc.linear_no_bias_r(r_noisy)               # broadcast over N_sample -> [..., N_sample, N_atom, c_atom]  (stock: c_l expanded + linear(r))
    _STATE["atom_ctx"] = ("enc", N_sample)
    q_l = enc.atom_transformer(q_l, c_l, C["enc_p_lm"], chunk_size=chunk_size)   # c (=c_l) and p (=p_lm) carry a singleton N_sample dim; the patched blocks read cached AdaLN/bias terms
    _STATE["atom_ctx"] = None
    a_token = aggregate_atom_to_token(x_atom=F.relu(enc.linear_no_bias_q(q_l)), atom_to_token_idx=atom_to_token_idx, n_token=C["n_token"], reduce="mean")   # stock line (protenix scatter mean)
    q_skip, c_skip, p_skip = q_l, c_l, C["enc_p_lm"]
    a_token = a_token.to(dtype=torch.float32)
    if inplace_safe:
        a_token += self.linear_no_bias_s(self.layernorm_s(s_single))
    else:
        a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s_single))
    _STATE["tok_ctx"] = True
    a_token = self.diffusion_transformer(a=a_token.to(dtype=torch.float32), s=s_single.to(dtype=torch.float32), z=C["_z_dummy"],
                                         inplace_safe=inplace_safe, chunk_size=chunk_size)
    _STATE["tok_ctx"] = None
    a_token = self.layernorm_a(a_token)
    # ---- decoder (stock, with hoisted biases inside the atom transformer)
    dec = self.atom_attention_decoder
    q = broadcast_token_to_atom(x_token=dec.linear_no_bias_a(a_token), atom_to_token_idx=atom_to_token_idx) + q_skip
    _STATE["atom_ctx"] = ("dec", N_sample)
    q = dec.atom_transformer(q, c_skip, p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size)
    _STATE["atom_ctx"] = None
    r_update = dec.linear_no_bias_out(dec.layernorm_q(q))
    return r_update




# --------------------------------------------------------------------------------------------------------------------------
# H5: step-invariant padding-mask bias of _local_attention (depends only on N_atom, n_queries, n_keys, inf, dtype)
# --------------------------------------------------------------------------------------------------------------------------
_MASK_CACHE = {}

def _local_attention_hoisted(q, k, v, n_queries, n_keys, attn_bias=None, trunked_attn_bias=None, inf=1e10, use_efficient_implementation=False,
                             inplace_safe=False, chunk_size=None):
    """protenix.model.modules.primitives._local_attention with the padding-mask bias (attn_bias is None branch) taken from a cache.
    Everything else is the stock code path verbatim (same rearrangement, same add, same _attention)."""
    import protenix.model.modules.primitives as PR
    if attn_bias is not None or _STATE["cache"] is None:
        return PR._pxd_stock_local_attention(q, k, v, n_queries, n_keys, attn_bias=attn_bias, trunked_attn_bias=trunked_attn_bias, inf=inf,
                                             use_efficient_implementation=use_efficient_implementation, inplace_safe=inplace_safe, chunk_size=chunk_size)
    _hit("local_attn_mask")
    assert q.shape == k.shape == v.shape
    n, d = q.shape[-2:]
    q_trunked, kv_trunked, padding_info = PR.rearrange_qk_to_dense_trunk(q=q, k=[k, v], dim_q=-2, dim_k=[-2, -2], n_queries=n_queries, n_keys=n_keys, compute_mask=False)
    q_pad_length, pad_left, pad_right = padding_info["q_pad"], padding_info["k_pad_left"], padding_info["k_pad_right"]
    key = (int(n), int(n_queries), int(n_keys), float(inf), q.dtype, q.device, len(q.shape[:-2]))
    attn_bias_trunked = _MASK_CACHE.get(key)
    if attn_bias_trunked is None:
        # stock lines (rearrange_to_dense_trunk, attn_bias is None branch)
        ab = q.new_zeros(*(1,) * len(q.shape[:-2]), n + q_pad_length, n + pad_left + pad_right)
        ab[..., :n, 0:pad_left] = -inf
        ab[..., :n, pad_left + n::] = -inf
        ab[..., n::, :] = -inf
        concat_split_data = PR.optimized_concat_split(ab, n_queries)
        attn_bias_trunked = concat_split_data.unfold(-1, n_keys, ab.shape[-1] + n_queries).transpose(-2, -3)
        _MASK_CACHE[key] = attn_bias_trunked
        if len(_MASK_CACHE) > 64:
            _MASK_CACHE.pop(next(iter(_MASK_CACHE)))
    k_trunked, v_trunked = kv_trunked[0], kv_trunked[1]
    if trunked_attn_bias is not None:
        attn_bias_trunked = attn_bias_trunked + trunked_attn_bias
    if chunk_size is not None:
        from functools import partial
        attn_inputs = {"q": q_trunked, "k": k_trunked, "v": v_trunked, "attn_bias": attn_bias_trunked}
        out = PR.chunk_layer(partial(PR._attention, use_efficient_implementation=use_efficient_implementation, inplace_safe=inplace_safe),
                             attn_inputs, chunk_size=chunk_size, no_batch_dims=len(attn_bias_trunked.shape[:-2]), _out=None)
    else:
        out = PR._attention(q=q_trunked, k=k_trunked, v=v_trunked, attn_bias=attn_bias_trunked, use_efficient_implementation=use_efficient_implementation, inplace_safe=inplace_safe)
    out = out.reshape(*out.shape[:-3], -1, out.shape[-1])
    if q_pad_length > 0:
        out = out[..., :-q_pad_length, :]
    return out


def _flatten_cache(C):
    out = {}
    for k, v in C.items():
        if k.startswith("_") or k in ("n_token", "n_sample_shape"):
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v
        elif isinstance(v, list):
            for i, e in enumerate(v):
                if isinstance(e, torch.Tensor):
                    out[f"{k}[{i}]"] = e
                elif isinstance(e, dict):
                    for kk, vv in e.items():
                        out[f"{k}[{i}].{kk}"] = vv
    return out


def _resolve_cache(C, N_sample):
    """Return the concrete cache for this denoiser call. 'shape' mode keys by the chunk N_sample (stock chunk loop uses at most two
    distinct chunk sizes: diffusion_chunk_size and the remainder); 'rows' mode has one cache for all N_sample."""
    if "_lazy" not in C:
        return C
    L = C["_lazy"]; key = N_sample if L["mode"] == "shape" else 0
    ent = C["_by_ns"].get(key)
    if ent is None:
        ent = prepare_cache(L["dm"], L["input_feature_dict"], L["s_inputs"], L["s_trunk"], L["z_trunk"], L["inplace_safe"],
                            n_sample_shape=(N_sample if L["mode"] == "shape" else 0))
        ent["_z_dummy"] = ent["pair_z"].new_zeros((ent["pair_z"].shape[-2], 1))   # DiffusionTransformer.forward only reads z.shape[-2] (>2000 -> empty_cache between blocks)
        C["_by_ns"][key] = ent
    return ent


def _apb_forward_hoisted(self, a, s, z, n_queries=None, n_keys=None, inplace_safe=False, chunk_size=None):
    """AttentionPairBias.forward with the pair bias (and, for atom blocks, the AdaLN(s) terms + output gate) taken from the cache."""
    ent = getattr(self, "_pxd_cache_slot", None)
    C = _STATE.get("cache_active") if _STATE["cache"] is not None else None
    if C is None or ent is None:
        return self._pxd_stock_forward(a=a, s=s, z=z, n_queries=n_queries, n_keys=n_keys, inplace_safe=inplace_safe, chunk_size=chunk_size)
    kind, idx = ent
    if kind == "tok":
        _hit("apb_tok")
        # a = AdaLN(a, s) with per-step s (stock); bias from cache
        a = self.layernorm_a(a=a, s=s)
        kv = self.layernorm_kv(a=a, s=s) if self.cross_attention_mode else a   # stock order: layernorm_kv sees the normalised a
        bias = C["tok_bias"][idx]                                          # [..., 1, H, N, N] broadcast over N_sample
        q = self.attention(q_x=a, kv_x=kv, attn_bias=_bcast_bias(bias, a), inplace_safe=inplace_safe)
        if inplace_safe:
            q *= torch.sigmoid(self.linear_a_last(s))
        else:
            q = torch.sigmoid(self.linear_a_last(s)) * q
        return q
    else:  # atom encoder/decoder block: s == c_l (step-invariant) -> AdaLN(s) parts and gates cached
        _hit("apb_atom")
        ad = C["enc_adaln" if kind == "enc" else "dec_adaln"][idx]
        bias = C["enc_bias" if kind == "enc" else "dec_bias"][idx]         # [..., 1, H, nb, nq, nk]
        ln = self.layernorm_a
        a_in = a
        a = ad["attn_gate_s"] * ln.layernorm_a(a_in) + ad["attn_bias_s"]   # == AdaptiveLayerNorm.forward(a, s)  (self.layernorm_a(a=a, s=s))
        if self.cross_attention_mode:                                      # stock: kv = self.layernorm_kv(a=a, s=s) applied to the ALREADY-normalised a (stock reassigns a first)
            kv = ad["kv_gate_s"] * self.layernorm_kv.layernorm_a(a) + ad["kv_bias_s"]
        else:
            kv = a
        q = self.attention(q_x=a, kv_x=kv, trunked_attn_bias=_bcast_bias(bias, a, trunked=True), n_queries=n_queries, n_keys=n_keys,
                           inplace_safe=inplace_safe, chunk_size=chunk_size)
        if inplace_safe:
            q *= ad["attn_out_gate"]
        else:
            q = ad["attn_out_gate"] * q
        return q


def _bcast_bias(bias, a, trunked=False):
    """bias has a singleton N_sample dim at -4 (tok: [...,1,H,N,N]) or -5 (atom: [...,1,H,nb,nq,nk]); a is [..., N_sample, N, c].
    Attention asserts bias.shape[:-3] == q.shape[:-2] (= [..., N_sample, H]) so expand the singleton to N_sample (a view, no copy)."""
    ns = a.shape[-3]
    if trunked:
        shp = list(bias.shape); shp[-5] = ns
    else:
        shp = list(bias.shape); shp[-4] = ns
    return bias.expand(*shp)


def _ctb_forward_hoisted(self, a, s):
    """ConditionedTransitionBlock.forward for atom blocks with cached AdaLN(s) terms and output gate; token blocks -> stock."""
    ent = getattr(self, "_pxd_cache_slot", None); C = _STATE.get("cache_active") if _STATE["cache"] is not None else None
    if C is None or ent is None or ent[0] == "tok":
        return self._pxd_stock_forward(a, s)
    _hit("ctb_atom")
    kind, idx = ent
    ad = C["enc_adaln" if kind == "enc" else "dec_adaln"][idx]
    a = ad["tr_gate_s"] * self.adaln.layernorm_a(a) + ad["tr_bias_s"]
    b = F.silu((self.linear_nobias_a1(a))) * self.linear_nobias_a2(a)
    return ad["tr_out_gate"] * self.linear_nobias_b(b)


# --------------------------------------------------------------------------------------------------------------------------
# sample_diffusion wrapper: prepare cache once per call, clear after
# --------------------------------------------------------------------------------------------------------------------------
def _make_sample_diffusion_wrapper(stock_fn):
    @functools.wraps(stock_fn)
    def wrapped(denoise_net, input_feature_dict, s_inputs, s_trunk, z_trunk, **kw):
        dm = denoise_net
        if not getattr(dm, "_pxd_hoist_ready", False):
            return stock_fn(denoise_net=denoise_net, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, **kw)
        inplace_safe = kw.get("inplace_safe", False)
        mode = os.environ.get("PXD_HOIST_MODE", "shape")      # 'shape' = shape-exact (evaluate hoisted ops on N_sample-expanded rows like stock; bitwise target) | 'rows' = un-expanded rows (faster prepare; last-ulp GEMM tiling differences possible)
        C = {"_lazy": dict(dm=dm, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, inplace_safe=inplace_safe, mode=mode), "_by_ns": {}}
        _STATE["cache"] = C
        try:
            return stock_fn(denoise_net=denoise_net, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, **kw)
        finally:
            _STATE["cache"] = None; _STATE["cache_active"] = None
            del C
    wrapped._pxd_hoist_wrapper = True
    return wrapped


def install(model=None):
    """Patch the module classes/instances. `model` = ProtenixDesign (runner.model). Idempotent. Returns stats dict."""
    if os.environ.get("PXD_HOIST", "1") in ("0", "false", "off"):
        return None
    import pxdesign.model.pxdesign as P
    import pxdesign.model.generator as G
    from protenix.model.modules.diffusion import DiffusionModule
    from protenix.model.modules.transformer import AttentionPairBias, ConditionedTransitionBlock
    if not _STATE["installed"]:
        # class-level patches with stock fall-through
        DiffusionModule._pxd_stock_f_forward = DiffusionModule.f_forward
        DiffusionModule.f_forward = _f_forward_hoisted
        AttentionPairBias._pxd_stock_forward = AttentionPairBias.forward
        AttentionPairBias.forward = _apb_forward_hoisted
        ConditionedTransitionBlock._pxd_stock_forward = ConditionedTransitionBlock.forward
        ConditionedTransitionBlock.forward = _ctb_forward_hoisted
        _STATE["stock_sample_diffusion_P"] = P.sample_diffusion
        _STATE["stock_sample_diffusion_G"] = G.sample_diffusion
        P.sample_diffusion = _make_sample_diffusion_wrapper(P.sample_diffusion)   # the name ProtenixDesign.sample_diffusion resolves
        import protenix.model.modules.primitives as PR
        if os.environ.get("PXD_HOIST_MASK", "1") not in ("0", "off") and not hasattr(PR, "_pxd_stock_local_attention"):
            PR._pxd_stock_local_attention = PR._local_attention
            PR._local_attention = _local_attention_hoisted          # Attention.forward resolves the module-global name at call time
        _STATE["installed"] = True
    if model is not None:
        dm = model.diffusion_module if hasattr(model, "diffusion_module") else model
        for i, blk in enumerate(dm.diffusion_transformer.blocks):
            blk.attention_pair_bias._pxd_cache_slot = ("tok", i); blk.conditioned_transition_block._pxd_cache_slot = ("tok", i)
        for i, blk in enumerate(dm.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks):
            blk.attention_pair_bias._pxd_cache_slot = ("enc", i); blk.conditioned_transition_block._pxd_cache_slot = ("enc", i)
        for i, blk in enumerate(dm.atom_attention_decoder.atom_transformer.diffusion_transformer.blocks):
            blk.attention_pair_bias._pxd_cache_slot = ("dec", i); blk.conditioned_transition_block._pxd_cache_slot = ("dec", i)
        dm._pxd_hoist_ready = True
    return _STATE["stats"]


def uninstall(model=None):
    import pxdesign.model.pxdesign as P
    from protenix.model.modules.diffusion import DiffusionModule
    from protenix.model.modules.transformer import AttentionPairBias, ConditionedTransitionBlock
    if _STATE["installed"]:
        DiffusionModule.f_forward = DiffusionModule._pxd_stock_f_forward
        AttentionPairBias.forward = AttentionPairBias._pxd_stock_forward
        ConditionedTransitionBlock.forward = ConditionedTransitionBlock._pxd_stock_forward
        P.sample_diffusion = _STATE["stock_sample_diffusion_P"]
        import protenix.model.modules.primitives as PR
        if hasattr(PR, "_pxd_stock_local_attention"):
            PR._local_attention = PR._pxd_stock_local_attention; del PR._pxd_stock_local_attention
        _MASK_CACHE.clear()
        _STATE["installed"] = False
    if model is not None:
        dm = model.diffusion_module if hasattr(model, "diffusion_module") else model
        dm._pxd_hoist_ready = False
    _STATE["cache"] = None


def stats():
    return _STATE["stats"]
