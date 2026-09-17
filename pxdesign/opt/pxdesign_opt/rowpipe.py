"""The package's ROWPIPE lever — `registry.PACKAGE_LEVERS["rowpipe"]`, applied by the package-lever hook (`stack.classify` → `sizeceil.apply`)
after the kit's `install(model)`, under the `big` mode and nowhere else.

  rowpipe   the hoist kit's `prepare_cache` (`opt/forward/hoist/pxd_xattempt/hoist.py:63-182`, once per `sample_diffusion` call) evaluates three
            pair-plane passes on the whole `[N_tok, N_tok, c]` fp32 plane — H1 the DiffusionConditioning pair path
            `Linear(LN(cat(z_trunk, relpe(features)))) + transition_z1 + transition_z2` (hoist.py:71-82), H2 the 16 token pair biases
            `permute(Linear(LN(pair_z)))` (hoist.py:91-97), and the AtomAttentionEncoder pair term `Linear(LN(pair_z)) → [N, N, 16]`
            (hoist.py:130) — and keeps `C["pair_z"]` (512 B × N_tok²) resident through the sampling although the per-step path reads only
            `pair_z.shape[-2]` (hoist.py:357). At N_tok 5288 the H1 line alone asks for > 60 GiB (cat [N,N,256] fp32 + its LayerNorm copy +
            the Linear output). This lever evaluates the same module calls on row slabs `[i0:i1, :, :]` of at most `slab_rows(N_tok)` token rows (`ROWS`; `GEMM_ROWS`
            by the device's compute capability, `GEMM_ROWS_BY_CC`) × all columns
            and writes each slab's H2 / encoder-term rows into preallocated outputs; a slab's relative-position one-hot features are built
            from the integer token features for rows `[i0, i1)` with the comparisons, clips, one-hot widths and feature order of
            `protenix/model/modules/embedders.py` `RelativePositionEncoding.forward`, then that module's own `linear_no_bias`. No
            `[N_tok, N_tok, c_z]` tensor exists at any time; `C["pair_z"]` is a `[CARRIER_ROWS, 0]` fp32 shape carrier (0 bytes; its row
            count is the one thing the per-step path reads, see `CARRIER_ROWS`). Resident after
            prepare: `tok_bias` 16 × `[1, 16, N, N]` fp32 (1 KiB × N_tok², what the token attention reads at every step) and the kit's
            atom-scale caches; transient at prepare: the encoder pair term `[1, N, N, 16]` fp32 (64 B × N_tok², freed once gathered into
            the local atom-pair layout) + one slab's intermediates (≈ 2.6 KiB × ROWS × N_tok). Every other line of `prepare_cache` (H1s,
            the atom-scale H3 part, H4, the AdaLN terms) is the kit's, verbatim. Class: fp32 reassociation — LayerNorm and Linear are
            row-independent, so a slab's rows see the plane's algebra; cuBLAS may select GEMM kernels by problem shape (tier 2, the
            `big` line's class; `tests/test_rowpipe.py` compares slab against plane on the stock modules from `stock/src`). Defined for `PXD_HOIST_MODE=rows` (the value `big` exports): a shape-mode prepare
            (`n_sample_shape` > 1, the `exact` line's N_sample-expanded evaluation) is refused by name — never slabbed silently.

No cross-call state: the kit builds its cache inside one `sample_diffusion` call (`hoist._make_sample_diffusion_wrapper`: created before,
cleared in `finally`) and keys nothing by `id()`; this module adds no cache of its own, so two feature dicts in one process never
share conditioning (`tests/test_rowpipe.py`).

`apply_rowpipe(hoist_mod)` rebinds `pxd_xattempt.hoist.prepare_cache` process-wide (the kit's `_resolve_cache` resolves that
name through the module's globals at call time), is idempotent, and returns the hook's evidence fields.
"""
import sys
import time
from typing import Any, Dict

TAG = "rowpipe"
ROWS = 256                                                   # token rows per slab at most: peak slab transient ≈ 2.6 KiB × ROWS × N_tok (0.7 GiB at N_tok 5788)
GEMM_ROWS = 32768                                            # token-pair rows (slab rows × N_tok = the M of every slab GEMM) per slab at most. cuBLAS selects fp32 GEMM kernels by
                                                             # problem shape; on the pinned H100 stack (compute capability 9.0) slab GEMMs of M ≤ 37 950 return the rows of
                                                             # stock's plane GEMMs (M = N_sample·N_tok²) bit for bit and slab GEMMs of M ≥ 70 400 do not (a different fp32
                                                             # summation order — measured at N_tok 275: slabs of 55 / 128 / 137 / 138 rows identical to stock, 256 / 275 rows
                                                             # not). The cap keeps every slab at or below that size; the lever's class stays tier 2 (kernel selection is the
                                                             # library's). A device whose capability is not in GEMM_ROWS_BY_CC runs this value.
GEMM_ROWS_BY_CC = {"9.0": GEMM_ROWS, "8.0": 65536}           # the cap by the CUDA device's compute capability (`gemm_rows_cap`). On the A100 (8.0, same stack) the library's
                                                             # selection runs the other way: slab GEMMs of M ≥ 20 800 return stock's plane rows bit for bit at every N_sample
                                                             # (1 / 5 / 10 / 16) and every N_tok measured (191–700; slabs up to the whole plane, M 490 000), slab GEMMs of
                                                             # M ≤ 20 437 do not (the 16 pair biases and the encoder term at every such M; the pair path in bands; the
                                                             # relative-position Linear below M ≈ 5 000). With the balanced schedule of `slab_rows`, 65536 keeps every slab
                                                             # of a 145–1452-token plane at M ≥ 20 800 (minimum 21 025, the 145-token plane itself; two slabs only from
                                                             # 257 tokens, M ≥ 32 896), where 32768 cuts 182–256-token planes into two slabs of M 16 562–32 768 (182–202
                                                             # tokens: a slab at M ≤ 20 437); wider planes keep slabs of ≤ 65536 // N_tok rows (10 at 5988 tokens) and
                                                             # their last, shorter slab may fall under 20 800. A plane under 145 tokens is one slab = the plane (h1's own
                                                             # shape; under the switch for h2 / the encoder term whenever N_sample·N_tok² is above it).
CARRIER_ROWS = 1                                             # C["pair_z"] = an empty [CARRIER_ROWS, 0] tensor. The kit's `_resolve_cache` (hoist.py:357) builds the
#   token transformer's `z` argument as `pair_z.new_zeros((pair_z.shape[-2], 1))`, and stock `DiffusionTransformer.forward` (transformer.py:447)
#   reads `z.shape[-2] > 2000` to call `torch.cuda.empty_cache()` between its 16 blocks — upstream's guard on reserved memory when the
#   [N_sample, 16, N, N] fp32 attention logits are large. With no post-prepare segments left for the allocator to reuse (this lever's point),
#   each per-block clear hands the logits' segments back to the driver and the next block re-allocates them through cudaMalloc: +24 % forward
#   at 4788 tokens on H100, while peak *allocated* memory — what decides the ceiling — is unchanged by a clear. A 1-row carrier keeps
#   the per-block clears off (forward at 4788 tokens then 6 % under `big` without this lever); the atom transformer's own clear
#   (transformer.py:928, N_atom > 20000, twice per step) and DiffusionConditioning's (diffusion.py:155, once per prepare) are untouched.
HOIST_MODULE = "pxd_xattempt.hoist"
RELPE_FEATURES = ("asym_id", "residue_index", "entity_id", "sym_id", "token_index")   # RelativePositionEncoding.input_feature
_STATE: Dict[str, Any] = {"prepares": 0, "last": {}}


# ------------------------------------------------------------------------------------------------------------- slab rows
def capability(device=None):
    """The compute capability "major.minor" of `device` (a torch.device) when it is a CUDA device (torch.cuda.get_device_capability); None for a
    CPU device or none given."""
    if device is None or getattr(device, "type", None) != "cuda":
        return None
    import torch
    cc = torch.cuda.get_device_capability(device)
    return "%d.%d" % (int(cc[0]), int(cc[1]))


def gemm_rows_cap(cc=None, device=None) -> int:
    """The GEMM_ROWS cap of a compute capability: `cc` as (major, minor) or "major.minor"; else the capability of `device` (`capability`).
    No capability (a CPU device, none given) or one outside GEMM_ROWS_BY_CC → GEMM_ROWS."""
    if cc is None:
        cc = capability(device)
    if cc is None:
        return GEMM_ROWS
    key = cc if isinstance(cc, str) else "%d.%d" % (int(cc[0]), int(cc[1]))
    return GEMM_ROWS_BY_CC.get(key, GEMM_ROWS)


def gemm_rows_words(cc) -> str:
    """`gemm_rows=<cap> cc=<major.minor|none>` plus ` (not in GEMM_ROWS_BY_CC: GEMM_ROWS)` when the capability has no row of its own — the
    prepare line's tail."""
    tabled = cc is None or cc in GEMM_ROWS_BY_CC
    return f"gemm_rows={gemm_rows_cap(cc=cc)} cc={cc or 'none'}" + ("" if tabled else " (not in GEMM_ROWS_BY_CC: GEMM_ROWS)")


def slab_rows(n_token: int, rows=None, gemm_rows=None) -> int:
    """Token rows per slab for an `n_token` plane: the cap is `min(ROWS, gemm_rows // n_token)` (at least 1; `gemm_rows` defaults to GEMM_ROWS,
    prepare_cache passes the device's `gemm_rows_cap`); the plane is cut into `n_slabs = ceil(n_token / cap)` slabs of
    `ceil(n_token / n_slabs)` rows each (≤ cap), the last slab taking the rows that remain — it may be smaller (1788 tokens under 32768: 99 slabs
    of 18 rows and one of 6). Both caps hold for every slab."""
    rows = ROWS if rows is None else rows
    gemm_rows = GEMM_ROWS if gemm_rows is None else gemm_rows
    n = max(1, int(n_token))
    cap = max(1, min(int(rows), int(gemm_rows) // n, n))
    n_slabs = -(-n // cap)
    return -(-n // n_slabs)


# ------------------------------------------------------------------------------------------------------------- pair rows (H1 slab)
def relpe_rows(relpe, feats, i0: int, i1: int):
    """`RelativePositionEncoding.forward` (embedders.py:136-240, the relative position encoding) for query rows [i0, i1) × all key columns: the same
    comparisons, clips, one-hot widths and feature order (a_rel_pos 66, a_rel_token 66, b_same_entity 1, a_rel_chain 6 → 139), then the
    module's own `linear_no_bias`. Returns [..., i1−i0, N_tok, c_z]."""
    import torch
    import torch.nn.functional as F
    asym, res, ent, sym, tok = (feats[k] for k in RELPE_FEATURES)
    r = slice(i0, i1)
    b_same_chain = (asym[..., r, None] == asym[..., None, :]).long()
    b_same_residue = (res[..., r, None] == res[..., None, :]).long()
    b_same_entity = (ent[..., r, None] == ent[..., None, :]).long()
    d_residue = torch.clip(input=res[..., r, None] - res[..., None, :] + relpe.r_max, min=0, max=2 * relpe.r_max) * b_same_chain \
        + (1 - b_same_chain) * (2 * relpe.r_max + 1)
    a_rel_pos = F.one_hot(d_residue, 2 * (relpe.r_max + 1))
    d_token = torch.clip(input=tok[..., r, None] - tok[..., None, :] + relpe.r_max, min=0, max=2 * relpe.r_max) * b_same_chain * b_same_residue \
        + (1 - b_same_chain * b_same_residue) * (2 * relpe.r_max + 1)
    a_rel_token = F.one_hot(d_token, 2 * (relpe.r_max + 1))
    d_chain = torch.clip(input=sym[..., r, None] - sym[..., None, :] + relpe.s_max, min=0, max=2 * relpe.s_max) * b_same_entity \
        + (1 - b_same_entity) * (2 * relpe.s_max + 1)
    a_rel_chain = F.one_hot(d_chain, 2 * (relpe.s_max + 1))
    x = torch.cat([a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain], dim=-1).float()
    return relpe.linear_no_bias(x)


def pair_rows(cond, feats, z_trunk, i0: int, i1: int, inplace_safe: bool, use_conditioning: bool = True):
    """`DiffusionConditioning.forward` pair lines (H1, diffusion.py:123-133) for rows [i0, i1): [..., i1−i0, N_tok, c_z]; dtype as stock (the cat
    promotes to fp32)."""
    import torch
    zt = z_trunk[..., i0:i1, :, :]
    if not use_conditioning:
        zt = 0 * zt
    pair_z = torch.cat(tensors=[zt, relpe_rows(cond.relpe, feats, i0, i1)], dim=-1)
    pair_z = cond.linear_no_bias_z(cond.layernorm_z(pair_z))
    if inplace_safe:
        pair_z += cond.transition_z1(pair_z)
        pair_z += cond.transition_z2(pair_z)
    else:
        pair_z = pair_z + cond.transition_z1(pair_z)
        pair_z = pair_z + cond.transition_z2(pair_z)
    return pair_z


def pair_derived(dm, input_feature_dict, z_trunk, inplace_safe: bool, use_conditioning: bool, rows: int):
    """H1 → H2 + the encoder pair term, in row slabs of `rows`. Returns (tok_bias: n_blocks × [..., 1, H, N, N], z16: [..., 1, N, N, c_atompair],
    carrier: [CARRIER_ROWS, 0] fp32, n_slabs)."""
    import torch
    from protenix.model.utils import expand_at_dim, permute_final_dims
    cond = dm.diffusion_conditioning
    enc = dm.atom_attention_encoder
    blocks = list(dm.diffusion_transformer.blocks)
    N = int(z_trunk.shape[-2])
    R = max(1, min(int(rows), N))
    tb = None
    z16 = None
    n_slabs = 0
    for i0 in range(0, N, R):
        i1 = min(N, i0 + R)
        zs = pair_rows(cond, input_feature_dict, z_trunk, i0, i1, inplace_safe, use_conditioning)   # H1 rows i0:i1  [..., r, N, c_z]
        z_e = expand_at_dim(zs, dim=-4, n=1)                                                        # stock f_forward: z_pair = expand_at_dim(pair_z, -4, N_sample); rows mode: 1
        z32 = z_e.to(dtype=torch.float32)                                                         # [..., 1, r, N, c_z]
        for j, blk in enumerate(blocks):                                                            # H2 rows i0:i1 of every block: permute(Linear(LN(z)), [2,0,1]) → [..., 1, H, r, N]
            apb = blk.attention_pair_bias
            b = permute_final_dims(apb.linear_nobias_z(apb.layernorm_z(z32)), [2, 0, 1])
            if tb is None:
                tb = [b.new_empty((*b.shape[:-2], N, N)) for _ in blocks]                           # [..., 1, H, N, N] fp32, contiguous (== the kit's _ns_slice(NS=1) layout)
            tb[j][..., i0:i1, :] = b
            del b
        e = enc.linear_no_bias_z(enc.layernorm_z(z_e))                                             # encoder pair term rows i0:i1: [..., 1, r, N, c_atompair]
        if z16 is None:
            z16 = e.new_empty((*e.shape[:-3], N, N, e.shape[-1]))
        z16[..., i0:i1, :, :] = e
        del zs, z_e, z32, e
        n_slabs += 1
    carrier = z_trunk.new_empty((CARRIER_ROWS, 0), dtype=torch.float32)                           # C["pair_z"]: hoist._resolve_cache reads .shape[-2] only (→ _z_dummy [CARRIER_ROWS, 1])
    return tb, z16, carrier, n_slabs


# ------------------------------------------------------------------------------------------------- prepare_cache (the kit's, slabbed)
def prepare_cache(dm, input_feature_dict, s_inputs, s_trunk, z_trunk, inplace_safe, use_conditioning=True, n_sample_shape=0):
    """`pxd_xattempt.hoist.prepare_cache` with H1 / H2 / the encoder pair term evaluated in row slabs (`pair_derived`) and no resident pair_z.
    Spans that differ from hoist.py: l.71-82 and l.90-100 → `pair_derived`; l.130-134 gathers the slab-built z16 and frees it; l.181; l.184.
    Every other line is the kit's."""
    if n_sample_shape not in (0, 1, None):
        raise RuntimeError(f"[{TAG}] the row-slab prepare is defined for PXD_HOIST_MODE=rows (un-expanded rows); a shape-exact prepare on "
                           f"N_sample={n_sample_shape} expanded rows (PXD_HOIST_MODE=shape, the exact line) was asked — refused")
    import torch
    import torch.nn.functional as F
    from protenix.model.utils import expand_at_dim, broadcast_token_to_atom, permute_final_dims
    from protenix.model.modules.primitives import rearrange_qk_to_dense_trunk, broadcast_token_to_local_atom_pair
    H = sys.modules[HOIST_MODULE]
    _ns_slice, HSTATE = H._ns_slice, H._STATE
    t0 = time.time()
    cond = dm.diffusion_conditioning
    C: Dict[str, Any] = {}
    # ---- H1 pair path + H2 token pair biases + the encoder pair term: row slabs; nothing [N_tok, N_tok, c_z]-shaped is materialised
    if not use_conditioning:
        s_trunk = 0 * s_trunk
    CC = capability(z_trunk.device); G = gemm_rows_cap(cc=CC)
    R = slab_rows(int(z_trunk.shape[-2]), gemm_rows=G)
    tb, z16, carrier, n_slabs = pair_derived(dm, input_feature_dict, z_trunk, inplace_safe, use_conditioning, R)
    C["pair_z"] = carrier                                           # [CARRIER_ROWS, 0] shape carrier (module doc, CARRIER_ROWS)
    # ---- H1s: single path base (before the per-step noise embedding is added)
    single_s = torch.cat(tensors=[s_trunk, s_inputs], dim=-1)
    C["single_base"] = cond.linear_no_bias_s(cond.layernorm_s(single_s))   # [..., N_tok, c_s]
    NS = 1                                                          # rows mode: un-expanded rows, one cache for every chunk N_sample
    C["n_sample_shape"] = NS
    C["tok_bias"] = tb                                              # n_blocks × [..., 1, H, N_tok, N_tok]
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
    n_token = s1.size(-2)
    c_l = c_l.unsqueeze(dim=-3) + broadcast_token_to_atom(x_token=enc.linear_no_bias_s(enc.layernorm_s(s1)), atom_to_token_idx=atom_to_token_idx)  # [...,1,N_atom,c_atom]
    p_lm = (p_lm.unsqueeze(dim=-5) + broadcast_token_to_local_atom_pair(z_token=z16,                                    # z16 == enc.linear_no_bias_z(enc.layernorm_z(z)), slab-built
            atom_to_token_idx=atom_to_token_idx, n_queries=enc.n_queries, n_keys=enc.n_keys, compute_mask=False)[0])          # [...,1,nb,nq,nk,c_atompair]
    del z16                                                          # the [1, N, N, c_atompair] plane is not kept past the gather
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
    del c_l_full, p_lm_full
    if torch.cuda.is_available(): torch.cuda.synchronize()
    HSTATE["stats"]["prepares"] += 1; HSTATE["stats"]["prepare_s"].append(round(time.time() - t0, 4))
    _nu = HSTATE.pop("_nonuniform_this_call", [])
    N = int(z_trunk.shape[-2])
    _STATE["prepares"] = int(_STATE["prepares"]) + 1
    _STATE["last"] = {"N_token": N, "rows": R, "n_slabs": n_slabs, "gemm_rows": G, "cc": CC, "prepare_s": round(time.time() - t0, 4),
                      "tok_bias_bytes": int(sum(t.numel() * t.element_size() for t in tb)), "pair_z_resident_bytes": 0}
    print(f"[pxd_hoist v1.2.2+rowpipe] prepare_cache NS={NS} rows={R} slabs={n_slabs} N_tok={N}: non-uniform N_sample row-blocks kept full = "   # the tag spelled out: the line as printed, byte for byte
          f"{sorted(set(_nu)) or 'none'} ({len(_nu)} tensors); pair_z not resident; {gemm_rows_words(CC)}", flush=True)
    return C


# ------------------------------------------------------------------------------------------------------------------------- apply
def apply_rowpipe(hoist_mod=None) -> dict:
    """Rebind `pxd_xattempt.hoist.prepare_cache` to this module's `prepare_cache`. Returns the evidence fields: `prepare_cache_rebound`
    (the module attribute is this module's function) and `resolved_by_global` (the kit's `_resolve_cache` reads the name from its globals)."""
    H = hoist_mod or sys.modules.get(HOIST_MODULE)
    if H is None:
        import importlib
        H = importlib.import_module(HOIST_MODULE)
    if getattr(H.prepare_cache, "_pxdesign_opt_lever", None) != "rowpipe":
        H._pxdesign_opt_stock_prepare_cache = H.prepare_cache
        prepare_cache._pxdesign_opt_lever = "rowpipe"
        H.prepare_cache = prepare_cache
    return {"prepare_cache_rebound": getattr(H.prepare_cache, "_pxdesign_opt_lever", None) == "rowpipe",
            "resolved_by_global": "prepare_cache" in getattr(getattr(H, "_resolve_cache", None), "__code__", prepare_cache.__code__).co_names}


def stats() -> dict:
    return {"prepares": _STATE["prepares"], "last": dict(_STATE["last"]), "rows": ROWS, "gemm_rows_default": GEMM_ROWS, "gemm_rows_by_cc": dict(GEMM_ROWS_BY_CC)}
