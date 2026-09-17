"""boltz2_opt.rowpair_heads — seams 16-19 of the tensor-parallel line (``boltz2_opt.rowpair``, ``--n_gpu P > 1``): the three heads that read
the trunk pair tensor — the diffusion conditioning, the diffusion sampler's token transformer and the confidence module — run on this rank's
ROWS of it (``rowpair.TrunkShard``: ``z_loc [1, R, N, 128]``, rows ``[lay.r0, lay.r1)``); no ``N x N x c`` tensor is built on a rank. The
mechanism (row layout, block schedules, every collective, the census) is the core's (``opt_core.mem.rowpair.{diffusion, heads, confidence,
dist, shard}``); what lives here is boltz 2.2.1's STATEMENT of each module over a row block: the stock sub-modules (their weights, LayerNorms,
linears, embeddings) called in the stock order and dtypes with the ``i`` operand of every pair statement sliced to the rows.

``HEADS`` (bound by ``rowpair._bind_heads``; call sites in ``rowpair._forward_sharded``):

  diffusion_conditioning_rows(model, T) -> dict            seam 18  DiffusionConditioning.forward (diffusion_conditioning.py:83-116): the
        conditioned pair ``z_cond`` as ROWS ``[1, R, N, token_z]`` (``diffusion.pair_cond_rows``: LN + linear of ``cat(z rows, rel-pos rows)`` per
        row block — the relative position encoding of the block from ``rowpair._relpos_rows``, never ``[N, N, ·]`` — then the two transitions added
        per block); the atom encoder (encodersv2.py:299-414) with its token-pair term ``z_to_p`` read from a diagonal BAND of the projected rows
        (``diffusion.band_plan / pair_band_rows / band_lookup``: ``q16 = z_to_p_trans(z rows)`` per row block folded into ``[N, 2W+1, atom_z]``, then
        the (query atom, key atom) -> (token, token) lookup; the stock ``einsum`` with the one-hot atom->token maps selects exactly these
        entries); the atom encoder / decoder biases as stock (atom windows: REPLICATED by design). The 24 token-transformer pair biases are NOT
        materialised: ``dict["token_trans_bias"]`` is a :class:`ZCondRows` (the z_cond rows + the per-layer projections), consumed by seam 19.
  sample_sharded(model, T, **sample_kw) -> dict            seam 19  AtomDiffusion.sample (diffusionv2.py:295-420) VERBATIM — schedule, noise,
        random augmentation, the atom attention encoder / decoder (REPLICATED by design: atoms) — with the score model's token
        DiffusionTransformer (transformersv2.py:97-139, 24 x DiffusionTransformerLayer = AdaLN -> AttentionPairBias (attentionv2.py:62-110) ->
        gated residual -> ConditionedTransitionBlock) run by ``diffusion.diffusion_transformer_sharded``: queries = this rank's token rows, keys /
        values from the replicated activation, pair-bias rows ``linear(LN(z_cond rows))`` of layer i per row block (cached for the roll-out when
        the core's budget admits: ``diff_bias_cache``), ONE all-gather of the updated rows per layer. The sampled coordinates are proven identical
        on every rank (``diffusion.sync_replicated``, mode ``guard``: a mismatch refuses by name).
  confidence_rows(model, T, x_pred, multiplicity, run_sequentially) -> dict     seams 16-17  ConfidenceModule.forward (confidencev2.py:109-240)
        one diffusion sample at a time (the stock ``run_sequentially`` recursion; samples concatenated as stock): the confidence pair input BORN
        FROM THE TRUNK ROWS per row block (``heads.embed_rows``: z_norm rows + rel-pos rows + token-bond / contact rows of the host-resident
        feature planes + the ``s_to_z`` outer sum rows + ``dist_bin_pairwise_embed(cdist(x_i rows, x_all))``), its Pairformer presharded
        (``rowpair._presharded_stack``), then ConfidenceHeads (confidencev2.py:275-495): PAE logits per row block (``heads.logit_rows``), PDE logits
        per row block of ``z + z^T`` (the rows of ``z^T`` by ONE shard transpose, ``rowpair._zT_rows``), each block consumed on the fly — the
        ``[N, N, 64]`` logits never exist: ``pae`` / ``pde`` rows ``[R, N]`` -> the ``[N, N]`` matrices on rank 0's HOST only
        (``dist.gather_rows_to_rank0_host``; ranks > 0 return their own rows there, named ``conf_pae_ranks=own_rows``); gPDE / giPDE as row-block
        partial sums all-reduced (``conf_gpde=rowsum``: a reordered fp32 sum, |d| ~ 1e-7, never bitwise); the pTM family (ptm, iptm, ligand_iptm,
        protein_iptm, pair_chains_iptm) as ROW PARTIAL SUMS: boltz's ``compute_ptms`` (confidence_utils.py) weighs every pair with ONE d0
        (``tm_function(pae_value, N_res)`` of the whole padded complex), so each statistic ``max_i sum_j T[i,j] M[i,j] / (sum_j M[i,j] + 1e-5)`` is
        separable per row: the block's ``T rows = sum_b probs * tm_value`` and every mask's rows are the stock statements with the ``i`` operand
        sliced, the per-row numerators / denominators ``[R, 2, 4 + C^2]`` are all-gathered (``shard.unshard_rows``: O(N C^2) floats) and the
        division + ``max`` run on every rank alike — no ``[N, N]`` tensor on any rank (``conf_finish=rowsum``); ``compute_frame_pred`` (the frame
        mask) is called verbatim; plddt / resolved / complex_plddt / complex_iplddt on the replicated ``s`` (the
        interface mask's ``max_j`` per row block of ``cdist``). ``pae_logits`` / ``pde_logits`` (training-loss inputs) are not returned.

REPLICATED BY DESIGN (small or atom-shaped; named in the census): ``s``, ``s_inputs``, the atom tensors ``q c p`` (windows) and the atom
biases, the sampler's coordinates and noise, plddt / resolved logits, the ``[N]`` interface mask. Refused by name: a P == 1 / no-group layout
(every entry: the adapter installs nothing at ``--n_gpu 1``), atom-level confidence heads (``token_level_confidence=False``), a compiled or
re-bound token transformer, a ``diffusion_conditioning`` dict whose ``token_trans_bias`` is not this module's :class:`ZCondRows`.
Dtypes: every statement runs the stock module under the autocast state of its stock call site (the token pair-bias projection of layer i,
stock-computed inside DiffusionConditioning.forward, re-enters that state when the sampler evaluates it: :class:`_Ambient`); nothing is cast
that the stock statement does not cast.

Census (``rowpair._STATE["calls"]``): ``zcond_blocks band_W band_extra_rows dit_layers_sharded dit_transformer_calls conf_samples
conf_embed_rows conf_pae_blocks conf_pde_blocks conf_transposes conf_ptm_stats conf_host_matrices sample_coords_rank0 denoiser_state_bcast denoiser_out_bcast``; schedule
words (``evidence.record_schedule``): ``diff_cond=rows diff_z_to_p=band diff_dit=local_q_rows conf_z=rows conf_heads=rows conf_finish=rowsum
conf_gpde=rowsum conf_pae_host=rank0`` next to the core's ``diff_* / conf_* / host_gather_*`` block facts.
"""
from __future__ import annotations

import types
from typing import Any, Callable, Dict, Optional

TAG = "[boltz2-opt]"


# ---------------------------------------------------------------- plumbing ----------------------------------------------------------------
def _rp():
    from boltz2_opt import rowpair as RP                                  # noqa: PLC0415  (rowpair imports this module lazily: no cycle)
    return RP


def _core():
    """The core seams this module binds, by name (a pinned opt_core without one is refused by name: ``rowpair._core_mod``)."""
    RP = _rp()
    return types.SimpleNamespace(D=RP._core_mod("dist"), SH=RP._core_mod("shard"), DF=RP._core_mod("diffusion"), HD=RP._core_mod("heads"),
                                 EV=RP._core_mod("evidence"))


def _count(key: str, n: int = 1) -> None:
    calls = _rp()._STATE["calls"]
    calls[key] = int(calls.get(key, 0)) + int(n)


def _refuse(msg: str):
    RP = _rp()
    return RP.Refused(f"refused: {msg} (boltz2_opt.rowpair_heads under n_gpu={RP._STATE.get('P')})")


class _Ambient:
    """The autocast state at a stock call site, re-entered where the same statement runs later (the sampler runs with autocast disabled; the
    token pair-bias projections are stock-computed inside DiffusionConditioning.forward under the caller's autocast)."""
    __slots__ = ("device_type", "enabled", "dtype")

    def __init__(self, device):
        import torch
        self.device_type = "cuda" if getattr(device, "type", str(device)) == "cuda" else "cpu"
        try:
            self.enabled = bool(torch.is_autocast_enabled(self.device_type))
            self.dtype = torch.get_autocast_dtype(self.device_type)
        except TypeError:                                                 # torch < 2.4 spelling
            self.enabled = bool(torch.is_autocast_enabled() if self.device_type == "cuda" else torch.is_autocast_cpu_enabled())
            self.dtype = torch.get_autocast_gpu_dtype() if self.device_type == "cuda" else torch.get_autocast_cpu_dtype()

    def enter(self):
        import torch
        return torch.autocast(self.device_type, dtype=self.dtype, enabled=self.enabled)

    def out_dtype(self, like):
        """The dtype a Linear returns at this call site."""
        return self.dtype if self.enabled else like.dtype


def _unwrapped(mod, what: str):
    if hasattr(mod, "_orig_mod"):
        raise _refuse(f"{what} is a compiled module (torch.compile) — the row statement calls its sub-modules; run compiled modules at --n_gpu 1")
    return mod


# ================================================================ seam 18: diffusion conditioning ================================================
class ZCondRows:
    """The conditioned pair tensor as this rank's ROWS (``z_loc [1, R, N, token_z]``) plus what the sharded token transformer derives from it:
    the per-layer pair-bias projections (``token_trans_proj_z``) and the autocast state they run under. Stands in for the stock
    ``token_trans_bias [B, N, N, 24*H]`` in the ``diffusion_conditioning`` dict: ``DiffusionModule.forward`` (diffusionv2.py:159) calls
    ``.float()`` on it and hands it to the token transformer, whose sharded form (:func:`sample_sharded`) reads the rows."""
    __slots__ = ("z_loc", "lay", "bias_layers", "amb", "c_z", "c_in")

    def __init__(self, z_loc, lay, bias_layers, amb, c_z: int, c_in: int):
        self.z_loc, self.lay, self.bias_layers, self.amb, self.c_z, self.c_in = z_loc, lay, bias_layers, amb, int(c_z), int(c_in)

    def float(self):                                                      # diffusionv2.py:159 — the widening cast happens per bias block in the attention statement
        return self

    @property
    def shape(self):
        raise _refuse("ZCondRows.shape: the stock DiffusionTransformer indexed the row-sharded conditioned pair — HEADS['sample_sharded'] is not bound")


def _atom_encoder_banded(enc, feats, s_trunk, z_loc, lay, C, band_rows: Optional[int]):
    """AtomEncoder.forward (encodersv2.py:299-414, ``structure_prediction=True``) statement by statement, with the token-pair term
    ``z_to_p[b, w, q, k] = z_to_p_trans(z)[b, token(q), token(k)]`` (stock: ``einsum('bijd,bwki,bwlj->bwkld')`` against the one-hot atom->token
    maps of the query / key slots — one non-zero term per real slot pair, 0 for a slot without an atom) read from the band of the projected
    ROWS (``pair_band_rows``: ``z_to_p_trans`` on this rank's rows per row block, folded and all-gathered as ``[1, N, 2W+1, atom_z]``; slots
    outside the band read whole rows the plan names — none for boltz's token-ordered atoms). Returns ``(q, c, p, to_keys)`` as stock."""
    import torch
    from functools import partial
    from torch.nn.functional import one_hot
    from boltz.model.modules.encodersv2 import get_indexing_matrix, single_to_keys
    with torch.autocast("cuda", enabled=False):
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"].bool()
        atom_ref_pos = feats["ref_pos"]
        atom_uid = feats["ref_space_uid"]
        atom_feats = [atom_ref_pos, feats["ref_charge"].unsqueeze(-1), feats["ref_element"]]
        if not enc.use_no_atom_char:
            atom_feats.append(feats["ref_atom_name_chars"].reshape(B, N, 4 * 64))
        if enc.use_atom_backbone_feat:
            atom_feats.append(feats["atom_backbone_feat"])
        if enc.use_residue_feats_atoms:
            res_feats = torch.cat([feats["res_type"], feats["modified"].unsqueeze(-1), one_hot(feats["mol_type"], num_classes=4).float()], dim=-1)
            atom_to_token = feats["atom_to_token"].float()
            atom_res_feats = torch.bmm(atom_to_token, res_feats)
            atom_feats.append(atom_res_feats)
        atom_feats = torch.cat(atom_feats, dim=-1)
        c = enc.embed_atom_features(atom_feats)
        W, H = enc.atoms_per_window_queries, enc.atoms_per_window_keys
        B, N = c.shape[:2]
        K = N // W
        keys_indexing_matrix = get_indexing_matrix(K, W, H, c.device)
        to_keys = partial(single_to_keys, indexing_matrix=keys_indexing_matrix, W=W, H=H)
        atom_ref_pos_queries = atom_ref_pos.view(B, K, W, 1, 3)
        atom_ref_pos_keys = to_keys(atom_ref_pos).view(B, K, 1, H, 3)
        d = atom_ref_pos_keys - atom_ref_pos_queries
        d_norm = torch.sum(d * d, dim=-1, keepdim=True)
        d_norm = 1 / (1 + d_norm)
        atom_mask_queries = atom_mask.view(B, K, W, 1)
        atom_mask_keys = to_keys(atom_mask.unsqueeze(-1).float()).view(B, K, 1, H).bool()
        atom_uid_queries = atom_uid.view(B, K, W, 1)
        atom_uid_keys = to_keys(atom_uid.unsqueeze(-1).float()).view(B, K, 1, H).long()
        v = (atom_mask_queries & atom_mask_keys & (atom_uid_queries == atom_uid_keys)).float().unsqueeze(-1)
        p = enc.embed_atompair_ref_pos(d) * v
        p = p + enc.embed_atompair_ref_dist(d_norm) * v
        p = p + enc.embed_atompair_mask(v) * v
        q = c
        # structure prediction branch
        atom_to_token = feats["atom_to_token"].float()
        s_to_c = enc.s_to_c_trans(s_trunk.float())
        s_to_c = torch.bmm(atom_to_token, s_to_c)
        c = c + s_to_c.to(c)
        tok = atom_to_token.argmax(dim=-1)                                        # [B, N_atom] the token of each atom slot
        has_tok = atom_to_token.sum(dim=-1) > 0                                   # a padded atom slot's one-hot row is all zero (its stock term is 0)
        q_idx = tok.view(B, K, W)
        q_ok = has_tok.view(B, K, W)
        k_idx = to_keys(tok.unsqueeze(-1).float()).view(B, K, H).long()           # the key slots' tokens through the same window indexing (exact small ints)
        k_ok = to_keys(has_tok.unsqueeze(-1).float()).view(B, K, H) > 0.5
        pair_valid = q_ok.unsqueeze(-1) & k_ok.unsqueeze(-2)                      # [B, K, W, H]
        plan = C.DF.band_plan(q_idx, k_idx, pair_valid, int(lay.N), max_w=None)   # as wide as the valid slots need: no row travels whole
        band, extras = C.DF.pair_band_rows(lambda zr: enc.z_to_p_trans(zr.float()), z_loc, lay, plan, rows=band_rows)
        z_to_p = C.DF.band_lookup(band, extras, plan)                             # [B, K, W, H, atom_z]
        del band, extras
        z_to_p = torch.where(pair_valid.unsqueeze(-1), z_to_p, z_to_p.new_zeros(()))
        calls = _rp()._STATE["calls"]
        calls["band_W"] = int(plan.W); calls["band_extra_rows"] = len(plan.extra_rows)
        p = p + z_to_p.to(p)
        del z_to_p
        p = p + enc.c_to_p_trans_q(c.view(B, K, W, 1, c.shape[-1]))
        p = p + enc.c_to_p_trans_k(to_keys(c).view(B, K, 1, H, c.shape[-1]))
        p = p + enc.p_mlp(p)
    return q, c, p, to_keys


def diffusion_conditioning_rows(model, T) -> Dict[str, Any]:
    """Seam 18 — ``DiffusionConditioning.forward`` (diffusion_conditioning.py:83-116) on the trunk rows. Returns the ``diffusion_conditioning``
    dict the sampler takes: ``q c to_keys atom_enc_bias atom_dec_bias`` as stock (replicated atom windows) and ``token_trans_bias`` = a
    :class:`ZCondRows` (the conditioned pair rows; seam 19 projects the per-layer biases from them per row block)."""
    import torch
    RP = _rp()
    C = _core()
    lay = C.D.require_sharded(T.lay, "diffusion_conditioning_rows")
    dc = model.diffusion_conditioning
    pc, enc = dc.pairwise_conditioner, dc.atom_encoder
    feats, z_loc = T.feats, T.z_loc
    amb = _Ambient(z_loc.device)
    R, N, c_z = int(z_loc.shape[-3]), int(z_loc.shape[-2]), int(z_loc.shape[-1])
    c_in = int(pc.dim_pairwise_init_proj[0].normalized_shape[0])                 # token_z + rel-pos features
    c_cond = int(pc.dim_pairwise_init_proj[1].out_features)
    L = len(dc.token_trans_proj_z)
    H = int(dc.token_trans_proj_z[0][1].out_features)
    c_pair = int(enc.z_to_p_trans[1].out_features)
    sched = C.DF.DiffusionSchedule.decide(lay, c_z=c_z, c_in=c_in, c_cond=c_cond, H=H, S=1, n_blocks=L, c_pair=c_pair, elt=int(z_loc.element_size()))

    def embed(z_rows, g0: int, g1: int):                                          # PairwiseConditioning.forward (encodersv2.py:210-218) on rows [g0, g1)
        rel = RP._relpos_rows(model.rel_pos, feats, g0, g1)                      # boltz2.py:508: relative_position_encoding = self.rel_pos(feats)
        z = torch.cat((z_rows, rel), dim=-1)
        _count("zcond_blocks")
        return pc.dim_pairwise_init_proj(z)

    transitions = [(lambda x, g0, g1, t=t: t(x)) for t in pc.transitions]         # `z = transition(z) + z`: added into the rows block by block
    plan = T.zplan                                                                # the ROLL-OUT ENTRY: under ROWPAIR_CONF_PARK_ZTRUNK the trunk shard is parked on pinned host and its
    plan.park_now()                                                               # device storage RELEASED here — before z_cond is allocated — so the roll-out holds z_cond rows, not
    src = plan.source()                                                           # z_cond + z_trunk; the conditioning reads the trunk rows per block from the park (census conf_ztrunk_entry=)
    zc = torch.empty((1, R, N, c_cond), dtype=amb.out_dtype(z_loc), device=z_loc.device)
    C.DF.pair_cond_rows(embed, src, lay, c_out=c_cond, rows=sched.cond_rows, transitions=transitions, out=zc)

    q, c, p, to_keys = _atom_encoder_banded(enc, feats, T.s, zc, lay, C, sched.band_rows)
    atom_enc_bias = torch.cat([layer(p) for layer in dc.atom_enc_proj_z], dim=-1)
    atom_dec_bias = torch.cat([layer(p) for layer in dc.atom_dec_proj_z], dim=-1)
    C.EV.record_schedule(diff_cond="rows", diff_z_to_p="band")
    return {"q": q, "c": c, "to_keys": to_keys, "atom_enc_bias": atom_enc_bias, "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": ZCondRows(zc, lay, list(dc.token_trans_proj_z), amb, c_z, c_in)}


# ================================================================ seam 19: the sampler's token transformer =========================================
def _check_dit_contract(layers) -> None:
    """H-qkvg: the row statement calls ``proj_q / proj_k / proj_v / proj_g / proj_o`` of every AttentionPairBias and expects ``proj_z`` to be the
    stock heads-first rearrangement (``compute_pair_bias=False``); a lever that fused or removed them is refused by name."""
    from torch import nn
    for i, ly in enumerate(layers):
        apb = getattr(ly, "pair_bias_attn", None)
        if apb is None or type(apb).__name__ != "AttentionPairBias":
            raise _refuse(f"dit_contract: token_transformer.layers[{i}].pair_bias_attn is {type(apb).__name__}, not AttentionPairBias")
        for a in ("proj_q", "proj_k", "proj_v", "proj_g", "proj_o"):
            if not isinstance(getattr(apb, a, None), nn.Linear):
                raise _refuse(f"dit_contract: token_transformer.layers[{i}].pair_bias_attn.{a} is {type(getattr(apb, a, None)).__name__}, not nn.Linear "
                              "(a lever re-bound the projections the row statement calls)")
        if getattr(apb, "compute_pair_bias", True) or any(isinstance(m, (nn.Linear, nn.LayerNorm)) for m in apb.proj_z.modules()):
            raise _refuse(f"dit_contract: token_transformer.layers[{i}].pair_bias_attn.proj_z computes a pair bias (compute_pair_bias=True); the token "
                          "transformer's biases come precomputed from DiffusionConditioning")
        for a in ("adaln", "output_projection", "transition", "post_lnorm"):
            if not isinstance(getattr(ly, a, None), nn.Module):
                raise _refuse(f"dit_contract: token_transformer.layers[{i}].{a} is missing")


def _dit_fns(C, ly, proj, amb: _Ambient, mask, multiplicity: int):
    """The five callables of one DiffusionTransformerLayer (transformersv2.py:168-197) for ``diffusion.dit_block_sharded``: AdaLN on all rows;
    K / V of all rows; AttentionPairBias (attentionv2.py:86-110) for a block of this rank's query rows against all keys with the layer's bias
    rows; the gated residual + ConditionedTransitionBlock + post-LN on this rank's rows; the layer's pair-bias projection of z_cond rows."""
    import torch
    apb = ly.pair_bias_attn
    H, hd = int(apb.num_heads), int(apb.head_dim)

    def norm(a, s):
        return ly.adaln(a, s)

    def kv(x):
        B = x.shape[0]
        return apb.proj_k(x).view(B, -1, H, hd), apb.proj_v(x).view(B, -1, H, hd)

    core = _dit_rows_core()                                                       # the core's pair-bias flash attention for the local query rows (T2), or None = the fp32 statement below

    def attn_fp32(q, k, v, bias, g_logits, B):                                    # AttentionPairBias's own statement (attentionv2.py:97-110) on the rows: fp32 logits [B, H, rows, N]
        with torch.autocast("cuda", enabled=False):
            a_ = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            a_ = a_ / (apb.head_dim ** 0.5) + bias.float()
            a_ = a_ + (1 - mask[:, None, None].float()) * -apb.inf
            a_ = a_.softmax(dim=-1)
            o = torch.einsum("bhij,bjhd->bihd", a_, v.float()).to(v.dtype)
        return g_logits.sigmoid() * o.reshape(B, -1, apb.c_s)

    def attn(x_q, kvp, bias_q, rows):
        k, v = kvp
        B = x_q.shape[0]
        q = apb.proj_q(x_q).view(B, -1, H, hd)
        bias = bias_q.repeat_interleave(multiplicity, 0)                         # proj_z = Rearrange('b ... h -> b h ...'): the core's [*, H, rows, N] layout already
        g = apb.proj_g(x_q)                                                       # gate logits [B, rows, H*hd] (the sigmoid is the kernel's epilogue / attn_fp32's)
        if core is not None and x_q.is_cuda:
            o = _dit_rows_served(core, apb, q, k, v, bias, mask, g, attn_fp32)   # [B, rows, H*hd] in x_q's dtype, or None = refused by name (booked once; the statement serves)
            if o is not None:
                return apb.proj_o(o)
        return apb.proj_o(attn_fp32(q, k, v, bias, g, B))

    def update(a, o_rows, s, rr):
        r0, r1 = rr
        s_r = s[..., r0:r1, :]
        b = ly.output_projection(s_r) * o_rows
        x = a[..., r0:r1, :] + b
        x = x + ly.transition(x, s_r)
        return ly.post_lnorm(x)

    def bias(z_rows):
        with amb.enter():                                                         # DiffusionConditioning.forward:107-110 — layer(z) under the conditioning call's autocast
            return proj(z_rows)

    return C.DF.DiTBlockFns(norm=norm, kv=kv, attn=attn, update=update, bias=bias)


# ----------------------------------------------------------------------------------------------------------------- the DiT rows' attention core
DIT_CORE_ENV = "ROWPAIR_DIT_CORE"          # apb_attn (default at n_gpu > 1) | torch: the token DiffusionTransformer's AttentionPairBias over this rank's query rows against all
                                           #  keys through the core's pair-bias flash attention (opt_core.attn.apb_core: the Triton `apb_attn` kernel, bf16 operands, fp32 online
                                           #  softmax, sigmoid gate fused — the numerics class of the one-GPU big line's fused bf16 DiT step) instead of the fp32 einsum statement
                                           #  ([B, 16, rows, N] fp32 logits per layer per step); a call the kernel does not serve runs the statement, named (census dit_rows_core=…)
_DIT_CORE: Dict[str, Any] = {"mod": None, "word": None, "why": None, "served": 0, "stock": 0, "refused": {}, "checked": None}


def _dit_rows_core():
    """Resolve ``ROWPAIR_DIT_CORE`` once: the core entry module (``opt_core.attn.apb_core``) for ``apb_attn``, None for ``torch`` or when the
    entry / its kernel is not importable (named in the census: ``dit_rows_core=unavailable:<why>``)."""
    if _DIT_CORE["word"] is not None:
        return _DIT_CORE["mod"]
    import os
    word = (os.environ.get(DIT_CORE_ENV, "") or "apb_attn").strip().lower()
    _DIT_CORE["word"] = word
    if word in ("torch", "0", "off"):
        return None
    if word != "apb_attn":
        raise _refuse(f"{DIT_CORE_ENV}={word!r}: one of apb_attn | torch (or unset)")
    try:
        import importlib
        mod = importlib.import_module("opt_core.attn.apb_core")
        mod.kernel()                                                            # resolves the carried kernel through the core's route; Unsupported('import:apb_attn') by name
        _DIT_CORE["mod"] = mod
    except Exception as e:  # noqa: BLE001 — an entry that cannot load is a NAMED state; the statement serves
        _DIT_CORE["why"] = f"{type(e).__name__}:{str(e)[:120]}"
    return _DIT_CORE["mod"]


def _dit_rows_served(core, apb, q, k, v, bias, mask, g, attn_fp32):
    """One AttentionPairBias core call on the kernel: q [B, rows, H, hd], k / v [B, N, H, hd] (bf16 copies, [B, H, ·, hd] views), the layer's
    bias rows [B|1, H, rows, N] as they are (fp32 or bf16, unit key stride), the key mask [B|1, N], the gate LOGITS [B, rows, H*hd]; returns
    ``sigmoid(g) * softmax(q·k/√hd + bias − inf·(1−mask)) v`` as [B, rows, H*hd] in q's dtype, or None when the entry refuses the call by name
    (``Unsupported``: booked under its event word; that call and — for a shape-class refusal — the later ones run the statement). The first
    served call of the process is checked against the fp32 statement on the same operands (census ``dit_rows_core_maxdiff``)."""
    import torch
    B, H, hd = int(q.shape[0]), int(q.shape[2]), int(q.shape[3])
    dt = torch.bfloat16
    try:
        o = core.pair_bias_attention(q.to(dt).permute(0, 2, 1, 3), k.to(dt).permute(0, 2, 1, 3), v.to(dt).permute(0, 2, 1, 3), bias, mask,
                                     gate=g.to(dt), num_samples=B, num_heads=H, layout="shnd", inf=float(apb.inf))      # [B, rows, H*hd]
    except core.Unsupported as e:
        ev = str(getattr(e, "event", None) or type(e).__name__)
        _DIT_CORE["refused"][ev] = _DIT_CORE["refused"].get(ev, 0) + 1
        _DIT_CORE["stock"] += 1
        return None
    if _DIT_CORE["checked"] is None:                                            # once per process: the kernel's rows vs the fp32 statement on these very operands
        ref = attn_fp32(q, k, v, bias, g, B)
        _DIT_CORE["checked"] = float((o.float() - ref.float()).abs().max())
    _DIT_CORE["served"] += 1
    return o.to(q.dtype)


def dit_core_report() -> Dict[str, Any]:
    """Census of the DiT rows' attention core: ``{word, state, why, served, stock, refused, maxdiff_first_call}`` (rowpair.report: tp_report.dit_core;
    schedule words dit_rows_core= dit_rows_core_served= dit_rows_core_stock= dit_rows_core_maxdiff=)."""
    word = _DIT_CORE["word"]
    state = "off" if word is None else ("torch" if _DIT_CORE["mod"] is None and _DIT_CORE["why"] is None else ("on" if _DIT_CORE["mod"] is not None else "unavailable"))
    return {"word": word, "state": state, "why": _DIT_CORE["why"], "served": int(_DIT_CORE["served"]), "stock": int(_DIT_CORE["stock"]),
            "refused": dict(_DIT_CORE["refused"]), "maxdiff_first_call": _DIT_CORE["checked"]}


# ----------------------------------------------------------------------------------------------------------------- the pair-bias cache decision
BIAS_CACHE_FREE_FRAC = 0.80                # the per-layer pair-bias row cache ([L, H, R, N] in the conditioned rows' dtype: 768·N²/P B at L = 24, H = 16, bf16) saves L x steps
BIAS_CACHE_MARGIN_B = 2 * 2 ** 30          #  projections of the z_cond rows per roll-out; the core's default budget is a FIXED 24 GB (ROWPAIR_DIFF_BIAS_CACHE_GB), which turns the
                                           #  cache off above N²/P = 31e6 (N > 7,900 at P = 2; 11,180 at 4; 15,800 at 8) however much memory is free. Rule here: the cache is
                                           #  also ON when it fits BIAS_CACHE_FREE_FRAC of the ranks' AGREED free bytes at the sampler's entry minus a margin (the trunk's shard is
                                           #  parked by then; the roll-out's other transients are row blocks). A caller's own pin (ROWPAIR_DIFF_BIAS_CACHE[_GB]) is left to the core.


def _bias_cache_from_free(C, lay, L: int, H: int, elt: int):
    """True = hand the core `bias_cache=True` ("given") because the agreed free memory admits the cache its fixed default budget would refuse; None =
    let the core decide (its default admits it, a caller pinned the decision, or no CUDA). Census ``diff_bias_cache_rule=core|free:<need_gib>/<free_gib>``."""
    import os
    if os.environ.get("ROWPAIR_DIFF_BIAS_CACHE", "").strip() or os.environ.get("ROWPAIR_DIFF_BIAS_CACHE_GB", "").strip():
        C.EV.record_schedule(diff_bias_cache_rule="core:pinned")
        return None
    P, N = int(lay.P), int(lay.N)
    r_max = max(int(lay.nrows(q)) for q in range(P)) if hasattr(lay, "nrows") else -(-N // P)
    need = int(L) * int(H) * r_max * N * int(elt)
    default_gb = float(getattr(C.DF, "BIAS_CACHE_GB_DEFAULT", 24.0))
    if need <= int(default_gb * 1e9):
        C.EV.record_schedule(diff_bias_cache_rule="core")
        return None
    try:
        free = (getattr(C.D, "agreed_free_bytes", None) or C.DF.agreed_free_bytes)()   # ONE small collective: min over the ranks (dist.agreed_free_bytes)
    except Exception:  # noqa: BLE001
        free = None
    if free is None:
        C.EV.record_schedule(diff_bias_cache_rule="core:no_free_reading")
        return None
    ok = need <= int(BIAS_CACHE_FREE_FRAC * int(free)) - BIAS_CACHE_MARGIN_B
    C.EV.record_schedule(diff_bias_cache_rule=f"{'free' if ok else 'core'}:{need / 2 ** 30:.1f}/{int(free) / 2 ** 30:.1f}")
    return True if ok else None


def sample_sharded(model, T, **sample_kw) -> Dict[str, Any]:
    """Seam 19 — ``AtomDiffusion.sample(**sample_kw)`` VERBATIM with the score model's token DiffusionTransformer bound to the row-sharded
    statement for the duration of the call (an instance-level ``forward``; restored on exit). ``sample_kw["diffusion_conditioning"]`` is the dict
    of :func:`diffusion_conditioning_rows`. Every denoiser call conditions on rank 0's state (``diffusion.sync_replicated``, bcast); at exit rank 0's coordinates replace every rank's after the cross-rank spread is recorded (``diff_rank_spread_A``; refused by name above ``SPREAD_MAX_A``)."""
    RP = _rp()
    C = _core()
    lay = C.D.require_sharded(T.lay, "sample_sharded")
    cond = sample_kw.get("diffusion_conditioning")
    zc = cond.get("token_trans_bias") if isinstance(cond, dict) else None
    if not isinstance(zc, ZCondRows):
        raise _refuse(f"sample_sharded: diffusion_conditioning['token_trans_bias'] is {type(zc).__name__}, not the row-sharded conditioned pair "
                      "(HEADS['diffusion_conditioning_rows'] and HEADS['sample_sharded'] are bound together)")
    sm = model.structure_module
    score = _unwrapped(sm.score_model, "structure_module.score_model")
    tt = _unwrapped(score.token_transformer, "score_model.token_transformer")
    if "forward" in vars(tt):
        raise _refuse("sample_sharded: score_model.token_transformer.forward is already re-bound on the instance")
    layers = list(tt.layers)
    L = len(layers)
    if L != len(zc.bias_layers):
        raise _refuse(f"sample_sharded: token_transformer has {L} layers, DiffusionConditioning projects {len(zc.bias_layers)} pair biases")
    if not getattr(tt, "pair_bias_attn", True):
        raise _refuse("sample_sharded: token_transformer.pair_bias_attn is False")
    _check_dit_contract(layers)
    H = int(layers[0].pair_bias_attn.num_heads)
    mult = int(sample_kw.get("multiplicity", 1) or 1)
    mps = sample_kw.get("max_parallel_samples")
    S = min(mult, int(mps)) if mps else mult                                      # samples per score-model call (diffusionv2.py:362)
    cache_given = _bias_cache_from_free(C, lay, L, H, int(zc.z_loc.element_size()))   # True when the agreed free memory admits the cache the core's fixed-GB default would refuse; None = the core decides
    sched = C.DF.DiffusionSchedule.decide(lay, c_z=zc.c_z, c_in=zc.c_in, c_cond=int(zc.z_loc.shape[-1]), H=H, S=S, n_blocks=L, c_pair=0,
                                          elt=int(zc.z_loc.element_size()), bias_cache=cache_given)
    cache = C.DF.PairBiasCache(enabled=bool(sched.bias_cache))

    def token_transformer_rows(a, s, bias=None, mask=None, to_keys=None, multiplicity=1):   # DiffusionTransformer.forward (transformersv2.py:97-139)
        if bias is not zc:
            raise _refuse("token_transformer: bias is not the row-sharded conditioned pair of this roll-out")
        if to_keys is not None:
            raise _refuse("token_transformer: to_keys given (an atom transformer) — only the token transformer is row-sharded")
        blocks = [_dit_fns(C, ly, zc.bias_layers[i], zc.amb, mask, int(multiplicity)) for i, ly in enumerate(layers)]
        out = C.DF.diffusion_transformer_sharded(blocks, a, s, zc.z_loc, lay, schedule=sched, bias_cache=cache)
        _count("dit_layers_sharded", L); _count("dit_transformer_calls")
        _dc = dit_core_report()                                                   # the DiT rows' attention core census (schedule words, refreshed per call)
        C.EV.record_schedule(dit_rows_core=(_dc["state"] if _dc["state"] != "on" else "apb_attn") + (f":{_dc['why']}" if _dc["why"] else ""),
                             dit_rows_core_served=_dc["served"], dit_rows_core_stock=_dc["stock"],
                             **({"dit_rows_core_refused": ",".join(f"{k}:{n}" for k, n in sorted(_dc["refused"].items()))} if _dc["refused"] else {}),
                             **({"dit_rows_core_maxdiff": round(_dc["maxdiff_first_call"], 5)} if _dc["maxdiff_first_call"] is not None else {}))
        return out

    if "preconditioned_network_forward" in vars(sm):
        raise _refuse("sample_sharded: structure_module.preconditioned_network_forward is already re-bound on the instance")
    pnf = sm.preconditioned_network_forward                                       # the bound stock method

    policy = RP.sync_policy()                                                     # bcast (det 0: rank 0 authoritative) | guard (strict)

    def pnf_rank0_state(noised_atom_coords, sigma, network_condition_kwargs):    # diffusionv2.py:388-396: the loop hands the denoiser an
        x0 = C.DF.sync_replicated(noised_atom_coords.contiguous(), "denoiser_state", mode=policy)   # advanced-index COPY of its noisy state and
        if x0.data_ptr() != noised_atom_coords.data_ptr():                        # STORES the return value (`atom_coords_denoised[ids] = chunk`), which
            noised_atom_coords.copy_(x0)                                          # its update statement (:523-525) then reads. So: the denoiser
        _count("denoiser_state_" + policy)                                        # conditions on rank 0's state (the sharded DiT mixes query rows of
        out = pnf(noised_atom_coords, sigma, network_condition_kwargs=network_condition_kwargs)      # ONE state) and RETURNS rank 0's output — the
        out0 = C.DF.sync_replicated(out.contiguous(), "denoiser_out", mode=policy)                  # only cross-rank-divergent term of the update
        _count("denoiser_out_" + policy)                                          # (replicated atom decoder, non-deterministic kernels); the noisy
        return out0                                                               # state itself is identical by construction: identical carried
                                                                                  # coordinates + identical draws (rank 0's generator states)

    tt.forward = token_transformer_rows                                           # nn.Module.__call__ resolves the instance attribute first
    sm.preconditioned_network_forward = pnf_rank0_state
    try:
        out = sm.sample(**sample_kw)                                              # the stock loop
    finally:
        for obj, name in ((tt, "forward"), (sm, "preconditioned_network_forward")):
            try:
                delattr(obj, name)
            except AttributeError:
                pass
    x = out.get("sample_atom_coords")
    if x is not None:                                                             # sampler exit: rank 0's coordinates on every rank (the confidence rows embed ONE structure),
        mine = x.contiguous().clone()                                             # after recording the cross-rank spread max|x_rank - x_rank0| (A)
        x0 = C.DF.sync_replicated(x.contiguous(), "sample_atom_coords", mode="bcast")
        spread = max(float(v) for v in C.D.comm().allgather_obj(float((mine - x0).abs().max())))
        del mine
        C.EV.record_schedule(diff_rank_spread_A=round(spread, 6))
        _count("sample_coords_rank0")
        if spread > SPREAD_MAX_A:
            raise _refuse(f"sample_sharded: cross-rank coordinate spread {spread:.3f} A > {SPREAD_MAX_A} A (the ranks' diffusion states diverged: "
                          "a replicated input differs across ranks)")
        out["sample_atom_coords"] = x0
    C.EV.record_schedule(diff_dit="local_q_rows", diff_noise=("bcast_rank0_state+rng_state_bcast" if policy == "bcast" else "guard"), sync_policy=policy)
    _release_zcond(zc, cache, C)                                                  # the conditioned pair rows' last reader was the roll-out (the confidence module embeds the trunk's z, not z_cond)
    return out


ZCOND_RELEASE_ENV = "ROWPAIR_ZCOND_RELEASE"   # =0: the conditioned pair rows z_cond [1, N/P, N, token_z] (and the roll-out's pair-bias row cache) stay allocated until Boltz2.forward
                                             #  drops its `diffusion_conditioning` dict at return; unset/1 (default): released at the sampler's exit — the
                                             #  confidence pass, whose transients set the item's final peak at large N, then runs without 256·N²/P B (+ the cache) of dead rows
                                             #  resident (census zcond_released_gib=<g>|kept)


def _release_zcond(zc, cache, C) -> None:
    import os
    import torch
    word = (os.environ.get(ZCOND_RELEASE_ENV, "") or "1").strip().lower()
    if word in ("0", "off", "keep"):
        C.EV.record_schedule(zcond_released_gib="kept")
        return
    freed = 0
    z = getattr(zc, "z_loc", None)
    if isinstance(z, torch.Tensor):
        freed += int(z.numel()) * int(z.element_size())
        zc.z_loc = torch.empty((0,), dtype=z.dtype, device=z.device)             # a reader after this point fails LOUDLY on shape, never reads stale rows
        del z
    for attr in ("store", "rows", "blocks", "_rows"):                            # the PairBiasCache's per-layer bias rows (core: `.store`), whatever the core names its store
        store = getattr(cache, attr, None)
        if isinstance(store, dict) and store:
            for v in store.values():
                if isinstance(v, torch.Tensor):
                    freed += int(v.numel()) * int(v.element_size())
            store.clear()
        elif isinstance(store, list) and store:
            for v in store:
                if isinstance(v, torch.Tensor):
                    freed += int(v.numel()) * int(v.element_size())
            store.clear()
    rel = getattr(cache, "release", None) or getattr(cache, "clear", None)
    if callable(rel):
        try:
            rel()
        except Exception:  # noqa: BLE001 — a cache without a release is dropped with the sampler's locals anyway
            pass
    C.EV.record_schedule(zcond_released_gib=round(freed / 2 ** 30, 3))
    _count("zcond_released")


SPREAD_MAX_A = 1.0                                                                # sampler exit: the largest tolerated cross-rank coordinate spread (A) before rank 0's replace every rank's


# ================================================================ seams 16-17: confidence =========================================================
def _confidence_one(model, T, x_pred, C, i: int = 0):
    """ConfidenceModule.forward for ONE sample (multiplicity 1) on the rows — confidence pass ``i`` of ``T.zplan``; returns the stock output
    dict (see the module docstring for where each entry lives)."""
    import torch
    from boltz.data import const
    from boltz.model.layers.confidence_utils import compute_aggregated_metric, compute_frame_pred, tm_function
    RP = _rp()
    cm = model.confidence_module
    ch = cm.confidence_heads
    if not ch.token_level_confidence:
        raise _refuse("confidence_rows: atom-level confidence heads (token_level_confidence=False) are not bound under n_gpu>1")
    lay = T.lay
    feats = T.feats
    z_loc = T.z_loc
    dev = z_loc.device
    N = int(lay.N)
    R = int(z_loc.shape[-3])
    r0 = int(lay.r0)
    multiplicity = 1
    # -- confidencev2.py:158-175: the single tracks (replicated)
    s_inputs = cm.s_inputs_norm(T.s_inputs)
    s = T.s
    if not cm.no_update_s:
        s = cm.s_norm(s)
    if cm.add_s_input_to_s:
        s = s + cm.s_input_to_s(s_inputs)
    s = s.repeat_interleave(multiplicity, 0)
    s_to_z_i = cm.s_to_z(s_inputs)                                                # [1, N, C]: the `i` and `j` operands of the outer sum, computed once
    s_to_z_j = cm.s_to_z_transpose(s_inputs)
    prod_i = cm.s_to_z_prod_in1(s_inputs) if cm.add_s_to_z_prod else None
    prod_j = cm.s_to_z_prod_in2(s_inputs) if cm.add_s_to_z_prod else None
    # -- confidencev2.py:184-195: representative-atom coordinates (replicated, [1, N, 3])
    token_to_rep_atom = feats["token_to_rep_atom"].repeat_interleave(multiplicity, 0)
    if len(x_pred.shape) == 4:
        B, mult, NA, _ = x_pred.shape
        x_pred = x_pred.reshape(B * mult, NA, -1)
    x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)

    # -- seam 16: the confidence pair input born from the trunk rows (confidencev2.py:163-198 with the `i` operands sliced to the block)
    def embed(z_rows, g0: int, g1: int):
        z = cm.z_norm(z_rows)
        if cm.add_z_input_to_z:
            fr = RP._feats_rows(feats, g0, g1, dev)
            relative_position_encoding = RP._relpos_rows(cm.rel_pos, feats, g0, g1)
            z = z + relative_position_encoding
            z = z + cm.token_bonds(fr["token_bonds"].float())
            if cm.bond_type_feature:
                z = z + cm.token_bonds_type(fr["type_bonds"].long())
            z = z + cm.contact_conditioning(fr)
        z = z + s_to_z_i[:, g0:g1, None, :] + s_to_z_j[:, None, :, :]
        if cm.add_s_to_z_prod:
            z = z + cm.s_to_z_prod_out(prod_i[:, g0:g1, None, :] * prod_j[:, None, :, :])
        d = torch.cdist(x_pred_repr[:, g0:g1], x_pred_repr)                       # rows [g0, g1) of cdist(x, x)
        distogram = (d.unsqueeze(-1) > cm.boundaries).sum(dim=-1).long()
        distogram = cm.dist_bin_pairwise_embed(distogram)
        z = z + distogram
        _count("conf_embed_rows")
        return z

    src, inplace = T.zplan.begin(i, lead_out=(1,), out_dtype=torch.float32)     # the trunk shard's form this pass (core heads.ZTrunkPlan): in place (this pass's pair input overwrites the
    z = C.HD.embed_rows(embed, src, lay, lead_out=(1,), inplace=inplace, out_dtype=torch.float32)   # shard: one sample, last use), parked (host copy live, device storage released before the
                                                                                  # [1, R, N, C] below is allocated; rows served per block) or resident; R rows, never N
    if callable(getattr(T.zplan, "retire", None)):                              # the z_trunk park's last reader was this embed on the LAST pass —
        C.EV.record_schedule(ztrunk_retire=str(T.zplan.retire(i)))                #  retire it BEFORE the pass's pair stack (a no-op by word on non-last passes / unparked forms)
    mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
    pair_mask = T.pair_mask
    stack = _unwrapped(cm.pairformer_stack, "confidence_module.pairformer_stack")
    s, zs = RP._presharded_stack(stack.layers, z[0], pair_mask[0], lay, bool(getattr(model, "use_kernels", False)), s=s, mask=mask)
    C.EV.record_schedule(conf_pairstack="inplace" if zs.data_ptr() == z.data_ptr() else "copy")   # the confidence Pairformer updates the pass's pair rows IN PLACE (core pair_block_): one shard-equivalent, no working copy
    z = zs.unsqueeze(0)                                                           # no residual adds, as upstream: s, z = s_t, z_t (confidencev2.py:203-204)

    # -- seam 17: ConfidenceHeads.forward (confidencev2.py:275-495) on rows
    asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    if ch.use_separate_heads:
        def _sep(intra, inter):
            def fn_rows(x_rows, g0, g1):
                is_same_chain = asym_id[:, g0:g1].unsqueeze(-1) == asym_id.unsqueeze(-2)
                is_different_chain = ~is_same_chain
                a = intra(x_rows) * is_same_chain.float().unsqueeze(-1)
                b = inter(x_rows) * is_different_chain.float().unsqueeze(-1)
                return b + a
            return fn_rows
        pae_fn = _sep(ch.to_pae_intra_logits, ch.to_pae_inter_logits)
        pde_fn = _sep(ch.to_pde_intra_logits, ch.to_pde_inter_logits)
    else:
        pae_fn = lambda x_rows, g0, g1: ch.to_pae_logits(x_rows)                  # noqa: E731
        pde_fn = lambda x_rows, g0, g1: ch.to_pde_logits(x_rows)                  # noqa: E731
    resolved_logits = ch.to_resolved_logits(s)
    plddt_logits = ch.to_plddt_logits(s)
    ligand_weight = 20
    non_interface_weight = 1
    interface_weight = 10
    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)
    is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    plddt = compute_aggregated_metric(plddt_logits)
    token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
    complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(dim=-1)
    rb, _src = C.HD.conf_rows(N, 64)
    # the interface mask: max over j of (d < 8) * different chain, per row block of cdist over ALL rows (replicated: O(rows x N) transient)
    token_interface_mask = torch.empty_like(token_pad_mask)
    for g0 in range(0, N, rb):
        g1 = min(N, g0 + rb)
        d = torch.cdist(x_pred_repr[:, g0:g1], x_pred_repr)
        is_contact = (d < 8).float()
        is_different_chain = (asym_id[:, g0:g1].unsqueeze(-1) != asym_id.unsqueeze(-2)).float()
        token_interface_mask[:, g0:g1] = torch.max(is_contact * is_different_chain * (1 - is_ligand_token)[:, g0:g1].unsqueeze(-1), dim=-1).values
        del d, is_contact, is_different_chain
    token_non_interface_mask = (1 - token_interface_mask) * (1 - is_ligand_token)
    iplddt_weight = (is_ligand_token * ligand_weight + token_interface_mask * interface_weight + token_non_interface_mask * non_interface_weight)
    complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(dim=-1) / torch.sum(token_pad_mask * iplddt_weight, dim=-1)

    # PAE rows: the pae matrix rows + compute_ptms (confidence_utils.py) as ROW PARTIAL SUMS — boltz weighs every pair with ONE d0 (N_res of the
    # whole padded complex), so each statistic max_i [sum_j T[i,j] M[i,j] / (sum_j M[i,j] + 1e-5)] is separable per row: the block's rows of
    # T = sum_b probs * tm_value and of every mask are the stock statements with the `i` operand sliced; no [N, N] tensor on any rank
    _, mask_collinear_pred = compute_frame_pred(x_pred, feats["frames_idx"], feats, multiplicity, inference=True)
    mask_pad = token_pad_mask
    maski = mask_collinear_pred.reshape(-1, mask_collinear_pred.shape[-1])
    is_protein_token = (token_type == const.chain_type_ids["PROTEIN"]).float()
    asym_ids_list = torch.unique(asym_id).tolist()
    n_chain = len(asym_ids_list)
    n_stat = 4 + n_chain * n_chain                                                # ptm, iptm, ligand_iptm, protein_iptm, then (idx1, idx2) in the stock loop order
    num_bins = int((ch.to_pae_intra_logits if ch.use_separate_heads else ch.to_pae_logits).out_features)
    bin_width = 32.0 / num_bins
    end = 32.0
    pae_value = torch.arange(start=0.5 * bin_width, end=end, step=bin_width, device=dev).unsqueeze(0)
    N_res = mask_pad.sum(dim=-1, keepdim=True)
    tm_value = tm_function(pae_value, N_res).unsqueeze(1).unsqueeze(2)            # [B, 1, 1, bins]
    numden = torch.zeros((R, 2, n_stat), dtype=torch.float32, device=dev)        # per local row: numerator / denominator of every statistic
    pae_rows = torch.empty((1, R, N), dtype=torch.float32, device=dev)
    for i0, i1, zr in C.HD.logit_rows(lambda x: x, z, lay, rb, bins=num_bins):
        g0, g1 = r0 + i0, r0 + i1
        pae_logits = pae_fn(zr, g0, g1)
        pae_rows[:, i0:i1] = compute_aggregated_metric(pae_logits, end=32)
        probs = torch.nn.functional.softmax(pae_logits, dim=-1)
        tm_expected_value = torch.sum(probs * tm_value, dim=-1)                   # rows [g0, g1) of stock's [B, N, N]
        del pae_logits, probs
        mi, pr, pc_ = maski[:, g0:g1, None], mask_pad[:, g0:g1, None], mask_pad[:, None, :]
        different = asym_id[:, None, :] != asym_id[:, g0:g1, None]
        masks = [mi * pc_ * pr,                                                                                       # pair_mask_ptm rows
                 mi * different * pc_ * pr,                                                                           # pair_mask_iptm rows
                 mi * different * pc_ * pr * ((is_ligand_token[:, g0:g1, None] * is_protein_token[:, None, :])
                                              + (is_protein_token[:, g0:g1, None] * is_ligand_token[:, None, :])),   # ligand_iptm_mask rows
                 mi * different * pc_ * pr * (is_protein_token[:, g0:g1, None] * is_protein_token[:, None, :])]       # protein_ipmt_mask rows
        masks += [mi * (asym_id[:, None, :] == idx1) * (asym_id[:, g0:g1, None] == idx2) * pc_ * pr for idx1 in asym_ids_list for idx2 in asym_ids_list]
        for k, M in enumerate(masks):
            numden[i0:i1, 0, k] = torch.sum(tm_expected_value * M, dim=-1)[0]
            numden[i0:i1, 1, k] = torch.sum(M, dim=-1)[0]
        del tm_expected_value, masks
        _count("conf_pae_blocks")
    full = C.SH.unshard_rows(numden, lay)                                         # [N, 2, n_stat] on every rank: O(N C^2) floats
    stats = torch.max(full[:, 0, :] / (full[:, 1, :] + 1e-5), dim=0).values      # the stock `torch.max(num / (den + 1e-5), dim=1).values` per statistic
    del full, numden
    ptm, iptm, ligand_iptm, protein_iptm = stats[0].reshape(1), stats[1].reshape(1), stats[2].reshape(1), stats[3].reshape(1)
    pair_chains_iptm = {idx1: {idx2: stats[4 + a * n_chain + b].reshape(1) for b, idx2 in enumerate(asym_ids_list)} for a, idx1 in enumerate(asym_ids_list)}
    _count("conf_ptm_stats", n_stat)
    # PDE rows over (z + z^T) rows: ONE shard transpose; gPDE / giPDE as row-block partial sums
    zt = None
    if not _pde_blocks_streamed():                                                # ROWPAIR_SYM_ZT=whole: ONE shard transpose, a whole [R, N, C] transposed copy resident through the loop
        zt = RP._zT_rows(z, lay)
        _count("conf_transposes")
        zt += z                                                                   # (z + z^T) rows, in place on the transposed rows
    pde_rows = torch.empty((1, R, N), dtype=torch.float32, device=dev)
    pd_rows = T.pd_loc[:, :, :, 0] if T.pd_loc.dim() == 5 else T.pd_loc              # boltz2.py:598: pred_distogram_logits = pdistogram[:, :, :, 0]
    contacts = torch.zeros((1, 1, 1, int(pd_rows.shape[-1])), dtype=pd_rows.dtype).to(dev)
    contacts[:, :, :, :20] = 1.0
    cols = torch.arange(N, device=dev)
    sums = torch.zeros((4, 1), dtype=torch.float32, device=dev)                   # [num_pde, den_pde, num_ipde, den_ipde]
    sym_rows = (C.HD.logit_rows(lambda x: x, zt, lay, rb, bins=num_bins) if zt is not None
                else _zzT_row_blocks(z, lay, rb, C))                             # (z + z^T) row blocks off the core's block-streamed transpose — no whole transposed shard
    for i0, i1, xr in sym_rows:
        g0, g1 = r0 + i0, r0 + i1
        if i1 <= i0:
            continue                                                              # an exchange round this rank has no rows in (uneven shards): the collective ran inside the generator
        pde_logits = pde_fn(xr, g0, g1)
        pde = compute_aggregated_metric(pde_logits, end=32)
        pde_rows[:, i0:i1] = pde
        pred_distogram_prob = torch.nn.functional.softmax(pd_rows[:, i0:i1], dim=-1).repeat_interleave(multiplicity, 0)
        prob_contact = (pred_distogram_prob * contacts).sum(-1)
        eye_rows = (cols.unsqueeze(0) == torch.arange(g0, g1, device=dev).unsqueeze(1)).to(token_pad_mask.dtype).unsqueeze(0)
        token_pad_pair_mask = token_pad_mask[:, g0:g1].unsqueeze(-1) * token_pad_mask.unsqueeze(-2) * (1 - eye_rows)
        token_pair_mask = token_pad_pair_mask * prob_contact
        sums[0] += (pde * token_pair_mask).sum(dim=(1, 2)); sums[1] += token_pair_mask.sum(dim=(1, 2))
        token_interface_pair_mask = token_pair_mask * (asym_id[:, g0:g1].unsqueeze(-1) != asym_id.unsqueeze(-2))
        sums[2] += (pde * token_interface_pair_mask).sum(dim=(1, 2)); sums[3] += token_interface_pair_mask.sum(dim=(1, 2))
        del pde_logits, pde, pred_distogram_prob, prob_contact, token_pair_mask, token_interface_pair_mask
        _count("conf_pde_blocks")
    del zt, sym_rows
    C.D.allreduce_(sums, "sum")
    complex_pde = sums[0] / sums[1]
    complex_ipde = sums[2] / (sums[3] + 1e-5)

    # the [N, N] output matrices: rank 0's HOST (the writer runs there); other ranks keep their rows (named)
    pae_full = C.D.gather_rows_to_rank0_host(pae_rows[0], lay)
    pde_full = C.D.gather_rows_to_rank0_host(pde_rows[0], lay)
    if pae_full is not None:
        _count("conf_host_matrices", 2)
        pae_out, pde_out = pae_full.unsqueeze(0), pde_full.unsqueeze(0)
    else:
        pae_out, pde_out = pae_rows, pde_rows
    C.EV.record_schedule(conf_z="rows", conf_heads="rows", conf_finish="rowsum", conf_gpde="rowsum", conf_pae_host="rank0",
                         conf_pae_ranks="own_rows", conf_rows_block=int(rb))
    out = dict(plddt_logits=plddt_logits, resolved_logits=resolved_logits, pde=pde_out, plddt=plddt, complex_plddt=complex_plddt,
               complex_iplddt=complex_iplddt, complex_pde=complex_pde, complex_ipde=complex_ipde, pae=pae_out, ptm=ptm, iptm=iptm,
               ligand_iptm=ligand_iptm, protein_iptm=protein_iptm, pair_chains_iptm=pair_chains_iptm)
    if cm.return_latent_feats:
        out["s_conf"] = s
        out["z_conf"] = z                                                         # rows (named): [1, R, N, C]
    return out


# ----------------------------------------------------------------------------------------------------------------- (z + z^T) row blocks
# The confidence PDE head reads (z + z^T) rows: under rowpair.SYM_ZT_ENV=blocks (default) from the core's block-streamed transpose (dist.transpose_blocks:
# one all-to-all per row block, transient O(N · rows · C)) instead of ONE whole shard transpose resident through the head (whole form: +512·N²/P B at the
# item's FINAL peak). Census pde_zT=blocks conf_zT_blocks=.


def _pde_blocks_streamed() -> bool:
    return bool(_rp()._sym_blocks_streamed())                                    # ONE word for both (z + z^T) readers: rowpair.SYM_ZT_ENV (ROWPAIR_SYM_ZT=blocks|whole)


def _zzT_row_blocks(z, lay, rb: int, C):
    """Yields ``(i0, i1, x)`` for this rank's local row blocks with ``x[0, w, j, :] = z[0, i0+w, j, :] + z^T rows`` — i.e. the (z + z^T) rows the
    PDE head reads — from the core's block-streamed transpose of the shard (``dist.transpose_blocks``: COLLECTIVE, every rank walks the same
    rounds; exhausted by the caller's loop). ``z``: ``[1, R, N, C]``."""
    import torch
    P = int(getattr(lay, "P", 0) or 0) or int(C.D.comm().world)
    r_max = max(int(lay.nrows(q)) for q in range(P))                            # the widest shard (the same on every rank: the exchange is COLLECTIVE, one step for all)
    step = max(1, min(int(rb), r_max))                                            # a block never spans more rows than a shard holds (keeps every staging buffer [N, step, C], never N x N)
    n = 0
    for i0, i1, zT_blk in C.D.transpose_blocks(z[0], lay, step=step):            # zT_blk [i1-i0, N, C]: a transposed view of the round's column block
        n += 1
        if i1 > i0:
            x = torch.empty_like(z[:, i0:i1])                                     # the block in the shard's own (contiguous row) layout: the head's Linear then runs the very GEMM the
            torch.add(z[:, i0:i1], zT_blk.unsqueeze(0), out=x)                   #  whole-transpose statement ran on these rows (a transposed-stride operand takes another path: not bitwise)
            yield i0, i1, x
        else:
            yield i0, i1, None
    C.EV.record_schedule(pde_zT="blocks", conf_zT_blocks=int(n))
    _count("conf_transposes")                                                     # the shard WAS transposed (block-streamed)


def _confidence_pass(model, T, x_pred, C, i: int):
    """Confidence pass ``i``: one sample's ``_confidence_one`` between ``T.zplan``'s ``begin(i)`` (inside, at the embed) and ``end(i)`` here, once
    the pass's pair rows are dead. ``device_needed_next=False`` on every pass: no statement of this forward reads the device trunk tensor whole
    after a confidence pass — the next pass embeds through ``begin(i + 1)`` (served by the live park), the distogram rows were taken in the trunk
    (``TrunkShard.pd_loc``), ``dict_out["z"]`` becomes :class:`rowpair.ConsumedRows` once the rows are gone (``rowpair._ztrunk_close``)."""
    out = _confidence_one(model, T, x_pred, C, i)
    T.zplan.end(i, device_needed_next=False)
    return out


def confidence_rows(model, T, x_pred, multiplicity: int = 1, run_sequentially: bool = False) -> Dict[str, Any]:
    """Seams 16-17 — ``ConfidenceModule.forward`` on the trunk rows, one diffusion sample at a time (the stock ``run_sequentially`` form for any
    ``multiplicity``; ``run_sequentially=False`` batches the samples in the stock module — the same statements per sample), outputs
    concatenated as stock (confidencev2.py:120-152)."""
    import torch
    C = _core()
    C.D.require_sharded(T.lay, "confidence_rows")
    multiplicity = int(multiplicity)
    _count("conf_samples", multiplicity)
    if multiplicity <= 1:
        return _confidence_pass(model, T, x_pred, C, 0)
    if x_pred.dim() == 4:
        x_pred = x_pred.reshape(-1, x_pred.shape[-2], x_pred.shape[-1])
    out_dicts = [_confidence_pass(model, T, x_pred[sample_idx:sample_idx + 1], C, sample_idx) for sample_idx in range(multiplicity)]
    out_dict: Dict[str, Any] = {}
    for key in out_dicts[0]:
        if key != "pair_chains_iptm":
            out_dict[key] = torch.cat([out[key] for out in out_dicts], dim=0)
        else:
            pair_chains_iptm = {}
            for chain_idx1 in out_dicts[0][key]:
                chains_iptm = {}
                for chain_idx2 in out_dicts[0][key][chain_idx1]:
                    chains_iptm[chain_idx2] = torch.cat([out[key][chain_idx1][chain_idx2] for out in out_dicts], dim=0)
                pair_chains_iptm[chain_idx1] = chains_iptm
            out_dict[key] = pair_chains_iptm
    return out_dict


HEADS: Dict[str, Optional[Callable]] = {"diffusion_conditioning_rows": diffusion_conditioning_rows, "sample_sharded": sample_sharded,
                                        "confidence_rows": confidence_rows}
