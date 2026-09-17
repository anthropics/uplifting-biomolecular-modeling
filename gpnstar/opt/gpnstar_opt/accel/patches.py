"""Exact-mode patches for GPN-Star (songlab-cal/gpn @ 6f28c81b, module gpn.star.model).

Every patch is applied in-place to a loaded *stock* GPNStarForMaskedLM / GPNStarModel and is
designed to leave numerics untouched -- either it runs the very same torch ops on tensors of the
very same shape/layout (so the same CUDA kernels are dispatched), or it only moves data
(gather / copy).  Identity is nevertheless checked on the GPU (the layer-0 self-check at run time, the engage tests), never assumed.

    P0 device_constants : phylogenetic-distance tensors resident on the GPU
                          (stock copies pairwise/in-clade distances host->device on every forward)
    P1 cached_constants : per-batch-shape constants computed ONCE by the stock code path and reused:
                          clade-mean distances (fp64 scatter_add), clade attention mask (one_hot -> host
                          sync in eager), the 1+16 FIRE evolutionary-time-bias MLPs, sinusoidal positions
    P2 source_gather    : source module driven by device-resident index tensors (stock indexes with
                          Python lists -> H2D copies every call, which also forbids CUDA-graph capture);
                          the singleton-clade embedding lookups are fused into one gather (pure copies)
    P3 kv_lut           : column-attention key/value rows of *singleton* clades come from a per-layer
                          (6 x H/2) lookup table -- a singleton clade embedding can take only 6 values
                          (one per token), so key(embed(token)) is precomputed with the stock GEMM;
                          multi-species clades keep their GEMM (on fewer rows)
    (b)/(c) helpers     : fused QKV / KV projections, SDPA backend swap -- for the identity experiments

Contract assumed by P1-P3 (true for all maintained inference paths, gpn.star.inference._target_species):
T = 1 target row and target_species == `target_index` (0 = the reference genome) for every window.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def core_model(model: nn.Module) -> nn.Module:
    """Return the GPNStarModel inside a GPNStarForMaskedLM (or the model itself)."""
    return model.model if hasattr(model, "model") and hasattr(model.model, "phylo_info") else model


@dataclass
class ExactState:
    """Shared per-model state for the patches (attached as model._exact_state)."""

    target_index: int = 0
    clade_species: list[list[int]] = field(default_factory=list)  # stock iteration order
    single_clade_pos: list[int] = field(default_factory=list)  # clade slots that are singletons
    single_species: list[int] = field(default_factory=list)  # their species column, same order
    multi_clade_pos: list[int] = field(default_factory=list)
    multi_species: list[list[int]] = field(default_factory=list)
    # device tensors
    single_species_dev: Tensor | None = None
    single_clade_pos_dev: Tensor | None = None
    multi_clade_pos_dev: Tensor | None = None
    multi_species_dev: list[Tensor] = field(default_factory=list)
    # per-forward stash (written by P2, read by P3)
    single_tokens: Tensor | None = None  # (B, L, S) int64
    multi_source_embeddings: Tensor | None = None  # (B, L, Cm, H) contiguous
    lut_enabled: bool = False
    stash_multi: bool = False
    caches: dict[Any, Any] = field(default_factory=dict)


def _state(model: nn.Module, target_index: int = 0) -> ExactState:
    core = core_model(model)
    st = getattr(core, "_exact_state", None)
    if st is None:
        st = ExactState(target_index=target_index)
        clade_dict = core.phylo_info.clade_dict
        st.clade_species = [list(species) for species in clade_dict.values()]  # == stock order
        for pos, sp in enumerate(st.clade_species):
            if len(sp) > 1:
                st.multi_clade_pos.append(pos)
                st.multi_species.append(sp)
            else:
                st.single_clade_pos.append(pos)
                st.single_species.append(sp[0])
        dev = next(core.parameters()).device
        st.single_species_dev = torch.tensor(st.single_species, dtype=torch.long, device=dev)
        st.single_clade_pos_dev = torch.tensor(st.single_clade_pos, dtype=torch.long, device=dev)
        st.multi_clade_pos_dev = torch.tensor(st.multi_clade_pos, dtype=torch.long, device=dev)
        st.multi_species_dev = [torch.tensor(sp, dtype=torch.long, device=dev) for sp in st.multi_species]
        core._exact_state = st
    return st


# --------------------------------------------------------------------------------------
# P0: device-resident constants
# --------------------------------------------------------------------------------------


def patch_device_constants(model: nn.Module) -> None:
    core = core_model(model)
    dev = next(core.parameters()).device
    pi = core.phylo_info
    pi.phylo_dist_pairwise = pi.phylo_dist_pairwise.to(dev)
    pi.in_clade_phylo_dist = pi.in_clade_phylo_dist.to(dev)
    pi.clade_labels = pi.clade_labels.to(dev)
    _state(model)


# --------------------------------------------------------------------------------------
# P1: cached constants (FIRE biases, clade-mean distances, attention mask, sinusoidal pos)
# --------------------------------------------------------------------------------------


class CachedByShape(nn.Module):
    """Wrap a module whose output depends only on a constant input; cache by input shape/dtype.
    The cached value is produced by the wrapped (stock) module itself, so it is bit-identical to
    what the stock module returns for that input (given deterministic kernels)."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._cache: dict[Any, Tensor] = {}

    def forward(self, x: Tensor) -> Tensor:
        key = (tuple(x.shape), x.dtype, x.device)
        out = self._cache.get(key)
        if out is None:
            out = self.inner(x)
            self._cache[key] = out
        return out


def _model_forward_cached(self, input_ids=None, source_ids=None, target_species=None, output_attentions=False, **kwargs):
    """GPNStarModel.forward with the target-species-dependent constants cached per batch shape.
    Mirrors gpn.star.model.GPNStarModel.forward line by line otherwise."""
    if input_ids is None or source_ids is None or target_species is None:
        raise ValueError("input_ids, source_ids, and target_species are required")
    st: ExactState = self._exact_state
    hidden_states = self.target_embedding(input_ids=input_ids)  # (B, L, T, H)

    clade_dict = self.phylo_info.clade_dict
    in_clade_phylo_dist = self.phylo_info.in_clade_phylo_dist.to(hidden_states.device)
    source_embeddings = self.source_embedding(source_ids, clade_dict, in_clade_phylo_dist)

    if not torch.is_tensor(target_species):
        target_species = torch.as_tensor(target_species)
    key = ("mask", tuple(target_species.shape), hidden_states.dtype)
    cached = st.caches.get(key)
    if cached is None:
        # first call for this batch shape: validate the contract (host sync allowed here), then
        # compute exactly as stock does and keep the result.
        ts = target_species.to(hidden_states.device)
        if not bool((ts == st.target_index).all()):
            raise ValueError("exact-mode cache assumes a constant target_species == target_index")
        phylo_dist_pairwise = self.phylo_info.phylo_dist_pairwise.to(hidden_states.device)
        clade_labels = self.phylo_info.clade_labels.to(hidden_states.device)
        phylo_dist = phylo_dist_pairwise[ts]
        phylo_dist = self.compute_clade_means(phylo_dist, clade_labels, len(clade_dict))
        target_clades = clade_labels[ts]
        attention_mask = F.one_hot(target_clades, num_classes=phylo_dist.size(-1)).to(hidden_states.dtype)
        attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask[:, None, None, :, :]
        cached = (phylo_dist, attention_mask, ts.clone())
        st.caches[key] = cached
    elif getattr(st, "check_inputs", True) and not (hidden_states.is_cuda and torch.cuda.is_current_stream_capturing()):
        # every later eager call: re-validate the contract (one small host sync).  A host sync is illegal while a CUDA
        # graph is being captured; GraphRunner validates target_species of every batch on the host side before replay.
        ts = target_species.to(hidden_states.device)
        if ts.shape != cached[2].shape or not torch.equal(ts, cached[2]):
            raise ValueError("gpnstar_exact: target_species differs from the value the exact-mode constants were built for "
                             f"(expected all == {st.target_index}); refusing to serve cached constants. Rebuild with make_exact(..., target_index=...) "
                             "or run the stock model for this batch.")
    phylo_dist, attention_mask = cached[0], cached[1]
    return self.encoder(
        hidden_states,
        source_embeddings,
        phylo_dist=phylo_dist,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        **kwargs,
    )


def patch_cached_constants(model: nn.Module, target_index: int = 0) -> None:
    core = core_model(model)
    patch_device_constants(model)
    st = _state(model, target_index)
    st.target_index = target_index
    # FIRE time-bias MLPs: source module (input = in-clade distances, constant) and every layer's
    # column attention (input = clade-mean distances, constant given target species).
    src = core.source_embedding
    if not isinstance(src.embed_positions, CachedByShape):
        src.embed_positions = CachedByShape(src.embed_positions)
    for layer in core.encoder.layer:
        ca = layer.attention.col_attention
        if not isinstance(ca.embed_positions, CachedByShape):
            ca.embed_positions = CachedByShape(ca.embed_positions)
    # sinusoidal positions: RoFormerSinusoidalPositionalEmbedding.forward(shape, past_len) -> cache by shape
    enc = core.encoder
    if not hasattr(enc.embed_positions, "_exact_cached"):
        orig = enc.embed_positions
        cache: dict[Any, Tensor] = {}
        orig_forward = orig.forward

        def cached_forward(input_ids_shape, past_key_values_length: int = 0, position_ids=None):
            k = (tuple(input_ids_shape)[:2], int(past_key_values_length))
            v = cache.get(k)
            if v is None or position_ids is not None:
                v = orig_forward(input_ids_shape, past_key_values_length, position_ids)
                if position_ids is None:
                    cache[k] = v
            return v

        orig.forward = cached_forward  # instance attribute shadows the bound method
        orig._exact_cached = True
    # model-level forward with cached clade means / mask
    core.forward = _model_forward_cached.__get__(core, type(core))


# --------------------------------------------------------------------------------------
# P2: source module with device index tensors + fused singleton gather
# --------------------------------------------------------------------------------------


def _source_forward_gather(self, source_ids, clade_dict, in_clade_phylo_dist):
    """GPNStarSourceModule.forward; same per-clade attn_pool calls (same shapes -> same kernels),
    singleton clades via ONE embedding gather, all indexing with device tensors."""
    st: ExactState = self._exact_state
    in_clade_time_bias = self.embed_positions(in_clade_phylo_dist[None, None, :] * self.time_scale)
    B, L, N = source_ids.shape
    src_int = source_ids.to(torch.int)  # stock casts each gathered slice; cast-then-gather == gather-then-cast
    C = len(st.clade_species)
    H = self.embed.embedding_dim

    single_tok = src_int.index_select(-1, st.single_species_dev)  # (B, L, S) int32, clade order
    out = torch.empty((B, L, C, H), dtype=self.embed.weight.dtype, device=src_int.device)  # the stage's one long-lived tensor, allocated first
    single_emb = self.embed(single_tok)  # (B, L, S, H)  == stack of stock per-species lookups
    out.index_copy_(2, st.single_clade_pos_dev, single_emb)
    del single_emb  # (B, L, S, H), the largest transient of the stage: released before the clade pooling runs, not at function exit

    bias_key = ("ictb", tuple(in_clade_time_bias.shape))  # in-clade distances are a model constant
    biases = st.caches.get(bias_key)
    if biases is None:
        biases = [in_clade_time_bias.index_select(-1, sp) for sp in st.multi_species_dev]
        st.caches[bias_key] = biases
    stash = st.lut_enabled or st.stash_multi
    multi = None  # the multi-species clade embeddings for P3/P3b/P4, written in place as they are pooled (no list of them + a stacked copy alive at once)
    for j, (pos, sp_dev, bias) in enumerate(zip(st.multi_clade_pos, st.multi_species_dev, biases)):
        pooled = self.attn_pool(src_int.index_select(-1, sp_dev), bias)  # (B, L, H)
        out[:, :, pos, :] = pooled  # (under autocast pooled may be half; copy casts, as torch.stack would not -- fast-mode only)
        if stash:
            if multi is None:
                multi = pooled.new_empty((B, L, len(st.multi_clade_pos), H))  # pooled's dtype, as torch.stack of the pooled list had
            multi[:, :, j, :] = pooled  # a copy of the same values torch.stack copied
        del pooled
    if stash:
        st.single_tokens = single_tok.long()  # (B, L, S) token ids of singleton-clade species
        st.multi_source_embeddings = multi if multi is not None else out.new_empty((B, L, 0, H))
    return out


def patch_source_gather(model: nn.Module) -> None:
    core = core_model(model)
    st = _state(model)
    src = core.source_embedding
    src._exact_state = st
    src.forward = _source_forward_gather.__get__(src, type(src))


# --------------------------------------------------------------------------------------
# P3: singleton-clade K/V lookup tables for the column cross-attention
# --------------------------------------------------------------------------------------


def _sdpa_math(q, k, v, mask, scale, dropout_p=0.0):
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=scale)


def _col_forward_lut(self, hidden_states, source_embeddings, attention_mask=None, evol_time_bias=None, output_attentions=False):
    """GPNStarColCrossAttention.forward with singleton-clade K/V from lookup tables."""
    if attention_mask is None or evol_time_bias is None:
        raise ValueError("attention_mask and evol_time_bias are required")
    if output_attentions:
        return self._stock_forward(hidden_states, source_embeddings, attention_mask, evol_time_bias, output_attentions)
    st: ExactState = self._exact_state
    lut = self._lut
    query_layer = self.transpose_for_scores(self.query(hidden_states))
    if st.lut_enabled and lut is not None:
        B, L, C, H = source_embeddings.shape
        src_multi = st.multi_source_embeddings  # (B, L, Cm, H)
        tok = st.single_tokens  # (B, L, S)
        Ah = self.all_head_size
        if lut.get("fuse_kv"):
            kv_multi = F.linear(src_multi, lut["w_kv"], lut["b_kv"])  # (B, L, Cm, 2Ah)
            k_multi, v_multi = kv_multi[..., :Ah], kv_multi[..., Ah:]
        else:
            k_multi = self.key(src_multi)
            v_multi = self.value(src_multi)
        K = k_multi.new_empty((B, L, C, Ah))
        V = v_multi.new_empty((B, L, C, Ah))
        if src_multi.shape[2] > 0:
            K.index_copy_(2, st.multi_clade_pos_dev, k_multi)
            V.index_copy_(2, st.multi_clade_pos_dev, v_multi)
        lk, lv = lut["k"], lut["v"]
        if lk.dtype != K.dtype:  # autocast (fast-mode experiments): LUT built in fp32, GEMM output in half
            lk, lv = lk.to(K.dtype), lv.to(V.dtype)
        K.index_copy_(2, st.single_clade_pos_dev, lk[tok])
        V.index_copy_(2, st.single_clade_pos_dev, lv[tok])
        key_layer = self.transpose_for_scores(K)
        value_layer = self.transpose_for_scores(V)
    else:
        key_layer = self.transpose_for_scores(self.key(source_embeddings))
        value_layer = self.transpose_for_scores(self.value(source_embeddings))

    attention_mask = attention_mask + evol_time_bias.to(query_layer.dtype)
    context_layer = _sdpa_math(
        query_layer, key_layer, value_layer, attention_mask,
        scale=1 / math.sqrt(self.attention_head_size),
        dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
    )
    context_layer = context_layer.transpose(-2, -3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return (context_layer,)


@torch.no_grad()
def build_kv_luts(model: nn.Module, mode: str = "tiled", ref_rows: int | None = None, fuse_kv: bool = False) -> None:
    """Precompute per-layer singleton-clade K/V tables.

    mode="small": lut = key(E) with E the (6, H) source token-embedding table (GEMM with M=6).
    mode="tiled": lut = key(tile(E) to `ref_rows` rows)[:6] -- the GEMM then has the same
                  (M, N, K) as the stock projection of the full (B*L*C, H) source-embedding matrix,
                  so cuBLAS dispatches the same kernel and every row is accumulated in the same order.
    """
    core = core_model(model)
    st = _state(model)
    E = core.source_embedding.embed.weight  # (6, H)
    if mode == "tiled":
        if ref_rows is None:
            raise ValueError("ref_rows (= B*L*C of the target workload) required for mode='tiled'")
        reps = -(-ref_rows // E.shape[0])
        X = E.repeat(reps, 1)[:ref_rows].contiguous()
    elif mode == "small":
        X = E.contiguous()
    else:
        raise ValueError(mode)
    for layer in core.encoder.layer:
        ca = layer.attention.col_attention.self
        k = ca.key(X)[: E.shape[0]].contiguous().clone()
        v = ca.value(X)[: E.shape[0]].contiguous().clone()
        lut = {"k": k, "v": v, "mode": mode, "ref_rows": ref_rows, "fuse_kv": fuse_kv}
        if fuse_kv:
            lut["w_kv"] = torch.cat([ca.key.weight, ca.value.weight], 0).contiguous()
            lut["b_kv"] = torch.cat([ca.key.bias, ca.value.bias], 0).contiguous()
        ca._lut = lut
    del X


def patch_kv_lut(model: nn.Module, mode: str = "tiled", ref_rows: int | None = None, fuse_kv: bool = False) -> None:
    """Install P3 (requires P2). Call again with a new ref_rows when the workload shape changes
    if mode='tiled'."""
    core = core_model(model)
    st = _state(model)
    if not hasattr(core.source_embedding, "_exact_state"):
        patch_source_gather(model)
    st.lut_enabled = True
    st.stash_multi = True
    for layer in core.encoder.layer:
        ca = layer.attention.col_attention.self
        if not hasattr(ca, "_stock_forward"):
            ca._stock_forward = ca.forward
            ca._exact_state = st
            ca._lut = None
            ca.forward = _col_forward_lut.__get__(ca, type(ca))
    build_kv_luts(model, mode=mode, ref_rows=ref_rows, fuse_kv=fuse_kv)


def disable_kv_lut(model: nn.Module) -> None:
    st = _state(model)
    st.lut_enabled = False


# --------------------------------------------------------------------------------------
# (b)/(c): attention-backend swap and fused projections for the ROW self-attention
# --------------------------------------------------------------------------------------


def _row_forward_variant(self, hidden_states, attention_mask=None, sinusoidal_pos=None, output_attentions=False):
    """GPNStarRowSelfAttention.forward with selectable SDPA backend / fused QKV (T must be 1 for
    the 4-D backends). self._variant in {"math", "math4d", "efficient", "flash", "cudnn"};
    self._fuse_qkv bool."""
    if output_attentions or hidden_states.shape[1] != 1:
        return self._stock_forward(hidden_states, attention_mask, sinusoidal_pos, output_attentions)
    variant = self._variant
    if self._fuse_qkv:
        qkv = F.linear(hidden_states[:, :1, ...], self._w_qkv, self._b_qkv)
        Ah = self.all_head_size
        query_layer = self.transpose_for_scores(qkv[..., :Ah])
        key_layer = self.transpose_for_scores(qkv[..., Ah : 2 * Ah])
        value_layer = self.transpose_for_scores(qkv[..., 2 * Ah :])
    else:
        query_layer = self.transpose_for_scores(self.query(hidden_states[:, :1, ...]))
        key_layer = self.transpose_for_scores(self.key(hidden_states[:, :1, ...]))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
    if sinusoidal_pos is not None:
        if self.rotary_value:
            query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer)
        else:
            query_layer, key_layer = self.apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer)
    scale = 1 / math.sqrt(self.attention_head_size)
    dropout_p = self.attention_probs_dropout_prob if self.training else 0.0
    if variant == "math":
        context_layer = _sdpa_math(query_layer, key_layer, value_layer, attention_mask, scale, dropout_p)
    else:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend = {
            "math4d": SDPBackend.MATH,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
        }[variant]
        B = query_layer.shape[0]
        q4 = query_layer[:, 0]  # (B, A, L, D)
        k4 = key_layer[:, 0]
        v4 = value_layer[:, 0]
        m4 = attention_mask[:, 0] if attention_mask is not None else None
        with sdpa_kernel([backend]):
            ctx4 = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=m4, dropout_p=dropout_p, scale=scale)
        context_layer = ctx4[:, None]  # (B, 1, A, L, D)
    context_layer = context_layer.transpose(-2, -3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return (context_layer,)


def patch_row_attention(model: nn.Module, variant: str = "math", fuse_qkv: bool = False) -> None:
    core = core_model(model)
    for layer in core.encoder.layer:
        ra = layer.attention.row_attention.self
        if not hasattr(ra, "_stock_forward"):
            ra._stock_forward = ra.forward
            ra.forward = _row_forward_variant.__get__(ra, type(ra))
        ra._variant = variant
        ra._fuse_qkv = fuse_qkv
        if fuse_qkv and not hasattr(ra, "_w_qkv"):
            ra._w_qkv = torch.cat([ra.query.weight, ra.key.weight, ra.value.weight], 0).contiguous()
            ra._b_qkv = torch.cat([ra.query.bias, ra.key.bias, ra.value.bias], 0).contiguous()


def _col_forward_backend(self, hidden_states, source_embeddings, attention_mask=None, evol_time_bias=None, output_attentions=False):
    """Column cross-attention with a 4-D SDPA backend (efficient / math4d); K/V as stock or LUT."""
    variant = self._col_variant
    if variant == "math" or output_attentions:
        return _col_forward_lut(self, hidden_states, source_embeddings, attention_mask, evol_time_bias, output_attentions)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    st: ExactState = self._exact_state
    lut = getattr(self, "_lut", None)
    query_layer = self.transpose_for_scores(self.query(hidden_states))  # (B, L, A, T, D)
    if st.lut_enabled and lut is not None:
        B, L, C, H = source_embeddings.shape
        src_multi = st.multi_source_embeddings
        tok = st.single_tokens
        Ah = self.all_head_size
        k_multi = self.key(src_multi)
        v_multi = self.value(src_multi)
        K = k_multi.new_empty((B, L, C, Ah))
        V = v_multi.new_empty((B, L, C, Ah))
        if src_multi.shape[2] > 0:
            K.index_copy_(2, st.multi_clade_pos_dev, k_multi)
            V.index_copy_(2, st.multi_clade_pos_dev, v_multi)
        lk, lv = lut["k"], lut["v"]
        if lk.dtype != K.dtype:
            lk, lv = lk.to(K.dtype), lv.to(V.dtype)
        K.index_copy_(2, st.single_clade_pos_dev, lk[tok])
        V.index_copy_(2, st.single_clade_pos_dev, lv[tok])
        key_layer = self.transpose_for_scores(K)
        value_layer = self.transpose_for_scores(V)
    else:
        key_layer = self.transpose_for_scores(self.key(source_embeddings))
        value_layer = self.transpose_for_scores(self.value(source_embeddings))
    attention_mask = attention_mask + evol_time_bias.to(query_layer.dtype)  # (B, 1, 1, T, C)
    B, L, A, T, D = query_layer.shape
    C = key_layer.shape[-2]
    q4 = query_layer.reshape(B * L, A, T, D)
    k4 = key_layer.reshape(B * L, A, C, D)
    v4 = value_layer.reshape(B * L, A, C, D)
    m4 = attention_mask.expand(B, L, attention_mask.shape[2], T, C).reshape(B * L, attention_mask.shape[2], T, C)
    backend = {"math4d": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION}[variant]
    with sdpa_kernel([backend]):
        ctx4 = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=m4, dropout_p=0.0, scale=1 / math.sqrt(self.attention_head_size))
    context_layer = ctx4.reshape(B, L, A, T, D)
    context_layer = context_layer.transpose(-2, -3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return (context_layer,)


def patch_col_attention_backend(model: nn.Module, variant: str = "math") -> None:
    core = core_model(model)
    st = _state(model)
    for layer in core.encoder.layer:
        ca = layer.attention.col_attention.self
        if not hasattr(ca, "_stock_forward"):
            ca._stock_forward = ca.forward
            ca._exact_state = st
            ca._lut = None
        ca._col_variant = variant
        ca.forward = _col_forward_backend.__get__(ca, type(ca))


# --------------------------------------------------------------------------------------
# convenience: apply an exact-mode "level"
# --------------------------------------------------------------------------------------


def apply_exact(model: nn.Module, level: int, *, lut_mode: str = "tiled", ref_rows: int | None = None, fuse_kv: bool = False) -> nn.Module:
    """level 0: stock; 1: P0; 2: P0+P1; 3: +P2; 4: +P3 (K/V LUT)."""
    model.eval()
    if level >= 1:
        patch_device_constants(model)
    if level >= 2:
        patch_cached_constants(model)
    if level >= 3:
        patch_source_gather(model)
    if level >= 4:
        patch_kv_lut(model, mode=lut_mode, ref_rows=ref_rows, fuse_kv=fuse_kv)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# --------------------------------------------------------------------------------------
# P3b / P4: unified source-side K/V path -- singleton rows + (optionally deduplicated)
# multi-species clade rows projected in ONE GEMM per layer, then gathered into (B, L, C, Ah)
# --------------------------------------------------------------------------------------
# Rationale: the column cross-attention keys/values depend only on the alignment column
# (species tokens), not on the target hidden state.  A singleton clade row can take 6 values;
# a multi-species clade row depends only on its k-tuple of tokens, which in real alignments
# (and across overlapping windows inside a batch) is massively redundant.  Stock recomputes
# key()/value() for all B*L*C rows in every layer (the bulk of the forward's FLOPs).
#
# Exactness rests on two measured properties of the fp32 cuBLAS GEMM used by nn.Linear:
#   (i)  identical input rows give identical output rows irrespective of row position, and
#   (ii) the per-row result does not depend on M (number of rows) once M >= min_rows
#        (kernel-selection plateau) -- probed empirically over M; min_rows is a parameter.


def _encode_tuples(tok: Tensor, powers: Tensor) -> Tensor:
    """(R, k) small-int tokens -> (R,) int64 codes, base 6 (k <= 24)."""
    return (tok.to(torch.int64) * powers).sum(-1)


def _dedup_codes(codes: Tensor) -> tuple[Tensor, Tensor]:
    """Deterministic dedup of a 1-D int64 tensor.
    Returns rep (U,) = flat index of the first occurrence of each unique code (ascending code
    order) and inv (R,) = unique id of every element.  Involves one host sync (U is dynamic)."""
    sorted_codes, order = torch.sort(codes, stable=True)
    is_first = torch.ones_like(sorted_codes, dtype=torch.bool)
    is_first[1:] = sorted_codes[1:] != sorted_codes[:-1]
    rep = order[is_first]
    group = torch.cumsum(is_first.to(torch.int64), 0) - 1
    inv = torch.empty_like(group)
    inv[order] = group
    return rep, inv


def _source_forward_unified(self, source_ids, clade_dict, in_clade_phylo_dist):
    """P2 source forward + preparation of the unified K/V input matrix and gather index.
    st.dedup=False -> P3b (static shapes, CUDA-graph safe); st.dedup=True -> P4 (dynamic U)."""
    st: ExactState = self._exact_state
    out = _source_forward_gather(self, source_ids, clade_dict, in_clade_phylo_dist)  # stashes multi embeddings
    B, L, N = source_ids.shape
    C = len(st.clade_species)
    H = self.embed.embedding_dim
    R = B * L
    E = self.embed.weight  # (6, H): the only values a singleton-clade embedding can take
    dev = source_ids.device
    single_tok = st.single_tokens  # (B, L, S) long
    multi = st.multi_source_embeddings  # (B, L, Cm, H)
    if multi.dtype != E.dtype:  # only under autocast (fast-mode experiments)
        multi = multi.to(E.dtype)
    Cm = multi.shape[2]
    M_stock = R * C
    st.kv_shape_used = (int(B), int(L))
    mode = st.dedup  # False (P3b) | True (P4 eager dedup) | "static" (P4s static-capacity dedup)
    override = st.mode_override.get(M_stock) if st.mode_override else None
    if st.force_stock_once:  # one-shot request (GraphRunner recomputing an overflowed/foreign batch eagerly): stock projections
        st.force_stock_once = False
        override = "stock"
    if override == "stock":
        # terminal fallback for this shape (every reduced-GEMM option was rejected by the self-check): do NOT build
        # the reduced input at all -> stock projections in every layer, ~stock memory, P0-P2 speed.  Still exact.
        st.kv_input = None
        st.kv_index = None
        st.kv_m_stock = M_stock
        st.kv_mode_used = "stock"
        st.kv_fallback_full = True
        st.multi_source_embeddings = None
        return out
    if override == "p3b":
        mode = False  # dedup was rejected for this shape -> static unified K/V (P3b), which is validated on its own
    st.kv_mode_used = mode
    oom = False
    try:  # the reduced input is EXTRA memory beside the stock tensors: if it does not fit, this shape runs the stock projections (exact) by rule
        return _build_reduced_input(self, st, out, source_ids, mode, B, L, C, H, R, E, dev, single_tok, multi, Cm, M_stock)
    except torch.cuda.OutOfMemoryError:
        oom = True  # handled below, once the failed attempt's tensors are released
    if oom:
        del multi
        _memory_fallback(st, M_stock, where="reduced K/V input")
    return out


# ---- memory plan of the reduced K/V route (decided per batch shape BEFORE the route allocates its per-forward workspace)
MEMORY_STOCK_LAYER_FACTOR = 5  # the stock column attention holds about five K-sized tensors at its peak (K, V, their per-head copies, the scaled key): what one layer needs on ANY route
MEMORY_ALLOC_OVERHEAD = {"default": 1.16, "expandable_segments": 1.05}  # reserved/allocated ratio the CUDA caching allocator reaches on this workload, per allocator layout
MEMORY_MARGIN_BYTES = 1 << 30
MEMORY_CHECK_SLICE_BYTES = 1 << 30  # the layer-0 self-check compares through the gather in slices of at most this size
_ALLOCATOR_SWITCHED = [False]  # set when an out-of-memory backstop turned expandable segments on for this process


def _reduced_route_bytes(rows_padded: int, hidden: int, out_features: int, m_stock: int, validate: bool, elem: int = 4) -> dict:
    """Bytes the reduced K/V route allocates BEYOND the stock projections for one forward of a shape: the reduced input X (alive through
    every layer), one reduced projection at a time, and - on the first forward of a (rows, stock rows) pair only - the self-check slice."""
    x = int(rows_padded) * int(hidden) * int(elem)
    proj = int(rows_padded) * int(out_features) * int(elem)
    check = min(MEMORY_CHECK_SLICE_BYTES, int(m_stock) * int(out_features) * int(elem)) if validate else 0
    return {"x_bytes": x, "projection_bytes": proj, "check_bytes": check, "extra_bytes": x + proj + check}


def _stock_layer_bytes(m_stock: int, out_features: int, elem: int = 4, factor: int = MEMORY_STOCK_LAYER_FACTOR) -> int:
    """Transient bytes the column attention of ONE layer allocates for a shape with m_stock K/V rows: `factor` K-sized tensors
    (stock's SDPA-math path: MEMORY_STOCK_LAYER_FACTOR; the fused K/V route states its own, fusedkv.FUSED_LAYER_FACTOR)."""
    return int(round(float(factor) * int(m_stock) * int(out_features) * int(elem)))


def _memory_decision(extra_bytes: int, stock_layer_bytes: int, budget_bytes: int, overhead: float, margin_bytes: int = MEMORY_MARGIN_BYTES,
                     releasable_bytes: int = 0, count_releasable: bool = False):
    """(fits, detail). The reduced route runs when what it adds, on top of what the stock attention needs for the shape anyway, fits the
    memory obtainable right now once the allocator's overhead and a margin are counted; otherwise the shape runs the stock projections.
    releasable_bytes = buffers of the source stage that are released before the layers run: they count as obtainable only where the
    allocator hands a released block back whatever the next request's size (expandable segments); under the default layout a released
    block of another size is not reliably reusable for the layers' K/V tensors and is left out."""
    budget = int(budget_bytes) + (int(releasable_bytes) if count_releasable else 0)
    need = int(round((int(extra_bytes) + int(stock_layer_bytes)) * float(overhead))) + int(margin_bytes)
    return need <= budget, {"extra_bytes": int(extra_bytes), "stock_layer_bytes": int(stock_layer_bytes), "need_bytes": int(need), "budget_bytes": budget,
                            "measured_budget_bytes": int(budget_bytes), "releasable_bytes": int(releasable_bytes), "overhead": float(overhead), "margin_bytes": int(margin_bytes)}


def _allocator_kind() -> str:
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") + "," + os.environ.get("PYTORCH_ALLOC_CONF", "")
    return "expandable_segments" if (_ALLOCATOR_SWITCHED[0] or re.search(r"expandable_segments\s*:\s*True", conf)) else "default"


def _memory_budget(dev) -> int:
    """Bytes obtainable for new allocations on dev: free device memory plus this process's reserved-but-unused allocator cache."""
    free, _total = torch.cuda.mem_get_info(dev)
    return int(free) + max(0, int(torch.cuda.memory_reserved(dev)) - int(torch.cuda.memory_allocated(dev)))


def _memory_plan(st, rows_padded: int, H: int, M_stock: int, dev, elem: int, validate: bool, releasable_bytes: int = 0):
    """(fits, plan) for the reduced route at this shape on this device, from the known sizes and the memory obtainable now."""
    if getattr(dev, "type", "cuda") != "cuda" and st.debug_memory_budget is None:
        return True, {}
    out_features = int(st.kv_out_features or H)
    r = _reduced_route_bytes(rows_padded, H, out_features, M_stock, validate, elem)
    budget = int(st.debug_memory_budget) if st.debug_memory_budget is not None else _memory_budget(dev)
    kind = _allocator_kind()
    fits, plan = _memory_decision(r["extra_bytes"], _stock_layer_bytes(M_stock, out_features, elem, getattr(st, "layer_factor", MEMORY_STOCK_LAYER_FACTOR)), budget, MEMORY_ALLOC_OVERHEAD[kind],
                                  releasable_bytes=int(releasable_bytes), count_releasable=(kind == "expandable_segments" and st.debug_memory_budget is None))
    plan.update(r)
    plan.update({"rows": int(rows_padded), "allocator": kind})
    return fits, plan


def _released_before_layers(out: Tensor, multi, blocks) -> int:
    """Bytes this stage still holds that are released before any layer runs: the storage behind the stashed multi-clade embeddings (a view
    can pin a larger base) unless it is the source embeddings' own, and the de-dup row blocks (copied into X, which the plan counts)."""
    keep = out.untyped_storage().data_ptr()
    seen, total = {keep}, 0
    for t in ([multi] if multi is not None else []) + list(blocks):
        try:
            s = t.untyped_storage()
            if s.data_ptr() not in seen:
                seen.add(s.data_ptr())
                total += int(s.nbytes())
        except Exception:  # noqa: BLE001
            pass
    return total


def _memory_stepaside(st, m_stock: int, plan: dict) -> None:
    """The plan says the reduced route does not fit at this shape: nothing of it is allocated, the shape runs the STOCK projections from
    now on (exact; the levers' other savings remain) and says so once. kv_mode_report(model)['memory_plans'] keeps the numbers."""
    st.kv_input = None
    st.kv_index = None
    st.multi_source_embeddings = None
    st.kv_fallback_full = True
    st.kv_mode_used = "stock"
    st.kv_m_stock = m_stock
    if st.mode_override is None:
        st.mode_override = {}
    first = st.mode_override.get(m_stock) != "stock"
    st.mode_override[m_stock] = "stock"
    if st.memory_fallbacks is None:
        st.memory_fallbacks = {}
    st.memory_fallbacks[m_stock] = st.memory_fallbacks.get(m_stock, 0) + 1
    if st.memory_plans is None:
        st.memory_plans = {}
    st.memory_plans[m_stock] = dict(plan)
    if first:
        B_, L_ = getattr(st, "kv_shape_used", ("?", "?"))
        g = 2 ** 30
        warnings.warn(f"gpnstar_exact: at batch shape B={B_} x L={L_} ({m_stock:,} stock K/V rows) the reduced K/V route needs about "
                      f"{plan.get('need_bytes', 0) / g:.1f} GiB ({plan.get('extra_bytes', 0) / g:.1f} GiB beyond the stock attention's own "
                      f"{plan.get('stock_layer_bytes', 0) / g:.1f} GiB, allocator layout '{plan.get('allocator')}') and {plan.get('budget_bytes', 0) / g:.1f} GiB is "
                      "obtainable -> this shape runs the STOCK projections (exact; stock speed and memory for the K/V projections). A smaller batch, "
                      "or PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set before the process starts, keeps the reduced route.", RuntimeWarning, stacklevel=2)


def _build_reduced_input(self, st, out, source_ids, mode, B, L, C, H, R, E, dev, single_tok, multi, Cm, M_stock):
    """The reduced K/V GEMM input X (distinct rows) and the gather index idx (B, L, C) for this forward, written in place: X is
    allocated once at its padded row count and filled block by block (no concatenation copy, no tiled duplicate)."""
    idx = torch.empty((B, L, C), dtype=torch.long, device=dev)
    # singleton slots -> rows 0..5 of X
    idx.index_copy_(2, st.single_clade_pos_dev, single_tok)
    blocks = [E]
    offset = E.shape[0]
    multi_view = None  # P3b: the multi rows as a (Cm, R, H) strided view of `multi`, copied straight into X below
    if not mode:
        # P3b: all multi rows, clade-major: row id = offset + j*R + (b*L + l)
        if Cm > 0:
            multi_view = multi.permute(2, 0, 1, 3).reshape(Cm, R, H)
            base = torch.arange(R, device=dev).view(B, L, 1) + offset
            idx.index_copy_(2, st.multi_clade_pos_dev, base + torch.arange(Cm, device=dev).view(1, 1, Cm) * R)
        st.n_unique_rows = offset + Cm * R
    elif mode != "static":
        # P4: dedup rows of each multi clade by its species-token tuple (dynamic U -> eager only)
        src = source_ids
        if st.code_powers is None:
            st.code_powers = [(6 ** torch.arange(len(sp), device=dev, dtype=torch.int64)) if len(sp) <= 24 else None for sp in st.multi_species]
        multi_flat = multi.reshape(R, Cm, H)
        u_counts = []
        for j, (sp_dev, pw) in enumerate(zip(st.multi_species_dev, st.code_powers)):
            pos = st.multi_clade_pos[j]
            if pw is None:  # clade too large to encode -> keep all rows
                blocks.append(multi_flat[:, j, :])
                idx[:, :, pos] = torch.arange(R, device=dev).view(B, L) + offset
                offset += R
                u_counts.append(R)
                continue
            codes = _encode_tuples(src.index_select(-1, sp_dev).reshape(R, -1), pw)
            rep, inv = _dedup_codes(codes)
            rows_j = multi_flat[rep, j, :]
            # A representative stands for a row only if the stock module produced the SAME values for both (identical species tokens
            # give identical pooled embeddings up to the pooling kernel's own rounding, which can differ for a few rows of a large
            # batch): rows that differ from their representative, bit for bit, are kept as rows of their own.
            differs = (multi_flat[:, j, :] != rows_j.index_select(0, inv)).any(-1)
            n_diff = int(differs.sum())
            if n_diff:
                own = differs.nonzero().squeeze(1)
                inv = inv.clone()
                inv[own] = int(rep.shape[0]) + torch.arange(n_diff, device=dev)
                rows_j = torch.cat([rows_j, multi_flat[own, j, :]], 0)
            blocks.append(rows_j)
            idx[:, :, pos] = inv.view(B, L) + offset
            U = int(rows_j.shape[0])
            offset += U
            u_counts.append(U)
        st.n_unique_rows = offset
        st.last_unique_counts = u_counts
    if mode == "static":
        # P4s: fixed-capacity dedup -- no host sync, static shapes, CUDA-graph capturable.
        # Per multi clade j a slab of cap_j rows (+1 trash row) inside one preallocated X buffer.
        # Unique ids come from a stable sort + cumsum; first occurrences are written to their slab row,
        # later duplicates to the trash row; ids >= cap_j (capacity overflow) raise a device flag that the
        # caller must check (results of that forward are then NOT valid -> rerun with the eager path).
        src = source_ids
        if st.code_powers is None:
            st.code_powers = [(6 ** torch.arange(len(sp), device=dev, dtype=torch.int64)) if len(sp) <= 24 else None for sp in st.multi_species]
        key = (B, L)
        plan = st.static_plans.get(key)
        if plan is not None and st.kv_validation.get((int(plan["X"].shape[0]), M_stock)) is False:
            plan = None  # padded size failed the kernel-equivalence check -> rebuild with the next ladder value
            st.static_plans.pop(key, None)
        if plan is None:
            caps = st.static_caps.get(key)
            if caps is None:
                raise RuntimeError(f"dedup='static' needs capacities for shape {key}: call set_static_capacities() first")
            caps = [min(int(c), R) if st.code_powers[j] is not None else R for j, c in enumerate(caps)]
            offs, o = [], E.shape[0]
            for c in caps:
                offs.append(o)
                o += c + 1  # +1 trash row
            total = _padded_rows(st, o, M_stock)
            with torch.inference_mode(False), torch.no_grad():  # normal (non-inference) tensors: reusable across calls / modes
                Xbuf = torch.zeros((max(total, o), H), dtype=E.dtype, device=dev)
                plan = {"caps": caps, "offs": offs, "rows": o, "X": Xbuf, "overflow": torch.zeros((), dtype=torch.bool, device=dev),
                        "n_unique": torch.zeros((len(caps),), dtype=torch.int64, device=dev)}
            st.static_plans[key] = plan
        Xbuf = plan["X"]
        Xbuf[: E.shape[0]].copy_(E)
        multi_flat = multi.reshape(R, Cm, H)
        ar = None
        for j, (sp_dev, pw) in enumerate(zip(st.multi_species_dev, st.code_powers)):
            pos = st.multi_clade_pos[j]
            cap, off = plan["caps"][j], plan["offs"][j]
            slab = Xbuf[off : off + cap + 1]
            if pw is None or cap >= R:  # no dedup for this clade: identity mapping (static)
                slab[:R].copy_(multi_flat[:, j, :])
                if ar is None:
                    ar = torch.arange(R, device=dev)
                idx[:, :, pos] = ar.view(B, L) + off
                plan["n_unique"][j].fill_(R)  # scalar fill (no H2D copy: CUDA-graph safe)
                continue
            codes = _encode_tuples(src.index_select(-1, sp_dev).reshape(R, -1), pw)
            sorted_codes, order = torch.sort(codes, stable=True)
            is_first = torch.ones_like(sorted_codes, dtype=torch.bool)
            is_first[1:] = sorted_codes[1:] != sorted_codes[:-1]
            group = torch.cumsum(is_first.to(torch.int64), 0) - 1  # unique id per sorted element
            plan["n_unique"][j].copy_(group[-1] + 1)
            torch.logical_or(plan["overflow"], group[-1] >= cap, out=plan["overflow"])
            gid = group.clamp(max=cap - 1)
            dest = torch.where(is_first, gid, torch.full_like(gid, cap))  # duplicates -> trash row `cap`
            slab.index_copy_(0, dest, multi_flat.index_select(0, order)[:, j, :])
            inv = torch.empty_like(gid)
            inv[order] = gid
            idx[:, :, pos] = inv.view(B, L) + off
        X = Xbuf
        st.n_unique_rows = plan["rows"]
    else:
        U = sum(int(b.shape[0]) for b in blocks) + (Cm * R if multi_view is not None else 0)
        target = _padded_rows(st, U, M_stock)
        fits, plan = _memory_plan(st, max(U, target), H, M_stock, dev, E.element_size(),
                                  bool(st.validate and st.kv_validation.get((max(U, target), M_stock)) is None),
                                  releasable_bytes=_released_before_layers(out, multi, blocks[1:]))
        st.memory_plan_last = plan
        if not fits:  # decided before anything of the route is allocated: this shape runs the stock projections, by name
            del blocks, multi_view
            _memory_stepaside(st, M_stock, plan)
            return out
        X = torch.empty((max(U, target), H), dtype=E.dtype, device=dev)
        o = 0
        for blk in blocks:
            X[o : o + blk.shape[0]].copy_(blk)
            o += int(blk.shape[0])
        del blocks
        if multi_view is not None:
            X[o : o + Cm * R].view(Cm, R, H).copy_(multi_view)  # one strided copy, no intermediate tensor
            o += Cm * R
        filled = U
        while filled < X.shape[0]:  # pad (by tiling: row i = row i mod U) to a row count whose GEMM kernel is validated against stock's
            n = min(filled, int(X.shape[0]) - filled)
            X[filled : filled + n].copy_(X[:n])
            filled += n
    st.kv_input = X.contiguous()
    st.kv_index = idx
    st.kv_m_stock = M_stock
    st.kv_fallback_full = False  # per-forward flag, set by layer 0 if validation fails
    st.multi_source_embeddings = None  # not needed downstream in the unified path (frees B*L*Cm*H floats)
    return out


def _padded_rows(st, U: int, M_stock: int) -> int:
    """Smallest ladder value >= max(U, min_rows) that is not known-bad for this M_stock.
    Ladder = multiples of st.row_quantum (default 4096), capped at M_stock."""
    q = max(1, int(st.row_quantum))
    cand = max(U, int(st.min_rows))
    cand = min(-(-cand // q) * q, M_stock)
    while cand < M_stock and st.kv_validation.get((cand, M_stock)) is False:
        nxt = min(-(-int(cand * 2) // q) * q, M_stock)  # geometric ladder: <= log2(M_stock/U) failed attempts
        cand = nxt if nxt > cand else M_stock
    return cand


def _col_forward_unified(self, hidden_states, source_embeddings, attention_mask=None, evol_time_bias=None, output_attentions=False):
    if attention_mask is None or evol_time_bias is None:
        raise ValueError("attention_mask and evol_time_bias are required")
    if output_attentions:
        return self._stock_forward(hidden_states, source_embeddings, attention_mask, evol_time_bias, output_attentions)
    st: ExactState = self._exact_state
    query_layer = self.transpose_for_scores(self.query(hidden_states))
    X, idx = st.kv_input, st.kv_index
    if st.kv_fallback_full or X is None:
        K, V = self.key(source_embeddings), self.value(source_embeddings)  # stock path for this forward
    else:
        oom = False
        try:
            K, V = _reduced_kv(self, st, X, idx, source_embeddings)
        except torch.cuda.OutOfMemoryError:  # the reduced path needs memory the stock projections do not: this shape runs stock's, by rule
            oom = True  # handled below, once the failed attempt's tensors are released
        if oom:
            X = idx = None
            _memory_fallback(st, int(st.kv_m_stock), where="reduced K/V projections")
            K, V = self.key(source_embeddings), self.value(source_embeddings)
    key_layer = self.transpose_for_scores(K)
    value_layer = self.transpose_for_scores(V)
    attention_mask = attention_mask + evol_time_bias.to(query_layer.dtype)
    context_layer = _sdpa_math(
        query_layer, key_layer, value_layer, attention_mask,
        scale=1 / math.sqrt(self.attention_head_size),
        dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
    )
    context_layer = context_layer.transpose(-2, -3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return (context_layer,)


def _equal_gathered(u: Tensor, idx: Tensor, ref: Tensor) -> bool:
    """torch.equal(u[idx], ref) evaluated in batch slices, so the comparison never materialises a second full-size tensor
    (a gather is pure data movement: slicing it changes no value)."""
    B = int(idx.shape[0])
    per_b = max(1, int(ref[0].numel()) * ref.element_size())
    step = max(1, (1 << 30) // per_b)
    for b0 in range(0, B, step):
        if not torch.equal(u[idx[b0 : b0 + step]], ref[b0 : b0 + step]):
            return False
    return True


def _reduced_kv(self, st, X: Tensor, idx: Tensor, source_embeddings: Tensor):
    """K and V of every clade row from the reduced GEMM over X and the gather index, one projection at a time so that at most
    ONE reduced projection is alive beside the gathered tensors (the transient memory stays within what the stock projections
    use), with the layer-0 self-check folded in: on the first forward of a (reduced rows, stock rows) pair each reduced
    projection is compared with the stock projection of the real data THROUGH the gather, bit for bit, before it is used."""
    check = bool(st.validate and getattr(self, "_is_layer0", False))
    vkey = (int(X.shape[0]), int(st.kv_m_stock))
    status = st.kv_validation.get(vkey) if check else True
    if check and status is False:  # known-bad pair (only reachable at rows == stock rows): stock projections
        _register_reject(st, getattr(st, "kv_mode_used", st.dedup), vkey)
        st.kv_fallback_full = True
        st.kv_input = None
        return self.key(source_embeddings), self.value(source_embeddings)
    if st.debug_force_oom:  # TEST HOOK ONLY (forced out-of-memory for fault-injection tests); never set in production
        st.debug_force_oom -= 1
        raise torch.cuda.OutOfMemoryError("forced by debug_force_oom")
    fused = bool(st.fuse_kv_unified)
    if fused:
        if not hasattr(self, "_w_kv"):
            self._w_kv = torch.cat([self.key.weight, self.value.weight], 0).contiguous()
            self._b_kv = torch.cat([self.key.bias, self.value.bias], 0).contiguous()
        kv = F.linear(X, self._w_kv, self._b_kv)
        Ah = self.all_head_size
        k_u, v_u = kv[:, :Ah], kv[:, Ah:]
    else:
        k_u, v_u = self.key(X), None  # (rows, Ah); the value projection is computed after K is gathered and k_u released
    if status is None:
        # One-time runtime check per (reduced rows, stock rows) pair: the reduced GEMM + gather must reproduce the stock
        # GEMM on the real data bit for bit (covers cuBLAS kernel-selection equivalence, row-position independence and the
        # dedup mapping at once).  Host sync; it runs during eager warm-up and is cached, so it never executes inside
        # CUDA-graph capture.  K is checked and gathered first, then V, so the reference tensors are alive one at a time.
        K_ref = self.key(source_embeddings)
        ok = _equal_gathered(k_u, idx, K_ref)
        if ok:
            del K_ref
            K = k_u[idx]
            k_u = None
            if v_u is None:
                v_u = self.value(X)
            V_ref = self.value(source_embeddings)
            ok = _equal_gathered(v_u, idx, V_ref)
            if ok:
                del V_ref
                V = v_u[idx]
            else:
                V = V_ref  # K passed its check (it IS the stock projection bit for bit); V is stock's own
            v_u = None
        else:
            k_u = v_u = None
            K, V = K_ref, self.value(source_embeddings)  # stock projections for this forward (exact)
        if st.debug_force_reject:  # TEST HOOK ONLY (forced self-check failure for fault-injection tests); never set in production
            st.debug_force_reject -= 1
            ok = False
        st.kv_validation[vkey] = ok
        mode_used = getattr(st, "kv_mode_used", st.dedup)
        st.validation_log.append({"rows": vkey[0], "stock_rows": vkey[1], "ok": ok, "mode": {False: "p3b", True: "dedup", "static": "sdedup"}.get(mode_used, str(mode_used))})
        if not ok:  # stay exact: stock projections in every layer of THIS forward; later forwards use the next option
            _register_reject(st, mode_used, vkey)
            st.kv_fallback_full = True
            st.kv_input = None  # free the reduced input right away (layers 1+ will not touch it)
        return K, V
    K = k_u[idx]  # gather -> (B, L, C, Ah)
    k_u = None
    if v_u is None:
        v_u = self.value(X)
    V = v_u[idx]
    return K, V


def _memory_fallback(st, m_stock: int, where: str) -> None:
    """The reduced K/V path ran out of device memory at this batch shape: release what it holds, run the STOCK projections for this
    shape from now on (exact; the levers' other savings remain) and say so once. kv_mode_report(model)['memory_fallbacks'] names it."""
    st.kv_input = None
    st.kv_index = None
    st.multi_source_embeddings = None
    st.kv_fallback_full = True
    st.kv_mode_used = "stock"
    if st.mode_override is None:
        st.mode_override = {}
    first = st.mode_override.get(m_stock) != "stock"
    st.mode_override[m_stock] = "stock"
    if st.memory_fallbacks is None:
        st.memory_fallbacks = {}
    st.memory_fallbacks[m_stock] = st.memory_fallbacks.get(m_stock, 0) + 1
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if not _ALLOCATOR_SWITCHED[0] and _allocator_kind() == "default":
        try:  # segments created from here on grow in place instead of fragmenting; existing cached blocks were just released
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            _ALLOCATOR_SWITCHED[0] = True
        except Exception:  # noqa: BLE001 - a runtime without the setter keeps its layout
            pass
    if first:
        B_, L_ = getattr(st, "kv_shape_used", ("?", "?"))
        warnings.warn(f"gpnstar_exact: the {where} did not fit in device memory for batch shape B={B_} x L={L_} ({m_stock:,} stock K/V rows) "
                      "-> this shape now runs the STOCK projections (exact; stock speed and memory for the K/V projections). "
                      "A smaller eval batch keeps the reduced path.", RuntimeWarning, stacklevel=2)


def _register_reject(st, mode_used, vkey) -> None:
    """Self-check rejected the reduced K/V GEMM built by `mode_used` at (rows, stock_rows) = vkey.
    Policy (per stock_rows value, i.e. per batch shape):
      dedup / sdedup : after `st.dedup_max_rejects` rejected paddings -> switch this shape to P3b (static unified K/V,
                       slower than de-dup yet faster than the stock projections, and validated on its own) instead of climbing the ladder to stock speed;
      p3b            : keeps climbing its (short) padding ladder; if the full-size padding (rows == stock_rows) is also
                       rejected, or after `st.p3b_max_rejects` rejections -> terminal 'stock' for this shape: the reduced
                       input is no longer built at all (stock projections, ~stock memory, P0-P2 speed; still exact)."""
    rows, m_stock = vkey
    fam = "dedup" if mode_used in (True, "static") else "p3b"
    key = (fam, m_stock)
    st.reject_counts[key] = st.reject_counts.get(key, 0) + 1
    B_, L_ = getattr(st, "kv_shape_used", ("?", "?"))
    shape_txt = f"batch shape B={B_} x L={L_} ({m_stock:,} stock K/V rows; reduced GEMM had {rows:,} rows)"
    before = st.mode_override.get(m_stock)
    if fam == "dedup":
        if st.on_dedup_reject == "p3b" and st.reject_counts[key] >= st.dedup_max_rejects:
            st.mode_override[m_stock] = "p3b"
            for shp in [k for k, pl in list((st.static_plans or {}).items()) if k[0] * k[1] * len(st.clade_species) == m_stock]:
                st.static_plans.pop(shp, None)  # free static-dedup slabs of this shape
        elif st.on_dedup_reject == "stock" and st.reject_counts[key] >= st.dedup_max_rejects:
            st.mode_override[m_stock] = "stock"
        elif rows >= m_stock:  # 'ladder' exhausted (full-size padding rejected too): nothing left to try
            st.mode_override[m_stock] = "stock"
    else:
        if rows >= m_stock or st.reject_counts[key] >= st.p3b_max_rejects:
            st.mode_override[m_stock] = "stock"
    after = st.mode_override.get(m_stock)
    if after != before:  # LOUD, once per decision: tell the user where this shape landed and why (outputs stay exact either way)
        if after == "p3b":
            warnings.warn(f"gpnstar_exact: the layer-0 reduced-vs-full K/V self-check rejected the DE-DUPLICATED GEMM for {shape_txt} "
                          f"{st.reject_counts[key]}x -> this shape now runs the STATIC unified K/V path (P3b: exact; slower than the de-duplicated GEMM, faster than the stock projections). "
                          "Outputs remain bitwise identical to stock (the kit never chunks a batch: chunked outputs would equal "
                          "stock-at-the-chunk-size, not stock-at-your-batch-size).", RuntimeWarning, stacklevel=2)
        elif after == "stock":
            warnings.warn(f"gpnstar_exact: every reduced K/V GEMM option was rejected by the self-check for {shape_txt} -> this shape now runs the "
                          "STOCK projections (exact, ~stock speed and memory; only the P0-P2 hygiene gains remain). See kv_mode_report(model).",
                          RuntimeWarning, stacklevel=2)


def kv_mode_report(model: nn.Module) -> dict:
    """Which K/V path each seen batch shape ended up on, and the self-check log."""
    st = _state(model)
    return {"configured": {False: "p3b", True: "dedup", "static": "sdedup"}.get(st.dedup, str(st.dedup)),
            "override_per_stock_rows": dict(st.mode_override or {}), "reject_counts": {f"{k[0]}@{k[1]}": v for k, v in (st.reject_counts or {}).items()},
            "validation_log": list(st.validation_log or []), "last_mode_used": {False: "p3b", True: "dedup", "static": "sdedup"}.get(getattr(st, "kv_mode_used", None), str(getattr(st, "kv_mode_used", None))),
            "memory_fallbacks": dict(st.memory_fallbacks or {}), "memory_plans": dict(st.memory_plans or {})}


def patch_unified_kv(model: nn.Module, *, dedup: bool, min_rows: int = 2048, fuse_kv: bool = False,
                     validate: bool = True, row_quantum: int = 4096, on_dedup_reject: str = "p3b",
                     dedup_max_rejects: int = 2, p3b_max_rejects: int = 3) -> None:
    """Install P3b (dedup=False) or P4 (dedup=True).  Supersedes patch_kv_lut.
    validate=True: the first forward of every new (padded rows, stock rows) pair checks at layer 0 that
    the reduced K/V GEMM reproduces the stock GEMM bitwise on the real data; if not, that forward falls
    back to the stock projection (still exact) and later forwards try the next padding on the ladder."""
    core = core_model(model)
    st = _state(model)
    patch_source_gather(model)  # ensures state on source module
    st.lut_enabled = False
    st.stash_multi = True
    if dedup not in (False, True, "static"):
        raise ValueError("dedup must be False, True or 'static'")
    st.dedup = dedup
    if not isinstance(getattr(st, "static_caps", None), dict):
        st.static_caps = {}
        st.static_plans = {}
    st.min_rows = int(min_rows)
    st.row_quantum = int(row_quantum)
    st.validate = bool(validate)
    st.fuse_kv_unified = fuse_kv
    st.kv_out_features = int(getattr(core.encoder.layer[0].attention.col_attention.self.key, "out_features", 0) or 0)
    if not hasattr(st, "code_powers") or st.code_powers is None:
        st.code_powers = None
    if not isinstance(getattr(st, "kv_validation", None), dict):
        st.kv_validation = {}
        st.validation_log = []
    if on_dedup_reject not in ("p3b", "stock", "ladder"):
        raise ValueError("on_dedup_reject must be 'p3b' (switch the shape to static unified K/V), 'stock', or 'ladder' (legacy: keep padding up to stock rows)")
    st.on_dedup_reject = on_dedup_reject
    st.dedup_max_rejects = int(dedup_max_rejects)
    st.p3b_max_rejects = int(p3b_max_rejects)
    if not isinstance(getattr(st, "mode_override", None), dict):
        st.mode_override = {}
        st.reject_counts = {}
    src = core.source_embedding
    src.forward = _source_forward_unified.__get__(src, type(src))
    for i, layer in enumerate(core.encoder.layer):
        ca = layer.attention.col_attention.self
        if not hasattr(ca, "_stock_forward"):
            ca._stock_forward = ca.forward
        ca._exact_state = st
        ca._is_layer0 = i == 0
        ca.forward = _col_forward_unified.__get__(ca, type(ca))


# extra state fields used by P3b/P4 (kept out of the dataclass signature for brevity)
ExactState.dedup = False
ExactState.min_rows = 2048
ExactState.fuse_kv_unified = False
ExactState.code_powers = None
ExactState.kv_input = None
ExactState.kv_index = None
ExactState.n_unique_rows = 0
ExactState.last_unique_counts = None
ExactState.row_quantum = 4096
ExactState.validate = True
ExactState.kv_validation = None
ExactState.validation_log = None
ExactState.kv_m_stock = 0
ExactState.kv_fallback_full = False
ExactState.static_caps = None
ExactState.static_plans = None
ExactState.mode_override = None      # {stock_rows: "p3b" | "stock"} decided by the self-check policy (_register_reject)
ExactState.reject_counts = None
ExactState.on_dedup_reject = "p3b"
ExactState.dedup_max_rejects = 2
ExactState.p3b_max_rejects = 3
ExactState.memory_fallbacks = None   # {stock_rows: count}: shapes whose reduced K/V path did not fit in memory and run the stock projections
ExactState.memory_plans = None       # {stock_rows: plan}: the numbers behind a step-aside decided before allocation
ExactState.memory_plan_last = None
ExactState.memory_layer_factor = None  # per-layer K-sized tensor count of the installed attention route (None = the stock attention's MEMORY_STOCK_LAYER_FACTOR)
ExactState.kv_out_features = 0       # output width of the K/V projections (layer 0), for the memory plan
ExactState.debug_memory_budget = None  # TEST HOOK ONLY: pretend this many bytes are obtainable
ExactState.debug_force_oom = 0
ExactState.kv_mode_used = None
ExactState.kv_shape_used = None
ExactState.debug_force_reject = 0    # test hook
ExactState.force_stock_once = False


def set_static_capacities(model: nn.Module, B: int, L: int, caps, *, margin: float = 1.0, quantum: int = 256) -> list[int]:
    """Register per-multi-clade row capacities for dedup='static' at batch shape (B, L).
    `caps` = iterable of observed unique-row counts per multi clade (e.g. max over calibration
    batches of ExactState.last_unique_counts from the eager dedup path); multiplied by `margin`,
    rounded up to a multiple of `quantum`.
    CALIBRATION GUIDANCE: take the max over >= 16 representative batches with margin 1.25; with only a few (~4)
    calibration batches use margin >= 2 at B=8 (capacities calibrated on few batches with a small margin overflow on most
    later batches).  An overflow never yields a wrong result
    if you check static_overflow() after each call, but every overflowed batch must be recomputed eagerly."""
    st = _state(model)
    if not isinstance(getattr(st, "static_caps", None), dict):
        st.static_caps = {}
        st.static_plans = {}
    q = max(1, int(quantum))
    c = [int(-(-int(math.ceil(x * margin)) // q) * q) for x in caps]
    st.static_caps[(int(B), int(L))] = c
    st.static_plans.pop((int(B), int(L)), None)
    return c


def _overflow_flag_tensor(model: nn.Module, B: int, L: int):
    """The device bool flag of the static-dedup plan for (B, L), or None if that shape has no static-dedup plan."""
    st = _state(model)
    plan = (st.static_plans or {}).get((int(B), int(L)))
    return None if plan is None else plan["overflow"]


def static_overflow(model: nn.Module, B: int, L: int, reset: bool = True) -> bool:
    """True if the last static-dedup forward(s) at shape (B, L) exceeded a clade capacity (host sync)."""
    st = _state(model)
    plan = (st.static_plans or {}).get((int(B), int(L)))
    if plan is None:
        return False
    flag = bool(plan["overflow"].item())
    if reset:
        plan["overflow"].zero_()
    return flag


def static_unique_counts(model: nn.Module, B: int, L: int) -> list[int] | None:
    st = _state(model)
    plan = (st.static_plans or {}).get((int(B), int(L)))
    return None if plan is None else [int(v) for v in plan["n_unique"].tolist()]


# --------------------------------------------------------------------------------------
# FAST mode: TF32 tensor-core matmuls with a selectable scope (numerics change -> not exact)
# --------------------------------------------------------------------------------------


def _tf32_api() -> str:
    """Which TF32 interface this process reads without error: 'legacy' (torch.backends.cuda.matmul.allow_tf32) or
    'precision' (torch.backends.cuda.matmul.fp32_precision = 'tf32' | 'ieee').  Once any code in the process has used the
    precision interface, reading the legacy flag raises; the scope then reads and writes the precision interface only, so
    the two are never mixed by this module."""
    try:
        torch.backends.cuda.matmul.allow_tf32  # noqa: B018 -- the read itself is the probe
        return "legacy"
    except RuntimeError:
        return "precision"


def _tf32_get() -> bool:
    """TF32 state of fp32 cuBLAS matmuls through the interface in use."""
    if _tf32_api() == "legacy":
        return bool(torch.backends.cuda.matmul.allow_tf32)
    return str(getattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")) == "tf32"


def _tf32_set(value: bool) -> None:
    """Set the TF32 state of fp32 cuBLAS matmuls through the interface in use."""
    if _tf32_api() == "legacy":
        torch.backends.cuda.matmul.allow_tf32 = bool(value)
    else:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if value else "ieee"


class _TF32Scope:
    """Forward pre/post hooks that switch TF32 for fp32 cuBLAS matmuls on (or off) for the duration of a module's forward and
    restore the previous state afterwards, through whichever TF32 interface the process uses (works under CUDA-graph
    capture: the cuBLAS kernel chosen at capture time is what gets replayed)."""

    def __init__(self, value: bool):
        self.value = value
        self.prev = []

    def pre(self, module, args, kwargs=None):
        self.prev.append(_tf32_get())
        _tf32_set(self.value)

    def post(self, module, args, output):
        _tf32_set(self.prev.pop() if self.prev else False)


FAST_POLICIES = ("tf32_all", "tf32_bulk", "tf32_ffn")


def apply_fast_policy(model: nn.Module, policy: str) -> None:
    """Install TF32 scopes.  Call AFTER apply_exact/patch_unified_kv.  The caller runs the model with
    torch.backends.cuda.matmul.allow_tf32 = True set globally (runtime.tf32(True)); the policy pins
    selected sub-modules back to strict fp32:
        tf32_all  : nothing pinned (every matmul TF32)
        tf32_bulk : fp32 for the source module (species embedding + clade attention pooling), the
                    column-attention key/value projections (tiny after dedup), and the MLM head;
                    TF32 for row-attention QKV/out, column Q/out, FFN, attention score/context matmuls
        tf32_ffn  : TF32 only inside the FFN (intermediate + output dense); everything else fp32
    """
    if policy not in FAST_POLICIES:
        raise ValueError(f"unknown policy {policy}; choose from {FAST_POLICIES}")
    core = core_model(model)
    handles = getattr(model, "_fast_handles", [])
    for h in handles:
        h.remove()
    handles = []

    def pin(module, value):
        sc = _TF32Scope(value)
        handles.append(module.register_forward_pre_hook(sc.pre))
        handles.append(module.register_forward_hook(sc.post))

    if policy == "tf32_bulk":
        pin(core.source_embedding, False)
        for layer in core.encoder.layer:
            ca = layer.attention.col_attention.self
            pin(ca.key, False)
            pin(ca.value, False)
        head = getattr(model, "cls", None)
        if head is not None:
            pin(head, False)
    elif policy == "tf32_ffn":
        pin(core, False)  # whole model strict ...
        for layer in core.encoder.layer:
            pin(layer.intermediate, True)  # ... except the FFN GEMMs
            pin(layer.output, True)
        # note: layer.output = dense + residual + LayerNorm (LayerNorm has no matmul; unaffected)
    model._fast_handles = handles
    model._fast_policy = policy
