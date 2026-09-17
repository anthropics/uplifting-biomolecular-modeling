"""OpenFold3's diffusion sampler on ``z_cond`` rows — ``SampleDiffusion`` / ``DiffusionModule`` (``structure/diffusion_module.py``),
``DiffusionConditioning`` (``layers/diffusion_conditioning.py``), ``DiffusionTransformer`` (``layers/diffusion_transformer.py``: 24 blocks of
``AttentionPairBias`` + ``ConditionedTransitionBlock`` x 200 steps), the atom attention encoder / decoder's token-pair term
(``NoisyPositionEmbedder.forward``'s ``convert_pair_rep_to_blocks(linear_z(layer_norm_z(zij)))``, ``core/utils/atom_attention_block_utils.py``)
— bound onto ``opt_core.mem.rowpair.diffusion`` (opt_core/mem/rowpair/API.md "diffusion"). Every row loop, block schedule and collective is
the core's; OpenFold3's submodules enter as callables; this module holds OpenFold3's attribute paths, its atom-window index statements and
the four per-call forward swaps.

The sampler loop is OpenFold3's own ``SampleDiffusion.forward`` (AF3 Alg. 18), UNCHANGED: ``sample_diffusion_rows`` calls it with
``zij_trunk`` = this rank's shard ``[1, 1, n_loc, N, 128]`` after installing, for the duration of the call and on these module INSTANCES only,
four forwards (``sharded_denoiser``):
  ``DiffusionModule.forward``        the stock statements, entered through ``diffusion.sync_replicated(xl_noisy)`` — the proof that the initial
                                     noise, the per-step noise and the random augmentation every rank drew for itself ARE identical
                                     (``ROWPAIR_DIFF_NOISE_SYNC``: ``guard`` (the line's default) | ``bcast`` | ``off``, recorded ``diff_noise_sync``)
  ``DiffusionConditioning.forward``  single half = the stock statements (replicated, depends on the noise level); pair half = the rollout's
                                     ``z_cond`` ROWS, computed ONCE per rollout (``diffusion.pair_cond_rows``: ``linear_z(layer_norm_z(cat([z rows,
                                     relpos rows])))`` then ``transition_z`` x2 in place per row block; relpos rows = ``trunk.relpos_rows``, the
                                     input embedder's one composition of the core's ``trunk.relpos_onehot_rows``)
  ``DiffusionTransformer.forward``   token queries = this rank's rows through every block (``diffusion.diffusion_transformer_sharded`` over
                                     ``DiTBlockFns`` of each ``DiffusionTransformerBlock``: AdaLN + K/V on the replicated activation, ``linear_q`` /
                                     ``_attention`` / ``_wrap_up`` on local query rows in sub-blocks of ``diff_q_rows`` with the block's pair-bias rows
                                     ``linear_z(layer_norm_z(z_cond rows))`` (``PairBiasCache`` across the steps when ``DiffusionSchedule`` says it
                                     fits, ``diff_bias_cache=on|off``), the AdaLN-zero gate + residual + ``conditioned_transition`` on local rows,
                                     ONE all_gather of the updated rows per block)
  ``NoisyPositionEmbedder.forward``  the stock statements with the token-pair term read from the rollout's BAND: ``diffusion.band_plan`` over the
                                     32-query x 128-key atom windows mapped to token pairs (``atom_window_plan``: ``convert_pair_rep_to_blocks``'s
                                     index statements), ``diffusion.pair_band_rows`` (``linear_z(layer_norm_z(z_cond rows))`` folded into the
                                     ``[N, 2W+1, 16]`` band per row block, ONCE per rollout; rows an out-of-band window needs travel whole, at most
                                     ``ROWPAIR_DIFF_BAND_EXTRA_MAX``), ``diffusion.band_lookup`` -> exactly stock's ``zp[b, q_idx, k_idx]``, then
                                     stock's ``masked_fill_`` / ``atom_pair_mask`` statements
Nothing ``[N, N, c]`` exists on any rank: resident per rank for the rollout are the ``z_cond`` shard (1 u), the bias cache
(``24 x 16 x Rmax x N`` fp32 when on) and the replicated band; transients are the core's row blocks (``DiffusionSchedule.decide``, printed as
``diff_cond_rows diff_bias_rows diff_q_rows diff_band_rows diff_bias_cache diff_rows_source diff_work_bytes``; pins ``ROWPAIR_DIFF_{COND,BIAS,Q,
BAND}_ROWS``, budgets ``ROWPAIR_DIFF_WORK_GB`` / ``ROWPAIR_DIFF_BIAS_CACHE(_GB)``; the band cap ``ROWPAIR_DIFF_BAND_W`` (default ``n_key - 1``)
and ``ROWPAIR_DIFF_BAND_EXTRA_MAX`` (default 256) are this adapter's explicit arguments of ``band_plan``).

REPLICATED BY DESIGN (named in the schedule census as ``diff_replicated=s,a,atoms,atom_attention``): the single conditioning ``si``, the token
activation ``a [1, S, N, 768]`` between blocks, the DiT K/V projections, every atom-level tensor (encoder / decoder windows, ``plm``, noise,
coordinates). Every rank runs the identical loop and issues OpenFold3's RNG calls in OpenFold3's order, so identically seeded ranks draw the
identical trajectory; ``sync_replicated`` proves it every step.

Numerics (API.md table): conditioning / bias / band projection / gate / transition are row-local (per-element the dense statement, GEMM M =
local rows); the attention is whole per query row over all N keys (stock ``_attention``: einsum, ``+= mask_bias``, ``+= pair bias``,
``softmax_no_cast``); gathers, band assembly and the extras broadcast are bit copies. The pair conditioning is step-invariant, so once per
rollout == stock's per-step recompute value for value.

Under the sample loop (``draws``: the stock-order draw stream of one chunk of samples) the roll-out is ``rollout_drawn``, OpenFold3's Alg. 18
statements with every batch-shaped draw made at the all-samples shape and sliced to the chunk; without it the stock loop above runs.

Refused by name: ``use_conditioning=False``, the DS4Sci / cuEq / triton / LMA attention routes, ``DiffusionTransformer(use_cross_attention)``,
a live ``DiffusionConditioning.chunk_size_tuner``, a P = 1 layout (the core's statements refuse it: at ``--n_gpu 1`` nothing here runs).
"""
from __future__ import annotations

import math
import os
import time
import sys
from contextlib import contextmanager
from typing import Optional

import torch

from . import core as C
from . import pairstack as PS
from . import trunk as TR

C.register("diffusion", ("DiffusionSchedule", "pair_cond_rows", "pair_bias_rows", "PairBiasCache", "DiTBlockFns", "dit_block_sharded",
                         "diffusion_transformer_sharded", "band_plan", "pair_band_rows", "band_lookup", "sync_replicated", "BandPlan"))

ENV_BAND_W = "ROWPAIR_DIFF_BAND_W"                  # band_plan(max_w=): cap on the band half-width in tokens (default n_key - 1); rows outside travel whole
ENV_BAND_EXTRA_MAX = "ROWPAIR_DIFF_BAND_EXTRA_MAX"  # advisory count of whole rows outside the band (default 256): more is a NOTE line, the plan runs as requested


class DiffusionRefused(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------------------- per-rollout state
class RolloutCache(object):
    """What one rollout computes ONCE on this rank: ``lay``, ``schedule`` (the core's ``DiffusionSchedule``), ``z_cond_loc [1, 1, n_loc, N, c_z]``,
    ``bias_cache`` (the core's ``PairBiasCache``), ``win`` (``atom_window_plan``), ``plan`` (the core's ``BandPlan``), ``plm_z`` (the atom
    encoder's token-pair term ``[1, 1, n_blocks, n_query, n_key, c_atom_pair]``, replicated), ``steps`` (denoiser calls served)."""

    __slots__ = ("lay", "schedule", "z_cond_loc", "bias_cache", "win", "plan", "plm_z", "steps", "rows_core")

    def __init__(self, lay, schedule, z_cond_loc, bias_cache, win, plan, plm_z, rows_core=None):
        self.lay, self.schedule, self.z_cond_loc, self.bias_cache = lay, schedule, z_cond_loc, bias_cache
        self.win, self.plan, self.plm_z = win, plan, plm_z
        self.steps = 0
        self.rows_core = rows_core                             # the rows attention core word admitted for this roll-out (None = the engine's _attention statement)


ENV_DIT_ROWS = "OF3TP_DIT_ROWS"                                # the kit switch (env.KIT_SWITCHES): big (default) | apb_attn | sba[...] | sdpa | naive | off
DIT_ROWS_DEFAULT = "big"


def dit_rows_word() -> str:
    """The DiT rows attention core word this process asks the core to admit (``OF3TP_DIT_ROWS``; ``off`` / ``engine`` / ``0`` = the engine's own ``_attention`` statement)."""
    w = os.environ.get(ENV_DIT_ROWS, DIT_ROWS_DEFAULT).strip().lower()
    return "" if w in ("off", "engine", "0", "none", "stock") else w


def _c_in(ln) -> int:
    c = getattr(ln, "c_in", None)
    if c is None:
        return int(ln.weight.shape[-1])
    return int(c[0] if isinstance(c, (tuple, list)) else c)


_DECIDED_ROWS = [None]                                          # the rows core word decide_schedule admitted last (build_rollout_cache hands it to the RolloutCache)


def decide_schedule(dm, lay, no_rollout_samples: int, z_loc):
    """``DiffusionSchedule.decide`` from OpenFold3's dimensions: ``c_z`` (the trunk shard's channels; ``z_loc`` = the shard tensor or the parked
    shard ``model.rollout_rows`` hands over — ``heads.ZTrunkPlan.source()``, read by row block in ``pair_cond_rows``), ``c_in`` = ``c_z + 139`` relpos features
    (``layer_norm_z``), ``c_cond`` (``linear_z.out_features``), ``H`` (DiT heads), ``S`` (rollout samples), ``n_blocks``, ``c_pair``
    (``c_atom_pair``), ``elt`` (the shard's element size)."""
    D = C.seam("diffusion")
    dc, dt = dm.diffusion_conditioning, dm.diffusion_transformer
    apb0 = dt.blocks[0].attention_pair_bias
    npe = dm.atom_attn_enc.noisy_position_embedder
    attn_core, w = None, dit_rows_word()                                     # the rows attention slot's static admission, once per roll-out (q_rows = Rmax under an admitted core;
    if w and hasattr(D, "dit_rows_core"):                                     #  q_rows_stock keeps the byte model's block for the statement's fallback chunks)
        act_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else z_loc.dtype
        attn_core, _sel, why = D.dit_rows_core(w, dtype=act_dtype, heads=int(apb0.mha.no_heads), head_dim=int(apb0.mha.c_hidden), samples=int(no_rollout_samples),
                                               device=getattr(z_loc, "device", None))
        C.comm().log(f"[diffusion] rows core word={w!r} act_dtype={str(act_dtype).rsplit('.', 1)[-1]} -> attn_core={attn_core} ({why})")
    _DECIDED_ROWS[0] = w if attn_core == "kernel" or w == "naive" else None
    kw = {} if attn_core is None else dict(attn_core=attn_core)
    return D.DiffusionSchedule.decide(lay, c_z=int(z_loc.shape[-1]), c_in=_c_in(dc.layer_norm_z), c_cond=int(dc.linear_z.out_features),
                                      H=int(apb0.mha.no_heads), S=int(no_rollout_samples), n_blocks=len(dt.blocks), c_pair=int(npe.linear_z.out_features),
                                      elt=int(torch.empty((), dtype=z_loc.dtype).element_size()), **kw)   # z_loc: the shard tensor or the parked shard (heads.ZTrunkPlan.source(); both carry dtype)


# ----------------------------------------------------------------------------------------------------------------- (1) conditioning
def conditioning_pair_rows(dc, batch, z_loc, lay, rows: Optional[int] = None):
    """The pair half of ``DiffusionConditioning.forward`` (``_embed_trunk_inputs`` :151-156 + ``_forward`` :168-170) on this rank's rows ->
    ``z_cond_loc [1, 1, n_loc, N, c_z]``: the core's ``pair_cond_rows`` with OpenFold3's statements as callables."""
    D, T = C.seam("diffusion"), C.seam("trunk")
    token_mask = batch["token_mask"]

    def embed_fn(z_rows, g0, g1):                            # relpos_zij = relpos_complex(batch, ...)   (rows g0:g1 only: the lazy relpos)
        relpos_zij = TR.relpos_rows(batch, g0, g1, dc.max_relative_idx, dc.max_relative_chain, dtype=z_rows.dtype)
        zij = torch.cat([z_rows, relpos_zij], dim=-1)         # zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
        return dc.linear_z(dc.layer_norm_z(zij))              # zij = self.linear_z(self.layer_norm_z(zij))

    def transition_fn(l):                                     # zij = zij + l(zij, mask=pair_token_mask, chunk_size)  — Transition.forward: mask.unsqueeze(-1), _transition
        def t(x_rows, g0, g1):
            return l._transition(x=x_rows, mask=T.pair_mask_rows(token_mask, g0, g1).unsqueeze(-1).to(dtype=x_rows.dtype))
        return t

    return D.pair_cond_rows(embed_fn, z_loc, lay, c_out=int(dc.linear_z.out_features), rows=rows, transitions=[transition_fn(l) for l in dc.transition_z])


def conditioning_single(dc, batch, t, si_input, si_trunk, chunk_size=None):
    """The single half of ``DiffusionConditioning.forward`` (``_embed_trunk_inputs`` :158-165 + ``_forward`` :172-174): replicated, depends on the
    noise level, stock statements."""
    token_mask = batch["token_mask"]
    si = torch.cat([si_trunk, si_input], dim=-1)
    si = dc.linear_s(dc.layer_norm_s(si))
    n = 0.25 * torch.log(t / dc.sigma_data)
    n = dc.fourier_emb(n.unsqueeze(-1))
    si = si + dc.linear_n(dc.layer_norm_n(n)).unsqueeze(-2)
    for l in dc.transition_s:
        si = si + l(si, mask=token_mask, chunk_size=chunk_size)
    return si


# ----------------------------------------------------------------------------------------------------------------- (2) DiffusionTransformer
def dit_block_fns(blk, mask, layer_norm_z, use_high_precision_attention: bool = False, _mask_trans: bool = True, rows_core=None, stock_q_rows=None):
    """``DiTBlockFns`` of one ``DiffusionTransformerBlock`` (``diffusion_transformer.py:142-176``; ``DiffusionAttentionPairBias``,
    ``attention_pair_bias.py:212-330``: AdaLN of the queries, ``linear_z`` of the pair rows, the ``linear_ada_out`` output gate;
    ``primitives/attention.py`` ``_prep_qkv`` / ``_attention`` / ``_wrap_up``): the stock statements, split at the core's seams.
    ``layer_norm_z`` is the DiffusionTransformer's ONE pair LayerNorm (``diffusion_transformer.py:246,303``: ``z = self.layer_norm_z(z)`` once,
    ahead of the blocks): a per-element statement of the pair rows, so it runs here on each bias row block ahead of the block's ``linear_z``
    (the ``PairBiasCache`` holds the projected rows across the steps: once per block per roll-out when it fits)."""
    F = C.fn("diffusion", "DiTBlockFns")
    apb = blk.attention_pair_bias
    mha = apb.mha
    H = int(mha.no_heads)
    attention = PS._attention_core()

    def norm(a, s):                                           # a = self.layer_norm_a(a, s)   (AdaLN; attention_pair_bias.py:318)
        return apb.layer_norm_a(a, s)

    def kv(x):                                                # _prep_bias' mask term + _prep_qkv's k / v (all rows, once per block)
        m = mask
        if m is None:
            m = x.new_ones(x.shape[:-1])
        batch_dims = x.shape[:-2]
        m = m.expand((*batch_dims, -1))
        mask_bias = (apb.inf * (m - 1))[..., None, None, :]
        k = mha.linear_k(x)
        v = mha.linear_v(x)
        k = k.view(k.shape[:-1] + (H, -1))
        v = v.view(v.shape[:-1] + (H, -1))
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        return k, v, mask_bias

    def attn(x_q, kv_, bias_q, rows):                         # _prep_qkv's q (apply_scale) ; _attention(q, k, v, [mask_bias, pair bias]) ; transpose ; _wrap_up
        k, v, mask_bias = kv_
        q = mha.linear_q(x_q)
        q = q.view(q.shape[:-1] + (H, -1))
        q = q.transpose(-2, -3)
        q /= math.sqrt(mha.c_hidden)
        o = attention(q, k, v, [mask_bias, bias_q], use_high_precision=use_high_precision_attention)
        o = o.transpose(-2, -3)
        return mha._wrap_up(o, x_q)

    def update(a, o_rows, s, rows):                           # AttentionPairBias.forward :236 ; DiffusionTransformerBlock.forward :160-187 on rows r0:r1
        r0, r1 = rows
        s_rows = s[..., r0:r1, :]
        o_rows = apb.sigmoid(apb.linear_ada_out(s_rows)) * o_rows       # attention_pair_bias.py:328
        a_rows = a[..., r0:r1, :] + o_rows
        trans_mask = (mask[..., r0:r1] if mask is not None else None) if _mask_trans else None
        return a_rows + blk.conditioned_transition(a=a_rows, s=s_rows, mask=trans_mask)

    def bias(z_rows):                                         # DiffusionTransformer.forward `z = layer_norm_z(z)` (:303) then _prep_bias `linear_z(z)` (:301); the core moves heads first == permute_final_dims(z, [2, 0, 1])
        return apb.linear_z(layer_norm_z(z_rows))

    # ---- sampler rows: (a) the pair-bias producer as weights (ROWPAIR_DIFF_BIAS=ln_proj on the line), (b) the query-block attention on the rows face --
    D = C.seam("diffusion")
    bias_into = None
    if hasattr(D, "DitBias"):                                # (a): LN(z rows) + linear_z written head-major straight into the [H, R, N] block; engine (the `bias` statement above) by name on refusal
        _ln = layer_norm_z
        bias_into = D.DitBias(engine=bias, ln_weight=getattr(_ln, "weight", None), ln_bias=getattr(_ln, "bias", None),
                              weight=apb.linear_z.weight, linear_bias=getattr(apb.linear_z, "bias", None), eps=float(getattr(_ln, "eps", 1e-5)))
    if rows_core and hasattr(D, "dit_attention_rows"):        # (b): k / v cast ONCE per block, q per query block; the `attn` statement above by name on refusal, in stock_q_rows chunks
        stock_attn, stock_kv = attn, kv                        # captured BEFORE the names are rebound (the fallback by name and the K/V projections)

        def kv(x):                                            # noqa: F811 - rebinding is the point
            k, v, mask_bias = stock_kv(x)
            return D.DitKV(D.cast16(k), D.cast16(v), mask, (k, v, mask_bias))

        def q_fn(x_q):                                        # _prep_qkv's q with apply_scale
            q = mha.linear_q(x_q)
            q = q.view(q.shape[:-1] + (H, -1)).transpose(-2, -3)
            return q / math.sqrt(mha.c_hidden)

        def out_fn(o, x_q):                                   # o [.., S, q, H*D] -> _wrap_up(o [.., q, H, D], q_x): the sigmoid(linear_g) gate + linear_o
            return mha._wrap_up(o.view(o.shape[:-1] + (H, -1)), x_q)

        attn = D.dit_attention_rows(q_fn=q_fn, out_fn=out_fn, stock_fn=stock_attn, num_heads=H, core_word=rows_core, scale=1.0, inf=float(apb.inf),
                                    stock_q_rows=stock_q_rows)
    if bias_into is None:
        return F(norm=norm, kv=kv, attn=attn, update=update, bias=bias)
    return F(norm=norm, kv=kv, attn=attn, update=update, bias=bias, bias_into=bias_into)


def diffusion_transformer_rows(dt, a, s, z_cond_loc, mask, cache: RolloutCache, use_high_precision_attention=False, _mask_trans=True, **flags):
    """``DiffusionTransformer.forward`` (token level) with queries = this rank's rows: the core's block loop over the blocks' ``DiTBlockFns``."""
    PS.refuse_kernel_flags("DiffusionTransformer", **flags)
    if any(getattr(b, "use_cross_attention", False) for b in dt.blocks):          # DiffusionTransformerBlock.use_cross_attention = n_query is not None (diffusion_transformer.py:77)
        raise DiffusionRefused("refused: DiffusionTransformer blocks with cross attention (n_query set) on pair rows (the token-level transformer has n_query=None)")
    D = C.seam("diffusion")
    fns = [dit_block_fns(b, mask, dt.layer_norm_z, use_high_precision_attention=use_high_precision_attention, _mask_trans=_mask_trans,
                         rows_core=getattr(cache, "rows_core", None), stock_q_rows=getattr(cache.schedule, "q_rows_stock", None)) for b in dt.blocks]   # :303 z = self.layer_norm_z(z), folded into each block's bias rows
    return D.diffusion_transformer_sharded(fns, a, s, z_cond_loc, cache.lay, schedule=cache.schedule, bias_cache=cache.bias_cache)


# ----------------------------------------------------------------------------------------------------------------- (3) atom-attention band
class AtomWindows(object):
    """``convert_pair_rep_to_blocks``' index statements (``atom_attention_block_utils.py``) for this batch: ``q_idx [F, nb, nq]`` / ``k_idx
    [F, nb, nk]`` token rows / columns of every (query slot, key slot), ``pair_valid`` (real query atom AND valid key slot), ``k_valid``
    (valid key slot, any query slot), plus what the stock ``masked_fill_`` / ``atom_pair_mask`` statements need."""
    __slots__ = ("batch_dims", "flat", "n_atom", "num_blocks", "pad", "q_idx", "k_idx", "key_block_idxs", "invalid_mask", "invalid_mask_f",
                 "atom_mask_e", "pair_valid", "k_valid", "n_query", "n_key")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])


def atom_window_plan(batch, lead_shape, n_query: int, n_key: int, device) -> AtomWindows:
    from openfold3.core.utils.atom_attention_block_utils import get_block_indices, get_query_block_padding
    atom_to_token_index = batch["atom_to_token_index"]
    atom_mask = batch["atom_mask"]
    batch_dims = tuple(lead_shape)
    n_atom = atom_to_token_index.shape[-1]
    flat_batch_size = int(math.prod(batch_dims))
    num_blocks = math.ceil(n_atom / n_query)
    pad_len_right_q = get_query_block_padding(n_atom=n_atom, n_query=n_query)
    atom_to_token_index_q = torch.nn.functional.pad(atom_to_token_index, (0, pad_len_right_q), value=0.0)
    q_indices = atom_to_token_index_q.reshape((flat_batch_size, num_blocks, n_query)).long()
    atom_mask_e = atom_mask.expand(*batch_dims, -1)
    key_block_idxs, invalid_mask = get_block_indices(atom_mask=atom_mask_e, n_query=n_query, n_key=n_key, device=device)
    atom_to_token_index_flat = atom_to_token_index.reshape(flat_batch_size, n_atom)
    key_block_idxs_flat = key_block_idxs.reshape(flat_batch_size, num_blocks * n_key)
    k_indices_flat = torch.gather(atom_to_token_index_flat, 1, key_block_idxs_flat.long())
    k_indices = k_indices_flat.reshape(flat_batch_size, num_blocks, n_key).long()
    invalid_mask_f = invalid_mask.reshape(flat_batch_size, num_blocks, n_key)
    q_real = (torch.arange(num_blocks * n_query, device=device) < n_atom).view(1, num_blocks, n_query, 1)
    k_valid = (~invalid_mask_f).unsqueeze(-2)                                                   # [F, nb, 1, nk]
    shape4 = (flat_batch_size, num_blocks, n_query, n_key)
    return AtomWindows(batch_dims=batch_dims, flat=flat_batch_size, n_atom=int(n_atom), num_blocks=int(num_blocks), pad=int(pad_len_right_q),
                       q_idx=q_indices, k_idx=k_indices, key_block_idxs=key_block_idxs, invalid_mask=invalid_mask, invalid_mask_f=invalid_mask_f,
                       atom_mask_e=atom_mask_e, pair_valid=(q_real & k_valid).expand(shape4), k_valid=k_valid.expand(shape4), n_query=int(n_query), n_key=int(n_key))


def band_plan_of(win: AtomWindows, N: int):
    """The core's ``BandPlan`` for these windows: ``W`` = what the REAL (query, key) pairs need, capped by ``ROWPAIR_DIFF_BAND_W`` (default
    ``n_key - 1``); every row a valid key slot reads outside that band — the padded query slots read row 0 (``pad(..., value=0)``) — travels
    whole (``extra_rows``, typically ``[0]``), so ``band_lookup`` is stock's gather at EVERY slot, padded ones included."""
    D = C.seam("diffusion")
    env_int = C.fn("dist", "env_int")
    max_w = env_int(ENV_BAND_W, win.n_key - 1)
    emax = env_int(ENV_BAND_EXTRA_MAX, 256)
    need = D.band_plan(win.q_idx, win.k_idx, win.pair_valid, N, max_w=max_w, max_extra_rows=BAND_EXTRA_UNCAPPED, record=False)   # W from the real pairs
    plan = D.band_plan(win.q_idx, win.k_idx, win.k_valid, N, max_w=need.W, max_extra_rows=BAND_EXTRA_UNCAPPED)                    # same W; extras for any valid key slot
    note = band_extra_note(len(plan.extra_rows), emax, plan.W)
    if note:                                                                                     # more whole rows than the advisory count: named once per process, the plan runs as requested
        _note_once(note)
    return plan


BAND_EXTRA_UNCAPPED = 1 << 40                      # band_plan(max_extra_rows=): no cap — ROWPAIR_DIFF_BAND_EXTRA_MAX is advisory (a NOTE line), never a refusal
_NOTED = set()


def band_extra_note(n_extra: int, emax: int, W: int) -> Optional[str]:
    """The NOTE line of a band plan that carries more whole pair rows than ``ROWPAIR_DIFF_BAND_EXTRA_MAX`` (default 256) advises: each such row travels as a
    full ``[N, c]`` broadcast per rank (device memory and all-gather volume grow with the count); None within the advisory count."""
    if int(n_extra) <= int(emax):
        return None
    return (f"NOTE diffusion band plan carries {int(n_extra)} whole pair rows outside the |k-q|<={int(W)} band (> {ENV_BAND_EXTRA_MAX}={int(emax)}, advisory): "
            f"proceeding with the requested band — each whole row is an [N, c] broadcast per rank, memory grows with the count; {ENV_BAND_W}=<w> widens the band")


def _note_once(note: str) -> None:
    """One stderr line per process per NOTE kind (the band plan is rebuilt per rollout)."""
    key = note.split(":")[0]
    if key in _NOTED:
        return
    _NOTED.add(key)
    print(f"[openfold3_ob0-opt rowpair] {note}", file=sys.stderr, flush=True)


def plm_from_band(band, extras, plan, win: AtomWindows):
    """== ``convert_pair_rep_to_blocks(batch, zp, n_query, n_key)``: ``zp[b, q_idx, k_idx]`` read from the band (the core's ``band_lookup``), then
    the stock ``masked_fill_`` / ``get_pair_atom_block_mask`` / reshape / ``* atom_pair_mask`` statements."""
    from openfold3.core.utils.atom_attention_block_utils import get_pair_atom_block_mask
    plm = C.fn("diffusion", "band_lookup")(band, extras, plan)                                  # [F, nb, nq, nk, C]
    F_, nb, nq, nk = win.flat, win.num_blocks, win.n_query, win.n_key
    plm.masked_fill_(win.invalid_mask_f[..., None, :, None].expand((F_, nb, nq, nk, plm.shape[-1])), 0)
    atom_pair_mask = get_pair_atom_block_mask(atom_mask=win.atom_mask_e, num_blocks=nb, n_query=nq, n_key=nk, pad_len_right_q=win.pad,
                                              key_block_idxs=win.key_block_idxs, invalid_mask=win.invalid_mask)
    plm = plm.reshape((*win.batch_dims, nb, nq, nk, plm.shape[-1]))
    plm = plm * atom_pair_mask.unsqueeze(-1)
    return plm


def atom_pair_term(dm, batch, z_cond_loc, lay, rows: Optional[int] = None):
    """``(win, plan, plm_z)``: the atom encoder's token-pair term ``convert_pair_rep_to_blocks(linear_z(layer_norm_z(z_cond)))``
    (``NoisyPositionEmbedder.forward`` :359-364) from the band of the ``z_cond`` shard — the core's ``pair_band_rows`` with the projection as
    the callable (the projected shard never exists), ONE all_gather of the band rows + the whole extra rows."""
    D = C.seam("diffusion")
    enc = dm.atom_attn_enc
    npe = enc.noisy_position_embedder
    win = atom_window_plan(batch, tuple(int(v) for v in z_cond_loc.shape[:-3]), int(enc.n_query), int(enc.n_key), z_cond_loc.device)
    plan = band_plan_of(win, int(lay.N))
    band, extras = D.pair_band_rows(lambda z_rows: npe.linear_z(npe.layer_norm_z(z_rows)), z_cond_loc, lay, plan, rows=rows)   # zij_trunk = self.linear_z(self.layer_norm_z(zij_trunk))
    plm_z = plm_from_band(band, extras, plan, win)
    del band, extras
    return win, plan, plm_z


def noisy_position_embedder_forward_band(npe, cache: RolloutCache, batch, cl, plm, si_trunk, zij_trunk, rl, n_query, n_key):
    """``NoisyPositionEmbedder.forward`` (``sequence_local_atom_attention.py:342-368``), stock statements; the token-pair term is the rollout's ``plm_z``."""
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
    si_trunk = npe.linear_s(npe.layer_norm_s(si_trunk))
    si_trunk = broadcast_token_feat_to_atoms(token_mask=batch["token_mask"], num_atoms_per_token=batch["num_atoms_per_token"], token_feat=si_trunk, token_dim=-2)
    cl = cl + si_trunk
    plm = plm + cache.plm_z                                    # zij_trunk = convert_pair_rep_to_blocks(linear_z(layer_norm_z(zij_trunk))); plm = plm + zij_trunk
    ql = cl + npe.linear_r(rl)
    return cl, plm, ql


# ----------------------------------------------------------------------------------------------------------------- rollout setup / denoiser
def build_rollout_cache(dm, batch, z_loc, lay, no_rollout_samples: int, log=None) -> RolloutCache:
    """Everything step-invariant, ONCE per rollout at a point every rank reaches: the schedule, the ``z_cond`` rows, the bias cache object, the
    atom windows + band plan + ``plm_z``. Logged once per rank (the census line ``[diffusion] rollout ...``)."""
    D = C.seam("diffusion")
    C.fn("dist", "require_sharded")(lay, "sample_diffusion_rows")
    t0 = time.time()
    sched = decide_schedule(dm, lay, no_rollout_samples, z_loc)
    z_cond_loc = conditioning_pair_rows(dm.diffusion_conditioning, batch, z_loc, lay, rows=sched.cond_rows)
    win, plan, plm_z = atom_pair_term(dm, batch, z_cond_loc, lay, rows=sched.band_rows)
    cache = RolloutCache(lay, sched, z_cond_loc, D.PairBiasCache(enabled=sched.bias_cache), win, plan, plm_z, rows_core=_DECIDED_ROWS[0])   # the admitted rows core (None -> the statement)
    (log or C.comm().log)(f"[diffusion] rollout N={lay.N} P={lay.P} rows {lay.r0}:{lay.r1} z_cond_loc={tuple(z_cond_loc.shape)} {sched!r} "
                          f"band W={plan.W} (need {plan.w_need}) extra_rows={list(plan.extra_rows)[:8]}{'...' if len(plan.extra_rows) > 8 else ''} "
                          f"plm_z={tuple(plm_z.shape)} setup {time.time() - t0:.2f}s")
    return cache


@contextmanager
def swapped_forward(module, fn):
    """``module.forward`` = ``fn`` on this INSTANCE for the duration (the class and every other instance untouched)."""
    had = "forward" in module.__dict__
    prev = module.__dict__.get("forward")
    module.forward = fn
    try:
        yield module
    finally:
        if had:
            module.forward = prev
        else:
            del module.forward


@contextmanager
def sharded_denoiser(dm, cache: RolloutCache):
    """For the duration: ``dm(...)`` is ``DiffusionModule.forward`` on the shard — the stock method with ``DiffusionConditioning`` /
    ``DiffusionTransformer`` / ``NoisyPositionEmbedder`` served from ``cache`` (the four swaps of the module docstring)."""
    D = C.seam("diffusion")
    dc, dt, npe = dm.diffusion_conditioning, dm.diffusion_transformer, dm.atom_attn_enc.noisy_position_embedder
    stock_forward = type(dm).forward

    def dm_forward(batch, xl_noisy, token_mask, atom_mask, t, si_input, si_trunk, zij_trunk, use_conditioning, chunk_size=None, **flags):
        use_high_precision_attention = bool(flags.pop("use_high_precision_attention", False))
        _mask_trans = bool(flags.pop("_mask_trans", True))
        PS.refuse_kernel_flags("DiffusionModule", **flags)
        if not use_conditioning:
            raise DiffusionRefused("refused: DiffusionModule(use_conditioning=False) on pair rows (inference conditions on the trunk; the rollout passes True)")
        synced = D.sync_replicated(xl_noisy, "diffusion.xl_noisy")              # initial noise + augmentation + per-step noise: rank 0's state under `bcast`, proven identical under `guard`
        if synced is not xl_noisy:                                          # `bcast`: rank 0's state lands IN PLACE, so the sampler loop that owns this tensor (its `xl_noisy + dt * delta`
            xl_noisy.copy_(synced)                                          # update, diffusion_module.py:413-415) continues from the same state on every rank, not only the denoiser
        cache.steps += 1
        return stock_forward(dm, batch=batch, xl_noisy=xl_noisy, token_mask=token_mask, atom_mask=atom_mask, t=t, si_input=si_input, si_trunk=si_trunk,
                             zij_trunk=zij_trunk, use_conditioning=True, chunk_size=chunk_size, use_high_precision_attention=use_high_precision_attention,
                             _mask_trans=_mask_trans, **{k: False for k in PS.KERNEL_FLAGS})

    def dc_forward(batch, t, si_input, si_trunk, zij_trunk, use_conditioning, chunk_size=None):
        if getattr(dc, "chunk_size_tuner", None) is not None and chunk_size is not None:
            raise DiffusionRefused("refused: DiffusionConditioning.tune_chunk_size=true under the tp line (the launcher's yaml pins tune_chunk_size=false)")
        return conditioning_single(dc, batch, t, si_input, si_trunk, chunk_size=chunk_size), cache.z_cond_loc

    def dt_forward(a, s, z, mask=None, use_high_precision_attention=False, _mask_trans=True, **flags):
        if z is not cache.z_cond_loc:
            raise DiffusionRefused(f"refused: DiffusionTransformer got z {tuple(z.shape)} that is not the rollout's z_cond shard")
        return diffusion_transformer_rows(dt, a, s, z, mask, cache, use_high_precision_attention=use_high_precision_attention, _mask_trans=_mask_trans, **flags)

    def npe_forward(batch, cl, plm, si_trunk, zij_trunk, rl, n_query, n_key):
        return noisy_position_embedder_forward_band(npe, cache, batch, cl, plm, si_trunk, zij_trunk, rl, n_query, n_key)

    with swapped_forward(dm, dm_forward), swapped_forward(dc, dc_forward), swapped_forward(dt, dt_forward), swapped_forward(npe, npe_forward):
        yield dm


# ----------------------------------------------------------------------------------------------------------------- entry (model.rollout_rows)
def sample_diffusion_rows(sd, batch, si_input, si_trunk, z_loc, lay, noise_schedule, no_rollout_samples, use_conditioning=True, chunk_size=None,
                          _mask_trans=True, use_high_precision_attention=False, draws=None, chunk=None, **flags):
    """``SampleDiffusion.forward`` with ``zij_trunk`` = this rank's shard ``[1, 1, n_loc, N, c_z]`` -> the replicated sampled positions
    ``[1, S, N_atom, 3]`` (identical on every rank). The loop is OpenFold3's (``rollout_drawn`` when the sample loop drives the roll-out: ``draws`` / ``chunk`` -> samples ``[chunk.start, chunk.stop)`` of ``no_rollout_samples``, ``[1, k, N_atom, 3]``; the
    z_cond rows and the pair-bias cache are then sized for ``k`` samples)."""
    PS.refuse_kernel_flags("SampleDiffusion", **flags)
    if not use_conditioning:
        raise DiffusionRefused("refused: SampleDiffusion(use_conditioning=False) on pair rows (inference conditions on the trunk)")
    dm = sd.diffusion_module
    k = int(no_rollout_samples) if chunk is None else int(chunk.stop - chunk.start)
    cache = build_rollout_cache(dm, batch, z_loc, lay, k)
    try:
        with sharded_denoiser(dm, cache):
            if draws is not None:
                out = rollout_drawn(sd, batch, si_input, si_trunk, z_loc, noise_schedule, no_rollout_samples, chunk_size=chunk_size,
                                     use_high_precision_attention=use_high_precision_attention, _mask_trans=_mask_trans, draws=draws, chunk=chunk)
            else:
                out = sd(batch=batch, si_input=si_input, si_trunk=si_trunk, zij_trunk=z_loc, noise_schedule=noise_schedule, no_rollout_samples=no_rollout_samples,
                                 use_conditioning=True, chunk_size=chunk_size, use_high_precision_attention=use_high_precision_attention, _mask_trans=_mask_trans,
                                 **{k: False for k in PS.KERNEL_FLAGS})
        n_steps = int(len(noise_schedule) - 1)
        expected = n_steps + pocket_restart_steps(batch, n_steps)           # + the pocket-conditioned second roll-out from its start step (diffusion_module.py:443-472), 0 without pocket features
        if cache.steps != expected:
            raise DiffusionRefused(f"refused: the sharded denoiser served {cache.steps} calls for a {n_steps}-step schedule (+{expected - n_steps} pocket-restart steps expected) "
                                   "(SampleDiffusion.forward did not reach it)")
        bc = cache.bias_cache
        C.comm().log(f"[diffusion] rollout done: {cache.steps} steps x {int(no_rollout_samples)} samples; bias cache {'on' if bc.enabled else 'off'} "
                     f"hits={bc.hits} misses={bc.misses} held={bc.nbytes / 1e9:.2f} GB")
        D = C.seam("diffusion")
        if hasattr(D, "dit_rows_census"):                                    # the two slots' census words (schedule record + rank log)
            words = dict(dit_rows=D.dit_rows_census(), dit_bias=D.dit_bias_census(), dit_rows_core=cache.rows_core or "engine", diff_bias_producer=os.environ.get("ROWPAIR_DIFF_BIAS", "") or "engine")
            try:
                C.fn("evidence", "record_schedule")(**words)
            except Exception:                                                # noqa: BLE001 - a core without the record call: the log line only
                pass
            C.comm().log("[diffusion] census " + " ".join(f"{k}={v}" for k, v in words.items()))
        return out
    finally:
        cache.bias_cache.clear()
        del cache


def pocket_restart_steps(batch, n_steps: int) -> int:
    """The denoiser calls of ``SampleDiffusion.forward``'s pocket-conditioned restart (``diffusion_module.py:443-472``): ``T - start_step`` with
    ``start_step = max(0, min(T - 1, round(pocket_sampling_start_frac * T)))`` when ``pocket_constraints._pocket_sampling_enabled(batch)``, else 0."""
    from openfold3.core.model.structure.pocket_constraints import _batch_scalar, _pocket_sampling_enabled
    if not _pocket_sampling_enabled(batch):
        return 0
    T = int(n_steps)
    start_frac = _batch_scalar(batch, "pocket_sampling_start_frac", float)
    start_step = max(0, min(T - 1, int(round(float(start_frac) * T))))
    return T - start_step


# ----------------------------------------------------------------------------------------------------------------- the sample loop's drawn roll-out
def _drawn(draw, fn, chunk):
    """``fn()`` (a draw at the ALL-samples shape) sliced to ``chunk`` along the sample dim through the sample loop's ``draws`` (None = the plain draw)."""
    return fn() if draw is None else draw.draw(fn, chunk)


def centre_random_augmentation_drawn(xl, atom_mask, scale_trans=1.0, draws=None, chunk=None, n_samples=None):
    """``augmentation.centre_random_augmentation`` (AF3 Alg. 19, ``structure/augmentation.py:44-76``; re-exported by ``diffusion_module``), statements
    and RNG order verbatim (``sample_rotations``, then the translation draw; the masked centroid's denominator clamped at 1); under a sample loop
    (``draws``) both draws are made at the ``n_samples`` shape and sliced to ``chunk`` (``opt_core.mem.ckpt.StockOrderDraws``), so every sample's
    augmentation is the batched roll-out's."""
    from openfold3.core.model.structure import augmentation as _aug
    lead = tuple(xl.shape[:-2]) if draws is None else (int(xl.shape[0]), int(n_samples))
    rots = _drawn(draws, lambda: _aug.sample_rotations(shape=lead, dtype=xl.dtype, device=xl.device), chunk)
    trans = scale_trans * _drawn(draws, lambda: torch.randn((*lead, 3), dtype=xl.dtype, device=xl.device), chunk)
    mean_xl = torch.sum(xl * atom_mask[..., None], dim=-2, keepdim=True) / torch.sum(atom_mask[..., None], dim=-2, keepdim=True).clamp(min=1)   # no 0-div (augmentation.py:68)
    pos_centered = xl - mean_xl
    pos_out = pos_centered @ rots.transpose(-1, -2) + trans[..., None, :]
    pos_out = pos_out * atom_mask[..., None]
    return pos_out


def rollout_drawn(sd, batch, si_input, si_trunk, zij_trunk, noise_schedule, no_rollout_samples, chunk_size=None,
                  use_high_precision_attention=False, _mask_trans=True, draws=None, chunk=None):
    """``SampleDiffusion.forward`` / ``_sample_rollout``'s loop (AF3 Alg. 18, ``diffusion_module.py:349-485``) statements verbatim under the sample
    loop (``draws``, ``chunk``: ``opt_core.mem.sample_loop``): the roll-out covers samples ``[chunk.start, chunk.stop)`` of ``no_rollout_samples`` and
    its three draws (initial noise, augmentation, per-step noise) are made at the ``no_rollout_samples`` shape and sliced, so every sample's noise is
    the batched roll-out's. Identical on every rank. The pocket-conditioned restart of ``:442-472`` is never entered here: a pocket query runs the
    stock forward in one batched pass (``model.rollout_rows``)."""
    from openfold3.core.model.structure.pocket_constraints import _pocket_sampling_enabled
    if _pocket_sampling_enabled(batch):
        raise DiffusionRefused("refused: the sample loop's chunked roll-out with a pocket-conditioned query (pocket_sampling_enabled): SampleDiffusion's "
                               "pocket restart selects parents across all samples — model.rollout_rows runs such a query as the stock forward in one batched pass")
    atom_mask = batch["atom_mask"]
    batch_dim, num_atoms = atom_mask.shape[0], atom_mask.shape[-1]
    S_all = int(no_rollout_samples)
    if draws is None:
        chunk = slice(0, S_all)
    xl = noise_schedule[0] * _drawn(draws, lambda: torch.randn((batch_dim, S_all, num_atoms, 3), device=atom_mask.device, dtype=atom_mask.dtype), chunk)
    for tau, c_tau in enumerate(noise_schedule[1:]):
        xl = centre_random_augmentation_drawn(xl=xl, atom_mask=atom_mask, draws=draws, chunk=chunk, n_samples=S_all)   # xl = centre_random_augmentation(xl=xl, atom_mask=atom_mask)
        gamma = sd.gamma_0 if c_tau > sd.gamma_min else 0
        t = noise_schedule[tau] * (gamma + 1)
        noise = sd.noise_scale * torch.sqrt(t ** 2 - noise_schedule[tau] ** 2) * _drawn(draws, lambda: torch.randn((batch_dim, S_all, *xl.shape[-2:]), device=xl.device, dtype=xl.dtype), chunk)   # torch.randn_like(xl) at the S shape, sliced
        xl_noisy = xl + noise
        xl_denoised = sd.diffusion_module(batch=batch, xl_noisy=xl_noisy, token_mask=batch["token_mask"], atom_mask=atom_mask, t=t.to(xl_noisy.device),
                                          si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk, use_conditioning=True, chunk_size=chunk_size,
                                          use_high_precision_attention=use_high_precision_attention, _mask_trans=_mask_trans)
        delta = (xl_noisy - xl_denoised) / t
        dt = c_tau - t
        xl = xl_noisy + sd.step_scale * dt * delta
    return xl
