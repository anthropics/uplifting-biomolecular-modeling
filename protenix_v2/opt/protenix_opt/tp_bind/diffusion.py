"""Protenix-v2's diffusion module (``protenix/model/modules/diffusion.py`` ``DiffusionConditioning`` / ``DiffusionModule``,
``modules/transformer.py`` ``DiffusionTransformer`` = 24 x ``DiffusionTransformerBlock`` (``AttentionPairBias``, 16 heads) and the
``AtomAttentionEncoder``'s token-pair input) with the pair representation ROW-SHARDED: this module holds Protenix's attribute paths and the
tensor plumbing of its statements; the row layout, the block schedule and every collective are ``opt_core.mem.rowpair.diffusion``.

Per roll-out ONCE: the conditioned pair rows ``pair_z[R, N, c_z]`` = ``prepare_cache`` on this rank's rows of the trunk pair shard
(``diffusion.pair_cond_rows``: ``linear_no_bias_z(layernorm_z(cat[z rows, relpe(relp rows)]))`` then ``+transition_z1``, ``+transition_z2``
row-local; the relative-position feature rows are :func:`protenix_opt.tp_bind.relpos.relp_rows`, never an ``[N, N, 139]`` tensor); the atom encoder's token-pair term ``linear_no_bias_z(layernorm_z(pair_z))[tok(l), tok(m)]`` read from a diagonal
BAND of the shard (``diffusion.band_plan`` / ``pair_band_rows`` / ``band_lookup``: as wide as the valid atom-pair slots need; no rank ever
gathers ``[N, N, c]``) and added to the stock replicated ``prepare_cache(r_l=None, z=None)``; the 24 per-block attention biases of the local
query rows cached when the core's budget affords them (``diffusion.PairBiasCache``, ``DiffusionSchedule.decide``), else recomputed per step
from the shard. Per denoising step (``DiffusionModule.forward`` with the pair path on rows): single conditioning, atom encoder / decoder
and ``layernorm_a`` are the stock statements on REPLICATED tensors (atom attention is sequence-local: replicated by design); the
DiffusionTransformer is ``diffusion.diffusion_transformer_sharded`` over ``diffusion.DiTBlockFns`` bound to each block: ``norm`` =
``layernorm_a(a, s)`` (AdaLN, all rows), ``kv`` = ``attention._prep_qkv`` (q / k / v projections at the stock row count), ``attn`` =
``_attention`` (SDPA) for this rank's query rows against all keys with the bias rows, then ``_wrap_up`` (gate, output projection) on those
rows, ``update`` = the AdaLN-zero gate ``sigmoid(linear_a_last(s))``, the residual and ``conditioned_transition_block`` on those rows,
``bias`` = ``conv2d(normalize(z rows), w)`` (``enable_efficient_fusion``) or ``linear_nobias_z(layernorm_z(z rows))``; the core re-replicates
the updated token rows once per block (one all-gather). Every SDPA call of a roll-out issues the same query length on every rank (local rows
zero-padded to the schedule's query block: the kernel configuration cannot differ by rank). The sampling loop is the stock
``Protenix.sample_diffusion`` -> ``generator.sample_diffusion`` (noise schedule, initial noise, random augmentation, per-step noise, Euler
update on replicated tensors); the RNG states are rank 0's at roll-out start (``dist.broadcast_obj``) and every step's noisy coordinates are
rank 0's on every rank at every denoiser call (``diffusion.sync_replicated``, ``ROWPAIR_DIFF_NOISE_SYNC=bcast`` under the line) and the
final coordinates are rank 0's on every rank (``adopt_rank0_coordinates``: cross-rank spread recorded as ``diff_rank_spread_A``, refused above 1 Å). Precision is the stock policy: ``skip_amp`` (autocast off,
fp32) or the ambient bf16 autocast; no statement here changes a dtype the stock statement would not.

Row blocks: the pair-conditioning, bias and band statements walk the layout's block ``B`` (the trunk's chunk grid: P-invariant numerics);
the query block is the whole local shard (``q_rows=0``: one SDPA call per block and rank, int32-capped by the core). ``DiffusionSchedule.decide``
prints them with the bias-cache decision (``diff_cond_rows diff_bias_rows diff_q_rows diff_band_rows diff_bias_cache diff_band_w ...``).
Refused by name (``RowpairRefused``): a P == 1 / replicated layout (the single-GPU line installs nothing; ``dist.require_sharded``), a missing
``pair_z`` cache (``enable_diffusion_shared_vars_cache`` off), a sampler-hook spec (``ROWPAIR_SAMPLER_HOOK``: the
stock loop has no hook statement), training-time guidance (``sample_diffusion.guidance.enable``), an ``enable_efficient_fusion`` flag at a
step other than the roll-out's. Entry points keep the carried seam's call signatures (``ptx_tp/trunk.py`` calls them through
``protenix_opt.tp_route``): :func:`tp_prepare_cache`, :func:`tp_sample_diffusion`, :func:`report` (``modes.sample_diffusion`` read by
``protenix_opt.report.unit_record``)."""
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from opt_core.mem.rowpair import RowpairRefused
from opt_core.mem.rowpair import ckpt as CK
from opt_core.mem.rowpair import diffusion as D
from opt_core.mem.rowpair import dist as DI
from opt_core.mem.rowpair import sampler_hook as SH
from opt_core.mem.rowpair.evidence import record_schedule

from .relpos import relp_rows

__all__ = ["tp_prepare_cache", "tp_sample_diffusion", "report", "adopt_rank0_coordinates", "bias_cache_decision", "RANK_SPREAD_MAX_A", "ENV_BAND_W", "SAMPLER_HOOK_ENVS",
           "ENV_BIAS_CACHE_POLICY", "BIAS_CACHE_POLICIES", "ENV_BIAS_CACHE_DTYPE", "BIAS_CACHE_DTYPES", "bias_cache_dtype", "FIT_FRAC", "FIT_MARGIN_GB", "HEADROOM_FRAC", "ENV_DIT_ROWS", "DIT_ROWS_DEFAULT", "dit_rows_word"]

WHAT = "protenix_opt.tp_bind.diffusion"
ENV_OP_TIMERS = "PTX_TP_OP_TIMERS"                          # engineering (=1): per-section device-synchronised seconds of the denoiser, one line per rank at sampler exit
ENV_BAND_W = D.ENV_BAND_W                                 # the adapter-read cap of the atom-pair band half-width (API.md); unset = as wide as needed
SAMPLER_HOOK_ENVS = (SH.ENV,)
ATOM_KEYS = ("ref_pos", "ref_charge", "ref_mask", "ref_element", "ref_atom_name_chars", "atom_to_token_idx", "d_lm", "v_lm", "pad_info")
_REPORT: Dict[str, Dict[str, Any]] = {"modes": {}, "buffers": {}}


def report() -> Dict[str, Dict[str, Any]]:
    """``{"modes": {"sample_diffusion": {mode, skip_amp, N, P, R, N_sample, N_step, fusion, bias_cache, band_w, rows}, "prepare_cache": {...}},
    "buffers": {name: bytes}}`` of the last roll-out in this rank (``protenix_opt.report.unit_record`` reads ``modes.sample_diffusion``)."""
    return {k: dict(v) for k, v in _REPORT.items()}


def _buffer(name: str, t) -> None:
    if t is None:
        _REPORT["buffers"].pop(name, None)
    else:
        _REPORT["buffers"][name] = int(t.numel()) * int(t.element_size())


def _amp_off(fp32: bool):
    """The stock precision policy of the diffusion module: ``autocasting_disable_decorator(skip_amp)`` disables autocast (fp32 statements);
    otherwise the ambient autocast applies."""
    return torch.autocast(device_type="cuda", enabled=False) if fp32 else _Null()


class _Null(object):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _up(t, fp32: bool):
    """``autocasting_disable_decorator``'s conditioned cast: floating tensors enter fp32 statements as fp32."""
    return t.to(torch.float32) if (fp32 and torch.is_floating_point(t) and t.dtype != torch.float32) else t


# ----------------------------------------------------------------------------------------------------------------- C1: pair conditioning rows
def tp_prepare_cache(cond_mod, input_feature_dict: Dict[str, Any], z_trunk_shard, layout, *, skip_amp: Optional[bool] = None,
                     inplace_safe: bool = False, chunk_rows: Optional[int] = None):
    """``DiffusionConditioning.prepare_cache`` on this rank's rows: returns the conditioned pair shard ``pair_z[R, N, c_z]`` (rows ``[r0, r1)``
    of the dense statement) in the dtype the stock statement produces under ``skip_amp``. ``chunk_rows``: the row block (default the layout
    block ``B``). ``inplace_safe`` is accepted for the call signature (the row statement adds the transitions in place into its own output)."""
    DI.require_sharded(layout, f"{WHAT}.tp_prepare_cache")
    if skip_amp is None:
        raise RowpairRefused(f"{WHAT}.tp_prepare_cache: skip_amp must be passed (configs.skip_amp.sample_diffusion); the precision policy is never guessed")
    fp32 = bool(skip_amp)
    rows = int(chunk_rows) if chunk_rows else int(layout.B)
    r_max, s_max = int(cond_mod.relpe.r_max), int(cond_mod.relpe.s_max)

    def embed(z_rows, g0: int, g1: int):                    # diffusion.py 93-100 on rows: linear_no_bias_z(layernorm_z(cat[z, relpe(relp)]))
        zc = _up(z_rows, fp32)
        rel = relp_rows(input_feature_dict, g0, g1, r_max, s_max)
        if rel.device != zc.device:
            rel = rel.to(zc.device)
        return cond_mod.linear_no_bias_z(cond_mod.layernorm_z(torch.cat([zc, cond_mod.relpe(rel)], dim=-1)))

    transitions = [lambda x, g0, g1: cond_mod.transition_z1(x), lambda x, g0, g1: cond_mod.transition_z2(x)]   # diffusion.py 101-105 (row-local)
    with _amp_off(fp32):
        probe = embed(z_trunk_shard[..., :1, :, :], layout.r0, layout.r0 + 1)          # one row: the dtype the stock statement yields here
        out = torch.empty(tuple(z_trunk_shard.shape[:-3]) + (layout.n_loc, layout.N, int(cond_mod.c_z)), dtype=probe.dtype, device=probe.device)
        del probe
        pair_z = D.pair_cond_rows(embed, z_trunk_shard, layout, c_out=int(cond_mod.c_z), rows=rows, transitions=transitions, out=out)
    _REPORT["modes"]["prepare_cache"] = dict(mode="tp", rows=rows, skip_amp=fp32, dtype=str(pair_z.dtype), N=layout.N, P=layout.P, R=layout.n_loc)
    _buffer("pair_z(shard)", pair_z)
    return pair_z


# ----------------------------------------------------------------------------------------------------------------- F2: the atom encoder's pair band
def _atom_pair_cache(enc, feats: Dict[str, Any], pair_z_shard, layout, fp32: bool, band_rows: int, band_w: Optional[int]):
    """``AtomAttentionEncoder.prepare_cache(r_l=True, z=pair_z)`` = the stock replicated ``prepare_cache(r_l=None, z=None)`` (everything but
    the token-pair term) ``.unsqueeze(-5) +`` the token-pair term of every atom-pair slot, read from the band of the projected shard
    (``transformer.py`` 806-817; ``primitives.broadcast_token_to_local_atom_pair``). Returns ``(p_lm, c_l, plan)``."""
    from protenix.model.modules.primitives import rearrange_qk_to_dense_trunk
    from protenix.utils.torch_utils import autocasting_disable_decorator
    kw = {k: feats[k] for k in ATOM_KEYS}
    p_base, c_l = autocasting_disable_decorator(fp32)(enc.prepare_cache)(**kw, r_l=None, z=None, inplace_safe=False)
    a2t = feats["atom_to_token_idx"]
    idx_q, idx_k, _ = rearrange_qk_to_dense_trunk(a2t, a2t, dim_q=-1, dim_k=-1, n_queries=enc.n_queries, n_keys=enc.n_keys, compute_mask=False)
    valid = feats["pad_info"]["mask_trunked"].to(torch.bool)
    lead = (1,) * max(0, 4 - valid.dim())                    # the core's plan takes [F, n_blocks, n_q(, n_k)] (F = 1: Protenix inference carries no batch dim)
    plan = D.band_plan(idx_q.long().reshape(lead + tuple(idx_q.shape)), idx_k.long().reshape(lead + tuple(idx_k.shape)),
                       valid.reshape(lead + tuple(valid.shape)), layout.N, max_w=band_w)
    with _amp_off(fp32):
        proj = lambda zr: enc.linear_no_bias_z(enc.layernorm_z(_up(zr, fp32)))     # noqa: E731  transformer.py 809 on rows
        band, extras = D.pair_band_rows(proj, pair_z_shard, layout, plan, rows=band_rows)
        y = D.band_lookup(band, extras, plan)                # [1, n_blocks, n_q, n_k, c_atompair]: the stock gather's values slot by slot
        del band, extras
        p_lm = p_base.unsqueeze(dim=-5) + (y[0] if lead else y)   # transformer.py 806-817: [1, n_blocks, n_q, n_k, c_atompair]
        del y
    return p_lm, c_l, plan


# ----------------------------------------------------------------------------------------------------------------- F4: DiffusionTransformer block callables
def _sdpa(att, q, k, v, bias, q_len: int, inplace_safe: bool):
    """``primitives._attention`` for ``q[*, H, n, d]`` with ``n`` zero-padded to ``q_len`` (every rank issues the same query length)."""
    from protenix.model.modules.primitives import _attention
    n = int(q.shape[-2])
    if 0 < n < q_len:
        pad = q_len - n
        q = torch.cat([q, q.new_zeros(tuple(q.shape[:-2]) + (pad, int(q.shape[-1])))], dim=-2)
        bias = torch.cat([bias, bias.new_zeros(tuple(bias.shape[:-2]) + (pad, int(bias.shape[-1])))], dim=-2)
    o = _attention(q=q, k=k, v=v, attn_bias=bias, use_efficient_implementation=att.use_efficient_implementation, inplace_safe=inplace_safe)
    return o[..., :n, :] if n < q_len else o


def _block_fns(blk, dm, *, fusion: bool, q_len: int, inplace_safe: bool, bias_dtype: Optional[torch.dtype] = None,
               rows_core: Optional[str] = None, stock_q_rows: Optional[int] = None) -> "D.DiTBlockFns":
    """One ``DiffusionTransformerBlock`` as the core's callables (``transformer.py`` 201-256, 303-353; ``primitives.Attention`` 718-880);
    ``bias_dtype``: the dtype the pair bias leaves ``bias()`` in (None = as computed, fp32; the cached set's dtype under :func:`bias_cache_decision`);
    ``rows_core``: the row word the roll-out admitted for the query-block attention (:func:`dit_rows_word`; the core's
    ``dit_attention_rows`` on ``kernels.apb.pair_bias_attention_rows`` — k / v projected and cast to bf16 ONCE per block, q per query block, the
    bias rows read in place, fp32 statistics / accumulation; the statement below is its fallback BY NAME in chunks of ``stock_q_rows``) or None =
    the engine statement (``_sdpa`` = ``primitives._attention`` on the query rows). Every block also carries ``bias_into`` (``DitBias``: the
    producer word ``ROWPAIR_DIFF_BIAS=engine|ln_proj``; engine = ``bias()`` value for value)."""
    from protenix.model.utils import permute_final_dims
    apb = blk.attention_pair_bias
    att = apb.attention
    H = int(att.num_heads)

    def norm(a, s):                                            # AttentionPairBias 246-249: AdaLN on all rows (replicated)
        return apb.layernorm_a(a=a, s=s)

    def kv(x):                                                 # Attention._prep_qkv: q / k / v projections (+ q scaling) at the stock row count
        return att._prep_qkv(q_x=x, kv_x=x, apply_scale=True)

    def attn(x_q, qkv, bias_q, rows):                          # Attention.forward 782-880 for query rows [g0, g1): SDPA vs all keys, then gate + linear_o on those rows
        g0, g1 = rows
        q, k, v = qkv
        b = bias_q.reshape((1,) * (q.dim() - bias_q.dim()) + tuple(bias_q.shape))
        o = _sdpa(att, q[..., g0:g1, :], k, v, b, q_len, inplace_safe)             # [*, H, rows, d]
        return att._wrap_up(o.transpose(-2, -3), x_q)                              # [*, rows, c_a]

    def update(a, o, s, rows):                                 # AttentionPairBias 251-256 + DiffusionTransformerBlock 348-353 on rows [r0, r1)
        r0, r1 = rows
        a_r, s_r = a[..., r0:r1, :], s[..., r0:r1, :]
        if inplace_safe:
            o *= torch.sigmoid(apb.linear_a_last(s_r))
            attn_out = o                                       # drop_path = identity at inference
            attn_out += a_r
        else:
            o = torch.sigmoid(apb.linear_a_last(s_r)) * o
            attn_out = o + a_r
        ff_out = blk.conditioned_transition_block(a=attn_out, s=s_r)
        return ff_out + attn_out

    if fusion:                                                 # diffusion.py 447-449 + transformer.py 181-185: conv2d(normalize(z) permuted, w_z * w_ln)
        def bias(z_rows):
            w = (apb.linear_nobias_z.weight * apb.layernorm_z.weight[None, :])[:, :, None, None]
            zn = permute_final_dims(dm.normalize(z_rows.to(dtype=torch.float32)), [2, 0, 1]).contiguous()   # [*, c_z, rows, N]
            squeeze = zn.dim() == 3
            y = F.conv2d(zn.unsqueeze(0) if squeeze else zn, w)                    # [*, H, rows, N]
            y = (y[0] if squeeze else y).movedim(-3, -1)                             # [*, rows, N, H] (the core stores it heads-first)
            return y if bias_dtype is None else y.to(dtype=bias_dtype)
    else:                                                      # transformer.py 187-190: linear_nobias_z(layernorm_z(z fp32))
        def bias(z_rows):
            y = apb.linear_nobias_z(apb.layernorm_z(z_rows.to(dtype=torch.float32)))
            return y if bias_dtype is None else y.to(dtype=bias_dtype)
    ln = apb.layernorm_z                                       # the producer as WEIGHTS for the core's one-pass LN + projection (ROWPAIR_DIFF_BIAS=ln_proj; engine = bias() above):
    bias_into = D.DitBias(engine=bias, ln_weight=getattr(ln, "weight", None), ln_bias=None if fusion else getattr(ln, "bias", None),   # the fused path folds the LN
                          weight=apb.linear_nobias_z.weight, linear_bias=None, eps=float(getattr(ln, "eps", 1e-5)))              # weight and has no LN offset term
    if rows_core:                                              # the query-block attention on the rows face; `attn` / `kv` above captured BEFORE the names are rebound
        stock_attn, stock_kv = attn, kv

        def kv(x):                                             # k / v of ALL rows once per block (bf16 under the tier word: cast16), the engine's (q, k, v) as the fallback's kv
            qkv = stock_kv(x)
            return D.DitKV(D.cast16(qkv[1]), D.cast16(qkv[2]), None, qkv)

        def q_fn(x_q):                                         # _prep_qkv's q for these query rows, UNSCALED (the face applies D ** -0.5)
            q = att.linear_q(x_q)
            return q.view(q.shape[:-1] + (H, -1)).transpose(-2, -3)

        def out_fn(o, x_q):                                    # o [.., S, q, H*D] -> _wrap_up(o [.., q, H, D], q_x): sigmoid(linear_g(q_x)) gate + linear_o
            return att._wrap_up(o.view(o.shape[:-1] + (H, -1)), x_q)

        attn = D.dit_attention_rows(q_fn=q_fn, out_fn=out_fn, stock_fn=stock_attn, num_heads=H, core_word=rows_core, scale=None,
                                    stock_q_rows=stock_q_rows)
    return D.DiTBlockFns(norm=norm, kv=kv, attn=attn, update=update, bias=bias, bias_into=bias_into)


class _Denoiser(object):
    """``DiffusionModule.forward`` (``diffusion.py`` 331-600) with the pair path on rows; called by the stock sampling loop in place of the module."""

    def __init__(self, dm, feats, pair_z_shard, p_lm, c_l, layout, sched, *, fp32: bool, fusion: bool, inplace_safe: bool,
                 cache_dtype: Optional[torch.dtype] = None, rows_core: Optional[str] = None):
        self.dm, self.feats, self.pair_z, self.p_lm, self.c_l, self.layout, self.sched = dm, feats, pair_z_shard, p_lm, c_l, layout, sched
        self.fp32, self.fusion, self.inplace_safe = fp32, fusion, inplace_safe
        q_rows = int(sched.q_rows)
        self.q_len = q_rows if 0 < q_rows < layout.Rmax else int(layout.Rmax)
        q_stock = int(getattr(sched, "q_rows_stock", 0) or 0)
        self.q_len_stock = q_stock if 0 < q_stock < layout.Rmax else int(layout.Rmax)   # the engine statement's query block (its own path, and the face's fallback chunks)
        self.cache_dtype = cache_dtype if bool(sched.bias_cache) else None             # a cached set's dtype; a per-row-block recompute stays as computed (fp32)
        self.rows_core = rows_core
        self.blocks = [_block_fns(b, dm, fusion=fusion, q_len=self.q_len_stock, inplace_safe=inplace_safe, bias_dtype=self.cache_dtype,
                                  rows_core=rows_core, stock_q_rows=self.q_len_stock if rows_core else None)
                       for b in dm.diffusion_transformer.blocks]
        self.cache = D.PairBiasCache(enabled=bool(sched.bias_cache))
        self.calls = 0
        self.timers = {} if os.environ.get(ENV_OP_TIMERS, "") == "1" else None      # engineering: device-synchronised seconds per denoiser section

    def _tick(self, name: str, t0: float) -> float:
        if self.timers is None:
            return 0.0
        torch.cuda.synchronize()
        t1 = time.time()
        self.timers[name] = self.timers.get(name, 0.0) + (t1 - t0)
        return t1

    def __call__(self, x_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk=None, pair_z=None, p_lm=None, c_l=None,
                 inplace_safe: bool = False, chunk_size: Optional[int] = None, use_conditioning: bool = True, enable_efficient_fusion: bool = False):
        from protenix.model.utils import expand_at_dim
        dm, feats = self.dm, self.feats
        if bool(enable_efficient_fusion) != self.fusion:
            raise RowpairRefused(f"{WHAT}: enable_efficient_fusion={enable_efficient_fusion} at a step, {self.fusion} at roll-out start")
        if p_lm is None or c_l is None:
            raise RowpairRefused(f"{WHAT}: the atom-pair cache (p_lm, c_l) must be built once per roll-out; a per-step prepare_cache on a shard is refused")
        synced = D.sync_replicated(x_noisy, "diffusion.x_noisy")                   # the augmentation + noise draws are replicated (guard | bcast | off)
        if synced is not x_noisy:                                                    # bcast handed back rank 0's tensor: write it INTO the sampler loop's own
            x_noisy.copy_(synced)                                                    # variable (this object is the tensor the stock loop's update statement reads:
                                                                                     # delta = (x_noisy - x_denoised) / t_hat; x_l = x_noisy + ...), never a rebound local
        with _amp_off(self.fp32):
            t0 = self._tick("sync", time.time()) if self.timers is not None else 0.0
            x_noisy, t_hat_noise_level, s_inputs, s_trunk = (_up(t, self.fp32) for t in (x_noisy, t_hat_noise_level, s_inputs, s_trunk))
            # DiffusionModule.forward 553-557: EDM input scaling
            r_noisy = x_noisy / torch.sqrt(dm.sigma_data ** 2 + t_hat_noise_level ** 2)[..., None, None]
            # f_forward 392-404: conditioning (the pair path is the cached shard; no clone: nothing below writes into it)
            s_single, _ = dm.diffusion_conditioning(t_hat_noise_level=t_hat_noise_level, relp_feature=None, s_inputs=s_inputs, s_trunk=s_trunk,
                                                    z_trunk=None, pair_z=self.pair_z, inplace_safe=False, use_conditioning=use_conditioning)
            s_trunk_e = expand_at_dim(s_trunk, dim=-3, n=1)
            t0 = self._tick("cond", t0)
            # f_forward 430-447: atom encoder (replicated; reads the cached p_lm / c_l, never the pair tensor)
            a_token, q_skip, c_skip, p_skip = dm.atom_attention_encoder(
                feats["atom_to_token_idx"], feats["ref_pos"], feats["ref_charge"], feats["ref_mask"], feats["ref_atom_name_chars"],
                feats["ref_element"], feats["d_lm"], feats["v_lm"], feats["pad_info"], r_l=r_noisy, s=s_trunk_e, z=self.pair_z, p_lm=p_lm,
                c_l=c_l, inplace_safe=inplace_safe, chunk_size=chunk_size)
            a_token = a_token.to(dtype=torch.float32)
            t0 = self._tick("atom_enc", t0)
            if inplace_safe:                                    # f_forward 449-456
                a_token += dm.linear_no_bias_s(dm.layernorm_s(s_single))
            else:
                a_token = a_token + dm.linear_no_bias_s(dm.layernorm_s(s_single))
            # f_forward 457-477: the DiffusionTransformer with token queries = this rank's rows (the core re-replicates a once per block)
            a_token = D.diffusion_transformer_sharded(self.blocks, a_token.to(dtype=torch.float32), s_single.to(dtype=torch.float32), self.pair_z,
                                                     self.layout, schedule=self.sched, bias_cache=self.cache)
            a_token = dm.layernorm_a(a_token)
            t0 = self._tick("dit", t0)
            # f_forward 478-500: atom decoder (replicated)
            r_update = dm.atom_attention_decoder(atom_to_token_idx=feats["atom_to_token_idx"], a=a_token, q_skip=q_skip, c_skip=c_skip,
                                                 p_skip=p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size)
            t0 = self._tick("atom_dec", t0)
            # DiffusionModule.forward 572-581: EDM output combination
            s_ratio = (t_hat_noise_level / dm.sigma_data)[..., None, None].to(r_update.dtype)
            x_denoised = 1 / (1 + s_ratio ** 2) * x_noisy + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio ** 2) * r_update
        self.calls += 1
        return x_denoised


# ----------------------------------------------------------------------------------------------------------------- the roll-out
def _sync_rng_from_rank0() -> str:
    """Every rank continues rank 0's python / numpy / torch CPU / CUDA RNG streams (the stock loop's ``randn`` draws are then identical)."""
    if not DI.is_dist():
        return "none"
    st = DI.broadcast_obj(CK.rng_state() if DI.world()[1] == 0 else None, src=0)
    if DI.world()[1] != 0:
        CK.set_rng_state(st)
    return "rank0"


def _refuse_hooks() -> None:
    for env in SAMPLER_HOOK_ENVS:
        if (os.environ.get(env) or "").strip():
            raise RowpairRefused(f"{WHAT}: {env}={os.environ[env]!r}: the stock sampling loop (generator.sample_diffusion) has no sampler-hook "
                                 f"statement; a hook needs the engine loop with the core's call sites (opt_core.mem.rowpair.sampler_hook)")


ENV_DIT_ROWS = "ROWPAIR_DIT_ROWS"                          # big (default) | apb_attn | sba[:tf32|ieee|tf32x3] | sdpa[:…] | naive | engine — the DiT query-block attention's
DIT_ROWS_DEFAULT = "big"                                 #   row word on the core's rows face (kernels.apb.pair_bias_attention_rows; the tier word =
                                                           #   the measured rows cell's winner, else apb_attn by name); engine | off = the engine statement (primitives._attention
                                                           #   per query block: the [S, H, q, N] fp32 logits). Census dit_rows=<row>:<n> (served) | stock:<kind>:<n> (fallback by name)
ENV_BIAS_CACHE_POLICY = "ROWPAIR_DIFF_BIAS_CACHE_POLICY"      # fit (default) | headroom | core — the rule of :func:`bias_cache_decision`
BIAS_CACHE_POLICIES = ("fit", "headroom", "core")
ENV_BIAS_CACHE_DTYPE = "ROWPAIR_DIFF_BIAS_CACHE_DTYPE"        # fp16 (default) | bf16 | fp32 — the dtype of a CACHED bias set (a recomputed row block stays fp32)
BIAS_CACHE_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
FIT_FRAC = 0.9                                             # policy fit: the share of the agreed usable bytes the cached set + margin may take
FIT_MARGIN_GB = 4.0                                        #   and the margin left for the sampler's own row-block transients (ROWPAIR_DIFF_WORK_GB class)
HEADROOM_FRAC = 0.8                                        # policy headroom: the share of the agreed allocator headroom (peak-neutral by construction)


def dit_rows_word() -> str:
    """The DiT rows attention word of this process (:data:`ENV_DIT_ROWS`; '' when the word is engine / off: nothing bound)."""
    w = (os.environ.get(ENV_DIT_ROWS) or DIT_ROWS_DEFAULT).strip().lower()
    return "" if w in ("engine", "off", "none", "stock", "0") else w


def bias_cache_dtype() -> torch.dtype:
    """The dtype of a CACHED DiT pair-bias set (:data:`ENV_BIAS_CACHE_DTYPE`, default fp16: the operand class the single-GPU line's DiT
    attention kernel runs these biases in on cc 9.0 — dit_attn's fp16 arms; a bias enters the fp32 logits by promotion). fp32 = the tag's."""
    word = (os.environ.get(ENV_BIAS_CACHE_DTYPE) or "fp16").strip().lower()
    if word not in BIAS_CACHE_DTYPES:
        raise RowpairRefused(f"{WHAT}: {ENV_BIAS_CACHE_DTYPE}={word!r}: one of {' | '.join(BIAS_CACHE_DTYPES)}")
    return BIAS_CACHE_DTYPES[word]


def _agreed_min(v: int) -> int:
    if not DI.is_dist():
        return int(v)
    c = DI.comm()
    t = torch.tensor([int(v)], dtype=torch.int64, device=c.device)
    c.allreduce_(t, "min")
    return int(t.item())


def bias_cache_decision(layout, *, cache_bytes: int):
    """Whether the DiffusionTransformer's 24 per-block pair biases of the local query rows (``n_blocks * H * Rmax * N * elt`` bytes; they
    depend on the conditioned pair rows only, so a cached set is computed ONCE per roll-out instead of once per row block per block per step)
    are cached this roll-out -> ``(decision, census word)``;
    ``decision`` None hands the rule to the core (``DiffusionSchedule.decide``: ``ROWPAIR_DIFF_BIAS_CACHE=0|1`` pins, else on iff the set fits
    ``ROWPAIR_DIFF_BIAS_CACHE_GB``). Every policy takes ONE decision for all ranks (an int64 min-allreduce of the measured bytes).

    ``fit`` (default: decide from measured free memory at sampler entry with a margin, bounded): on iff ``cache_bytes +
    FIT_MARGIN_GB <= FIT_FRAC * usable`` with ``usable`` = device-free bytes (``mem_get_info``) + the allocator's own free pool (``reserved -
    allocated``) — it cannot OOM, and where the set does not fit the core recomputes per row block (word ``fit:off:<need>/<usable>GiB``).
    ``headroom``: on iff the set fits :data:`HEADROOM_FRAC` of ``max_memory_allocated - memory_allocated`` (the trunk's high-water mark:
    the item's peak cannot rise; stricter). ``core``: the core's rule. A caller's ``ROWPAIR_DIFF_BIAS_CACHE=0|1`` pin always wins (word ``env``)."""
    pin = (os.environ.get(D.ENV_BIAS_CACHE) or "").strip().lower()
    if pin in ("0", "1", "on", "off"):
        return None, "env"
    policy = (os.environ.get(ENV_BIAS_CACHE_POLICY) or "fit").strip().lower()
    if policy not in BIAS_CACHE_POLICIES:
        raise RowpairRefused(f"{WHAT}: {ENV_BIAS_CACHE_POLICY}={policy!r}: one of {' | '.join(BIAS_CACHE_POLICIES)}")
    if policy == "core" or not torch.cuda.is_available():
        return None, policy
    if policy == "headroom":
        room = _agreed_min(int(torch.cuda.max_memory_allocated()) - int(torch.cuda.memory_allocated()))
        on = int(cache_bytes) <= int(room * HEADROOM_FRAC)
    else:
        free, _total = torch.cuda.mem_get_info()
        room = _agreed_min(int(free) + int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated()))
        on = int(cache_bytes) + int(FIT_MARGIN_GB * 2 ** 30) <= int(room * FIT_FRAC)
    return on, "%s:%s:%.1f/%.1fGiB" % (policy, "on" if on else "off", cache_bytes / 2 ** 30, room / 2 ** 30)


def tp_sample_diffusion(model, input_feature_dict: Dict[str, Any], s_inputs, s_trunk, z_trunk_shard, pair_z_shard, layout, N_sample: int,
                        N_step: int, **kw):
    """``Protenix.sample_diffusion`` (its config plumbing and autocast policy) -> ``generator.sample_diffusion`` (the stock loop) with the
    denoiser of this module. ``kw``: ``skip_amp`` (required: ``configs.skip_amp.sample_diffusion``), ``inplace_safe`` (default True as stock
    inference), ``noise_schedule`` (default ``model.inference_noise_scheduler(N_step)``). ``z_trunk_shard`` is accepted for the call signature
    and not read (the conditioned shard replaces it, as stock passes ``z_trunk=None`` beside a ``pair_z`` cache). Returns the coordinates
    ``[N_sample, N_atom, 3]``, replicated."""
    DI.require_sharded(layout, f"{WHAT}.tp_sample_diffusion")
    _refuse_hooks()
    skip_amp = kw.pop("skip_amp", None)
    if skip_amp is None:
        raise RowpairRefused(f"{WHAT}.tp_sample_diffusion: skip_amp must be passed (configs.skip_amp.sample_diffusion)")
    fp32 = bool(skip_amp)
    inplace_safe = bool(kw.pop("inplace_safe", True))
    noise_schedule = kw.pop("noise_schedule", None)
    if kw:
        raise RowpairRefused(f"{WHAT}.tp_sample_diffusion: unknown arguments {sorted(kw)} (no mode overrides: the row statements are the only path)")
    if pair_z_shard is None:
        raise RowpairRefused(f"{WHAT}.tp_sample_diffusion: pair_z shard is None (enable_diffusion_shared_vars_cache off): the conditioned pair "
                             f"rows are built once per roll-out by tp_prepare_cache, never per step")
    guidance = model.configs.sample_diffusion.to_dict().get("guidance") or {}
    if guidance.get("enable"):
        raise RowpairRefused(f"{WHAT}.tp_sample_diffusion: sample_diffusion.guidance.enable: guided sampling calls the denoiser outside the row schedule")
    dm = model.diffusion_module
    enc, dt = dm.atom_attention_encoder, dm.diffusion_transformer
    fusion = bool(getattr(model, "enable_efficient_fusion", False))
    B = int(layout.B)
    band_w = int(os.environ[ENV_BAND_W]) if (os.environ.get(ENV_BAND_W) or "").strip() else None
    H = int(dt.blocks[0].attention_pair_bias.n_heads)
    elt = 4 if fp32 else 2
    cache_dtype = bias_cache_dtype()
    cache_elt = torch.empty((), dtype=cache_dtype).element_size()
    cache_on, cache_word = bias_cache_decision(layout, cache_bytes=len(dt.blocks) * H * int(layout.Rmax) * int(layout.N) * cache_elt)
    rows_word = dit_rows_word()                                # the query-block attention's row word, admitted ONCE per roll-out (static, identical on every rank)
    att0 = dt.blocks[0].attention_pair_bias.attention
    attn_core, _sel, rows_why = D.dit_rows_core(rows_word, dtype=torch.float32 if fp32 else torch.bfloat16, heads=H, head_dim=int(att0.c_hidden), samples=int(N_sample))
    rows_core = rows_word if (attn_core == "kernel" or rows_word == "naive") else None
    sched = D.DiffusionSchedule.decide(layout, c_z=int(dm.c_z), c_in=2 * int(dm.c_z), c_cond=int(dm.c_z), H=H, S=int(N_sample), n_blocks=len(dt.blocks),
                                     c_pair=int(enc.c_atompair), elt=elt, cond_rows=B, bias_rows=B, q_rows=0, band_rows=B, bias_cache=cache_on, attn_core=attn_core)
    dtype_word = str(cache_dtype).replace("torch.", "") if sched.bias_cache else "fp32:recompute"
    record_schedule(diff_bias_cache_policy=cache_word, diff_bias_cache_dtype=dtype_word, dit_rows_word=rows_word or "engine")
    os.write(2, (f"[protenix-opt] TP-DIFF sampler_entry rank={layout.rank}/{layout.P} diff_bias_cache={int(bool(sched.bias_cache))} "
                 f"diff_bias_cache_policy={cache_word} diff_bias_cache_dtype={dtype_word} bias_rows={sched.bias_rows} q_rows={sched.q_rows} "
                 f"dit_rows_word={rows_word or 'engine'} dit_rows_core={rows_core or 'engine'}({rows_why}) dit_bias_word={D.dit_bias_word() if hasattr(D, 'dit_bias_word') else 'engine'} "
                 f"N={layout.N} R={layout.n_loc}\n").encode())
    rng = _sync_rng_from_rank0()
    if noise_schedule is None:
        noise_schedule = model.inference_noise_scheduler(N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype)
    p_lm, c_l, plan = _atom_pair_cache(enc, input_feature_dict, pair_z_shard, layout, fp32, sched.band_rows, band_w)
    _buffer("p_lm", p_lm)
    denoise = _Denoiser(dm, input_feature_dict, pair_z_shard, p_lm, c_l, layout, sched, fp32=fp32, fusion=fusion, inplace_safe=inplace_safe,
                        cache_dtype=cache_dtype, rows_core=rows_core)
    _REPORT["modes"]["sample_diffusion"] = dict(mode="tp", skip_amp=fp32, N=layout.N, P=layout.P, R=layout.n_loc, N_sample=int(N_sample), N_step=int(N_step),
                                                fusion=fusion, bias_cache=bool(sched.bias_cache), rows=B, q_len=denoise.q_len, band_w=int(plan.W),
                                                band_w_need=int(plan.w_need), band_extra_rows=len(plan.extra_rows), rng_sync=rng)
    record_schedule(diff_engine="protenix_v2", diff_fusion=int(fusion), diff_q_len=denoise.q_len, diff_rng_sync=rng)
    stock = getattr(type(model).sample_diffusion, "__wrapped__", type(model).sample_diffusion)   # Protenix.sample_diffusion itself (kit wrappers add census only)
    # the loop carries pair_z=None (the denoiser holds the shard: the loop's fp32 cast of its tensor arguments never copies the pair rows)
    try:
        coords = stock(model, denoise_net=denoise, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=None,
                       pair_z=None, p_lm=p_lm, c_l=c_l, N_sample=N_sample, noise_schedule=noise_schedule, inplace_safe=inplace_safe,
                       enable_efficient_fusion=fusion)
    finally:
        rows_census = D.dit_rows_census() if hasattr(D, "dit_rows_census") else "none"
        bias_census = D.dit_bias_census() if hasattr(D, "dit_bias_census") else "none"
        record_schedule(diff_denoise_calls=denoise.calls, diff_bias_cache_bytes=denoise.cache.nbytes, dit_rows=rows_census, dit_bias=bias_census)
        _REPORT["modes"]["sample_diffusion"]["denoise_calls"] = denoise.calls
        _REPORT["modes"]["sample_diffusion"]["dit_rows"] = rows_census
        os.write(2, (f"[protenix-opt] TP-DIFF sampler_done rank={layout.rank}/{layout.P} calls={denoise.calls} N_sample={int(N_sample)} "
                     f"dit_rows={rows_census} dit_bias={bias_census} cache_gib={denoise.cache.nbytes / 2 ** 30:.2f} "
                     f"max_alloc_gib={(torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else 0:.2f}\n").encode())
        if denoise.timers is not None:
            os.write(2, (f"[protenix-opt] TP-OPTIMERS sampler rank={layout.rank}/{layout.P} calls={denoise.calls} N_sample={int(N_sample)} "
                         f"bias_cache={int(bool(sched.bias_cache))} cache_gib={denoise.cache.nbytes / 2 ** 30:.2f} q_len={denoise.q_len} "
                         + " ".join("%s=%.2f" % kv for kv in denoise.timers.items()) + "\n").encode())
        denoise.cache.clear()
        del denoise, p_lm, c_l
        _buffer("p_lm", None)
    return adopt_rank0_coordinates(coords)


RANK_SPREAD_MAX_A = 1.0                                   # sampler exit: the cross-rank spread of the final coordinates above this is refused by name


def adopt_rank0_coordinates(coords):
    """Sampler exit: every rank continues with RANK 0's coordinates (the confidence head's sharded pair rows then embed identical
    structures on every rank). The cross-rank spread ``max |x_rank - x_rank0|`` (Å) is all-reduced and recorded first
    (``diff_rank_spread_A``; 0 under deterministic kernels, ~1e-3 Å otherwise: the loop's replicated statements are recomputed per rank)
    and a spread above ``RANK_SPREAD_MAX_A`` is refused by name — the per-call state broadcast (``diffusion.x_noisy``) keeps the ranks on
    one trajectory; this closes the last denoiser call's output. Without a process group: ``coords`` unchanged."""
    if not DI.is_dist():
        return coords
    mine = coords.detach().clone()
    synced = D.sync_replicated(coords, "diffusion.x_out", mode="bcast")                  # rank 0's tensor on every rank, then the guard
    spread = (mine.float() - synced.float()).abs().amax().reshape(1)
    DI.allreduce_(spread, op="max")
    spread_a = float(spread.item())
    record_schedule(diff_noise="bcast_rank0_state", diff_rank_spread_A=f"{spread_a:.3e}")
    _REPORT["modes"]["sample_diffusion"]["rank_spread_A"] = spread_a
    P, rank = DI.world()
    os.write(2, (f"[protenix-opt] TP-DIFF sampler_exit rank={rank}/{P} diff_noise=bcast_rank0_state diff_rank_spread_A={spread_a:.3e} "
                 f"noise_sync={os.environ.get('ROWPAIR_DIFF_NOISE_SYNC', 'guard')} (refused above {RANK_SPREAD_MAX_A} A)\n").encode())   # fd 2: the rank's stderr (the module imports os / torch / the core / protenix only)
    if spread_a > RANK_SPREAD_MAX_A:
        raise RowpairRefused(f"{WHAT}: sampler exit: the ranks' final coordinates differ by {spread_a:.3f} A (> {RANK_SPREAD_MAX_A} A): "
                             f"the replicated sampling loop diverged across ranks (diff_rank_spread_A)")
    return synced
