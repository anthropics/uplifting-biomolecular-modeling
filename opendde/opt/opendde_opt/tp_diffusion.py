"""The structural-token DIFFUSION stage of the ``big`` line under ``--n_gpu P > 1``: the diffusion pair conditioning and the diffusion
transformer's pair biases are this rank's ROWS of the structural pair space, bound to the core's row-sharded diffusion drivers
(:mod:`opt_core.mem.rowpair.diffusion`); nothing ``[N_st, N_st, c]`` is resident on a device.

Structural tokens: ``N_st`` rows (the structural-token expansion of the ``N`` residue tokens, ``N_st ~ 1.3-2 N``); this rank owns rows
``[r0, r1)`` of ``Layout.auto(N_st, P, rank)`` (:func:`struct_layout`) — a layout of its own, independent of the residue trunk's.

Statements (each is the engine's, on rows; per-element arithmetic identical to the dense statement, GEMM ``M`` = local rows):

  * pair conditioning ONCE per roll-out (``DiffusionConditioning.prepare_cache``): ``z_cond[i] = transitions(linear(LN(cat[proj(z_st[i]),
    relpe(relp rows i)])))`` for local rows ``i`` -> ``z_cond [R_st, N_st, c_pair]`` device-resident for the roll-out
    (``diffusion.pair_cond_rows``; the structural pair rows are read from wherever the structural stage left them — host-resident
    ``HostPair`` / host tensor (H2D per row block, only rows ``[r0, r1)`` are ever read), a host or device row block ``[R_st, N_st, c_z]``,
    or a :class:`opendde_opt.tp.RowShard`; named ``zstruct_src`` in the census).
  * the 24 transformer blocks' pair biases ``linear(LN(z_cond rows))`` (or the fused ``conv2d(normalize(z_cond rows))`` form when the
    model runs ``enable_efficient_fusion``) per local row block, cached across the steps x samples of the roll-out when the schedule says
    they fit (``diffusion.PairBiasCache``; ``diff_bias_cache=on|off``).
  * ``DiffusionTransformer``: per block, AdaLN + K/V of all tokens (replicated), attention with QUERIES = local rows (+ the structural
    extra attention bias rows), the AdaLN-zero gate, residual and conditioned transition on local rows, ONE all-gather of the updated rows
    (``diffusion.diffusion_transformer_sharded`` with :func:`dit_block_fns`).
  * the atom encoder's token-pair term ``p_lm += linear(LN(z_cond[q_tok, k_tok]))``: the projection is row-local, the (query token, key
    token) lookup reads a diagonal BAND of the projected rows all-gathered ``[N_st, 2W+1, c_atompair]`` plus the named out-of-band rows
    (``diffusion.band_plan`` / ``pair_band_rows`` / ``band_lookup``), ONCE per roll-out into the cached ``p_lm``.
  * the sampler loop, noise schedule, random augmentation and noise draws are the engine's (``generator.sample_diffusion``), identically
    seeded on every rank; ``diffusion.sync_replicated`` makes the noisy positions agree at every denoiser call (rank 0's state broadcast,
    or every rank's own proven bitwise under ``ROWPAIR_DIFF_NOISE_SYNC=guard`` / the deterministic recipe: ``opendde_opt.tp._noise_sync``)
    and rank 0's sampled coordinates replace every rank's at the end (cross-rank spread recorded; a divergent rank is refused by name).

Replicated by design (named here and in the census): the single conditioning ``s`` (per step, O(N_st)), the token activation ``a
[S, N_st, c_token]`` between blocks, atoms / coordinates, the atom attention encoder and decoder, ``p_lm`` / ``c_l``, the structural extra
attention bias when the caller hands it whole (``dit_extra_bias=full``; ``rows`` when handed this rank's rows: :func:`shard_extra_bias`).

Refused by name (``RowpairRefused``): ``P == 1`` / no process group (at ``--n_gpu 1`` the engine's own diffusion module runs: this module
installs nothing), a Fold-CP transformer variant (``OPENDDE_FOLDCP_MODE=distributed``), ``cross_attention_mode`` blocks, an
``enable_efficient_fusion`` value at a denoiser call other than the primed one, ``use_conditioning=False``, training-free guidance, a
structural pair source whose shape is not ``[N_st, N_st, c]`` / ``[R_st, N_st, c]``.

Census (``fields()`` and ``opendde_opt.tp.STATS``): ``zcond_rows`` (+R_st per roll-out), ``dit_local_queries`` (+R_st per block call),
``diff_denoise_calls``, ``diff_band_terms``, ``zstruct_src``, ``zstruct_host_gb``, ``dit_bias=ln_linear|fused_conv``, ``dit_extra_bias``,
``struct_layout``, and the core's schedule record (``diff_cond_rows diff_bias_rows diff_q_rows diff_band_rows diff_bias_cache
diff_rows_source diff_band_W diff_band_extra_rows diff_noise_sync``).
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
from typing import Any, Optional

__all__ = ["StructDiffusion", "struct_layout", "prime", "prepare_cache_sharded", "denoise_sharded", "sharded_denoiser",
           "run_sample_diffusion_stage", "shard_extra_bias", "dit_block_fns", "fields", "STATS", "EXTRA_BIAS_KEY", "EXTRA_BIAS_ROWS_KEY"]

EXTRA_BIAS_KEY = "structural_pair_attn_bias"              # the structural stage's extra attention bias in the (structural) feature dict
EXTRA_BIAS_ROWS_KEY = "_rowpair_struct_extra_bias_rows"   # marker: EXTRA_BIAS_KEY holds this rank's rows [R_st, N_st] (shard_extra_bias)
ENV_BAND_W = "ROWPAIR_DIFF_BAND_W"                        # band_plan(max_w=): unset = as wide as the valid atom pairs need (no row travels whole)
ENV_BAND_EXTRA_MAX = "ROWPAIR_DIFF_BAND_EXTRA_MAX"        # band_plan(max_extra_rows=)
STATS = {"zcond_rows": 0, "dit_local_queries": 0, "diff_denoise_calls": 0, "diff_band_terms": 0, "diff_rollouts": 0,
         "zstruct_src": None, "zstruct_host_gb": 0.0, "dit_bias": None, "dit_extra_bias": None, "struct_layout": None, "noise_guards": 0, "rank_spread_A": None,
         "dit_attn_kernel": None, "dit_attn_kernel_calls": 0, "dit_attn_stock_calls": 0, "dit_attn_unsupported": {}, "dit_attn_maxerr": None,
         "dit_attn_cast": {}, "dit_bias_form": None, "dit_bias_calls": 0}
ENV_DIT_BIAS = "ODDE_TP_DIT_BIAS"                           # the per-block pair-bias statement of the sharded roll-out: folded (default: LayerNorm statistics by one
                                                           # reduction + ONE [c_pair -> H] GEMM with the LayerNorm affine folded into the projection) | ln_linear (the engine's
                                                           # layernorm_z -> linear_nobias_z statement through the LayerNorm extension)
ENV_DIT_KERNEL = "ODDE_TP_DIT_KERNEL"                       # the DiT local-row attention kernel of the sharded roll-out: apb_attn (default: the core's rectangular
                                                           # pair-bias attention kernel, queries = this rank's rows, keys = all N_st) | stock (the engine's _attention statement)
ENV_DIT_KERNEL_CHECK = "ODDE_TP_DIT_KERNEL_CHECK"           # =<n>: the first n served calls also run the engine's statement and record max |kernel - stock| (diagnostic; 0)
REPLICATED_BY_DESIGN = "s_single,a[S,N_st,c_token],atoms,atom_encoder,atom_decoder,p_lm,c_l"
_LAYOUTS: dict = {}


# ----------------------------------------------------------------------------------------------------------------- plumbing
def _core():
    from opt_core.mem.rowpair import diffusion as DF, dist as D, RowpairRefused
    return DF, D, RowpairRefused


def _refuse(msg: str):
    from opt_core.mem.rowpair import RowpairRefused
    from . import tp as _tp
    raise RowpairRefused(f"{_tp.LEVER}: refused: diffusion stage: {msg}")


def _cfg(node, key: str, default=None):
    """``node[key]`` / ``node.key`` of an engine config node (mapping or attribute style), else ``default``."""
    if node is None:
        return default
    if hasattr(node, "get"):
        try:
            v = node.get(key, default)
            return default if v is None else v
        except Exception:
            pass
    return getattr(node, key, default)


def _bump(key: str, n: int = 1) -> None:
    STATS[key] = STATS.get(key, 0) + n
    from . import tp as _tp                                  # the adapter's census carries the two reserved keys
    if key in ("zcond_rows", "dit_local_queries"):
        _tp.STATS[key] = _tp.STATS.get(key, 0) + n


def struct_layout(n_struct: int):
    """The row layout of the structural pair space ``[N_st, N_st]`` over the ranks of the active group (``Layout.auto(N_st, P, rank)``:
    the core's block grid; refused by name when no grid shards ``N_st`` over ``P``). Cached per ``(N_st, P, rank)``."""
    _DF, D, _ = _core()
    from . import tp as _tp
    P, r = _tp._group()
    key = (int(n_struct), int(P), int(r))
    lay = _LAYOUTS.get(key)
    if lay is None:
        lay = D.Layout.auto(int(n_struct), int(P), int(r), lever=_tp.LEVER)
        _LAYOUTS[key] = lay
    STATS["struct_layout"] = lay.facts()
    return lay


class _ZRows(object):
    """This rank's structural pair rows ``[R_st, N_st, c_z]`` as a (host or device) tensor VIEW plus where they live. Sources: a
    ``HostPair``-like object (``.t`` = the ``[N_st, N_st, c]`` host tensor), a tensor ``[N_st, N_st, c]`` (host or device: only rows
    ``[r0, r1)`` are viewed), a row block ``[R_st, N_st, c]``, or a :class:`opendde_opt.tp.RowShard` of THIS layout."""
    __slots__ = ("rows", "src", "host_gb", "c")

    def __init__(self, z, lay):
        import torch
        from . import tp as _tp
        full_gb = 0.0
        if isinstance(z, _tp.RowShard):
            if int(z.lay.N) != lay.N or int(z.lay.r0) != lay.r0 or int(z.lay.R) != lay.R:
                _refuse(f"RowShard layout {z.lay!r} is not the structural layout {lay!r}")
            t, src = z.z, "device_rows" if z.z.device.type != "cpu" else "host_rows"
        elif hasattr(z, "t") and torch.is_tensor(getattr(z, "t")):                 # the offload unit's HostPair (host-resident full z_struct)
            t, src = z.t, "hostpair_full"
        elif torch.is_tensor(z):
            t, src = z, None
        else:
            _refuse(f"structural pair source {type(z).__name__} is not a HostPair / tensor / RowShard")
        if t.dim() != 3 or int(t.shape[-2]) != lay.N:
            _refuse(f"structural pair source {tuple(t.shape)}: expected [N_st={lay.N}, N_st, c] or [R_st={lay.R}, N_st, c]")
        if int(t.shape[0]) == lay.N and lay.N != lay.R:
            full_gb = t.numel() * t.element_size() / 1e9 if t.device.type == "cpu" else 0.0
            if t.device.type != "cpu" and src is None:
                src = "device_full"                                                  # a device-resident full z_struct: NAMED (the caller's floor, not this module's)
            t = t[lay.r0:lay.r1]
            src = src or "host_full"
        elif int(t.shape[0]) == lay.R:
            src = src or ("host_rows" if t.device.type == "cpu" else "device_rows")
        else:
            _refuse(f"structural pair source rows={int(t.shape[0])}: neither N_st={lay.N} nor this rank's R_st={lay.R}")
        self.rows, self.src, self.host_gb, self.c = t, src, full_gb, int(t.shape[-1])


# ----------------------------------------------------------------------------------------------------------------- engine statements on rows
def _single_cond(dc, t_hat_noise_level, s_inputs, s_trunk, inplace_safe: bool):
    """== ``DiffusionConditioning.forward`` single path (diffusion.py: cat -> linear(LN) -> + noise embedding -> two transitions);
    replicated (O(N_st) per step)."""
    import torch
    single_s = torch.cat(tensors=[s_trunk, s_inputs], dim=-1)
    single_s = dc.linear_no_bias_s(dc.layernorm_s(single_s))
    noise_ratio = (t_hat_noise_level / dc.sigma_data).clamp(min=1e-10)
    noise_n = dc.fourier_embedding(t_hat_noise_level=torch.log(input=noise_ratio) / 4).to(single_s.dtype)
    single_s = single_s.unsqueeze(dim=-3) + dc.linear_no_bias_n(dc.layernorm_n(noise_n)).unsqueeze(dim=-2)
    if inplace_safe:
        single_s += dc.transition_s1(single_s)
        single_s += dc.transition_s2(single_s)
    else:
        single_s = single_s + dc.transition_s1(single_s)
        single_s = single_s + dc.transition_s2(single_s)
    return single_s


def _cond_embed_fn(dc, relp, zrows: _ZRows, device):
    """``embed(z_rows, g0, g1)``: == ``DiffusionConditioning.prepare_cache`` up to the transitions, on GLOBAL rows ``[g0, g1)``:
    ``linear(LN(cat[project_z_trunk(z_rows), relpe.linear(relp rows)])))`` with the relative-position rows materialised lazily
    (``relp.materialize(rows, all)``: no ``[N_st, N_st, c_rel]`` one-hot) and the structural rows moved to the device per block."""
    import torch

    def embed(z_rows, g0, g1):
        zr = (z_rows.to(device, non_blocking=False) if z_rows.device != device else z_rows).to(dtype=torch.float32)   # a bf16 structural shard (struct_pair_bf16) is read as fp32 rows
        zt = dc._project_z_trunk(zr) if dc.compress_pair_z else zr
        rel = relp.materialize(slice(g0, g1), slice(None)) if hasattr(relp, "materialize") else relp[..., g0:g1, :, :]
        rel = dc.relpe.linear_no_bias(rel.to(device))
        out = dc._project_pair_z(torch.cat([zt, rel], dim=-1))
        del zr, zt, rel
        return out
    return embed


def _cond_transitions(dc):
    """The two pair transitions of ``_apply_pair_z_transitions`` as row-local callables (``x += t1(x)`` over all rows, then ``t2``)."""
    return [lambda x, g0, g1: dc.transition_z1(x), lambda x, g0, g1: dc.transition_z2(x)]


def _extra_rows(extra, lay):
    """``(tensor, offset)``: the structural extra attention bias as handed — whole ``[.., N_st, N_st]`` (offset 0: rows indexed
    globally) or this rank's rows ``[.., R_st, N_st]`` (offset r0) — or ``(None, 0)``."""
    if extra is None:
        return None, 0
    if int(extra.shape[-1]) != lay.N:
        _refuse(f"{EXTRA_BIAS_KEY} {tuple(extra.shape)}: columns != N_st={lay.N}")
    if int(extra.shape[-2]) == lay.N and lay.N != lay.R:
        return extra, 0
    if int(extra.shape[-2]) == lay.R:
        return extra, lay.r0
    _refuse(f"{EXTRA_BIAS_KEY} {tuple(extra.shape)}: rows are neither N_st={lay.N} nor this rank's R_st={lay.R}")


def shard_extra_bias(feats: dict, lay) -> None:
    """The structural extra attention bias of ``feats`` -> this rank's rows ``[.., R_st, N_st]`` (a contiguous copy; the whole tensor is
    released). Idempotent (marker :data:`EXTRA_BIAS_ROWS_KEY`). Call only when nothing after the diffusion stage reads the whole bias."""
    if feats.get(EXTRA_BIAS_ROWS_KEY) or feats.get(EXTRA_BIAS_KEY) is None:
        return
    t = feats[EXTRA_BIAS_KEY]
    if int(t.shape[-2]) != lay.N or int(t.shape[-1]) != lay.N:
        _refuse(f"shard_extra_bias: {EXTRA_BIAS_KEY} {tuple(t.shape)} is not [.., N_st={lay.N}, N_st]")
    feats[EXTRA_BIAS_KEY] = t[..., lay.r0:lay.r1, :].contiguous()
    feats[EXTRA_BIAS_ROWS_KEY] = True


def dit_block_fns(blk, lay, *, extra=None, fusion: bool = False, normalize=None, inplace_safe: bool = False):
    """``diffusion.DiTBlockFns`` of one opendde ``DiffusionTransformerBlock`` (transformer.py: ``AttentionPairBias`` with ``has_s`` +
    ``ConditionedTransitionBlock``): ``norm`` = the AdaLN of all tokens; ``kv`` = K / V heads of all tokens (``Attention.linear_k/v``);
    ``attn`` = Q of the query rows (``linear_q``, scaled), logits + local pair-bias rows (+ the structural extra bias rows ``[g0, g1)``:
    ``_add_extra_attn_bias_to_chunk``), ``_align_bias_to_query``, the engine's ``_attention`` kernel, ``_wrap_up`` (query gate, output
    projection) — statement for statement ``_standard_multihead_attention_stream_pair_bias`` on one row chunk; ``update`` =
    ``sigmoid(linear_a_last(s_rows)) * o + a_rows`` then ``+ conditioned_transition_block(., s_rows)``; ``bias`` = ``linear_nobias_z(
    layernorm_z(z_rows))`` (``fusion``: ``conv2d(normalize(z_rows), W * ln_w)``, the engine's ``enable_efficient_fusion`` form)."""
    DF, _D, _ = _core()
    import torch
    import torch.nn.functional as F
    from opendde.model.modules.primitives import _attention
    apb, ctb = blk.attention_pair_bias, blk.conditioned_transition_block
    att = apb.attention
    if getattr(apb, "cross_attention_mode", False):
        _refuse("DiffusionTransformerBlock with cross_attention_mode=True (separate K/V AdaLN) is not wired")
    if not getattr(apb, "has_s", True):
        _refuse("DiffusionTransformerBlock without single conditioning (has_s=False)")
    ex, ex_off = _extra_rows(extra, lay)
    H = int(att.num_heads)
    scale = 1.0 / math.sqrt(att.c_hidden)
    on_cuda = bool(getattr(getattr(apb.linear_nobias_z, "weight", None), "is_cuda", False))   # CPU tensors (the kit's gloo / CPU suites): both words below resolve to the ENGINE'S
    kern = _dit_kernel() if on_cuda else "stock"                                    # statements by name — the CUDA kernels cannot run there and the folded projection is a CUDA
    STATS["dit_attn_kernel"] = kern if on_cuda else "stock:cpu"                     # performance statement, so rows stay bitwise the dense module on CPU; on CUDA:
                                                                                    # "apb_attn" (the core's rectangular pair-bias kernel, default) | "stock"

    def norm(a, s):
        return apb.layernorm_a(a=a, s=s)                                           # AdaLN of every token (K / V need all rows)

    def kv(x):
        k = att.linear_k(x)
        v = att.linear_v(x)
        k = k.view(k.shape[:-1] + (H, -1)).transpose(-2, -3)                       # == Attention._prep_qkv (k, v of all rows, once per block)
        v = v.view(v.shape[:-1] + (H, -1)).transpose(-2, -3)
        return (k, v)

    def attn(x_q, kvh, bias_q, g):
        g0, g1 = g
        q = att.linear_q(x_q)
        q = q.view(q.shape[:-1] + (H, -1)).transpose(-2, -3)
        q = q / math.sqrt(att.c_hidden)                                             # == _prep_qkv(apply_scale=True)
        bias = bias_q                                                               # [*, H, q, N_st] (heads first, this block's query rows)
        if ex is not None:
            bias = apb._add_extra_attn_bias_to_chunk(bias, ex, g0 - ex_off, g1 - ex_off)
        bias = apb._align_bias_to_query(bias, x_q, n_pair_dims=2)
        o = _apb_rows(q, kvh[0], kvh[1], bias) if kern == "apb_attn" else None      # [*, q, H, d] from the core kernel, or None (named: dit_attn_unsupported)
        if o is None:
            STATS["dit_attn_stock_calls"] += 1
            o = _attention(q=q, k=kvh[0], v=kvh[1], attn_bias=bias, use_efficient_implementation=att.use_efficient_implementation,
                           inplace_safe=inplace_safe)                               # [*, H, q, d]: softmax over ALL N_st keys per query row
            o = o.transpose(-2, -3)
        return att._wrap_up(o, x_q)                                                 # query gate + output projection -> [*, q, c_a]

    def update(a, o_rows, s, rr):
        r0, r1 = rr
        s_rows = s[..., r0:r1, :]
        x = torch.sigmoid(apb.linear_a_last(s_rows)) * o_rows                       # adaLN-Zero output gate (AttentionPairBias.forward, has_s)
        x = x + a[..., r0:r1, :]                                                    # attn_out + a
        return ctb(a=x, s=s_rows) + x                                               # ff_out + attn_out

    form = _dit_bias_form() if on_cuda else "ln_linear"                          # folded (default on CUDA) | ln_linear (the engine's statements below; always on CPU)
    STATS["dit_bias_form"] = (("fused_conv" if fusion else "ln_linear") if form == "ln_linear" else ("folded_fused" if fusion else "folded_ln_linear")) + ("" if on_cuda else ":cpu")
    if fusion and normalize is None:
        _refuse("enable_efficient_fusion: the DiffusionModule.normalize LayerNorm is required for the fused pair-bias form")
    if form == "ln_linear" and fusion:
        w = (apb.linear_nobias_z.weight * apb.layernorm_z.weight[None, :])[:, :, None, None]

        def bias(z_rows):                                                           # == f_forward: normalize(z) -> [c, rows, N] -> conv2d
            STATS["dit_bias_calls"] += 1
            zn = normalize(z_rows.to(dtype=torch.float32))
            b = F.conv2d(zn.permute(2, 0, 1).contiguous(), w.to(zn.dtype))         # [H, rows, N_st]
            return b.permute(1, 2, 0)                                               # [rows, N_st, H] (the core writes heads first)
    elif form == "ln_linear":
        def bias(z_rows):
            STATS["dit_bias_calls"] += 1
            return apb.linear_nobias_z(apb.layernorm_z(z_rows.to(dtype=torch.float32)))   # [rows, N_st, H]; a bf16 structural shard is read as fp32 rows
    else:
        ln, lin = apb.layernorm_z, apb.linear_nobias_z
        fold = {}

        def _folded(device):
            """Once per block: the statement's LayerNorm affine folded into the projection. fused form (``conv2d(normalize(z), W_z * ln_w)``):
            scale = ln_w * normalize.weight (if any), shift = normalize.bias (if any), eps = normalize.eps — the engine's fused form carries
            no layernorm_z bias, and neither does this; LN->linear form: scale = ln_w, shift = ln_b, eps = layernorm_z.eps.
            W' = diag(scale) . W_z^T [c, H];  s = colsum(W') [H];  c0 = W_z . shift (+ linear bias if any) [H]."""
            if not fold:
                with torch.no_grad():
                    w = lin.weight.detach().to(device=device, dtype=torch.float32)   # [H, c]
                    c = int(w.shape[1])
                    one, zero = torch.ones(c, device=device), None
                    lw = ln.weight.detach().to(device=device, dtype=torch.float32) if getattr(ln, "weight", None) is not None else one
                    if fusion:
                        nw = getattr(normalize, "weight", None); nb = getattr(normalize, "bias", None)
                        scale = lw * (nw.detach().to(device=device, dtype=torch.float32) if nw is not None else one)
                        shift = nb.detach().to(device=device, dtype=torch.float32) if nb is not None else zero
                        eps = float(getattr(normalize, "eps", 1e-5))
                    else:
                        scale = lw
                        lb = getattr(ln, "bias", None)
                        shift = lb.detach().to(device=device, dtype=torch.float32) if lb is not None else zero
                        eps = float(getattr(ln, "eps", 1e-5))
                    Wt = (w * scale[None, :]).contiguous()                            # [H, c] (= W'^T)
                    c0 = (w @ shift) if shift is not None else torch.zeros(int(w.shape[0]), device=device)
                    lb2 = getattr(lin, "bias", None)
                    if lb2 is not None and not fusion:
                        c0 = c0 + lb2.detach().to(device=device, dtype=torch.float32)
                    fold.update(Wt=Wt, s=Wt.sum(1), c0=c0, eps=eps)
            return fold

        def bias(z_rows):                                                           # == the statement above, refactored: rstd * (W' z - mu s) + c0, written HEADS FIRST
            STATS["dit_bias_calls"] += 1                                            # (one var_mean reduction + one [H, c] x [c, rows*N] GEMM; no LayerNorm pass, no
            f = _folded(z_rows.device)                                              # [rows, N, c] fp32 temporary, no transpose copy)
            z = z_rows.to(dtype=torch.float32)
            rows, n, c = int(z.shape[-3]), int(z.shape[-2]), int(z.shape[-1])
            if z.dim() != 3:
                _refuse(f"pair-bias rows: z rows {tuple(z.shape)} carry a batch dim (the folded form serves [rows, N_st, c])")
            zf = z.reshape(rows * n, c)
            var, mu = torch.var_mean(zf, dim=-1, unbiased=False)                    # [rows*N]
            y = torch.matmul(f["Wt"], zf.t())                                       # [H, rows*N] (TN GEMM on the row-major rows: no operand copy)
            y.sub_(f["s"][:, None] * mu[None, :]).mul_(torch.rsqrt(var.add_(f["eps"]))[None, :]).add_(f["c0"][:, None])
            return y.view(-1, rows, n).permute(1, 2, 0)                             # [rows, N_st, H] view of the contiguous [H, rows, N_st] (the core's heads-first copy is a memcpy)

    return DF.DiTBlockFns(norm, kv, attn, update, bias)


def _dit_bias_form() -> str:
    """``ODDE_TP_DIT_BIAS``: ``folded`` (default) | ``ln_linear``; anything else refused by name. Consulted for CUDA modules only: on CPU the
    engine's statement runs (``dit_bias_form=…:cpu``)."""
    v = (os.environ.get(ENV_DIT_BIAS) or "folded").strip()
    if v not in ("folded", "ln_linear"):
        _refuse(f"{ENV_DIT_BIAS}={v!r} (folded | ln_linear)")
    return v


def _dit_kernel() -> str:
    """``ODDE_TP_DIT_KERNEL``: ``apb_attn`` (default) | ``stock``; anything else refused by name. Consulted for CUDA modules only: on CPU the
    engine's ``_attention`` statement runs (``dit_attn_kernel=stock:cpu``)."""
    v = (os.environ.get(ENV_DIT_KERNEL) or "apb_attn").strip()
    if v not in ("apb_attn", "stock"):
        _refuse(f"{ENV_DIT_KERNEL}={v!r} (apb_attn | stock)")
    return v


def _apb_rows(q, k, v, bias):
    """The DiT's local-row attention on the core's rectangular pair-bias kernel (``opt_core.kernels.apb_attn.apb_attention``: online
    softmax over all ``N_st`` keys for this rank's query rows, bf16 / fp16 operands, fp32 bias). ``q [*, H, q, d]`` (scaled),
    ``k / v [*, H, N_st, d]``, ``bias [*, H, q, N_st]`` -> ``[*, q, H, d]`` in q's dtype, or None when the kernel names the operands
    unsupported (``Unsupported.event`` counted in ``dit_attn_unsupported``; the engine's statement then serves the call). The first
    ``ODDE_TP_DIT_KERNEL_CHECK`` served calls also run the engine's statement and keep the largest |difference| (``dit_attn_maxerr``)."""
    import torch
    from opt_core.kernels import apb_attn as _ak
    H, nq, d = int(q.shape[-3]), int(q.shape[-2]), int(q.shape[-1])
    nk = int(k.shape[-2])
    lead = tuple(q.shape[:-3])
    S = 1
    for x in lead:
        S *= int(x)
    try:
        qs, ks, vs = q.reshape(S, H, nq, d), k.reshape(-1, H, nk, d), v.reshape(-1, H, nk, d)
        if qs.dtype == torch.float32:                                                # the sampler's fp32 statement (upstream runs the roll-out autocast-off below its own
            qs, ks, vs = qs.to(torch.bfloat16), ks.to(torch.bfloat16), vs.to(torch.bfloat16)   # size gate): bf16 operands, fp32 accumulation and bias, output back in fp32 — the
            STATS["dit_attn_cast"]["float32->bfloat16"] = STATS["dit_attn_cast"].get("float32->bfloat16", 0) + 1   # x1 provider's operand form at this site (dit_attn fast/big rows)
        if int(ks.shape[0]) != S:                                                    # k / v of one sample broadcast against S query samples: expand (a view)
            ks, vs = ks.expand(S, H, nk, d), vs.expand(S, H, nk, d)
        b = bias.reshape(-1, H, nq, nk) if bias.dim() > 4 or bias.dim() < 4 else bias
        if int(b.shape[0]) not in (1, S):
            b = b.expand(S, H, nq, nk) if int(b.shape[0]) == 1 else b.reshape(S, H, nq, nk)
        if int(b.shape[0]) == S and S > 1 and b.stride(0) == 0:                     # a broadcast view of one plane: hand the kernel the shared plane
            b = b[:1]
        for t in (qs, ks, vs):
            if t.stride(-1) != 1:
                raise _ak.Unsupported("stride:d_not_unit", "reshape left a non-unit head-dim stride")
        o = _ak.apb_attention(qs, ks, vs, b, scale=1.0)                              # [S, q, H*d] in q's dtype (q carries 1/sqrt(d) already)
    except _ak.Unsupported as e:
        ev = getattr(e, "event", None) or type(e).__name__
        STATS["dit_attn_unsupported"][ev] = STATS["dit_attn_unsupported"].get(ev, 0) + 1
        return None
    STATS["dit_attn_kernel_calls"] += 1
    o = o.view(*lead, nq, H, d)
    if o.dtype != q.dtype:
        o = o.to(q.dtype)
    n_check = int(os.environ.get(ENV_DIT_KERNEL_CHECK, "0") or 0)
    if n_check > 0 and STATS["dit_attn_kernel_calls"] <= n_check:
        from opendde.model.modules.primitives import _attention
        ref = _attention(q=q, k=k, v=v, attn_bias=bias, use_efficient_implementation=False, inplace_safe=False).transpose(-2, -3)
        err = float((o.float() - ref.float()).abs().max())
        STATS["dit_attn_maxerr"] = max(err, STATS["dit_attn_maxerr"] or 0.0)
    return o


# ----------------------------------------------------------------------------------------------------------------- the roll-out context
class StructDiffusion(object):
    """One prediction's row-sharded diffusion conditioning (built by :func:`prime`): ``lay`` (structural layout), ``sched``
    (``DiffusionSchedule``), ``zc`` (this rank's conditioned pair rows ``[R_st, N_st, c_pair]``, device), ``cache`` (``PairBiasCache``),
    ``blocks`` (``DiTBlockFns`` per transformer block), ``fusion`` (the primed ``enable_efficient_fusion``), ``n_sample``. It is ALSO the
    object the sampler carries as ``pair_z`` (the stock stage code only tests it against None): the installed denoiser recognises it."""

    def __init__(self, model, lay):
        self.model, self.lay = model, lay
        self.sched = self.zc = self.cache = self.blocks = None
        self.fusion = bool(getattr(model, "enable_efficient_fusion", False))
        self.n_sample = 1
        self.zsrc = None
        self.band_facts = {}
        self.denoise_calls = 0

    # the offload unit's stage timer / diagnostics probe pair_z like a tensor: describe the LOCAL rows, refuse tensor conversion by name
    @property
    def shape(self):
        return tuple(self.zc.shape) if self.zc is not None else (0,)

    def numel(self) -> int:
        return int(self.zc.numel()) if self.zc is not None else 0

    def element_size(self) -> int:
        return int(self.zc.element_size()) if self.zc is not None else 4

    def clone(self):
        _refuse("pair_z.clone(): the conditioned pair is this rank's rows (a dense DiffusionConditioning.forward reached a sharded roll-out)")

    def to(self, *a, **k):
        _refuse("pair_z.to(): the conditioned pair is this rank's rows (a dense consumer reached a sharded roll-out)")


def prime(model, input_feature_dict: dict, z_struct, *, lay=None, n_sample: Optional[int] = None,
          enable_efficient_fusion: Optional[bool] = None) -> StructDiffusion:
    """ONCE per roll-out, at a point every rank reaches: the structural layout (``lay`` or :func:`struct_layout` of the source's
    ``N_st``), the ``DiffusionSchedule`` (row blocks from the agreed budget / pins; recorded), the conditioned pair rows ``z_cond
    [R_st, N_st, c_pair]`` (``pair_cond_rows`` over the structural rows of ``z_struct``: HostPair / tensor / RowShard, see
    :class:`_ZRows`), the pair-bias cache and the per-block callables. ``n_sample``: the samples per denoiser call (the attention
    logits transient ``S x H x q x N_st`` is budgeted with it)."""
    DF, D, _ = _core()
    import torch
    if os.environ.get("OPENDDE_FOLDCP_MODE", "single") == "distributed":
        _refuse("OPENDDE_FOLDCP_MODE=distributed: the Fold-CP transformer variants (forward_foldcp_local_z / _window) are not this line's")
    dm = model.diffusion_module
    dc = dm.diffusion_conditioning
    n_struct = None
    from . import tp as _tp
    if isinstance(z_struct, _tp.RowShard):
        n_struct = int(z_struct.lay.N)
    elif hasattr(z_struct, "t") and torch.is_tensor(getattr(z_struct, "t")):
        n_struct = int(z_struct.t.shape[-2])
    elif torch.is_tensor(z_struct):
        n_struct = int(z_struct.shape[-2])
    if lay is None:
        if n_struct is None:
            _refuse(f"structural pair source {type(z_struct).__name__}: cannot read N_st")
        lay = struct_layout(n_struct)
    D.require_sharded(lay, "opendde diffusion stage (tp_diffusion.prime)")
    ctx = StructDiffusion(model, lay)
    if enable_efficient_fusion is not None:
        ctx.fusion = bool(enable_efficient_fusion)
    zrows = _ZRows(z_struct, lay)
    ctx.zsrc = zrows
    STATS["zstruct_src"], STATS["zstruct_host_gb"], STATS["struct_layout"] = zrows.src, round(zrows.host_gb, 2), lay.facts()
    device = next(dm.parameters()).device
    cpd = int(dc.c_z_pair_diffusion)
    tr = dm.diffusion_transformer
    S = int(n_sample or 1)
    ctx.n_sample = S
    ctx.sched = DF.DiffusionSchedule.decide(lay, c_z=zrows.c, c_in=zrows.c + 2 * cpd, c_cond=cpd, H=int(tr.n_heads), S=S,
                                          n_blocks=int(tr.n_blocks), c_pair=int(dm.atom_attention_encoder.linear_no_bias_z.weight.shape[0]),
                                          elt=4)
    relp = input_feature_dict["relp"]
    out = torch.empty((lay.R, lay.N, cpd), dtype=torch.float32, device=device)     # the roll-out's resident: R_st x N_st x c_pair fp32
    ctx.zc = DF.pair_cond_rows(_cond_embed_fn(dc, relp, zrows, device), zrows.rows, lay, c_out=cpd, rows=ctx.sched.cond_rows or None,
                               transitions=_cond_transitions(dc), out=out)
    _bump("zcond_rows", lay.R)
    ctx.cache = DF.PairBiasCache(enabled=ctx.sched.bias_cache)
    extra = input_feature_dict.get(EXTRA_BIAS_KEY)
    STATS["dit_extra_bias"] = "none" if extra is None else ("rows" if input_feature_dict.get(EXTRA_BIAS_ROWS_KEY) or
                                                             (int(extra.shape[-2]) == lay.R and lay.R != lay.N) else "full")
    STATS["dit_bias"] = "fused_conv" if ctx.fusion else "ln_linear"
    ctx.blocks = [dit_block_fns(b, lay, extra=extra, fusion=ctx.fusion, normalize=getattr(dm, "normalize", None)) for b in tr.blocks]
    STATS["diff_rollouts"] += 1
    _tp._mark("diffusion_primed") if hasattr(_tp, "_mark") else None
    return ctx


def token_pair_term(enc, input_feature_dict: dict, ctx: StructDiffusion):
    """== ``AtomAttentionEncoder._add_token_pair_context_to_atom_pair``'s token-pair term ``linear_no_bias_z(layernorm_z(z[q_tok, k_tok]))``
    for every (window, query slot, key slot) -> ``[n_windows, n_queries, n_keys, c_atompair]``, read from the diagonal band of the
    projected conditioned rows (``band_plan`` on the engine's own window index map ``rearrange_qk_to_dense_trunk(atom_to_token_idx)``,
    valid slots = ``pad_info['mask_trunked']``; ``pair_band_rows``; ``band_lookup``). Slots the engine pads (masked with -inf downstream)
    hold band values of unspecified pairs."""
    DF, _D, _ = _core()
    from opendde.model.modules.primitives import rearrange_qk_to_dense_trunk
    lay = ctx.lay
    a2t = input_feature_dict["atom_to_token_idx"]
    idx_q, idx_k, _ = rearrange_qk_to_dense_trunk(a2t, a2t, dim_q=-1, dim_k=-1, n_queries=enc.n_queries, n_keys=enc.n_keys, compute_mask=False)
    valid = input_feature_dict["pad_info"]["mask_trunked"]
    if valid is None or tuple(valid.shape[-3:]) != (int(idx_q.shape[-2]), int(idx_q.shape[-1]), int(idx_k.shape[-1])):
        _refuse(f"pad_info['mask_trunked'] {None if valid is None else tuple(valid.shape)} vs windows {tuple(idx_q.shape)} x {int(idx_k.shape[-1])}")
    q_idx = idx_q.long().reshape(1, int(idx_q.shape[-2]), int(idx_q.shape[-1]))
    k_idx = idx_k.long().reshape(1, int(idx_k.shape[-2]), int(idx_k.shape[-1]))
    pv = valid.bool().reshape((1,) + tuple(valid.shape[-3:]))
    max_w = int(os.environ[ENV_BAND_W]) if os.environ.get(ENV_BAND_W, "").strip() else None
    max_extra = int(os.environ[ENV_BAND_EXTRA_MAX]) if os.environ.get(ENV_BAND_EXTRA_MAX, "").strip() else None
    plan = DF.band_plan(q_idx.to(ctx.zc.device), k_idx.to(ctx.zc.device), pv.to(ctx.zc.device), lay.N, max_w=max_w, max_extra_rows=max_extra)
    ctx.band_facts = plan.facts()
    proj = lambda z_rows: enc.linear_no_bias_z(enc.layernorm_z(z_rows))            # noqa: E731 — [rows, N_st, c_atompair], row-local
    band, extras = DF.pair_band_rows(proj, ctx.zc, lay, plan, rows=ctx.sched.band_rows or None)
    out = DF.band_lookup(band, extras, plan)[0]                                     # [nb, nq, nk, c_atompair] == proj(z_cond)[idx_q, idx_k]
    del band, extras
    _bump("diff_band_terms")
    return out


def prepare_cache_sharded(model, input_feature_dict: dict, ctx: StructDiffusion) -> dict:
    """== ``OpenDDE.prepare_diffusion_cache_for_sampling`` (opendde.py; the diffusion shared-variables cache, == ``--enable_cache true``,
    which the structural stage requires) with the pair image this rank's ROWS: ``pair_z`` = ``ctx`` (carried by the sampler to the
    installed denoiser), ``p_lm / c_l`` = the atom encoder's ``prepare_cache`` WITHOUT the dense token-pair gather, plus
    :func:`token_pair_term` added into ``p_lm`` exactly where the engine adds it (``p_lm.unsqueeze(-5)`` + the term per window)."""
    from opendde.utils.torch_utils import autocasting_disable_decorator
    dm = model.diffusion_module
    enc = dm.atom_attention_encoder
    f = input_feature_dict

    def build():
        p_lm, c_l = enc.prepare_cache(ref_pos=f["ref_pos"], ref_charge=f["ref_charge"], ref_mask=f["ref_mask"], ref_element=f["ref_element"],
                                      ref_atom_name_chars=f["ref_atom_name_chars"], atom_to_token_idx=f["atom_to_token_idx"], d_lm=f["d_lm"],
                                      v_lm=f["v_lm"], pad_info=f["pad_info"], r_l=None, z=None, inplace_safe=False)
        term = token_pair_term(enc, f, ctx)                                         # [nb, nq, nk, c_atompair]
        p_lm = p_lm.unsqueeze(dim=-5)                                               # the sample dim the engine inserts before the add
        if term.dim() == p_lm.dim() - 1:
            term = term.unsqueeze(dim=-5)
        p_lm += term.to(p_lm.dtype)
        del term
        return [p_lm, c_l]
    skip_amp = getattr(getattr(model, "configs", None), "skip_amp", None)
    plc = autocasting_disable_decorator(skip_amp.sample_diffusion)(build)() if skip_amp is not None else build()
    return {"pair_z": ctx, "pair_z_spec": None, "p_lm/c_l": plc}


def denoise_sharded(dm, ctx: StructDiffusion, *, x_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, p_lm, c_l,
                    inplace_safe: bool = False, chunk_size=None, enable_efficient_fusion: bool = False, use_conditioning: bool = True, **unused):
    """== ``DiffusionModule.forward`` -> ``f_forward`` (diffusion.py, the single-process cached-pair path: no Fold-CP mesh, no activation
    checkpoint) with the pair conditioning this rank's rows: EDM input scaling, the single conditioning (replicated), the atom attention
    encoder on the cached ``p_lm / c_l`` (replicated), ``a += linear(LN(s))``, the transformer = ``diffusion.diffusion_transformer_sharded``
    (queries = local rows, one all-gather per block), ``layernorm_a``, the atom attention decoder (replicated), EDM output scaling. The
    noisy positions are PROVEN replicated first (``diffusion.sync_replicated``)."""
    DF, _D, _ = _core()
    import torch
    if not use_conditioning:
        _refuse("use_conditioning=False under a sharded roll-out (the cached pair conditioning is the conditioned one)")
    if bool(enable_efficient_fusion) != bool(ctx.fusion):
        _refuse(f"enable_efficient_fusion={enable_efficient_fusion} at the denoiser call vs {ctx.fusion} when the roll-out was primed")
    if p_lm is None or c_l is None:
        _refuse("the atom encoder cache (p_lm / c_l) is required under a sharded roll-out (prepare_cache_sharded builds it)")
    lay = ctx.lay
    f = input_feature_dict
    dc = dm.diffusion_conditioning
    synced = DF.sync_replicated(x_noisy, "diffusion.x_noisy", mode=_tp_noise_sync())    # the per-call INPUT STATE (after noise + augmentation): rank 0's at det 0 (bcast), proven bitwise at det 1 (guard)
    if synced is not x_noisy:                                                       # bcast handed back rank 0's tensor: land it IN PLACE on the sampler's loop variable
        x_noisy.copy_(synced)                                                       # (generator.sample_diffusion's `x_noisy`, the tensor its update `x_l = x_noisy + eta*dt*delta` reads) — never a rebound local
        del synced
    STATS["noise_guards"] += 1
    r_noisy = x_noisy / torch.sqrt(dm.sigma_data ** 2 + t_hat_noise_level ** 2)[..., None, None]
    N_sample = r_noisy.size(-3)
    assert t_hat_noise_level.size(-1) == N_sample
    s_single = _single_cond(dc, t_hat_noise_level, s_inputs, s_trunk, inplace_safe)          # [..., S, N_st, c_s]
    s_trunk_e = s_trunk.unsqueeze(dim=-3)                                           # == expand_at_dim(s_trunk, -3, 1)
    z_none = torch.empty(0, device=ctx.zc.device)                                   # the encoder asserts z is not None; with the cache it never reads it
    a_token, q_skip, c_skip, p_skip = dm.atom_attention_encoder(
        f["atom_to_token_idx"], f["ref_pos"], f["ref_charge"], f["ref_mask"], f["ref_atom_name_chars"], f["ref_element"], f["d_lm"], f["v_lm"],
        f["pad_info"], r_l=r_noisy, s=s_trunk_e, z=z_none, p_lm=p_lm, c_l=c_l, inplace_safe=inplace_safe, chunk_size=chunk_size)
    a_token = a_token.to(dtype=torch.float32)
    if inplace_safe:
        a_token += dm.linear_no_bias_s(dm.layernorm_s(s_single))
    else:
        a_token = a_token + dm.linear_no_bias_s(dm.layernorm_s(s_single))
    a_token = DF.diffusion_transformer_sharded(ctx.blocks, a_token.to(dtype=torch.float32), s_single.to(dtype=torch.float32), ctx.zc, lay,
                                               schedule=ctx.sched, bias_cache=ctx.cache)
    _bump("dit_local_queries", lay.R * len(ctx.blocks))
    a_token = dm.layernorm_a(a_token)
    r_update = dm.atom_attention_decoder(atom_to_token_idx=f["atom_to_token_idx"], a=a_token, q_skip=q_skip, c_skip=c_skip, p_skip=p_skip,
                                         inplace_safe=inplace_safe, chunk_size=chunk_size)
    s_ratio = (t_hat_noise_level / dm.sigma_data)[..., None, None].to(r_update.dtype)
    x_denoised = (1 / (1 + s_ratio ** 2) * x_noisy + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio ** 2) * r_update).to(r_update.dtype)
    ctx.denoise_calls += 1
    _bump("diff_denoise_calls")
    return x_denoised


@contextlib.contextmanager
def sharded_denoiser(dm, ctx: StructDiffusion):
    """While active, ``dm(...)`` (``DiffusionModule.__call__``, what the engine's sampler calls as ``denoise_net``) with ``pair_z=ctx`` runs
    :func:`denoise_sharded`; any other ``pair_z`` reaches the class's own ``forward`` unchanged. The hook is instance-level and removed
    on exit (levers patch the CLASS; they stay where they are)."""
    def forward(x_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk=None, pair_z=None, p_lm=None, c_l=None,
                inplace_safe=False, chunk_size=None, use_conditioning=True, enable_efficient_fusion=False, pair_z_spec=None, **kw):
        if pair_z is not ctx:
            return type(dm).forward(dm, x_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l,
                                    inplace_safe=inplace_safe, chunk_size=chunk_size, use_conditioning=use_conditioning,
                                    enable_efficient_fusion=enable_efficient_fusion, pair_z_spec=pair_z_spec, **kw)
        if pair_z_spec is not None:
            _refuse("pair_z_spec (Fold-CP shard metadata) under a row-sharded roll-out")
        return denoise_sharded(dm, ctx, x_noisy=x_noisy, t_hat_noise_level=t_hat_noise_level, input_feature_dict=input_feature_dict,
                               s_inputs=s_inputs, s_trunk=s_trunk, p_lm=p_lm, c_l=c_l, inplace_safe=inplace_safe, chunk_size=chunk_size,
                               enable_efficient_fusion=enable_efficient_fusion, use_conditioning=use_conditioning)
    forward._rowpair_tp = True
    had = "forward" in dm.__dict__
    prev = dm.__dict__.get("forward")
    dm.forward = forward
    try:
        yield ctx
    finally:
        if had:
            dm.forward = prev
        else:
            del dm.forward


def run_sample_diffusion_stage(model, stock_stage, ctx: StructDiffusion, **kw):
    """``OpenDDE.run_sample_diffusion_stage`` (``stock_stage``: whatever is bound — the stock method or the offload unit's timer wrapper)
    with the denoiser row-sharded for the roll-out and the sampled coordinates PROVEN replicated (``diffusion.sync_replicated``).
    ``kw['cache']`` must be :func:`prepare_cache_sharded`'s; ``kw['z']`` is not read by the sharded path (the stock stage passes it as
    ``z_trunk`` only when there is no cache) and is replaced by None. Training-free guidance is refused by name (its engine steps call
    the denoiser outside this module's accounting)."""
    DF, _D, _ = _core()
    cache = kw.get("cache") or {}
    if cache.get("pair_z") is not ctx:
        _refuse("run_sample_diffusion_stage: cache['pair_z'] is not the primed sharded conditioning (prepare_cache_sharded)")
    cfg = getattr(model, "configs", None)
    guidance = _cfg(_cfg(cfg, "sample_diffusion"), "guidance")
    if guidance is not None and bool(_cfg(guidance, "enable", False)):
        _refuse("training-free guidance (configs.sample_diffusion.guidance.enable) under a sharded roll-out")
    kw = dict(kw)
    kw["z"] = None
    n = kw.get("N_sample")
    dcs = _cfg(_cfg(cfg, "infer_setting"), "sample_diffusion_chunk_size")
    per_call = min(int(n), int(dcs)) if (n and dcs) else (int(n) if n else ctx.n_sample)
    if per_call != ctx.n_sample:
        ctx.n_sample = per_call                                                     # census only: the schedule was budgeted at prime
    with sharded_denoiser(model.diffusion_module, ctx):
        coords = stock_stage(model, **kw)
    coords = _rank0_coordinates(coords, DF, _D)
    STATS["noise_guards"] += 1
    return coords


SPREAD_REFUSE_A = 1.0                                       # det 0: cross-rank coordinate spread above this at the sampler exit is refused by name (expected ~1e-3 A); det 1: any non-zero spread is refused


def _tp_noise_sync() -> str:
    from . import tp as _tp
    return _tp._noise_sync()


def _rank0_coordinates(coords, DF, D):
    """Sampler exit under n_gpu > 1: rank 0's coordinates replace every rank's (the confidence head's sharded rows then embed IDENTICAL
    coordinates), after the cross-rank spread ``max |x_rank - x_rank0|`` is all-reduced and recorded (``diff_rank_spread_A``); a spread above
    ``SPREAD_REFUSE_A`` is refused by name (a rank whose roll-out diverged)."""
    import torch
    mine = coords.detach().clone()
    x0 = DF.sync_replicated(coords, "diffusion.coordinate", mode="bcast")            # rank 0's tensor on every rank (then the core's guard)
    spread = (mine - x0).abs().max().reshape(1).to(torch.float32)
    D.allreduce_(spread, op="max")
    val = float(spread.item())
    STATS["rank_spread_A"] = max(float(STATS.get("rank_spread_A") or 0.0), val)
    try:
        from opt_core.mem.rowpair import evidence as _ev
        _ev.record_schedule(diff_rank_spread_A=round(STATS["rank_spread_A"], 6))
    except Exception:
        pass
    from . import tp as _tp
    if _tp._det_level_on() and val != 0.0:                                          # det 1: STRICT — every rank's roll-out is bitwise rank 0's, so any spread is a defect
        _refuse(f"diffusion.coordinate: cross-rank spread {val:.3e} A != 0 at the sampler exit under the deterministic recipe (det 1 is bitwise across ranks)")
    if val > SPREAD_REFUSE_A:                                                       # det 0: bounded — tier-2 drift only
        _refuse(f"diffusion.coordinate: cross-rank spread {val:.3f} A > {SPREAD_REFUSE_A} A at the sampler exit (a rank's roll-out diverged)")
    return x0


def release(ctx: Optional[StructDiffusion]) -> None:
    """Drop the roll-out's device residents (z_cond rows, the bias cache) once the coordinates exist."""
    if ctx is None:
        return
    if ctx.cache is not None:
        ctx.cache.clear()
    ctx.zc = None
    ctx.blocks = None


def fields() -> list:
    """Census pairs of this stage (the adapter appends them to its evidence line)."""
    lay = STATS.get("struct_layout") or {}
    out = [("diff_stage", "rows" if STATS["diff_rollouts"] else "unused"), ("zstruct_src", STATS["zstruct_src"]), ("zstruct_host_gb", STATS["zstruct_host_gb"]),
           ("struct_N", lay.get("N")), ("struct_rows", f"{lay.get('r0')}:{lay.get('r1')}" if lay else None), ("zcond_rows", STATS["zcond_rows"]),
           ("dit_local_queries", STATS["dit_local_queries"]), ("dit_bias", STATS["dit_bias"]), ("dit_extra_bias", STATS["dit_extra_bias"]),
           ("diff_denoise_calls", STATS["diff_denoise_calls"]), ("diff_band_terms", STATS["diff_band_terms"]), ("diff_noise_guards", STATS["noise_guards"]), ("diff_rank_spread_A", STATS["rank_spread_A"]),
           ("dit_attn_kernel", STATS["dit_attn_kernel"]), ("dit_attn_kernel_calls", STATS["dit_attn_kernel_calls"]), ("dit_attn_stock_calls", STATS["dit_attn_stock_calls"]),
           ("dit_attn_unsupported", ",".join(f"{k}:{v}" for k, v in sorted(STATS["dit_attn_unsupported"].items())) or "none"), ("dit_attn_maxerr", STATS["dit_attn_maxerr"]),
           ("dit_attn_cast", ",".join(f"{k}:{v}" for k, v in sorted(STATS["dit_attn_cast"].items())) or "none"), ("dit_bias_form", STATS["dit_bias_form"]), ("dit_bias_calls", STATS["dit_bias_calls"]),
           ("diff_replicated_by_design", REPLICATED_BY_DESIGN)]
    try:
        from opt_core.mem.rowpair import evidence as _ev
        sch = dict(_ev.schedule_fields())
        out += [(k, sch[k]) for k in sorted(sch) if k.startswith("diff_")]
    except Exception:                                                               # an older core without the schedule record: the pairs above stand
        pass
    return out