# Portions derived from Protenix v2.0.0 (https://github.com/bytedance/Protenix), Copyright 2024 ByteDance and/or its affiliates,
# used under the Apache License, Version 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ptx_tp/diffusion.py -- tensor-parallel (row-sharded pair) diffusion conditioning + sampling for
Protenix-v2 inference (protenix 2.0.0 wheel; model 'protenix-v2': c_z=256, c_s=384, c_token=768,
24 x 16-head DiffusionTransformer blocks, atom encoder/decoder 3 blocks each).

Census ops covered: C1 (DiffusionConditioning.prepare_cache, row-local), C2 (single conditioning,
replicated), F1 (noise schedule / randn / augmentation: replicated with rank-identical RNG streams),
F2 (AtomAttentionEncoder.prepare_cache: band-gather of pair_z rows + replicated), F3 (token aggregation,
replicated), F4 (DiffusionTransformer: row-split attention + all-gather; replicated below
PTX_TP_DIFF_REPLICATE_BELOW tokens), F5 (AtomAttentionDecoder, replicated).

Conventions (shared TP contract): pair_z shard = pair_z[r0:r1, :, :] contiguous on rank `layout.rank`;
token activations a [N_sample, N, 768] replicated between blocks; coordinates replicated.  Every
O(N^2) statement runs on the rank's own rows over stock's chunk grids in GLOBAL coordinates; no split-K
across ranks, no partial-sum all-reduce anywhere.  Stock statements are CALLED (not re-implemented)
wherever the statement is replicated, so levers installed on the stock modules (PTX_XL_TRANS_CHUNK, the fused
Transition) apply inside TP exactly as in the single-card lever stack.

What stock 2.0.0 caches: with enable_diffusion_shared_vars_cache (CLI default on)
stock computes pair_z = prepare_cache(relp, z_trunk) and (p_lm, c_l) = atom_encoder.prepare_cache(z=pair_z)
ONCE per forward (model/protenix.py 536-557); the 24 per-block pair biases [16, N, N] are NOT cached --
with enable_efficient_fusion (CLI default on) each block recomputes bias = conv2d(normalize(pair_z), w_b)
every step (transformer.py 173-178, diffusion.py 447-449), i.e. 24 x 200 bias evaluations per sample.
The single-card switch PTX_XL_FREE=diffcache refers to freeing the pair_z cache before the confidence head.  This module
adds an OPTIONAL rows-only bias cache [24, 16, R, N] (PTX_TP_DIFFCACHE_GB budget; arithmetic identical to
the per-step evaluation because the same conv statement on the same row chunks produces it).

Environment (read at call time):
  PTX_TP_DIFF_REPLICATE_BELOW=3841   token transformer + atom band run REPLICATED (stock statements on the
                                     all-gathered pair_z) when N < this (the diffusion sampler runs fp32
                                     at every N; fp32 SDPA/GEMM M-invariance is not guaranteed).
                                     0 => always row-split.
  PTX_TP_DIFFCACHE_GB=8              per-rank budget for persistent pair-derived diffusion buffers:
                                     bias cache [n_blk,16,R,N] if it fits, else normalized-z cache
                                     [256,R,N] fp32 (fusion mode) if it fits, else stream (recompute
                                     per block per step from the pair_z shard; no persistent buffer).
  PTX_TP_DIFF_ATTN=rowsplit          rowsplit: queries = rows I (bias rows I local) -> all-gather rows of the attention output.
                                     replicated: bias rows I all-gathered per block into the full [1,16,N,N] bias, attention and
                                     every other statement over ALL rows on every rank (stock shapes; redundant compute; transient
                                     16*N*N*elt bytes per block) -- fallback when row-split SDPA is not query-count invariant.
  PTX_TP_DIFF_ROWLIN=0               0: only SDPA (query rows I) and the pair bias (rows I) are row-split;
                                     the O(N) projections/gating/transition run on all rows (stock GEMM
                                     shapes; all-gather of the gated attention output rows before linear_o).
                                     1: gating/linear_o/linear_a_last/ConditionedTransitionBlock on rows I,
                                     then all-gather a rows (fewer FLOPs; row-count-dependent GEMM shapes).
  PTX_TP_PREP_CHUNK=128              prepare_cache row-chunk (GLOBAL grid) when PTX_XL_PREP_CHUNK is unset;
                                     PTX_XL_PREP_CHUNK / PTX_XL_PREP_CHUNK_BUDGET_MB (single-card semantics: auto
                                     or rows) take precedence when set.
  PTX_TP_BIAS_CHUNK=128              row chunk (GLOBAL grid) for normalize/conv/LN+linear of the pair bias
                                     and of the atom-pair z projection.
  PTX_TP_BAND_CHUNK=256              dense-trunk windows per all_gather round in the band-gather.
  PTX_TP_SDPA_QPAD=1                 row-split SDPA: zero-pad this rank's query/bias rows to layout.Rmax so all ranks call SDPA with
                                     the same s_q (the SDPA kernel configuration can depend on s_q when it is small, e.g. bf16 cuDNN at s_q <= 256).
  PTX_TP_F2_GATHER_BELOW=3841        fp32 regime only: below this N the atom-pair cache p_lm is computed by the STOCK statement on a
                                     once-gathered pair_z (freed right after) instead of the band-gather (the fp32 skinny GEMM
                                     c_z->16 is GEMM-M dependent, so the band-gather is not bitwise-equal to it in fp32; it is in bf16). 0 => always band.
  PTX_TP_RNG_SYNC=1                  broadcast torch(CPU+CUDA)/numpy/python RNG states from rank 0 at the
                                     start of tp_sample_diffusion (rank 0 keeps its own = stock stream).
  PTX_DET=1                          the deterministic recipe (ptx_tp.det_recipe()): per-step allreduce_checksum(x_noisy) + final coords checksum.
  PTX_TP_STOCK_LOOP=0                1: call generator.sample_diffusion itself (no hook points) instead of the mirrored loop.
  PTX_TP_SDPA_BACKEND=''             process-wide torch SDPA backend pin applied by apply_sdpa_backend_from_env():
                                     'efficient' | 'cudnn' | 'math' | 'flash,efficient' ... (comma list) | '' (torch default).
                                     A LEVER: it changes which kernel every F.scaled_dot_product_attention call uses (stock
                                     modules included), not only this module's calls.
"""
import contextlib
import math
import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as tdist
import torch.nn.functional as F

import ptx_tp
from . import dist as D
from .relp import get_relp_rows

try:  # thin row-block accessors (contract: LOCAL row indices, views on plain tensors)
    from .dist import zrows, zwrite
except ImportError:  # pragma: no cover
    def zrows(z, i0, i1):
        return z[i0:i1]

    def zwrite(z, i0, i1, x):
        if x.data_ptr() != z[i0:i1].data_ptr():
            z[i0:i1].copy_(x)

from protenix.model.generator import sample_diffusion as _stock_sample_diffusion
from protenix.model.modules.primitives import _attention, rearrange_qk_to_dense_trunk
from protenix.model.utils import expand_at_dim, flatten_final_dims, permute_final_dims
from protenix.utils.torch_utils import autocasting_disable_decorator

__all__ = [
    "tp_prepare_cache",
    "tp_atom_prepare_cache",
    "TPDiffusion",
    "tp_diffusion_transformer",
    "tp_sample_diffusion",
    "tp_sampling_loop",
    "sync_rng_from_rank0",
    "set_sdpa_backends",
    "apply_bias_chunk_lever",
    "remove_bias_chunk_lever",
    "apply_sdpa_backend_from_env",
    "report",
]


# ============================================================================================ env / bookkeeping
def _env_int(name, default):
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _env_float(name, default):
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _cfg():
    return dict(
        replicate_below=_env_int("PTX_TP_DIFF_REPLICATE_BELOW", 3841),
        diffcache_gb=_env_float("PTX_TP_DIFFCACHE_GB", 8.0),
        rowlin=_env_int("PTX_TP_DIFF_ROWLIN", 0),
        attn_mode=(os.environ.get("PTX_TP_DIFF_ATTN", "rowsplit").strip() or "rowsplit"),
        prep_chunk=_env_int("PTX_TP_PREP_CHUNK", 128),
        bias_chunk=_env_int("PTX_TP_BIAS_CHUNK", 128),
        band_chunk=_env_int("PTX_TP_BAND_CHUNK", 256),
        rng_sync=_env_int("PTX_TP_RNG_SYNC", 1),
        debug=int(ptx_tp.det_recipe()),
        stock_loop=_env_int("PTX_TP_STOCK_LOOP", 0),
    )


_REPORT: Dict[str, Any] = {"buffers": {}, "modes": {}, "notes": []}


def _world():
    if tdist.is_available() and tdist.is_initialized():
        return tdist.get_world_size(), tdist.get_rank()
    return 1, 0


def _owner_of_rows(layout, idx: torch.Tensor) -> torch.Tensor:
    """rank owning each GLOBAL row index, from the contract attribute layout.bounds (contiguous row ranges)."""
    starts = torch.tensor([q0 for (q0, q1) in layout.bounds], dtype=torch.long, device=idx.device)
    return (torch.searchsorted(starts, idx.long().contiguous(), right=True) - 1).clamp_(0, layout.P - 1)


def report() -> Dict[str, Any]:
    """Sizes of every persistent buffer this module created (bytes), chosen modes, per-phase timings."""
    return {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in _REPORT.items()}


def _reg_buffer(name: str, t: Optional[torch.Tensor]):
    if t is None:
        _REPORT["buffers"].pop(name, None)
        return
    _REPORT["buffers"][name] = dict(bytes=int(t.numel() * t.element_size()), shape=list(t.shape), dtype=str(t.dtype))


def _default_skip_amp(N: int) -> bool:
    # the sampler runs with autocast disabled (fp32) at every N: upstream's own setting for every size protenix-v2
    # accepts, kept above the size guard by the guard lift (runner_hooks.install_guard_lift); the seams are called
    # with the runner's configs.skip_amp.sample_diffusion and this default only answers a caller that passes none.
    return True


def _amp_off(skip_amp: bool):
    return torch.autocast(device_type="cuda", enabled=False) if skip_amp else contextlib.nullcontext()


def _cast32(t, skip_amp: bool):
    if skip_amp and torch.is_tensor(t) and torch.is_floating_point(t):
        return t.to(dtype=torch.float32)
    return t


def _prep_chunk_rows(N: int, cfg: Dict[str, Any]) -> int:
    """Row chunk for prepare_cache: PTX_XL_PREP_CHUNK semantics when set (rows | <n>k | <n>m | auto within
    PTX_XL_PREP_CHUNK_BUDGET_MB), else PTX_TP_PREP_CHUNK (default 128 = the layout block, P-invariant)."""
    p = os.environ.get("PTX_XL_PREP_CHUNK", "0").strip().lower()
    if p not in ("", "0", "off"):
        if p == "auto":
            budget = int(os.environ.get("PTX_XL_PREP_CHUNK_BUDGET_MB", "4096")) * 2**20
            per_row = N * (512 * 4 * 2 + 256 * 4 * 2 + 512 * 4 * 2 + 139 * 4)
            return max(1, int(budget // max(per_row, 1)))
        s, mult = p, 1
        if s.endswith("k"):
            mult, s = 1024, s[:-1]
        elif s.endswith("m"):
            mult, s = 1024 * 1024, s[:-1]
        return max(1, int(float(s) * mult))
    return max(1, int(cfg["prep_chunk"]))


def _gather_token_rows(x_I: torch.Tensor, layout: "D.Layout") -> torch.Tensor:
    """x_I [..., R, X] (token rows at dim -2) -> [..., N, X] replicated (bit-preserving all_gather)."""
    if layout.P == 1 or layout.replicated:
        return x_I
    xt = x_I.movedim(-2, 0).contiguous()  # [R, ..., X]
    full = D.all_gather_rows(xt, layout)  # [N, ..., X]
    return full.movedim(0, -2).contiguous()


# ============================================================================================ C1 prepare_cache
def tp_prepare_cache(
    cond_mod,
    input_feature_dict: Dict[str, Any],
    z_trunk_shard: torch.Tensor,
    layout: "D.Layout",
    *,
    skip_amp: Optional[bool] = None,
    inplace_safe: bool = False,
    chunk_rows: Optional[int] = None,
) -> torch.Tensor:
    """Row-local DiffusionConditioning.prepare_cache (diffusion.py 85-105).

    rows I of cat[z_trunk, relpe(relp rows I)] -> layernorm_z -> linear_no_bias_z -> +transition_z1 ->
    +transition_z2, iterated over the GLOBAL row-chunk grid (PTX_XL_PREP_CHUNK when set, else
    PTX_TP_PREP_CHUNK=128), in the dtype/autocast context stock uses: the stock call site wraps
    prepare_cache in autocasting_disable_decorator(configs.skip_amp.sample_diffusion) => when skip_amp
    (True at every N under the guard lift) autocast is disabled and z is up-cast to fp32; a caller passing
    skip_amp=False gets the ambient bf16 autocast.
    relp rows come from input_feature_dict['relp'].rows() (a LazyRelp) / a dense relp tensor /
    the per-token index vectors (ptx_tp.relp) -- never an [N, N, 139] tensor beyond one row chunk.
    Returns the pair_z shard [R, N, c_z] (fp32 when skip_amp, else the autocast dtype), which stays
    row-sharded for the whole sampling loop.  layout.replicated => stock statement on the full z."""
    cfg = _cfg()
    N = layout.N
    if skip_amp is None:
        skip_amp = _default_skip_amp(N)
        _REPORT["notes"].append(f"tp_prepare_cache: skip_amp not given, using upstream rule -> {skip_amp}")
    r_max, s_max = cond_mod.relpe.r_max, cond_mod.relpe.s_max
    if layout.replicated:
        relp = input_feature_dict.get("relp", None) if isinstance(input_feature_dict, dict) else None
        if relp is None or not (torch.is_tensor(relp) or hasattr(relp, "rows")):
            relp = get_relp_rows(input_feature_dict, 0, N, r_max, s_max)
        with _amp_off(skip_amp):
            out = cond_mod.prepare_cache(_cast32(relp, skip_amp), _cast32(z_trunk_shard, skip_amp), inplace_safe)
        _reg_buffer("pair_z(full,replicated)", out)
        return out
    rows = int(chunk_rows) if chunk_rows else _prep_chunk_rows(N, cfg)
    _REPORT["modes"]["prep_chunk_rows"] = rows
    r0, R = layout.r0, layout.R
    out = None
    with _amp_off(skip_amp):
        for (c0, c1) in layout.chunks(rows):
            zc = _cast32(zrows(z_trunk_shard, c0 - r0, c1 - r0), skip_amp)
            rel = get_relp_rows(input_feature_dict, c0, c1, r_max, s_max)
            if rel.device != zc.device:
                rel = rel.to(zc.device)
            pz = torch.cat([zc, cond_mod.relpe(rel)], dim=-1)  # [C, N, 2*c_z]
            del rel
            pz = cond_mod.linear_no_bias_z(cond_mod.layernorm_z(pz))
            pz = pz + cond_mod.transition_z1(pz)
            pz = pz + cond_mod.transition_z2(pz)
            if out is None:
                out = torch.empty((R, N, pz.shape[-1]), dtype=pz.dtype, device=pz.device)
            zwrite(out, c0 - r0, c1 - r0, pz)
            del pz
    if out is None:  # this rank owns no rows
        if skip_amp:
            dt = torch.float32
        elif z_trunk_shard.is_cuda and torch.is_autocast_enabled():
            dt = torch.get_autocast_gpu_dtype()
        else:
            dt = z_trunk_shard.dtype
        out = torch.empty((0, N, cond_mod.c_z), dtype=dt, device=z_trunk_shard.device)
    _reg_buffer("pair_z_shard", out)
    return out


# ============================================================================================ F2 band-gather
def _atom_feature_kwargs(input_feature_dict: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        ref_pos=input_feature_dict["ref_pos"],
        ref_charge=input_feature_dict["ref_charge"],
        ref_mask=input_feature_dict["ref_mask"],
        ref_element=input_feature_dict["ref_element"],
        ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
        atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
        d_lm=input_feature_dict["d_lm"],
        v_lm=input_feature_dict["v_lm"],
        pad_info=input_feature_dict["pad_info"],
    )


def _z_to_atompair_rows(enc, pair_z_shard: torch.Tensor, layout: "D.Layout", chunk: int) -> torch.Tensor:
    """rows I of linear_no_bias_z(layernorm_z(pair_z)) (transformer.py 823) -> [R, N, c_atompair], computed on
    the GLOBAL bias-chunk grid."""
    r0, R, N = layout.r0, layout.R, layout.N
    out = None
    for (c0, c1) in layout.chunks(chunk):
        zt = enc.linear_no_bias_z(enc.layernorm_z(zrows(pair_z_shard, c0 - r0, c1 - r0)))
        if out is None:
            out = torch.empty((R, N, zt.shape[-1]), dtype=zt.dtype, device=zt.device)
        out[c0 - r0: c1 - r0].copy_(zt)
        del zt
    if out is None:
        probe = enc.linear_no_bias_z(enc.layernorm_z(pair_z_shard.new_zeros((1, 1, enc.c_z))))
        out = torch.empty((0, N, enc.c_atompair), dtype=probe.dtype, device=pair_z_shard.device)
    return out


def _band_gather(enc, pair_z_shard: torch.Tensor, atom_to_token_idx: torch.Tensor, layout: "D.Layout",
                 bias_chunk: int, band_chunk: int) -> torch.Tensor:
    """The z-term of AtomAttentionEncoder.prepare_cache (transformer.py 819-830 + primitives.py 882-956):
    broadcast_token_to_local_atom_pair(linear_no_bias_z(layernorm_z(pair_z)))[0] -> [n_trunks, 32, 128, c_atompair],
    assembled from row shards: each rank evaluates the entries whose token ROW it owns, the per-window
    partial bands are all-gathered and every entry is COPIED from its owner (pure selection: exact)."""
    nq, nk = enc.n_queries, enc.n_keys
    idx_q, idx_k, _ = rearrange_qk_to_dense_trunk(
        atom_to_token_idx, atom_to_token_idx, dim_q=-1, dim_k=-1, n_queries=nq, n_keys=nk, compute_mask=False
    )
    idx_q = idx_q.long()  # [n_tr, nq]   (padded slots index token 0, exactly like stock)
    idx_k = idx_k.long()  # [n_tr, nk]
    assert idx_q.dim() == 2, "band-gather supports an un-batched atom_to_token_idx [N_atom] (inference)"
    n_tr = idx_q.shape[0]
    zt = _z_to_atompair_rows(enc, pair_z_shard, layout, bias_chunk)  # [R, N, c]
    c = zt.shape[-1]
    r0, r1, R, P = layout.r0, layout.r1, layout.R, layout.P
    owner = _owner_of_rows(layout, idx_q)  # [n_tr, nq]
    mine = (idx_q >= r0) & (idx_q < r1)
    loc = (idx_q - r0).clamp(min=0, max=max(R - 1, 0))
    out = torch.empty((n_tr, nq, nk, c), dtype=zt.dtype, device=zt.device)
    G = max(1, int(band_chunk))
    single = (P == 1) or layout.replicated or not (tdist.is_available() and tdist.is_initialized())
    for t0 in range(0, n_tr, G):
        t1 = min(n_tr, t0 + G)
        g = t1 - t0
        if R > 0:
            yl = zt[loc[t0:t1, :, None].expand(-1, -1, nk), idx_k[t0:t1, None, :].expand(-1, nq, -1), :]
            yl = yl.masked_fill(~mine[t0:t1, :, None, None], 0)
        else:
            yl = torch.zeros((g, nq, nk, c), dtype=zt.dtype, device=zt.device)
        if single:
            out[t0:t1] = yl
            continue
        yl = yl.contiguous()
        Y = torch.empty((P,) + tuple(yl.shape), dtype=yl.dtype, device=yl.device)
        tdist.all_gather_into_tensor(Y, yl)
        sel = owner[t0:t1][None, :, :, None, None].expand(1, g, nq, nk, c)
        out[t0:t1] = torch.gather(Y, 0, sel).squeeze(0)
        del Y, yl, sel
    return out


def tp_atom_prepare_cache(enc, input_feature_dict: Dict[str, Any], pair_z_shard: torch.Tensor, layout: "D.Layout",
                          *, skip_amp: Optional[bool] = None):
    """(p_lm, c_l) of AtomAttentionEncoder.prepare_cache(..., r_l=True, z=pair_z) (transformer.py 728-831) with
    pair_z row-sharded: c_l and the d_lm/v_lm terms of p_lm are the STOCK statements (replicated, atom-local);
    the token-pair term is band-gathered from the shards.  Called like stock's cache site, i.e. through
    autocasting_disable_decorator(skip_amp).  Both outputs replicated."""
    cfg = _cfg()
    if skip_amp is None:
        skip_amp = _default_skip_amp(layout.N)
    dec = autocasting_disable_decorator(skip_amp)
    kw = _atom_feature_kwargs(input_feature_dict)
    gather_below = _env_int("PTX_TP_F2_GATHER_BELOW", 3841)
    if layout.replicated:
        p_lm, c_l = dec(enc.prepare_cache)(**kw, r_l=True, z=pair_z_shard, inplace_safe=False)
    elif skip_amp and layout.N < gather_below:
        # fp32 regime below PTX_TP_F2_GATHER_BELOW: the fp32 skinny Linear c_z->c_atompair is GEMM-M dependent (not bitwise-equal when
        # evaluated on row blocks), so gather pair_z ONCE, run the stock statement, free it (same path as mode=replicated)
        full = D.all_gather_rows(pair_z_shard.contiguous(), layout)
        p_lm, c_l = dec(enc.prepare_cache)(**kw, r_l=True, z=full, inplace_safe=False)
        del full
        _REPORT["modes"]["atom_prepare_cache"] = "gather-once(stock statement)"
    else:
        p_base, c_l = dec(enc.prepare_cache)(**kw, r_l=None, z=None, inplace_safe=False)
        with _amp_off(skip_amp):
            y = _band_gather(enc, _cast32(pair_z_shard, skip_amp), input_feature_dict["atom_to_token_idx"], layout,
                             cfg["bias_chunk"], cfg["band_chunk"])
            p_lm = p_base.unsqueeze(dim=-5) + y  # [1, n_trunks, 32, 128, c_atompair] exactly like stock
        del y
    _reg_buffer("p_lm", p_lm)
    _reg_buffer("c_l", c_l)
    return p_lm, c_l


# ============================================================================================ F4 transformer state
class TPDiffusion:
    """State for one sampling call: row-split DiffusionTransformer over the pair_z shard + the TP denoiser
    (DiffusionModule.forward mirror).  Token activations a stay replicated between blocks."""

    def __init__(self, diffusion_module, pair_z_shard: torch.Tensor, layout: "D.Layout", *, skip_amp: bool,
                 enable_efficient_fusion: bool, rowlin: Optional[int] = None, diffcache_gb: Optional[float] = None,
                 bias_chunk: Optional[int] = None, debug: Optional[int] = None, attn_mode: Optional[str] = None):
        cfg = _cfg()
        self.attn_mode = str(cfg["attn_mode"] if attn_mode is None else attn_mode)
        assert self.attn_mode in ("rowsplit", "replicated"), self.attn_mode
        self.dm = diffusion_module
        self.layout = layout
        self.skip_amp = bool(skip_amp)
        self.fusion = bool(enable_efficient_fusion)
        self.rowlin = int(cfg["rowlin"] if rowlin is None else rowlin)
        self.bias_chunk = int(cfg["bias_chunk"] if bias_chunk is None else bias_chunk)
        self.debug = int(cfg["debug"] if debug is None else debug)
        # decorator semantics of the stock call site: pair_z is up-cast to fp32 when skip_amp
        self.pair_z = _cast32(pair_z_shard, self.skip_amp)
        dt = self.dm.diffusion_transformer
        self.blocks = dt.blocks
        self.n_blocks = len(self.blocks)
        self.H = dt.n_heads
        self._bias_cache = [None] * self.n_blocks
        self._zn = None
        self._step = 0
        R, N = layout.R, layout.N
        budget = float(cfg["diffcache_gb"] if diffcache_gb is None else diffcache_gb) * 2**30
        bias_elt = 4 if (self.skip_amp or not torch.cuda.is_available()) else 2  # conv/linear output dtype under autocast
        bias_bytes = self.n_blocks * self.H * layout.Rmax * N * bias_elt
        zn_bytes = self.dm.c_z * layout.Rmax * N * 4
        if bias_bytes <= budget:
            self.mode = "bias-cache"
        elif self.fusion and zn_bytes <= budget:
            self.mode = "zn-cache"
        else:
            self.mode = "stream"
        _REPORT["modes"]["transformer"] = dict(mode=self.mode, rowlin=self.rowlin, attn_mode=self.attn_mode, fusion=self.fusion, skip_amp=self.skip_amp,
                                               bias_cache_bytes=bias_bytes if self.mode == "bias-cache" else 0,
                                               zn_cache_bytes=zn_bytes if self.mode == "zn-cache" else 0,
                                               R=R, N=N, P=layout.P, bias_chunk=self.bias_chunk)
        _reg_buffer("pair_z_shard(sampling)", self.pair_z)

    # ---------------------------------------------------------------- pair-bias rows
    def _zn_rows(self, i0: int, i1: int) -> torch.Tensor:
        """normalized + permuted z rows (diffusion.py 447-449) -> [1, c_z, i1-i0, N] (LOCAL rows)."""
        if self._zn is not None:
            return self._zn[:, :, i0:i1, :]
        zc = zrows(self.pair_z, i0, i1).unsqueeze(0)  # [1, C, N, c_z]  (stock: expand_at_dim(pair_z, -4, 1))
        z = self.dm.normalize(zc.to(dtype=torch.float32))
        return permute_final_dims(z, [2, 0, 1]).contiguous()

    def _build_zn_cache(self):
        lay = self.layout
        pieces = [self._zn_rows(c0 - lay.r0, c1 - lay.r0) for (c0, c1) in lay.chunks(self.bias_chunk)]
        if pieces:
            zn = torch.cat(pieces, dim=-2) if len(pieces) > 1 else pieces[0]
        else:
            zn = torch.empty((1, self.dm.c_z, 0, lay.N), dtype=torch.float32, device=self.pair_z.device)
        self._zn = zn
        _reg_buffer("zn_cache[1,c_z,R,N]", zn)

    def bias_rows(self, b: int) -> torch.Tensor:
        """pair bias of block b for query rows I: [1, H, R, N] (AttentionPairBias.standard_multihead_attention)."""
        if self._bias_cache[b] is not None:
            return self._bias_cache[b]
        lay = self.layout
        apb = self.blocks[b].attention_pair_bias
        pieces = []
        if self.fusion:
            weight = (apb.linear_nobias_z.weight * apb.layernorm_z.weight[None, :])[:, :, None, None]
        for (c0, c1) in lay.chunks(self.bias_chunk):
            i0, i1 = c0 - lay.r0, c1 - lay.r0
            if self.fusion:
                bias = F.conv2d(self._zn_rows(i0, i1), weight)  # [1, H, C, N] contiguous (as stock)
            else:
                zc = zrows(self.pair_z, i0, i1).unsqueeze(0).to(dtype=torch.float32)  # stock: z_pair.to(fp32)
                bias = apb.linear_nobias_z(apb.layernorm_z(zc))  # [1, C, N, H]; permuted below as a VIEW (stock layout)
            pieces.append(bias)
        if pieces:
            if self.fusion:
                bias_I = torch.cat(pieces, dim=-2) if len(pieces) > 1 else pieces[0]
            else:
                bcat = torch.cat(pieces, dim=-3) if len(pieces) > 1 else pieces[0]  # [1, R, N, H] contiguous
                bias_I = permute_final_dims(bcat, [2, 0, 1])  # [1, H, R, N] non-contiguous view, same strides pattern as stock
        else:
            dtp = torch.float32
            if self.pair_z.is_cuda and torch.is_autocast_enabled():
                dtp = torch.get_autocast_gpu_dtype()
            bias_I = torch.empty((1, self.H, 0, lay.N), dtype=dtp, device=self.pair_z.device)
        if self.mode == "bias-cache":
            self._bias_cache[b] = bias_I
            if b == self.n_blocks - 1:
                tot = sum(int(x.numel() * x.element_size()) for x in self._bias_cache if x is not None)
                _REPORT["buffers"]["bias_cache[n_blk][1,H,R,N]"] = dict(bytes=tot, shape=[self.n_blocks] + list(bias_I.shape), dtype=str(bias_I.dtype))
        return bias_I

    # ---------------------------------------------------------------- one block (transformer.py 257-366 + 40-256)
    def _sdpa_rows(self, att, q_I, k, v, bias_I, inplace_safe, _allow_pad: bool = True):
        """_attention over query rows with (a) the query-padding guard PTX_TP_SDPA_QPAD=1 (default): the rank's query rows (and
        bias rows) are zero-padded to layout.Rmax so that EVERY rank issues the SDPA call with the same s_q = Rmax (>= 512 whenever
        N > 3,840 and P <= 8) -- SDPA kernel configuration was observed to depend on s_q for s_q <= 256 (bf16, cuDNN); and (b) the
        int32 guard (every launch < 2^31 mask elements)."""
        Rq, N = q_I.shape[-2], k.shape[-2]
        Rmax = int(self.layout.Rmax)
        if _allow_pad and _env_int("PTX_TP_SDPA_QPAD", 1) and 0 < Rq < Rmax and not self.layout.replicated:
            pad = Rmax - Rq
            q_P = torch.cat([q_I, q_I.new_zeros(q_I.shape[:-2] + (pad, q_I.shape[-1]))], dim=-2)
            if bias_I.is_contiguous():  # fused bias layout (stock: conv2d output, contiguous [.., H, R, N])
                b_P = torch.cat([bias_I, bias_I.new_zeros(bias_I.shape[:-2] + (pad, bias_I.shape[-1]))], dim=-2)
            else:  # non-fusion layout (stock: Linear output [.., R, N, H] contiguous seen through a permuted VIEW) -> pad in that layout
                bt = bias_I.transpose(-3, -2).transpose(-2, -1)  # [.., R, N, H]
                bt = torch.cat([bt.contiguous(), bt.new_zeros(bt.shape[:-3] + (pad,) + tuple(bt.shape[-2:]))], dim=-3)
                b_P = permute_final_dims(bt, [2, 0, 1])  # [.., H, Rmax, N] view, stock strides
            out = self._sdpa_rows(att, q_P, k, v, b_P, inplace_safe, _allow_pad=False)
            return out[..., :Rq, :]
        if Rq == 0:
            return q_I.new_empty(q_I.shape[:-1] + (v.shape[-1],))
        max_rows = max(1, (2**31 - 1) // max(1, self.H * N))
        if Rq <= max_rows:
            return _attention(q=q_I, k=k, v=v, attn_bias=bias_I,
                              use_efficient_implementation=att.use_efficient_implementation, inplace_safe=inplace_safe)
        if max_rows >= 128:
            max_rows = (max_rows // 128) * 128
        outs = []
        for a0 in range(0, Rq, max_rows):
            a1 = min(Rq, a0 + max_rows)
            outs.append(_attention(q=q_I[..., a0:a1, :], k=k, v=v, attn_bias=bias_I[..., a0:a1, :],
                                   use_efficient_implementation=att.use_efficient_implementation, inplace_safe=inplace_safe))
        return torch.cat(outs, dim=-2)

    def block_forward(self, b: int, a: torch.Tensor, s: torch.Tensor, inplace_safe: bool = False) -> torch.Tensor:
        lay = self.layout
        r0, r1 = lay.r0, lay.r1
        blk = self.blocks[b]
        apb = blk.attention_pair_bias
        att = apb.attention
        # AttentionPairBias.forward: input projection (AdaLN), replicated
        a_ln = apb.layernorm_a(a=a, s=s)
        bias_I = self.bias_rows(b)  # [1, H, R, N]
        # Attention.forward (global attention with bias): q/k/v projections on all rows (stock statement),
        # queries restricted to rows I, keys/values all tokens
        q, k, v = att._prep_qkv(q_x=a_ln, kv_x=a_ln, apply_scale=True)  # [Ns, H, N, d]
        if self.attn_mode == "replicated":
            # full bias [1, H, N, N] = all-gather of every rank's bias rows (data movement only), then the STOCK
            # statements over all rows (SDPA, gating + linear_o, output gating, transition) on every rank
            if lay.P == 1 or lay.replicated:
                bias_full = bias_I
            elif self.fusion:  # stock: conv2d output [1, H, N, N] contiguous
                bias_full = D.all_gather_rows(bias_I.permute(2, 0, 1, 3).contiguous(), lay).permute(1, 2, 0, 3).contiguous()
            else:  # stock: Linear output [1, N, N, H] contiguous, then permute_final_dims([2,0,1]) VIEW -> same strides here
                g = D.all_gather_rows(bias_I.permute(2, 0, 3, 1).contiguous(), lay)  # [N, 1, N, H] contiguous
                bias_full = permute_final_dims(g.permute(1, 0, 2, 3), [2, 0, 1])  # [1, H, N, N] view, strides (.., 1, N*H, H)
            o = _attention(q=q, k=k, v=v, attn_bias=bias_full, use_efficient_implementation=att.use_efficient_implementation,
                           inplace_safe=inplace_safe)  # [Ns, H, N, d] (stock statement, all rows)
            del q, k, v, bias_full
            o = att._wrap_up(o.transpose(-2, -3), a_ln)  # gating + linear_o (stock)
            return self._block_tail(blk, apb, o, a, s, inplace_safe)
        q_I = q[..., r0:r1, :]
        if len(bias_I.shape) != len(q_I.shape):
            bias_I = bias_I.unsqueeze(dim=-3)
        o_I = self._sdpa_rows(att, q_I, k, v, bias_I, inplace_safe)  # [Ns, H, R, d]
        del q, k, v, q_I
        o_I = o_I.transpose(-2, -3)  # [Ns, R, H, d]
        if self.rowlin:
            # _wrap_up + output projection + residual + transition on rows I, then all-gather a rows
            o = att._wrap_up(o_I, a_ln[..., r0:r1, :])  # [Ns, R, c_a]
            out_I = self._block_tail(blk, apb, o, a[..., r0:r1, :], s[..., r0:r1, :], inplace_safe)
            return _gather_token_rows(out_I, lay)
        # rowlin == 0: gate rows I with the full-row gate (same elements as stock), gather, then every O(N)
        # statement on all rows with stock shapes
        if att.linear_g is not None:
            g = att.sigmoid(att.linear_g(a_ln))
            g = g.view(g.shape[:-1] + (att.num_heads, -1))
            o_I = o_I * g[..., r0:r1, :, :]
        o_I = flatten_final_dims(o_I, num_dims=2)  # [Ns, R, H*d]
        o = _gather_token_rows(o_I, lay)  # [Ns, N, H*d]
        del o_I
        o = att.linear_o(o)
        return self._block_tail(blk, apb, o, a, s, inplace_safe)

    @staticmethod
    def _block_tail(blk, apb, o: torch.Tensor, a: torch.Tensor, s: torch.Tensor, inplace_safe: bool) -> torch.Tensor:
        """AttentionPairBias output gating + DiffusionTransformerBlock residual/transition, with stock's IN-PLACE
        statements when inplace_safe (transformer.py 246-251, 348-353): under bf16 autocast `a *= gate` and
        `attn_out += a` keep the residual stream in the attention output's dtype (bf16), unlike the out-of-place form."""
        if inplace_safe:
            o *= torch.sigmoid(apb.linear_a_last(s))
            attn_out = o  # drop_path = Identity at inference
            attn_out += a
        else:
            o = torch.sigmoid(apb.linear_a_last(s)) * o
            attn_out = o + a
        ff_out = blk.conditioned_transition_block(a=attn_out, s=s)
        return ff_out + attn_out

    def transformer(self, a: torch.Tensor, s: torch.Tensor, inplace_safe: bool = False, chunk_size: Optional[int] = None,
                    blocks: Optional[range] = None) -> torch.Tensor:
        """DiffusionTransformer.forward (24 blocks) row-split; a, s replicated fp32 [Ns, N, c]."""
        if self.mode == "zn-cache" and self._zn is None:
            self._build_zn_cache()
        for b in (range(self.n_blocks) if blocks is None else blocks):
            a = self.block_forward(b, a, s, inplace_safe=inplace_safe)
        return a

    # ---------------------------------------------------------------- denoiser (diffusion.py 331-600 mirror)
    def denoise(self, x_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk=None, pair_z=None,
                p_lm=None, c_l=None, inplace_safe: bool = False, chunk_size: Optional[int] = None,
                use_conditioning: bool = True, enable_efficient_fusion: bool = False):
        """Same signature as DiffusionModule.forward; pair_z/z_trunk arguments are ignored (the shard held by
        this state is used).  Everything except the token transformer is the stock statement, replicated."""
        dm = self.dm
        assert bool(enable_efficient_fusion) == self.fusion, "enable_efficient_fusion differs from the state's setting"
        assert p_lm is not None and c_l is not None, "TP denoiser requires the (band-gathered) p_lm/c_l cache"
        if self.debug:
            D.allreduce_checksum(x_noisy, f"x_noisy[step {self._step}]")
            D.allreduce_checksum(t_hat_noise_level, f"t_hat[step {self._step}]")
        # DiffusionModule.forward: scale positions
        r_noisy = x_noisy / torch.sqrt(dm.sigma_data**2 + t_hat_noise_level**2)[..., None, None]
        # f_forward: conditioning (pair path cached -> single path only; stock statements)
        s_single, _ = dm.diffusion_conditioning(
            t_hat_noise_level, None, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=None, pair_z=self.pair_z,
            inplace_safe=False, use_conditioning=use_conditioning,
        )
        s_trunk_e = expand_at_dim(s_trunk, dim=-3, n=1)
        a_token, q_skip, c_skip, p_skip = dm.atom_attention_encoder(
            input_feature_dict["atom_to_token_idx"], input_feature_dict["ref_pos"], input_feature_dict["ref_charge"],
            input_feature_dict["ref_mask"], input_feature_dict["ref_atom_name_chars"], input_feature_dict["ref_element"],
            input_feature_dict["d_lm"], input_feature_dict["v_lm"], input_feature_dict["pad_info"],
            r_l=r_noisy, s=s_trunk_e, z=self.pair_z, p_lm=p_lm, c_l=c_l, inplace_safe=inplace_safe, chunk_size=chunk_size,
        )
        a_token = a_token.to(dtype=torch.float32)
        if inplace_safe:
            a_token += dm.linear_no_bias_s(dm.layernorm_s(s_single))
        else:
            a_token = a_token + dm.linear_no_bias_s(dm.layernorm_s(s_single))
        a_token = self.transformer(a_token.to(dtype=torch.float32), s_single.to(dtype=torch.float32),
                                   inplace_safe=inplace_safe, chunk_size=chunk_size)
        a_token = dm.layernorm_a(a_token)
        r_update = dm.atom_attention_decoder(
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"], a=a_token, q_skip=q_skip, c_skip=c_skip, p_skip=p_skip,
            inplace_safe=inplace_safe, chunk_size=chunk_size,
        )
        # DiffusionModule.forward: recombine
        s_ratio = (t_hat_noise_level / dm.sigma_data)[..., None, None].to(r_update.dtype)
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio**2) * r_update
        ).to(r_update.dtype)
        self._step += 1
        return x_denoised

    __call__ = denoise


def tp_diffusion_transformer(diffusion_module, a: torch.Tensor, s: torch.Tensor, pair_z_shard: torch.Tensor, layout: "D.Layout", *,
                             skip_amp: bool, enable_efficient_fusion: bool, inplace_safe: bool = False, blocks: Optional[range] = None,
                             **state_kw) -> torch.Tensor:
    """Convenience: row-split DiffusionTransformer (all 24 blocks, or `blocks`) on replicated a/s -> replicated a."""
    st = TPDiffusion(diffusion_module, pair_z_shard, layout, skip_amp=skip_amp, enable_efficient_fusion=enable_efficient_fusion, **state_kw)
    return st.transformer(a, s, inplace_safe=inplace_safe, blocks=blocks)


# ============================================================================================ F1 RNG
def sync_rng_from_rank0() -> None:
    """Make every rank's RNG state identical to rank 0's (torch CPU + CUDA default generators, numpy global
    RandomState used by scipy Rotation.random in centre_random_augmentation, python `random`).  Rank 0 keeps
    its own state, i.e. the stock stream."""
    P, rank = _world()
    if P == 1:
        return
    import random as _random
    import numpy as _np
    state = None
    if rank == 0:
        state = dict(
            torch_cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            numpy=_np.random.get_state(),
            python=_random.getstate(),
        )
    state = D.broadcast_obj(state, src=0)
    if rank != 0:
        torch.set_rng_state(state["torch_cpu"].cpu())
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda"].cpu())
        _np.random.set_state(state["numpy"])
        _random.setstate(state["python"])


class _DebugDenoiser:
    def __init__(self, dm):
        self.dm = dm
        self.step = 0

    def __call__(self, **kw):
        D.allreduce_checksum(kw["x_noisy"], f"x_noisy[step {self.step}]")
        self.step += 1
        return self.dm(**kw)


# ============================================================================================ single-card bias-chunk lever
_BIAS_LEVER = {}


def apply_bias_chunk_lever(chunk: Optional[int] = None) -> bool:
    """Single-card lever (arithmetic per element unchanged; deterministic; bitwise-equal to stock unless the conv/GEMM kernels are
    shape dependent): AttentionPairBias.standard_multihead_attention computes the pair bias in row chunks on the GLOBAL grid
    range(0, N, chunk) -- exactly the statement TPDiffusion.bias_rows evaluates per rank -- instead of one conv2d / Linear over [N, N].
    chunk defaults to PTX_TP_BIAS_CHUNK (128).  remove_bias_chunk_lever() restores stock."""
    from protenix.model.modules import transformer as _T
    if _BIAS_LEVER:
        return False
    ch = int(chunk or _cfg()["bias_chunk"])
    _BIAS_LEVER["orig"] = _T.AttentionPairBias.standard_multihead_attention
    _BIAS_LEVER["chunk"] = ch

    def standard_multihead_attention(self, q, kv, z, inplace_safe=False, enable_efficient_fusion=False):
        N_rows = z.shape[-2] if enable_efficient_fusion else z.shape[-3]
        pieces = []
        if enable_efficient_fusion:  # z: [..., c_z, N, N] normalized fp32
            weight = (self.linear_nobias_z.weight * self.layernorm_z.weight[None, :])[:, :, None, None]
            for c0 in range(0, N_rows, ch):
                pieces.append(F.conv2d(z[..., c0:c0 + ch, :], weight))
            bias = torch.cat(pieces, dim=-2) if len(pieces) > 1 else pieces[0]
        else:  # z: [..., N, N, c_z]
            for c0 in range(0, N_rows, ch):
                pieces.append(self.linear_nobias_z(self.layernorm_z(z[..., c0:c0 + ch, :, :])))
            bias = torch.cat(pieces, dim=-3) if len(pieces) > 1 else pieces[0]
            bias = permute_final_dims(bias, [2, 0, 1])
        return self.attention(q_x=q, kv_x=kv, attn_bias=bias, inplace_safe=inplace_safe)

    _T.AttentionPairBias.standard_multihead_attention = standard_multihead_attention
    _REPORT["modes"]["bias_chunk_lever"] = ch
    return True


def remove_bias_chunk_lever() -> bool:
    from protenix.model.modules import transformer as _T
    if not _BIAS_LEVER:
        return False
    _T.AttentionPairBias.standard_multihead_attention = _BIAS_LEVER.pop("orig")
    _BIAS_LEVER.pop("chunk", None)
    _REPORT["modes"].pop("bias_chunk_lever", None)
    return True


# ============================================================================================ SDPA backend lever
_SDPA_FLAGS = dict(flash="enable_flash_sdp", efficient="enable_mem_efficient_sdp", cudnn="enable_cudnn_sdp", math="enable_math_sdp")


def set_sdpa_backends(names) -> Dict[str, bool]:
    """Enable exactly the listed SDPA backends process-wide (names from {flash, efficient, cudnn, math}); [] or None =>
    enable all (torch default).  Returns the resulting flag state."""
    names = [n.strip().lower() for n in (names or []) if n.strip()]
    for key, fn in _SDPA_FLAGS.items():
        enable = (not names) or (key in names)
        getattr(torch.backends.cuda, fn)(enable)
    return sdpa_backend_state()


def sdpa_backend_state() -> Dict[str, bool]:
    q = dict(flash="flash_sdp_enabled", efficient="mem_efficient_sdp_enabled", cudnn="cudnn_sdp_enabled", math="math_sdp_enabled")
    out = {}
    for k, fn in q.items():
        f = getattr(torch.backends.cuda, fn, None)
        out[k] = bool(f()) if callable(f) else None
    return out


def apply_sdpa_backend_from_env() -> Dict[str, bool]:
    spec = os.environ.get("PTX_TP_SDPA_BACKEND", "").strip()
    st = set_sdpa_backends(spec.split(",") if spec else [])
    _REPORT["modes"]["sdpa_backends"] = st
    return st


# ============================================================================================ sampling loop (mirror + hooks)
def tp_sampling_loop(
    denoise_net,
    input_feature_dict: Dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: Optional[torch.Tensor],
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
    guidance_configs: Optional[dict] = None,
    sampler_hook=None,
    debug: int = 0,
) -> torch.Tensor:
    """generator.sample_diffusion (generator.py 123-288) statement by statement (training-free guidance excluded), with
    three optional hook points: init_noise / post_step / post_update.  hook None => identical arithmetic to stock."""
    from protenix.model.utils import centre_random_augmentation

    if guidance_configs is not None:
        try:
            from protenix.tfg import parse_tfg_config
            if parse_tfg_config(guidance_configs).enable:
                raise NotImplementedError("training-free guidance is not supported by ptx_tp.tp_sampling_loop")
        except ImportError:
            pass
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype
    h_init = getattr(sampler_hook, "init_noise", None) if sampler_hook is not None else None
    h_step = getattr(sampler_hook, "post_step", None) if sampler_hook is not None else None
    h_upd = getattr(sampler_hook, "post_update", None) if sampler_hook is not None else None

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        x_l = noise_schedule[0] * torch.randn(size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype)
        if h_init is not None:
            x_l = h_init(x_l)
            if debug:
                D.allreduce_checksum(x_l, "hook.init_noise")
        for step_i, (c_tau_last, c_tau) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
            x_l = centre_random_augmentation(x_input_coords=x_l, N_sample=1).squeeze(dim=-3).to(dtype)
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)
            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(size=x_l.shape, device=device, dtype=dtype)
            t_hat = t_hat.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, chunk_n_sample).to(dtype)
            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
            )
            if h_step is not None:
                x_denoised = h_step(step_i, t_hat, x_noisy, x_denoised)
                if debug:
                    D.allreduce_checksum(x_denoised, f"hook.post_step[{step_i}]")
            delta = (x_noisy - x_denoised) / t_hat[..., None, None]
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta
            if h_upd is not None:
                x_l = h_upd(step_i, x_l)
                if debug:
                    D.allreduce_checksum(x_l, f"hook.post_update[{step_i}]")
        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
        for i in range(no_chunks):
            chunk_n_sample = diffusion_chunk_size if i < no_chunks - 1 else N_sample - i * diffusion_chunk_size
            chunk_x_l = _chunk_sample_diffusion(chunk_n_sample, inplace_safe=inplace_safe)
            x_l.append(chunk_x_l)
        x_l = torch.cat(x_l, -3)
    return x_l


# ============================================================================================ sampling
def tp_sample_diffusion(model, input_feature_dict: Dict[str, Any], s_inputs: torch.Tensor, s_trunk: torch.Tensor,
                        z_trunk_shard: Optional[torch.Tensor], pair_z_shard: torch.Tensor, layout: "D.Layout",
                        N_sample: int, N_step: int, **kw) -> torch.Tensor:
    """Protenix.sample_diffusion (protenix.py 314-343) + generator.sample_diffusion (generator.py 123-288) with the
    pair representation row-sharded.  The stock sampling loop itself is CALLED (noise schedule, randn init,
    centre_random_augmentation, predictor noise, Euler update, N_sample chunking are stock statements on
    replicated tensors with rank-identical RNG streams); the denoiser is either the stock DiffusionModule
    (replicated mode, N < PTX_TP_DIFF_REPLICATE_BELOW: pair_z all-gathered once) or the TP denoiser.
    kw: skip_amp (default configs.skip_amp.sample_diffusion), inplace_safe (default True, as stock inference),
        noise_schedule (default model.inference_noise_scheduler(N_step)), force_mode ('replicated'|'tp'),
        rowlin, diffcache_gb (TPDiffusion overrides), sampler_hook (an object with init_noise / post_step / post_update),
        use_stock_loop (call generator.sample_diffusion itself; no hook points).  z_trunk_shard is accepted for interface symmetry
        (stock passes z_trunk=None when the pair_z cache exists) and is not read.
    Returns coordinates [N_sample, N_atom, 3], replicated."""
    cfg = _cfg()
    configs = model.configs
    skip_amp = kw.pop("skip_amp", None)
    if skip_amp is None:
        skip_amp = bool(configs.skip_amp.sample_diffusion)
    inplace_safe = kw.pop("inplace_safe", True)
    noise_schedule = kw.pop("noise_schedule", None)
    force_mode = kw.pop("force_mode", None)
    rowlin = kw.pop("rowlin", None)
    diffcache_gb = kw.pop("diffcache_gb", None)
    sampler_hook = kw.pop("sampler_hook", None)
    attn_mode = kw.pop("attn_mode", None)
    use_stock_loop = kw.pop("use_stock_loop", None)
    if use_stock_loop is None:
        use_stock_loop = bool(cfg["stock_loop"])
    if sampler_hook is not None and use_stock_loop:
        raise ValueError("a sampler hook requires the mirrored loop (PTX_TP_STOCK_LOOP=0)")
    _REPORT["modes"]["sampler_hook"] = None if sampler_hook is None else type(sampler_hook).__name__
    if kw:
        raise TypeError(f"tp_sample_diffusion: unexpected kwargs {sorted(kw)}")
    if noise_schedule is None:
        noise_schedule = model.inference_noise_scheduler(N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype)
    fusion = bool(getattr(model, "enable_efficient_fusion", configs.enable_efficient_fusion))
    dm = model.diffusion_module
    enc = dm.atom_attention_encoder
    if force_mode is not None:
        mode = force_mode
    elif layout.replicated or layout.P == 1 or layout.N < cfg["replicate_below"]:
        mode = "replicated"
    else:
        mode = "tp"
    if layout.replicated:
        mode = "replicated"
    _REPORT["modes"]["sample_diffusion"] = dict(mode=mode, skip_amp=skip_amp, fusion=fusion, N=layout.N, P=layout.P,
                                                N_sample=N_sample, N_step=N_step)
    if cfg["rng_sync"]:
        sync_rng_from_rank0()
    dec = autocasting_disable_decorator(skip_amp)
    # Protenix.sample_diffusion plumbing
    _configs = {key: configs.sample_diffusion.get(key) for key in ["gamma0", "gamma_min", "noise_scale_lambda", "step_scale_eta"]}
    _configs.update(
        {
            "attn_chunk_size": (configs.infer_setting.chunk_size if not model.training else None),
            "diffusion_chunk_size": (configs.infer_setting.sample_diffusion_chunk_size if not model.training else None),
        }
    )
    guidance = configs.sample_diffusion.to_dict().get("guidance") if hasattr(configs.sample_diffusion, "to_dict") else None
    _configs.update({"guidance_configs": guidance})
    if mode == "tp":
        try:
            from protenix.tfg import parse_tfg_config
            if parse_tfg_config(guidance).enable:
                raise NotImplementedError("training-free guidance is not supported by the TP denoiser")
        except ImportError:
            pass

    if mode == "replicated":
        pair_z = pair_z_shard if (layout.replicated or layout.P == 1) else D.all_gather_rows(pair_z_shard.contiguous(), layout)
        _reg_buffer("pair_z(full,replicated transformer)", pair_z)
        p_lm, c_l = dec(enc.prepare_cache)(**_atom_feature_kwargs(input_feature_dict), r_l=True, z=pair_z, inplace_safe=False)
        _reg_buffer("p_lm", p_lm)
        _reg_buffer("c_l", c_l)
        denoise_net = _DebugDenoiser(dm) if cfg["debug"] else dm
        pz_arg = pair_z
    else:
        state = TPDiffusion(dm, pair_z_shard, layout, skip_amp=skip_amp, enable_efficient_fusion=fusion, rowlin=rowlin,
                            diffcache_gb=diffcache_gb, attn_mode=attn_mode)
        p_lm, c_l = tp_atom_prepare_cache(enc, input_feature_dict, pair_z_shard, layout, skip_amp=skip_amp)
        denoise_net = state.denoise
        pz_arg = pair_z_shard

    loop_fn = _stock_sample_diffusion if use_stock_loop else tp_sampling_loop
    loop_kw = {} if use_stock_loop else dict(sampler_hook=sampler_hook, debug=cfg["debug"])
    coords = dec(loop_fn)(
        **loop_kw,
        denoise_net=denoise_net,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=None,
        pair_z=pz_arg,
        p_lm=p_lm,
        c_l=c_l,
        noise_schedule=noise_schedule,
        N_sample=N_sample,
        inplace_safe=inplace_safe,
        enable_efficient_fusion=fusion,
        **_configs,
    )
    if cfg["debug"]:
        D.allreduce_checksum(coords, "coords")
    return coords
