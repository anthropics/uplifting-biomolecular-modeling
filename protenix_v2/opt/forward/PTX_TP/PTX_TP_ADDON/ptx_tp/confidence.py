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
"""Tensor-parallel ConfidenceHead + DistogramHead/contact probabilities for Protenix v2.

Row-sharded pair representation: rank r owns z[r0:r1, :, :] (layout from ptx_tp.dist.Layout).
Everything below reproduces the *statements* of

  protenix/model/modules/confidence.py  ConfidenceHead.forward / memory_efficient_forward (26-348)
  protenix/model/modules/head.py        DistogramHead.forward (43-56)
  protenix/model/protenix.py            contact_probs = compute_contact_prob(distogram_head(z)) (579-585)

on row shards.  Rules used to make the result bitwise identical to the single-device code:
  * statements whose inputs are replicated (s path, linear_no_bias_s1/s2, cdist of the
    representative atoms, pLDDT / resolved einsums) are executed on the FULL replicated tensor
    (same kernel, same shape as stock) and row-sliced afterwards;
  * row-local statements on z (distance one-hot embedding, pae_ln/pde_ln + linears, distogram
    linear, softmax) are executed per global-grid row chunk (layout.chunks) - the only
    P-dependence is the GEMM row count M;
  * z^T rows come from ptx_tp.dist.transpose_shards (bit-preserving), once per sample for the
    PDE head (on the 4-block pairformer output shard) and once for the distogram logits shard;
  * the 4-block pairformer goes through ptx_tp.pairformer.tp_pairformer_stack.

The [N_sample, N, N, 64] PAE / PDE logits never exist at once: they are produced as
[row_chunk, N, 64] blocks, each pushed through ptx_tp.blockreduce.RowBlockReducer (expected
PAE / PDE rows fp16, pTM-family partial terms, gpde inputs); ptx_tp.summary then finishes on
rank 0 with stock's formulas.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import torch

from protenix.model.sample_confidence import compute_contact_prob
from protenix.model.utils import broadcast_token_to_atom, one_hot

from ptx_tp import dist as tpd
from ptx_tp.blockreduce import ChainIndex, RowBlockReducer, max_row_chunk

_DEFAULT_BINS = {"pae": (0.0, 32.0, 64), "pde": (0.0, 32.0, 64)}   # configs_base.loss.pae / .pde


def _pairformer_api(pairformer_fn: Optional[Callable]) -> Callable:
    if pairformer_fn is not None:
        return pairformer_fn
    from ptx_tp.pairformer import tp_pairformer_stack  # shared API name
    return tp_pairformer_stack


def _amp_off(t: torch.Tensor):
    return torch.autocast(device_type="cuda" if t.is_cuda else "cpu", enabled=False)


def token_is_ligand_from_atoms(token_asym_id: torch.Tensor, atom_to_token_idx: torch.Tensor,
                               atom_is_polymer: torch.Tensor) -> torch.Tensor:
    """stock sample_confidence.py 96-100."""
    atom_is_ligand = (1 - atom_is_polymer).long()
    token_is_ligand = torch.zeros_like(token_asym_id).scatter_add(0, atom_to_token_idx, atom_is_ligand)
    return token_is_ligand > 0


# ============================================================================ distogram / contacts
@torch.no_grad()
def tp_distogram_contact_rows(distogram_head, z_shard, layout, *,
                              min_bin: float = 2.3125, max_bin: float = 21.6875, no_bins: int = 64,
                              thres: float = 8.0, row_chunk: Optional[int] = None) -> torch.Tensor:
    """Rows r0:r1 (fp32 [R, N]) of stock
         contact_probs = autocasting_disable_decorator(True)(compute_contact_prob)(distogram_head(z), ...)
    DistogramHead.forward symmetrises the LOGITS before the softmax (head.py 55:
    logits = linear(z); logits = logits + logits.transpose(-2, -3)), so z^T information is needed:
    z column blocks are streamed with ptx_tp.dist.transpose_blocks and the linear is applied to
    both the row block and the transposed block (logits^T[i, j] = linear(z[j, i])); no logits shard
    and no transposed copy of z is ever held.  Replicated layout -> stock statement (TP no-op)."""
    if layout.replicated:
        logits = distogram_head(z_shard)
        with _amp_off(logits):
            return compute_contact_prob(logits.to(torch.float32), min_bin=min_bin, max_bin=max_bin,
                                        no_bins=no_bins, thres=thres)
    N, R = layout.N, layout.R
    step = _row_step(N, max(no_bins, tpd.zrows(z_shard, 0, min(1, R)).shape[-1]), row_chunk)
    lin = distogram_head.linear
    out = torch.empty((R, N), dtype=torch.float32, device=tpd.zrows(z_shard, 0, min(1, R)).device)
    for (i0, i1, zT_blk) in tpd.transpose_blocks(z_shard, layout, step=step):
        # NB: bf16 GEMMs are bitwise M-invariant only for CONTIGUOUS row-major inputs; a strided
        # (transposed-view) operand selects a different cuBLAS kernel -> .contiguous().
        l = lin(tpd.zrows(z_shard, i0, i1)) + lin(zT_blk.contiguous())   # ambient autocast, as stock
        with _amp_off(l):                                       # autocasting_disable_decorator(True): fp32 cast + amp off
            out[i0:i1] = compute_contact_prob(l.to(torch.float32), min_bin=min_bin, max_bin=max_bin,
                                              no_bins=no_bins, thres=thres)
        del l, zT_blk
    return out


def _row_step(N: int, width: int, row_chunk: Optional[int]) -> int:
    """Row block for streamed loops: divides 128 (P-invariant grid) and honours the int32 rule."""
    step = max_row_chunk(N, width)
    if row_chunk:
        step = min(step, int(row_chunk))
    for d in (128, 64, 32, 16, 8, 4, 2, 1):
        if d <= step:
            return d
    return 1


# ============================================================================ confidence head pieces
@torch.no_grad()
def tp_confidence_zinit_rows(head, s_inputs: torch.Tensor, z_trunk_shard: torch.Tensor, layout,
                             *, use_embedding: bool = True, inplace_safe: bool = True,
                             consume_z_trunk: bool = False) -> torch.Tensor:
    """confidence.py 185-201 on rows:  z = (linear_no_bias_s1(s_inputs)[None,:,:] + linear_no_bias_s2(s_inputs)[:,None,:]) + z_trunk."""
    r0, R = layout.r0, layout.R
    s1 = head.linear_no_bias_s1(s_inputs)                   # full replicated GEMMs (same statement as stock)
    s2 = head.linear_no_bias_s2(s_inputs)
    # z_rows = z_init_rows + z_trunk_rows, block-streamed through the storage-agnostic accessors.
    # (a + b == b + a bitwise, so zadd(copy_of_z_trunk, z_init_block) == stock's z_init + z_trunk.)
    z_rows = z_trunk_shard if consume_z_trunk else tpd.zclone(z_trunk_shard)
    for (i0, i1) in tpd.zblocks(R):
        if not use_embedding:
            tpd.zwrite(z_rows, i0, i1, tpd.zrows(z_rows, i0, i1) * 0)
        tpd.zadd(z_rows, i0, i1, s1[..., None, :, :] + s2[..., r0 + i0: r0 + i1, None, :])
    del s1, s2
    return z_rows


@torch.no_grad()
def tp_distance_embed_rows_(head, z_pair_rows: torch.Tensor, x_pred_rep_coords: torch.Tensor, layout,
                            *, row_chunk: Optional[int] = None) -> torch.Tensor:
    """confidence.py 277-304 on rows (in place on the private per-sample copy z_pair_rows):
         d = cdist(x_rep, x_rep)  (fp32, amp off; full [N,N] then row slice -> same kernel as stock)
         z += linear_no_bias_d(one_hot(d));  z += linear_no_bias_d_wo_onehot(d[..., None])"""
    N, r0, R = layout.N, layout.r0, layout.R
    with _amp_off(x_pred_rep_coords):
        xr = x_pred_rep_coords.to(torch.float32)
        d_rows = torch.cdist(xr, xr)[r0: r0 + R]          # [R, N]
    chunk = row_chunk or max_row_chunk(N, head.num_bins)
    for (c0, c1) in layout.chunks(chunk):
        lo, hi = c0 - r0, c1 - r0
        tpd.zadd(z_pair_rows, lo, hi, head.linear_no_bias_d(
            one_hot(x=d_rows[lo:hi], lower_bins=head.lower_bins, upper_bins=head.upper_bins)))
        tpd.zadd(z_pair_rows, lo, hi, head.linear_no_bias_d_wo_onehot(d_rows[lo:hi].unsqueeze(dim=-1)))
    return z_pair_rows


@torch.no_grad()
def tp_plddt_resolved(head, s_single: torch.Tensor, input_feature_dict: dict):
    """confidence.py 318-345 (replicated s path, verbatim)."""
    s_single = s_single.to(torch.float32)
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    atom_to_tokatom_idx = input_feature_dict["atom_to_tokatom_idx"]
    with _amp_off(s_single):
        a = broadcast_token_to_atom(x_token=s_single, atom_to_token_idx=atom_to_token_idx)
        plddt_pred = torch.einsum("...nc,ncb->...nb", head.plddt_ln(a), head.plddt_weight[atom_to_tokatom_idx])
        resolved_pred = torch.einsum("...nc,ncb->...nb", head.resolved_ln(a), head.resolved_weight[atom_to_tokatom_idx])
    return plddt_pred, resolved_pred


@torch.no_grad()
def tp_pae_pde_logits_chunks(head, z_pair_rows, layout, *, row_chunk: Optional[int] = None):
    """Iterator over (c0, c1, pae_logits [c,N,b], pde_logits [c,N,b]) fp32 for GLOBAL rows [c0, c1) of
    this rank (ascending) - confidence.py 318, 327-331:
         z = z.float();  pae = linear_no_bias_pae(pae_ln(z));  pde = linear_no_bias_pde(pde_ln(z + z^T))
    z^T rows arrive block-streamed from ptx_tp.dist.transpose_blocks (one all_to_all per block; no
    transposed copy of the shard); the fp32 upcast is done per block."""
    N, r0, R = layout.N, layout.r0, layout.R
    step = _row_step(N, head.c_z, row_chunk)
    for (i0, i1, zT_blk) in tpd.transpose_blocks(z_pair_rows, layout, step=step):
        zc = tpd.zrows(z_pair_rows, i0, i1).to(torch.float32)          # "Upcast after pairformer", per block
        with _amp_off(zc):
            pae = head.linear_no_bias_pae(head.pae_ln(zc))
            zsum = (zc + zT_blk.to(torch.float32)).contiguous()          # z_pair + z_pair.transpose(-2, -3)
            pde = head.linear_no_bias_pde(head.pde_ln(zsum))
        del zc, zsum, zT_blk
        yield r0 + i0, r0 + i1, pae, pde


# ============================================================================ the whole head
@torch.no_grad()
def tp_confidence_head(head, input_feature_dict: dict, s_inputs: torch.Tensor, s_trunk: torch.Tensor,
                       z_trunk_shard: torch.Tensor, pair_mask: Optional[torch.Tensor], x_pred_coords: torch.Tensor,
                       layout, *, triangle_multiplicative: str = "torch", triangle_attention: str = "torch",
                       inplace_safe: bool = True, chunk_size: Optional[int] = None,
                       use_embedding: bool = True,
                       contact_rows: Optional[torch.Tensor] = None, finish: str = "exact",
                       bin_cfg: Optional[dict] = None, row_chunk: Optional[int] = None,
                       pairformer_fn: Optional[Callable] = None, keep_value_rows: bool = True,
                       consume_z_trunk: bool = False, timers: Optional[dict] = None,
                       mem_probe: bool = False) -> dict:
    """TP ConfidenceHead.forward (confidence.py 133-256 + 258-348).

    Inputs follow the shared TP contract: s_inputs / s_trunk / x_pred_coords replicated,
    z_trunk_shard = z_trunk[r0:r1] (dtype as stock, bf16 under the runner's autocast),
    pair_mask as stock (None at inference).  Call it inside the same autocast context stock uses
    (skip_amp.confidence_head is False for protenix-v2 -> bf16 autocast).

    Returns a dict:
      always            plddt [N_s, N_atom, b_plddt], resolved [N_s, N_atom, 2] (replicated, fp32), mode
      mode "block"      reducers: list of ptx_tp.blockreduce.RowBlockReducer (one per sample, this rank's rows),
                        chains: ChainIndex   -> finish with ptx_tp.summary.tp_full_data_and_summary
      mode "replicated" (layout.replicated) stock forward outputs on every rank (pae/pde full everywhere)
    contact_rows ([R, N] fp32 from tp_distogram_contact_rows) feeds the gpde family.
    """
    bins = dict(_DEFAULT_BINS)
    if bin_cfg:
        bins.update(bin_cfg)
    t_ = timers if timers is not None else {}

    if layout.replicated:
        plddt, pae, pde, resolved = head(
            input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk_shard,
            pair_mask=pair_mask, x_pred_coords=x_pred_coords, use_embedding=use_embedding,
            triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
            inplace_safe=inplace_safe, chunk_size=chunk_size)
        return {"mode": "replicated", "plddt": plddt, "pae": pae, "pde": pde, "resolved": resolved}

    pf = _pairformer_api(pairformer_fn)
    N, r0, r1, R = layout.N, layout.r0, layout.r1, layout.R
    _z0 = tpd.zrows(z_trunk_shard, 0, min(1, R))
    assert z_trunk_shard.shape[0] == R and _z0.shape[-2] == N, (tuple(z_trunk_shard.shape), R, N)
    N_token = s_inputs.shape[-2]
    assert N_token == N

    # ---- forward() prologue (178-204)
    s_trunk = head.input_strunk_ln(torch.clamp(s_trunk, min=-512, max=512))
    x_rep_atom_mask = input_feature_dict["distogram_rep_atom_mask"].bool()
    x_pred_rep_coords = x_pred_coords[..., x_rep_atom_mask, :]
    N_sample = x_pred_rep_coords.size(-3)
    tic = time.time()
    z_rows = tp_confidence_zinit_rows(head, s_inputs, z_trunk_shard, layout, use_embedding=use_embedding,
                                      inplace_safe=inplace_safe, consume_z_trunk=consume_z_trunk)
    t_["zinit"] = time.time() - tic

    assert contact_rows is not None and contact_rows.shape == (R, N), "the row-block reduction needs contact_rows [R, N]"
    feat = input_feature_dict
    atom_is_polymer = 1 - feat["is_ligand"]
    chains = ChainIndex(feat["asym_id"], feat["has_frame"],
                        token_is_ligand_from_atoms(feat["asym_id"], feat["atom_to_token_idx"], atom_is_polymer))

    plddt_preds, resolved_preds, reducers = [], [], []
    for i in range(N_sample):
        # ---- memory_efficient_forward (258-348) on rows
        z_pair = tpd.zclone(z_rows) if N_sample > 1 else z_rows      # z_rows is private (or consumed): in place
        tic = time.time()
        tp_distance_embed_rows_(head, z_pair, x_pred_rep_coords[..., i, :, :], layout, row_chunk=row_chunk)
        t_[f"dist_embed_{i}"] = time.time() - tic
        tic = time.time()
        s_single, z_pair = pf(head.pairformer_stack, s_trunk.clone() if inplace_safe else s_trunk, z_pair, layout,
                              pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative,
                              triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        t_[f"pairformer_{i}"] = time.time() - tic
        tic = time.time()
        plddt_pred, resolved_pred = tp_plddt_resolved(head, s_single, input_feature_dict)
        t_[f"plddt_{i}"] = time.time() - tic
        tic = time.time()
        if mem_probe and s_trunk.is_cuda:                     # tests only: peak of the heads/reduce phase
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            _mem_base = torch.cuda.memory_allocated()
        red = RowBlockReducer(chains, r0, r1, s_trunk.device, pae_bins=bins["pae"], pde_bins=bins["pde"],
                              finish=finish, keep_value_rows=keep_value_rows)
        for c0, c1, pae_c, pde_c in tp_pae_pde_logits_chunks(head, z_pair, layout, row_chunk=row_chunk):
            red.consume(c0, c1, pae_c, pde_c, contact_rows[c0 - r0: c1 - r0])
            del pae_c, pde_c
        reducers.append(red)
        t_[f"heads_{i}"] = time.time() - tic
        if mem_probe and s_trunk.is_cuda:
            torch.cuda.synchronize()
            t_[f"heads_peak_delta_gib_{i}"] = (torch.cuda.max_memory_allocated() - _mem_base) / 2 ** 30
            t_[f"heads_peak_abs_gib_{i}"] = torch.cuda.max_memory_allocated() / 2 ** 30
        del z_pair
        plddt_preds.append(plddt_pred)
        resolved_preds.append(resolved_pred)

    return {"mode": "block",
            "plddt": torch.stack(plddt_preds, dim=-3),          # [..., N_s, N_atom, b_plddt]
            "resolved": torch.stack(resolved_preds, dim=-3),
            "reducers": reducers,
            "chains": chains}
