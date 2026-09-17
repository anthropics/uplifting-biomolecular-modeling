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
"""Tensor-parallel TemplateEmbedder (Protenix-2.x pairformer.py 919-1098).

Activity check (protenix 2.0.0, CLI `--use_msa true`, no template flag ->
configs_inference.use_template = False): InferenceTemplateFeaturizer.make_template_feature
is called unconditionally and, without hits, assembles TemplateFeatures.empty_template_features
padded to max_templates = 4, so input_feature_dict DOES contain "template_aatype"
[4, N] (+ template_distogram / masks / unit vectors, all zero).  For protenix-v2
(template_embedder.n_blocks = 2) the TemplateEmbedder therefore RUNS in every recycling
cycle and its output depends on z through linear_no_bias_z(LN(z)) -> it is NOT a no-op.

TP pattern: v_t = linear_z(LN(z_I)) + linear_a(at_t[I]) on the rank's rows; the c=64 pair
stack of each template runs on [R, N, 64] shards through the pair-stack primitives
(ptx_tp.msa.tp_pair_stack_block); LN_v, the template average, ReLU and linear_u are
row-local.  Returns u_shard [R, N, c_z] (or the python int 0 exactly like stock).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from protenix.data.constants import STD_RESIDUES_WITH_GAP
from protenix.model.utils import expand_at_dim

from ptx_tp.dist import Layout
from ptx_tp._h2util import zblocks, zrows
from ptx_tp.msa import tp_pair_stack_block, zadd_rows

__all__ = ["tp_template_embedder", "template_embedder_is_active", "compact_template_features",
           "template_pair_rows", "DENSE_TEMPLATE_KEYS"]


def template_embedder_is_active(template_embedder, input_feature_dict: Dict[str, Any]) -> bool:
    """True iff the stock TemplateEmbedder.forward would do work (not `return 0`)."""
    return ("template_aatype" in input_feature_dict) and template_embedder.n_blocks >= 1



DENSE_TEMPLATE_KEYS = ("template_distogram", "template_pseudo_beta_mask", "template_unit_vector",
                       "template_backbone_frame_mask")  # [T,N,N,39], [T,N,N], [T,N,N,3], [T,N,N] fp32 = 44 fp32/pair
DENSE_FREE_MARKER = "template_pair_dense_free"
PB1D, BB1D = "template_pseudo_beta_mask_1d", "template_backbone_frame_mask_1d"


def compact_template_features(input_feature_dict: Dict[str, Any], *, recheck: bool = True) -> bool:
    """Launcher helper (rank 0, before broadcasting features): drop the dense [T,N,N,44] template
    pair tensors when they can be rebuilt losslessly per row block from per-token vectors.

    Lossless condition (true for the featurizer's DUMMY/padding templates, i.e. the
    `use_template=False` path): template_distogram and template_unit_vector are identically 0 and
    the two 2-D masks equal the outer products of their diagonals (masks are {0,1} products, so
    diag(pb2d) == pb1d).  Then the dict gains
        template_pseudo_beta_mask_1d [T,N], template_backbone_frame_mask_1d [T,N],
        template_pair_dense_free = tensor(1)
    and loses the four dense keys (template_aatype / atom_positions / atom_mask are kept).
    Returns True if compacted; False (dict untouched) when real template coordinates are present
    (then tp_template_embedder slices rows from the dense tensors instead).
    """
    d = input_feature_dict
    if "template_aatype" not in d or any(k not in d for k in DENSE_TEMPLATE_KEYS):
        return False
    pb2d = d["template_pseudo_beta_mask"]
    bb2d = d["template_backbone_frame_mask"]
    pb1d = torch.diagonal(pb2d, dim1=-2, dim2=-1).contiguous()  # [T, N]
    bb1d = torch.diagonal(bb2d, dim1=-2, dim2=-1).contiguous()
    if recheck:
        ok = (not bool(d["template_distogram"].any())) and (not bool(d["template_unit_vector"].any()))
        ok = ok and bool(torch.equal(pb2d, pb1d[..., :, None] * pb1d[..., None, :]))
        ok = ok and bool(torch.equal(bb2d, bb1d[..., :, None] * bb1d[..., None, :]))
        if not ok:
            return False
    d[PB1D], d[BB1D] = pb1d, bb1d
    d[DENSE_FREE_MARKER] = torch.ones((), dtype=torch.long, device=pb1d.device)
    for k in DENSE_TEMPLATE_KEYS:
        del d[k]
    return True


def template_pair_rows(d: Dict[str, Any], template_id: int, g0: int, g1: int):
    """(dgram [n,N,39], pb2d [n,N], unit_vector [n,N,3], bb2d [n,N]) for GLOBAL rows g0:g1 of
    template `template_id`, either sliced from the dense featurizer tensors or rebuilt
    (bitwise-identically) from the per-token vectors of a compacted dict."""
    I = slice(g0, g1)
    if "template_distogram" in d:  # dense path (real templates or un-compacted dict)
        return (d["template_distogram"][template_id][I], d["template_pseudo_beta_mask"][template_id][I],
                d["template_unit_vector"][template_id][I], d["template_backbone_frame_mask"][template_id][I])
    if DENSE_FREE_MARKER not in d:
        raise KeyError("template features are neither dense nor compacted (compact_template_features)")
    pb1d = d[PB1D][template_id]
    bb1d = d[BB1D][template_id]
    N = pb1d.shape[-1]
    n = g1 - g0
    dev, dt = pb1d.device, torch.float32
    pb2d = pb1d[I][:, None] * pb1d[None, :]  # [n, N]  (same fp32 product as the featurizer)
    bb2d = bb1d[I][:, None] * bb1d[None, :]
    dgram = torch.zeros((n, N, 39), dtype=dt, device=dev)
    uv = torch.zeros((n, N, 3), dtype=dt, device=dev)
    return dgram, pb2d.to(dt), uv, bb2d.to(dt)


def tp_template_embedder(
    te,
    input_feature_dict: Dict[str, Any],
    z_shard,
    layout: Layout,
    *,
    pair_mask: Optional[torch.Tensor] = None,
    triangle_attention: str = "torch",
    triangle_multiplicative: str = "torch",
    inplace_safe: bool = False,
    chunk_size: Optional[int] = None,
    add_to_z: bool = False,
):
    """TemplateEmbedder.forward on the row shard.

    Returns u_shard [R, N, c_z] (the trunk does `z += u`), or - with add_to_z=True -
    applies the update blockwise in place (zadd) AFTER u is complete and returns z_shard.
    Returns the python int 0 exactly like stock when the embedder is inactive.
    """
    if "template_aatype" not in input_feature_dict or te.n_blocks < 1:
        return 0
    if layout.replicated:
        u = te(
            input_feature_dict, z_shard, pair_mask=pair_mask,
            triangle_attention=triangle_attention, triangle_multiplicative=triangle_multiplicative,
            inplace_safe=inplace_safe, chunk_size=chunk_size,
        )
        if add_to_z:
            z_shard += u
            return z_shard
        return u
    assert pair_mask is None, "the stock trunk calls TemplateEmbedder without pair_mask (-> all ones)"
    R, N = layout.R, layout.N
    num_templates = input_feature_dict["template_aatype"].shape[0]
    # w = linear_z(LN(z)) is template-independent (stock recomputes the identical GEMM per
    # template): computed once, z read in 128-row blocks.
    blocks = zblocks(R) if R > 0 else [(0, 0)]
    w = torch.cat([te.linear_no_bias_z(te.layernorm_z(zrows(z_shard, i0, i1))) for (i0, i1) in blocks], dim=0)
    z_dtype = zrows(z_shard, 0, min(R, 1)).dtype
    query_num_channels = zrows(z_shard, 0, min(R, 1)).shape[-1]
    u = 0
    for template_id in range(num_templates):
        u = u + _tp_single_template_forward(
            te, template_id, input_feature_dict, w, z_dtype, layout,
            triangle_attention=triangle_attention, triangle_multiplicative=triangle_multiplicative,
            inplace_safe=inplace_safe, chunk_size=chunk_size,
        )
    u = u / (1e-7 + num_templates)
    u = te.linear_no_bias_u(te.relu(u))
    assert u.shape == (R, N, query_num_channels)
    if add_to_z:
        zadd_rows(z_shard, u, layout)
        return z_shard
    return u


def _template_pair_features_rows(te, template_id: int, d: Dict[str, Any], g0: int, g1: int,
                                 z_dtype, N: int) -> torch.Tensor:
    """`at` (pairformer.py 1054-1086) for GLOBAL rows g0:g1 -> [g1-g0, N, 108]."""
    I = slice(g0, g1)
    n = g1 - g0
    asym_id = d["asym_id"]
    multichain_mask = (asym_id[I][:, None] == asym_id[None, :]).to(z_dtype)  # rows I of [N, N]
    pair_mask = torch.ones((n, N), dtype=z_dtype, device=asym_id.device)  # rows I of z.new_ones([N, N])
    to_concat = []
    dgram, pseudo_beta_mask_2d, unit_vector, backbone_mask_2d = template_pair_rows(d, template_id, g0, g1)
    dgram = dgram * multichain_mask[..., None] * pair_mask[..., None]
    pseudo_beta_mask_2d = pseudo_beta_mask_2d * multichain_mask * pair_mask
    to_concat.append(dgram)
    to_concat.append(pseudo_beta_mask_2d.unsqueeze(-1))
    aatype = d["template_aatype"][template_id]  # [N]
    aatype = F.one_hot(aatype, num_classes=len(STD_RESIDUES_WITH_GAP))  # [N, 32]
    to_concat.append(expand_at_dim(aatype, dim=-3, n=n))  # [n, N, 32]: [i, j] = aatype[j]
    to_concat.append(expand_at_dim(aatype[I], dim=-2, n=N))  # [n, N, 32]: [i, j] = aatype[i]
    unit_vector = unit_vector * multichain_mask[..., None] * pair_mask[..., None]
    to_concat.append(unit_vector)
    backbone_mask_2d = backbone_mask_2d * multichain_mask * pair_mask
    to_concat.append(backbone_mask_2d.unsqueeze(-1))
    return torch.concat(to_concat, dim=-1)  # [n, N, 108]


def _tp_single_template_forward(
    te, template_id: int, d: Dict[str, Any], w: torch.Tensor, z_dtype, layout: Layout, *,
    triangle_attention, triangle_multiplicative, inplace_safe, chunk_size,
) -> torch.Tensor:
    R, N, r0 = layout.R, layout.N, layout.r0
    # v = linear_z(z) + linear_a(at), assembled in 128-row blocks (at is fp32 [., N, 108])
    v = None
    for (i0, i1) in (zblocks(R) if R > 0 else [(0, 0)]):
        at = _template_pair_features_rows(te, template_id, d, r0 + i0, r0 + i1, z_dtype, N)
        blk = w[i0:i1] + te.linear_no_bias_a(at)
        if v is None:
            v = blk.new_empty((R,) + tuple(blk.shape[1:]))
        v[i0:i1] = blk
        del at, blk
    # PairformerStack(c_s=0): checkpoint_blocks at inference == sequential blocks
    for blk_mod in te.pairformer_stack.blocks:
        v = tp_pair_stack_block(
            blk_mod, v, layout,
            triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
            inplace_safe=inplace_safe, chunk_size=chunk_size,
        )
    v = te.layernorm_v(v)
    return v
