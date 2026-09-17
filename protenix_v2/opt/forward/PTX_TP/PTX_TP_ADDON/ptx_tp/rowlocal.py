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
"""
ptx_tp.rowlocal -- row-local pieces of the Pairformer block.

tp_transition(mod, x_shard)                        pair Transition on the shard (calls the
                                                   installed Transition.forward; compatible with
                                                   row-chunking wrappers such as PTX_XL_TRANS_CHUNK)
tp_transition_add_(mod, z_shard)                   z[i0:i1] += mod(z[i0:i1]) in row blocks (no whole-shard transient)
tp_attention_pair_bias_single(apb, s, z_shard, L)  AttentionPairBias(a=s, s=None, z=z) for the single rep:
                                                   bias rows linear_nobias_z(layernorm_z(z_I)) -> all_gather ->
                                                   full [h, N, N] bias; then EITHER the stock attention
                                                   statement replicated on every rank (default; exact by
                                                   construction) OR queries = rows I, keys = all N, output
                                                   rows all_gathered (shard_queries=True; exact only if the
                                                   SDPA kernel is query-count invariant -- see unit tests).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

from .dist import Layout, all_gather_rows, is_dist
from .contract import INT32_MAX, max_rows_per_launch

__all__ = ["tp_transition", "tp_transition_add_", "tp_attention_pair_bias_single", "tp_single_transition"]


def _permute_final_dims(tensor: torch.Tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def _row_block(x_shard: torch.Tensor) -> int:
    env = os.environ.get("PTX_TP_TRANS_ROWS")
    if env:
        return max(1, int(env))
    R = x_shard.shape[0]
    if x_shard.numel() * 4 <= INT32_MAX:          # hidden = 4*c intermediate stays < 2**31 too
        return max(R, 1)
    N, C = x_shard.shape[-2], x_shard.shape[-1]
    return max(1, min(R, max_rows_per_launch(N, 4 * C)))


def tp_transition(mod, x_shard: torch.Tensor, layout: Optional[Layout] = None) -> torch.Tensor:
    """Row-local Transition: returns mod(x_shard) (the update).  Rows are processed in
    blocks only when needed for the int32 rule (PTX_TP_TRANS_ROWS overrides)."""
    R = x_shard.shape[0]
    rb = _row_block(x_shard)
    if rb >= R:
        return mod(x_shard)
    out = torch.empty_like(x_shard)
    for i0 in range(0, R, rb):
        i1 = min(R, i0 + rb)
        out[i0:i1] = mod(x_shard[i0:i1])
    return out


def tp_transition_add_(mod, z_shard: torch.Tensor, layout: Optional[Layout] = None, rows: Optional[int] = None) -> torch.Tensor:
    """In-place z_shard[i0:i1] += mod(z_shard[i0:i1]) over row blocks (Transition is
    pointwise in (i,j), so updating earlier rows never feeds later rows)."""
    R = z_shard.shape[0]
    rb = rows or _row_block(z_shard)
    if rb >= R:
        z_shard += mod(z_shard)
        return z_shard
    for i0 in range(0, R, rb):
        i1 = min(R, i0 + rb)
        z_shard[i0:i1] += mod(z_shard[i0:i1])
    return z_shard


def tp_single_transition(mod, s: torch.Tensor) -> torch.Tensor:
    """single-rep Transition (s replicated): the stock call."""
    return mod(s)


def tp_attention_pair_bias_single(apb, s: torch.Tensor, z_shard: torch.Tensor, layout: Layout, *,
                                  shard_queries: bool = False, inplace_safe: bool = False) -> torch.Tensor:
    """Pairformer's AttentionPairBias(a=s, s=None, z=z) with z row-sharded; returns the
    update for s (replicated on every rank; caller does s = s + update)."""
    assert not apb.has_s, "Pairformer single-rep attention has has_s=False"
    assert not getattr(apb, "cross_attention_mode", False)
    if layout.replicated or not is_dist() or layout.P == 1:
        return apb(a=s, s=None, z=z_shard)
    # ---- AttentionPairBias.forward: a = layernorm_a(a)
    a = apb.layernorm_a(s)
    # ---- standard_multihead_attention (enable_efficient_fusion=False): bias = linear_nobias_z(layernorm_z(z))
    R = z_shard.shape[0]
    rb = max_rows_per_launch(z_shard.shape[1], z_shard.shape[2])
    if rb >= R:
        bias_rows = apb.linear_nobias_z(apb.layernorm_z(z_shard))                    # [R, N, h]
    else:
        bias_rows = torch.cat([apb.linear_nobias_z(apb.layernorm_z(z_shard[i0:min(R, i0 + rb)]))
                               for i0 in range(0, R, rb)], dim=0)
    local_bias = bool(shard_queries) and os.environ.get("PTX_TP_APB_LOCAL_BIAS", "1") == "1"
    if local_bias:      # sharded queries only ever read bias rows r0:r1 == this rank's own rows -> no [N,N,h] all_gather (31 GB at 31k)
        bias = _permute_final_dims(bias_rows.contiguous(), [2, 0, 1])                    # [h, R, N]
        del bias_rows
    else:
        bias_full = all_gather_rows(bias_rows.contiguous(), layout)                       # [N, N, h]
        del bias_rows
        bias = _permute_final_dims(bias_full, [2, 0, 1])                                  # [h, N, N]
    att = apb.attention
    if not shard_queries:
        a = att(q_x=a, kv_x=a, attn_bias=bias, inplace_safe=inplace_safe)
        return a
    # ---- sharded queries: primitives.Attention.forward with q rows = I, k/v = all
    from protenix.model.modules.primitives import _attention
    r0, r1 = layout.r0, layout.r1
    q = att.linear_q(a[r0:r1])
    k = att.linear_k(a)
    v = att.linear_v(a)
    q = q.view(q.shape[:-1] + (att.num_heads, -1)).transpose(-2, -3)
    k = k.view(k.shape[:-1] + (att.num_heads, -1)).transpose(-2, -3)
    v = v.view(v.shape[:-1] + (att.num_heads, -1)).transpose(-2, -3)
    q = q / math.sqrt(att.c_hidden)
    attn_bias = bias if local_bias else bias[..., r0:r1, :]
    if len(attn_bias.shape) != len(q.shape):
        attn_bias = attn_bias.unsqueeze(dim=-3)
    o = _attention(q=q, k=k, v=v, attn_bias=attn_bias, use_efficient_implementation=att.use_efficient_implementation,
                   inplace_safe=inplace_safe)                                          # [h, R, c]
    o = o.transpose(-2, -3)
    o = att._wrap_up(o, a[r0:r1])                                                     # [R, c_s]
    return all_gather_rows(o.contiguous(), layout)
