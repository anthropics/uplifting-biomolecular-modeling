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
ptx_tp.triatt -- tensor-parallel TriangleAttention (Protenix-v2).

tp_triatt(mod, z_shard, layout, *, starting, triangle_attention, chunk_size, inplace_safe) -> update_shard

starting node: bias rows linear(LN(z_I)) computed locally -> all_gather_rows -> the
full [1, H, N, N] triangle bias on every rank (this IS stock's `triangle_bias`
tensor) -> the STOCK TriangleAttention statement for the rank's rows, iterating
stock's chunk_layer grid in GLOBAL coordinates and calling the same
`mod.mha(...)` (hence the same kernel: 'torch' or 'cuequivariance').
ending node: in PairformerBlock `tri_att_end` is a starting=True module applied
to z.transpose(-2,-3); TP: transpose_shards(z) -> starting statement with the
module's own weights -> transpose_shards back (update returned in row layout).

Large-N provisions: head-chunked kernel calls (PTX_TP_TRIATT_HEADS_PER_CALL,
default all heads when every launch stays < 2**31 elements) with sliced
q/k/v/bias; the gathered bias stays in the trunk dtype (bf16) and is upcast
per head group only inside the cuEquivariance call.

PAD8 arm (PTX_TP_TRIATT_PAD8=1 or pad8=True; default OFF):
the cuEquivariance kernel takes its Blackwell (sm100) fast path only for sequence length % 8 == 0 and mask=None.
PAD8 pads q/k/v along the sequence axis to a multiple of 8 (zeros), pads the
bias with 0 for pad query rows and -1e9 for pad KEY columns, calls the kernel
with mask=None and drops the pad query rows.  Requires an all-ones pair mask
(the trunk).  On sm100 this selects a DIFFERENT kernel than the stock call form
=> rounding-level differences vs stock; on sm80/sm90 the same kernel runs.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

import torch

from .dist import Layout, all_gather_rows, is_dist, transpose_shards
from .contract import INT32_MAX

__all__ = ["tp_triatt", "tp_triatt_", "SUPPORTED_TRIANGLE_ATTENTION"]

SUPPORTED_TRIANGLE_ATTENTION = ("torch", "cuequivariance")


def _permute_final_dims(tensor: torch.Tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def _heads_per_call(H: int, N: int, rows: int, triangle_attention: str, override: Optional[int]) -> int:
    env = os.environ.get("PTX_TP_TRIATT_HEADS_PER_CALL")
    if override is not None:
        return max(1, min(H, int(override)))
    if env:
        return max(1, min(H, int(env)))
    hg = H
    # bias slice [1, hg, N, N] (and its fp32 copy) must stay < 2**31 elements
    while hg > 1 and hg * N * N > INT32_MAX:
        hg -= 1
    if triangle_attention == "torch":
        # torch path materialises logits [rows, hg, N, N]
        while hg > 1 and rows * hg * N * N > INT32_MAX:
            hg -= 1
    return hg


PAD8_NEG = -1e9


def _pad8_enabled(override: Optional[bool]) -> bool:
    if override is not None:
        return bool(override)
    return os.environ.get("PTX_TP_TRIATT_PAD8", "0") == "1"


def _mha_heads(mha, q_x: torch.Tensor, kv_x: torch.Tensor, biases: List[torch.Tensor], triangle_attention: str,
               hpc: int, pad8: bool = False) -> torch.Tensor:
    """`Attention.forward` (protenix.model.triangular.layers) with the kernel
    invoked per head group of `hpc` heads (and optionally the PAD8 call form for
    the cuEquivariance kernel).  hpc >= no_heads and not pad8 -> the stock method
    itself is called (identical code path)."""
    H = mha.no_heads
    if hpc >= H and not (pad8 and triangle_attention == "cuequivariance"):
        return mha(q_x=q_x, kv_x=kv_x, biases=biases, triangle_attention=triangle_attention)
    from protenix.model.triangular.layers import _attention, cuequivariance_triangular_attn
    import torch.nn.functional as F
    assert triangle_attention in ("torch", "cuequivariance"), triangle_attention
    q, k, v = mha._prep_qkv(q_x, kv_x, apply_scale=triangle_attention in ["torch", "triattention"])
    if q.shape[-2] <= 16:
        triangle_attention = "torch"   # (stock quirk reproduced: q stays unscaled in this fallback)
    outs = []
    if triangle_attention == "cuequivariance":
        scale = 1.0 / math.sqrt(mha.c_hidden)
        Nk = k.shape[-2]
        padn = (-Nk) % 8 if pad8 else 0
        mask_b = None if pad8 else (biases[0] == 0).bool()
        # exact arm must keep stock's dense-mask call form (mask=None would select a different kernel on cc>=10)
        assert pad8 or (mask_b is not None and torch.is_tensor(mask_b)), "exact arm must pass the dense mask tensor"
        for h0 in range(0, H, hpc):
            h1 = min(H, h0 + hpc)
            if h0 == 0 and h1 == H:
                qg, kg, vg = q, k, v
            else:
                qg, kg, vg = q[..., h0:h1, :, :].contiguous(), k[..., h0:h1, :, :].contiguous(), v[..., h0:h1, :, :].contiguous()
            bias_g = biases[1][..., h0:h1, :, :].float()
            if padn:
                qg = F.pad(qg, (0, 0, 0, padn))
                kg = F.pad(kg, (0, 0, 0, padn))
                vg = F.pad(vg, (0, 0, 0, padn))
                bias_g = F.pad(bias_g, (0, padn, 0, padn))
                bias_g[..., :, Nk:] = PAD8_NEG
            assert bias_g.numel() <= INT32_MAX and qg.numel() <= INT32_MAX
            o_g = cuequivariance_triangular_attn(qg, kg, vg, bias_g, mask_b, scale)[0]
            if padn:
                o_g = o_g[..., :Nk, :]
            outs.append(o_g)
            del bias_g, qg, kg, vg
    else:
        for h0 in range(0, H, hpc):
            h1 = min(H, h0 + hpc)
            assert q.shape[0] * (h1 - h0) * q.shape[-2] * k.shape[-2] <= INT32_MAX
            outs.append(_attention(q[..., h0:h1, :, :], k[..., h0:h1, :, :], v[..., h0:h1, :, :],
                                   [biases[0], biases[1][..., h0:h1, :, :]]))
    o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-3)          # [*, H, Q, c_hidden]
    del outs
    o = o.transpose(-2, -3)
    return mha._wrap_up(o, q_x)


def _start_from_ln(mod, x: torch.Tensor, layout: Layout, *, triangle_attention: str, chunk_size: Optional[int],
                   inplace_safe: bool, heads_per_call: Optional[int] = None,
                   bias_head_chunk: Optional[int] = None, pad8: Optional[bool] = None) -> torch.Tensor:
    """stock TriangleAttention.forward from `x = self.layer_norm(x)` onwards,
    for the rank's rows; x = LN(z rows) [R, N, C]."""
    R, N = x.shape[0], x.shape[1]
    mask = x.new_ones(x.shape[:-1])                                   # [R, N]  (stock: x.new_ones)
    mask_bias = (mod.inf * (mask - 1))[..., :, None, None, :]         # [R, 1, 1, N]
    tb_rows = mod.linear(x)                                           # [R, N, H]
    H = tb_rows.shape[-1]
    bhc = bias_head_chunk or int(os.environ.get("PTX_TP_TRIATT_BIAS_HEAD_CHUNK", "0") or 0)
    if bhc and bhc < H:
        tb_full = torch.empty((N, N, H), dtype=tb_rows.dtype, device=tb_rows.device)
        for h0 in range(0, H, bhc):
            h1 = min(H, h0 + bhc)
            tb_full[..., h0:h1].copy_(all_gather_rows(tb_rows[..., h0:h1].contiguous(), layout))
    else:
        tb_full = all_gather_rows(tb_rows.contiguous(), layout)       # [N, N, H] == stock linear(x) on all rows
    del tb_rows
    triangle_bias = _permute_final_dims(tb_full, (2, 0, 1)).unsqueeze(-4)   # [1, H, N, N]
    hpc = _heads_per_call(H, N, (chunk_size or max(R, 1)), triangle_attention, heads_per_call)
    p8 = _pad8_enabled(pad8)
    if R == 0:
        return x
    if chunk_size is None:
        return _mha_heads(mod.mha, x, x, [mask_bias, triangle_bias], triangle_attention, hpc, p8)
    # stock chunk_layer over the flattened batch dim (= rows), grid in GLOBAL coordinates
    out = x if inplace_safe else None
    for c0, c1 in layout.chunks(chunk_size):
        l0, l1 = c0 - layout.r0, c1 - layout.r0
        o = _mha_heads(mod.mha, x[l0:l1], x[l0:l1], [mask_bias[l0:l1], triangle_bias], triangle_attention, hpc, p8)
        if out is None:
            out = o.new_zeros((R,) + tuple(o.shape[1:]))
        out[l0:l1] = o
        del o
    return out


def _triatt_rows(N: int, override: Optional[int] = None) -> int:
    if override:
        return int(override)
    env = os.environ.get("PTX_TP_TRIATT_ROWS")
    if env:
        return max(1, int(env))
    return 256 if N <= 16384 else 128


def _autocast_dtype(x: torch.Tensor) -> torch.dtype:
    if x.is_cuda and torch.is_autocast_enabled():
        try:
            return torch.get_autocast_dtype("cuda")
        except Exception:  # pragma: no cover
            return torch.get_autocast_gpu_dtype()
    return x.dtype


def _gather_triangle_bias(mod, X: torch.Tensor, layout: Layout, rb: int, bias_head_chunk: Optional[int] = None):
    """triangle_bias = permute_final_dims(linear(LN(X_all_rows)), (2,0,1)).unsqueeze(-4) as a VIEW
    over a [N, N, H] buffer (identical layout to stock), computed from LN of local row blocks
    (row-local) and all-gathered.  Peak = the [N,N,H] buffer + local [R,N,H] rows + one LN block."""
    R, N = X.shape[0], X.shape[1]
    H = mod.linear.weight.shape[0]
    tb_rows = None
    for i0 in range(0, R, rb):
        i1 = min(R, i0 + rb)
        t = mod.linear(mod.layer_norm(X[i0:i1]))                      # [rb, N, H]
        if tb_rows is None:
            tb_rows = torch.empty((R, N, H), dtype=t.dtype, device=X.device)
        tb_rows[i0:i1] = t
        del t
    if tb_rows is None:                                                  # zero-row rank
        tb_rows = torch.empty((0, N, H), dtype=_autocast_dtype(X), device=X.device)
    bhc = bias_head_chunk or int(os.environ.get("PTX_TP_TRIATT_BIAS_HEAD_CHUNK", "0") or 0)
    if layout.P == 1 or layout.replicated or not is_dist():
        tb_full = tb_rows
    elif bhc and bhc < H:
        tb_full = torch.empty((N, N, H), dtype=tb_rows.dtype, device=tb_rows.device)
        for h0 in range(0, H, bhc):
            h1 = min(H, h0 + bhc)
            tb_full[..., h0:h1].copy_(all_gather_rows(tb_rows[..., h0:h1].contiguous(), layout))
    else:
        tb_full = all_gather_rows(tb_rows.contiguous(), layout)          # [N, N, H]
    del tb_rows
    return _permute_final_dims(tb_full, (2, 0, 1)).unsqueeze(-4)         # [1, H, N, N] view


def _attn_rows_add_(mod, X: torch.Tensor, layout: Layout, triangle_bias: torch.Tensor, *, triangle_attention: str,
                    chunk_size: Optional[int], heads_per_call: Optional[int], pad8: Optional[bool], rb: int) -> torch.Tensor:
    """X[b] += mha(q_x=LN(X[b]), kv_x=LN(X[b]), biases=[mask_bias_b, triangle_bias]) over row blocks b.
    Row i's q/k/v/gate come only from row i and the bias was built beforehand, so the per-block
    in-place residual add is the identical arithmetic to stock's `z += tri_att(z)` (elementwise
    bf16 adds of the same values).  Blocks = stock's chunk grid in GLOBAL coordinates clipped to
    the rank's rows (sub-split to <= rb rows), or rb-row blocks when chunk_size is None."""
    R, N = X.shape[0], X.shape[1]
    H = mod.linear.weight.shape[0]
    if chunk_size is not None and not (layout.P == 1 and layout.N != R):
        blocks = []
        for c0, c1 in layout.chunks(chunk_size):
            l0, l1 = c0 - layout.r0, c1 - layout.r0
            for b0 in range(l0, l1, rb):
                blocks.append((b0, min(l1, b0 + rb)))
    else:
        blocks = [(i0, min(R, i0 + rb)) for i0 in range(0, R, rb)]
    hpc = _heads_per_call(H, N, rb, triangle_attention, heads_per_call)
    p8 = _pad8_enabled(pad8)
    for (l0, l1) in blocks:
        if l1 <= l0:
            continue
        x_b = mod.layer_norm(X[l0:l1])                                   # [rb, N, C]
        mask_bias = torch.zeros((l1 - l0, 1, 1, N), dtype=X.dtype, device=X.device)   # inf * (ones - 1) == +0.0
        o = _mha_heads(mod.mha, x_b, x_b, [mask_bias, triangle_bias], triangle_attention, hpc, p8)
        del x_b, mask_bias
        X[l0:l1] += o
        del o
    return X


def _owns_storage(t: torch.Tensor) -> bool:
    try:
        st = t.untyped_storage()
        return t.is_contiguous() and t.storage_offset() == 0 and st.nbytes() == t.numel() * t.element_size()
    except Exception:
        return False


def _release_storage(t: torch.Tensor) -> bool:
    """Free the memory behind `t` (all Python references stay valid objects; data is gone).
    Used for z while it lives on as zT (z == transpose(zT) exactly, nothing is lost)."""
    if os.environ.get("PTX_TP_TRIATT_FREE_Z", "1") != "1" or not _owns_storage(t):
        return False
    t.untyped_storage().resize_(0)
    return True


def _restore_storage(t: torch.Tensor) -> None:
    t.untyped_storage().resize_(t.numel() * t.element_size())


def tp_triatt_(mod, z_shard: torch.Tensor, layout: Layout, *, starting: bool, triangle_attention: str = "torch",
               chunk_size: Optional[int] = None, inplace_safe: bool = True, heads_per_call: Optional[int] = None,
               bias_head_chunk: Optional[int] = None, pad8: Optional[bool] = None,
               block_rows: Optional[int] = None, transpose_block_rows: Optional[int] = None) -> torch.Tensor:
    """IN-PLACE tensor-parallel triangle attention residual:  z_shard += TriAtt(z)  (starting) or
    the ending-node statement of PairformerBlock (z^T += TriAtt_end(z^T), expressed on shards).
    Returns z_shard (same tensor object / storage).  TWO-SHARD RULE: peak device memory =
      starting: 1 shard + [N,N,H] bias + O(block) transients;
      ending  : 2 shards during the two streamed transposes; during attention z's storage is
                RELEASED (PTX_TP_TRIATT_FREE_Z=1, default; only when z owns its storage) so
                1 shard (zT) + bias + transients; z's storage is re-allocated and refilled by the
                back-transpose (all outside references to z remain valid).
    Bitwise identical to `z += tp_triatt(...)` / stock (per-block elementwise adds of identical values).
    Env: PTX_TP_TRIATT_ROWS (row block, default 256; 128 above 16k tokens),
         PTX_TP_TRANSPOSE_BLOCK_ROWS (streamed transpose window; default 256 when the shard > 2 GB),
         PTX_TP_TRIATT_HEADS_PER_CALL, PTX_TP_TRIATT_BIAS_HEAD_CHUNK, PTX_TP_TRIATT_PAD8, PTX_TP_TRIATT_FREE_Z."""
    assert getattr(mod, "starting", True), "PairformerBlock's tri_att modules are starting=True (block transposes z)"
    if triangle_attention not in SUPPORTED_TRIANGLE_ATTENTION:
        raise NotImplementedError(f"tp_triatt_ supports {SUPPORTED_TRIANGLE_ATTENTION}, got {triangle_attention!r}")
    N = layout.N
    rb = _triatt_rows(N, block_rows)
    akw = dict(triangle_attention=triangle_attention, chunk_size=chunk_size, heads_per_call=heads_per_call, pad8=pad8, rb=rb)
    single = layout.replicated or not is_dist() or layout.P == 1
    if single:
        L1 = Layout(z_shard.shape[0], 1, 0)
        X = z_shard if starting else z_shard.transpose(-2, -3).contiguous()
        tb = _gather_triangle_bias(mod, X, L1, rb, bias_head_chunk)
        _attn_rows_add_(mod, X, L1, tb, **akw)
        del tb
        if not starting:
            z_shard.copy_(X.transpose(-2, -3))
        return z_shard
    if starting:
        tb = _gather_triangle_bias(mod, z_shard, layout, rb, bias_head_chunk)
        _attn_rows_add_(mod, z_shard, layout, tb, **akw)
        del tb
        return z_shard
    # ---- ending node: work in the transposed frame (this IS stock's z.transpose(-2,-3) statement)
    shard_bytes = layout.Rmax * N * z_shard.shape[-1] * z_shard.element_size()
    tbr = transpose_block_rows or int(os.environ.get("PTX_TP_TRANSPOSE_BLOCK_ROWS", "0") or 0) or (256 if shard_bytes > 2 * 2**30 else None)
    zT = transpose_shards(z_shard, layout, block_rows=tbr)               # 2 shards (momentarily)
    freed = _release_storage(z_shard)                                     # -> 1 shard (z data == zT^T, nothing lost)
    if os.environ.get("PTX_TP_VERBOSE", "0") == "1" and layout.rank == 0:
        print(f"[ptx_tp.triatt] ending node N={N}: z storage released during attention: {freed} "
              f"(z owns storage: {_owns_storage(z_shard) or freed})", flush=True)
    tb = None
    try:
        tb = _gather_triangle_bias(mod, zT, layout, rb, bias_head_chunk)
        _attn_rows_add_(mod, zT, layout, tb, **akw)
        tb = None
        if freed:
            _restore_storage(z_shard)                                     # 2 shards again (uninitialised z)
            freed = False
        transpose_shards(zT, layout, block_rows=tbr or 128, out=z_shard)  # streamed back INTO z's storage
    finally:
        del tb, zT                                                        # never leave the zT copy / bias alive (also on exceptions)
        if freed:
            _restore_storage(z_shard)
    return z_shard


def tp_triatt(mod, z_shard: torch.Tensor, layout: Layout, *, starting: bool, triangle_attention: str = "torch",
              chunk_size: Optional[int] = None, inplace_safe: bool = False,
              heads_per_call: Optional[int] = None, bias_head_chunk: Optional[int] = None,
              pad8: Optional[bool] = None) -> torch.Tensor:
    """Update-returning API: returns the attention UPDATE for the rank's rows, in row
    layout [R, N, C] (caller adds it).  Memory: starting = z + update (2 shard-eq); ending = 3
    shard-eq momentarily.  Prefer the in-place `tp_triatt_` (two-shard rule) at large N."""
    assert getattr(mod, "starting", True), "PairformerBlock's tri_att modules are starting=True (block transposes z)"
    if triangle_attention not in SUPPORTED_TRIANGLE_ATTENTION:
        raise NotImplementedError(f"tp_triatt supports {SUPPORTED_TRIANGLE_ATTENTION}, got {triangle_attention!r}")
    kw = dict(triangle_attention=triangle_attention, chunk_size=chunk_size, inplace_safe=inplace_safe,
              heads_per_call=heads_per_call, bias_head_chunk=bias_head_chunk, pad8=pad8)
    if layout.replicated or not is_dist() or layout.P == 1:
        if _pad8_enabled(pad8) and triangle_attention == "cuequivariance":
            L1 = Layout(z_shard.shape[0], 1, 0)
            zin = z_shard if starting else z_shard.transpose(-2, -3).contiguous()
            u = _start_from_ln(mod, mod.layer_norm(zin), L1, **kw)
            return u if starting else u.transpose(-2, -3)
        if starting:
            return mod(z_shard, mask=None, chunk_size=chunk_size, triangle_attention=triangle_attention,
                       inplace_safe=inplace_safe)
        zT = z_shard.transpose(-2, -3).contiguous()
        u = mod(zT, mask=None, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
        return u.transpose(-2, -3)
    if starting:
        x = mod.layer_norm(z_shard)
        return _start_from_ln(mod, x, layout, **kw)
    zT = transpose_shards(z_shard, layout)          # this IS stock's z.transpose(-2,-3).contiguous(), sharded
    x = mod.layer_norm(zT)
    del zT
    uT = _start_from_ln(mod, x, layout, **kw)
    del x
    u = transpose_shards(uT, layout)
    del uT
    return u
