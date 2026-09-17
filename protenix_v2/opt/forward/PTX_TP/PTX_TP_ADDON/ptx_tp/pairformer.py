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
ptx_tp.pairformer -- tensor-parallel PairformerBlock / PairformerStack (Protenix-v2).

Reproduces PairformerBlock.forward's inplace branch op order on a row shard:
    z = tri_mul_out(z) (+=, fused) ; z = tri_mul_in(z) (+=) ; z += tri_att_start(z) ;
    [transpose] z += tri_att_end(z) [transpose back] ; z += pair_transition(z) ;
    if c_s > 0: s = s + attention_pair_bias(a=s, z=z) ; s = s + single_transition(s)
pair_mask must be None (as in the Protenix trunk; every mask is then all-ones).
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

import torch

from .dist import Layout, is_dist
from .trimul import tp_trimul
from .triatt import tp_triatt, tp_triatt_
from .rowlocal import tp_attention_pair_bias_single, tp_transition_add_, tp_single_transition

__all__ = ["tp_pairformer_block", "tp_pairformer_stack"]


def release_device_caches(reset_pinned_pool: bool = False, empty_cache: bool = True) -> dict:
    """Driver hook: the pair-stack modules keep NO module-level device buffers between calls (ring/slab/
    all-to-all/bias/assembler/stash buffers are call-local and closed in `finally`).  This runs a
    cyclic GC pass (frees tensors kept alive only by reference cycles or by a caught exception's
    traceback that the caller has since dropped), optionally drops the pinned HOST pool used by
    bcache/zcopy='host', and returns the allocator counters (bytes) for logging."""
    import gc
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    gc.collect()
    if reset_pinned_pool:
        try:
            from .trimul import _PinnedPool
            _PinnedPool.buf = None
        except Exception:
            pass
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    return dict(allocated_before=before, allocated_after=after, freed_by_gc=before - after,
                reserved=(torch.cuda.memory_reserved() if torch.cuda.is_available() else 0))


def memguard(op_name: str, z_shard, layout: Layout, *, max_shards: Optional[float] = None,
             slack_gb: Optional[float] = None) -> Optional[float]:
    """TWO-SHARD RULE check before a pair op: at N >= PTX_TP_MEMGUARD_N (default 15000) assert
    torch.cuda.memory_allocated() <= max_shards (PTX_TP_MEMGUARD_SHARDS, default 2.0) x shard bytes
    + slack (PTX_TP_MEMGUARD_SLACK_GB, default 24) and fail fast with the op name
    (PTX_TP_MEMGUARD_STRICT=0 only prints).  Returns allocated / shard or None when inactive."""
    if not (torch.is_tensor(z_shard) and z_shard.is_cuda):
        return None
    N = layout.N
    if N < int(os.environ.get("PTX_TP_MEMGUARD_N", "15000")):
        return None
    shard = max(1, layout.Rmax * N * int(z_shard.shape[-1]) * z_shard.element_size())
    ms = float(max_shards if max_shards is not None else os.environ.get("PTX_TP_MEMGUARD_SHARDS", "2.0"))
    slack = float(slack_gb if slack_gb is not None else os.environ.get("PTX_TP_MEMGUARD_SLACK_GB", "24")) * 1e9
    alloc = torch.cuda.memory_allocated(z_shard.device)
    limit = ms * shard + slack
    msg = (f"[ptx_tp.memguard] rank {layout.rank} before {op_name}: allocated {alloc / 1e9:.1f} GB = "
           f"{alloc / shard:.2f} shard-eq (shard {shard / 1e9:.1f} GB, limit {ms:g} x shard + {slack / 1e9:g} GB = {limit / 1e9:.1f} GB, "
           f"reserved {torch.cuda.memory_reserved(z_shard.device) / 1e9:.1f} GB)")
    if os.environ.get("PTX_TP_VERBOSE", "0") == "1" or alloc > limit:
        print(msg, flush=True)
    if alloc > limit and os.environ.get("PTX_TP_MEMGUARD_STRICT", "1") == "1":
        raise RuntimeError(msg)
    return alloc / shard



def tp_pairformer_block(block, s: Optional[torch.Tensor], z_shard: torch.Tensor, layout: Layout, *,
                        pair_mask=None, triangle_multiplicative: str = "torch", triangle_attention: str = "torch",
                        inplace_safe: bool = True, chunk_size: Optional[int] = None,
                        shard_single_queries: bool = False, trimul_kwargs: Optional[dict] = None,
                        triatt_kwargs: Optional[dict] = None, timers: Optional[dict] = None,
                        mem: Optional[dict] = None):
    """-> (s, z_shard).  With inplace_safe=True z_shard is updated in place (stock
    semantics); with inplace_safe=False the input shard is left untouched and a new
    shard is returned (same arithmetic as the inplace branch)."""
    assert pair_mask is None, "tp_pairformer_block supports pair_mask=None (Protenix trunk)"
    if triangle_multiplicative != "torch":
        if os.environ.get("PTX_TP_ALLOW_TRIMUL_KERNEL_SUBSTITUTION", "0") == "1":
            triangle_multiplicative = "torch"   # not bitwise-equal to a cuEquivariance TriMul run (explicit opt-in)
        else:
            raise NotImplementedError("tp_pairformer_block reproduces the 'torch' triangle_multiplicative statement; "
                                      "run the reference with --trimul_kernel torch or set "
                                      "PTX_TP_ALLOW_TRIMUL_KERNEL_SUBSTITUTION=1 to accept a TIER-2 substitution")
    if layout.replicated or not is_dist() or layout.P == 1:
        return block(s, z_shard, None, triangle_multiplicative=triangle_multiplicative,
                     triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
    tk = dict(trimul_kwargs or {})
    ak = dict(triatt_kwargs or {})
    z = z_shard if inplace_safe else z_shard.clone()

    def tick(name, t0):
        if timers is not None:
            if z.is_cuda:
                torch.cuda.synchronize()
            timers[name] = timers.get(name, 0.0) + (time.time() - t0)
        if mem is not None and z.is_cuda:
            mem[name] = max(mem.get(name, 0), torch.cuda.max_memory_allocated())

    def now():
        if (timers is not None or mem is not None) and z.is_cuda:
            torch.cuda.synchronize()
            if mem is not None:
                torch.cuda.reset_peak_memory_stats()
        return time.time()

    t0 = now(); memguard("tri_mul_out", z, layout); z = tp_trimul(block.tri_mul_out, z, layout, outgoing=True, with_add=True, **tk); tick("tri_mul_out", t0)
    t0 = now(); memguard("tri_mul_in", z, layout); z = tp_trimul(block.tri_mul_in, z, layout, outgoing=False, with_add=True, **tk); tick("tri_mul_in", t0)
    t0 = now(); memguard("tri_att_start", z, layout)
    tp_triatt_(block.tri_att_start, z, layout, starting=True, triangle_attention=triangle_attention,
               chunk_size=chunk_size, **ak)                       # in place: z += TriAttStart(z), streamed per row block
    tick("tri_att_start", t0)
    t0 = now(); memguard("tri_att_end", z, layout)
    tp_triatt_(block.tri_att_end, z, layout, starting=False, triangle_attention=triangle_attention,
               chunk_size=chunk_size, **ak)                       # in place, transposed frame, z storage released meanwhile
    tick("tri_att_end", t0)
    t0 = now(); memguard("pair_transition", z, layout); tp_transition_add_(block.pair_transition, z, layout); tick("pair_transition", t0)
    if block.c_s > 0:
        t0 = now()
        s = s + tp_attention_pair_bias_single(block.attention_pair_bias, s, z, layout,
                                              shard_queries=(shard_single_queries or layout.N >= int(os.environ.get("PTX_TP_APB_SHARD_ABOVE", "15000"))))   # sharded queries above PTX_TP_APB_SHARD_ABOVE tokens
        tick("attention_pair_bias", t0)
        t0 = now(); s = s + tp_single_transition(block.single_transition, s); tick("single_transition", t0)
    return s, z


def tp_pairformer_stack(stack, s, z_shard, layout: Layout, *, block_callback: Optional[Callable] = None, **kw):
    """Loop over stack.blocks (inference: no activation checkpointing).  -> (s, z_shard)"""
    for bi, block in enumerate(stack.blocks):
        s, z_shard = tp_pairformer_block(block, s, z_shard, layout, **kw)
        if block_callback is not None:
            block_callback(bi, s, z_shard)
    return s, z_shard
