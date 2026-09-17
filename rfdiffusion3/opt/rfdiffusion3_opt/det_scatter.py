"""Deterministic replacements for atomic scatter-mean / scatter-add over an atom->token index (the deterministic recipe's statement
(3), det.py): the SAME mean upstream computes, in a FIXED reduction order.

  stock:  out[i] = mean_{j: idx[j]==i} src[j]   computed with CUDA atomicAdd (index_reduce / scatter_add_) -> the summation order over j
          is undefined -> run-to-run differences of a few ulp on a few % of elements.
  here:   the same mean as a *segment* reduction with a fixed order: (1) stable-sort the index (no-op when atoms of a token are
          contiguous, upstream's layout), (2) pad every token's atoms into a [n_token, max_atoms_per_token] gather table (built from the
          index only, cached per structure), (3) gather src rows and reduce with torch.sum over the padded axis in fp32 (one fixed tree
          order per (shape, dtype, device) -> bitwise repeatable run to run and under CUDA-graph replay), (4) divide by the per-token
          count (clamped at 1, as upstream does).
  Precision: accumulation in fp32 regardless of src dtype (a constant of the recipe, not a switch).

The entry point mirrors the upstream signature — foundry_scatter_mean(zeros, dim, index, source) == foundry.utils.torch.scatter_mean —
and falls back to the stock implementation for layouts it does not handle (dim other than -2 / index not 1-D or [..., N] with identical
rows), so behaviour outside the atom->token aggregation is unchanged.
"""
from __future__ import annotations
import torch
from torch import Tensor

_TABLE_CACHE: dict = {}
# Accumulation dtype: "fp32" (default; for fp32 inputs identical precision class to the stock ops; for bf16 inputs under autocast
# this is MORE precise than index_reduce's bf16 atomics -> a precision change for bf16 call sites, stated in the README) or
# "src" (accumulate in the source dtype; torch.sum still uses fp32 partial sums internally for bf16, so results are >= as accurate).
ACCUMULATE = "fp32"                                   # the recipe's accumulation dtype: a constant (fp32 regardless of the source dtype)
_MAX_CACHE = 64
STATS = dict(calls=0, fallbacks=0, table_builds=0)


def _gather_table(index: Tensor, dim_size: int):
    """index: 1-D long [N] -> (order [N] long, table [dim_size, max_per] long into `order`-sorted rows, valid mask, counts).
    Cached on (data_ptr, N, dim_size, device, version) — the atom->token map is constant through a trajectory."""
    key = (index.data_ptr(), index.numel(), int(dim_size), index.device, index._version, index.dtype)
    ent = _TABLE_CACHE.get(key)
    if ent is not None:
        return ent
    # [capture-safety] Under CUDA-graph capture a table CANNOT be built (bincount/.item() = host sync, and the
    # capture pool gives fresh data_ptrs, so the identity key above misses even for the same map, e.g. when a call site does
    # `.long()` on an int32 map every call).  The graph samplers of this kit run >= 2 eager warm-up iterations of the SAME fold on
    # the side stream right before capture, so the most recent entry with the same (numel, dim_size, device, dtype) IS that fold's
    # map: reuse it.  If no such entry exists the original error surfaces (never a silent atomics fallback).  A wrong table would
    # make a graph-replayed run differ from the eager one (never a silent pass).
    if index.is_cuda and torch.cuda.is_current_stream_capturing():
        shape_key = (index.numel(), int(dim_size), index.device, index.dtype)
        for k in reversed(list(_TABLE_CACHE)):
            if (k[1], k[2], k[3], k[5]) == shape_key:
                STATS["capture_shape_key_hits"] = STATS.get("capture_shape_key_hits", 0) + 1
                return _TABLE_CACHE[k]
    STATS["table_builds"] += 1
    idx = index.reshape(-1).long()
    N = idx.numel()
    # stable sort => fixed order of atoms within a token (original order); contiguous layouts give order == arange
    sorted_idx, order = torch.sort(idx, stable=True)
    counts = torch.bincount(sorted_idx, minlength=dim_size)[:dim_size]                    # [dim_size] (indices >= dim_size are dropped like scatter would error)
    max_per = int(counts.max().item()) if N > 0 else 1                                     # one host sync per NEW table only (cached afterwards)
    max_per = max(max_per, 1)
    starts = torch.cumsum(counts, 0) - counts                                              # [dim_size]
    ar = torch.arange(max_per, device=idx.device)
    pos = starts[:, None] + ar[None, :]                                                    # [dim_size, max_per] positions into the sorted list
    valid = ar[None, :] < counts[:, None]
    pos = torch.where(valid, pos, torch.zeros_like(pos)).clamp_(max=max(N - 1, 0))
    table = order[pos] if N > 0 else pos                                                   # rows of src (original numbering)
    ent = (table, valid, counts)
    if len(_TABLE_CACHE) >= _MAX_CACHE:
        _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
    _TABLE_CACHE[key] = ent
    return ent


def segment_reduce_rows(src: Tensor, index: Tensor, dim_size: int, reduce: str = "mean") -> Tensor:
    """src [..., N, C], index [N] long -> out [..., dim_size, C]; reduce in {'mean','sum'}; deterministic; fp32 accumulate."""
    table, valid, counts = _gather_table(index, dim_size)
    lead = src.shape[:-2]; N, C = src.shape[-2], src.shape[-1]
    g = src[..., table.reshape(-1), :].reshape(*lead, table.shape[0], table.shape[1], C)   # [..., dim_size, max_per, C]
    acc = torch.float32 if ACCUMULATE == "fp32" else src.dtype
    g = g.to(acc) * valid.to(acc)[..., None]                                               # zero the padding rows
    s = g.sum(dim=-2)                                                                      # fixed reduction order for a given shape (torch.sum accumulates
                                                                                           # low-precision inputs in fp32 internally on CUDA and CPU)
    if reduce == "mean":
        s = s / counts.clamp(min=1).to(s.dtype)[:, None]
    return s.to(src.dtype)


# ---------------------------------------------------------------- foundry ----------------------------------------------------------------
def foundry_scatter_mean(zeros: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
    """Drop-in for foundry.utils.torch.scatter_mean (zeros (...,I,C), dim=-2 or source.dim()-2, index 1-D [N], source (...,N,C))."""
    STATS["calls"] += 1
    ndim = source.dim(); d = dim + ndim if dim < 0 else dim
    if d != ndim - 2 or index.dim() != 1 or zeros.shape[-1] != source.shape[-1]:
        STATS["fallbacks"] += 1
        return zeros.index_reduce(dim, index, source, "mean", include_self=False)
    out = segment_reduce_rows(source, index, zeros.shape[-2], "mean")
    if zeros.shape[:-2] != out.shape[:-2]:
        out = out.expand(zeros.shape) if out.dim() == zeros.dim() else out.reshape(zeros.shape)
    return (zeros + out) if zeros.requires_grad else out.to(zeros.dtype)      # zeros is all-zero by contract (include_self=False)
