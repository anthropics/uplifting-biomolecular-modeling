"""Small helpers of the msa / template seams that are NOT part of the ptx_tp.dist contract:
an in-place tensor broadcast and thin z-shard row accessors."""
from __future__ import annotations

import torch
import torch.distributed as dist


def broadcast_tensor(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    """In-place broadcast of a tensor from rank src (no-op when not distributed)."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast(t, src=src)
    return t


# --------------------------------------------------------------------------- #
# Thin z-shard accessors (plain bf16 tensors; kept as trivial helpers
# so op code reads/updates z in 128-row LOCAL blocks).  LOCAL row indices; tail block allowed.
# --------------------------------------------------------------------------- #
ZBLK = 128


def zrows(z, i0: int, i1: int) -> torch.Tensor:
    """rows i0:i1 of the shard as a compute-dtype tensor (a VIEW for plain tensors)."""
    return z[i0:i1]


def zwrite(z, i0: int, i1: int, x: torch.Tensor) -> None:
    """copy rows back; no-op if x IS the view returned by zrows."""
    dst = z[i0:i1]
    if x.data_ptr() == dst.data_ptr() and x.shape == dst.shape and x.stride() == dst.stride():
        return
    dst.copy_(x)


def zadd(z, i0: int, i1: int, delta: torch.Tensor) -> None:
    """in-place z[i0:i1] += delta."""
    z[i0:i1] += delta


def zblocks(R: int, blk: int = ZBLK):
    """local row blocks [(i0, i1), ...] of size blk (tail allowed)."""
    return [(i0, min(R, i0 + blk)) for i0 in range(0, R, blk)]
