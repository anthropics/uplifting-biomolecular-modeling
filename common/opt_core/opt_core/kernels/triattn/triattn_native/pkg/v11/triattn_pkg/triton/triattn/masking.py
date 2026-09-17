"""Key-mask handling shared by k10 / k11 / k12: the batch element's OR pattern of kept keys is folded into the staged bias, and a
per-row key-tile table tells the kernels which rows still need the per-key select and how many key tiles each row visits.

For mask [B, N, 1, 1, S] (True = keep):
  orpat[b, k]      = any_i mask[b, i, k]  -- keys kept by at least one pair row; keys outside it get the -1e9 / -2**30 mask
                     sentinel written into the staged bias, so rows whose mask EQUALS the OR pattern (every row, for engine
                     padding masks) run the mask-free code path and skip the key tiles past the last OR-kept key;
  rowinfo[b, i, :] = (n_tiles, sel): key tiles of BLOCK_N the row visits (up to its last kept key; ALL tiles for a row that
                     keeps nothing, which yields the uniform average of v over the real keys like cuEquivariance), and the first
                     tile from which the per-key select must run (the tile of the first key where the row differs from the OR
                     pattern; n_all = never for rows equal to it; 0 for empty rows).  A CTA serving several rows visits
                     max(n_tiles) and selects from min(sel).
Three small launches (memset, 2-D OR kernel with atomic max, row kernel), no host synchronisation (CUDA-graph capturable);
`census(rowinfo, n_all)` counts (mask-free rows, per-key-select rows) on demand.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .launch import launch

_LAUNCHES = {}

@triton.jit
def _orpat_kernel(Mask, OrPat, smb, smi, N, SK, KB: tl.constexpr, RB: tl.constexpr):
    """OrPat[b, k] (int32, zero-initialised) |= any over rows r0..r0+RB of Mask[b, r, k]; one [RB, KB] tile per program."""
    pid_k = tl.program_id(0)
    pid_r = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    cols = pid_k * KB + tl.arange(0, KB)
    rows = pid_r * RB + tl.arange(0, RB)
    c_ok = cols < SK
    m = tl.load(Mask + b * smb + rows.to(tl.int64)[:, None] * smi + cols[None, :], mask=(rows < N)[:, None] & c_ok[None, :], other=0)
    anyk = tl.max((m != 0).to(tl.int32), 0)
    tl.atomic_max(OrPat + b * SK + cols, anyk, mask=c_ok & (anyk != 0), sem="relaxed")


@triton.jit
def _rowinfo_kernel(Mask, OrPat, RowInfo, smb, smi, N, SK, n_all,
                    BN: tl.constexpr, RB: tl.constexpr, KB: tl.constexpr, FOLD: tl.constexpr):
    """RowInfo[b, r] = (n_tiles, sel) for rows r0..r0+RB (see module doc).  FOLD: compare against OrPat; else against 'every
    key kept'."""
    pid_r = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    rows = pid_r * RB + tl.arange(0, RB)
    r_ok = rows < N
    last_kept = tl.full([RB], -1, tl.int32)
    first_diff = tl.full([RB], 2147483647, tl.int32)
    mrow = Mask + b * smb + rows.to(tl.int64)[:, None] * smi
    for k0 in range(0, SK, KB):
        cols = k0 + tl.arange(0, KB)
        c_ok = cols < SK
        if FOLD:
            orp = tl.load(OrPat + b * SK + cols, mask=c_ok, other=0) != 0
        else:
            orp = c_ok                      # reference pattern = every key kept (nothing folded into the bias)
        kept = tl.load(mrow + cols[None, :], mask=r_ok[:, None] & c_ok[None, :], other=0) != 0
        last_kept = tl.maximum(last_kept, tl.max(tl.where(kept, cols[None, :], -1), 1))
        diff = (kept != orp[None, :]) & c_ok[None, :]
        first_diff = tl.minimum(first_diff, tl.min(tl.where(diff, cols[None, :], 2147483647), 1))
    empty = last_kept < 0
    n_t = tl.where(empty, n_all, (last_kept + BN) // BN)
    same = first_diff == 2147483647
    sel = tl.where(empty, 0, tl.where(same, n_all, first_diff // BN))
    dst = RowInfo + (b * N + rows.to(tl.int64)) * 2
    tl.store(dst, n_t, mask=r_ok)
    tl.store(dst + 1, sel, mask=r_ok)


def mask_tables(mask_u8: torch.Tensor, bn: int, fold: bool = True):
    """mask_u8 [B, N, S] uint8 (unit stride on S) -> (orpat int32 [B, S] or None, rowinfo int32 [B, N, 2], n_all).
    fold=False: reference pattern 'every key kept' (rows with any masked key select from their first masked key's tile)."""
    B, N, SK = mask_u8.shape
    assert mask_u8.stride(2) == 1
    dev = mask_u8.device
    n_all = triton.cdiv(SK, bn)
    rowinfo = torch.empty((B, N, 2), dtype=torch.int32, device=dev)
    if fold:
        orpat = torch.zeros((B, SK), dtype=torch.int32, device=dev)
        KB, RB = 128, 64
        launch(_LAUNCHES, _orpat_kernel, (triton.cdiv(SK, KB), triton.cdiv(N, RB), B),
               (mask_u8, orpat, mask_u8.stride(0), mask_u8.stride(1), N, SK), dict(KB=KB, RB=RB), num_warps=4)
    else:
        orpat = None
    launch(_LAUNCHES, _rowinfo_kernel, (triton.cdiv(N, 16), B),
           (mask_u8, orpat if fold else mask_u8, rowinfo, mask_u8.stride(0), mask_u8.stride(1), N, SK, n_all),
           dict(BN=bn, RB=16, KB=256, FOLD=fold), num_warps=4)
    return orpat, rowinfo, n_all


def census(rowinfo: torch.Tensor, n_all: int) -> torch.Tensor:
    """(rows on the mask-free path = equal to the reference pattern, rows using the per-key select), int64 device tensor."""
    sel = rowinfo[..., 1]
    return torch.stack([(sel >= n_all).sum(), (sel < n_all).sum()])
