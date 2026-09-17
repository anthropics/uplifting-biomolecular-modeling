"""Bench-harness plug-ins of the Triton / Gluon kernels (contract: fn(q, k, v, bias, mask=None, scale=None) -> out [B, N, H, S, D]).

    --kernel tri=candidate:tri[:bf16bias]   the entry of record: k12 (Gluon, persistent, sm_90) for square bf16/fp16 pair attention
                                    with D in {16, 32} and S >= 256 (16-bit bias) / 320 (fp32 bias); k10 (Triton, any CUDA GPU) for
                                    everything else it supports (D in {16, 32, 64, 128}, S_q != S_kv, small S, non-Hopper GPUs).
                                    With a 16-bit bias whose S is a multiple of 64 an unmasked call is ONE kernel launch (the bias is
                                    read in place); an fp32 bias is staged to 16-bit tiles first (exact for bf16-representable values)
    --kernel k10=candidate:k10      the Triton kernel alone
    --kernel k12=candidate:k12      the Gluon kernel alone (raises Unsupported outside its domain)

Masks: any [B,N,1,1,S] bool key mask (padding prefixes per row, interior zeros, fully-masked rows -> uniform average of v like
cuEquivariance); the batch OR pattern is folded into the staged bias so rows equal to it run mask-free and skip dead key tiles
(masking.py).  Strided q/k/v views (engine projection layouts, column-attention transposes, 4-D operands unsqueezed by the caller) are read in place: k12 through rank-2 TMA views, k10 by pointer arithmetic; a copy is a named fallback (INFO['q_copied'...]).  Inputs outside
the supported domain raise `Unsupported` (a NotImplementedError) naming the reason -- nothing degrades silently.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triattn.k10 import triattn_k10  # noqa: E402
from triattn.k11 import Unsupported  # noqa: E402
from triattn.k12 import triattn_k12  # noqa: E402

BIAS_STAGING = "fp32 in; k12: 16-bit tiles (exact for bf16-representable bias), k10: 16-bit tiles only when lossless else fp32"
K10_CONFIG = {16: dict(ROWS=4, BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2, PIPE=1),
              32: dict(ROWS=4, BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2, PIPE=1),
              64: dict(ROWS=2, BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2, PIPE=1),
              128: dict(ROWS=1, BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=2, PIPE=0)}   # per head_dim: shared-memory / register budget
K12_CONFIG = dict(ROWS=3, BLOCK_N=64, STAGES=3)
K12_MIN_S_16 = 256        # k12 from here when the bias arrives in the 16-bit dtype (no staging launch: one kernel per unmasked call)
K12_MIN_S_32 = 320        # ... and from here when an fp32 bias has to be staged first (below, k10's lighter CTAs win)
K12_DIMS = (16, 32)       # K12_CONFIG's shared-memory / register budget (D=64/128 go to k10)
K10_DIMS = (16, 32, 64, 128)


def _check(q, k, v):
    if q.dim() != 5 or k.dim() != 5 or v.dim() != 5:
        raise Unsupported(f"unsupported rank: q/k/v must be [B,N,H,S,D], got {q.dim()}/{k.dim()}/{v.dim()}")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise Unsupported(f"unsupported dtype {q.dtype} (bf16 / fp16 only)")
    if q.shape[-1] not in K10_DIMS:
        raise Unsupported(f"unsupported head_dim {q.shape[-1]} (supported: {K10_DIMS})")
    if not q.is_cuda:
        raise Unsupported("CUDA tensors only")


def k10(q, k, v, bias, mask=None, scale=None):
    _check(q, k, v)
    return triattn_k10(q, k, v, bias, mask, scale, config=K10_CONFIG[q.shape[-1]])


def k12(q, k, v, bias, mask=None, scale=None):
    _check(q, k, v)
    return triattn_k12(q, k, v, bias, mask, scale, config=K12_CONFIG)


def tri(q, k, v, bias, mask=None, scale=None):
    _check(q, k, v)
    S_q, S_k, D = q.shape[-2], k.shape[-2], q.shape[-1]
    min_s = K12_MIN_S_16 if bias.dtype == q.dtype else K12_MIN_S_32
    if S_q == S_k and S_q >= min_s and D in K12_DIMS and _is_sm90(q.device):
        return triattn_k12(q, k, v, bias, mask, scale, config=K12_CONFIG)
    return triattn_k10(q, k, v, bias, mask, scale, config=K10_CONFIG[q.shape[-1]])


_SM90 = {}


def _is_sm90(dev) -> bool:
    r = _SM90.get(dev)
    if r is None:
        r = torch.cuda.get_device_capability(dev) == (9, 0)
        _SM90[dev] = r
    return r
