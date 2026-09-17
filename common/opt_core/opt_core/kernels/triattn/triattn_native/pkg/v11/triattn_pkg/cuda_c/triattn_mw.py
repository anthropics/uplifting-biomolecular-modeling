"""Triangle-attention forward on H100 — many-warp `mma.sync` kernel (csrc/triattn_mw.cu).

    out = triangle_attention(q, k, v, bias, mask=None, scale=None)

q, k, v: [B, N, H, S, D=32] (or 4-D [N, H, S, D]) bf16 CUDA tensors with stride(-1) == 1 and the other strides non-negative
multiples of 8 elements -- contiguous tensors, ending-node transposed views and the engines' strided projection views all
qualify and are read in place (K/V through TMA descriptors, Q by plain loads); anything else is copied once, counted in FALLBACKS["copy"] (a named
fallback, never silent); N (pair rows) may differ from S (sequence); S <= 4096 (larger raises Unsupported); bias: [B, 1, H, S, S] (or [1, H, S, S]) fp32 |
bf16 | fp16, shared by the N pair rows (staged once per call to an fp32 copy pre-divided by `scale`, exact to fp32 rounding);
mask: [B, N, 1, 1, S] bool, True = attend, any pattern (a pair row with no attendable key attends uniformly to all S keys, as
cuEquivariance does); scale defaults to D**-0.5.  Returns a new contiguous [B, N, H, S, D] bf16 tensor.  Deterministic; safe
under CUDA-graph capture after the first (building) call.  The extension is JIT-built on first use (nvcc, sm_90a, ~1 min).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GEOM = "3x4"   # 3 pair rows (12 warps), 4-deep ring: 224 KB of shared memory, one CTA per SM
_EXTS = {}             # geometry -> compiled extension
# Named fallbacks / slow paths (counted, never silent):
#   "copy"      q/k/v operands copied to a TMA-legal layout (stays 0 on contiguous, transposed and engine-strided inputs);
#   "fix_tiles" CTA tiles (3 pair rows x 128 queries) recomputed by the max-tracking fix pass because a row overflowed the max-free
#               softmax (logit swings beyond ~2^100 within a row; never seen on model data);
#   "list_rowgroups" row groups (3 pair rows, all heads and q tiles) computed by the general list pass because a row's mask differs
#               from the batch OR of the masks (device counters; reading FALLBACKS synchronises).
class Unsupported(NotImplementedError):
    """Input outside what this kernel serves (typed refusal; callers route to another kernel)."""


class _Fallbacks(dict):
    """FALLBACKS with the device-side censuses folded in on every read (reading synchronises with the device)."""
    def _sync(self):
        tiles = rows = 0
        for b in _FIX.values():
            c = b[:2].tolist(); tiles += int(c[0]); rows += int(c[1])
        dict.__setitem__(self, "fix_tiles", tiles); dict.__setitem__(self, "list_rowgroups", rows)
    def __getitem__(self, k): self._sync(); return dict.__getitem__(self, k)
    def get(self, k, d=None): self._sync(); return dict.get(self, k, d)
    def items(self): self._sync(); return dict.items(self)
    def values(self): self._sync(); return dict.values(self)
    def keys(self): self._sync(); return dict.keys(self)
    def __iter__(self): self._sync(); return dict.__iter__(self)
    def __repr__(self): self._sync(); return dict.__repr__(self)
    def copy(self): self._sync(); return dict(dict.items(self))


FALLBACKS = _Fallbacks(copy=0, fix_tiles=0, list_rowgroups=0)
_FIX = {}   # device -> int32 [3 + 3 * max CTA tiles]: [0] census of fixed tiles, [1] census of list-pass row groups, [2] this call's count, [3:] tile list


def fix_tiles() -> int:
    """CTA tiles recomputed by the fix pass since import (synchronises with the device)."""
    return FALLBACKS["fix_tiles"]


def _fix_buffer(device, n: int) -> "torch.Tensor":
    b = _FIX.get(device)
    if b is None or b.numel() < n:
        nb = torch.zeros(max(n, 1 << 16), dtype=torch.int32, device=device)
        if b is not None:
            nb[0] = b[0]
        _FIX[device] = b = nb
    return b


def _build(verbose: bool = False, geom: str = ""):
    """Compiled extension for CTA geometry `geom` = "RxST" or "RxSTmB": R pair rows per CTA tile (4 warps each), ring depth ST,
    B CTAs per SM (launch bound); "" = MW_GEOM from the environment, else the default 3x4.  One extension per geometry, cached."""
    geom = geom or os.environ.get("MW_GEOM", "") or DEFAULT_GEOM
    ext = _EXTS.get(geom)
    if ext is None:
        from torch.utils.cpp_extension import load
        r_, rest = geom.split("x"); st_, _, minb = rest.partition("m")
        flags = ["-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a", "-lineinfo", "-Xptxas", "-v",
                 "--expt-relaxed-constexpr", "-DNDEBUG", f"-DMW_R={int(r_)}", f"-DMW_ST={int(st_)}", f"-DMW_MINB={int(minb or 1)}"]
        timeline = bool(os.environ.get("MW_TIMELINE"))            # dev instrumentation (timeline_mw.py); never in timed builds
        if timeline: flags.append("-DMW_TIMELINE")
        ext = _EXTS[geom] = load(name=f"triattn_mw_ext_g{geom}" + ("_tl" if timeline else ""), sources=[os.path.join(_HERE, "csrc", "triattn_mw.cu")],
                                 extra_include_paths=[os.path.join(_HERE, "csrc")], extra_cuda_cflags=flags, extra_cflags=["-O3", "-std=c++17"],
                                 verbose=verbose)
    return ext


def _tma_ok(t: torch.Tensor) -> bool:      # K, V: TMA descriptors -- d stride 1, other strides multiples of 8 elements, 16-byte base
    return t.stride(-1) == 1 and all(int(s) % 8 == 0 and int(s) >= 0 for s in t.stride()[:-1]) and t.data_ptr() % 16 == 0


def _ldq_ok(t: torch.Tensor) -> bool:      # Q: 4-byte loads -- d stride 1, other strides even, 4-byte base
    return t.stride(-1) == 1 and all(int(s) % 2 == 0 and int(s) >= 0 for s in t.stride()[:-1]) and t.data_ptr() % 4 == 0


def stage_mask(mask: torch.Tensor, B: int, N: int, S: int, ext=None):
    """[B,N,1,1,S] mask (True = attend) -> (mask_u8 [B,N,S], keyany [B,ceil64(S)] uint8, rowkind [B,N] uint8, irregular row-group
    list, ktend [B] int32): keyany = keys some row attends (folded into the staged bias as -inf columns); rowkind 1 = the row's mask
    differs from keyany (recomputed by the general kernel); ktend = key tiles up to the batch's last attended key."""
    if mask.dim() == 5 and (mask.shape[2] != 1 or mask.shape[3] != 1):
        raise Unsupported(f"mask shape {tuple(mask.shape)}: per-head / per-query masks are not served (expect [B,N,1,1,S])")
    if mask.numel() != B * N * S:
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match B={B}, N={N}, S={S}")
    m = mask.reshape(B, N, S)
    if m.dtype == torch.bool:
        m = m.contiguous().view(torch.uint8)
    elif m.dtype != torch.uint8:
        m = (m != 0).view(torch.uint8)
    m = m.contiguous()
    keyany, rowkind, irr, ktend = (ext or _build()).stage_mask(m)
    return m, keyany, rowkind, irr, ktend


def stage_bias(bias: torch.Tensor, scale: float, keyany: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B,1,H,S,S] -> fp32 [B,H,ceil128(S),ceil64(S)] / scale, permuted so one 16-byte shared-memory load is one mma C fragment
    (see stage_bias_kernel), with -inf on keys >= S and on keys no row attends (keyany from stage_mask).  Reusable across calls
    that share bias, scale AND mask (pass as bias_staged=)."""
    return _build().stage_bias(bias[:, 0], float(scale), keyany)


def triangle_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                       mask: Optional[torch.Tensor] = None, scale: Optional[float] = None, *,
                       bias_staged: Optional[torch.Tensor] = None, dbg: int = 0, tl: Optional[torch.Tensor] = None, geom: str = "") -> torch.Tensor:
    ext = _build(geom=geom)
    squeeze = q.dim() == 4                                         # [N,H,S,D] operands, [1,H,S,S] bias, [N,1,1,S] mask
    if squeeze:
        q, k, v, bias = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), bias.reshape(1, *bias.shape[-4:])
        if mask is not None: mask = mask.reshape(1, *mask.shape[-4:]) if mask.dim() >= 4 else mask.reshape(1, mask.shape[0], 1, 1, mask.shape[-1])
    B, N, H, S, D = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"q/k/v shape mismatch {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise Unsupported(f"dtype {q.dtype}: bf16 only")
    if D != 32:
        raise Unsupported(f"D={D}: D=32 only")
    if S > 4096:
        raise Unsupported(f"S={S}: S <= 4096 only")
    if not q.is_cuda:
        raise Unsupported("CUDA tensors only")
    if tuple(bias.shape) != (B, 1, H, S, S):
        raise ValueError(f"bias shape {tuple(bias.shape)} != {(B, 1, H, S, S)}")
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    ops = []
    for t, ok in ((q, _ldq_ok(q)), (k, _tma_ok(k)), (v, _tma_ok(v))):
        if not ok:
            FALLBACKS["copy"] += 1; t = t.contiguous()
        ops.append(t)
    q_, k_, v_ = ops
    mask_u8 = keyany = rowkind = irr = ktend = None
    if mask is not None:
        mask_u8, keyany, rowkind, irr, ktend = stage_mask(mask, B, N, S, ext)
    if bias_staged is None:
        bias_staged = ext.stage_bias(bias[:, 0], float(scale), keyany)
    out = torch.empty(B, N, H, S, D, dtype=torch.bfloat16, device=q.device)
    fix = _fix_buffer(q.device, ext.fix_elems(B, N, H, S))
    ext.fwd(q_, k_, v_, bias_staged, mask_u8, rowkind, irr, ktend, float(scale), out, fix, int(dbg), tl)
    return out[0] if squeeze else out


def reference(q, k, v, bias, mask=None, scale=None, dtype=torch.float32):
    """Plain PyTorch reference with the same fully-masked-row convention (numerics_suite.reference)."""
    from numerics_suite import reference as ref
    return ref(q, k, v, bias, mask, scale, dtype=dtype)

