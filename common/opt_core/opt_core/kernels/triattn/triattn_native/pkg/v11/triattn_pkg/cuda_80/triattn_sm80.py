"""Triangle-attention forward for A100-class GPUs (sm_80, cc 8.0) -- the `mma.sync` / `ldmatrix` / `cp.async` member of this kernel family
(device code csrc/triattn_sm80.cuh, launch csrc/launch_sm80.cuh, torch binding csrc/sm80_binding.cu).

    out = triangle_attention_sm80(q, k, v, bias, mask=None, scale=None)
    out, lse = triangle_attention_sm80(q, k, v, bias, mask=None, scale=None, return_lse=True)

q, k, v: [B, N, H, S, D] (or 4-D [N, H, S, D]) bf16 CUDA tensors, D in {16, 32, 64}, stride(-1) == 1 and the other strides
non-negative multiples of 8 elements -- contiguous tensors, ending-node transposed views and strided projection views all qualify and
are read in place (no copies, ever); N (pair rows) may differ from S (sequence); bias: [B, 1, H, S, S] (or [1, H, S, S]) fp32 | bf16 |
fp16 with any strides, shared by the N pair rows (staged once per call to an fp32 copy pre-divided by `scale`, exact to fp32 rounding);
mask: [B, N, 1, 1, S] bool, True = attend, any pattern per pair row (a pair row with no attendable key attends uniformly to all S keys,
as cuequivariance_ops_torch does); scale defaults to D ** -0.5.  Returns a new contiguous [B, N, H, S, D] bf16 tensor; with
return_lse=True also lse [B, N, H, S] fp32 = per query row log2(sum_k 2^(x_k)) with x = (scale * q.k + bias) * log2(e) the kernel's
log2-domain logit (a fully-masked row: log2(S)).  Deterministic run to run.  CUDA graphs -- capture-safe: the only module-level device
state a captured call refers to is this device's fix buffer (fix list + SAFE-tile census) and its 4-int mask census; both are created or
grown ONLY by eager calls, kept for the life of the process (a superseded fix buffer is never freed or resized, so a graph captured on it
keeps replaying into memory this module still owns) and grown geometrically (few generations, < ~2x the largest call's need in total);
everything else (staged bias, mask tables, out, lse) is allocated inside the call, i.e. owned by the graph's pool.  Capture therefore
works after one eager call at the same or a larger fix-list size (B*H*ceil(S/64)*ceil(N/4) CTA tiles) on that device -- what a warm-up
before capture provides; a capture that would have to create device state is refused by name (below) instead of recording its zero-fill
into the graph.  Inputs outside this contract raise `Unsupported` (a NotImplementedError) BEFORE any work: non-bf16 q/k/v (fp16
included), D not in {16, 32, 64}, S_q != S_kv, per-query / per-head masks, per-row biases, stride(-1) != 1 or strides that are not
multiples of 8 elements, a device that is not cc 8.0, fix-buffer growth requested while the current stream is capturing.  The extension
is JIT-built on first use (nvcc, sm_80 only, about a minute; cached under TORCH_EXTENSIONS_DIR).

Numerics: the family's max-free streaming softmax in exp2 units (integer row offsets, exact power-of-two renormalisations, rows seeded
once from their first finite logit), exact fp32 pair bias as the mma accumulator's initial value, row sums from the same bf16 P the P.V
mma consumes; CTA tiles the max-free pass cannot finish (a non-finite value or a zero row sum -- logit swings beyond ~2^100 within a row)
are recomputed by the exact running-max instantiation of the same tile routine from a per-call fix list: FALLBACKS["fix_tiles"] counts
them (0 on model tensors).  Masks: keys masked in every row of a batch element become -inf columns of the staged bias; every row streams
only its own live key tiles; a row whose attended keys are one contiguous run applies per-key masking only on the tiles holding the run's
ends; a row with holes ("ragged") applies 32-bit mask words on every tile (FALLBACKS["general_rows"] counts such rows, FALLBACKS[
"list_rowgroups"] the CTA row groups holding one); FALLBACKS["copy"] stays 0 (operands are never copied).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

__version__ = "1.0.0"
BIAS_STAGING = "fp32 exact"
NOTES = ("sm_80 member of the triangle-attention kernel family: mma.sync.m16n8k16 bf16->fp32, ldmatrix, cp.async multistage ring "
         "(<= 163 KB shared memory), max-free exp2 streaming softmax, exact fp32 pair-bias staging in accumulator-fragment order, per-row "
         "live key-tile ranges (ragged masks hot), fix-list SAFE pass with a visible census, int64 addressing, no operand copies.")

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None            # the compiled extension (triattn_sm80_ext)
_FIX = {}              # device -> [int32 fix buffers]: [0] census of SAFE-pass tiles (all calls), [1] reserved, [2] this call's count, [3:] tile list.
                       # A larger call appends a new generation (>= 2x the last); earlier ones are never freed (CUDA graphs captured on them keep
                       # replaying into them); generations are created by eager calls only (see _fix_buffer).
_CENSUS = {}           # device -> int32[4]: [0] unused, [1] ragged rows, [2] fully-masked rows, [3] row groups holding a ragged row (all calls)
_FIX_MIN = 1 << 18     # first generation: 1 MB = S <= ~2300 at N = S, H = 4 (either geometry)


class Unsupported(NotImplementedError):
    """Input outside what this kernel serves (typed refusal, raised before any work; callers route to another kernel)."""


class _Fallbacks(dict):
    """FALLBACKS with the device-side censuses folded in on every read (reading synchronises with the device)."""
    def _sync(self):
        tiles = 0
        for gens in _FIX.values():
            for b in gens:
                tiles += int(b[0].item())
        rag = uni = grp = 0
        for c in _CENSUS.values():
            v = c.tolist(); rag += int(v[1]); uni += int(v[2]); grp += int(v[3])
        dict.__setitem__(self, "fix_tiles", tiles); dict.__setitem__(self, "general_rows", rag)
        dict.__setitem__(self, "uniform_rows", uni); dict.__setitem__(self, "list_rowgroups", grp)
    def __getitem__(self, k): self._sync(); return dict.__getitem__(self, k)
    def get(self, k, d=None): self._sync(); return dict.get(self, k, d)
    def items(self): self._sync(); return dict.items(self)
    def values(self): self._sync(); return dict.values(self)
    def keys(self): self._sync(); return dict.keys(self)
    def __iter__(self): self._sync(); return dict.__iter__(self)
    def __repr__(self): self._sync(); return dict.__repr__(self)
    def copy(self): self._sync(); return dict(dict.items(self))


# Named slow paths (counted, never silent): "fix_tiles" CTA tiles recomputed by the exact (SAFE) pass; "general_rows" pair rows whose
# mask has holes (per-tile mask words in the hot pass); "list_rowgroups" CTA row groups holding such a row; "uniform_rows" fully-masked
# pair rows (uniform mean of v); "copy" operand copies (always 0: non-conforming layouts are refused by name instead).
FALLBACKS = _Fallbacks(copy=0, fix_tiles=0, general_rows=0, uniform_rows=0, list_rowgroups=0)


def fix_tiles() -> int:
    """CTA tiles recomputed by the SAFE pass since import (synchronises with the device)."""
    return FALLBACKS["fix_tiles"]


def _dims_wanted():
    env = os.environ.get("TRIATTN_SM80_DIMS", "")          # development: build a subset, e.g. "32"
    return sorted({int(x) for x in env.split(",") if x.strip()}) if env else [16, 32, 64]


def _build(verbose: bool = False):
    """The compiled extension (JIT through torch.utils.cpp_extension.load; sm_80 code only).  Cached in-process and on disk."""
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load
    dims = _dims_wanted()
    csrc = os.path.join(_HERE, "csrc")
    sources = [os.path.join(csrc, "sm80_binding.cu")] + [os.path.join(csrc, f"inst_d{d}.cu") for d in dims]
    flags = ["-O3", "-std=c++17", "-gencode", "arch=compute_80,code=sm_80", "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas", "-v", "-DNDEBUG"]
    flags += [f"-DTS_HAVE_D{d}" for d in dims]
    extra = os.environ.get("TRIATTN_SM80_NVCC_FLAGS", "")    # development A/B only (e.g. "-DTS_ST32=2 -DTS_R32=3"); leave unset
    if extra:
        flags += extra.split()
    name = "triattn_sm80_ext" if (dims == [16, 32, 64] and not extra) else "triattn_sm80_ext_" + "_".join(map(str, dims)) + (f"_{abs(hash(extra)) % 100000}" if extra else "")
    arch_prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"                       # cpp_extension otherwise adds -gencode for the visible device (+PTX)
    try:
        _EXT = load(name=name, sources=sources, extra_include_paths=[csrc], extra_cuda_cflags=flags,
                    extra_cflags=["-O3", "-std=c++17"] + [f"-DTS_HAVE_D{d}" for d in dims], verbose=verbose)
    finally:
        if arch_prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_prev
    return _EXT


def _capturing(device) -> bool:
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _fix_buffer(device, n: int) -> "torch.Tensor":
    """This device's fix buffer with room for n int32: the newest generation.  A too-small newest generation is superseded by one of
    max(n, 2 x its size) elements and is never freed or resized -- a CUDA graph captured on it keeps writing its fix list and census there and
    FALLBACKS sums the census over all generations -- so a process holds at most ~log2(n_max / 2^18) + 2 generations, < ~2 x 4 n_max bytes in
    all (N = S = 4096, H = 4: 3 generations, ~7 MB).  A generation is zero-filled device memory: created while the current stream is being
    captured, the fill would become part of the graph (every replay would wipe the census, and the buffer would be undefined until the first
    replay), so growth during capture is refused by name -- one eager call at this fix-list size or a larger one makes the generation (torch's
    own capture protocol, warm-up before capture, does exactly that).  The 4-int mask census is created with the first generation."""
    gens = _FIX.setdefault(device, [])
    if not gens or gens[-1].numel() < n:
        if _capturing(device):
            raise Unsupported(f"fix-buffer growth to {n} int32 (largest generation {gens[-1].numel() if gens else 0}) requested while the current "
                              f"stream is capturing a CUDA graph: make one eager (non-captured) call at this shape or a larger one on this device first")
        gens.append(torch.zeros(max(n, _FIX_MIN, 2 * gens[-1].numel() if gens else 0), dtype=torch.int32, device=device))
        _census_buffer(device)
    return gens[-1]


def _census_buffer(device) -> "torch.Tensor":
    c = _CENSUS.get(device)
    if c is None:
        if _capturing(device):
            raise Unsupported("the mask census buffer of this device does not exist yet and the current stream is capturing a CUDA graph: "
                              "make one eager (non-captured) call on this device first")
        _CENSUS[device] = c = torch.zeros(4, dtype=torch.int32, device=device)
    return c


def _require_cc80(device) -> None:
    cc = tuple(torch.cuda.get_device_capability(device))
    if cc != (8, 0):
        raise Unsupported(f"device cc {cc[0]}.{cc[1]}: this member is built and measured for cc 8.0 (A100-class) only")


def _layout_problem(name: str, t: torch.Tensor) -> Optional[str]:
    if t.stride(-1) != 1:
        return f"{name}.stride(-1) = {t.stride(-1)} (must be 1)"
    for d, s in enumerate(t.stride()[:-1]):
        if int(s) % 8 != 0 or int(s) < 0:
            return f"{name}.stride({d}) = {int(s)} (must be a non-negative multiple of 8 elements)"
    if t.data_ptr() % 16 != 0:
        return f"{name} base address not 16-byte aligned"
    return None


def stage_mask(mask: torch.Tensor, B: int, N: int, S: int, R: int, device):
    """[B,N,1,1,S] mask (True = attend) -> (keyany [B,ceil64(S)] uint8, rows [B*N,4] int32, maskw int32): keyany = keys some row attends
    (folded into the staged bias as -inf columns); rows = per pair row {a, e, kind} (live key interval / kind); maskw = 32-key mask words."""
    if mask.dim() == 5 and (mask.shape[2] != 1 or mask.shape[3] != 1):
        raise Unsupported(f"mask shape {tuple(mask.shape)}: per-head / per-query masks are not served (expect [B,N,1,1,S])")
    if mask.numel() != B * N * S:
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match B={B}, N={N}, S={S}")
    _require_cc80(mask.device)
    census = _census_buffer(device)                                # before any work (refused by name during a first-ever capture)
    m = mask.reshape(B, N, S)
    if m.dtype == torch.bool:
        m = m.contiguous().view(torch.uint8)
    elif m.dtype != torch.uint8:
        m = (m != 0).contiguous().view(torch.uint8)
    m = m.contiguous()
    keyany, rows, maskw = _build().stage_mask(m, census, int(R))
    return keyany, rows, maskw


def stage_bias(bias: torch.Tensor, scale: float, keyany: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B,1,H,S,S] -> fp32 [B,H,ceil128(S),ceil64(S)] = bias / scale in accumulator-fragment order (see stage_bias_kernel), -inf on keys
    >= S and on keys no row attends (keyany from stage_mask).  Reusable across calls that share bias, scale AND mask (pass bias_staged=)."""
    _require_cc80(bias.device)
    return _build().stage_bias(bias[:, 0], float(scale), keyany)


def geometry(D=32, small=False):
    """The built geometry for head dim D: the large-S instantiation, or (small=True) the small-S one (64-query tiles) the forward uses when
    S <= small_max for any layout, or S <= small_max_rows when the q/k/v position strides are <= 512 elements (not transposed views)."""
    r, qg, st, warps, bm, smem_hot, smem_safe, rw, small_max, small_max_rows = _build().geometry(int(D), bool(small))
    return dict(D=int(D), R=r, RW=rw, QG=qg, STAGES=st, warps=warps, threads=32 * warps, BM=bm, BN=64, smem_hot=smem_hot, smem_safe=smem_safe,
                small=bool(small), small_max=small_max, small_max_rows=small_max_rows)


def triangle_attention_sm80(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                            mask: Optional[torch.Tensor] = None, scale: Optional[float] = None, return_lse: bool = False, *,
                            bias_staged: Optional[torch.Tensor] = None, dbg: int = 0):
    squeeze = q.dim() == 4                                         # [N,H,S,D] operands, [1,H,S,S] bias, [N,1,1,S] mask
    if squeeze:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        bias = bias.reshape(1, *bias.shape[-4:]) if bias.dim() >= 4 else bias
        if mask is not None:
            mask = mask.reshape(1, *mask.shape[-4:]) if mask.dim() >= 4 else mask.reshape(1, mask.shape[0], 1, 1, mask.shape[-1])
    if q.dim() != 5:
        raise Unsupported(f"q rank {q.dim()}: expect [B,N,H,S,D] or [N,H,S,D]")
    B, N, H, S, D = q.shape
    # ---- typed refusals, before any work ----
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise Unsupported(f"q/k/v dtype {q.dtype}/{k.dtype}/{v.dtype}: bf16 only (fp16 is not served by this member)")
    if D not in (16, 32, 64):
        raise Unsupported(f"D={D}: head dim must be 16, 32 or 64")
    if not (q.is_cuda and k.is_cuda and v.is_cuda and bias.is_cuda and (mask is None or mask.is_cuda)):
        raise Unsupported("CUDA tensors only")
    if k.dim() != 5 or v.dim() != 5 or k.shape[:3] != q.shape[:3] or k.shape[4] != D or v.shape != k.shape:
        raise Unsupported(f"q/k/v shapes {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}: expect identical [B,N,H,S,D]")
    if k.shape[3] != S:
        raise Unsupported(f"S_q={S} != S_kv={k.shape[3]}: only square (S_q == S_kv) triangle attention is served")
    if bias.dim() != 5 or bias.shape[0] != B or bias.shape[2] != H or bias.shape[3] != S or bias.shape[4] != S:
        raise Unsupported(f"bias shape {tuple(bias.shape)}: expect [B,1,H,S,S] (one pair bias shared by the N rows)")
    if bias.shape[1] != 1:
        raise Unsupported(f"bias shape {tuple(bias.shape)}: per-row biases (dim 1 = {bias.shape[1]}) are not triangle attention; expect [B,1,H,S,S]")
    if bias.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise Unsupported(f"bias dtype {bias.dtype}: fp32 | bf16 | fp16")
    if mask is not None:
        if mask.dim() != 5 or mask.shape[0] != B or mask.shape[1] != N or mask.shape[4] != S:
            raise Unsupported(f"mask shape {tuple(mask.shape)}: expect [B,N,1,1,S] (one key mask per pair row)")
        if mask.shape[2] != 1 or mask.shape[3] != 1:
            raise Unsupported(f"mask shape {tuple(mask.shape)}: per-head / per-query masks are not served (expect [B,N,1,1,S])")
    for name, t in (("q", q), ("k", k), ("v", v)):
        why = _layout_problem(name, t)
        if why:
            raise Unsupported(f"operand layout: {why}")
    dev = q.device
    _require_cc80(dev)
    if D not in _dims_wanted():
        raise Unsupported(f"D={D}: not in this build's head dims {_dims_wanted()} (TRIATTN_SM80_DIMS)")
    # ---- work ----
    ext = _build()
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    geo = ext.geometry(D, False)
    dense = all(t.stride(3) <= 512 for t in (q, k, v))
    if S <= int(geo[8]) or (dense and S <= int(geo[9])):
        geo = ext.geometry(D, True)                                # the small-S geometry fwd will use (row-group census granularity)
    R = int(geo[0])
    fix = _fix_buffer(dev, int(ext.fix_elems(B, N, H, S, D)))       # module state first: its one refusal (growth during capture) precedes any work
    keyany = rows = maskw = None
    if mask is not None:
        keyany, rows, maskw = stage_mask(mask, B, N, S, R, dev)
    if bias_staged is None:
        bias_staged = ext.stage_bias(bias[:, 0], float(scale), keyany)
    out = torch.empty(B, N, H, S, D, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(B, N, H, S, dtype=torch.float32, device=dev) if return_lse else None
    ext.fwd(q, k, v, bias_staged, rows, maskw, float(scale), out, fix, lse, int(dbg))
    if squeeze:
        out = out[0]
        lse = lse[0] if lse is not None else None
    return (out, lse) if return_lse else out


triangle_attention = triangle_attention_sm80        # family-style alias


def reference(q, k, v, bias, mask=None, scale=None, dtype=torch.float32):
    """Plain PyTorch reference with the same fully-masked-row convention (cuda_c/numerics_suite.reference)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "cuda_c"))
    from numerics_suite import reference as ref
    return ref(q, k, v, bias, mask, scale, dtype=dtype)
