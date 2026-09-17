"""Python entry point for the sm_90a triangle-attention forward kernel (csrc/).

    out = triangle_attention(q, k, v, bias, mask=None, scale=None)

q, k, v: [B, N, H, S, D] bf16 CUDA tensors (any strides with stride(-1) == 1 and the other strides multiples of 8 elements —
ending-node transposed views qualify; anything else is copied once); bias: [B, 1, H, S, S] fp32 or bf16 (shared by the N
rows); mask: [B, N, 1, 1, S] bool, True = attend (interior zeros allowed; a row with no attendable key attends uniformly to
all S keys, as cuEquivariance does); scale defaults to D**-0.5. Returns a contiguous [B, N, H, S, D] bf16 tensor.
The extension is JIT-built on first use (torch.utils.cpp_extension, nvcc -arch=sm_90a, CUTLASS headers from $CUTLASS_PATH).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None


def _build(verbose: bool = False):
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load
    cutlass = os.environ.get("CUTLASS_PATH", "/opt/cutlass")
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
             "-gencode", "arch=compute_90a,code=sm_90a", "-DNDEBUG", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "--ftemplate-backtrace-limit=0", "-lineinfo",
             "-Xcompiler", "-Wno-psabi", "-diag-suppress", "177,550", "-Xptxas", "-v"]
    if os.environ.get("TRIATTN_MICROBENCH", "0") == "1":
        flags.append("-DTRIATTN_MICROBENCH")           # also build the F4 wgmma-throughput microkernel (binding `wgmma_bench`)
    _EXT = load(name="triattn_sm90_ext", sources=[os.path.join(_HERE, "csrc", "triattn_binding.cu")],
                extra_include_paths=[os.path.join(_HERE, "csrc"), os.path.join(cutlass, "include")],
                extra_cuda_cflags=flags, extra_cflags=["-O3", "-std=c++17"], verbose=verbose)
    return _EXT


def _tma_ok(t: torch.Tensor) -> bool:
    return t.stride(-1) == 1 and all(int(s) % 8 == 0 for s in t.stride()[:-1]) and t.data_ptr() % 16 == 0


def prepare_bias(bias: torch.Tensor, S: int) -> torch.Tensor:
    """[B,1,H,S,S] fp32|bf16 -> [B,H,S,S8] bf16 with S8 = S rounded up to 8 (TMA needs 16-byte row strides)."""
    B, _, H = bias.shape[0], bias.shape[1], bias.shape[2]
    b = bias[:, 0]
    S8 = (S + 7) // 8 * 8
    if b.dtype == torch.bfloat16 and S8 == S and _tma_ok(b):
        return b
    out = torch.empty(B, H, S, S8, dtype=torch.bfloat16, device=bias.device)
    out[..., :S].copy_(b)
    if S8 != S:
        out[..., S:].zero_()
    return out


BIAS_STAGING = "bf16-staged (lossy for fp32 bias)"   # the kernel reads the pair bias as bf16; an fp32 bias is converted once per call
NOTES = "sm_90a warp-specialized: 1 TMA producer + 2 consumer warpgroups, 4 pair rows per CTA share each bias tile, [V|1] PV GEMM row sums, MUFU-section ping-pong"
__version__ = "r4-ring3-live-c6"

# (block_n, rows, cluster_q, stages, variant) of the variant of record; must be instantiated in csrc/triattn_binding.cu TRIATTN_VARIANTS
CONFIG = dict(block_n=64, rows=4, cluster_q=1, stages=12, variant=1)


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, mask: Optional[torch.Tensor] = None,
            scale: Optional[float] = None) -> torch.Tensor:
    """Harness plug-in entry point (bench/README contract): the variant of record."""
    return triangle_attention(q, k, v, bias, mask, scale, **CONFIG)


_COUNTERS = {}


def _device_counters(dev):
    """int64[8] per device, [0] = row-steps that took the exact-max path after the max-free seed (read via fallback_counts())."""
    key = torch.device(dev).index
    if key not in _COUNTERS:
        _COUNTERS[key] = torch.zeros(8, dtype=torch.int64, device=dev)
    return _COUNTERS[key]


def fallback_counts() -> dict:
    """Named fallback events so far (host-side copies + device-side exact-max row-steps; the latter syncs the device)."""
    d = dict(FALLBACK_COUNTS)
    d["maxfree_exact_rowsteps"] = int(sum(int(t[0].item()) for t in _COUNTERS.values()))
    return d


STREAM_VARIANTS = tuple(range(9, 40))          # chunk-streaming consumer: bias staged in MMA-fragment order
_FRAG_IDX = {}


def _frag_index(device, block_m: int):
    """Element order of one block_m x 64 bf16 bias tile as the streaming consumer reads it from shared memory:
    [key half c][warpgroup][warp][accumulator row mi][lane][8 elements] -> row m = 64 wg + 16 warp + lane/4 + 8 mi,
    key n = 32 c + 8 (e/2) + 2 (lane%4) + e%2 (the wgmma m64nN fp32 accumulator layout)."""
    key = (str(device), block_m)
    if key not in _FRAG_IDX:
        nwg = block_m // 64
        f = torch.arange(block_m * 64)
        e, lane, mi, warp, wg, c = f % 8, (f // 8) % 32, (f // 256) % 2, (f // 512) % 4, (f // 2048) % nwg, f // (2048 * nwg)
        m = wg * 64 + warp * 16 + lane // 4 + 8 * mi
        n = c * 32 + 8 * (e // 2) + 2 * (lane % 4) + (e % 2)
        _FRAG_IDX[key] = (m * 64 + n).to(device)
    return _FRAG_IDX[key]


def _acc_index(device, block_m: int):
    """Element order of one block_m x 64 tile in wgmma ACCUMULATOR order: [key half c][warpgroup][warp][lane][16 values v], value v of a
    thread = accumulator register v: e = v%2, mi = (v/2)%2, nj = v/4 -> row 64 wg + 16 warp + lane/4 + 8 mi, key 32 c + 8 nj + 2 (lane%4) + e."""
    key = ("acc", str(device), block_m)
    if key not in _FRAG_IDX:
        nwg = block_m // 64
        f = torch.arange(block_m * 64)
        v, lane, warp, wg, c = f % 16, (f // 16) % 32, (f // 512) % 4, (f // 2048) % nwg, f // (2048 * nwg)
        e, mi, nj = v % 2, (v // 2) % 2, v // 4
        m = wg * 64 + warp * 16 + lane // 4 + 8 * mi
        n = c * 32 + 8 * nj + 2 * (lane % 4) + e
        _FRAG_IDX[key] = (m * 64 + n).to(device)
    return _FRAG_IDX[key]


def stage_bias_acc32(bias16: torch.Tensor, S: int, scale: float, block_m: int = 128) -> torch.Tensor:
    """[B,1,H,S,S] or [B,H,S,>=S] -> fp32 [B*H, ceil(S/block_m), ceil(S/64), block_m*64] = bias/scale in accumulator order (the streaming
    consumer loads it straight into the S accumulator; the QK product accumulates on top); keys >= S as -inf."""
    b4 = bias16 if bias16.dim() == 4 else bias16[:, 0]
    B, H = b4.shape[:2]
    nq, nk = -(-S // block_m), -(-S // 64)
    buf = torch.zeros(B, H, nq * block_m, nk * 64, dtype=torch.float32, device=bias16.device)
    if nk * 64 > S:
        buf[..., S:] = float("-inf")
    buf[:, :, :S, :S] = b4[:, :, :S, :S].float() * (1.0 / scale)
    t = buf.view(B, H, nq, block_m, nk, 64).permute(0, 1, 2, 4, 3, 5).reshape(B * H, nq, nk, block_m * 64)
    return t.index_select(-1, _acc_index(bias16.device, block_m)).contiguous()


def stage_bias_for(cfg: dict, bias16: torch.Tensor, S: int, scale: float) -> torch.Tensor:
    """The staged bias tensor the streaming variant `cfg` (block_n, rows, cluster_q, stages, variant) consumes."""
    ext = _build()
    args = (cfg["block_n"], cfg["rows"], cfg["cluster_q"], cfg["stages"], cfg["variant"])
    bm = ext.block_m(*args)
    return stage_bias_acc32(bias16, S, scale, bm) if ext.bias_f32(*args) == 1 else stage_bias_frag(bias16, S, bm)


def stage_bias_frag(bias16: torch.Tensor, S: int, block_m: int = 128) -> torch.Tensor:
    """[B,1,H,S,S] or [B,H,S,>=S] bf16 -> [B*H, ceil(S/block_m), ceil(S/64), block_m*64] bf16: every (q-tile, k-tile) block in
    fragment order, keys >= S as -inf (excluded), query rows >= S as 0 (never stored)."""
    b4 = bias16 if bias16.dim() == 4 else bias16[:, 0]          # [B,H,S,>=S]
    B, H = b4.shape[:2]
    nq, nk = -(-S // block_m), -(-S // 64)
    buf = torch.zeros(B, H, nq * block_m, nk * 64, dtype=torch.bfloat16, device=bias16.device)
    if nk * 64 > S:
        buf[..., S:] = float("-inf")
    buf[:, :, :S, :S] = b4[:, :, :S, :S]
    t = buf.view(B, H, nq, block_m, nk, 64).permute(0, 1, 2, 4, 3, 5).reshape(B * H, nq, nk, block_m * 64)
    return t.index_select(-1, _frag_index(bias16.device, block_m)).contiguous()


class Unsupported(NotImplementedError):
    """Raised (before any work) for inputs this kernel does not serve, naming the reason; dispatchers route on it."""


# named fallback events (E1/C4): a copy made because an operand's strides are not TMA-legal, and fp32->bf16 bias re-staging
FALLBACK_COUNTS = {"qkv_contiguous_copy": 0, "bias_restaged": 0, "mask_u8_copy": 0}


def _check(q, k, v, bias, mask):
    if q.dim() != 5:
        raise Unsupported(f"triattn_cuda: q/k/v must be [B,N,H,S,D] (or 4-D [N,H,S,D]); got rank {q.dim()}")
    B, N, H, S, D = q.shape
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise Unsupported(f"triattn_cuda: dtype bf16 only (got {q.dtype})")
    if D != 32:
        raise Unsupported(f"triattn_cuda: head_dim 32 only (got {D})")
    if tuple(k.shape) != (B, N, H, S, D) or tuple(v.shape) != (B, N, H, S, D):
        raise Unsupported(f"triattn_cuda: k/v shape {tuple(k.shape)}/{tuple(v.shape)} != q shape {tuple(q.shape)} (keys S must equal queries S)")
    if bias.dim() != 5 or bias.shape[0] != B or bias.shape[1] != 1 or bias.shape[2] != H or bias.shape[3] != S or bias.shape[4] != S:
        raise Unsupported(f"triattn_cuda: bias must be [B,1,H,S,S] = {[B, 1, H, S, S]} shared by all N pair rows (got {list(bias.shape)})")
    if bias.dtype not in (torch.bfloat16, torch.float32, torch.float16):
        raise Unsupported(f"triattn_cuda: bias dtype {bias.dtype}")
    if mask is not None:
        if mask.dtype not in (torch.bool, torch.uint8):
            raise Unsupported(f"triattn_cuda: mask dtype bool/uint8 (got {mask.dtype})")
        if mask.numel() != B * N * S or mask.shape[-1] != S:
            raise Unsupported(f"triattn_cuda: mask must be [B,N,1,1,S] (got {list(mask.shape)})")
    if S > 4096:
        raise Unsupported(f"triattn_cuda: S <= 4096 (got {S})")
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise Unsupported("triattn_cuda: sm_90a (H100/H200) only")


def triangle_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                       mask: Optional[torch.Tensor] = None, scale: Optional[float] = None, *,
                       block_n: int = 64, rows: int = 4, cluster_q: int = 1, stages: int = 12, variant: int = 1, trace: Optional[torch.Tensor] = None,
                       bias_frag: Optional[torch.Tensor] = None) -> torch.Tensor:
    squeeze = False
    if q.dim() == 4:                                   # [N,H,S,D] operands (no batch dim): OF3 trunk
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        if bias.dim() == 4: bias = bias.unsqueeze(0)
        if mask is not None and mask.dim() == 4: mask = mask.unsqueeze(0)
        squeeze = True
    _check(q, k, v, bias, mask)
    B, N, H, S, D = q.shape
    ext = _build()
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    qkv = []
    for t in (q, k, v):
        if _tma_ok(t):
            qkv.append(t)
        else:                                          # named fallback: strides not TMA-legal (inner stride != 1 or not 16-byte multiples)
            FALLBACK_COUNTS["qkv_contiguous_copy"] += 1
            qkv.append(t.contiguous())
    bias16 = prepare_bias(bias, S)
    if bias16 is not bias and bias16.data_ptr() != bias.data_ptr():
        FALLBACK_COUNTS["bias_restaged"] += 1          # fp32/fp16 -> bf16 conversion, or a non-TMA-legal bf16 view copied (one O(H*S*S) pass)
    mask_u8 = None
    if mask is not None:
        m = mask.reshape(B, N, S)
        m = m.view(torch.uint8) if m.dtype == torch.bool else m      # same bytes, no conversion kernel
        if not m.is_contiguous():
            FALLBACK_COUNTS["mask_u8_copy"] += 1
            m = m.contiguous()
        mask_u8 = m
    out = torch.empty(B, N, H, S, D, dtype=q.dtype, device=q.device)
    counters = _device_counters(q.device)
    if variant in STREAM_VARIANTS and bias_frag is None:
        bm = ext.block_m(block_n, rows, cluster_q, stages, variant)
        bias_frag = (stage_bias_acc32(bias16, S, float(scale), bm) if ext.bias_f32(block_n, rows, cluster_q, stages, variant) == 1
                     else stage_bias_frag(bias16, S, bm))
    ext.fwd(qkv[0], qkv[1], qkv[2], bias16, mask_u8, float(scale), out, block_n, rows, cluster_q, stages, variant, trace, counters, bias_frag)
    return out[0] if squeeze else out
