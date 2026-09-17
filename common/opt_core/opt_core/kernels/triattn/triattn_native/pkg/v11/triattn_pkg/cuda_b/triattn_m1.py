"""Python entry point for the M1 kernel (csrc/m1/, contract in README.md): three consumer warpgroups (one pair row each), 64x32 S
chunks streamed max-free, pair bias / scale staged once per call in fp32 MMA-fragment order and loaded into the accumulators from
shared memory.

    out = triangle_attention_m1(q, k, v, bias, mask=None, scale=None, *, flags=0)

q, k, v: [B, N, H, S, 32] bf16 (stride(-1) == 1, other strides multiples of 8 elements); bias [B, 1, H, S, S] fp32 | bf16 | fp16;
any S; mask [B, N, 1, 1, S] bool or None. Keys masked in every row of a batch element are folded into -inf bias columns; a CTA streams
the union of its rows' live key-column ranges and each row consumes only the tiles of its own range (K/V loads, QK/PV work), applying its
mask words only on the tiles that can hold a boundary of its range (a ragged row: on all); fully-masked rows return the uniform mean of v. CTA tiles the max-free hot pass cannot finish (sum/output outside the representable window, or a row
with no finite logit in its first chunk) are recomputed by the exact SAFE instantiation via a fix list: census FALLBACKS["fix_tiles"].
flags: 16 = clock trace, others = ablations (see the kernel header).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None
FLAGS = [0]


def _all_flags():
    env = os.environ.get("TRIATTN_M1_FLAGS")           # dev override: "0;16;2"
    hot = [int(x) for x in env.split(";")] if env else list(FLAGS)
    base = {f & ~1024 for f in hot}
    return sorted(base | {(f & ~256) | 1024 for f in base})    # every hot instantiation gets its SAFE (fix-list) partner (never clustered)


def _generate_sources(build_dir: str):
    inst = os.path.join(build_dir, "inst_m1")
    os.makedirs(inst, exist_ok=True)
    srcs, decls, rows = [], [], []
    for fl in _all_flags():
        body = (f'#include "launch_m1.cuh"\nnamespace triattn_m1 {{\n'
                f"void run_m1_{fl}(Args const& a) {{ launch_m1<Traits<{fl}>>(a); }}\n"
                f"int64_t smem_m1_{fl}() {{ return smem_bytes_m1<Traits<{fl}>>(); }}\n}}\n")
        path = os.path.join(inst, f"m1_{fl}.cu")
        if not os.path.exists(path) or open(path).read() != body:
            open(path, "w").write(body)
        srcs.append(path)
        decls.append(f"void run_m1_{fl}(Args const&); int64_t smem_m1_{fl}();")
        rows.append(f"    t[{fl}] = Entry{{&run_m1_{fl}, smem_m1_{fl}()}};")
    table = "\n".join(decls) + "\nstatic std::map<int, Entry> make_table() {\n    std::map<int, Entry> t;\n" + "\n".join(rows) + "\n    return t;\n}\n"
    tpath = os.path.join(inst, "table_m1.inc")
    if not os.path.exists(tpath) or open(tpath).read() != table:
        open(tpath, "w").write(table)
    return srcs, inst


PTXAS_134 = "/usr/local/cuda-13.4/bin/ptxas"   # assembler of record when present (identical output bytes, faster SASS at large S); TRIATTN_PTXAS=image disables


def _toolkit_with_ptxas(ptxas: str) -> str:
    """A CUDA_HOME shim: the image toolkit (nvcc, cicc, headers, cudart) with only bin/ptxas replaced by `ptxas`."""
    import torch.utils.cpp_extension as ce
    home = ce.CUDA_HOME or "/usr/local/cuda"
    assert not os.path.basename(home).startswith("cuda_home_"), "CUDA_HOME already points at a ptxas shim"
    shim = os.path.join("/tmp", "cuda_home_" + os.path.basename(os.path.dirname(os.path.dirname(ptxas))))
    bindir = os.path.join(shim, "bin")
    os.makedirs(bindir, exist_ok=True)
    for entry in os.listdir(home):                              # everything but bin/ -> symlink to the image toolkit
        dst = os.path.join(shim, entry)
        if entry != "bin" and not os.path.lexists(dst):
            os.symlink(os.path.join(home, entry), dst)
    for entry in os.listdir(os.path.join(home, "bin")):
        dst = os.path.join(bindir, entry)
        if not os.path.lexists(dst):
            os.symlink(ptxas if entry == "ptxas" else os.path.join(home, "bin", entry), dst)
    return shim


class _ScopedToolkit:
    """Points torch's JIT extension builder at a CUDA_HOME shim for the duration of ONE load() and restores the process state afterwards
    (torch.utils.cpp_extension.CUDA_HOME and the CUDA_HOME / CUDA_PATH environment variables), so kernels built later in the same process
    keep the image toolkit."""
    _UNSET = object()

    def __init__(self, home):
        self.home = home

    def __enter__(self):
        import torch.utils.cpp_extension as ce
        self.ce = ce
        self.saved_module = ce.CUDA_HOME
        self.saved_env = {k: os.environ.get(k, self._UNSET) for k in ("CUDA_HOME", "CUDA_PATH")}
        if self.home is not None:
            ce.CUDA_HOME = self.home
            os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"] = self.home
        return self

    def __exit__(self, *exc):
        self.ce.CUDA_HOME = self.saved_module
        for k, v in self.saved_env.items():
            if v is self._UNSET:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _build(verbose: bool = False):
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch.utils.cpp_extension as ce
    from torch.utils.cpp_extension import load, _get_build_directory
    name = "triattn_m1_ext"
    use_134 = os.environ.get("TRIATTN_PTXAS", "") != "image" and os.path.exists(PTXAS_134)
    home = _toolkit_with_ptxas(PTXAS_134) if use_134 else None
    if use_134:
        print(f"[triattn_m1] assembling with {PTXAS_134} via CUDA_HOME={home} (scoped to this build)", flush=True)
    else:
        print(f"[triattn_m1] assembling with the image toolkit ptxas (CUDA_HOME={ce.CUDA_HOME})", flush=True)
    build_dir = _get_build_directory(name, verbose)
    srcs, inst = _generate_sources(build_dir)
    cutlass = os.environ.get("CUTLASS_PATH", "/opt/cutlass")
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
             "-gencode", "arch=compute_90a,code=sm_90a", "-DNDEBUG", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "--ftemplate-backtrace-limit=0", "-lineinfo",
             "-Xcompiler", "-Wno-psabi", "-diag-suppress", "177,550", "-Xptxas", "-v"]
    csrc = os.path.join(_HERE, "csrc", "m1")
    with _ScopedToolkit(home):
        _EXT = load(name=name, sources=[os.path.join(csrc, "m1_binding.cu")] + srcs,
                    extra_include_paths=[csrc, os.path.join(_HERE, "csrc"), inst, os.path.join(cutlass, "include")],
                    extra_cuda_cflags=flags, extra_cflags=["-O3", "-std=c++17"], verbose=verbose)
    return _EXT


def _tma_ok(t: torch.Tensor) -> bool:
    return t.stride(-1) == 1 and all(int(s) % 8 == 0 for s in t.stride()[:-1]) and t.data_ptr() % 16 == 0


class Unsupported(NotImplementedError):
    """Inputs this kernel refuses by name (the caller routes them elsewhere)."""


class _Fallbacks(dict):
    """Census of named fallback paths. fix_tiles = CTA tiles recomputed by the exact (SAFE) pass, accumulated on the device and read
    back (synchronising) only when this entry is accessed."""
    def __getitem__(self, key):
        if key == "fix_tiles":
            return int(_FIX_TOTAL[0].item()) if _FIX_TOTAL is not None else 0
        return dict.__getitem__(self, key)


_FIX_TOTAL = None      # device int32[2]: [CTA tiles recomputed by the SAFE pass, debug counter]
_MASK_COUNTS = None    # device int32[2]: irregular rows (per-row mask words in the hot pass), fully-masked rows (uniform mean of v)
FALLBACKS = _Fallbacks(qkv_copy=0, mask_copy=0, fix_tiles=0)


def debug_counter():
    return 0 if _FIX_TOTAL is None else int(_FIX_TOTAL[1].item())


def mask_census():
    """(irregular_rows, uniform_rows) accumulated over all calls (synchronises)."""
    return (0, 0) if _MASK_COUNTS is None else tuple(int(x) for x in _MASK_COUNTS.tolist())


def triangle_attention_m1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, mask: Optional[torch.Tensor] = None,
                           scale: Optional[float] = None, *, flags: int = 0, trace: Optional[torch.Tensor] = None) -> torch.Tensor:
    global _FIX_TOTAL, _MASK_COUNTS
    if q.dim() == 4:                                   # [N, H, S, D] operands (no batch dim): unsqueeze, no copy
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        if mask is not None and mask.dim() == 4:
            mask = mask.unsqueeze(0)
    B, N, H, S, D = q.shape
    if not (q.is_cuda and k.is_cuda and v.is_cuda and bias.is_cuda and (mask is None or mask.is_cuda)):
        raise Unsupported("M1: every operand must be a CUDA tensor")
    if D != 32:
        raise Unsupported(f"M1: head dim {D} (only 32)")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise Unsupported(f"M1: q/k/v dtype {q.dtype} (only bf16)")
    if k.shape != q.shape or v.shape != q.shape:
        raise Unsupported("M1: q, k, v must have identical shapes [B, N, H, S, 32]")
    ext = _build()
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    qkv = []
    for t in (q, k, v):
        if _tma_ok(t):
            qkv.append(t)
        else:
            FALLBACKS["qkv_copy"] += 1
            qkv.append(t.contiguous())
    bias5 = bias if bias.dim() == 5 else bias.unsqueeze(0)
    if tuple(bias5.shape) != (B, 1, H, S, S):
        raise Unsupported(f"M1: bias shape {tuple(bias.shape)} (need [B, 1, H, S, S]; per-row bias is not triangle attention)")
    maskw = rowkind = kcend = kcstart = keyany = rowkc0 = rowkc1 = None
    if mask is not None:
        if tuple(mask.shape) != (B, N, 1, 1, S):
            raise Unsupported(f"M1: mask shape {tuple(mask.shape)} (need [B, N, 1, 1, S]: one key mask per pair row; per-query / per-head masks are not triangle attention)")
        if mask.dtype != torch.bool:
            FALLBACKS["mask_copy"] += 1
            mask = mask != 0
        if _MASK_COUNTS is None or _MASK_COUNTS.device != q.device:
            _MASK_COUNTS = torch.zeros(2, dtype=torch.int32, device=q.device)
        maskw, keyany, rowkind, kcend, kcstart, rowkc0, rowkc1 = ext.stage_mask(mask, _MASK_COUNTS)
    n_ctas = ((S + 127) // 128) * ((N + 2) // 3 + 1) * B * H
    fix = torch.empty(1 + 3 * n_ctas, dtype=torch.int32, device=q.device)   # fix list: [0] = count (zeroed by stage_bias), then CTA-tile triples
    bias_staged = ext.stage_bias(bias5, float(scale), keyany, fix)
    out = torch.empty(B, N, H, S, D, dtype=q.dtype, device=q.device)
    if _FIX_TOTAL is None or _FIX_TOTAL.device != q.device:
        _FIX_TOTAL = torch.zeros(2, dtype=torch.int32, device=q.device)   # [fix tiles, debug counter]
    ext.fwd(qkv[0], qkv[1], qkv[2], bias_staged, float(scale), out, fix, _FIX_TOTAL, maskw, rowkind, kcend, kcstart, rowkc0, rowkc1, int(flags), trace)   # hot -> SAFE; the SAFE pass adds this call's fix count to _FIX_TOTAL
    return out
