"""The prebuilt kernel library of kit v0_ew (libew_progen2.so, built by build.py from kernels.cu with the pinned
stack's nvcc; BUILD.json records the stack it was built on and the kernel images it carries): loaded by PATH through ctypes,
every launcher on the caller's current CUDA stream. The library binds the CUDA runtime only (no libtorch symbol), so the load
asks one thing of the running torch: the same CUDA runtime MAJOR as the build's (BUILD.json `nvcc` release; any torch build on
that runtime loads it — a different major refuses by name, with the rebuild command; the kit's apply turns that refusal into
its levers stepping aside by name). Before the bytes are mapped they are held to the SHA256SUMS line beside the library
(``sums_refusal``): unlisted or altered bytes refuse by name exactly as an absent library does. ``kernel_image`` says which
image serves a card: BUILD.json `gencode` — a SASS image of the card's major at or below its minor (sm_80 serves 8.0 / 8.6 /
8.9; sm_90 9.0; sm_100 10.x), else the PTX image the driver JIT-compiles for a newer card (compute_90 serves 12.0), else none
(a card older than every image). Wrappers take torch tensors, assert the layouts the kernels index, and refuse a non-zero
launcher return.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = "libew_progen2.so"
SOURCE = "kernels.cu"
SUMS = "SHA256SUMS"                     # beside the library: `<sha256>  libew_progen2.so` (sha256sum format), the line load() holds it to
_lib = None
_record = None
_SUMS_VERDICTS: dict = {}               # library path -> None (listed, digest equal) | the refusal reason; hashed once per process
BINARIES_HELD: dict = {}                # library path -> sha256 of every library that passed the hold, in load order


class KitRefused(RuntimeError):
    """A pin / precondition of the kit failed: the process fails closed."""


def _cudart_preload():
    """torch's own libcudart (the nvidia-cuda-runtime wheel of the image) must be the runtime our launchers bind to."""
    import glob
    cands = []
    try:
        import nvidia.cuda_runtime as nvrt
        cands += glob.glob(os.path.join(os.path.dirname(nvrt.__file__), "lib", "libcudart.so.12*"))
    except Exception:  # noqa: BLE001
        pass
    cands += [m.split()[-1] for m in open("/proc/self/maps").read().splitlines() if "libcudart.so" in m]
    for c in cands:
        if os.path.exists(c):
            return ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
    return None


REBUILD = "python -m engines.progen2.kits.v0_ew.build (cwd opt/forward) rebuilds it on this stack"


def _read_sums(path: str) -> dict:
    """A ``<sha256>  <name>`` file (sha256sum format; blank lines and ``#`` comments skipped) as {name: sha256}."""
    out: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError(f"{path}: not a '<sha256>  <name>' line: {ln!r}")
            out[parts[1].strip()] = parts[0].lower()
    return out


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sums_refusal(path: str):
    """``None`` when the library at ``path`` is a line of the SHA256SUMS beside it and its bytes re-hash to that line. Otherwise the
    reason, one clause — no SHA256SUMS, not listed, unreadable, or the digest differing — after one ``[kit v0_ew] SHA256SUMS: refused
    <library>: <reason>`` line on stderr; load() then refuses the library by name exactly as it refuses an absent one. A library outside
    this directory with no SHA256SUMS beside it (a build.py --out DIR of one's own) is not the shipped binary: ``None``, unchecked.
    One verdict per path per process."""
    p = os.path.abspath(path)
    if p in _SUMS_VERDICTS:
        return _SUMS_VERDICTS[p]
    shipped = os.path.dirname(p) == HERE
    name = os.path.basename(p) if shipped else p                       # the shipped library by name; another by path
    sums = os.path.join(os.path.dirname(p), SUMS)
    reason = digest = None
    if not os.path.isfile(sums):
        if shipped:
            reason = f"no {SUMS} beside it"
    else:
        try:
            want = _read_sums(sums).get(os.path.basename(p))
        except (OSError, ValueError):
            want = None
        if want is None:
            reason = f"not listed in {SUMS}"
        else:
            try:
                digest = _sha256_file(p)
            except OSError as e:
                reason = f"unreadable ({type(e).__name__})"
            else:
                if digest != want:
                    reason = f"sha256 {digest[:16]} != {SUMS} {want[:16]}"
    _SUMS_VERDICTS[p] = reason
    if reason is None:
        if digest is not None:
            BINARIES_HELD[p] = digest
    else:
        print(f"[kit v0_ew] {SUMS}: refused {name}: {reason}", file=sys.stderr, flush=True)
    return reason


def cuda_major_of_build(build: dict):
    """The CUDA runtime major the library was built for: BUILD.json's nvcc release line (`release 12.8` -> 12), else the +cuXYZ tag of the
    torch recorded beside it; None when the record names neither."""
    import re
    for line in build.get("nvcc") or []:
        m = re.search(r"release (\d+)\.", str(line))
        if m:
            return int(m.group(1))
    m = re.search(r"\+cu(\d{2,3})", str(build.get("torch") or ""))
    return int(m.group(1)[:2]) if m else None


def cuda_major_of_torch():
    """The running torch's CUDA runtime major (torch.version.cuda `12.8` -> 12); None for a CPU-only torch."""
    v = getattr(torch.version, "cuda", None)
    return int(str(v).split(".")[0]) if v else None


def build_record(path: str | None = None) -> dict:
    """BUILD.json beside the library (`path` = the library's path): the stack it was built on, its gencode list; {} when absent."""
    p = os.path.join(os.path.dirname(path or os.path.join(HERE, BINARY)), "BUILD.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def kernel_image(build: dict, cc: tuple) -> tuple:
    """The image of the library that runs on a card of compute capability `cc` = (major, minor), from BUILD.json's `gencode` list:
    ("sass", arch) — a real image of the card's major whose minor is at or below the card's (binary compatible within a major);
    ("ptx", arch) — the highest virtual arch at or below the card, JIT-compiled by the driver at load (forward compatible);
    ("unknown", None) — the record names no gencode; (None, reason) — no image can run there."""
    import re
    sass, ptx = [], []
    for g in build.get("gencode") or []:
        m = re.search(r"arch=compute_(\d+)\w*,code=(sm|compute)_(\d+)", str(g))
        if m:
            n = int(m.group(3))
            (sass if m.group(2) == "sm" else ptx).append((n // 10, n % 10))
    if not sass and not ptx:
        return "unknown", None
    cc = (int(cc[0]), int(cc[1]))
    real = [a for a in sass if a[0] == cc[0] and a[1] <= cc[1]]
    if real:
        return "sass", max(real)
    virtual = [a for a in ptx if a <= cc]
    if virtual:
        return "ptx", max(virtual)
    return None, (f"no kernel image of {BINARY} runs on sm_{cc[0]}{cc[1]} (SASS {', '.join('sm_%d%d' % a for a in sorted(sass)) or 'none'}: same major, minor at or below the card's; "
                  f"PTX {', '.join('compute_%d%d' % a for a in sorted(ptx)) or 'none'}: a card at or above it) — {REBUILD}")


def load(path: str | None = None, check_stack: bool = True):
    """Load the library once. The stack check is the CUDA runtime major (cuda_major_of_build vs cuda_major_of_torch): the library binds
    cudart alone, so every torch build on that runtime major loads it; another major (or a CPU-only torch) refuses by name."""
    global _lib, _record
    if _lib is not None:
        return _lib
    path = path or os.path.join(HERE, BINARY)
    if not os.path.exists(path):
        raise KitRefused(f"kit v0_ew: prebuilt library missing at {path} — {REBUILD}")
    why = sums_refusal(path)
    if why is not None:                                                # unlisted / altered bytes: refused by name, as an absent library is
        raise KitRefused(f"kit v0_ew: {BINARY} at {path} refused ({why}) — {REBUILD}")
    build = build_record(path)
    if check_stack:
        built, running = cuda_major_of_build(build), cuda_major_of_torch()
        if built is not None and running != built:
            raise KitRefused(f"kit v0_ew: {BINARY} binds the CUDA {built} runtime (built with {(build.get('nvcc') or ['nvcc'])[0]} beside torch {build.get('torch')}); "
                             f"the running torch {torch.__version__} carries CUDA runtime {getattr(torch.version, 'cuda', None)} — {REBUILD}")
    _cudart_preload()
    lib = ctypes.CDLL(path)
    lib.ew_version.restype = ctypes.c_int
    lib.ew_gelu_new.restype = ctypes.c_int
    lib.ew_gelu_new.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_void_p]
    lib.ew_gelu_new_autocast.restype = ctypes.c_int
    lib.ew_gelu_new_autocast.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_void_p]
    lib.ew_rotary_split_qkv.restype = ctypes.c_int
    lib.ew_rotary_split_qkv.argtypes = ([ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int] + [ctypes.c_int] * 7 +
                                        [ctypes.c_void_p, ctypes.c_void_p] +
                                        [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64] * 3 + [ctypes.c_void_p])
    lib.ew_residual_add2.restype = ctypes.c_int
    lib.ew_residual_add2.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.ew_attn_glue.restype = ctypes.c_int
    lib.ew_attn_glue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int] + [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_float,
                                 ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.ew_layer_norm.restype = ctypes.c_int
    lib.ew_layer_norm.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    if lib.ew_version() != 1:
        raise KitRefused(f"kit v0_ew: {BINARY} reports launcher table version {lib.ew_version()}, this module binds version 1 — {REBUILD}")
    _lib = lib
    _record = {"path": path, "sha256": BINARIES_HELD.get(os.path.abspath(path)), "build": build}   # sha256: the SHA256SUMS line it was held to
    return lib


def record() -> dict:
    if _record is None:
        raise KitRefused("kit v0_ew: library not loaded")
    return dict(_record)


def _stream(t: torch.Tensor):
    return ctypes.c_void_p(torch.cuda.current_stream(t.device).cuda_stream)


def _check(rc: int, what: str):
    if rc != 0:
        raise KitRefused(f"kit v0_ew: {what} launcher returned {rc} ({torch.cuda.CudaError if rc > 0 else 'argument refused'})")


def _is_half(t: torch.Tensor) -> int:
    if t.dtype == torch.float16:
        return 1
    if t.dtype == torch.float32:
        return 0
    raise KitRefused(f"kit v0_ew: dtype {t.dtype} is not one of the stock's (fp16, fp32)")


def gelu_new(x: torch.Tensor, c0: float, c1: float) -> torch.Tensor:
    """The 8-op gelu_new chain in ONE kernel; c0/c1 = the chain's opmath scalars for x's dtype (patches.gelu_scalars)."""
    if not x.is_contiguous():
        x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    if n:
        _check(load().ew_gelu_new(x.data_ptr(), y.data_ptr(), n, _is_half(x), float(c0), float(c1), _stream(x)), "gelu_new")
    return y


def gelu_new_autocast(x: torch.Tensor, c0: float, c1: float, out_half: bool) -> torch.Tensor:
    """The autocast chain on an fp16 x (pow in fp32, everything after it fp32, 0.5*x fp16): fp32 output (the stock tensor) or
    fp16 (the fc_out autocast cast folded)."""
    if x.dtype != torch.float16:
        raise KitRefused(f"kit v0_ew: the autocast gelu chain takes an fp16 input (got {x.dtype})")
    if not x.is_contiguous():
        x = x.contiguous()
    y = torch.empty(x.shape, dtype=torch.float16 if out_half else torch.float32, device=x.device)
    n = x.numel()
    if n:
        _check(load().ew_gelu_new_autocast(x.data_ptr(), y.data_ptr(), n, 1 if out_half else 0, float(c0), float(c1), _stream(x)), "gelu_new_autocast")
    return y


def rotary_split_qkv(qkv: torch.Tensor, H: int, hd: int, rd: int, offset: int, costab: torch.Tensor, sintab: torch.Tensor,
                     q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor) -> None:
    """qkv (B, L, 3E) in the regime dtype (last dim contiguous); tables (n_pos, rd/2) fp32 contiguous; q_out/k_out fp32 and
    v_out (qkv dtype) are 4-D views (B, L, H, hd) with unit stride on hd — the stock's buffers or cache-slot views."""
    B, L, E3 = qkv.shape
    E = E3 // 3
    assert E * 3 == E3 and H * hd == E and H % 8 == 0 and hd % 2 == 0 and rd % 2 == 0 and 0 < rd <= hd, (qkv.shape, H, hd, rd)
    assert qkv.stride(2) == 1, qkv.stride()
    assert costab.dtype == torch.float32 and sintab.dtype == torch.float32 and costab.is_contiguous() and sintab.is_contiguous()
    assert costab.shape == sintab.shape and costab.shape[1] == rd // 2 and offset + L <= costab.shape[0], (costab.shape, offset, L)
    for t, dt in ((q_out, torch.float32), (k_out, torch.float32), (v_out, qkv.dtype)):
        assert t.dtype == dt and t.shape == (B, L, H, hd) and t.stride(3) == 1 and t.device == qkv.device, (t.dtype, t.shape, t.stride())
    _check(load().ew_rotary_split_qkv(qkv.data_ptr(), qkv.stride(0), qkv.stride(1), _is_half(qkv), B, L, H, hd, rd, E, int(offset),
                                      costab.data_ptr(), sintab.data_ptr(),
                                      q_out.data_ptr(), q_out.stride(0), q_out.stride(1), q_out.stride(2),
                                      k_out.data_ptr(), k_out.stride(0), k_out.stride(1), k_out.stride(2),
                                      v_out.data_ptr(), v_out.stride(0), v_out.stride(1), v_out.stride(2), _stream(qkv)), "rotary_split_qkv")


def residual_add2(a: torch.Tensor, f: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """RN_out( RN_ab(a + f) + r ): the block's two residual adds with the stock's dtype promotion."""
    assert a.shape == f.shape == r.shape and a.dtype == f.dtype, (a.shape, f.shape, r.shape, a.dtype, f.dtype)
    a = a.contiguous(); f = f.contiguous(); r = r.contiguous()
    out_dtype = torch.promote_types(a.dtype, r.dtype)
    out = torch.empty(a.shape, dtype=out_dtype, device=a.device)
    n = a.numel()
    if n:
        _check(load().ew_residual_add2(a.data_ptr(), f.data_ptr(), r.data_ptr(), out.data_ptr(), n, _is_half(a), _is_half(r),
                                       _is_half(out), _stream(a)), "residual_add2")
    return out


def attn_glue(w: torch.Tensor, key_length: int, inv_scale: float, masked_value: float, amask: torch.Tensor | None) -> torch.Tensor:
    """where(causal, RN(w * inv_scale), masked_value) + amask, w (B, H, Lq, Lk) contiguous; amask (B, 1, 1, Lk) or None."""
    assert w.dim() == 4 and w.is_contiguous(), (w.shape, w.stride())
    B, H, Lq, Lk = w.shape
    out = torch.empty_like(w)
    if amask is not None:
        assert amask.shape == (B, 1, 1, Lk) or amask.shape == (1, 1, 1, Lk), amask.shape
        am_ptr, am_half, am_sb, am_sk = amask.data_ptr(), _is_half(amask), (amask.stride(0) if amask.shape[0] == B else 0), amask.stride(3)
    else:
        am_ptr, am_half, am_sb, am_sk = None, 0, 0, 0
    _check(load().ew_attn_glue(w.data_ptr(), out.data_ptr(), _is_half(w), B, H, Lq, Lk, int(key_length), float(inv_scale), float(masked_value),
                               am_ptr, am_half, am_sb, am_sk, _stream(w)), "attn_glue")
    return out


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, variant: int, stats: bool = False):
    """ATen's vectorized layer norm replica on (M, N) rows; x, weight, bias in one dtype (fp16 or fp32), N % 8 == 0."""
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    assert weight.dtype == x.dtype and bias.dtype == x.dtype and weight.shape == (N,) and bias.shape == (N,) and weight.is_contiguous() and bias.is_contiguous()
    assert N % 8 == 0, N
    M = x2.shape[0]
    y = torch.empty_like(x2)
    mean = rstd = None
    if stats:
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    if M:
        _check(load().ew_layer_norm(x2.data_ptr(), weight.data_ptr(), bias.data_ptr(), y.data_ptr(),
                                    mean.data_ptr() if stats else None, rstd.data_ptr() if stats else None,
                                    M, N, float(eps), _is_half(x2), int(variant), _stream(x2)), "layer_norm")
    y = y.reshape(x.shape)
    return (y, mean, rstd) if stats else y
