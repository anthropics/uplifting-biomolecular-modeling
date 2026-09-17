"""triattn_exact.cuda_mma — hand-CUDA mma.sync route (sm_80 code path, runs on sm_80/sm_90).

Interface:
    attention(q, k, v, bias, mask=None, scale=None) -> out      (library semantics; strided inputs served in place)
    supports(q, k, v, bias, mask, scale, lib_version, device) -> (bool, reason)
Refusal = raise triattn_exact.Refused(reason).

Build (development path): nvcc (from the CUDA toolkit on PATH / $CUDA_HOME) compiles csrc/cuda_mma/triattn_v3.cu (located via
_paths.csrc_dir) into a shared library cached under _paths.cache_dir() ($TRIATTN_EXACT_CACHE; keyed by source hash + nvcc
version + flags); loaded with ctypes; kernels are launched on torch's current stream (so CUDA-graph capture of attention()
works).  No torch C++ extension / ninja needed.  Production serves the same kernels from prebuilt cubins (_prebuilt).

Exploration build: every undecided arithmetic clause is a runtime switch (`sw=` dict); DEFAULT_SW holds the current
locked values (each switch names the discriminating test that fixed it).
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import subprocess
import threading

import torch

try:
    from .. import Refused
except Exception:  # pragma: no cover  # hygiene: no-cuda (import-time fallback definition for standalone use)
    class Refused(RuntimeError):
        def __init__(self, reason, cell=None):
            self.reason = reason; self.cell = cell or {}
            super().__init__(f"triattn_exact refused: {reason} cell={self.cell}")

from .. import _paths

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = _paths.csrc_dir("cuda_mma", "triattn_fwd.cu")

# ---- arithmetic switches (see csrc/cuda_mma/triattn_fwd.cu struct Sw) ----
DEFAULT_SW = dict(
    combine=0,        # fma(d, scale, bias)
    mask_mode=0,      # masked -> x = -1e9 (replace)
    mask_value=-1.0e9,
    tail_mode=1,      # keys >= S inside the last 64-tile behave as MASKED keys (x=-1e9, V=0)
    exp_mode=21,      # ex2.approx.ftz.f32(x2 - m2)  (20/21/22/23 indistinguishable so far)
    lsum_mode=10,     # = kind 2 (S = tile terms summed from 0, thread-partial; l = fma(l, alpha, S)) | allreduce once at end, order xor1 then xor2 (bit3)
    lsum_bf16=0,
    lsum_inner=0,     # in-thread term order: sequential over (subtile j, col c)
    norm_mode=2,      # rcp.approx.ftz(l) * o
    key_reverse=0,
    pv_rev=0,
    inf_guard=1,      # m==-inf -> subtract 0; l==0 or NaN -> output unnormalised acc
    first_special=1,
    alpha_mode=0,
    qk_korder=0,      # d 0..15 then 16..31, C=0 chained
    ftz=1,            # all fp32 ops .ftz (canary 'tiny' family: library flushes denormal outputs to +-0)
    o_mode=0,         # O *= alpha then O is the live MMA accumulator of P.V (FA2 style) [smoke suite excludes o_mode 1/2]
    l2_mode=1,        # log2 domain: x2 = RN(fma(d,s,b)*log2e)
)
DEFAULT_BLOCK_N = 64
DEFAULT_NWARPS = 4
SW_KEYS = ["combine", "mask_mode", "mask_value", "tail_mode", "exp_mode", "lsum_mode", "lsum_bf16", "lsum_inner", "norm_mode", "key_reverse",
           "pv_rev", "inf_guard", "first_special", "alpha_mode", "qk_korder", "l2_mode", "o_mode", "ftz"]


class _LaunchArgs(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.c_void_p), ("k", ctypes.c_void_p), ("v", ctypes.c_void_p), ("bias", ctypes.c_void_p), ("bias_bf16", ctypes.c_int),
        ("mask", ctypes.c_void_p), ("out", ctypes.c_void_p), ("aux_m", ctypes.c_void_p), ("aux_l", ctypes.c_void_p),
        ("B", ctypes.c_int), ("N", ctypes.c_int), ("H", ctypes.c_int), ("S", ctypes.c_int),
        ("sqB", ctypes.c_longlong), ("sqN", ctypes.c_longlong), ("sqH", ctypes.c_longlong), ("sqS", ctypes.c_longlong),
        ("skB", ctypes.c_longlong), ("skN", ctypes.c_longlong), ("skH", ctypes.c_longlong), ("skS", ctypes.c_longlong),
        ("svB", ctypes.c_longlong), ("svN", ctypes.c_longlong), ("svH", ctypes.c_longlong), ("svS", ctypes.c_longlong),
        ("sbB", ctypes.c_longlong), ("sbH", ctypes.c_longlong), ("sbQ", ctypes.c_longlong), ("sbK", ctypes.c_longlong),
        ("smB", ctypes.c_longlong), ("smN", ctypes.c_longlong),
        ("scale", ctypes.c_float), ("block_n", ctypes.c_int), ("nwarps", ctypes.c_int),
        ("combine", ctypes.c_int), ("mask_mode", ctypes.c_int), ("mask_value", ctypes.c_float), ("tail_mode", ctypes.c_int),
        ("exp_mode", ctypes.c_int), ("lsum_mode", ctypes.c_int), ("lsum_bf16", ctypes.c_int), ("lsum_inner", ctypes.c_int), ("norm_mode", ctypes.c_int),
        ("key_reverse", ctypes.c_int), ("pv_rev", ctypes.c_int), ("inf_guard", ctypes.c_int), ("first_special", ctypes.c_int),
        ("alpha_mode", ctypes.c_int), ("qk_korder", ctypes.c_int), ("l2_mode", ctypes.c_int), ("o_mode", ctypes.c_int), ("ftz", ctypes.c_int),
        ("stream", ctypes.c_void_p),
    ]

class _LaunchArgsV1(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.c_void_p), ("k", ctypes.c_void_p), ("v", ctypes.c_void_p), ("bias", ctypes.c_void_p), ("bias_bf16", ctypes.c_int),
        ("mask", ctypes.c_void_p), ("out", ctypes.c_void_p),
        ("B", ctypes.c_int), ("N", ctypes.c_int), ("H", ctypes.c_int), ("S", ctypes.c_int),
        ("sqB", ctypes.c_longlong), ("sqN", ctypes.c_longlong), ("sqH", ctypes.c_longlong), ("sqS", ctypes.c_longlong),
        ("skB", ctypes.c_longlong), ("skN", ctypes.c_longlong), ("skH", ctypes.c_longlong), ("skS", ctypes.c_longlong),
        ("svB", ctypes.c_longlong), ("svN", ctypes.c_longlong), ("svH", ctypes.c_longlong), ("svS", ctypes.c_longlong),
        ("sbB", ctypes.c_longlong), ("sbH", ctypes.c_longlong), ("sbQ", ctypes.c_longlong), ("sbK", ctypes.c_longlong),
        ("smB", ctypes.c_longlong), ("smN", ctypes.c_longlong),
        ("scale", ctypes.c_float), ("G", ctypes.c_int), ("nwarps", ctypes.c_int), ("bias_fast", ctypes.c_int),
        ("stream", ctypes.c_void_p),
    ]



_lock = threading.Lock()


def _nvcc():
    for c in (os.environ.get("CUDA_HOME", ""), "/usr/local/cuda"):
        p = os.path.join(c, "bin", "nvcc")
        if c and os.path.exists(p):
            return p
    return "nvcc"


_SRC_V1 = _paths.csrc_dir("cuda_mma", "triattn_v1.cu")
_SRC_V2 = _paths.csrc_dir("cuda_mma", "triattn_v2.cu")
_SRC_V3 = _paths.csrc_dir("cuda_mma", "triattn_v3.cu")
_SRC_V4 = _paths.csrc_dir("cuda_mma", "triattn_v4.cu")
_libs = {}


def build(which="v0", verbose=False, extra_flags=()):
    """Compile (or reuse) the shared library for kernel `which` ('v0' exploration build | 'v1' speed build); returns its path."""
    path = _paths.kernel_source("cuda_mma", os.path.basename({"v0": _SRC, "v1": _SRC_V1, "v2": _SRC_V2, "v3": _SRC_V3, "v4": _SRC_V4}[which]))
    with open(path, "rb") as f:
        src = f.read()
    try:
        ver = subprocess.run([_nvcc(), "--version"], capture_output=True, text=True).stdout
    except Exception as e:  # hygiene: no-cuda (host-side probe for the nvcc binary)
        raise Refused(f"nvcc not available: {e}")
    archs = os.environ.get("TRIATTN_EXACT_ARCHS", "80;90").split(";")
    if which == "v4":
        archs = ["90a"]                                                  # clusters + TMA multicast: Hopper-only code path
    flags = ["-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC", "-lineinfo", "--fmad=false"]   # every FMA in the kernels is explicit
    for a_ in archs:
        flags += ["-gencode", f"arch=compute_{a_},code=sm_{a_}"]
    if which == "v4":
        flags += ["-lcuda"]
    flags += list(extra_flags)
    key = hashlib.sha1(src + ver.encode() + " ".join(flags).encode()).hexdigest()[:16]
    cache = _cache_dir()
    so = os.path.join(cache, f"cuda_mma_{which}_{key}.so")
    if not os.path.exists(so):
        tmp = so + f".{os.getpid()}.tmp"
        cmd = [_nvcc(), *flags, "-o", tmp, path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("nvcc failed:\n" + r.stdout + r.stderr)
        if verbose:
            print(r.stdout, r.stderr)
        os.replace(tmp, so)
    return so


def _load(which="v0"):
    with _lock:
        lib = _libs.get(which)
        if lib is None:
            so = build(which)
            lib = ctypes.CDLL(so)
            if which == "v0":
                lib.triattn_fwd_launch.argtypes = [ctypes.POINTER(_LaunchArgs)]
                lib.triattn_fwd_launch.restype = ctypes.c_int
                assert lib.triattn_sizeof_launchargs() == ctypes.sizeof(_LaunchArgs), (lib.triattn_sizeof_launchargs(), ctypes.sizeof(_LaunchArgs))
            elif which == "v1":
                lib.triattn_v1_launch.argtypes = [ctypes.POINTER(_LaunchArgsV1)]
                lib.triattn_v1_launch.restype = ctypes.c_int
                assert lib.triattn_v1_sizeof_launchargs() == ctypes.sizeof(_LaunchArgsV1), (lib.triattn_v1_sizeof_launchargs(), ctypes.sizeof(_LaunchArgsV1))
            elif which == "v2":
                lib.triattn_v2_launch.argtypes = [ctypes.POINTER(_LaunchArgsV2)]
                lib.triattn_v2_launch.restype = ctypes.c_int
                assert lib.triattn_v2_sizeof_launchargs() == ctypes.sizeof(_LaunchArgsV2), (lib.triattn_v2_sizeof_launchargs(), ctypes.sizeof(_LaunchArgsV2))
            elif which == "v3":
                lib.triattn_v3_launch.argtypes = [ctypes.POINTER(_LaunchArgsV2)]
                lib.triattn_v3_launch.restype = ctypes.c_int
                lib.triattn_v3_vscan.argtypes = [ctypes.POINTER(_LaunchArgsV2), ctypes.c_void_p]
                lib.triattn_v3_vscan.restype = ctypes.c_int
                assert lib.triattn_v3_sizeof_launchargs() == ctypes.sizeof(_LaunchArgsV2), (lib.triattn_v3_sizeof_launchargs(), ctypes.sizeof(_LaunchArgsV2))
            else:
                lib.triattn_v4_launch.argtypes = [ctypes.POINTER(_LaunchArgsV2)]
                lib.triattn_v4_launch.restype = ctypes.c_int
                assert lib.triattn_v4_sizeof_launchargs() == ctypes.sizeof(_LaunchArgsV2), (lib.triattn_v4_sizeof_launchargs(), ctypes.sizeof(_LaunchArgsV2))
            _libs[which] = lib
        return lib



def _as5d(t, name):
    if t.dim() == 4:
        return t.unsqueeze(0)
    if t.dim() != 5:
        raise Refused(f"{name} must be 4-D or 5-D, got {t.dim()}-D")
    return t


def default_scale(D):
    return 1.0 / math.sqrt(D)


def _check_inputs(q, k, v, bias, mask):
    q = _as5d(q, "q"); k = _as5d(k, "k"); v = _as5d(v, "v"); bias = _as5d(bias, "bias")
    B, N, H, S, D = q.shape
    if D != 32:
        raise Refused(f"head_dim {D} not served (D=32 only)")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise Refused(f"dtype {q.dtype}/{k.dtype}/{v.dtype} not served (bf16 only)")
    if tuple(k.shape) != (B, N, H, S, D) or tuple(v.shape) != (B, N, H, S, D):
        raise Refused(f"k/v shape {tuple(k.shape)}/{tuple(v.shape)} != q shape {tuple(q.shape)} (Q==K required)")
    if not q.is_cuda:
        raise Refused(f"q on {q.device}: CUDA tensors only")
    for name, t in (("k", k), ("v", v), ("bias", bias), ("mask", mask)):          # every tensor on q.device (a host pointer would fault)
        if t is not None and t.device != q.device:
            raise Refused(f"{name} is on {t.device} but q is on {q.device}: all inputs must be on the same CUDA device")
    if S < 1 or N < 1 or B < 1:
        raise Refused(f"empty problem B={B} N={N} S={S}")
    for name, t, align in (("q", q, 16), ("k", k, 16), ("v", v, 16)):              # 16B base + 16B rows (the library faults otherwise)
        if t.stride(-1) != 1:
            raise Refused(f"{name} last-dim stride {t.stride(-1)} != 1 (need contiguous head_dim)")
        if t.data_ptr() % align:
            raise Refused(f"{name} base address not {align}-byte aligned")
        need = align // 2
        for i, s_ in enumerate(t.stride()[:-1]):
            if t.shape[i] > 1 and s_ % need:
                raise Refused(f"{name} stride[{i}]={s_} not a multiple of {need} elements ({align}B rows needed)")
    if bias.dtype not in (torch.float32, torch.bfloat16):
        raise Refused(f"bias dtype {bias.dtype} not served (fp32 or bf16)")
    if bias.dim() != 5 or bias.shape[1] != 1 or bias.shape[2] != H or bias.shape[3] != S or bias.shape[4] != S or bias.shape[0] != B:
        raise Refused(f"bias shape {tuple(bias.shape)} not [B,1,H,S,S] with B={B} (the library rejects other batch dims)")
    if bias.data_ptr() % bias.element_size():
        raise Refused("bias base address misaligned")
    if mask is not None:
        mask = _as5d(mask, "mask")
        if mask.dtype != torch.bool:
            mask = mask != 0
        if tuple(mask.shape) != (B, N, 1, 1, S):
            raise Refused(f"mask shape {tuple(mask.shape)} not [B,N,1,1,S]")
        if mask.stride(-1) != 1 and S > 1:
            mask = mask.contiguous()  # tiny [B,N,S] bytes; not one of the large strided inputs
    return q, k, v, bias, mask


def fallback_threshold():
    """Under the default environment the library serves S_q <= T (T=100) with a torch fallback whose bits differ
    from the kernel's; CUEQ_TRIATTN_FALLBACK_THRESHOLD moves T."""
    v = os.environ.get("CUEQ_TRIATTN_FALLBACK_THRESHOLD")
    try:
        return int(float(v)) if v is not None and v != "" else 100
    except ValueError:
        return 100


DEFAULT_IMPL = "v3"


def pick_G(B, N, H, S, nwarps=4):
    """Pair-rows per CTA sharing one bias tile: largest G in {4,2,1} that still yields >= 2 waves of CTAs."""
    bm = nwarps * 16
    for G in (4, 2, 1):
        ctas = ((S + bm - 1) // bm) * H * B * ((N + G - 1) // G)
        if ctas >= 2 * 132 or G == 1:
            return G
    return 1


def _bias_fast(b5):
    es = b5.element_size()
    if b5.stride(-1) != 1 and b5.shape[-1] > 1:
        return 0
    for d in (0, 2, 3):
        if b5.shape[d] > 1 and (b5.stride(d) * es) % 16 != 0:
            return 0
    return 1 if b5.data_ptr() % 16 == 0 else 0


class _LaunchArgsV2(ctypes.Structure):
    _fields_ = list(_LaunchArgsV1._fields_) + [("cfg", ctypes.c_int), ("pad_", ctypes.c_int), ("vbad", ctypes.c_void_p)]


VSCAN_MIN_QBLOCKS = int(os.environ.get("TRIATTN_EXACT_VSCAN_QBLOCKS", "4"))   # pre-pass V-finiteness scan when >= this many q-blocks share it


# v2 configurations: cfg id -> (G pair-rows per CTA, BM query rows per CTA, threads, stages)
V2_CFG = {0: (4, 64, 256, 2), 1: (2, 64, 256, 2), 2: (2, 64, 128, 2), 3: (1, 64, 128, 2), 4: (2, 128, 256, 2), 5: (2, 128, 256, 3),
          6: (4, 128, 512, 2), 7: (4, 128, 512, 3), 8: (4, 64, 256, 3),
          9: (4, 16, 128, 2), 10: (4, 32, 256, 2)}                    # (G, BM, threads, stages); 9/10 = small-BM configs for TAIL rows (v3 only)
TAIL_SPLIT = os.environ.get("TRIATTN_EXACT_TAIL_SPLIT", "1") == "1"   # serve S % BM leftover query rows with a second small-BM launch


def plan_rows(S, BM):
    """-> (main_rows, tail_rows, tail_cfg): split the query rows so the last partial BM-row block does not pad S up to a multiple
    of BM (the S=64k+1 cliff). tail_cfg in {9 (BM16), 10 (BM32), 0 (BM64)}; no split when the leftover fills >= 3/4 of a block."""
    rem = S % BM
    if not TAIL_SPLIT or rem == 0 or S - rem < 256 or rem > (3 * BM) // 4:     # measured: the extra launch only pays off from S >= 257
        return S, 0, None
    tail_cfg = 9 if rem <= 16 else (10 if rem <= 32 else (0 if rem <= 64 else None))
    if tail_cfg is None:
        return S, 0, None
    return S - rem, rem, tail_cfg
V2_DEFAULT_ORDER = [4, 0, 2, 3]      # preference for auto-pick: BM128/G2 when >= 2 CTAs per SM result, else smaller CTAs


_CFG_RELCOST = {4: 1.00, 0: 1.09, 2: 1.20, 3: 1.35}   # measured relative cost per padded element (H100, S=512..1024)


def pick_cfg_id(B, N, H, S):
    """Cheapest config by (padded rows x padded pair-rows x relative per-element cost), among configs that fill the GPU
    (>= 2 CTAs per SM); falls back to the smallest CTA shape for tiny grids."""
    best, best_cost = 3, None
    for cfg in V2_DEFAULT_ORDER:
        G, BM, thr, nst = V2_CFG[cfg]
        qblk, ngrp = (S + BM - 1) // BM, (N + G - 1) // G
        ctas = qblk * H * B * ngrp
        if ctas < 2 * 132 and cfg != 3:
            continue
        main_rows, tail_rows, tail_cfg = plan_rows(S, BM)
        rows = qblk * BM if not tail_rows else main_rows + V2_CFG[tail_cfg][1] + 24     # +24 rows ~ cost of the extra launch
        cost = rows * ngrp * G * _CFG_RELCOST.get(cfg, 1.5)
        if best_cost is None or cost < best_cost:
            best, best_cost = cfg, cost
    return best


V2_CONFIGS = [(8, 8), (4, 8), (2, 8), (2, 4), (1, 4)]   # legacy (G, nwarps) names


def pick_cfg_v2(B, N, H, S):
    """(G, nwarps) for v2: most pair-rows per CTA that still gives >= ~2 CTAs per SM (2 x 132)."""
    qblocks = (S + 63) // 64
    for G, nw in V2_CONFIGS[1:]:                      # (8,8) only on request for now
        ctas = qblocks * H * B * ((N + G - 1) // G)
        if ctas >= 2 * 132:
            return G, nw
    return 1, 4


_LEGACY_V2 = {(4, 8): 0, (2, 8): 1, (2, 4): 2, (1, 4): 3}


def _stage_bias(b5):
    """Data movement is free, arithmetic is not: when the bias view cannot be streamed with 16-byte cp.async (key stride != 1,
    e.g. the ENGINE permuted [B,S,S,H] view, or a row pitch / base that is not 16-byte aligned, e.g. odd S), stage it once into a
    contiguous buffer whose row pitch is padded to 16 bytes. Values are unchanged, so the arithmetic is unchanged."""
    if _bias_fast(b5) or os.environ.get("TRIATTN_EXACT_BIAS_INPLACE") == "1":
        return b5
    Bb, _, H, S, _ = b5.shape
    per16 = 16 // b5.element_size()
    Sp = (S + per16 - 1) // per16 * per16
    buf = torch.empty((Bb, 1, H, S, Sp), dtype=b5.dtype, device=b5.device)
    st = buf[..., :S]
    st.copy_(b5)
    return st


def _cache_dir():
    return _paths.cache_dir()          # $TRIATTN_EXACT_CACHE (see LAYOUT.md); never inside the package


def _fill_launch_args(a, q5, k5, v5, b5, m5, out, scale):
    B, N, H, S, D = q5.shape
    a.q, a.k, a.v, a.bias = q5.data_ptr(), k5.data_ptr(), v5.data_ptr(), b5.data_ptr()
    a.bias_bf16 = 1 if b5.dtype == torch.bfloat16 else 0
    a.mask = m5.data_ptr() if m5 is not None else None
    a.out = out.data_ptr()
    a.B, a.N, a.H, a.S = B, N, H, S
    a.sqB, a.sqN, a.sqH, a.sqS = [q5.stride(i) for i in range(4)]
    a.skB, a.skN, a.skH, a.skS = [k5.stride(i) for i in range(4)]
    a.svB, a.svN, a.svH, a.svS = [v5.stride(i) for i in range(4)]
    a.sbB = b5.stride(0) if b5.shape[0] == B else 0
    a.sbH, a.sbQ, a.sbK = b5.stride(2), b5.stride(3), b5.stride(4)
    if m5 is not None:
        a.smB = m5.stride(0) if m5.shape[0] == B else 0
        a.smN = m5.stride(1)
    else:
        a.smB = a.smN = 0
    a.scale = float(default_scale(D) if scale is None else scale)
    a.bias_fast = _bias_fast(b5)
    a.stream = torch.cuda.current_stream(q5.device).cuda_stream


def _attention_v1(q5, k5, v5, b5, m5, scale, G=None, nwarps=None, impl="v1", cfg=None, cx=None):
    B, N, H, S, D = q5.shape
    if impl in ("v2", "v3", "v4"):
        if cfg is None:
            cfg = _LEGACY_V2.get((G, nwarps)) if (G and nwarps) else pick_cfg_id(B, N, H, S)
            if cfg is None:
                raise ValueError(f"no v2 config for G={G} nwarps={nwarps}")
        G = V2_CFG[cfg][0]
    nwarps = int(nwarps or 4)
    G = int(G or pick_G(B, N, H, S, nwarps))
    if B * ((N + G - 1) // G) > 65535:
        raise Refused(f"B*ceil(N/G)={B * ((N + G - 1) // G)} > 65535 grid limit", cell=dict(B=B, N=N))
    lib = _load(impl)
    if impl != "v1":
        b5 = _stage_bias(b5)
    out = torch.empty((B, N, H, S, D), dtype=torch.bfloat16, device=q5.device)
    a = _LaunchArgsV1() if impl == "v1" else _LaunchArgsV2()
    tail_rows, tail_cfg = 0, None
    if impl in ("v2", "v3", "v4"):
        a.cfg = int(cfg)
    vb = None
    if impl == "v3":
        main_rows, tail_rows, tail_cfg = plan_rows(S, V2_CFG[int(cfg)][1])
        a.pad_ = int(tail_rows)                      # >0: this (main) launch serves rows [0, S - tail_rows)
        nqblk = (S + V2_CFG[int(cfg)][1] - 1) // V2_CFG[int(cfg)][1] + (1 if tail_rows else 0)
        if m5 is not None and nqblk >= VSCAN_MIN_QBLOCKS:
            # exactness guard for masked-tile skipping (0*NaN must propagate): one pre-pass over the masked part of V marks
            # (b,n,h,tile) tiles holding NaN/Inf; cheaper than every q-block CTA re-scanning them (the in-kernel fallback)
            vb = torch.empty(B * N * H * ((S + 63) // 64), dtype=torch.uint8, device=q5.device)
    if impl == "v4":
        if cfg not in (0, 2, 3, 4, 5):
            cfg = 4; a.cfg = 4
        BM = V2_CFG[cfg][1]
        qblocks = (S + BM - 1) // BM
        cxv = cx if cx else (4 if qblocks % 4 == 0 else (2 if qblocks % 2 == 0 else (2 if qblocks >= 3 else 1)))
        a.pad_ = int(cxv)
    _fill_launch_args(a, q5, k5, v5, b5, m5, out, scale)
    a.G, a.nwarps = G, nwarps
    with torch.cuda.device(q5.device):
        if vb is not None:
            rc = lib.triattn_v3_vscan(ctypes.byref(a), ctypes.c_void_p(vb.data_ptr()))
            if rc != 0:
                raise RuntimeError(f"triattn_v3_vscan failed rc={rc}")
            a.vbad = vb.data_ptr()
        rc = (lib.triattn_v1_launch(ctypes.byref(a)) if impl == "v1" else
              lib.triattn_v2_launch(ctypes.byref(a)) if impl == "v2" else
              lib.triattn_v3_launch(ctypes.byref(a)) if impl == "v3" else lib.triattn_v4_launch(ctypes.byref(a)))
        if rc == 0 and tail_rows:                    # second launch: the S % BM leftover rows with a small-BM config
            a.cfg = int(tail_cfg); a.G = V2_CFG[int(tail_cfg)][0]; a.nwarps = V2_CFG[int(tail_cfg)][2] // 32
            a.pad_ = -int(tail_rows)
            rc = lib.triattn_v3_launch(ctypes.byref(a))
    if rc != 0:
        raise RuntimeError(f"triattn_{impl}_launch failed rc={rc} (G={G}, nwarps={nwarps})")
    return out


def attention(q, k, v, bias, mask=None, scale=None, return_aux=False, *, sw=None, block_n=None, nwarps=None, debug_aux=False,
              force_kernel=False, impl=None, G=None, cfg=None, cx=None, **unexpected):
    """Forward triangle attention (library semantics). Exploration keyword args: sw (switch overrides), block_n, nwarps,
    force_kernel (serve S <= fallback threshold with kernel arithmetic, e.g. to compare against return_aux=True library calls)."""
    if return_aux:
        raise Refused("return_aux=True is not served by the cuda_mma route (library lse/max side outputs not reproduced here)")
    if unexpected:
        raise Refused(f"unsupported keyword arguments {sorted(unexpected)} (e.g. kv_lengths)")
    q5, k5, v5, b5, m5 = _check_inputs(q, k, v, bias, mask)
    B, N, H, S, D = q5.shape
    if not force_kernel and S <= fallback_threshold():
        raise Refused(f"S={S} <= CUEQ_TRIATTN_FALLBACK_THRESHOLD={fallback_threshold()}: the library takes its torch fallback path "
                      f"here (different arithmetic); kernel path only", cell=dict(S=S))
    if B * N > 65535:
        raise Refused(f"B*N={B*N} > 65535 grid limit of this launcher", cell=dict(B=B, N=N))
    if impl is None:
        impl = "v0" if (sw or block_n or debug_aux) else DEFAULT_IMPL
    if impl in ("v1", "v2", "v3", "v4"):
        return _attention_v1(q5, k5, v5, b5, m5, scale, G=G, nwarps=nwarps, impl=impl, cfg=cfg, cx=cx)
    lib = _load("v0")
    out = torch.empty((B, N, H, S, D), dtype=torch.bfloat16, device=q5.device)
    aux_m = aux_l = None
    if debug_aux:
        aux_m = torch.empty((B, N, H, S), dtype=torch.float32, device=q5.device)
        aux_l = torch.empty((B, N, H, S), dtype=torch.float32, device=q5.device)
    swd = dict(DEFAULT_SW)
    if sw:
        for kk in sw:
            if kk not in swd:
                raise ValueError(f"unknown switch {kk}")
        swd.update(sw)
    a = _LaunchArgs()
    a.q, a.k, a.v, a.bias = q5.data_ptr(), k5.data_ptr(), v5.data_ptr(), b5.data_ptr()
    a.bias_bf16 = 1 if b5.dtype == torch.bfloat16 else 0
    a.mask = m5.data_ptr() if m5 is not None else None
    a.out = out.data_ptr(); a.aux_m = aux_m.data_ptr() if aux_m is not None else None; a.aux_l = aux_l.data_ptr() if aux_l is not None else None
    a.B, a.N, a.H, a.S = B, N, H, S
    a.sqB, a.sqN, a.sqH, a.sqS = [q5.stride(i) for i in range(4)]
    a.skB, a.skN, a.skH, a.skS = [k5.stride(i) for i in range(4)]
    a.svB, a.svN, a.svH, a.svS = [v5.stride(i) for i in range(4)]
    a.sbB = b5.stride(0) if b5.shape[0] == B else 0
    a.sbH, a.sbQ, a.sbK = b5.stride(2), b5.stride(3), b5.stride(4)
    if m5 is not None:
        a.smB = m5.stride(0) if m5.shape[0] == B else 0
        a.smN = m5.stride(1)
    else:
        a.smB = a.smN = 0
    a.scale = float(default_scale(D) if scale is None else scale)
    a.block_n = int(block_n or DEFAULT_BLOCK_N); a.nwarps = int(nwarps or DEFAULT_NWARPS)
    for kk in SW_KEYS:
        setattr(a, kk, swd[kk])
    a.stream = torch.cuda.current_stream(q5.device).cuda_stream
    with torch.cuda.device(q5.device):
        rc = lib.triattn_fwd_launch(ctypes.byref(a))
    if rc != 0:
        raise RuntimeError(f"triattn_fwd_launch failed rc={rc} (block_n={a.block_n}, nwarps={a.nwarps})")
    # library semantics: 4-D inputs return a 5-D [1,N,H,S,D] output -> never squeeze
    if debug_aux:
        return out, aux_m, aux_l
    return out


# ---- capability (proof of equality lives in CELLS.json, not here) ----
DEVICE_CLASSES_SERVED = ("H100_SXM", "H100_PCIe", "H100_NVL", "H200", "A100_SXM_80GB", "A100_PCIe_80GB")   # sm_80 code path (face names)
LIB_VERSIONS_SERVED = ("0.11.1", "0.10.0")
CAPABILITY = ("bf16 q/k/v, D=32, any B/N/H, S > CUEQ_TRIATTN_FALLBACK_THRESHOLD (default 100; kernel path only), "
              "bias fp32|bf16 [B,1,H,S,S] (any strides), mask None|bool-like [B,N,1,1,S], q/k/v last-dim contiguous with 16-B aligned "
              "base and rows (strided ENGINE views served in place), all tensors on one CUDA device, sm_80+ (built for 8.0;9.0)")


def device_class(device):
    """Map a torch device / device name / face device-class string to the face's DEVICE_CLASSES names."""
    if isinstance(device, str) and device in DEVICE_CLASSES_SERVED:
        return device
    try:
        name = torch.cuda.get_device_name(device) if not isinstance(device, str) else device
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 - an unknown device spec maps to its string
        name = str(device)
    n = name.upper().replace("-", " ")
    if "H200" in n:
        return "H200"
    if "H100" in n:
        if "PCIE" in n:
            return "H100_PCIe"
        if "NVL" in n:
            return "H100_NVL"
        return "H100_SXM"
    if "A100" in n:
        return "A100_PCIe_80GB" if "PCIE" in n else "A100_SXM_80GB"
    return name


_device_class = device_class          # back-compat alias


def supports(q, k, v, bias, mask, scale, lib_version, device):
    """CAPABILITY ONLY: can this build technically serve the call under its own implementation constraints?
    Proof of equality lives in CELLS.json, not here.  Returns (bool, reason)."""
    try:
        q5, k5, v5, b5, m5 = _check_inputs(q, k, v, bias, mask)
    except Refused as e:
        return False, e.reason
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as e:                                            # malformed inputs -> a reason, never an exception
        return False, f"input check failed: {type(e).__name__}: {e}"
    if str(lib_version) not in LIB_VERSIONS_SERVED:
        return False, f"cuequivariance_torch=={lib_version} not characterised by this route (serves {LIB_VERSIONS_SERVED})"
    dc = device_class(device)
    if dc not in DEVICE_CLASSES_SERVED:
        return False, f"device class {dc!r} not served (sm_80/sm_90 build: {DEVICE_CLASSES_SERVED})"
    S = q5.shape[-2]
    if S <= fallback_threshold():
        return False, (f"S={S} <= CUEQ_TRIATTN_FALLBACK_THRESHOLD={fallback_threshold()}: the library takes its torch fallback path here "
                       f"(different arithmetic); kernel path only")
    if scale is not None:
        try:
            float(scale)
        except Exception:  # hygiene: no-cuda (python float conversion)
            return False, f"scale {scale!r} is not a number"
    return True, f"cuda_mma v3 kernel path: bf16 D=32 S={S} on {dc}, cuequivariance_torch=={lib_version}"


# ---- test entry points for schedule variants (same arithmetic) ----
def attention_v0(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v0")


def attention_g1(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v1", G=1)


def attention_g2(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v1", G=2)


def attention_g4(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v1", G=4)


def attention_w8(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v1", G=2, nwarps=8)


def attention_v1(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v1")


def attention_v2(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v2")


def attention_v2_g4w8(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v2", G=4, nwarps=8)


def attention_v2_g2w8(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v2", G=2, nwarps=8)


def attention_v2_g8w8(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v2", G=8, nwarps=8)


def attention_v2_g2w4(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v2", G=2, nwarps=4)


def _mk_cfg_fn(c):
    def f(q, k, v, bias, mask=None, scale=None):
        return attention(q, k, v, bias, mask, scale, impl="v2", cfg=c)
    f.__name__ = f"attention_v2_cfg{c}"
    return f


for _c in range(9):
    globals()[f"attention_v2_cfg{_c}"] = _mk_cfg_fn(_c)


def _mk_cfg_fn3(c):
    def f(q, k, v, bias, mask=None, scale=None):
        return attention(q, k, v, bias, mask, scale, impl="v3", cfg=c)
    f.__name__ = f"attention_v3_cfg{c}"
    return f


for _c in range(9):
    globals()[f"attention_v3_cfg{_c}"] = _mk_cfg_fn3(_c)


def attention_v3(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v3")


# ----------------------------------------------------------------------------------------------------------------
# Variant builds for configuration sweeps: build_variant(**knobs) -> handle; attention_variant(handle, ...)
# knobs: nwr (row-warps: 4|8 -> BM=64|128), nsplit (n-groups of warps: 1|2|4), gw (pair-rows per warp: 1|2|4),
#        nst (cp.async stages: 2|3|4), minb (min CTAs/SM for __launch_bounds__, default by warps), skip (0|1),
#        loadskip (0|1), maxreg (ptxas -maxrregcount, optional), arch ("90"|"80;90"), extra (list of extra nvcc flags)
# The arithmetic per output element is identical for every variant (only schedule / residency changes).
# ----------------------------------------------------------------------------------------------------------------
_VARIANTS = {}


def build_variant(nwr=8, nsplit=1, gw=2, nst=2, minb=None, skip=1, loadskip=1, maxreg=None, arch=None, extra=None, verbose=False, qsmem=0):
    """Knobs: nwr row-warps (BM=16*nwr) | nsplit warp groups over pair-rows | gw pair-rows per warp (G=nsplit*gw share the bias tile) |
    nst cp.async stages | minb min CTAs/SM for __launch_bounds__ | skip/loadskip exact masked-tile skipping | maxreg | qsmem Q tile in smem |
    arch e.g. '80' | extra nvcc flags (e.g. ['-DTRIATTN_PERF=8'] for attribution builds, NOT exact)."""
    import subprocess, hashlib
    key = (nwr, nsplit, gw, nst, minb, skip, loadskip, maxreg, arch, tuple(extra or ()), qsmem)
    if key in _VARIANTS:
        return _VARIANTS[key]
    src = open(_SRC_V3, "rb").read()
    flags = ["-DTRIATTN_VARIANT", f"-DVAR_NWR={nwr}", f"-DVAR_NSPLIT={nsplit}", f"-DVAR_GW={gw}", f"-DVAR_NST={nst}",
             f"-DTRIATTN_SKIP={skip}", f"-DTRIATTN_LOADSKIP={loadskip}", f"-DTRIATTN_QSMEM={int(qsmem)}"]
    if minb is not None:
        flags.append(f"-DTRIATTN_MINB(NW)={minb}")
    if maxreg:
        flags += ["-Xptxas", f"-maxrregcount={maxreg}"]
    flags += list(extra or [])
    archs = (arch or os.environ.get("TRIATTN_EXACT_ARCHS", "80;90")).split(";")
    h = hashlib.sha256(src + " ".join(flags + archs).encode()).hexdigest()[:16]
    so = os.path.join(_cache_dir(), f"cuda_mma_v3var_{h}.so")
    if not os.path.exists(so):
        gencode = []
        for a_ in archs:
            gencode += ["-gencode", f"arch=compute_{a_},code=sm_{a_}"]
        cmd = ["nvcc", "-O3", "-std=c++17", "--shared", "-Xcompiler", "-fPIC", "--fmad=false"] + gencode + flags + ["-o", so + ".tmp", _SRC_V3]
        if verbose:
            cmd.insert(1, "-Xptxas"); cmd.insert(2, "-v")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("nvcc failed: " + r.stderr[-4000:])
        if verbose:
            print(r.stderr[-3000:])
        os.replace(so + ".tmp", so)
    lib = ctypes.CDLL(so)
    lib.triattn_v3var_launch.argtypes = [ctypes.POINTER(_LaunchArgsV2)]
    lib.triattn_v3var_launch.restype = ctypes.c_int
    lib.triattn_v3var_vscan.argtypes = [ctypes.POINTER(_LaunchArgsV2), ctypes.c_void_p]
    lib.triattn_v3var_vscan.restype = ctypes.c_int
    assert lib.triattn_v3var_sizeof_launchargs() == ctypes.sizeof(_LaunchArgsV2)
    handle = {"lib": lib, "G": lib.triattn_v3var_G(), "BM": lib.triattn_v3var_BM(), "threads": lib.triattn_v3var_threads(), "knobs": key, "so": so}
    _VARIANTS[key] = handle
    return handle


def attention_variant(handle, q, k, v, bias, mask=None, scale=None, force_kernel=False):
    """Run a build_variant() kernel with the same input checks / refusals as attention()."""
    q5, k5, v5, b5, m5 = _check_inputs(q, k, v, bias, mask)
    B, N, H, S, D = q5.shape
    if not force_kernel and S <= fallback_threshold():
        raise Refused(f"S={S} <= CUEQ_TRIATTN_FALLBACK_THRESHOLD={fallback_threshold()}: the library takes its torch fallback path here")
    if scale is None:
        scale = default_scale(D)
    lib = handle["lib"]
    b5 = _stage_bias(b5)
    out = torch.empty((B, N, H, S, D), dtype=torch.bfloat16, device=q5.device)
    a = _LaunchArgsV2()
    _fill_launch_args(a, q5, k5, v5, b5, m5, out, scale)
    a.G = handle["G"]; a.nwarps = handle["threads"] // 32
    with torch.cuda.device(q5.device):
        if m5 is not None and (S + handle["BM"] - 1) // handle["BM"] >= VSCAN_MIN_QBLOCKS:
            vb = torch.empty(B * N * H * ((S + 63) // 64), dtype=torch.uint8, device=q5.device)
            rc = lib.triattn_v3var_vscan(ctypes.byref(a), ctypes.c_void_p(vb.data_ptr()))
            if rc != 0:
                raise RuntimeError(f"variant vscan failed rc={rc}")
            a.vbad = vb.data_ptr()
        rc = lib.triattn_v3var_launch(ctypes.byref(a))
    if rc != 0:
        raise RuntimeError(f"variant launch failed rc={rc} knobs={handle['knobs']}")
    return out


def make_variant_fn(**knobs):
    """Candidate factory: an attention(q, k, v, bias, mask=None, scale=None) callable bound to build_variant(**knobs)."""
    h = build_variant(**knobs)

    def f(q, k, v, bias, mask=None, scale=None):
        return attention_variant(h, q, k, v, bias, mask, scale)
    f.__name__ = "attention_variant_" + "_".join(f"{k_}{v_}" for k_, v_ in sorted(knobs.items()))
    return f


def _env_variant():
    """TRIATTN_EXACT_VARIANT='nwr=8,gw=2,nst=3' -> module attribute `attention_env_variant` uses that build (command-line use)."""
    spec = os.environ.get("TRIATTN_EXACT_VARIANT")
    if not spec:
        return None
    knobs = {}
    for kv in spec.split(","):
        k_, v_ = kv.split("=")
        knobs[k_.strip()] = int(v_) if v_.strip().lstrip("-").isdigit() else v_.strip()
    return make_variant_fn(**knobs)


def attention_env_variant(q, k, v, bias, mask=None, scale=None):
    f = globals().get("_ENV_VARIANT_FN")
    if f is None:
        f = _env_variant()
        if f is None:
            raise Refused("TRIATTN_EXACT_VARIANT not set")
        globals()["_ENV_VARIANT_FN"] = f
    return f(q, k, v, bias, mask, scale)


def _mk_v4_fn(c, cxv):
    def f(q, k, v, bias, mask=None, scale=None):
        return attention(q, k, v, bias, mask, scale, impl="v4", cfg=c, cx=cxv)
    f.__name__ = f"attention_v4_cfg{c}_cx{cxv}"
    return f


for _c in (0, 2, 3, 4, 5):
    for _cx in (1, 2, 4):
        globals()[f"attention_v4_cfg{_c}_cx{_cx}"] = _mk_v4_fn(_c, _cx)


def attention_v4(q, k, v, bias, mask=None, scale=None):
    return attention(q, k, v, bias, mask, scale, impl="v4")
