"""exactln — a bit-for-bit replica of torch's CUDA ``layer_norm`` forward (ATen ``vectorized_layer_norm_kernel<T, float>``, torch 2.12.0) on a
faster schedule: one warp per row, many rows per CTA, rows gathered through the input's own leading strides (no ``.contiguous()`` copy of a
transposed pair tensor), gamma/beta held in registers.  The per-element arithmetic — the online Welford update, the shuffle-down combine tree of
the (32, 4) CTA ATen launches per row, the IEEE reciprocal / division, the non-FTZ ``rsqrtf``, the fused affine — is the sequence ATen's sm_90
binary executes, written with explicit round-to-nearest intrinsics (``exactln_fwd.cu`` carries the recovered specification in its header).
Same operands, same operations, same order => the same bits for every row, at any row count and any leading layout.

Served (``plan``): CUDA tensors, x fp32 or bf16 with ``x.stride(-1) == 1``, normalized over the last dimension only, C % 4 == 0 and
4 <= C <= 1024, weight/bias both given or both absent (fp32, or bf16 when x is bf16 — ``F.layer_norm``'s own dtype rule), every row start
16-byte aligned (8 for bf16) — exactly the conditions under which ATen takes its vectorized kernel (a contiguous-but-misaligned input takes ATen's
``RowwiseMomentsCUDAKernel`` pair, another arithmetic: refused here by name, ``aten_rowwise_path``).  A leading layout that is not an
(outer, inner) pair of strides is copied contiguous first (what ATen does for every non-contiguous input; a copy changes no value).

Compiled at first use per (C, affine, dtype) with NVRTC through ``cuda.bindings`` (pinned in the kit's lock; no nvcc, no new dependency) -- or,
where that package is absent, through the same libraries bound with ctypes (``cubind.py``) -- loaded and launched through the CUDA driver API
on torch's current stream.  ``precompile(widths)`` builds a set up front.  ``prebuilt/`` ships the (form, C, affine) instantiations the provider's cells
serve as one CUBIN module per arch (index.json: source sha256, options, lowered names; build_prebuilt.py): where the bundle carries the kernel
a serving process loads it and compiles nothing (``facts()["prebuilt_hits"]``; ``OPT_CORE_EXACTLN_PREBUILT=0`` forces the compile path).
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import tempfile
import threading
from typing import Any, Dict, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(HERE, "exactln_fwd.cu")
MAX_C = 1024
BLOCK_THREADS = 256                     # 8 warps per CTA
BLOCKS_PER_SM = 8                       # resident CTAs per SM the grid is sized for (grid-stride beyond)
TORCH_PIN = "2.12.0"                    # the torch whose ATen kernel arithmetic exactln_fwd.cu replicates (checked by the adapter, reported on the lever line)

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {"module_src": None, "kernels": {}, "cubin": {}, "sm_count": None, "cc": None, "compiles": 0, "compile_s": 0.0, "disk_hits": 0, "bindings": None,
                          "prebuilt_hits": 0, "prebuilt_modules": {}, "prebuilt_index": None, "prebuilt_off": None}
BINDINGS_ENV = "OPT_CORE_EXACTLN_BINDINGS"      # "" | "ctypes": force the ctypes bindings (cubind.py) even where cuda.bindings is importable
PREBUILT_ENV = "OPT_CORE_EXACTLN_PREBUILT"      # "" | "0": never serve the shipped CUBIN bundles (prebuilt/): compile through NVRTC / the disk cache as before
PREBUILT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prebuilt")   # index.json + exactln_fwd-sm<cc>.cubin: every (form, C, affine) the provider's
                                                                                        # cells serve, compiled ONCE from exactln_fwd.cu (build_prebuilt.py) -- a serving
                                                                                        # process loads a module and never compiles (facts()["prebuilt_hits"])


def _bindings(name: str):
    """``cuda.bindings.<name>`` (driver | nvrtc) when that package is importable, else the same entry points bound through ctypes
    (``cubind`` beside this file: libcuda + the libnvrtc torch loads) -- the images without the cuda-python wheel are served too; the
    compiler, source and options are identical either way, so are the bits.  ``facts()["bindings"]`` says which serves."""
    if os.environ.get(BINDINGS_ENV, "") != "ctypes":
        try:
            import importlib
            mod = importlib.import_module("cuda.bindings." + name)
            _STATE["bindings"] = _STATE["bindings"] or "cuda.bindings"
            return mod
        except ImportError:
            pass
    from . import cubind
    _STATE["bindings"] = "ctypes"
    return getattr(cubind, name)
STATS: Dict[str, int] = {}
CACHE_ENV = ("MODEL_OPT_JIT_ROOT", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR")   # the CUBIN disk cache lives under the kit's own JIT root words (an `exactln/` sibling of the Triton / torch-extension caches), else ~/.cache/exactln — no new environment name (REVIEW A14.3); keyed by sha256(source, name expression, options, nvrtc version)


class Unsupported(Exception):
    """A call the replica does not serve, by reason word (the caller takes torch's own layer_norm for it)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _cnt(k: str, n: int = 1) -> None:
    STATS[k] = STATS.get(k, 0) + n


# ----------------------------------------------------------------------------------------------------------------- driver / NVRTC plumbing
def _cu():
    return _bindings("driver")


def _check(res, what: str):
    """cuda.bindings calls return (err, *values); raise by name on error, return the values."""
    err = res[0]
    if int(err) != 0:
        raise RuntimeError(f"exactln: {what} failed: {err}")
    return res[1:] if len(res) > 1 else ()


def _source() -> bytes:
    if _STATE["module_src"] is None:
        with open(SOURCE_FILE, "rb") as f:
            _STATE["module_src"] = f.read()
    return _STATE["module_src"]


def _device_facts() -> Tuple[str, int]:
    if _STATE["cc"] is None:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        _STATE["cc"] = f"{p.major}{p.minor}"
        _STATE["sm_count"] = int(p.multi_processor_count)
    return _STATE["cc"], _STATE["sm_count"]


_T = {"float32": "float", "bfloat16": "exactln::bf16_t"}


def _name_expr(tin: str, tpar: str, tout: str, C: int, affine: int, rpw: int) -> str:
    if tin == "resid":                       # the fused residual-add + LayerNorm kernel: key ("resid", TU, TLN, C, affine, rpw)
        return f"exactln::exactln_resid_fwd<{_T[tpar]}, {_T[tout]}, {int(C)}, {int(affine)}, {int(rpw)}>"
    return f"exactln::exactln_fwd<{_T[tin]}, {_T[tpar]}, {_T[tout]}, {int(C)}, {int(affine)}, {int(rpw)}>"


_CACHE_DIRS: dict = {}


def _private_cache_dir(path: str, tag: str, own_only: bool = False) -> str:
    """``path`` (created 0700 when absent) when nothing another account could have written would be loaded from it; else — one stderr line:
    what, why, the fix — a fresh directory private to this process, where its build products are compiled again. Refused: a directory (or,
    when group or other can enter it, a file in it) that is writable by group or other, or whose owner is neither this uid nor root — uid 0
    accepts any owner (a container's root reading a bind-mounted host directory). ``own_only`` (the per-user default under the shared
    temporary directory): this uid alone, and never a symbolic link. No digest kept beside a file would add to this: whoever can write the
    directory can rewrite the digest, so the owner / mode rule is the check."""
    import stat
    if path in _CACHE_DIRS:
        return _CACHE_DIRS[path]
    os.makedirs(path, mode=0o700, exist_ok=True)
    uid, why = os.geteuid(), None
    names = [""] + (sorted(os.listdir(path)) if os.stat(path).st_mode & 0o011 else [])
    for name in names:
        p = os.path.join(path, name) if name else path
        st = os.lstat(p) if own_only else os.stat(p)
        if own_only and not name and (os.path.islink(p) or st.st_uid != uid):
            why = f"{p} is a symbolic link or belongs to uid {st.st_uid}, not to this process (uid {uid}); fix: remove it, or name a directory of your own"
        elif st.st_mode & 0o022 and not stat.S_ISLNK(st.st_mode):     # a link's own mode says nothing (os.stat above already followed it unless own_only)
            why = f"{p} is writable by group or other (mode {st.st_mode & 0o7777:04o}); fix: chmod go-w {p}"
        elif uid != 0 and st.st_uid not in (uid, 0):
            why = f"{p} belongs to uid {st.st_uid}, not to this process (uid {uid}) or root; fix: name a cache directory of your own, or chown {p}"
        if why is not None:
            break
    out = path
    if why is not None:
        out = tempfile.mkdtemp(prefix=tag + "-")
        print(f"[opt_core] {tag}: REFUSED cache directory {path}: {why} — nothing in it is loaded; this process compiles again, privately, in {out}",
              file=sys.stderr, flush=True)
    _CACHE_DIRS[path] = out
    return out


def _cache_dir() -> Optional[str]:
    for k in CACHE_ENV:
        v = os.environ.get(k)
        if v:
            d = os.path.join(v if k == "MODEL_OPT_JIT_ROOT" else os.path.dirname(v.rstrip("/")), "exactln")
            break
    else:
        d = os.path.join(os.path.expanduser("~"), ".cache", "exactln")
    try:                                               # cubins are loaded from here as found: never from a directory another account could write
        d = _private_cache_dir(d, "exactln")
        return d if os.access(d, os.W_OK) else None
    except OSError:
        return None


def is_compiled(tin: str, tpar: str, tout: str, C: int, affine: int) -> bool:
    return (tin, tpar, tout, int(C), int(affine), _rows_per_warp(int(C))) in _STATE["kernels"]


def _nvrtc_version() -> str:
    nvrtc = _bindings("nvrtc")
    r = nvrtc.nvrtcVersion()
    return f"{r[1]}.{r[2]}" if int(r[0]) == 0 else "?"


def _prebuilt_index() -> Optional[dict]:
    """prebuilt/index.json when it exists, is switched on, and was built from THIS exactln_fwd.cu with these compile options (else None, once)."""
    import hashlib
    if _STATE["prebuilt_index"] is not None or _STATE["prebuilt_off"]:
        return _STATE["prebuilt_index"]
    why = None
    if os.environ.get(PREBUILT_ENV, "") == "0":
        why = "env"
    else:
        ipath = os.path.join(PREBUILT_DIR, "index.json")
        if not os.path.isfile(ipath):
            why = "no index.json"
        else:
            try:
                idx = json.load(open(ipath))
            except (OSError, ValueError) as e:
                idx, why = None, "index unreadable: %s" % e
            if idx is not None:
                if idx.get("source_sha256") != hashlib.sha256(_source()).hexdigest():
                    why = "built from another exactln_fwd.cu (source sha256 differs)"
                elif list(idx.get("opts", [])) != [o.decode() for o in _opts(None)]:
                    why = "built with other compile options"
                else:
                    _STATE["prebuilt_index"] = idx
    if why is not None:
        _STATE["prebuilt_off"] = why
    return _STATE["prebuilt_index"]


def _prebuilt_function(expr: str, cc: str):
    """(CUfunction, CUmodule) of `expr` from the shipped bundle for sm_<cc>, or None (no bundle for the arch / kernel not in it / load refused)."""
    import hashlib
    idx = _prebuilt_index()
    if idx is None:
        return None
    b = idx.get("bundles", {}).get("sm%s" % cc)
    if not b or expr not in b.get("kernels", {}):
        return None
    cu = _cu()
    mod = _STATE["prebuilt_modules"].get(cc)
    if mod is None:
        path = os.path.join(PREBUILT_DIR, b["file"])
        try:
            data = open(path, "rb").read()
        except OSError:
            return None
        from opt_core.gates import binary_refusal  # noqa: PLC0415
        why = binary_refusal(path, data)
        if why:
            _STATE["prebuilt_off"] = "bundle %s refused: %s" % (b["file"], why)
            return None
        if hashlib.sha256(data).hexdigest() != b.get("sha256"):
            _STATE["prebuilt_off"] = "bundle sha256 differs from index.json (%s)" % b["file"]
            return None
        torch.cuda.current_stream()
        _ensure_ctx()
        res = cu.cuModuleLoadData(data)
        if int(res[0]) != 0:                                   # e.g. a driver older than the bundle's toolkit: compile instead, say why once
            _STATE["prebuilt_off"] = "cuModuleLoadData(%s) -> %s" % (b["file"], res[0])
            return None
        mod = res[1]
        _STATE["prebuilt_modules"][cc] = mod
    res = cu.cuModuleGetFunction(mod, b["kernels"][expr].encode())
    if int(res[0]) != 0:
        return None
    return (res[1], mod)


def _opts(cc: Optional[str]):
    """The NVRTC options of every exactln compile (arch first; None = without the arch option, for the bundle index)."""
    # no --use_fast_math, no --ftz: the fp32 operations are explicit intrinsics; rsqrtf keeps its non-FTZ expansion (as ATen's build)
    tail = [b"--std=c++17", b"-default-device", b"--ftz=false", b"--prec-div=true", b"--fmad=true"]
    return ([f"--gpu-architecture=sm_{cc}".encode()] + tail) if cc else tail


def _compile(tin: str, tpar: str, tout: str, C: int, affine: int, rpw: int):
    """The CUfunction of one instantiation (cached): from the shipped CUBIN bundle for this arch when it carries the kernel (no compile), else
    NVRTC-compiled for this card (or read from the disk cache) and loaded."""
    import hashlib
    import time
    key = (tin, tpar, tout, int(C), int(affine), int(rpw))
    if key in _STATE["kernels"]:
        return _STATE["kernels"][key]
    cc, _ = _device_facts()
    t0 = time.time()
    expr = _name_expr(tin, tpar, tout, C, affine, rpw)
    pf = _prebuilt_function(expr, cc)
    if pf is not None:                                 # shipped: no compile in this process
        _STATE["kernels"][key] = pf
        _STATE["prebuilt_hits"] += 1
        return pf
    opts = _opts(cc)
    digest = hashlib.sha256(_source() + b"\0" + expr.encode() + b"\0" + b" ".join(opts) + b"\0" + _nvrtc_version().encode()).hexdigest()[:24]
    cdir = _cache_dir()
    cpath = os.path.join(cdir, f"{digest}.cubin") if cdir else None
    cubin = lowered = None
    if cpath and os.path.isfile(cpath) and os.path.isfile(cpath + ".name"):
        try:
            cubin = open(cpath, "rb").read(); lowered = open(cpath + ".name", "rb").read()
            _STATE["disk_hits"] += 1
        except OSError:
            cubin = lowered = None
    if cubin is None:
        cubin, lowered = _nvrtc(expr, opts)
        if cpath:
            try:                                       # atomic publish: another worker may be writing the same entry
                tmp = f"{cpath}.{os.getpid()}.tmp"
                for blob, dst in ((cubin, cpath), (lowered, cpath + ".name")):   # 0644 whatever the umask: what _private_cache_dir accepts next time
                    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), "wb") as f:
                        f.write(blob)
                    os.replace(tmp, dst)
            except OSError:
                pass
    cu = _cu()
    torch.cuda.current_stream()                    # torch's primary context exists and is current on this thread after any CUDA call
    _ensure_ctx()
    module = _check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")[0]
    func = _check(cu.cuModuleGetFunction(module, lowered), "cuModuleGetFunction")[0]
    _STATE["kernels"][key] = (func, module)
    _STATE["cubin"][key] = len(cubin)
    _STATE["compiles"] += 1
    _STATE["compile_s"] += time.time() - t0
    return _STATE["kernels"][key]


def _nvrtc(expr: str, opts) -> Tuple[bytes, bytes]:
    """One NVRTC compile of the carried source for the name expression `expr`: (cubin, lowered name)."""
    nvrtc = _bindings("nvrtc")
    prog = _check(nvrtc.nvrtcCreateProgram(_source(), b"exactln_fwd.cu", 0, [], []), "nvrtcCreateProgram")[0]
    try:
        _check(nvrtc.nvrtcAddNameExpression(prog, expr.encode()), "nvrtcAddNameExpression")
        err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
        if int(err) != 0:
            n = _check(nvrtc.nvrtcGetProgramLogSize(prog), "nvrtcGetProgramLogSize")[0]
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"exactln: NVRTC compile of {expr} failed:\n{log.decode(errors='replace')[:4000]}")
        lowered = _check(nvrtc.nvrtcGetLoweredName(prog, expr.encode()), "nvrtcGetLoweredName")[0]
        lowered = bytes(lowered) if not isinstance(lowered, bytes) else lowered
        n = _check(nvrtc.nvrtcGetCUBINSize(prog), "nvrtcGetCUBINSize")[0]
        cubin = b" " * n
        _check((nvrtc.nvrtcGetCUBIN(prog, cubin)[0],), "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(prog)
    return bytes(cubin), bytes(lowered)


def _ensure_ctx() -> None:
    """Make torch's primary context current on this thread for the driver-API calls (it is after any torch CUDA op on the thread)."""
    cu = _cu()
    ctx = _check(cu.cuCtxGetCurrent(), "cuCtxGetCurrent")[0]
    if int(ctx) == 0:
        dev = _check(cu.cuDeviceGet(torch.cuda.current_device()), "cuDeviceGet")[0]
        pctx = _check(cu.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")[0]
        _check((cu.cuCtxSetCurrent(pctx)[0],), "cuCtxSetCurrent")


def precompile(widths=(64, 128), dtypes=("float32",), affine=(3,)) -> Dict[str, Any]:
    """Build the kernels for `widths` x `dtypes` x `affine` now (e.g. before a CUDA-graph capture); returns compile facts."""
    for dt in dtypes:
        for C in widths:
            for a in (affine if isinstance(affine, (tuple, list)) else (affine,)):
                _compile(dt, dt, dt, int(C), int(a), _rows_per_warp(int(C)))
    return facts()


def facts() -> Dict[str, Any]:
    cc, sms = _device_facts() if torch.cuda.is_available() else (None, None)
    return {"torch": torch.__version__, "torch_pin": TORCH_PIN, "cc": cc, "sm_count": sms, "bindings": _STATE["bindings"], "compiles": _STATE["compiles"], "disk_hits": _STATE["disk_hits"],
            "prebuilt_hits": _STATE["prebuilt_hits"], "prebuilt_off": _STATE["prebuilt_off"], "cache_dir": _cache_dir(),
            "compile_s": round(_STATE["compile_s"], 2), "kernels": [(f"resid:u={k[1]}>ln={k[2]}:C{k[3]}:a{k[4]}:r{k[5]}" if k[0] == "resid" else f"{k[0]}/{k[1]}>{k[2]}:C{k[3]}:a{k[4]}:r{k[5]}") for k in _STATE["kernels"]],
            "stats": dict(STATS)}


# ----------------------------------------------------------------------------------------------------------------- the plan (eligibility)
def _rows_per_warp(C: int) -> int:
    return 2 if C <= 64 else 1


def _collapse_leading(sizes, strides) -> Optional[Tuple[int, int, int, int]]:
    """Leading dims (all but the last) -> (rows, inner, stride_outer, stride_inner) when they form at most two stride groups, else None.
    Size-1 dims are dropped; adjacent dims merge when the outer stride equals inner stride * inner size."""
    dims = [(int(s), int(st)) for s, st in zip(sizes, strides) if int(s) != 1]
    rows = 1
    for s, _ in dims:
        rows *= s
    if not dims:
        return 1, 1, 0, 0
    groups = [list(dims[-1])]                      # innermost leading dim first
    for s, st in reversed(dims[:-1]):
        gs, gst = groups[-1]
        if st == gst * gs:
            groups[-1][0] = gs * s                 # merge into the current group
        else:
            groups.append([s, st])
    if len(groups) == 1:
        return rows, rows, 0, groups[0][1]
    if len(groups) == 2:
        (inner, st_in), (outer, st_out) = groups[0], groups[1]
        return rows, inner, st_out, st_in
    return None


def plan(x: torch.Tensor, normalized_shape, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], widen: bool = False) -> Dict[str, Any]:
    """The launch plan for a call, or Unsupported(reason) — the reasons are the census words of the caller.  ``widen``: the autocast form of a
    call on a bf16 x — parameters fp32, arithmetic on float(x), fp32 output (ATen runs its float kernel on autocast's fp32 copy of x; the replica
    reads the bf16 storage and widens in the load, which is that copy's value)."""
    if not x.is_cuda:
        raise Unsupported("not_cuda")
    ns = tuple(int(v) for v in (normalized_shape if isinstance(normalized_shape, (tuple, list, torch.Size)) else (normalized_shape,)))
    if len(ns) != 1 or x.dim() < 1 or int(x.shape[-1]) != ns[0]:
        raise Unsupported("normalized_shape")        # ATen serves multi-dim normalized shapes with the same kernel over N = prod; not needed here
    C = ns[0]
    if C % 4 != 0 or C < 4 or C > MAX_C:
        raise Unsupported(f"width:{C}")
    if x.dtype == torch.float32:
        tin, esize = "float32", 4
    elif x.dtype == torch.bfloat16:
        tin, esize = "bfloat16", 2
    else:
        raise Unsupported(f"dtype:{x.dtype}".replace("torch.", ""))
    if widen and tin != "bfloat16":
        raise Unsupported("widen_form")
    pdtype, psize = (torch.float32, 4) if widen else (x.dtype, esize)
    tpar = "float32" if widen else tin
    tout = "float32" if widen else tin
    for name, t in (("weight", weight), ("bias", bias)):
        if t is not None:
            if not t.is_cuda or t.dtype != pdtype or tuple(t.shape) != (C,):
                raise Unsupported(f"{name}_form")     # F.layer_norm itself raises on a dtype mismatch; never reached from nn.LayerNorm
            if not t.is_contiguous():
                raise Unsupported(f"{name}_noncontig")  # ATen: expect_contiguous copies it; rare — not served
            if t.data_ptr() % (4 * psize) != 0:
                raise Unsupported("aten_rowwise_path")  # ATen's can_vectorize(gamma/beta) fails -> RowwiseMoments arithmetic
    affine = (1 if weight is not None else 0) | (2 if bias is not None else 0)
    if x.numel() == 0:
        raise Unsupported("empty")
    if x.stride(-1) != 1 and x.shape[-1] != 1:
        copy = True                                    # ATen copies; so do we (below)
        lay = None
    else:
        lay = _collapse_leading(x.shape[:-1], x.stride()[:-1])
        copy = lay is None
    align = 4 * esize
    if not copy:
        rows, inner, st_out, st_in = lay
        aligned = (x.data_ptr() % align == 0) and (st_in * esize) % align == 0 and (st_out * esize) % align == 0
        if not aligned:
            if x.is_contiguous() and not widen:
                raise Unsupported("aten_rowwise_path")  # contiguous but misaligned: ATen does NOT copy and takes the non-vectorized kernels
            copy = True                                # non-contiguous (or autocast's fresh fp32 copy): ATen works on a fresh aligned buffer; so do we
    if copy:
        rows = 1
        for s in x.shape[:-1]:
            rows *= int(s)
        inner, st_out, st_in = rows, 0, C
    if rows >= 2 ** 31 - 2 ** 20:
        raise Unsupported("rows_int32")
    return {"C": C, "tin": tin, "tpar": tpar, "tout": tout, "affine": affine, "rows": int(rows), "inner": int(inner), "stride_outer": int(st_out),
            "stride_inner": int(st_in), "copy": bool(copy), "rpw": _rows_per_warp(C), "widen": bool(widen)}


# ----------------------------------------------------------------------------------------------------------------- the launch
def layer_norm(x: torch.Tensor, normalized_shape, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None,
               eps: float = 1e-5, widen: bool = False, out_dtype: Optional[torch.dtype] = None, _plan: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps) bit for bit (see module doc for the served set; raises Unsupported
    otherwise).  ``widen``: the bf16-autocast form (bf16 x, fp32 parameters, fp32 arithmetic and output).  ``out_dtype=torch.bfloat16`` on an
    fp32 result: the caller's following ``.to(torch.bfloat16)`` fused into the store (round-to-nearest-even, the cast's own rounding)."""
    sig = call_signature(x, normalized_shape, weight, bias, widen, out_dtype)
    ent = _PLANS.get(sig)
    if ent is None:
        p = _plan if _plan is not None else plan(x, normalized_shape, weight, bias, widen=widen)
        tout_ = p["tout"] if out_dtype is None else {torch.float32: "float32", torch.bfloat16: "bfloat16"}[out_dtype]
        func_, _m = _compile(p["tin"], p["tpar"], tout_, p["C"], p["affine"], p["rpw"])
        blocks_, row_step_ = _grid(p["rows"], p["rpw"])
        ent = (p, _Launcher(func_, _LN_SPEC), blocks_, row_step_, {"float32": torch.float32, "bfloat16": torch.bfloat16}[tout_])
        if len(_PLANS) < 4096:
            _PLANS[sig] = ent
    p, launcher, blocks, row_step, ydt = ent
    C, rows = p["C"], p["rows"]
    if p["copy"]:
        x = x.contiguous()
        _cnt("copied_input")
    y = torch.empty(x.shape, dtype=ydt, device=x.device)
    g = weight if weight is not None else x
    b = bias if bias is not None else x
    stream = torch.cuda.current_stream(x.device)
    with torch.cuda.device(x.device):
        launcher.launch((x.data_ptr(), g.data_ptr(), b.data_ptr(), y.data_ptr(), int(rows), int(p["inner"]), int(p["stride_outer"]), int(p["stride_inner"]),
                         float(eps), int(row_step)), blocks, stream.cuda_stream)
    _cnt(f"served:C{C}")
    return y


# ----------------------------------------------------------------------------------------------------------------- the residual add fused with the next LayerNorm [E4]
def resid_layer_norm(z: torch.Tensor, u: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], eps: float = 1e-5,
                     ln_transposed: bool = False, ln_dtype: torch.dtype = torch.bfloat16, precompiled_only: bool = False):
    """(z + u, LN) in one pass over the pair tensor: ``z_new = z + u`` — torch's promote-add of the bf16 (or fp32) update ``u`` onto the fp32
    ``z`` (one round-to-nearest FADD per element: the same bits as ``torch.add``) — and ``LN = layer_norm(z_new[.transpose(-2, -3)], (C,),
    weight, bias, eps).to(ln_dtype)`` by the replica on the fp32 sums (the bits the next block's own LayerNorm call produces reading z_new back).
    ``u`` may be any (outer, inner)-strided row layout of z's shape (the ending node's ``y.transpose(-2, -3)`` view is gathered, no copy);
    ``ln_transposed`` scatters the LayerNorm rows in the transposed pair order (the ending node's input), contiguous.  z: fp32 CUDA, contiguous,
    [..., N1, N2, C]; a leading batch > 1 with the transposed form is served slice by slice.  Returns (z_new, ln).  Raises Unsupported(reason)."""
    rsig = ("resid", tuple(z.shape), tuple(z.stride()), z.dtype, z.data_ptr() & 15, tuple(u.shape), tuple(u.stride()), u.dtype, u.data_ptr() & 15, z.device.index,
            None if weight is None else weight.data_ptr(), None if bias is None else bias.data_ptr(), bool(ln_transposed), ln_dtype)
    ent = _PLANS.get(rsig)
    if ent is not None:                                   # this exact call form was planned (and its kernel built) before
        rows, C, u_inner, u_so, u_si, l_inner, l_so, l_si, launcher, blocks, row_step, ln_shape = ent
        z_new = torch.empty_like(z)
        ln = torch.empty(ln_shape, dtype=ln_dtype, device=z.device)
        g = weight if weight is not None else z
        b = bias if bias is not None else z
        stream = torch.cuda.current_stream(z.device)
        with torch.cuda.device(z.device):
            launcher.launch((z.data_ptr(), u.data_ptr(), g.data_ptr(), b.data_ptr(), z_new.data_ptr(), ln.data_ptr(), rows, u_inner, u_so, u_si,
                             l_inner, l_so, l_si, float(eps), row_step), blocks, stream.cuda_stream)
        _cnt(f"resid:C{C}")
        return z_new, ln
    if not (z.is_cuda and u.is_cuda and z.dtype == torch.float32 and u.dtype in (torch.bfloat16, torch.float32)):
        raise Unsupported("resid_form")
    if tuple(u.shape) != tuple(z.shape) or z.dim() < 3:
        raise Unsupported("resid_shape")
    C = int(z.shape[-1])
    if C % 4 != 0 or C < 4 or C > MAX_C:
        raise Unsupported(f"width:{C}")
    if not z.is_contiguous() or z.data_ptr() % 16 != 0:
        raise Unsupported("resid_z_layout")
    for name, t in (("weight", weight), ("bias", bias)):
        if t is not None and (t.dtype != torch.float32 or tuple(t.shape) != (C,) or not t.is_contiguous() or t.data_ptr() % 16 != 0):
            raise Unsupported(f"{name}_form")
    affine = (1 if weight is not None else 0) | (2 if bias is not None else 0)
    rows = z.numel() // C
    if rows == 0:
        raise Unsupported("empty")
    if rows >= 2 ** 31 - 2 ** 20:
        raise Unsupported("rows_int32")
    usize = 2 if u.dtype == torch.bfloat16 else 4
    if u.stride(-1) != 1:
        raise Unsupported("resid_u_layout")
    lay = _collapse_leading(u.shape[:-1], u.stride()[:-1])
    if lay is None:
        raise Unsupported("resid_u_layout")
    _, u_inner, u_so, u_si = lay
    if u.data_ptr() % (4 * usize) or (u_so * usize) % (4 * usize) or (u_si * usize) % (4 * usize):
        raise Unsupported("resid_u_align")
    N1, N2 = int(z.shape[-3]), int(z.shape[-2])
    tu = "bfloat16" if u.dtype == torch.bfloat16 else "float32"
    tln = {torch.bfloat16: "bfloat16", torch.float32: "float32"}[ln_dtype]
    rpw = _rows_per_warp(C)
    key = ("resid", tu, tln, C, affine, rpw)
    if precompiled_only and key not in _STATE["kernels"]:
        raise Unsupported("capturing_uncompiled")
    if ln_transposed and rows != N1 * N2:
        # a leading batch (templates, diffusion multiplicity): one launch per [N1, N2, C] slice — the transposed scatter is a 2-level row map per slice
        lead = rows // (N1 * N2)
        zf, uf = z.reshape(lead, N1, N2, C), u.reshape(lead, N1, N2, C) if u.is_contiguous() else None
        z_new = torch.empty_like(z); ln = torch.empty(tuple(z.shape[:-3]) + (N2, N1, C), dtype=ln_dtype, device=z.device)
        znf, lnf = z_new.view(lead, N1, N2, C), ln.view(lead, N2, N1, C)
        for i in range(lead):
            ui = uf[i] if uf is not None else u[(slice(None),) * 0 + tuple(int(v) for v in _unravel(i, z.shape[:-3]))]
            zi, li = resid_layer_norm(zf[i], ui, weight, bias, eps, ln_transposed=True, ln_dtype=ln_dtype, precompiled_only=precompiled_only)
            znf[i].copy_(zi); lnf[i].copy_(li)
        _cnt("resid_batched_slices", lead)
        return z_new, ln
    if ln_transposed:
        l_inner, l_so, l_si = N2, C, N1 * C             # input row (i, j) -> output row (j, i) of the contiguous [.., N2, N1, C] result
        ln_shape = tuple(z.shape[:-3]) + (N2, N1, C)
    else:
        l_inner, l_so, l_si = rows, 0, C
        ln_shape = tuple(z.shape)
    func, _module = _compile(*key)
    blocks, row_step = _grid(rows, rpw)
    launcher = _Launcher(func, _RESID_SPEC)
    if len(_PLANS) < 4096 and (not precompiled_only):
        _PLANS[rsig] = (int(rows), int(C), int(u_inner), int(u_so), int(u_si), int(l_inner), int(l_so), int(l_si), launcher, blocks, row_step, ln_shape)
    z_new = torch.empty_like(z)
    ln = torch.empty(ln_shape, dtype=ln_dtype, device=z.device)
    g = weight if weight is not None else z
    b = bias if bias is not None else z
    stream = torch.cuda.current_stream(z.device)
    with torch.cuda.device(z.device):
        launcher.launch((z.data_ptr(), u.data_ptr(), g.data_ptr(), b.data_ptr(), z_new.data_ptr(), ln.data_ptr(), int(rows), int(u_inner), int(u_so), int(u_si),
                         int(l_inner), int(l_so), int(l_si), float(eps), int(row_step)), blocks, stream.cuda_stream)
    _cnt(f"resid:C{C}")
    return z_new, ln


def precompile_resid(widths=(128,), u_dtypes=("bfloat16",), ln_dtypes=("bfloat16",), affine=(3,)) -> Dict[str, Any]:
    for C in widths:
        for tu in u_dtypes:
            for tl in ln_dtypes:
                for a in affine:
                    _compile("resid", tu, tl, int(C), int(a), _rows_per_warp(int(C)))
    return facts()


def _unravel(i: int, shape) -> tuple:
    out = []
    for d in reversed([int(v) for v in shape]):
        out.append(i % d); i //= d
    return tuple(reversed(out))


# ----------------------------------------------------------------------------------------------------------------- fast launch path (plan cache + prepacked kernel parameters)
class _Launcher:
    """cuLaunchKernel with a prebuilt void** parameter block: per call only the argument VALUES are poked (pointers / ints), no tuple parsing.
    One instance per cached call signature; the argument layout is fixed by `spec` = sequence of ctypes types."""

    __slots__ = ("func", "store", "params", "addr", "n")

    def __init__(self, func, spec):
        self.func = func
        self.store = [t() for t in spec]
        self.n = len(spec)
        arr = (ctypes.c_void_p * self.n)()
        for i, c in enumerate(self.store):
            arr[i] = ctypes.addressof(c)
        self.params = arr
        self.addr = ctypes.addressof(arr)

    def launch(self, values, blocks: int, stream_handle) -> None:
        st = self.store
        for i in range(self.n):
            st[i].value = values[i]
        _check((_cu().cuLaunchKernel(self.func, blocks, 1, 1, BLOCK_THREADS, 1, 1, 0, stream_handle, self.addr, 0)[0],), "cuLaunchKernel")


_LN_SPEC = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_double, ctypes.c_int)
_RESID_SPEC = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong,
               ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_double, ctypes.c_int)
_PLANS: Dict[tuple, Any] = {}          # call signature -> (plan, launcher, blocks, row_step, out torch dtype)


def _grid(rows: int, rpw: int):
    _, sms = _device_facts()
    warps_per_block = BLOCK_THREADS // 32
    warps = (rows + rpw - 1) // rpw
    blocks = max(1, min((warps + warps_per_block - 1) // warps_per_block, sms * BLOCKS_PER_SM))
    return blocks, blocks * warps_per_block * rpw


def call_signature(x: torch.Tensor, normalized_shape, weight, bias, widen: bool, out_dtype) -> tuple:
    """Everything plan() depends on, as a hashable key (shape, strides, dtypes, 16-byte alignment classes, parameter identity)."""
    return (tuple(x.shape), tuple(x.stride()), x.dtype, x.data_ptr() & 15, x.device.index,
            None if weight is None else (weight.data_ptr(), weight.dtype), None if bias is None else (bias.data_ptr(), bias.dtype),
            tuple(normalized_shape) if isinstance(normalized_shape, (tuple, list, torch.Size)) else (int(normalized_shape),), bool(widen), out_dtype)
