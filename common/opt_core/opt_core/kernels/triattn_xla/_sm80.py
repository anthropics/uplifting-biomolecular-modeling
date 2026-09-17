"""Row cuda_80: the sm_80 member of kernels/triattn/triattn_native's kernel family (package generation 11, kernel directory cuda_80: mma.sync /
ldmatrix / cp.async multistage ring; max-free streaming softmax in exp2 units, fp32 pair bias staged once per call in MMA-fragment order, per-row
live key-tile ranges for masks, an exact SAFE pass over the CTA tiles the hot pass lists; two CTA geometries per head dim chosen by S) built as
bin/cuda/sm_80/libtriattn_sm80_xla.so (the kernel directory's device header, launch template and instantiation units compiled unmodified; the torch
binding's host side restated over raw pointers in csrc/triattn_sm80_xla.cu; C entry) and called through the launcher's `triattn_xla_sm80_fwd` FFI
target.  bf16, head_dim 16 / 32 / 64, S_q == S_k, bias fp32 | bf16, key mask or none, compute capability 8.0 (built and measured for sm_80; other
8.x devices are refused by name).  Forward-only (refused by name under differentiation); the same library serves the differentiable row's forward
with its per-row log2-sum-exp output (module _vjp, word cuda80_lse)."""
import ctypes
import os
from typing import Dict, Optional

from . import PKG_DIR, FALLBACK, Refused, binaries
from ...gates import binary_refusal

_STATE: Dict[str, object] = {}
_CALLS: Dict[str, object] = {}
SERVED = {"calls": 0}
TARGET = "triattn_xla_sm80_fwd"
HEAD_DIMS = (16, 32, 64)
FIX_LIST = 3
# CTA geometry per head dim (== cuda_80/csrc/launch_sm80.cuh Geo / GeoS): rows per CTA R = WR*RW, queries per CTA BM = 32*QG; "big" = large-S, "small" = small-S tiles
GEOMETRY = {16: {"big": (2, 128), "small": (4, 64)}, 32: {"big": (4, 128), "small": (4, 64)}, 64: {"big": (1, 128), "small": (2, 64)}}


def lib_entry() -> Optional[dict]:
    for b in binaries("cuda"):
        if b.get("role") == "sm80_fwd":
            return b
    return None


def fix_elems(B: int, N: int, H: int, S: int, D: int) -> int:
    """int32 entries of the fix buffer (== the library's triattn_sm80_xla_fix_elems: FIX_LIST + 3 x the CTA-tile count of the larger geometry)."""
    def tiles(R, BM):
        return (-(-S // BM)) * (-(-N // R)) * B * H
    g = GEOMETRY[int(D)]
    return FIX_LIST + 3 * max(tiles(*g["big"]), tiles(*g["small"]))


def cannot_serve(cc: str, dtype: str, D: int, SQ: int, SK: int, N: int, H: int, B: int, has_mask: bool, bias_dtype=None) -> Optional[str]:
    if cc != "8.0":
        return "compute capability %s (the binary is sm_80 and measured on 8.0 only)" % cc
    if dtype != "bf16":
        return "dtype %s (bf16 only)" % dtype
    if D not in HEAD_DIMS:
        return "head_dim %d (16 / 32 / 64)" % D
    if SQ != SK:
        return "SQ %d != SK %d (keys == queries only)" % (SQ, SK)
    if not (1 <= SQ <= 65536):
        return "S out of range (1..65536)"
    if B * H > 65535 or -(-N // 1) > 65535:
        return "grid limits (B*H <= 65535, N <= 65535)"
    s64, s128 = -(-SQ // 64) * 64, -(-SQ // 128) * 128
    if B * H * s128 * s64 * 4 > 8 << 30:
        return "staged bias %d MiB exceeds 8 GiB (B*H*ceil128(S)*ceil64(S)*4 B)" % ((B * H * s128 * s64 * 4) >> 20)
    if bias_dtype is not None:
        bd = str(getattr(bias_dtype, "name", bias_dtype)).lower()
        if bd not in ("float32", "bfloat16", "fp32", "bf16"):
            return "bias dtype %s (f32 / bf16)" % bd
    ent = lib_entry()
    if ent is None:
        return "no sm_80 library (role sm80_fwd) in this package's manifest"
    if not os.path.isfile(os.path.join(PKG_DIR, ent["file"])):
        return "sm_80 library %s missing on disk" % ent["file"]
    why = binary_refusal(os.path.join(PKG_DIR, ent["file"]))
    if why:
        return "sm_80 library %s refused: %s" % (ent["file"], why)
    err = _STATE.get("load_error")
    if err:
        return str(err)
    return None


def load():
    """Load the sm_80 library and install its entry in the launcher; cached; failures are Refused by name (and remembered for cannot_serve)."""
    if "lib" in _STATE:
        return _STATE["lib"]
    from . import _launch
    launcher = _launch.load()
    if not hasattr(launcher, "txla_set_sm80_entry"):
        _STATE["load_error"] = "this launcher build has no txla_set_sm80_entry (rebuild csrc/cubin_launch.cc)"
        raise Refused("triattn_xla: cuda_80: %s; fallback: %s" % (_STATE["load_error"], FALLBACK))
    launcher.txla_set_sm80_entry.argtypes = [ctypes.c_void_p]
    ent = lib_entry()
    if ent is None:
        raise Refused("triattn_xla: cuda_80: no sm_80 library in manifest.json; fallback: %s" % FALLBACK)
    path = os.path.join(PKG_DIR, ent["file"])
    why = binary_refusal(path)
    if why:
        _STATE["load_error"] = "%s refused: %s" % (ent["file"], why)
        raise Refused("triattn_xla: cuda_80: %s refused: %s; fallback: %s" % (path, why, FALLBACK))
    try:
        lib = ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_LOCAL", 0))
    except OSError as e:
        _STATE["load_error"] = "cannot load %s (%s)" % (ent["file"], e)
        raise Refused("triattn_xla: cuda_80: cannot load %s (%s); fallback: %s" % (path, e, FALLBACK))
    lib.triattn_sm80_xla_describe.restype = ctypes.c_char_p
    lib.triattn_sm80_xla_fix_elems.restype = ctypes.c_longlong; lib.triattn_sm80_xla_fix_elems.argtypes = [ctypes.c_int] * 5
    lib.triattn_sm80_xla_dims.restype = ctypes.c_int; lib.triattn_sm80_xla_dims.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.triattn_sm80_xla_geometry.restype = ctypes.c_int; lib.triattn_sm80_xla_geometry.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    dims = (ctypes.c_int * 3)(); n = lib.triattn_sm80_xla_dims(dims)
    built = tuple(int(dims[i]) for i in range(n))
    # the Python-side geometry table must agree with the library (fix-buffer sizing happens at trace time, without a device)
    for D in built:
        g = (ctypes.c_int * 10)()
        for small, key in ((0, "big"), (1, "small")):
            if lib.triattn_sm80_xla_geometry(D, small, g) == 0 and (int(g[0]), int(g[4])) != GEOMETRY[D][key]:
                _STATE["load_error"] = "geometry table mismatch for D %d %s: library (R %d, BM %d) vs module %s" % (D, key, int(g[0]), int(g[4]), GEOMETRY[D][key])
                raise Refused("triattn_xla: cuda_80: %s; fallback: %s" % (_STATE["load_error"], FALLBACK))
        if int(lib.triattn_sm80_xla_fix_elems(1, 700, 4, 700, D)) != fix_elems(1, 700, 4, 700, D):
            _STATE["load_error"] = "fix_elems mismatch for D %d" % D
            raise Refused("triattn_xla: cuda_80: %s; fallback: %s" % (_STATE["load_error"], FALLBACK))
    entry = ctypes.cast(lib.triattn_sm80_xla_fwd, ctypes.c_void_p).value
    launcher.txla_set_sm80_entry(ctypes.c_void_p(entry))
    _STATE["lib"] = lib; _STATE["path"] = path; _STATE["describe"] = lib.triattn_sm80_xla_describe().decode(); _STATE["dims"] = built
    return lib


def scratch_shapes(B: int, N: int, H: int, S: int, D: int, has_mask: bool, want_lse: bool = False):
    """(name, shape, dtype-name) of the outputs after `out`, in the FFI ret order: [lse], bias_staged, fix, census, then with a mask keyany, rows, maskw, rgflag."""
    s64, s128, nkt = -(-S // 64) * 64, -(-S // 128) * 128, -(-S // 64)
    sh = []
    if want_lse:
        sh.append(("lse", (B, N, H, S), "float32"))
    sh += [("bias_staged", (B * H * s128 * s64,), "float32"), ("fix", (fix_elems(B, N, H, S, D),), "int32"), ("census", (4,), "int32")]
    if has_mask:
        sh += [("keyany", (B * s64,), "uint8"), ("rows", (B * N * 4,), "int32"), ("maskw", (B * N * nkt * 2,), "int32"), ("rgflag", (B * N,), "int32")]
    return sh


def scratch_bytes(B: int, N: int, H: int, S: int, D: int, has_mask: bool, want_lse: bool = False) -> int:
    size = {"float32": 4, "int32": 4, "uint8": 1}
    tot = 0
    for _, shp, dt in scratch_shapes(B, N, H, S, D, has_mask, want_lse):
        n = 1
        for x in shp:
            n *= int(x)
        tot += n * size[dt]
    return tot


def _call(qshape, B: int, N: int, H: int, S: int, D: int, has_mask: bool, scale: float, layout: int, want_lse: bool = False):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from . import _launch
    ffi = _launch.jax_ffi_module()
    outs = [jax.ShapeDtypeStruct(tuple(int(x) for x in qshape), jnp.bfloat16)] + [jax.ShapeDtypeStruct(shp, getattr(jnp, dt)) for _, shp, dt in scratch_shapes(B, N, H, S, D, has_mask, want_lse)]
    call = _launch._ffi_call(ffi, TARGET, outs)
    flags = (1 if has_mask else 0) | (2 if layout == 1 else 0) | (4 if want_lse else 0)
    fe = fix_elems(B, N, H, S, D)

    def f(*inputs):
        return call(*inputs, scale=np.float32(scale), flags=np.int64(flags), fix_elems=np.int64(fe))
    return f


def forward(q, k, v, bias, mask_u8, scale: float, layout: int):
    load()
    B, N, D = int(q.shape[0]), int(q.shape[1]), int(q.shape[4])
    if layout == 0:
        H, S = int(q.shape[2]), int(q.shape[3])
    else:
        S, H = int(q.shape[2]), int(q.shape[3])
    has_mask = mask_u8 is not None
    key = "B%d N%d H%d S%d D%d|L%d|m%d|%s|%.10g" % (B, N, H, S, D, layout, int(has_mask), str(bias.dtype), scale)
    fn = _CALLS.get(key)
    if fn is None:
        call = _call(q.shape, B, N, H, S, D, has_mask, scale, layout)

        def raw(qq, kk, vv, bb, mm_):
            ins = [qq, kk, vv, bb] + ([mm_] if has_mask else [])
            return call(*ins)[0]

        from ._k2b import _forward_only
        fn = _forward_only(raw, "cuda_80", has_mask)
        _CALLS[key] = fn
    SERVED["calls"] += 1
    return fn(q, k, v, bias, mask_u8)


def forward_with_lse(q, k, v, bias, mask_u8, scale: float, layout: int):
    """(out, lse2): the same kernels with the per-row log2-sum-exp output (fp32 [B,N,H,S], log2 units; a fully-masked row: log2(S)) -- the differentiable
    row's forward on cc 8.0.  Unguarded (module _vjp wraps it in its custom_vjp)."""
    load()
    B, N, D = int(q.shape[0]), int(q.shape[1]), int(q.shape[4])
    if layout == 0:
        H, S = int(q.shape[2]), int(q.shape[3])
    else:
        S, H = int(q.shape[2]), int(q.shape[3])
    has_mask = mask_u8 is not None
    key = "lse|B%d N%d H%d S%d D%d|L%d|m%d|%s|%.10g" % (B, N, H, S, D, layout, int(has_mask), str(bias.dtype), scale)
    call = _CALLS.get(key)
    if call is None:
        call = _call(q.shape, B, N, H, S, D, has_mask, scale, layout, want_lse=True)
        _CALLS[key] = call
    ins = [q, k, v, bias] + ([mask_u8] if has_mask else [])
    outs = call(*ins)
    SERVED["calls"] += 1
    return outs[0], outs[1]


def status() -> dict:
    ent = lib_entry()
    return {"loaded": "lib" in _STATE, "path": _STATE.get("path"), "describe": _STATE.get("describe"), "dims": _STATE.get("dims"), "load_error": _STATE.get("load_error"),
            "library": {k: ent.get(k) for k in ("file", "sha256", "requires", "nvcc", "ptxas", "package_generation")} if ent else None,
            "served": dict(SERVED), "compiled_calls": len(_CALLS)}
