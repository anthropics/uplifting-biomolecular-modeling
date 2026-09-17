"""Row triattn_native: the sm_90a "M1" triangle-attention forward of kernels/triattn/triattn_native (package generation 11, its cuda_b kernel: three pair
rows per CTA, warp-specialized TMA producer, bias staged once per call in MMA-fragment order, a max-free hot pass + an exact SAFE pass over
the CTA tiles the hot pass lists) built as bin/cuda/sm_90a/libtriattn_m1_xla.so (kernel header unmodified; host side restated over raw
pointers in csrc/triattn_m1_xla.cu; plain C entry) and called through the launcher's `triattn_xla_m1_fwd` FFI target.  bf16, head_dim 32,
S_q == S_k, bias fp32 | bf16, key mask or none, compute capability 9.0.  Forward-only (refused by name under differentiation)."""
import ctypes
import os
from typing import Dict, Optional

from . import PKG_DIR, FALLBACK, Refused, binaries
from ...gates import binary_refusal

_STATE: Dict[str, object] = {}
_CALLS: Dict[str, object] = {}
SERVED = {"calls": 0}
TARGET = "triattn_xla_m1_fwd"


def lib_entry() -> Optional[dict]:
    for b in binaries("cuda"):
        if b.get("role") == "m1_fwd":
            return b
    return None


def cannot_serve(cc: str, dtype: str, D: int, SQ: int, SK: int, N: int, H: int, B: int, has_mask: bool, bias_dtype=None) -> Optional[str]:
    if cc != "9.0":
        return "compute capability %s (the binary is sm_90a: 9.0 only)" % cc
    if dtype != "bf16":
        return "dtype %s (bf16 only)" % dtype
    if D != 32:
        return "head_dim %d (32 only)" % D
    if SQ != SK:
        return "SQ %d != SK %d (keys == queries only)" % (SQ, SK)
    if not (1 <= SQ <= 65536):
        return "S out of range (1..65536)"
    nq = -(-SQ // 128)
    if B * H > 65535 or (-(-N // 3) + 1) > 65535:
        return "grid limits (B*H <= 65535)"
    if B * H * nq * 4 * (nq + 1) * 4096 * 4 > 8 << 30:
        return "staged bias %d MiB exceeds 8 GiB (B*H*ceil(S/128)*(4*ceil(S/128)+4)*16 KiB)" % ((B * H * nq * 4 * (nq + 1) * 4096 * 4) >> 20)
    if bias_dtype is not None:
        bd = str(getattr(bias_dtype, "name", bias_dtype)).lower()
        if bd not in ("float32", "bfloat16", "fp32", "bf16"):
            return "bias dtype %s (f32 / bf16)" % bd
    ent = lib_entry()
    if ent is None:
        return "no M1 library (role m1_fwd) in this package's manifest"
    if not os.path.isfile(os.path.join(PKG_DIR, ent["file"])):
        return "M1 library %s missing on disk" % ent["file"]
    if not os.environ.get("TRIATTN_XLA_M1_LIB"):
        why = binary_refusal(os.path.join(PKG_DIR, ent["file"]))
        if why:
            return "M1 library %s refused: %s" % (ent["file"], why)
    err = _STATE.get("load_error")
    if err:
        return str(err)
    return None


def load():
    """Load the M1 library and install its entry in the launcher; cached; failures are Refused by name (and remembered for cannot_serve)."""
    if "lib" in _STATE:
        return _STATE["lib"]
    from . import _launch
    launcher = _launch.load()
    if not hasattr(launcher, "txla_set_m1_entry"):
        _STATE["load_error"] = "this launcher build has no txla_set_m1_entry (rebuild csrc/cubin_launch.cc)"
        raise Refused("triattn_xla: triattn_native: %s; fallback: %s" % (_STATE["load_error"], FALLBACK))
    launcher.txla_set_m1_entry.argtypes = [ctypes.c_void_p]
    ent = lib_entry()
    if ent is None:
        raise Refused("triattn_xla: triattn_native: no M1 library in manifest.json; fallback: %s" % FALLBACK)
    path = os.environ.get("TRIATTN_XLA_M1_LIB") or os.path.join(PKG_DIR, ent["file"])     # TRIATTN_XLA_M1_LIB: another build of the same library (measurement A/B), by absolute path
    why = binary_refusal(path)
    if why:
        _STATE["load_error"] = "%s refused: %s" % (ent["file"], why)
        raise Refused("triattn_xla: triattn_native: %s refused: %s; fallback: %s" % (path, why, FALLBACK))
    try:
        lib = ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_LOCAL", 0))
    except OSError as e:
        _STATE["load_error"] = "cannot load %s (%s)" % (ent["file"], e)
        raise Refused("triattn_xla: triattn_native: cannot load %s (%s); fallback: %s" % (path, e, FALLBACK))
    lib.triattn_m1_xla_describe.restype = ctypes.c_char_p
    lib.triattn_m1_xla_smem_bytes.restype = ctypes.c_longlong; lib.triattn_m1_xla_smem_bytes.argtypes = [ctypes.c_int]
    entry = ctypes.cast(lib.triattn_m1_xla_fwd, ctypes.c_void_p).value
    launcher.txla_set_m1_entry(ctypes.c_void_p(entry))
    _STATE["lib"] = lib; _STATE["path"] = path; _STATE["describe"] = lib.triattn_m1_xla_describe().decode()
    return lib


def scratch_shapes(B: int, N: int, H: int, S: int, has_mask: bool):
    """(name, shape, dtype-name) of the scratch outputs the library expects, in the FFI ret order after `out`."""
    nq = -(-S // 128); W4 = 4 * (nq + 1)
    n_ctas = nq * (-(-N // 3) + 1) * B * H                      # the torch host code's fix-list bound (one spare row group)
    sh = [("bias_staged", (B * H * nq * W4 * 4096,), "float32"), ("fix", (1 + 3 * n_ctas,), "int32"), ("fix_total", (2,), "int32")]
    if has_mask:
        sh += [("words", (B * N * W4,), "int32"), ("keyany", (B * W4,), "int32"), ("rowkind", (B * N,), "uint8"), ("kcend", (B,), "int32"), ("kcstart", (B,), "int32"),
               ("rowkc0", (B * N,), "int32"), ("rowkc1", (B * N,), "int32"), ("counts", (2,), "int32")]
    return sh


def scratch_bytes(B: int, N: int, H: int, S: int, has_mask: bool) -> int:
    size = {"float32": 4, "int32": 4, "uint8": 1}
    tot = 0
    for _, shp, dt in scratch_shapes(B, N, H, S, has_mask):
        n = 1
        for x in shp:
            n *= int(x)
        tot += n * size[dt]
    return tot


def _call(qshape, B: int, N: int, H: int, S: int, has_mask: bool, scale: float, layout: int):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from . import _launch
    ffi = _launch.jax_ffi_module()
    outs = [jax.ShapeDtypeStruct(tuple(int(x) for x in qshape), jnp.bfloat16)] + [jax.ShapeDtypeStruct(shp, getattr(jnp, dt)) for _, shp, dt in scratch_shapes(B, N, H, S, has_mask)]
    call = _launch._ffi_call(ffi, TARGET, outs)
    flags = (1 if has_mask else 0) | (2 if layout == 1 else 0)

    def f(*inputs):
        return call(*inputs, scale=np.float32(scale), flags=np.int64(flags))
    return f


def forward(q, k, v, bias, mask_u8, scale: float, layout: int):
    load()
    B, N = int(q.shape[0]), int(q.shape[1])
    if layout == 0:
        H, S = int(q.shape[2]), int(q.shape[3])
    else:
        S, H = int(q.shape[2]), int(q.shape[3])
    has_mask = mask_u8 is not None
    key = "B%d N%d H%d S%d|L%d|m%d|%s|%.10g" % (B, N, H, S, layout, int(has_mask), str(bias.dtype), scale)
    fn = _CALLS.get(key)
    if fn is None:
        call = _call(q.shape, B, N, H, S, has_mask, scale, layout)

        def raw(qq, kk, vv, bb, mm_):
            ins = [qq, kk, vv, bb] + ([mm_] if has_mask else [])
            return call(*ins)[0]

        from ._k2b import _forward_only
        fn = _forward_only(raw, "triattn_native", has_mask)
        _CALLS[key] = fn
    SERVED["calls"] += 1
    return fn(q, k, v, bias, mask_u8)


def status() -> dict:
    ent = lib_entry()
    return {"loaded": "lib" in _STATE, "path": _STATE.get("path"), "describe": _STATE.get("describe"), "load_error": _STATE.get("load_error"),
            "library": {k: ent.get(k) for k in ("file", "sha256", "requires", "cutlass_tag", "nvcc", "ptxas")} if ent else None,
            "served": dict(SERVED), "compiled_calls": len(_CALLS)}
