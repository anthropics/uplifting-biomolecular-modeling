"""Row cuda_sm90a: the sm_90a CUDA triangle-attention forward (kernels/triattn/cuda_sm90a/csrc/, built as bin/cuda/sm_90a/libtriattn_mw_cuda.so with a plain C
entry) called through the launcher's `triattn_xla_cuda_fwd` FFI target; and its lse variant (libtriattn_mw_cuda_lse.so = the same sources + csrc/cuda_lse.patch,
which also stores each query row's log2-sum-exp: the differentiable row's forward on cc 9.0, forward_lse_call).  Scratch buffers (staged bias, mask tables, fix-up list) are
outputs of the FFI call that the caller discards, so XLA owns every byte."""
import ctypes
import os
from typing import Dict, Optional

from . import PKG_DIR, FALLBACK, Refused, binaries
from ...gates import binary_refusal

_STATE: Dict[str, object] = {}
_CALLS: Dict[str, object] = {}
SERVED = {"calls": 0}
R, BM, BN = 3, 128, 64           # the kernel's fixed geometry (pair rows per CTA tile, query tile, key tile) -- restated for scratch sizing; checked against the library's own fix_elems at load


def lib_entry(role: str = "fwd") -> Optional[dict]:
    """manifest entry of the CUDA library: role "fwd" (libtriattn_mw_cuda.so, row cuda_sm90a) or "fwd_lse" (libtriattn_mw_cuda_lse.so: the same kernel + a
    per-row log-sum-exp store, the differentiable row's forward on cc 9.0)."""
    for b in binaries("cuda"):
        if b.get("role", "fwd") == role:
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
    if B * H > 65535 or -(-N // R) > 65535:
        return "grid limits (B*H <= 65535, N <= 196605)"
    if bias_dtype is not None:
        bd = str(getattr(bias_dtype, "name", bias_dtype)).lower()
        if bd not in ("float32", "bfloat16", "float16", "fp32", "bf16", "fp16"):
            return "bias dtype %s (f32 / bf16 / f16)" % bd
    ent = lib_entry()
    if ent is None:
        return "no CUDA library in this package's manifest"
    if not os.path.isfile(os.path.join(PKG_DIR, ent["file"])):
        return "CUDA library %s missing on disk" % ent["file"]
    why = binary_refusal(os.path.join(PKG_DIR, ent["file"]))
    if why:
        return "CUDA library %s refused: %s" % (ent["file"], why)
    return None


def load():
    """Load the CUDA library and install its forward entry in the launcher; cached; failures are Refused by name."""
    if "lib" in _STATE:
        return _STATE["lib"]
    from . import _launch
    launcher = _launch.load()
    ent = lib_entry()
    if ent is None:
        raise Refused("triattn_xla: cuda_sm90a: no CUDA library in manifest.json; fallback: %s" % FALLBACK)
    path = os.path.join(PKG_DIR, ent["file"])
    why = binary_refusal(path)
    if why:
        raise Refused("triattn_xla: cuda_sm90a: %s refused: %s; fallback: %s" % (ent["file"], why, FALLBACK))
    try:
        lib = ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_LOCAL", 0))
    except OSError as e:
        raise Refused("triattn_xla: cuda_sm90a: cannot load %s (%s); fallback: %s" % (path, e, FALLBACK))
    lib.triattn_mw_cuda_fix_elems.restype = ctypes.c_longlong; lib.triattn_mw_cuda_fix_elems.argtypes = [ctypes.c_longlong] * 4
    lib.triattn_mw_cuda_smem_bytes.restype = ctypes.c_longlong
    lib.triattn_mw_cuda_describe.restype = ctypes.c_char_p
    if int(lib.triattn_mw_cuda_fix_elems(1, 7, 2, 300)) != fix_elems(1, 7, 2, 300):
        raise Refused("triattn_xla: cuda_sm90a: the library's tile geometry differs from this module's (fix_elems); fallback: %s" % FALLBACK)
    entry = ctypes.cast(lib.triattn_mw_cuda_fwd, ctypes.c_void_p).value
    launcher.txla_set_cuda_entry(ctypes.c_void_p(entry))
    _STATE["lib"] = lib; _STATE["path"] = path; _STATE["describe"] = lib.triattn_mw_cuda_describe().decode()
    return lib


def load_lse():
    """Load the lse CUDA library (role fwd_lse) and install its entry in the launcher; cached; failures are Refused by name."""
    if "lib_lse" in _STATE:
        return _STATE["lib_lse"]
    from . import _launch
    launcher = _launch.load()
    if not hasattr(launcher, "txla_set_cuda_lse_entry"):
        raise Refused("triattn_xla: cuda_lse: this launcher build has no txla_set_cuda_lse_entry (rebuild csrc/cubin_launch.cc)")
    ent = lib_entry("fwd_lse")
    if ent is None:
        raise Refused("triattn_xla: cuda_lse: no lse CUDA library (role fwd_lse) in manifest.json")
    path = os.path.join(PKG_DIR, ent["file"])
    why = binary_refusal(path)
    if why:
        raise Refused("triattn_xla: cuda_lse: %s refused: %s" % (ent["file"], why))
    try:
        lib = ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_LOCAL", 0))
    except OSError as e:
        raise Refused("triattn_xla: cuda_lse: cannot load %s (%s)" % (path, e))
    lib.triattn_mw_cuda_fix_elems.restype = ctypes.c_longlong; lib.triattn_mw_cuda_fix_elems.argtypes = [ctypes.c_longlong] * 4
    lib.triattn_mw_cuda_describe.restype = ctypes.c_char_p
    if int(lib.triattn_mw_cuda_fix_elems(1, 7, 2, 300)) != fix_elems(1, 7, 2, 300):
        raise Refused("triattn_xla: cuda_lse: the library's tile geometry differs from this module's (fix_elems)")
    entry = ctypes.cast(lib.triattn_mw_cuda_fwd_lse, ctypes.c_void_p).value
    launcher.txla_set_cuda_lse_entry(ctypes.c_void_p(entry))
    _STATE["lib_lse"] = lib; _STATE["path_lse"] = path; _STATE["describe_lse"] = lib.triattn_mw_cuda_describe().decode()
    return lib


def fix_elems(B: int, N: int, H: int, S: int) -> int:
    return 3 + 3 * (-(-S // BM)) * (-(-N // R)) * B * H


def _out_types(qshape, B: int, N: int, H: int, S: int, has_mask: bool, lse: bool):
    """The FFI call's outputs: out (bf16, q's shape), the scratch the library stages into (XLA-owned, discarded by the caller) and, for the lse variant,
    ret 11 = fp32 [B,N,H,S] per-row log2-sum-exp."""
    import jax
    import jax.numpy as jnp
    S64, S128 = -(-S // 64) * 64, -(-S // 128) * 128
    YG, nkt = -(-N // R), -(-S // BN)
    u8, i32, f32 = jnp.uint8, jnp.int32, jnp.float32
    one = (1,)
    outs = [jax.ShapeDtypeStruct(tuple(qshape), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, H, S128, S64), f32),
            jax.ShapeDtypeStruct((B, S64) if has_mask else one, u8),
            jax.ShapeDtypeStruct((B, N) if has_mask else one, u8),
            jax.ShapeDtypeStruct((1 + B * YG,) if has_mask else one, i32),
            jax.ShapeDtypeStruct((B * YG,) if has_mask else one, i32),
            jax.ShapeDtypeStruct((B,) if has_mask else one, i32),
            jax.ShapeDtypeStruct((B, N, nkt, 2) if has_mask else one, i32),
            jax.ShapeDtypeStruct((B, N, nkt) if has_mask else one, u8),
            jax.ShapeDtypeStruct((B, N) if has_mask else one, i32),
            jax.ShapeDtypeStruct((fix_elems(B, N, H, S) + 1,), i32)]
    if lse:
        outs.append(jax.ShapeDtypeStruct((B, N, H, S), f32))
    return outs


def forward_lse_call(B: int, N: int, H: int, S: int, D: int, has_mask: bool, scale: float, bias_dtype: str):
    """raw(q, k, v, bias, mask_or_None) -> (out, lse2) through the lse library (layout BNHSD, bf16): the differentiable row's forward on cc 9.0.
    out is the arithmetic of row cuda_sm90a (bit-identical); lse2 = fp32 [B,N,H,S] log2-sum-exp of each query row."""
    from . import _launch
    load_lse()
    key = "lse|B%d N%d H%d S%d D%d|m%d|%s|%.10g" % (B, N, H, S, D, int(has_mask), bias_dtype, scale)
    raw = _CALLS.get(key)
    if raw is None:
        call = _launch.cuda_call(_out_types((B, N, H, S, D), B, N, H, S, has_mask, True), scale, (1 if has_mask else 0) | 2)

        def raw(qq, kk, vv, bb, mm_=None):
            ins = [qq, kk, vv, bb] + ([mm_] if has_mask else [])
            r = call(*ins)
            return r[0], r[11]
        raw.cubin = "cuda_sm90a+lse"
        _CALLS[key] = raw
    return raw


def forward(q, k, v, bias, mask_u8, scale: float, layout: int):
    import jax
    import jax.numpy as jnp
    from . import _launch
    load()
    B, N = int(q.shape[0]), int(q.shape[1]); D = int(q.shape[4])
    if layout == 0:
        H, S = int(q.shape[2]), int(q.shape[3])
    else:
        S, H = int(q.shape[2]), int(q.shape[3])
    has_mask = mask_u8 is not None
    key = "B%d N%d H%d S%d|L%d|m%d|%s|%.10g" % (B, N, H, S, layout, int(has_mask), str(bias.dtype), scale)
    fn = _CALLS.get(key)
    if fn is None:
        call = _launch.cuda_call(_out_types(tuple(int(x) for x in q.shape), B, N, H, S, has_mask, False), scale, (1 if has_mask else 0) | (int(layout) << 4))

        def raw(qq, kk, vv, bb, mm_):
            ins = [qq, kk, vv, bb] + ([mm_] if has_mask else [])
            return call(*ins)[0]

        from ._k2b import _forward_only
        fn = _forward_only(raw, "cuda_sm90a", has_mask)
        _CALLS[key] = fn
    SERVED["calls"] += 1
    return fn(q, k, v, bias, mask_u8)


def status() -> dict:
    ent = lib_entry()
    lse = lib_entry("fwd_lse")
    return {"loaded": "lib" in _STATE, "path": _STATE.get("path"), "describe": _STATE.get("describe"), "manifest": {k: ent.get(k) for k in ("file", "sha256", "nvcc", "arch")} if ent else None,
            "lse": {"loaded": "lib_lse" in _STATE, "describe": _STATE.get("describe_lse"), "manifest": {k: lse.get(k) for k in ("file", "sha256", "abi_version")} if lse else None},
            "served": dict(SERVED)}
