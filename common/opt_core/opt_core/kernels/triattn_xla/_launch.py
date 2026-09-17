"""The XLA-FFI launcher library: pick the build for this jaxlib's FFI API version, load it (ctypes), register its two FFI targets with
jax.ffi once, describe launches, build ffi_call callables.  A launch is SELF-DESCRIBING: the cubin key + digest, kernel name, grid, shared
memory and the parameter recipe are attributes of the custom call, and the launcher reads the cubin from the package on first use in a process --
so an executable restored from a persistent compilation cache runs in a process that never traced it.  Every failure is a named `Refused`."""
import ctypes
import os
import re
from typing import Dict, List, Optional, Sequence

from . import PKG_DIR, FALLBACK, Refused, binaries
from ...gates import binary_refusal

TARGET_LAUNCH = "triattn_xla_launch"      # previous interface (spec registered per process); kept exported by the launcher for one version, unused here
TARGET_RUN = "triattn_xla_run"            # self-describing launch: every parameter travels as an attribute of the custom call
TARGET_CUDA = "triattn_xla_cuda_fwd"
TARGET_BGEMM = "xla_cublas_bgemm_nt"     # launcher 1.5: X[c] = A[c].B[c]^T on bf16 planes inside one buffer through the jaxlib-shipped cuBLAS (kernels/trimul_xla contraction)
TARGET_GENERIC = "xla_cubin_call"        # launcher 1.5: a kernel from a cubin under a NAMED package root, by-value struct parameters + tensor maps (kernels/trimul_xla)
_STATE: Dict[str, object] = {}
_SPECS: Dict[str, int] = {}


def jax_ffi_module():
    """jax.ffi with the four names this package uses, or Refused naming what is missing (the `n/a: jax <ver> lacks ffi` case)."""
    try:
        import jax
    except ImportError as e:
        raise Refused("triattn_xla: n/a: jax is not importable (%s); fallback: %s" % (e, FALLBACK))
    ffi = getattr(jax, "ffi", None)
    missing = [n for n in ("ffi_call", "register_ffi_target", "pycapsule", "include_dir") if ffi is None or not hasattr(ffi, n)]
    if missing:
        raise Refused("triattn_xla: n/a: jax %s lacks jax.ffi (%s missing); fallback: %s" % (getattr(jax, "__version__", "?"), ", ".join(missing) if ffi else "module jax.ffi", FALLBACK))
    return ffi


def ffi_api_version() -> str:
    """'<major>.<minor>' of the XLA-FFI API this jaxlib was built with, read from the headers it ships (jax.ffi.include_dir())."""
    if "api" in _STATE:
        return _STATE["api"]
    ffi = jax_ffi_module()
    inc = ffi.include_dir()
    path = os.path.join(inc, "xla", "ffi", "api", "c_api.h")
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError as e:
        raise Refused("triattn_xla: cannot read the XLA-FFI API header of this jaxlib (%s: %s); fallback: %s" % (path, e, FALLBACK))
    ma = re.search(r"#define XLA_FFI_API_MAJOR (\d+)", txt); mi = re.search(r"#define XLA_FFI_API_MINOR (\d+)", txt)
    if not ma or not mi:
        raise Refused("triattn_xla: XLA_FFI_API_MAJOR/MINOR not found in %s; fallback: %s" % (path, FALLBACK))
    _STATE["api"] = "%s.%s" % (ma.group(1), mi.group(1))
    return _STATE["api"]


def launcher_builds() -> List[dict]:
    return binaries("launcher")


def launcher_path(api: Optional[str] = None) -> str:
    api = api or ffi_api_version()
    override = os.environ.get("TRIATTN_XLA_LAUNCHER")            # a build outside the package (bring-up on a new jaxlib line)
    if override:
        return override
    for b in launcher_builds():
        if b.get("ffi_api_version") == api:
            p = os.path.join(PKG_DIR, b["file"])
            if os.path.isfile(p):
                why = binary_refusal(p)
                if why:
                    raise Refused("triattn_xla: launcher build %s refused: %s; fallback: %s" % (b["file"], why, FALLBACK))
                return p
            raise Refused("triattn_xla: launcher build %s listed in manifest.json is missing on disk; fallback: %s" % (b["file"], FALLBACK))
    have = ", ".join("%s (jaxlib %s)" % (b.get("ffi_api_version"), "/".join(b.get("jaxlib_lines", [])) or "?") for b in launcher_builds()) or "none"
    import jax
    raise Refused("triattn_xla: no launcher build for XLA-FFI API %s (jax %s); builds shipped: %s; fallback: %s" % (api, getattr(jax, "__version__", "?"), have, FALLBACK))


def load():
    """The loaded launcher (ctypes CDLL) with both FFI targets registered; cached."""
    if "lib" in _STATE:
        return _STATE["lib"]
    ffi = jax_ffi_module()
    path = launcher_path()
    try:
        lib = ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_LOCAL", 0))
    except OSError as e:
        raise Refused("triattn_xla: cannot load the launcher %s (%s); fallback: %s" % (path, e, FALLBACK))
    lib.txla_register.restype = ctypes.c_int
    lib.txla_register.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.txla_set_cuda_entry.argtypes = [ctypes.c_void_p]
    if hasattr(lib, "txla_set_cuda_lse_entry"):
        lib.txla_set_cuda_lse_entry.argtypes = [ctypes.c_void_p]
    lib.txla_ffi_api_version.restype = ctypes.c_char_p
    built = lib.txla_ffi_api_version().decode()
    if built != ffi_api_version():
        raise Refused("triattn_xla: launcher %s was built for XLA-FFI API %s, this jaxlib has %s; fallback: %s" % (path, built, ffi_api_version(), FALLBACK))
    if not hasattr(lib, "TriattnXlaRun"):
        raise Refused("triattn_xla: launcher %s predates the self-describing launch target (rebuild csrc/cubin_launch.cc); fallback: %s" % (path, FALLBACK))
    lib.txla_set_root.argtypes = [ctypes.c_char_p]; lib.txla_root.restype = ctypes.c_char_p
    lib.txla_run_launches.restype = ctypes.c_ulonglong; lib.txla_run_launches.argtypes = [ctypes.c_char_p]
    try:
        ffi.register_ffi_target(TARGET_RUN, ffi.pycapsule(lib.TriattnXlaRun), platform="CUDA", api_version=1)
        ffi.register_ffi_target(TARGET_CUDA, ffi.pycapsule(lib.TriattnXlaCudaFwd), platform="CUDA", api_version=1)
        if hasattr(lib, "TriattnXlaM1Fwd"):
            ffi.register_ffi_target("triattn_xla_m1_fwd", ffi.pycapsule(lib.TriattnXlaM1Fwd), platform="CUDA", api_version=1)
        if hasattr(lib, "TriattnXlaSm80Fwd"):
            ffi.register_ffi_target("triattn_xla_sm80_fwd", ffi.pycapsule(lib.TriattnXlaSm80Fwd), platform="CUDA", api_version=1)
            _STATE["sm80_target"] = True
        ffi.register_ffi_target(TARGET_LAUNCH, ffi.pycapsule(lib.TriattnXlaLaunch), platform="CUDA", api_version=1)
        if hasattr(lib, "XlaCubinCall"):
            ffi.register_ffi_target(TARGET_GENERIC, ffi.pycapsule(lib.XlaCubinCall), platform="CUDA", api_version=1)
            lib.txla_add_root.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            _STATE["generic"] = True
        if hasattr(lib, "XlaCublasBgemm"):
            ffi.register_ffi_target(TARGET_BGEMM, ffi.pycapsule(lib.XlaCublasBgemm), platform="CUDA", api_version=1)
            lib.txla_set_cublas.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_char_p]; lib.txla_cublas_state.restype = ctypes.c_char_p
            _STATE["bgemm"] = True
            _STATE["cublas_install"] = install_cublas(lib)
    except (TypeError, ValueError, RuntimeError, AttributeError) as e:
        raise Refused("triattn_xla: jax.ffi.register_ffi_target failed (%s: %s); fallback: %s" % (type(e).__name__, e, FALLBACK))
    lib.txla_set_root(PKG_DIR.encode())          # cubins are read from the package on first use in any process (executables restored from a compilation cache included)
    _STATE["lib"] = lib; _STATE["path"] = path
    # The CUDA libraries' entries are installed here, at registration, not at trace time: a process that only LOADS a cached executable never traces.
    try:
        from . import _cuda
        _cuda.load()
    except Exception as e:            # noqa: BLE001  (Refused or an image the library cannot load in: the row is then refused by name at trace time, as before)
        _STATE["cuda_eager"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    try:
        from . import _cuda
        _cuda.load_lse()
    except Exception as e:            # noqa: BLE001
        _STATE["cuda_lse_eager"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    try:
        from . import _native
        _native.load()
    except Exception as e:            # noqa: BLE001
        _STATE["native_eager"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    try:
        from . import _sm80
        _sm80.load()
    except Exception as e:            # noqa: BLE001  (an image / card the sm_80 library does not serve: the row is refused by name at trace time)
        _STATE["sm80_eager"] = "%s: %s" % (type(e).__name__, str(e)[:200])
    return lib


def register_spec(label: str, entry: dict, kernel_name: str, shared: int, num_warps: int, grid: Sequence[int], kinds: Sequence[int], ivals: Sequence[int],
                  fvals: Sequence[float], n_trailing_null: int) -> dict:
    """The launch description of one static call signature: the attribute set of the `triattn_xla_run` custom call (nothing is registered in the
    launcher -- the returned dict IS the spec).  entry = the cubin's manifest entry (arch, key, sha256)."""
    if label in _SPECS:
        return _SPECS[label]
    load()
    toks = []
    for kd, iv, fv in zip(kinds, ivals, fvals):
        kd = int(kd)
        if kd == 0: toks.append("b%d" % int(iv))
        elif kd == 1: toks.append("o%d" % int(iv))
        elif kd == 2: toks.append("i%d" % int(iv))
        elif kd == 3: toks.append("l%d" % int(iv))
        elif kd == 4: toks.append("f" + float(fv).hex())        # C99 hex float: exact through strtod
        else: toks.append("n")
    g3 = tuple(int(x) for x in grid) + (1,) * (3 - len(grid))
    spec = {"key": "%s/%s" % (entry["arch"], entry["key"]), "sha": str(entry["sha256"])[:16], "kname": kernel_name, "shared": int(shared), "warps": int(num_warps),
            "nnull": int(n_trailing_null), "gx": g3[0], "gy": g3[1], "gz": g3[2], "params": ",".join(toks), "label": label}
    _SPECS[label] = spec
    return spec


def _ffi_call(ffi, target: str, out_types):
    """ffi.ffi_call(target, ...) with vmap_method="sequential" where this jax accepts it: a vmapped trace that reaches the bare call (no
    custom_vmap rule above it) then runs one launch per sample with the registered shapes instead of raising (module _batching)."""
    from . import _batching
    try:
        call = ffi.ffi_call(target, list(out_types), has_side_effect=False, vmap_method="sequential")
        _batching._STATE["sequential_ffi"] = "sequential"
    except TypeError:
        call = ffi.ffi_call(target, list(out_types), has_side_effect=False)
        _batching._STATE["sequential_ffi"] = "unsupported by this jax"
    return call


def launch_call(spec: dict, out_types):
    """f(*inputs) -> tuple(outputs): jax.ffi.ffi_call on the self-describing launch target; the spec travels as attributes of the custom call."""
    import numpy as np
    ffi = jax_ffi_module()
    call = _ffi_call(ffi, TARGET_RUN, out_types)
    attrs = dict(key=spec["key"], sha=spec["sha"], kname=spec["kname"], shared=np.int64(spec["shared"]), warps=np.int64(spec["warps"]), nnull=np.int64(spec["nnull"]),
                 gx=np.int64(spec["gx"]), gy=np.int64(spec["gy"]), gz=np.int64(spec["gz"]), params=spec["params"], label=spec["label"])

    def f(*inputs):
        return call(*inputs, **attrs)
    return f


def launch_count(spec) -> int:
    """Launches of a spec's cubin key in this process (every static shape that uses the same cubin counts)."""
    lib = _STATE.get("lib")
    if lib is None:
        return 0
    key = spec["key"] if isinstance(spec, dict) else str(spec)
    return int(lib.txla_run_launches(key.encode()))


def cuda_call(out_types, scale: float, flags: int):
    import numpy as np
    ffi = jax_ffi_module()
    call = _ffi_call(ffi, TARGET_CUDA, out_types)
    sc, fl = np.float32(scale), np.int64(flags)

    def f(*inputs):
        return call(*inputs, scale=sc, flags=fl)
    return f


def status() -> dict:
    out = {"loaded": "lib" in _STATE, "path": _STATE.get("path"), "target": TARGET_RUN, "specs": len(_SPECS), "cuda_eager": _STATE.get("cuda_eager"), "cuda_lse_eager": _STATE.get("cuda_lse_eager"), "native_eager": _STATE.get("native_eager"), "sm80_eager": _STATE.get("sm80_eager"), "generic_target": bool(_STATE.get("generic")), "cublas_target": bool(_STATE.get("bgemm")), "cublas": _STATE.get("cublas_install"), "roots": dict(_STATE.get("roots", {})),
           "builds": [{"ffi_api_version": b.get("ffi_api_version"), "jaxlib_lines": b.get("jaxlib_lines"), "file": b.get("file")}
                                                                             for b in launcher_builds()], "specs": len(_SPECS)}
    try:
        out["ffi_api_version"] = ffi_api_version()
    except Refused as e:
        out["ffi_api_version"] = None; out["refused"] = str(e)
    return out


# ---------------------------------------------------------------------------------------------------------------- generic cubin call (launcher 1.5)
def add_root(name: str, directory: str) -> None:
    """Register a named package root for the generic target: cubins named by `path` attributes are read from `directory` on first use in any process."""
    lib = load()
    if not _STATE.get("generic"):
        raise Refused("triattn_xla: launcher %s predates the generic cubin target xla_cubin_call (launcher 1.5; rebuild csrc/cubin_launch.cc); fallback: %s" % (_STATE.get("path"), FALLBACK))
    lib.txla_add_root(name.encode(), directory.encode())
    _STATE.setdefault("roots", {})[name] = directory


def has_generic_target() -> bool:
    load()
    return bool(_STATE.get("generic"))


def f32_token(x: float) -> str:
    """`f<hexfloat>`: an fp32 scalar parameter / struct field exactly as the tracer meant it."""
    return "f" + float(x).hex()


def struct_token(size: int, fields: Sequence[str]) -> str:
    """`S<size>:<f1;f2;...>` -- one by-value struct parameter.  Field tokens: b<i> o<i> n i<v> q<v> f<hex> z<k> t<b|o><i>:<map>; the caller inserts the padding
    (z<k>) so the fields add up to exactly `size` bytes (the launcher checks it)."""
    for f in fields:
        assert "," not in f and ";" not in f and "|" not in f, f
    return "S%d:%s" % (int(size), ";".join(fields))


def tmap_token(dt: str, dims: Sequence[int], strides_bytes: Sequence[int], box: Sequence[int], swizzle: int = 128, l2: int = 128, oob: int = 0) -> str:
    """One tensor-map spec for the `tmaps` attribute (innermost dimension first, as cuTensorMapEncodeTiled takes them)."""
    assert len(strides_bytes) == len(dims) - 1 and len(box) == len(dims)
    return "dt=%s,dims=%s,str=%s,box=%s,swz=%d,l2=%d,oob=%d" % (dt, "/".join(str(int(d)) for d in dims), "/".join(str(int(x)) for x in strides_bytes), "/".join(str(int(b)) for b in box), swizzle, l2, oob)


def cubin_call(*, root: str, path: str, sha: str, kname: str, shared: int, block: int, grid: Sequence[int], grid_rule: int, params: Sequence[str], tmaps: Sequence[str],
               out_types, label: str):
    """f(*inputs) -> tuple(outputs): jax.ffi.ffi_call on the generic target `xla_cubin_call`; everything the launch needs travels as attributes (self-describing;
    executables restored from a persistent compilation cache run in a fresh process once the owning package registered its root)."""
    import numpy as np
    ffi = jax_ffi_module()
    if not has_generic_target():
        raise Refused("triattn_xla: launcher %s predates the generic cubin target (launcher 1.5); fallback: %s" % (_STATE.get("path"), FALLBACK))
    call = _ffi_call(ffi, TARGET_GENERIC, out_types)
    g = list(grid) + [1] * (3 - len(grid))
    attrs = dict(root=root, path=path, sha=sha, kname=kname, shared=np.int64(shared), bx=np.int64(block), gx=np.int64(g[0]), gy=np.int64(g[1]), gz=np.int64(g[2]),
                 grule=np.int64(grid_rule), params=",".join(params), tmaps="|".join(tmaps), label=label)

    def f(*inputs):
        return call(*inputs, **attrs)
    return f


def cublas_candidates() -> List[str]:
    """Paths of the cuBLAS library the running jaxlib ships (pip wheels: nvidia/cublas/lib, nvidia/cu13/lib); the launcher also tries the sonames already in the process."""
    import glob as _glob, sys as _sys
    out = []
    for base in list(_sys.path):
        if not base or not os.path.isdir(base):
            continue
        for pat in ("nvidia/cublas/lib/libcublas.so.*", "nvidia/cu1*/lib/libcublas.so.*", "nvidia/*/lib/libcublas.so.*"):
            for f in sorted(_glob.glob(os.path.join(base, pat))):
                if "Lt" not in os.path.basename(f) and f not in out:
                    out.append(f)
    return out


def _mapped_cublas() -> List[str]:
    """libcublas.so.* files already mapped into this process (Linux), e.g. the one the running XLA client loaded."""
    out = []
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if parts and "/libcublas.so" in parts[-1] and "Lt" not in os.path.basename(parts[-1]) and parts[-1] not in out:
                    out.append(parts[-1])
    except OSError:
        pass
    return out


def _mapped_cublaslt_dirs() -> List[str]:
    out = []
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if parts and "/libcublasLt.so" in parts[-1] and os.path.dirname(parts[-1]) not in out:
                    out.append(os.path.dirname(parts[-1]))
    except OSError:
        pass
    return out


def install_cublas(lib) -> str:
    """Resolve cublasCreate_v2 / cublasSetStream_v2 / cublasSetWorkspace_v2 / cublasGemmStridedBatchedEx and hand them to the launcher.  The library must be
    the SAME cuBLAS the XLA client uses (a second libcublas next to an already-loaded libcublasLt of another release fails its calls): order = a libcublas
    already mapped in this process > the libcublas beside an already-mapped libcublasLt > the pip wheel next to jaxlib (nvidia/cublas/lib) > bare sonames.
    Returns a status string; never raises."""
    import glob as _glob
    cands = _mapped_cublas()
    for d in _mapped_cublaslt_dirs():
        cands += [f for f in sorted(_glob.glob(os.path.join(d, "libcublas.so.*"))) if "Lt" not in os.path.basename(f)]
    cands += cublas_candidates() + ["libcublas.so.13", "libcublas.so.12"]
    seen = set(); ordered = [c for c in cands if not (c in seen or seen.add(c))]
    tried = []
    for c in ordered:
        try:
            cl = ctypes.CDLL(c, mode=getattr(ctypes, "RTLD_LOCAL", 0))
            fns = [ctypes.cast(getattr(cl, n), ctypes.c_void_p).value for n in ("cublasCreate_v2", "cublasSetStream_v2", "cublasSetWorkspace_v2", "cublasGemmStridedBatchedEx")]
            if not all(fns[:2]) or not fns[3]:
                tried.append("%s (symbols missing)" % c); continue
            lib.txla_set_cublas(fns[0], fns[1], fns[2] or None, fns[3], c.encode())
            _STATE["cublas_lib"] = cl                      # keep the library mapped
            _STATE["cublas_path"] = c
            return "installed: %s" % c
        except (OSError, AttributeError) as e:
            tried.append("%s (%s)" % (c, str(e)[:60]))
    return "unavailable: " + "; ".join(tried)[:600]


def cublas_state() -> str:
    lib = _STATE.get("lib")
    if lib is None or not _STATE.get("bgemm"):
        return "unavailable (launcher without the cuBLAS target)"
    return lib.txla_cublas_state().decode()


def cublas_bgemm_nt(out_types, *, a_buf: int, a_off: int, b_buf: int, b_off: int, batch: int, m: int, n: int, k: int, ws: int, label: str):
    """f(*inputs) -> (X [batch, m, n] bf16, workspace): X[c] = A[c] . B[c]^T with A / B = bf16 [batch, m|n, k] planes at element offsets a_off / b_off of inputs a_buf / b_buf."""
    import numpy as np
    ffi = jax_ffi_module()
    load()
    if not _STATE.get("bgemm"):
        raise Refused("triattn_xla: launcher %s has no cuBLAS GEMM target (launcher 1.5); fallback: %s" % (_STATE.get("path"), FALLBACK))
    call = _ffi_call(ffi, TARGET_BGEMM, out_types)
    attrs = dict(a_buf=np.int64(a_buf), a_off=np.int64(a_off), b_buf=np.int64(b_buf), b_off=np.int64(b_off), batch=np.int64(batch), m=np.int64(m), n=np.int64(n), k=np.int64(k), ws=np.int64(ws), label=label)

    def f(*inputs):
        return call(*inputs, **attrs)
    return f
