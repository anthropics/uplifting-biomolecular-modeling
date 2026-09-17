"""kernels.trimul.esm_v61.cudrv -- the CUDA driver entry points the carried package calls (module load, function attributes, kernel launch,
tensor-map encode), resolved to the ``cuda.bindings`` wheel when the image has it, else to a ctypes binding of the same names over
``libcuda.so.1`` (the driver library every CUDA image mounts).  The carried module reaches the driver through ONE accessor (``_drv() ->
(driver, nvrtc)``); the face points that accessor at ``modules()`` below, so the package's host code runs as written on images with or without
the wheel and nothing in ``sys.modules`` is added or replaced (another user of the wheel in the same process is unaffected).

    from opt_core.kernels.trimul.esm_v61 import cudrv
    word = cudrv.ensure()                 # "cuda.bindings:<version>" | "ctypes:libcuda.so.1"; ensure(binding="ctypes") forces the second form
    driver, nvrtc = cudrv.modules()       # the pair the carried module's accessor returns under the ensured binding
    cudrv.func_attrs(cu_function)         # {"regs": n, "local_bytes": n, "binary_version": n} of a loaded function under either binding (the load check)

Standard library at import; ``libcuda.so.1`` is opened at the first ``ensure`` of the ctypes form.  The ctypes form implements exactly the calls,
enumerators and return conventions (``(CUresult, value...)`` tuples, ``CUtensorMap.opaque``) the carried module uses -- nothing else of the wheel.
Its NVRTC half is a named refusal: a prebuilt cubin never needs the compiler, and a compile attempt through this binding says so.
"""
import ctypes
import sys
import types

__all__ = ["ensure", "binding", "modules", "func_attrs", "driver_version", "Unresolvable"]

_STATE = {"binding": None, "lib": None, "mods": None}


class Unresolvable(ImportError):
    """Neither the wheel nor libcuda.so.1 is loadable (named reason)."""


# ------------------------------------------------------------------------------------------------------------------ enumerators (cuda.h values)
def _enum(name, members):
    cls = type(name, (int,), {"__repr__": lambda self: "%s(%d)" % (name, int(self))})
    for k, v in members.items():
        setattr(cls, k, cls(v))
    cls.members = dict(members)
    return cls


CUresult = _enum("CUresult", {
    "CUDA_SUCCESS": 0, "CUDA_ERROR_INVALID_VALUE": 1, "CUDA_ERROR_OUT_OF_MEMORY": 2, "CUDA_ERROR_NOT_INITIALIZED": 3, "CUDA_ERROR_DEINITIALIZED": 4,
    "CUDA_ERROR_NO_DEVICE": 100, "CUDA_ERROR_INVALID_DEVICE": 101, "CUDA_ERROR_INVALID_IMAGE": 200, "CUDA_ERROR_INVALID_CONTEXT": 201,
    "CUDA_ERROR_NO_BINARY_FOR_GPU": 209, "CUDA_ERROR_UNSUPPORTED_PTX_VERSION": 222, "CUDA_ERROR_INVALID_SOURCE": 300, "CUDA_ERROR_FILE_NOT_FOUND": 301,
    "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED": 303, "CUDA_ERROR_INVALID_HANDLE": 400, "CUDA_ERROR_NOT_FOUND": 500, "CUDA_ERROR_NOT_READY": 600,
    "CUDA_ERROR_ILLEGAL_ADDRESS": 700, "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES": 701, "CUDA_ERROR_LAUNCH_FAILED": 719, "CUDA_ERROR_NOT_SUPPORTED": 801,
    "CUDA_ERROR_UNKNOWN": 999})
CUfunction_attribute = _enum("CUfunction_attribute", {
    "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK": 0, "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES": 1, "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES": 2,
    "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES": 3, "CU_FUNC_ATTRIBUTE_NUM_REGS": 4, "CU_FUNC_ATTRIBUTE_PTX_VERSION": 5, "CU_FUNC_ATTRIBUTE_BINARY_VERSION": 6,
    "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA": 7, "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES": 8, "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT": 9})
CUtensorMapDataType = _enum("CUtensorMapDataType", {
    "CU_TENSOR_MAP_DATA_TYPE_UINT8": 0, "CU_TENSOR_MAP_DATA_TYPE_UINT16": 1, "CU_TENSOR_MAP_DATA_TYPE_UINT32": 2, "CU_TENSOR_MAP_DATA_TYPE_INT32": 3,
    "CU_TENSOR_MAP_DATA_TYPE_UINT64": 4, "CU_TENSOR_MAP_DATA_TYPE_INT64": 5, "CU_TENSOR_MAP_DATA_TYPE_FLOAT16": 6, "CU_TENSOR_MAP_DATA_TYPE_FLOAT32": 7,
    "CU_TENSOR_MAP_DATA_TYPE_FLOAT64": 8, "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16": 9, "CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ": 10, "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32": 11,
    "CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ": 12})
CUtensorMapInterleave = _enum("CUtensorMapInterleave", {"CU_TENSOR_MAP_INTERLEAVE_NONE": 0, "CU_TENSOR_MAP_INTERLEAVE_16B": 1, "CU_TENSOR_MAP_INTERLEAVE_32B": 2})
CUtensorMapSwizzle = _enum("CUtensorMapSwizzle", {"CU_TENSOR_MAP_SWIZZLE_NONE": 0, "CU_TENSOR_MAP_SWIZZLE_32B": 1, "CU_TENSOR_MAP_SWIZZLE_64B": 2, "CU_TENSOR_MAP_SWIZZLE_128B": 3})
CUtensorMapL2promotion = _enum("CUtensorMapL2promotion", {"CU_TENSOR_MAP_L2_PROMOTION_NONE": 0, "CU_TENSOR_MAP_L2_PROMOTION_L2_64B": 1,
                                                          "CU_TENSOR_MAP_L2_PROMOTION_L2_128B": 2, "CU_TENSOR_MAP_L2_PROMOTION_L2_256B": 3})
CUtensorMapFloatOOBfill = _enum("CUtensorMapFloatOOBfill", {"CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE": 0, "CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA": 1})
nvrtcResult = _enum("nvrtcResult", {"NVRTC_SUCCESS": 0, "NVRTC_ERROR_OUT_OF_MEMORY": 1, "NVRTC_ERROR_PROGRAM_CREATION_FAILURE": 2, "NVRTC_ERROR_INVALID_INPUT": 3,
                                    "NVRTC_ERROR_INVALID_PROGRAM": 4, "NVRTC_ERROR_INVALID_OPTION": 5, "NVRTC_ERROR_COMPILATION": 6,
                                    "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE": 7, "NVRTC_ERROR_INTERNAL_ERROR": 11})
ENUMS = ("CUresult", "CUfunction_attribute", "CUtensorMapDataType", "CUtensorMapInterleave", "CUtensorMapSwizzle", "CUtensorMapL2promotion", "CUtensorMapFloatOOBfill")


def cuuint64_t(v):
    return int(v)


def cuuint32_t(v):
    return int(v)


class CUtensorMap(object):
    """The 128-byte opaque descriptor; ``opaque`` = its 16 64-bit words (what the carried module copies into the kernel argument)."""
    __slots__ = ("opaque", "_buf")

    def __init__(self):
        self._buf = (ctypes.c_uint8 * 192)()                              # 128 bytes at a 64-byte boundary inside
        self.opaque = [0] * 16

    def _address(self):
        a = ctypes.addressof(self._buf)
        return (a + 63) & ~63

    def _read(self):
        words = (ctypes.c_uint64 * 16).from_address(self._address())
        self.opaque = [int(w) for w in words]


# ------------------------------------------------------------------------------------------------------------------ libcuda through ctypes
_SIGS = {                                                                   # name -> argtypes (restype is CUresult as c_int everywhere)
    "cuInit": [ctypes.c_uint],
    "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)],
    "cuCtxGetDevice": [ctypes.POINTER(ctypes.c_int)],
    "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
    "cuCtxSetCurrent": [ctypes.c_void_p],
    "cuModuleLoadData": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p],
    "cuModuleGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
    "cuModuleUnload": [ctypes.c_void_p],
    "cuFuncSetAttribute": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
    "cuFuncGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p],
    "cuLaunchKernel": [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
    "cuTensorMapEncodeTiled": [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                               ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    "cuDriverGetVersion": [ctypes.POINTER(ctypes.c_int)],
}


def _lib():
    lib = _STATE["lib"]
    if lib is not None:
        return lib
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        raise Unresolvable("libcuda.so.1 not loadable (%s)" % str(e).split("\n")[0][:80])
    for name, argtypes in _SIGS.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            raise Unresolvable("libcuda.so.1 lacks %s (driver older than the tensor-map API)" % name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
    _STATE["lib"] = lib
    lib.cuInit(0)
    return lib


def _res(rc):
    return CUresult(int(rc))


def _handle(h):
    if h is None:
        return ctypes.c_void_p(0)
    if isinstance(h, ctypes.c_void_p):
        return h
    if isinstance(h, ctypes.Array) or isinstance(h, ctypes._Pointer):
        return ctypes.cast(h, ctypes.c_void_p)
    return ctypes.c_void_p(int(h))


def _current_context(lib):
    """A current context on this thread before a module load: the primary context of the current (else the first) device, as the runtime API
    the framework uses would make current itself at its next call."""
    ctx = ctypes.c_void_p(0)
    lib.cuCtxGetCurrent(ctypes.byref(ctx))
    if ctx.value:
        return
    dev = ctypes.c_int(0)
    pctx = ctypes.c_void_p(0)
    if lib.cuDevicePrimaryCtxRetain(ctypes.byref(pctx), dev) == 0 and pctx.value:
        lib.cuCtxSetCurrent(pctx)


def cuInit(flags=0):
    return (_res(_lib().cuInit(int(flags))),)


def cuDriverGetVersion():
    v = ctypes.c_int(0)
    rc = _lib().cuDriverGetVersion(ctypes.byref(v))
    return (_res(rc), int(v.value))


def cuGetErrorName(err):
    s = ctypes.c_char_p()
    rc = _lib().cuGetErrorName(int(err), ctypes.byref(s))
    return (_res(rc), s.value if s.value is not None else b"CUDA_ERROR_%d" % int(err))


def cuModuleLoadData(image):
    lib = _lib()
    _current_context(lib)
    if isinstance(image, (bytearray, memoryview)):
        image = bytes(image)
    mod = ctypes.c_void_p(0)
    rc = lib.cuModuleLoadData(ctypes.byref(mod), image)
    return (_res(rc), mod)


def cuModuleGetFunction(module, name):
    fn = ctypes.c_void_p(0)
    rc = _lib().cuModuleGetFunction(ctypes.byref(fn), _handle(module), name if isinstance(name, bytes) else str(name).encode())
    return (_res(rc), fn)


def cuModuleUnload(module):
    return (_res(_lib().cuModuleUnload(_handle(module))),)


def cuFuncSetAttribute(func, attrib, value):
    return (_res(_lib().cuFuncSetAttribute(_handle(func), int(attrib), int(value))),)


def cuFuncGetAttribute(attrib, func):
    v = ctypes.c_int(0)
    rc = _lib().cuFuncGetAttribute(ctypes.byref(v), int(attrib), _handle(func))
    return (_res(rc), int(v.value))


def cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared_bytes, stream, kernel_params, extra):
    kp = _handle(kernel_params) if kernel_params not in (0, None) else ctypes.c_void_p(0)
    ex = _handle(extra) if extra not in (0, None) else ctypes.c_void_p(0)
    st = _handle(getattr(stream, "value", stream) if not isinstance(stream, int) else stream)
    rc = _lib().cuLaunchKernel(_handle(f), int(gx), int(gy), int(gz), int(bx), int(by), int(bz), int(shared_bytes), st, kp, ex)
    return (_res(rc),)


def cuTensorMapEncodeTiled(data_type, rank, global_address, global_dim, global_strides, box_dim, element_strides, interleave, swizzle, l2_promotion, oob_fill):
    rank = int(rank)
    gd = (ctypes.c_uint64 * max(rank, 1))(*[int(v) for v in global_dim])
    gs = (ctypes.c_uint64 * max(rank - 1, 1))(*[int(v) for v in global_strides][:max(rank - 1, 0)] or [0])
    bd = (ctypes.c_uint32 * max(rank, 1))(*[int(v) for v in box_dim])
    es = (ctypes.c_uint32 * max(rank, 1))(*[int(v) for v in element_strides])
    tm = CUtensorMap()
    addr = getattr(global_address, "value", global_address)
    rc = _lib().cuTensorMapEncodeTiled(ctypes.c_void_p(tm._address()), int(data_type), rank, ctypes.c_void_p(int(addr)), gd, gs, bd, es,
                                       int(interleave), int(swizzle), int(l2_promotion), int(oob_fill))
    tm._read()
    return (_res(rc), tm)


# ------------------------------------------------------------------------------------------------------------------ NVRTC: named refusal
def nvrtcVersion():
    return (nvrtcResult.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE, 0, 0)          # the compiler is not part of this binding: prebuilt cubins only


def _nvrtc_refused(*_a, **_k):
    return (nvrtcResult.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,)


_DRIVER_NAMES = ("CUresult", "CUfunction_attribute", "CUtensorMapDataType", "CUtensorMapInterleave", "CUtensorMapSwizzle", "CUtensorMapL2promotion",
                 "CUtensorMapFloatOOBfill", "CUtensorMap", "cuuint64_t", "cuuint32_t", "cuInit", "cuDriverGetVersion", "cuGetErrorName", "cuModuleLoadData",
                 "cuModuleGetFunction", "cuModuleUnload", "cuFuncSetAttribute", "cuFuncGetAttribute", "cuLaunchKernel", "cuTensorMapEncodeTiled")
_NVRTC_NAMES = {"nvrtcResult": nvrtcResult, "nvrtcVersion": nvrtcVersion, "nvrtcCreateProgram": _nvrtc_refused, "nvrtcCompileProgram": _nvrtc_refused,
                "nvrtcGetProgramLogSize": _nvrtc_refused, "nvrtcGetProgramLog": _nvrtc_refused, "nvrtcGetCUBINSize": _nvrtc_refused, "nvrtcGetCUBIN": _nvrtc_refused}


def _shim_modules():
    me = sys.modules[__name__]
    drv = types.ModuleType("cuda.bindings.driver", "ctypes binding of libcuda.so.1 (opt_core.kernels.trimul.esm_v61.cudrv)")
    for n in _DRIVER_NAMES:
        setattr(drv, n, getattr(me, n))
    drv.__opt_core_binding__ = "ctypes:libcuda.so.1"
    nv = types.ModuleType("cuda.bindings.nvrtc", "NVRTC is not part of the ctypes binding (prebuilt cubins only)")
    for n, v in _NVRTC_NAMES.items():
        setattr(nv, n, v)
    nv.__opt_core_binding__ = "ctypes:libcuda.so.1"
    return drv, nv


def ensure(binding=None):
    """Resolve the binding once per process: ``binding=None`` -> the wheel when importable with the tensor-map entry points, else the ctypes form;
    ``"ctypes"`` -> the ctypes form even when the wheel exists; ``"wheel"`` -> the wheel or ``Unresolvable``.  A later call asking for the other
    form after one is resolved raises ``Unresolvable`` (handles of the two forms do not mix in one process).  Returns the binding word."""
    if binding not in (None, "wheel", "ctypes"):
        raise ValueError("binding must be None, 'wheel' or 'ctypes' (got %r)" % (binding,))
    have = _STATE["binding"]
    if have is not None:
        if binding is None or have.startswith("cuda.bindings" if binding == "wheel" else "ctypes"):
            return have
        raise Unresolvable("binding %s already resolved in this process (asked for %s)" % (have, binding))
    if binding != "ctypes":
        try:
            from cuda.bindings import driver as d
            from cuda.bindings import nvrtc as n
            if hasattr(d, "cuTensorMapEncodeTiled") and hasattr(d, "cuLaunchKernel") and hasattr(d, "cuModuleLoadData") and hasattr(d, "cuFuncGetAttribute"):
                ver = "?"
                try:
                    from cuda.bindings import __version__ as ver                 # type: ignore
                except ImportError:
                    try:
                        import cuda as _c
                        ver = getattr(_c, "__version__", "?")
                    except ImportError:
                        pass
                _STATE["mods"] = (d, n)
                _STATE["binding"] = "cuda.bindings:%s" % ver
                return _STATE["binding"]
            if binding == "wheel":
                raise Unresolvable("cuda.bindings lacks the tensor-map entry points (wheel older than 12.0)")
        except ImportError as e:
            if binding == "wheel":
                raise Unresolvable("cuda.bindings not importable (%s)" % str(e).split("\n")[0][:80])
    _lib()                                                                     # raises Unresolvable by name when libcuda is absent
    _STATE["mods"] = _shim_modules()
    _STATE["binding"] = "ctypes:libcuda.so.1"
    return _STATE["binding"]


def binding():
    return _STATE["binding"]


def modules():
    """(driver, nvrtc) under the ensured binding: the wheel's two modules, or the ctypes form's two module objects (never registered in sys.modules)."""
    ensure()
    return _STATE["mods"]


def driver_version():
    """The driver's CUDA version as (major, minor) through the ensured binding (None when no binding resolves or the call fails)."""
    try:
        d = modules()[0]
    except Unresolvable:
        return None
    r = d.cuDriverGetVersion()
    rc, v = (r[0], r[1]) if isinstance(r, tuple) and len(r) > 1 else (r, 0)
    v = int(v)
    return (v // 1000, (v % 1000) // 10) if int(rc) == 0 else None


def func_attrs(func):
    """{"regs", "local_bytes", "binary_version"} of a loaded CUfunction under the ensured binding (the load check of a prebuilt cubin)."""
    d = modules()[0]
    A = d.CUfunction_attribute
    out = {}
    for key, att in (("regs", A.CU_FUNC_ATTRIBUTE_NUM_REGS), ("local_bytes", A.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES), ("binary_version", A.CU_FUNC_ATTRIBUTE_BINARY_VERSION)):
        r = d.cuFuncGetAttribute(att, func)
        rc, val = (r[0], r[1]) if isinstance(r, tuple) and len(r) > 1 else (r, -1)
        out[key] = int(val) if int(rc) == 0 else None
    return out
