"""trimul_native._driver -- the CUDA driver entry points this package calls, bound over ``libcuda.so.1`` with the standard library
(ctypes), or over the ``cuda.bindings`` wheel when that is importable and asked for.  Both forms call the same driver functions and
produce identical results; the ctypes form has no dependency beyond the driver library every CUDA image mounts.

    from trimul_native import _driver as D
    drv = D.get()                       # resolve once per process: env TRIMUL_NATIVE_DRIVER = auto (default) | ctypes | cuda_bindings
    drv.word                            # "ctypes:libcuda.so.1" | "cuda.bindings:<version>"
    drv.driver_version()                # e.g. 13000  (cuDriverGetVersion)
    mod = drv.module_load_data(cubin_bytes); fn = drv.module_get_function(mod, "kernel_name")
    drv.func_get_attribute(D.FUNC_ATTR["NUM_REGS"], fn)
    drv.func_set_attribute(fn, D.FUNC_ATTR["MAX_DYNAMIC_SHARED_SIZE_BYTES"], nbytes)
    drv.launch_kernel(fn, (gx, gy, gz), (bx, by, bz), smem_bytes, stream_handle, kernel_params_void_pp)
    tm = drv.tensor_map_encode_tiled(dtype, rank, global_address, dims, strides_bytes, box, elem_strides, interleave, swizzle, l2, oob)  # -> 128 bytes

Every driver call is checked; a non-zero CUresult raises ``DriverError`` carrying the decoded error NAME (``cuGetErrorName``), the numeric
code and the entry point.  Nothing here imports torch; the caller (``launch.py``) makes the framework's primary context current before a
module load and passes the framework's stream handle to a launch.  Standard library at import; ``libcuda.so.1`` (or the wheel) is opened at
the first ``get()``.
"""
import ctypes
import os
import threading

__all__ = ["get", "driver", "reset", "DriverError", "Unresolvable", "FUNC_ATTR", "TMAP_DTYPE", "TMAP_SWIZZLE", "TMAP_INTERLEAVE", "TMAP_L2",
           "TMAP_OOB", "ERROR_NAMES"]

# ---------------------------------------------------------------------------------------------------------------- cuda.h enumerators
FUNC_ATTR = {"MAX_THREADS_PER_BLOCK": 0, "SHARED_SIZE_BYTES": 1, "CONST_SIZE_BYTES": 2, "LOCAL_SIZE_BYTES": 3, "NUM_REGS": 4, "PTX_VERSION": 5,
             "BINARY_VERSION": 6, "CACHE_MODE_CA": 7, "MAX_DYNAMIC_SHARED_SIZE_BYTES": 8, "PREFERRED_SHARED_MEMORY_CARVEOUT": 9}
DEVICE_ATTR = {"COMPUTE_CAPABILITY_MAJOR": 75, "COMPUTE_CAPABILITY_MINOR": 76, "MULTIPROCESSOR_COUNT": 16, "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN": 97}
TMAP_DTYPE = {"uint8": 0, "uint16": 1, "uint32": 2, "int32": 3, "uint64": 4, "int64": 5, "float16": 6, "float32": 7, "float64": 8, "bfloat16": 9,
              "float32_ftz": 10, "tfloat32": 11, "tfloat32_ftz": 12}
TMAP_INTERLEAVE = {"none": 0, "16B": 1, "32B": 2}
TMAP_SWIZZLE = {"none": 0, "32B": 1, "64B": 2, "128B": 3, 0: 0, 32: 1, 64: 2, 128: 3}
TMAP_L2 = {"none": 0, "64B": 1, "128B": 2, "256B": 3}
TMAP_OOB = {"none": 0, "nan_request_zero_fma": 1}
STREAM_CAPTURE_STATUS = {0: "none", 1: "active", 2: "invalidated"}
ERROR_NAMES = {0: "CUDA_SUCCESS", 1: "CUDA_ERROR_INVALID_VALUE", 2: "CUDA_ERROR_OUT_OF_MEMORY", 3: "CUDA_ERROR_NOT_INITIALIZED", 4: "CUDA_ERROR_DEINITIALIZED",
               100: "CUDA_ERROR_NO_DEVICE", 101: "CUDA_ERROR_INVALID_DEVICE", 200: "CUDA_ERROR_INVALID_IMAGE", 201: "CUDA_ERROR_INVALID_CONTEXT",
               209: "CUDA_ERROR_NO_BINARY_FOR_GPU", 218: "CUDA_ERROR_INVALID_PTX", 220: "CUDA_ERROR_NVLINK_UNCORRECTABLE", 221: "CUDA_ERROR_JIT_COMPILER_NOT_FOUND",
               222: "CUDA_ERROR_UNSUPPORTED_PTX_VERSION", 225: "CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC", 300: "CUDA_ERROR_INVALID_SOURCE", 301: "CUDA_ERROR_FILE_NOT_FOUND",
               400: "CUDA_ERROR_INVALID_HANDLE", 500: "CUDA_ERROR_NOT_FOUND", 600: "CUDA_ERROR_NOT_READY", 700: "CUDA_ERROR_ILLEGAL_ADDRESS",
               701: "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", 702: "CUDA_ERROR_LAUNCH_TIMEOUT", 719: "CUDA_ERROR_LAUNCH_FAILED", 801: "CUDA_ERROR_NOT_SUPPORTED",
               900: "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED", 901: "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED", 999: "CUDA_ERROR_UNKNOWN"}
TENSOR_MAP_BYTES = 128
TENSOR_MAP_ALIGN = 64


class Unresolvable(ImportError):
    """No driver binding resolves (neither libcuda.so.1 nor the wheel); the message is the named reason."""


class DriverError(RuntimeError):
    """A driver call returned a non-zero CUresult.  ``.code`` the number, ``.name`` the decoded enumerator, ``.call`` the entry point."""

    def __init__(self, call, code, name, detail=""):
        RuntimeError.__init__(self, "%s -> %s (%d)%s" % (call, name, code, (": " + detail) if detail else ""))
        self.call, self.code, self.name = call, int(code), name


# ---------------------------------------------------------------------------------------------------------------- the ctypes form
_SIGS = {   # entry point -> argtypes; restype is CUresult (c_int) everywhere
    "cuInit": [ctypes.c_uint],
    "cuDriverGetVersion": [ctypes.POINTER(ctypes.c_int)],
    "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    "cuGetErrorString": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
    "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "cuDeviceGetName": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
    "cuDeviceGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
    "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
    "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)],
    "cuCtxSetCurrent": [ctypes.c_void_p],
    "cuCtxGetDevice": [ctypes.POINTER(ctypes.c_int)],
    "cuModuleLoadData": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p],
    "cuModuleGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
    "cuModuleUnload": [ctypes.c_void_p],
    "cuFuncGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p],
    "cuFuncSetAttribute": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
    "cuLaunchKernel": [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
    "cuStreamIsCapturing": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)],
}
_OPTIONAL_SIGS = {   # present on drivers with the tensor-map API (CUDA 12.0+); absence is reported by name at the first tensor-map call, not at load
    "cuTensorMapEncodeTiled": [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                               ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
}


class _CtypesDriver(object):
    word = "ctypes:libcuda.so.1"

    def __init__(self):
        try:
            lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise Unresolvable("libcuda.so.1 not loadable (%s)" % str(e).split("\n")[0][:120])
        self._missing = set()
        for name, argtypes in list(_SIGS.items()) + list(_OPTIONAL_SIGS.items()):
            fn = None
            for cand in (name + "_v2", name) if name in ("cuCtxGetCurrent",) else (name,):
                fn = getattr(lib, cand, None)
                if fn is not None:
                    break
            if fn is None:
                if name in _OPTIONAL_SIGS:
                    self._missing.add(name)
                    continue
                raise Unresolvable("libcuda.so.1 lacks %s" % name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
            setattr(self, "_" + name, fn)
        self._lib = lib
        self._check("cuInit", lib.cuInit(0))

    # -- errors
    def error_name(self, code):
        code = int(code)
        s = ctypes.c_char_p()
        try:
            if self._cuGetErrorName(code, ctypes.byref(s)) == 0 and s.value:
                return s.value.decode()
        except Exception:
            pass
        return ERROR_NAMES.get(code, "CUDA_ERROR_%d" % code)

    def _check(self, call, rc, detail=""):
        if rc != 0:
            raise DriverError(call, rc, self.error_name(rc), detail)

    # -- version / devices / contexts
    def driver_version(self):
        v = ctypes.c_int(0)
        self._check("cuDriverGetVersion", self._cuDriverGetVersion(ctypes.byref(v)))
        return int(v.value)

    def device_count(self):
        v = ctypes.c_int(0)
        self._check("cuDeviceGetCount", self._cuDeviceGetCount(ctypes.byref(v)))
        return int(v.value)

    def device_get(self, ordinal):
        d = ctypes.c_int(0)
        self._check("cuDeviceGet", self._cuDeviceGet(ctypes.byref(d), int(ordinal)))
        return int(d.value)

    def device_name(self, dev):
        buf = ctypes.create_string_buffer(256)
        self._check("cuDeviceGetName", self._cuDeviceGetName(buf, 256, int(dev)))
        return buf.value.decode()

    def device_attribute(self, attr, dev):
        v = ctypes.c_int(0)
        self._check("cuDeviceGetAttribute", self._cuDeviceGetAttribute(ctypes.byref(v), int(attr), int(dev)))
        return int(v.value)

    def ctx_get_current(self):
        c = ctypes.c_void_p(0)
        self._check("cuCtxGetCurrent", self._cuCtxGetCurrent(ctypes.byref(c)))
        return int(c.value or 0)

    def ctx_get_device(self):
        d = ctypes.c_int(-1)
        self._check("cuCtxGetDevice", self._cuCtxGetDevice(ctypes.byref(d)))
        return int(d.value)

    def primary_ctx_retain(self, dev):
        c = ctypes.c_void_p(0)
        self._check("cuDevicePrimaryCtxRetain", self._cuDevicePrimaryCtxRetain(ctypes.byref(c), int(dev)))
        return int(c.value or 0)

    def ctx_set_current(self, ctx):
        self._check("cuCtxSetCurrent", self._cuCtxSetCurrent(ctypes.c_void_p(int(ctx))))

    # -- modules / functions
    def module_load_data(self, image):
        if isinstance(image, (bytearray, memoryview)):
            image = bytes(image)
        buf = ctypes.create_string_buffer(image, len(image))          # keeps a NUL-safe copy alive for the call
        m = ctypes.c_void_p(0)
        self._check("cuModuleLoadData", self._cuModuleLoadData(ctypes.byref(m), buf))
        return int(m.value)

    def module_get_function(self, module, name):
        f = ctypes.c_void_p(0)
        nm = name if isinstance(name, bytes) else str(name).encode()
        self._check("cuModuleGetFunction", self._cuModuleGetFunction(ctypes.byref(f), ctypes.c_void_p(int(module)), nm), nm.decode())
        return int(f.value)

    def module_unload(self, module):
        self._check("cuModuleUnload", self._cuModuleUnload(ctypes.c_void_p(int(module))))

    def func_get_attribute(self, attr, func):
        v = ctypes.c_int(0)
        self._check("cuFuncGetAttribute", self._cuFuncGetAttribute(ctypes.byref(v), int(attr), ctypes.c_void_p(int(func))))
        return int(v.value)

    def func_set_attribute(self, func, attr, value):
        self._check("cuFuncSetAttribute", self._cuFuncSetAttribute(ctypes.c_void_p(int(func)), int(attr), int(value)))

    def launch_kernel(self, func, grid, block, smem_bytes, stream, kernel_params):
        """``kernel_params``: a ctypes array of c_void_p (one pointer per kernel argument, each at host memory holding that argument's bytes)."""
        kp = ctypes.cast(kernel_params, ctypes.c_void_p) if kernel_params is not None else ctypes.c_void_p(0)
        rc = self._cuLaunchKernel(ctypes.c_void_p(int(func)), int(grid[0]), int(grid[1]), int(grid[2]), int(block[0]), int(block[1]), int(block[2]),
                                  int(smem_bytes), ctypes.c_void_p(int(stream)), kp, ctypes.c_void_p(0))
        self._check("cuLaunchKernel", rc)

    def stream_is_capturing(self, stream):
        v = ctypes.c_int(0)
        self._check("cuStreamIsCapturing", self._cuStreamIsCapturing(ctypes.c_void_p(int(stream)), ctypes.byref(v)))
        return STREAM_CAPTURE_STATUS.get(int(v.value), str(v.value))

    # -- tensor maps
    def has_tensor_maps(self):
        return "cuTensorMapEncodeTiled" not in self._missing

    def tensor_map_encode_tiled(self, dtype, rank, global_address, global_dim, global_strides, box_dim, element_strides,
                                interleave=0, swizzle=0, l2_promotion=0, oob_fill=0):
        """Returns the 128 descriptor bytes.  ``global_dim``/``box_dim``/``element_strides``: rank entries, innermost dimension first;
        ``global_strides``: rank-1 byte strides of dimensions 1..rank-1 (multiples of 16)."""
        if not self.has_tensor_maps():
            raise DriverError("cuTensorMapEncodeTiled", 500, "CUDA_ERROR_NOT_FOUND", "libcuda.so.1 lacks the tensor-map API (driver older than CUDA 12.0)")
        rank = int(rank)
        gd = (ctypes.c_uint64 * rank)(*[int(v) for v in global_dim])
        gs = (ctypes.c_uint64 * max(rank - 1, 1))(*([int(v) for v in global_strides][:rank - 1] or [0]))
        bd = (ctypes.c_uint32 * rank)(*[int(v) for v in box_dim])
        es = (ctypes.c_uint32 * rank)(*[int(v) for v in element_strides])
        raw = (ctypes.c_uint8 * (TENSOR_MAP_BYTES + TENSOR_MAP_ALIGN))()
        addr = (ctypes.addressof(raw) + TENSOR_MAP_ALIGN - 1) & ~(TENSOR_MAP_ALIGN - 1)
        rc = self._cuTensorMapEncodeTiled(ctypes.c_void_p(addr), int(dtype), rank, ctypes.c_void_p(int(global_address)), gd, gs, bd, es,
                                          int(interleave), int(swizzle), int(l2_promotion), int(oob_fill))
        self._check("cuTensorMapEncodeTiled", rc, "dtype=%d rank=%d dim=%s strides=%s box=%s" % (int(dtype), rank, list(global_dim), list(global_strides), list(box_dim)))
        return ctypes.string_at(addr, TENSOR_MAP_BYTES)


# ---------------------------------------------------------------------------------------------------------------- the cuda.bindings form
class _WheelDriver(object):
    """The same entry points through the ``cuda.bindings`` wheel (12.0+: it must carry the tensor-map API).  Return conventions of the wheel
    (``(CUresult, value...)`` tuples) are unwrapped here so both forms present one interface."""

    def __init__(self):
        try:
            from cuda.bindings import driver as d
        except ImportError as e:
            raise Unresolvable("cuda.bindings not importable (%s)" % str(e).split("\n")[0][:120])
        for need in ("cuTensorMapEncodeTiled", "cuLaunchKernel", "cuModuleLoadData", "cuFuncGetAttribute", "cuStreamIsCapturing"):
            if not hasattr(d, need):
                raise Unresolvable("cuda.bindings lacks %s" % need)
        ver = "?"
        try:
            from cuda.bindings import __version__ as ver          # type: ignore
        except ImportError:
            try:
                import cuda as _c
                ver = getattr(_c, "__version__", "?")
            except ImportError:
                pass
        self.d = d
        self.word = "cuda.bindings:%s" % ver
        self._unwrap("cuInit", d.cuInit(0))

    def _unwrap(self, call, r, detail=""):
        rc = r[0] if isinstance(r, tuple) else r
        if int(rc) != 0:
            raise DriverError(call, int(rc), self.error_name(int(rc)), detail)
        if isinstance(r, tuple):
            return r[1] if len(r) == 2 else r[1:]
        return None

    def error_name(self, code):
        try:
            r = self.d.cuGetErrorName(self.d.CUresult(int(code)))
            if int(r[0]) == 0 and r[1]:
                return r[1].decode() if isinstance(r[1], bytes) else str(r[1])
        except Exception:
            pass
        return ERROR_NAMES.get(int(code), "CUDA_ERROR_%d" % int(code))

    def driver_version(self):
        return int(self._unwrap("cuDriverGetVersion", self.d.cuDriverGetVersion()))

    def device_count(self):
        return int(self._unwrap("cuDeviceGetCount", self.d.cuDeviceGetCount()))

    def device_get(self, ordinal):
        return int(self._unwrap("cuDeviceGet", self.d.cuDeviceGet(int(ordinal))))

    def device_name(self, dev):
        r = self._unwrap("cuDeviceGetName", self.d.cuDeviceGetName(256, self.d.CUdevice(int(dev))))
        return (r.decode() if isinstance(r, bytes) else str(r)).split("\x00")[0]

    def device_attribute(self, attr, dev):
        return int(self._unwrap("cuDeviceGetAttribute", self.d.cuDeviceGetAttribute(self.d.CUdevice_attribute(int(attr)), self.d.CUdevice(int(dev)))))

    def ctx_get_current(self):
        return int(self._unwrap("cuCtxGetCurrent", self.d.cuCtxGetCurrent()) or 0)

    def ctx_get_device(self):
        return int(self._unwrap("cuCtxGetDevice", self.d.cuCtxGetDevice()))

    def primary_ctx_retain(self, dev):
        return int(self._unwrap("cuDevicePrimaryCtxRetain", self.d.cuDevicePrimaryCtxRetain(self.d.CUdevice(int(dev)))) or 0)

    def ctx_set_current(self, ctx):
        self._unwrap("cuCtxSetCurrent", self.d.cuCtxSetCurrent(self.d.CUcontext(int(ctx))))

    def module_load_data(self, image):
        if isinstance(image, (bytearray, memoryview)):
            image = bytes(image)
        return int(self._unwrap("cuModuleLoadData", self.d.cuModuleLoadData(image)))

    def module_get_function(self, module, name):
        nm = name if isinstance(name, bytes) else str(name).encode()
        return int(self._unwrap("cuModuleGetFunction", self.d.cuModuleGetFunction(self.d.CUmodule(int(module)), nm), nm.decode()))

    def module_unload(self, module):
        self._unwrap("cuModuleUnload", self.d.cuModuleUnload(self.d.CUmodule(int(module))))

    def func_get_attribute(self, attr, func):
        return int(self._unwrap("cuFuncGetAttribute", self.d.cuFuncGetAttribute(self.d.CUfunction_attribute(int(attr)), self.d.CUfunction(int(func)))))

    def func_set_attribute(self, func, attr, value):
        self._unwrap("cuFuncSetAttribute", self.d.cuFuncSetAttribute(self.d.CUfunction(int(func)), self.d.CUfunction_attribute(int(attr)), int(value)))

    def launch_kernel(self, func, grid, block, smem_bytes, stream, kernel_params):
        kp = ctypes.addressof(kernel_params) if kernel_params is not None else 0
        self._unwrap("cuLaunchKernel", self.d.cuLaunchKernel(self.d.CUfunction(int(func)), int(grid[0]), int(grid[1]), int(grid[2]), int(block[0]), int(block[1]),
                                                             int(block[2]), int(smem_bytes), self.d.CUstream(int(stream)), kp, 0))

    def stream_is_capturing(self, stream):
        v = self._unwrap("cuStreamIsCapturing", self.d.cuStreamIsCapturing(self.d.CUstream(int(stream))))
        return STREAM_CAPTURE_STATUS.get(int(v), str(v))

    def has_tensor_maps(self):
        return True

    def tensor_map_encode_tiled(self, dtype, rank, global_address, global_dim, global_strides, box_dim, element_strides,
                                interleave=0, swizzle=0, l2_promotion=0, oob_fill=0):
        d = self.d
        rank = int(rank)
        tm = self._unwrap("cuTensorMapEncodeTiled",
                          d.cuTensorMapEncodeTiled(d.CUtensorMapDataType(int(dtype)), rank, int(global_address),
                                                   [d.cuuint64_t(int(v)) for v in global_dim], [d.cuuint64_t(int(v)) for v in list(global_strides)[:rank - 1]],
                                                   [d.cuuint32_t(int(v)) for v in box_dim], [d.cuuint32_t(int(v)) for v in element_strides],
                                                   d.CUtensorMapInterleave(int(interleave)), d.CUtensorMapSwizzle(int(swizzle)),
                                                   d.CUtensorMapL2promotion(int(l2_promotion)), d.CUtensorMapFloatOOBfill(int(oob_fill))),
                          "dtype=%d rank=%d dim=%s strides=%s box=%s" % (int(dtype), rank, list(global_dim), list(global_strides), list(box_dim)))
        words = [int(w) for w in tm.opaque]
        if len(words) != 16:
            raise DriverError("cuTensorMapEncodeTiled", 999, "CUDA_ERROR_UNKNOWN", "unexpected CUtensorMap layout in cuda.bindings (%d words)" % len(words))
        buf = (ctypes.c_uint64 * 16)(*words)
        return ctypes.string_at(ctypes.addressof(buf), TENSOR_MAP_BYTES)


# ---------------------------------------------------------------------------------------------------------------- resolution
_LOCK = threading.Lock()
_STATE = {"drv": None, "error": None}


def get(binding=None):
    """Resolve the binding once per process and return the driver object.  ``binding``: None -> env ``TRIMUL_NATIVE_DRIVER`` (``auto`` |
    ``ctypes`` | ``cuda_bindings``; default ``auto`` = the wheel when importable with the tensor-map API, else ctypes).  A later call naming
    the other form after one is resolved raises ``Unresolvable`` (one binding per process).  Raises ``Unresolvable`` with the reason when
    nothing resolves."""
    want = (binding or os.environ.get("TRIMUL_NATIVE_DRIVER", "auto") or "auto").strip().lower().replace(".", "_").replace("-", "_")
    if want not in ("auto", "ctypes", "cuda_bindings", "wheel"):
        raise ValueError("binding must be auto | ctypes | cuda_bindings (got %r)" % (binding,))
    if want == "wheel":
        want = "cuda_bindings"
    with _LOCK:
        drv = _STATE["drv"]
        if drv is not None:
            if want == "auto" or drv.word.startswith("ctypes" if want == "ctypes" else "cuda.bindings"):
                return drv
            raise Unresolvable("binding %s already resolved in this process (asked for %s)" % (drv.word, want))
        errs = []
        if want in ("auto", "cuda_bindings"):
            try:
                drv = _WheelDriver()
            except Unresolvable as e:
                errs.append(str(e))
                if want == "cuda_bindings":
                    raise
        if drv is None:
            try:
                drv = _CtypesDriver()
            except Unresolvable as e:
                errs.append(str(e))
                raise Unresolvable("; ".join(errs))
        _STATE["drv"] = drv
        return drv


def driver(binding=None, device=None):
    """The parameter-block launch surface the op assembly uses (``launch.BlockDriver``: load / load_unit / function / attrs / launch /
    encode_tiled over this binding, with the framework's primary context made current before loads).  Imports ``launch`` (and the framework)
    lazily; this module itself stays framework-free."""
    from . import launch
    return launch.block_driver(binding, device)


def reset():
    """Forget the resolved binding (tests only; handles obtained from the old binding must not be used afterwards)."""
    with _LOCK:
        _STATE["drv"] = None
