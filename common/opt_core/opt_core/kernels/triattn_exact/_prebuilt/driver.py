"""CUDA driver-API entry points needed to load a cubin and launch its kernels from Python.

Only libcuda.so.1 (installed with the NVIDIA driver) is required: no nvcc, no CUDA toolkit, no torch C++ extension.
Two interchangeable backends issue the same driver calls: a ctypes binding of libcuda (default) and NVIDIA's cuda-python
bindings (TRIATTN_EXACT_DRIVER=cuda-python).  Kernel parameters are marshalled by the caller into one ctypes buffer and
passed as a void** array, so launched bits cannot depend on the backend.

Every failure surfaces as CudaDriverError, a subclass of triattn_exact.Refused (typed, by name).  Launches go to the stream
the caller names and nothing here synchronizes, so callers stay CUDA-graph capturable; function attributes are set once at
module load, never at launch time.
"""
from __future__ import annotations

import ctypes
import os
import threading

from . import Refused

CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
CU_FUNC_ATTRIBUTE_NUM_REGS = 4
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
CUDA_ERROR_NOT_FOUND = 500

# cuTensorMapEncodeTiled enums (CUDA driver API)
CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0
CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7
CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9
CU_TENSOR_MAP_INTERLEAVE_NONE = 0
CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_128B = 0, 1, 2, 3
CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 0, 1, 2, 3
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0
TENSOR_MAP_BYTES = 128          # sizeof(CUtensorMap), 64-byte aligned inside kernel parameter structs


class CudaDriverError(Refused):
    """A CUDA driver call failed (or the driver library is unusable).  Typed refusal: .fn, .code, .reason."""

    def __init__(self, fn, code, msg):
        self.fn, self.code = fn, code
        Refused.__init__(self, f"cuda driver: {fn} failed with error {code} ({msg})", {"driver_call": fn, "code": code})


def _load_libcuda():
    last = None
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            last = e
    raise CudaDriverError("dlopen(libcuda.so.1)", -1, f"NVIDIA driver library not loadable: {last}")


class _CtypesBackend:
    name = "ctypes"

    def __init__(self):
        self.lib = _load_libcuda()
        vp, ci, cu = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
        self._sig("cuInit", [cu]); self._sig("cuDriverGetVersion", [ctypes.POINTER(ci)])
        self._sig("cuDeviceGet", [ctypes.POINTER(ci), ci]); self._sig("cuDeviceGetAttribute", [ctypes.POINTER(ci), ci, ci])
        self._sig("cuDevicePrimaryCtxRetain", [ctypes.POINTER(vp), ci])
        self._sig("cuCtxGetCurrent", [ctypes.POINTER(vp)]); self._sig("cuCtxSetCurrent", [vp])
        self._sig("cuModuleLoadData", [ctypes.POINTER(vp), ctypes.c_char_p]); self._sig("cuModuleUnload", [vp])
        self._sig("cuModuleGetFunction", [ctypes.POINTER(vp), vp, ctypes.c_char_p])
        self._sig("cuFuncGetAttribute", [ctypes.POINTER(ci), ci, vp]); self._sig("cuFuncSetAttribute", [vp, ci, ci])
        self._sig("cuLaunchKernel", [vp, cu, cu, cu, cu, cu, cu, cu, vp, ctypes.POINTER(vp), ctypes.POINTER(vp)])
        self._sig("cuGetErrorString", [ci, ctypes.POINTER(ctypes.c_char_p)])
        self._sig("cuOccupancyMaxActiveBlocksPerMultiprocessor", [ctypes.POINTER(ci), vp, ci, ctypes.c_size_t])
        for opt, args in (("cuModuleGetFunctionCount", [ctypes.POINTER(cu), vp]),          # CUDA >= 12.4
                          ("cuModuleEnumerateFunctions", [ctypes.POINTER(vp), cu, vp]),
                          ("cuFuncGetName", [ctypes.POINTER(ctypes.c_char_p), vp]),
                          ("cuTensorMapEncodeTiled", [vp, ci, cu, vp, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                                                      ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ci, ci, ci, ci])):
            try:
                self._sig(opt, args)
            except AttributeError:
                pass
        self._launch = self.lib.cuLaunchKernel

    def _sig(self, name, argtypes):
        f = getattr(self.lib, name)
        f.argtypes = argtypes; f.restype = ctypes.c_int

    def _chk(self, fn, rc):
        if rc != 0:
            s = ctypes.c_char_p()
            try:
                self.lib.cuGetErrorString(rc, ctypes.byref(s)); msg = (s.value or b"?").decode()
            except Exception:  # hygiene: no-cuda (error-string lookup while raising)
                msg = "?"
            raise CudaDriverError(fn, int(rc), msg)

    def init(self):
        self._chk("cuInit", self.lib.cuInit(0))

    def driver_version(self):
        v = ctypes.c_int(); self._chk("cuDriverGetVersion", self.lib.cuDriverGetVersion(ctypes.byref(v))); return v.value

    def device_get(self, ordinal):
        d = ctypes.c_int(); self._chk("cuDeviceGet", self.lib.cuDeviceGet(ctypes.byref(d), ordinal)); return d.value

    def device_attr(self, attr, dev):
        v = ctypes.c_int(); self._chk("cuDeviceGetAttribute", self.lib.cuDeviceGetAttribute(ctypes.byref(v), attr, dev)); return v.value

    def primary_ctx_retain(self, dev):
        c = ctypes.c_void_p(); self._chk("cuDevicePrimaryCtxRetain", self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(c), dev)); return c.value or 0

    def ctx_get_current(self):
        c = ctypes.c_void_p(); self._chk("cuCtxGetCurrent", self.lib.cuCtxGetCurrent(ctypes.byref(c))); return c.value or 0

    def ctx_set_current(self, ctx):
        self._chk("cuCtxSetCurrent", self.lib.cuCtxSetCurrent(ctypes.c_void_p(ctx)))

    def module_load_data(self, image):
        buf = ctypes.create_string_buffer(image, len(image))
        m = ctypes.c_void_p(); self._chk("cuModuleLoadData", self.lib.cuModuleLoadData(ctypes.byref(m), buf))
        return m.value or 0

    def module_get_function(self, mod, name):
        f = ctypes.c_void_p()
        rc = self.lib.cuModuleGetFunction(ctypes.byref(f), ctypes.c_void_p(mod), name.encode())
        if rc == CUDA_ERROR_NOT_FOUND:
            return 0
        self._chk("cuModuleGetFunction", rc); return f.value or 0

    def module_functions(self, mod):
        """[(name, handle)] via cuModuleEnumerateFunctions, or [] when the driver predates it."""
        if not hasattr(self.lib, "cuModuleGetFunctionCount"):
            return []
        n = ctypes.c_uint(); self._chk("cuModuleGetFunctionCount", self.lib.cuModuleGetFunctionCount(ctypes.byref(n), ctypes.c_void_p(mod)))
        arr = (ctypes.c_void_p * max(1, n.value))()
        self._chk("cuModuleEnumerateFunctions", self.lib.cuModuleEnumerateFunctions(arr, n.value, ctypes.c_void_p(mod)))
        out = []
        for i in range(n.value):
            s = ctypes.c_char_p(); self._chk("cuFuncGetName", self.lib.cuFuncGetName(ctypes.byref(s), ctypes.c_void_p(arr[i])))
            out.append(((s.value or b"").decode(), arr[i] or 0))
        return out

    def func_get_attr(self, attr, f):
        v = ctypes.c_int(); self._chk("cuFuncGetAttribute", self.lib.cuFuncGetAttribute(ctypes.byref(v), attr, ctypes.c_void_p(f))); return v.value

    def func_set_attr(self, f, attr, value):
        self._chk("cuFuncSetAttribute", self.lib.cuFuncSetAttribute(ctypes.c_void_p(f), attr, value))

    def occupancy(self, f, block_threads, dyn_smem):
        n = ctypes.c_int()
        self._chk("cuOccupancyMaxActiveBlocksPerMultiprocessor",
                  self.lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(n), ctypes.c_void_p(f), int(block_threads), int(dyn_smem)))
        return n.value

    def encode_tiled(self, dst_addr, dtype, rank, base_ptr, dims, strides_bytes, box, elem_strides, interleave, swizzle, l2promo, oobfill):
        """cuTensorMapEncodeTiled into the 128-byte, 64-byte-aligned buffer at dst_addr.  dims/box/elem_strides have `rank` entries,
        strides_bytes has rank-1 entries (byte strides of dimensions 1..rank-1)."""
        if not hasattr(self.lib, "cuTensorMapEncodeTiled"):
            raise CudaDriverError("cuTensorMapEncodeTiled", -1, "entry point absent from this driver (needs a CUDA 12 driver)")
        gd = (ctypes.c_uint64 * rank)(*[int(x) for x in dims]); gs = (ctypes.c_uint64 * max(1, rank - 1))(*[int(x) for x in strides_bytes])
        bd = (ctypes.c_uint32 * rank)(*[int(x) for x in box]); es = (ctypes.c_uint32 * rank)(*[int(x) for x in elem_strides])
        rc = self.lib.cuTensorMapEncodeTiled(ctypes.c_void_p(dst_addr), int(dtype), rank, ctypes.c_void_p(base_ptr), gd, gs, bd, es,
                                             int(interleave), int(swizzle), int(l2promo), int(oobfill))
        return int(rc)

    def launch(self, f, grid, block, smem, stream, kernel_params):
        """kernel_params: a ctypes array of c_void_p holding the address of each kernel argument."""
        rc = self._launch(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], smem, stream, kernel_params, None)
        if rc != 0:
            self._chk("cuLaunchKernel", rc)


class _CudaPythonBackend(_CtypesBackend):
    """The same calls through NVIDIA's cuda-python.  Handles are converted to/from ints so module caches and argument marshalling
    are shared with the ctypes backend; tensor-map encoding and occupancy use the ctypes entry points of the same libcuda."""
    name = "cuda-python"

    def __init__(self):
        _CtypesBackend.__init__(self)
        try:
            from cuda.bindings import driver as cbd     # cuda-python >= 12.8 layout
        except ImportError:
            from cuda import cuda as cbd                # older layout
        self.cbd = cbd
        self._CUfunction = cbd.CUfunction; self._CUstream = cbd.CUstream

    def _chk2(self, fn, err):
        if int(err) != 0:
            try:
                msg = self.cbd.cuGetErrorString(err)[1]
                msg = msg.decode() if isinstance(msg, bytes) else str(msg)
            except Exception:  # hygiene: no-cuda (error-string lookup while raising)
                msg = "?"
            raise CudaDriverError(fn, int(err), msg)

    def init(self):
        self._chk2("cuInit", self.cbd.cuInit(0)[0])

    def module_load_data(self, image):
        err, m = self.cbd.cuModuleLoadData(image); self._chk2("cuModuleLoadData", err); return int(m)

    def module_get_function(self, mod, name):
        err, f = self.cbd.cuModuleGetFunction(self.cbd.CUmodule(mod), name.encode())
        if int(err) == CUDA_ERROR_NOT_FOUND:
            return 0
        self._chk2("cuModuleGetFunction", err); return int(f)

    def func_get_attr(self, attr, f):
        err, v = self.cbd.cuFuncGetAttribute(self.cbd.CUfunction_attribute(attr), self.cbd.CUfunction(f)); self._chk2("cuFuncGetAttribute", err); return int(v)

    def func_set_attr(self, f, attr, value):
        self._chk2("cuFuncSetAttribute", self.cbd.cuFuncSetAttribute(self.cbd.CUfunction(f), self.cbd.CUfunction_attribute(attr), int(value))[0])

    def launch(self, f, grid, block, smem, stream, kernel_params):
        err, = self.cbd.cuLaunchKernel(self._CUfunction(f), grid[0], grid[1], grid[2], block[0], block[1], block[2], smem,
                                       self._CUstream(stream), ctypes.addressof(kernel_params), 0)
        if int(err) != 0:
            self._chk2("cuLaunchKernel", err)


_backend = None
_backend_lock = threading.Lock()


def backend():
    """The process-wide driver backend (lazy).  TRIATTN_EXACT_DRIVER=ctypes (default) | cuda-python."""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                want = os.environ.get("TRIATTN_EXACT_DRIVER", "ctypes")
                if want == "cuda-python":
                    try:
                        b = _CudaPythonBackend()
                    except ImportError as e:
                        raise CudaDriverError("import cuda-python", -1, f"TRIATTN_EXACT_DRIVER=cuda-python but the package is not importable: {e}")
                elif want in ("ctypes", "auto", ""):
                    b = _CtypesBackend()
                else:
                    raise CudaDriverError("backend", -1, f"TRIATTN_EXACT_DRIVER={want!r} is not one of ctypes|cuda-python")
                b.init()
                _backend = b
    return _backend


class Device:
    """Per-device state: primary context (the one torch uses), compute capability, opt-in smem limit, SM count, loaded modules."""
    _devices = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, ordinal):
        d = cls._devices.get(ordinal)
        if d is None:
            with cls._lock:
                d = cls._devices.get(ordinal)
                if d is None:
                    d = cls(ordinal); cls._devices[ordinal] = d
        return d

    def __init__(self, ordinal):
        be = backend()
        self.ordinal = ordinal
        self.dev = be.device_get(ordinal)
        self.cc = (be.device_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.dev), be.device_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.dev))
        self.smem_optin = be.device_attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, self.dev)
        self.sm_count = be.device_attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, self.dev)
        self.ctx = be.primary_ctx_retain(self.dev)
        self.modules = {}

    def make_current(self):
        be = backend()
        if be.ctx_get_current() != self.ctx:
            be.ctx_set_current(self.ctx)


class Module:
    """A cubin loaded on one device plus its kernel handles by entry name.  The max-dynamic-shared-memory attribute of every
    kernel is raised once here to (device opt-in limit - static smem), so no attribute call happens at launch time."""

    def __init__(self, device, image, entry_names):
        be = backend()
        device.make_current()
        self.device = device
        self.handle = be.module_load_data(image)
        self.funcs, self.max_dyn_smem, self.static_smem, self.regs = {}, {}, {}, {}
        enumerated = None
        for name in entry_names:
            f = be.module_get_function(self.handle, name)
            if not f:                                            # internal-linkage entry: look it up by enumeration
                if enumerated is None:
                    enumerated = dict(be.module_functions(self.handle))
                f = enumerated.get(name, 0)
            if not f:
                raise CudaDriverError("cuModuleGetFunction", CUDA_ERROR_NOT_FOUND, f"kernel entry {name[:48]}... not in cubin")
            st = be.func_get_attr(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f)
            mx = device.smem_optin - st
            be.func_set_attr(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, mx)
            self.funcs[name] = f; self.max_dyn_smem[name] = mx; self.static_smem[name] = st
            self.regs[name] = be.func_get_attr(CU_FUNC_ATTRIBUTE_NUM_REGS, f)
        self._occ = {}

    def occupancy(self, name, block_threads, dyn_smem):
        """Resident CTAs per SM for kernel `name` at this block size and dynamic smem (cached)."""
        key = (name, block_threads, dyn_smem)
        v = self._occ.get(key)
        if v is None:
            self.device.make_current()
            v = backend().occupancy(self.funcs[name], block_threads, dyn_smem)
            self._occ[key] = v
        return v


def load_module(device_ordinal, cubin_sha256, image_fn, entry_names):
    """Load (once per device and cubin) the image image_fn() returns; entry_names are resolved eagerly."""
    dev = Device.get(device_ordinal)
    m = dev.modules.get(cubin_sha256)
    if m is None:
        with Device._lock:
            m = dev.modules.get(cubin_sha256)
            if m is None:
                m = Module(dev, image_fn(), entry_names)
                dev.modules[cubin_sha256] = m
    return m
