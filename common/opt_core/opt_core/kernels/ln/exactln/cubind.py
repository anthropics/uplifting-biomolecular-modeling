"""cubind — the CUDA driver API and NVRTC entry points exactln calls, bound through ``ctypes`` (``libcuda.so.1`` from the driver;
``libnvrtc.so.*`` from the libraries torch itself ships / loads, or the CUDA toolkit), with the calling and return convention of
``cuda.bindings`` — every call returns ``(err, *values)``, handles are plain integers, ``err`` is an ``int`` whose ``str()`` names the
error — so exactln's plumbing is unchanged whichever binding serves.  ``driver`` and ``nvrtc`` are the two namespaces; nothing is loaded
until the first call (importing this module on a machine without a GPU is fine).

Used by exactln when the ``cuda.bindings`` package is absent from the image (or when ``OPT_CORE_EXACTLN_BINDINGS=ctypes`` asks for it);
the compiled code is the same either way: the same NVRTC library compiles the same source with the same options.
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
import threading
from typing import List, Optional

__all__ = ["driver", "nvrtc", "describe", "available"]

_LOCK = threading.Lock()


class _Err(int):
    """An error code that prints its name (cuda.bindings returns enum members; exactln formats them with f"{err}")."""
    _names = {}

    def __new__(cls, code: int, name: Optional[str] = None):
        obj = int.__new__(cls, int(code))
        obj._name = name
        return obj

    def __str__(self) -> str:
        return "%s(%d)" % (self._name, int(self)) if self._name else "%d" % int(self)

    __repr__ = __str__


_OK = _Err(0, "SUCCESS")


def _load_first(cands: List[str], what: str) -> ctypes.CDLL:
    errs = []
    for c in cands:
        try:
            return ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:  # noqa: PERF203
            errs.append("%s: %s" % (c, str(e)[:80]))
    raise OSError("cubind: could not load %s; tried:\n  %s" % (what, "\n  ".join(errs) or "(no candidates)"))


def _nvrtc_candidates() -> List[str]:
    """libnvrtc sonames / paths in preference order: the CUDA major torch was built with first; sonames the loader (or torch's own preload)
    already resolves; then the pip `nvidia/cuda_nvrtc` wheel beside torch; torch/lib; the toolkit."""
    majors: List[str] = []
    try:
        import torch
        cv = (torch.version.cuda or "").split(".")
        if cv and cv[0].isdigit():
            majors.append(cv[0])
            if int(cv[0]) <= 11 and len(cv) > 1:
                majors.insert(0, "%s.%s" % (cv[0], cv[1]))            # CUDA 11 sonames carry the minor (libnvrtc.so.11.2)
        troot = os.path.dirname(os.path.abspath(torch.__file__))
    except Exception:  # noqa: BLE001
        troot = None
    for m in ("13", "12", "11.2"):
        if m not in majors:
            majors.append(m)
    cands = ["libnvrtc.so.%s" % m for m in majors]
    roots = [p for p in sys.path if isinstance(p, str) and os.path.isdir(p)]
    if troot:
        roots.insert(0, os.path.dirname(troot))
    seen = set()
    for r in roots:
        for pat in (os.path.join(r, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so*"), os.path.join(r, "nvidia", "cu*", "lib", "libnvrtc.so*")):
            for f in sorted(glob.glob(pat)):
                if "builtins" not in os.path.basename(f) and f not in seen:
                    seen.add(f); cands.append(f)
    if troot:
        for f in sorted(glob.glob(os.path.join(troot, "lib", "libnvrtc*.so*"))):
            if "builtins" not in os.path.basename(f) and f not in seen:
                seen.add(f); cands.append(f)
    for pat in ("/usr/local/cuda/lib64/libnvrtc.so*", "/usr/local/cuda-*/lib64/libnvrtc.so*", "/usr/local/cuda/targets/*/lib/libnvrtc.so*", "/usr/lib/x86_64-linux-gnu/libnvrtc.so*"):
        for f in sorted(glob.glob(pat)):
            if "builtins" not in os.path.basename(f) and f not in seen:
                seen.add(f); cands.append(f)
    cands.append("libnvrtc.so")
    return cands


class _Driver:
    """``cuda.bindings.driver`` subset: cuInit, cuCtxGetCurrent, cuCtxSetCurrent, cuDeviceGet, cuDevicePrimaryCtxRetain, cuModuleLoadData,
    cuModuleGetFunction, cuModuleUnload, cuLaunchKernel, cuGetErrorName.  Handles in / out are ints."""

    def __init__(self):
        self._lib = None

    def _l(self) -> ctypes.CDLL:
        if self._lib is None:
            with _LOCK:
                if self._lib is None:
                    lib = _load_first(["libcuda.so.1", "libcuda.so"], "the CUDA driver library")
                    vp, ci, cu = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
                    lib.cuInit.argtypes = [cu]; lib.cuInit.restype = ci
                    lib.cuGetErrorName.argtypes = [ci, ctypes.POINTER(ctypes.c_char_p)]; lib.cuGetErrorName.restype = ci
                    lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(vp)]; lib.cuCtxGetCurrent.restype = ci
                    lib.cuCtxSetCurrent.argtypes = [vp]; lib.cuCtxSetCurrent.restype = ci
                    lib.cuDeviceGet.argtypes = [ctypes.POINTER(ci), ci]; lib.cuDeviceGet.restype = ci
                    lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(vp), ci]; lib.cuDevicePrimaryCtxRetain.restype = ci
                    lib.cuModuleLoadData.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p]; lib.cuModuleLoadData.restype = ci
                    lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]; lib.cuModuleGetFunction.restype = ci
                    lib.cuModuleUnload.argtypes = [vp]; lib.cuModuleUnload.restype = ci
                    lib.cuLaunchKernel.argtypes = [vp, cu, cu, cu, cu, cu, cu, cu, vp, vp, vp]; lib.cuLaunchKernel.restype = ci
                    rc = lib.cuInit(0)                                         # idempotent; torch's runtime has normally done it already
                    if rc != 0:
                        raise OSError("cubind: cuInit failed: %d" % rc)
                    self._lib = lib
        return self._lib

    def _e(self, rc: int) -> _Err:
        if rc == 0:
            return _OK
        name = ctypes.c_char_p()
        try:
            self._l().cuGetErrorName(int(rc), ctypes.byref(name))
            nm = name.value.decode() if name.value else None
        except Exception:  # noqa: BLE001
            nm = None
        return _Err(rc, nm)

    def cuInit(self, flags: int = 0):
        return (self._e(self._l().cuInit(int(flags))),)

    def cuCtxGetCurrent(self):
        ctx = ctypes.c_void_p()
        rc = self._l().cuCtxGetCurrent(ctypes.byref(ctx))
        return (self._e(rc), int(ctx.value or 0))

    def cuCtxSetCurrent(self, ctx):
        return (self._e(self._l().cuCtxSetCurrent(ctypes.c_void_p(int(ctx)))),)

    def cuDeviceGet(self, ordinal: int):
        dev = ctypes.c_int()
        rc = self._l().cuDeviceGet(ctypes.byref(dev), int(ordinal))
        return (self._e(rc), int(dev.value))

    def cuDevicePrimaryCtxRetain(self, dev):
        ctx = ctypes.c_void_p()
        rc = self._l().cuDevicePrimaryCtxRetain(ctypes.byref(ctx), int(dev))
        return (self._e(rc), int(ctx.value or 0))

    def cuModuleLoadData(self, image):
        mod = ctypes.c_void_p()
        buf = bytes(image)                                                     # the driver copies the image during the call
        rc = self._l().cuModuleLoadData(ctypes.byref(mod), buf)
        return (self._e(rc), int(mod.value or 0))

    def cuModuleGetFunction(self, module, name):
        fn = ctypes.c_void_p()
        rc = self._l().cuModuleGetFunction(ctypes.byref(fn), ctypes.c_void_p(int(module)), bytes(name))
        return (self._e(rc), int(fn.value or 0))

    def cuModuleUnload(self, module):
        return (self._e(self._l().cuModuleUnload(ctypes.c_void_p(int(module)))),)

    def cuLaunchKernel(self, f, gx, gy, gz, bx, by, bz, shared_mem, stream, kernel_params, extra=0):
        """``kernel_params``: the integer address of a void*[] block (what exactln's launcher builds), or 0; ``stream``: a CUstream handle
        (``torch.cuda.Stream.cuda_stream``) or an object with ``.cuda_stream`` / ``__int__``."""
        if hasattr(stream, "cuda_stream"):
            stream = stream.cuda_stream
        try:
            sh = int(stream) if stream is not None else 0
        except TypeError:
            sh = int(getattr(stream, "getPtr", lambda: 0)())
        kp = int(kernel_params) if kernel_params else 0
        ex = int(extra) if extra else 0
        rc = self._l().cuLaunchKernel(ctypes.c_void_p(int(f)), int(gx), int(gy), int(gz), int(bx), int(by), int(bz), int(shared_mem),
                                     ctypes.c_void_p(sh), ctypes.c_void_p(kp), ctypes.c_void_p(ex))
        return (self._e(rc),)


class _Nvrtc:
    """``cuda.bindings.nvrtc`` subset: nvrtcVersion, nvrtcCreateProgram, nvrtcDestroyProgram, nvrtcAddNameExpression, nvrtcCompileProgram,
    nvrtcGetProgramLogSize / Log, nvrtcGetLoweredName, nvrtcGetCUBINSize / CUBIN, nvrtcGetPTXSize / PTX, nvrtcGetErrorString.
    Output buffers follow cuda.bindings: the caller passes a ``bytes`` / ``bytearray`` of the queried size and it is filled in place."""

    def __init__(self):
        self._lib = None
        self.path = None

    def _l(self) -> ctypes.CDLL:
        if self._lib is None:
            with _LOCK:
                if self._lib is None:
                    lib = _load_first(_nvrtc_candidates(), "the NVRTC library")
                    vp, ci, cs = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
                    lib.nvrtcVersion.argtypes = [ctypes.POINTER(ci), ctypes.POINTER(ci)]; lib.nvrtcVersion.restype = ci
                    lib.nvrtcGetErrorString.argtypes = [ci]; lib.nvrtcGetErrorString.restype = ctypes.c_char_p
                    lib.nvrtcCreateProgram.argtypes = [ctypes.POINTER(vp), ctypes.c_char_p, ctypes.c_char_p, ci, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
                    lib.nvrtcCreateProgram.restype = ci
                    lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(vp)]; lib.nvrtcDestroyProgram.restype = ci
                    lib.nvrtcAddNameExpression.argtypes = [vp, ctypes.c_char_p]; lib.nvrtcAddNameExpression.restype = ci
                    lib.nvrtcCompileProgram.argtypes = [vp, ci, ctypes.POINTER(ctypes.c_char_p)]; lib.nvrtcCompileProgram.restype = ci
                    lib.nvrtcGetProgramLogSize.argtypes = [vp, ctypes.POINTER(cs)]; lib.nvrtcGetProgramLogSize.restype = ci
                    lib.nvrtcGetProgramLog.argtypes = [vp, ctypes.c_void_p]; lib.nvrtcGetProgramLog.restype = ci
                    lib.nvrtcGetLoweredName.argtypes = [vp, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]; lib.nvrtcGetLoweredName.restype = ci
                    lib.nvrtcGetCUBINSize.argtypes = [vp, ctypes.POINTER(cs)]; lib.nvrtcGetCUBINSize.restype = ci
                    lib.nvrtcGetCUBIN.argtypes = [vp, ctypes.c_void_p]; lib.nvrtcGetCUBIN.restype = ci
                    lib.nvrtcGetPTXSize.argtypes = [vp, ctypes.POINTER(cs)]; lib.nvrtcGetPTXSize.restype = ci
                    lib.nvrtcGetPTX.argtypes = [vp, ctypes.c_void_p]; lib.nvrtcGetPTX.restype = ci
                    self.path = getattr(lib, "_name", None)
                    self._lib = lib
        return self._lib

    def _e(self, rc: int) -> _Err:
        if rc == 0:
            return _OK
        try:
            s = self._l().nvrtcGetErrorString(int(rc))
            nm = s.decode() if s else None
        except Exception:  # noqa: BLE001
            nm = None
        return _Err(rc, nm)

    @staticmethod
    def _fill(dst, src: bytes) -> None:
        """Write ``src`` into the caller's buffer ``dst``: a bytearray / writable buffer, or a FRESH ``bytes`` object of the queried size
        (cuda.bindings fills those in place too; never pass an interned literal)."""
        n = min(len(src), len(dst))
        if n <= 0:
            return
        if isinstance(dst, bytes):
            ctypes.memmove(ctypes.cast(ctypes.c_char_p(dst), ctypes.c_void_p).value, src, n)
        else:
            ctypes.memmove((ctypes.c_char * n).from_buffer(dst), src, n)

    def nvrtcVersion(self):
        a, b = ctypes.c_int(), ctypes.c_int()
        rc = self._l().nvrtcVersion(ctypes.byref(a), ctypes.byref(b))
        return (self._e(rc), int(a.value), int(b.value))

    def nvrtcCreateProgram(self, src, name, num_headers=0, headers=None, include_names=None):
        prog = ctypes.c_void_p()
        hs = list(headers or []); ins = list(include_names or [])
        n = int(num_headers or 0)
        H = (ctypes.c_char_p * max(1, n))(*[bytes(h) for h in hs[:n]]) if n else None
        I = (ctypes.c_char_p * max(1, n))(*[bytes(i) for i in ins[:n]]) if n else None
        self._src = bytes(src)                                                 # keep the source alive for the program's lifetime
        rc = self._l().nvrtcCreateProgram(ctypes.byref(prog), self._src, bytes(name) if name is not None else None, n, H, I)
        return (self._e(rc), int(prog.value or 0))

    def nvrtcDestroyProgram(self, prog):
        p = ctypes.c_void_p(int(prog))
        return (self._e(self._l().nvrtcDestroyProgram(ctypes.byref(p))),)

    def nvrtcAddNameExpression(self, prog, name_expression):
        return (self._e(self._l().nvrtcAddNameExpression(ctypes.c_void_p(int(prog)), bytes(name_expression))),)

    def nvrtcCompileProgram(self, prog, num_options, options):
        opts = [bytes(o) for o in (options or [])][: int(num_options)]
        arr = (ctypes.c_char_p * max(1, len(opts)))(*opts)
        return (self._e(self._l().nvrtcCompileProgram(ctypes.c_void_p(int(prog)), len(opts), arr)),)

    def nvrtcGetProgramLogSize(self, prog):
        n = ctypes.c_size_t()
        rc = self._l().nvrtcGetProgramLogSize(ctypes.c_void_p(int(prog)), ctypes.byref(n))
        return (self._e(rc), int(n.value))

    def nvrtcGetProgramLog(self, prog, buf):
        n = self.nvrtcGetProgramLogSize(prog)[1]
        tmp = ctypes.create_string_buffer(max(1, n))
        rc = self._l().nvrtcGetProgramLog(ctypes.c_void_p(int(prog)), tmp)
        self._fill(buf, tmp.raw[:n])
        return (self._e(rc),)

    def nvrtcGetLoweredName(self, prog, name_expression):
        out = ctypes.c_char_p()
        rc = self._l().nvrtcGetLoweredName(ctypes.c_void_p(int(prog)), bytes(name_expression), ctypes.byref(out))
        return (self._e(rc), bytes(out.value) if out.value is not None else b"")

    def nvrtcGetCUBINSize(self, prog):
        n = ctypes.c_size_t()
        rc = self._l().nvrtcGetCUBINSize(ctypes.c_void_p(int(prog)), ctypes.byref(n))
        return (self._e(rc), int(n.value))

    def nvrtcGetCUBIN(self, prog, buf):
        n = self.nvrtcGetCUBINSize(prog)[1]
        tmp = ctypes.create_string_buffer(max(1, n))
        rc = self._l().nvrtcGetCUBIN(ctypes.c_void_p(int(prog)), tmp)
        self._fill(buf, tmp.raw[:n])
        return (self._e(rc),)

    def nvrtcGetPTXSize(self, prog):
        n = ctypes.c_size_t()
        rc = self._l().nvrtcGetPTXSize(ctypes.c_void_p(int(prog)), ctypes.byref(n))
        return (self._e(rc), int(n.value))

    def nvrtcGetPTX(self, prog, buf):
        n = self.nvrtcGetPTXSize(prog)[1]
        tmp = ctypes.create_string_buffer(max(1, n))
        rc = self._l().nvrtcGetPTX(ctypes.c_void_p(int(prog)), tmp)
        self._fill(buf, tmp.raw[:n])
        return (self._e(rc),)


driver = _Driver()
nvrtc = _Nvrtc()


def available() -> bool:
    """True when both libraries load on this machine (loads them)."""
    try:
        driver._l(); nvrtc._l()
        return True
    except OSError:
        return False


def describe() -> dict:
    """Which libraries serve (after first use): {'driver': soname, 'nvrtc': path, 'nvrtc_version': 'M.m' | None}."""
    out = {"driver": getattr(driver._lib, "_name", None), "nvrtc": nvrtc.path, "nvrtc_version": None}
    if nvrtc._lib is not None:
        r = nvrtc.nvrtcVersion()
        out["nvrtc_version"] = "%d.%d" % (r[1], r[2]) if int(r[0]) == 0 else None
    return out
