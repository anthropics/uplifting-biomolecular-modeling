"""trimul_native.launch -- load this package's arch-keyed cubins into the framework's primary context and launch their kernels on the
framework's current stream, with kernel arguments packed from a declared signature.

    from trimul_native import launch as L
    unit = L.load_unit("probe")                        # build/<arch of the current device>/probe.cubin, manifest-verified, cached per device
    k = unit.kernel("probe_cp_async")                  # a Kernel: .regs .local_bytes .static_smem .max_threads; .set_max_dynamic_smem(nbytes)
    k.launch(grid=(gx, gy, 1), block=(128, 1, 1), args=[src, dst, L.i32(rows), L.i32(cols), L.i32(ld), L.f32(scale)])
    tm = L.tensor_map(t2d, box=(64, 64))               # a TensorMap (128 descriptor bytes + the tensor kept alive) for a TMA kernel parameter
    k2.launch(grid, block, args=[L.Struct([tm, dst, L.i32(rows), ...])])      # one by-value __grid_constant__ parameter struct

Conventions (the contract the kernels in csrc/ are written against):

* arguments: a ``torch.Tensor`` packs as its data pointer (u64); ``int`` as i32, ``float`` as f32 unless wrapped (``i32 u32 i64 u64 f32 f64``);
  ``TensorMap`` as the 128-byte CUtensorMap (64-byte aligned inside a struct); ``Struct([...])`` as ONE by-value aggregate laid out with C
  rules (natural alignment per field, tensor maps at 64, size padded to the largest alignment) -- the ``__grid_constant__`` parameter-struct
  form; ``bytes`` as raw bytes.  One ``kernelParams`` entry per top-level argument; the driver copies each from host memory at launch, so
  nothing here allocates device memory and a launch under stream capture records exactly what an eager launch would run.
* stream: ``torch.cuda.current_stream(device)`` unless a raw handle is passed.  No host synchronisation anywhere in a launch.
* module load happens ONCE per (device, cubin digest) at ``load_unit`` -- call it (or the face's ``check``) before capturing a CUDA graph:
  a module load inside a global-mode capture is refused by the driver, a launch is not.
* dynamic shared memory above 48 KiB needs ``set_max_dynamic_smem`` once per function (done at load from the manifest's record when the
  unit declares it, else by the caller).
* host-lean serving: ``Kernel.launch(args)`` packs every argument per call (tens of microseconds of Python for a long list).  A serve that
  repeats a shape builds an ``ArgPack`` ONCE (``k.argpack(args)``: one contiguous host buffer + the void** array), keeps it in the caller's
  cache, patches only the slots that change (``pack.set_ptr(i, tensor)`` / ``set_i32`` / ``set_f32`` / ``set_tensor_map``) and calls
  ``k.launch_packed(grid, block, pack)`` -- one driver call, no Python allocation.  ``tensor_map_cached(cache, tensor, box, ...)`` reuses an
  encoded descriptor while (address, sizes, strides, box, swizzle) repeat.
"""
import ctypes
import os
import struct as _struct
import threading

from . import _driver
from . import manifest as _manifest

__all__ = ["load_unit", "Unit", "Kernel", "ArgPack", "TensorMap", "Struct", "tensor_map", "tensor_map_cached", "i32", "u32", "i64", "u64", "f32",
           "f64", "arch_of_device", "current_stream_handle", "pack_args", "LoadError", "BlockDriver", "block_driver", "unit_path"]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # <tree>/python/trimul_native/ -> <tree>/ (source tree and sealed package alike)
DEFAULT_BUILD_DIR = os.environ.get("TRIMUL_NATIVE_BUILD_DIR") or os.path.join(ROOT, "build")     # env override: load cubins built elsewhere (same manifest layout)
ARCH_OF_CC = {(9, 0): "sm_90a", (8, 0): "sm_80", (8, 6): "sm_80", (8, 7): "sm_80", (8, 9): "sm_80"}   # sm_80 SASS is binary-compatible with every 8.x device


class LoadError(RuntimeError):
    """A unit cannot be loaded; ``.kind`` is the refusal word (no_cubin:<arch> | manifest:<what> | driver_unavailable:<why> | cc_unsupported:<cc>
    | load:<driver error name> | no_kernel:<name>)."""

    def __init__(self, kind, detail=""):
        RuntimeError.__init__(self, kind + ((" (" + detail + ")") if detail else ""))
        self.kind = kind


# ---------------------------------------------------------------------------------------------------------------- typed scalars
class _Scalar(object):
    __slots__ = ("fmt", "value", "align")

    def __init__(self, fmt, value):
        self.fmt = fmt
        self.value = value
        self.align = _struct.calcsize(fmt)

    def pack(self):
        return _struct.pack("<" + self.fmt, self.value)


def i32(v): return _Scalar("i", int(v))
def u32(v): return _Scalar("I", int(v))
def i64(v): return _Scalar("q", int(v))
def u64(v): return _Scalar("Q", int(v))
def f32(v): return _Scalar("f", float(v))
def f64(v): return _Scalar("d", float(v))


class TensorMap(object):
    """The 128 descriptor bytes of a CUtensorMap plus the tensor it addresses (kept alive as long as this object is)."""
    __slots__ = ("raw", "keep", "spec")
    align = _driver.TENSOR_MAP_ALIGN

    def __init__(self, raw, keep=None, spec=None):
        assert len(raw) == _driver.TENSOR_MAP_BYTES, len(raw)
        self.raw, self.keep, self.spec = raw, keep, spec

    def pack(self):
        return self.raw


class Struct(object):
    """A by-value aggregate kernel parameter (C layout).  Fields: tensors / ints / floats / typed scalars / TensorMap / bytes / nested Struct."""
    __slots__ = ("fields",)

    def __init__(self, fields):
        self.fields = list(fields)

    def layout(self):
        blob, align = bytearray(), 1
        for f in self.fields:
            b, a = _pack_one(f)
            pad = (-len(blob)) % a
            blob += b"\x00" * pad
            blob += b
            align = max(align, a)
        blob += b"\x00" * ((-len(blob)) % align)
        return bytes(blob), align

    def pack(self):
        return self.layout()[0]

    @property
    def align(self):
        return self.layout()[1]


def _pack_one(a):
    """-> (bytes, alignment) of one argument."""
    try:
        import torch
        is_tensor = isinstance(a, torch.Tensor)
    except ImportError:
        is_tensor = False
    if is_tensor:
        return _struct.pack("<Q", a.data_ptr()), 8
    if isinstance(a, (_Scalar,)):
        return a.pack(), a.align
    if isinstance(a, TensorMap):
        return a.raw, TensorMap.align
    if isinstance(a, Struct):
        return a.layout()
    if isinstance(a, bool):
        return _struct.pack("<i", int(a)), 4
    if isinstance(a, int):
        if not (-2 ** 31 <= a < 2 ** 31):
            raise TypeError("int argument %d outside i32; wrap it (i64/u64)" % a)
        return _struct.pack("<i", a), 4
    if isinstance(a, float):
        return _struct.pack("<f", a), 4
    if isinstance(a, (bytes, bytearray)):
        return bytes(a), 1
    if a is None:
        return _struct.pack("<Q", 0), 8                                  # a null pointer
    raise TypeError("unsupported kernel argument type %s" % type(a).__name__)


class _Packed(object):
    """Host buffers of the packed arguments + the void** array cuLaunchKernel takes; keep it alive until the launch call returns."""
    __slots__ = ("bufs", "array")

    def __init__(self, args):
        self.bufs = []
        ptrs = []
        for a in args:
            b, al = _pack_one(a)
            al = max(al, 8)
            raw = (ctypes.c_uint8 * (len(b) + al))()
            addr = (ctypes.addressof(raw) + al - 1) & ~(al - 1)
            ctypes.memmove(addr, b, len(b))
            self.bufs.append(raw)
            ptrs.append(addr)
        self.array = (ctypes.c_void_p * max(len(ptrs), 1))(*ptrs) if ptrs else None


def pack_args(args):
    return _Packed(args)


class ArgPack(object):
    """A reusable packed argument list: ONE host buffer holding every top-level argument at a 64-byte-aligned slot plus the void** array,
    built once from a template argument list; slots are patched in place between launches.  Slot indices are the positions in the
    template list; a ``Struct`` argument is one slot whose fields are patched by byte offset (``struct_offsets(i)`` lists them)."""
    __slots__ = ("raw", "base", "offsets", "sizes", "aligns", "array", "field_offsets", "keep")

    def __init__(self, args):
        blobs = [_pack_one(a) for a in args]
        self.offsets, self.sizes, self.aligns, self.field_offsets = [], [], [], []
        total = 0
        for (b, al), a in zip(blobs, args):
            total = (total + 63) & ~63
            self.offsets.append(total)
            self.sizes.append(len(b))
            self.aligns.append(al)
            self.field_offsets.append(_struct_field_offsets(a) if isinstance(a, Struct) else None)
            total += len(b)
        self.raw = (ctypes.c_uint8 * (total + 64))()
        self.base = (ctypes.addressof(self.raw) + 63) & ~63
        for (b, _), off in zip(blobs, self.offsets):
            ctypes.memmove(self.base + off, b, len(b))
        self.array = (ctypes.c_void_p * max(len(blobs), 1))(*[self.base + off for off in self.offsets])
        self.keep = [a for a in args if isinstance(a, TensorMap)] + [f for a in args if isinstance(a, Struct) for f in a.fields if isinstance(f, TensorMap)]

    def _addr(self, slot, field=None):
        off = self.offsets[slot]
        if field is not None:
            fo = self.field_offsets[slot]
            if fo is None:
                raise IndexError("slot %d is not a Struct" % slot)
            off += fo[field]
        return self.base + off

    def struct_offsets(self, slot):
        return list(self.field_offsets[slot] or [])

    def set_ptr(self, slot, tensor_or_int, field=None):
        v = tensor_or_int if isinstance(tensor_or_int, int) else tensor_or_int.data_ptr()
        ctypes.c_uint64.from_address(self._addr(slot, field)).value = v

    def set_i32(self, slot, v, field=None):
        ctypes.c_int32.from_address(self._addr(slot, field)).value = int(v)

    def set_u32(self, slot, v, field=None):
        ctypes.c_uint32.from_address(self._addr(slot, field)).value = int(v)

    def set_i64(self, slot, v, field=None):
        ctypes.c_int64.from_address(self._addr(slot, field)).value = int(v)

    def set_f32(self, slot, v, field=None):
        ctypes.c_float.from_address(self._addr(slot, field)).value = float(v)

    def set_bytes(self, slot, b, field=None):
        ctypes.memmove(self._addr(slot, field), bytes(b), len(b))

    def set_tensor_map(self, slot, tm, field=None):
        ctypes.memmove(self._addr(slot, field), tm.raw, _driver.TENSOR_MAP_BYTES)
        self.keep.append(tm)


def _struct_field_offsets(st):
    offs, pos = [], 0
    for f in st.fields:
        b, a = _pack_one(f)
        pos += (-pos) % a
        offs.append(pos)
        pos += len(b)
    return offs


# ---------------------------------------------------------------------------------------------------------------- device / context / stream
def _torch():
    try:
        import torch
    except ImportError as e:
        raise LoadError("driver_unavailable:torch_not_importable", str(e).split("\n")[0][:120])
    if not torch.cuda.is_available():
        raise LoadError("driver_unavailable:no_cuda_device")
    return torch


def _device_index(device=None):
    torch = _torch()
    if device is None:
        return torch.cuda.current_device()
    d = torch.device(device)
    return torch.cuda.current_device() if d.index is None else int(d.index)


def arch_of_device(device=None):
    """(arch word or None, (major, minor)) for a torch device."""
    torch = _torch()
    cc = tuple(torch.cuda.get_device_capability(_device_index(device)))
    return ARCH_OF_CC.get(cc), cc


def current_stream_handle(device=None):
    torch = _torch()
    return int(torch.cuda.current_stream(_device_index(device)).cuda_stream)


def driver():
    try:
        return _driver.get()
    except _driver.Unresolvable as e:
        raise LoadError("driver_unavailable:%s" % str(e).replace(" ", "_")[:120])


def _make_context_current(idx):
    """The framework's primary context of device ``idx`` current on this thread (what the runtime API would do at its next call)."""
    torch = _torch()
    torch.cuda.init()
    with torch.cuda.device(idx):
        torch.empty(1, device="cuda:%d" % idx)                           # forces primary-context creation on that device
        drv = driver()
        dev = drv.device_get(idx)
        cur = drv.ctx_get_current()
        if cur:
            try:
                if drv.ctx_get_device() == dev:
                    return drv
            except _driver.DriverError:
                pass
        ctx = drv.primary_ctx_retain(dev)
        drv.ctx_set_current(ctx)
        return drv


# ---------------------------------------------------------------------------------------------------------------- units and kernels
class Kernel(object):
    """One __global__ function of a loaded unit."""

    def __init__(self, unit, name, handle):
        self.unit, self.name, self.handle = unit, name, handle
        drv = unit.drv
        A = _driver.FUNC_ATTR
        self.regs = drv.func_get_attribute(A["NUM_REGS"], handle)
        self.local_bytes = drv.func_get_attribute(A["LOCAL_SIZE_BYTES"], handle)
        self.static_smem = drv.func_get_attribute(A["SHARED_SIZE_BYTES"], handle)
        self.max_threads = drv.func_get_attribute(A["MAX_THREADS_PER_BLOCK"], handle)
        self.binary_version = drv.func_get_attribute(A["BINARY_VERSION"], handle)
        self.max_dynamic_smem = None

    def attrs(self):
        return {"regs": self.regs, "local_bytes": self.local_bytes, "static_smem": self.static_smem, "max_threads": self.max_threads,
                "binary_version": self.binary_version, "max_dynamic_smem": self.max_dynamic_smem}

    def set_max_dynamic_smem(self, nbytes):
        self.unit.drv.func_set_attribute(self.handle, _driver.FUNC_ATTR["MAX_DYNAMIC_SHARED_SIZE_BYTES"], int(nbytes))
        self.max_dynamic_smem = int(nbytes)

    def launch(self, grid, block, args, smem=0, stream=None):
        """Launch on ``stream`` (raw handle) or torch's current stream of the unit's device.  ``args``: see the module docstring.  Returns
        nothing; no synchronisation.  The packed host buffers live until cuLaunchKernel returns (the driver has copied them by then)."""
        g = tuple(int(x) for x in grid) + (1,) * (3 - len(grid))
        b = tuple(int(x) for x in block) + (1,) * (3 - len(block))
        if self.max_dynamic_smem is not None and smem > self.max_dynamic_smem:
            raise ValueError("%s: dynamic smem %d B > the opted-in maximum %d B" % (self.name, smem, self.max_dynamic_smem))
        packed = _Packed(args)
        st = current_stream_handle(self.unit.device_index) if stream is None else int(stream)
        self.unit.drv.launch_kernel(self.handle, g, b, int(smem), st, packed.array)
        return None

    @staticmethod
    def argpack(args):
        """Build a reusable ``ArgPack`` from a template argument list (see ``launch_packed``)."""
        return ArgPack(args)

    def launch_packed(self, grid, block, pack, smem=0, stream=None):
        """Launch with a prebuilt ``ArgPack`` (slots patched by the caller beforehand): one driver call, no packing, no allocation.
        ``grid`` / ``block``: 3-tuples of ints (no normalisation here -- this is the per-call fast path)."""
        st = current_stream_handle(self.unit.device_index) if stream is None else stream
        self.unit.drv.launch_kernel(self.handle, grid, block, smem, st, pack.array)


class Unit(object):
    """One loaded cubin (a compilation unit of csrc/) on one device."""

    def __init__(self, name, arch, device_index, drv, module, entry, cubin_path):
        self.name, self.arch, self.device_index, self.drv, self.module, self.entry, self.cubin_path = name, arch, device_index, drv, module, entry, cubin_path
        self._kernels = {}
        self._lock = threading.Lock()

    def kernel(self, fname):
        with self._lock:
            k = self._kernels.get(fname)
            if k is None:
                try:
                    h = self.drv.module_get_function(self.module, fname)
                except _driver.DriverError as e:
                    raise LoadError("no_kernel:%s" % fname, "%s in %s" % (e.name, self.cubin_path))
                k = Kernel(self, fname, h)
                want = (self.entry.get("kernels") or {}).get(fname, {})
                if want.get("max_dynamic_smem"):
                    k.set_max_dynamic_smem(int(want["max_dynamic_smem"]))
                self._kernels[fname] = k
            return k

    def kernel_names(self):
        return sorted((self.entry.get("kernels") or {}).keys())

    def loadcheck(self):
        """{kernel: {regs, local_bytes, want_regs, want_local, ok}} -- the ptxas fingerprint recorded at build read back off the loaded functions."""
        out = {}
        for fname, want in sorted((self.entry.get("kernels") or {}).items()):
            k = self.kernel(fname)
            ok = want.get("regs") in (None, k.regs) and (k.local_bytes == 0) == (int(want.get("spill_stores", 0) or 0) == 0 and int(want.get("stack", 0) or 0) == 0)
            out[fname] = {"regs": k.regs, "local_bytes": k.local_bytes, "static_smem": k.static_smem, "want_regs": want.get("regs"),
                          "want_smem": want.get("smem"), "ok": bool(ok)}
        return out


_UNITS = {}
_UNITS_LOCK = threading.Lock()


def load_unit(unit, arch=None, device=None, build_dir=None, verify=True):
    """Load ``<build_dir>/<arch>/<unit>.cubin`` (arch of the device unless given) into the primary context of ``device``, after the manifest
    checks (cubin sha256; source sha256 when the sources are carried), once per (device, cubin digest).  Raises LoadError(kind) by name."""
    build_dir = os.path.abspath(build_dir or DEFAULT_BUILD_DIR)
    idx = _device_index(device)
    dev_arch, cc = arch_of_device(idx)
    arch = arch or dev_arch
    if arch is None:
        raise LoadError("cc_unsupported:%d.%d" % cc)
    cubin_path = os.path.join(build_dir, arch, unit + ".cubin")
    if not os.path.isfile(cubin_path):
        raise LoadError("no_cubin:%s" % arch, "%s absent" % cubin_path)
    try:
        entry = (_manifest.verify_unit(build_dir, unit, arch, check_sources=True, source_root=ROOT) if verify      # sources resolve against THIS tree's csrc/
                 else _manifest.entry(build_dir, unit, arch))
    except _manifest.ManifestError as e:
        raise LoadError("manifest:%s" % e.kind, str(e))
    key = (idx, entry.get("cubin_sha256") or cubin_path)
    with _UNITS_LOCK:
        u = _UNITS.get(key)
        if u is not None:
            return u
        drv = _make_context_current(idx)
        with open(cubin_path, "rb") as f:
            image = f.read()
        try:
            module = drv.module_load_data(image)
        except _driver.DriverError as e:
            raise LoadError("load:%s" % e.name, "%s (%s, driver %d)" % (cubin_path, arch, drv.driver_version()))
        u = Unit(unit, arch, idx, drv, module, entry, cubin_path)
        _UNITS[key] = u
        return u


def loaded_units():
    return dict(_UNITS)


# ---------------------------------------------------------------------------------------------------------------- tensor maps
_TMAP_DTYPE_OF_TORCH = {"torch.bfloat16": "bfloat16", "torch.float16": "float16", "torch.float32": "float32", "torch.float64": "float64",
                        "torch.uint8": "uint8", "torch.int32": "int32", "torch.int64": "int64"}


def tensor_map(tensor, box, dims=None, strides_bytes=None, swizzle="none", interleave="none", l2="128B", oob="none", elem_strides=None, dtype=None):
    """Encode a tiled CUtensorMap over ``tensor`` (host-side cuTensorMapEncodeTiled).  By default the map describes the tensor itself:
    ``dims`` = its sizes innermost-first, ``strides_bytes`` = the byte strides of dimensions 1.. innermost-first (must be multiples of 16; the
    innermost dimension must be contiguous).  ``box``: the tile extent per dimension, innermost-first (innermost box bytes: a multiple of 16,
    <= the swizzle span when swizzled; every box extent <= 256).  Out-of-bounds elements of a box read as zero (``oob='none'``).  Returns a
    ``TensorMap`` usable as a kernel argument (directly or inside a ``Struct``); it keeps ``tensor`` alive."""
    drv = driver()
    tdt = dtype or _TMAP_DTYPE_OF_TORCH.get(str(tensor.dtype))
    if tdt is None:
        raise TypeError("no tensor-map data type for %s" % tensor.dtype)
    esz = tensor.element_size()
    if dims is None:
        if tensor.stride(-1) != 1:
            raise ValueError("tensor_map: the innermost dimension must be contiguous (stride %d)" % tensor.stride(-1))
        dims = [int(s) for s in reversed(tensor.shape)]
        strides_bytes = [int(st) * esz for st in reversed(tensor.stride()[:-1])]
    rank = len(dims)
    if not (1 <= rank <= 5):
        raise ValueError("tensor_map: rank %d outside 1..5" % rank)
    if strides_bytes is None or len(strides_bytes) != rank - 1:
        raise ValueError("tensor_map: %d strides for rank %d (need rank-1, innermost-first, bytes)" % (0 if strides_bytes is None else len(strides_bytes), rank))
    for s in strides_bytes:
        if s % 16:
            raise ValueError("tensor_map: stride %d B is not a multiple of 16" % s)
    if (tensor.data_ptr() % 16) != 0:
        raise ValueError("tensor_map: base address not 16-byte aligned")
    box = [int(b) for b in box]
    if len(box) != rank or any(b < 1 or b > 256 for b in box):
        raise ValueError("tensor_map: box %s must have %d extents in 1..256" % (box, rank))
    if (box[0] * esz) % 16:
        raise ValueError("tensor_map: innermost box extent %d x %d B is not a multiple of 16 bytes" % (box[0], esz))
    es = [int(e) for e in (elem_strides or [1] * rank)]
    raw = drv.tensor_map_encode_tiled(_driver.TMAP_DTYPE[tdt], rank, tensor.data_ptr(), dims, strides_bytes, box, es,
                                      _driver.TMAP_INTERLEAVE[interleave], _driver.TMAP_SWIZZLE[swizzle], _driver.TMAP_L2[l2], _driver.TMAP_OOB[oob])
    return TensorMap(raw, keep=tensor, spec={"dtype": tdt, "dims": dims, "strides_bytes": strides_bytes, "box": box, "swizzle": swizzle})


def tensor_map_cached(cache, tensor, box, **kw):
    """``tensor_map`` with reuse: the descriptor of (address, dtype, sizes, strides, box, options) is encoded once and kept in ``cache`` (a dict
    the caller owns, e.g. the serve cache).  A descriptor embeds the base address, so a tensor at a new address is a new entry; entries are
    small (the 128 bytes + the key) and the caller bounds the dict's lifetime."""
    key = ("trimul_native.tmap", tensor.data_ptr(), str(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride()), tuple(int(b) for b in box),
           tuple(sorted((k, str(v)) for k, v in kw.items())))
    tm = cache.get(key)
    if tm is None:
        tm = tensor_map(tensor, box, **kw)
        tm = TensorMap(tm.raw, keep=None, spec=tm.spec)                    # the cache must not pin the tensor: the key already names its address
        cache[key] = tm
    return tm


# ---------------------------------------------------------------------------------------------------------------- parameter-block driver surface
class BlockDriver(object):
    """The minimal launch surface the op assembly (kernel.py) is written against, over this package's driver binding and context handling:

        load(cubin_bytes) -> module                 function(module, name, smem_bytes) -> fn   (opts in dynamic smem above 48 KiB)
        attrs(fn) -> {"regs", "local_bytes", "const_bytes"}
        launch(fn, grid, block, smem_bytes, stream_handle, param_bytes)     one by-value parameter block (bytes) = one kernelParams entry
        encode_tiled(dtype 'bf16'|'f32', base_ptr, dims, strides_bytes, box, swizzle_bytes=128, l2promo_bytes=128) -> bytes(128)
        load_unit(name) -> module of build/<arch>/<name>.cubin after the manifest checks (the release route; ``load`` takes raw bytes)

    The framework's primary context is made current before a module load; launches never synchronise."""
    _DT = {"bf16": "bfloat16", "f32": "float32", "fp32": "float32", "f16": "float16"}
    _L2 = {0: "none", 64: "64B", 128: "128B", 256: "256B"}

    def __init__(self, binding=None, device=None):
        self.drv = _driver.get(binding)
        self.device_index = _device_index(device)
        self.word = self.drv.word
        self._bufs = []

    def load(self, cubin_bytes):
        _make_context_current(self.device_index)
        try:
            return self.drv.module_load_data(bytes(cubin_bytes))
        except _driver.DriverError as e:
            raise LoadError("load:%s" % e.name, str(e))

    def load_unit(self, name, build_dir=None, verify=True):
        return load_unit(name, device=self.device_index, build_dir=build_dir, verify=verify).module

    def function(self, module, name, smem_bytes=0):
        try:
            fn = self.drv.module_get_function(module, name)
        except _driver.DriverError as e:
            raise LoadError("no_kernel:%s" % name, str(e))
        if int(smem_bytes) > 48 * 1024:
            self.drv.func_set_attribute(fn, _driver.FUNC_ATTR["MAX_DYNAMIC_SHARED_SIZE_BYTES"], int(smem_bytes))
        return fn

    def attrs(self, fn):
        A = _driver.FUNC_ATTR
        return {"regs": self.drv.func_get_attribute(A["NUM_REGS"], fn), "local_bytes": self.drv.func_get_attribute(A["LOCAL_SIZE_BYTES"], fn),
                "const_bytes": self.drv.func_get_attribute(A["CONST_SIZE_BYTES"], fn)}

    def launch(self, fn, grid, block, smem_bytes, stream, param_bytes):
        buf = ctypes.create_string_buffer(bytes(param_bytes), len(param_bytes))
        arr = (ctypes.c_void_p * 1)(ctypes.addressof(buf))
        self.drv.launch_kernel(fn, (int(grid[0]), int(grid[1]), int(grid[2])), (int(block[0]), int(block[1]), int(block[2])), int(smem_bytes),
                               int(stream) if not isinstance(stream, int) else stream, arr)
        return buf                                                          # the driver copied the block at enqueue; returned for symmetry

    def encode_tiled(self, dtype, base_ptr, dims, strides_bytes, box, swizzle_bytes=128, l2promo_bytes=128):
        rank = len(dims)
        return self.drv.tensor_map_encode_tiled(_driver.TMAP_DTYPE[self._DT.get(dtype, dtype)], rank, int(base_ptr), [int(v) for v in dims],
                                               [int(v) for v in strides_bytes], [int(v) for v in box], [1] * rank, _driver.TMAP_INTERLEAVE["none"],
                                               _driver.TMAP_SWIZZLE[int(swizzle_bytes)], _driver.TMAP_L2[self._L2[int(l2promo_bytes)]], _driver.TMAP_OOB["none"])


_BLOCK_DRIVERS = {}


def block_driver(binding=None, device=None):
    """One BlockDriver per (process, device)."""
    idx = _device_index(device)
    with _UNITS_LOCK:
        bd = _BLOCK_DRIVERS.get(idx)
        if bd is None:
            bd = BlockDriver(binding, idx)
            _BLOCK_DRIVERS[idx] = bd
        return bd


def unit_path(unit, arch, build_dir=None):
    """Path of build/<arch>/<unit>.cubin (existence not checked)."""
    return os.path.join(build_dir or DEFAULT_BUILD_DIR, arch, unit + ".cubin")
