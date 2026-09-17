"""Record, WITHOUT a GPU, everything fpf_trimul_v4's served path decides before and at launch for one (N, C, D, direction, mask, residual) call:
the table.json row it selects (key + resolve_cfg k1/k3), the plane extent Np, every device allocation (shape, dtype), the Triton launch grids and
constexpr tile arguments of K1/K3, and the cuBLAS bmm operand shapes.  The kernels themselves are replaced by recorders; tensors live on the
'meta' device; the CUDA queries (capability, device name, free memory) are stubbed.  Used by tests/test_trimul_v4_admission.py against the
current tree and by the recorded fixture generator against a reference tree, so "launch arguments are identical" is a JSON equality.

    record_all(pkg, cells_path, sizes, channels) -> {"<N>x<C>x<dir>x<mask>x<res>": {...}}     pkg = an imported fpf_trimul_v4 package (any name)
"""
import contextlib
import json


_MISSING = object()


def _grid_rec(store, name):
    class _Launch:
        def __getitem__(self, grid):
            def call(*args, **kw):
                rec = {"kernel": name, "grid": [int(x) for x in grid],
                       "constexpr": {k: (int(v) if isinstance(v, (bool, int)) else str(v)) for k, v in sorted(kw.items())},
                       "n_args": len(args),
                       "int_args": [int(a) for a in args if isinstance(a, int) and not isinstance(a, bool)],     # N, Np, strides, row extents, batch offsets: the scalar launch geometry
                       "arg_shapes": [list(a.shape) for a in args if hasattr(a, "shape")]}
                descs = [{"view": list(a.shape), "strides": list(a.strides), "box": list(a.block_shape)} for a in args if isinstance(a, _Desc)]
                if descs:
                    rec["descriptors"] = descs                                                              # host TMA descriptors (kdesc cells): the viewed tensor's shape / strides and the box
                store.append(rec)
            return call
    return _Launch()


class _Desc:
    """Stand-in for triton's host TensorDescriptor: what kdesc.launch_k1d / launch_k3d build per operand (the viewed tensor's shape and strides, the box shape)."""
    def __init__(self, t, block_shape):
        self.shape = tuple(int(s) for s in t.shape); self.strides = tuple(int(s) for s in t.stride()); self.block_shape = tuple(int(b) for b in block_shape)

    @classmethod
    def from_tensor(cls, t, block_shape):
        return cls(t, block_shape)


@contextlib.contextmanager
def patched(pkg, cells_path, free_bytes=80 * 1024 ** 3, cc=(9, 0), triton_mm="3.3", reserved_bytes=0, allocated_bytes=0, oom_at=None):
    """Route the package's CUDA queries and kernels to recorders; yields the launch log (a list).  free_bytes = the driver's free memory;
    reserved_bytes / allocated_bytes = the caching allocator's figures; oom_at = 1-based index of the device allocation that raises OutOfMemoryError."""
    import torch
    K, CELLS, G, KD = pkg.kernels, pkg.cells, pkg.generic, pkg.kdesc
    log = []
    saved = {}

    def setp(obj, name, val):
        saved.setdefault((id(obj), name), (obj, name, getattr(obj, name, _MISSING)))
        if val is _MISSING:
            delattr(obj, name)
        else:
            setattr(obj, name, val)

    real_empty, real_empty_like = torch.empty, torch.empty_like

    OOM = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    n_alloc = [0]

    def maybe_oom():
        n_alloc[0] += 1
        if oom_at is not None and n_alloc[0] == int(oom_at):
            log.append({"oom": n_alloc[0]})
            raise OOM("CUDA out of memory (recorder: allocation %d)" % n_alloc[0])

    def empty(*size, **kw):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(int(s) for s in size)
        dt = kw.get("dtype", torch.get_default_dtype())
        maybe_oom()
        log.append({"alloc": "empty", "shape": list(shape), "dtype": str(dt), "device": str(kw.get("device"))})
        return real_empty(shape, dtype=dt, device="meta")

    def empty_like(t, **kw):
        maybe_oom()
        log.append({"alloc": "empty_like", "shape": list(t.shape), "dtype": str(kw.get("dtype", t.dtype))})
        return real_empty(tuple(t.shape), dtype=kw.get("dtype", t.dtype), device="meta")

    def bmm(a, b, out=None):
        log.append({"kernel": "bmm", "a": list(a.shape), "b": list(b.shape), "a_strides": list(a.stride()), "b_strides": list(b.stride()), "out": list(out.shape) if out is not None else None})
        return out

    setp(torch, "empty", empty); setp(torch, "empty_like", empty_like); setp(torch, "bmm", bmm)
    setp(torch.cuda, "mem_get_info", lambda device=None: (int(free_bytes), int(free_bytes) + (8 << 30)))
    setp(torch.cuda, "memory_reserved", lambda device=None: int(reserved_bytes)); setp(torch.cuda, "memory_allocated", lambda device=None: int(allocated_bytes))
    setp(torch.cuda, "get_device_capability", lambda device=None: tuple(cc))
    setp(torch.cuda, "get_device_name", lambda device=None: "recorder (cc %d.%d)" % tuple(cc))
    setp(K, "_k1c", _grid_rec(log, "_k1c")); setp(K, "_k1t", _grid_rec(log, "_k1t")); setp(K, "_k3c", _grid_rec(log, "_k3c"))
    setp(KD, "_k1d", _grid_rec(log, "_k1d")); setp(KD, "_k3d", _grid_rec(log, "_k3d")); setp(KD, "TensorDescriptor", _Desc)     # the host-descriptor cells: their launchers run, kernels + descriptors recorded
    setp(K, "ensure_allocator", lambda: None)
    setp(CELLS, "triton_mm", lambda: triton_mm)
    has_desc = tuple(int(x) for x in str(triton_mm).split(".")[:2]) >= (3, 4)          # the tensor-descriptor (TMA) API arrives with triton 3.4: mock the CAPABILITY per label,
    setp(K, "HAS_DESC", has_desc); setp(KD, "HAS_TD", has_desc)                         # not the box's own triton, so the recorded fixture replays on any stack
    import triton as _triton
    if has_desc and not hasattr(_triton, "set_allocator"):
        setp(_triton, "set_allocator", lambda fn: None)
    elif not has_desc and hasattr(_triton, "set_allocator"):
        setp(_triton, "set_allocator", _MISSING)
    setp(CELLS, "TABLE_PATH", cells_path); setp(CELLS, "KIT_TABLE_PATH", None); setp(CELLS, "_CELL_CACHE", {}); setp(CELLS, "INFO", {})   # hermetic: the process's kit table (FPF_TRIMUL_V4_CELLS) is not part of the record
    setp(CELLS, "_log", lambda msg: None); setp(G, "_log", lambda msg: None)
    setp(G, "probe", lambda *a, **k: None)                                               # the warm numerics probe launches kernels: not under a CPU replay (tests/gpu holds it)
    setp(CELLS, "_OFF", {"why": None})
    try:
        yield log
    finally:
        for obj, name, val in saved.values():
            if val is _MISSING:
                if hasattr(obj, name):
                    delattr(obj, name)
            else:
                setattr(obj, name, val)


class _CudaLike:
    """A [N, N, C] (B None) or [B, N, N, C] tensor stand-in: the attributes the served path reads before launch (no storage of that size is ever allocated)."""

    def __init__(self, N, C, dtype, B=None):
        import torch
        shape = (N, N, C) if B is None else (int(B), N, N, C)
        self.shape = torch.Size(shape); self.dtype = dtype; self.device = torch.device("cuda", 0); self.is_cuda = True
        self._t = torch.empty(shape, dtype=dtype, device="meta")

    def dim(self): return len(self.shape)
    def is_contiguous(self): return True
    def contiguous(self): return self
    def element_size(self): return self._t.element_size()
    def unsqueeze(self, d): return _Batch(self)
    def __getitem__(self, i): return _CudaLike(self.shape[-2], self.shape[-1], self.dtype) if self.dim() == 4 else self._t[i]
    def __getattr__(self, name): return getattr(self._t, name)


class _Batch:
    """z.unsqueeze(0) of a rank-3 stand-in: a [1, N, N, C] view answering the launcher's reads with z's own attributes."""
    def __init__(self, z):
        import torch
        self.z = z; self.shape = torch.Size((1,) + tuple(z.shape)); self.dtype = z.dtype; self.device = z.device; self.is_cuda = True
    def __getitem__(self, i): return self.z
    def dim(self): return 4
    def is_contiguous(self): return True
    def contiguous(self): return self
    def element_size(self): return self.z.element_size()
    def __getattr__(self, name): return getattr(self.z, name)


def _weights(C, D):
    """A packed-weights stand-in with the shapes the launcher reads (meta device)."""
    import torch
    w = {"C": C, "D": D, "has_bias": False}
    shapes = {"ln_in_w": (C,), "ln_in_b": (C,), "wgT_in": (C, 2 * D), "wpT_in": (C, 2 * D), "ln_out_w": (D,), "ln_out_b": (D,), "wz": (C, D), "wg_out": (C, C)}
    for k, shp in shapes.items():
        w[k] = torch.empty(shp, device="meta")
    return w


def record_call(pkg, cells_path, N, C, D, outgoing, with_mask, residual, dtype_name="bfloat16", pad=16, B=None, **patch_kw):
    """B None: the [N, N, C] call; B >= 1: the [B, N, N, C] call (mask, when given, is per-b [B, N, N])."""
    import torch
    dtype = getattr(torch, dtype_name)
    K, G = pkg.kernels, pkg.generic
    w = _weights(C, D)
    z = _CudaLike(N, C, dtype, B)
    mask = (torch.ones((N, N) if B is None else (int(B), N, N), dtype=torch.float32, device="meta")) if with_mask else None
    real_is_tensor = torch.is_tensor
    with patched(pkg, cells_path, **patch_kw) as log:
        torch.is_tensor = lambda t: True if isinstance(t, (_CudaLike, _Batch)) else real_is_tensor(t)
        try:
            try:
                cfg = G._check(z, mask, w, None, None) if G._check.__code__.co_argcount == 5 else G._check(z, mask, w, None, None, pad)
                sel = {"cfg": cfg, "cell_info": {k: v for k, v in (pkg.cells.INFO.get(str(z.device)) or {}).items() if k in ("cc", "triton", "source")}}
                k1, k3 = K.resolve_cfg(cfg, C, D, False)
                sel["resolved"] = {"k1": k1, "k3": k3}
                K.trimul_v4_forward(z, bool(outgoing), mask, w, cfg, residual=bool(residual), stock_round=True, pad=pad)
                sel["Np"] = int(K.ceil_to(N, pad))
                sel["log"] = log[:]
            except G.TrimulUnsupported as e:
                sel = {"refused": e.reason}
        finally:
            torch.is_tensor = real_is_tensor
    return json.loads(json.dumps(sel, sort_keys=True, default=str))


def record_generic_call(pkg, cells_path, N, C, D, B, outgoing=True, with_mask=True, residual=False, dtype_name="bfloat16", pad=16, kernel_entry=False, **patch_kw):
    """The SERVING layer on a [B, N, N, C] stand-in: generic.trimul_packed (kernel_entry False) or kernels.trimul_v4_forward directly (True).  Returns the launch log, the
    generic COUNTS after the call (batch words), and the refusal / limit name when one was raised — nothing of that size is allocated, so the int32 / grid boundaries of
    one launch set are reachable here."""
    import copy, torch
    dtype = getattr(torch, dtype_name)
    K, G = pkg.kernels, pkg.generic
    w = _weights(C, D)
    z = _CudaLike(N, C, dtype, B)
    mask = torch.ones((int(B), N, N), dtype=torch.float32, device="meta") if with_mask else None
    real_is_tensor = torch.is_tensor
    saved = copy.deepcopy(G.COUNTS)
    with patched(pkg, cells_path, **patch_kw) as log:
        torch.is_tensor = lambda t: True if isinstance(t, (_CudaLike, _Batch)) else real_is_tensor(t)
        try:
            rec = {"B": int(B), "N": N, "C": C}
            try:
                if kernel_entry:
                    cfg = G._check(z, mask, w, None, None) if G._check.__code__.co_argcount == 5 else G._check(z, mask, w, None, None, pad)
                    K.trimul_v4_forward(z, bool(outgoing), mask, w, cfg, residual=bool(residual), stock_round=True, pad=pad)
                else:
                    G.trimul_packed(z, mask, outgoing=bool(outgoing), weights=w, residual=bool(residual), pad=pad)
            except K.BatchLimit as e:                   # the launch-set limit word (a TrimulUnsupported subclass: caught first to record which entry refused)
                rec["batch_limit"] = e.reason
            except G.TrimulUnsupported as e:
                rec["refused"] = e.reason
            rec["log"] = log[:]
            rec["counts"] = copy.deepcopy(G.COUNTS)
        finally:
            torch.is_tensor = real_is_tensor
            launches = saved.pop("kernels", None)
            G.COUNTS.clear(); G.COUNTS.update(saved); G.COUNTS["kernels"] = K.LAUNCHES          # the per-kernel tally stays ONE dict (kernels.LAUNCHES), aliased into COUNTS
            if launches is not None:
                K.LAUNCHES.clear(); K.LAUNCHES.update(launches)
    return rec


def record_all(pkg, cells_path, sizes=(101, 256, 400, 705, 1000, 1400, 2048), channels=(128, 256), B=None, **patch_kw):
    out = {}
    for N in sizes:
        for C in channels:
            for outgoing in (True, False):
                for with_mask in (False, True):
                    for residual in (False, True):
                        key = "N%d_C%d_%s_%s_%s" % (N, C, "out" if outgoing else "in", "mask" if with_mask else "nomask", "res" if residual else "nores") + ("" if B is None else "_B%d" % B)
                        out[key] = record_call(pkg, cells_path, N, C, C, outgoing, with_mask, residual, B=B, **patch_kw)
    return out


RECORD_SIZES, RECORD_CHANNELS, RECORD_STACKS, RECORD_ABOVE, RECORD_BATCH = (101, 256, 400, 705, 1000, 1400, 2048), (128, 256), ("3.3", "3.7"), (3072, 4096), {"B": 8, "sizes": (101, 256, 705)}


def write_launch_record(pkg, cells_path, path, reference):
    """Record this tree's launcher into the recorded fixture tests/test_trimul_v4_admission.py holds it to: per triton stack the [N,N,C] calls over RECORD_SIZES x
    RECORD_CHANNELS ('records'), the sizes above the former ceiling ('above_2048'), and the [B,N,N,C] calls of RECORD_BATCH ('batched')."""
    import hashlib
    doc = {"meta": {"reference": reference, "cc": "9.0", "triton_stacks": list(RECORD_STACKS), "sizes": list(RECORD_SIZES), "channels": list(RECORD_CHANNELS),
                    "cells_fixture_sha256": hashlib.sha256(open(cells_path, "rb").read()).hexdigest(),
                    "what": "per call: selected cells row (cfg + source key), resolve_cfg k1/k3, Np, every device allocation (shape, dtype), K1/K3 launch grid + constexpr tile args + argument shapes, bmm operand shapes/strides",
                    "above_2048": "N in {3072, 4096} x C in {128, 256} x direction x mask x residual, both triton stacks",
                    "batched": "z [B, N, N, C] with B = %d, N in %s x C in {128, 256} x direction x per-b mask x residual, both triton stacks: ONE K1 launch (grid axis 2 = B), ONE bmm over B*D planes, ONE K3 launch" % (RECORD_BATCH["B"], list(RECORD_BATCH["sizes"]))},
           "records": {}, "above_2048": {}, "batched": {}}
    for st in RECORD_STACKS:
        doc["records"][st] = record_all(pkg, cells_path, RECORD_SIZES, RECORD_CHANNELS, triton_mm=st)
        doc["above_2048"][st] = record_all(pkg, cells_path, RECORD_ABOVE, RECORD_CHANNELS, triton_mm=st)
        doc["batched"][st] = record_all(pkg, cells_path, RECORD_BATCH["sizes"], RECORD_CHANNELS, B=RECORD_BATCH["B"], triton_mm=st)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True); fh.write("\n")
    return doc
