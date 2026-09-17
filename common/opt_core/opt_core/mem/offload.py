"""Host offload of pair / trunk tensors: pinned-RAM parking, whole-tensor residency per call, and row/column-block streaming.

Contract. ONE registry lever, ``host_park`` (family ``offload``; declared with :func:`opt_core.mem.registry.register` at import, so
``mem.apply((..., "host_park"), ctx)`` selects it — the kit's switch ``switches={"host_park": on/off}``, its settings in
``ctx.settings["host_park"]``), with two lines under that one switch: the *resident* line parks named tensors (or every
parameter / buffer of a named module) in pinned host memory and brings a whole tensor back to the device per call
(:meth:`HostPark.resident`, :meth:`HostPark.park_module`; the census names them ``resident:tensor`` / ``resident:module``); the
*streamed* line never materialises the parked pair tensor on the device — :func:`stream_blocks` runs a block function over row or
column blocks through a fixed-size device window of ``window`` block buffers, prefetching block k+1 on a side CUDA stream while block
k computes and writing block k's result back on the same side stream (census: ``streamed``). Everything the lever needs is here and
named: the settings (:class:`Settings`, read through ``ctx.setting``: ``pin_max_gb`` · ``min_tokens`` · ``block`` · ``window`` ·
``cols`` · ``host_reserve_gb``; the device is the hook ``ctx.hooks["host_park"]["device"]``), the preconditions (``applies(ctx)`` =
:func:`_applies`, then :meth:`HostPark.check`), the size gate (:meth:`HostPark.gate` — the crossing is a recorded event and a census
``skip`` ``below_gate:<n>/<min>`` when the input is under ``min_tokens``), the pinned budget (:class:`PinPool`: a request
above ``pin_max_gb`` or above the host's available RAM, or a failed pin, is a refusal — never a pageable buffer), the copy path
(rows = contiguous cudaMemcpyAsync through torch; columns = pitched ``cudaMemcpy2DAsync`` through the process's own libcudart, or
the recorded ``cols="rowloop"`` alternative — one memcpy per row, the same bytes, never chosen silently), the stream discipline
(per-slot events ready / done / written; a fence from the compute stream before the side stream writes any device buffer the
compute stream allocated; a slot is refilled only after its write-back event; the compute stream never touches a block before its
ready event; :func:`stream_blocks` returns only after every write-back landed; :meth:`HostPark.release` / ``close`` quiesce both
streams before a pinned buffer is freed, because the pitched cudart copies are invisible to torch's pinned allocator), and the
record: the :class:`opt_core.mem.registry.Applied` of the lever (settings in force, ``undo`` = ``close``), the census marks / skips
on the kit's ``AppliedRecord``, and :meth:`HostPark.record` (the lever's detailed record: settings, every check, every event, every
refusal, the lines engaged, pinned peak, bytes moved, the copy library found) for the kit's manifest.

Refusals are BY NAME: every silently degraded path is an :class:`OffloadRefusal` here — a
:class:`opt_core.mem.registry.RefusalError` whose precondition is a key of the closed catalogue :data:`REFUSALS` (the lever's declared
``preconditions``); the tests hold every raised name to it. No stock fallback, no pageable fallback, no ``return original`` when a
name is not parked, no untested copy path; a refusal raised on a run-time path propagates (the unit fails by name) and is noted on
the kit's record. The lever is inference-only (``grad_enabled``) and eager-only (``graphs``).

Exactness. The resident line is ``bitwise``: parking and restoring are byte copies and the op runs on the device at its own shapes.
The streamed line is exact only for a block-separable ``fn`` — every output row (column) block depends on its own input block alone
(pair transitions, layer norms over the channel dim, per-row softmax / row-wise attention, elementwise updates); a triangle
multiplication or an update that reads other rows is NOT streamable by this helper (an engine streams it through
:meth:`HostTensor.rows` / :meth:`HostTensor.cols` itself). Block-separable is necessary, not sufficient, for ``bitwise``: a GEMM inside
``fn`` runs at M = block·N instead of N·N and cuBLAS may select a different kernel (a different reduction order) — one source
measured a Hopper kernel switch near M ≈ 1e6 rows. The caller therefore DECLARES the label per call (``exact="bitwise" | "band" | "measured"``
with a reason); the record keeps the declaration and the kit's equality run measures it. Nothing here re-orders arithmetic on
its own: the copies are exact and ``fn`` is the engine's own code.

Adapter hook spec — what an engine passes. (1) ``"host_park"`` on its big line and the settings in ``ctx.settings["host_park"]``
(``pin_max_gb`` and ``min_tokens`` are the engine's line values — no library default; the flags
``ctx.settings["host_park"]`` keys ``pin_max_gb, min_tokens, block, window, cols, host_reserve_gb``; a string value is cast) and the device in
``ctx.hooks["host_park"]["device"]``; after ``mem.apply(...)`` the handle is ``park = park_of(ctx)``. (2) The token count of the item
for :meth:`HostPark.gate` (inside the kit's ``unit_begin`` / ``unit_end``). (3) Module references and attribute paths for the
resident line: ``park.park_module(module, "name")`` for the modules whose weights leave the device between calls (the diffusion /
confidence stacks while the trunk runs), ``park.park("z_init", tensor)`` for the pair tensors an engine holds across blocks or
recycles (the recycling copy, template pair, the diffusion pair cache). (4) For the streamed line: the pair tensor in
``[..., N, N, C]`` layout (leading dims flattened as a batch; every leading entry is copied as one slab), the block dim (``"rows"`` =
the -3 axis, ``"cols"`` = the -2 axis), and a block-separable ``fn(block, i0, i1) -> block`` closing over the engine's module (the
transition, the row-wise attention, the layer norm), with its ``exact`` label and reason. A line-wide ``block`` larger than an
item's N is clamped to N as a recorded event. The engine class map lives in the engine's adapter. The mechanism is engine-free: no
engine name, import, shape or default in this file.

Every mechanism here is engine-free: this module names no kit and no engine.

Torch is imported lazily: importing this module needs only the standard library and ``opt_core.mem.registry`` (the core's rule);
every function that touches a tensor imports torch at call time.
"""
from __future__ import annotations

import ctypes
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .registry import EXACT_LABELS, Applied, Ctx, Refusal, RefusalError, refuse, register, setting_ref
from .torch_hostpair import GiB, PinPool as _PinPool, PinRefused, host_mem_available_bytes as _host_mem_available_bytes

LEVER = "host_park"
FAMILY = "offload"
SETTINGS = ("pin_max_gb", "min_tokens", "block", "window", "cols", "host_reserve_gb")     # the keys of ctx.settings["host_park"]

# ----------------------------------------------------------------------------------------------------------------- refusals (by name)
REFUSALS: Dict[str, str] = {
    "no_cuda": "the settings name a cuda device and torch reports no cuda device",
    "bad_setting": "a setting is missing, malformed or out of range (the value and the range are in the details)",
    "pin_budget_exceeded": "a pinned allocation would take the pinned total above pin_max_gb (never pageable instead)",
    "host_ram_insufficient": "a pinned allocation exceeds the host's MemAvailable minus host_reserve_gb (never pageable, never swap)",
    "host_ram_unknown": "/proc/meminfo is unreadable so the host RAM check cannot run (the check is a precondition, not optional)",
    "pin_alloc_failed": "torch could not pin the buffer (locked-memory limit); a pageable buffer is not a substitute",
    "cudart_unavailable": "cols='pitched' needs cudaMemcpy2DAsync from a loadable libcudart and none was found (cols='rowloop' is the named alternative)",
    "cudart_error": "a cudaMemcpy2DAsync call returned a non-zero cudaError_t (the code is in the details)",
    "not_parked": "the name is not parked (the source kits returned the original tensor here)",
    "already_parked": "the name is already parked; park once, restore or unpark first",
    "not_contiguous": "only a contiguous tensor is parked (a strided source would be copied through a device temporary of its own size)",
    "layout_unsupported": "the streamed line needs a [..., N, N, C] tensor (N == N on the two token axes), a block dim of rows or cols, and dense blocks",
    "window_invalid": "block must be in [1, N] and window >= 2 (one buffer computing, one prefetching)",
    "fn_shape_mismatch": "fn returned a block whose shape or dtype differs from its input block (the streamed contract)",
    "exact_unlabelled": "a streamed call must declare exact='bitwise' | 'band' | 'measured' with a reason (recorded, then measured)",
    "device_mismatch": "a tensor handed to the lever lives on a device other than the one the settings name",
    "graphs": "CUDA-graph capture is in the composed line (ctx.graphs) or the current stream is capturing: the side-stream ring and the host copies cannot be captured — the trunk runs eagerly",
    "grad_enabled": "autograd is enabled (a requires_grad tensor or grad mode on a streamed call): the lever is inference-only; run under torch.no_grad()",
    "not_applied": "host_park was not applied on this ctx (put 'host_park' on the big line and mem.apply(...) first)",
    "closed": "the HostPark was closed; its buffers are released",
}

_COLS_MODES = ("pitched", "rowloop")


class OffloadRefusal(RefusalError):
    """A named refusal of the lever: a :class:`opt_core.mem.registry.RefusalError` whose :class:`Refusal` names ``host_park``, the
    precondition ``kind`` (a key of :data:`REFUSALS`, kept as ``.name``), the one-line ``reason`` and the ``details`` that decided it."""

    def __init__(self, kind: str, reason: str, **details: Any):
        assert kind in REFUSALS, kind
        self.name, self.reason, self.details = kind, reason, details
        super().__init__(refuse(LEVER, kind, reason, **details))

    def gate(self):
        """The refusal in the core's :class:`opt_core.gates.Gate` form (name, ok=False, reason, details)."""
        from opt_core.gates import Gate

        return Gate(name=f"{LEVER}:{self.name}", ok=False, reason=self.reason, details=dict(self.details))


# ----------------------------------------------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Settings:
    """The lever's named settings (all recorded). ``pin_max_gb`` and ``min_tokens`` have no default: the engine's big line states them.

    ``block`` rows (columns) per streamed block; ``window`` device block buffers (>= 2); ``cols`` = ``"pitched"`` (cudaMemcpy2DAsync,
    one call per column block per leading entry) or ``"rowloop"`` (one memcpy per row — the same bytes, more launches, no cudart
    dependency); ``host_reserve_gb`` = host RAM kept free of pinned buffers; ``device`` = ``"cuda"`` (or ``"cuda:<i>"``) for the lever,
    ``"cpu"`` for the CPU test line, which runs the same block schedule with plain copies, no pinning and no streams, and makes no
    memory claim (an engine adapter never passes it).
    """

    pin_max_gb: float
    min_tokens: int
    block: int = 256
    window: int = 2
    cols: str = "pitched"
    host_reserve_gb: float = 4.0
    device: str = "cuda"

    def __post_init__(self):
        if isinstance(self.pin_max_gb, bool) or not (isinstance(self.pin_max_gb, (int, float)) and self.pin_max_gb > 0):
            raise OffloadRefusal("bad_setting", "pin_max_gb must be a positive GiB value", pin_max_gb=self.pin_max_gb)
        if isinstance(self.min_tokens, bool) or not (isinstance(self.min_tokens, int) and self.min_tokens >= 0):
            raise OffloadRefusal("bad_setting", "min_tokens must be an int >= 0", min_tokens=self.min_tokens)
        if isinstance(self.block, bool) or not (isinstance(self.block, int) and self.block >= 1):
            raise OffloadRefusal("bad_setting", "block must be an int >= 1", block=self.block)
        if isinstance(self.window, bool) or not (isinstance(self.window, int) and self.window >= 2):
            raise OffloadRefusal("bad_setting", "window must be an int >= 2", window=self.window)
        if self.cols not in _COLS_MODES:
            raise OffloadRefusal("bad_setting", f"cols must be one of {_COLS_MODES}", cols=self.cols)
        if not (isinstance(self.host_reserve_gb, (int, float)) and self.host_reserve_gb >= 0):
            raise OffloadRefusal("bad_setting", "host_reserve_gb must be >= 0", host_reserve_gb=self.host_reserve_gb)
        if not (self.device == "cpu" or self.device == "cuda" or self.device.startswith("cuda:")):
            raise OffloadRefusal("bad_setting", "device must be 'cpu', 'cuda' or 'cuda:<i>'", device=self.device)

    @classmethod
    def from_ctx(cls, ctx: Ctx) -> "Settings":
        """The settings through the core grammar: ``ctx.setting("host_park", <name>)`` = the kit's ``ctx.settings["host_park"][<name>]``
        when set, else the kit's ``ctx.settings["host_park"][<name>]``, else the class default (``pin_max_gb`` / ``min_tokens`` have
        none: absent = ``bad_setting`` by name); ``device`` is the hook ``ctx.hooks["host_park"]["device"]`` (default ``"cuda"``). A
        flag that fails its cast is the registry's refusal naming the flag; every value read is noted on the record."""
        casts = {"pin_max_gb": float, "min_tokens": int, "block": int, "window": int, "cols": str, "host_reserve_gb": float}
        vals: Dict[str, Any] = {}
        for key in SETTINGS:
            v = ctx.setting(LEVER, key, None, cast=casts[key])
            if v is not None:
                vals[key] = v
        missing = [k for k in ("pin_max_gb", "min_tokens") if k not in vals]
        if missing:
            raise OffloadRefusal("bad_setting", "required settings absent (the engine's big line states them; no library default)",
                                 missing=missing, settings=[setting_ref(LEVER, m) for m in missing])
        vals["device"] = ctx.hook(LEVER, "device", "cuda")
        return cls(**vals)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------------------------------------------- host RAM + cudart
def host_mem_available_bytes(meminfo: str = "/proc/meminfo") -> int:
    """MemAvailable from /proc/meminfo in bytes (the package's one reader, :func:`opt_core.mem.torch_hostpair.host_mem_available_bytes`);
    ``host_ram_unknown`` when it cannot be read."""
    try:
        return _host_mem_available_bytes(meminfo, lever=LEVER)
    except PinRefused as r:                                       # the reader's one kind
        raise OffloadRefusal("host_ram_unknown", r.detail) from None


_CUDART: Dict[str, Any] = {"lib": None, "path": None, "tried": False}


def _loaded_cudart_paths() -> List[str]:
    """Paths of every libcudart mapped into this process (the copy torch loaded comes first in the candidate order)."""
    paths: List[str] = []
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                parts = line.split()
                p = parts[-1] if len(parts) >= 6 else ""
                if "libcudart" in os.path.basename(p) and p not in paths:
                    paths.append(p)
    except OSError:
        pass
    return paths


def cudart() -> Tuple[Any, str]:
    """The process's libcudart with ``cudaMemcpy2DAsync`` bound (the copy already mapped by torch first, then the nvidia wheel,
    then torch/lib, then the soname); ``cudart_unavailable`` by name when none loads. Cached per process."""
    if _CUDART["tried"]:
        if _CUDART["lib"] is None:
            raise OffloadRefusal("cudart_unavailable", "no libcudart with cudaMemcpy2DAsync loadable via ctypes", tried=_CUDART["path"])
        return _CUDART["lib"], _CUDART["path"]
    _CUDART["tried"] = True
    cands = _loaded_cudart_paths()
    try:
        import nvidia.cuda_runtime as ncr  # type: ignore

        d = os.path.join(os.path.dirname(ncr.__file__), "lib")
        cands += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.startswith("libcudart.so")]
    except Exception:
        pass
    try:
        import torch

        d = os.path.join(os.path.dirname(torch.__file__), "lib")
        cands += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.startswith("libcudart")]
    except Exception:
        pass
    cands += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
    tried = []
    for c in cands:
        try:
            lib = ctypes.CDLL(c)
            f = lib.cudaMemcpy2DAsync
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                          ctypes.c_int, ctypes.c_void_p]
            _CUDART["lib"], _CUDART["path"] = lib, c
            return lib, c
        except Exception:
            tried.append(c)
    _CUDART["path"] = tried
    raise OffloadRefusal("cudart_unavailable", "no libcudart with cudaMemcpy2DAsync loadable via ctypes", tried=tried)


_H2D, _D2H = 1, 2  # cudaMemcpyKind


def memcpy2d_async(dst_ptr: int, dpitch: int, src_ptr: int, spitch: int, width: int, height: int, kind: int, stream: int) -> None:
    """One pitched copy of ``height`` rows of ``width`` bytes (``cudaMemcpy2DAsync``); a non-zero cudaError_t is ``cudart_error``."""
    lib, _ = cudart()
    rc = lib.cudaMemcpy2DAsync(ctypes.c_void_p(dst_ptr), ctypes.c_size_t(dpitch), ctypes.c_void_p(src_ptr), ctypes.c_size_t(spitch),
                               ctypes.c_size_t(width), ctypes.c_size_t(height), ctypes.c_int(kind), ctypes.c_void_p(stream))
    if rc != 0:
        raise OffloadRefusal("cudart_error", "cudaMemcpy2DAsync failed", cudaError=int(rc), memcpy_kind=kind, width=width, height=height)


def _dense_block(t3) -> bool:
    """``[n, w, c]`` with a dense ``[w, c]`` inner plane (any row pitch): what a pitched copy can address."""
    return t3.dim() == 3 and t3.stride(2) == 1 and t3.stride(1) == t3.shape[2]


# ----------------------------------------------------------------------------------------------------------------- streams (cuda | cpu)
class _Streams:
    """The side copy stream and events on CUDA; every method a no-op on CPU (the same schedule, no concurrency)."""

    def __init__(self, device: str):
        self.cuda = device != "cpu"
        if self.cuda:
            import torch

            self.torch = torch
            self.dev = torch.device(device)
            self.copy = torch.cuda.Stream(device=self.dev)
        else:
            self.torch = None
            self.dev = None
            self.copy = None

    def compute(self):
        return self.torch.cuda.current_stream(self.dev) if self.cuda else None

    def event(self):
        return self.torch.cuda.Event() if self.cuda else None

    def record(self, ev, stream):
        if self.cuda:
            ev.record(stream)

    def wait(self, stream, ev):
        if self.cuda:
            stream.wait_event(ev)

    def fence(self, src, dst):
        """``dst`` waits everything enqueued on ``src`` so far (a device buffer allocated on ``src`` may be written on ``dst``)."""
        if self.cuda and src is not dst:
            ev = self.event()
            ev.record(src)
            dst.wait_event(ev)

    @contextmanager
    def on(self, stream):
        if self.cuda:
            with self.torch.cuda.stream(stream):
                yield
        else:
            yield

    def raw(self, stream) -> int:
        return int(stream.cuda_stream) if self.cuda else 0

    def sync(self, stream):
        if self.cuda:
            stream.synchronize()


# ----------------------------------------------------------------------------------------------------------------- pinned budget
class PinPool(_PinPool):
    """The lever's binding of the package's ONE pinned host-buffer pool (:class:`opt_core.mem.torch_hostpair.PinPool`): budget
    ``pin_max_gb``, host-RAM reserve ``host_reserve_gb`` (request + reserve must fit MemAvailable), plain host tensors on
    ``device="cpu"`` (recorded ``pinned=False``; only the budget is enforced — the CPU test line, no memory claim), refusals worded as
    this lever's (:class:`OffloadRefusal` ``pin_budget_exceeded`` · ``host_ram_insufficient`` · ``host_ram_unknown`` · ``pin_alloc_failed``);
    ``alloc`` never returns a pageable buffer; ``release`` returns the bytes to the budget."""

    def __init__(self, settings: Settings):
        super().__init__(int(settings.pin_max_gb * GiB), reserve_bytes=int(settings.host_reserve_gb * GiB), pin=settings.device != "cpu",
                         pageable=False, lever=LEVER, avail=lambda: host_mem_available_bytes())
        self.s = settings

    def alloc(self, shape: Sequence[int], dtype, tag: str):
        try:
            return super().alloc(shape, dtype, tag)
        except PinRefused as r:                                   # the pool's kinds ARE this lever's refusal names (one literal raise site each)
            if r.kind == "pin_budget_exceeded":
                raise OffloadRefusal("pin_budget_exceeded", r.detail, **r.details) from None
            if r.kind == "host_ram_insufficient":
                raise OffloadRefusal("host_ram_insufficient", r.detail, **r.details) from None
            if r.kind == "host_ram_unknown":
                raise OffloadRefusal("host_ram_unknown", r.detail, **r.details) from None
            if r.kind == "pin_alloc_failed":
                raise OffloadRefusal("pin_alloc_failed", r.detail, **r.details) from None
            raise OffloadRefusal(r.kind, r.detail, **r.details) from None


# ----------------------------------------------------------------------------------------------------------------- a parked tensor
class HostTensor:
    """A parked tensor: ``host`` (pinned, contiguous, the parked shape), block accessors that copy on the given stream (the compute
    stream by default) and ``to_device`` / ``put`` for the whole tensor. Layout for rows / cols: ``[..., N, N, C]`` — leading dims are
    one flattened batch; ``rows(i0, i1)`` → ``[..., i1-i0, N, C]``; ``cols(j0, j1)`` → ``[..., N, j1-j0, C]`` (pitched or rowloop).
    Every accessor is ASYNCHRONOUS on its stream and returns no event: synchronise the stream before reading ``host`` after a
    ``put*`` (``resident`` / ``stream_blocks`` do). Pinned-buffer lifetime: torch tracks the copies it issues (``copy_`` / ``to``)
    and gates the pinned block's reuse on their events; the pitched ``cudaMemcpy2DAsync`` copies are invisible to torch, so
    :meth:`HostPark.release` / :meth:`HostPark.close` synchronise the side and compute streams themselves before a buffer is freed."""

    def __init__(self, park: "HostPark", name: str, host, shape: Tuple[int, ...], dtype):
        self.park, self.name, self.host, self.shape, self.dtype = park, name, host, tuple(shape), dtype
        self.pinned = park.pool.pin
        self.nbytes = host.numel() * host.element_size()

    # --- layout helpers ---------------------------------------------------------------------------------------------------------
    def _pair_dims(self) -> Tuple[int, int, int, int]:
        """(B, N, N, C) of the flattened layout; ``layout_unsupported`` otherwise."""
        if len(self.shape) < 3 or self.shape[-3] != self.shape[-2]:
            raise self.park._refuse("layout_unsupported", "rows/cols need [..., N, N, C]", name=self.name, shape=self.shape)
        b = int(math.prod(self.shape[:-3])) if len(self.shape) > 3 else 1
        return b, self.shape[-3], self.shape[-2], self.shape[-1]

    def _flat(self):
        b, n, _, c = self._pair_dims()
        return self.host.view(b, n, n, c)

    def _out(self, shape, out, stream):
        """A device buffer for a block: the caller's ``out`` (fenced by the caller) or a fresh one, fenced from the compute stream."""
        import torch

        st = self.park.streams
        if out is not None:
            return out
        g = torch.empty(shape, dtype=self.dtype, device=self.park.device)
        st.fence(st.compute(), stream)
        return g

    # --- whole tensor ----------------------------------------------------------------------------------------------------------
    def to_device(self, stream=None):
        """The whole parked tensor on the device (H2D on ``stream`` or the compute stream); the host copy stays parked."""
        st = self.park.streams
        stream = stream or st.compute()
        with st.on(stream):
            g = self.host.to(self.park.device, non_blocking=True, copy=True)
        self.park._moved(h2d=self.nbytes)
        return g

    def put(self, g, stream=None) -> None:
        """Write a whole device tensor back (D2H) into the parked buffer; shape and dtype must match."""
        self.park._check_like(self, g)
        st = self.park.streams
        stream = stream or st.compute()
        with st.on(stream):
            self.host.copy_(g, non_blocking=True)
        self.park._moved(d2h=self.nbytes)

    # --- row blocks: contiguous slabs per leading entry -------------------------------------------------------------------------
    def _check_out(self, out, shape) -> None:
        if out is not None and (tuple(out.shape) != tuple(shape) or out.dtype != self.dtype):
            raise self.park._refuse("fn_shape_mismatch", "out buffer differs from the block's shape/dtype", want=(tuple(shape), str(self.dtype)),
                                    got=(tuple(out.shape), str(out.dtype)))

    def rows(self, i0: int, i1: int, out=None, stream=None):
        """Row block ``[..., i1-i0, N, C]`` on the device (H2D into ``out`` or a fresh buffer), asynchronous on ``stream``."""
        b, n, _, c = self._pair_dims()
        st = self.park.streams
        stream = stream or st.compute()
        h = self._flat()
        self._check_out(out, (b, i1 - i0, n, c))
        out = self._out((b, i1 - i0, n, c), out, stream)
        with st.on(stream):
            for k in range(b):
                out[k].copy_(h[k, i0:i1], non_blocking=True)
        self.park._moved(h2d=out.numel() * out.element_size())
        return out.view(self.shape[:-3] + (i1 - i0, n, c))

    def put_rows(self, i0: int, i1: int, g, stream=None) -> None:
        """Write a row block back (D2H), asynchronous on ``stream``."""
        b, n, _, c = self._pair_dims()
        st = self.park.streams
        stream = stream or st.compute()
        h = self._flat()
        gv = g.reshape(b, i1 - i0, n, c)
        with st.on(stream):
            for k in range(b):
                h[k, i0:i1].copy_(gv[k], non_blocking=True)
        self.park._moved(d2h=g.numel() * g.element_size())

    # --- column blocks: pitched 2-D copies (or the rowloop alternative; plain copies on the CPU line) --------------------------
    def cols(self, j0: int, j1: int, out=None, stream=None):
        """Column block ``[..., N, j1-j0, C]`` on the device (pitched cudaMemcpy2DAsync, or rowloop), asynchronous on ``stream``."""
        b, n, _, c = self._pair_dims()
        st = self.park.streams
        stream = stream or st.compute()
        h = self._flat()
        w = j1 - j0
        self._check_out(out, (b, n, w, c))
        out = self._out((b, n, w, c), out, stream)
        es = self.host.element_size()
        if not st.cuda:
            for k in range(b):
                out[k].copy_(h[k, :, j0:j1])
        elif self.park.cols_mode == "pitched":
            for k in range(b):
                if not _dense_block(out[k]):
                    raise self.park._refuse("layout_unsupported", "column block destination is not dense over [w, C]", strides=tuple(out[k].stride()))
                memcpy2d_async(out[k].data_ptr(), out[k].stride(0) * es, h[k].data_ptr() + j0 * c * es, h[k].stride(0) * es,
                               w * c * es, n, _H2D, st.raw(stream))
        else:
            with st.on(stream):
                for k in range(b):
                    for i in range(n):
                        out[k, i].copy_(h[k, i, j0:j1], non_blocking=True)
        self.park._moved(h2d=out.numel() * out.element_size())
        return out.view(self.shape[:-3] + (n, w, c))

    def put_cols(self, j0: int, j1: int, g, stream=None) -> None:
        """Write a column block back (D2H), asynchronous on ``stream``; the pitched path needs a dense ``[N, w, C]`` source per
        leading entry (``layout_unsupported`` otherwise — never a temporary made on another stream)."""
        b, n, _, c = self._pair_dims()
        st = self.park.streams
        stream = stream or st.compute()
        h = self._flat()
        w = j1 - j0
        gv = g.reshape(b, n, w, c)
        es = self.host.element_size()
        if not st.cuda:
            for k in range(b):
                h[k, :, j0:j1].copy_(gv[k])
        elif self.park.cols_mode == "pitched":
            for k in range(b):
                src = gv[k]
                if not _dense_block(src):
                    raise self.park._refuse("layout_unsupported", "column block source is not dense over [w, C]", strides=tuple(src.stride()))
                memcpy2d_async(h[k].data_ptr() + j0 * c * es, h[k].stride(0) * es, src.data_ptr(), src.stride(0) * es,
                               w * c * es, n, _D2H, st.raw(stream))
        else:
            with st.on(stream):
                for k in range(b):
                    for i in range(n):
                        h[k, i, j0:j1].copy_(gv[k, i], non_blocking=True)
        self.park._moved(d2h=g.numel() * g.element_size())

    def block(self, dim: str, a: int, b: int, out=None, stream=None):
        return self.rows(a, b, out, stream) if dim == "rows" else self.cols(a, b, out, stream)

    def put_block(self, dim: str, a: int, b: int, g, stream=None) -> None:
        (self.put_rows if dim == "rows" else self.put_cols)(a, b, g, stream)


# ----------------------------------------------------------------------------------------------------------------- module parking
class ModulePark:
    """Every parameter / buffer of a module parked in pinned host memory (``p.data`` is the host buffer); ``resident`` brings them to the
    device for one call: as a context, or through the forward hooks installed by :meth:`HostPark.park_module` (device copies
    allocated on the compute stream, filled on the side stream behind a fence, the compute stream waits their ready event; dropped
    after the call — inference only, never written back)."""

    def __init__(self, park: "HostPark", name: str, module, what: str):
        self.park, self.name, self.module, self.what = park, name, module, what
        self.entries: List[Tuple[str, Any, Any]] = []       # (qualified name, parameter-or-buffer, host tensor)
        self.nbytes = 0
        self._hooks: List[Any] = []
        self._dev: List[Any] = []

    def _tensors(self):
        if self.what in ("parameters", "all"):
            for n, p in self.module.named_parameters():
                yield n, p
        if self.what in ("buffers", "all"):
            for n, b in self.module.named_buffers():
                yield n, b

    def resident_enter(self):
        import torch

        st = self.park.streams
        comp = st.compute()
        side = st.copy if st.cuda else None
        devs = [torch.empty(h.shape, dtype=h.dtype, device=self.park.device) for _, _, h in self.entries]
        st.fence(comp, side)
        ev = st.event()
        with st.on(side):
            for d, (_, _, h) in zip(devs, self.entries):
                d.copy_(h, non_blocking=True)
        st.record(ev, side)
        st.wait(comp, ev)
        for (_, t, _), d in zip(self.entries, devs):
            t.data = d
        self._dev = devs
        self.park._moved(h2d=self.nbytes)

    def resident_exit(self):
        for (_, t, h) in self.entries:
            t.data = h
        self._dev = []

    @contextmanager
    def resident(self):
        self.resident_enter()
        try:
            yield self.module
        finally:
            self.resident_exit()


# ----------------------------------------------------------------------------------------------------------------- the lever
class HostPark:
    """The ``host_park`` lever: parked tensors and modules under one pinned budget, one side stream, one record.

    ``check()`` runs the preconditions (``no_cuda``; ``cudart_unavailable`` when ``cols='pitched'``; the host RAM probe) and records
    them; ``gate(n_tokens)`` records the size-gate crossing; ``park`` / ``park_like`` / ``get`` / ``unpark`` / ``release`` /
    ``resident`` are the resident line; ``park_module`` / ``unpark_module`` park a module's tensors with per-call residency;
    :meth:`stream_blocks` is the streamed line; ``record()`` is the applied-record entry; ``close()`` releases everything (a closed
    park refuses by name). The lever never frees or mutates a device tensor it did not allocate.
    """

    def __init__(self, settings: Settings, tag: str = LEVER, ctx: Optional[Ctx] = None):
        self.s = settings
        self.tag = tag
        self.ctx = ctx                             # the kit's registry context: census marks / skips / notes land on ctx.record
        self.device = settings.device
        self.cols_mode = settings.cols
        self.pool = PinPool(settings)
        self.streams: Optional[_Streams] = None
        self.tensors: Dict[str, HostTensor] = {}
        self.modules: Dict[str, ModulePark] = {}
        self.events: List[Dict[str, Any]] = []
        self.refusals: List[Dict[str, Any]] = []
        self.checks: List[Dict[str, Any]] = []
        self.lines: List[str] = []                 # lines engaged, in order: resident:tensor | resident:module | streamed
        self.gate_crossed: Optional[bool] = None
        self.first_refusal: Optional[str] = None
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.cudart_path: Optional[str] = None
        self.closed = False
        self._t0 = time.time()

    # --- bookkeeping ------------------------------------------------------------------------------------------------------------
    def _event(self, kind: str, **kw: Any) -> None:
        self.events.append({"event": kind, "t": round(time.time() - self._t0, 3), **kw})

    def _refuse(self, kind: str, reason: str, **details: Any) -> OffloadRefusal:
        r = OffloadRefusal(kind, reason, **details)
        self._note(r)
        return r

    def _note(self, r: OffloadRefusal) -> None:
        self.refusals.append({"name": r.name, "reason": r.reason, "details": r.details})
        if self.first_refusal is None:
            self.first_refusal = r.name
        rec = getattr(self.ctx, "record", None)
        if rec is not None and hasattr(rec, "note"):
            rec.note(f"{LEVER} refused:{r.name}: {r.reason}")

    @property
    def mode_ran(self) -> Optional[str]:
        """The row-level mode field: the lines engaged (``resident:tensor`` · ``resident:module`` · ``streamed``, joined by ``+`` in
        the order engaged), else ``refused:<first refusal>``, else ``below_gate`` when the gate said so, else None."""
        if self.lines:
            return "+".join(self.lines)
        if self.first_refusal is not None:
            return f"refused:{self.first_refusal}"
        if self.gate_crossed is False:
            return "below_gate"
        return None

    def _engaged(self, line: str) -> None:
        if line not in self.lines:
            self.lines.append(line)
        rec = getattr(self.ctx, "record", None)
        if rec is not None and hasattr(rec, "mark"):
            rec.mark(LEVER, detail=line)

    def _moved(self, h2d: int = 0, d2h: int = 0) -> None:
        self.h2d_bytes += h2d
        self.d2h_bytes += d2h

    def _open(self) -> None:
        if self.closed:
            raise self._refuse("closed", "HostPark.close() was called")

    def _check_like(self, ht: HostTensor, g) -> None:
        if tuple(g.shape) != ht.shape or g.dtype != ht.dtype:
            raise self._refuse("fn_shape_mismatch", "tensor differs from the parked shape/dtype", name=ht.name,
                               parked=(ht.shape, str(ht.dtype)), given=(tuple(g.shape), str(g.dtype)))

    def _alloc(self, shape, dtype, tag: str):
        try:
            return self.pool.alloc(shape, dtype, tag)
        except OffloadRefusal as r:
            self._note(r)
            raise

    # --- preconditions ------------------------------------------------------------------------------------------------------------
    def check(self) -> None:
        """Preconditions, each recorded; the first absent one is a refusal by name (``no_cuda`` · ``host_ram_unknown`` ·
        ``cudart_unavailable`` · ``graphs`` when the ctx composes on CUDA-graph capture); creates the side stream."""
        import torch

        self._open()
        if self.ctx is not None and getattr(self.ctx, "graphs", False):
            raise self._refuse("graphs", "ctx.graphs is set: the composed line captures CUDA graphs")
        if self.device != "cpu":
            if not torch.cuda.is_available():
                raise self._refuse("no_cuda", f"settings.device={self.device} but torch.cuda.is_available() is False")
            self.checks.append({"check": "cuda", "ok": True, "device": torch.cuda.get_device_name(torch.device(self.device))})
            try:
                avail = host_mem_available_bytes()
            except OffloadRefusal as r:
                self._note(r)
                raise
            self.checks.append({"check": "host_ram", "ok": True, "avail_gb": round(avail / GiB, 2), "reserve_gb": self.s.host_reserve_gb})
            if self.cols_mode == "pitched":
                try:
                    _, self.cudart_path = cudart()
                except OffloadRefusal as r:
                    self._note(r)
                    raise
                self.checks.append({"check": "cudart", "ok": True, "path": self.cudart_path})
        else:
            self.checks.append({"check": "cpu_line", "ok": True, "note": "no pinning, no streams, no memory claim"})
        if self.streams is None:
            self.streams = _Streams(self.device)
        self._event("check", ok=True)

    def gate(self, n_tokens: int) -> bool:
        """The size gate: True when ``n_tokens >= min_tokens`` (the lever engages), False = ``mode_ran below_gate``; both recorded."""
        self._open()
        crossed = int(n_tokens) >= self.s.min_tokens
        self._event("gate", n_tokens=int(n_tokens), min_tokens=self.s.min_tokens, crossed=crossed)
        self.gate_crossed = crossed
        rec = getattr(self.ctx, "record", None)
        if not crossed and rec is not None and hasattr(rec, "skip"):
            rec.skip(LEVER, f"below_gate:{int(n_tokens)}/{self.s.min_tokens}")
        return crossed

    def _ready(self) -> _Streams:
        if self.streams is None:
            self.check()
        return self.streams  # type: ignore[return-value]

    def _check_device(self, t, what: str) -> None:
        import torch

        want = torch.device(self.device)
        if t.device.type != want.type or (want.index is not None and t.device.index not in (None, want.index)):
            raise self._refuse("device_mismatch", f"{what} is on {t.device}, settings.device={self.device}")

    # --- resident line: tensors ---------------------------------------------------------------------------------------------------
    def park(self, name: str, t, stream=None) -> HostTensor:
        """Park ``t`` (contiguous, on the settings' device) under ``name``: a pinned buffer of its shape, D2H on ``stream`` (the
        compute stream by default), synchronised before return so the caller may drop ``t`` (its device memory is the caller's)."""
        self._open()
        st = self._ready()
        if name in self.tensors:
            raise self._refuse("already_parked", f"{name} is parked", name=name)
        self._check_device(t, name)
        if getattr(t, "requires_grad", False):
            raise self._refuse("grad_enabled", f"{name} requires grad: the parked buffer would carry autograd history", name=name)
        if not t.is_contiguous():
            raise self._refuse("not_contiguous", f"{name} is not contiguous", name=name, strides=tuple(t.stride()))
        host = self._alloc(tuple(t.shape), t.dtype, name)
        stream = stream or st.compute()
        with st.on(stream):
            host.copy_(t, non_blocking=True)
        st.sync(stream)
        ht = HostTensor(self, name, host, tuple(t.shape), t.dtype)
        self.tensors[name] = ht
        self._moved(d2h=ht.nbytes)
        self._event("park", name=name, shape=list(t.shape), dtype=str(t.dtype), gb=round(ht.nbytes / GiB, 4), pinned=ht.pinned)
        self._engaged("resident:tensor")
        return ht

    def park_like(self, name: str, shape: Sequence[int], dtype) -> HostTensor:
        """An EMPTY parked buffer (the output of a streamed op, a double buffer); no D2H."""
        self._open()
        self._ready()
        if name in self.tensors:
            raise self._refuse("already_parked", f"{name} is parked", name=name)
        host = self._alloc(tuple(shape), dtype, name)
        ht = HostTensor(self, name, host, tuple(shape), dtype)
        self.tensors[name] = ht
        self._event("park_like", name=name, shape=list(shape), dtype=str(dtype), gb=round(ht.nbytes / GiB, 4), pinned=ht.pinned)
        return ht

    def get(self, name: str) -> HostTensor:
        """The parked tensor by name; ``not_parked`` by name otherwise (never the original)."""
        self._open()
        ht = self.tensors.get(name)
        if ht is None:
            raise self._refuse("not_parked", f"{name} is not parked", name=name, parked=sorted(self.tensors))
        return ht

    def unpark(self, name: str, stream=None):
        """Restore ``name`` to the device (H2D) and release its host buffer; returns the device tensor."""
        ht = self.get(name)
        g = ht.to_device(stream)
        self.release(name)
        self._event("unpark", name=name)
        return g

    def _quiesce(self) -> None:
        """Every copy in flight on the side and compute streams has landed (the pitched cudart copies are invisible to torch's
        pinned allocator, so the lever fences them itself before a pinned buffer is freed)."""
        st = self.streams
        if st is not None and st.cuda:
            st.sync(st.copy)
            st.sync(st.compute())

    def release(self, name: str) -> None:
        """Drop a parked tensor's host buffer (its bytes return to the budget) after every in-flight copy landed."""
        ht = self.get(name)
        self._quiesce()
        del self.tensors[name]
        self.pool.release(ht.host)
        ht.host = None
        self._event("release", name=name)

    def rename(self, old: str, new: str) -> None:
        """Re-label a parked tensor (a streamed output becomes the next block's input without a copy)."""
        ht = self.get(old)
        if new in self.tensors:
            raise self._refuse("already_parked", f"{new} is parked", name=new)
        del self.tensors[old]
        ht.name = new
        self.tensors[new] = ht

    @contextmanager
    def resident(self, name: str, writeback: bool = True) -> Iterator[Any]:
        """Per-call residency of a parked tensor: the whole tensor H2D on enter (yielded), D2H back into the same buffer on exit
        when ``writeback`` (the op's result replaces the parked bytes; synchronised), the device copy dropped on exit."""
        ht = self.get(name)
        st = self.streams
        g = ht.to_device()
        self._event("resident_enter", name=name)
        try:
            yield g
        finally:
            if writeback:
                ht.put(g)
                st.sync(st.compute())
            self._event("resident_exit", name=name, writeback=writeback)
            del g

    # --- resident line: modules -------------------------------------------------------------------------------------------------
    def park_module(self, module, name: str, what: str = "parameters", hooks: bool = True) -> ModulePark:
        """Park every parameter (``what="parameters"``), buffer (``"buffers"``) or both (``"all"``) of ``module`` under ``name``; with
        ``hooks`` the module's forward brings them to the device for the call and drops the copies after (inference only)."""
        self._open()
        self._ready()
        if name in self.modules:
            raise self._refuse("already_parked", f"module {name} is parked", name=name)
        if what not in ("parameters", "buffers", "all"):
            raise self._refuse("bad_setting", "what must be parameters | buffers | all", what=what)
        mp = ModulePark(self, name, module, what)
        try:
            for qn, t in mp._tensors():
                self._check_device(t, f"{name}.{qn}")
                if not t.data.is_contiguous():
                    raise self._refuse("not_contiguous", f"{name}.{qn} is not contiguous", name=f"{name}.{qn}")
                host = self._alloc(tuple(t.shape), t.dtype, f"{name}.{qn}")
                host.copy_(t.data, non_blocking=False)
                mp.entries.append((qn, t, host))
                mp.nbytes += host.numel() * host.element_size()
        except OffloadRefusal:
            for _, _, h in mp.entries:
                self.pool.release(h)
            raise
        for _, t, host in mp.entries:
            t.data = host
        if hooks:
            mp._hooks.append(module.register_forward_pre_hook(lambda m, inp, mp=mp: mp.resident_enter()))
            mp._hooks.append(module.register_forward_hook(lambda m, inp, out, mp=mp: mp.resident_exit()))
        self.modules[name] = mp
        self._moved(d2h=mp.nbytes)
        self._event("park_module", name=name, what=what, n=len(mp.entries), gb=round(mp.nbytes / GiB, 4), hooks=hooks)
        self._engaged("resident:module")
        return mp

    def unpark_module(self, name: str) -> None:
        """Restore the module's tensors to the device for good, remove the hooks, release the host buffers."""
        self._open()
        mp = self.modules.get(name)
        if mp is None:
            raise self._refuse("not_parked", f"module {name} is not parked", name=name, parked=sorted(self.modules))
        for h in mp._hooks:
            h.remove()
        for _, t, host in mp.entries:
            t.data = host.to(self.device, non_blocking=False, copy=True)
            self.pool.release(host)
        self._moved(h2d=mp.nbytes)
        del self.modules[name]
        self._event("unpark_module", name=name)

    # --- streamed line ------------------------------------------------------------------------------------------------------------
    def stream_blocks(self, fn: Callable, z, dim: str, block: Optional[int] = None, window: Optional[int] = None, out=None,
                      exact: Optional[str] = None, reason: str = "") -> HostTensor:
        """:func:`stream_blocks` with ``z`` / ``out`` given as parked names or :class:`HostTensor`; block / window default to the settings."""
        zt = self.get(z) if isinstance(z, str) else z
        ot = self.get(out) if isinstance(out, str) else out
        return stream_blocks(fn, zt, dim, self.s.block if block is None else block, self.s.window if window is None else window,
                             out=ot, exact=exact, reason=reason)

    # --- record -------------------------------------------------------------------------------------------------------------------
    def record(self) -> Dict[str, Any]:
        """The applied-record entry: settings, ``mode_ran`` (``resident:tensor`` / ``resident:module`` / ``streamed`` joined by ``+``
        | ``below_gate`` | ``refused:<name>`` | None), ``lines_ran``, ``gate_crossed``, checks, events, refusals, pinned peak / now,
        bytes moved, the cudart path, the parked names."""
        return {
            "lever": LEVER, "family": FAMILY, "tag": self.tag, "settings": self.s.as_dict(), "mode_ran": self.mode_ran, "ctx": self.ctx is not None,
            "lines_ran": list(self.lines), "gate_crossed": self.gate_crossed,
            "checks": list(self.checks), "events": list(self.events), "refusals": list(self.refusals),
            "pinned_peak_gb": round(self.pool.bytes_peak / GiB, 4), "pinned_now_gb": round(self.pool.bytes_now / GiB, 4),
            "pinned": self.pool.pin, "h2d_gb": round(self.h2d_bytes / GiB, 4), "d2h_gb": round(self.d2h_bytes / GiB, 4),
            "cudart": self.cudart_path,
            "parked": {n: {"shape": list(t.shape), "dtype": str(t.dtype), "gb": round(t.nbytes / GiB, 4)} for n, t in self.tensors.items()},
            "parked_modules": {n: {"what": m.what, "n": len(m.entries), "gb": round(m.nbytes / GiB, 4)} for n, m in self.modules.items()},
            "closed": self.closed,
        }

    def close(self) -> None:
        """Release every parked buffer and module (modules are restored to the device first); the park refuses afterwards."""
        if self.closed:
            return
        self._quiesce()
        for name in list(self.modules):
            self.unpark_module(name)
        for name in list(self.tensors):
            self.release(name)
        self.closed = True
        self._event("close")


def describe() -> Dict[str, Any]:
    """The lever's static description for a registry / mode line: name, family, lines, exactness per line with its reason, settings."""
    return {
        "name": LEVER, "family": FAMILY,
        "lines": {
            "resident": {"exact": "bitwise", "reason": "park/restore are byte copies; the op runs on the device at its own shapes"},
            "streamed": {"exact": "declared per call (bitwise | band | measured)",
                         "reason": "block-separable fn required; a GEMM in fn runs at M = block*N and cuBLAS may pick another kernel — measured"},
        },
        "settings": list(SETTINGS), "hooks": ("device",), "refusals": sorted(REFUSALS), "exact_labels": list(EXACT_LABELS),
    }


# ----------------------------------------------------------------------------------------------------------------- the registry entry
def _applies(ctx: Ctx) -> Optional[Refusal]:
    """The lever's precondition check for :func:`opt_core.mem.apply`: the settings resolve (``bad_setting`` / the flag's refusal), the
    ctx is not a graph-capturing line (``graphs``), and on a cuda device: cuda present (``no_cuda``), host RAM readable
    (``host_ram_unknown``), libcudart loadable when ``cols='pitched'`` (``cudart_unavailable``). No allocation, no stream."""
    try:
        settings = Settings.from_ctx(ctx)
    except RefusalError as e:
        return e.refusal
    if getattr(ctx, "graphs", False):
        return refuse(LEVER, "graphs", REFUSALS["graphs"])
    if settings.device != "cpu":
        try:
            import torch
        except ImportError as e:
            return refuse(LEVER, "no_cuda", f"torch is not importable: {e}")
        if not torch.cuda.is_available():
            return refuse(LEVER, "no_cuda", f"device={settings.device} but torch.cuda.is_available() is False")
        try:
            host_mem_available_bytes()
            if settings.cols == "pitched":
                cudart()
        except RefusalError as e:
            return e.refusal
    return None


@register(LEVER, family=FAMILY, exact="measured",
          exact_reason="resident line bitwise (park / restore are byte copies; the op runs at its own shapes); streamed line declared "
                       "per stream_blocks call (a block-separable fn is necessary, a shape-independent kernel choice is not given) and "
                       "measured by the identity row",
          applies=_applies, description="pinned-RAM parking of pair tensors / modules with per-call residency and row/column-block streaming",
          preconditions=tuple(REFUSALS), settings=SETTINGS)
def host_park(ctx: Ctx) -> Applied:
    """Apply (the registry contract): build the :class:`HostPark` from the ctx's settings and ``device`` hook, run :meth:`HostPark.check`
    (each precondition recorded), attach it as ``ctx.host_park`` (:func:`park_of`), and return the :class:`Applied` whose ``settings``
    are the values in force, whose ``undo`` is :meth:`HostPark.close`. The lever marks the census itself from its own paths
    (``resident:tensor`` / ``resident:module`` / ``streamed`` through ``ctx.record.mark``; ``below_gate`` through ``ctx.record.skip``);
    ``park.record()`` is the lever's detailed record for the kit's manifest."""
    settings = Settings.from_ctx(ctx)
    park = HostPark(settings, tag=ctx.tag, ctx=ctx)
    park.check()
    ctx.host_park = park                        # the adapter's handle (a dataclass instance takes the attribute)
    return Applied(lever=LEVER, settings={**settings.as_dict(), "cudart": park.cudart_path, "pinned": park.pool.pin},
                   sites=("ctx.host_park",),
                   notes=["HostPark attached as ctx.host_park (park_of(ctx)); lines mark the census as resident:tensor / resident:module / "
                          "streamed; the gate skips as below_gate:<n>/<min>; park.record() is the lever's detailed record"],
                   undo=park.close)


def park_of(ctx: Ctx) -> "HostPark":
    """The :class:`HostPark` the lever attached to ``ctx`` at apply time; ``not_applied`` by name otherwise."""
    park = getattr(ctx, "host_park", None)
    if park is None:
        raise OffloadRefusal("not_applied", REFUSALS["not_applied"], prefix=getattr(ctx, "prefix", None))
    return park


# ----------------------------------------------------------------------------------------------------------------- stream_blocks
def stream_blocks(fn: Callable, z: HostTensor, dim: str, block: int, window: int, out: Optional[HostTensor] = None,
                  exact: Optional[str] = None, reason: str = "") -> HostTensor:
    """Run ``fn(block_dev, a, b) -> block_dev`` over the ``dim`` (``"rows"`` | ``"cols"``) blocks of a parked pair tensor ``z``
    through ``window`` input and ``window`` output device block buffers; results land in ``out`` (a parked buffer of z's shape;
    ``None`` = in place into ``z``, valid because a block-separable fn reads only the block it writes). Schedule per block k with
    slot s = k % window: the side stream H2D-fills input slot s (behind the slot's previous done event) and records ready[s]; the
    compute stream waits ready[s], runs ``fn`` (shape/dtype held to the input block: ``fn_shape_mismatch``), waits written[s]
    (the slot's previous write-back), copies the result into output slot s, records done[s]; the side stream waits done[s],
    D2H-writes output slot s to the host, records written[s]; blocks k+1 … k+window-1 are in flight meanwhile. Returns after the
    side stream drained (every write-back landed; the host bytes are complete). ``exact`` must be declared (see the module
    docstring); the call is one recorded event."""
    import torch

    if not isinstance(z, HostTensor):
        raise OffloadRefusal("not_parked", f"z must be a parked HostTensor, got {type(z).__name__} (park it first)")
    park = z.park
    park._open()
    st = park._ready()
    if exact not in EXACT_LABELS or not str(reason or "").strip():
        raise park._refuse("exact_unlabelled", f"exact={exact!r}, reason={reason!r}; declare one of {EXACT_LABELS} with a non-empty reason",
                           fn=getattr(fn, "__name__", "?"))
    if torch.is_grad_enabled():
        raise park._refuse("grad_enabled", "torch grad mode is on for a streamed call (autograd would hold every block's intermediates)")
    if st.cuda and torch.cuda.is_current_stream_capturing():
        raise park._refuse("graphs", "the current CUDA stream is capturing a graph")
    if dim not in ("rows", "cols"):
        raise park._refuse("layout_unsupported", "dim must be 'rows' or 'cols'", dim=dim)
    b, n, _, c = z._pair_dims()
    if block > n:
        park._event("block_clamped", block=block, N=n)      # a line-wide block on a smaller item: one block of N rows, recorded
        block = n
    if block < 1 or window < 2:
        raise park._refuse("window_invalid", "block >= 1 and window >= 2", block=block, window=window, N=n)
    if out is not None and (out.shape != z.shape or out.dtype != z.dtype):
        raise park._refuse("fn_shape_mismatch", "out differs from z in shape/dtype", out=(out.shape, str(out.dtype)), z=(z.shape, str(z.dtype)))
    dst = out if out is not None else z
    nb = math.ceil(n / block)
    lead = z.shape[:-3]
    bshape = (b, block, n, c) if dim == "rows" else (b, n, block, c)
    comp = st.compute()
    side = st.copy if st.cuda else None
    bufs = [torch.empty(bshape, dtype=z.dtype, device=park.device) for _ in range(window)]
    obufs = [torch.empty(bshape, dtype=z.dtype, device=park.device) for _ in range(window)]
    st.fence(comp, side)
    ready = [st.event() for _ in range(window)]
    done = [st.event() for _ in range(window)]
    written = [st.event() for _ in range(window)]
    used = [False] * window

    def bounds(k: int) -> Tuple[int, int]:
        return k * block, min(n, (k + 1) * block)

    def slot_view(buf, a: int, e: int):
        return buf[:, : e - a] if dim == "rows" else buf[:, :, : e - a]

    def issue_h2d(k: int) -> None:
        s = k % window
        a, e = bounds(k)
        if used[s]:
            st.wait(side, done[s])
        z.block(dim, a, e, out=slot_view(bufs[s], a, e), stream=side)
        st.record(ready[s], side)

    for k in range(min(window, nb)):
        issue_h2d(k)
    t0 = time.time()
    for k in range(nb):
        s = k % window
        a, e = bounds(k)
        st.wait(comp, ready[s])
        blk = slot_view(bufs[s], a, e).view(lead + ((e - a, n, c) if dim == "rows" else (n, e - a, c)))
        o = fn(blk, a, e)
        if not torch.is_tensor(o) or tuple(o.shape) != tuple(blk.shape) or o.dtype != blk.dtype:
            raise park._refuse("fn_shape_mismatch", "fn must return a block of the input block's shape and dtype", k=k,
                               got=(tuple(getattr(o, "shape", ())), str(getattr(o, "dtype", None))), want=(tuple(blk.shape), str(blk.dtype)))
        if used[s]:
            st.wait(comp, written[s])
        oslot = slot_view(obufs[s], a, e)
        oslot.copy_(o.reshape(oslot.shape))
        del o
        st.record(done[s], comp)
        st.wait(side, done[s])
        dst.put_block(dim, a, e, oslot, stream=side)
        st.record(written[s], side)
        used[s] = True
        if k + window < nb:
            issue_h2d(k + window)
    if st.cuda:
        comp.wait_stream(side)
        st.sync(side)
    del bufs, obufs
    park._event("stream", name=z.name, out=dst.name, dim=dim, block=block, window=window, n_blocks=nb, N=n, C=c, batch=b,
                exact=exact, reason=reason, cols_mode=park.cols_mode if dim == "cols" else None, s=round(time.time() - t0, 3))
    park._engaged("streamed")
    return dst
