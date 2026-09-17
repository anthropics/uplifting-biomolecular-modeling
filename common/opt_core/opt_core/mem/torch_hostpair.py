"""A pinned-host mirror of a pair tensor served to the device in row blocks, a pinned buffer pool, and the row-split guard for
``layer_norm`` inputs at or above 2**31 elements (torch imported lazily inside the functions).

Mechanism. An offload line keeps the pair representation ``[N, N, C]`` (or ``[B, N, N, C]``) in host RAM and hands each trunk sub-op
one row block on the device: device peak becomes O(rows x N x C) for every row-wise op; the price is PCIe traffic (a line built on
this is slower than the resident line below the single-card ceiling and exists for inputs beyond it). Pinned (page-locked) host
memory is what makes the H2D/D2H copies DMA transfers; torch's caching host allocator never returns freed pinned blocks to the OS,
so buffers are pooled by (numel, dtype) instead of re-allocated per call. The LN guard exists because a single ``layer_norm`` launch
over >= 2**31 (torch >= 2.8: 2**32) elements is outside what some torch builds compute correctly; normalising per leading-dim block
is the same arithmetic per row (mean/var are per row).

API (refusals are :class:`opt_core.mem.MemLeverRefused`; CUDA OOM propagates):
    PinPool(max_bytes, ...)            THE pinned host-buffer pool of the package (every pinned host byte a memory line, the output
                                       writer or a host mirror holds is allocated here): ``alloc(shape, dtype, tag)`` -> a fresh host
                                       tensor, page-locked unless ``pin=False`` (plain host tensors: the CPU line), under a byte budget
                                       (``max_bytes``; None = unbudgeted) and a host-RAM check (``reserve_bytes``: request + reserve must
                                       fit MemAvailable; None = no check); a request the pool cannot honour is a :class:`PinRefused`
                                       naming the kind (``pin_budget_exceeded`` · ``host_ram_insufficient`` · ``host_ram_unknown`` ·
                                       ``pin_alloc_failed``) — or, for a pool built with ``pageable=True``, a pageable buffer COUNTED in
                                       ``n_pageable`` with a first-per-kind entry in ``events`` (never silent); ``release(t)`` returns the
                                       bytes to the budget; ``bytes_now`` · ``bytes_peak`` · ``n_alloc``; thread-safe
    host_mem_available_bytes()         MemAvailable of /proc/meminfo in bytes (``host_ram_unknown`` when unreadable)
    HostChunks(nbytes, pool, pin=, chunk=) an EXACT-SIZE host copy of a contiguous tensor's bytes as a list of ``uint8`` chunks from ``pool``
                                       (``ROWPAIR_PARK_CHUNK_GIB`` GiB each, default 8, a power of two — torch's caching host allocator rounds
                                       every pinned block up to the next power of two, so one 147.5 GiB tensor costs 256 GiB but 8 GiB chunks
                                       cost 8 GiB; the tail is split into power-of-two pieces >= 64 MiB; ``=0``: one allocation as before):
                                       ``store(t)`` / ``load(t)`` (all bytes), ``store_range`` / ``load_range`` (a byte range <-> any contiguous
                                       tensor: row slabs), ``release()``; facts ``where`` / ``chunks`` / ``alloc_bytes`` / ``pinned_bytes``; byte
                                       copies only (bit-exact). The host-buffer statement of the row-shard parks and the tri-mul row mirror
    chunk_bytes() / chunk_sizes(n, c)  the env word -> bytes per chunk; the allocation sizes of a chunked copy of n bytes
    PinnedPool(lever)                  DEPRECATED alias: the ``(numel, dtype)``-keyed buffer cache over
                                       :class:`PinPool` — ``get(shape, dtype, pin)`` -> a VIEW of the cached buffer for that key (one live
                                       user per key); a refused page-lock raises unless ``pageable_ok=True`` (then ``ledger`` counts
                                       ``host_pageable_buffers``); ``bytes()``, ``is_pinned``, ``drop(key=None)``
    HostPair(z | shape, ...)           ``rows(r0, r1)`` -> device block; ``put_rows(r0, r1, blk)``; ``cols``/``put_cols`` through a
                                       staging buffer; ``from_tensor(z, rows)``; ``to_tensor(device)``; ``nbytes``; ledger keys
                                       ``h2d_bytes`` / ``d2h_bytes``
    ln_rowsplit(fn, x, limit, ...)     ``fn`` (a layer_norm-like row-wise callable) applied per leading-dim block when
                                       ``x.numel() >= limit``; else ``fn(x)`` unchanged — the building block of a kit's LN guard; a split
                                       dim inside the normalized dims is refused by name
    LayerNormGuard(limit, lever)       ``install()`` wraps ``torch.nn.functional.layer_norm`` process-wide with :func:`ln_rowsplit`
                                       (idempotent; ``remove()`` restores; ``hits`` counted in the ledger as ``ln_guard_hits``) — a kit
                                       whose stock LayerNorm class calls ``F.layer_norm`` is covered; one that does not wraps its own
                                       class's forward with :func:`ln_rowsplit` instead
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Sequence, Tuple

from . import Ledger, MemLeverRefused, row_blocks
from .torch_rowchunk import torch_module

__all__ = ["PinPool", "PinRefused", "PIN_REFUSALS", "host_mem_available_bytes", "PinnedPool", "HostPair", "HostChunks", "chunk_bytes",
           "chunk_sizes", "ENV_CHUNK_GIB", "ln_rowsplit", "LayerNormGuard", "LN_LIMIT_DEFAULT", "GiB"]

LN_LIMIT_DEFAULT = 1 << 31


GiB = 1 << 30
PIN_REFUSALS = {                                   # kind -> what it means (the vocabulary a lever re-words its own refusal names from)
    "pin_budget_exceeded": "the pinned total would exceed the pool's byte budget",
    "host_ram_insufficient": "request + reserve exceeds the host's MemAvailable",
    "host_ram_unknown": "MemAvailable could not be read (/proc/meminfo)",
    "pin_alloc_failed": "the driver refused to page-lock the buffer",
}


class PinRefused(MemLeverRefused):
    """A :class:`PinPool` refusal: ``kind`` is one of :data:`PIN_REFUSALS`, ``details`` the figures (GiB) the reason sentence quotes."""

    def __init__(self, lever: str, kind: str, reason: str, **details: Any):
        super().__init__(lever, f"{kind}: {reason}")
        self.kind, self.details = str(kind), dict(details)
        self.detail = str(reason)                                   # the sentence without the kind prefix (``reason`` carries both)


def host_mem_available_bytes(meminfo: str = "/proc/meminfo", lever: str = "pinpool") -> int:
    """MemAvailable from /proc/meminfo in bytes; :class:`PinRefused` ``host_ram_unknown`` when it cannot be read."""
    try:
        with open(meminfo) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError as e:
        raise PinRefused(lever, "host_ram_unknown", f"{meminfo}: {e}") from None
    raise PinRefused(lever, "host_ram_unknown", f"{meminfo} has no MemAvailable line")


class PinPool:
    """THE pool of pinned (page-locked) host buffers. ``alloc`` hands out a FRESH host tensor of the requested shape and dtype under the
    pool's byte budget and host-RAM check, page-locked unless the pool (or the call) says ``pin=False``; a request it cannot honour is
    refused BY NAME (:class:`PinRefused`) — never a silent pageable buffer. A pool built with ``pageable=True`` (the output writer's
    staging pool) answers a refused request with a pageable buffer instead, COUNTED: ``n_pageable`` and a first-per-kind ``events`` entry
    the caller prints as its FALLBACK evidence. ``release(t)`` returns a buffer's bytes to the budget (the tensor itself goes back to
    torch's caching host allocator when the caller drops it; re-use policies — byte buckets, typed keys — are the caller's, over this
    allocator). Checks, in order: budget (``bytes_now + request <= max_bytes``) · host RAM (pinned requests only: ``request +
    reserve_bytes <= avail()``) · the page-lock itself. ``avail`` is the MemAvailable reader (:func:`host_mem_available_bytes` unless given)."""
    def __init__(self, max_bytes: Optional[int] = None, *, reserve_bytes: Optional[int] = None, pin: bool = True, pageable: bool = False,
                 lever: str = "pinpool", avail=None):
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.reserve_bytes = None if reserve_bytes is None else int(reserve_bytes)
        self.pin = bool(pin)
        self.pageable = bool(pageable)
        self.lever = str(lever)
        self._avail = avail if avail is not None else (lambda: host_mem_available_bytes(lever=self.lever))
        self.bytes_now = 0
        self.bytes_peak = 0
        self.n_alloc = 0
        self.n_pageable = 0
        self.events: list = []
        self._live: Dict[int, int] = {}
        self._lock = threading.RLock()

    def alloc(self, shape: Sequence[int], dtype: Any, tag: str = "buf", pin: Optional[bool] = None) -> Any:
        """A fresh host tensor ``shape`` x ``dtype`` (page-locked unless ``pin=False`` here or at construction), or :class:`PinRefused`."""
        return self.alloc_counted(shape, dtype, tag, pin)[0]

    def alloc_counted(self, shape: Sequence[int], dtype: Any, tag: str = "buf", pin: Optional[bool] = None) -> Tuple[Any, Optional[str]]:
        """``(tensor, None)`` for a request served as asked; ``(pageable tensor, <kind>)`` when a ``pageable=True`` pool answered a refused
        request with a pageable buffer (the caller's FALLBACK evidence names ``kind``); :class:`PinRefused` otherwise."""
        torch = torch_module(self.lever)
        shape = tuple(int(s) for s in shape)
        numel = 1
        for s in shape:
            numel *= s
        nbytes = numel * int(torch.empty((), dtype=dtype).element_size())
        pin = self.pin if pin is None else bool(pin)
        gb = round(nbytes / GiB, 3)
        with self._lock:                                       # reserved before allocating: concurrent callers respect the budget
            if self.max_bytes is not None and self.bytes_now + nbytes > self.max_bytes:
                return self._deny(torch, shape, dtype, "pin_budget_exceeded", f"{tag}: pinned total would exceed pin_max_gb", tag=tag,
                                  request_gb=gb, pinned_gb=round(self.bytes_now / GiB, 3), pin_max_gb=round(self.max_bytes / GiB, 3))
            self.bytes_now += nbytes
        try:
            if pin:
                if self.reserve_bytes is not None:
                    avail = int(self._avail())
                    if nbytes + self.reserve_bytes > avail:
                        raise PinRefused(self.lever, "host_ram_insufficient", f"{tag}: request + host_reserve_gb exceeds MemAvailable", tag=tag,
                                         request_gb=gb, avail_gb=round(avail / GiB, 3), host_reserve_gb=round(self.reserve_bytes / GiB, 3))
                try:
                    t = torch.empty(shape, dtype=dtype, pin_memory=True)
                except RuntimeError as e:
                    raise PinRefused(self.lever, "pin_alloc_failed", f"{tag}: {str(e)[:160]}", tag=tag, request_gb=gb) from None
            else:
                t = torch.empty(shape, dtype=dtype)
        except BaseException as exc:
            with self._lock:
                self.bytes_now -= nbytes
            if isinstance(exc, PinRefused):
                return self._deny(torch, shape, dtype, exc.kind, exc.detail, **exc.details)
            raise
        with self._lock:
            self.bytes_peak = max(self.bytes_peak, self.bytes_now)
            self.n_alloc += 1
            self._live[id(t)] = nbytes
        return t, None

    def _deny(self, torch, shape, dtype, kind: str, reason: str, **details):
        """Refuse by name — or, for a ``pageable=True`` pool, hand out a pageable buffer outside the budget, counted and recorded."""
        if not self.pageable:
            raise PinRefused(self.lever, kind, reason, **details)
        with self._lock:
            self.n_pageable += 1
            if all(e["kind"] != kind for e in self.events):
                self.events.append({"kind": kind, "reason": reason, **details})
        return torch.empty(shape, dtype=dtype), kind

    def live(self) -> Tuple[int, int]:
        """``(count, bytes)`` of the buffers this pool handed out that were not released back to it (the 'kept' figure of the pinned-pool shrink line)."""
        with self._lock:
            return len(self._live), int(sum(self._live.values()))

    def release(self, t) -> None:
        with self._lock:
            self.bytes_now -= self._live.pop(id(t), 0)


class PinnedPool:
    """DEPRECATED alias: the ``(numel, dtype)``-keyed host-buffer cache over :class:`PinPool`.
    ``get`` returns a view with the requested shape over the cached buffer of that key (one live user per key at a time — an offload
    line owns its pool); page-locking, its refusal and the pageable policy are :class:`PinPool`'s."""

    def __init__(self, lever: str = "hostpair", pageable_ok: bool = False, ledger: Optional[Ledger] = None):
        self.lever = str(lever)
        self.pageable_ok = bool(pageable_ok)
        self.ledger = ledger
        self.pool = PinPool(None, pageable=self.pageable_ok, lever=self.lever)
        self._lock = threading.Lock()
        self._bufs: Dict[Tuple[int, Any], Any] = {}
        self._pinned: Dict[Tuple[int, Any], bool] = {}

    def get(self, shape: Sequence[int], dtype: Any = None, pin: bool = True) -> Any:
        torch = torch_module(self.lever)
        dtype = dtype or torch.float32
        numel = 1
        for s in shape:
            numel *= int(s)
        key = (numel, dtype)
        with self._lock:
            buf = self._bufs.get(key)
            if buf is None:
                want_pin = bool(pin) and torch.cuda.is_available()
                try:
                    buf = self.pool.alloc((numel,), dtype, tag=f"{numel}x{dtype}", pin=want_pin)
                except PinRefused as r:
                    raise MemLeverRefused(self.lever, f"pinned host allocation of {numel * _itemsize(torch, dtype) / 2 ** 30:.2f} GiB "
                                                      f"refused ({r.kind}: {r.detail[:120]}); pageable_ok not declared") from None
                pinned = bool(want_pin and buf.is_pinned())
                if pin and not pinned and self.ledger is not None:
                    self.ledger.count("host_pageable_buffers")
                self._bufs[key] = buf
                self._pinned[key] = pinned
                if self.ledger is not None:
                    self.ledger.count("host_pinned_bytes" if pinned else "host_pageable_bytes", numel * _itemsize(torch, dtype))
            return buf.view(*[int(s) for s in shape])

    def is_pinned(self, shape: Sequence[int], dtype: Any) -> bool:
        numel = 1
        for s in shape:
            numel *= int(s)
        return bool(self._pinned.get((numel, dtype), False))

    def bytes(self) -> int:
        torch = torch_module(self.lever)
        with self._lock:
            return sum(int(b.numel()) * _itemsize(torch, b.dtype) for b in self._bufs.values())

    def drop(self, key: Optional[Tuple[int, Any]] = None) -> None:
        with self._lock:
            keys = list(self._bufs) if key is None else [key]
            for k in keys:
                buf = self._bufs.pop(k, None)
                self._pinned.pop(k, None)
                if buf is not None:
                    self.pool.release(buf)


def _itemsize(torch, dtype) -> int:
    try:
        return int(torch.empty((), dtype=dtype).element_size())
    except Exception:  # noqa: BLE001
        return 4



# ----------------------------------------------------------------------------------------------------------------- host cache trim
def proc_mem() -> Dict[str, float]:
    """This process's ``VmRSS`` / ``VmHWM`` / ``VmLck`` / ``VmPin`` from ``/proc/self/status`` in GiB (keys absent when unreadable) — the
    resident-set evidence of :func:`host_cache_trim` (pinned host pages count in VmRSS; ``VmPin``/``VmLck`` stay 0 for CUDA-pinned pages on
    most kernels: they are pinned by the driver, not mlock'ed)."""
    out: Dict[str, float] = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                k = line.split(":", 1)[0]
                if k in ("VmRSS", "VmHWM", "VmLck", "VmPin", "VmSize"):
                    out[k] = round(int(line.split()[1]) / (1024.0 * 1024.0), 3)      # kB -> GiB
    except (OSError, ValueError, IndexError):
        pass
    return out


def host_cached_bytes(lever: str = "pool") -> Optional[Dict[str, int]]:
    """torch's caching HOST allocator statistics when this torch exposes them (``torch.cuda.host_memory_stats``, torch >= 2.5):
    ``{"allocated": live bytes, "reserved": bytes held from the OS (live + cached-free)}``; None when the build has no such call."""
    torch = torch_module(lever)
    fn = getattr(getattr(torch, "cuda", None), "host_memory_stats", None)
    if fn is None:
        return None
    try:
        st = fn()
    except Exception:  # noqa: BLE001 - a statistics probe (memory instrument, OOM census NOT_REROUTES): a binding that refuses (no CUDA context) has no statistics; the [pool] line says `cached-free n/a`
        return None
    def pick(*names):
        for n in names:
            v = st.get(n)
            if isinstance(v, (int, float)):
                return int(v)
        return None
    alloc = pick("allocated_bytes.current", "allocated_bytes.allocated", "active_bytes.current")
    res = pick("reserved_bytes.current", "segment.current_bytes", "reserved_bytes.allocated")
    if alloc is None and res is None:
        return None
    return {"allocated": alloc if alloc is not None else -1, "reserved": res if res is not None else -1}


def host_cache_trim(lever: str = "pool") -> Dict[str, Any]:
    """Return torch's CACHED-FREE pinned host blocks to the OS. torch's caching host allocator keeps every freed page-locked block for
    re-use (``cudaHostAlloc`` is slow), so a released park / row mirror stays resident until this call: ``torch._C._host_emptyCache()``
    (torch >= 2.5) ``cudaFreeHost``s every cached block no tensor uses; LIVE pinned tensors are untouched (no tensor value can change).
    Returns ``{"call": <name>|None, "available": bool, "ok": bool, "error": str|None, "reserved_before"/"reserved_after": bytes|None,
    "rss_before"/"rss_after": GiB}`` — a build without the binding answers ``available=False`` BY NAME (the caller says so; never a silent no-op)."""
    torch = torch_module(lever)
    call = None
    for name in ("_host_emptyCache", "_cuda_hostEmptyCache", "_cuda_host_emptyCache"):
        fn = getattr(torch._C, name, None)
        if callable(fn):
            call = name
            break
    if call is None:
        mem = getattr(getattr(torch, "cuda", None), "memory", None)
        fn = getattr(mem, "host_empty_cache", None) or getattr(getattr(torch, "cuda", None), "host_empty_cache", None)
        if callable(fn):
            call = "torch.cuda.host_empty_cache"
    before = host_cached_bytes(lever); rss0 = proc_mem().get("VmRSS")
    out: Dict[str, Any] = {"call": call, "available": call is not None, "ok": False, "error": None,
                           "reserved_before": (before or {}).get("reserved"), "allocated_before": (before or {}).get("allocated"), "rss_before": rss0}
    if call is None:
        out["error"] = "no host-allocator empty-cache binding in torch " + str(getattr(torch, "__version__", "?"))
        out["rss_after"] = rss0; out["reserved_after"] = out["reserved_before"]
        return out
    try:
        import gc
        gc.collect()                                                            # dead HostChunks tensors must be collected before their blocks count as cached-free
        fn()
        out["ok"] = True
    except Exception as e:  # noqa: BLE001 - a memory instrument (OOM census NOT_REROUTES), never a gate: a refusing binding is NAMED on the [pool] line (TRIM UNAVAILABLE: <error>)
        out["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    after = host_cached_bytes(lever)
    out["reserved_after"] = (after or {}).get("reserved"); out["allocated_after"] = (after or {}).get("allocated"); out["rss_after"] = proc_mem().get("VmRSS")
    return out

class HostPair:
    """A pair tensor resident in host memory (pinned via ``pool`` when a CUDA device is the target), served in row blocks.
    ``HostPair(shape, dtype, device, pool)`` allocates; ``HostPair.from_tensor(z, rows, ...)`` copies an existing tensor in (row block by
    row block, so a device ``z`` never needs a second full-size device copy). Row blocks along ``row_dim`` (default: dim -3 of
    ``[.., N, N, C]``) are contiguous slabs (plain DMA); column blocks go through a pooled staging buffer. ``ledger`` records ``h2d_bytes``
    and ``d2h_bytes``."""

    def __init__(self, shape: Sequence[int], dtype: Any = None, device: Any = "cuda", pool: Optional[PinnedPool] = None,
                 ledger: Optional[Ledger] = None, row_dim: int = -3, lever: str = "hostpair"):
        torch = torch_module(lever)
        self.lever = str(lever)
        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) < 3:
            raise MemLeverRefused(lever, f"HostPair needs a pair shape [.., N, N, C], got {self.shape}")
        self.row_dim = row_dim % len(self.shape)
        self.col_dim = self.row_dim + 1
        if self.col_dim >= len(self.shape) - 1 or self.shape[self.row_dim] != self.shape[self.col_dim]:
            raise MemLeverRefused(lever, f"HostPair shape {self.shape} is not square on dims ({self.row_dim}, {self.col_dim})")
        self.n = self.shape[self.row_dim]
        self.dtype = dtype or torch.float32
        self.device = torch.device(device)
        self.ledger = ledger
        self.pool = pool if pool is not None else PinnedPool(lever, ledger=ledger)
        self.t = self.pool.get(self.shape, self.dtype, pin=(self.device.type == "cuda"))
        self.pinned = self.pool.is_pinned(self.shape, self.dtype)
        self._stage_key = None

    @property
    def nbytes(self) -> int:
        return int(self.t.numel()) * int(self.t.element_size())

    def _count(self, key: str, t: Any) -> None:
        if self.ledger is not None:
            self.ledger.count(key, int(t.numel()) * int(t.element_size()))

    def rows(self, r0: int, r1: int, non_blocking: bool = False) -> Any:
        """``z[.., r0:r1, :, :]`` as a device tensor (H2D of a contiguous slab)."""
        x = self.t.narrow(self.row_dim, r0, r1 - r0)
        self._count("h2d_bytes", x)
        return x.to(self.device, non_blocking=non_blocking)

    def put_rows(self, r0: int, r1: int, blk: Any, non_blocking: bool = False) -> None:
        dst = self.t.narrow(self.row_dim, r0, r1 - r0)
        if tuple(blk.shape) != tuple(dst.shape):
            raise MemLeverRefused(self.lever, f"put_rows({r0},{r1}): block {tuple(blk.shape)} != slab {tuple(dst.shape)}")
        self._count("d2h_bytes", blk)
        dst.copy_(blk, non_blocking=non_blocking)

    def cols(self, c0: int, c1: int) -> Any:
        """``z[.., :, c0:c1, :]`` as a CONTIGUOUS device tensor (strided host read through a pooled staging buffer)."""
        src = self.t.narrow(self.col_dim, c0, c1 - c0)
        stage = self.pool.get(tuple(src.shape), self.dtype, pin=(self.device.type == "cuda"))
        if stage.data_ptr() == self.t.data_ptr():
            raise MemLeverRefused(self.lever, "cols(): staging buffer aliases the pair buffer (column block as large as the tensor)")
        stage.copy_(src)
        self._count("h2d_bytes", stage)
        return stage.to(self.device)

    def put_cols(self, c0: int, c1: int, blk: Any) -> None:
        dst = self.t.narrow(self.col_dim, c0, c1 - c0)
        if tuple(blk.shape) != tuple(dst.shape):
            raise MemLeverRefused(self.lever, f"put_cols({c0},{c1}): block {tuple(blk.shape)} != slab {tuple(dst.shape)}")
        self._count("d2h_bytes", blk)
        dst.copy_(blk)

    @classmethod
    def from_tensor(cls, z: Any, rows: int = 256, device: Any = None, pool: Optional[PinnedPool] = None,
                    ledger: Optional[Ledger] = None, row_dim: int = -3, lever: str = "hostpair") -> "HostPair":
        torch = torch_module(lever)
        dev = device if device is not None else (z.device if z.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
        hp = cls(tuple(z.shape), z.dtype, dev, pool=pool, ledger=ledger, row_dim=row_dim, lever=lever)
        for r0, r1 in row_blocks(hp.n, rows):
            hp.put_rows(r0, r1, z.narrow(hp.row_dim, r0, r1 - r0))
        return hp

    def to_tensor(self, device: Any = None) -> Any:
        x = self.t.to(self.device if device is None else device)
        self._count("h2d_bytes", x)
        return x

    def free(self) -> None:
        """Drop this pair's pooled buffer (the pool keeps nothing for its key afterwards)."""
        key = (int(self.t.numel()), self.dtype)
        self.t = None
        self.pool.drop(key)


# ----------------------------------------------------------------------------------------------------------------- exact-size host copies
ENV_CHUNK_GIB = "ROWPAIR_PARK_CHUNK_GIB"        # GiB per pinned host chunk of a chunked host copy (default 8; 0 = one allocation of the whole size)
CHUNK_GIB_DEFAULT = 8
CHUNK_TAIL_MIN = 64 << 20                        # the tail of a chunked copy is split into power-of-two pieces down to this size (waste < this)


def chunk_bytes(chunk_gib=None, lever: str = "hostchunks") -> int:
    """Bytes per host chunk: ``chunk_gib`` (else ``ROWPAIR_PARK_CHUNK_GIB``, default :data:`CHUNK_GIB_DEFAULT`) GiB; 0 = one allocation of the
    whole size (no chunking). A negative / non-numeric word is refused by name."""
    import os
    word = os.environ.get(ENV_CHUNK_GIB, "").strip() if chunk_gib is None else str(chunk_gib)
    if not word:
        return CHUNK_GIB_DEFAULT * GiB
    try:
        g = float(word)
    except ValueError:
        raise MemLeverRefused(lever, f"{ENV_CHUNK_GIB}={word!r}: a number of GiB (0 = one allocation) is required") from None
    if g < 0:
        raise MemLeverRefused(lever, f"{ENV_CHUNK_GIB}={word!r}: must be >= 0")
    return int(g * GiB)


def next_pow2(n: int) -> int:
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def chunk_sizes(nbytes: int, chunk: int) -> list:
    """The allocation sizes of a chunked host copy of ``nbytes``: ``chunk`` (a power of two is exact under torch's caching host allocator,
    which rounds every pinned block UP to the next power of two) repeated, then the tail as descending power-of-two pieces no smaller than
    :data:`CHUNK_TAIL_MIN`, the last piece rounded up to that size (so the rounded total exceeds ``nbytes`` by less than ``CHUNK_TAIL_MIN``).
    ``chunk <= 0``: ONE allocation of ``nbytes`` (today's single host tensor: the allocator rounds it to the next power of two)."""
    nbytes, chunk = int(nbytes), int(chunk)
    if nbytes <= 0:
        return []
    if chunk <= 0 or nbytes <= CHUNK_TAIL_MIN:
        return [nbytes]
    sizes = [chunk] * (nbytes // chunk)
    rest = nbytes - chunk * len(sizes)
    while rest > CHUNK_TAIL_MIN:
        p = 1 << (rest.bit_length() - 1)                    # largest power of two <= rest
        sizes.append(p)
        rest -= p
    if rest > 0:
        sizes.append(CHUNK_TAIL_MIN)
    return sizes


class HostChunks:
    """An exact-size host copy of a CONTIGUOUS tensor's bytes as a list of host CHUNKS (``uint8`` tensors of :func:`chunk_sizes`) allocated
    through a :class:`PinPool` (page-locked when ``pin``; a refused page-lock answered pageable by a ``pageable=True`` pool is NAMED in
    ``fallback`` / ``where``): the one host-buffer statement of the row-shard family's parks and of the triangle-multiplication row mirror. Why
    chunks: torch's caching host allocator rounds each pinned allocation up to the next power of two (a 147.5 GiB tensor occupies a 256 GiB
    block); chunks of a power-of-two size occupy exactly their size and are re-used across parks through the allocator's free lists.
    ``store(t)`` copies ``t``'s bytes in (device -> host: ``non_blocking`` copies on the current stream; the caller synchronizes before reading on
    the host or releasing device storage), ``load(t)`` copies them back into a tensor of the same byte size, ``load_range(dst, offset)`` fills any
    contiguous tensor from byte ``offset`` on (row slabs of a parked shard), ``store_range(src, offset)`` the inverse; ``release()`` returns every
    chunk to the pool. Facts: ``nbytes`` (payload), ``alloc_bytes`` (sum of chunk sizes), ``pinned_bytes`` (what the pinned chunks occupy in the
    allocator: each rounded to the next power of two), ``n_chunks``, ``pinned`` (every chunk page-locked), ``where`` (``host_pinned`` | ``host`` (plain,
    ``pin=False``) | ``host_pageable:<kind>``), ``fallback`` (``<kind>`` | None). Byte copies only: bit-exact by construction."""

    def __init__(self, nbytes: int, pool: "PinPool", *, pin: bool = True, chunk: Optional[int] = None, tag: str = "host_chunks",
                 lever: str = "hostchunks"):
        torch = torch_module(lever)
        self.lever, self.tag, self.pool = str(lever), str(tag), pool
        self.nbytes = int(nbytes)
        self.chunk = chunk_bytes(None, lever) if chunk is None else int(chunk)
        self.sizes = chunk_sizes(self.nbytes, self.chunk)
        self.chunks: list = []
        self.fallback: Optional[str] = None
        self.pinned = bool(pin)
        self.alloc_bytes = 0
        self.pinned_bytes = 0
        complete = False
        try:
            for i, sz in enumerate(self.sizes):
                t, kind = pool.alloc_counted((sz,), torch.uint8, tag=f"{tag}[{i}]", pin=bool(pin))
                self.chunks.append(t)
                self.alloc_bytes += sz
                is_pinned = bool(pin) and kind is None and bool(getattr(t, "is_pinned", lambda: False)())
                if is_pinned:
                    self.pinned_bytes += next_pow2(sz)
                else:
                    self.pinned = False
                if kind is not None and self.fallback is None:
                    self.fallback = str(kind)
            complete = True
        finally:
            if not complete:                                     # a refused chunk: the chunks already taken go back to the pool, the refusal propagates
                self.release()
        if not self.sizes:
            self.pinned = False
        self.where = (f"host_pageable:{self.fallback}" if self.fallback is not None else ("host_pinned" if self.pinned else "host"))

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def facts(self) -> Dict[str, Any]:
        return {"where": self.where, "fallback": self.fallback, "pinned": bool(self.pinned), "chunks": self.n_chunks,
                "gib": round(self.nbytes / GiB, 3), "alloc_gib": round(self.alloc_bytes / GiB, 3), "pinned_gib": round(self.pinned_bytes / GiB, 3),
                "chunk_gib": round(self.chunk / GiB, 3)}

    # --- byte views ----------------------------------------------------------------------------------------------------------------
    def _flat(self, t):
        """``t``'s bytes as a flat ``uint8`` view (no copy): contiguous tensors only."""
        if not t.is_contiguous():
            raise MemLeverRefused(self.lever, f"{self.tag}: a contiguous tensor is required (strides {tuple(t.stride())})")
        torch = torch_module(self.lever)
        if t.dtype == torch.uint8:
            return t.reshape(-1)
        return t.detach().reshape(-1).view(torch.uint8)

    def _span(self, offset: int, n: int):
        """Yield ``(chunk, c0, c1, d0, d1)``: chunk bytes ``c0:c1`` <-> payload bytes ``offset + d0 : offset + d1`` covering ``[offset, offset + n)``."""
        offset, n = int(offset), int(n)
        if offset < 0 or n < 0 or offset + n > self.nbytes:
            raise MemLeverRefused(self.lever, f"{self.tag}: byte range [{offset}, {offset + n}) outside [0, {self.nbytes})")
        if self.released:
            raise MemLeverRefused(self.lever, f"{self.tag}: released")
        base, d = 0, 0
        for ch, sz in zip(self.chunks, self.sizes):
            if n <= 0:
                break
            if offset < base + sz:
                c0 = offset - base
                take = min(sz - c0, n)
                yield ch, c0, c0 + take, d, d + take
                d += take
                offset += take
                n -= take
            base += sz

    @property
    def released(self) -> bool:
        return bool(self.sizes) and not self.chunks

    # --- copies --------------------------------------------------------------------------------------------------------------------
    def store_range(self, src, offset: int = 0, non_blocking: bool = True) -> None:
        """Copy the bytes of the contiguous tensor ``src`` into payload bytes ``[offset, offset + src.nbytes)``."""
        flat = self._flat(src)
        for ch, c0, c1, d0, d1 in self._span(offset, int(flat.numel())):
            ch[c0:c1].copy_(flat[d0:d1], non_blocking=non_blocking)

    def load_range(self, dst, offset: int = 0, non_blocking: bool = True):
        """Fill the contiguous tensor ``dst`` from payload bytes ``[offset, offset + dst.nbytes)``; returns ``dst``."""
        flat = self._flat(dst)
        for ch, c0, c1, d0, d1 in self._span(offset, int(flat.numel())):
            flat[d0:d1].copy_(ch[c0:c1], non_blocking=non_blocking)
        return dst

    def store(self, t, non_blocking: bool = True) -> None:
        """Copy ALL of ``t``'s bytes in (``t.nbytes == nbytes`` required)."""
        n = int(t.numel()) * int(t.element_size())
        if n != self.nbytes:
            raise MemLeverRefused(self.lever, f"{self.tag}: store of {n} B into a {self.nbytes} B host copy")
        self.store_range(t, 0, non_blocking)

    def load(self, t, non_blocking: bool = True):
        """Copy ALL bytes back into ``t`` (same byte size); returns ``t``."""
        n = int(t.numel()) * int(t.element_size())
        if n != self.nbytes:
            raise MemLeverRefused(self.lever, f"{self.tag}: load of a {self.nbytes} B host copy into {n} B")
        return self.load_range(t, 0, non_blocking)

    def release(self) -> None:
        """Return every chunk to the pool (the tensors go back to torch's caching host allocator: pinned blocks stay cached for re-use)."""
        chunks, self.chunks = self.chunks, []
        for ch in chunks:
            self.pool.release(ch)


def ln_rowsplit(fn, x: Any, limit: int = LN_LIMIT_DEFAULT, *, dim: Optional[int] = None, per: Optional[int] = None,
                normalized_ndim: int = 1, ledger: Optional[Ledger] = None, key: str = "ln_guard_hits", lever: str = "ln_guard") -> Any:
    """``fn(x)`` when ``x.numel() < limit`` (or x has no dim outside the normalized ones); else ``fn`` applied to blocks of ``x`` along
    ``dim`` written into one ``empty_like(x)`` output. ``normalized_ndim``: how many TRAILING dims ``fn`` normalizes over (layer_norm's
    ``len(normalized_shape)``); ``dim``: the dim to split (default: the first dim of size > 1 outside the normalized dims) — a ``dim`` inside
    the normalized dims raises :class:`MemLeverRefused` (splitting it would change the statistics); ``per``: rows per block (default: the
    largest count keeping a block under ``limit // 2`` elements). ``fn`` must be row-wise over the leading dims."""
    torch = torch_module(lever)
    nd = int(normalized_ndim)
    if nd < 1:
        raise MemLeverRefused(lever, f"normalized_ndim must be >= 1 (got {normalized_ndim})")
    lead = x.dim() - nd                                   # dims [0, lead) are row dims; [lead, x.dim()) are normalized
    if int(x.numel()) < int(limit) or lead < 1:
        return fn(x)
    d = dim
    if d is not None:
        d = d % x.dim()
        if d >= lead:
            raise MemLeverRefused(lever, f"split dim {dim} lies inside the normalized dims (last {nd} of shape {tuple(x.shape)})")
    else:
        d = next((i for i, s in enumerate(x.shape[:lead]) if int(s) > 1), None)
        if d is None:
            return fn(x)
    n = int(x.shape[d])
    inner = int(x.numel()) // n
    rows = int(per) if per else max(1, (int(limit) // 2) // max(inner, 1))
    out = torch.empty_like(x)
    for r0, r1 in row_blocks(n, rows):
        out.narrow(d, r0, r1 - r0).copy_(fn(x.narrow(d, r0, r1 - r0)))
    if ledger is not None:
        ledger.count(key)
    return out


class LayerNormGuard:
    """Process-wide wrap of ``torch.nn.functional.layer_norm`` with :func:`ln_rowsplit` (``torch.nn.LayerNorm.forward`` calls it through
    the module attribute, so stock LayerNorm modules are covered). ``install()`` is idempotent and returns True when it installed;
    ``remove()`` restores the original. A kit scopes it by installing only inside its memory line; ``hits`` are counted in ``ledger``."""

    def __init__(self, limit: int = LN_LIMIT_DEFAULT, lever: str = "ln_guard", ledger: Optional[Ledger] = None):
        self.limit = int(limit)
        self.lever = str(lever)
        self.ledger = ledger if ledger is not None else Ledger()
        self._orig = None

    def install(self) -> bool:
        torch = torch_module(self.lever)
        F = torch.nn.functional
        if getattr(F.layer_norm, "_opt_core_ln_guard", None) is not None:
            return False
        orig = F.layer_norm
        guard = self

        def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):  # noqa: A002 — torch's own signature
            nd = len(normalized_shape) if hasattr(normalized_shape, "__len__") else 1
            return ln_rowsplit(lambda blk: orig(blk, normalized_shape, weight, bias, eps), input, guard.limit, normalized_ndim=nd,
                               ledger=guard.ledger, lever=guard.lever)
        layer_norm._opt_core_ln_guard = guard
        layer_norm.__wrapped__ = orig
        self._orig = orig
        F.layer_norm = layer_norm
        return True

    def remove(self) -> bool:
        torch = torch_module(self.lever)
        F = torch.nn.functional
        if self._orig is None or getattr(F.layer_norm, "_opt_core_ln_guard", None) is not self:
            return False
        F.layer_norm = self._orig
        self._orig = None
        return True

    @property
    def hits(self) -> int:
        return int(self.ledger.get("ln_guard_hits", 0) or 0)
