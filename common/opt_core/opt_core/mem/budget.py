"""Memory-budget arithmetic for memory lines: how many rows of a pair-shaped operation fit a byte budget, and the device's free-byte
figure a kit budgets against (pure functions; :func:`device_free_bytes` imports torch lazily and is the only one that needs it).

A kit's row count / channel chunk / capture cap is a VALUE in its mode table; when a kit sizes it at run time instead ("auto"), it does
so from these functions so the arithmetic (and its refusal when nothing fits) is worded once. The registry levers take the same values:
the ``rows`` setting / ``table`` hook of the ``chunk_*`` levers (:mod:`opt_core.mem.chunk`) is a value or :func:`rows_within` of a budget the
adapter states; their ``auto`` TOKEN threshold (:class:`opt_core.mem.chunk.AutoCalibration`, a quadratic peak model of the un-levered path)
budgets against the device total read by :func:`opt_core.arch.device_memory` — the raw driver figures; :func:`device_free_bytes` here is the
allocator-aware free figure (reserved-but-unallocated slack counted as free).

API:
    nbytes(shape, itemsize)                                  product(shape) * itemsize
    rows_within(budget_bytes, bytes_per_row, n, ...)         the largest row count r <= n with r * bytes_per_row <= budget_bytes, rounded
                                                             down to ``multiple``; below ``floor`` -> MemLeverRefused (nothing fits: the line
                                                             refuses by name rather than run a degenerate block size)
    device_free_bytes(device=None)                           free bytes on the CUDA device as the caching allocator can still hand out:
                                                             the driver's free (opt_core.arch.device_memory) + (reserved - allocated); None without CUDA
    budget_bytes(frac, reserve_bytes, device=None)           ``device_free_bytes() * frac - reserve_bytes`` (None without CUDA)
"""
from __future__ import annotations

from typing import Optional, Sequence

from . import MemLeverRefused

__all__ = ["nbytes", "rows_within", "device_free_bytes", "budget_bytes"]


def nbytes(shape: Sequence[int], itemsize: int) -> int:
    n = int(itemsize)
    for s in shape:
        n *= int(s)
    return n


def rows_within(budget_bytes: int, bytes_per_row: int, n: int, floor: int = 1, multiple: int = 1, lever: str = "budget") -> int:
    """Largest ``r`` with ``floor <= r <= n``, ``r % multiple == 0`` (unless ``r == n``), and ``r * bytes_per_row <= budget_bytes``.
    ``n`` itself is returned when everything fits. Nothing fitting at ``floor`` raises :class:`MemLeverRefused` naming ``lever``."""
    n, floor, multiple = int(n), max(1, int(floor)), max(1, int(multiple))
    if n < 1:
        raise ValueError("rows_within: n must be >= 1")
    bpr = max(1, int(bytes_per_row))
    r = int(budget_bytes) // bpr
    if r >= n:
        return n
    r -= r % multiple
    if r < floor:
        raise MemLeverRefused(lever, f"budget {int(budget_bytes) / 2 ** 30:.2f} GiB fits {int(budget_bytes) // bpr} rows of "
                                     f"{bpr / 2 ** 20:.1f} MiB; floor is {floor} (multiple {multiple})")
    return r


def _device_index(torch, device) -> int:
    """The CUDA ordinal of ``device`` (None = the current device; an int, a ``torch.device`` or a ``"cuda:<i>"`` string otherwise)."""
    if device is None:
        return int(torch.cuda.current_device())
    idx = getattr(torch.device(device) if isinstance(device, str) else device, "index", device)
    return int(torch.cuda.current_device()) if idx is None else int(idx)


def device_free_bytes(device=None) -> Optional[int]:
    """The driver's free bytes on ``device`` (read through :func:`opt_core.arch.device_memory`, the package's one device-memory reader) plus
    the caching allocator's reusable slack (reserved - allocated); None when torch has no CUDA device (the caller then uses its table value)."""
    try:
        import torch  # noqa: WPS433 — lazy by contract
    except Exception:  # noqa: BLE001
        return None
    if not torch.cuda.is_available():
        return None
    from .. import arch

    free = arch.device_memory(_device_index(torch, device))["free_bytes"]
    if free is None:
        return None
    return int(free) + int(torch.cuda.memory_reserved(device)) - int(torch.cuda.memory_allocated(device))


def budget_bytes(frac: float, reserve_bytes: int = 0, device=None) -> Optional[int]:
    """The byte budget of an 'auto' sized block: ``device_free_bytes(device) * frac - reserve_bytes`` (None without CUDA)."""
    if not 0.0 < float(frac) <= 1.0:
        raise ValueError(f"budget frac must be in (0, 1] (got {frac!r})")
    free = device_free_bytes(device)
    if free is None:
        return None
    return max(0, int(free * float(frac)) - int(reserve_bytes))
