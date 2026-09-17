"""Per-fold peak GPU memory — the ``PEAK`` line every pass prints beside its ``PHASE`` line, on the stock route and under every kit mode alike
(stock_fold.fold_items, the one prediction loop of every route), in the release's one grammar:

    PEAK item=<id> alloc_gib=<f> reserved_gib=<f>
    PEAK-NOTE item=<id> seed=<s> resident_gib=<f>

``alloc_gib`` / ``reserved_gib``: ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved`` over the fold window (the peak counters are reset
when the window opens), in GiB (bytes / 2^30), one PEAK line per (item, seed) pass — nothing else on that line. The PEAK-NOTE line carries the
pass's seed and ``resident_gib``: ``torch.cuda.memory_allocated`` after the fold's result is released and the caching allocator emptied — what the
process carries into the next fold (a value that grows fold over fold names an accumulation). Without torch / CUDA in the process (the CPU stub)
no PEAK line is printed, only ``PEAK-NOTE item=<id> cuda=absent`` — never a guessed value. Reading only: no argument, result or setting is touched.
"""
import sys
from typing import List, Optional, Tuple

GIB = float(2 ** 30)


API = ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved", "memory_allocated")


def _cuda():
    """torch.cuda when torch is loaded, CUDA is available and the memory-statistics API is present; else None (the values read NA)."""
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    try:
        ok = cuda is not None and bool(cuda.is_available()) and all(callable(getattr(cuda, name, None)) for name in API)
    except Exception:  # noqa: BLE001
        ok = False
    return cuda if ok else None


def begin() -> bool:
    """Open a fold window: reset the peak counters. Returns whether CUDA statistics are available."""
    cuda = _cuda()
    if cuda is None:
        return False
    cuda.reset_peak_memory_stats()
    return True


def peak() -> Tuple[Optional[float], Optional[float]]:
    """(max allocated, max reserved) in GiB since begin(), or (None, None) without CUDA."""
    cuda = _cuda()
    if cuda is None:
        return None, None
    return cuda.max_memory_allocated() / GIB, cuda.max_memory_reserved() / GIB


def resident() -> Optional[float]:
    """Allocated GiB now (after the loop released the fold), or None without CUDA."""
    cuda = _cuda()
    return None if cuda is None else cuda.memory_allocated() / GIB


def fmt(v: float) -> str:
    return f"{float(v):.2f}"


def lines(item_id: str, seed: int, alloc: Optional[float], reserved: Optional[float], resident_now: Optional[float]) -> List[str]:
    """The pass's PEAK line and its PEAK-NOTE line — or the one ``cuda=absent`` note when the statistics are not available."""
    if alloc is None or reserved is None:
        return [f"PEAK-NOTE item={item_id} cuda=absent"]
    note = f"PEAK-NOTE item={item_id} seed={'none' if seed is None else int(seed)}" + (f" resident_gib={fmt(resident_now)}" if resident_now is not None else " resident_gib=unread")
    return [f"PEAK item={item_id} alloc_gib={fmt(alloc)} reserved_gib={fmt(reserved)}", note]
