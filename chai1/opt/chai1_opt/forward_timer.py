"""The kit's one timing line, the same statement on every route. ``install(chai1_mod, prefix)`` wraps the module attribute
``chai_lab.chai1.run_folding_on_context`` — the call upstream's ``run_inference`` makes once per trunk sample and the driver kit makes once per
seed; both reach it through the module attribute, so one wrapper serves every caller of the process (the ``off`` route installs it in its clean
stock process, ``stock_fold.py``; the driver route in the driver's report window, ``driver.py``; the in-process route in ``chai1_opt.enable``,
``stack.py``). Timing only: ``torch.cuda.synchronize()`` before and after the call, ``time.perf_counter``; the arguments and the return value pass
through untouched and nothing inside the fold is wrapped. One line per fold on stderr:

    <prefix> FORWARD item=<item> out=<seed dir> tokens=<n|na> crop=<size|na> forward_s=<sec> (run_folding_on_context, cuda-synced)

``item`` = the name of the directory above the fold's ``output_dir`` (the input's key) and ``out`` = its basename (``seed_<s>``), ``na`` without an
``output_dir``; ``tokens`` = the feature context's ``structure_context.num_tokens`` and ``crop`` = the size upstream pads it to
(``chai_lab.data.collate.utils.pad_size`` over ``AVAILABLE_MODEL_SIZES``), ``na`` when the context does not carry them or exceeds the largest size
(upstream then raises inside the fold itself). ``<prefix>`` is ``report.PREFIX`` on the kit routes, ``report.STOCK_PREFIX`` on ``off``. Every fold's
record rides ``LOG`` (``{item, out, tokens, crop, forward_s}``); ``tally_fields`` restates their count and sum on the EXIT line
(``forward_calls=<n> forward_s_total=<sec>``).

The CUDA caching allocator's peak counters restart at fold entry (``reset_peak``: ``torch.cuda.reset_peak_memory_stats()``; nothing without a CUDA
allocator), so ``torch.cuda.max_memory_allocated()`` read after a fold is that fold's peak, not the process's: the routes' per-seed-fold rows report
it as ``max_mem_alloc_gb`` (``stock_fold.py`` through ``read_peak``; the driver kit's ``chai_worker.py`` through ``torch.cuda``).
"""
import os
import sys
import time
from typing import List, Optional

PHRASE = "(run_folding_on_context, cuda-synced)"
LOG: List[dict] = []
LEVER = "forward_timer"
GIB = float(2 ** 30)


def line(prefix: str, item: str, out: str, tokens, crop, forward_s: float) -> str:
    return f"{prefix} FORWARD item={item} out={out} tokens={tokens} crop={crop} forward_s={forward_s:.3f} {PHRASE}"


def _allocator():
    """``torch.cuda`` when a CUDA allocator is present (``torch.cuda.is_available()``), else None — the one gate of the peak counters."""
    import torch
    return torch.cuda if torch.cuda.is_available() else None


def reset_peak() -> bool:
    """Restart the allocator's peak counters (``torch.cuda.reset_peak_memory_stats()``) — at fold entry; False without a CUDA allocator."""
    cuda = _allocator()
    if cuda is None:
        return False
    cuda.reset_peak_memory_stats()
    return True


def read_peak() -> Optional[dict]:
    """``{"alloc_gib", "reserved_gib"}`` = ``max_memory_allocated`` / ``max_memory_reserved`` in GiB since the last ``reset_peak``; None without a
    CUDA allocator."""
    cuda = _allocator()
    if cuda is None:
        return None
    return {"alloc_gib": cuda.max_memory_allocated() / GIB, "reserved_gib": cuda.max_memory_reserved() / GIB}


def _tokens_and_crop(feature_context):
    tokens = crop = "na"
    try:
        tokens = int(feature_context.structure_context.num_tokens)
    except Exception:  # noqa: BLE001  — a context without the attribute is timed all the same
        return tokens, crop
    try:
        from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES, pad_size
        crop = int(pad_size(tokens, AVAILABLE_MODEL_SIZES))
    except Exception:  # noqa: BLE001  — above upstream's largest size upstream raises in the fold itself; the line says crop=na
        crop = "na"
    return tokens, crop


def install(chai1_mod, prefix: str, stream=None) -> bool:
    """Wrap ``chai1_mod.run_folding_on_context`` once (idempotent; returns True when this call installed it). ``run_inference`` and the kit
    driver both reach the function through the module attribute, so one wrapper serves every caller in the process."""
    cur = chai1_mod.run_folding_on_context
    if getattr(cur, "chai1_opt_lever", "") == LEVER:
        return False
    orig = cur

    def run_folding_on_context(feature_context, *args, **kw):
        import torch
        sync = torch.cuda.is_available()
        out_dir = kw.get("output_dir")
        reset_peak()                                                                             # the allocator's peak counters restart at fold entry (the rows' max_mem_alloc_gb reads them after the fold)
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = orig(feature_context, *args, **kw)
        if sync:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tokens, crop = _tokens_and_crop(feature_context)
        out = os.fspath(out_dir) if out_dir is not None else "na"
        rec = {"item": os.path.basename(os.path.dirname(out.rstrip("/"))) or "na" if out != "na" else "na", "out": os.path.basename(out.rstrip("/")) if out != "na" else "na",
               "tokens": tokens, "crop": crop, "forward_s": round(dt, 3)}
        LOG.append(rec)
        s = stream if stream is not None else sys.stderr
        s.write(line(prefix, rec["item"], rec["out"], tokens, crop, dt) + "\n")
        s.flush()
        return result
    run_folding_on_context.chai1_opt_lever = LEVER
    run_folding_on_context.__wrapped__ = orig
    chai1_mod.run_folding_on_context = run_folding_on_context
    return True


def uninstall(chai1_mod) -> None:
    cur = chai1_mod.run_folding_on_context
    if getattr(cur, "chai1_opt_lever", "") == LEVER:
        chai1_mod.run_folding_on_context = cur.__wrapped__


def last(n: int = 1) -> List[dict]:
    return LOG[-n:] if n > 0 else []


def total_s() -> float:
    return round(sum(r["forward_s"] for r in LOG), 3)


def tally_fields() -> List[str]:
    """``forward_calls=<n> forward_s_total=<sec>`` for the EXIT tally (empty when nothing was timed)."""
    return [f"forward_calls={len(LOG)} forward_s_total={total_s()}"] if LOG else []
