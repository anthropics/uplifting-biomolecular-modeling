"""ptx1_keep_pool — lever `keep_pool` (EXACT class: caching-allocator policy only; no tensor value depends on it).

What stock does (pinned protenix 1.1.0, the installed package): the inference path calls ``torch.cuda.empty_cache()``
  * ``protenix/model/modules/confidence.py:204``  ConfidenceHead.forward, every item, INSIDE the forward (after ``del z_init``);
  * ``protenix/model/modules/confidence.py:233``  ConfidenceHead.forward, per sample, items with N_token > 2000 (pae/pde ``.cpu()`` offload path);
  * ``protenix/model/modules/confidence.py:347``  ConfidenceHead.memory_efficient_forward, per sample, N_token > 2000;
  * ``runner/inference.py:497`` (and :510 on the error path)  InferenceRunner's per-item loop, after every item (outside the item's forward).
Each call returns every cached, unused caching-allocator segment to the driver (synchronous cudaFree / cuMemUnmap) and the very next phase
re-allocates the same bytes from the driver. ``STOCK_SITES`` pins these statements for the census (tests/test_keep_pool.py checks them
against the installed package).

The lever: the stock call sites INSIDE the model forward (the three ``confidence.py`` sites: files of the stock ``protenix`` package) become
counted no-ops — the blocks the diffusion sampler freed stay cached and serve the confidence head (no synchronous release, no re-growth from the
driver inside the forward). The per-item site AFTER the item (``runner/inference.py``, outside the forward and outside the PHASE window) is LEFT
AS STOCK (passes through, counted), so the process's device profile across items is stock's. Every other caller passes through unchanged to the
callable that was installed before this one (the real ``torch.cuda.empty_cache`` or the sampler-graph lever's ``EmptyCacheGuard``, whichever is
current at install): the kit's own deliberate releases (the sampler-graph eviction release, the exact TriMul's buffer release, opt_core.mem's
cache_release seams under big, the guard's deferred flushes) keep their meaning. A stock call that arrives THROUGH the guard (guard installed
after this lever: ``EmptyCacheGuard.__call__`` -> here) is recognised by walking past that frame (``RELAY_FRAMES``).

Caller identification: the calling frame's source file, resolved once per code object: a file of the installed stock ``protenix`` package = a
model-forward site -> skipped; the stock ``runner`` package and anything else -> passed through.

Memory: ``max_memory_allocated`` is unchanged by construction (the live tensors are the same); the reserved pool inside an item is the sampler's
instead of a fresh confidence-head pool. One arm per process: ``install()`` is idempotent, ``uninstall()`` restores, ``report()`` is the census
the kit's LEVER line reads (levers_ptx1.describe()["keep_pool"]).
"""
from __future__ import annotations

import os
import sys

STOCK_TOPLEVEL = ("protenix", "runner")                     # the stock packages (site census); only SKIP_TOPLEVEL callers are swallowed
SKIP_TOPLEVEL = ("protenix",)                               # model-forward sites (confidence head); runner (between items) passes through
STOCK_SITES = (                                             # pinned protenix 1.1.0 inference-path sites (census; tests/test_keep_pool.py greps the installed package)
    "protenix/model/modules/confidence.py:204", "protenix/model/modules/confidence.py:233", "protenix/model/modules/confidence.py:347",
    "runner/inference.py:497", "runner/inference.py:510",
)
SKIPPED_SITES = tuple(x for x in STOCK_SITES if x.startswith("protenix/"))
RELAY_FRAMES = {("graphed.py", "__call__")}                 # infopt_graphs EmptyCacheGuard.__call__ relays stock calls to its `orig` (= this callable when it installed after us)
WRAPPER_CODES = set()                                       # code objects of other transparent empty_cache wrappers -> register_wrapper()
_STATE = {"installed": False, "orig": None, "skipped": {}, "passed": {}, "file_class": {}, "errors": 0}


def register_wrapper(fn) -> None:
    """Declare `fn` (another transparent torch.cuda.empty_cache wrapper) so caller identification looks through it."""
    code = getattr(fn, "__code__", None)
    if code is not None:
        WRAPPER_CODES.add(code)


def _classify(code) -> tuple:
    """(is_stock_model_site, site) for a code object — resolved once and memoised (a handful of distinct callers per process)."""
    key = id(code)
    hit = _STATE["file_class"].get(key)
    if hit is not None and hit[2] is code:
        return hit[0], hit[1]
    fn = (code.co_filename or "").replace("\\", "/")
    parts = fn.split("/")
    skip = False; site = os.path.basename(fn)
    for i in range(len(parts) - 1, -1, -1):                 # innermost top-level package dir named protenix|runner (site-packages/protenix/..., src/runner/...)
        if parts[i] in STOCK_TOPLEVEL and (i == 0 or parts[i - 1] in ("site-packages", "src", "dist-packages") or parts[i - 1].endswith((".egg", "stock"))):
            skip = parts[i] in SKIP_TOPLEVEL; site = "/".join(parts[i:]); break
    _STATE["file_class"][key] = (skip, site, code)
    return skip, site


def _caller_frame():
    f = sys._getframe(2)                                    # 0 = _caller_frame, 1 = empty_cache (ours), 2 = the caller
    while f is not None and (f.f_code in WRAPPER_CODES or (os.path.basename(f.f_code.co_filename), f.f_code.co_name) in RELAY_FRAMES):
        f = f.f_back                                        # relayed by EmptyCacheGuard.__call__ / a registered wrapper: judge THEIR caller
    return f


def empty_cache() -> None:
    """torch.cuda.empty_cache under lever keep_pool: stock model-forward sites -> counted no-op; every other caller -> the previous callable."""
    try:
        f = _caller_frame()
        is_stock, site = _classify(f.f_code) if f is not None else (False, "?")
        if f is not None:
            site = f"{site}:{f.f_lineno}"
    except Exception:                                       # never let bookkeeping break a release: pass through
        _STATE["errors"] += 1; is_stock, site = False, "?"
    if is_stock:
        _STATE["skipped"][site] = _STATE["skipped"].get(site, 0) + 1
        return None
    _STATE["passed"][site] = _STATE["passed"].get(site, 0) + 1
    return _STATE["orig"]()


empty_cache._ptx_keep_pool = True


def install() -> dict:
    """Replace torch.cuda.empty_cache (and torch.cuda.memory.empty_cache) with the policy callable. Idempotent. Returns report()."""
    import torch
    if _STATE["installed"]:
        return report()
    cur = torch.cuda.empty_cache
    if getattr(cur, "_ptx_keep_pool", False):
        _STATE["installed"] = True
        return report()
    _STATE["orig"] = cur
    _STATE["over"] = getattr(cur, "__qualname__", type(cur).__name__)
    torch.cuda.empty_cache = empty_cache
    try:
        import torch.cuda.memory as _m
        if _m.empty_cache is cur or not getattr(_m.empty_cache, "_ptx_keep_pool", False):
            _m.empty_cache = empty_cache
    except Exception:
        pass
    _STATE["installed"] = True
    return report()


def uninstall() -> None:
    import torch
    if not _STATE["installed"]:
        return
    if torch.cuda.empty_cache is empty_cache:
        torch.cuda.empty_cache = _STATE["orig"]
        try:
            import torch.cuda.memory as _m
            if _m.empty_cache is empty_cache:
                _m.empty_cache = _STATE["orig"]
        except Exception:
            pass
    _STATE["installed"] = False


def report() -> dict:
    """The lever's census: installed, what it wrapped, per-site skipped / passed counts and totals, bookkeeping errors."""
    return {"lever": "keep_pool", "installed": _STATE["installed"], "over": _STATE.get("over"),
            "skipped": dict(_STATE["skipped"]), "passed": dict(_STATE["passed"]),
            "skipped_total": sum(_STATE["skipped"].values()), "passed_total": sum(_STATE["passed"].values()), "errors": _STATE["errors"]}
