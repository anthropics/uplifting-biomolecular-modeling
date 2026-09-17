"""protenix_ptx_keep_pool (module; lever `keep_pool`). EXACT class: allocator policy only.

What stock does (pinned Protenix 2.0.0, `stock/src`): the inference path calls ``torch.cuda.empty_cache()``
  * ``protenix/model/modules/confidence.py:204``  ConfidenceHead.forward, every item, INSIDE the timed forward (after ``del z_init``);
  * ``protenix/model/modules/confidence.py:233``  ConfidenceHead.forward, per sample, items with N_token > 2000 (pae/pde ``.cpu()`` offload path);
  * ``protenix/model/modules/confidence.py:347``  ConfidenceHead.memory_efficient_forward, per sample, N_token > 2000;
  * ``runner/inference.py:505`` (and :518 on the error path)  infer_predict, after every item (outside the item's window — but the pool it
    releases is re-grown by cudaMalloc / cuMemMap inside the NEXT item's window).
Each call returns every cached, unused caching-allocator segment to the driver (synchronous cudaFree / cuMemUnmap, tens of ms) and the very
next phase re-allocates the same bytes from the driver. No tensor value depends on it.

The lever: the stock call sites INSIDE the model forward (the three ``confidence.py`` sites: files of the stock ``protenix`` package) become
counted no-ops — the blocks the diffusion sampler freed stay cached and serve the confidence head (no synchronous release, no re-growth from the
driver inside the timed forward). The per-item site AFTER the item (``runner/inference.py``, outside the forward and outside the kit's PHASE
window) is deliberately LEFT AS STOCK (passes through, counted): today's graphed sampler re-captures on every item and the kit then returns the
dropped private pool with its own ``empty_cache`` inside the next item's sampler phase (the ``graphed.py`` eviction release), so keeping the
pool across items would only move the between-item free work into that in-forward release. Every other caller passes through unchanged to the callable that was installed before us (the real ``torch.cuda.empty_cache`` or the
kit's sampler-graph ``EmptyCacheGuard``, whichever is current at install): the kit's own deliberate releases (fpf_stackgraph capture pool
accounting, the sampler eviction release, fpf_trunkgraph capture recovery, ptx_lazy_init, opt_core.mem cache_release seams in big, the
guard's deferred flushes) keep their meaning. A stock call that arrives THROUGH the guard (guard installed after us: ``EmptyCacheGuard.__call__``
-> us) or through a registered instrumentation wrapper is recognised by walking past those frames. Hazard #44 (graphed.py): NOT releasing
inside the forward is the safe direction.

Caller identification: the calling frame's source file, resolved once per code object: a file of the installed / pinned stock ``protenix``
package = a model-forward site -> skipped; the stock ``runner`` package and anything else -> passed through. ``STOCK_SITES`` lists the five
pinned inference-path sites for the census / the tree test (tests/test_keep_pool.py greps the pinned stock tree and fails when a site moves
or a new one appears, like big's drop_bond_mask reader test). Import name: protenix_ptx_keep_pool (kit rule: new top-level modules are protenix_-prefixed); ``SKIPPED_SITES`` is the subset the lever swallows.

Memory semantics: ``max_memory_allocated`` (PEAK alloc_gib) is unchanged by construction (live tensors are the same); ``max_memory_reserved``
per item is unchanged within noise (the confidence head is served from the sampler's cached blocks instead of fresh segments — its working set is
smaller than the sampler's); between items the pool is released exactly as stock does, so the process's NVML profile across items is stock's.
Rides exact and fast; big inherits fast's set (its cache_release seams are kit calls and pass through).

Switch: ``PTX_KEEP_POOL=1`` (exported by env.sh for ARM=E and ARM=T; big pre-sets 0). ``install()`` is idempotent; ``report()`` returns the
counters; an atexit line ``[ptx_keep_pool] SUMMARY ...`` and a PTX_LEVER_REPORT jsonl record are the run's evidence.
"""
from __future__ import annotations

import atexit
import json
import os
import sys

ENV = "PTX_KEEP_POOL"
STOCK_TOPLEVEL = ("protenix", "runner")                     # the stock packages (site census); only SKIP_TOPLEVEL callers are swallowed
SKIP_TOPLEVEL = ("protenix",)                               # model-forward sites (confidence head); runner (between items) passes through
STOCK_SITES = (                                             # pinned Protenix 2.0.0 inference-path sites (census + tests/test_keep_pool.py)
    "protenix/model/modules/confidence.py:204", "protenix/model/modules/confidence.py:233", "protenix/model/modules/confidence.py:347",
    "runner/inference.py:505", "runner/inference.py:518",
)
SKIPPED_SITES = tuple(x for x in STOCK_SITES if x.startswith("protenix/"))
RELAY_FRAMES = {("graphed.py", "__call__")}                 # infopt_graphs EmptyCacheGuard.__call__ relays stock calls to its `orig` (= us when it installed after us)
WRAPPER_CODES = set()                                       # code objects of other transparent empty_cache wrappers (instrumentation) -> register_wrapper()
_STATE = {"installed": False, "orig": None, "skipped": {}, "passed": {}, "file_class": {}, "errors": 0}


def register_wrapper(fn) -> None:
    """Declare `fn` (another transparent torch.cuda.empty_cache wrapper, e.g. a probe) so caller identification looks through it."""
    code = getattr(fn, "__code__", None)
    if code is not None:
        WRAPPER_CODES.add(code)


def enabled(environ=None) -> bool:
    v = (environ if environ is not None else os.environ).get(ENV, "")
    if v in ("", "0"):
        return False
    if v != "1":
        raise ValueError(f"{ENV}={v!r}: 1 (keep the caching-allocator pool across the stock empty_cache sites) or unset/0")
    return True


def _classify(code) -> tuple:
    """(is_stock, site) for a code object — resolved once and memoised (a handful of distinct callers per process)."""
    key = id(code)
    hit = _STATE["file_class"].get(key)
    if hit is not None and hit[2] is code:
        return hit[0], hit[1]
    fn = (code.co_filename or "").replace("\\", "/")
    parts = fn.split("/")
    skip = False; site = os.path.basename(fn)
    for i in range(len(parts) - 1, -1, -1):                 # innermost top-level package dir named protenix|runner (site-packages/protenix/..., stock/src/runner/...)
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


MARK = "KEEP_POOL"                   # marker family: KEEP_POOL:on(...)


def install() -> str:
    """Replace torch.cuda.empty_cache (and torch.cuda.memory.empty_cache) with the policy callable. Idempotent. Returns the marker text `on(...)`."""
    import torch
    if _STATE["installed"]:
        return "on(already)"
    cur = torch.cuda.empty_cache
    if getattr(cur, "_ptx_keep_pool", False):
        _STATE["installed"] = True
        return "on(already)"
    _STATE["orig"] = cur
    torch.cuda.empty_cache = empty_cache
    try:
        import torch.cuda.memory as _m
        if _m.empty_cache is cur or not getattr(_m.empty_cache, "_ptx_keep_pool", False):
            _m.empty_cache = empty_cache
    except Exception:
        pass
    _STATE["installed"] = True
    if not _STATE.get("atexit"):
        atexit.register(_atexit); _STATE["atexit"] = True
    return f"on(skip={len(SKIPPED_SITES)}of{len(STOCK_SITES)},over={getattr(cur, '__qualname__', type(cur).__name__)})"


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
    return {"lever": "keep_pool", "installed": _STATE["installed"], "skipped": dict(_STATE["skipped"]), "passed": dict(_STATE["passed"]),
            "skipped_total": sum(_STATE["skipped"].values()), "passed_total": sum(_STATE["passed"].values()), "errors": _STATE["errors"]}


def _atexit() -> None:
    r = report()
    try:
        print(f"[ptx_keep_pool] SUMMARY skipped={r['skipped_total']} passed={r['passed_total']} errors={r['errors']} "
              f"skipped_sites={json.dumps(r['skipped'], sort_keys=True)} passed_sites={json.dumps(r['passed'], sort_keys=True)}", flush=True)
    except Exception:
        pass
    p = os.environ.get("PTX_LEVER_REPORT")
    if p:
        try:
            with open(p, "a") as fh:
                fh.write(json.dumps({"ptx_keep_pool": r, "pid": os.getpid()}) + "\n")
        except Exception:
            pass


def apply():                                                 # alias entry point: same as install()
    return install()
