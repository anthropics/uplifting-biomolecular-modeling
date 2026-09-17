"""keep_pool — the tree's own lever on the CUDA caching-allocator policy (exact class: placement only).

Stock OpenDDE 1.1.1 returns the caching allocator's pool to the driver (`torch.cuda.empty_cache()`: synchronous cudaFree / cuMemUnmap of every
cached, unused segment) INSIDE the model forward of every item on one GPU, and the next phase re-grows the same bytes from the driver:
  * `opendde/model/opendde.py:580`   `_prepare_structural_features` — between the trunk and the structural stage (direct call);
  * `opendde/model/opendde.py:1598`  `run_post_confidence_outputs_stage` (direct call);
  * `opendde/model/modules/confidence.py:298`  ConfidenceHead, after `z_init` (through `opendde.utils.torch_utils.cleanup_device_memory`);
  * `opendde/model/modules/confidence.py:553`  before the PAE / PDE logits (through `cleanup_device_memory`);
  * `opendde/model/modules/confidence.py:813`  items with N_token > 2000 (through `cleanup_device_memory`).
(`SKIPPED_SITES`; the Fold-CP sites of `opendde.py` / `confidence.py` / `distributed/foldcp/*` run only on a Fold-CP mesh.) The runner's
releases — `runner/inference.py:1653` (per seed, before the items), `:775` via `_cleanup_batch_synchronized` (after every item), `:1870` (per
seed, with a synchronize), `:188` (error path) — are OUTSIDE the forward and are LEFT AS STOCK (passed through, counted): between items the
process's memory profile is the stock's. No tensor value depends on any of it.

The lever: `torch.cuda.empty_cache` (and `torch.cuda.memory.empty_cache`) become this module's `empty_cache`, which identifies the deciding caller —
the calling frame, walking past `opendde/utils/torch_utils.py` (the stock's `cleanup_device_memory` / `_clear_accelerator_cache` relay: its
`gc.collect()` / `synchronize` still run as the caller asked) and past registered transparent wrappers — and: a frame whose file lies in the
stock MODEL package (`opendde/model/...`) -> counted no-op (the blocks the previous phase freed stay cached and serve the next); any other caller
(the stock `runner`, the kit's own units — the offload unit, `opt_core.mem` seams, the big adapter — torch itself) -> the callable that was
installed before us, unchanged. Peak `memory_allocated` is unchanged by construction (the live tensors are the same); peak `memory_reserved`
per item is to be read from the run (the confidence head is served from the sampler's cached blocks). STATS: installed, skipped / passed per
`file:line` site, errors (a bookkeeping failure passes the call through — never a swallowed release by accident).
"""
from __future__ import annotations

import os
import sys

RELAY_FILES = ("opendde/utils/torch_utils.py",)              # frames walked past: the stock's cleanup relay (cleanup_device_memory -> _clear_accelerator_cache)
MODEL_MARK = "/opendde/model/"                                # a deciding frame under the stock model package = an in-forward site -> skipped
SKIPPED_SITES = ("opendde/model/opendde.py:580", "opendde/model/opendde.py:1598", "opendde/model/modules/confidence.py:298",
                 "opendde/model/modules/confidence.py:553", "opendde/model/modules/confidence.py:813")   # the pin's single-GPU in-forward sites (tests grep the stock tree)
PASSED_SITES = ("runner/inference.py:188", "runner/inference.py:775", "runner/inference.py:1653", "runner/inference.py:1870")
WRAPPER_CODES = set()                                         # code objects of other transparent empty_cache wrappers -> register_wrapper()
STATS = {"installed": False, "skipped": {}, "passed": {}, "skipped_total": 0, "passed_total": 0, "errors": 0}
_ST = {"orig": None}


class ActivationError(RuntimeError):
    """torch (or torch.cuda.empty_cache) is absent: the kit's activation fails by name."""


def register_wrapper(fn) -> None:
    """Declare `fn` (another transparent torch.cuda.empty_cache wrapper) so caller identification looks through it."""
    code = getattr(fn, "__code__", None)
    if code is not None:
        WRAPPER_CODES.add(code)


def _norm(path: str) -> str:
    return (path or "").replace("\\", "/")


def _site(path: str, lineno: int) -> str:
    p = _norm(path)
    i = max(p.rfind("/opendde/"), p.rfind("/runner/"))          # the innermost package mark (a stock tree under a directory named opendde/ still reads runner/...)
    if i >= 0:
        return f"{p[i + 1:]}:{lineno}"
    return f"{os.path.basename(p)}:{lineno}"


def classify(frame) -> tuple:
    """(skip, site) for the deciding caller of an empty_cache call: walk past the relay files and registered wrappers, then a frame under the
    stock model package is an in-forward site (skip=True); anything else passes."""
    f = frame
    while f is not None and (f.f_code in WRAPPER_CODES or _norm(f.f_code.co_filename).endswith(RELAY_FILES)):
        f = f.f_back
    if f is None:
        return False, "?"
    fn = _norm(f.f_code.co_filename)
    return (MODEL_MARK in fn), _site(fn, f.f_lineno)


def empty_cache() -> None:
    """torch.cuda.empty_cache under lever keep_pool: stock in-forward sites -> counted no-op; every other caller -> the previous callable."""
    try:
        skip, site = classify(sys._getframe(1))
    except Exception:                                        # never let bookkeeping swallow a release: pass through
        STATS["errors"] += 1
        skip, site = False, "?"
    if skip:
        STATS["skipped"][site] = STATS["skipped"].get(site, 0) + 1
        STATS["skipped_total"] += 1
        return None
    STATS["passed"][site] = STATS["passed"].get(site, 0) + 1
    STATS["passed_total"] += 1
    return _ST["orig"]()


empty_cache._keep_pool = True


def install() -> None:
    """Replace torch.cuda.empty_cache (and torch.cuda.memory.empty_cache) with the policy callable. Idempotent."""
    if STATS["installed"]:
        return
    try:
        import torch
        cur = torch.cuda.empty_cache
    except Exception as e:  # noqa: BLE001
        raise ActivationError(f"keep_pool: torch.cuda.empty_cache is not importable ({e!r})") from None
    if getattr(cur, "_keep_pool", False):
        STATS["installed"] = True
        return
    _ST["orig"] = cur
    torch.cuda.empty_cache = empty_cache
    try:
        import torch.cuda.memory as _m
        if not getattr(_m.empty_cache, "_keep_pool", False):
            _m.empty_cache = empty_cache
    except Exception:  # noqa: BLE001
        pass
    STATS["installed"] = True


def uninstall() -> None:
    if not STATS["installed"]:
        return
    try:
        import torch
        if torch.cuda.empty_cache is empty_cache and _ST["orig"] is not None:
            torch.cuda.empty_cache = _ST["orig"]
        import torch.cuda.memory as _m
        if _m.empty_cache is empty_cache and _ST["orig"] is not None:
            _m.empty_cache = _ST["orig"]
    except Exception:  # noqa: BLE001
        pass
    STATS["installed"] = False


def _reset() -> None:
    """Test hook: the wrapper removed, counters cleared."""
    uninstall()
    STATS["skipped"].clear(); STATS["passed"].clear(); STATS["skipped_total"] = STATS["passed_total"] = STATS["errors"] = 0


def kit_stats() -> dict:
    return {"installed": STATS["installed"], "skipped": sum(STATS["skipped"].values()), "passed": sum(STATS["passed"].values()),
            "errors": STATS["errors"], "skipped_sites": dict(STATS["skipped"]), "passed_sites": dict(STATS["passed"])}


def fallbacks(planned) -> list:
    """The lever's named events at exit: torch imported and CUDA used but the wrapper not in place (another unit re-bound empty_cache without
    relaying), or bookkeeping errors."""
    if "keep_pool" not in (planned or ()):
        return []
    out = []
    t = sys.modules.get("torch")
    if t is not None and STATS["installed"] and not getattr(getattr(t.cuda, "empty_cache", None), "_keep_pool", False):
        out.append("keep_pool: torch.cuda.empty_cache was re-bound after the lever without relaying to it")
    if STATS["errors"]:
        out.append(f"keep_pool: {STATS['errors']} caller classifications failed (those calls were passed through)")
    return out
