"""opt_core.tools.graph_audit.audit — capture-safety instruments for CUDA-graph work (observe only; no numerics are touched).

Lifted from the vendored ``infopt_graphs.audit`` module two co-folding kits carried byte-for-byte but for one docstring line
(sibling-map lineage L017); the instruments below are that module unchanged except that torch is imported on first use
(the core imports no framework at module top level) and the docstrings name no engine.

* SyncCensus      : context manager counting every host-synchronizing CUDA call inside a region, attributed to the
                    innermost NON-torch Python frame (file:line:func) and to the innermost frame overall.  Built on
                    torch.cuda.set_sync_debug_mode("warn"|"error").  mode="error" raises at the first sync (dry-run gate).
* LaunchCounter   : torch.profiler census of CUDA runtime launches (cudaLaunchKernel*, cudaGraphLaunch, memcpy/memset,
                    stream syncs) and of device kernel events for a region.
* rng_fingerprint : md5 of every RNG a co-folding model consumes (python random, numpy global, torch CPU, torch CUDA).
* RNGGuard        : asserts that a region consumes NO random numbers from any of those streams.
* audit_capture_safety(fn, ...) : dry-run fn once on a side stream under SyncCensus(mode="error") + RNGGuard and report
                    (syncs, rng use, allocation delta, peak memory).  Run it before capturing anything.
"""
from __future__ import annotations

import collections
import contextlib
import hashlib
import os
import random
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

def _torch():
    """torch, imported on first use: the core imports no framework at module top level (an importer pays the import when it measures)."""
    import torch
    return torch


def _torch_dir() -> str:
    """The directory of the torch package in use (frames under it are attributed to torch, not to the caller)."""
    return os.path.dirname(getattr(_torch(), "__file__", None) or os.sep)


def _frames_for_site() -> Tuple[str, str]:
    """(innermost non-torch frame 'file:line:func', innermost frame 'file:line:func') of the current stack."""
    st = traceback.extract_stack()
    here = os.path.abspath(__file__)
    inner = None
    outer = None
    for fr in reversed(st):
        fn = fr.filename
        if fn == here or fn.endswith("warnings.py"):
            continue
        if inner is None:
            inner = f"{fn}:{fr.lineno}:{fr.name}"
        if not fn.startswith(_torch_dir()) and "site-packages/torch/" not in fn:
            outer = f"{fn}:{fr.lineno}:{fr.name}"
            break
    return outer or "?", inner or "?"


@dataclass
class SyncCensus:
    mode: str = "warn"
    keep_messages: int = 20
    total: int = 0
    by_site: collections.Counter = field(default_factory=collections.Counter)
    by_inner: collections.Counter = field(default_factory=collections.Counter)
    messages: List[str] = field(default_factory=list)
    _prev_mode: Any = None
    _prev_show: Any = None
    _cm: Any = None

    def __enter__(self):
        torch = _torch()
        assert self.mode in ("warn", "error")
        self._prev_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.synchronize()  # drain pending work: count only what happens inside the region
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("always")
        self._prev_show = warnings.showwarning
        census = self

        def _show(message, category, filename, lineno, file=None, line=None):
            msg = str(message)
            if "synchroniz" in msg.lower():
                census.total += 1
                site, inner = _frames_for_site()
                census.by_site[site] += 1
                census.by_inner[inner] += 1
                if len(census.messages) < census.keep_messages:
                    census.messages.append(f"{msg.strip()[:160]} @ {site}")
            else:
                census._prev_show(message, category, filename, lineno, file, line)

        warnings.showwarning = _show
        torch.cuda.set_sync_debug_mode(self.mode)
        return self

    def __exit__(self, exc_type, exc, tb):
        torch = _torch()
        torch.cuda.set_sync_debug_mode(self._prev_mode if self._prev_mode is not None else 0)
        warnings.showwarning = self._prev_show
        self._cm.__exit__(exc_type, exc, tb)
        return False

    def report(self, top: int = 25) -> Dict[str, Any]:
        return {"total_syncs": self.total, "by_site": self.by_site.most_common(top),
                "by_inner_frame": self.by_inner.most_common(top), "examples": self.messages[:10]}


_LAUNCH_NAMES = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchKernelEx", "cuLaunchKernel", "cuLaunchKernelEx",
                 "cudaLaunchCooperativeKernel")
_GRAPH_LAUNCH = ("cudaGraphLaunch",)
_MEM_NAMES = ("cudaMemcpyAsync", "cudaMemsetAsync", "cudaMemcpy", "cudaMemset", "cudaMemcpy2DAsync")
_SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")


class LaunchCounter:
    """with LaunchCounter() as lc: ...; lc.summary().  Profiling adds host overhead: never time a profiled call."""

    def __init__(self, record_shapes: bool = False, top: int = 30):
        from torch.profiler import ProfilerActivity, profile
        self._prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=record_shapes)
        self.top = top
        self.events = None

    def __enter__(self):
        self._prof.__enter__()
        return self

    def __exit__(self, *a):
        self._prof.__exit__(*a)
        self.events = self._prof.events()
        return False

    def summary(self) -> Dict[str, Any]:
        c = collections.Counter()
        kern_time = collections.Counter()
        kern_count = collections.Counter()
        dev_kernels = 0
        dev_time_us = 0.0
        for e in self.events:
            n = e.name
            dt = getattr(e, "device_type", None)
            is_dev = dt is not None and str(dt).endswith("CUDA")
            if is_dev:
                dev_kernels += 1
                t = float(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0)) or 0.0)
                dev_time_us += t
                kern_time[n] += t
                kern_count[n] += 1
                continue
            if n in _LAUNCH_NAMES:
                c["kernel_launches"] += 1
            elif n in _GRAPH_LAUNCH:
                c["graph_launches"] += 1
            elif n in _MEM_NAMES:
                c["memcpy_memset"] += 1
            elif n in _SYNC_NAMES:
                c["stream_syncs"] += 1
        top = [{"kernel": k[:110], "count": kern_count[k], "self_ms": round(v / 1e3, 3)} for k, v in kern_time.most_common(self.top)]
        return {"kernel_launches": c["kernel_launches"], "graph_launches": c["graph_launches"], "memcpy_memset": c["memcpy_memset"],
                "stream_syncs": c["stream_syncs"], "device_kernels": dev_kernels,
                "device_kernel_time_ms": round(dev_time_us / 1e3, 3), "top_kernels": top,
                "atomics_and_fused_attention": kernel_histogram(self.events)}


_ATOMIC_PATTERNS = ("scatter", "index_add", "index_put", "atomic", "masked_scatter", "embedding_backward", "fmha", "flash", "sdpa",
                    "cudnn", "segment_reduce", "bincount", "histc", "nonzero", "unique", "sort", "cumsum")


def kernel_histogram(events, patterns=_ATOMIC_PATTERNS) -> Dict[str, Any]:
    """From torch.profiler events: {pattern: {kernel_name: count}} for device kernels whose name matches one of the
    patterns (case-insensitive).  Lists the nondeterminism-relevant kernels (atomics / fused attention) in a region."""
    out: Dict[str, Dict[str, int]] = {p: {} for p in patterns}
    total = 0
    for e in events:
        dt = getattr(e, "device_type", None)
        if not (dt is not None and str(dt).endswith("CUDA")):
            continue
        total += 1
        n = e.name
        nl = n.lower()
        for p in patterns:
            if p in nl:
                d = out[p]
                d[n[:120]] = d.get(n[:120], 0) + 1
    return {"device_kernels": total, "matches": {p: v for p, v in out.items() if v}}


def rng_fingerprint() -> Dict[str, str]:
    import numpy as np
    torch = _torch()
    out = {"py": hashlib.md5(repr(random.getstate()).encode()).hexdigest()[:12],
           "np": hashlib.md5(np.random.get_state()[1].tobytes()).hexdigest()[:12],
           "torch_cpu": hashlib.md5(torch.random.get_rng_state().numpy().tobytes()).hexdigest()[:12]}
    if torch.cuda.is_available():
        out["torch_cuda"] = hashlib.md5(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()[:12]
    return out


class RNGGuard:
    def __init__(self, raise_on_use: bool = True):
        self.raise_on_use = raise_on_use
        self.before = None
        self.after = None
        self.consumed: List[str] = []

    def __enter__(self):
        self.before = rng_fingerprint()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.after = rng_fingerprint()
        self.consumed = [k for k in self.before if self.before[k] != self.after[k]]
        if self.consumed and self.raise_on_use and exc_type is None:
            raise RuntimeError(f"RNGGuard: region consumed random numbers from {self.consumed}")
        return False


def audit_capture_safety(fn: Callable, *args, stream: Optional[torch.cuda.Stream] = None, **kwargs) -> Dict[str, Any]:
    """Dry-run fn(*args, **kwargs) once on a side stream with sync detector 'error' + RNG guard.
    fn must already be warmed up (lazy inits such as cuDNN plan caches / Triton compiles are synchronizing)."""
    torch = _torch()
    s = stream or torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    a0 = torch.cuda.memory_allocated()
    rep: Dict[str, Any] = {"ok": False, "syncs": None, "rng_consumed": None, "error": None}
    try:
        with torch.cuda.stream(s):
            with RNGGuard(raise_on_use=False) as rg, SyncCensus(mode="error") as sc:
                out = fn(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        rep["ok"] = True
        rep["syncs"] = sc.report()
        rep["rng_consumed"] = rg.consumed
        del out
    except Exception as e:  # noqa
        rep["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        rep["trace"] = traceback.format_exc()[-1500:]
    rep["alloc_delta_bytes"] = torch.cuda.memory_allocated() - a0
    rep["peak_alloc_bytes"] = torch.cuda.max_memory_allocated()
    return rep


@contextlib.contextmanager
def sync_census(mode: str = "warn"):
    with SyncCensus(mode=mode) as c:
        yield c
