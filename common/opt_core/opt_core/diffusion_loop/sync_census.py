"""The host-sync census of one sampling step: where a denoising loop makes the CPU wait for the GPU, engine-free.

Contract. :func:`sync_census` runs a callable under torch's CUDA sync debug mode ``"warn"`` and returns a :class:`SyncCensus` — the count
of synchronizing calls the callable made and one record per distinct call site (``file:line`` of the frame the warning was attributed to,
its count, and the warning text) — so an adapter can name the per-step ``.item()`` / ``.cpu()`` / ``.nonzero()`` / host-branch sites a
sync-free lever would replace, and a kit's test can hold a step to a sync count (:meth:`SyncCensus.at_most` returns an
:class:`opt_core.gates.Gate`). The previous debug mode is restored on exit. It changes no numerics and is not a lever: a development aid and
a regression guard. torch is imported inside :func:`sync_census` only; a test may pass a stand-in through ``torch_module``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

_SYNC_TEXT = "synchroniz"        # torch's warning text: "called a synchronizing CUDA operation"


@dataclass
class SyncSite:
    where: str          # "file:line"
    count: int
    text: str


@dataclass
class SyncCensus:
    syncs: int = 0
    sites: List[SyncSite] = field(default_factory=list)
    result: Any = None

    def evidence_fields(self) -> dict:
        return {"syncs": self.syncs, "sites": len(self.sites)}

    def at_most(self, n: int, name: str = "sync_census"):
        """Gate: ok when ``syncs <= n``; refused naming the count and the top sites otherwise."""
        from opt_core.gates import Gate
        details = {"syncs": self.syncs, "limit": int(n), "sites": [(s.where, s.count) for s in self.sites[:8]]}
        if self.syncs <= n:
            return Gate(name=name, ok=True, details=details)
        top = ", ".join(f"{s.where}×{s.count}" for s in self.sites[:3])
        return Gate(name=name, ok=False, details=details, reason=f"step made {self.syncs} host syncs > {n} allowed (top sites: {top})")


def sync_census(fn: Callable[[], Any], torch_module: Any = None) -> SyncCensus:
    """Run ``fn()`` once with ``torch.cuda.set_sync_debug_mode("warn")`` and count the sync warnings by call site. ``fn`` should be ONE
    step of the loop on its production inputs (the census is per call site, so one step names every site)."""
    torch = torch_module
    if torch is None:
        import torch  # noqa: F811 — lazy by contract
    previous = torch.cuda.get_sync_debug_mode()
    records: list = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            result = fn()
        records = list(caught)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    by_site: dict = {}
    other: list = []
    for w in records:
        text = str(w.message)
        if _SYNC_TEXT in text.lower():
            where = f"{w.filename}:{w.lineno}"
            site = by_site.get(where)
            if site is None:
                by_site[where] = SyncSite(where=where, count=1, text=text.splitlines()[0][:200])
            else:
                site.count += 1
        else:
            other.append(w)
    for w in other:                                   # warnings that are not ours are re-emitted, not swallowed
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    sites = sorted(by_site.values(), key=lambda s: (-s.count, s.where))
    return SyncCensus(syncs=sum(s.count for s in sites), sites=sites, result=result)
