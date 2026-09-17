"""torch.compile plumbing of a compile lever (strategy ``F3.torch_compile``): the dynamo boundary marker, the one installer, the census.

Contract. A kit that compiles upstream sub-modules does three generic things and one specific thing. Generic (this module): mark the
functions dynamo must not trace into — memo look-ups, host-side decisions, hoist accessors — as graph-break boundaries WHEN the lever is on and
leave them untouched when it is off (:func:`nodynamo`); wrap a list of ``(owner, attribute)`` targets in ``torch.compile`` exactly once, with a
per-code recompile budget large enough that dynamo never falls back to eager silently (:class:`CompileRecord.install`, ``cache_size_limit``);
and report what happened in dynamo's own numbers (:meth:`CompileRecord.census`: graphs compiled, frames ok/total, graph breaks, recompile-limit
hits) on one ``LEVER`` line with a fail-closed gate (a target list that wrapped fewer modules than declared, a recompile-limit hit, more graph
breaks than the kit allows). Specific (the kit): WHICH targets — upstream's own list, never restated — and WHEN to install (e.g. at the first
roll-out inside the kit's hoist + graph replay). No numerics policy lives here: the compiled kernels' class is the kit's mode-table fact
(inductor is tier 2 unless a kit's equality leg proves otherwise), ``mode`` / ``dynamic`` (torch's tri-state None | True | False, printed ``dynamic=auto|1|0``) / ``fullgraph`` / ``backend`` pass through unchanged.
Torch is imported lazily inside the calls; nothing at module import (Python 3.8 floor).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import report

__all__ = ["nodynamo", "install", "CompileRecord", "dynamo_counters", "cache_entries"]

STRATEGY = "F3.torch_compile"
IMPL = "torch_compile"


def _torch():
    t = sys.modules.get("torch")
    if t is None:
        import torch as t  # noqa: PLC0415
    return t


def nodynamo(fn: Callable, *, active: bool) -> Callable:
    """``fn`` marked as a dynamo boundary when ``active`` (the compile lever is on): it runs eagerly and a compiled caller breaks its graph around the
    call (``torch.compiler.disable``); ``fn`` itself, untouched, when not active — an exact line whose compile lever is off calls the very function."""
    if not active:
        return fn
    torch = _torch()
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is None:                                   # torch < 2.1 spelling
        import torch._dynamo as _dynamo  # noqa: PLC0415
        disable = _dynamo.disable
    return disable(fn)


def dynamo_counters() -> Dict[str, int]:
    """Dynamo's own process-wide tallies, flattened: ``graphs`` (unique graphs compiled), ``calls_captured``, ``frames_ok``, ``frames_total``,
    ``frames_skipped`` (= total - ok: frames dynamo attempted and did NOT convert — they run eager), ``graph_breaks`` (all reasons summed),
    ``graph_break_reasons`` (distinct), ``recompile_limit_hits`` (every counter entry, in any category, whose category or key names the recompile /
    cache-size limit — the spelling moved between torch releases: ``unimplemented['Dynamo recompile limit exceeded …']``, ``graph_break[…cache_size_limit…]``,
    ``recompile_limit`` / ``cache_size_limit`` categories; 0 is the only healthy value). Zeros before anything compiled or without torch._dynamo."""
    zero = {"graphs": 0, "calls_captured": 0, "frames_ok": 0, "frames_total": 0, "frames_skipped": 0, "graph_breaks": 0, "graph_break_reasons": 0,
            "recompile_limit_hits": 0}
    try:
        from torch._dynamo.utils import counters  # noqa: PLC0415
    except ImportError:
        return zero
    stats, frames, gb = counters.get("stats", {}), counters.get("frames", {}), counters.get("graph_break", {})
    limit_hits = 0
    for cat, tally in list(counters.items()):
        cat_l = str(cat).lower()
        for key, v in list(tally.items()):
            text = cat_l + " " + str(key).lower()
            if ("recompile" in text and "limit" in text) or "cache_size_limit" in text or "recompile_limit" in text:
                limit_hits += int(v)
    ok, total = int(frames.get("ok", 0)), int(frames.get("total", 0))
    return {"graphs": int(stats.get("unique_graphs", 0)), "calls_captured": int(stats.get("calls_captured", 0)), "frames_ok": ok, "frames_total": total,
            "frames_skipped": max(0, total - ok), "graph_breaks": int(sum(int(v) for v in gb.values())), "graph_break_reasons": len(gb),
            "recompile_limit_hits": int(limit_hits)}


def _code_of(obj):
    """The code object dynamo caches compiled entries on for a compile target: a function's own, a bound method's function's, a module's ``forward``."""
    fn = getattr(obj, "_torchdynamo_orig_callable", obj)
    fwd = getattr(fn, "forward", None)
    if fwd is not None and not hasattr(fn, "__code__"):
        fn = fwd
    fn = getattr(fn, "__func__", fn)
    return getattr(fn, "__code__", None)


def cache_entries(code) -> Optional[int]:
    """How many compiled cache entries dynamo holds on ``code`` (``None`` when the running torch has no such introspection)."""
    if code is None:
        return None
    try:
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return len(_debug_get_cache_entry_list(code))
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return None


def _inductor_cache_word() -> str:
    d = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not d:
        return "unset"
    try:
        warm = os.path.isdir(d) and any(True for _ in os.scandir(d))
    except OSError:
        warm = False
    return ("warm:" if warm else "cold:") + d


def dynamic_word(dynamic: Optional[bool]) -> str:
    """The census word of torch.compile's ``dynamic`` tri-state: ``auto`` (None: automatic dynamic shapes after the first recompile), ``1`` (True:
    symbolic shapes from the start), ``0`` (False: static shapes — one compile per distinct shape, under the recompile budget)."""
    return "auto" if dynamic is None else ("1" if dynamic else "0")


class CompileRecord:
    """One compile lever's installs and census. ``install(targets)`` wraps ``getattr(owner, attr)`` of every ``(owner, attr)`` pair with
    ``torch.compile(sub, dynamic=, mode=, fullgraph=, backend=)`` and sets it back — once: a target already wrapped by this record, or already a
    dynamo ``OptimizedModule`` / compiled function, counts ``already``; a missing attribute counts ``missing`` (named in ``missing_names``) — and raises
    dynamo's per-code recompile budget to at least ``cache_size_limit`` first (one code object serving several static shapes plus a resume function per
    boundary needs more than dynamo's default 8; past the budget dynamo falls back to eager SILENTLY, so the floor is recorded and a limit hit fails the
    gate). :meth:`timed` accumulates the seconds of the calls the kit brackets as compiling (the first call per shape). :meth:`census` / :meth:`fields` /
    :meth:`line` / :meth:`gate` report it."""

    def __init__(self, name: str, *, dynamic: Optional[bool] = False, mode: Optional[str] = None, fullgraph: bool = False, backend: str = "inductor",
                 cache_size_limit: int = 64, log: Optional[Callable[[str], None]] = None):
        self.name = str(name)
        self.dynamic = None if dynamic is None else bool(dynamic)   # torch.compile's tri-state: None = automatic dynamic after the first recompile, True = symbolic shapes, False = static
        self.mode, self.fullgraph, self.backend = mode, bool(fullgraph), str(backend)
        self.cache_size_limit = int(cache_size_limit)
        self._log = log
        self._lock = threading.RLock()
        self.stats_ = {"installs": 0, "wrapped": 0, "already": 0, "missing": 0, "declared": 0, "compile_s": 0.0, "timed_calls": 0}
        self.wrapped_names: List[str] = []
        self.missing_names: List[str] = []
        self.cache_size_limit_applied: Optional[int] = None
        self._codes: List[Tuple[str, Any]] = []            # (label, code object) per wrapped target: where dynamo's per-code cache entries are counted
        self._base = None                                  # dynamo counters at first install (the census is this record's delta)

    def _say(self, line: str) -> None:
        if self._log is not None:
            self._log(f"[compile:{self.name}] {line}")

    @staticmethod
    def _is_compiled(obj) -> bool:
        mod = type(obj).__module__ or ""
        if type(obj).__name__ == "OptimizedModule" or mod.startswith("torch._dynamo"):
            return True
        return bool(getattr(obj, "_torchdynamo_orig_callable", None)) or bool(getattr(obj, "_opt_core_compiled", False))

    def install(self, targets: Sequence[Tuple[Any, str]]) -> int:
        """Wrap each ``(owner, attr)`` target once; returns the number wrapped by THIS call. The kit decides when to call it."""
        torch = _torch()
        import torch._dynamo as _dynamo  # noqa: PLC0415
        with self._lock:
            if self._base is None:
                self._base = dynamo_counters()
            applied = None
            for knob in ("recompile_limit", "cache_size_limit"):           # one budget, two spellings across torch releases (aliases where both exist)
                cur = getattr(_dynamo.config, knob, None)
                if isinstance(cur, int):
                    setattr(_dynamo.config, knob, max(cur, self.cache_size_limit))
                    applied = int(getattr(_dynamo.config, knob)) if applied is None else min(applied, int(getattr(_dynamo.config, knob)))
            for knob in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):   # the per-process total must not undercut the per-code floor
                cur = getattr(_dynamo.config, knob, None)
                if isinstance(cur, int):
                    setattr(_dynamo.config, knob, max(cur, 4 * self.cache_size_limit))
            self.cache_size_limit_applied = applied if applied is not None else self.cache_size_limit
            n = 0
            self.stats_["installs"] += 1
            for owner, attr in targets:
                self.stats_["declared"] += 1
                label = f"{type(owner).__name__}.{attr}"
                sub = getattr(owner, attr, None)
                if sub is None:
                    self.stats_["missing"] += 1
                    self.missing_names.append(label)
                    continue
                if self._is_compiled(sub) or label in self.wrapped_names:
                    self.stats_["already"] += 1
                    continue
                kw = dict(dynamic=self.dynamic, fullgraph=self.fullgraph, backend=self.backend)
                if self.mode is not None:
                    kw["mode"] = self.mode
                compiled = torch.compile(sub, **kw)
                try:
                    setattr(compiled, "_opt_core_compiled", True)
                except (AttributeError, TypeError):         # a wrapper type that refuses attributes is still recognised by its module name
                    pass
                self._codes.append((label, _code_of(sub)))
                setattr(owner, attr, compiled)
                self.wrapped_names.append(label)
                self.stats_["wrapped"] += 1
                n += 1
            self._say(f"torch.compile(dynamic={self.dynamic}, mode={self.mode}, backend={self.backend}) on {n} target(s) {self.wrapped_names[-n:] if n else []}; "
                      f"cache_size_limit={self.cache_size_limit_applied}")
            return n

    @contextmanager
    def timed(self):
        """Bracket the call(s) that compile (the first call per static shape): their wall seconds accumulate in ``compile_s``."""
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            with self._lock:
                self.stats_["compile_s"] += time.perf_counter() - t0
                self.stats_["timed_calls"] += 1

    def census(self) -> Dict[str, Any]:
        """This record's installs plus dynamo's tallies since its first install (process-wide counters, differenced against the install-time base)."""
        now = dynamo_counters()
        base = self._base or {k: 0 for k in now}
        delta = {k: int(now[k]) - int(base.get(k, 0)) for k in now}
        with self._lock:
            d = dict(self.stats_)
            limit = self.cache_size_limit_applied if self.cache_size_limit_applied is not None else self.cache_size_limit
            entries = {label: cache_entries(code) for label, code in self._codes}
            known = [n for n in entries.values() if n is not None]
            d.update(name=self.name, dynamic=dynamic_word(self.dynamic), mode=self.mode or "default", backend=self.backend, fullgraph=int(self.fullgraph),
                     cache_size_limit=limit, wrapped_names=list(self.wrapped_names), missing_names=list(self.missing_names),
                     inductor_cache=_inductor_cache_word(), cache_entries=entries, cache_entries_max=(max(known) if known else None),
                     at_limit=[label for label, n in entries.items() if n is not None and n >= int(limit)], **delta)
            d["compile_s"] = round(float(d["compile_s"]), 3)
            return d

    def state(self) -> str:
        return "on" if self.stats_["wrapped"] else "skipped"

    def fields(self) -> List[Tuple[str, Any]]:
        c = self.census()
        return [("compiled", c["wrapped"]), ("declared", c["declared"]), ("already", c["already"]), ("missing", c["missing"]), ("dynamic", c["dynamic"]),
                ("mode", c["mode"]), ("backend", c["backend"]), ("cache_size_limit", c["cache_size_limit"]), ("compile_s", c["compile_s"]),
                ("graphs", c["graphs"]), ("frames", f"{c['frames_ok']}/{c['frames_total']}"), ("frames_skipped", c["frames_skipped"]),
                ("graph_breaks", c["graph_breaks"]), ("recompile_limit_hits", c["recompile_limit_hits"]),
                ("cache_entries_max", "na" if c["cache_entries_max"] is None else f"{c['cache_entries_max']}/{c['cache_size_limit']}"),
                ("inductor_cache", c["inductor_cache"].split(":")[0])] + ([("at_limit", ",".join(c["at_limit"]))] if c["at_limit"] else [])

    def words(self) -> List[str]:
        """The named uncertainty this record carries (:mod:`opt_core.report` ``words``): ``cache=at_limit(<targets>)`` when a wrapped target's
        per-code cache holds ``cache_size_limit`` entries — every compiled shape still runs compiled; a FURTHER distinct shape would run eager,
        and if it does dynamo's ``recompile_limit_hits`` counts it and :meth:`gate` refuses on that count. Evidence, never a gate sentence."""
        c = self.census()
        return [report.word("cache", "at_limit", *c["at_limit"])] if c["at_limit"] else []

    def line(self, tag: str, name: Optional[str] = None, strategy: Optional[str] = STRATEGY) -> str:
        """``[<tag>] LEVER name=<name> state=on|skipped [reason=nothing_wrapped] impl=torch_compile origin=core strategy=F3.torch_compile compiled=<n>
        declared=<n> already=<n> missing=<n> dynamic=<auto|1|0> mode=… backend=… cache_size_limit=… compile_s=… graphs=… frames=<ok>/<total> graph_breaks=…
        recompile_limit_hits=… inductor_cache=<unset|cold|warm> [at_limit=<targets>]`` (:func:`opt_core.report.lever_line`; ``at_limit`` only when a
        per-code cache is at the budget)."""
        st = self.state()
        reason = None if st == "on" else "nothing_wrapped"
        return report.lever_line(tag, name or self.name, st, *self.fields(), reason=reason, impl=IMPL, origin="core", strategy=strategy)

    def gate(self, *, expect_wrapped: Optional[int] = None, max_graph_breaks: Optional[int] = None, max_recompile_limit_hits: int = 0,
             max_frames_skipped: Optional[int] = 0, require_frames: bool = False) -> List[str]:
        """The fail-closed sentences (empty = pass): fewer targets wrapped than ``expect_wrapped`` (default: every declared target), any missing
        target, and the two independent signs that compiled code ran EAGER without a word — recompile-limit hits above ``max_recompile_limit_hits``
        (dynamo's counters, every spelling) and frames dynamo attempted but did not convert above ``max_frames_skipped`` (``None`` = not gated) —
        then graph breaks above ``max_graph_breaks`` when given, and ``require_frames`` with no frame compiled (installed but never ran compiled).
        A per-code cache AT the budget is not a sentence: nothing ran eager yet; it is the word ``cache=at_limit(<targets>)`` (:meth:`words`) and the
        line's ``at_limit=`` field, and the recompile-limit count above gates the moment a shape does run eager."""
        c, out = self.census(), []
        want = c["declared"] if expect_wrapped is None else int(expect_wrapped)
        if c["wrapped"] + c["already"] < want:
            out.append(f"{self.name}: wrapped {c['wrapped']} (+{c['already']} already) of {want} declared targets (missing: {','.join(c['missing_names']) or 'none'})")
        elif c["missing"]:
            out.append(f"{self.name}: missing targets {','.join(c['missing_names'])}")
        if c["recompile_limit_hits"] > int(max_recompile_limit_hits):
            out.append(f"{self.name}: recompile_limit_hits={c['recompile_limit_hits']} (dynamo fell back to eager past cache_size_limit={c['cache_size_limit']})")
        if max_frames_skipped is not None and c["frames_skipped"] > int(max_frames_skipped):
            out.append(f"{self.name}: frames_skipped={c['frames_skipped']} (frames dynamo attempted and did not convert run eager) above the allowed {int(max_frames_skipped)}")
        if max_graph_breaks is not None and c["graph_breaks"] > int(max_graph_breaks):
            out.append(f"{self.name}: graph_breaks={c['graph_breaks']} above the allowed {int(max_graph_breaks)}")
        if require_frames and c["wrapped"] and not c["frames_ok"]:
            out.append(f"{self.name}: installed on {c['wrapped']} target(s) but dynamo compiled no frame (never ran compiled)")
        return out

    def partial(self) -> Optional[str]:
        problems = self.gate()
        return "; ".join(problems) if problems else None


def install(targets: Sequence[Tuple[Any, str]], name: str = STRATEGY, **kw) -> CompileRecord:
    """One-call form: a :class:`CompileRecord` with ``install(targets)`` already applied (``kw``: dynamic, mode, fullgraph, backend, cache_size_limit, log)."""
    rec = CompileRecord(name, **kw)
    rec.install(targets)
    return rec
