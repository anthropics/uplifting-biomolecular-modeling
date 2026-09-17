"""Compile-cache witness: whether a process's compiled-kernel caches were warm, proven on one line.

Contract. A kit that ships or stages compiled-kernel caches (Triton, inductor, torch extensions, the driver's ComputeCache) proves the
warm start by CONTENT, never by asserting it. :func:`snapshot` records ``sha256 + mtime_ns + bytes`` of every file under a cache
directory (or of the named entries only — the kit's own files inside a shared directory). :func:`cache_witness` compares the snapshot
taken before the first kernel launch with the one taken after the warm-up forward: an entry whose bytes or mtime changed was rewritten,
an entry that vanished was evicted, a file that appeared is a kernel this process compiled (named by its Triton kernel group when the
file is a ``__grp__<kernel>.json``); with a :class:`CompileCounter` installed, the number of Triton compile calls that took at least
``real_compile_s`` (a cache load takes milliseconds, a real compile does not) joins the verdict. A file either snapshot could not read is counted as ``unreadable`` and is never
evidence of a warm cache. ``HIT`` = nothing changed, nothing missing, nothing new, nothing unreadable and no real compile; anything else
is ``MISS`` with the counts that say why. :func:`line_fields` is the
activation-evidence hook (``{key: str}`` in line order); :func:`witness_line` prints those fields in the tree's line grammar —
``[<tag>] CACHE name=<name> verdict=HIT|MISS|n/a entries=… changed=… missing=… new=… unreadable=… groups=… compiles=…`` — one line per cache
per process. The witness observes; installing, relocating or extracting a cache is the kit's (or the cold-start
module's) business. Standard library at top level; :class:`CompileCounter` imports ``triton`` inside :meth:`CompileCounter.install` and
records ``NOT bound (<why>)`` when it cannot (a witness without a counter still judges the files).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Iterable, Mapping, Optional

from .. import report

VERDICTS = ("HIT", "MISS", "n/a")
GROUP_PREFIX, GROUP_SUFFIX = "__grp__", ".json"        # Triton's put_group file: one per compiled kernel, named by the kernel


def snapshot(root: Optional[str], only: Optional[Iterable[str]] = None) -> dict:
    """``{relpath: {"sha256", "mtime_ns", "bytes", "unreadable"}}`` of every file under ``root`` (``only``: of those relative paths alone).
    A missing or empty directory is an empty snapshot. A file that vanished between the walk and the read (another process's cache write)
    is not an entry; a file that exists but cannot be read is an entry with ``unreadable`` = the error's class name and no digest — the
    witness counts it by name, it never passes as warm."""
    out: dict = {}
    if not root or not os.path.isdir(root):
        return out
    wanted = None if only is None else set(only)
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, root)
            if wanted is not None and rel not in wanted:
                continue
            try:
                st = os.stat(p)
                with open(p, "rb") as fh:
                    digest = hashlib.sha256(fh.read()).hexdigest()
            except FileNotFoundError:
                continue
            except OSError as e:
                out[rel] = {"sha256": None, "mtime_ns": None, "bytes": None, "unreadable": type(e).__name__}
                continue
            out[rel] = {"sha256": digest, "mtime_ns": st.st_mtime_ns, "bytes": st.st_size, "unreadable": None}
    return out


def kernel_groups(relpaths: Iterable[str]) -> list:
    """The Triton kernel names among ``relpaths`` (files ``__grp__<kernel>.json``), sorted, each once."""
    names = set()
    for r in relpaths:
        b = os.path.basename(r)
        if b.startswith(GROUP_PREFIX) and b.endswith(GROUP_SUFFIX):
            names.add(b[len(GROUP_PREFIX):-len(GROUP_SUFFIX)])
    return sorted(names)


def cache_witness(before: Mapping, after: Mapping, *, own: Optional[Iterable[str]] = None, real_compiles: Optional[int] = None,
                  listed: int = 20) -> dict:
    """The verdict of one cache directory (module contract). ``before`` / ``after``: :func:`snapshot` of the directory before the first
    launch and after the warm-up. ``own``: the kit's own entries (relative paths) — ``changed`` and ``missing`` are judged on those alone and
    a new file that is one of them does not count as compiled here; default: every entry of ``before``. ``real_compiles``: the
    :class:`CompileCounter`'s count over the same span (None: no counter — the files alone decide). An entry either snapshot could not
    read is ``unreadable`` (counted, listed) and the verdict is ``MISS``: a cache that cannot be read cannot be shown warm. Returns
    ``{verdict, n_entries, n_changed, n_missing, n_new, n_unreadable, changed, missing, new, unreadable, groups, real_compiles}`` with the
    lists cut at ``listed``."""
    own_set = set(before) if own is None else set(own)
    unreadable = sorted(r for r in set(before) | set(after) if (before.get(r) or {}).get("unreadable") or (after.get(r) or {}).get("unreadable"))
    bad = set(unreadable)
    changed = sorted(r for r in own_set if r in before and r in after and r not in bad and (after[r]["sha256"] != before[r]["sha256"] or after[r]["mtime_ns"] != before[r]["mtime_ns"]))
    missing = sorted(r for r in own_set if r in before and r not in after)
    new = sorted(r for r in after if r not in before and r not in own_set)
    real = None if real_compiles is None else int(real_compiles)
    hit = not changed and not missing and not new and not unreadable and (real is None or real == 0)
    return {"verdict": "HIT" if hit else "MISS", "n_entries": len(after), "n_changed": len(changed), "n_missing": len(missing), "n_new": len(new),
            "n_unreadable": len(unreadable), "changed": changed[:listed], "missing": missing[:listed], "new": new[:listed], "unreadable": unreadable[:listed],
            "groups": kernel_groups(new), "real_compiles": real}


def line_fields(name: str, witness: Optional[Mapping] = None, *, reason: Optional[str] = None, **fields) -> dict:
    """The activation-evidence fields of one cache witness, ``{key: str}`` in line order (the sub-package's hook convention): ``name,
    verdict, entries, changed, missing, new, unreadable, groups, compiles`` then the caller's ``fields`` (e.g. ``key=<jit_cache key> dir_rule=kept|keyed``).
    ``witness`` None gives ``name, verdict=n/a, reason`` — a route with no cache to judge says so instead of printing nothing."""
    if witness is None:
        out = {"name": str(name), "verdict": "n/a", "reason": str(reason or "no cache directory on this route")}
    else:
        w = witness
        out = {"name": str(name), "verdict": str(w["verdict"]), "entries": str(w["n_entries"]), "changed": str(w["n_changed"]), "missing": str(w["n_missing"]),
               "new": str(w["n_new"]), "unreadable": str(w["n_unreadable"]), "groups": report.join(w["groups"]),
               "compiles": "none" if w["real_compiles"] is None else str(w["real_compiles"])}
    for k, v in fields.items():
        out[str(k)] = report.kv((k, v)).split("=", 1)[1]
    return out


def witness_line(tag: str, name: str, witness: Optional[Mapping] = None, *, reason: Optional[str] = None, **fields) -> str:
    """``[<tag>] CACHE name=<name> verdict=… entries=… changed=… missing=… new=… unreadable=… groups=… compiles=… <fields>`` — :func:`line_fields`
    in the tree's line grammar, one line per cache per process."""
    return f"{report.prefix(tag)} CACHE {report.kv(*line_fields(name, witness, reason=reason, **fields).items())}"


class CompileCounter:
    """Counts Triton compile calls and their durations for the process: a call that takes at least ``real_compile_s`` is a real compile,
    a shorter one a cache load. :meth:`install` wraps the module-level ``triton.runtime.jit.compile`` (the name older Tritons call) and the
    bound ``compile`` of every ``JITFunction`` (or ``Autotuner.fn``) found in the already-imported modules whose names start with one of
    ``module_prefixes`` (newer Tritons bind compile per instance) — the kit names its own kernel modules; nothing else is touched.
    :meth:`report` → ``{real_compiles, calls, max_call_s, per_call, bound}``; :meth:`uninstall` restores every wrapped attribute."""

    def __init__(self, real_compile_s: float = 0.5, module_prefixes: Iterable[str] = ()):
        self.real_compile_s = float(real_compile_s)
        self.module_prefixes = tuple(module_prefixes)
        self.n = 0
        self.calls: list = []
        self.max_s = 0.0
        self.where: Optional[str] = None
        self._wrapped: list = []

    def _wrap(self, orig):
        me = self

        def compile(src, *args, **kwargs):  # noqa: A001 — the wrapped attribute's own name
            t = time.perf_counter()
            try:
                return orig(src, *args, **kwargs)
            finally:
                dt = time.perf_counter() - t
                name = getattr(getattr(src, "fn", None), "__name__", None) or getattr(src, "name", None) or type(src).__name__
                me.calls.append((str(name), round(dt, 4)))
                me.max_s = max(me.max_s, dt)
                if dt >= me.real_compile_s:
                    me.n += 1
        return compile

    def install(self) -> "CompileCounter":
        """Bind the counter (idempotent per instance). ``bound`` in :meth:`report` names what was wrapped, or ``NOT bound (<why>)``."""
        if self.where is not None:
            return self
        bound = []
        try:
            from triton.runtime import jit as jitmod                       # noqa: PLC0415 — triton only when a kit asks for the counter
            from triton.runtime.jit import JITFunction                     # noqa: PLC0415
            try:
                from triton.runtime.autotuner import Autotuner             # noqa: PLC0415
            except Exception:  # noqa: BLE001
                Autotuner = ()                                             # noqa: N806
            if callable(getattr(jitmod, "compile", None)):
                orig = jitmod.compile
                jitmod.compile = self._wrap(orig)
                self._wrapped.append((jitmod, "compile", orig))
                bound.append("triton.runtime.jit.compile (module global)")
            seen = set()
            n_inst = 0
            if self.module_prefixes:
                for mname, mod in list(sys.modules.items()):
                    if mod is None or not mname.startswith(self.module_prefixes):
                        continue
                    for attr in dir(mod):
                        obj = getattr(mod, attr, None)
                        if Autotuner and isinstance(obj, Autotuner):
                            obj = getattr(obj, "fn", None)
                        if isinstance(obj, JITFunction) and id(obj) not in seen and "compile" in getattr(obj, "__dict__", {}):
                            seen.add(id(obj))
                            orig = obj.compile
                            obj.compile = self._wrap(orig)
                            self._wrapped.append((obj, "compile", orig))
                            n_inst += 1
            if n_inst:
                bound.append(f"JITFunction.compile on {n_inst} kernel instance(s) under {','.join(self.module_prefixes)}")
        except Exception as e:  # noqa: BLE001 — no triton, or a triton without these names: recorded, the files still decide
            bound.append(f"NOT bound ({type(e).__name__}: {str(e)[:80]})")
        self.where = "; ".join(bound) or "NOT bound (no compile attribute found)"
        return self

    def uninstall(self) -> None:
        for obj, attr, orig in reversed(self._wrapped):
            try:
                setattr(obj, attr, orig)
            except Exception:  # noqa: BLE001
                pass
        self._wrapped = []

    def report(self) -> dict:
        return {"real_compiles": self.n, "calls": len(self.calls), "max_call_s": round(self.max_s, 4), "per_call": self.calls[-40:], "bound": self.where}
