"""The CUDA caching-allocator policy of a process: ``PYTORCH_CUDA_ALLOC_CONF`` exported at activation under named refusals, and the
allocator's own record read back as evidence (torch imported lazily; :func:`conf_for`, :func:`env_row` and the refusal logic of
:func:`export` need no torch at all, so a kit that composes a CHILD process's environment uses them without importing torch).

Mechanism. ``expandable_segments:True`` lets torch's caching allocator grow and shrink segments instead of fragmenting fixed blocks:
reserved stays close to allocated at large token counts, so a line that dies of fragmentation near its ceiling gains headroom.
``garbage_collection_threshold:<f>`` additionally returns unused cached blocks once reserved memory passes fraction ``f`` of the card.
Both are PLACEMENT policies: no kernel input changes, so outputs are unchanged by construction — the equality record of the kit still
qualifies it for its engine. The setting is read by torch at its FIRST CUDA allocation, hence the two refusals: CUDA already initialised in
this process (the export would be a silent no-op) and an environment that already names another configuration (the caller's own
setting is never overridden in silence). A third, opt-in refusal is the graph-pool hazard: memory captured into CUDA-graph private pools
is not returned by these policies, so a kit whose line keeps graph capture on declares ``graphs_on=True`` and gets a refusal unless it
also passes ``allow_with_graphs=True`` (its mode table then says why the composition is intended).

API:
    POLICIES                         ``("expandable", "capped")``
    conf_for(policy, gc_threshold)   the ``PYTORCH_CUDA_ALLOC_CONF`` value of a policy
    env_row(policy, gc_threshold)    ``{"PYTORCH_CUDA_ALLOC_CONF": conf}`` for a child process's environment ({} for policy None)
    parse_conf(value) / format_conf  the ONE parser / formatter of the variable's ``key:value,…`` text
    torch_state()                    torch imported? version, CUDA initialised?, runtime API present? — without importing torch
    allocator_settings()             the ONE read-back: ``torch.cuda.memory._snapshot()["allocator_settings"]`` or None (nothing to read)
    write_conf(conf, environ, via)   the ONE writer: the variable (``env``) and, for an initialised allocator, the runtime API (``runtime_api``)
    export(policy, ...)              set the variable in ``environ`` (default ``os.environ``) or raise MemLeverRefused; returns the facts
                                     dict of the evidence line: ``alloc=<policy> alloc_conf=<value> alloc_export=<exported|present>``
    effective()                      the allocator's own record once CUDA is initialised: ``{"expandable": True|False|None, "source": ...}``
                                     (None = pending: CUDA not initialised yet / torch not imported; an unreadable record reads False —
                                     an unreadable record is never a pass). The record is ``torch.cuda.memory._snapshot()
                                     [\"allocator_settings\"]``, a private torch surface: a torch build without it reads
                                     ``unreadable:no-allocator_settings:torch<ver>`` — the kit checks ``alloc_source=snapshot`` on
                                     ITS stack's first run (seen: snapshot on torch 2.13.0+cu130) and treats unreadable as a refusal
    facts(policy, environ)           the evidence-line facts without exporting (for a report at exit): policy, conf in env, effective
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Mapping, MutableMapping, Optional

from . import MemLeverRefused

__all__ = ["POLICIES", "ENV", "conf_for", "env_row", "parse_conf", "format_conf", "torch_state", "allocator_settings", "write_conf", "export",
           "effective", "facts"]

ENV = "PYTORCH_CUDA_ALLOC_CONF"
POLICIES = ("expandable", "capped")
GC_THRESHOLD_DEFAULT = 0.5
_EXPANDABLE = "expandable_segments:True"


def conf_for(policy: str, gc_threshold: float = GC_THRESHOLD_DEFAULT) -> str:
    """``expandable`` -> ``expandable_segments:True``; ``capped`` -> ``expandable_segments:True,garbage_collection_threshold:<gc>``."""
    if policy == "expandable":
        return _EXPANDABLE
    if policy == "capped":
        g = float(gc_threshold)
        if not 0.0 < g < 1.0:
            raise ValueError(f"gc_threshold must be in (0, 1) (got {gc_threshold!r})")
        return f"{_EXPANDABLE},garbage_collection_threshold:{g:g}"
    raise MemLeverRefused("alloc", f"unknown allocator policy {policy!r} (known: {','.join(POLICIES)})")


def env_row(policy: Optional[str], gc_threshold: float = GC_THRESHOLD_DEFAULT) -> Dict[str, str]:
    """The environment row a kit hands a child process for ``policy`` (``{}`` when ``policy`` is None / ``"off"``)."""
    if policy in (None, "", "off"):
        return {}
    return {ENV: conf_for(policy, gc_threshold)}


def parse_conf(value: Optional[str]) -> Dict[str, str]:
    """``"a:1,b:2"`` -> ``{"a": "1", "b": "2"}`` (blank entries skipped; a malformed entry raises ValueError by name). The ONE parser of the variable."""
    out: Dict[str, str] = {}
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{ENV} entry {item!r} is not key:value")
        k, _, v = item.partition(":")
        out[k.strip()] = v.strip()
    return out


def format_conf(conf: Mapping[str, str]) -> str:
    """``{"a": "1", "b": "2"}`` -> ``"a:1,b:2"`` (insertion order)."""
    return ",".join(f"{k}:{v}" for k, v in conf.items())


def torch_state() -> Dict[str, object]:
    """``{"imported", "version", "cuda_initialized", "runtime_api"}`` of THIS process without importing torch (``sys.modules`` only);
    ``runtime_api`` = torch has ``torch.cuda.memory._set_allocator_settings`` (the setting can reach an initialised allocator)."""
    t = sys.modules.get("torch")
    if t is None:
        return {"imported": False, "version": None, "cuda_initialized": None, "runtime_api": None}
    cuda = getattr(t, "cuda", None)
    try:
        initialised = bool(cuda.is_initialized()) if cuda is not None else None
    except Exception:  # noqa: BLE001
        initialised = None
    mem = getattr(cuda, "memory", None)
    return {"imported": True, "version": getattr(t, "__version__", None), "cuda_initialized": initialised,
            "runtime_api": bool(mem is not None and hasattr(mem, "_set_allocator_settings"))}


def _cuda_initialized() -> Optional[bool]:
    """True/False from ``torch.cuda.is_initialized()`` when torch is already imported in this process; None when it is not (importing
    torch here would be wrong: the export must precede the process's own first import when the kit arranges it so)."""
    return torch_state()["cuda_initialized"]                                                    # type: ignore[return-value]


def allocator_settings() -> Optional[Dict[str, object]]:
    """The settings the CUDA caching allocator holds NOW: ``torch.cuda.memory._snapshot()["allocator_settings"]`` (a private torch surface)
    — the ONE read-back; None when torch is not imported, CUDA is unavailable or not initialised (nothing to read), or the build has no
    such record. Raises what ``_snapshot`` raises."""
    t = sys.modules.get("torch")
    if t is None or not t.cuda.is_available() or not t.cuda.is_initialized():
        return None
    snap = t.cuda.memory._snapshot()
    s = snap.get("allocator_settings") if isinstance(snap, Mapping) else None
    return dict(s) if isinstance(s, Mapping) else None


def write_conf(conf: str, environ: Optional[MutableMapping[str, str]] = None, *, via: str = "env") -> None:
    """The ONE writer of the allocator configuration: ``via="env"`` sets ``PYTORCH_CUDA_ALLOC_CONF=conf`` in ``environ`` (default
    ``os.environ``; torch reads it at its first CUDA allocation); ``via="runtime_api"`` also hands ``conf`` to
    ``torch.cuda.memory._set_allocator_settings`` (an allocator already initialised in this process; torch imported already by then).
    The refusal logic (when writing is wrong) belongs to the callers: :func:`export` and the memory mode's ``expandable_segments`` lever."""
    if via not in ("env", "runtime_api"):
        raise ValueError(f"write_conf via {via!r} is not env|runtime_api")
    if via == "runtime_api":
        t = sys.modules.get("torch")
        if t is None or not torch_state()["runtime_api"]:
            raise MemLeverRefused("alloc", "runtime_api write needs torch imported with torch.cuda.memory._set_allocator_settings")
        t.cuda.memory._set_allocator_settings(conf)
    if environ is not None or via == "env":
        env = os.environ if environ is None else environ
        env[ENV] = conf


def export(policy: str, environ: Optional[MutableMapping[str, str]] = None, *, lever: str = "alloc",
           gc_threshold: float = GC_THRESHOLD_DEFAULT, graphs_on: bool = False, allow_with_graphs: bool = False,
           cuda_initialized: Optional[bool] = None) -> Dict[str, object]:
    """Export the policy into ``environ`` (default: this process's ``os.environ``) and return the evidence facts
    ``{"alloc": policy, "alloc_conf": conf, "alloc_export": "exported" | "present"}`` (``present``: the variable already carried exactly
    this value). Refusals (:class:`MemLeverRefused` naming ``lever``): unknown policy; the variable set to a DIFFERENT value; CUDA already
    initialised in this process when ``environ`` is this process's (``cuda_initialized`` overrides the probe — pass False when exporting
    for a child process through a copy of os.environ); ``graphs_on`` without ``allow_with_graphs`` (graph private pools do not shrink
    under these policies — compose them only where the mode table says so)."""
    conf = conf_for(policy, gc_threshold)
    env = os.environ if environ is None else environ
    if graphs_on and not allow_with_graphs:
        raise MemLeverRefused(lever, f"{ENV}={conf} with CUDA-graph capture on: graph private pools are not returned by the allocator "
                                     f"policy (declare allow_with_graphs where the mode table composes them on purpose)")
    have = env.get(ENV)
    if have not in (None, "") and have != conf:
        raise MemLeverRefused(lever, f"{ENV}={have!r} in the environment names another allocator configuration (expected {conf!r} or unset)")
    this_process = environ is None or environ is os.environ
    inited = cuda_initialized if cuda_initialized is not None else (_cuda_initialized() if this_process else False)
    if inited:
        raise MemLeverRefused(lever, f"{ENV}={conf}: torch's CUDA allocator is already initialised in this process (the setting is read at "
                                     f"the first CUDA allocation) — export before any CUDA work")
    if have == conf:
        return {"alloc": policy, "alloc_conf": conf, "alloc_export": "present"}
    write_conf(conf, env)
    return {"alloc": policy, "alloc_conf": conf, "alloc_export": "exported"}


def effective() -> Dict[str, object]:
    """The allocator's own record of expandable segments in THIS process: ``{"expandable": True|False|None, "source": <str>}``.
    ``None`` (source ``pending``) before torch is imported or CUDA is initialised; ``True``/``False`` from
    ``torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]`` (source ``snapshot``); an unreadable record gives
    ``False`` with source ``unreadable:<why>`` — never a pass."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {"expandable": None, "source": "pending:torch-not-imported"}
    try:
        if not torch.cuda.is_available():
            return {"expandable": False, "source": "no-cuda"}
        if not torch.cuda.is_initialized():
            return {"expandable": None, "source": "pending:cuda-not-initialized"}
        settings = allocator_settings()
        if settings is None or "expandable_segments" not in settings:
            return {"expandable": False, "source": f"unreadable:no-allocator_settings:torch{getattr(torch, '__version__', '?')}"}
        return {"expandable": bool(settings.get("expandable_segments")) is True, "source": "snapshot"}
    except Exception as exc:  # noqa: BLE001
        return {"expandable": False, "source": f"unreadable:{type(exc).__name__}:torch{getattr(torch, '__version__', '?')}"}


def facts(policy: Optional[str], environ: Optional[MutableMapping[str, str]] = None) -> Dict[str, object]:
    """Evidence-line facts at report time (no export): ``alloc`` (policy or none), ``alloc_conf`` (the variable as carried),
    ``alloc_effective`` (true|false|pending per :func:`effective`)."""
    env = os.environ if environ is None else environ
    eff = effective()
    e = eff["expandable"]
    return {"alloc": policy or None, "alloc_conf": env.get(ENV) or None,
            "alloc_effective": ("pending" if e is None else str(bool(e)).lower()), "alloc_source": eff["source"]}
