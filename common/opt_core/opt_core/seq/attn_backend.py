"""Attention-backend pin: the one attention implementation a kit's test record holds for, declared as data and checked as a gate.

Contract. A kit's equality test record is a statement about ONE attention code path (a flash-attn wheel of one version, the
package's SDPA call, an eager chain, a carried kernel). A box that silently serves another path (a fallthrough in the
package's own backend cascade, a neighbouring wheel version, a card the wheel has no image for) runs numerics nobody tested.
:func:`declare` records the path as a :class:`BackendPin`; :meth:`BackendPin.probe` is the kit's gate (an
:class:`opt_core.gates.Gate` named ``attn_backend``): every required distribution present at its version, every
forbidden distribution absent, every required module importable, the card's ``sm`` in the tested set. Nothing is imported to
answer it — distribution metadata and import specs only — so the gate runs before torch. :meth:`BackendPin.line` is the fragment
the kit puts on its one ACTIVE line (:meth:`BackendPin.line_fields` the same as a dict): ``attn_backend=<name> <dist>=<version>... sm=<smNN|none>``, so a
run record proves which backend served. The pin adds no lever and changes no numerics.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional

from ..gates import Gate, dist_version
from ..report import kv

GATE_NAME = "attn_backend"


def module_present(name: str) -> bool:
    """``name`` (``pkg`` or ``pkg.sub.mod``) resolves to a file on the import path WITHOUT importing anything: the top-level
    package through :func:`importlib.util.find_spec`, each deeper part through :class:`importlib.machinery.PathFinder` over the
    parent's search locations (``find_spec`` of a dotted name would import the parent)."""
    parts = name.split(".")
    try:
        spec = importlib.util.find_spec(parts[0])
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    for part in parts[1:]:
        locations = spec.submodule_search_locations
        if not locations:
            return False
        spec = importlib.machinery.PathFinder.find_spec(part, list(locations))
        if spec is None:
            return False
    return True


@dataclass(frozen=True)
class BackendPin:
    """The attention path of one kit.

    ``backend``   the kit's name for the path (printed as ``attn_backend=<backend>``)
    ``versions``  distribution name -> version (``None`` = present at any version)
    ``modules``   importable module names the path needs (``pkg.sub`` allowed; resolved without importing)
    ``forbidden`` distribution names whose presence changes the path (a package cascade that prefers them when installed)
    ``arch``      the tested ``smNN`` strings; empty = any card
    """

    engine_tag: str
    backend: str
    versions: Mapping[str, Optional[str]] = field(default_factory=dict)
    modules: tuple = ()
    forbidden: tuple = ()
    arch: frozenset = frozenset()

    def found_versions(self) -> dict:
        """Distribution name -> installed version (``None`` when absent) for every named and forbidden distribution."""
        return {d: dist_version(d) for d in list(self.versions) + list(self.forbidden)}

    def probe(self, gpu: Optional[Mapping] = None, found: Optional[Mapping[str, Optional[str]]] = None) -> Gate:
        """The gate. ``gpu`` is the kit's own probe dict (``{"sm"|"cc", ...}`` from :mod:`opt_core.gates`); ``found`` overrides the
        metadata lookup (distribution -> version or ``None``). Every finding is in ``details``; the reason names every problem found (total accounting), joined in check order."""
        found = dict(found) if found is not None else self.found_versions()
        sm = _sm_of(gpu)
        details = {"backend": self.backend, "versions": dict(self.versions), "found": dict(found), "version_ok": {}, "modules": {},
                   "forbidden": {}, "arch": sorted(self.arch), "sm": sm}
        problems = []
        for dist, want in self.versions.items():
            have = found.get(dist)
            details["version_ok"][dist] = have is not None and (want is None or have == want)
            if have is None:
                problems.append(f"{dist} not installed (the {self.backend} path of record needs {dist}{'==' + want if want else ''})")
            elif want is not None and have != want:
                problems.append(f"{dist}=={have}, the certificate holds for {dist}=={want}")
        for dist in self.forbidden:
            have = found.get(dist)
            details["forbidden"][dist] = have
            if have is not None:
                problems.append(f"{dist}=={have} is installed and changes the attention path (the path of record runs without it)")
        for mod in self.modules:
            ok = module_present(mod)
            details["modules"][mod] = ok
            if not ok:
                problems.append(f"module {mod} not importable")
        if self.arch:
            if sm is None:
                problems.append(f"GPU sm unknown; the {self.backend} path is certified on {','.join(sorted(self.arch))}")
            elif sm not in self.arch:
                problems.append(f"GPU is {sm}; the {self.backend} path is certified on {','.join(sorted(self.arch))}")
        details["problems"] = list(problems)
        if problems:
            return Gate(name=GATE_NAME, ok=False, reason="attention backend — " + "; ".join(problems), details=details)
        return Gate(name=GATE_NAME, ok=True, details=details)

    def line_fields(self, gpu: Optional[Mapping] = None, found: Optional[Mapping[str, Optional[str]]] = None) -> Dict[str, str]:
        """``{"attn_backend": <backend>, <dist>: <version|"none">..., "sm": <smNN|"none">}`` — the activation-evidence fields for the kit's ACTIVE line."""
        found = dict(found) if found is not None else self.found_versions()
        out = {"attn_backend": self.backend}
        for d in self.versions:
            out[d] = _s(found.get(d))
        out["sm"] = _s(_sm_of(gpu))
        return out

    def line(self, gpu: Optional[Mapping] = None, found: Optional[Mapping[str, Optional[str]]] = None) -> str:
        """``attn_backend=<backend> <dist>=<version|none>... sm=<smNN|none>`` (:func:`opt_core.report.kv` over :meth:`line_fields`)."""
        return kv(*self.line_fields(gpu, found).items())


def declare(engine_tag: str, backend: str, versions: Optional[Mapping[str, Optional[str]]] = None,
            arch: Optional[Iterable[str]] = None, modules: Iterable[str] = (), forbidden: Iterable[str] = ()) -> BackendPin:
    """The kit's attention path as a :class:`BackendPin` (see the class for the fields)."""
    return BackendPin(engine_tag=str(engine_tag), backend=str(backend), versions=dict(versions or {}), modules=tuple(modules),
                      forbidden=tuple(forbidden), arch=frozenset(str(a) for a in (arch or ())))


def _s(v: Optional[str]) -> str:
    return "none" if v is None else str(v)


def _sm_of(gpu: Optional[Mapping]) -> Optional[str]:
    if not isinstance(gpu, Mapping):
        return None
    sm = gpu.get("sm")
    if sm in (None, "") and gpu.get("cc") not in (None, ""):
        sm = "sm" + str(gpu["cc"]).replace(".", "")
    if sm in (None, ""):
        return None
    sm = str(sm)
    return sm if sm.startswith("sm") else "sm" + sm
