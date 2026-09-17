"""Resident tables: positional / rotary tables built ONCE by the engine's own builder and served for the life of the process, plus the
pin-and-assert rule for lazily allocated device caches a captured CUDA graph reads by address.

Contract. Two objects, no numerics of their own.

:class:`ResidentTables` wraps the engine's STOCK builder (the function that computes a sin/cos or positional table): ``get(*args)``
builds once per key and returns the same object after — the bytes are the builder's, so every value served is the stock's by
construction; a kit's adapter slices what it needs (a slice is a view of the same bytes). ``freeze()`` closes the table set: a
``get`` that would BUILD after the freeze raises :class:`PinDrift` by name (a table built after a graph capture is a table the
graph does not read — the request is refused, never served). ``assert_pinned()`` re-reads the fingerprint (data pointer, shape,
dtype, device) of every table and names the first that moved. ``census()`` is the record for the kit's manifest; ``line_fields()`` /
``line()`` the fields / fragment for its ACTIVE line (``tables=<name>:<n>built/<n>hits frozen=<0|1>``).

:class:`LazyCachePins` watches caches the ENGINE owns and reallocates lazily (a rotary module's cos / sin tables rebuilt when a
longer sequence, another device or another dtype arrives; the kit names the attributes): ``pin(prebuild)`` runs the kit's prebuild (the module's
own cache builder at the maximum shape — no forward) and snapshots every watched attribute; ``assert_pinned()`` before a replay
raises :class:`PinDrift` naming module, attribute and the field that changed. A captured graph holds device ADDRESSES: a cache
reallocated after capture is freed memory under the graph, so the pins make a silent reallocation impossible to miss.

Fingerprints are duck-typed (``data_ptr()``, ``shape``, ``dtype``, ``device``): nothing here imports torch.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from ..report import kv

class PinDrift(RuntimeError):
    """A pinned table moved, or a table would be built after the freeze (the message names which)."""


def fingerprint(value: Any) -> Tuple:
    """Equality of a table as a graph sees it: ``(data_ptr, shape, dtype, device)`` for a tensor-like object; element-wise over a
    tuple / list / dict of them; ``("object", id)`` otherwise."""
    if hasattr(value, "data_ptr") and hasattr(value, "shape"):
        return ("tensor", int(value.data_ptr()), tuple(int(s) for s in value.shape), str(getattr(value, "dtype", None)),
                str(getattr(value, "device", None)))
    if isinstance(value, (tuple, list)):
        return ("seq",) + tuple(fingerprint(v) for v in value)
    if isinstance(value, dict):
        return ("map",) + tuple((k, fingerprint(v)) for k, v in sorted(value.items(), key=lambda kv_: repr(kv_[0])))
    return ("object", id(value))


def default_key(*args, **kwargs) -> Hashable:
    """The cache key of a builder call: tensor-like arguments by ``(shape, dtype, device)`` (never by content or address), everything
    else by value."""
    def one(a):
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            return ("tensor", tuple(int(s) for s in a.shape), str(a.dtype), str(getattr(a, "device", None)))
        return a
    return tuple(one(a) for a in args) + tuple(sorted((k, one(v)) for k, v in kwargs.items()))


class ResidentTables:
    """Build-once tables from the engine's own builder.

    ``builder(*args, **kwargs)`` is the stock function (its output is served unchanged); ``key_fn(*args, **kwargs)`` maps a call to
    its cache key (default :func:`default_key`); ``name`` labels the census and the line.
    """

    def __init__(self, builder: Callable[..., Any], key_fn: Optional[Callable[..., Hashable]] = None, name: str = "tables"):
        self.builder = builder
        self.key_fn = key_fn or default_key
        self.name = str(name)
        self._tables = {}        # key -> value
        self._pins = {}          # key -> fingerprint at build
        self.frozen = False
        self.counters = {"built": 0, "hits": 0, "refused": 0}

    def get(self, *args, **kwargs) -> Any:
        """The table for this call: built by the builder on first sight of its key, the same object after. After :meth:`freeze` a
        new key raises :class:`PinDrift`."""
        key = self.key_fn(*args, **kwargs)
        if key in self._tables:
            self.counters["hits"] += 1
            return self._tables[key]
        if self.frozen:
            self.counters["refused"] += 1
            raise PinDrift(f"{self.name}: a table for {key!r} would be built after the freeze "
                           f"(built: {sorted(repr(k) for k in self._tables)}); build it before the capture")
        value = self.builder(*args, **kwargs)
        self._tables[key] = value
        self._pins[key] = fingerprint(value)
        self.counters["built"] += 1
        return value

    def keys(self) -> List[Hashable]:
        return list(self._tables)

    def freeze(self) -> dict:
        """Close the table set (call after the last prebuild, before the first capture). Returns the census."""
        self.frozen = True
        return self.census()

    def assert_pinned(self) -> None:
        """Every built table has the fingerprint it was built with; the first difference raises :class:`PinDrift`."""
        for key, want in self._pins.items():
            now = fingerprint(self._tables[key])
            if now != want:
                raise PinDrift(f"{self.name}: table {key!r} moved: {want} -> {now} (a .to(), resize or rebuild after the pin)")

    def clear(self) -> None:
        """Forget every table (the kit's disengage); unfreezes."""
        self._tables.clear()
        self._pins.clear()
        self.frozen = False

    def census(self) -> dict:
        return {"name": self.name, "built": self.counters["built"], "hits": self.counters["hits"], "refused": self.counters["refused"],
                "frozen": self.frozen, "keys": [repr(k) for k in self._tables]}

    def line_fields(self) -> Dict[str, str]:
        """``{"tables": "<name>:<built>built/<hits>hits", "frozen": "0"|"1"}`` — the activation-evidence fields for the kit's ACTIVE line."""
        return {"tables": f"{self.name}:{self.counters['built']}built/{self.counters['hits']}hits", "frozen": str(int(self.frozen))}

    def line(self) -> str:
        """``tables=<name>:<built>built/<hits>hits frozen=<0|1>``."""
        return kv(*self.line_fields().items())


def find_modules(model: Any, attrs: Sequence[str]) -> List[Tuple[str, Any]]:
    """``(path, module)`` for every sub-module of ``model`` carrying any of ``attrs`` (``model.named_modules()`` order). ``attrs`` are the
    kit's names for its rotary / positional module's cache attributes (engine glue: the module has no default)."""
    attrs = tuple(attrs)
    if not attrs:
        raise ValueError("find_modules: attrs is empty — name the cache attributes to watch")
    return [(n, m) for n, m in model.named_modules() if any(hasattr(m, a) for a in attrs)]


class LazyCachePins:
    """Pins over engine-owned lazily built caches.

    ``modules`` is a non-empty sequence of ``(path, module)`` (see :func:`find_modules`); ``attrs`` the watched tensor attributes (the
    kit's names — no default); ``extra`` scalar attributes recorded beside them (e.g. the length the caches were built to). An empty
    ``modules`` or ``attrs`` raises :class:`ValueError` by name: a pin over nothing is not a pin.
    """

    def __init__(self, modules: Iterable[Tuple[str, Any]], attrs: Sequence[str], extra: Sequence[str] = (), name: str = "lazy_caches"):
        self.modules = list(modules)
        self.attrs = tuple(attrs)
        self.extra = tuple(extra)
        self.name = str(name)
        if not self.modules:
            raise ValueError(f"{self.name}: no modules to watch (find_modules found none carrying {self.attrs}) — the kit's attribute names do not match its model")
        if not self.attrs:
            raise ValueError(f"{self.name}: attrs is empty — name the cache attributes to watch")
        self.pinned = None       # the snapshot, or None before pin()
        self.checks = 0

    def snapshot(self) -> List[Dict[str, Any]]:
        """One record per (module, watched attribute holding a tensor): ``{module, attr, data_ptr, shape, dtype, device, <extra>...}``."""
        out = []
        for path, m in self.modules:
            extra = {e: getattr(m, e, None) for e in self.extra}
            for a in self.attrs:
                t = getattr(m, a, None)
                if t is None or not hasattr(t, "data_ptr"):
                    continue
                rec = {"module": path, "attr": a, "data_ptr": int(t.data_ptr()), "shape": tuple(int(s) for s in t.shape),
                       "dtype": str(getattr(t, "dtype", None)), "device": str(getattr(t, "device", None))}
                rec.update(extra)
                out.append(rec)
        return out

    def pin(self, prebuild: Optional[Callable[[], Any]] = None, min_len: Optional[int] = None, len_attr: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run ``prebuild`` (the kit's call of the modules' own cache builder at the maximum shape), snapshot, and hold the snapshot as
        the pins. With ``min_len``, ``len_attr`` (one of ``extra``) names the module attribute holding the built length; a module built
        shorter raises :class:`PinDrift` (the prebuild did not take)."""
        if prebuild is not None:
            prebuild()
        snap = self.snapshot()
        if not snap:
            raise PinDrift(f"{self.name}: nothing to pin — no watched attribute {self.attrs} holds a tensor on {len(self.modules)} modules")
        if min_len is not None:
            if not len_attr or len_attr not in self.extra:
                raise ValueError(f"{self.name}: min_len needs len_attr, one of extra={self.extra}")
            short = sorted({r["module"] for r in snap if r.get(len_attr) is None or int(r[len_attr]) < int(min_len)})
            if short:
                raise PinDrift(f"{self.name}: caches not built to {min_len} on {short}")
        self.pinned = snap
        return snap

    def assert_pinned(self) -> None:
        """The caches are the pinned ones (same pointer, shape, dtype, device, recorded extras); the first difference raises
        :class:`PinDrift`. Call before every replay."""
        if self.pinned is None:
            raise PinDrift(f"{self.name}: assert_pinned before pin")
        self.checks += 1
        now = self.snapshot()
        if len(now) != len(self.pinned):
            raise PinDrift(f"{self.name}: {len(now)} cache tensors now, {len(self.pinned)} pinned")
        for w, n in zip(self.pinned, now):
            for k in w:
                if n.get(k) != w[k]:
                    raise PinDrift(f"{self.name}: {w['module']}.{w['attr']} changed: {k} {w[k]!r} -> {n.get(k)!r} "
                                   f"(a longer forward, a .to() or a dtype change after the pin)")

    def census(self) -> dict:
        return {"name": self.name, "modules": len(self.modules), "pins": len(self.pinned or []), "checks": self.checks,
                "pinned": self.pinned is not None}

    def line_fields(self) -> Dict[str, str]:
        """``{"pins": "<name>:<n>", "checks": "<n>"}`` — the activation-evidence fields for the kit's ACTIVE line."""
        return {"pins": f"{self.name}:{len(self.pinned or [])}", "checks": str(self.checks)}

    def line(self) -> str:
        """``pins=<name>:<n> checks=<n>``."""
        return kv(*self.line_fields().items())
