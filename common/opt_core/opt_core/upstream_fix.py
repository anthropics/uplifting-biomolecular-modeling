"""``--upstream-fix <ID>[,<ID>…]``: the release tree's one grammar for confirmed upstream issues that a kit fixes ON REQUEST ONLY.

The core names no issue and no kit. A kit registers its own entries — ``Registry.register(ID, apply, summary=…, refuse=…)`` for a
callable, ``Registry.register_file(ID, path, module_name=…)`` for an issue file loaded by path only when a run names its ID — and
drives the flag through the registry: :func:`parse` turns the flag's text into the ordered, de-duplicated ID list; ``resolve(ids, run)``
refuses an unknown ID by name (:class:`UnknownUpstreamFix`, listing the registered IDs) and a fix whose own run precondition does not
hold (:class:`UpstreamFixRefused`, the entry's ``refuse(run)`` words); ``apply(ids, log=…)`` installs the fixes in the calling process in
the order given and prints one line per fix through the kit's own logger, ``UPSTREAM-FIX <ID> applied file=<file> targets=<t,…>``
(:data:`APPLIED_LINE`; the kit's logger adds its own line prefix). A fix that cannot be installed propagates its exception: the caller
refuses the run by name with :data:`NOT_APPLIED`. With no IDs nothing is resolved, loaded, applied or printed — the default path is
upstream's behaviour as shipped. There is no ``all`` word: every fix is named.

An issue file (``register_file``) is a Python file defining ``ID`` (== the registered ID), ``apply(log=None) -> dict`` (the record:
``summary``, ``targets`` …) and optionally ``refuse(run: dict) -> str | None`` (the words of an unmet precondition); it is loaded by
path under the kit's module name (:func:`load_by_path`, memoised in ``sys.modules`` so a second apply finds the same module) and never
sits on ``sys.path``.

Standard library only: a stock caller that installs a requested fix may hold this module beside the stock proof machinery
(``opt_core.stock_proof.CORE_ALLOWED_WITH_UPSTREAM_FIX``) and nothing else of the core.
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Callable, Dict, List, Optional

FLAG = "--upstream-fix"
NOT_APPLIED = "UPSTREAM-FIX NOT APPLIED"                                   # the refusal words when a requested fix cannot be installed (the run never proceeds without it)
APPLIED_LINE = "UPSTREAM-FIX {id} applied file={file} targets={targets}"   # the one census line per applied fix (targets: comma-joined, '-' when none)

__all__ = ["FLAG", "NOT_APPLIED", "APPLIED_LINE", "UnknownUpstreamFix", "UpstreamFixRefused", "Registry", "parse", "load_by_path", "loaded",
           "applied_line", "ids_of"]


class UnknownUpstreamFix(ValueError):
    """``--upstream-fix`` names an ID the kit did not register (refused by name before anything launches)."""


class UpstreamFixRefused(ValueError):
    """A requested fix's own run precondition does not hold (the entry's ``refuse(run)`` words, e.g. ``--upstream-fix <ID> requires
    --use-templates true``): refused by name before anything launches — a set flag is applied or the run does not start."""


def parse(value: Optional[str]) -> List[str]:
    """The flag's value → the IDs in order, duplicates dropped: ``None``/empty → ``[]``; ``"A-001"``; ``"A-001,A-002"`` (spaces around items allowed)."""
    ids: List[str] = []
    for item in (value or "").split(","):
        item = item.strip()
        if item and item not in ids:
            ids.append(item)
    return ids


def applied_line(record: dict) -> str:
    """The census line of one applied fix from its record (``id``, ``file``, ``targets``)."""
    return APPLIED_LINE.format(id=record["id"], file=record.get("file") or "-", targets=",".join(record.get("targets") or []) or "-")


def ids_of(records: Optional[List[dict]]) -> List[str]:
    return [r["id"] for r in (records or [])]


def load_by_path(module_name: str, path: str, fid: str):
    """An issue file as a module: loaded by ``path`` under ``module_name`` (registered in ``sys.modules`` so a second load returns the same
    module; removed again when its import fails). The file must define ``ID == fid`` and a callable ``apply``; anything else is an
    :class:`ImportError` naming the file."""
    mod = sys.modules.get(module_name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    executed = False
    try:
        spec.loader.exec_module(mod)
        executed = True
    finally:
        if not executed:                                   # the file's own import failed: leave no half-module behind (the exception propagates)
            sys.modules.pop(module_name, None)
    if getattr(mod, "ID", None) != fid or not callable(getattr(mod, "apply", None)):
        sys.modules.pop(module_name, None)
        raise ImportError(f"{path}: an upstream issue file defines ID == {fid!r} (found {getattr(mod, 'ID', None)!r}) and apply()")
    return mod


def loaded(module_prefix: str) -> List[str]:
    """The IDs whose issue file is loaded in this process under ``<module_prefix>.<ID>`` (empty on every default run)."""
    return sorted(n[len(module_prefix) + 1:] for n in sys.modules if n.startswith(module_prefix + "."))


class _Entry:
    __slots__ = ("fid", "apply", "summary", "refuse", "file", "_loader", "_extra")

    def __init__(self, fid, apply, summary, refuse, file, loader=None, extra=None):
        self.fid, self.apply, self.summary, self.refuse, self.file, self._loader, self._extra = fid, apply, summary, refuse, file, loader, extra


class Registry:
    """A kit's registered upstream-issue fixes, in registration order. ``where``: the kit's words appended to the unknown-ID refusal
    (after ``known: …``), e.g. where its issue files live and which README section lists them."""

    def __init__(self, *, where: str = ""):
        self.where = where
        self._entries: Dict[str, _Entry] = {}

    # ---------------------------------------------------------------------------------------------------------------- registration
    def register(self, fid: str, apply: Callable, *, summary: str = "", refuse: Optional[Callable] = None, file: Optional[str] = None) -> None:
        """Register one fix: ``apply(log=…) -> dict`` (its record: ``summary``, ``targets`` …), ``refuse(run) -> words | None`` (its run
        precondition; None = none), ``file`` (the record's and the census line's ``file=`` word). A second entry under the same ID is refused."""
        if not fid or "," in fid or fid != fid.strip():
            raise ValueError(f"an upstream fix ID is a non-empty token without commas or surrounding spaces (got {fid!r})")
        if fid in self._entries:
            raise ValueError(f"two upstream issue entries carry the ID {fid!r}")
        if not callable(apply):
            raise ValueError(f"upstream fix {fid!r}: apply is not callable")
        self._entries[fid] = _Entry(fid, apply, summary, refuse, file)

    def register_file(self, fid: str, path: str, *, module_name: str, file: Optional[str] = None, summary: str = "",
                      extra: Optional[Callable] = None) -> None:
        """Register an issue FILE, loaded by path (:func:`load_by_path` under ``module_name``) only when a run names ``fid``: its ``apply``
        is the file's ``apply``, its precondition the file's optional ``refuse``; ``extra(module) -> dict`` adds kit words to the record."""
        def loader(_name=module_name, _path=path, _fid=fid):
            return load_by_path(_name, _path, _fid)

        def apply(log=None, _loader=loader):
            return _loader().apply(log=log)

        def refuse(run, _loader=loader):
            check = getattr(_loader(), "refuse", None)
            return check(dict(run)) if callable(check) else None

        self.register(fid, apply, summary=summary, refuse=refuse, file=file)
        self._entries[fid]._loader, self._entries[fid]._extra = loader, extra

    # ---------------------------------------------------------------------------------------------------------------- the flag
    def ids(self) -> List[str]:
        return list(self._entries)

    def file_of(self, fid: str) -> Optional[str]:
        """The registered ``file`` word of an entry (None when not registered or registered without one)."""
        e = self._entries.get(fid)
        return e.file if e is not None else None

    def known(self) -> str:
        """The registered IDs as the refusal lists them: sorted, comma-joined, ``none`` when empty."""
        return ", ".join(sorted(self._entries)) or "none"

    def resolve(self, ids: List[str], run: Optional[dict] = None) -> List[str]:
        """The requested IDs, each registered; :class:`UnknownUpstreamFix` naming the first unknown one and the known set. With ``run``
        (the kit's words for the run, e.g. ``{"verb", "use_templates"}``) each entry's ``refuse(run)`` precondition is checked:
        :class:`UpstreamFixRefused` ``--upstream-fix <ID> <words>``. Nothing is loaded or checked for no IDs."""
        ids = list(ids or [])
        if not ids:
            return []
        for fid in ids:
            if fid not in self._entries:
                raise UnknownUpstreamFix(f"unknown upstream fix {fid!r} ({FLAG}); known: {self.known()}{self.where}")
        if run is not None:
            for fid in ids:
                check = self._entries[fid].refuse
                words = check(dict(run)) if callable(check) else None
                if words:
                    raise UpstreamFixRefused(f"{FLAG} {fid} {words}")
        return ids

    def apply(self, ids: List[str], *, log: Callable[[str], None]) -> List[dict]:
        """Install the requested fixes in THIS process, in the order given: call each entry's ``apply(log=log)``, complete its record
        (``id``, ``file``, the entry's extra words) and print :data:`APPLIED_LINE` through ``log``. Returns the records (``[]``, nothing
        printed, for no IDs). An exception from a fix propagates: the caller refuses the run by name (:data:`NOT_APPLIED`)."""
        records: List[dict] = []
        for fid in self.resolve(ids):
            entry = self._entries[fid]
            rec = dict(entry.apply(log=log) or {})
            rec["id"] = fid
            if entry.file is not None:
                rec["file"] = entry.file
            if entry._extra is not None:
                rec.update(entry._extra(entry._loader()) or {})
            log(applied_line(rec))
            records.append(rec)
        return records
