"""``--upstream-fix <ID>[,<ID>…]``: confirmed upstream issues, fixed on request only (README 'Known upstream issues').

Each switchable issue is one file ``<tree>/upstream_issues/<ID>_<slug>.py`` beside opt/ — the fix as a runtime rebinding plus its docstring
(what upstream does · evidence · what the fix changes · expected effect on outputs). The directory is never on ``sys.path`` and nothing
imports a file from it unless a run names its ID: ``design`` / ``warm`` resolve the flag before anything launches (an unknown ID is refused
by name; so is a fix whose own run precondition does not hold — the issue file's optional ``refuse(run)``), and the arm process that runs
the loop (stock_design.py, every mode, ``off`` included) loads the file by path, calls its ``apply()`` and prints
``[ef2inv-opt] UPSTREAM-FIX <ID> applied …``; the run's run.json / opt_manifest.json record the IDs under ``upstream_fix``. With the flag
absent no arm, lever or timer differs in any way and no module from upstream_issues/ is loaded: without the flag every mode runs the
upstream library's behaviour as installed. A fix stock needs merely to run (the stock exception, STOCK.md) is not an upstream issue in this sense: it has no file here,
no ID on the flag, and every mode applies it identically (patches.py).

This module imports nothing beyond the standard library and ``.report``.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Tuple

from .report import PREFIX

DIR = "upstream_issues"                              # <tree>/upstream_issues/<ID>_<slug>.py
FLAG = "--upstream-fix"
MODULE_PREFIX = "ef2inv_opt_upstream_issues"         # sys.modules name of a loaded fix: f"{MODULE_PREFIX}.{ID}" (loaded by path; the IDs are not importable names)
NOT_APPLIED = "UPSTREAM-FIX NOT APPLIED"             # the refusal words when a requested fix cannot be installed (the run never proceeds without it)


class UnknownUpstreamFix(ValueError):
    """``--upstream-fix`` names an ID with no file under upstream_issues/ (refused by name before anything launches)."""


class UpstreamFixRefused(ValueError):
    """A requested fix's own run precondition does not hold (the issue file's ``refuse(run)`` words): refused by name before anything
    launches — a set flag is applied or the run does not start."""


def available(home: str) -> Dict[str, str]:
    """{ID: path} of the switchable issue files in the tree: ``<home>/upstream_issues/<ID>_<slug>.py`` (the ID is the file name up to its first ``_``)."""
    out: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(home, DIR, "*_*.py"))):
        fid = os.path.basename(path).split("_", 1)[0]
        if fid in out:
            raise ValueError(f"two upstream issue files carry the ID {fid!r}: {out[fid]} and {path}")
        out[fid] = path
    return out


def parse(value: Optional[str]) -> List[str]:
    """The flag's value → the IDs in order, duplicates dropped: ``None``/empty → ``[]``; ``"EF2INV-0003"``; ``"EF2INV-0003,EF2INV-0004"`` (spaces around items allowed)."""
    ids: List[str] = []
    for item in (value or "").split(","):
        item = item.strip()
        if item and item not in ids:
            ids.append(item)
    return ids


def resolve(ids: List[str], home: str, run: Optional[dict] = None) -> List[Tuple[str, str]]:
    """[(ID, path)] for the requested IDs; :class:`UnknownUpstreamFix` naming the first unknown one and the known set. With ``run`` (the
    CLI's words for the run: ``{"verb", "mode"}``) each issue file is loaded and its optional ``refuse(run)`` precondition checked:
    :class:`UpstreamFixRefused` ``--upstream-fix <ID> <words>``. Nothing is loaded for no IDs."""
    if not ids:
        return []
    avail = available(home)
    for fid in ids:
        if fid not in avail:
            raise UnknownUpstreamFix(f"unknown upstream fix {fid!r} ({FLAG}); known: {', '.join(sorted(avail)) or 'none'} — one file per issue under {DIR}/ "
                                     f"(README 'Known upstream issues')")
    if run is not None:
        for fid in ids:
            refuse = getattr(load(fid, avail[fid]), "refuse", None)
            words = refuse(run) if callable(refuse) else None
            if words:
                raise UpstreamFixRefused(f"{FLAG} {fid} {words}")
    return [(fid, avail[fid]) for fid in ids]


def module_name(fid: str) -> str:
    return f"{MODULE_PREFIX}.{fid}"


def load(fid: str, path: str):
    """The issue file as a module (by path; registered in sys.modules under :func:`module_name` so a second apply finds the same module)."""
    name = module_name(fid)
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if getattr(mod, "ID", None) != fid or not callable(getattr(mod, "apply", None)):
        sys.modules.pop(name, None)
        raise ImportError(f"{path}: an upstream issue file defines ID == {fid!r} (found {getattr(mod, 'ID', None)!r}) and apply()")
    return mod


def loaded() -> List[str]:
    """The IDs whose issue file is loaded in this process (empty on every default run)."""
    return sorted(n[len(MODULE_PREFIX) + 1:] for n in sys.modules if n.startswith(MODULE_PREFIX + "."))


def _log(msg: str) -> None:
    print(f"{PREFIX} {msg}", file=sys.stderr, flush=True)


def apply(ids: List[str], home: str, log=None) -> List[dict]:
    """Install the requested fixes in THIS process, in the order given: load each file, call its ``apply(log=…)``, print
    ``UPSTREAM-FIX <ID> applied file=… targets=…``. Returns the records for run.json / the manifest (``[{"id", "file", "summary", "targets", …}]``);
    ``[]`` (nothing loaded, nothing printed) for no IDs. An exception from a fix propagates: the caller refuses the run by name."""
    log = log or _log
    records: List[dict] = []
    for fid, path in resolve(ids, home):
        mod = load(fid, path)
        rec = dict(mod.apply(log=log) or {})
        rec["id"] = fid
        rec["file"] = f"{DIR}/{os.path.basename(path)}"
        log(f"UPSTREAM-FIX {fid} applied file={rec['file']} targets={','.join(rec.get('targets') or []) or '-'}")
        records.append(rec)
    return records


def ids_of(records: Optional[List[dict]]) -> List[str]:
    return [r["id"] for r in (records or [])]
