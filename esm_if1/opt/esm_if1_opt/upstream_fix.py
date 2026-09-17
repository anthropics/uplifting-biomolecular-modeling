"""``--upstream-fix <ID>[,<ID>…]``: confirmed upstream issues, fixed on request only (README 'Known upstream issues').

Each switchable issue is one file ``<tree>/upstream_issues/<ID>_<slug>.py`` beside opt/ — the fix as a runtime rebinding plus its docstring
(what upstream does · evidence · what the fix changes · expected effect on outputs). The directory is never on ``sys.path`` and nothing
imports a file from it unless a run names its ID: ``design`` resolves the flag before anything launches (an unknown ID is refused by name,
a usage error), hands the resolved files to the process that runs the model — the proven stock child under ``off`` (after its proof), the
kit child under ``fast`` — which loads each by path, calls its ``apply()`` and prints ``[esm_if1-opt] UPSTREAM-FIX <ID> applied (<words>)``;
opt_manifest.json records the IDs under ``kit.upstream_fix``. With the flag absent no mode differs in any way and no module from
upstream_issues/ is loaded: every mode runs upstream's behaviour exactly as shipped. This engine has no stock exception.

Child hand-off grammar (both children): the leading tokens ``--upstream-fix <path>[,<path>…]`` of the child's argument list (``split_argv``).
This module imports nothing beyond the standard library.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Tuple

PREFIX = "[esm_if1-opt]"
DIR = "upstream_issues"                              # <tree>/upstream_issues/<ID>_<slug>.py
FLAG = "--upstream-fix"
MODULE_PREFIX = "esm_if1_opt_upstream_issues"        # sys.modules name of a loaded fix: f"{MODULE_PREFIX}.{ID}" (loaded by path; the IDs are not importable names)


class UnknownUpstreamFix(ValueError):
    """``--upstream-fix`` names an ID with no file under upstream_issues/ (refused by name before anything launches)."""


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
    """The flag's value → the IDs in order, duplicates dropped: ``None``/empty → ``[]``; ``"ESMIF1-001"``; ``"A,B"`` (spaces around items allowed)."""
    ids: List[str] = []
    for item in (value or "").split(","):
        item = item.strip()
        if item and item not in ids:
            ids.append(item)
    return ids


def resolve(ids: List[str], home: str) -> List[Tuple[str, str]]:
    """[(ID, path)] for the requested IDs; :class:`UnknownUpstreamFix` naming the first unknown one and the known set. Nothing is loaded."""
    if not ids:
        return []
    avail = available(home)
    for fid in ids:
        if fid not in avail:
            raise UnknownUpstreamFix(f"unknown upstream fix {fid!r} ({FLAG}); known: {', '.join(sorted(avail)) or 'none'} — one file per issue under {DIR}/ "
                                     f"(README 'Known upstream issues')")
    return [(fid, avail[fid]) for fid in ids]


def child_tokens(resolved: List[Tuple[str, str]]) -> List[str]:
    """The leading tokens a child receives: ``[--upstream-fix, "<path>,<path>"]`` or ``[]``."""
    return [FLAG, ",".join(p for _, p in resolved)] if resolved else []


def split_argv(argv: List[str]) -> Tuple[List[str], List[str]]:
    """``(fix file paths, the rest)`` of a child's argument list: a leading ``--upstream-fix <paths>`` pair is the hand-off, anything else is the payload."""
    argv = list(argv)
    if len(argv) >= 2 and argv[0] == FLAG:
        return [p for p in argv[1].split(",") if p], argv[2:]
    return [], argv


def load(path: str):
    """Import one issue file by path (never through sys.path); cached in ``sys.modules`` under ``MODULE_PREFIX.<ID>``."""
    fid = os.path.basename(path).split("_", 1)[0]
    name = f"{MODULE_PREFIX}.{fid}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def apply_files(paths: List[str]) -> List[dict]:
    """In the process that runs the model: load each file, call its ``apply()``, print ``UPSTREAM-FIX <ID> applied (<words>)``; returns the records."""
    records = []
    for path in paths:
        mod = load(path)
        rec = dict(mod.apply())
        rec["path"] = path
        records.append(rec)
        sys.stderr.write(f"{PREFIX} UPSTREAM-FIX {mod.ID} applied ({getattr(mod, 'WORDS', 'see ' + os.path.basename(path))})\n")
        sys.stderr.flush()
    return records
