"""The carry gate over the carried kit files: ``carry_check`` establishes that every file each add-on directory asked for (the add-on table in
stack) is actually carried under ``opt/forward/`` — full presence accounting against kit_required_files.json (generated once from the git tree,
paths only, never a digest — a carried file's version is the git commit, not a sum restated in this tree). It gates the modes that run kit code
(stack.activate)."""
from __future__ import annotations

import json
import os
from typing import Iterable, Optional

REQUIRED_FILES_NAME = "kit_required_files.json"


def _required_files() -> dict:
    """{add-on name (the add-on table in stack): [relative paths, sorted]} — generated once from `git ls-tree -r HEAD -- opt/forward/<kit>` (regenerate
    with the same command on a real change to an add-on's carried file set) and committed beside this module, so the required list is
    present wherever the package is installed, independent of whether opt/forward/ itself made it there intact."""
    with open(os.path.join(os.path.dirname(__file__), REQUIRED_FILES_NAME), encoding="utf-8") as f:
        return json.load(f)


def carry_check(kits: Optional[Iterable[str]] = None) -> dict:
    """The add-on directories named by the first argument (names from the add-on table in stack; default all): every file _required_files()[name] names is present under
    opt/forward/<rel>/, and every file actually found under there (__pycache__ and *.py[co] excluded) is accounted for.
    Returns the add-on names checked plus 'files', 'ok', 'missing', 'extra' — `missing` names each absent required file, relative to opt/ (the identifying file,
    the add-on's identifying file first if it is itself missing, so the existing single-file refusal message still names it); `extra` names
    files present on disk but not in the required list, relative to opt/. `files` counts the required files found present."""
    from .stack import KITS, opt_home
    opt = opt_home()
    names = list(kits) if kits is not None else list(KITS)
    required = _required_files()
    res = {"kits": names, "files": 0, "ok": False, "missing": [], "extra": []}
    for name in names:
        rel, marker = KITS[name]
        root = os.path.join(opt, rel)
        want = set(required[name])
        missing_here = sorted(p for p in want if not os.path.isfile(os.path.join(root, p)))
        if marker in missing_here:                                       # the identifying file leads, so the single-file refusal names it first
            missing_here.remove(marker)
            missing_here.insert(0, marker)
        res["missing"].extend(os.path.join(rel, p) for p in missing_here)
        res["files"] += len(want) - len(missing_here)
        found = set()
        for d, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            reldir = os.path.relpath(d, root)
            for fn in fs:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                found.add(fn if reldir == "." else os.path.join(reldir, fn))
        res["extra"].extend(os.path.join(rel, p) for p in sorted(found - want))
    res["ok"] = not res["missing"]
    return res
