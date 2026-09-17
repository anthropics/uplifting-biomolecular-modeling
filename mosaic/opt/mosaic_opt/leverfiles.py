"""`run.sh install`, its last software step — the kit's lever files placed inside the installed `mosaic` package (`site-packages/mosaic/fast/`),
the one place `fast` and `big` execute them from (`levers.py`: a per-step lever runs from the installed package's copy, never from a tree or
tools/ file; `exact` and `off` do not need it). The files are the kit's `mosaic_fast/*.py` (`registry.KIT_FAST_DIR`
under the kit home, `stack.kit_home()`); the installed package is found without importing it (`stack.installed_mosaic_dir()`).

The copy is add-only and idempotent: a file already byte-equal to the kit's is left alone, an absent or differing one is written, nothing is
ever removed — a `.py` in the installed directory that the kit does not ship is named (the routes' `check` refuses it by name:
`stack.installed_fast_check`) and left for the user to delete. A read-only environment whose copy is already the kit's therefore passes
with nothing written. ``python -m mosaic_opt.leverfiles [--dry-run]``; exit 0 done (or nothing to do), 1 a write failed, 3 `mosaic` is not
installed on this interpreter.
"""
from __future__ import annotations

import filecmp
import os
import shutil
import sys
from typing import Optional

from .registry import KIT_FAST_DIR

PREFIX = "[mosaic-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_NOT_INSTALLED = 0, 1, 3


def source_dir(kit: Optional[str] = None) -> str:
    """The kit's lever files (`<kit home>/mosaic_fast`)."""
    from . import stack
    return os.path.join(kit or stack.kit_home(), KIT_FAST_DIR)


def plan(kit: Optional[str] = None, mosaic_dir: Optional[str] = None) -> dict:
    """What the copy would do: per kit file `same` | `differs` | `absent` against `<mosaic_dir>/fast/`, plus `extra` (installed `.py` the kit
    does not ship). `mosaic_dir` None = the installed package of this interpreter (None again in the result when it is not installed)."""
    from . import stack
    src = source_dir(kit)
    md = stack.installed_mosaic_dir() if mosaic_dir is None else mosaic_dir
    target = os.path.join(md, "fast") if md else None
    names = sorted(f for f in os.listdir(src) if f.endswith(".py"))
    files = {}
    for fn in names:
        got = os.path.join(target, fn) if target else None
        if not got or not os.path.isfile(got):
            files[fn] = "absent"
        else:
            files[fn] = "same" if filecmp.cmp(os.path.join(src, fn), got, shallow=False) else "differs"
    extra = sorted(f for f in os.listdir(target) if f.endswith(".py") and f not in files) if target and os.path.isdir(target) else []
    return {"source": src, "mosaic_dir": md, "target": target, "files": files, "extra": extra}


def install(kit: Optional[str] = None, mosaic_dir: Optional[str] = None, dry_run: bool = False) -> dict:
    """Copy every `absent` / `differs` file of `plan()` into `<mosaic_dir>/fast/` (created when missing); `written` names them. Raises
    OSError from the copy (a read-only environment whose copy differs) and ValueError when `mosaic` is not installed."""
    p = plan(kit, mosaic_dir)
    if not p["target"]:
        raise ValueError("mosaic is not installed on this interpreter (no `mosaic` package on sys.path): install the pinned stack first (README.md 'Install')")
    todo = [fn for fn, st in p["files"].items() if st != "same"]
    p["written"] = [] if dry_run else todo
    p["would_write"] = todo
    if todo and not dry_run:
        os.makedirs(p["target"], exist_ok=True)
        for fn in todo:
            shutil.copyfile(os.path.join(p["source"], fn), os.path.join(p["target"], fn))
    return p


def summary(p: dict, dry_run: bool = False) -> str:
    n, total = len(p["would_write"]), len(p["files"])
    same = total - n
    verb = "would be written" if dry_run else "written"
    head = (f"{PREFIX} lever files: {total}/{total} already the kit's at {p['target']} — nothing {verb}" if n == 0 else
            f"{PREFIX} lever files: {n} of {total} {verb} into {p['target']} ({same} already the kit's): {', '.join(p['would_write'])}")
    tail = f"; extra files there the kit does not ship (left in place; `check` refuses them by name): {', '.join(p['extra'])}" if p["extra"] else ""
    return head + tail + " — fast and big run from that copy"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    dry_run = argv == ["--dry-run"]
    if argv and not dry_run:
        print(f"usage: python -m mosaic_opt.leverfiles [--dry-run]  (got {' '.join(argv)!r})", file=sys.stderr)
        return 2
    try:
        p = install(dry_run=dry_run)
    except ValueError as e:
        print(f"{PREFIX} REFUSED: {e}", file=sys.stderr)
        return EXIT_NOT_INSTALLED
    except OSError as e:
        print(f"{PREFIX} FAILED: the lever files could not be written ({type(e).__name__}: {e}) — a read-only environment needs them placed at build time", file=sys.stderr)
        return EXIT_FAIL
    print(summary(p, dry_run), flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
