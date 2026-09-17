"""The kit's JIT-cache tree helper — ``python -s -m boltz2_opt.jitcache relocate --root <MODEL_OPT_JIT_ROOT>``.

Why. A COPIED Triton cache tree is not relocatable on triton 3.7: beside a compiled kernel's files (``<TRITON_CACHE_DIR>/<hash>/<name>.cubin``,
``.json``, ``.ptx``, ``.ttir``, …) Triton writes a group file ``__grp__<name>.json`` = ``{"child_paths": {"<child>": "<ABSOLUTE path>"}}`` and,
at lookup, opens the children BY THOSE ABSOLUTE PATHS (``FileCacheManager.get_group``: a child whose recorded path does not exist is dropped,
the kernel counts as absent and is compiled again). A tree compiled under root A and served under root B therefore misses silently — which is
the image's warm layer: ``run.sh warm`` fills the warm machine's ``MODEL_OPT_JIT_ROOT``, the tree is tarred there and unpacked to
``/kit/boltz2/.jit`` (environment/Dockerfile), another absolute root unless the warm machine used that very path. ``relocate`` re-roots such a
tree IN PLACE: every group file under the root is rewritten so each child path names the child file that sits beside the group file (the group
file's own directory + the child's file name — where Triton put it), keeping the ``<stack key>/triton/<hash>/<file>`` layout as it is. Idempotent:
a tree compiled in place, or already relocated, is unchanged (``already``); a child that is not beside its group file is left as recorded and
counted (``unresolved``); a group file that is not Triton's JSON is left alone (``invalid``). Caches compiled in place (``run.sh warm`` on the
serving machine, a user's ``MODEL_OPT_JIT_ROOT``, the kit's exactln cubin directory — no group files) need nothing.

    python -s -m boltz2_opt.jitcache relocate --root /kit/boltz2/.jit
    [boltz2-opt jitcache] relocate root=/kit/boltz2/.jit groups=41 rewritten=41 already=0 unresolved=0 invalid=0
"""
import json
import os
import sys
from typing import Dict, Iterator, List, Optional

PREFIX = "[boltz2-opt jitcache]"
GROUP_PREFIX = "__grp__"                      # triton.runtime.cache.FileCacheManager: `__grp__<filename>` beside the kernel's files
CHILD_PATHS = "child_paths"                   # the group file's one key: {child file name: absolute path}


def group_files(root: str) -> Iterator[str]:
    """Every Triton group file under ``root`` (any depth: ``<stack key>/triton/<hash>/__grp__*.json`` and whatever else was tarred in)."""
    for dirpath, _dirs, files in os.walk(os.fspath(root)):
        for f in sorted(files):
            if f.startswith(GROUP_PREFIX) and f.endswith(".json"):
                yield os.path.join(dirpath, f)


def relocate_group(path: str) -> str:
    """Re-root ONE group file; returns ``rewritten`` | ``already`` | ``unresolved`` (at least one child not beside the file; the others are
    still re-rooted) | ``invalid`` (not Triton's group JSON: untouched)."""
    try:
        with open(path) as f:
            data = json.load(f)
        children = data.get(CHILD_PATHS) if isinstance(data, dict) else None
        if not isinstance(children, dict):
            return "invalid"
    except (OSError, ValueError):
        return "invalid"
    here = os.path.dirname(os.path.abspath(path))
    new: Dict[str, str] = {}
    changed = unresolved = 0
    for child, recorded in children.items():
        recorded = os.fspath(recorded) if isinstance(recorded, (str, os.PathLike)) else str(recorded)
        beside = os.path.join(here, os.path.basename(recorded) or str(child))
        if os.path.exists(beside):
            new[child] = beside
            changed += int(beside != recorded)
        else:                                                          # not beside its group file: left as recorded (Triton drops it at lookup, as before)
            new[child] = recorded
            unresolved += 1
    if changed:
        data[CHILD_PATHS] = new
        tmp = path + ".relocate.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)                                          # atomic: a reader never meets a half-written group file
    if unresolved:
        return "unresolved"
    return "rewritten" if changed else "already"


def relocate(root: str) -> Dict[str, object]:
    """Re-root every Triton group file under ``root`` in place (see the module text). Returns the census
    ``{"root", "groups", "rewritten", "already", "unresolved", "invalid"}``; a missing root is ``groups=0``."""
    out: Dict[str, object] = {"root": os.fspath(root), "groups": 0, "rewritten": 0, "already": 0, "unresolved": 0, "invalid": 0}
    if not os.path.isdir(os.fspath(root)):
        return out
    for g in group_files(root):
        out["groups"] = int(out["groups"]) + 1
        word = relocate_group(g)
        out[word] = int(out[word]) + 1
    return out


def line(census: Dict[str, object]) -> str:
    return f"{PREFIX} relocate " + " ".join(f"{k}={census[k]}" for k in ("root", "groups", "rewritten", "already", "unresolved", "invalid"))


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 3 and argv[0] == "relocate" and argv[1] == "--root":
        print(line(relocate(argv[2])), flush=True)
        return 0
    if len(argv) == 2 and argv[0] == "relocate" and argv[1].startswith("--root="):
        print(line(relocate(argv[1].split("=", 1)[1])), flush=True)
        return 0
    sys.stderr.write("usage: python -s -m boltz2_opt.jitcache relocate --root <MODEL_OPT_JIT_ROOT>\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
