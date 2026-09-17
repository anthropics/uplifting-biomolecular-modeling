"""The kit registry: where the kit is, that its tree is complete, and the state of the ten target files in an interpreter.

The kit is carried under `opt/forward/xattempt_addon/` (`kit_home()`); its own bytes are this repo's tracked files — the
git commit identifies them, so nothing here re-hashes them against a stored digest. `MANIFEST.json` "files" lists every path the
tree carries; `check_tree()` checks that
listing is on disk and that no stray file sits beside it. The installer's two EXTERNAL-state tables — `STOCK_SHA256_upstream_4010e3e.txt`
(the eight upstream files at the pin) and `STOCK_SHA256_upstream_faf4299_extra.txt` (upstream's later `layer_utils.py`) — fingerprint states
that are not tracked at those paths in this tree (an upstream checkout); `classify(site)` names the state of each target
file with the installer's own vocabulary (`install.sh:21-27`: patched | stock | stock-head | absent-ok | unknown) by hashing
the interpreter's live tree against `patched/` directly (the kit's own carried bytes) and against those two tables, without
executing anything — the gate `check` and `enable()` use before a subprocess exists. State CHANGES go through the kit's installer
only (install.py).
`checkpoint_sha256(path)` is the checkpoint digest every reader shares (the manifest's `checkpoint` block, the KERNELS line's `ckpt=` word):
the 2.7 GB read is paid once per (path, size, mtime) and kept in `$RFDIFFUSION3_OPT_CACHE/checkpoint_sha256.json` (`digest_cache_path`).
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import subprocess
from typing import Dict, List, Optional, Tuple

KIT_RELPATH = os.path.join("opt", "forward", "xattempt_addon")
ENV_HOME = "MODEL_OPT"                                  # the rfdiffusion3/ directory (configs export it; default: relative to this package)
TABLES = {"stock": "STOCK_SHA256_upstream_4010e3e.txt",                # the installer's two EXTERNAL-state reference tables (an upstream checkout at the
          "stock-head": "STOCK_SHA256_upstream_faf4299_extra.txt"}   # pin, upstream's later layer_utils.py — not this tree's own bytes)
NEW_FILES = ("rfd3/model/hoist.py", "rfd3/model/cudagraph_sampler.py")   # absent in stock: install.sh's `absent-ok`


def tree_home() -> str:
    """The rfdiffusion3/ directory: MODEL_OPT when set, else two levels above this package (opt/rfdiffusion3_opt/ -> rfdiffusion3/)."""
    env = os.environ.get(ENV_HOME)
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def kit_home() -> str:
    return os.path.join(tree_home(), KIT_RELPATH)


def pins() -> dict:
    """stock/PINS.json: the upstream pin, the stock file shas, the pinned checkpoint and the stock arm's environment rules."""
    with open(os.path.join(tree_home(), "stock", "PINS.json"), encoding="utf-8") as fh:
        return json.load(fh)


def check_pins_module():
    """stock/check_pins.py loaded from its file — the one place the pin check lives (standard library only); shared by the
    activation core (stack) and the stock caller (stock_design), which loads nothing else of this package."""
    import importlib.util
    path = os.path.join(tree_home(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("rfdiffusion3_check_pins", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_table(name: str, kit: Optional[str] = None) -> Dict[str, str]:
    """A kit sha table (`<sha>  <path>` lines) -> {path: sha}."""
    out = {}
    with open(os.path.join(kit or kit_home(), name), encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                sha, path = line.split()
                out[path] = sha
    return out


def manifest(kit: Optional[str] = None) -> dict:
    with open(os.path.join(kit or kit_home(), "MANIFEST.json"), encoding="utf-8") as fh:
        return json.load(fh)


def carried_files(kit: Optional[str] = None) -> List[str]:
    """The MANIFEST "files" entries: every relative path the tree carries of the kit (no digest — the bytes are the tracked files;
    the git commit identifies them)."""
    return list(manifest(kit)["files"])


def check_tree(kit: Optional[str] = None) -> dict:
    """Every file MANIFEST.json "files" names is present under the kit directory, and no stray file sits beside them.
    Tamper detection (a present-but-edited file) is never a stored digest — inside a git checkout, `git diff` against
    HEAD is the check (the commit identifies the bytes); a kit copied somewhere with no `.git` (carried into an install) has
    nothing in-repo to compare against, and that is a named skip (`git_check`), never a silent one.
    Returns {"ok", "kit", "checked", "missing": [...], "stray": [...], "version", "modified": [...],
    "git_check": "clean"|"modified"|"skipped: <reason>"}."""
    kit = kit or kit_home()
    res = {"ok": False, "kit": kit, "checked": 0, "missing": [], "stray": [], "version": None, "modified": [], "git_check": None}
    if not os.path.isfile(os.path.join(kit, "MANIFEST.json")):
        res["reason"] = f"kit MANIFEST.json missing under {kit}"
        return res
    man = manifest(kit)
    res["version"] = man.get("version")
    want = carried_files(kit)
    for rel in want:
        if os.path.isfile(os.path.join(kit, rel)):
            res["checked"] += 1
        else:
            res["missing"].append(rel)
    known = set(want) | {"MANIFEST.json"}
    for root, dirs, files in os.walk(kit):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), kit).replace(os.sep, "/")
            if rel not in known and "__pycache__" not in rel:
                res["stray"].append(rel)
    try:
        is_repo = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=kit, capture_output=True, text=True)
    except OSError as ex:
        res["git_check"] = f"skipped: git not available ({ex})"
        is_repo = None
    if is_repo is not None and is_repo.returncode == 0 and is_repo.stdout.strip() == "true":
        present = [rel for rel in want if rel not in res["missing"]]
        diff = subprocess.run(["git", "diff", "--name-only", "HEAD", "--"] + present, cwd=kit, capture_output=True, text=True) if present else None
        if diff is not None and diff.returncode == 0:
            res["modified"] = sorted(l for l in diff.stdout.splitlines() if l)
            res["git_check"] = "modified" if res["modified"] else "clean"
        elif diff is None:
            res["git_check"] = "clean"        # nothing present to diff (all missing; already named above)
        else:
            res["git_check"] = f"skipped: git diff failed ({diff.stderr.strip()[:100]})"
    elif is_repo is not None:
        res["git_check"] = "skipped: not a git checkout"
    res["ok"] = not (res["missing"] or res["stray"] or res["modified"]) and res["version"] is not None
    return res


def target_files(kit: Optional[str] = None) -> List[str]:
    """The ten files install.sh overlays, as `rfd3/...` paths (every file under the kit's `patched/` directory, sorted).
    `__pycache__` is skipped: importing (or byte-compiling) a module under `patched/` — which every interpreter that
    activates the kit does — writes bytecode cache there; it is never a target file."""
    kit = kit or kit_home()
    base = os.path.join(kit, "patched")
    out = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            out.append(os.path.relpath(os.path.join(root, f), base).replace(os.sep, "/"))
    return sorted(out)


def site_dir(package: str = "rfd3") -> Optional[str]:
    """The site-packages that holds the installed package, located without executing it (the path finder alone, so the package's own
    autoload finder is not consulted by the probe); None when not installed."""
    spec = importlib.machinery.PathFinder.find_spec(package)
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def classify(site: Optional[str] = None, kit: Optional[str] = None) -> Dict[str, str]:
    """{rfd3/... path: state} for the ten target files of the interpreter's rfd3 tree, with install.sh's vocabulary and rule
    (`state_of`, install.sh:21-27): patched > stock > stock-head > absent-ok (a new file that is absent) > unknown; `absent`
    for a stock file that is missing. `site` defaults to this interpreter's. "patched" is a direct hash against the kit's own
    `patched/<rel>` (the tracked bytes); the other two states are the installer's external reference tables."""
    kit = kit or kit_home()
    site = site or site_dir()
    if site is None:
        return {}
    tables = {name: read_table(fn, kit) for name, fn in TABLES.items()}
    out = {}
    for rel in target_files(kit):
        p = os.path.join(site, rel) if site else None
        if p is None or not os.path.isfile(p):
            out[rel] = "absent-ok" if rel in NEW_FILES else "unknown"     # install.sh: an absent stock file is `unknown`
            continue
        sha = sha256_of(p)
        if sha == sha256_of(os.path.join(kit, "patched", rel)):
            out[rel] = "patched"
            continue
        state = "unknown"
        for name in ("stock", "stock-head"):
            if tables[name].get(rel) == sha:
                state = name
                break
        out[rel] = state
    return out


def summarize(states: Dict[str, str]) -> Tuple[str, str]:
    """(label, detail): `patched` when all ten are patched; `pristine` when the eight are stock and the two new files absent;
    `stock-head`, `mixed`, `unknown`, or `not-installed` (rfd3 not importable: classify() returned nothing)."""
    vals = list(states.values())
    n = len(vals)
    if not vals:
        return "not-installed", "no target files"
    if all(v == "patched" for v in vals):
        return "patched", f"{n}/{n} patched"
    if all(v == "stock" or (v == "absent-ok" and k in NEW_FILES) for k, v in states.items()):
        return "pristine", f"{n - len(NEW_FILES)}/{n - len(NEW_FILES)} stock, {len(NEW_FILES)} kit files absent"
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    label = "mixed"
    if set(counts) <= {"stock", "stock-head", "absent-ok"} and counts.get("stock-head"):
        label = "stock-head"
    elif counts.get("unknown"):
        label = "unknown"
    return label, ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


CACHE_ENV = "RFDIFFUSION3_OPT_CACHE"                    # the digest cache directory (default: ~/.cache/rfdiffusion3_opt)


def digest_cache_path() -> str:
    """The digest cache file: $RFDIFFUSION3_OPT_CACHE/checkpoint_sha256.json, default under the user's cache directory."""
    base = os.environ.get(CACHE_ENV) or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "rfdiffusion3_opt")
    return os.path.join(base, "checkpoint_sha256.json")


def checkpoint_sha256(path: str) -> Tuple[str, str]:
    """(sha256, source) of a checkpoint file: `cache` when the file's (absolute path, bytes, mtime_ns) were hashed before — the
    2.7 GB read is paid once per file, not once per `design` — else hashed now (`hashed`) and recorded; a cache that cannot be
    written is ignored."""
    st = os.stat(path)
    key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    cache_path = digest_cache_path()
    try:
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    if isinstance(cache, dict) and isinstance(cache.get(key), str):
        return cache[key], "cache"
    sha = sha256_of(path)
    cache = cache if isinstance(cache, dict) else {}
    cache[key] = sha
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = f"{cache_path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=1, sort_keys=True)
        os.replace(tmp, cache_path)
    except OSError:
        pass
    return sha, "hashed"
