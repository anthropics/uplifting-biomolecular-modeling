"""The models the kit serves and their checkpoints, from ONE source per fact: repository ids, revisions, species counts and native
windows from the lever tree's own table (accel/__init__.py MODELS, read by AST — nothing of the levers is imported by reading it), the
per-file digests of each staged snapshot from the tree's stock/PINS.json (`weights.<key>.files`). The primary model is the one the
install step stages by default. A model is named by its key (v100-200m) or its repository id (songlab/gpn-star-hg38-v100-200m)."""
from __future__ import annotations

import ast
import json
import os
from typing import Dict, List, Optional

from ._names import ACCEL_DIR

PRIMARY = "v100-200m"
ALLOW_PATTERNS = None                                               # the whole snapshot is staged: every file the pins list is then present to check


def opt_home() -> str:
    """The gpnstar/opt directory (this package's parent)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tree_home() -> str:
    """The gpnstar/ directory: beside opt/ (the package is installed editable from the tree)."""
    return os.path.dirname(opt_home())


def pins_json_path() -> str:
    return os.path.join(tree_home(), "stock", "PINS.json")


def load_pins(path: Optional[str] = None) -> dict:
    path = path or pins_json_path()
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def models(accel_dir: Optional[str] = None) -> Dict[str, dict]:
    """{key: {repo, revision, n_species, native_window, ...}} — the MODELS table of accel/__init__.py, evaluated as a literal (dict calls
    of constants), never imported."""
    path = os.path.join(accel_dir or ACCEL_DIR, "__init__.py")
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODELS" for t in node.targets):
            out = {}
            if not isinstance(node.value, ast.Dict):
                raise ValueError(f"{path}: MODELS is not a dict literal — the file changed shape")
            for k, v in zip(node.value.keys, node.value.values):
                key = ast.literal_eval(k)
                if isinstance(v, ast.Call) and getattr(v.func, "id", None) == "dict":
                    out[key] = {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords}
                else:
                    out[key] = ast.literal_eval(v)
            return out
    raise ValueError(f"{path}: no top-level MODELS assignment — the file changed shape")


def gpn_commit(accel_dir: Optional[str] = None) -> str:
    """The upstream commit the levers are written against (accel/__init__.py GPN_COMMIT, by AST)."""
    from opt_core.modes import literal_assignment
    return literal_assignment(os.path.join(accel_dir or ACCEL_DIR, "__init__.py"), "GPN_COMMIT")


def keys() -> List[str]:
    return list(models())


def resolve(name: Optional[str]) -> Optional[str]:
    """The model key for a key, a repository id, a repository tail, or a snapshot directory of the hub cache layout; None when the name is
    not one of the served models (the caller words it: unpinned)."""
    if name is None or not str(name).strip():
        return PRIMARY
    n = str(name).strip().rstrip("/")
    table = models()
    if n in table:
        return n
    low = n.lower()
    for k, m in table.items():
        repo = m["repo"].lower()
        if low == repo or low == repo.split("/")[-1] or f"models--{repo.replace('/', '--')}" in low.replace(os.sep, "/"):
            return k
    return None


def checkpoint(key: str, pins: Optional[dict] = None) -> dict:
    """{key, repo, rev, files: {relpath: {sha256, bytes}}, n_species, native_window} — repo/rev from MODELS, files from PINS.json."""
    m = models()[key]
    files = {}
    try:
        w = weights_entry(key, pins if pins is not None else load_pins())
    except (OSError, ValueError):
        w = {}
    raw = w.get("files") or {}
    for rel, ent in raw.items():
        if isinstance(ent, dict):
            files[rel] = {"sha256": ent.get("sha256"), "bytes": ent.get("size_bytes", ent.get("bytes"))}
        else:
            files[rel] = {"sha256": str(ent), "bytes": None}
    pins_rev = w.get("snapshot_commit") or w.get("rev")
    if pins_rev and pins_rev != m["revision"]:
        files = {}                                            # digests of another revision are not this checkpoint's
    return {"key": key, "repo": m["repo"], "rev": m["revision"], "files": files, "n_species": m.get("n_species"), "native_window": m.get("native_window"),
            "pins_rev": pins_rev}


def weights_entry(key: str, pins: dict) -> dict:
    """The stock/PINS.json weights entry of a model: keyed by repository id (the tree's schema) or by key."""
    w = pins.get("weights") or {}
    repo = models()[key]["repo"]
    ent = w.get(repo) or w.get(key) or {}
    return ent if isinstance(ent, dict) else {}


def label(key: Optional[str], repo: Optional[str] = None, rev: Optional[str] = None) -> str:
    """`<repo tail>@<rev8>` — the ACTIVE line's model= value."""
    if key is not None:
        m = models()[key]
        repo, rev = m["repo"], rev or m["revision"]
    tail = (repo or "none").split("/")[-1]
    return f"{tail}@{(rev or 'none')[:8]}"


def snapshot_dir(key: str, hf_home: Optional[str] = None) -> str:
    """The hub-cache snapshot directory of the pinned revision under an HF_HOME (the library's own layout)."""
    m = models()[key]
    root = hf_home or os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    hub = os.environ.get("HF_HUB_CACHE") if hf_home is None and os.environ.get("HF_HUB_CACHE") else os.path.join(root, "hub")
    return os.path.join(hub, f"models--{m['repo'].replace('/', '--')}", "snapshots", m["revision"])


def cross_check(pins: Optional[dict] = None) -> List[str]:
    """Disagreements between stock/PINS.json and the lever tree's tables: the upstream commit, per-model repo / rev. [] = agree or absent."""
    try:
        pj = pins if pins is not None else load_pins()
    except OSError:
        return []
    except ValueError as e:
        return [f"{pins_json_path()}: unreadable ({e})"]
    bad = []
    up = ((pj.get("upstream") or {}).get("gpn") or {})
    if up.get("commit") and up["commit"] != gpn_commit():
        bad.append(f"stock commit: PINS.json {up['commit']} != accel GPN_COMMIT {gpn_commit()}")
    table = models()
    by_repo = {m["repo"]: k for k, m in table.items()}
    for name, w in (pj.get("weights") or {}).items():
        if not isinstance(w, dict) or not (w.get("snapshot_commit") or w.get("rev")):
            continue                                          # the file's notes (root, layout), not a checkpoint
        k = by_repo.get(name) or (name if name in table else None)
        if k is None:
            bad.append(f"weights {name}: in PINS.json, not in accel MODELS")
            continue
        rev = w.get("snapshot_commit") or w.get("rev")
        if rev != table[k]["revision"]:
            bad.append(f"weights {name}: PINS.json revision {rev} != MODELS {table[k]['revision']}")
    return bad
