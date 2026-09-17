"""Variants — the model sizes, one per process — and their checkpoints, from ONE source: the kit's own pins module
(`opt/forward/engines/e1/kits/pins.py`, `WEIGHTS`: HF repo, revision, the weights file's sha256 and size, the hidden width the kit
detects a size by). The module is loaded from its file path (importlib, stdlib-only imports inside it), so no kit package is imported
by reading the registry — the stock runner does the same (stock_score.py). The tree's `stock/PINS.json`, when present, must agree
with the pins module (`cross_check`); it is never a second source of the values.

Variant names are `WEIGHTS`' keys (150m | 300m | 600m); the environment switch is E1_VARIANT (`stack.ENV_VARIANT`).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Dict, List, Optional

from ._names import PINS_MODULE_NAME                  # noqa: F401 — the one names module
PINS_RELPATH = os.path.join("engines", "e1", "kits", "pins.py")

_LOADED = {}


def load_pins(forward_root: str):
    """The kit's pins module, loaded from `<forward_root>/engines/e1/kits/pins.py` by file path (cached per path). When the kit package
    has already imported it in this process (`engines.e1.kits.pins`, the exact route), that object is returned instead — one source."""
    kp = sys.modules.get("engines.e1.kits.pins")
    if kp is not None and getattr(kp, "WEIGHTS", None):
        return kp
    path = os.path.abspath(os.path.join(forward_root, PINS_RELPATH))
    if path in _LOADED:
        return _LOADED[path]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"kit pins module not found at {path}")
    spec = importlib.util.spec_from_file_location(PINS_MODULE_NAME, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PINS_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    _LOADED[path] = mod
    return mod


def variants(pins) -> List[str]:
    return list(pins.WEIGHTS)


def check_variant(variant: Optional[str], pins) -> Optional[str]:
    """The variant name, normalised (None stays None: the caller decides whether it is required); ValueError for an unknown one."""
    if variant is None:
        return None
    v = str(variant).strip().lower()
    if v not in pins.WEIGHTS:
        raise ValueError(f"unknown variant {variant!r} (expected {'|'.join(variants(pins))})")
    return v


def checkpoint(variant: str, pins) -> Dict:
    """{repo, rev, sha256, bytes, hidden_size, file} for the variant, as the pins module states them."""
    w = pins.WEIGHTS[variant]
    return {"variant": variant, "repo": w.get("repo"), "rev": w.get("rev"), "sha256": w.get("sha256"), "bytes": w.get("bytes"),
            "hidden_size": w.get("hidden_size"), "file": w.get("file")}


def snapshot_dir(variant: str, pins, hf_home: Optional[str] = None) -> str:
    """The HF-cache snapshot directory of the pinned revision — the kit's own resolution (`pins.weights_snapshot`); this is the
    `--model-name` the package passes to the upstream CLI (a directory, so nothing is fetched)."""
    return pins.weights_snapshot(variant, hf_home)


def cross_check(pins, pins_json_path: str) -> List[str]:
    """Disagreements between the tree's stock/PINS.json (when it exists) and the pins module: stock commit, per-variant repo/rev/sha256.
    Returns [] when the file is absent or agrees. The JSON's shape is the one the tree's stock/ carries: {"upstream": {"E1": {"commit"}},
    "weights": {<variant>: {"repo", "rev", "sha256"}, ...notes}} — keys that are absent are not compared; non-dict weights entries are the file's notes."""
    if not os.path.isfile(pins_json_path):
        return []
    try:
        pj = json.load(open(pins_json_path, "r", encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{pins_json_path}: unreadable ({e})"]
    bad = []
    up = (pj.get("upstream") or {}).get(pins.STOCK.get("package", "E1")) or {}
    if up.get("commit") and up["commit"] != pins.STOCK["commit"]:
        bad.append(f"stock commit: PINS.json {up['commit']} != pins.STOCK {pins.STOCK['commit']}")
    for v, w in (pj.get("weights") or {}).items():
        if not isinstance(w, dict):
            continue                                                # the file's own notes (layout, cite), not a variant
        if v not in pins.WEIGHTS:
            bad.append(f"weights {v}: in PINS.json, not in pins.WEIGHTS")
            continue
        for k in ("repo", "rev", "sha256"):
            if w.get(k) and w[k] != pins.WEIGHTS[v].get(k):
                bad.append(f"weights {v}.{k}: PINS.json {w[k]} != pins.WEIGHTS {pins.WEIGHTS[v].get(k)}")
    return bad
