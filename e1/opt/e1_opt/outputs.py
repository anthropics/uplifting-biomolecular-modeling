"""Outputs of the wrapper command: the tool's own file only — `score` writes upstream's `--output-path` (columns as the tool writes
them); the kit writes no file of its own beside it. `scores_path(out_dir, item)` is the per-item layout `<out_dir>/<item>/scores.csv`.

`listing(path)` reads a scores.csv back: sha256, byte size, the header and the number of data rows (lines after the header).
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict

SCORES_NAME = "scores.csv"


def item_out_dir(out_dir: str, item: str) -> str:
    return os.path.join(os.path.abspath(out_dir), item)


def scores_path(out_dir: str, item: str) -> str:
    return os.path.join(item_out_dir(out_dir, item), SCORES_NAME)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def listing(path: str) -> Dict:
    """{present, path, sha256, bytes, header, n_rows}: n_rows counts the non-empty lines after the header."""
    if not os.path.isfile(path):
        return {"present": False, "path": path, "sha256": None, "bytes": None, "header": None, "n_rows": 0}
    header, n = None, 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                header = line.rstrip("\r\n")
                continue
            if line.strip():
                n += 1
    return {"present": True, "path": path, "sha256": sha256_file(path), "bytes": os.path.getsize(path), "header": header, "n_rows": n}
