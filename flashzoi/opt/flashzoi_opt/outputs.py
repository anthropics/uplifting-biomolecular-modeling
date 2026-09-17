"""The one output schema of `pred` (both routes): `<out>/<item>.npy` = the documented call's return verbatim (np.save of the
float32 (1, 4, 6144, 7611) array — the values the call produced, in C order), `<out>/rows.jsonl` = one JSON row per item
{item, shape, dtype, wall_s, ok, error?} written row by row as items complete (wall_s = the documented call's seconds, bookkeeping,
never compared across runs), `<out>/opt_manifest.json` (manifest.py); the stock route (flashzoi/opt/flashzoi_opt/stock_pred.py) writes the same
item files and rows and its own `<out>/opt_manifest.json` (mode off) carrying the environment proof under `stock_env_proof`. The .npy files carry
the bytes; a digest of them is the reader's to take (sha256_array here is what `warm` prints for its one window).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

ROWS_FILE = "rows.jsonl"
PROOF_KEY = "stock_env_proof"                                          # the key of the stock route's opt_manifest.json holding the environment proof


def sha256_array(y: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()


def item_path(out_dir: str, item: str) -> str:
    return os.path.join(out_dir, f"{item}.npy")


def write_item(out_dir: str, item: str, y: np.ndarray) -> dict:
    """np.save the array (C order; a C-contiguous array takes np.save's direct path, any other is reordered first) under a temporary name, rename
    it into place, and return the row fields (without the wall): the stock caller's row schema, key for key."""
    os.makedirs(out_dir, exist_ok=True)
    y = np.ascontiguousarray(y)
    p = item_path(out_dir, item)
    tmp = p + ".tmp.npy"
    np.save(tmp, y)
    os.replace(tmp, p)
    return {"item": item, "shape": list(y.shape), "dtype": str(y.dtype)}     # the stock caller's row fields exactly (opt/flashzoi_opt/stock_pred.py: + wall_s, ok by the loop); no key of the package's own — the file is <item>.npy by the item rule


def append_row(out_dir: str, row: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, ROWS_FILE), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


def read_rows(out_dir: str) -> list:
    p = os.path.join(out_dir, ROWS_FILE)
    if not os.path.isfile(p):
        return []
    return [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip()]


def write_json(out_dir: str, name: str, obj) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.write("\n")
    os.replace(tmp, p)
    return p
