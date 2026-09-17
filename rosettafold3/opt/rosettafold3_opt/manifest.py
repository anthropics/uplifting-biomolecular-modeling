"""The checkpoint facts of the WEIGHTS line: the file's path and size, its sha256 when computed, and the pin's (stock/PINS.json "weights")."""
from __future__ import annotations

import os


def checkpoint_info(path: str | None, pins: dict | None, sha: str | None = None) -> dict:
    w = ((pins or {}).get("weights") or {}).get("rf3") or {}
    info = {"path": path, "bytes": os.path.getsize(path) if path and os.path.isfile(path) else None,
            "pinned_sha256": w.get("sha256"), "pinned_bytes": w.get("bytes"), "pinned_filename": w.get("filename"),
            "sha256_matches_pin": None}
    if sha is not None:
        info["sha256"] = sha
        info["sha256_matches_pin"] = (sha == w.get("sha256"))
    elif info["bytes"] is not None and w.get("bytes") is not None:
        info["bytes_match_pin"] = (info["bytes"] == w.get("bytes"))
    return info
