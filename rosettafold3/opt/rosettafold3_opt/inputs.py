"""Inputs: the upstream CLI's own JSON (``rf3 fold inputs=<file>``; ``models/rf3/README.md:96-114``) — one item or a list of
items, each ``{"name": ..., "components": [{"seq"|"ccd_code"|..., "chain_id", "chain_type"?, "msa_path"?}, ...]}``.

``warm``'s helper: it validates the shape of the add-on's public input, selects its one item by name and writes it as the file ``rf3 fold``
reads (``pred`` hands its ``--input`` to ``inputs=`` verbatim and never reads it). It never rewrites an item's content.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List


class InputError(ValueError):
    pass


def load(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        raise InputError(f"input file not found: {path}")
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise InputError(f"{path}: not JSON ({e})") from e
    items = data if isinstance(data, list) else [data]
    for i, it in enumerate(items):
        if not isinstance(it, dict) or "components" not in it:
            raise InputError(f"{path}: item {i} has no 'components' (the upstream JSON schema)")
        if not isinstance(it["components"], list) or not it["components"]:
            raise InputError(f"{path}: item {i} ({it.get('name')}) has an empty 'components' list")
        for j, c in enumerate(it["components"]):
            if not isinstance(c, dict) or not any(k in c for k in ("seq", "ccd_code", "smiles", "sdf_path", "cif_path", "path")):
                raise InputError(f"{path}: item {i} component {j} names neither seq nor ccd_code/smiles/sdf_path")
    names = [it.get("name") for it in items]
    if len(set(names)) != len(names):
        raise InputError(f"{path}: duplicate item names {names}")
    return items


def write(items: List[Dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=1)
        fh.write("\n")
    return path

