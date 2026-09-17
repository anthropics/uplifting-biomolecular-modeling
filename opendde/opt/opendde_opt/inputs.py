"""Inputs: upstream's query JSON, the `pred` input, read as upstream reads it.

Upstream query JSON (upstream's `docs/infer_json_format.md`): a list of jobs `{name, modelSeeds, sequences: [{proteinChain: {sequence,
count, unpairedMsaPath?, pairedMsaPath?, templatesPath?, modifications?, ...}} | {rnaSequence | dnaSequence: {...}} | {ligand: {ligand:
"CCD_xxx" | SMILES, count}} | ...]}` — every entity kind, count and field upstream accepts is a `pred` input as it stands (no chain has a
role here: a single chain, a homomer, a protein with RNA/DNA/ligands, templated chains are all just jobs); `load_query` only insists on
the file being a JSON list (or one job object) so a malformed file is refused by name before the model starts.
"""
from __future__ import annotations

import json
import os
from opt_core import gates as _gates



sha256_file = _gates.sha256_file                                         # the core's file digest (one helper for every kit)


def load_query(path: str) -> list[dict]:
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not all(isinstance(j, dict) and "sequences" in j for j in data):
        raise ValueError(f"{path}: not an upstream query JSON (a list of jobs with `sequences`)")
    for j in data:
        j.setdefault("name", "job")
    return data


def describe_query(path: str) -> dict:
    jobs = load_query(path)
    return {"path": os.path.abspath(path), "sha256": sha256_file(path), "n_items": len(jobs), "items": [j.get("name") for j in jobs],
            "seeds": {j.get("name"): j.get("modelSeeds") for j in jobs}}


