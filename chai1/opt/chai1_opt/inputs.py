"""Inputs for ``pred``: the items file, translated to the kit driver's own input spec (the one translation; host-side, no torch).

The kit driver takes ``fasta:<path>`` specs, comma-joined, plus one ``--msa_dir`` of chai ``.aligned.pqt`` files matched by sequence
(``chai_worker.py`` usage; ``chai_proto.input_spec`` / ``fasta_spec``); every FASTA is folded verbatim, as chai-lab reads it. The package's
items file is a JSON document of the same facts:

    {"msa_dir": "msas/",                                   # optional: the run's .aligned.pqt directory (every route folds an item with msa_directory=<it> only when
                                                          # a chain's .aligned.pqt is there, else None — chai-lab's default; absent: None for every item)
     "items": [{"id": "1BRS_1to1", "fasta": "1BRS_1to1.fasta", "seeds": [0, 1, 2]}]}

``id`` defaults to the FASTA stem; ``seeds`` is optional (settings.seeds_for). Also accepted: the kit's own ``tests/public_inputs/PACK.json``
(``entries`` with ``name`` / ``fasta``), a bare JSON list of items, and a ``.fasta`` path (one item). Relative paths resolve against the items
file. The item key — the output directory name — is the kit's own (``chai_proto.fasta_spec(...)["key"]``, the FASTA stem): the package calls
that function, it does not restate it; two items with one stem are refused by name.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Item:
    id: str
    fasta: str                                  # absolute path
    seeds: Optional[List[int]] = None
    key: Optional[str] = None                   # filled by resolve_keys (the kit's rule)

    @property
    def uid(self) -> str:
        """The kit driver's input spec for this item."""
        return f"fasta:{self.fasta}"


@dataclass
class Items:
    items: List[Item]
    msa_dir: Optional[str] = None
    source: Optional[str] = None


def _abs(base: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def load(path: str) -> Items:
    path = os.path.abspath(path)
    base = os.path.dirname(path)
    if path.lower().endswith((".fasta", ".fa")):
        return Items([Item(id=os.path.splitext(os.path.basename(path))[0], fasta=path)], source=path)
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    msa_dir = None
    if isinstance(doc, dict):
        msa_dir = doc.get("msa_dir")
        raw = doc.get("items", doc.get("entries"))
        if raw is None:
            raise ValueError(f"{path}: no 'items' (or 'entries') list")
    elif isinstance(doc, list):
        raw = doc
    else:
        raise ValueError(f"{path}: expected a JSON object or list")
    items = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict) or "fasta" not in r:
            raise ValueError(f"{path}: item {i} has no 'fasta'")
        fasta = _abs(base, r["fasta"])
        if not os.path.isfile(fasta):
            raise FileNotFoundError(f"{path}: item {i}: FASTA not found: {fasta}")
        ident = r.get("id") or r.get("name") or os.path.splitext(os.path.basename(fasta))[0]
        seeds = r.get("seeds")
        items.append(Item(id=str(ident), fasta=fasta, seeds=([int(s) for s in seeds] if seeds else None)))
    if not items:
        raise ValueError(f"{path}: no items")
    return Items(items, msa_dir=(_abs(base, msa_dir) if msa_dir else None), source=path)


def resolve_keys(its: Items, chai_proto) -> Items:
    """Fill ``key`` with the kit's own ``fasta_spec`` (the module object is passed in: only the pred process imports the kit). Refuses two
    items that would share an output directory."""
    seen = {}
    for it in its.items:
        it.key = chai_proto.fasta_spec(it.fasta)["key"]
        if it.key in seen:
            raise ValueError(f"items {seen[it.key]!r} and {it.id!r} share the kit key {it.key!r} (the same FASTA stem: one output directory)")
        seen[it.key] = it.id
    return its


def uids_arg(items: List[Item]) -> str:
    """The driver's first positional argument: comma-joined specs (the kit driver's own ``uids`` grammar, ``chai_worker.py`` argparse)."""
    return ",".join(it.uid for it in items)


def to_plan(its: Items, items: List[Item], seeds_by_item: dict, st: dict, det_level: int, msa_dir: Optional[str]) -> dict:
    """The stock caller's input (one process per invocation, every item): everything it needs without importing the kit — the fold
    keywords (settings.from_args ``fold``), the knobs given a non-default value (``non_default``: the pass-through gate's allowance), ``device``
    (None = stock's default), det, MSAs, items (``seeds`` None = no seed named: stock's unseeded call)."""
    return {"schema": "chai1_opt.plan/2", "fold": dict(st["fold"]), "non_default": dict(st.get("non_default") or {}), "device": st.get("device"),
            "det": int(det_level), "msa_dir": msa_dir,
            "items": [{"id": it.id, "key": it.key, "fasta": it.fasta, "seeds": (list(seeds_by_item[it.key]) if seeds_by_item[it.key] else None)}
                      for it in items]}
