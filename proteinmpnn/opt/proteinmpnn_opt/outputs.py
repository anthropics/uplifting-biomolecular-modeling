"""Outputs of one design pass: the listing (every output file, by relative path) and the per-directory counts.

protein_mpnn_run.py writes, under --out_folder: ``seqs/<name>.fa`` (the header carries score, global_score, the chain lists,
git_hash and seed: :384), ``scores/<name>.npz`` (score, global_score) and ``probs/<name>.npz`` (probs, log_probs, S, mask, chain_order, ...)
— the last two only when the write gates --save_score 1 / --save_probs 1 are on; ``score_only/``, ``conditional_probs_only/``,
``unconditional_probs_only/`` for the scoring tasks.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

BASE_DIRS = ("seqs", "scores", "probs", "score_only", "conditional_probs_only", "unconditional_probs_only")


def listing(out_dir: str, variant: str) -> List[str]:
    """The relative path of every output file of the pass (the subdirectories of the variant's writer), sorted."""
    out: List[str] = []
    for d in BASE_DIRS:
        for p in sorted(glob.glob(os.path.join(out_dir, d, "*"))):
            if os.path.isfile(p):
                out.append(os.path.relpath(p, out_dir))
    return out


def counts(listing_: List[str]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for rel in listing_:
        c[rel.split(os.sep)[0]] = c.get(rel.split(os.sep)[0], 0) + 1
    return c
