"""The outputs of a design pass and their census. Two layouts: ``--out DIR`` writes ``<DIR>/seqs/<stem>.fasta`` per structure (a directory
or one file in); upstream's ``--outpath FILE`` writes that one FASTA for one input structure (its default ``output/sampled_seqs.fasta`` applies
when one file is given with neither flag). Records are upstream's own format (``>sampled_seq_<i>`` then the sequence). The run record
(``opt_manifest.json``, ``stock_env_proof.json``, ``timing.jsonl``) goes to the output directory: ``--out`` itself, or the directory holding
``--outpath``.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .batched import DEFAULT_OUTPATH

SEQS_DIR = "seqs"


def seqs_dir(out_dir: str) -> str:
    return os.path.join(os.path.abspath(out_dir), SEQS_DIR)


def fasta_path(out_dir: str, stem: str) -> str:
    return os.path.join(seqs_dir(out_dir), stem + ".fasta")


def fasta_records(path: str) -> List[str]:
    """The sequences of a FASTA file in order (header lines dropped; a record's lines joined)."""
    seqs: List[str] = []
    cur = None
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.startswith(">"):
                if cur is not None:
                    seqs.append(cur)
                cur = ""
            elif cur is not None:
                cur += ln.strip()
    if cur is not None:
        seqs.append(cur)
    return seqs


class OutputError(ValueError):
    """--out and --outpath both given, --outpath for more than one structure, or a directory input without --out."""


def plan(out, outpath, files: List[str], stems: List[str]) -> Tuple[str, Dict[str, str]]:
    """``(out_dir, {stem: fasta path})`` — absolute paths — for the parsed ``--out`` / ``--outpath`` values over the resolved inputs."""
    if out and outpath:
        raise OutputError(f"--out {out} and --outpath {outpath} both given: --out DIR (one FASTA per structure under DIR/{SEQS_DIR}/) or --outpath FILE (one structure), not both")
    if out:
        return os.path.abspath(out), {s: fasta_path(out, s) for s in stems}
    if len(files) != 1:
        raise OutputError(f"{len(files)} structures and no --out: a directory of structures needs --out DIR (--outpath names one FASTA for one structure)")
    path = os.path.abspath(outpath or DEFAULT_OUTPATH)
    return os.path.dirname(path), {stems[0]: path}


def census_paths(paths: Dict[str, str], num_samples: int) -> Dict[str, object]:
    """``{n_items, n_items_complete, items_complete, item_failed: {stem: reason}, n_records, incomplete: "k/n" | None}`` over ``{stem: fasta path}``."""
    failed: Dict[str, str] = {}
    n_records = 0
    complete = []
    for s, p in paths.items():
        if not os.path.isfile(p):
            failed[s] = "missing"
            continue
        n = len(fasta_records(p))
        n_records += n
        if n != num_samples:
            failed[s] = f"{n}/{num_samples} records"
        else:
            complete.append(s)
    n_items = len(paths)
    return {"n_items": n_items, "n_items_complete": len(complete), "items_complete": len(complete) == n_items, "item_failed": failed,
            "n_records": n_records, "incomplete": (None if len(complete) == n_items else f"{n_items - len(complete)}/{n_items}")}


def census(out_dir: str, stems: List[str], num_samples: int) -> Dict[str, object]:
    """The census of the ``--out DIR`` layout: ``<out>/seqs/<stem>.fasta`` per stem."""
    return census_paths({s: fasta_path(out_dir, s) for s in stems}, num_samples)
