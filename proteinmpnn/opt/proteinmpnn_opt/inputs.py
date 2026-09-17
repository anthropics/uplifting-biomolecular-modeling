"""Inputs of one design pass, built once per run and handed to whichever arm runs.

``--input`` is a directory of PDB files, an existing ``parsed.jsonl`` (protein_mpnn_run.py's ``--jsonl_path``) or one PDB file
(its ``--pdb_path``; ``classify_base``). A directory is parsed by the parser of the route
— the stock helper ``$MPNN_DIR/helper_scripts/parse_multiple_chains.py`` (mode ``off``, in the clean subprocess) or the kit's
read-once ``kit/fast_parse.py`` (mode ``exact``; byte-identical output is the kit's contract) — into ``<out>/parsed.jsonl``. One PDB file is
protein_mpnn_run.py's ``--pdb_path`` on the stock route (its own parse_PDB; every chain designed, or ``--pdb_path_chains``; it reads no
``--chain_id_jsonl`` then) and, on the kit route, the kit parser's one entry from a staged directory holding that file with the chain assignment
protein_mpnn_run.py builds for it (``write_pdb_assignment``). Chains:
a caller's ``--chain_id_jsonl`` (upstream's format, ``{name: [[designed chains], [fixed chains]]}``) is handed to the design process
unchanged; without one both routes run upstream's default — ``protein_mpnn_run.py`` given no ``--chain_id_jsonl`` (``chain_id_dict`` None:
``protein_mpnn_utils.tied_featurize`` designs every chain of the entry and fixes none). The stock route then passes no chain flag at all; the
kit worker, whose ``--chain_id_jsonl`` is a required argument, is handed ``UNASSIGNED`` (``write_unassigned``): one JSON line ``null``, which
the worker loads as that same None and hands to the same ``tied_featurize`` — upstream's own default branch, no chain rule of this package's.
Order: a parsed.jsonl the routes make from a PDB directory keeps the parse helper's own order (the file system's glob order, the same on both routes
of one box: kit/fast_parse.py globs as the stock helper does); a parsed.jsonl the caller gives is read as given. A caller who wants one order on
every box parses once and passes the parsed.jsonl as ``--input``.
"""
from __future__ import annotations

import glob
import json
import os
from typing import List, Optional

PARSED = "parsed.jsonl"
UNASSIGNED = "unassigned.jsonl"                        # the kit worker's --chain_id_jsonl when the caller gives none (write_unassigned): staged, removed with the stage
PDB_ASSIGNED = "pdb_chains.jsonl"                      # the kit worker's --chain_id_jsonl for a single-PDB input (write_pdb_assignment): staged, removed with the stage


class InputError(ValueError):
    """An input that is not a PDB directory / file or a parsed.jsonl."""


def write_unassigned(path: str) -> str:
    """The kit worker's ``--chain_id_jsonl`` when the caller gives none: one JSON line ``null``. The worker reads line one with json.loads
    (as protein_mpnn_run.py reads its --chain_id_jsonl, :69-73) and hands the None to protein_mpnn_utils.tied_featurize — upstream's default
    when no --chain_id_jsonl is given: every chain of each entry designed, none fixed. Returns ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("null\n")
    return path


def pdb_chain_assignment(parsed_jsonl: str, designed: Optional[List[str]]) -> dict:
    """The chain assignment protein_mpnn_run.py builds for a ``--pdb_path`` input, from the parsed entry of that file: its statements — every chain
    letter of the entry (``seq_chain_<X>`` keys, in order), designed = ``--pdb_path_chains`` (``designed``) else all of them, fixed = the others —
    as ``{name: [designed, fixed]}`` (upstream's --chain_id_jsonl form; it reads no --chain_id_jsonl for a --pdb_path input)."""
    with open(parsed_jsonl, encoding="utf-8") as fh:
        entry = json.loads(fh.readline())
    all_chain_list = [item[-1:] for item in list(entry) if item[:9] == "seq_chain"]
    designed_chain_list = [str(item) for item in designed] if designed else all_chain_list
    fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
    return {entry["name"]: [designed_chain_list, fixed_chain_list]}


def write_pdb_assignment(parsed_jsonl: str, designed: Optional[List[str]], path: str) -> str:
    """Write ``pdb_chain_assignment`` as the kit worker's ``--chain_id_jsonl`` (one JSON line). Returns ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(pdb_chain_assignment(parsed_jsonl, designed)) + "\n")
    return path


def pdb_files(directory: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(directory, "*.pdb")))
    if not files:
        raise InputError(f"no *.pdb file in {directory}")
    return files


def classify_base(path: str) -> str:
    """'dir' | 'parsed' | 'pdb' for a base-variant input: a directory of PDB files (parsed first), a parsed jsonl (protein_mpnn_run.py's
    ``--jsonl_path``) or one PDB file (its ``--pdb_path``)."""
    if os.path.isdir(path):
        return "dir"
    if os.path.isfile(path) and path.endswith((".jsonl", ".json")):
        return "parsed"
    if os.path.isfile(path) and path.endswith(".pdb"):
        return "pdb"
    raise InputError(f"--input {path}: not a directory of PDB files, a parsed.jsonl nor a .pdb file")


def parse_count(parsed_jsonl: str) -> int:
    with open(parsed_jsonl, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
