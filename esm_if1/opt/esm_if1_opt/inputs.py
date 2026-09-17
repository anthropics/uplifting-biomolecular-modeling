"""The structure inputs of a design pass: upstream's positional ``pdbfile`` or ``--input`` (exactly one of them) names one ``.pdb`` / ``.cif``
file (what upstream's loaders open) or — ``--input`` only — a directory whose every ``*.pdb`` / ``*.cif`` (sorted by name = design order) is
one backbone. File stems name the outputs of a directory run (``<out>/seqs/<stem>.fasta``), so they must be unique.
"""
from __future__ import annotations

import glob
import os
from typing import List, Tuple

SUFFIXES = (".pdb", ".cif")


class InputError(ValueError):
    """No structure file named, both spellings given, or non-unique stems."""


def input_arg(ns) -> str:
    """The one input path of the parsed arguments: the positional ``pdbfile`` (upstream's spelling) or ``--input`` — exactly one."""
    pos, opt = getattr(ns, "pdbfile", None), getattr(ns, "input", None)
    if pos and opt:
        raise InputError(f"two inputs given (pdbfile {pos} and --input {opt}): name the structure file or directory once")
    if not (pos or opt):
        raise InputError("no input: give the structure file (pdbfile, as upstream's sample_sequences.py takes it) or --input <dir|file>")
    return pos or opt


def resolve(input_path: str) -> Tuple[List[str], List[str]]:
    """``(files, stems)`` in design order."""
    p = os.path.abspath(input_path)
    if os.path.isdir(p):
        files = sorted(f for s in SUFFIXES for f in glob.glob(os.path.join(p, "*" + s)))
        if not files:
            raise InputError(f"--input {input_path}: no *.pdb / *.cif file in the directory")
    elif os.path.isfile(p):
        files = [p]
    else:
        raise InputError(f"--input {input_path}: no such file or directory")
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        raise InputError(f"--input {input_path}: structure file stems are not unique ({', '.join(dup[:5])}): one FASTA is written per stem")
    return files, stems
