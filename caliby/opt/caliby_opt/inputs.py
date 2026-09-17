"""Input resolution for ``design`` / ``warm``: structure files for the design writer.

``--input`` accepts, in any mix: structure files (.pdb, .cif, .pdb.gz, .cif.gz — what caliby.clean_pdbs reads), directories (their
structure files, sorted), and list files (.txt/.list: one path per line, relative to the list file). The writer designs them in
sorted order (xcaliby_design.py:56 ``sorted(a.inputs)``). ``public_inputs()`` are the kit's seven public example structures
(the upstream repository's examples, carried in opt/forward/xattempt_addon/tests/public_inputs/), the inputs of ``warm``.
"""
from __future__ import annotations

import glob
import os
from typing import List

from . import stack

STRUCTURE_SUFFIXES = (".pdb", ".cif", ".pdb.gz", ".cif.gz", ".mmcif")


def is_structure(path: str) -> bool:
    low = path.lower()
    return any(low.endswith(s) for s in STRUCTURE_SUFFIXES)


def resolve(items: List[str]) -> List[str]:
    out: List[str] = []
    for it in items:
        if os.path.isdir(it):
            files = sorted(p for p in glob.glob(os.path.join(it, "*")) if is_structure(p))
            if not files:
                raise FileNotFoundError(f"no structure files under {it}")
            out += files
        elif is_structure(it):
            if not os.path.isfile(it):
                raise FileNotFoundError(it)
            out.append(it)
        elif it.lower().endswith((".txt", ".list")):
            base = os.path.dirname(os.path.abspath(it))
            with open(it, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    p = line if os.path.isabs(line) else os.path.join(base, line)
                    if not os.path.isfile(p):
                        raise FileNotFoundError(f"{it}: {line}")
                    out.append(p)
        else:
            raise ValueError(f"not a structure file, directory or list: {it}")
    seen, uniq = set(), []
    for p in out:
        a = os.path.abspath(p)
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return sorted(uniq)


def public_inputs_dir() -> str:
    return os.path.join(stack.kit_dir(stack.KIT_ADDON), "tests", "public_inputs")


def public_inputs() -> List[str]:
    return resolve([public_inputs_dir()])


def smallest_public_input() -> str:
    files = public_inputs()
    return min(files, key=os.path.getsize)
