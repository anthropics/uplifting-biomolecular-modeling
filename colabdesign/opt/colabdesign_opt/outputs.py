"""Outputs of a design read back: the file set (design.pdb / design.fasta / trajectory.jsonl) with sha256 and byte counts, the names
missing from it, the trajectory's rows."""
from __future__ import annotations

import json
import os
from typing import Dict, List

from .names import OUTPUTS, TRAJECTORY, sha256_file




def files(out_dir: str, names=OUTPUTS) -> Dict[str, dict]:
    out = {}
    for n in names:
        p = os.path.join(out_dir, n)
        out[n] = {"present": os.path.isfile(p), "sha256": sha256_file(p) if os.path.isfile(p) else None,
                  "bytes": os.path.getsize(p) if os.path.isfile(p) else None}
    return out


def trajectory_rows(out_dir: str) -> List[dict]:
    p = os.path.join(out_dir, TRAJECTORY)
    with open(p, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def complete(fs: Dict[str, dict], names=OUTPUTS) -> List[str]:
    """The output names missing from the directory ([] when the set is complete)."""
    return [n for n in names if not fs.get(n, {}).get("present")]
