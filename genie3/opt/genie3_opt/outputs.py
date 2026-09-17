"""The design outputs of a pass.

``list_outputs(out_dir, problems, since, names)``: the ``<problem>/pdbs/<name>.pdb`` files upstream's writer produced (src/genie3/generation/runner/
postprocess.py:150-163; the driver uses the same writer) IN THIS PASS — written at or after ``since`` and, when the request's design names are
known, among those names — by name: the manifest's ``outputs`` record and the pass's completeness count. Files an earlier pass left in the same
directory are not this pass's. The stock line's ``generation_stats/`` file and either line's logs / timings are not design outputs.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional


UNCONDITIONAL = "unconditional"        # the pseudo-problem of an unconditional request: its designs land in <out>/pdbs/ (postprocess.py:69-82), not under a problem directory


def pdb_dir(out_dir: str, problem: str) -> str:
    """Where upstream's writer puts a problem's designs: <out>/<problem>/pdbs/ (target / motif, postprocess.py:108,150), <out>/pdbs/ (unconditional, :69)."""
    return os.path.join(out_dir, "pdbs") if problem == UNCONDITIONAL else os.path.join(out_dir, problem, "pdbs")


def problem_dirs(out_dir: str, problems: Optional[List[str]] = None) -> List[str]:
    if problems:
        return [p for p in problems if os.path.isdir(pdb_dir(out_dir, p))]
    if not os.path.isdir(out_dir):
        return []
    found = sorted(d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d, "pdbs")))
    if os.path.isdir(os.path.join(out_dir, "pdbs")):
        found.append(UNCONDITIONAL)
    return found


def list_outputs(out_dir: str, problems: Optional[List[str]] = None, since: Optional[float] = None, names: Optional[Dict[str, List[str]]] = None) -> dict:
    """{"n_pdb", "problems": {problem: [design names]}} over the pass's problem directories (the named ones, else every one present): the PDB files
    written at or after ``since`` (None = any time) whose design name is among ``names[problem]`` (None = any name)."""
    out: Dict[str, object] = {"n_pdb": 0, "problems": {}}
    for p in problem_dirs(out_dir, problems):
        d = pdb_dir(out_dir, p)
        want = set(names[p]) if names and p in names else None
        found = []
        for f in sorted(os.listdir(d)):
            if not f.endswith(".pdb"):
                continue
            if want is not None and f[:-4] not in want:
                continue
            if since is not None and os.path.getmtime(os.path.join(d, f)) + 1.0 < since:
                continue
            found.append(f[:-4])
        out["problems"][p] = found
        out["n_pdb"] += len(found)
    return out
