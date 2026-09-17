"""The outputs of a design pass, as upstream writes them: `<prefix>_<i>.pdb`, `<prefix>_<i>.trb`, and under `inference.write_trajectory=True`
`<dir>/traj/<name>_<i>_Xt-1_traj.pdb` / `_pX0_traj.pdb` (stock/src/scripts/run_inference.py:52-56,143-176; the resident drivers write the same
set, drivers/rfd_bench.py:192-200). `list_outputs` counts, per case, which of the requested design indices are on disk — the manifest's
`outputs` block and the exit rule's `n_pdb` (report.verdict: outputs short of the request are exit 1).
"""
from __future__ import annotations

import os
from typing import Dict, List


def case_prefix(case: dict, out_dir: str) -> str:
    """The output prefix of a case: its own `prefix` (upstream's inference.output_prefix, the command line's target) or `<out_dir>/<name>/des` (a warm-up row)."""
    p = case.get("prefix")
    return os.path.abspath(p) if p else os.path.join(os.path.abspath(out_dir), case["name"], "des")


def case_indices(case: dict) -> List[int]:
    s = int(case.get("startnum", 0))
    return list(range(s, s + int(case["num_designs"])))


def list_case(case: dict, out_dir: str) -> dict:
    """`{prefix, indices, designs: {i: {pdb, trb, traj: [names]}}, n_pdb, n_trb, n_traj}` for the requested indices of one case."""
    prefix = case_prefix(case, out_dir)
    d, base = os.path.dirname(prefix), os.path.basename(prefix)
    traj_dir = os.path.join(d, "traj")
    designs: Dict[str, dict] = {}
    for i in case_indices(case):
        rec = {"pdb": os.path.isfile(f"{prefix}_{i}.pdb"), "trb": os.path.isfile(f"{prefix}_{i}.trb"),
               "traj": sorted(f for f in (os.listdir(traj_dir) if os.path.isdir(traj_dir) else []) if f.startswith(f"{base}_{i}_") and f.endswith("_traj.pdb"))}
        designs[str(i)] = rec
    return {"prefix": prefix, "indices": case_indices(case), "designs": designs,
            "n_pdb": sum(1 for r in designs.values() if r["pdb"]), "n_trb": sum(1 for r in designs.values() if r["trb"]),
            "n_traj": sum(len(r["traj"]) for r in designs.values())}


def list_outputs(cases: List[dict], out_dir: str) -> dict:
    """The requested designs on disk, per case and in total (`n_pdb`, `n_trb`, `n_traj`, `expected`)."""
    per = {c["name"]: list_case(c, out_dir) for c in cases}
    return {"n_pdb": sum(v["n_pdb"] for v in per.values()), "n_trb": sum(v["n_trb"] for v in per.values()), "n_traj": sum(v["n_traj"] for v in per.values()),
            "expected": sum(len(v["indices"]) for v in per.values()), "cases": per}
