"""Inputs: the design case — a target PDB file, its chains, the binder length — and what the package can say about it before the arm runs.

The token count of the case is the residues ColabDesign keeps for the target chains plus the binder length (`prep.py:221-226`:
`_target_len = residue_index.shape[0]` after `prep_pdb(pdb_filename, chain=chains)`, `_lengths = [_target_len, binder_len]`). The
package counts CA atoms per chain from the file (ATOM records, altloc blank or A) as the estimate for the RUN line; the recorded value
is what the script reads back from the model (`sum(af._lengths)`, the run record's `tokens`).
"""
from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional


class Target(NamedTuple):
    path: str
    chains: List[str]
    residues: Dict[str, int]                               # per chain: residues with a CA atom
    sha256: str

    @property
    def target_res(self) -> int:
        return sum(self.residues.values())


from .names import sha256_file


def parse_chains(chains: str) -> List[str]:
    out = [c.strip() for c in str(chains).replace("+", ",").split(",") if c.strip()]
    if not out:
        raise ValueError(f"no chain in {chains!r}")
    for c in out:
        if len(c) != 1:
            raise ValueError(f"chain ids are single characters; got {c!r}")
    return out


def ca_residues(path: str, chains: List[str]) -> Dict[str, int]:
    """Residues with a CA atom per requested chain (PDB ATOM records; altloc blank or 'A'; keyed by chain, resSeq, iCode)."""
    seen = {c: set() for c in chains}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA" or line[16] not in (" ", "A"):
                continue
            c = line[21]
            if c in seen:
                seen[c].add(line[22:27])
    return {c: len(seen[c]) for c in chains}


def target(path: str, chains: str) -> Target:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"target PDB not found: {path}")
    cl = parse_chains(chains)
    res = ca_residues(path, cl)
    empty = [c for c in cl if res[c] == 0]
    if empty:
        raise ValueError(f"chain(s) {','.join(empty)} have no CA atom in {path}")
    return Target(os.path.abspath(path), cl, res, sha256_file(path))


def case(t: Target, binder_len: int, seed: int, mode: str, iterations: dict, hotspot: Optional[str] = None) -> dict:
    """The case record: one trajectory. `iterations` = the settings file's four phase counts (settings.iteration_counts): `steps` = soft +
    temp + hard graded steps at most (BindCraft's gates may stop earlier), `greedy_rounds` the forward-only rounds; `hotspot` = BindCraft's
    target_hotspot_residues string or None."""
    if int(binder_len) <= 0:
        raise ValueError(f"--binder-len must be positive; got {binder_len}")
    it = dict(iterations)
    return {"mode": mode, "target": t.path, "target_sha256": t.sha256, "chains": ",".join(t.chains), "residues_per_chain": dict(t.residues),
            "target_res": t.target_res, "binder_len": int(binder_len), "tokens": t.target_res + int(binder_len), "hotspot": hotspot or None,
            "seed": int(seed),
            "steps": int(it["soft"]) + int(it["temp"]) + int(it["hard"]), "greedy_rounds": int(it["greedy"]), "iterations": it}
