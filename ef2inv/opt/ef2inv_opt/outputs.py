"""The output set of a design and its readers.

  design.pdb / design.cif   the hallucinated complex as stock materializes it: the hero-critic refold by the model the gradient flowed
                            through (ESMFold2-Experimental-Fast, the first hero critic, stock file l.1277-1282; num_loops=3, 200 sampling
                            steps, confidence on, l.1168-1170). Stock writes no structure of its own and materializes no design-time
                            structure (the design step's distogram is consumed by the losses, l.1078-1094); no extra forward is added.
  critic_<name>.pdb         the same for every hero critic (four)
  critics.json              per critic: iptm, final_loss, iptm_proxy scores, designed_sequence
  design.fasta              the final hard sequence: target and binder records, then the binder alone
  trajectory.jsonl          one object per step: the cookbook's step_losses (its keys, its order: the loss terms, plm_loss, total_loss,
                            time, peak_allocated_gib, peak_reserved_gib) plus step, temperature, replicate
  steps.jsonl               one object per fold call: k (call index), kind (design | critic), wall_s (CUDA-synchronised), tokens, confidence
  run.json                  the record: settings, deviations, the case, the environment proof, timings, peak memory
  run.log                   the process's log (every logging record + the kit's own lines)
  opt_manifest.json         written by the launcher: shas, the activation report, refusals, the mode's evidence
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List

NAMES = ("design.pdb", "design.cif", "critics.json", "design.fasta", "trajectory.jsonl", "steps.jsonl", "run.json", "run.log", "opt_manifest.json")

def _num(v):
    try:
        import torch
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()
            return v.item() if v.numel() == 1 else v.tolist()
    except ImportError:
        pass
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    return v


def write_trajectory(path: str, trajectory: Dict[int, dict], temperatures: Dict[int, float], replicates: Dict[int, str]) -> int:
    n = 0
    with open(path, "w") as f:
        for step in sorted(trajectory):
            row: Dict[str, Any] = {"step": step}
            row.update({k: _num(v) for k, v in trajectory[step].items()})
            row["temperature"] = temperatures.get(step)
            row["replicate"] = replicates.get(step)
            f.write(json.dumps(row) + "\n"); n += 1
    return n


def write_steps(path: str, calls: Iterable[dict]) -> int:
    n = 0
    with open(path, "w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n"); n += 1
    return n


def write_fasta(path: str, target_name: str, target_seq: str, binder_seq: str) -> None:
    with open(path, "w") as f:
        f.write(f">{target_name}|target\n{target_seq}\n>{target_name}|binder\n{binder_seq}\n>binder\n{binder_seq}\n")


def write_complex(pdb_path: str, cif_path: str, complex_obj) -> Dict[str, Any]:
    """ProteinComplex.to_pdb / to_mmcif_string (esm/utils/structure/protein_complex.py l.361, l.1009)."""
    complex_obj.to_pdb(pdb_path)
    with open(cif_path, "w") as f:
        f.write(complex_obj.to_mmcif_string())
    return {"pdb": os.path.basename(pdb_path), "cif": os.path.basename(cif_path), "n_atoms": int(len(complex_obj.atom_array)) if hasattr(complex_obj, "atom_array") else None}


def write_critics(path: str, results: List[dict], design_dir: str) -> List[dict]:
    rows = []
    for i, r in enumerate(results):
        name = str(r.get("critic_name", f"critic{i}"))
        safe = name.replace("/", "_").replace("biohub_", "")
        row = {"index": i, "critic_name": name, "iptm": _num(r.get("iptm")), "final_loss": _num(r.get("final_loss")),
               "designed_sequence": r.get("designed_sequence"), "iptm_proxy": _num(r.get("iptm_proxy")),
               "distogram_iptm_proxy": _num(r.get("distogram_iptm_proxy")), "batch_idx": _num(r.get("batch_idx")),
               "seed": _num(r.get("seed")), "scores": _num(r.get("scores"))}
        c = r.get("complex")
        if c is not None:                                              # design(batch_size=B): one result per (critic, batch_idx); trajectory 0 keeps the plain name, trajectory k > 0 writes critic_<name>_b<k>.* (nothing overwritten)
            k = r.get("batch_idx")
            stem = f"critic_{safe}" if not k else f"critic_{safe}_b{int(k)}"
            row["files"] = write_complex(os.path.join(design_dir, f"{stem}.pdb"), os.path.join(design_dir, f"{stem}.cif"), c)
        rows.append(row)
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def listing(design_dir: str) -> Dict[str, Dict[str, Any]]:
    import hashlib
    out = {}
    for n in sorted(os.listdir(design_dir)):
        p = os.path.join(design_dir, n)
        if os.path.isfile(p):
            out[n] = {"sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "bytes": os.path.getsize(p)}
    return out


def complete(design_dir: str) -> List[str]:
    """The names of the output set missing from the directory (empty = complete)."""
    return [n for n in NAMES if not os.path.isfile(os.path.join(design_dir, n))]
