"""``CALIBY_FAST_POTTS_PARAMS``'s input gate: the switch word and the predicate deciding, per design call, whether the decoder's
sparse (B, N, K, C, C) couplings J are kept (the mechanism defined for untied positions) or upstream's dense fold runs. No torch
here: the lever file passes plain values in.

The gate keeps J sparse only when no structure of the batch carries a symmetry group (upstream's positional-constraint CSV column
``symmetry_pos``: positions tied to sample one residue type together). With symmetry-tied positions the call takes upstream's
``_fold_symmetry_pos`` — the dense N x N fold, stock arithmetic, the same outputs as ``off`` for that step — so the sparse-J step
and `CALIBY_X_SPARSE_EXACT` (which rebuilds the dense energy from a sparse J) are idle for that call. This is the lever's declared
gate (``registry.GATE_SYMMETRY_DENSE_J``), reported once per call as ``[caliby-opt] LEVER name=CALIBY_FAST_POTTS_PARAMS
state=skipped reason=symmetry_dense_J impl=atom_mpnn_denoiser origin=kit structures=<with groups>/<batch> groups=<n> J=dense
arithmetic=stock [idle=CALIBY_X_SPARSE_EXACT]`` and recorded (``report.note_gate``); the run stays the mode's run (exit code
unchanged) — a gate is not a fallback.
"""
import os
from typing import Sequence

from .registry import GATE_SYMMETRY_DENSE_J

SWITCH = "CALIBY_FAST_POTTS_PARAMS"
SPARSE_EXACT_SWITCH = "CALIBY_X_SPARSE_EXACT"


def group_counts(symmetry_pos: Sequence) -> list:
    """The number of symmetry groups per structure of the batch (upstream's ``symmetry_pos``: per structure, a list of groups of tied positions)."""
    return [len(groups or ()) for groups in symmetry_pos]


def keep_sparse(level: int, symmetry_pos: Sequence) -> bool:
    """True when level 2 keeps the sparse couplings for this call: ``level >= 2`` and no structure of the batch has a symmetry group. False =
    the caller runs upstream's dense fold (level 0 / 1 always; level 2 on a batch with symmetry groups, after saying the gate line once)."""
    if level < 2:
        return False
    counts = group_counts(symmetry_pos)
    if not any(counts):
        return True
    from . import report                                               # the core-backed report module: imported where a line is said, not at lever-module load
    evidence = {"structures": f"{sum(1 for n in counts if n)}/{len(counts)}", "groups": sum(counts), "J": "dense", "arithmetic": "stock"}
    if (os.environ.get(SPARSE_EXACT_SWITCH) or "0").strip() not in ("", "0"):
        evidence["idle"] = SPARSE_EXACT_SWITCH                         # the exact-sparse-energy lever has no sparse J to act on in this call
    report.note_gate(SWITCH, GATE_SYMMETRY_DENSE_J, **evidence)
    return False
