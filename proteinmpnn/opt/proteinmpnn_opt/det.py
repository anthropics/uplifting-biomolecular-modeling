"""The deterministic recipe: ProteinMPNN's stock run is bit-reproducible from the command line alone, so the recipe is a set of
argument values, not a set of library switches — the stock command line's own options, given per pass and handed to both routes alike
(settings.py); nothing is exported to the environment. What mode exact cannot serve of them is refused by name before any process (settings.worker_refuses).
The command line's ``--det 0|1`` is therefore inert: accepted so the family's identity spelling runs unchanged, it selects nothing (both routes are
deterministic by construction at the pass's own ``--seed``; ``--seed 0`` is upstream's random seed per process on either value).

The recipe (protein_mpnn_run.py): an explicit non-zero ``--seed`` (0 = a random seed per process, :430, on both routes alike — the worker
draws it as upstream draws it and every .fa header carries it; two runs agree only at one given seed; the seed is applied to torch, random and numpy
at :29-31) and ``--backbone_noise 0.0`` (the only value the worker reproduces; another is refused by name). Every other option of the design pass —
the sequence counts, the temperatures, the residue dictionaries — is served as given. The stock CLI's RNG stream runs across the whole jsonl in
dataset order in one process, every (temperature, batch) round of each backbone in turn; the worker replays that stream (``--mode stream``), so
equality with stock is defined at matched (seed, backbone, temperature, sample number) with the same parsed.jsonl and the same chain assignment on
both arms (the caller's --chain_id_jsonl verbatim, or none on either: upstream's default; for one PDB file the assignment upstream builds for it) —
inputs.py hands the same to either arm.
"""
from __future__ import annotations

from typing import List

from opt_core import det as core_det

ENV_RECIPE = core_det.PRODUCTION            # opt_core.det.Recipe level 0 — no env, unset or pythonpath entry: this engine's recipe is argument values only

RECIPE: List[str] = ["explicit non-zero --seed", "--backbone_noise 0.0",
                     "one parsed.jsonl + one chain assignment (the caller's --chain_id_jsonl, or none) shared by both arms (dataset order = RNG order)"]


def recipe() -> List[str]:
    return list(RECIPE)


def describe() -> str:
    """``det=0 env=none unset=none pythonpath=0 (production numerics)`` — ENV_RECIPE in the core's one-line form (opt_core.det.describe)."""
    return core_det.describe(ENV_RECIPE)

