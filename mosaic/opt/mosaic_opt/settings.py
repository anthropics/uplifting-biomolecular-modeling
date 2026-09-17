"""Settings: the recipe the driver runs (its own constants and argparse defaults), read from the kit file, never transcribed.

The driver hard-codes the notebook recipe (`tools/public_design_run.py`: `Boltz2Loss(recycling_steps=1, sampling_steps=25,
deterministic=True)`, loss `2·BinderTargetContact + WithinBinderContact + 5·InverseFoldingSequenceRecovery(ProteinMPNN v_48_020, T=0.01)`,
`simplex_APGM` 75 steps @0.1 then 50 steps @0.5 (scale 1.5), the Boltz-2 refold of the final PSSM — the literals of tools/recipe.py, the
one recipe module) and exposes `--binder-length`, `--target-copies`, `--target-fasta`, `--first-record`, `--msa`, `--epitope`, `--steps1`, `--steps2`, `--seed`, and the lever flags `--weights` / `--features-in` / `--features-sha` / `--levers` (its `add_argument` calls). ``driver_defaults()`` reads those defaults from the file's `add_argument` calls (ast, no import); ``effective()`` passes the
driver's own knobs through under the driver's names and defaults (`--steps1`, `--steps2`), nothing else.
The target, its MSA and the epitope are inputs, not settings: `--target-fasta` (default: the public target barstar 1BRS:D; one record, or
record 1 with `--first-record`), `--msa` (the target chain's precomputed alignment = upstream's `TargetChain(use_msa=True)` with upstream's MSA-server
fetch done ahead of time; absent = single-sequence) and `--epitope` (target residue positions the contact loss is restricted to; absent = the
notebook's unrestricted term) — the same in every mode. Upstream's notebook value the driver does not default to — `binder_length = 75` (`examples/boltz_notebook.py:107`; the
driver's 80) — is stated in `STOCK.md`.
"""
import ast
import os
from typing import Dict, Optional

from .modes import DRIVER_RELPATH

DRIVER_FLAGS = ("--seed", "--binder-length", "--target-copies", "--weights", "--features-in", "--features-sha", "--features-out", "--steps1", "--steps2", "--target-fasta", "--first-record", "--msa", "--epitope")


def driver_defaults(kit_home: str) -> Dict[str, object]:
    """`{flag: default}` for every `p.add_argument(...)` of the driver (ast; no import of the driver, which runs on import)."""
    path = os.path.join(kit_home, DRIVER_RELPATH)
    tree = ast.parse(open(path, "r", encoding="utf-8").read(), filename=path)
    out: Dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args:
            flag = ast.literal_eval(node.args[0])
            default = None
            for kw in node.keywords:
                if kw.arg == "default":
                    default = ast.literal_eval(kw.value)
            out[flag] = default
    missing = [f for f in DRIVER_FLAGS if f not in out]
    if missing:
        raise ValueError(f"{path}: expected driver flags missing: {missing}")
    return out


def effective(kit_home: str, steps1: Optional[int] = None, steps2: Optional[int] = None) -> dict:
    """The driver settings a run uses: the driver's own defaults, then the caller's flags; `{"--steps1", "--steps2"}`."""
    d = driver_defaults(kit_home)
    eff = {"--steps1": d["--steps1"], "--steps2": d["--steps2"]}
    if steps1 is not None:
        eff["--steps1"] = int(steps1)
    if steps2 is not None:
        eff["--steps2"] = int(steps2)
    return eff


def flags(eff: dict) -> list:
    return ["--steps1", str(eff["--steps1"]), "--steps2", str(eff["--steps2"])]
