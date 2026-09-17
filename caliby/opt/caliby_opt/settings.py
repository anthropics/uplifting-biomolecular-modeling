"""Turns a lever set into the keyword arguments for the kit's design writer, and can describe those settings.
The writer is opt/forward/xattempt_addon/tests/xcaliby_design.py: clean_pdbs -> load_model -> [seed] ->
sample | generate_ensembles -> ensemble_sample, writing seq_des_outputs.csv, raw/samples, cleaned/*.cif, timing.json.
Nothing about that call sequence is re-implemented here.
Every knob is upstream's own keyword; each is forwarded only when the command line gives it, so an
unset knob keeps upstream's own default (caliby/api.py, caliby/configs/seq_des/inference.yaml,
caliby/configs/eval/sampling/*.yaml).
DEFAULT_MODEL_NAME is ``caliby``; DEFAULT_NUM_SEQS_PER_PDB is 1.
ENSEMBLE_VARIANT (ensemble32) adds the ensemble-only knobs: --num_samples_per_pdb, --pp_batch_size,
--sampling_yaml_path, --max_num_conformers, --include_primary_conformer, --use_primary_res_type.
--seed is otherwise unset, meaning upstream's own unseeded call; given, the writer calls Lightning's
``seed_everything`` before sampling and passes it to generate_ensembles(seed=).
DET_LEVELS' --det 1 requires --seed plus torch.backends.cudnn.deterministic=True / benchmark=False,
the same on every mode.
"""
from __future__ import annotations

from typing import Dict, List, Optional

DET_LEVELS = (0, 1)                                                    # --det: 0 = nothing applied; 1 = seed + cuDNN deterministic flags (upstream's scripts' recipe)
DEFAULT_MODEL_NAME = "caliby"                                          # caliby/api.py load_model(model_name="caliby"): the checkpoint a run reads when --model_name is not given
DEFAULT_NUM_SEQS_PER_PDB = 1                                           # caliby/api.py:100; inference.yaml:5 — the count ``incomplete`` and the DESIGNS line check against

# upstream's keyword -> (writer flag, kind); kind: value | flag (boolean spelled true|false) | words (several values). Order = the writer's argv order.
PASS_THROUGH: Dict[str, str] = {
    "model_name": "value", "device": "value", "sampling_cfg_path": "value",
    "num_seqs_per_pdb": "value", "batch_size": "value", "omit_aas": "value", "temperature": "value", "num_workers": "value",
    "verbose": "bool", "sampling_overrides": "words", "pos_constraint_csv": "value", "clean_workers": "value",
}
ENSEMBLE_ONLY: Dict[str, str] = {
    "num_samples_per_pdb": "value", "pp_batch_size": "value", "sampling_yaml_path": "value", "max_num_conformers": "value",
    "include_primary_conformer": "bool", "use_primary_res_type": "bool",
}
ENSEMBLE_VARIANT = "ensemble32"


def ensemble_only_given(knobs: dict) -> List[str]:
    """The ensemble-only flags a command gave (cli.usage_refusal names them on the single variant)."""
    return [f"--{k}" for k in ENSEMBLE_ONLY if knobs.get(k) is not None]


def _fmt(kind: str, v) -> List[str]:
    if kind == "bool":
        return ["true" if v else "false"]
    if kind == "words":
        return [str(w) for w in v]
    return [str(v)]


def writer_args(variant: str, inputs: List[str], out_dir: str, *, seed: Optional[int] = None, det: int = 0, **knobs) -> List[str]:
    """The argument list for xcaliby_design.py (after its path): the inputs, the output directory, the route (``--ensemble`` on the
    ensemble32 variant), ``--det``, ``--seed`` when given, and every upstream knob the command gave — nothing else, so a knob not given
    runs at upstream's default."""
    unknown = sorted(set(knobs) - set(PASS_THROUGH) - set(ENSEMBLE_ONLY))
    if unknown:
        raise ValueError(f"not design knobs: {unknown}")
    args = ["--inputs", *inputs, "--out_dir", out_dir, "--det", str(int(det or 0))]
    if seed is not None:
        args += ["--seed", str(int(seed))]
    if variant == ENSEMBLE_VARIANT:
        args += ["--ensemble"]
    table = {**PASS_THROUGH, **(ENSEMBLE_ONLY if variant == ENSEMBLE_VARIANT else {})}
    for k, kind in table.items():
        v = knobs.get(k)
        if v is None or (kind == "words" and not v):
            continue
        args += [f"--{k}", *_fmt(kind, v)]
    return args


def describe(variant: str, *, seed: Optional[int] = None, det: int = 0, **knobs) -> dict:
    """The manifest's ``settings`` block: the route, the seed / det recipe, the knobs the command gave (``given``) and the two values the
    package itself reads — the checkpoint the weights step checks and the sequences-per-structure count the DESIGNS line counts against —
    at upstream's default when not given."""
    given = {k: knobs[k] for k in list(PASS_THROUGH) + list(ENSEMBLE_ONLY) if knobs.get(k) is not None}
    return {"variant": variant, "ensemble": variant == ENSEMBLE_VARIANT, "seed": seed, "det": int(det or 0),
            "model_name": knobs.get("model_name") or DEFAULT_MODEL_NAME,
            "num_seqs_per_pdb": int(knobs.get("num_seqs_per_pdb") or DEFAULT_NUM_SEQS_PER_PDB),
            "given": given}
