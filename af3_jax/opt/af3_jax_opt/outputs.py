"""The output file set of a pass: per prediction ``<out_dir>/<name>/seed-<s>_sample-<k>/`` the stock script writes ``*_model.cif``,
``*_confidences.json`` and ``*_summary_confidences.json``; ``count_predictions`` counts the seed-level models written, ``expected_predictions``
the number the command line asks for (the DONE line's ``predictions=<n>/<expected>``)."""
from __future__ import annotations

import fnmatch
import os
from typing import List, Optional

PATTERNS = ("*_model.cif", "*_confidences.json")                    # *confidences.json also matches *_summary_confidences.json


def output_files(out_dir: str) -> List[str]:
    """Relative paths of the prediction file set under out_dir (seed-level directories only), sorted."""
    found = []
    for d, _, fs in os.walk(out_dir):
        if "seed-" not in os.path.basename(d):
            continue
        for f in fs:
            if any(fnmatch.fnmatch(f, p) for p in PATTERNS):
                found.append(os.path.relpath(os.path.join(d, f), out_dir))
    return sorted(found)


def prediction_files(out_dir: str) -> dict:
    """{seed-level model.cif path relative to out_dir: mtime_ns} present now — snapshot before the model process starts, count what changed
    after: upstream accepts an --output_dir that already holds a finished job (it writes the new job into a timestamped sibling directory, never
    over the old one), so a pass counts only the models IT wrote — new paths, or paths rewritten since the snapshot."""
    return {rel: os.stat(os.path.join(out_dir, rel)).st_mtime_ns for rel in output_files(out_dir) if rel.endswith("_model.cif")}


def count_predictions(out_dir: str, before: Optional[dict] = None) -> int:
    """Seed-level model.cif count — one per (seed, diffusion sample); with ``before`` (prediction_files taken before the launch), only the
    models written since (absent from the snapshot, or rewritten after it)."""
    now = prediction_files(out_dir)
    if before is None:
        return len(now)
    return sum(1 for rel, mt in now.items() if rel not in before or mt > before[rel])


def num_seeds(argv: List[str]):
    """The run's ``--num_seeds`` (run_alphafold.py ``flags.DEFINE_integer('num_seeds', None, ...)``): when stated, the stock script expands
    EVERY fold input to that many seeds (folding_input.Input.with_multiple_seeds: the input must carry exactly one seed and the count must
    exceed 1 — otherwise upstream itself raises); None when absent (the inputs' own modelSeeds count). Read, never validated here."""
    from .settings import flag_int
    return flag_int(argv, "--num_seeds")


def expected_predictions(argv: List[str], n_seeds: int) -> int:
    """What count_predictions must find for a run: the inputs' seeds (inputs.seeds_in, which honours ``--num_seeds``) x the run's
    ``--num_diffusion_samples`` (the stock script writes one seed-level model.cif per sample) — the flag's value when the caller states it
    on the command line in either spelling (``--num_diffusion_samples=3`` / ``--num_diffusion_samples 3``), else the flag's own default read
    from the stock source (settings.upstream_defaults), never a guess. ``--num_recycles`` and ``--buckets`` do not change the count."""
    from .settings import flag_int, upstream_defaults
    v = flag_int(argv, "--num_diffusion_samples")
    if v is None:
        v = int(upstream_defaults()["num_diffusion_samples"])
    return n_seeds * v
