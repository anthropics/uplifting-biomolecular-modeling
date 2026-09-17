"""The deterministic recipe — the stock-side switches of the KERNELS-OFF stock (`pred --mode off --det 1`: the anchor of the kits' own
kernels-off lines); `exact` itself is compared with the stock as shipped, no recipe.

Level 1 (the kits' kernels-off recipe): one YAML per process, a fixed `--seed`, `--num_workers 1` (single-threaded featurization),
kernels off (`--no_kernels`: the torch triangle path; run-to-run bitwise), `--diffusion_samples 1` (the worker's shape). Upstream itself
sets `torch.set_float32_matmul_precision("highest")` and `seed_everything(seed)` (main.py:1096,1103); the worker replays the same RNG
position per (input, seed) (bz_worker_lev.py:2-25). The kernels-on stock (upstream's default, the pinned settings) IS process-reproducible
on H100 (the same input and seed give the same files run to run) — so the recipe's kernels-off switch is not a reproducibility aid but
the choice of anchor: `pred --mode off` leaves kernels at upstream's default unless the caller passes `--no_kernels`, and `--det 1` adds
the level-1 switches that are not already among the caller's own `boltz predict` options.
"""
from __future__ import annotations

from typing import List, Optional

LEVELS = (0, 1)


def stock_args(level: Optional[int], present: List[str]) -> List[str]:
    """Level-1 switches not already present in `present` (the stock arguments so far)."""
    if not level:
        return []
    if int(level) not in LEVELS:
        raise ValueError(f"unknown --det level {level}: {LEVELS}")
    out: List[str] = []
    if "--num_workers" not in present:
        out += ["--num_workers", "1"]
    if "--no_kernels" not in present:
        out += ["--no_kernels"]
    return out


def describe(level: Optional[int]) -> dict:
    return {"level": int(level or 0), "switches": ["--num_workers 1", "--no_kernels", "one YAML per process", "--seed <S>"] if level else [],
            "open": "kernels-off at the pinned settings"}
