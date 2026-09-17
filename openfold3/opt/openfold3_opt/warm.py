"""warm — one public-input prediction under the mode: imports, the Triton JIT (the trunk-kernels flash kernel and the fused pair cells),
the first graph captures. Uses the fast-inference kit's own public input builder
(``tests/public_inputs.py <tiling[,tiling…]> <outdir>``: PDB 1BRS barnase:barstar tiled 1:1 .. 6:6 = 199 .. 1194 tokens, single-sequence
MSAs; several tilings are several queries of ONE query JSON, folded in turn by the one process, so the size-dependent kernel variants and
captures of each are met once here) and the stock configuration; one seed,
one sample (``cli.warm_argv``: cli.WARM_KNOBS — MSA server off, templates off, both counts 1 — through the same argument builder as ``pred``).
The prediction runs in-process through ``pred``'s route (the same as any prediction under the mode).
"""
from __future__ import annotations

import os
import subprocess
import sys

from . import modes

PUBLIC_INPUTS = os.path.join("tests", "public_inputs.py")
DEFAULT_TILING = "1to1"


TILINGS = tuple(f"{n}to{n}" for n in range(1, 7))          # public_inputs.py MAX_TILING = 6: n barnase + n barstar chains = n x 199 tokens


def tilings_of(spec: str) -> list:
    """`--tiling` value -> the tiling list: one name or a comma-separated list (order kept); an unknown or duplicate name is a ValueError
    naming it (the caller's usage error), before anything is built."""
    out = [t.strip() for t in str(spec or DEFAULT_TILING).split(",") if t.strip()] or [DEFAULT_TILING]
    bad = [t for t in out if t not in TILINGS]
    if bad:
        raise ValueError(f"--tiling: unknown tiling {','.join(bad)} (one of {'|'.join(TILINGS)}, or a comma-separated list of them)")
    if len(set(out)) != len(out):
        raise ValueError(f"--tiling: duplicate tiling in {spec!r}")
    return out


def tiling_tokens(tiling: str) -> int:
    return int(tiling[0]) * 199


def public_query(home: str, out_dir: str, tiling: str = DEFAULT_TILING) -> str:
    """Build the kit's public query (one query per tiling of `tiling`, comma-separated) into `out_dir` with the kit's own script; returns the
    query JSON path (`<out_dir>/q.json`)."""
    tiling = ",".join(tilings_of(tiling))
    script = os.path.join(home, modes.KITS["fast_inference"], PUBLIC_INPUTS)
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run([sys.executable, script, tiling, out_dir], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"public_inputs.py failed: {r.stderr.strip()[:400]}")
    q = os.path.join(out_dir, "q.json")
    if not os.path.isfile(q):
        raise RuntimeError(f"public_inputs.py wrote no {q}")
    return q

