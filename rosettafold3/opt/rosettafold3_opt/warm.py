"""``warm``: one ``pred`` over the kit's bundled public input so the mode's caches are filled.

The first fold of an input-size class compiles the Triton kernels the mode uses (the kit's, cuEquivariance's) into ``TRITON_CACHE_DIR``
(configs/*.env): the fused LayerNorm-linear / transition kernels specialise on the pair-row bucket (< 256, 256-632, >= 633 tokens) and several
kernels on the 16-divisibility of their row counts (the token count, its square for the pair kernels), so a later fold in a class already met
compiles nothing; a token count itself divisible by 16 is one class the bundled tiles do not cover (its first fold compiles a few kernels more). ``warm`` folds the four 1BRS
tiles of ``public_inputs/1brs_tiles.json`` (199, 398, 597, 796 tokens: the three buckets; the largest's pair rows 16-divisible) in one
process at ``rf3 fold``'s own defaults (``WARM_OVERRIDES``: none), then deletes the outputs (``--out <dir>`` keeps them there). What stays
per process whatever the cache holds: the sampler graph's capture and the kernels' autotune replay on the first fold of each token count
(seconds). ``--mode off`` is refused: the stock arm has no warm (``pred --mode off`` is its one command; run.sh refuses it the same way).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Optional

from . import fold as _fold
from . import inputs as _inputs
from . import report as _report
from . import stack as _stack


class WarmError(RuntimeError):
    pass

WARM_INPUT = "1brs_tiles.json"             # every item of the bundled public input: the four 1BRS tiles, smallest first
WARM_OVERRIDES: tuple = ()          # rf3 fold at its own defaults: warm states no sampling setting
WARM_SEED = 0


def warm_items() -> list:
    """The bundled public input's items (inputs.load: validated, in file order)."""
    return _inputs.load(os.path.join(_stack.kit_home(), "public_inputs", WARM_INPUT))


def public_input(out_dir: str) -> str:
    return _inputs.write(warm_items(), os.path.join(out_dir, "input", WARM_INPUT))


def run(mode: str, *, ckpt: Optional[str] = None, out_dir: Optional[str] = None, log_path: Optional[str] = None, allow_partial: bool = False) -> dict:
    if (mode or "").strip().lower() == "off":
        raise WarmError("warm --mode off is refused: the stock arm has no warm (pred --mode off is its one command)")
    od = out_dir or tempfile.mkdtemp(prefix="rosettafold3_warm_")
    t0 = time.time()
    inp = public_input(od)
    rec = _fold.run(mode, inputs=inp, out_dir=od, ckpt=ckpt, overrides=WARM_OVERRIDES, seeds=[WARM_SEED], log_path=log_path or os.path.join(od, "warm.log"),
                    allow_partial=allow_partial)
    res = {"status": rec["status"], "mode": mode, "items": [it["name"] for it in _inputs.load(inp)], "wall_s": round(time.time() - t0, 1), "out_dir": od,
           "runs": rec["runs"], "tree_state": rec["tree_state"]}
    if rec["status"] == "PASS" and out_dir is None:
        shutil.rmtree(od, ignore_errors=True)
        res["out_dir"] = None
    return res


def summary_line(res: dict) -> str:
    return f"{_report.PREFIX} warm {res['status']} mode={res['mode']} items={len(res['items'])}:{WARM_INPUT} wall_s={res['wall_s']} tree={res['tree_state']}"
