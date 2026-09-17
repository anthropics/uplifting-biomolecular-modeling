"""warm — one public-input pass through ``design`` in the requested mode on this box: the kit's CUDA-graph capture and the worker's
own device line, measured here (the activation line's GPU and the kit README's probe statement are the dry-run values; this is the
run). The kit keys no compile cache of its own — the CUDA graphs are captured per process, and the worker's one Triton kernel is compiled on
first use into Triton's own cache — so warm exercises the route more than it fills a cache: ``WARM PASS`` = the pass exited 0 and wrote its outputs.

Inputs: the upstream examples inside the checkout — ``$MPNN_DIR/inputs/PDB_monomers/pdbs/`` (two monomers) — so nothing is fetched.
Outputs go to a temporary directory unless ``--keep``; opt_manifest.json is written either way and its path is reported. The pass's stock
options are ``WARM_STOCK_ARGS`` (one seeded sequence per backbone, with the score / probability arrays requested: ``--save_score 1 --save_probs 1``).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Optional, Sequence

from . import manifest, modes, report, stack

BASE_INPUT = os.path.join("inputs", "PDB_monomers", "pdbs")           # under MPNN_DIR
# The stock options of warm's one pass (settings.parse accepts them; the kit worker serves every value here).
WARM_STOCK_ARGS = ["--num_seq_per_target", "1", "--batch_size", "1", "--sampling_temp", "0.1", "--seed", "37", "--backbone_noise", "0.0",
                   "--save_score", "1", "--save_probs", "1", "--omit_AAs", "X", "--model_name", "v_48_020"]


def public_input(variant: str) -> str:
    mdir = stack.mpnn_dir() or ""
    p = os.path.join(mdir, BASE_INPUT)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"{p}: the upstream example inputs are not under {stack.ENV_MPNN_DIR}={mdir!r}")
    return p


def run(mode: str, variant: str, keep: bool = False, allow_partial: bool = False, opt_out: Sequence[str] = (), bb_batch: Optional[int] = None) -> dict:
    """``rc`` is design's own exit (3 on a partial activation unless ``allow_partial``, forwarded as ``--allow-partial``; ``opt_out`` names the
    probe-gated levers left out, forwarded as their ``--<lever> 0``; ``bb_batch`` forwarded as ``--bb_batch``). The pass's lines and its child's
    output are relayed as ``design`` relays them; the outputs go to a temporary directory, kept with ``keep``."""
    from . import cli
    variant = modes.check_variant(variant)
    out = tempfile.mkdtemp(prefix=f"proteinmpnn_opt_warm_{variant}_{mode}_")
    t0 = time.time()
    argv = ["--mode", mode, "--variant", variant, "--input", public_input(variant), "--out", out]
    if allow_partial:
        argv.append("--allow-partial")
    for name in opt_out:
        argv += [modes.OPT_OUTS[name], "0"]
    if bb_batch is not None:
        argv += ["--bb_batch", str(bb_batch)]
    argv += WARM_STOCK_ARGS
    rc = cli.cmd_design(argv)
    wall = time.time() - t0
    n_fa = len([f for f in os.listdir(os.path.join(out, "seqs"))]) if os.path.isdir(os.path.join(out, "seqs")) else 0
    res = {"status": "PASS" if rc == 0 and n_fa > 0 else "FAIL", "rc": rc, "wall_s": round(wall, 1), "n_fa": n_fa, "out": out, "mode": mode, "variant": variant,
           "manifest": os.path.join(out, manifest.FILENAME)}
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
        res["out"] = None
        res["manifest"] = None
    return res


def summary_line(res: dict) -> str:
    return (f"{report.PREFIX} WARM {res['status']} mode={res['mode']} variant={res['variant']} rc={res['rc']} wall={res['wall_s']}s fa_files={res['n_fa']}"
            + (f" out={res['out']}" if res.get("out") else ""))
