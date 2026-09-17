"""`warm` — one prediction of upstream's smallest documented example (the README's 9-residue chain, no MSA; `WARM_QUERY` / `WARM_FLAGS` below)
through `pred` on the selected mode/line, in a subprocess: Triton JIT and the tuned tile cache happen
here instead of on the first real item. The outputs are deleted unless `--keep`; `opt_manifest.json` is kept beside the log."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from . import modes, nostdin
from . import report as _report


WARM_QUERY = [{"name": "tiny", "modelSeeds": [101], "sequences": [{"proteinChain": {"sequence": "ACDEFGHIK", "count": 1}}]}]   # upstream README "Running Your First Prediction", verbatim
WARM_FLAGS = ["--cycle", "10", "--step", "200", "--sample", "1", "--dtype", "bf16", "--model_name", "opendde_v1", "--use_msa", "false",
              "--use_template", "false", "--use_rna_msa", "false", "--need_atom_confidence", "true"]   # upstream's flags for the warm-up item: one sample, no MSA / templates, the tree's
                                                                                              # tested dtype (bf16) so the JIT and tile caches it fills are the ones a bf16 run reads


def run(mode_or_line: str, out_dir: str | None = None, *, keep: bool = False) -> dict:
    out_dir = os.path.abspath(out_dir or tempfile.mkdtemp(prefix="opendde_warm_"))
    os.makedirs(out_dir, exist_ok=True)
    q = os.path.join(out_dir, "tiny.json")
    with open(q, "w") as fh:
        json.dump(WARM_QUERY, fh, indent=1)
    pred_dir = os.path.join(out_dir, "pred")
    cmd = [sys.executable, "-m", "opendde_opt", "pred", "--mode", mode_or_line, "-i", q, "-o", pred_dir, *WARM_FLAGS]
    t0 = time.time()
    log_path = os.path.join(out_dir, "warm.log")
    with open(log_path, "w") as logf:
        rc = subprocess.call(cmd, stdin=nostdin.CHILD_STDIN, stdout=logf, stderr=subprocess.STDOUT)
    wall = round(time.time() - t0, 1)
    mp = os.path.join(pred_dir, "opt_manifest.json")
    man = json.load(open(mp)) if os.path.isfile(mp) else None
    if man:
        shutil.copy(mp, os.path.join(out_dir, "opt_manifest.json"))
    n_files = (man or {}).get("outputs", {}).get("n_files") if man else None
    if not keep and os.path.isdir(pred_dir):
        shutil.rmtree(pred_dir, ignore_errors=True)
    status = "PASS" if (rc == 0 and man and (n_files or 0) > 0) else "FAIL"
    return {"status": status, "rc": rc, "wall_s": wall, "log": log_path, "out": out_dir, "selection": mode_or_line, "n_files": n_files,
            "levers": (man or {}).get("activation", {}).get("levers_applied") if man else None, "cmd": cmd}


def summary_line(r: dict) -> str:
    return f"{_report.PREFIX} WARM {r['status']} {r['selection']} rc={r['rc']} wall_s={r['wall_s']} files={r.get('n_files')} levers={r.get('levers')} log={r['log']}"
