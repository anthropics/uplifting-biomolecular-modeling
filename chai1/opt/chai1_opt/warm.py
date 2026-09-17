"""warm — one prediction on the kit's own public input (``tests/public_inputs/1BRS_1to1.fasta``, 199 protein tokens, seed 0) through
``pred`` in a subprocess, so that the route is exercised end to end once before a real run: the weights are located and loaded, the
ACTIVE / ready lines are seen, the first fold's one-off costs (TorchScript profiling of the traced ESM, cuBLAS handle creation) are paid.
``exact`` compiles nothing; ``fast`` JIT-compiles the add-on's Triton kernels on first use per crop into Triton's own cache
(``TRITON_CACHE_DIR``), which a plain warm fills for crop 256 only. ``warm --crops all|N,N`` adds one synthetic single-chain protein input per
requested crop size (``CROPS``: chai-lab's seven model sizes) folded in ONE pred process at ``CACHE_FOLD`` (the real run's shapes — five diffusion
samples — on the shortest schedule: two denoiser steps, one trunk recycle, no ESM), so the mode's kernel caches under the JIT root (Triton's,
Inductor's compiled-step kernels and FX-graph entries) are filled for every crop a later run meets; what stays per process is the compiled step's
own trace (README, first use). It folds at its own fold settings ``WARM_FOLD`` (one diffusion sample, the batch kept on the GPU: the shortest fold), given to
``pred`` as stock's flags; they are warm's constant, not a command-line choice. Outputs go to a temporary directory unless ``--keep``.

Result: ``{"status": PASS|FAIL, "activation", "ready", "exit_code", "wall_s", "log", "predictions"}``.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

from . import report as _report, settings, stack

PUBLIC_INPUT_RELPATH = os.path.join("tests", "public_inputs", "1BRS_1to1.fasta")
WARM_FOLD = {"num_diffn_samples": 1, "low_memory": False}   # warm's own fold settings (run_inference keywords): one diffusion sample, the batch on the GPU — the shortest fold

CROPS = (256, 384, 512, 768, 1024, 1536, 2048)                # chai_lab 0.6.1 AVAILABLE_MODEL_SIZES: the crop sizes a run can meet
CACHE_FOLD = {"num_diffn_timesteps": 2, "num_trunk_recycles": 1, "use_esm_embeddings": False, "low_memory": False}   # a real fold's shapes (5 samples), shortest schedule
CROP_MARGIN = 16                                              # synthetic chain length = crop - CROP_MARGIN tokens (pads to that crop)


def parse_crops(word: Optional[str]) -> tuple:
    """``--crops`` value -> crop sizes: None/"" -> (), "all" -> CROPS, "512,1024" -> those (each must be one of CROPS; refused by name otherwise)."""
    if not word:
        return ()
    if word.strip().lower() == "all":
        return CROPS
    out = []
    for w in word.replace(" ", "").split(","):
        if not w:
            continue
        n = int(w)
        if n not in CROPS:
            raise ValueError(f"--crops: {n} is not one of chai-lab's model sizes {CROPS}")
        if n not in out:
            out.append(n)
    return tuple(out)


def crop_inputs(dirpath: str, crops) -> str:
    """One synthetic single-chain protein FASTA per crop (poly-alanine of crop - CROP_MARGIN residues, chai's `>protein|name=` header) and the
    items JSON over them (seed 0); returns the items JSON path."""
    import json
    os.makedirs(dirpath, exist_ok=True)
    items = []
    for c in crops:
        fa = os.path.join(dirpath, f"warm_crop{c}.fasta")
        with open(fa, "w", encoding="utf-8") as f:
            f.write(f">protein|name=warm-crop{c}\n" + "A" * (int(c) - CROP_MARGIN) + "\n")
        items.append({"fasta": fa, "seeds": [0]})
    path = os.path.join(dirpath, "warm_crops.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, indent=1)
    return path


def public_input() -> str:
    p = os.path.join(stack.kit_home(), PUBLIC_INPUT_RELPATH)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"the kit's public input is missing: {p}")
    return p


def pred_command(mode: str, *, input_path: str, out_dir: str, tag: str, det_level: int, seeds: str = "0", fold: Optional[dict] = None) -> list:
    """``python -m chai1_opt pred`` for one input through the mode's own route (off: the stock caller in a clean process; a kit mode: the
    driver) at the fold settings ``fold`` (run_inference keyword -> value, handed on as stock's flags; default WARM_FOLD) — the command warm runs."""
    from . import settings
    return ([sys.executable, "-m", "chai1_opt", "pred", "--mode", mode] + settings.fold_flags(WARM_FOLD if fold is None else fold) +
            ["--input", input_path, "--seeds", str(seeds), "--out_dir", out_dir, "--tag", tag, "--det", str(int(det_level))])


def run(mode: str, *, det_level: int = 0, log_path: Optional[str] = None, echo: bool = True, keep: bool = False, crops=()) -> dict:
    tmp = tempfile.mkdtemp(prefix="chai1_opt_warm_")
    out_dir = os.path.join(tmp, "out")
    log_path = log_path or os.path.join(tmp, "warm.log")
    crops = tuple(crops or ())
    if crops:                                                                  # the cache filler: every requested crop in one pred process at CACHE_FOLD
        cmd = pred_command(mode, input_path=crop_inputs(os.path.join(tmp, "inputs"), crops), out_dir=out_dir, tag="warm", det_level=det_level, seeds="0", fold=CACHE_FOLD)
    else:
        cmd = pred_command(mode, input_path=public_input(), out_dir=out_dir, tag="warm", det_level=det_level, seeds="0")
    phases = {}
    t0 = time.time(); activation = None; ready = None; lines = []
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for ln in p.stdout:
            lines.append(ln); log.write(ln); log.flush()
            if echo:
                sys.stderr.write(ln); sys.stderr.flush()
            if (_report.PREFIX + " ACTIVE ") in ln or (_report.PREFIX + " NOT ACTIVE") in ln:
                activation = ln.strip()
            if " ready route=" in ln:
                ready = ln.strip()
            if " FORWARD item=" in ln and "warm_crop" in ln and "forward_s=" in ln:   # per-crop seconds of the cache-filling folds (the kit's forward timer line)
                m = re.search(r"warm_crop(\d+)\b.*?forward_s=([\d.]+)", ln)
                if m:
                    phases[int(m.group(1))] = float(m.group(2))
        rc = p.wait()
    wall = time.time() - t0
    n_cif = len(glob.glob(os.path.join(out_dir, "warm", "*", "seed_0", "pred.model_idx_*.cif")))
    partial = " PARTIAL " in (activation or "")                                # never a PASS on a partial activation (warm exercises the whole mode; a PARTIAL line is status FAIL here, not a refusal)
    want = len(crops) if crops else 1
    res = {"status": "PASS" if (rc == 0 and n_cif >= want and not partial and (mode == "off" or (activation or "").find(" ACTIVE ") >= 0)) else "FAIL",
           "mode": mode, "fold": dict(CACHE_FOLD if crops else WARM_FOLD), "det": int(det_level), "activation": activation, "ready": ready, "exit_code": rc,
           "wall_s": round(wall, 1), "log": log_path, "predictions": n_cif, "command": cmd, "crops": {c: phases.get(c) for c in crops}}
    if rc == 3:
        res["reason"] = f"pred refused the activation (exit 3): {activation or 'no NOT ACTIVE line captured'}"   # the activation's own refusal words
    elif partial:
        res["reason"] = f"partial activation: {activation}"
    elif rc != 0:
        res["reason"] = f"pred exited {rc}"
    elif n_cif == 0:
        res["reason"] = "no prediction written"
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        res["out_dir"] = out_dir
    return res


def summary_line(res: dict) -> str:
    crops = res.get("crops") or {}
    cw = (" crops=" + ",".join(f"{c}:{('%.0fs' % t) if t is not None else '?'}" for c, t in crops.items())) if crops else ""
    return (f"{_report.PREFIX} WARM {res.get('status')} mode={res.get('mode')} predictions={res.get('predictions')} "
            f"rc={res.get('exit_code')} wall={res.get('wall_s')}s{cw} log={res.get('log')}" + (f" reason={res['reason']}" if res.get("reason") else ""))
