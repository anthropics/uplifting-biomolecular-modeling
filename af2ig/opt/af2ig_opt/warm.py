"""``af2ig-opt warm`` — the kit's warm-up process: one public complex through the stock line, outputs discarded; or, with ``--lengths L1,L2,…``
, the CACHE warm-up: one synthetic two-chain complex per total length through the ``--mode`` line, so JAX's compilation cache (ccache) and
the program store (L13) hold those lengths — the first real run at such a length then loads instead of compiling (README "First run at a new length").

The kit's rule: "Always run one throw-away process first in a fresh container: the first
process in a container pays 2-3x longer XLA compiles than every later process (driver-level caches)". This runs that process — the
smallest public complex the kit ships (``tests/inputs/pdbs/1brs_bb_1to1.pdb``, 195 residues) through ``af2ig-opt pred --mode off`` into a
temporary directory — and prints ``WARM PASS|FAIL wall=<s>``. It carries no lever: the effect is the container's (driver/PTX caches, the
libraries' pages), the same for every arm that follows.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

from opt_core import process as _core_process

from . import registry, report as _report, stack


def _relay(line: str) -> None:                                 # the warm-up process's own output, line by line as it lands
    sys.stderr.write(line); sys.stderr.flush()

WARM_INPUT = "1brs_bb_1to1.pdb"                              # tests/inputs/index.csv: the smallest public complex (195 residues)


def run(out_dir=None, timeout: float = 1200.0, det=None, stream=None) -> dict:
    stream = stream or sys.stderr
    try:
        kit = stack.kit_home()
    except stack.ActivationError as e:
        print(f"{_report.PREFIX} WARM FAIL: {e}", file=stream, flush=True)
        return {"ok": False, "inactive": True, "reason": str(e)}
    src = os.path.join(kit, registry.PUBLIC_INPUTS, WARM_INPUT)
    keep = out_dir is not None
    out_dir = os.path.abspath(out_dir) if out_dir else tempfile.mkdtemp(prefix="af2ig_warm_")
    in_dir = os.path.join(out_dir, "input"); os.makedirs(in_dir, exist_ok=True)
    shutil.copyfile(src, os.path.join(in_dir, WARM_INPUT))
    cmd = [sys.executable, "-m", "af2ig_opt", "pred", "--mode", "off", "--pdbdir", in_dir, "--out", os.path.join(out_dir, "out")] + (["--det", str(det)] if det is not None else [])
    print(f"{_report.PREFIX} WARM start input={WARM_INPUT} out={out_dir} timeout={timeout:.0f}s", file=stream, flush=True)
    t0 = time.time()
    run = _core_process.run_logged(cmd, timeout_s=timeout, on_line=_relay, what="warm")
    rc = None if run.timed_out else run.rc
    wall = time.time() - t0
    ok = rc == 0
    print(f"{_report.PREFIX} WARM {'PASS' if ok else 'FAIL'} rc={rc if rc is not None else 'timeout'} wall={wall:.1f}s out={out_dir if keep else 'discarded'}", file=stream, flush=True)
    if not keep:
        shutil.rmtree(out_dir, ignore_errors=True)
    return {"ok": ok, "inactive": rc == 3, "rc": rc, "wall_s": wall, "out": out_dir if keep else None}


# ---- warm the caches for given lengths -------------------------------------------------------------------------------
MAX_LENGTHS = 16

def parse_lengths(text: str) -> list:
    """``"400,800, 1200"`` -> [400, 800, 1200] (positive ints, deduplicated in order); ValueError names a bad token."""
    out = []
    for tok in str(text).replace(" ", "").split(","):
        if not tok:
            continue
        if not tok.isdigit() or int(tok) < 16:
            raise ValueError(f"--lengths: '{tok}' is not a residue count >= 16")
        if int(tok) not in out:
            out.append(int(tok))
    if not out:
        raise ValueError("--lengths: no length given")
    if len(out) > MAX_LENGTHS:
        raise ValueError(f"--lengths: at most {MAX_LENGTHS} lengths per warm-up")
    return out


def _backbone(n: int, origin=(0.0, 0.0, 0.0)):
    """An ideal alpha-helical poly-GLY backbone of n residues (N, CA, C, O per residue), by NeRF placement from ideal internal coordinates —
    geometry only has to featurise; the model's answer is discarded."""
    import math
    import numpy as np
    b = {"N-CA": 1.458, "CA-C": 1.525, "C-N": 1.329, "C-O": 1.229}
    ang = {"N-CA-C": 111.2, "CA-C-N": 116.2, "C-N-CA": 121.7, "CA-C-O": 120.8}
    phi, psi, omega = -57.8, -47.0, 180.0
    def place(a, b_, c, bond, angle, torsion):
        angle, torsion = math.radians(angle), math.radians(torsion)
        bc = c - b_; bc /= np.linalg.norm(bc); nrm = np.cross(b_ - a, bc); nrm /= np.linalg.norm(nrm); m = np.stack([bc, np.cross(nrm, bc), nrm], 1)
        d = np.array([-bond * math.cos(angle), bond * math.sin(angle) * math.cos(torsion), bond * math.sin(angle) * math.sin(torsion)])
        return c + m @ d
    N0 = np.array([0.0, 0.0, 0.0]); CA0 = np.array([b["N-CA"], 0.0, 0.0]); C0 = CA0 + np.array([-math.cos(math.radians(ang["N-CA-C"])), math.sin(math.radians(ang["N-CA-C"])), 0.0]) * b["CA-C"]
    atoms = [[N0, CA0, C0]]
    for _ in range(1, n):
        pN, pCA, pC = atoms[-1]
        N1 = place(pN, pCA, pC, b["C-N"], ang["CA-C-N"], psi); CA1 = place(pCA, pC, N1, b["N-CA"], ang["C-N-CA"], omega); C1 = place(pC, N1, CA1, b["CA-C"], ang["N-CA-C"], phi)
        atoms.append([N1, CA1, C1])
    out = []
    for i, (N_, CA_, C_) in enumerate(atoms):
        nxt = atoms[i + 1][0] if i + 1 < n else place(N_, CA_, C_, b["C-N"], ang["CA-C-N"], psi)
        O_ = place(nxt, CA_, C_, b["C-O"], ang["CA-C-O"], 180.0)
        out.append({"N": N_, "CA": CA_, "C": C_, "O": O_})
    o = np.array(origin)
    return [{k: v + o for k, v in r.items()} for r in out]


def write_complex(path: str, total_len: int) -> tuple:
    """A synthetic binder (chain A) + target (chain B) poly-GLY complex of `total_len` residues in PDB format; returns (binder_len, target_len)."""
    nb = max(16, min(80, total_len // 4)); nt = total_len - nb
    lines, serial = [], 1
    for chain, n, origin in (("A", nb, (0.0, 0.0, 0.0)), ("B", nt, (30.0, 0.0, 0.0))):
        for i, res in enumerate(_backbone(n, origin), start=1):
            for name in ("N", "CA", "C", "O"):
                x, y, z = res[name]
                lines.append(f"ATOM  {serial:5d}  {name:<3s} GLY {chain}{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}")
                serial += 1
        lines.append("TER")
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return nb, nt


def run_lengths(lengths, mode: str, det=None, out_dir=None, timeout: float = 3600.0, stream=None) -> dict:
    """The cache warm-up: one synthetic complex per total length through ``af2ig-opt pred --mode <mode> [--det N]`` into a temporary directory
    (discarded). The mode's own line runs — its L7 precompile pass compiles every length (the compilation cache stores the executables, the L13
    store the serialized programs where the mode's programs are storable) — so a later real run at those lengths loads them. ``--mode off`` keeps
    no cache: said, and the stock warm-up runs instead."""
    stream = stream or sys.stderr
    if mode == "off":
        print(f"{_report.PREFIX} WARM note: --mode off keeps no compilation cache or program store — --lengths has nothing to warm; running the stock warm-up", file=stream, flush=True)
        return run(out_dir=out_dir, det=det, stream=stream)
    try:
        stack.kit_home()
    except stack.ActivationError as e:
        print(f"{_report.PREFIX} WARM FAIL: {e}", file=stream, flush=True)
        return {"ok": False, "inactive": True, "reason": str(e)}
    keep = out_dir is not None
    out_dir = os.path.abspath(out_dir) if out_dir else tempfile.mkdtemp(prefix="af2ig_warm_")
    in_dir = os.path.join(out_dir, "input"); os.makedirs(in_dir, exist_ok=True)
    for L in lengths:
        write_complex(os.path.join(in_dir, f"warm_L{L:05d}.pdb"), L)
    cmd = [sys.executable, "-m", "af2ig_opt", "pred", "--mode", mode, "--pdbdir", in_dir, "--out", os.path.join(out_dir, "out"), "--allow-partial"] + (["--det", str(det)] if det is not None else [])
    word = ",".join(str(L) for L in lengths)
    print(f"{_report.PREFIX} WARM start lengths={word} mode={mode} det={det if det is not None else 'default'} out={out_dir} timeout={timeout:.0f}s (synthetic two-chain poly-GLY complexes; outputs discarded)", file=stream, flush=True)
    t0 = time.time()
    r = _core_process.run_logged(cmd, timeout_s=timeout, on_line=_relay, what="warm-lengths")
    rc = None if r.timed_out else r.rc
    wall = time.time() - t0; ok = rc == 0
    print(f"{_report.PREFIX} WARM {'PASS' if ok else 'FAIL'} rc={rc if rc is not None else 'timeout'} wall={wall:.1f}s out={out_dir if keep else 'discarded'} lengths={word} mode={mode}", file=stream, flush=True)
    if not keep:
        shutil.rmtree(out_dir, ignore_errors=True)
    return {"ok": ok, "inactive": rc == 3, "rc": rc, "wall_s": wall, "out": out_dir if keep else None, "lengths": lengths, "mode": mode}
