"""warm — one scoring of the package's public fixture through `score`: the JIT / autotune warm-up of a mode on this box.

`e1-opt warm --variant V [--mode M] [--det D]` runs `python -m e1_opt score --variant V --mode M --parent-path <fixture>/parent.fasta --mutants-path <fixture>/mutants.fasta --output-path <tmp>/warm/scores.csv` as a
subprocess (its lines — the activation line, the kit's or the stock runner's own, the score line — are relayed verbatim) on the
fixture under `tests/fixtures/warm/` (a SHORT SYNTHETIC parent sequence and a few single-substitution mutants; not a natural protein, no
biological claim: its only purpose is to drive the mode's route once so the persistent caches — TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR —
are filled for the shape class it exercises; the kit's own apply performs its own warm-up as well). PASS = the scoring exited 0 and
wrote scores.csv with one row per mutant. Verdict line: `[e1-opt] WARM PASS|FAIL mode=<m> variant=<v> rows=<n> triton_cache=<a>-><b>
inductor_cache=<a>-><b> rc=<rc> wall=<s>`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

from . import outputs, report
from ._names import EXIT_NOT_ACTIVE                                            # the one exit-code table

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "fixtures", "warm")


def _count_files(env_var: str) -> Optional[int]:
    d = os.environ.get(env_var)
    if not d or not os.path.isdir(d):
        return None
    n = 0
    for _, _, files in os.walk(d):
        n += len(files)
    return n


def run(mode: str, variant: Optional[str], *, det: int = 0, fixture_dir: Optional[str] = None) -> dict:
    fixture_dir = fixture_dir or FIXTURE_DIR
    with open(os.path.join(fixture_dir, "mutants.fasta"), encoding="utf-8") as fh:
        n_mutants = sum(1 for line in fh if line.startswith(">"))       # one FASTA record per variant: the rows the tool's scores.csv must have
    tmp = tempfile.mkdtemp(prefix="e1_opt_warm_")
    scores = os.path.join(tmp, "warm", "scores.csv")
    before = {"triton": _count_files("TRITON_CACHE_DIR"), "inductor": _count_files("TORCHINDUCTOR_CACHE_DIR")}
    cmd = [sys.executable, "-m", "e1_opt", "score", "--mode", mode, "--parent-path", os.path.join(fixture_dir, "parent.fasta"), "--mutants-path", os.path.join(fixture_dir, "mutants.fasta"),
           "--output-path", scores, "--det", str(int(det))]
    if variant:
        cmd += ["--variant", variant]
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)
    lines = []
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line.rstrip("\n"))
    rc = p.wait()
    wall = time.perf_counter() - t0
    lst = outputs.listing(scores)
    after = {"triton": _count_files("TRITON_CACHE_DIR"), "inductor": _count_files("TORCHINDUCTOR_CACHE_DIR")}
    variant_seen = variant
    for s in lines:
        m = report.RE_ACTIVE.match(s) or report.RE_OFF.match(s)
        if m:
            variant_seen = m.group("variant")
    ok = rc == 0 and lst["present"] and lst["n_rows"] == n_mutants
    res = {"status": "PASS" if ok else "FAIL", "mode": mode, "variant": variant_seen, "det": det, "exit_code": rc, "wall_s": round(wall, 1),
           "rows": lst["n_rows"], "expected_rows": n_mutants, "scores_sha256": lst["sha256"], "cache_files": {"before": before, "after": after},
           "command": cmd, "fixture": fixture_dir}
    if rc == EXIT_NOT_ACTIVE:
        res["reason"] = "the scoring was refused (NOT ACTIVE)"
    elif rc != 0:
        res["reason"] = f"score exited {rc}"
    elif not lst["present"]:
        res["reason"] = "no scores.csv written"
    elif lst["n_rows"] != n_mutants:
        res["reason"] = f"scores.csv has {lst['n_rows']} rows, the fixture has {n_mutants} mutants"
    shutil.rmtree(tmp, ignore_errors=True)
    report.emit(report.warm_line(res["status"], mode, variant_seen, rows=lst["n_rows"],
                                 triton_cache=f"{before['triton']}->{after['triton']}", inductor_cache=f"{before['inductor']}->{after['inductor']}",
                                 rc=rc, wall=f"{res['wall_s']}s", **({"reason": res["reason"]} if res.get("reason") else {})))
    return res
