"""``warm`` — one design job on a public input through ``design`` so that the JIT work of a fresh box is done before a timed run:
upstream's fast-LayerNorm extension build (``protenix/model/layer_norm/torch_ext_compile.py``, built inside the installed ``protenix``
package — ``layer_norm.py`` passes its own directory as ``build_directory``) and the first sampler call. The input is the first task of the lever kit's ``inputs/tasks_3targets.json`` (5O45, chain A 1-116, binder 75) on the
kit's vendored copy of that public structure (``inputs/targets/``); ``--N_sample 1`` (the smallest design job) unless overridden. Reports the
activation and application lines seen, the design count and the file counts of ``TORCH_EXTENSIONS_DIR`` before/after. A reused ``--out_dir`` that
already holds the warm design is a FAIL by name (the child printed NOTHING RAN: upstream skipped the pair, no sampler call warmed anything).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Callable, List, Optional, Tuple

ACTIVATION_RX, APPLIED_RX, NOT_ACTIVE_RX, NOTHING_RAN_RX = "] ACTIVE mode=", "] APPLIED model#", "] NOT ACTIVE", "] NOTHING RAN:"
WARM_SEED = 317000                                                          # the warm-up job's seed: any fixed seed makes the warm job reproducible (upstream would otherwise derive one from the clock)
INPUTS_RELPATH = "inputs"                                                   # under the lever kit (opt/forward/hoist): the warm-up's public inputs
TASKS_RELPATH = os.path.join(INPUTS_RELPATH, "tasks_3targets.json")      # three design tasks on public wwPDB entries (5O45, 1TNF, 3DI3)
TARGETS_RELPATH = os.path.join(INPUTS_RELPATH, "targets")                # the vendored copies of those entries (inputs/targets/SOURCES.json)


class WarmError(RuntimeError):
    pass


class ProcessTimeout(WarmError):
    def __init__(self, timeout: float, wall: float, what: str = "warm-up"):
        super().__init__(f"{what} exceeded --timeout {timeout:g} s: process group killed at {wall:.1f} s")
        self.timeout, self.wall = timeout, wall


def run_logged(cmd: List[str], on_line: Callable[[str], None], timeout: Optional[float] = None, **popen_kw) -> Tuple[int, bool]:
    """Run `cmd` in its own process group with stdout+stderr merged, calling ``on_line`` per line; returns (exit_code, timed_out).
    The shared core's runner (`opt_core.process.run_logged`: the group is killed on timeout and on every exit path)."""
    from opt_core.process import run_logged as _run_logged
    r = _run_logged(cmd, timeout_s=timeout, on_line=on_line, what="warm-transcript", **popen_kw)
    return r.rc, r.timed_out


def stage_tasks(kit: str, out_dir: str) -> dict:
    """A copy of the kit's inputs/tasks_3targets.json with absolute paths to the vendored targets (checked present), written under
    ``out_dir``; returns its path and the staged target files (presence is the gate; nothing is digested)."""
    targets = os.path.join(kit, TARGETS_RELPATH)
    src = os.path.join(kit, TASKS_RELPATH)
    with open(src, "r", encoding="utf-8") as fh:
        tasks = json.load(fh)
    staged = []
    for x in tasks:
        name = os.path.basename(x["condition"]["structure_file"])
        p = os.path.join(targets, name)
        if not os.path.isfile(p):
            raise WarmError(f"public target {name} is not vendored at {targets}")
        staged.append(name)
        x["condition"]["structure_file"] = os.path.abspath(p)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "tasks_3targets.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(tasks, fh, indent=1)
    return {"tasks": dst, "targets_dir": targets, "targets": staged, "source": src}


def _cache_count() -> int:
    d = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not d or not os.path.isdir(d):
        return 0
    return sum(len(f) for _, _, f in os.walk(d))


def run(mode: str, out_dir: str | None = None, timeout: float | None = 1800, n_sample: int = 1, echo: bool = True, keep: bool = False) -> dict:
    from . import report, stack
    from .outputs import PREDICTIONS_DIR
    tmp = out_dir or tempfile.mkdtemp(prefix="pxdesign-opt-warm-")
    staged = stage_tasks(stack.kit_home(), tmp)
    with open(staged["tasks"], "r", encoding="utf-8") as fh:
        tasks = json.load(fh)
    one = os.path.join(tmp, "task_warm.json")
    with open(one, "w", encoding="utf-8") as fh:
        json.dump(tasks[:1], fh, indent=1)
    run_dir = os.path.join(tmp, "out")
    cmd = [sys.executable, "-m", "pxdesign_opt", "design", "--mode", mode, "--tasks", one, "--out_dir", run_dir, "--N_sample", str(n_sample), "--seeds", str(WARM_SEED)]
    before = _cache_count()
    lines, t0 = [], time.time()
    log_path = os.path.join(tmp, "warm.log")
    with open(log_path, "w", encoding="utf-8") as log:
        def on_line(line):
            lines.append(line); log.write(line); log.flush()
            if echo:
                sys.stderr.write(line); sys.stderr.flush()
        rc, killed = run_logged(cmd, on_line, timeout=timeout)
    wall = time.time() - t0
    activation = next((ln.strip() for ln in lines if ACTIVATION_RX in ln or NOT_ACTIVE_RX in ln), None)
    applied = next((ln.strip() for ln in lines if APPLIED_RX in ln), None)
    done = next((ln.strip() for ln in lines if report.DONE_RE.match(ln.strip())), None)   # the design verb's DONE | INCOMPLETE line (report.py owns the grammar)
    nothing_ran = next((ln.strip() for ln in lines if NOTHING_RAN_RX in ln), None)      # a reused --out_dir already holding the warm design: upstream skipped it, no sampler call ran (report.nothing_ran_line)
    n_cif = len(glob.glob(os.path.join(run_dir, "*", "seed_*", PREDICTIONS_DIR, "*.cif")))
    ok = rc == 0 and not killed and n_cif == n_sample and (mode == "off" or applied) and not nothing_ran
    res = {"status": "PASS" if ok else "FAIL", "mode": mode, "activation": activation, "applied": applied, "done": done or nothing_ran, "exit_code": rc, "killed": killed,
           "timed_out": killed, "wall_s": round(wall, 1), "log": log_path, "designs": n_cif, "torch_ext_files": {"before": before, "after": _cache_count()}, "command": cmd, "inputs": staged}
    if killed:
        err = ProcessTimeout(timeout, wall)
        res["error"] = type(err).__name__; res["reason"] = str(err)
    elif rc != 0:
        res["reason"] = f"design exited {rc}" + (" (NOT ACTIVE: the lever was not applied, or not wholly)" if rc == 3 else "")
    elif nothing_ran:
        res["reason"] = f"nothing ran: {run_dir} already held the warm design and upstream skipped it (no sampler call, nothing warmed); pass a fresh --out_dir"
    elif n_cif != n_sample:
        res["reason"] = f"{n_cif} design(s) written, expected {n_sample}"
    if not keep and out_dir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        res["out_dir"] = tmp
    return res


def summary_line(res: dict) -> str:
    from .report import PREFIX
    tc = res.get("torch_ext_files") or {}
    return (f"{PREFIX} WARM {res.get('status')} mode={res.get('mode')} designs={res.get('designs')} torch_ext={tc.get('before')}->{tc.get('after')} "
            f"rc={res.get('exit_code')} wall={res.get('wall_s')}s log={res.get('log')}" + (f" reason={res['reason']}" if res.get("reason") else ""))
