"""warm — one prediction on the kit's own public input (the first input of ``tests/w4_public_slice.json``, 1BRS, single sequence)
through ``pred`` in a subprocess, so that the Triton kernels of the mode are JIT-compiled into the cache (``TRITON_CACHE_DIR``)
and the activation / application lines are seen once before a real run. The kit's imports and JIT happen at the first fold,
so the warm-up is a real fold at one seed; its outputs go to a temporary directory.

Result: ``{"status": PASS|FAIL, "activation", "applied", "exit_code", "wall_s", "log", "triton_cache_files": {before, after}}``.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
import time
from typing import Callable, List, Optional, Tuple

from .report import EXIT_NOT_ACTIVE, PREFIX

PUBLIC_ITEMS_RELPATH = os.path.join("tests", "w4_public_slice.json")
ACTIVATION_RX = f"{PREFIX} ACTIVE "
APPLIED_RX = f"{PREFIX} APPLIED "
NOT_ACTIVE_RX = f"{PREFIX} NOT ACTIVE"


def run_logged(cmd: List[str], on_line: Callable[[str], None], **popen_kw) -> int:
    """Run `cmd` in its own process group with stdout+stderr merged, calling ``on_line`` for every line as it arrives; returns the exit
    code (opt_core.process.run_logged, no deadline)."""
    from opt_core.process import run_logged as _run
    return _run(cmd, timeout_s=None, on_line=on_line, **popen_kw).rc


def public_items() -> str:
    from . import stack
    p = os.path.join(stack.kit_home(), PUBLIC_ITEMS_RELPATH)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"the kit's public items file is missing: {p}")
    return p


def _cache_count() -> Optional[int]:
    d = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    if not os.path.isdir(d):
        return 0
    return sum(1 for _ in glob.glob(os.path.join(d, "**", "*"), recursive=True))


def run(mode: str, variant: str, log_path: Optional[str] = None, echo: bool = True, keep: bool = False, line: Optional[str] = None) -> dict:
    """One pred in a subprocess; ``status`` PASS iff it exited 0 with a prediction and an APPLIED line under a kit mode; ``exit_code`` is
    pred's own (3 = NOT ACTIVE: the pinned installation not met, or a lever of the mode's set could not run on this device — pred refuses by name)."""
    import json
    items = public_items()
    with open(items, "r", encoding="utf-8") as fh:
        first = json.load(fh)[0]
    tmp = tempfile.mkdtemp(prefix="esmfold2_opt_warm_")
    items_one = os.path.join(tmp, "warm_item.json")
    with open(items_one, "w", encoding="utf-8") as fh:
        json.dump([first], fh)
    out_dir = os.path.join(tmp, "out")
    log_path = log_path or os.path.abspath(f"esmfold2_opt_warm_{time.strftime('%Y%m%dT%H%M%S')}.log")
    cmd = [sys.executable, "-m", "esmfold2_opt", "pred", "--mode", mode, "--variant", variant, "--input", items_one, "--out_dir", out_dir, "--seeds", "0"]
    before = _cache_count()
    t0 = time.time()
    lines = []
    with open(log_path, "w", encoding="utf-8") as log:
        def on_line(line: str) -> None:
            lines.append(line); log.write(line); log.flush()
            if echo:
                sys.stderr.write(line); sys.stderr.flush()
        rc = run_logged(cmd, on_line)
    wall = time.time() - t0
    activation = next((ln.strip() for ln in lines if ACTIVATION_RX in ln or NOT_ACTIVE_RX in ln), None)
    applied = next((ln.strip() for ln in lines if APPLIED_RX in ln), None)
    from . import outputs                                                               # here, not at import: outputs needs numpy, check does not
    n_cif = len(glob.glob(os.path.join(out_dir, outputs.CIF_DIR, "*.cif")))                # the driver file set (outputs.py): <out_dir>/cif_all/*.cif
    res = {"status": "PASS" if (rc == 0 and n_cif > 0 and (mode == "off" or applied)) else "FAIL", "mode": mode, "variant": variant,
           "activation": activation, "applied": applied, "exit_code": rc, "wall_s": round(wall, 1), "log": log_path,
           "predictions": n_cif, "triton_cache_files": {"before": before, "after": _cache_count()}, "command": cmd}
    if rc != 0:
        res["reason"] = f"pred exited {rc}" + (" (levers not active: the NOT ACTIVE line names why)" if rc == EXIT_NOT_ACTIVE else "")
    elif n_cif == 0:
        res["reason"] = "no prediction written"
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        res["out_dir"] = out_dir
    return res


def summary_line(res: dict) -> str:
    tc = res.get("triton_cache_files") or {}
    return (f"{PREFIX} WARM {res.get('status')} mode={res.get('mode')} variant={res.get('variant')} predictions={res.get('predictions')} "
            f"triton_cache={tc.get('before')}->{tc.get('after')} rc={res.get('exit_code')} wall={res.get('wall_s')}s log={res.get('log')}"
            + (f" reason={res['reason']}" if res.get("reason") else ""))
