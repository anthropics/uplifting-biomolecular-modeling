"""``warm`` — one public request through the mode's line: upstream's binder-design example (examples/binder_design/experiment.yaml,
problem 01_bhrf1) at ``--n 1``, so that the driver's set-up, the checkout's imports and one CUDA-graph capture have run on this box
before a served or measured pass. Mode fast's line also JIT-compiles the shared core's fused TriangleMultiplication kernel (L7, Triton) on its
first call at or above the kernel's token floor — the example's pair extent is above it, so a fast warm compiles it into Triton's cache; the
exact line compiles nothing (eager kernels under a CUDA graph). Beyond that the warm's value is the Python / CUDA-context start-up and the first
capture, both paid per process; the record is the design's manifest under ``<out>/first_design/``.
"""
from __future__ import annotations

import os
from typing import Optional

from . import design as _design, report as _report, stack

PUBLIC_REQUEST = "examples/binder_design/experiment.yaml"          # under the checkout (stock/genie3-d77ae5ac.tar.gz member; selections 01_bhrf1, n_sample 5)


def warm_line(res_out: dict) -> str:
    """``WARM <status> mode=<m> rc=<rc> first_design=<n_pdb> levers_missing=<ids|none> levers_declined=<ids|none> out=<dir>`` — every token read from
    the design's own manifest (``res_out['first_design']``, filled by `run` from design.run's return): the mode is the mode the pass RAN
    (``opt_manifest.json`` ``mode``; the mode asked for only when the pass was refused before it ran); ``levers_declined`` names planned levers that declined this request by name (L7
    under the kernel's token floor)."""
    fd = res_out.get("first_design") or {}
    return (f"{_report.PREFIX} WARM {res_out.get('status')} mode={res_out.get('mode')} rc={res_out.get('rc')} first_design={fd.get('n_pdb')} "
            f"levers_missing={','.join(fd.get('levers_missing') or []) or 'none'} levers_declined={','.join(fd.get('levers_declined') or []) or 'none'} out={res_out.get('out_dir')}")


def run(mode: Optional[str], out_dir: str, request_path: Optional[str] = None, n: int = 1, python: Optional[str] = None) -> dict:
    """The result dict: ``status`` = the design's own (ok | incomplete | failed | refused) and ``rc`` its exit code — the design's, never
    collapsed; ``mode`` = the mode the design pass ran, from its manifest; ``levers_missing`` names planned levers that could not run (the mode
    refused by name, rc 3); ``levers_declined`` planned levers that declined the request by name."""
    out_dir = os.path.abspath(out_dir)
    rep = stack.activate(mode, dry_run=True, python=python)
    res_out = {"out_dir": out_dir, "mode": rep.get("mode"), "activation": rep, "status": "refused"}
    if rep.get("would_refuse") or rep.get("reason"):
        res_out["reason"] = rep.get("reason") or "; ".join(rep["would_refuse"])
        _report.emit(f"{_report.PREFIX} WARM refused: {res_out['reason']}")
        return res_out
    if request_path is None:
        request_path = os.path.join(rep["genie3_root"], *PUBLIC_REQUEST.split("/"))
    rc, man = _design.run(request_path, os.path.join(out_dir, "first_design"), rep["mode"], tag="warm", n_sample=n, python=python)
    res_out["first_design"] = {"rc": rc, "status": man.get("status"), "out_dir": os.path.join(out_dir, "first_design"), "n_pdb": (man.get("outputs") or {}).get("n_pdb"),
                               "ready_s": (man.get("driver_pass") or man.get("stock") or {}).get("ready_s"), "levers_missing": man.get("levers_missing") or [],
                               "levers_declined": man.get("levers_declined") or [], "incomplete": man.get("incomplete"), "reason": man.get("reason")}
    res_out["mode"] = man.get("mode", rep["mode"])                     # the mode the pass ran (the manifest's); a pass refused before it started carries no mode of its own: the mode asked for
    res_out["status"] = "ok" if rc == 0 else (man.get("status") or "failed")
    res_out["rc"] = rc
    _report.emit(warm_line(res_out))
    return res_out
