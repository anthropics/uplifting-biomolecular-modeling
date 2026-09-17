"""opt_manifest.json — the record written beside the outputs of every ``design`` / ``warm`` run: the activation report (the same report
the startup line prints), the stock pin record with the weights file the pass loaded (``pins.detail.weights``: digest and verdict), the
stock options given, the inputs, the outputs (the file listing and the counts), the kit's own evidence lines and the exit code.
``read(out_dir)`` loads it back; ``FILENAME`` is the one name every reader uses.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
from typing import Optional

from opt_core import gates as core_gates, manifest as core_manifest

from . import __version__, det, stack

FILENAME = "opt_manifest.json"
SCHEMA = "proteinmpnn_opt/opt_manifest/1"
EXCLUDED_REPORT_KEYS = ("tree_home",)


def build(report: Optional[dict], command: str, argv: Optional[list] = None, exit_code: Optional[int] = None, stock_args: Optional[list] = None,
          inputs: Optional[dict] = None, outputs: Optional[dict] = None, kit: Optional[dict] = None, stock: Optional[dict] = None,
          wall_s: Optional[float] = None, extra: Optional[dict] = None) -> dict:
    rep = {k: v for k, v in (report or {}).items() if k not in EXCLUDED_REPORT_KEYS}
    pins = rep.get("pins") or {}
    man = {
        "schema": SCHEMA,
        "written_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": command,
        "argv": list(argv) if argv is not None else None,
        "exit_code": exit_code,
        "wall_s": round(wall_s, 3) if wall_s is not None else None,
        "mode": rep.get("mode"),
        "variant": rep.get("variant"),
        "route": rep.get("route"),
        "line": rep.get("line"),
        "active": bool(rep.get("active")),
        "applied": rep.get("applied"),
        "levers_applied": list(rep.get("levers_applied") or []),
        "levers_fallback": list(rep.get("levers_fallback") or []),
        "levers_unavailable": list(rep.get("levers_unavailable") or []),
        "levers_unobserved": list(rep.get("levers_unobserved") or []),         # requested, no field in the worker's record: named, not judged
        "partial": list(rep.get("partial") or []),                             # the requested levers without evidence of application
        "allow_partial": bool(rep.get("allow_partial")),
        "partial_detection": rep.get("partial_detection"),                    # `none (<why>)` when the run leaves no lever record to judge
        "opted_out": rep.get("opted_out") or [],                              # probe-gated levers the command line left out by name (--hybrid_gemm 0)
        "refused": rep.get("refused") or [],                                  # a pass the mode cannot serve, refused by name before any process: the options named (reason carries the mechanisms)
        "gated": rep.get("gated") or {},                                       # a kit gate that switched a lever off and said so: not partial
        "bb_batch": rep.get("bb_batch"),
        "probe": rep.get("probe"),
        "reason": rep.get("reason"),
        "gpu": rep.get("gpu"),
        "target_gpu": rep.get("target_gpu"),
        "gpu_matches_target": rep.get("gpu_matches_target"),
        "stack_key": rep.get("stack_key"),
        "proteinmpnn_version": rep.get("proteinmpnn_version"),
        "package_version": rep.get("package_version") or __version__,
        "core": core_manifest.core_block(),                                   # the opt_core this pass imported: version and package location
        "det": det.describe(),                                                # the environment recipe in the core's one-line form (det=0: argument values only)
        "pins": {"pinned": pins.get("pinned"), "detail": pins.get("detail"), "findings": pins.get("findings"), "notes": pins.get("notes")},   # detail.weights: the weights file the pass loaded, its verdict; notes: the checkout against the pin (reported, never a gate)
        "kits": rep.get("kits"),
        "stock_args": list(stock_args) if stock_args is not None else None,        # the stock command line's own options as given (settings.parse); absent = upstream's defaults
        "inputs": inputs,
        "outputs": outputs,
        "kit": kit,
        "stock": stock,
        "env": {stack.ENV_MODE: os.environ.get(stack.ENV_MODE), stack.ENV_VARIANT: os.environ.get(stack.ENV_VARIANT),
                stack.ENV_ALLOW_PARTIAL: os.environ.get(stack.ENV_ALLOW_PARTIAL),
                stack.ENV_MPNN_DIR: os.environ.get(stack.ENV_MPNN_DIR)},
        "python": platform.python_version(),
        "torch": core_gates.dist_version("torch"),
        "numpy": core_gates.dist_version("numpy"),
        "activation_report": rep,
    }
    if extra:
        man.update(extra)
    return man


def write(out_dir: str, report: Optional[dict], **kw) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILENAME)
    man = build(report, **kw)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read(out_dir_or_path: str) -> dict:
    path = out_dir_or_path if out_dir_or_path.endswith(".json") else os.path.join(out_dir_or_path, FILENAME)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
