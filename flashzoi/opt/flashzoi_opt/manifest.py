"""opt_manifest.json — what was active when a set of outputs was produced, written beside the outputs (the activation report's keys
plus this model's axes). Keys:

  schema, written_at, mode, active, components_applied, components_fallback, components_unavailable, partial, partial_detection (the evidence
  the partial verdict is read from: stack.PARTIAL_DETECTION), allow_partial (`--allow-partial` given: a partial run proceeds
  to its own exit; without it a partial run exits 3, the outputs and this file kept), incomplete (`<ok>/<items>` when an item failed:
  the outputs are short of the request, exit 1 — never conflated with partial), reason, package_version, gpu,
  command, argv, exit_code, python, torch, flashzoi_opt_env, stack_key, kit_arm, pins (the stack + package pins as read from
  stock/PINS.json), weights (per replicate: repo, revision, model.safetensors sha256 as asserted at load), settings (settings.py
  as_dict + the numerics read back from torch), det (the deterministic recipe as applied, or null), items (n, ok, failed),
  replicates (per attached model: apply_s = the kit's KitRunner.apply_s, describe = the kit's describe(), effective_flags, counts,
  jit_after_job), activation_report (the whole report, in-memory objects dropped).

The stock route (opt/flashzoi_opt/stock_pred.py) writes the file itself: mode "off", the script, argv, det, numerics read back, weights, versions, gpu and
the environment proof under `stock_env_proof` — one run record beside the outputs on every route.
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import sys

SCHEMA = "flashzoi_opt.opt_manifest/1"
FILENAME = "opt_manifest.json"
DROP_KEYS = ("resolution",)                                   # in-memory objects that do not serialise


def _torch_version():
    m = sys.modules.get("torch")
    return getattr(m, "__version__", None) if m is not None else None


def build(report: dict | None, *, command: str, argv=None, exit_code=None, settings: dict | None = None, det: dict | None = None,
          weights: list | None = None, items: dict | None = None, replicates: list | None = None, extra: dict | None = None) -> dict:
    rep = {k: v for k, v in (report or {}).items() if k not in DROP_KEYS}
    gpu = rep.get("gpu")
    man = {"schema": SCHEMA, "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "mode": rep.get("mode"), "active": bool(rep.get("active")), "components_applied": rep.get("components_applied") or [],
           "components_fallback": rep.get("components_fallback") or [], "components_unavailable": rep.get("components_unavailable") or [], "partial": bool(rep.get("partial")),
           "partial_detection": rep.get("partial_detection"),
           "reason": rep.get("reason"), "package_version": rep.get("package_version"), "gpu": gpu if isinstance(gpu, dict) else ({"name": gpu} if gpu else None),
           "command": command, "argv": list(argv) if argv is not None else None, "exit_code": exit_code, "python": platform.python_version(),
           "torch": _torch_version(), "flashzoi_opt_env": os.environ.get("FLASHZOI_OPT"), "stack_key": rep.get("stack_key"), "kit_arm": rep.get("kit_arm"),
           "device_class": rep.get("device_class"), "drift": list(rep.get("drift") or []),   # the environment as named on the ACTIVE line: pinned / record / unpinned class, drift items
           "pins": rep.get("pins"), "weights": weights, "settings": settings, "det": det, "items": items, "replicates": replicates, "activation_report": rep}
    if extra:
        man.update(extra)
    return man


def write(out_dir: str, report: dict | None, **kw) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILENAME)
    man = build(report, **kw)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read(out_dir_or_path: str) -> dict:
    path = out_dir_or_path if out_dir_or_path.endswith(".json") else os.path.join(out_dir_or_path, FILENAME)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
