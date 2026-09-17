"""``opt_manifest.json`` beside the outputs: the activation report of the run plus what identifies the run — the pins, the GPU,
the stack key, the configure line, the seed(s), the command line, the exit code, the timing records the lever modules wrote, the
accelerator census account (``kernels``), the core block (``core``: the opt_core version and package dir the run imported) — one schema
for every command and every mode (the stock arm carries its clean-environment record under ``stock_env_proof``). ``partial`` lists the
levers of the mode that could not run on the box (``[]`` when the activation is whole)."""
from __future__ import annotations

import json
import os
import platform
import sys
from typing import Optional

from opt_core import manifest as core_manifest

from . import modes, stack

FILENAME = "opt_manifest.json"
SCHEMA = 1


def build(rep: Optional[dict], command: str, argv=None, exit_code=None, extra: Optional[dict] = None, designs: Optional[dict] = None) -> dict:
    rep = dict(rep or {})
    p = stack.pins()
    man = {
        "schema": SCHEMA,
        "package": "boltzgen_opt",
        "package_version": rep.get("package_version") or stack.package_version(),
        "core": core_manifest.core_block(),                               # the opt_core this run imported: version + package dir
        "mode": rep.get("mode"),
        "form": rep.get("form"),
        "active": bool(rep.get("active")),
        "levers_applied": rep.get("levers_applied"),
        "levers_fallback": rep.get("levers_fallback"),
        "fallback_reasons": rep.get("fallback_reasons"),                # why each lever that fell back did (the classifier's own words)
        "levers_unavailable": rep.get("levers_unavailable"),
        "partial": stack.partial_levers(rep),                             # the levers that fell back when the activation is partial; [] when whole
        "designs": designs,                                                # the design census (design.designs_census; the DESIGNS line) — its one place: step, requested, produced, oom_skipped (the batches upstream's own handler skipped on a CUDA out-of-memory, every mode), featurizer_skipped, stale, reuse, output_dir, config; null before a run
        "switches": rep.get("switches"),
        "boltzgen": {"version": rep.get("boltzgen_version") or stack.boltzgen_version(), "pin": p["boltzgen_version"]},
        "gpu": rep.get("gpu"),
        "gpu_gate": rep.get("gpu_gate"),
        "stack_key": rep.get("stack_key"),
        "command": command,
        "argv": list(argv) if argv is not None else None,
        "exit_code": exit_code,
        "boltzgen_opt_env": os.environ.get(modes.ENV),
        "cache": os.environ.get(stack.ENV_CACHE),
        "python": platform.python_version(),
        "torch": rep.get("torch") or getattr(sys.modules.get("torch"), "__version__", None),
        "activation_report": rep,
    }
    if extra:
        man.update(extra)
    return man


def write(out_dir: str, rep: Optional[dict], **kw) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILENAME)
    man = build(rep, **kw)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read(out_dir_or_path: str) -> dict:
    path = out_dir_or_path if out_dir_or_path.endswith(".json") else os.path.join(out_dir_or_path, FILENAME)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
