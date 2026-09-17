"""``opt_manifest.json`` beside the outputs: the activation report of the run, the installed-tree proof, the overlay, the settings, the command.

Written by ``design``, ``warm`` (their child process writes the activation part at activation; the parent completes it with the
exit code and the writer's ``timing.json``). One schema for every mode: both arms
carry the installed tree's state and digest (proven the pinned upstream's), the stock arm ``stock_env_proof`` (every kit and package
switch proven absent), the exact arm ``overlay`` (module -> kit file and digest); ``core`` is the shared core the package imported
(``opt_core.manifest.core_block``).
The lever fields: ``levers_applied`` at activation; ``levers_fallback`` (the partial list) / ``fallback_reasons`` /
``partial`` completed at the child's exit from the exit tally (``activate.completion``), with the tally itself under ``exit_tally``;
``lever_gates`` ({lever: {gate: calls}}: a lever's declared input gate met at call time — upstream's lines for that step, the same outputs;
never ``partial``); ``incomplete`` (``<n>/<m>``, the parent's
count of designs written against the request; None when complete) is the other named state — exit 1, never conflated with ``partial``.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import sys
from typing import Optional

from . import __version__, stack

FILENAME = "opt_manifest.json"
SCHEMA = "caliby_opt/opt_manifest/2"


def _torch_version() -> Optional[str]:
    t = sys.modules.get("torch")
    return getattr(t, "__version__", None) if t is not None else None


def _core_block() -> dict:
    """The shared core this process imports (``opt_core.manifest.core_block``). Every manifest writer (the design children, ``warm``)
    passed the core pin gate first, so the pinned core is importable here; refusals other than the core's still write a manifest."""
    from opt_core.manifest import core_block
    return core_block()


def build(report: Optional[dict], *, command: Optional[str] = None, argv: Optional[list] = None, exit_code: Optional[int] = None,
          settings: Optional[dict] = None, extra: Optional[dict] = None) -> dict:
    rep = dict(report or {})
    gpu = rep.get("gpu")
    man = {
        "schema": SCHEMA,
        "written_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": rep.get("mode"),
        "variant": rep.get("variant"),
        "active": bool(rep.get("active")),
        "row": rep.get("row_label"),
        "row_source": rep.get("row_source"),
        "switches": dict(rep.get("switches") or {}),
        "lever_classes": rep.get("lever_classes"),
        "levers_applied": list(rep.get("levers_applied") or []),
        "levers_fallback": list(rep.get("levers_fallback") or []),
        "fallback_reasons": dict(rep.get("fallback_reasons") or {}),
        "levers_unavailable": list(rep.get("levers_unavailable") or []),
        "partial": bool(rep.get("partial")),
        "lever_gates": dict(rep.get("lever_gates") or {}),
        "incomplete": rep.get("incomplete"),
        "row_alternatives": list(rep.get("row_alternatives") or []),
        "clean_workers": rep.get("clean_workers"),
        "tree_state": rep.get("tree_state"),
        "site_packages_state": rep.get("site_packages_state"),
        "tree_digest": rep.get("tree_digest"),
        "overlay": rep.get("overlay") or {},
        "core": _core_block(),
        "reason": rep.get("reason"),
        "package_version": rep.get("package_version") or __version__,
        "upstream": rep.get("upstream") or stack.upstream_versions(),
        "caliby_version": rep.get("caliby_version"),
        "gpu": gpu if isinstance(gpu, dict) else ({"name": gpu} if gpu else None),
        "target_gpu": rep.get("target_gpu"),
        "target_gpu_note": rep.get("target_gpu_note"),
        "stack_key": rep.get("stack_key"),
        "compiler": rep.get("compiler"),
        "weights": rep.get("weights"),
        "stock_env_proof": rep.get("stock_env_proof"),
        "settings": settings,
        "command": command,
        "argv": list(argv) if argv is not None else None,
        "exit_code": exit_code,
        "caliby_opt_env": os.environ.get(stack.ENV_MODE),
        "caliby_variant_env": os.environ.get(stack.ENV_VARIANT),
        "python": platform.python_version(),
        "torch": _torch_version() or (gpu or {}).get("torch") if isinstance(gpu, dict) else _torch_version(),
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
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def update(out_dir: str, **fields) -> str:
    """Merge fields into an existing manifest (the parent completes the child's manifest after the run)."""
    path = os.path.join(out_dir, FILENAME)
    man = read(path) if os.path.isfile(path) else build(None)
    man.update({k: v for k, v in fields.items() if v is not None})
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
        fh.write("\n")
    os.replace(tmp, path)
    return path
