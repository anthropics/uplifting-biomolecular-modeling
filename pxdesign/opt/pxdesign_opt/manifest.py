"""``opt_manifest.json`` — the kit's one file beside upstream's own outputs, written by the design verb on every mode (monitoring only:
nothing in it is read back or judged): the activation report (or the stock proof's summary for mode off), the upstream options (values, the
caller's given words, seeds), the command line, the exit code, the design counts in upstream's layout (outputs.py; no digests), the timings,
the records under `stamps` (batch facts, the built model's LayerNorm class, the KERNELS line's fields as printed — stamps.py) and
the stack — the box as found with the shared core's block (`opt_core.manifest.stack_block`: python, torch, cuda, triton, the
GPU probe, the JIT-cache key, the core's version). Written atomically (`opt_core.manifest.write`)."""
from __future__ import annotations

import os
import platform
import sys
import time

from opt_core import manifest as _core_manifest

from .outputs import MANIFEST_NAME

FILENAME = MANIFEST_NAME
SCHEMA = "pxdesign_opt.opt_manifest.v3"
NOT_RECORDED = ("logged", "exit_judged", "exit_problems")       # the activation report's in-process latches (line de-duplication, the exit verdict's stand-down flag and its
                                                                # scratch list) — not run evidence; the census sentences are recorded once, as `package_gate`


def _torch_version():
    t = sys.modules.get("torch")
    return getattr(t, "__version__", None) if t else None


def stack_block(rep: dict | None) -> dict:
    """The box as found: the core's stack block keyed by this kit's GPU probe and JIT-cache key (nothing imported for it)."""
    rep = rep or {}
    gpu = rep.get("gpu") or {}
    key = os.environ.get("MODEL_OPT_STACK_KEY")                  # the key the JIT caches are keyed by (configs/h100.env), when set
    if key is None and gpu.get("torch"):
        from .modes import jit_cache_key
        key = jit_cache_key(gpu.get("torch"), gpu.get("cuda"), gpu.get("cc"), strict=False)   # display only (the box as found): a part the probe could not establish reads `unknown`
    return _core_manifest.stack_block(gpu=gpu or None, torch_version=_torch_version() or gpu.get("torch"), cuda=gpu.get("cuda"), stack_key=key)


def build(report: dict | None, *, mode: str, command: str, argv=None, exit_code=None, options=None, outputs=None, timings=None,
          stock_proof=None, stamps=None, extra=None) -> dict:
    from . import __version__
    rep = {k: v for k, v in dict(report or {}).items() if k not in NOT_RECORDED}
    man = {
        "schema": SCHEMA,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package": "pxdesign_opt",
        "package_version": rep.get("package_version") or __version__,
        "mode": mode,
        "tier": rep.get("tier"),
        "active": bool(rep.get("active")),
        "levers_planned": rep.get("levers_planned") or [],
        "levers_applied": rep.get("levers_applied") or [],
        "levers_fallback": rep.get("levers_fallback") or [],
        "levers_skipped": rep.get("levers_skipped") or [],
        "env_exported": rep.get("env") or {},
        "applications": rep.get("applications") or [],
        "reason": rep.get("reason"),
        "upstream": rep.get("upstream"),
        "gpu": rep.get("gpu"),
        "stack": stack_block(rep),
        "options": options,
        "timings": timings,
        "outputs": outputs,
        "stock_env_proof": stock_proof,
        "stamps": stamps,
        "command": command,
        "argv": list(argv) if argv is not None else None,
        "exit_code": exit_code,
        "pxdesign_opt_env": os.environ.get("PXDESIGN_OPT"),
        "python": platform.python_version(),
        "torch": _torch_version(),
        "activation_report": rep or None,
    }
    if extra:
        man.update(extra)
    return man


def write(out_dir: str, report: dict | None, **kw) -> str:
    path = os.path.join(out_dir, FILENAME)
    return _core_manifest.write(path, build(report, **kw))


def read(out_dir_or_path: str) -> dict:
    path = out_dir_or_path if out_dir_or_path.endswith(".json") else os.path.join(out_dir_or_path, FILENAME)
    return _core_manifest.read(path)
