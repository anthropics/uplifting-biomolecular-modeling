"""The run's record, assembled in memory at the end of every route (pred, warm): the activation report, the worker evidence,
the settings, the inputs and the outputs account. Nothing is written beside the outputs — the printed lines (report.py) are the kit's one
surface; the record is kept for in-process callers (``LAST``, the latest run of this process) and the kit's tests."""
from __future__ import annotations

import datetime
import hashlib
import os
import platform
import sys
from typing import Optional

from opt_core import manifest as core_manifest

from . import __version__, report as rep, stack


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path: str) -> Optional[str]:
    try:
        return stack.sha256_file(path)
    except OSError:
        return None


def build(route: str, mode: str, *, report: Optional[dict] = None, evidence: Optional[dict] = None, inputs: Optional[list] = None,
          outputs: Optional[dict] = None, settings: Optional[dict] = None, det: Optional[dict] = None,
          staged: Optional[dict] = None, command: Optional[list] = None, rc: Optional[int] = None, extra: Optional[dict] = None,
          kernels_census: Optional[dict] = None) -> dict:
    from . import modes
    row = modes.resolve(mode)
    gpu = stack.gpu_probe(); gpu_class = stack.gpu_class_check(gpu, os.environ.get("MODEL_OPT_TARGET_GPU"))
    m = {"package": "boltz2_opt", "package_version": __version__, "core": core_manifest.core_block(), "written_utc": _utc(), "route": route, "mode": row["mode"], "tier": row["tier"],
         "levers": list(row["levers"]), "env_row": dict(row["env"]), "kernels": row.get("kernels"), "evidence_env": dict(modes.EVIDENCE_ENV.get(row["mode"], {})),
         "pinned_route": modes.PINNED_ROUTE, "python": sys.version.split()[0],
         "platform": platform.platform(), "gpu": gpu, "target_gpu": gpu_class["target"], "gpu_class": gpu_class, "gpu_supported": gpu_class["supported"],
         "boltz_cache": stack.cache_dir(), "report": report, "evidence": evidence,
         "settings": settings, "det": det, "staged_kit_files": staged, "inputs": inputs, "outputs": outputs,
         "command": command, "rc": rc, "kernels_census": kernels_census, "tally": rep.tally_snapshot()}
    if extra:
        m.update(extra)
    return m


LAST: Optional[dict] = None                                          # the latest run's record of this process (None: no route completed yet)


def record(m: dict) -> dict:
    """Keep ``m`` as this process's latest run record (in memory only) and return it."""
    global LAST
    LAST = m
    return m


def input_records(items: list) -> list:
    return [{"name": it["name"], "yaml": os.path.abspath(it["yaml"]), "sha256": sha256_of(it["yaml"])} for it in items]


def output_records(out_dir: str, items: list, seeds: list) -> dict:
    recs = {}
    for it in items:
        for s in seeds:
            files = stack.output_files(out_dir, it["name"], s)
            d = os.path.join(out_dir, "by_seed", it["name"], f"s{s}")
            recs[f"{it['name']}/s{s}"] = {f: sha256_of(os.path.join(d, f)) for f in files}
    return recs
