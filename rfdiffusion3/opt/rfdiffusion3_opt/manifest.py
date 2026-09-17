"""opt_manifest.json — what was active when a set of designs was produced, written beside the outputs (the shared report keys, the
core block `opt_core.manifest.core_block`, plus this kit's axes: the interpreter and its file state, the kit file shas, the checkpoint sha).

``write(out_dir, report, ...)`` records the activation report (mode, environment delta, the shared lever keys `levers_applied` /
`levers_fallback` / `levers_unavailable` / `partial` and the exit rule's record — `partial_reason` (the engine's own reason a lever
has no evidence of application) and `allow_partial` (true when `--allow-partial` / RFDIFFUSION3_OPT_ALLOW_PARTIAL=1 let a partial run
proceed; a partial run without it exits 3, `exit_code`) — interpreter label + the ten file states, kit version and sha check, pins, GPU),
the package and upstream versions, the checkpoint (path, bytes, sha256 against stock/PINS.json — the digest from the package's cache
when the file is unchanged since it was last hashed, `sha256_source`), the line's overrides and the
literal command line (`settings.values`: the key=value tokens as passed; `settings.route`: the route of modes.ROUTES the line runs under), the spec (the line's
`inputs=` value; the path and size when it is a file), the output files by name, size and sha256 (`outputs`; this manifest itself is
not one), the stock environment proof of an `off` run, the KERNELS census of the pass (`kernels`: route, the accelerator words, the
printed line and its stack facts, the guard's refusal if any, dynamo's counters, upstream's per-roll-out seconds and the per-item stamps,
the atom-attention path counts, the checkpoint the engine loaded), and a UTC timestamp. Everything volatile about a run lives here and
nowhere else.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import platform
import sys
from typing import Dict, List, Optional, Tuple

from . import registry
from .registry import CACHE_ENV, checkpoint_sha256, digest_cache_path   # noqa: F401 — the checkpoint digest and its cache live in the core-free `registry`; served here for the manifest's callers

SCHEMA = "rfdiffusion3_opt.manifest/3"
FILENAME = "opt_manifest.json"
EXCLUDED_REPORT_KEYS = ("resolution",)
OUTPUT_GLOBS = ("*.cif.gz", "*.json")
NOT_OUTPUTS = (FILENAME,)   # the package's one file beside the designs: never listed as an output
LEVER_KEYS = ("levers_applied", "levers_fallback", "levers_unavailable", "partial", "partial_reason", "allow_partial")   # the shared report keys + the exit rule's record, copied to the manifest's top level
_LEVER_DEFAULTS = {"partial": False, "partial_reason": None, "allow_partial": False}                 # for a report without the key (a stock run)


def _dist_version(name: str) -> Optional[str]:
    from opt_core.gates import dist_version
    return dist_version(name)


def package_version() -> Optional[str]:
    v = _dist_version("rfdiffusion3_opt") or _dist_version("rfdiffusion3-opt")
    if v:
        return v
    pkg = sys.modules.get("rfdiffusion3_opt")
    return getattr(pkg, "__version__", None) if pkg is not None else None


def upstream_versions() -> dict:
    return {"rc-foundry": _dist_version("rc-foundry"), "torch": _dist_version("torch"), "triton": _dist_version("triton"),
            "atomworks": _dist_version("atomworks"), "biotite": _dist_version("biotite"), "lightning": _dist_version("lightning"),
            "numpy": _dist_version("numpy"), "hydra-core": _dist_version("hydra-core")}


def _core_block() -> dict:
    """The core this process imports, whatever opt_core.manifest.core_block currently reports (at minimum version, package_dir);
    the pin gate's own record (pinned vs imported) is in the activation report."""
    from opt_core.manifest import core_block
    return core_block()


def checkpoint_info(path: Optional[str], pins: Optional[dict] = None, hash_it: bool = True) -> Optional[dict]:
    if not path:
        return None
    info: Dict[str, object] = {"path": os.path.abspath(path), "exists": os.path.isfile(path)}
    if info["exists"]:
        info["bytes"] = os.path.getsize(path)
        if hash_it:
            info["sha256"], info["sha256_source"] = checkpoint_sha256(path)
    if pins:
        want = pins.get("checkpoint", {})
        info["pinned_sha256"] = want.get("sha256")
        info["source_filename"] = want.get("source_filename")
        if info.get("sha256"):
            info["is_pinned_checkpoint"] = info["sha256"] == want.get("sha256")
    return info


def spec_info(spec: Optional[str]) -> Optional[dict]:
    """The line's `inputs=` value as given; the absolute path and size when it names a file (anything else — upstream's comma list, a name —
    is upstream's to interpret)."""
    if not spec:
        return None
    info: Dict[str, object] = {"value": spec, "is_file": os.path.isfile(spec)}
    if info["is_file"]:
        info["path"], info["bytes"] = os.path.abspath(spec), os.path.getsize(spec)
    return info


def output_listing(out_dir: str) -> List[dict]:
    files = []
    for pat in OUTPUT_GLOBS:
        for p in sorted(glob.glob(os.path.join(out_dir, pat))):
            if os.path.basename(p) in NOT_OUTPUTS:
                continue
            files.append({"file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": registry.sha256_of(p)})
    return files


def build(report: Optional[dict], *, settings: Optional[dict] = None, command: Optional[List[str]] = None, argv: Optional[List[str]] = None,
          exit_code: Optional[int] = None, checkpoint: Optional[str] = None, spec: Optional[str] = None, out_dir: Optional[str] = None,
          pins: Optional[dict] = None, stock_env_proof: Optional[dict] = None, extra: Optional[dict] = None, hash_checkpoint: bool = True,
          kernels: Optional[dict] = None) -> dict:
    rep = {k: v for k, v in (report or {}).items() if k not in EXCLUDED_REPORT_KEYS}
    man = {
        "schema": SCHEMA,
        "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "mode": rep.get("mode"),
        "active": rep.get("active"),
        "env_delta": rep.get("env"),
        **{k: rep.get(k, _LEVER_DEFAULTS.get(k, [])) for k in LEVER_KEYS},
        "interpreter": rep.get("interpreter"),
        "kit": rep.get("kit"),
        "kit_version": rep.get("kit_version"),
        "core": _safe(_core_block),
        "package_version": rep.get("package_version") or package_version(),
        "upstream": upstream_versions(),
        "pins": rep.get("pins"),
        "gpu": rep.get("gpu"),
        "checkpoint": checkpoint_info(checkpoint, pins, hash_checkpoint),
        "spec": spec_info(spec),
        "settings": settings,
        "command": command,
        "argv": list(argv) if argv is not None else None,
        "exit_code": exit_code,
        "outputs": output_listing(out_dir) if out_dir else None,
        "stock_env_proof": stock_env_proof,
        "kernels": kernels,                                  # the KERNELS census of the pass (census.record(): route, words, the line, refused, dynamo counters, upstream's per-roll-out clock, items, attn_path, ckpt_path)
        "rfdiffusion3_opt_env": os.environ.get("RFDIFFUSION3_OPT"),
        "python": platform.python_version(),
        "activation_report": rep,
    }
    if extra:
        man.update(extra)
    return man


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def write(out_dir: str, report: Optional[dict], **kw) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILENAME)
    man = build(report, out_dir=kw.pop("out_dir", out_dir), **kw)
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
