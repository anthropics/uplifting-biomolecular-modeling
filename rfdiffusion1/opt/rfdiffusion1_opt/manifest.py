"""``opt_manifest.json`` — the run record written beside the outputs by ``design`` (design.py, serve.py) at the end of every pass.

Keys: ``schema``, ``package_version``, ``started_utc`` / ``finished_utc``, ``out_dir``, ``tag``, ``mode`` / ``tier`` / ``attach`` /
``line`` / ``levers_planned`` / ``env_row`` / ``flags`` (the mode's resolution, modes.py), ``overrides`` (upstream's hydra overrides
as typed), ``compose_overrides`` + ``driver_fixed`` (the values composed on the driver line, driver_run.py), ``det`` (the deterministic recipe's
level, det.py), ``cases`` (the targets as run: name, pdb, contigs, hotspots, num_designs, startnum, prefix), ``activation`` (the
activation report, stack.activate), ``weights`` (the checkpoint directory and the checkpoint names this pass loads), ``core`` (the shared core's
own block), then per route: ``stock`` (env_stripped, must_be_absent, passes[{case, rc, cmd, proof, log}]) or
``driver_passes`` [{tag, cases, rc, cmd, log, timings_json, evidence, io, numerics}] (+ ``serve`` on the packed
line), ``outputs`` (outputs.list_outputs: the requested designs on disk), ``levers_evidenced`` / ``levers_missing`` / ``forbidden_lines``,
``status`` / ``partial`` / ``partial_reason`` / ``incomplete`` / ``exit_code`` / ``reason`` (report.verdict).
"""
from __future__ import annotations

import json
import os
import time
from typing import List

from . import __version__
from .modes import DRIVER_FIXED, DRIVER_FIXED_DOC, Resolution

SCHEMA = "rfdiffusion1_opt.manifest/2"
NAME = "opt_manifest.json"


def default_checkpoint() -> str:
    """The checkpoint every line loads unless a case names another: stock/PINS.json ``weights.checkpoint``."""
    from . import stack
    return stack.pins()["weights"]["checkpoint"]


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def core_block() -> dict:
    """The shared core's own report of itself (opt_core.manifest.core_block) for the opt_core this process imported — passed through as
    read, never parsed for specific fields here."""
    from opt_core.manifest import core_block as _core_block
    return _core_block()


def checkpoint_record(weights_dir: str | None, cases: List[dict]) -> dict:
    """The checkpoint directory and the checkpoint files this pass loads (the default one, plus any per-case `ckpt` / `ckpt_path` override),
    each with `present` — a user's weights are always accepted."""
    names = [default_checkpoint()] + sorted({c["ckpt"] for c in cases if c.get("ckpt")})
    files = {n: {"present": bool(weights_dir) and os.path.isfile(os.path.join(weights_dir, n))} for n in names}
    for c in cases:
        if c.get("ckpt_path"):
            files[c["ckpt_path"]] = {"present": os.path.isfile(c["ckpt_path"]), "override_path": True}
    return {"dir": weights_dir, "files": files}


def start(activation: dict, res: Resolution, cases: List[dict], out_dir: str, tag: str, *, det: bool = False) -> dict:
    return {"schema": SCHEMA, "package_version": __version__, "started_utc": _utc(), "out_dir": out_dir, "tag": tag,
            "mode": res.mode, "overrides": list(res.settings.stock_overrides), "tier": res.tier, "attach": res.attach, "line": res.line,
            "levers_planned": list(res.levers), "env_row": dict(res.env), "flags": list(res.flags), "override_driver_flags": list(res.settings.driver_flags),
            "served": bool(res.served),
            "compose_overrides": list(res.settings.compose_overrides) if res.attach == "driver" else [],   # composed in place of the driver's constants and carried onto its configuration (driver_run.py)
            "driver_fixed": {"keys": list(DRIVER_FIXED), "source": DRIVER_FIXED_DOC} if res.attach == "driver" else None,
            "det": int(bool(det)), "activation": dict(activation), "cases": cases,
            "weights": checkpoint_record(activation.get("weights"), cases), "core": core_block(), "status": "running"}


def finish(man: dict, out_dir: str) -> str:
    man["finished_utc"] = _utc()
    if "driver_passes" in man:
        applied, missing, forb = set(), set(), 0
        for p in man["driver_passes"]:
            ev = p.get("evidence") or {}
            applied.update(ev.get("applied", [])); missing.update(ev.get("missing", [])); forb += len(ev.get("forbidden", []))
        man["levers_evidenced"] = sorted(applied)
        from . import stack as _stack
        man["activation"]["proven"] = bool(_stack.status().get("proven")); man["activation"]["proven_by"] = _stack.status().get("proven_by")   # whether, and by which evidence, the ACTIVE documented line was printed
        man["levers_missing"] = sorted(missing)
        man["forbidden_lines"] = forb
    path = os.path.join(out_dir, NAME)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
    return path
