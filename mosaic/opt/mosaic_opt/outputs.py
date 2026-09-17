"""The driver's own file set, read back: results.json's run fields, which levers its manifest shows, the files written.

Every route of the package ends in the kit driver's own outputs under `<out>/<tag>/` — `results.json`, `pssm_seed<s>.npz` (x0, x1, best1,
x2, best2, loss_traj), `refold_seed<s>.npz` (pae, plddt, coords) (the driver's two `np.savez` calls, `tools/public_design_run.py`); nothing is re-implemented
or re-written. ``RUN_KEYS`` are the run's result fields the package reports (the trajectory hash, the PSSM / refold digests the driver
records, the sequences, the refold metric, the numeric-state flag); ``classify()`` reads the levers the driver's manifest records
(registry.LEVERS probes).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

from .registry import LEVERS

RESULTS_NAME = "results.json"
DIGEST_KEYS: List[str] = ["x1_sha16", "best1_sha16", "x2_sha16", "best2_sha16", "refold_coords_sha16", "refold_pae_sha16", "refold_plddt_sha16"]   # the driver's PSSM / refold digests (results.json run)
RUN_KEYS: List[str] = ["loss_traj_sha256", "x0_sha16"] + DIGEST_KEYS + ["seq_stage2_x", "seq_stage2_best", "refold_iptm", "numeric_state_unchanged"]
META_KEYS_EXCLUDED = ("t_*", "host_class", "pid", "identity_key_pre", "identity_key_post", "jax_cache_listing_sha16")


def results_path(out_dir: str, tag: str) -> str:
    return os.path.join(out_dir, tag, RESULTS_NAME)


def read_results(out_dir: str, tag: str) -> Optional[dict]:
    p = results_path(out_dir, tag)
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_fields(results: Optional[dict]) -> dict:
    run = (results or {}).get("run") or {}
    return {k: run.get(k) for k in RUN_KEYS}


def run_summary(results: Optional[dict]) -> dict:
    """What the driver's `[run]` line prints, from its record: tokens, timings, loss[0], loss[-1], the run fields."""
    if not results:
        return {"status": "no results.json"}
    m, run = results.get("manifest") or {}, results.get("run") or {}
    lt = run.get("loss_traj") or []
    return {"status": results.get("status"), "tag": m.get("tag"), "n_tokens": m.get("n_tokens"), "load_path": m.get("load_path"),
            "features_source": (m.get("features") or {}).get("source"), "features_sha256": (m.get("features") or {}).get("sha256"),
            "t_load_boltz2_s": m.get("t_load_boltz2_s"), "t_features_s": m.get("t_features_s"), "t_first_iter_s": run.get("t_first_iter_s"),
            "t_iter_stage1_steady_s": run.get("t_iter_stage1_steady_s"), "t_design_opt_s": run.get("t_design_opt_s"), "t_refold_s": run.get("t_refold_s"),
            "t_process_total_s": results.get("t_process_total_s"), "loss_0": lt[0] if lt else None, "loss_last": lt[-1] if lt else None, "n_steps": len(lt),
            "boltz2_ckpt_sha256": m.get("boltz2_ckpt_sha256"), "param_fingerprint": m.get("param_fingerprint"), "versions": m.get("versions"),
            "jax_devices": m.get("jax_devices"), "gpu_nvidia_smi": (m.get("host_class") or {}).get("gpu_nvidia_smi"),
            "identity_key_pre": m.get("identity_key_pre"), "identity_key_post": m.get("identity_key_post"), "error": results.get("error"),
            **run_fields(results)}


def _get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _predicate(value, pred: str) -> bool:
    if pred == "nonempty":
        return bool(value)
    if pred.startswith("startswith:"):
        return isinstance(value, str) and value.startswith(pred.split(":", 1)[1])
    if pred.startswith("equals:"):
        return isinstance(value, str) and value == pred.split(":", 1)[1]
    if pred == "never":
        return False
    raise ValueError(f"unknown probe predicate {pred!r}")


def levers_record(results: Optional[dict]) -> dict:
    """The driver's own record of the per-step levers live in its process (`manifest["levers"]`: {<id>: {"state": "on", …}}; {} when none)."""
    return dict(((results or {}).get("manifest") or {}).get("levers") or {})


def classify(results: Optional[dict]) -> Dict[str, bool]:
    """Per lever, whether the driver's manifest shows it applied in that process (registry probes)."""
    man = (results or {}).get("manifest") or {}
    out = {}
    for name, lv in LEVERS.items():
        where, key, pred = lv.probe
        out[name] = _predicate(_get(man if where == "manifest" else {}, key), pred)
    return out


def incomplete(rc: int, killed: bool, results: Optional[dict], requested: int = 1) -> Optional[str]:
    """The driver's outputs short of the design — the driver exited 0, was not killed, and left no results file: the shared
    ``"<ok>/<requested>"`` form (``"0/1"`` for the one design a verb runs); ``None`` when the design is there or the run failed outright."""
    return f"0/{requested}" if rc == 0 and not killed and not results else None


def written_files(out_dir: str, tag: str) -> List[str]:
    return sorted(os.path.relpath(p, out_dir) for p in glob.glob(os.path.join(out_dir, tag, "*")) if os.path.isfile(p))
