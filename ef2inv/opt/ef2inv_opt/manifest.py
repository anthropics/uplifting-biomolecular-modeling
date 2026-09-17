"""``opt_manifest.json`` — what ran, from which bytes, with what evidence.

Fields: ``mode`` (the table row), ``stack`` (python / torch / cuda / device from run.json, the config's stack key), ``pins`` (the upstream file
presence and the kit's version/file-count facts), ``files_on_path`` (sha256 of each file of the kit the arm could import: the cookbook it drove, the ``k/`` modules,
the `modal` stand-in), ``upstream_fix`` (the IDs the arm applied under ``--upstream-fix``; ``[]`` without the flag), ``case``, ``settings`` + ``deviations`` (the run's effective ``model_switches`` — chunk_size, kernel_backend: the mode's values or the user's — each named there, user flags also under ``overrides`` with ``COMPILE``), ``det``, ``attention`` (the arm's attention words and the flags / callables behind them, attention.py), ``activation`` (the before/after report of a kit arm; ``stock_lever_lines`` of a
stock arm), ``refusals`` / ``fallback`` / ``partial`` (named; a non-empty list = exit 3 unless ``allow_partial``), ``timing``, ``outputs``
(name -> sha256, bytes), ``launch`` (exit code, wall, argv, the environment proof).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

from . import outputs as O
from .modes import Mode, kit_paths, SDK_STANDIN
FASTKIT_MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastkit.py")


def _sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def files_on_path(mode: Mode, model_opt: str, cookbook_stock: str) -> Dict[str, str]:
    kp = kit_paths(model_opt)
    out = {}
    out[cookbook_stock] = _sha(cookbook_stock)                          # the one stock file every mode runs
    out[SDK_STANDIN] = _sha(SDK_STANDIN)                                # the `modal` stand-in every arm process registers (stock_design.prepare_arm_process)
    if mode.is_kit:
        out[FASTKIT_MODULE] = _sha(FASTKIT_MODULE)                          # the kit's cookbook-level integration installed on the module (fastkit.py)
        for fn in sorted(os.listdir(kp["k_dir"])):
            if fn.endswith(".py"):
                p = os.path.join(kp["k_dir"], fn); out[p] = _sha(p)
    return out


def switch_records(mode: Mode, run: dict, switches: Optional[dict], user_overrides: Optional[dict]):
    """(model_switches, overrides, deviation words). ``model_switches`` = the effective chunk_size / kernel_backend of the run — what the arm
    applied (run.json ``model_switches``) over what the launcher passed (``switches``), "shipped" = no setter call; ``overrides`` = a kit mode's
    user flags that differ from its values (``user_overrides``; a flag on off is a setting, never an override) plus the arm's own ``overrides``; the words name every effective switch
    that is not shipped as the mode's value (the stock definition on off / exact) or as a user override."""
    eff = {"chunk_size": mode.chunk_size, "kernel_backend": mode.kernel_backend}
    eff.update(switches or {}); eff.update(run.get("model_switches") or {})
    user = dict(user_overrides or {})
    overrides = dict(run.get("overrides") or {}); overrides.update(user)
    words = [f"override {k}={v!r} (the design verb's flag)" for k, v in (run.get("overrides") or {}).items() if k not in eff]
    for k in ("chunk_size", "kernel_backend"):
        if k in user:
            words.append(f"override {k}={user[k]!r} (user flag; the mode's value is {getattr(mode, k)!r})")
        elif eff[k] != "shipped":
            words.append(f"model switch {k}={eff[k]!r} (the {mode.name} mode's value; stock = chunk None + cuequivariance backend)")
    return eff, (overrides or None), words


def write(design_dir: str, mode: Mode, model_opt: str, pins: dict, cookbook_stock: str, launch: dict, stack_key: Optional[str], facts: Optional[dict] = None,
          allow_partial: bool = False, switches: Optional[dict] = None, user_overrides: Optional[dict] = None) -> dict:
    run = {}
    rj = os.path.join(design_dir, "run.json")
    if os.path.isfile(rj):
        run = json.load(open(rj))
    act = {}
    aj = os.path.join(design_dir, "activation.json")
    if os.path.isfile(aj):
        act = json.load(open(aj))
    model_switches, overrides, switch_words = switch_records(mode, run, switches, user_overrides)
    refusals = list(act.get("before", {}).get("refusals", [])) + list(act.get("after", {}).get("refusals", [])) + \
        ([f"stock arm log carries kit lines: {act['stock_lever_lines'][:3]}"] if act.get("stock_lever_lines") else [])
    missing = O.complete(design_dir) if launch.get("exit_code") == 0 else O.complete(design_dir)
    missing = [m for m in missing if m != "opt_manifest.json"]
    man: Dict[str, Any] = {
        "engine": "ef2inv", "mode": {"name": mode.name, "kit_switch": mode.kit_switch, "levers": mode.levers,
                                     "numerics_class": mode.numerics_class, "tier": mode.tier},
        "stack": {"key": stack_key, **{k: run.get("launch", {}).get(k) for k in ("python", "torch", "cuda", "device", "nvidia_smi_memory_total_mib", "torch_total_mib")},   # key = torch<v>-cu<NNN>-sm<cc> of the running stack (modes.jit_cache_key), beside its versions
                  "hardware_gate": (facts or {}).get("hardware")},
        "weights": (facts or {}).get("weights"),
        "stock_patch": next(("upstream_bug_0002" for p in (run.get("patches") or []) if p.get("applied") and p["name"] == "upstream_bug_0002_transition_addmm_dtype"), None),
        "stock_patch_function_sha256": next((p.get("function_sha256") for p in (run.get("patches") or []) if p.get("applied") and p["name"] == "upstream_bug_0002_transition_addmm_dtype"), None),
        "attention": run.get("attention"),
        "pins": {"upstream": pins.get("upstream")},
        "files_on_path": files_on_path(mode, model_opt, cookbook_stock),
        "case": run.get("case"), "settings": run.get("settings"), "deviations": list((run.get("settings") or {}).get("deviations") or []) + [f"patch {p['name']}: applied={p['applied']}" + (f" impl={p['impl']}" if "impl" in p else "") + f" ({p['source']})" for p in (run.get("patches") or [])] + ([f"det level {run['det']['level']}: {run['det']['applied']}"] if (run.get("det") or {}).get("level") else [])
                      + switch_words,
        "model_switches": model_switches, "overrides": overrides,
        "shipped_constants": run.get("shipped_constants"), "det": run.get("det"), "patches": run.get("patches"),
        "upstream_fix": [r.get("id") for r in (run.get("upstream_fix") or [])],           # --upstream-fix: the IDs of the upstream-issue fixes the ARM applied (upstream_fix.py); [] = the upstream library as installed
        "activation": act, "refusals": refusals, "fallback": act.get("after", {}).get("fallback", []),
        "partial": refusals or missing, "missing_outputs": missing, "allow_partial": bool(allow_partial),
        "evidence": ("applied" if (mode.is_kit and act.get("after", {}).get("applied")) else
                     ("stock" if not mode.is_kit and not act.get("stock_lever_lines") else "partial")),
        "timing": run.get("timing"), "critics": run.get("critics"), "design_structure": run.get("design_structure"),
        "launch": {**launch, "proof": run.get("launch", {}).get("proof")},
        "outputs": O.listing(design_dir),
    }
    with open(os.path.join(design_dir, "opt_manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man
