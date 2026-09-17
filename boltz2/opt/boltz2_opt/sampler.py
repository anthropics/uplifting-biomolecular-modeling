"""boltz2_opt.sampler — the engine adapter of the BOLTZ2-FASTER `sampler` levers (opt/forward/sampler/src/bz_sampler*.py) onto Boltz-2's
diffusion sampler (``boltz.model.modules.diffusionv2.AtomDiffusion.sample``, ``DiffusionModule`` / ``DiffusionTransformer``).

Attached by ``worker_launch --attach sampler`` right after ``boltz.model.modules.diffusionv2`` has executed. Every lever is ONE row env
word: the row sets it, the adapter reads it, ``apply()`` installs it or refuses BY NAME, and ``report()`` states per lever
``state=on|off|skipped`` with a reason; ``off`` (the word absent) leaves the stock statement in place.

  BOLTZ_SAMPLER_ROLLOUT=graph|eager   registry lever `rollout` (exact tier): the boundary-graph roll-out of AtomDiffusion.sample (bz_sampler.py
                                      module doc: schedule read once, RNG pre-drawn in stock order, one CUDA graph per step boundary, SVD/det
                                      eager, guard prints without per-step syncs, in-graph Euler with host-folded fp32 scalars). `eager` = the
                                      same statements without the graph (falsification control).
  BOLTZ_SAMPLER_ALIGN=jacobi64        registry lever `align_jacobi64` (fast tier): sync-free in-graph 3x3 SVD/det (fp64 Jacobi) for the rigid alignment.
  BOLTZ_SAMPLER_ALIGN=aligncap        registry lever `align_aligncap` (exact tier): WASTE's bitwise cusolverDnSgesvd seam (opt/forward/waste/aligncap.py) between replays.
  BOLTZ_SAMPLER_DIT=<words>           registry lever `dit_fused` (fast tier; bz_sampler_dit.py): the fused token-transformer step, under the roll-out or —
                                      without BOLTZ_SAMPLER_ROLLOUT — from the stock eager loop (the memory row: bf16,weights=sample); words: see that module.

Refusals by name (exit 3 through worker_launch): the kit's per-step graph patch requested in the same row (BOLTZ_GRAPH_DIFFUSION set: both own
AtomDiffusion.sample), a word that is not a value, a failed install precondition (e.g. the scalar-division probe). Fail-closed gate at exit:
the roll-out must have served every sample() call that was in scope (scope words counted by name), no capture failure, and
AtomDiffusion.sample calls must have reached the roll-out (calls > 0 when the worker predicted anything).
"""
import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional

TAG = "boltz2-opt"
NAME = "sampler"
BUNDLE = "forward/sampler/src"                                   # opt/-relative: the directory carrying bz_sampler*.py (stack.kit_path resolves it)
FILES = (f"{BUNDLE}/bz_sampler.py", f"{BUNDLE}/bz_sampler_dit.py")
SWITCH_ROLLOUT = "BOLTZ_SAMPLER_ROLLOUT"                         # graph | eager
SWITCH_ALIGN = "BOLTZ_SAMPLER_ALIGN"                             # jacobi64 (absent = stock gesvd, bitwise)
SWITCH_DIT = "BOLTZ_SAMPLER_DIT"                                 # the fused-step words (bz_sampler_dit)
SWITCH_MAX_TOKENS = "BOLTZ_SAMPLER_MAX_TOKENS"                   # the memory row's token ceiling for the roll-out group (modes.BIG_SAMPLER_CEILINGS states the number per card; a
                                                                 # non-number = no ceiling): predictions above it sample on the stock eager loop by name (scope word above_max_tokens)
GRAPH_PATCH_SWITCH = "BOLTZ_GRAPH_DIFFUSION"                     # the kit's per-step graph patch: must be absent when the roll-out is on
LEVERS = ("rollout", "align_jacobi64", "align_aligncap", "dit_fused")               # registry levers this adapter installs (worker_launch reads it)
EXPECTED_SCOPE = ("steering", "guidance_contact_constraints", "guidance_template_force", "above_max_tokens")   # scope words a healthy run may show, by name: --use_potentials (the kit drops the
# levers for that run anyway) and inputs whose yaml carries force=true contact/pocket constraints or forced templates (stock's guidance loop does real host-scheduled
# per-step work there -> the stock sampler serves that call; counted on the LEVER line as scope=<word>:<n>)
_STATE: Dict[str, Any] = {"mod": None, "applied": [], "words": {}, "refused": None, "levers": {}}


def _load():
    """Import bz_sampler from the carried bundle directory (that directory alone enters sys.path for the import). Idempotent."""
    if _STATE["mod"] is not None:
        return _STATE["mod"]
    from . import stack
    src = stack.kit_path(BUNDLE)
    path = os.path.join(src, "bz_sampler.py")
    if not os.path.isfile(path):
        raise RuntimeError(f"{NAME}: kit file missing: {path}")
    spec = importlib.util.spec_from_file_location("bz_sampler", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bz_sampler"] = mod
    spec.loader.exec_module(mod)
    _STATE["mod"] = mod
    return mod


def words(environ=None) -> Dict[str, str]:
    env = os.environ if environ is None else environ
    return {k: env.get(k, "").strip().lower() for k in (SWITCH_ROLLOUT, SWITCH_ALIGN, SWITCH_DIT, SWITCH_MAX_TOKENS) if env.get(k, "").strip()}


def requested(environ=None) -> bool:
    return bool({k: v for k, v in words(environ).items() if k != SWITCH_MAX_TOKENS})   # the ceiling word alone requests nothing


def apply(spec: Optional[dict] = None) -> List[str]:
    """Install the requested levers (idempotent). Returns the registry lever names installed; raises RuntimeError naming a refusal."""
    if _STATE["applied"]:
        return list(_STATE["applied"])
    w = words() if spec is None else dict(spec)
    _STATE["words"] = w
    if not w:
        return []
    if os.environ.get(GRAPH_PATCH_SWITCH, "").strip().lower() not in ("", "0", "off") and w.get(SWITCH_ROLLOUT):
        raise RuntimeError(f"{NAME}: {SWITCH_ROLLOUT}={w[SWITCH_ROLLOUT]} with {GRAPH_PATCH_SWITCH}={os.environ.get(GRAPH_PATCH_SWITCH)}: "
                           "the roll-out and the kit's per-step graph patch both own AtomDiffusion.sample — refused (a row sets one of them)")
    mod = _load()
    rollout = w.get(SWITCH_ROLLOUT, "off")
    kabsch = w.get(SWITCH_ALIGN, "gesvd") or "gesvd"
    acmod = None
    if kabsch == "aligncap":                                          # WASTE's bitwise gesvd seam: staged beside the worker as aligncap.py, else the kit tree's copy
        try:
            import aligncap as acmod
        except ImportError:
            f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "forward", "waste", "aligncap.py")
            if not os.path.isfile(f):
                raise RuntimeError(f"[{TAG}] {SWITCH_ALIGN}=aligncap: opt/forward/waste/aligncap.py is not in this tree and no aligncap module is staged — refused by name")
            spec = importlib.util.spec_from_file_location("aligncap", f); acmod = importlib.util.module_from_spec(spec); sys.modules["aligncap"] = acmod; spec.loader.exec_module(acmod)
    dit = [x for x in (w.get(SWITCH_DIT, "") or "").split(",") if x.strip()]
    mt = (w.get(SWITCH_MAX_TOKENS) or "").strip()
    max_tokens = int(mt) if mt.isdigit() else 0                        # `card` (unresolved: no card probed) or absent = no ceiling
    installed = mod.install(rollout=rollout, kabsch=kabsch, dit=dit or None, aligncap_module=acmod, max_tokens=max_tokens)   # raises RuntimeError by name on a bad word / failed precondition
    applied = []
    levers = {}
    if rollout in ("graph", "eager"):
        applied.append("rollout"); levers["rollout"] = {"state": "on", "word": rollout, "div_recipe": mod.STATS.get("div_recipe"), "max_tokens": max_tokens or None}
    else:
        levers["rollout"] = {"state": "off", "reason": "word_absent"}
    if kabsch in ("jacobi64", "device"):
        applied.append("align_jacobi64"); levers["align_jacobi64"] = {"state": "on", "word": kabsch}
    else:
        levers["align_jacobi64"] = {"state": "off", "reason": "word_absent" if kabsch == "gesvd" else f"word={kabsch}"}
    if kabsch == "aligncap":
        applied.append("align_aligncap"); levers["align_aligncap"] = {"state": "on", "word": kabsch}
    else:
        levers["align_aligncap"] = {"state": "off", "reason": "word_absent" if kabsch == "gesvd" else f"word={kabsch}"}
    if dit:
        applied.append("dit_fused"); levers["dit_fused"] = {"state": "on", "word": ",".join(dit)}
    else:
        levers["dit_fused"] = {"state": "off", "reason": "word_absent"}
    _STATE.update(applied=applied, levers=levers)
    sys.stderr.write(f"[{TAG}] {NAME}: installed {installed} (div_recipe={mod.STATS.get('div_recipe')})\n")
    return list(applied)


def verdict() -> Dict[str, Any]:
    """Fail-closed: refused when a sample() call in scope was not served by the roll-out (scope word outside EXPECTED_SCOPE), on a capture
    failure, or when the worker predicted and no call reached the roll-out."""
    mod = _STATE["mod"]
    if mod is None or not _STATE["applied"]:
        return {"ok": False, "idle": False, "reason": "not applied"}
    st = mod.STATS
    bad_scope = {k: n for k, n in (st.get("scope") or {}).items() if k not in EXPECTED_SCOPE}
    if bad_scope:
        return {"ok": False, "idle": False, "reason": "scope:" + ",".join(f"{k}={n}" for k, n in bad_scope.items())}
    if st.get("capture_failed"):
        return {"ok": False, "idle": False, "reason": f"capture_failed={st['capture_failed']}"}
    if "dit_fused" in _STATE["applied"]:
        ds = ((mod.report().get("dit") or {}).get("stats") or {})
        bad = {k: n for k, n in (ds.get("scope") or {}).items()}
        if bad:
            return {"ok": False, "idle": False, "reason": "dit_scope:" + ",".join(f"{k}={n}" for k, n in bad.items())}
        if (st.get("samples", 0) > 0 or st.get("calls", 0) > 0) and ds.get("served", 0) == 0:     # under the roll-out (samples) or the stock eager loop (calls reached sample())
            return {"ok": False, "idle": False, "reason": "dit_fused installed but served no token-transformer call"}
    if "rollout" in _STATE["applied"] and st.get("calls", 0) > 0 and st.get("samples", 0) == 0 and not st.get("scope"):
        return {"ok": False, "idle": False, "reason": "calls reached the roll-out but none was sampled"}
    if st.get("calls", 0) == 0:
        return {"ok": True, "idle": True, "reason": "idle: no AtomDiffusion.sample call reached the roll-out (nothing predicted)"}
    return {"ok": True, "idle": False, "reason": None}


def line() -> str:
    mod = _STATE["mod"]
    st = mod.STATS if mod is not None else {}
    lv = _STATE["levers"]
    parts = [f"[{TAG}] LEVER name=rollout state={lv.get('rollout', {}).get('state', 'off')}"]
    if lv.get("rollout", {}).get("state") == "on":
        cs = st.get("capture_s") or []
        parts.append(f"word={lv['rollout']['word']} calls={st.get('calls', 0)} samples={st.get('samples', 0)} replays={st.get('replays', 0)} "
                     f"captures={st.get('captures', 0)} capture_s_total={round(sum(cs), 3)} capture_s_max={max(cs) if cs else 0} capture_failed={st.get('capture_failed', 0)} "
                     f"kabsch={st.get('kabsch')} div_recipe={st.get('div_recipe')} predraw={st.get('predraw') or '-'} "
                     f"scope={','.join(f'{k}:{n}' for k, n in (st.get('scope') or {}).items()) or '-'} guard_prints={st.get('guard_prints', 0)} "
                     f"max_tokens={lv['rollout'].get('max_tokens') or '-'} token_gated={st.get('token_gated', 0)}")
    else:
        parts.append(f"reason={lv.get('rollout', {}).get('reason', '-')}")
    dit = ((mod.report() or {}).get("dit") or {}) if mod is not None else {}
    ds = dit.get("stats") or {}
    parts.append(f"| LEVER name=dit_fused state={lv.get('dit_fused', {}).get('state', 'off')}")
    if lv.get("dit_fused", {}).get("state") == "on":
        parts.append(f"word={lv['dit_fused']['word']} gemm={ds.get('gemm')} attn={ds.get('attn')} layers={ds.get('layers')} launches_per_layer={ds.get('launches_per_layer')} "
                     f"calls={ds.get('calls', 0)} served={ds.get('served', 0)} scope={','.join(f'{k}:{n}' for k, n in (ds.get('scope') or {}).items()) or '-'} "
                     f"packs={ds.get('packs', 0)} pack_gib_last={ds.get('pack_gib')} weights_mib={ds.get('weights_mib')} weights={ds.get('weights') or 'resident'} "
                     f"weights_builds={ds.get('weights_builds', '-')} host={'rollout' if lv.get('rollout', {}).get('state') == 'on' else 'eager'} out_dtype_probe={ds.get('out_dtype_probe')} mask_fold={ds.get('mask_fold')}")
    parts.append(f"| LEVER name=align_jacobi64 state={lv.get('align_jacobi64', {}).get('state', 'off')} | LEVER name=align_aligncap state={lv.get('align_aligncap', {}).get('state', 'off')}")
    return " ".join(parts)


def report() -> Dict[str, Any]:
    mod = _STATE["mod"]
    rep = mod.report() if mod is not None else None
    return {"applied": list(_STATE["applied"]), "levers": dict(_STATE["levers"]), "words": dict(_STATE["words"]), "line": line(),
            "gate": verdict(), "patched": (["AtomDiffusion.sample"] if "rollout" in _STATE["applied"] else []), "module": rep}
