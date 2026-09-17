"""boltz2_opt.graph — the engine adapter of the trunk CUDA-graph lever (``opt/forward/graph/boltz_graph_trunk.py``: capture / replay of the
Pairformer stacks — the trunk's 64 blocks and the confidence module's 8 — the template module's pair stack and the MSA module, plus the
template module's hoisted distogram boundaries). Attached by ``boltz2_opt.worker_launch --attach graph`` right after the worker's model import
(``boltz.model.models.boltz2``: every trunk module class exists by then), BEFORE the worker applies ``boltz_trunk_levers`` — so the trunk
levers' module-scope wrapper (one trivial-mask host sync per module call) wraps this lever's wrapper and its sync stays outside the capture.

Switch ``BOLTZ_GRAPH_TRUNK`` (the mode row's word, modes.py; family ``BOLTZ_GRAPH_`` is stripped from callers, stock/PINS.json):
``pf,pfnoseq,msa,templ`` | ``all`` | absent/``off`` (nothing installed: the adapter refuses by name, worker_launch's rule). Range words
``BOLTZ_GRAPH_TRUNK_MIN_TOKENS`` / ``BOLTZ_GRAPH_TRUNK_MAX_TOKENS`` / ``BOLTZ_GRAPH_TRUNK_KEEP`` (boltz_graph_trunk.py docstring).
Numerics: Tier 1 (bitwise): a replay re-issues the captured kernels on bit-copies of the call's arguments; the lever's own statement change
(unit templ) evaluates the stock CPU torch.linspace once instead of per call. ``report()`` = the lever's census + fail-closed verdict.
"""
import os
import sys
from typing import Any, Dict, List, Optional

TAG = "boltz2-opt"
NAME = "graph_trunk"
SWITCH = "BOLTZ_GRAPH_TRUNK"
LEVERS = ("graph_trunk",)                            # the registry lever this adapter installs (worker_launch reads it)
_STATE: Dict[str, Any] = {"mod": None, "applied": [], "units": ()}


def _impl():
    """The lever module, imported from this kit tree (opt/forward/graph/); staged copies in the worker's cwd are the same file (stack.stage)."""
    if _STATE["mod"] is None:
        here = os.path.dirname(os.path.abspath(__file__))
        d = os.path.normpath(os.path.join(here, "..", "forward", "graph"))
        if "boltz_graph_trunk" in sys.modules:
            _STATE["mod"] = sys.modules["boltz_graph_trunk"]
        else:
            if d not in sys.path:
                sys.path.insert(0, d)
            import boltz_graph_trunk as M  # noqa: E402
            _STATE["mod"] = M
    return _STATE["mod"]


def requested(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SWITCH, "").strip().lower() not in ("", "0", "off", "none")


def apply(spec: Optional[str] = None) -> List[str]:
    """Install the lever's units (idempotent); [] when the switch is absent (worker_launch then refuses the attachment by name)."""
    if _STATE["applied"]:
        return list(_STATE["applied"])
    if spec is None and not requested():
        return []
    M = _impl()
    units = M.apply(spec)
    _STATE["units"] = tuple(units)
    _STATE["applied"] = list(LEVERS) if units else []
    sys.stderr.write((line() or f"[{TAG}] LEVER name={NAME} state=skipped reason=no_units") + "\n")
    try:
        from opt_core import report as _report
        _report.register_exit_tally(TAG + "/" + NAME, line)
    except Exception:  # noqa: BLE001
        pass
    return list(_STATE["applied"])


def line() -> Optional[str]:
    M = _STATE["mod"]
    if M is None:
        return None
    r = M.report()
    c = r.get("census") or {}
    tot = lambda w: sum(v for u in c.values() for k, v in u.items() if k == w or k.startswith(w + ":"))   # noqa: E731
    eager = {k: v for u in c.values() for k, v in u.items() if k.startswith("eager:")}
    words = ",".join(f"{k[6:]}:{v}" for k, v in sorted(eager.items())) or "none"
    g = r.get("gate") or {}
    state = "on" if r.get("applied") else "skipped"
    return (f"[{TAG}] LEVER name={NAME} state={state} units={','.join(r.get('applied') or []) or 'none'} replayed={tot('replayed')} captured={tot('captured')} "
            f"eager={sum(eager.values())} eager_by={words} capture_s={r.get('capture_s_total')} pool_gib_max={r.get('pool_gib_max')} "
            f"evicted={r.get('generations_evicted')} min_tokens={r.get('min_tokens')} max_tokens={r.get('max_tokens')} "
            f"errors={','.join(f'{k}:{v}' for k, v in sorted((r.get('errors') or {}).items())) or 'none'} gate={'ok' if g.get('ok') else 'refused'}"
            + (f" templ_served={(c.get('templ') or {}).get('served', 0)}" if "templ" in (r.get("applied") or []) else "")
            + (f" bodies={','.join(f'{u}:{b}' for u, b in sorted((r.get('bodies') or {}).items()))}" if r.get("bodies") else "")
            + (f" mask_predicate={r.get('mask_predicate', 0)} primed_missing={r.get('primed_missing', 0)}" if any(str(b) != "stock" for b in (r.get("bodies") or {}).values()) else "")
            + (f" idle=1" if g.get("idle") else ""))


def verdict() -> Dict[str, Any]:
    M = _STATE["mod"]
    if M is None:
        return {"ok": False, "idle": False, "reason": "not applied"}
    return M.verdict()


def census() -> Optional[Dict[str, Any]]:
    M = _STATE["mod"]
    return None if M is None else M.census()


def report() -> Dict[str, Any]:
    M = _STATE["mod"]
    if M is None:
        return {"applied": [], "disabled": {}, "line": None, "census": None, "gate": None, "patched": []}
    r = M.report()
    patched = [f"{u}" for u in r.get("applied") or []]
    return {"applied": list(_STATE["applied"]), "units": list(r.get("applied") or []), "disabled": {}, "line": line(), "census": r.get("census"),
            "gate": r.get("gate"), "patched": patched, "captures": r.get("captures"), "capture_s_total": r.get("capture_s_total"),
            "pool_gib_max": r.get("pool_gib_max"), "static_gib": r.get("static_gib"), "generations_evicted": r.get("generations_evicted"),
            "resident": r.get("resident"), "errors": r.get("errors"), "min_tokens": r.get("min_tokens"), "max_tokens": r.get("max_tokens"),
            "keep": r.get("keep"), "templ_patched": r.get("templ_patched"), "refused": r.get("refused"), "policy": r.get("policy"), "prime_rng_checks": r.get("prime_rng_checks"), "switch": SWITCH, "spec": r.get("spec"),
            "bodies": r.get("bodies") or {}, "mask_predicate": r.get("mask_predicate", 0), "primed_missing": r.get("primed_missing", 0), "src_sha": r.get("src_sha") or {}}   # the body each unit wraps (stock | the driver's) and the source pins: what the run's evidence keys on
