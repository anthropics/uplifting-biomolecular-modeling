"""boltz2_opt — explicit interface to the Boltz-2 inference optimizations.

    import boltz2_opt
    report = boltz2_opt.enable("exact")      # or "fast" / "off"; idempotent, once per process
    boltz2_opt.status()                      # the last activation report

or, without code changes, `BOLTZ2_OPT=<mode> ...`: the package's .pth installs a lazy import hook that runs the first time the upstream
model family (`boltz.model`, `boltz.main`) is imported. Nothing is imported at interpreter start beyond this module; torch and boltz
load only when a mode is activated.

Boltz-2's optimizations run on the kits' persistent worker (one process: the model built once, the stock per-(input, seed) RNG
streams replayed, every input parsed in a fresh child of a pre-CUDA zygote as stock's one-process-per-input CLI parses it (prep.py), the
levers applied by the kit's own worker code from its activation row): `boltz2-opt pred --mode exact|fast|big` launches
that worker on the caller's YAMLs (modes.PINNED_ROUTE). The other route —
the levers applied inside a stock `boltz predict` process (a sitecustomize hook, or `enable()` before the upstream API) — has no kit
line that composes all three levers (modes.CLI_ROUTE), so `enable("exact"|"fast"|"big")` in-process and `BOLTZ2_OPT=<mode>` on the stock
CLI are refused by name: the report says `active: False` with the kit fact as `reason`, and under BOLTZ2_OPT the process exits 3 rather
than running stock silently. `enable()` still resolves and gates the mode on this box (kit files present, the stock pin,
BOLTZ_CACHE, a GPU) and returns the worker-route plan as `plan`.

Modes (`modes.MODES`, exactly four, named by guarantee): "off" = stock (the upstream CLI in a clean subprocess); "exact" = the kits' worker
kit line with upstream's fused kernels on (the worker's `--kernels on`; trunk row `resid,mask2` — the levers that act beside the fused
kernels — + CUDA-graph sampler + DiT hoist + the core's exact-class pair-track cells; byte-identical to the stock as shipped); "fast" =
exact's composition with the Tier-2 attention core / TriMul / chunked transitions (Tier 2 band vs the shipped stock); "big" = the memory mode
(fast's trunk and pair-track levers under the providers' `big` tier words + the engine adapter boltz2_opt.big: row-chunked transition / conditioner, pre-confidence
free on the core's memory primitives, eager sampler, expandable segments; lower peak memory, Tier 2). Any other mode name
is refused by name. `registry.LEVERS` describes each lever; the
kits' own code applies them (stack.py stages the kit files and launches the kit worker).

The contract: `enable(mode)` returns the activation report (a dict; `active` says whether the levers are on in THIS process, `reason` says
why not, `pinned_route` names where they are), `status()` returns the last report. Late activation: `enable()` may run any time
after `boltz` is imported and is refused by name once a Boltz2 instance exists (constructor counter armed at the first call) or a kit
lever module reports itself applied in this process. Every refusal is one line on stdout: `[boltz2-opt] NOT ACTIVE: <reason>`.
"""
from __future__ import annotations

from typing import Optional

__version__ = "0.3.36"
__all__ = ["enable", "status", "MODES", "ActivationError", "registry", "modes", "stack", "manifest"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated in this process (route refused by name, pins, kit files, no GPU)."""


_LAST: dict = {}


def enable(mode: str, *, strict: bool = False, trigger: Optional[str] = None) -> dict:
    """Resolve, gate and — where a kit line exists for this process — activate `mode` ("exact" | "fast" | "off"). Idempotent per process:
    a second call with the same mode returns the first report; a different mode is refused. Returns the activation report; with
    ``strict=True`` a refusal raises ``ActivationError`` (after printing the NOT ACTIVE line). An absent or older shared core is the
    kit's core pin gate's to refuse (``_core_gate.gate``, then ``_autoload.require_core``: the NOT ACTIVE line, SystemExit(3)) before anything of the core is imported."""
    from ._core_gate import gate
    from ._autoload import TAG, require_core
    gate(__file__, TAG)
    require_core()
    from . import modes, report as rep, stack
    global _LAST
    try:
        row = modes.resolve(mode)
    except ValueError as e:
        r = _inactive(str(mode), str(e), trigger); return _finish(r, strict)
    m = row["mode"]
    if _LAST and _LAST.get("requested"):
        if _LAST["requested"] == m:
            return dict(_LAST)
        r = _inactive(m, f"mode {_LAST['requested']!r} was already requested in this process; one mode per process", trigger)
        return _finish(r, strict)
    if m == "off":
        r = _inactive(m, "mode off: stock — nothing is applied; `boltz2-opt pred --mode off` runs the stock CLI in a clean, proven subprocess", trigger)
        r["requested"] = m; _LAST = r; return _finish(r, strict=False)
    stack.arm_instance_counter()
    plan = stack.gate(m, need_gpu=True)
    late = stack.late_activation_block()
    if late:
        r = _inactive(m, f"late activation refused: {late}", trigger, plan)
    elif plan["reasons"]:
        r = _inactive(m, "; ".join(plan["reasons"]), trigger, plan)
    else:
        r = _inactive(m, f"in-process route of {m} is not a kit line: {row['cli_route']}", trigger, plan)
        r["gated"] = True
    r["requested"] = m; _LAST = r
    return _finish(r, strict)


def _inactive(mode: str, reason: str, trigger: Optional[str], plan: Optional[dict] = None) -> dict:
    from . import modes, stack
    r = {"active": False, "mode": mode, "reason": reason, "pinned_route": modes.PINNED_ROUTE, "gated": False, "trigger": trigger,
         "package_version": __version__, "levers_applied": [], "levers_fallback": [], "partial": False,
         "gpu": (plan or {}).get("gpu", {}) and (plan or {}).get("gpu", {}).get("name"), "boltz_version": ((plan or {}).get("pins") or {}).get("version")}
    if plan is not None:
        r["plan"] = {k: plan[k] for k in ("mode", "tier", "route", "levers", "env", "worker_base", "stage", "reasons", "cache")}
    return r


def _finish(r: dict, strict: bool) -> dict:
    from . import report as rep
    rep.say(rep.not_active_line(r["reason"]))
    if strict:
        raise ActivationError(r["reason"])
    return dict(r)


def status() -> dict:
    """The last activation report, or ``{"active": False, "reason": "no activation requested"}`` before any."""
    return dict(_LAST) if _LAST else {"active": False, "reason": "no activation requested", "mode": None}


def _reset_for_tests() -> None:
    global _LAST
    _LAST = {}


def __getattr__(name):
    if name in ("registry", "modes", "stack", "manifest"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    if name == "MODES":
        from .modes import MODES
        return MODES
    raise AttributeError(name)
