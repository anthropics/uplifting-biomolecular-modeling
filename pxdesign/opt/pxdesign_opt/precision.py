"""The numerics policy of each mode (`opt_core.precision.policy.Policy`) and the package lever `tf32`.

  STOCK  `Policy("stock", matmul="highest")` — the engine's own numerics: the sampler's statements run in fp32 under
         `autocasting_disable_decorator(configs.skip_amp.sample_diffusion=True)` (protenix/model/utils.py; pxdesign/model/pxdesign.py:127-152)
         with torch's shipped fp32 matmul precision `highest` (IEEE fp32 products: no `allow_tf32`, `set_float32_matmul_precision` or
         TF32 environment switch anywhere under stock/src). Modes off / exact hold it: `attest()` READS the live switches against it
         after checkpoint load and records the verdict (`precision_policy=stock precision_attested=1|0`); nothing is set.
  TF32   `Policy("tf32", matmul="high", cudnn_tf32=True)` — the package lever `tf32` (`fast` and `big`): the process's fp32 matmuls (condition
         embedder and the 400-step sampler alike) run as TF32 tensor-core products with fp32 accumulation; every Linear / matmul statement is
         stock's. Tolerance-class by construction (tier 2): the rounding of every GEMM moves. `apply_tf32()` sets it through the core
         (`policy.apply`) at the package-lever hook and returns the hook's evidence fields (the PACKAGE line's `tf32_policy=tf32 tf32_matmul=high
         tf32_cudnn_tf32=True tf32_live=<matmul>/<tf32 words> tf32_changed=<fields> tf32_applied=True`).

The TF32 library overrides (`NVIDIA_TF32_OVERRIDE`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`) are never refused: torch folds them into its matmul
switch, the activation report notes their presence (`override_words`), and the KERNELS line and the LEVER tf32 line print the LIVE switches on every
route — a run whose numerics an override changed says so in its own log. `exit_check` re-reads the switches at exit against the plan; the
design verb records it in the manifest and exits NOT ACTIVE (3) when the `tf32` lever was REQUESTED and did not hold (`exit_problems`: a mode
is all of its levers); a stock-policy mode whose live switches moved is reported (`ok=0`), not refused.
"""
from __future__ import annotations

import os

from typing import Optional

from opt_core.precision import policy as _policy

Policy = _policy.Policy
STOCK = Policy("stock", matmul="highest", note="fp32 sampler under autocasting_disable_decorator(skip_amp.sample_diffusion); torch's shipped fp32 matmul precision; upstream sets no TF32 switch")
TF32 = Policy("tf32", matmul="high", cudnn_tf32=True, note="fp32 matmuls as TF32 tensor-core products, fp32 accumulate; every statement stock's")
LEVER = "tf32"
IMPL = "opt_core.precision.policy"
STRATEGY = "F4.tf32_matmul"                                       # the LEVER line's strategy word: the core's canonical id for the TF32 matmul policy (opt_core/STRATEGIES.json)


def policy_of(mode: str, package_levers=None) -> Policy:
    """The policy a process runs under: TF32 when the APPLIED plan includes the `tf32` package lever (the mode's set, or
    `package_levers` when given), else STOCK."""
    from .modes import KIT_MODES, check_mode
    levers = KIT_MODES[check_mode(mode)].package_levers if package_levers is None else package_levers
    return TF32 if LEVER in levers else STOCK


def exit_check(planned: Policy, torch=None) -> dict:
    """Re-read the live switches NOW (at exit, after the run) against the planned policy: `precision_planned`, `precision_live`, `precision_ok`
    (1/0; None = unreadable, named) and the mismatch words. The design verb files it in the manifest (`exit_problems` decides the exit)."""
    try:
        mm = _policy.mismatches(planned, torch=torch)
        live = _policy.live(torch=torch)
    except Exception as e:  # noqa: BLE001 — a torch without the precision getters: unreadable, named, never a guess
        return {"precision_planned": planned.name, "precision_ok": None, "precision_unreadable": type(e).__name__}
    return {"precision_planned": planned.name, "precision_live": _live_word(live), "precision_ok": 0 if mm else 1,
            **({"precision_mismatch": ",".join(f"{k}:{v[0]}!={v[1]}" for k, v in sorted(mm.items()))} if mm else {})}


def exit_problems(check: dict) -> list:
    """The `tf32` lever's engagement check (empty = pass): planned TF32 but the live switches disagree or are unreadable at exit. A STOCK plan is
    never a problem sentence — the record says `ok=0|None` and the run stands."""
    if check.get("precision_planned") != TF32.name or check.get("precision_ok") == 1:
        return []
    if check.get("precision_ok") is None:
        return [f"precision: lever tf32 requested but the live switches are unreadable ({check.get('precision_unreadable')}) — --mode exact runs without it"]
    return [f"precision: lever tf32 requested but live {check.get('precision_live')} ({check.get('precision_mismatch')}) — --mode exact runs without it"]


def override_words(environ=None) -> str:
    """The TF32 library override variables present in the environment as ``NAME=value`` words ("" when none) — reported, never refused."""
    environ = os.environ if environ is None else environ
    return ",".join(f"{h}={environ.get(h, '')}" for h in _policy.tf32_override_env(environ))


def attest(torch=None) -> dict:
    """Read the live switches against STOCK (nothing set): the evidence fields of a mode that holds the stock policy."""
    try:
        mm = _policy.mismatches(STOCK, torch=torch)
        live = _policy.live(torch=torch)
    except Exception as e:  # noqa: BLE001 — a torch without the precision getters: the verdict is `unreadable`, named, never a guess
        return {"precision_policy": STOCK.name, "precision_attested": None, "precision_unreadable": type(e).__name__}
    return {"precision_policy": STOCK.name, "precision_attested": 0 if mm else 1, "precision_live": _live_word(live),
            **({"precision_mismatch": ",".join(f"{k}:{v[0]}!={v[1]}" for k, v in sorted(mm.items()))} if mm else {})}


def apply_tf32(_hoist=None, torch=None) -> dict:
    """The package-lever hook's applier: set TF32 through the core and return the evidence fields (`applied` is the required one)."""
    rec = _policy.apply(TF32, torch=torch)
    mm = _policy.mismatches(TF32, torch=torch)
    return {"policy": TF32.name, "matmul": TF32.matmul, "cudnn_tf32": TF32.cudnn_tf32, "live": rec.get("live") or _live_word(_policy.live(torch=torch)),
            "changed": ",".join(rec.get("changed") or []) or "none", "applied": not mm,
            **({"mismatch": ",".join(f"{k}:{v[0]}!={v[1]}" for k, v in sorted(mm.items()))} if mm else {})}


def _live_word(live: dict) -> str:
    return "/".join(str(live.get(k)) for k in ("matmul", "matmul_tf32", "cudnn_tf32") if k in live) or "unknown"


def lever_line(tag: str, fields: Optional[dict], planned: bool) -> str:
    """The LEVER line of `tf32` for the exit census: on with the policy words and the live switches RE-READ now (`live=`), or off / skipped."""
    from opt_core import report as _core
    f = dict(fields or {})
    if planned and f.get("applied"):
        try:
            live = _live_word(_policy.live())
        except Exception:  # noqa: BLE001 — telemetry: the live words unreadable at exit are printed as such
            live = "unreadable"
        return _core.lever_line(tag, LEVER, "on", impl=IMPL, origin="core", strategy=STRATEGY, policy=TF32.name, matmul=TF32.matmul,
                                cudnn_tf32=TF32.cudnn_tf32, live=live)
    if planned:
        return _core.lever_line(tag, LEVER, "skipped", reason=(f.get("mismatch") or "not_applied").replace(" ", ""), impl=IMPL, origin="core", strategy=STRATEGY)
    return _core.lever_line(tag, LEVER, "off", impl=IMPL, origin="core", strategy=STRATEGY)
