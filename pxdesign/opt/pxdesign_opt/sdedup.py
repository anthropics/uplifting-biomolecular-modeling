"""`sdedup` — the token path's single conditioning evaluated on ONE sample row (`opt_core.capture.hoist.RowDedup`, strategy F7.row_dedup).

What it removes. Every denoiser call of the sampler receives `t_hat_noise_level` of shape `[..., N_sample]` holding one value — the step's
noise level broadcast over the batch (protenix/model/generator.py: `t_hat = c_tau_last * (gamma + 1)`, a scalar of the schedule, expanded) —
so the single conditioning `s = f(s_trunk, s_inputs, t_hat)` (DiffusionConditioning.forward, single path: `fourier_embedding` →
`layernorm_n` → `linear_no_bias_n` → `transition_s1` / `transition_s2`) has `N_sample` identical rows, and so has everything computed from `s`
alone: `linear_no_bias_s(layernorm_s(s))` of DiffusionModule.f_forward and, in each of the 24 token DiffusionTransformerBlocks, AdaLN's
`linear_s(s)` / `linear_nobias_s(s)` (AttentionPairBias and ConditionedTransitionBlock) and the two adaLN-Zero output gates
`linear_a_last(s)` / `linear_s(s)` — six `c_s → c_token` projections per block, ≈ 21 % of a token block's GEMM work at any batch. With `s`
carried as `[..., 1, N_tok, c_s]` each of those stock statements runs on `N_tok` rows instead of `N_sample · N_tok` and broadcasts over the
sample dimension inside the stock elementwise products that consume it (`sigmoid(linear_s(s)) * a + linear_nobias_s(s)`, …; no statement
changes).

How it attaches. Through the hoist kit's own extension point: `hoist._f_forward_hoisted` asks `pxd_xattempt.fuse.state()["sdedup"]` and, when
set, takes the single conditioning from `pxd_xattempt.fuse._cond_forward_single_dedup(...)` (opt/forward/hoist/pxd_xattempt/hoist.py:197-205).
This module IS that `fuse`: `apply()` binds it as the `fuse` attribute of the carried `pxd_xattempt` package (a run-time attribute, like the
kit's own class patches; no carried byte changes) — so the lever acts only where the hoist lever acts (a model `install()` tagged) and only in
a process whose mode plans it. The row logic is the core's `RowDedup` (ONE object for this one site): the uniformity read of `t_hat` across
the sample dimension, the evaluation on one row, the counts. A call whose `t_hat` rows differ evaluates the stock rows — RowDedup's
`fallback_by=nonuniform`, named and counted, folded into the design verb's exit by `gate()` (expected count: 0); never silent.

Numerics. Exact algebra, not bitwise: cuBLAS selects an fp32 GEMM kernel by its M, so the same products are summed in another order and the
400-step sampler amplifies the last bit (outputs are not byte-identical to stock even under the det recipe; the saving is one row's work in
N_sample of the single path). Tier 2 — the `fast` and `big` lines. The site is tolerance class (`KLASS`, RowDedup's `klass`; the LEVER line says `class=tolerance`): the core's
per-call equality demand (a `torch.equal` between the one-row and the full-row statement, an option of `RowDedup`) is meaningful
for exact sites only and stays 0 here.
"""
from __future__ import annotations

import inspect
import sys

NAME = "sdedup"
STRATEGY = "F7.row_dedup"
IMPL = "opt_core.capture.hoist.RowDedup"
KLASS = "tolerance"                                             # the numerics class the mode table declares for this site (registry: tier 2); RowDedup refuses an equality demand for it
HOOK_TOKEN = "_fuse._cond_forward_single_dedup("                # the carried lever's call into this hook (hoist.py:202): the source guard of apply()
SAMPLE_DIM_IN, SAMPLE_DIM_OUT = -1, -3                          # t_hat: [..., N_sample]; s: [..., N_sample, N_tok, c_s]
_STATE = {"on": False, "dedup": None}


def dedup():
    """The ONE RowDedup of this site (constructed on first use; tolerance class, no per-call equality demand)."""
    if _STATE["dedup"] is None:
        from opt_core.capture.hoist import RowDedup
        _STATE["dedup"] = RowDedup(NAME, klass=KLASS, verify_every=0, strict=True, materialize=False)   # klass: the class the mode table declares for this site (tolerance; RowDedup refuses verify_every > 0 there)
    return _STATE["dedup"]


def state() -> dict:
    """The hook's switchboard as hoist.py reads it (`state()["sdedup"]`)."""
    return {"sdedup": bool(_STATE["on"]), "qkvg": False}


def _single_path(cond, single_base, inplace_safe: bool):
    """DiffusionConditioning.forward's single path below the step-invariant base (protenix/model/modules/diffusion.py; the same statements
    hoist.py:206-216 replays), as a function of `t_hat_noise_level` alone — row-independent along the sample dimension."""
    import torch

    def fn(t_hat_noise_level):
        noise_n = cond.fourier_embedding(t_hat_noise_level=torch.log(input=t_hat_noise_level / cond.sigma_data) / 4).to(single_base.dtype)
        single_s = single_base.unsqueeze(dim=-3) + cond.linear_no_bias_n(cond.layernorm_n(noise_n)).unsqueeze(dim=-2)
        if inplace_safe:
            single_s += cond.transition_s1(single_s)
            single_s += cond.transition_s2(single_s)
        else:
            single_s = single_s + cond.transition_s1(single_s)
            single_s = single_s + cond.transition_s2(single_s)
        return single_s
    return fn


def _cond_forward_single_dedup(cond, t_hat_noise_level, s_inputs, s_trunk, inplace_safe, single_base=None):
    """The hook hoist.py calls: `[..., 1, N_tok, c_s]` when RowDedup served the call (uniform `t_hat`, one row evaluated) or there is one
    sample (`single`); the stock `[..., N_sample, N_tok, c_s]` rows when RowDedup evaluated the full rows — `fallback` (non-uniform `t_hat`,
    counted) or `bypassed` (inside a CUDA-graph capture; not reachable on this tree's routes, handled all the same); None when the
    step-invariant base is not supplied (hoist.py then runs its own stock lines). The outcome is read from RowDedup's own counters."""
    if single_base is None or not _STATE["on"]:
        return None
    D = dedup()
    before = D.stats()
    y = D.apply(_single_path(cond, single_base, inplace_safe), t_hat_noise_level, SAMPLE_DIM_IN, out_dim=SAMPLE_DIM_OUT)
    after = D.stats()
    one_row = any(int(after.get(k) or 0) != int(before.get(k) or 0) for k in ("served", "single"))
    if not one_row:
        return y                                                                  # fallback / bypassed: the stock rows, evaluated in full (counted by RowDedup under its own name)
    return y.narrow(SAMPLE_DIM_OUT, 0, 1)                                         # one evaluated row (served: the broadcast view's row 0; single: the row itself); the consumers broadcast it over N_sample

def apply(hoist_mod=None) -> dict:
    """The package-lever hook's applier: bind this module as `pxd_xattempt.fuse` and switch the site on. Evidence fields: `sdedup_hook`
    (the PACKAGE line's `sdedup_hook`, the required one: the carried lever consults the hook AND the binding is this module), `class`."""
    import pxd_xattempt
    me = sys.modules[__name__]
    other = getattr(pxd_xattempt, "fuse", None)
    if other is not None and other is not me:
        return {"hook": False, "reason": f"pxd_xattempt.fuse is already bound to {getattr(other, '__name__', other)!r}"}
    hoist_mod = hoist_mod or sys.modules.get("pxd_xattempt.hoist")
    try:
        consulted = HOOK_TOKEN in inspect.getsource(hoist_mod._f_forward_hoisted)
    except Exception:  # noqa: BLE001
        consulted = False
    if not consulted:
        return {"hook": False, "reason": "hoist._f_forward_hoisted does not consult pxd_xattempt.fuse (source guard)"}
    pxd_xattempt.fuse = me
    sys.modules.setdefault("pxd_xattempt.fuse", me)
    dedup()
    _STATE["on"] = True
    return {"hook": True, "class": KLASS}


def stats() -> dict:
    D = _STATE["dedup"]
    return dict(D.stats()) if D is not None else {}


def gate() -> list:
    """Exit-gate sentences (empty = pass): no fallback (a non-uniform `t_hat` is not expected on any route of this tree) and — when the run had
    dedup-able calls at all (calls beyond `single` / `bypassed`: N_sample > 1) — at least one served call. An N_sample 1 run (every call
    `single`) passes: there was nothing to deduplicate."""
    D = _STATE["dedup"]
    if not _STATE["on"] or D is None:
        return []
    st = D.stats()
    dedupable = int(st.get("calls") or 0) - int(st.get("single") or 0) - int(st.get("bypassed") or 0)
    return list(D.gate(expect_fallback=(), max_fallback=0, require_served=dedupable > 0))

def lever_line(tag: str, planned: bool) -> str:
    """The LEVER line for the exit census: RowDedup's evidence line when live, else off."""
    from opt_core import report as _core
    D = _STATE["dedup"]
    if planned and _STATE["on"] and D is not None:
        return D.evidence_line(tag, name=NAME, strategy=STRATEGY)                 # the core's evidence words, class=<klass> among them
    if planned:
        return _core.lever_line(tag, NAME, "skipped", reason="not_applied", impl=IMPL, origin="core", strategy=STRATEGY)
    return _core.lever_line(tag, NAME, "off", impl=IMPL, origin="core", strategy=STRATEGY)


def reset_for_tests() -> None:
    _STATE.update(on=False, dedup=None)
    mod = sys.modules.get("pxd_xattempt")
    if mod is not None and getattr(mod, "fuse", None) is sys.modules.get(__name__):
        delattr(mod, "fuse")
    sys.modules.pop("pxd_xattempt.fuse", None)
