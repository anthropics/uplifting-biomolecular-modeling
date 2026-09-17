"""Lever `pallas` — the kit adapter of the tree's ONE AF2 Pallas flash-attention kernel, `opt_core.kernels.pallas_attn`, served through
`opt_core.kernels.pallas_attn_serve` (F1: the served attention call, the eligibility rule, the per-process Ledger, the haiku class rebinding).

Scope on this engine: EVERY eligible `Attention` call of the AF-Multimer design model (`all_calls=True`: pair-biased or not) — per traced
program 9 calls: the Evoformer block's MSA row attention (pair bias), MSA column attention (no pair bias), triangle attention ×2; the extra-MSA
block's triangle attention ×2 and its MSA row attention — 64 channels over 8 heads = 8 per head, under the kernel's 16 floor: stock's math, the
ONE fallback by design (`head_dim_lt_16`); the template pair stack's triangle attention ×2 (16 per head). No per-call size rule (`min_tokens=0`).
Numerics class `precision`: bf16 rounding points move (exact online softmax, fp32 accumulation, forward and backward — `jax.custom_vjp` with separate
dQ and dK/dV kernels, no atomics: run-to-run bitwise). The Ledger DECLARES the structural per-call step-aside words (EXPECTED_FALLBACKS); any other fallback
reason, a kernel error, or zero served calls refuses the arm's gate (`LEDGER.gate()`, fail-closed). Evidence: ONE LEVER line at process exit (`opt_core.report.register_exit_tally`):

    [colabdesign-opt] LEVER name=F1.pallas_attn state=on impl=pallas_attn@<sha8> origin=core served=8 fallback=1 fallback_by=head_dim_lt_16:1
                      min_tokens=0 shapes=B..xH8xS..xD32:..,… scope=all_calls source=exit pid=<pid>
"""
from __future__ import annotations

from typing import Optional

from opt_core import report as _core_report
from opt_core.kernels import pallas_attn_serve as F1

from .names import LEVER_PALLAS, TAG

EXPECTED_FALLBACKS = (F1.HEAD_DIM_LT_16, F1.NO_PAIR_BIAS, F1.KEY_DIM_NE_VALUE_DIM, F1.BIAS_FORM, F1.BELOW_KEYS_RULE, F1.BELOW_SIZE_RULE)   # structural per-call step-asides BY NAME (stock's op serves the call; in this model `head_dim_lt_16`: the extra-MSA row attention, 8 per head, one call per traced program); environment words stay undeclared
ATTENTION_FWD_KERNELS = ("af2_flash_fwd",)                                            # the pallas_call name of this lever's attention FORWARD (opt_core af2_flash_pallas._fwd); declaring it marks this module an attention-kernel lever (levers.ATTENTION_KERNEL_LEVERS: proj / txla compose on one)
SCOPE = "all_calls"                              # every eligible call, pair-biased or not
LEDGER: Optional[F1.Ledger] = None
MARKER = "_opt_core_pallas_attn"                 # pallas_attn_serve's marker on the rebound Attention class
REFUSALS = (F1.Refusal,)                         # the kernel's named cannot-run (jax floor, no Pallas API, no GPU backend): levers.install steps the lever aside by name
NUMERICS = "precision"                           # registry.LEVERS[pallas].numerics: bf16 rounding points move, the mathematics is stock's


def off_line(reason: str) -> str:
    """The lever's line in a kit-route mode that does not select it: state=off, zero counters (impl = the kernel name; nothing is imported)."""
    return _core_report.lever_line(TAG, F1.LEVER, "off", reason=reason, impl=F1.NAME, origin="core")


def exit_line() -> str:
    """The lever's ONE evidence line of this process: the Ledger's census (served / fallback by reason / shapes) + scope."""
    if LEDGER is None:
        return off_line("not_installed")
    if LEDGER.origin is None:
        LEDGER.origin = F1.kernel_origin()
    if LEDGER.impl in (None, F1.NAME):
        LEDGER.impl = F1.kernel_impl()
    return LEDGER.line(TAG, scope=SCOPE, source="exit")


def install() -> dict:
    """Require the kernel (jax floor, Pallas, GPU backend — a named `F1.Refusal` otherwise), rebind `colabdesign.af.alphafold.model.modules.Attention`
    through F1 with this engine's scope, register the exit line. Idempotent. Call before the model function is traced. Returns the probe facts."""
    global LEDGER
    facts = F1.require()
    from colabdesign.af.alphafold.model import modules as CDM
    if LEDGER is None:
        LEDGER = F1.ledger(min_tokens=0, expected=EXPECTED_FALLBACKS)
    F1.enable([CDM], ledger=LEDGER, all_calls=True)
    from .kernels import register_exit_line
    register_exit_line(LEVER_PALLAS, exit_line)                               # the package's one exit printer (kernels/__init__.py): this lever's line at process exit
    return {"impl": F1.kernel_impl(), "origin": F1.kernel_origin(), "scope": SCOPE, "expected_fallbacks": list(EXPECTED_FALLBACKS),
            "jax": facts.get("jax") if isinstance(facts, dict) else None}


def uninstall() -> None:
    """Restore the stock `Attention` class (F1.disable); the Ledger and the exit line stay (the census of what this process served)."""
    import sys
    mods = sys.modules.get("colabdesign.af.alphafold.model.modules")
    if mods is not None:
        F1.disable([mods])


def installed() -> bool:
    """THIS lever serves the Attention class: opt_core's interception marker is on it and the ledger behind it is this module's (lever triatt
    points the same interception at its own kernels and ledger)."""
    import sys
    mods = sys.modules.get("colabdesign.af.alphafold.model.modules")
    A = getattr(mods, "Attention", None) if mods else None
    return bool(A is not None and getattr(A, MARKER, False) and LEDGER is not None and getattr(A, "_ledger", LEDGER) is LEDGER)


def evidence() -> dict:
    """The Ledger's fields ({} when not installed) and the gate verdict."""
    if LEDGER is None:
        return {}
    g = LEDGER.gate()
    return {**LEDGER.fields(), "scope": SCOPE, "gate_ok": bool(g.ok), "gate_reason": g.reason}
