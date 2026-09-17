"""Lever `triatt` — the AF-Multimer design model's attention kernel for EVERY haiku `Attention` call, taken from the shared core's JAX-family
provider `opt_core.kernels.pallas` BY TIER WORD (`fast`, through `kernels/provider.py`): the provider's measured row for the model's main cell
(triangle attention, 4 heads x 32, the differentiated call) among the rows this adapter binds — `cd_triatt`
(`opt_core/kernels/pallas_triatt/{triatt_attn,attbwd_dkdv}.py`: Pallas/Triton flash attention with the pair bias entering through the tensor
core, an fp32 additive key-mask row and mask-free kernels chosen per call, forward AND backward — a two-kernel deterministic backward with the
pair-bias gradient reduced from per-group partial sums) — served through the tree's ONE haiku `Attention` interception,
`opt_core.kernels.pallas_attn_serve` (F1: the served call, the eligibility rule, the class rebinding, the per-process Ledger), with
`op=<row module>.attention`. The interception takes ONE op per process: the word is resolved ONCE, for the model's main cell, at install
(`provider.admit`); what the provider's table names for every OTHER cell this run served (template stack 16 per head, extra-MSA row attention
8 per head, MSA row attention over 2 sequences, per size bucket) is resolved from the served shapes at exit and printed beside it — `row=` is
what ran, `cells=` / `tier=` what the table names per cell, `uncovered=` the cells it has no number for — a census, never a guess. A word the
provider refuses, or one that names a row this adapter does not bind for the main cell (the stock statement `xla`), is the lever's refusal BY
NAME at install (`state=skipped reason=cannot_run`). It supersedes lever `pallas` (the same call sites, the same tier): a mode that holds both
runs this one and prints pallas `state=off reason=replaced`.

Scope: every `Attention` call, pair-biased or not (`all_calls=True`) — per traced program the Evoformer block's MSA row attention and triangle
attention x2; the extra-MSA block's triangle attention x2 and MSA row attention (8 per head: zero-padded to 16 — exact — so
`pad_head_dim_below_min=True`); the template pair stack's triangle attention x2 (16 per head). The ONE fallback by design: calls with fewer
than MIN_KEYS = 16 keys — the MSA column attention over N_seq = 2 sequences, a [B,H,2,2] softmax that is not kernel work — run STOCK's
method verbatim (bitwise; reason `below_keys_rule`, F1's `min_keys` rule), declared in EXPECTED_FALLBACKS; the Ledger's gate is fail-closed
on any other fallback reason, any kernel error, or zero served calls.
Numerics class `precision`: stock's mathematics (softmax(q k^T / sqrt(d) + mask + pair bias) v, exact — not approximate — online softmax);
bf16 products with fp32 accumulation and fp32 softmax statistics, P and dS rounded to bf16 before their products (as stock's XLA path rounds
its bf16 logits/weights); float32 inputs at XLA's DEFAULT product class (tf32, stock's on sm_90; `ieee` available in the op, no other word).
Deterministic: no atomics anywhere — run-to-run bitwise, gradients included. Evidence: ONE LEVER line at process exit through the package's
exit printer (`kernels.register_exit_line`):

    [colabdesign-opt] LEVER name=triatt state=on impl=triatt_attn@<sha8> origin=core served=<n> fallback=<m> fallback_by=below_keys_rule:<m> min_tokens=0
                      shapes=B..xH4xS..xD32:..,… numerics=precision precision=bf16 f32_precision=tf32 bwd=deterministic kernel=<sha8> scope=all_calls
                      provider=opt_core.kernels.pallas@<core> word=fast row=cd_triatt:<served calls> cells=<arm:calls,…> tier=<arm:cells,…>
                      [uncovered=<n>:<family/direction>,…] source=exit pid=<pid>
"""
from __future__ import annotations

from typing import Optional

from opt_core import report as _core_report
from opt_core.counters import Ledger
from opt_core.kernels import pallas_attn_serve as F1

from ..names import TAG
from . import provider

LEVER = "triatt"                                   # registry id = the LEVER line's name=
NUMERICS = "precision"                             # registry.LEVERS[triatt].numerics: rounding points move (bf16 P/dS, online softmax), mathematics is stock's
SUPERSEDES = ("pallas",)                           # the same call sites in the same tier (registry row: supersedes=)
ATTENTION_FWD_KERNELS = ("triatt_fwd",)            # the pallas_call name prefix of this lever's attention FORWARD kernels (triatt_attn._fwd names them
                                                   # triatt_fwd_<m|u>_G<G>_bq<bq>_bk<bk>_w<warps>_s<stages>); declaring it marks this module an attention-kernel lever (levers.ATTENTION_KERNEL_LEVERS)
MIN_KEYS = 16                                      # calls with fewer keys (the MSA column attention over N_seq = 2 sequences) run STOCK's method verbatim
EXPECTED_FALLBACKS = (F1.BELOW_KEYS_RULE, F1.NO_PAIR_BIAS, F1.KEY_DIM_NE_VALUE_DIM, F1.BIAS_FORM, F1.BELOW_SIZE_RULE, F1.HEAD_DIM_LT_16)   # structural per-call step-asides BY NAME; environment words stay undeclared —         # ... and are the ONE fallback reason this lever declares (bitwise stock; a [B,H,2,2] softmax is not kernel work)
SCOPE = "all_calls"
MARKER = "_opt_core_pallas_attn"                   # pallas_attn_serve's marker on the rebound Attention class (the interception is F1's)
ORIGIN = "core"                                    # the implementation lives in the shared core (provider row cd_triatt); the adapter is this module + F1


class Refusal(RuntimeError):
    """The provider refused the lever's word by name (levers.install steps the lever aside: state=skipped reason=cannot_run)."""


REFUSALS = (F1.Refusal, Refusal)                   # jax floor / no Pallas-Triton lowering / no GPU backend / the provider's refusal: the lever steps aside by name
LEDGER: Optional[Ledger] = None
_K = None                                          # the row module the word resolved to at install (provider.admit): its `attention` is the op served


def kernel_impl() -> str:
    """The line's impl=: `triatt_attn@<sha8>` — the row module @ the first 8 hex digits of its source's sha256 (origin=core)."""
    return provider.kernel_word(LEVER)


def kernel_sha8() -> str:
    """sha256[:8] of the kernel module's bytes (the line's kernel= evidence; the same digits impl= carries)."""
    return kernel_impl().rsplit("@", 1)[-1]


def kernel_module():
    """The row module serving this process (None before install)."""
    return _K


def _binding_words() -> dict:
    """The provider tokens of the line: `provider= word= row= cells= tier= [uncovered=]` — `row=<admitted row>:<served calls>` is what RAN (the
    interception serves ONE op per process: the row the word resolved to for the model's main cell); `cells=<arm>:<traced calls>,…` what the
    provider's table names, cell by cell, for the shapes this run served (a per-call binding would serve exactly that; `xla` = the stock
    statement); `tier=<arm>:<cells>,…` the same by cell count; `uncovered=` the served cells the table has no measured number for."""
    b = provider.BINDINGS[LEVER]
    if LEDGER is not None:
        try:                                                                  # every served cell resolved through the provider (memoised per family / size bucket)
            provider.note_cells(LEVER, dict(LEDGER.shapes))
        except Exception as e:                                                # noqa: BLE001 - a census fault prints as a note, never in place of the line
            b.notes["census_error"] = f"{type(e).__name__}: {e}"
    f = provider.facts(LEVER)                                                 # provider, word, row (= per-cell arms : traced calls), tier, [uncovered], [census_error]
    served_row = b.admitted.row if b.admitted is not None else provider.bound_row(LEVER)
    out = {"provider": f["provider"], "word": f["word"], "row": f"{served_row}:{LEDGER.served if LEDGER is not None else 0}", "cells": f["row"], "tier": f["tier"]}
    for k in ("uncovered", "census_error"):
        if k in f:
            out[k] = f[k]
    return out


def _words() -> dict:
    d = _K.describe_op(_K.attention) if _K is not None else {}
    return {"numerics": NUMERICS, "precision": "bf16", "f32_precision": d.get("f32_precision", "tf32"), "bwd": "deterministic", "kernel": kernel_sha8(),
            "scope": SCOPE, **_binding_words(), "source": "exit"}


def off_line(reason: str) -> str:
    """The lever's line in a kit-route mode that does not select it: state=off, zero counters (nothing of the kernel is imported)."""
    return _core_report.lever_line(TAG, LEVER, "off", reason=reason, impl=kernel_impl(), origin=ORIGIN)


def exit_line() -> str:
    """The lever's ONE evidence line of this process: the Ledger's census (served / fallback by reason / shapes) + the numerics words."""
    if LEDGER is None:
        return off_line("not_installed")
    return LEDGER.line(TAG, **_words())


def install() -> dict:
    """Require the Pallas/Triton GPU path (F1.require: a named `F1.Refusal` otherwise), rebind `colabdesign.af.alphafold.model.modules.Attention`
    through F1's interception with op = this kit's kernel, every call served (all_calls, head dims below 16 padded), register the exit line.
    Idempotent. Call before any model function is traced. Returns the install facts."""
    global LEDGER, _K
    facts = F1.require()
    try:
        K = _K = provider.admit(LEVER)                                        # the tier word -> the provider's row for the model's main cell -> its module (cd_triatt: pallas_triatt.triatt_attn)
    except provider.ProviderRefusal as e:
        raise Refusal(f"triatt: {e}") from None
    from colabdesign.af.alphafold.model import modules as CDM
    from . import register_exit_line
    if LEDGER is None:
        LEDGER = Ledger(LEVER, impl=kernel_impl(), origin=ORIGIN, min_tokens=0, expected=EXPECTED_FALLBACKS)
    F1.enable([CDM], ledger=LEDGER, all_calls=True, op=K.attention, pad_head_dim_below_min=True, min_keys=MIN_KEYS)
    register_exit_line(LEVER, exit_line)
    return {"impl": LEDGER.impl, "origin": ORIGIN, "scope": SCOPE, "expected_fallbacks": list(EXPECTED_FALLBACKS), "min_keys": MIN_KEYS, "knobs": K.describe_op(K.attention),
            "row": provider.BINDINGS[LEVER].admitted.row, "word": provider.BINDINGS[LEVER].word, "provider": provider.provider_word(),
            "jax": facts.get("jax") if isinstance(facts, dict) else None}


def uninstall() -> None:
    """Restore the stock `Attention` class (F1.disable); the Ledger and the exit line stay (the census of what this process served)."""
    import sys
    mods = sys.modules.get("colabdesign.af.alphafold.model.modules")
    if mods is not None:
        F1.disable([mods])


def installed() -> bool:
    import sys
    mods = sys.modules.get("colabdesign.af.alphafold.model.modules")
    A = getattr(mods, "Attention", None) if mods else None
    return bool(A is not None and getattr(A, MARKER, False) and LEDGER is not None and getattr(A, "_ledger", LEDGER) is LEDGER)


def evidence() -> dict:
    """The Ledger's fields ({} when not installed), the numerics words and the gate verdict."""
    if LEDGER is None:
        return {}
    g = LEDGER.gate()
    return {**LEDGER.fields(), **_words(), "gate_ok": bool(g.ok), "gate_reason": g.reason, "binding": provider.report(LEVER)}
