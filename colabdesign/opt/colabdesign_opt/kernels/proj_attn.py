"""Lever `proj` — the input projections of every kernel-served `Attention` call as ONE GEMM (track PROJ of BINDCRAFT-FASTER).

Stock (`colabdesign/af/alphafold/model/modules.py: Attention.__call__`, and the served form `opt_core.kernels.pallas_attn_serve.
served_attention_call` the `pallas` lever installs) compute the query, key, value and gating projections as FOUR contractions of the SAME
LayerNorm'd activation `q_data` (`'bqa,ahc->…'` ×3 + `'bqc,chv->bqhv'`): four cuBLAS GEMMs with K = C (128 for the pair stack), each
reading the whole [B·S, C] activation — at ≥ 500 tokens these K=128 products sit on the HBM roofline (PROFILE M2: ≈64 FLOP/B, 195–222
TFLOP/s), so their cost IS the bytes they read and write. This lever rebinds `modules.Attention` (a subclass with the SAME class name, created
through haiku's metaclass, so the module name and the parameter tree are unchanged — the stock weights `query_w`, `key_w`, `value_w`,
`gating_w`, `gating_b`, `output_w`, `output_b` load as they are) so that a self-attention call the kernel serves computes

    y = q_data · [ W_q | W_k | W_v | W_gate ]        ONE GEMM, [B·S, C] × [C, 2·H·dk + 2·H·dv], the activation read once

and slices q, k, v (laid heads-major for the kernel, as the served path lays them) and the gate logits out of `y` (when the call is
traced with q_data and m_data as two arrays — stock's sub-batched forward slices them separately from the same activation — the form
is TWO GEMMs, q|gate over q_data and k|v over m_data: correct for any pair, each input still read once). The backward pass
inherits the fusion: ONE dX GEMM (K = 2·H·dk + 2·H·dv, fp32 accumulation, one bf16 rounding) replaces four dX GEMMs and three bf16
partial-sum passes, ONE dW GEMM reads the activation once instead of four times. The attention core is the `pallas` lever's kernel, called
exactly as the served path calls it (same eligibility rule `pallas_attn_serve.ineligible`, same mask / pair-bias forms, the softmax scale from
the ORIGINAL key dim, the same census key counted in the `pallas` lever's ledger — its LEVER line stays the truth about kernel-served calls).

Numerics class: precision (2 = re-associated accumulation) — the same dot products, a different GEMM tiling in the forward pass and one
fp32 accumulation instead of four bf16-rounded partial sums in the backward pass; never claimed bitwise. The softmax scale is left
to the op exactly as the served path passes it (key_dim**-0.5); folding it into q was measured as lever `qscale` and rejected for the
kit, within the mode's accuracy band . Requires the kit's ACTIVE attention-kernel lever (`pallas` today, `triatt` when it supersedes it): this lever composes
ON the serve layer's record for the modules package — its op and its ledger — refuses to install without one (no second attention core,
no silent fallback), and honours the op contract op(q, k, v, pair_bias, key_mask, scale) = "applies exactly `scale`" (key_dim**-0.5,
as the served path passes it; proj never scales q itself — the scale is applied exactly once).
Calls the kernel does not serve (the `pallas` lever's named fallbacks, e.g. `head_dim_lt_16`) run the class below unchanged and are
COUNTED here by reason. Evidence: one LEVER line at exit

    [colabdesign-opt] LEVER name=proj state=on impl=fused_qkvg@kit origin=kit served=<n> fallback=<n> fallback_by=<reason:n,…|none>
                      shapes=<BxSxC>xX<cols>:<n>,… numerics=precision precision=bf16 requires=attn_kernel active=<the active
                      lever's ledger name, e.g. F1.pallas_attn> traced=<n>

(an `opt_core.counters.Ledger`: `served` = calls computed by the one-GEMM form, `fallback` = calls handed to the class below by reason —
`EXPECTED_FALLBACKS` declares the by-design one (the pallas lever's `head_dim_lt_16`); `traced` = Attention calls traced in this process).
The line is printed at exit by the package's one exit printer (`kernels.register_exit_line`) once KIT registers the lever (registry.LEVERS).
Nothing changes a decision — no argument, no environment variable: a mode IS its lever set (modes.py).
"""
from __future__ import annotations

from typing import Optional

from opt_core.counters import Ledger
from opt_core.kernels import pallas_attn_serve as F1

from ..names import TAG

LEVER = "proj"                                       # the lever id (registry.LEVERS row: KIT) = the LEVER line's name=
IMPL = "fused_qkvg@kit"                              # origin=kit: the implementation lives in this package (impl=<name>@kit)
NUMERICS = "precision"                               # registry numerics class: the same dot products, re-associated (one GEMM; one fp32 dX accumulation)
PRECISION = "bf16"                                   # product class of the projections, forward and backward: bf16 operands, fp32 accumulate (stock's class for them)
MARKER = "_colabdesign_opt_proj_attn"
REQUIRES = "attn_kernel"                             # the kit's ACTIVE attention-kernel lever — whichever registered lever rebinds `modules.Attention`
                                                     # through the serve layer (`pallas` = F1's op today; `triatt` when it supersedes pallas): proj
                                                     # composes on the serve layer's record for the modules package (its op AND its ledger), never on
                                                     # a kernel it imports. OP CONTRACT (shared with lever triatt): op(q, k, v, pair_bias,        
                                                     # key_mask, scale) applies exactly `scale` to q.k^T and nothing else; proj passes key_dim**-0.5 exactly as
                                                     # the served call does and never scales q itself - the softmax scale is applied EXACTLY ONCE.
PASS_NO_GATING = "no_gating"                         # config.gating False: nothing in AF-Multimer's attention modules; kept as a named reason
EXPECTED_FALLBACKS = (F1.HEAD_DIM_LT_16, PASS_NO_GATING, F1.NO_PAIR_BIAS, F1.KEY_DIM_NE_VALUE_DIM, F1.BIAS_FORM, F1.BELOW_KEYS_RULE, F1.BELOW_SIZE_RULE)   # every STRUCTURAL per-call step-aside word this lever can print (a call whose module config / shape is not kernel work runs stock's projections BY NAME: the extra-MSA row attention (8 per head), modules without gating (AF2-ptm monomer template stack), attention without a pair bias, ...); environment words (backend_not_gpu, probe failures) stay undeclared = fail-closed


class LeverError(RuntimeError):
    pass


REFUSALS = (LeverError, F1.Refusal)                  # cannot-run here: no attention-kernel lever active (LeverError) or the kernel's own refusal (F1.Refusal)
LEDGER: Optional[Ledger] = None                      # this lever's per-process census (served = one-GEMM calls; fallback = handed to the class below, by reason)
_STATE = {"traced": 0, "installed_on": None, "exit_line_registered": False, "active": None}


def _ledger() -> Ledger:
    global LEDGER
    if LEDGER is None:
        LEDGER = Ledger(LEVER, impl=IMPL, origin="kit", expected=EXPECTED_FALLBACKS)
    return LEDGER


def serve_record(modules_pkg=None) -> dict:
    """The serve layer's record for colabdesign's `modules` package — {"module", "stock_cls", "ledger", "op"} as the ACTIVE attention-kernel
    lever enabled it (`pallas_attn_serve.enable(..., ledger=, op=)`). LeverError when no kernel lever is active on it."""
    if modules_pkg is None:
        from colabdesign.af.alphafold.model import modules as modules_pkg  # noqa: WPS433
    rec = F1._ENABLED.get(id(modules_pkg))
    if rec is None:
        raise LeverError(f"lever {LEVER} requires the kit's attention-kernel lever active on {modules_pkg.__name__} (the serve layer has no record)")
    return rec


def active_op(rec: dict):
    """The op the active kernel lever serves attention with: its own (`enable(op=...)`, e.g. triatt's) or, when it enabled the serve layer's
    default (pallas: op=None), the carried kernel module's `flash_attention` — resolved at trace time exactly as the served call resolves it."""
    return rec.get("op") or F1.kernel_module().flash_attention


def _eligibility_kwargs(rec: dict) -> dict:
    """The served call's per-call rule inputs: every call (all_calls), the active ledger's size gate (min_tokens; 0 on this engine), no padding."""
    mt = getattr(rec.get("ledger"), "min_tokens", None) or 0
    return {"all_calls": True, "min_tokens": int(mt), "min_keys": 0, "pad_head_dim": False}


def fused_projection_weights(*weights):
    """[C, Σ H·d] — stock weights `[C, H, d]` side by side (a small bf16 concat per call: 128×512 elements for the pair stack)."""
    import jax.numpy as jnp  # noqa: WPS433
    C = weights[0].shape[0]
    return jnp.concatenate([w.reshape(C, -1) for w in weights], axis=1)


def project(x, weights):
    """x `[B, S, C]` · [w0 | w1 | …] as ONE GEMM → the projections `[B, S, H, d_i]` in the weights' order (token-major; views of one buffer)."""
    import jax.numpy as jnp  # noqa: WPS433
    W = fused_projection_weights(*weights)
    y = jnp.einsum("bsa,ax->bsx", x, W)
    B, S = y.shape[0], y.shape[1]
    outs, o = [], 0
    for w in weights:
        H, d = int(w.shape[1]), int(w.shape[2])
        outs.append(y[..., o:o + H * d].reshape(B, S, H, d)); o += H * d
    return outs, int(W.shape[1])


def heads_major(x):
    """`[B, S, H, d]` → `[B, H, S, d]` (the kernel's layout; XLA materialises it, as it does for the served path's 'bqa,ahc->bhqc')."""
    import jax.numpy as jnp  # noqa: WPS433
    return jnp.transpose(x, (0, 2, 1, 3))


def split_fused(y, num_head: int, key_dim: int, value_dim: int, gating: bool):
    """y `[B, S, X]` (X = q|k|v|gate columns) → q, k, v heads-major `[B, H, S, d]` and the gate logits `[B, S, H, dv]` (or None)."""
    B, S = y.shape[0], y.shape[1]
    o = 0
    q = y[..., o:o + num_head * key_dim].reshape(B, S, num_head, key_dim); o += num_head * key_dim
    k = y[..., o:o + num_head * key_dim].reshape(B, S, num_head, key_dim); o += num_head * key_dim
    v = y[..., o:o + num_head * value_dim].reshape(B, S, num_head, value_dim); o += num_head * value_dim
    g = y[..., o:o + num_head * value_dim].reshape(B, S, num_head, value_dim) if gating else None
    return heads_major(q), heads_major(k), heads_major(v), g


def fused_attention_call(self, q_data, m_data, bias, nonbatched_bias=None) -> Optional[object]:
    """The one-GEMM form of a served self-attention call for the haiku `Attention` instance `self` (called inside its own name scope).
    Returns None (counted by reason) when the call is not for this lever: the class below then runs unchanged."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    import haiku as hk  # noqa: WPS433

    _STATE["traced"] += 1
    rec = serve_record(_STATE["installed_on"])
    num_head = self.config.num_head
    key_dim = self.config.get("key_dim", int(q_data.shape[-1])) // num_head
    value_dim = self.config.get("value_dim", int(m_data.shape[-1])) // num_head
    why = F1.ineligible(key_dim, value_dim, nonbatched_bias is not None, getattr(bias, "shape", None), n_queries=int(q_data.shape[-2]),
                        n_keys=int(m_data.shape[-2]), **_eligibility_kwargs(rec))
    if why is None and not self.config.gating:
        why = PASS_NO_GATING
    if why is not None:
        _ledger().fallback(why)
        return None
    glorot_uniform = lambda: hk.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")  # noqa: E731
    C = int(q_data.shape[-1])
    q_w = hk.get_parameter("query_w", shape=(C, num_head, key_dim), dtype=q_data.dtype, init=glorot_uniform())
    k_w = hk.get_parameter("key_w", shape=(C, num_head, key_dim), dtype=q_data.dtype, init=glorot_uniform())
    v_w = hk.get_parameter("value_w", shape=(C, num_head, value_dim), dtype=q_data.dtype, init=glorot_uniform())
    g_w = hk.get_parameter("gating_w", shape=(C, num_head, value_dim), dtype=q_data.dtype, init=hk.initializers.Constant(0.0))
    g_b = hk.get_parameter("gating_b", shape=(num_head, value_dim), dtype=q_data.dtype, init=hk.initializers.Constant(1.0))
    if q_data is m_data:                                                               # self-attention traced on one array: ONE GEMM q|k|v|gate
        (q, k, v, gate_logits), cols = project(q_data, [q_w, k_w, v_w, g_w]); form = f"X{cols}"
    else:                                                                              # two tracers (stock's sub-batched forward slices q_data and m_data
        (q, gate_logits), cq = project(q_data, [q_w, g_w])                             # from the same array separately): TWO GEMMs, q|gate and k|v —
        (k, v), ckv = project(m_data, [k_w, v_w]); form = f"X{cq}+{ckv}"              # correct for any q/m pair, each input still read once
    q, k, v = heads_major(q), heads_major(k), heads_major(v)
    scale = float(key_dim ** (-0.5))                                                   # the ORIGINAL key dim's, as the served path passes it
    kmask = bias[:, 0, 0, :] > F1.MASKED_BIAS_THRESHOLD
    sq, sk = q.shape[2], k.shape[2]
    nb = jnp.zeros((num_head, sq, sk), q.dtype) if nonbatched_bias is None else nonbatched_bias.astype(q.dtype)
    census = F1.shape_key(q)
    op = active_op(rec)                                                                # the ACTIVE kernel lever's op (pallas: F1; triatt: its own)
    weighted_avg = op(q, k, v, nb, kmask, scale)                                       # [B, H, S, dv]; contract: the op applies exactly `scale`
    weighted_avg = jnp.swapaxes(weighted_avg, 1, 2)                                    # [B, S, H, dv]
    weighted_avg *= jax.nn.sigmoid(gate_logits + g_b)
    init = hk.initializers.Constant(0.0) if self.global_config.zero_init else glorot_uniform()
    o_w = hk.get_parameter("output_w", shape=(num_head, value_dim, self.output_dim), dtype=q_data.dtype, init=init)
    o_b = hk.get_parameter("output_b", shape=(self.output_dim,), dtype=q_data.dtype, init=hk.initializers.Constant(0.0))
    if rec.get("ledger") is not None:
        rec["ledger"].serve(census)                                                    # the kernel served this call: the ACTIVE lever's census stays whole
    _ledger().serve(f"{q_data.shape[0]}x{q_data.shape[1]}x{C}x{form}")
    return jnp.einsum("bqhc,hco->bqo", weighted_avg, o_w) + o_b


def install() -> dict:
    """Rebind `colabdesign.af.alphafold.model.modules.Attention` ON TOP of the `pallas` lever's rebinding (idempotent; marker MARKER).
    Refuses without the `pallas` lever installed on that module object. Call before any model function is traced."""
    from colabdesign.af.alphafold.model import modules as CDM  # noqa: WPS433
    A = CDM.Attention
    if getattr(A, MARKER, False):
        return info()
    rec = serve_record(CDM)                                              # LeverError unless a kernel lever is active on the package
    if not getattr(A, "_opt_core_pallas_attn", False):                   # the serve layer's marker on the class it rebound
        raise LeverError(f"lever {LEVER} requires the kit's attention-kernel lever installed first (modules.Attention is not the served class)")
    _STATE["active"] = getattr(rec.get("ledger"), "name", None) or "unknown"
    below_call = A.__call__

    def _call(self, q_data, m_data, bias, nonbatched_bias=None):
        out = fused_attention_call(self, q_data, m_data, bias, nonbatched_bias)
        return below_call(self, q_data, m_data, bias, nonbatched_bias) if out is None else out

    class Attention(A):                              # noqa: D101 — SAME name, defined in a class body so haiku's metaclass wraps __call__
        _colabdesign_opt_proj_attn = True
        _below_cls = A

        def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
            return _call(self, q_data, m_data, bias, nonbatched_bias)

    Attention.__qualname__ = "Attention"
    Attention.__module__ = A.__module__
    CDM.Attention = Attention
    _STATE["installed_on"] = CDM
    _ledger()
    from .. import registry                          # noqa: WPS433
    if LEVER in registry.LEVERS:                     # the package's one exit printer (kernels/__init__.py) prints this lever's line at exit, registry order
        from . import register_exit_line             # noqa: WPS433
        _STATE["exit_line_registered"] = bool(register_exit_line(LEVER, exit_line)) or _STATE["exit_line_registered"]
    return info()


def uninstall() -> bool:
    """Restore the class this lever wrapped (the `pallas` lever's); re-trace afterwards. True when something was restored."""
    import sys
    CDM = sys.modules.get("colabdesign.af.alphafold.model.modules")      # nothing to undo in a process that never imported the model
    if CDM is None:
        _STATE["installed_on"] = None
        return False
    A = CDM.Attention
    if getattr(A, MARKER, False):
        CDM.Attention = A._below_cls
        _STATE["installed_on"] = None
        return True
    return False


def installed() -> bool:
    import sys
    m = sys.modules.get("colabdesign.af.alphafold.model.modules")
    return bool(m and getattr(getattr(m, "Attention", None), MARKER, False))


def info() -> dict:
    return {"lever": LEVER, "impl": IMPL, "numerics": NUMERICS, "requires": REQUIRES, "active": _STATE.get("active"), "installed": installed(),
            "expected_fallbacks": list(EXPECTED_FALLBACKS), "exit_line_registered": bool(_STATE["exit_line_registered"])}


def evidence() -> dict:
    led = _ledger()
    g = led.gate(require_served=False)
    return {**led.fields(), "traced": int(_STATE["traced"]), "gate_ok": bool(g.ok), "gate_reason": g.reason}


def _evidence_pairs() -> dict:
    return {"numerics": NUMERICS, "precision": PRECISION, "requires": REQUIRES, "active": _STATE.get("active") or "none",
            "traced": int(_STATE["traced"])}


def exit_line() -> str:
    """The lever's ONE line as it stands (state=on with the census when installed; the ledger's skipped/partial words otherwise)."""
    return _ledger().line(TAG, "on" if installed() else None, **_evidence_pairs())


def lever_line(state: Optional[str] = None, reason: Optional[str] = None) -> str:
    return _ledger().line(TAG, state if state is not None else ("on" if installed() else "off"), reason, **_evidence_pairs())


def off_line(reason: Optional[str] = None) -> str:
    """The lever's line in a mode that does not select it: state=off, zero counters."""
    return Ledger(LEVER, impl=IMPL, origin="kit", expected=EXPECTED_FALLBACKS).line(TAG, "off", reason, numerics=NUMERICS, precision=PRECISION, requires=REQUIRES)
