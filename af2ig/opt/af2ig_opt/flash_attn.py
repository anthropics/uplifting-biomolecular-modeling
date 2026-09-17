"""L8 — the flash-attention core for the vendored AlphaFold's `Attention` module: the kit adapter over the tree's shared Pallas kernel
(``opt_core.kernels.pallas_attn`` through its serve layer ``opt_core.kernels.pallas_attn_serve``; strategy ``F1.pallas_attn`` of the
model-opt tree; no kit size gate).

The kit's patch 05 (``patches/05_flash_attention.diff``) gives ``alphafold/model/modules.py`` ``Attention.__call__`` one branch: with
``global_config.use_flash_attention`` set (the driver's ``-flash_attn``) the attention core — logits = q·kᵀ + mask bias (+ pair bias),
softmax, weighted sum over v — of every call the kernel takes is computed by :func:`core`; the projections before it and the gating /
output projection after it are the module's own, unchanged. A call the kernel does not take (the serve layer's ONE eligibility rule
``pallas_attn_serve.ineligible``: CPU backend, unequal or < 16 per-head widths, a mask bias not ``[b, 1, 1, k]``, a call without a pair bias;
or fewer keys than the size gate) returns ``None`` and the module runs its stock ops for it. Scope = the calls that carry a pair bias
(triangle attention start/end, MSA row attention: the scope the kernel's home kit ships; MSA-column and template attention stay on the
stock ops). Every decision is counted in the serve layer's :class:`Ledger` (served
shapes; fallbacks by reason name) — the driver copies :func:`census` into its ``flash_attn`` timer record after each newly compiled
shape and prints :func:`line` (the tree's per-lever evidence line, ``[af2ig-opt] LEVER name=F1.pallas_attn state=on|skipped …``) once at
exit, so a run with the lever requested and no kernel in any compiled program is visible in its records (``stack.applied``: L8 with no
served call is a partial activation, exit 3).

Numerics: not bitwise with the stock ops (online-softmax re-association; the kernel's f32 products at the tier's class (modes.F32_PRODUCTS) — tensor-core TF32 products
with f32 accumulation, the same class as the stock's default-precision XLA einsums it replaces; af2ig is f32 end to end) — a
tier-2 lever, composed on the fast line (``modes.resolve`` refuses it under exact and under big on its exact base).

No kit size gate and no kit precision word — the provider (opt_core.kernels.pallas.serve.attention, by tier word) decides the row per cell.
"""
from __future__ import annotations

from typing import Optional

from . import TAG
FACE = "attention"                                 # the provider's serve face this lever binds BY TIER WORD (opt_core.kernels.pallas.serve.attention: pallas_attn | rowshared | cd_triatt | cudnn | … | xla per cell); no kit size gate, no kit precision word
IMPL = "opt_core.pallas.serve.attention"
                                                   # 'tf32' = tensor cores, the class of the stock's default-precision XLA einsums the kernel replaces; 'ieee' is 18-26x slower at these dims and no closer to the stock; 'bf16' is a wider class
_STATE = {"ledger": None, "gate": None, "serve": None, "all_calls": False, "op": None}


def EXPECTED_FALLBACKS(F1) -> tuple:
    """The fallback reasons a healthy af2ig run shows (the Ledger's ``expected`` set: any other reason makes the run partial): below the size gate,
    a call without a pair bias (outside the scope), a per-head width under 16 (this AlphaFold's template-stack attention: 16 calls per program)."""
    return (F1.BELOW_SIZE_RULE, F1.NO_PAIR_BIAS, F1.HEAD_DIM_LT_16)


def _serve():
    """``opt_core.kernels.pallas_attn_serve`` — a core without it is an error naming it (the lever was asked for; no silent stock run)."""
    if _STATE["serve"] is None:
        try:
            from opt_core.kernels import pallas_attn_serve as F1
        except ImportError as e:
            raise RuntimeError(f"-flash_attn: opt_core.kernels.pallas_attn_serve is not importable from this opt_core ({e}); install the "
                               f"tree's core at the kit's pin (opt/pyproject.toml [tool.opt_core]) or drop the flag") from None
        _STATE["serve"] = F1
    return _STATE["serve"]



def configure() -> dict:
    """The lever's fixed settings (pure; no jax): a fresh Ledger; the scope is the calls that carry a pair bias (MSA row attention, triangle attention start/end
    when L10 is off, the template pair stack). No size gate — the provider decides per cell. Returns ``{min_tokens: 0, all_calls, gate_source: 'provider'}``."""
    F1 = _serve()
    all_calls = False
    _STATE.update(gate=None, all_calls=all_calls, ledger=F1.ledger(min_tokens=0, expected=EXPECTED_FALLBACKS(F1), origin=F1.kernel_origin()))
    return {"min_tokens": 0, "all_calls": all_calls, "gate_source": "provider"}


def setup() -> dict:
    """Resolve the kernel scope for this process (before any design): :func:`configure`, then the serve layer's ``require()`` (jax / Pallas / gpu present — a
    named ``Refusal`` makes the MODE refuse by name, ``_fused.refuse``). The attention core itself is bound BY TIER WORD through opt_core's provider
    (``serve.attention``): ``_STATE['op']`` is that binding. Returns the probe updated with the switches."""
    conf = configure()
    F1 = _serve()
    try:
        probe = dict(F1.require())
    except F1.Refusal as r:                                             # the kernel cannot serve in this process (jax below the Pallas floor, Pallas / the kernel absent, a CPU backend): the MODE
        from . import _fused                                            # refuses by name — a mode is all of its levers, never a run under the mode's name without this one
        word = f"cannot_run:{r.kind}(pallas_attn)"
        _fused.refuse("L8", "-flash_attn", word, r, _ledger().line(TAG, state="skipped", reason=word, lever="L8", flag="-flash_attn", dtype="f32"))
    from . import _fused, modes
    P, PS = _fused.provider()
    word = modes.tier_word()
    def op(q, k, v, pair, key_mask, scale, *, kind, n_seq=None):       # the provider's attention face by tier word: q/k/v [b, S, H, D] (the module's layout), pair bias [H, Sq, Sk], key mask [b, Sk]
        return PS.attention(q, k, v, pair, key_mask, scale, word=word, kind=kind, n_seq=n_seq, layout="BSHD")
    _STATE["op"] = op
    probe.update(conf); probe.update(word=word, binding=IMPL)
    return probe


def _ledger():
    if _STATE["ledger"] is None:
        raise RuntimeError("af2ig_opt.flash_attn: setup() was not called before the model was traced (predict_pdb.py calls it when -flash_attn is given)")
    return _STATE["ledger"]


def census() -> dict:
    """The Ledger's fields for the driver's ``flash_attn`` timer record: ``{name, impl, origin, state, served, fallback, fallback_by, errors, min_tokens, shapes,
    partial, calls, first}`` + the tier word and the provider rows that served (``providers``: the provider's own arm names)."""
    from . import _fused, modes
    out = dict(_ledger().fields())
    out.pop("facts", None)
    out["gate_source"] = "provider"
    out["all_calls"] = bool(_STATE["all_calls"])
    out.update(precision=modes.F32_PRODUCTS, word=modes.tier_word(), providers=_fused.served_arms(FACE))
    return out


def _precision_word() -> str:
    """The float32 product class the tier hands the provider (modes.F32_PRODUCTS)."""
    from . import modes
    return modes.F32_PRODUCTS


def providers_word() -> str:
    from . import _fused
    return _fused.arms_word(_fused.served_arms(FACE))


def line() -> str:
    """The per-lever evidence line of this process (the core's LEVER grammar, from the Ledger; ``providers=`` = the provider rows that served)."""
    return _ledger().line(TAG, lever="L8", flag="-flash_attn", dtype="f32", scope="all" if _STATE["all_calls"] else "pair_bias", precision=_precision_word(), providers=providers_word())


def emit() -> str:
    """Print :func:`line` once on stderr (the driver's exit)."""
    return _serve().emit_line(_ledger(), TAG, lever="L8", flag="-flash_attn", dtype="f32", scope="all" if _STATE["all_calls"] else "pair_bias", precision=_precision_word(), providers=providers_word())


def reason_for(key_dim: int, value_dim: int, has_pair_bias: bool, bias_shape: tuple, n_keys: int, backend: Optional[str] = None) -> Optional[str]:
    """Why a call stays on the stock ops (a reason name of the serve layer's scope rules), or None when the provider takes it (no kit size floor)."""
    F1 = _serve()
    return F1.ineligible(key_dim, value_dim, has_pair_bias=has_pair_bias, mask_bias_shape=bias_shape, all_calls=bool(_STATE["all_calls"]), backend=backend)


def kind_for(n_heads: int, head_dim: int) -> str:
    """The provider's attention family kind for an AlphaFold `Attention` call with a pair bias, by its head geometry: 8 heads = the MSA row attention (n_seq = the
    call's batch rows; d < 16 = the extra-MSA stack), 4 heads d 16 = the template pair stack's triangle attention, otherwise triangle attention (`tri`)."""
    if int(n_heads) == 8:
        return "extramsa" if int(head_dim) < 16 else "msarow"
    if int(n_heads) == 4 and int(head_dim) == 16:
        return "tmpl"
    return "tri"


def core(q, k, v, bias, nonbatched_bias, key_dim: int, value_dim: int):
    """The attention core of one `Attention` call. ``q`` ``[b, N_q, h, d]`` already scaled by d**-0.5 (as the module computes it), ``k`` / ``v``
    ``[b, N_k, h, d]``, ``bias`` the mask bias ``1e9 * (mask - 1)`` as ``[b, 1, 1, N_k]``, ``nonbatched_bias`` the pair bias ``[h, N_q, N_k]`` or None.
    Returns the weighted average ``[b, N_q, h, d]`` in q's dtype, or None for a call left on the stock ops (counted with its reason). Served by
    opt_core's provider BY TIER WORD (``serve.attention``, layout BSHD) — the row per cell is the provider's; a row that cannot engage steps aside by name
    inside the provider and its next measured arm serves (the stock statement last)."""
    import jax.numpy as jnp
    F1 = _serve()
    ledger = _ledger()
    why = reason_for(key_dim, value_dim, nonbatched_bias is not None, tuple(int(x) for x in bias.shape), int(k.shape[1]))
    if why is not None:
        ledger.fallback(why)
        return None
    key_mask = bias[:, 0, 0, :] > -1e8                                                        # the module's mask bias is 1e9 * (mask - 1)
    b, sq, h, d = (int(x) for x in q.shape)
    pair = None if nonbatched_bias is None else nonbatched_bias.astype(q.dtype)
    op = _STATE["op"]
    if op is None:
        raise RuntimeError("af2ig_opt.flash_attn: setup() was not called before the model was traced (predict_pdb.py calls it when -flash_attn is given)")
    kind = kind_for(h, d)
    out = op(q, k, v, pair, key_mask, 1.0, kind=kind, n_seq=(b if kind in ("msarow", "extramsa") else None))   # q carries the d**-0.5 scale already
    ledger.serve(F1.shape_key(jnp.swapaxes(q, 1, 2)))
    return out.astype(q.dtype)                                                                # [b, N_q, h, d]                                            # [b, N_q, h, d]
