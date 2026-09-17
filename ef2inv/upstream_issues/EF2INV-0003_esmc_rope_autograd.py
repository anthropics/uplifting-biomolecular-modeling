"""EF2INV-0003 — ESM-C's rotary embedding runs outside autograd when flash-attn is installed (transformers fork `ef32577f`).

Applied only on request: ``run.sh design|warm --upstream-fix EF2INV-0003`` (every mode, ``off`` included; opt/ef2inv_opt/upstream_fix.py
loads this file by path in the arm process before any model is built and prints ``[ef2inv-opt] UPSTREAM-FIX EF2INV-0003 applied …``;
run.json ``upstream_fix`` / ``patches`` and opt_manifest.json ``upstream_fix`` / ``deviations`` record it; the ATTN line reads
``esmc_rope=torch``). Nothing imports this file otherwise; no lever, mode or config references it. Without the flag ESM-C runs as the
fork imports it (``esmc_rope=flash_triton`` on a box with flash-attn and a CUDA device).

WHAT UPSTREAM DOES. ``transformers/models/esmc/modeling_esmc.py`` l.56-62 imports ``flash_attn.ops.triton.rotary.apply_rotary`` and sets
``_flash_attn_rotary_available = torch.cuda.is_available()``; ``RotaryEmbedding.forward`` (l.424-429) then rotates q and k with that
function on every ESM-C attention layer. In flash-attn 2.8.3 ``flash_attn/ops/triton/rotary.py`` ``apply_rotary`` (l.102-149) is the raw
kernel driver — it allocates ``output = torch.empty_like(x)`` and launches the Triton kernel with no ``torch.autograd.Function`` around
it (the autograd-aware entry point is ``flash_attn.layers.rotary.apply_rotary_emb`` / ``ApplyRotaryEmb``, whose backward re-launches the
kernel with ``conjugate=True``; the fork does not use it). The rotated q/k therefore carry no ``grad_fn``: a loss differentiated through
ESM-C keeps its value / residual / MLP paths and silently loses the attention-score (q·k) path. Nothing raises.

WHERE IT ENTERS THE DESIGN LOOP. The cookbook's step (stock/src/cookbook/tutorials/binder_design.py l.1084-1098) takes
``plm_grad = torch.autograd.grad(plm_loss.mean(), logits)`` through ESMC-6B (``LM_MASK_PASSES`` masked passes of ``LM_LOSS_BATCH_SIZE`` per
step) and sets ``logits.grad = normalize(structure_grad) + 0.15 · normalize(plm_grad)`` (0.05 for antibodies). The language-model
gradient is L2-normalised before weighting, so the missing path changes the DIRECTION of that term, not only its size. The fold's own use
of ESM-C hidden states is forward-only; there the two paths differ by bf16 rounding at most (the torch path rotates in the activation
dtype, the Triton kernel accumulates in fp32 and rounds once).

EVIDENCE. On the pinned stack, as imported: ``modeling_esmc._flash_attn_rotary_available is True``; for q, k with ``requires_grad=True``,
``RotaryEmbedding(64).cuda()(q, k)`` returns tensors with ``grad_fn is None`` and ``requires_grad False``; after this fix the rotated q
carries a ``grad_fn`` and ``d(sum(q_rot))/dq`` is non-zero.

WHAT THE FIX CHANGES. One module attribute: ``modeling_esmc._flash_attn_rotary_available = False`` (rebound after the cookbook import,
before any model is built), so ``RotaryEmbedding.forward`` takes its own ``_apply_rotary_emb_torch`` branch (l.255-271: plain tensor
arithmetic that autograd differentiates) on every ESM-C layer, in the design step's pseudo-perplexity term and in the fold's
language-model feature pass alike. No file is edited; the record (``name esmc_rope_pin``, ``impl torch``, ``as_imported``, ``applied``)
is the one opt/ef2inv_opt/patches.py ``pin_esmc_rope`` returns.

EXPECTED EFFECT ON OUTPUTS. Different design trajectories from the same seed (the 0.15-weighted language-model gradient gains its
attention-score path); forward values equal to bf16 rounding. Speed: rotary embedding is elementwise work beside ESMC-6B's GEMMs — a few
eager tensor ops per attention layer in place of one fused kernel, on the arm that carries the flag; small next to a design step.
The upstream remedy is a one-line import change in the fork (``flash_attn.layers.rotary.apply_rotary_emb`` in place of
``flash_attn.ops.triton.rotary.apply_rotary``, with its (batch, seqlen, nheads, headdim) layout).
"""
ID = "EF2INV-0003"
SUMMARY = "ESM-C RoPE on the fork's differentiable torch path (modeling_esmc._flash_attn_rotary_available = False): the pseudo-perplexity gradient keeps its attention-score path"
MODULE = "transformers.models.esmc.modeling_esmc"
FLAG_ATTR = "_flash_attn_rotary_available"
TARGETS = [f"{MODULE}.{FLAG_ATTR}"]


def apply(log=None, module=None) -> dict:
    """Rebind the fork's flag through the package's one RoPE pin (opt/ef2inv_opt/patches.py ``pin_esmc_rope()``); ``module`` injects a
    stand-in object for the CPU tests. Returns that record plus ``summary`` / ``targets`` (upstream_fix.apply adds ``id`` / ``file`` and
    prints the UPSTREAM-FIX line)."""
    from ef2inv_opt import patches as PT
    rec = dict(PT.pin_esmc_rope(module=module))
    rec["summary"] = SUMMARY
    rec["targets"] = list(TARGETS)
    return rec
