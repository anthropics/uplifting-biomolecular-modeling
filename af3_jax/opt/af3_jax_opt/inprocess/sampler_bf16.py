"""SAMPLER_BF16 — the diffusion sampler's f32 GEMM islands run with bf16 tensor-core operands (kind=precision; numerics NOT bitwise: a
tolerance-class fast lever). Shared strategy id ``F8.sampler_lowp``; implemented as a two-function Haiku rebinding of
``diffusion_transformer`` plus a scope on ``diffusion_head.sample``.

What stock runs in f32 (alphafold3/model/network/diffusion_head.py ``DiffusionHead.__call__``; the FlashPairformer hoist's
``HoistDiffusionHead.__call__`` keeps the same casts): the trunk embeddings reach the sampler as f32 (model.py ``recycle_body``:
``embeddings['pair'|'single'].astype(jnp.float32)``) and the head casts ``act``, ``trunk_single_cond``, ``trunk_pair_cond``, ``sequence_mask``
to f32 before the 24-block diffusion transformer (diffusion_head.py lines 299 and 314-317), so every module the 200-step x num_samples loop
traces — the per-step single conditioning, the atom cross-attention encoder (3 blocks) and decoder (3 blocks), and the 24 DiT blocks
(adaptive LayerNorm, q/k/v/gate projections, attention, output projection, adaptive-zero gate, SwiGLU transition) — computes on f32
operands (``hm.Linear`` follows its input dtype; ``utils.bfloat16_context`` only casts parameters a bf16 INPUT requests). The trunk and the
confidence head run bf16 under ``GlobalConfig.bfloat16 = 'all'``; the sampler is the model's one f32 island.

What changes: ONLY the operand dtype of the GEMMs inside the sampler's transformer blocks. Two functions of diffusion_transformer are
rebound, active only while ``diffusion_head.sample`` is being traced (the scope both the stock ``Model._sample_diffusion`` and the hoisted
sampler enter; the pair-conditioning hoist, the trunk and the confidence head are outside it) and only for an f32 input (a bf16 caller —
the trunk pairformer's single attention, which calls the same functions — is untouched, so the rebinding is inert outside the island):
  * ``adaptive_layernorm``: the LayerNorm statistics of the residual stream ``x`` and of ``single_cond`` are computed in f32 exactly as stock
    (``hm.LayerNorm`` on the f32 tensors); the AdaLN scale / bias projections read bf16 operands; the modulation ``sigmoid(scale) * x + bias``
    promotes to f32; the block input is handed on as bf16 — so the q / k / v / gating projections, the attention's second product, the
    SwiGLU transition (``tokamax.gated_linear_unit``) and the output projection (``transition2``) all read bf16 operands with f32 accumulation
    inside the GEMM (XLA's bf16 dot) and ONE bf16 rounding of each result.
  * ``adaptive_zero_init``: the adaptive-zero gate projection of ``single_cond`` reads bf16 operands; the gated block output returns to the
    caller, whose residual add ``act += ...`` is on the f32 stream (the hk.layer_stack carry stays f32: the stream, every residual sum and every
    LayerNorm input keep stock's dtype).
Kept in f32 BY RULE (the fork accumulates or pins them there): LayerNorm statistics; softmax logits and statistics (the stock attention casts
q, k to f32 for the logits einsum — unchanged; under DATTN the flash kernel takes bf16 q/k/v and accumulates logits / softmax / output in
f32); every ``precision='highest'`` projection of the head (per-atom position embedding, the trunk
conditioning projections, ``single_cond_embedding_projection``, the decoder's position update); the residual streams (token [N, 768] and
atom [subsets, 32|128, 128]); the sampler's own arithmetic (noise, Karras update) and the pair conditioning + per-block pair logits.
Every call site is served because each resolves the module attribute at call time: diffusion_transformer ``self_attention`` /
``cross_attention`` / ``transition_block`` name ``adaptive_layernorm`` and ``adaptive_zero_init`` as module globals, DATTN's replacement
``self_attention`` calls ``DT.adaptive_layernorm`` / ``DT.adaptive_zero_init``, the hoist's step calls ``DT.self_attention`` /
``DT.transition_block``. ``report()["sites"]`` counts traced casts by the stock ``name`` argument (``token`` = the DiT and the single
transformer's transitions; ``atom`` = the atom transformer encoder/decoder). The UNCONDITIONED transition blocks — the pair and single conditioning paths of
``DiffusionHead._conditioning`` (per step, or once per sample under the FlashPairformer hoist) — are left stock (f32) by rule, so the lever's
sites are the same set on every line of the kit (fast; the memory mode's single-card line, whose SAMPLES_PER_PASS sampler enters ``scope``;
its row-sharded line, whose attention transcription resolves the two functions at call time).

Switch: ``AF3_JAX_SAMPLER_BF16=1`` in the model process's environment (set by the mode table for a composition that names SAMPLER_BF16;
modes.TREE_LEVER_ENV; the package's prefix, declared in stack.LEVER_SWITCH_ENV). Evidence: ``report()`` -> {"installed", "traced" (cast
sites traced inside the sampler scope), "sites", "passed" (calls left stock: outside the scope or not f32), "recipe"} — the tree's launcher
prints ``sbf16=<traced casts>|off sbf16_sites=<site:n,...|none>`` on its SERVED line. Ablation: ``MODEL_OPT_LEVERS_OFF=SAMPLER_BF16``
removes the variable (modes.with_levers_off), nothing is rebound, every line is the base's.
"""
import os

ENV_SWITCH = "AF3_JAX_SAMPLER_BF16"
REBINDS = ("alphafold3.model.network.diffusion_transformer:adaptive_layernorm", "alphafold3.model.network.diffusion_transformer:adaptive_zero_init", "alphafold3.model.network.diffusion_head:sample")   # what install() rebinds (module:attribute) — the two conditioning functions (scoped wrappers) and the sampler entry that opens the scope; the launcher's INSTALL_ORDER table (fpf_launch.py) and tests/test_install_order.py read it
RECIPE = "bf16_operands(f32_stream,f32_ln_stats,f32_softmax,highest_pinned_f32)"   # the precision words of this lever (the probe and the notes print them)
_STATE = {"installed": False, "depth": 0, "traced": 0, "passed": 0, "sites": {}, "stock": {}}


def wanted(environ=os.environ) -> bool:
    return environ.get(ENV_SWITCH, "") == "1"


def _site(name) -> str:
    return "atom" if "atom" in str(name) else "token"


def _make(DT):
    """The two stock functions (diffusion_transformer.py ``adaptive_layernorm`` / ``adaptive_zero_init``) line for line, with the operand
    casts of this lever at the marked points; taken only inside the sampler scope on an f32 input, else the stock function runs."""
    import jax
    import jax.numpy as jnp
    hm = DT.hm
    LOW = jnp.bfloat16
    stock_aln, stock_azi = DT.adaptive_layernorm, DT.adaptive_zero_init

    def adaptive_layernorm(x, single_cond, name):
        if _STATE["depth"] <= 0 or x.dtype != jnp.float32 or single_cond is None:          # outside the sampler, not an f32 island, or an UNCONDITIONED
            _STATE["passed"] += 1                                                           # transition (the pair / single conditioning paths, diffusion_head
            return stock_aln(x, single_cond, name)                                          # `_conditioning`): stock, f32 — the same sites on every line of the kit
        _STATE["traced"] += 1
        site = _site(name)
        _STATE["sites"][site] = _STATE["sites"].get(site, 0) + 1
        x = hm.LayerNorm(name=f"{name}layer_norm", use_fast_variance=False, create_scale=False, create_offset=False)(x)   # f32 statistics on the f32 stream (stock)
        single_cond = hm.LayerNorm(name=f"{name}single_cond_layer_norm", use_fast_variance=False, create_offset=False)(single_cond)  # f32 statistics (stock)
        single_cond = single_cond.astype(LOW)                                               # LEVER: the AdaLN projections read bf16 operands
        single_scale = hm.Linear(x.shape[-1], initializer="zeros", use_bias=True, name=f"{name}single_cond_scale")(single_cond)
        single_bias = hm.Linear(x.shape[-1], initializer="zeros", name=f"{name}single_cond_bias")(single_cond)
        x = jax.nn.sigmoid(single_scale) * x + single_bias                                  # promotes to f32 (x is f32): the modulation as stock
        return x.astype(LOW)                                                                # LEVER: the block's GEMMs read bf16 operands

    def adaptive_zero_init(x, num_channels, single_cond, global_config, name):
        if _STATE["depth"] <= 0 or single_cond is None or single_cond.dtype != jnp.float32 or x.dtype != LOW:
            _STATE["passed"] += 1
            return stock_azi(x, num_channels, single_cond, global_config, name)
        output = hm.Linear(num_channels, name=f"{name}transition2")(x)                     # bf16 operands (x is the bf16 block output), as every GEMM of the block
        cond = hm.Linear(output.shape[-1], initializer="zeros", use_bias=True, bias_init=-2.0, name=f"{name}adaptive_zero_cond")(
            single_cond.astype(LOW))                                                        # LEVER: the gate projection reads bf16 operands
        return jax.nn.sigmoid(cond) * output                                                # bf16; the caller's residual add is on its f32 stream

    return adaptive_layernorm, adaptive_zero_init


class scope:
    """The sampler scope as a context: raised by the kit's ``diffusion_head.sample`` wrapper below while it traces, and entered by any lever
    that re-implements ``sample``'s body instead of calling it (the memory mode's SAMPLES_PER_PASS chunked sampler reads it off the function
    it wraps: ``getattr(sample, "_sampler_bf16_scope")``) — so the lever engages on every line that names it, never silently absent."""
    def __enter__(self):
        _STATE["depth"] += 1
        return self

    def __exit__(self, *exc):
        _STATE["depth"] -= 1
        return False


def _scoped_sample(stock_sample):
    """``diffusion_head.sample`` with the scope raised while it traces: the denoising steps (single conditioning, atom encoder, DiT, atom
    decoder) are inside; the hoist's precompute, the trunk and the confidence head are outside."""
    def sample(denoising_step, batch, key, config):
        with scope():
            return stock_sample(denoising_step=denoising_step, batch=batch, key=key, config=config)
    sample._sampler_bf16_scoped = True
    sample._sampler_bf16_scope = scope
    sample._sampler_bf16_inner = stock_sample                                 # NOT ``__wrapped_stock__``: that name is the memory mode's own mark of ITS chunked sampler (big_levers: 'applied twice' guard)
    return sample


def install() -> bool:
    """Rebind diffusion_transformer.adaptive_layernorm / adaptive_zero_init and scope diffusion_head.sample (idempotent). Returns whether
    the lever is installed in this process."""
    if _STATE["installed"]:
        return True
    from alphafold3.model.network import diffusion_transformer as DT
    from alphafold3.model.network import diffusion_head as DH
    _STATE["stock"] = {"aln": DT.adaptive_layernorm, "azi": DT.adaptive_zero_init, "sample": DH.sample}
    DT.adaptive_layernorm, DT.adaptive_zero_init = _make(DT)
    DH.sample = _scoped_sample(DH.sample)
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if _STATE["installed"]:
        from alphafold3.model.network import diffusion_transformer as DT
        from alphafold3.model.network import diffusion_head as DH
        DT.adaptive_layernorm, DT.adaptive_zero_init, DH.sample = _STATE["stock"]["aln"], _STATE["stock"]["azi"], _STATE["stock"]["sample"]
        _STATE["installed"] = False


def report() -> dict:
    return {"installed": _STATE["installed"], "traced": _STATE["traced"], "passed": _STATE["passed"], "sites": dict(_STATE["sites"]),
            "recipe": RECIPE}
