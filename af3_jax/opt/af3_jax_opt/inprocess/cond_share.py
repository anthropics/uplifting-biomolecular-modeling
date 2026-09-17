"""COND_SHARE — the diffusion sampler's per-step conditioning computed ONCE per denoising step, shared by the num_samples (5) samples of that
step, instead of once per sample (kind=schedule; strategy LOCAL.sample_invariant_share).

What stock does (alphafold3/model/network/diffusion_head.py ``sample``): the 200-step sampler is ``hk.scan(hk.vmap(apply_denoising_step,
in_axes=(0, None)), init, noise_levels[1:], unroll=4)`` with a PER-SAMPLE carry ``(key, positions, noise_level_prev)``. ``noise_level_prev``
holds the same number for every sample (the previous element of the schedule), but because it travels in the vmapped carry it is a batched
value, so ``t_hat = noise_level_prev * (1 + gamma)`` is batched and everything the denoising step derives from ``t_hat`` alone is traced
once PER SAMPLE: the noise embedding and the single conditioning (``DiffusionHead._conditioning``'s single half: LayerNorm + projection +
two transitions on [N, 384]), and — inside each of the 24 diffusion-transformer blocks x 2 (attention, transition) — the AdaLN
``single_cond_layer_norm`` + ``single_cond_scale`` / ``single_cond_bias`` projections and the adaptive-zero ``adaptive_zero_cond``
projection ([N, 384] x [384, 768|1536] GEMMs at M = num_samples * N instead of N).

What changes: ONLY where ``t_hat`` comes from. The scan's per-step input becomes the pair ``(noise_level, noise_level_prev)`` =
``(noise_levels[1:], noise_levels[:-1])`` — the same numbers the stock carry held, now an UNBATCHED scan input (``in_axes=(0, None)``) — and
the carry is ``(key, positions)``. Under ``hk.vmap`` a value that does not depend on a batched input is traced unbatched, so the single
conditioning, every per-block conditioning projection and the per-block conditioning LayerNorms (48 x [N, 384] per step) are computed once
per step at M = N and broadcast over the sample axis only where they meet the per-sample stream (``sigmoid(scale) * x + bias``,
``sigmoid(cond) * output``). Every per-sample quantity (key split, augmentation, noise, positions, the network on the per-sample stream) is
unchanged; the arithmetic per element is the stock expression on the stock dtypes. Whether the result is bitwise identical to the lever off
depends on XLA/cuBLAS choosing the same kernels at the smaller M and the same fusions around the broadcast — a property of the compiled
program, not something this source guarantees.

Scope: ``diffusion_head.sample`` is REPLACED by this module's ``sample`` (same signature, same body but for the carry/input split). It is
installed FIRST among the tree levers so SAMPLER_BF16's scope wrapper and ATOM_COND_HOIST's precompute wrapper wrap it (fpf_launch
TREE_LEVERS order). The memory mode's SAMPLES_PER_PASS chunked sampler re-implements ``sample``'s body (big_levers), so this lever is not
part of that mode's composition.
Switch: ``AF3_JAX_COND_SHARE=1`` (``MODEL_OPT_LEVERS_OFF=COND_SHARE`` removes it where the mode table sets it).
Prints: the launcher prints ``XLEVER COND_SHARE state=on traced=<n>`` from ``report()`` -> {"installed", "traced" (sample calls traced
through this body)}.
"""
import os

ENV_SWITCH = "AF3_JAX_COND_SHARE"
REBINDS = ("alphafold3.model.network.diffusion_head:sample",)   # what install() rebinds (module:attribute) — REPLACED (a re-implementation, not a wrapper): installed FIRST so the levers that wrap `sample` wrap this body (fpf_launch INSTALL ORDER; tests/test_install_order.py)
REPLACES = REBINDS                                                # the targets this lever replaces rather than wraps (the install-order test allows one replacer per target, installed before every wrapper)
_STATE = {"installed": False, "traced": 0, "stock": None}


def wanted(environ=os.environ) -> bool:
    return environ.get(ENV_SWITCH, "") == "1"


def _make(DH):
    import haiku as hk
    import jax
    import jax.numpy as jnp

    def sample(denoising_step, batch, key, config):
        """diffusion_head.sample line for line; the carry is (key, positions) per sample, the scan input is (noise_level, noise_level_prev)
        shared by the samples of a step."""
        _STATE["traced"] += 1
        mask = batch.predicted_structure_info.atom_mask

        def apply_denoising_step(carry, levels):
            key, positions = carry
            noise_level, noise_level_prev = levels                        # LEVER: both from the schedule (unbatched under hk.vmap), not from the per-sample carry
            key, key_noise, key_aug = jax.random.split(key, 3)
            positions = DH.random_augmentation(rng_key=key_aug, positions=positions, mask=mask)
            gamma = config.gamma_0 * (noise_level > config.gamma_min)
            t_hat = noise_level_prev * (1 + gamma)
            noise_scale = config.noise_scale * jnp.sqrt(jnp.maximum(t_hat**2 - noise_level_prev**2, 0.0))
            noise = noise_scale * jax.random.normal(key_noise, positions.shape)
            positions_noisy = positions + noise
            positions_denoised = denoising_step(positions_noisy, t_hat)   # t_hat unbatched: the step's conditioning is traced once for the num_samples samples
            grad = (positions_noisy - positions_denoised) / t_hat
            d_t = noise_level - t_hat
            positions_out = positions_noisy + config.step_scale * d_t * grad
            return (key, positions_out), positions_out

        num_samples = config.num_samples
        noise_levels = DH.noise_schedule(jnp.linspace(0, 1, config.steps + 1))
        key, noise_key = jax.random.split(key)
        positions = jax.random.normal(noise_key, (num_samples,) + mask.shape + (3,))
        positions *= noise_levels[0]
        init = (jax.random.split(key, num_samples), positions)
        step = hk.vmap(apply_denoising_step, in_axes=(0, None), split_rng=(not hk.running_init()))
        result, _ = hk.scan(step, init, (noise_levels[1:], noise_levels[:-1]), unroll=4)
        _, positions_out = result
        final_dense_atom_mask = jnp.tile(mask[None], (num_samples, 1, 1))
        return {"atom_positions": positions_out, "mask": final_dense_atom_mask}

    sample._cond_share = True
    return sample


def install() -> bool:
    if _STATE["installed"]:
        return True
    from alphafold3.model.network import diffusion_head as DH
    _STATE["stock"] = DH.sample
    DH.sample = _make(DH)
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if _STATE["installed"]:
        from alphafold3.model.network import diffusion_head as DH
        DH.sample = _STATE["stock"]
        _STATE["installed"] = False


def report() -> dict:
    return {"installed": _STATE["installed"], "traced": _STATE["traced"]}
