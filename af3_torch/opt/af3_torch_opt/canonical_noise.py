"""Canonical diffusion noise (lever ``canonical_noise``): the sampler's random draws are made at the input's own token count — the length
xfold runs at as shipped (no padding) — and laid into the model's padded length, so an input's noise for its real atoms is the noise stock
draws for it, whatever length the kit pads to.

Stock (``xfold/alphafold3.py`` ``AlphaFold3._sample_diffusion`` / ``_apply_denoising_step``) draws, in this order: the initial positions
``randn((S,) + mask.shape + (3,))``, then per sample per step the augmentation's rotation ``randn(2, 3)`` and translation ``randn(3)``
(``diffusion_head.random_augmentation``) and the step noise ``randn(positions.shape)`` — the token axis of ``mask`` / ``positions`` is the
PADDED length, so the same seed gives different noise at a different padded length. Under the ``kernel_tile`` padding policy (fast /
big: ``modes.MODE_PADDING``) the model runs at a padded length ``n`` above the input's token count; here every shape-dependent draw is
made at the canonical length ``C`` = the input's real token count (``batch.token_features.seq_length``: the length stock runs at) and
laid into rows ``0..C-1`` of the model's ``n`` token rows (padding is appended at the end of the token axis, so those rows are the same
tokens in both); rows ``C..n-1`` — padding, masked out of every output — get zeros and no draw. The generator then advances exactly as
stock's does, draw for draw. With ``n == C`` (no padding: ``off`` / ``exact``) the values are the stock statements' bit for bit. The
statements mirror the kit's own methods (``tests/test_kit_statement_mirrors.py`` compares the restatement's source against the kit's,
live, at test time); the class and the kit's files are untouched — ``install`` binds the two methods below to the model instance.
Census: ``COUNTS`` (``take()`` per item).
"""
from __future__ import annotations

import types

COUNTS = {"trajectories": 0, "model_len": None, "canonical_len": None}


def take() -> dict:
    out = dict(COUNTS)
    COUNTS.update(trajectories=0, model_len=None, canonical_len=None)
    return out


def _apply_denoising_step(self, batch, embeddings, positions, noise_level_prev, mask, noise_level):
    """``AlphaFold3._apply_denoising_step`` with the step noise drawn at the canonical length (``self._canonical_len``) and sliced."""
    import torch
    from xfold.nn import diffusion_head

    positions = diffusion_head.random_augmentation(
        positions=positions, mask=mask
    )

    gamma = self.gamma_0 * (noise_level > self.gamma_min)
    t_hat = noise_level_prev * (1 + gamma)

    noise_scale = self.noise_scale * \
        torch.sqrt(t_hat**2 - noise_level_prev**2)
    C = int(self._canonical_len); n = int(positions.shape[0])
    noise = noise_scale * _rows(torch.randn(size=(C,) + tuple(positions.shape[1:]), device=noise_scale.device), n, 0)
    positions_noisy = positions + noise

    if getattr(self.diffusion_head, "use_step_graph", False):
        positions_denoised = self.diffusion_head.forward_graphed(positions_noisy, t_hat, batch, embeddings, True)
    else:
        positions_denoised = self.diffusion_head(positions_noisy=positions_noisy,
                                                 noise_level=t_hat,
                                                 batch=batch,
                                                 embeddings=embeddings,
                                                 use_conditioning=True)
    grad = (positions_noisy - positions_denoised) / t_hat

    d_t = noise_level - t_hat
    positions_out = positions_noisy + self.step_scale * d_t * grad

    return positions_out, noise_level


def _sample_diffusion(self, batch, embeddings):
    """``AlphaFold3._sample_diffusion`` with the initial positions drawn at the canonical length and sliced."""
    import torch
    from xfold.nn import diffusion_head

    if getattr(self, "sample_batch", 0):
        return self._sample_diffusion_batched(batch, embeddings)   # lever 'sbatch': the samples advanced together per step (above)

    mask = batch.predicted_structure_info.atom_mask
    num_samples = self.num_samples

    device = mask.device

    noise_levels = diffusion_head.noise_schedule(
        torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

    n = int(mask.shape[0]); C = int(batch.token_features.seq_length.reshape(-1)[0].item())   # the input's real token count: the length stock draws at
    if C > n:
        raise RuntimeError(f"canonical_noise: the input's token count {C} exceeds the model's padded length {n} (the batch is inconsistent)")
    self._canonical_len = C
    COUNTS["trajectories"] += int(num_samples); COUNTS["model_len"] = n; COUNTS["canonical_len"] = C
    positions = _rows(torch.randn((num_samples, C) + tuple(mask.shape[1:]) + (3,), device=device), n, 1).contiguous()
    positions *= noise_levels[0]

    noise_level = torch.tile(noise_levels[None, 0], (num_samples,))

    if getattr(self.diffusion_head, "use_hoist", False) or getattr(self.diffusion_head, "use_step_graph", False):
        self.diffusion_head.prime_static(batch, embeddings)   # step-invariant conditioning, once per trajectory

    for sample_idx in range(num_samples):
        for step_idx in range(self.diffusion_steps):
            positions[sample_idx], noise_level[sample_idx] = self._apply_denoising_step(
                batch, embeddings, positions[sample_idx], noise_level[sample_idx], mask, noise_levels[1 + step_idx])

    final_dense_atom_mask = torch.tile(mask[None], (num_samples, 1, 1))

    return {'atom_positions': positions, 'mask': final_dense_atom_mask}


def _draw_rows(self, batch, n, num_samples):
    """``AlphaFold3._draw_rows`` under this lever: the kit's sample-batched sampler (lever 'sbatch', ``_sample_diffusion_batched``) makes its
    draws at the rows this returns and lays them into ``n`` (``diffusion_head.lay_rows``, the statement ``_rows`` below restates) — the
    canonical length, found and booked exactly as ``_sample_diffusion`` above finds and books it."""
    C = int(batch.token_features.seq_length.reshape(-1)[0].item())   # the input's real token count: the length stock draws at
    if C > n:
        raise RuntimeError(f"canonical_noise: the input's token count {C} exceeds the model's padded length {n} (the batch is inconsistent)")
    self._canonical_len = C
    COUNTS["trajectories"] += int(num_samples); COUNTS["model_len"] = n; COUNTS["canonical_len"] = C
    return C


def _rows(draw, n, dim):
    """`draw` (token axis `dim`, C rows) laid into `n` token rows: itself when n == C, else zero rows appended (padding tokens: no draw, masked out of every output)."""
    import torch
    C = int(draw.shape[dim])
    if n == C:
        return draw
    pad = list(draw.shape); pad[dim] = n - C
    return torch.cat([draw, torch.zeros(pad, dtype=draw.dtype, device=draw.device)], dim=dim)


def install(model) -> dict:
    """Bind the two methods to ``model`` (an xfold AlphaFold3). Idempotent."""
    if getattr(model, "_canonical_noise", False):
        return {"installed": True, "already": True}
    for name in ("_sample_diffusion", "_apply_denoising_step", "_draw_rows", "gamma_0", "gamma_min", "noise_scale", "step_scale", "diffusion_head"):
        if not hasattr(model, name):
            raise RuntimeError(f"canonical_noise: {type(model).__name__} has no {name!r} (the kit's AlphaFold3 changed shape)")
    model._canonical_len = None
    model._sample_diffusion = types.MethodType(_sample_diffusion, model)
    model._apply_denoising_step = types.MethodType(_apply_denoising_step, model)
    model._draw_rows = types.MethodType(_draw_rows, model)               # the sample-batched sampler's draw length (lever 'sbatch')
    model._canonical_noise = True
    return {"installed": True, "already": False}
