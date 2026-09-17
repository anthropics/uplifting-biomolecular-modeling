"""The batched line's per-batch reverse-diffusion sampling — ONE function, called once per batch by g3batch.py: the initial noise over the
batch's ``gt_atom_positions`` (upstream sampler.py:98), then per DDIM step one denoiser call (the CUDA-graph wrapper `gd` when the line
replays graphs, else the model's own forward) and the step arithmetic with its noise draw (upstream ddim.py ``_step``: ``torch.randn_like(xs)``
on every step but the last). The statement order and every RNG draw are upstream's, so the generator advances exactly as stock's sampler does
at the same batch size; nothing in this module times, counts or prints anything.
"""
from __future__ import annotations

import torch


def sample_batch(gd, model, bd, tab, SB, mask, ddim_math):
    """Sample one batch: returns ``xs``, the batch's denoised CA positions after the last DDIM step ([B, n_token, 3]).

    gd         the CUDA-graph denoiser wrapper for this batch's shape (g3fast.GraphedDenoiser: eager pre-part + graph replay), or None = eager
    model      the denoiser module (used when gd is None: ``model(batch=bd, xl=xs, t=s / n_timestep)["xl"]``, upstream's call)
    bd         the batch's feature dict (``gt_atom_positions`` gives the noise its shape, dtype and device)
    tab        the sync-free step tables (g3fast.StepTables: ``steps``, ``is_last``, ``n_timestep`` and the DDIM coefficients ddim_math reads)
    SB         the per-step timestep vectors for this batch size (SB[k] = ``torch.Tensor([step] * B).int()`` on the device, built once per B)
    mask       ``bd["gt_atom_mask"]``
    ddim_math  g3fast.ddim_math — the DDIM update of one step (x_s, the denoiser's x0 estimate, the mask, the step's noise -> x_{s - step})
    """
    xs = torch.randn_like(bd["gt_atom_positions"])           # sampler.py:98
    for k in range(len(tab.steps)):
        s_vec = SB[k]
        out_xl = gd(xs, s_vec) if gd is not None else model(batch=bd, xl=xs, t=s_vec / tab.n_timestep)["xl"]
        noise = None if tab.is_last[k] else torch.randn_like(xs)   # ddim.py _step: torch.randn_like(xs)
        xs = ddim_math(tab, k, s_vec, xs, out_xl, mask, noise)
    return xs
