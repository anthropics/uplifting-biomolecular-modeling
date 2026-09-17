"""Lever U0: skip the scipy truncated-normal weight initialisation when the weights are about to be overwritten by a checkpoint.
OpenFold3 0.4.1 builds the model with OpenFold-style per-Linear scipy.stats.truncnorm.rvs() init (CPU-bound, for the 155k model) and then
load_state_dict()s the checkpoint over it. For inference the init values are dead. This module patches the init functions to cheap
no-ops while OF3_FAST_INIT=1. Numerics: EXACT (every initialised tensor is overwritten by load_state_dict(strict=True); we assert that
by checking load_state_dict's missing-keys list is empty or == ['model.version_tensor'] as stock does).
Usage: import of3_fastinit; of3_fastinit.enable()  (before the model is constructed)  -- or PYTHONPATH sitecustomize hook with OF3_FAST_INIT=1.
"""
import os

_ENABLED = False

def enable():
    global _ENABLED
    if _ENABLED:
        return
    import torch
    import openfold3.core.model.primitives.initialization as I
    import openfold3.core.model.primitives.linear as L
    def _noop(weights, *a, **k):
        return weights
    names = ["trunc_normal_init_", "lecun_normal_init_", "he_normal_init_", "glorot_uniform_init_", "final_init_", "gating_init_", "kaiming_normal_init_"]
    for n in names:
        if hasattr(I, n): setattr(I, n, _noop)
        if hasattr(L, n): setattr(L, n, _noop)
    # nn.Linear.__init__ itself calls reset_parameters (kaiming_uniform) -> also dead work; skip it for openfold Linear only
    _orig_reset = torch.nn.Linear.reset_parameters
    def reset_parameters(self):
        if isinstance(self, L.Linear):
            return
        return _orig_reset(self)
    torch.nn.Linear.reset_parameters = reset_parameters
    _ENABLED = True
