"""Fast model construction (lever `fast_init`, switch OF3_FAST_INIT=1): skip the CPU-side weight initialisation that the checkpoint load overwrites.
OpenFold3 builds the model with per-Linear init functions (primitives/initialization.py) plus nn.Linear.reset_parameters and then
load_state_dict()s the checkpoint over it; for inference the init values are dead. OpenFold3 0.5.0's inference runner already skips the
truncated-normal draws (experiment_runner.skip_random_init); enable() turns the remaining init functions and OpenFold3 Linear's
reset_parameters into no-ops, so model construction does no initialisation work. Numerics: exact (every initialised
tensor is overwritten by load_state_dict(strict=True), whose missing-keys list stock checks to be empty or == ['model.version_tensor']).
Usage: of3_fastinit.enable() before the model is constructed; the of3_levers sitecustomize hook calls it when OF3_FAST_INIT=1.
"""

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
