"""
DDIM sampler configuration.

Default configuration for DDIM (Denoising Diffusion Implicit Model) sampling.
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import required_placeholder


config = ConfigDict(
    {
        "direction_scale": required_placeholder(float),
        "eta": 1.0,
        "n_sample_step": 100,
        "noise_scale": 1.0,
        "verbose": False,
        "predict_sequence": False,
        "predict_sidechain": False,
    }
)
