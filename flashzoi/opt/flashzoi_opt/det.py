"""The deterministic recipe (`--det`), the same recipe on both routes: the process-level switches that make two runs of the
documented call byte-identical on one machine. Applied by the stock caller and the kit driver alike — stock-side switches, never a mode.

    env   CUBLAS_WORKSPACE_CONFIG=:4096:8      (before CUDA initialises; the CLI exports it into the process before torch is imported)
    torch.manual_seed(0)                       (inert: the documented call samples nothing; recorded)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    (the TF32 switches stay at torch's own defaults: the stock's numerics under every recipe)
    batch 1, torch.autocast("cuda") (fp16 on the pinned stack), the model's fp32 head (autocast disabled inside the package)
"""
from __future__ import annotations

import os

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_VALUE = ":4096:8"
CUBLAS_ACCEPTED = (":4096:8", ":16:8")
SEED = 0
RECIPE = {"seed": SEED, "deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
          "tf32": "untouched (torch's defaults)", "cublas_workspace_config": CUBLAS_VALUE, "batch": 1, "autocast": "cuda"}


class DetRecipeError(RuntimeError):
    """The recipe cannot be honoured in this process (CUBLAS_WORKSPACE_CONFIG missing after CUDA initialised)."""


def export_env(environ=None) -> dict:
    """Set CUBLAS_WORKSPACE_CONFIG (kept when already an accepted value); call before torch is imported."""
    environ = os.environ if environ is None else environ
    if environ.get(CUBLAS_ENV) not in CUBLAS_ACCEPTED:
        environ[CUBLAS_ENV] = CUBLAS_VALUE
    return {CUBLAS_ENV: environ[CUBLAS_ENV]}


def apply() -> dict:
    """Apply the recipe in this process (torch imported here); returns what torch reads back plus the env value."""
    import torch
    if os.environ.get(CUBLAS_ENV) not in CUBLAS_ACCEPTED:
        if torch.cuda.is_initialized():
            raise DetRecipeError(f"{CUBLAS_ENV} must be set (:4096:8) before CUDA initialises; it is {os.environ.get(CUBLAS_ENV)!r} and CUDA is initialised")
        export_env()
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from .settings import read_back
    out = read_back()
    out[CUBLAS_ENV] = os.environ.get(CUBLAS_ENV)
    out["seed"] = SEED
    return out
