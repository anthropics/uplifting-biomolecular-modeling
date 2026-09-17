"""enformer_opt — inference optimizations for Enformer (upstream ``enformer-pytorch`` 0.8.12), one mode: ``exact``.

The kit is a drop-in: with it enabled, every ``enformer_pytorch.Enformer`` instance in the process (however it was built — ``from_pretrained``,
``Enformer.from_hparams``, your own code) runs its forward through the kit's levers and returns the same bytes as the stock forward, faster.
Nothing about the model's interface changes: same call forms, same outputs, same dtypes.

    ENFORMER_OPT=exact python your_script.py        # no code change (the package's .pth arms it at interpreter start)
    import enformer_opt; enformer_opt.enable()       # or explicitly, before the first forward

Levers (all four, always; kits/v0_2 under opt/forward/enformer_kit): ``poscache`` (positional embeddings computed once per length and reused),
``fused`` (the trunk's elementwise passes — conv bias adds, BatchNorm+GELU, residual adds, the attention-pool tail — as five exact kernels),
``xattn`` (one exact kernel chain for the relative-position attention), ``graph`` (CUDA-graph replay of the trunk, one graph per batch size). The first forward of each model pays a one-time, stated cost
(the extension load and one graph capture for that batch size; a later batch size captures once at its first call); every line the package
prints starts with ``[enformer-opt]`` on stderr. ``enable()`` refuses by name (``NOT ACTIVE``, nothing applied, stock untouched) when no CUDA
device is visible, when no build of the kit's kernels runs on the device (below sm_80), when the kit's files are missing, or when the installed
``enformer-pytorch`` is not the pinned version; a device or torch build that merely differs from the tested ones (H100 / A100, torch
2.13.0+cu130) is named on the line and served. ``python -m enformer_opt check`` (``run.sh check``) is the dry run: the same resolution, nothing
applied.
"""
from ._runtime import (PREFIX, ActivationError, apply, check, disable, enable, status)   # noqa: F401

__version__ = "1.0.0"
__all__ = ["enable", "disable", "apply", "check", "status", "ActivationError", "PREFIX", "__version__"]
