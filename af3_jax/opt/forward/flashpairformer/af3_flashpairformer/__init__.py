"""af3_flashpairformer - fused Pallas (Triton) GPU kernels for the AlphaFold 3 pairformer: TriangleMultiplication and
GridSelfAttention (triangle attention), as a drop-in for the official google-deepmind/alphafold3 Haiku modules
(v3.0.x releases and main) and the sokrypton/alphafold3 fork (OF3-p2 weights). Inference only. Zero weight changes.

    import af3_flashpairformer            # before the first forward; env AF3_FLASHPAIRFORMER=off|trimul|triattn|both (default both)

or explicitly: af3_flashpairformer.install("both") / .uninstall() / .status().
env AF3_DIFFUSION_HOIST=1 (or af3_flashpairformer.install_hoist()) additionally hoists the step-invariant diffusion conditioning out of the
200-step sampling loop (module diffusion_hoist.py; not a Pallas kernel; numerics: not guaranteed bitwise).
Outside the served shapes (bf16 pair activations, square N x N with N a multiple of the core's kernel tile
(opt_core.kernels.fpf_pallas_serve.required_multiple), C % 16 == 0, head dim in {16,32,64})
the patched classes fall back to the stock code path call by call.
NUMERICS: not bitwise identical to stock (bf16 re-association; tolerance class).
"""
import os as _os
from .patch import install, uninstall, status, MODES, served_report  # noqa: F401

__version__ = "1.2.0"
_mode = _os.environ.get("AF3_FLASHPAIRFORMER", "both").strip().lower()
if _mode not in ("off", "0", "none", "false", ""):
    install(_mode)


def install_hoist():
    from . import diffusion_hoist
    return diffusion_hoist.install()


if _os.environ.get("AF3_DIFFUSION_HOIST", "0").strip().lower() in ("1", "true", "on", "yes"):
    install_hoist()
