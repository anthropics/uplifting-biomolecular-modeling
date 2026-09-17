"""protenix_fpf_ditfast — fused diffusion-sampler levers for the Protenix-v2 kit (2026-09):

  cond_dedupe   DiffusionConditioning once per step on the sample-invariant noise level, broadcast over the 5 samples      install_cond_dedupe(model)
  dit_fused     the 24-block token DiffusionTransformer as one fused forward (merged GEMMs + Triton row kernels + hoisted bias;  install_dit_fast(model)
                attention = fpf_apb.dit_apb, lever dit_attn, REQUIRED)
  dit_lowp      precision word of dit_fused's a-path GEMMs / attention operands: PTX_DIT_LOWP = off | bf16 | fp16                (read by install_dit_fast)
  atom_fused    both 3-block atom transformers as fused stacks (attention = fpf_apb.atom_apb, lever atom_attn, REQUIRED)         install_atom_fast(model)

The kit calls the install functions from its runner.inference.InferenceRunner.__init__ seam AFTER fpf_clisampler (graphed sampler + DiT hoist),
sampler_fuse (DIT_FUSE) and fpf_apb installed theirs. Each install prints one marker line ([protenix_fpf_ditfast] DITFAST:on(...) / ATOMFAST:on(...) /
CONDDEDUPE:on(...)), registers an EXIT census line, and raises LeverRefused("<lever>: <reason>") BY NAME when it cannot engage (no fallback).
Provenance: first-party code of this kit; no third-party source. Runtime needs: torch, triton (the image's), protenix_fpf_apb.
"""
from ._plumbing import LeverRefused, exits as report  # noqa: F401
from .cond_dedupe import install as install_cond_dedupe  # noqa: F401
from .dit_fast import install as install_dit_fast, FastTokenStack  # noqa: F401
from .atom_fast import install as install_atom_fast, FastAtomStack  # noqa: F401

__version__ = "0.9.0"
