"""protenix_fpf_apb — fused attention-with-pair-bias kernels for the Protenix-v2 diffusion sampler (levers ``dit_attn`` and ``atom_attn``).

  apb_views(q, k, v, bias, g=None, ...)   flash attention + an additive pair bias shared by the S diffusion samples (DiT token site: 16 heads x 48)
  dit_apb(qkvg, bias, S, N, ...)          the same kernel on one fused q|k|v|g GEMM output (the fused-projection dit_fast layer signature)
  atom_apb(q, k, v, bias, g=None, ...)    the atom encoder/decoder's 32-query x 128-key local-window attention, one launch per call
  pf_bias(z, ln_w, ln_b, W, eps)          fused LayerNorm(z) + Linear(c_z -> heads) pair-bias producer for the Pairformer AttentionPairBias (lever ``pf_attn``)
  install_dit_attn(model) / install_atom_attn(model) / install_pf_attn(model) / report()   the levers' install functions (``install.py``), called from the kit's
                                          ``runner.inference.InferenceRunner.__init__`` seam beside sampler_fuse

Provenance: written for this kit by the fpf_apb authors (2026-09); no third-party source. Requires triton >= 3.3 (host-side
TMA tensor descriptors; without them the bias tile is read with plain loads — a named cell variant, not a fallback) and an NVIDIA GPU with
tensor cores (cells measured on sm_90; other cards engage the sm_90 cell and are named).
"""
from .apb_triton import apb_views, dit_apb, apb_config, LAST_LAUNCH  # noqa: F401
from .atom_triton import atom_apb  # noqa: F401
from .pf_triton import pf_bias  # noqa: F401
from .install import install_dit_attn, install_atom_attn, install_pf_attn, report, CELLS, STATE  # noqa: F401

__version__ = "0.2.2"
