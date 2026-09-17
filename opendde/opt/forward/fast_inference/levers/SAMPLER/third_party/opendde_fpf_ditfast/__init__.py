"""opendde_fpf_ditfast — fused diffusion-sampler levers for OpenDDE 1.1.1: cond_dedupe / dit_fused (+dit_lowp) / atom_fused.

  cond_dedupe   DiffusionConditioning once per step on the sample-invariant noise level, broadcast over the samples              install_cond_dedupe(model)
  dit_fused     the 24-block token DiffusionTransformer as one fused forward (merged GEMMs + Triton row kernels + hoisted bias;     install_dit_fast(model)
                attention = odde_apb_bind.dit_packed, lever dit_attn_apb, REQUIRED)
  dit_lowp      precision word of dit_fused's a-path GEMMs / attention operands: ODDE_DIT_LOWP = off | bf16 | fp16                  (read by install_dit_fast)
  atom_fused    both 3-block atom transformers as fused stacks (attention = odde_apb_bind.atom_views, lever atom_attn_apb, REQUIRED)  install_atom_fast(model)

Engine binding (marked `odde:` in each file): module paths opendde.model.*; switch words ODDE_COND_DEDUPE, ODDE_DIT_FUSED, ODDE_DIT_LOWP, ODDE_ATOM_FUSED
(the attention words ODDE_DIT_ATTN / ODDE_ATOM_ATTN are served by levers/SAMPLER/odde_apb_bind through the core provider); the DiT hoist object is
levers/DITFAST odde_addon._ACTIVE["dit_hoist"] (`cached(name, producer)` slot protocol); OpenDDE's `extra_attn_bias` argument of the token stack (the
structural pair attention bias) is folded into the hoisted per-block bias exactly as transformer.py AttentionPairBias.standard_multihead_attention does;
the deeper instance overrides the hoist puts on the atom blocks are stripped (this unit owns those blocks); per-lever COUNTS dicts feed the kit's
ran-or-refuse census (opendde_opt/ran.py). The row kernels are the shared core's (`opt_core.kernels.apb.ditfast.kernels` / `.atom_kernels`); no kernel
file lives here.

odde_sampler.install calls the install functions AFTER the DiT hoist and the attention levers are installed. Each install prints one marker line
([opendde_fpf_ditfast] DITFAST:on(...) / ATOMFAST:on(...) / CONDDEDUPE:on(...)), registers an EXIT census line, and raises LeverRefused("<lever>: <reason>")
BY NAME when it cannot engage (no fallback). First-party code of this kit; no third-party source. Runtime needs: torch, triton, opt_core, odde_apb_bind.
"""
from ._plumbing import LeverRefused, exits as report  # noqa: F401
from .cond_dedupe import install as install_cond_dedupe  # noqa: F401
from .dit_fast import install as install_dit_fast, FastTokenStack  # noqa: F401
from .atom_fast import install as install_atom_fast, FastAtomStack  # noqa: F401

__version__ = "0.9.0+odde2"    # the package's version word (printed on the marker lines)
