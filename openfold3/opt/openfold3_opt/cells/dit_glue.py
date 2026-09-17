"""The `dit_glue` lever (fast class): the token diffusion transformer block (`DiffusionTransformerBlock.forward`, 24 blocks x 200 steps) as ONE
fused schedule — AdaLN row kernel, one q|k|v|gate GEMM, pair-bias flash attention on the GEMM's column slices in place, one output GEMM,
gate + residual in one pass, the conditioned SwiGLU transition as [a|b] GEMM -> swiglu -> GEMM -> gate/mask/residual kernel; refusals by name
run the stock block, counted. The implementation is the tree's (`opt_core.of3_sampler.dit_glue` + `dit_rows`); this module binds openfold3_opt's
switches and prefix and re-exports its record. A call that asks `use_high_precision_attention` keeps the stock block (openfold3 0.4.1 never asks it in the rollout).

Switches: OPENFOLD3_OPT_DIT_GLUE=1; OPENFOLD3_OPT_DIT_GLUE_MIN_TOKENS (size gate, default 0); OPENFOLD3_OPT_DIT_GLUE_ROWS=all|block (all, the default: the conditioned
transition and AdaLN outside the block schedule — the atom transformer's, fp32 — served class-wide by the same row schedules; block: the block
schedule only); OPENFOLD3_OPT_DIT_GLUE_CORE=auto|dtk (the attention core: the
tree's `opt_core.attn.apb_core` entry when carried, else `dtk_kernels.flash_bias_attn`). Exit line `[openfold3-opt/dit_glue] LEVER name=dit_glue …`."""
from opt_core.of3_sampler import dit_glue as _core

from . import apb_word                                          # the family's provider word (bound at install, before the core resolves)

ENV = "OPENFOLD3_OPT_DIT_GLUE"
ENV_MIN = "OPENFOLD3_OPT_DIT_GLUE_MIN_TOKENS"
ENV_CORE = "OPENFOLD3_OPT_DIT_GLUE_CORE"
ENV_ROWS = "OPENFOLD3_OPT_DIT_GLUE_ROWS"
_core.configure(PREFIX="[openfold3-opt/dit_glue]", ENV=ENV, ENV_MIN=ENV_MIN, ENV_CORE=ENV_CORE, ENV_ROWS=ENV_ROWS, HIGH_PRECISION="honour",
                M_DIT="openfold3.core.model.layers.diffusion_transformer")

STATE = _core.STATE
VALUES, CORE_VALUES, ROWS_VALUES, KERNEL, CORE_MODULE = _core.VALUES, _core.CORE_VALUES, _core.ROWS_VALUES, _core.KERNEL, _core.CORE_MODULE
requested, census_line = _core.requested, _core.census_line


def install(environ=None):
    """Bind the family's provider word, then the tree's install (idempotent)."""
    if requested(environ):
        apb_word.bind(environ)
    return _core.install(environ)
