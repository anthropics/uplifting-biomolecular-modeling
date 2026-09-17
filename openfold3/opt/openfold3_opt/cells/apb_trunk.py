"""The `apb_trunk` lever (fast class): the pairformer single track's attention with pair bias (`AttentionPairBias.forward` of the 48 trunk
blocks per recycle pass and of the confidence head's pairformer per sample; not the diffusion transformer's AdaLN instances) — the pair bias
produced head-major by one GEMM straight from LayerNorm(z) and the attention as one launch of the core's pair-bias kernel
(`opt_core.attn.apb_core`); refusals by name run the stock forward, counted. The implementation is the tree's (`opt_core.of3_sampler.apb_trunk`);
this module binds openfold3_opt's switches and prefix and re-exports its record. openfold3 0.4.1 never asks `use_high_precision_attention` in
the trunk: a call that does keeps the stock forward (HIGH_PRECISION "honour").

Switches: OPENFOLD3_OPT_APB_TRUNK=1 (the fast line exports it); OPENFOLD3_OPT_APB_TRUNK_MIN_TOKENS (token gate, default 0); attribution knobs
OPENFOLD3_OPT_APB_TRUNK_HIGH_PRECISION=override|honour (default honour) and OPENFOLD3_OPT_APB_TRUNK_SCOPE=trunk+confidence|trunk (default
trunk+confidence; `trunk` runs the confidence head's calls on the stock forward, counted `scope:confidence`);
OPENFOLD3_OPT_APB_TRUNK_PRODUCER=ln_proj|mm (the pair-bias producer: the tree's fused LayerNorm+projection kernel, the default, or the module's
LayerNorm + one head-major GEMM). Refused beside the trunk-kernels add-on's `OF3T_APB` route. Exit line `[openfold3-opt/apb_trunk] LEVER name=apb_trunk …`."""
from opt_core.of3_sampler import apb_trunk as _core

from . import apb_word                                          # the family's provider word (bound at install, before the core resolves)

ENV = "OPENFOLD3_OPT_APB_TRUNK"
ENV_MIN = "OPENFOLD3_OPT_APB_TRUNK_MIN_TOKENS"
ENV_HIGH_PRECISION = "OPENFOLD3_OPT_APB_TRUNK_HIGH_PRECISION"
ENV_SCOPE = "OPENFOLD3_OPT_APB_TRUNK_SCOPE"
ENV_PRODUCER = "OPENFOLD3_OPT_APB_TRUNK_PRODUCER"
CONFLICT_ENV = "OF3T_APB"
_core.configure(PREFIX="[openfold3-opt/apb_trunk]", ENV=ENV, ENV_MIN=ENV_MIN, ENV_HIGH_PRECISION=ENV_HIGH_PRECISION, ENV_SCOPE=ENV_SCOPE,
                ENV_PRODUCER=ENV_PRODUCER, PRODUCER_DEFAULT="ln_proj", CONFLICT_ENV=CONFLICT_ENV, HIGH_PRECISION="honour",
                M_APB="openfold3.core.model.layers.attention_pair_bias",
                M_CONF="openfold3.core.model.heads.prediction_heads")          # PairformerEmbedding: the confidence head's pairformer (scope word, served_confidence)

STATE, VALUES, CORE_MODULE = _core.STATE, _core.VALUES, _core.CORE_MODULE
requested, census_line, use_producer = _core.requested, _core.census_line, _core.use_producer


def install(environ=None):
    """Bind the family's provider word, then the tree's install (idempotent)."""
    if requested(environ):
        apb_word.bind(environ)
    return _core.install(environ)
plan, served_forward, headmajor_bias, lnproj_bias, PRODUCERS = _core.plan, _core.served_forward, _core.headmajor_bias, _core.lnproj_bias, _core.PRODUCERS
