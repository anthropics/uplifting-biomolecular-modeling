"""The `exactln` lever (exact class; the `exact` line) and the `ln_provider` lever (tolerance class; the `fast` and `big` lines): openfold3's
`LayerNorm` primitive bound to the shared core's LayerNorm PROVIDER (`opt_core.kernels.ln`) by the line's TIER WORD — `exact` on the exact line
(an exact-class row only where the provider's table marks it bitwise with ATen for the running card and stack and its first call proves
`torch.equal` in-process, else ATen by name), `fast` / `big` on those lines (the provider's fastest row within each cell's tolerance,
capture-aware). The implementation is `openfold3_ob0_opt.of3_exactln` (kit-namespaced);
this module binds openfold3_ob0_opt's switches, knobs and prefixes and re-exports its record for the stack's probe and the exit census.

Switches: OPENFOLD3_OB0_OPT_EXACTLN=1 (knob OPENFOLD3_OB0_OPT_EXACTLN_WORD=<tier word | provider row>, default `exact`); OPENFOLD3_OB0_OPT_LN_PROVIDER=1 with
OPENFOLD3_OB0_OPT_LN_TIER=fast|big (the line's word; knob OPENFOLD3_OB0_OPT_LN_WORD=<tier word | provider row>). Exit line `[openfold3_ob0-opt/exactln] LEVER
name=exactln …` / `[openfold3_ob0-opt/ln_provider] LEVER name=ln_provider …` (word, stack, served / kept counts by name, cells, proven / witnessed)."""
from openfold3_ob0_opt import of3_exactln as _core

ENV = "OPENFOLD3_OB0_OPT_EXACTLN"
ENV_WORD = "OPENFOLD3_OB0_OPT_EXACTLN_WORD"
ENV_PROVIDER = "OPENFOLD3_OB0_OPT_LN_PROVIDER"
ENV_TIER = "OPENFOLD3_OB0_OPT_LN_TIER"
ENV_PROVIDER_WORD = "OPENFOLD3_OB0_OPT_LN_WORD"
PREFIX = "[openfold3_ob0-opt/exactln]"
PREFIX_PROVIDER = "[openfold3_ob0-opt/ln_provider]"
_core.configure(ENV=ENV, ENV_WORD=ENV_WORD, ENV_PROVIDER=ENV_PROVIDER, ENV_TIER=ENV_TIER, ENV_PROVIDER_WORD=ENV_PROVIDER_WORD, PREFIX=PREFIX,
                PREFIX_PROVIDER=PREFIX_PROVIDER, M_NORM="openfold3.core.model.primitives.normalization",
                M_MODEL="openfold3.projects.of3_all_atom.model", MODEL_CLASS="OpenFold3", CASTCACHE_MODULE="openfold3_ob0_opt.cells.castcache")


STATE = _core.STATE                                           # ONE record for the binding (either lever); STATE["lever"] names the one switched on
VALUES, TIER_VALUES = _core.VALUES, _core.TIER_VALUES
install, serving, word, word_sourced, lever, census_line = _core.install, _core.serving, _core.word, _core.word_sourced, _core.lever, _core.census_line


def requested(environ=None) -> bool:
    return _core.requested(environ, which="exactln")
