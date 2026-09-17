"""The `ln_provider` lever (tolerance class; the `fast` and `big` lines): openfold3's `LayerNorm` primitive bound to the shared core's LayerNorm
provider by the line's tier word (`fast` | `big`). ONE binding with the exact line's `exactln` lever (openfold3_opt.of3_exactln, configured by
cells/exactln.py); this module is the lever's own name for the hook, the stack's probe and the exit census.
Switches: OPENFOLD3_OPT_LN_PROVIDER=1, OPENFOLD3_OPT_LN_TIER=fast|big (exported by the line), knob OPENFOLD3_OPT_LN_WORD=<tier word | provider row>."""
from . import exactln as _binding
from openfold3_opt import of3_exactln as _core

ENV, ENV_TIER, ENV_WORD, PREFIX = _binding.ENV_PROVIDER, _binding.ENV_TIER, _binding.ENV_PROVIDER_WORD, _binding.PREFIX_PROVIDER
STATE = _core.STATE                                           # the binding's one record (cells/exactln.py); STATE["lever"] == "ln_provider" when this lever is the one on
VALUES, TIER_VALUES = _core.VALUES, _core.TIER_VALUES
install, serving, word, word_sourced, lever, census_line = _core.install, _core.serving, _core.word, _core.word_sourced, _core.lever, _core.census_line


def requested(environ=None) -> bool:
    return _core.requested(environ, which="ln_provider")
