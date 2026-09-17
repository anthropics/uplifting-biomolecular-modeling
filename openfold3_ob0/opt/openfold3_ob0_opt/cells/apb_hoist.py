"""The `apb_hoist` lever (exact class; the `exact` and `fast` lines): `AttentionPairBias._prep_bias` rebuilds the key-mask bias
`inf * (mask - 1)` on every call from a mask that is constant for the stack call (48 pairformer blocks per recycle pass share one); the term is
memoised on the mask's storage / version / shape and the pair-bias term computed exactly as stock — bitwise the line without it. On the fast
line the `apb_trunk` cell serves the pairformer / confidence instances itself and passes the rest here. The implementation is the tree's
(`opt_core.of3_trunk.apb_hoist`); this module binds openfold3_ob0_opt's switch and prefix.

Switch: OPENFOLD3_OB0_OPT_APB_HOIST=1. Exit line `[openfold3_ob0-opt/apb_hoist] LEVER name=apb_hoist state=on hit=… fill=… fallback=…`."""
from opt_core.of3_trunk import apb_hoist as _core

ENV = "OPENFOLD3_OB0_OPT_APB_HOIST"
_core.configure(PREFIX="[openfold3_ob0-opt/apb_hoist]", ENV=ENV, M_APB="openfold3.core.model.layers.attention_pair_bias")

STATE = _core.STATE
VALUES = _core.VALUES
requested, install, census_line, serving = _core.requested, _core.install, _core.census_line, _core.serving
