"""The `castcache` lever (exact class; the `exact` and `fast` lines): openfold3's `Linear` / `LayerNorm` primitives cast their fp32 weight and
bias to bf16 on every bf16 call (two copy kernels per call, a large share of the trunk's launches at small token counts); the bf16 copies are memoised per
module (keyed on the fp32 tensors' storage + version) and handed to the same F.linear / F.layer_norm call — bitwise the line without it. The
implementation is the tree's (`opt_core.of3_trunk.castcache`); this module binds openfold3_ob0_opt's switch and prefix and re-exports its record
for the stack's probe and the exit census.

Switch: OPENFOLD3_OB0_OPT_CASTCACHE=1. Exit line `[openfold3_ob0-opt/castcache] LEVER name=castcache state=on modules=… mib=… refresh=… served=…`."""
from opt_core.of3_trunk import castcache as _core

ENV = "OPENFOLD3_OB0_OPT_CASTCACHE"
_core.configure(PREFIX="[openfold3_ob0-opt/castcache]", ENV=ENV, M_LINEAR="openfold3.core.model.primitives.linear",
                M_NORM="openfold3.core.model.primitives.normalization")

STATE = _core.STATE
VALUES = _core.VALUES
requested, install, census_line, serving = _core.requested, _core.install, _core.census_line, _core.serving
