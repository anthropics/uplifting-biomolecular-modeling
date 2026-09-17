"""`tuner_guard` — this kit's binding of `opt_core.of3_trunk.tuner_guard` (exact class): the engine's per-stack chunk-size tuner compares
the argument record of its last tuning with the next chunked call's and RAISES when the two differ in structure (the confidence stack's batched
call at or below its per-sample token cutoff vs its per-sample calls above it: tensor ranks differ) — one `fast` process predicting a query of
<= 750 tokens and then a larger one failed every larger query in the confidence head. The lever answers such a comparison "changed" so the
tuner re-tunes, counted on the census. Switch `OPENFOLD3_OPT_TUNER_GUARD=1` (the `fast` line exports it; the stock configuration never tunes the
batched call, so `exact` does not need it)."""
from opt_core.of3_trunk import tuner_guard as _core

ENV = "OPENFOLD3_OPT_TUNER_GUARD"
_core.configure(ENV="OPENFOLD3_OPT_TUNER_GUARD", PREFIX="[openfold3-opt/tuner_guard]", M_CHUNK="openfold3.core.utils.chunk_utils")

STATE = _core.STATE
VALUES = _core.VALUES
requested, install, census_line, serving = _core.requested, _core.install, _core.census_line, _core.serving
