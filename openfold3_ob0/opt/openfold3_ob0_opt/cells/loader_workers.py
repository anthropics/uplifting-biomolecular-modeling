"""`loader_workers` — this kit's binding of the predict-loader worker cap (exact class; `openfold3_ob0_opt.of3_loader`): one query x
one seed does not need the engine's ten forked featurisation workers (`data_module_args.num_workers` default 10: each costs start-up time and
host memory, and nine of them idle); the predict DataLoader gets max(1, min(items, configured)) workers — the same worker featurises the same item
with the same seed: byte-identical outputs. Switches `OPENFOLD3_OB0_OPT_LOADER_WORKERS=1` (the exact, fast and big/resident lines export it)
and `OPENFOLD3_OB0_OPT_LOADER_WORKERS_CAP=items|none` (`none` = the engine's configured count: the lever off by name)."""
from .. import of3_loader as _core

ENV = "OPENFOLD3_OB0_OPT_LOADER_WORKERS"
ENV_CAP = "OPENFOLD3_OB0_OPT_LOADER_WORKERS_CAP"
_core.configure(ENV=ENV, ENV_CAP=ENV_CAP, PREFIX="[openfold3_ob0-opt/loader_workers]", M_DATA="openfold3.core.data.framework.data_module")

STATE = _core.STATE
VALUES = _core.VALUES
requested, install, census_line, serving, plan = _core.requested, _core.install, _core.census_line, _core.serving, _core.plan
