"""Cell `fastjson` — this kit's binding of `of3_fastjson` (exact class, in bytes): the engine's output writer
(`openfold3.core.runners.writer`, `OF3OutputWriter.write_confidence_scores`) renders each sample's aggregated and full confidence JSON
(`json.dumps(obj, indent=4, cls=NumpyEncoder)`) through a subclass of its `NumpyEncoder` whose `encode()` builds the same characters row by
row at C level instead of CPython's pure-Python indent encoder — every output file identical, the item's wall time after the model forward
shorter. Switch `OPENFOLD3_OPT_FASTJSON=1` (the `exact`, `fast` and `big` lines export it); knob `OPENFOLD3_OPT_FASTJSON_ROUTE=fast|stock`
(default `fast`; `stock` keeps the class installed and answers every call with the stock encoder — for comparison). Installed by the package at
activation on every line (openfold3_opt.stack.activate), not by a hook: the writer is engine plumbing every line shares."""
from . import of3_fastjson as _core

ENV = "OPENFOLD3_OPT_FASTJSON"
ENV_ROUTE = "OPENFOLD3_OPT_FASTJSON_ROUTE"
_core.configure(ENV="OPENFOLD3_OPT_FASTJSON", ENV_ROUTE="OPENFOLD3_OPT_FASTJSON_ROUTE", PREFIX="[openfold3-opt/fastjson]",
                M_WRITER="openfold3.core.runners.writer", ENCODER_ATTR="NumpyEncoder")

STATE = _core.STATE
VALUES = _core.VALUES
ROUTES = _core.ROUTES
requested, route, install, uninstall, census_line, serving = _core.requested, _core.route, _core.install, _core.uninstall, _core.census_line, _core.serving
