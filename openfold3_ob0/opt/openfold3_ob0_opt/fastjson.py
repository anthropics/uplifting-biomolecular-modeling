"""The `fastjson` lever — this kit's binding of `cells/of3_fastjson.py` (exact class, in bytes): the engine's output writer
(`openfold3.core.runners.writer`, `OF3OutputWriter.write_confidence_scores`) renders each sample's aggregated confidence JSON and — under
the runner's default `full_confidence_output_format: json` — the full confidence JSON (`json.dumps(obj, indent=4, cls=NumpyEncoder)`; OpenFold3
0.5.0's `NumpyEncoder` rounds float16 / float32 arrays to 3 / 6 decimals before `tolist()`, the fast route calls that same conversion) through
a subclass of its `NumpyEncoder` whose `encode()` builds the same characters row by row at C level instead of CPython's pure-Python indent
encoder — every output file identical, the item's wall time after the model forward shorter; the `npz` full-confidence format never reaches
`json` (nothing to serve, the aggregated file still is). Switch `OPENFOLD3_OB0_OPT_FASTJSON=1` (the `exact`, `fast` and `big` lines export
it); knob `OPENFOLD3_OB0_OPT_FASTJSON_ROUTE=fast|stock` (default `fast`; `stock` keeps the class installed and answers every call with the
stock encoder — the ablation arm). Installed by the package's activation in every active process (stack.ACTIVATION_MODULES; registry kit
`package`), not by a hook: the writer is engine plumbing every line shares. Marker `[openfold3_ob0-opt/fastjson] installed`; exit census
`[openfold3_ob0-opt/fastjson] LEVER name=fastjson state=on|armed|refused|off route=fast|stock patched=<0|1> served=<n> fallback=<n> stock=<n>
arrays=<n> bytes=<n> s=<seconds>`."""
from .cells import of3_fastjson as _core

ENV = "OPENFOLD3_OB0_OPT_FASTJSON"
ENV_ROUTE = "OPENFOLD3_OB0_OPT_FASTJSON_ROUTE"
TARGET = "openfold3.core.runners.writer"
_core.configure(ENV="OPENFOLD3_OB0_OPT_FASTJSON", ENV_ROUTE="OPENFOLD3_OB0_OPT_FASTJSON_ROUTE", PREFIX="[openfold3_ob0-opt/fastjson]",
                M_WRITER=TARGET, ENCODER_ATTR="NumpyEncoder")

STATE = _core.STATE
VALUES = _core.VALUES
ROUTES = _core.ROUTES
requested, route, install, uninstall, census_line, serving = _core.requested, _core.route, _core.install, _core.uninstall, _core.census_line, _core.serving
