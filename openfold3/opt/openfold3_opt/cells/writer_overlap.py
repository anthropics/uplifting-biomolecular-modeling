"""Cell `writer_overlap` — this kit's binding of `of3_writer_overlap` (exact class, in bytes): the engine's prediction writer
(`openfold3.core.runners.writer.OF3OutputWriter.on_predict_batch_end` -> `write_all_outputs`: per diffusion sample one mmCIF and two indented
JSON texts) renders and writes item k's files in one writer worker (a forked child process by default) while item k+1 is featurised and
forwarded, the device-to-host copies and the callback's bookkeeping staying where the engine runs them; every item drained and tallied
before the engine's summary — every output file identical, the wall time between two forwards shorter. Switch `OPENFOLD3_OPT_WRITER_OVERLAP=1`
(the `exact`, `fast` and `big` (resident) lines export it); knob `OPENFOLD3_OPT_WRITER_OVERLAP_ROUTE=process|thread|sync` (default
`process`; `sync` keeps the patch installed and writes every item where the engine writes it — for comparison). Installed by the package at
activation (openfold3_opt.stack.activate, modes.ACTIVATION_CELLS), not by a hook: the writer is engine plumbing every line shares. The
worker process moves `fastjson`'s counters; they are merged back into that cell's record per item (CHILD_CENSUS)."""
from . import of3_writer_overlap as _core
from . import fastjson as _fastjson

ENV = "OPENFOLD3_OPT_WRITER_OVERLAP"
ENV_ROUTE = "OPENFOLD3_OPT_WRITER_OVERLAP_ROUTE"
def _fastjson_snapshot() -> dict:
    st = _fastjson.STATE
    return {k: (dict(st[k]) if isinstance(st[k], dict) else st[k]) for k in ("served", "stock", "fallback", "arrays", "bytes", "seconds", "fallback_by")}


def _fastjson_merge(delta: dict) -> None:
    st = _fastjson.STATE
    for k, v in delta.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                st[k][kk] = st[k].get(kk, 0) + vv
        else:
            st[k] = st.get(k, 0) + v


_core.configure(ENV=ENV, ENV_ROUTE=ENV_ROUTE, PREFIX="[openfold3-opt/writer_overlap]", M_WRITER="openfold3.core.runners.writer", WRITER_CLASS="OF3OutputWriter",
                DIGESTS={"on_predict_batch_end": ("4c1eaf36b439d73d",)}, CHILD_CENSUS=[("fastjson", _fastjson_snapshot, _fastjson_merge)])

STATE = _core.STATE
VALUES = _core.VALUES
ROUTES = _core.ROUTES
requested, route, install, uninstall, census_line, serving, drain = (_core.requested, _core.route, _core.install, _core.uninstall, _core.census_line,
                                                                     _core.serving, _core.drain)
