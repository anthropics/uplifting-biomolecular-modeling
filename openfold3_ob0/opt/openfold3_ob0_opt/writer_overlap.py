"""The `writer_overlap` lever — this kit's binding of `cells/of3_writer_overlap.py` (exact class, in bytes): the engine's prediction writer
(`openfold3.core.runners.writer.OF3OutputWriter.on_predict_batch_end` -> `write_all_outputs`: per diffusion sample one mmCIF through biotite and
two indented JSON texts — or the npz —, on the main thread between two forwards) re-stated so the callback's bookkeeping and the device-to-host
copies the writer makes first stay where the engine runs them and the rendering + writing — the writer instance's own `write_all_outputs` on those
host copies, `fastjson` applying unchanged — runs in one writer worker (a forked child process by default) while item k+1 is featurised and
forwarded; at most two items in flight, every item drained and tallied into the callback's own counters before the engine's summary and at exit;
`write_features` / `write_latent_outputs` / a distributed predict written inline by name — every output file identical, the wall time between two
forwards shorter, the model forward untouched. Switch `OPENFOLD3_OB0_OPT_WRITER_OVERLAP=1`
(the `exact`, `fast` and `big` (resident) lines export it); knob `OPENFOLD3_OB0_OPT_WRITER_OVERLAP_ROUTE=process|thread|sync` (default `process`;
`sync` = installed, every item written where the engine writes it: the ablation arm). Installed by the package's activation in every active process
(stack.ACTIVATION_MODULES; registry kit `package`). Marker `[openfold3_ob0-opt/writer_overlap] installed`; exit census `[openfold3_ob0-opt/writer_overlap]
LEVER name=writer_overlap state=on|armed|refused|off route=… items=<n> overlapped=<n> inline=<reason:n|none> host_copy_s= wait_s= drain_s= write_s= write_exc=
worker=<pid|thread|->`."""
from .cells import of3_writer_overlap as _core
from . import fastjson as _fastjson

ENV = "OPENFOLD3_OB0_OPT_WRITER_OVERLAP"
ENV_ROUTE = "OPENFOLD3_OB0_OPT_WRITER_OVERLAP_ROUTE"
TARGET = "openfold3.core.runners.writer"


def _fastjson_snapshot() -> dict:
    st = _fastjson.STATE
    return {k: (dict(st[k]) if isinstance(st[k], dict) else st[k]) for k in ("served", "stock", "fallback", "arrays", "bytes", "seconds", "fallback_by") if k in st}


def _fastjson_merge(delta: dict) -> None:
    st = _fastjson.STATE
    for k, v in delta.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                st[k][kk] = st[k].get(kk, 0) + vv
        else:
            st[k] = st.get(k, 0) + v


_core.configure(ENV=ENV, ENV_ROUTE=ENV_ROUTE, PREFIX="[openfold3_ob0-opt/writer_overlap]", M_WRITER=TARGET, WRITER_CLASS="OF3OutputWriter",
                DIGESTS={"on_predict_batch_end": ("4c1eaf36b439d73d",)},      # OpenFold3 0.5.0 core/runners/writer.py OF3OutputWriter.on_predict_batch_end
                CHILD_CENSUS=[("fastjson", _fastjson_snapshot, _fastjson_merge)])

STATE = _core.STATE
VALUES = _core.VALUES
ROUTES = _core.ROUTES
requested, route, install, uninstall, census_line, serving, drain = (_core.requested, _core.route, _core.install, _core.uninstall, _core.census_line,
                                                                     _core.serving, _core.drain)
