"""The `triatt_exact` lever (exact class; the `exact` line): the trunk's triangle attention — the cuEquivariance library call stock's
`_cueq_triangle_attn` makes for every pair stack above the library's 100-token threshold (pairformer, MSA-module, confidence head; the
template stack's head dim 16 stays the library's by the cell table) — served through the core's triangle-attention provider on word `exact`
(`opt_core.kernels.triattn`, cell table `TRIATTN_CELLS.json`): at this pin the cell names the stock library call on every card and it
serves, by name; an exact-class row a cell names would serve a call class only after its first eager call proved `torch.equal` against the
library on the call's own operands.  Bitwise with the line without it.  The binding is `openfold3_ob0_opt.of3_triattn`;
this module binds openfold3_ob0_opt's switch, prefix and engine module and re-exports the record for the stack's probe and the exit census.

Switch: OPENFOLD3_OB0_OPT_TRIATT_EXACT=1.  Evidence `[openfold3_ob0-opt/triattn] triatt_exact: installed …` at install (of3_triattn's prefix) and
the exit line `[openfold3_ob0-opt/triatt_exact] LEVER name=triatt_exact state=on word=exact rows=… fallback=… cells=… proven=… bits_differ=…`
(state=refused reason=no_cueq:triangle_attention | no_cuda_device | of3t_triatt_routes when it cannot bind)."""
import atexit
import os
import sys

from .. import of3_triattn as _t

ENV = "OPENFOLD3_OB0_OPT_TRIATT_EXACT"
PREFIX = "[openfold3_ob0-opt/triatt_exact]"
M_ATT = "openfold3.core.model.primitives.attention"
VALUES = ("1",)
STATE = _t.EXACT


def requested(environ=None) -> bool:
    v = ((os.environ if environ is None else environ).get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def serving() -> bool:
    return bool(STATE.get("installed")) and STATE.get("state") == "on"


def census_line() -> str:
    r = STATE.get("router")
    body = r.fields() if r is not None else f"reason={STATE.get('reason') or 'not_installed'}"
    return f"{PREFIX} LEVER name=triatt_exact state={STATE.get('state', 'off')} {body}"


def install(environ=None) -> dict:
    if STATE.get("installed") or not requested(environ):
        return STATE
    env = os.environ if environ is None else environ
    if (env.get("OF3T_TRIATT") or "").strip():                       # the trunk-kernels add-on routes TriangleAttention's kernel flags on that switch (no kit line
        STATE.update(installed=True, state="refused", reason="of3t_triatt_routes")   #  sets it beside this lever): one owner per statement, by name
        sys.stderr.write(f"{PREFIX} REFUSED: OF3T_TRIATT={env.get('OF3T_TRIATT')} routes TriangleAttention (trunk_kernels add-on) — nothing rebound\n")
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
        return STATE
    _t.install_exact(M_ATT, word="exact", tag="triatt_exact")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
