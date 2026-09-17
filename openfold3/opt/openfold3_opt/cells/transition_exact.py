"""The `transition_exact` lever (exact class; the `exact` line): the trunk's SwiGLU transitions (`layers.transition.SwiGLUTransition`: the 48
pairformer pair transitions, the MSA module's 4 and the confidence head's 4 pair-stack transitions — c 128, hidden 512; the template pair
stack's c 64) bound to the shared core's ONE transition provider (`opt_core.kernels.transition`; cell table TRANSITION_CELLS.json) by the tier
word `exact` under this engine's statement form (`form='liger'`: OpenFold3 0.4.1's `LigerSiLUMulFunction(linear_a(x), linear_b(x))` ->
linear_out — silu(fp32 a) * fp32 b rounded once), fed the module's own LayerNorm output: where the provider's table names a kernel row the
exact-tier winner for the cell on this process's stack — bitwise against the statement there, from the stack's token floor up, and not slower
than the statement (row `v1:liger`: everything after the LayerNorm in one Triton kernel) —
the call launches it; where the table names the stock statement the winner, carries no vouch for this stack or row count, or has no cell, the
call runs the module's own statement, counted by the provider's word.  No kernel, tile table or token floor in the kit.  The n*c hidden never
reaches HBM on served calls.  fp32 calls outside bf16 autocast (the diffusion module's transitions) and module variants run the module's own
statement, counted.  Bitwise the line without it.  The binding is `openfold3_opt.of3_transition`; this module binds the kit's switch, prefix
and engine module and re-exports its state for the stack's probe and the exit census.

Switch: OPENFOLD3_OPT_TRANSITION_EXACT=1.  Evidence `[openfold3-opt/transition_exact] installed: ...` at install and the exit line
`[openfold3-opt/transition_exact] LEVER name=transition_exact state=on word=exact form=liger core=<opt_core version> vouch=<stack|none>
served=<n> fallback=<n> [fallback:<reason>=<n> ...] rows=<row>:<n>,...|none cells=<cell key>:<n>,...|none widths=c<C>:<n>,...|none
first=<shape:dtype|none>` (state=refused reason=pair_transition_routes | no-provider:... | no-SwiGLUTransition._transition when it cannot bind)."""
import atexit
import importlib
import os
import sys

from .. import of3_transition as _t

ENV = "OPENFOLD3_OPT_TRANSITION_EXACT"
PREFIX = _t.PREFIX
M_TRANSITION = "openfold3.core.model.layers.transition"
VALUES = ("1",)
STATE = _t.STATE


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
    body = _t.census() if STATE.get("installed") and STATE.get("state") == "on" else f" reason={STATE.get('reason') or 'not_installed'}"
    return f"{PREFIX} LEVER name=transition_exact state={STATE.get('state', 'off')}{body}"


def install(environ=None) -> dict:
    if STATE.get("installed") or STATE.get("state") == "refused" or not requested(environ):
        return STATE
    env = os.environ if environ is None else environ
    pair = [w for w in (env.get("OPENFOLD3_OPT_PAIR") or "").split(":") if w]
    if "pair_transition" in pair:                                         # the fast line's pair cell routes SwiGLUTransition.forward itself (cells/pairfused.py):
        STATE.update(installed=True, state="refused", reason="pair_transition_routes")   #  one owner per statement, by name
        sys.stderr.write(f"{PREFIX} REFUSED: OPENFOLD3_OPT_PAIR names pair_transition (the fast line's fused transition cell owns SwiGLUTransition) — nothing rebound\n")
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
        return STATE
    mod = sys.modules.get(M_TRANSITION) or importlib.import_module(M_TRANSITION)
    _t.install(mod)
    if STATE.get("state") == "refused":
        STATE["installed"] = True                                         # the probe reads installed + state: a refused binding is unavailable, named on the exit line
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
