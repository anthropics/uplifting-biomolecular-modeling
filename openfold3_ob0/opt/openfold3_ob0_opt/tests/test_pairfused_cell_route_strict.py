"""The trimul provider's small-N route under strict. The hazard: without the exemption below, the fast and big lines fail BY NAME on every
item of <= 100 tokens (DNA duplexes, short RNAs, small interfaces) with `[openfold3_ob0-opt/pairfused] trimul_v4: the core refused the trunk
shape (c_z=128, N=24): cell:cueq — OPENFOLD3_OB0_OPT_PAIR_STRICT=1 …`, while items of >= 101 tokens complete.

Mechanism: the core's ONE triangle-multiplication table (opt_core.kernels.trimul TRIMUL_CELLS.json) gives the c_z-128 `N<=100` token bucket to
the stock op on EVERY tier (fast = exact = big = cueq, cc 9.0 and 8.0) — below fpf_trimul_v4's own floor (generic.N_MIN = 101) the vendor
TriMul is the right statement.  of3_trimul names that verdict `cell:<row>` (STOCK_ROWS cueq / torch_math) and the pair cell runs the line's own
TriMul for the class, counted `fallback:cell:cueq=<n>` / `line=cell:cueq:<n>` — a ROUTE by design, the provider-era spelling of the `n<101`
word.  STRICT_EXEMPT carries `cell:` so strict (set by the mode table on the fast and big lines) does not turn the route into a fatal; every other
trunk-shape refusal word still fails the run under strict.
CPU only: the table is the shipped one; the provider is a stub; no torch tensor is built."""
import types

import pytest

from openfold3_ob0_opt import of3_trimul
from openfold3_ob0_opt.cells import pairfused
from opt_core.kernels import trimul as KT


@pytest.fixture
def strict_state():
    import copy
    saved = {k: (copy.copy(v) if isinstance(v, (dict, list)) else v) for k, v in pairfused.STATE.items()}
    pairfused.STATE["counts"] = {n: {"served": 0, "fallback": 0} for n in pairfused.LEVER_NAMES}
    pairfused.STATE["reasons"] = {n: {} for n in pairfused.LEVER_NAMES}
    if "degraded" in pairfused.STATE:
        pairfused.STATE["degraded"] = {n: {} for n in pairfused.LEVER_NAMES}
    pairfused.STATE["strict"] = True
    yield pairfused.STATE
    pairfused.STATE.clear(); pairfused.STATE.update(saved)


@pytest.mark.parametrize("residency", [None, "fp32"])                     # bf16 z (bf16 autocast) and the fp32-residency trunk (f32z_bf16)
def test_the_core_table_names_a_row_for_every_small_class_and_stock_rows_are_the_route(residency):
    """The shipped table answers every c_z-128 class of 1..100 tokens on the measured capabilities (a cell exists: the provider never meets an
    unkeyed small class), and wherever that answer is one of the provider's STOCK_ROWS the kit's route applies (this file's other tests).  The
    verdict itself belongs to the core's table (the stock op on every tier below 101 tokens, or a kernel row per class and column where the
    table names one) -- not pinned here."""
    cells = KT.table()["cells"]
    for cc in ("9.0", "8.0"):
        for n in (1, 12, 24, 33, 61, 81, 88, 100):
            for direction in ("outgoing", "incoming"):
                key, _exact, _note = KT.cell_key(cc, "bf16", 128, 128, n, direction, residency=residency)
                assert key is not None and "|C128|H128|" in key, (cc, n, key)
                cell = cells[key]
                for tier in ("fast", "exact", "big"):
                    assert cell[tier] in of3_trimul.STOCK_ROWS or cell[tier] in KT.ROW_NAMES, (key, tier, cell[tier])
    assert set(of3_trimul.STOCK_ROWS) >= {"cueq", "torch_math"}

def test_the_router_names_a_stock_row_verdict_cell_and_plans_the_line(monkeypatch):
    """of3_trimul.Router.plan on a class whose row is a STOCK_ROW: Plan kind 'line', reason `cell:<row>` (the census token), memoised per class."""
    class StubProvider:                                                     # the core provider's face the router uses: eligible(call) leaves the Selection on call.extra
        def eligible(self, call):
            call.extra["trimul_selection"] = types.SimpleNamespace(row="cueq", cell="9.0|bf16|C128|H128|N<=100|out|fwd")
    import opt_core.trimul as T
    monkeypatch.setattr(T, "by_word", lambda *a, **k: StubProvider())
    monkeypatch.setattr(of3_trimul.Router, "facts", staticmethod(lambda z4, mod: ("9.0", "bf16", 128, 128, 24)))
    r = of3_trimul.Router("fast", weights_of=lambda mod: {})
    plan = r.plan(None, None, None, "outgoing", None)
    assert (plan.kind, plan.row, plan.reason) == ("line", "cueq", "cell:cueq")
    assert plan.key == "9.0|bf16|C128|H128|N<=100|out|fwd" and r.cells[plan.key] == "line(cell:cueq)"
    assert r.plan(None, None, None, "outgoing", None) is plan                # memoised: the table walk happens once per class


def test_a_stock_row_verdict_is_a_route_under_strict_and_every_other_trunk_word_still_fails(strict_state):
    C = pairfused.TRUNK_C
    assert "cell:" in pairfused.STRICT_EXEMPT
    for word in ("cell:cueq", "cell:torch_math", "cell:e01:stock_block(256<keys<512)", "n<101", "no-cell:fpf:prologue:128x4x32:9.0|3.7" if "no-cell:" in pairfused.STRICT_EXEMPT else "n<101"):
        pairfused._strict_refusal("trimul_v4", word, C, 24)               # routes: return
    for word in ("dtype:float16", "mask-batch", "mask-shape", "refused", "fold:rows", "module:Foo"):
        with pytest.raises(RuntimeError):
            pairfused._strict_refusal("trimul_v4", word, C, 24)           # defects of the line (or nothing served at all): fail the run by name
    pairfused._strict_refusal("trimul_v4", "dtype:float16", 64, 24)        # the template width is never the trunk: a route whatever the word


def test_the_route_is_counted_on_the_census_not_as_a_degradation(strict_state):
    st = strict_state
    st.update(installed=True, levers=["trimul_v4"], cells={})
    for _ in range(3):
        pairfused._count("trimul_v4", False, "cell:cueq", C=pairfused.TRUNK_C)
    pairfused._count("trimul_v4", True)
    line = [l for l in pairfused.census_lines() if "name=trimul_v4 " in l][0]
    assert line.startswith("[openfold3_ob0-opt/pairfused] LEVER name=trimul_v4 state=on served=1 fallback=3 ") and " fallback:cell:cueq=3" in line, line
    if hasattr(pairfused, "fallbacks"):                                    # the kit whose EXIT line aggregates the cells' degradations (`fallbacks=`): a route is never one
        assert " degraded=0 " in line, line
        assert pairfused.fallbacks() == {}
        pairfused._count("trimul_v4", False, "dtype:float16", C=pairfused.TRUNK_C)
        assert pairfused.fallbacks() == {"trimul_v4:dtype:float16": 1}
