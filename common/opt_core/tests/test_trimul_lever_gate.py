"""kernels.trimul / opt_core Lever served-floor gates: an all-stock-by-cell answer set is a named aside (stock_row_by_cell), not a floor
violation; a launch fallback / error keeps the violation; served kernel rows unchanged (CPU; fake providers)."""
from opt_core import trimul as LT
from opt_core.counters import Ledger, is_stock_by_cell, stock_by_cell_aside, STOCK_BY_CELL_ASIDE


class _Z:
    def __init__(self, n, c):
        self.shape = (n, n, c); self.dtype = "bfloat16"

    def dim(self):
        return 3


def _call(n=300, c=128, d="outgoing"):
    return LT.Call(None, _Z(n, c), None, d, False, lambda: "stock")


def _lever(fn, elig, expected):
    return LT.Lever("acme-opt", "fast", provider=LT.custom("fake", fn, elig, "0"), min_tokens=0, expected=expected)


def test_lever_served_floor_all_stock_by_cell_is_a_named_aside_and_launch_fallbacks_keep_the_violation():
    def stock_by_cell(call):                                                   # the provider names the cell's stock row for EVERY call (the engine's own statement serves it)
        raise LT.Refused("stock:cueq")

    def ok(call):
        return None

    # (1) n calls, all answered stock-by-cell, 0 fallbacks of any other kind -> aside stock_row_by_cell, gate passes (both directions)
    lv = _lever(lambda call: "kernel", stock_by_cell, expected=("stock:cueq",))
    for i in range(6):
        assert lv.serve(_call(d="outgoing" if i % 2 else "incoming")) == "stock"
    g = lv.gate()
    assert g.ok and g.reason is None and g.details.get("aside") == STOCK_BY_CELL_ASIDE and ("aside:" + STOCK_BY_CELL_ASIDE) in g.words, g
    assert lv.census()["served"] == 0 and lv.census()["fallback"] == {"stock:cueq": 6}
    assert "aside=stock_row_by_cell" in lv.line() and "gate=ok" in lv.line(), lv.line()
    # the same answers UNDECLARED by the kit: unexpected fallback refuses as before (the aside needs the kit's declaration)
    lvu = _lever(lambda call: "kernel", stock_by_cell, expected=())
    lvu.serve(_call()); assert not lvu.gate().ok and "unexpected fallback" in lvu.gate().reason
    # (2) the same but ONE launch fallback (a kernel error rerouted to the engine's forward) -> violation as today
    state = {"n": 0}

    def elig_ok(call):
        return None

    def boom_once_then_stock(call):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("launch failed: illegal memory access")
        raise LT.Refused("stock:cueq")
    lv2 = _lever(boom_once_then_stock, elig_ok, expected=("stock:cueq",))
    for _ in range(5):
        assert lv2.serve(_call()) == "stock"
    g2 = lv2.gate()
    assert not g2.ok and "RuntimeError" in g2.reason, g2
    # (2b) a non-stock fallback among the stock answers (a row that refused at launch by name) -> 'served 0 of n' violation kept
    state3 = {"n": 0}

    def elig_mixed(call):
        state3["n"] += 1
        raise LT.Refused("row_error" if state3["n"] == 3 else "stock:cueq")
    lv3 = _lever(lambda call: "kernel", elig_mixed, expected=("stock:cueq", "row_error"))
    for _ in range(5):
        lv3.serve(_call())
    g3 = lv3.gate()
    assert not g3.ok and "served 0 of 5 calls" in g3.reason, g3
    # a size-gated lever that never served is still a violation (below_min_tokens is not an answer by cell)
    lv4 = LT.Lever("acme-opt", "fast", provider=LT.custom("fake", lambda call: "kernel", ok, "0"), min_tokens=10 ** 6, expected=("below_min_tokens",))
    lv4.serve(_call()); assert not lv4.gate().ok and "served 0 of 1 calls" in lv4.gate().reason
    # (3) kernel rows served >= floor -> unchanged (ok, no aside)
    lv5 = _lever(lambda call: "kernel", ok, expected=("stock:cueq",))
    for _ in range(4):
        assert lv5.serve(_call()) == "kernel"
    g5 = lv5.gate()
    assert g5.ok and not g5.details.get("aside") and lv5.census()["served"] == 4 and "aside=" not in lv5.line()


def test_ledger_served_floor_all_stock_by_cell_is_a_named_aside_and_launch_fallbacks_keep_the_violation():
    # (1) every call answered with the statement / stock row by cell, declared -> aside, gate passes
    L = Ledger("F7.triattn", impl="triattn_core", origin="core", expected=("stock_statement", "stock"))
    for _ in range(5):
        L.fallback("stock_statement")
    L.fallback("stock", 3)
    g = L.gate()
    assert g.ok and g.details["aside"] == STOCK_BY_CELL_ASIDE and ("aside:" + STOCK_BY_CELL_ASIDE) in g.words and g.details["served"] == 0, g
    # (2) one launch fallback among them (row_error: a row failed to build / launch, the module served) -> violation as today
    L2 = Ledger("F7.triattn", impl="triattn_core", origin="core", expected=("stock_statement", "row_error"))
    for _ in range(5):
        L2.fallback("stock_statement")
    L2.fallback("row_error")
    g2 = L2.gate()
    assert not g2.ok and "0 served" in g2.reason, g2
    # (2c) a counted kernel error -> refused as today
    L3 = Ledger("F4.transition", expected=("stock",)); L3.fallback("stock"); L3.error(RuntimeError("x"))
    assert not L3.gate().ok and "kernel error" in L3.gate().reason
    # (3) served >= 1 -> unchanged; require_served=False unchanged
    L4 = Ledger("F4.transition", expected=("stock",)); L4.serve("300x128"); L4.fallback("stock", 2)
    assert L4.gate().ok and not L4.gate().details.get("aside")
    L5 = Ledger("F9.ln", expected=("no_cell",)); L5.fallback("no_cell")
    assert not L5.gate().ok and L5.gate(require_served=False).ok
    # the predicate: answers by cell vs launch fallbacks / size gates / refusals by shape
    for r in ("stock", "stock:cueq", "stock_statement", "statement", "statement:layer_norm", "stock_row_by_cell", "cueq:c_hidden=128!=c_z=64", "torch_math", "stock(cueq)"):
        assert is_stock_by_cell(r), r
    for r in ("row_error", "error:RuntimeError", "below_min_tokens", "mode_stock", "no_cell", "v4:c_z:384", "tx_sm90a:no_prebuilt:cp311", "class_differs", "refused", "", None):
        assert not is_stock_by_cell(r), r
    assert stock_by_cell_aside({"stock": 2}, 0, {}) == STOCK_BY_CELL_ASIDE and stock_by_cell_aside({"stock": 2}, 1, {}) is None
    assert stock_by_cell_aside({"stock": 2, "row_error": 1}, 0, {}) is None and stock_by_cell_aside({"stock": 2}, 0, {"error:X": 1}) is None and stock_by_cell_aside({}, 0, {}) is None
