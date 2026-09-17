"""boltz2_opt.rowfloor: the per-card served ROW RULE arithmetic of a bitwise pair-track variant (pairblock.CARD_ROWS, compute capability 8.0):
(min_rows, max_rows, piece_rows) -> served, or the word naming why not."""
import pytest


def test_row_rule_helpers():
    from .. import pairblock as PB, rowfloor as RF
    P = RF.PIECE_ROWS_SM80
    assert P == 2 ** 20 - 32 == 1048544
    assert PB.CARD_ROWS == {"8.0": {"cueq": (5857, 3 * P, P)}}
    assert RF.WORDS == ("below_min_rows", "above_max_rows", "trailing_piece") and set(RF.WORDS) <= set(PB.EXPECTED)
    assert RF.range_for(PB.CARD_ROWS, "cueq", "8.0") == (5857, 3145632, 1048544)
    for v in ("flash", "k2b"):
        assert RF.range_for(PB.CARD_ROWS, v, "8.0") is None, (v, "Tier 2: no rule")
    assert RF.range_for(PB.CARD_ROWS, "cueq", "9.0") is None and RF.range_for(PB.CARD_ROWS, "cueq", None) is None and RF.range_for(PB.CARD_ROWS, None, "8.0") is None
    rng = (3137, 3 * P, P)
    assert RF.outside(3136, rng) == "below_min_rows" and RF.outside(3137, rng) is None and RF.outside(1048543, rng) is None and RF.outside(3 * P, rng) == "above_max_rows"
    assert RF.outside(P, rng) == "trailing_piece" and RF.outside(P + 3136, rng) == "trailing_piece" and RF.outside(P + 3137, rng) is None, "a pieced call is served exactly when its trailing piece clears the floor"
    assert RF.outside(5, None) is None and RF.facts(None) == {} and RF.facts(rng) == {"min_rows": 3137, "max_rows": 3145632, "piece_rows": 1048544}
    assert RF.call_facts(1048543, rng) == {} and RF.call_facts(1440000, rng) == {"pieces": 2, "trailing": 391456} and RF.call_facts(1440000, None) == {} and RF.call_facts(1440000, (3137, 3 * P, 0)) == {}
    torch = pytest.importorskip("torch")
    assert RF.rows_of(torch.zeros(1, 56, 56, 128)) == 3136 and RF.rows_of(torch.zeros(1, 57, 57, 128)) == 3249 and RF.rows_of(torch.zeros(5, 20, 20, 128)) == 2000 and RF.rows_of(torch.zeros(7, 128)) == 7
    assert 1023 * 1023 < 1048544 <= 1024 * 1024 and 76 * 76 < 5857 <= 77 * 77 and 1773 * 1773 < 3 * P <= 1774 * 1774, "token edges of the pairblock rule: 77..1,773 tokens, the piece from 1,024"


def test_the_piece_arithmetic_of_the_row_rule():
    """rowfloor.pieces / outside on compute capability 8.0's piece size (2**20 - 32 rows): rows -> (pieces, trailing) and served-or-word at the
    row counts the models and the identity probes run at (N tokens -> N·N rows), for a 3,137-row and the pairblock 5,857-row floor."""
    from .. import pairblock as PB, rowfloor as RF
    P = RF.PIECE_ROWS_SM80
    lo, pb = (3137, 3 * P, P), RF.range_for(PB.CARD_ROWS, "cueq", "8.0")
    table = [  # rows, (pieces, trailing), 3,137-floor word, pairblock word
        (56 * 56, (1, 3136), "below_min_rows", "below_min_rows"), (57 * 57, (1, 3249), None, "below_min_rows"), (77 * 77, (1, 5929), None, None),
        (1023 * 1023, (1, 1046529), None, None), (1024 * 1024, (2, 32), "trailing_piece", "trailing_piece"), (1025 * 1025, (2, 2081), "trailing_piece", "trailing_piece"),
        (1026 * 1026, (2, 4132), None, "trailing_piece"), (1027 * 1027, (2, 6185), None, None), (1200 * 1200, (2, 391456), None, None),
        (1449 * 1449, (3, 2513), "trailing_piece", "trailing_piece"), (1450 * 1450, (3, 5412), None, "trailing_piece"), (1451 * 1451, (3, 8313), None, None),
        (1773 * 1773, (3, 1046441), None, None), (1774 * 1774, (4, 1444), "above_max_rows", "above_max_rows")]
    for rows, pt, w_lo, w_pb in table:
        assert RF.pieces(rows, lo) == pt, rows
        assert RF.outside(rows, lo) == w_lo and RF.outside(rows, pb) == w_pb, rows
