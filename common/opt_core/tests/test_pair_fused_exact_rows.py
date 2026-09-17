"""attn.pair_fused "exact_rows": the exact construction's served-row rule per compute capability (the stock GEMMs' cuBLAS piece-remainder
rule), read as data — pure python, no torch. The rule refuses BY NAME ('exact-rows:below_min_rows' | 'exact-rows:above_max_rows') and never
substitutes; a card without an entry serves every row count; kernels.transition's exact tier honours the same statement."""
import pytest

from opt_core.attn import pair_fused as PF
from opt_core.kernels import transition as T

A100 = "A100:torch2.13.0+cu130/3.7.1"
H100 = "H100:torch2.13.0+cu130/3.7.1"


def test_the_table_states_one_rule_for_cc80_and_none_for_cc90():
    ex = PF.cells()["exact_rows"]
    assert ex["rule"] and ex["words"] == {"below": "below_min_rows", "above": "above_max_rows"}
    card = ex["cards"]["8.0"]
    assert (card["piece_rows"], card["max_rows"], card["min_remainder"], card["top_margin"]) == (1048576, 1048544, 161424, 8192)
    assert card["pieces"] == {"transition": {"min_rows": 4096}, "prologue": {"min_rows": 5857}}
    assert card["measured_on"] and card["evidence"]
    assert "9.0" not in ex["cards"]
    for piece in PF.EXACT_ROWS_PIECES:
        assert PF.exact_rows_rule("9.0", piece) is None and PF.exact_rows_word(piece, "9.0", 1025 * 1025) is None and PF.exact_rows_facts("9.0", piece) is None
        assert PF.exact_rows_rule(None, piece) is None
    assert PF.exact_rows_rule("8.0", "epilogue") is None                       # a piece the rule does not name: nothing refused


@pytest.mark.parametrize("piece,floor", [("transition", 4096), ("prologue", 5857)])
def test_the_words_at_the_measured_boundaries_cc80(piece, floor):
    w = lambda rows: PF.exact_rows_word(piece, "8.0", rows)
    assert w(floor - 1) == "below_min_rows" and w(floor) is None and w(0) == "below_min_rows"
    assert w(1023 * 1023) is None and w(1048543) is None                        # below max_rows: served
    assert w(1048544) == "above_max_rows"                                       # 2^20 - 32 rows and more: the remainder decides
    assert w(1025 * 1025) == "above_max_rows"                                   # remainder 2,049
    assert w(1028 * 1028) == "above_max_rows"                                   # remainder 8,208 (measured NOT identical)
    assert w(1099 * 1099) == "above_max_rows" and w(1100 * 1100) is None        # remainder 159,225 | 161,424 (the smallest measured identical)
    for n in (1152, 1200, 1280, 1340, 1445, 1536, 1771, 1819, 2045):            # measured identical / inside the served band
        assert w(n * n) is None, n
    for n in (1446, 1502, 1772, 1818, 2046, 2048):                              # remainder above 2^20 - 8192 or below 161,424 (2048: remainder 0)
        assert w(n * n) == "above_max_rows", n
    f = PF.exact_rows_facts("8.0", piece)
    assert f == {"cc": "8.0", "min_rows": floor, "max_rows": 1048544, "piece": 1048576, "piece_min": 161424, "piece_top_margin": 8192,
                 "below": "exact-rows:below_min_rows", "above": "exact-rows:above_max_rows"}


def test_the_transition_providers_exact_tier_honours_the_rule_by_name_and_the_tolerance_words_do_not():
    for n in (1025, 1028, 1099, 2048):
        with pytest.raises(T.Refusal) as r:
            T.select("exact", c=128, hidden=512, n_tokens=n, cc="8.0", stack=A100, ln_given=True, rows_count=n * n)
        assert r.value.kind == "exact_rows_above_max_rows_on_cc_8.0" and r.value.fallback == "torch_swiglu", (n, r.value.kind)
        f = T.select("fast", c=128, hidden=512, n_tokens=n, cc="8.0", stack=A100, rows_count=n * n)      # tolerance class: served as before
        assert f.row in T.ROW_NAMES and f.row not in T.STOCK_ROWS, (n, f.line())
        b = T.select("big", c=128, hidden=512, n_tokens=n, cc="8.0", stack=A100, rows_count=n * n)
        assert b.row in T.ROW_NAMES and b.row not in T.STOCK_ROWS, (n, b.line())
    for n in (64, 400, 1023, 1100, 1200, 1445, 1536):                                                    # inside the served set: the vouched kernel arm
        s = T.select("exact", c=128, hidden=512, n_tokens=n, cc="8.0", stack=A100, ln_given=True, rows_count=n * n)
        assert s.row == "v1" and s.tier == "exact", (n, s.line())
    s = T.select("exact", c=128, hidden=512, n_tokens=1025, cc="8.0", stack=A100, ln_given=True)         # no row count given: the vouch by size decides (planning calls)
    assert s.row == "v1"
    with pytest.raises(T.Refusal) as r:                                                                   # under the transition floor the size vouch already refuses (64 tokens = 4096 rows)
        T.select("exact", c=128, hidden=512, n_tokens=63, cc="8.0", stack=A100, ln_given=True, rows_count=63 * 63)
    assert r.value.kind.startswith("exact_vouch_below_64_tokens_on_") or r.value.kind == "exact_rows_below_min_rows_on_cc_8.0", r.value.kind
    for n in (1025, 1028, 2048):                                                                          # cc 9.0: no rule, the kernel arm at every vouched size
        s = T.select("exact", c=128, hidden=512, n_tokens=n, cc="9.0", stack=H100, ln_given=True, rows_count=n * n)
        assert s.row == "v1", (n, s.line())
    assert T.table()["exact_rows_rule"]
