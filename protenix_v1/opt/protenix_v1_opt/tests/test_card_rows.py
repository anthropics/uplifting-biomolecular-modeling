"""The exact-class pair levers' served ROWS per card: the rule is the shared core's (opt_core.attn.pair_fused "exact_rows": compute capability
8.0 only — the fused triangle-attention block (gblock, piece `prologue`) from 5857 rows = inputs of 77 tokens, the exact transition (xtr /
tmpl_xtr, piece `transition`) from 4096 rows = 64 tokens, both below 1048544 rows = inputs up to 1023 tokens; from that top on a call is served
when its remainder piece rows mod 2^20 lies within [161424, 2^20 - 8192] — inputs of 1100..1445, 1503..1771, 1819..2045 tokens; no entry for
9.0), the kit's per-call reading of it (levers_ptx1.exact_rows_word -> the census words stock:below_min_rows / stock:above_max_rows; the
provider's exact tier refuses the transition's rows by the same statement), and the exit rule's reading (report.range_state: a run whose every
call is outside the served rows idles the lever — `range-off`, state=on, not partial; the LEVER line names `min_rows= max_rows= range=`; a card
without an entry adds nothing to the line and keeps the served-0 rule). The kit carries NO row table of its own (kit 0.2.39): the lever file is
read (AST) to prove that — torch-free; the helper test imports the lever module where torch and the shared core are importable."""
import ast
import os
import sys

import pytest

from protenix_v1_opt import kit as K, modes, report as R
from protenix_v1_opt.tests._ditfast_account import DF_OK as _DF_OK

EXACT = modes.resolve("exact")
RANGE_80 = (5857, 1048544)                                                       # gblock on 8.0 (the core's prologue piece)
XTR_80 = (4096, 1048544)                                                         # xtr on 8.0 (the core's transition piece)
PIECES_80 = (1048576, 161424, 8192)                                              # (piece rows, smallest served remainder, top margin) on 8.0


def _assigned_names():
    tree = K._levers_module(None)
    return {t.id for node in tree.body if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name)}


def test_the_kit_carries_no_row_table_the_core_states_the_rule():
    names = _assigned_names()
    assert "CARD_ROWS" not in names and "CARD_PIECES" not in names                # the kit-side card table left with kit 0.2.39: the core's exact_rows is the one statement
    assert {"RANGED_LEVERS", "RANGED_PIECES", "BELOW_MIN_ROWS", "ABOVE_MAX_ROWS"} <= names
    src = open(K.levers_file(None)).read()
    for fn in ("def row_range(", "def row_piece(", "def row_range_word("):
        assert fn not in src, fn
    assert "exact_rows_word" in src and "exact_rows_facts" in src               # read from opt_core.attn.pair_fused
    PF = pytest.importorskip("opt_core.attn.pair_fused")
    if not hasattr(PF, "exact_rows_rule"):
        pytest.skip("opt_core below the exact_rows rule")
    g = PF.exact_rows_rule("8.0", "prologue"); t = PF.exact_rows_rule("8.0", "transition")
    assert (g["min_rows"], g["max_rows"]) == RANGE_80 and (t["min_rows"], t["max_rows"]) == XTR_80
    assert (t["piece_rows"], t["min_remainder"], t["top_margin"]) == PIECES_80 == (g["piece_rows"], g["min_remainder"], g["top_margin"])
    assert PF.exact_rows_rule("9.0", "prologue") is None and PF.exact_rows_rule("9.0", "transition") is None


def test_no_environment_spelling_reads_or_sets_the_range():
    src = open(K.levers_file(None)).read()
    for spelling in ("GBLOCK_MIN_ROWS", "GBLOCK_MAX_ROWS", "XTR_MAX_ROWS", "PTX1_A100"):
        assert spelling not in src, spelling


def test_the_kits_words_follow_the_cores_rule_per_call():
    """levers_ptx1's census words for a refused row count == the core's rule (torch + opt_core importable; the lever module imported)."""
    L = pytest.importorskip("levers_ptx1")
    PF = pytest.importorskip("opt_core.attn.pair_fused")
    if not hasattr(PF, "exact_rows_word"):
        pytest.skip("opt_core below the exact_rows rule")
    assert L._rows_census_word("below_min_rows") == L.BELOW_MIN_ROWS == "stock:below_min_rows"
    assert L._rows_census_word("exact_rows_above_max_rows_on_cc_8.0") == L.ABOVE_MAX_ROWS == "stock:above_max_rows"
    assert L._rows_census_word("exact-rows:above_max_rows") == L.ABOVE_MAX_ROWS and L._rows_census_word("no_cell:x") is None
    assert L.RANGED_PIECES == {"gblock": "prologue", "xtr": "transition"}
    for n, ok in ((1024, False), (1025, False), (1028, False), (1100, True), (1200, True), (1340, True), (1445, True), (1446, False), (1503, True), (1536, True), (1772, False), (2045, True)):
        for piece in ("prologue", "transition"):                                 # from the top on: the remainder piece decides, both levers alike
            w = PF.exact_rows_word(piece, "8.0", n * n)
            assert (w is None) is ok, (piece, n)
            if not ok:
                assert L._rows_census_word(w) == L.ABOVE_MAX_ROWS                # the same word: a stock route by name
            assert PF.exact_rows_word(piece, "9.0", n * n) is None               # 9.0: no rule
    assert PF.exact_rows_word("prologue", "8.0", 45 * 45) == "below_min_rows" and PF.exact_rows_word("prologue", "8.0", 77 * 77) is None
    assert PF.exact_rows_word("transition", "8.0", 63 * 63) == "below_min_rows" and PF.exact_rows_word("transition", "8.0", 64 * 64) is None
    assert PF.exact_rows_word("transition", "8.0", 1023 * 1023) is None


def _exact_account(triattn_counts, rows=None):
    """A kit account (levers_ptx1.describe(), JSON-safe) of the exact arm: the TriMul and transitions served; the triangle attention as given."""
    acc = {"cfg": {"trimul": "exact", **{lv: True for lv in EXACT.levers}},
           "counts": {"dit": {"apb:fp16": 48}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "atom": {"apb:tf32rn": 12}, "trimul": {"exact": 2160, "stock:path": 320}, "triattn": dict(triattn_counts), "transition": {"xtr:C=128": 1084, "stock:C=384": 1008}},
           "tg": {"max_tokens": 4096, "fastln": {"ok": True, "how": "prebuilt"}, "stats": {"captures": 1, "graphed_calls": 9, "eager_gate": 10},
                  "graphs": {"0": {"entries": 1, "stats": {"replay": 9, "captures": 1}}}},
           "keep_pool": {"installed": True, "skipped_total": 3, "passed_total": 1, "errors": 0}, "summary_hostidx": {"installed": True, "samples": 5, "delegated": 0, "errors": 0}, "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 3.9, "patched": 12}, "ditfast": _DF_OK, "templ": {"template_dedupe": {"installed": True, "calls": 1, "evaluated": 2, "reused": 2, "stock": {}}, "tmpl_triatt": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul_exact": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_xtr": {"on": True, "routed_total": 4, "stock": {}, "fallback": {}}, "tmpl_pairfused": {"on": True, "calls": 8}}, "dit_attn_exact": {"installed": True, "calls": 72, "routes": {"kernel": 72}}, "apb": {"dit": {"engaged": True, "fp16": True, "opd": "fp16", "cell_key": "9.0", "installed_on": 24, "calls": 48}, "atom": {"engaged": True, "opd": "tf32rn", "cell_key": "9.0", "installed_on": 6, "calls": 12}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "sampler": {"graphs": True, "prep": {"on": True, "parts": "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec", "poison": "ok", "stats": {}, "aside": None}, "hoist_installed": True, "fastln": {"how": "prebuilt"}, "errors": [],
                       "sampler": {"captures": 1, "replays": 398, "eager_steps": 0, "bypass": 0}, "hoist": {"hits": 339, "records": 1, "bypass": 0}}}
    if rows is not None:
        acc["card_rows"] = rows
    return acc


GB_FACTS = {"cc": "8.0", "min_rows": RANGE_80[0], "max_rows": RANGE_80[1], "below": "stock:below_min_rows", "above": "stock:above_max_rows"}
XTR_FACTS = {"cc": "8.0", "min_rows": XTR_80[0], "max_rows": XTR_80[1], "below": "stock:below_min_rows", "above": "stock:above_max_rows"}
ROWS_80 = {"gblock": GB_FACTS, "xtr": XTR_FACTS}                                # levers_ptx1.row_range_facts() on the A100 under the exact arm (the core's numbers)


def _line(ev, lever):
    lines = [l for l in R.lever_lines(ev, {"mode": "exact"}) if f" lever={lever}" in l]
    assert len(lines) == 1, lines
    return lines[0]


def _gblock_line(ev):
    return _line(ev, "gblock")


def test_an_input_wholly_below_the_range_idles_the_lever_not_partial():
    acc = _exact_account({"stock:below_min_rows": 2080, "stock:c=64": 320, "stock:gate": 80}, ROWS_80)   # a 45-token input on the A100: every c=128 call under min_rows
    ev = R.kit_evidence(acc, EXACT.trimul, EXACT.levers, [{"N_token": 45}])
    assert ev["gblock"]["served"] == 0 and ev["gblock"]["fallback"] == {}
    assert ev["gblock"]["rows"] == {"min_rows": 5857, "max_rows": 1048544, "calls_below": 2080, "calls_above": 0, "state": "range-off"}
    partial, _ = R.partial_of(ev)
    assert "gblock" not in partial
    line = _gblock_line(ev)
    assert " state=on " in line and " served=0 gated=2480 gated_by=stock:below_min_rows:2080,stock:c=64:320,stock:gate:80 " in line
    assert " min_rows=5857 max_rows=1048544 range=range-off " in line


def test_an_input_above_the_range_idles_both_ranged_levers():
    acc = _exact_account({"stock:above_max_rows": 1920, "stock:c=64": 320, "stock:gate": 80}, ROWS_80)   # a >= 1024-token input on the A100: the block AND the fused transition over max_rows
    acc["counts"]["transition"] = {"stock:above_max_rows": 1084, "stock:C=64": 220, "stock:C=384": 1008}
    ev = R.kit_evidence(acc, EXACT.trimul, EXACT.levers, [{"N_token": 1024}])
    assert ev["gblock"]["rows"]["state"] == "range-off" and ev["gblock"]["rows"]["calls_above"] == 1920
    assert ev["xtr"]["rows"] == {"min_rows": 4096, "max_rows": 1048544, "calls_below": 0, "calls_above": 1084, "state": "range-off"}
    assert R.partial_of(ev) == ([], None)
    assert " range=range-off " in _gblock_line(ev)
    xl = _line(ev, "xtr")
    assert " state=on " in xl and " served=0 gated=2312 gated_by=stock:above_max_rows:1084,stock:C=64:220,stock:C=384:1008 " in xl and " min_rows=4096 max_rows=1048544 range=range-off " in xl


def test_inside_the_range_the_levers_serve_and_the_lines_name_the_range():
    acc = _exact_account({"gblock": 2080, "stock:c=64": 320, "stock:gate": 80}, ROWS_80)                # a 995-token input on the A100
    ev = R.kit_evidence(acc, EXACT.trimul, EXACT.levers, [{"N_token": 995}])
    assert ev["gblock"]["rows"]["state"] == "served" and ev["xtr"]["rows"]["state"] == "served" and R.partial_of(ev) == ([], None)
    line = _gblock_line(ev)
    assert " state=on " in line and " served=2080 " in line and " min_rows=5857 max_rows=1048544 range=served " in line
    assert " served=1084 " in _line(ev, "xtr") and " min_rows=4096 max_rows=1048544 range=served " in _line(ev, "xtr")


def test_a_card_without_a_range_adds_nothing_and_keeps_the_served0_rule():
    for rows in (None, {}):                                                      # 9.0: levers_ptx1.row_range_facts() is {} (or the key absent in an older account)
        acc = _exact_account({"gblock": 2080, "stock:c=64": 320, "stock:gate": 80}, rows)
        ev = R.kit_evidence(acc, EXACT.trimul, EXACT.levers, [{"N_token": 995}])
        assert "rows" not in ev["gblock"] and "rows" not in ev["xtr"]
        for line in (_gblock_line(ev), _line(ev, "xtr")):
            assert "min_rows=" not in line and "max_rows=" not in line and " range=" not in line
        acc0 = _exact_account({"stock:gate": 30}, rows)                            # no block call and no range: served 0 is partial, the gate named (the rule every card had)
        partial, reason = R.partial_of(R.kit_evidence(acc0, EXACT.trimul, EXACT.levers))
        assert partial == ["gblock"] and reason == "gblock served 0 calls (gated {'stock:gate': 30})"


def test_a_refused_call_stays_partial_whatever_the_range():
    acc = _exact_account({"stock:below_min_rows": 2000, "fallback:gblock:x_ln-shape": 80}, ROWS_80)
    partial, reason = R.partial_of(R.kit_evidence(acc, EXACT.trimul, EXACT.levers))
    assert partial == ["gblock"] and "gblock fallback" in reason
