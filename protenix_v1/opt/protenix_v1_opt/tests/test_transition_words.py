"""`xtr` / `ttr`: the pair (c=128 x 512) and MSA (c=64 x 256) transitions through the shared core's transition provider (opt_core.kernels.transition)
by TIER word only — xtr asks `exact` (the exact construction: the module's LayerNorm output handed in; refused by name -> the stock module, bytes
== off), ttr asks `fast` (`big` under --mode big). Class contracts only: the tier word passed, the refusal-by-name path, the census words the
report reads; no assertion on the row the core serves today. The retired rider word `ttrv2` (kit 0.2.22-0.2.38) is folded into ttr. No GPU needed."""
import ast

import pytest

from .conftest import KIT


def test_words_wired_and_the_rider_word_retired():
    from protenix_v1_opt import kit as K, modes as M, report as R
    _, levers = K.lever_grammar(KIT)
    assert "xtr" in levers and "ttr" in levers and "ttrv2" not in levers
    assert M.LEVERS["xtr"]["class"] == "exact" and M.LEVERS["ttr"]["class"] == "tolerance" and "ttrv2" not in M.LEVERS
    assert "xtr" in K.parse_arm(M.KIT_MODES["exact"].arm, KIT)[1]
    for m in ("fast", "big"):
        lv = K.parse_arm(M.KIT_MODES[m].arm, KIT)[1]
        assert "ttr" in lv and "xtr" not in lv and "ttrv2" not in lv, m
    with pytest.raises(Exception):
        K.parse_arm("fast+gflash+ttr+ttrv2", KIT)                                          # the retired word is refused by the grammar, by name
    assert "ttrv2" not in R.STRATEGY_IDS


def test_the_tier_word_per_mode_is_stated_in_the_lever_file():
    """levers_ptx1.transition_tier: xtr -> exact; ttr -> fast, big under --mode big (kit_mode()) — read from the source (torch-free)."""
    from protenix_v1_opt import kit as K
    src = open(K.levers_file(KIT)).read()
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "transition_tier"]
    assert len(fn) == 1
    body = ast.get_source_segment(src, fn[0])
    assert 'return "exact"' in body and 'return "big" if kit_mode() == "big" else "fast"' in body
    assert "TTRV2_WORD" not in src and "def _provider_transition(" not in src                # no rider: one binding
    seg = src[src.index("def _transition_forward("):src.index("# =====", src.index("def _transition_forward("))]
    assert "T.select(word," in seg and "T.transition(xs," in seg and "word=word" in seg      # the provider by the tier word: select (plan) then transition (serve)
    assert "ln_given=(lever == \"xtr\")" in seg and "x_ln=(self.layernorm1(xs) if lever == \"xtr\" else None)" in seg   # the exact construction hands the module's LayerNorm output in
    assert "rows_count=rows" in seg                                                          # the card's served-row rule is the provider's to apply (exact_rows)
    assert "sel.row in T.STOCK_ROWS" in seg and "return orig(self, x)" in seg                # a stock-row tier / a refusal -> the module's own forward, by name
    for gone in ("row_range_word(", "CARD_ROWS", "impl=impl", "PF.transition(xs, T"):       # no kit cell table, no kit-side pair_fused statement for these calls
        assert gone not in seg, gone


def test_provider_face_serves_the_kits_shapes_by_tier_word():
    T = pytest.importorskip("opt_core.kernels.transition")
    for w in ("exact", "fast", "big"):
        assert w in T.TIER_WORDS, w
    assert T.cell_word(128, 512, "pair") == "pair_c128_n4" and T.cell_word(64, 256, "rows") == "rows_c64_n4"    # the two shapes the levers own on protenix 1.1.0
    H = "H100:torch2.13.0+cu130/3.7.1"
    for w in ("fast", "big"):
        for c, h, fam in ((128, 512, "pair"), (64, 256, "rows")):
            s = T.select(w, c=c, hidden=h, n_tokens=800, cc="9.0", stack=H, family=fam, rows_count=800 * 800)
            assert s.tier == w and s.row in T.ROW_NAMES, (w, c, s.line())
    s = T.select("exact", c=128, hidden=512, n_tokens=800, cc="9.0", stack=H, family="pair", rows_count=800 * 800, ln_given=True)
    assert s.tier == "exact" and (s.row not in T.STOCK_ROWS)                                # the pair transition's exact tier on the kit's H100 stack is a vouched kernel row
    try:                                                                                     # a shape without a cell is refused BY NAME with the module floor (never substituted)
        T.select("fast", c=128, hidden=256, n_tokens=800, cc="9.0", stack=H, family="pair")
    except T.Refusal as r:
        assert r.kind and r.fallback in T.STOCK_ROWS
    if hasattr(__import__("opt_core.attn.pair_fused", fromlist=["x"]), "exact_rows_word"):   # cc 8.0: the served-row rule refuses the 1,028-token pair rows by name in the exact tier only
        A = "A100:torch2.13.0+cu130/3.7.1"
        with pytest.raises(T.Refusal) as r:
            T.select("exact", c=128, hidden=512, n_tokens=1028, cc="8.0", stack=A, family="pair", rows_count=1028 * 1028, ln_given=True)
        assert r.value.kind == "exact_rows_above_max_rows_on_cc_8.0" and r.value.fallback in T.STOCK_ROWS
        assert T.select("fast", c=128, hidden=512, n_tokens=1028, cc="8.0", stack=A, family="pair", rows_count=1028 * 1028).tier == "fast"


def test_evidence_words_served_gated_refused():
    """report.kit_evidence reads COUNTS['transition']: `<lever>:C=<c>` served; `stock:*` words (dtype / width gate, the provider's refusal by name,
    a stock-row tier, the card's served rows) gated — never partial; `fallback:*` (a provider error) partial. The account's `transition` entry
    carries the tier word (LEVER fact `word=`) and the Selection lines (`selections=`)."""
    from protenix_v1_opt import report as R
    cell = "9.0|bf16|pair_c128_n4|N<=800|eager|fwd"
    levers = {"cfg": {"trimul": "fast", "ttr": True},
              "counts": {"trimul": {"fast": 40}, "transition": {"ttr:C=128": 40, "ttr:C=64": 8, "stock:C=384": 96, "stock:ttr:no_cell:pair_c128_n2": 4}},
              "transition": {"lever": "ttr", "word": "fast", "selections": {"ttr:pair:c128x512": "transition:v2:fast word=fast tier=fast cell=%s" % cell}, "refusals": {"no_cell:pair_c128_n2": 4}}}
    ev = R.kit_evidence(levers, "fast", ("ttr",))
    assert ev["ttr"]["served"] == 48 and ev["ttr"]["fallback"] == {} and ev["ttr"]["gated"] == {"stock:C=384": 96, "stock:ttr:no_cell:pair_c128_n2": 4}
    assert ev["ttr"]["facts"] == ["word=fast"] and ev["ttr"]["selections"] == ["transition:v2:fast word=fast tier=fast cell=%s" % cell]
    assert R.partial_of(ev) == ([], None)
    line = [l for l in R.lever_lines(ev, {"mode": "fast"}) if l.endswith(" lever=ttr")][0]
    assert " served=48 gated=100 " in line and " word=fast " in line and " selections=transition:v2:fast" in line
    levers["counts"]["transition"]["fallback:ttr:error:RuntimeError"] = 1                 # a provider error after engagement: partial, named
    ev = R.kit_evidence(levers, "fast", ("ttr",))
    assert ev["ttr"]["fallback"] == {"fallback:ttr:error:RuntimeError": 1} and R.partial_of(ev)[0] == ["ttr"]
    xl = {"cfg": {"trimul": "exact", "xtr": True}, "counts": {"trimul": {"exact": 40}, "transition": {"stock:above_max_rows": 1084, "stock:xtr:row=torch_swiglu": 8, "stock:C=384": 100}},
          "card_rows": {"xtr": {"cc": "8.0", "min_rows": 4096, "max_rows": 1048544, "below": "stock:below_min_rows", "above": "stock:above_max_rows"}},
          "transition": {"lever": "xtr", "word": "exact", "selections": {}, "refusals": {"exact_rows_above_max_rows_on_cc_8.0": 1084}}}
    ev = R.kit_evidence(xl, "exact", ("xtr",), [{"N_token": 1028}])                        # a 1,028-token input on cc 8.0: every pair call outside the served rows — by design, not partial
    assert ev["xtr"]["served"] == 0 and ev["xtr"]["rows"]["state"] == "range-off" and R.partial_of(ev) == ([], None)
