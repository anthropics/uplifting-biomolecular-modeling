"""The attention-pair-bias head sub-word of the FPF add-on (``rf3fpf/fpf_rf3_apb_rows.py``, the shared core's provider ``opt_core.kernels.apb``):
``apb.<word>`` leaves today's arms byte for byte (``apb`` = the statement core after the fused producer); a word outside the vocabulary is refused
by name; a row the provider refuses is DECLINED BY NAME to the statement, printed and counted; the SELECT / CENSUS lines carry the provider's
``describe()``.  CPU only: selections over synthetic facts."""
import os
import re
import sys

import pytest

from rosettafold3_opt import modes, stack

A = pytest.importorskip("opt_core.kernels.apb")
RF3FPF = os.path.join(stack.fpf_home(), "rf3fpf")


@pytest.fixture
def R(monkeypatch):
    monkeypatch.syspath_prepend(RF3FPF)
    sys.modules.pop("fpf_rf3_apb_rows", None)
    import fpf_rf3_apb_rows as R
    R.reset_census()
    R.STATE["apb"]["word"] = None; R.STATE["apb"]["on"] = False
    R._PROC["stack"] = None
    yield R
    R.reset_census()
    sys.modules.pop("fpf_rf3_apb_rows", None)


def test_todays_arms_are_untouched_by_the_sub_word_grammar(R):
    for arm in ("fast+gflash+ttr+apb+res@L1", "tg+sapb@L1", "fast+apb+res@L1", "fast", "stock+tg@L1", "fast+gflash+ttr+apb+res+tg@L1.warm"):
        assert R.strip_components(arm) == (arm, {"apb": None}), arm
    R.set_apb_word(None)
    assert R.STATE["apb"]["word"] == "stmt" and R.core_word() is None                            # default: the statement (the adapter's code path unchanged)


def test_sub_words_parse_and_unknown_words_are_refused_by_name(R):
    assert R.strip_components("fast+gflash+ttr+apb.fpf_apb+res@L1") == ("fast+gflash+ttr+apb+res@L1", {"apb": "fpf_apb"})
    assert R.strip_components("fast+apb.sdpa+res+tg@L1.warm") == ("fast+apb+res+tg@L1.warm", {"apb": "sdpa"})
    assert R.strip_components("fast+apb.fast@L1") == ("fast+apb@L1", {"apb": "fast"})
    assert R.parse_component("apb") == ("apb", None) and R.parse_component("apb.apb_attn") == ("apb", "apb_attn") and R.parse_component("sapb") == (None, None)
    for bad in ("apb.turbo", "apb.ds4sci", "apb.dit_exact", "apb.naive", "apb.exact", "apb."):
        with pytest.raises(ValueError):
            R.strip_components("fast+" + bad + "@L1")
    with pytest.raises(ValueError):
        R.set_apb_word("l3a")
    assert R.COMPONENT_WORDS["apb"][0] == "stmt"


def test_the_adapter_parses_the_sub_word_and_binds_the_provider():
    src = open(os.path.join(stack.fpf_home(), modes.FPF_ADAPTER_RELPATH), encoding="utf-8").read()
    assert "APBR.core_word()" in src and "APBR.serve_core(" in src and "APBR.strip_components" in src and re.search(r"served:triton\+", src)


def test_row_words_select_that_row_and_tier_words_the_cells_winner(R, capsys):
    R.set_apb_word("fpf_apb")
    sel, r, cell = R._select_one(A, (9, 0), "bf16", 16, 24, 800, 1, "fpf_apb", False)
    assert cell == "pf_h16d24" and sel.row == "fpf_apb" and r is None, A.describe(sel)
    R.set_apb_word("fast")
    for cc in ((9, 0), (8, 0)):
        for cap in (False, True):
            sel, r, cell = R._select_one(A, cc, "bf16", 16, 24, 1200, 1, "fast", cap)
            assert r is None and sel.row in A.ROW_NAMES + tuple(getattr(A, "STOCK_ROWS", ())) and sel.row != "ds4sci", (cc, cap, A.describe(sel))
            if cap:
                assert sel.capture_safe, A.describe(sel)


def test_a_refusing_row_is_declined_by_name_to_the_statement(R, capsys):
    R.set_apb_word("apb_attn")
    sel, r, cell = R._select_one(A, (9, 0), "fp32", 16, 24, 400, 1, "apb_attn", False)          # apb_attn is a bf16/fp16 row: fp32 operands refused by name
    ent = {"sel": sel if r is None else None, "refusal": r, "call_refusal": None, "calls": 0, "cell": cell, "key": ("apb_attn", (9, 0), "fp32", 16, 24, 400, 1, False)}
    assert r is not None and r.row == "apb_attn" and "dtype" in r.kind
    R._SEL[ent["key"]] = ent
    R._say_select(ent)
    out = capsys.readouterr().out
    assert "[fpf_rf3 apb] SELECT component=apb asked=apb_attn cc=9.0 dtype=fp32 h=16 d=24 n=400 s=1 capture=False cell=pf_h16d24 -> stmt" in out
    assert "REFUSED row=apb_attn kind=%s -> stmt by name" % r.kind in out
    R._count(ent, "fp32", 16, 24, False)
    line = R.census_line("apb")
    assert line.startswith("[fpf_rf3 apb] CENSUS component=apb word=apb_attn on=False served=0 rows=- declined=refused:apb_attn:%s:1" % r.kind)


def test_geometry_without_a_cell_gets_the_row_word_anyway_and_tier_words_by_name(R):
    R.set_apb_word("fpf_apb")
    sel, r, cell = R._select_one(A, (9, 0), "bf16", 4, 16, 400, 1, "fpf_apb", False)            # the template track's would-be geometry (c_s=0 there: no APB module) — no cell, row word still that row or a named refusal
    assert cell is None
    assert (r is None and sel.row == "fpf_apb") or (r is not None and r.row == "fpf_apb")


def test_the_statements_broadcasting_is_planned_not_assumed(R):
    """RF3's einsum core broadcasts Q/K/V's leading dims against the bias's: the confidence head hands an UNBATCHED single [I, H, D] and a
    per-sample pair [S, I, J, H] and gets a per-sample output (guards against dropping the sample dim).  plan_shapes returns
    (lead, S, Sb, I, J, D) or None (declined by name) exactly as torch would broadcast."""
    assert R.plan_shapes((400, 16, 24), (400, 400, 16), 16) == ((), 1, 1, 400, 400, 24)                       # pairformer: both unbatched
    assert R.plan_shapes((400, 16, 24), (5, 400, 400, 16), 16) == ((5,), 5, 5, 400, 400, 24)                # confidence head: single shared, pair per sample
    assert R.plan_shapes((400, 16, 24), (1, 400, 400, 16), 16) == ((1,), 1, 1, 400, 400, 24)
    assert R.plan_shapes((5, 400, 16, 24), (1, 400, 400, 16), 16) == ((5,), 5, 1, 400, 400, 24)             # batched queries, shared planes
    assert R.plan_shapes((5, 400, 16, 24), (5, 400, 400, 16), 16) == ((5,), 5, 5, 400, 400, 24)
    assert R.plan_shapes((1, 5, 200, 16, 24), (5, 200, 200, 16), 16) == ((1, 5), 5, 5, 200, 200, 24)
    assert R.plan_shapes((5, 400, 16, 24), (3, 400, 400, 16), 16) is None                                    # not broadcastable: declined by name
    assert R.plan_shapes((2, 1, 400, 16, 24), (1, 3, 400, 400, 16), 16) is None                              # bias batched where q broadcasts AND shared where q is batched: no [1|S] plane set

