"""The xtr lever binds the shared core's transition provider (opt_core.kernels.transition) by its EXACT tier word: per call shape the provider
names the row it has vouched bitwise on this stack at this size (served) or the statements BY NAME (routed, counted); no exactness table in
the kit.  CPU tests against the real provider table with the kit's two stack words."""
import pytest

opt_core = pytest.importorskip("opt_core")
from opt_core.kernels import transition as TR          # noqa: E402
from rosettafold3_opt import pf, report                # noqa: E402

H100 = "H100:torch2.13.0+cu130/3.7.1"
A100 = "A100:torch2.13.0+cu130/3.7.1"


def setup_function(_):
    pf._DECISIONS.clear(); pf.CENSUS.clear()
    pf.STATE.update({"on": False, "serve": (), "routes": {}, "rows": {}, "reason": None})


def test_the_lever_asks_the_exact_tier_word_and_serves_the_vouched_row():
    """128-wide pair transitions: v1 (bitwise, the module's LayerNorm given) from 257 tokens on 9.0 and from 64 tokens on 8.0; below the vouch,
    the 384-wide single transition and the c=64 tracks are the statements BY NAME (the provider's word on the census route)."""
    assert pf.WORD == "exact" and pf.PROVIDER == "opt_core.kernels.transition"
    assert pf.decide(TR, 128, 512, 400, 400 * 400, "9.0", H100) == (True, "served:128x512", "v1")
    assert pf.decide(TR, 128, 512, 1200, 1200 * 1200, "9.0", H100) == (True, "served:128x512", "v1")
    assert pf.decide(TR, 128, 256, 400, 400 * 400, "9.0", H100) == (True, "served:128x256", "v1")
    serve, word, row = pf.decide(TR, 128, 512, 199, 199 * 199, "9.0", H100)            # below the 9.0 vouch: the cell's exact winner is the statement
    assert (serve, row) == (False, "torch_swiglu") and word == "route:128x512:stock:torch_swiglu"
    assert pf.decide(TR, 128, 512, 199, 199 * 199, "8.0", A100) == (True, "served:128x512", "v1")   # 8.0: vouched from 64 tokens up
    serve, word, row = pf.decide(TR, 128, 512, 20, 400, "8.0", A100)
    assert not serve and word.startswith("route:128x512:exact_vouch_below_64_tokens")
    assert pf.decide(TR, 384, 1536, 400, 400, "9.0", H100, "single")[1].startswith("route:384x1536:stock:")   # the single track (family single): the statements by its cell
    assert pf.family_of(TR, (400, 400, 128), 128, 512) == "pair" and pf.family_of(TR, (1, 400, 400, 128), 128, 256) == "pair"      # the pair track, square token dims
    assert pf.family_of(TR, (1, 400, 384), 384, 1536) == "single" and pf.family_of(TR, (400, 384), 384, 768) == "single" and pf.family_of(TR, (5, 400, 384), 384, 768) == "single"
    assert pf.family_of(TR, (16, 400, 64), 64, 256) == "pair" and pf.family_of(TR, (2, 400, 400, 64), 64, 256) == "pair"          # MSA rows / template pair stack: the provider's rows family via pair
    assert pf.family_of(TR, (3, 400, 128), 128, 512) == "pair"                                                                        # a non-square 128-wide input: no single cell for it -> pair
    assert pf.decide(TR, 64, 256, 400, 400 * 400, "9.0", H100)[:2] == (False, "route:64x256:stock:torch_swiglu")
    serve, word, _ = pf.decide(TR, 384, 768, 400, 400, "9.0", H100, "single")             # the single track's n=2 transition: its cell names the statements (no UNCOVERED token)
    assert not serve and word.startswith("route:384x768:stock:")
    serve, word, _ = pf.decide(TR, 384, 768, 400, 400 * 400, "9.0", H100, "pair")         # a (width, family) without a cell: refused by name -> the statements
    assert not serve and word.startswith("route:384x768:no_cell")


def test_no_exactness_table_in_the_kit():
    for name in ("CARD_ROWS", "row_floor", "stock_keys", "serve_set", "launch_decision", "LAUNCH_VARIANTS", "floor_route"):
        assert not hasattr(pf, name), name


def test_census_problems_and_the_named_routed_state():
    pf.STATE.update({"on": True, "serve": ("128x512",), "routes": {}, "rows": {"128x512": "v1"}})
    pf.CENSUS.update({"served:128x512": 300, "route:384x1536:stock:torch_swiglu": 232, "route:128x512:stock:torch_swiglu": 4})
    c = pf.census()
    assert (c["calls"], c["served"], c["routed"], c["fallback"]) == (536, 300, 236, 0) and pf.problems() == [] and not pf.routed_by_name()
    d = pf.describe()
    assert d["ok"] and d["construction"] == "kernels.transition(word=exact,x_ln=given)" and d["rows"] == {"128x512": "v1"} and "below_floor" not in d
    assert ("word", "exact") in pf.lever_evidence() and ("rows", "128x512:v1") in pf.lever_evidence()
    pf.CENSUS.clear(); pf.CENSUS.update({"route:128x512:stock:torch_swiglu": 40, "route:384x1536:stock:torch_swiglu": 20})   # a small input: every call routed by the provider's word
    assert pf.problems() == [] and pf.routed_by_name() and pf.describe()["below_floor"] is True and pf.describe()["ok"]
    pf.CENSUS.update({"fallback:128x512:v1:some_refusal": 1})                              # a refusal at launch of a selected row fails the run by name
    assert pf.problems() and "fallback:128x512:v1:some_refusal=1" in pf.problems()[0] and not pf.describe()["ok"]
    pf.CENSUS.clear()
    assert pf.problems() == ["no Transition call reached the xtr lever in this process (installed but never ran)"]


def test_a_cuda_less_helper_process_is_a_named_no_op(monkeypatch):
    import types, sys
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    d = pf.enable()
    assert d["on"] is False and d["reason"] == pf.CPU_PROCESS and pf.problems() == []


def test_ttr_keys_leave_the_addons_widths_to_it_by_name(monkeypatch):
    import sys, types
    adp = types.ModuleType(pf.TTR_ADAPTER); adp.CFG2 = {"transition": "triton"}
    monkeypatch.setitem(sys.modules, pf.TTR_ADAPTER, adp)
    def prev(self, X): return X
    prev.__module__ = pf.TTR_ADAPTER
    assert pf.ttr_keys(prev) == ("128x512", "128x256", "64x256", "64x128")
    adp.CFG2["transition"] = "stock"
    assert pf.ttr_keys(prev) == ()
    def other(self, X): return X
    assert pf.ttr_keys(other) == ()


def test_the_exit_tallys_transition_size_words():
    """The tally reads a transition decline below the provider's size floor (fallback:N<=<n>:<word>, or the older row-floor word fallback:M<<n>) and
    an unserved channel class (fallback:C=<c>,HID=<h>[...]) as size words: a silent ttr on a small input is size-gated, not silent."""
    assert report.ttr_size_decline("fallback:M<4096") and report.ttr_size_decline("fallback:N<=20:exact_vouch_below_64_tokens_on_A100")
    assert report.ttr_size_decline("fallback:C=384,HID=1536") and not report.ttr_size_decline("fallback:cpu/no-autocast")
