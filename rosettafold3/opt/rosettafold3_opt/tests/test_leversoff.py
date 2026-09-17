"""MODEL_OPT_LEVERS_OFF (leversoff.py): named levers leave the mode at the one resolver, refusals by name, the
withheld= token on the ACTIVE / EXIT lines and reason=withheld on the LEVER lines (CPU: no torch)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rosettafold3_opt import big, leversoff, modes, registry, report  # noqa: E402


@pytest.fixture(autouse=True)
def _no_request(monkeypatch):
    monkeypatch.delenv(leversoff.ENV, raising=False)
    yield


def resolve(mode, req, monkeypatch):
    monkeypatch.setenv(leversoff.ENV, req)
    return modes.resolve(mode)


def test_unset_or_blank_is_the_mode_byte_for_byte(monkeypatch):
    base = modes.resolve("fast")
    assert base.withheld == [] and leversoff.requested({}) == []
    monkeypatch.setenv(leversoff.ENV, " , ")
    again = modes.resolve("fast")
    assert again == base and leversoff.token(again.withheld) == ""


def test_requested_folds_duplicates_and_whitespace():
    assert leversoff.requested({leversoff.ENV: " fpf_ttr, mkdit ,fpf_ttr"}) == ["fpf_ttr", "mkdit"]


def test_an_fpf_component_leaves_the_arm_string(monkeypatch):
    r = resolve("fast", "fpf_ttr", monkeypatch)
    assert r.fpf_arm == "fast.fast+gflash+apb.fast+res+tg+xmul.eager+xln+msa@L1.warm" and "fpf_ttr" not in r.levers and "fpf_ttr" in r.levers_off and r.withheld == ["fpf_ttr"]
    r = resolve("fast", "fpf_trimul,fpf_ttr,fpf_res", monkeypatch)
    assert r.fpf_arm == "gflash+apb.fast+tg+xmul.eager+xln+msa@L1.warm" and r.fpf_components == ["gflash", "apb.fast", "tg", "xmul.eager", "xln", "msa"]          # `fast` trimul leaves the body: the adapter's stock trimul
    with pytest.raises(leversoff.WithholdError, match="fpf_tg stays on and needs one of fpf_sapb,fpf_apb"):
        resolve("fast", "fpf_trimul,fpf_gflash,fpf_ttr,fpf_apb,fpf_res", monkeypatch)              # the trunk graph needs a pair-bias cache: name fpf_tg with fpf_apb
    r = resolve("fast", "fpf_trimul,fpf_gflash,fpf_ttr,fpf_apb,fpf_res,fpf_tg,fpf_xmul,fpf_xln,fpf_msa,warm", monkeypatch)
    assert r.fpf_arm is None and r.levers == ["graph", "graph_safe_ops", "hoist", "mkdit", "confhoist", "confln", "hostlean", "prefetch", "awrite"]   # no FPF lever left and no seam lever: an arm-less row


def test_a_package_lever_leaves_kit_levers(monkeypatch):
    r = resolve("exact", "xtr", monkeypatch)
    assert r.kit_levers == ["confhoist", "hostlean", "prefetch", "awrite"] and r.fpf_arm == "tg+sapb+xatt+xmul.eager+xln+smsa@L1.warm" and r.withheld == ["xtr"]


def test_a_switch_lever_is_exported_zero_and_the_arms_lever_step_leaves_with_it(monkeypatch):
    with pytest.raises(leversoff.WithholdError, match="warm: rides the FPF arm's lever step, which leaves with hoist"):
        resolve("fast", "hoist,mkdit", monkeypatch)                                          # the step's sub-step lever is never implied off: name it
    r = resolve("fast", "hoist,warm,mkdit", monkeypatch)
    assert r.switches == {"RF3_CUDAGRAPH": "1", "RF3_HOIST": "0"} and r.fpf_arm == "fast.fast+gflash+ttr+apb.fast+res+tg+xmul.eager+xln+msa" and r.fpf_lever_state is None and "warm" in r.withheld
    assert "hoist" in r.levers_off and "mkdit" not in r.levers
    r = resolve("exact", "graph,graph_safe_ops,warm", monkeypatch)
    assert r.switches["RF3_CUDAGRAPH"] == "0" and r.fpf_arm == "tg+sapb+xatt+xmul.eager+xln+smsa"


def test_the_seam_lever_keeps_a_stock_arm(monkeypatch):
    r = resolve("big", "fpf_trimul,fpf_gflash,fpf_ttr,fpf_apb,fpf_res,fpf_xmul,fpf_xln,fpf_msa", monkeypatch)   # every FPF component of the row withheld (a sub-worded one, msa.pwa / fast.fast, leaves with its lever)
    assert r.fpf_arm == "stock" and r.kit_levers == ["dtk", "confhoist", "confln", "hostlean", "prefetch", "awrite"]                              # dtk installs through the adapter's dattn seam: the adapter stays imported, no kernel


def test_a_big_memory_lever_becomes_the_lines_own_off_switch(monkeypatch):
    r = resolve("big", "big_triatt_chunk,fpf_res", monkeypatch)
    assert r.withheld == ["big_triatt_chunk", "fpf_res"] and r.fpf_arm == "fast.big+gflash+ttr+apb.big+xmul.eager+xln+msa.pwa"
    assert leversoff.big_switches() == {"triatt_chunk": False}
    assert big.switches_for(1)["triatt_chunk"] is False


@pytest.mark.parametrize("mode,req,words", [
    ("fast", "bogus", ["bogus: not a lever of this kit"]),
    ("exact", "mkdit", ["mkdit: not in mode exact's lever set"]),
    ("fast", "big_triatt_chunk", ["not in mode fast's lever set"]),
    ("fast", "mem", ["mem: the memory policy rides every kit row"]),
    ("big", "rowpair", ["rowpair: the row-sharded pair stack IS the --n_gpu line"]),
    ("fast", "hoist", ["hoist: required by mkdit, which stays on (withhold mkdit with it)"]),
    ("big", "hoist", ["required by big_atom_pair_local"]),
    ("exact", "graph", ["withhold graph,graph_safe_ops together"]),
    ("exact", "graph_safe_ops", ["withhold graph,graph_safe_ops together"]),
    ("exact", "fpf_sapb", ["fpf_tg stays on and needs one of fpf_sapb,fpf_apb"]),
    ("fast", "fpf_trimul,fpf_ttr", ["fpf_res stays on and needs one of fpf_trimul,fpf_ttr"]),
    ("exact", "graph,graph_safe_ops,hoist,fpf_sapb,fpf_tg,fpf_xatt,fpf_xmul,fpf_xln,fpf_smsa,warm,xtr,confhoist,hostlean,prefetch,awrite", ["removes every lever of mode exact: that is --mode off"]),
    ("off", "fpf_ttr", ["--mode off applies no lever"]),
])
def test_refusals_are_by_name(monkeypatch, mode, req, words):
    monkeypatch.setenv(leversoff.ENV, req)
    with pytest.raises(ValueError) as ei:
        modes.resolve(mode)
    assert isinstance(ei.value, leversoff.WithholdError)
    for w in words:
        assert w in str(ei.value), str(ei.value)


def test_every_registry_lever_of_every_mode_is_withholdable_alone_or_named_why(monkeypatch):
    """Total accounting: for each kit mode, each lever of its set either leaves the row alone or is refused with a sentence naming it."""
    for mode in modes.KIT_MODES:
        monkeypatch.delenv(leversoff.ENV, raising=False)
        base = modes.resolve(mode)
        for name in leversoff.mode_set(base):
            monkeypatch.setenv(leversoff.ENV, name)
            try:
                r = modes.resolve(mode)
                assert name in r.withheld and name not in r.levers
            except leversoff.WithholdError as e:
                assert name in str(e)


def test_active_lever_and_exit_lines_carry_the_words(monkeypatch):
    r = resolve("fast", "fpf_ttr,mkdit", monkeypatch)
    rep = {"active": True, "mode": r.mode, "row": r.row, "switch_line": "RF3_CUDAGRAPH=1,RF3_HOIST=1", "levers": r.levers, "levers_off": r.levers_off,
           "withheld": r.withheld, "fpf": {"arm": r.fpf_arm}, "applied": "deferred", "gpu": None, "upstream": {}, "n_gpu": 1}
    line = report.active_line(rep)
    assert " withheld=fpf_ttr,mkdit" in line and "fpf=fast.fast+gflash+apb.fast+res+tg+xmul.eager+xln+msa@L1.warm" in line
    t = report.tally(rep)
    assert t["withheld"] == ["fpf_ttr", "mkdit"] and " withheld=fpf_ttr,mkdit" in report.tally_line(t)
    states = report.lever_states(rep, t)
    assert states["fpf_ttr"] == {"state": "off", "reason": "withheld"} or (states["fpf_ttr"]["state"], states["fpf_ttr"]["reason"]) == ("off", "withheld")
    assert (states["mkdit"]["state"], states["mkdit"]["reason"]) == ("off", "withheld")
    assert states["fpf_gflash"]["reason"] != "withheld"


def test_a_refused_request_is_the_not_active_report_never_a_traceback(monkeypatch):
    """stack.activate (check / the drop-in hook) returns the mode's own report with the refusal as its reason (cmd_check prints the line and
    exits 3); fold.run raises NotActive after printing the NOT ACTIVE line (cmd_pred exits 3)."""
    from rosettafold3_opt import stack
    monkeypatch.setenv(leversoff.ENV, "bogus")
    monkeypatch.setitem(stack.STATE, "report", None)
    rep = stack.activate("fast", dry_run=True)
    assert rep.get("reason", "").startswith("MODEL_OPT_LEVERS_OFF refused") and "bogus" in rep["reason"] and rep["row"] == "fast"
    assert rep["withheld"] == [] and "fpf_ttr" in rep["levers"]                        # the mode as it is, named on the line
    assert leversoff.without_request(lambda: os.environ.get(leversoff.ENV)) is None and os.environ[leversoff.ENV] == "bogus"


def test_big_folds_ttr_and_transition_chunk_steps_aside_by_name(monkeypatch):
    """The composed big arm keeps fast's ttr; transition_chunk is off BY PROPERTY (big.ARM_SERVES → site_owned:fpf_ttr) and comes back
    when a run withholds fpf_ttr."""
    monkeypatch.delenv(leversoff.ENV, raising=False)
    r = modes.resolve("big")
    assert r.fpf_arm == "fast.big+gflash+ttr+apb.big+res+xmul.eager+xln+msa.pwa" and "ttr" in r.fpf_components and "gflash" in r.fpf_components
    assert big.site_owned(1) == {"transition_chunk": big.ARM_SERVES["transition_chunk"][1]} and big.switches_for(1) == {"transition_chunk": False}
    assert "transition_chunk" not in big.expected_levers(None, 1)
    monkeypatch.setenv(leversoff.ENV, "fpf_ttr")
    r = modes.resolve("big")
    assert r.fpf_arm == "fast.big+gflash+apb.big+res+xmul.eager+xln+msa.pwa" and big.site_owned(1) == {} and "transition_chunk" in big.expected_levers(None, 1)
    monkeypatch.setenv(leversoff.ENV, "fpf_gflash")                                   # without the fused triangle attention: the row-block statement at every size
    assert modes.resolve("big").fpf_arm == "fast.big+ttr+apb.big+res+xmul.eager+xln+msa.pwa" and "triatt_chunk" in big.expected_levers(None, 1)


def test_compile_is_an_accepted_absent_word_reported_none(monkeypatch):
    """``compile`` (run.sh --no-compile = MODEL_OPT_LEVERS_OFF=compile) names a lever class this kit does not have: accepted in every
    mode (off too), it withholds nothing, rides the resolution as ``absent`` and the lines say ``compile=none``; a real lever beside it is
    withheld as before; an unknown name beside it is still refused by name."""
    base = modes.resolve("fast")
    r = resolve("fast", "compile", monkeypatch)
    assert r.absent == ["compile"] and r.withheld == [] and r.levers == base.levers and r.fpf_arm == base.fpf_arm and r.switches == base.switches
    assert leversoff.requested() == [] and leversoff.requested_absent() == ["compile"] and leversoff.absent_token(r.absent) == " compile=none"
    r = resolve("fast", "compile,mkdit", monkeypatch)
    assert r.absent == ["compile"] and r.withheld == ["mkdit"] and "mkdit" not in r.kit_levers
    r = resolve("off", "compile", monkeypatch)
    assert r.absent == ["compile"] and r.withheld == [] and r.levers == []                      # the stock route: nothing to withhold, the word is still accepted
    with pytest.raises(leversoff.WithholdError, match="torch_compile: not a lever of this kit"):
        resolve("fast", "compile,torch_compile", monkeypatch)
    from rosettafold3_opt import report
    line = report.active_line({"active": True, "mode": "fast", "row": "fast", "absent": ["compile"], "switch_line": "x", "levers": [], "tree_line": "tree=patched(5/5) sp"})
    assert " compile=none" in line and "withheld=" not in line
