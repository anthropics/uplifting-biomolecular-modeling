"""The exit tally: the kit module's own counters, or a line saying it never ran; the JSON copy pred reads for its verdict."""
import json
import sys
import types

import pytest

from .. import report


def test_tally_when_nothing_ran():
    t = report.tally({"mode": "exact", "row": "exact"})
    assert t["graph_flags_imported"] is False
    assert report.tally_line(t) == "[rosettafold3-opt] EXIT mode=exact: rf3.graph_flags never imported in this process (no fold ran here) n_gpu=1 sharding=none"


def test_tally_reads_the_kit_module(monkeypatch, tmp_path):
    gf = types.ModuleType("rf3.graph_flags")
    gf.describe = lambda: {"RF3_CUDAGRAPH": "1", "RF3_HOIST": True}
    gf.HOIST_STATS = {"rollouts": 2, "entries_last": 7, "calls_reused": 398}
    gf.LAST_CAPTURE = {"capture_ms": 812.5, "n_replay": 199}
    monkeypatch.setitem(sys.modules, "rf3.graph_flags", gf)
    t = report.tally({"mode": "exact", "row": "exact"})
    line = report.tally_line(t)
    assert line.startswith("[rosettafold3-opt] EXIT mode=exact rollouts=2 entries_last=7 capture=") and line.endswith(" n_gpu=1 sharding=none")
    assert '"n_replay": 199' in line and '"RF3_HOIST": true' in line
    # the exit hooks: the line through the core's registry (once per process per tag), the JSON copy from this module's own hook
    import atexit
    import io
    from opt_core import report as core_report
    p = tmp_path / "tally.json"
    calls = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a: calls.append((fn, a)))
    core_report._TALLIES.pop(report.TAG, None)
    assert report.register_exit_tally({"mode": "exact", "row": "exact"}, tally_file=str(p)) is True
    assert report.register_exit_tally({"mode": "exact", "row": "exact"}, tally_file=str(p)) is False      # once per process
    assert len(calls) == 2 and calls[0][1][0] == report.TAG and calls[1][1] == ()
    calls[1][0]()                                                                                            # the JSON hook (registered last, runs first)
    assert json.load(open(p))["HOIST_STATS"]["entries_last"] == 7
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    calls[0][0](*calls[0][1])                                                                               # the core prints the whole line
    assert err.getvalue() == line + "\n"
    core_report._TALLIES.pop(report.TAG, None)


def test_tally_carries_the_kernel_route_census_in_json_only():
    rep = {"mode": "fast", "row": "fast", "fpf": {"arm": "fast+gflash@L1", "kernels": {"fpf_trimul_v4": {"resolved": "/core/fpf_trimul_v4"}}, "kernels_imported": {"fpf_trimul_v4": "/core/fpf_trimul_v4"}}}
    t = report.tally(rep)
    assert t["kernels"] == {"routed": {"fpf_trimul_v4": "/core/fpf_trimul_v4"}, "executed_from": {"fpf_trimul_v4": "/core/fpf_trimul_v4"}}
    assert "kernels" not in report.tally_line(t) and "/core/" not in report.tally_line(t)


def test_lines_grammar():
    rep = {"active": True, "mode": "exact", "switch_line": "RF3_CUDAGRAPH=1,RF3_HOIST=1", "tree_line": "tree=patched(5/5) site-packages=/sp",
           "upstream": {"foundry": "0.2.1.dev13+g4010e3e2e", "commit": "4010e3e2e7350edada3e25a45c908c6bf407df4d", "torch": "2.13.0+cu130"},
           "gpu": {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90"}, "levers": ["graph", "graph_safe_ops", "hoist"], "applied": "deferred"}
    assert report.active_line(rep) == ("[rosettafold3-opt] ACTIVE mode=exact row=RF3_CUDAGRAPH=1,RF3_HOIST=1 tree=patched(5/5) "
                                       "foundry=0.2.1.dev13+g4010e3e2e@4010e3e2 torch=2.13.0+cu130 gpu=NVIDIA H100 80GB HBM3(sm90) "
                                       "levers=graph,graph_safe_ops,hoist applied=deferred n_gpu=1 sharding=none")
    assert report.active_line({"active": False, "reason": "x"}) == "[rosettafold3-opt] NOT ACTIVE: x"
    assert report.gpu_label(None) == "none"


def test_fpf_probes_read_the_adapters_own_counters():
    """Each FPF lever's probe against the adapter's record shapes (describe() / describe_v2()); a size-gated dattn call counts as reached."""
    from rosettafold3_opt import registry
    desc = {"mode": "fast", "served": 96, "fallback": {"c=64/d=64": 4}, "errors": 0}
    desc2 = {"cfg": {}, "counts": {"triattn": {"served:gflash": 40, "served:gflash+gatek": 8, "fallback:C=32": 2}, "transition": {"served:triton:C=128": 96, "served:triton:C=64+res": 4},
                                   "apb": {"served:triton": 48, "served:safe": 0}},
             "trunk_graph": {"replays": 7, "captures": []}, "dattn": {"calls": 0, "fallback": 200}, "res": {"fused": 96, "unfused": 4}}
    got = {n: report.fpf_probe(n, lv.probe, desc, desc2) for n, lv in registry.FPF_LEVERS.items()}
    assert got == {"fpf_trimul": 96, "fpf_gflash": 48, "fpf_ttr": 100, "fpf_apb": 48, "fpf_sapb": 0, "fpf_tg": 7, "fpf_res": 96, "fpf_dattn": 200, "fpf_xmul": 0, "fpf_xln": 0, "fpf_xatt": 0, "fpf_msa": 0, "fpf_smsa": 0}, got
    assert report.fpf_probe("fpf_dattn", registry.LEVERS["fpf_dattn"].probe, desc, {"dattn": {"calls": 0, "fallback": 0}}) == 0
    with pytest.raises(ValueError):
        report.fpf_probe("x", ("describe", "k", "v"), desc, desc2)


def test_lever_lines_are_the_cores_pinned_grammar():
    """One LEVER line per registry lever, byte-locked: the core's lever_line order (name, state, [reason], impl, origin), then strategy and
    the lever's evidence; one-token reasons (not_in_mode:<m>, serve_only, no_fold, below_gate, probe_zero:<kind>)."""
    from .. import registry
    t = {"mode": "fast", "graph_flags_imported": True, "describe": {"RF3_CUDAGRAPH": "1", "RF3_GRAPH_SAFE_OPS": True},
         "HOIST_STATS": {"rollouts": 1, "entries_last": 33}, "LAST_CAPTURE": {"capture_ms": 1631.5, "T": 49, "D": 5, "warmup": 3},
         "mem": {"policy": "capped", "releases": 2},
         "kernels": {"routed": {"fpf_trimul_v4": "/core/kernels/fpf_trimul_v4", "flash_triattn": "/core/kernels/flash_triattn.py"}},
         "fpf": {"served": 1088, "fallback": {"c=64/d=64": 40}, "errors": 0, "describe": {"served": 1088, "mode": "fast"},
                 "describe_v2": {"counts": {"triattn": {"served:gflash+gatek": 1128}, "transition": {"served:triton:C=128": 42}, "apb": {"served:triton": 504}},
                                 "res": {"fused": 1532, "unfused": 40}, "trunk_graph": {}, "dattn": {"calls": 0, "fallback": 0}}},
         "dtk": {"on": True, "impl": "/core/kernels/dtk_kernels.py", "origin": "core", "bias": "relayout", "min_tokens": 400,
                 "census": {"calls": 96, "served": 0, "gated": 96, "gated_by": {"lt_min": 96}, "fallback": 0, "fallback_by": {}}, "ok": True, "reason": None}}
    rep = {"levers": ["graph", "graph_safe_ops", "hoist", "fpf_trimul", "fpf_gflash", "fpf_ttr", "fpf_apb", "fpf_res", "dtk"]}
    t["levers"] = report.lever_states(rep, t)
    lines = report.lever_lines(t)
    assert len(lines) == len(registry.LEVERS) and all(l.startswith("[rosettafold3-opt] LEVER name=") for l in lines)
    by = {l.split("name=")[1].split()[0]: l for l in lines}
    assert by["graph"] == "[rosettafold3-opt] LEVER name=graph state=on impl=patched/rf3/diffusion_samplers/inference_sampler.py origin=kit strategy=F3.cuda_graph_sampler capture_ms=1631.5 steps=49 samples=5 warmup=3"
    assert by["fpf_tg"] == "[rosettafold3-opt] LEVER name=fpf_tg state=off reason=not_in_mode:fast impl=rf3fpf/fpf_rf3_adapter.py origin=kit strategy=F3.cuda_graph_trunk"
    assert by["fpf_trimul"] == "[rosettafold3-opt] LEVER name=fpf_trimul state=on impl=opt_core.kernels.fpf_trimul_v4 origin=core strategy=F2.fpf_trimul_fast executes_from=/core/kernels/fpf_trimul_v4 served=1088 fallback=c64/d64:40 errors=0"
    assert by["dtk"] == "[rosettafold3-opt] LEVER name=dtk state=skipped reason=below_gate impl=opt_core.kernels.dtk_kernels origin=core strategy=F5.flash_attn_dense executes_from=/core/kernels/dtk_kernels.py bias=relayout served=0 fallback=0 min_tokens=400 gated=96 calls=96"
    assert by["mem"] == "[rosettafold3-opt] LEVER name=mem state=on impl=opt_core.mem.torch_alloc+mem.py origin=core strategy=F7.expandable_segments policy=capped releases=2"
    t2 = dict(t, graph_flags_imported=False); t2["levers"] = report.lever_states(rep, t2)
    assert {v["reason"] for k, v in t2["levers"].items() if k in rep["levers"]} == {"no_fold"}


def _fake_adapter(monkeypatch, replays):
    import types
    adp = types.ModuleType("fpf_rf3_adapter")
    adp.describe = lambda: {"mode": "stock", "served": 0, "fallback": {}, "errors": 0}
    adp.describe_v2 = lambda: {"cfg": {"apb": "safe", "levers": True}, "counts": {"apb": {"served:safe": 96}}, "trunk_graph": {"replays": replays, "fallbacks": {}},
                               "dattn": {"calls": 0, "fallback": 0}, "res": {"fused": 0, "unfused": 0}}
    monkeypatch.setitem(sys.modules, "fpf_rf3_adapter", adp)


def _budget(monkeypatch, graphed, skipped):
    from rosettafold3_opt import tgbudget as tgb
    monkeypatch.setattr(tgb, "STATE", {"on": True, "max_i": 1000, "reason": None})
    c = __import__("collections").Counter({**({"graphed:1000": graphed} if graphed else {}), **({"skipped:1400": skipped} if skipped else {})})
    monkeypatch.setattr(tgb, "CENSUS", c)


REP_EXACT = {"mode": "exact", "row": "exact", "levers": ["graph", "graph_safe_ops", "hoist", "fpf_tg", "fpf_sapb", "xtr"], "fpf": {"arm": "tg+sapb@L1", "applied": "configured"}}


def test_a_trunk_graph_skipped_by_the_token_budget_is_a_named_route_not_a_silent_lever(monkeypatch):
    """exact at I > tgbudget.MAX_I: every Recycler call took the pre-graph loop by the budget rule, so the graph never replayed —
    reached and routed by name (levers_budget_skipped), the tally ok; the EXIT line carries the field with max_i and the sizes."""
    _fake_adapter(monkeypatch, replays=0); _budget(monkeypatch, graphed=0, skipped=10)
    ft = report.fpf_tally(REP_EXACT)
    assert ft["ok"] is True and ft["levers_silent"] == [] and ft["levers_budget_skipped"] == ["fpf_tg"] and ft["levers_acted"] == ["fpf_sapb"]
    assert ft["tg_budget_skipped"] == {"max_i": 1000, "by_key": {"skipped:1400": 10}}
    line = report.fpf_tally_line(ft)
    assert " ok=True" in line and "levers_budget_skipped=fpf_tg(max_i=1000,skipped:1400=10)" in line and "levers_silent=none" in line


def test_a_trunk_graph_within_budget_must_replay(monkeypatch):
    """I <= max_i: the calls were handed to the graph (graphed > 0) — replays > 0 is acted; a graph that got calls and never replayed is silent."""
    _fake_adapter(monkeypatch, replays=7); _budget(monkeypatch, graphed=10, skipped=0)
    ft = report.fpf_tally(REP_EXACT)
    assert ft["ok"] is True and "fpf_tg" in ft["levers_acted"] and ft["levers_budget_skipped"] == []
    _fake_adapter(monkeypatch, replays=0)
    ft = report.fpf_tally(REP_EXACT)
    assert ft["ok"] is False and ft["levers_silent"] == ["fpf_tg"] and ft["levers_budget_skipped"] == [] and "silent lever(s): fpf_tg" in ft["reason"]


def test_a_silent_trunk_graph_without_a_skip_reason_still_fails(monkeypatch):
    """No budget census at all (the budget not installed / no call reached it): replays == 0 is a silent lever, ok=False."""
    _fake_adapter(monkeypatch, replays=0)
    from rosettafold3_opt import tgbudget as tgb
    monkeypatch.setattr(tgb, "STATE", {"on": False, "max_i": None, "reason": None})
    ft = report.fpf_tally(REP_EXACT)
    assert ft["ok"] is False and ft["levers_silent"] == ["fpf_tg"] and ft.get("levers_budget_skipped") == []


REP_FAST = {"mode": "fast", "row": "fast", "levers": ["graph", "hoist", "fpf_trimul", "fpf_gflash", "fpf_ttr", "fpf_apb", "fpf_res"], "fpf": {"arm": "fast+gflash+ttr+apb+res@L1", "applied": "configured"}}


def _fast_adapter(monkeypatch, served, fallback, errors=0, transition=None, res=None):
    """The adapter's records of a fast fold: the trimul kernel's served / fallback census as given, the transition / res census as given
    (default: acting), every other component acting."""
    adp = types.ModuleType("fpf_rf3_adapter")
    adp.describe = lambda: {"mode": "fast", "served": served, "fallback": dict(fallback), "errors": errors}
    adp.describe_v2 = lambda: {"cfg": {"apb": "triton", "triattn": "gflash", "transition": "triton", "res": True}, "trunk_graph": {"replays": 0, "fallbacks": {}},
                               "counts": {"apb": {"served:triton": 480}, "triattn": {"served:gflash": 960}, "transition": dict(transition if transition is not None else {"served:triton": 480})},
                               "res": dict(res if res is not None else {"fused": 960, "unfused": 0})}
    monkeypatch.setitem(sys.modules, "fpf_rf3_adapter", adp)
    from rosettafold3_opt import tgbudget as tgb
    monkeypatch.setattr(tgb, "STATE", {"on": False, "max_i": None, "reason": None})


def test_a_trimul_below_its_token_floor_is_a_named_route_not_a_silent_lever(monkeypatch):
    """fast on an input of <= 100 tokens (a20 K17: a 25-model protein-RNA unit): every pair-stack call lay below fpf_trimul_v4's N_MIN, the
    kernel declined each by its own word (`N<101`; the c=64 template track is never served) and the vendor op ran — reached and routed by
    the named rule: levers_size_gated, the tally ok, the EXIT line carries the census, the LEVER line says skipped below_gate."""
    _fast_adapter(monkeypatch, served=0, fallback={"N<101": 1080, "c=64/d=64": 40})
    ft = report.fpf_tally(REP_FAST)
    assert ft["ok"] is True and ft["reason"] is None and ft["levers_silent"] == [] and ft["levers_size_gated"] == ["fpf_trimul"], ft
    assert ft["levers_acted"] == ["fpf_gflash", "fpf_ttr", "fpf_apb", "fpf_res"] and ft["size_gated"] == {"fpf_trimul": {"N<101": 1080, "c=64/d=64": 40}}
    line = report.fpf_tally_line(ft)
    assert " ok=True" in line and "levers_silent=none" in line and "levers_size_gated=fpf_trimul(N<101=1080,c=64/d=64=40)" in line and "reason=" not in line
    t = {"mode": "fast", "graph_flags_imported": True, "fpf": ft, "LAST_CAPTURE": {"capture_ms": 800.0, "n_replay": 199}, "HOIST_STATS": {"rollouts": 1, "entries_last": 7}}
    st = report.lever_states(REP_FAST, t)
    assert (st["fpf_trimul"]["state"], st["fpf_trimul"]["reason"]) == ("skipped", "below_gate") and st["fpf_trimul"]["served"] == 0 and st["fpf_trimul"]["fallback"] == {"N<101": 1080, "c64/d64": 40}
    assert st["fpf_gflash"]["state"] == "on"
    assert report.trimul_size_decline("N<101") and report.trimul_size_decline("N<129") and report.trimul_size_decline("c=64/d=64")
    assert not any(report.trimul_size_decline(k) for k in ("vanilla-path-not-served", "dtype=torch.float16", "no-cell:9.0", "declined", "N=64"))


@pytest.mark.parametrize("served,fallback,errors", [
    (0, {"c=64/d=64": 40}, 0),                                   # only the template track declined: the pair stack never reached the kernel — silent
    (0, {"N<101": 1080, "vanilla-path-not-served": 4}, 0),      # one decline that is the machine's / configuration's, not the input's — silent
    (0, {}, 0),                                                  # nothing counted at all — silent
    (0, {"N<101": 1080, "c=64/d=64": 40}, 2),                    # adapter errors close the gate whatever the declines say
])
def test_a_silent_trimul_without_the_size_reason_still_fails(monkeypatch, served, fallback, errors):
    _fast_adapter(monkeypatch, served=served, fallback=fallback, errors=errors)
    ft = report.fpf_tally(REP_FAST)
    assert ft["ok"] is False and ft["levers_size_gated"] == [] and "size_gated" not in ft
    assert ("silent lever(s): fpf_trimul" in ft["reason"]) or (errors and f"{errors} adapter error(s)" in ft["reason"])
    assert "levers_size_gated" not in report.fpf_tally_line(ft)


def test_a_trimul_that_served_is_acted_whatever_its_small_input_declines(monkeypatch):
    """A fold of several inputs, some below the floor: served > 0 is acted; the declines stay on the census (fallback=) as before."""
    _fast_adapter(monkeypatch, served=960, fallback={"N<101": 120, "c=64/d=64": 40})
    ft = report.fpf_tally(REP_FAST)
    assert ft["ok"] is True and "fpf_trimul" in ft["levers_acted"] and ft["levers_size_gated"] == [] and ft["fallback"] == {"N<101": 120, "c=64/d=64": 40}


def test_an_input_under_every_kernel_floor_names_all_three_size_gates(monkeypatch):
    """fast on an input under 64 tokens: the trimul (N<101), the pair transition (M = I*I < 4096 rows: the adapter's `fallback:M<4096`) and
    with them the residual fusion that rides their epilogues (fused 0, unfused counted) are all size-gated — named in row order, ok."""
    _fast_adapter(monkeypatch, served=0, fallback={"N<101": 1080, "c=64/d=64": 40}, transition={"fallback:M<4096": 528, "fallback:cpu/no-autocast": 2}, res={"fused": 0, "unfused": 1440})
    ft = report.fpf_tally(REP_FAST)
    assert ft["ok"] is False and ft["levers_silent"] == ["fpf_ttr", "fpf_res"], "a transition decline that is the machine's (no autocast) keeps ttr silent, and res with it"
    _fast_adapter(monkeypatch, served=0, fallback={"N<101": 1080, "c=64/d=64": 40}, transition={"fallback:M<4096": 528, "fallback:C=384,HID=1536": 480}, res={"fused": 0, "unfused": 1440})
    ft = report.fpf_tally(REP_FAST)
    assert ft["ok"] is True and ft["levers_silent"] == [] and ft["levers_size_gated"] == ["fpf_trimul", "fpf_ttr", "fpf_res"] and ft["levers_acted"] == ["fpf_gflash", "fpf_apb"]
    assert ft["size_gated"] == {"fpf_trimul": {"N<101": 1080, "c=64/d=64": 40}, "fpf_ttr": {"M<4096": 528, "C=384,HID=1536": 480}, "fpf_res": {"unfused": 1440}}
    assert "levers_size_gated=fpf_trimul(N<101=1080,c=64/d=64=40);fpf_ttr(C=384,HID=1536=480,M<4096=528);fpf_res(unfused=1440)" in report.fpf_tally_line(ft)
    t = {"mode": "fast", "graph_flags_imported": True, "fpf": ft, "LAST_CAPTURE": {"capture_ms": 800.0, "n_replay": 199}, "HOIST_STATS": {"rollouts": 1, "entries_last": 7}}
    st = report.lever_states(REP_FAST, t)
    assert [(st[n]["state"], st[n]["reason"]) for n in ("fpf_trimul", "fpf_ttr", "fpf_res")] == [("skipped", "below_gate")] * 3


def test_res_is_size_gated_when_the_transition_served_only_outside_the_pair_block(monkeypatch):
    """An input of 64..100 tokens with MSA rows enough for the MSA track's transitions: ttr acted there, every pair-block transition and trimul
    lay below their floors, so no residual could fuse — res is size-gated, not silent."""
    _fast_adapter(monkeypatch, served=0, fallback={"N<101": 1080, "c=64/d=64": 40}, transition={"fallback:M<4096": 96, "served:triton": 4}, res={"fused": 0, "unfused": 1440})
    ft = report.fpf_tally(REP_FAST)
    assert ft["ok"] is True and ft["levers_size_gated"] == ["fpf_trimul", "fpf_res"] and "fpf_ttr" in ft["levers_acted"] and ft["levers_silent"] == []


@pytest.mark.parametrize("transition,res,silent", [
    ({"fallback:cpu/no-autocast": 528}, {"fused": 0, "unfused": 1440}, ["fpf_ttr", "fpf_res"]),        # a machine decline of the transition: ttr fails by name, res with it
    ({"fallback:M<4096": 96, "served:triton": 4, "fallback:cpu/no-autocast": 2}, {"fused": 0, "unfused": 1440}, ["fpf_res"]),   # one transition decline that is not the floor: res's silence is not shown to be size
    ({"fallback:M<4096": 528}, {"fused": 0, "unfused": 0}, ["fpf_res"]),                               # the block forward was never reached (unfused 0): res truly silent
])
def test_res_and_ttr_stay_silent_when_the_silence_is_not_their_size_gate(monkeypatch, transition, res, silent):
    _fast_adapter(monkeypatch, served=0, fallback={"N<101": 1080, "c=64/d=64": 40}, transition=transition, res=res)
    ft = report.fpf_tally(REP_FAST)
    assert ft["ok"] is False and ft["levers_silent"] == silent and "fpf_trimul" in ft["levers_size_gated"]


def test_the_adapter_binds_the_transition_provider_by_tier_word_and_declines_by_name():
    """Source contract (the adapter needs rf3 to import): the fused transition is the shared core's provider under the arm's tier word (fast | big);
    a width whose cell names the statements declines with the channel-class word, a size below the provider's floor with a size word; no kit row floor."""
    import os
    from rosettafold3_opt import stack
    src = open(os.path.join(stack.fpf_home(), "rf3fpf", "fpf_rf3_adapter.py")).read()
    assert "TTR_MIN_ROWS" not in src and "lnl_fused as RFU\n        c = _cache(self)\n        if \"tr_w\"" not in src
    assert 'TTR_PROVIDER = "opt_core.kernels.transition"' in src and "TR.transition(X, W, word=_ttr_word()" in src and "TR.select(_ttr_word()" in src
    assert '"fallback:C=%d,HID=%d" % (C, HID)' in src and '"fallback:N<=%d:%s" % (I, kind)' in src and '"served:%s:C=%d" % (sel.row, C)' in src
    assert report.TTR_ROW_FLOOR.fullmatch("fallback:M<4096") and report.TTR_ROW_FLOOR.fullmatch("fallback:N<=49:exact_vouch_below_64_tokens") and not report.TTR_ROW_FLOOR.fullmatch("fallback:C=128,HID=512")


# ---- the size matrix (a20 K17b): the adapter's censuses of single-chain protein folds at 28 / 49 / 71 / 81 tokens, as the fast arm writes them
# (10 recycles: 48 pairformer blocks' pair transitions C=128 -> M = I*I rows, the single track's C=384 transitions on every input, the MSA
# module's C=64 transitions over S*I rows; trimul below N_MIN=101 on all four; the residual fuses only with a served kernel).
def _census(tokens):
    single = {"fallback:C=384,HID=1536": 500, "fallback:C=384,HID=768": 8}                        # every input: the kernel serves C in {64,128} only
    if tokens * tokens < 4096:                                                                     # 28, 49: every pair transition below the row floor; MSA rows too
        transition = {"fallback:M<4096": 602, **single}; res = {"fused": 0, "unfused": 1560}
    else:                                                                                          # 71, 81: pair transitions served (+res), a few MSA rows below the floor
        transition = {"served:triton:C=64+res": 20, "fallback:M<4096": 40, "served:triton:C=128": 62, "served:triton:C=128+res": 480, **single}
        res = {"fused": 500, "unfused": 1060}
    return dict(served=0, fallback={"N<101": 1080, "c=64/d=64": 40}, transition=transition, res=res)


@pytest.mark.parametrize("tokens,size_gated,acted", [
    (28, ["fpf_trimul", "fpf_ttr", "fpf_res"], ["fpf_gflash", "fpf_apb"]),
    (49, ["fpf_trimul", "fpf_ttr", "fpf_res"], ["fpf_gflash", "fpf_apb"]),          # the site's 8its census verbatim: was 'silent lever(s): fpf_ttr,fpf_res'
    (71, ["fpf_trimul"], ["fpf_gflash", "fpf_ttr", "fpf_apb", "fpf_res"]),
    (81, ["fpf_trimul"], ["fpf_gflash", "fpf_ttr", "fpf_apb", "fpf_res"]),          # the site's 8fhx / 8d79 censuses: ok before and after
])
def test_the_size_matrix_every_lever_below_its_gate_is_size_gated_and_the_fold_is_ok(monkeypatch, tokens, size_gated, acted):
    c = _census(tokens)
    _fast_adapter(monkeypatch, served=c["served"], fallback=c["fallback"], transition=c["transition"], res=c["res"])
    ft = report.fpf_tally(REP_FAST)
    assert ft["ok"] is True and ft["reason"] is None and ft["levers_silent"] == [], (tokens, ft.get("reason"))
    assert ft["levers_size_gated"] == size_gated and [n for n in ft["levers_acted"] if n.startswith("fpf_")] == acted
    line = report.fpf_tally_line(ft)
    assert "levers_silent=none ok=True levers_size_gated=fpf_trimul(N<101=1080,c=64/d=64=40)" in line
    if tokens < 64:
        assert ";fpf_ttr(C=384,HID=1536=500,C=384,HID=768=8,M<4096=602);fpf_res(unfused=1560)" in line, line
        t = {"mode": "fast", "graph_flags_imported": True, "fpf": ft, "LAST_CAPTURE": {"capture_ms": 800.0, "n_replay": 199}, "HOIST_STATS": {"rollouts": 1, "entries_last": 7}}
        st = report.lever_states(REP_FAST, t)
        assert [(st[n]["state"], st[n]["reason"]) for n in ("fpf_trimul", "fpf_ttr", "fpf_res")] == [("skipped", "below_gate")] * 3 and st["fpf_apb"]["state"] == "on"


@pytest.mark.parametrize("transition,res,silent", [
    ({"fallback:C=384,HID=1536": 500, "fallback:C=384,HID=768": 8}, {"fused": 0, "unfused": 1560}, ["fpf_ttr", "fpf_res"]),                 # no row-floor word at all: the pair transitions never met the kernel -> true silence above the gate
    ({"fallback:M<4096": 602, "fallback:C=384,HID=1536": 500, "fallback:cpu/no-autocast": 3}, {"fused": 0, "unfused": 1560}, ["fpf_ttr", "fpf_res"]),   # one machine decline: fails by name
    ({}, {"fused": 0, "unfused": 0}, ["fpf_ttr", "fpf_res"]),                                                                              # the transition wrapper never ran
])
def test_above_the_gate_or_by_the_machine_the_transition_stays_silent(monkeypatch, transition, res, silent):
    _fast_adapter(monkeypatch, served=0, fallback={"N<101": 1080, "c=64/d=64": 40}, transition=transition, res=res)
    ft = report.fpf_tally(REP_FAST)
    assert ft["ok"] is False and ft["levers_silent"] == silent and ft["levers_size_gated"] == ["fpf_trimul"]
    assert report.ttr_size_decline("fallback:C=384,HID=1536") and report.ttr_size_decline("fallback:M<4096") and not report.ttr_size_decline("fallback:cpu/no-autocast")
