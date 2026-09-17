"""The upstream-outcome census (upstream.py) and the exit verdict's `no_call_reached:<zero_items|early_stop>` rule: a completed run in which
upstream predicted nothing (skip_existing) or early-stopped every input (early_stopping_plddt_threshold) exits with stock's code, its levers
named unreached; a lever withheld by MODEL_OPT_LEVERS_OFF is off by design; a run where an input DID roll out keeps failing closed on a
silent lever. CPU only: fake trainer classes, a fake adapter record."""
import sys
import types

import pytest

from rosettafold3_opt import fold, report, upstream


class _FakeTrainer:
    def validation_step(self, batch=None, batch_idx=0, compute_metrics=True, should_early_stop_fn=None):
        stop = bool(should_early_stop_fn and should_early_stop_fn())
        return {"network_output": {"early_stopped": stop, "mean_plddt": 0.3} if stop else {"xyz": 1}, "metrics_output": {}}


def _fresh_trainers(monkeypatch):
    mod = types.ModuleType("rf3.trainers.rf3")

    class RF3Trainer(_FakeTrainer):
        def validation_step(self, *a, **kw):
            return _FakeTrainer.validation_step(self, *a, **kw)

    class RF3TrainerWithConfidence(RF3Trainer):
        def validation_step(self, *a, **kw):                # a subclass step calling its base: ONE item
            return RF3Trainer.validation_step(self, *a, **kw)

    mod.RF3Trainer, mod.RF3TrainerWithConfidence = RF3Trainer, RF3TrainerWithConfidence
    monkeypatch.setattr(upstream, "STATE", dict(upstream.STATE, armed=False, installed=False, items=0, early_stopped=0, classes=[]))
    return mod


def test_the_census_counts_items_and_early_stops_and_returns_upstreams_result_untouched(monkeypatch):
    mod = _fresh_trainers(monkeypatch)
    d = upstream.enable(trainers=mod)
    assert d["installed"] and upstream.no_rollout_reason() == "zero_items"                     # nothing predicted yet: `Found 0 structures to predict!`
    t = mod.RF3TrainerWithConfidence()
    out = t.validation_step(batch={}, should_early_stop_fn=lambda: True)                        # early-stopped after the recycle-1 probe
    assert out["network_output"]["early_stopped"] is True and upstream.describe()["items"] == 1 and upstream.describe()["early_stopped"] == 1
    assert upstream.no_rollout_reason() == "early_stop"
    t.validation_step(batch={})                                                                 # a second input rolls out
    assert upstream.describe() == {"installed": True, "items": 2, "early_stopped": 1, "rolled_out": 1, "reason": None}
    assert upstream.no_rollout_reason() is None
    upstream.enable(trainers=mod); upstream.enable(trainers=mod)                                # idempotent: one wrapper per class
    assert mod.RF3Trainer.__dict__["validation_step"].__name__ == upstream.WRAP_NAME and not upstream._already_wrapped(mod.RF3Trainer.__dict__["validation_step"].__wrapped__)
    t.validation_step(batch={})
    assert upstream.describe()["items"] == 3


def test_an_uninstalled_census_names_nothing(monkeypatch):
    monkeypatch.setattr(upstream, "STATE", dict(upstream.STATE, installed=False, items=0, early_stopped=0))
    assert upstream.no_rollout_reason() is None                                                # no census: the verdict stays fail-closed


def _fake_fpf(monkeypatch, served=0):
    adp = types.ModuleType("fpf_rf3_adapter")
    adp.describe = lambda: {"mode": "fast", "served": served, "fallback": {}, "errors": 0}
    adp.describe_v2 = lambda: {"cfg": {"levers": True}, "counts": {}, "trunk_graph": {"replays": 0, "fallbacks": {}}}
    monkeypatch.setitem(sys.modules, "fpf_rf3_adapter", adp)
    from rosettafold3_opt import tgbudget as tgb
    monkeypatch.setattr(tgb, "STATE", {"on": False, "max_i": None, "reason": None})


def _census(monkeypatch, items, early):
    monkeypatch.setattr(upstream, "STATE", dict(upstream.STATE, installed=True, items=items, early_stopped=early))


REP = {"mode": "fast", "levers": ["fpf_trimul", "fpf_gflash", "fpf_ttr"], "fpf": {"arm": "fast.fast+gflash+ttr@L1", "applied": "configured"}}


def test_zero_items_every_fpf_lever_unreached_by_name_and_the_verdict_holds(monkeypatch):
    """`skip_existing=true` into an out_dir that already holds the outputs: 'Found 0 structures to predict!' — no forward, every kernel lever
    reached no call: named `no_call_reached:zero_items`, ok (the run exits with stock's code)."""
    _fake_fpf(monkeypatch); _census(monkeypatch, items=0, early=0)
    ft = report.fpf_tally(REP)
    assert ft["ok"] is True and ft["levers_silent"] == [] and ft["levers_unreached"] == ["fpf_trimul", "fpf_gflash", "fpf_ttr"]
    assert ft["unreached"]["fpf_trimul"] == "no_call_reached:zero_items"
    line = report.fpf_tally_line(ft)
    assert " ok=True" in line and "levers_silent=none" in line and "levers_unreached=fpf_trimul(no_call_reached:zero_items);fpf_gflash(no_call_reached:zero_items);fpf_ttr(no_call_reached:zero_items)" in line
    assert fold.fpf_failures([42], [{"seed": 42, "rc": 0}], {"42": {"fpf": ft}}) == []


def test_early_stop_of_every_input_is_named_not_silent(monkeypatch):
    """`early_stopping_plddt_threshold` (upstream default 0.5) stops a low-confidence input after the recycle-1 probe: no roll-out — the DiT /
    sampler / roll-out kernels reach no call: `no_call_reached:early_stop`."""
    _fake_fpf(monkeypatch); _census(monkeypatch, items=1, early=1)
    ft = report.fpf_tally(REP)
    assert ft["ok"] is True and ft["levers_unreached"] == ["fpf_trimul", "fpf_gflash", "fpf_ttr"] and ft["unreached"]["fpf_gflash"] == "no_call_reached:early_stop"


def test_a_silent_lever_in_a_process_that_rolled_out_still_fails_closed(monkeypatch):
    _fake_fpf(monkeypatch); _census(monkeypatch, items=2, early=1)                             # one input rolled out: the kernels HAD calls to serve
    ft = report.fpf_tally(REP)
    assert ft["ok"] is False and ft["levers_silent"] == ["fpf_trimul", "fpf_gflash", "fpf_ttr"] and ft["levers_unreached"] == []
    assert "silent lever(s)" in ft["reason"]
    assert fold.fpf_failures([42], [{"seed": 42, "rc": 0}], {"42": {"fpf": ft}}) == ["seed 42: FPF silent lever(s): fpf_trimul,fpf_gflash,fpf_ttr"]


def test_a_withheld_lever_is_off_by_design_never_silent(monkeypatch):
    """MODEL_OPT_LEVERS_OFF=fpf_tg pred --mode exact: the run withheld the trunk graph by name; the verdict does not count it silent."""
    _fake_fpf(monkeypatch, served=10); _census(monkeypatch, items=3, early=0)
    adp = sys.modules["fpf_rf3_adapter"]
    adp.describe_v2 = lambda: {"cfg": {"levers": True}, "counts": {"apb": {"served:safe": 96}}, "trunk_graph": {"replays": 0, "fallbacks": {}}}
    rep = {"mode": "exact", "levers": ["fpf_tg", "fpf_sapb"], "withheld": ["fpf_tg"], "fpf": {"arm": "sapb@L1", "applied": "configured"}}
    ft = report.fpf_tally(rep)
    assert ft["ok"] is True and ft["levers_silent"] == [] and "fpf_tg" not in ft["levers_acted"] and ft["levers_acted"] == ["fpf_sapb"]


def test_package_levers_whose_seam_no_input_reached_are_named_ok(monkeypatch):
    """mkdit / dtk 'no … call reached … (installed but never ran)' and xtr's, in a process where upstream rolled out nothing: ok by name; a block
    with calls, or a process that rolled out, is untouched."""
    _census(monkeypatch, items=1, early=1)
    out = {"upstream": upstream.describe(),
           "mkdit": {"on": True, "ok": False, "census": {"calls": 0}, "reason": "no DiffusionTransformer token call reached the mkdit seam in this process (the forward is installed but never ran)"},
           "dtk": {"on": True, "ok": False, "census": {"calls": 0}, "reason": "no AttentionPairBiasDiffusion call reached the dtk seam"},
           "xtr": {"on": True, "ok": True, "census": {"calls": 680, "served": 316}, "reason": None}}
    report.unreached_blocks(out)
    assert out["mkdit"]["ok"] is True and out["mkdit"]["reason"] == "no_call_reached:early_stop" and out["dtk"]["unreached"] == "early_stop" and out["xtr"]["reason"] is None
    assert fold.lever_failures("mkdit", [{"seed": 42, "rc": 0}], {"42": out}) == []
    _census(monkeypatch, items=2, early=1)                                                     # an input rolled out: the seam HAD calls to reach — fails closed
    out2 = {"upstream": upstream.describe(), "mkdit": {"on": True, "ok": False, "census": {"calls": 0}, "reason": "no call"}}
    report.unreached_blocks(out2)
    assert out2["mkdit"]["ok"] is False and fold.lever_failures("mkdit", [{"seed": 42, "rc": 0}], {"42": out2}) == ["seed 42: mkdit no call"]


def test_big_verdict_when_upstream_predicted_nothing(monkeypatch):
    _census(monkeypatch, items=0, early=0)
    t = {"upstream": upstream.describe(), "big": {"units": 0, "census": None, "exit": {"exit_code": 3, "reasons": {"big": "trigger never executed"}}}}
    assert fold.big_failures([42], [{"seed": 42, "rc": 0}], {"42": t}) == []
    t2 = {"upstream": dict(upstream.describe(), items=2, rolled_out=2), "big": {"units": 0, "census": None, "exit": {"exit_code": 3, "reasons": {"big": "trigger never executed"}}}}
    assert fold.big_failures([42], [{"seed": 42, "rc": 0}], {"42": t2}) != []                # inputs rolled out and the mode never applied: a failure


def test_exit_line_names_upstream_only_when_it_skipped_or_stopped_something(monkeypatch):
    _census(monkeypatch, items=1, early=1)
    t = {"mode": "fast", "graph_flags_imported": True, "upstream": upstream.describe(), "n_gpu": 1, "sharding": "none"}
    assert " upstream=items:1,early_stopped:1,rolled_out:0,no_rollout:early_stop " in report.tally_line(t) + " "
    _census(monkeypatch, items=3, early=0)
    t["upstream"] = upstream.describe()
    assert " upstream=" not in report.tally_line(t)                                            # a plain run's EXIT line is unchanged (the activation pins hold)
