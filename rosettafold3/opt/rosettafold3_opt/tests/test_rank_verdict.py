"""``pred --n_gpu P > 1``: the package levers' verdict reads EVERY rank's exit tally (fold.lever_rank_failures), and a lever the
row-sharding adapter turns off by name (rowpair.CONFLICTS: ``dtk``) is a named disengagement in each rank's tally (dtk.decline), on the
LEVER line (report.lever_states) and on the RANKS line — never a failure and never silent. ``n_gpu == 1`` keeps fold.lever_failures."""
import copy

import pytest

from rosettafold3_opt import dtk, fold, report, rowpair, stack

SERVED = {"on": True, "impl": "/core/opt_core/kernels/dtk_kernels.py", "origin": "core", "bias": "relayout", "min_tokens": 400,
          "census": {"calls": 48, "served": 48, "gated": 0, "fallback": 0}, "ok": True, "reason": None}
DEAD = dict(SERVED, census={"calls": 0, "served": 0, "gated": 0, "fallback": 0}, ok=False,
            reason="no AttentionPairBiasDiffusion call reached the dtk seam in this process (the rewrite is installed but never ran)")
FALLBACK = dict(SERVED, census={"calls": 48, "served": 47, "gated": 0, "fallback": 1, "fallback_by": {"bias_shape": 1}}, ok=False,
                reason="dtk fallbacks: fallback:bias_shape=1")


@pytest.fixture
def declined():
    saved = copy.deepcopy(dtk.STATE)
    try:
        dtk.STATE.update(on=False, conflict=None, conflict_reason=None)
        yield dtk.decline("n_gpu", rowpair.CONFLICTS["dtk"])
    finally:
        dtk.STATE.clear()
        dtk.STATE.update(saved)


def _run(seed=42, rc=0):
    return {"seed": seed, "rc": rc, "n_gpu": 2, "ranks_reported": 2}


def test_dtk_is_named_in_rowpair_conflicts():
    assert "dtk" in rowpair.CONFLICTS and "dtk" in rowpair.plan(2, "big")["conflicts"] and rowpair.plan(1, "big")["conflicts"] == {}
    assert stack._rowpair_conflicts()["dtk"] == rowpair.CONFLICTS["dtk"]


def test_decline_describes_the_conflict_by_name(declined):
    assert declined["on"] is False and declined["conflict"] == "n_gpu" and declined["ok"] is True and declined["census"] is None
    assert declined["reason"].startswith("off:conflict:n_gpu: ") and dtk.problems() == []


def test_lever_line_state_is_off_conflict(declined):
    st = report.lever_states({"levers": ["dtk"]}, {"graph_flags_imported": True, "mode": "big", "dtk": declined})
    assert st["dtk"]["state"] == "off" and st["dtk"]["reason"] == "conflict:n_gpu", st["dtk"]


def test_all_ranks_declined_passes_and_the_census_names_it(declined):
    bad, recs = fold.lever_rank_failures("dtk", [_run()], {"42": {"dtk": declined}}, {"42": {1: {"dtk": dict(declined)}}}, 2)
    assert bad == [], bad
    assert recs == [{"lever": "dtk", "seed": "42", "n_gpu": 2, "ranks_reported": 2, "state": "off:conflict:n_gpu",
                     "calls": 0, "served": 0, "gated": 0, "fallback": 0, "per_rank_calls": "0,0"}], recs


def test_all_ranks_served_passes_with_summed_counters():
    bad, recs = fold.lever_rank_failures("dtk", [_run()], {"42": {"dtk": SERVED}}, {"42": {1: {"dtk": SERVED}}}, 2)
    assert bad == [] and recs[0]["state"] == "on" and recs[0]["calls"] == 96 and recs[0]["fallback"] == 0 and recs[0]["per_rank_calls"] == "48,48", recs


def test_a_rank_whose_seam_no_call_reached_fails_by_name():
    bad, recs = fold.lever_rank_failures("dtk", [_run()], {"42": {"dtk": SERVED}}, {"42": {1: {"dtk": DEAD}}}, 2)
    assert len(bad) == 1 and bad[0].startswith("seed 42 rank 1: dtk no AttentionPairBiasDiffusion call reached"), bad
    assert recs[0]["state"] == "mixed" and recs[0]["per_rank_calls"] == "48,0", recs


def test_any_fallback_on_any_rank_fails_by_name():
    bad, _ = fold.lever_rank_failures("dtk", [_run()], {"42": {"dtk": FALLBACK}}, {"42": {1: {"dtk": SERVED}}}, 2)
    assert bad == ["seed 42 rank 0: dtk dtk fallbacks: fallback:bias_shape=1"], bad


def test_a_missing_rank_tally_fails_by_name(declined):
    bad, recs = fold.lever_rank_failures("dtk", [_run()], {"42": {"dtk": declined}}, {"42": {}}, 2)
    assert bad == ["seed 42 rank 1: no dtk tally (the lever's block was not written at that rank's exit)"], bad
    assert recs[0]["ranks_reported"] == 1 and recs[0]["state"] == "mixed", recs


def test_ranks_in_different_states_fail_by_name(declined):
    bad, recs = fold.lever_rank_failures("dtk", [_run()], {"42": {"dtk": declined}}, {"42": {1: {"dtk": SERVED}}}, 2)
    assert len(bad) == 1 and bad[0].startswith("seed 42: dtk ranks disagree (rank 0=off:conflict:n_gpu, rank 1=on)"), bad
    assert recs[0]["state"] == "mixed"


def test_a_failed_run_is_not_judged_twice():
    assert fold.lever_rank_failures("dtk", [_run(rc=1)], {}, {}, 2) == ([], [])


def test_n_gpu_1_verdict_is_the_single_tally_one():
    assert fold.lever_failures("dtk", [{"seed": 42, "rc": 0}], {"42": {"dtk": SERVED}}) == []
    assert fold.lever_failures("dtk", [{"seed": 42, "rc": 0}], {"42": {"dtk": DEAD}}) == [f"seed 42: dtk {DEAD['reason']}"]
    assert fold.lever_failures("dtk", [{"seed": 42, "rc": 0}], {"42": {}}) == ["seed 42: no dtk tally (the lever's counters were not written at exit)"]
