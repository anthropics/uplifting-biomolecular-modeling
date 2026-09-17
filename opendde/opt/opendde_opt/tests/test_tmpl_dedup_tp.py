"""tmpl_dedup on the row-sharded line (`--mode big --n_gpu P`, line BIG_TP): a rank process never enters TemplateEmbedder.forward (the
template pair blocks run on the rank's row shard through the core pair-block driver), so the lever — installed, counter 0 — is INERT BY DESIGN
there (`replaced_by=rowpair_tp`, LEVER state=off with the reason, the run complete), never `lever_never_ran:tmpl_dedup` (which refused every x2
prediction PARTIAL, exit 3, before this rule). On one card the lever stays inside its domain: a zero counter there IS the defect class."""
from opendde_opt import modes, ran, registry


def test_tmpl_dedup_rides_the_tp_line_inert_by_design_on_every_rank():
    assert "tmpl_dedup" not in modes.LINES["BIG_TP"].levers and "tmpl_dedup" in modes.BIG_TP_DROP and "tmpl_dedup" in ran.COUNTERS   # off the row-sharded line by name (0.2.39) …
    e = registry.ENGAGEMENT["tmpl_dedup"]                                                # … and inert by design in any --n_gpu P>1 process that still plans it
    assert e.single_gpu and e.single_gpu.startswith("replaced_by=rowpair_tp")
    for rank in (0, 1):                                                                  # both rank processes of --n_gpu 2 (rank 0 featurises; neither embeds templates by the module)
        facts = {"det": True, "n_gpu": 2, "rank": rank, "token_floors": {"q": 612}, "levers": list(modes.LINES["BIG_TP"].levers) + ["tmpl_dedup"]}
        ok, why = ran.engagement("tmpl_dedup", facts)
        assert ok is False and why.startswith("replaced_by=rowpair_tp")
        assert ran.inert_by_design(["tmpl_dedup"], facts) == {"tmpl_dedup": f"lever_inert_by_design:tmpl_dedup({why})"}
        assert ran.never_ran(["tmpl_dedup"], facts) == []                              # counter 0 outside its domain: not the defect class


def test_tmpl_dedup_on_one_card_is_inside_its_domain():
    facts = {"det": True, "n_gpu": 1, "rank": None, "token_floors": {"q": 612}, "levers": list(modes.LINES["S1"].levers)}
    assert ran.engagement("tmpl_dedup", facts) == (True, None)
    assert ran.never_ran(["tmpl_dedup"], facts) == ["lever_never_ran:tmpl_dedup"]      # a zero counter on one card IS named (ran.COUNTERS reads the live module: 0 here)


def test_a_rank_report_keeps_tmpl_dedup_inert_through_the_exit_fold(monkeypatch):
    """The exit-time fold (stack.refresh() after the predicted one) must not re-claim the lever applied once the predicted report demoted it
    to inert by design: the rank's LEVER row reads state=off with the reason."""
    from opendde_opt import report, stack, tmpldedup
    monkeypatch.setattr(tmpldedup, "kit_stats", lambda: {**tmpldedup.STATS, "installed": True, "calls": 0, "bypass": 0})
    why = "lever_inert_by_design:tmpl_dedup(replaced_by=rowpair_tp: ...)"
    rep = {"active": True, "mode": "big", "line": "BIG_TP", "levers_planned": ["tmpl_dedup"], "levers_applied": [], "levers_inert": ["tmpl_dedup"],
           "levers_inert_reasons": {"tmpl_dedup": why}, "levers_fallback": []}
    monkeypatch.setattr(stack, "_REPORT", rep)
    out = stack.refresh()                                                                # the unpredicted exit-time fold
    assert "tmpl_dedup" in out["levers_inert"] and "tmpl_dedup" not in out["levers_applied"] and not out.get("partial")
    assert report.lever_state("tmpl_dedup", out) == ("off", why)


def test_the_offload_unit_turns_tmpl_dedup_off_by_name():
    """On one card with the offload unit composed in (line BIG_F above the offload size gate) the offloaded trunk stage embeds the templates
    through its own streamed driver: TemplateEmbedder.forward is not entered there either, so the row is turned off BY NAME with the unit
    (modes.BIG_KIT_ROWS_OFF["pair_offload"]) — LEVER state=off, the run complete — instead of a zero counter refusing PARTIAL
    (lever_never_ran:tmpl_dedup at xl1400 before 0.2.39). The resident big line keeps it."""
    assert "tmpl_dedup" in modes.BIG_KIT_ROWS_OFF["pair_offload"]
    off = modes.big_line("BIG_F", ("pair_offload",))                                           # the offload unit composed in
    assert "tmpl_dedup" not in off.levers and any(x.startswith("pair_offload") for x in off.levers)
    res = modes.big_line("BIG_F", ())                                                         # resident big (below the offload size gate)
    assert "tmpl_dedup" in res.levers
