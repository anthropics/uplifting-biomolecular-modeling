"""A lever's state word is a function of its counters (levers.status, THE one reader): engaged ⇔ a work counter moved; a memoised CHECK never
counts as work; pair_assembly idle on padded batches (a binder_length range) is its DECLARED gate — `state=skipped reason=padded_batches`,
`levers_gated=` on TALLY / KIT-RECORD, `gated=` on EXIT, exit unaffected — and a lever whose site was never reached stays `never_engaged`
(partial, exit 3). CPU only: counters are set by hand, nothing of torch or upstream is imported."""
import inspect
import json
from collections import defaultdict

import pytest

from opt_core import report as core_report

from complexa_opt import activate, kit_design, levers, modes, report, settings, stock_design


@pytest.fixture
def counters(monkeypatch):
    """Hand-set lever counters on a clean registry state: `counters(lever, **n)`; the exact set reads as installed (census / active)."""
    monkeypatch.setattr(levers, "_STATS", defaultdict(lambda: defaultdict(int)))
    monkeypatch.setattr(levers, "_ACTIVE", set(modes.levers_of("exact")))
    for n in modes.levers_of("exact"):
        for k in levers.LEVERS[n].zeroed:
            levers._STATS[n][k] += 0

    def _set(lever, **n):
        levers._STATS[lever].update(n)
    return _set


def test_memo_books_a_check_apart_from_work_and_a_check_never_engages(counters):
    calls = []
    with levers.predict_step_scope():
        for _ in range(3):
            assert levers._memo(("uniform", 1), lambda: calls.append(1) or False, "pair_assembly", levers.CHECK) is False
    levers._memo(("uniform", 1), lambda: False, "pair_assembly", levers.CHECK)                      # outside a predict_step: computed, counted unscoped
    c = levers.census()["pair_assembly"]
    assert len(calls) == 1 and c["check_computed"] == 1 and c["check_served"] == 2 and c["check_unscoped"] == 1
    assert c["assembled"] == 0 and c["padded"] == 0                                                 # seeded at install: printed even at zero
    assert not levers.engaged("pair_assembly") and levers.status("pair_assembly") == ("skipped", levers.NEVER_ENGAGED) and not levers.gated("pair_assembly")
    with levers.predict_step_scope():                                                                # target_hoist's memo IS its work: bare counters engage
        levers._memo(("block", 1), lambda: 7, "target_hoist")
        levers._memo(("block", 1), lambda: 7, "target_hoist")
    assert levers.census()["target_hoist"] == {"computed": 1, "served": 1} and levers.status("target_hoist") == ("on", None)


def test_pair_assembly_state_follows_assembled_and_padded(counters):
    counters("pair_assembly", check_computed=8, check_served=3192, padded=3200)                     # a binder_length range: every call padded
    assert levers.status("pair_assembly") == ("skipped", "padded_batches") and levers.gated("pair_assembly") and not levers.engaged("pair_assembly")
    counters("pair_assembly", assembled=1600, padded=1600)                                           # mixed: engaged, both counts on the line
    assert levers.status("pair_assembly") == ("on", None) and not levers.gated("pair_assembly")
    counters("pair_assembly", assembled=3200, padded=0)                                              # a fixed binder_length [L, L]: every call assembled
    assert levers.status("pair_assembly") == ("on", None)
    assert "padded_batches" in levers.GATE_REASONS and levers.NEVER_ENGAGED not in levers.GATE_REASONS
    assert set(levers.GATES) <= set(levers.LEVERS) and all(g.counter in levers.LEVERS[n].zeroed for n, g in levers.GATES.items())


def _arm(monkeypatch, tmp_path, mode="exact"):
    monkeypatch.setattr(activate, "_STATE", {"mode": mode, "tier": modes.TIERS[mode], "levers": tuple(modes.levers_of(mode)), "gpu": {"name": None, "cc": None, "label": "none", "class": "none"},
                                             "upstream": "1.1.0", "torch": "none", "trigger": "test", "t_enable": 0.0, "record_path": str(tmp_path / "kit_1.json"), "active": True})


def test_exit_report_names_the_gate_and_prices_nothing(counters, monkeypatch, tmp_path):
    _arm(monkeypatch, tmp_path)
    counters("onehot_f32", served=10); counters("target_hoist", computed=2, served=798); counters("loop_desync", served=800)
    counters("pair_assembly", check_computed=2, check_served=798, padded=800)
    lines = {ln.split(" name=")[1].split()[0]: ln for ln in activate.lever_lines()}
    assert lines["pair_assembly"].startswith("[complexa-opt] LEVER name=pair_assembly state=skipped reason=padded_batches impl=complexa_opt.levers origin=kit assembled=0 ")
    assert " padded=800 " in lines["pair_assembly"] and lines["pair_assembly"].endswith(" tier=exact")
    assert lines["target_hoist"].startswith("[complexa-opt] LEVER name=target_hoist state=on impl=complexa_opt.levers origin=kit computed=2 ")
    assert activate.buckets() == {"on": ["onehot_f32", "target_hoist", "loop_desync"], "gated": ["pair_assembly"], "skipped": []}
    t = activate.tally_line()
    assert " mode=exact state=gated " in t and " levers_on=onehot_f32,target_hoist,loop_desync levers_skipped=none peak_alloc_gib=" in t and t.endswith(" levers_gated=pair_assembly")
    activate._write_record(final=True)
    doc = json.loads((tmp_path / "kit_1.json").read_text())
    assert doc["levers"]["pair_assembly"]["state"] == "skipped" and doc["levers"]["pair_assembly"]["reason"] == "padded_batches" and doc["levers"]["pair_assembly"]["counters"]["padded"] == 800
    assert doc["gated"] == ["pair_assembly"] and doc["partial"] == [] and doc["final"] is True
    # the parent reads it back: state gated, nothing priced, the gate's calls summed
    s = kit_design.read_records(str(tmp_path), "exact", modes.levers_of("exact"))
    assert s["state"] == "gated" and s["levers_on"] == ["onehot_f32", "target_hoist", "loop_desync"] and s["levers_gated"] == ["pair_assembly"] and s["levers_skipped"] == []
    assert s["partial"] == [] and s["gates"] == {"pair_assembly": {"reason": "padded_batches", "padded": 800}}
    k = report.kit_record_line(s)
    assert " state=gated levers_on=onehot_f32,target_hoist,loop_desync levers_skipped=none " in k and k.endswith(" levers_gated=pair_assembly")
    rep = {"active": True, "partial": s["partial"], "gated": [f"{n}:{s['gates'][n]['reason']}" for n in s["levers_gated"]]}
    v = core_report.verdict(0, rep, allow_partial=False, incomplete=None)
    assert v["exit_code"] == core_report.EXIT_OK and v["partial"] == [] and v["gated"] == ["pair_assembly:padded_batches"]
    x = report.exit_line(mode="exact", route="kit", rc=0, written=8, expected=8, wall=1.0, incomplete=None, exit_code=v["exit_code"], partial=v["partial"], gated=v["gated"])
    assert x.endswith(" designs_written=8 designs_expected=8 wall=1.0 gated=pair_assembly:padded_batches exit=0") and " partial=" not in x


def test_a_fixed_length_run_prints_what_it_always_printed_plus_a_trailing_levers_gated_none(counters, monkeypatch, tmp_path):
    _arm(monkeypatch, tmp_path)
    counters("onehot_f32", served=10); counters("target_hoist", computed=2, served=798); counters("loop_desync", served=800)
    counters("pair_assembly", check_computed=2, check_served=798, assembled=800)
    pa = [ln for ln in activate.lever_lines() if " name=pair_assembly " in ln][0]
    assert pa.startswith("[complexa-opt] LEVER name=pair_assembly state=on impl=complexa_opt.levers origin=kit assembled=800 ") and " padded=0 " in pa
    t = activate.tally_line()
    assert " state=complete " in t and " levers_on=onehot_f32,target_hoist,pair_assembly,loop_desync levers_skipped=none peak_alloc_gib=" in t and t.endswith(" levers_gated=none")
    activate._write_record(final=True)
    s = kit_design.read_records(str(tmp_path), "exact", modes.levers_of("exact"))
    assert s["state"] == "complete" and s["levers_gated"] == [] and s["partial"] == []
    assert report.kit_record_line(s).endswith(" levers_gated=none")
    x = report.exit_line(mode="exact", route="kit", rc=0, written=8, expected=8, wall=1.0, incomplete=None, exit_code=0, partial=[], gated=[])
    assert x.endswith(" wall=1.0 exit=0") and " gated=" not in x


def test_a_lever_never_reached_stays_a_priced_partial(counters, monkeypatch, tmp_path):
    _arm(monkeypatch, tmp_path)
    counters("onehot_f32", served=10); counters("loop_desync", served=800)
    counters("target_hoist", fallback_stock=800)                                                     # the pair factory is not the protein-target one: never hoisted, never assembled
    assert levers.status("target_hoist") == ("skipped", "never_engaged") and levers.status("pair_assembly") == ("skipped", "never_engaged")
    assert activate.buckets() == {"on": ["onehot_f32", "loop_desync"], "gated": [], "skipped": ["target_hoist", "pair_assembly"]}
    assert " state=partial " in activate.tally_line()
    activate._write_record(final=True)
    s = kit_design.read_records(str(tmp_path), "exact", modes.levers_of("exact"))
    assert s["state"] == "partial" and s["partial"] == ["target_hoist", "pair_assembly"] and s["levers_gated"] == []
    v = core_report.verdict(0, {"partial": s["partial"], "gated": []}, allow_partial=False, incomplete=None)
    assert v["exit_code"] == core_report.EXIT_NOT_ACTIVE


def test_records_that_disagree_are_priced_by_name(tmp_path):
    d = tmp_path
    lev = {n: {"state": "on", "tier": "exact", "counters": {}} for n in modes.levers_of("exact")}
    (d / "kit_1.json").write_text(json.dumps({"pid": 1, "mode": "fast", "final": True, "levers": lev}))
    s = kit_design.read_records(str(d), "exact", modes.levers_of("exact"))
    assert s["state"] == "partial" and s["partial"] == ["record_of_another_mode"]                    # a state word that says partial always carries a priced reason
    (d / "kit_2.json").write_text(json.dumps({"pid": 2, "mode": "exact", "final": False, "levers": lev}))
    (d / "kit_1.json").write_text(json.dumps({"pid": 1, "mode": "exact", "final": True, "levers": lev}))
    s = kit_design.read_records(str(d), "exact", modes.levers_of("exact"))
    assert s["state"] == "partial" and s["partial"] == ["record_not_final"]
    gl = dict(lev, pair_assembly={"state": "skipped", "reason": "padded_batches", "tier": "exact", "counters": {"padded": 5}})
    (d / "kit_2.json").write_text(json.dumps({"pid": 2, "mode": "exact", "final": True, "levers": gl}))
    s = kit_design.read_records(str(d), "exact", modes.levers_of("exact"))                          # one process assembled, the other was gated: gated, not partial
    assert s["state"] == "gated" and s["levers_gated"] == ["pair_assembly"] and s["gates"]["pair_assembly"]["padded"] == 5 and s["partial"] == []


def test_attn_sdpa_engages_by_its_dispatch_not_by_its_mask_check(counters):
    src = inspect.getsource(levers._attn_sdpa)
    assert '_STATS["attn_sdpa"]["served"] += 1' in src and '"attn_sdpa", CHECK)' in src                # every call it serves is booked bare; the mask check is a check
    counters("attn_sdpa", check_computed=12, check_served=120)
    assert levers.status("attn_sdpa") == ("skipped", levers.NEVER_ENGAGED) and "attn_sdpa" not in levers.GATES   # a check alone never reads as engagement (and attn_sdpa declares no gate)
    counters("attn_sdpa", served=67320)
    assert levers.status("attn_sdpa") == ("on", None)


def test_an_uncountable_design_count_is_named_before_the_launch(tmp_path, capsys):
    ov = ["++generation.dataloader.dataset.nres.nsamples=${n}"]
    stock_design.prelaunch(str(tmp_path), "02_PDL1", ov, settings.values_of(ov))
    err = capsys.readouterr().err
    assert "[complexa-opt] NOTE designs_expected=unknown: generation.dataloader.dataset.nres.nsamples=${n} × generation.dataloader.dataset.nrepeat_per_sample=1 is not" in err
    assert report.outputs_line("kit", "02_PDL1", 3, None, "/o/inference/x").split(" designs_expected=")[1].startswith("unknown ")
    x = report.exit_line(mode="off", route="stock", rc=0, written=3, expected=None, wall=1.0, incomplete=None, exit_code=0)
    assert " designs_expected=unknown " in x
    ov = ["++generation.dataloader.dataset.nres.nsamples=8"]
    stock_design.prelaunch(str(tmp_path), "02_PDL1", ov, settings.values_of(ov))
    assert "designs_expected" not in capsys.readouterr().err                                         # a countable run says nothing
