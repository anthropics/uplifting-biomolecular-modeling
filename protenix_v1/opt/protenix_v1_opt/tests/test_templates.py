"""The template guard (templates.py): the slot census, the all-dummy event and count, the dropped-hit events around the stock featurizer,
the EXIT-line fields. CPU only; the stock featurizer is a stand-in class installed under the stock module name."""

import pytest

from protenix_v1_opt import report as R
from protenix_v1_opt import templates as T


@pytest.fixture(autouse=True)
def fresh_tally(monkeypatch):
    monkeypatch.setattr(T, "_TALLY", {k: (False if k == "installed" else 0) for k in T._TALLY})
    monkeypatch.setattr(T, "_DECLARED", {})
    yield


def feats(masks, asym):
    """masks: [T][N] 0/1 lists under the first MASK_KEYS name; asym: [N] chain ids."""
    return {T.MASK_KEYS[0]: masks, "asym_id": asym}


def test_census_counts_real_slots_and_per_chain():
    c = T.census(feats([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]], [0, 0, 1, 1]))
    assert c["slots"] == 4 and c["real"] == 2 and c["dummy"] == 2
    assert c["per_template"] == [True, False, True, False]
    assert c["per_chain"] == {"0": 1, "1": 1}


def test_census_none_without_template_masks():
    assert T.census({"asym_id": [0, 0]}) is None


def test_all_atom_mask_trailing_dims_reduce():
    c = T.census({"template_all_atom_mask": [[[0, 0], [0, 1]], [[0, 0], [0, 0]]], "asym_id": [5, 5]})
    assert c["key"] == "template_all_atom_mask" and c["real"] == 1 and c["per_chain"] == {"5": 1}


def test_untemplated_run_censuses_nothing(capsys):
    assert T.check_item({"use_template": False}, {"input_feature_dict": feats([[0, 0]], [0, 0])}) is None
    assert T.tally()["items"] == 1 and T.tally()["items_templated"] == 0
    assert capsys.readouterr().err == ""
    assert T.exit_fields() == {"templates": "off"}


def test_all_dummy_templated_item_runs_as_the_stock_runs_it(capsys):
    """An item whose declared templates all dropped: the census line, the `all_dummy` line and count, and the item proceeds (no refusal,
    no non-zero exit) — untemplated, exactly as the stock runs it."""
    c = T.check_item({"use_template": True}, {"sample_name": "8IO9", "input_feature_dict": feats([[0, 0, 0]] * 4, [0, 0, 1])})
    assert c["real"] == 0 and c["slots"] == 4
    err = capsys.readouterr().err
    assert f"{R.PREFIX} TEMPLATE event=census item=8IO9 slots=4 real=0 dummy=4 per_chain=0:0,1:0" in err
    assert f"{R.PREFIX} TEMPLATE event=all_dummy item=8IO9 slots=4" in err and "refused" not in err
    assert T.tally()["all_dummy"] == 1 and R.template_exit_line().endswith(" tmpl_all_dummy=1")


def test_dummy_substitute_without_masks_counts_as_all_dummy():
    """use_template=true and a feature dict with no template mask at all (the dataloader's dummy branch) = 0 real slots: counted, proceeds."""
    c = T.check_item({"use_template": "true"}, {"input_feature_dict": {"asym_id": [0]}})
    assert c["real"] == 0 and T.tally()["all_dummy"] == 1


# ---------------------------------------------------------------------------------------------- declared templates (the input JSON's templatesPath)
def _row(tmp_path, *items):
    """An input JSON of `items` = (name, templatesPath or None per protein chain...) -> configs naming it."""
    import json as _json
    rows = [{"name": n, "sequences": [{"proteinChain": dict({"sequence": "MK", "count": 1}, **({"templatesPath": t} if t else {}))} for t in ts]
                                     + [{"dnaSequence": {"sequence": "ACGT", "count": 2}}]} for n, *ts in items]
    p = tmp_path / "row.json"; p.write_text(_json.dumps(rows))
    T._DECLARED.clear()
    return {"use_template": True, "input_json_path": str(p)}


def test_declared_templates_counts_the_chains_that_name_a_path(tmp_path):
    cfg = _row(tmp_path, ("A", "/t/a.hhr", None), ("B",), ("C", None))
    assert [T.declared_templates(cfg, {"sample_index": i, "sample_name": n}, n) for i, n in enumerate("ABC")] == [1, 0, 0]
    assert T.declared_templates(cfg, {"sample_name": "A"}, "A") == 1                          # by name when no index travels with the item
    assert T.declared_templates(cfg, {"sample_index": 0}, "nope") is None                     # index/name disagree and the name is unknown: undecidable
    assert T.declared_templates({"use_template": True}, {}, "A") is None                       # no input_json_path: the census alone decides


def test_templated_item_decision_is_unchanged(tmp_path, capsys):
    """(1) an item that declares a template: real slots -> the census line and it proceeds; (3) all dummy -> the `all_dummy` line and count, and it proceeds as the stock runs it."""
    cfg = _row(tmp_path, ("7VCU", "/t/7vcu.hhr"), ("8IO9", "/t/8io9.a3m"))
    c = T.check_item(cfg, {"sample_name": "7VCU", "sample_index": 0, "input_feature_dict": feats([[1, 1, 0], [0, 0, 0]], [0, 0, 1])})
    assert c["real"] == 1
    assert T.check_item(cfg, {"sample_name": "8IO9", "sample_index": 1, "input_feature_dict": feats([[0, 0, 0]] * 4, [0, 0, 1])})["real"] == 0
    err = capsys.readouterr().err
    assert f"{R.PREFIX} TEMPLATE event=census item=7VCU slots=2 real=1 dummy=1 per_chain=0:1,1:0" in err
    assert f"{R.PREFIX} TEMPLATE event=census item=8IO9 slots=4 real=0 dummy=4 per_chain=0:0,1:0" in err
    assert f"{R.PREFIX} TEMPLATE event=all_dummy item=8IO9 slots=4" in err
    assert "event=none" not in err and T.tally()["all_dummy"] == 1 and T.tally()["none"] == 0


def test_an_item_that_declares_no_template_runs_untemplated_by_name(tmp_path, capsys):
    """(2) use_template=true for the run, no templatesPath on the item (e.g. a DNA duplex, or protein chains without one): the stock's dummy
    slots, one `TEMPLATE event=none` line, no census, no refusal."""
    cfg = _row(tmp_path, ("entity_dsdna_1bna_duplex",), ("plain_protein", None, None))
    assert T.check_item(cfg, {"sample_name": "entity_dsdna_1bna_duplex", "sample_index": 0, "input_feature_dict": feats([[0, 0]] * 4, [0, 0])}) is None
    assert T.check_item(cfg, {"sample_name": "plain_protein", "sample_index": 1, "input_feature_dict": {"asym_id": [0, 1]}}) is None
    err = capsys.readouterr().err
    assert f"{R.PREFIX} TEMPLATE event=none item=entity_dsdna_1bna_duplex declared=0" in err
    assert f"{R.PREFIX} TEMPLATE event=none item=plain_protein declared=0" in err
    assert "event=refused" not in err and "event=census" not in err
    t = T.tally(); assert (t["items"], t["items_templated"], t["none"], t["all_dummy"]) == (2, 0, 2, 0)
    assert R.template_exit_line() == f"{R.PREFIX} TEMPLATE event=exit templates=on tmpl_items=0 tmpl_slots=0 tmpl_real=0 tmpl_searches=0 tmpl_hits=0 tmpl_kept=0 tmpl_dropped=0 tmpl_none=2"


def test_a_row_mixing_templated_and_template_less_items_proceeds(tmp_path, capsys):
    """(4) one row, one templated item with a real slot and one item declaring none: both proceed; the EXIT fields count each kind."""
    cfg = _row(tmp_path, ("7VCU", "/t/7vcu.hhr"), ("entity_dsdna_1bna_duplex",))
    assert T.check_item(cfg, {"sample_name": "7VCU", "sample_index": 0, "input_feature_dict": feats([[1, 0]], [0, 0])})["real"] == 1
    assert T.check_item(cfg, {"sample_name": "entity_dsdna_1bna_duplex", "sample_index": 1, "input_feature_dict": feats([[0, 0]] * 4, [0, 0])}) is None
    err = capsys.readouterr().err
    assert "TEMPLATE event=census item=7VCU slots=1 real=1 dummy=0" in err and "TEMPLATE event=none item=entity_dsdna_1bna_duplex declared=0" in err
    assert "event=refused" not in err
    assert R.template_exit_line().endswith(" tmpl_items=1 tmpl_slots=1 tmpl_real=1 tmpl_searches=0 tmpl_hits=0 tmpl_kept=0 tmpl_dropped=0 tmpl_none=1")
