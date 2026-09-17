"""The template census guard (protenix_opt/templates.py): dropped hits are named, the per-chain census is printed, an item whose
DECLARED template slots are all dummy is named on a census line and runs as stock runs it; a run with use_template off names the skipped file; nothing declared under
use_template = one `templates=none` census line, no refusal, the item runs as stock (also beside a templated item of the same row).
Fakes stand in for the stock featurizer classes (the guard wraps whatever the two sites hold)."""
import sys
import types

import pytest

from protenix_opt import templates as T


class _Result:
    def __init__(self, features, errors=(), warnings=()):
        self.features, self.errors, self.warnings = list(features), list(errors), list(warnings)


def _fake_modules(monkeypatch, plan):
    """Install fake stock modules: TemplateHitFeaturizer.get_templates returns plan[seq] = (features, errors); make_template_feature
    mimics the stock loop (per protein chain with templatesPath and use_template: get_templates; assemble)."""
    tu = types.ModuleType("protenix.data.template.template_utils")

    class TemplateHitFeaturizer:
        def get_templates(self, sequence_uid, query_sequence, hits, max_template_date=None):
            feats, errs = plan[query_sequence]
            return _Result(feats, errs), {"track": 1}
    tu.TemplateHitFeaturizer = TemplateHitFeaturizer
    tf = types.ModuleType("protenix.data.template.template_featurizer")

    class InferenceTemplateFeaturizer:
        @staticmethod
        def make_template_feature(bioassembly, atom_array, use_template=True, online_template_featurizer=None):
            n = 0
            for info in bioassembly:
                c = info.get("proteinChain")
                if c and c.get("templatesPath") and use_template and online_template_featurizer:
                    res, _ = online_template_featurizer.get_templates(sequence_uid=c["sequence"], query_sequence=c["sequence"], hits=["h"] * c.get("_hits", 2))
                    n += len(res.features)
            return {"n_real": n}
    tf.InferenceTemplateFeaturizer = InferenceTemplateFeaturizer
    for name, mod in (("protenix", types.ModuleType("protenix")), ("protenix.data", types.ModuleType("protenix.data")),
                      ("protenix.data.template", types.ModuleType("protenix.data.template")), (tu.__name__, tu), (tf.__name__, tf)):
        monkeypatch.setitem(sys.modules, name, mod)
    return tu, tf


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for k, v in {"installed": [], "items": 0, "items_no_templates": 0, "chains_requested": 0, "chains_skipped_use_template_off": 0, "hits": 0, "real": 0,
                 "dropped": [], "warnings": 0, "all_dummy_items": 0, "calls": []}.items():
        monkeypatch.setitem(T._STATE, k, v)
    import opt_core.autoload as A
    monkeypatch.setattr(A, "_PATCHES", {})
    yield


BIO = [{"proteinChain": {"sequence": "MKV" * 10, "count": 1, "templatesPath": "/x/chainA.a3m", "_hits": 4}},
       {"ligand": {"ligand": "CCD_ATP", "count": 1}},
       {"proteinChain": {"sequence": "GGS" * 9, "count": 2, "templatesPath": "/x/chainB.hhr", "_hits": 3}}]


def test_partial_census_printed_and_drops_named(monkeypatch, capsys):
    tu, tf = _fake_modules(monkeypatch, {"MKV" * 10: (["f1", "f2"], ["Error processing hit: TemplateAtomMaskAllZerosError x", "Error processing hit: CaDistanceError y"]),
                                        "GGS" * 9: (["g1", "g2", "g3"], [])})
    assert T.install().startswith("TEMPLATE_CENSUS:")
    out = tf.InferenceTemplateFeaturizer.make_template_feature(bioassembly=BIO, atom_array=None, use_template=True, online_template_featurizer=tu.TemplateHitFeaturizer())
    assert out == {"n_real": 5}
    txt = capsys.readouterr().out
    assert txt.count("[protenix-opt] TEMPLATE DROPPED seq=") == 2 and "TemplateAtomMaskAllZerosError" in txt
    assert f"[protenix-opt] TEMPLATES entity=0 seq={T.seq_id('MKV' * 10)} path=chainA.a3m hits=4 real=2/4 dropped=2" in txt
    assert f"[protenix-opt] TEMPLATES entity=2 seq={T.seq_id('GGS' * 9)} path=chainB.hhr hits=3 real=3/4 dropped=0" in txt
    assert "templates=none" not in txt
    s = T.state()
    assert (s["items"], s["items_no_templates"], s["chains_requested"], s["hits"], s["real"], s["dropped"], s["all_dummy_items"]) == (1, 0, 2, 7, 5, 2, 0)
    assert dict(T.evidence())["templ_real"] == 5


def test_all_dummy_item_is_named_and_runs_as_stock(monkeypatch, capsys):
    """Declared templates whose every slot comes out dummy: one `TEMPLATES templates=all_dummy` line, the item counted, and the stock
    featurizer's own result returned unchanged (the item runs exactly as stock runs it — nothing stock accepts is refused)."""
    tu, tf = _fake_modules(monkeypatch, {"MKV" * 10: ([], ["Error processing hit: TemplateAtomMaskAllZerosError a"] * 4), "GGS" * 9: ([], ["Error processing hit: missing cif"] * 3)})
    T.install()
    out = tf.InferenceTemplateFeaturizer.make_template_feature(bioassembly=BIO, atom_array=None, use_template=True, online_template_featurizer=tu.TemplateHitFeaturizer())
    assert out is not None
    txt = capsys.readouterr().out
    assert txt.count("TEMPLATE DROPPED") == 7 and "real=0/4 dropped=4" in txt and "real=0/4 dropped=3" in txt
    lines = [l for l in txt.splitlines() if "TEMPLATES templates=all_dummy" in l]
    assert len(lines) == 1 and lines[0].startswith("[protenix-opt] TEMPLATES templates=all_dummy entities=0,2 hits=7 real=0/4 reasons=") and "TemplateAtomMaskAllZerosError" in lines[0]
    assert T.state()["all_dummy_items"] == 1 and ("templ_all_dummy", 1) in T.evidence()
    assert not hasattr(T, "TemplateSlotsAllDummy")

def test_use_template_off_names_the_skipped_files_and_never_refuses(monkeypatch, capsys):
    tu, tf = _fake_modules(monkeypatch, {"MKV" * 10: ([], []), "GGS" * 9: ([], [])})
    T.install()
    tf.InferenceTemplateFeaturizer.make_template_feature(bioassembly=BIO, atom_array=None, use_template=False, online_template_featurizer=tu.TemplateHitFeaturizer())
    txt = capsys.readouterr().out
    assert txt.count("skipped=use_template_off") == 2 and "TEMPLATE DROPPED" not in txt
    assert T.state()["chains_skipped_use_template_off"] == 2 and T.state()["all_dummy_items"] == 0


NO_TEMPLATES = [{"proteinChain": {"sequence": "MKV", "count": 1}}, {"dnaSequence": {"sequence": "ACGT", "count": 2}}, {"ligand": {"ligand": "CCD_ATP", "count": 1}}]


def test_no_templates_declared_under_use_template_logs_none_and_runs(monkeypatch, capsys):
    """(2) an item that names no templatesPath, use_template on: one `templates=none` census line, no refusal, the stock result returned."""
    tu, tf = _fake_modules(monkeypatch, {})
    T.install()
    out = tf.InferenceTemplateFeaturizer.make_template_feature(bioassembly=NO_TEMPLATES, atom_array=None, use_template=True, online_template_featurizer=tu.TemplateHitFeaturizer())
    assert out == {"n_real": 0}
    assert capsys.readouterr().out == "[protenix-opt] TEMPLATES templates=none requested=0 entities=3\n"
    s = T.state()
    assert (s["items"], s["items_no_templates"], s["chains_requested"], s["all_dummy_items"]) == (1, 1, 0, 0)


def test_no_templates_declared_use_template_off_prints_nothing(monkeypatch, capsys):
    """use_template off and nothing declared: nothing to name (stock never reads a template there) — no line, no refusal."""
    tu, tf = _fake_modules(monkeypatch, {})
    T.install()
    tf.InferenceTemplateFeaturizer.make_template_feature(bioassembly=NO_TEMPLATES, atom_array=None, use_template=False, online_template_featurizer=tu.TemplateHitFeaturizer())
    assert capsys.readouterr().out == "" and (T.state()["items"], T.state()["items_no_templates"], T.state()["all_dummy_items"]) == (1, 0, 0)


def test_row_mixing_templated_and_template_less_items_both_proceed(monkeypatch, capsys):
    """(4) one run (use_template on for the row): a templated item then a template-less item — both featurise, neither is refused."""
    tu, tf = _fake_modules(monkeypatch, {"MKV" * 10: (["f1", "f2"], []), "GGS" * 9: (["g1"], ["Error processing hit: CaDistanceError y"])})
    T.install()
    feat = tu.TemplateHitFeaturizer()
    out1 = tf.InferenceTemplateFeaturizer.make_template_feature(bioassembly=BIO, atom_array=None, use_template=True, online_template_featurizer=feat)
    out2 = tf.InferenceTemplateFeaturizer.make_template_feature(bioassembly=NO_TEMPLATES, atom_array=None, use_template=True, online_template_featurizer=feat)
    assert (out1, out2) == ({"n_real": 3}, {"n_real": 0})
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"[protenix-opt] TEMPLATE DROPPED seq={T.seq_id('GGS' * 9)} reason=Error processing hit: CaDistanceError y",
                     f"[protenix-opt] TEMPLATES entity=0 seq={T.seq_id('MKV' * 10)} path=chainA.a3m hits=4 real=2/4 dropped=0",
                     f"[protenix-opt] TEMPLATES entity=2 seq={T.seq_id('GGS' * 9)} path=chainB.hhr hits=3 real=1/4 dropped=1",
                     "[protenix-opt] TEMPLATES templates=none requested=0 entities=3"]
    s = T.state()
    assert (s["items"], s["items_no_templates"], s["chains_requested"], s["real"], s["all_dummy_items"]) == (2, 1, 2, 3, 0)


def test_install_before_import_patches_at_import(monkeypatch, capsys):
    """The sites are armed before the stock modules exist; importing them afterwards installs the wrappers (autoload seam)."""
    for m in list(sys.modules):
        if m.startswith("protenix"):
            monkeypatch.delitem(sys.modules, m, raising=False)
    marker = T.install()
    assert marker.startswith("TEMPLATE_CENSUS:")
    tu, tf = _fake_modules(monkeypatch, {"MKV" * 10: (["f"], []), "GGS" * 9: (["g"], [])})
    import opt_core.autoload as A
    for p in A._PATCHES.values():                    # the finder patches a real import; the fakes are injected, so drive the patch the way exec_module would
        if p.state != "installed":
            p.install()
    tf.InferenceTemplateFeaturizer.make_template_feature(bioassembly=BIO, atom_array=None, use_template=True, online_template_featurizer=tu.TemplateHitFeaturizer())
    assert "real=1/4" in capsys.readouterr().out


def test_census_item_pure():
    calls = [{"seq": T.seq_id("A" * 5), "hits": 4, "real": 0, "dropped": 4}]
    cen = T.census_item([{"proteinChain": {"sequence": "A" * 5, "count": 1, "templatesPath": "t.a3m"}}], True, True, calls)
    assert cen["all_dummy"] is True and cen["real_total"] == 0 and cen["requested"] == 1
    cen = T.census_item([{"proteinChain": {"sequence": "A" * 5, "count": 1, "templatesPath": "t.a3m"}}], False, True, [])
    assert cen["all_dummy"] is False and cen["rows"][0]["skipped"] == "use_template_off"
    cen = T.census_item(NO_TEMPLATES, True, True, [])
    assert (cen["requested"], cen["active"], cen["all_dummy"]) == (0, 0, False), "nothing declared is never all-dummy"


