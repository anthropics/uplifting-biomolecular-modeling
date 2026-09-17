"""The target entry file: one item mapped to a mapping (the file's shape is checked, its values never); the Hydra override rendering."""
import json
import os

import pytest

from complexa_opt import inputs


def write(tmp_path, doc, name="e.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def entry(tmp_path, **over):
    pdb = tmp_path / "target.pdb"
    pdb.write_text("ATOM\n")
    e = {"source": "custom", "target_filename": "7dz1A_N200", "target_path": str(pdb), "target_input": "A1-115", "hotspot_residues": ["A56", "A115"], "binder_length": [80, 80], "pdb_id": None}
    e.update(over)
    return e


def test_load_entry_and_overrides(tmp_path):
    e = entry(tmp_path)
    item, got = inputs.load_entry(write(tmp_path, {"7dz1A_N200": e}))
    assert item == "7dz1A_N200" and got == e
    assert inputs.overrides(item, got) == ["++generation.task_name=7dz1A_N200",
                                           "++generation.target_dict_cfg={7dz1A_N200:{source:'custom',target_filename:'7dz1A_N200',"
                                           f"target_path:'{e['target_path']}',"
                                           "target_input:'A1-115',hotspot_residues:['A56','A115'],binder_length:[80,80],pdb_id:null}}"]
    assert inputs.binder_length(got) == 80
    # upstream's own file form (top key target_dict_cfg) with one entry
    item2, got2 = inputs.load_entry(write(tmp_path, {"target_dict_cfg": {"x_1": dict(e, source="custom")}}, "f.json"))
    assert item2 == "x_1" and inputs.overrides(item2, got2)[1].startswith("++generation.target_dict_cfg={x_1:{source:'custom',target_filename:'7dz1A_N200',")


@pytest.mark.parametrize("doc_fn, words", [
    (lambda e: {"a": e, "b": e}, "exactly one top-level key"),
    (lambda e: {}, "exactly one top-level key"),
    (lambda e: [e], "exactly one top-level key"),
    (lambda e: {"a": "notamapping"}, "not a mapping"),
    (lambda e: {" ": e}, "item name is empty"),
])
def test_only_the_files_shape_is_refused(tmp_path, doc_fn, words):
    with pytest.raises(inputs.InputError, match=words):
        inputs.load_entry(write(tmp_path, doc_fn(entry(tmp_path))))


@pytest.mark.parametrize("doc_fn", [
    lambda e: {"a": dict(e, steps=400)},                                          # extra keys: upstream receives them
    lambda e: {"a": {k: v for k, v in e.items() if k != "hotspot_residues"}},     # fewer keys: upstream's configuration answers
    lambda e: {"a": dict(e, target_path="/nonexistent/x.pdb")},                  # a missing target file: upstream names it
    lambda e: {"a": dict(e, target_input="A1 115", hotspot_residues=["56"], binder_length=[90, 80], pdb_id="1jz7 A", source="")},
    lambda e: {"1234": e},                                                         # a numeric-looking name: Hydra types it, upstream answers
    lambda e: {"lig_1": {"ligand": "ATP", "smiles": "C1=CC=CC=C1", "binder_length": [60, 90]}},   # a ligand-style entry keeps its own keys
])
def test_values_and_key_sets_pass_through(tmp_path, doc_fn):
    item, got = inputs.load_entry(write(tmp_path, doc_fn(entry(tmp_path))))
    toks = inputs.overrides(item, got)
    assert toks[0] == f"++generation.task_name={item}" and toks[1].startswith("++generation.target_dict_cfg={" + item + ":{")
    for k in got:
        assert f"{k}:" in toks[1]


def test_relative_target_path_resolves_against_the_entry_dir(tmp_path):
    case = tmp_path / "n200" / "1gpbA_n200"
    case.mkdir(parents=True)
    (case / "target.pdb").write_text("ATOM\n")
    e = dict(entry(tmp_path), target_path="1gpbA_n200/target.pdb")
    p = tmp_path / "n200" / "1gpbA_n200.json"
    p.write_text(json.dumps({"1gpbA_n200": e}))
    item, got = inputs.load_entry(str(p))
    assert got["target_path"] == str(case / "target.pdb") and os.path.isabs(got["target_path"])
    assert f"target_path:'{case / 'target.pdb'}'," in inputs.overrides(item, got)[1]
    here = os.getcwd()
    try:                                                                       # independent of the caller's cwd
        os.chdir(str(tmp_path))
        assert inputs.load_entry(os.path.join("n200", "1gpbA_n200.json"))[1]["target_path"] == str(case / "target.pdb")
    finally:
        os.chdir(here)


def test_missing_file_and_bad_json(tmp_path):
    with pytest.raises(inputs.InputError, match="no such file"):
        inputs.load_entry(str(tmp_path / "nope.json"))
    p = tmp_path / "e.txt"
    p.write_text("{not json")
    with pytest.raises(inputs.InputError, match="not valid JSON"):
        inputs.load_entry(str(p))


def test_render_quotes_and_escapes_instead_of_refusing():
    assert inputs.render(None) == "null" and inputs.render(True) == "true" and inputs.render(3) == "3" and inputs.render([1, "A5"]) == "[1,'A5']"
    assert inputs.render("has space") == "'has space'"
    assert inputs.render("it's") == "'it\\'s'"
    assert inputs.render({"k": [1, None]}) == "{k:[1,null]}"


def test_the_dictionary_token_parses_and_merges_under_hydra(tmp_path):
    """The reason for the dictionary form: Hydra's KEY grammar cannot spell `target_dict_cfg.1gpbA_N200.source` (a path element starting with a
    digit), while the dictionary VALUE form parses, keeps the item name a string key, types every value, and MERGES into upstream's table."""
    pytest.importorskip("hydra")
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from hydra._internal.config_loader_impl import ConfigLoaderImpl
    from omegaconf import OmegaConf
    pdb = tmp_path / "target.pdb"; pdb.write_text("ATOM\n")
    e = dict(entry(tmp_path), target_path=str(pdb))
    item, got = inputs.load_entry(write(tmp_path, {"1gpbA_N200": e}, "e.json"))
    toks = inputs.overrides(item, got)
    parser = OverridesParser.create()
    with pytest.raises(Exception):
        parser.parse_override("++generation.target_dict_cfg.1gpbA_N200.source='x'")
    cfg = OmegaConf.create({"generation": {"task_name": "02_PDL1", "target_dict_cfg": {"02_PDL1": {"source": "x"}}}})
    OmegaConf.set_struct(cfg, True)
    ConfigLoaderImpl._apply_overrides_to_config(parser.parse_overrides(toks), cfg)
    g = cfg.generation
    assert g.task_name == "1gpbA_N200" and isinstance(g.task_name, str) and "02_PDL1" in g.target_dict_cfg
    t = g.target_dict_cfg["1gpbA_N200"]
    assert (t.source, t.target_filename, t.target_path, t.target_input, list(t.hotspot_residues), list(t.binder_length), t.pdb_id) == \
           ("custom", "7dz1A_N200", str(pdb), "A1-115", ["A56", "A115"], [80, 80], None)
    # a quoted value with an escaped quote and a space parses too (render escapes instead of refusing)
    cfg2 = OmegaConf.create({"generation": {"target_dict_cfg": {}}})
    ConfigLoaderImpl._apply_overrides_to_config(parser.parse_overrides(["++generation.target_dict_cfg={a:{pdb_id:" + inputs.render("it's a b") + "}}"]), cfg2)
    assert cfg2.generation.target_dict_cfg["a"].pdb_id == "it's a b"
