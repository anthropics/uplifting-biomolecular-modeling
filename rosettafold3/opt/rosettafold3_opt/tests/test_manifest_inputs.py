"""warm's input helper (the add-on's public tiles: schema, selection, write) and the output comparator (the CLI's file set, _entry.time excluded, ranking_score at production)."""
import json
import os

from .. import inputs, stack


def test_inputs_schema_and_selection(tmp_path):
    p = tmp_path / "in.json"
    p.write_text(json.dumps([{"name": "a", "components": [{"seq": "AAA", "chain_id": "A", "msa_path": "msas/a.a3m"}]},
                             {"name": "b", "components": [{"ccd_code": "HEM", "chain_id": "B"}]}]))
    (tmp_path / "msas").mkdir()
    (tmp_path / "msas" / "a.a3m").write_text(">a\nAAA\n")
    items = inputs.load(str(p))
    out = inputs.write([it for it in items if it["name"] == "a"], str(tmp_path / "o" / "sel.json"))
    assert json.load(open(out))[0]["name"] == "a"
    for bad in ([{"name": "x"}], [{"name": "x", "components": []}], [{"name": "x", "components": [{"chain_id": "A"}]}],
                [{"name": "x", "components": [{"seq": "A", "chain_id": "A"}]}, {"name": "x", "components": [{"seq": "A", "chain_id": "A"}]}]):
        p.write_text(json.dumps(bad))
        try:
            inputs.load(str(p))
        except inputs.InputError:
            continue
        raise AssertionError(bad)
    # the kit's public tiles load through the same reader
    tiles = inputs.load(os.path.join(stack.kit_home(), "public_inputs", "1brs_tiles.json"))
    assert [t["name"] for t in tiles] == ["1brs_barnase_barstar_1to1", "1brs_barnase_barstar_2to2", "1brs_barnase_barstar_3to3", "1brs_barnase_barstar_4to4"]
