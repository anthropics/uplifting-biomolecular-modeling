"""The template census and its words: an item whose fold input declares no templates runs untemplated as stock does and says
`templates=none`; declared-but-all-dummy runs untemplated too and is counted (dummy_only); a templated item is untouched; a pass mixing
templated and template-less items runs both."""
import json
import os

from af3_torch_opt import cli
from af3_torch_opt import forward as fwd


def _input(tmp, name, n_templates, seeds=(1,)):
    """A fold input whose one protein chain declares `n_templates` templates (the AlphaFold 3 JSON's per-chain `templates` list)."""
    d = tmp / "in"; d.mkdir(exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps({"name": name, "modelSeeds": list(seeds),
                             "sequences": [{"protein": {"id": "A", "sequence": "ACDEFGHIK", "templates": [{"mmcif": f"t{i}", "queryIndices": [0], "templateIndices": [0]} for i in range(n_templates)]}}]}))
    return str(p)


def _lines(err, tag):
    return [l for l in err.splitlines() if l.startswith(f"[af3-torch-opt] {tag} ")]



def test_words():
    assert cli.templates_word({"templates_declared": 0, "templates_live": 0}) == cli.TEMPLATES_NONE == "none"
    assert cli.templates_word({"templates_declared": 2, "templates_live": 2}) == "2/2" and cli.templates_word({"templates_declared": 2, "templates_live": 0}) == "0/2"
    assert cli.templates_word({}) is None                                                     # a step that never reached the item: no word
    c = cli.templates_census([{"forwards": [{"templates_declared": 0, "templates_live": 0}, {"templates_declared": 2, "templates_live": 2},
                                            {"templates_declared": 1, "templates_live": 0}, {}]}])
    assert c == {"items": 4, "declared": 3, "live": 2, "dummy_only": 1, "none": 1, "slots": 0, "slots_live": 0}                 # the record with no template fields (never reached) is not a `none` item


def test_templated_and_template_less_items_in_one_pass(box, capsys):
    """A pass over a templated item (1 template, featurised live) and a template-less one: both run; the templated
    item's word is 1/1, the template-less one's is `none`; TEMPLATES counts none=1 refused=0; the model process is told only about the
    declared item (--templates-declared t=1)."""
    out = box["tmp"] / "out"
    rc = cli.main(["pred", "--mode", "fast", "--json_path", _input(box["tmp"], "t", 1), "--json_path", _input(box["tmp"], "u", 0), "--output_dir", str(out)])
    err = capsys.readouterr().err
    assert rc == 0, err[-3000:]
    items = _lines(err, "ITEM")
    assert len(items) == 2 and " name=t seed=1 " in items[0] and " ok=1 " in items[0] and " templates=1/1 " in items[0], items
    assert " name=u seed=1 " in items[1] and " ok=1 " in items[1] and " templates=none " in items[1], items
    (t,) = _lines(err, "TEMPLATES")
    assert t == "[af3-torch-opt] TEMPLATES items=2 declared=1 live=1 refused=0 reason=None dummy_only=0 none=1 real=1/8 form=dense", t
    assert " templates=1/1 " in _lines(err, "DONE")[0] and " items=2/2 failed=none " in _lines(err, "DONE")[0]
    fj = cli.last_run()["reports"]["forward"]
    assert [(i["name"], i["templates_declared"], i["templates_live"], i["ok"]) for i in fj["items"]] == [("t", 1, 1, True), ("u", 0, 0, True)]
    from .conftest import stub_calls
    fwd_argv = [c for c in stub_calls(box) if os.path.basename(c[1]) == "forward.py"][0]
    assert [fwd_argv[i + 1] for i, a in enumerate(fwd_argv) if a == "--templates-declared"] == ["t=1"]   # the template-less item is not declared to the model process


def test_declared_but_all_dummy_runs_untemplated_exit_0_and_is_counted(box, capsys, monkeypatch):
    """The templated item's slots all featurise dummy (STUB_TEMPLATES_LIVE d=0): the model runs it untemplated as stock does, its outputs are
    written (items 2/2), its ITEM line says templates=0/2, the TEMPLATES census counts dummy_only=1, exit 0; the template-less neighbour says `none`."""
    monkeypatch.setenv("STUB_TEMPLATES_LIVE", "d=0")
    out = box["tmp"] / "out"
    rc = cli.main(["pred", "--mode", "fast", "--json_path", _input(box["tmp"], "d", 2), "--json_path", _input(box["tmp"], "p", 0), "--output_dir", str(out)])
    err = capsys.readouterr().err
    assert rc == 0, err[-3000:]
    items = _lines(err, "ITEM")
    d = [l for l in items if " name=d " in l]; pl = [l for l in items if " name=p " in l]
    assert d and " ok=1 " in d[0] and " templates=0/2 " in d[0], items
    assert pl and " ok=1 " in pl[0] and " templates=none " in pl[0], items
    (t,) = _lines(err, "TEMPLATES")
    assert t == "[af3-torch-opt] TEMPLATES items=2 declared=2 live=0 refused=0 reason=None dummy_only=1 none=1 real=0/8 form=dense", t
    done = _lines(err, "DONE")[0]
    assert " items=2/2 failed=none " in done and " rc=0 " in done, done


def test_declared_counts_every_chain_copy(tmp_path):
    """cli.templates_declared: a protein entry naming several chain ids declares its templates once per copy (a homodimer with four
    templates declares eight — the units the model process counts live templates in); non-protein entries and null lists declare none."""
    import json as _json
    p = tmp_path / "h.json"
    p.write_text(_json.dumps({"name": "h", "modelSeeds": [1], "sequences": [
        {"protein": {"id": ["A", "B"], "sequence": "MKV", "templates": [{"mmcif": "", "queryIndices": [], "templateIndices": []}] * 4}},
        {"protein": {"id": "C", "sequence": "MKV", "templates": [{"mmcif": "", "queryIndices": [], "templateIndices": []}]}},
        {"protein": {"id": "D", "sequence": "MKV", "templates": None}},
        {"ligand": {"id": "L", "ccdCodes": ["ATP"]}}]}))
    assert cli.templates_declared(str(p)) == 4 * 2 + 1


def test_live_templates_are_counted_per_chain():
    """forward.template_entries_live: over every chain (asym_id) the slots whose atom mask has an atom on that chain's tokens, summed;
    template_slots_live stays per slot (the TEMPLATES line's real=<live slots>/<slots>)."""
    try:
        import numpy as np
    except ImportError:          # these tests run on a stdlib interpreter too; this count needs numpy (the model process always has it)
        return
    T, N = 4, 6
    mask = np.zeros((T, N, 24), dtype=np.float32)
    mask[0, :, 0] = 1            # slot 0: atoms on both chains
    mask[1, :3, 1] = 1           # slot 1: chain 1 only
    mask[3, 2, 5] = 1; mask[3, 4, 5] = 1   # slot 3: one atom on each chain; slot 2: dummy
    batch = {"template_atom_mask": mask, "asym_id": np.array([1, 1, 1, 2, 2, 2])}
    assert fwd.template_slots_total(batch) == 4 and fwd.template_slots_live(batch) == 3
    assert fwd.template_entries_live(batch) == 3 + 2                                   # chain 1: slots 0, 1, 3; chain 2: slots 0, 3
    padded = {"template_atom_mask": np.concatenate([mask, np.zeros((T, 2, 24), np.float32)], 1), "asym_id": np.array([1, 1, 1, 2, 2, 2, 0, 0])}
    assert fwd.template_entries_live(padded) == 5                                      # padding tokens (asym_id 0) are no chain
    assert fwd.template_entries_live({"template_atom_mask": mask}) == 3                 # no asym_id: the slot count
