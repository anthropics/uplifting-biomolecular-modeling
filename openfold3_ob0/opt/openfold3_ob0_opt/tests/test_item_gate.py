"""The offload port's item gate decided PER ITEM (cli.item_gate_groups / run_item_groups, modes.of3o_item_groups, inputs.polymer_tokens_each): under
`pred --mode big` a query set whose members fall on both sides of OF3O_MIN_TOKENS runs as one `pred` pass per side — below-gate items on the line a
below-gate set resolves to (modes.of3o_aside), the rest on the port —; a set on one side runs as one pass exactly as written; exact / fast / off and the
row-sharded line never split. CPU contracts: the partition, the subset documents, the passes' argv and environment, the summary lines, the rc rule."""
import json
import os
import re
import sys

import pytest

from openfold3_ob0_opt import cli, inputs, modes, report
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
RESIDENT = modes.LINES[("big", "resident")]
GATE = modes.OF3O_GATE_DEFAULT


def _query(n_res, ligands=0):
    chains = [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "M" * n_res}]
    chains += [{"molecule_type": "ligand", "chain_ids": [c], "ccd_codes": ["ATP"]} for c in "LMNOP"[:ligands]]
    return {"chains": chains}


def _set(**sizes):
    return {"seeds": [42], "queries": {k: _query(v) for k, v in sizes.items()}}


def _write(tmp_path, doc, name="q.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def _ns(tmp_path, doc, mode="big", extra=(), name="q.json"):
    q = _write(tmp_path, doc, name)
    argv = ["pred", "--mode", mode, "--query-json", q, "--output-dir", str(tmp_path / "out"), "--ckpt", "/w.pt", "--num-diffusion-samples", "2",
            "--use-msa-server", "false", "--use-templates", "true", *extra]
    a = cli.build_parser().parse_args(argv)
    a._argv, a._environ = list(argv), {"X": "1"}
    return a


def test_per_query_token_counts_the_largest_and_the_subset_document():
    qs = _set(small=400, big=1500)
    qs["queries"]["lig"] = _query(120, ligands=2)
    each = inputs.polymer_tokens_each(qs)
    assert each == {"small": (400, 0), "big": (1500, 0), "lig": (120, 2)} and list(each) == ["small", "big", "lig"]
    assert inputs.polymer_tokens(qs) == (1500, 0) == max(each.values())                       # the largest member, as before (the graphs / conf gates)
    sub = inputs.subset(qs, ["lig", "small"])
    assert list(sub["queries"]) == ["small", "lig"] and sub["seeds"] == [42]                    # document order kept; every other top-level key unchanged
    assert inputs.polymer_tokens(sub) == (400, 0) and sub["queries"]["small"] == qs["queries"]["small"]
    with pytest.raises(inputs.InputError):
        inputs.subset(qs, ["nope"])


def test_each_item_is_judged_on_its_own_count_as_a_set_of_that_item_alone():
    env = {"MODEL_OPT_TARGET_GPU": "H100"}
    g = modes.of3o_item_groups(RESIDENT, {"a": 400, "b": 1500}, env)
    assert (g["below"], g["engaged"], g["min_tokens"], g["source"]) == (("a",), ("b",), modes.OF3O_GATE_BY_CARD["H100"], "card:H100")
    assert modes.of3o_item_groups(RESIDENT, {"a": 400, "b": 800}, env)["engaged"] == ()          # one side: nothing to split
    assert modes.of3o_item_groups(RESIDENT, {"a": 1500, "b": 2000}, env)["below"] == ()
    assert modes.of3o_item_groups(RESIDENT, {"a": GATE - 1, "b": GATE}, {}) == {"below": ("a",), "engaged": ("b",), "tokens": {"a": GATE - 1, "b": GATE}, "min_tokens": GATE, "source": "default"}
    assert modes.of3o_item_groups(RESIDENT, {"a": 400, "b": 1500}, {"OF3O_MIN_TOKENS": "none"})["below"] == ()      # the knob `none`: every item on the port
    e = modes.of3o_item_groups(RESIDENT, {"a": 400, "b": 600, "c": 1500}, {"OF3O_MIN_TOKENS": "500"})              # the knob applies per item the same way
    assert (e["below"], e["engaged"], e["min_tokens"], e["source"]) == (("a",), ("b", "c"), 500, "env")
    assert modes.of3o_item_groups(RESIDENT, {"z": 0, "a": 400}, {})["engaged"] == ("z",)         # no polymer count: unknown -> the port (the memory fail-safe, as that item alone resolves)
    others = [k for k in modes.LINES if k != ("big", "resident")]
    assert others and all(modes.of3o_item_groups(modes.LINES[k], {"a": 400, "b": 1500}, {}) is None for k in others)   # no offload hook on the line: nothing to decide
    for n, names in ((400, g["below"]), (1500, g["engaged"])):                                     # each side runs what a homogeneous set of its items resolves to today
        side = modes.effective_line("big", "resident", env, n)
        assert (side == modes.of3o_aside(RESIDENT)) is (n < modes.OF3O_GATE_BY_CARD["H100"]) and (side is RESIDENT) is (n >= modes.OF3O_GATE_BY_CARD["H100"])


def test_the_summary_line_grammar():
    g = modes.of3o_item_groups(RESIDENT, {"a": 400, "b": 1500, "c": 2000}, {"MODEL_OPT_TARGET_GPU": "H100"})
    line = modes.item_gate_line(g)
    assert line == f"ITEM GATE: below=1 (400) line={modes.OF3O_ASIDE_WORD} | engaged=2 (1500,2000) line=resident gate={modes.OF3O_GATE_BY_CARD['H100']} source=card:H100"
    assert re.fullmatch(r"ITEM GATE: below=\d+ \([\d,]+\) line=[\w-]+ \| engaged=\d+ \([\d,]+\) line=\w+ gate=(\d+|none) source=(card:\w+|env|default)", line)
    assert modes.ITEM_GATE_NAME == "ITEM GATE" and cli.ITEM_GROUP_ORDER == ("below", "engaged") and cli.ITEM_GROUP_LINE["below"] == modes.OF3O_ASIDE_WORD


def test_the_query_json_flag_is_swapped_in_every_form_argparse_accepts():
    for argv in (["pred", "--query-json", "old.json", "--output-dir", "o"], ["pred", "--query_json", "old.json", "--output-dir", "o"],
                 ["pred", "--query-json=old.json", "--output-dir", "o"], ["pred", "--mode", "big", "--query-j", "old.json", "--output-dir", "o"]):
        out = cli.swap_query_json(argv, "new.json")
        assert out is not None and "old.json" not in " ".join(out) and out[-2:] == ["--output-dir", "o"]
        assert cli.build_parser().parse_args(out).query_json == "new.json"
    assert cli.swap_query_json(["pred", "--output-dir", "o"], "n.json") is None                      # absent / given twice: not a call this layer re-issues
    assert cli.swap_query_json(["pred", "--query-json", "a", "--query_json", "b"], "n.json") is None


def test_which_calls_split(tmp_path, monkeypatch):
    monkeypatch.delenv("OF3O_MIN_TOKENS", raising=False)
    monkeypatch.delenv("MODEL_OPT_TARGET_GPU", raising=False)
    a = _ns(tmp_path, _set(small=400, big=1500))
    g = cli.item_gate_groups(a, "big", "resident")
    assert g["below"] == ("small",) and g["engaged"] == ("big",) and g["query_set"]["seeds"] == [42] and g["line"] == "resident"
    assert all(cli.item_gate_groups(a, "big", tp) is None for tp in modes.TP_LINES)             # the row-sharded line: its launcher's, never split here
    for mode, el in (("fast", None), ("exact", "cueq"), ("off", None)):
        assert cli.item_gate_groups(_ns(tmp_path, _set(small=400, big=1500), mode=mode), mode, el) is None
    assert cli.item_gate_groups(_ns(tmp_path, _set(a=400, b=800)), "big", "resident") is None    # one side of the gate: one pass, as written
    assert cli.item_gate_groups(_ns(tmp_path, _set(a=1500, b=2000)), "big", "resident") is None
    assert cli.item_gate_groups(_ns(tmp_path, _set(only=400)), "big", "resident") is None
    b = _ns(tmp_path, _set(small=400, big=1500))
    del b._argv
    assert cli.item_gate_groups(b, "big", "resident") is None                                    # not issued through main(): no argv to re-issue -> one pass
    c = _ns(tmp_path, _set(small=400, big=1500))
    c.query_json = str(tmp_path / "missing.json")
    c._argv = cli.swap_query_json(c._argv, c.query_json)
    assert cli.item_gate_groups(c, "big", "resident") is None                                    # unreadable: upstream refuses it by name on the one-pass route
    monkeypatch.setenv("OF3O_MIN_TOKENS", "none")
    assert cli.item_gate_groups(a, "big", "resident") is None                                    # the knob `none`: every item engages -> one pass
    monkeypatch.setenv("OF3O_MIN_TOKENS", "500")
    d = _ns(tmp_path, _set(x=400, y=600))
    assert cli.item_gate_groups(d, "big", "resident")["engaged"] == ("y",)                       # the knob applies per item
    monkeypatch.setenv("OF3O_MIN_TOKENS", "many")
    assert cli.item_gate_groups(a, "big", "resident") is None                                    # a malformed knob: refused by name on the one-pass route, as before


def test_the_passes_reissue_the_call_once_per_side_with_the_subset(tmp_path, monkeypatch, capsys):
    doc = _set(small=400, big=1500)
    doc["queries"]["mid"] = _query(700)
    a = _ns(tmp_path, doc, extra=("--num-model-seeds", "1", "--det", "1"))
    calls = []

    def fake_call(cmd, env=None):
        qj = cmd[cmd.index("--query-json") + 1]
        with open(qj, encoding="utf-8") as fh:
            sub = json.load(fh)
        calls.append({"cmd": list(cmd), "env": env, "queries": list(sub["queries"]), "doc": sub, "qj": qj, "existed": os.path.isfile(qj)})
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    monkeypatch.setattr(cli, "group_structures", lambda out_dir, names: 2 * len(names))             # each pass wrote 1 seed x 2 samples per item
    monkeypatch.delenv("OF3O_MIN_TOKENS", raising=False)
    g = cli.item_gate_groups(a, "big", "resident")
    assert cli.run_item_groups(a, g) == 0
    assert [c["queries"] for c in calls] == [["small", "mid"], ["big"]]                              # the below-gate pass first, the port's last; document order inside a pass
    for c in calls:
        assert c["cmd"][:4] == [sys.executable, "-m", "openfold3_ob0_opt", "pred"] and c["existed"] and c["env"] == {"X": "1"}
        assert c["doc"]["seeds"] == [42] and all(c["doc"]["queries"][n] == doc["queries"][n] for n in c["queries"])   # the subset: the same document, fewer entries
        assert cli.swap_query_json(c["cmd"][3:], "Q") == cli.swap_query_json(a._argv, "Q")             # the same call but the query json: mode, seeds, samples, msa / template words, det, ckpt
    assert not os.path.exists(os.path.dirname(calls[0]["qj"]))                                       # the subset jsons are removed afterwards
    err = capsys.readouterr().err
    assert f"{report.PREFIX} ITEM GATE: below=2 (400,700) line={modes.OF3O_ASIDE_WORD} | engaged=1 (1500) line=resident gate={GATE} source=default" in err
    assert re.search(r"ITEM GATE pass 1/2 below: items=small,mid line=[\w-]+ query_json=\S+q\.below\.json", err)
    assert re.search(r"ITEM GATE pass 1/2 below: rc=0 wall_s=[\d.]+ structures=4/4", err)
    assert re.search(r"ITEM GATE pass 2/2 engaged: items=big line=resident query_json=\S+q\.engaged\.json", err)
    assert re.search(r"ITEM GATE pass 2/2 engaged: rc=0 wall_s=[\d.]+ structures=2/2", err)
    assert "ITEM GATE: passes=2 rc=0 (below=0 engaged=0) rollups=last-pass" in err        # no rollups written by the stub: the line says so


def test_the_worst_pass_rc_is_returned_and_a_short_pass_is_incomplete(tmp_path, monkeypatch, capsys):
    assert cli.worst_rc([0, 0]) == 0 and cli.worst_rc([0, 3]) == 3 and cli.worst_rc([5, 1]) == 1 and cli.worst_rc([3, 2]) == 2 and cli.worst_rc([1, -9]) == -9
    monkeypatch.delenv("OF3O_MIN_TOKENS", raising=False)
    a = _ns(tmp_path, _set(small=400, big=1500))
    rcs = iter([0, 3])
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, env=None: next(rcs))
    monkeypatch.setattr(cli, "group_structures", lambda out_dir, names: 0)                           # nothing written: the rc-0 pass is incomplete here (its own rule counts the whole directory)
    assert cli.run_item_groups(a, cli.item_gate_groups(a, "big", "resident")) == 1                 # incomplete (1) is worse than not active (3)
    err = capsys.readouterr().err
    assert "pass 1/2 below: rc=0 -> 1 (incomplete)" in err and "structures=0/2" in err and "pass 2/2 engaged: rc=3 wall_s=" in err
    assert "ITEM GATE: passes=2 rc=1 (below=1 engaged=3) rollups=last-pass" in err
    assert cli.group_structures(str(tmp_path / "none"), ["a", "b"]) == 0                              # no per-query directory yet: nothing counted


def test_cmd_pred_splits_only_a_straddling_big_set(tmp_path, monkeypatch):
    sentinel = 77
    monkeypatch.setattr(cli, "run_item_groups", lambda a, groups: sentinel)
    monkeypatch.setattr(cli, "weights_check", lambda *args, **kw: ({}, 9))                          # the one-pass route reaches the weights gate first: 9 = it took that route, one process
    for k in ("OPENFOLD3_OB0_OPT", "OF3O_MIN_TOKENS", "MODEL_OPT_TARGET_GPU", "OPENFOLD3_OB0_OPT_N_GPU"):
        monkeypatch.delenv(k, raising=False)
    two = _write(tmp_path, _set(small=400, big=1500), "two.json")
    one = _write(tmp_path, _set(a=400, b=800), "one.json")
    high = _write(tmp_path, _set(a=1500, b=2000), "high.json")
    base = ["--output-dir", str(tmp_path / "o"), "--ckpt", str(tmp_path / "w.pt")]
    assert cli.main(["pred", "--mode", "big", "--query-json", two, *base]) == sentinel              # both sides of the gate: the passes
    assert cli.main(["pred", "--mode", "big", "--query-json", one, *base]) == 9                     # below the gate: one pass, as written
    assert cli.main(["pred", "--mode", "big", "--query-json", high, *base]) == 9                    # at or above it: one pass, the port
    assert cli.main(["pred", "--mode", "fast", "--query-json", two, *base]) == 9                      # fast / exact: never split
    assert cli.main(["pred", "--mode", "exact", "--query-json", two, *base]) == 9
    monkeypatch.setenv("OF3O_MIN_TOKENS", "none")
    assert cli.main(["pred", "--mode", "big", "--query-json", two, *base]) == 9                     # the knob `none`: every item engages, one pass


_SUMMARY = ("\n" + "=" * 50 + "\n    PREDICTION SUMMARY (COMPLETE)    \n" + "=" * 50 + "\nTotal Queries Processed: {t}\n  - Successful Queries:  {s}\n  - Failed Queries:      {f}"
            "{names}\n" + "=" * 50 + "\n")                                                        # upstream's core/runners/writer.py _write_summary, byte for byte


def _upstream_rollups(out, names, failed=()):
    """What one `run_openfold predict` pass leaves at the top of the output directory for its query subset."""
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(_SUMMARY.format(t=len(names), s=len(names) - len(failed), f=len(failed), names=(f"\n\nFailed Queries: {', '.join(sorted(failed))}" if failed else "")))
    with open(os.path.join(out, "inference_query_set.json"), "w") as fh:
        json.dump({"seeds": [42], "queries": {n: {"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "M"}], "use_msas": True} for n in names}}, fh, indent=4)


def test_upstreams_run_rollups_describe_the_whole_call_after_the_passes(tmp_path, monkeypatch, capsys):
    doc = _set(small=400, big=1500)
    doc["queries"]["mid"] = _query(700)
    a = _ns(tmp_path, doc)
    out = str(tmp_path / "out")

    def fake_call(cmd, env=None):                                                                   # each pass overwrites the rollups with its own subset's, as upstream does
        with open(cmd[cmd.index("--query-json") + 1]) as fh:
            names = list(json.load(fh)["queries"])
        _upstream_rollups(out, names, failed=[n for n in names if n == "mid"])
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    monkeypatch.setattr(cli, "group_structures", lambda out_dir, names: 2 * len(names))
    monkeypatch.delenv("OF3O_MIN_TOKENS", raising=False)
    assert cli.run_item_groups(a, cli.item_gate_groups(a, "big", "resident")) == 0
    one = tmp_path / "one"
    _upstream_rollups(str(one), ["small", "big", "mid"], failed=["mid"])                              # what ONE pass over the whole set writes
    assert (tmp_path / "out" / "summary.txt").read_text() == (one / "summary.txt").read_text()        # the counts summed, upstream's format byte for byte
    merged = json.loads((tmp_path / "out" / "inference_query_set.json").read_text())
    assert list(merged["queries"]) == ["small", "big", "mid"] and merged["seeds"] == [42]              # the union, in the query file's order
    assert merged == json.loads((one / "inference_query_set.json").read_text())
    assert "rollups=summary.txt,inference_query_set.json" in capsys.readouterr().err
    # a pass that left no readable rollup (it died first): the last pass's copies stay as written, and the closing line says so
    b = _ns(tmp_path, _set(lo=400, hi=1500), name="b.json")
    b.output_dir = str(tmp_path / "out_b")
    b._argv[b._argv.index("--output-dir") + 1] = b.output_dir
    seq = iter([lambda names: None, lambda names: _upstream_rollups(b.output_dir, names)])
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, env=None: (next(seq)(list(json.load(open(cmd[cmd.index("--query-json") + 1]))["queries"])), 0)[1])
    assert cli.run_item_groups(b, cli.item_gate_groups(b, "big", "resident")) == 0
    assert json.loads(open(os.path.join(b.output_dir, "inference_query_set.json")).read())["queries"].keys() == {"hi"}
    assert "rollups=last-pass" in capsys.readouterr().err
    assert cli.read_run_rollups(str(tmp_path / "nowhere")) == {"summary": None, "query_set": None}


def test_the_pass_tally_reads_one_runner_yaml_or_the_flag_repeated(tmp_path):
    a = _ns(tmp_path, _set(small=400, big=1500), extra=("--num-model-seeds", "3"))
    q = a.query_json
    assert cli.pass_expected(q, a) == 2 * 3 * 2                                                     # 2 queries x 3 seeds x 2 samples (the flags)
    b = _ns(tmp_path, _set(small=400, big=1500))
    y1 = tmp_path / "one.yml"; y1.write_text("experiment_settings:\n  seeds: [1, 2, 3, 4]\n")
    y2 = tmp_path / "two.yml"; y2.write_text("model_update:\n  custom: {}\n")
    b.runner_yaml = str(y1)
    assert cli.pass_expected(q, b) == 2 * 4 * 2                                                     # one caller yaml with a seed list
    b.runner_yaml = [str(y2), str(y1)]
    assert cli.pass_expected(q, b) == 2 * 4 * 2                                                     # the flag repeated: the seed list in one of them counts
    b.runner_yaml = [str(tmp_path / "missing.yml")]
    assert cli.pass_expected(q, b) is None                                                          # unreadable: no count (the pass's own rule still gates)
    b.runner_yaml = None
    assert cli.pass_expected(q, b) == 2 * 1 * 2
