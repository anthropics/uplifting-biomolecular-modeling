"""`--use_template true` is accepted on every route once its inputs are frozen (templates.py / frozen.template_problems): every protein
chain names a hits file, every hit's cif is where the engine opens it, upstream's two template assets are under the root, kalign is
executable; the model process's root is the weights root or an overlay whose search_database/mmcif is the query's cif directory; the
engine's `Found <n> templates` lines are counted against the hits the query names, on the stock route (transcript) and the kit lines
(the featurizer's logger) alike."""
import json
import logging
import os

import pytest

from opendde_opt import cli, frozen, settings, templates
from opendde_opt.tests import _stubs


def _flags(*stated):
    """settings.effective for a `pred` call stating ``stated`` (upstream's defaults otherwise) — the dict frozen.problems reads."""
    import argparse
    ap = argparse.ArgumentParser(); cli._common_pred(ap)
    return settings.effective(ap.parse_args(["-i", "q", "-o", "o", *stated]), [], _stubs.TREE)


HITS = """>query/1-12
MKVLAAGIDEKQ
>1abc_A/3-14 mol:protein length:40 some homolog
--VLAAGIDEKQ
>2xyz_B/1-10 mol:protein length:12
MKVLAAGIDE--
"""
HITS1 = """>query/1-12
MKVLAAGIDEKQ
>1abc_A/3-14 mol:protein length:40 some homolog
--VLAAGIDEKQ
"""


def _kalign(tmp_path, monkeypatch, present=True):
    b = tmp_path / "bin"; b.mkdir(exist_ok=True)
    if present:
        k = b / "kalign"; k.write_text("#!/bin/sh\nexit 0\n"); k.chmod(0o755)
    monkeypatch.setenv("PATH", str(b))


def _root(tmp_path, files=frozen.REQUIRED_FILES + frozen.TEMPLATE_FILES, cifs=("1abc", "2xyz"), obsolete=None):
    root = tmp_path / "w"
    for rel in files:
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"{}" if rel.endswith(".json") else b"x")
    if obsolete is not None:
        (root / templates.OBSOLETE_REL).write_text(json.dumps(obsolete))
    m = root / templates.MMCIF_REL; m.mkdir(parents=True, exist_ok=True)
    for c in cifs:
        (m / f"{c}.cif").write_text("data_x\n")
    return str(root)


def _query(tmp_path, chains, name="q"):
    d = tmp_path / name; d.mkdir(exist_ok=True)
    seqs = []
    for i, (seq, hits) in enumerate(chains):
        ch = {"sequence": seq, "unpairedMsaPath": "/m/u.a3m", "count": 1}
        if hits is not None:
            hp = d / "templates" / f"{i}.hits.a3m"; hp.parent.mkdir(parents=True, exist_ok=True); hp.write_text(hits)
            ch["templatesPath"] = str(hp)
        seqs.append({"proteinChain": ch})
    q = d / f"{name}.json"
    q.write_text(json.dumps([{"name": name, "sequences": seqs, "modelSeeds": [101]}]))
    return str(q)


def test_use_template_is_upstreams_flag_read_by_one_rule_on_every_mode():
    """`--use_template true` (stated on `pred`, or passed after `--`) is the whole switch: the frozen gate and cli read it through settings.effective /
    frozen._flag, the same on the stock caller and every kit line."""
    eff = _flags()
    assert eff["use_template"] == "false" and not frozen._truthy(frozen._flag("use_template", eff, []))          # upstream's default
    assert frozen._truthy(frozen._flag("use_template", eff, ["--use_template", "true"]))
    assert frozen._truthy(frozen._flag("use_template", _flags("--use_template", "true"), []))


def test_a_flag_after_the_separator_is_one_statement_in_either_spelling():
    """`-- --use_template=true` and `-- --use_template true` are the same statement: settings.effective (the cli's reading, the PRED line) agrees
    with frozen._flag (the gate's reading) in both spellings, the last occurrence winning across spellings; the same for every upstream flag read
    after `--` (e.g. `--dtype=fp32` is the caller's dtype: the stock base's pair is not added beside it)."""
    import argparse
    ap = argparse.ArgumentParser(); cli._common_pred(ap)
    a0 = ap.parse_args(["-i", "q", "-o", "o"])
    for extra in (["--use_template", "true"], ["--use_template=true"], ["--cycle", "4", "--use_template=true"], ["--use_template=True", "--cycle=4"]):
        eff = settings.effective(a0, extra, _stubs.TREE)
        assert frozen._truthy(eff["use_template"]) and frozen._truthy(frozen._flag("use_template", eff, extra)), extra
    assert settings.effective(a0, ["--use_template=false", "--use_template", "true"], _stubs.TREE)["use_template"] == "true"
    assert settings.effective(a0, ["--use_template", "true", "--use_template=false"], _stubs.TREE)["use_template"] == "false"
    assert settings.effective(a0, ["--cycle=4"], _stubs.TREE)["cycle"] == "4" and settings.effective(a0, ["-c", "4"], _stubs.TREE)["cycle"] == "4"
    assert settings.effective(a0, ["--dtype=fp32"], _stubs.TREE)["dtype"] == "fp32" and settings.base_args(a0, ["--dtype=fp32"]) == []
    assert settings.effective(a0, ["--use_template"], _stubs.TREE)["use_template"] == "false"        # a bare trailing flag states nothing


def test_use_template_is_accepted_when_its_inputs_are_frozen(tmp_path, monkeypatch):
    _kalign(tmp_path, monkeypatch)
    root = _root(tmp_path)
    q = _query(tmp_path, [("MKVLAAGIDEKQ", HITS), ("GGSGG", ">query/1-5\nGGSGG\n")])           # chain 1: a query-only hits file = no template, spelled by a file
    assert frozen.problems(root, q, _flags("--use_template", "true"), []) == []
    assert frozen.problems(root, q, _flags(), ["--use_template", "true"]) == []                 # the pass-through flag: the same rule
    rows = templates.hits_of(json.load(open(q)))
    assert [r["pdb_ids"] for r in rows] == [["1abc", "2xyz"], []]
    assert templates.named(rows) == {"MKVLAAGIDEKQ": {2}}


def test_every_template_precondition_is_named(tmp_path, monkeypatch):
    _kalign(tmp_path, monkeypatch, present=False)
    root = _root(tmp_path, files=frozen.REQUIRED_FILES, cifs=("1abc",))                      # no template assets, one cif of two
    q = _query(tmp_path, [("MKVLAAGIDEKQ", HITS), ("GGSGG", None)])                            # chain 1 without templatesPath
    p = frozen.problems(root, q, _flags("--use_template", "true"), [])
    heads = [x.split(":")[0] for x in p]
    assert heads[:2] == ["common/release_date_cache.json absent under " + root + " (stock downloads it under --use_template true",
                         "common/obsolete_to_successor.json absent under " + root + " (stock downloads it under --use_template true"], heads
    assert heads[2] == "kalign absent" and "runner/batch_inference.py:494-497" in p[2]
    assert heads[3] == "task 'q' sequences[1]" and "without templatesPath" in p[3] and "template_search.py:187-241" in p[3]
    assert heads[4] == "task 'q' sequences[0]" and "template 2xyz" in p[4] and p[4].endswith("(the stock fetches it from PDBe: template_utils.py:407-427)") and len(p) == 5
    assert frozen.problems(root, q, _flags(), []) == []                      # use_template false: none of it is read
    # the hit id is mapped through upstream's obsolete map first: the SUCCESSOR's cif is the one the engine opens
    _kalign(tmp_path, monkeypatch)
    root2 = _root(tmp_path / "r2", obsolete={"2XYZ": "9ZZZ"})
    q2 = _query(tmp_path, [("MKVLAAGIDEKQ", HITS)], name="q2")
    p2 = frozen.problems(root2, q2, _flags("--use_template", "true"), [])
    assert len(p2) == 1 and "template 2xyz" in p2[0] and p2[0].split(": ")[2].endswith("9zzz.cif absent (the stock fetches it from PDBe"), p2
    open(os.path.join(root2, templates.MMCIF_REL, "9zzz.cif"), "w").write("data_y\n")
    assert frozen.problems(root2, q2, _flags("--use_template", "true"), []) == []
    hp = tmp_path / "bad.txt"; hp.write_text(HITS)
    with pytest.raises(templates.TemplateInputError):
        templates.pdb_ids_of(str(hp))


def test_the_cif_directory_is_one_and_the_model_root_overlays_it(tmp_path):
    root = _root(tmp_path, cifs=())
    q = _query(tmp_path, [("MKVLAAGIDEKQ", HITS), ("MKV", HITS)])
    rows = templates.hits_of(json.load(open(q)))
    d = templates.mmcif_dir_of(rows, None)
    assert d == str(tmp_path / "q" / "templates")                                             # the hits files' own directory: <dir>/<pdb>.cif beside them
    (tmp_path / "q" / "templates" / "mmcif").mkdir()
    assert templates.mmcif_dir_of(rows, None) == str(tmp_path / "q" / "templates" / "mmcif")   # … or its mmcif/ subdirectory when that exists
    assert templates.mmcif_dir_of(rows, str(tmp_path / "elsewhere")) == str(tmp_path / "elsewhere")
    rows2 = rows + [dict(rows[0], path=str(tmp_path / "other" / "x.hits.a3m"))]
    with pytest.raises(templates.TemplateInputError):
        templates.mmcif_dir_of(rows2, None)
    open(os.path.join(d, "1abc.cif"), "w").write("data_x\n")                                  # a cif beside the hits files
    out = tmp_path / "o"; out.mkdir()
    mr = templates.model_root(root, d, str(out))
    assert mr == str(out / templates.OVERLAY_NAME) and os.path.islink(os.path.join(mr, "checkpoint")) and os.path.islink(os.path.join(mr, "common"))
    assert os.path.realpath(os.path.join(mr, templates.MMCIF_REL)) == os.path.realpath(d)
    assert os.path.isfile(os.path.join(mr, "checkpoint", "opendde.pt")) and os.path.isfile(os.path.join(mr, templates.MMCIF_REL, "1abc.cif"))
    assert templates.model_root(root, d, str(out)) == mr                                     # rebuilt in place
    assert templates.model_root(root, os.path.join(root, templates.MMCIF_REL), str(out)) == root   # the weights root's own directory: no overlay
    assert templates.model_root(root, None, str(out)) == root
    assert [x[2] for x in templates.missing_cifs(rows, os.path.join(mr, templates.MMCIF_REL), {})] == ["2xyz", "2xyz"]


def test_the_engines_template_count_is_the_census(tmp_path):
    rows = templates.hits_of(json.load(open(_query(tmp_path, [("MKVLAAGIDEKQ", HITS), ("GGSGG", None)]))))
    line = "2026-09-06 [template_featurizer.py:314] INFO opendde.data.template.template_featurizer: Found {n} templates for sequence {s}"
    ok = templates.census([line.format(n=2, s="MKVLAAGIDEKQ"), line.format(n=0, s="GGSGG"), "noise"], rows)
    assert ok["ok"] and ok["sequences"] == {"MKVLAAGIDEKQ": {"named": [2], "found": [2]}} and ok["notes"] == []
    assert templates.census([line.format(n=2, s="MKVLAAGIDEKQ")] * 4, rows)["ok"]               # the same sequence reported more than once (several tasks / chains): every report must equal the named count
    short = templates.census([line.format(n=1, s="MKVLAAGIDEKQ")], rows)                     # upstream dropped a hit (parser / date / duplicate prefilter): a NOTE, never an exit code
    assert not short["ok"] and short["notes"] == ["sequence MKVLAAGIDEKQ: named [2] templates, the engine found [1]"]
    none = templates.census([], rows)
    assert not none["ok"] and "reported none" in none["notes"][0]
    two = templates.hits_of(json.load(open(_query(tmp_path, [("MKVLAAGIDEKQ", HITS), ("MKVLAAGIDEKQ", HITS1)]))))   # two chains of one sequence naming different hits files: upstream accepts it, so does the census
    same = templates.census([line.format(n=2, s="MKVLAAGIDEKQ"), line.format(n=1, s="MKVLAAGIDEKQ")], two)
    assert same["ok"] and same["sequences"]["MKVLAAGIDEKQ"] == {"named": [1, 2], "found": [2, 1]} and same["notes"] == []
    root_handlers = list(logging.getLogger().handlers)
    c = templates.FoundCollector().attach()
    try:
        logging.getLogger(templates.LOGGER_NAME).warning("Found 2 templates for sequence MKVLAAGIDEKQ")
        logging.getLogger("elsewhere").warning("Found 9 templates for sequence MKVLAAGIDEKQ")
    finally:
        c.detach()
    assert c.lines == ["Found 2 templates for sequence MKVLAAGIDEKQ"] and logging.getLogger().handlers == root_handlers
    assert templates.census(c.lines, rows)["ok"]


def test_pred_no_longer_refuses_use_template_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENDDE_ROOT_DIR", raising=False); monkeypatch.delenv("OPENDDE_OPT", raising=False)
    _kalign(tmp_path, monkeypatch)
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ.get("PATH", ""))
    q = _query(tmp_path, [("MKVLAAGIDEKQ", HITS)])
    rc = cli.main(["pred", "--mode", "off", "-i", q, "-o", str(tmp_path / "o"), "--use_template", "true"])
    io = capsys.readouterr()
    assert rc == cli.EXIT_NOT_ACTIVE and "use_template=true:" not in io.err                  # refused for the unset root and the absent cifs, never for the flag
    assert "OPENDDE_ROOT_DIR is not set" in io.err and "template 1abc" in io.err and "template 2xyz" in io.err
    assert "TEMPLATES use_template=true protein_chains=1 templated=1 hits_named=2" in io.out


def test_the_census_names_real_slots_and_form():
    """C5 words: `real=<r>/<T>` = template slots holding a real hit (largest `Found <n>` over the templated sequences, capped at the
    featurizer's 4 slots), `form=dense` for one model process."""
    rows = [{"task": "t", "chain": "A", "sequence": "SEQA", "path": "a.hits.a3m", "pdb_ids": ["1abc", "2abc"]},
            {"task": "t", "chain": "B", "sequence": "SEQB", "path": "b.hits.a3m", "pdb_ids": []}]
    c = templates.census(["Found 2 templates for sequence SEQA"], rows)
    assert (c["ok"], c["real"], c["slots"], c["form"]) == (True, 2, 4, "dense")
    c = templates.census(["Found 9 templates for sequence SEQA"], rows)                     # capped at the featurizer's slots; the count difference is a NOTE
    assert (c["ok"], c["real"], c["slots"]) == (False, 4, 4)
    assert templates.census([], rows)["real"] == 0


def test_rank_transcripts_are_one_census_keyed_on_rank_0():
    """--n_gpu P: rank 0 alone runs the featurizer (data_form=rank0_bcast): its `Found <n>` lines are the counts; a receiver reports none — one that
    does ran a featurizer of its own (NOTE, ok False); form=row_born only when every rank's rowpair census names its template rows as born."""
    from opendde_opt import tp
    rows = [{"task": "t", "chain": "A", "sequence": "SEQA", "path": "a.hits.a3m", "pdb_ids": ["1abc"]}]
    born = f"[opendde-opt] LEVER name=rowpair_tp ... inputs {templates.BORN_WORD}(R=32,N=61):0.001GB/full=0.002GB"
    logs = {0: f"Found 1 templates for sequence SEQA\n{born}\n", 1: f"{born}\n"}                       # rank 1 received rank 0's features: no `Found` line, its rows born
    c = templates.census_ranks(logs, rows)
    assert (c["ok"], c["real"], c["form"], c["notes"]) == (True, 1, "row_born", [])
    c = templates.census_ranks({0: logs[0], 1: f"Found 1 templates for sequence SEQA\n{born}\n"}, rows)   # a receiver that ran a featurizer of its own: named, ok False
    assert c["ok"] is False and any("rank 1 reported {'SEQA': [1]}" in n and "rank 0 alone featurises" in n for n in c["notes"]), c["notes"]
    c = templates.census_ranks({0: f"Found 0 templates for sequence SEQA\n{born}\n", 1: logs[1]}, rows)   # rank 0 found fewer than named: the census NOTE (as on one GPU)
    assert c["ok"] is False and any("sequence SEQA" in n for n in c["notes"])
    c = templates.census_ranks({0: logs[0], 1: ""}, rows)                                                # a rank without the born word: form unverified, named
    assert c["form"] == "unverified" and any("ranks [1]" in n for n in c["notes"]) and c["ok"] is True
    c = templates.census_ranks({1: logs[1]}, rows)                                                       # no rank-0 transcript: named, ok False
    assert c["ok"] is False and any("rank 0 left no transcript" in n for n in c["notes"])
    tp.STATS["template_rows_born"], tp.STATS["layout"] = 3, {"R": 32, "N": 61}                             # the word census_ranks keys on IS the line's census word
    try:
        assert templates.BORN_WORD in tp._inputs_word()
    finally:
        tp.STATS["template_rows_born"], tp.STATS["layout"] = 0, None


def test_rank_processes_are_handed_the_weights_root_and_the_cif_directory_by_flag(tmp_path):
    """--n_gpu P under --use_template true: every rank's command states the WEIGHTS root (never the parent's overlay) and the one cif
    directory the parent resolved; without templates the ranks' command is unchanged."""
    import argparse
    a = argparse.Namespace(input=str(tmp_path / "q.json"), det=0, root=None, allow_partial=False, dtype=None, cycle=None, step=None, sample=None,
                           model_name=None, use_msa=None, use_template=None, use_rna_msa=None, need_atom_confidence=None, seeds=None)
    plain = cli.rank_argv(a, 2, str(tmp_path), ["--use_template", "true"])(1)
    templ = cli.rank_argv(a, 2, str(tmp_path), ["--use_template", "true"], templates=([], "/cifs", "/weights", str(tmp_path / "opendde_root")))(1)
    assert "--root" not in plain and "--template_mmcif_dir" not in plain
    assert templ[templ.index("--root") + 1] == "/weights" and templ[templ.index("--template_mmcif_dir") + 1] == "/cifs"
    assert templ[-3:] == ["--", "--use_template", "true"] and plain[-3:] == templ[-3:]
    assert [x for x in templ if x not in ("--root", "/weights", "--template_mmcif_dir", "/cifs")] == plain   # nothing else moves
