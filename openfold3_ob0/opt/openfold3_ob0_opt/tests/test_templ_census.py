"""templ_census.py: the TEMPLATES DECLARED line counts what a query declares (polymer chain entries with / without a template key, the
template files named and how many exist), prints once from the primary process, feeds the exit tally, and modifies nothing."""
import ast, inspect, io, os

import importlib


def _live():
    """The package's live modules (other tests re-import the package: a module object imported at collection time may be stale)."""
    return importlib.import_module("openfold3_ob0_opt.report"), importlib.import_module("openfold3_ob0_opt.templ_census")


def _qs(tmp_path, present=1, absent=1):
    files = []
    for i in range(present):
        p = tmp_path / f"t{i}.a3m"; p.write_text(">x\nAC\n"); files.append(str(p))
    files += [str(tmp_path / f"absent{i}.a3m") for i in range(absent)]
    chains = [{"molecule_type": "protein", "sequence": "ACD", "chain_ids": ["A", "B"], "template_alignment_file_path": files[0] if files else None},
              {"molecule_type": "protein", "sequence": "ACD", "chain_ids": ["C"], "template_cif_paths": files[1:]},
              {"molecule_type": "rna", "sequence": "ACGU", "chain_ids": ["D"]},
              {"molecule_type": "ligand", "smiles": "CCO", "chain_ids": ["L"]}]
    return {"queries": {"q1": {"chains": chains}, "q2": {"chains": [{"molecule_type": "dna", "sequence": "ACGT", "chain_ids": ["E"]}]}}}


def test_counts_files_and_the_line(tmp_path):
    report, templ_census = _live()
    qs = _qs(tmp_path, present=1, absent=2)
    c = templ_census.census(qs)
    assert (c["queries"], c["chains_templated"], c["chains_untemplated"], c["template_files"], c["template_files_resolved"]) == (2, 2, 2, 3, 1), c
    assert c["missing"] == [str(tmp_path / "absent0.a3m"), str(tmp_path / "absent1.a3m")]
    line = templ_census.line(c)
    assert line == (f"[openfold3_ob0-opt] TEMPLATES DECLARED queries=2 chains_templated=2 chains_untemplated=2 files_resolved=1/3 "
                    f"missing={tmp_path / 'absent0.a3m'},{tmp_path / 'absent1.a3m'}"), line
    none = templ_census.census({"queries": {"q": {"chains": [{"molecule_type": "protein", "sequence": "AC", "chain_ids": ["A"]}]}}})
    assert templ_census.line(none) == "[openfold3_ob0-opt] TEMPLATES DECLARED queries=1 chains_templated=0 chains_untemplated=1 files_resolved=0/0"


def test_record_prints_once_from_the_primary_process_and_feeds_the_exit_tally(tmp_path, monkeypatch):
    report, templ_census = _live()
    templ_census.reset()
    assert templ_census.counters() == {}
    buf = io.StringIO()
    c = templ_census.record(_qs(tmp_path, 1, 0), stream=buf)
    assert buf.getvalue().count("\n") == 1 and buf.getvalue().startswith("[openfold3_ob0-opt] TEMPLATES DECLARED queries=2 chains_templated=1 ")   # the second chain names no file with absent=0? see below
    assert templ_census.counters() == {"templ_declared": c["chains_templated"], "templ_untemplated": c["chains_untemplated"], "templ_files": c["template_files"], "templ_files_resolved": c["template_files_resolved"]}
    monkeypatch.setattr(report, "process_role", lambda: "worker")
    buf2 = io.StringIO(); templ_census.record(_qs(tmp_path, 1, 0), stream=buf2)
    assert buf2.getvalue() == ""                                                              # a worker takes the census silently
    templ_census.reset()
    q = tmp_path / "q.json"; q.write_text('{"queries": {"q": {"chains": [{"molecule_type": "protein", "sequence": "AC", "chain_ids": ["A"]}]}}}')
    monkeypatch.setattr(report, "process_role", lambda: "main")
    buf3 = io.StringIO()
    assert templ_census.record_file(str(q), stream=buf3)["chains_untemplated"] == 1 and "chains_untemplated=1" in buf3.getvalue()
    assert templ_census.record_file(str(tmp_path / "absent.json")) is None and templ_census.record_file(None) is None
    templ_census.reset()


def test_the_module_modifies_nothing_and_reads_no_switch():
    report, templ_census = _live()
    src = inspect.getsource(templ_census)
    assert "os.environ" not in src and "meta_path" not in src and "setattr" not in src          # counts only: no switch, no finder, no patch
    tree = ast.parse(src)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m and m.startswith("openfold3.") for m in imported), imported                # nothing of upstream is imported or wrapped
