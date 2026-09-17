"""`pred` takes upstream's query JSON as upstream takes it: no chain plays a role. A single chain, a homomer, a protein with RNA / DNA /
ligand (CCD or SMILES) entities and templated chains — hits files named per chain, two chains of one sequence naming different hits files —
all load, describe and pass the template census as information; the package has no `serve` verb, no pack format and no query adapter."""
import json
import os

import pytest

from opendde_opt import cli, inputs, templates

MONOMER = [{"name": "mono", "modelSeeds": [101], "sequences": [{"proteinChain": {"sequence": "MKTAYIAKQRQISFVKSHFSRQ", "count": 1}}]}]
HOMOMER = [{"name": "homo3", "sequences": [{"proteinChain": {"sequence": "MKTAYIAKQRQISFVKSHFSRQ", "count": 3, "unpairedMsaPath": "/m/a.a3m"}}]}]
MIXED = [{"name": "mixed", "modelSeeds": [7, 8], "sequences": [
    {"proteinChain": {"sequence": "MKVLAAGIDEKQ", "count": 1, "modifications": [{"ptmType": "CCD_SEP", "ptmPosition": 3}]}},
    {"rnaSequence": {"sequence": "GGGAAACCC", "count": 1}},
    {"dnaSequence": {"sequence": "ATGCATGC", "count": 2}},
    {"ligand": {"ligand": "CCD_ATP", "count": 1}},
    {"ligand": {"ligand": "CC(=O)Oc1ccccc1C(=O)O", "count": 1}}]}]
HITS2 = ">query/1-12\nMKVLAAGIDEKQ\n>1abc_A/3-14 mol:protein length:40\n--VLAAGIDEKQ\n>2xyz_B/1-10 mol:protein length:12\nMKVLAAGIDE--\n"
HITS1 = ">query/1-12\nMKVLAAGIDEKQ\n>1abc_A/3-14 mol:protein length:40\n--VLAAGIDEKQ\n"


def _write(tmp_path, name, jobs):
    p = tmp_path / f"{name}.json"; p.write_text(json.dumps(jobs)); return str(p)


@pytest.mark.parametrize("name,jobs,n_items", [("mono", MONOMER, 1), ("homo3", HOMOMER, 1), ("mixed", MIXED, 1), ("two_jobs", MONOMER + MIXED, 2), ("one_object", MONOMER[0], 1)])
def test_role_less_queries_load_as_upstream_reads_them(tmp_path, name, jobs, n_items):
    p = _write(tmp_path, name, jobs)
    loaded = inputs.load_query(p)
    assert len(loaded) == n_items and all("sequences" in j for j in loaded)
    d = inputs.describe_query(p)
    assert d["n_items"] == n_items and len(d["sha256"]) == 64
    assert templates.hits_of(loaded) == [] or all(not r["pdb_ids"] for r in templates.hits_of(loaded))   # no templatesPath: nothing templated, nothing refused


def test_templated_chains_of_one_sequence_may_name_different_hits(tmp_path):
    d = tmp_path / "t"; d.mkdir()
    (d / "a.hits.a3m").write_text(HITS2); (d / "b.hits.a3m").write_text(HITS1)
    jobs = [{"name": "tt", "sequences": [{"proteinChain": {"sequence": "MKVLAAGIDEKQ", "count": 1, "templatesPath": str(d / "a.hits.a3m")}},
                                          {"proteinChain": {"sequence": "MKVLAAGIDEKQ", "count": 1, "templatesPath": str(d / "b.hits.a3m")}}]}]
    rows = templates.hits_of(inputs.load_query(_write(tmp_path, "tt", jobs)))
    assert [len(r["pdb_ids"]) for r in rows] == [2, 1]
    c = templates.census(["Found 2 templates for sequence MKVLAAGIDEKQ", "Found 1 templates for sequence MKVLAAGIDEKQ"], rows)
    assert c["ok"] and c["notes"] == []                                                          # accepted, as upstream accepts it
    short = templates.census(["Found 1 templates for sequence MKVLAAGIDEKQ", "Found 1 templates for sequence MKVLAAGIDEKQ"], rows)
    assert not short["ok"] and len(short["notes"]) == 1                                          # a NOTE for the record — cli._templates_line prints it; no exit code reads it
    assert "_templates_verdict" not in open(cli.__file__).read() and "TEMPLATES refused" not in open(cli.__file__).read()


def test_the_package_has_no_serving_surface():
    assert set(cli.COMMANDS) == {"pred", "check", "warm"}
    assert not any(hasattr(inputs, n) for n in ("query_to_pack", "load_pack", "describe_pack"))
    pkg = os.path.dirname(cli.__file__)
    assert not os.path.exists(os.path.join(pkg, "serve.py")) and not os.path.exists(os.path.join(pkg, "_served_census.py"))
    import argparse
    ap = argparse.ArgumentParser(prog="pred"); cli._common_pred(ap)
    assert ap.parse_args(["-i", "q.json", "-o", "o"]).input == "q.json"
    for gone in ("--hash-checkpoint", "--json"):                                                  # removed pred toggles are usage errors, not silently accepted
        with pytest.raises(SystemExit) as e:
            ap.parse_args(["-i", "q.json", "-o", "o", gone])
        assert e.value.code == 2
