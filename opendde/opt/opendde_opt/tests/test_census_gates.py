"""The frozen-weights boot gate: every stock download site named and closed before the stock process starts."""
import json


from opendde_opt import cli, frozen, settings
from opendde_opt.tests import _stubs


def _flags(*stated):
    """settings.effective for a `pred` call stating ``stated`` (upstream's defaults otherwise) — the dict frozen.problems reads."""
    import argparse
    ap = argparse.ArgumentParser(); cli._common_pred(ap)
    return settings.effective(ap.parse_args(["-i", "q", "-o", "o", *stated]), [], _stubs.TREE)


def _root(tmp_path, *present):
    for rel in present:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return str(tmp_path)


def _query(tmp_path, chains, name="q"):
    q = tmp_path / f"{name}.json"
    q.write_text(json.dumps([{"name": "t1", "sequences": [{"proteinChain": c} for c in chains], "modelSeeds": [101]}]))
    return str(q)


def test_an_unset_root_and_every_absent_file_is_named(tmp_path):
    q = _query(tmp_path, [{"sequence": "MK", "unpairedMsaPath": "/m/u.a3m"}])
    rec = _flags()
    p = frozen.problems(None, q, rec, [])
    assert len(p) == 1 and "OPENDDE_ROOT_DIR is not set — see README Variables" in p[0] and "download.py:283-316" in p[0]
    root = _root(tmp_path / "w", "checkpoint/opendde.pt")
    p = frozen.problems(root, q, rec, [])
    assert [x.split(" absent")[0] for x in p] == ["common/components.cif", "common/components.cif.rdkit_mol.pkl"] and all("download.py:254-294" in x for x in p)
    assert frozen.problems(_root(tmp_path / "w", *frozen.REQUIRED_FILES), q, rec, []) == []
    # with the tree: upstream's size rule — a present CCD file of another size is a download upstream, refused here by name
    root = _root(tmp_path / "w", *frozen.REQUIRED_FILES)
    p = frozen.problems(root, q, rec, [], tree=_stubs.TREE)
    assert len(p) == 2 and all("upstream's manifest says" in x for x in p) and p[0].startswith("common/components.cif under "), p
    sizes = frozen.managed_sizes(_stubs.TREE)
    assert sizes["common/components.cif"] == 490777362 and sizes["common/components.cif.rdkit_mol.pkl"] == 142498117
    assert frozen.problems(_stubs.weights_root(str(tmp_path / "w2")), q, rec, [], tree=_stubs.TREE) == []     # the stub root carries the sizes (sparse)


def test_a_protein_chain_without_precomputed_msa_is_refused_under_use_msa(tmp_path):
    root = _root(tmp_path / "w", *frozen.REQUIRED_FILES); rec = _flags()
    q = _query(tmp_path, [{"sequence": "MK", "unpairedMsaPath": "/m/u.a3m"}, {"sequence": "MKV"}])
    p = frozen.problems(root, q, rec, [])
    assert len(p) == 1 and "sequences[1]" in p[0] and "MMseqs2 service" in p[0] and "msa_search.py:181-225" in p[0]
    q2 = _query(tmp_path, [{"sequence": "MK", "msa": {"precomputed_msa_dir": "/m/d", "pairing_db": "uniref100"}}], "q2")   # the old form is admitted (converted by the stock first)
    assert frozen.problems(root, q2, rec, []) == []
    assert frozen.problems(root, q, _flags("--use_msa", "false", "--sample", "1"), []) == []                                  # use_msa false: single sequence, no service
    assert frozen.problems(root, q, _flags("--use_msa", "false", "--sample", "1"), ["--use_msa", "true"]) != []               # a pass-through flag counts


def test_rna_msa_is_refused_and_templates_name_their_preconditions(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "nobin"))                                                # no kalign on this PATH
    root = _root(tmp_path / "w", *frozen.REQUIRED_FILES)
    q = _query(tmp_path, [{"sequence": "MK", "unpairedMsaPath": "/m/u.a3m"}])
    p = frozen.problems(root, q, _flags(), ["--use_template", "true", "--use_rna_msa=true"])
    heads = [x.split(":")[0] for x in p]                                                                # --use_template true is read, never refused as a flag:
    assert heads == ["common/release_date_cache.json absent under " + root + " (stock downloads it under --use_template true",   # its assets, kalign and the
                     "common/obsolete_to_successor.json absent under " + root + " (stock downloads it under --use_template true",  # chain's hits file are named
                     "kalign absent", "task 't1' sequences[0]", "use_rna_msa=true"], heads


def test_pred_refuses_before_the_stock_starts(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENDDE_ROOT_DIR", raising=False); monkeypatch.delenv("OPENDDE_OPT", raising=False)
    q = _query(tmp_path, [{"sequence": "MK"}])
    rc = cli.main(["pred", "--mode", "off", "-i", q, "-o", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_NOT_ACTIVE and "frozen-weights gate REFUSED (2 named)" in err and "OPENDDE_ROOT_DIR is not set" in err and "MMseqs2 service" in err
    assert not (tmp_path / "o" / "opt_manifest.json").exists()                                            # nothing ran, nothing written
