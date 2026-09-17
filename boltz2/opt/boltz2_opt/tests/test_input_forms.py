"""`--input` takes what `boltz predict` takes: a YAML or FASTA file, or a directory of them (boltz/main.py check_inputs) — on the kit modes one
or more such paths (worker.expand_inputs / items_from_yamls), on `off` one path per process with every record of it accounted
(cli.stock_unit_account)."""
import json
import os

import pytest

from .. import cli, worker


def _touch(p, text="version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKV\n      msa: empty\n"):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(text); return p


def test_expand_inputs_is_boltz_predicts_check_inputs(tmp_path):
    d = str(tmp_path / "in"); a = _touch(os.path.join(d, "a.yaml")); f = _touch(os.path.join(d, "b.fasta"), ">A|protein|empty\nMKV\n"); y2 = _touch(os.path.join(d, "c.yml"))
    lone = _touch(str(tmp_path / "lone.yaml"))
    assert worker.expand_inputs([lone, d]) == [lone, a, f, y2], "a file is itself; a directory is its entries, sorted"
    items = worker.items_from_yamls([d])
    assert [it["name"] for it in items] == ["a", "b", "c"] and items[1]["yaml"] == os.path.abspath(f), "the record id is the file stem, FASTA included"
    _touch(os.path.join(d, "notes.txt"), "x")
    with pytest.raises(ValueError, match="Unable to parse filetype .txt"):        # boltz predict's own refusal of another file type in the directory
        worker.expand_inputs([d])
    os.remove(os.path.join(d, "notes.txt")); os.makedirs(os.path.join(d, "sub"))
    with pytest.raises(ValueError, match="instead of .fasta or .yaml"):          # ... and of a nested directory
        worker.expand_inputs([d])
    with pytest.raises(FileNotFoundError):
        worker.expand_inputs([str(tmp_path / "absent.yaml")])
    with pytest.raises(ValueError, match="share the record name"):
        worker.items_from_yamls([lone, _touch(str(tmp_path / "x" / "lone.fasta"), ">A|protein|empty\nMKV\n")])


def test_the_stock_route_accounts_every_record_of_a_directory_input(tmp_path):
    """stock_unit_account over a directory: boltz writes boltz_results_<dir name>/predictions/<record>/ per file; each record is ok, skipped
    (no record of that id in processed/manifest.json) or outputs short — counts close."""
    d = str(tmp_path / "set"); _touch(os.path.join(d, "p.yaml")); _touch(os.path.join(d, "q.fasta"), ">A|protein|empty\nMKV\n"); _touch(os.path.join(d, "r.yaml"))
    out = str(tmp_path / "o"); res = os.path.join(out, "boltz_results_set")
    os.makedirs(os.path.join(res, "predictions", "p")); open(os.path.join(res, "predictions", "p", "p_model_0.cif"), "w").write("x")
    os.makedirs(os.path.join(res, "processed")); json.dump({"records": [{"id": "p"}, {"id": "r"}]}, open(os.path.join(res, "processed", "manifest.json"), "w"))
    acct = cli.stock_unit_account(out, d, 5, 0)
    assert (acct["expected"], acct["ok"], acct["failed"]) == (3, 1, 2) and acct["status"] == "incomplete"
    why = {u["name"]: u["reason"] for u in acct["failed_units"]}
    assert why["q"].startswith("stock parser skipped the input") and why["r"].startswith("outputs short") and [s["name"] for s in acct["skipped"]] == ["q"] and not acct["all_skipped"]
    one = _touch(str(tmp_path / "one.fasta"), ">A|protein|empty\nMKV\n")                     # a FASTA file on off: its stem is the record, as for a YAML
    res1 = os.path.join(out, "boltz_results_one"); os.makedirs(os.path.join(res1, "predictions", "one")); open(os.path.join(res1, "predictions", "one", "one_model_0.cif"), "w").write("x")
    assert cli.stock_unit_account(out, one, 5, 0)["status"] == "complete"
