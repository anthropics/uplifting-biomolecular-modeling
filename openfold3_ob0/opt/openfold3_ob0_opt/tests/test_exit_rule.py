"""The exit rule of the CLI verbs (cli.exit_rule / cli.expected_structures): rc first; outputs short -> EXIT_FAIL `incomplete`; a partial
line -> EXIT_NOT_ACTIVE (a mode is all of its levers: no opt-out); the expected count = queries x seeds x samples from the query JSON, the settings and
the runner yaml (a seeds list in the yaml counts, an int is one seed, none is upstream's [42])."""
import json
import os

import pytest

from openfold3_ob0_opt.tests import _stubs
from openfold3_ob0_opt import cli, modes

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_exit_rule_branches():
    assert cli.exit_rule("t", 7, 10, 10, False, True)[0] == 7                                     # the runner's rc as it is
    code, gate = cli.exit_rule("t", 0, 3, 10, False, True)
    assert code == cli.EXIT_FAIL and gate["incomplete"] and gate["reason"].startswith("incomplete: 3/10")
    code, gate = cli.exit_rule("t", 0, 10, 10, True, False)                                         # a partial line: NOT ACTIVE by name (rc 3), no opt-out — a mode is all of its levers
    assert code == cli.EXIT_NOT_ACTIVE and gate["partial"] and gate["exit_code"] == cli.EXIT_NOT_ACTIVE and "allow_partial" not in gate
    code, gate = cli.exit_rule("t", 0, 10, 10, False, False)                                        # arm_complete False alone is partial too
    assert code == cli.EXIT_NOT_ACTIVE and gate["partial"]
    code, gate = cli.exit_rule("t", 0, 10, 10, False, True)
    assert code == cli.EXIT_OK and gate["reason"] == "ok"
    code, gate = cli.exit_rule("t", 0, 4, None, False, True)                                         # the count unknown: rc and the line only
    assert code == cli.EXIT_OK and not gate["incomplete"] and "unknown" in gate["reason"]
    assert cli.exit_rule("t", 0, 10, 10, False, None)[0] == cli.EXIT_OK                            # arm_complete None (stock) is not partial


def test_expected_structures(tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"queries": {"a": {}, "b": {}, "c": {}}}), encoding="utf-8")
    stock = os.path.join(HOME, modes.STOCK_YAML)
    assert cli.expected_structures(str(q), stock, None, None) == 3 * 1 * 5                         # upstream's seed list [42], 5 samples
    assert cli.expected_structures(str(q), stock, 5, 1) == 15                                       # --num-model-seeds 5 --num-diffusion-samples 1
    assert cli.expected_structures(str(q), _stubs.caller_yaml(tmp_path), None, 5) == 3 * 5 * 5        # a caller yaml's five-seed list
    assert cli.expected_structures(str(q), os.path.join(HOME, modes.KERNELS_OFF_YAML), None, 1) == 3 * 1 * 1   # a yaml without a seed list: upstream's [42]
    assert cli.expected_structures(str(tmp_path / "missing.json"), stock, None, None) is None


def test_no_partial_opt_out():
    """A partial line exits 3 by name and no verb takes an opt-out for it (a mode is all of its levers)."""
    ap = cli.build_parser()
    for verb, extra in (("pred", ["--query-json", "q", "--output-dir", "o"]), ("warm", ["--out", "o"])):
        with pytest.raises(SystemExit):
            ap.parse_args([verb, "--allow-partial"] + extra)


def test_no_record_file_beside_the_outputs_and_a_rank_hands_its_record_to_the_launcher(tmp_path, monkeypatch):
    """cli.gated_manifest writes NO file into the output directory; a rank of the row-sharded line (OF3TP_RANK) writes its run record as
    <OPENFOLD3_OB0_OPT_RECORDS>/rank<r>.json — the launcher's per-launch records directory (tp.launch reads rank 0's back and removes it)."""
    import json
    from openfold3_ob0_opt import manifest
    out = tmp_path / "out"; out.mkdir()
    rep = {"active": True, "mode": "fast", "line": "-", "levers_requested": [], "partial": False, "arm_complete": True}
    monkeypatch.delenv(manifest.RECORDS_ENV, raising=False); monkeypatch.delenv("OF3TP_RANK", raising=False)
    assert cli.gated_manifest(str(out), rep, "pred --mode fast", 0, None, False, command="pred", argv=[], checkpoint=None, det=0) == cli.EXIT_OK
    assert os.listdir(out) == []
    recs = tmp_path / "records"; recs.mkdir()
    monkeypatch.setenv(manifest.RECORDS_ENV, str(recs)); monkeypatch.setenv("OF3TP_RANK", "1")
    assert cli.gated_manifest(str(out), rep, "pred --mode big (tp rank 1)", 0, None, False, command="pred", argv=[], checkpoint=None, det=0) == cli.EXIT_OK
    rec = json.load(open(recs / "rank1.json"))
    assert os.listdir(out) == [] and rec["exit_code"] == 0 and rec["exit_rule"]["reason"].startswith("ok") and rec["schema"] == manifest.SCHEMA


def test_confidence_not_written_is_a_named_non_zero_exit():
    # structure first: 1 sample, upstream caught the heads death (rc 0), the structure-only file is on disk but NOT counted (n_cif 0) -> named EXIT_FAIL
    code, gate = cli.exit_rule("t", 0, 0, 1, False, True, conf_not_written=1)
    assert code == cli.EXIT_FAIL and gate["incomplete"] and gate["confidence_not_written"] == 1
    assert gate["reason"].startswith("incomplete: confidence not written for 1/1 structures")
    code, gate = cli.exit_rule("t", 0, 1, 2, False, True, conf_not_written=1)     # a 2-sample run that died in sample 2's heads: named too, before the plain count branch
    assert code == cli.EXIT_FAIL and gate["reason"].startswith("incomplete: confidence not written for 1/2")
    assert cli.exit_rule("t", 4, 0, 1, False, True, conf_not_written=1)[0] == 4      # the runner's rc still comes first


def test_structure_only_files_are_not_counted(tmp_path, monkeypatch):
    import json as _json
    from openfold3_ob0_opt import manifest
    seed = tmp_path / "q" / "seed_1"; seed.mkdir(parents=True)
    for k in (1, 2):
        (seed / f"q_seed_1_sample_{k}_model.cif").write_text("data_x\n#\n", encoding="utf-8")
    (seed / "q_seed_1_sample_2_structure_first.json").write_text(_json.dumps({"structure_file": "q_seed_1_sample_2_model.cif", "confidence_written": False}), encoding="utf-8")
    monkeypatch.setattr(manifest, "coordinate_defect", lambda path: None)
    assert manifest.count_structures(str(tmp_path)) == 1                                   # sample 2 is structure-only: not a structure of the tally
    blk = manifest.structure_first_block(str(tmp_path))
    assert blk["confidence_not_written"] == ["q_seed_1_sample_2_model.cif"] and blk["notes"] and blk["notes"][0].startswith("confidence not written: 1")
    (seed / "q_seed_1_sample_2_structure_first.json").write_text(_json.dumps({"structure_file": "q_seed_1_sample_2_model.cif", "confidence_written": True}), encoding="utf-8")
    assert manifest.count_structures(str(tmp_path)) == 2
