"""big_bf16_c16_predict.yml = the big lines' pinned-plan configuration (OFFLOAD_C16_YAML) plus upstream's bf16 trainer block (FAST_BF16_YAML's)."""
import os

import yaml

from openfold3_opt import modes

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_composition():
    doc = yaml.safe_load(open(os.path.join(HOME, modes.BIG_BF16_C16_YAML)))
    base = yaml.safe_load(open(os.path.join(HOME, modes.OFFLOAD_C16_YAML)))
    bf16 = yaml.safe_load(open(os.path.join(HOME, modes.FAST_BF16_YAML)))
    assert doc["model_update"] == base["model_update"]
    assert doc["pl_trainer_args"] == bf16["pl_trainer_args"] == {"precision": "bf16-mixed"}
    assert set(doc) == {"model_update", "pl_trainer_args"}


def test_big_lines_select_it():
    from openfold3_opt import cli
    want = os.path.join(HOME, modes.BIG_BF16_C16_YAML)
    for name in ("resident", "tp"):
        line = modes.LINES[("big", name)]
        assert line.runner_yaml == modes.BIG_BF16_C16_YAML, name
        assert cli.row_yaml(HOME, line, None, mode="big") == want, name           # P=1 (resident) and the --n_gpu 2|4|8 owner (tp)
    assert modes.LINES[("fast", None)].runner_yaml is None                                  # the fast line keeps its precision parameter's yamls
    assert cli.row_yaml(HOME, modes.LINES[("fast", None)], "bf16") == os.path.join(HOME, modes.FAST_BF16_YAML)
    import tempfile, yaml
    own = os.path.join(HOME, modes.KERNELS_OFF_YAML)                                           # a caller's --runner-yaml: composed under the big configuration — the member's keys (bf16-mixed, chunk 16, tuner off) laid on
    got = cli.row_yaml(HOME, modes.LINES[("big", "resident")], None, out_dir=tempfile.mkdtemp(), runner_yaml=own, mode="big")
    doc, member = yaml.safe_load(open(got)), yaml.safe_load(open(want))
    assert os.path.basename(got) == "runner_composed_" + os.path.basename(own) and doc["pl_trainer_args"] == member["pl_trainer_args"] and doc["model_update"]["custom"] == member["model_update"]["custom"]


def test_big_active_line_names_the_precision():
    from openfold3_opt import report
    res = modes.resolve("big", HOME, environ={}, n_tokens=1012)
    assert res.precision == "bf16-mixed"
    res2 = modes.resolve("big", HOME, environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OPT_N_GPU": "2"})
    assert res2.precision == "bf16-mixed"                                                     # the tp line (P > 1) too
    line = report.activation_line({"active": True, "mode": "big", "line": "resident", "openfold3_version": "0.4.1", "precision": res.precision, "levers_requested": [], "hooks": []})
    assert " precision=bf16-mixed" in line


def test_tp_pinned_yaml_keeps_the_precision_block(tmp_path):
    from openfold3_opt import tp
    out = tmp_path / "tp_predict.yml"
    tp.pinned_yaml(os.path.join(HOME, modes.BIG_BF16_C16_YAML), 16, str(out))
    doc = yaml.safe_load(open(out))
    assert doc["pl_trainer_args"] == {"precision": "bf16-mixed"}
