"""big_bf16_c16_predict.yml = the big lines' pinned-plan configuration: the kernels-off eval triple (upstream's Triton triangle kernels,
cuEquivariance and DS4Sci all off — 0.5.0 defaults the Triton kernels on, so the key is written), the chunk-size autotuner off, the pair-stack
chunk plan pinned at 16 rows, and upstream's bf16 trainer block; it is the runner yaml of both big compositions (resident, tp)."""
import os

import yaml

from openfold3_ob0_opt import modes
from openfold3_ob0_opt.tests import _stubs

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BIG_CHUNK = 16
KERNELS_OFF = {"use_triton_triangle_kernels": False, "use_cueq_triangle_kernels": False, "use_deepspeed_evo_attention": False}


def _eval(doc):
    return doc["model_update"]["custom"]["settings"]["memory"]["eval"]


def test_composition():
    doc = yaml.safe_load(open(os.path.join(HOME, modes.BIG_BF16_C16_YAML)))
    off = yaml.safe_load(open(os.path.join(HOME, modes.KERNELS_OFF_YAML)))
    assert set(doc) == {"model_update", "pl_trainer_args"}
    assert doc["pl_trainer_args"] == {"precision": "bf16-mixed"}
    ev = _eval(doc)
    assert ev == {**KERNELS_OFF, "tune_chunk_size": False, "chunk_size": BIG_CHUNK}, ev
    assert {k: v for k, v in _eval(off).items() if k in KERNELS_OFF} == KERNELS_OFF                      # the kernels-off configuration's own triple, key for key
    assert doc["model_update"]["presets"] == off["model_update"]["presets"] == ["predict"]


def test_big_lines_select_it():
    from openfold3_ob0_opt import cli
    want = os.path.join(HOME, modes.BIG_BF16_C16_YAML)
    for name in ("resident", "tp"):
        line = modes.LINES[("big", name)]
        assert line.runner_yaml == modes.BIG_BF16_C16_YAML, name
        assert cli.row_yaml(HOME, line, mode="big") == want, name                 # P=1 (resident) and the --n_gpu 2|4|8 owner (tp)
    assert modes.LINES[("fast", None)].runner_yaml == modes.STOCK_YAML                  # the fast line keeps the stock runner yaml
    own = os.path.join(HOME, modes.KERNELS_OFF_YAML)
    composed = cli.row_yaml(HOME, modes.LINES[("big", "resident")], runner_yaml=own, mode="big")     # a caller's --runner-yaml is composed UNDER the line's yaml
    assert os.path.basename(composed).startswith("runner_composed_big_resident_"), composed
    doc = yaml.safe_load(open(composed))
    assert _eval(doc) == {**KERNELS_OFF, "tune_chunk_size": False, "chunk_size": BIG_CHUNK}, _eval(doc)   # composes to the line's own eval block
    assert doc["pl_trainer_args"]["precision"] == "bf16-mixed"


def test_off_lays_the_big_yaml_over_the_stock_member():
    """`pred --mode off --runner-yaml <the big lines' yaml>` lays the file OVER the stock member (its keys win: every eval kernel flag, the chunk plan
    and the trainer precision it writes) — stock on the big lines' configuration (the resident line's equality arm), the same effective configuration
    the file alone states; `exact` composes it under the stock member like any caller yaml."""
    from openfold3_ob0_opt import cli
    want = os.path.join(HOME, modes.BIG_BF16_C16_YAML)
    for det in (0, 1):
        got = cli.row_yaml(HOME, modes.LINES[("off", None)], runner_yaml=modes.BIG_BF16_C16_YAML, mode="off", det=det)
        assert os.path.basename(got).startswith(cli.COMPOSED_PREFIX + "off_"), got
        assert _eval(yaml.safe_load(open(got))) == _eval(yaml.safe_load(open(want))) and yaml.safe_load(open(got))["pl_trainer_args"]["precision"] == "bf16-mixed"
    assert _eval(yaml.safe_load(open(want))) == {**KERNELS_OFF, "tune_chunk_size": False, "chunk_size": BIG_CHUNK}
    assert not cli.stock_family(HOME, modes.BIG_BF16_C16_YAML)                                          # not the stock family: `exact` composes it under the stock member
    ex = cli.row_yaml(HOME, modes.LINES[("exact", modes.EXACT_LINE)], runner_yaml=modes.BIG_BF16_C16_YAML, mode="exact", det=1)
    assert os.path.basename(ex).startswith("runner_composed_exact_"), ex


def test_big_active_line_names_the_precision():
    from openfold3_ob0_opt import report
    res = modes.resolve("big", HOME, environ={}, n_tokens=1012)
    assert res.precision == "bf16-mixed"
    res2 = modes.resolve("big", HOME, environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OB0_OPT_N_GPU": "2"}, n_gpu=2)
    assert res2.precision == "bf16-mixed"                                                     # the tp line (P > 1) too
    line = report.activation_line({"active": True, "mode": "big", "line": "resident", "openfold3_version": _stubs.PIN, "precision": res.precision, "levers_requested": [], "hooks": []})
    assert " precision=bf16-mixed" in line


def test_tp_pinned_yaml_over_the_c16_base(tmp_path):
    """The ×P launcher pins its plan over an already kernels-off base: the pinned document's eval kernel triple and precision equal the base's,
    only the chunk plan / tuner paths / trunk offload switches are the launcher's (tp.pins_overlay)."""
    from openfold3_ob0_opt import tp
    out = tmp_path / "tp_predict.yml"
    pins = tp.pinned_yaml(os.path.join(HOME, modes.BIG_BF16_C16_YAML), 32, str(out))
    doc = yaml.safe_load(open(out))
    assert doc["pl_trainer_args"] == {"precision": "bf16-mixed"}
    ev = _eval(doc)
    assert {k: ev[k] for k in KERNELS_OFF} == KERNELS_OFF and ev["use_lma"] is False and ev["tune_chunk_size"] is False and ev["chunk_size"] == 32
    assert [o for o in pins["overrides"] if "use_" in o] == [], pins["overrides"]                          # no kernel switch had to be overridden on this base
    assert any(o.startswith("model_update.custom.settings.memory.eval.chunk_size (16 -> 32)") for o in pins["overrides"]), pins["overrides"]
