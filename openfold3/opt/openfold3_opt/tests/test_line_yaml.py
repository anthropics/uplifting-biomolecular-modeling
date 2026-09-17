"""The runner yaml a call runs under (cli.row_yaml): the mode's own member (cli.member_yaml), or a caller's `--runner-yaml` — verbatim under
`off`, composed under the line under a kit mode (runner_yaml.compose: the keys the mode requires laid on top and named)."""
import os

import pytest

from openfold3_opt import cli, modes
from openfold3_opt.tests import _stubs

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_row_yaml_precedence(tmp_path):
    import yaml
    ex = modes.LINES[("exact", "cueq")]
    assert ex.runner_yaml is None and modes.LINES[("fast", None)].runner_yaml is None          # no line names a yaml: the kernel flags follow the mode (cli.kernel_policy)
    assert cli.kernel_policy("off") == cli.kernel_policy("exact") == {"use_cueq_triangle_kernels": True, "use_deepspeed_evo_attention": True}
    assert cli.kernel_policy("off", 1) == cli.kernel_policy("exact", 2) == {"use_cueq_triangle_kernels": True, "use_deepspeed_evo_attention": False}
    assert cli.kernel_policy("fast") is None and cli.kernel_policy("big", 1) is None and cli.kernel_policy(None) is None
    assert cli.row_yaml(HOME) == os.path.join(HOME, modes.KERNELS_OFF_YAML)                     # no mode named: the kernels-off base
    assert cli.row_yaml(HOME, ex, mode="exact") == os.path.join(HOME, modes.STOCK_YAML) == cli.row_yaml(HOME, mode="off")
    assert cli.row_yaml(HOME, ex, mode="exact", det=1) == os.path.join(HOME, modes.STOCK_DET_YAML) == cli.row_yaml(HOME, mode="off", det=2)
    assert cli.row_yaml(HOME, modes.LINES[("fast", None)], "fp32", mode="fast", det=1) == os.path.join(HOME, modes.KERNELS_OFF_YAML)
    own = _stubs.caller_yaml(tmp_path)                                                               # a caller's --runner-yaml: an OVERLAY on every route — composed under the stock configuration under off, under the mode's otherwise; the caller's file untouched
    off = cli.row_yaml(HOME, runner_yaml=own, out_dir=str(tmp_path / "o"), mode="off")
    assert off == str(tmp_path / "o" / ("runner_composed_" + os.path.basename(own))) and yaml.safe_load(open(own)) == yaml.safe_load(open(_stubs.caller_yaml(tmp_path / "again")))
    off_doc, stock_doc = yaml.safe_load(open(off)), yaml.safe_load(open(os.path.join(HOME, modes.STOCK_YAML)))
    assert off_doc["pl_trainer_args"] == stock_doc["pl_trainer_args"] and off_doc["model_update"]["custom"]["settings"]["memory"]["eval"]["use_cueq_triangle_kernels"] is True   # stock's keys pinned on top
    assert off_doc["experiment_settings"]["seeds"] == [42, 66, 101, 2024, 8888] and off_doc["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 10       # the caller's other keys as written
    for kw, member in ((dict(line=ex, mode="exact", det=1), modes.STOCK_DET_YAML), (dict(line=modes.LINES[("big", "resident")], mode="big"), modes.BIG_BF16_C16_YAML),
                       (dict(line=modes.LINES[("fast", None)], precision="bf16", mode="fast"), modes.FAST_BF16_YAML)):
        got = cli.row_yaml(HOME, runner_yaml=own, out_dir=str(tmp_path / "o"), **kw)
        want = yaml.safe_load(open(os.path.join(HOME, member)))
        doc = yaml.safe_load(open(got))
        assert os.path.basename(got) == "runner_composed_" + os.path.basename(own) and doc["pl_trainer_args"]["precision"] == want["pl_trainer_args"]["precision"], kw   # the member's keys laid on
        assert yaml.safe_load(open(own)) == yaml.safe_load(open(_stubs.caller_yaml(tmp_path / "again2")))                                                                # the caller's file untouched
    assert cli.row_yaml(HOME, modes.LINES[("big", "resident")], mode="big") == os.path.join(HOME, modes.BIG_BF16_C16_YAML)   # a line's own yaml (the big lines)


def test_yaml_kernel_flag_read():
    assert cli.yaml_ds4sci_off(os.path.join(HOME, modes.KERNELS_OFF_YAML)) and cli.yaml_ds4sci_off(os.path.join(HOME, modes.BIG_BF16_C16_YAML)) and cli.yaml_ds4sci_off(os.path.join(HOME, modes.STOCK_DET_YAML))
    assert not cli.yaml_ds4sci_off(os.path.join(HOME, modes.SHIPPED_YAML)) and not cli.yaml_ds4sci_off(os.path.join(HOME, modes.STOCK_YAML))
    on = lambda p: tuple(f for f, v in cli.yaml_kernel_flags(os.path.join(HOME, p)).items() if v)
    assert on(modes.STOCK_YAML) == ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels") and on(modes.STOCK_DET_YAML) == ("use_cueq_triangle_kernels",) and on(modes.KERNELS_OFF_YAML) == ()


def test_yaml_kernel_flag_every_spelling(tmp_path):
    """The gate reads the parsed value, not a lowercase line: every YAML false spelling and the flow style turn the kernel off; a non-boolean is refused."""
    head = "model_update:\n  presets: [predict]\n  custom:\n    settings:\n      memory:\n        eval:\n"
    for i, spelled in enumerate(("false", "False", "FALSE", "no", "off", "No", "OFF")):
        p = tmp_path / f"off_{i}.yml"; p.write_text(head + f"          use_deepspeed_evo_attention: {spelled}\n")
        assert cli.yaml_ds4sci_off(str(p)), spelled
    p = tmp_path / "flow.yml"; p.write_text("model_update: {presets: [predict], custom: {settings: {memory: {eval: {use_deepspeed_evo_attention: false}}}}}\n")
    assert cli.yaml_ds4sci_off(str(p))
    for i, spelled in enumerate(("true", "True", "yes", "on")):
        p = tmp_path / f"on_{i}.yml"; p.write_text(head + f"          use_deepspeed_evo_attention: {spelled}\n")
        assert not cli.yaml_ds4sci_off(str(p)), spelled
    p = tmp_path / "absent.yml"; p.write_text("model_update:\n  presets: [predict]\n")
    assert not cli.yaml_ds4sci_off(str(p))
    p = tmp_path / "bad.yml"; p.write_text(head + "          use_deepspeed_evo_attention: maybe\n")
    with pytest.raises(ValueError):
        cli.yaml_ds4sci_off(str(p))


def test_the_exact_line_lays_the_stock_members_keys_on_and_the_fast_line_pins_the_kernels_off(tmp_path, capsys):
    """A caller's yaml: `exact` lays the stock member's keys on (the det member under --det: cuEquivariance on, DS4Sci off); `fast` (graphed)
    composes it under its configuration: the four alternative kernel flags pinned off and the precision bf16, every override named on stderr."""
    import yaml
    ex = modes.LINES[("exact", "cueq")]; fast = modes.LINES[("fast", None)]
    for yml in (modes.STOCK_DET_YAML, modes.KERNELS_OFF_YAML, modes.SHIPPED_YAML):
        p = os.path.join(HOME, yml)
        got = cli.row_yaml(HOME, ex, None, runner_yaml=p, out_dir=str(tmp_path), mode="exact", det=1)
        if yml == modes.STOCK_DET_YAML:
            assert got == os.path.join(HOME, modes.STOCK_DET_YAML)                                                # the stock family follows the det level
        else:                                                                                                       # anything else: the det member's keys laid on (cuEquivariance on, DS4Sci off, bf16-mixed)
            assert os.path.basename(got) == "runner_composed_" + os.path.basename(yml) and cli.yaml_kernel_flags(got) == {"use_deepspeed_evo_attention": False, "use_cueq_triangle_kernels": True}, yml
    capsys.readouterr()
    for yml in (modes.STOCK_YAML, modes.STOCK_DET_YAML, modes.SHIPPED_YAML):
        got = cli.row_yaml(HOME, fast, "bf16", runner_yaml=os.path.join(HOME, yml), out_dir=str(tmp_path), mode="fast")
        err = capsys.readouterr().err
        doc = yaml.safe_load(open(got)); ev = doc["model_update"]["custom"]["settings"]["memory"]["eval"]
        assert os.path.basename(got) == "runner_composed_" + os.path.basename(yml) and doc["pl_trainer_args"]["precision"] == "bf16-mixed", yml
        assert all(ev[f] is False for f in ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma")), (yml, ev)
        assert "composed under the fast configuration (opt/openfold3_opt/fast_bf16_predict.yml) -> " in err, err
        written = (yaml.safe_load(open(os.path.join(HOME, yml))).get("model_update") or {}).get("custom", {}).get("settings", {}).get("memory", {}).get("eval", {})
        for f in ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma"):
            if written.get(f) is True:                                              # written on: overridden, named `True -> False`
                assert f"model_update.custom.settings.memory.eval.{f} (True -> False)" in err, (yml, f, err)
            elif f not in written:                                                  # not written (upstream's default): the line's `false` set, named as set
                assert f"model_update.custom.settings.memory.eval.{f}=False" in err and "fast sets runner-yaml keys the caller's yaml leaves unset:" in err, (yml, f, err)
    own = cli.row_yaml(HOME, fast, "bf16", runner_yaml=os.path.join(HOME, modes.KERNELS_OFF_YAML), out_dir=str(tmp_path), mode="fast")
    assert "overrides runner-yaml keys" not in capsys.readouterr().err and cli.yaml_kernel_flags(own) == {"use_deepspeed_evo_attention": False, "use_cueq_triangle_kernels": False}


def test_a_runner_yaml_naming_either_stock_file_is_the_stock_family(tmp_path):
    """`--runner-yaml <STOCK_YAML | STOCK_DET_YAML>` under off / exact names the family: the det level picks the member exactly as without the flag
    (a caller that names the speed yaml under --det 1 still runs the det yaml on both arms); a staged copy counts by content; fast is untouched (refused by name elsewhere)."""
    import shutil
    ex = modes.LINES[("exact", "cueq")]
    for named in (modes.STOCK_YAML, modes.STOCK_DET_YAML):
        p = os.path.join(HOME, named)
        assert cli.stock_family(HOME, p) and cli.stock_family(HOME, named)
        assert cli.row_yaml(HOME, ex, runner_yaml=p, mode="exact", det=1) == os.path.join(HOME, modes.STOCK_DET_YAML)
        assert cli.row_yaml(HOME, ex, runner_yaml=p, mode="exact", det=0) == os.path.join(HOME, modes.STOCK_YAML)
        assert cli.row_yaml(HOME, runner_yaml=p, mode="off", det=2) == os.path.join(HOME, modes.STOCK_DET_YAML)
    copy = tmp_path / "staged_stock.yml"; shutil.copy(os.path.join(HOME, modes.STOCK_YAML), copy)
    assert cli.stock_family(HOME, str(copy)) and cli.row_yaml(HOME, runner_yaml=str(copy), mode="off", det=1) == os.path.join(HOME, modes.STOCK_DET_YAML)
    other = _stubs.caller_yaml(tmp_path)                                                            # any other yaml under exact: composed under the det member's keys
    assert not cli.stock_family(HOME, other) and os.path.basename(cli.row_yaml(HOME, ex, runner_yaml=other, out_dir=str(tmp_path / "x"), mode="exact", det=1)) == "runner_composed_" + os.path.basename(other)
    got = cli.row_yaml(HOME, modes.LINES[("fast", None)], "fp32", runner_yaml=os.path.join(HOME, modes.STOCK_YAML), out_dir=str(tmp_path / "f"), mode="fast", det=1)
    assert os.path.basename(got) == "runner_composed_" + os.path.basename(modes.STOCK_YAML) and cli.yaml_kernel_flags(got) == {"use_deepspeed_evo_attention": False, "use_cueq_triangle_kernels": False}   # fast: no stock family; the graphed line pins the kernels off
