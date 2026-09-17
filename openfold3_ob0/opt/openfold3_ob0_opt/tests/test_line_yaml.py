"""The runner yaml a call runs under (cli.row_yaml): the caller's own, else the line's (the ds4sci line: upstream's shipped configuration),
else the mode's configuration — and the ds4sci line's refusal of a yaml that turns the DS4Sci kernel off (cli.line_yaml_check)."""
import os

import pytest
import yaml

from openfold3_ob0_opt.tests import _stubs
from openfold3_ob0_opt import cli, modes
from openfold3_ob0_opt.tests import _stubs

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_row_yaml_precedence(tmp_path):
    import yaml
    ex = modes.LINES[("exact", modes.EXACT_LINE)]
    assert ex.runner_yaml is None and modes.LINES[("fast", None)].runner_yaml == modes.STOCK_YAML   # exact follows the det level (cli.kernel_policy); fast names the stock runner yaml
    assert cli.kernel_policy("off") == cli.kernel_policy("exact") == modes.STOCK_YAML           # the stock configuration member, by mode
    assert cli.kernel_policy("off", 1) == cli.kernel_policy("exact", 2) == modes.STOCK_DET_YAML
    assert cli.kernel_policy("fast") is None and cli.kernel_policy("big", 1) is None and cli.kernel_policy(None) is None
    for kw in ({}, {"mode": "big"}):                                                      # no route names a configuration for (no mode) / (big without a line): refused by name (cli.UnnamedRoute)
        with pytest.raises(cli.UnnamedRoute):
            cli.row_yaml(HOME, **kw)
    assert cli.row_yaml(HOME, ex, mode="exact") == os.path.join(HOME, modes.STOCK_YAML) == cli.row_yaml(HOME, mode="off")
    assert cli.row_yaml(HOME, ex, mode="exact", det=1) == os.path.join(HOME, modes.STOCK_DET_YAML) == cli.row_yaml(HOME, mode="off", det=2)
    assert cli.row_yaml(HOME, modes.LINES[("fast", None)], mode="fast", det=1) == os.path.join(HOME, modes.STOCK_YAML)   # the fast line: the stock runner yaml itself
    own = _stubs.caller_yaml(tmp_path)                                                               # a caller's --runner-yaml is an OVERLAY on the route's member on every mode, off included; the caller's file itself is never touched
    got = cli.row_yaml(HOME, runner_yaml=own, mode="off")                                             # off: laid over the stock member (the caller's keys win, the member's other keys stay)
    assert got != own and os.path.basename(got).startswith(cli.COMPOSED_PREFIX) and yaml.safe_load(open(got))["pl_trainer_args"] == yaml.safe_load(open(os.path.join(HOME, modes.STOCK_YAML)))["pl_trainer_args"]
    assert yaml.safe_load(open(got))["model_update"]["custom"]["architecture"]["shared"]["num_recycles"] == 10                      # the caller's key
    for kw, member in ((dict(line=ex, mode="exact", det=1), modes.STOCK_DET_YAML), (dict(line=modes.LINES[("fast", None)], mode="fast"), modes.STOCK_YAML)):
        got = cli.row_yaml(HOME, runner_yaml=own, **kw)
        assert not got.startswith(str(tmp_path)) and os.path.basename(got).startswith(cli.COMPOSED_PREFIX), (kw, got)   # composed into a temporary directory, never among the outputs
        assert cli.yaml_kernel_flags(got) == cli.yaml_kernel_flags(os.path.join(HOME, member)), kw                                        # the member's kernel triple wins
        assert yaml.safe_load(open(own)) == yaml.safe_load(open(_stubs.caller_yaml(tmp_path / "again"))), kw                              # the caller's file untouched
    assert cli.row_yaml(HOME, modes.LINES[("big", "resident")], mode="big") == os.path.join(HOME, modes.BIG_BF16_C16_YAML)   # a line's own yaml (the big line)


def test_yaml_kernel_flag_read(tmp_path):
    shipped_on = tuple(f for f, v in cli.KERNEL_FLAG_DEFAULTS.items() if v)                      # what upstream ships on (0.5.0: the Triton triangle kernels)
    assert cli.yaml_ds4sci_off(os.path.join(HOME, modes.KERNELS_OFF_YAML)) and cli.yaml_ds4sci_off(os.path.join(HOME, modes.SHIPPED_YAML)) and cli.yaml_ds4sci_off(os.path.join(HOME, modes.STOCK_YAML))
    ds4sci, all3, shipped32 = _stubs.kernel_yaml(tmp_path, "d.yml", ds4sci=True), _stubs.kernel_yaml(tmp_path, "e.yml", cueq=True, ds4sci=True, precision="bf16-mixed"), _stubs.kernel_yaml(tmp_path, "s.yml", precision="32-true")
    assert not cli.yaml_ds4sci_off(ds4sci) and not cli.yaml_ds4sci_off(all3)
    assert cli.yaml_kernels_on(os.path.join(HOME, modes.SHIPPED_YAML)) == shipped_on == cli.yaml_kernels_on(shipped32)                      # absent flags = the shipped triple
    assert cli.yaml_kernels_on(os.path.join(HOME, modes.KERNELS_OFF_YAML)) == ()
    assert set(cli.yaml_kernels_on(all3)) == set(cli.KERNEL_FLAG_DEFAULTS)                                                                   # all three on
    assert cli.yaml_kernels_on(os.path.join(HOME, modes.STOCK_YAML)) == ("use_triton_triangle_kernels", "use_cueq_triangle_kernels") or set(cli.yaml_kernels_on(os.path.join(HOME, modes.STOCK_YAML))) == {"use_triton_triangle_kernels", "use_cueq_triangle_kernels"}


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
    assert cli.yaml_ds4sci_off(str(p))                                                              # absent = upstream's shipped default: off (0.5.0)
    p = tmp_path / "bad.yml"; p.write_text(head + "          use_deepspeed_evo_attention: maybe\n")
    with pytest.raises(ValueError):
        cli.yaml_ds4sci_off(str(p))


def test_the_exact_line_takes_every_yaml_and_the_fast_line_only_its_capture_safe_set(tmp_path):
    ex = modes.LINES[("exact", modes.EXACT_LINE)]; fast = modes.LINES[("fast", None)]
    others = [_stubs.kernel_yaml(tmp_path, "c32.yml", cueq=True, precision="32-true"), _stubs.kernel_yaml(tmp_path, "dbf.yml", ds4sci=True, precision="bf16-mixed"), _stubs.kernel_yaml(tmp_path, "e32n.yml", cueq=True, ds4sci=True, pinned=True)]
    for yml in (modes.STOCK_YAML, modes.STOCK_DET_YAML, modes.SHIPPED_YAML, modes.KERNELS_OFF_YAML, others[0]):
        assert cli.line_yaml_check(HOME, ex, os.path.join(HOME, yml), "pred --mode exact") is None        # the exact line captures graphs: the capture-safe set (stock's cuEq, Triton, kernels off) runs
    for yml in (modes.SHIPPED_YAML, modes.STOCK_YAML, modes.STOCK_DET_YAML, modes.KERNELS_OFF_YAML, others[0], _stubs.kernel_yaml(tmp_path, "abf.yml", precision="bf16-mixed")):
        assert cli.line_yaml_check(HOME, fast, os.path.join(HOME, yml), "pred --mode fast") is None, yml   # the capture-safe set: upstream's Triton kernels, stock's cuEquivariance kernels, or kernels off
    refused = [y for y in others if "d" in os.path.basename(y) or "e32n" in os.path.basename(y)]           # DS4Sci on: refused for capture by name
    assert len(refused) == 2
    for yml in refused:
        msg = cli.line_yaml_check(HOME, fast, yml, "pred --mode fast")
        assert msg and cli.GRAPHS_YAML_REFUSED in msg and "--mode exact" in msg and modes.STOCK_YAML in msg and "use_triton_triangle_kernels" not in msg.split("turns on")[1].split("—")[0], yml

def test_a_runner_yaml_naming_either_stock_file_is_the_stock_family(tmp_path):
    """`--runner-yaml <STOCK_YAML | STOCK_DET_YAML>` under off / exact names the family: the det level picks the member exactly as without the flag
    (a caller that names the speed yaml under --det 1 still runs the det yaml on both arms); a staged copy counts by content; fast is untouched (refused by name elsewhere)."""
    import shutil
    ex = modes.LINES[("exact", modes.EXACT_LINE)]
    for named in (modes.STOCK_YAML, modes.STOCK_DET_YAML):
        p = os.path.join(HOME, named)
        assert cli.stock_family(HOME, p) and cli.stock_family(HOME, named)
        assert cli.row_yaml(HOME, ex, runner_yaml=p, mode="exact", det=1) == os.path.join(HOME, modes.STOCK_DET_YAML)
        assert cli.row_yaml(HOME, ex, runner_yaml=p, mode="exact", det=0) == os.path.join(HOME, modes.STOCK_YAML)
        assert cli.row_yaml(HOME, runner_yaml=p, mode="off", det=2) == os.path.join(HOME, modes.STOCK_DET_YAML)
    copy = tmp_path / "staged_stock.yml"; shutil.copy(os.path.join(HOME, modes.STOCK_YAML), copy)
    assert cli.stock_family(HOME, str(copy)) and cli.row_yaml(HOME, runner_yaml=str(copy), mode="off", det=1) == os.path.join(HOME, modes.STOCK_DET_YAML)
    other = _stubs.caller_yaml(tmp_path)                                                            # any other yaml is composed under the member (cli.compose_runner_yaml)
    got = cli.row_yaml(HOME, ex, runner_yaml=other, mode="exact", det=1)
    assert not cli.stock_family(HOME, other) and got != other and os.path.basename(got).startswith(cli.COMPOSED_PREFIX) and cli.yaml_kernel_flags(got) == cli.yaml_kernel_flags(os.path.join(HOME, modes.STOCK_DET_YAML))
    got = cli.row_yaml(HOME, modes.LINES[("fast", None)], runner_yaml=os.path.join(HOME, modes.STOCK_YAML), mode="fast", det=1)      # under fast the stock yaml is a caller's yaml like any other: composed under the line's member (the same file: nothing overridden)
    assert cli.yaml_kernel_flags(got) == cli.yaml_kernel_flags(os.path.join(HOME, modes.STOCK_YAML))
