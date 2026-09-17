"""The graphed lines' runner-yaml contract (cli.line_yaml_check over modes.line_graphed): `sampler` and `fast` capture the diffusion sampler in
CUDA graphs (so does `composed`) and run the kernels-off attention configuration only — a runner yaml that leaves upstream's DS4Sci evoformer attention on or turns
the cuEquivariance triangle kernels on is refused BY NAME before the run (the graphs refuse to capture those kernels: a run would fail every query
at capture). Resolving `fast` through its precision words never yields graphs together with those kernels; the pred and
check verbs refuse the shipped configuration under `fast` with the words and rc 2 before any activation."""
import os

import pytest

from openfold3_ob0_opt import cli, modes
from openfold3_ob0_opt.tests import _stubs

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HEAD = "model_update:\n  presets: [predict]\n  custom:\n    settings:\n      memory:\n        eval:\n"


GRAPHED = {("fast", None), ("exact", modes.EXACT_LINE)}          # the lines whose switches carry OF3_CUDA_GRAPHS=1


def test_graphed_lines():
    assert {k for k, ln in modes.LINES.items() if modes.line_graphed(ln)} == GRAPHED == {k for k, ln in modes.LINES.items() if ln.env.get("OF3_CUDA_GRAPHS") == "1"}
    assert all(modes.graphed(m, l) == modes.line_graphed(ln) for (m, l), ln in modes.LINES.items())
    assert not any(modes.line_graphed(modes.LINES[("big", l)]) for l in modes.BIG_LINES) and modes.line_graphed(modes.LINES[("exact", modes.EXACT_LINE)])


def test_fast_resolves_to_a_capture_safe_yaml():
    """Whatever the image carries (cuEquivariance, deepspeed's op), the fast line's own yaml — the stock runner yaml, at its one precision —
    stays inside the capture-safe kernel set (modes.CAPTURE_SAFE_KERNEL_FLAGS): graphs and a kernel they are not shown to capture never meet
    through the package's own resolution."""
    ln = modes.LINES[("fast", None)]
    seen = set()
    yml = cli.row_yaml(HOME, ln, mode="fast")
    assert set(cli.yaml_kernels_on(yml)) <= set(modes.CAPTURE_SAFE_KERNEL_FLAGS), yml
    assert cli.line_yaml_check(HOME, ln, yml, "pred --mode fast") is None
    seen.add(os.path.relpath(yml, HOME))
    assert seen == {modes.STOCK_YAML} == {ln.runner_yaml}
    assert cli.yaml_kernel_flags(os.path.join(HOME, modes.SHIPPED_YAML)) == dict(cli.KERNEL_FLAG_DEFAULTS)            # the shipped yaml sets no flag: upstream's defaults
    assert cli.yaml_kernels_on(os.path.join(HOME, modes.SHIPPED_YAML)) == tuple(f for f, v in cli.KERNEL_FLAG_DEFAULTS.items() if v)


def test_graphed_lines_refuse_a_ds4sci_yaml_by_name(tmp_path):
    shipped = os.path.join(HOME, modes.SHIPPED_YAML)
    cueq = tmp_path / "cueq.yml"; cueq.write_text(HEAD + "          use_triton_triangle_kernels: true\n          use_cueq_triangle_kernels: true\n")
    both = tmp_path / "both.yml"; both.write_text(HEAD + "          use_triton_triangle_kernels: false\n          use_deepspeed_evo_attention: true\n          use_cueq_triangle_kernels: True\n")
    for key in sorted(GRAPHED, key=str):
        ln = modes.LINES[key]
        assert cli.line_yaml_check(HOME, ln, shipped, "pred --mode x") is None                                     # upstream's Triton kernels: capture-safe (modes.CAPTURE_SAFE_KERNEL_FLAGS)
        assert cli.line_yaml_check(HOME, ln, os.path.join(HOME, modes.KERNELS_OFF_YAML), "pred") is None            # kernels off: capture-safe
        assert cli.line_yaml_check(HOME, ln, str(cueq), "pred --mode x") is None                                   # stock's cuEquivariance kernels: capture-safe (modes.CUEQ_CAPTURES)
        assert cli.line_yaml_check(HOME, ln, os.path.join(HOME, modes.STOCK_YAML), "pred --mode x") is None
        msg = cli.line_yaml_check(HOME, ln, str(both), "pred --mode x")
        assert msg and cli.GRAPHS_YAML_REFUSED in msg and "turns on use_deepspeed_evo_attention" in msg and "use_cueq_triangle_kernels" not in msg.split("turns on")[1].split("—")[0] and modes.STOCK_YAML in msg and "--mode exact" in msg, msg
    # the lines that do not capture graphs take the kernels-on yamls they are documented with
    for yml in (modes.STOCK_YAML, modes.STOCK_DET_YAML, shipped):
        assert cli.line_yaml_check(HOME, modes.LINES[("exact", modes.EXACT_LINE)], yml if os.path.isabs(yml) else os.path.join(HOME, yml), "pred") is None
    assert cli.line_yaml_check(HOME, modes.LINES[("big", "resident")], os.path.join(HOME, modes.BIG_BF16_C16_YAML), "pred") is None   # the big line's own yaml
    bad = tmp_path / "bad.yml"; bad.write_text(HEAD + "          use_cueq_triangle_kernels: perhaps\n")
    with pytest.raises(ValueError):
        cli.line_yaml_check(HOME, modes.LINES[("fast", None)], str(bad), "pred")


def test_pred_and_check_refuse_fast_with_a_third_party_kernel_yaml_before_the_run(tmp_path, capsys):
    """The CLI route: a caller's `--runner-yaml` that turns DS4Sci on under `fast` is COMPOSED under the line's base (its kernel triple wins, the
    overridden key named on ONE note line) — never refused; an unknown exact line is refused by name before anything runs."""
    w = tmp_path / "w.pt"; w.write_bytes(b"weights")
    ds4 = _stubs.kernel_yaml(tmp_path, "ds4.yml", ds4sci=True)                                  # a third-party kernel outside the capture-safe set (DS4Sci)
    d = _stubs.stub_dist()
    try:
        _stubs.reset_package()
        cli.main(["check", "--mode", "fast", "--runner-yaml", ds4])        # (the dry run's rc follows the levers present in this tree; the words are the point)
        err = capsys.readouterr().err
        assert cli.GRAPHS_YAML_REFUSED not in err and f"composed under the fast configuration ({modes.STOCK_YAML})" in err and "use_deepspeed_evo_attention (True -> False)" in err, err[-900:]
    finally:
        _stubs.reset_package()
        _stubs.unstub_dist(d)


def test_runner_yaml_relative_spelling_resolves_against_the_tree_and_a_missing_one_is_refused(tmp_path, capsys, monkeypatch):
    """`--runner-yaml opt/openfold3_ob0_opt/shipped_predict.yml` (the tree's spelling) from any working directory is the tree's file; a yaml found in
    neither the working directory nor the tree is refused by name (usage exit 2), never handed to the runner."""
    assert cli.resolve_runner_yaml(None, HOME) is None
    monkeypatch.chdir(tmp_path)
    assert cli.resolve_runner_yaml(modes.SHIPPED_YAML, HOME) == os.path.join(HOME, modes.SHIPPED_YAML)
    here = tmp_path / "mine.yml"; here.write_text("model_update:\n  presets: [predict]\n")
    assert cli.resolve_runner_yaml("mine.yml", HOME) == str(here) and cli.resolve_runner_yaml(str(here), HOME) == str(here)
    with pytest.raises(ValueError, match="no such file"):
        cli.resolve_runner_yaml("nowhere/none.yml", HOME)
    w = tmp_path / "w.pt"; w.write_bytes(b"weights")
    d = _stubs.stub_dist()
    try:
        _stubs.reset_package()
        _stubs.kernel_yaml(tmp_path, "ds4.yml", ds4sci=True)
        cli.main(["check", "--mode", "fast", "--runner-yaml", "ds4.yml"])
        err = capsys.readouterr().err
        assert f"runner yaml {tmp_path / 'ds4.yml'} composed under the fast configuration" in err, err[-600:]   # the relative spelling resolves to the working directory's file, which reaches the composition
        _stubs.reset_package()
        rc = cli.main(["pred", "--mode", "fast", "--runner-yaml", "nowhere/none.yml", "--query-json", "q.json", "--output-dir", str(tmp_path / "o"), "--ckpt", str(w)])
        assert rc == cli.EXIT_USAGE and "no such file" in capsys.readouterr().err
    finally:
        _stubs.reset_package()
        _stubs.unstub_dist(d)


def test_a_missing_line_yaml_is_refused_by_name_not_a_traceback(tmp_path):
    ln = modes.LINES[("fast", None)]
    msg = cli.line_yaml_check(HOME, ln, str(tmp_path / "not_here.yml"), "check --mode fast")
    assert msg and cli.MISSING_YAML in msg and "not_here.yml" in msg
