"""The graphed lines' runner-yaml contract (runner_yaml.line_pins over modes.line_graphed): `fast` captures the diffusion sampler in CUDA graphs and
runs the kernels-off attention configuration only — the graphs do not capture upstream's DS4Sci evoformer attention or the cuEquivariance triangle
kernels, so a caller's runner yaml that leaves or turns either on is composed under the line: the four third-party kernel flags pinned off, every
override named on stderr before the run. Resolving `fast` through its precision words never yields graphs together with those kernels."""
import os

import pytest

from openfold3_opt import cli, modes, runner_yaml
from openfold3_opt.tests import _stubs

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HEAD = "model_update:\n  presets: [predict]\n  custom:\n    settings:\n      memory:\n        eval:\n"


GRAPHED = {("fast", None), ("exact", "cueq")}                          # the lines whose switches carry OF3_CUDA_GRAPHS=1 (exact: up to its own 512-token cap)
KERNELS_OFF_GRAPHED = {("fast", None)}                                 # ... of which those over the kernels-off base pin a caller's kernel flags off (exact keeps stock's kernels)


def test_graphed_lines():
    assert {k for k, ln in modes.LINES.items() if modes.line_graphed(ln)} == GRAPHED == {k for k, ln in modes.LINES.items() if ln.env.get("OF3_CUDA_GRAPHS") == "1"}
    assert all(modes.graphed(m, l) == modes.line_graphed(ln) for (m, l), ln in modes.LINES.items())
    assert not any(modes.line_graphed(modes.LINES[("big", l)]) for l in modes.BIG_LINES) and modes.line_graphed(modes.LINES[("exact", "cueq")])
    assert modes.LINES[("exact", "cueq")].graphs_max_tokens == 512 and modes.LINES[("fast", None)].graphs_max_tokens is None   # the exact line's own cap; fast captures at every size
    assert modes.stock_kernel_line(modes.LINES[("exact", "cueq")]) and not modes.stock_kernel_line(modes.LINES[("fast", None)])


def test_fast_resolves_to_a_kernels_off_yaml_under_every_precision_word():
    """Whatever the image carries (cuEquivariance, deepspeed's op), the fast line's own yaml — under every precision word the route accepts —
    leaves both third-party kernel flags off: graphs and those kernels never meet through the package's own resolution."""
    ln = modes.LINES[("fast", None)]
    seen = set()
    for precision in (None, "fp32", "bf16"):
        yml = cli.row_yaml(HOME, ln, precision, mode="fast")
        assert not any(cli.yaml_kernel_flags(yml).values()), (precision, yml)
        seen.add(os.path.relpath(yml, HOME))
    assert seen == {modes.KERNELS_OFF_YAML, modes.FAST_BF16_YAML}
    assert cli.yaml_kernel_flags(os.path.join(HOME, modes.STOCK_YAML)) == {"use_deepspeed_evo_attention": True, "use_cueq_triangle_kernels": True}
    assert cli.yaml_kernel_flags(os.path.join(HOME, modes.SHIPPED_YAML)) == {"use_deepspeed_evo_attention": True, "use_cueq_triangle_kernels": False}
    assert [f for f, v in cli.yaml_kernel_flags(os.path.join(HOME, modes.SHIPPED_YAML)).items() if v] == ["use_deepspeed_evo_attention"]


def test_graphed_lines_pin_a_kernels_on_yaml_off_by_name(tmp_path, capsys):
    """A graphed line composes a caller's kernels-on yaml under the line: every third-party kernel flag the yaml leaves or turns on is pinned off
    in the written composition and named (`… True -> False`, or `added` where the yaml did not write the key); the lines that do not capture graphs
    take such a yaml as given (their own path back)."""
    import yaml
    shipped = os.path.join(HOME, modes.SHIPPED_YAML)
    cueq = tmp_path / "cueq.yml"; cueq.write_text(HEAD + "          use_deepspeed_evo_attention: false\n          use_cueq_triangle_kernels: true\n")
    both = tmp_path / "both.yml"; both.write_text(HEAD + "          use_deepspeed_evo_attention: true\n          use_cueq_triangle_kernels: True\n")
    assert {k for k in GRAPHED if not modes.stock_kernel_line(modes.LINES[k])} == KERNELS_OFF_GRAPHED
    for key in sorted(KERNELS_OFF_GRAPHED, key=str):
        ln = modes.LINES[key]
        pins = runner_yaml.line_pins(ln, os.path.join(HOME, modes.FAST_BF16_YAML))
        assert all(pins["model_update"]["custom"]["settings"]["memory"]["eval"][f] is False for f in runner_yaml.KERNEL_EVAL_FLAGS) and pins["pl_trainer_args"] == {"precision": "bf16-mixed"}
        for src, named in ((shipped, ()), (str(cueq), ("use_cueq_triangle_kernels",)), (str(both), ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels"))):
            capsys.readouterr()
            got = cli.row_yaml(HOME, ln, modes.FAST_PRECISION, runner_yaml=src, out_dir=str(tmp_path / "o"), mode=key[0])
            err = capsys.readouterr().err
            assert not any(cli.yaml_kernel_flags(got).values()) and yaml.safe_load(open(got))["pl_trainer_args"]["precision"] == "bf16-mixed", (src, got)
            for f in named:
                assert f"model_update.custom.settings.memory.eval.{f} (True -> False)" in err, (src, err)
            if src == shipped:                                                      # DS4Sci absent = upstream's default on: the line's `false` is SET, named as set
                assert "sets runner-yaml keys the caller's yaml leaves unset:" in err and "use_deepspeed_evo_attention=False" in err, err
    # the lines that do not capture graphs lay their member's keys on such a yaml and leave the rest as written: exact keeps a caller's DS4Sci
    # flag (its det member writes it off: named as an override) and sets its cuEquivariance flag on; the one-GPU big line sets its DS4Sci off
    ex = cli.row_yaml(HOME, modes.LINES[("exact", "cueq")], None, runner_yaml=shipped, out_dir=str(tmp_path / "e"), mode="exact", det=1)
    assert cli.yaml_kernel_flags(ex) == {"use_deepspeed_evo_attention": False, "use_cueq_triangle_kernels": True}
    c16 = os.path.join(HOME, modes.OFFLOAD_SHIPPED_C16_YAML)
    bg = cli.row_yaml(HOME, modes.LINES[("big", "resident")], None, runner_yaml=c16, out_dir=str(tmp_path / "b"), mode="big")
    assert os.path.basename(bg) == "runner_composed_" + os.path.basename(c16) and cli.yaml_kernel_flags(bg)["use_deepspeed_evo_attention"] is False
    bad = tmp_path / "bad.yml"; bad.write_text("- not\n- a mapping\n")
    with pytest.raises(ValueError):
        cli.row_yaml(HOME, modes.LINES[("fast", None)], modes.FAST_PRECISION, runner_yaml=str(bad), out_dir=str(tmp_path / "x"), mode="fast")


def test_check_composes_a_kernels_on_yaml_under_the_fast_line(tmp_path, capsys):
    """The CLI route: `check --mode fast --runner-yaml <shipped>` names the composition (the DS4Sci flag pinned off) and reports the graphed line —
    the graph levers stay requested; no usage exit."""
    d = _stubs.stub_dist()
    try:
        _stubs.reset_package()
        rc = cli.main(["check", "--mode", "fast", "--runner-yaml", os.path.join(HOME, modes.SHIPPED_YAML)])
        err = capsys.readouterr().err
        assert rc != cli.EXIT_USAGE and "composed under the fast configuration (opt/openfold3_opt/fast_bf16_predict.yml) -> " in err and "OPENFOLD3_OPT_GRAPHS_MAX_TOKENS=off" not in err, (rc, err[-900:])
        last = [l for l in err.splitlines() if l.startswith("[openfold3-opt] DRY-RUN mode=fast")][-1]
        assert "OF3_CUDA_GRAPHS=1" in last.split(" @ ")[0] and "cuda_graphs" in last.split("levers_requested=")[1].split()[0], last
    finally:
        _stubs.reset_package(); _stubs.unstub_dist(d)


def test_runner_yaml_relative_spelling_resolves_against_the_tree_and_a_missing_one_is_refused(tmp_path, capsys, monkeypatch):
    """`--runner-yaml opt/openfold3_opt/shipped_predict.yml` (the tree's spelling) from any working directory is the tree's file; a yaml found in
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
        rc = cli.main(["check", "--mode", "fast", "--runner-yaml", modes.SHIPPED_YAML, "--ckpt", str(w)])
        assert rc != cli.EXIT_USAGE and "runner yaml opt/openfold3_opt/shipped_predict.yml composed under the fast configuration (opt/openfold3_opt/fast_bf16_predict.yml) -> " in capsys.readouterr().err   # the relative spelling from another cwd reaches the composition as the tree's file (the graphs turn off; the run goes on to the weights gate)
        _stubs.reset_package()
        rc = cli.main(["pred", "--mode", "fast", "--runner-yaml", "nowhere/none.yml", "--query-json", "q.json", "--output-dir", str(tmp_path / "o"), "--ckpt", str(w)])
        assert rc == cli.EXIT_USAGE and "no such file" in capsys.readouterr().err
    finally:
        _stubs.reset_package()
        _stubs.unstub_dist(d)
