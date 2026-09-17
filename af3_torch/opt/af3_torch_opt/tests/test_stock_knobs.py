"""The stock CLI's own knobs on `pred`, under their upstream names and defaults — `--num_diffusion_samples` (5) and `--fastnn` /
`--nofastnn` (on) from xfold's run_alphafold.py, `--num_recycles` / `--diffusion_steps` from the model constructor — and how each mode
serves them: off / exact run xfold's SHIPPED Triton fastnn kernels (forward.py --fastnn 1; forward.apply_fastnn carries the stock CLI's own
three statements), `--nofastnn` under off is the eager port, exact refuses `--nofastnn` and fast / big refuse an explicit `--fastnn` by
name before anything runs; the SETTINGS line names the resolved protocol; there is no named preset."""
import json
import os
import re
import tarfile

from af3_torch_opt import cli, modes, stack

from .conftest import archive_or_skip

from .test_cli_manifest import _inputs

SWITCHES = ("layer_norm_implementation", "dot_product_attention_implementation", "gated_linear_unit_implementation")
SETTINGS_DEFAULT = "[af3-torch-opt] SETTINGS num_recycles=None num_diffusion_samples=5 diffusion_steps=None fastnn={fastnn} confidence=once_per_sample"


def _forward_calls(box):
    if not os.path.exists(box["log"]):                    # no stub process ran at all
        return []
    return [c for c in (json.loads(l) for l in open(box["log"])) if any(str(x).endswith("/forward.py") for x in c)]


def test_no_preset_machinery():
    assert not hasattr(modes, "SETTINGS_PRESETS") and not hasattr(modes, "MODE_SETTINGS") and not hasattr(modes, "settings_refusal")
    assert modes.FASTNN_MODES == ("off", "exact") and modes.STOCK_NUM_DIFFUSION_SAMPLES == modes.UPSTREAM_SAMPLES == 5
    assert modes.fastnn_refusal("off", None) is None and modes.fastnn_refusal("off", True) is None and modes.fastnn_refusal("off", False) is None
    assert modes.fastnn_refusal("exact", False).startswith("--nofastnn is served under --mode off only, not exact")
    for m in ("fast", "big"):
        assert modes.fastnn_refusal(m, None) is None and modes.fastnn_refusal(m, False) is None
        assert modes.fastnn_refusal(m, True).startswith(f"--fastnn is served under --mode off | exact only, not {m}")


def test_xfold_ships_fastnn_torch_and_the_cli_flips_it(tmp_path):
    """The by-file facts: xfold's fastnn module defaults every switch to torch (the eager path `off --nofastnn` runs), the stock CLI's
    `--fastnn` (default True) is the one place they become triton — forward.apply_fastnn carries those three statements verbatim — and
    `--num_diffusion_samples` defaults to 5 there."""
    archive_or_skip()
    cfg = open(os.path.join(stack.kit_home(), "af3_torch", "xfold", "fastnn", "config.py"), encoding="utf-8").read()
    assert re.findall(r'^(\w+_implementation) = "(\w+)"$', cfg, re.M) == [(k, "torch") for k in SWITCHES]
    P = stack.pins(); arch = P["upstream"]["archive"]
    with tarfile.open(os.path.join(stack.home(), "stock", arch["file"]), "r:gz") as t:
        cli_src = t.extractfile(f"xfold-{P['upstream']['commit']}/run_alphafold.py").read().decode()
    assert re.search(r"_USE_FASTNN = flags\.DEFINE_bool\(\s*'fastnn',\s*True,", cli_src)
    assert re.search(r"_NUM_DIFFUSION_SAMPLES = flags\.DEFINE_integer\(\s*'num_diffusion_samples',\s*5,", cli_src)
    stmts = [f"fastnn_config.{k} = 'triton'" for k in SWITCHES]
    block = cli_src[cli_src.index("if _USE_FASTNN.value is True:"):][:400]
    assert all(s_ in block for s_ in stmts), block
    fwd = open(os.path.join(os.path.dirname(cli.__file__), "forward.py"), encoding="utf-8").read()
    body = fwd[fwd.index("def apply_fastnn("): fwd.index("\ndef ", fwd.index("def apply_fastnn(") + 1)]
    assert "from xfold.fastnn import config as fastnn_config" in body and all(s_ in body for s_ in stmts), body
    assert fwd.index('rep["fastnn"] = apply_fastnn(a.fastnn)') < fwd.index("model = A.build_model("), "set before the model is built"
    assert 'ap.add_argument("--fastnn", type=int, choices=(0, 1), default=0' in fwd


def test_off_runs_the_stock_kernels_by_default_and_names_them(box, tmp_path, capsys):
    """`off` with no flag = xfold as shipped: forward.py gets --fastnn 1 and --num_samples 5; the ACTIVE line carries no settings token; the
    SETTINGS line and the manifest's activation report name the protocol; bare `--fastnn` and `--fastnn=true` are the same run."""
    for extra in ((), ("--fastnn",), ("--fastnn=true",)):
        out = tmp_path / ("o" + str(len(extra)) + (extra[0][-4:] if extra else ""))
        assert cli.main(["pred", "--mode", "off", *extra, "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out)]) == 0
        err = capsys.readouterr().err
        active = [l for l in err.splitlines() if "] ACTIVE " in l][0]
        assert " mode=off " in active and "settings=" not in active, active
        assert SETTINGS_DEFAULT.format(fastnn=1) in err, err
        fwd = _forward_calls(box)[-1]
        assert fwd[fwd.index("--fastnn") + 1] == "1" and fwd[fwd.index("--num_samples") + 1] == "5" and "--num_recycles" not in fwd and "--diffusion_steps" not in fwd
        man = cli.last_run()
        assert man["activation"]["settings"] == {"num_recycles": None, "num_diffusion_samples": 5, "diffusion_steps": None, "fastnn": 1, "confidence": "once_per_sample"} and "settings_preset" not in man["activation"]


def test_nofastnn_under_off_is_the_eager_port(box, tmp_path, capsys):
    for flag in ("--nofastnn", "--fastnn=false"):
        assert cli.main(["pred", "--mode", "off", flag, "--json_path", _inputs(box["tmp"], ("e",))[0], "--output_dir", str(tmp_path / flag.strip("-=").replace("=", ""))]) == 0
        err = capsys.readouterr().err
        assert SETTINGS_DEFAULT.format(fastnn=0) in err, err
        assert "--fastnn" not in _forward_calls(box)[-1]


def test_fast_and_big_run_the_kits_kernels_and_refuse_an_explicit_fastnn(box, tmp_path, capsys, monkeypatch):
    """fast: no --fastnn on the model argv and no SETTINGS line at the defaults; `--nofastnn` is accepted (the same run, named); an explicit
    `--fastnn` is refused by name before anything runs (rc 3, one NOT ACTIVE line, no process). exact refuses `--nofastnn` the same way."""
    assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("f",))[0], "--output_dir", str(tmp_path / "f")]) == 0
    err = capsys.readouterr().err
    assert "] SETTINGS " not in err and "--fastnn" not in _forward_calls(box)[-1]
    assert cli.main(["pred", "--mode", "fast", "--nofastnn", "--json_path", _inputs(box["tmp"], ("fn",))[0], "--output_dir", str(tmp_path / "fn")]) == 0
    err = capsys.readouterr().err
    assert SETTINGS_DEFAULT.format(fastnn=0) in err and "--fastnn" not in _forward_calls(box)[-1]
    n = len(_forward_calls(box))
    for mode, flag, reason in (("fast", "--fastnn", "--fastnn is served under --mode off | exact only, not fast"), ("big", "--fastnn", "--fastnn is served under --mode off | exact only, not big"),
                               ("exact", "--nofastnn", "--nofastnn is served under --mode off only, not exact")):
        assert cli.main(["pred", "--mode", mode, flag, "--json_path", _inputs(box["tmp"], (f"r{mode}",))[0], "--output_dir", str(tmp_path / ("r" + mode))]) == 3
        lines = [l for l in capsys.readouterr().err.splitlines() if l.startswith("[af3-torch-opt]")]
        assert len(lines) == 1 and lines[0].startswith(f"[af3-torch-opt] NOT ACTIVE mode={mode} n_gpu=1 reason={reason} ("), lines
    assert len(_forward_calls(box)) == n                                             # nothing ran for the refused three
    monkeypatch.setenv(modes.ENV_MODE, "fast")                                       # the mode from the environment refuses the same way
    assert cli.main(["pred", "--fastnn", "--json_path", _inputs(box["tmp"], ("renv",))[0], "--output_dir", str(tmp_path / "env")]) == 3


def test_the_knobs_pass_through_verbatim_in_every_mode(box, tmp_path, capsys):
    """--num_recycles / --num_diffusion_samples / --diffusion_steps reach forward.py as --num_recycles / --num_samples / --diffusion_steps in
    every mode, and the SETTINGS line names them (fastnn stays the mode's: 1 under off / exact, 0 = the kit's kernels under fast / big)."""
    for mode, fastnn in (("off", 1), ("exact", 1), ("fast", 0), ("big", 0)):
        assert cli.main(["pred", "--mode", mode, "--num_recycles", "1", "--num_diffusion_samples", "1", "--diffusion_steps", "2", "--json_path", _inputs(box["tmp"], (f"k{mode}",))[0], "--output_dir", str(tmp_path / ("k" + mode))]) == 0
        err = capsys.readouterr().err
        assert f"[af3-torch-opt] SETTINGS num_recycles=1 num_diffusion_samples=1 diffusion_steps=2 fastnn={fastnn} confidence=once_per_sample" in err, err
        fwd = _forward_calls(box)[-1]
        assert (fwd[fwd.index("--num_recycles") + 1], fwd[fwd.index("--num_samples") + 1], fwd[fwd.index("--diffusion_steps") + 1]) == ("1", "1", "2") and (("--fastnn" in fwd) == bool(fastnn))


def test_fastnn_flag_spellings():
    assert cli.fastnn_flag("true") is True and cli.fastnn_flag("1") is True and cli.fastnn_flag("False") is False and cli.fastnn_flag("0") is False
    import argparse, pytest
    with pytest.raises(argparse.ArgumentTypeError):
        cli.fastnn_flag("maybe")
