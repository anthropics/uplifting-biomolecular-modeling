"""The CLI: usage, the one-mode rule, the dry run on a box without the stack (named gates, rc 3), the stock check (rc 0), out_dir parsing."""
import json
import os

import pytest

from .conftest import n_kit_files
from protenix_v1_opt import cli
from protenix_v1_opt import kit as K
from protenix_v1_opt.modes import DEFAULT_MODE, KIT_MODES


def test_usage_and_unknown_command(capsys):
    assert cli.main([]) == 2
    assert cli.main(["--help"]) == 0
    assert cli.main(["fold"]) == 2


def test_mode_disagreement_is_refused(monkeypatch):
    monkeypatch.setenv("PROTENIX_V1_OPT", "fast")
    assert cli.main(["check", "--mode", "exact"]) == 2


def test_check_reports_the_gates(capsys):
    """rc 3 with the gates named on a box without the stack (no pinned protenix / no GPU); rc 0 with the GPU named on the pinned stack."""
    rc = cli.main(["check"])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == DEFAULT_MODE == "fast" and out["arm"] == KIT_MODES[DEFAULT_MODE].arm == "fast+gflash+tricuda+ttr+sg+hoist+keep_pool+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+sampler_prep+pfattn+opm_fused+pwa_fused+cond_dedupe+dit_fused+dit_lowp+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused"   # the default: the fast line (the amortized warm steady-state is the timing that counts)
    assert out["kit_files"]["ok"] is True and out["kit_files"]["files"] == n_kit_files()
    assert rc == (0 if out["ok"] else 3)
    if out["ok"]:
        assert out["gates"] == [] and out["protenix_version"] == "1.1.0" and out["gpu"]["sm"]
    else:
        joined = " ".join(out["gates"])
        assert "protenix" in joined or "GPU" in joined or "torch" in joined


def test_check_off_is_the_stock_route(capsys):
    assert cli.main(["check", "--mode", "off"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["arm"] == "stock" and out["gates"] == []


def test_check_det_flag_and_env(monkeypatch, capsys):
    monkeypatch.setenv("PROTENIX_V1_OPT_DET", "1")
    cli.main(["check", "--mode", "exact"])
    assert json.loads(capsys.readouterr().out)["det"] is True
    cli.main(["check", "--mode", "exact", "--det", "0"])
    assert json.loads(capsys.readouterr().out)["det"] is False


def test_stock_arguments_pass_through_verbatim(monkeypatch):
    """Everything but the kit's own flags (--mode/--det/--n_gpu/--allow-partial) is stock's: it reaches the stock CLI unchanged, in order,
    nothing prepended (no kit-side settings table) — `--cycle 1 --step 2 --sample 1` included."""
    seen = {}
    monkeypatch.setattr(cli, "_run_stock", lambda stock_args, det: seen.update(args=stock_args, det=det) or 0)
    monkeypatch.setattr(cli.K, "frozen_weights_check", lambda *a, **k: {})       # the weights boot gate admits (test_frozen_weights.py holds it to its contract): the argument pass-through is what is under test
    assert cli.main(["pred", "--mode", "off", "--det", "0", "--seeds", "101", "--cycle", "1", "--step", "2", "--sample", "1", "--input", "x.json", "--out_dir", "o"]) == 0
    assert seen["det"] is False and seen["args"] == ["--seeds", "101", "--cycle", "1", "--step", "2", "--sample", "1", "--input", "x.json", "--out_dir", "o"]
    seen.clear()
    assert cli.main(["pred", "--mode", "off", "--input", "x.json", "--out_dir", "o"]) == 0 and seen["args"] == ["--input", "x.json", "--out_dir", "o"]   # no flag: stock's own defaults, nothing injected
    assert not hasattr(cli, "settings") and "--settings" not in cli.USAGE


def test_out_dir_parsing():
    assert cli._out_dir(["-i", "x.json"]) == "./output"
    assert cli._out_dir(["-i", "x.json", "-o", "/tmp/o"]) == "/tmp/o"
    assert cli._out_dir(["--out_dir", "/a", "--out_dir=/b"]) == "/b"


def test_kit_input_json_makes_the_bundled_msa_dirs_absolute(tmp_path):
    """The bundled 1BRS input names its MSA directories relative to protenix_v1/ (the README runs it from there with no edit); warm's copy is absolute."""
    src = json.load(open(os.path.join(K.kit_home(), "inputs", "p995_1brs.json"), encoding="utf-8"))
    rel = [e["proteinChain"]["msa"]["precomputed_msa_dir"] for e in src[0]["sequences"]]
    assert rel == ["opt/forward/v05_addon/inputs/msa/barnase", "opt/forward/v05_addon/inputs/msa/barstar"] and all(os.path.isdir(os.path.join(K.tree_home(), r)) for r in rel)
    p = cli.kit_input_json("p995_1brs", str(tmp_path))
    dirs = [e["proteinChain"]["msa"]["precomputed_msa_dir"] for e in json.load(open(p, encoding="utf-8"))[0]["sequences"]]
    assert all(os.path.isabs(d) and os.path.isdir(d) for d in dirs) and dirs == [os.path.join(K.tree_home(), r) for r in rel]
    with pytest.raises(cli.CliError):
        cli.kit_input_json("nope", str(tmp_path))


def test_check_names_a_composition_refusal_as_a_gate(monkeypatch, capsys):
    """`check --mode big` under a retired / mistyped PROTENIX_V1_BIG_* name: the composition refuses by name and check reports it as
    the mode's gate (NOT ACTIVE line, exit 3) — never a traceback."""
    from protenix_v1_opt import big as B, cli, report as R
    B.reset()                                                                  # the composition is sized once per process: afresh here
    monkeypatch.setenv("PROTENIX_V1_BIG_RELP_LEAN", "0")
    rc = cli.main(["check", "--mode", "big"])
    B.reset()
    err = capsys.readouterr().err
    assert rc == R.EXIT_NOT_ACTIVE
    assert f"{R.PREFIX} NOT ACTIVE: mode big: undeclared PROTENIX_V1_BIG_RELP_LEAN: the line's levers are one set, not switchable" in err, err[-600:]

