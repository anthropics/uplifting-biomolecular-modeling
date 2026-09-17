"""The exit rule on partial activation, every entry point: report.partial_line's literal fixed parts; pred / serve / warm print it after a
run (stack.partial_activation); --allow-partial records the opt-out and proceeds; the stock route
refuses the flag; the worker route refuses the stock route's arguments (never ignores them)."""
import os
import subprocess
import sys

import pytest

from .. import cli, modes, report as rep, stack


@pytest.fixture(autouse=True)
def _tally():
    rep.reset_tally(); yield; rep.reset_tally()


def test_the_family_line_literal_parts():
    assert rep.EXIT_NOT_ACTIVE == 3
    assert rep.PARTIAL_REFUSED_FMT == "{prefix} NOT ACTIVE: partial activation — {detail}; exit {code} (--allow-partial records and proceeds)"
    assert rep.PARTIAL_ALLOWED_FMT == "{prefix} PARTIAL allowed: {detail} (--allow-partial, recorded)"
    assert rep.partial_line(["mask2: x"], False) == "[boltz2-opt] NOT ACTIVE: partial activation — mask2: x; exit 3 (--allow-partial records and proceeds)"
    assert rep.partial_line(["mask2: x", "f2: y"], True) == "[boltz2-opt] PARTIAL allowed: mask2: x; f2: y (--allow-partial, recorded)"
    assert rep.partial_lines({"levers_fallback": ["mask2: x"], "gates": {"f2": "size gate"}}) == [rep.partial_line(["mask2: x"], True), "[boltz2-opt] GATE f2: size gate"]
    assert rep.partial_lines({"levers_fallback": [], "gates": {}}) == []


def test_the_stock_route_refuses_the_flag_and_the_worker_route_refuses_stock_arguments(tmp_path, capsys):
    assert cli.main(["check", "--mode", "off", "--allow-partial"]) == rep.EXIT_USAGE
    assert "--allow-partial names a kit fallback of a row lever" in capsys.readouterr().err
    y = str(tmp_path / "a.yaml"); open(y, "w").write("version: 1\n")
    assert cli.main(["pred", "--mode", "exact", "--input", y, "--out_dir", str(tmp_path / "o"), "--", "--use_potentials"]) == rep.EXIT_USAGE   # (boltz predict's own options by name are inputs of every mode: test_predict_settings_flags.py)
    err = capsys.readouterr().err
    assert "belong to the stock route (--mode off)" in err and "--mode exact" in err, err
    assert cli.main(["pred", "--mode", "exact", "--input", y, "--out_dir", str(tmp_path / "o"), "--det", "1"]) == rep.EXIT_USAGE      # the deterministic recipe is the stock route's
    assert "--det belong to the stock route (--mode off)" in capsys.readouterr().err
    for gone in (["--settings", "kit"], ["--no_pae"]):                                                  # no presets, no kit-invented settings switches: not options of this CLI at all
        with pytest.raises(SystemExit) as ex:
            cli.main(["pred", "--mode", "exact", "--input", y, "--out_dir", str(tmp_path / "o")] + gone)
        assert ex.value.code == 2
    capsys.readouterr()
    assert not os.path.exists(tmp_path / "o")


def test_the_command_line_end_to_end(tmp_path):
    """The refusal as a caller sees it: the process exit code and the line, `check --mode exact` on a box with no CUDA device."""
    env = {k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}
    env["BOLTZ_CACHE"] = str(tmp_path / "nocache")
    r = subprocess.run([sys.executable, "-m", "boltz2_opt", "check", "--mode", "exact"], capture_output=True, text=True, env=env)
    assert r.returncode in (0, 3), r.stdout + r.stderr
    if r.returncode == 3:                                                    # this box: the gate (no GPU / cache) or the partial plan
        assert "[boltz2-opt] NOT ACTIVE:" in r.stdout and "DRY-RUN ok" not in r.stdout


def test_a_mask_scope_error_is_a_partial_activation():
    """boltz_trunk_levers.py:143 counts mask_scope_error when the mask-elision scope cannot classify a mask (that call runs the stock mask
    arithmetic). The counter feeds the partial-activation gate: a non-zero count under a row carrying mask2 is a FALLBACK —
    the verb exits by the partial-activation rule — never a silent counter; zero is quiet."""
    assert "mask2" in modes.levers("exact") and "mask2" in modes.levers("fast")
    base = {"levers": ["resid", "mask2"], "graph": "graph", "hoist_level": 2}          # other evidence keys classify nothing
    assert stack.partial_activation("exact", dict(base, mask_scope_errors=0)) == ([], {})
    fb, gates = stack.partial_activation("exact", dict(base, mask_scope_errors=3))
    assert fb == ["mask2: 3 calls could not classify the pair mask (mask_scope_error) and ran the stock mask arithmetic"] and gates == {}
    fb, _ = stack.partial_activation("fast", dict(base, mask_scope_errors=1))
    assert len(fb) == 1 and fb[0].startswith("mask2: 1 calls")
    src = open(stack.__file__).read()
    assert 'ev["mask_scope_errors"] = int(st.get("mask_scope_error") or 0)' in src, "evidence() reads the worker's stats.mask_scope_error"
