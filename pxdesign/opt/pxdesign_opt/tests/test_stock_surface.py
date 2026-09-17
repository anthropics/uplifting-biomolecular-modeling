"""The design verb takes what upstream's `pxdesign infer` takes: upstream's flag spellings (`-i/--input`, `-o/--dump_dir`,
`--load_checkpoint_dir`; `--tasks/--out_dir/--ckpt_dir` accepted too), boolean knobs in upstream's own words, `--seeds` optional on every mode
(one clock-derived seed, upstream's rule), a YAML or JSON task file, a dump directory in any state, any further `pxdesign infer` argument passed
through unchanged; LAYERNORM_TYPE is set from `--use_fast_ln` on every mode alike; the stock pins are a report of `check`, never a gate."""
import json
import os

import pytest

from . import _stubs

from .conftest import run_routes_text

def test_boolean_knobs_take_upstreams_words(tree):
    """`--use_msa` / `--use_fast_ln` accept every word upstream's get_bool_value accepts (true|t|yes|y|1, false|f|no|n|0), rendered true|false
    for the child; any other word is the usage error upstream itself raises on it."""
    import argparse
    from pxdesign_opt import options
    pins = json.load(open(os.path.join(tree, "stock", "PINS.json")))
    ap = argparse.ArgumentParser(); options.add_arguments(ap)
    for word, want in (("1", "true"), ("yes", "true"), ("T", "true"), ("0", "false"), ("n", "false"), ("False", "false")):
        a = ap.parse_args(["--use_msa", word])
        assert options.from_args(a, pins=pins).values["use_msa"] == want, word
    with pytest.raises(SystemExit) as e:
        ap.parse_args(["--use_msa", "maybe"])
    assert e.value.code == 2
    assert options.bool_word("Y") == "true" and options.bool_word("no") == "false"


def test_design_takes_upstreams_flag_spellings_and_forwards_the_rest():
    """`-i/--input`, `-o/--dump_dir`, `--load_checkpoint_dir` parse to the verb's own fields (`--tasks/--out_dir/--ckpt_dir` too); an argument the
    verb does not name is kept for `pxdesign infer` in the caller's order (a `--` separator is dropped); check/warm still refuse an unknown flag."""
    from pxdesign_opt import cli
    ap = cli.build_parser()
    for argv in (["design", "-i", "t.json", "-o", "o", "--load_checkpoint_dir", "/c"], ["design", "--input", "t.json", "--dump_dir", "o", "--ckpt_dir", "/c"],
                 ["design", "--tasks", "t.json", "--out_dir", "o", "--ckpt_dir", "/c"]):
        a, rest = ap.parse_known_args(argv)
        assert (a.tasks, a.out_dir, a.ckpt_dir, rest) == ("t.json", "o", "/c", []), argv
    a, rest = ap.parse_known_args(["design", "-i", "t.json", "-o", "o", "--sample_diffusion_chunk_size", "2", "--det", "1", "--", "--use_deepspeed_evo_attention", "true"])
    assert a.det == 1 and [x for x in rest if x != "--"] == ["--sample_diffusion_chunk_size", "2", "--use_deepspeed_evo_attention", "true"]
    with pytest.raises(SystemExit) as e:
        cli.main(["check", "--mode", "exact", "--sample_diffusion_chunk_size", "2"])
    assert e.value.code == cli.EXIT_USAGE


def test_layernorm_type_follows_use_fast_ln_on_every_mode(tree):
    """`options.apply_layernorm_env`: `--use_fast_ln true` (upstream's default) -> LAYERNORM_TYPE=fast_layernorm, `false` -> the variable removed
    (OpenFold's LayerNorm) — one rule for the kit modes' own environment and the stock child's; the default word is configs/<gpu>.env's export."""
    from pxdesign_opt import options
    pins = json.load(open(os.path.join(tree, "stock", "PINS.json")))
    for start in ({}, {"LAYERNORM_TYPE": "fast_layernorm"}, {"LAYERNORM_TYPE": "torch"}):
        env = dict(start); options.apply_layernorm_env(options.resolve(pins=pins), env)
        assert env == {"LAYERNORM_TYPE": "fast_layernorm"}, start
        env = dict(start); options.apply_layernorm_env(options.resolve(use_fast_ln="false", pins=pins), env)
        assert "LAYERNORM_TYPE" not in env, start
    cfg = open(os.path.join(tree, "configs", "h100.env"), encoding="utf-8").read()
    assert "export LAYERNORM_TYPE=${LAYERNORM_TYPE-" + options.LAYERNORM_VALUE + "}" in cfg


def test_use_fast_ln_false_is_served_by_the_kit_modes(fresh_stack, monkeypatch, capsys, tmp_path):
    """`--use_fast_ln false` on a kit mode is no usage error: the verb clears LAYERNORM_TYPE and proceeds to activation (refused here for the test
    box, which has no GPU — never for the flag); `--seeds` absent is not refused either."""
    from pxdesign_opt import cli
    stack = fresh_stack
    _stubs.install_torch(cuda=False); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack, cuda=False)
    monkeypatch.setenv("LAYERNORM_TYPE", "fast_layernorm")
    tasks = tmp_path / "tasks.json"; tasks.write_text("[]")
    rc = cli.main(["design", "--mode", "fast", "-i", str(tasks), "-o", str(tmp_path / "out"), "--use_fast_ln", "false", "--seeds", "1"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_NOT_ACTIVE and "not served" not in err and "NOT ACTIVE" in err, err
    stack.reset_for_tests(); _stubs.fake_box(monkeypatch, stack, cuda=False)
    rc = cli.main(["design", "--mode", "exact", "--tasks", str(tasks), "--out_dir", str(tmp_path / "out")])          # no --seeds: not a usage error on a kit mode
    err = capsys.readouterr().err
    assert rc == cli.EXIT_NOT_ACTIVE and "--seeds is required" not in err and "NOT ACTIVE" in err


def test_check_reports_the_stock_pins(fresh_stack, monkeypatch, capsys):
    """`check` prints one `[pxdesign-opt] PINS pinned=<0|1> stack=<status> …` line (stack.pins_report through stock/check_pins.py); on an
    installed upstream that is not the pinned stock a kit mode's dry run names the refusal (`would_refuse=`, exit 3 — the code design's NOT
    ACTIVE exits with) while `--mode off` is not gated; `--json` carries the report under "pins_check"."""
    from pxdesign_opt import cli
    stack = fresh_stack
    _stubs.install_torch(); _stubs.install_upstream()
    _stubs.fake_box(monkeypatch, stack, pinned=False)
    assert cli.main(["check", "--mode", "exact"]) == 3                                 # the mode would refuse (not the pinned stock): the refusal's code
    err = capsys.readouterr().err
    assert "[pxdesign-opt] PINS pinned=0 stack=OPEN differs=pxdesign 0.1.0: installed from git commit deadbeef; want f788441" in err, err
    assert "would_refuse='the installed upstream is not the pinned stock — pxdesign 0.1.0: installed from git commit deadbeef; want f788441 (STOCK.md; run.sh install)'" in err, err
    stack.reset_for_tests(); _stubs.fake_box(monkeypatch, stack, pinned=False)
    assert cli.main(["check", "--mode", "off"]) == 0
    capsys.readouterr()
    stack.reset_for_tests(); _stubs.fake_box(monkeypatch, stack, pinned=True)
    assert cli.main(["check", "--mode", "exact", "--json"]) == 0
    out, err = capsys.readouterr()
    assert "[pxdesign-opt] PINS pinned=1 stack=OPEN" in err and json.loads(out)["pins_check"]["pinned"] is True


def test_exit_codes_are_the_shared_cores_four(tree):
    """0 ok | 1 failed / outputs incomplete | 2 usage | 3 not active / stock environment not clean — no kit-specific 4 or 5."""
    from pxdesign_opt import cli, report, stock_infer
    assert (cli.EXIT_OK, cli.EXIT_FAIL, cli.EXIT_USAGE, cli.EXIT_NOT_ACTIVE) == (0, 1, 2, 3) == (report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE)
    assert stock_infer.EXIT_NOT_STOCK == 3 and stock_infer.EXIT_INCOMPLETE == 1
    for mod in (cli, report, stock_infer):
        assert not hasattr(mod, "EXIT_UNCLEAN") and getattr(mod, "EXIT_INCOMPLETE", 1) == 1
    src = open(os.path.join(tree, "run.sh"), encoding="utf-8").read()
    assert "5 incomplete" not in src and "check_pins.py" not in run_routes_text(src)        # the install step checks the stock pins once; the run routes never gate on them


def test_seed_rule_and_done_word_without_seeds():
    """No --seeds: infer_loop.seed_list derives exactly one seed with upstream's function; given seeds pass as ints. The DONE line's first word is
    INCOMPLETE when designs != expected (kit modes and stock alike name a short run)."""
    from pxdesign_opt import infer_loop, report
    assert infer_loop.seed_list([], lambda ns: ns // 10, now_ns=4242) == [424] and infer_loop.seed_list(["7", 8], lambda ns: 0) == [7, 8]
    assert report.done_line({"mode": "fast", "n_designs": 4, "n_tasks": 1, "n_seeds": 1, "expected": 5}).startswith("[pxdesign-opt] INCOMPLETE mode=fast designs=4 ")
    assert report.done_line({"mode": "fast", "n_designs": 5, "n_tasks": 1, "n_seeds": 1, "expected": 5}).startswith("[pxdesign-opt] DONE mode=fast designs=5 ")
    assert report.DONE_RE.match(report.done_line({"mode": "fast", "n_designs": 4, "n_tasks": 1, "n_seeds": 1, "expected": 5})).group("word") == "INCOMPLETE"


