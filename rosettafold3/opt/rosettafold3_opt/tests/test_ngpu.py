"""--n_gpu P, the memory mode's resource axis: the rules and their words are ``opt_core.mem.ngpu``'s (the one producer); this tree
gates with them before any fold process starts and prints their tokens in the ACTIVE / EXIT lines."""
import io
import sys
import types

import re
import pytest

from .. import cli, fold, report, stack

ngpu = stack._core.load("mem.ngpu")


def test_tokens_are_the_cores_exact_text():
    assert report.ngpu_fields({"n_gpu": 1}) == "n_gpu=1 sharding=none" == ngpu.active_fields(1)
    assert report.ngpu_fields({"n_gpu": 2, "sharding": "rowpair"}) == "n_gpu=2 sharding=rowpair" == ngpu.active_fields(2, "rowpair")
    assert report.ngpu_fields({}) == "n_gpu=1 sharding=none"                                   # absent == 1 (the base arms pass no flag)


def test_exit_line_carries_the_tokens():
    t = {"mode": "big", "graph_flags_imported": False, "n_gpu": 1, "sharding": "none"}
    assert report.tally_line(t) == "[rosettafold3-opt] EXIT mode=big: rf3.graph_flags never imported in this process (no fold ran here) n_gpu=1 sharding=none"
    t = {"mode": "big", "graph_flags_imported": True, "n_gpu": 4, "sharding": "rowpair"}
    assert report.tally_line(t).startswith("[rosettafold3-opt] EXIT mode=big rollouts=0 ") and report.tally_line(t).endswith(" n_gpu=4 sharding=rowpair")


@pytest.mark.parametrize("mode", ["exact", "fast", "off"])
def test_n_gpu_above_one_refused_outside_big_by_name(mode):
    with pytest.raises(stack.ActivationError) as e:
        stack.n_gpu_gate(mode, 2)
    assert str(e.value) == ngpu.REFUSE_MODE == "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)"
    assert stack.n_gpu_gate(mode, 1) == {"n_gpu": 1, "sharding": "none", "visible_gpus": None}   # P=1 is every mode's single-GPU path


def test_fewer_visible_than_requested_refused_by_name_never_auto_sized(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(stack.ActivationError) as e:
        stack.n_gpu_gate("big", 2)
    assert str(e.value) == "refused: n_gpu=2 visible=1" == ngpu.visible_refusal(2, 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert stack.visible_gpus() == 0


def test_bad_values_are_refused(monkeypatch):
    for bad in (0, -1, "two"):
        with pytest.raises(stack.ActivationError, match="a positive integer is required"):
            stack.n_gpu_gate("big", bad)
    monkeypatch.setenv(stack.ENV_N_GPU, "3")
    assert stack.n_gpu_requested(None) == 3 and stack.n_gpu_requested(2) == 2       # --n_gpu wins over the variable; absent -> the variable; else 1
    monkeypatch.delenv(stack.ENV_N_GPU)
    assert stack.n_gpu_requested(None) == 1


def test_env_name_declared():
    assert stack.ENV_N_GPU == "ROSETTAFOLD3_OPT_N_GPU" and stack.ENV_N_GPU in stack.ENV_NAMES
    from .. import _autoload
    assert stack.ENV_N_GPU in _autoload.ENV_NAMES


def test_run_refuses_before_any_process(monkeypatch, tmp_path, capsys):
    """fold.run(fast, n_gpu=2): the NOT ACTIVE line (the core's sentence) and NotActive; no interpreter is started."""
    inp = tmp_path / "in.json"; inp.write_text("[]")
    ck = tmp_path / "w.ckpt"; ck.write_text("x")
    monkeypatch.setattr(fold, "run_kit", lambda *a, **k: pytest.fail("no fold process may start"))
    with pytest.raises(fold.NotActive):
        fold.run("fast", inputs=str(inp), out_dir=str(tmp_path / "o"), ckpt=str(ck), seeds=[42], n_gpu=2)
    assert "[rosettafold3-opt] NOT ACTIVE: refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)" in capsys.readouterr().err


def test_cli_maps_the_refusal_to_exit_3(monkeypatch, tmp_path):
    def refuse(*a, **k):
        raise fold.NotActive("refused: n_gpu=2 visible=1")
    monkeypatch.setattr(fold, "run", refuse)
    inp = tmp_path / "in.json"
    inp.write_text('{"name": "x", "sequences": []}')
    assert cli.main(["pred", "--input", str(inp), "--out_dir", str(tmp_path / "o"), "--mode", "big", "--n_gpu", "2"]) == cli.EXIT_NOT_ACTIVE == 3


def test_kit_env_exports_the_axis_to_the_fold_process(monkeypatch):
    from ..modes import resolve
    res = resolve("exact", stack.kit_home(), fpf_home=stack.fpf_home())
    env = fold.kit_env(res, environ={}, n_gpu=1)
    assert env[stack.ENV_N_GPU] == "1" and env["ROSETTAFOLD3_OPT"] == "exact"


def test_rowpair_lever_line_names_the_axis():
    """The row-sharded pair stack's LEVER line: off with reason n_gpu:1 under big at P = 1; on with the axis tokens when installed at P > 1;
    strategy = the core's canonical id (F7.tensor_parallel)."""
    from .. import registry
    assert registry.LEVERS["rowpair"].strategy == ngpu.TP_LEVER == "F7.tensor_parallel"
    t1 = {"mode": "big", "graph_flags_imported": True, "n_gpu": 1, "sharding": "none"}
    st = report.lever_states({"levers": ["graph"], "mode": "big"}, t1)["rowpair"]
    assert st["state"] == "off" and st["reason"] == "n_gpu:1" and st["origin"] == "core"
    t2 = {"mode": "big", "graph_flags_imported": True, "n_gpu": 2, "sharding": "rowpair", "rowpair": {"installed": True, "sites": ["trimul", "triatt"], "state": {"layout": "B64", "peak_gib": 31.5}}}
    st = report.lever_states({"levers": ["graph", "rowpair"], "mode": "big"}, t2)["rowpair"]
    assert st["state"] == "on" and st["n_gpu"] == 2 and st["sharding"] == "rowpair" and st["sites"] == 2 and st["peak_gib"] == 31.5
    t2["levers"] = report.lever_states({"levers": ["graph", "rowpair"], "mode": "big"}, t2)
    line = [l for l in report.lever_lines(t2) if "name=rowpair" in l][0]
    assert line.startswith("[rosettafold3-opt] LEVER name=rowpair state=on impl=opt_core.mem.rowpair+rowpair.py origin=core strategy=F7.tensor_parallel n_gpu=2 sharding=rowpair"), line
    t3 = {"mode": "fast", "graph_flags_imported": True, "n_gpu": 1}
    assert report.lever_states({"levers": ["graph"], "mode": "fast"}, t3)["rowpair"]["reason"] == "not_in_mode:fast"


def test_launch_p1_is_the_plain_child_and_p_gt_1_goes_through_the_adapters_launcher(monkeypatch, tmp_path, capsys):
    """P = 1: one subprocess, nothing of the launcher imported; P = 2: rowpair.launch_argv's runner with its kwargs (one launcher seam);
    no seed token under P > 1: refused by name before any rank starts."""
    from .. import rowpair
    calls = {}
    class R: returncode = 0
    monkeypatch.setattr(fold.subprocess, "run", lambda cmd, **k: (calls.setdefault("p1", (cmd, k)), R())[1])
    cmd = ["py", "-m", "rf3.cli", "fold", "inputs=/i.json", "out_dir=/o/seed-42", "ckpt_path=/w.ckpt", "seed=42"]
    assert fold.launch(cmd, {"A": "1"}, 1) == 0 and calls["p1"][0] == cmd and fold.cmd_seed(cmd) == 42
    def fake_launch_argv(c, env, p):
        calls["argv"] = (list(c), dict(env), p)
        return (lambda stdout=None, timeout=None, **kw: (calls.setdefault("kw", (kw, timeout)), 7)[1]), {"n_gpu": p, "log_dir": "/o/seed-42.ranks"}
    monkeypatch.setattr(rowpair, "launch_argv", fake_launch_argv)
    assert fold.launch(cmd, {"A": "1"}, 2) == 7                                                    # the runner's rc is the fold's rc
    assert calls["argv"] == (cmd, {"A": "1"}, 2) and calls["kw"] == ({"n_gpu": 2, "log_dir": "/o/seed-42.ranks"}, None)
    calls.clear()
    assert fold.launch(cmd[:-1], {"A": "1"}, 2) == 7                                               # no seed= token: ONE seed is drawn for every rank and named
    drawn = calls["argv"][0]
    assert drawn[:-1] == cmd[:-1] and drawn[-1].startswith("seed=") and fold.cmd_seed(drawn) is not None
    assert re.search(r"n_gpu=2 seed=\d+ drawn=1", capsys.readouterr().err)


def test_the_gate_reads_the_adapters_plan(monkeypatch):
    """n_gpu_gate(big, 2) with 2 GPUs visible: the adapter's plan(p, mode) must name sharding=rowpair and at least one install, else refused by name."""
    from .. import rowpair
    monkeypatch.setattr(stack, "visible_gpus", lambda: 2)
    monkeypatch.setattr(stack, "in_rank_process", lambda: False)
    g = stack.n_gpu_gate("big", 2)
    assert g["n_gpu"] == 2 and g["sharding"] == "rowpair"
    monkeypatch.setattr(rowpair, "plan", lambda p, mode=None: {"sharding": "rowpair", "installs": [], "reason": "nothing to shard here"})
    with pytest.raises(stack.ActivationError, match=r"refused: n_gpu=2: the row-sharding adapter declines \(nothing to shard here\)"):
        stack.n_gpu_gate("big", 2)


def test_pred_n_gpu_reaches_the_launcher(monkeypatch, tmp_path):
    """`pred --mode big --n_gpu 2` starts the ranks with P = 2 — the axis is threaded run → run_kit → kit_env / launch; a fold that
    ran one process under --n_gpu 2 would be a silent fallback (this test is its guard)."""
    seen = {}
    inp = tmp_path / "in.json"; inp.write_text("[]")
    ck = tmp_path / "w.ckpt"; ck.write_text("x")
    monkeypatch.setattr(stack, "n_gpu_gate", lambda mode, n: {"n_gpu": int(n or 1), "sharding": "rowpair" if int(n or 1) > 1 else "none", "visible_gpus": 2})
    def fake_run_kit(mode, **k):
        seen["run_kit_n_gpu"] = k.get("n_gpu")
        return {"status": "PASS", "runs": [], "tallies": {}, "python": "py", "tree_state": "patched", "tree": "x", "failures": []}
    monkeypatch.setattr(fold, "run_kit", fake_run_kit)
    monkeypatch.setattr(fold._settings, "describe", lambda toks=None: {})
    fold.run("big", inputs=str(inp), out_dir=str(tmp_path / "o"), ckpt=str(ck), seeds=[42], n_gpu=2)
    assert seen == {"run_kit_n_gpu": 2}


def test_a_fold_on_fewer_gpus_than_requested_is_never_a_pass(capsys):
    """Fail-closed: the P the fold processes report at exit (tally n_gpu + one tally per rank) must equal --n_gpu, else
    `NOT ACTIVE: reason=n_gpu_mismatch requested=P active=Q ranks_reported=R` and status FAIL (the CLI maps it to exit 3)."""
    runs = [{"seed": 42, "rc": 0, "ranks_reported": 1}]
    assert fold.n_gpu_failures(2, runs, {"42": {"n_gpu": 1}}) == ["n_gpu_mismatch requested=2 active=1 ranks_reported=1 (seed 42)"]
    assert fold.n_gpu_failures(1, runs, {"42": {"n_gpu": 1}}) == []
    assert fold.n_gpu_failures(2, [{"seed": 42, "rc": 0, "ranks_reported": 2}], {"42": {"n_gpu": 2}}) == []
    assert fold.n_gpu_failures(2, [{"seed": 42, "rc": 0, "ranks_reported": 1}], {"42": {"n_gpu": 2}}) == ["n_gpu_mismatch requested=2 active=2 ranks_reported=1 (seed 42)"]
    assert fold.n_gpu_failures(4, [{"seed": 7, "rc": 0}], {}) == ["n_gpu_mismatch requested=4 active=0 ranks_reported=0 (seed 7)"]


def test_only_measured_n_gpu_values_are_accepted(monkeypatch):
    """--n_gpu outside stack.N_GPU_SUPPORTED (the values with a measured statement) is refused by name before any GPU is probed."""
    monkeypatch.setattr(stack, "visible_gpus", lambda: 8)
    monkeypatch.setattr(stack, "in_rank_process", lambda: False)
    assert 1 in stack.N_GPU_SUPPORTED and 2 in stack.N_GPU_SUPPORTED
    bad = max(stack.N_GPU_SUPPORTED) + 1
    with pytest.raises(stack.ActivationError, match=rf"refused: n_gpu={bad} is not a supported value on rosettafold3 \(supported: "):
        stack.n_gpu_gate("big", bad)
    with pytest.raises(stack.ActivationError, match=r"requires --mode big"):      # the mode rule still comes first
        stack.n_gpu_gate("fast", bad)
