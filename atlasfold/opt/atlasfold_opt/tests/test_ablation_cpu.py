"""MODEL_OPT_LEVERS_OFF — the ablation switch (CPU; no model, no GPU): parsing, by-name refusals, the row filter, the ACTIVE token, the
LEVER state=off lines' grammar, and check's PLAN lines."""
import types
import pytest

from atlasfold_opt import ablation as A, modes, registry


def test_env_word_and_tokens_are_the_family_convention():
    assert A.ENV == "MODEL_OPT_LEVERS_OFF" and A.REASON == "levers_off" and A.TOKEN == "ablated"


def test_requested_parses_in_order_folding_blanks_and_duplicates():
    assert A.requested({}) == [] and A.requested({A.ENV: ""}) == [] and A.requested({A.ENV: " , "}) == []
    assert A.requested({A.ENV: "diffusion_bf16, trimul_v4 ,diffusion_bf16,,denoiser_graph"}) == ["diffusion_bf16", "trimul_v4", "denoiser_graph"]


def test_validate_passes_members_and_refuses_by_name():
    assert A.validate("fast", []) == []
    assert A.validate("fast", ["diffusion_bf16", "denoiser_graph"]) == ["diffusion_bf16", "denoiser_graph"]
    with pytest.raises(A.AblationError, match="bogus: not a lever of this kit"):
        A.validate("fast", ["bogus"])
    with pytest.raises(A.AblationError, match="trimul_v4: not in mode exact's row"):
        A.validate("exact", ["trimul_v4"])
    with pytest.raises(A.AblationError, match="mode off applies no lever"):
        A.validate("off", ["diffusion_bf16"])
    with pytest.raises(A.AblationError, match="lever_report: the lever report itself"):
        A.validate("fast", ["lever_report"])
    everything = [l for l in modes.MODES["exact"] if l != "lever_report"]
    with pytest.raises(A.AblationError, match="lever_report: the lever report itself"):
        A.validate("exact", everything + ["lever_report"])
    assert A.validate("exact", everything) == everything          # lever_report stays: not empty, allowed


def test_row_without_keeps_the_row_order():
    row = A.row_without("fast", ["flash_triattn", "trimul_v4"])
    assert row == [l for l in modes.MODES["fast"] if l not in ("flash_triattn", "trimul_v4")]
    assert A.row_without("exact", []) == modes.MODES["exact"]
    assert A.token(["a", "b"]) == "a,b" and A.token([]) == ""


def test_lever_lines_follow_the_core_grammar():
    pytest.importorskip("opt_core")
    lines = A.lever_lines("atlasfold-opt", ["triatt_block", "trimul_v4", "ln_bf16"])
    assert len(lines) == 3
    assert lines[0].startswith("[atlasfold-opt] LEVER name=LOCAL.atlasfold.triatt_block state=off reason=levers_off impl=atlasfold_opt.hooks.triatt_block origin=")
    assert lines[1].startswith("[atlasfold-opt] LEVER name=F2.trimul state=off reason=levers_off impl=") and " lever=trimul_v4" in lines[1]
    assert " origin=kit" in lines[2] and " lever=ln_bf16" in lines[2]
    for ln in lines:
        assert all("=" in tok for tok in ln.split(" ")[2:]), ln      # every token after the verb is key=value, no blanks inside values


def _stub_check(monkeypatch):
    from atlasfold_opt import _core, stack, cli
    from atlasfold_opt.hooks import denoiser_graph as DG
    monkeypatch.setattr(_core, "gate", lambda: {"installed": {"version": "x"}, "pinned": {"version": "x", "path": "p"}})
    monkeypatch.setattr(stack, "_gpu_probe", lambda: {"name": "stub", "cc": "9.0", "torch": "2.7.1"})
    monkeypatch.setattr(stack, "_stock_facts", lambda: {"importable": True, "dist_version": None, "cuequivariance_torch": None})
    monkeypatch.setattr(cli, "kernel_probe", lambda: [])
    monkeypatch.setattr(DG, "probe", lambda: {"kernel": "cuda_graph", "ok": False, "routed": False, "resolved": None, "reason": "capture_failed:RuntimeError: boom"})
    monkeypatch.delenv("ATLASFOLD_WEIGHTS_DIR", raising=False)
    return cli


def test_check_prints_plan_state_off_and_the_dry_run_token(monkeypatch, capsys):
    cli = _stub_check(monkeypatch)
    monkeypatch.setenv(A.ENV, "denoiser_graph,diffusion_bf16")
    rc = cli.cmd_check(types.SimpleNamespace(mode="fast", weights=None), [])
    err = capsys.readouterr().err
    assert rc == 0, err                                                 # denoiser_graph ablated: a box that cannot capture is not refused for it
    assert "DRY-RUN mode=fast ablated=denoiser_graph,diffusion_bf16,graph_reuse config=" in err   # graph_reuse requires denoiser_graph: ablated with it, by name
    assert "PLAN lever=denoiser_graph state=off reason=levers_off class=fast" in err and "PLAN lever=diffusion_bf16 state=off reason=levers_off" in err
    assert "PLAN lever=trimul_v4 class=fast" in err


def test_check_refuses_an_unknown_name_by_name(monkeypatch, capsys):
    cli = _stub_check(monkeypatch)
    monkeypatch.setenv(A.ENV, "no_such_lever")
    rc = cli.cmd_check(types.SimpleNamespace(mode="fast", weights=None), [])
    err = capsys.readouterr().err
    assert rc == 3 and "NOT ACTIVE: reason=levers_off_refused: MODEL_OPT_LEVERS_OFF refused" in err and "no_such_lever: not a lever of this kit" in err


def test_unset_leaves_check_byte_for_byte(monkeypatch, capsys):
    cli = _stub_check(monkeypatch)
    monkeypatch.delenv(A.ENV, raising=False)
    cli.cmd_check(types.SimpleNamespace(mode="exact", weights=None), [])
    err = capsys.readouterr().err
    assert " ablated=" not in err and "reason=levers_off" not in err


def test_dependency_closure_steps_aside_by_name():
    # a lever OF another lever's served path goes off WITH its prerequisite, named, in row order — never a partial activation
    assert A.dropped("fast", ["triatt_block"]) == [("triattn_core", "requires:triatt_block(levers_off)"), ("pair_block_residual", "requires:triatt_block(levers_off)")]
    assert A.dropped("fast", ["pair_transition"]) == [("pair_block_residual", "requires:pair_transition(levers_off)")]
    assert A.dropped("fast", ["triattn_core"]) == [] and A.dropped("exact", ["ln_bf16"]) == [] and A.dropped("fast", []) == []
    why = A.reasons("fast", ["pair_transition", "triatt_block"])
    assert list(why) == ["pair_transition", "triatt_block", "triattn_core", "pair_block_residual"]          # the request as given, then the closure in row order
    assert why["triatt_block"] == "levers_off" and why["triattn_core"] == "requires:triatt_block(levers_off)" and why["pair_block_residual"] == "requires:triatt_block(levers_off)"
    row = A.row_without("fast", ["triatt_block"])
    assert "triatt_block" not in row and "triattn_core" not in row and "pair_block_residual" not in row and "pair_transition" in row and row[0] == "trimul_v4"
    lines = A.lever_lines("[t]", list(A.reasons("fast", ["triatt_block"])), A.reasons("fast", ["triatt_block"]))
    assert any("name=LOCAL.atlasfold.triattn_core state=off reason=requires:triatt_block(levers_off)" in l for l in lines)
    assert any("name=LOCAL.atlasfold.triatt_block state=off reason=levers_off" in l for l in lines)


def test_check_names_the_closure_and_exits_0(monkeypatch, capsys):
    cli = _stub_check(monkeypatch)
    monkeypatch.setenv(A.ENV, "triatt_block")
    rc = cli.cmd_check(types.SimpleNamespace(mode="fast", weights=None), [])
    err = capsys.readouterr().err
    assert "DRY-RUN mode=fast ablated=triatt_block,triattn_core,pair_block_residual config=" in err
    assert "PLAN lever=triatt_block state=off reason=levers_off" in err
    assert "PLAN lever=triattn_core state=off reason=requires:triatt_block(levers_off) class=fast" in err
    assert "PLAN lever=pair_block_residual state=off reason=requires:triatt_block(levers_off)" in err
    assert "PLAN lever=pair_transition class=fast" in err and rc in (0, 3)     # rc 3 only from the stubbed cuda_graph probe (denoiser_graph kept), never from the closure
