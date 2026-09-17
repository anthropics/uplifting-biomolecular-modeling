"""The big line's partial-unit opt-out: a flag on pred / warm (usage-refused under the other modes), exported to the fold processes as
ROSETTAFOLD3_BIG_ALLOW_PARTIAL=1 only from the flag (a caller's value is dropped), read kit-side in the fold process and handed to the core's
registry as ``allow_partial``; the partial verdict's lines name --allow-partial."""
import json

import pytest

from .. import big, cli, fold, modes


def test_the_flag_parses_on_pred_and_warm_and_is_refused_by_name_outside_big(tmp_path, capsys, monkeypatch):
    inp = tmp_path / "in.json"; inp.write_text(json.dumps([{"name": "x", "components": [{"seq": "AAAA", "chain_id": "A"}]}]))
    for argv in (["pred", "--input", str(inp), "--out_dir", str(tmp_path / "o"), "--mode", "exact", "--allow-partial"],
                 ["pred", "--input", str(inp), "--out_dir", str(tmp_path / "o"), "--mode", "off", "--allow-partial"],
                 ["warm", "--mode", "fast", "--allow-partial"]):
        assert cli.main(argv) == cli.EXIT_USAGE
        err = capsys.readouterr().err
        assert "--allow-partial is the big line's partial-unit opt-out; --mode" in err and "has no partial units" in err, err
    seen = {}
    monkeypatch.setattr(fold, "run", lambda mode, **k: (seen.update(k), {"status": "PASS", "runs": [], "tree_state": "patched", "tree": "x", "settings": {}})[1])
    monkeypatch.setattr(cli, "_ckpt", lambda a: str(inp))
    monkeypatch.setattr(fold, "summary_line", lambda rec: "x")
    cli.main(["pred", "--input", str(inp), "--out_dir", str(tmp_path / "o2"), "--mode", "big", "--allow-partial"])
    assert seen.get("allow_partial") is True
    seen.clear()
    cli.main(["pred", "--input", str(inp), "--out_dir", str(tmp_path / "o3"), "--mode", "big"])
    assert seen.get("allow_partial") is False


def test_pred_exports_the_word_to_its_fold_processes_from_the_flag_only(monkeypatch):
    res = modes.resolve(mode="big")
    base = {"PATH": "/usr/bin", big.ALLOW_PARTIAL_ENV: "1"}                       # a caller's value: dropped
    env = fold.kit_env(res, environ=dict(base), n_gpu=1)
    assert big.ALLOW_PARTIAL_ENV not in env
    env = fold.kit_env(res, environ={"PATH": "/usr/bin"}, n_gpu=1, allow_partial=True)
    assert env[big.ALLOW_PARTIAL_ENV] == "1" and big.allow_partial_of(env) is True
    assert big.allow_partial_of({"PATH": "/usr/bin"}) is False and big.allow_partial_of({big.ALLOW_PARTIAL_ENV: "0"}) is False


def test_the_fold_process_reads_the_word_and_hands_it_to_the_core_as_allow_partial(monkeypatch):
    """big.selection: the line's levers with explicit switches (only the site-owned levers off under n_gpu > 1), the opt-out read kit-side
    from the environment word; the core reads no environment."""
    calls = []

    class FakeMem:
        OPT_OUT = "--allow-partial"

        def selection(self, levers, *, switches=None, allow_partial=False):
            calls.append({"levers": list(levers), "switches": dict(switches or {}), "allow_partial": allow_partial})
            return "sel"
    monkeypatch.setattr(big, "_mem", lambda: FakeMem())
    monkeypatch.setattr(big, "compose", lambda environ=None: (None, type("L", (), {"levers": tuple(big.LEVERS)})()))
    assert big.selection({big.ALLOW_PARTIAL_ENV: "1"}) == "sel"
    assert calls[-1] == {"levers": list(big.LEVERS), "switches": {lv: False for lv in big.arm_served()}, "allow_partial": True}   # transition_chunk: its site is the arm's ttr (big.ARM_SERVES)
    big.selection({}, n_gpu=2)
    assert calls[-1]["allow_partial"] is False and calls[-1]["switches"] == {lv: False for lv in {**big.ROWPAIR_OWNS, **big.arm_served()}} == big.switches_for(2)
    assert big.switches_for(1) == {lv: False for lv in big.arm_served()}          # {'transition_chunk': False}: the arm's ttr holds that site (big.ARM_SERVES)


def test_the_partial_lines_name_the_opt_out_of_the_route(monkeypatch):
    """Ctx.opt_out: --allow-partial in a fold pred / warm started (ROSETTAFOLD3_OPT_TALLY_FILE set), the environment word in an rf3 process the
    user activated through the environment; the words are the core's PARTIAL lines."""
    from .. import stack
    assert big.opt_out_of({stack.ENV_TALLY_FILE: "/o/.tally_seed-42.json"}) == "--allow-partial"
    assert big.opt_out_of({"ROSETTAFOLD3_OPT": "big"}) == "ROSETTAFOLD3_BIG_ALLOW_PARTIAL=1" == f"{big.ALLOW_PARTIAL_ENV}=1"
    record = big._core.load("mem.record")
    line = record.PARTIAL_REFUSED.format(prefix="[rosettafold3-opt]", detail="opm_chunk: u1 never marked", opt_out="--allow-partial")
    assert line == "[rosettafold3-opt] NOT ACTIVE: big partial — opm_chunk: u1 never marked; exit 3 (--allow-partial records and proceeds)"
    line = record.PARTIAL_ALLOWED.format(prefix="[rosettafold3-opt]", detail="opm_chunk: u1 never marked", opt_out="ROSETTAFOLD3_BIG_ALLOW_PARTIAL=1")
    assert line == "[rosettafold3-opt] PARTIAL allowed: opm_chunk: u1 never marked (ROSETTAFOLD3_BIG_ALLOW_PARTIAL=1, recorded)"


def test_a_lever_word_in_the_environment_is_refused_by_name():
    """The line is one lever set: ROSETTAFOLD3_BIG_<LEVER> / _<SETTING> words are undeclared names (refused at start by the hook, the CLI and
    the activation); the environment route's opt-out word is the one declared name under the prefix."""
    from .. import _autoload, stack
    assert _autoload.BIG_NAMES == (big.ALLOW_PARTIAL_ENV,)
    env = {"ROSETTAFOLD3_BIG_OPM_CHUNK": "0", "ROSETTAFOLD3_BIG_OPM_CHUNK_ROWS": "64", big.ALLOW_PARTIAL_ENV: "1", "ROSETTAFOLD3_OPT": "big"}
    assert stack.undeclared_env(env) == {"ROSETTAFOLD3_BIG_OPM_CHUNK": "0", "ROSETTAFOLD3_BIG_OPM_CHUNK_ROWS": "64"}
