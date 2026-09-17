"""The `--n_gpu` axis on the record lines: the exact token text `n_gpu=P sharding=rowpair` (a big/tp rank of P GPUs) and
`n_gpu=1 sharding=none` (every one-GPU route) on the ACTIVE, DRY-RUN and exit lines and on the tp lever's LEVER line — produced by
opt_core.mem.ngpu (the one producer every cofold kit imports), never spelled in the kit."""
from opt_core.mem import ngpu

from openfold3_ob0_opt import modes, report, stack
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()


def test_token_text_comes_from_the_core_producer():
    assert report.ngpu_fields({"n_gpu": 2}) == ngpu.active_fields(2, "rowpair") == "n_gpu=2 sharding=rowpair"
    assert report.ngpu_fields({"n_gpu": 1}) == report.ngpu_fields({}) == ngpu.active_fields(1, "rowpair") == "n_gpu=1 sharding=none"
    assert report.ngpu_fields({"n_gpu": 8}) == "n_gpu=8 sharding=rowpair" and report.SHARDING_SCHEME == "rowpair" and "rowpair" in ngpu.SCHEMES


def test_active_and_exit_lines_carry_the_tokens():
    one = {"active": True, "mode": "fast", "line": None, "openfold3_version": "0.4.1", "levers_requested": ["fast_init"], "hooks": ["fast_inference"], "n_gpu": 1}
    assert " n_gpu=1 sharding=none" in report.activation_line(one)
    assert " arm_complete=pending n_gpu=1 sharding=none " in report.exit_tally_line(one, 1) + " "
    rank = dict(one, mode="big", line="tp", levers_requested=["fast_init", "tp_shard_s"], levers_applied=["tp_shard_s"], hooks=["tp", "fast_inference"], n_gpu=4)
    assert " n_gpu=4 sharding=rowpair" in report.activation_line(rank)
    ex = report.exit_tally_line(rank, 1)
    assert " n_gpu=4 sharding=rowpair" in ex.splitlines()[0]
    tp_lever = [l for l in ex.splitlines() if l.startswith("[openfold3_ob0-opt] LEVER name=tp_shard_s ")]
    assert len(tp_lever) == 1 and "state=on" in tp_lever[0] and " n_gpu=4 sharding=rowpair" in tp_lever[0], tp_lever
    dry = {"dry_run": True, "env": {"OF3TP_TRIMUL_SUB": "128"}, "mode": "big", "line_spelling": "tp", "openfold3_version": "0.4.1", "levers_requested": ["tp_shard_s"], "hooks": ["tp"], "n_gpu": 2}
    assert " n_gpu=2 sharding=rowpair" in report.activation_line(dry)
    assert "n_gpu=" not in report.activation_line({"active": False, "reason": "x"})          # a refusal line carries the reason only


def test_a_rank_process_reports_its_world_and_one_gpu_routes_report_one():
    assert modes.rank_world({"OF3TP_WORLD": "4"}) == 4 and modes.rank_world({}) == 1 and modes.rank_world({"OF3TP_WORLD": "x"}) == 1
    rep = stack.activate("big", dry_run=True, n_gpu=2, home=HOME, environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OB0_OPT_HOME": HOME}, log=False)
    assert rep["n_gpu"] == 2 and " n_gpu=2 sharding=rowpair" in report.activation_line(rep), report.activation_line(rep)
    rep = stack.activate("big", dry_run=True, home=HOME, environ={"OPENFOLD3_OB0_OPT_HOME": HOME}, log=False)
    assert rep["n_gpu"] == 1 and rep["line"] == modes.BIG_RESIDENT and " n_gpu=1 sharding=none" in report.activation_line(rep)
    # the structural rule: at n_gpu=1 NOTHING of the tensor-parallel line is installed — no tp hook directory on the path, no OF3TP_* switch
    # exported, the tp lever absent from the line (today's one-GPU bytes; modes.LINES is the table)
    assert "tp" not in rep["hooks"] and "tp_shard_s" not in rep["levers_requested"] and not [k for k in rep["env"] if k.startswith("OF3TP_")], rep["hooks"]
    for n in ("1", None):
        r1 = modes.resolve("big", HOME, environ={"OPENFOLD3_OB0_OPT_N_GPU": n} if n else {})
        assert r1.line == modes.BIG_RESIDENT and "tp" not in r1.hooks and not any("tp_rowpair" in d for d in r1.hook_dirs)
    rep = stack.activate("fast", dry_run=True, home=HOME, environ={"OPENFOLD3_OB0_OPT_HOME": HOME}, log=False)
    assert rep["n_gpu"] == 1 and " n_gpu=1 sharding=none" in report.activation_line(rep)
