"""--n_gpu P (the memory mode's resource axis, tp.py): refused BY NAME under exact / fast / a line and with fewer than P cards (the
core's sentences, exit 3); the ACTIVE / EXIT / PRED lines carry the core's exact token text (`n_gpu=P sharding=rowpair`, `n_gpu=1
sharding=none`); the rank processes' command; the derived line BIG_TP; nothing installed at n_gpu=1."""
import json
import os
import sys

import pytest

pytest.importorskip("opt_core.mem.ngpu", reason="the pinned core is older than the 0.4.0 producers this file exercises (opt_core.mem.ngpu / rowpair): skipped by name until the re-pin")

from opendde_opt import cli, modes, report, stack, tp
from opendde_opt.tests import _stubs


def test_token_text_is_the_cores():
    assert tp.fields(1) == "n_gpu=1 sharding=none"
    assert tp.fields(2) == "n_gpu=2 sharding=rowpair"
    assert tp.fields(8) == "n_gpu=8 sharding=rowpair"


def test_active_line_carries_the_tokens():
    ln = report.activation_line({"mode": "big", "active": True, "levers_applied": ["rowpair_tp"], "n_gpu": 4})
    assert " n_gpu=4 sharding=rowpair" in ln and ln.startswith(report.PREFIX + " ACTIVE mode=big")
    ln1 = report.activation_line({"mode": "fast", "active": True, "levers_applied": ["arm_u"]})
    assert " n_gpu=1 sharding=none" in ln1
    assert " n_gpu=1 sharding=none" in report.exit_tally_line(pid=1)


@pytest.mark.parametrize("mode", ["off", "exact", "fast", None])           # every mode but big, and no mode at all
def test_n_gpu_above_one_is_refused_outside_big(mode):
    with pytest.raises(tp.TpRefused) as ei:
        tp.check(2, mode, visible=8)
    assert str(ei.value) == "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)"
    assert tp.check(1, mode, visible=0) == 1                       # P=1 passes everywhere, nothing probed


def test_fewer_visible_cards_than_p_is_refused_by_name():
    with pytest.raises(tp.TpRefused) as ei:
        tp.check(4, "big", visible=2)
    assert str(ei.value) == "refused: n_gpu=4 visible=2"
    assert tp.check(4, "big", visible=4) == 4
    with pytest.raises(tp.TpRefused):
        tp.check(0, "big", visible=4)                            # a usage error: positive integers only


@pytest.mark.parametrize("sel", [["--mode", "off"], ["--mode", "exact"], ["--mode", "fast"], []], ids=["off", "exact", "fast", "no-mode"])
def test_cli_refuses_n_gpu_above_one_outside_big_before_anything_runs(tmp_path, capsys, monkeypatch, sel):
    """`pred --n_gpu 2` under off / exact / fast / no --mode (= fast): the core's sentence on the NOT ACTIVE line and exit 3, before any model
    process, stock subprocess or rank launcher starts (the stock caller is stood in and must not be reached)."""
    monkeypatch.setenv("MODEL_OPT", _stubs.TREE); monkeypatch.delenv("OPENDDE_OPT", raising=False)
    reached = []
    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: reached.append(a) or 0)
    monkeypatch.setattr(cli, "_call_relay", lambda *a, **k: (reached.append(a), (0, ""))[1])
    q = tmp_path / "q.json"; q.write_text('[{"name": "x", "sequences": [{"proteinChain": {"sequence": "ACDE", "count": 1}}]}]')
    with pytest.raises(cli.CliError) as ei:
        cli.cmd_pred([*sel, "--n_gpu", "2", "-i", str(q), "-o", str(tmp_path / "o")])
    assert ei.value.code == cli.EXIT_NOT_ACTIVE and reached == []
    out = capsys.readouterr().out
    assert "NOT ACTIVE mode=" in out and "n_gpu=2 reason='refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)'" in out, out


def test_rank_processes_run_this_command_rank0_owning_the_out_dir(tmp_path):
    import argparse
    a = argparse.Namespace(input=str(tmp_path / "q.json"), det="0", root=None, allow_partial=True, sample="5", seeds="101")
    argv_of = cli.rank_argv(a, 4, str(tmp_path / "o"), ["--use_msa", "false"])
    r0, r3 = argv_of(0), argv_of(3)
    assert r0[1:4] == ["-m", "opendde_opt", "pred"] and r0[r0.index("--mode") + 1] == "big" and r0[r0.index("--n_gpu") + 1] == "4"
    assert r0[r0.index("-o") + 1] == str(tmp_path / "o")
    assert r3[r3.index("-o") + 1] == os.path.join(str(tmp_path / "o"), tp.RANK_SUBDIR, "rank3")
    assert "--allow-partial" in r0 and r0[-3:] == ["--", "--use_msa", "false"]
    i = r0.index("--sample"); assert r0[i:i + 4] == ["--sample", "5", "--seeds", "101"] and "--settings" not in r0   # the stated upstream flags reach every rank verbatim


def test_big_tp_is_the_big_line_without_the_offload_unit_plus_rowpair():
    """`BIG_TP` = BIG_F's composition minus the offload unit (every pair track is row-sharded across the ranks: the unit's host-streamed
    stages, diffz, the LayerNorm guard and free_templ are served by the row shard) plus rowpair_tp; same allocator; the kit path without the unit."""
    line = tp.register_line()
    base = modes.LINES["BIG_F"]
    assert modes.LINES[tp.LINE_NAME] is line and tp.LINE_NAME in modes.BIG_LINES and line.tier == "tier2"
    assert tp.LEVER == "rowpair_tp" and line.levers[-2:] == ("rowpair_tp",) + modes.TP_KERNEL_LEVERS   # the row-sharded pair stack + its row-block tri-attention kernel lever close the line
    assert not (set(line.levers) & set(modes._OFFLOAD_LEVERS)) and base.exports["ODDE_OFFLOAD"] == "all"
    assert not any(k.startswith("ODDE_OFFLOAD") for k in line.exports)
    assert line.path_order == modes.KIT_PATH_ORDER and line.allocator == base.allocator
    assert set(line.levers) - {tp.LEVER, modes.TP_STRUCT_BF16_LEVER, *modes.TP_KERNEL_LEVERS} <= set(base.levers)   # the row-sharded line's own levers: rowpair_tp + struct_pair_bf16 + the row-block tri-attention kernel lever
    assert set(line.exports) == (set(base.exports) - {k for k in base.exports if k.startswith("ODDE_OFFLOAD")} - set(modes._OFFLOAD_EXPORTS)) | set(modes.TP_EXPORTS)


def test_rank_environment_detection(monkeypatch):
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    assert not tp.in_rank_process()
    monkeypatch.setenv("ROWPAIR_WORLD", "2"); monkeypatch.setenv("ROWPAIR_RANK", "1")
    assert tp.in_rank_process() and tp.rank() == 1
    assert tp.rank_out_dir("/o", 0) == "/o" and tp.rank_out_dir("/o", 1) == "/o/.rowpair/rank1"


def test_nothing_is_installed_at_n_gpu_1(monkeypatch):
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    monkeypatch.setattr(tp, "STATS", dict(tp.STATS, installed=False, armed=False, n_gpu=1))
    tp.install()
    assert not tp.STATS["installed"] and not tp.STATS["armed"] and "patches" not in tp.STATS
    assert tp.notes([tp.LEVER]) == ["rowpair_tp: n_gpu=1 — the adapter installs nothing; the engine's single-card statements run"]
    assert tp.fallbacks([tp.LEVER]) == []


# ---------------------------------------------------------------------------------------------- the axis never drops silently (fail-closed)
def _query(tmp_path):
    q = tmp_path / "q.json"; q.write_text(json.dumps([{"name": "a", "sequences": [{"proteinChain": {"sequence": "MKT", "count": 1}}]}]))
    return q


def test_requested_p_reaches_the_launcher(monkeypatch, tmp_path):
    """`pred --mode big --n_gpu 2`: the parent hands P=2 to the core's launcher and every rank's argv carries `--n_gpu 2`."""
    monkeypatch.setattr(cli._frozen, "problems", lambda *a, **k: [])                   # the frozen-weights gate is not under test here
    from opt_core.mem.rowpair import launch
    seen = {}
    def fake_run(n, argv, **kw):
        seen["n"] = n; seen["argv1"] = kw["argv_of"](1); seen["isolate"] = kw.get("isolate_devices")
        return [{"rank": r, "rc": 0, "wall_s": 1.0} for r in range(n)]
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    monkeypatch.setattr(launch, "run_rank_processes", fake_run)
    monkeypatch.setattr(tp, "check", lambda n, mode, visible=None: int(n))          # P cards "visible" on this CPU box
    out = tmp_path / "o"
    (out).mkdir(); (out / "opt_manifest.json").write_text(json.dumps({"activation": {"active": True, "n_gpu": 2}}))   # what rank 0 writes
    rc = cli.main(["pred", "--mode", "big", "--n_gpu", "2", "-i", str(_query(tmp_path)), "-o", str(out)])
    assert rc == cli.EXIT_OK and seen["n"] == 2 and seen["isolate"] is True
    assert seen["argv1"][seen["argv1"].index("--n_gpu") + 1] == "2" and seen["argv1"][seen["argv1"].index("-o") + 1].endswith("/.rowpair/rank1")


def test_a_run_that_folded_on_fewer_cards_is_refused_by_name(monkeypatch, tmp_path, capsys):
    """The launcher ran one rank for `--n_gpu 2`, or rank 0 reported n_gpu=1: NOT ACTIVE reason=n_gpu_mismatch, exit 3 — never a pass."""
    monkeypatch.setattr(cli._frozen, "problems", lambda *a, **k: [])                   # the frozen-weights gate is not under test here
    from opt_core.mem.rowpair import launch
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    monkeypatch.setattr(tp, "check", lambda n, mode, visible=None: int(n))
    out = tmp_path / "o"; out.mkdir()
    (out / "opt_manifest.json").write_text(json.dumps({"activation": {"active": True, "n_gpu": 1}}))
    monkeypatch.setattr(launch, "run_rank_processes", lambda n, argv, **kw: [{"rank": r, "rc": 0, "wall_s": 1.0} for r in range(n)])
    assert cli.main(["pred", "--mode", "big", "--n_gpu", "2", "-i", str(_query(tmp_path)), "-o", str(out)]) == cli.EXIT_NOT_ACTIVE
    assert "NOT ACTIVE mode=big reason=n_gpu_mismatch requested=2 active=1 (rank 0's activation report)" in capsys.readouterr().out
    monkeypatch.setattr(launch, "run_rank_processes", lambda n, argv, **kw: [{"rank": 0, "rc": 0, "wall_s": 1.0}])
    assert cli.main(["pred", "--mode", "big", "--n_gpu", "2", "-i", str(_query(tmp_path)), "-o", str(out)]) == cli.EXIT_NOT_ACTIVE
    assert "n_gpu_mismatch requested=2 active=1 (rank processes the launcher ran)" in capsys.readouterr().out


def test_a_rank_started_for_another_world_refuses_by_name(monkeypatch, tmp_path, capsys):
    """Inside a rank process (ROWPAIR_WORLD=4) asked for `--n_gpu 2`: refused before anything activates."""
    monkeypatch.setattr(cli._frozen, "problems", lambda *a, **k: [])                   # the frozen-weights gate is not under test here
    monkeypatch.setenv("ROWPAIR_WORLD", "4"); monkeypatch.setenv("ROWPAIR_RANK", "1")
    rc = cli.main(["pred", "--mode", "big", "--n_gpu", "2", "-i", str(_query(tmp_path)), "-o", str(tmp_path / "o2")])
    assert rc == cli.EXIT_NOT_ACTIVE
    assert "reason=n_gpu_mismatch requested=2 active=4 (the launcher's world of this rank process)" in capsys.readouterr().out
    assert tp.mismatch(2, 2, "x") is None and tp.mismatch(2, 1, "x") == "n_gpu_mismatch requested=2 active=1 (x)"


# ------------------------------------------------------------------------------- `--mode off`: the plain stock caller (no kit-composed multi-GPU launch)
def test_the_stock_caller_is_the_plain_interpreter_and_upstreams_own_flags_pass_verbatim():
    """The stock caller: `python -s -m opendde_opt.stock_pred … -- <upstream argv>` — no launcher, nothing the kit composes about multi-GPU;
    upstream's own options (its Fold-CP switches included) are whatever the caller stated after `--`, handed on unchanged."""
    cmd1, _ = cli.stock_command(_stubs.TREE, ["pred", "-i", "q.json", "-o", "/o"], "/o/stock_env_proof.json")
    assert cmd1[:4] == [sys.executable, "-s", "-m", "opendde_opt.stock_pred"] and "--n-gpu" not in cmd1
    assert "torch.distributed.run" not in cmd1 and cmd1[cmd1.index("--") + 1:] == ["pred", "-i", "q.json", "-o", "/o"]
    theirs = ["pred", "-i", "q.json", "-o", "/o", "--foldcp_mode", "distributed", "--foldcp_size_cp", "2"]        # upstream's flags, stated by the caller after `--`
    cmd2, _ = cli.stock_command(_stubs.TREE, theirs, "/o/stock_env_proof.json")
    assert cmd2[cmd2.index("--") + 1:] == theirs


# ------------------------------------------------------------------------------------------------ `--dtype`: upstream's switch on any mode, run as requested


def test_pred_dtype_bf16_under_a_kit_mode_is_the_base_and_says_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    rc = cli.main(["pred", "--mode", "fast", "--dtype", "bf16", "-i", str(_query(tmp_path)), "-o", str(tmp_path / "o3")])
    out = capsys.readouterr().out
    assert rc != cli.EXIT_USAGE and "NOTE" not in out and "refused: --dtype" not in out


def test_off_stated_dtype_passes_verbatim_and_unstated_flags_are_absent(monkeypatch, tmp_path):
    """`pred --mode off --dtype fp32`: the stock argv carries exactly the stated upstream flag (once) plus the checkpoint path; nothing else is injected."""
    seen = {}
    monkeypatch.setattr(cli._frozen, "problems", lambda *a, **k: [])
    def fake_call(cmd, env=None, stdin=None):
        seen["cmd"] = list(cmd); seen["stdin"] = stdin; return 0
    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    monkeypatch.setattr(stack, "activate", lambda *a, **k: {"active": False, "mode": "off", "reason": "stock"})
    cli.main(["pred", "--mode", "off", "--dtype", "fp32", "-i", str(_query(tmp_path)), "-o", str(tmp_path / "o4")])
    args = seen["cmd"][seen["cmd"].index("--") + 1:]
    assert args[:5] == ["pred", "-i", str(_query(tmp_path)), "-o", str(tmp_path / "o4")]
    assert args[5:7] == ["--dtype", "fp32"] and args.count("--dtype") == 1                       # stated once, verbatim, right after -o
    assert not any(a in args for a in ("--cycle", "--step", "--sample", "--seeds", "--use_msa", "--model_name"))   # unstated: upstream's defaults, nothing passed
    assert "torch.distributed.run" not in seen["cmd"]
    assert seen["stdin"] is cli.subprocess.DEVNULL                                 # the stock child is launched off the caller's stdin (nostdin)
    cli.main(["pred", "--mode", "off", "-i", str(_query(tmp_path)), "-o", str(tmp_path / "o5")])   # nothing stated: the stock base's --dtype bf16 is the one flag passed (then the checkpoint path if any)
    args = seen["cmd"][seen["cmd"].index("--") + 1:]
    assert args[5:7] == ["--dtype", "bf16"] and args.count("--dtype") == 1 and not any(a in args for a in ("--cycle", "--seeds", "--sample"))
    cli.main(["pred", "--mode", "off", "-i", str(_query(tmp_path)), "-o", str(tmp_path / "o6"), "--", "--dtype", "fp32"])   # --dtype after `--`: passed where the caller put it, no base flag added
    args = seen["cmd"][seen["cmd"].index("--") + 1:]
    assert args.count("--dtype") == 1 and args[-2:] == ["--dtype", "fp32"]


# ------------------------------------------------------------------- the run's seeds under P > 1: ONE decision in the launcher's parent (settings.run_seed)
def _jobs(*seeded):
    """A query's jobs, one per argument: `modelSeeds` present where True."""
    return [dict({"name": f"j{i}", "sequences": [{"proteinChain": {"sequence": "MKT", "count": 1}}]}, **({"modelSeeds": [7 + i]} if s else {}))
            for i, s in enumerate(seeded)]


def test_run_seed_is_one_decision_cli_json_drawn_or_refused():
    from opendde_opt import settings
    assert settings.RUNSEED_SOURCES == ("cli", "json", "drawn") and settings.RUNSEED_SPACE == (1, 65536)      # upstream's own random-seed range
    assert settings.run_seed({"seeds": "5,6"}, _jobs(False), 2) == settings.RunSeed("5,6", "cli", [])          # stated: every rank states it itself, nothing added
    assert settings.run_seed({"seeds": "5"}, _jobs(True, False), 2).source == "cli"                            # stated wins over a part-seeded query (upstream: --seeds overrides modelSeeds)
    assert settings.run_seed({}, _jobs(True, True), 2) == settings.RunSeed(None, "json", [])                   # every job seeded: upstream reads them per job on every rank
    assert settings.run_seed({}, _jobs(False, False), 4, draw=lambda: 4242) == settings.RunSeed(4242, "drawn", ["--seeds", "4242"])   # drawn ONCE, stated to every rank
    v = settings.run_seed({}, _jobs(False), 2).value
    assert isinstance(v, int) and settings.RUNSEED_SPACE[0] <= v <= settings.RUNSEED_SPACE[1]
    empty = _jobs(False); empty[0]["modelSeeds"] = []
    assert settings.run_seed({}, empty, 2, draw=lambda: 9) == settings.RunSeed(9, "drawn", ["--seeds", "9"])   # an empty list carries no seeds (upstream's own test)
    with pytest.raises(ValueError) as ei:
        settings.run_seed({}, _jobs(True, False, False), 2)
    assert str(ei.value) == "runseed: 2 of 3 jobs carry no modelSeeds under --n_gpu 2 — state --seeds, or give every job modelSeeds"
    assert settings.run_seed_fields(settings.RunSeed("5,6", "cli", [])) == "runseed=5,6 source=cli"
    assert settings.run_seed_fields(settings.RunSeed(None, "json", [])) == "runseed=- source=json"
    assert settings.run_seed_fields(settings.RunSeed(4242, "drawn", ["--seeds", "4242"])) == "runseed=4242 source=drawn"


def _launch(monkeypatch, tmp_path, capsys, args, jobs):
    """`pred --mode big --n_gpu 2 <args>` on a query of `jobs`, the core's launcher stood in: (rc, {rank: argv} | None when never launched, stdout)."""
    monkeypatch.setattr(cli._frozen, "problems", lambda *a, **k: [])                   # the frozen-weights gate is not under test here
    from opt_core.mem.rowpair import launch
    seen = {}
    def fake_run(n, argv, **kw):
        seen["argv"] = {r: kw["argv_of"](r) for r in range(n)}
        return [{"rank": r, "rc": 0, "wall_s": 1.0} for r in range(n)]
    monkeypatch.delenv("ROWPAIR_WORLD", raising=False)
    monkeypatch.setattr(launch, "run_rank_processes", fake_run)
    monkeypatch.setattr(tp, "check", lambda n, mode, visible=None: int(n))          # P cards "visible" on this CPU box
    q = tmp_path / "jobs.json"; q.write_text(json.dumps(jobs))
    out = tmp_path / "o"
    out.mkdir(exist_ok=True); (out / "opt_manifest.json").write_text(json.dumps({"activation": {"active": True, "n_gpu": 2}}))   # what rank 0 writes
    rc = cli.main(["pred", "--mode", "big", "--n_gpu", "2", "-i", str(q), "-o", str(out), *args])
    return rc, seen.get("argv"), capsys.readouterr().out


def test_a_drawn_run_seed_reaches_every_rank_as_seeds(monkeypatch, tmp_path, capsys):
    """No `--seeds`, no `modelSeeds`: the parent draws ONE seed (upstream's range) and every rank's argv states `--seeds <v>` where a caller's
    own `--seeds` would stand; the LAUNCH line names it (`runseed=<v> source=drawn`) before `ranks=`; the launcher manifest records it."""
    from opendde_opt import settings
    monkeypatch.setattr(settings.random, "randint", lambda lo, hi: 31337)
    rc, argv, out = _launch(monkeypatch, tmp_path, capsys, ["--sample", "1"], _jobs(False))
    assert rc == cli.EXIT_OK, out
    for r in (0, 1):
        av = argv[r]; i = av.index("--seeds")
        assert av[i - 2:i + 3] == ["--sample", "1", "--seeds", "31337", "--det"] and av.count("--seeds") == 1, av   # after the stated flags, once, the same on every rank
    assert " line=BIG_TP runseed=31337 source=drawn ranks=2 data_form=rank0_bcast " in out, out   # + who featurises under P>1 (the core's word)
    assert json.load(open(tmp_path / "o" / tp.RANK_SUBDIR / "opt_manifest.json"))["runseed"] == {"value": 31337, "source": "drawn", "ranks": 2}


@pytest.mark.parametrize("args", [["--seeds", "7"], ["-s", "7"], ["--", "--seeds", "7"], ["--", "-s", "7"]], ids=["seeds", "s", "dashdash-seeds", "dashdash-s"])
def test_stated_seeds_are_the_run_seed_and_nothing_is_added(monkeypatch, tmp_path, capsys, args):
    """`--seeds` / upstream's `-s`, on `pred` or after `--`: source=cli, the ranks carry exactly what the caller stated (once), nothing drawn."""
    from opendde_opt import settings
    monkeypatch.setattr(settings.random, "randint", lambda lo, hi: pytest.fail("a stated --seeds draws nothing"))
    rc, argv, out = _launch(monkeypatch, tmp_path, capsys, args, _jobs(False, False))
    assert rc == cli.EXIT_OK, out
    for r in (0, 1):
        av = argv[r][4:]                                                              # past `python -m opendde_opt pred`
        flags = [t for t in av if t in ("--seeds", "-s")]
        assert len(flags) == 1 and av[av.index(flags[0]) + 1] == "7", av
        if args[0] == "--":
            assert av[av.index("--") + 1:] == args[1:]                                # after `--`: verbatim where the caller put it
    assert " runseed=7 source=cli ranks=2 " in out, out
    assert json.load(open(tmp_path / "o" / tp.RANK_SUBDIR / "opt_manifest.json"))["runseed"] == {"value": "7", "source": "cli", "ranks": 2}


def test_json_model_seeds_are_the_run_seed(monkeypatch, tmp_path, capsys):
    """Every job carries `modelSeeds`: source=json, no `--seeds` reaches any rank (upstream reads the jobs' seeds on every rank alike)."""
    from opendde_opt import settings
    monkeypatch.setattr(settings.random, "randint", lambda lo, hi: pytest.fail("a fully seeded query draws nothing"))
    rc, argv, out = _launch(monkeypatch, tmp_path, capsys, [], _jobs(True, True))
    assert rc == cli.EXIT_OK, out
    assert all("--seeds" not in argv[r] and "-s" not in argv[r][4:] for r in (0, 1)), argv
    assert " runseed=- source=json ranks=2 " in out, out
    assert json.load(open(tmp_path / "o" / tp.RANK_SUBDIR / "opt_manifest.json"))["runseed"] == {"value": None, "source": "json", "ranks": 2}


def test_a_query_seeded_in_part_is_refused_before_any_rank_starts(monkeypatch, tmp_path, capsys):
    rc, argv, out = _launch(monkeypatch, tmp_path, capsys, [], _jobs(True, False))
    assert rc == cli.EXIT_USAGE and argv is None                                      # the launcher was never reached
    assert (f"PRED refused: exit {cli.EXIT_USAGE} reason=runseed: 1 of 2 jobs carry no modelSeeds under --n_gpu 2 — state --seeds, "
            "or give every job modelSeeds") in out, out
    assert " LAUNCH " not in out


# ------------------------------------------------------------------- the rank processes' hash seed: ONE `PYTHONHASHSEED` for the P interpreters (the core's launcher)
def _rank_seeds(monkeypatch, tmp_path, capsys, sub):
    """Launch 2 real rank processes (the core's launcher on CPU, each printing its PYTHONHASHSEED) through tp.run_ranks: their two values and the
    launcher's stderr (its `RANKENV` line)."""
    import re
    from opt_core.mem.rowpair import launch
    real = launch.run_rank_processes
    monkeypatch.setattr(launch, "run_rank_processes", lambda *a, **k: real(*a, **{**k, "cpu_ok": True}))   # the real launcher; only the visible-card gate is lifted
    out = tmp_path / sub
    tp.run_ranks(2, lambda r: [sys.executable, "-c", "import os; print('hashseed', os.environ.get('PYTHONHASHSEED'), flush=True)"], str(out), run_timeout_s=120)
    vals = []
    for r in (0, 1):
        m = re.search(r"^hashseed (\S+)$", (out / tp.RANK_SUBDIR / f"rank{r}.log").read_text(), re.M)
        assert m, (r, (out / tp.RANK_SUBDIR / f"rank{r}.log").read_text())
        vals.append(m.group(1))
    return vals, capsys.readouterr().err


def test_rank_processes_inherit_a_preset_hash_seed(monkeypatch, tmp_path, capsys):
    """A `PYTHONHASHSEED` set in the caller's environment reaches every rank unchanged (source=inherited), named once on the launcher's RANKENV line."""
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    vals, err = _rank_seeds(monkeypatch, tmp_path, capsys, "preset")
    assert vals == ["123", "123"] and "[opendde-opt] RANKENV hashseed=123 source=inherited ranks=2" in err, (vals, err)   # the core launcher's line, under the kit's tag


def test_rank_processes_get_one_hash_seed_when_none_is_set(monkeypatch, tmp_path, capsys):
    """No `PYTHONHASHSEED` in the caller's environment: the launcher decides ONE value for the run and every rank process gets it (source=default)."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    (a, b), err = _rank_seeds(monkeypatch, tmp_path, capsys, "unset")
    assert a == b and a not in ("None", "") and f"[opendde-opt] RANKENV hashseed={a} source=default ranks=2" in err, (a, b, err)
