"""The `--n_gpu P` axis (ngpu.py over opt_core.mem.ngpu): absent == 1; the exact ACTIVE/EXIT token text; every refusal by name with exit 3;
usage errors exit 2. CPU only: the verbs are driven up to the refusal (nothing past it needs a GPU)."""
import json
import os

import pytest

from opt_core.mem import ngpu as NG

from protenix_v1_opt import cli, ngpu
from protenix_v1_opt import report as R


def test_absent_and_explicit_one_are_the_same_P(monkeypatch):
    monkeypatch.delenv(ngpu.ENV_N_GPU, raising=False)
    assert ngpu.effective(None) == 1 and ngpu.effective("1") == 1 and ngpu.effective(1) == 1
    monkeypatch.setenv(ngpu.ENV_N_GPU, "2")
    assert ngpu.effective(None) == 2 and ngpu.effective("1") == 1          # the command line wins over the environment


@pytest.mark.parametrize("bad", ["0", "abc", ""])          # one representative per branch of ngpu.effective/check_n_gpu: post-parse
def test_non_positive_or_non_integer_is_a_usage_error(bad, monkeypatch):    # range check ("0"; "-1" is the same branch), parse failure
    monkeypatch.delenv(ngpu.ENV_N_GPU, raising=False)                       # ("abc"; "1.5" is the same branch), and effective()'s own
    if bad == "":                                                          # empty-counts-as-absent case ("") — R5: trimmed from 5 to 3
        assert ngpu.effective(bad) == 1                                     # empty == absent
        return
    with pytest.raises(ValueError):
        ngpu.effective(bad)


def test_token_text_is_exact():
    """Downstream log parsers grep these exact tokens."""
    assert ngpu.fields(1) == "n_gpu=1 sharding=none"
    assert ngpu.fields(2) == "n_gpu=2 sharding=rowpair"                    # P=4 dropped (R5): same P>1 branch as P=2, no added coverage
    assert R.ngpu_fields({"n_gpu": 1}) == "n_gpu=1 sharding=none" and R.ngpu_fields({}) == "n_gpu=1 sharding=none"
    assert R.exit_line("big", 0, 2) == f"{R.PREFIX} EXIT mode=big rc=0 n_gpu=2 sharding=rowpair"
    assert R.exit_line("fast", 3, 1) == f"{R.PREFIX} EXIT mode=fast rc=3 n_gpu=1 sharding=none"


def test_active_line_carries_the_tokens():
    line = R.activation_line({"mode": "big", "arm": "stock", "cfg": {}, "n_gpu": 1})
    assert " n_gpu=1 sharding=none " in line
    line = R.activation_line({"mode": "big", "arm": "stock", "cfg": {}, "n_gpu": 2})
    assert " n_gpu=2 sharding=rowpair " in line


@pytest.mark.parametrize("mode", ["fast", "off"])          # "exact" dropped (R5): refuse_unless_big has no per-mode branching, so
def test_P_above_one_is_refused_by_name_outside_big(mode):               # "exact"/"fast" are identical code paths; "off" is kept
    with pytest.raises(NG.NGpuRefused) as ei:                               # separately since stack.activate special-cases mode=="off"
        ngpu.admit(2, mode, visible=8)                                     # nearby (a future refactor could accidentally exempt it)
    assert str(ei.value.reason) == NG.REFUSE_MODE == "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)"


def test_fewer_visible_devices_than_P_is_refused_by_name():
    with pytest.raises(NG.NGpuRefused) as ei:
        ngpu.admit(2, "big", visible=1)
    assert str(ei.value.reason) == "refused: n_gpu=2 visible=1"
    with pytest.raises(NG.NGpuRefused) as ei:
        ngpu.admit(4, "big", visible=0)
    assert str(ei.value.reason) == "refused: n_gpu=4 visible=0"


def test_one_is_admitted_under_every_mode():
    for mode in ("exact", "fast", "big", "off"):
        assert ngpu.admit(1, mode, visible=0) == 1                          # P=1 owns no visible-device rule here (the engine's GPU gate does)


def test_the_served_P_set_is_named():
    assert 1 in ngpu.SUPPORTED
    for p in (2, 4, 8):
        if p in ngpu.SUPPORTED:
            assert ngpu.admit(p, "big", visible=p) == p
        else:
            with pytest.raises(NG.NGpuRefused) as ei:
                ngpu.admit(p, "big", visible=p)
            assert str(ei.value.reason).startswith(f"refused: n_gpu={p} not served")


def test_pred_refuses_P2_under_fast_before_any_weight_loads(monkeypatch, capsys, tmp_path):
    called = {"weights": 0}
    from protenix_v1_opt import kit as K
    monkeypatch.setattr(K, "frozen_weights_check", lambda *a, **k: called.__setitem__("weights", called["weights"] + 1))
    rc = cli.main(["pred", "--mode", "fast", "--n_gpu", "2", "--input", "x.json", "--out_dir", str(tmp_path)])
    assert rc == R.EXIT_NOT_ACTIVE == 3
    assert called["weights"] == 0
    err = capsys.readouterr().err.splitlines()
    assert f"{R.PREFIX} NOT ACTIVE: {NG.REFUSE_MODE}" in err
    assert f"{R.PREFIX} EXIT mode=fast rc=3 n_gpu=2 sharding=rowpair" in err
    assert list(tmp_path.iterdir()) == []                                # a refused P writes nothing


@pytest.mark.parametrize("mode_argv", [[], ["--mode", "exact"], ["--mode", "off"]])
def test_pred_refuses_P2_outside_big_including_the_default_mode(monkeypatch, capsys, tmp_path, mode_argv):
    """`--n_gpu 2` with --mode exact / --mode off / NO --mode (the default, fast): refused by name before anything runs — one line
    `NOT ACTIVE: refused: n_gpu>1 requires --mode big …`, exit 3, no weight check (the shared core's rule, opt_core.mem.ngpu.refuse_unless_big)."""
    called = {"weights": 0}
    from protenix_v1_opt import kit as K
    monkeypatch.delenv("PROTENIX_V1_OPT", raising=False)
    monkeypatch.setattr(K, "frozen_weights_check", lambda *a, **k: called.__setitem__("weights", called["weights"] + 1))
    rc = cli.main(["pred", *mode_argv, "--n_gpu", "2", "--input", "x.json", "--out_dir", str(tmp_path)])
    assert rc == R.EXIT_NOT_ACTIVE == 3 and called["weights"] == 0
    err = capsys.readouterr().err
    assert f"{R.PREFIX} NOT ACTIVE: {NG.REFUSE_MODE}" in err and "n_gpu>1 requires --mode big" in NG.REFUSE_MODE
    mode = mode_argv[1] if mode_argv else "fast"
    assert f"{R.PREFIX} EXIT mode={mode} rc=3 n_gpu=2 sharding=rowpair" in err



def test_pred_refuses_P2_under_big_without_devices(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ngpu, "visible_devices", lambda: 0)
    rc = cli.main(["pred", "--mode", "big", "--n_gpu", "2", "--input", "x.json", "--out_dir", str(tmp_path)])
    assert rc == 3
    err = capsys.readouterr().err
    assert f"{R.PREFIX} NOT ACTIVE: refused: n_gpu=2 visible=0" in err


def test_pred_n_gpu_zero_is_usage(capsys, tmp_path):
    rc = cli.main(["pred", "--mode", "big", "--n_gpu", "0", "--input", "x.json", "--out_dir", str(tmp_path)])
    assert rc == R.EXIT_USAGE == 2


def test_check_reports_the_axis_and_refuses_by_name(monkeypatch, capsys):
    monkeypatch.setattr(ngpu, "visible_devices", lambda: 1)
    rc = cli.main(["check", "--mode", "big", "--n_gpu", "2"])
    out = capsys.readouterr()
    rep = json.loads(out.out)
    assert rep["n_gpu"] == 2 and rep["sharding"] == "rowpair" and rep["ok"] is False
    assert rep["gates"][0] == "refused: n_gpu=2 visible=1"
    assert rc == 3 and f"{R.PREFIX} EXIT mode=big rc=3 n_gpu=2 sharding=rowpair" in out.err
    rc = cli.main(["check", "--mode", "exact", "--n_gpu", "2"])
    assert rc == 3 and NG.REFUSE_MODE in capsys.readouterr().err


def test_activate_reads_the_environment_spelling(monkeypatch, capsys):
    from protenix_v1_opt import stack
    monkeypatch.setattr(stack, "_REPORT", None)
    monkeypatch.setenv(ngpu.ENV_N_GPU, "2")
    rep = stack.activate("fast", strict=False, trigger="runner")
    assert rep["active"] is False and rep["reason"] == NG.REFUSE_MODE and rep["n_gpu"] == 2 and rep["sharding"] == "rowpair"
    assert f"{R.PREFIX} NOT ACTIVE: {NG.REFUSE_MODE}" in capsys.readouterr().err


def test_env_name_is_declared_on_both_routes():
    from protenix_v1_opt import _autoload as A
    from protenix_v1_opt import stack
    assert ngpu.ENV_N_GPU in A.DECLARED and ngpu.ENV_N_GPU in stack.PACKAGE_ENV


def _stub_parent(monkeypatch, tmp_path, seen, active_p, n_logs=None):
    """The launching process of `pred --n_gpu 2`: weights check off, two devices visible, the launcher replaced by a recorder that writes
    the ranks' transcripts with an ACTIVE line reporting `active_p`."""
    from protenix_v1_opt import kit as K
    from protenix_v1_opt import rowpair
    monkeypatch.setattr(K, "frozen_weights_check", lambda *a, **k: {})
    monkeypatch.setattr(ngpu, "visible_devices", lambda: 8)
    for name in ngpu.regime_names():                                                    # the launch WRITES the regime into os.environ: every name restored / removed at teardown
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name)
    monkeypatch.delenv(rowpair.L.ENV_RANK, raising=False) if hasattr(rowpair.L, "ENV_RANK") else None

    def fake_run_ranks(n_gpu, argv, out_dir, on_line=None):
        from protenix_v1_opt import big as B
        seen.update(n_gpu=n_gpu, argv=list(argv), environ={k: os.environ.get(k) for k in ngpu.regime_names()})
        n = n_gpu if n_logs is None else n_logs
        os.makedirs(os.path.join(out_dir, "rowpair"), exist_ok=True)
        for r in range(n):
            with open(rowpair.rank_log(out_dir, r), "w") as fh:
                fh.write(f"{R.PREFIX} ACTIVE mode=big arm=x levers=- det=0 partial=- {ngpu.fields(active_p)} gpu='NVIDIA H100 80GB HBM3' sm=9.0\n")
        return [{"rank": r, "rc": 0} for r in range(n)]
    monkeypatch.setattr(rowpair, "run_ranks", fake_run_ranks)


def test_the_requested_P_reaches_the_launcher_and_the_ranks(monkeypatch, tmp_path, capsys):
    import os as _os
    seen = {}
    _stub_parent(monkeypatch, tmp_path, seen, active_p=2)
    rc = cli.main(["pred", "--mode", "big", "--n_gpu", "2", "--input", "x.json", "--out_dir", str(tmp_path)])
    assert rc == 0, capsys.readouterr().err[-600:]
    assert seen["n_gpu"] == 2
    i = seen["argv"].index("--n_gpu"); assert seen["argv"][i + 1] == "2"       # every rank runs `pred … --n_gpu 2 …` (rowpair.rank_argv keeps the axis)
    assert not [k for k in seen["environ"] if k.startswith("PROTENIX_V1_BIG_")], seen["environ"]                    # no lever switch travels through the environment: the superseded levers are off by each rank's in-process selection
    assert all(seen["environ"][k] == v for k, v in ngpu.TP_EXPORTS.items()), seen["environ"]                          # the ranks inherit the ×P line's exports (the core's ROWPAIR_* names) from the launching process
    assert f"{R.PREFIX} EXIT mode=big rc=0 n_gpu=2 sharding=rowpair" in capsys.readouterr().err


def test_a_rank_reporting_fewer_devices_is_refused_by_name_never_a_pass(monkeypatch, tmp_path, capsys):
    seen = {}
    _stub_parent(monkeypatch, tmp_path, seen, active_p=1)                       # the ranks folded on ONE device although 2 were asked
    rc = cli.main(["pred", "--mode", "big", "--n_gpu", "2", "--input", "x.json", "--out_dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 3 and f"{R.PREFIX} NOT ACTIVE: reason=n_gpu_mismatch requested=2 active=rank0:1" in err, err[-600:]


def test_fewer_rank_transcripts_than_P_is_refused_by_name(monkeypatch, tmp_path, capsys):
    seen = {}
    _stub_parent(monkeypatch, tmp_path, seen, active_p=2, n_logs=1)
    rc = cli.main(["pred", "--mode", "big", "--n_gpu", "2", "--input", "x.json", "--out_dir", str(tmp_path)])
    assert rc == 3 and "reason=n_gpu_mismatch requested=2 active=1_ranks" in capsys.readouterr().err


def test_mismatch_words():
    assert ngpu.mismatch_reason(2, 1) == "reason=n_gpu_mismatch requested=2 active=1"


def test_n_gpu_above_one_states_the_lines_large_input_regime(monkeypatch):
    """P>1 exports the row-sharded pair stack's host-parking levers (TP_EXPORTS, the core's ROWPAIR_* names) into every rank's environment —
    a caller's value is recorded as the caller's — and turns the four single-GPU levers whose sites the sharded path replaces OFF by name
    through the line's in-process selection (big.rowpair_switches), never through the environment."""
    from protenix_v1_opt import big as B
    assert set(ngpu.SUPERSEDED_LEVERS) == {"relp_lean", "recycle_carry", "diffusion_cond_chunk", "conf_head_chunk"} and all(l in B.LINE for l in ngpu.SUPERSEDED_LEVERS)
    assert B.rowpair_switches(2) == {l: False for l in ngpu.SUPERSEDED_LEVERS} and B.rowpair_switches(8) == B.rowpair_switches(2) and B.rowpair_switches(1) == {}
    env = {}
    out = ngpu.large_input_regime(env)
    assert env == dict(ngpu.TP_EXPORTS) and all(out[k] == (v, "n_gpu>1:tp_exports") for k, v in ngpu.TP_EXPORTS.items())
    assert set(env) == set(ngpu.regime_names()) == set(out) and not [k for k in env if k.startswith("PROTENIX_V1_BIG_")]   # regime_names() is exactly what the regime writes
    env = {"ROWPAIR_PARK_ZINIT": "0"}
    assert ngpu.large_input_regime(env)["ROWPAIR_PARK_ZINIT"] == ("0", "caller")


def test_n_gpu_above_one_exports_the_row_sharded_lines_host_parks(monkeypatch):
    """The ×P line's exports (ngpu.TP_EXPORTS) are the core's own `ROWPAIR_*` names at the line's values: `large_input_regime`
    states each in the environment the rank processes inherit (source `n_gpu>1:tp_exports`) unless the caller set the name — a caller's
    value, `0` included, wins and is recorded as the caller's (the core reads the name: opt_core.mem.rowpair.trunk.ShardPark reads
    ROWPAIR_PARK_ZINIT when the adapter passes park=None, which tp.get_pairformer_output_tp does)."""
    torch = pytest.importorskip("torch")
    import inspect
    from opt_core.mem.rowpair import trunk as TK, heads as HD, msa_host as MH, pairstack as PS
    assert ngpu.TP_EXPORTS == {"ROWPAIR_PARK_ZINIT": "recompute", "ROWPAIR_MSA_HOST": "rank0", "ROWPAIR_CONF_PARK_ZTRUNK": "1", "ROWPAIR_FREE_ZTRUNK": "1",
                               "ROWPAIR_TRANSPOSE_INPLACE": "1", "ROWPAIR_RANK_THREADS": "auto", "ROWPAIR_HOST_SLAB": "lease",
                               "ROWPAIR_TRIATT_STAGE": "once"}
    monkeypatch.setenv("ROWPAIR_PARK_ZINIT", "recompute")                                     # the line's value: no host copy, the init statements re-run per recycle row block
    assert TK.zinit_park_mode() == "recompute" if hasattr(TK, "zinit_park_mode") else True
    # every exported name is one the core READS, at the adapter's call form (park=None / free=None / host_mode()):
    for value, on in (("1", True), ("0", False)):
        monkeypatch.setenv("ROWPAIR_PARK_ZINIT", value)                                       # tp.get_pairformer_output_tp -> run_trunk_sharded -> ShardPark(park=None)
        assert TK.ShardPark(torch.zeros(1, 4, 4, 8), park=None, force_host=True).park is on, value
        monkeypatch.setenv("ROWPAIR_CONF_PARK_ZTRUNK", value); monkeypatch.setenv("ROWPAIR_FREE_ZTRUNK", value)   # tp.conf_plan -> ZTrunkPlan(free=None, park=None)
        plan = HD.ZTrunkPlan(torch.zeros(4, 4, 8), passes=1)
        assert (plan.park, plan.free) == (on, on), value
        monkeypatch.setenv("ROWPAIR_TRANSPOSE_INPLACE", value)                                # tp.pair_stack_tp -> pairstack.pair_stack_ -> pair_block_(transpose_inplace=None)
        assert PS.env_flag("TRANSPOSE_INPLACE", False) is on and "env_flag(\"TRANSPOSE_INPLACE\"" in inspect.getsource(PS.pair_block_)
    monkeypatch.setenv("ROWPAIR_MSA_HOST", "rank0"); assert MH.host_mode() == "rank0"       # tp.msa_host_hold / _msa_host_place / msa_module_tp read host_mode()
    monkeypatch.setenv("ROWPAIR_MSA_HOST", "0"); assert MH.host_mode() is None
    for name in ngpu.TP_EXPORTS:
        monkeypatch.delenv(name, raising=False)                                             # the words this test did not set above (RANK_THREADS, HOST_SLAB, TRIATT_STAGE) may be absent
    env = {}
    out = ngpu.large_input_regime(env)
    for name, value in ngpu.TP_EXPORTS.items():
        assert env[name] == value and out[name] == (value, "n_gpu>1:tp_exports"), name
    env = {"ROWPAIR_PARK_ZINIT": "0"}                                                         # the caller's 0 wins, by name
    out = ngpu.large_input_regime(env)
    assert env["ROWPAIR_PARK_ZINIT"] == "0" and out["ROWPAIR_PARK_ZINIT"] == ("0", "caller")
    env = {"ROWPAIR_PARK_ZINIT": " "}                                                         # a blank value is unset: the line's value is stated
    out = ngpu.large_input_regime(env)
    assert env["ROWPAIR_PARK_ZINIT"] == "recompute" and out["ROWPAIR_PARK_ZINIT"] == ("recompute", "n_gpu>1:tp_exports")
    # the launching process states the exports before the ranks start (cli._run_ranks) and every rank states them again at activation (stack.activate)
    import inspect
    from protenix_v1_opt import cli, stack
    assert "large_input_regime(os.environ)" in inspect.getsource(cli._run_ranks) and "large_input_regime(os.environ)" in inspect.getsource(stack.activate)


def test_rowpair_line_replaces_the_base_arm_kernel_levers_by_name(monkeypatch):
    """Under `--n_gpu P>1` (the core launcher's ROWPAIR_WORLD in every rank process) the base arm's fast / gflash / ttr levers own no
    site under the row-sharded statements: gates_for names them `replaced_by_rowpair` in off_by_property (report.verdict excuses a lever that
    served 0 calls only BY NAME — never a blanket exemption, never partial + exit 3); at P == 1 nothing is named."""
    from protenix_v1_opt import big as B
    from opt_core.mem.rowpair import launch as L
    assert L.ENV_WORLD == "ROWPAIR_WORLD"
    g1 = B.gates_for(992, {})
    assert g1["rowpair_world"] == 1 and not any(v.startswith("replaced_by_rowpair") for v in g1["off_by_property"].values())
    g2 = B.gates_for(992, {L.ENV_WORLD: "2"})
    assert g2["rowpair_world"] == 2
    for lv in ("fast", "gflash", "ttr"):
        assert g2["off_by_property"][lv].startswith("replaced_by_rowpair at n_gpu 2"), (lv, g2["off_by_property"].get(lv))


def test_replicated_sync_policy_follows_the_determinism_level(monkeypatch):
    """det 1 -> guard (strict bitwise; a mismatch is a defect), det 0 -> bcast (rank 0 authoritative at every sync point); an explicit
    ROWPAIR_DIFF_NOISE_SYNC is the named opt-in over either default; any other value is refused by name."""
    from protenix_v1_opt import rowpair as RP
    assert RP.sync_policy(True, {}) == {"mode": "guard", "det": True, "source": "det"}
    assert RP.sync_policy(False, {}) == {"mode": "bcast", "det": False, "source": "det"}
    assert RP.sync_policy(False, {RP.SYNC_ENV: "guard"})["mode"] == "guard" and RP.sync_policy(True, {RP.SYNC_ENV: "bcast"})["mode"] == "bcast"
    with pytest.raises(RP.D.RowpairRefused):
        RP.sync_policy(False, {RP.SYNC_ENV: "off"})
