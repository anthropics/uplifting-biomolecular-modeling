"""THE EXIT RULE (cli.py): a running verb never exits 0 with a degradation only in a status field.

(a) a named fallback (partial activation) -> EXIT_NOT_ACTIVE by name, no opt-out (a mode is all of its levers on the card, never a subset
    under its name), at activation and again after the run once the activation report is reconciled with the kit's end-of-run records
    (the graphed sampler installs late);
(b) a documented gate is not a fallback: recorded under `gates` / `gate_overrides`, rc 0;
(c) outputs short -> the OUTPUTS line's `incomplete` (expected/found/missing), EXIT_FAIL, on both routes;
(d) the env route records the same states (`exit_gate: env-route-records`); the CLI verb gates (`exit_gate: cli`);
(e) check: a partial dry run exits EXIT_NOT_ACTIVE by name, as a dry run that does not activate does;
    the deterministic recipe (--det 1, PTX_DET=1) changes no lever: the graphed sampler stays in the expected set."""
import json
import os
import sys
import types

import pytest

from protenix_opt import cli, det, outputs, report, stack
from protenix_opt.modes import MODES
from protenix_opt.tests import _stock_stub
from protenix_opt.tests.conftest import HAS_DURABLE_INSTALL
from protenix_opt.tests.test_cli_passthrough import CoreStub, fake_stock            # noqa: F401  (fixture: the in-process stock stand-in)
from protenix_opt.tests.test_stock_route import stub_runner                          # noqa: F401  (fixture: the on-disk stock stand-in)

KIT = stack.kit_home()
LATE_REFUSAL = {"requested": True, "installed": False, "det": False, "why": "install failed: no infopt_graphs", "graph": True, "hoist": True}
INSTALLED = {"requested": True, "installed": True, "det": False, "why": None, "graph": True, "hoist": True, "max_tokens": 0,
             "sampler": {"graphed": 5, "stock": 0}}


@pytest.fixture
def partial_core(monkeypatch):
    """The core interface with one lever fallen back (the ACTIVE partial form)."""
    stub = CoreStub(partial=True)
    monkeypatch.setattr(cli, "activate", lambda core, mode, dry_run=False, det_level=None: stub.enable(mode))
    monkeypatch.setattr(cli, "check_mode", lambda core, mode: mode)
    monkeypatch.setattr(report, "register_exit_tally", lambda: stub.events.append("tally") or True)
    monkeypatch.delenv("PTX_LEVER_REPORT", raising=False)
    return stub


@pytest.fixture
def late_record(monkeypatch):
    """Install a stand-in for the kit's fpf_clisampler module whose report() returns the record given."""
    def _install(rec):
        m = types.ModuleType("fpf_clisampler.clisampler")
        m.report = lambda: dict(rec)
        monkeypatch.setitem(sys.modules, "fpf_clisampler.clisampler", m)
        return m
    yield _install


# ------------------------------------------------------------------------------------------------ (a) partial at activation
def test_pred_partial_exits_not_active_by_name(partial_core, fake_stock, tmp_path, capsys):
    """A mode is all of its levers on the card: with one unable to run it refuses by name before the stock CLI runs — no opt-out."""
    out = tmp_path / "o"
    rc = cli.main(["pred", "--mode", "exact", "--input", "x", "--out_dir", str(out)])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_NOT_ACTIVE and fake_stock["main_args"] == [], "refused before the stock CLI runs"
    assert "[protenix-opt] NOT ACTIVE: partial activation refused (at activation) — PWAZ: stub: refused(PWAZ: no cell on this GPU)" in err
    assert "--allow-partial" not in err and "refuses rather than run a subset under its name" in err
    assert not out.exists(), "refused before the stock CLI: nothing written"


def test_allow_partial_flag_is_gone(partial_core, fake_stock, tmp_path, capsys):
    """The retired opt-out is no port option: it changes nothing (the partial mode still refuses) and the names are gone."""
    out = tmp_path / "o"
    rc = cli.main(["pred", "--mode", "exact", "--allow-partial", "--input", "x", "--out_dir", str(out)])
    assert rc == cli.EXIT_NOT_ACTIVE and fake_stock["main_args"] == []
    assert not hasattr(cli, "ALLOW_PARTIAL") and not hasattr(cli, "split_flag") and not hasattr(cli, "note_partial")


# ------------------------------------------------------------------------------------------------ (a) partial after the run: late records
def test_pred_late_fallback_is_reconciled_and_exits_not_active(monkeypatch, fake_stock, late_record, tmp_path, capsys):
    """The kit installs the graphed sampler after the activation report; its end-of-run record says installed False -> the lever
    moves to the fallbacks with the kit's why, the FINAL line says so, and the verb exits 3 after the work (outputs in place)."""
    stub = CoreStub(applied=["sampler_graph", "t1_fused_transition"])
    monkeypatch.setattr(cli, "activate", lambda core, mode, dry_run=False, det_level=None: stub.enable(mode))
    monkeypatch.setattr(cli, "check_mode", lambda core, mode: mode)
    monkeypatch.setattr(report, "register_exit_tally", lambda: True)
    monkeypatch.delenv("PTX_LEVER_REPORT", raising=False)
    late_record(LATE_REFUSAL)
    out = tmp_path / "o"
    rc = cli.main(["pred", "--mode", "exact", "--input", "x", "--out_dir", str(out)])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_NOT_ACTIVE and len(fake_stock["main_args"]) == 1, "the run happened; the exit code says what the records say"
    assert "[protenix-opt] ACTIVE mode=exact" in err and "(levers installed later are reconciled at the end)" in err
    assert ("[protenix-opt] FINAL mode=exact n_gpu=1 sharding=none levers=t1_fused_transition fallbacks=sampler_graph partial=true reconciled_with=clisampler "
            "moves=sampler_graph:applied->fallback") in err
    assert "NOT ACTIVE: partial activation refused (after the run: the kit's end-of-run records) — sampler_graph: clisampler: installed=False — install failed: no infopt_graphs" in err
    assert "[protenix-opt] OUTPUTS complete " in err, "the work happened and its outputs are in place"
    # a record that says installed True: the lever stays applied, plain exit
    late_record(INSTALLED)
    rc = cli.main(["pred", "--mode", "exact", "--input", "x", "--out_dir", str(out)])
    err = capsys.readouterr().err
    assert rc == 0 and "[protenix-opt] FINAL mode=exact n_gpu=1 sharding=none levers=sampler_graph,t1_fused_transition fallbacks=none partial=false reconciled_with=clisampler" in err
    assert " PARTIAL " not in err and "NOT ACTIVE" not in err, "(b) the installed lever's own counters are a documented gate, not a fallback"


def test_reconcile_moves_every_way():
    rep = {"mode": "exact", "active": True, "levers_applied": ["sampler_graph", "sampler_graph_cache_policy", "stackgraph", "t1"], "levers_fallback": [],
           "fallback_reasons": {}}
    r = stack.reconcile(rep, {"clisampler": LATE_REFUSAL, "stackgraph": {"installed": True, "captures": 3, "replays": 9, "refused": 1, "oom_skips": 0}})
    assert r["levers_applied"] == ["stackgraph", "t1"] and r["levers_fallback"] == ["sampler_graph", "sampler_graph_cache_policy"] and r["partial"] is True
    assert r["fallback_reasons"]["sampler_graph"] == "clisampler: installed=False — install failed: no infopt_graphs"
    assert r["reconciled"]["moves"] == {"sampler_graph": "applied -> fallback", "sampler_graph_cache_policy": "applied -> fallback"}
    assert r["gates"] == {"stackgraph": {"captures": 3, "replays": 9, "refused": 1, "oom_skips": 0}}, "stackgraph's memory guard / size gate: recorded, not a fallback"
    assert "excluded_by_det" not in r and "det_exclusion_reasons" not in r, "the deterministic recipe has no lever exclusion"
    # a refusal recorded under the recipe (det True in the kit's record) is a fallback like any other: partial, the kit's reason kept
    det_refusal = dict(LATE_REFUSAL, det=True, why="PTX_DET=1: refused for a reason of the kit's own")
    r = stack.reconcile(rep, {"clisampler": det_refusal})
    assert r["partial"] is True and r["levers_fallback"] == ["sampler_graph", "sampler_graph_cache_policy"]
    assert r["fallback_reasons"]["sampler_graph"] == "clisampler: installed=False — PTX_DET=1: refused for a reason of the kit's own"
    # a fallback lever whose record says installed True: applied
    fb = dict(rep, levers_applied=["t1"], levers_fallback=["sampler_graph", "sampler_graph_cache_policy"],
              fallback_reasons={"sampler_graph": "x", "sampler_graph_cache_policy": "x"})
    r = stack.reconcile(fb, {"clisampler": INSTALLED})
    assert r["levers_applied"] == ["t1", "sampler_graph", "sampler_graph_cache_policy"] and r["levers_fallback"] == [] and r["partial"] is False
    assert r["reconciled"]["moves"] == {"sampler_graph": "fallback -> applied", "sampler_graph_cache_policy": "fallback -> applied"}
    # a record without `installed`, a report() that raised: no move, named
    r = stack.reconcile(rep, {"clisampler": {"report_error": "RuntimeError('x')"}})
    assert r["partial"] is False and r["reconciled"] == {"records": ["clisampler"], "moves": {}, "record_errors": {"clisampler": "RuntimeError('x')"}}


def test_late_records_reads_the_module_then_the_file(late_record, tmp_path, monkeypatch):
    assert stack.late_records(None, None) == {}, "no kit module loaded, no file: no record"
    late_record(LATE_REFUSAL)
    assert stack.late_records(None, None) == {"clisampler": LATE_REFUSAL}
    p = tmp_path / "lr.jsonl"
    p.write_text(json.dumps({"pid": 7, "stackgraph": {"installed": True, "captures": 1}}) + "\n" + json.dumps({"pid": 7, "clisampler": {"installed": True}}) + "\n")
    recs = stack.late_records(str(p), 7)
    assert recs["stackgraph"] == {"installed": True, "captures": 1} and recs["clisampler"] == LATE_REFUSAL, "the live module's report() wins over the file"
    m = late_record({"installed": True})
    m.report = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    assert stack.late_records(None, None) == {"clisampler": {"report_error": "RuntimeError('boom')"}}


# ------------------------------------------------------------------------------------------------ the det recipe changes no lever


def test_det_switch_reads_the_environment():
    assert det.is_det({"PTX_DET": "1"}) and not det.is_det({"PTX_DET": "0"}) and not det.is_det({})
    assert not hasattr(det, "EXCLUDED_LEVERS") and not hasattr(det, "excluded"), "the recipe carries no lever exclusion"


def test_classify_is_the_same_with_and_without_the_recipe():
    with_det = stack._classify("exact", [], {"PTX_DET": "1"})
    without = stack._classify("exact", [], {})
    assert with_det == without, "PTX_DET=1 changes no lever's classification"
    on, fb, skipped, why = with_det
    assert sorted(on + fb + skipped) == sorted(MODES["exact"]), "total accounting"
    assert {"sampler_graph", "sampler_graph_cache_policy"} <= set(on + fb), "the graphed sampler is judged by its markers under the recipe, never dropped"


def test_kit_sampler_lever_has_no_det_refusal():
    """The kit's CLI-path sampler lever installs under PTX_DET=1: no branch of its install() keys on the recipe."""
    src = open(os.path.join(KIT, "src", "fpf_clisampler", "clisampler.py"), encoding="utf-8").read()
    assert "PTX_SAMPLER_GRAPH_DET" not in src and "_DET_OK" not in src


def test_dry_run_under_det_keeps_the_graphed_sampler(monkeypatch):
    """check --det 1: the dry run's expected set is the mode's full set, the graphed sampler included; nothing named excluded."""
    import io
    from contextlib import redirect_stderr
    from protenix_opt.tests.test_supported_gpu import H100
    monkeypatch.setattr(stack, "protenix_version", lambda: "2.0.0")
    monkeypatch.setattr(stack, "gpu_probe_smi", lambda: H100)
    err = io.StringIO()
    with redirect_stderr(err):
        r = stack.activate("exact", dry_run=True, det=True)
    with redirect_stderr(io.StringIO()):
        r0 = stack.activate("exact", dry_run=True, det=False)
    assert r["det"] is True and r0["det"] is False
    assert {"sampler_graph", "sampler_graph_cache_policy"} <= set(r["levers_applied"]), "--det 1 keeps the graphed sampler in the expected set"
    assert r["levers_applied"] == r0["levers_applied"] and r["levers_fallback"] == r0["levers_fallback"], "the recipe changes no lever"
    assert "excluded_by_det" not in r and "excluded_by_det" not in err.getvalue()


# ------------------------------------------------------------------------------------------------ (b) gate overrides
def test_force_is_a_recorded_gate_override(monkeypatch):
    monkeypatch.delenv(stack.ENV_FORCE, raising=False)
    assert stack._base("exact")["gate_overrides"] == {}
    monkeypatch.setenv(stack.ENV_FORCE, "1")
    assert stack._base("exact")["gate_overrides"] == {stack.ENV_FORCE: "1"}


# ------------------------------------------------------------------------------------------------ (c) outputs short -> incomplete
def test_outputs_layout_matches_the_stock_dumper():
    dumper = open(os.path.join(stack.tree_home(), "stock", "src", "runner", "dumper.py"), encoding="utf-8").read()
    assert 'f"{sample_name}_sample_{rank}.cif"' in dumper and outputs.CIF == "{name}_sample_{k}.cif"
    assert 'f"{sample_name}_summary_confidence_sample_{rank}.json"' in dumper and outputs.SUMMARY == "{name}_summary_confidence_sample_{k}.json"
    assert 'os.path.join(dump_dir, "predictions")' in dumper and outputs.PRED_SUBDIR == "predictions"
    assert 'f"seed_{seed}"' in dumper and outputs.SEED_DIR == "seed_{seed}"


def test_outputs_census(tmp_path):
    inp = _stock_stub.input_json(str(tmp_path / "in.json"), names=("a", "b"))
    params = {"input": inp, "out_dir": str(tmp_path / "o"), "seeds": "1,2", "sample": 3}
    exp = outputs.expected(params)
    assert exp["entries"] == ["a", "b"] and exp["seeds"] == ["1", "2"] and exp["samples"] == 3 and len(exp["files"]) == 2 * 2 * 3 * 2 and exp["problems"] == []
    assert exp["files"][0] == os.path.join("a", "seed_1", "predictions", "a_sample_0.cif") and exp["files"][1].endswith("a_summary_confidence_sample_0.json")
    cen = outputs.census(params["out_dir"], params)
    assert cen["status"] == "incomplete" and cen["found"] == 0 and cen["expected"] == 24 and "output directory absent" in cen["problems"][0]
    _stock_stub.write_outputs(params)
    cen = outputs.census(params["out_dir"], params)
    assert cen["status"] == "complete" and cen["found"] == cen["expected"] == 24 and cen["missing"] == []
    os.remove(os.path.join(params["out_dir"], exp["files"][-1]))
    cen = outputs.census(params["out_dir"], params)
    assert cen["status"] == "incomplete" and cen["found"] == 23 and cen["missing"] == [exp["files"][-1]]
    assert outputs.line(cen, "[p]").startswith("[p] OUTPUTS incomplete 23/24 (entries=2 seeds=2 samples=3) missing=[")
    # a dataset directory between out_dir and the entry (the stock's dataset_name) is found too
    os.rename(os.path.join(params["out_dir"], "a"), os.path.join(params["out_dir"], "ds"))
    os.makedirs(os.path.join(params["out_dir"], "ds2")); os.rename(os.path.join(params["out_dir"], "ds"), os.path.join(params["out_dir"], "ds2", "a"))
    assert outputs.census(params["out_dir"], params)["found"] == 23
    # unknown: the input cannot be read, the entries cannot be named, no out_dir
    assert outputs.census(params["out_dir"], dict(params, input=str(tmp_path / "missing.json")))["status"] == "unknown"
    (tmp_path / "bad.json").write_text("[{\"sequences\": []}]")
    cen = outputs.census(params["out_dir"], dict(params, input=str(tmp_path / "bad.json")))
    assert cen["status"] == "unknown" and cen["problems"] == [f"{tmp_path / 'bad.json'}: entry 0 without a name"]
    assert outputs.census(None, params)["problems"] == ["no --out_dir in the stock parameters"]
    assert "unknown (completeness not provable)" in outputs.line(cen, "[p]")
    d = tmp_path / "dir"; d.mkdir(); _stock_stub.input_json(str(d / "1.json"), ("c",)); _stock_stub.input_json(str(d / "2.json"), ("d",))
    assert outputs.expected(dict(params, input=str(d)))["entries"] == ["c", "d"]


@pytest.mark.parametrize("mode", ["exact", "off"])
def test_pred_incomplete_outputs_exit_fail_on_both_routes(mode, monkeypatch, tmp_path, capsys, request):
    """Expected 10 (1 entry x 1 seed x 5 samples, a CIF + a summary each), found 9: the OUTPUTS line says incomplete, EXIT_FAIL — on the
    in-process kit arm (the fake stock) and on the stock route (the on-disk stub in a clean subprocess)."""
    if mode == "off":
        if not HAS_DURABLE_INSTALL:
            pytest.skip("protenix_opt is reachable only via this process's own PYTHONPATH, not a durable site install (pip install -e "
                        "opt); the stock 'clean subprocess' route strips PYTHONPATH and needs the real thing")
        request.getfixturevalue("stub_runner")                                            # the on-disk stock stand-in (test_stock_route)
        monkeypatch.setenv("PYTHONPATH", sys.path[0])
    else:
        request.getfixturevalue("fake_stock")
        stub = CoreStub()
        monkeypatch.setattr(cli, "activate", lambda core, mode, dry_run=False, det_level=None: stub.enable(mode))
        monkeypatch.setattr(cli, "check_mode", lambda core, mode: mode)
        monkeypatch.setattr(report, "register_exit_tally", lambda: True)
    monkeypatch.delenv("PTX_LEVER_REPORT", raising=False)
    monkeypatch.setenv(_stock_stub.SHORT_ENV, "1")                                      # the stub leaves the last expected file unwritten
    out = tmp_path / "o"
    rc = cli.main(["pred", "--mode", mode, "--input", "x", "--out_dir", str(out), "--seeds", "7", "--sample", "5"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_FAIL, err[-800:]
    assert "[protenix-opt] OUTPUTS incomplete 9/10 (entries=1 seeds=1 samples=5) missing=['x/seed_7/predictions/x_summary_confidence_sample_4.json']" in err
    # a stock failure keeps its own exit code; the census is still printed
    monkeypatch.delenv(_stock_stub.SHORT_ENV)
    rc = cli.main(["pred", "--mode", mode, "--input", "FAIL", "--out_dir", str(out / "f")])
    assert rc == 5 and "[protenix-opt] OUTPUTS incomplete " in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ (d) the env route records
def test_env_route_records_the_states_and_names_its_exit_gate(monkeypatch, late_record, tmp_path, capsys):
    monkeypatch.setattr(report, "_EXIT_GATE", {"value": report.EXIT_GATE_ENV})
    monkeypatch.setattr(stack, "_REPORT", {"mode": "exact", "active": True, "levers_applied": ["sampler_graph", "t1"], "levers_fallback": [],
                                            "package_version": "0", "protenix_version": "2.0.0", "gpu": {"name": "X", "sm": "sm90"}})
    late_record(LATE_REFUSAL)
    rec = report._activation_record()
    assert rec["exit_gate"] == "env-route-records" and rec["partial"] is True and rec["levers_fallback"] == ["sampler_graph"] and rec["levers_applied"] == ["t1"]
    assert rec["reconciled"]["moves"] == {"sampler_graph": "applied -> fallback"} and "excluded_by_det" not in rec
    p = tmp_path / "lr.jsonl"
    monkeypatch.setenv("PTX_LEVER_REPORT", str(p))
    monkeypatch.setattr(report, "_memory_stats", lambda: None)
    line = report.exit_line()
    assert line.startswith("[protenix-opt] FINAL mode=exact n_gpu=1 sharding=none levers=t1 fallbacks=sampler_graph partial=true reconciled_with=clisampler moves=sampler_graph:applied->fallback exit_gate=env-route-records\n")
    assert "[protenix-opt] EXIT pid=" in line
    written = [json.loads(l) for l in p.read_text().splitlines()]
    assert written[-1]["protenix_opt"]["exit_gate"] == "env-route-records" and written[-1]["protenix_opt"]["partial"] is True
    report.set_exit_gate(cli.EXIT_GATE_CLI)
    assert report._activation_record()["exit_gate"] == "cli"
    assert not report.exit_line().startswith("[protenix-opt] FINAL"), "the CLI verb prints its own FINAL line"


# ------------------------------------------------------------------------------------------------ (e) check
def _dry(mode, partial):
    rep = {"active": False, "dry_run": True, "mode": mode, "env": {"PTX_BLK": "2"}, "levers_applied": ["a"], "levers_fallback": ["b"] if partial else [],
           "partial": partial, "fallback_reasons": {"b": "the kit's reason"} if partial else {}, "reason": "dry run: resolved and gated, nothing applied"}
    report.log_activation(rep); rep["logged"] = True
    return rep


@pytest.mark.parametrize("verb", ["check"])
def test_check_partial_dry_run(verb, monkeypatch, capsys):
    monkeypatch.setattr(stack, "activate", lambda mode, strict=False, trigger=None, dry_run=False, det=None: _dry(mode, partial=True))
    monkeypatch.setattr(cli, "check_console_script", lambda core: {"status": "ok", "ok": True})
    rc = cli.main([verb, "--mode", "exact"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_NOT_ACTIVE
    assert "[protenix-opt] NOT ACTIVE: partial activation refused (dry run) — b: the kit's reason" in err and "--allow-partial" not in err
    with pytest.raises(SystemExit):                       # the retired opt-out is no check option (argparse rejects it)
        cli.main(["check", "--mode", "exact", "--allow-partial"])
    capsys.readouterr()
    monkeypatch.setattr(stack, "activate", lambda mode, strict=False, trigger=None, dry_run=False, det=None: _dry(mode, partial=False))
    assert cli.main([verb, "--mode", "exact"]) == 0
    monkeypatch.setattr(stack, "activate", lambda mode, strict=False, trigger=None, dry_run=False, det=None:
                        (lambda r: (report.log_activation(r), r)[1])({"active": False, "dry_run": True, "mode": mode, "reason": "no CUDA device visible"}))
    assert cli.main([verb, "--mode", "exact"]) == cli.EXIT_NOT_ACTIVE, "a dry run that would not activate: the same code"


