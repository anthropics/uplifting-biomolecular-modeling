"""The design verb's run accounting on the stub stack (no GPU, no weights): the census of upstream's resume rule (`skipped_existing`, NOTHING
RAN, the manifest left untouched), the exit rule (`opt_core.report.verdict`: the manifest's exit_code is the code returned), the exit census
(hook never ran / designs written while no sampling call went through the levers → NOT ACTIVE 3), and the PXDESIGN_OPT route's exit verdict."""
import json
import os

import pytest

from pxdesign_opt import cli, manifest, outputs, report, stack as _stack_mod

from . import _stubs


def _box(monkeypatch, stack, tmp_path):
    _stubs.install_torch()
    _stubs.install_upstream()
    _stubs.fake_box(monkeypatch, stack)
    return _stubs.stage_weights(tmp_path, monkeypatch, stack.read_pins())


def _tasks(tmp_path, names=("t5o45", "t1tnf")):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps([{"name": n} for n in names]))
    return p


def _fake_run(out_dir, tasks, n_sample, *, build_runner=True, prepare=True, short=0):
    """Stand-in for infer_loop.run: upstream's InferenceRunner is built (the hook fires: the REAL kit install() on the stub model), then every
    (task, seed) pair NOT already dumped (upstream's resume rule: SUCCESS_FILE present → "already dumped", skipped) gets N_sample design CIFs and
    its SUCCESS_FILE, and the hoist's prepare counter moves once per sampled pair (as one sample_diffusion call through the levers does)."""
    def run(argv, seeds, det, log=print, on_runner=None):
        if build_runner:
            from pxdesign.runner.inference import InferenceRunner
            InferenceRunner(configs=None)
        import pxd_xattempt.hoist as H
        for t in tasks:
            for s in seeds:
                d = os.path.join(out_dir, t, f"seed_{int(s)}")
                if os.path.exists(os.path.join(d, outputs.SUCCESS_FILE)):
                    continue                                                   # upstream: "Skip sample=…: already dumped."
                os.makedirs(os.path.join(d, outputs.PREDICTIONS_DIR), exist_ok=True)
                for i in range(n_sample - short):
                    open(os.path.join(d, outputs.PREDICTIONS_DIR, f"{t}_sample_{i}.cif"), "w").close()
                open(os.path.join(d, outputs.SUCCESS_FILE), "w").close()
                if prepare:
                    H._STATE["stats"]["prepares"] += 1
        return {"load_s": 0.5, "per_seed_s": {str(s): 1.0 for s in seeds}, "n_items": len(tasks), "seeds": [int(s) for s in seeds], "det": int(det),
                "seeding": [], "dump_dir": out_dir, "N_sample": n_sample, "batch": {}, "layernorm": {}, "kernels": {}}
    return run


def _design(monkeypatch, tmp_path, out, ckpt, fake, mode="exact", n_sample=2, seeds="101"):
    from pxdesign_opt import infer_loop
    monkeypatch.setattr(infer_loop, "run", fake)
    return cli.main(["design", "--mode", mode, "--tasks", str(_tasks(tmp_path)), "--out_dir", str(out), "--seeds", seeds, "--N_sample", str(n_sample),
                     "--load_checkpoint_dir", str(ckpt)])


def test_resume_census_counts_upstreams_rule(tmp_path):
    """outputs.count_designs against the pre-run census: skipped_existing / skipped_pairs / produced / nothing_ran (scope job), never nothing_ran on scope dir."""
    out = tmp_path / "out"
    for t, s in (("a", 1), ("b", 1)):
        d = out / t / f"seed_{s}" / outputs.PREDICTIONS_DIR; d.mkdir(parents=True)
        (d / f"{t}_sample_0.cif").write_text("")
    (out / "a" / "seed_1" / outputs.SUCCESS_FILE).write_text("")                  # pair a/seed_1 was dumped by an earlier run; b/seed_1 by this one
    existed = outputs.dumped_pairs(str(out))
    assert existed == ["a/seed_1"]
    c = outputs.count_designs(str(out), ["a", "b"], [1], 1, existed=existed)
    assert (c["n_pairs"], c["skipped_existing"], c["skipped_pairs"], c["produced"], c["nothing_ran"], c["complete"]) == (2, 1, ["a/seed_1"], 1, False, True)
    c = outputs.count_designs(str(out), ["a"], [1], 1, existed=existed)
    assert c["nothing_ran"] is True and c["produced"] == 0 and c["skipped_existing"] == 1
    c = outputs.count_designs(str(out), [], None, 1, n_tasks=2, existed=existed)    # scope dir: the job's pairs unknown — no NOTHING RAN claim
    assert c["scope"] == "dir" and c["nothing_ran"] is False and c["skipped_existing"] == 1 and c["produced"] == 1
    assert outputs.count_designs(str(out), ["a"], [1], 1)["skipped_existing"] == 0   # no census given: nothing counted as skipped


def test_run_lines_name_the_census():
    res = {"mode": "fast", "n_designs": 10, "n_tasks": 2, "n_seeds": 1, "expected": 10, "scope": "job", "skipped_existing": 1, "out_dir": "/x", "wall_s": 3.0}
    line = report.done_line(res)
    assert line.startswith("[pxdesign-opt] DONE mode=fast designs=10 tasks=2 seeds=1 skipped_existing=1") and " s/design=" in line and report.DONE_RE.match(line)
    assert "skipped_existing=n/a" in report.done_line({k: v for k, v in res.items() if k != "skipped_existing"})
    nr = report.nothing_ran_line(dict(res, n_pairs=2, skipped_existing=2, complete=True, nothing_ran=True))
    assert nr.startswith("[pxdesign-opt] NOTHING RAN: 2/2 pairs already dumped (upstream resume)") and "no mode claim made (requested=fast)" in nr
    assert "opt_manifest.json not written" in nr and "mode=fast" not in nr and not report.DONE_RE.match(nr)
    assert report.nothing_ran_line(dict(res, n_pairs=2, skipped_existing=2, complete=False)).endswith(f"designs on disk != tasks x seeds x N_sample: exit {report.EXIT_FAIL}")
    assert "opt_manifest.json not written; stock_env_proof.json written by" in report.nothing_ran_line(dict(res, n_pairs=2, skipped_existing=2, complete=True, nothing_ran_notes=["stock_env_proof.json written by this invocation's stock child"]))


def test_design_writes_the_manifest_after_the_verdict_and_names_skips(fresh_stack, monkeypatch, capsys, tmp_path):
    """A complete run: DONE skipped_existing=0, manifest exit_code 0 == rc. The same job again: upstream skips every pair → NOTHING RAN, no DONE
    line, the manifest left byte for byte as the first run wrote it, rc 0. A third seed added: DONE skipped_existing=2, the manifest names the
    skipped pairs and counts what this run produced."""
    stack = fresh_stack
    ckpt = _box(monkeypatch, stack, tmp_path)
    out = tmp_path / "out"
    rc = _design(monkeypatch, tmp_path, out, ckpt, _fake_run(str(out), ["t5o45", "t1tnf"], 2))
    err = capsys.readouterr().err
    assert rc == 0, err
    assert "[pxdesign-opt] APPLIED model#1 levers=h1,h2,h3,h4,h5 fallbacks=none" in err
    assert "[pxdesign-opt] DONE mode=exact designs=4 tasks=2 seeds=1 skipped_existing=0 " in err and "NOTHING RAN" not in err
    man = manifest.read(str(out))
    assert man["exit_code"] == 0 and man["mode"] == "exact" and man["outputs"]["produced"] == 4 and man["outputs"]["skipped_pairs"] == []
    first = (out / outputs.MANIFEST_NAME).read_bytes()
    # the same job on the same -o: every pair already dumped
    stack.reset_for_tests(); _stubs.uninstall(); _rebox(monkeypatch, stack)
    rc = _design(monkeypatch, tmp_path, out, ckpt, _fake_run(str(out), ["t5o45", "t1tnf"], 2))
    err = capsys.readouterr().err
    assert rc == 0, err
    assert "[pxdesign-opt] NOTHING RAN: 2/2 pairs already dumped (upstream resume)" in err and "requested=exact" in err
    assert "] DONE mode=" not in err and "] INCOMPLETE mode=" not in err
    assert (out / outputs.MANIFEST_NAME).read_bytes() == first                       # the earlier run's manifest, untouched
    # one new seed beside the dumped one: a partial resume, named on the DONE line and in the manifest
    stack.reset_for_tests(); _stubs.uninstall(); _rebox(monkeypatch, stack)
    rc = _design(monkeypatch, tmp_path, out, ckpt, _fake_run(str(out), ["t5o45", "t1tnf"], 2), seeds="101,102")
    err = capsys.readouterr().err
    assert rc == 0, err
    assert "[pxdesign-opt] DONE mode=exact designs=8 tasks=2 seeds=2 skipped_existing=2 " in err
    man = manifest.read(str(out))
    assert man["exit_code"] == 0 and man["outputs"]["skipped_pairs"] == ["t5o45/seed_101", "t1tnf/seed_101"] and man["outputs"]["produced"] == 4


def _rebox(monkeypatch, stack):
    _stubs.install_torch(); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack)


def test_incomplete_run_records_exit_1_in_the_manifest(fresh_stack, monkeypatch, capsys, tmp_path):
    """Outputs short of tasks x seeds x N_sample: INCOMPLETE line, rc 1, and the manifest — written after the verdict — says exit_code 1."""
    stack = fresh_stack
    ckpt = _box(monkeypatch, stack, tmp_path)
    out = tmp_path / "out"
    rc = _design(monkeypatch, tmp_path, out, ckpt, _fake_run(str(out), ["t5o45", "t1tnf"], 2, short=1))
    err = capsys.readouterr().err
    assert rc == cli.EXIT_FAIL == 1, err
    assert "[pxdesign-opt] INCOMPLETE mode=exact designs=2 tasks=2 seeds=1 skipped_existing=0 " in err
    man = manifest.read(str(out))
    assert man["exit_code"] == 1 and man["incomplete"] == "designs 2/4"


def test_hook_never_ran_is_not_active_3_in_process_and_manifest(fresh_stack, monkeypatch, capsys, tmp_path):
    """Designs appear but upstream's runner was never built through the hook: the exit census names it, rc 3, manifest exit_code 3."""
    stack = fresh_stack
    ckpt = _box(monkeypatch, stack, tmp_path)
    out = tmp_path / "out"
    rc = _design(monkeypatch, tmp_path, out, ckpt, _fake_run(str(out), ["t5o45", "t1tnf"], 2, build_runner=False, prepare=False))
    err = capsys.readouterr().err
    assert rc == cli.EXIT_NOT_ACTIVE == 3, err
    assert "[pxdesign-opt] NOT ACTIVE: lever run-time census: " + _stack_mod.HOOK_NEVER_RAN + " — exit 3" in err
    assert "] DONE mode=exact designs=4 " in err                                       # outputs kept and counted; the claim withdrawn by name above it
    man = manifest.read(str(out))
    assert man["exit_code"] == 3 and man["package_gate"] == [_stack_mod.HOOK_NEVER_RAN] and man["active"] is True
    assert "exit_judged" not in man["activation_report"] and "exit_problems" not in man["activation_report"]   # the verdict's latch is not run evidence


def test_designs_written_while_no_sampling_call_went_through_the_levers_is_3(fresh_stack, monkeypatch, capsys, tmp_path):
    stack = fresh_stack
    ckpt = _box(monkeypatch, stack, tmp_path)
    out = tmp_path / "out"
    rc = _design(monkeypatch, tmp_path, out, ckpt, _fake_run(str(out), ["t5o45", "t1tnf"], 2, prepare=False))
    err = capsys.readouterr().err
    assert rc == 3, err
    assert "NOT ACTIVE: lever run-time census: hoist: 4 design(s) written this run but no sample_diffusion call went through the installed levers (prepares=0)" in err
    assert manifest.read(str(out))["exit_code"] == 3


def test_incomplete_wins_over_a_census_problem(fresh_stack, monkeypatch, capsys, tmp_path):
    """The core's precedence: outputs short → 1 first; the census sentence still prints (not headed NOT ACTIVE: it did not decide the code)."""
    stack = fresh_stack
    ckpt = _box(monkeypatch, stack, tmp_path)
    out = tmp_path / "out"
    rc = _design(monkeypatch, tmp_path, out, ckpt, _fake_run(str(out), ["t5o45", "t1tnf"], 2, prepare=False, short=1))
    err = capsys.readouterr().err
    assert rc == 1, err
    assert "[pxdesign-opt] lever run-time census (recorded; exit 1 decided by the INCOMPLETE count): hoist: 2 design(s) written" in err and "NOT ACTIVE" not in err
    assert manifest.read(str(out))["exit_code"] == 1


def test_a_run_that_raises_propagates_with_no_manifest_and_no_run_line(fresh_stack, monkeypatch, capsys, tmp_path):
    """The kit's rule for a run that fails on its own (an OOM, any error): it propagates as raised — no exit code in its place, no manifest
    claiming anything, no DONE / INCOMPLETE line; the traceback is the record (test_oom_propagation locks the absence of a broad handler)."""
    stack = fresh_stack
    ckpt = _box(monkeypatch, stack, tmp_path)
    out = tmp_path / "out"; out.mkdir()

    def boom(argv, seeds, det, log=print, on_runner=None):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")
    with pytest.raises(RuntimeError):
        _design(monkeypatch, tmp_path, out, ckpt, boom)
    err = capsys.readouterr().err
    assert "] DONE mode=" not in err and "] INCOMPLETE mode=" not in err and not (out / outputs.MANIFEST_NAME).exists()


def test_env_route_exit_verdict(fresh_stack, monkeypatch, capsys):
    """PXDESIGN_OPT route (trigger set): the verdict is registered once; nothing to judge while nothing of the model ran; a model built without the
    hook → NOT ACTIVE 3; the hook ran and the census is clean → None; a census problem → the NOT ACTIVE line naming PXDESIGN_OPT=off, code 3,
    forced through the core's forced_exit by the atexit hook; a verb that judged the exit (exit_census) makes it stand down."""
    stack = fresh_stack
    _rebox(monkeypatch, stack)
    forced = []
    monkeypatch.setattr(_stack_mod._core_report, "forced_exit", lambda code: forced.append(code))
    shut = []
    monkeypatch.setattr(_stack_mod.logging, "shutdown", lambda *a, **k: shut.append(len(forced)))   # records how many exits were forced BEFORE each shutdown
    stack.activate("exact", strict=True, trigger="pxdesign")
    assert _stack_mod._VERDICT["registered"] is True and stack.register_exit_verdict() is False
    assert stack.exit_verdict() is None                                                # a help text / an input check: no application, no model built
    import pxdesign.model.pxdesign as P
    P.ProtenixDesign()                                                                 # a model built in this process that the hook never reached
    assert stack.exit_verdict() == 3
    err = capsys.readouterr().err
    assert "[pxdesign-opt] NOT ACTIVE: lever run-time census: " + _stack_mod.HOOK_NEVER_RAN in err and "PXDESIGN_OPT=off runs stock" in err
    from pxdesign.runner.inference import InferenceRunner
    InferenceRunner(configs=None)                                                      # the hook runs: levers installed
    stack._REPORT.pop("exit_judged", None)
    assert stack.exit_verdict() is None                                                # clean census: the process keeps its own status
    monkeypatch.setattr(_stack_mod, "runtime_gate", lambda rep=None: ["sdedup: planned and installed but served nothing"])
    stack._REPORT.pop("exit_judged", None)
    _stack_mod._exit_verdict_hook()
    assert forced == [3] and shut == [0] and "sdedup: planned and installed but served nothing" in capsys.readouterr().err   # upstream's log handlers shut down BEFORE the forced status
    stack.exit_census()                                                                # a verb priced the exit: the verdict stands down
    assert stack.exit_verdict() is None
    _stack_mod._exit_verdict_hook()
    assert forced == [3]                                                               # nothing forced when there is nothing to judge
    monkeypatch.setattr(_stack_mod, "exit_census", lambda produced=None: (_ for _ in ()).throw(KeyError("stats")))
    stack._REPORT.pop("exit_judged", None)
    _stack_mod._exit_verdict_hook()                                                    # fail-closed: a verdict that cannot be read is named and forces 3
    assert forced == [3, 3] and shut == [0, 1] and "NOT ACTIVE: exit census unreadable at interpreter exit: KeyError('stats')" in capsys.readouterr().err


def test_cli_route_registers_no_exit_verdict_hook(fresh_stack, monkeypatch):
    """The verb owns its exit: activation without a trigger registers the EXIT tally only (the verdict is the PXDESIGN_OPT route's)."""
    stack = fresh_stack
    _rebox(monkeypatch, stack)
    monkeypatch.setattr(_stack_mod, "_VERDICT", {"registered": False})
    stack.activate("exact", strict=True)
    assert _stack_mod._VERDICT["registered"] is False
