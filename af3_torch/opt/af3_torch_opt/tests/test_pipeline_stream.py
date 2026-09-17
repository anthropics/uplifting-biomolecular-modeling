"""The streamed process chain (package levers `prefetch` / `write_behind`, pipeline_stream.py): the hand-off protocol's waits resolve to
ready / withdrawn exactly as the sequential chain would have handed the item on (or not), a ready marker written before a terminal marker is
never withdrawn, the SEEDS line round-trips, a background Job announces / marks its step done / returns run_step's record, the overlap
arithmetic; and through cli.cmd_pred on the stub box: the chain streams under the shipped modes (three processes launched together, the same
records), runs sequentially when the two levers are named in MODEL_OPT_LEVERS_OFF, steps aside by name under --n_gpu > 1, and goes on
sequentially from the featurise report when the featuriser never announces."""
import json
import os
import threading
import time

from af3_torch_opt import cli, modes, pipeline_stream as PS, registry

from .conftest import stub_calls
from .test_cli_manifest import _inputs


def test_levers_are_registered_package_levers_of_every_mode_but_off():
    for lever in PS.STREAM_LEVERS:
        assert lever in registry.PACKAGE_LEVERS and registry.EVIDENCE[lever] == "package" and registry.STRATEGY[lever].startswith("F6.")
        assert registry.BITWISE_EVIDENCE[lever] == {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True}
        assert all((lever in modes.MODE_PACKAGE_LEVERS[m]) == (m != "off") for m in modes.MODES)
    assert registry.STRATEGY["prefetch"] == "F6.item_ordering_prefetch" and registry.STRATEGY["write_behind"] == "F6.output_overlap"


def _seed_dir(tmp_path, name="a", seed=1):
    d = tmp_path / "work" / name / f"seed-{seed}"
    d.mkdir(parents=True)
    return d


def test_await_batch_ready_and_withdrawn(tmp_path):
    root = str(tmp_path / "work"); d = _seed_dir(tmp_path)
    batch = str(d / "batch.npz")
    # ready after a short wait
    threading.Timer(0.1, lambda: PS.touch(str(d / PS.BATCH_READY))).start()
    w = PS.await_batch(batch, root, poll_s=0.01)
    assert w["ready"] and w["wait_s"] >= 0.05
    # the item's featurisation failed: withdrawn by that name
    d2 = _seed_dir(tmp_path, "b"); PS.touch(str(d2.parent / PS.ITEM_FAILED))
    assert PS.await_batch(str(d2 / "batch.npz"), root, poll_s=0.01) == {"ready": False, "wait_s": 0.0, "reason": "featurise_failed"}
    # the featuriser gone without this seed: withdrawn
    d3 = _seed_dir(tmp_path, "c"); PS.touch(PS.step_done_path(root, "featurise"))
    assert PS.await_batch(str(d3 / "batch.npz"), root, poll_s=0.01)["reason"] == "featuriser_gone"
    # a ready marker written before the terminal marker is honoured even when the terminal marker is seen first in the loop
    d4 = _seed_dir(tmp_path, "d"); PS.touch(str(d4 / PS.BATCH_READY)); PS.touch(str(d4.parent / PS.ITEM_OK))
    assert PS.await_batch(str(d4 / "batch.npz"), root, poll_s=0.01)["ready"] is True
    # the item complete (featurise.ok) but this seed never handed over: withdrawn (a seed the wrapper listed that the featuriser did not produce)
    d5 = _seed_dir(tmp_path, "e"); PS.touch(str(d5.parent / PS.ITEM_OK))
    assert PS.await_batch(str(d5 / "batch.npz"), root, poll_s=0.01)["reason"] == "seed_not_featurised"


def test_await_results_states(tmp_path):
    root = str(tmp_path / "work")
    # every seed handed over while the model process is live: ready, early
    d = _seed_dir(tmp_path, "a"); (d / "batch.pkl").write_bytes(b"P"); PS.touch(str(d.parent / PS.ITEM_OK)); PS.touch(str(d / PS.RESULT_READY))
    w = PS.await_results(str(d.parent), root, poll_s=0.01)
    assert (w["state"], w["early"], w["seeds"]) == ("ready", 1, 1)
    # featurisation failed: withdrawn
    d2 = _seed_dir(tmp_path, "b"); PS.touch(str(d2.parent / PS.ITEM_FAILED))
    assert PS.await_results(str(d2.parent), root, poll_s=0.01)["state"] == "withdrawn"
    # two seeds, one result, model process gone: ready (postprocess names the missing seed as it always has), not early
    d3 = _seed_dir(tmp_path, "c", 1); d3b = _seed_dir(tmp_path, "c", 2)
    for x in (d3, d3b):
        (x / "batch.pkl").write_bytes(b"P")
    PS.touch(str(d3.parent / PS.ITEM_OK)); PS.touch(str(d3 / PS.RESULT_READY)); PS.touch(PS.step_done_path(root, "forward"))
    w = PS.await_results(str(d3.parent), root, poll_s=0.01)
    assert (w["state"], w["early"], w["seeds"]) == ("ready", 0, 2)
    # no result at all once the model process is gone: withdrawn (the sequential chain never hands such an input to the writers)
    d4 = _seed_dir(tmp_path, "dd"); (d4 / "batch.pkl").write_bytes(b"P"); PS.touch(str(d4.parent / PS.ITEM_OK))
    assert PS.await_results(str(d4.parent), root, poll_s=0.01) == {"state": "withdrawn", "reason": "no_result", "wait_s": 0.0, "early": 0}


def test_await_results_waits_for_the_last_seed(tmp_path):
    root = str(tmp_path / "work")
    d1 = _seed_dir(tmp_path, "a", 1); d2 = _seed_dir(tmp_path, "a", 2)
    for x in (d1, d2):
        (x / "batch.pkl").write_bytes(b"P")
    PS.touch(str(d1.parent / PS.ITEM_OK)); PS.touch(str(d1 / PS.RESULT_READY))
    threading.Timer(0.15, lambda: PS.touch(str(d2 / PS.RESULT_READY))).start()
    w = PS.await_results(str(d1.parent), root, poll_s=0.01)
    assert w["state"] == "ready" and w["early"] == 1 and w["wait_s"] >= 0.1


def test_seeds_line_round_trip():
    ln = PS.seeds_line({"a.af3": [1, 2], "b": [7]})
    assert ln.startswith(PS.SEEDS_MARK) and PS.parse_seeds_line(ln + "\n") == {"a.af3": [1, 2], "b": [7]}
    assert PS.parse_seeds_line('{"a": [1]}') is None and PS.parse_seeds_line("SEEDS not-json") is None and PS.parse_seeds_line('SEEDS ["a"]') is None


def test_job_announces_marks_done_and_returns_the_record(tmp_path):
    root = str(tmp_path / "work"); os.makedirs(root)
    seen = []

    def fake_run(step, argv, env, log_path, on_line=None):
        on_line("noise\n"); on_line(PS.seeds_line({"x": [3]}) + "\n"); time.sleep(0.05)
        return {"step": step, "rc": 0, "ok": True, "wall_s": 0.05, "argv": argv, "log": log_path}

    j = PS.Job("featurise", root, fake_run, ["py", "featurise.py"], {}, os.path.join(root, "f.log"), on_line=seen.append)
    assert j.wait_announced() == {"x": [3]}
    rec = j.join()
    assert rec["step"] == "featurise" and rec["ok"] and os.path.exists(PS.step_done_path(root, "featurise")) and seen[0] == "noise\n"
    a, b = j.span
    assert b is not None and b >= a
    # a step that never announces releases its waiter when it ends, with None
    j2 = PS.Job("forward", root, lambda *a, **k: {"step": "forward", "rc": 1, "ok": False, "wall_s": 0.0}, [], {}, os.path.join(root, "g.log"))
    assert j2.wait_announced() is None and j2.join()["rc"] == 1 and os.path.exists(PS.step_done_path(root, "forward"))
    # an exception in run is re-raised on join, and the done marker is still written
    def boom(*a, **k):
        raise RuntimeError("x")
    j3 = PS.Job("postprocess", root, boom, [], {}, os.path.join(root, "h.log"))
    try:
        j3.join(); raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    assert os.path.exists(PS.step_done_path(root, "postprocess"))


def test_overlap_arithmetic():
    assert PS.overlap_s([(0.0, 10.0), (2.0, 12.0), (11.0, 13.0)]) == 9.0      # walls 10+10+2=22, union 13 → 9 s ran concurrently
    assert PS.overlap_s([(0.0, 1.0), (2.0, 3.0)]) == 0.0 and PS.overlap_s([]) == 0.0 and PS.overlap_s([(0.0, None)]) == 0.0


# ---- through cmd_pred on the stub box ----

def _lever_lines(err):
    return {l.split("name=")[1].split()[0]: l for l in err.splitlines() if l.startswith("[af3-torch-opt] LEVER ")}


def test_pred_streams_under_the_shipped_modes(box, capsys):
    out = box["tmp"] / "out"
    ins = _inputs(box["tmp"], seeds=(1, 2))
    assert cli.main(["pred", "--mode", "exact", "--json_path", ins[0], "--json_path", ins[1], "--output_dir", str(out)]) == 0
    M = cli.last_run(); err = capsys.readouterr().err
    assert M["activation"]["stream"] == {"selected": ["prefetch", "write_behind", "feat_par"], "engaged": ["feat_par", "prefetch", "write_behind"], "skipped": {}}   # two inputs: two featurisers (feat_par)
    assert M["reports"]["featurise"]["workers"] == 2 and M["activation"]["feat_par"]["workers"] == 2 and [it["name"] for it in M["reports"]["featurise"]["items"]] == ["a", "b"]
    calls = {os.path.basename(c[1]): c for c in stub_calls(box)}
    assert calls["featurise.py"][calls["featurise.py"].index("--stream") + 1] == "1"
    work = os.path.join(str(out), cli.WORK_DIR)
    assert calls["forward.py"][calls["forward.py"].index("--stream-root") + 1] == work and calls["postprocess.py"][calls["postprocess.py"].index("--follow") + 1] == work
    assert [s["step"] for s in M["steps"]] == ["featurise", "forward", "postprocess"]                 # the records in chain order, one STEP line each
    assert err.count("[af3-torch-opt] STEP step=") == 4 and err.count("[af3-torch-opt] COMMAND step=") == 4   # two featuriser processes (feat_par), the model process, the writers
    levers = _lever_lines(err)
    assert " state=on " in levers["prefetch"] and "strategy=F6.item_ordering_prefetch" in levers["prefetch"] and " items=4 " in levers["prefetch"] and " engaged=1 " in levers["prefetch"]
    assert " state=on " in levers["write_behind"] and " published=4 " in levers["write_behind"] and " items=2 " in levers["write_behind"] and " early=" in levers["write_behind"]   # early = inputs written while the model process was still live (0..2 here: the stub's model process ends at once)
    assert M["items"][0]["ok"] and M["items"][1]["ok"] and (out / "a" / "a_model.cif").exists() and not os.path.exists(work)   # a clean pred: the stock layout only
    assert "PARTIAL" not in err and " DONE ok=1 rc=0 items=2/2 " in err


def test_levers_off_runs_the_sequential_chain(box, capsys, monkeypatch):
    monkeypatch.setenv(modes.ENV_LEVERS_OFF, "prefetch,write_behind")
    out = box["tmp"] / "out"
    ins = _inputs(box["tmp"])
    assert cli.main(["pred", "--mode", "fast", "--json_path", ins[0], "--json_path", ins[1], "--output_dir", str(out)]) == 0
    M = cli.last_run(); err = capsys.readouterr().err
    assert M["activation"]["stream"] == {"selected": ["feat_par"], "engaged": [], "skipped": {"feat_par": "requires:prefetch"}}   # feat_par rides prefetch: named, one featuriser
    assert sum(1 for c in stub_calls(box) if os.path.basename(c[1]) == "featurise.py") == 1
    calls = stub_calls(box)
    assert [os.path.basename(c[1]) for c in calls] == ["featurise.py", "forward.py", "postprocess.py"]
    flat = [x for c in calls for x in c]
    assert "--stream" not in flat and "--stream-root" not in flat and "--follow" not in flat
    levers = _lever_lines(err)
    assert " state=off reason=levers_off " in levers["prefetch"] and " state=off reason=levers_off " in levers["write_behind"]
    assert err.rstrip().splitlines()[-1].startswith("[af3-torch-opt] DONE ok=1 rc=0") and "levers_off=prefetch,write_behind" in err


def test_n_gpu_above_one_steps_aside_by_name(box, capsys, monkeypatch):
    monkeypatch.setattr(modes, "N_GPU_SUPPORTED", (1, 2, 4)); monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    out = box["tmp"] / "out"
    assert cli.main(["pred", "--mode", "big", "--n_gpu", "2", "--json_path", _inputs(box["tmp"])[0], "--output_dir", str(out)]) == 0
    M = cli.last_run(); err = capsys.readouterr().err
    assert M["activation"]["stream"]["engaged"] == [] and M["activation"]["stream"]["skipped"] == {"prefetch": "n_gpu>1:rowpair_ranks", "write_behind": "n_gpu>1:rowpair_ranks", "feat_par": "n_gpu>1:rowpair_ranks"}
    fwd = [c for c in stub_calls(box) if os.path.basename(c[1]) == "forward.py"][0]
    assert "prefetch" not in fwd[fwd.index("--package-levers") + 1] and "--stream-root" not in fwd
    levers = _lever_lines(err)
    assert " state=skipped reason=skipped:n_gpu>1:rowpair_ranks " in levers["prefetch"] and " state=skipped " in levers["write_behind"]
    assert "PARTIAL" not in err and "scope=package" not in err


def test_a_featuriser_that_never_announces_goes_on_sequentially(box, capsys, monkeypatch):
    monkeypatch.setenv("STUB_NO_SEEDS", "1")
    out = box["tmp"] / "out"
    assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"])[0], "--output_dir", str(out)]) == 0
    M = cli.last_run()
    assert M["activation"]["stream"]["engaged"] == [] or M["activation"]["stream"]["engaged"] == ["write_behind"]   # prefetch did not engage (no announcement); the writers may still follow the model process
    assert "prefetch" not in M["activation"]["stream"]["engaged"] and M["items"][0]["ok"]
    assert [s["step"] for s in M["steps"]] == ["featurise", "forward", "postprocess"]


def test_release_sources_carry_the_levers():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    readme = open(os.path.join(here, "README.md"), encoding="utf-8").read(); changes = open(os.path.join(here, "CHANGES.md"), encoding="utf-8").read()
    for lever in PS.STREAM_LEVERS:
        assert f"`{lever}`" in readme and f"`{lever}`" in changes, lever
