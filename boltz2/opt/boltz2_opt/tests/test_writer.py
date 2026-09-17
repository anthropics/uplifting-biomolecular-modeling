"""boltz2_opt.writer — the output writer off the critical path (registry lever ``writer_overlap``, switch ``BOLTZ_WRITER=overlap``).

CPU tests (torch + numpy required; skipped without them): the switch words / dispositions / LEVER line grammar and the fail-closed gate; the
background mechanism end to end on host tensors with the thread backend over a stand-in writer function (submit -> write -> the worker's move ->
collect -> join: the files land under by_seed/<name>/s<seed>/ with the bytes the same function writes inline, in FIFO order per input); a write
that raises is FAILED BY NAME (collect() names the unit and the reason, the census counts it, the gate refuses, join still returns); the
registry / modes / attach wiring (word, lever, attachment beside the featurizer in every row; off by name at n_gpu > 1; an ablation takes the
word and the attachment). With boltz importable: the writer PROCESS is spawned through the zygote (forked before CUDA), serves a move, and a
write it cannot perform comes back failed by name — never a silent loss.
"""
import json
import os
import types

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from .. import writer as WR   # noqa: E402
from .. import prep as PREP   # noqa: E402


def _fake_write(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
    """A stand-in for BoltzWriter.write_on_batch_end with the same reads: files under output_dir/<record.id>/ from host tensors."""
    if prediction["exception"]:
        self.failed += 1
        return
    for rec in batch["record"]:
        d = os.path.join(str(self.output_dir), rec.id); os.makedirs(d, exist_ok=True)
        if rec.id == "boom":
            raise RuntimeError("disk on fire")
        np.savez_compressed(os.path.join(d, f"pae_{rec.id}_model_0.npz"), pae=prediction["pae"].cpu().numpy())
        with open(os.path.join(d, f"confidence_{rec.id}_model_0.json"), "w") as f:
            json.dump({"confidence_score": round(prediction["confidence_score"].reshape(-1)[0].item(), 6), "plddt0": prediction["plddt"][0].item()}, f, indent=4)
        with open(os.path.join(d, f"{rec.id}_model_0.cif"), "w") as f:
            f.write("data_" + rec.id + "\n" + " ".join(f"{v:.2f}" for v in (prediction["plddt"] * 100).tolist()) + "\n")


def _writer_stub(out):
    return types.SimpleNamespace(data_dir=str(out / "processed"), output_dir=str(out / "predictions"), output_format="mmcif", boltz2=True, write_embeddings=False, failed=0)


def _pred(n, score):
    return {"exception": False, "pae": torch.arange(n * n, dtype=torch.float32).reshape(1, n, n) / 7.0, "plddt": torch.linspace(0.1, 0.9, n),
            "confidence_score": torch.tensor([score]), "coords": torch.zeros(1, n, 3), "masks": torch.ones(1, n)}


@pytest.fixture()
def thread_backend(monkeypatch, tmp_path):
    WR.reset_for_tests()
    monkeypatch.setattr(WR, "_writer_for", lambda cfg: types.SimpleNamespace(**{**cfg, "failed": 0}))
    WR._start("thread", _fake_write)
    yield tmp_path
    WR.reset_for_tests()


def test_switch_words_dispositions_line_and_gate(monkeypatch):
    WR.reset_for_tests()
    assert WR.words({}) == [] and not WR.requested({}) and WR.levers_of({}) == []
    assert WR.requested({"BOLTZ_WRITER": "overlap"}) and WR.levers_of({"BOLTZ_WRITER": "overlap,thread"}) == ["writer_overlap"]
    assert WR.backend_of({"BOLTZ_WRITER": "overlap"}) == "process" and WR.backend_of({"BOLTZ_WRITER": "overlap,thread"}) == "thread"
    assert "unknown word" in WR.problems({"BOLTZ_WRITER": "overlap,bogus"})[0] and WR.problems({"BOLTZ_WRITER": "thread"})
    monkeypatch.setenv("BOLTZ_WRITER", "overlap")
    monkeypatch.setattr(WR.sys, "argv", ["bz_worker.py", "--pipeline", "0"])
    assert WR.apply() == [] and WR.dispositions() == {"writer_overlap": "skipped_by_name:pipeline0"}
    ln = WR.line()
    assert ln.startswith("[boltz2-opt] LEVER name=writer_overlap state=skipped reason=pipeline0 impl=boltz2_opt.writer origin=kit") and " " not in ln.split("execution=")[1], ln
    # the gate: fail-closed until joined with everything served and nothing failed / fallen back
    WR.STATE.update(installed=True, disposition=None, backend="process", forked_before_cuda=True, submitted=3, served=3, moves=3, moves_done=3, joined=False)
    assert not WR.gate()["ok"] and "not_joined" in WR.gate()["why"]
    WR.STATE.update(joined=True)
    g = WR.gate(); ln = WR.line()
    assert g["ok"] and "state=on" in ln and "backend=process" in ln and "served=3" in ln and "failed=0" in ln and "execution=executed:3" in ln, (g, ln)
    WR.STATE.update(failed=1)
    assert not WR.gate()["ok"] and "failed_writes" in WR.gate()["why"]
    WR.STATE.update(failed=0, fallback=1, fallback_by={"helper_died:gone": 1}, dead="helper_died:gone")
    g = WR.gate()
    assert not g["ok"] and "fallbacks" in g["why"] and "dead" in g["why"] and "fallback_by=helper_died:gone:1" in WR.line() and "execution=dead:" in WR.line()
    WR.STATE.update(fallback=0, fallback_by={}, dead=None, forked_before_cuda=False)
    assert not WR.gate()["ok"] and "forked_after_cuda" in WR.gate()["why"]
    WR.STATE.update(forked_before_cuda=True, served=2)
    assert not WR.gate()["ok"] and "unserved" in WR.gate()["why"]
    r = WR.report()
    assert r["installed"] and r["applied"] == ["writer_overlap"] and "gate" in r and "line" in r and r["submitted"] == 3
    WR.reset_for_tests()
    assert not WR.STATE["installed"] and WR.report()["applied"] == []


def test_background_write_move_collect_join_thread_backend(thread_backend):
    out = thread_backend
    stub = _writer_stub(out)
    recs = [types.SimpleNamespace(id="alpha")]
    # item 1: alpha seed 0 — background write, then the worker's move queued behind it
    WR.begin_item("alpha", 0)
    assert WR._write_on_batch_end(stub, None, None, _pred(5, 0.75), None, {"record": recs}, 0, 0) is None
    assert WR.STATE["submitted"] == 1 and WR.STATE["items"] == 1
    pred_dir = out / "predictions" / "alpha"; dst0 = out / "by_seed" / "alpha" / "s0"
    assert WR.finish_item("alpha", 0, pred_dir, dst0) is None            # pending: queued (None), the list arrives with collect()
    # item 2: alpha seed 1 (same input name: FIFO behind seed 0's write AND move, so the predictions dir never mixes seeds)
    WR.begin_item("alpha", 1)
    WR._write_on_batch_end(stub, None, None, _pred(5, 0.5), None, {"record": recs}, 0, 0)
    dst1 = out / "by_seed" / "alpha" / "s1"
    assert WR.finish_item("alpha", 1, pred_dir, dst1) is None
    # a prediction boltz skipped: inline by name, nothing written, the stub's failed counter as stock's
    WR.begin_item("alpha", 2)
    WR._write_on_batch_end(stub, None, None, {"exception": True}, None, {"record": recs}, 0, 0)
    assert WR.STATE["inline"] == 1 and WR.STATE["inline_by"] == {"exception": 1} and stub.failed == 1
    assert WR.finish_item("alpha", 2, pred_dir, out / "by_seed" / "alpha" / "s2") in (None, [])   # queued behind the unit's earlier jobs (None) or inline once they drained ([]: nothing to move): either is the stock result
    WR.wait_item("alpha")
    c = WR.join()
    assert c["complete"] and c["served"] == 2 and c["moves_done"] == c["moves"] and c["failed"] == 0, c
    done = WR.collect()
    moves = [d for d in done if d["kind"] == "move" and d["ok"]]
    assert sorted((d["name"], d["seed"]) for d in moves)[:2] == [("alpha", 0), ("alpha", 1)]
    for dst, score in ((dst0, 0.75), (dst1, 0.5)):
        names = sorted(os.listdir(dst))
        assert names == ["alpha_model_0.cif", "confidence_alpha_model_0.json", "pae_alpha_model_0.npz"], names
        assert json.load(open(dst / "confidence_alpha_model_0.json"))["confidence_score"] == score
    assert os.listdir(pred_dir) == []
    # bytes: the background files == the same function inline on the same values
    ref = out / "ref"; stub2 = types.SimpleNamespace(**{**vars(stub), "output_dir": str(ref)})
    _fake_write(stub2, None, None, _pred(5, 0.75), None, {"record": recs}, 0, 0)
    for f in os.listdir(dst0):
        assert open(dst0 / f, "rb").read() == open(ref / "alpha" / f, "rb").read(), f
    g = WR.gate(); ln = WR.line()
    assert g["ok"] and "state=on" in ln and "backend=thread" in ln and "served=2" in ln and "inline_by=exception:1" in ln and "moves=" in ln and "bytes_written=" in ln, (g, ln)
    assert WR.STATE["bytes_written"] == sum(os.path.getsize(d / f) for d in (dst0, dst1) for f in os.listdir(d))


def test_a_write_that_raises_fails_the_unit_by_name(thread_backend):
    out = thread_backend
    stub = _writer_stub(out)
    WR.begin_item("boom", 7)
    WR._write_on_batch_end(stub, None, None, _pred(4, 0.1), None, {"record": [types.SimpleNamespace(id="boom")]}, 0, 0)
    assert WR.finish_item("boom", 7, out / "predictions" / "boom", out / "by_seed" / "boom" / "s7") is None
    WR.begin_item("fine", 0)
    WR._write_on_batch_end(stub, None, None, _pred(4, 0.2), None, {"record": [types.SimpleNamespace(id="fine")]}, 0, 0)
    WR.finish_item("fine", 0, out / "predictions" / "fine", out / "by_seed" / "fine" / "s0")
    c = WR.join()
    assert not c["complete"] and c["failed"] == 1 and c["served"] == 1 and WR.failed_count() == 1, c
    bad = [d for d in WR.collect() if not d["ok"]]
    assert len(bad) == 1 and bad[0]["kind"] == "write" and (bad[0]["name"], bad[0]["seed"]) == ("boom", 7) and "disk on fire" in bad[0]["error"], bad
    assert any("disk on fire" in e for e in c["errors"])
    g = WR.gate()
    assert not g["ok"] and "failed_writes" in g["why"] and "failed=1" in WR.line()
    assert sorted(os.listdir(out / "by_seed" / "fine" / "s0")) == ["confidence_fine_model_0.json", "fine_model_0.cif", "pae_fine_model_0.npz"]   # the other unit is whole
    # after join every call is the stock call inline (fallback counted by name)
    WR._write_on_batch_end(stub, None, None, _pred(4, 0.3), None, {"record": [types.SimpleNamespace(id="late")]}, 0, 0)
    assert WR.STATE["fallback"] == 1 and WR.STATE["fallback_by"] == {"joined": 1} and os.path.isfile(out / "predictions" / "late" / "late_model_0.cif")


def test_tied_scores_take_the_stock_call_inline(thread_backend):
    out = thread_backend
    stub = _writer_stub(out)
    p = _pred(3, 0.4); p["confidence_score"] = torch.tensor([0.4, 0.4])
    WR.begin_item("tie", 0)
    WR._write_on_batch_end(stub, None, None, p, None, {"record": [types.SimpleNamespace(id="tie")]}, 0, 0)
    assert WR.STATE["inline_by"] == {"tied_scores": 1} and WR.STATE["submitted"] == 0
    p["confidence_score"] = torch.tensor([0.4, float("nan")])
    WR._write_on_batch_end(stub, None, None, p, None, {"record": [types.SimpleNamespace(id="tie")]}, 0, 0)
    assert WR.STATE["inline_by"] == {"tied_scores": 2}
    p["confidence_score"] = torch.tensor([0.4, 0.5])
    WR._write_on_batch_end(stub, None, None, p, None, {"record": [types.SimpleNamespace(id="tie")]}, 0, 0)
    assert WR.STATE["submitted"] == 1
    WR.join()
    assert WR.gate()["ok"]


def test_registry_modes_attach_wiring():
    from .. import modes, registry, worker_launch
    L = registry.LEVERS["writer_overlap"]
    assert L["switch"] == "BOLTZ_WRITER" and L["value"].startswith("overlap") and L["tier"] == 1
    assert registry.LEVER_IDS["writer_overlap"]["strategy"] == WR.STRATEGY and registry.LEVER_IDS["writer_overlap"]["impl"] == WR.IMPL
    assert worker_launch.ATTACH["writer"]["module"] == "boltz2_opt.writer" and worker_launch.ATTACH["writer"]["report_key"] == "writer_report"
    assert modes.LEVER_ATTACH["writer_overlap"] == "writer"
    for m in ("exact", "fast", "big"):
        row = modes.resolve(m)
        assert row["env"].get("BOLTZ_WRITER") == "overlap", m
        assert "writer_overlap" in row["levers"] and row["levers"].index("writer_overlap") < row["levers"].index("prefetch"), (m, row["levers"])
        assert "writer" in row["attach"] and row["attach"].index("writer") < row["attach"].index("prefetch") and row["attach"][-1] == "prefetch", (m, row["attach"])
        assert list(row["env"])[-1] == "BOLTZ_PREFETCH" and list(row["env"])[-2] == "BOLTZ_WRITER", (m, list(row["env"])[-3:])
        # the row's ablation entry takes the word and the attachment off by name
        r2 = modes.resolve(m); modes.take_off(r2, "writer_overlap", "ablation", "off")
        assert "BOLTZ_WRITER" not in r2["env"] and "writer" not in r2["attach"] and "writer_overlap" not in r2["levers"] and r2["off"]["writer_overlap"] == "ablation", m
    # the xP line keeps its inline writer, by name
    assert "writer_overlap" in modes.TP_DROPS["big"]
    P0 = modes.n_gpu_set()
    try:
        modes.set_n_gpu(2)
        rx = modes.resolve("big")
        assert "writer_overlap" not in rx["levers"] and "BOLTZ_WRITER" not in rx["env"] and "writer" not in rx["attach"] and rx["tp_off"]["writer_overlap"].startswith("xP_line_not_measured")
    finally:
        modes.set_n_gpu(P0)


def test_the_writer_process_through_the_zygote(tmp_path):
    """boltz importable: the writer process is spawned through the zygote (no CUDA in its lineage), serves a move, names a write it cannot do."""
    pytest.importorskip("boltz")
    WR.reset_for_tests()
    z = PREP.zygote()
    if z is None:
        pytest.skip("no zygote in this process")
    h = WR.Helper()
    try:
        assert h.alive() and h.helper_cuda_initialized is False
        src = tmp_path / "predictions" / "x"; src.mkdir(parents=True); (src / "x_model_0.cif").write_text("data_x\n")
        rep = h.call(("move", 1, str(src), str(tmp_path / "by_seed" / "x" / "s0")))
        assert rep[0] == "ok" and rep[2]["files"] == ["x_model_0.cif"] and (tmp_path / "by_seed" / "x" / "s0" / "x_model_0.cif").read_text() == "data_x\n"
        cfg = {"data_dir": str(tmp_path / "nowhere"), "output_dir": str(tmp_path / "predictions"), "output_format": "mmcif", "boltz2": True, "write_embeddings": False}
        from boltz.data.types import Record  # noqa: F401  (the helper unpickles boltz records; a namespace stands in for one here)
        payload = WR._to_numpy({"exception": False, "coords": torch.zeros(1, 3, 3), "masks": torch.ones(1, 3), "confidence_score": torch.tensor([0.5])})
        rep = h.call(("write", 2, cfg, payload, [types.SimpleNamespace(id="x")]))
        assert rep[0] == "err" and rep[1] == 2 and rep[2], rep          # the structure npz is not there: named, not lost
        assert h.alive()                                                # the writer process survives a failed job
    finally:
        h.close()
