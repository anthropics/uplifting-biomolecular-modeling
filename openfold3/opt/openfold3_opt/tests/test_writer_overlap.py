"""`writer_overlap` (cells/of3_writer_overlap.py + cells/writer_overlap.py): the engine's writer callback re-stated so item k's files are rendered
and written by one writer worker (forked process | thread) while item k+1 runs — on a file-backed stub of the engine's writer whose
`on_predict_batch_end` IS the engine's text (sliced from stock/src, so the kit's digest pin is proven against the tree): every route writes the
files the inline route writes, byte for byte; the callback's bookkeeping (total / success / failed / failed_queries) equals the engine's,
a failing item included; write_features / a distributed predict are written inline by name; on_predict_end sees every item drained; the census
words; the kit wiring (every kit line but off / tp exports the switch, the registry row, the probe, the .pth gate's declaration, LEVERS_OFF)."""
import ast
import importlib
import json
import os
import sys
import textwrap
import types

import numpy as np
import pytest

from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()
STOCK_WRITER = os.path.join(HOME, "stock", "src", "openfold3", "core", "runners", "writer.py")


def _engine_method_text(name="on_predict_batch_end"):
    src = open(STOCK_WRITER, encoding="utf-8").read()
    lines = src.splitlines(keepends=True)
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ClassDef) and n.name == "OF3OutputWriter":
            for m in n.body:
                if isinstance(m, ast.FunctionDef) and m.name == name:
                    return "".join(lines[(m.decorator_list[0].lineno if m.decorator_list else m.lineno) - 1:m.end_lineno])
    raise AssertionError(name)


class _FakeTensor:
    """Enough of torch.Tensor for the writer's statements: x[b], .cpu(), .float(), .numpy(), .detach(), contiguity."""
    def __init__(self, a, device="cuda"):
        self.a, self.device = np.asarray(a), device
    def detach(self): return self
    def cpu(self): return _FakeTensor(self.a.copy() if self.device != "cpu" else self.a, "cpu")
    def float(self): return self if self.a.dtype == np.float32 else _FakeTensor(self.a.astype(np.float32), self.device)
    def is_contiguous(self): return self.a.flags["C_CONTIGUOUS"]
    def contiguous(self): return _FakeTensor(np.ascontiguousarray(self.a), self.device)
    def numpy(self):
        assert self.device == "cpu", "numpy() of a device tensor"
        return self.a
    def __getitem__(self, i): return _FakeTensor(self.a[i], self.device)
    @property
    def shape(self): return self.a.shape


def _live():
    """The package's modules as sys.modules holds them now (tests/_stubs.fresh() drops them between tests; a stale module object would pickle the
    job's leaves by a dead class reference)."""
    g = globals()
    g["_autoload"], g["modes"], g["registry"], g["stack"] = (importlib.import_module("openfold3_opt." + n) for n in ("_autoload", "modes", "registry", "stack"))
    g["core"] = importlib.import_module("openfold3_opt.cells.of3_writer_overlap")
    g["cell"] = importlib.import_module("openfold3_opt.cells.writer_overlap")


_live()


@pytest.fixture(autouse=True)
def live_modules():
    _live()
    yield


@pytest.fixture()
def engine(tmp_path, monkeypatch, live_modules):
    """A fake `torch` and a file-backed stub writer module `nfw_stub_writer` whose OF3OutputWriter carries the ENGINE'S on_predict_batch_end text."""
    torch = types.ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch.from_numpy = lambda a: _FakeTensor(a, "cpu")
    torch.set_num_threads = lambda n: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    body = textwrap.indent(textwrap.dedent(_engine_method_text()), "    ")
    src = textwrap.dedent('''
        import json, logging, os
        from pathlib import Path
        import numpy as np
        import torch
        logger = logging.getLogger(__name__)
        SUMMARY = {}
        def _take_batch_dim(x, b):
            if isinstance(x, torch.Tensor):
                return x[b].cpu().float().numpy() if len(x.shape) > 1 else x
            if isinstance(x, dict):
                return {k: _take_batch_dim(v, b) for k, v in x.items()}
            return x
        class OF3OutputWriter:
            def __init__(self, output_dir, write_features=False, write_latent_outputs=False):
                self.output_dir = Path(output_dir); self.write_features = write_features; self.write_latent_outputs = write_latent_outputs
                self.total_count = 0; self.success_count = 0; self.failed_count = 0; self.failed_queries = []
            def write_all_outputs(self, batch, outputs, confidence_scores):
                for b in range(len(batch["atom_array"])):
                    seed, qid = batch["seed"][b], batch["query_id"][b]
                    prefix = self.output_dir / qid / f"seed_{seed}" / f"{qid}_seed_{seed}"
                    prefix.parent.mkdir(parents=True, exist_ok=True)
                    coords = outputs["atom_positions_predicted"][b].cpu().float().numpy()
                    conf = _take_batch_dim(confidence_scores, b)
                    if qid == "boom":
                        raise RuntimeError("the engine's writer failed on this item")
                    for s in range(coords.shape[0]):
                        text = json.dumps({"atoms": batch["atom_array"][b], "xyz": coords[s].tolist(), "plddt": conf["plddt"][s].tolist(),
                                           "pae": conf["pae"][s].tolist(), "ptm": conf["nested"]["ptm"][s].tolist(), "tag": conf["tag"]}, indent=1)
                        Path(f"{prefix}_sample_{s + 1}.json").write_text(text)
                    if self.write_features:
                        Path(f"{prefix}_features.txt").write_text("features")
            def on_predict_end(self, trainer, pl_module):
                SUMMARY.update(total=self.total_count, success=self.success_count, failed=self.failed_count, failed_queries=list(self.failed_queries))
        ''') + "\n" + body + "\n"
    # ^ the engine's method text, re-indented into the stub class (dedent -> 4 spaces: inspect.getsource + dedent gives the digest the kit pins)
    d = tmp_path / "stubpkg"; d.mkdir()
    (d / "nfw_stub_writer.py").write_text(src)
    monkeypatch.syspath_prepend(str(d))
    sys.modules.pop("nfw_stub_writer", None)
    saved = {k: core.__dict__[k] for k in core.CONFIGURABLE}
    core.uninstall()
    core.STATE.update(items=0, overlapped=0, inline={}, host_copy_s=0.0, wait_s=0.0, drain_s=0.0, write_s=0.0, failed=0, worker="-")
    core.configure(ENV="NFW_STUB_WRITER_OVERLAP", ENV_ROUTE="NFW_STUB_WRITER_OVERLAP_ROUTE", M_WRITER="nfw_stub_writer", WRITER_CLASS="OF3OutputWriter",
                   DIGESTS=dict(cell._core.DIGESTS if False else {"on_predict_batch_end": saved["DIGESTS"]["on_predict_batch_end"]}), CHILD_CENSUS=[])
    yield types.SimpleNamespace(tmp=tmp_path, torch=torch)
    core.uninstall()
    sys.modules.pop("nfw_stub_writer", None)
    core.configure(**saved)


def _item(qid, seed=42, n=7, s=3, tag="x"):
    rs = np.random.RandomState(abs(hash(qid)) % 2 ** 31)
    batch = {"atom_array": [[qid, "CA", "CB"]], "seed": [seed], "query_id": [qid], "repeated_sample": False, "gpu_feature": _FakeTensor(rs.rand(1, n, 5))}
    outputs = {"atom_positions_predicted": _FakeTensor(rs.rand(1, s, n, 3).astype(np.float32)),
               "confidence_scores": {"plddt": _FakeTensor(rs.rand(1, s, n).astype(np.float64)), "pae": _FakeTensor(rs.rand(1, s, n, n).astype(np.float32)),
                                     "nested": {"ptm": _FakeTensor(rs.rand(1, s, 2))}, "tag": tag}}
    return batch, outputs


def _predict(engine, route, qids, out, trainer=None, **writer_kw):
    """One predict epoch through the patched callback: install(route) -> import the writer -> items -> on_predict_end; returns (files, SUMMARY, STATE)."""
    environ = {"NFW_STUB_WRITER_OVERLAP": "1", "NFW_STUB_WRITER_OVERLAP_ROUTE": route}
    st = core.install(environ)
    assert st["installed"] and st["route"] == route
    W = importlib.import_module("nfw_stub_writer")
    assert getattr(W.OF3OutputWriter.on_predict_batch_end, core.MARK, False), core.STATE      # patched at the engine's import (digest accepted: the engine's text)
    w = W.OF3OutputWriter(out, **writer_kw)
    trainer = trainer if trainer is not None else types.SimpleNamespace(world_size=1)     # no num_predict_batches: every item overlapped (the loop's length unknown)
    for i, q in enumerate(qids):
        batch, outputs = _item(q)
        w.on_predict_batch_end(trainer, None, (batch, outputs), batch, i)
    w.on_predict_end(trainer, None)
    files = {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    summary, state = dict(W.SUMMARY), json.loads(json.dumps(core.STATE, default=str))
    core.uninstall()
    sys.modules.pop("nfw_stub_writer", None)
    core.STATE.update(items=0, overlapped=0, inline={}, host_copy_s=0.0, wait_s=0.0, drain_s=0.0, write_s=0.0, failed=0, worker="-")
    return files, summary, state


def test_the_kits_digest_pin_is_the_engines_callback_text_in_the_tree():
    text = textwrap.dedent(_engine_method_text())
    import hashlib
    assert hashlib.sha256(text.encode()).hexdigest()[:16] in cell._core.DIGESTS["on_predict_batch_end"] == ("4c1eaf36b439d73d",)
    assert "self.write_all_outputs(" in text and "del batch, outputs" in text and 'batch.get("repeated_sample")' in text   # the statements the port re-states


@pytest.mark.parametrize("route", ["thread", "process"])
def test_every_route_writes_the_inline_routes_files_byte_for_byte_with_the_engines_bookkeeping(engine, route):
    qids = ["q1", "boom", "q2", "q3"]
    ref_files, ref_sum, ref_state = _predict(engine, "sync", qids, engine.tmp / "ref")
    files, summ, state = _predict(engine, route, qids, engine.tmp / route)
    assert files and files == ref_files                                                    # same names, same bytes (3 items x 3 samples; the failed item wrote nothing in both)
    assert sorted(files) == sorted(f"{q}/seed_42/{q}_seed_42_sample_{s}.json" for q in ("q1", "q2", "q3") for s in (1, 2, 3))
    assert ref_sum == summ == {"total": 4, "success": 3, "failed": 1, "failed_queries": ["boom"]}          # on_predict_end saw every item drained and tallied, the engine's way
    assert ref_state["inline"] == {"route_sync": 4} and ref_state["overlapped"] == 0 and ref_state["items"] == 4
    assert state["inline"] == {} and state["overlapped"] == 4 and state["items"] == 4 and state["failed"] == 1 and state["route"] == route
    assert (state["worker"] == "thread") == (route == "thread") and state["write_s"] > 0 and state["state"] == "on"
    assert (engine.tmp / route / "boom" / "seed_42").is_dir()                               # the writer's mkdir made on the main thread (PredictTimer writes timing.json there next)


def test_features_latent_and_a_distributed_predict_are_written_inline_by_name(engine):
    files, summ, state = _predict(engine, "process", ["q1", "q2"], engine.tmp / "feat", write_features=True)
    assert state["inline"] == {"features_or_latent": 2} and state["overlapped"] == 0 and summ["success"] == 2
    assert any(k.endswith("_features.txt") for k in files)
    environ = {"NFW_STUB_WRITER_OVERLAP": "1", "NFW_STUB_WRITER_OVERLAP_ROUTE": "thread"}
    core.install(environ)
    W = importlib.import_module("nfw_stub_writer")
    w = W.OF3OutputWriter(engine.tmp / "dist")
    batch, outputs = _item("q9")
    w.on_predict_batch_end(types.SimpleNamespace(world_size=2), None, (batch, outputs), batch, 0)
    w.on_predict_end(None, None)
    assert core.STATE["inline"] == {"world_size": 1} and W.SUMMARY["success"] == 1


def test_a_failed_forward_and_a_repeated_item_are_the_engines_bookkeeping_verbatim(engine):
    environ = {"NFW_STUB_WRITER_OVERLAP": "1", "NFW_STUB_WRITER_OVERLAP_ROUTE": "thread"}
    core.install(environ)
    W = importlib.import_module("nfw_stub_writer")
    w = W.OF3OutputWriter(engine.tmp / "bk")
    batch, outputs = _item("q1")
    w.on_predict_batch_end(None, None, None, batch, 0)                                      # outputs None: the forward failed -> failed bookkeeping, nothing queued
    rep = dict(batch, repeated_sample=True)
    w.on_predict_batch_end(None, None, (rep, outputs), rep, 1)                              # repeated: skipped entirely
    w.on_predict_end(None, None)
    assert W.SUMMARY == {"total": 1, "success": 0, "failed": 1, "failed_queries": ["q1"]} and core.STATE["items"] == 0


def test_switch_words_refusals_and_the_census_line(engine):
    assert core.requested({}) is False and core.requested({"NFW_STUB_WRITER_OVERLAP": "1"}) is True
    with pytest.raises(ValueError):
        core.requested({"NFW_STUB_WRITER_OVERLAP": "yes"})
    with pytest.raises(ValueError):
        core.route({"NFW_STUB_WRITER_OVERLAP_ROUTE": "async"})
    assert core.route({}) == "process" and core.ROUTES == ("process", "thread", "sync")
    st = core.install({"NFW_STUB_WRITER_OVERLAP": "1", "NFW_STUB_WRITER_OVERLAP_ROUTE": "bogus"})
    assert st["state"] == "refused" and st["reason"].startswith("bad_word:") and core.FINDER not in sys.meta_path      # a word the cell does not know: steps aside by name, the engine untouched
    line = core.census_line()
    for word in ("LEVER name=writer_overlap", "state=refused", "route=", "items=0", "overlapped=0", "inline=none", "host_copy_s=", "wait_s=", "drain_s=", "write_s=", "write_exc=0", "worker="):
        assert word in line, (word, line)
    core.uninstall()
    core.configure(DIGESTS={"on_predict_batch_end": ("0000000000000000",)})                  # a writer text the port does not re-state: refused by name at the engine's import
    core.install({"NFW_STUB_WRITER_OVERLAP": "1"})
    W = importlib.import_module("nfw_stub_writer")
    assert core.STATE["state"] == "refused" and core.STATE["reason"].startswith("digest:on_predict_batch_end=") and not getattr(W.OF3OutputWriter.on_predict_batch_end, core.MARK, False)


def test_every_kit_line_but_off_and_tp_exports_the_switch_and_the_kit_names_the_lever_everywhere():
    for key, ln in modes.LINES.items():
        carries = key[0] != "off" and key != ("big", "tp")
        assert (ln.env.get(cell.ENV) == "1") is carries and ("writer_overlap" in ln.levers) is carries and cell.ENV_ROUTE not in ln.env, key
    lv = registry.LEVERS["writer_overlap"]
    assert (lv.kit, lv.cls, lv.tier, lv.env_keys, lv.target) == ("cells", registry.OUTPUT, registry.EXACT, modes.WRITER_OVERLAP_ENVS, registry.T_WRITER)
    assert set(lv.modes) == {"exact", "fast", "big/resident"} and lv.marker == "[openfold3-opt/writer_overlap] installed" and lv.source == "opt/openfold3_opt/cells/writer_overlap.py"
    assert registry.tested_sm("writer_overlap") == ("sm90",) and "writer_overlap" in registry.ARCH_NOTES
    assert modes.WRITER_OVERLAP_ENVS == (cell.ENV, cell.ENV_ROUTE) and modes.WRITER_OVERLAP_ENVS in modes.CELL_FAMILIES and modes.WRITER_OVERLAP_ENVS in modes.ACTIVATION_FAMILIES
    assert "writer_overlap" in modes.ACTIVATION_CELLS and "writer_overlap" in stack._PROBES and modes.LEVER_SWITCHES["writer_overlap"] == modes.WRITER_OVERLAP_ENVS
    assert cell.ENV in _autoload.DECLARED and cell.ENV_ROUTE in _autoload.DECLARED
    assert cell._core.M_WRITER == registry.T_WRITER and cell._core.PREFIX == "[openfold3-opt/writer_overlap]" and [c[0] for c in cell._core.CHILD_CENSUS] == ["fastjson"]
    r = modes.resolve("exact", HOME, environ={modes.ENV_LEVERS_OFF: "writer_overlap"}, n_tokens=400)
    assert "writer_overlap" not in r.levers and cell.ENV not in r.exports and "fastjson" in r.levers            # ablated alone, by name, the kit's way


def test_fastjsons_worker_side_counters_are_merged_back_into_its_record():
    from openfold3_opt.cells import fastjson
    snap, merge = cell._fastjson_snapshot, cell._fastjson_merge
    before = snap()
    fastjson.STATE["served"] += 3; fastjson.STATE["bytes"] += 100; fastjson.STATE["fallback_by"]["X"] = fastjson.STATE["fallback_by"].get("X", 0) + 2
    delta = core._delta(before, snap())
    fastjson.STATE["served"] -= 3; fastjson.STATE["bytes"] -= 100; fastjson.STATE["fallback_by"]["X"] -= 2
    ref = snap()
    merge(delta)
    assert fastjson.STATE["served"] == ref["served"] + 3 and fastjson.STATE["bytes"] == ref["bytes"] + 100 and fastjson.STATE["fallback_by"]["X"] == ref["fallback_by"]["X"] + 2
    merge({k: (-v if not isinstance(v, dict) else {kk: -vv for kk, vv in v.items()}) for k, v in delta.items()})   # restore


def test_the_finder_composes_with_another_finder_for_the_writer_module_instead_of_recursing(engine):
    """fastjson arms its own finder for the same module and delegates the find over sys.meta_path without a guard: with both armed the import
    must resolve once, both loaders' hooks run, no RecursionError (seen as: RecursionError, fastjson refused)."""
    seen = []

    class _OtherWriterFinder:                                                    # fastjson's delegation text (cells/of3_fastjson.py _WriterFinder.find_spec), hooking a marker
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "nfw_stub_writer":
                return None
            spec = None
            for finder in sys.meta_path:
                if finder is self or type(finder).__name__ == type(self).__name__:
                    continue
                try:
                    spec = finder.find_spec(fullname, path, target)
                except Exception:  # noqa: BLE001
                    spec = None
                if spec is not None:
                    break
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module
            def exec_module(module, _orig=orig):
                _orig(module); seen.append("other"); module.OTHER_PATCHED = True
            spec.loader.exec_module = exec_module
            return spec

    other = _OtherWriterFinder()
    for order in ("mine_first", "other_first"):
        sys.modules.pop("nfw_stub_writer", None); core.uninstall(); seen.clear()
        sys.meta_path.insert(0, other)
        try:
            core.install({"NFW_STUB_WRITER_OVERLAP": "1"})                         # inserts FINDER at 0 (mine first) ...
            if order == "other_first":
                sys.meta_path.remove(other); sys.meta_path.insert(0, other)        # ... or the other one ahead of it
            W = importlib.import_module("nfw_stub_writer")
            assert getattr(W.OF3OutputWriter.on_predict_batch_end, core.MARK, False) and W.OTHER_PATCHED is True and seen and core.STATE["state"] == "on", order
        finally:
            if other in sys.meta_path:
                sys.meta_path.remove(other)
            core.uninstall(); sys.modules.pop("nfw_stub_writer", None)


def test_host_copies_cast_exactly_the_leaves_the_writer_casts(engine):
    f64 = core.to_host(_FakeTensor(np.zeros((2, 3), dtype=np.float64)))          # rank >= 2: the writer's .cpu().float() -> float32
    i1 = core.to_host(_FakeTensor(np.arange(4, dtype=np.int64)))                  # rank 1: handed on as it is -> dtype kept
    assert f64.a.dtype == np.float32 and i1.a.dtype == np.int64
    nested = core.to_host({"a": [_FakeTensor(np.ones((1, 2), dtype=np.float16))], "s": "text", "n": 3})
    assert nested["a"][0].a.dtype == np.float32 and nested["s"] == "text" and nested["n"] == 3


def test_the_predict_end_retires_the_worker_and_a_child_deaf_to_the_sentinel_is_still_ended(engine, monkeypatch):
    """The hang this prevents: multiprocessing's exit handler terminates daemonic children and joins them; a fork that inherited Lightning's
    SIGTERM notifier never left. Now: the worker is closed at on_predict_end, the child takes SIGTERM's default disposition, and close() is
    bounded (sentinel -> SIGTERM -> SIGKILL)."""
    import signal
    got = []
    prev = signal.signal(signal.SIGTERM, lambda *a: got.append(a))                        # the predicting process's own SIGTERM handler (Lightning's does not exit)
    try:
        environ = {"NFW_STUB_WRITER_OVERLAP": "1"}
        core.install(environ)
        W = importlib.import_module("nfw_stub_writer")
        w = W.OF3OutputWriter(engine.tmp / "life")
        batch, outputs = _item("q1")
        w.on_predict_batch_end(None, None, (batch, outputs), batch, 0)
        worker = core._WORKER["obj"]
        assert worker is not None and worker.kind == "process" and worker.p.is_alive()
        w.on_predict_end(None, None)                                                            # drained, written, and the worker retired here
        assert W.SUMMARY["success"] == 1 and core._WORKER["obj"] is None and not worker.p.is_alive() and worker.p.exitcode == 0
        # a second predict in the process: a new worker; make it deaf to the sentinel -> close() ends it by signal, bounded
        w2 = W.OF3OutputWriter(engine.tmp / "life2")
        b2, o2 = _item("q2")
        w2.on_predict_batch_end(None, None, (b2, o2), b2, 0)
        worker2 = core._WORKER["obj"]; core.drain(w2)
        monkeypatch.setattr(core, "CLOSE_JOIN_S", 0.5); monkeypatch.setattr(core, "TERM_JOIN_S", 5.0)
        monkeypatch.setattr(worker2.jobq, "put", lambda *_a, **_k: (_ for _ in ()).throw(BrokenPipeError("no sentinel")))
        t = __import__("time").perf_counter()
        core._retire()
        assert not worker2.p.is_alive() and worker2.p.exitcode == -signal.SIGTERM and __import__("time").perf_counter() - t < 5.0 and got == []
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_the_last_item_of_the_loop_is_written_inline_earlier_items_overlapped(engine):
    tr = types.SimpleNamespace(world_size=1, num_predict_batches=[3])
    ref_files, ref_sum, _ = _predict(engine, "sync", ["q1", "q2", "q3"], engine.tmp / "last_ref")
    files, summ, state = _predict(engine, "process", ["q1", "q2", "q3"], engine.tmp / "last", trainer=tr)
    assert files == ref_files and summ == ref_sum == {"total": 3, "success": 3, "failed": 0, "failed_queries": []}
    assert state["inline"] == {"last_item": 1} and state["overlapped"] == 2 and state["items"] == 3
    assert core._is_last_item(tr, 2) and not core._is_last_item(tr, 1) and not core._is_last_item(None, 5)
    tr.num_predict_batches = [float("inf")]
    assert not core._is_last_item(tr, 10)
