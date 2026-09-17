"""Lever awrite (awrite.py + sidecar.py): an item's writers run in one forked side process on the item's own objects, flushed before
run() returns; a writer failure re-runs the stock statement in the fold process. CPU: a stand-in engine module with upstream's writer
names and call order (dump_ranking_scores, dump_top_ranked_outputs, RF3Output.dump per sample); byte identity of the written tree
against a stock run is the acceptance."""
import json
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from .test_prefetch import _REFUSED  # noqa: E402

pytestmark = pytest.mark.skipif(_REFUSED is not None, reason=f"platform refuses fork/pipe side processes: {_REFUSED}")

from .. import awrite as aw
from .. import sidecar as sc

PARENT = os.getpid()


FAIL = {"in_writer": None}                                   # a path fragment whose CIF write fails in the side process only


def dump_json_compact_arrays(obj, f):
    f.write(json.dumps(obj, indent=2))


def to_cif_file(atom_array, path, file_type="cif", include_entity_poly=False):
    if FAIL["in_writer"] and FAIL["in_writer"] in str(path) and os.getpid() != PARENT:
        raise OSError(f"simulated writer failure on {path}")
    with open(f"{path}.{file_type}", "w") as f:
        f.write("data_x\n" + "\n".join(" ".join(f"{v:.3f}" for v in row) for row in np.asarray(atom_array)))


@dataclass
class RF3Output:                                             # module level: the writer's objects must pickle (upstream's dataclass does)
    example_id: str
    atom_array: object
    summary_confidences: dict = field(default_factory=dict)
    confidences: dict | None = None
    sample_idx: int = 0
    seed: int = 0

    def dump(self, out_dir, file_type="cif", dump_full_confidences=True):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / f"{self.example_id}_seed-{self.seed}_sample-{self.sample_idx}"
        to_cif_file(self.atom_array, Path(f"{base}_model"), file_type=file_type)
        with open(f"{base}_summary_confidences.json", "w") as f:
            dump_json_compact_arrays(self.summary_confidences, f)
        if dump_full_confidences and self.confidences:
            with open(f"{base}_confidences.json", "w") as f:
                dump_json_compact_arrays(self.confidences, f)


STOCK_DUMP = RF3Output.dump


def dump_ranking_scores(outputs, out_dir, example_id):
    with open(Path(out_dir) / f"{example_id}_ranking_scores.csv", "w") as f:
        f.write("seed,sample,ranking_score\n" + "".join(f"{o.seed},{o.sample_idx},{o.summary_confidences.get('ranking_score')}\n" for o in outputs))


def dump_top_ranked_outputs(outputs, out_dir, example_id, file_type="cif"):
    best = max(outputs, key=lambda o: o.summary_confidences.get("ranking_score", float("-inf")))
    to_cif_file(best.atom_array, Path(out_dir) / f"{example_id}_model", file_type=file_type)
    with open(Path(out_dir) / f"{example_id}_summary_confidences.json", "w") as f:
        dump_json_compact_arrays(best.summary_confidences, f)
    if best.confidences:
        with open(Path(out_dir) / f"{example_id}_confidences.json", "w") as f:
            dump_json_compact_arrays(best.confidences, f)
    return best


def make_engine_module(fail_in_writer=None):
    """A stand-in rf3.inference_engines.rf3 with upstream's writer names and run()'s writer block (stock names looked up on the module)."""
    FAIL["in_writer"] = fail_in_writer
    RF3Output.dump = STOCK_DUMP
    m = types.ModuleType("rf3.inference_engines.rf3")

    class RF3InferenceEngine:
        def __init__(self):
            self.pipeline = None

        def _construct_pipeline(self, cfg=None):
            self.pipeline = object()

        def run(self, items, out_dir, n_samples=3):
            if self.pipeline is None:
                self._construct_pipeline({})
            rng = np.random.RandomState(0)
            for ex in items:
                example_out_dir = Path(out_dir) / ex
                example_out_dir.mkdir(parents=True, exist_ok=True)
                outs = [m.RF3Output(example_id=ex, atom_array=rng.rand(5, 3).astype(np.float32),
                                     summary_confidences={"ranking_score": round(float(rng.rand()), 4), "chain_pair_pae": [[None, 1.5], [2.25, None]]},
                                     confidences={"pae": rng.rand(4, 4).round(2).tolist(), "atom_plddts": [0.5, 0.75]}, sample_idx=i, seed=42)
                        for i in range(n_samples)]
                m.dump_ranking_scores(outs, example_out_dir, ex)
                m.dump_top_ranked_outputs(outs, example_out_dir, ex, file_type="cif")
                for o in outs:
                    o.dump(out_dir=example_out_dir / f"seed-{o.seed}_sample-{o.sample_idx}", file_type="cif", dump_full_confidences=True)
            return None

    m.dump_json_compact_arrays, m.to_cif_file = dump_json_compact_arrays, to_cif_file
    m.RF3Output, m.dump_ranking_scores, m.dump_top_ranked_outputs, m.RF3InferenceEngine = RF3Output, dump_ranking_scores, dump_top_ranked_outputs, RF3InferenceEngine
    return m


def tree_bytes(root):
    out = {}
    for d, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(d, fn)
            out[os.path.relpath(p, root)] = open(p, "rb").read()
    return out


def reset_lever():
    aw.STATE.update({"armed": False, "installed": False, "on": False, "reason": None, "off_reason": None, "conflict": None, "n_gpu": 1, "writer_pid": None,
                     "forked_before_cuda": None})
    aw.CENSUS.clear(); aw.ITEMS.clear()
    if aw._W["car"] is not None:
        aw._W["car"].stop()
    aw._W.update({"car": None, "pending": {}, "item": None, "outs": None, "n_call": 0})
    aw._ORIG.clear()


@pytest.fixture()
def lever():
    reset_lever()
    yield aw
    m = sys.modules.get("rf3.inference_engines.rf3")
    if m is not None and "dump" in aw._ORIG:
        aw.disable(m)
    reset_lever()
    sc.stop_all()


def test_written_tree_is_byte_identical_and_flushed_before_run_returns(lever, monkeypatch, tmp_path):
    items = ["ex_a", "ex_b", "ex_c"]
    make_engine_module().RF3InferenceEngine().run(items, tmp_path / "stock")
    m = make_engine_module()
    monkeypatch.setitem(sys.modules, m.__name__, m)
    stock = tree_bytes(tmp_path / "stock")
    assert len(stock) == 3 * (1 + 3 + 3 * 3)                            # csv + top (cif, 2 json) + per sample (cif, 2 json)
    rep = {"levers": ["prefetch", "awrite"], "n_gpu": 1, "levers_applied": []}
    lever.arm(rep, lambda *a: pytest.fail("no watch expected"))
    assert rep["awrite"]["installed"] and "awrite" in rep["levers_applied"] and m.RF3InferenceEngine.run.__name__ == "run_awrite"
    m.RF3InferenceEngine().run(items, tmp_path / "kit")                 # returns after AWRITE flush: every file on disk
    got = tree_bytes(tmp_path / "kit")
    assert got == stock
    c = aw.census()
    assert aw.STATE["on"] and aw.STATE["writer_pid"] and aw.STATE["forked_before_cuda"] is True
    assert c["items"] == 3 and c["calls"] == 3 * 5 and c["calls_done"] == 15 and c["files_done"] == 3 * 13 and c["errors"] == 0 and c["files_here"] == 0
    toks = dict(aw.lever_tokens())
    assert toks["files"] == 39 and toks["here"] == 0 and toks["errors"] == 0 and toks["nonfinite"] == 0


def test_a_writer_failure_reruns_the_stock_statement_here(lever, monkeypatch, tmp_path):
    make_engine_module().RF3InferenceEngine().run(["ex_a", "ex_b"], tmp_path / "stock")
    m = make_engine_module(fail_in_writer="ex_b_seed-42_sample-1_model")   # sample 1 of ex_b fails in the side process only
    monkeypatch.setitem(sys.modules, m.__name__, m)
    lever.arm({"levers": ["awrite"], "n_gpu": 1}, lambda *a: None)
    m.RF3InferenceEngine().run(["ex_a", "ex_b"], tmp_path / "kit")
    assert tree_bytes(tmp_path / "kit") == tree_bytes(tmp_path / "stock")
    c = aw.census()
    assert c["errors"] == 1 and c["calls_here"] == 1 and c["files_here"] == 3 and c["files_done"] == 2 * 13 - 3


def test_named_step_asides_and_finite_word(lever, monkeypatch, tmp_path):
    make_engine_module().RF3InferenceEngine().run(["e1"], tmp_path / "stock")
    m = make_engine_module()
    monkeypatch.setitem(sys.modules, m.__name__, m)
    rep = aw.arm({"levers": ["awrite"], "n_gpu": 2}, lambda *a: None)    # the row-sharded line: declined by name at activation (conflict:n_gpu) — no writer process, the engine untouched
    m.RF3InferenceEngine().run(["e1"], tmp_path / "kit")
    assert tree_bytes(tmp_path / "kit") == tree_bytes(tmp_path / "stock")
    assert aw.STATE["writer_pid"] is None and rep["awrite"]["conflict"] == "n_gpu" and aw.STATE["conflict"] == "n_gpu" and aw.census()["here"] == {} and not aw.STATE["installed"]
    O = types.SimpleNamespace
    ok = [O(atom_array=O(coord=np.zeros((2, 3))), summary_confidences={"a": 1.0, "m": [[None, 2.0]]}, sample_idx=0)]
    assert aw.finite_word(ok) == "ok"
    bad = [O(atom_array=O(coord=np.array([[0.0, np.nan, 1.0]])), summary_confidences={}, sample_idx=3)]
    assert aw.finite_word(bad) == "NONFINITE:coord:sample3"
    inf = [O(atom_array=O(coord=np.zeros((1, 3))), summary_confidences={"overall_pae": float("inf")}, sample_idx=1)]
    assert aw.finite_word(inf) == "NONFINITE:overall_pae:sample1"
    assert not aw.arm({"levers": ["graph"], "n_gpu": 1}, lambda *a: None).get("awrite")   # a row without the lever: untouched


def test_registry_words():
    d = aw.describe()
    assert {"on", "installed", "reason", "off_reason", "writer_pid", "forked_before_cuda", "census", "items", "rule"} <= set(d)
    assert [k for k, _ in aw.lever_tokens(d)] == ["files", "here", "errors", "nonfinite", "flush_wait_ms"]

