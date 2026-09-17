"""mem.rowpair.structure_first (K.T24a, CPU): the early write goes through the ENGINE's static writer entry (passed in) with a 0.00 pLDDT column to
the path that writer will overwrite; sidecars say 'confidence not written' until mark_final sees an actual rewrite (outputs present AND sha256
changed); flips are keyed to the batch's query/seed; batch size 1 is asserted by name; the off switch and a missing writer are named; the census
words ride evidence.record_schedule; scan() yields the run record's block, the structure-only paths and the note."""
import json, os, types
import numpy as np
import pytest

from opt_core.mem.rowpair import structure_first as SF
from opt_core.mem.rowpair import evidence as EV


class FakeWriter:
    """Stands in for the engine's output-writer callback: `output_dir`, `structure_format`, and the static writer, which records (coords, plddt)
    and writes bytes that depend on the coordinates AND the pLDDT column, like the real structure file."""
    calls = []

    def __init__(self, output_dir):
        self.output_dir, self.structure_format = output_dir, "cif"

    @staticmethod
    def write_structure_prediction(atom_array, predicted_coords, plddt, output_file, make_ost_compatible=True):
        FakeWriter.calls.append((np.asarray(predicted_coords).copy(), np.asarray(plddt).copy(), str(output_file)))
        with open(output_file, "w") as fh:
            fh.write("coords %s\nbfac %s\n" % (np.asarray(predicted_coords).round(3).tolist(), np.asarray(plddt).round(2).tolist()))


class X:                                                # x[0].detach().float().cpu().numpy() -> [k, N_atom, 3]
    def __init__(self, arr): self.arr = arr
    def __getitem__(self, i): return X(self.arr[i])
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.arr


IS_WRITER = lambda cb: isinstance(cb, FakeWriter)
WRITE = FakeWriter.write_structure_prediction


@pytest.fixture()
def engine(tmp_path):
    FakeWriter.calls = []
    SF.reset(); EV.reset_schedule()
    writer = FakeWriter(str(tmp_path / "out"))
    pl = types.SimpleNamespace(trainer=types.SimpleNamespace(callbacks=[object(), writer]))
    batch = {"query_id": ["q1"], "seed": [7], "atom_array": [types.SimpleNamespace(n=5)]}
    SF.capture(pl, batch, IS_WRITER)
    return writer, batch, pl


def side_of(path):
    return json.load(open(path.rsplit("_model.", 1)[0] + SF.SIDE_SUFFIX))


def test_capture_finds_the_engine_writer_and_batch(engine):
    writer, batch, _pl = engine
    assert SF.STATE["writer"] is writer and SF.STATE["batch"] is batch


def test_write_early_then_mark_final(engine):
    writer, batch, _pl = engine
    x = X(np.arange(2 * 5 * 3, dtype=np.float32).reshape(1, 2, 5, 3))
    lines = []
    paths = SF.write_early(batch, x, 3, WRITE, log=lines.append)        # samples 4 and 5 (1-based names) of a pass starting at sample index 3
    assert [os.path.basename(p) for p in paths] == ["q1_seed_7_sample_4_model.cif", "q1_seed_7_sample_5_model.cif"]
    for p in paths:
        assert os.path.isfile(p)
        side = side_of(p)
        assert side["confidence_written"] is False and side["note"].startswith("confidence not written") and side["structure_first_sha256"]
    (c0, b0, _f0), (c1, b1, _f1) = FakeWriter.calls                      # the static writer got the pass's coordinates and an all-zero pLDDT column
    assert np.array_equal(c0, x.arr[0, 0]) and np.array_equal(c1, x.arr[0, 1]) and not b0.any() and b0.shape == (5,)
    assert lines and lines[0].startswith("[structure_first] ") and "wrote 2 structure(s) BEFORE the confidence heads" in lines[0] and "census structure_first=on" in lines[0]
    sched = EV.schedule()                                                 # the census words are recorded for real (the rank record's tp_schedule)
    assert sched["structure_first"] == "on" and sched["structure_first_files"] == 2 and "structure_first_write_s" in sched
    blk = SF.scan(writer.output_dir)
    assert blk["files"] == 2 and len(blk["confidence_not_written"]) == 2 and len(blk["structure_only_paths"]) == 2
    assert sorted(os.path.basename(p) for p in blk["structure_only_paths"]) == sorted(os.path.basename(p) for p in paths)
    assert blk["notes"][0].startswith("confidence not written: 2 structure-only model file(s)")
    # a prediction that died inside the heads: the engine's hook still runs with outputs=None -> nothing flips, the note stays, the census says 2
    assert SF.mark_final(batch, outputs=None, log=lines.append) == 0
    assert EV.schedule()["structure_first_not_written"] == 2 and any("confidence NOT written" in l for l in lines)
    # unchanged bytes with outputs present (the writer skipped the file): still not flipped
    assert SF.mark_final(batch, outputs=("batch", "outputs"), log=lines.append) == 0
    for (c, _b, f) in FakeWriter.calls[:2]:                              # the engine's writer runs after the heads (the fake, with a real pLDDT column)
        FakeWriter.write_structure_prediction(None, c, np.full((5,), 87.5), f)
    assert SF.mark_final(batch, outputs=("batch", "outputs"), log=lines.append) == 2
    blk = SF.scan(writer.output_dir)
    assert blk["files"] == 2 and blk["confidence_not_written"] == [] and blk["structure_only_paths"] == [] and blk["notes"] == []
    side = side_of(paths[0])
    assert side["confidence_written"] is True and side["final_sha256"] != side["structure_first_sha256"]
    assert EV.schedule()["structure_first_not_written"] == 0
    assert SF.mark_final(batch, outputs=("batch", "outputs"), log=lines.append) == 0     # a later batch end does not re-flip / re-count final sidecars


def test_flips_are_keyed_to_the_batch_query_and_seed(engine):
    writer, batch, pl = engine
    x = X(np.zeros((1, 1, 5, 3), dtype=np.float32))
    p1 = SF.write_early(batch, x, 0, WRITE)
    batch2 = {"query_id": ["q2"], "seed": [7], "atom_array": [types.SimpleNamespace(n=5)]}
    SF.capture(pl, batch2, IS_WRITER)
    p2 = SF.write_early(batch2, x, 0, WRITE)
    FakeWriter.write_structure_prediction(None, np.zeros((5, 3)), np.full((5,), 50.0), p2[0])    # only q2 was rewritten by the engine
    assert SF.mark_final(batch2, outputs=("b", "o")) == 1
    assert side_of(p2[0])["confidence_written"] is True and side_of(p1[0])["confidence_written"] is False   # q1's pending sidecar untouched by q2's batch end
    assert EV.schedule()["structure_first_not_written"] == 1


def test_batch_size_is_asserted_by_name(engine):
    _writer, _batch, pl = engine
    batch = {"query_id": ["a", "b"], "seed": [1, 1], "atom_array": [object(), object()]}
    SF.capture(pl, batch, IS_WRITER)
    lines = []
    assert SF.write_early(batch, X(np.zeros((2, 1, 5, 3), dtype=np.float32)), 0, WRITE, log=lines.append) == []
    assert SF.words()["structure_first"] == "skipped:batch_size_2" and "batch size 2 != 1" in lines[0] and FakeWriter.calls == []


def test_off_switch_and_missing_writer_are_named(engine):
    writer, batch, _pl = engine
    x = X(np.zeros((1, 1, 5, 3), dtype=np.float32))
    lines = []
    assert SF.write_early(batch, x, 0, WRITE, log=lines.append, enabled=False, switch="OF3TP_STRUCTURE_FIRST") == [] and SF.words()["structure_first"] == "off"
    assert "OF3TP_STRUCTURE_FIRST=0" in lines[0] and EV.schedule()["structure_first"] == "off"
    SF.STATE["writer"] = None; SF.STATE["batch"] = None
    assert SF.write_early({"query_id": ["q"]}, x, 0, WRITE, log=lines.append) == [] and SF.words()["structure_first"].startswith("skipped:")


def test_slow_write_is_named(engine, monkeypatch):
    writer, batch, _pl = engine
    monkeypatch.setattr(SF, "WRITE_WARN_S", -1.0)
    lines = []
    SF.write_early(batch, X(np.zeros((1, 1, 5, 3), dtype=np.float32)), 0, WRITE, log=lines.append)
    assert "SLOW" in lines[0]
