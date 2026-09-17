"""The route driver's pure parts: FASTA reading, the arms' constructions, the mode / environment agreement (usage exit 2 before any model loads)."""
import importlib.util
import os
import tempfile
import unittest
from unittest import mock

from evo2_opt import pins

spec = importlib.util.spec_from_file_location("evo2_route", os.path.join(pins.tree_root(), pins.ROUTE_RELPATH))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


class TestDriver(unittest.TestCase):
    def test_read_fasta(self):
        with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fh:
            fh.write(">w0 first window\nacgt\nAC\n\n>w1\nGGGG\n"); p = fh.name
        self.assertEqual(R.read_fasta(p), [("w0", "ACGTAC"), ("w1", "GGGG")]); os.remove(p)

    def test_constructions(self):
        self.assertEqual(R.construction("off", "evo2_7b"), {"use_kernels": True})
        self.assertEqual(R.construction("exact", "evo2_7b"), {"use_kernels": True})
        self.assertEqual(R.MODES, ("exact", "fast", "off"))                # the score vocabulary: no other mode word
        self.assertEqual(R.construction("off", "evo2_40b"), {})              # use_kernels=True scores NaN over the two-device split
        self.assertEqual(R.construction("exact", "evo2_40b"), {})

    def test_score_writes_scores_and_run_record(self):
        """The score verb on a fake `evo2` package (CPU): one score_sequences call, scores.jsonl one row per record, run.json the run's record."""
        import json, sys, types
        fake = types.ModuleType("evo2")
        class Evo2:
            def __init__(self, name, local_path=None, **kw): self.kw = kw
            def score_sequences(self, seqs, batch_size=1): return [-(i + 1) / 8.0 for i in range(len(seqs))]
        fake.Evo2 = Evo2
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules, {"evo2": fake}), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EVO2_OPT", None)
            fa = os.path.join(d, "w.fa"); open(fa, "w").write(">a\nACGT\n>b\nGGCC\n")
            with mock.patch.object(R, "env_clean_line", return_value=("[evo2-route stock] ENV-CLEAN ok: test", None)):
                self.assertEqual(R.main(["--input", fa, "--out_dir", d, "--mode", "off", "--batch_size", "2"]), 0)
            rows = [json.loads(l) for l in open(os.path.join(d, "scores.jsonl"))]
            self.assertEqual([(r["id"], r["index"], r["length"]) for r in rows], [("a", 0, 4), ("b", 1, 4)])
            rec = json.load(open(os.path.join(d, "run.json")))
            self.assertEqual(rec, {"model": "evo2_7b", "mode": "off", "batch_size": 2, "n_sequences": 2, "n_scored": 2, "status": "ok"})

    def test_mode_environment_agreement_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as d:
            fa = os.path.join(d, "w.fa"); open(fa, "w").write(">w\nACGT\n")
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch("sys.stderr"):
                self.assertEqual(R.main(["--input", fa, "--out_dir", d, "--mode", "exact"]), 2)     # exact without EVO2_OPT=exact
            with mock.patch.dict(os.environ, {"EVO2_OPT": "exact"}, clear=True), mock.patch("sys.stderr"):
                self.assertEqual(R.main(["--input", fa, "--out_dir", d, "--mode", "off"]), 2)       # a stock arm under the kit's switch


if __name__ == "__main__":
    unittest.main()
