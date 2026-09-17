"""Legal requests that used to refuse now print ONE named NOTE line and proceed, decided up front (no fallback, no silent switch):
the batched configure() branch's t11-off-at-B>1 words, the user's x4 threshold (EF2_X4_MIN_TOKENS); the multi-GPU line's
one-item-per-fold-call domain is refused BY NAME with the knob named. CPU only."""
import io
import os
import unittest
from contextlib import redirect_stderr
from unittest import mock

from esmfold2_opt import big, modes, stack

HERE = os.path.dirname(os.path.abspath(__file__))
FWD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "forward")


class TestX4ThresholdKnob(unittest.TestCase):
    def setUp(self):
        big.X4_NOTES["threshold_set"] = 0
        self.knobs = {"esmc_offload": True, "esmc_min_tok": modes.X4_MIN_TOKENS, "own": False}

    def test_default_is_the_lines_threshold_and_prints_nothing(self):
        err = io.StringIO()
        with redirect_stderr(err):
            out = big.x4_threshold(dict(self.knobs), environ={})
        self.assertEqual(out["esmc_min_tok"], modes.X4_MIN_TOKENS); self.assertEqual(modes.X4_MIN_TOKENS, 1500); self.assertEqual(err.getvalue(), ""); self.assertEqual(big.X4_NOTES["threshold_set"], 0)

    def test_env_sets_the_threshold_with_one_note(self):
        err = io.StringIO()
        with redirect_stderr(err):
            out = big.x4_threshold(dict(self.knobs), environ={big.ENV_X4_MIN_TOKENS: "900"})
        self.assertEqual(out["esmc_min_tok"], 900); self.assertEqual(big.X4_NOTES["threshold_set"], 1)
        self.assertIn("[esmfold2-opt] NOTE x4 threshold EF2_X4_MIN_TOKENS=900 (line default 1500): the ESMC-6B host offload engages at inputs >= 900 tokens", err.getvalue())
        with redirect_stderr(io.StringIO()):
            self.assertEqual(big.x4_threshold(dict(self.knobs), environ={big.ENV_X4_MIN_TOKENS: "0"})["esmc_min_tok"], 0)      # 0 = engage at any size
        off = dict(self.knobs, esmc_offload=False)
        self.assertEqual(big.x4_threshold(off, environ={big.ENV_X4_MIN_TOKENS: "900"}), off)                            # a line without x4: the knob has nothing to tune, untouched

    def test_a_malformed_value_is_refused_by_name(self):
        for bad in ("-5", "lots", "1.5"):
            with self.assertRaises(stack.ActivationError) as cm:
                big.x4_threshold(dict(self.knobs), environ={big.ENV_X4_MIN_TOKENS: bad})
            self.assertIn("EF2_X4_MIN_TOKENS=", str(cm.exception)); self.assertIn("non-negative integer", str(cm.exception))

    def test_census_reads_the_threshold_in_force(self):
        c = big.x4_census({"esmc_offload": True, "esmc_min_tok": 900}, {"esmc_offload_events": 1}, tokens=[950])
        self.assertEqual((c["state"], c["threshold"], c["expected"]), ("engaged", 900, 1))
        c = big.x4_census({"esmc_offload": True, "esmc_min_tok": 1500}, {"esmc_offload_events": 0}, tokens=[950])
        self.assertEqual((c["state"], c["threshold"]), ("off-by-size", 1500))


class TestMultiGpuDomainRefusalsNameTheKnob(unittest.TestCase):
    def test_one_item_per_fold_call_words(self):
        src = open(os.path.join(os.path.dirname(HERE), "rowpair.py")).read() + open(os.path.join(os.path.dirname(HERE), "rowpair_heads.py")).read()
        self.assertIn("one item per fold call is the row-sharded line's domain", src); self.assertIn("or use --n_gpu 1", src); self.assertIn("fold the items separately", src)


if __name__ == "__main__":
    unittest.main()
