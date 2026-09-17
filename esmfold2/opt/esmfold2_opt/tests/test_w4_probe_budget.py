"""K.21 R2, behaviourally: the W4 first-call probe runs below the graph budget and is skipped BY NAME above it, deciding on the pair's
extent L that the caller passes (T9: pair.shape[1]; T10: x.shape[-2] — never the flattened (B·L², C) rows of T10's output), and the skip
line prints that L. Needs torch + the carried driver importable (triton); CPU tensors suffice — skipped by name otherwise."""
import contextlib
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "forward", "fast_inference", "driver")


def _w4():
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"torch not importable: {e!r}")
    if not os.path.isfile(os.path.join(DRIVER_DIR, "ef2_w4.py")):
        raise unittest.SkipTest("carried driver not beside the package")
    sys.path.insert(0, DRIVER_DIR)
    try:
        import ef2_w4
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"ef2_w4 not importable here: {e!r}")
    return ef2_w4


class TestW4ProbeBudgetGate(unittest.TestCase):
    def setUp(self):
        self.w4 = _w4()
        import torch
        self.torch = torch
        self.saved = dict(self.w4._IDPROBE)
        self.w4._IDPROBE.update(enabled=True, budget=1300, done={})

    def tearDown(self):
        self.w4._IDPROBE.clear(); self.w4._IDPROBE.update(self.saved)

    def _probe(self, name, L, C=8):
        """One probe call as the T10 site makes it: the FLATTENED (B·L², C) output, the reference thunk, and tokens = the pair extent L."""
        torch = self.torch
        out2d = torch.zeros(L * L, C)                    # T10's flattened rows: B·L² of them — the number the gate must NOT read
        calls = []
        def ref():
            calls.append(1); return out2d.clone()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.w4._identity_probe(name, out2d, ref, tokens=L)
        return calls, buf.getvalue(), self.w4._IDPROBE["done"].get(name)

    def test_below_the_budget_the_probe_runs_for_every_lever(self):
        for name in ("T10", "T9/outgoing", "T9/incoming"):
            calls, printed, rec = self._probe(name, L=200)          # 200 pair tokens ≤ 1300, although the flattened output has 40000 rows
            self.assertEqual(len(calls), 1, f"{name}: the reference op must run below the budget")
            self.assertIsNotNone(rec); self.assertNotIn("skipped", rec); self.assertNotIn("error", rec, rec); self.assertTrue(rec["identical"], rec)
            self.assertIn(f"identity probe {name}", printed); self.assertIn("IDENTICAL", printed); self.assertNotIn("skipped", printed)

    def test_above_the_budget_the_probe_is_skipped_by_name_with_the_pair_extent(self):
        for name in ("T10", "T9/outgoing"):
            calls, printed, rec = self._probe(name, L=1400, C=2)
            self.assertEqual(calls, [], f"{name}: no reference buffers above the budget")
            self.assertEqual((rec["skipped"], rec["tokens"], rec["budget"]), (True, 1400, 1300))
            self.assertIn(f"identity probe {name} skipped", printed); self.assertIn("1400 tokens > EF2_GRAPH_BUDGET_TOKENS=1300", printed)
            self.assertNotIn(f"{1400 * 1400} tokens", printed)         # the flattened row count is never reported as tokens (it appears only inside the printed shape)

    def test_budget_zero_captures_everything_and_probes_everything(self):
        self.w4._IDPROBE.update(budget=0, done={})
        calls, printed, rec = self._probe("T10", L=1400, C=2)
        self.assertEqual(len(calls), 1); self.assertNotIn("skipped", rec); self.assertNotIn("error", rec, rec)


if __name__ == "__main__":
    unittest.main()
