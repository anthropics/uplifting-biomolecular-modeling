"""The PEAK line (esmfold2_opt/peakmem.py) in the release's one grammar — `PEAK item=<id> alloc_gib=<f> reserved_gib=<f>`, nothing else on the line;
the pass's seed and resident GiB on a separate PEAK-NOTE line; without torch / CUDA (this CPU stub) NO PEAK line, one `PEAK-NOTE item=<id> cuda=absent`."""
import re
import sys
import types
import unittest
from unittest import mock

from esmfold2_opt import peakmem

PEAK = re.compile(r"^PEAK item=(\S+) alloc_gib=(\d+\.\d+) reserved_gib=(\d+\.\d+)$")          # the reader's grammar: two positional float groups after item, no other token


class PeakLine(unittest.TestCase):
    def test_grammar_with_values(self):
        out = peakmem.lines("PDL1x70_d0_wt", 42, 74.02, 78.61, 23.4)
        self.assertEqual(out, ["PEAK item=PDL1x70_d0_wt alloc_gib=74.02 reserved_gib=78.61", "PEAK-NOTE item=PDL1x70_d0_wt seed=42 resident_gib=23.40"])
        self.assertRegex(out[0], PEAK); self.assertNotIn("seed=", out[0])

    def test_no_cuda_prints_no_peak_line_only_the_absent_note(self):
        fake = types.ModuleType("torch"); fake.cuda = types.SimpleNamespace(is_available=lambda: False)
        avail_no_stats = types.ModuleType("torch"); avail_no_stats.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
        for mods in ({"torch": fake}, {"torch": types.ModuleType("torch")}, {"torch": avail_no_stats}):   # CUDA not available; a torch without .cuda; a cuda namespace without the statistics API (a test double)
            with mock.patch.dict(sys.modules, mods):
                self.assertFalse(peakmem.begin())
                self.assertEqual(peakmem.peak(), (None, None)); self.assertIsNone(peakmem.resident())
                out = peakmem.lines("x", 0, *peakmem.peak(), peakmem.resident())
            self.assertEqual(out, ["PEAK-NOTE item=x cuda=absent"])
            self.assertFalse([l for l in out if PEAK.match(l) or l.startswith("PEAK ")])
        with mock.patch.dict(sys.modules):
            sys.modules.pop("torch", None)                                                       # torch not loaded at all
            self.assertEqual(peakmem.lines("x", 1, *peakmem.peak(), peakmem.resident()), ["PEAK-NOTE item=x cuda=absent"])

    def test_cuda_counters_are_read_and_reset(self):
        calls = []
        cuda = types.SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: calls.append("reset"),
                                     max_memory_allocated=lambda: 3 * 2 ** 30, max_memory_reserved=lambda: int(4.5 * 2 ** 30), memory_allocated=lambda: 2 ** 29)
        fake = types.ModuleType("torch"); fake.cuda = cuda
        with mock.patch.dict(sys.modules, {"torch": fake}):
            self.assertTrue(peakmem.begin()); self.assertEqual(calls, ["reset"])
            self.assertEqual(peakmem.lines("it", 7, *peakmem.peak(), peakmem.resident()), ["PEAK item=it alloc_gib=3.00 reserved_gib=4.50", "PEAK-NOTE item=it seed=7 resident_gib=0.50"])

    def test_the_loop_prints_it_on_every_route(self):
        """stock_fold.fold_items (the one loop of every route) resets the counters before the fold window and prints after the PHASE line (source order)."""
        import inspect
        from esmfold2_opt import stock_fold
        src = inspect.getsource(stock_fold.fold_items)
        i_begin, i_fold, i_peak, i_phase, i_line = (src.index(k) for k in ("peakmem.begin()", "builder.fold(", "peakmem.peak()", "phases.line(", "peakmem.lines("))
        self.assertTrue(i_begin < i_fold < i_peak < i_phase < i_line)


if __name__ == "__main__":
    unittest.main()
