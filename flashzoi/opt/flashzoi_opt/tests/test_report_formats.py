"""The lines are formatted in report.py and nowhere else: the ACTIVE line (once per process), NOT ACTIVE, DRY-RUN, the per-item
line, the EXIT tally and its counters."""
import io
import re
import unittest

from flashzoi_opt import report


def _rep(**kw):
    base = {"mode": "exact", "active": True, "gpu": {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0"}, "components_planned": ["a", "b"], "components_unavailable": []}
    base.update(kw)
    return base


class TestReportFormats(unittest.TestCase):
    def setUp(self):
        report._ACTIVE_PRINTED["done"] = False
        report.ITEMS.update(items=0, ok=0, failed=0)

    def test_active_line(self):
        self.assertEqual(report.activation_line(_rep()), "[flashzoi-opt] ACTIVE mode=exact variant=- gpu=NVIDIA H100 80GB HBM3 cc=9.0 components=a,b (partial: none)")
        self.assertEqual(report.activation_line(_rep(components_unavailable=["b"])), "[flashzoi-opt] ACTIVE mode=exact variant=- gpu=NVIDIA H100 80GB HBM3 cc=9.0 components=a,b (partial: b)")

    def test_active_printed_once(self):
        s = io.StringIO()
        report.log_activation(_rep(), stream=s); report.log_activation(_rep(), stream=s)
        self.assertEqual(s.getvalue().count("ACTIVE mode=exact"), 1)

    def test_not_active_and_dry_run(self):
        self.assertEqual(report.not_active_line("why"), "[flashzoi-opt] NOT ACTIVE: why")
        line = report.dry_run_line(_rep(knobs_line="numerics=tf32", stack_key="9.0|3.1", would_refuse=None))
        self.assertTrue(line.startswith("[flashzoi-opt] DRY-RUN mode=exact variant=- gpu=NVIDIA H100 80GB HBM3 cc=9.0 components=a,b knobs=numerics=tf32 stack_key=9.0|3.1 would_refuse=none"), line)

    def test_pred_and_exit(self):
        self.assertEqual(report.pred_line("chr1_1_2", 1.23456), "[flashzoi-opt] pred chr1_1_2 s0 samples=1 1.235s")
        report.note_item(True); report.note_item(False); report.note_item(True)
        line = report.exit_tally_line("exact", wall_s=12.34)
        self.assertEqual(line, "[flashzoi-opt] EXIT mode=exact items=3 ok=2 failed=1 wall=12.3s partial=none")
        self.assertRegex(report.exit_tally_line("off"), r"^\[flashzoi-opt\] EXIT mode=off items=3 ok=2 failed=1 wall=\d+\.\ds partial=none$")

    def test_partial_line(self):
        """partial_line(): the refusal form (exit_code given) vs the --allow-partial form (exit_code None) -- the only two
        exercises of PARTIAL_FMT / PARTIAL_ALLOWED_FMT; every other *_FMT constant is already pinned by a full rendered-line
        comparison elsewhere in this file."""
        self.assertEqual(report.partial_line("graph: why", 3), "[flashzoi-opt] NOT ACTIVE: partial activation — graph: why; exit 3 (--allow-partial records and proceeds)")
        self.assertEqual(report.partial_line("graph: why"), "[flashzoi-opt] PARTIAL allowed: graph: why (--allow-partial, recorded)")

    def test_no_other_module_prints_the_prefix_lines(self):
        """Every '[flashzoi-opt]' literal lives in report.py (the other modules format through it)."""
        import glob, os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for p in glob.glob(os.path.join(here, "*.py")):
            if os.path.basename(p) in ("report.py", "_autoload.py"):       # _autoload prints one line before the package is importable (unknown env mode)
                continue
            src = open(p, encoding="utf-8").read()
            code = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
            self.assertNotRegex(code, re.compile(r'(f?"\[flashzoi-opt\] (ACTIVE|NOT ACTIVE|DRY-RUN|EXIT|pred) )'), f"{p} formats a line outside report.py")


if __name__ == "__main__":
    unittest.main()
