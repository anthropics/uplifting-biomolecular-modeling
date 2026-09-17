"""`pred_bw --items <file>` at the package level: many items in one process on the fast arm — the line becomes the kit's entry script with its
own --items loop (opt/kit_ho/tf/pred_bw_fast.py --items <file> <stock args minus -r/-op/-os>), the mode line ends multi=N, and every other
shape is refused by name: mode off, the items file missing or malformed, -r/-op/-os on the line. The child's items reader and refusals are
tested by file (no GPU)."""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from chrombpnet_opt import stack, report, cli
from . import _stubs

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "kit_ho", "tf", "pred_bw_fast.py")   # the one-source form: the kit's entry script's --items loop


def items_file(rows):
    d = tempfile.mkdtemp(); p = os.path.join(d, "items.tsv")
    with open(p, "w") as fh:
        fh.write("# regions\toutput_prefix\tstats\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    return p


REAL_TREE = bool(_stubs.real_kit() and _stubs.real_kit("kit_ho"))            # the real tree: both kits present (the script refuses without the carried kit beside it)


class MultiTests(unittest.TestCase):
    def setUp(self):
        self.bed = os.path.join(tempfile.mkdtemp(), "a.bed"); open(self.bed, "w").write("chr1\t1000\t3000\t.\t0\t.\t0\t0\t0\t1000\n")
        self.items = items_file([(self.bed, "/tmp/x/a", "/tmp/x/a.stats"), (self.bed, "/tmp/x/b", "")])
        self.stock = ["-g", "g.fa", "-c", "c.sizes", "-cm", "m.h5", "-bw", "o.bw"]

    @unittest.skipUnless(REAL_TREE, "opt/kit and opt/kit_ho are not both in this tree")
    def test_plain_line_without_items(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            rep = stack.activate("fast", dry_run=True, quiet=True, trigger="env", args=self.stock + ["-r", self.bed, "-op", "/tmp/x/a"])
            self.assertTrue(rep["active"]); self.assertIsNone(rep.get("multi")); self.assertNotIn("multi=", report.mode_line(rep))
            self.assertTrue(rep["line"][1].endswith(os.path.join("kit_ho", "tf", "pred_bw_fast.py")), rep["line"])      # the fast mode's kit (its one-item form)

    @unittest.skipUnless(REAL_TREE, "opt/kit and opt/kit_ho are not both in this tree")
    def test_items(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertEqual(stack.multi_items(self.items), 2)
            rep = stack.activate("fast", dry_run=True, quiet=True, trigger="env", args=self.stock, items=self.items)
            self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["multi"], 2)
            self.assertTrue(report.mode_line(rep).endswith(" multi=2"), report.mode_line(rep))
            self.assertTrue(rep["kit"].endswith(os.path.join("opt", "kit_ho")))
            self.assertEqual(rep["line"][1:4], [CHILD, "--items", os.path.abspath(self.items)]); self.assertEqual(rep["line"][4:], self.stock)

    @unittest.skipUnless(REAL_TREE, "opt/kit and opt/kit_ho are not both in this tree")
    def test_refusals(self):
        cases = [("/nonexistent/items.tsv", self.stock, "not a file"),
                 (self.items, self.stock + ["-r", self.bed], "every item names its own"),
                 (self.items, self.stock + ["-op", "/tmp/x/a"], "every item names its own"),
                 (items_file([("only-one-column",)]), self.stock, "needs `regions")]
        for items, args, word in cases:
            with mock.patch.dict(os.environ, {}, clear=False):
                rep = stack.activate("fast", dry_run=True, quiet=True, trigger="env", args=args, items=items)
                self.assertFalse(rep["active"], (items, args)); self.assertIn(word, rep.get("reason") or "", (items, args, rep.get("reason")))

    def test_mode_off_refused(self):
        rep = stack.activate("off", dry_run=True, quiet=True, trigger="env", args=self.stock, items=self.items)
        self.assertFalse(rep["active"]); self.assertIn("mode off", rep.get("reason") or "")

    def test_cli_flag(self):
        ap = cli.build_parser()
        ns, extras = ap.parse_known_args(["pred_bw", "--items", self.items, "-g", "g.fa", "-bw", "o.bw"])      # the stock arguments pass through (cli.main)
        self.assertEqual(ns.items, self.items); self.assertEqual(extras, ["-g", "g.fa", "-bw", "o.bw"])
        ns, extras = ap.parse_known_args(["pred_bw", "-g", "g.fa"])
        self.assertIsNone(ns.items); self.assertEqual(extras, ["-g", "g.fa"])

    @unittest.skipUnless(REAL_TREE, "opt/kit and opt/kit_ho are not both in this tree")      # the script refuses without the carried kit beside it
    def test_child_refuses_by_name(self):
        env = dict(os.environ); env.pop("CHROMBPNET_OPT", None)
        env.pop("CHROMBPNET_FASTKIT_RECORD", None)
        r = subprocess.run([sys.executable, CHILD, "--items", self.items] + self.stock + ["-r", self.bed], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr); self.assertIn("[pred_bw_fast] NOT ACTIVE multi: ", r.stdout); self.assertIn("not served", r.stdout); self.assertNotIn("Traceback", r.stderr)
        for extra in (["--no-metrics-overlap"], ["--forward", "k1"], ["--chunk=2048"]):                  # the entry script has no lever flags: the stock arguments and --items only (argparse refuses the word, exit 2)
            for e2 in (env, dict(env, CHROMBPNET_FASTKIT_RECORD=os.path.join(tempfile.mkdtemp(), "record.json"))):
                r = subprocess.run([sys.executable, CHILD, "--items", self.items] + self.stock + extra, capture_output=True, text=True, env=e2)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr); self.assertIn("unrecognized arguments: %s" % extra[0].split("=")[0], r.stderr); self.assertNotIn("Traceback", r.stderr)
        bad = items_file([("/nonexistent.bed", "/tmp/x/a")])
        r = subprocess.run([sys.executable, CHILD, "--items", bad] + self.stock, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 3); self.assertIn("[pred_bw_fast] NOT ACTIVE multi: ", r.stdout); self.assertIn("not found", r.stdout)
        dup = items_file([(self.bed, "/tmp/x/a"), (self.bed, "/tmp/x/a")])
        r = subprocess.run([sys.executable, CHILD, "--items", dup] + self.stock, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 3); self.assertIn("[pred_bw_fast] NOT ACTIVE multi: ", r.stdout); self.assertIn("named twice", r.stdout)



if __name__ == "__main__":
    unittest.main()
