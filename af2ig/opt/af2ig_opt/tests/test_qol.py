"""QoL (0.7.3.1): the --no-compile alias / compile= word, warm --lengths parsing and the synthetic complex writer."""
import math
import os
import tempfile
import unittest

from af2ig_opt import registry, warm
from af2ig_opt.tests import _stubs
import tempfile as _tf


class TestNoCompileAlias(unittest.TestCase):
    def test_check_names_compile_stock_jit(self):
        with _tf.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            rc, so, err = _stubs.run_cli(["check", "--mode", "exact"], env)
            self.assertEqual(rc, 0, err); self.assertIn(" compile=stock_jit", err); self.assertNotIn("no-compile", err)
            rc, so, err = _stubs.run_cli(["check", "--mode", "exact", "--no-compile"], env)
            self.assertEqual(rc, 0, err); self.assertIn(" compile=stock_jit(no-compile:inherent;no_kit_compile_lever)", err)
            self.assertNotIn("unknown id", err)                                                    # the tree's word is known: nothing to drop, named — not an unknown id
            self.assertIn("levers=L6,L1,U1,L7,L13,L15,L16,ccache", err)                            # nothing dropped
            rc, so, err = _stubs.run_cli(["check", "--mode", "big"], dict(env, **{registry.LEVERS_OFF_ENV: "compile,L8"}))
            self.assertEqual(rc, 0, err); self.assertIn("compile=stock_jit(no-compile:inherent;no_kit_compile_lever)", err); self.assertIn("levers_off=L8", err)
            rc, so, err = _stubs.run_cli(["check", "--mode", "off", "--no-compile"], env)
            self.assertEqual(rc, 0, err); self.assertNotIn(" compile=", err)                      # the stock line carries no kit word (precompile= is the stock line's own token)


class TestWarmLengths(unittest.TestCase):
    def test_parse_lengths(self):
        self.assertEqual(warm.parse_lengths("400,800, 1200,400"), [400, 800, 1200])
        for bad in ("", "abc", "8", "400,-1"):
            with self.assertRaises(ValueError):
                warm.parse_lengths(bad)

    def test_write_complex(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.pdb"); nb, nt = warm.write_complex(p, 400)
            self.assertEqual(nb + nt, 400); self.assertEqual(nb, 80)
            atoms = [l for l in open(p).read().splitlines() if l.startswith("ATOM")]
            self.assertEqual(len(atoms), 4 * 400)
            self.assertEqual({l[21] for l in atoms}, {"A", "B"}); self.assertEqual(atoms[0][17:20], "GLY")
            ca = [l for l in atoms if l[12:16].strip() == "CA"]
            xyz = lambda l: (float(l[30:38]), float(l[38:46]), float(l[46:54]))
            d01 = math.dist(xyz(ca[0]), xyz(ca[1])); self.assertTrue(3.6 < d01 < 4.0, d01)      # consecutive CA ~3.8 A: a sane backbone
            dAB = math.dist(xyz(ca[0]), xyz(ca[nb])); self.assertTrue(dAB > 10.0, dAB)          # the two chains do not sit on top of each other

    def test_cli_accepts_lengths_and_off_note(self):
        with _tf.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            rc, so, err = _stubs.run_cli(["warm", "--mode", "off", "--lengths", "400"], env)
            self.assertIn("WARM note: --mode off keeps no compilation cache", err)


if __name__ == "__main__":
    unittest.main()
