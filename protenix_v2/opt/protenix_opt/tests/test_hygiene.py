"""Kit hygiene (protenix_opt 0.3.31): (1) the stock subprocess's CUEQ-CACHE census — where cuequivariance_ops' triton cache manager would read
tuned-tile json in the CHILD's environment and how many are there, resolved as the library resolves it, counted without creating the directory,
printed next to ENV-CLEAN (evidence only, nothing redirected); (2) `pred --det 1` restores the detref copy on EVERY exit path — a failing pred and
a terminated pred leave the protenix package tree as it was (the by-hand refusal in det.plan() stays the backstop for a kill -9)."""
import hashlib
import os
import signal
import tempfile
import unittest
from unittest import mock

from protenix_opt import cli, det, stock_pred


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class TestCueqUserCacheCensus(unittest.TestCase):
    def test_env_variable_wins(self):
        c = stock_pred.cueq_user_cache_census({"CUEQ_TRITON_CACHE_DIR": "/opt/cache/cueq", "XDG_CACHE_HOME": "/xdg", "HOME": "/home/u"})
        self.assertEqual((c["dir"], c["source"], c["exists"], c["json"]), ("/opt/cache/cueq", "env", False, 0))

    def test_xdg_then_home_like_platformdirs(self):
        self.assertEqual(stock_pred.cueq_user_cache_census({"XDG_CACHE_HOME": "/xdg", "HOME": "/home/u"})["dir"], "/xdg/cuequivariance-triton")
        self.assertEqual(stock_pred.cueq_user_cache_census({"XDG_CACHE_HOME": "", "HOME": "/home/u"})["dir"], "/home/u/.cache/cuequivariance-triton")
        self.assertEqual(stock_pred.cueq_user_cache_census({"XDG_CACHE_HOME": "relative/ignored", "HOME": "/home/u"})["dir"], "/home/u/.cache/cuequivariance-triton",
                         "platformdirs ignores a relative XDG_CACHE_HOME")

    def test_counts_json_without_creating_the_directory(self):
        with tempfile.TemporaryDirectory() as home:
            c = stock_pred.cueq_user_cache_census({"HOME": home})
            self.assertEqual((c["exists"], c["json"]), (False, 0)); self.assertFalse(os.path.exists(os.path.join(home, ".cache")), "the census never creates the dir")
            d = os.path.join(home, ".cache", "cuequivariance-triton"); os.makedirs(d)
            for f in ("triangle_attention.9.0.json", "trimul.9.0.json", "notes.txt"):
                open(os.path.join(d, f), "w").write("{}")
            c = stock_pred.cueq_user_cache_census({"HOME": home})
            self.assertEqual((c["dir"], c["source"], c["exists"], c["json"]), (d, "home", True, 2))
            self.assertEqual(stock_pred.cueq_cache_line(c), f"[protenix-opt stock] CUEQ-CACHE cueq_user_cache={d} source=home exists=1 json=2")

    def test_the_line_is_printed_next_to_env_clean(self):
        src = open(stock_pred.__file__, encoding="utf-8").read()
        i = src.index('ENV-CLEAN ok: absent='); j = src.index("print(cueq_cache_line(", i)
        self.assertLess(j - i, 600, "the census line follows the ENV-CLEAN line in run()")


class _FakeTree:
    """A protenix package dir with utils/scatter_utils.py and an FPF home with src/detref/{scatter_utils.py, det_segment_reduce.py}."""
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = self.tmp.name
        self.pkg = os.path.join(r, "site", "protenix"); os.makedirs(os.path.join(self.pkg, "utils"))
        self.target = os.path.join(self.pkg, "utils", "scatter_utils.py"); open(self.target, "w").write("STOCK scatter_utils\n")
        self.fpf = os.path.join(r, "fpf"); os.makedirs(os.path.join(self.fpf, "src", "detref"))
        open(os.path.join(self.fpf, "src", "detref", "scatter_utils.py"), "w").write("DETREF scatter_utils\n")
        open(os.path.join(self.fpf, "src", "detref", "det_segment_reduce.py"), "w").write("DETREF det_segment_reduce\n")
        self.stock_sha = _sha(self.target)

    def plan(self):
        with mock.patch("protenix_opt.kits.missing_kit_files", return_value=[]):
            return det.plan(fpf_home=self.fpf, package_dir=self.pkg)

    def assert_restored(self, tc):
        tc.assertEqual(_sha(self.target), self.stock_sha, "scatter_utils.py carries the stock bytes again")
        tc.assertFalse(os.path.exists(self.target + det.BACKUP_SUFFIX), "the backup is gone")
        tc.assertFalse(os.path.exists(os.path.join(self.pkg, "utils", "det_segment_reduce.py")), "the det-only file is gone")
        tc.assertEqual(self.plan()["problems"], [], "the next --det 1 plans clean")


class TestDetRestoresOnEveryExit(unittest.TestCase):
    def setUp(self):
        self.tree = _FakeTree(); self.addCleanup(self.tree.tmp.cleanup)

    def _patched(self):
        p = det.Patch(self.tree.plan()).apply()
        self.assertNotEqual(_sha(self.tree.target), self.tree.stock_sha, "applied: the detref bytes are in place")
        return p

    def _pred_with(self, patch, body):
        """cli.pred's guard structure around a route body: det_patch -> guard_signals -> try body finally restore + unguard."""
        with mock.patch.object(cli, "det_patch", return_value=patch), mock.patch.object(cli, "_pred_routes", side_effect=body), \
             mock.patch.object(cli, "check_mode"), mock.patch.object(cli.tp, "refusal", return_value=None), \
             mock.patch.object(cli._frozen, "check", return_value=None), mock.patch.object(cli, "weights_note"):
            return cli.cmd_pred(["--mode", "exact", "--det", "1", "--input", "x.json", "--out_dir", "o"])

    def test_a_failing_pred_restores_the_tree(self):
        patch = self._patched()
        def body(*a, **k): raise RuntimeError("PATCH failure inside the model")
        with self.assertRaises(RuntimeError):
            self._pred_with(patch, body)
        self.tree.assert_restored(self); self.assertTrue(patch.restored)

    def test_a_terminated_pred_restores_the_tree(self):
        patch = self._patched()
        def body(*a, **k): os.kill(os.getpid(), signal.SIGTERM); raise AssertionError("not reached: the guard raises det.Terminated")
        with self.assertRaises(det.Terminated) as cm:
            self._pred_with(patch, body)
        self.assertEqual(cm.exception.code, 128 + int(signal.SIGTERM))
        self.tree.assert_restored(self)
        self.assertNotEqual(signal.getsignal(signal.SIGTERM), det._raise_terminated, "the guard is lifted after the run")

    def test_a_clean_pred_restores_the_tree_and_returns_the_code(self):
        patch = self._patched()
        self.assertEqual(self._pred_with(patch, lambda *a, **k: 0), 0)
        self.tree.assert_restored(self)

    def test_the_backstop_stays(self):
        """A copy nobody restored (kill -9): the next plan refuses by name until restored by hand — unchanged."""
        det.Patch(self.tree.plan()).apply()                     # applied and abandoned (atexit would restore at interpreter exit; plan() sees the tree now)
        probs = self.tree.plan()["problems"]
        self.assertTrue(any("already carries the detref bytes" in p for p in probs) and any("a backup from an earlier run is present" in p for p in probs), probs)


if __name__ == "__main__":
    unittest.main()
