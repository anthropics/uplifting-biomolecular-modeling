"""`warm`'s results directory: private and fresh without ``--out``; a given ``--out`` another account could have written into is refused
by name before anything is launched (colabfold_batch unpickles result files it finds in a results directory)."""
import contextlib
import io
import os
import shutil
import stat
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
if OPT not in sys.path:
    sys.path.insert(0, OPT)

from colabfold_opt import cli, warm  # noqa: E402


def args(out=None):
    return SimpleNamespace(command="warm", mode=None, n_gpu=1, det=0, out=out, json=False, help=False)


class TestWarmResultsDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="colabfold_warm_out_")
        self.launched = []                                                  # the results directory of every pred launch
        self.made = []                                                      # private directories mkdtemp made (removed in tearDown)

        def fake_pred(a, mode, argv):
            self.launched.append(argv[1])
            return 0, {"active": True, "verdict": {}, "result_dir": argv[1]}
        self.patch = mock.patch.object(cli, "pred", fake_pred)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        for d in self.made + [self.tmp]:
            shutil.rmtree(d, ignore_errors=True)

    def run_warm(self, out=None):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc, rec = warm.main(args(out), "fast", [])
        return rc, err.getvalue()

    def test_no_out_is_a_fresh_private_directory_named_before_the_launch(self):
        rc1, err1 = self.run_warm()
        rc2, err2 = self.run_warm()
        self.made += self.launched
        self.assertEqual((rc1, rc2), (0, 0))
        self.assertEqual(len(self.launched), 2)
        a, b = self.launched
        self.assertNotEqual(a, b)                                            # a name no other process predicts: never a fixed one
        for d, err in ((a, err1), (b, err2)):
            self.assertTrue(os.path.basename(d).startswith(warm.OUT_PREFIX), d)
            self.assertEqual(os.path.dirname(d), os.path.abspath(tempfile.gettempdir()))
            st = os.lstat(d)
            self.assertTrue(stat.S_ISDIR(st.st_mode)); self.assertEqual(st.st_uid, os.geteuid())
            self.assertEqual(stat.S_IMODE(st.st_mode), 0o700, oct(st.st_mode))
            first = err.splitlines()[0]
            self.assertTrue(first.startswith(f"[colabfold-opt] WARM mode=fast out={d} "), first)   # where it writes, before the minutes-long launch
        self.assertNotEqual(os.path.basename(a), "colabfold_opt_warm")

    def test_out_writable_by_others_is_refused_by_name_and_nothing_is_launched(self):
        for mode in (0o777, 0o1777, 0o770, 0o757):
            d = os.path.join(self.tmp, f"shared_{mode:o}"); os.mkdir(d); os.chmod(d, mode)
            rc, err = self.run_warm(d)
            self.assertEqual(rc, cli.EXIT_USAGE, err)
            self.assertEqual(self.launched, [])
            self.assertIn(f"[colabfold-opt] WARM REFUSED mode=fast rc=2 reason=out_dir out={d}: {d} is writable by group or other (mode {mode:04o})", err)

    def test_out_of_another_owner_is_refused_unless_root(self):
        d = os.path.join(self.tmp, "theirs"); os.mkdir(d, 0o755)
        real = os.stat(d)
        fake = os.stat_result((real.st_mode, real.st_ino, real.st_dev, real.st_nlink, os.geteuid() + 1, real.st_gid, real.st_size, real.st_atime, real.st_mtime, real.st_ctime))
        with mock.patch.object(warm.os, "stat", return_value=fake), mock.patch.object(warm.os, "geteuid", return_value=fake.st_uid - 1 or 4242):
            why = warm.out_refusal(d)
        self.assertTrue(why and why.startswith(f"{d} belongs to uid {fake.st_uid}, not to this process"), why)
        with mock.patch.object(warm.os, "stat", return_value=fake), mock.patch.object(warm.os, "geteuid", return_value=0):
            self.assertIsNone(warm.out_refusal(d))                          # uid 0 accepts any owner (a container's root over a bind-mounted host directory)
        root_owned = os.stat_result((real.st_mode,) + tuple(real)[1:4] + (0,) + tuple(real)[5:10])
        with mock.patch.object(warm.os, "stat", return_value=root_owned):
            self.assertIsNone(warm.out_refusal(d))                          # root's own directory is nobody else's

    def test_out_absent_or_own_is_used_as_given(self):
        absent = os.path.join(self.tmp, "later", "warm")
        rc, err = self.run_warm(absent)
        self.assertEqual((rc, self.launched), (0, [absent]))
        self.assertFalse(os.path.exists(absent))                            # colabfold_batch makes it; warm does not
        self.assertNotIn("WARM mode=fast out=", err)                        # the caller named it: no where-line
        own = os.path.join(self.tmp, "own"); os.mkdir(own); os.chmod(own, 0o755)
        rc, err = self.run_warm(own)
        self.assertEqual((rc, self.launched[-1]), (0, own))
        here = os.getcwd()
        try:
            os.chdir(self.tmp)
            rc, err = self.run_warm("own")                                   # relative: made absolute, as before
        finally:
            os.chdir(here)
        self.assertEqual((rc, self.launched[-1]), (0, own))


if __name__ == "__main__":
    unittest.main()
