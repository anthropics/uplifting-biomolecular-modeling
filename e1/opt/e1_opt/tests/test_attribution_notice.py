"""`run.sh` prints the Profluent-E1 attribution notice once, on stderr, at the start of every verb — before any step of the verb runs —
and names the licence file the kit carries (stock/src/LICENSE, with NOTICE and ATTRIBUTION beside it). A word that is not a verb gets the
usage text and no notice. CPU only: the install verb's own usage error ends the run before python is needed."""
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")
PREFIX = "[e1-kit] Built with Profluent-E1"


class AttributionNotice(unittest.TestCase):
    def run_sh(self, args):
        tmp = tempfile.mkdtemp(prefix="e1_notice_")
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp}
        return subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=tmp)

    def test_notice_is_the_first_line_of_a_verb(self):
        r = self.run_sh(["install", "--bogus"])                          # a verb; its usage error stops it right after the notice
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        err = r.stderr.splitlines()
        self.assertTrue(err and err[0].startswith(PREFIX), r.stderr)
        self.assertIn("Profluent-E1 Clickthrough License Agreement", err[0])
        self.assertIn(os.path.join(TREE, "stock", "src", "LICENSE"), err[0])
        self.assertEqual(sum(1 for ln in err if ln.startswith(PREFIX)), 1, r.stderr)   # said once
        self.assertNotIn(PREFIX, r.stdout)                                # stderr only: stdout stays the verb's own

    def test_named_files_are_carried(self):
        for name in ("LICENSE", "NOTICE", "ATTRIBUTION"):
            self.assertTrue(os.path.isfile(os.path.join(TREE, "stock", "src", name)), name)

    def test_no_notice_without_a_verb(self):
        r = self.run_sh(["frobnicate"])
        self.assertEqual(r.returncode, 2)
        self.assertNotIn(PREFIX, r.stdout + r.stderr)

    def test_notice_precedes_every_verb_in_the_script(self):
        text = open(RUN_SH, encoding="utf-8").read().splitlines()
        gate = [i for i, ln in enumerate(text) if ln.startswith('case "$CMD" in score|check|warm|install)')]
        self.assertEqual(len(gate), 1)
        self.assertTrue(text[gate[0] + 1].startswith('echo "' + PREFIX), text[gate[0] + 1])   # the line right after the verb gate, ahead of every verb's branch


if __name__ == "__main__":
    unittest.main()
