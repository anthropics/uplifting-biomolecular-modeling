"""The kit modes' runner (opt/forward/xattempt_addon/src/xa_run.py) takes the pipeline module whose code it executes, bg_inproc.py, from its
own tree — opt/forward/fast_inference/src, reached from the runner's real location — before any sys.path entry: a bg_inproc.py in the working
directory or on a relative / foreign PYTHONPATH entry never stands in for it (it is named on one stderr line and not run), a runner reached
through a symbolic link still finds its own tree's copy, and a launch from another working directory changes nothing. A runner copied out of
the tree layout is served by PYTHONPATH as before. Run on the runner's own bytes in a scratch tree whose partner module only records that it
ran (no GPU, no boltzgen)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from boltzgen_opt import modes, stack

RUNNER = os.path.join(stack.opt_home(), modes.KIT_XATTEMPT, modes.RUNNER)
RECORDER = "import json, os, sys\njson.dump({'file': __file__, 'argv': sys.argv[1:], 'cwd': os.getcwd()}, open(os.environ['XA_TEST_RECORD'], 'w'))\n"
PLANT = "import os\nopen(os.environ['XA_TEST_RECORD'] + '.planted', 'w').write(__file__)\n"


class TestRunnerPartnerPath(unittest.TestCase):
    def setUp(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML absent (the runner reads steps.yaml with it)")
        self.td = os.path.realpath(tempfile.mkdtemp(prefix="bg_runner_"))
        self.addCleanup(shutil.rmtree, self.td, ignore_errors=True)
        fwd = os.path.join(self.td, "tree", "opt", "forward")
        self.addon = os.path.join(fwd, "xattempt_addon", "src"); os.makedirs(self.addon)
        self.partner = os.path.join(fwd, "fast_inference", "src"); os.makedirs(self.partner)
        shutil.copy(RUNNER, self.addon)
        with open(os.path.join(self.partner, "bg_inproc.py"), "w") as fh:
            fh.write(RECORDER)
        self.run_dir = os.path.join(self.td, "job"); os.makedirs(self.run_dir)
        with open(os.path.join(self.run_dir, "steps.yaml"), "w") as fh:
            fh.write("steps:\n- name: design\n  config_file: design.yaml\n")
        self.cwd = os.path.join(self.td, "elsewhere"); os.makedirs(self.cwd)                   # another working directory, holding a planted bg_inproc.py
        with open(os.path.join(self.cwd, "bg_inproc.py"), "w") as fh:
            fh.write(PLANT)
        self.record = os.path.join(self.td, "record.json")

    def _run(self, runner, pythonpath):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "BG_GRAPH")}
        env.update({"XA_FAST_INIT": "0", "XA_HOIST": "0", "XA_TEST_RECORD": self.record, "PYTHONPATH": pythonpath})   # no lever import: the partner lookup alone
        return subprocess.run([sys.executable, runner, self.run_dir, "7", "design"], cwd=self.cwd, env=env, capture_output=True, text=True, timeout=300)

    def _assert_own_partner_ran(self, p):
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(self.record) as fh:
            rec = json.load(fh)
        self.assertEqual(os.path.realpath(rec["file"]), os.path.join(self.partner, "bg_inproc.py"))
        self.assertEqual(rec["argv"], [self.run_dir, "7", "design"])
        self.assertEqual(rec["cwd"], self.cwd)                                                   # launched from the other working directory
        self.assertFalse(os.path.exists(self.record + ".planted"), "the planted bg_inproc.py ran")

    def test_the_kit_launch_from_another_working_directory_is_silent(self):
        p = self._run(os.path.join(self.addon, "xa_run.py"), os.pathsep.join([self.addon, self.partner]))   # PYTHONPATH as the package composes it: absolute, add-on first
        self._assert_own_partner_ran(p)
        self.assertNotIn("NOT RUN", p.stderr)

    def test_a_planted_module_in_the_working_directory_or_first_on_pythonpath_is_named_and_not_run(self):
        for pythonpath in (os.pathsep.join([self.cwd, self.partner]), os.pathsep.join(["", self.partner]), os.pathsep.join([".", self.addon, self.partner])):
            with self.subTest(pythonpath=pythonpath):
                for f in (self.record, self.record + ".planted"):
                    if os.path.exists(f):
                        os.remove(f)
                p = self._run(os.path.join(self.addon, "xa_run.py"), pythonpath)
                self._assert_own_partner_ran(p)
                lines = [l for l in p.stderr.splitlines() if "NOT RUN" in l]
                self.assertEqual(len(lines), 1, p.stderr)                                        # one line, by name: the stray file, the file that runs, the remedy
                self.assertIn(os.path.join(self.cwd, "bg_inproc.py"), lines[0])
                self.assertIn(os.path.join(self.partner, "bg_inproc.py"), lines[0])
                self.assertIn("remove that file or that PYTHONPATH entry", lines[0])

    def test_a_runner_reached_through_a_symbolic_link_finds_its_own_tree(self):
        link = os.path.join(self.cwd, "xa_run.py"); os.symlink(os.path.join(self.addon, "xa_run.py"), link)
        p = self._run(link, os.pathsep.join([self.cwd, ""]))                                     # the partner directory is not even on PYTHONPATH
        self._assert_own_partner_ran(p)

    def test_pythonpath_still_serves_a_runner_outside_the_tree_layout(self):
        lone = os.path.join(self.td, "lone"); os.makedirs(lone); shutil.copy(RUNNER, lone)       # no ../../fast_inference/src beside it: the first PYTHONPATH entry holding the module, as before
        os.remove(os.path.join(self.cwd, "bg_inproc.py"))
        p = self._run(os.path.join(lone, "xa_run.py"), self.partner)
        self._assert_own_partner_ran(p)
        self.assertNotIn("NOT RUN", p.stderr)
        p = self._run(os.path.join(lone, "xa_run.py"), lone)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("bg_inproc.py not found", p.stderr)


if __name__ == "__main__":
    unittest.main()
