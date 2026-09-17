"""Every entry with the shared core (opt_core) ABSENT: refused by name by the core pin gate (the package's copy of the core's kit template,
statement one of each entry) — `NOT ACTIVE: reason=core_missing:opt_core (pinned >= v<version> at <path>; nothing importable as opt_core
on sys.path)`, exit 3 — never a traceback, never the stock body. CPU only: subprocesses of this interpreter started with `-S` (no site
directories: nothing importable as opt_core) with the package and a stub `colabfold.batch` on PYTHONPATH."""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from colabfold_opt import _autoload, report

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoreMissing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        stub = os.path.join(self.tmp, "stub"); os.makedirs(os.path.join(stub, "colabfold"))
        open(os.path.join(stub, "colabfold", "__init__.py"), "w").close()
        open(os.path.join(stub, "colabfold", "batch.py"), "w").write(textwrap.dedent('''\
            import os
            def run(queries, result_dir, **kw):
                open(os.environ["STUB_MARK"], "w").write("stock body ran")
            '''))
        self.stub = stub
        self.mark = os.path.join(self.tmp, "mark")

    def run_trigger(self, **env):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "PYTHONPATH"))}
        e.update(PYTHONPATH=os.pathsep.join([self.stub, OPT]), STUB_MARK=self.mark, **env)
        code = "import colabfold_opt._autoload as a; a.install(); import colabfold.batch as b; b.run(queries=[], result_dir='x')"
        return subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, env=e)

    def test_refused_by_name_exit_3_no_stock_body(self):
        for mode in ("fast", "exact", "big"):
            r = self.run_trigger(COLABFOLD_OPT=mode)
            self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, r.stderr)
            self.assertIn("[colabfold-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v", r.stderr); self.assertIn("; nothing importable as opt_core on sys.path)\n", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertFalse(os.path.exists(self.mark), "the stock body ran")

    def test_command_line_refused_by_name(self):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "PYTHONPATH"))}
        e.update(PYTHONPATH=os.pathsep.join([self.stub, OPT]))
        for argv in (["check", "--mode", "fast"], ["check", "--mode", "big", "--n_gpu", "2"], ["pred", "--mode", "exact", "x.a3m", "o"]):
            r = subprocess.run([sys.executable, "-S", "-m", "colabfold_opt", *argv], capture_output=True, text=True, env=e)
            self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, (argv, r.stderr))
            self.assertIn("[colabfold-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v", r.stderr, argv); self.assertNotIn("Traceback", r.stderr, argv)

    def test_off_is_untouched(self):
        r = self.run_trigger(COLABFOLD_OPT="off")                     # no finder: the stub's own body runs, the core is never imported
        self.assertEqual(r.returncode, 0, r.stderr); self.assertTrue(os.path.exists(self.mark))

    def test_pairs(self):
        self.assertEqual((_autoload.PREFIX, _autoload.EXIT_NOT_ACTIVE, f"[{_autoload.TAG}]"), (report.PREFIX, report.EXIT_NOT_ACTIVE, report.PREFIX))
        from colabfold_opt import _core_gate
        self.assertEqual(_core_gate.EXIT_NOT_ACTIVE, report.EXIT_NOT_ACTIVE)

    def test_in_process_route_refuses_by_name(self):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "PYTHONPATH"))}
        e.update(PYTHONPATH=os.pathsep.join([self.stub, OPT]))
        for call in ("colabfold_opt.enable('fast', queries=[])", "colabfold_opt.enable('big')", "colabfold_opt.check('exact')", "colabfold_opt.status()"):
            r = subprocess.run([sys.executable, "-S", "-c", f"import colabfold_opt; {call}; print('returned')"], capture_output=True, text=True, env=e)
            self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, (call, r.stderr)); self.assertNotIn("returned", r.stdout, call)
            self.assertIn("[colabfold-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v", r.stderr, call); self.assertNotIn("Traceback", r.stderr, call)


if __name__ == "__main__":
    unittest.main()
