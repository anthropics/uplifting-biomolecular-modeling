"""`--mode big` / `--n_gpu` against a shared core OLDER than the axis (no `opt_core.mem.ngpu`, no `opt_core.mem.rowpair_jax`): every entry
route refuses BY NAME with exit 3 before anything resolves — never a traceback, never the stock body, `exact` / `fast` untouched. CPU only: a
subprocess of this interpreter over an AMPUTATED COPY of the very core it imports (the two modules removed), so the test exercises the words on
every core, the pinned one or a newer one; the wholly-absent core is test_autoload_core_missing.py's."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import opt_core
from colabfold_opt import report

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORDS = "[colabfold-opt] NOT ACTIVE: core_missing:opt_core.mem.ngpu (--mode big and --n_gpu need the shared core's opt_core.mem.ngpu and opt_core.mem.rowpair_jax)"


class TestOlderCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        core = os.path.join(cls.tmp, "oldcore")
        shutil.copytree(os.path.dirname(os.path.abspath(opt_core.__file__)), os.path.join(core, "opt_core"), ignore=shutil.ignore_patterns("__pycache__"))
        for gone in ("mem/ngpu.py", "mem/rowpair_jax"):                 # the axis' producers, absent in a core older than the axis
            p = os.path.join(core, "opt_core", gone)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)
        cls.core = core
        stub = os.path.join(cls.tmp, "stub"); os.makedirs(os.path.join(stub, "colabfold"))
        open(os.path.join(stub, "colabfold", "__init__.py"), "w").close()
        open(os.path.join(stub, "colabfold", "batch.py"), "w").write(textwrap.dedent("""\
            import os
            def run(queries, result_dir, **kw):
                open(os.environ["STUB_MARK"], "w").write("stock body ran")
            """))
        cls.stub = stub

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, True)

    def env(self, **extra):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "PYTHONPATH"))}
        e.update(PYTHONPATH=os.pathsep.join([self.core, self.stub, OPT]), STUB_MARK=os.path.join(self.tmp, "mark"), **extra)   # the amputated core FIRST: it shadows the installed one
        return e

    def test_the_copy_is_the_older_core(self):
        r = subprocess.run([sys.executable, "-c", "import opt_core, importlib.util as u; print(opt_core.__file__); "
                            "print(u.find_spec('opt_core.mem.ngpu') is None, u.find_spec('opt_core.mem.rowpair_jax') is None)"],
                           capture_output=True, text=True, env=self.env())
        self.assertEqual(r.returncode, 0, r.stderr); self.assertTrue(r.stdout.splitlines()[0].startswith(self.core), r.stdout)
        self.assertEqual(r.stdout.splitlines()[1], "True True")

    def test_command_line_routes_refuse_by_name(self):
        for argv in (["check", "--mode", "big"], ["check", "--mode", "big", "--n_gpu", "2"], ["check", "--mode", "fast", "--n_gpu", "2"],
                     ["pred", "--mode", "big", "x.a3m", os.path.join(self.tmp, "o")]):
            r = subprocess.run([sys.executable, "-m", "colabfold_opt", *argv], capture_output=True, text=True, env=self.env())
            self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, (argv, r.stderr))
            self.assertIn(WORDS, r.stderr, argv); self.assertNotIn("Traceback", r.stderr, argv)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "o")))

    def test_trigger_route_refuses_by_name_no_stock_body(self):
        code = "import colabfold_opt._autoload as a; a.install(); import colabfold.batch as b; b.run(queries=[], result_dir='x')"
        for extra in ({"COLABFOLD_OPT": "big"}, {"COLABFOLD_OPT": "big", "COLABFOLD_OPT_N_GPU": "2"}, {"COLABFOLD_OPT": "fast", "COLABFOLD_OPT_N_GPU": "4"}):
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=self.env(**extra), cwd=self.tmp)   # the refusal's manifest lands under the test's dir, never the tree
            self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, (extra, r.stderr))
            self.assertIn("NOT ACTIVE: core_missing:opt_core.mem.ngpu", r.stderr, extra); self.assertNotIn("Traceback", r.stderr, extra)
            self.assertFalse(os.path.exists(os.path.join(self.tmp, "mark")), f"the stock body ran {extra}")

    def test_in_process_route_refuses_by_name(self):
        code = ("import colabfold_opt, json; rep = colabfold_opt.enable('big', queries=[('q', 'A' * 700, None)]); "
                "print(json.dumps({'active': rep['active'], 'reason': rep.get('reason')}))")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=self.env())
        self.assertEqual(r.returncode, 0, r.stderr); self.assertNotIn("Traceback", r.stderr)
        self.assertIn("NOT ACTIVE", r.stderr); self.assertIn("core_missing:opt_core.mem.ngpu", r.stderr)
        rep = json.loads(r.stdout.strip().splitlines()[-1]); self.assertFalse(rep["active"]); self.assertIn("core_missing:opt_core.mem.ngpu", rep["reason"])
        r = subprocess.run([sys.executable, "-c", code.replace("enable('big',", "enable('big', strict=True,")], capture_output=True, text=True, env=self.env())
        self.assertNotEqual(r.returncode, 0); self.assertIn("ActivationError", r.stderr); self.assertIn("core_missing:opt_core.mem.ngpu", r.stderr)   # strict: the refusal raises by name

    def test_exact_and_fast_at_p1_are_untouched_by_the_older_core(self):
        r = subprocess.run([sys.executable, "-c", "from colabfold_opt import stack; print(stack.gate_n_gpu('fast', 1, None), stack.gate_n_gpu('exact', 1, 8))"],
                           capture_output=True, text=True, env=self.env())
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "None None"), r.stderr)


class TestWrongVersionCore(unittest.TestCase):
    """A shared core that IMPORTS but is older than the pin (a full copy of the running core with another __version__): the activation's
    first gate (the opt_core pin) refuses by name — `opt_core pinned >= v<want> at <path>, installed v<have> at <root>` — on the command
    line, the trigger route and the in-process route; rc 3 wherever a process exits, no traceback, no stock body."""

    WORDS = "[colabfold-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned "

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        core = os.path.join(cls.tmp, "wrongcore")
        shutil.copytree(os.path.dirname(os.path.abspath(opt_core.__file__)), os.path.join(core, "opt_core"), ignore=shutil.ignore_patterns("__pycache__"))
        init = os.path.join(core, "opt_core", "__init__.py"); src = open(init).read()
        import re
        src, n = re.subn(r'(?m)^__version__\s*=\s*"[^"]*"', '__version__ = "0.2.5"', src); assert n == 1, "opt_core/__init__.py names __version__ once"
        open(init, "w").write(src)                                       # an older version than the pin's floor — what the gate reads
        cls.core = core
        stub = os.path.join(cls.tmp, "stub"); os.makedirs(os.path.join(stub, "colabfold"))
        open(os.path.join(stub, "colabfold", "__init__.py"), "w").close()
        open(os.path.join(stub, "colabfold", "batch.py"), "w").write(textwrap.dedent("""\
            import os
            def run(queries, result_dir, **kw):
                open(os.environ["STUB_MARK"], "w").write("stock body ran")
            """))
        cls.stub = stub

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, True)

    def env(self, **extra):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "PYTHONPATH", "MODEL_OPT"))}
        e.update(PYTHONPATH=os.pathsep.join([self.core, self.stub, OPT]), STUB_MARK=os.path.join(self.tmp, "mark"), **extra)
        return e

    def test_the_copy_is_an_older_core(self):
        r = subprocess.run([sys.executable, "-c", "from opt_core import gates; c = gates.imported_core(); print(c['package_dir']); print(c['version'])"],
                           capture_output=True, text=True, env=self.env())
        self.assertEqual(r.returncode, 0, r.stderr); self.assertTrue(r.stdout.splitlines()[0].startswith(self.core), r.stdout)
        self.assertEqual(r.stdout.splitlines()[1], "0.2.5")
        from opt_core import gates
        pin = gates.core_pin(os.path.join(OPT, "pyproject.toml"))
        self.assertNotEqual("0.2.5", pin["version"])

    def test_command_line_refuses_by_name(self):
        for argv in (["check", "--mode", "fast"], ["check", "--mode", "exact"], ["check", "--mode", "big", "--n_gpu", "2"],
                     ["pred", "--mode", "fast", "x.a3m", os.path.join(self.tmp, "o")]):
            r = subprocess.run([sys.executable, "-m", "colabfold_opt", *argv], capture_output=True, text=True, env=self.env())
            self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, (argv, r.stderr)); self.assertNotIn("Traceback", r.stderr, argv)
            self.assertIn(self.WORDS, r.stderr, argv); self.assertIn("installed v0.2.5 at " + self.core, r.stderr, argv)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "o")))

    def test_trigger_route_refuses_by_name_no_stock_body(self):
        code = "import colabfold_opt._autoload as a; a.install(); import colabfold.batch as b; b.run(queries=[('q', 'A' * 700, None)], result_dir='x')"
        for extra in ({"COLABFOLD_OPT": "fast"}, {"COLABFOLD_OPT": "exact"}, {"COLABFOLD_OPT": "big"}):
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=self.env(**extra))
            self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, (extra, r.stderr)); self.assertIn(self.WORDS, r.stderr, extra)
            self.assertNotIn("Traceback", r.stderr, extra); self.assertFalse(os.path.exists(os.path.join(self.tmp, "mark")), f"the stock body ran {extra}")

    def test_in_process_route_refuses_by_name(self):
        for call in ("colabfold_opt.enable('fast', queries=[('q', 'A' * 700, None)])", "colabfold_opt.check('exact')", "colabfold_opt.status()"):
            r = subprocess.run([sys.executable, "-c", f"import colabfold_opt; {call}; print('returned')"], capture_output=True, text=True, env=self.env())
            self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, (call, r.stderr)); self.assertNotIn("returned", r.stdout, call)
            self.assertIn(self.WORDS, r.stderr, call); self.assertNotIn("Traceback", r.stderr, call)


if __name__ == "__main__":
    unittest.main()
