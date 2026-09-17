"""The autoload route in a throwaway venv with the package installed editable from a temp copy of opt/ (`pip install -e <copy>`: the
tree itself is never written to): the .pth lands in site-packages; nothing beyond `colabfold_opt` and `colabfold_opt._autoload` is
imported at interpreter start (no jax, no colabfold, no numpy, no other module of the package), with or without COLABFOLD_OPT; the finder
is installed for every COLABFOLD_OPT value but off; the trigger (a stub colabfold.batch on PYTHONPATH) hooks `run` and, when the gates
refuse or the mode is unknown, prints NOT ACTIVE and exits 3 before the stock body runs; the launch forms (the import form
fires the finder, `-m` does not); the console script; the wheel carries the .pth at its root. Network: pip needs setuptools for the
build (one venv per test module)."""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPORT_PROBE = ("import sys; print(sorted(m for m in sys.modules if m.split('.')[0] in "
                "('jax', 'jaxlib', 'colabfold', 'alphafold', 'numpy', 'haiku', 'af2_pallas_attn', 'af2_flash_pallas', 'colabfold_opt')));"
                "print(any(type(f).__name__ == 'Finder' for f in sys.meta_path))")
AT_START = "['colabfold_opt', 'colabfold_opt._autoload']"
STUB_BATCH = textwrap.dedent('''\
    import os
    def run(queries, result_dir, num_models=5, is_complex=True, data_dir=None, **kw):
        open(os.path.join(os.environ["STUB_MARK"]), "w").write("stock body ran")
        return 0
    def main():
        return run(queries=[("q", "A" * 800, None, None)], result_dir=os.environ["STUB_RES"], data_dir="/nonexistent")
    if __name__ == "__main__":
        import sys
        sys.exit(main())
    ''')


class TestStartup(unittest.TestCase):
    venv = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.venv = os.path.join(cls.tmp, "venv")
        subprocess.run([sys.executable, "-m", "venv", cls.venv], check=True, capture_output=True)
        cls.py = os.path.join(cls.venv, "bin", "python")
        tree_copy = os.path.join(cls.tmp, "tree")                        # the installs and builds write beside their source: never the tree
        shutil.copytree(os.path.dirname(OPT), tree_copy, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", ".pytest_cache", "*.whl", "src"))
        cls.opt_copy = os.path.join(tree_copy, "opt")
        import opt_core                                                  # the core this process imports, installed into the venv first (a kit = the core + itself)
        core_dir = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))
        r = subprocess.run([cls.py, "-m", "pip", "install", "-q", "-e", core_dir, "-e", cls.opt_copy], capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"editable install failed (pip needs setuptools from the index): {r.stderr[-500:]}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def env(self, **kw):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "PYTHONPATH"))}
        e.update(kw)
        return e

    def run_py(self, code, **env):
        return subprocess.run([self.py, "-c", code], capture_output=True, text=True, env=self.env(**env))

    def test_pth_installed(self):
        pths = glob.glob(os.path.join(self.venv, "lib", "python*", "site-packages", "colabfold_opt_autoload.pth"))
        self.assertEqual(len(pths), 1)
        tree_pth = open(os.path.join(self.opt_copy, "colabfold_opt_autoload.pth")).read()
        self.assertEqual(open(pths[0]).read(), tree_pth)                             # the installed .pth is the tree's file, byte for byte (the build backend ships it)
        self.assertIn("import colabfold_opt._autoload", tree_pth)

    def test_nothing_imported_at_start(self):
        r = self.run_py(IMPORT_PROBE)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.splitlines()[0], AT_START)                          # the package and _autoload alone: no modes/stack
        self.assertEqual(r.stdout.splitlines()[1], "False")
        r = self.run_py(IMPORT_PROBE, COLABFOLD_OPT="fast")
        self.assertEqual(r.stdout.splitlines()[0], AT_START); self.assertEqual(r.stdout.splitlines()[1], "True")
        r = self.run_py(IMPORT_PROBE, COLABFOLD_OPT="off")
        self.assertEqual(r.stdout.splitlines()[1], "False")
        for value in ("turbo", "exact"):                                              # any non-off value installs the finder (an unknown one is refused at the trigger import, by name)
            r = self.run_py(IMPORT_PROBE, COLABFOLD_OPT=value)
            self.assertEqual((r.stdout.splitlines()[0], r.stdout.splitlines()[1], r.stderr), (AT_START, "True", ""), value)

    def test_trigger_hooks_run_and_refuses_before_the_stock_body(self):
        stub = os.path.join(self.tmp, "stubpath"); os.makedirs(os.path.join(stub, "colabfold"), exist_ok=True)
        open(os.path.join(stub, "colabfold", "__init__.py"), "w").write("")
        open(os.path.join(stub, "colabfold", "batch.py"), "w").write(STUB_BATCH)
        mark = os.path.join(self.tmp, "mark.txt"); res = os.path.join(self.tmp, "res")
        code = "import colabfold.batch as b; print(getattr(b.run, '_colabfold_opt_hook', False)); b.main()"
        r = self.run_py(code, COLABFOLD_OPT="fast", PYTHONPATH=stub, STUB_MARK=mark, STUB_RES=res)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertEqual(r.stdout.strip(), "True")
        self.assertIn("[colabfold-opt] NOT ACTIVE: ", r.stderr); self.assertIn("stock pin: colabfold is not installed", r.stderr)
        self.assertFalse(os.path.exists(mark))                                       # stock never ran silently under COLABFOLD_OPT
        r = self.run_py(code, PYTHONPATH=stub, STUB_MARK=mark, STUB_RES=res)          # no COLABFOLD_OPT: untouched
        self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(r.stdout.strip(), "False"); self.assertTrue(os.path.exists(mark))
        os.remove(mark)
        r = self.run_py(code, COLABFOLD_OPT="fast", COLABFOLD_OPT_SIZE_RULE="maybe", PYTHONPATH=stub, STUB_MARK=mark, STUB_RES=res)
        self.assertEqual(r.returncode, 3); self.assertIn("NOT ACTIVE: undeclared variable(s) COLABFOLD_OPT_SIZE_RULE (the names this kit reads: ", r.stderr); self.assertFalse(os.path.exists(mark))   # a name under the prefix nothing reads: refused at the trigger import
        r = self.run_py(code, COLABFOLD_OPT="exact", PYTHONPATH=stub, STUB_MARK=mark, STUB_RES=res)   # a kit mode: hooked and gated like fast (no colabfold here: refused by name, stock never runs)
        self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("[colabfold-opt] NOT ACTIVE: ", r.stderr); self.assertIn("stock pin: colabfold is not installed", r.stderr); self.assertFalse(os.path.exists(mark))
        r = self.run_py(code, COLABFOLD_OPT="turbo", PYTHONPATH=stub, STUB_MARK=mark, STUB_RES=res)
        self.assertEqual(r.returncode, 3); self.assertIn("NOT ACTIVE: 'turbo' is not a mode (off|exact|fast", r.stderr); self.assertFalse(os.path.exists(mark))

    def test_find_spec_probe_then_import_still_fires(self):
        """A bare `importlib.util.find_spec('colabfold.batch')` probe before the real import does not disarm the finder: the import
        that follows is hooked and refused (rc 3, the kit's own line), the stock body never runs; a probe of another name is inert."""
        stub = os.path.join(self.tmp, "stubpath3"); os.makedirs(os.path.join(stub, "colabfold"), exist_ok=True)
        open(os.path.join(stub, "colabfold", "__init__.py"), "w").write("")
        open(os.path.join(stub, "colabfold", "batch.py"), "w").write(STUB_BATCH)
        mark = os.path.join(self.tmp, "mark3.txt"); res = os.path.join(self.tmp, "res3")
        code = ("import importlib.util, sys; from colabfold_opt import _autoload; "
                "s1 = importlib.util.find_spec('colabfold.batch'); s2 = importlib.util.find_spec('json'); "
                "print(sum(isinstance(f, _autoload.Finder) for f in sys.meta_path), s1 is not None, s2 is not None); "
                "import colabfold.batch as b; print(getattr(b.run, '_colabfold_opt_hook', False)); b.main()")
        r = self.run_py(code, COLABFOLD_OPT="fast", PYTHONPATH=stub, STUB_MARK=mark, STUB_RES=res)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertEqual(r.stdout.split("\n")[:2], ["1 True True", "True"])           # the finder still installed after both probes; the import hooked
        self.assertIn("[colabfold-opt] NOT ACTIVE: ", r.stderr); self.assertFalse(os.path.exists(mark))
        r = self.run_py(code, COLABFOLD_OPT="turbo", PYTHONPATH=stub, STUB_MARK=mark, STUB_RES=res)   # an unknown selection after a probe: refused by name
        self.assertEqual(r.returncode, 3); self.assertIn("NOT ACTIVE: 'turbo' is not a mode (off|exact|fast", r.stderr); self.assertFalse(os.path.exists(mark))

    def test_unknown_selection_exits_at_the_trigger_import(self):
        """COLABFOLD_OPT outside the mode table (a mistyped value): the bare import of the trigger prints the kit's NOT ACTIVE line and
        exits EXIT_NOT_ACTIVE (3) — never print-and-return; the stock body is unreachable. A mode of the table (any case, padded) hooks."""
        stub = os.path.join(self.tmp, "stubpath4"); os.makedirs(os.path.join(stub, "colabfold"), exist_ok=True)
        open(os.path.join(stub, "colabfold", "__init__.py"), "w").write("")
        open(os.path.join(stub, "colabfold", "batch.py"), "w").write(STUB_BATCH)
        mark = os.path.join(self.tmp, "mark4.txt"); res = os.path.join(self.tmp, "res4")
        for value, line in (("turbo", "NOT ACTIVE: 'turbo' is not a mode (off|exact|fast"), (" Exact", None), ("FAST ", None)):
            r = self.run_py("import colabfold.batch; print('imported')", COLABFOLD_OPT=value, PYTHONPATH=stub, STUB_MARK=mark, STUB_RES=res)
            if line is None:                                                          # a known selection (normalised: strip + lower) imports and hooks; nothing runs
                self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(r.stdout.strip(), "imported"); continue
            self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("[colabfold-opt] " + line, r.stderr); self.assertEqual(r.stdout, "")
            self.assertFalse(os.path.exists(mark))

    def test_undeclared_variable_name_is_refused_at_the_trigger_import(self):
        """A variable under the kit's prefix that nothing reads (COLABFOLD_OPT_MODE=fast, a mistyped name) installs the finder whatever
        COLABFOLD_OPT says; the trigger import prints the NOT ACTIVE line naming it and exits 3 — the stock body never runs."""
        stub = os.path.join(self.tmp, "stubpath5"); os.makedirs(os.path.join(stub, "colabfold"), exist_ok=True)
        open(os.path.join(stub, "colabfold", "__init__.py"), "w").write("")
        open(os.path.join(stub, "colabfold", "batch.py"), "w").write(STUB_BATCH)
        mark = os.path.join(self.tmp, "mark5.txt"); res = os.path.join(self.tmp, "res5")
        for extra in ({"COLABFOLD_OPT_MODE": "fast"}, {"COLABFOLD_OPT": "fast", "COLABFOLD_OPTS": "1"}, {"COLABFOLD_OPT": "off", "COLABFOLD_OPT_SIZE": "on"}):
            r = self.run_py("import colabfold.batch; print('imported')", PYTHONPATH=stub, STUB_MARK=mark, STUB_RES=res, **extra)
            name = [k for k in extra if k not in ("COLABFOLD_OPT",)][0]
            self.assertEqual(r.returncode, 3, (extra, r.stderr)); self.assertEqual(r.stdout, "")
            self.assertIn(f"[colabfold-opt] NOT ACTIVE: undeclared variable(s) {name} (the names this kit reads: COLABFOLD_OPT, ", r.stderr)
            self.assertFalse(os.path.exists(mark))
        r = self.run_py(IMPORT_PROBE, COLABFOLD_OPT_DATA_DIR="/nowhere")                 # a declared name alone: no finder, nothing fires
        self.assertEqual((r.returncode, r.stdout.splitlines()[1]), (0, "False"), r.stderr)

    def test_launch_forms(self):
        """The import form (the console script's body) fires the finder; `python -m colabfold.batch` runs the stock body silently under
        COLABFOLD_OPT=fast (runpy executes the module through the loader's get_code, not exec_module) — which is why the launcher never
        uses it (stock_pred.cli_argv)."""
        from colabfold_opt import stock_pred
        stub = os.path.join(self.tmp, "stubpath2"); os.makedirs(os.path.join(stub, "colabfold"), exist_ok=True)
        open(os.path.join(stub, "colabfold", "__init__.py"), "w").write("")
        open(os.path.join(stub, "colabfold", "batch.py"), "w").write(STUB_BATCH)
        mark = os.path.join(self.tmp, "mark2.txt"); res = os.path.join(self.tmp, "res2")
        env = self.env(COLABFOLD_OPT="fast", PYTHONPATH=stub, STUB_MARK=mark, STUB_RES=res)
        sys.path.insert(0, stub); self.addCleanup(sys.path.remove, stub)               # colabfold (the stub) importable: the import form exists
        argv = stock_pred.cli_argv(self.py)                                            # no colabfold_batch beside the venv's python: the import form
        self.assertEqual(argv, [self.py, "-c", stock_pred.IMPORT_FORM])
        r = subprocess.run(argv, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("[colabfold-opt] NOT ACTIVE: ", r.stderr); self.assertFalse(os.path.exists(mark))
        r = subprocess.run([self.py, "-m", "colabfold.batch"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0); self.assertNotIn("NOT ACTIVE", r.stderr); self.assertTrue(os.path.exists(mark))   # silent stock: the form is never used

    def test_console_script(self):
        exe = os.path.join(self.venv, "bin", "colabfold-opt")
        self.assertTrue(os.path.isfile(exe))
        r = subprocess.run([exe, "--help"], capture_output=True, text=True, env=self.env())
        self.assertEqual(r.returncode, 0); self.assertIn("usage: colabfold-opt <command>", r.stdout)
        r = subprocess.run([exe, "check", "--mode", "off"], capture_output=True, text=True, env=self.env())
        self.assertEqual(r.returncode, 0); self.assertIn("NOT ACTIVE mode=off (stock: nothing applied)", r.stderr)
        r = subprocess.run([exe, "pred", "--mode", "turbo"], capture_output=True, text=True, env=self.env())
        self.assertEqual(r.returncode, 2); self.assertIn("'turbo' is not a mode (off|exact|fast", r.stderr)
        r = subprocess.run([self.py, "-m", "colabfold_opt", "check", "--mode", "off"], capture_output=True, text=True, env=self.env())
        self.assertEqual(r.returncode, 0)

    def test_wheel_carries_the_pth_at_root(self):
        wd = os.path.join(self.tmp, "wheels")
        r = subprocess.run([self.py, "-m", "pip", "wheel", "-q", "--no-deps", "-w", wd, self.opt_copy], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        whl = glob.glob(os.path.join(wd, "colabfold_opt-*.whl"))
        self.assertEqual(len(whl), 1)
        with zipfile.ZipFile(whl[0]) as z:
            names = z.namelist()
            self.assertIn("colabfold_opt_autoload.pth", names)
            record = [n for n in names if n.endswith(".dist-info/RECORD")][0]
            self.assertIn("colabfold_opt_autoload.pth,sha256=", z.read(record).decode())
            self.assertFalse([n for n in names if n.startswith("colabfold_opt/forward")])   # the kit is not in the wheel: it is resolved from the tree
