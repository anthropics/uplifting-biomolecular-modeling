"""The stock caller: the clean environment, the proof, the execv into the driver, the refusals."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

from af2ig_opt import cli, stack, stock_cli
from . import _stubs


class TestStockCaller(unittest.TestCase):
    def test_clean_env(self):
        env = {"AF2IG_OPT": "exact", "AF2IG_OPT_FORCE": "1", "JAX_COMPILATION_CACHE_DIR": "/c", "AF2_SUBBATCH_SIZE": "0", "CUDA_MPS_PIPE_DIRECTORY": "/x",
               "XLA_FLAGS": "--xla_gpu_autotune_level=0", "AF2_PARAMS": "/p", "AF2IG_DIR": "/d", "PATH": "/bin"}
        prefixes = ["AF2IG_OPT", "JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "AF2_SUBBATCH_SIZE", "CUDA_MPS_"]
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(stock_cli.stock_environment(stack.pins())[0], prefixes)              # stock/PINS.json stock_environment.must_be_absent_prefixes
            out, stripped = cli.child_env(None)
        self.assertEqual(stripped, ["AF2IG_OPT", "AF2IG_OPT_FORCE", "AF2_SUBBATCH_SIZE", "CUDA_MPS_PIPE_DIRECTORY", "JAX_COMPILATION_CACHE_DIR"])
        self.assertEqual(set(out), {"XLA_FLAGS", "AF2_PARAMS", "AF2IG_DIR", "PATH"})

    def test_stock_route_passes_only_inputs_and_outputs(self):
        """The stock route adds nothing to the driver's defaults: its argv is the I/O switches (PINS.json stock_environment) and nothing else."""
        argv = cli.driver_args("/in", "/out", "/params", "/out/timers.jsonl")
        self.assertEqual(argv, ["-pdbdir", "/in", "-outpdbdir", "/out/pdbs", "-scorefilename", "/out/out.sc", "-checkpoint_name", "/out/check.point",
                                "-af2_dir", "/params", "-timers", "/out/timers.jsonl"])
        self.assertEqual([a for a in argv if a.startswith("-")], ["-pdbdir", "-outpdbdir", "-scorefilename", "-checkpoint_name", "-af2_dir", "-timers"])

    def test_proof_flags(self):
        pins = {"stock_environment": {"must_be_absent_prefixes": ["AF2IG_OPT"], "recipe_exceptions": ["XLA_FLAGS"], "pass_through": []}}
        P = stock_cli.proof(pins, 0, sys.executable, [sys.executable, "-I", "-u", "d.py", "-pdbdir", "x", "-fast"], [])
        self.assertEqual(P["lever_flags_in_argv"], ["-fast"])
        self.assertFalse(P["ok"])

    def test_fresh_interpreter_runs_the_driver(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            in_dir, names = _stubs.inputs(tmp, 1)
            out = os.path.join(tmp, "out"); os.makedirs(out)
            driver = os.path.join(env["AF2IG_DIR"], "predict_pdb.py")
            cmd = [sys.executable, "-I", "-m", "af2ig_opt.stock_cli", "--pins", os.path.join(tree, "stock", "PINS.json"),
                   "--det", "0", "--python", sys.executable, "--driver", driver, "--", "-pdbdir", in_dir, "-outpdbdir", os.path.join(out, "pdbs"),
                   "-scorefilename", os.path.join(out, "out.sc"), "-checkpoint_name", os.path.join(out, "check.point"), "-af2_dir", env["AF2_PARAMS"], "-timers", os.path.join(out, "timers.jsonl")]
            env = {k: v for k, v in env.items() if not k.startswith("AF2IG_OPT")}   # what cli.child_env hands the caller: the package's switches stripped
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            prefixes = stock_cli.stock_environment(json.load(open(os.path.join(tree, "stock", "PINS.json"))))[0]
            self.assertIn(f"[af2ig-opt stock] ENV-CLEAN ok: absent={','.join(prefixes)} kit_modules=none kit_dirs=none no_user_site=True autoload=none "
                          f"package_modules=af2ig_opt det=0\n", r.stderr)                            # the clean-process census (opt_core.stock_proof.clean_sentence), once, before execv
            self.assertEqual(r.stderr.count("ENV-CLEAN"), 1); self.assertNotIn("NOT STOCK", r.stderr)
            self.assertTrue(os.path.isfile(os.path.join(out, "pdbs", names[0] + "_af2pred.pdb")))
            first = json.loads(open(os.path.join(out, "timers.jsonl")).readline())                 # the driver's own -timers records at the path this test handed it
            self.assertEqual(first["kind"], "proc_start"); self.assertEqual(first["argv"][0], driver)   # execv reached the driver itself, under the interpreter's -I -u
            self.assertNotIn("-fast", first["argv"])
            self.assertEqual(sorted(os.listdir(out)), ["check.point", "out.sc", "pdbs", "timers.jsonl"])   # the caller writes no file of its own
            # a forbidden name present -> refused before anything runs
            bad = dict(env, JAX_COMPILATION_CACHE_DIR="/tmp/c")
            out2 = os.path.join(tmp, "out2"); os.makedirs(out2)
            cmd2 = list(cmd); cmd2[cmd2.index("-outpdbdir") + 1] = os.path.join(out2, "pdbs")
            r = subprocess.run(cmd2, env=bad, capture_output=True, text=True)
            self.assertEqual(r.returncode, stock_cli.EXIT_REFUSED)
            self.assertIn("REFUSED", r.stderr)
            self.assertIn("[af2ig-opt stock] NOT STOCK: forbidden env ['JAX_COMPILATION_CACHE_DIR'], kit modules [], kit dirs [], autoload [], kit sitecustomize None, "
                          "torch loaded False; package modules beyond the caller [], tree dirs on sys.path [], lever flags in argv []; nothing ran", r.stderr)
            self.assertNotIn("ENV-CLEAN", r.stderr)
            self.assertEqual(os.listdir(out2), [])                                                     # refused: nothing ran, nothing written

    def test_refusals_by_name_before_the_driver_starts(self):
        """A kit directory on sys.path, an armed autoload finder, a package module beyond the caller's two: each refuses with exit 3, the
        NOT STOCK line names it, and the driver never starts (os.execv, the start, is never reached)."""
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            pins_path = os.path.join(tree, "stock", "PINS.json")
            kit_dir = os.path.join(tree, "opt", "forward", "af2ig_kit"); os.makedirs(kit_dir)          # the stand-in tree's kit directory (PINS.json kit.dir), on disk so a sys.path entry can point at it

            def run(tag):
                argv = ["--pins", pins_path, "--det", "0", "--python", sys.executable,
                        "--driver", os.path.join(env["AF2IG_DIR"], "predict_pdb.py"), "--", "-pdbdir", "/in", "-outpdbdir", "/out/pdbs"]
                err = io.StringIO()
                clean = {k: v for k, v in os.environ.items() if not k.startswith(("AF2IG_OPT", "AF2_SUBBATCH", "JAX_COMPILATION", "JAX_PERSISTENT", "CUDA_MPS_"))}
                with unittest.mock.patch.dict(os.environ, clean, clear=True), unittest.mock.patch.object(stock_cli.os, "execv") as start, contextlib.redirect_stderr(err):
                    rc = stock_cli.main(argv)
                return rc, err.getvalue(), start

            # this test process holds af2ig_opt.cli, af2ig_opt.stack, … (imported above): package modules beyond the caller's two, by name
            rc, err, start = run("modules")
            self.assertEqual(rc, stock_cli.EXIT_REFUSED); start.assert_not_called()
            self.assertIn("[af2ig-opt stock] NOT STOCK: ", err); self.assertIn("package modules beyond the caller [", err); self.assertIn("'af2ig_opt.cli'", err)
            self.assertIn("; nothing ran", err); self.assertIn("[af2ig-opt] stock_cli REFUSED: ", err); self.assertNotIn("ENV-CLEAN", err)
            # a kit directory on sys.path: named by the core proof (kit dirs) and by the caller's own tree check
            sys.path.insert(0, kit_dir)
            try:
                rc, err, start = run("kitdir")
            finally:
                sys.path.remove(kit_dir)
            self.assertEqual(rc, stock_cli.EXIT_REFUSED); start.assert_not_called()
            self.assertIn(f"kit dirs ['{kit_dir}']", err); self.assertIn(f"tree dirs on sys.path ['{kit_dir}']", err)
            # an armed autoload finder on the meta path: named by the core proof (autoload)
            Finder = type("StubKitFinder", (), {"armed": True, "find_spec": lambda self, *a, **k: None}); Finder.__module__ = "stubkit._autoload"
            finder = Finder(); sys.meta_path.insert(0, finder)
            try:
                rc, err, start = run("autoload")
            finally:
                sys.meta_path.remove(finder)
            self.assertEqual(rc, stock_cli.EXIT_REFUSED); start.assert_not_called()
            self.assertIn("autoload ['StubKitFinder']", err)


if __name__ == "__main__":
    unittest.main()
