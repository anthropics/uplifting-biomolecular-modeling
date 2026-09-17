"""The seed lever (L1, opt/af2ig_opt/seed.py): the patch applies to the kit's driver and changes exactly the three seed sites; a run
without --seed never builds the seeded copy; --seed N runs the seeded copy with `-seed N` (both routes), records it in the manifest and the
lines; a checkout the patch does not apply to refuses (nothing runs)."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

from af2ig_opt import cli, seed, stack

from . import _stubs

REAL_DRIVER = os.path.join(_stubs.TREE, "opt", "forward", "af2ig_kit", "patches", "patched_files", "af2_initial_guess", "predict_pdb.py")


class TestSeedPatch(unittest.TestCase):
    def test_patch_yields_seeded_driver(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "af2_initial_guess"); os.makedirs(src); shutil.copyfile(REAL_DRIVER, os.path.join(src, "predict_pdb.py"))
            with unittest.mock.patch.dict(os.environ, {"TMPDIR": os.path.join(tmp, "tmp")}):
                os.makedirs(os.path.join(tmp, "tmp")); tempfile.tempdir = None
                info = seed.seeded_dir(src)
            self.assertEqual(sorted(info), ["dir", "driver"])
            self.assertTrue(info["dir"].startswith(os.path.join(tmp, "tmp")), info["dir"])              # a temp dir, never the run's outputs
            self.assertFalse(info["dir"].startswith(src))
            before = open(REAL_DRIVER, encoding="utf-8").read().splitlines(); after = open(info["driver"], encoding="utf-8").read().splitlines()
            self.assertEqual(len(after), len(before) + 1)                                   # one argparse line added
            changed = [(a, b) for a, b in zip(before, [l for l in after if not l.startswith('parser.add_argument( "-seed"')]) if a != b]
            self.assertEqual(len(changed), 3, changed)                                       # the three seed sites, nothing else
            self.assertTrue(all("args.seed" in b and ("random_seed=0" in a or "PRNGKey(0)" in a) for a, b in changed), changed)
            self.assertEqual(sum(1 for l in after if 'parser.add_argument( "-seed", type=int, default=0' in l), 1)
            self.assertEqual(_stubs.sha256(os.path.join(src, "predict_pdb.py")), _stubs.sha256(REAL_DRIVER))      # the source checkout untouched
            seed.remove(info); self.assertFalse(os.path.exists(os.path.dirname(info["dir"])))       # removed on request, nothing left behind
            tempfile.tempdir = None

    def test_patch_applies_without_fuzz_and_the_seeded_driver_compiles(self):
        """0.8.2 (the eval's item #22): levers/04_seed_switch.diff was stale against the kit's own patched driver — `patch` applied it with fuzz and
        its first hunk landed the `-seed` argument inside the two-line `-af2_dir` call, so every `pred --seed N` died on a SyntaxError at
        predict_pdb.py:93 (items 0/14). The diff is regenerated against the current driver: it applies with fuzz 0 and no reject, the seeded
        driver compiles, and the `-seed` switch (int, default 0) feeds exactly the three seed sites."""
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "af2_initial_guess"); os.makedirs(src); shutil.copyfile(REAL_DRIVER, os.path.join(src, "predict_pdb.py"))
            dry = subprocess.run(["patch", "-p2", "--dry-run", "--fuzz=0", "-i", seed.PATCH], cwd=src, capture_output=True, text=True)
            self.assertEqual(dry.returncode, 0, dry.stdout + dry.stderr)                     # every hunk's context is the driver's own text: nothing needs fuzz
            self.assertNotIn("fuzz", dry.stdout + dry.stderr); self.assertNotIn("FAILED", dry.stdout + dry.stderr)
            with unittest.mock.patch.dict(os.environ, {"TMPDIR": os.path.join(tmp, "tmp")}):
                os.makedirs(os.path.join(tmp, "tmp")); tempfile.tempdir = None
                info = seed.seeded_dir(src)
            try:
                self.assertEqual([f for f in os.listdir(info["dir"]) if f.endswith((".rej", ".orig"))], [])   # no reject, no backup: a clean application
                pc = subprocess.run([sys.executable, "-m", "py_compile", info["driver"]], capture_output=True, text=True)
                self.assertEqual(pc.returncode, 0, pc.stdout + pc.stderr)                    # the seeded driver is a Python program (0.8.1's was not: SyntaxError at :93)
                text = open(info["driver"], encoding="utf-8").read(); lines = text.splitlines()
                k = next(i for i, l in enumerate(lines) if l.startswith('parser.add_argument( "-seed", type=int, default=0'))
                self.assertTrue(lines[k - 1].rstrip().endswith(")") and lines[k + 1].startswith('parser.add_argument( "-timers"'), lines[k - 1:k + 2])   # after the COMPLETE -af2_dir call, before -timers
                self.assertEqual(text.count("random_seed=args.seed"), 1); self.assertEqual(text.count("jax.random.PRNGKey(args.seed)"), 2)
                self.assertEqual([l for l in lines if ("random_seed=0" in l or "PRNGKey(0)" in l) and not l.lstrip().startswith(("af2_util.check_residue_distances", "jax.random.PRNGKey(0), scores"))], [])   # no fixed seed left at a code site (the module docstring names stock's constants)
            finally:
                seed.remove(info); tempfile.tempdir = None

    def test_a_garbled_seeded_driver_refuses_by_name(self):
        """The lever is fail-closed past `patch` too: a seeded copy that does not compile raises (the caller exits EXIT_FAIL with the line) and is removed — nothing runs it."""
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "af2_initial_guess"); os.makedirs(src); shutil.copyfile(REAL_DRIVER, os.path.join(src, "predict_pdb.py"))
            real_run = subprocess.run
            def garble(argv, **kw):                                   # patch applies, then the copy is corrupted the way 0.8.1's stale hunk corrupted it
                r = real_run(argv, **kw)
                drv = os.path.join(kw["cwd"], "predict_pdb.py"); t = open(drv, encoding="utf-8").read()
                open(drv, "w", encoding="utf-8").write(t.replace('parser.add_argument( "-timers"', 'parser.add_argument( "-af2_dir", type=str,\nparser.add_argument( "-timers"', 1))
                return r
            before = set(os.listdir(tempfile.gettempdir()))
            with unittest.mock.patch.object(seed.subprocess, "run", garble), self.assertRaises(RuntimeError) as cm:
                seed.seeded_dir(src)
            self.assertIn("does not compile", str(cm.exception))
            self.assertEqual({d for d in os.listdir(tempfile.gettempdir()) if d.startswith("af2ig_seeded_")} - before, set())

    def test_patch_does_not_apply_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "af2_initial_guess"); os.makedirs(src)
            with open(os.path.join(src, "predict_pdb.py"), "w") as fh:
                fh.write("print('not the driver')\n")
            before = set(os.listdir(tempfile.gettempdir()))
            with self.assertRaises(RuntimeError):
                seed.seeded_dir(src)
            self.assertEqual({d for d in os.listdir(tempfile.gettempdir()) if d.startswith("af2ig_seeded_")} - before, set())   # the failed copy is removed too


class TestSeedCli(unittest.TestCase):
    def _stub_seeded(self, af2ig_dir):                       # the stub checkout is not the pinned driver: stand in for the copy+patch step, keep its record shape (a temp dir)
        dest = os.path.join(tempfile.mkdtemp(prefix="af2ig_seeded_"), "af2_initial_guess"); shutil.copytree(af2ig_dir, dest)
        self.seeded_dirs.append(dest)
        return {"dir": dest, "driver": os.path.join(dest, "predict_pdb.py")}

    def test_no_seed_never_builds_the_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); in_dir, _ = _stubs.inputs(tmp, 2); out = os.path.join(tmp, "out")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "off", "--pdbdir", in_dir, "--out", out], env)
            self.assertEqual(rc, 0, err); self.assertFalse(os.path.exists(os.path.join(out, "seeded_checkout")))
            self.assertNotIn("-seed", _stubs.records(so)[0]["argv"])
            self.assertNotIn("seed=", err)

    def test_seed_zero_is_the_unseeded_driver_and_never_builds_the_copy(self):
        """0.8.2: seed 0 through the lever is the unseeded driver, bitwise (seed.py) — so `--seed 0` short-circuits: no seeded copy is built,
        no `-seed` flag reaches the driver, no seed word is printed; the run is the mode's own line on both routes (the eval's item #22)."""
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); in_dir, _ = _stubs.inputs(tmp, 2)
            for mode in ("off", "exact"):
                out = os.path.join(tmp, "out0_" + mode)
                buf, obuf = io.StringIO(), io.StringIO()
                built = unittest.mock.Mock(side_effect=AssertionError("--seed 0 built the seeded copy"))
                stack._PINS_MOD = None
                try:
                    with unittest.mock.patch.dict(os.environ, env, clear=True), unittest.mock.patch.object(cli._seed, "seeded_dir", built), contextlib.redirect_stderr(buf), \
                            contextlib.redirect_stdout(obuf):
                        rc = cli.main(["pred", "--mode", mode, "--seed", "0", "--pdbdir", in_dir, "--out", out])
                finally:
                    stack._PINS_MOD = None
                err, so = buf.getvalue(), obuf.getvalue()
                self.assertEqual(rc, 0, err); built.assert_not_called()
                argv = _stubs.records(so)[0]["argv"]
                self.assertNotIn("-seed", argv)                                                   # the driver ran unseeded: its own constants
                self.assertEqual(next(a for a in argv if a.endswith("predict_pdb.py")), os.path.join(env["AF2IG_DIR"], "predict_pdb.py"))   # the checkout's own driver, not a copy
                self.assertNotIn("seed=", err)                                                    # no seed word on the STOCK / EXIT lines: this is not the af2ig-seeded arm

    def test_seed_runs_the_seeded_copy_on_both_routes(self):
        self.seeded_dirs = []
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); in_dir, _ = _stubs.inputs(tmp, 2)
            for mode, arm in (("off", "af2ig-seeded"), ("exact", "exact")):
                out = os.path.join(tmp, "out_" + mode)
                rc, _, err = _stubs.run_cli(["pred", "--mode", mode, "--pdbdir", in_dir, "--out", os.path.join(tmp, "out_plain_" + mode)], env); self.assertEqual(rc, 0, err)
                buf, obuf = io.StringIO(), io.StringIO()      # in-process (the mock must reach the CLI): the driver child is the stub, as in run_cli
                stack._PINS_MOD = None   # stock/check_pins.py is imported once per process by path: re-read from the stub tree, then dropped again
                try:
                    with unittest.mock.patch.dict(os.environ, env, clear=True), unittest.mock.patch.object(cli._seed, "seeded_dir", self._stub_seeded), contextlib.redirect_stderr(buf), \
                            contextlib.redirect_stdout(obuf):
                        rc = cli.main(["pred", "--mode", mode, "--seed", "7", "--pdbdir", in_dir, "--out", out])
                finally:
                    stack._PINS_MOD = None
                err, so = buf.getvalue(), obuf.getvalue()
                self.assertEqual(rc, 0, err)
                records = _stubs.records(so); argv = records[0]["argv"]                             # the driver's own proc_start record: its argv
                k = argv.index("-seed"); self.assertEqual(argv[k + 1], "7")
                drv = next(a for a in argv if a.endswith("predict_pdb.py")); self.assertEqual(drv, os.path.join(self.seeded_dirs[-1], "predict_pdb.py"))
                self.assertFalse(drv.startswith(out)); self.assertFalse(os.path.exists(os.path.dirname(self.seeded_dirs[-1])))   # a temp copy, removed after the run
                self.assertEqual(sorted(os.listdir(out)), sorted(os.listdir(os.path.join(tmp, "out_plain_" + mode))))   # a seeded run leaves exactly a plain run's files
                self.assertEqual([r["argv"][r["argv"].index("-seed") + 1] for r in records if r["kind"] == "proc_start"], ["7"])   # the argv fact (the driver's own proc_start record)
                self.assertRegex(err, r"EXIT pid=\d+ mode=%s .* seed=7" % mode)
                if mode == "off":
                    self.assertRegex(err, r"STOCK line='.* -seed 7' det=off stripped=\S+ seed=7")


if __name__ == "__main__":
    unittest.main()
