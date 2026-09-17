"""The launcher's building blocks in-process (driver.strip_env / driver.compose / driver.launch: every must-be-absent prefix and the
package's own variables stripped -- no arm carries a kit or lever variable; the argv per route; the run through the tree's runner, lines
teed, wall-clock deadline) and the two launchers end to end in their own process on the stub stack (the environment proof -- clean / a
forbidden variable / a kit directory on the path; the stock proof file; the kit arm installing its mode's levers through the package's
installer; nothing written but the outputs)."""

import os
import subprocess
import sys
import tempfile
import unittest

from . import _stubs
from colabdesign_opt import driver, modes, names, report, stack




# ============================================================ driver.strip_env / compose / launch: pure-function unit tests
class TestDriverUnit(unittest.TestCase):
    def test_strip_env(self):
        base = {"PATH": "/bin", "AF2M_LEVERS": "nosub", "AF2M_SUBBATCH_GRAD": "4", "AF_PALLAS_ATTN": "1", "COLABDESIGN_OPT": "fast", "COLABDESIGN_OPT_LEVERS": "x",
                "COLABDESIGN_OPT_HOME": "/t", "KEEP_XLA_DEFAULTS": "1", "XLA_FLAGS": "--x", "COLABDESIGN_PARAMS_DIR": "/data/af2_params", "MODEL_OPT": "/t"}
        env, dropped = driver.strip_env(base, ["AF2M_", "AF_PALLAS_", "COLABDESIGN_OPT", "KEEP_XLA_DEFAULTS"], {"AF2M_LEVERS": "nosub,pallas"})
        self.assertEqual(dropped, ["AF2M_LEVERS", "AF2M_SUBBATCH_GRAD", "AF_PALLAS_ATTN", "COLABDESIGN_OPT", "COLABDESIGN_OPT_HOME", "COLABDESIGN_OPT_LEVERS", "KEEP_XLA_DEFAULTS"])
        self.assertEqual(env, {"PATH": "/bin", "XLA_FLAGS": "--x", "COLABDESIGN_PARAMS_DIR": "/data/af2_params", "MODEL_OPT": "/t", "AF2M_LEVERS": "nosub,pallas"})

    @unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
    def test_compose(self):
        base = {"PATH": "/bin", "AF2M_LEVERS": "nosub", "COLABDESIGN_OPT": "off"}
        argv, env, notes = driver.compose(modes.resolve("off"), ["--starting-pdb", "t.pdb"], out_dir="/o", base_env=base)
        self.assertEqual(argv[:4], [sys.executable, "-s", "-m", "colabdesign_opt.stock_launch"])
        self.assertEqual(argv[4:6], ["--pins", stack.pins_path()])
        self.assertEqual(argv[6:], ["--", "--starting-pdb", "t.pdb"])
        self.assertEqual(env, {"PATH": "/bin", "PYTHONDONTWRITEBYTECODE": "1"}); self.assertEqual(notes["env_dropped"], ["AF2M_LEVERS", "COLABDESIGN_OPT"]); self.assertEqual(notes["env_kept"], {})
        for mode in ("fast",):
            argv, env, notes = driver.compose(modes.resolve(mode), ["--starting-pdb", "t.pdb"], out_dir="/o", base_env=base)
            self.assertEqual(argv[3], "colabdesign_opt.kit_launch")
            self.assertEqual(argv[4:8], ["--pins", stack.pins_path(), "--mode", mode]); self.assertEqual(argv[8:], ["--", "--starting-pdb", "t.pdb"])
            self.assertEqual(env, {"PATH": "/bin", "PYTHONDONTWRITEBYTECODE": "1"}); self.assertEqual(notes["env_kept"], {})

    def test_launch_and_timeout(self):
        tmp = tempfile.mkdtemp()
        res = driver.launch([sys.executable, "-c", "import sys; print('a'); print('b', file=sys.stderr); sys.exit(4)"], dict(os.environ), cwd=tmp,
                            log_path=os.path.join(tmp, "ab.log"), timeout=30, echo=lambda s: None)
        self.assertEqual((res["exit_code"], res["timed_out"], sorted(res["lines"])), (4, False, ["a", "b"]))
        res = driver.launch([sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"], dict(os.environ), cwd=tmp,
                            log_path=os.path.join(tmp, "x.log"), timeout=1, echo=lambda s: None)
        self.assertTrue(res["timed_out"]); self.assertNotEqual(res["exit_code"], 0); self.assertEqual(res["lines"], ["x"])
        res = driver.launch([sys.executable, "-c", "print('hello')"], dict(os.environ), cwd=tmp, log_path=os.path.join(tmp, "run.log"), timeout=30, echo=lambda s: None)
        self.assertEqual((res["exit_code"], res["lines"]), (0, ["hello"])); self.assertEqual(open(res["log"]).read(), "hello\n")


# ============================================================ the two launchers end to end, on the stub stack, in their own process
@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
@unittest.skipIf(_stubs.bindcraft_stack_missing(), _stubs.SKIP_BINDCRAFT)
class TestLaunchers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cd_opt_launch_")
        cls.site = _stubs.make_stub_stack(cls.tmp)
        cls.target = _stubs.write_target(os.path.join(cls.tmp, "t.pdb"), ("A",), 108)
        _stubs.make_params(os.path.join(cls.tmp, "params"))
        cls.env = _stubs.child_env(cls.site)

    def design_args(self, out, binder_len=100):
        return ["--", "--starting-pdb", self.target, "--chains", "A", "--binder-len", str(binder_len), "--seed", "0", "--params-dir", os.path.join(self.tmp, "params"), "--out", out]

    def stock(self, out, **env):
        cmd = _stubs.arm_cmd("stock", out + "_probe.json", ["--pins", _stubs.PINS, *self.design_args(out)])
        return subprocess.run(cmd, env=dict(self.env, **env), capture_output=True, text=True, cwd=self.tmp)

    def kit(self, out, mode="fast", binder_len=100, extra=(), **env):
        cmd = _stubs.arm_cmd("kit", out + "_probe.json", ["--pins", _stubs.PINS, "--mode", mode, *extra, *self.design_args(out, binder_len)])
        return subprocess.run(cmd, env=dict(self.env, **env), capture_output=True, text=True, cwd=self.tmp)

    def runs(self, out):
        """The run records of the arm under `out`, by seed (bindcraft.run_design's return values, captured test-side by tests/_stubs.ARM_PROBE)."""
        return _stubs.read_probe(out + "_probe.json")["runs"]

    def lever_lines(self, r, name):
        return [l for l in r.stderr.splitlines() if l.startswith(f"[colabdesign-opt] LEVER name={name} ")]

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_stock_clean(self):
        out = os.path.join(self.tmp, "stock_ok")
        r = self.stock(out)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("[colabdesign-opt stock] ENV-CLEAN ok:", r.stderr)
        self.assertIn(" kit_modules=none ", r.stderr); self.assertIn(" upstream_loaded=none ", r.stderr); self.assertIn(" forbidden_present=none ", r.stderr)   # the proof is the line: no kit module, no upstream module, no forbidden variable
        self.assertIn("core_modules=none autoload_armed=none", r.stderr)                # opt_core.stock_proof: the stock process holds no core module beyond the proof
        self.assertEqual(sorted(f for f in os.listdir(out) if os.path.isfile(os.path.join(out, f))), ["design.fasta", "design.pdb", "failure_csv.csv", "trajectory.jsonl"])   # the outputs and BindCraft's own csv: no other file (its Trajectory/… directories beside them)
        run = self.runs(out)["0"]
        self.assertEqual(run["script_sha256"], names.sha256_file(os.path.join(os.path.dirname(stack.__file__), "stock_design.py")))
        self.assertNotIn("[colabdesign-opt] LEVER", r.stdout + r.stderr)                 # the stock arm runs no lever
        self.assertTrue(run["colabdesign_file"].startswith(self.site))                 # which colabdesign ran: the stub site's
        self.assertNotIn("stock_files", run); self.assertNotIn("vendored_files", run["bindcraft"])   # the run record carries no file inventory (the pins identify the trees)
        self.assertGreater(run["timing"]["ready_s"], 0)

    def test_stock_refuses_a_kit_variable(self):
        out = os.path.join(self.tmp, "stock_bad")
        r = self.stock(out, AF2M_LEVERS="nosub")
        self.assertEqual(r.returncode, 3); self.assertIn("ENV-CLEAN FAIL", r.stderr); self.assertIn("forbidden_present=AF2M_LEVERS", r.stderr)
        self.assertNotIn(f"{names.PREFIX} {report.RUN_HEAD}", r.stderr); self.assertFalse(os.path.isdir(out) and os.listdir(out))   # refused before any design: no [run] line, no output

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_kit_fast_steps_aside_by_name_without_the_kernel(self):
        """`fast` = every lever. The stub jax has no Pallas and no GPU backend and the stand-in design surface is not ColabDesign's pinned loop:
        after the clean-process proof every lever that cannot engage HERE steps aside BY NAME at install (`LEVER … state=skipped
        reason=cannot_run detail=…`; proj / txla `reason=no_attention_kernel` without a kernel lever) and the arm runs the design with
        the rest — the kit rule: a kit accepts everything stock accepts, a lever never refuses the mode and never disappears silently.
        (nosub's decisions and LEVER lines at either side of the size gate: test_levers, test_activation.)"""
        out = os.path.join(self.tmp, "fast_aside")
        r = self.kit(out, "fast")
        self.assertEqual(r.returncode, 0, r.stderr[-2500:])
        self.assertIn("[colabdesign-opt kit] ENV-CLEAN ok:", r.stderr); self.assertIn("forbidden_present=none", r.stderr)
        self.assertNotIn("REFUSED", r.stderr)
        self.assertRegex(r.stderr, r"\[colabdesign-opt\] LEVER name=(hoist_prev|trimul_pallas|F1\.pallas_attn|triatt) state=skipped reason=cannot_run (\S+ )*detail=\S+ source=install")
        self.assertRegex(r.stderr, r"\[colabdesign-opt\] LEVER name=proj state=skipped reason=no_attention_kernel ")
        self.assertIn(f"{names.PREFIX} {report.RUN_HEAD}", r.stderr)                     # the design ran with the levers that engaged

    def test_kit_refuses_a_kit_variable(self):
        for var in ("AF2M_LEVERS", "AF_PALLAS_ATTN_PRECISE_BWD", "COLABDESIGN_OPT"):
            r = self.kit(os.path.join(self.tmp, f"kit_{var}"), "fast", **{var: "1"})
            self.assertEqual(r.returncode, 3, var); self.assertIn(f"forbidden_present={var}", r.stderr)

if __name__ == "__main__":
    unittest.main()
