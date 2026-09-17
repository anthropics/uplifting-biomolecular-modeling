"""The deterministic recipe as a package surface (det.py): its levels and names, the environment it exports before torch, `apply()` on a
stub torch (the switches it sets, the scatter-mean it rebinds), the stock child's `--det 0|1` parsing (stock_design.main: applied AFTER the
environment proof and BEFORE upstream is imported, never on a process that failed its proof, named on the STOCK line), the stock command
that carries the level to the child, and the level's word on the activation lines. No torch here: the tensor-side module (det_scatter) is
replaced by a stub where `apply()` reaches it."""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from .. import _autoload, cli, det, census, report, stack, stock_design
from . import _fixtures as fx


def _stub_torch(calls: list) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0+stub"
    torch.use_deterministic_algorithms = lambda mode, warn_only=False: calls.append(("use_deterministic_algorithms", mode, warn_only))
    torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=False, benchmark=True))
    return torch


class TestLevels(unittest.TestCase):
    def test_the_table(self):
        self.assertEqual((det.LEVELS, det.DEFAULT_LEVEL, det.ENV_LEVEL), ((0, 1), 0, "RFDIFFUSION3_OPT_DET"))
        self.assertEqual(det.RECIPE_ENV, ("CUBLAS_WORKSPACE_CONFIG", "FOUNDRY_DET_SCATTER"))
        self.assertEqual((det.level(0), det.level(1), det.level("1")), (0, 1, 1))
        for bad in (2, -1, "3"):
            with self.assertRaises(ValueError) as cm:
                det.level(bad)
            self.assertIn("levels are (0, 1)", str(cm.exception))
        self.assertEqual((det.word(0), det.word(1)), ("det=0", "det=1"))

    def test_level_from_env(self):
        self.assertEqual(det.level_from_env({}), 0)
        self.assertEqual(det.level_from_env({det.ENV_LEVEL: ""}), 0)
        self.assertEqual(det.level_from_env({det.ENV_LEVEL: "0"}), 0)
        self.assertEqual(det.level_from_env({det.ENV_LEVEL: " 1 "}), 1)
        with self.assertRaises(ValueError) as cm:
            det.level_from_env({det.ENV_LEVEL: "2"})
        self.assertEqual(str(cm.exception), "RFDIFFUSION3_OPT_DET='2': levels are 0|1")      # by name
        with mock.patch.dict(os.environ, {det.ENV_LEVEL: "1"}):
            self.assertEqual(det.level_from_env(), 1)                                        # the process's own environment by default

    def test_env_before_torch(self):
        self.assertEqual(det.env_before_torch(0), {})
        self.assertEqual(det.env_before_torch(1), {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "FOUNDRY_DET_SCATTER": "1"})
        self.assertEqual(tuple(det.env_before_torch(1)), det.RECIPE_ENV)
        with self.assertRaises(ValueError):
            det.env_before_torch(2)

    def test_the_names_are_the_stock_arms_forbidden_names_and_the_level_a_package_switch(self):
        """The recipe's two variables are must-be-absent names of the stock arm (an inherited copy never reaches the stock child, which sets
        its own at --det 1); the level's variable is a declared package switch (stripped from the child with the others)."""
        self.assertTrue(set(det.RECIPE_ENV) <= set(stack.lever_names()))
        self.assertIn(det.ENV_LEVEL, _autoload.ENV_NAMES); self.assertIn(det.ENV_LEVEL, stack.PACKAGE_ENV)
        self.assertNotIn(det.ENV_LEVEL, stack.strip_env({det.ENV_LEVEL: "1", "FOUNDRY_DET_SCATTER": "1", "HOME": "/h"}))
        self.assertIn("det", __import__("rfdiffusion3_opt")._LAZY); self.assertNotIn("det_scatter", __import__("rfdiffusion3_opt")._LAZY)   # the tensor side is not a package name


class TestApply(unittest.TestCase):
    def test_level_0_touches_nothing(self):
        env = {}
        with mock.patch.dict(sys.modules, {"torch": None}):                                  # an `import torch` here would raise ImportError
            self.assertEqual(det.apply(0, environ=env), {"level": 0})
        self.assertEqual(env, {})

    def test_level_1_sets_the_variables_torchs_switches_and_the_scatter(self):
        calls, env = [], {}
        torch = _stub_torch(calls)
        upstream = types.ModuleType("foundry.utils.torch")
        orig = upstream.scatter_mean = lambda zeros, dim, index, source: "upstream"
        early = types.ModuleType("rfd3.some.caller"); early.scatter_mean = orig             # a module that bound the name before the recipe was applied
        other = types.ModuleType("rfd3.other"); other.scatter_mean = lambda *a: "unrelated"  # a same-named function that is not upstream's: left alone
        scat = types.ModuleType("rfdiffusion3_opt.det_scatter")
        scat.foundry_scatter_mean = lambda zeros, dim, index, source: "fixed-order"
        mods = {"torch": torch, "foundry": types.ModuleType("foundry"), "foundry.utils": types.ModuleType("foundry.utils"), "foundry.utils.torch": upstream,
                "rfd3.some.caller": early, "rfd3.other": other, "rfdiffusion3_opt.det_scatter": scat}
        with mock.patch.dict(sys.modules, mods):
            res = det.apply(1, environ=env)
            self.assertEqual(env, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "FOUNDRY_DET_SCATTER": "1"})
            self.assertEqual(calls, [("use_deterministic_algorithms", True, True)])            # warn_only: upstream's non-deterministic ops warn, not raise
            self.assertEqual((torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark), (True, False))
            self.assertEqual({k: res[k] for k in ("level", "CUBLAS_WORKSPACE_CONFIG", "FOUNDRY_DET_SCATTER", "deterministic_algorithms", "warn_only", "cudnn_deterministic", "cudnn_benchmark", "torch_imported_before")},
                             {"level": 1, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "FOUNDRY_DET_SCATTER": "1", "deterministic_algorithms": True, "warn_only": True,
                              "cudnn_deterministic": True, "cudnn_benchmark": False, "torch_imported_before": True})
            self.assertEqual(res["scatter"], {"target": "foundry.utils.torch.scatter_mean", "installed": "yes", "rebound": ["rfd3.some.caller"]})
            self.assertIsNot(upstream.scatter_mean, orig); self.assertIs(upstream.scatter_mean.__wrapped__, orig)
            self.assertIs(early.scatter_mean, upstream.scatter_mean)                            # rebound to the same wrapper
            self.assertEqual(other.scatter_mean(), "unrelated")
            dev = lambda kind: types.SimpleNamespace(device=types.SimpleNamespace(type=kind))
            self.assertEqual(upstream.scatter_mean(dev("cuda"), 0, None, None), "fixed-order")   # served by det_scatter ...
            self.assertEqual(upstream.scatter_mean(dev("mps"), 0, None, None), "upstream")       # ... except on upstream's own MPS branch
            self.assertEqual(det.install_scatter()["installed"], "already")                      # idempotent
            self.assertEqual(det.apply(1, environ=env)["scatter"]["installed"], "already")


class TestStockChild(unittest.TestCase):
    """stock_design.main's own `--det` (before `--`): parsed and removed from upstream's argv, recorded in the proof, applied once AFTER the
    environment proof passed and BEFORE upstream runs, named on the STOCK line; a failed proof applies nothing."""

    def setUp(self):
        census.reset()
        self.events = []
        self.proof = {"ok": True, "reasons": [], "python": "/venv/bin/python", "forbidden_names_checked": list(stack.lever_names()), "forbidden_present": [],
                      "upstream_env": {}, "pins": {"distribution": {"version": "0.2.1.dev13+g4010e3e2e"}, "notes": ["rc-foundry is an index install (a note)"], "pristine": True}}

    def tearDown(self):
        census.reset()

    def _main(self, argv, ok=True):
        proof = dict(self.proof, ok=ok, reasons=[] if ok else ["rfd3 tree is not pristine: rfd3/model/x.py=patched"])
        def env_proof(**kw):
            self.events.append("proof"); return json.loads(json.dumps(proof))
        def apply(lv, environ=None):
            self.events.append(("apply", lv)); return {"level": lv, "scatter": {"installed": "yes"}}
        def run_stock(args):
            self.events.append(("run", list(args))); return 0
        err = io.StringIO()
        with mock.patch.object(stock_design, "env_proof", env_proof), mock.patch.object(det, "apply", apply), mock.patch.object(stock_design, "run_stock", run_stock), \
                contextlib.redirect_stderr(err):
            rc = stock_design.main(list(argv))
        lines = [l for l in err.getvalue().splitlines() if l.startswith(stock_design.PREFIX)]
        return rc, lines

    def test_det_1_is_applied_after_the_proof_and_named(self):
        with tempfile.TemporaryDirectory() as td:
            pj = os.path.join(td, "proof.json")
            rc, lines = self._main(["--proof-json", pj, "--det", "1", "--", "design", "out_dir=/o", "inputs=/s.json"])
            self.assertEqual(rc, 0)
            self.assertEqual(self.events, ["proof", ("apply", 1), ("run", ["design", "out_dir=/o", "inputs=/s.json"])])   # proof, then the recipe once, then upstream — `--det 1` is not upstream's token
            self.assertEqual([l.split()[2] for l in lines[:2]], ["PINS", "STOCK"])                  # the pin note first, then the STOCK line
            self.assertEqual(lines[0], "[rfdiffusion3-opt stock] PINS rc-foundry is an index install (a note)")
            self.assertEqual(lines[1], "[rfdiffusion3-opt stock] STOCK rc-foundry=0.2.1.dev13+g4010e3e2e tree=pristine python=/venv/bin/python forbidden_absent=%d upstream_env=none det=1" % len(stack.lever_names()))
            written = json.load(open(pj))
            self.assertEqual((written["det"], written["after"]["exit_code"]), ({"level": 1, "scatter": {"installed": "yes"}}, 0))   # the proof file carries what apply() returned

    def test_det_0_and_absent_apply_nothing(self):
        for argv in (["--det", "0", "--", "design", "out_dir=/o"], ["--", "design", "out_dir=/o"], ["design", "out_dir=/o"]):
            self.events.clear(); census.reset()
            rc, lines = self._main(argv)
            self.assertEqual(rc, 0, argv)
            self.assertEqual(self.events, ["proof", ("run", ["design", "out_dir=/o"])], argv)
            self.assertTrue(lines[1].endswith(" upstream_env=none det=0"), lines)

    def test_det_after_the_separator_is_upstreams_token(self):
        rc, lines = self._main(["--", "design", "out_dir=/o", "--det", "1"])
        self.assertEqual(self.events, ["proof", ("run", ["design", "out_dir=/o", "--det", "1"])])
        self.assertTrue(lines[1].endswith(" det=0"), lines)

    def test_a_failed_proof_applies_nothing(self):
        rc, lines = self._main(["--det", "1", "--", "design", "out_dir=/o"], ok=False)
        self.assertEqual(rc, stock_design.EXIT_NOT_STOCK)
        self.assertEqual(self.events, ["proof"])                                                    # no recipe, no upstream
        self.assertEqual(lines, ["[rfdiffusion3-opt stock] PINS rc-foundry is an index install (a note)", "[rfdiffusion3-opt stock] NOT STOCK: rfd3 tree is not pristine: rfd3/model/x.py=patched"])

    def test_levels_and_usage(self):
        with self.assertRaises(ValueError):
            self._main(["--det", "2", "--", "design", "out_dir=/o"])                                # the design verb's argparse keeps 2 away; the child names it if it arrives
        rc, lines = self._main(["--det", "1", "--", "predict"])
        self.assertEqual((rc, self.events), (2, []))
        self.assertIn("[--det 0|1] -- design", lines[0])

    def test_the_stock_command_carries_the_level(self):
        self.assertEqual(cli.stock_command("/venv/bin/python", "/t/" + cli.PROOF_FILENAME, ["out_dir=/o", "inputs=/s.json"], det=1),
                         ["/venv/bin/python", "-I", "-m", "rfdiffusion3_opt.stock_design", "--proof-json", "/t/" + cli.PROOF_FILENAME, "--det", "1", "--", "design", "out_dir=/o", "inputs=/s.json"])   # the proof path is the caller's (cli._design_off: a temporary file folded into opt_manifest.json), never the out_dir
        self.assertEqual(cli.stock_command("/p", "/t/p.json", ["out_dir=/o"])[6:9], ["--det", "0", "--"])       # default 0, always spelled, before the separator


class TestActivationWord(unittest.TestCase):
    """`det=<level>` on the DRY-RUN / ACTIVE lines (report.activation_line), the level from --det or, on the .pth route, from
    RFDIFFUSION3_OPT_DET; an invalid level is a named refusal."""

    def test_the_word_follows_the_report(self):
        rep = {"mode": "exact", "env": {"RFD3_HOIST": "1"}, "env_line": "RFD3_HOIST=1", "active": True, "interpreter": {"label": "patched", "detail": "10/10 patched"},
               "applied": "at-import", "attn_pin": True, "alloc_conf": "unset", "det": {"level": 1}, "trigger": "design"}
        self.assertEqual(report.activation_line(rep), "[rfdiffusion3-opt] ACTIVE mode=exact env=RFD3_HOIST=1 interpreter=patched(10/10 patched) foundry=None gpu=%s kit=None applied=at-import attn_pin=1 alloc_conf=unset det=1 trigger=design" % report._gpu_str(None))   # the word sits after the allocator word, before the trigger
        self.assertIn(" alloc_conf=unset det=0 trigger=design", report.activation_line(dict(rep, det={"level": 0})))
        self.assertIn(" det=1 trigger=", report.activation_line(dict(rep, det={"level": 1, "scatter": {"installed": "yes"}})))   # apply()'s record: the level word only
        self.assertNotIn("det=", report.activation_line({k: v for k, v in rep.items() if k != "det"}))

    def test_dry_run_reads_the_level_from_the_switch(self):
        with tempfile.TemporaryDirectory() as td:
            site, bin_dir = fx.make_site(os.path.join(td, "p"), "patched"), fx.fake_gpu(os.path.join(td, "bin"))
            r = fx.run_cli(["check", "--mode", "exact"], site, bin_dir=bin_dir)
            self.assertEqual(r.returncode, 0, r.stderr); self.assertRegex(r.stderr, r"DRY-RUN mode=exact .* alloc_conf=\S+ det=0( |$)")
            r = fx.run_cli(["check", "--mode", "exact"], site, env={det.ENV_LEVEL: "1"}, bin_dir=bin_dir)
            self.assertEqual(r.returncode, 0, r.stderr); self.assertRegex(r.stderr, r"DRY-RUN mode=exact .* det=1( |$)")   # a dry run applies nothing (no torch here): the level is read and named
            r = fx.run_cli(["check", "--mode", "off"], fx.make_site(os.path.join(td, "s"), "pristine"), env={det.ENV_LEVEL: "1"}, bin_dir=bin_dir)
            self.assertEqual(r.returncode, 0, r.stderr); self.assertIn(" det=1 (stock: nothing exported)", r.stderr)
            r = fx.run_cli(["check", "--mode", "exact"], site, env={det.ENV_LEVEL: "2"}, bin_dir=bin_dir)
            self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("RFDIFFUSION3_OPT_DET='2': levels are 0|1", r.stderr)
            r = fx.run_cli(["design", "--mode", "exact", "--det", "2", "--ckpt", "/c.ckpt", "inputs=/s.json", f"out_dir={os.path.join(td, 'o')}"], site, bin_dir=bin_dir)
            self.assertEqual(r.returncode, 2); self.assertIn("--det: invalid choice: 2", r.stderr)   # the design verb: argparse, det.LEVELS


if __name__ == "__main__":
    unittest.main()
