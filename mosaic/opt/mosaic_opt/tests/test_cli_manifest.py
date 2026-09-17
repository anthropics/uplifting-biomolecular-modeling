"""The command layer against the stub kit (the stand-in driver): `check` lines and exit codes, `design --mode off` through the stock
caller with its ENV-CLEAN line, `design --mode exact` refused until the shape is warm and then composed from the row (the
driver's manifest shows P1/P2/P3), `warm` producing the shape's files, the exit tally, the exit codes, the usage.

Bare-interpreter form: every child these tests spawn is the stand-in driver (`_stubs.STUB_DRIVER`, run by `sys.executable`: argparse, hashlib,
json, os, sys, time — stdlib only) with the gates stubbed (`_stubs.stub_gates`); no jax/torch stack, no installed package or venv and no
GPU route is needed, so the class runs green under pytest alone and skips by name only when the release tree is not around the package."""
import hashlib
import shutil
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import pytest

from . import _stubs, core_src
from mosaic_opt import cli, modes, outputs, stack
from mosaic_opt.stack import WEIGHTS_MISSING_MARKER


def shown(*on):
    """The driver's per-lever booleans over EVERY registry lever: True for the ids named."""
    from mosaic_opt import registry
    return {k: k in on for k in registry.LEVERS}


INSTALL_VERDICT = stack.installed_fast_label(stack.installed_fast_check())          # this box's installed-mosaic/fast verdict, the token the activation lines carry (not-installed | kit-bytes@<dir>)


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestCli(unittest.TestCase):
    def setUp(self):
        self.state = _stubs.StubState(); self.state.restore()
        self.mp = pytest.MonkeyPatch()
        self.tmp = tempfile.mkdtemp(prefix="mosaic_opt_cli_")
        self.kit = _stubs.make_stub_kit(self.tmp)
        for k in ("XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "MOSAIC_OPT", "MOSAIC_OPT_CACHE_DIR", "MOSAIC_OPT_FORCE", "MODEL_OPT_TARGET_GPU", "STUB_DRIVER_FAIL", "STUB_DRIVER_SLEEP", "STUB_DRIVER_DROP", "STUB_DRIVER_NO_RESULTS"):
            self.mp.delenv(k, raising=False)
        self.mp.setenv("MOSAIC_OPT_HOME", _stubs.TREE)
        self.mp.setenv("MOSAIC_OPT_KIT", self.kit)
        self.root = os.path.join(self.tmp, "jitcache"); os.makedirs(self.root)
        self.mp.setenv("MOSAIC_OPT_CACHE_ROOT", self.root)
        self.weights = os.path.join(self.tmp, "weights"); os.makedirs(os.path.join(self.weights, "boltz")); open(os.path.join(self.weights, "boltz", "boltz2_conf.ckpt"), "wb").write(b"x")
        self.mp.setenv("MOSAIC_CACHE_DIR", self.weights)
        _stubs.stub_gates(self.mp)

    def tearDown(self):
        self.mp.undo(); self.state.restore()

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = cli.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def warm_shape(self, key="L80_c1"):
        d = os.path.join(self.root, key); os.makedirs(os.path.join(d, "xla_cache"), exist_ok=True)
        open(os.path.join(d, f"features_{key}.npz"), "wb").write(b"frozen")
        open(os.path.join(d, "xla_cache", "xla_autotune_results.pb"), "wb").write(b"pb")
        return d

    def test_usage(self):
        rc, out, err = self.run_cli([])
        self.assertEqual(rc, 2); self.assertIn("usage:", err)
        rc, out, err = self.run_cli(["--help"])
        self.assertEqual(rc, 0); self.assertIn("design", out); self.assertIn(" | ".join(modes.MODES), out)
        for word in modes.NOT_WIRED:                                                    # a tier word with no lever wired: usage error, named
            rc, out, err = self.run_cli(["check", "--mode", word])
            self.assertEqual(rc, 2); self.assertIn(f"unknown mode '{word}'", err); self.assertIn("tier word", err)

    def test_check_lines(self):
        rc, out, err = self.run_cli(["check", "--mode", "off"])
        self.assertEqual(rc, 0); self.assertIn("DRY-RUN mode=off route=driver row=A_stock1[stock]", err); self.assertIn("shape=L80_c1(169 tokens)", err)
        self.assertIn(" mosaic.fast=" + INSTALL_VERDICT, err)                                # not-installed on a box without upstream mosaic; kit-bytes@<dir> on the kit image
        rc, out, err = self.run_cli(["check", "--mode", "exact"])                                          # the exact row; the shape is not warm
        self.assertEqual(rc, 0); self.assertIn("DRY-RUN mode=exact route=driver row=B_p1populate_p2[P1,P2,P3]", err)
        self.assertIn("not warm", err); self.assertIn("features=absent", err); self.assertNotIn("would_refuse", err)
        self.warm_shape()
        rc, out, err = self.run_cli(["check", "--mode", "exact", "--json"])
        self.assertEqual(rc, 0); rep = json.loads(out)
        self.assertEqual(rep["row"], "C_p1warm_p2"); self.assertEqual(rep["levers_planned"], ["P1", "P2", "P3"]); self.assertTrue(rep["features"]["present"])
        self.assertEqual(rep["env"]["JAX_COMPILATION_CACHE_DIR"], os.path.join(self.root, "L80_c1", "xla_cache"))
        self.assertIn("row=C_p1warm_p2[P1,P2,P3]", err); self.assertIn("features=present", err)
        self.mp.delenv("MOSAIC_OPT_CACHE_ROOT")
        rc, out, err = self.run_cli(["check", "--mode", "exact"])
        self.assertEqual(rc, 3); self.assertIn("would_refuse='MOSAIC_OPT_CACHE_ROOT unset", err)

    def test_check_refuses_without_gpu(self):
        _stubs.stub_gates(self.mp, gpu={"name": None, "cc": None, "sm": None, "probe": "none"})
        rc, out, err = self.run_cli(["check", "--mode", "exact"])
        self.assertEqual(rc, 3); self.assertIn("would_refuse=", err); self.assertIn("no GPU visible", err)

    def test_design_off_runs_the_stock_caller(self):
        out_dir = os.path.join(self.tmp, "out_off")
        rc, out, err = self.run_cli(["design", "--mode", "off", "--out", out_dir, "--seed", "5"])
        self.assertEqual(rc, 0, err)
        self.assertIn("NOT ACTIVE: mode off: stock mosaic", err); self.assertIn("ENV-CLEAN ok", err); self.assertIn("[mosaic-opt] [run] off_L80_c1_s5:", err)
        self.assertIn("DONE mode=off rc=0", err)
        res = outputs.read_results(out_dir, "off_L80_c1_s5")
        self.assertEqual(res["status"], "ok"); self.assertEqual(res["manifest"]["load_path"][:12], "stock Boltz2")
        self.assertTrue(res["manifest"]["features"]["source"].startswith("in-process"))
        seen = res["manifest"]["env_seen"]
        for k in ("XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "MOSAIC_OPT"):
            self.assertNotIn(k, seen)
        self.assertNotIn("XLA_PYTHON_CLIENT_PREALLOCATE", seen); self.assertEqual(seen["MOSAIC_CACHE_DIR"], self.weights)   # --det 0, the default: nothing of the recipe exported
        rc, out, err = self.run_cli(["design", "--mode", "off", "--out", os.path.join(self.tmp, "out_off_det1"), "--seed", "5", "--det", "1"])
        self.assertEqual(rc, 0, err)
        seen1 = outputs.read_results(os.path.join(self.tmp, "out_off_det1"), "off_L80_c1_s5")["manifest"]["env_seen"]
        self.assertEqual(seen1["PYTHONUNBUFFERED"], "1"); self.assertNotIn("XLA_PYTHON_CLIENT_PREALLOCATE", seen1)   # --det 1: the recipe's stdio setting reaches the stock arm; no allocator variable at any level
        self.assertEqual(outputs.classify(res), shown()); self.assertEqual(outputs.run_summary(res).get("n_tokens"), 169)
        self.assertEqual(outputs.written_files(out_dir, "off_L80_c1_s5"), ["off_L80_c1_s5/pssm_seed5.npz", "off_L80_c1_s5/refold_seed5.npz", "off_L80_c1_s5/results.json"])
        self.assertEqual(sorted(os.listdir(out_dir)), ["off_L80_c1_s5"])                          # the driver's file set only: no side file beside it
        self.assertIn("stock subprocess:", err); self.assertIn("mosaic_opt.stock_design", err)

    def test_input_options_refused_by_name_or_named_on_every_route(self):
        """--target-fasta with more than one record is a usage error (exit 2, `fasta_multi_record`, both remedies named) on every command BEFORE any
        model work; `--first-record` runs record 1 and the run SAYS so: the driver's `INPUT target_fasta records=<n> used=1 (--first-record)` line
        relayed under the package tag and the record facts in results.json. `--epitope` out of range / misspelled is exit 2 by name; a valid one
        reaches the driver on the stock route and the kit route alike (an input option, every mode) with its INPUT line and manifest record, and
        keys the shape (`ep<sha8>`), so `warm` / `design --mode exact` carry it through both of their driver launches."""
        multi = os.path.join(self.tmp, "two.fasta"); open(multi, "w").write(">t1 first\nMKTAYIAKQRQISFVKSHFS\n>t2\nGGGG\n")
        for cmd in (["check", "--mode", "off"], ["check", "--mode", "exact"], ["design", "--mode", "off", "--out", os.path.join(self.tmp, "o_refused")],
                    ["design", "--mode", "fast", "--out", os.path.join(self.tmp, "o_refused2")], ["warm", "--mode", "exact"]):
            rc, out, err = self.run_cli(cmd + ["--target-fasta", multi])
            self.assertEqual(rc, 2, (cmd, err)); self.assertIn("fasta_multi_record", err); self.assertIn("--first-record", err); self.assertIn("holds 2 FASTA records", err)
            self.assertNotIn("ACTIVE", err); self.assertNotIn("[run]", err)                      # refused before anything of the mode or the driver
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "o_refused")))
        rc, out, err = self.run_cli(["check", "--mode", "off", "--epitope", "90"])                 # the public target has 89 residues
        self.assertEqual(rc, 2, err); self.assertIn("epitope_out_of_range", err); self.assertIn("1..89", err)
        rc, out, err = self.run_cli(["check", "--mode", "off", "--target-fasta", multi, "--first-record", "--epitope", "3,21"])
        self.assertEqual(rc, 2, err); self.assertIn("epitope_out_of_range", err); self.assertIn("1..20", err)
        rc, out, err = self.run_cli(["check", "--mode", "off", "--epitope", "5-3"])
        self.assertEqual(rc, 2, err); self.assertIn("epitope_format", err)
        rc, out, err = self.run_cli(["check", "--mode", "off", "--first-record"])
        self.assertEqual(rc, 2, err); self.assertIn("first_record_without_fasta", err)
        # the stock route: record 1 of the file, two copies, an epitope — named on INPUT lines, recorded, passed to the driver as its own flags
        out_dir = os.path.join(self.tmp, "out_inputs_off")
        rc, out, err = self.run_cli(["design", "--mode", "off", "--out", out_dir, "--tag", "in_off", "--target-fasta", multi, "--first-record", "--target-copies", "2", "--epitope", "5, 2,4-5"])
        self.assertEqual(rc, 0, err)
        self.assertIn("[mosaic-opt] INPUT target_fasta records=2 used=1 (--first-record)", err)
        self.assertIn("[mosaic-opt] INPUT epitope residues=2,4-5 positions=3 target_length=20 copies=2 epitope_idx=6 (--epitope: 1-based target residue positions, applied on every target copy)", err)
        self.assertIn("DONE mode=off rc=0", err)
        man = outputs.read_results(out_dir, "in_off")["manifest"]
        self.assertEqual((man["args"]["first_record"], man["args"]["epitope"], man["args"]["target_copies"]), (True, "2,4-5", 2))   # the canonical spelling is what the driver receives
        self.assertEqual((man["target"]["fasta_records"], man["target"]["fasta_record_used"], man["target"]["first_record"]), (2, 1, True))
        self.assertEqual((man["epitope"]["residues"], man["epitope"]["idx"], man["epitope"]["copies"]), ([2, 4, 5], [1, 3, 4, 21, 23, 24], 2))
        # the kit route (exact, a cold shape: warmed first — both warm launches and the design launch carry the input options; the shape key names the epitope)
        out_x = os.path.join(self.tmp, "out_inputs_exact")
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", out_x, "--tag", "in_x", "--target-fasta", multi, "--first-record", "--epitope", "4-5,2"])
        self.assertEqual(rc, 0, err)
        import hashlib as _h
        ep8 = _h.sha256(b"2,4-5").hexdigest()[:8]
        self.assertRegex(err, r"shape t1-[0-9a-f]{8}_ep" + ep8 + r"_L80_c1 not warm under .*: P1,P3 step aside by name")   # a cold shape: named, nothing warmed in the command
        self.assertEqual(err.count("INPUT target_fasta records=2 used=1 (--first-record)"), 1)       # the design, the one driver process of the command (nothing warmed inside it), names its inputs
        self.assertIn("ACTIVE mode=exact route=driver", err); self.assertIn("DONE mode=exact rc=0", err)
        manx = outputs.read_results(out_x, "in_x")["manifest"]
        self.assertEqual((manx["args"]["epitope"], manx["epitope"]["idx"], manx["target"]["first_record"]), ("2,4-5", [1, 3, 4], True))
        plain = os.path.join(self.tmp, "out_plain")
        rc, out, err = self.run_cli(["design", "--mode", "off", "--out", plain, "--tag", "plain"])
        self.assertEqual(rc, 0, err); self.assertNotIn("INPUT ", err)                               # nothing narrowed: no INPUT line, and the manifest says so
        manp = outputs.read_results(plain, "plain")["manifest"]
        self.assertEqual((manp["epitope"], manp["args"]["first_record"], manp["args"]["epitope"], manp["target"]["first_record"]), (None, False, None, False))

    def test_design_exact_warms_a_cold_shape_first_then_composes_from_the_row(self):
        """A cold shape (no frozen features / P1 files under the cache root) is a cache miss, named — never a refusal: `design` warms it
        — P1 and P3 step aside by name and the design compiles and featurizes as stock does; `warm` fills the shape ahead of time."""
        out0 = os.path.join(self.tmp, "out_exact_cold")
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", out0, "--target-copies", "2"])
        self.assertEqual(rc, 0, err)
        self.assertIn("[mosaic-opt] shape L80_c2 not warm under ", err); self.assertIn("P1,P3 step aside by name", err); self.assertNotIn("NOT ACTIVE", err)
        self.assertNotIn("WARM ", err)                                                                # nothing is warmed inside the command
        self.assertIn("ACTIVE mode=exact route=driver row=C_p1warm_p2[P2]-aside[P1,P3] levers=P2 unavailable=none p1=none(shape_not_warm:_run.sh_warm)", err)
        self.assertFalse(os.path.isfile(os.path.join(self.root, "L80_c2", "features_L80_c2.npz")))      # no features frozen, no P1 files written: `warm` does that
        seen = outputs.read_results(out0, "exact_L80_c2_s0")["manifest"]["env_seen"]
        self.assertNotIn("JAX_COMPILATION_CACHE_DIR", seen); self.assertNotIn("XLA_FLAGS", seen)         # the design compiled and featurized as stock does (P2's --weights fastinit only)
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", os.path.join(self.tmp, "out_exact_cold2"), "--target-copies", "2"])
        self.assertEqual(rc, 0, err); self.assertIn("not warm", err)                               # still cold: the second process says so again, by name
        out_dir = os.path.join(self.tmp, "out_exact")
        d = self.warm_shape()
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", out_dir, "--seed", "7", "--tag", "run7"])
        self.assertEqual(rc, 0, err)
        self.assertIn("ACTIVE mode=exact route=driver row=C_p1warm_p2[P1,P2,P3] levers=P1,P2,P3 unavailable=none p1=load:", err)
        res = outputs.read_results(out_dir, "run7")
        seen = res["manifest"]["env_seen"]
        self.assertEqual(seen["JAX_COMPILATION_CACHE_DIR"], os.path.join(d, "xla_cache"))
        self.assertEqual(seen["XLA_FLAGS"], f"--xla_gpu_load_autotune_results_from={os.path.join(d, 'xla_cache', 'xla_autotune_results.pb')}")
        self.assertEqual(seen["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"], "0"); self.assertEqual(seen["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"], "0")
        self.assertNotIn("MOSAIC_OPT", seen); self.assertNotIn("XLA_PYTHON_CLIENT_PREALLOCATE", seen)              # --det 0, the default
        self.assertEqual(res["manifest"]["args"]["weights"], "fastinit"); self.assertEqual(res["manifest"]["args"]["seed"], 7)
        self.assertEqual(outputs.classify(res), shown("P1", "P2", "P3")); self.assertNotIn(" partial=", err)
        self.assertEqual(sorted(os.listdir(out_dir)), ["run7"])                                     # the driver's file set only
        self.assertNotIn("WARNING", err)

    def test_design_exact_exit_code_follows_the_driver(self):
        self.warm_shape()
        self.mp.setenv("STUB_DRIVER_FAIL", "1")
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", os.path.join(self.tmp, "out_fail")])
        self.assertEqual(rc, 1); self.assertIn("DONE mode=exact rc=1", err); self.assertNotIn(" partial=", err); self.assertNotIn(" incomplete=", err)
        self.mp.setenv("STUB_DRIVER_DROP", "P2")                                                    # failed and partial: 1 (the run failed), never 3
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", os.path.join(self.tmp, "out_fail2")])
        self.assertEqual(rc, 1); self.assertNotIn("NOT ACTIVE", err)

    def test_design_exact_partial_is_exit_3_and_recorded(self):
        """The driver's manifest does not show a lever of the row applied (P2: the stock load path under --weights fastinit): a partial
        activation — recorded on the DONE line, exit 3; --allow-partial records `allow_partial` and the exit is the run's own."""
        self.warm_shape()
        self.mp.setenv("STUB_DRIVER_DROP", "P2")
        out_dir = os.path.join(self.tmp, "out_partial")
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", out_dir, "--tag", "p"])
        self.assertEqual(rc, 3, err)
        self.assertIn(f"[mosaic-opt] NOT ACTIVE: partial activation — P2 — the driver's manifest does not show them applied (see {outputs.results_path(out_dir, 'p')}); exit 3 (--allow-partial records and proceeds)", err)
        self.assertIn("DONE mode=exact rc=3", err); self.assertIn(" partial=P2 (the driver's manifest does not show them applied) out=", err)
        self.assertEqual(outputs.classify(outputs.read_results(out_dir, "p")), shown("P1", "P3")); self.assertNotIn(" allow_partial=1", err)
        self.assertTrue(os.path.isfile(outputs.results_path(out_dir, "p")))                          # the run itself completed: partial, not failed
        self.mp.setenv("MOSAIC_OPT_ALLOW_PARTIAL", "1")                                       # the in-process route's word: the CLI verbs ignore it
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", os.path.join(self.tmp, "out_env_ignored")])
        self.assertEqual(rc, 3, err); self.assertNotIn(" allow_partial=1", err); self.assertNotIn("PARTIAL allowed", err)
        self.mp.delenv("MOSAIC_OPT_ALLOW_PARTIAL")
        out2 = os.path.join(self.tmp, "out_allowed_flag")
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", out2, "--allow-partial"])
        self.assertEqual(rc, 0, err)
        self.assertIn("[mosaic-opt] PARTIAL allowed: P2 — the driver's manifest does not show them applied (see ", err); self.assertIn(" (--allow-partial, recorded)", err)
        self.assertIn(" partial=P2 (the driver's manifest does not show them applied) allow_partial=1 out=", err)
        self.mp.setenv("STUB_DRIVER_DROP", "P1,P3")
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", os.path.join(self.tmp, "out_p13")])
        self.assertEqual(rc, 3); self.assertIn(" partial=P1,P3 (the driver's manifest does not show them applied) out=", err)

    def test_design_incomplete_outputs_is_exit_1_never_partial(self):
        """The driver exits 0 without its results file: `incomplete: 0/1`, exit 1 — a different condition from partial (which stays [])."""
        self.warm_shape()
        self.mp.setenv("STUB_DRIVER_NO_RESULTS", "1")
        out_dir = os.path.join(self.tmp, "out_inc")
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", out_dir])
        self.assertEqual(rc, 1, err); self.assertIn("design FAILED: incomplete — the driver exited 0 without", err); self.assertIn(" incomplete=0/1 out=", err)
        self.assertNotIn(" partial=", err)
        rc, out, err = self.run_cli(["design", "--mode", "off", "--out", os.path.join(self.tmp, "out_inc_off")])
        self.assertEqual(rc, 1, err); self.assertIn(" incomplete=0/1 out=", err)

    def test_warm_partial_populate_row_is_exit_3(self):
        """The populate row's manifest is read back for every lever of the row (P1 dump, P2, P3): one not shown applied is a partial
        activation — WARM NOT ACTIVE, exit 3; --allow-partial records it and the shape is warm."""
        self.mp.setenv("STUB_DRIVER_DROP", "P2")
        rc, out, err = self.run_cli(["warm", "--target-copies", "2"])
        self.assertEqual(rc, 3, err)
        self.assertIn("[mosaic-opt] NOT ACTIVE: partial activation — P2 — the driver's manifest does not show them applied (populate row warm_B_populate, see ", err)
        self.assertIn("; exit 3 (--allow-partial records and proceeds)", err)
        self.assertIn("WARM NOT ACTIVE mode=exact shape=L80_c2 features=present autotune=present", err); self.assertIn(" partial=P2 (the driver's manifest does not show them applied) out=", err)
        d = os.path.join(self.root, "L80_c2")
        self.assertEqual(outputs.classify(outputs.read_results(os.path.join(d, "warm"), "warm_B_populate")), shown("P1", "P3")); self.assertNotIn(" allow_partial=1", err)
        shutil.rmtree(d)                                                                          # a warm shape is refused: remove its directory to warm it again
        rc, out, err = self.run_cli(["warm", "--target-copies", "2", "--allow-partial"])
        self.assertEqual(rc, 0, err)
        self.assertIn("WARM PASS mode=exact shape=L80_c2", err); self.assertIn(" partial=P2 (the driver's manifest does not show them applied) allow_partial=1 out=", err)
        self.assertFalse(os.path.exists(os.path.join(d, "warm", "warm_result.json")))            # the shape's files and the driver's own outputs only
        self.mp.delenv("STUB_DRIVER_DROP"); self.mp.setenv("STUB_DRIVER_FAIL", "1")
        shutil.rmtree(d)
        rc, out, err = self.run_cli(["warm", "--target-copies", "2"])
        self.assertEqual(rc, 1); self.assertIn("WARM FAIL", err)                                       # the run failed: 1

    def test_warm_makes_the_shape_ready(self):
        rc, out, err = self.run_cli(["warm", "--target-copies", "2"])
        self.assertEqual(rc, 0, err)
        d = os.path.join(self.root, "L80_c2")
        self.assertTrue(os.path.isfile(os.path.join(d, "features_L80_c2.npz")))
        self.assertIn("WARM PASS mode=exact shape=L80_c2 features=present", err)
        a = outputs.read_results(os.path.join(d, "warm"), "warm_A_stock")
        self.assertEqual(a["manifest"]["args"]["target_copies"], 2); self.assertEqual(a["manifest"]["args"]["weights"], "torch"); self.assertNotIn("XLA_FLAGS", a["manifest"]["env_seen"])
        b = outputs.read_results(os.path.join(d, "warm"), "warm_B_populate")
        self.assertIn("--xla_gpu_dump_autotune_results_to=", b["manifest"]["env_seen"]["XLA_FLAGS"]); self.assertEqual(b["manifest"]["args"]["weights"], "fastinit")
        self.assertEqual(b["manifest"]["features"]["source"], "frozen npz " + os.path.join(d, "features_L80_c2.npz"))
        self.assertIn("autotune=present", err)
        self.assertTrue(os.path.isfile(os.path.join(d, "xla_cache", "xla_autotune_results.pb")))
        rc, out, err = self.run_cli(["check", "--mode", "exact", "--target-copies", "2"])
        self.assertEqual(rc, 0); self.assertIn("row=C_p1warm_p2[P1,P2,P3] shape=L80_c2(258 tokens)", err); self.assertNotIn("not warm", err)
        self.assertIn(" mosaic.fast=" + INSTALL_VERDICT, err)                                # the install verdict of this box (not-installed | kit-bytes@<dir>)

    def test_warm_refuses_a_warm_shape(self):
        """The populate row always starts from an empty directory: a warm shape is refused by name; removing its directory warms it again."""
        d = self.warm_shape("L80_c2"); stale = os.path.join(d, "xla_cache", "stale_entry"); open(stale, "wb").write(b"old")
        rc, out, err = self.run_cli(["warm", "--target-copies", "2"])
        self.assertEqual(rc, 2); self.assertIn("already holds its features file and 2 P1 entries", err); self.assertIn(f"remove {d} to warm the shape again", err)
        self.assertTrue(os.path.isfile(stale)); self.assertEqual(open(os.path.join(d, "features_L80_c2.npz"), "rb").read(), b"frozen")   # nothing touched
        shutil.rmtree(d)
        rc, out, err = self.run_cli(["warm", "--target-copies", "2"])
        self.assertEqual(rc, 0, err)
        self.assertFalse(os.path.exists(stale))
        self.assertNotEqual(open(os.path.join(d, "features_L80_c2.npz"), "rb").read(), b"frozen")                          # re-frozen by arm A
        self.assertIn("WARM PASS mode=exact shape=L80_c2 features=present autotune=present", err)

    def test_exit_tally_line_at_process_exit(self):
        self.warm_shape()
        env = {**os.environ, "PYTHONPATH": _stubs.OPT_DIR + os.pathsep + core_src(), "MOSAIC_OPT_FORCE": "1"}          # a real process: the pin gates are not stubbed there
        out = subprocess.run([sys.executable, "-m", "mosaic_opt", "design", "--mode", "off", "--out", os.path.join(self.tmp, "out_tally")], capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("[mosaic-opt] EXIT pid=", out.stderr); self.assertIn("no lever applied in this process", out.stderr)
        self.assertTrue(out.stderr.rstrip().splitlines()[-1].startswith("[mosaic-opt] EXIT pid="), "the tally is the last line")
        self.assertTrue(out.stderr.rstrip().endswith("no lever applied in this process jax_cache=off autotune=off via=none autotune_sha256=none cache_entries=0"), out.stderr.rstrip().splitlines()[-1])

    def test_precision_variable_refused_before_launch(self):
        self.mp.setenv("JAX_DEFAULT_MATMUL_PRECISION", "highest")
        rc, out, err = self.run_cli(["design", "--mode", "off", "--out", os.path.join(self.tmp, "out_prec")])
        self.assertEqual(rc, 3); self.assertIn("JAX_DEFAULT_MATMUL_PRECISION must be unset", err)

    # boltz has no offline switch, so a missing checkpoint is a refusal everywhere a real design could reach its unguarded download
    # -- EXCEPT through `warm`, whose own Step 0 (stage_weights, the kit's sha256-checked fetch) is the sanctioned staging path and
    # must still be able to run when the checkpoint is exactly what's missing. End-to-end through cmd_warm/cmd_design (stack.activate's
    # dry-run gate + the driver route), not just data_path_gate() in isolation.

    def test_warm_proceeds_with_the_checkpoint_missing(self):
        """warm's own dry-run gate does not block Step 0 on a missing checkpoint -- that is exactly what Step 0 exists to fix."""
        os.remove(os.path.join(self.weights, "boltz", "boltz2_conf.ckpt"))
        rc, out, err = self.run_cli(["warm", "--target-copies", "2"])
        self.assertEqual(rc, 0, err)                                                          # not blocked by data_path_gate's refusal
        self.assertNotIn(WEIGHTS_MISSING_MARKER, err)
        self.assertIn("WARM PASS mode=exact shape=L80_c2", err)                                # every step ran: nothing was skipped by a refusal

    def test_design_refuses_before_load_with_the_checkpoint_missing(self):
        """design (a real model load, unlike warm) refuses by name before reaching boltz's own unguarded download."""
        self.warm_shape()
        os.remove(os.path.join(self.weights, "boltz", "boltz2_conf.ckpt"))
        out_dir = os.path.join(self.tmp, "out_no_ckpt")
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", out_dir])
        self.assertEqual(rc, 2)                                                               # cli.py's direct check: EXIT_USAGE (cli.py:164)
        self.assertIn(WEIGHTS_MISSING_MARKER, err); self.assertIn("mosaic-opt warm", err)
        self.assertFalse(os.path.exists(out_dir) and os.listdir(out_dir))                     # refused before any driver run, not just before load

    def test_design_refuses_with_cache_dir_unset(self):
        """An unset MOSAIC_CACHE_DIR resolves to boltz's own default cache -- the same unguarded-download gap as a missing checkpoint
        under a set directory -- so this refuses too: an unset cache dir has no legitimately-deferred case to distinguish (the kit's
        own configs always set it), unlike an absent-but-optional component elsewhere in this gate."""
        self.warm_shape()
        self.mp.delenv("MOSAIC_CACHE_DIR")
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", os.path.join(self.tmp, "out_unset")])
        self.assertEqual(rc, 2); self.assertIn("MOSAIC_CACHE_DIR unset", err)

    def test_design_inert_with_the_checkpoint_present(self):
        """The default (checkpoint present, set up by setUp for every test in this class): the gate is inert on this path -- the only
        panel-relevant one, since a warm always stages the checkpoint first."""
        self.warm_shape()
        out_dir = os.path.join(self.tmp, "out_present")
        rc, out, err = self.run_cli(["design", "--mode", "exact", "--out", out_dir, "--tag", "t"])
        self.assertEqual(rc, 0, err)
        self.assertNotIn(WEIGHTS_MISSING_MARKER, err); self.assertNotIn("MOSAIC_CACHE_DIR unset", err)
        self.assertTrue(os.path.isfile(outputs.results_path(out_dir, "t")))


if __name__ == "__main__":
    unittest.main()


class TestPartialVerdictGrammar(unittest.TestCase):
    """The ONE partial verdict (report.partial_verdict_line, printed by report.exit_for for design / warm / check): refused —
    `[mosaic-opt] NOT ACTIVE: partial activation — <levers> — <reason>; exit 3 (--allow-partial records and proceeds)`, exit 3; allowed —
    `[mosaic-opt] PARTIAL allowed: <levers> — <reason> (--allow-partial, recorded)`, exit 0; whole: silent, 0; failed: 1 whatever the partial state."""

    def test_literal_parts_and_exit_codes(self):
        from mosaic_opt import report
        rep = {"partial": True, "levers_fallback": ["P2"], "levers_applied": ["P1", "P2", "P3"]}
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(report.exit_for(0, False, rep, False, what="see r.json"), report.EXIT_NOT_ACTIVE)
        line = err.getvalue().strip()
        self.assertEqual(line, "[mosaic-opt] NOT ACTIVE: partial activation — P2 — the driver's manifest does not show them applied (see r.json); exit 3 (--allow-partial records and proceeds)")
        self.assertTrue(line.startswith(report.PREFIX + " " + report.PARTIAL_HEAD) and line.endswith("; exit 3 " + report.PARTIAL_TAIL))
        self.assertIs(rep["allow_partial"], False)
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(report.exit_for(0, False, rep, True, what="see r.json"), report.EXIT_OK)
        line = err.getvalue().strip()
        self.assertEqual(line, "[mosaic-opt] PARTIAL allowed: P2 — the driver's manifest does not show them applied (see r.json) (--allow-partial, recorded)")
        self.assertTrue(line.startswith(report.PREFIX + " " + report.ALLOWED_HEAD) and line.endswith(" " + report.ALLOWED_TAIL))
        self.assertIs(rep["allow_partial"], True)
        plan = {"partial": True, "levers_unavailable": ["P2", "P3"], "unavailable_reason": "call-site replacements"}
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(report.exit_for(0, False, plan, False, what="check"), 3)
        self.assertEqual(err.getvalue().strip(), "[mosaic-opt] NOT ACTIVE: partial activation — P2,P3 — the plan is partial on this box (call-site replacements); exit 3 (--allow-partial records and proceeds)")
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(report.exit_for(0, False, {"partial": False}, False), 0)
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(report.exit_for(1, False, dict(rep), True), report.EXIT_FAIL)
        self.assertEqual(report.exit_for(0, True, dict(rep), True), report.EXIT_FAIL)
        from mosaic_opt import cli
        self.assertIs(cli.exit_for, report.exit_for); self.assertEqual((cli.EXIT_OK, cli.EXIT_FAIL, cli.EXIT_USAGE, cli.EXIT_NOT_ACTIVE), (0, 1, 2, 3))


class TestDefaultModeIsFast(unittest.TestCase):
    """The package default is `fast` wherever a fast tier ships (modes.DEFAULT_MODE, the sibling kits' rule): `design` without --mode composes the fast
    row — and on a box whose installed mosaic does not carry the tier's lever module it is NOT ACTIVE by name (exit 3), never a stock run."""
    def test_default(self):
        self.assertEqual(modes.DEFAULT_MODE, "fast"); self.assertIn("fast", modes.MODES)
