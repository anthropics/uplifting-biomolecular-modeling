"""GPU accounting: the GPU never gates activation. The kit's own per-key behaviour stands and the package reports it — `partial`,
`levers_unavailable` (from the kit's own applied/refused records), `gpu`, `gpu_supported` (stack.SUPPORTED_CC is data for the README
sentence only). With levers left off the startup line names them; with none left off it is the plain ACTIVE line.
`check` reports the same from the resolved environment. Mocked capabilities: 8.9 (L40S: no kit README row) and 9.0 (H100); the
installed Triton is mocked too (the kernel key is `<cc>|<triton major.minor>` of whatever is installed). cc 8.0 (A100) has its own
kit README row: tests/test_a100_rows.py."""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock

import protenix_opt
from protenix_opt import cli, stack, kits, modes
from opt_core import instances
from protenix_opt.modes import MODES

L40S = {"name": "NVIDIA L40S", "sm": "sm89", "cc": "8.9", "probe": "nvidia-smi"}
H100 = {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90", "cc": "9.0", "probe": "nvidia-smi"}
SOME_MARKERS = ["T1:fused", "DEADSKIP:hooked", "LAZY_INIT:1"]          # what a kit might report on a GPU where its kernel levers refuse
ALL_ON = [n for n in MODES["exact"] if not stack.LEVERS[n].row_dependent]
ROW_DEP = [n for n in MODES["exact"] if stack.LEVERS[n].row_dependent]


class _Base(unittest.TestCase):
    def setUp(self):
        stack._REPORT = None
        self._env = dict(os.environ)
        self._path = list(sys.path)
        self._meta = list(sys.meta_path)                                  # an activation installs finders (the instance counter, the kernel routes)
        for k in ("PROTENIX_OPT", "PROTENIX_OPT_FORCE"):
            os.environ.pop(k, None)

    def tearDown(self):
        stack._REPORT = None
        instances._COUNTERS.clear()
        os.environ.clear(); os.environ.update(self._env)
        sys.path[:] = self._path
        sys.meta_path[:] = self._meta


class TestCc89(_Base):
    def test_activates_and_names_what_the_kit_left_off(self):
        err = io.StringIO()
        with mock.patch.object(stack, "protenix_version", return_value="2.0.0"), mock.patch.object(stack, "gpu_probe_smi", return_value=L40S), mock.patch.object(stack, "gpu_info", return_value=L40S), \
             mock.patch.object(modes, "triton_version", return_value="3.7.1"), mock.patch.object(stack, "_apply", return_value=(SOME_MARKERS, {})), redirect_stderr(err):
            r = protenix_opt.enable("exact")
            r2 = protenix_opt.enable("exact", strict=True)          # the autoload route: active, so strict does not raise; no second line
        self.assertTrue(r["active"]); self.assertEqual(r["mode"], "exact"); self.assertTrue(r["partial"])
        self.assertIs(r["gpu_supported"], False); self.assertEqual(r["gpu"]["cc"], "8.9")
        self.assertIn("PTX_BLK", os.environ, "the mode's environment is exported; the kit's own gates decide per lever")
        self.assertEqual(r["levers_unavailable"], r["levers_fallback"]); self.assertTrue(r["levers_unavailable"])
        for n in ("blk2_block_path", "trimul_core_exact", "stackgraph"):
            self.assertIn(n, r["levers_unavailable"], n)
        for n in ("t1_fused_transition", "deadskip", "lazy_init"):
            self.assertIn(n, r["levers_applied"], n)
        self.assertEqual(sorted(r["levers_applied"] + r["levers_unavailable"] + r["levers_not_in_arm"]), sorted(MODES["exact"]), "total accounting")
        self.assertEqual(r["row_key"], "other", "cc 8.9 has no kit README row"); self.assertEqual(r["levers_not_in_arm"], ["pad8", "glue_v2", "mk_pf", "dit_attn_exact", "triatt_exact", "atom_attn_exact", "triatt_prologue_cuda"])
        lines = err.getvalue().splitlines()
        card = [l for l in lines if l.startswith("[protenix-opt] CARD LEVER SET ")]
        self.assertEqual(len(card), 1, err.getvalue()); self.assertIn("kernel_key=8.9|3.7: not in the arm on this card — pad8 (", card[0]); self.assertIn("glue_v2 (", card[0]); self.assertIn("mk_pf (", card[0])
        lines = [l for l in lines if not l.startswith("[protenix-opt] CARD LEVER SET ")]
        self.assertEqual(len(lines), 1, err.getvalue())
        self.assertEqual(lines[0], "[protenix-opt] ACTIVE mode=exact n_gpu=1 sharding=none (partial on NVIDIA L40S, cc 8.9: kernel levers unavailable — "
                         + ", ".join(r["levers_unavailable"]) + ")")
        self.assertEqual(r2["levers_unavailable"], r["levers_unavailable"])

    def test_probe_is_never_overwritten_by_torch(self):
        """The nvidia-smi (or injected) probe stays; torch's view is recorded beside it and a disagreement is a note."""
        other = {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90", "cc": "9.0", "probe": "torch"}
        err = io.StringIO()
        with mock.patch.object(stack, "protenix_version", return_value="2.0.0"), mock.patch.object(stack, "gpu_probe_smi", return_value=L40S), \
             mock.patch.object(stack, "gpu_info", return_value=other), mock.patch.object(stack, "_apply", return_value=(SOME_MARKERS, {})), redirect_stderr(err):
            r = protenix_opt.enable("exact")
        self.assertEqual((r["gpu"]["name"], r["gpu"]["cc"], r["gpu"]["sm"], r["gpu"]["probe"]), ("NVIDIA L40S", "8.9", "sm89", "nvidia-smi"))
        self.assertEqual(r["gpu"]["torch"]["cc"], "9.0"); self.assertTrue(any("GPU probes disagree" in n for n in r["notes"]))
        self.assertIn("(partial on NVIDIA L40S, cc 8.9:", err.getvalue())

    def test_check_reports_the_same_from_the_resolved_environment(self):
        err = io.StringIO()
        with mock.patch.object(stack, "protenix_version", return_value="2.0.0"), mock.patch.object(stack, "gpu_probe_smi", return_value=L40S), redirect_stderr(err):
            r = stack.activate("exact", dry_run=True)
            rc = cli.main(["check", "--mode", "exact"])
        self.assertTrue(r["dry_run"]); self.assertFalse(r["active"]); self.assertIs(r["gpu_supported"], False)
        self.assertEqual(r["levers_unavailable"], r["levers_fallback"]); self.assertEqual(r["partial"], bool(r["levers_unavailable"]))
        self.assertTrue(all(stack.LEVERS[n].probe == "env" or n in stack.BLK2_LEVERS for n in r["levers_unavailable"]), "only switch-level and card-level facts before activation")
        self.assertTrue({"blk2_block_path", "blk2_chunked_exact"} <= set(r["levers_unavailable"]), "cc 8.9 has no BLK2 cells: the block path cannot engage there (named, the run refuses)")
        self.assertTrue(all(r["fallback_reasons"][n].startswith(stack.NO_BLK2_CELLS) for n in ("blk2_block_path", "blk2_chunked_exact")))
        self.assertEqual(sorted(r["levers_applied"] + r["levers_unavailable"] + r["levers_not_in_arm"]), sorted(MODES["exact"]))
        self.assertNotIn("PTX_BLK", os.environ); self.assertIsNone(stack._REPORT)
        self.assertEqual(sum(1 for l in err.getvalue().splitlines() if l.startswith("[protenix-opt] WEIGHTS ")), 1)   # check names the checkpoint's digest state once (a note, never a gate)
        self.assertEqual(sum(1 for l in err.getvalue().splitlines() if l.startswith("[protenix-opt] CARD LEVER SET ")), 2)   # once per dry run: this card's row lacks pad8 / glue_v2 / mk_pf
        lines = [l for l in err.getvalue().splitlines() if not l.startswith(("[protenix-opt] KIT ", "[protenix-opt] CHECK console script:", "[protenix-opt] WEIGHTS ", "[protenix-opt] CARD LEVER SET "))]   # check's kit-bytes lines precede its activation line; the console-script advisory follows it
        self.assertEqual(len(lines), 2 + (1 if r["partial"] else 0)); self.assertEqual(lines[0], lines[1])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE if r["partial"] else 0, "a partial dry run refuses by name: the code pred exits with")
        if r["partial"]:
            self.assertTrue(lines[2].startswith("[protenix-opt] NOT ACTIVE: partial activation refused (dry run) — "), lines[2]); self.assertNotIn("--allow-partial", lines[2])
        self.assertEqual(sum(1 for l in err.getvalue().splitlines() if l.startswith("[protenix-opt] KIT ")), 0)                # no per-unit file account
        self.assertTrue(lines[0].startswith("[protenix-opt] DRY-RUN mode=exact"))
        if r["partial"]:
            self.assertIn("(partial on NVIDIA L40S, cc 8.9: kernel levers unavailable — " + ", ".join(r["levers_unavailable"]) + ")", lines[0])
        else:
            self.assertNotIn("partial", lines[0])


class TestCc90(_Base):
    def test_plain_active_line_when_nothing_is_left_off(self):
        err = io.StringIO()
        with mock.patch.object(stack, "protenix_version", return_value="2.0.0"), mock.patch.object(stack, "gpu_probe_smi", return_value=H100), mock.patch.object(stack, "gpu_info", return_value=H100), \
             mock.patch.object(stack, "_apply", return_value=(SOME_MARKERS, {})), \
             mock.patch.object(stack, "_classify", return_value=(ALL_ON, [], ROW_DEP, {})), redirect_stderr(err):
            r = protenix_opt.enable("exact")
        self.assertTrue(r["active"]); self.assertFalse(r["partial"]); self.assertEqual(r["levers_unavailable"], []); self.assertIs(r["gpu_supported"], True)
        self.assertIn("PTX_BLK", os.environ)
        lines = err.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("[protenix-opt] ACTIVE mode=exact n_gpu=1 sharding=none protenix=2.0.0 gpu="), lines[0])
        self.assertIn(" levers=", lines[0]); self.assertNotIn("partial", lines[0])

    def test_partial_line_on_9_0_when_the_kit_refuses(self):
        """The GPU family does not decide the line: what the kit's own records say does."""
        err = io.StringIO()
        with mock.patch.object(stack, "protenix_version", return_value="2.0.0"), mock.patch.object(stack, "gpu_probe_smi", return_value=H100), mock.patch.object(stack, "gpu_info", return_value=H100), \
             mock.patch.object(stack, "_apply", return_value=(SOME_MARKERS, {})), redirect_stderr(err):
            r = protenix_opt.enable("exact")
        self.assertTrue(r["partial"]); self.assertIs(r["gpu_supported"], True)
        self.assertTrue(err.getvalue().startswith("[protenix-opt] ACTIVE mode=exact n_gpu=1 sharding=none (partial on NVIDIA H100 80GB HBM3, cc 9.0: kernel levers unavailable — "))

    def test_dry_run_unchanged(self):
        err = io.StringIO()
        with mock.patch.object(stack, "protenix_version", return_value="2.0.0"), mock.patch.object(stack, "gpu_probe_smi", return_value=H100), redirect_stderr(err):
            r = stack.activate("exact", dry_run=True)
        self.assertTrue(r["dry_run"]); self.assertIs(r["gpu_supported"], True); self.assertIn("PTX_BLK", r["env"])
        served = "trimul_core_exact"                                                                                # the cc-9.0 kit README rows select the package's exact TriMul provider (PTX_E_TRIMUL=tx); a key without a row keeps fpf_trimul_exact
        self.assertIn(served, r["levers_applied"]); self.assertIn("DRY-RUN mode=exact", err.getvalue())


if __name__ == "__main__":
    unittest.main()
