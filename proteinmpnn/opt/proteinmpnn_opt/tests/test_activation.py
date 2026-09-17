"""Activation: the gates (carried bytes, stock pin, GPU), the report fields, idempotence, the refusal of a second mode and of late activation,
the env-vs-CLI disagreement rule. The stock pin and the GPU are stubbed (no checkout, no GPU on the test box); the kit bytes are the tree's."""
import os
import shutil
import tempfile
import unittest
from unittest import mock

import proteinmpnn_opt
from proteinmpnn_opt import ActivationError, modes, report, stack

PINNED = {"pinned": True, "findings": [], "detail": {"head": "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57", "commit": "26ec57ac976ade5379920dbd43c7f97a91cf82de"}}
UNPINNED = {"pinned": False, "findings": ["MPNN_DIR='/nowhere' is not a directory"], "detail": {}}


GPU = {"name": "NVIDIA H100 80GB HBM3", "mem_mib": 81559, "cc": "9.0", "sm": "sm_90", "count": 1}


class Base(unittest.TestCase):
    def setUp(self):
        stack.reset_for_tests()
        self._env = dict(os.environ)
        for k in stack.PACKAGE_ENV + (stack.ENV_TARGET_GPU, stack.ENV_TARGET_GPU_MEM):
            os.environ.pop(k, None)
        self.p_pins = mock.patch.object(stack, "check_pins", return_value=dict(PINNED))
        self.p_gpu = mock.patch.object(stack, "gpu_info", return_value=dict(GPU))          # a GPU box unless a test says otherwise (a CPU box refuses the kit modes by name: TestGates.test_cpu_box_refuses_by_name)
        self.p_pins.start(); self.p_gpu.start()

    def tearDown(self):
        self.p_pins.stop(); self.p_gpu.stop()
        os.environ.clear(); os.environ.update(self._env)
        stack.reset_for_tests()


class TestGates(Base):
    def test_carried_files_are_present(self):
        n_ok, bad = stack.check_kits()
        self.assertEqual(bad, [])
        self.assertEqual(n_ok, sum(len(v) for v in stack.KIT_REQUIRED_FILES.values()))

    def test_dry_run_exact_reports_the_plan(self):
        rep = stack.activate("exact", "soluble", dry_run=True)
        self.assertFalse(rep["active"]); self.assertTrue(rep["dry_run"]); self.assertIsNone(rep["reason"])
        self.assertEqual(rep["route"], "worker"); self.assertEqual(rep["bb_batch"], 16)
        self.assertEqual(rep["proteinmpnn_version"], "8907e667")
        self.assertTrue(rep["kits"]["present"])
        self.assertEqual(rep["probe"]["verdict"], "measured at launch"); self.assertTrue(rep["probe"]["requested"])   # the exact line requests the probe-gated --hybrid_gemm
        self.assertEqual(rep["levers_fallback"], []); self.assertEqual(rep["partial"], []); self.assertEqual(rep["levers_unavailable"], [])
        self.assertEqual(rep["gpu"]["sm"], "sm_90"); self.assertEqual(rep["probe"]["kit_observed"], "PASS")

    def test_cpu_box_refuses_by_name(self):
        """A mode is all of its levers: without a CUDA device the worker's CUDA-graph levers cannot run, so the mode refuses by name (the plan names them
        in ``partial``; ``design`` raises, nothing launches) — never a run under the mode's name with a subset; ``--allow-partial`` is the one recorded override."""
        with mock.patch.object(stack, "gpu_info", return_value=None):
            rep = stack.activate("exact", "soluble", dry_run=True)
            self.assertIsNone(rep["gpu"]); self.assertFalse(rep["active"])
            self.assertEqual(rep["partial"], ["graph_rng", "single_graph", "fused_draw"]); self.assertEqual(set(rep["levers_unavailable"]), {"graph_rng", "single_graph", "fused_draw"})
            self.assertIn("mode exact is all of its levers and graph_rng,single_graph,fused_draw cannot run on this box (no CUDA device", rep["reason"])
            self.assertIn("--mode off runs the stock command line here", rep["reason"]); self.assertIn("--allow-partial", rep["reason"])
            with self.assertRaises(ActivationError) as cm:                          # design / warm: refused before anything launches
                proteinmpnn_opt.enable("exact", "soluble")
            self.assertIn("all of its levers", str(cm.exception)); self.assertFalse(proteinmpnn_opt.status()["active"])
            rep = stack.activate("exact", "soluble", dry_run=True, allow_partial=True)   # the recorded override: the levers that can run do
            self.assertIsNone(rep["reason"]); self.assertEqual(rep["partial"], ["graph_rng", "single_graph", "fused_draw"]); self.assertTrue(rep["allow_partial"])
            os.environ[stack.ENV_ALLOW_PARTIAL] = "1"                                 # its environment spelling
            rep = proteinmpnn_opt.enable("exact", "soluble")
            self.assertTrue(rep["active"]); self.assertEqual(rep["partial"], ["graph_rng", "single_graph", "fused_draw"])


    def test_gpu_target_and_probe_kit_observed(self):
        h100 = {"name": "NVIDIA H100 80GB HBM3", "mem_mib": 81559, "cc": "9.0", "sm": "sm_90", "count": 1}
        nvl = {"name": "NVIDIA H100 NVL", "mem_mib": 95830, "cc": "9.0", "sm": "sm_90", "count": 1}
        os.environ[stack.ENV_TARGET_GPU] = "H100"; os.environ[stack.ENV_TARGET_GPU_MEM] = "81559"
        with mock.patch.object(stack, "gpu_info", return_value=h100):
            rep = stack.activate("exact", "soluble", dry_run=True)
            self.assertTrue(rep["gpu_matches_target"]); self.assertFalse(rep["partial"]); self.assertEqual(rep["levers_unavailable"], [])
            self.assertEqual(rep["probe"]["kit_observed"], "PASS")
            self.assertEqual((rep["target_gpu"], rep["target_gpu_mem_mib"]), ("H100", 81559))
            self.assertIn(" target=H100/81559MiB match=yes ", report.activation_line(rep))
            os.environ[stack.ENV_TARGET_GPU] = "L40S"
            rep = stack.activate("exact", "soluble", dry_run=True)
            self.assertFalse(rep["gpu_matches_target"]); self.assertIn(" target=L40S/81559MiB match=no ", report.activation_line(rep))
        os.environ[stack.ENV_TARGET_GPU] = "H100"
        with mock.patch.object(stack, "gpu_info", return_value=nvl):           # the name contains the target; the memory total is another class
            rep = stack.activate("exact", "soluble", dry_run=True)
            self.assertFalse(rep["gpu_matches_target"]); self.assertIsNone(rep["reason"])   # reported, not refused (the kit's own refusals only)
            self.assertIn("gpu=NVIDIA H100 NVL(sm_90) target=H100/81559MiB match=no", report.activation_line(rep))
            del os.environ[stack.ENV_TARGET_GPU_MEM]                             # without the memory total the gate is the name only
            self.assertTrue(stack.activate("exact", "soluble", dry_run=True)["gpu_matches_target"])
        self.assertTrue(stack.gpu_matches_target(h100, "H100", 81559)); self.assertFalse(stack.gpu_matches_target(nvl, "H100", 81559))
        self.assertTrue(stack.gpu_matches_target(dict(h100, mem_mib=81000), "H100", 81559))     # within 1 %
        self.assertFalse(stack.gpu_matches_target(dict(h100, mem_mib=None), "H100", 81559))
        self.assertIsNone(stack.gpu_matches_target(None, "H100", 81559)); self.assertIsNone(stack.gpu_matches_target(h100, None))
        del os.environ[stack.ENV_TARGET_GPU]
        self.assertNotIn("target=", report.activation_line(stack.activate("exact", "soluble", dry_run=True)))

    def test_fast_refused_by_name_in_dry_run_and_enable(self):
        pointer = "proteinmpnn ships no fast tier: select --mode exact"
        rep = stack.activate("fast", "soluble", dry_run=True)
        self.assertFalse(rep["active"]); self.assertEqual(rep["reason"], pointer)
        with self.assertRaises(modes.UnsupportedMode) as cm:                       # refused by name before any gate: a ModeError (ValueError), the pointer its text
            proteinmpnn_opt.enable("fast", "vanilla")
        self.assertEqual(str(cm.exception), pointer); self.assertIsInstance(cm.exception, ValueError)
        os.environ[stack.ENV_MODE] = "fast"
        with self.assertRaises(modes.UnsupportedMode):
            proteinmpnn_opt.enable(None, "soluble")
        self.assertFalse(proteinmpnn_opt.status()["active"])

    def test_unpinned_stock_refuses(self):
        with mock.patch.object(stack, "check_pins", return_value=dict(UNPINNED)):
            rep = stack.activate("exact", "soluble", dry_run=True)
            self.assertIn("stock pin", rep["reason"])
            with self.assertRaises(ActivationError) as cm:
                proteinmpnn_opt.enable("exact", "soluble")
            self.assertIn("stock pin", str(cm.exception))

    def test_weights_record_is_carried_never_a_reason(self):
        """The weights file of the pass (stock/check_pins.py check_weights) is named, not gated: a not-pinned record leaves no reason, the report and
        the line carry it, and the model_name of the pass reaches the pin check (the file it digests is the one the pass loads)."""
        record = {"file": "/opt/ProteinMPNN/vanilla_model_weights/pmft_v1.pt", "sha256": "ab" * 32, "pinned": None, "verdict": "not_pinned",
                  "line": "weights sha256=abababababab NOT PINNED — proceeding (stock/PINS.json names the tested weights)"}
        with mock.patch.object(stack, "check_pins", return_value={**PINNED, "detail": {**PINNED["detail"], "weights": record}}) as cp:
            rep = stack.activate("exact", "vanilla", dry_run=True, model_name="pmft_v1")
            cp.assert_called_once_with("vanilla", "pmft_v1", None)
            self.assertIsNone(rep["reason"]); self.assertEqual(rep["weights"], record)
            self.assertEqual(report.weights_line(rep["weights"]), "[proteinmpnn-opt] " + record["line"])
            rep = proteinmpnn_opt.enable("exact", "vanilla")                            # the API names no model_name: the variant's default file
            self.assertEqual(cp.call_args, mock.call("vanilla", None, None)); self.assertTrue(rep["active"]); self.assertEqual(rep["weights"]["verdict"], "not_pinned")
            self.assertEqual(proteinmpnn_opt.status()["weights"], record)
        self.assertIsNone(report.weights_line(None)); self.assertIsNone(report.weights_line({"file": "/x", "sha256": None, "line": None}))

class TestProcessRules(Base):
    def test_status_before_activation(self):
        s = proteinmpnn_opt.status()
        self.assertFalse(s["active"]); self.assertIn("not activated", s["reason"])

    def test_enable_is_idempotent_and_one_mode_per_process(self):
        r1 = proteinmpnn_opt.enable("exact", "soluble")
        self.assertTrue(r1["active"]); self.assertEqual(r1["applied"], "armed")
        r2 = proteinmpnn_opt.enable("exact", "soluble")
        self.assertIs(r1, r2)
        self.assertIs(proteinmpnn_opt.status(), r1)
        with self.assertRaises(ActivationError) as cm:
            proteinmpnn_opt.enable("exact", "vanilla")
        self.assertIn("already active", str(cm.exception))
        with self.assertRaises(ActivationError):
            proteinmpnn_opt.enable("off", "soluble")

    def test_late_activation_refused_by_name(self):
        proteinmpnn_opt.enable("exact", "soluble")
        stack.mark_launched()
        self.assertEqual(proteinmpnn_opt.status()["applied"], "launched")
        with self.assertRaises(ActivationError) as cm:
            proteinmpnn_opt.enable("exact", "soluble")
        self.assertIn("late activation refused", str(cm.exception))
        stack.reset_for_tests()
        stack.mark_launched()                                    # a launch without activation (mode off) also closes the process
        with self.assertRaises(ActivationError):
            proteinmpnn_opt.enable("exact", "soluble")

    def test_env_route_and_disagreement(self):
        os.environ[stack.ENV_MODE] = "exact"; os.environ[stack.ENV_VARIANT] = "vanilla"
        self.assertEqual(stack.resolve_mode_and_variant(None, None), ("exact", "vanilla"))
        self.assertEqual(stack.resolve_mode_and_variant("exact", "vanilla"), ("exact", "vanilla"))
        with self.assertRaises(ActivationError):
            stack.resolve_mode_and_variant("off", None)
        with self.assertRaises(ActivationError):
            stack.resolve_mode_and_variant(None, "soluble")
        rep = proteinmpnn_opt.enable()
        self.assertEqual((rep["mode"], rep["variant"]), ("exact", "vanilla"))
        del os.environ[stack.ENV_MODE]; del os.environ[stack.ENV_VARIANT]
        with self.assertRaises(modes.ModeError):                                 # no --mode, no $PROTEINMPNN_OPT: no default mode on this engine — a usage error naming off|exact
            stack.resolve_mode_and_variant(None, None)

    def test_kit_env_strips_package_variables(self):
        os.environ[stack.ENV_MODE] = "exact"; os.environ[stack.ENV_MPNN_DIR] = "/opt/ProteinMPNN"
        env = stack.kit_env()
        self.assertNotIn(stack.ENV_MODE, env); self.assertEqual(env[stack.ENV_MPNN_DIR], "/opt/ProteinMPNN"); self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
