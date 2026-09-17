
"""The activation contract on CPU (gates stubbed, the kit's repro_cache stood in for): refusals by name (no GPU, pins, no cache
directory, precision variables, P1 already on, backend initialised, a Boltz2 instance), the in-process report (P1 applied, P2/P3 —
the driver's call-site levers — stepping aside by name: not partial), idempotency, the mode change refusal, the instance counter, the exit
tally line, `off`.

Bare-interpreter form: no child process and no stack — the kit's `repro_cache` is a fake module installed in `sys.modules` and the gates are
stubbed, so the class runs green under pytest alone (no jax/torch, no installed package or venv, no GPU) and skips by name only when the
release tree is not around the package."""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

import pytest

from . import _stubs
from mosaic_opt import ActivationError, report, stack
import mosaic_opt


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestActivationRules(unittest.TestCase):
    def setUp(self):
        self.state = _stubs.StubState()
        self.state.restore()
        self.mp = pytest.MonkeyPatch()
        self.tmp = tempfile.mkdtemp(prefix="mosaic_opt_act_")
        for k in ("XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION", "MOSAIC_OPT", "MOSAIC_OPT_CACHE_DIR",
                  "MOSAIC_OPT_CACHE_ROOT", "MOSAIC_OPT_FORCE", "MOSAIC_CACHE_DIR", "MODEL_OPT_TARGET_GPU"):
            self.mp.delenv(k, raising=False)
        self.mp.setenv("MOSAIC_OPT_HOME", _stubs.TREE)
        self.fake = _stubs.FakeReproCache()
        self.mp.setattr(stack, "p1_lever", lambda: self.fake)

    def tearDown(self):
        self.mp.undo()
        self.state.restore()

    def _ready(self, gpu=None):
        _stubs.stub_gates(self.mp, gpu=gpu)
        self.mp.setenv("MOSAIC_OPT_CACHE_DIR", os.path.join(self.tmp, "p1"))

    def test_no_gpu_refused_by_name(self):
        _stubs.stub_gates(self.mp, gpu={"name": None, "cc": None, "sm": None, "probe": "nvidia-smi rc=9"})
        self.mp.setenv("MOSAIC_OPT_CACHE_DIR", os.path.join(self.tmp, "p1"))
        rep = mosaic_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("no GPU visible", rep["reason"])
        self.assertEqual(self.fake.calls, [])
        with self.assertRaises(ActivationError):
            stack.activate("exact", strict=True)

    def test_pins_refused_by_name(self):
        _stubs.stub_gates(self.mp, pins_ok=False)
        rep = mosaic_opt.enable("exact")
        self.assertIn("not at the pinned commit", rep["reason"])
        self.state.restore(); self.mp.setattr(stack, "p1_lever", lambda: self.fake)
        _stubs.stub_gates(self.mp, versions_ok=False)                                      # the stack's versions differ from the kit's pins: named on the activation line, never refused
        self.mp.setenv("MOSAIC_OPT_CACHE_DIR", os.path.join(self.tmp, "p1"))
        rep = mosaic_opt.enable("exact")
        self.assertTrue(rep["active"], rep.get("reason")); self.assertTrue(any("stack differs from the kit's pins" in n for n in rep["notes"]), rep["notes"])
        self.assertIn("notes=stack differs from the kit's pins: stub", report.activation_line(rep))

    def test_no_cache_directory_refused_by_name(self):
        _stubs.stub_gates(self.mp)
        rep = mosaic_opt.enable("exact")
        self.assertFalse(rep["active"])
        self.assertIn("MOSAIC_OPT_CACHE_DIR", rep["reason"]); self.assertIn("MOSAIC_OPT_CACHE_ROOT", rep["reason"])
        self.assertEqual(self.fake.calls, [])

    def test_cache_root_gives_the_campaign_directory(self):
        _stubs.stub_gates(self.mp)
        self.mp.setenv("MOSAIC_OPT_CACHE_ROOT", os.path.join(self.tmp, "root"))
        rep = mosaic_opt.enable("exact")
        self.assertTrue(rep["active"])
        self.assertEqual(rep["p1"]["cache_dir"], os.path.join(self.tmp, "root", stack.CAMPAIGN_DIRNAME))

    def test_precision_variables_refused(self):
        self._ready()
        self.mp.setenv("JAX_ENABLE_X64", "1")
        rep = mosaic_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("JAX_ENABLE_X64", rep["reason"]); self.assertIn("must be unset", rep["reason"])

    def test_in_process_route_applies_p1_and_names_the_call_site_levers_aside(self):
        self._ready()
        err = io.StringIO()
        with redirect_stderr(err):
            rep = mosaic_opt.enable("exact")
        self.assertTrue(rep["active"])
        self.assertEqual(rep["route"], "in-process")
        self.assertEqual(rep["levers_applied"], ["P1"])
        self.assertEqual(rep["levers_unavailable"], []); self.assertFalse(rep["partial"])   # P2, P3 are the driver's own call-site replacements: they step aside BY NAME, never a partial activation
        self.assertEqual(rep["levers_driver_only"], ["P2", "P3"]); self.assertEqual(rep["levers_aside"], {"P2": "driver_only", "P3": "driver_only"})
        self.assertEqual(rep["driver_only_reason"], stack.CALL_SITE_REASON); self.assertNotIn("unavailable_reason", rep)
        self.assertEqual(rep["p1"]["autotune"], "dump")                         # first process: no autotune file yet
        self.assertEqual(self.fake.calls, [os.path.join(self.tmp, "p1")])
        self.assertIn("--xla_gpu_dump_autotune_results_to=", os.environ["XLA_FLAGS"])
        line = err.getvalue()
        self.assertIn("[mosaic-opt] ACTIVE mode=exact route=in-process row=C_p1warm_p2[P1]-aside[P2:driver_only,P3:driver_only] ", line)
        self.assertIn("levers=P1 unavailable=none p1=dump:", line); self.assertNotIn(" partial=", line); self.assertNotIn("NOT ACTIVE", line)
        self.assertNotIn("allow_partial", line); self.assertFalse(rep["allow_partial"])
        self.assertEqual(report.partial_levers(rep), []); self.assertEqual(report.partial_field(rep), "")
        genuine = dict(rep, partial=True, levers_unavailable=["P1"], unavailable_reason="stub")   # the verdict's one meaning, rendered: a lever the route should apply and did not
        self.assertEqual(report.partial_levers(genuine), ["P1"]); self.assertIn(" partial=P1 (", report.activation_line(genuine)); self.assertEqual(report.partial_levers(dict(genuine, partial=False)), [])
        self.assertIn(" allow_partial=1", report.activation_line(dict(genuine, allow_partial=True)))   # the opt-out, recorded beside it
        self.assertTrue(report.allow_partial_env({"MOSAIC_OPT_ALLOW_PARTIAL": "1"})); self.assertFalse(report.allow_partial_env({})); self.assertFalse(report.allow_partial(False))
        self.assertEqual(mosaic_opt.status()["mode"], "exact")
        self.assertTrue(rep["row_line"].startswith("C_p1warm_p2[P1]-aside[P2:driver_only,P3:driver_only] "), rep["row_line"])
        # the exit tally reads the kit's own P1 key
        self.assertIsNotNone(report._TALLY["fn"])
        self.assertIn("EXIT pid=", report._TALLY["fn"]()); self.assertIn("route=in-process", report._TALLY["fn"]())

    def test_autotune_present_means_load(self):
        self._ready()
        os.makedirs(os.path.join(self.tmp, "p1"), exist_ok=True)
        open(os.path.join(self.tmp, "p1", "xla_autotune_results.pb"), "wb").write(b"x")
        rep = mosaic_opt.enable("exact")
        self.assertEqual(rep["p1"]["autotune"], "load")
        self.assertIn("--xla_gpu_load_autotune_results_from=", os.environ["XLA_FLAGS"])

    def test_idempotent_and_mode_change_refused(self):
        self._ready()
        with redirect_stderr(io.StringIO()):
            r1 = mosaic_opt.enable("exact")
            r2 = mosaic_opt.enable("exact")
        self.assertEqual(r1["pid"], r2["pid"]); self.assertEqual(self.fake.calls, [os.path.join(self.tmp, "p1")])
        r3 = mosaic_opt.enable("off")
        self.assertFalse(r3["active"]); self.assertTrue(r3.get("refused")); self.assertIn("already active as mode=exact", r3["reason"])
        with self.assertRaises(ActivationError):
            mosaic_opt.enable("off", strict=True)

    def test_p1_already_on_by_the_environment_row_is_refused(self):
        self._ready()
        self.mp.setenv("XLA_FLAGS", "--xla_gpu_load_autotune_results_from=/x/xla_autotune_results.pb")
        rep = mosaic_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("P1 already on in this process", rep["reason"]); self.assertIn("XLA_FLAGS", rep["reason"])
        self.assertEqual(self.fake.calls, [])
        self.state.restore(); self.mp.setattr(stack, "p1_lever", lambda: self.fake); self._ready()
        self.mp.delenv("XLA_FLAGS", raising=False); self.mp.setenv("JAX_COMPILATION_CACHE_DIR", "/x")
        rep = mosaic_opt.enable("exact")
        self.assertIn("JAX_COMPILATION_CACHE_DIR", rep["reason"])

    def test_backend_initialised_is_refused_by_name(self):
        self._ready()
        self.mp.setattr(stack, "backend_initialised", lambda: (True, "jax backend initialised (cuda)"))
        rep = mosaic_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("backend initialised", rep["reason"]); self.assertIn("pcc.enable", rep["reason"])
        self.assertEqual(self.fake.calls, [])

    def test_existing_model_instance_is_refused_by_name(self):
        self._ready()
        sys.path.insert(0, _stubs.fake_mosaic_package(self.tmp))
        import mosaic.models.boltz2 as mb
        keep = mb.Boltz2()
        rep = mosaic_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("1 Boltz2 instance(s) already exist", rep["reason"])
        self.assertEqual(self.fake.calls, [])
        del keep

    def test_instance_counter_counts_constructions_after_activation(self):
        self._ready()
        sys.path.insert(0, _stubs.fake_mosaic_package(self.tmp))
        with redirect_stderr(io.StringIO()):
            rep = mosaic_opt.enable("exact")                                      # mosaic not imported yet: the finder arms
        self.assertTrue(rep["active"])
        self.assertEqual(rep["instances"], {"n": 0, "method": "none", "built": 0})
        import mosaic.models.boltz2 as mb
        a, b = mb.Boltz2(), mb.Boltz2()
        chk = stack.instance_check()
        self.assertEqual((chk["n"], chk["method"], chk["built"]), (2, "counted", 2))
        del a
        self.assertEqual(stack.instance_check()["n"], 1)
        del b

    def test_instance_counter_wraps_an_already_imported_class(self):
        self._ready()
        sys.path.insert(0, _stubs.fake_mosaic_package(self.tmp))
        import mosaic.models.boltz2 as mb
        with redirect_stderr(io.StringIO()):
            rep = mosaic_opt.enable("exact")
        self.assertTrue(rep["active"])
        x = mb.Boltz2()
        self.assertEqual(stack.instance_check()["n"], 1)
        del x

    def test_off_is_inactive_without_touching_anything(self):
        rep = mosaic_opt.enable("off")
        self.assertFalse(rep["active"]); self.assertIn("stock", rep["reason"])
        self.assertEqual(self.fake.calls, []); self.assertNotIn("XLA_FLAGS", os.environ)
        self.assertEqual(mosaic_opt.status()["mode"], "off")

    def test_target_gpu_note(self):
        self._ready()
        self.mp.setenv("MODEL_OPT_TARGET_GPU", "H200")
        with redirect_stderr(io.StringIO()):
            rep = mosaic_opt.enable("exact")
        self.assertTrue(rep["active"])
        self.assertTrue(any("MODEL_OPT_TARGET_GPU=H200" in n for n in rep["notes"]))

    def test_target_gpu_note_by_product_name(self):
        """The target GPU word against the product name of the card seen: another GPU type is a note; the memory size is recorded on the
        activation line (`report.gpu_label`), never compared."""
        self.mp.setenv("MOSAIC_OPT_HOME", _stubs.TREE); self.mp.setenv("MODEL_OPT_TARGET_GPU", "H100")
        self.assertIsNone(stack.target_gpu_note(_stubs.gpu_h100()))
        self.assertIsNone(stack.target_gpu_note({"name": "NVIDIA H100 NVL", "sm": "sm90", "memory_mib": 95830}))
        self.assertIn("another GPU type", stack.target_gpu_note({"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920}))
        self.assertIsNone(stack.target_gpu_note({"name": None}))                                                     # no GPU name read: no claim
        self.assertEqual(report.gpu_label(_stubs.gpu_h100()), "NVIDIA H100 80GB HBM3(sm90,81559MiB)")
        self.assertEqual(report.gpu_label({"name": "NVIDIA H100 NVL", "sm": "sm90", "memory_mib": 95830}), "NVIDIA H100 NVL(sm90,95830MiB)")

    def test_activation_line_carries_the_install_verdict(self):
        self._ready()
        err = io.StringIO()
        with redirect_stderr(err):
            rep = mosaic_opt.enable("exact")
        box = stack.installed_fast_check()                                                      # this box: no upstream mosaic (not-installed) or the kit image (kit-bytes@<dir>)
        self.assertTrue(rep["active"]); self.assertEqual(rep["kit_install"]["installed"], bool(box.get("installed")))
        line = err.getvalue()
        self.assertIn("gpu=NVIDIA H100 80GB HBM3(sm90,81559MiB)", line); self.assertIn(" mosaic.fast=" + stack.installed_fast_label(box), line)
        self.assertIn("kit_install", stack.status())

    def test_installed_fast_missing_a_kit_file_is_refused(self):
        import shutil
        self._ready()
        site = os.path.join(self.tmp, "site"); pkg = os.path.join(site, "mosaic"); os.makedirs(pkg); open(os.path.join(pkg, "__init__.py"), "w").close()
        shutil.copytree(os.path.join(_stubs.KIT, "mosaic_fast"), os.path.join(pkg, "fast"))
        os.remove(os.path.join(pkg, "fast", "repro_cache.py"))
        sys.path.insert(0, site)
        with redirect_stderr(io.StringIO()):
            rep = mosaic_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("installed mosaic/fast at", rep["reason"]); self.assertIn("ABSENT(repro_cache.py)", rep["reason"])
        self.assertEqual(self.fake.calls, [])
        self.assertIn("mosaic.fast=ABSENT(repro_cache.py)", report.activation_line(stack.activate("exact", dry_run=True)))

    def test_dry_run_applies_nothing(self):
        self._ready()
        rep = stack.activate("exact", dry_run=True)
        self.assertTrue(rep["dry_run"]); self.assertFalse(rep["active"]); self.assertIsNone(rep["would_refuse"])
        self.assertEqual(self.fake.calls, []); self.assertNotIn("XLA_FLAGS", os.environ)
        self.assertIsNone(stack._REPORT)
        line = report.activation_line(rep)
        self.assertIn("DRY-RUN mode=exact route=in-process row=C_p1warm_p2[P1]-aside[P2:driver_only,P3:driver_only] levers=P1 p1=", line)   # the plan names the call-site levers aside, as enable() does
        self.assertNotIn(" partial=", line); self.assertEqual((rep["partial"], rep["levers_unavailable"], rep["levers_driver_only"]), (False, [], ["P2", "P3"]))


if __name__ == "__main__":
    unittest.main()

