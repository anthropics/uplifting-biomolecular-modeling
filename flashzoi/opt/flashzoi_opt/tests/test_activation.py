"""The activation core on a CPU box: the gates in order (dry run), enable()/status() for exact (the default), the class-level hook and the
refusal of a non-CUDA instance, idempotency and the one-mode-per-process rule, the late-activation rule (a foreign runner under other
knobs refuses; one under the mode's knobs is adopted), the attach path through a stub kit (the ACTIVE line once, the instance-level
forward, settle(), remove()). The kit's tables are used throughout (copied into the stub kit at test time)."""
import io
import os
import sys
import types
import unittest

import flashzoi_opt
from flashzoi_opt import modes, report, stack
from flashzoi_opt.tests import _stubs


class _Dev:
    def __init__(self, t):
        self.type = t

    def __repr__(self):
        return self.type


class TestGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kit = _stubs.require_kit()
        cls.table = modes.kit_table(cls.kit)
        _stubs.PINNED_GPU = cls.table["PINS"]["device_names"][0]

    def setUp(self):
        _stubs.reset_activation()

    def tearDown(self):
        _stubs.reset_activation()

    def test_kit_missing(self):
        t = _stubs.Tree(with_pins=True).enter()
        try:
            rep = stack.activate("exact", dry_run=True)
            self.assertIn("kit not found", rep["would_refuse"])
        finally:
            t.exit()

    def test_pins_missing_then_mismatch_then_switches_then_gpu(self):
        t = _stubs.Tree(with_pins=False, kit=self.kit).enter()
        try:
            rep = stack.activate("exact", dry_run=True)
            self.assertIn("stock/PINS.json not found", rep["would_refuse"])
            import json
            json.dump(_stubs.pins_stub(), open(os.path.join(t.root, "stock", "PINS.json"), "w"))
            undo = _stubs.stub_box(versions={"borzoi-pytorch": "0.5.0"})                    # another upstream release is another "stock": the ONE environment refusal
            try:
                rep = stack.activate("exact", dry_run=True)
                self.assertIn("stock pin not met: borzoi-pytorch 0.5.0 != pinned 0.5.1", rep["would_refuse"])
            finally:
                undo()
            undo = _stubs.stub_box(versions={"triton": "3.0.0"})                            # stack drift: named on the line (drift=[...]), served
            try:
                rep = stack.activate("exact", dry_run=True)
                self.assertIsNone(rep.get("would_refuse"), rep.get("would_refuse")); self.assertIn("stack triton 3.0.0 != pinned 3.1.0", rep["drift"])
                self.assertIn(" drift=[stack triton 3.0.0 != pinned 3.1.0]", report.dry_run_line(rep))
            finally:
                undo()
            undo = _stubs.stub_box()
            try:
                os.environ["NVIDIA_TF32_OVERRIDE"] = "0"                                   # a library numerics override: named as drift (both arms run under it), never refused
                rep = stack.activate("exact", dry_run=True)
                self.assertIsNone(rep.get("would_refuse")); self.assertEqual(rep["drift"], ["NVIDIA_TF32_OVERRIDE=0"]); self.assertEqual(rep["env_noted"], ["NVIDIA_TF32_OVERRIDE"])
                del os.environ["NVIDIA_TF32_OVERRIDE"]
                rep = stack.activate("exact", dry_run=True)
                self.assertIsNone(rep.get("would_refuse"), rep.get("would_refuse")); self.assertEqual(rep["drift"], [])
                self.assertNotIn("drift=", report.dry_run_line(rep))                        # the pinned environment: the line unchanged
                self.assertEqual(rep["components_planned"], list(self.table["LEVERS"]))
                self.assertEqual(rep["knobs"], self.table["knob_defaults"])
                self.assertEqual(rep["stack_key"], "9.0|3.1")
                self.assertTrue(rep["dry_run"]); self.assertFalse(rep["active"])
            finally:
                undo()
            undo = _stubs.stub_box(gpu_name="NVIDIA L40S")                                    # a GPU no pinned class or class record names: the levers engage, the class is named unpinned
            try:
                rep = stack.activate("exact", dry_run=True)
                self.assertIsNone(rep.get("would_refuse"), rep.get("would_refuse")); self.assertEqual(rep["device_class"], "unpinned")
                self.assertEqual(len(rep["drift"]), 1); self.assertTrue(rep["drift"][0].startswith("gpu NVIDIA L40S cc "), rep["drift"])
            finally:
                undo()
            rep = stack.activate("exact", dry_run=True)                       # the real CPU box: no GPU or a missing pin, never active
            self.assertTrue(rep.get("would_refuse"))
        finally:
            t.exit()

    def test_exact_default_dry_run(self):
        t = _stubs.Tree(kit=self.kit).enter()
        undo = _stubs.stub_box()
        try:
            d = stack.activate(None, dry_run=True)                                     # no mode given: the package default (exact)
            self.assertEqual(d["mode"], "exact"); self.assertIsNone(d.get("would_refuse"), d.get("would_refuse"))
            self.assertEqual(d["knobs"], dict(self.table["knob_defaults"])); self.assertEqual(d["knobs"]["numerics"], "tf32"); self.assertEqual(d["apply_line"], "kit.KitRunner(model)")
            self.assertEqual(d["numerics"], "tf32"); self.assertEqual(d["components_planned"], list(self.table["LEVERS"]))
            e = stack.activate("exact", dry_run=True)                                  # exact selected by name: the kit's defaults
            self.assertEqual(e["mode"], "exact"); self.assertEqual(e["knobs"], self.table["knob_defaults"]); self.assertEqual(e["apply_line"], "kit.KitRunner(model)"); self.assertEqual(e["numerics"], "tf32")
            e = stack.activate("exact", dry_run=True)
            self.assertEqual(e["apply_line"], "kit.KitRunner(model)"); self.assertEqual(e["numerics"], "tf32")
            with self.assertRaises(ValueError):
                flashzoi_opt.enable("turbo")
        finally:
            undo(); t.exit()

    def test_off_is_inactive_by_design(self):
        t = _stubs.Tree(kit=self.kit).enter()
        try:
            rep = flashzoi_opt.enable("off")
            self.assertFalse(rep["active"]); self.assertEqual(rep["mode"], "off"); self.assertIn("stock", rep["reason"])
            self.assertIn("already active as mode=off", flashzoi_opt.enable("exact")["reason"])
        finally:
            t.exit()

    def test_status_before_enable(self):
        s = flashzoi_opt.status()
        self.assertFalse(s["active"]); self.assertIn("has not run", s["reason"])


class TestHookAndLateRule(unittest.TestCase):
    """enable() on the stubbed box with the REAL kit module (imports on CPU): the hook, the CPU refusal, idempotency."""

    @classmethod
    def setUpClass(cls):
        cls.kit = _stubs.require_kit()
        _stubs.require_upstream()
        cls.table = modes.kit_table(cls.kit)
        _stubs.PINNED_GPU = cls.table["PINS"]["device_names"][0]

    def setUp(self):
        _stubs.reset_activation()
        self.tree = _stubs.Tree(kit=self.kit).enter()
        self.undo = _stubs.stub_box()

    def tearDown(self):
        self.undo(); self.tree.exit(); _stubs.reset_activation()

    def test_enable_arms_and_cpu_forward_refuses(self):
        import torch
        rep = flashzoi_opt.enable("exact")
        self.assertTrue(rep["active"]); self.assertEqual(rep["applied"], "deferred"); self.assertEqual(rep["components_applied"], [])
        self.assertEqual(rep["numerics"], "tf32"); self.assertEqual(rep["knobs"], self.table["knob_defaults"])
        self.assertEqual(rep["components_planned"], list(self.table["LEVERS"])); self.assertEqual(os.environ.get(modes.ENV_MODE), "exact")
        self.assertEqual(rep["gpu"]["name"], _stubs.PINNED_GPU); self.assertEqual(rep["gpu"]["cc"], "9.0")
        self.assertEqual(rep["borzoi_pytorch_version"], "0.5.1"); self.assertEqual(rep["package_version"], flashzoi_opt.__version__)
        from borzoi_pytorch.pytorch_borzoi_model import Borzoi
        self.assertTrue(hasattr(Borzoi.forward, "__wrapped__"))
        m = _stubs.small_borzoi()
        x = torch.zeros(1, 4, 524288); x[:, 0, :] = 1
        err = io.StringIO(); saved = sys.stderr; sys.stderr = err
        try:
            with self.assertRaises(flashzoi_opt.ActivationError) as cm:
                m(x)
        finally:
            sys.stderr = saved
        self.assertIn("device cpu", str(cm.exception)); self.assertIn("NOT ACTIVE: model instance on device cpu", err.getvalue())
        self.assertEqual(flashzoi_opt.status()["applied"], "deferred")
        self.assertIs(flashzoi_opt.enable("exact")["active"], True)                       # idempotent
        r2 = flashzoi_opt.enable("off")
        self.assertFalse(r2["active"]); self.assertIn("already active as mode=exact", r2["reason"])
        with self.assertRaises(flashzoi_opt.ActivationError):
            flashzoi_opt.enable("off", strict=True)

    def test_instances_before_enable_are_allowed_and_noted(self):
        m = _stubs.small_borzoi()
        rep = flashzoi_opt.enable("exact")
        self.assertTrue(rep["active"]); self.assertGreaterEqual(rep["instances_at_enable"], 1)
        self.assertTrue(any("existed at activation" in n for n in rep["notes"]))
        del m

    def test_foreign_runner_other_knobs_refuses(self):
        m = _stubs.small_borzoi()
        m._flashzoi_kit_runner = types.SimpleNamespace(numerics_knob="ieee")                 # a runner constructed under another numerics knob than the mode's
        rep = flashzoi_opt.enable("exact")
        self.assertFalse(rep["active"]); self.assertIn("kit runner already exists", rep["reason"]); self.assertIn("'ieee'", rep["reason"])

    def test_foreign_runner_same_knobs_is_adopted(self):
        m = _stubs.small_borzoi()
        m._flashzoi_kit_runner = types.SimpleNamespace(numerics_knob="tf32", components=frozenset(self.table["LEVERS"]), apply_s={}, device_name="x")   # the mode's own knob value: adopted
        err = io.StringIO(); saved = sys.stderr; sys.stderr = err
        try:
            rep = flashzoi_opt.enable("exact")
        finally:
            sys.stderr = saved
        self.assertTrue(rep["active"]); self.assertEqual(rep["applied"], "attached"); self.assertEqual(rep["models"][0]["trigger"], "adopted")
        self.assertIn("[flashzoi-opt] ACTIVE mode=exact variant=- gpu=", err.getvalue())


class TestAttachThroughStubKit(unittest.TestCase):
    """The attach path end to end on CPU: a stub kit (the kit's tables) whose KitRunner installs the instance-level forward; the
    model's device is stubbed as cuda. The ACTIVE line prints once at the first attach (after the kit's own line), every forward after
    the attach is the kit's instance forward, settle() reads the kit's evidence, remove() restores."""

    @classmethod
    def setUpClass(cls):
        real = _stubs.require_kit()
        _stubs.require_upstream()
        cls.table = modes.kit_table(real)
        _stubs.PINNED_GPU = cls.table["PINS"]["device_names"][0]

    def setUp(self):
        _stubs.reset_activation(); _stubs.forget_kit_modules()
        self.tree = _stubs.Tree().enter()
        self.stub_kit = _stubs.write_stub_kit(self.tree.root, self.table, _stubs.PINNED_GPU)
        self.undo = _stubs.stub_box()
        self._dev = stack._model_device
        stack._model_device = lambda model: _Dev("cuda")

    def tearDown(self):
        stack._model_device = self._dev
        self.undo(); self.tree.exit(); _stubs.reset_activation(); _stubs.forget_kit_modules()

    def test_attach_at_first_forward(self):
        import torch
        rep = flashzoi_opt.enable("exact")
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["kit_root"], self.stub_kit)
        kit = sys.modules[modes.KIT_MODULE]
        m1, m2 = _stubs.small_borzoi(), _stubs.small_borzoi()
        x = torch.zeros(1, 4, 8)
        err = io.StringIO(); saved = sys.stderr; sys.stderr = err
        try:
            y1 = m1(x); y1b = m1(x); y2 = m2(x)
        finally:
            sys.stderr = saved
        self.assertEqual(y1[0], "kit"); self.assertEqual(y2[0], "kit")
        self.assertEqual(len(kit.CALLS), 2, "one KitRunner per instance, at its first forward")
        self.assertEqual(kit.CALLS[0], {"model": id(m1), "numerics": "tf32"})
        self.assertEqual(err.getvalue().count("[flashzoi-opt] ACTIVE mode=exact"), 1)
        self.assertIn("components=" + ",".join(self.table["LEVERS"]), err.getvalue())
        self.assertIn("forward", m1.__dict__)                                                  # the kit's instance-level forward shadows the hook
        self.assertEqual(m1._flashzoi_kit_runner.forwards, 2)
        st = flashzoi_opt.status()
        self.assertEqual(st["applied"], "attached"); self.assertEqual(len(st["models"]), 2)
        self.assertEqual(st["components_applied"], sorted(self.table["LEVERS"]))
        self.assertEqual(st["models"][0]["knobs"], {"numerics": "tf32"})
        self.assertEqual(st["numerics"], "tf32"); self.assertEqual(st["class_label"], "exact"); self.assertTrue(st["route_class"])
        from borzoi_pytorch.pytorch_borzoi_model import Borzoi
        self.assertEqual(Borzoi.forward(m1, x)[0], "kit")                                    # an explicit class-level call lands in the hook and dispatches to the kit
        settled = stack.settle()
        self.assertFalse(settled["partial"]); self.assertEqual(settled["components_fallback"], [])
        self.assertEqual(settled["models"][0]["counts"], {"predict_calls": 3})
        out = stack.remove()
        self.assertEqual(out["runner0"], {"forward_restored": True}); self.assertNotIn("forward", m1.__dict__)
        self.assertFalse(flashzoi_opt.status()["active"]); self.assertFalse(hasattr(Borzoi.forward, "__wrapped__"))

    def test_exact_is_the_kits_defaults(self):
        """mode exact: the apply line receives the model and nothing else (the kit's defaults, numerics='tf32'); the ACTIVE line says mode=exact; the report reads the class back."""
        import torch
        rep = flashzoi_opt.enable("exact")
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["numerics"], "tf32")
        kit = sys.modules[modes.KIT_MODULE]
        m = _stubs.small_borzoi()
        err = io.StringIO(); saved = sys.stderr; sys.stderr = err
        try:
            y = m(torch.zeros(1, 4, 8))
        finally:
            sys.stderr = saved
        self.assertEqual(y[0], "kit")
        self.assertEqual(kit.CALLS, [{"model": id(m), "numerics": "tf32"}])
        self.assertIn("[flashzoi-opt] ACTIVE mode=exact variant=- gpu=", err.getvalue())
        st = flashzoi_opt.status()
        self.assertEqual(st["components_applied"], sorted(self.table["LEVERS"]))
        self.assertEqual(st["numerics"], "tf32"); self.assertEqual(st["class_label"], "exact"); self.assertEqual(st["models"][0]["route_class"], kit.CALLS and m._flashzoi_kit_runner.route_class)
        self.assertEqual(flashzoi_opt.enable("off")["reason"][:27], "already active as mode=exact"[:27])

    def test_apply_to_eager_and_partial(self):
        rep = flashzoi_opt.enable("exact")
        self.assertTrue(rep["active"], rep.get("reason"))
        m = _stubs.small_borzoi()
        err = io.StringIO(); saved = sys.stderr; sys.stderr = err
        try:
            rec = stack.apply_to(m, trigger="pred"); rec2 = stack.apply_to(m, trigger="pred")
        finally:
            sys.stderr = saved
        self.assertEqual(rec["trigger"], "pred"); self.assertEqual(rec2["model_index"], rec["model_index"])
        self.assertEqual(rec["describe"]["arm"], self.table["ARM"]); self.assertEqual(rec["numerics"], "tf32")
        r = m._flashzoi_kit_runner
        r.effective_flags = lambda: {lv: (lv != "graph") for lv in self.table["LEVERS"]}
        settled = stack.settle()
        self.assertTrue(settled["partial"]); self.assertEqual(settled["components_fallback"], ["graph"]); self.assertNotIn("graph", settled["components_applied"])
        self.assertIn("(partial: graph)", report.activation_line(settled))

    def test_kit_table_disagreement_refuses(self):
        init = os.path.join(self.stub_kit, modes.KIT_INIT_RELPATH)
        src = open(init).read().replace("from ._wrap import", "LEVERS = LEVERS[:-1]\nfrom ._wrap import")     # the live module drifts from its literals
        open(init, "w").write(src)
        err = io.StringIO(); saved = sys.stderr; sys.stderr = err
        try:
            rep = flashzoi_opt.enable("exact")
        finally:
            sys.stderr = saved
        self.assertFalse(rep["active"]); self.assertIn("kit table disagrees with its file", rep["reason"]); self.assertIn("NOT ACTIVE", err.getvalue())


if __name__ == "__main__":
    unittest.main()
