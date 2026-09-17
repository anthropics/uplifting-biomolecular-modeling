"""The model-instance count behind the late-activation rule: a constructor wrap on the upstream model class, registered at activation —
on the class when its module is already imported (existing instances found once by a gc scan), at the module's import otherwise (a
meta-path finder) — and the gc scan only when the class cannot be wrapped. The activation report notes that fallback."""
import gc
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest

from esmfold2_opt import stack
from esmfold2_opt.tests import _stubs

FRESH = {"state": None, "finder": None}


def _fake_module(name: str, cls_name: str = stack.MODEL_CLASS):
    mod = types.ModuleType(name)

    class Model:
        def __init__(self, config, *inputs, **kwargs):
            self.config = config
    Model.__name__ = Model.__qualname__ = cls_name
    setattr(mod, cls_name, Model)
    return mod, Model


class _Frozen(type):
    def __setattr__(cls, name, value):
        raise TypeError("frozen class")


class TestCounter(unittest.TestCase):
    def setUp(self):
        self._saved = (stack.MODEL_MODULE, dict(stack._INSTANCES), list(sys.meta_path))
        stack._INSTANCES.clear(); stack._INSTANCES.update(FRESH)
        self.name = f"_ef2_counter_test_{id(self)}"
        stack.MODEL_MODULE = self.name

    def tearDown(self):
        stack.MODEL_MODULE = self._saved[0]
        stack._INSTANCES.clear(); stack._INSTANCES.update(self._saved[1])
        sys.meta_path[:] = self._saved[2]
        sys.modules.pop(self.name, None)

    def test_not_imported_means_no_instances(self):
        self.assertEqual(stack.instance_check(), {"n": 0, "method": "none", "built": 0})
        self.assertEqual(stack.model_instances(), 0)

    def test_counted_path_on_an_imported_class(self):
        mod, Model = _fake_module(self.name)
        sys.modules[self.name] = mod
        before = Model("pre")                                              # exists before the counter: found once by the gc scan
        sig = inspect.signature(Model)
        chk = stack.register_instance_counter()
        self.assertEqual(chk, {"n": 1, "method": "counted", "built": 0})
        self.assertEqual(stack._INSTANCES["state"]["gc_seeded"], 1)
        self.assertEqual(inspect.signature(Model), sig, "the class signature is kept")
        a = Model("a"); b = Model("b", 1, x=2)
        self.assertEqual(stack.instance_check(), {"n": 3, "method": "counted", "built": 2})
        self.assertEqual((a.config, b.config), ("a", "b"))
        del a, before
        self.assertEqual(stack.instance_check()["n"], 1, "no gc pass needed: a released instance leaves the count at once")
        stack.register_instance_counter()                                  # idempotent: one wrap per class
        Model("c")
        self.assertEqual(stack.instance_check()["built"], 3)
        self.assertEqual(stack.model_instances(), 1)
        del b

    def test_finder_path_wraps_the_class_at_its_import(self):
        tmp = tempfile.mkdtemp(prefix="ef2_counter_")
        try:
            with open(os.path.join(tmp, self.name + ".py"), "w", encoding="utf-8") as fh:
                fh.write(textwrap.dedent(f'''
                    class {stack.MODEL_CLASS}:
                        def __init__(self, config):
                            self.config = config
                '''))
            sys.path.insert(0, tmp)
            try:
                chk = stack.register_instance_counter()
                self.assertEqual(chk, {"n": 0, "method": "none", "built": 0})
                self.assertIsInstance(sys.meta_path[0], stack._ModelClassFinder)
                self.assertIs(stack._INSTANCES["finder"], sys.meta_path[0])
                stack.register_instance_counter()                          # idempotent: one finder
                self.assertEqual(sum(isinstance(f, stack._ModelClassFinder) for f in sys.meta_path), 1)
                mod = importlib.import_module(self.name)                   # the import happens on the program's own schedule
                self.assertFalse(any(isinstance(f, stack._ModelClassFinder) for f in sys.meta_path), "the finder fires once and removes itself")
                self.assertIs(stack._INSTANCES["state"]["cls"], getattr(mod, stack.MODEL_CLASS))
                self.assertEqual(stack.instance_check(), {"n": 0, "method": "counted", "built": 0})
                m = getattr(mod, stack.MODEL_CLASS)("cfg")
                self.assertEqual(stack.instance_check(), {"n": 1, "method": "counted", "built": 1})
                self.assertEqual(m.config, "cfg")
                del m
                self.assertEqual(stack.model_instances(), 0)
            finally:
                sys.path.remove(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_gc_scan_when_the_class_cannot_be_wrapped(self):
        mod = types.ModuleType(self.name)
        Model = _Frozen(stack.MODEL_CLASS, (), {"__init__": lambda self, config=None: None})
        setattr(mod, stack.MODEL_CLASS, Model)
        sys.modules[self.name] = mod
        chk = stack.register_instance_counter()
        self.assertEqual(chk, {"n": 0, "method": "gc", "built": 0})
        self.assertFalse(stack._INSTANCES["state"]["wrapped"])
        m = Model()
        self.assertEqual(stack.instance_check(), {"n": 1, "method": "gc", "built": 0})
        del m
        gc.collect()
        self.assertEqual(stack.model_instances(), 0)


class TestActivationUsesTheCounter(unittest.TestCase):
    """Through enable(), in a subprocess with the stub upstream: the refusal names the method, and the gc fallback is a note."""

    @classmethod
    def setUpClass(cls):
        cls.real_kit = _stubs.require_kit()
        cls.pins = _stubs.require_pins()
        cls.tmp = tempfile.mkdtemp(prefix="ef2_counter_act_")
        cls.site = _stubs.write_stub_upstream(cls.tmp, cls.pins)
        cls.kit = _stubs.write_stub_kit(cls.tmp, cls.real_kit)
        cls.tree = _stubs.write_stub_tree(cls.tmp, cls.pins)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, script, **extra):
        env = _stubs.env_for_stub(self.site, self.kit, self.tree, **extra)
        r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        return json.loads(r.stdout.strip().splitlines()[-1]), r.stderr

    def test_counter_registered_before_the_model_module_is_imported(self):
        out, _ = self._run(textwrap.dedent('''
            import json, sys
            import esmfold2_opt
            from esmfold2_opt import stack
            r = esmfold2_opt.enable("fast", "fast")                          # before upstream is imported: the finder waits for the module
            out = {"active": r["active"], "notes": r["notes"], "finder_armed": any(isinstance(f, stack._ModelClassFinder) for f in sys.meta_path)}
            import transformers.models.esmfold2.modeling_esmfold2 as M
            out["finder_after_import"] = any(isinstance(f, stack._ModelClassFinder) for f in sys.meta_path)
            out["check_after_import"] = stack.instance_check()
            m = M.ESMFold2Model.from_pretrained("biohub/ESMFold2")
            out["check_with_model"] = stack.instance_check()
            del m
            out["check_after_del"] = stack.instance_check()
            print(json.dumps(out, default=str))
        '''))
        self.assertTrue(out["active"]); self.assertTrue(out["finder_armed"]); self.assertFalse(out["finder_after_import"])
        self.assertEqual(out["check_after_import"], {"n": 0, "method": "counted", "built": 0})
        self.assertEqual(out["check_with_model"], {"n": 1, "method": "counted", "built": 1})
        self.assertEqual(out["check_after_del"], {"n": 0, "method": "counted", "built": 1})
        self.assertFalse([n for n in out["notes"] if "gc scan" in n])

    def test_refusal_names_the_method_and_the_gc_fallback_is_noted(self):
        out, _ = self._run(textwrap.dedent('''
            import json, sys
            import esmfold2_opt
            from esmfold2_opt import stack
            import transformers.models.esmfold2.modeling_esmfold2 as M
            class Frozen(type):
                def __setattr__(cls, name, value):
                    raise TypeError("frozen class")
            M.ESMFold2Model = Frozen("ESMFold2Model", (), {"__init__": lambda self, config=None: None})   # a class the wrap cannot touch
            m = M.ESMFold2Model()
            r = esmfold2_opt.enable("fast", "fast")
            out = {"refused": {"active": r["active"], "reason": r.get("reason")}, "check": stack.instance_check()}
            del m
            import gc; gc.collect()
            r = esmfold2_opt.enable("fast", "fast")
            out["active"] = r["active"]; out["notes"] = r["notes"]
            print(json.dumps(out, default=str))
        '''))
        self.assertFalse(out["refused"]["active"]); self.assertIn("1 ESMFold2Model instance(s) already exist in this process (gc)", out["refused"]["reason"])
        self.assertEqual(out["check"], {"n": 1, "method": "gc", "built": 0})
        self.assertTrue(out["active"])
        self.assertIn("ESMFold2Model instances are counted by a gc scan: the class could not be wrapped", out["notes"])


if __name__ == "__main__":
    unittest.main()
