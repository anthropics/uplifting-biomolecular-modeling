"""The mode table is the kit's own bytes: what modes.kit_table reads from the kit files equals what the kit module exposes live (its
LEVERS, LEVER_CLASS, PINS, ARM, and KitRunner's knob defaults and accepted values); `exact` resolves to exactly the kit's LEVERS at the
kit's default knobs (`KitRunner(model)`, numerics='tf32': the kit's one numerics route, a value the kit's own check
accepts); `off` carries nothing; the registry has one entry per kit component, carries the knob under the mode and names the not-wired knob."""
import importlib
import inspect
import os
import sys
import unittest

from flashzoi_opt import modes, registry
from flashzoi_opt.tests import _stubs


class TestModesMatchKit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kit = _stubs.require_kit()
        cls.table = modes.kit_table(cls.kit)

    def test_table_is_read_not_typed(self):
        t = self.table
        self.assertIsInstance(t["LEVERS"], tuple); self.assertGreater(len(t["LEVERS"]), 0)
        self.assertEqual(set(t["LEVERS"]), set(t["LEVER_CLASS"]) & set(t["LEVERS"]))
        for lv in t["LEVERS"]:
            self.assertIn(lv, t["LEVER_CLASS"], f"kit component {lv} has no class statement in the kit's LEVER_CLASS")
        self.assertIn("device_names", t["PINS"]); self.assertTrue(t["PINS"]["device_names"])
        self.assertEqual(t["knob_defaults"], {"numerics": "tf32"})                  # the kit's own default (the apply line's knob) — the contract of `exact`
        self.assertIn("tf32", t["knob_choices"]["numerics"]); self.assertEqual(set(t["knob_choices"]), {"numerics"})   # the kit's own accepted values (read from its check)
        self.assertTrue(t["ARM"])

    def test_live_module_equals_file(self):
        _stubs.require_upstream()
        if self.kit not in sys.path:
            sys.path.insert(0, self.kit)
        kit = importlib.import_module(modes.KIT_MODULE)
        self.assertEqual(tuple(kit.LEVERS), self.table["LEVERS"])
        self.assertEqual(dict(kit.LEVER_CLASS), dict(self.table["LEVER_CLASS"]))
        self.assertEqual(list(kit.PINS["device_names"]), list(self.table["PINS"]["device_names"]))
        self.assertEqual(kit.ARM, self.table["ARM"])
        sig = inspect.signature(kit.KitRunner.__init__).parameters
        self.assertEqual({k: sig[k].default for k in modes.KIT_KNOBS}, self.table["knob_defaults"])
        self.assertTrue(callable(kit.remove))

    def test_resolve_exact_is_the_kit_composition(self):
        r = modes.resolve("exact", self.kit)
        self.assertEqual(r.components, self.table["LEVERS"])
        self.assertEqual(r.knobs, self.table["knob_defaults"]); self.assertEqual(r.constructor_kwargs, {})
        self.assertEqual(r.apply_line, "kit.KitRunner(model)")
        self.assertEqual(r.kit_arm, self.table["ARM"])
        self.assertEqual(r.component_line, ",".join(self.table["LEVERS"]))

    def test_exact_is_the_kits_one_numerics_route(self):
        r = modes.resolve("exact", self.kit)
        self.assertEqual(r.constructor_kwargs, {}); self.assertEqual(r.apply_line, "kit.KitRunner(model)")
        self.assertEqual(r.knobs, dict(self.table["knob_defaults"])); self.assertEqual(r.knob_line, "numerics=tf32")
        self.assertEqual(self.table["knob_defaults"]["numerics"], "tf32"); self.assertEqual(self.table["knob_choices"]["numerics"], ("tf32",))   # ONE numerics route in the kit's own check
        saved = modes.MODE_ARGS["exact"]
        modes.MODE_ARGS["exact"] = {"numerics": "bf16"}
        try:
            with self.assertRaises(ValueError):
                modes.resolve("exact", self.kit)                                                 # a value the kit's own check does not accept
        finally:
            modes.MODE_ARGS["exact"] = saved

    def test_off_and_unknown(self):
        self.assertEqual(modes.resolve("off", self.kit).components, ())
        with self.assertRaises(ValueError):
            modes.check_mode("turbo")
        self.assertEqual(modes.DEFAULT_MODE, "exact")    # the package default: a run that names no mode runs exact
        self.assertEqual(modes.MODES, ("off", "exact"))
        self.assertEqual(modes.KIT_MODES, ("exact",))

    def test_default_mode_from_env(self):
        saved = os.environ.pop(modes.ENV_MODE, None)
        try:
            self.assertEqual(modes.check_mode(None), "exact")                      # the package default
            os.environ[modes.ENV_MODE] = "off"; self.assertEqual(modes.check_mode(None), "off")
            os.environ[modes.ENV_MODE] = "exact"; self.assertEqual(modes.check_mode(None), "exact")
            os.environ[modes.ENV_MODE] = "turbo"
            with self.assertRaises(ValueError):
                modes.check_mode(None)
        finally:
            os.environ.pop(modes.ENV_MODE, None)
            if saved is not None:
                os.environ[modes.ENV_MODE] = saved

    def test_registry_from_kit_table(self):
        lv = registry.components(table=self.table)
        self.assertEqual(tuple(lv), self.table["LEVERS"])
        for name, L in lv.items():
            self.assertEqual(L.kit_class, self.table["LEVER_CLASS"][name])
            self.assertEqual(L.switch, registry.APPLY_LINE)
        self.assertEqual(registry.MODE_KNOBS["exact"]["numerics"], "tf32"); self.assertEqual(tuple(registry.MODE_KNOBS), modes.KIT_MODES)
        self.assertEqual({m: modes.MODE_ARGS[m].get("numerics", self.table["knob_defaults"]["numerics"]) for m in modes.KIT_MODES}, {m: registry.MODE_KNOBS[m]["numerics"] for m in modes.KIT_MODES})
        self.assertEqual(registry.device_names(table=self.table), tuple(self.table["PINS"]["device_names"]))

    def test_stack_and_jit_keys(self):
        self.assertEqual(modes.stack_key("9.0", "3.1.0"), "9.0|3.1")
        self.assertEqual(modes.jit_cache_key(version="2.5.1+cu124", cc="90"), "torch2.5.1-cu124-sm90")
        self.assertEqual(modes.jit_cache_key(version="2.5.1", cuda="124", cc="90"), "torch2.5.1-cu124-sm90")
        self.assertEqual(modes.jit_cache_key(version="2.5.1", cuda="12.4", cc="9.0"), "torch2.5.1-cu124-sm90")   # the pins' dotted forms name the same key


if __name__ == "__main__":
    unittest.main()
