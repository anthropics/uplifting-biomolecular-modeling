"""The kit's call-surface gate (CPU): route decision, passthrough counting, one shape's stores at a time, the hook rule."""
import collections
import unittest

from evo2_opt.kit import gate as GT
from evo2_opt.kit import hooks as HK


class _Mod:
    def __init__(self, hooks=False, w1=False):
        self._forward_hooks = collections.OrderedDict({1: object()}) if hooks else collections.OrderedDict()
        self._forward_pre_hooks = collections.OrderedDict()
        self._v40_w1 = w1


class _Model:
    def __init__(self, mods):
        self.mods = mods

    def named_modules(self):
        return list(self.mods.items())


class TestGate(unittest.TestCase):
    def test_decide_route(self):
        self.assertEqual(GT.decide_route(1, 8192, None, None), ("kit", None))
        self.assertEqual(GT.decide_route(7, 12345, None, None), ("kit", None))            # every (batch, length)
        self.assertEqual(GT.decide_route(1, 1, {"mha": 0}, None)[0], "stock")
        self.assertEqual(GT.decide_route(1, 8, None, object())[0], "stock")

    def test_passthrough_counts_and_announces_once(self):
        ctr = collections.Counter(); g = GT.Gate(ctr); lines = []
        g.log = type("L", (), {"write": lambda self, s: lines.append(s), "flush": lambda self: None})()
        for _ in range(3):
            with g.passthrough("inference_params|generate"):
                self.assertEqual(g.mode, "stock")
            self.assertEqual(g.mode, "kit")
        self.assertEqual((ctr["gate_passthrough"], ctr["passthrough:inference_params"]), (3, 3))
        self.assertEqual(len([l for l in lines if l.strip()]), 1)

    def test_one_shape_resident_at_a_time(self):
        ctr = collections.Counter(); released = []
        sm = GT.ShapeManager({"a": lambda: released.append("a")}, ctr, model_stores={"m": lambda: released.append("m")})
        self.assertFalse(sm.ensure(1, 8192)); self.assertFalse(sm.ensure(1, 8192))       # same shape: nothing released
        self.assertTrue(sm.ensure(2, 8192)); self.assertEqual(released, ["a"])           # a new shape releases the shape-bound stores only
        sm.release(full=True); self.assertEqual(released, ["a", "a", "m"])

    def test_gated_wrapper_and_forward_gate(self):
        ctr = collections.Counter(); g = GT.Gate(ctr); g.log = type("L", (), {"write": lambda self, s: None, "flush": lambda self: None})()
        w = GT.gated(g, lambda x: ("kit", x), lambda x: ("stock", x), "f")
        self.assertEqual(w(1), ("kit", 1))
        with g.passthrough("hooks|test"):
            self.assertEqual(w(1), ("stock", 1))
        sm = GT.ShapeManager({}, ctr)

        class X:
            shape = (2, 16)
        calls = []
        fwd = GT.forward_gate(lambda self, x, ip=None, pm=None: calls.append(g.mode) or "out", g, sm)
        self.assertEqual(fwd(object(), X()), "out"); self.assertEqual(fwd(object(), X(), {"mha": 1}), "out")
        self.assertEqual(calls, ["kit", "stock"]); self.assertEqual(sm.shape, None)        # the passthrough released and reset the shape

    def test_hook_rule(self):
        self.assertEqual(HK.decide(_Model({"": _Mod(), "blocks.0": _Mod()}), 1, 8, None, None), ("kit", None))
        r, why = HK.decide(_Model({"blocks.3": _Mod(hooks=True)}), 1, 8, None, None)
        self.assertEqual(r, "stock"); self.assertTrue(why.startswith("hooks|"))
        with self.assertRaises(HK.KitRefusedHooks):
            HK.decide(_Model({"blocks.3.projections": _Mod(hooks=True)}), 1, 8, None, None)
        late = _Model({"": _Mod(), "blocks.0": _Mod()})                                     # a hook registered after the first decision is seen by the next
        self.assertEqual(HK.decide(late, 1, 8, None, None), ("kit", None))
        late.mods["blocks.0"]._forward_hooks[7] = object()
        self.assertEqual(HK.decide(late, 1, 8, None, None)[0], "stock")
        del late.mods["blocks.0"]._forward_hooks[7]
        self.assertEqual(HK.decide(late, 1, 8, None, None), ("kit", None))


if __name__ == "__main__":
    unittest.main()
