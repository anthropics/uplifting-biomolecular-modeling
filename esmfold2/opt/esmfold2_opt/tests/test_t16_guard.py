"""t16's data-dependent guard (stack.guards_after_run): the CuTe transition kernel's own call counters are the evidence of application —
lever on and no Transition / PairTransition call served => 'unreached' (the lever leaves levers_applied for partial, exit != 0 by the
all-or-refuse verdict); calls outside the kernel's scope that took the previous statements => a 'gated' note (exit unchanged)."""
import collections
import sys
import types
import unittest

from esmfold2_opt import stack


class T16Guard(unittest.TestCase):
    def _with_module(self, on, **stats):
        m = types.ModuleType("ef2_transition_cute")
        m.STATS = collections.Counter(stats)
        m.levers_on = lambda: {"t16": on}
        old = sys.modules.get("ef2_transition_cute")
        sys.modules["ef2_transition_cute"] = m
        self.addCleanup(lambda: (sys.modules.__setitem__("ef2_transition_cute", old) if old is not None else sys.modules.pop("ef2_transition_cute", None)))
        return stack.guards_after_run()

    def test_unreached_when_no_call_served(self):
        g = self._with_module(True, transition_calls=0, pair_transition_calls=0)
        self.assertEqual(g["t16"][0], "unreached")

    def test_gated_note_on_fallthrough(self):
        g = self._with_module(True, transition_calls=960, pair_transition_calls=160, fallthrough_transition=2)
        self.assertEqual(g["t16"][0], "gated"); self.assertIn("fallthrough_transition=2", g["t16"][1]); self.assertIn("1120 served", g["t16"][1])

    def test_silent_when_served_or_off(self):
        self.assertNotIn("t16", self._with_module(True, transition_calls=960, pair_transition_calls=0))
        self.assertNotIn("t16", self._with_module(False, transition_calls=0, pair_transition_calls=0))

    def test_settle_marks_unreached_partial(self):
        self._with_module(True, transition_calls=0, pair_transition_calls=0)
        rep = stack.settle_guards({"levers_applied": ["t9", "t16"], "partial": []})
        self.assertNotIn("t16", rep["levers_applied"]); self.assertIn("t16", rep["partial"]); self.assertIn("t16", rep["levers_fallback"])


if __name__ == "__main__":
    unittest.main()
