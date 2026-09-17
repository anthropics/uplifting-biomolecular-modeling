"""AF_PALLAS_ATTN_ALL=1 (the kernel widened to MSA-column / template attention) is refused by name at activation: the gate's reason, the
dry-run report, and no refusal when the switch is absent or 0. No jax, no GPU."""
import unittest

from colabfold_opt import registry, stack


class TestSwitchRefusal(unittest.TestCase):
    def test_widening_refused_by_name(self):
        r = stack.gate_switches({"AF_PALLAS_ATTN_ALL": "1"})
        self.assertTrue(r.startswith("AF_PALLAS_ATTN_ALL=1 refused: MSA-column and template attention through the kernel are outside the measured band"), r)
        self.assertIn("short-key probe", r); self.assertIn("n_templates <= 4", r)

    def test_absent_or_zero_passes(self):
        for env in ({}, {"AF_PALLAS_ATTN_ALL": "0"}, {"AF_PALLAS_ATTN_ALL": ""}, {"AF_PALLAS_ATTN": "1"}):
            self.assertIsNone(stack.gate_switches(env), env)

    def test_table_is_registry_data(self):
        self.assertEqual(list(registry.REFUSED_SWITCHES), ["AF_PALLAS_ATTN_ALL"])
        self.assertIn("AF_PALLAS_ATTN_ALL", registry.NOT_WIRED)                      # still the kit's own switch, in no mode; refused when set

    def test_check_reports_the_refusal(self):
        import os
        os.environ["AF_PALLAS_ATTN_ALL"] = "1"
        try:
            rep = stack.check("fast", print_line=False)
        finally:
            del os.environ["AF_PALLAS_ATTN_ALL"]
        self.assertFalse(rep["active"]); self.assertIn("AF_PALLAS_ATTN_ALL=1 refused:", rep["reason"])


if __name__ == "__main__":
    unittest.main()
