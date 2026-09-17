"""P5 memlevers' capacity-gate word `fit` (CHANGES.md 0.3.12): grammar, canonical spec, refusal of a bare `fit`, the gate arithmetic under a stated pool, the default
setting's describe() unchanged. CPU jax is enough (the lever file imports jax / equinox / einops at its top)."""
import importlib.util
import os
import unittest

from . import _stubs
from mosaic_opt import registry

P5_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "memlevers.py")



@unittest.skipIf(any(importlib.util.find_spec(m) is None for m in ("jax", "equinox", "einops")), "memlevers imports jax / equinox / einops (CPU jax is enough)")
class MemleversFit(unittest.TestCase):
    """P5's capacity gate word `fit` (CHANGES.md 0.3.12): grammar, the canonical spec, the refusal of a bare `fit`, the gate arithmetic under a stated pool,
    and the default setting's describe() unchanged (no `fit` token unless the word is in force)."""
    def setUp(self):
        spec = importlib.util.spec_from_file_location("memlevers_under_test", os.path.join(_stubs.KIT, "mosaic_fast", "memlevers.py"))
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)

    def test_grammar(self):
        m = self.m
        self.assertEqual(m.parse("pf8+sub+fit"), {"triatt_chunk": None, "pf_group": 8, "sub_remat": True, "fit": True})
        self.assertEqual(m.spec_of(m.parse("pf8+sub+fit")), "pf8+sub+fit"); self.assertEqual(m.spec_of(m.parse(None)), "pf8+sub")
        self.assertNotIn("fit", m.parse(None)); self.assertEqual(m.parse("tri64+pf8"), {"triatt_chunk": 64, "pf_group": 8, "sub_remat": False})   # every other spec keeps its shape
        with self.assertRaises(ValueError) as cm:
            m.parse("fit")
        self.assertIn("memlevers_refused", str(cm.exception))
        self.assertNotIn("fit", m.describe())                                          # default: the LEVER line of the big row is unchanged

    def test_gate_arithmetic(self):
        m = self.m
        m.MEM["fit"] = True
        m._pool_limit_gib = lambda: 59.4                                                # H100-80GB default pool (0.75 of the card)
        self.assertTrue(m._fits(500)); self.assertTrue(m._fits(800)); self.assertFalse(m._fits(1100))
        m._pool_limit_gib = lambda: 29.6                                                # a 40 GB card's default pool
        self.assertTrue(m._fits(500)); self.assertFalse(m._fits(800))
        m._pool_limit_gib = lambda: None                                                # no memory statistics: never aside
        self.assertFalse(m._fits(200))
        m.MEM.pop("fit")
        m._pool_limit_gib = lambda: 59.4
        self.assertFalse(m._fits(200))




class BigRowPin(unittest.TestCase):
    """0.3.24: the big row pins P5 at `pf8+sub` (the memory schedule at every input size; `pf8+sub+fit`, the pin from 0.3.15 to 0.3.23, stays a spec P5's grammar accepts); the fast row carries no P5."""
    def test_pin(self):
        from mosaic_opt import modes
        self.assertEqual(modes.specs_of("big").get("P5"), "pf8+sub")
        self.assertNotIn("P5", modes.specs_of("fast"))
        self.assertIn("P5", modes.KIT_MODES["big"]["levers"]); self.assertNotIn("P5", modes.KIT_MODES["fast"]["levers"])
        src = open(P5_FILE).read()
        self.assertIn('elif part == "fit": out["fit"] = True', src); self.assertIn('FIT_ASIDE = "aside:fits"', src)


if __name__ == "__main__":
    unittest.main()
