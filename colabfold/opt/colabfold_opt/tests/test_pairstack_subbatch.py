"""SUBBATCH (subbatch.py: the value decided through opt_core.jax_design.subbatch_policy and written into every model configuration) on stubs —
no jax kernels, no GPU; and the retired word FPF_TRIMUL (no lever of any mode since 0.2.6; its module pairstack.py left the tree at 0.2.17)."""
import types
import unittest

from colabfold_opt import modes, registry, stack, subbatch

GIB = 2 ** 30


class TestSubbatchDecision(unittest.TestCase):
    def test_policy_branches_are_named(self):
        fits = subbatch.decide(1000, 80 * GIB)
        self.assertEqual((fits["value"], fits["source"], fits["tokens"], fits["stock_value"]), (128, "auto:fits", 1000, 4))
        self.assertGreater(fits["est_gib"], 0); self.assertEqual(fits["device_gib"], 80.0)
        big = subbatch.decide(4000, 80 * GIB)                                # the chunk-128 program's estimate exceeds the device: stock's 4, named
        self.assertEqual((big["value"], big["source"]), (4, "auto:exceeds"))
        nodev = subbatch.decide(1000, None)                                  # no device size: stock's 4, named
        self.assertEqual((nodev["value"], nodev["source"]), (4, "auto:no_device"))
        notok = subbatch.decide(None, 80 * GIB)                              # no token count (the Python route without queries): the chunk, named
        self.assertEqual((notok["value"], notok["source"]), (128, "requested:no_tokens"))
        for P in (2, 4, 8):                                                  # 0.2.21: under the row-sharded pair stack (big --n_gpu P > 1) the chunk, named auto:rowpair, at any size
            tp = subbatch.decide(4000, 80 * GIB, n_gpu=P)
            self.assertEqual((tp["value"], tp["source"], tp["tokens"], tp["est_gib"], tp["device_gib"]), (128, "auto:rowpair", 4000, None, 80.0))
        self.assertEqual((subbatch.decide(None, None, n_gpu=2)["value"], subbatch.decide(None, None, n_gpu=2)["source"]), (128, "auto:rowpair"))
        self.assertEqual(subbatch.decide(4000, 80 * GIB, n_gpu=1), big)             # n_gpu=1 is the one-GPU decision, field for field

    def test_peak_fit_passes_through_the_measured_points(self):
        for tokens, gib in subbatch.PEAK_POINTS:                              # the fit + 10 % margin reads above every measured point and within 15 % of it
            est = subbatch.PEAK(tokens) / GIB
            self.assertGreaterEqual(est, gib); self.assertLess(est, gib * 1.15)

    def test_enable_wraps_model_config_with_marker_and_counts(self):
        cfgmod = types.ModuleType("cfg_stub")
        cfgmod.model_config = lambda name: types.SimpleNamespace(model=types.SimpleNamespace(global_config=types.SimpleNamespace(subbatch_size=4)))
        subbatch.reset_for_tests()
        try:
            dec = subbatch.enable(tokens=1400, device_bytes=80 * GIB, module=cfgmod)
            self.assertEqual(dec["value"], 128)
            self.assertTrue(subbatch.marker_present(cfgmod)); self.assertTrue(getattr(cfgmod.model_config, subbatch.MARKER))
            cfg = cfgmod.model_config("model_1_multimer_v3")
            self.assertEqual(cfg.model.global_config.subbatch_size, 128)
            self.assertEqual((subbatch._STATE["enabled"], subbatch._STATE["calls"], subbatch._STATE["source"]), (True, 1, "auto:fits"))
            subbatch.enable(tokens=5000, device_bytes=80 * GIB, module=cfgmod)      # re-decides and re-wraps the STOCK function (no double wrap)
            self.assertEqual(cfgmod.model_config("m").model.global_config.subbatch_size, 4)
            subbatch.disable()
            self.assertFalse(subbatch.marker_present(cfgmod)); self.assertEqual(cfgmod.model_config("m").model.global_config.subbatch_size, 4)
        finally:
            subbatch.reset_for_tests()

    def test_registry_row(self):
        lv = registry.LEVERS[modes.SUBBATCH_LEVER]
        self.assertEqual((lv.impl, lv.origin, lv.strategy, lv.kit_file), ("opt_core.jax_design.subbatch_policy", "core", "F7.jax_subbatch", "colabfold_opt/subbatch.py"))
        self.assertEqual(subbatch.STRATEGY, lv.strategy); self.assertEqual(subbatch.STOCK_VALUE, 4); self.assertEqual(subbatch.CHUNK, 128)


class TestFpfTrimulRetired(unittest.TestCase):
    def test_the_word_is_no_lever_of_the_modes_or_the_registry(self):
        self.assertNotIn("FPF_TRIMUL", registry.LEVERS); self.assertNotIn("FPF_TRIMUL", registry.MARKERS)
        for mode in modes.MODES:
            self.assertNotIn("FPF_TRIMUL", modes.TABLE[mode][0])


if __name__ == "__main__":
    unittest.main()
