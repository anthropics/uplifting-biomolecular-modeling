"""MSA_COL_CUDNN + PALLAS_MSA (msa_col_cudnn.py / msa_attn.py): the routing rule of the two levers that bind the bias-free / narrow-head attention
sites to the shared core's JAX-family provider by tier word, the class chain, the floor by name, the registry rows and LEVER tokens.  CPU only
(tests/_stubs.py; no call reaches a kernel here: the served body needs jax + haiku and is exercised on the GPU stack)."""
import unittest

from colabfold_opt.tests import _stubs
from colabfold_opt import modes, msa_attn, msa_col_cudnn, registry, report, stack


class TestRule(unittest.TestCase):
    def test_classify_names_the_provider_kind_of_the_two_call_classes(self):
        self.assertEqual(msa_attn.classify(32, 32, False), "msacol")                 # MSA column attention: bias-free, 8 x 32
        self.assertEqual(msa_attn.classify(8, 8, True), "extramsa_slab512")          # extra-MSA rows: pair-biased, 8 channels (< 16: padded inside the face)
        self.assertIsNone(msa_attn.classify(32, 32, True))                           # pair-biased 32-channel heads: AF_PALLAS_ATTN's / TRIATTN_XLA's, uncounted here
        self.assertIsNone(msa_attn.classify(16, 16, True))                           # template pair stack heads (16): not narrow
        self.assertIsNone(msa_attn.classify(32, 24, False))                          # unequal q / v widths: the wrapped class
        self.assertEqual((msa_attn.KIND_COLUMN, msa_attn.KIND_NARROW, msa_attn.NARROW_BELOW, msa_attn.COLUMN_ROW_ELSEWHERE), ("msacol", "extramsa_slab512", 16, "cudnn"))

    def test_key_mask_of_reads_stock_mask_bias_only(self):
        import numpy as np
        bias = (1e9 * (np.array([[1, 1, 0, 0]], np.float32) - 1.0))[:, None, None, :]
        self.assertEqual(msa_attn.key_mask_of(bias).tolist(), [[True, True, False, False]])
        self.assertIsNone(msa_attn.key_mask_of(np.zeros((1, 8, 4, 4), np.float32)))   # a full [b,h,q,k] bias is not a key mask: mask_form
        self.assertIsNone(msa_attn.key_mask_of(None))

    def test_tier_word_is_the_mode_word(self):
        self.assertEqual([msa_attn.tier_word({modes.ENV: m}) for m in ("fast", "big", "exact", "off", "")], ["fast", "big", "exact", "fast", "fast"])

    def test_column_prefer_leaves_cudnn_to_the_column_lever(self):
        rows = msa_attn.column_prefer()
        self.assertNotIn("cudnn", rows); self.assertIn("xla", rows); self.assertIn("pallas_attn", rows)

    def test_no_kit_floor_or_row_pin_remains(self):
        for mod in (msa_attn, msa_col_cudnn):
            for name in ("MIN_KEYS", "BELOW_KEYS_RULE", "OP", "sdpa", "probe", "ROW", "KERNEL"):
                self.assertFalse(hasattr(mod, name), (mod.__name__, name))
        self.assertIn("no_cell_family", registry.STEP_ASIDE_RULES)


class TestLever(unittest.TestCase):
    def setUp(self):
        self.mods, self.saved = _stubs.install()
        self.gates = _stubs.gates_pass(stack)
        stack.reset_for_tests()

    def tearDown(self):
        stack.reset_for_tests()
        _stubs.gates_restore(stack, self.gates)
        _stubs.remove(self.saved)

    def test_enable_rebinds_the_class_bound_now_and_disable_restores_it(self):
        m = self.mods["alphafold.model.modules"]
        stock = m.Attention
        msa_attn.enable(m)
        self.assertTrue(msa_attn.marker_present(m)); self.assertTrue(issubclass(m.Attention, stock))
        msa_col_cudnn.enable(m)
        self.assertTrue(msa_col_cudnn.marker_present(m)); self.assertTrue(getattr(m.Attention, msa_attn.MARKER, False))   # over PALLAS_MSA's class
        self.assertEqual((msa_col_cudnn._STATE["word"], msa_col_cudnn._STATE["cc"]), ("fast", "9.0"))
        msa_col_cudnn.disable(); msa_attn.disable()
        self.assertIs(m.Attention, stock)

    def test_floor_refusals_by_name(self):
        m = self.mods["alphafold.model.modules"]
        msa_col_cudnn.CC = "7.5"
        with self.assertRaises(msa_col_cudnn.Refusal) as cm:
            msa_col_cudnn.enable(m)
        self.assertEqual(cm.exception.kind, "cc_below_8_0"); self.assertTrue(str(cm.exception).startswith("MSA_COL_CUDNN: cc_below_8_0"))
        self.assertFalse(msa_col_cudnn.marker_present(m))

    def test_registry_rows_lever_lines_and_tables(self):
        for name, strategy in ((modes.MSA_LEVER, "F5.flash_attn_dense"), (modes.COL_LEVER, "F5.sdpa_cudnn")):
            lv = registry.LEVERS[name]
            self.assertEqual((lv.impl, lv.origin, lv.strategy), ("opt_core.kernels.pallas:attention", "core", strategy))
        msa_col_cudnn.reset_for_tests(); msa_attn.reset_for_tests()
        self.assertEqual(report.lever_line(modes.COL_LEVER, "on", **{k: v for k, v in msa_col_cudnn._STATE.items() if k != "enabled"}),
                         "[colabfold-opt] LEVER name=MSA_COL_CUDNN state=on impl=opt_core.kernels.pallas:attention origin=core strategy=F5.sdpa_cudnn "
                         "word=none served=none cells=none calls=0 fallbacks=0 fallback_by=none shapes=none precision=none cc=none")
        self.assertEqual(report.lever_line(modes.MSA_LEVER, "on", **{k: v for k, v in msa_attn._STATE.items() if k != "enabled"}),
                         "[colabfold-opt] LEVER name=PALLAS_MSA state=on impl=opt_core.kernels.pallas:attention origin=core strategy=F5.flash_attn_dense "
                         "word=none served=none cells=none calls=0 fallbacks=0 fallback_by=none shapes=none cc=none")
        self.assertEqual(modes.SUPERSEDES[modes.COL_LEVER], (modes.MSA_LEVER,)); self.assertNotIn(modes.COL_LEVER, modes.ONE_DEVICE_PAIR_LEVERS)   # 0.2.23: kept on at --n_gpu P > 1


if __name__ == "__main__":
    unittest.main()
