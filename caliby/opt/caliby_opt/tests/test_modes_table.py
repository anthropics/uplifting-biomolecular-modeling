"""The mode table (modes.py): the rows equal the expected switch sets switch for switch, in the order exported
(a shell comment on a switch line never leaks into a row), the resolver's rules hold (the default mode, the three clean forms
of fast/single and the flag that selects them, no sub-selector of a row (a mode is all of its levers), the ensemble line without a clean lever; exact on single
refused by name, fast on ensemble32 = the ensemble row; default fast on both variants), the single rows under `fast` are the single route's rows switch for
switch (the levers the single route always shipped; the mode names the guarantee, tier 2), and the environment route's mode names are the table's."""
import unittest

from .. import _autoload, modes, registry

# the expected rows, switch for switch
ROW_X = {"CALIBY_FAST_SAMPLER": "2", "CALIBY_FAST_POTTS_PARAMS": "2", "CALIBY_X_SPARSE_EXACT": "2", "CALIBY_X_LCP": "1", "CALIBY_X_CLEAN": "0", "CALIBY_X_BG_CIF": "fork",
         "CALIBY_X_MULTISEQ": "1", "CALIBY_X_CIF_WORKERS": "8"}
ROW_XL = dict(ROW_X, CALIBY_X_CLEAN="loader")
ROW_XP = dict(ROW_X, CALIBY_X_CLEAN="proc")
ROW_E32 = {"CALIBY_FAST_SAMPLER": "2", "CALIBY_FAST_POTTS_PARAMS": "2", "CALIBY_X_SPARSE_EXACT": "2", "CALIBY_X_LCP": "1", "CALIBY_X_TIED_DET": "1",
           "CALIBY_X_ENS_WORKERS": "8", "CALIBY_X_PP_CACHE": "1", "CALIBY_X_BG_CIF": "fork", "CALIBY_X_PP_NOSYNC": "1", "CALIBY_X_PP_FASTPDB": "1",
           "CALIBY_X_MULTISEQ": "1", "CALIBY_X_CIF_WORKERS": "8"}


class TestModeTable(unittest.TestCase):
    def test_modes_and_defaults(self):
        self.assertEqual(modes.MODES, ("off", "exact", "fast"))
        self.assertEqual(modes.KIT_MODE_OF, {"single": "fast", "ensemble32": "exact"})   # each variant's one kit mode
        self.assertEqual(modes.TIER_OF, {"exact": 1, "fast": 2})
        self.assertEqual((modes.default_mode("single"), modes.default_mode("ensemble32"), modes.default_mode(None)), ("fast", "fast", "fast"))   # never exact by default
        self.assertEqual(modes.DEFAULT_MODE, "fast")
        self.assertEqual(modes.MODES_OF, {"single": ("off", "fast"), "ensemble32": ("off", "exact", "fast")})
        self.assertEqual(modes.VARIANTS, ("single", "ensemble32"))
        self.assertEqual(modes.DEFAULT_VARIANT, "single")
        self.assertEqual(modes.CLEAN_LEVER, "loader")
        self.assertFalse(hasattr(modes, "LCP_KERNELS") or hasattr(modes, "LCP_OFF"))   # no sub-selector of a row: a mode is all of its levers
        self.assertEqual(set(_autoload.MODES), set(modes.MODES))       # the environment route names the table's modes

    def test_rows(self):
        t = modes.table()
        self.assertEqual(t["single/serial"], ROW_X)
        self.assertEqual(t["single/loader"], ROW_XL)
        self.assertEqual(t["ensemble32"], ROW_E32)
        self.assertEqual(modes.line_switches("A CALIBY_X_CLEAN=loader CALIBY_X_BG_CIF=fork   # a comment: CALIBY_X_CLEAN=0"),
                         {"CALIBY_X_CLEAN": "loader", "CALIBY_X_BG_CIF": "fork"})
        self.assertEqual(list(t["single/serial"]), list(ROW_X))        # the order exported
        self.assertEqual(list(t["ensemble32"]), list(ROW_E32))

    def test_clean_lever_values_are_the_two_rows(self):
        for key, sw in modes.table().items():
            self.assertIn(sw.get("CALIBY_X_CLEAN"), ("0", "loader", None), key)   # the clean lever's only forms: serial (row X) and loader (row XL); the ensemble row does not export it
        self.assertEqual(modes.resolve("fast", "single", clean_workers=8).row.key, "single/loader")
        self.assertFalse(hasattr(modes, "CLEAN_LEVERS"))                # no lever selector outside the mode words: --clean_workers N>1 is the loader row

    def test_row_labels_and_shapes(self):
        self.assertEqual({key: (row.label, row.variant, row.clean_workers, row.ensemble) for key, row in modes.ROWS.items()},
                         {"single/serial": ("X", "single", 1, 0), "single/loader": ("XL", "single", 4, 0),
                          "ensemble32": ("ensemble command", "ensemble32", 1, 32)})

    def test_no_sub_selector(self):
        """resolve() takes the mode, the variant and the clean workers only: every kit row is exported whole (CALIBY_X_LCP=1 on all)."""
        with self.assertRaises(TypeError):
            modes.resolve("fast", "single", lcp_kernel=0)
        for r in (modes.resolve("fast", "single"), modes.resolve("fast", "single", clean_workers=4), modes.resolve("exact", "ensemble32")):
            self.assertEqual((r.env["CALIBY_X_LCP"], r.env, r.source), ("1", modes.row_switches(r.row), f"modes.py:{r.row.key}"))

    def test_resolver_rules(self):
        r = modes.resolve("off", "single")
        self.assertEqual((r.row, r.env, r.source), (None, {}, None))
        r = modes.resolve("fast", "single")
        self.assertEqual((r.row.key, r.env, r.clean_workers), ("single/serial", ROW_X, 1))
        self.assertEqual(r.source, "modes.py:single/serial")
        r = modes.resolve("fast", "single", clean_workers=4)
        self.assertEqual((r.row.key, r.env, r.label), ("single/loader", ROW_XL, "XL"))
        r = modes.resolve("exact", "ensemble32", clean_workers=4)     # the ensemble row carries no clean lever: N clean workers are upstream's own parallel clean, the row unchanged
        self.assertEqual((r.row.key, r.env, r.clean_workers), ("ensemble32", ROW_E32, 4))
        r = modes.resolve("exact", "ensemble32")
        self.assertEqual((r.row.key, r.env, r.source), ("ensemble32", ROW_E32, "modes.py:ensemble32"))
        self.assertEqual(r.line, " ".join(f"{k}={v}" for k, v in ROW_E32.items()))
        self.assertEqual(modes.resolve("fast", None).variant, "single")
        with self.assertRaises(ValueError) as cm:                     # exact on single: refused by name, with the pointer
            modes.resolve("exact", "single")
        self.assertEqual(str(cm.exception), modes.REFUSED[("exact", "single")])
        self.assertIn("exact refused: the single-sequence route differs from stock deterministically (energy last-bit class); use --mode fast on single", str(cm.exception))
        rf = modes.resolve("fast", "ensemble32")                       # fast on ensemble32: the row exact names, switch for switch (bit-identical satisfies fast)
        self.assertEqual((rf.row.key, rf.env, rf.source), ("ensemble32", modes.resolve("exact", "ensemble32").env, "modes.py:ensemble32"))
        self.assertEqual([modes.mode_refusal(m, v) for v, ms in modes.MODES_OF.items() for m in ms], [None] * 5)
        self.assertEqual(sorted(modes.REFUSED), [("exact", "single")])
        self.assertTrue(modes.mode_refusal("turbo", "single").startswith("unknown mode 'turbo'"))
        with self.assertRaises(ValueError):
            modes.resolve("exact", "e32")
        with self.assertRaises(ValueError):
            modes.resolve("fast", "single", clean_workers=0)

    def test_single_fast_rows(self):
        """`fast` on single exports exactly the single route's rows — row X and XL, whole: the mode names the guarantee (tier 2), not
        other levers."""
        self.assertEqual(modes.resolve("fast", "single").env, ROW_X)
        self.assertIs(modes.resolve("fast", "single").row, modes.ROWS["single/serial"])
        self.assertEqual(modes.resolve("fast", "single", clean_workers=4).env, ROW_XL)
        for key in ("single/serial", "single/loader"):
            self.assertEqual(modes.ROWS[key].variant, "single")
            self.assertEqual(modes.table()[key], modes.row_switches(modes.ROWS[key]))
        self.assertEqual(modes.resolve("exact", "ensemble32").env, ROW_E32)                  # the ensemble row unchanged under exact
        self.assertIs(modes.resolve("exact", "ensemble32").row, modes.ROWS["ensemble32"])

    def test_alternatives_named(self):
        r = modes.resolve("fast", "single")
        self.assertEqual(r.alternatives, ("clean lever",))
        r = modes.resolve("exact", "ensemble32")
        self.assertEqual(r.alternatives, ("ensemble workers",))
        for k in r.alternatives:
            self.assertIn(k, modes.ALTERNATIVES)
        self.assertEqual(modes.resolve("off", "single").alternatives, ())
        self.assertFalse(hasattr(modes, "OPEN_SLOTS"))

    def test_every_switch_is_registered(self):
        for key, sw in modes.table().items():
            for name in sw:
                self.assertIn(name, registry.LEVERS, f"{key}: {name}")
                lever = registry.LEVERS[name]
                self.assertIn(lever.cls, (registry.CLASS_FORWARD, registry.CLASS_DATAPATH))
        single = registry.for_variant("single")
        for name in ROW_XP:
            self.assertIn(name, single)
        for name in ("CALIBY_X_TIED_DET", "CALIBY_X_ENS_WORKERS", "CALIBY_X_PP_CACHE", "CALIBY_X_PP_NOSYNC", "CALIBY_X_PP_FASTPDB"):
            self.assertNotIn(name, single)


if __name__ == "__main__":
    unittest.main()
