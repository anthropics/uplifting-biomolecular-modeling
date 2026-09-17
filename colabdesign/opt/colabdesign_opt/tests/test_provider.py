"""kernels/provider.py — the binding of the kernel levers to the shared core's JAX-family provider: every lever's word is either a ROW word naming
a row this kit's adapter binds or a TIER word of the provider narrowed to those rows; the row module is located in the core WITHOUT importing it
(impl= words on GPU-less hosts); a row word resolves to its own row in any cell (measured or not) on both cards; a tier word resolves, in every
cell, to a row this adapter binds or to the stock statement BY NAME (`xla`, the per-call step-aside `cell_xla`) — the class contract, never the
provider's current winner by name; and the printed facts are blank-free tokens.  Pure Python (the provider's selection imports no jax)."""
import re
import unittest

from colabdesign_opt.kernels import provider as P

TIER_WORDS = tuple(P.provider().TIER_WORDS)                                                 # exact | fast | big — the shared provider's tier vocabulary


class TestBindings(unittest.TestCase):
    def setUp(self):
        P.reset_for_tests()

    def test_words_name_bindable_rows_and_modules_are_located(self):
        core = P.provider()
        for lever, b in P.BINDINGS.items():
            if b.word in core.TIER_WORDS:                                                  # a tier binding: the tier this kit's kernel levers belong to
                self.assertEqual(b.word, P.TIER_WORD, lever)
            else:                                                                          # a row binding: the row word (bitwise the kit's own kernels)
                self.assertIn(b.word, b.rows, lever)
            self.assertTrue(b.rows, lever)
            for row, path in b.rows.items():
                self.assertTrue(path.startswith("opt_core.kernels."), path)
                self.assertIsNotNone(P.row_file(lever, row), f"{lever}: {path} not found in the core")
            self.assertRegex(P.kernel_word(lever), r"^[a-z_]+@[0-9a-f]{8}$")

    def test_bound_word_serves_a_bindable_row_or_the_stock_statement_by_name_in_every_cell_on_both_cards(self):
        core = P.provider()
        for lever, b in P.BINDINGS.items():
            fam = core.family(b.op, **b.main)
            tier = b.word in core.TIER_WORDS
            for cc in ("9.0", "8.0"):
                b.reset(); b.stack = ("0.6", cc)                                            # one card per process: the per-process cell cache is card-blind
                for n in (195, 500, 800, 3000):
                    arm, cfg = P._cell(lever, n, "bf16", None, 1, **b.main)
                    row = arm.split("@")[0].split(":")[0]
                    if tier:                                                               # the class contract: one of this adapter's rows, or the stock method by its name
                        self.assertIn(row, set(b.rows) | {"xla"}, f"{lever} {cc} {n}: {arm}")
                        self.assertIsInstance(cfg, dict, f"{lever} {cc} {n}")
                        if row == "xla":                                                   # resolve() hands such a call to the stock method; its census word:
                            self.assertEqual(P.fallback_word(lever, n, "bf16", **b.main), P.CELL_FALLBACK + "xla", f"{lever} {cc} {n}")
                    else:                                                                  # the default rule: a row word serves exactly that row, no launch setting
                        self.assertEqual(row, b.word, f"{lever} {cc} {n}: {arm}")
                        self.assertEqual(cfg, {}, f"{lever}: a row word carries no launch setting")
            f = P.facts(lever)
            for k in ("provider", "word", "row", "tier"):
                self.assertIn(k, f)
            for k, v in f.items():
                self.assertNotRegex(str(v), r"\s", f"{lever}.{k}={v!r}")
            self.assertTrue(f["provider"].startswith(P.PROVIDER + "@"))

    def test_tier_bound_levers_admit_on_both_measured_cards_of_this_line(self):
        """the model's main cell (MAIN_TOKENS, the binding's main family, differentiated) under a tier word names a row this adapter binds on this
        kit's pinned jax line for cc 9.0 and 8.0, or the stock statement by name where that is the measured winner -- admission installs the lever there (a card / line the provider has not measured names the stock
        statement: the lever steps aside by name at install, tested through the refusal path below)."""
        core = P.provider()
        for lever, b in P.BINDINGS.items():
            if b.word not in core.TIER_WORDS:
                continue
            fam = core.family(b.op, **b.main)
            for cc in ("9.0", "8.0"):
                sel = core._select("0.6", cc, b.dtype, b.op, fam, P.MAIN_TOKENS, b.direction, word=b.word, prefer=tuple(b.rows))
                self.assertIn(sel.row, set(b.rows) | {"xla"}, f"{lever} {cc}: {sel}")          # a row this adapter binds, or the stock statement BY NAME where the table measured it faster on that card (the class contract; which one is the provider's measurement)

    def test_unbindable_word_is_refused_by_name(self):
        b = P.BINDINGS["trimul"]
        keep = b.word
        try:
            b.word = "no_such_row_word"
            b.stack = ("0.6", "9.0")
            arm, _ = P._cell("trimul", 500, "bf16", None, 1, **b.main)
            self.assertTrue(arm.startswith("refused_"), arm)                              # the census names the refusal; resolve() hands the call to the stock method
            mod, cfg = P.resolve("trimul", 500, "bf16", **b.main)
            self.assertIsNone(mod)
            self.assertTrue(P.fallback_word("trimul", 500, "bf16", **b.main).startswith(P.CELL_FALLBACK + "refused_"))
        finally:
            b.word = keep
            P.reset_for_tests()


if __name__ == "__main__":
    unittest.main()
