"""chai1_eager.transition_core — the CLASS contract of the tier-word binding (CPU; the provider's table read through its own face):
the binding names no provider row; every mode asks exactly its own tier word; at each of the trunk's four transition cells, on both cards and every
crop the kit runs, the tier word either resolves a row of the provider (an exact-tier row is measured bitwise and never slower than the stock arm)
or refuses BY NAME with a stock fallback (the statement serves, counted) — never a silent choice."""
import importlib.util
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                       # <kit>
CORE = os.path.abspath(os.path.join(KIT, "..", "common", "opt_core"))
TC_PATH = os.path.join(KIT, "opt", "forward", "eager_trunk", "chai1_eager", "transition_core.py")
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from opt_core.kernels import transition as T  # noqa: E402

CLASSES = (((256, 512), "pair"), ((256, 1024), "pair"), ((64, 256), "rows"), ((384, 768), "single"))   # (c, hidden), row family — the trunk's four Transition classes
CROPS = (256, 384, 512, 768, 1024, 1536, 2048)
STACKS = {"9.0": "H100:torch2.13.0+cu130/3.7.1", "8.0": "A100:torch2.13.0+cu130/3.7.1"}


def _load_tc():
    spec = importlib.util.spec_from_file_location("chai1_eager_transition_core_under_test", TC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TierWordBinding(unittest.TestCase):
    def test_modes_ask_their_own_tier_word_and_nothing_else(self):
        tc = _load_tc()
        self.assertEqual(tc.TIER, {"exact": "exact", "fast": "fast", "big": "big"})
        for mode, word in tc.TIER.items():
            self.assertIn(word, T.TIER_WORDS)
            self.assertEqual(tc.tier_for(mode), word)
        with self.assertRaises(KeyError):
            tc.tier_for("faithful")

    def test_the_binding_names_no_provider_row(self):
        src = open(TC_PATH, encoding="utf-8").read()
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        body = code.split('"""', 2)[2] if code.count('"""') >= 2 else code                   # past the module docstring
        for row in T.ROW_NAMES:
            if row in T.STOCK_ROWS:
                continue
            self.assertNotRegex(body, r"['\"]%s['\"@:]" % re.escape(row), row)             # no row word literal in the binding's code

    def test_declared_fallback_words_match_the_pair_track(self):
        tc = _load_tc()
        self.assertEqual(tc.EXPECTED_FALLBACKS, ("stock", "no_cell", "refused", "class_differs"))
        from chai1_opt import pairtrack
        self.assertEqual(pairtrack.EXPECTED_FALLBACKS["transition"], tc.EXPECTED_FALLBACKS)

    def test_family_words_are_read_off_the_operand(self):
        tc = _load_tc()

        class _X:
            def __init__(self, nd): self._nd = nd
            def dim(self): return self._nd
        self.assertEqual(tc.family_of(_X(4), 256), "pair")
        self.assertEqual(tc.family_of(_X(3), 384), "single")
        self.assertEqual(tc.family_of(_X(4), 64), "rows")
        for (c, h), fam in CLASSES:
            self.assertIn(fam, ("pair", "rows", "single"))

    def test_every_class_card_crop_resolves_a_row_or_refuses_by_name(self):
        """CLASS contract: whatever the shared core's cells say today, the tier word at the trunk's cells is a served row (in the provider's row set; exact rows
        bitwise and not slower than the stock arm where measured) or a Refusal by name whose fallback is a stock row."""
        for cc, st in STACKS.items():
            for (c, h), fam in CLASSES:
                for tier in ("exact", "fast", "big"):
                    for n in CROPS:
                        rows = n * n if fam != "single" else n
                        try:
                            s = T.select(tier, c=c, hidden=h, n_tokens=n, family=fam, cc=cc, stack=st, ln_given=(tier == "exact"), rows_count=rows)
                        except T.Refusal as e:
                            self.assertTrue(e.kind and e.fallback in T.STOCK_ROWS + T.ROW_NAMES, (cc, c, h, tier, n, str(e)))
                            continue
                        self.assertIn(s.row, T.ROW_NAMES, (cc, c, h, tier, n, s))
                        self.assertEqual(s.tier, tier)
                        if tier == "exact" and s.row not in T.STOCK_ROWS and s.stack_measured:
                            self.assertEqual(s.cls, "bitwise", s)
                            self.assertGreaterEqual(s.x_stock, 1.0, s)


if __name__ == "__main__":
    unittest.main()
