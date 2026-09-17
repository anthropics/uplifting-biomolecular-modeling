"""Lever ln_core (protenix_opt 0.3.51): the model's standalone LayerNorm calls bound to the shared core's LayerNorm provider by TIER WORD.

Class contracts (CPU, no device): the tier word per mode comes from modes.PACKAGE_POST (fast -> fast, big -> big, exact -> none: the
library op fast_layernorm by name); the word RESOLVES THROUGH THE PROVIDER (opt_core.kernels.ln.select) for this kit's bf16o cells on both
cards -- never a row-name pin in the kit; the binding unit executes only CARRIED rows and leaves STOCK rows / uncovered widths on the
module's statement by name; ablation removes the switch."""
import os
import sys
import unittest

from protenix_opt import modes, registry, stack, report, ablation

_SRC = os.path.join(stack.opt_home(), "forward", "flashpairformer", "src")                              # the unit lives in the FlashPairformer tree's src (on the worker's path via env.sh PYTHONPATH)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class LnCoreContract(unittest.TestCase):
    def test_lever_row_and_modes(self):
        lv = registry.LEVERS["ln_core"]
        self.assertEqual(lv.env_keys, ("PTX_LN_TIER",)); self.assertEqual(lv.tier, registry.TOLERANCE); self.assertTrue(lv.extra)
        self.assertIn("ln_core", modes.MODES["fast"]); self.assertIn("ln_core", modes.MODES["big"]); self.assertIn("ln_core", modes.big_levers())
        self.assertNotIn("ln_core", modes.MODES["exact"])                  # exact = the library op by name: nothing bound (no provider row is bitwise to fast_layernorm)
        self.assertEqual(modes.PACKAGE_POST["fast"]["PTX_LN_TIER"], "fast"); self.assertEqual(modes.PACKAGE_POST["big"]["PTX_LN_TIER"], "big")
        self.assertNotIn("PTX_LN_TIER", modes.PACKAGE_POST["exact"])
        self.assertIn("ln_core", stack.MARKERS); self.assertEqual(report.IMPL["ln_core"], ("opt_core.kernels.ln", "core"))
        self.assertEqual(ablation.switches("ln_core"), ("PTX_LN_TIER",))

    def test_unit_words_and_byname_default(self):
        import protenix_ptx_ln_core as U
        self.assertEqual(U.WORDS, ("fast", "big")); self.assertEqual(U.ENV, "PTX_LN_TIER")
        r = U.report()
        self.assertFalse(r["installed"]); self.assertEqual(r["calls"], 0); self.assertEqual(r["rows"], {})
        old = os.environ.pop("PTX_LN_TIER", None)
        try:
            os.environ["PTX_LN_TIER"] = "exact"
            with self.assertRaises(U.Unavailable):                          # exact is not a word this unit binds
                U.install()
        finally:
            os.environ.pop("PTX_LN_TIER", None)
            if old is not None:
                os.environ["PTX_LN_TIER"] = old

    def test_tier_words_resolve_through_the_provider(self):
        """Both cards, this kit's stack word, the bf16o pair cells at the ladder sizes: every tier word the kit exports resolves to a provider Selection
        (a carried row or the named stock row) -- the kit pins no row name; the exact word names the stock arm fast_layernorm (library op by name)."""
        from opt_core.kernels import ln as LN
        stacks = {"9.0": "H100:torch2.13.0+cu130/3.7.1/cueq0.11.1", "8.0": "A100:torch2.13.0+cu130/3.7.1/cueq0.11.1"}
        carried, stock = set(LN.ROW_NAMES), set(LN.STOCK_ROWS)
        executed = 0
        for cc, stk in stacks.items():
            for n in (400, 800, 1200):
                rows = n * n
                cw = LN.cell_for_rows(rows, n, 256)
                self.assertIsNotNone(cw, (cc, n))
                for word in ("fast", "big"):
                    sel = LN._select(cc, "bfloat16", cw, n, word=word, widen=True, out="bf16", stack=stk, C=256, rows=rows, has_triton=True, has_nvrtc=True)
                    self.assertIn(sel.row, carried | stock, (cc, n, word, sel.row))
                    executed += sel.row in carried
                ex = LN._select(cc, "bfloat16", cw, n, word="exact", widen=True, out="bf16", stack=stk, C=256, rows=rows, has_triton=True, has_nvrtc=True)
                self.assertIn(ex.row, {"fast_layernorm"} | set(LN.EXACT_ROWS), (cc, n))   # exact: the cells' stock arm by name, or (opt_core >= 0.5.125.0) the bitwise-vouched exactln row on a proven cc — the kit's exact mode still binds nothing here (ln_core is fast | big)
        self.assertGreater(executed, 0, "the provider executes a carried row for at least one (card, size, word) of this kit's pair LayerNorm")

    def test_decide_classifies_rows(self):
        import protenix_ptx_ln_core as U

        class Sel:
            def __init__(self, row, variant=None): self.row, self.variant = row, variant

        class Face:
            STOCK_ROWS = ("aten", "aten_autocast", "fast_layernorm")
            class Refusal(Exception):
                kind = "named_row"
            def __init__(self, answer): self.answer = answer
            def cell_for_rows(self, rows, n, C): return None if C == 16 else "pair_c%d" % C
            def select(self, cc, dt, cw, n, **kw):
                if isinstance(self.answer, Exception): raise self.answer
                return self.answer
        U._STATE["word"] = "fast"
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch absent")
        x = None
        self.assertEqual(U._decide(Face(Sel("fastln", "lp")), (9, 0), x, 256, 160000, True, True)[0], "core")
        self.assertEqual(U._decide(Face(Sel("fast_layernorm")), (9, 0), x, 256, 160000, True, True), ("byname", "stock:fast_layernorm"))
        self.assertEqual(U._decide(Face(Sel("fastln", "lp")), (9, 0), x, 16, 100, True, True), ("byname", "uncovered"))
        self.assertEqual(U._decide(Face(Face.Refusal("x")), (9, 0), x, 256, 100, True, True)[0], "byname")
        U._STATE["word"] = None


if __name__ == "__main__":
    unittest.main()
