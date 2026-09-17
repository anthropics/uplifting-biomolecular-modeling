"""The LayerNorm adapter (opt/forward/chai1_exactln/serve.py) binds the shared core's provider BY TIER WORD: class contracts on a CPU box.

For every kit mode's word (exact -> 'exact', fast -> 'fast', big -> 'big'), on both cards (cc 9.0 / 8.0) at the kit's stack word, every
LayerNorm class the trunk runs (pair c256 / MSA c64 / OPM-out c512 in the bf16 kind, TriMul's and the confidence head's fp32 c256, the single
track's c384 in both kinds) at every crop of the ladder decides through the provider WITHOUT a refusal or an uncovered cell, and the exact word
never names a tolerance-class row (an exact-class row where the provider vouches it on the stack, else the statement BY NAME).  Never asserts
the core's cell winner by name (the cells are the core's; the contract is the class)."""
import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "chai1_exactln", "serve.py"))
STACKS = {"9.0": "H100:torch2.13.0+cu130/3.7.1/cueq0.11.1", "8.0": "A100:torch2.13.0+cu130/3.7.1/cueq0.11.1"}   # the kit image's provider stack words (img_chai_cu130)
CROPS = (256, 384, 512, 768, 1024, 1536, 2048)
CLASSES = [("bf16", 256, lambda n: n * n), ("bf16", 64, lambda n: 16384 * n), ("bf16", 64, lambda n: 4096 * n), ("bf16", 512, lambda n: n * n), ("bf16", 384, lambda n: n),
           ("f32", 256, lambda n: n * n), ("f32", 384, lambda n: n)]


def _load():
    spec = importlib.util.spec_from_file_location("chai1_exactln_serve_undertest", SERVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _X:                                   # a duck tensor for _decide: dtype word + nothing the provider's alignment probe can read (-> aligned)
    def __init__(self, kind):
        self.dtype = "bf16" if kind == "bf16" else "fp32"


class TestTierWordBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.sv = _load()
            cls.LN = cls.sv.package()
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"opt_core.kernels.ln not importable here: {e}")

    def _arm(self, mode, cc):
        sv, LN = self.sv, self.LN
        sv.reset()
        sv._S.update(installed=True, mode=mode, word=sv.TIER_WORDS[mode], stack=STACKS[cc], cc=cc, has_triton=True, has_nvrtc=True)
        return sv, LN

    def test_words_are_the_tiers_and_big_never_shares_fast(self):
        self.assertEqual(self.sv.TIER_WORDS, {"exact": "exact", "fast": "fast", "big": "big"})
        self.assertEqual(self.sv.PROVIDER, "opt_core.kernels.ln")
        for w in self.sv.TIER_WORDS.values():
            self.assertIn(w, self.LN.TIER_WORDS)

    def test_every_class_decides_through_the_provider_without_refusal_or_uncovered_cell(self):
        for cc in STACKS:
            for mode in ("exact", "fast", "big"):
                sv, LN = self._arm(mode, cc)
                for kind, C, rf in CLASSES:
                    for n in CROPS:
                        rows = rf(n)
                        arm, sel, fb, exact_class = sv._decide(kind, _X(kind), C, None, None, rows, True)
                        tag = (cc, mode, kind, C, n, rows, arm, fb)
                        self.assertFalse(fb and fb.startswith("refused"), tag)              # a tier word never refuses
                        self.assertIsNotNone(sel, tag)
                        self.assertTrue(sel.size_measured, tag)                             # a measured cell at or above the call's rows: no UNCOVERED_CELL at ladder shapes
                        if arm is None:
                            self.assertTrue(fb.startswith("statement:"), tag)              # the statement BY NAME (a stock row is the cell's winner)
                            self.assertIn(sel.row, LN.STOCK_ROWS, tag)
                        else:
                            self.assertIn(sel.row, LN.ROW_NAMES, tag); self.assertNotIn(sel.row, LN.STOCK_ROWS, tag)
                        if mode == "exact":                                                 # exact = an exact-class row (bit-compared at run time) or the statement by name; never tolerance-class
                            self.assertTrue(sel.row in LN.EXACT_ROWS or sel.row in LN.STOCK_ROWS, tag)
                            self.assertEqual(exact_class, sel.row in LN.EXACT_ROWS, tag)

    def test_exact_word_on_an_unvouched_stack_names_the_statement_never_a_replica(self):
        sv, LN = self._arm("exact", "9.0")
        sv._S["stack"] = "H100:torch9.9.9+cu999/9.9.9/nocueq"                                 # a stack no cell vouches the replica on
        arm, sel, fb, exact_class = sv._decide("bf16", _X("bf16"), 256, None, None, 512 * 512, True)
        self.assertIsNone(arm); self.assertTrue(fb.startswith("statement:")); self.assertIn(sel.row, LN.STOCK_ROWS)

    def test_without_the_replicas_bindings_the_exact_word_names_the_statement(self):
        sv, LN = self._arm("exact", "9.0")
        sv._S["has_nvrtc"] = False
        arm, sel, fb, _ = sv._decide("f32", _X("f32"), 256, None, None, 512 * 512, True)
        self.assertIsNone(arm); self.assertTrue(fb.startswith("statement:"))

    def test_census_words_and_the_gate(self):
        sv, LN = self._arm("fast", "9.0")
        self.assertEqual(sv.gate(), (False, "no_calls"))                                   # installed but the hook never saw a call: refused at exit
        sv._S["classes"][("bf16", 256, True, True, 4, "c")] = (None, None, "statement:aten_autocast", False)
        sv._cnt(sv._S["fallback"], "statement:aten_autocast"); sv._cnt(sv._S["statement"], "aten_autocast")   # what ln() counts for a class the provider names the statement for
        self.assertEqual(sv.gate(), (True, ""))                                             # `statement:<row>` is an expected, named word
        sv._cnt(sv._S["fallback"], "dtype:float16")
        ok, why = sv.gate(); self.assertFalse(ok); self.assertIn("unexpected_fallback=dtype:float16", why)
        line = sv.line("[chai1-opt]")
        for tok in ("LEVER exactln pairtrack word=fast provider=opt_core.kernels.ln", "served=0", "statement=aten_autocast:1", "gate=REFUSED:"):
            self.assertIn(tok, line)
        sv.reset(); self.assertFalse(sv._S["installed"])


if __name__ == "__main__":
    unittest.main()
