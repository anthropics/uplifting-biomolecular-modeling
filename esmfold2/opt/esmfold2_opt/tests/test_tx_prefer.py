"""`tx` w4.19: under the BIG word the eager pair-TriMul calls of classes N <= 512 on class 9.x ask the provider with the row preference
tx_sm90a first (opt_core's cell winner there, row `native`, is host-bound per call in eager use); fast / exact / a row word, a capturing stream,
N >= 513 and class 8.0 never carry a preference -- their composition is unchanged.  CPU tests: the rule itself (per word x N x class x capture),
the LEVER word, and the binding's resolution against the REAL provider face (pure selection: a preference the cell does not serve never lands
the class on the stock row -- the word's own answer stands)."""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DRV = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "driver"))
if DRV not in sys.path:
    sys.path.insert(0, DRV)

SIZES = (20, 128, 200, 256, 257, 400, 448, 512, 513, 600, 800, 1200)


def _ef2_w4(tc):
    try:
        import ef2_w4
    except ModuleNotFoundError as e:                                   # the esm / transformers forks + triton: the kit stack (the GPU image runs these tests there)
        tc.skipTest(f"ef2_w4 needs the kit stack ({e.name})")
    return ef2_w4


class TestTxEagerPrefer(unittest.TestCase):
    def test_constants(self):
        W = _ef2_w4(self)
        self.assertEqual(W.TX_EAGER_PREFER, ("tx_sm90a",)); self.assertEqual(W.TX_EAGER_PREFER_MAX_N, 512)
        self.assertEqual(W.TX_EAGER_PREFER_WORDS, ("big",)); self.assertEqual(W.TX_EAGER_PREFER_CC, ("9.",))

    def test_rule_per_word_size_class_and_capture(self):
        """The composition grid: prefer present ONLY for (big, class 9.x, N <= 512, not capturing)."""
        W = _ef2_w4(self)
        saved = dict(W._TX)
        try:
            for word in ("fast", "big", "exact", "tx_sm90a", "native"):
                for cc in ("9.0", "8.0", "10.0", None):
                    W._TX.update(word=word, cc=cc)
                    self.assertEqual(W.prefer_word(), "tx_sm90a:eager:N<=512" if (word == "big" and cc == "9.0") else None, (word, cc))
                    for N in SIZES:
                        for capturing in (False, True):
                            want = ("tx_sm90a",) if (word == "big" and cc == "9.0" and N <= 512 and not capturing) else None
                            self.assertEqual(W._tx_prefer(N, capturing), want, (word, cc, N, capturing))
        finally:
            W._TX.clear(); W._TX.update(saved)

    def test_resolution_against_the_real_provider_face(self):
        """ef2_w4._tx_select on the shared core's pure selection (no device): under big + prefer every class N <= 512 on the pinned 9.0 stacks resolves
        to a KERNEL row the binding serves with the preference kept only when the cell served the preferred row; a class never lands on a stock row
        because of the preference; fast resolves exactly as the word alone does (no preference asked)."""
        import torch
        W = _ef2_w4(self)
        from opt_core.kernels import trimul as TRI
        z = torch.zeros(1, 8, 8, 256, dtype=torch.bfloat16)
        saved = dict(W._TX)
        try:
            for stack in ("H100:2.13.0+cu130/3.7.1/nocueq", "H100:2.13.0+cu130/3.7.1/cueq0.11.1", "H100:2.11.0+cu128/3.6.0/cueq0.10.0"):
                for word in ("big", "fast"):
                    W._TX.clear(); W._TX.update(saved)
                    W._TX.update(on=True, bound=True, mod=TRI, word=word, cc="9.0", stack=stack, abi=None, has_cueq=False, classes={}, rows={}, refused={}, printed=set(), stock_classes=0)
                    for N in SIZES:
                        for d in ("outgoing", "incoming"):
                            pref = W._tx_prefer(N, False)
                            self.assertEqual(pref, ("tx_sm90a",) if (word == "big" and N <= 512) else None)
                            rec = W._tx_select(z, N, d, 256, pref)
                            alone = TRI.select("9.0", "bf16", 256, 256, N, d, word=word, stack=stack, has_cueq=False)
                            self.assertEqual(rec["prefer_asked"], pref, (stack, word, N, d))
                            if pref is None:                                           # no preference: the word's own answer, verbatim
                                self.assertEqual(rec["cell_row"], alone.row, (stack, word, N, d)); self.assertIsNone(rec["prefer"])
                            elif rec["prefer"] is not None:                            # the cell served the preferred row: it is the class's row, the serving calls carry the preference
                                self.assertEqual(rec["cell_row"], "tx_sm90a", (stack, word, N, d)); self.assertEqual(rec["prefer"], ("tx_sm90a",))
                            else:                                                      # the cell does not serve it on this stack: the word's own answer, never a stock row by preference
                                self.assertEqual(rec["cell_row"], alone.row, (stack, word, N, d))
                            if rec["cell_row"] in TRI.STOCK_ROWS:
                                self.assertIn(alone.row, TRI.STOCK_ROWS, (stack, word, N, d))   # a stock resolution is the word's, not the preference's
            # the pinned stacks (torch 2.13.0+cu130): the preferred row IS measured and admitted in every cell up to 512 tokens -> served under big
            for stack in ("H100:2.13.0+cu130/3.7.1/nocueq", "H100:2.13.0+cu130/3.7.1/cueq0.11.1"):
                for N in (128, 256, 257, 400, 512):
                    for d in ("outgoing", "incoming"):
                        sel = TRI.select("9.0", "bf16", 256, 256, N, d, word="big", prefer=("tx_sm90a",), stack=stack, has_cueq=False)
                        self.assertEqual(sel.row, "tx_sm90a", (stack, N, d, TRI.describe(sel)))
            with self.assertRaises(TRI.Refusal):                                       # class 8.0 never serves an sm_90a object, preference or not (the rule never asks there)
                TRI.select("8.0", "bf16", 256, 256, 400, "outgoing", word="tx_sm90a")
        finally:
            W._TX.clear(); W._TX.update(saved)

    def test_lever_line_carries_the_preference_under_big_only(self):
        W = _ef2_w4(self)
        from esmfold2_opt import report
        saved = dict(W._TX)
        try:
            sys.modules.setdefault("ef2_w4", W)
            for word, cc, want in (("big", "9.0", "tx_sm90a:eager:N<=512"), ("fast", "9.0", None), ("exact", "9.0", None), ("big", "8.0", None)):
                W._TX.update(on=True, bound=True, word=word, cc=cc, stack="H100:stub", abi="stub", classes={}, rows={}, refused={}, stock_classes=0)
                ev = report.lever_evidence("tx")
                self.assertEqual(ev.get("prefer"), want, (word, cc)); self.assertEqual(ev["word"], word)
                keys = list(ev)
                self.assertEqual(keys[:6], ["word", "row", "cell", "stack", "abi", "cc"]); self.assertEqual(keys[-2:], ["refused", "stock_classes"])   # format-stable: prefer sits between cc and refused
        finally:
            W._TX.clear(); W._TX.update(saved)


if __name__ == "__main__":
    unittest.main()
