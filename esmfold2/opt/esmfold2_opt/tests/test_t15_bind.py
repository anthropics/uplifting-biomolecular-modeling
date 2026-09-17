"""t15 / t15msa / xtr serve through ONE provider call (ef2_pair_v2.serve) BY TIER WORD; the module holds no kernel; a stock resolution or a
Refusal keeps the module's statement by name."""
import ast, os, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "driver", "ef2_pair_v2.py"))).read()


class TestT15Bind(unittest.TestCase):
    def test_words_env_and_route(self):
        tree = ast.parse(SRC)
        consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                  if getattr(t, "id", None) in ("WORD_ENV", "MODE_ENV", "TIER_OF_MODE")}
        self.assertEqual(consts, {"WORD_ENV": "EF2_PAIR_WORD", "MODE_ENV": "ESMFOLD2_OPT", "TIER_OF_MODE": {"fast": "fast", "big": "big", "exact": "exact"}})
        serve = ast.get_source_segment(SRC, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "serve"))
        self.assertLess(serve.index("TR.select(bound_word()"), serve.index("TR.transition("))
        for token in ("sel.row in TR.STOCK_ROWS", "except TR.Refusal", "_name_once(", "return None", '("esmpair", "esmfused") if residual else ("pair", "swiglu")'):
            self.assertIn(token, serve)
        for fn, call in (("_transition_forward_v2", 'serve(self, x, residual=True, kind="t15")'), ("_pair_transition_residual_v2", 'serve(pt, pair, residual=True, kind="t15msa")'),
                         ("_pair_transition_forward_xtr", 'serve(self, x, residual=False, kind="xtr", x_ln=n)')):
            body = ast.get_source_segment(SRC, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn))
            self.assertIn(call, body)
        self.assertNotIn("triton.jit", SRC)                                                   # no kernel in the kit module

    def test_core_carries_the_family(self):
        from opt_core.kernels import transition as TR
        self.assertIn("esmpair_c256_n4+esmfused", TR.CELL_WORDS)
        self.assertTrue({"fast", "big", "exact"} <= set(TR.TIER_WORDS))


if __name__ == "__main__":
    unittest.main()
