"""t16 serves through the shared core's transition provider BY TIER WORD (the mode's own word: fast | big), holds no kernel, resolves the word
for the call class first (a stock resolution or a Refusal keeps the statement by name), and the tree's core resolves every tier word to SOME
served row for this engine's pair Transition cell on both classes (a class contract: never a particular row by name)."""
import ast, os, re, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_OPT = os.path.normpath(os.path.join(HERE, "..", ".."))
DRIVER = os.path.join(KIT_OPT, "forward", "fast_inference", "driver")
SRC = open(os.path.join(DRIVER, "ef2_transition_cute.py")).read()


class TestT16Bind(unittest.TestCase):
    def test_words_and_env(self):
        tree = ast.parse(SRC)
        consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                  if getattr(t, "id", None) in ("WORD_ENV", "MODE_ENV", "TIER_OF_MODE", "DEFAULT_WORD")}
        self.assertEqual(consts["WORD_ENV"], "EF2_T16_WORD"); self.assertEqual(consts["MODE_ENV"], "ESMFOLD2_OPT")
        self.assertEqual(consts["TIER_OF_MODE"], {"fast": "fast", "big": "big", "exact": "exact"})   # big binds `big` literally
        self.assertEqual(consts["DEFAULT_WORD"], "fast")

    def test_no_kernel_in_the_module_and_no_shipped_object(self):
        self.assertNotIn("triton.jit", SRC); self.assertNotIn("nvrtc_sources", SRC); self.assertNotIn("AUDITED_SHA256", SRC)
        self.assertFalse(os.path.exists(os.path.join(DRIVER, "ef2_transition_cute.cuh")))
        self.assertFalse(os.path.exists(os.path.join(DRIVER, "prebuilt", "sm_90a", "ef2_transition_cute.cubin")))

    def test_run_resolves_the_word_first_and_steps_aside_by_name(self):
        tree = ast.parse(SRC)
        body = ast.get_source_segment(SRC, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run"))
        self.assertLess(body.index("TR.select(bound_word()"), body.index("TR.transition("))        # the tier word resolved for the call class, then the resolved row serves
        for token in ("sel.row in TR.STOCK_ROWS", "except TR.Refusal", 'STATS["face_refused"]', "_name_once(", "return None"):
            self.assertIn(token, body)
        for fn in ("_transition_forward_cute", "_pair_transition_forward_cute"):
            fwd = ast.get_source_segment(SRC, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn))
            self.assertIn("_run(self, x, residual=", fwd); self.assertIn("if y is None:", fwd)

    def test_lever_line_names_the_binding(self):
        ev = ast.get_source_segment(SRC, next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "kernel_evidence"))
        for token in ('"word"', '"provider"', '"core"', 'ev["rows"]', 'ev["refused"]'):
            self.assertIn(token, ev)

    def test_every_tier_word_resolves_for_the_engine_cell_on_both_classes(self):
        txt = open(os.path.join(KIT_OPT, "pyproject.toml")).read()
        v = re.search(r'\[tool\.opt_core\][^\[]*?version = "([0-9.]+)"', txt, re.S).group(1)
        self.assertGreaterEqual(tuple(int(p) for p in v.split(".")), (0, 5, 114, 0))               # the transition face's `big` tier word
        from opt_core.kernels import transition as TR
        self.assertTrue({"fast", "big", "exact"} <= set(TR.TIER_WORDS))
        for cc in ("9.0", "8.0"):
            for word in ("fast", "big", "exact"):
                try:
                    sel = TR.select(word, c=256, hidden=1024, n_tokens=800, dtype="bf16", direction="fwd", family="esmpair", residual=True, cc=cc, form="esmfused")
                    self.assertTrue(sel.row in TR.ROW_NAMES or sel.row in TR.STOCK_ROWS, (cc, word, sel.row))
                except TR.Refusal as r:                                                         # admitted: a refusal BY NAME carries the fallback row word
                    self.assertTrue(getattr(r, "fallback", None), (cc, word))


if __name__ == "__main__":
    unittest.main()
