"""Lever xtr (the MSA module's PairTransition through the shared core's transition provider by word, LayerNorm given; exact class): it is a
registry lever of the pair field on the Full model only, the exact line's server set carries it and the fast line's does not (t16 / t15msa own
PairTransition there — configure refuses the combination by name), the source guard pins the upstream function it re-issues, the LEVER line
names its words, and the kit's core pin floor reaches the core release that carries the provider (kernels.transition, 0.5.34)."""
import ast, os, re, unittest

from esmfold2_opt import modes, registry, report

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_OPT = os.path.normpath(os.path.join(HERE, "..", ".."))
DRIVER = os.path.join(KIT_OPT, "forward", "fast_inference", "driver")


def _server_table():
    src = open(os.path.join(DRIVER, "ef2_server.py")).read()
    return ast.literal_eval(next(n.value for n in ast.parse(src).body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "MODES" for t in n.targets)))


class TestXtr(unittest.TestCase):
    def test_registry_row(self):
        lv = registry.LEVERS["xtr"]
        self.assertEqual((lv.field, lv.kit_file, lv.tier_vs_kit_line, lv.probe), (registry.FIELD_PAIR, "ef2_pair_v2.py", "bitwise", ("pair", "xtr")))
        self.assertEqual(tuple(lv.variants), tuple(registry.FULL_MODEL))            # the MSA module exists on the Full model only
        self.assertEqual(registry.STRATEGY["xtr"], "LOCAL.fused_transition")
        self.assertIsNone(lv.classes)                                                # ONE exact set on every class (test_class_sets): the lever's words step aside by name where the provider refuses

    def test_exact_set_carries_it_fast_set_does_not(self):
        table = _server_table()
        def pair_of(server_mode):
            groups = dict(g.split(":", 1) if ":" in g else (g, "") for g in table[server_mode][3].split(";"))
            return groups.get("pair", "").split(",")
        self.assertIn("xtr", pair_of(modes.KIT_MODES["exact"].server_mode))
        fast_pair = pair_of(modes.KIT_MODES["fast"].server_mode)
        self.assertNotIn("xtr", fast_pair)                                          # t16 (9.0) / t15msa (8.0) serve PairTransition on the fast line
        self.assertTrue({"t16", "t15msa"} & set(fast_pair))

    def test_configure_refuses_xtr_beside_t16_by_name(self):
        src = open(os.path.join(DRIVER, "ef2_server.py")).read()
        self.assertIn('"xtr" in pair_flags and ("t16" in pair_flags or "t15msa" in pair_flags)', src)
        self.assertIn('PAIR_V2_LEVERS = ("t15", "t15msa", "xtr")', src)

    def test_source_guard_pins_pair_transition(self):
        src = open(os.path.join(DRIVER, "ef2_srcguard.py")).read()
        self.assertRegex(src, r'"xtr": \(\("transformers\.models\.esmfold2\.modeling_esmfold2", "PairTransition\.forward"\),\)')

    def test_lever_module_words_no_windows(self):
        src = open(os.path.join(DRIVER, "ef2_pair_v2.py")).read()
        tree = ast.parse(src)
        self.assertNotIn("XTR_ROWS", src); self.assertNotIn("XTR_WORDS", src)          # no kit row floors / class word table: the provider's cells decide, by the tier word
        fwd = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pair_transition_forward_xtr")
        body = ast.get_source_segment(src, fwd)
        self.assertIn("self.norm(x)", body)                                         # the module's own LayerNorm statement, then the provider with x_ln given
        self.assertIn('residual=False, kind="xtr", x_ln=n', body)
        self.assertIn("c in (256, 128)", body)                                      # pair_transition (256) and msa_transition (128)

    def test_leverfold_keys_and_line_evidence(self):
        self.assertTrue({"xtr_calls", "xtr_fallthrough"} <= set(report.LEVERFOLD_KEYS["ef2_pair_v2"]))
        src = open(os.path.join(os.path.dirname(HERE), "report.py")).read()
        self.assertIn('m.bind_words().items()', src)

    def test_core_pin_floor_reaches_the_provider(self):
        txt = open(os.path.join(KIT_OPT, "pyproject.toml")).read()
        v = re.search(r'\[tool\.opt_core\][^\[]*?version = "([0-9.]+)"', txt, re.S).group(1)
        self.assertGreaterEqual(tuple(int(p) for p in v.split(".")), (0, 5, 34, 0))


if __name__ == "__main__":
    unittest.main()
