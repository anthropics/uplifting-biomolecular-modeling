"""Lever xte (the trunk's fused inference Transition through the shared core's transition provider row esm_fused_exact by word; exact class): it
is a registry lever of the pair field on both models, the exact line's server set carries it and the fast line's does not (t16 / t15 own the
trunk Transition there — configure refuses the combination by name), the shim pins the upstream function it re-issues through the source guard,
resolves its word by class with a bitwise install check, the LEVER line names its words, and the kit's core pin floor reaches the core release
that carries the row (kernels.transition esm_fused_exact, 0.5.61)."""
import ast, os, re, unittest

from esmfold2_opt import modes, registry, report

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_OPT = os.path.normpath(os.path.join(HERE, "..", ".."))
DRIVER = os.path.join(KIT_OPT, "forward", "fast_inference", "driver")


def _server_table():
    src = open(os.path.join(DRIVER, "ef2_server.py")).read()
    return ast.literal_eval(next(n.value for n in ast.parse(src).body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "MODES" for t in n.targets)))


def _pair_of(table, server_mode):
    groups = dict(g.split(":", 1) if ":" in g else (g, "") for g in table[server_mode][3].split(";"))
    return groups.get("pair", "").split(",")


class TestXte(unittest.TestCase):
    def test_registry_row(self):
        lv = registry.LEVERS["xte"]
        self.assertEqual((lv.field, lv.kit_file, lv.tier_vs_kit_line, lv.probe), (registry.FIELD_PAIR, "ef2_xte.py", "bitwise", ("xte", "xte")))
        self.assertEqual(tuple(lv.variants), tuple(registry.ALL_VARIANTS))          # the trunk Transition exists on both models
        self.assertEqual(registry.STRATEGY["xte"], "LOCAL.fused_transition")
        self.assertIsNone(lv.classes)                                                # ONE exact set on every class: the word steps aside by name where no class entry / the provider refuses

    def test_exact_set_carries_it_fast_set_does_not(self):
        table = _server_table()
        self.assertIn("xte", _pair_of(table, modes.KIT_MODES["exact"].server_mode))
        fast_pair = _pair_of(table, modes.KIT_MODES["fast"].server_mode)
        self.assertNotIn("xte", fast_pair)                                          # t16 (9.0) / t15 (8.0) serve the trunk Transition on the fast line (measured faster per call; tolerance class)
        self.assertTrue({"t16", "t15"} & set(fast_pair))

    def test_configure_refuses_xte_beside_t16_by_name_and_installs_after_pair_v2(self):
        src = open(os.path.join(DRIVER, "ef2_server.py")).read()
        self.assertIn('"xte" in pair_flags and ("t16" in pair_flags or "t15" in pair_flags)', src)
        self.assertIn('XTE_LEVERS = ("xte",)', src)
        self.assertLess(src.index("ef2_pair_v2.install(model, trunk="), src.index("ef2_xte.install(model)"))   # the instance bindings sit over ef2_pair_v2's class forward

    def test_shim_words_guard_and_fallthrough(self):
        src = open(os.path.join(DRIVER, "ef2_xte.py")).read()
        tree = ast.parse(src)
        consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                  if getattr(t, "id", None) in ("UPSTREAM_SOURCES", "LEVERS", "XTE_C", "FUSED_EPS")}
        self.assertEqual(consts["LEVERS"], ("xte",))
        self.assertNotIn("XTE_ROWS", src); self.assertNotIn("XTE_WORDS", src)          # no kit row window / class word table: the provider's cells decide, by the tier word
        self.assertEqual(consts["UPSTREAM_SOURCES"], (("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"),))
        srcguard = open(os.path.join(DRIVER, "ef2_srcguard.py")).read()
        self.assertIn('("transformers.models.esmfold2.modeling_esmfold2_common", "Transition.forward"):', srcguard)   # the digest the shim checks is pinned
        fwd = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_xte_forward"))
        for token in ("xte_fallthrough", "C.Transition.forward(self, x)", 'P.serve(self, x, residual=True, kind="xte")', "self._can_use_fused_path(x)", "if y is None:"):
            self.assertIn(token, fwd)
        inst = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "install"))
        for token in ('ef2_srcguard.check("xte", keys=UPSTREAM_SOURCES)', "eligible_modules(model)", "no_eligible_module", "P._pack(m)"):
            self.assertIn(token, inst)
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"install", "uninstall", "levers_on", "refusal", "stats", "evidence", "describe"} <= names)

    def test_stack_and_report_rows(self):
        stack_src = open(os.path.join(os.path.dirname(HERE), "stack.py")).read()
        self.assertIn('("xte", "ef2_xte", lambda m: m.levers_on())', stack_src)
        self.assertIn('sys.modules["ef2_xte"].refusal()', stack_src)
        self.assertEqual(tuple(report.LEVERFOLD_KEYS["ef2_xte"]), ("xte_calls", "xte_fallthrough"))
        self.assertIn("ef2_xte", report.LEVER_MODULES_V2)
        rep_src = open(os.path.join(os.path.dirname(HERE), "report.py")).read()
        self.assertIn('elif name == "xte":', rep_src)

    def test_core_pin_floor_reaches_the_row(self):
        txt = open(os.path.join(KIT_OPT, "pyproject.toml")).read()
        v = re.search(r'\[tool\.opt_core\][^\[]*?version = "([0-9.]+)"', txt, re.S).group(1)
        self.assertGreaterEqual(tuple(int(p) for p in v.split(".")), (0, 5, 114, 0))


if __name__ == "__main__":
    unittest.main()
