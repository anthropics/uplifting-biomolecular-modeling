"""Per-class lever sets: a mode is ONE set in the server's table; the box's compute-capability class decides which of its transition kernels
serve (modes.class_drop over registry Lever.classes / SUPERSEDED_ON). On 9.0 (H100 / H200) t16 supersedes t15 / t15msa / t10; on every other class
t16 steps aside BY CLASS (never a refusal) and t15 / t15msa / t10 serve; an unknown class subtracts nothing. The pair TriMul (lever tx) is ONE binding
on every class: the shared core's TriMul provider decides the row per capability under the set's tier word, so no class subtracts it
(registry.STEPS_ASIDE_ON is empty). The words: DRY-RUN / APPLIED ``not_for_class=<names>``, LEVER ``state=skipped reason=not_for_class:<class>:<reason>``."""
import os
import re
import unittest

from esmfold2_opt import modes, registry, report

KIT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "forward", "fast_inference"))
SM90_ONLY = ("t16",)
RETIRED = ("t9", "sigmoid", "incnt", "formtab", "k3cute", "lnfold")            # the kit's own TriMul kernels: served by the shared core's provider under the tier word (lever tx)


class TestClassSets(unittest.TestCase):
    def test_registry_declares_the_classes(self):
        for name in SM90_ONLY:
            self.assertEqual(registry.LEVERS[name].classes, ("9.0",), name)
        self.assertEqual(registry.SUPERSEDED_ON["t15"], {"9.0": "t16"})
        self.assertEqual(registry.SUPERSEDED_ON["t15msa"], {"9.0": "t16"}); self.assertEqual(registry.SUPERSEDED_ON["t10"], {"9.0": "t16"})
        self.assertIsNone(registry.LEVERS["t15"].classes)
        self.assertIsNone(registry.LEVERS["tx"].classes)                                 # every class: the provider's cell table decides the row per card
        self.assertEqual(registry.STEPS_ASIDE_ON, {})
        for name in RETIRED:
            self.assertNotIn(name, registry.LEVERS, name)
        self.assertEqual(registry.class_word("9.0"), "sm90"); self.assertEqual(registry.class_word(None), "unknown")

    def test_fast_on_sm90_runs_t16_and_tx(self):
        res = modes.resolve("fast", "full_msa", KIT, cc="9.0")
        self.assertEqual(res.not_for_class, {"t15": "superseded_by_t16", "t15msa": "superseded_by_t16", "t10": "superseded_by_t16"})
        for name in SM90_ONLY + ("tx",):
            self.assertIn(name, res.levers_for_variant, name)
        for name in ("t15", "t15msa", "t10"):
            self.assertNotIn(name, res.levers_for_variant); self.assertIn(name, res.levers_off)
        self.assertLess(res.levers.index("t5"), res.levers.index("tx"))              # the trimul group installs right after W4
        self.assertLess(res.levers.index("tx"), res.levers.index("ax"))

    def test_fast_on_sm80_keeps_t15_and_tx(self):
        """On 8.0 (A100) t15 / t15msa / t10 serve the transitions and the pair TriMul binding stays (the provider's 8.0 cells decide its row) under
        fast and big alike; every other class keeps it too."""
        for mode in ("fast", "big"):
            res = modes.resolve(mode, "full_msa", KIT, cc="8.0")
            self.assertEqual(res.not_for_class, {n: "serves_sm90" for n in SM90_ONLY}, mode)
            for name in ("t15", "t15msa", "t10", "tx"):
                self.assertIn(name, res.levers_for_variant, name)
            for name in SM90_ONLY:
                self.assertNotIn(name, res.levers_for_variant); self.assertIn(name, res.levers_off)
            for gone in RETIRED:
                with self.assertRaises(ValueError):                                      # not a lever of the kit any more: naming it in the ablation variable is an unknown token
                    modes.resolve(mode, "fast", KIT, cc="8.0", ablate=gone)
            self.assertNotIn("tx", modes.resolve(mode, "fast", KIT, cc="8.0", ablate="tx").levers_for_variant)   # individually switchable on every class
        self.assertEqual(modes.resolve("fast", "fast", KIT, cc="10.0").not_for_class, {n: "serves_sm90" for n in SM90_ONLY})
        for cc in ("10.0", "9.0", "8.0", None):
            self.assertIn("tx", modes.resolve("fast", "fast", KIT, cc=cc).levers_for_variant, cc)
        self.assertIsNone(registry.class_excludes(registry.LEVERS["tx"], "8.0")); self.assertIsNone(registry.class_excludes(registry.LEVERS["tx"], "9.0"))
        self.assertIsNone(registry.class_excludes(registry.LEVERS["tx"], None))          # an unknown class subtracts nothing

    def test_unknown_class_subtracts_nothing_and_exact_is_untouched(self):
        res = modes.resolve("fast", "full_msa", KIT, cc=None)
        self.assertEqual(res.not_for_class, {}); self.assertIsNone(res.cc)
        for name in SM90_ONLY + ("t15", "t15msa", "tx"):
            self.assertIn(name, res.levers)
        for cc in ("9.0", "8.0", None):
            ex = modes.resolve("exact", "full_msa", KIT, cc=cc)
            self.assertEqual(ex.not_for_class, {}, cc)
            self.assertFalse(set(SM90_ONLY) & set(ex.levers))
            self.assertNotIn("tx", ex.levers)                                             # the exact set's pair TriMul is the upstream fused statement itself (no byte-vouched exact-class row to bind)

    def test_big_inherits_the_class_policy_and_ablation_composes(self):
        b = modes.resolve("big", "full_msa", KIT, cc="9.0")
        self.assertEqual(b.not_for_class, {"t15": "superseded_by_t16", "t15msa": "superseded_by_t16", "t10": "superseded_by_t16"})
        self.assertIn("t16", b.levers_for_variant); self.assertIn("tx", b.levers_for_variant)
        a = modes.resolve("fast", "fast", KIT, cc="9.0", ablate="tx")
        self.assertNotIn("tx", a.levers_for_variant); self.assertIn("t16", a.levers_for_variant)
        t = modes.resolve("fast", "full_msa", KIT, cc="9.0", ablate="t16")                # an ablated t16 supersedes nothing: t15 / t15msa / t10 serve the 9.0 box as they serve 8.0
        self.assertEqual(t.not_for_class, {}); self.assertIn("t15", t.levers_for_variant); self.assertIn("t10", t.levers_for_variant); self.assertNotIn("t16", t.levers)
        with self.assertRaises(ValueError):                                              # a lever another class serves is not in this box's set: naming it is an unknown-token error
            modes.resolve("fast", "fast", KIT, cc="8.0", ablate="t16")

    def test_target_word_names_a_class(self):
        self.assertEqual(modes.target_cc("H100"), "9.0"); self.assertEqual(modes.target_cc("h200"), "9.0")
        self.assertEqual(modes.target_cc("A100"), "8.0"); self.assertEqual(modes.target_cc("B200"), "10.0")
        self.assertIsNone(modes.target_cc(None)); self.assertIsNone(modes.target_cc("mystery"))

    def test_composition_by_class(self):
        """The server refuses t16 beside t15 / t15msa BY NAME; the class-resolved sets pass and carry the TriMul word on both classes."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("ef2_server_under_test", os.path.join(KIT, "driver", "ef2_server.py"))
        srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
        with self.assertRaises(ValueError) as cm:
            srv.check_composition("opt14_msa", off=())                                    # the raw table carries t15, t15msa AND t16: the package always subtracts one side by class
        self.assertIn("t16 together with t15", str(cm.exception))
        sm90 = modes.resolve("fast", "full_msa", KIT, cc="9.0"); sm80 = modes.resolve("fast", "full_msa", KIT, cc="8.0")
        self.assertEqual(srv.check_composition("opt14_msa", off=set(sm90.levers_off))["pair"], ["t16"])
        self.assertEqual(srv.check_composition("opt14_msa", off=set(sm80.levers_off))["pair"], ["t15", "t15msa"])
        self.assertEqual(srv.check_composition("opt14_msa", off=set(sm80.levers_off))["trimul"], ["tx"])
        self.assertEqual(srv.check_composition("opt14_msa", off=set(sm90.levers_off))["trimul"], ["tx"])
        self.assertEqual(srv.check_composition("opt7x", off=set(modes.resolve("exact", "full_msa", KIT, cc="8.0").levers_off))["trimul"], [])

    def test_the_words(self):
        rep = {"mode": "fast", "variant": "full_msa", "server_line": "opt14_msa", "levers_not_for_class": {"t15": "superseded_by_t16", "t15msa": "superseded_by_t16", "t10": "superseded_by_t16"}, "class_cc": "9.0",
               "levers_planned": ["fused", "t16"], "levers_not_for_variant": []}
        self.assertEqual(report.class_word(rep), " not_for_class=t15,t15msa,t10")
        self.assertEqual(report.class_word({"levers_not_for_class": {}}), "")
        rec = {"model_index": 0, "levers_applied": ["fused", "t16"], "levers_fallback": [], "levers_substituted": {}, "fallback_reasons": {}, "levers_not_for_variant": []}
        by = {re.search(r" name=(\S+)", l).group(1): l for l in report.lever_lines(rep, rec)}
        self.assertIn(" state=skipped reason=not_for_class:sm90:superseded_by_t16 ", by["t15"])
        self.assertIn(" state=on impl=ef2_transition_cute.py origin=kit strategy=LOCAL.fused_transition ", by["t16"])
        rep80 = dict(rep, levers_not_for_class={n: "serves_sm90" for n in SM90_ONLY}, class_cc="8.0")
        by80 = {re.search(r" name=(\S+)", l).group(1): l for l in report.lever_lines(rep80, dict(rec, levers_applied=["fused", "t15", "tx"]))}
        self.assertIn(" state=skipped reason=not_for_class:sm80:serves_sm90 impl=ef2_transition_cute.py ", by80["t16"])
        self.assertIn(" state=on impl=ef2_w4.py origin=kit strategy=F2.fpf_trimul_fast ", by80["tx"])
        for gone in RETIRED:
            self.assertNotIn(gone, by80)
        self.assertTrue(report.applied_line(rep, rec).endswith(" not_for_variant=none not_for_class=t15,t15msa,t10"))


if __name__ == "__main__":
    unittest.main()
