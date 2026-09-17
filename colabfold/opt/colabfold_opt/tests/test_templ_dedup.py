"""TEMPL_DEDUP — identical template rows embedded once: the multimer TemplateEmbedding rebound to the dedup subclass and model.RunModel to
the counting subclass (markers, idempotence, disable); the host census (T rows / U distinct consecutive rows per predict; the monomer route and
a template-less configuration counted by name — registry.STEP_ASIDE_RULES, so a run of such predicts is a by-design step-aside, never PARTIAL);
the registry row and its LEVER line; the mode tuples; the ablation name; big at --n_gpu P>1 drops it by name. No jax, no GPU (the dedup
body itself is traced only in the box: CHANGES.md 0.2.10 carries its measurements)."""
import os
import tempfile
import types
import unittest

import numpy as np

from colabfold_opt import ablation, manifest, modes, registry, report, templ_dedup
from colabfold_opt.tests import _stubs


def feats(rows, n=7, monomer=False):
    """A feature dict with template rows built from small integers: equal integers give equal rows."""
    aat = np.stack([np.full((n,), r, np.int32) for r in rows]); pos = np.stack([np.full((n, 37, 3), float(r), np.float32) for r in rows])
    msk = np.stack([np.full((n, 37), float(r % 2), np.float32) for r in rows])
    key = "template_all_atom_masks" if monomer else "template_all_atom_mask"
    return {"template_aatype": aat, "template_all_atom_positions": pos, key: msk, "aatype": np.zeros((n,), np.int32)}


def cfg(multimer=True, templates=True, dropout=False):
    gc = ({"multimer_mode": multimer} if multimer else {}) | ({"eval_dropout": True} if dropout else {})
    return types.SimpleNamespace(model=types.SimpleNamespace(global_config=_Get(gc), embeddings_and_evoformer=types.SimpleNamespace(template=_Get({"enabled": templates}))))


class _Get(dict):
    def __getattr__(self, k):
        return self[k]


class TestCensus(unittest.TestCase):
    def setUp(self):
        templ_dedup.reset_for_tests()

    def test_consecutive_unique(self):
        self.assertEqual(templ_dedup.consecutive_unique(feats([0, 0, 0, 0])), (4, 1))          # colabfold's default: four mock rows → one embedded, three reuse
        self.assertEqual(templ_dedup.consecutive_unique(feats([1, 2, 3, 4])), (4, 4))          # four real templates: each embedded (the lever steps aside by the data)
        self.assertEqual(templ_dedup.consecutive_unique(feats([5, 6, 0, 0])), (4, 3))          # two hits + zero padding: the padding row embedded once
        self.assertEqual(templ_dedup.consecutive_unique(feats([5, 0, 5, 0])), (4, 4))          # only the PREVIOUS row is compared (the scan's carry): no reuse across a different row
        self.assertEqual(templ_dedup.consecutive_unique(feats([3])), (1, 1))
        self.assertIsNone(templ_dedup.consecutive_unique(feats([0, 0], monomer=True)))          # the monomer dict names the mask in the plural: not the multimer arrays
        self.assertIsNone(templ_dedup.consecutive_unique({"aatype": 1}))

    def test_census_tallies_and_step_asides(self):
        st = templ_dedup._STATE
        self.assertIsNone(templ_dedup.census(feats([0, 0, 0, 0]), cfg())); self.assertIsNone(templ_dedup.census(feats([0, 0, 0, 0]), cfg()))
        self.assertIsNone(templ_dedup.census(feats([1, 2, 3, 4]), cfg()))
        self.assertEqual((st["predicts"], st["templates"], st["embedded"], st["reused"], st["unique_hist"], st["fallbacks"], st["fallback_by"]), (3, 12, 6, 6, "1:2,4:1", 0, "none"))
        self.assertEqual(templ_dedup.census(feats([0, 0], monomer=True), cfg(multimer=False)), "monomer_route")
        self.assertEqual(templ_dedup.census(feats([0, 0], monomer=True), None), "monomer_route")                                     # no config: the monomer feature names say it
        self.assertEqual(templ_dedup.census(feats([0, 0, 0, 0]), cfg(templates=False)), "template_disabled")
        self.assertEqual((st["predicts"], st["fallbacks"], st["fallback_by"]), (6, 3, "monomer_route:2,template_disabled:1"))
        for rule in (templ_dedup.MONOMER_ROUTE, templ_dedup.TEMPLATE_DISABLED):
            self.assertIn(rule, registry.STEP_ASIDE_RULES)

    def test_a_monomer_run_is_a_step_aside_by_name_not_partial(self):
        templ_dedup.census(feats([0], monomer=True), cfg(multimer=False)); templ_dedup.census(feats([0], monomer=True), cfg(multimer=False))
        st = dict(templ_dedup._STATE, enabled=True)
        self.assertEqual((st["calls"], st["fallbacks"], st["fallback_by"]), (0, 2, "monomer_route:2"))
        self.assertEqual(manifest.stepped_aside_rule(st), "monomer_route")
        served = dict(st, calls=1)                                                                                                      # a multimer predict traced the dedup body: served, nothing to excuse
        self.assertIsNone(manifest.stepped_aside_rule(served))
        self.assertIsNone(manifest.stepped_aside_rule(dict(st, fallbacks=0, fallback_by="none")))                                       # nothing counted at all stays PARTIAL


class TestRebinding(unittest.TestCase):
    def setUp(self):
        self.mods, self.saved = _stubs.install()
        self.mm, self.model = self.mods["alphafold.model.modules_multimer"], self.mods["alphafold.model.model"]
        self.stock, self.stock_model = self.mm.TemplateEmbedding, self.model.RunModel

    def tearDown(self):
        from colabfold_opt import device_resident
        templ_dedup.reset_for_tests(); device_resident.reset_for_tests(); _stubs.remove(self.saved)

    def test_enable_rebinds_both_classes_idempotently_and_disable_restores(self):
        self.assertFalse(templ_dedup.marker_present(self.mm))
        st = templ_dedup.enable(multimer=self.mm, model=self.model)
        self.assertTrue(st["enabled"]); self.assertTrue(templ_dedup.marker_present(self.mm))
        cls, mcls = self.mm.TemplateEmbedding, self.model.RunModel
        self.assertTrue(issubclass(cls, self.stock)); self.assertIs(cls._templ_dedup_wrapped, self.stock); self.assertEqual((cls.__name__, cls.__module__), ("TemplateEmbedding", self.stock.__module__))
        self.assertTrue(issubclass(mcls, self.stock_model)); self.assertTrue(getattr(mcls, templ_dedup.CENSUS_MARKER)); self.assertEqual(mcls.__name__, "RunModel")
        templ_dedup.enable(multimer=self.mm, model=self.model)                                                                          # idempotent: no second wrapping
        self.assertIs(self.mm.TemplateEmbedding, cls); self.assertIs(self.model.RunModel, mcls)
        self.assertEqual(registry.marker_present(modes.TEMPL_LEVER), True)                                                              # the registry's probe reads the same marker
        templ_dedup.disable()
        self.assertIs(self.mm.TemplateEmbedding, self.stock); self.assertIs(self.model.RunModel, self.stock_model); self.assertFalse(templ_dedup._STATE["enabled"])

    def test_the_counting_runmodel_counts_then_runs_the_wrapped_predict(self):
        templ_dedup.enable(multimer=self.mm, model=self.model)
        m = self.model.RunModel(config=cfg(), params={"w": [1.0]})
        out, r = m.predict(feats([0, 0, 0, 0], n=2), random_seed=0)
        self.assertEqual(r, 3); self.assertIn("ranking_confidence", out)                                                                 # the stub's stock predict ran (3 recycles)
        st = templ_dedup._STATE
        self.assertEqual((st["predicts"], st["templates"], st["embedded"], st["reused"], st["unique_hist"]), (1, 4, 1, 3, "1:1"))
        self.assertTrue(getattr(self.model.RunModel, "_templ_dedup_census"))


    def test_dropout_runs_stocks_body_uncounted(self):
        cls = templ_dedup.enable(multimer=self.mm, model=self.model)  and self.mm.TemplateEmbedding
        inst = cls.__new__(cls); inst.config, inst.global_config, inst.name = None, _Get({"eval_dropout": True}), "template_embedding"
        out = cls.__call__(inst, None, {"template_aatype": None}, None, None, False, safe_key="k")
        self.assertEqual(out, ("stock", {"template_aatype": None})); self.assertEqual(templ_dedup._STATE["calls"], 0)          # stock's body ran; no dedup trace counted
        inst.global_config = _Get({})
        out = cls.__call__(inst, None, {"template_aatype": None}, None, None, True, safe_key="k")                             # is_training likewise
        self.assertEqual(out[0], "stock")

    def test_census_over_device_resident(self):
        from colabfold_opt import device_resident
        device_resident.enable(self.model)                                                                                              # the table's order: DEVICE_RESIDENT first, the census wraps its class
        templ_dedup.enable(multimer=self.mm, model=self.model)
        self.assertTrue(getattr(self.model.RunModel, device_resident.MARKER)); self.assertTrue(getattr(self.model.RunModel, templ_dedup.CENSUS_MARKER))
        self.assertTrue(device_resident.marker_present(self.model))


class TestFiling(unittest.TestCase):
    def test_registry_row_and_lever_line(self):
        lv = registry.LEVERS[modes.TEMPL_LEVER]
        self.assertEqual((lv.impl, lv.origin, lv.strategy, lv.kit_file, lv.cls, lv.in_mode), ("colabfold_opt.templ_dedup", "kit", "LOCAL.colabfold.template_dedup", "colabfold_opt/templ_dedup.py", "forward", True))
        self.assertEqual(templ_dedup.STRATEGY, lv.strategy); self.assertEqual(lv.probe, ("module_state", "colabfold_opt.templ_dedup", "enabled"))
        self.assertEqual(registry.MARKERS[modes.TEMPL_LEVER], ("class_attr", "alphafold.model.modules_multimer", "TemplateEmbedding", "_templ_dedup"))
        templ_dedup.reset_for_tests()
        ev = {k: v for k, v in templ_dedup._STATE.items() if k != "enabled"}
        self.assertEqual(report.lever_line(modes.TEMPL_LEVER, "on", None, **ev),
                         "[colabfold-opt] LEVER name=TEMPL_DEDUP state=on impl=colabfold_opt.templ_dedup origin=kit strategy=LOCAL.colabfold.template_dedup calls=0 predicts=0 templates=0 embedded=0 reused=0 unique_hist=none fallbacks=0 fallback_by=none sites=multimer")
        self.assertEqual(report.lever_line(modes.TEMPL_LEVER, "off", templ_dedup.N_GPU_REASON),
                         "[colabfold-opt] LEVER name=TEMPL_DEDUP state=off reason=n_gpu>1 impl=colabfold_opt.templ_dedup origin=kit strategy=LOCAL.colabfold.template_dedup")

    def test_modes_and_ablation_name(self):
        self.assertIn(modes.TEMPL_LEVER, modes.TABLE["fast"][0]); self.assertIn(modes.TEMPL_LEVER, modes.TABLE["big"][0]); self.assertNotIn(modes.TEMPL_LEVER, modes.TABLE["exact"][0])
        self.assertGreater(modes.TABLE["fast"][0].index(modes.TEMPL_LEVER), modes.TABLE["fast"][0].index(modes.COL_LEVER))                 # applied after the attention levers in fast; before ROWPAIR in big
        self.assertLess(modes.TABLE["big"][0].index(modes.TEMPL_LEVER), modes.TABLE["big"][0].index(modes.TP_LEVER))
        self.assertIn((modes.TEMPL_LEVER, modes.TEMPL_LEVER_MODULE), modes.ONE_DEVICE_LEVERS); self.assertIn(modes.TEMPL_LEVER, modes.ONE_DEVICE_PAIR_LEVERS)
        fast = list(modes.TABLE["fast"][0])
        self.assertEqual(ablation.validate("fast", ["TEMPL_DEDUP"], fast), ["TEMPL_DEDUP"])
        self.assertEqual(templ_dedup.FEATURES, ("template_aatype", "template_all_atom_positions", "template_all_atom_mask"))           # modules_multimer.py:841-843, the multimer feature names


class TestVerdictOnUnservedRoutes(unittest.TestCase):
    """X1: a run whose every predict TEMPL_DEDUP could not serve — the monomer/ptm route (colabfold builds model_3's config, alphafold2_ptm's
    other template body) or a model built without the template embedder — ends by the NAMED by-design word on its LEVER line and exit 0, never
    as a partial activation (exit 3). Through the manifest's own verdict, as the model process writes it."""

    def setUp(self):
        templ_dedup.reset_for_tests()
        self.tmp = tempfile.mkdtemp(); self.res = os.path.join(self.tmp, "run"); os.makedirs(self.res)
        self.saved = os.environ.pop(manifest.ENV_LAUNCH_ID, None)

    def tearDown(self):
        if self.saved is not None:
            os.environ[manifest.ENV_LAUNCH_ID] = self.saved
        templ_dedup.reset_for_tests()

    def verdict(self, td_state):
        rep = {"active": True, "mode": "fast", "levers": ["AF_PALLAS_ATTN", "TEMPL_DEDUP"], "levers_applied": ["AF_PALLAS_ATTN", "TEMPL_DEDUP"], "reason": None}
        kit = {"enabled": True, "calls": 12, "fallbacks": 0}
        manifest.write(self.res, rep); manifest.record_exit(self.res, kit, {"AF_PALLAS_ATTN": kit, "TEMPL_DEDUP": td_state})
        man = manifest.read(self.res)
        return man, manifest.verdict(self.res, "fast", [], 0)

    def test_monomer_route_and_template_disabled_are_by_design_rc_0(self):
        for route, config in (("monomer_route", cfg(multimer=False)), ("template_disabled", cfg(templates=False)), ("dropout_enabled", cfg(dropout=True))):
            with self.subTest(route=route):
                templ_dedup.reset_for_tests()
                f = feats([0], monomer=True) if route == "monomer_route" else feats([0, 0, 0, 0])
                for _ in range(3):
                    self.assertEqual(templ_dedup.census(f, config), route)
                man, v = self.verdict(dict(templ_dedup._STATE, enabled=True))
                self.assertEqual((man["partial"], man["stepped_aside"].get("TEMPL_DEDUP")), ([], route), man)
                self.assertEqual((v["ok"], v["rc"]), (True, 0), v)

    def test_nothing_counted_stays_partial_rc_3(self):
        man, v = self.verdict(dict(templ_dedup._STATE, enabled=True))                                                                  # calls=0, no predict seen: the lever never engaged
        self.assertEqual(man["partial"], ["TEMPL_DEDUP"]); self.assertEqual((v["ok"], v["rc"], v["reason"]), (False, report.EXIT_NOT_ACTIVE, "kernel_not_engaged"))


if __name__ == "__main__":
    unittest.main()
