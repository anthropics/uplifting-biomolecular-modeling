"""The activation report states the CUDA-graph switches as the levers run with them in this process (`effective`): a mode override
wins, else the caller's EF2_GRAPH_BUDGET_TOKENS / EF2_GRAPH_LRU_SAMPLER, else the package's default (EF2_GRAPH_LRU_SAMPLER=2, exported at activation),
else the driver's default — each with its source. CPU only."""
import os
import re
import types
import unittest

from esmfold2_opt import modes, stack

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(os.path.dirname(os.path.dirname(HERE)), "forward", "fast_inference", "driver", "ef2_opt.py")


class TestEffectiveGraphSettings(unittest.TestCase):
    def test_default_env_mode_precedence(self):
        plain = types.SimpleNamespace(overrides={})
        eff = stack.effective_graph_settings(plain, environ={})
        self.assertEqual({k: eff[k] for k in ("graph_budget_tokens", "graph_budget_tokens_source", "graph_lru_sampler", "graph_lru_sampler_source")},
                         {"graph_budget_tokens": 0, "graph_budget_tokens_source": "default", "graph_lru_sampler": 2, "graph_lru_sampler_source": "package"})
        self.assertEqual({k: eff[k] for k in ("graph_budget_tokens_trunk", "graph_budget_tokens_encoder", "graph_budget_tokens_sampler", "graph_budget_tokens_sampler_source", "recycle_graph_max_tokens")},
                         {"graph_budget_tokens_trunk": -1, "graph_budget_tokens_encoder": -1, "graph_budget_tokens_sampler": -1, "graph_budget_tokens_sampler_source": "default", "recycle_graph_max_tokens": 512})
        self.assertEqual([eff[k + "_effective"] for k in stack.SITE_KEYS], [0, 0, 0])                 # -1 = the site inherits the per-shape budget (0: no cap)
        self.assertEqual(modes.PACKAGE_GRAPH_DEFAULTS, {modes.ENV_GRAPH_LRU_SAMPLER: 2})               # the package default for every kit mode: two sampler step graphs per generation
        self.assertEqual(modes.PACKAGE_GRAPH_BUDGET_SAMPLER, {modes.ENV_GRAPH_BUDGET_SAMPLER: 1536})  # the sampler site under exact / fast: captured up to 1536 tokens, eager (references only) above
        self.assertEqual(modes.PACKAGE_GRAPH_BUDGET_FULL, {modes.ENV_GRAPH_BUDGET: 800}); self.assertEqual(modes.FULL_BUDGET_MODES, ("exact", "fast"))   # the Full model's per-shape budget = big's line budget and the Fast model's sites
        self.assertEqual(modes.PACKAGE_GRAPH_BUDGET_FASTMODEL, {modes.ENV_GRAPH_BUDGET_TRUNK: 800, modes.ENV_GRAPH_BUDGET_ENCODER: 800}); self.assertEqual(modes.FASTMODEL_BUDGET_SERVER_VARIANT, "fast")
        self.assertEqual(stack.package_graph_exports(plain, environ={}), {modes.ENV_GRAPH_LRU_SAMPLER: "2"})   # exported at activation when nobody sets it ...
        self.assertEqual(stack.activation_exports(plain, environ={}), {modes.ENV_GRAPH_LRU_SAMPLER: "2"})
        env = {modes.ENV_GRAPH_BUDGET: "1300", modes.ENV_GRAPH_LRU_SAMPLER: "1"}
        eff = stack.effective_graph_settings(plain, environ=env)
        self.assertEqual({k: eff[k] for k in ("graph_budget_tokens", "graph_budget_tokens_source", "graph_lru_sampler", "graph_lru_sampler_source")},
                         {"graph_budget_tokens": 1300, "graph_budget_tokens_source": "env", "graph_lru_sampler": 1, "graph_lru_sampler_source": "env"})
        self.assertEqual([eff[k + "_effective"] for k in stack.SITE_KEYS], [1300, 1300, 1300])        # the caller's per-shape budget governs every site it does not set itself
        self.assertEqual(stack.package_graph_exports(plain, environ=env), {})                          # ... and never over the caller's own value
        self.assertEqual(stack.package_graph_exports(plain, environ=dict(env, **{modes.ENV_GRAPH_BUDGET_SAMPLER: "1945"})), {})
        self.assertEqual(stack.effective_graph_settings(plain, environ=dict(env, **{modes.ENV_GRAPH_BUDGET_SAMPLER: "1945"}))["graph_budget_tokens_sampler_effective"], 1945)
        mk = types.SimpleNamespace(overrides={modes.ENV_MK: "1"})
        self.assertEqual(stack.activation_exports(mk, environ={}), {modes.ENV_MK: "1", modes.ENV_GRAPH_LRU_SAMPLER: "2"})   # beside the mode's own overrides
        big = types.SimpleNamespace(overrides=dict(modes.BIG_FAST.overrides))                     # the memory line's own word (no capture) wins over the caller's
        eff = stack.effective_graph_settings(big, environ=dict(env, **{modes.ENV_GRAPH_CAPTURE: "1"}))
        self.assertEqual((eff["graph_capture"], eff["graph_capture_source"]), (0, "mode"))
        self.assertEqual((eff["graph_budget_tokens"], eff["graph_budget_tokens_source"]), (1300, "env"))   # a caller's budget is reported as given; it governs nothing while capture is off
        self.assertEqual((eff["graph_lru_sampler"], eff["graph_lru_sampler_source"]), (1, "env"))
        self.assertEqual((stack.effective_graph_settings(plain, environ={})["graph_capture"], stack.effective_graph_settings(plain, environ={})["graph_capture_source"]), (1, "default"))

    def test_defaults_are_the_drivers(self):
        if not os.path.isfile(DRIVER):
            self.skipTest("carried driver not beside the package")
        src = open(DRIVER).read()
        self.assertRegex(src, r'graph_budget_tokens = int\(os\.environ\.get\("' + modes.ENV_GRAPH_BUDGET + r'", "%d"\)\)' % modes.GRAPH_ENV_DEFAULTS[modes.ENV_GRAPH_BUDGET])
        self.assertRegex(src, r'lru_sampler = int\(os\.environ\.get\("' + modes.ENV_GRAPH_LRU_SAMPLER + r'", "%d"\)\)' % modes.GRAPH_ENV_DEFAULTS[modes.ENV_GRAPH_LRU_SAMPLER])
        for env_name, attr in ((modes.ENV_GRAPH_BUDGET_TRUNK, "graph_budget_tokens_trunk"), (modes.ENV_GRAPH_BUDGET_ENCODER, "graph_budget_tokens_encoder"),
                               (modes.ENV_GRAPH_BUDGET_SAMPLER, "graph_budget_tokens_sampler"), (modes.ENV_RECYCLE_GRAPH_MAX, "recycle_graph_max_tokens")):
            self.assertRegex(src, attr + r' = int\(os\.environ\.get\("' + env_name + r'", "%d"\)\)' % modes.GRAPH_ENV_DEFAULTS[env_name])   # the per-site budgets and rg's ceiling: the driver's own defaults
            self.assertNotIn(env_name, modes.SWITCHES)
        self.assertRegex(src, r'graph_capture = os\.environ\.get\("' + modes.ENV_GRAPH_CAPTURE + r'", "%d"\)' % modes.GRAPH_ENV_DEFAULTS[modes.ENV_GRAPH_CAPTURE])   # the capture policy: on unless a line turns it off
        self.assertIn("if _capture_off(where):", src)                                                     # ... decided first in _over_graph_budget, for every site (the roll-out asks the sampler site)
        self.assertNotIn(modes.ENV_GRAPH_CAPTURE, modes.SWITCHES)
        self.assertNotIn(modes.ENV_GRAPH_BUDGET, modes.SWITCHES); self.assertNotIn(modes.ENV_GRAPH_LRU_SAMPLER, modes.SWITCHES)     # caller values reach the driver

    def test_report_carries_the_field_and_activation_exports_the_default(self):
        import inspect
        self.assertIn('"effective": effective_graph_settings(res)', inspect.getsource(stack._resolution_fields))
        body = inspect.getsource(stack._activate_body)
        self.assertIn("exports = activation_exports(res)", body); self.assertIn("os.environ.update(exports)", body)
        self.assertNotIn("os.environ.update(res.overrides)", body)


class TestFullModelGraphBudget(unittest.TestCase):
    """K.21: the Full model under exact / fast gets the per-shape graph budget 800 as a PACKAGE default (exported at activation when nobody sets it);
    the Fast model, off and big are untouched; the caller's own EF2_GRAPH_BUDGET_TOKENS (0 included) always wins."""
    def test_full_exact_fast_default_800_and_nothing_else(self):
        R = lambda mode, sv, ov=None: types.SimpleNamespace(mode=mode, server_variant=sv, overrides=dict(ov or {}))
        for mode in ("exact", "fast"):
            eff = stack.effective_graph_settings(R(mode, "full"), environ={})
            self.assertEqual((eff["graph_budget_tokens"], eff["graph_budget_tokens_source"]), (800, "package"))
            self.assertEqual(stack.package_graph_exports(R(mode, "full"), environ={}), {modes.ENV_GRAPH_BUDGET: "800", modes.ENV_GRAPH_LRU_SAMPLER: "2", modes.ENV_GRAPH_BUDGET_SAMPLER: "1536"})
            self.assertEqual([eff[k + "_effective"] for k in stack.SITE_KEYS], [800, 800, 1536])               # the Full model: trunk / encoder inherit 800, the sampler site captured up to 1536
            eff = stack.effective_graph_settings(R(mode, "fast"), environ={})                                  # the Fast model: no per-shape budget; the trunk / encoder SITES at 800, the sampler site at 1536
            self.assertEqual((eff["graph_budget_tokens"], eff["graph_budget_tokens_source"]), (0, "default"))
            self.assertEqual([eff[k + "_effective"] for k in stack.SITE_KEYS], [800, 800, 1536])
            self.assertEqual(stack.package_graph_exports(R(mode, "fast"), environ={}),
                             {modes.ENV_GRAPH_LRU_SAMPLER: "2", modes.ENV_GRAPH_BUDGET_TRUNK: "800", modes.ENV_GRAPH_BUDGET_ENCODER: "800", modes.ENV_GRAPH_BUDGET_SAMPLER: "1536"})
            capped = stack.effective_graph_settings(R(mode, "fast"), environ={modes.ENV_GRAPH_BUDGET: "1300"})   # a caller's per-shape budget governs every site of the Fast model (the site defaults yield)
            self.assertEqual([capped[k + "_effective"] for k in stack.SITE_KEYS], [1300, 1300, 1300])
        for sv in ("full", "fast"):                                                                              # big: NO CUDA graph at any site (the memory line's rule) — the line's own word, no budget word, no package site default
            eff = stack.effective_graph_settings(R("big", sv, modes.BIG_FAST.overrides), environ={})
            self.assertEqual((eff["graph_capture"], eff["graph_capture_source"]), (0, "mode"))
            self.assertEqual((eff["graph_budget_tokens"], eff["graph_budget_tokens_source"]), (0, "default"))
            self.assertEqual(stack.package_graph_exports(R("big", sv, modes.BIG_FAST.overrides), environ={}), {modes.ENV_GRAPH_LRU_SAMPLER: "2"})   # nothing but the every-mode LRU word
            self.assertEqual(stack.activation_exports(R("big", sv, modes.BIG_FAST.overrides), environ={}),
                             {modes.ENV_GRAPH_CAPTURE: "0", modes.ENV_W4_IDPROBE: "0", modes.ENV_GRAPH_LRU_SAMPLER: "2"})
        for mode in ("exact", "fast"):                                                                           # the graph lines: capture on by the driver's default (the budgets decide)
            self.assertEqual(stack.effective_graph_settings(R(mode, "full"), environ={})["graph_capture"], 1)
        off = stack.effective_graph_settings(R("off", "full"), environ={})
        self.assertEqual((off["graph_budget_tokens"], off["graph_budget_tokens_source"]), (0, "default"))

    def test_the_callers_value_wins_zero_included(self):
        R = types.SimpleNamespace(mode="fast", server_variant="full", overrides={})
        for raw, want in (("900", 900), ("0", 0), ("4000", 4000)):
            eff = stack.effective_graph_settings(R, environ={modes.ENV_GRAPH_BUDGET: raw})
            self.assertEqual((eff["graph_budget_tokens"], eff["graph_budget_tokens_source"]), (want, "env"))
            self.assertNotIn(modes.ENV_GRAPH_BUDGET, stack.package_graph_exports(R, environ={modes.ENV_GRAPH_BUDGET: raw}))

    def test_w4_probe_is_gated_by_the_same_budget(self):
        src = open(os.path.join(os.path.dirname(DRIVER), "ef2_w4.py")).read() if os.path.isfile(DRIVER) else None
        if src is None:
            self.skipTest("carried driver not beside the package")
        self.assertIn('budget=int(os.environ.get("%s", "0") or 0)' % modes.ENV_GRAPH_BUDGET, src)          # the probe reads the same switch
        self.assertIn('if _IDPROBE["budget"] > 0 and tokens > _IDPROBE["budget"]:', src)                     # and skips by name above it (a recorded, printed event)
        self.assertIn("identity probe {name} skipped on the first served call", src)
        self.assertIn("tokens=pair.shape[1]", src); self.assertIn("tokens=x.shape[-2]", src)                 # the gate decides on the pair extent L the callers pass (behaviour: test_w4_probe_budget.py)


class TestActiveLineWord(unittest.TestCase):
    def test_active_and_dry_run_lines_end_with_the_effective_sampler_budget(self):
        from esmfold2_opt import report
        eff = stack.effective_graph_settings(types.SimpleNamespace(overrides={}), environ={})
        rep = {"active": True, "mode": "exact", "variant": "fast", "server_line": "opt7x", "upstream": {}, "gpu": {"name": "H100", "sm": "sm90"},
               "levers_planned": ["fused"], "levers_fallback": [], "applied": "deferred", "effective": eff}
        self.assertTrue(report.activation_line(rep).endswith(" applied=deferred compile=n/a graph_lru_sampler=2 graph_budgets=0/0/0"), report.activation_line(rep))   # compile=n/a: nothing in stock ESMFold2 or this kit compiles (nothing here compiles)
        dry = dict(rep, active=False, dry_run=True, server_mode="opt7x", effective=dict(eff, graph_lru_sampler=1, graph_budget_tokens_trunk_effective=800, graph_budget_tokens_encoder_effective=800))
        self.assertTrue(report.activation_line(dry).endswith(" compile=n/a graph_lru_sampler=1 graph_budgets=800/800/0"), report.activation_line(dry))   # trunk/encoder/sampler as ef2_opt's sites will run
        self.assertNotIn("graph_lru_sampler", report.activation_line({k: v for k, v in rep.items() if k != "effective"}))   # a report without the field: no graph word
        self.assertIn(" compile=n/a", report.activation_line({k: v for k, v in rep.items() if k != "effective"}))          # the compile word is a constant of every ACTIVE / DRY-RUN line
        off = dict(rep, mode="big", line="fast", server_line="opt14_msa+EF2_GRAPH_CAPTURE=0,EF2_W4_IDENTITY_PROBE=0", effective=dict(eff, graph_capture=0))
        self.assertTrue(report.activation_line(off).endswith(" applied=deferred compile=n/a graph_lru_sampler=2 graph_capture=off"), report.activation_line(off))   # the memory line: no budget word, capture off
        ab = dict(rep, ablate_tokens=["disto", "dit.gemm=fp32"])                                            # the ablation variable set: ablate=<tokens> right after applied= (never under a mode's plain name)
        self.assertIn(" applied=deferred ablate=disto,dit.gemm=fp32 compile=n/a graph_lru_sampler=2", report.activation_line(ab))
        self.assertEqual(report.graph_words({"effective": eff}), " compile=n/a graph_lru_sampler=2 graph_budgets=0/0/0")
        self.assertEqual(report.graph_words({}), " compile=n/a")


if __name__ == "__main__":
    unittest.main()
