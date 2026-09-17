"""The kernel's per-call fallbacks are a named, gated event: the census at exit (manifest.fallback_census: calls, fallbacks, share against
registry.FALLBACK_MAX_SHARE and the documented FALLBACK_CLASSES), the `fallback_excess` state (recorded, named on ONE line, never an exit code:
those calls ran the stock operation; no kernel call at all is `kernel_not_engaged`, rc 3: a mode is all of its levers), and the activation's patch-marker gate (enabled=True with no patched
Attention class is a refusal by name, not an ACTIVE line). Stubs only: no jax, no GPU."""
import io
import json
import os
import sys
import tempfile
import types
import unittest
import unittest.mock
from contextlib import redirect_stderr

import colabfold_opt
from colabfold_opt import manifest, modes, registry, report, stack
from colabfold_opt.tests import _stubs
from colabfold_opt.tests.test_activation import Base

ACTIVE_REP = {"active": True, "mode": "fast", "levers": ["AF_PALLAS_ATTN"], "levers_applied": ["AF_PALLAS_ATTN"], "reason": None}


class TestCensus(unittest.TestCase):
    def test_share_against_the_documented_class(self):
        c = manifest.fallback_census({"enabled": True, "calls": 492, "fallbacks": 132})       # a b800 run: 132 / 624 = 0.2115, within
        self.assertEqual((c["calls"], c["fallbacks"], c["within"], c["max_share"]), (492, 132, True, registry.FALLBACK_MAX_SHARE))
        self.assertAlmostEqual(c["share"], 132 / 624); self.assertEqual(c["classes"], list(registry.FALLBACK_CLASSES)); self.assertEqual(len(c["classes"]), 2)
        self.assertIs(manifest.fallback_census({"enabled": True, "calls": 10, "fallbacks": 10})["within"], False)     # 0.5 > 0.30: beyond the class
        edge = manifest.fallback_census({"enabled": True, "calls": 7, "fallbacks": 3})                                  # 0.30 exactly: within (<=)
        self.assertEqual((round(edge["share"], 2), edge["within"]), (0.3, True))
        none = manifest.fallback_census({"enabled": True, "calls": 0, "fallbacks": 0})                                  # nothing counted: undecided, not excess
        self.assertEqual((none["share"], none["within"]), (None, None))
        self.assertIsNone(manifest.fallback_census(None)); self.assertIsNone(manifest.fallback_census({"enabled": True}))

    def test_excess_needs_an_active_report_and_a_share_beyond_the_class(self):
        beyond, within = {"enabled": True, "calls": 10, "fallbacks": 10}, {"enabled": True, "calls": 492, "fallbacks": 132}
        self.assertEqual(manifest.excess_fallbacks({"activation_report": ACTIVE_REP, "kit_state_exit": beyond}), ["AF_PALLAS_ATTN"])
        self.assertEqual(manifest.excess_fallbacks({"activation_report": ACTIVE_REP, "kit_state_exit": within}), [])
        self.assertEqual(manifest.excess_fallbacks({"activation_report": {**ACTIVE_REP, "active": False}, "kit_state_exit": beyond}), [])   # a hold / refusal is its own state
        self.assertEqual(manifest.excess_fallbacks({"activation_report": ACTIVE_REP, "kit_state_exit": None}), [])
        d = manifest.fallback_detail(["AF_PALLAS_ATTN"], manifest.fallback_census(beyond))
        self.assertTrue(d.startswith("AF_PALLAS_ATTN: EXIT calls=10 fallbacks=10 share=0.500 above 0.3 ("), d)
        for cls in registry.FALLBACK_CLASSES:
            self.assertIn(cls, d)


class TestVerdict(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.res = os.path.join(self.tmp, "run"); os.makedirs(self.res)
        self.saved = os.environ.pop(manifest.ENV_LAUNCH_ID, None)

    def tearDown(self):
        if self.saved is not None:
            os.environ[manifest.ENV_LAUNCH_ID] = self.saved

    def write(self, state):
        manifest.write(self.res, dict(ACTIVE_REP))
        manifest.record_exit(self.res, state)
        return manifest.read(self.res)

    def test_record_exit_writes_the_census_and_the_excess(self):
        man = self.write({"enabled": True, "calls": 10, "fallbacks": 10})
        self.assertEqual((man["fallback_census"]["share"], man["fallback_census"]["within"], man["fallback_excess"], man["partial"]), (0.5, False, ["AF_PALLAS_ATTN"], []))
        man = self.write({"enabled": True, "calls": 492, "fallbacks": 132})
        self.assertEqual((man["fallback_census"]["within"], man["fallback_excess"], man["partial"]), (True, [], []))
        man = self.write({"enabled": True, "calls": 0, "fallbacks": 5})                                                 # no kernel call: kernel_not_engaged names it, not fallback_excess
        self.assertEqual((man["partial"], man["fallback_excess"]), (["AF_PALLAS_ATTN"], ["AF_PALLAS_ATTN"]))

    def test_excess_is_named_and_never_an_exit_code(self):
        self.write({"enabled": True, "calls": 10, "fallbacks": 10})
        v = manifest.verdict(self.res, "fast", [], 0)                                                                     # the kernel engaged and fell back beyond the class: the run's own verdict, the state recorded and named
        self.assertEqual((v["ok"], v["rc"], v["reason"], v["fallback_excess"], v["partial"], v["exit_by"]), (True, 0, None, ["AF_PALLAS_ATTN"], [], "verdict"))
        self.assertIn("share=0.500 above 0.3", v["detail"]); self.assertEqual(v["fallback_census"]["fallbacks"], 10)
        self.assertEqual(report.fallback_excess_line(v["detail"]),
                         "[colabfold-opt] PARTIAL fallback_excess: " + v["detail"] + "; the calls that fell back ran the stock operation — recorded, exit 0")
        v = manifest.verdict(self.res, "fast", [], report.EXIT_NOT_ACTIVE)                                              # a model process that exited 3 did so for a refusal of its own, named on its line
        self.assertEqual((v["rc"], v["reason"], v["exit_by"]), (report.EXIT_NOT_ACTIVE, "not_active", "model_process"))
        self.write({"enabled": True, "calls": 0, "fallbacks": 10})                                                       # no kernel call at all: kernel_not_engaged — rc 3 by name, --allow-partial the escape
        v = manifest.verdict(self.res, "fast", [], 0)
        self.assertEqual((v["ok"], v["rc"], v["reason"]), (False, report.EXIT_NOT_ACTIVE, "kernel_not_engaged"))
        self.assertTrue(report.partial_exit_line(v["detail"]).endswith("(the kernel ran no attention call); exit 3"), report.partial_exit_line(v["detail"]))
        self.assertNotIn("allow_partial", v)
        self.write({"enabled": True, "calls": 492, "fallbacks": 132})                                                    # within the class: PASS, nothing to name
        v = manifest.verdict(self.res, "fast", [], 0)
        self.assertEqual((v["ok"], v["rc"], v["fallback_excess"], v["detail"]), (True, 0, [], ""))
        self.assertEqual(manifest.verdict(self.res, "off", [], 0)["fallback_excess"], [])                               # the stock route carries no census


MSA_REP = {"active": True, "mode": "fast", "levers": ["AF_PALLAS_ATTN", "PALLAS_MSA"], "levers_applied": ["AF_PALLAS_ATTN", "PALLAS_MSA"], "reason": None}


class TestSteppedAsideByRule(unittest.TestCase):
    """A lever whose EVERY call stepped aside by a named size rule (PALLAS_MSA on a 20-residue single sequence: `calls=0 fallbacks=16
    fallback_by=below_keys_rule:16`, msa_attn.MIN_KEYS) ran the stock operation by design at that size — its LEVER line says so; that is not a
    partial activation: `pred`'s verdict is rc 0. `calls=0 fallbacks=0`, or fallbacks without a sanctioned rule, stay PARTIAL (rc 3)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(); self.res = os.path.join(self.tmp, "run"); os.makedirs(self.res)
        self.saved = os.environ.pop(manifest.ENV_LAUNCH_ID, None)

    def tearDown(self):
        if self.saved is not None:
            os.environ[manifest.ENV_LAUNCH_ID] = self.saved

    def write(self, msa_state, kit_state=None):
        kit_state = kit_state or {"enabled": True, "calls": 20, "fallbacks": 4}
        manifest.write(self.res, dict(MSA_REP))
        manifest.record_exit(self.res, kit_state, {"AF_PALLAS_ATTN": kit_state, "PALLAS_MSA": msa_state})
        return manifest.read(self.res)

    def test_every_call_below_the_key_floor_is_by_design_rc_0(self):
        st = {"enabled": True, "calls": 0, "fallbacks": 16, "fallback_by": "below_keys_rule:16", "shapes": "none", "min_keys": 128}
        self.assertEqual(manifest.stepped_aside_rule(st), "below_keys_rule")
        man = self.write(st)
        self.assertEqual((man["partial"], man["stepped_aside"]), ([], {"PALLAS_MSA": "below_keys_rule"}))
        v = manifest.verdict(self.res, "fast", [], 0)
        self.assertEqual((v["ok"], v["rc"], v["reason"], v["partial"], v["stepped_aside"]), (True, 0, None, [], {"PALLAS_MSA": "below_keys_rule"}))
        err = io.StringIO()
        with redirect_stderr(err):                                                                                          # the strict hook's own exit reading: no PARTIAL line, no exit
            with unittest.mock.patch.object(stack, "_STATE", {**stack._STATE, "report": dict(MSA_REP), "manifest_dir": self.res}), \
                 unittest.mock.patch.object(stack, "lever_states", lambda: {"AF_PALLAS_ATTN": {"enabled": True, "calls": 20, "fallbacks": 4}, "PALLAS_MSA": st}), \
                 unittest.mock.patch.object(stack, "kit_state", lambda: {"enabled": True, "calls": 20, "fallbacks": 4}):
                stack.post_run_partial(strict=True)
        self.assertNotIn("partial activation", err.getvalue())
        both = {"enabled": True, "calls": 0, "fallbacks": 6, "fallback_by": "below_keys_rule:4,below_size_rule:2"}                # two sanctioned rules: still by design
        self.assertEqual(manifest.stepped_aside_rule(both), "below_keys_rule,below_size_rule")
        self.assertEqual(registry.STEP_ASIDE_RULES, ("below_keys_rule", "below_size_rule", "cell", "monomer_route", "template_disabled", "dropout_enabled", "no_cell_family", "stock_by_name", "cell_stock", "no_cell"))
        cell = {"enabled": True, "calls": 0, "fallbacks": 108, "fallback_by": "cell:108"}                                             # TRIATTN_XLA: the provider's cell named another row for every call — by measurement, by name
        self.assertEqual(manifest.stepped_aside_rule(cell), "cell")
        self.assertIsNone(manifest.stepped_aside_rule({"enabled": True, "calls": 0, "fallbacks": 3, "fallback_by": "refused:3"}))     # the bridge refused every call at run time: stays partial
        tw = {"enabled": True, "calls": 0, "fallbacks": 8, "fallback_by": "stock_by_name:8"}                                          # TRIMUL_PALLAS on a card / jax line with no measured cell: the provider's tier word named the stock statement for every class — by design
        self.assertEqual(manifest.stepped_aside_rule(tw), "stock_by_name")

    def test_no_call_and_no_named_rule_stays_partial_rc_3(self):
        for st in ({"enabled": True, "calls": 0, "fallbacks": 0, "fallback_by": "none", "min_keys": 128},                   # nothing counted at all: PARTIAL
                   {"enabled": True, "calls": 0, "fallbacks": 3, "fallback_by": "kernel_unavailable:3"},                     # a fallback that is not a size rule: PARTIAL
                   {"enabled": True, "calls": 0, "fallbacks": 5, "fallback_by": "below_keys_rule:3,bias_form:2"},            # one call outside the sanctioned rules: PARTIAL
                   {"enabled": True, "calls": 0, "fallbacks": 5, "fallback_by": "below_keys_rule:3"},                        # counts that do not add up: PARTIAL
                   {"enabled": True, "calls": 0, "fallbacks": 5}):                                                          # no census at all (the carried kernel's shape): PARTIAL
            with self.subTest(st=st):
                self.assertIsNone(manifest.stepped_aside_rule(st))
                man = self.write(st)
                self.assertEqual((man["partial"], man["stepped_aside"]), (["PALLAS_MSA"], {}))
                v = manifest.verdict(self.res, "fast", [], 0)
                self.assertEqual((v["ok"], v["rc"], v["reason"], v["partial"]), (False, report.EXIT_NOT_ACTIVE, "kernel_not_engaged", ["PALLAS_MSA"]))
                self.assertIn("PALLAS_MSA: EXIT calls=0 (no model call went through the lever)", v["detail"])
        served = {"enabled": True, "calls": 12, "fallbacks": 4, "fallback_by": "below_keys_rule:4"}                          # served calls: the ordinary pass, nothing stepped aside wholesale
        self.assertIsNone(manifest.stepped_aside_rule(served)); self.assertEqual(self.write(served)["partial"], [])


class TestStrictHookExit(unittest.TestCase):
    """stack.post_run_partial under the strict hook: an ACTIVE process whose exit counters fall back beyond the class prints the ONE
    `PARTIAL fallback_excess:` line (standalone; `pred` prints it from its verdict) and never exits for it; no kernel call at all is a PARTIAL
    activation — the family line and exit 3 (a mode is all of its levers)."""
    def setUp(self):
        stack.reset_for_tests()
        self.saved_mod = sys.modules.get(modes.KIT_MODULE)
        km = types.ModuleType(modes.KIT_MODULE); km._STATE = {"enabled": True, "calls": 10, "fallbacks": 10}
        sys.modules[modes.KIT_MODULE] = km; self.km = km
        stack._STATE["report"] = dict(ACTIVE_REP)
        self.env = {k: os.environ.pop(k, None) for k in (manifest.ENV_LAUNCH_ID,)}

    def tearDown(self):
        sys.modules.pop(modes.KIT_MODULE, None)
        if self.saved_mod is not None:
            sys.modules[modes.KIT_MODULE] = self.saved_mod
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        stack.reset_for_tests()

    def test_excess_prints_one_line_and_returns(self):
        err = io.StringIO()
        with redirect_stderr(err):
            stack.post_run_partial(strict=True)                                                                            # returns: the run's own exit
        self.assertIn("[colabfold-opt] PARTIAL fallback_excess: AF_PALLAS_ATTN: EXIT calls=10 fallbacks=10 share=0.500 above 0.3 (", err.getvalue())
        self.assertIn("; the calls that fell back ran the stock operation — recorded, exit 0\n", err.getvalue())
        self.assertNotIn("NOT ACTIVE", err.getvalue())
        os.environ[manifest.ENV_LAUNCH_ID] = "L1"                                                                         # launched by pred: pred prints the line from its verdict, the model process stays quiet
        err = io.StringIO()
        with redirect_stderr(err):
            stack.post_run_partial(strict=True)
        self.assertEqual(err.getvalue(), "")

    def test_no_kernel_call_exits_3_with_the_family_line(self):
        self.km._STATE.update(calls=0, fallbacks=5)                                                                        # the optimization did not run: kernel_not_engaged
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            stack.post_run_partial(strict=True)
        self.assertEqual(cm.exception.code, report.EXIT_NOT_ACTIVE)
        self.assertIn("[colabfold-opt] NOT ACTIVE: partial activation — AF_PALLAS_ATTN: EXIT calls=0 ", err.getvalue())
        self.assertIn("(the kernel ran no attention call); exit 3\n", err.getvalue())
        os.environ[manifest.ENV_LAUNCH_ID] = "L1"                                                                         # launched by pred: the model process exits 3 all the same (pred reads rc 3 + `partial`)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            stack.post_run_partial(strict=True)
        self.assertEqual(cm.exception.code, report.EXIT_NOT_ACTIVE)
        stack.post_run_partial(strict=False)                                                                               # the explicit non-strict route: no exit, the manifest's `partial` is the record

    def test_within_the_class_is_silent(self):
        self.km._STATE.update(calls=492, fallbacks=132)
        err = io.StringIO()
        with redirect_stderr(err):
            stack.post_run_partial(strict=True)
        self.assertEqual(err.getvalue(), "")


class TestPatchMarkerGate(Base):
    """enable() that reports enabled=True without patching alphafold.model.modules.Attention (the module absent at enable time) is refused by
    name at activation — before the run — instead of an ACTIVE line followed by a stock run."""
    def test_enabled_without_a_patched_class_is_not_active(self):
        import importlib
        kp = stack.kit_pythonpath()
        if kp not in sys.path:
            sys.path.insert(0, kp)
        km = importlib.import_module(modes.KIT_MODULE)                                                                    # the stub kit's enable() made to patch nothing and still say enabled=True
        km.enable = lambda *a, **k: km._STATE.__setitem__("enabled", True)
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600, 705))
        self.assertFalse(rep["active"]); self.assertEqual((rep["levers_applied"], rep["levers_unavailable"]), ([], ["DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"]))
        self.assertIn("carries no _pallas_patched marker: the patch did not land", rep["reason"])
        self.assertIn("[colabfold-opt] NOT ACTIVE: AF_PALLAS_ATTN: enable() returned enabled=True but alphafold.model.modules.Attention carries no _pallas_patched marker", err.getvalue())
        self.assertNotIn("ACTIVE mode=fast", err.getvalue().replace("NOT ACTIVE", ""))
        stack.reset_for_tests(); sys.modules[modes.KIT_MODULE]._STATE["enabled"] = False
        with redirect_stderr(io.StringIO()), self.assertRaises(stack.ActivationError):
            stack.activate("fast", queries=_stubs.queries(600, 705), strict=True)

    def test_the_patched_class_passes(self):
        with redirect_stderr(io.StringIO()):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600, 705))
        self.assertTrue(rep["active"]); self.assertTrue(rep["patch_marker"])


if __name__ == "__main__":
    unittest.main()


class TestZeroTrafficIsPartial(unittest.TestCase):
    """0.2.15: a lever with zero counted traffic at exit (calls=0 and no rule-attributed fallback, no superseding lever's calls) is a PARTIAL
    activation whatever its class marker reads — the 0.2.14 `route_absent` clause is withdrawn (no by-design instance on either AlphaFold route:
    every route lever is always counted, as calls or as by-rule fallbacks)."""

    def test_the_five_class_levers_at_zero_traffic_are_partial(self):
        from colabfold_opt import manifest
        applied = ["DEVICE_RESIDENT", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"]
        zero = {"calls": 0, "fallbacks": 0, "fallback_by": "none"}
        man = {"activation_report": {"active": True, "mode": "fast", "levers": applied, "levers_applied": applied},
               "kit_state_exit": {"enabled": True, "calls": 20, "fallbacks": 8},
               "lever_states_exit": {"DEVICE_RESIDENT": {"calls": 20}, "TRIMUL_PALLAS": dict(zero), "PALLAS_MSA": dict(zero), "TRIATTN_XLA": dict(zero),
                                     "MSA_COL_CUDNN": dict(zero), "TEMPL_DEDUP": dict(zero), "TRANSITION": dict(zero)},
               "markers_exit": {n: True for n in ("TRIMUL_PALLAS", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION")}}   # a bound marker does not excuse zero traffic
        self.assertEqual(manifest.partial_levers(man), ["TRIMUL_PALLAS", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"])
        rule = {"calls": 0, "fallbacks": 8, "fallback_by": "below_keys_rule:8"}                                             # the ptm route's real reading: by the size rule, not partial
        man["lever_states_exit"].update(PALLAS_MSA=rule, MSA_COL_CUDNN={"calls": 0, "fallbacks": 4, "fallback_by": "below_keys_rule:4"},
                                        TRIATTN_XLA={"calls": 0, "fallbacks": 20, "fallback_by": "below_size_rule:20"}, TRIMUL_PALLAS={"calls": 8, "fallbacks": 0, "fallback_by": "none"},
                                        TEMPL_DEDUP={"calls": 0, "predicts": 5, "fallbacks": 5, "fallback_by": "monomer_route:5"},
                                        TRANSITION={"calls": 0, "fallbacks": 48, "fallback_by": "cell_stock:44,no_cell:4"})                     # the provider's word named the stock statement for every class: by design
        self.assertEqual(manifest.partial_levers(man), [])
        self.assertFalse(hasattr(manifest, "route_absent"))
