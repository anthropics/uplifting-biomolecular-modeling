"""The ablation switch MODEL_OPT_LEVERS_OFF (ablation.py): its grammar on this kit's lever names, its refusals by name, the `ablated=` token
on the ACTIVE / DRY-RUN lines, the ablated levers' LEVER lines at exit (`state=off reason=ablated`), the exports an ablated lever loses, the
exit census that expects only the kept levers, and the routes (activation, dry run, command line, the `.pth` finder, the stock child's
environment). CPU only: the activation stubs of test_activation (no jax, no GPU, no colabfold)."""
import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import colabfold_opt
from colabfold_opt import _autoload, ablation, cli, modes, registry, report, stack, stock_pred
from colabfold_opt.tests import _stubs
from colabfold_opt.tests.test_activation import Base, TestExitTally

ENV = ablation.ENV
FAST = list(modes.TABLE["fast"][0])                      # DEVICE_RESIDENT, SUBBATCH, TRIMUL_PALLAS, AF_PALLAS_ATTN, PALLAS_MSA, TRIATTN_XLA (0.2.7)


def _enable(mode, **kw):
    err = io.StringIO()
    with redirect_stderr(err):
        rep = colabfold_opt.enable(mode, queries=_stubs.queries(600), **kw)
    return rep, err.getvalue()


def _check(mode, **kw):
    err = io.StringIO()
    with redirect_stderr(err):
        rep = stack.check(mode, **kw)
    return rep, err.getvalue()


class TestGrammar(unittest.TestCase):
    def test_requested_parsing(self):
        self.assertEqual(ablation.requested({}), [])
        self.assertEqual(ablation.requested({ENV: ""}), [])
        self.assertEqual(ablation.requested({ENV: " , ,"}), [])
        self.assertEqual(ablation.requested({ENV: "SUBBATCH"}), ["SUBBATCH"])
        self.assertEqual(ablation.requested({ENV: " SUBBATCH , TRIMUL_PALLAS,SUBBATCH ,"}), ["SUBBATCH", "TRIMUL_PALLAS"])   # whitespace ignored, duplicates folded, order kept

    def test_names_spelled_once(self):
        self.assertEqual((ENV, _autoload.LEVERS_OFF_ENV), ("MODEL_OPT_LEVERS_OFF", "MODEL_OPT_LEVERS_OFF"))   # the tree's one name; _autoload spells it again (imports nothing at start)
        self.assertEqual((ablation.REASON, ablation.TOKEN, report.ABLATED), ("ablated",) * 3)
        self.assertEqual(ablation.token([]), ""); self.assertEqual(ablation.token(["SUBBATCH", "PALLAS_MSA"]), " ablated=SUBBATCH,PALLAS_MSA")

    def test_validate_accepts_mode_levers_and_deployment_levers(self):
        self.assertEqual(ablation.validate("fast", ["SUBBATCH"], FAST), ["SUBBATCH"])
        self.assertEqual(ablation.validate("fast", ["TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "SUBBATCH"], FAST), ["TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "SUBBATCH", "TRIATTN_XLA"])   # TRIATTN_XLA rides AF_PALLAS_ATTN's binding: ablated with it, by name
        self.assertEqual(ablation.validate("fast", ["TRIATTN_XLA"], FAST), ["TRIATTN_XLA"])                     # alone: the binding stays, narrowed to the provider's other rows
        self.assertEqual(ablation.validate("fast", ["XLA_CACHE"], FAST), ["XLA_CACHE"])                     # a deployment lever: placed by the activation, so removable (registry.DEPLOYMENT)
        self.assertEqual(ablation.validate("exact", ["XLA_CACHE"], modes.TABLE["exact"][0]), ["XLA_CACHE"])
        self.assertIn("XLA_CACHE", registry.DEPLOYMENT)

    def test_validate_refusals_by_name(self):
        cases = [
            ("off", ["SUBBATCH"], (), 1, "mode off applies no lever, there is nothing to ablate"),
            ("fast", ["bogus"], FAST, 1, "bogus: not a lever of this kit (registry levers: "),
            ("fast", ["subbatch"], FAST, 1, "subbatch: not a lever of this kit"),                          # matched as spelled: the LEVER lines' upper-case names
            ("exact", ["SUBBATCH"], modes.TABLE["exact"][0], 1, "SUBBATCH: not in mode exact's lever set (DEVICE_RESIDENT; deployment levers XLA_CACHE)"),
            ("fast", ["ROWPAIR"], FAST, 1, "ROWPAIR: not in mode fast's lever set"),
            ("fast", ["AF_PALLAS_ATTN_ALL"], FAST, 1, "AF_PALLAS_ATTN_ALL: not in mode fast's lever set"),  # a carried switch the kit reads but no mode wires
            ("exact", ["DEVICE_RESIDENT"], modes.TABLE["exact"][0], 1, "the request removes every lever of mode exact (DEVICE_RESIDENT): that is --mode off, not an ablation"),
            ("fast", list(FAST), FAST, 1, "the request removes every lever of mode fast"),
            ("big", ["ROWPAIR"], FAST, 1, "ROWPAIR: the --n_gpu axis of mode big, not removable by MODEL_OPT_LEVERS_OFF (at --n_gpu 1 it installs nothing"),
            ("big", ["ROWPAIR"], [l for l in modes.TABLE["big"][0] if l != "TRIMUL_PALLAS"], 4, "(at --n_gpu 4 it IS the arm; --n_gpu 1 runs the unsharded lever set)"),
            ("big", ["TRIMUL_PALLAS"], [l for l in modes.TABLE["big"][0] if l != "TRIMUL_PALLAS"], 2, "TRIMUL_PALLAS: not applied by mode big at --n_gpu 2 (ROWPAIR owns triangle multiplication at --n_gpu P > 1), nothing to ablate"),
        ]
        for mode, names, levers, n_gpu, text in cases:
            with self.subTest(mode=mode, names=names, n_gpu=n_gpu):
                with self.assertRaises(ablation.AblationError) as cm:
                    ablation.validate(mode, names, levers, n_gpu)
                self.assertIn(text, str(cm.exception)); self.assertTrue(str(cm.exception).startswith("MODEL_OPT_LEVERS_OFF"), str(cm.exception))
        with self.assertRaises(ablation.AblationError) as cm:                                                  # every problem named at once
            ablation.validate("fast", ["bogus", "ROWPAIR"], FAST)
        self.assertIn("bogus: not a lever of this kit", str(cm.exception)); self.assertIn("ROWPAIR: not in mode fast's lever set", str(cm.exception))

    def test_apply_drops_levers_and_the_carried_switch(self):
        res = modes.resolve("fast")
        self.assertEqual(res["kit_env"], {modes.KIT_SWITCH: "1"})
        out = ablation.apply(res, ["SUBBATCH"])
        self.assertEqual(list(out["levers"]), [l for l in FAST if l != "SUBBATCH"]); self.assertEqual(out["kit_env"], {modes.KIT_SWITCH: "1"})
        out = ablation.apply(res, ["AF_PALLAS_ATTN"])
        self.assertEqual(list(out["levers"]), [l for l in FAST if l != "AF_PALLAS_ATTN"]); self.assertEqual(out["kit_env"], {})   # the switch leaves the exports with its lever
        self.assertEqual(ablation.switches(["AF_PALLAS_ATTN", "SUBBATCH"]), (modes.KIT_SWITCH,)); self.assertEqual(ablation.switches(["SUBBATCH"]), ())
        self.assertEqual(list(res["levers"]), FAST)                                                             # the table's own tuple untouched


class TestActivation(Base):
    """The in-process route on the activation stubs: the mode minus the named levers, everything else exactly the mode's."""

    def setUp(self):
        super().setUp()
        self.env_saved[ENV] = os.environ.pop(ENV, None)

    def test_unset_is_the_mode_byte_for_byte(self):
        rep, err = _enable("fast")
        self.assertTrue(rep["active"], err); self.assertNotIn("ablated=", err); self.assertNotIn("levers_ablated", rep)
        self.assertEqual(rep["levers_applied"], FAST)

    def test_fast_minus_subbatch(self):
        os.environ[ENV] = "SUBBATCH"
        rep, err = _enable("fast")
        self.assertTrue(rep["active"], err)
        self.assertEqual(rep["levers_ablated"], ["SUBBATCH"]); self.assertEqual(rep["levers_applied"], [l for l in FAST if l != "SUBBATCH"])
        self.assertNotIn("subbatch", rep)                                                                        # its module never ran: no sub-batch decision in the report
        self.assertIn("ACTIVE mode=fast levers=DEVICE_RESIDENT,TRIMUL_PALLAS,AF_PALLAS_ATTN,PALLAS_MSA,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION tokens=600-600 ", err)
        self.assertIn(" ablated=SUBBATCH", err); self.assertNotIn("NOT ACTIVE", err)
        self.assertEqual(os.environ.get(modes.KIT_SWITCH), "1")                                                 # the carried lever kept: its switch exported as the mode does

    def test_fast_minus_the_carried_lever_exports_no_switch_and_pops_a_callers(self):
        os.environ[ENV] = "AF_PALLAS_ATTN"
        rep, err = _enable("fast")
        self.assertTrue(rep["active"], err)
        self.assertEqual(rep["levers_applied"], ["DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "PALLAS_MSA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"]); self.assertEqual(rep["kit_env"], {})
        self.assertNotIn(modes.KIT_SWITCH, os.environ)                                                          # never exported …
        self.assertIsNone(self.kit_module())                                                                     # … and the carried adapter never imported: PALLAS_MSA wraps the stock class
        self.assertIn(" ablated=AF_PALLAS_ATTN,TRIATTN_XLA", err)                                                # the bridge row rides the binding: ablated with it, by name

    def test_a_callers_export_of_an_ablated_switch_is_unset(self):
        os.environ[ENV] = "AF_PALLAS_ATTN,PALLAS_MSA"
        os.environ[modes.KIT_SWITCH] = "1"                                                                       # someone's export — the ablation removes it before any lever runs (the adapter is never imported, so no late-activation question arises)
        rep, err = _enable("fast")
        self.assertTrue(rep["active"], err); self.assertNotIn(modes.KIT_SWITCH, os.environ)
        self.assertEqual(rep["levers_applied"], ["DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"]); self.assertIn(" ablated=AF_PALLAS_ATTN,PALLAS_MSA,TRIATTN_XLA", err)

    def test_deployment_lever_ablated(self):
        os.environ[ENV] = "XLA_CACHE"
        with mock.patch.dict(os.environ, {"COLABFOLD_OPT_JIT_ROOT": os.path.join(self.tmp, "jit")}):
            rep, err = _enable("fast")
        self.assertTrue(rep["active"], err); self.assertEqual(rep["levers_applied"], FAST)                      # the mode's levers all on: a deployment lever is outside the tuple
        self.assertEqual((rep["xla_cache"]["state"], rep["xla_cache"]["reason"]), ("off", "ablated")); self.assertFalse(os.path.exists(os.path.join(self.tmp, "jit")))
        self.assertIn(" ablated=XLA_CACHE", err)

    def test_refused_by_name_in_process(self):
        for mode, val, text in (("fast", "bogus", "bogus: not a lever of this kit"), ("fast", "ROWPAIR", "ROWPAIR: not in mode fast's lever set"),
                                ("exact", "DEVICE_RESIDENT", "that is --mode off, not an ablation"), ("off", "SUBBATCH", "mode off applies no lever")):
            with self.subTest(mode=mode, val=val):
                stack.reset_for_tests(); os.environ[ENV] = val; os.environ.pop(modes.KIT_SWITCH, None)
                rep, err = _enable(mode)
                self.assertFalse(rep.get("active"), err); self.assertIn("[colabfold-opt] NOT ACTIVE: MODEL_OPT_LEVERS_OFF", err); self.assertIn(text, err)
                self.assertNotIn("ACTIVE mode=", err.replace("NOT ACTIVE", "")); self.assertNotIn("mode=off (stock: nothing applied)", err)   # never the mode's line under a refused name
                self.assertIsNone(self.kit_module())                                                             # nothing applied
        stack.reset_for_tests(); os.environ[ENV] = "bogus"
        with redirect_stderr(io.StringIO()), self.assertRaises(colabfold_opt.ActivationError):
            colabfold_opt.enable("fast", queries=_stubs.queries(600), strict=True)                               # the strict route (the hook's): the line, then the error → exit 3

    def test_dry_run_names_the_ablation_or_the_refusal(self):
        os.environ[ENV] = "PALLAS_MSA"
        rep, err = _check("fast")
        self.assertEqual(rep["levers_ablated"], ["PALLAS_MSA"]); self.assertIsNone(rep.get("would_refuse"))
        self.assertIn("DRY-RUN mode=fast levers=DEVICE_RESIDENT,SUBBATCH,TRIMUL_PALLAS,AF_PALLAS_ATTN,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION tokens=none ", err); self.assertIn(" ablated=PALLAS_MSA", err)
        self.assertIsNone(self.kit_module()); self.assertNotIn(modes.KIT_SWITCH, os.environ)                     # a dry run touches nothing
        stack.reset_for_tests(); os.environ[ENV] = "nosuch"
        rep, err = _check("fast")
        self.assertIn("MODEL_OPT_LEVERS_OFF refused — nosuch: not a lever of this kit", rep["would_refuse"]); self.assertIn("DRY-RUN mode=fast", err); self.assertIn("would_refuse=", err)

    def test_command_line_routes(self):
        def main(argv):
            err, out = io.StringIO(), io.StringIO()
            with redirect_stderr(err), redirect_stdout(out):
                rc = cli.main(list(argv))
            return rc, err.getvalue() + out.getvalue()
        os.environ[ENV] = "SUBBATCH"
        rc, text = main(["check", "--mode", "off"])                                                              # under off: refused before any gate, rc 3, the house line
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("[colabfold-opt] NOT ACTIVE: MODEL_OPT_LEVERS_OFF=SUBBATCH: mode off applies no lever", text)
        rc, text = main(["pred", "--mode", "off", os.path.join(self.tmp, "absent.a3m"), os.path.join(self.tmp, "o", "pred")])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("NOT ACTIVE: MODEL_OPT_LEVERS_OFF=SUBBATCH", text); self.assertFalse(os.path.exists(os.path.join(self.tmp, "o")))   # nothing launched, nothing made
        rc, text = main(["check", "--mode", "fast"])                                                             # a kit mode: the dry run carries the token, rc 0
        self.assertEqual(rc, 0, text); self.assertIn(" ablated=SUBBATCH", text)
        stack.reset_for_tests(); os.environ[ENV] = "ROWPAIR"
        rc, text = main(["check", "--mode", "fast"])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("would_refuse=", text); self.assertIn("ROWPAIR: not in mode fast's lever set", text)


class TestExitLines(TestExitTally):
    """At interpreter exit (subprocesses): an ablated lever's LEVER line reads `state=off reason=ablated`; the kept levers' lines are the mode's."""

    def test_ablated_levers_read_off_ablated(self):
        r = self.run_py("""
            import colabfold_opt
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
            import af2_pallas_attn as km
            km._STATE["calls"] += 3
            print("active", rep["active"], rep["levers_ablated"], rep["levers_applied"])
            """, env_extra={ENV: "SUBBATCH,PALLAS_MSA,XLA_CACHE"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("active True ['SUBBATCH', 'PALLAS_MSA', 'XLA_CACHE'] ['DEVICE_RESIDENT', 'TRIMUL_PALLAS', 'AF_PALLAS_ATTN', 'TRIATTN_XLA', 'MSA_COL_CUDNN', 'TEMPL_DEDUP', 'TRANSITION']", r.stdout)
        self.assertIn("ACTIVE mode=fast levers=DEVICE_RESIDENT,TRIMUL_PALLAS,AF_PALLAS_ATTN,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION tokens=600-600 ", r.stderr); self.assertIn(" ablated=SUBBATCH,PALLAS_MSA,XLA_CACHE", r.stderr)
        tail = r.stderr.strip().splitlines()[-12:]
        self.assertEqual(tail[0], "[colabfold-opt] EXIT _STATE enabled=True calls=3 fallbacks=0")
        self.assertTrue(tail[1].startswith("[colabfold-opt] LEVER name=DEVICE_RESIDENT state=on "), tail)
        self.assertEqual(tail[2], "[colabfold-opt] LEVER name=SUBBATCH state=off reason=ablated impl=opt_core.jax_design.subbatch_policy origin=core strategy=F7.jax_subbatch")
        self.assertTrue(tail[3].startswith("[colabfold-opt] LEVER name=TRIMUL_PALLAS state=on impl=opt_core.kernels.pallas.serve:triangle_multiplication origin=core "), tail)
        self.assertRegex(tail[4], r"^\[colabfold-opt\] LEVER name=AF_PALLAS_ATTN state=on impl=opt_core.kernels.pallas origin=core strategy=F1.flash_triatt calls=3 fallbacks=0 word=fast provider_calls=0 served=none sites=none cells=none provider=\S+$")
        self.assertEqual(tail[5], "[colabfold-opt] LEVER name=PALLAS_MSA state=off reason=ablated impl=opt_core.kernels.pallas:attention origin=core strategy=F5.flash_attn_dense")
        self.assertTrue(tail[6].startswith("[colabfold-opt] LEVER name=TRIATTN_XLA state=on impl=opt_core.kernels.triattn_xla origin=core strategy=F1.flash_triatt calls=0 fallbacks=0 "), tail)
        self.assertTrue(tail[7].startswith("[colabfold-opt] LEVER name=MSA_COL_CUDNN state=on impl=opt_core.kernels.pallas:attention origin=core strategy=F5.sdpa_cudnn word=fast served=none cells=none calls=0 fallbacks=0 "), tail)
        self.assertTrue(tail[8].startswith("[colabfold-opt] LEVER name=TEMPL_DEDUP state=on impl=colabfold_opt.templ_dedup origin=kit strategy=LOCAL.colabfold.template_dedup calls=0 predicts="), tail)
        self.assertTrue(tail[9].startswith("[colabfold-opt] LEVER name=TRANSITION state=on impl=opt_core.kernels.pallas.serve:transition origin=core strategy=LOCAL.fused_transition calls=0 fallbacks=0 fallback_by=none word=fast "), tail)
        self.assertEqual(tail[10], "[colabfold-opt] LEVER name=ROWPAIR state=off reason=not_in_mode impl=opt_core.mem.rowpair_jax origin=core strategy=F7.tensor_parallel")
        self.assertEqual(tail[11], "[colabfold-opt] LEVER name=XLA_CACHE state=off reason=ablated impl=opt_core.capture.xla_cache origin=core strategy=F3.jit_cache_keyed")

    def test_the_carried_lever_ablated_prints_no_state_line_and_no_partial(self):
        """AF_PALLAS_ATTN ablated: the carried adapter is not in the process's lever set — no `EXIT _STATE` line, no partial-activation census
        against its counters (an ablated lever is not expected to engage); PALLAS_MSA kept engages over the stock class."""
        r = self.run_py("""
            import colabfold_opt
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
            import sys
            print("active", rep["active"], rep["levers_applied"], "af2_pallas_attn" in sys.modules)
            """, env_extra={ENV: "AF_PALLAS_ATTN"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("active True ['DEVICE_RESIDENT', 'SUBBATCH', 'TRIMUL_PALLAS', 'PALLAS_MSA', 'MSA_COL_CUDNN', 'TEMPL_DEDUP', 'TRANSITION'] False", r.stdout)
        self.assertIn("[colabfold-opt] LEVER name=TRIATTN_XLA state=off reason=ablated impl=opt_core.kernels.triattn_xla origin=core strategy=F1.flash_triatt", r.stderr)
        self.assertNotIn("EXIT _STATE", r.stderr); self.assertNotIn("partial activation", r.stderr)
        self.assertIn("[colabfold-opt] LEVER name=AF_PALLAS_ATTN state=off reason=ablated impl=opt_core.kernels.pallas origin=core strategy=F1.flash_triatt", r.stderr)
        self.assertIn("[colabfold-opt] LEVER name=PALLAS_MSA state=on ", r.stderr)


class TestRoutes(unittest.TestCase):
    def test_finder_installed_for_the_switch_under_off_or_unset(self):
        self.assertIsNone(_autoload.install({}))
        self.assertIsNone(_autoload.install({"COLABFOLD_OPT": "off"}))
        self.assertIsNone(_autoload.install({"COLABFOLD_OPT": "off", ENV: " , "}))                              # blank = no ablation
        saved = list(__import__("sys").meta_path)
        try:
            for env in ({ENV: "SUBBATCH"}, {"COLABFOLD_OPT": "off", ENV: "SUBBATCH"}):
                __import__("sys").meta_path[:] = [f for f in saved if not isinstance(f, _autoload.Finder)]
                f = _autoload.install(env)
                self.assertIsInstance(f, _autoload.Finder); self.assertEqual(f.levers_off, "SUBBATCH")          # the trigger import will refuse it by name (nothing to ablate)
            __import__("sys").meta_path[:] = [f for f in saved if not isinstance(f, _autoload.Finder)]
            f = _autoload.install({"COLABFOLD_OPT": "fast", ENV: "SUBBATCH"})
            self.assertIsInstance(f, _autoload.Finder); self.assertEqual(f.levers_off, "")                      # a kit mode: the activation validates the names, not the finder
        finally:
            __import__("sys").meta_path[:] = saved

    def test_the_stock_child_never_carries_the_switch(self):
        env = stock_pred.stock_env({"PATH": "/bin", ENV: "SUBBATCH", "COLABFOLD_OPT": "off", "MODEL_OPT_TARGET_GPU": "H100"})
        self.assertNotIn(ENV, env); self.assertNotIn("COLABFOLD_OPT", env); self.assertEqual(env["MODEL_OPT_TARGET_GPU"], "H100")   # the exact name goes; the tree's other MODEL_OPT_* names are deployment variables stock ignores
        with mock.patch.object(stock_pred, "launch_form", return_value={"form": "colabfold_batch"}):   # the proof records the launch form; no colabfold on this interpreter
            p = stock_pred.proof({ENV: "x", "PATH": "/bin"}, ["/usr/bin/colabfold_batch"])
        self.assertFalse(p["ok"]); self.assertEqual(p["env_present"], [ENV])


if __name__ == "__main__":
    unittest.main()
