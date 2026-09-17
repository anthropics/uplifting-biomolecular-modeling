"""Runs on which colabfold_batch builds no model exit as stock does (0), by name, in every mode: `--num-models 0` / `--msa-only`
(`no_model_run=num_models_0`), a results directory whose every job is already complete — colabfold resumes: it skips a job whose
`<job>.result.zip` (the `--zip` form) or `<job>.done.txt` exists (`all_jobs_done`) —, and `--af3-json`, which returns before `run()`
(`af3_json`); the applied levers such a run leaves without a call are `idle` (the IDLE line: `levers=not_applicable:<names>`; their LEVER
lines: `no_model_run=<case>`), never a partial activation — while a run that DID build a model with a lever left unserved stays PARTIAL,
exit 3. Two layers, both on the package's real code paths: the run hook in-process over test_activation's stand-in `colabfold.batch` whose
`run_jobs` keeps stock's per-job completion facts exactly (stack.hook_run → no_model_run_census → post_run_partial → the exit record →
manifest.verdict), and `pred` over test_cli_manifest's stub `colabfold_batch`, which lays the results directory out as stock leaves it."""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr

from colabfold_opt import inputs, manifest, report, stack
from colabfold_opt.tests import _stubs
from colabfold_opt.tests.test_activation import Base as HookBase
from colabfold_opt.tests.test_cli_manifest import A3M, Base as PredBase
import json  # noqa: E402


class TestRunHook(HookBase):
    """The model process: `colabfold.batch.run` entered through the strict hook (the env route on its own — this process is the entry point,
    so the hook prints the IDLE line itself)."""

    def hooked(self, work):
        cb = self.mods["colabfold.batch"]
        cb.run = cb.run_jobs                                                     # stock's job loop with its completion facts and a stand-in model
        os.environ["COLABFOLD_OPT_WORK_DIR"] = work                             # `pred` names a work directory: the manifest goes there
        stack.hook_run("fast", strict=True, trigger="colabfold.batch", module=cb)
        return cb

    def run_main(self, cb, res, **flags):
        err, code = io.StringIO(), None
        with redirect_stderr(err):
            try:
                cb.main(queries=_stubs.queries(199), result_dir=res, data_dir=None, **flags)
            except SystemExit as e:
                code = e.code
        return code, err.getvalue()

    def at_exit(self):
        """What interpreter exit does: the LEVER lines, then the manifest's exit record (report._print_exit_tally → manifest.record_exit)."""
        err = io.StringIO()
        with redirect_stderr(err):
            report._print_exit_tally()
        return err.getvalue()

    def dirs(self):
        return tempfile.mkdtemp(dir=self.tmp), tempfile.mkdtemp(dir=self.tmp)

    def test_num_models_0_leaves_the_levers_idle_by_name_exit_0(self):
        work, res = self.dirs()
        cb = self.hooked(work)
        code, err = self.run_main(cb, res, num_models=0)
        self.assertIsNone(code, err)                                             # no exit 3: the run keeps stock's own exit
        self.assertEqual(sorted(os.listdir(res)), ["q0.a3m", "q0.pickle"])       # as stock leaves it: the MSA and the input features, no marker, no model file
        self.assertIn("[colabfold-opt] IDLE no_model_run=num_models_0 levers=not_applicable:", err)
        self.assertIn("colabfold_batch built no model (num_models=0)", err); self.assertIn("not a partial activation; exit 0\n", err)
        self.assertNotIn("NOT ACTIVE", err); self.assertNotIn("partial activation —", err)
        man = manifest.read(manifest.path(work))
        self.assertEqual((man["no_model_run"], man["activation_report"]["no_model_run"]), ("num_models_0", "num_models_0"))   # the model process's own decision, where `pred` reads it
        applied = list(man["activation_report"]["levers_applied"])
        self.assertTrue(applied)
        tally = self.at_exit()
        lever_lines = [l for l in tally.splitlines() if "] LEVER name=" in l and " state=on " in l]
        idle_named = [l.split("name=")[1].split()[0] for l in lever_lines if " no_model_run=num_models_0" in l]
        man = manifest.read(manifest.path(work))                                # the exit record
        self.assertEqual((man["partial"], sorted(man["idle"])), ([], sorted(idle_named)))    # every idle lever says so on its own line; none is partial
        self.assertTrue(set(man["idle"]) <= set(applied)); self.assertTrue(man["idle"])
        v = manifest.verdict(work, "fast", ["q0"], 0, result_dir=res)           # the parent's reading of the same record + the results directory
        self.assertEqual((v["ok"], v["rc"], v["reason"], v["no_model_run"], v["missing"], v["partial"], sorted(v["idle"])),
                         (True, 0, None, "num_models_0", [], [], sorted(man["idle"])))
        self.assertEqual(v["completion"], {"q0": None})                         # no marker is owed when no model was asked

    def test_a_results_directory_already_complete_is_resumed_idle_by_name_exit_0(self):
        work, res = self.dirs()
        open(os.path.join(res, "q0.done.txt"), "w").close()                      # a finished job as stock leaves it: the marker and its files
        open(os.path.join(res, "q0_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"), "w").write("ATOM\n")
        cb = self.hooked(work)
        code, err = self.run_main(cb, res, num_models=5)
        self.assertIsNone(code, err)
        self.assertIn({"skipped": "q0"}, cb.RUN_LOG); self.assertFalse([e for e in cb.RUN_LOG if "predicted" in e])   # colabfold skipped the job: no model, no lever call
        self.assertIn("[colabfold-opt] IDLE no_model_run=all_jobs_done levers=not_applicable:", err)
        self.assertIn("(1/1: q0.done.txt)", err); self.assertIn("not a partial activation; exit 0\n", err); self.assertNotIn("NOT ACTIVE", err)
        self.at_exit()
        man = manifest.read(manifest.path(work))
        self.assertEqual((man["no_model_run"], man["partial"]), ("all_jobs_done", [])); self.assertTrue(man["idle"])
        v = manifest.verdict(work, "fast", ["q0"], 0, result_dir=res)
        self.assertEqual((v["ok"], v["rc"], v["no_model_run"], v["missing"], v["completion"]), (True, 0, "all_jobs_done", [], {"q0": "done.txt"}))

    def test_a_zipped_results_directory_is_complete_by_colabfolds_own_test(self):
        work, res = self.dirs()
        open(os.path.join(res, "q0.result.zip"), "wb").close()                   # --zip's finished job: the archive alone, no marker (batch.py:1643-1651)
        cb = self.hooked(work)
        code, err = self.run_main(cb, res, num_models=5, zip_results=True)
        self.assertIsNone(code, err)
        self.assertIn({"skipped": "q0"}, cb.RUN_LOG)
        self.assertIn("[colabfold-opt] IDLE no_model_run=all_jobs_done levers=not_applicable:", err); self.assertIn("(1/1: q0.result.zip)", err)
        self.at_exit()
        v = manifest.verdict(work, "fast", ["q0"], 0, result_dir=res)
        self.assertEqual((v["ok"], v["rc"], v["no_model_run"], v["missing"], v["completion"]), (True, 0, "all_jobs_done", [], {"q0": "result.zip"}))

    def test_a_run_that_built_a_model_with_an_unserved_lever_is_partial_exit_3(self):
        """The standing rule: a model DID run (colabfold's facts: a model asked for, the job not complete — the stand-in model wrote its files and the
        marker) and levers of the mode read calls=0 with no by-design word — a PARTIAL activation, the house line, exit 3, whatever the
        counters; `idle` is empty, `no_model_run` is None."""
        work, res = self.dirs()
        cb = self.hooked(work)
        code, err = self.run_main(cb, res, num_models=1)
        self.assertEqual(code, 3, err)
        self.assertIn("[colabfold-opt] NOT ACTIVE: partial activation —", err); self.assertNotIn("] IDLE ", err)
        self.assertIn({"predicted": "q0", "models": 1}, cb.RUN_LOG); self.assertTrue(os.path.isfile(os.path.join(res, "q0.done.txt")))
        self.at_exit()
        man = manifest.read(manifest.path(work))
        self.assertIsNone(man["no_model_run"]); self.assertEqual(man["idle"], []); self.assertTrue(man["partial"])
        v = manifest.verdict(work, "fast", ["q0"], 3, result_dir=res)
        self.assertEqual((v["ok"], v["rc"], v["reason"], v["idle"], v["no_model_run"]), (False, 3, "kernel_not_engaged", [], None)); self.assertTrue(v["partial"])

    def test_overwrite_existing_results_recomputes_so_the_partial_rule_stands(self):
        """`--overwrite-existing-results` (keep_existing_results False): colabfold recomputes a finished job — a model runs, the census says
        so from run()'s own argument, and unserved levers are PARTIAL as ever."""
        work, res = self.dirs()
        open(os.path.join(res, "q0.done.txt"), "w").close()
        cb = self.hooked(work)
        code, err = self.run_main(cb, res, num_models=1, keep_existing_results=False)
        self.assertEqual(code, 3, err); self.assertIn({"predicted": "q0", "models": 1}, cb.RUN_LOG); self.assertNotIn("] IDLE ", err)


class TestPred(PredBase):
    """`pred` over the stub colabfold_batch: the verdict reads colabfold's own completion artefacts and the no-model cases in every mode."""

    def setUp(self):
        super().setUp()
        self.gates = _stubs.gates_pass(stack)                                    # the parent's gates pass on this CPU box

    def tearDown(self):
        _stubs.gates_restore(stack, self.gates)
        super().tearDown()

    def pred(self, mode, name, *opts):
        rd = os.path.join(self.out(name), "pred")
        err = io.StringIO()
        with redirect_stderr(err):
            rc = self.main(["pred", "--mode", mode, A3M, rd, "--data", self.data, *opts])
        return rc, err.getvalue(), rd

    def idle_lines(self, err):
        return [l for l in err.splitlines() if l.startswith("[colabfold-opt] IDLE ")]

    def test_zip_a_jobs_completion_is_its_result_zip(self):
        rc, err, rd = self.pred("off", "z", "--zip")
        self.assertEqual(rc, 0, err)                                             # stock exits 0 with the outputs in the archive: so does pred
        files = os.listdir(rd)
        self.assertIn("1BRS_AD.result.zip", files); self.assertNotIn("1BRS_AD.done.txt", files); self.assertFalse([f for f in files if f.endswith(".pdb")])
        v = self.mans[rd]["verdict"]
        self.assertEqual((v["reason"], v["missing"], v["completion"], v["no_model_run"]), (None, [], {"1BRS_AD": "result.zip"}, None))
        self.assertEqual(self.idle_lines(err), [])                               # a model ran: no IDLE line
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 7})
        rc, err, rd2 = self.pred("fast", "zf", "--zip")
        self.assertEqual(rc, 0, err); self.assertEqual(self.mans[rd2]["verdict"]["completion"], {"1BRS_AD": "result.zip"})
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls_per_job": 112})   # launched again into the zipped directory: colabfold skips the job (result.zip), no lever call
        err = io.StringIO()
        with redirect_stderr(err):
            rc = self.main(["pred", "--mode", "fast", A3M, rd2, "--data", self.data, "--zip"])
        self.assertEqual(rc, 0, err.getvalue())
        v = self.mans[rd2]["verdict"]
        self.assertEqual((v["no_model_run"], v["calls"], v["partial"], v["idle"]), ("all_jobs_done", 0, [], ["AF_PALLAS_ATTN"]))
        [line] = self.idle_lines(err.getvalue())
        self.assertIn("no_model_run=all_jobs_done levers=not_applicable:AF_PALLAS_ATTN jobs=1: every job was complete in the results directory before the run (1/1: 1BRS_AD.result.zip)", line)

    def test_msa_only_and_num_models_0_owe_no_marker_and_leave_the_levers_idle(self):
        rc, err, rd = self.pred("off", "m", "--msa-only")
        self.assertEqual(rc, 0, err)                                             # stock: MSAs and features, no model, exit 0
        files = sorted(f for f in os.listdir(rd) if f.startswith("1BRS_AD"))
        self.assertEqual(files, ["1BRS_AD.a3m", "1BRS_AD.pickle"])              # no marker, no model file
        self.assertEqual(self.idle_lines(err), ["[colabfold-opt] IDLE no_model_run=num_models_0 levers=none jobs=1: colabfold_batch built no model (--msa-only): "
                                                "MSAs and input features only, no done.txt written; exit 0"])
        v = self.mans[rd]["verdict"]
        self.assertEqual((v["reason"], v["missing"], v["no_model_run"], v["idle"]), (None, [], "num_models_0", []))
        self.assertEqual(self.mans[rd]["settings"]["models_per_seed"], 0)
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 0})   # a kit mode: the levers applied, no call reached them — no model was asked
        rc, err, rd = self.pred("fast", "n", "--num-models", "0")
        self.assertEqual(rc, 0, err); self.assertNotIn("NOT ACTIVE", err)
        self.assertEqual(self.idle_lines(err), ["[colabfold-opt] IDLE no_model_run=num_models_0 levers=not_applicable:AF_PALLAS_ATTN jobs=1: colabfold_batch built no model "
                                                "(--num-models 0): MSAs and input features only, no done.txt written — the mode's levers had no model call to serve: "
                                                "not a partial activation; exit 0"])
        rec = self.mans[rd]; v = rec["verdict"]
        self.assertEqual((v["ok"], v["calls"], v["partial"], v["idle"], v["no_model_run"], rec["idle"], rec["partial"], rec["no_model_run"], rec["exit_code"]),
                         (True, 0, [], ["AF_PALLAS_ATTN"], "num_models_0", ["AF_PALLAS_ATTN"], [], "num_models_0", 0))
        rc, err, rd = self.pred("exact", "e", "--msa-only", "--num-models", "3")   # --msa-only wins over --num-models (main(): num_models = 0)
        self.assertEqual(rc, 0, err); self.assertEqual(self.mans[rd]["verdict"]["no_model_run"], "num_models_0")

    def test_af3_json_returns_before_run_the_hook_not_entered_by_design(self):
        os.environ["STUB_MANIFEST"] = json.dumps({"active": True, "calls": 7})   # would be written by run() — which --af3-json never enters
        rc, err, rd = self.pred("fast", "a", "--af3-json")
        self.assertEqual(rc, 0, err)                                             # not hook_never_fired: colabfold_batch returned before its run() by its own flag
        self.assertEqual(sorted(f for f in os.listdir(rd) if f.startswith("1BRS_AD")), ["1BRS_AD.a3m", "1BRS_AD.json"])
        self.assertEqual(self.idle_lines(err), ["[colabfold-opt] IDLE no_model_run=af3_json levers=none jobs=1: colabfold_batch wrote the AlphaFold 3 input JSON (--af3-json) "
                                                "and returned before its run(): the mode's hook was not entered, nothing was applied; exit 0"])
        v = self.mans[rd]["verdict"]
        self.assertEqual((v["reason"], v["active"], v["no_model_run"], v["missing"]), (None, None, "af3_json", []))
        self.assertNotIn("NOT ACTIVE", err)
        rc, err, rd = self.pred("off", "b", "--af3-json")
        self.assertEqual(rc, 0, err); self.assertEqual(self.mans[rd]["verdict"]["no_model_run"], "af3_json")
        os.environ.pop("STUB_MANIFEST")                                          # without the flag the same launch is hook_never_fired as before (the rule this case is the exception to)
        rc, err, rd = self.pred("fast", "c")
        self.assertEqual((rc, self.mans[rd]["verdict"]["reason"]), (3, "hook_never_fired"))


class TestReadings(unittest.TestCase):
    """The record-level readings: the decision reads colabfold's facts; idle versus partial is the no-model case alone; the line's format."""

    def test_the_decision_reads_colabfolds_own_facts(self):
        d = tempfile.mkdtemp()
        try:
            self.assertIsNone(manifest.no_model_run(5, ["a", "b"], d))
            open(os.path.join(d, "a.done.txt"), "w").close()
            self.assertIsNone(manifest.no_model_run(5, ["a", "b"], d))          # one job still to run: a model is expected
            open(os.path.join(d, "b.result.zip"), "wb").close()
            self.assertEqual(manifest.no_model_run(5, ["a", "b"], d), "all_jobs_done")
            self.assertIsNone(manifest.no_model_run(5, ["a", "b"], d, keep_existing=False))   # --overwrite-existing-results recomputes
            self.assertEqual(manifest.no_model_run(0, ["a", "b"], d), "num_models_0")        # asked for none: that is the case, whatever the directory holds
            self.assertEqual(manifest.no_model_run("0", ["c"], d), "num_models_0")
            self.assertEqual(manifest.no_model_run(5, ["c"], d, af3_json=True), "af3_json")
            self.assertIsNone(manifest.no_model_run(5, [], d)); self.assertIsNone(manifest.no_model_run(None, ["c"], d))
            self.assertEqual((manifest.completion(d, "a"), manifest.completion(d, "b"), manifest.completion(d, "c"), manifest.completion(None, "a")), ("done.txt", "result.zip", None, None))
            open(os.path.join(d, "a.result.zip"), "wb").close()
            self.assertEqual(manifest.completion(d, "a"), "result.zip")          # both present: colabfold tests the archive first
            self.assertEqual(manifest.NO_MODEL_RUN_CASES, ("num_models_0", "all_jobs_done", "af3_json"))
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_idle_versus_partial_is_the_no_model_case_alone(self):
        zero = {"calls": 0, "fallbacks": 0, "fallback_reasons": {}}
        names = ["DEVICE_RESIDENT", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA"]
        man = {"activation_report": {"active": True, "mode": "fast", "levers_applied": names}, "kit_state_exit": dict(zero),
               "lever_states_exit": {n: dict(zero) for n in names if n != "AF_PALLAS_ATTN"}}
        self.assertEqual((manifest.partial_levers(man), manifest.idle_levers(man)), (names, []))       # a model was expected: zero traffic is PARTIAL
        for where in ("top", "report"):
            m = json.loads(json.dumps(man))
            if where == "top":
                m["no_model_run"] = "all_jobs_done"
            else:
                m["activation_report"]["no_model_run"] = "num_models_0"
            self.assertEqual((manifest.partial_levers(m), manifest.idle_levers(m)), ([], names), where)   # no model was built: the same levers are idle, none partial
        m = dict(man, no_model_run="something_else")                             # not a case of the tool: no excuse
        self.assertEqual((manifest.no_model_case(m), manifest.partial_levers(m)), (None, names))
        ran = json.loads(json.dumps(man)); ran["lever_states_exit"]["DEVICE_RESIDENT"] = {"calls": 20, "fallbacks": 0}   # a model DID run through one lever: the others at zero are partial
        self.assertEqual(manifest.partial_levers(ran), [n for n in names if n != "DEVICE_RESIDENT"]); self.assertEqual(manifest.idle_levers(ran), [])

    def test_the_idle_line(self):
        self.assertEqual(report.idle_line("num_models_0", ["A", "B"], 2, "detail"), "[colabfold-opt] IDLE no_model_run=num_models_0 levers=not_applicable:A,B jobs=2: detail; exit 0")
        self.assertEqual(report.idle_line("af3_json", [], 1, "d"), "[colabfold-opt] IDLE no_model_run=af3_json levers=none jobs=1: d; exit 0")
        self.assertIn("no_model_run=", report.IDLE_FMT); self.assertEqual(report.IDLE_LEVERS, "not_applicable")

    def test_job_names_are_colabfolds_numbering(self):
        """`--jobname-prefix`: the index zero-filled to the width of the query COUNT (batch.py:1393-1396) — ten queries are <p>_00 … <p>_09."""
        keep = lambda s: s                                                        # noqa: E731
        self.assertEqual(inputs.job_names(_stubs.queries(*([10] * 10)), "job", safe_filename=keep), [f"job_{n:02d}" for n in range(10)])
        self.assertEqual(inputs.job_names(_stubs.queries(10, 20), "job", safe_filename=keep), ["job_0", "job_1"])
        self.assertEqual(inputs.job_names(_stubs.queries(10), "job", safe_filename=keep), ["job_0"])
        self.assertEqual(inputs.job_names([("a b", "AA", None, None)], None, safe_filename=lambda s: s.replace(" ", "_")), ["a_b"])


if __name__ == "__main__":
    unittest.main()
