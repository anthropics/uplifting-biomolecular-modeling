"""The command layer: the mode-vs-environment disagreement refusal (exit 2), an unknown mode (exit 2), off restricted to pred, check as a
dry run on the stubbed box for exact (the default) (the DRY-RUN line, exit 0 / 3), the manifest's shape, and the exit rule after a
run (TestExitRule: partial -> 3 unless --allow-partial, recorded; a failed item -> incomplete, exit 1)."""
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

from flashzoi_opt import cli, manifest, modes, report, stack
from flashzoi_opt.tests import _stubs


def _run(argv):
    err, out = io.StringIO(), io.StringIO()
    se, so = sys.stderr, sys.stdout
    sys.stderr, sys.stdout = err, out
    try:
        rc = cli.main(argv)
    finally:
        sys.stderr, sys.stdout = se, so
    return rc, err.getvalue(), out.getvalue()


class TestCli(unittest.TestCase):
    def setUp(self):
        _stubs.reset_activation()
        self.saved = os.environ.pop(modes.ENV_MODE, None)

    def tearDown(self):
        _stubs.reset_activation()
        if self.saved is not None:
            os.environ[modes.ENV_MODE] = self.saved

    def test_usage(self):
        self.assertEqual(_run([])[0], cli.EXIT_USAGE)
        self.assertEqual(_run(["frobnicate"])[0], cli.EXIT_USAGE)
        self.assertEqual(_run(["--help"])[0], cli.EXIT_OK)

    def test_mode_vs_env_disagreement(self):
        os.environ[modes.ENV_MODE] = "off"
        rc, err, _ = _run(["check", "--mode", "exact"])
        self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("disagrees with FLASHZOI_OPT=off", err)
        self.assertEqual(cli.effective_mode("off"), "off")
        os.environ[modes.ENV_MODE] = "exact"
        self.assertEqual(cli.effective_mode(None), "exact")
        del os.environ[modes.ENV_MODE]
        self.assertEqual(cli.effective_mode(None), modes.DEFAULT_MODE)
        with self.assertRaises(cli.CliError):
            cli.effective_mode("turbo")

    def test_unknown_mode_exit_2(self):
        rc, err, _ = _run(["check", "--mode", "turbo"])
        self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("unknown mode 'turbo'", err)

    def test_off_only_pred(self):
        for cmd in ("check", "warm"):
            rc, err, _ = _run([cmd, "--mode", "off"])
            self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("--mode off has one command, pred", err)

    def test_tf32_override_env_is_no_refusal_on_the_stock_route(self):
        """A TF32 library override in the environment (stack.TF32_OVERRIDE_ENV) is never a refusal: the stock route goes on to its own checks
        (here a missing --input: a usage error, not exit 3) — stock runs under the override as stock does (the kit routes: TestExitRule)."""
        from flashzoi_opt import stack
        for name in stack.TF32_OVERRIDE_ENV:
            os.environ[name] = "0"
            try:
                rc, err, _ = _run(["pred", "--mode", "off", "--input", "/nonexistent/in", "--out", "/nonexistent/out"])
                self.assertEqual(rc, cli.EXIT_USAGE, err); self.assertNotIn("NOT ACTIVE", err)
            finally:
                os.environ.pop(name, None)

    def test_pred_missing_input(self):
        rc, err, _ = _run(["pred", "--mode", "exact", "--input", "/nonexistent", "--out", "y"])
        self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("not a directory", err)

    def test_check_dry_run(self):
        kit = _stubs.require_kit()
        t = _stubs.Tree(kit=kit).enter()
        try:
            rc, err, _ = _run(["check", "--mode", "exact"])                     # the real CPU box: refused (no GPU / a pin)
            self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("[flashzoi-opt] DRY-RUN mode=exact variant=-", err); self.assertIn("would_refuse='", err)
            undo = _stubs.stub_box()
            try:
                rc, err, out = _run(["check", "--mode", "exact", "--json"])
                self.assertEqual(rc, cli.EXIT_OK, err); self.assertIn("would_refuse=none", err); self.assertIn("knobs=numerics=tf32 ", err)
                rep = json.loads(out)
                self.assertTrue(rep["dry_run"]); self.assertFalse(rep["active"]); self.assertEqual(rep["stack_key"], "9.0|3.1")
                self.assertEqual(rep["apply_line"], "kit.KitRunner(model)")
                rc, err, out = _run(["check", "--json"])                            # no --mode, no env: the package default (exact)
                self.assertEqual(rc, cli.EXIT_OK, err); self.assertIn("[flashzoi-opt] DRY-RUN mode=exact variant=-", err); self.assertIn("knobs=numerics=tf32 ", err)
                self.assertEqual(json.loads(out)["apply_line"], "kit.KitRunner(model)")
                self.assertFalse(stack.status()["active"])                      # nothing armed by a dry run
            finally:
                undo()
        finally:
            t.exit()


class TestManifest(unittest.TestCase):
    def test_shape(self):
        d = tempfile.mkdtemp()
        try:
            rep = {"active": True, "mode": "exact", "components_applied": ["a"], "components_fallback": [], "components_unavailable": [], "partial": False, "package_version": "0.1.0",
                   "gpu": {"name": "g", "cc": "9.0"}, "stack_key": "9.0|3.1", "kit_arm": "v", "pins": {"package": {}, "stack": {}}, "resolution": object()}
            p = manifest.write(d, rep, command="pred", argv=["pred"], exit_code=0, settings={"batch": 1}, det=None, weights=[{"repo": "r"}], items={"items": 1, "ok": 1, "failed": 0}, replicates=[{"apply_s": {}}])
            m = manifest.read(d)
            for k in ("schema", "written_at", "mode", "active", "components_applied", "components_fallback", "components_unavailable", "partial", "reason", "package_version", "gpu", "command", "argv",
                      "exit_code", "python", "torch", "flashzoi_opt_env", "stack_key", "kit_arm", "pins", "weights", "settings", "det", "items", "replicates", "activation_report"):
                self.assertIn(k, m, k)
            self.assertEqual(m["schema"], manifest.SCHEMA); self.assertNotIn("resolution", m["activation_report"]); self.assertEqual(m["items"]["ok"], 1)
            self.assertEqual(os.path.basename(p), "opt_manifest.json")
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


class _Dev:
    def __init__(self, t):
        self.type = t


class TestExitRule(unittest.TestCase):
    """The exit rule of pred / warm after the run, on the stubbed box with the stub kit (no GPU, no upstream): the weights load and
    the loop are stubbed at the package's own seams (weights.load_replicates, loop.run_items) and the kit's evidence is the stub
    KitRunner's effective_flags() — a lever of the mode without a true flag is components_fallback: the run is PARTIAL, recorded in the
    manifest (partial, partial_detection, allow_partial), the outputs kept, exit 3; --allow-partial proceeds to the run's own exit;
    a failed item is incomplete (exit 1), never partial; the jobs form exits 3 once after every job's manifest."""

    @classmethod
    def setUpClass(cls):
        real = _stubs.require_kit()
        cls.table = modes.kit_table(real)
        _stubs.PINNED_GPU = cls.table["PINS"]["device_names"][0]

    def setUp(self):
        _stubs.reset_activation(); _stubs.forget_kit_modules()
        self.tree = _stubs.Tree().enter()
        _stubs.write_stub_kit(self.tree.root, self.table, _stubs.PINNED_GPU)
        self.undo = _stubs.stub_box()
        self.tmp = tempfile.mkdtemp(prefix="flashzoi-exit-rule-")
        self.inp = os.path.join(self.tmp, "in"); os.makedirs(self.inp)
        for name in ("w1", "w2"):
            np.save(os.path.join(self.inp, name + ".npy"), np.zeros((4, 8), dtype=np.uint8))
        from flashzoi_opt import loop, weights
        from flashzoi_opt import settings as _settings
        self.saved = {"install_hook": stack._install_hook, "model_device": stack._model_device, "load": weights.load_replicates, "run_items": loop.run_items,
                      "apply_numerics": _settings.apply_numerics, "read_back": _settings.read_back, "torch": sys.modules.get("torch")}
        stack._install_hook = lambda: None                                    # the class-level hook needs the upstream module; pred applies eagerly
        stack._model_device = lambda model: _Dev("cuda")
        _settings.apply_numerics = lambda st: {}
        _settings.read_back = lambda: {}
        torch = types.ModuleType("torch"); torch.cuda = types.SimpleNamespace(synchronize=lambda: None); sys.modules["torch"] = torch
        self.flags = {}                                                       # lever -> flag the stub runner reports after the run (default: all true)
        self.fail_items = 0

        def load_replicates(pins, replicates, device, after_each=None):
            m = types.SimpleNamespace()
            if after_each is not None:
                after_each(m, {"repo": replicates[0], "sha256": "0" * 64})
            return [m], [{"repo": replicates[0], "sha256": "0" * 64}]

        def run_items(models, items, out, st, device):
            os.makedirs(out, exist_ok=True)
            r = models[0]._flashzoi_kit_runner
            flags = dict(self.flags)
            r.effective_flags = lambda: {lv: flags.get(lv, True) for lv in r.components}
            n = len(items)
            return {"items": n, "ok": n - self.fail_items, "failed": self.fail_items}
        weights.load_replicates = load_replicates
        loop.run_items = run_items

    def tearDown(self):
        from flashzoi_opt import loop, weights
        from flashzoi_opt import settings as _settings
        stack._install_hook = self.saved["install_hook"]; stack._model_device = self.saved["model_device"]
        weights.load_replicates = self.saved["load"]; loop.run_items = self.saved["run_items"]
        _settings.apply_numerics = self.saved["apply_numerics"]; _settings.read_back = self.saved["read_back"]
        if self.saved["torch"] is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self.saved["torch"]
        self.undo(); self.tree.exit(); _stubs.reset_activation(); _stubs.forget_kit_modules()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pred(self, *extra):
        out = os.path.join(self.tmp, "out")
        rc, err, _ = _run(["pred", "--mode", "exact", "--input", self.inp, "--out", out, *extra])
        man = json.load(open(os.path.join(out, "opt_manifest.json"))) if os.path.exists(os.path.join(out, "opt_manifest.json")) else None
        return rc, err, man

    def test_tf32_override_env_is_named_drift_at_the_gate(self):
        """The kit routes NAME a TF32 library override (environment drift) and run: pred prints ` drift=[<VAR>=1]` on its ACTIVE line and ends
        on its own exit, the manifest recording env_noted / drift; check prints the same suffix on the DRY-RUN line and exits 0."""
        for name in stack.TF32_OVERRIDE_ENV:
            os.environ[name] = "1"
            try:
                rc, err, man = self._pred()
                self.assertNotIn("NOT ACTIVE: TF32", err); self.assertIn(f" drift=[{name}=1]", err)
                self.assertIsNotNone(man); self.assertEqual(man["activation_report"]["env_noted"], [name]); self.assertEqual(man["drift"], [f"{name}=1"]); self.assertEqual(man["device_class"], "pinned")
                rc, err, _ = _run(["check", "--mode", "exact"])
                self.assertEqual(rc, cli.EXIT_OK, err); self.assertIn(f" drift=[{name}=1]", err)
                rep = stack.activate("exact", dry_run=True, trigger="check")
                self.assertEqual(rep["env_hits"], []); self.assertEqual(rep["env_noted"], [name]); self.assertEqual(rep["env_checked"], list(stack.FORBIDDEN_ENV))
            finally:
                os.environ.pop(name, None)
                _stubs.reset_activation()

    def test_pred_partial_exits_3_recorded_outputs_kept(self):
        self.flags = {"graph": False}
        rc, err, man = self._pred()
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE, err)
        self.assertIn("[flashzoi-opt] NOT ACTIVE: partial activation — graph: components_fallback (no true KitRunner.effective_flags() entry after the run); "
                      "exit 3 (--allow-partial records and proceeds)", err)                                                # the family grammar, verbatim
        self.assertNotIn("PARTIAL allowed", err)
        self.assertTrue(man["partial"]); self.assertEqual(man["components_fallback"], ["graph"]); self.assertEqual(man["exit_code"], 3)
        self.assertFalse(man["allow_partial"]); self.assertEqual(man["partial_detection"], stack.PARTIAL_DETECTION); self.assertNotIn("incomplete", man)
        self.assertEqual(man["items"], {"items": 2, "ok": 2, "failed": 0})
        self.assertEqual(man["replicates"][0]["effective_flags"]["graph"], False)
        self.assertNotIn("graph", man["components_applied"])

    def test_a_model_the_kit_cannot_attach_to_refuses_the_mode_by_name(self):
        """The kit's constructor raising (a kernel that does not compile or launch here, a model shape it does not serve) is `cannot run`:
        the MODE refuses by name — one NOT ACTIVE line naming the kit's error, exit 3, no item runs (a mode is all of its levers; it never
        runs under its name with a subset, and nothing falls back to the stock forward under the mode's name)."""
        def raising(model, **kw):
            raise RuntimeError("Triton launch failed: no kernel image for this device")
        real_import = stack.import_kit
        def import_kit(root, table):
            kit = real_import(root, table); kit.KitRunner = raising; return kit
        from flashzoi_opt import loop
        real_run_items = loop.run_items; ran = []
        loop.run_items = lambda *a, **k: ran.append(a) or {"items": 0, "ok": 0, "failed": 0}
        stack.import_kit = import_kit
        try:
            rc, err, man = self._pred()
            self.assertEqual(rc, cli.EXIT_NOT_ACTIVE, err)
            self.assertIn("[flashzoi-opt] NOT ACTIVE: kit.KitRunner(model", err)
            self.assertIn("cannot run here: RuntimeError: Triton launch failed: no kernel image for this device — mode exact refused", err)
            self.assertNotIn("FALLBACK", err); self.assertEqual(ran, [])                                            # no stock forward under the mode's name, no item run
            _stubs.reset_activation(); _stubs.forget_kit_modules()
            rc, err, _ = self._pred("--allow-partial")                                                            # --allow-partial is about lever evidence after a run, not about a mode that cannot run
            self.assertEqual(rc, cli.EXIT_NOT_ACTIVE, err); self.assertIn("cannot run here", err)
        finally:
            stack.import_kit = real_import; loop.run_items = real_run_items

    def test_pred_allow_partial_proceeds_to_the_runs_own_exit(self):
        self.flags = {"graph": False}
        rc, err, man = self._pred("--allow-partial")
        self.assertEqual(rc, cli.EXIT_OK, err)
        self.assertNotIn("NOT ACTIVE", err)
        self.assertIn("[flashzoi-opt] PARTIAL allowed: graph: components_fallback (no true KitRunner.effective_flags() entry after the run) (--allow-partial, recorded)", err)
        self.assertTrue(man["partial"]); self.assertTrue(man["allow_partial"]); self.assertEqual(man["components_fallback"], ["graph"]); self.assertEqual(man["exit_code"], 0)
        self.fail_items = 1                                                   # partial AND a failed item: the run's own exit is 1, both recorded
        rc, err, man = self._pred("--allow-partial")
        self.assertEqual(rc, cli.EXIT_FAILED, err); self.assertTrue(man["partial"]); self.assertEqual(man["incomplete"], "1/2")

    def test_pred_failed_item_is_incomplete_exit_1_not_partial(self):
        self.fail_items = 1
        rc, err, man = self._pred()
        self.assertEqual(rc, cli.EXIT_FAILED, err)
        self.assertNotIn("NOT ACTIVE", err)
        self.assertFalse(man["partial"]); self.assertEqual(man["components_fallback"], []); self.assertEqual(man["incomplete"], "1/2"); self.assertFalse(man["allow_partial"])
        self.fail_items = 0                                                   # a complete, fully effective run records neither
        rc, err, man = self._pred()
        self.assertEqual(rc, cli.EXIT_OK, err); self.assertFalse(man["partial"]); self.assertNotIn("incomplete", man); self.assertFalse(man["allow_partial"])

    def test_pred_jobs_partial_exits_3_once_after_every_jobs_manifest(self):
        self.flags = {"pinned": False}
        outs = [os.path.join(self.tmp, f"job{i}") for i in (1, 2)]
        jobs = os.path.join(self.tmp, "jobs.tsv")
        open(jobs, "w").write("".join(f"{self.inp}\t{o}\n" for o in outs))
        rc, err, _ = _run(["pred", "--mode", "exact", "--jobs", jobs])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE, err)
        self.assertEqual(err.count("NOT ACTIVE"), 1)
        self.assertIn("[flashzoi-opt] NOT ACTIVE: partial activation — pinned in 2/2 job(s) (1,2): components_fallback (no true KitRunner.effective_flags() entry after the run); "
                      "exit 3 (--allow-partial records and proceeds)", err)
        self.assertIn("[flashzoi-opt multi] DONE jobs=2", err)
        for o in outs:
            man = json.load(open(os.path.join(o, "opt_manifest.json")))
            self.assertTrue(man["partial"]); self.assertEqual(man["exit_code"], 3); self.assertFalse(man["allow_partial"]); self.assertEqual(man["multi"]["jobs"], 2)
        _stubs.reset_activation(); _stubs.forget_kit_modules()
        rc, err, _ = _run(["pred", "--mode", "exact", "--jobs", jobs, "--allow-partial"])
        self.assertEqual(rc, cli.EXIT_OK, err); self.assertNotIn("NOT ACTIVE", err)
        self.assertEqual(err.count("[flashzoi-opt] PARTIAL allowed: pinned in 2/2 job(s) (1,2): components_fallback"), 1)
        self.assertTrue(json.load(open(os.path.join(outs[1], "opt_manifest.json")))["allow_partial"])

    def test_warm_partial_follows_the_running_verbs_rule(self):
        from flashzoi_opt import warm
        saved = warm.run
        seen = {}

        def run(mode, allow_partial=False):
            seen["allow_partial"] = allow_partial
            return {"status": "PARTIAL", "mode": mode, "forwards": 1, "triton_cache_files": {"before": 0, "after": 3}, "wall_s": 1.0, "partial": True,
                    "components_fallback": ["graph"], "partial_detection": stack.PARTIAL_DETECTION, "allow_partial": allow_partial, "reason": "components_fallback=graph"}
        warm.run = run
        try:
            rc, err, _ = _run(["warm", "--mode", "exact"])
            self.assertEqual(rc, cli.EXIT_NOT_ACTIVE, err); self.assertFalse(seen["allow_partial"])
            self.assertIn("[flashzoi-opt] WARM PARTIAL mode=exact forwards=1 triton_cache=0->3 wall=1.0s reason=components_fallback=graph", err)
            self.assertIn("[flashzoi-opt] NOT ACTIVE: partial activation — graph: components_fallback (no true KitRunner.effective_flags() entry after the run); "
                          "exit 3 (--allow-partial records and proceeds)", err)
            rc, err, _ = _run(["warm", "--mode", "exact", "--allow-partial"])
            self.assertEqual(rc, cli.EXIT_OK, err); self.assertTrue(seen["allow_partial"]); self.assertNotIn("NOT ACTIVE", err)
            self.assertIn("[flashzoi-opt] PARTIAL allowed: graph: components_fallback (no true KitRunner.effective_flags() entry after the run) (--allow-partial, recorded)", err)
        finally:
            warm.run = saved

    def test_settle_reads_the_planned_levers_against_the_kits_flags(self):
        planned = list(self.table["LEVERS"])
        class R:
            components = frozenset(planned)
            apply_s = {}
            def __init__(self, eff):
                self._eff = eff
            def effective_flags(self):
                if isinstance(self._eff, Exception):
                    raise self._eff
                return self._eff
            def all_counts(self):
                return {}
            def jit_after_job(self):
                return {}
        stack._REPORT = {"active": True, "mode": "exact", "components_planned": planned, "models": []}
        stack._RUNNERS[:] = [R({lv: True for lv in planned})]
        rep = stack.settle()
        self.assertFalse(rep["partial"]); self.assertEqual(rep["components_applied"], planned); self.assertEqual(rep["partial_detection"], stack.PARTIAL_DETECTION)
        absent = {lv: True for lv in planned if lv != planned[0]}             # a planned lever ABSENT from the flags: no evidence = fallback
        stack._RUNNERS[:] = [R(absent), R(dict({lv: True for lv in planned}, **{planned[1]: False}))]
        rep = stack.settle()
        self.assertTrue(rep["partial"]); self.assertEqual(rep["components_fallback"], sorted(planned[:2]))
        self.assertEqual(rep["models"][0]["components_fallback"], [planned[0]]); self.assertEqual(rep["models"][1]["components_fallback"], [planned[1]])
        stack._RUNNERS[:] = [R(RuntimeError("flags unreadable"))]             # the flags unreadable: nothing evidenced, every planned lever a fallback
        rep = stack.settle()
        self.assertTrue(rep["partial"]); self.assertEqual(rep["components_fallback"], sorted(planned)); self.assertIn("flags unreadable", rep["models"][0]["effective_flags"]["error"])
        self.assertTrue(report.exit_tally_line("exact", wall_s=1.0).endswith(f" partial={','.join(sorted(planned))}"))

    def test_pinned_lever_is_evidenced_by_the_helper_route(self):
        """The kit's own `pinned` flag reads its lease pool (the host route); the package's route is the documented helper over the device path,
        so the flag is false on every package run — the evidence is the helper's per-call stamp (stack.note_helper_call), read by
        settle() through helper_pinned_evidence and recorded per runner as pinned_evidence."""
        import sys, types
        planned = list(self.table["LEVERS"]); self.assertIn("pinned", planned)
        class R:
            components = frozenset(planned)
            apply_s = {}
            _helper_route = stack.STOCK_HELPER_MODULE + ".predict_tracks -> helper: code-swap on the bound object (stub)"
            def __init__(self, counts, route=None):
                self._counts = counts
                if route is not None:
                    self._helper_route = route
            def effective_flags(self):
                return dict({lv: True for lv in planned}, pinned=False)      # the kit's flag on the package route: no lease ever taken
            def all_counts(self):
                return dict(self._counts)
            def jit_after_job(self):
                return {}
        fake = types.ModuleType(stack.HELPER_MODULE)
        def served(landing_ran):                                                # the helper serving a call: LAST_CALL cleared and stamped afresh (predict_tracks_fast.py:113)
            fake.LAST_CALL.clear(); fake.LAST_CALL.update(landing_requested="direct", landing_ran=landing_ran, kind="all", overlap=True)
        fake.LAST_CALL = {}; served("direct")
        saved = sys.modules.get(stack.HELPER_MODULE); sys.modules[stack.HELPER_MODULE] = fake
        try:
            stack._REPORT = {"active": True, "mode": "exact", "components_planned": planned, "models": []}
            stack._HELPER_CALLS.clear()
            self.assertEqual(stack.note_helper_call(), {"landing_requested": "direct", "landing_ran": "direct", "kind": "all", "overlap": True})
            served("direct"); stack.note_helper_call()
            self.assertEqual(stack._HELPER_CALLS, {"direct": 2})
            stack.note_helper_call()                                                # a call the helper did not serve: the stamp is the one already counted
            self.assertEqual(stack._HELPER_CALLS, {"direct": 2, "stale stamp": 1})
            stack._RUNNERS[:] = [R({"predict_calls": 3, "predict_device_calls": 3})]
            self.assertTrue(stack.settle()["partial"]); self.assertEqual(stack.settle()["components_fallback"], ["pinned"])
            stack._HELPER_CALLS.clear(); stack._HELPER_CALLS.update(direct=2)
            stack._RUNNERS[:] = [R({"predict_calls": 2, "predict_device_calls": 2, "predict_host_calls": 0, "leases": 0})]
            rep = stack.settle()                                                    # the control: every call on the device route, every landing pinned
            self.assertFalse(rep["partial"]); self.assertEqual(rep["components_fallback"], []); self.assertEqual(rep["components_applied"], planned)
            self.assertTrue(rep["models"][0]["effective_flags"]["pinned"]); self.assertIn("pinned host landing on every call = True", rep["models"][0]["pinned_evidence"])
            served("stock (cpu)")
            stack.note_helper_call()                                                # one call through the stock body: the lever is not evidenced
            stack._RUNNERS[:] = [R({"predict_calls": 3, "predict_device_calls": 3})]
            rep = stack.settle()
            self.assertTrue(rep["partial"]); self.assertEqual(rep["components_fallback"], ["pinned"]); self.assertIn("'stock (cpu)': 1", rep["models"][0]["pinned_evidence"])
            stack._RUNNERS[:] = [R({"predict_calls": 3, "predict_device_calls": 2, "predict_host_calls": 1})]   # a host-route call: no stamp for it
            stack._HELPER_CALLS.clear(); stack._HELPER_CALLS.update(direct=3)
            self.assertTrue(stack.settle()["partial"])
            stack._HELPER_CALLS.clear()                                             # the environment route: no loop call counted — the runner's route + the last stamp
            served("pool")
            stack._RUNNERS[:] = [R({"predict_calls": 5, "predict_device_calls": 5})]
            rep = stack.settle()
            self.assertFalse(rep["partial"]); self.assertIn("environment route (no per-call count): helper routed = True", rep["models"][0]["pinned_evidence"])
            stack._RUNNERS[:] = [R({"predict_calls": 5, "predict_device_calls": 5}, route="not routed (FZ_KIT_HELPER=off)")]
            rep = stack.settle()
            self.assertTrue(rep["partial"]); self.assertEqual(rep["components_fallback"], ["pinned"])
            del sys.modules[stack.HELPER_MODULE]                                    # the helper module never loaded: `no stamp`
            stack.note_helper_call(); self.assertEqual(stack._HELPER_CALLS, {"no stamp": 1})
            stack._RUNNERS[:] = [R({"predict_calls": 1, "predict_device_calls": 1})]
            self.assertTrue(stack.settle()["partial"])
        finally:
            stack._HELPER_CALLS.clear()
            if saved is not None:
                sys.modules[stack.HELPER_MODULE] = saved
            else:
                sys.modules.pop(stack.HELPER_MODULE, None)
