"""Activation on stubs (a stand-in kit directory through COLABFOLD_OPT_KIT, stub colabfold.batch / alphafold.model.modules, the gates
monkeypatched): the kit's own row applied in-process (its directory on sys.path, AF_PALLAS_ATTN=1 exported, its enable() called, its
_STATE read back) at every input size; idempotent; late activation refused by name; strict raises after the
line; `off` applies nothing; the dry run touches nothing; the hook on colabfold.batch.run sees queries / result_dir / data_dir and writes
the manifest beside the outputs; the EXIT tally at interpreter exit (subprocesses). No jax, no GPU."""
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr

import colabfold_opt
from colabfold_opt import manifest, modes, registry, report, stack
try:
    from opt_core import __version__ as PROVIDER_V          # the shared core's version: AF_PALLAS_ATTN's LEVER line names the provider it bound (provider=)
except ImportError:  # pragma: no cover
    PROVIDER_V = "None"
from colabfold_opt.tests import _stubs

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import opt_core                                                                       # noqa: E402 — the shared core's import root, handed to child interpreters explicitly
CORE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.kit = _stubs.make_kit_dir(self.tmp)
        self.env_saved = {k: os.environ.get(k) for k in ("COLABFOLD_OPT_KIT", "AF_PALLAS_ATTN", "COLABFOLD_OPT", "COLABFOLD_OPT_WORK_DIR", "COLABFOLD_OPT_DATA_DIR", "MODEL_OPT_TARGET_GPU")}
        for k in self.env_saved:
            os.environ.pop(k, None)
        os.environ["COLABFOLD_OPT_KIT"] = self.kit
        self.log = []
        self.mods, self.saved = _stubs.install(self.log)
        self.gates = _stubs.gates_pass(stack)
        self.path_saved = list(sys.path)
        stack.reset_for_tests()                                          # also unregisters the exit tally (report.reset_tally_for_tests)

    def tearDown(self):
        _stubs.gates_restore(stack, self.gates)
        _stubs.remove(self.saved)
        sys.path[:] = self.path_saved
        for k, v in self.env_saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        stack.reset_for_tests()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def kit_module(self):
        return sys.modules.get("af2_pallas_attn")


class TestEnable(Base):
    def test_fast_applies_the_kits_row(self):
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600, 705))
        self.assertTrue(rep["active"]); self.assertIsNone(rep["reason"])
        self.assertEqual(rep["levers_applied"], ["DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"]); self.assertEqual(rep["levers_unavailable"], []); self.assertFalse(rep["partial"])
        self.assertTrue(getattr(self.mods["alphafold.model.model"].RunModel, "_device_resident", False)); self.assertEqual(rep["markers"], {"DEVICE_RESIDENT": True, "SUBBATCH": True, "TRIMUL_PALLAS": True, "AF_PALLAS_ATTN": True, "PALLAS_MSA": True, "TRIATTN_XLA": True, "MSA_COL_CUDNN": True, "TEMPL_DEDUP": True, "TRANSITION": True})
        self.assertEqual(os.environ["AF_PALLAS_ATTN"], "1")
        self.assertIn(os.path.join(self.kit, "af2_pallas_flash"), sys.path)
        km = self.kit_module()
        self.assertIsNotNone(km); self.assertTrue(km._STATE["enabled"]); self.assertEqual(km.ENABLE_CALLS, [False])   # AF_PALLAS_ATTN_ALL left at the kit default
        self.assertEqual(rep["kit_state"], {"enabled": True, "calls": 0, "fallbacks": 0})
        self.assertTrue(getattr(self.mods["alphafold.model.modules"].Attention, "_pallas_patched", False))
        self.assertEqual((rep["tokens_min"], rep["tokens_max"], rep["queries"]), (600, 705, 2)); self.assertNotIn("size_rule", rep)
        self.assertEqual(rep["key"], "9.0|0.5.3"); self.assertEqual(rep["stack_key"], "jax0.5.3-cu12.9-sm90")
        self.assertEqual((rep["colabfold_version"], rep["alphafold_colabfold_version"], rep["jax_version"], rep["package_version"]), ("1.6.1", "2.3.13", "0.5.3", colabfold_opt.__version__))
        self.assertIn("[colabfold-opt] ACTIVE mode=fast levers=DEVICE_RESIDENT,SUBBATCH,TRIMUL_PALLAS,AF_PALLAS_ATTN,PALLAS_MSA,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION tokens=600-705 queries=2", err.getvalue())
        self.assertTrue(report.tally_registered())
        self.assertEqual(colabfold_opt.status()["active"], True)
        # idempotent
        with redirect_stderr(io.StringIO()):
            again = colabfold_opt.enable("fast", queries=_stubs.queries(600, 705))
        self.assertEqual(again["active"], True); self.assertEqual(km.ENABLE_CALLS, [False])

    def test_off_applies_nothing(self):
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("off")
        self.assertFalse(rep["active"]); self.assertEqual(rep["mode"], "off"); self.assertEqual(rep["levers"], [])
        self.assertNotIn("AF_PALLAS_ATTN", os.environ); self.assertIsNone(self.kit_module())
        self.assertEqual(err.getvalue().strip(), "[colabfold-opt] NOT ACTIVE mode=off (stock: nothing applied)")

    def test_exact_applies_the_transfer_lever_alone(self):
        """`exact` = DEVICE_RESIDENT: the carried kit is neither imported nor switched on."""
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("exact", queries=_stubs.queries(199))
        self.assertTrue(rep["active"]); self.assertEqual((rep["levers"], rep["levers_applied"]), (["DEVICE_RESIDENT"], ["DEVICE_RESIDENT"]))
        self.assertNotIn("AF_PALLAS_ATTN", os.environ); self.assertIsNone(self.kit_module()); self.assertIsNone(rep["kit_state"]); self.assertIsNone(rep["patch_marker"])
        self.assertEqual(rep["lever_states"], {"DEVICE_RESIDENT": {"enabled": True, "calls": 0, "uploads": 0, "hits": 0, "fetched_bytes": 0, "fetch": "last", "deferred": 0, "deferred_fetches": 0}})
        self.assertIn("[colabfold-opt] ACTIVE mode=exact levers=DEVICE_RESIDENT tokens=199-199 queries=1 ", err.getvalue())
        self.assertTrue(getattr(self.mods["alphafold.model.model"].RunModel, "_device_resident", False))
        self.assertFalse(getattr(self.mods["alphafold.model.modules"].Attention, "_pallas_patched", False))

    def test_unknown_mode(self):
        with self.assertRaises(colabfold_opt.UnsupportedMode):
            colabfold_opt.enable("turbo")
        self.assertIs(colabfold_opt.UnsupportedMode, modes.UnsupportedMode)

    def test_gates_refuse_by_name(self):
        stack.gpu_info = lambda index=0: None
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertFalse(rep["active"]); self.assertIn("no NVIDIA GPU visible (nvidia-smi)", rep["reason"])
        self.assertIn("[colabfold-opt] NOT ACTIVE: no NVIDIA GPU visible (nvidia-smi)", err.getvalue())
        self.assertNotIn("AF_PALLAS_ATTN", os.environ); self.assertIsNone(self.kit_module())
        stack.reset_for_tests()
        stack.gpu_info = lambda index=0: {"name": "NVIDIA T4", "memory_mib": 15360, "compute_cap": 7.5, "count": 1}   # an untested part: ACTIVE with every lever, the capability named on the line — never a refusal
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertTrue(rep["active"]); self.assertIsNone(rep["reason"]); self.assertIn("compute capability 7.5 < 8.0 (NVIDIA T4): untested part, the levers engage", rep["notes"])
        self.assertEqual(rep["levers_applied"], ["DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"])
        self.assertIn(" ACTIVE mode=fast levers=DEVICE_RESIDENT,SUBBATCH,TRIMUL_PALLAS,AF_PALLAS_ATTN,PALLAS_MSA,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION ", err.getvalue()); self.assertIn(" notes=compute capability 7.5 < 8.0 (NVIDIA T4)", err.getvalue())
        stack.reset_for_tests(); sys.modules.pop("af2_pallas_attn", None); stack.gpu_info = lambda index=0: dict(_stubs.GPU_H100)
        stack.versions = lambda: dict(_stubs.VERSIONS, jax="0.8.1")                     # an untested jax: the same
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertTrue(rep["active"]); self.assertIn("jax 0.8.1 outside the tested range 0.5.3 - 0.7.x: untested, the levers engage", rep["notes"])
        self.assertEqual(rep["levers_applied"], ["DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"])
        self.assertIn(" notes=jax 0.8.1 outside the tested range", err.getvalue()); self.assertNotIn("NOT ACTIVE", err.getvalue())
        stack.reset_for_tests(); sys.modules.pop("af2_pallas_attn", None); stack.versions = lambda: dict(_stubs.VERSIONS)
        stack.pins_check = lambda: (["stock pin: colabfold 1.5.5 != 1.6.1 (stock/PINS.json)"], {})
        with redirect_stderr(io.StringIO()):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertIn("stock pin: colabfold 1.5.5 != 1.6.1", rep["reason"])
        stack.reset_for_tests(); _stubs.gates_restore(stack, self.gates); self.gates = _stubs.gates_pass(stack)
        stack.weights_check = lambda d, refresh=False: ([f"parameters under {d}: marker missing"], {})
        with redirect_stderr(io.StringIO()):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertIn("marker missing", rep["reason"])

    def test_a_gate_that_cannot_run_is_a_refusal_not_a_traceback(self):
        def boom():
            raise FileNotFoundError(2, "No such file or directory", "/nonexistent/stock/PINS.json")
        stack.pins_check = boom
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertFalse(rep["active"]); self.assertIn("gate failed: FileNotFoundError: [Errno 2] No such file or directory: '/nonexistent/stock/PINS.json'", rep["reason"])
        self.assertIn("[colabfold-opt] NOT ACTIVE: gate failed: FileNotFoundError", err.getvalue()); self.assertNotIn("Traceback", err.getvalue())
        stack.reset_for_tests()
        with redirect_stderr(io.StringIO()), self.assertRaises(colabfold_opt.ActivationError):
            colabfold_opt.enable("fast", queries=_stubs.queries(600), strict=True)

    def test_not_wired_switches_are_named_on_the_line(self):
        os.environ["AF_PALLAS_ATTN_PRECISE_BWD"] = "1"                                  # a backward-only kit switch no mode sets: named on the line, not refused
        self.addCleanup(os.environ.pop, "AF_PALLAS_ATTN_PRECISE_BWD", None)
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertTrue(rep["active"]); self.assertEqual(rep["switches_not_wired"]["AF_PALLAS_ATTN_PRECISE_BWD"], True)
        line = [ln for ln in err.getvalue().splitlines() if " ACTIVE mode=fast" in ln][0]
        self.assertTrue(line.endswith(" not_wired=AF_PALLAS_ATTN_PRECISE_BWD"), line)

    def test_widening_switch_is_refused_by_name(self):
        os.environ["AF_PALLAS_ATTN_ALL"] = "1"                                          # MSA-column / template attention through the kernel: outside the measured band
        self.addCleanup(os.environ.pop, "AF_PALLAS_ATTN_ALL", None)
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertFalse(rep["active"]); self.assertTrue(rep["reason"].startswith("AF_PALLAS_ATTN_ALL=1 refused:"), rep["reason"])
        self.assertEqual(rep["levers_applied"], []); self.assertFalse(registry.patch_marker_present())
        self.assertIn("NOT ACTIVE: AF_PALLAS_ATTN_ALL=1 refused:", err.getvalue())          # a gate refusal: the NOT ACTIVE line

    def test_reset_unregisters_the_tally(self):
        with redirect_stderr(io.StringIO()):
            colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertTrue(report.tally_registered())
        stack.reset_for_tests()
        self.assertFalse(report.tally_registered())

    def test_strict_raises_after_the_line(self):
        stack.gpu_info = lambda index=0: None
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(colabfold_opt.ActivationError):
            colabfold_opt.enable("fast", queries=_stubs.queries(600), strict=True)
        self.assertIn("NOT ACTIVE: no NVIDIA GPU visible", err.getvalue())

    def test_late_activation_refused(self):
        os.environ["AF_PALLAS_ATTN"] = "1"                                    # someone else's switch: the kit enables itself at import
        sys.path.insert(0, os.path.join(self.kit, "af2_pallas_flash"))
        import af2_pallas_attn as km  # noqa: F401
        self.assertTrue(km._STATE["enabled"])
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertFalse(rep["active"]); self.assertIn("late activation refused: AF_PALLAS_ATTN reports enabled=True already", rep["reason"])

    def test_target_gpu_note(self):
        os.environ["MODEL_OPT_TARGET_GPU"] = "A100"
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertTrue(rep["active"]); self.assertEqual(rep["notes"], ["MODEL_OPT_TARGET_GPU=A100 but the visible GPU is NVIDIA H100 80GB HBM3"])
        self.assertIn(" notes=MODEL_OPT_TARGET_GPU=A100 but the visible GPU is NVIDIA H100 80GB HBM3", err.getvalue())

    def test_no_queries_hooks_the_run(self):
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.enable("fast")
        self.assertFalse(rep["active"]); self.assertTrue(rep["hooked"]); self.assertIn("decided at the colabfold.batch.run call", rep["reason"])
        self.assertTrue(getattr(self.mods["colabfold.batch"].run, "_colabfold_opt_hook", False))
        self.assertIsNone(self.kit_module()); self.assertNotIn("AF_PALLAS_ATTN", os.environ)
        out = os.path.join(self.tmp, "work"); os.environ[manifest.ENV_WORK_DIR] = out   # the launch's work directory as `pred` names it: the manifest goes there, never under the run's result directory
        with redirect_stderr(err):
            ret = self.mods["colabfold.batch"].main(queries=_stubs.queries(600, 800), result_dir=os.path.join(self.tmp, "res"), data_dir="/d")
        self.assertEqual(ret, "stock-ran"); self.assertEqual(len(self.log), 1)          # the stock run followed the activation
        self.assertTrue(colabfold_opt.status()["active"]); self.assertEqual(colabfold_opt.status()["data_dir"], "/d")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "res", manifest.FILENAME))); os.environ.pop(manifest.ENV_WORK_DIR)
        man = manifest.read(out)
        self.assertTrue(man["active"]); self.assertEqual(man["mode"], "fast"); self.assertEqual(man["tokens_min"], 600); self.assertEqual(man["route"], "hook")
        self.assertEqual(man["kit_state_at_activation"], {"enabled": True, "calls": 0, "fallbacks": 0})


class TestCheck(Base):
    def test_dry_run_touches_nothing(self):
        err = io.StringIO()
        with redirect_stderr(err):
            rep = colabfold_opt.check("fast")
        self.assertFalse(rep["active"]); self.assertNotIn("would_refuse", rep); self.assertEqual(rep["levers"], ["DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"])
        self.assertNotIn("AF_PALLAS_ATTN", os.environ); self.assertIsNone(self.kit_module()); self.assertEqual(sys.path, self.path_saved)
        self.assertIn("[colabfold-opt] DRY-RUN mode=fast levers=DEVICE_RESIDENT,SUBBATCH,TRIMUL_PALLAS,AF_PALLAS_ATTN,PALLAS_MSA,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION tokens=none queries=none colabfold=1.6.1", err.getvalue())
        self.assertEqual(colabfold_opt.status(), {"active": False, "reason": "no activation in this process"})
        stack.gpu_info = lambda index=0: None
        with redirect_stderr(err):
            rep = colabfold_opt.check("fast")
        self.assertIn("no NVIDIA GPU visible", rep["would_refuse"])                    # no device at all: nothing of a kit mode can run — the one hardware refusal
        stack.gpu_info = lambda index=0: {"name": "Tesla T4", "memory_mib": 15360, "compute_cap": 7.5, "count": 1}
        stack.versions = lambda: dict(_stubs.VERSIONS, jax="0.9.1")
        try:
            err2 = io.StringIO()
            with redirect_stderr(err2):
                rep = colabfold_opt.check("fast")
        finally:
            stack.versions = lambda: dict(_stubs.VERSIONS)
        self.assertNotIn("would_refuse", rep)                                            # the environment is never a refusal: an untested part and an untested jax are notes on the line
        self.assertEqual([n for n in rep["notes"] if n.startswith(("compute capability", "jax "))],
                         ["jax 0.9.1 outside the tested range 0.5.3 - 0.7.x: untested, the levers engage",
                          "compute capability 7.5 < 8.0 (Tesla T4): untested part, the levers engage"])
        line = err2.getvalue().splitlines()[-1]
        self.assertIn("[colabfold-opt] DRY-RUN mode=fast levers=DEVICE_RESIDENT,SUBBATCH,TRIMUL_PALLAS,AF_PALLAS_ATTN,PALLAS_MSA,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION ", line); self.assertIn(" notes=", line); self.assertIn("compute capability 7.5 < 8.0 (Tesla T4)", line)
        self.assertEqual((stack.gate_gpu({"name": "Tesla T4", "memory_mib": 1, "compute_cap": 7.5, "count": 1}), stack.gate_jax("0.4.0")), (None, None))
        with redirect_stderr(err):
            self.assertEqual(colabfold_opt.check("off")["mode"], "off")
        self.assertIn("NOT ACTIVE mode=off (stock: nothing applied)", err.getvalue())


class TestHook(Base):
    def test_hook_sees_the_run_and_refuses_strictly(self):
        cb = self.mods["colabfold.batch"]
        stock_run = cb.run
        run = stack.hook_run("fast", strict=True, trigger="colabfold.batch", module=cb)
        self.assertIs(cb.run, run); self.assertIs(run._colabfold_opt_stock, stock_run)
        self.assertIs(stack.hook_run("fast", module=cb), run)                # installed once
        out = os.path.join(self.tmp, "w1"); os.environ[manifest.ENV_WORK_DIR] = out    # the launch's work directory as `pred` names it
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            cb.main(queries=_stubs.queries(199), result_dir=os.path.join(self.tmp, "r1"), data_dir="/d")
        self.assertEqual(cm.exception.code, 3); self.assertEqual(len(self.log), 1)   # activated at the run call (any input size), the stock run ran, the kernel made no call: partial, exit 3 by name after it
        man = manifest.read(out)
        self.assertEqual((man["active"], man["mode"], man["tokens_min"], man["route"]), (True, "fast", 199, "env"))
        self.assertIn("[colabfold-opt] ACTIVE mode=fast levers=DEVICE_RESIDENT,SUBBATCH,TRIMUL_PALLAS,AF_PALLAS_ATTN,PALLAS_MSA,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION tokens=199-199 queries=1 ", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "r1", manifest.FILENAME)))   # nothing of the package under the run's result directory
        os.environ.pop(manifest.ENV_WORK_DIR)
        self.fresh_process(); err = io.StringIO()                                      # without a work directory (a bare env-route run): no manifest anywhere, the lines are the record
        with redirect_stderr(err), self.assertRaises(SystemExit):
            cb.main(queries=_stubs.queries(600), result_dir=os.path.join(self.tmp, "r1b"), data_dir="/d")
        self.assertEqual(len(self.log), 2); self.assertIn(" ACTIVE mode=fast ", err.getvalue()); self.assertEqual(os.listdir(os.path.join(self.tmp, "r1b")) if os.path.isdir(os.path.join(self.tmp, "r1b")) else [], [])
        self.fresh_process(); stack.gpu_info = lambda index=0: None
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            cb.main(queries=_stubs.queries(600), result_dir=os.path.join(self.tmp, "r2"), data_dir="/d")
        self.assertEqual(cm.exception.code, 3); self.assertEqual(len(self.log), 2)   # refused: exit 3, the stock run never started

    def test_post_run_partial_exits_by_name(self):
        """The env route owns the process exit: an active run whose kernel engaged no call by the time the stock `run` returns (the stub's
        Attention is never called: calls=0) prints the house line and exits 3 — a mode is all of its levers, there is no opt-out; under `pred`'s
        launch id the model process exits 3 the same (pred's verdict reads rc 3 + `partial`); the explicit non-strict route prints nothing and
        exits nothing (the manifest's `partial` at exit is its record)."""
        cb = self.mods["colabfold.batch"]
        stack.hook_run("fast", strict=True, trigger="colabfold.batch", module=cb)
        out = os.path.join(self.tmp, "p1"); err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            cb.main(queries=_stubs.queries(199), result_dir=out, data_dir="/d")
        self.assertEqual((cm.exception.code, len(self.log)), (3, 1))                  # the stock run happened; the exit is by name after it
        self.assertIn(" ACTIVE mode=fast levers=DEVICE_RESIDENT,SUBBATCH,TRIMUL_PALLAS,AF_PALLAS_ATTN,PALLAS_MSA,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION tokens=199-199 queries=1 ", err.getvalue())
        self.assertIn("[colabfold-opt] NOT ACTIVE: partial activation — DEVICE_RESIDENT: EXIT calls=0 (no model call went through the lever); SUBBATCH: EXIT calls=0 (no model call went through the lever); TRIMUL_PALLAS: EXIT calls=0 (no model call went through the lever); AF_PALLAS_ATTN: EXIT calls=0 (the kernel ran no attention call); PALLAS_MSA: EXIT calls=0 (no model call went through the lever); TRIATTN_XLA: EXIT calls=0 (no model call went through the lever); MSA_COL_CUDNN: EXIT calls=0 (no model call went through the lever); TEMPL_DEDUP: EXIT calls=0 (no model call went through the lever); TRANSITION: EXIT calls=0 (no model call went through the lever); exit 3\n", err.getvalue())
        self.fresh_process(); os.environ[manifest.ENV_LAUNCH_ID] = "abc"; err = io.StringIO()   # launched by pred: the model process exits 3 the same, its line in the log
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            cb.main(queries=_stubs.queries(199), result_dir=os.path.join(self.tmp, "p3"), data_dir="/d")
        self.assertEqual(cm.exception.code, 3); self.assertIn("[colabfold-opt] NOT ACTIVE: partial activation — ", err.getvalue())
        os.environ.pop(manifest.ENV_LAUNCH_ID)
        self.fresh_process(); err = io.StringIO()
        cb2 = _stubs.make_colabfold(self.log)["colabfold.batch"]                     # a fresh stub module: the explicit, non-strict hook
        with redirect_stderr(err):
            stack.hook_run("fast", strict=False, module=cb2)                            # the explicit route: no exit, no line; the report and the manifest carry the state
            self.assertEqual(cb2.main(queries=_stubs.queries(199), result_dir=os.path.join(self.tmp, "p4"), data_dir="/d"), "stock-ran")
        self.assertNotIn("PARTIAL", err.getvalue()); self.assertNotIn("NOT ACTIVE", err.getvalue())

    def fresh_process(self):
        """The next activation as a new process would see it: the package state reset and the stub kit's module dropped (its _STATE
        `enabled` would otherwise read as an activation from outside the package — the late-activation rule)."""
        stack.reset_for_tests(); sys.modules.pop("af2_pallas_attn", None)

    def test_activation_during_a_run_is_late(self):
        stack._STATE["running"] = 1
        with redirect_stderr(io.StringIO()):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertIn("a prediction run is in progress", rep["reason"])
        stack._STATE["running"] = 0


class TestExitTally(unittest.TestCase):
    """The EXIT line at interpreter exit, in subprocesses: the kit's counters read in memory at exit; `none` when the kit was never imported."""

    def run_py(self, code, env_extra=None):
        tmp = tempfile.mkdtemp()
        kit = _stubs.make_kit_dir(tmp)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN")) and k != "JAX_COMPILATION_CACHE_DIR"}   # an image that presets the compile cache dir would make XLA_CACHE `kept`: these lines lock the unset case
        env.update({"COLABFOLD_OPT_KIT": kit, "PYTHONPATH": os.pathsep.join([OPT, CORE_ROOT]), "PYTHONDONTWRITEBYTECODE": "1"}); env.update(env_extra or {})   # the package and the shared core, importable wherever this interpreter found them
        pre = textwrap.dedent("""
            import sys
            from colabfold_opt.tests import _stubs
            from colabfold_opt import stack
            mods, saved = _stubs.install(); _stubs.gates_pass(stack)
            """)
        r = subprocess.run([sys.executable, "-c", pre + textwrap.dedent(code)], capture_output=True, text=True, env=env, cwd=tmp)
        self.addCleanup(shutil.rmtree, tmp, True)
        return r

    def test_tally_printed_at_exit_with_the_final_counters(self):
        r = self.run_py("""
            import colabfold_opt
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600))
            import af2_pallas_attn as km
            km._STATE["calls"] += 7
            print("active", rep["active"])
            """)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("active True", r.stdout)
        tail = r.stderr.strip().splitlines()[-12:]                                     # the carried kit's `_STATE` line, then one LEVER line per lever (the table's nine + the axis + the deployment lever)
        self.assertEqual(tail, ["[colabfold-opt] EXIT _STATE enabled=True calls=7 fallbacks=0",
                                "[colabfold-opt] LEVER name=DEVICE_RESIDENT state=on impl=colabfold_opt.device_resident origin=kit strategy=F6.host_sync_elimination calls=0 uploads=0 hits=0 fetched_bytes=0 fetch=last deferred=0 deferred_fetches=0",
                                "[colabfold-opt] LEVER name=SUBBATCH state=on impl=opt_core.jax_design.subbatch_policy origin=core strategy=F7.jax_subbatch calls=0 value=128 source=auto:fits tokens=600 stock_value=4 est_gib=3.13 device_gib=79.65",
                                "[colabfold-opt] LEVER name=TRIMUL_PALLAS state=on impl=opt_core.kernels.pallas.serve:triangle_multiplication origin=core strategy=F2.trimul calls=0 fallbacks=0 fallback_by=none word=fast rows=none cells=none shapes=none precision=none cc=9.0 core=stub jax=0.5.3",
                                "[colabfold-opt] LEVER name=AF_PALLAS_ATTN state=on impl=opt_core.kernels.pallas origin=core strategy=F1.flash_triatt calls=7 fallbacks=0 word=fast provider_calls=0 served=none sites=none cells=none provider=" + PROVIDER_V,
                                "[colabfold-opt] LEVER name=PALLAS_MSA state=on impl=opt_core.kernels.pallas:attention origin=core strategy=F5.flash_attn_dense word=fast served=none cells=none calls=0 fallbacks=0 fallback_by=none shapes=none cc=none",
                                "[colabfold-opt] LEVER name=TRIATTN_XLA state=on impl=opt_core.kernels.triattn_xla origin=core strategy=F1.flash_triatt calls=0 fallbacks=0 fallback_by=none rows=none sites=none shapes=none word=fast cc=9.0 core=1.1.0",
                                "[colabfold-opt] LEVER name=MSA_COL_CUDNN state=on impl=opt_core.kernels.pallas:attention origin=core strategy=F5.sdpa_cudnn word=fast served=none cells=none calls=0 fallbacks=0 fallback_by=none shapes=none precision=none cc=9.0",
                                "[colabfold-opt] LEVER name=TEMPL_DEDUP state=on impl=colabfold_opt.templ_dedup origin=kit strategy=LOCAL.colabfold.template_dedup calls=0 predicts=0 templates=0 embedded=0 reused=0 unique_hist=none fallbacks=0 fallback_by=none sites=multimer",
                                "[colabfold-opt] LEVER name=TRANSITION state=on impl=opt_core.kernels.pallas.serve:transition origin=core strategy=LOCAL.fused_transition calls=0 fallbacks=0 fallback_by=none word=fast rows=none served=0 cells=none shapes=none precision=none cc=9.0 core=stub jax=0.5.3",
                                "[colabfold-opt] LEVER name=ROWPAIR state=off reason=not_in_mode impl=opt_core.mem.rowpair_jax origin=core strategy=F7.tensor_parallel",
                                "[colabfold-opt] LEVER name=XLA_CACHE state=skipped reason=no_cache_root impl=opt_core.capture.xla_cache origin=core strategy=F3.jit_cache_keyed"], r.stderr)

    @unittest.skipUnless(all(importlib.util.find_spec(n) is not None for n in ("opt_core.mem.ngpu", "opt_core.mem.rowpair_jax")), "needs the axis' core modules")
    def test_big_p1_activation_leaves_predict_structure_stock(self):
        """The ACTIVATION route (`enable('big')` at the default P = 1, the axis not in the process's lever set): nothing of the axis is
        installed — `colabfold.batch.predict_structure` is stock's own object, a prediction with template slots (real or mock, asked or
        not) reaches stock as stock runs it, and the axis' exit line reads `state=off reason=n_gpu=1`."""
        r = self.run_py("""
            import colabfold.batch as B
            orig = B.predict_structure
            import colabfold_opt
            rep = colabfold_opt.enable("big", queries=_stubs.queries(600))
            from colabfold_opt import big
            print("active", rep["active"], rep["n_gpu"], rep["levers_applied"], B.predict_structure is orig)
            print("plain", B.predict_structure(prefix="p", feature_dict={"template_all_atom_mask": [[[0.0] * 37] * 5] * 4}, use_templates=False))
            print("mock", B.predict_structure(prefix="q1", feature_dict={"template_all_atom_mask": [[[0.0] * 37] * 5] * 4}, use_templates=True))
            """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("active True 1 ['DEVICE_RESIDENT', 'SUBBATCH', 'TRIMUL_PALLAS', 'AF_PALLAS_ATTN', 'PALLAS_MSA', 'TRIATTN_XLA', 'MSA_COL_CUDNN', 'TEMPL_DEDUP', 'TRANSITION'] True", r.stdout, r.stdout + r.stderr)
        self.assertIn("plain stock-predicted", r.stdout); self.assertIn("mock stock-predicted", r.stdout); self.assertNotIn("NOT ACTIVE", r.stderr)
        self.assertIn("LEVER name=ROWPAIR state=off reason=n_gpu=1 ", r.stderr); self.assertNotIn("template_guard=", r.stderr)

    def test_exact_prints_its_lever_lines_only(self):
        r = self.run_py("""
            import colabfold_opt
            rep = colabfold_opt.enable("exact", queries=_stubs.queries(199))
            m = mods["alphafold.model.model"].RunModel(params={"w": [1.0]})
            out = m.apply(m.params, None, {"prev": {}})
            print("active", rep["active"], sorted(out))
            """)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("active True ['aatype', 'distogram', 'mean_plddt', 'prev', 'ranking_confidence', 'structure_module', 'tol']", r.stdout)
        tail = r.stderr.strip().splitlines()[-11:]                                     # no `_STATE` line: the kernel is not in the mode
        self.assertRegex(tail[0], r"^\[colabfold-opt\] LEVER name=DEVICE_RESIDENT state=on impl=colabfold_opt.device_resident origin=kit strategy=F6.host_sync_elimination calls=1 uploads=[0-9]+ hits=0 fetched_bytes=[0-9]+ fetch=last deferred=[0-9]+ deferred_fetches=[0-9]+$")
        self.assertEqual(tail[1], "[colabfold-opt] LEVER name=SUBBATCH state=off reason=not_in_mode impl=opt_core.jax_design.subbatch_policy origin=core strategy=F7.jax_subbatch")
        self.assertEqual(tail[2], "[colabfold-opt] LEVER name=TRIMUL_PALLAS state=off reason=not_in_mode impl=opt_core.kernels.pallas.serve:triangle_multiplication origin=core strategy=F2.trimul")
        self.assertEqual(tail[3], "[colabfold-opt] LEVER name=AF_PALLAS_ATTN state=off reason=not_in_mode impl=opt_core.kernels.pallas origin=core strategy=F1.flash_triatt")
        self.assertEqual(tail[4], "[colabfold-opt] LEVER name=PALLAS_MSA state=off reason=not_in_mode impl=opt_core.kernels.pallas:attention origin=core strategy=F5.flash_attn_dense")
        self.assertEqual(tail[5], "[colabfold-opt] LEVER name=TRIATTN_XLA state=off reason=not_in_mode impl=opt_core.kernels.triattn_xla origin=core strategy=F1.flash_triatt")
        self.assertEqual(tail[6], "[colabfold-opt] LEVER name=MSA_COL_CUDNN state=off reason=not_in_mode impl=opt_core.kernels.pallas:attention origin=core strategy=F5.sdpa_cudnn")
        self.assertEqual(tail[7], "[colabfold-opt] LEVER name=TEMPL_DEDUP state=off reason=not_in_mode impl=colabfold_opt.templ_dedup origin=kit strategy=LOCAL.colabfold.template_dedup")
        self.assertEqual(tail[8], "[colabfold-opt] LEVER name=TRANSITION state=off reason=not_in_mode impl=opt_core.kernels.pallas.serve:transition origin=core strategy=LOCAL.fused_transition")
        self.assertEqual(tail[9], "[colabfold-opt] LEVER name=ROWPAIR state=off reason=not_in_mode impl=opt_core.mem.rowpair_jax origin=core strategy=F7.tensor_parallel")
        self.assertEqual(tail[10], "[colabfold-opt] LEVER name=XLA_CACHE state=skipped reason=no_cache_root impl=opt_core.capture.xla_cache origin=core strategy=F3.jit_cache_keyed")

    def test_manifest_records_the_exit_state(self):
        r = self.run_py("""
            import os, json, colabfold_opt
            out = os.path.abspath("work"); os.environ["COLABFOLD_OPT_WORK_DIR"] = out     # the launch's work directory as `pred` names it
            colabfold_opt.enable("fast")
            mods["colabfold.batch"].main(queries=_stubs.queries(700), result_dir=os.path.abspath("res"), data_dir="/d")
            import af2_pallas_attn as km
            km._STATE["calls"] = 12
            print(out)
            """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("EXIT _STATE enabled=True calls=12 fallbacks=0", r.stderr)
        man = manifest.read(r.stdout.strip().splitlines()[-1])
        self.assertEqual(man["kit_state_exit"], {"enabled": True, "calls": 12, "fallbacks": 0})
        self.assertEqual(man["kit_state_at_activation"], {"enabled": True, "calls": 0, "fallbacks": 0}); self.assertTrue(man["active"])
