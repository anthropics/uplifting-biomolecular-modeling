"""The env route: the finder installed only for a MOSAIC_OPT mode word, firing once after the trigger package's body, exit 3 on refusal
(stock never runs silently), the row's call-site levers (P2, P3: the driver's own replacements) stepping aside BY NAME on this route —
named on the ACTIVE line's row token, never partial, never an exit — for exact, fast and big, the partial verdict kept for a lever the
route should apply and did not, the kit driver never levered by the hook, nothing imported at interpreter start beyond the package (the
.pth text itself: test_core_adoption.py; the real .pth on an absent / older core: test_core_gate_routes.py). The documented form end to end
(`MOSAIC_OPT=fast python -c 'import mosaic'` in a fresh interpreter) needs the pinned upstream packages and a GPU for the activation's own
gates: a quick check on a GPU machine, not a case of this CPU file — the hook's path is exercised here through the real finder on a stand-in `mosaic` package."""
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

import pytest

from . import _stubs, core_src
from .test_ablation_switch import fake_modules                     # every per-step lever served by the recording fake: fast / big install in-process without the stack
from mosaic_opt import _autoload, levers, report, stack
import mosaic_opt


class TestAutoloadInstall(unittest.TestCase):
    def setUp(self):
        self.state = _stubs.StubState(); self.state.restore()

    def tearDown(self):
        self.state.restore()

    def test_install_only_for_a_mode(self):
        self.assertIsNone(_autoload.install({}))
        self.assertIsNone(_autoload.install({"MOSAIC_OPT": "off"}))
        self.assertIsNone(_autoload.install({"MOSAIC_OPT": ""}))
        for word, words in (("bogus", "NOT ACTIVE: unknown MOSAIC_OPT='bogus'; expected one of ('fast', 'exact', 'big', 'off'); exit 3"),):   # every tier word has a row on this version: only an unknown word refuses
            f = _autoload.install({"MOSAIC_OPT": word})                                    # a word the hook does not serve REFUSES at the trigger (exit 3), never runs stock
            self.assertIsInstance(f, _autoload.Finder); self.assertIs(sys.meta_path[0], f); self.assertIn(words, f.refuse)
            err = io.StringIO()
            with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                f._fire("mosaic")                                                           # what the trigger's loader calls once the upstream package's body has run
            self.assertEqual(cm.exception.code, 3); self.assertIn(words, err.getvalue()); self.assertNotIn(f, sys.meta_path)
        f = _autoload.install({"MOSAIC_OPT": "exact"})
        self.assertIsInstance(f, _autoload.Finder); self.assertIs(sys.meta_path[0], f)
        self.assertIs(_autoload.install({"MOSAIC_OPT": "exact"}), f)                    # idempotent
        f.armed = False; sys.meta_path.remove(f)

    def test_trigger_is_the_upstream_package(self):
        self.assertEqual(_autoload.TRIGGERS, ("mosaic",))
        self.assertEqual(_autoload.MODES, ("fast", "exact", "big", "off"))


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestAutoloadFires(unittest.TestCase):
    def setUp(self):
        self.state = _stubs.StubState(); self.state.restore()
        self.mp = pytest.MonkeyPatch()
        self.tmp = tempfile.mkdtemp(prefix="mosaic_opt_auto_")
        for k in ("XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "MOSAIC_OPT", "MOSAIC_OPT_CACHE_DIR", "MOSAIC_OPT_CACHE_ROOT", "MOSAIC_CACHE_DIR"):
            self.mp.delenv(k, raising=False)
        self.mp.setenv("MOSAIC_OPT_HOME", _stubs.TREE)
        self.fake = _stubs.FakeReproCache()
        self.mp.setattr(stack, "p1_lever", lambda: self.fake)
        sys.path.insert(0, _stubs.fake_mosaic_package(self.tmp))

    def tearDown(self):
        self.mp.undo(); self.state.restore()

    def _fire(self, mode):
        """Arm the hook for `mode` and import the stand-in upstream package: (finder, status report, stderr text). No SystemExit expected."""
        f = _autoload.install({"MOSAIC_OPT": mode})
        err = io.StringIO()
        with redirect_stderr(err):
            import mosaic  # noqa: F401  (the fake upstream package: its body runs, then the hook fires once)
        self.assertEqual(f.fired, "mosaic"); self.assertNotIn(f, sys.meta_path)
        return f, mosaic_opt.status(), err.getvalue()

    def _active_line(self, text):
        lines = [l for l in text.splitlines() if l.startswith("[mosaic-opt] ACTIVE ")]
        self.assertEqual(len(lines), 1, text)
        self.assertNotIn(" partial=", text); self.assertNotIn("NOT ACTIVE", text); self.assertNotIn("PARTIAL allowed", text)   # a step-aside is not a partial activation: no verdict, no exit
        return lines[0]

    def test_exact_fires_after_the_trigger_body_applies_p1_and_names_p2_p3_aside(self):
        """MOSAIC_OPT=exact: the hook applies P1 after the trigger body; P2 and P3 — call-site replacements only the kit driver makes — step aside
        BY NAME on the row token (`-aside[P2:driver_only,P3:driver_only]`): `unavailable=none`, not partial, no verdict line, and the process
        continues (no SystemExit) whether or not MOSAIC_OPT_ALLOW_PARTIAL is set."""
        _stubs.stub_gates(self.mp)
        self.mp.setenv("MOSAIC_OPT_CACHE_DIR", os.path.join(self.tmp, "p1")); self.mp.delenv("MOSAIC_OPT_ALLOW_PARTIAL", raising=False)
        f, rep, text = self._fire("exact")
        self.assertTrue(rep["active"]); self.assertEqual(rep["trigger"], "mosaic"); self.assertEqual(rep["levers_applied"], ["P1"])
        self.assertEqual((rep["partial"], rep["levers_unavailable"], rep["levers_driver_only"], rep["levers_aside"]),
                         (False, [], ["P2", "P3"], {"P2": "driver_only", "P3": "driver_only"}))
        self.assertIs(rep["allow_partial"], False); self.assertNotIn("unavailable_reason", rep); self.assertIn("load_stock_fast_init", rep["driver_only_reason"])
        self.assertEqual(report.partial_levers(rep), []); self.assertEqual(self.fake.calls, [os.path.join(self.tmp, "p1")])
        line = self._active_line(text)
        self.assertIn("ACTIVE mode=exact route=in-process row=C_p1warm_p2[P1]-aside[P2:driver_only,P3:driver_only] (stood in for in-process by pcc.enable()) "
                      "levers=P1 unavailable=none p1=dump:", line)
        self.assertIn("P2 steps aside: a call-site replacement only the kit driver applies (`mosaic-opt design`); this route runs the row's other levers", line)   # notes=
        self.assertTrue(rep["row_line"].startswith("C_p1warm_p2[P1]-aside[P2:driver_only,P3:driver_only] "), rep["row_line"])

    def test_fast_installs_the_per_step_levers_in_process_and_p2_steps_aside(self):
        """MOSAIC_OPT=fast without MOSAIC_OPT_CACHE_ROOT: the per-step levers install in this process (the ONE installer), P1 steps aside by name
        (no cache root: free text, off the row token, `p1=none(…)`), P2 by name as `P2:driver_only` — every lever accounted, nothing partial, no exit."""
        _stubs.stub_gates(self.mp)
        self.mp.delenv("MOSAIC_OPT_ALLOW_PARTIAL", raising=False)
        with fake_modules():
            f, rep, text = self._fire("fast")
            per_step = ["E1", "P6", "K1", "E10", "F6", "F8", "F9", "P7"]
            self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["levers_applied"], per_step); self.assertEqual(levers.installed()["levers"], per_step)
            self.assertEqual((rep["partial"], rep["levers_unavailable"], rep["levers_driver_only"]), (False, [], ["P2"]))
            self.assertEqual(rep["levers_aside"], {"P1": stack.P1_ASIDE_REASON, "P2": "driver_only"}); self.assertEqual(self.fake.calls, [])   # no cache root: P1 aside, pcc never called
            line = self._active_line(text)
            self.assertIn("ACTIVE mode=fast route=in-process row=T_fast[E1,P6,K1,E10,F6,F8,F9,P7]-aside[P1,P2:driver_only] (stood in for in-process by levers.install()) "
                          "levers=E1,P6,K1,E10,F6,F8,F9,P7 unavailable=none p1=none(MOSAIC_OPT_CACHE_ROOT_unset) ", line)
            self.assertIn("[mosaic-opt] LEVER name=E1 state=on ", text); self.assertIn("[mosaic-opt] LEVER name=P5 state=off reason=mode:fast", text)

    def test_big_installs_p5_in_process_and_p2_steps_aside(self):
        """`big` in-process (`enable("big")`, the same body the hook calls): P5 is a per-step lever — it installs here like the rest; only P2 is
        the driver's; with a cache root P1 applies in its transparent form (`p1=off:<root>/campaign_big/…` — the route's own directory)."""
        _stubs.stub_gates(self.mp)
        self.mp.setenv("MOSAIC_OPT_CACHE_ROOT", os.path.join(self.tmp, "root")); self.mp.delenv("MOSAIC_OPT_ALLOW_PARTIAL", raising=False)
        with fake_modules():
            err = io.StringIO()
            with redirect_stderr(err):
                rep = mosaic_opt.enable("big", strict=True)                                # strict: a refusal would raise ActivationError — none does
            per_step = ["E1", "P6", "K1", "E10", "F6", "F8", "F9", "P5", "P7"]
            self.assertTrue(rep["active"]); self.assertEqual(rep["levers_applied"], ["P1"] + per_step)
            self.assertEqual((rep["partial"], rep["levers_unavailable"], rep["levers_driver_only"], rep["levers_aside"]), (False, [], ["P2"], {"P2": "driver_only"}))
            self.assertEqual(len(self.fake.calls), 1); self.assertTrue(self.fake.calls[0].startswith(os.path.join(self.tmp, "root")))
            line = self._active_line(err.getvalue())
            self.assertIn("ACTIVE mode=big route=in-process row=U_big[P1,E1,P6,K1,E10,F6,F8,F9,P5,P7]-aside[P2:driver_only] (stood in for in-process by pcc.enable() + levers.install()) "
                          "levers=P1,E1,P6,K1,E10,F6,F8,F9,P5,P7 unavailable=none p1=off:", line)
            dry = stack.activate("big", dry_run=True)                                          # `check`, the in-process plan: the same words, nothing partial
            self.assertEqual((dry["partial"], dry["levers_unavailable"], dry["levers_driver_only"], dry["levers_planned"]), (False, [], ["P2"], ["P1"] + per_step))
            self.assertIn("DRY-RUN mode=big route=in-process row=U_big[P1,E1,P6,K1,E10,F6,F8,F9,P5,P7]-aside[P2:driver_only] levers=P1,E1,P6,K1,E10,F6,F8,F9,P5,P7 p1=", report.activation_line(dry))
            self.assertNotIn(" partial=", report.activation_line(dry))

    def test_a_lever_the_route_should_apply_and_did_not_is_partial_and_fails_closed(self):
        """`partial` keeps ONE meaning at the hook: a report that comes back partial (a lever this route should apply and did not) prints the one
        verdict and the process exits 3; MOSAIC_OPT_ALLOW_PARTIAL=1 records the opt-out (`PARTIAL allowed:`) and proceeds. A step-aside never
        reaches this gate (the cases above)."""
        rep = {"active": True, "mode": "fast", "route": "in-process", "partial": True, "levers_unavailable": ["K1"], "unavailable_reason": "stub: K1 not applied on this route",
               "levers_applied": ["E1"], "row_line": "T_fast[E1,K1]", "p1": {}, "upstream": {}, "gpu": None}
        self.mp.setattr(mosaic_opt, "enable", lambda mode, **kw: dict(rep))
        self.mp.delenv("MOSAIC_OPT_ALLOW_PARTIAL", raising=False)
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            _autoload.enable("fast", strict=True, trigger="mosaic")
        self.assertEqual(cm.exception.code, 3)
        self.assertIn("[mosaic-opt] NOT ACTIVE: partial activation — K1 — the plan is partial on this box (stub: K1 not applied on this route); exit 3 (--allow-partial records and proceeds)", err.getvalue())
        self.mp.setenv("MOSAIC_OPT_ALLOW_PARTIAL", "1")
        err = io.StringIO()
        with redirect_stderr(err):
            out = _autoload.enable("fast", strict=True, trigger="mosaic")
        self.assertIs(out["allow_partial"], True); self.assertEqual(report.partial_levers(out), ["K1"])
        self.assertIn("[mosaic-opt] PARTIAL allowed: K1 — the plan is partial on this box (stub: K1 not applied on this route) (--allow-partial, recorded)", err.getvalue())
        self.assertIn(" partial=K1 ", report.activation_line(out) + " "); self.assertIn(" allow_partial=1", report.activation_line(out))

    def test_refusal_exits_3_instead_of_running_stock(self):
        _stubs.stub_gates(self.mp, gpu={"name": None, "cc": None, "sm": None, "probe": "none"})
        self.mp.setenv("MOSAIC_OPT_CACHE_DIR", os.path.join(self.tmp, "p1"))
        _autoload.install({"MOSAIC_OPT": "exact"})
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            import mosaic  # noqa: F401
        self.assertEqual(cm.exception.code, 3)
        self.assertIn("NOT ACTIVE: no GPU visible", err.getvalue())

    def test_the_kit_driver_is_never_levered_by_the_hook(self):
        """The kit driver run by hand with MOSAIC_OPT=exact exported: the hook refuses (exit 3) instead of levering the stock arm silently."""
        _stubs.stub_gates(self.mp)
        self.mp.setenv("MOSAIC_OPT_CACHE_DIR", os.path.join(self.tmp, "p1"))
        self.assertIsNone(_autoload.driver_process_refusal(["my_script.py"]))
        self.assertIn("public_design_run.py", _autoload.driver_process_refusal([os.path.join(_stubs.KIT, "tools", "public_design_run.py")]))
        self.mp.setattr(sys, "argv", [os.path.join(_stubs.KIT, "tools", "public_design_run.py"), "--tag", "A_stock1"])
        _autoload.install({"MOSAIC_OPT": "exact"})
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            import mosaic  # noqa: F401
        self.assertEqual(cm.exception.code, 3)
        self.assertIn("NOT ACTIVE: MOSAIC_OPT is set but this process is the kit driver", err.getvalue())
        self.assertEqual(self.fake.calls, []); self.assertFalse(mosaic_opt.status()["active"])
        self.assertEqual(_autoload.DRIVER_BASENAME, os.path.basename(stack.KIT_MARKER))

    def test_enable_disarms_the_finder(self):
        _stubs.stub_gates(self.mp)
        self.mp.setenv("MOSAIC_OPT_CACHE_DIR", os.path.join(self.tmp, "p1"))
        f = _autoload.install({"MOSAIC_OPT": "exact"})
        with redirect_stderr(io.StringIO()):
            mosaic_opt.enable("exact")
        self.assertNotIn(f, sys.meta_path); self.assertFalse(f.armed)


class TestInterpreterStart(unittest.TestCase):
    def test_import_mosaic_opt_imports_nothing_heavy(self):
        code = "import sys, mosaic_opt, mosaic_opt._autoload; print(sorted(m for m in sys.modules if m.split('.')[0] in ('jax','jaxlib','torch','numpy','mosaic','joltz','boltz')))"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "MOSAIC_OPT": "exact", "PYTHONPATH": _stubs.OPT_DIR + os.pathsep + core_src()})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")
        code = "import sys, mosaic_opt, mosaic_opt._autoload; print(sorted(m for m in sys.modules if m.startswith('opt_core')))"   # under the variable: the core's finder module alone
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "MOSAIC_OPT": "exact", "PYTHONPATH": _stubs.OPT_DIR + os.pathsep + core_src()})
        self.assertEqual(out.stdout.strip(), "['opt_core', 'opt_core.autoload']", out.stderr)
        code = "import sys, mosaic_opt, mosaic_opt._autoload; print(sorted(m for m in sys.modules if m.startswith('opt_core')))"   # unset: nothing of the core in a stock process
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**{k: v for k, v in os.environ.items() if k != 'MOSAIC_OPT'}, "PYTHONPATH": _stubs.OPT_DIR + os.pathsep + core_src()})
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)
        code = "import sys, mosaic_opt; print([m for m in sys.modules if m.startswith('mosaic_opt.') and m != 'mosaic_opt._autoload'])"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": _stubs.OPT_DIR})
        self.assertEqual(out.stdout.strip(), "[]", "importing the package alone must load no submodule (the installed .pth's _autoload aside)")


if __name__ == "__main__":
    unittest.main()
