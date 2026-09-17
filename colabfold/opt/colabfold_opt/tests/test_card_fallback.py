"""A lever that cannot run on the box's GPU refuses the MODE by name — a mode is all of its levers, so no run goes out under the mode's name with a
subset of them. The vehicle is TRIMUL_PALLAS's floor (trimul_pallas.require: a part below compute capability 8.0 raises `Refusal(cc_below_8_0)`):
`NOT ACTIVE: enable() failed: TRIMUL_PALLAS: Refusal(…); exit 3`, every other refusing lever named on the SAME line. An untested 8.x part engages
the lever (the provider's tier word decides the row per call; its LEVER line's `cc=` names the part). The one-line word of a table-backed lever without a tile row
(`report.LEVER_REFUSED_FMT`, `NOT ACTIVE: <LEVER>=fallback:no_tiles_cc<NN>(<kind>): …`) is a report.py format unit (test_modes_lock). Stubs only: no jax, no GPU."""
import io
import sys
import tempfile
from contextlib import redirect_stderr

from colabfold_opt import manifest, report, stack, trimul_pallas
from colabfold_opt.tests import _stubs
from colabfold_opt.tests.test_activation import Base

REFUSED_HEAD = "[colabfold-opt] NOT ACTIVE: enable() failed: TRIMUL_PALLAS: Refusal("


class TestCardFallback(Base):
    def below_floor(self):
        kern = sys.modules[_stubs.STUB_TRIMUL]; kern.CC = "7.5"
        return kern

    def tearDown(self):
        sys.modules[_stubs.STUB_TRIMUL].CC = "9.0"
        super().tearDown()

    def test_a_part_below_the_floor_refuses_the_mode_by_name_exit_3(self):
        self.below_floor()
        out = tempfile.mkdtemp(dir=self.tmp)
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(stack.ActivationError):
            stack.activate("fast", queries=_stubs.queries(600, 705), strict=True, manifest_dir=out)
        lines = err.getvalue().splitlines()
        self.assertTrue(lines[0].startswith(REFUSED_HEAD), lines[0]); self.assertIn("cc_below_8_0", lines[0])   # ONE line: the lever and its named refusal
        self.assertFalse(any(" ACTIVE mode=" in l for l in lines))                                   # never an ACTIVE line for a mode whose lever cannot engage
        rep = stack.status()
        self.assertFalse(rep["active"]); self.assertEqual(rep["levers_unavailable"], list(rep["levers"]))
        self.assertTrue(rep["reason"].startswith("enable() failed: TRIMUL_PALLAS: "), rep["reason"])
        self.assertFalse(trimul_pallas.marker_present(self.mods["alphafold.model.modules"]))       # the class was NOT rebound
        man = manifest.read(manifest.path(out))
        self.assertFalse(man["active"]); self.assertNotIn("lever_fallbacks", man)
        v = manifest.verdict(out, "fast", [], report.EXIT_NOT_ACTIVE)                               # pred's verdict of that model process: rc 3, not active
        self.assertEqual((v["ok"], v["rc"]), (False, report.EXIT_NOT_ACTIVE))

    def test_non_strict_route_reports_inactive(self):
        self.below_floor()
        err = io.StringIO()
        with redirect_stderr(err):
            rep = stack.activate("fast", queries=_stubs.queries(600, 705), strict=False)
        self.assertFalse(rep["active"]); self.assertFalse(rep.get("lever_fallbacks"))
        self.assertTrue(err.getvalue().splitlines()[0].startswith(REFUSED_HEAD))

    def test_every_refusal_is_named_in_one_line(self):
        self.below_floor()
        aserve = sys.modules[_stubs.STUB_ATTN_SERVE]
        def refuse2(*a, **k):
            raise RuntimeError("pallas_missing")
        saved = aserve.require; aserve.require = refuse2
        try:
            err = io.StringIO()
            with redirect_stderr(err):
                rep = stack.activate("fast", queries=_stubs.queries(600, 705), strict=False)
        finally:
            aserve.require = saved
        line = err.getvalue().splitlines()[0]
        self.assertTrue(line.startswith(REFUSED_HEAD), line)
        self.assertIn("PALLAS_MSA: RuntimeError('pallas_missing')", line)                          # the second refusal is named too: one pass, one line
        self.assertFalse(rep["active"]); self.assertEqual(rep["levers_unavailable"], list(rep["levers"]))

    def test_an_untested_8x_part_engages_the_lever(self):
        kern = sys.modules[_stubs.STUB_TRIMUL]; kern.CC = "8.6"
        out = tempfile.mkdtemp(dir=self.tmp); err = io.StringIO()
        with redirect_stderr(err):
            rep = stack.activate("fast", queries=_stubs.queries(600, 705), strict=True, manifest_dir=out)
        self.assertTrue(rep["active"]); self.assertIn("TRIMUL_PALLAS", rep["levers"]); self.assertFalse(rep.get("lever_fallbacks"))
        self.assertNotIn("NOT ACTIVE", err.getvalue()); self.assertNotIn("lever_fallbacks", manifest.read(manifest.path(out)))
        st = stack.trimul_lever_state()
        self.assertEqual((st["word"], st["cc"]), ("fast", "8.6"))                                   # what the exit LEVER name=TRIMUL_PALLAS line renders as word=<tier> cc=<cc>: the mode's tier word, the part; the row per call is the provider's

    def test_a_gpu_that_serves_every_lever_writes_no_fallback_key(self):
        out = tempfile.mkdtemp(dir=self.tmp)
        with redirect_stderr(io.StringIO()):
            rep = stack.activate("fast", queries=_stubs.queries(600, 705), strict=True, manifest_dir=out)
        self.assertTrue(rep["active"])
        self.assertNotIn("lever_fallbacks", manifest.read(manifest.path(out)))                     # nothing held: the manifest is what it was

    def test_the_table_backed_refusal_word_keeps_its_format(self):
        word = "fallback:no_tiles_cc75(trimul)"
        self.assertEqual(report.lever_refused_line({"lever_fallbacks": {"SOME_LEVER": word}}),
                         f"[colabfold-opt] NOT ACTIVE: SOME_LEVER={word}: the lever cannot run on this GPU (no tile table for its compute capability) — a mode is all of its levers; "
                         "exit 3 (--mode off runs stock)")
