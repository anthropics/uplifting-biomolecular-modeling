"""Lever F9 (transition_fused): the fused pair-transition kernels — the kit module mosaic/fast/transition_fused.py; the registry entry names the module,
its class (fast: bf16 operand rounding, another summation order — never bitwise) and route; the spec grammar (route words `zres` | `sub`, kernel
settings `:k=v`, canonical spelling, named refusals); describe() is FLAT single tokens in every state; configure before install is refused by name;
P7's ONE extension point for the transition line (halfpair.set_tz_body / TZ_SLOT / _tz_line) exists and defaults to P7's own line. The arithmetic is a
GPU kernel (Pallas-Triton): its agreement with the XLA chain is the lever's own probe at configure (refused `probe_failed:<kind>` by name) and the
identity suite's band, not a CPU test."""
import importlib.util
import os
import sys
import unittest

from . import _stubs
from mosaic_opt import registry

LID = "F9"
KIT_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "transition_fused.py")
HALFPAIR_FILE = os.path.join(_stubs.KIT, "mosaic_fast", "halfpair.py")


def _load(path=KIT_FILE, name="transition_fused_under_test"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    saved = {k: sys.modules.get(k) for k in _stubs.STANDINS}
    sys.modules.update(_stubs._standins())
    try:
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestTransitionFusedLever(unittest.TestCase):
    def test_registry_entry(self):
        lv = registry.LEVERS[LID]
        self.assertEqual(lv.kit_file, f"{registry.KIT_FAST_DIR}/transition_fused.py")
        self.assertTrue(os.path.isfile(os.path.join(_stubs.KIT, lv.kit_file)))
        self.assertEqual((lv.klass, lv.route, lv.module, lv.origin, lv.tier), ("fast", "install", "mosaic.fast.transition_fused", "kit", "tier2"))
        self.assertIn(LID, registry.INSTALL)
        self.assertEqual(lv.needs, ())                                                       # P7 off → a named aside, never a refusal: no declared need

    def test_row_order_when_wired(self):
        """Wherever F9 rides a tier row it installs after F8 and BEFORE P5 / P7 (P7 reads the slot at trace time; P5's `sub` wraps the class call it finds)."""
        from mosaic_opt import modes
        for mode in ("fast", "big"):
            row = modes.KIT_MODES[mode]["levers"]
            if LID not in row:
                continue
            self.assertLess(row.index("F8"), row.index(LID)); self.assertLess(row.index(LID), row.index("P7"))
            if "P5" in row:
                self.assertLess(row.index(LID), row.index("P5"))
        self.assertNotIn(LID, modes.KIT_MODES["exact"]["levers"])

    def test_grammar(self):
        M = _load()
        self.assertEqual(M.SETTING, "zres"); self.assertEqual(M.WORDS, ("zres", "sub"))
        d = M.parse(None)
        self.assertEqual((d["on"], d["spec"], d["route"]), (True, "zres", "zres")); self.assertEqual(d["cfg"], M.CFG)
        self.assertEqual(M.parse("sub")["route"], "sub"); self.assertEqual(M.parse(" ZRES ")["spec"], "zres")
        for off in ("stock", "off", "none", "xla", ""):
            self.assertEqual((M.parse(off)["on"], M.parse(off)["spec"]), (False, "stock"), off)
        d = M.parse("zres:tb=64:t=128")
        self.assertEqual((d["cfg"]["tb"], d["cfg"]["t"], d["cfg"]["tf"]), (64, 128, M.CFG["tf"]))
        self.assertEqual(d["spec"], "zres:t=128:tb=64")                                     # canonical: CFG key order, defaults dropped
        self.assertEqual(M.parse("sub:rnd=1")["spec"], "sub")                                # a default value is not spelled
        for bad in ("fused", "zres:bogus=1", "zres:t=x", "pf", "zres+sub"):
            with self.assertRaises(M.Refusal) as cm:
                M.parse(bad)
            self.assertEqual(cm.exception.reason, "unknown_spec", bad)

    def test_describe_is_flat_and_configure_before_install_refused(self):
        M = _load(name="transition_fused_describe")
        d = M.describe()
        for k, v in d.items():
            self.assertIsInstance(v, (str, int, float, bool, type(None)), k)
            self.assertNotIn(" ", str(v), k)
        for k in ("lever", "spec", "route", "probe", "served", "served_zres", "served_sub", "f32_stock", "unserved", "aside", "impl", "origin", "fwd_tile", "bwd_tile", "rnd"):
            self.assertIn(k, d)
        self.assertEqual((d["lever"], d["installed"], d["on"], d["spec"], d["origin"], d["served"]), ("F9", 0, 0, "none", "kit", 0))
        with self.assertRaises(M.Refusal) as cm:
            M.configure(None)
        self.assertEqual(cm.exception.reason, "not_installed")
        self.assertIsNone(M.gate())                                                          # off and not installed: nothing to fail
        self.assertEqual(M.ENV_REQUIRED, {}); self.assertEqual(M.PATCHED, ("joltz.Transition.__call__",))

    def test_halfpair_extension_point(self):
        """P7 carries the ONE door F9's `zres` route uses: set_tz_body / TZ_SLOT, `_tz_line` falls to P7's own `z + _sub(...)` when no body is set or the
        body declines (NotImplemented), and describe() names the body word (flat)."""
        H = _load(HALFPAIR_FILE, name="halfpair_for_f9")
        self.assertTrue(callable(getattr(H, "set_tz_body", None))); self.assertIn("fn", H.TZ_SLOT)
        self.assertIsNone(H.TZ_SLOT["fn"]); self.assertEqual(H.describe()["tz_body"], "none")
        calls = []

        class _Mod:                                                                           # a stand-in sub-layer: records the call, returns its input
            def __call__(self, z):
                calls.append("mod"); return z

        H._cast_module = lambda mod, dtype: mod                                               # no equinox on a CPU box: identity cast
        H._half = lambda: "bf16"

        class _Z:                                                                             # a stand-in activation with astype / +
            dtype = "f32"
            def astype(self, d): return self
            def __add__(self, o): calls.append("add"); return self

        z = _Z()
        H._tz_line(_Mod(), False, z)                                                          # tz off: the module's own call + the residual add
        self.assertEqual(calls, ["mod", "add"]); calls.clear()
        H.set_tz_body(lambda mod, zz: NotImplemented, "F9:zres")
        self.assertEqual(H.describe()["tz_body"], "F9:zres")
        H._tz_line(_Mod(), True, z)                                                           # body declines → P7's own line
        self.assertEqual(calls, ["mod", "add"]); calls.clear()
        served = []
        H.set_tz_body(lambda mod, zz: (served.append(1), zz)[1], "F9:zres")
        out = H._tz_line(_Mod(), True, z)
        self.assertIs(out, z); self.assertEqual((calls, served, H.CENSUS["tz_body"]), ([], [1], 1))
        H.set_tz_body(None)
        self.assertIsNone(H.TZ_SLOT["fn"]); self.assertEqual(H.describe()["tz_body"], "none")



    def test_probe_failure_is_a_named_step_aside(self):
        """Class contract, any card: when the kernels cannot lower / compile / agree on the device (the configure probe), the lever steps aside BY NAME —
        state on, route `aside`, describe/TRANSITION `aside=probe_failed:<kind>`, no slot taken, bf16 calls run joltz's own body, and the gate passes."""
        M = _load()
        M._probe = lambda route: "compile"                       # the device verdict, whatever the card
        M._cc = lambda: "0.0"
        class _T:                                               # a stand-in joltz.Transition whose body must be what runs
            def __call__(self, x): return ("stock_body", x)
        J = type("J", (), {"Transition": _T})
        M._joltz = lambda: J
        M._halfpair = lambda: None
        M.install()
        d = M.configure(None)
        self.assertEqual((d["on"], d["route"], d["slot"], d["aside"]), (True, "aside", "none", "probe_failed:compile"))
        self.assertIn("aside=probe_failed:compile", M.emit_line("t"))
        self.assertEqual(_T()(3), ("stock_body", 3))                 # the rebound class call routes to the original body while aside
        M.gate()                                                 # passes: stood aside by name, nothing unserved
        M.uninstall()


if __name__ == "__main__":
    unittest.main()
