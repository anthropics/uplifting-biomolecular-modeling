"""The mode table named by guarantee (three modes validated by the core's ModeTable, the lever set per mode, resolve and its refusals,
the autoload's import-free copy of the names) and this kit's adoption of the shared core (opt_core): the build backend is the tree's
template byte-for-byte, the `[tool.opt_core]` pin names the imported core and its forward_numerics globs cover the BindCraft flow's real
files, the partial-activation lines and exit codes are the tree's form of the core's own record."""

import os
import sys
import unittest

import opt_core
from opt_core import gates as core_gates, report as core_report

from colabdesign_opt import _autoload, modes, names, registry, report
from colabdesign_opt.names import PREFIX
from colabdesign_opt.tests._stubs import tree_present


# ============================================================ the mode table: guarantees, resolve, refusals, the autoload's copy
class TestModes(unittest.TestCase):
    def test_table(self):
        self.assertEqual(list(modes.MODES), ["off", "exact", "fast"])
        self.assertEqual(modes.TABLE.modes, ("off", "exact", "fast")); self.assertEqual(modes.TABLE.default, "fast")
        self.assertEqual(modes.DEFAULT_MODE, "fast")                    # the package default: fast wherever a fast mode ships
        self.assertEqual({m: modes.MODES[m].levers for m in modes.MODES}, {"off": (), "exact": ("compilecache", "lowercache", "parcompile", "hoist_prev"), "fast": ("compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla")})
        self.assertEqual({m: (modes.MODES[m].tier, modes.MODES[m].numerics_class, modes.MODES[m].route) for m in modes.MODES},
                         {"off": (None, None, "stock"), "exact": (1, "exact", "kit"), "fast": (2, "precision", "kit")})
        self.assertEqual((modes.SIZE_GATE_TOKENS, modes.STOCK_SUBBATCH), (384, 4))
        for m in modes.MODES.values():
            for lever in m.levers:
                self.assertIn(lever, registry.LEVERS)

    def test_resolve(self):
        r = modes.resolve(None)                                         # no mode = the default
        self.assertEqual((r.mode, r.levers, r.tier, r.numerics_class, r.route), ("fast", ("compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"), 2, "precision", "kit"))
        self.assertEqual(modes.resolve(" Fast ").levers, ("compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"))            # names are stripped and lowercased
        r = modes.resolve("off")
        self.assertEqual((r.mode, r.levers, r.tier, r.route), ("off", (), None, "stock"))
        r = modes.resolve("exact")                                                        # the exact tier: the exact-class levers, tier 1, class 1
        self.assertEqual((r.mode, r.levers, r.tier, r.numerics_class, r.route), ("exact", ("compilecache", "lowercache", "parcompile", "hoist_prev"), 1, "exact", "kit"))
        with self.assertRaises(ValueError) as cm:
            modes.resolve("turbo")                                                          # an unknown name through the one path, the one text
        self.assertEqual(str(cm.exception), "unknown mode 'turbo' (expected off|exact|fast, or exact-no-<lever>[-no-<lever>...] without the named levers of exact's compilecache+lowercache+parcompile+hoist_prev, fast-no-<lever>[-no-<lever>...] without the named levers of fast's compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla)")
        self.assertEqual(modes.describe(modes.resolve("fast")), "mode=fast levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla route=kit")

    def test_env_readers(self):
        self.assertIsNone(modes.mode_from_env({}))
        self.assertEqual(modes.mode_from_env({"COLABDESIGN_OPT": " Fast "}), "fast")
        self.assertFalse(hasattr(modes, "allow_partial_from_env")); self.assertFalse(hasattr(modes, "ENV_ALLOW_PARTIAL"))   # no opt-out variable: an installed lever that did not engage is a defect; a lever that cannot run steps aside at install, named

    def test_autoload_names_equal_the_table(self):
        """The hook reads the mode names from names.py when a mode is set (nothing of the package at interpreter start): names == the table."""
        self.assertEqual(names.MODE_NAMES, tuple(modes.MODES)); self.assertEqual(_autoload.ENV, modes.ENV)
        self.assertFalse(any(hasattr(_autoload, a) for a in ("MODES", "KIT_DIR_MARKERS", "NEVER_LEVERED")))   # no module-level names binding in the hook
        f = _autoload.install({"COLABDESIGN_OPT": "not_a_mode"})
        try:
            self.assertIn("expected " + "|".join(modes.MODES), f.refusal)                  # the refusal names the table's modes
        finally:
            sys.meta_path[:] = [x for x in sys.meta_path if x is not f]



# ============================================================ this kit's adoption of the shared core: the pin, the build backend, the cache key
HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))                     # colabdesign/opt


class CoreAdoption(unittest.TestCase):
    def test_partial_lines_are_the_core_templates(self):
        d = "nosub: fallback: grad_fn subbatch=4 (auto: peak exceeds the device)"
        self.assertTrue(core_report.PARTIAL_REFUSED.format(prefix=PREFIX, detail=d).startswith(f"{PREFIX} {report.PARTIAL_HEAD}{d}; exit 3 ("))   # the head is the tree's form; the tail is this kit's (no opt-out flag to name)
        self.assertEqual(f"{PREFIX} {report.partial_line(d)}", f"{PREFIX} NOT ACTIVE: partial activation — {d}; exit 3 (an installed lever that did not engage is a defect, named - a lever that cannot run here steps aside at install instead, state=skipped, exit 0)")
        self.assertFalse(hasattr(report, "partial_allowed_line")); self.assertEqual(report.PARTIAL_HEAD, "NOT ACTIVE: partial activation — ")
        self.assertEqual((report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE), (0, 1, 2, 3))

    @unittest.skipUnless(tree_present(), "needs the release tree (opt/pyproject.toml, common/opt_core/kit_template)")
    def test_pin_and_backend(self):
        pin = core_gates.read_pin_table(os.path.join(OPT, "pyproject.toml"))
        self.assertTrue({"path", "version"} <= set(pin), sorted(pin))       # the two string keys the gate reads (its stdlib fallback reader on python < 3.11 skips arrays)
        import glob
        import re
        tree = os.path.dirname(OPT)
        text = open(os.path.join(OPT, "pyproject.toml"), encoding="utf-8").read()
        block = re.search(r"^forward_numerics\s*=\s*\[(.*?)^\]", text, re.S | re.M)               # the declared numerics globs, read from the file on any interpreter
        self.assertIsNotNone(block, "opt/pyproject.toml declares no [tool.opt_core] forward_numerics")
        globs = re.findall(r'"([^"]+)"', block.group(1))
        if "forward_numerics" in pin:
            self.assertEqual(list(pin["forward_numerics"]), globs)
        self.assertGreater(len(globs), 10)
        for pattern in globs:                                                          # every declared numerics glob names files of this tree
            self.assertTrue(glob.glob(os.path.join(tree, pattern), recursive=True), pattern)
        for rel in ("stock/src/bindcraft/functions/colabdesign_utils.py", "stock/src/bindcraft/functions/biopython_utils.py", "stock/src/bindcraft/functions/generic_utils.py",
                    "stock/src/bindcraft/settings_advanced/default_4stage_multimer.json",
                    "opt/colabdesign_opt/bindcraft.py", "opt/colabdesign_opt/units.py", "opt/colabdesign_opt/settings.py", "opt/colabdesign_opt/stock_design.py",
                    "opt/colabdesign_opt/nosub.py", "opt/colabdesign_opt/pallas.py", "opt/colabdesign_opt/levers.py", "opt/colabdesign_opt/modes.py", "opt/colabdesign_opt/registry.py",
                    "opt/colabdesign_opt/kernels/__init__.py", "opt/colabdesign_opt/launch.py", "opt/colabdesign_opt/kit_launch.py", "opt/colabdesign_opt/stock_launch.py", "stock/PINS.json", "configs/h100.env"):
            self.assertTrue(any(os.path.join(tree, rel) in glob.glob(os.path.join(tree, pat), recursive=True) for pat in globs), rel)   # the BindCraft flow's numerics files are declared
        from colabdesign_opt._core_gate import version_tuple                          # the pin is a FLOOR (>=), never equality: a newer core than the pin serves this kit
        self.assertLessEqual(version_tuple(pin["version"]), version_tuple(opt_core.__version__), (pin["version"], opt_core.__version__))
        g = core_gates.core_pin_check(os.path.join(OPT, "pyproject.toml"))
        self.assertTrue(g.ok, g.reason)                                               # the imported core meets the pinned floor
        template = os.path.join(OPT, pin["path"], "kit_template", "_build_backend.py")
        with open(template, "rb") as a, open(os.path.join(OPT, "_build_backend.py"), "rb") as b:
            self.assertEqual(a.read(), b.read())                                       # the tree's one build backend, byte-for-byte


if __name__ == "__main__":
    unittest.main()
