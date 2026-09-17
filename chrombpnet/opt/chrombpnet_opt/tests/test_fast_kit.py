"""The fast mode's kit at the package level: opt/kit_ho (the tf/ side: the finish-stage overlap, the PNG helpers and the one bigWig writer
interpreter spawned at process start) with the carried kit opt/kit beside it; no switch selects it. These tests run on the real tree
and are skipped when the carried kit is not in it."""
import glob, os
import unittest

from chrombpnet_opt import stack, report, registry
from . import _stubs


@unittest.skipUnless(_stubs.real_kit() and _stubs.real_kit("kit_ho"), "opt/kit and opt/kit_ho are not both in this tree")
class FastKitTests(unittest.TestCase):
    def test_kit_home_is_kit_ho(self):
        k = stack.kit_home()
        self.assertTrue(k.endswith(os.path.join("opt", "kit_ho")), k)
        self.assertTrue(os.path.isfile(os.path.join(k, "tf", "pred_bw_fast.py")))
        self.assertTrue(os.path.isdir(os.path.join(os.path.dirname(k), "kit", "torch")))          # the carried kit beside it (torch/, caches)
        self.assertTrue(stack.kit_version(k).endswith("+ho2"), stack.kit_version(k))
        self.assertEqual(stack.kit_tables_root(k), os.path.join(os.path.dirname(k), "kit"))

    def test_kit_env_composes_no_kit_switch(self):
        kenv, stripped, cache = stack.kit_env({"HOME": "/x"}, False, stack.kit_home(), "k1", "H100")
        self.assertEqual([n for n in registry.KIT_SWITCH_NAMES if n in kenv], [])                   # the kit's own switch names are never composed by the package

    def test_kit_ho_route_files_present(self):
        integ = stack.kit_integrity(stack.kit_home(), scope="route")
        self.assertGreater(integ["n_present"], 0)
        self.assertEqual(integ["missing"], [])

    def test_one_fast_kit_tree(self):
        """One fast-kit tree: the tf/ side (entry script, chrombpnet_fastkit, the seed hook, the tf/ CPU tests, the regime self-test) lives
        in opt/kit_ho only; the carried kit opt/kit beside it holds the torch side (and the driver caches at run time), no tf/."""
        k = stack.kit_home(); carried = stack.carried_kit(k)
        self.assertNotEqual(carried, k)
        self.assertTrue(os.path.isfile(os.path.join(k, "tf", "det_subprocess", "sitecustomize.py")))
        self.assertTrue(glob.glob(os.path.join(k, "tf", "test_*_cpu.py")) and not os.path.isfile(os.path.join(k, "tf", "selftest_regimes.py")))   # the kit's CPU tests ship beside the code; the regime self-test driver does not ship in the kit
        self.assertTrue(os.path.isdir(os.path.join(carried, "torch", "chrombpnet_k1")))
        self.assertFalse(os.path.exists(os.path.join(carried, "tf")))
        self.assertEqual(stack.kit_tables_root(k), carried)

    def test_mode_line(self):
        rep = {"active": True, "mode": "fast", "route": "k1", "kit_version": "0.12.73+ho2", "gpu": {"class": "H100"}, "det": False}
        self.assertTrue(report.mode_line(rep).endswith(" kit=0.12.73+ho2 gpu=H100 det=0 precision=tf32"), report.mode_line(rep))


if __name__ == "__main__":
    unittest.main()
