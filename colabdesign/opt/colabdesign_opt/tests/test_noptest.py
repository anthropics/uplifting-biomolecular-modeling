"""The pytest stand-in (_noptest.py) does what the kit's script-run tests need: raises / skip / importorskip / mark.skipif / mark.parametrize /
main; and this file itself runs as a plain script through it (`python test_noptest.py`)."""
import os
import subprocess
import sys
import unittest

try:
    import pytest                                          # noqa: F401
except ImportError:
    from colabdesign_opt.tests import _noptest as pytest   # noqa: F401
from colabdesign_opt.tests import _noptest as N


class TestStandIn(unittest.TestCase):
    def test_raises(self):
        with N.raises(ValueError, match="bad .* value") as e:
            raise ValueError("bad mode value")
        self.assertIsInstance(e["value"], ValueError)
        with self.assertRaises(AssertionError):
            with N.raises(ValueError):
                pass
        with self.assertRaises(AssertionError):
            with N.raises(ValueError, match="x"):
                raise ValueError("y")

    def test_skips_and_marks(self):
        with self.assertRaises(unittest.SkipTest):
            N.skip("why")
        with self.assertRaises(unittest.SkipTest):
            N.importorskip("no_such_module_xyz")
        self.assertIs(N.importorskip("json"), sys.modules["json"])
        self.assertEqual(N.mark.skipif(False, reason="r")(lambda: 7)(), 7)
        with self.assertRaises(unittest.SkipTest):
            N.mark.skipif(True, reason="r")(lambda: 7)()
        seen = []
        N.mark.parametrize("a, b", [(1, 2), (3, 4)])(lambda a, b: seen.append((a, b)))()
        N.mark.parametrize("n", [5, 6])(lambda n: seen.append(n))()
        self.assertEqual(seen, [(1, 2), (3, 4), 5, 6])

    @unittest.skipIf(os.environ.get("COLABDESIGN_OPT_NOPTEST_CHILD"), "the script run itself (no recursion)")
    def test_runs_as_a_script(self):
        r = subprocess.run([sys.executable, __file__], capture_output=True, text=True, env=dict(os.environ, COLABDESIGN_OPT_NOPTEST_CHILD="1"), timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn("4 passed, 0 failed, 1 skipped (pytest stand-in)", r.stdout)


class TestMonkeyPatch(unittest.TestCase):
    def test_setattr_setenv_undo(self):
        import os, types
        obj = types.SimpleNamespace(a=1)
        mp = N.MonkeyPatch()
        mp.setattr(obj, "a", 2); mp.setattr(obj, "b", 3, raising=False); mp.setenv("COLABDESIGN_OPT_NOPTEST_X", "v"); mp.setitem(os.environ, "COLABDESIGN_OPT_NOPTEST_Y", "w")
        self.assertEqual((obj.a, obj.b, os.environ["COLABDESIGN_OPT_NOPTEST_X"], os.environ["COLABDESIGN_OPT_NOPTEST_Y"]), (2, 3, "v", "w"))
        mp.delenv("COLABDESIGN_OPT_NOPTEST_Y"); self.assertNotIn("COLABDESIGN_OPT_NOPTEST_Y", os.environ)
        with self.assertRaises(AttributeError):
            mp.setattr(obj, "zzz", 1)
        mp.undo()
        self.assertEqual((obj.a, hasattr(obj, "b")), (1, False)); self.assertNotIn("COLABDESIGN_OPT_NOPTEST_X", os.environ); self.assertNotIn("COLABDESIGN_OPT_NOPTEST_Y", os.environ)


def test_function_style():                                 # collected by pytest AND by _noptest.main
    with N.raises(ZeroDivisionError):
        1 / 0


if __name__ == "__main__":
    raise SystemExit(N.main([__file__]))
