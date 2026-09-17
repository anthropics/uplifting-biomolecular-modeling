"""The kit carried under opt/forward/af2ig_kit: the one directory under opt/forward/ (its bytes are the git commit's — no
count, digest or manifest of its files), its MANIFEST.json the kit that stock/PINS.json names, and every file the package resolves inside
it present (registry.DRIVER, the checkout's patch series; the patched copies: test_stock_archives)."""
import json
import os
import unittest

from af2ig_opt import registry
from . import _stubs


class TestKitCarry(unittest.TestCase):
    def test_carry(self):
        kit = _stubs.KIT
        self.assertEqual(sorted(os.listdir(os.path.dirname(kit))), ["af2ig_kit"])   # opt/forward/ carries kit bytes only
        man = json.load(open(os.path.join(kit, "MANIFEST.json"), encoding="utf-8"))
        pins = json.load(open(os.path.join(_stubs.TREE, "stock", "PINS.json"), encoding="utf-8"))
        self.assertEqual(man["name"], pins["kit"]["name"])
        self.assertEqual(os.path.normpath(pins["kit"]["dir"]), os.path.relpath(kit, _stubs.TREE))
        for rel in (registry.DRIVER, *pins["checkout"]["patches"]):
            self.assertTrue(os.path.isfile(os.path.join(kit, rel)), rel)


if __name__ == "__main__":
    unittest.main()
