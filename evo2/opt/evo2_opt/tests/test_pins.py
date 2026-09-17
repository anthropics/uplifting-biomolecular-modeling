"""stock/PINS.json as the package reads it: the tree root, the stack by Transformer Engine presence, the schema the readers rely on."""
import os
import unittest
from unittest import mock

from evo2_opt import pins


class TestPins(unittest.TestCase):
    def test_tree_root_and_schema(self):
        doc = pins.load()
        self.assertTrue(os.path.isfile(os.path.join(pins.tree_root(), "run.sh")))
        self.assertEqual({"evo2", "vtx"}, set(doc["upstream"]))
        self.assertEqual(doc["stock_environment"]["must_be_absent_names"], ["EVO2_OPT"])
        self.assertIn("evo2_7b", doc["checkpoints"]); self.assertIn("sha256", doc["checkpoints"]["evo2_40b"])
        for st in doc["stacks"].values():
            self.assertIn("transformer_engine", st["pins"]); self.assertIn("evo2_7b", st["serves"])
        self.assertTrue(all({"sm", "memory_mib"} <= set(v) for v in doc["gpus"].values()))
        with mock.patch.dict(os.environ, {"EVO2_OPT_HOME": "/elsewhere"}):
            self.assertEqual(pins.pins_path(), "/elsewhere/stock/PINS.json")

    def test_stack_by_te_presence(self):
        doc = pins.load()
        self.assertEqual(pins.stack_of(doc, True), "img_full"); self.assertEqual(pins.stack_of(doc, False), "img_a100")


if __name__ == "__main__":
    unittest.main()
