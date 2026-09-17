"""stock/check_pins.py: version normalisation, the GPU-only pins, the requirement lines, the checkout and weights checks on stand-ins."""
import importlib.util
import json
import os
import tempfile
import unittest

from . import _stubs

spec = importlib.util.spec_from_file_location("cp", os.path.join(_stubs.TREE, "stock", "check_pins.py")); cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)


class TestCheckPins(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(cp.normalize("12.9.2 (4th version field: 10)"), "12.9.2.10")
        self.assertEqual(cp.normalize("1.11.0"), "1.11")
        self.assertEqual(cp.normalize("0.5.3"), "0.5.3")

    def test_gpu_only(self):
        pins = {"pins": {"jax": "0.5.3", "nvidia-cudnn-cu12": "9.23.2.1"}, "pins_gpu_only": ["nvidia-cudnn-cu12"]}
        bad, d = cp.check(pins, installed={"jax": "0.5.3"}, gpu=False)
        self.assertEqual(bad, []); self.assertFalse(d["nvidia-cudnn-cu12"]["required"])
        bad, d = cp.check(pins, installed={"jax": "0.5.3"}, gpu=True)
        self.assertEqual(len(bad), 1)
        bad, d = cp.check(pins, installed={"jax": "0.5.3", "nvidia_cudnn_cu12": "9.23.2.1"}, gpu=True)
        self.assertEqual(bad, [])

    def test_checkout_and_weights_on_stand_ins(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            pins = json.load(open(os.path.join(tree, "stock", "PINS.json")))
            self.assertEqual(cp.check_checkout(pins, os.path.dirname(env["AF2IG_DIR"])), ([], 1))
            self.assertEqual(cp.check_weights(pins, env["AF2_PARAMS"])[0], [])
            with open(os.path.join(env["AF2IG_DIR"], "extra.py"), "w") as fh:
                fh.write("x")
            bad, n = cp.check_checkout(pins, os.path.dirname(env["AF2IG_DIR"]))
            self.assertEqual(len(bad), 1); self.assertIn("not a file of the pinned checkout", bad[0])
            self.assertEqual(cp.check_weights(pins, tmp)[0][0].endswith("not found"), True)
            v = cp.weights_verdict(pins, env["AF2_PARAMS"]); self.assertEqual((v["present"], v["pinned"]), (True, True)); self.assertEqual(cp.weights_words(v), f"weights=params_model_1_ptm.npz sha256={v['sha256'][:12]} (pinned)")
            other = os.path.join(tmp, "other"); os.makedirs(os.path.join(other, "params"))
            with open(os.path.join(other, "params", "params_model_1_ptm.npz"), "wb") as fh: fh.write(b"other bytes")
            self.assertEqual(cp.check_weights(pins, other)[0], [])                                    # present = accepted whatever the bytes
            v = cp.weights_verdict(pins, other); self.assertEqual((v["present"], v["pinned"]), (True, False)); self.assertTrue(cp.weights_words(v).startswith("weights sha256=") and cp.weights_words(v).endswith(cp.WEIGHTS_NOT_PINNED.format(pin12=v["pin"][:12])))


if __name__ == "__main__":
    unittest.main()
