"""The pins gate (stack.pins_check on the PINS.json schema; public vs local version tags) and the one weights loader's digest words
(weights.assert_pins: sha256 / byte count / config.json against stock/PINS.json — the pinned digest says pinned, any other digest says
not pinned, both load and the weights line says which; only a repository with no pin or a file not in the cache refuses, by name).
The tree's standalone stock caller (opt/flashzoi_opt/stock_pred.py) words its replicates the same way: its copy is exercised here on the same files."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest

from flashzoi_opt import stack, weights
from flashzoi_opt.tests import _stubs


class TestPinsGate(unittest.TestCase):
    def test_schema(self):
        t = _stubs.Tree().enter()
        try:
            pins = stack.read_pins()
            for k in ("package", "stack", "weights", "config_json_sha256"):
                self.assertIn(k, pins)
            p = os.path.join(t.root, "stock", "PINS.json")
            bad = _stubs.pins_stub(); del bad["stack"]["flash-attn"]; json.dump(bad, open(p, "w"))
            with self.assertRaises(KeyError):
                stack.read_pins()
        finally:
            t.exit()

    def test_versions_public_vs_local(self):
        pins = _stubs.pins_stub()
        ok = dict(_stubs.PINNED)
        self.assertEqual(stack.pins_check(pins, ok), [])
        self.assertEqual(stack.pins_check(pins, dict(ok, torch="2.5.1")), [])                       # a CPU wheel without the local tag meets the public pin
        bad = stack.pins_check(pins, dict(ok, torch="2.6.0+cu124", **{"flash-attn": None}))
        self.assertEqual(bad, ["torch 2.6.0+cu124 != pinned 2.5.1", "flash-attn not installed != pinned 2.7.0.post2"])
        self.assertEqual(stack.pins_check(pins, dict(ok, **{"borzoi-pytorch": "0.5.0"})), [])          # the stock pin is stock_pin_check's (the one refusal); pins_check names stack drift only
        self.assertEqual(stack.stock_pin_check(pins, dict(ok, **{"borzoi-pytorch": "0.5.0"})), ["borzoi-pytorch 0.5.0 != pinned 0.5.1"])
        self.assertEqual(stack.stock_pin_check(pins, dict(ok, torch="2.6.0")), [])
        self.assertEqual(stack.pins_check(pins, dict(ok, python="3.12.1")), ["python 3.12 != pinned 3.11"])

    def test_installed_versions_shape(self):
        v = stack.installed_versions()
        for k in ("borzoi-pytorch", "torch", "triton", "flash-attn", "transformers", "python"):
            self.assertIn(k, v)


class TestWeightsPins(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.files = {}
        for name, data in (("model.safetensors", b"abcd"), ("config.json", b"{}")):
            p = os.path.join(self.d, name); open(p, "wb").write(data); self.files[name] = p
        self._resolve = weights.resolve_file
        weights.resolve_file = lambda repo, filename, revision: self.files[filename]

    def tearDown(self):
        weights.resolve_file = self._resolve
        shutil.rmtree(self.d, ignore_errors=True)

    def _pins(self, **over):
        w = {"revision": "0" * 40, "model.safetensors_sha256": hashlib.sha256(b"abcd").hexdigest(), "bytes": 4}
        w.update(over)
        return {"weights": {"johahi/flashzoi-replicate-0": w}, "config_json_sha256": hashlib.sha256(b"{}").hexdigest()}

    def test_pinned_digest_says_pinned(self):
        rec = weights.assert_pins(self._pins(), "johahi/flashzoi-replicate-0")
        self.assertEqual(rec["bytes"], 4); self.assertEqual(rec["model.safetensors_sha256"], hashlib.sha256(b"abcd").hexdigest()); self.assertIn("config.json_sha256", rec)
        self.assertEqual((rec["word"], rec["differs"]), (weights.WEIGHTS_WORDS[0], []))
        self.assertEqual(weights.weights_words(rec), f"weights=johahi/flashzoi-replicate-0@000000000000 sha256={hashlib.sha256(b'abcd').hexdigest()[:12]} (pinned)")
        err = io.StringIO()
        line = weights.say_weights(rec, stream=err)
        self.assertEqual(err.getvalue(), line + "\n"); self.assertTrue(line.startswith("[flashzoi-opt] weights=johahi/flashzoi-replicate-0@"))

    def test_unknown_digests_say_not_pinned_and_load(self):
        """A sha256, byte-count or config.json difference is worded, named in the record and never raised: the checkpoint loads."""
        sha12 = hashlib.sha256(b"abcd").hexdigest()[:12]
        rec = weights.assert_pins(self._pins(**{"model.safetensors_sha256": "0" * 64}), "johahi/flashzoi-replicate-0")     # no exception: it loads
        self.assertEqual(rec["word"], weights.WEIGHTS_WORDS[1]); self.assertEqual(len(rec["differs"]), 1); self.assertIn("sha256", rec["differs"][0])
        self.assertEqual(rec["pinned_sha256"], "0" * 64); self.assertEqual(rec["model.safetensors_sha256"][:12], sha12)              # the digest is the record either way
        line = weights.weights_words(rec)
        self.assertTrue(line.startswith(f"weights sha256={sha12} NOT PINNED — the kit's statements hold for the pinned weights only (johahi/flashzoi-replicate-0@"), line)
        self.assertTrue(weights.WEIGHTS_NOTICE.startswith("NOT PINNED — ")); self.assertIn(weights.WEIGHTS_NOTICE, line)
        err = io.StringIO()
        self.assertEqual(weights.say_weights(rec, stream=err), err.getvalue().rstrip("\n")); self.assertIn(f"[flashzoi-opt] weights sha256={sha12} NOT PINNED", err.getvalue())
        rec = weights.assert_pins(self._pins(bytes=5), "johahi/flashzoi-replicate-0")
        self.assertEqual(rec["word"], weights.WEIGHTS_WORDS[1]); self.assertIn("bytes", rec["differs"][0])
        pins = self._pins(); pins["config_json_sha256"] = "1" * 64
        rec = weights.assert_pins(pins, "johahi/flashzoi-replicate-0")
        self.assertEqual(rec["word"], weights.WEIGHTS_WORDS[1]); self.assertIn("config.json", rec["differs"][0])

    def test_missing_pin_and_missing_file_refuse_by_name(self):
        with self.assertRaises(weights.WeightsError) as cm:
            weights.assert_pins({"weights": {}}, "johahi/flashzoi-replicate-0")
        self.assertIn("no complete weights pin for johahi/flashzoi-replicate-0", str(cm.exception))
        def gone(repo, filename, revision):
            raise weights.WeightsError(f"{repo}@{revision[:12]} {filename} is not in the offline hub cache")
        weights.resolve_file = gone
        with self.assertRaises(weights.WeightsError) as cm:
            weights.assert_pins(self._pins(), "johahi/flashzoi-replicate-0")
        self.assertIn("model.safetensors is not in the offline hub cache", str(cm.exception))

    def test_stock_caller_words_its_replicates_the_same_way(self):
        """stock_pred.py (imports nothing of this package) on the same files: pinned / not-pinned lines with its own prefix, a load either way."""
        script = os.path.join(stack.opt_home(), "flashzoi_opt", "stock_pred.py")
        spec = importlib.util.spec_from_file_location("stock_pred_script_weights", script)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.resolve_cached = lambda repo, filename, revision: self.files[filename]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rec = mod.assert_pins(self._pins(), "johahi/flashzoi-replicate-0")
            bad = mod.assert_pins(self._pins(**{"model.safetensors_sha256": "0" * 64}), "johahi/flashzoi-replicate-0")
        self.assertEqual((rec["word"], bad["word"]), mod.WEIGHTS_WORDS); self.assertEqual(mod.WEIGHTS_WORDS, weights.WEIGHTS_WORDS); self.assertEqual(mod.WEIGHTS_NOTICE, weights.WEIGHTS_NOTICE)
        lines = err.getvalue().splitlines()
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(lines[0].startswith("[flashzoi-stock] weights=johahi/flashzoi-replicate-0@000000000000 sha256=") and lines[0].endswith("(pinned)"), lines[0])
        self.assertTrue(lines[1].startswith(f"[flashzoi-stock] weights sha256={hashlib.sha256(b'abcd').hexdigest()[:12]} NOT PINNED — "), lines[1])
        with self.assertRaises(RuntimeError):
            mod.assert_pins({"weights": {}}, "johahi/flashzoi-replicate-0")                          # no pin: the refusal stays

    @unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "huggingface_hub not installed: weights.resolve_file's offline miss is its refusal")
    def test_offline_miss_refuses(self):
        weights.resolve_file = self._resolve
        saved = {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_OFFLINE")}
        os.environ["HF_HOME"] = self.d; os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            with self.assertRaises(weights.WeightsError) as cm:
                weights.assert_pins(self._pins(), "johahi/flashzoi-replicate-0")
            self.assertIn("not in the offline hub cache", str(cm.exception))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
