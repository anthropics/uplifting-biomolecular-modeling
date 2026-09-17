"""The weights word is decided by the sha256 pinned digest (stock/PINS.json "weights"), digested afresh at activation: pinned -> a `WEIGHTS pinned
sha256` line and no warning; any other digest (same size or not) -> `WEIGHTS unknown ...; proceeding` BY NAME and the run proceeds (rc 0 at every
entry point: the gate returns no reason); an absent file refuses. CPU only, on a small synthetic pin set laid out under a temporary HF_HOME."""
import copy
import hashlib
import os
import tempfile
import unittest

from esmfold2_opt import stack


def _small_pins():
    """The kit's pins with every weights file shrunk to a few known bytes (pinned digest = sha256 of those bytes)."""
    pins = copy.deepcopy(stack.pins()); content = {}
    for repo, w in pins["weights"].items():
        for name, d in w["files"].items():
            b = f"{repo}/{name}/of-record".encode()
            d["size_bytes"] = len(b); d["sha256"] = hashlib.sha256(b).hexdigest(); content[(repo, name)] = b
    return pins, content


def _lay_out(root, pins, content, variant="fast", tamper=None, resize=None, omit=None):
    for repo, rel, _size in stack.pinned_weight_files(pins, variant):
        name = os.path.basename(rel)
        if omit == rel:
            continue
        b = content[(repo, name)]
        if tamper == rel:
            b = bytes([b[0] ^ 1]) + b[1:]                                  # same size, another digest
        if resize == rel:
            b = b + b"+"                                                 # another size (and digest)
        p = os.path.join(root, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b)


class TestUnknownCheckpointWarnsAndProceeds(unittest.TestCase):
    def setUp(self):
        self.pins, self.content = _small_pins()
        self.files = stack.pinned_weight_files(self.pins, "fast"); self.assertTrue(self.files)
        self.rel = next(rel for _r, rel, _s in self.files if rel.endswith("model.safetensors"))

    def _gate(self, hf):
        why, notes = stack.data_path_gate({"HF_HOME": hf}, variant="fast", pins=self.pins)
        old = os.environ.get("HF_HOME"); os.environ["HF_HOME"] = hf
        try:
            status = stack.weights_status("fast", self.pins)
        finally:
            os.environ.pop("HF_HOME") if old is None else os.environ.__setitem__("HF_HOME", old)
        return why, notes, status

    def test_pinned_digests_pass_with_the_pinned_line_and_no_warning(self):
        with tempfile.TemporaryDirectory() as hf:
            _lay_out(hf, self.pins, self.content)
            why, notes, status = self._gate(hf)
            self.assertIsNone(why); self.assertFalse([n for n in notes if n.startswith("WEIGHTS unknown")], notes)
            ok = [n for n in notes if n.startswith("WEIGHTS pinned sha256")]; self.assertEqual(len(ok), 1, notes)
            self.assertEqual(status["word"], stack.WEIGHTS_PINNED); self.assertIn("pinned", stack.WEIGHTS_PINNED)
            self.assertEqual(status["sha256"]["model.safetensors"], self.pins["weights"]["biohub/ESMFold2-Fast"]["files"]["model.safetensors"]["sha256"])

    def test_same_size_other_digest_warns_by_name_and_proceeds(self):
        with tempfile.TemporaryDirectory() as hf:
            _lay_out(hf, self.pins, self.content, tamper=self.rel)
            why, notes, status = self._gate(hf)
            self.assertIsNone(why, "an unknown checkpoint is never a refusal")
            warn = [n for n in notes if n.startswith("WEIGHTS unknown")]
            self.assertEqual(len(warn), 1, notes); self.assertIn(f"{self.rel} sha256=", warn[0]); self.assertIn("(pinned ", warn[0]); self.assertIn("proceeding", warn[0])
            self.assertNotIn("bytes (pinned", warn[0])                                       # the size matched: the digest alone decided
            self.assertEqual(status["word"], stack.WEIGHTS_UNKNOWN)

    def test_other_size_warns_by_name_and_proceeds(self):
        with tempfile.TemporaryDirectory() as hf:
            _lay_out(hf, self.pins, self.content, resize=self.rel)
            why, notes, _ = self._gate(hf)
            self.assertIsNone(why)
            warn = [n for n in notes if n.startswith("WEIGHTS unknown")]
            self.assertEqual(len(warn), 1, notes); self.assertIn("bytes (pinned", warn[0])

    def test_an_absent_file_refuses_by_name(self):
        with tempfile.TemporaryDirectory() as hf:
            _lay_out(hf, self.pins, self.content, omit=self.rel)
            why, _, status = self._gate(hf)
            self.assertIsNotNone(why); self.assertIn("absent", why); self.assertIn(self.rel, why); self.assertEqual(status["word"], stack.WEIGHTS_ABSENT)

    def test_pins_carry_a_sha256_for_every_weights_file(self):
        real = stack.pins()
        for repo, w in real["weights"].items():
            for name, d in w["files"].items():
                self.assertRegex(d.get("sha256", ""), r"^[0-9a-f]{64}$", (repo, name)); self.assertIsInstance(d.get("size_bytes"), int, (repo, name))


if __name__ == "__main__":
    unittest.main()
