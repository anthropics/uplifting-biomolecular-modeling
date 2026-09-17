"""The stock pin: PINS.json checkout relpaths = the vendored archive's members with the kit's own patched copies;
stock/src/af2_initial_guess/predict.py = the archive member."""
import hashlib
import json
import os
import tarfile
import unittest

from . import _stubs

STOCK = os.path.join(_stubs.TREE, "stock")


def _members(tf):
    out = {}
    for m in tf.getmembers():
        if m.isfile() and "__pycache__" not in m.name.split("/") and not m.name.endswith(".pyc"):
            with tf.extractfile(m) as fh:
                out[m.name.split("/", 1)[1]] = hashlib.sha256(fh.read()).hexdigest()      # a tar member's bytes in memory (no file path): not a file digest, so not opt_core.gates.sha256_file
    return out


class TestStockArchives(unittest.TestCase):
    def setUp(self):
        self.pins = json.load(open(os.path.join(STOCK, "PINS.json"), encoding="utf-8"))
        with tarfile.open(os.path.join(STOCK, self.pins["upstream"]["archive"])) as tf:
            self.members = _members(tf)

    def test_checkout_files(self):
        C = self.pins["checkout"]
        kit_patched = os.path.join(_stubs.KIT, "patches", "patched_files")
        for rel in C["patched_relpaths"]:
            self.assertTrue(os.path.isfile(os.path.join(kit_patched, rel)), rel)   # the kit carries its own patched copy of every patched relpath
        for rel in C["relpaths"]:
            if rel in C["patched_relpaths"]:
                continue
            self.assertIn(rel, self.members, rel)                          # every other relpath of the checkout is a member of the vendored archive
        self.assertEqual(len(C["relpaths"]), C["n_files"])
        self.assertEqual(len([r for r in self.members if r.startswith(C["subtree"] + "/")]) + 1, C["n_files"])   # + predict_pdb.py, new

    def test_src_copy(self):
        self.assertEqual(_stubs.sha256(os.path.join(STOCK, "src", "af2_initial_guess", "predict.py")), self.members["af2_initial_guess/predict.py"])


if __name__ == "__main__":
    unittest.main()
