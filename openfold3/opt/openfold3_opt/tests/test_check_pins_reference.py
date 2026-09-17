"""stock/check_pins.py in a tree WITHOUT the wheel: the install is checked against its own RECORD and, file for file, against the tagged
source stock/src (upstream test-data files may be absent there and are counted by name); an edited install, a source file that differs,
a package file with no source counterpart, a tree with neither reference, and --wheel-vs-source without the wheel are each refused by
name. (With the wheel present the reference is the wheel's RECORD: test_install_verb, the clean tree.)"""
import base64
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

HOME = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
FILES = {"openfold3/__init__.py": b"V = 1\n", "openfold3/projects/of3_all_atom/config/model_config.py": b"C = 2\n", "openfold3/tests/test_data/x.npz": b"\0data"}


def _b64(digest):
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


class CheckPinsWithoutWheel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.stock = os.path.join(self.tmp, "kit", "stock"); os.makedirs(self.stock)
        shutil.copy(os.path.join(HOME, "stock", "check_pins.py"), self.stock)
        pins = json.load(open(os.path.join(HOME, "stock", "PINS.json"), encoding="utf-8"))
        json.dump(pins, open(os.path.join(self.stock, "PINS.json"), "w"))
        self.pins = pins
        self.site = os.path.join(self.tmp, "site")                                   # an installed openfold3 <pinned version> with a RECORD
        for name, data in FILES.items():
            p = os.path.join(self.site, name); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data)
        di = os.path.join(self.site, f"openfold3-{pins['openfold3_version']}.dist-info"); os.makedirs(di)
        open(os.path.join(di, "METADATA"), "w").write(f"Metadata-Version: 2.1\nName: openfold3\nVersion: {pins['openfold3_version']}\n")
        rec = [f"{n},sha256={_b64(hashlib.sha256(d).digest())},{len(d)}" for n, d in FILES.items()] + [f"openfold3-{pins['openfold3_version']}.dist-info/METADATA,,", f"openfold3-{pins['openfold3_version']}.dist-info/RECORD,,"]
        open(os.path.join(di, "RECORD"), "w").write("\n".join(rec) + "\n")
        for name, data in FILES.items():                                              # the tagged source: the two text files; the data file is not in this tree
            if name.endswith(".npz"): continue
            p = os.path.join(self.stock, "src", name); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data)
        sys.path.insert(0, self.site)
        spec = importlib.util.spec_from_file_location("check_pins_t", os.path.join(self.stock, "check_pins.py"))
        self.cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.cp)

    def tearDown(self):
        sys.path.remove(self.site); shutil.rmtree(self.tmp)

    def run_main(self, *argv):
        err = io.StringIO()
        with redirect_stderr(err): rc = self.cp.main(list(argv))
        return rc, err.getvalue()

    def test_install_checked_against_record_and_tag(self):
        rc, err = self.run_main()
        self.assertEqual(rc, 0, err)
        self.assertIn("installed==RECORD 3/3 files ==tag 0.4.1 d12f5955 2/2 source files (1 upstream test-data files not in this tree)", err)
        self.assertIn("wheel not in this tree (install route: python -m pip install --no-deps openfold3==", err)

    def test_an_edited_install_is_refused_by_name(self):
        open(os.path.join(self.site, "openfold3/projects/of3_all_atom/config/model_config.py"), "wb").write(b"C = 3\n")
        rc, err = self.run_main()
        self.assertEqual(rc, 3); self.assertIn("NOT STOCK: 1 installed files differ from the install RECORD: openfold3/projects/of3_all_atom/config/model_config.py", err)

    def test_a_source_file_that_differs_is_refused_by_name(self):
        open(os.path.join(self.stock, "src", "openfold3/__init__.py"), "wb").write(b"V = 9\n")
        rc, err = self.run_main()
        self.assertEqual(rc, 3); self.assertIn("differ from the tagged source stock/src (tag 0.4.1): ['openfold3/__init__.py']", err)

    def test_a_package_file_without_a_source_counterpart_is_refused_by_name(self):
        os.remove(os.path.join(self.stock, "src", "openfold3/__init__.py"))
        rc, err = self.run_main()
        self.assertEqual(rc, 3); self.assertIn("1 installed files have no counterpart in the tagged source stock/src (tag 0.4.1): ['openfold3/__init__.py']", err)

    def test_a_tree_with_neither_reference_is_refused_by_name(self):
        shutil.rmtree(os.path.join(self.stock, "src"))
        rc, err = self.run_main()
        self.assertEqual(rc, 3); self.assertIn("nor stock/src/openfold3 is in this tree: no pin reference", err)

    def test_wheel_vs_source_names_the_absent_wheel(self):
        rc, err = self.run_main("--wheel-vs-source")
        self.assertEqual(rc, 3); self.assertIn("--wheel-vs-source needs stock/openfold3-0.4.1-py3-none-any.whl, which is not in this tree", err)


if __name__ == "__main__":
    unittest.main()
