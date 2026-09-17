"""The shipped op library is held to its SHA256SUMS line before TensorFlow maps it (ops.sums_refusal, ops.load; ops/build.sh writes the line).

The tree: every compiled binary the package ships (.so, .cubin, .cubin.xz, .ptx, .fatbin — today ops/edm_ops.so) is a `<sha256>  <name>`
line of the SHA256SUMS in its own directory, digest equal, so nothing shipped is refused. The rule: listed and equal passes (recorded in
ops.HELD); a zeroed or differing digest, an unlisted library, or no SHA256SUMS at all is a refusal by name with one stderr line, and load()
raises FileNotFoundError before tf.load_op_library is reached — the absent-library path. No TensorFlow is needed: a stand-in module records
whether load_op_library was called.
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, OPT)
from enformer_deepmind_opt import ops  # noqa: E402

PKG = os.path.dirname(os.path.dirname(os.path.abspath(ops.__file__)))                                 # enformer_deepmind_opt/
BINARY_SUFFIXES = (".so", ".cubin", ".xz", ".ptx", ".fatbin")
ZERO = "0" * 64


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _binaries(package):
    out = []
    for root, dirs, files in os.walk(package):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith(BINARY_SUFFIXES)]
    return out


def _read_sums(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for n, ln in enumerate(f, 1):
            if not ln.strip() or ln.startswith("#"):
                continue
            digest, rel = ln.rstrip("\n").split(None, 1)
            assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), (path, n)      # sha256sum format
            assert not rel.startswith("/") and ".." not in rel.split("/"), (path, n)
            out[rel.strip()] = digest
    return out


class _State(unittest.TestCase):
    """Each test starts from empty verdict caches, the real SO_PATH and no loaded library; a stand-in `tensorflow` records load_op_library."""

    def setUp(self):
        self._saved = (ops.SO_PATH, ops._lib, dict(ops._VERDICTS), list(ops.HELD), sys.modules.get("tensorflow", None))
        ops._VERDICTS.clear(); del ops.HELD[:]; ops._lib = None
        self.loaded = []
        fake = types.ModuleType("tensorflow")
        fake.__version__ = ops.build_info().get("tensorflow")
        fake.load_op_library = lambda path: (self.loaded.append(path), ("module", path))[1]
        sys.modules["tensorflow"] = fake
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        ops.SO_PATH, ops._lib, verdicts, held, tf = self._saved
        ops._VERDICTS.clear(); ops._VERDICTS.update(verdicts); del ops.HELD[:]; ops.HELD.extend(held)
        if tf is None:
            sys.modules.pop("tensorflow", None)
        else:
            sys.modules["tensorflow"] = tf
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _copy(self, sums_text):
        """A copy of the shipped library in a directory of its own, with the given SHA256SUMS text beside it (None: no SHA256SUMS)."""
        d = tempfile.mkdtemp(dir=self.tmp)
        so = os.path.join(d, "edm_ops.so")
        shutil.copyfile(ops.SO_PATH, so)
        if sums_text is not None:
            with open(os.path.join(d, "SHA256SUMS"), "w", encoding="utf-8") as f:
                f.write(sums_text)
        return so


class TestTree(_State):
    def test_every_shipped_binary_is_a_sha256sums_line_with_its_digest(self):
        bins = _binaries(PKG)
        self.assertGreaterEqual(len(bins), 1)
        self.assertIn(os.path.abspath(ops.SO_PATH), [os.path.abspath(p) for p in bins])
        for p in bins:
            sums = os.path.join(os.path.dirname(p), "SHA256SUMS")
            self.assertTrue(os.path.isfile(sums), f"{os.path.relpath(p, PKG)}: no SHA256SUMS in its directory")
            listed = _read_sums(sums)
            self.assertIn(os.path.basename(p), listed, f"{os.path.relpath(p, PKG)}: not listed in {os.path.relpath(sums, PKG)}")
            self.assertEqual(_sha(p), listed[os.path.basename(p)], f"{os.path.relpath(p, PKG)}: sha256 differs from {os.path.relpath(sums, PKG)}")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            refused = [(os.path.relpath(p, PKG), ops.sums_refusal(p)) for p in bins if ops.sums_refusal(p) is not None]
        self.assertEqual(refused, []); self.assertEqual(err.getvalue(), "")                          # the loader's reading: nothing shipped is refused
        self.assertEqual(ops.HELD, [(os.path.relpath(p, PKG).replace(os.sep, "/"), _sha(p)[:16]) for p in bins])

    def test_the_sums_line_and_build_json_name_the_same_library(self):
        listed = _read_sums(ops.SUMS_PATH)
        self.assertEqual(list(listed), ["edm_ops.so"])
        self.assertEqual(listed["edm_ops.so"], ops.build_info()["object"]["sha256"])                  # build.sh writes both from the one object


class TestRule(_State):
    def test_listed_and_equal_loads(self):
        so = self._copy("%s  edm_ops.so\n" % _sha(ops.SO_PATH))
        ops.SO_PATH = so
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            lib = ops.load()
        self.assertEqual(lib, ("module", so)); self.assertEqual(self.loaded, [so])                  # tf.load_op_library reached once, with the file
        self.assertEqual(err.getvalue(), ""); self.assertEqual(ops.HELD, [(so, _sha(so)[:16])])

    def test_zeroed_digest_is_refused_by_name_and_nothing_is_loaded(self):
        so = self._copy("%s  edm_ops.so\n" % ZERO)
        ops.SO_PATH = so
        why = "sha256 %s != %s %s" % (_sha(so)[:16], os.path.join(os.path.dirname(so), "SHA256SUMS"), ZERO[:16])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(FileNotFoundError) as cm:
                ops.load()
            self.assertEqual(ops.sums_refusal(so), why)                                                 # one verdict per path: no second hash, no second line
        self.assertIn("is not the shipped library (%s)" % why, str(cm.exception)); self.assertIn("build.sh", str(cm.exception))
        self.assertEqual(err.getvalue(), "[enformer-deepmind-opt] SHA256SUMS: refused %s: %s\n" % (so, why))
        self.assertEqual(self.loaded, []); self.assertIsNone(ops._lib); self.assertEqual(ops.HELD, [])

    def test_altered_bytes_are_refused(self):
        so = self._copy("%s  edm_ops.so\n" % _sha(ops.SO_PATH))
        with open(so, "r+b") as f:
            f.seek(64); b = f.read(1); f.seek(64); f.write(bytes([b[0] ^ 1]))                            # one bit of the copy flipped after its line was written
        ops.SO_PATH = so
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                ops.load()
        self.assertTrue(ops.sums_refusal(so).startswith("sha256 %s != " % _sha(so)[:16])); self.assertEqual(self.loaded, [])

    def test_unlisted_library_is_refused(self):
        so = self._copy("%s  other.so\n" % _sha(ops.SO_PATH))
        ops.SO_PATH = so
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(FileNotFoundError):
                ops.load()
        self.assertEqual(ops.sums_refusal(so), "not listed in %s" % os.path.join(os.path.dirname(so), "SHA256SUMS"))
        self.assertIn("refused %s: not listed in " % so, err.getvalue()); self.assertEqual(self.loaded, [])

    def test_no_sha256sums_is_refused(self):
        so = self._copy(None)
        ops.SO_PATH = so
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                ops.load()
        self.assertEqual(ops.sums_refusal(so), "no SHA256SUMS in its directory lists it"); self.assertEqual(self.loaded, [])

    def test_absent_library_is_the_same_refusal_class(self):
        ops.SO_PATH = os.path.join(self.tmp, "nowhere", "edm_ops.so")
        with self.assertRaises(FileNotFoundError):
            ops.load()
        self.assertEqual(self.loaded, [])

    def test_the_bytes_already_read_are_what_is_hashed(self):
        so = self._copy("%s  edm_ops.so\n" % _sha(ops.SO_PATH))
        with contextlib.redirect_stderr(io.StringIO()):
            why = ops.sums_refusal(so, data=b"in-hand")
        self.assertEqual(why, "sha256 %s != %s %s" % (hashlib.sha256(b"in-hand").hexdigest()[:16], os.path.join(os.path.dirname(so), "SHA256SUMS"), _sha(so)[:16]))


class TestBuildScript(unittest.TestCase):
    def test_build_sh_writes_the_sums_line_it_loads_against(self):
        with open(os.path.join(os.path.dirname(ops.SO_PATH), "build.sh"), encoding="utf-8") as f:
            text = f.read()
        self.assertIn('open(os.path.join(here, "SHA256SUMS"), "w")', text)
        self.assertIn('build["object"]["sha256"] + "  edm_ops.so\\n"', text)


if __name__ == "__main__":
    unittest.main()
