"""The shipped extension objects are held to their SHA256SUMS lines before anything loads them (kits.v0_2.sums_refusal, called by
kits.v0_2.check_files — the one place the package resolves the objects it hands to the loaders).

The tree: every compiled object the kit ships under opt/forward (.so, .cubin, .cubin.xz, .ptx, .fatbin — the two objects in bin/ and the two
in classes/a100_80gb/) is a `<sha256>  <name>` line of the SHA256SUMS in its own directory, digest equal, and those lines name the same
objects BUILD.json and CLASS_PINS.json describe, so nothing shipped is refused. The rule: listed and equal passes (recorded in HELD); a
zeroed or differing digest, altered bytes, an unlisted object, or no SHA256SUMS at all is a refusal by name with one stderr line, and
check_files raises FileNotFoundError — the missing-object path, which the package reports as NOT ACTIVE — so no path reaches a loader.
No torch and no GPU are needed: the device probe is pinned to "no device", where the kit's own objects are the ones resolved.
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from enformer_opt import _runtime as R

K = R._kit()                                                                                   # engines.enformer.kits.v0_2, from the vendored closure
from engines.enformer.kits import class_pins  # noqa: E402  (importable once _kit() put the closure on the path)

FORWARD = os.path.join(R.OPT_HOME, "forward")
CLASS_DIR = os.path.join(FORWARD, "classes", "a100_80gb")
BINARY_SUFFIXES = (".so", ".cubin", ".xz", ".ptx", ".fatbin")
ZERO = "0" * 64
LOADER_MODULES = ("enformer_fastkit", "enformer_xattn")                                        # the modules that map an object (attach imports them after check_files)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _binaries(root):
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        out += [os.path.join(base, f) for f in sorted(files) if f.endswith(BINARY_SUFFIXES)]
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
    """Each test starts from empty verdict caches and a device probe that reports no device (the kit's own objects resolve); loader modules
    imported during a test are recorded."""

    def setUp(self):
        self._saved = (dict(K._VERDICTS), list(K.HELD))
        K._VERDICTS.clear(); del K.HELD[:]
        self._probe = mock.patch.object(class_pins, "device_sm", lambda: None)
        self._probe.start()
        self._modules_before = set(sys.modules)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        self._probe.stop()
        verdicts, held = self._saved
        K._VERDICTS.clear(); K._VERDICTS.update(verdicts); del K.HELD[:]; K.HELD.extend(held)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def loaded(self):
        """Loader modules imported since setUp (a refusal must leave this empty)."""
        return sorted(m for m in set(sys.modules) - self._modules_before if m.rsplit(".", 1)[-1] in LOADER_MODULES)

    def _copy(self):
        """Copies of the kit's own two objects in a directory of their own, no SHA256SUMS beside them yet -> the so_paths dict check_files takes."""
        d = tempfile.mkdtemp(dir=self.tmp)
        paths = {}
        for key in K.BINARIES:
            paths[key] = os.path.join(d, K.KIT_PINS[key])
            shutil.copyfile(os.path.join(R.BIN_DIR, K.KIT_PINS[key]), paths[key])
        return paths

    def _lines(self, paths, **override):
        """SHA256SUMS text listing both copies by their real digests, with per-key overrides (digest string, or None to leave the key out)."""
        text = ""
        for key, p in paths.items():
            digest = override.get(key, _sha(p))
            if digest is not None:
                text += "%s  %s\n" % (digest, os.path.basename(p))
        return text


class TestTree(_State):
    def test_every_shipped_object_is_a_sha256sums_line_with_its_digest(self):
        bins = _binaries(FORWARD)
        self.assertEqual(sorted(os.path.relpath(p, FORWARD) for p in bins),
                         sorted(os.path.join(sub, K.KIT_PINS[key]) for sub in ("bin", os.path.join("classes", "a100_80gb")) for key in K.BINARIES))
        for p in bins:
            sums = os.path.join(os.path.dirname(p), "SHA256SUMS")
            self.assertTrue(os.path.isfile(sums), f"{os.path.relpath(p, FORWARD)}: no SHA256SUMS in its directory")
            listed = _read_sums(sums)
            self.assertIn(os.path.basename(p), listed, f"{os.path.relpath(p, FORWARD)}: not listed in {os.path.relpath(sums, FORWARD)}")
            self.assertEqual(_sha(p), listed[os.path.basename(p)], f"{os.path.relpath(p, FORWARD)}: sha256 differs from {os.path.relpath(sums, FORWARD)}")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            refused = [(os.path.relpath(p, FORWARD), K.sums_refusal(p)) for p in bins if K.sums_refusal(p) is not None]
        self.assertEqual(refused, []); self.assertEqual(err.getvalue(), "")                          # the loader's reading: nothing shipped is refused
        self.assertEqual(K.HELD, [(os.path.relpath(p, FORWARD).replace(os.sep, "/"), _sha(p)[:16]) for p in bins])
        self.assertEqual(os.path.abspath(K.FORWARD_DIR), os.path.abspath(FORWARD))

    def test_the_sums_lines_name_the_objects_build_json_and_class_pins_describe(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(K.__file__)), "BUILD.json"), encoding="utf-8") as f:
            build = json.load(f)
        own = {os.path.basename(k): v["sha256"] for k, v in build["objects"].items()}
        self.assertEqual(_read_sums(os.path.join(R.BIN_DIR, "SHA256SUMS")), own)                     # bin/: the two objects of BUILD.json
        self.assertIn("SHA256SUMS", build["rebuild"])                                                 # the rebuild note names the line a rebuilt object needs
        with open(os.path.join(CLASS_DIR, class_pins.FILE), encoding="utf-8") as f:
            cp = json.load(f)
        self.assertEqual(_read_sums(os.path.join(CLASS_DIR, "SHA256SUMS")), {n: b["sha256"] for n, b in cp["binaries"].items()})   # the class build's


class TestRule(_State):
    def test_listed_and_equal_passes_into_the_stamp(self):
        paths = self._copy()
        with open(os.path.join(os.path.dirname(paths["fused_so"]), "SHA256SUMS"), "w", encoding="utf-8") as f:
            f.write(self._lines(paths))                                                                # both copies listed by their digests
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            stamp = K.check_files(paths)
        for key in K.BINARIES:
            self.assertEqual(stamp[f"{key}_path"], paths[key]); self.assertEqual(stamp[f"{key}_sha256"], _sha(paths[key]))
        self.assertIsNone(stamp["class_pins"]); self.assertEqual(err.getvalue(), "")
        self.assertEqual(K.HELD, [(paths[key], _sha(paths[key])[:16]) for key in K.BINARIES])       # outside opt/forward: named by full path

    def test_zeroed_digest_is_refused_by_name_and_nothing_loads(self):
        paths = self._copy()
        so = paths["fused_so"]; sums = os.path.join(os.path.dirname(so), "SHA256SUMS")
        with open(sums, "w", encoding="utf-8") as f:
            f.write(self._lines(paths, fused_so=ZERO))
        why = "sha256 %s != %s %s" % (_sha(so)[:16], sums, ZERO[:16])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(FileNotFoundError) as cm:
                K.check_files(paths)
            self.assertEqual(K.sums_refusal(so), why)                                                 # one verdict per path: no second hash, no second line
        self.assertIn("kit v0.2: %s is not the shipped object (%s)" % (so, why), str(cm.exception)); self.assertIn("BUILD.json", str(cm.exception))
        self.assertEqual(err.getvalue(), "[enformer-opt] SHA256SUMS: refused %s: %s\n" % (so, why))
        self.assertEqual(K.HELD, []); self.assertEqual(self.loaded(), [])

    def test_altered_bytes_are_refused(self):
        paths = self._copy()
        so = paths["xattn_so"]
        with open(os.path.join(os.path.dirname(so), "SHA256SUMS"), "w", encoding="utf-8") as f:
            f.write(self._lines(paths))                                                                # both lines written from the copies as they are
        with open(so, "r+b") as f:
            f.seek(64); b = f.read(1); f.seek(64); f.write(bytes([b[0] ^ 1]))                            # then one bit of one copy flipped
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(FileNotFoundError) as cm:
                K.check_files(paths)
        self.assertTrue(K.sums_refusal(so).startswith("sha256 %s != " % _sha(so)[:16])); self.assertIn(so, str(cm.exception))
        self.assertEqual([rel for rel, _ in K.HELD], [paths["fused_so"]]); self.assertEqual(self.loaded(), [])   # the untouched copy passed, the altered one did not

    def test_unlisted_object_is_refused(self):
        paths = self._copy()
        so = paths["fused_so"]; sums = os.path.join(os.path.dirname(so), "SHA256SUMS")
        with open(sums, "w", encoding="utf-8") as f:
            f.write(self._lines(paths, fused_so=None) + "%s  other.so\n" % _sha(so))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(FileNotFoundError):
                K.check_files(paths)
        self.assertEqual(K.sums_refusal(so), "not listed in %s" % sums)
        self.assertIn("refused %s: not listed in " % so, err.getvalue()); self.assertEqual(self.loaded(), [])

    def test_no_sha256sums_is_refused(self):
        paths = self._copy()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                K.check_files(paths)
        self.assertEqual(K.sums_refusal(paths["fused_so"]), "no SHA256SUMS in its directory lists it"); self.assertEqual(self.loaded(), [])

    def test_missing_object_is_the_same_refusal_class(self):
        paths = {key: os.path.join(self.tmp, "nowhere", K.KIT_PINS[key]) for key in K.BINARIES}
        with self.assertRaises(FileNotFoundError) as cm:
            K.check_files(paths)
        self.assertIn("missing", str(cm.exception)); self.assertEqual(K.HELD, []); self.assertEqual(self.loaded(), [])

    def test_the_bytes_already_read_are_what_is_hashed(self):
        paths = self._copy()
        so = paths["fused_so"]; sums = os.path.join(os.path.dirname(so), "SHA256SUMS")
        with open(sums, "w", encoding="utf-8") as f:
            f.write(self._lines(paths))
        with contextlib.redirect_stderr(io.StringIO()):
            why = K.sums_refusal(so, data=b"in-hand")
        self.assertEqual(why, "sha256 %s != %s %s" % (hashlib.sha256(b"in-hand").hexdigest()[:16], sums, _sha(so)[:16]))


class TestPackageLine(_State):
    """Through the package: an object that is not the shipped one makes `check` / `enable` say NOT ACTIVE by name (the missing-file reason
    class), with the refusal line before it; nothing is hooked."""

    def test_resolve_reports_not_active_by_name(self):
        paths = self._copy()
        d = os.path.dirname(paths["fused_so"]); sums = os.path.join(d, "SHA256SUMS")
        with open(sums, "w", encoding="utf-8") as f:
            f.write(self._lines(paths, xattn_so=ZERO))
        dev = {"name": "NVIDIA H100 80GB HBM3", "sm": (9, 0), "memory_mib": 81559, "torch": R.TESTED_TORCH, "cuda": "13.0", "cudnn": 92000}
        err = io.StringIO()
        with contextlib.redirect_stderr(err), mock.patch.object(R, "_device", lambda: dev), mock.patch.object(R, "_upstream_version", lambda: R.UPSTREAM_VERSION), \
                mock.patch.object(R, "BIN_DIR", d):
            rep = R.check()
        why = "sha256 %s != %s %s" % (_sha(paths["xattn_so"])[:16], sums, ZERO[:16])
        self.assertFalse(rep["active"]); self.assertFalse(R._HOOK["installed"])
        self.assertEqual(rep["reason"], "FileNotFoundError: kit v0.2: %s is not the shipped object (%s) — restore it, or rebuild it and its SHA256SUMS line (BUILD.json)" % (paths["xattn_so"], why))
        self.assertEqual(err.getvalue().splitlines(), ["[enformer-opt] SHA256SUMS: refused %s: %s" % (paths["xattn_so"], why),
                                                       "[enformer-opt] NOT ACTIVE mode=exact reason=%s" % rep["reason"]])
        self.assertEqual(self.loaded(), [])


if __name__ == "__main__":
    unittest.main()
