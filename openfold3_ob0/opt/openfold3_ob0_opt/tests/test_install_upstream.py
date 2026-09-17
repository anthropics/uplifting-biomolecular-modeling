"""stock/install_upstream.py — the three steps of `run.sh install` that make stock openfold3 present (the pinned wheel into stock/, the
distribution into the environment, upstream's source tree into stock/src), each on a COPY of stock/ in a temporary directory with injected
fetchers and a stand-in pin table: present / fetched / handed-over / off-the-pin / failed-fetch, and what is written in each case. No network,
no pip, no upstream import, CPU only."""
import hashlib
import importlib.util
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
PKG_FILES = {"openfold3/__init__.py": b"# stand-in package\n", "openfold3/core/model.py": b"X = 1\n"}   # the stand-in wheel's packaged files (RECORD lists them)
TREE_ONLY = {"examples/query.json": b"{}\n", "LICENSE": b"Apache\n"}                                    # tagged-tree files outside the package


def _b64(digest_bytes):
    import base64
    return base64.urlsafe_b64encode(digest_bytes).rstrip(b"=").decode()


def make_wheel(path):
    """A stand-in wheel: the packaged files + a dist-info RECORD with their sha256 (the form check_pins.record_hashes reads)."""
    import zipfile
    lines = []
    with zipfile.ZipFile(path, "w") as z:
        for name, data in PKG_FILES.items():
            z.writestr(name, data); lines.append(f"{name},sha256={_b64(hashlib.sha256(data).digest())},{len(data)}")
        lines.append("openfold3-0.5.0.dist-info/RECORD,,")
        z.writestr("openfold3-0.5.0.dist-info/RECORD", "\n".join(lines) + "\n")
    return path


def make_archive(path, top="openfold-3-cccc", files=None, extra_member=None):
    """A stand-in repository archive: one top directory holding the package files (byte-identical to the wheel's unless `files` says otherwise) + tree-only files."""
    files = {**PKG_FILES, **TREE_ONLY, **(files or {})}
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}"); info.size = len(data); t.addfile(info, io.BytesIO(data))
        if extra_member:
            info = tarfile.TarInfo(extra_member); info.size = 1; t.addfile(info, io.BytesIO(b"x"))
    return path


class _Stock:
    """A temporary copy of stock/ (check_pins.py, install_upstream.py) with a stand-in PINS.json pinned to the stand-in wheel; loads
    install_upstream from there so HERE is the copy."""
    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="ob0_upstream_")
        self.stock = os.path.join(self.dir, "stock"); os.makedirs(self.stock)
        for f in ("check_pins.py", "install_upstream.py"): shutil.copy(os.path.join(TREE, "stock", f), self.stock)
        self.elsewhere = os.path.join(self.dir, "elsewhere"); os.makedirs(self.elsewhere)
        self.good_wheel = make_wheel(os.path.join(self.elsewhere, "openfold3-0.5.0-py3-none-any.whl"))
        data = open(self.good_wheel, "rb").read()
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))      # the real table's shape; wheel digest / size re-pinned to the stand-in
        pins["wheel"]["sha256"], pins["wheel"]["bytes"] = hashlib.sha256(data).hexdigest(), len(data)
        pins["source"]["files"] = len(PKG_FILES) + len(TREE_ONLY)
        pins["upstream"]["repo"], pins["upstream"]["commit"] = "https://example.org/up/openfold-3", "c" * 40
        with open(os.path.join(self.stock, "PINS.json"), "w", encoding="utf-8") as fh: json.dump(pins, fh, indent=1)
        spec = importlib.util.spec_from_file_location("install_upstream_undertest", os.path.join(self.stock, "install_upstream.py"))
        self.M = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.M)
        self.pins = self.M.CP.read_pins(os.path.join(self.stock, "PINS.json"))
        self.wheel_dst = os.path.join(self.stock, "openfold3-0.5.0-py3-none-any.whl")
        self.src_dst = os.path.join(self.stock, "src")

    def cleanup(self): shutil.rmtree(self.dir, ignore_errors=True)


class WheelStep(unittest.TestCase):
    def setUp(self): self.S = _Stock(); self.out = io.StringIO(); self.fetched = []
    def tearDown(self): self.S.cleanup()

    def download(self, req, dest):                                      # the injected PyPI route: copies the stand-in wheel into dest
        self.fetched.append(req); return shutil.copy(self.S.good_wheel, dest)

    def test_absent_is_fetched_judged_and_placed(self):
        rc = self.S.M.wheel(self.S.pins, download=self.download, out=self.out)
        self.assertEqual(rc, 0, self.out.getvalue()); self.assertTrue(os.path.isfile(self.S.wheel_dst))
        self.assertEqual(self.fetched, ["openfold3==0.5.0"]); self.assertIn("WHEEL OK", self.out.getvalue()); self.assertIn("fetched from PyPI (openfold3==0.5.0)", self.out.getvalue())
        self.assertEqual([f for f in os.listdir(self.S.stock) if f.startswith(".wheel-")], [])            # the staging directory is gone

    def test_present_at_the_pin_is_only_judged(self):
        shutil.copy(self.S.good_wheel, self.S.wheel_dst)
        rc = self.S.M.wheel(self.S.pins, download=self.download, out=self.out)
        self.assertEqual((rc, self.fetched), (0, []), self.out.getvalue()); self.assertIn("— present", self.out.getvalue())

    def test_present_off_the_pin_is_refused_and_left_in_place(self):
        with open(self.S.wheel_dst, "wb") as fh: fh.write(b"not the wheel")
        rc = self.S.M.wheel(self.S.pins, download=self.download, out=self.out)
        self.assertEqual((rc, self.fetched), (1, []), self.out.getvalue()); self.assertIn("REFUSED", self.out.getvalue())
        self.assertEqual(open(self.S.wheel_dst, "rb").read(), b"not the wheel")

    def test_a_fetched_file_off_the_pin_is_refused_and_nothing_is_placed(self):
        def bad(req, dest):
            p = os.path.join(dest, "openfold3-0.5.0-py3-none-any.whl"); open(p, "wb").write(b"tampered"); return p
        rc = self.S.M.wheel(self.S.pins, download=bad, out=self.out)
        self.assertEqual(rc, 1, self.out.getvalue()); self.assertIn("REFUSED", self.out.getvalue()); self.assertFalse(os.path.exists(self.S.wheel_dst))

    def test_a_failed_fetch_names_the_offline_remedy(self):
        def down(req, dest): raise OSError("network unreachable")
        rc = self.S.M.wheel(self.S.pins, download=down, out=self.out)
        self.assertEqual(rc, 1); self.assertIn("FAILED", self.out.getvalue()); self.assertIn("run.sh install --wheel FILE", self.out.getvalue()); self.assertFalse(os.path.exists(self.S.wheel_dst))

    def test_a_copy_handed_over_is_judged_then_placed_without_a_fetch(self):
        rc = self.S.M.wheel(self.S.pins, source=self.S.good_wheel, download=self.download, out=self.out)
        self.assertEqual((rc, self.fetched), (0, []), self.out.getvalue()); self.assertIn("placed from --wheel", self.out.getvalue()); self.assertTrue(os.path.isfile(self.S.wheel_dst))
        S2 = _Stock()
        try:
            other = os.path.join(S2.elsewhere, "other.whl"); open(other, "wb").write(b"other bytes")
            out = io.StringIO(); rc = S2.M.wheel(S2.pins, source=other, download=self.download, out=out)
            self.assertEqual(rc, 1, out.getvalue()); self.assertIn("REFUSED", out.getvalue()); self.assertFalse(os.path.exists(S2.wheel_dst))
            out = io.StringIO(); rc = S2.M.wheel(S2.pins, source=os.path.join(S2.elsewhere, "absent.whl"), download=self.download, out=out)
            self.assertEqual(rc, 1); self.assertIn("is not a file", out.getvalue())
        finally:
            S2.cleanup()

    def test_the_pypi_route_is_pip_download_of_the_pinned_requirement(self):
        """pip_download runs `<python> -m pip download --no-deps --only-binary :all: … <requirement>` and returns the one wheel written (a stand-in runner records the call)."""
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd); dest = cmd[cmd.index("-d") + 1]; shutil.copy(self.S.good_wheel, dest)
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        dl = os.path.join(self.S.elsewhere, "dl"); os.makedirs(dl)
        real = self.S.M.subprocess.run; self.S.M.subprocess.run = fake_run
        try:
            got = self.S.M.pip_download("openfold3==0.5.0", dl)
        finally:
            self.S.M.subprocess.run = real
        cmd = calls[-1]
        self.assertEqual(cmd[1:4], ["-m", "pip", "download"]); self.assertIn("--no-deps", cmd); self.assertIn(":all:", cmd); self.assertEqual(cmd[-1], "openfold3==0.5.0")
        self.assertTrue(got.endswith(".whl"))


class DistributionStep(unittest.TestCase):
    def setUp(self): self.S = _Stock(); self.out = io.StringIO(); self.installed = []
    def tearDown(self): self.S.cleanup()

    def test_an_installed_openfold3_is_left_to_the_pin_check(self):
        rc = self.S.M.distribution(self.S.pins, install=self.installed.append, version=lambda: "0.5.0", out=self.out)
        self.assertEqual((rc, self.installed), (0, []), self.out.getvalue()); self.assertIn("INSTALLED: openfold3 0.5.0 is installed", self.out.getvalue())
        out = io.StringIO(); rc = self.S.M.distribution(self.S.pins, install=self.installed.append, version=lambda: "0.4.1", out=out)   # another version: not reinstalled here — check_pins refuses it by name right after
        self.assertEqual((rc, self.installed), (0, [])); self.assertIn("openfold3 0.4.1 is installed", out.getvalue())

    def test_an_absent_openfold3_is_installed_from_the_pinned_wheel(self):
        state = {"v": None}
        def install(path): self.installed.append(path); state["v"] = "0.5.0"
        rc = self.S.M.distribution(self.S.pins, install=install, version=lambda: state["v"], out=self.out)
        self.assertEqual(rc, 0, self.out.getvalue()); self.assertEqual(self.installed, [self.S.wheel_dst]); self.assertIn("INSTALLED: openfold3 0.5.0 from stock/", self.out.getvalue())

    def test_a_failed_pip_install_is_relayed(self):
        def install(path): raise RuntimeError("pip install --no-deps x.whl exited 1: ['read-only file system']")
        rc = self.S.M.distribution(self.S.pins, install=install, version=lambda: None, out=self.out)
        self.assertEqual(rc, 1); self.assertIn("FAILED", self.out.getvalue()); self.assertIn("read-only", self.out.getvalue())


class SourceStep(unittest.TestCase):
    def setUp(self):
        self.S = _Stock(); self.out = io.StringIO(); self.urls = []
        shutil.copy(self.S.good_wheel, self.S.wheel_dst)                # the source check reads the wheel's RECORD (step 1 ran)
    def tearDown(self): self.S.cleanup()

    def download(self, url, dest, **kw):
        self.urls.append(url); return make_archive(os.path.join(dest, "src.tar.gz"), **kw)

    def test_absent_is_fetched_extracted_checked_and_placed(self):
        rc = self.S.M.source(self.S.pins, download=self.download, out=self.out)
        self.assertEqual(rc, 0, self.out.getvalue())
        self.assertEqual(self.urls, ["https://example.org/up/openfold-3/archive/" + "c" * 40 + ".tar.gz"])   # <repo>/archive/<commit>.tar.gz from the pin table
        self.assertTrue(os.path.isfile(os.path.join(self.S.src_dst, "openfold3", "core", "model.py"))); self.assertTrue(os.path.isfile(os.path.join(self.S.src_dst, "examples", "query.json")))
        self.assertIn("SOURCE OK", self.out.getvalue()); self.assertIn("2/2 packaged files byte-identical", self.out.getvalue()); self.assertIn("4 files)", self.out.getvalue())
        self.assertEqual([f for f in os.listdir(self.S.stock) if f.startswith(".src-")], [])

    def test_present_and_identical_is_only_checked(self):
        self.S.M.extract_tree(make_archive(os.path.join(self.S.elsewhere, "a.tgz")), self.S.src_dst)
        rc = self.S.M.source(self.S.pins, download=self.download, out=self.out)
        self.assertEqual((rc, self.urls), (0, []), self.out.getvalue()); self.assertIn("— present", self.out.getvalue())

    def test_present_but_edited_is_refused_and_left_in_place(self):
        self.S.M.extract_tree(make_archive(os.path.join(self.S.elsewhere, "a.tgz"), files={"openfold3/core/model.py": b"X = 2  # edited\n"}), self.S.src_dst)
        rc = self.S.M.source(self.S.pins, download=self.download, out=self.out)
        self.assertEqual((rc, self.urls), (1, []), self.out.getvalue()); self.assertIn("REFUSED", self.out.getvalue()); self.assertIn("NOT STOCK", self.out.getvalue())
        self.assertTrue(os.path.isdir(self.S.src_dst))

    def test_a_fetched_tree_off_the_tag_is_refused_and_nothing_is_placed(self):
        rc = self.S.M.source(self.S.pins, download=lambda u, d: self.download(u, d, files={"openfold3/core/model.py": b"X = 3\n"}), out=self.out)
        self.assertEqual(rc, 1, self.out.getvalue()); self.assertIn("REFUSED", self.out.getvalue()); self.assertFalse(os.path.exists(self.S.src_dst))
        self.assertEqual([f for f in os.listdir(self.S.stock) if f.startswith(".src-")], [])

    def test_an_archive_with_a_member_outside_its_tree_is_refused(self):
        rc = self.S.M.source(self.S.pins, download=lambda u, d: self.download(u, d, extra_member="../escape.txt"), out=self.out)
        self.assertEqual(rc, 1, self.out.getvalue()); self.assertIn("FAILED: extracting", self.out.getvalue()); self.assertFalse(os.path.exists(self.S.src_dst))
        self.assertFalse(os.path.exists(os.path.join(self.S.dir, "escape.txt")))

    def test_a_failed_fetch_names_the_offline_remedy_and_a_handed_over_archive_needs_no_fetch(self):
        def down(url, dest): raise OSError("network unreachable")
        rc = self.S.M.source(self.S.pins, download=down, out=self.out)
        self.assertEqual(rc, 1); self.assertIn("run.sh install --src TARBALL", self.out.getvalue()); self.assertFalse(os.path.exists(self.S.src_dst))
        tgz = make_archive(os.path.join(self.S.elsewhere, "handed.tar.gz"))
        out = io.StringIO(); rc = self.S.M.source(self.S.pins, tarball=tgz, download=down, out=out)
        self.assertEqual(rc, 0, out.getvalue()); self.assertIn("extracted from --src", out.getvalue()); self.assertTrue(os.path.isdir(self.S.src_dst))


class MainRoutes(unittest.TestCase):
    def setUp(self): self.S = _Stock()
    def tearDown(self): self.S.cleanup()

    def test_wheel_only_stops_after_the_distribution(self):
        M, seen = self.S.M, []
        real = (M.wheel, M.distribution, M.source)
        M.wheel = lambda pins, source=None, out=None, **k: seen.append(("wheel", source)) or 0
        M.distribution = lambda pins, out=None, **k: seen.append(("dist",)) or 0
        M.source = lambda pins, tarball=None, out=None, **k: seen.append(("src", tarball)) or 0
        try:
            self.assertEqual(M.main(["--wheel-only"], out=io.StringIO()), 0); self.assertEqual(seen, [("wheel", None), ("dist",)])
            seen.clear(); self.assertEqual(M.main(["--wheel", "/x.whl", "--src", "/s.tgz"], out=io.StringIO()), 0)
            self.assertEqual(seen, [("wheel", "/x.whl"), ("dist",), ("src", "/s.tgz")])
            seen.clear(); M.wheel = lambda pins, source=None, out=None, **k: seen.append(("wheel", source)) or 1     # a refused wheel stops the sequence with its code
            self.assertEqual(M.main([], out=io.StringIO()), 1); self.assertEqual(seen, [("wheel", None)])
            self.assertEqual(M.main(["--bogus"], out=io.StringIO()), 2)
        finally:
            M.wheel, M.distribution, M.source = real

    def test_the_kits_own_tree_is_stock_at_the_pin(self):
        """The tree under test: its wheel (when present) is the pinned wheel, and its stock/src (when present) is the tagged tree — the real table."""
        spec = importlib.util.spec_from_file_location("install_upstream_real", os.path.join(TREE, "stock", "install_upstream.py"))
        M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
        pins = M.CP.read_pins()
        self.assertEqual(M.archive_url(pins), f"{pins['upstream']['repo']}/archive/{pins['upstream']['commit']}.tar.gz")
        w = M.wheel_path(pins)
        if os.path.isfile(w): self.assertTrue(M.judge_wheel(w, pins)[0], "stock/ wheel is not the one stock/PINS.json pins")
        if os.path.isdir(os.path.join(TREE, "stock", "src")) and os.path.isfile(w):
            bad, d = M.CP.check_wheel_vs_source(pins); self.assertEqual(bad, [], bad); self.assertEqual(d["identical"], d["files"])


if __name__ == "__main__":
    unittest.main()
