"""Every carried/vendored file this repo tracks is the git commit's own bytes -- checked live against its upstream archive member,
never against a stored digest of an in-repo file. Two trees: the vendored ColabDesign source under stock/src/ (excluding bindcraft/,
covered separately below) against its pinned tar.gz; the vendored BindCraft tree (stock/src/bindcraft/, `names.BINDCRAFT_DIR`) against
upstream.bindcraft's archive and file list; the one file that tree does not carry — BindCraft's prebuilt DSSP executable, fetched by
`run.sh install` (upstream.bindcraft.fetched) — is checked against its pin when present and is absent from the archive by recipe. The kit
itself carries no kernel or lever copy (the levers import the shared core by name) and nothing under opt/ but the package. The pinned settings (PINS.json `settings.file`) names the loader's default file; the loader refuses an
absent or re-pointed default by name (`SettingsError`) before reading it."""

import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import unittest
from . import _stubs
from colabdesign_opt import names, settings


# ============================================================ ColabDesign side: stock/src/, excluding bindcraft/ (below)
STOCK = os.path.join(_stubs.TREE, "stock")
COLABDESIGN_ARCHIVE = "colabdesign-e31a56fe.tar.gz"
COLABDESIGN_PREFIX = "colabdesign-e31a56fe/"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def colabdesign_src_files():
    """stock/src/, relative, '/'-separated, excluding the bindcraft/ subtree (its own archive-membership test is
    TestBindcraftVendoredBytes, below)."""
    root = os.path.join(STOCK, "src")
    out = []
    for r, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in fs:
            rel = os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/")
            if not rel.startswith("bindcraft/"):
                out.append(rel)
    return sorted(out)


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestStockArchives(unittest.TestCase):
    def test_extracts_are_archive_members(self):
        """Every extracted ColabDesign file is its archive's member byte for byte: stock/src/<path> ∈ the ColabDesign archive (its
        repository root) — the vendored design step nobody edits."""
        archive = os.path.join(STOCK, COLABDESIGN_ARCHIVE)
        if not os.path.isfile(archive):                                                   # a bundle cut without the upstream archives: named, not failed
            self.skipTest(f"{COLABDESIGN_ARCHIVE}: upstream archive not included in this tree (online install route only)")
        with tarfile.open(archive) as tf:
            members = {m.name: m for m in tf.getmembers() if m.isfile()}
            self.assertEqual(sorted(n for n in members if n.count("/") == 1), [COLABDESIGN_PREFIX + "LICENSE.txt", COLABDESIGN_PREFIX + "README.md", COLABDESIGN_PREFIX + "setup.py"])
            for rel in colabdesign_src_files():
                member = COLABDESIGN_PREFIX + rel
                self.assertIn(member, members, rel)
                disk = sha(os.path.join(STOCK, "src", rel))
                self.assertEqual(hashlib.sha256(tf.extractfile(members[member]).read()).hexdigest(), disk, rel)

    def test_dssp_is_executable(self):
        dssp = os.path.join(STOCK, "src", "bindcraft", "functions", "dssp")
        if not os.path.isfile(dssp):                                                      # not carried in the tree: fetched at install (upstream.bindcraft.fetched)
            self.skipTest("functions/dssp is fetched by `run.sh install`, not carried; it is not in place in this checkout")
        self.assertTrue(os.access(dssp, os.X_OK), "functions/dssp must be executable (run.sh install sets mode 755, as install_bindcraft.sh chmod +x)")

    def test_the_archive_words(self):
        """stock/check_pins.py names every upstream archive either way: `included` when the file is there, `not included: …` for a kit directory
        without it, with how that upstream is installed there — reported, never a finding."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("colabdesign_check_pins_ti", os.path.join(STOCK, "check_pins.py"))
        cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
        pins = json.load(open(os.path.join(STOCK, "PINS.json"), encoding="utf-8"))
        here = cp.upstream_archives(pins)
        self.assertEqual({k: v["path"] for k, v in here.items()}, {"colabdesign": "stock/" + COLABDESIGN_ARCHIVE, "bindcraft": pins["upstream"]["bindcraft"]["archive"]})
        for name, d in here.items():
            self.assertEqual(d["included"], os.path.isfile(os.path.join(_stubs.TREE, d["path"])), name)
        with tempfile.TemporaryDirectory() as empty:
            gone = cp.upstream_archives(pins, home=empty)
        self.assertEqual({k: v["included"] for k, v in gone.items()}, {"colabdesign": False, "bindcraft": False})
        self.assertEqual(cp.upstream_archive_lines(gone), [
            f'upstream archive stock/{COLABDESIGN_ARCHIVE}: not included: online install route only, pip install --no-deps "{pins["upstream"]["colabdesign"]["install"]}"',
            f"upstream archive {pins['upstream']['bindcraft']['archive']}: not included: the vendored tree stock/src/bindcraft/ runs as shipped"])
        self.assertEqual(cp.upstream_archive_lines({k: dict(v, included=True) for k, v in gone.items()}), [f"upstream archive stock/{COLABDESIGN_ARCHIVE}: included", f"upstream archive {pins['upstream']['bindcraft']['archive']}: included"])
        bad, detail = cp.check(pins, stack=False, gpu=False)
        self.assertEqual(detail["upstream_archives"], here); self.assertFalse(any("archive" in b for b in bad))   # never a finding

    def test_the_vendored_set_matches_upstream_extracted(self):
        """upstream.colabdesign.extracted names exactly the files under stock/src/ (excluding bindcraft/): total accounting, mirroring the
        bindcraft side below."""
        extracted = set(_stubs.pins()["upstream"]["colabdesign"]["extracted"])
        on_disk = {"src/" + rel for rel in colabdesign_src_files()}
        self.assertEqual(len(_stubs.pins()["upstream"]["colabdesign"]["extracted"]), len(extracted), "upstream.colabdesign.extracted lists a path twice")
        self.assertEqual(sorted(on_disk - extracted), [], "vendored on disk but absent from upstream.colabdesign.extracted")
        self.assertEqual(sorted(extracted - on_disk), [], "listed in upstream.colabdesign.extracted but absent under stock/src/")



# ============================================================ BindCraft side: stock/src/bindcraft/ + the pinned settings
UPSTREAM = "bindcraft"                                                        # stock/PINS.json upstream.<name> of the vendored tree
VENDORED = os.path.join(_stubs.TREE, names.BINDCRAFT_DIR)                     # colabdesign/stock/src/bindcraft
SRC_PREFIX = os.path.relpath(VENDORED, STOCK).replace(os.sep, "/") + "/"      # "src/bindcraft/": the upstream.bindcraft.extracted path prefix
FETCHED = ("functions/dssp",)                                                 # the binary BindCraft's design step runs (optimise_beta's DSSP check): not carried, fetched +x by run.sh install (upstream.bindcraft.fetched)
NOT_DISTRIBUTED = ("functions/dssp", "functions/DAlphaBall.gcc")              # BindCraft's two prebuilt executables: in neither stock/src/ nor the carried archive (its recipe excludes them)

def vendored_files():
    """Every file under the vendored tree, relative to it, '/'-separated (bytecode caches excluded, as TestKitIntegrity walks opt/forward, below;
    the install-fetched FETCHED files and a leftover `.part` of an interrupted fetch excluded: they are not the tree's, fetched_files_present covers them)."""
    out = []
    for root, dirs, files in os.walk(VENDORED):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        out += [os.path.relpath(os.path.join(root, f), VENDORED).replace(os.sep, "/") for f in files]
    return sorted(r for r in out if r not in FETCHED and not (r.endswith(".part") and r[:-5] in FETCHED))


def fetched_files_present():
    """The FETCHED files in place in this checkout (after `run.sh install`: all of them; in a fresh checkout: none)."""
    return [rel for rel in FETCHED if os.path.isfile(os.path.join(VENDORED, rel))]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestBindcraftVendoredBytes(unittest.TestCase):
    def setUp(self):
        self.pins = _stubs.pins()
        self.up = self.pins["upstream"][UPSTREAM]

    # ------------------------------------------------------------------------------------------------ the file set: total accounting
    def test_the_vendored_set_matches_upstream_extracted(self):
        on_disk = {SRC_PREFIX + rel for rel in vendored_files()}
        extracted = set(self.up["extracted"])
        self.assertTrue(on_disk, f"no vendored file under {VENDORED}")
        self.assertEqual(len(self.up["extracted"]), len(extracted), "upstream.%s.extracted lists a path twice" % UPSTREAM)
        self.assertEqual(sorted(on_disk - extracted), [], f"vendored on disk but absent from upstream.{UPSTREAM}.extracted")
        self.assertEqual(sorted(extracted - on_disk), [], f"listed in upstream.{UPSTREAM}.extracted but absent under {names.BINDCRAFT_DIR}")
        fetched = self.up["fetched"]["files"]
        self.assertEqual(sorted(fetched), sorted(SRC_PREFIX + rel for rel in FETCHED), f"upstream.{UPSTREAM}.fetched.files is not the fetched set")
        self.assertEqual(sorted(extracted & set(fetched)), [], "a file is listed both as extracted (carried) and as fetched (not carried)")
        for rel in fetched_files_present():                                                  # in place (an installed tree): the pinned bytes, executable
            rec = fetched[SRC_PREFIX + rel]; p = os.path.join(VENDORED, rel)
            self.assertEqual((names.sha256_file(p), os.path.getsize(p)), (rec["sha256"], rec["bytes"]), f"{rel} in place is not the pinned file (run.sh install refuses it by name)")
            self.assertTrue(os.stat(p).st_mode & stat.S_IXUSR, f"{rel} is in place without its executable bit (run.sh install sets mode {rec['mode']})")

    def test_fetched_files_are_pinned_to_the_upstream_commit(self):
        """Each install-fetched file: its URL is the raw file of the pinned repository AT THE PINNED COMMIT (never a branch), with a full sha256, a
        byte count and mode 755 — what stock/check_pins.py --fetch checks before it places or keeps the file."""
        m = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", self.up["repo"]); self.assertIsNotNone(m, self.up["repo"])
        fetched = self.up["fetched"]["files"]; self.assertTrue(fetched)
        for key, rec in fetched.items():
            self.assertTrue(key.startswith(SRC_PREFIX), key); rel = key[len(SRC_PREFIX):]
            self.assertEqual(rec["url"], f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{self.up['commit']}/{rel}", key)
            self.assertRegex(rec["sha256"], r"^[0-9a-f]{64}$"); self.assertGreater(rec["bytes"], 0); self.assertEqual(rec["mode"], "755")
            self.assertIn(rel, NOT_DISTRIBUTED)
        self.assertIn("fetch", self.up["install"]); self.assertTrue(self.up["fetched"].get("why"))

    # ------------------------------------------------------------------------------------------------ the bytes: disk == archive member, live, not against a stored pin
    def test_every_vendored_file_is_its_archive_member_byte_for_byte(self):
        archive = os.path.join(_stubs.TREE, self.up["archive"])
        if not os.path.isfile(archive):                                                          # a bundle cut without the upstream archives: named, not failed
            self.skipTest(f"{os.path.basename(self.up['archive'])}: upstream archive not included in this tree (the vendored files run as shipped)")
        self.assertIn(self.up["commit"][:8], os.path.basename(archive), "the archive name does not carry the pinned commit")
        prefix = re.search(r"--prefix=(\S+)", self.up["archive_recipe"]).group(1)              # 'bindcraft-efb5bfeb/': the recipe's one root
        self.assertIn(self.up["commit"], self.up["archive_recipe"], "the recipe archives another commit than the pin")
        excluded = re.findall(r"':\(exclude\)([^']+)'", self.up["archive_recipe"])                 # the recipe's pathspec exclusions: BindCraft's two prebuilt executables
        self.assertEqual(sorted(excluded), sorted(NOT_DISTRIBUTED), "the archive recipe does not exclude exactly the two prebuilt executables")
        with tarfile.open(archive) as tf:
            members = {m.name: m for m in tf.getmembers()}
            stray = sorted(n for n in members if not (n + "/").startswith(prefix))
            self.assertEqual(stray, [], f"archive members outside the recipe's --prefix={prefix}")
            self.assertEqual(sorted(n for n in members if n[len(prefix):] in NOT_DISTRIBUTED), [], "the carried archive holds a prebuilt executable this tree does not distribute")
            self.assertEqual(tf.pax_headers.get("comment"), self.up["commit"], "the archive's pax comment is not the pinned commit (git archive writes it)")
            for rel in vendored_files():
                key, member = SRC_PREFIX + rel, prefix + rel
                self.assertIn(member, members, f"{key} is vendored but {member} is not a member of {self.up['archive']}")
                self.assertTrue(members[member].isfile(), f"{member} is not a regular file in the archive")
                disk = names.sha256_file(os.path.join(VENDORED, rel))
                self.assertEqual(_sha256_bytes(tf.extractfile(members[member]).read()), disk, f"{key}: the vendored bytes differ from the archive member (vendored code is never edited)")

    # ------------------------------------------------------------------------------------------------ the pinned settings
    def test_pins_settings_declare_the_loaders_file(self):
        rec = self.pins[settings.PIN_KEY]
        self.assertEqual(os.path.normpath(rec["file"]), os.path.normpath(settings.SETTINGS_RELPATH), "PINS.json settings.file is not the loader's file")
        key = os.path.relpath(os.path.join(_stubs.TREE, rec["file"]), STOCK).replace(os.sep, "/")
        self.assertTrue(key.startswith(SRC_PREFIX), f"the pinned settings {rec['file']!r} live outside the vendored tree")
        self.assertEqual(settings.pinned_relpath(_stubs.TREE), rec["file"])
        loaded = settings.load(_stubs.TREE)
        on_disk = names.sha256_file(os.path.join(_stubs.TREE, rec["file"]))
        self.assertEqual((loaded["sha256"], loaded["file"], loaded["name"]), (on_disk, settings.SETTINGS_RELPATH, settings.NAME))
        with open(os.path.join(_stubs.TREE, rec["file"]), "r", encoding="utf-8") as fh:
            self.assertEqual(loaded["advanced"], json.load(fh), "load() returns something other than the file's own dict")
        self.assertEqual(set(settings.iteration_counts(loaded["advanced"])), {"soft", "temp", "hard", "greedy"})

    def test_the_loader_refuses_by_name_before_reading(self):
        """A copy of the tree's PINS.json + settings file loads; an absent file and a PINS record pointing at another file are refused
        by name; a PINS record with no pin at all is refused by name too."""
        rec = self.pins[settings.PIN_KEY]
        tmp = tempfile.mkdtemp(prefix="cd_opt_settings_")
        try:
            os.makedirs(os.path.join(tmp, "stock"))
            pins_copy = os.path.join(tmp, "stock", "PINS.json")
            shutil.copy(_stubs.PINS, pins_copy)
            target = os.path.join(tmp, settings.SETTINGS_RELPATH)
            with self.assertRaises(settings.SettingsError) as cm:                              # absent: refused by name
                settings.load(tmp)
            self.assertIn("absent", str(cm.exception))
            os.makedirs(os.path.dirname(target))
            shutil.copy(os.path.join(_stubs.TREE, rec["file"]), target)
            loaded = settings.load(tmp)                                                        # the pristine copy loads
            self.assertEqual(loaded["sha256"], names.sha256_file(target))
            pins = json.load(open(pins_copy, encoding="utf-8"))                                # re-point the pin's file
            pins[settings.PIN_KEY]["file"] = rec["file"].replace(settings.NAME, settings.NAME + "_flexible")
            json.dump(pins, open(pins_copy, "w", encoding="utf-8"))
            with self.assertRaises(settings.SettingsError) as cm:
                settings.load(tmp)
            self.assertIn("expected", str(cm.exception))
            del pins[settings.PIN_KEY]                                                         # no pin at all: refused by name
            json.dump(pins, open(pins_copy, "w", encoding="utf-8"))
            with self.assertRaises(settings.SettingsError):
                settings.pinned_relpath(tmp)
            with self.assertRaises(settings.SettingsError):
                settings.load(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================ the kit carries no kernel or lever copy: its levers are adapters over the shared core
@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestKitIntegrity(unittest.TestCase):
    def test_no_forward_directory(self):
        """The kit carries nothing under opt/ but the package and its build files: no file under opt/forward/ (no add-on code, patches or licence texts of one)."""
        carried = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(_stubs.OPT_DIR, "forward")) for f in fs]
        self.assertEqual(carried, [])

    def test_no_kernel_or_lever_copy_is_carried(self):
        """The levers are this package's adapters over the shared core: no `af2_flash_pallas.py`, `colabdesign_pallas_attn.py` or
        `af2m_levers.py` exists anywhere under the kit, and the adapters import the core modules by name."""
        for r, _, fs in os.walk(_stubs.TREE):
            for f in fs:
                self.assertNotIn(f, ("af2_flash_pallas.py", "colabdesign_pallas_attn.py", "af2m_levers.py"), os.path.join(r, f))
        src_p = open(os.path.join(_stubs.PKG_DIR, "pallas.py")).read()
        src_n = open(os.path.join(_stubs.PKG_DIR, "nosub.py")).read()
        self.assertIn("from opt_core.kernels import pallas_attn_serve as F1", src_p)
        self.assertIn("from opt_core.jax_design import subbatch_policy", src_n)
        self.assertIn("from opt_core.attn.size_gate import SizeGate", src_n)


if __name__ == "__main__":
    unittest.main()
