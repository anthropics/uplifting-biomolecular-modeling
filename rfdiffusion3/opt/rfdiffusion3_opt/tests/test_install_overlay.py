"""install.sh --classify / --rebind / apply / --uninstall on fake site-packages trees: the OVERLAY_PRESTATE classes (pristine | current |
previous:<id> | foreign:<files>), the OVERLAY_APPLIED line, the overlay marker, and the rule that a foreign file is never touched by --rebind
(apply --force stays the manual escape)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from .. import install, registry
from . import _fixtures as fx

MARKER_REL = os.path.join("rfd3", ".xattempt_addon_overlay.json")
ID_RE = r"[0-9][^/ ]*/[0-9a-f]{12}"                       # <MANIFEST version>/<overlay id, 12 hex>


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class OverlayVerbs(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.td = self._td.name
        self.kit = registry.kit_home()

    def tearDown(self):
        self._td.cleanup()

    def sh(self, site, *args, kit=None):
        """install.sh <args> against `site` (a `python` wrapper first on PATH, as the installer locates site-packages)."""
        kit = kit or self.kit
        bin_dir = fx.fake_python_on_path(os.path.join(self.td, "bin"), site)
        env = dict(os.environ, PATH=bin_dir + os.pathsep + os.environ.get("PATH", ""), PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run(["bash", os.path.join(kit, "install.sh"), *args], env=env, capture_output=True, text=True, cwd=kit)
        return r.returncode, [l for l in r.stdout.splitlines() if l.strip()], r.stdout + r.stderr

    def classify(self, site, kit=None):
        rc, lines, both = self.sh(site, "--classify", kit=kit)
        self.assertEqual(rc, 0, both)
        self.assertEqual(len(lines), 1, both)
        self.assertTrue(lines[0].startswith("OVERLAY_PRESTATE: "), both)
        return lines[0][len("OVERLAY_PRESTATE: "):]

    def newer_kit(self):
        """A copy of the kit whose patched/hoist.py carries one more tagged line: the tree of a LATER version relative to what is installed."""
        dst = os.path.join(self.td, "kit_next", "xattempt_addon")
        shutil.copytree(self.kit, dst)
        with open(os.path.join(dst, "patched", "rfd3", "model", "hoist.py"), "a") as fh:
            fh.write("\n# [RFD3_HOIST] a later revision of this module\n")
        return dst

    def test_pristine_rebind_installs_and_marks_then_current_is_a_no_op(self):
        site = fx.make_site(self.td, "pristine")
        self.assertEqual(self.classify(site), "pristine")
        rc, lines, both = self.sh(site, "--rebind")
        self.assertEqual(rc, 0, both)
        self.assertEqual(lines[0], "OVERLAY_PRESTATE: pristine", both)
        self.assertRegex(lines[1], r"^OVERLAY_APPLIED: 10 files id=" + ID_RE + r"$", both)
        self.assertEqual(len(lines), 2, both)
        marker = json.load(open(os.path.join(site, MARKER_REL)))
        self.assertEqual(set(marker), {"name", "version", "overlay_id", "files"})
        self.assertTrue(marker["name"].startswith("RFD3_XATTEMPT_ADDON"), marker["name"])
        self.assertEqual(lines[1].split("id=")[1], f"{marker['version']}/{marker['overlay_id'][:12]}")
        self.assertEqual(marker["files"], {rel: _sha(os.path.join(self.kit, "patched", rel)) for rel in registry.target_files()})
        for rel in registry.target_files():                                   # the sidecars: pristine backups for upstream's files, absent-marks for the two new ones
            side = ".hoist_absent" if rel in registry.NEW_FILES else ".hoist_orig"
            self.assertTrue(os.path.isfile(os.path.join(site, rel + side)), rel + side)
        self.assertEqual(self.classify(site), "current")
        rc, lines, both = self.sh(site, "--rebind")
        self.assertEqual((rc, lines), (0, ["OVERLAY_PRESTATE: current"]), both)
        self.assertEqual(self.sh(site, "--check-only")[0], 0)

    def test_an_earlier_marked_install_is_previous_with_its_id_and_rebinds(self):
        site = fx.make_site(self.td, "pristine")
        self.assertEqual(self.sh(site, "--rebind")[0], 0)
        old = json.load(open(os.path.join(site, MARKER_REL)))
        nxt = self.newer_kit()
        self.assertEqual(self.classify(site, kit=nxt), f"previous:{old['version']}/{old['overlay_id'][:12]}")
        rc, lines, both = self.sh(site, "apply", kit=nxt)                       # plain apply refuses an earlier install by name (rc 4) and points at --rebind
        self.assertEqual(rc, 4, both)
        self.assertIn("--rebind", both)
        rc, lines, both = self.sh(site, "--rebind", kit=nxt)
        self.assertEqual(rc, 0, both)
        self.assertRegex(lines[0], r"^OVERLAY_PRESTATE: previous:" + ID_RE + r"$", both)
        self.assertRegex(lines[1], r"^OVERLAY_APPLIED: 10 files id=" + ID_RE + r"$", both)
        new = json.load(open(os.path.join(site, MARKER_REL)))
        self.assertNotEqual(new["overlay_id"], old["overlay_id"])
        self.assertEqual(_sha(os.path.join(site, "rfd3/model/hoist.py")), _sha(os.path.join(nxt, "patched", "rfd3/model/hoist.py")))
        sb = fx.stock_bytes()                                                  # the pristine backups survive a re-base: uninstall still leads back to upstream
        for rel in registry.target_files():
            if rel not in registry.NEW_FILES:
                self.assertEqual(_sha(os.path.join(site, rel + ".hoist_orig")), hashlib.sha256(sb[rel]).hexdigest(), rel)
        self.assertEqual(self.classify(site, kit=nxt), "current")
        self.assertRegex(self.classify(site), r"^previous:" + ID_RE + r"$")   # and this tree now sees the newer install as a previous version of itself

    def test_an_earlier_install_without_a_marker_is_previous_unmarked_by_its_sidecars_and_tags(self):
        site = fx.make_site(self.td, "pristine")
        self.assertEqual(self.sh(site, "--rebind")[0], 0)
        os.remove(os.path.join(site, MARKER_REL))                               # an install older than the marker
        nxt = self.newer_kit()
        self.assertEqual(self.classify(site, kit=nxt), "previous:unmarked")
        rc, lines, both = self.sh(site, "--rebind", kit=nxt)
        self.assertEqual(rc, 0, both)
        self.assertEqual(lines[0], "OVERLAY_PRESTATE: previous:unmarked", both)
        self.assertRegex(lines[1], r"^OVERLAY_APPLIED: 10 files id=" + ID_RE + r"$", both)
        self.assertEqual(self.classify(site, kit=nxt), "current")

    def test_a_foreign_file_is_named_and_rebind_touches_nothing(self):
        site = fx.make_site(self.td, "pristine")
        victim = os.path.join(site, "rfd3/model/layers/attention.py")
        open(victim, "w").write("# some other edit of this file\n")
        before = {rel: (_sha(os.path.join(site, rel)) if os.path.exists(os.path.join(site, rel)) else None) for rel in registry.target_files()}
        self.assertEqual(self.classify(site), "foreign:rfd3/model/layers/attention.py")
        rc, lines, both = self.sh(site, "--rebind")
        self.assertEqual(rc, 4, both)
        self.assertEqual(lines[0], "OVERLAY_PRESTATE: foreign:rfd3/model/layers/attention.py", both)
        self.assertFalse(any(l.startswith("OVERLAY_APPLIED") for l in lines), both)
        after = {rel: (_sha(os.path.join(site, rel)) if os.path.exists(os.path.join(site, rel)) else None) for rel in registry.target_files()}
        self.assertEqual(after, before)
        self.assertFalse(os.path.exists(os.path.join(site, MARKER_REL)))
        self.assertFalse(any(os.path.exists(os.path.join(site, rel + s)) for rel in registry.target_files() for s in (".hoist_orig", ".hoist_absent")))
        self.assertEqual(self.sh(site, "apply")[0], 4)                          # plain apply refuses too …
        rc, lines, both = self.sh(site, "apply", "--force")                     # … --force is the manual escape: backs the foreign file up and installs over it
        self.assertEqual(rc, 0, both)
        self.assertEqual(open(victim + ".hoist_orig").read(), "# some other edit of this file\n")
        self.assertEqual(self.classify(site), "current")

    def test_a_tagged_file_without_this_installers_sidecar_is_foreign(self):
        """The tags alone are not identification: an upstream file carrying `[RFD3_` text but no pristine .hoist_orig beside it is foreign."""
        site = fx.make_site(self.td, "pristine")
        victim = os.path.join(site, "rfd3/model/layers/blocks.py")
        with open(victim, "a") as fh:
            fh.write("\n# [RFD3_HOIST] hand-edited\n")
        self.assertEqual(self.classify(site), "foreign:rfd3/model/layers/blocks.py")
        shutil.copyfile(victim, victim + ".hoist_orig")                         # a backup that is not upstream's bytes does not identify it either
        self.assertEqual(self.classify(site), "foreign:rfd3/model/layers/blocks.py")

    def test_a_hand_edit_of_a_marked_install_is_foreign_even_with_its_sidecar_and_tags(self):
        """Once an apply wrote the marker, a target file is identified by the marker's record only: an edited overlay file (pristine backup
        beside it, tags inside it) is foreign — --rebind leaves it alone; the sidecar rule speaks only for installs older than the marker."""
        site = fx.make_site(self.td, "pristine")
        self.assertEqual(self.sh(site, "--rebind")[0], 0)
        with open(os.path.join(site, "rfd3/model/hoist.py"), "a") as fh:
            fh.write("\n# [RFD3_HOIST] edited in place\n")
        self.assertEqual(self.classify(site), "foreign:rfd3/model/hoist.py")
        self.assertEqual(self.sh(site, "--rebind")[0], 4)
        os.remove(os.path.join(site, MARKER_REL))                               # the same bytes on a marker-less install read as an earlier version (the sidecar rule)
        self.assertEqual(self.classify(site), "previous:unmarked")

    def test_a_mix_of_patched_and_pristine_files_is_previous_partial_and_rebinds(self):
        site = fx.make_site(self.td, "mixed")                                    # blocks.py pristine, the rest == patched/
        self.assertEqual(self.classify(site), "previous:partial")
        rc, lines, both = self.sh(site, "--rebind")
        self.assertEqual(rc, 0, both)
        self.assertEqual(lines[0], "OVERLAY_PRESTATE: previous:partial", both)
        self.assertRegex(lines[1], r"^OVERLAY_APPLIED: 10 files id=" + ID_RE + r"$", both)
        self.assertEqual(self.classify(site), "current")

    def test_uninstall_restores_upstream_and_drops_the_marker(self):
        site = fx.make_site(self.td, "pristine")
        self.assertEqual(self.sh(site, "--rebind")[0], 0)
        rc, lines, both = self.sh(site, "--uninstall")
        self.assertEqual(rc, 0, both)
        self.assertFalse(os.path.exists(os.path.join(site, MARKER_REL)))
        self.assertEqual(self.classify(site), "pristine")

    def test_the_overlay_id_is_a_digest_of_the_patched_bytes_not_a_stored_value(self):
        """No table in the tree carries the id: it is recomputed from patched/ (a tree with one more byte has another id)."""
        site = fx.make_site(self.td, "pristine")
        self.sh(site, "--rebind")
        this_id = json.load(open(os.path.join(site, MARKER_REL)))["overlay_id"]
        for root, _, files in os.walk(self.kit):
            for f in files:
                if not root.startswith(os.path.join(self.kit, "patched")):
                    self.assertNotIn(this_id[:12], open(os.path.join(root, f), errors="replace").read(), os.path.join(root, f))
        site2 = fx.make_site(os.path.join(self.td, "two"), "pristine")
        self.sh(site2, "--rebind", kit=self.newer_kit())
        self.assertNotEqual(json.load(open(os.path.join(site2, MARKER_REL)))["overlay_id"], this_id)

    def test_the_package_installer_knows_the_two_verbs(self):
        self.assertEqual(install.FLAGS["classify"], "--classify")
        self.assertEqual(install.FLAGS["rebind"], "--rebind")


if __name__ == "__main__":
    unittest.main()
