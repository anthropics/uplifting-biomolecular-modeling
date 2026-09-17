"""The kit tree and the ten-file classification: the carried subset equals MANIFEST.json's own file listing (no in-repo file is
re-hashed against a stored digest — the git commit identifies the carried bytes), the evidence files are absent, and
registry.classify names states exactly as the kit's install.sh --status does. The stock archive (the upstream snapshot the kit's
installer classifies against): its members are the recipe's paths, the kit's new files are genuinely absent from it, and the
image-of-record facts (the freeze, the apex wheel's build and archive metadata) agree with stock/PINS.json; stock/check_pins.py's
distribution rule (a git install at the pinned commit, the archive by sha256, or a local checkout — else refused)."""
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest

from .. import install, registry, stack
from . import _fixtures as fx


class TestKitBytes(unittest.TestCase):
    def test_carried_subset_matches_manifest(self):
        res = registry.check_tree()
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["checked"], 19)                       # MANIFEST.json `files`: what the tree carries (the kit's files; no design inputs)
        self.assertEqual([f for f in registry.manifest()["files"] if f.startswith("inputs/")], [])   # design inputs are not kit bytes
        self.assertEqual(res["version"], "1.8")
        self.assertEqual(res["git_check"], "clean")                # this checkout's carried files match HEAD

    def test_a_tampered_carried_file_is_caught_by_git_not_a_stored_digest(self):
        """A present-but-edited carried file: no stored digest to check it against, so the check is `git diff` against
        HEAD on a real commit (a fresh `git init` + commit of a copy of the kit tree, not the checkout under test —
        isolated from whatever else this working tree happens to hold open). Edited -> ok=False, named in `modified`;
        a copy with no `.git` at all -> the same file goes undetected, but that is a named skip, never a silent one."""
        kit = registry.kit_home()
        with tempfile.TemporaryDirectory() as td:
            g = os.path.join(td, "git_copy")
            shutil.copytree(kit, g)
            run = lambda *a: subprocess.run(a, cwd=g, capture_output=True, text=True, check=True)
            run("git", "init", "-q")
            run("git", "config", "user.email", "t@example.com")
            run("git", "config", "user.name", "t")
            run("git", "add", "-A")
            run("git", "commit", "-q", "-m", "snapshot")
            clean = registry.check_tree(g)
            self.assertEqual((clean["ok"], clean["git_check"], clean["modified"]), (True, "clean", []))
            edited_rel = "install.sh"
            with open(os.path.join(g, edited_rel), "a") as fh:
                fh.write("\n# edited\n")
            tampered = registry.check_tree(g)
            self.assertEqual((tampered["ok"], tampered["git_check"], tampered["modified"]), (False, "modified", [edited_rel]))

            plain = os.path.join(td, "no_git_copy")
            shutil.copytree(kit, plain)
            with open(os.path.join(plain, edited_rel), "a") as fh:
                fh.write("\n# edited, but no .git here\n")
            no_git = registry.check_tree(plain)
            self.assertTrue(no_git["ok"], no_git)                              # nothing to compare against: not a false failure
            self.assertTrue(no_git["git_check"].startswith("skipped:"), no_git)  # but the skip is named, not silent
            self.assertEqual(no_git["modified"], [])

            empty_path = os.path.join(td, "empty_path")             # a `git` binary that plainly doesn't exist, not just no repo
            os.makedirs(empty_path)
            old_path = os.environ.get("PATH")
            os.environ["PATH"] = empty_path
            try:
                no_binary = registry.check_tree(g)             # g: the real git checkout from above, now unreachable without the binary
            finally:
                if old_path is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = old_path
            self.assertTrue(no_binary["ok"], no_binary)                                # activation must not crash when git is absent
            self.assertTrue(no_binary["git_check"].startswith("skipped:"), no_binary)   # named skip, not a raised exception

    def test_manifest_carries_no_record(self):
        """MANIFEST.json states the kit (name, version, what it is, class, upstream, flags, the compile gate's frame counts) and lists its files; it
        carries no measurement, result or provenance record."""
        man = registry.manifest()
        self.assertEqual(man["version"], "1.8")
        self.assertEqual(set(man), {"name", "version", "what", "class", "upstream", "flags", "compile_frames_skipped", "files"})
        for rel in ("results", "tools", "tests", "patches"):
            self.assertFalse(os.path.exists(os.path.join(registry.kit_home(), rel)), rel)

    def test_every_file_in_the_kit_directory_is_a_carried_manifest_member(self):
        """The kit directory holds exactly the 25 carried members plus MANIFEST.json itself — nothing else."""
        kit = registry.kit_home()
        on_disk = sorted(os.path.relpath(os.path.join(r, f), kit) for r, _, fs in os.walk(kit) for f in fs if "__pycache__" not in r)
        want = registry.carried_files()
        self.assertEqual(on_disk, sorted(want + ["MANIFEST.json"]))

    def test_stock_table_and_pins_do_not_duplicate_the_rfd3_hashes(self):
        """The kit's own upstream-at-pin table (registry.TABLES["stock"]) is the one place the eight rfd3/* stock hashes are recorded;
        PINS.json "stock_files_sha256" carries only foundry/utils/torch.py (no tracked home inside the kit), not a copy of the kit's table."""
        kit_stock = registry.read_table(registry.TABLES["stock"])
        pins = stack.pins()["stock_files_sha256"]
        self.assertEqual(set(pins), {"foundry/utils/torch.py"})
        self.assertEqual(len(kit_stock), 8)
        self.assertTrue(all(k.startswith("rfd3/") for k in kit_stock))
        self.assertFalse(set(kit_stock) & set(pins))

    def test_target_files(self):
        t = registry.target_files()
        self.assertEqual(len(t), 10)
        self.assertEqual(set(registry.NEW_FILES), {p for p in t if p not in registry.read_table(registry.TABLES["stock"])})

    def test_a_stray_or_missing_kit_file_is_detected(self):
        """check_tree checks tree shape (every carried path present, nothing stray beside them) — not byte content: the git commit is
        what identifies the carried files, not a stored digest, so a file's content is never re-hashed against one."""
        with tempfile.TemporaryDirectory() as td:
            k2 = os.path.join(td, "kit")
            shutil.copytree(registry.kit_home(), k2)
            open(os.path.join(k2, "results.txt"), "w").write("x")
            res = registry.check_tree(k2)
            self.assertFalse(res["ok"])
            self.assertIn("results.txt", res["stray"])
            os.remove(os.path.join(k2, "results.txt"))
            os.remove(os.path.join(k2, "install.sh"))
            res = registry.check_tree(k2)
            self.assertFalse(res["ok"])
            self.assertIn("install.sh", res["missing"])


class TestClassify(unittest.TestCase):
    def _status_via_install_sh(self, site):
        with tempfile.TemporaryDirectory() as td:
            bin_dir = fx.fake_python_on_path(os.path.join(td, "bin"), site)
            env = dict(os.environ)
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
            r = subprocess.run(["bash", os.path.join(registry.kit_home(), "install.sh"), "--status"], env=env, capture_output=True, text=True, cwd=registry.kit_home())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        st = install.parse_status(r.stdout)
        self.assertEqual(os.path.realpath(st.pop("site-packages")), os.path.realpath(site))
        return st

    def test_states_agree_with_install_sh(self):
        for state, label in (("patched", "patched"), ("pristine", "pristine"), ("mixed", "mixed"), ("unknown", "unknown")):
            with tempfile.TemporaryDirectory() as td:
                site = fx.make_site(td, state)
                mine = registry.classify(site)
                theirs = self._status_via_install_sh(site)
                self.assertEqual(mine, theirs, state)
                self.assertEqual(registry.summarize(mine)[0], label, (state, mine))

    def test_not_installed(self):
        self.assertEqual(registry.classify("/nonexistent/site"), {p: ("absent-ok" if p in registry.NEW_FILES else "unknown") for p in registry.target_files()})
        self.assertEqual(registry.summarize({})[0], "not-installed")


class TestInstallCommandLine(unittest.TestCase):
    def test_install_names_a_missing_upstream_instead_of_its_traceback(self):
        """`python -m rfdiffusion3_opt.install status` in an interpreter without `rfd3`: one `[rfdiffusion3-opt] install status: rfd3 is not
        importable in <python> …` line and the installer's own 'no rfd3' code 2 — not the probe child's raw ModuleNotFoundError traceback."""
        import io
        import sys
        from contextlib import redirect_stderr, redirect_stdout
        from .. import _autoload
        probe = subprocess.run([sys.executable, "-c", "import rfd3"], capture_output=True)
        self.assertNotEqual(probe.returncode, 0, "precondition: this test interpreter has no rfd3")
        err, out = io.StringIO(), io.StringIO()
        with redirect_stderr(err), redirect_stdout(out):
            rc = install.main(["status"])
        self.assertEqual(rc, install.EXIT_NO_RFD3)
        self.assertEqual(rc, 2)
        lines = [l for l in err.getvalue().splitlines() if l]                 # the package's printer starts every line fresh (a leading newline)
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("[%s] install status: rfd3 is not importable in %s (ModuleNotFoundError: No module named 'rfd3')" % (_autoload.TAG, sys.executable)), lines[0])
        self.assertNotIn("Traceback", err.getvalue())
        self.assertEqual(out.getvalue(), "")


class TestStockArchive(unittest.TestCase):
    def setUp(self):
        self.stock = os.path.join(registry.tree_home(), "stock")
        self.pins = stack.pins()
        self.sums = {"foundry-4010e3e.tar.gz"}   # the apex wheel is built by the recipe (PINS pinned_stack.apex.build) and placed in stock/wheels/ for a container build, never a committed file of the tree
        self.lock = os.path.join(registry.tree_home(), self.pins["pinned_stack"]["in_job_pins"])   # environment/requirements.lock: the one list of the stack

    def test_stock_dir_contents_and_pinned_stack_facts(self):
        for name in self.sums:
            self.assertTrue(os.path.isfile(os.path.join(self.stock, name)), name)
        rec = self.pins["pinned_stack"]
        for key in ("tested_on", "recipe", "python", "torch", "cuda", "triton", "gpu", "apex", "in_job_pins"):
            self.assertIn(key, rec)
        for key in ("image", "base_image", "image_recipe", "image_label", "image_layer", "freeze"):          # the stack is stated by versions + recipe; no image id is a key of its own (tested_on is the one documentation line)
            self.assertNotIn(key, rec)
        self.assertNotIn("wheel_archive", rec["apex"]); self.assertNotIn("kit_owner_copy", self.pins["checkpoint"])   # no volume / mount location in the pins: where files live is the box's (README "Variables")
        self.assertTrue(self.pins["checkpoint"]["url"].startswith("https://"))                               # the pinned checkpoint has a public route
        resolved = [l.strip() for l in open(self.lock, encoding="utf-8") if l.strip() and not l.startswith("#")]
        self.assertIn(f"torch=={rec['torch'].split('+')[0]}", resolved); self.assertIn(f"triton=={rec['triton']}", resolved)   # the in-job pins carry the pinned stack's versions ...
        self.assertIn(f"apex=={rec['apex']['version']}", resolved)                                                             # ... and the apex line both interpreters carry
        self.assertEqual((rec["apex"]["wheel"], rec["apex"]["wheel_sha256"]), ("apex-0.1-cp312-cp312-linux_x86_64.whl", "2da21fe056fd4d14542a228b309ed9e19f0500cd400778ea2b502d3c17d05080"))
        self.assertIn("APEX_CPP_EXT=1 APEX_CUDA_EXT=1", rec["apex"]["build"])                                 # the build recipe travels with the pin
        self.assertNotIn(rec["apex"]["wheel"], self.sums)
        self.assertFalse(os.path.exists(os.path.join(self.stock, rec["apex"]["wheel"])))
        for gone in ("pip_freeze.txt", "IMAGE_LAYER.json", "APEX_BUILD.json"):                                    # no image freeze or build record in the tree
            self.assertFalse(os.path.exists(os.path.join(self.stock, gone)), gone)

    def test_recipe_paths_and_prefix(self):
        recipe = self.pins["archive_recipe"]
        paths = re.search(r"\(foundry: (.*?)\)", recipe).group(1).split()
        prefix = re.search(r"--prefix=(\S+)", recipe).group(1)
        self.assertEqual(prefix, fx.ARCHIVE_PREFIX)
        with tarfile.open(fx.ARCHIVE, "r:gz") as tf:
            names = tf.getnames()
        self.assertTrue(all(n.startswith(prefix.rstrip("/")) for n in names))
        rel = {n[len(prefix):] for n in names if n.startswith(prefix)}
        for p in paths:
            self.assertTrue(p in rel or any(r.startswith(p + "/") for r in rel), p)
        for must in ("pyproject.toml", "README.md", "LICENSE.md", "models/rfd3/configs/inference_engine/rfdiffusion3.yaml", "models/rfd3/docs/examples/demo.json"):
            self.assertIn(must, rel)
        self.assertFalse(any(r.startswith(".git") for r in rel))

    def test_kit_new_files_are_not_in_stock(self):
        """The archived upstream snapshot carries the one file PINS still tracks a stock hash for (foundry/utils/torch.py — no other
        tracked home); the two files the kit adds (hoist.py, cudagraph_sampler.py) are genuinely new, absent from stock."""
        sb = fx.stock_bytes()
        for rel in self.pins["stock_files_sha256"]:
            self.assertIn(rel, sb)
        for rel in self.pins["kit_new_files_absent_in_stock"]:
            self.assertNotIn(rel, sb)

    def test_version_string_and_commit(self):
        f = self.pins["upstream"]["foundry"]
        self.assertEqual(f["version"], "0.2.1.dev13+g4010e3e2e")
        self.assertTrue(f["commit"].startswith("4010e3e2e"))
        self.assertIn("SETUPTOOLS_SCM_PRETEND_VERSION=0.2.1.dev13+g4010e3e2e", self.pins["archive_recipe"])


class TestCheckPins(unittest.TestCase):
    """stock/check_pins.py's distribution rule: the pinned version string and a source that names the commit — a git install at the
    commit, the archive by sha256, or a local checkout / extracted archive (the version string carries the commit hash) — else refused."""

    def _state(self, **kw):
        import tempfile as _tempfile
        cp = stack._check_pins_module()
        with _tempfile.TemporaryDirectory() as td:
            site = os.path.join(td, "site")
            os.makedirs(site)
            fx.dist_info(site, **kw)
            import sys
            old = list(sys.path)
            sys.path.insert(0, site)
            try:
                return cp.distribution_state(stack.pins()["upstream"]["foundry"])
            finally:
                sys.path[:] = old

    def test_sources(self):
        self.assertTrue(self._state()["pinned"])                                                       # git at the pin
        self.assertFalse(self._state(commit="0" * 40)["pinned"])                                       # git at another commit
        self.assertFalse(self._state(version="0.2.0")["pinned"])                                       # another version at the pin commit
        d = self._state(kind="dir")
        self.assertTrue(d["pinned"]); self.assertEqual(d["source"], "a local directory")               # the kit's own install route
        self.assertFalse(self._state(kind="dir", version="0.2.1.dev14+gfaf42990")["pinned"])          # a checkout at HEAD
        n = self._state(kind="none")
        self.assertFalse(n["pinned"]); self.assertEqual(n["source"], "a non-git, non-archive source")  # an index install


if __name__ == "__main__":
    unittest.main()
