"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (stub `python`, `tar` and `patch` on PATH record every invocation in a
scratch copy of the tree; nothing is installed, nothing is written into this tree), the patched-tree recipe against the pin with the real tools
(unpack stock/*.tar.gz, apply opt/forward/af2ig_kit/patches, stock/check_pins.py --checkout), the weights step (af2ig_opt.weights.fetch with an
injected archive stream and pin table; no network), and environment/ against stock/PINS.json. CPU only."""
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/, environment/
RUN_SH = os.path.join(TREE, "run.sh")
PATCHES = os.path.join(TREE, "opt", "forward", "af2ig_kit", "patches")

STUB_PYTHON = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes per call kind by STUB_RC_PROBE / _PIP / _CHECKOUT / _PINS / _WEIGHTS
printf 'python %s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py --checkout"*) exit "${STUB_RC_CHECKOUT:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m af2ig_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""
STUB_TAR = """#!/bin/bash
# stub tar: records the call and lays down the one file run.sh looks for (-C <dir> is the tree)
printf 'tar %s\\n' "$*" >> "$STUB_LOG"
while [ $# -gt 0 ]; do [ "$1" = -C ] && { mkdir -p "$2/dl_binder_design/af2_initial_guess" && : > "$2/dl_binder_design/af2_initial_guess/predict_pdb.py"; }; shift; done
exit "${STUB_RC_TAR:-0}"
"""
STUB_PATCH = """#!/bin/bash
printf 'patch %s < %s\\n' "$*" "$(basename "$(readlink -f /dev/stdin)")" >> "$STUB_LOG"
exit "${STUB_RC_PATCH:-0}"
"""


def _write_exe(path, text):
    with open(path, "w") as f: f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class InstallVerbArguments(unittest.TestCase):
    """run.sh copied into a scratch tree (stock/ and opt/ linked from this one) with stub tools first on PATH: the calls, their order, the exit codes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="af2ig_install_verb_")
        self.tree = os.path.join(self.tmp, "af2ig"); os.makedirs(self.tree)
        shutil.copy(RUN_SH, os.path.join(self.tree, "run.sh"))
        for d in ("stock", "opt", "configs"): os.symlink(os.path.join(TREE, d), os.path.join(self.tree, d))
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        _write_exe(os.path.join(self.bin, "python"), STUB_PYTHON); _write_exe(os.path.join(self.bin, "tar"), STUB_TAR); _write_exe(os.path.join(self.bin, "patch"), STUB_PATCH)
        self.log = os.path.join(self.tmp, "calls.log")
        self.core, self.opt, self.checkout = f"{self.tree}/../common/opt_core", f"{self.tree}/opt", f"{self.tree}/dl_binder_design"
        self.archive = os.path.join(self.tree, "stock", "dl_binder_design-cafa3853.tar.gz")
        self.patches = sorted(f for f in os.listdir(PATCHES) if re.match(r"[0-9][0-9]_.*\.diff$", f))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_sh(self, args, **rc):
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        r = subprocess.run(["bash", os.path.join(self.tree, "run.sh")] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        calls = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        return r.returncode, calls, r.stdout + r.stderr

    def fresh_sequence(self, weights=None):
        seq = [f"python -m pip install -e {self.core} -e {self.opt}", f"tar -xzf {self.archive} -C {self.tree}"]
        seq += [f"patch -d {self.checkout} -p1 -s < {p}" for p in self.patches]
        seq += [f"python -I {self.tree}/stock/check_pins.py --checkout {self.checkout}", f"python -I {self.tree}/stock/check_pins.py --gpu"]
        return seq + ([f"python -m af2ig_opt.weights {weights}"] if weights else [])

    def test_install_is_pip_editable_then_the_patched_tree_then_the_pin_check(self):
        rc, calls, out = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertTrue(len(self.patches) >= 8 and self.patches[0].startswith("00_"), self.patches)
        self.assertEqual(calls[0].split(" -c ")[0], "python -I"); self.assertIn("import os,sys", calls[0])   # the installed-from-this-tree probe (fails here: nothing is installed) ...
        self.assertEqual(calls[1:], self.fresh_sequence())                                              # ... so pip runs, then the tree is unpacked, patched in series order, and checked, then the pins
        self.assertIn(f"export AF2IG_DIR={self.checkout}/af2_initial_guess", out)

    def test_install_weights_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/w/params"], ["install", "--weights=/w/params"]):
            if os.path.exists(self.log): os.remove(self.log)
            shutil.rmtree(self.checkout, ignore_errors=True)
            rc, calls, out = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[1:], self.fresh_sequence(weights="/w/params"))
            self.assertIn("AF2_PARAMS=/w/params", out)

    def test_usage_errors_exit_2_and_call_nothing(self):
        for args in (["install", "--bogus"], ["install", "--weights"], ["install", "--weights", "--x"], ["install", "--config", "h100"], ["install", "--mode", "fast"], ["install", "extra"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, calls, out = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out)); self.assertEqual(calls, [], args); self.assertIn("run.sh:", out)

    def test_failures_stop_the_sequence_with_the_documented_codes(self):
        rc, calls, out = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual((rc, len(calls)), (1, 2), out); self.assertIn("the install failed", out)
        shutil.rmtree(self.checkout, ignore_errors=True); os.remove(self.log)
        rc, calls, out = self.run_sh(["install"], patch=2)
        self.assertEqual((rc, len(calls)), (1, 4), out); self.assertIn("did not apply", out)          # probe, pip, tar, the first patch
        shutil.rmtree(self.checkout, ignore_errors=True); os.remove(self.log)
        rc, calls, out = self.run_sh(["install"], checkout=3)
        self.assertEqual(rc, 1, out); self.assertTrue(calls[-1].endswith(f"--checkout {self.checkout}"), calls); self.assertIn("not the pinned, patched upstream tree", out)
        shutil.rmtree(self.checkout, ignore_errors=True); os.remove(self.log)
        rc, calls, out = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual(rc, 3, out); self.assertTrue(calls[-1].endswith("check_pins.py --gpu"), calls); self.assertIn("refused by the pin check", out)   # pins unmet: no weights step
        shutil.rmtree(self.checkout, ignore_errors=True); os.remove(self.log)
        rc, calls, out = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, calls[1:]), (1, self.fresh_sequence(weights="/w")), out)                 # the weights step's own exit code is the verb's

    def test_a_tree_already_installed_skips_pip_and_keeps_its_patched_tree(self):
        """The container image case: both packages resolve from this tree (probe exits 0) and the patched tree exists — pip, tar and patch are not
        run (a read-only image cannot), the tree and the pins are checked and --weights proceeds."""
        os.makedirs(os.path.join(self.checkout, "af2_initial_guess")); open(os.path.join(self.checkout, "af2_initial_guess", "predict_pdb.py"), "w").close()
        rc, calls, out = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls[1:], [f"python -I {self.tree}/stock/check_pins.py --checkout {self.checkout}", f"python -I {self.tree}/stock/check_pins.py --gpu", "python -m af2ig_opt.weights /w"])
        self.assertIn("installed from this tree already", out); self.assertIn("already — kept", out)

    def test_a_directory_in_the_way_is_named_not_overwritten(self):
        os.makedirs(self.checkout)                                                                       # exists, but holds no patched driver
        rc, calls, out = self.run_sh(["install"])
        self.assertEqual(rc, 1, out); self.assertNotIn("tar", " ".join(c.split()[0] for c in calls)); self.assertIn("remove it and re-run", out)


@unittest.skipUnless(shutil.which("tar") and shutil.which("patch"), "tar and GNU patch on PATH: the patched-tree recipe runs the real tools")
class PatchedTreeRecipe(unittest.TestCase):
    """The three install lines with the real tools in a scratch directory: the archive in stock/ unpacked, the kit's series applied in glob order, and
    stock/check_pins.py --checkout accepting the result (every pinned file present, every patched file byte-equal to the pin)."""

    def test_unpack_patch_check(self):
        tmp = tempfile.mkdtemp(prefix="af2ig_tree_")
        try:
            archives = [f for f in os.listdir(os.path.join(TREE, "stock")) if f.startswith("dl_binder_design-") and f.endswith(".tar.gz")]
            self.assertEqual(len(archives), 1, archives)
            subprocess.run(["tar", "-xzf", os.path.join(TREE, "stock", archives[0]), "-C", tmp], check=True)
            checkout = os.path.join(tmp, "dl_binder_design")
            series = sorted(f for f in os.listdir(PATCHES) if re.match(r"[0-9][0-9]_.*\.diff$", f))
            for p in series:
                with open(os.path.join(PATCHES, p), "rb") as fh:
                    r = subprocess.run(["patch", "-d", checkout, "-p1", "-s"], stdin=fh, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, (p, r.stdout, r.stderr))
            self.assertTrue(os.path.isfile(os.path.join(checkout, "af2_initial_guess", "predict_pdb.py")))
            r = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--checkout", checkout], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            n = json.load(open(os.path.join(TREE, "stock", "PINS.json")))["checkout"]["n_files"]
            self.assertIn(f"checkout: {n}/{n} pinned files present", r.stdout)
            open(os.path.join(checkout, "af2_initial_guess", "predict_pdb.py"), "a").write("\n# edited\n")   # an edited tree is refused by name
            r = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--checkout", checkout], capture_output=True, text=True)
            self.assertEqual(r.returncode, 3); self.assertIn("predict_pdb.py", r.stdout + r.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _archive(members):
    """An uncompressed tar stream holding {name: bytes} — the shape of the parameter archive."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __init__(self, data):
        super().__init__(data); self.length = len(data)


class WeightsStep(unittest.TestCase):
    """af2ig_opt.weights.fetch with an injected archive stream (records the URL asked for), a pin table shaped like stock/PINS.json `weights`, and a
    matcher computing the verdict the routes' gate would (pinned / sha256 / present)."""

    REL = "params/params_model_1_ptm.npz"

    def setUp(self):
        from af2ig_opt import weights
        self.w = weights
        self.dir = tempfile.mkdtemp(prefix="af2ig_params_")
        self.member = b"the pinned parameter bytes"
        self.tar = _archive({"params_model_1.npz": b"another model", "params_model_1_ptm.npz": self.member, "LICENSE": b"CC BY 4.0"})
        self.pins = {"weights": {"env": "AF2_PARAMS", "file": self.REL, "sha256": _sha(self.member), "bytes": len(self.member),
                                 "source": {"url": "https://example.invalid/params.tar", "sha256": _sha(self.tar), "bytes": len(self.tar), "members": 3}}}
        self.asked = []

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def opener(self, data=None, exc=None):
        def op(url):
            self.asked.append(url)
            if exc: raise exc
            return _Resp(self.tar if data is None else data)
        return op

    def matcher(self, d):
        p = os.path.join(d, self.REL)
        if not os.path.isfile(p): return {"pinned": None, "sha256": None, "bad": [f"weights_missing: {p}"]}
        h = _sha(open(p, "rb").read())
        return {"pinned": h == self.pins["weights"]["sha256"], "sha256": h, "pin": self.pins["weights"]["sha256"], "bad": []}

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.w.fetch(self.dir, pins=self.pins, opener=kw.pop("opener", self.opener()), matcher=kw.pop("matcher", self.matcher), out=out)
        return rc, out.getvalue()

    def test_absent_file_is_fetched_from_the_pinned_archive_and_checked(self):
        rc, out = self.fetch()
        self.assertEqual(rc, self.w.EXIT_OK, out)
        self.assertEqual(self.asked, [self.pins["weights"]["source"]["url"]])                                     # the URL is the pin table's, nothing else
        self.assertEqual(open(os.path.join(self.dir, self.REL), "rb").read(), self.member)
        self.assertEqual(sorted(os.listdir(os.path.join(self.dir, "params"))), ["params_model_1_ptm.npz"])         # the other members are not kept; no temporary file remains
        self.assertIn("(the pinned archive)", out); self.assertIn("WEIGHTS OK: 1/1", out); self.assertIn(f"export AF2_PARAMS={self.dir}", out)

    def test_present_file_is_kept_and_only_checked(self):
        os.makedirs(os.path.join(self.dir, "params")); open(os.path.join(self.dir, self.REL), "wb").write(self.member)
        rc, out = self.fetch(opener=self.opener(exc=AssertionError("must not fetch")))
        self.assertEqual((rc, self.asked), (self.w.EXIT_OK, []), out); self.assertIn("present", out)

    def test_a_file_off_the_pin_is_refused_and_left_in_place(self):
        os.makedirs(os.path.join(self.dir, "params")); open(os.path.join(self.dir, self.REL), "wb").write(b"corrupt")
        rc, out = self.fetch()
        self.assertEqual(rc, self.w.EXIT_FAIL); self.assertIn("REFUSED", out); self.assertIn(_sha(b"corrupt")[:16], out)
        self.assertEqual(open(os.path.join(self.dir, self.REL), "rb").read(), b"corrupt")                          # never deleted

    def test_a_repacked_archive_is_named_but_the_member_digest_decides(self):
        other = _archive({"params_model_1_ptm.npz": self.member})                                                   # same member, different archive bytes
        rc, out = self.fetch(opener=self.opener(data=other))
        self.assertEqual(rc, self.w.EXIT_OK, out); self.assertIn("NOT the digest stock/PINS.json weights.source pins", out); self.assertIn("WEIGHTS OK", out)
        bad = _archive({"params_model_1_ptm.npz": b"tampered"})
        shutil.rmtree(os.path.join(self.dir, "params"))
        rc, out = self.fetch(opener=self.opener(data=bad))
        self.assertEqual(rc, self.w.EXIT_FAIL, out); self.assertIn("REFUSED", out)
        self.assertEqual(open(os.path.join(self.dir, self.REL), "rb").read(), b"tampered")                         # fetched, off the pin: left in place for inspection

    def test_transfer_errors_and_an_archive_without_the_member_fail_by_name(self):
        rc, out = self.fetch(opener=self.opener(exc=OSError("network unreachable")))
        self.assertEqual(rc, self.w.EXIT_FAIL); self.assertIn("FAILED fetching", out); self.assertIn("network unreachable", out)
        rc, out = self.fetch(opener=self.opener(data=_archive({"LICENSE": b"x"})))
        self.assertEqual(rc, self.w.EXIT_FAIL); self.assertIn("holds no member named params_model_1_ptm.npz", out)
        self.assertFalse(os.path.exists(os.path.join(self.dir, self.REL)))

    def test_no_source_url_in_the_pins_fails_by_name(self):
        self.pins["weights"]["source"] = {}
        rc, out = self.fetch()
        self.assertEqual(rc, self.w.EXIT_FAIL); self.assertIn("names no archive URL", out); self.assertEqual(self.asked, [])

    def test_the_pin_table_is_stock_pins_json(self):
        """The real table: one file under params/, a sha256, and a source archive with an https URL — what fetch reads, nothing transcribed here."""
        W = json.load(open(os.path.join(TREE, "stock", "PINS.json")))["weights"]
        self.assertEqual((W["env"], W["file"]), ("AF2_PARAMS", self.REL)); self.assertRegex(W["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(W["source"]["url"].startswith("https://") and W["source"]["url"].endswith(".tar"), W["source"]["url"]); self.assertRegex(W["source"]["sha256"], r"^[0-9a-f]{64}$")

    def test_usage(self):
        self.assertEqual(self.w.main([]), self.w.EXIT_USAGE); self.assertEqual(self.w.main(["--dir"]), self.w.EXIT_USAGE)


class EnvironmentFiles(unittest.TestCase):
    """environment/ builds the pinned stack: the Dockerfile's base image is stock/PINS.json `pinned_stack.base_image`, its interpreter the pinned python,
    the lock and `run.sh install` its install steps; the lock carries every `pins` entry at the pinned version; apptainer.def converts that image."""

    def setUp(self):
        self.pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
        self.df = open(os.path.join(TREE, "environment", "Dockerfile"), encoding="utf-8").read()
        self.lock = {}
        for line in open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                n, v = line.split("=="); self.lock[n.lower().replace("_", "-")] = v

    def test_dockerfile_builds_the_pinned_stack(self):
        m = re.search(r"^FROM (\S+)$", self.df, re.M); self.assertIsNotNone(m)
        self.assertEqual(m.group(1), self.pins["pinned_stack"]["base_image"]); self.assertIn("-runtime-", m.group(1))
        self.assertRegex(self.df, r"cpython-%s\.\d+\+" % re.escape(self.pins["pinned_stack"]["python"]))
        self.assertIn("COPY af2ig/environment/requirements.lock ", self.df); self.assertIn("--no-deps -r", self.df); self.assertIn("RUN bash run.sh install", self.df)
        self.assertIn("AF2IG_DIR=/kit/af2ig/dl_binder_design/af2_initial_guess", self.df); self.assertNotIn("AF2_PARAMS=", self.df.split("\nFROM", 1)[1])   # the tree's location is the image's; the weights root is never baked in
        self.assertRegex(open(os.path.join(TREE, "environment", "apptainer.def"), encoding="utf-8").read(), r"(?m)^From: af2ig-kit:")

    def test_lock_carries_every_pin_at_the_pinned_version(self):
        sys.path.insert(0, os.path.join(TREE, "stock"))
        try:
            import check_pins as cp
        finally:
            sys.path.pop(0)
        self.assertTrue(set(self.pins["pins_gpu_only"]) <= set(self.pins["pins"]))
        for name, want in self.pins["pins"].items():
            self.assertIn(name, self.lock); self.assertEqual(cp.normalize(self.lock[name]), cp.normalize(want), name)
        self.assertEqual(len(self.lock), len(set(self.lock)))
        self.assertNotIn("pip", self.lock); self.assertNotIn("setuptools", self.lock)                                  # the interpreter's bundled pip and setuptools are not lock lines


if __name__ == "__main__":
    unittest.main()
