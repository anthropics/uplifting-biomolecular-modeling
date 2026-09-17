"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), the lever-file step (`mosaic_opt.leverfiles` against a stand-in installed `mosaic` package in a temporary directory), and the weights
step's digest gate (`mosaic_opt.weights.main` with an injected fetch routine and pin table; no network, no upstream). CPU only."""
import hashlib
import io
import os
import stat
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from . import OPT_DIR  # noqa: F401  (tests/__init__ puts the pinned core on sys.path when it is not installed)
from mosaic_opt import leverfiles, registry, stack, weights

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PIP / STUB_RC_PINS / STUB_RC_LEVERS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m mosaic_opt.leverfiles"*) exit "${STUB_RC_LEVERS:-0}" ;;
  *"-m mosaic_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mosaic_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        self.stub = os.path.join(self.bin, "python")
        with open(self.stub, "w") as f: f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, args, **rc):
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        lines = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        self.probes = [c for c in lines if c.startswith("-I -c import os,sys")]           # the installed-from-this-tree probe (one per install call)
        return r.returncode, r.stdout + r.stderr, [c for c in lines if not c.startswith("-I -c import os,sys")]

    def test_install_runs_pip_then_the_pin_check_then_the_lever_files(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py", "-m mosaic_opt.leverfiles"], calls)
        self.assertEqual(len(self.probes), 1)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/mosaic_cache"], ["install", "--weights=/data/mosaic_cache"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[3:], ["-m mosaic_opt.weights /data/mosaic_cache"], calls)
            self.assertEqual(len(calls), 4, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--json"], ["install", "--weights="], ["install", "--bogus"],
                     ["install", "extra"], ["install", "--mode", "exact"], ["install", "--config", "h100"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls))
            self.assertIn("run.sh:", out)

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        for failing, code, ncalls in (("pip", 1, 1), ("pins", 3, 2), ("levers", 1, 3), ("weights", 1, 4)):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(["install", "--weights", "/w"], **{failing: code})
            self.assertEqual((rc, len(calls)), (code, ncalls), (failing, out, calls))

    def test_the_pip_step_is_skipped_when_the_tree_is_installed_already(self):
        rc, out, calls = self.run_sh(["install"], probe=0)
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py", "-m mosaic_opt.leverfiles"], calls)
        self.assertIn("installed from this tree already", out)

    def test_the_other_verbs_are_untouched_by_install_parsing(self):
        text = open(RUN_SH).read()
        self.assertIn('case "$CMD" in design|check|warm|install) ;; *) usage ;; esac', text)
        self.assertEqual(text.count("python -m mosaic_opt.leverfiles"), 1)
        self.assertEqual(text.count("python -m mosaic_opt.weights"), 1)


class LeverFiles(unittest.TestCase):
    """The kit's lever files into a stand-in installed `mosaic` package: add-only, idempotent, extras named and kept."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mosaic_leverfiles_")
        self.mosaic_dir = os.path.join(self.tmp, "site-packages", "mosaic"); os.makedirs(self.mosaic_dir)
        open(os.path.join(self.mosaic_dir, "__init__.py"), "w").close()
        self.src = os.path.join(stack.kit_home(), registry.KIT_FAST_DIR)
        self.names = sorted(f for f in os.listdir(self.src) if f.endswith(".py"))

    def test_source_is_the_kit_directory(self):
        self.assertEqual(leverfiles.source_dir(), self.src)
        self.assertIn("flashattn.py", self.names)
        self.assertIn("__init__.py", self.names)

    def test_first_install_copies_every_file_then_nothing(self):
        p = leverfiles.install(mosaic_dir=self.mosaic_dir)
        self.assertEqual(sorted(p["written"]), self.names)
        for fn in self.names:
            self.assertEqual(open(os.path.join(self.src, fn), "rb").read(), open(os.path.join(self.mosaic_dir, "fast", fn), "rb").read())
        again = leverfiles.install(mosaic_dir=self.mosaic_dir)
        self.assertEqual(again["written"], [])
        self.assertTrue(all(v == "same" for v in again["files"].values()), again["files"])
        self.assertIn("nothing written", leverfiles.summary(again))
        fc = stack.installed_fast_check(mosaic_dir=self.mosaic_dir)                        # the routes' own view of the result: installed and clean
        self.assertTrue(fc["installed"] and fc["clean"], fc)

    def test_a_differing_file_is_rewritten_and_an_extra_file_is_named_and_kept(self):
        leverfiles.install(mosaic_dir=self.mosaic_dir)
        target = os.path.join(self.mosaic_dir, "fast")
        with open(os.path.join(target, "flashattn.py"), "a") as f: f.write("\n# local edit\n")
        with open(os.path.join(target, "stray.py"), "w") as f: f.write("x = 1\n")
        dry = leverfiles.install(mosaic_dir=self.mosaic_dir, dry_run=True)
        self.assertEqual((dry["would_write"], dry["written"], dry["extra"]), (["flashattn.py"], [], ["stray.py"]))
        self.assertIn("would be written", leverfiles.summary(dry, dry_run=True))
        p = leverfiles.install(mosaic_dir=self.mosaic_dir)
        self.assertEqual((p["written"], p["extra"]), (["flashattn.py"], ["stray.py"]))
        self.assertTrue(os.path.isfile(os.path.join(target, "stray.py")))                  # add-only: never removed
        self.assertIn("stray.py", leverfiles.summary(p))
        self.assertEqual(open(os.path.join(self.src, "flashattn.py"), "rb").read(), open(os.path.join(target, "flashattn.py"), "rb").read())

    def test_no_installed_mosaic_is_a_refusal(self):
        with self.assertRaises(ValueError):
            leverfiles.install(mosaic_dir="")
        p = leverfiles.plan(mosaic_dir="")
        self.assertIsNone(p["target"])
        self.assertTrue(all(v == "absent" for v in p["files"].values()))


class WeightsStep(unittest.TestCase):
    """`python -m mosaic_opt.weights DIR` with the fetch routine injected: the digest gate against the pin, files left in place."""
    GOOD = b"boltz2 checkpoint bytes"

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="mosaic_weights_")
        self.pin = {"boltz2_conf.ckpt": {"sha256": hashlib.sha256(self.GOOD).hexdigest()}}
        self.calls = []

    def ensure(self, payload):
        def _ensure(cache_root):
            self.calls.append(str(cache_root))
            bdir = os.path.join(str(cache_root), "boltz"); os.makedirs(os.path.join(bdir, "mols"), exist_ok=True)
            ck = os.path.join(bdir, "boltz2_conf.ckpt"); fetched = not os.path.isfile(ck)
            if fetched:
                with open(ck, "wb") as f: f.write(payload)
            for i in range(3): open(os.path.join(bdir, "mols", f"m{i}.pkl"), "wb").close()
            return {"cache_dir": bdir, "boltz2_conf_ckpt_sha256": hashlib.sha256(open(ck, "rb").read()).hexdigest(), "fetched_now": fetched, "n_mols": 3}
        return _ensure

    def run_main(self, argv, payload=GOOD):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = weights.main(argv, ensure=self.ensure(payload), pin=self.pin)
        return rc, out.getvalue() + err.getvalue()

    def test_fetch_then_ok_then_idempotent(self):
        rc, out = self.run_main([self.dir])
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 1/1", out); self.assertIn("fetched now", out); self.assertIn("CCD molecules: 3 files", out)
        self.assertEqual(os.environ.get(stack.LIBRARY_CACHE_ENV), self.dir)
        self.assertTrue(os.path.isfile(os.path.join(self.dir, stack.WEIGHTS_RELPATH)))
        rc, out = self.run_main([self.dir])
        self.assertEqual(rc, 0, out)
        self.assertIn("present — kept and checked", out); self.assertNotIn("fetched now", out)
        self.assertEqual(self.calls, [self.dir, self.dir])

    def test_a_digest_off_the_pin_is_refused_and_the_file_kept(self):
        rc, out = self.run_main([self.dir], payload=b"not the pinned bytes")
        self.assertEqual(rc, 1, out)
        self.assertIn("REFUSED", out); self.assertIn(self.pin["boltz2_conf.ckpt"]["sha256"], out)
        self.assertTrue(os.path.isfile(os.path.join(self.dir, stack.WEIGHTS_RELPATH)))       # left in place for inspection

    def test_a_failing_fetch_is_exit_1_with_its_words(self):
        def boom(cache_root): raise OSError("network unreachable")
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            rc = weights.main([self.dir], ensure=boom, pin=self.pin)
        self.assertEqual(rc, 1); self.assertIn("network unreachable", err.getvalue())

    def test_usage(self):
        for argv in ([], ["--weights"], [self.dir, "extra"], [""]):
            rc, out = self.run_main(argv)
            self.assertEqual(rc, 2, (argv, out))

    # -- the CCD archive (boltz/mols.tar, unpacked to mols/ by the same downloader) is the second pinned unit ---------------------------------
    TAR = b"the ccd archive bytes"

    def with_archive(self, tar=TAR, write=True):
        """the pin table with "ccd".sha256 and an ensure that also leaves boltz/mols.tar (as boltz's downloader does) unless ``write`` is off."""
        self.pin = dict(self.pin, ccd={"sha256": hashlib.sha256(self.TAR).hexdigest()})
        inner = self.ensure(self.GOOD)
        def _ensure(cache_root):
            rec = inner(cache_root)
            if write:
                with open(os.path.join(rec["cache_dir"], "mols.tar"), "wb") as f: f.write(tar)
            return rec
        return _ensure

    def run_with(self, ensure):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = weights.main([self.dir], ensure=ensure, pin=self.pin)
        return rc, out.getvalue() + err.getvalue()

    def test_the_ccd_archive_at_its_pin_passes_as_the_second_pinned_file(self):
        rc, out = self.run_with(self.with_archive())
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 2/2 pinned files match stock/PINS.json (boltz/boltz2_conf.ckpt, boltz/mols.tar, fetched now)", out)
        self.assertIn(f"boltz/mols.tar: sha256 {hashlib.sha256(self.TAR).hexdigest()}", out)

    def test_a_ccd_archive_off_its_pin_is_refused_by_name_and_kept(self):
        rc, out = self.run_with(self.with_archive(tar=b"tampered archive"))
        self.assertEqual(rc, 1, out)
        self.assertIn(f"REFUSED: {os.path.join(self.dir, 'boltz', 'mols.tar')} sha256 {hashlib.sha256(b'tampered archive').hexdigest()} is not the pin's", out)
        self.assertEqual(open(os.path.join(self.dir, "boltz", "mols.tar"), "rb").read(), b"tampered archive"); self.assertNotIn("WEIGHTS OK", out)

    def test_an_existing_unpacked_ccd_directory_without_the_archive_is_accepted_with_one_note(self):
        """The layout an earlier install leaves (boltz2_conf.ckpt + unpacked mols/, no mols.tar): rc 0, the checkpoint judged, the molecules
        counted and the missing archive said in the one WEIGHTS OK line — `install --weights DIR && … check` keeps working for a returning user."""
        rc, out = self.run_with(self.with_archive(write=False))
        self.assertEqual(rc, 0, out)
        ok = [l for l in out.splitlines() if "WEIGHTS OK: 1/1 pinned file matches stock/PINS.json" in l]
        self.assertEqual(len(ok), 1, out)
        self.assertIn("CCD molecules: 3 files under", ok[0]); self.assertIn("present from an earlier install without the archive boltz/mols.tar", ok[0])
        self.assertIn(f"remove {os.path.join(self.dir, 'boltz', 'mols')} to re-fetch and verify", ok[0]); self.assertNotIn("REFUSED", out)

    def test_neither_archive_nor_molecules_is_refused_by_name(self):
        self.pin = dict(self.pin, ccd={"sha256": hashlib.sha256(self.TAR).hexdigest()})
        def hollow(cache_root):                                       # a fetch that leaves the checkpoint but no molecules and no archive
            bdir = os.path.join(str(cache_root), "boltz"); os.makedirs(bdir, exist_ok=True)
            with open(os.path.join(bdir, "boltz2_conf.ckpt"), "wb") as f: f.write(self.GOOD)
            return {"cache_dir": bdir, "boltz2_conf_ckpt_sha256": hashlib.sha256(self.GOOD).hexdigest(), "fetched_now": True, "n_mols": 0}
        rc, out = self.run_with(hollow)
        self.assertEqual(rc, 1, out)
        self.assertIn(f"REFUSED: {os.path.join(self.dir, 'boltz', 'mols.tar')} is absent and", out); self.assertNotIn("WEIGHTS OK", out)

    def test_stock_pins_carry_the_ccd_archive_digest(self):
        ccd = stack.pins()["weights"]["ccd"]
        self.assertRegex(ccd["sha256"], r"^[0-9a-f]{64}$"); self.assertEqual(weights.ccd_pin(stack.pins()["weights"]), ccd["sha256"])
        self.assertIn("mols.tar", ccd["path"]); self.assertEqual(weights.CCD_ARCHIVE, "mols.tar")
        self.assertIsNone(weights.ccd_pin({"boltz2_conf.ckpt": {"sha256": "0" * 64}}))

    def test_the_pin_is_the_kit_tools_own_constant(self):
        """stock/PINS.json "weights" and the fetch tool's asserted digest name the same checkpoint bytes (the tool is what `warm` and this step call)."""
        F = weights.fetch_tool()
        self.assertEqual(F.BOLTZ2_CONF_CKPT_SHA256, stack.pins()["weights"][weights.PINNED_FILE]["sha256"])
        self.assertTrue(callable(F.ensure_weights))
        self.assertEqual(stack.WEIGHTS_RELPATH, os.path.join("boltz", weights.PINNED_FILE))


if __name__ == "__main__":
    unittest.main()
