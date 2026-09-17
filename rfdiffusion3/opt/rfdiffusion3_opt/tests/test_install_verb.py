"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step (rfdiffusion3_opt.weights: upstream's installer bound to the pinned registry entry, then the size and digest
gate — with injected stand-ins; no network, no upstream). CPU only."""
import hashlib
import io
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest

from .. import registry

TREE = registry.tree_home()                                              # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m rfdiffusion3_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rfd3_install_verb_")
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

    def test_install_runs_pip_then_the_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py --expect any"], calls)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/weights/rfdiffusion3"], ["install", "--weights=/data/weights/rfdiffusion3"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["-m rfdiffusion3_opt.weights /data/weights/rfdiffusion3"], calls)
            self.assertEqual(len(calls), 3, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--json"], ["install", "--weights="], ["install", "--bogus"],
                     ["install", "extra"], ["install", "--mode", "exact"], ["install", "--config", "h100"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls))
            self.assertIn("run.sh", out)

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual((rc, len(calls)), (1, 1), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual((rc, len(calls)), (3, 2), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, len(calls)), (1, 3), (out, calls))

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit) in both interpreters: `install` there names the skip and goes on to
        the pin check (and --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py --expect any", "-m rfdiffusion3_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_the_other_verbs_keep_their_dispatch(self):
        text = open(RUN_SH, encoding="utf-8").read()
        self.assertEqual(re.search(r'case "\$CMD" in ([a-z|]+)\) ;; ([a-z|]+)\) ;;', text).groups(), ("design|check|warm", "install"))


class WeightsStep(unittest.TestCase):
    """rfdiffusion3_opt.weights.fetch: upstream's installer when the file is absent, then bytes and sha256 against the pin — a mismatch fails by
    name and the file stays."""

    def setUp(self):
        from .. import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="rfd3_weights_")
        self.payload = b"checkpoint-bytes"
        self.pin = {"registry_name": "rfd3", "filename": "rfd3_latest.ckpt", "url": "https://upstream.example/pub/rfd3/rfd3_x.ckpt",
                    "sha256": hashlib.sha256(self.payload).hexdigest(), "bytes": len(self.payload)}
        self.calls = []

    def installer(self, directory):
        self.calls.append(directory)
        p = os.path.join(directory, self.pin["filename"])
        if not os.path.exists(p):                                        # upstream's rule: an existing file is left alone
            with open(p, "wb") as f: f.write(self.payload)

    @staticmethod
    def digest(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest(), "hashed"

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, pin=kw.pop("pin", self.pin), installer=kw.pop("installer", self.installer), digest=kw.pop("digest", self.digest), out=out, **kw)
        return rc, out.getvalue()

    def test_fetched_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("rfd3_latest.ckpt: fetching", out); self.assertIn("WEIGHTS OK: rfd3_latest.ckpt", out)
        self.assertIn(f"export RFD3_CKPT={os.path.join(self.dir, 'rfd3_latest.ckpt')}", out)
        self.assertEqual(self.calls, [self.dir])

    def test_a_present_file_is_kept_and_checked_without_upstream(self):
        with open(os.path.join(self.dir, "rfd3_latest.ckpt"), "wb") as f: f.write(self.payload)
        rc, out = self.fetch(installer=lambda d: self.fail("upstream's installer must not run for a present file"))
        self.assertEqual(rc, 0, out); self.assertIn("rfd3_latest.ckpt: present", out)

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        p = os.path.join(self.dir, "rfd3_latest.ckpt")
        with open(p, "wb") as f: f.write(b"X" * len(self.payload))                                  # right size, wrong bytes
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out); self.assertIn("REFUSED", out); self.assertIn("not the pin's", out); self.assertTrue(os.path.exists(p))

    def test_a_short_file_is_refused_by_size_before_hashing(self):
        p = os.path.join(self.dir, "rfd3_latest.ckpt")
        with open(p, "wb") as f: f.write(b"short")
        rc, out = self.fetch(digest=lambda path: self.fail("a file of the wrong size is not hashed"))
        self.assertEqual(rc, 1, out); self.assertIn("is 5 bytes", out); self.assertTrue(os.path.exists(p))

    def test_a_fetch_error_is_relayed_and_nothing_deleted(self):
        def broken(directory):
            with open(os.path.join(directory, "rfd3_latest.ckpt"), "wb") as f: f.write(b"par")
            raise OSError("connection reset")
        rc, out = self.fetch(installer=broken)
        self.assertEqual(rc, 1, out); self.assertIn("FAILED fetching rfd3_latest.ckpt: OSError: connection reset", out)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "rfd3_latest.ckpt")))

    def test_usage(self):
        self.assertEqual(self.W.main([]), 2); self.assertEqual(self.W.main(["--weights"]), 2); self.assertEqual(self.W.main(["a", "b"]), 2)


class UpstreamInstallerBinding(unittest.TestCase):
    """weights.upstream_installer binds upstream's `foundry install` step to the pinned registry entry and refuses an entry that no longer says
    what stock/PINS.json recorded; the pin's record is upstream's own constants at the pinned commit (read from the stock archive)."""

    def setUp(self):
        self.pin = registry.pins()["checkpoint"]
        self.rec = []
        reg = types.ModuleType("foundry.inference_engines.checkpoint_registry")
        Entry = lambda url, filename: types.SimpleNamespace(url=url, filename=filename, description="", sha256=None)   # noqa: E731
        reg.REGISTERED_CHECKPOINTS = {self.pin["registry_name"]: Entry(self.pin["url"], self.pin["filename"]), "elsewhere": Entry("https://upstream.example/x.ckpt", "x.ckpt")}
        cli = types.ModuleType("foundry_cli.download_checkpoints"); cli.install_model = lambda name, d, force=False: self.rec.append((name, str(d), force))
        self.mods = {"foundry": types.ModuleType("foundry"), "foundry.inference_engines": types.ModuleType("foundry.inference_engines"),
                     "foundry.inference_engines.checkpoint_registry": reg, "foundry_cli": types.ModuleType("foundry_cli"), "foundry_cli.download_checkpoints": cli}
        self.saved = {k: sys.modules.get(k) for k in self.mods}
        sys.modules.update(self.mods)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v

    def test_binds_upstreams_step_to_the_pinned_entry(self):
        from .. import weights
        weights.upstream_installer(self.pin)("/data/w")
        self.assertEqual(self.rec, [(self.pin["registry_name"], "/data/w", False)])

    def test_an_entry_off_the_pin_is_refused_before_any_transfer(self):
        from .. import weights
        for pin in (dict(self.pin, url=self.pin["url"] + ".moved"), dict(self.pin, filename="rfd3_other.ckpt"), dict(self.pin, registry_name="absent")):
            with self.assertRaises(RuntimeError): weights.upstream_installer(pin)
        self.assertEqual(self.rec, [])

    def test_the_pin_records_upstreams_constants(self):
        archive = os.path.join(TREE, registry.pins()["upstream"]["foundry"]["archive"])
        with tarfile.open(archive) as tf:
            member = [m for m in tf.getmembers() if m.name.endswith("src/foundry/inference_engines/checkpoint_registry.py")][0]
            text = tf.extractfile(member).read().decode()
        block = re.search(r'"%s": RegisteredCheckpoint\((.*?)\)' % re.escape(self.pin["registry_name"]), text, re.S).group(1)
        self.assertIn(f'url="{self.pin["url"]}"', block); self.assertIn(f'filename="{self.pin["filename"]}"', block)


if __name__ == "__main__":
    unittest.main()
