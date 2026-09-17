"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step's digest gate (openfold3_opt.weights.fetch with an injected route and pin table; no network, no upstream). CPU only."""
import hashlib
import io
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS / STUB_RC_PROBE
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m openfold3_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="openfold3_install_verb_")
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
        self.assertEqual(len(calls), 2, calls)
        self.assertEqual(calls[0], f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt")
        self.assertEqual(calls[1], f"-I {TREE}/stock/check_pins.py")

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/openfold3"], ["install", "--weights=/data/openfold3"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["-m openfold3_opt.weights /data/openfold3"], calls)
            self.assertEqual(len(calls), 3, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--det"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                     ["install", "--mode", "exact"], ["install", "--config", "h100"], ["install", "--exact-line", "cueq"]):
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
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py", "-m openfold3_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_usage_names_the_verb(self):
        """An unknown verb prints the usage block (exit 2) and calls nothing; the block lists `install [--weights DIR]` beside pred / serve / check / warm."""
        rc, out, calls = self.run_sh(["bogus-verb"])
        self.assertEqual((rc, calls), (2, []), out); self.assertIn("run.sh install [--weights DIR]", out); self.assertIn("run.sh pred", out)


class WeightsDigestGate(unittest.TestCase):
    """openfold3_opt.weights.fetch: fetch through the route, then every file's sha256 against the pin — a mismatch or a file without a route fails by name."""

    def setUp(self):
        from openfold3_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="openfold3_weights_")
        self.payload = {"of3-p2-155k.pt": b"checkpoint-bytes"}
        self.files = [{"local": k, "sha256": hashlib.sha256(v).hexdigest(), "url": f"https://bucket.s3.amazonaws.com/dir/{k}"} for k, v in self.payload.items()]
        self.calls = []

    def route(self, entry):
        local = entry["local"]
        def call():
            self.calls.append(local)
            with open(os.path.join(self.dir, local), "wb") as f: f.write(self.payload[local])
        return call

    def matcher(self, d, files):
        """pinned_matcher's contract over the test's pin table: {status pinned|unknown, files, unknown [{local, sha256}], seconds}."""
        unknown = []
        for f in files:
            p = os.path.join(d, f["local"]); digest = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else "absent"
            if digest != f["sha256"]: unknown.append({"local": f["local"], "sha256": digest})
        return {"status": "unknown" if unknown else "pinned", "files": len(files), "unknown": unknown, "seconds": 0.0}

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, files=kw.pop("files", self.files), route=kw.pop("route", self.route), matcher=kw.pop("matcher", self.matcher), out=out, **kw)
        return rc, out.getvalue()

    def test_fetched_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 1/1", out); self.assertIn(f"export OPENFOLD3_CKPT={os.path.join(self.dir, 'of3-p2-155k.pt')}", out)
        self.assertEqual(self.calls, ["of3-p2-155k.pt"])
        self.assertTrue(os.path.isfile(os.path.join(self.dir, "of3-p2-155k.pt")))

    def test_a_present_file_is_kept_and_checked(self):
        with open(os.path.join(self.dir, "of3-p2-155k.pt"), "wb") as f: f.write(b"checkpoint-bytes")
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("of3-p2-155k.pt: present", out); self.assertEqual(self.calls, [])

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        p = os.path.join(self.dir, "of3-p2-155k.pt")
        with open(p, "wb") as f: f.write(b"other bytes")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        self.assertIn("REFUSED: 1 of 1", out); self.assertIn("of3-p2-155k.pt", out.splitlines()[-1])
        self.assertTrue(os.path.isfile(p)); self.assertEqual(open(p, "rb").read(), b"other bytes")

    def test_a_file_without_a_url_is_refused(self):
        rc, out = self.fetch(route=lambda entry: None)
        self.assertEqual(rc, 1, out)
        self.assertIn("carries no url for of3-p2-155k.pt", out)
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_fetch_error_is_relayed(self):
        def broken(local):
            def call(): raise ConnectionError("network unreachable")
            return call
        rc, out = self.fetch(route=broken)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED fetching of3-p2-155k.pt: ConnectionError: network unreachable", out)
        self.assertEqual(os.listdir(self.dir), [])                       # nothing written, nothing deleted

    def test_pins_weights_file_is_in_upstreams_registry(self):
        """stock/PINS.json weights.file is a file upstream's parameter registry serves (openfold3/entry_points/parameters.py
        OPENFOLD_MODEL_CHECKPOINT_REGISTRY, read from the pinned source as text), it is the registry's default checkpoint, and not a legacy entry —
        so upstream_route has a download for it."""
        import json, re
        src = open(os.path.join(TREE, "stock", "src", "openfold3", "entry_points", "parameters.py"), encoding="utf-8").read()
        entries = dict(re.findall(r'"([A-Za-z0-9_.-]+)":\s*CheckpointEntry\(\s*(?:#[^\n]*\n\s*)?file_name="([^"]+)"', src))
        default = re.search(r'DEFAULT_CHECKPOINT_NAME = "([^"]+)"', src).group(1)
        legacy = re.findall(r'"([^"]+)"', re.search(r"LEGACY_CHECKPOINTS = \[([^\]]*)\]", src).group(1))
        pinned = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))["weights"]["file"]
        self.assertIn(pinned, entries.values(), entries)
        self.assertEqual(entries[default], pinned, (default, entries))
        self.assertTrue(all(entries[n] != pinned for n in legacy), (legacy, entries))

    def test_route_table(self):
        """upstream_route's rule: the pin-table entry's url decides the transport — an S3 https URL → upstream's S3 client
        (bucket, key, target path); any other https URL → the plain download; no url → no route. Both transports injected; nothing fetched."""
        from pathlib import Path
        rec = []
        route = self.W.upstream_route(self.dir, s3_download=lambda b, k, p: rec.append(("s3", b, k, str(p))), https_download=lambda u, p: rec.append(("https", u, p)))
        route({"local": "of3-p2-155k.pt", "url": "https://openfold.s3.amazonaws.com/staging/of3-p2-155k.pt"})()
        route({"local": "b.pt", "url": "https://bucket-x.s3.us-east-1.amazonaws.com/dir/b.pt"})()
        route({"local": "c.pt", "url": "https://example.org/files/c.pt"})()
        self.assertIsNone(route({"local": "d.pt"}))
        self.assertIsNone(route({"local": "e.pt", "url": ""}))
        self.assertEqual(rec, [("s3", "openfold", "staging/of3-p2-155k.pt", str(Path(self.dir) / "of3-p2-155k.pt")),
                               ("s3", "bucket-x", "dir/b.pt", str(Path(self.dir) / "b.pt")),
                               ("https", "https://example.org/files/c.pt", os.path.join(self.dir, "c.pt"))])
        self.assertEqual(self.W.s3_bucket_key("http://openfold.s3.amazonaws.com/x"), None)          # not https
        self.assertEqual(self.W.s3_bucket_key("https://openfold.s3.amazonaws.com/"), None)          # no key

    def test_pin_table_carries_the_url(self):
        """pinned_files() = stock/PINS.json weights (file, sha256, url): the url is an https URL naming the pinned file."""
        files = self.W.pinned_files(TREE)
        self.assertEqual(len(files), 1)
        f = files[0]
        self.assertTrue(f["url"].startswith("https://") and f["url"].endswith("/" + f["local"]), f)
        self.assertEqual(len(f["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
