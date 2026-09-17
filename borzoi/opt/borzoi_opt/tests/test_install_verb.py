"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), the weights step's digest gate (borzoi_opt.weights.fetch with an injected fetcher and pin table; no network), the pin check's
refusal on an interpreter without the pinned upstream, and the agreement of environment/ with stock/PINS.json. CPU only; no TensorFlow."""
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/, environment/
RUN_SH = os.path.join(TREE, "run.sh")
PINS = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m borzoi_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="borzoi_install_verb_")
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

    def test_install_runs_pip_on_the_kit_alone_then_the_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"-m pip install --no-deps --no-build-isolation -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py"], calls)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/borzoi"], ["install", "--weights=/data/borzoi"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["-m borzoi_opt.weights /data/borzoi"], calls)
            self.assertEqual(len(calls), 3, calls)
            self.assertIn("/data/borzoi/f0/model0_best.h5", out)               # the closing line names the <model_file> argument the directory now holds

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                     ["install", "--mode", "exact"], ["install", "--config", "h100"], ["install", "--det"]):
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
        --weights) — a read-only image cannot re-run pip. The stub answers the installed-from-this-tree probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py", "-m borzoi_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_usage_lists_the_verb(self):
        rc, out, calls = self.run_sh(["bogus"])
        self.assertEqual((rc, calls), (2, []), out); self.assertIn("run.sh install [--weights DIR]", out)


class WeightsDigestGate(unittest.TestCase):
    """borzoi_opt.weights.fetch: fetch what is absent, then every file's sha256 against the pin — a mismatch or a failed transfer fails by name."""

    def setUp(self):
        from borzoi_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="borzoi_weights_")
        self.payload = {"f0/model0_best.h5": b"fold-0 weights", "f1/model0_best.h5": b"fold-1 weights"}
        self.files = {k: {"url": f"https://upstream.example/borzoi/{k}", "sha256": hashlib.sha256(v).hexdigest(), "size_bytes": len(v)} for k, v in self.payload.items()}
        self.calls = []

    def fetch_one(self, url, dest):
        self.calls.append((url, dest))
        rel = url.split("/borzoi/", 1)[1]
        with open(dest, "wb") as f: f.write(self.payload[rel])

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, files=kw.pop("files", self.files), fetch_one=kw.pop("fetch_one", self.fetch_one), out=lambda s: out.write(s + "\n"), **kw)
        return rc, out.getvalue()

    def test_all_fetched_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 2/2", out)
        self.assertEqual([u for u, _ in self.calls], [self.files[k]["url"] for k in sorted(self.files)])
        for k in self.files: self.assertTrue(os.path.isfile(os.path.join(self.dir, k)))
        self.assertIn(os.path.join(self.dir, "f0", "model0_best.h5"), out.splitlines()[-1])     # the closing line names the <model_file> path

    def test_a_present_file_is_kept_and_checked_not_fetched(self):
        p = os.path.join(self.dir, "f0", "model0_best.h5"); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(self.payload["f0/model0_best.h5"])
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("present:  f0/model0_best.h5", out)
        self.assertEqual([u for u, _ in self.calls], [self.files["f1/model0_best.h5"]["url"]])

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        p = os.path.join(self.dir, "f0", "model0_best.h5"); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(b"other bytes")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        self.assertIn("REFUSED:  f0/model0_best.h5", out); self.assertIn("WEIGHTS NOT READY: 1 of 2", out.splitlines()[-1])
        self.assertTrue(os.path.isfile(p)); self.assertEqual(open(p, "rb").read(), b"other bytes")

    def test_a_fetch_error_is_relayed_and_nothing_is_left_half_written(self):
        def broken(url, dest): raise ConnectionError("network unreachable")
        rc, out = self.fetch(fetch_one=broken)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED:   f0/model0_best.h5: ConnectionError: network unreachable", out)
        self.assertEqual(sorted(os.listdir(self.dir)), ["f0", "f1"])            # the per-file directories, no partial file
        self.assertEqual(os.listdir(os.path.join(self.dir, "f0")), [])

    def test_download_writes_through_a_part_file(self):
        """weights.download streams into <dest>.part and renames into place: an interrupted transfer never leaves a file under the final name."""
        import urllib.request
        dest = os.path.join(self.dir, "f9", "model0_best.h5")
        class Resp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        saved = urllib.request.urlopen
        urllib.request.urlopen = lambda url, timeout=None: Resp(b"streamed bytes")
        try:
            self.W.download("https://upstream.example/x", dest)
        finally:
            urllib.request.urlopen = saved
        self.assertEqual(open(dest, "rb").read(), b"streamed bytes"); self.assertFalse(os.path.exists(dest + ".part"))

    def test_the_wired_table_is_the_pin_tables_fetchable_entries(self):
        """weights.wired() = the stock/PINS.json `weights` entries with a url and a digest (f0/model0_best.h5 at this pin); prose entries are skipped;
        every url is HTTPS on upstream's bucket and ends with the file's own name."""
        table = self.W.wired(PINS)
        self.assertEqual(sorted(table), ["f0/model0_best.h5"])
        for rel, e in table.items():
            self.assertTrue(e["url"].startswith("https://storage.googleapis.com/seqnn-share/borzoi/"), e["url"])
            self.assertTrue(e["url"].endswith("/" + rel), (rel, e["url"]))
            self.assertRegex(e["sha256"], r"^[0-9a-f]{64}$"); self.assertGreater(e["size_bytes"], 0)


class PinCheckRefusesAnUnpinnedInterpreter(unittest.TestCase):
    """stock/check_pins.py on this interpreter (the kit's test environment: no baskerville, no borzoi, no stock entry): exit 3, one REFUSED line per
    missing piece, the stack note on stderr — and it imports nothing outside the standard library."""

    def test_refusal_lines_and_exit_code(self):
        env = {k: v for k, v in os.environ.items() if k not in ("BORZOI_DIR", "PYTHONPATH")}
        env["PATH"] = "/usr/bin:/bin"
        r = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py")], capture_output=True, text=True, env=env)
        try:
            import baskerville  # noqa: F401 — an environment that does carry the pinned upstream is judged by the install step itself, not here
            self.skipTest("baskerville is importable here: the refusal path is not this environment's")
        except ImportError:
            pass
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        refused = [l for l in r.stderr.splitlines() if l.startswith("REFUSED: ")]
        self.assertTrue(any("baskerville is not installed" in l for l in refused), r.stderr)
        self.assertTrue(any("borzoi is not installed" in l for l in refused), r.stderr)
        self.assertTrue(any("stock entry not found: borzoi_sad.py on PATH (BORZOI_DIR unset)" in l for l in refused), r.stderr)
        self.assertEqual(subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--bogus"], capture_output=True, text=True).returncode, 2)

    def test_standard_library_only(self):
        src = open(os.path.join(TREE, "stock", "check_pins.py"), encoding="utf-8").read()
        mods = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", src, re.M))
        stdlib = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else None
        if stdlib is None: self.skipTest("sys.stdlib_module_names needs Python 3.10")
        self.assertTrue(all(m.split(".")[0] in stdlib for m in mods), sorted(mods))


class EnvironmentAgreesWithThePins(unittest.TestCase):
    """environment/ is written from stock/PINS.json's facts: the base image tag carries the pinned Python, the lock pins every `pins` entry at its
    version and names both upstream commits, the Dockerfile installs the two archives PINS names with the versions PINS records."""

    def setUp(self):
        env = os.path.join(TREE, "environment")
        self.dockerfile = open(os.path.join(env, "Dockerfile"), encoding="utf-8").read()
        self.lock = [l.strip() for l in open(os.path.join(env, "requirements.lock"), encoding="utf-8") if l.strip() and not l.startswith("#")]
        self.apptainer = open(os.path.join(env, "apptainer.def"), encoding="utf-8").read()

    def test_base_image_is_the_pinned_python(self):
        m = re.search(r"^FROM python:([0-9.]+)-slim-bookworm$", self.dockerfile, re.M)
        self.assertTrue(m, "FROM line"); self.assertEqual(m.group(1), PINS["python"])

    def test_lock_pins_the_pin_table(self):
        pinned = dict(l.split("==", 1) for l in self.lock if "==" in l and not l.startswith("-e "))
        norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
        by_norm = {norm(k): v for k, v in pinned.items()}
        for name, version in PINS["pins"].items():
            self.assertEqual(by_norm.get(norm(name)), version, name)
        self.assertEqual(len(self.lock), len(set(self.lock)))                  # no duplicate lines

    def test_lock_and_dockerfile_carry_the_upstream_pins(self):
        for name, up in PINS["upstream"].items():
            line = [l for l in self.lock if l.startswith("-e git+") and l.endswith(f"#egg={name}")]
            self.assertEqual(len(line), 1, name); self.assertIn(f"{up['repo']}.git@{up['commit']}#egg=", line[0])
            self.assertIn(f"borzoi/{up['archive']}", self.dockerfile)           # COPY'd from the kit's stock/
            self.assertIn(f"/opt/{name}", self.dockerfile)
            version = up["version"].split(" ", 1)[0]                             # "1.0.0 (…)" → "1.0.0"
            self.assertIn(f"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_{name.upper()}={version}", self.dockerfile)

    def test_install_step_and_apptainer_source(self):
        self.assertIn("RUN bash run.sh install", self.dockerfile)
        self.assertIn("ENV BORZOI_DIR=/opt/borzoi", self.dockerfile)             # upstream's variable: the stock entry is $BORZOI_DIR/src/scripts/borzoi_sad.py
        self.assertRegex(self.apptainer, r"(?m)^Bootstrap: docker-daemon\nFrom: borzoi-kit:dev$")


if __name__ == "__main__":
    unittest.main()
