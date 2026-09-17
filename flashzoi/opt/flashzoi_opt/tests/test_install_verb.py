"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step's digest gate (flashzoi_opt.weights.fetch with an injected transfer and pin table, checked by stock/check_pins.py's
own weights check on real files; no network, no huggingface_hub). CPU only."""
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
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m flashzoi_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flashzoi_install_verb_")
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
        self.assertEqual(calls, [f"-m pip install -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py --package-only", f"-I {TREE}/stock/check_pins.py --quiet"], calls)   # the kit alone (flashzoi_opt imports nothing from common/opt_core), then the stock pin (the gate), then the stack pins (named when they differ, never a gate)
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/weights/flashzoi/hf"], ["install", "--weights=/data/weights/flashzoi/hf"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[3:], ["-m flashzoi_opt.weights /data/weights/flashzoi/hf"], calls)
            self.assertEqual(len(calls), 4, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--det"], ["install", "--weights="], ["install", "--bogus"],
                     ["install", "extra"], ["install", "--mode", "exact"], ["install", "--config", "h100"], ["fetch"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls))
            self.assertEqual(self.probes, [], (args, self.probes))

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual((rc, len(calls)), (1, 1), (out, calls)); self.assertIn("the install failed", out)
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual((rc, len(calls)), (3, 2), (out, calls)); self.assertIn("the stock is not at its pin", out)   # the stock pin is the one gate (exit 3); the stack pins after it are a NOTE
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, len(calls)), (1, 4), (out, calls))

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit/flashzoi/opt): `install` there names the skip and goes on to the pin
        check (and --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py --package-only", f"-I {TREE}/stock/check_pins.py --quiet", "-m flashzoi_opt.weights /w"], calls)   # stock pin, stack note, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_no_python_on_path_is_named(self):
        os.remove(self.stub)
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual((rc, calls), (2, []), out); self.assertIn("no python on PATH", out)


class WeightsDigestGate(unittest.TestCase):
    """flashzoi_opt.weights.fetch: each pinned replicate's config.json + model.safetensors into the hub cache DIR through the transfer, then
    stock/check_pins.py's weights check over DIR — a digest, a byte count or a missing file fails by name and nothing is deleted."""

    def setUp(self):
        from flashzoi_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="flashzoi_weights_")
        self.saved = {k: os.environ.get(k) for k in ("HF_HUB_CACHE", "HF_HUB_OFFLINE")}
        os.environ.pop("HF_HUB_OFFLINE", None)
        cfg = b'{"model_type": "borzoi"}'
        self.payload, w = {}, {}
        for k, rev in ((0, "a" * 40), (1, "b" * 40)):
            repo = f"johahi/flashzoi-replicate-{k}"; blob = f"weights-of-replicate-{k}".encode()
            self.payload[(repo, "model.safetensors")] = blob; self.payload[(repo, "config.json")] = cfg
            w[repo] = {"revision": rev, "model.safetensors_sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        self.pins = {"weights": w, "config_json_sha256": hashlib.sha256(cfg).hexdigest()}
        self.calls = []

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

    def download(self, repo, filename, revision):
        """huggingface_hub's contract over the test's payload: the file lands at snapshots/<revision>/<filename>; a present file is returned as is."""
        self.calls.append((repo, filename, revision))
        p = self.W.snapshot_file(self.dir, repo, revision, filename)
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f: f.write(self.payload[(repo, filename)])
        return p

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, pins=kw.pop("pins", self.pins), download=kw.pop("download", self.download), out=out, **kw)   # checker: the real stock/check_pins.py weights check
        return rc, out.getvalue()

    def expected_calls(self):
        return [(repo, f, w["revision"]) for repo, w in self.pins["weights"].items() for f in self.W.FETCHED_FILES]

    def test_all_fetched_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 2/2 replicates (4 files)", out); self.assertIn(f"export FLASHZOI_WEIGHTS={self.dir}", out)
        self.assertEqual(self.calls, self.expected_calls())
        self.assertEqual(os.environ.get("HF_HUB_CACHE"), self.dir)                      # the hub cache the check (and every route) reads
        for (repo, f), _ in self.payload.items():
            self.assertTrue(os.path.isfile(self.W.snapshot_file(self.dir, repo, self.pins["weights"][repo]["revision"], f)))

    def test_a_present_file_is_kept_and_checked(self):
        repo = "johahi/flashzoi-replicate-1"; rev = self.pins["weights"][repo]["revision"]
        for f in self.W.FETCHED_FILES: self.download(repo, f, rev)
        self.calls.clear()
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.calls, [c for c in self.expected_calls() if c[0] != repo])   # replicate-1's files were present: no transfer, still checked
        self.assertIn(f"{repo}@{rev[:12]} model.safetensors: present", out)

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        repo = "johahi/flashzoi-replicate-0"; rev = self.pins["weights"][repo]["revision"]
        p = self.W.snapshot_file(self.dir, repo, rev, "model.safetensors"); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(b"weights-of-replicate-X")                      # same byte count as the pin, another digest
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        self.assertIn(f"{repo}: model.safetensors sha256 ", out); self.assertIn("REFUSED: 1 finding(s) above — 1 of 2 replicates", out)
        self.assertEqual(open(p, "rb").read(), b"weights-of-replicate-X")                   # left in place, never deleted
        self.assertIn("johahi/flashzoi-replicate-1: pinned", out)

    def test_a_byte_count_off_the_pin_is_refused_by_name(self):
        repo = "johahi/flashzoi-replicate-1"; rev = self.pins["weights"][repo]["revision"]
        p = self.W.snapshot_file(self.dir, repo, rev, "model.safetensors"); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(b"short")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out); self.assertIn(f"{repo}: model.safetensors is 5 bytes; want ", out)

    def test_a_fetch_error_is_relayed(self):
        def boom(repo, filename, revision): raise OSError("connection reset by peer")
        rc, out = self.fetch(download=boom)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED fetching johahi/flashzoi-replicate-0@aaaaaaaaaaaa config.json: OSError: connection reset by peer", out)
        self.assertIn("*.incomplete", out)

    def test_a_transfer_that_lands_nothing_is_named(self):
        rc, out = self.fetch(download=lambda repo, filename, revision: "/nonexistent")
        self.assertEqual(rc, 1, out); self.assertIn("is not at", out)

    def test_offline_switch_is_refused_by_name_before_any_transfer(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        rc, out = self.fetch(download=None)                                               # the real hub_downloader: refused before huggingface_hub is imported
        self.assertEqual(rc, 1, out); self.assertIn("HF_HUB_OFFLINE=1 is set", out)
        self.assertEqual(self.calls, [])

    def test_offline_switch_is_inert_when_nothing_is_fetched(self):
        for repo, w in self.pins["weights"].items():
            for f in self.W.FETCHED_FILES: self.download(repo, f, w["revision"])
        os.environ["HF_HUB_OFFLINE"] = "1"; self.calls.clear()
        rc, out = self.fetch(download=None)
        self.assertEqual(rc, 0, out); self.assertEqual(self.calls, [])

    def test_snapshot_layout_is_the_one_check_pins_reads(self):
        self.assertEqual(self.W.snapshot_file("/w", "johahi/flashzoi-replicate-3", "336e45f8", "config.json"),
                         "/w/models--johahi--flashzoi-replicate-3/snapshots/336e45f8/config.json")

    def test_usage(self):
        self.assertEqual(self.W.main([]), 2); self.assertEqual(self.W.main(["--weights"]), 2); self.assertEqual(self.W.main(["a", "b"]), 2)


if __name__ == "__main__":
    unittest.main()
