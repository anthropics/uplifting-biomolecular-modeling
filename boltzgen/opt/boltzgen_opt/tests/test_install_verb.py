"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step (boltzgen_opt.weights: the digest gate with an injected route and pin table, the upstream route's call shape with a
stand-in downloader, the refs it points, and PINS' six files against upstream's own artifact table read from the wheel). CPU only, no network."""
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")
SNAP = "0123456789abcdef0123456789abcdef01234567"

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS / STUB_RC_PROBE
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m boltzgen_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="boltzgen_install_verb_")
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
        self.assertEqual(calls, [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py"], calls)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/boltzgen-cache"], ["install", "--weights=/data/boltzgen-cache"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["-m boltzgen_opt.weights /data/boltzgen-cache"], calls)
            self.assertEqual(len(calls), 3, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--bogus"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                     ["install", "--mode", "exact"], ["install", "--config", "h100"], ["--mode", "fast", "install"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls))
            self.assertIn("run.sh:", out)

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
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py", "-m boltzgen_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_the_other_verbs_do_not_reach_the_install_step(self):
        """design/check/warm go past the install block to the package probe (the stub imports nothing, so the probe's line is the NOT ACTIVE one)."""
        rc, out, calls = self.run_sh(["check", "--mode", "off"])
        self.assertNotIn("pip install -e", " ".join(c for c in calls if c.startswith("-m pip")))
        self.assertFalse(any("check_pins.py" in c or "boltzgen_opt.weights" in c for c in calls), calls)


class WeightsDigestGate(unittest.TestCase):
    """boltzgen_opt.weights.fetch: fetch through the route, point the refs, then every file's sha256 against the pin — a mismatch, an absent file or a
    path without a route fails by name; nothing is deleted."""

    def setUp(self):
        from boltzgen_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="boltzgen_weights_")
        self.payload = {f"models--boltzgen--boltzgen-1/snapshots/{SNAP}/a.ckpt": b"ckpt-a", f"models--boltzgen--boltzgen-1/snapshots/{SNAP}/b.ckpt": b"ckpt-b",
                        f"datasets--boltzgen--inference-data/snapshots/{SNAP}/mols.zip": b"mols"}
        self.files = [{"file": k, "sha256": hashlib.sha256(v).hexdigest()} for k, v in self.payload.items()]
        self.calls = []

    def route(self, rel):
        def call():
            self.calls.append(rel)
            p = os.path.join(self.dir, rel)
            if not os.path.exists(p):                                   # huggingface_hub's rule: fetched only when absent
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f: f.write(self.payload[rel])
        return call

    @staticmethod
    def digest(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, files=kw.pop("files", self.files), route=kw.pop("route", self.route), digest=kw.pop("digest", self.digest), out=out, **kw)
        return rc, out.getvalue()

    def test_all_fetched_pinned_and_the_refs_point_at_the_snapshot(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 3/3", out); self.assertIn(f"export BOLTZGEN_CACHE={self.dir}", out)
        self.assertEqual(self.calls, [f["file"] for f in self.files])
        for f in self.files: self.assertTrue(os.path.isfile(os.path.join(self.dir, f["file"])))
        for repo in ("models--boltzgen--boltzgen-1", "datasets--boltzgen--inference-data"):
            self.assertEqual(open(os.path.join(self.dir, repo, "refs", "main")).read(), SNAP)
        self.assertEqual(out.count("refs/main: written"), 2, out)
        rc, out = self.fetch()                                               # the idempotent second run: everything present, refs already right, no fetch
        self.assertEqual(rc, 0, out); self.assertEqual(out.count(": present"), 3, out); self.assertEqual(out.count("names the pinned snapshot"), 2, out)

    def test_a_ref_naming_another_snapshot_is_repointed_by_name(self):
        ref = os.path.join(self.dir, "models--boltzgen--boltzgen-1", "refs", "main"); os.makedirs(os.path.dirname(ref))
        with open(ref, "w") as f: f.write("f" * 40)
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out); self.assertIn("was ffffffffffff, repointed", out); self.assertEqual(open(ref).read(), SNAP)

    def test_a_present_file_is_kept_and_checked(self):
        rel = self.files[2]["file"]; p = os.path.join(self.dir, rel); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(b"mols")
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out); self.assertIn(f"{rel}: present", out); self.assertNotIn(rel, self.calls)

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        rel = self.files[0]["file"]; p = os.path.join(self.dir, rel); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(b"other bytes")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        self.assertIn("REFUSED: 1 of 3", out); self.assertIn(rel, out.splitlines()[-1])
        self.assertTrue(os.path.isfile(p)); self.assertEqual(open(p, "rb").read(), b"other bytes")

    def test_a_path_without_an_upstream_route_is_refused(self):
        rc, out = self.fetch(files=self.files + [{"file": "elsewhere/file.bin", "sha256": "0" * 64}], route=lambda rel: None if rel.startswith("elsewhere/") else self.route(rel))
        self.assertEqual(rc, 1, out); self.assertIn("FAILED: elsewhere/file.bin is not a Hugging Face cache path", out)
        self.assertIsNone(self.W.upstream_route(self.dir, download=lambda *a, **k: None)("elsewhere/file.bin"))   # the real route: no Hugging Face artifact, no call

    def test_a_route_error_is_relayed_by_name(self):
        def broken(rel):
            def call(): raise OSError("network unreachable")
            return call
        rc, out = self.fetch(route=broken)
        self.assertEqual(rc, 1, out); self.assertIn("FAILED fetching", out); self.assertIn("network unreachable", out)


class UpstreamRoute(unittest.TestCase):
    """weights.upstream_route: PINS' cache path → upstream's hf_hub_download call (repo, file, repo type, the pinned snapshot as revision, boltzgen as
    the library name, DIR as cache_dir); a path that names no Hugging Face artifact has no route. The downloader is a stand-in recording the call."""

    def test_the_call_shape_per_repository_kind(self):
        from boltzgen_opt import weights
        rec = []
        route = weights.upstream_route("/w", download=lambda *a, **k: rec.append((a, k)))
        route(f"models--boltzgen--boltzgen-1/snapshots/{SNAP}/boltzgen1_diverse.ckpt")()
        route(f"datasets--boltzgen--inference-data/snapshots/{SNAP}/mols.zip")()
        self.assertIsNone(route("models--boltzgen--boltzgen-1/refs/main")); self.assertIsNone(route("elsewhere/file.bin"))
        self.assertEqual(rec, [(("boltzgen/boltzgen-1", "boltzgen1_diverse.ckpt"), {"repo_type": "model", "revision": SNAP, "library_name": "boltzgen", "cache_dir": "/w"}),
                               (("boltzgen/inference-data", "mols.zip"), {"repo_type": "dataset", "revision": SNAP, "library_name": "boltzgen", "cache_dir": "/w"})])

    def test_pins_weights_are_upstreams_artifact_table_at_the_pinned_snapshots(self):
        """stock/PINS.json weights.files = every artifact of upstream's ARTIFACTS table (boltzgen/cli/boltzgen.py in the stock wheel: repo, file name,
        model|dataset), each at the snapshot PINS names for its repository — so the weights step fetches exactly what upstream's `download all` would."""
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
        w = pins["weights"]
        with zipfile.ZipFile(os.path.join(TREE, "stock", os.path.basename(pins["wheel"]["file"]))) as z:
            cli = z.read("boltzgen/cli/boltzgen.py").decode()
        table = re.search(r"\nARTIFACTS[^=]*=\s*\{(.*?)\n\}\n", cli, re.S).group(1)
        upstream = sorted(re.findall(r'"huggingface:([^:"]+):([^"]+)",?\s*"(model|dataset)"', re.sub(r"\s+", " ", table)))
        self.assertEqual(len(upstream), 6, upstream)
        from boltzgen_opt import weights
        mine = []
        for row in w["files"]:
            a = weights.parse_cache_path(row["file"]); self.assertIsNotNone(a, row["file"])
            mine.append((a["repo_id"], a["filename"], a["repo_type"]))
            self.assertEqual(a["revision"], w["model_snapshot"] if a["repo_type"] == "model" else w["data_snapshot"], row["file"])
            self.assertEqual(a["repo_id"], w["model_repo"] if a["repo_type"] == "model" else w["data_repo"], row["file"])
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(sorted(mine), upstream)


if __name__ == "__main__":
    unittest.main()
