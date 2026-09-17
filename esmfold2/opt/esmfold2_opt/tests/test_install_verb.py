"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step (esmfold2_opt.weights.install_weights with an injected downloader and pin table: the hub-cache paths it fills, the
refs/main pointer, the sha256 check by the routes' own comparison; no network, no upstream). CPU only."""
import hashlib
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
CUBINS = f"-I {TREE}/opt/forward/fast_inference/driver/ef2_nvjit.py cubins"   # the shipped-cubins check (CPU): manifest, sha256s, source keys against this tree
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_PINS / STUB_RC_CUBINS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"ef2_nvjit.py cubins"*) exit "${STUB_RC_CUBINS:-0}" ;;
  *"-m esmfold2_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="esmfold2_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        self.stub = os.path.join(self.bin, "python")
        with open(self.stub, "w") as f: f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, args, extra_env=None, **rc):
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()}); env.update(extra_env or {})
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        lines = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        self.probes = [c for c in lines if c.startswith("-I -c import os,sys")]           # the installed-from-this-tree probe (one per install call)
        return r.returncode, r.stdout + r.stderr, [c for c in lines if not c.startswith("-I -c import os,sys")]

    def test_install_runs_pip_then_the_pin_check_then_the_shipped_cubins_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py", CUBINS], calls)
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/esmfold2_hf"], ["install", "--weights=/data/esmfold2_hf"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], [CUBINS, "-m esmfold2_opt.weights /data/esmfold2_hf"], calls)
            self.assertEqual(len(calls), 4, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--bogus"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                     ["install", "--mode", "exact"], ["install", "--config", "h100"], ["install", "--variant", "fast"]):
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
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], cubins=1)
        self.assertEqual((rc, len(calls)), (1, 3), (out, calls)); self.assertIn("driver/prebuilt/build_prebuilt.py", out)   # the shipped cubins do not match this tree: named, rc 1, no weights step
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, len(calls)), (1, 4), (out, calls))

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py", CUBINS, "-m esmfold2_opt.weights /w"], calls)   # pin check, cubins, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_the_packages_switches_do_not_reach_pip_or_the_weights_step(self):
        """ESMFOLD2_OPT* names are removed from the environment of the pip and weights interpreters (the start-up hook would act on them there)."""
        marker = os.path.join(self.tmp, "envdump")
        with open(self.stub, "w") as f:
            f.write(STUB.replace("printf '%s\\n' \"$*\" >> \"$STUB_LOG\"", "printf '%s|%s\\n' \"$*\" \"${ESMFOLD2_OPT:-unset}\" >> \"$STUB_LOG\""))
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], extra_env={"ESMFOLD2_OPT": "fast"})
        self.assertEqual(rc, 0, out)
        self.assertEqual([c.split("|")[1] for c in calls], ["unset", "fast", "unset", "unset"], calls)   # pip: cleaned; check_pins: the caller's environment; cubins, weights: cleaned


def fake_pins(payload):
    """A pin table of stock/PINS.json's shape over two stand-in repositories: {repo: {snapshot_commit, files: {name: {sha256, size_bytes}}}}."""
    repos = {"org/ModelA": "a" * 40, "org/ModelB": "b" * 40}
    pins = {"weights": {}}
    for (repo, name), data in payload.items():
        w = pins["weights"].setdefault(repo, {"snapshot_commit": repos[repo], "files": {}})
        w["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    return pins


class WeightsStep(unittest.TestCase):
    """esmfold2_opt.weights.install_weights: every pinned file fetched into the hub cache at its pinned commit, refs/main pointed at that commit,
    then every file's sha256 against the pin — a mismatch, an absent file or a failed transfer fails by name and deletes nothing."""

    def setUp(self):
        from esmfold2_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="esmfold2_weights_")
        self.memo = tempfile.mkdtemp(prefix="esmfold2_weights_memo_")
        self._saved = os.environ.get("ESMFOLD2_OPT_WEIGHTS_MEMO_DIR"); os.environ["ESMFOLD2_OPT_WEIGHTS_MEMO_DIR"] = self.memo
        self.payload = {("org/ModelA", "model.safetensors"): b"weights-a", ("org/ModelA", "config.json"): b"{}", ("org/ModelB", "model.safetensors"): b"weights-b"}
        self.pins = fake_pins(self.payload)
        self.calls = []
        self.lines = []

    def tearDown(self):
        if self._saved is None: os.environ.pop("ESMFOLD2_OPT_WEIGHTS_MEMO_DIR", None)
        else: os.environ["ESMFOLD2_OPT_WEIGHTS_MEMO_DIR"] = self._saved

    def dest(self, repo, commit, name):
        return os.path.join(self.W.repo_dir(self.dir, repo), "snapshots", commit, name)

    def fetch(self, repo, commit, name):                                  # the downloader's contract: writes the snapshot entry, returns its path
        self.calls.append((repo, commit, name))
        p = self.dest(repo, commit, name); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f: f.write(self.payload[(repo, name)])
        return p

    def install(self, **kw):
        rc = self.W.install_weights(self.dir, fetch=kw.pop("fetch", self.fetch), pins=kw.pop("pins", self.pins), log=self.lines.append, **kw)
        return rc, "\n".join(self.lines)

    def test_all_fetched_pointed_and_pinned(self):
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 3/3", out)
        self.assertEqual(self.calls, [(r, self.pins["weights"][r]["snapshot_commit"], n) for r, c, n, s in self.W.pinned_files(self.pins)])
        for repo, w in self.pins["weights"].items():
            with open(os.path.join(self.W.repo_dir(self.dir, repo), "refs", "main")) as fh:
                self.assertEqual(fh.read(), w["snapshot_commit"])
            for name in w["files"]:
                self.assertTrue(os.path.isfile(self.dest(repo, w["snapshot_commit"], name)))

    def test_a_present_file_is_kept_and_checked_and_a_rerun_fetches_nothing(self):
        rc, out = self.install(); self.assertEqual(rc, 0, out)
        self.calls.clear(); self.lines.clear()
        rc, out = self.install()
        self.assertEqual(rc, 0, out); self.assertEqual(self.calls, []); self.assertIn("present", out); self.assertNotIn("refs/main of", out)

    def test_a_digest_off_the_pin_fails_by_name_and_is_left_in_place(self):
        repo = "org/ModelB"; commit = self.pins["weights"][repo]["snapshot_commit"]
        p = self.dest(repo, commit, "model.safetensors"); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(b"other bytes")
        rc, out = self.install()
        self.assertEqual(rc, 1, out)
        self.assertIn("off its pin", out); self.assertIn("models--org--ModelB", out); self.assertIn("WEIGHTS FAILED: 0 absent, 1 off their pin, of 3", out)
        self.assertTrue(os.path.isfile(p))
        with open(p, "rb") as f: self.assertEqual(f.read(), b"other bytes")

    def test_a_failed_transfer_is_relayed_and_counted(self):
        def broken(repo, commit, name):
            if name == "config.json": raise ConnectionError("network unreachable")
            return self.fetch(repo, commit, name)
        rc, out = self.install(fetch=broken)
        self.assertEqual(rc, 1, out)
        self.assertIn("org/ModelA config.json: the transfer failed (ConnectionError: network unreachable)", out)
        self.assertIn("1 absent", out); self.assertIn("1 transfer(s) failed", out)

    def test_an_existing_ref_is_repointed_and_the_old_value_named(self):
        repo = "org/ModelA"; ref = os.path.join(self.W.repo_dir(self.dir, repo), "refs", "main"); os.makedirs(os.path.dirname(ref))
        with open(ref, "w") as f: f.write("c" * 40)
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        self.assertIn(f"refs/main of {repo} -> {'a' * 12} (was {'c' * 12})", out)
        with open(ref) as f: self.assertEqual(f.read(), "a" * 40)

    def test_the_real_pin_table_names_sixteen_files_of_three_repositories(self):
        from esmfold2_opt import stack
        files = self.W.pinned_files(stack.pins())
        self.assertEqual(len(files), 16); self.assertEqual(len({r for r, *_ in files}), 3)
        self.assertTrue(all(len(c) == 40 and s > 0 for _r, c, _n, s in files))
        rels = {os.path.join("hub", "models--" + r.replace("/", "--"), "snapshots", c, n) for r, c, n, _s in files}
        self.assertEqual(rels, {rel for _repo, rel, _size in stack.pinned_weight_files(stack.pins(), None)})   # the same files the routes check


if __name__ == "__main__":
    unittest.main()
