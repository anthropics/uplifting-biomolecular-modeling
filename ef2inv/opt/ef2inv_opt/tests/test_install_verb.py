"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (stub `python` / `uv` executables on PATH record every
invocation; nothing is installed), and the weights step (ef2inv_opt.weights.fetch with an injected downloader over a synthetic pin table laid out
as a Hugging Face cache: the fetch-only-what-is-missing rule, refs/main pointing, the sha256 gate through stock/check_pins.py weights_gate).
No network, no GPU, no torch."""
import hashlib
import io
import os
import stat
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")

STUB_PYTHON = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes per call kind by STUB_RC_PROBE / STUB_RC_HASPIP / STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS
printf 'python %s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip --version"*) exit "${STUB_RC_HASPIP:-0}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m ef2inv_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""
STUB_UV = """#!/bin/bash
printf 'uv %s\\n' "$*" >> "$STUB_LOG"
exit "${STUB_RC_UV:-0}"
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ef2inv_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        for name, text in (("python", STUB_PYTHON), ("uv", STUB_UV)):
            p = os.path.join(self.bin, name)
            with open(p, "w") as f: f.write(text)
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, args, with_uv=True, **rc):
        if not with_uv and os.path.exists(os.path.join(self.bin, "uv")): os.remove(os.path.join(self.bin, "uv"))
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        lines = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        self.probes = [c for c in lines if c.startswith("python -I -c import os,sys") or c == "python -m pip --version"]   # the installed-from-this-tree probe and the pip probe
        return r.returncode, r.stdout + r.stderr, [c for c in lines if c not in self.probes]

    def test_install_runs_pip_then_the_software_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"python -m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"python -I {TREE}/stock/check_pins.py --checks software"], calls)

    def test_no_pip_module_installs_with_uv_into_the_python_on_path(self):
        """The pinned stack's environment carries no pip (environment/Dockerfile: uv makes it): the verb says so and installs with `uv pip --python <that python>`."""
        rc, out, calls = self.run_sh(["install"], haspip=1)
        self.assertEqual(rc, 0, out); self.assertIn("carries no pip module — installing with uv pip", out)
        self.assertEqual(calls, [f"uv pip install --python {self.bin}/python -e {TREE}/../common/opt_core -e {TREE}/opt", f"python -I {TREE}/stock/check_pins.py --checks software"], calls)
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install"], haspip=1, uv=1)                  # uv's own failure is the install's failure (1)
        self.assertEqual((rc, len(calls)), (1, 1), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install"], with_uv=False, haspip=1)         # no pip and no uv: refused by name, nothing attempted
        self.assertEqual((rc, calls), (3, []), (out, calls)); self.assertIn("no pip module and uv is not on PATH", out)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/hf"], ["install", "--weights=/data/hf"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["python -m ef2inv_opt.weights /data/hf"], calls)
            self.assertEqual(len(calls), 3, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--json"], ["install", "--weights="], ["install", "--bogus"],
                     ["install", "extra"], ["install", "--mode", "exact"], ["install", "--config", "h100"]):
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

    def test_a_tree_already_installed_skips_the_installer_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run the installer. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"python -I {TREE}/stock/check_pins.py --checks software", "python -m ef2inv_opt.weights /w"], calls)   # pin check, weights — no installer
        self.assertEqual([p for p in self.probes if "import os,sys" in p].__len__(), 1, self.probes)


class WeightsStep(unittest.TestCase):
    """ef2inv_opt.weights.fetch over a synthetic two-repository pin table: fetch only what is missing, point refs/main, then the sha256 gate."""

    def setUp(self):
        from ef2inv_opt import weights
        self.W = weights; self.cp = weights.check_pins_module(TREE)
        self.dir = tempfile.mkdtemp(prefix="ef2inv_weights_")
        self.payload = {"org/model-a": {"config.json": b'{"a": 1}\n', "model.safetensors": b"\x00" * 64},
                        "org/model-b": {"config.json": b'{"b": 2}\n', "model-00001-of-00002.safetensors": b"\x01" * 32, "model-00002-of-00002.safetensors": b"\x02" * 32}}
        self.table = {repo: {"snapshot_commit": hashlib.sha1(repo.encode()).hexdigest(), "refs_main": hashlib.sha1(repo.encode()).hexdigest(),
                             "files": {rel: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)} for rel, data in files.items()}}
                      for repo, files in self.payload.items()}
        self.calls = []

    def fetcher(self, repo, commit):
        """Stands in for huggingface_hub.snapshot_download: lays the repository's files out under hub/models--org--x/snapshots/<commit>/ (no refs)."""
        self.calls.append((repo, commit))
        snap = self.cp.snapshot_dir(self.dir, repo, {"snapshot_commit": commit})
        for rel, data in self.payload[repo].items():
            p = os.path.join(snap, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f: f.write(data)
        return snap

    def run_fetch(self, fetcher=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = self.W.fetch(self.dir, fetcher=fetcher or self.fetcher, cp=self.cp, table=self.table)
        return rc, out.getvalue() + err.getvalue()

    def test_fetches_every_repository_then_passes_the_gate(self):
        rc, out = self.run_fetch()
        self.assertEqual(rc, 0, out); self.assertIn("WEIGHTS OK: 2/2 snapshots, 5 files", out); self.assertIn(f"export HF_HOME={self.dir}", out)
        self.assertEqual(self.calls, [(r, self.table[r]["snapshot_commit"]) for r in self.table])
        for repo, rec in self.table.items():                                     # refs/main points at the pin: from_pretrained(repo) resolves offline
            ref = os.path.join(self.cp.repo_dir(self.dir, repo), "refs", "main")
            self.assertEqual(open(ref).read().strip(), rec["snapshot_commit"])
        self.assertEqual(os.environ.get("HF_HOME"), self.dir)

    def test_present_snapshots_are_kept_and_not_fetched(self):
        self.run_fetch(); self.calls.clear()
        rc, out = self.run_fetch()
        self.assertEqual((rc, self.calls), (0, []), out); self.assertEqual(out.count("kept and checked, no fetch"), 2, out)

    def test_a_differing_file_fails_by_name_and_stays(self):
        self.run_fetch()
        p = os.path.join(self.cp.snapshot_dir(self.dir, "org/model-a", self.table["org/model-a"]), "model.safetensors")
        with open(p, "wb") as f: f.write(b"\xff" * 64)                          # same size, other bytes: only the digest sees it
        self.calls.clear()
        rc, out = self.run_fetch()
        self.assertEqual(rc, 1, out); self.assertEqual(self.calls, [], "a complete snapshot is not re-fetched")
        self.assertIn("REFUSED org/model-a: unpinned", out); self.assertIn("model.safetensors: sha256", out); self.assertTrue(os.path.isfile(p))
        self.assertIn("WEIGHTS FAILED: 1/2", out)

    def test_a_fetch_that_raises_is_named_and_the_gate_refuses_the_missing_snapshot(self):
        def broken(repo, commit):
            if repo == "org/model-b": raise OSError("network unreachable")
            return self.fetcher(repo, commit)
        rc, out = self.run_fetch(fetcher=broken)
        self.assertEqual(rc, 1, out); self.assertIn("FAILED fetching org/model-b", out); self.assertIn("network unreachable", out)
        self.assertIn("REFUSED org/model-b: missing", out)

    def test_refs_main_naming_another_revision_is_repointed_and_said(self):
        repo = "org/model-a"; ref = os.path.join(self.cp.repo_dir(self.dir, repo), "refs", "main")
        os.makedirs(os.path.dirname(ref)); open(ref, "w").write("0" * 40)
        rc, out = self.run_fetch()
        self.assertEqual(rc, 0, out); self.assertIn(f"{repo}: refs/main was 000000000000, now {self.table[repo]['snapshot_commit'][:12]}", out)
        self.assertEqual(open(ref).read().strip(), self.table[repo]["snapshot_commit"])

    def test_usage(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(self.W.main([]), 2); self.assertEqual(self.W.main(["--help"]), 2); self.assertEqual(self.W.main(["a", "b"]), 2)


if __name__ == "__main__":
    unittest.main()
