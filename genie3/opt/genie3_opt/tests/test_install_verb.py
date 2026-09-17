"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step (genie3_opt.weights.fetch with an injected download runner and census; no network, no upstream): upstream's
command is built from upstream's own script, present files are excluded and kept, the digest gate refuses by name. CPU only."""
import hashlib
import io
import json
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")
DOWNLOAD_SH = os.path.join(TREE, "stock", "src", "scripts", "setup", "download.sh")   # upstream's downloader (the pinned copy)

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m genie3_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="genie3_install_verb_")
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
        for args in (["install", "--weights", "/data/genie3_downloads"], ["install", "--weights=/data/genie3_downloads"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["-m genie3_opt.weights /data/genie3_downloads"], calls)
            self.assertEqual(len(calls), 3, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--n"], ["install", "--weights="], ["install", "--bogus"],
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

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py", "-m genie3_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_the_other_verbs_are_untouched_by_install_arguments(self):
        rc, out, calls = self.run_sh(["design", "--weights", "/w"])       # --weights is install's: design hands it to the package like any argument
        self.assertNotIn("install", out)


class UpstreamCommand(unittest.TestCase):
    """The download command is upstream's (scripts/setup/download.sh --weights): its repository id and pattern are read from the pinned script."""

    def setUp(self):
        from genie3_opt import weights
        self.W = weights
        self.script = open(DOWNLOAD_SH, encoding="utf-8").read()

    def test_constants_are_read_from_upstreams_script(self):
        repo, pattern = self.W.upstream_constants(self.script)
        self.assertEqual((repo, pattern), ("yeqinglin/genie3", "pretrained/**"))
        src = json.load(open(os.path.join(TREE, "stock", "PINS.json")))["weights"]["step=600000.ckpt"]["source"]   # the pin card cites the same command
        self.assertIn(f"hf download {repo} --include '{pattern}'", src)
        self.assertIn(f"{self.W.LOCAL_SUBDIR}/checkpoints/step=600000.ckpt", src)                                      # and the layout the pattern lands

    def test_command_shape(self):
        cmd = self.W.hf_command("yeqinglin/genie3", "pretrained/**", "/data/w", exclude=["pretrained/v1/config.yaml"])
        self.assertEqual(os.path.basename(cmd[0]), "hf")
        self.assertEqual(cmd[1:], ["download", "yeqinglin/genie3", "--include", "pretrained/**", "--exclude", "pretrained/v1/config.yaml", "--local-dir", "/data/w"])

    def test_a_script_of_another_shape_raises(self):
        with self.assertRaises(RuntimeError):
            self.W.upstream_constants("HF_REPO='x'\necho no weights branch\n")


class WeightsStep(unittest.TestCase):
    """genie3_opt.weights.fetch: upstream's command for what is absent, then every pinned file's sha256 against the pin — a mismatch or an absent
    file fails by name, files stay in place."""

    def setUp(self):
        from genie3_opt import stack, weights
        self.W, self.WEIGHT_FILES = weights, stack.WEIGHT_FILES
        self.dir = tempfile.mkdtemp(prefix="genie3_weights_")
        self.payload = {rel: f"bytes-of-{rel}".encode() for rel in self.WEIGHT_FILES}
        self.pins = {rel: hashlib.sha256(v).hexdigest() for rel, v in self.payload.items()}
        self.commands = []
        self.script = open(DOWNLOAD_SH, encoding="utf-8").read()

    def path(self, rel):
        return os.path.join(self.dir, self.W.LOCAL_SUBDIR, *rel.split("/"))

    def write(self, rel, data):
        os.makedirs(os.path.dirname(self.path(rel)), exist_ok=True)
        with open(self.path(rel), "wb") as f: f.write(data)

    def runner(self, cmd):
        """Stands in for `hf download`: writes every pinned file the command does not exclude (upstream's tool writes what is absent or stale)."""
        self.commands.append(cmd)
        excluded = {cmd[i + 1] for i, a in enumerate(cmd) if a == "--exclude"}
        self.assertEqual(cmd[cmd.index("--local-dir") + 1], self.dir)
        for rel in self.WEIGHT_FILES:
            if f"{self.W.LOCAL_SUBDIR}/{rel}" not in excluded: self.write(rel, self.payload[rel])
        return 0

    def census(self, d):
        """manifest.weights_record's contract over the test's pin table: files{rel: {sha256, pin_sha256, pinned} | {missing}}, missing, pinned, lines."""
        out = {"dir": d, "files": {}, "missing": [], "pinned": True, "lines": []}
        for rel in self.WEIGHT_FILES:
            p = os.path.join(d, *rel.split("/"))
            if os.path.isfile(p):
                got = hashlib.sha256(open(p, "rb").read()).hexdigest(); ok = got == self.pins[rel]
                out["files"][rel] = {"path": p, "sha256": got, "pin_sha256": self.pins[rel], "pinned": ok}
                out["lines"].append(f"weights={os.path.basename(rel)} sha256={got[:12]} {'(pinned)' if ok else 'NOT PINNED'}")
                out["pinned"] = out["pinned"] and ok
            else:
                out["files"][rel] = {"missing": True, "path": p}; out["missing"].append(p); out["pinned"] = False
        return out

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, runner=kw.pop("runner", self.runner), census=kw.pop("census", self.census), script_text=kw.pop("script_text", self.script), out=out)
        return rc, out.getvalue()

    def test_all_fetched_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 2/2", out)
        self.assertIn(f"export GENIE3_WEIGHTS={os.path.join(self.dir, self.W.LOCAL_SUBDIR)}", out)
        self.assertEqual(len(self.commands), 1, self.commands)
        self.assertEqual(self.commands[0][1:5], ["download", "yeqinglin/genie3", "--include", "pretrained/**"])
        self.assertNotIn("--exclude", self.commands[0])
        for rel in self.WEIGHT_FILES: self.assertTrue(os.path.isfile(self.path(rel)))

    def test_a_present_file_is_excluded_from_the_download_and_checked(self):
        self.write("config.yaml", self.payload["config.yaml"])
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("config.yaml: present", out)
        cmd = self.commands[0]
        self.assertEqual(cmd[cmd.index("--exclude") + 1], f"{self.W.LOCAL_SUBDIR}/config.yaml")

    def test_both_present_downloads_nothing(self):
        for rel in self.WEIGHT_FILES: self.write(rel, self.payload[rel])
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.commands, [])
        self.assertIn("nothing to download", out)

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        self.write(self.WEIGHT_FILES[0], b"corrupt")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        self.assertIn("REFUSED", out); self.assertIn(self.WEIGHT_FILES[0], out); self.assertNotIn("WEIGHTS OK", out)
        self.assertEqual(open(self.path(self.WEIGHT_FILES[0]), "rb").read(), b"corrupt")   # left in place: excluded from the download, never deleted

    def test_a_failed_download_stops_before_the_census(self):
        rc, out = self.fetch(runner=lambda cmd: 7, census=lambda d: self.fail("census consulted after a failed download"))
        self.assertEqual(rc, 1, out); self.assertIn("FAILED: the download exited 7", out)

    def test_a_download_that_leaves_a_file_absent_is_refused_by_name(self):
        rc, out = self.fetch(runner=lambda cmd: 0)                        # exits 0, writes nothing
        self.assertEqual(rc, 1, out); self.assertIn("REFUSED", out); self.assertIn("absent after the download", out)

    def test_no_hf_tool_is_named_not_worked_around(self):
        from unittest import mock
        with mock.patch.object(self.W.sysconfig, "get_path", return_value=os.path.join(self.dir, "no-such-bin")):
            rc, out = self.fetch(runner=None)                              # the real runner: the tool's absence is the failure, by name
        self.assertEqual(rc, 1, out); self.assertIn("`hf` tool is not installed", out)

    def test_usage(self):
        self.assertEqual(self.W.main([]), 2)
        self.assertEqual(self.W.main(["--weights"]), 2)


if __name__ == "__main__":
    unittest.main()
