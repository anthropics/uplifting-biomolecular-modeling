"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step (complexa_opt.weights: upstream's downloader run under a scratch project root whose ckpts/ is DIR, then the
digest gate — an injected script, route and pin table; no network, no upstream). CPU only."""
import hashlib
import io
import os
import re
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
  *"-m complexa_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""
PIP_CALL = re.compile(r"^-m pip install -e \S+/\.\./common/opt_core -e \S+/opt$")      # the core first, then the kit, both editable, from this tree
PINS_CALL = re.compile(r"^-I \S+/stock/check_pins\.py$")                                # the pin check, isolated mode, no selector: software pins


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="complexa_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        self.stub = os.path.join(self.bin, "python")
        with open(self.stub, "w") as f: f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, args, **rc):
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        if os.path.exists(self.log): os.remove(self.log)                        # one call sequence per invocation
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        calls = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        return r.returncode, calls, r.stdout + r.stderr

    def test_plain_install_probes_then_installs_core_and_kit_then_checks_pins(self):
        rc, calls, out = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(calls), 3, calls)
        self.assertIn("-I -c import os,sys", calls[0])                                  # is the kit installed from this tree already?
        self.assertRegex(calls[1], PIP_CALL)
        self.assertRegex(calls[2], PINS_CALL)

    def test_already_installed_from_this_tree_skips_pip_keeps_the_pin_check(self):
        rc, calls, out = self.run_sh(["install"], probe=0)
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(calls), 2, calls)
        self.assertRegex(calls[1], PINS_CALL)
        self.assertIn("the pip step is skipped", out)

    def test_weights_dir_runs_the_weights_step_last(self):
        rc, calls, out = self.run_sh(["install", "--weights", "/data/w"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(calls), 4, calls)
        self.assertRegex(calls[2], PINS_CALL)
        self.assertEqual(calls[3], "-m complexa_opt.weights /data/w")
        rc, calls, _ = self.run_sh(["install", "--weights=/data/w2"])
        self.assertEqual((rc, calls[-1]), (0, "-m complexa_opt.weights /data/w2"))

    def test_usage_errors_exit_2_before_anything_runs(self):
        for args in (["install", "--weights"], ["install", "--weights", "--x"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"]):
            rc, calls, out = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out)); self.assertEqual(calls, [], args); self.assertIn("usage: run.sh install [--weights DIR]", out)

    def test_failures_are_named_with_their_exit_codes(self):
        rc, calls, out = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual(rc, 1); self.assertIn("the install failed", out); self.assertEqual(len(calls), 2)          # probe, pip; no pin check, no weights
        rc, calls, out = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual(rc, 3); self.assertIn("refused by the pin check", out); self.assertEqual(len(calls), 3)    # weights step never reached
        rc, calls, out = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual(rc, 1); self.assertEqual(len(calls), 4)                                                     # the weights step's own status

    def test_no_python_on_path_is_refused_by_name(self):
        os.remove(self.stub)
        rc, calls, out = self.run_sh(["install"])
        self.assertEqual(rc, 3); self.assertIn("no python on PATH", out); self.assertEqual(calls, [])


def _write(path, data):
    with open(path, "wb") as f: f.write(data)
    return hashlib.sha256(data).hexdigest()


FAKE_DOWNLOADER = """#!/bin/bash
# stands in for upstream's env/download_startup.sh: the same location rule (project root = the parent of this script's directory, files into
# <project root>/ckpts/) and the same keep-if-present behaviour; the payload is the test's bytes, the exit status is $FAKE_RC
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
[ "$1" = "--complexa" ] || exit 9
cd "$PROJECT_ROOT"; mkdir -p ./ckpts
for f in complexa.ckpt complexa_ae.ckpt; do [ -s "./ckpts/$f" ] || printf '%s' "${FAKE_PAYLOAD:-weights-of-$f}" > "./ckpts/$f"; done
echo "fake downloader ran in $PROJECT_ROOT" ; exit "${FAKE_RC:-0}"
"""


class WeightsStep(unittest.TestCase):
    """complexa_opt.weights.fetch / upstream_route with an injected script and pin table: files land in DIR through the scratch root's ckpts
    link, the checkout is never written to, the digest gate is the verdict, nothing is deleted."""

    def setUp(self):
        from complexa_opt import stack, weights
        self.stack, self.W = stack, weights
        self.tmp = tempfile.mkdtemp(prefix="complexa_weights_")
        self.dir = os.path.join(self.tmp, "w"); os.makedirs(self.dir)
        self.checkout = os.path.join(self.tmp, "checkout"); os.makedirs(os.path.join(self.checkout, "env"))
        self.script = os.path.join(self.checkout, "env", "download_startup.sh")
        with open(self.script, "w") as f: f.write(FAKE_DOWNLOADER)
        self.payload = {n: f"weights-of-{n}".encode() for n in ("complexa.ckpt", "complexa_ae.ckpt")}
        self.files = [{"file": n, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in self.payload.items()]
        self._pins_before = stack._PINS
        stack._PINS = {"weights": {"files": self.files}}                       # the kit's own comparator (stack.weights_gate) against this table
        self.gate = lambda d: stack.weights_gate({stack.ENV_WEIGHTS: d}, with_sha=True)

    def tearDown(self):
        self.stack._PINS = self._pins_before

    def fetch(self, route, files=None):
        buf = io.StringIO()
        rc = self.W.fetch(self.dir, files=files or self.files, route=route, gate=self.gate, out=buf)
        return rc, buf.getvalue()

    def test_upstreams_script_runs_under_a_scratch_root_whose_ckpts_is_dir(self):
        route = self.W.upstream_route(self.dir, script=self.script)
        self.assertEqual(route(), 0)
        self.assertEqual(sorted(os.listdir(self.dir)), ["complexa.ckpt", "complexa_ae.ckpt"])           # upstream's lines wrote straight into DIR
        self.assertEqual(sorted(os.listdir(self.checkout)), ["env"])                                     # the checkout is never written to
        self.assertEqual(os.listdir(os.path.join(self.checkout, "env")), ["download_startup.sh"])
        leftovers = [d for d in os.listdir(tempfile.gettempdir()) if d.startswith("complexa_weights_") and os.path.isdir(os.path.join(tempfile.gettempdir(), d))
                     and os.path.islink(os.path.join(tempfile.gettempdir(), d, "ckpts"))]
        self.assertEqual(leftovers, [])                                                                   # the scratch root (link + copy) is removed

    def test_absent_files_are_fetched_by_upstreams_route_then_pass_the_digest_gate(self):
        rc, text = self.fetch(self.W.upstream_route(self.dir, script=self.script))
        self.assertEqual(rc, self.W.EXIT_OK, text)
        self.assertIn("complexa.ckpt: fetching", text); self.assertIn("exited 0; the pin check below is the verdict", text)
        self.assertIn("(pinned)", text); self.assertIn("WEIGHTS OK: 2/2", text); self.assertIn(f"export CKPT_PATH={self.dir}", text)

    def test_present_and_pinned_fetches_nothing(self):
        for n, b in self.payload.items(): _write(os.path.join(self.dir, n), b)
        called = []
        rc, text = self.fetch(lambda: called.append(1) or 0)
        self.assertEqual((rc, called), (self.W.EXIT_OK, []), text)
        self.assertIn("complexa.ckpt: present", text); self.assertNotIn("fetching", text)

    def test_the_scripts_exit_status_is_not_the_verdict_the_digest_is(self):
        os.environ["FAKE_RC"] = "7"                                             # upstream's script reports failure but the files are right: OK
        try:
            rc, text = self.fetch(self.W.upstream_route(self.dir, script=self.script))
        finally:
            del os.environ["FAKE_RC"]
        self.assertEqual(rc, self.W.EXIT_OK, text); self.assertIn("exited 7", text)

    def test_a_file_off_its_pin_is_refused_by_name_and_left_in_place(self):
        _write(os.path.join(self.dir, "complexa.ckpt"), b"not the pinned bytes!")                          # wrong size
        _write(os.path.join(self.dir, "complexa_ae.ckpt"), b"X" * len(self.payload["complexa_ae.ckpt"]))  # right size, wrong digest
        rc, text = self.fetch(lambda: 0)
        self.assertEqual(rc, self.W.EXIT_FAIL, text)
        self.assertIn("REFUSED: 2 finding(s)", text); self.assertIn("complexa.ckpt: 21 bytes, the pin is", text)
        self.assertIn("complexa_ae.ckpt: bytes do not hash to the pin", text); self.assertIn("UNPINNED", text)
        self.assertEqual(open(os.path.join(self.dir, "complexa.ckpt"), "rb").read(), b"not the pinned bytes!")   # never deleted

    def test_an_absent_file_after_the_route_ran_is_refused(self):
        rc, text = self.fetch(lambda: 0)                                        # a route that fetched nothing
        self.assertEqual(rc, self.W.EXIT_FAIL, text); self.assertIn("complexa.ckpt: absent from", text)

    def test_a_route_that_cannot_run_is_named(self):
        def broken(): raise RuntimeError("upstream's checkout cannot be located")
        rc, text = self.fetch(broken)
        self.assertEqual(rc, self.W.EXIT_FAIL); self.assertIn("FAILED fetching complexa.ckpt, complexa_ae.ckpt: RuntimeError: upstream's checkout cannot be located", text)

    def test_upstream_script_is_located_through_the_checkout_rule(self):
        os.environ[self.stack.ENV_UPSTREAM] = self.checkout                     # LOCAL_CODE_PATH names the checkout: its env/download_startup.sh
        try:
            self.assertEqual(self.W.upstream_script(), self.script)
            os.environ[self.stack.ENV_UPSTREAM] = self.tmp                      # a directory that is not upstream's checkout: named
            with self.assertRaisesRegex(RuntimeError, "is absent: .* is not upstream's checkout at the pin"):
                self.W.upstream_script()
        finally:
            del os.environ[self.stack.ENV_UPSTREAM]

    def test_usage(self):
        self.assertEqual(self.W.main([]), self.W.EXIT_USAGE); self.assertEqual(self.W.main(["--weights"]), self.W.EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()
