"""`run.sh install [--make-venv] [--weights DIR] [--stock-python P] [--opt-python P]` — the verb's argument handling and call sequence (stub
interpreters stand in for the stock and the patched python and record every invocation; nothing is installed), and the weights route's digest
gate (rosettafold3_opt.weights.fetch with an injected downloader and pin table; no network, no upstream). CPU only."""
import hashlib
import io
import os
import shutil
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/, configs/
RUN_SH = os.path.join(TREE, "run.sh")
CKPT = "rf3_foundry_01_24_latest_remapped.ckpt"

STUB = """#!/bin/bash
# stub interpreter <role>: one line per invocation in $STUB_LOG ("<role>: <argv>" plus the PYTHONPATH head when set); exit codes per call kind from
# STUB_RC_<KIND>_<ROLE> (else STUB_RC_<KIND>, else 0). On `-m rosettafold3_opt install ... --make-venv` it creates the --opt-python it is given
# as a copy of the opt stub template ($STUB_OPT_TEMPLATE), the way the package's --make-venv creates opt/venv/bin/python.
role=__ROLE__
printf '%s: %s%s\\n' "$role" "$*" "${PYTHONPATH:+ [PYTHONPATH=${PYTHONPATH%%:*}]}" >> "$STUB_LOG"
rc() { local k="STUB_RC_$1_${role^^}"; local g="STUB_RC_$1"; echo "${!k:-${!g:-0}}"; }
case "$*" in
  *"-m pip install"*"common/opt_core"*) exit "$(rc PIPCORE)" ;;
  *"-m pip install"*) exit "$(rc PIPKIT)" ;;
  *"-m rosettafold3_opt install"*)
     if [[ "$*" == *"--make-venv"* ]]; then
       opt=$(echo "$*" | sed -n 's/.*--opt-python \\([^ ]*\\).*/\\1/p'); mkdir -p "$(dirname "$opt")"; cp "$STUB_OPT_TEMPLATE" "$opt"; chmod +x "$opt"
     fi
     exit "$(rc INSTALL)" ;;
  *"check_pins.py"*) exit "$(rc PINS)" ;;
  *"-m rosettafold3_opt.weights"*) exit "$(rc WEIGHTS)" ;;
  *) exit 0 ;;
esac
"""


def _write_stub(path, role):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(STUB.replace("__ROLE__", role))
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class InstallVerbArguments(unittest.TestCase):
    """run.sh install drives two interpreters: every pip / package / pin / weights call goes to the PATCHED one; the stock one gets the pin check
    (and, with --make-venv on a tree without opt/venv, the package's own install command run from the tree)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rf3_install_verb_")
        self.stock = _write_stub(os.path.join(self.tmp, "stockenv", "bin", "python"), "stock")
        self.opt_template = _write_stub(os.path.join(self.tmp, "opt_template_python"), "opt")
        self.opt = os.path.join(self.tmp, "optenv", "bin", "python")            # created per test (present) or left absent (the --make-venv cases)
        self.log = os.path.join(self.tmp, "calls.log")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_opt(self):
        shutil.copy(self.opt_template, _write_stub(self.opt, "opt"))
        return self.opt

    def run_sh(self, args, run_sh=RUN_SH, extra_env=None, **rc):
        env = {"PATH": "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log, "STUB_OPT_TEMPLATE": self.opt_template}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        env.update(extra_env or {})
        r = subprocess.run(["bash", run_sh, "install"] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        calls = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        return r, calls

    @staticmethod
    def kinds(calls):
        out = []
        for c in calls:
            role, argv = c.split(": ", 1)
            kind = ("pipcore" if "-m pip install" in argv and "common/opt_core" in argv else
                    "pipkit" if "-m pip install" in argv else
                    "install" if "-m rosettafold3_opt install" in argv else
                    "pins" if "check_pins.py" in argv else
                    "weights" if "-m rosettafold3_opt.weights" in argv else "other")
            out.append(f"{role}:{kind}")
        return out

    def test_plain_install_on_an_existing_patched_interpreter(self):
        opt = self.make_opt()
        r, calls = self.run_sh(["--stock-python", self.stock, "--opt-python", opt])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.kinds(calls), ["opt:pipcore", "opt:pipkit", "opt:pins", "stock:pins", "opt:install"])
        self.assertIn("--no-deps -e " + os.path.join(TREE, "..", "common", "opt_core"), calls[0])   # the shared core beside the kit, editable
        self.assertTrue(calls[1].endswith("-e " + os.path.join(TREE, "opt")), calls[1])                                # then the kit, editable
        self.assertIn("-I " + os.path.join(TREE, "stock", "check_pins.py"), calls[2])                                   # the pin check, isolated mode, on both
        self.assertIn(f"-m rosettafold3_opt install --stock-python {self.stock} --opt-python {opt}", calls[4])        # the package's own install command, both interpreters named
        self.assertIn("DONE stock=" + self.stock, r.stdout); self.assertIn("opt=" + opt, r.stdout)
        self.assertIn("ROSETTAFOLD3_OPT_CKPT=<dir>/" + CKPT, r.stdout); self.assertIn("run.sh check --config h100", r.stdout)

    def test_interpreters_from_the_environment_and_flags_passed_on(self):
        opt = self.make_opt()
        r, calls = self.run_sh(["--json", "--log", "/tmp/x.log"], extra_env={"ROSETTAFOLD3_OPT_STOCK_PYTHON": self.stock, "ROSETTAFOLD3_OPT_PYTHON": opt})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.kinds(calls), ["opt:pipcore", "opt:pipkit", "opt:pins", "stock:pins", "opt:install"])
        self.assertIn(f"--stock-python {self.stock} --opt-python {opt} --json --log /tmp/x.log", calls[4])            # the package's other flags, verbatim, after the interpreters

    def test_weights_dir_runs_the_weights_step_last_in_both_spellings(self):
        opt = self.make_opt()
        for args in (["--weights", "/w/rf3"], ["--weights=/w/rf3"]):
            if os.path.exists(self.log): os.unlink(self.log)
            r, calls = self.run_sh(["--stock-python", self.stock, "--opt-python", opt] + args)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(self.kinds(calls), ["opt:pipcore", "opt:pipkit", "opt:pins", "stock:pins", "opt:install", "opt:weights"])
            self.assertTrue(calls[-1].endswith("-m rosettafold3_opt.weights /w/rf3"), calls[-1])
            self.assertIn(f"ROSETTAFOLD3_OPT_CKPT=/w/rf3/{CKPT}", r.stdout)
            self.assertNotIn("--weights", calls[4])                                                                     # --weights is run.sh's, never the package's

    def test_no_patched_interpreter_without_make_venv_is_refused_by_name(self):
        r, calls = self.run_sh(["--stock-python", self.stock, "--opt-python", self.opt])
        self.assertEqual(r.returncode, 3); self.assertEqual(calls, [])
        self.assertIn("no patched interpreter at " + self.opt, r.stderr); self.assertIn("--make-venv", r.stderr)

    def test_make_venv_derives_the_patched_interpreter_from_the_stock_one_first(self):
        r, calls = self.run_sh(["--make-venv", "--stock-python", self.stock, "--opt-python", self.opt])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.kinds(calls), ["stock:install", "opt:pipcore", "opt:pipkit", "opt:pins", "stock:pins", "opt:install"])
        self.assertIn(f"-m rosettafold3_opt install --stock-python {self.stock} --opt-python {self.opt} --make-venv", calls[0])
        self.assertIn("[PYTHONPATH=" + os.path.join(TREE, "opt") + "]", calls[0])                                        # run from the tree: nothing is installed anywhere yet
        self.assertIn("--make-venv", calls[5])                                                                            # the package's flag stays the package's on the second call too

    def test_no_addon_is_the_packages_flag_on_both_install_calls(self):
        r, calls = self.run_sh(["--make-venv", "--no-addon", "--stock-python", self.stock, "--opt-python", self.opt])   # the container image's install line
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.kinds(calls), ["stock:install", "opt:pipcore", "opt:pipkit", "opt:pins", "stock:pins", "opt:install"])
        self.assertIn("--make-venv --no-addon", calls[0]); self.assertIn("--make-venv --no-addon", calls[5])              # passed on verbatim, never consumed by run.sh
        self.assertIn("DONE stock=" + self.stock, r.stdout)

    def test_no_stock_interpreter_is_refused_by_name(self):
        opt = self.make_opt()
        r, calls = self.run_sh(["--stock-python", os.path.join(self.tmp, "nope", "python"), "--opt-python", opt])
        self.assertEqual(r.returncode, 3); self.assertEqual(calls, []); self.assertIn("no stock interpreter", r.stderr)

    def test_failures_stop_the_step_with_their_code(self):
        opt = self.make_opt()
        base = ["--stock-python", self.stock, "--opt-python", opt, "--weights", "/w"]
        r, calls = self.run_sh(base, pipcore=1)
        self.assertEqual((r.returncode, self.kinds(calls)), (1, ["opt:pipcore"]))
        os.unlink(self.log); r, calls = self.run_sh(base, pipkit=1)
        self.assertEqual((r.returncode, self.kinds(calls)), (1, ["opt:pipcore", "opt:pipkit"]))
        os.unlink(self.log); r, calls = self.run_sh(base, pins_stock=1)
        self.assertEqual((r.returncode, self.kinds(calls)), (3, ["opt:pipcore", "opt:pipkit", "opt:pins", "stock:pins"]))
        self.assertIn("not installed at the pin on " + self.stock, r.stderr)
        os.unlink(self.log); r, calls = self.run_sh(base, install=3)
        self.assertEqual((r.returncode, self.kinds(calls)), (3, ["opt:pipcore", "opt:pipkit", "opt:pins", "stock:pins", "opt:install"]))   # the package's refusal passes through
        os.unlink(self.log); r, calls = self.run_sh(base, weights=1)
        self.assertEqual((r.returncode, self.kinds(calls)[-1]), (1, "opt:weights")); self.assertNotIn("DONE", r.stdout)
        os.unlink(self.log); r, calls = self.run_sh(["--make-venv", "--stock-python", self.stock, "--opt-python", os.path.join(self.tmp, "v", "bin", "python")], install_stock=3)
        self.assertEqual((r.returncode, self.kinds(calls)), (3, ["stock:install"]))                                       # the package's refusal on the first (tree) call passes through

    def test_a_tree_without_the_shared_core_beside_it_is_refused_by_name(self):
        kit = os.path.join(self.tmp, "bundle", "rosettafold3"); os.makedirs(kit)
        shutil.copy(RUN_SH, os.path.join(kit, "run.sh"))
        opt = self.make_opt()
        r, calls = self.run_sh(["--stock-python", self.stock, "--opt-python", opt], run_sh=os.path.join(kit, "run.sh"))
        self.assertEqual(r.returncode, 3); self.assertEqual(calls, [])
        self.assertIn("the shared core is not at " + os.path.join(kit, "..", "common", "opt_core"), r.stderr)

    def test_usage_names_the_install_flags(self):
        r = subprocess.run(["bash", RUN_SH], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        for word in ("run.sh install [--make-venv] [--weights DIR]", "--stock-python", "--no-addon", "ROSETTAFOLD3_OPT_CKPT"):
            self.assertIn(word, r.stderr)


class WeightsFetchDigestGate(unittest.TestCase):
    """rosettafold3_opt.weights.fetch: a file already there is kept and checked; an absent one is fetched by the route and checked; a digest that
    is not the pin's is exit 1 with the file left in place; a failed or empty fetch is exit 1; nothing is ever fetched when the file is present."""

    def setUp(self):
        from rosettafold3_opt import weights
        self.weights = weights
        self.tmp = tempfile.mkdtemp(prefix="rf3_weights_fetch_")
        self.good = b"the pinned checkpoint bytes"
        self.pins = {"weights": {"rf3": {"filename": CKPT, "sha256": hashlib.sha256(self.good).hexdigest(), "bytes": len(self.good), "url": "https://example.invalid/" + CKPT}}}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def route_writing(self, data, rc=0, calls=None):
        def route(directory):
            (calls if calls is not None else []).append(directory)
            if data is not None:
                with open(os.path.join(directory, CKPT), "wb") as f: f.write(data)
            return rc
        return route

    def test_present_and_pinned_is_ok_without_fetching(self):
        d = os.path.join(self.tmp, "w"); os.makedirs(d)
        with open(os.path.join(d, CKPT), "wb") as f: f.write(self.good)
        calls, out = [], io.StringIO()
        rc = self.weights.fetch(d, pins=self.pins, route=self.route_writing(b"never", calls=calls), out=out)
        self.assertEqual((rc, calls), (0, []))
        self.assertIn("WEIGHTS OK: " + os.path.join(d, CKPT), out.getvalue()); self.assertIn("export ROSETTAFOLD3_OPT_CKPT=" + os.path.join(d, CKPT), out.getvalue())

    def test_absent_is_fetched_into_a_created_dir_then_checked(self):
        d = os.path.join(self.tmp, "new", "dir"); calls, out = [], io.StringIO()
        rc = self.weights.fetch(d, pins=self.pins, route=self.route_writing(self.good, calls=calls), out=out)
        self.assertEqual((rc, calls), (0, [d])); self.assertTrue(os.path.isfile(os.path.join(d, CKPT)))
        self.assertIn("fetching " + CKPT, out.getvalue()); self.assertIn("foundry install rf3 --checkpoint-dir " + d, out.getvalue()); self.assertIn("WEIGHTS OK", out.getvalue())

    def test_wrong_digest_is_refused_and_left_in_place(self):
        d = os.path.join(self.tmp, "w"); out = io.StringIO()
        rc = self.weights.fetch(d, pins=self.pins, route=self.route_writing(b"some other checkpoint"), out=out)
        self.assertEqual(rc, 1); self.assertTrue(os.path.isfile(os.path.join(d, CKPT)))
        self.assertIn("REFUSED: " + os.path.join(d, CKPT), out.getvalue()); self.assertIn("left in place", out.getvalue()); self.assertNotIn("WEIGHTS OK", out.getvalue())

    def test_failed_or_empty_fetch_is_exit_1(self):
        out = io.StringIO()
        self.assertEqual(self.weights.fetch(os.path.join(self.tmp, "a"), pins=self.pins, route=self.route_writing(None, rc=7), out=out), 1)
        self.assertIn("downloader exited 7", out.getvalue())
        out = io.StringIO()
        self.assertEqual(self.weights.fetch(os.path.join(self.tmp, "b"), pins=self.pins, route=self.route_writing(None, rc=0), out=out), 1)
        self.assertIn("is not at " + os.path.join(self.tmp, "b", CKPT), out.getvalue())

    def test_a_pin_table_without_digest_is_exit_1_and_usage_is_2(self):
        out = io.StringIO()
        self.assertEqual(self.weights.fetch(self.tmp, pins={"weights": {"rf3": {"filename": CKPT}}}, route=self.route_writing(self.good), out=out), 1)
        self.assertEqual(self.weights.main([]), 2); self.assertEqual(self.weights.main(["--dir", "x"]), 2)

    def test_the_default_route_is_upstreams_downloader_on_this_interpreter(self):
        import sys
        seen = {}
        real = self.weights.subprocess.run
        class R: returncode = 0
        def fake(cmd, *a, **k): seen["cmd"] = cmd; return R()
        self.weights.subprocess.run = fake
        try:
            self.assertEqual(self.weights.upstream_fetch("/w/rf3"), 0)
        finally:
            self.weights.subprocess.run = real
        self.assertEqual(seen["cmd"], [sys.executable, "-m", "foundry_cli.download_checkpoints", "install", "rf3", "--checkpoint-dir", "/w/rf3"])


if __name__ == "__main__":
    unittest.main()
