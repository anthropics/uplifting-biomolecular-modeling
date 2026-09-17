"""`run.sh install [--config h100] [--variant p2] [--weights DIR] [--checkpoint_dir D]` — the verb's argument handling and call sequence: a stub `python`
on PATH records every invocation (nothing is installed, fetched or converted). The sequence is pip (skipped when the probe says both packages come from this
tree) → the config, when named → the pin check → the parameters step when a root is named (--weights DIR, or AF3_JAX_PARAMS_ROOT). CPU only."""
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/, configs/
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG (+ the parameters root the call saw); exit codes per call kind from STUB_RC_PROBE / _PIP / _PINS / _WEIGHTS
printf '%s%s\\n' "$(printf '%s' "$*" | tr '\\n' ' ')" "${AF3_JAX_PARAMS_ROOT:+ [AF3_JAX_PARAMS_ROOT=$AF3_JAX_PARAMS_ROOT]}" >> "$STUB_LOG"   # one line per invocation (the config's second call spans lines)
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"-c import af3_jax_opt"*) exit 0 ;;
  *"config_exports"*) echo "export AF3_JAX_PY=\\${AF3_JAX_PY:-/stub/venv/bin/python}"; exit 0 ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m af3_jax_opt.convert"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""
PROBE_PREFIX = "-I -c import os,sys"
CONFIG_CALLS = ("-c import af3_jax_opt", "config_exports")


class InstallVerb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="af3_jax_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        self.stub = os.path.join(self.bin, "python")
        with open(self.stub, "w") as f: f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def read_log(self):
        if not os.path.exists(self.log): return []
        with open(self.log) as fh: return fh.read().splitlines()

    def run_sh(self, args, env=None, **rc):
        e = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        e.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()}); e.update(env or {})
        if os.path.exists(self.log): os.remove(self.log)
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=e, cwd=self.tmp)
        lines = self.read_log()
        self.probes = [c for c in lines if c.startswith(PROBE_PREFIX)]                    # the installed-from-this-tree probe (one per install call)
        self.config_calls = [c for c in lines if any(k in c for k in CONFIG_CALLS)]      # configs/h100.env's own two calls
        return r.returncode, r.stdout + r.stderr, [c for c in lines if not c.startswith(PROBE_PREFIX) and not any(k in c for k in CONFIG_CALLS)]

    PIP = f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt"
    PINS = f"-I {TREE}/stock/check_pins.py"

    def test_install_runs_pip_then_the_pin_check_and_names_the_missing_root(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [self.PIP, self.PINS], calls)
        self.assertEqual(len(self.probes), 1); self.assertEqual(self.config_calls, [])
        self.assertIn("no parameters root named", out)

    def test_weights_dir_adds_the_parameters_step_last_with_that_root(self):
        for args in (["install", "--weights", "/data/af3_params"], ["install", "--weights=/data/af3_params"]):
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, (args, out))
            self.assertEqual(calls, [self.PIP, self.PINS, "-m af3_jax_opt.convert install [AF3_JAX_PARAMS_ROOT=/data/af3_params]"], calls)

    def test_variant_and_checkpoint_dir_pass_to_the_parameters_step(self):
        rc, out, calls = self.run_sh(["install", "--variant", "p2", "--weights", "/w", "--checkpoint_dir", "/ck"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls[-1], "-m af3_jax_opt.convert install --variant p2 --checkpoint_dir /ck [AF3_JAX_PARAMS_ROOT=/w]", calls)

    def test_the_root_from_the_environment_runs_the_parameters_step_too(self):
        rc, out, calls = self.run_sh(["install"], env={"AF3_JAX_PARAMS_ROOT": "/env/root"})
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls[-1], "-m af3_jax_opt.convert install [AF3_JAX_PARAMS_ROOT=/env/root]", calls)
        rc, out, calls = self.run_sh(["install", "--weights", "/flag/root"], env={"AF3_JAX_PARAMS_ROOT": "/env/root"})
        self.assertEqual(calls[-1], "-m af3_jax_opt.convert install [AF3_JAX_PARAMS_ROOT=/flag/root]", calls)      # --weights DIR is the root of the step

    def test_config_is_sourced_after_pip_and_before_the_pin_check(self):
        rc, out, calls = self.run_sh(["install", "--config", "h100"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [self.PIP, self.PINS], calls)
        self.assertEqual(len(self.config_calls), 2, self.config_calls)
        order = "\n".join(self.read_log())
        self.assertLess(order.index("-m pip install"), order.index("-c import af3_jax_opt")); self.assertLess(order.index("config_exports"), order.index("check_pins.py"))
        rc, out, calls = self.run_sh(["install", "--config", "nosuch"])
        self.assertEqual(rc, 2, out); self.assertEqual(calls, []); self.assertIn("no such config: nosuch", out)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--variant", "p2"], ["install", "--weights="], ["install", "--mode", "exact"], ["bogus"]):
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls)); self.assertEqual(self.probes, [], args)
            self.assertIn("run.sh", out)

    def test_parameters_flags_without_a_root_are_refused_by_name_after_the_install(self):
        rc, out, calls = self.run_sh(["install", "--variant", "p2", "--checkpoint_dir", "/ck"])
        self.assertEqual(rc, 2, out)
        self.assertEqual(calls, [self.PIP, self.PINS], calls)
        self.assertIn("the parameters step needs its root", out)

    def test_already_installed_from_this_tree_skips_pip(self):
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [self.PINS, "-m af3_jax_opt.convert install [AF3_JAX_PARAMS_ROOT=/w]"], calls)
        self.assertIn("the pip step is skipped", out)
        self.assertEqual(len(self.probes), 1)

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual((rc, calls), (1, [self.PIP]), out)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual((rc, calls), (3, [self.PIP, self.PINS]), out); self.assertIn("refused by the pin check", out)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, len(calls)), (1, 3), out)
        rc, out, calls = self.run_sh(["install", "--variant", "p3", "--weights", "/w"], weights=2)                    # the parameters step's own refusal (argparse: unknown variant) passes through
        self.assertEqual(rc, 2, out)

    def test_no_python_on_path_is_named(self):
        os.remove(self.stub)
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 3, out); self.assertIn("no python on PATH", out); self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
