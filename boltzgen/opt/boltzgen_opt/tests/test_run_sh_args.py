"""run.sh's argument scan: `--config` / `--mode` are the script's own options only before the first `--`; the `--` and every token after it
reach `python -m boltzgen_opt <cmd>` untouched (upstream's `configure --config <step> k=v` words ride there), and a config named before
`--` is still sourced."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from boltzgen_opt import stack

KIT = os.path.dirname(os.path.abspath(stack.opt_home()))          # boltzgen/: run.sh and configs/ live here (opt_home() is boltzgen/opt)
SHIM = """#!/bin/bash
# `python` on PATH for run.sh: records the package launch, runs every probe (`python -c …`, the config's) on the real interpreter
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "boltzgen_opt" ]; then
  { printf 'ARGV'; printf '\\t%s' "$@"; printf '\\n'; printf 'TARGET_GPU=%s\\n' "${MODEL_OPT_TARGET_GPU:-}"; } > "$RUNSH_ARGV_OUT"; exit 0
fi
exec REAL_PYTHON "$@"
"""


class TestRunShArgumentScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_runsh_")
        bindir = os.path.join(cls.tmp, "bin")
        os.makedirs(bindir)
        shim = os.path.join(bindir, "python")
        with open(shim, "w") as fh:
            fh.write(SHIM.replace("REAL_PYTHON", os.path.abspath(sys.executable)))   # abspath, not realpath: a venv interpreter keeps its site
        os.chmod(shim, 0o755)
        cls.env = dict(os.environ, PATH=bindir + os.pathsep + os.environ.get("PATH", ""), RUNSH_ARGV_OUT=os.path.join(cls.tmp, "argv.txt"))
        for k in ("BOLTZGEN_OPT", "MODEL_OPT_TARGET_GPU", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "MODEL_OPT_JIT_ROOT", "MODEL_OPT_STACK_KEY"):
            cls.env.pop(k, None)
        cls.env["BOLTZGEN_CACHE"] = cls.tmp                              # the config's one required variable (configs/h100.env refuses by name without it)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_sh(self, *argv):
        out = self.env["RUNSH_ARGV_OUT"]
        if os.path.exists(out):
            os.remove(out)
        r = subprocess.run(["bash", os.path.join(KIT, "run.sh"), *argv], env=self.env, capture_output=True, text=True, cwd=self.tmp, timeout=120)
        rec = {}
        if os.path.exists(out):
            for ln in open(out).read().splitlines():
                if ln.startswith("ARGV\t"):
                    rec["argv"] = ln.split("\t")[1:]
                elif ln.startswith("TARGET_GPU="):
                    rec["target_gpu"] = ln.split("=", 1)[1]
        return r, rec

    def test_config_words_after_the_separator_are_upstreams(self):
        tail = ["--", "--diffusion_batch_size", "16", "--config", "design", "compile_pairformer=true", "compile_structure=true"]
        r, rec = self.run_sh("design", "--config", "h100", "--mode", "off", "s.yaml", "--output", "o", "--seed", "42", *tail)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertEqual(rec["argv"], ["-m", "boltzgen_opt", "design", "--mode", "off", "s.yaml", "--output", "o", "--seed", "42", *tail])
        self.assertEqual(rec["target_gpu"], "H100")                      # configs/h100.env was sourced: `--config h100` before `--` is the script's
        self.assertNotIn("no such config", r.stderr)

    def test_mode_and_config_after_the_separator_do_not_set_the_scripts(self):
        tail = ["--", "--mode", "x", "--config", "nosuch", "--config=alsonot"]
        r, rec = self.run_sh("design", "--mode", "off", "s.yaml", "--output", "o", *tail)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertEqual(rec["argv"], ["-m", "boltzgen_opt", "design", "--mode", "off", "s.yaml", "--output", "o", *tail])   # MODE stayed `off`; the tail verbatim
        self.assertEqual(rec["target_gpu"], "")                          # no config sourced
        self.assertNotIn("no such config", r.stderr)

    def test_an_unknown_config_before_the_separator_is_still_refused(self):
        r, rec = self.run_sh("design", "--config", "nosuch", "s.yaml", "--output", "o", "--", "--config", "design", "k=v")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no such config: nosuch", r.stderr)
        self.assertEqual(rec, {})                                          # nothing launched

    def test_the_config_refuses_by_name_without_BOLTZGEN_CACHE(self):
        env = {k: v for k, v in self.env.items() if k != "BOLTZGEN_CACHE"}
        if os.path.exists(self.env["RUNSH_ARGV_OUT"]):
            os.remove(self.env["RUNSH_ARGV_OUT"])
        r = subprocess.run(["bash", os.path.join(KIT, "run.sh"), "check", "--config", "h100", "--mode", "off"], env=env, capture_output=True, text=True, cwd=self.tmp, timeout=120)
        self.assertEqual(r.returncode, 3, r.stderr[-800:])
        self.assertEqual(r.stderr.count("NOT ACTIVE"), 1, r.stderr[-800:])
        self.assertIn("BOLTZGEN_CACHE is not set", r.stderr)
        self.assertFalse(os.path.exists(self.env["RUNSH_ARGV_OUT"]))       # nothing launched

    def test_the_a100_config_is_h100s_parameters_with_its_gpu_word(self):
        """configs/a100.env sets MODEL_OPT_TARGET_GPU=A100 (unless pre-set) and sources configs/h100.env: `--config a100` reaches the package
        like `--config h100`, keys the JIT caches the same way, and refuses by name without BOLTZGEN_CACHE (one NOT ACTIVE line, rc 3)."""
        r, rec = self.run_sh("check", "--config", "a100", "--mode", "exact")
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertEqual((rec["argv"], rec["target_gpu"]), (["-m", "boltzgen_opt", "check", "--mode", "exact"], "A100"))
        src = f"source {os.path.join(KIT, 'configs', 'a100.env')} && echo \"G=$MODEL_OPT_TARGET_GPU T=${{TRITON_CACHE_DIR:-}} X=${{TORCH_EXTENSIONS_DIR:-}} O=${{HF_HUB_OFFLINE:-}}\""
        root = os.path.join(self.tmp, "jit")
        r = subprocess.run(["bash", "-c", src], env=dict(self.env, MODEL_OPT_JIT_ROOT=root, MODEL_OPT_STACK_KEY="k1"), capture_output=True, text=True, cwd=self.tmp, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn(f"G=A100 T={root}/k1/triton X={root}/k1/torch_extensions O=1", r.stdout)
        r = subprocess.run(["bash", "-c", src], env=dict(self.env, MODEL_OPT_STACK_KEY="k1", MODEL_OPT_TARGET_GPU="A100-80GB"), capture_output=True, text=True, cwd=self.tmp, timeout=120)
        self.assertIn("G=A100-80GB T= X= O=1", r.stdout)                                    # a launcher's own GPU word is kept; no JIT root: nothing keyed
        env = {k: v for k, v in self.env.items() if k != "BOLTZGEN_CACHE"}
        r = subprocess.run(["bash", "-c", src], env=dict(env, MODEL_OPT_STACK_KEY="k1"), capture_output=True, text=True, cwd=self.tmp, timeout=120)
        self.assertEqual((r.returncode, r.stderr.count("NOT ACTIVE")), (3, 1), r.stderr[-800:])
        self.assertIn("BOLTZGEN_CACHE is not set", r.stderr)

    def test_the_jit_root_keys_the_caches_only_when_set(self):
        """MODEL_OPT_JIT_ROOT set: TRITON_CACHE_DIR / TORCH_EXTENSIONS_DIR = <root>/<stack key>/{triton,torch_extensions} unless already set;
        unset: the config exports neither (the tools' own defaults apply)."""
        src = f"source {os.path.join(KIT, 'configs', 'h100.env')} && echo \"T=${{TRITON_CACHE_DIR:-}} X=${{TORCH_EXTENSIONS_DIR:-}}\""
        root = os.path.join(self.tmp, "jit")
        r = subprocess.run(["bash", "-c", src], env=dict(self.env, MODEL_OPT_JIT_ROOT=root, MODEL_OPT_STACK_KEY="k1"), capture_output=True, text=True, cwd=self.tmp, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn(f"T={root}/k1/triton X={root}/k1/torch_extensions", r.stdout)
        r = subprocess.run(["bash", "-c", src], env=dict(self.env, MODEL_OPT_JIT_ROOT=root, MODEL_OPT_STACK_KEY="k1", TRITON_CACHE_DIR="/keep/me"), capture_output=True, text=True, cwd=self.tmp, timeout=120)
        self.assertIn(f"T=/keep/me X={root}/k1/torch_extensions", r.stdout)          # a pre-set tool variable is honored as given
        r = subprocess.run(["bash", "-c", src], env=dict(self.env, MODEL_OPT_STACK_KEY="k1"), capture_output=True, text=True, cwd=self.tmp, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("T= X=", r.stdout)                                                 # no root: nothing exported

    def test_the_separator_needs_a_command_first(self):
        r, rec = self.run_sh("--", "design")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(rec, {})

    def test_the_verbs_are_install_design_check_warm(self):
        """run.sh's one `case "$CMD"` line admits exactly install | design | check | warm; anything else is usage (exit 2, nothing launched);
        the script appends nothing of its own to the package's argv."""
        text = open(os.path.join(KIT, "run.sh"), encoding="utf-8").read()
        case = [ln for ln in text.splitlines() if ln.startswith('case "$CMD" in')]
        self.assertEqual(len(case), 1, case)
        verbs = re.findall(r"(?:^| )([a-z|]+)\) ", case[0].split(" in ", 1)[1])
        self.assertEqual(sorted(v for alt in verbs for v in alt.split("|")), ["check", "design", "install", "warm"])
        self.assertIn('python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt" ||', text)   # install: the shared core and this package, editable (then stock/check_pins.py; test_install_verb.py has the sequence)
        self.assertIn('python -I "$HERE/stock/check_pins.py" ||', text)
        self.assertNotIn("PACK_K", text); self.assertNotIn("--pack $", text)
        for word in ("serve", "selftest", "step"):                            # anything outside the four is usage
            r, rec = self.run_sh(word, "--mode", "exact")
            self.assertEqual((r.returncode, rec), (2, {}), (word, r.stderr[-300:]))   # usage, nothing launched
        r, rec = self.run_sh("check", "--mode", "exact")
        self.assertEqual((r.returncode, rec["argv"]), (0, ["-m", "boltzgen_opt", "check", "--mode", "exact"]))
        r, rec = self.run_sh("design", "--mode", "fast", "s.yaml", "--output", "o", "--seed", "1", "--num_designs", "3")
        self.assertEqual((r.returncode, rec["argv"]), (0, ["-m", "boltzgen_opt", "design", "--mode", "fast", "s.yaml", "--output", "o", "--seed", "1", "--num_designs", "3"]))   # the caller's flags, untouched, nothing appended


if __name__ == "__main__":
    unittest.main()
