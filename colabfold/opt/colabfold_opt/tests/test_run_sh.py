"""run.sh and configs/h100.env on a stub `python` (records every call; answers the package probes): the usage refusals (no command,
unknown mode rc 2, a --mode / COLABFOLD_OPT disagreement), `--mode exact` passed through, the default mode read from the
package, the pin check before every route, the exec line, and the config's exports (deployment parameters only: no mode, no kernel switch; the compile-cache root)."""
import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TREE = os.path.dirname(OPT)
STUB_PY = textwrap.dedent('''\
    #!/bin/bash
    # stand-in python (tests only): logs argv; answers the colabfold_opt probes (rc from STUB_IMPORT_RC, stderr from STUB_IMPORT_ERR), the DEFAULT_MODE / stack_key probes, check_pins (rc from STUB_PINS_RC)
    printf '%s\\n' "$*" >> "$STUB_LOG"
    case "$*" in
      *"modes.DEFAULT_MODE"*) echo fast; exit 0 ;;
      *"stack_key()"*) [ "${STUB_CORE_RC:-0}" = 0 ] && echo jaxstub-custub-smstub; exit "${STUB_CORE_RC:-0}" ;;
      *"find_spec('colabfold_opt')"*|*"import colabfold_opt"*|*"colabfold_opt._autoload"*) [ -n "${STUB_IMPORT_ERR:-}" ] && echo "$STUB_IMPORT_ERR" >&2; exit "${STUB_IMPORT_RC:-0}" ;;
      *check_pins.py*) exit "${STUB_PINS_RC:-0}" ;;
      *) echo "stub-exec: $*"; exit "${STUB_EXEC_RC:-0}" ;;
    esac
    ''')


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        p = os.path.join(self.bin, "python"); open(p, "w").write(STUB_PY); os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
        self.log = os.path.join(self.tmp, "calls.log")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_sh(self, *args, env=None):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "MODEL_OPT"))}
        e["PATH"] = self.bin + os.pathsep + e.get("PATH", ""); e["STUB_LOG"] = self.log
        e.update(env or {})
        r = subprocess.run(["bash", os.path.join(TREE, "run.sh"), *args], capture_output=True, text=True, env=e, cwd=self.tmp)
        calls = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        return r, calls


class TestRunSh(Base):
    def test_usage(self):
        r, _ = self.run_sh(); self.assertEqual(r.returncode, 2); self.assertIn("run.sh pred", r.stderr)
        r, _ = self.run_sh("bogus"); self.assertEqual(r.returncode, 2)
        r, _ = self.run_sh("pred", "--config", "nosuch"); self.assertEqual(r.returncode, 2); self.assertIn("no such config", r.stderr)

    def test_exact_passes_through_and_unknown_is_refused(self):
        r, calls = self.run_sh("pred", "--mode", "exact", "x.a3m", "o")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue([c for c in calls if "-m colabfold_opt pred --mode exact x.a3m o" in c], calls)   # a mode of the table: the package decides, run.sh only names it
        n = len(calls)
        r, calls = self.run_sh("check", "--mode", "turbo"); self.assertEqual(r.returncode, 2); self.assertIn("'turbo' is not a mode (off|exact|fast|big", r.stderr)
        self.assertFalse([c for c in calls[n:] if "-m colabfold_opt" in c])                                                 # refused before the package is called

    def test_disagreement_and_off_restrictions(self):
        r, _ = self.run_sh("check", "--mode", "fast", env={"COLABFOLD_OPT": "off"})
        self.assertEqual(r.returncode, 2); self.assertIn("disagrees with COLABFOLD_OPT=off", r.stderr)
        r, calls = self.run_sh("check", "--mode", "off"); self.assertEqual(r.returncode, 0)
        self.assertTrue(calls[-1].endswith("-m colabfold_opt check --mode off"), calls)

    def test_default_mode_from_the_package(self):
        r, calls = self.run_sh("check")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("no --mode and no COLABFOLD_OPT: the package default mode, fast (opt/colabfold_opt/modes.py DEFAULT_MODE)", r.stderr)
        self.assertTrue(calls[-1].endswith("-m colabfold_opt check --mode fast"))
        r, calls = self.run_sh("pred", "x", "o", env={"COLABFOLD_OPT": "off"})
        self.assertTrue(calls[-1].endswith("-m colabfold_opt pred --mode off x o"), calls)

    def test_pins_gate_before_every_route(self):
        r, calls = self.run_sh("check", "--mode", "off", env={"STUB_PINS_RC": "3"})
        self.assertEqual(r.returncode, 3); self.assertIn("not installed as pinned", r.stderr)
        self.assertTrue(any("check_pins.py --quiet" in c for c in calls)); self.assertFalse([c for c in calls if "-m colabfold_opt" in c])
        r, _ = self.run_sh("check", "--mode", "fast", env={"STUB_IMPORT_RC": "4"})                                      # find_spec found no colabfold_opt: the one case worded 'not installed'
        self.assertEqual(r.returncode, 3); self.assertIn("colabfold_opt is not installed", r.stderr)

    def test_a_startup_refusal_of_the_interpreter_passes_through_the_probe(self):
        """The kit's autoload .pth refuses at interpreter start-up under COLABFOLD_OPT=<mode> (core pin gate / hook not importable): the
        probe prints those words and exits with that rc — never 'not installed' (run.sh and the config alike)."""
        refusal = "[colabfold-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v0.5.8 at x, installed v0.4.4 at y"
        r, calls = self.run_sh("check", "--mode", "fast", env={"STUB_IMPORT_RC": "3", "STUB_IMPORT_ERR": refusal})
        self.assertEqual(r.returncode, 3); self.assertIn(refusal, r.stderr); self.assertNotIn("not installed", r.stderr)
        self.assertFalse([c for c in calls if "-m colabfold_opt" in c])                                                # nothing launched
        r, _ = self.run_sh("check", "--mode", "fast", env={"STUB_IMPORT_RC": "5", "STUB_IMPORT_ERR": "some other start-up failure"})
        self.assertEqual(r.returncode, 5); self.assertIn("some other start-up failure", r.stderr); self.assertNotIn("not installed", r.stderr)
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "MODEL_OPT"))}
        e["PATH"] = self.bin + os.pathsep + e.get("PATH", ""); e["STUB_LOG"] = self.log; e.update(STUB_IMPORT_RC="3", STUB_IMPORT_ERR=refusal)
        r = subprocess.run(["bash", "-c", "source configs/h100.env; echo rc=$?; echo n=$(env | grep -c '^MODEL_OPT_STACK_KEY=')"], capture_output=True, text=True, env=e, cwd=TREE)
        self.assertIn("rc=3", r.stdout); self.assertIn("n=0", r.stdout); self.assertIn(refusal, r.stderr); self.assertNotIn("not installed", r.stderr)

    def test_passthrough_order(self):
        r, calls = self.run_sh("pred", "--config", "h100", "--mode", "fast", "in.a3m", "o", "--num-recycle", "3",
                              env={"COLABFOLD_OPT_DATA_DIR": "/data/af2_params"})                                    # the deployment names the parameters root; the config requires it
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(calls[-1].endswith("-m colabfold_opt pred --mode fast in.a3m o --num-recycle 3"), calls[-1])


class TestConfig(Base):
    def test_h100_env_exports(self):
        probe = "source configs/h100.env && env | grep -E '^(MODEL_OPT|COLABFOLD_OPT|AF_PALLAS|PYTHONDONT)' | sort"
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "MODEL_OPT"))}
        e["PATH"] = self.bin + os.pathsep + e.get("PATH", ""); e["STUB_LOG"] = self.log
        r = subprocess.run(["bash", "-c", "export COLABFOLD_OPT_DATA_DIR=/data/af2_params; " + probe], capture_output=True, text=True, env=e, cwd=TREE)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = dict(ln.split("=", 1) for ln in r.stdout.splitlines())
        self.assertEqual(got, {"MODEL_OPT": TREE, "MODEL_OPT_TARGET_GPU": "H100", "COLABFOLD_OPT_DATA_DIR": "/data/af2_params",
                               "MODEL_OPT_STACK_KEY": "jaxstub-custub-smstub", "PYTHONDONTWRITEBYTECODE": "1",
                               "COLABFOLD_OPT_JIT_ROOT": os.path.join(e.get("HOME", "/root"), ".cache", "colabfold_opt", "jit")})   # the XLA_CACHE lever's root (box-local default)
        self.assertNotIn("COLABFOLD_OPT", got); self.assertNotIn("COLABFOLD_OPT_SIZE_RULE", got)                        # no mode, no opt-in, no lever in the config
        r = subprocess.run(["bash", "-c", "source configs/h100.env; echo rc=$?; echo n=$(env | grep -cE '^(COLABFOLD_OPT_DATA_DIR|MODEL_OPT_STACK_KEY|COLABFOLD_OPT_JIT_ROOT)=')"], capture_output=True, text=True, env=e, cwd=TREE)
        self.assertIn("rc=2", r.stdout); self.assertIn("n=0", r.stdout)                                                # the parameters root has no default: unset, the config refuses by name and exports nothing more
        self.assertIn("COLABFOLD_OPT_DATA_DIR is not set", r.stderr); self.assertNotRegex(r.stderr, r"=/\S*weights")
        r = subprocess.run(["bash", "-c", "export COLABFOLD_OPT_DATA_DIR=/w MODEL_OPT_TARGET_GPU=A100; " + probe], capture_output=True, text=True, env=e, cwd=TREE)
        got = dict(ln.split("=", 1) for ln in r.stdout.splitlines())
        self.assertEqual((got["COLABFOLD_OPT_DATA_DIR"], got["MODEL_OPT_TARGET_GPU"]), ("/w", "A100"))            # pre-set values win
        e["STUB_IMPORT_RC"] = "4"                                                                                       # find_spec found no colabfold_opt
        r = subprocess.run(["bash", "-c", "source configs/h100.env"], capture_output=True, text=True, env=e, cwd=TREE)
        self.assertEqual(r.returncode, 2); self.assertIn("NOT ACTIVE: colabfold_opt is not installed", r.stderr)

    def test_configs(self):
        self.assertEqual(sorted(os.listdir(os.path.join(TREE, "configs"))), ["a100.env", "h100.env", "h200.env"])

    def test_a100_env_is_h100_env_for_compute_capability_80(self):
        """configs/a100.env exports the same names as configs/h100.env; only the target GPU word differs (A100), and its text names a100 where h100.env names h100."""
        probe = "source configs/{cfg}.env && env | grep -E '^(MODEL_OPT|COLABFOLD_OPT|AF_PALLAS|PYTHONDONT)' | sort"
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "MODEL_OPT"))}
        e["PATH"] = self.bin + os.pathsep + e.get("PATH", ""); e["STUB_LOG"] = self.log
        got = {}
        for cfg in ("h100", "a100"):
            r = subprocess.run(["bash", "-c", "export COLABFOLD_OPT_DATA_DIR=/data/af2_params; " + probe.format(cfg=cfg)], capture_output=True, text=True, env=e, cwd=TREE)
            self.assertEqual(r.returncode, 0, r.stderr)
            got[cfg] = dict(ln.split("=", 1) for ln in r.stdout.splitlines())
        self.assertEqual((got["h100"].pop("MODEL_OPT_TARGET_GPU"), got["a100"].pop("MODEL_OPT_TARGET_GPU")), ("H100", "A100"))
        self.assertEqual(got["h100"], got["a100"])
        code = lambda cfg: [ln.split("#")[0].rstrip() for ln in open(os.path.join(TREE, "configs", cfg + ".env")) if not ln.startswith("#")]
        self.assertEqual([l.replace("a100", "h100").replace("A100", "H100") for l in code("a100")], code("h100"))   # the statements are h100.env's with the two words swapped

    def test_config_propagates_a_refused_core_probe(self):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "MODEL_OPT"))}
        e["PATH"] = self.bin + os.pathsep + e.get("PATH", ""); e["STUB_LOG"] = self.log; e["STUB_CORE_RC"] = "3"     # the stack-key probe refused (an absent shared core): rc 3 propagated, nothing more exported
        r = subprocess.run(["bash", "-c", "source configs/h100.env; echo rc=$?; echo n=$(env | grep -c '^MODEL_OPT_STACK_KEY=')"], capture_output=True, text=True, env=e, cwd=TREE)
        self.assertIn("rc=3", r.stdout); self.assertIn("n=0", r.stdout); self.assertIn("MODEL_OPT_STACK_KEY could not be read", r.stderr)


class TestCoreAbsentThroughRunSh(unittest.TestCase):
    """The REAL interpreter behind `python` on PATH (started without its site directories), the package importable, the shared core ABSENT:
    `bash run.sh check|pred` and `source configs/h100.env` refuse by name through the core pin gate — `NOT ACTIVE: reason=core_missing:opt_core
    (pinned >= v… at …; nothing importable as opt_core on sys.path)`, rc 3 — before the mode, the pins or the command line are reached."""

    def setUp(self):
        import sys
        self.tmp = tempfile.mkdtemp()
        b = os.path.join(self.tmp, "bin"); os.makedirs(b)
        py = os.path.join(b, "python")                                                        # this interpreter WITHOUT its site directories: nothing importable as opt_core
        open(py, "w").write(f"#!/bin/sh\nexec {sys.executable} -S \"$@\"\n"); os.chmod(py, 0o755)
        self.bin = b

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def env(self):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "MODEL_OPT", "PYTHONPATH"))}
        e.update(PATH=self.bin + os.pathsep + e.get("PATH", ""), PYTHONPATH=OPT)
        return e

    def test_check_and_pred_refused_by_name_rc_3(self):
        for args in (["check", "--mode", "fast"], ["check", "--mode", "big", "--n_gpu", "2"], ["pred", "--mode", "exact", "x.a3m", "o"], ["check"],
                     ["check", "--config", "h100", "--mode", "fast"]):
            r = subprocess.run(["bash", os.path.join(TREE, "run.sh"), *args], capture_output=True, text=True, env=self.env(), cwd=self.tmp)
            self.assertEqual(r.returncode, 3, (args, r.stderr))
            self.assertIn("[colabfold-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v", r.stderr, args); self.assertIn("nothing importable as opt_core on sys.path)", r.stderr, args)
            self.assertNotIn("Traceback", r.stderr, args); self.assertFalse(os.path.exists(os.path.join(self.tmp, "o")))

    def test_config_alone_refused_by_name_rc_3(self):
        r = subprocess.run(["bash", "-c", f"source {os.path.join(TREE, 'configs', 'h100.env')}; echo rc=$?; echo n=$(env | grep -c '^MODEL_OPT_STACK_KEY=')"],
                           capture_output=True, text=True, env=self.env(), cwd=self.tmp)
        self.assertIn("rc=3", r.stdout); self.assertIn("n=0", r.stdout)
        self.assertIn("[colabfold-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v", r.stderr); self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
