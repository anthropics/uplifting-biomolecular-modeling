"""configs/h100.env and run.sh: the config fills unset deployment parameters and sets no mode or lever; run.sh's parsing rules
(mode vs RFDIFFUSION3_OPT, unknown modes, the upstream pin gate; the core pin gate on these routes: test_core_gate_routes)
— without a real GPU or the upstream package (a fake
torch+GPU stub stands in for the config's own JIT-cache-key computation, which now refuses rather than guessing; see
modes.test_jit_cache_key_raises_never_silent_unknown for that refusal path itself)."""
import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from .. import registry, stack
from . import _fixtures as fx

TREE = registry.tree_home()
_FAKE_TORCH_DIR = tempfile.mkdtemp(prefix="rfd3_test_config_fake_torch_")
atexit.register(shutil.rmtree, _FAKE_TORCH_DIR, ignore_errors=True)
fx.fake_torch(_FAKE_TORCH_DIR, cuda="13.0", sm=(9, 0))


def _bash(script, env=None, cwd=None):
    base = {k: v for k, v in os.environ.items() if k not in stack.lever_names() and k not in stack.PACKAGE_ENV}
    base["PATH"] = os.path.dirname(sys.executable) + os.pathsep + base.get("PATH", "")
    base["PYTHONPATH"] = _FAKE_TORCH_DIR + (os.pathsep + base["PYTHONPATH"] if base.get("PYTHONPATH") else "")
    if env:
        base.update(env)
    return subprocess.run(["bash", "-c", script], env=base, capture_output=True, text=True, cwd=cwd or TREE)


class TestConfig(unittest.TestCase):
    def test_config_fills_only_deployment_parameters(self):
        td = tempfile.mkdtemp(prefix="rfd3_test_config_tmp_"); self.addCleanup(shutil.rmtree, td, ignore_errors=True)
        r = _bash("source configs/h100.env && env | sort", env={"TMPDIR": td})
        self.assertEqual(r.returncode, 0, r.stderr)
        got = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
        self.assertEqual(os.path.realpath(got["MODEL_OPT"]), os.path.realpath(TREE))
        self.assertEqual(got["MODEL_OPT_TARGET_GPU"], "H100")
        self.assertNotIn("RFD3_CKPT", got)                                      # a weights path is the box's: no default (design / warm refuse by name without RFD3_CKPT or --ckpt)
        self.assertNotIn("FOUNDRY_CHECKPOINT_DIRS", got)                         # follows RFD3_CKPT only when that is set (test_config_keeps_preset_values)
        self.assertNotIn("MODEL_OPT_JIT_ROOT", got)                             # the JIT-cache root is optional and has no default ...
        self.assertRegex(got["MODEL_OPT_STACK_KEY"], r"^torch[0-9.]+-cu130-sm90$")            # the stack key is still computed (the fake torch: CUDA 13.0, sm90)
        scratch = os.path.join(td, "model_opt_jit-uid%d" % os.getuid())         # ... but the three caches never reach a shared default: unset, they are keyed under the private per-user scratch root
        self.assertEqual([got.get(v) for v in ("TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR")],
                         [f"{scratch}/{got['MODEL_OPT_STACK_KEY']}/{c}" for c in ("triton", "torch_extensions", "inductor")])
        self.assertEqual(os.stat(scratch).st_mode & 0o777, 0o700); self.assertFalse(os.path.islink(scratch)); self.assertEqual(os.lstat(scratch).st_uid, os.getuid())
        self.assertEqual(got["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(got["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")        # torch's allocator mode of the launch line (a kit-mode multi-design pass at long lengths completes only under it)
        for name in list(stack.lever_names()) + list(stack.upstream_switches()):
            self.assertNotIn(name, got, name)
        self.assertNotIn("RFDIFFUSION3_STOCK_PYTHON", got)                     # a venv path is not a deployment parameter

    def test_config_keeps_preset_values(self):
        r = _bash("source configs/h100.env && echo $RFD3_CKPT $FOUNDRY_CHECKPOINT_DIRS $MODEL_OPT_TARGET_GPU $TRITON_CACHE_DIR", env={"RFD3_CKPT": "/data/x.ckpt", "MODEL_OPT_TARGET_GPU": "H200", "TRITON_CACHE_DIR": "/tmp/tc"})
        self.assertEqual(r.stdout.split(), ["/data/x.ckpt", "/data", "H200", "/tmp/tc"])            # a set RFD3_CKPT is exported with upstream's search path following it; a pre-set cache directory whose parent exists is kept
        r = _bash("source configs/h100.env && echo $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TORCHINDUCTOR_CACHE_DIR", env={"MODEL_OPT_JIT_ROOT": "/tmp/jc"})
        key = _bash("source configs/h100.env && echo $MODEL_OPT_STACK_KEY").stdout.strip()
        self.assertEqual(r.stdout.split(), [f"/tmp/jc/{key}/triton", f"/tmp/jc/{key}/torch_extensions", f"/tmp/jc/{key}/inductor"])   # MODEL_OPT_JIT_ROOT set: all three caches keyed under it, <root>/<stack key>/<cache>
        r = _bash("source configs/h100.env && echo $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR", env={"MODEL_OPT_JIT_ROOT": "/tmp/jc", "TRITON_CACHE_DIR": "/tmp/tc"})
        self.assertEqual(r.stdout.split(), ["/tmp/tc", f"/tmp/jc/{key}/inductor"])                                                  # ... a pre-set cache directory still wins

    def test_config_keys_an_unwritable_or_unset_root_under_a_per_user_scratch_root(self):
        """A MODEL_OPT_JIT_ROOT the process cannot write is left as set (an unset one stays unset) and the three caches go under
        ${TMPDIR:-/tmp}/model_opt_jit-<uid>, made with mode 0700; a path there that is a symbolic link, open to group or others (or another
        user's) is refused by name and no cache variable is exported."""
        shim = 'mkdir() { [ "${1-}" != -p ] || [ "${2-}" != "$MODEL_OPT_JIT_ROOT" ] || return 1; command mkdir "$@"; }; '   # the root's own mkdir -p fails, as on a read-only mount (root writes through any mode)
        key = _bash("source configs/h100.env && echo $MODEL_OPT_STACK_KEY").stdout.strip()
        with tempfile.TemporaryDirectory() as td:
            scratch = os.path.join(td, "model_opt_jit-uid%d" % os.getuid())
            r = _bash(shim + 'source configs/h100.env && echo "$MODEL_OPT_JIT_ROOT ${TRITON_CACHE_DIR:-unset} ${TORCH_EXTENSIONS_DIR:-unset} ${TORCHINDUCTOR_CACHE_DIR:-unset}"', env={"MODEL_OPT_JIT_ROOT": "/ro/root", "TMPDIR": td})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.split(), ["/ro/root", f"{scratch}/{key}/triton", f"{scratch}/{key}/torch_extensions", f"{scratch}/{key}/inductor"])
            self.assertEqual(os.stat(scratch).st_mode & 0o777, 0o700)
            self.assertFalse(os.path.islink(scratch))
            os.rmdir(scratch); bait = os.path.join(td, "bait"); os.mkdir(bait); os.symlink(bait, scratch)
            r = _bash(shim + 'source configs/h100.env && echo "$MODEL_OPT_JIT_ROOT ${TRITON_CACHE_DIR:-unset} ${TORCH_EXTENSIONS_DIR:-unset} ${TORCHINDUCTOR_CACHE_DIR:-unset}"', env={"MODEL_OPT_JIT_ROOT": "/ro/root", "TMPDIR": td})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.split(), ["/ro/root", "unset", "unset", "unset"])
            self.assertIn(f"{scratch} refused", r.stderr)
            self.assertEqual(os.listdir(bait), [])
            # no root at all and the per-user path planted (a link here; another owner or an open mode alike): refused by name, nothing exported, nothing behind the link
            r = _bash('source configs/h100.env && echo "${MODEL_OPT_JIT_ROOT:-unset} ${TRITON_CACHE_DIR:-unset} ${TORCH_EXTENSIONS_DIR:-unset} ${TORCHINDUCTOR_CACHE_DIR:-unset}"', env={"TMPDIR": td})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.split(), ["unset", "unset", "unset", "unset"])
            self.assertIn(f"{scratch} refused", r.stderr)
            self.assertEqual(os.listdir(bait), [])
            os.remove(scratch); os.mkdir(scratch); os.chmod(scratch, 0o770)
            r = _bash('source configs/h100.env && echo "${TRITON_CACHE_DIR:-unset}"', env={"TMPDIR": td})
            self.assertEqual((r.returncode, r.stdout.split()), (0, ["unset"])); self.assertIn(f"{scratch} refused", r.stderr)

    def test_config_refuses_without_the_package(self):
        with tempfile.TemporaryDirectory() as td:
            py = fx.make_venv(td, os.path.join(td, "empty"))
            os.makedirs(os.path.join(td, "empty"), exist_ok=True)
            # a venv whose .pth points at nothing of the package: `python -c "import rfdiffusion3_opt"` fails there
            import glob
            pth = glob.glob(os.path.join(td, "venv", "lib", "python*", "site-packages", "fake_stock.pth"))[0]
            open(pth, "w").write(os.path.join(td, "empty") + "\n")
            r = subprocess.run(["bash", "-c", "source configs/h100.env; echo rc=$?"], env={"PATH": os.path.dirname(py) + os.pathsep + os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp")},
                               capture_output=True, text=True, cwd=TREE)
            self.assertIn("rc=3", r.stdout)                                   # not importable: rc 3, before the core pin gate could even run
            self.assertIn("configs/h100.env: rfdiffusion3_opt is not importable", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


class TestRunSh(unittest.TestCase):
    def test_usage_and_mode_rules(self):
        r = _bash("./run.sh"); self.assertEqual(r.returncode, 2)
        header = open(os.path.join(TREE, "run.sh"), encoding="utf-8").read().splitlines()
        header = header[1:header.index("set -euo pipefail")]                                # usage prints the WHOLE header comment (line 2 up to the `set -euo pipefail` line, by marker)
        self.assertEqual([l for l in r.stderr.splitlines() if l.startswith("#")], header)
        self.assertIn("5 KERNELS refused", r.stderr); self.assertIn("is refused by name too", r.stderr)   # the exit-code sentences, the header's last: what refuses (deployment), what reports (pins), what declines (CFG)
        self.assertIn("are REPORTED", r.stderr); self.assertNotIn("check_pins.py --expect pristine (3)", r.stderr)   # no pin gate in the check order
        r = _bash("./run.sh predict"); self.assertEqual(r.returncode, 2)
        r = _bash("./run.sh check --config nosuch"); self.assertEqual(r.returncode, 2); self.assertIn("no such config", r.stderr)

    def test_pins_are_a_report_the_interpreter_is_the_gate(self):
        """This interpreter has no rc-foundry and no rfd3: `check` prints stock/check_pins.py's report (never a gate) and reaches the package,
        which prints one PINS line per finding and stops on the INTERPRETER — the kit not installed (exact), the tree not pristine (off) —
        rc 3 either way, by the package's own lines."""
        r = _bash("./run.sh check --mode exact"); self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("check_pins: rc-foundry None: installed from not installed; want version 0.2.1.dev13+g4010e3e2e from", r.stderr)   # the report, printed and passed over
        self.assertIn("[rfdiffusion3-opt] PINS rc-foundry None: installed from not installed", r.stderr)                                 # the package's own note line for the same finding
        self.assertIn("[rfdiffusion3-opt] PINS rfd3 is not importable in this interpreter", r.stderr)
        self.assertIn("DRY-RUN mode=exact env=RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1 interpreter=not-installed(no target files)", r.stderr)
        self.assertIn('would_refuse="the kit is not installed in this interpreter', r.stderr)
        r = _bash("./run.sh check --mode off"); self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("DRY-RUN mode=off env=none interpreter=not-installed(no target files)", r.stderr)
        self.assertIn("stock interpreter = this one: tree not-installed (no target files); pins (reported): rc-foundry None: installed from not installed", r.stderr)
        self.assertNotIn("NOT STOCK", r.stderr)

    def test_stock_route_reaches_the_package_on_a_pristine_interpreter(self):
        with tempfile.TemporaryDirectory() as td:
            site = fx.make_site(td, "pristine")
            py = fx.make_venv(td, site)
            env = {"RFDIFFUSION3_STOCK_PYTHON": py, "MODEL_OPT": TREE, "PYTHONPATH": site, "PATH": os.path.dirname(py) + os.pathsep + os.environ.get("PATH", "")}
            r = _bash("./run.sh check --mode off", env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("DRY-RUN mode=off", r.stderr)
            self.assertRegex(r.stderr, r"stock interpreter (= this one: tree pristine \(8/8 stock, 2 kit files absent\); pins ok|\S+: pristine)")
            self.assertIn("rc-foundry 0.2.1.dev13+g4010e3e2e: pinned (commit 4010e3e2e7350edada3e25a45c908c6bf407df4d); rfd3 tree pristine at ", r.stderr)   # stock/check_pins.py's report, printed by `check` and passed over
            # run.sh carries no mode list: a value that is not a mode reaches the package and is refused there, with its reason
            r = _bash("./run.sh check --mode turbo", env=env)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("invalid choice: 'turbo'", r.stderr)
            r = _bash("./run.sh check --mode exact_lowmem", env=env)                     # a lever subset is not a mode either: the same refusal
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("invalid choice: 'exact_lowmem'", r.stderr)
            self.assertNotIn("DRY-RUN", r.stderr)
            r = _bash("./run.sh check --mode off", env=dict(env, RFDIFFUSION3_OPT="exact"))   # --mode wins over the variable: no disagreement refusal
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
