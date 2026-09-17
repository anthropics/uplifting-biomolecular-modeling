"""Packaging and interpreter start: the documented install (`pip install -e common/opt_core -e rfdiffusion3/opt`) works in a fresh venv
WITHOUT torch or the upstream package, ships the generated rfdiffusion3_opt_autoload.pth into site-packages, and that .pth leaves
`python -c pass` unchanged — no torch, no upstream, a few-millisecond autoload import (the core pin gate reads two files under a set
RFDIFFUSION3_OPT) — with and without RFDIFFUSION3_OPT set; under the installed .pth an absent or below-pin core is refused by name (exit 3),
never swallowed. Also builds the wheel and checks the .pth sits at its root with a RECORD entry, and that the console script answers. (A
non-editable install of the package carries no opt/pyproject.toml within reach of the package: the core pin gate refuses it by name,
`core_pin_unreadable` — the documented install is editable.) The copy under test is byte-compiled once (as pip compiles a regular install), so
the budgets measure the modules' import work, not source compilation: an editable tree run under `PYTHONDONTWRITEBYTECODE=1`
(configs/h100.env) with no `__pycache__` additionally pays that compilation at every start — a few milliseconds more on a CPU box, by
the `-X importtime` log — which stays two orders of magnitude under a torch import.

Runs in a throwaway venv created from this interpreter (~15-40 s; the build backend's setuptools comes through pip's build
isolation); venv/pip failures are test failures, not skips."""
import glob
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

from .. import _autoload, registry

N_RUNS = 15
WALL_BUDGET_FACTOR = 2.0             # the .pth may at most double the installed interpreter's own start-up, measured with the .pth set aside (a torch import costs > 1 s, i.e. > 20x)
IMPORTTIME_BUDGET_US = 5000          # cumulative -X importtime for rfdiffusion3_opt + rfdiffusion3_opt._autoload, RFDIFFUSION3_OPT unset: the inert path (two small modules, nothing else)
IMPORTTIME_BUDGET_MODE_US = 30000    # the same under RFDIFFUSION3_OPT=exact: + the core pin gate (tomllib + json on two small files) + the core's finder; a torch import is > 1 000 000
PTH = "rfdiffusion3_opt_autoload.pth"
OPT_HOME = os.path.join(registry.tree_home(), "opt")


def _git_ignored(path):
    """True | False when git can answer for `path`; None outside a repository or without git (an extracted tree)."""
    try:
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=os.path.dirname(path), capture_output=True)
    except FileNotFoundError:
        return None
    return None if r.returncode == 128 else r.returncode == 0


def _wall(python, env, n=N_RUNS):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        subprocess.run([python, "-c", "pass"], env=env, check=True)
        times.append(time.perf_counter() - t)
    return statistics.median(times)


class TestStartup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.venv = os.path.join(cls.tmp, "venv")
        r = subprocess.run([sys.executable, "-m", "venv", cls.venv], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("venv unavailable (a precondition of this test, not a reason to skip): " + r.stderr[-300:])
        cls.python = os.path.join(cls.venv, "bin", "python")
        cls.env = {"PATH": os.path.join(cls.venv, "bin") + os.pathsep + os.environ.get("PATH", ""), "HOME": cls.tmp,
                   "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1", "MODEL_OPT": registry.tree_home()}
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            if k in os.environ:
                cls.env[k] = os.environ[k]
        cls.opt_copy = os.path.join(cls.tmp, "opt")                               # built from a copy: setuptools writes build/ beside the sources
        shutil.copytree(OPT_HOME, cls.opt_copy, ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__", ".pytest_cache"))
        subprocess.run([cls.python, "-m", "compileall", "-q", os.path.join(cls.opt_copy, "rfdiffusion3_opt")], env=cls.env, check=True, capture_output=True)   # byte-compiled once, as pip compiles a regular install (PYTHONDONTWRITEBYTECODE keeps the interpreter from doing it at start-up)
        cls.core_dir = _autoload.core_gate()["installed"]["root"]                 # the kit is installed as the core plus itself, both editable (README, run.sh install)
        r = subprocess.run([cls.python, "-m", "pip", "install", "-q", "-e", cls.core_dir, "-e", cls.opt_copy], env=cls.env, capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("pip install failed: " + r.stderr[-1500:])
        cls.site = subprocess.run([cls.python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], env=cls.env,
                                  capture_output=True, text=True, check=True).stdout.strip()
        pth = os.path.join(cls.site, PTH)                                          # the control: the same installed venv with the kit's .pth line set aside
        os.rename(pth, pth + ".off")
        try:
            cls.wall_before = _wall(cls.python, cls.env)
        finally:
            os.rename(pth + ".off", pth)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_no_build_residue_in_the_tree(self):
        """setuptools writes `build/` beside the sources it builds — these tests build from a copy (setUpClass), so the tree carries
        none. The documented editable install of the tree (`pip install -e opt`, tests/README.md) leaves `opt/rfdiffusion3_opt.egg-info/`
        in place: tolerated when the repository ignores it (`*.egg-info/`), never tracked."""
        self.assertFalse(os.path.exists(os.path.join(OPT_HOME, "build")))
        egg = os.path.join(OPT_HOME, "rfdiffusion3_opt.egg-info")
        if os.path.exists(egg):
            self.assertNotEqual(_git_ignored(egg), False, "opt/rfdiffusion3_opt.egg-info exists and is not git-ignored")

    def test_no_torch_no_upstream_in_the_venv(self):
        r = subprocess.run([self.python, "-c", "import importlib.util as u; print([u.find_spec(m) is None for m in ('torch','rfd3','foundry')])"],
                           env=self.env, capture_output=True, text=True, check=True)
        self.assertEqual(r.stdout.strip(), "[True, True, True]")

    def test_pth_installed(self):
        self.assertTrue(os.path.isfile(os.path.join(self.site, PTH)), sorted(os.listdir(self.site)))
        text = open(os.path.join(self.site, PTH), encoding="utf-8").read()
        self.assertEqual(text, open(os.path.join(OPT_HOME, PTH), encoding="utf-8").read())    # the kit's generated .pth, byte for byte (test_core_adoption: == the backend's pth_text)
        self.assertTrue(text.startswith("# opt_core autoload guard: package=rfdiffusion3_opt env=RFDIFFUSION3_OPT tag=%s exit=3\nimport " % _autoload.TAG), text)

    def test_interpreter_start_unchanged(self):
        after = _wall(self.python, self.env)
        self.assertLessEqual(after, WALL_BUDGET_FACTOR * self.wall_before, f"start-up {self.wall_before:.4f}s -> {after:.4f}s (budget {WALL_BUDGET_FACTOR}x)")
        r = subprocess.run([self.python, "-c", "import sys; print(sorted(m for m in sys.modules if m.startswith(('rfdiffusion3_opt','torch','rfd3','foundry'))))"],
                           env=self.env, capture_output=True, text=True, check=True)
        self.assertEqual(r.stdout.strip(), "['rfdiffusion3_opt', 'rfdiffusion3_opt._autoload']")
        r = subprocess.run([self.python, "-c", "import sys; print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch','rfd3','foundry')), "
                            "[type(f).__name__ for f in sys.meta_path if type(f).__module__=='opt_core.autoload'])"],
                           env=dict(self.env, RFDIFFUSION3_OPT="exact"), capture_output=True, text=True, check=True)
        self.assertEqual(r.stdout.strip(), "[] ['Finder']")                   # the core's finder armed from the kit's spec, nothing heavy imported
        r = subprocess.run([self.python, "-c", "import sys; print(sorted(m for m in sys.modules if m.startswith('opt_core')))"],
                           env=dict(self.env, RFDIFFUSION3_OPT="exact"), capture_output=True, text=True, check=True)
        self.assertEqual(r.stdout.strip(), "['opt_core', 'opt_core.autoload']")   # the finder module alone: no gates, no table

    def test_importtime_budget(self):
        for mode, budget in ((None, IMPORTTIME_BUDGET_US), ("exact", IMPORTTIME_BUDGET_MODE_US)):
            env = dict(self.env, RFDIFFUSION3_OPT=mode) if mode else self.env
            r = subprocess.run([self.python, "-X", "importtime", "-c", "pass"], env=env, capture_output=True, text=True, check=True)
            us = {m.group(2).strip(): int(m.group(1)) for m in re.finditer(r"import time:\s+\d+ \|\s+(\d+) \|(.*)", r.stderr)}
            total = us.get("rfdiffusion3_opt", 0) + us.get("rfdiffusion3_opt._autoload", 0)
            self.assertGreater(total, 0, r.stderr[-500:])
            self.assertLess(total, budget, (mode, total))
            self.assertEqual("rfdiffusion3_opt._core_gate" in us, mode is not None, (mode, sorted(m for m in us if m.startswith("rfdiffusion3_opt"))))   # the gate module loads under a kit mode only

    def test_unknown_mode_is_refused_at_start(self):
        """An unknown value is refused at interpreter start on the real route (the installed .pth): the kit's NOT ACTIVE line, exit 3,
        nothing run (a lever subset such as `exact_lowmem` is such a value); an undeclared RFDIFFUSION3_OPT_* name is refused at start in every
        process, a declared one refuses nothing."""
        r = subprocess.run([self.python, "-c", "print('ran')"], env=dict(self.env, RFDIFFUSION3_OPT="turbo"), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip(), r.stderr.strip()), (3, "", "[rfdiffusion3-opt] NOT ACTIVE: 'turbo' is not a mode (exact|fast|off)"))
        r = subprocess.run([self.python, "-c", "print('ran')"], env=dict(self.env, RFDIFFUSION3_OPT="exact_lowmem"), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip(), r.stderr.strip()), (3, "", "[rfdiffusion3-opt] NOT ACTIVE: 'exact_lowmem' is not a mode (exact|fast|off)"))
        r = subprocess.run([self.python, "-c", "import sys; print([f.mode for f in sys.meta_path if type(f).__module__=='opt_core.autoload'])"],
                           env=dict(self.env, RFDIFFUSION3_OPT="exact"), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip(), r.stderr.strip()), (0, "['exact']", ""))            # a mode arms the finder; nothing printed before the trigger
        r = subprocess.run([self.python, "-c", "print('ran')"], env=dict(self.env, RFDIFFUSION3_OPT_ALLOW_PARTAIL="1"), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (3, ""), r.stderr)
        self.assertIn("[rfdiffusion3-opt] NOT ACTIVE: undeclared RFDIFFUSION3_OPT_ALLOW_PARTAIL in the environment", r.stderr)
        r = subprocess.run([self.python, "-c", "print('ran')"], env=dict(self.env, RFDIFFUSION3_OPT_ALLOW_PARTIAL="1"), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip(), r.stderr.strip()), (0, "ran", ""))

    def test_core_missing_is_refused_not_swallowed(self):
        """With RFDIFFUSION3_OPT set and opt_core NOT importable (or importable below the pin), the INSTALLED .pth line does not fall
        through to stock: `site` would swallow an exception raised by a .pth line and carry on (rc 0, no line); the core pin gate refuses by
        name at interpreter start — one line, exit 3 — and with the variable unset the same line is inert. `-S` keeps the venv's site-packages
        (the editable core) out of reach; the package comes in by path. The gate locates a candidate core by regex, never by import: a
        stray opt_core whose __init__.py would raise on import still gets a byte-literal version reading and a name-only mismatch line."""
        code = "import site, sys; site.addpackage(sys.argv[1], sys.argv[2], None); print('fell through to stock')"
        env = dict(self.env, RFDIFFUSION3_OPT="exact", PYTHONPATH=self.opt_copy)
        r = subprocess.run([self.python, "-S", "-c", code, self.site, PTH], env=env, capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip(), r.stderr.count("NOT ACTIVE")), (3, "", 1), r.stderr)
        self.assertIn("[%s] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v" % _autoload.TAG, r.stderr)
        self.assertNotIn("Traceback", r.stderr); self.assertNotIn("Error processing line", r.stderr)
        from .. import _core_gate
        want_v = _core_gate.read_table(os.path.join(OPT_HOME, "pyproject.toml"), "tool.opt_core")["version"]
        unimportable = os.path.join(self.tmp, "nocore"); os.makedirs(os.path.join(unimportable, "opt_core"), exist_ok=True)
        open(os.path.join(unimportable, "opt_core", "__init__.py"), "w").write('__version__ = "0.0.0"\nraise ImportError("never imported: the gate locates, it does not import")\n')
        r = subprocess.run([self.python, "-S", "-c", code, self.site, PTH], env=dict(env, PYTHONPATH=unimportable + os.pathsep + self.opt_copy), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip(), r.stderr.count("NOT ACTIVE")), (3, "", 1), r.stderr)
        self.assertIn("NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v%s at" % want_v, r.stderr)
        self.assertIn(", installed v0.0.0 at %s" % unimportable, r.stderr)
        r = subprocess.run([self.python, "-S", "-c", code, self.site, PTH], env=dict(env, RFDIFFUSION3_OPT=""), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip(), r.stderr.strip()), (0, "fell through to stock", ""))

    def test_console_script_and_module(self):
        exe = os.path.join(self.venv, "bin", "rfdiffusion3-opt")
        self.assertTrue(os.path.isfile(exe))
        r = subprocess.run([exe, "--help"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0); self.assertIn("design", r.stdout); self.assertIn("warm", r.stdout)
        r = subprocess.run([self.python, "-m", "rfdiffusion3_opt"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        r = subprocess.run([self.python, "-m", "rfdiffusion3_opt", "check", "--mode", "off"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 3); self.assertIn("NOT ACTIVE mode=off", r.stderr.replace("DRY-RUN mode=off", "NOT ACTIVE mode=off"))   # no upstream in this venv: not pristine

    def test_wheel_ships_pth_at_root(self):
        wd = os.path.join(self.tmp, "wheels")
        r = subprocess.run([self.python, "-m", "pip", "wheel", "-q", "--no-deps", "-w", wd, self.opt_copy], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        whl = glob.glob(os.path.join(wd, "rfdiffusion3_opt-*.whl"))
        self.assertEqual(len(whl), 1, whl)
        with zipfile.ZipFile(whl[0]) as zf:
            names = zf.namelist()
            self.assertIn(PTH, names)
            record = next(n for n in names if n.endswith(".dist-info/RECORD"))
            self.assertIn(PTH + ",sha256=", zf.read(record).decode())
            self.assertEqual(len([n for n in names if n == record]), 1)


if __name__ == "__main__":
    unittest.main()
