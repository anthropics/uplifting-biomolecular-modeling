"""Packaging and interpreter start: `pip install -e common/opt_core -e boltzgen/opt` works in a fresh venv WITHOUT torch or upstream, ships
boltzgen_opt_autoload.pth into site-packages, and that .pth leaves `python -c pass` unchanged — no torch, no upstream, a
sub-millisecond autoload import — with and without BOLTZGEN_OPT set; the console script answers; the wheel carries the .pth at its
root with a RECORD entry. Runs in a throwaway venv created from this interpreter (no system site-packages; the build backend's
setuptools comes from the venv: `pip install setuptools` there, or from the build isolation when the index is reachable);
venv/pip failures are test failures, not skips."""
import glob
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

import opt_core
from boltzgen_opt import stack

N_RUNS = 12
WALL_BUDGET_FACTOR = 2.0             # the .pth may at most double the interpreter's own start-up (a torch import costs > 1 s, i.e. > 20x)
IMPORTTIME_BUDGET_US = 8000          # cumulative -X importtime for boltzgen_opt + boltzgen_opt._autoload
PROOF_BOX_ENV = "MODEL_OPT_PROOF_BOX"   # =1 on a dedicated box: the wall-clock budgets below hold there; a shared box is noise
PTH = "boltzgen_opt_autoload.pth"


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
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_startup_")
        cls.venv = os.path.join(cls.tmp, "venv")
        r = subprocess.run([sys.executable, "-m", "venv", cls.venv], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("venv unavailable (a precondition of this test, not a reason to skip): " + r.stderr[-300:])
        cls.python = os.path.join(cls.venv, "bin", "python")
        cls.env = {"PATH": os.path.join(cls.venv, "bin") + os.pathsep + os.environ.get("PATH", ""), "HOME": cls.tmp, "LANG": "C.UTF-8",
                   "PYTHONDONTWRITEBYTECODE": "1", "KMP_AFFINITY": "disabled"}
        cls.opt = stack.opt_home()
        cls.core = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))     # common/opt_core: the pinned core installs with the package (run.sh install)
        r = subprocess.run([cls.python, "-m", "pip", "install", "-q", "-e", cls.core, "-e", cls.opt], env=cls.env, capture_output=True, text=True)
        if r.returncode != 0:                                              # no index: try with the interpreter's own setuptools
            r2 = subprocess.run([cls.python, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-index", "-e", cls.core, "-e", cls.opt],
                                env=dict(cls.env, PYTHONPATH=os.pathsep.join(p for p in sys.path if "site-packages" in p)), capture_output=True, text=True)
            if r2.returncode != 0:
                raise AssertionError("editable install failed:\n" + r.stderr[-800:] + "\n" + r2.stderr[-800:])
        cls.site = subprocess.run([cls.python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], env=cls.env, capture_output=True, text=True, check=True).stdout.strip()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pth_installed_and_no_upstream_needed(self):
        self.assertTrue(os.path.isfile(os.path.join(self.site, PTH)), os.listdir(self.site))
        self.assertEqual(open(os.path.join(self.site, PTH)).read(), open(os.path.join(self.opt, PTH)).read())     # the tree's generated .pth, byte for byte (its text: test_core_adoption)
        r = subprocess.run([self.python, "-c", "import torch"], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0, "the throwaway venv must not have torch: the test proves the package needs none")
        r = subprocess.run([self.python, "-c", "import boltzgen_opt, sys; print(boltzgen_opt.__version__); print('torch' in sys.modules, 'boltzgen' in sys.modules)"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split()[1:], ["False", "False"])

    def test_interpreter_start_unchanged(self):
        code = "import sys; print(sorted(n for n in sys.modules if n.startswith(('boltzgen', 'torch', 'numpy')))); print(any(type(f).__name__ == 'Finder' for f in sys.meta_path))"
        r = subprocess.run([self.python, "-c", code], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip().splitlines(), ["['boltzgen_opt', 'boltzgen_opt._autoload']", "False"])
        r = subprocess.run([self.python, "-c", code], env=dict(self.env, BOLTZGEN_OPT="exact"), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip().splitlines(), ["['boltzgen_opt', 'boltzgen_opt._autoload']", "True"])
        r = subprocess.run([self.python, "-c", "pass"], env=dict(self.env, BOLTZGEN_OPT="faster"), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stderr), (0, ""))                # not a mode: refused at the first `import boltzgen`, never at start

    @unittest.skipUnless(os.environ.get(PROOF_BOX_ENV) == "1", f"a wall-clock budget: runs on a dedicated proof box ({PROOF_BOX_ENV}=1); a shared box is noise")
    def test_startup_wall_and_importtime(self):
        no_pth = dict(self.env, PYTHONNOUSERSITE="1")
        with_pth = dict(self.env, BOLTZGEN_OPT="exact")
        r = subprocess.run([self.python, "-S", "-c", "pass"], env=no_pth, check=True)   # -S: no site -> no .pth, the floor
        base = _wall(self.python, no_pth)
        # the floor without site processing at all
        t_floor = []
        for _ in range(N_RUNS):
            t = time.perf_counter(); subprocess.run([self.python, "-S", "-c", "pass"], env=no_pth, check=True); t_floor.append(time.perf_counter() - t)
        floor = statistics.median(t_floor)
        armed = _wall(self.python, with_pth)
        self.assertLess(armed, max(floor, base) * WALL_BUDGET_FACTOR + 0.05, f"floor {floor:.4f}s site {base:.4f}s armed {armed:.4f}s")
        r = subprocess.run([self.python, "-X", "importtime", "-c", "pass"], env=with_pth, capture_output=True, text=True, check=True)
        total = 0
        for ln in r.stderr.splitlines():
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) == 3 and parts[2] in ("boltzgen_opt", "boltzgen_opt._autoload"):     # ours; pip's editable finder is not counted
                total += int(parts[1]) if parts[1].isdigit() else 0
        self.assertLess(total, IMPORTTIME_BUDGET_US, r.stderr[-1500:])

    def test_console_script_and_wheel(self):
        r = subprocess.run([os.path.join(self.venv, "bin", "boltzgen-opt"), "--help"], env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("design  <spec.yaml> --output DIR", r.stdout)
        wd = os.path.join(self.tmp, "wheel")
        r = subprocess.run([self.python, "-m", "pip", "wheel", "-q", "--no-deps", "-w", wd, self.opt], env=self.env, capture_output=True, text=True)
        if r.returncode != 0:
            r = subprocess.run([self.python, "-m", "pip", "wheel", "-q", "--no-deps", "--no-build-isolation", "--no-index", "-w", wd, self.opt],
                               env=dict(self.env, PYTHONPATH=os.pathsep.join(p for p in sys.path if "site-packages" in p)), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        whl = glob.glob(os.path.join(wd, "boltzgen_opt-*.whl"))
        self.assertEqual(len(whl), 1)
        with zipfile.ZipFile(whl[0]) as z:
            names = z.namelist()
            self.assertIn(PTH, names)
            record = next(n for n in names if n.endswith(".dist-info/RECORD"))
            self.assertIn(PTH + ",sha256=", z.read(record).decode())
            self.assertTrue(any(n.startswith("boltzgen_opt/tests/") for n in names))


if __name__ == "__main__":
    unittest.main()
