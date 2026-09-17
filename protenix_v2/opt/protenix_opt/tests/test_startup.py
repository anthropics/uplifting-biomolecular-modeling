"""Packaging and interpreter start: `pip install -e common/opt_core -e protenix_v2/opt` works, ships protenix_opt_autoload.pth into
site-packages, and that .pth leaves `python -c pass` unchanged — no torch, no protenix, nothing of the core with PROTENIX_OPT unset or
off, a sub-millisecond autoload import — with and without PROTENIX_OPT set; a known mode arms the core's finder with the folded selection.
Also builds the wheel and checks the .pth sits at its root with a RECORD entry.

Runs in a throwaway venv created from this interpreter (~15 s); venv/pip failures are test failures, not skips."""
import glob
import json
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

from protenix_opt import _core, stack

N_RUNS = 15
WALL_BUDGET_S = 0.030                # median start-up delta allowed for the .pth with no selection (a torch import costs > 1 s)
WALL_BUDGET_GATE_S = 0.090           # under a selection: + the pre-import core pin gate and the core's finder
IMPORTTIME_BUDGET_US = 5000          # cumulative -X importtime for protenix_opt + protenix_opt._autoload with no selection (a stock process)
IMPORTTIME_BUDGET_GATE_US = 60000    # under a selection: + the pre-import core pin gate (importlib.util, json, re, one find_spec, the pin + MANIFEST reads) and the core's finder
FINDERS = "[type(f).__module__ + '.' + type(f).__name__ for f in sys.meta_path if 'autoload' in type(f).__module__]"


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
        r = subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", cls.venv], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("venv unavailable (a precondition of this test, not a reason to skip): " + r.stderr[-300:])
        cls.python = os.path.join(cls.venv, "bin", "python")
        cls.env = {"PATH": os.path.join(cls.venv, "bin") + os.pathsep + os.environ.get("PATH", ""), "HOME": cls.tmp,
                   "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        cls.wall_before = _wall(cls.python, cls.env)
        # Building from the venv's own site-packages (--no-build-isolation) needs setuptools >= 70.1 (PEP 660 editable installs since 64,
        # bdist_wheel bundled since 70.1 — older setuptools needs the separate `wheel` package, which a --system-site-packages venv of an
        # image python does not have: "invalid command 'bdist_wheel'"). Below that the build is isolated (pip fetches the pinned backend).
        probe = subprocess.run([cls.python, "-c", "import setuptools; print(setuptools.__version__)"], env=cls.env, capture_output=True, text=True)
        m = re.match(r"(\d+)\.(\d+)", probe.stdout.strip()) if probe.returncode == 0 else None
        st_ver = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        cls.build_args = ["--no-build-isolation"] if st_ver >= (70, 1) else ["--use-pep517"]
        cls.setuptools = probe.stdout.strip() or "absent"
        cls.install_cmd = [cls.python, "-m", "pip", "install", "-q", *cls.build_args, "--config-settings", "editable_mode=compat",
                           "-e", _core.pinned_path(), "-e", stack.opt_home()]                # the kit is installed as the core plus itself
        r = subprocess.run(cls.install_cmd, env=cls.env, capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"pip install -e failed (a precondition of this test, not a reason to skip): setuptools {cls.setuptools}; "
                                 f"{' '.join(cls.install_cmd)}\n" + r.stderr[-1500:])
        cls.site = json.loads(subprocess.run([cls.python, "-c", "import json, site; print(json.dumps(site.getsitepackages()))"],
                                             env=cls.env, capture_output=True, text=True, check=True).stdout)[0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _modules(self, extra_env, roots=("protenix_opt", "torch", "protenix", "runner")):
        env = dict(self.env, **extra_env)
        out = subprocess.run([self.python, "-c", "import sys, json; print(json.dumps(sorted(m for m in sys.modules "
                              f"if m.split('.')[0] in {roots!r})))"], env=env, capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    def test_pth_is_installed(self):
        pth = os.path.join(self.site, "protenix_opt_autoload.pth")
        self.assertTrue(os.path.isfile(pth), os.listdir(self.site))
        import importlib.util                                                                          # the text is the build backend's generated guard (opt/_build_backend.py pth_text), never hand-written
        spec = importlib.util.spec_from_file_location("_kit_build_backend", os.path.join(os.path.dirname(_core.PYPROJECT), "_build_backend.py"))
        bb = importlib.util.module_from_spec(spec); spec.loader.exec_module(bb)
        self.assertEqual(open(pth).read(), bb.pth_text("protenix_opt", "PROTENIX_OPT", "protenix-opt"))
        self.assertIn("import protenix_opt._autoload", open(pth).read())
        record = glob.glob(os.path.join(self.site, "protenix_opt-*.dist-info", "RECORD"))[0]
        self.assertIn("protenix_opt_autoload.pth,sha256=", open(record).read(), "pip uninstall must remove the .pth")

    def test_interpreter_start_imports_nothing_heavy(self):
        roots = ("protenix_opt", "opt_core", "torch", "protenix", "runner")
        for extra in ({}, {"PROTENIX_OPT": "off"}, {"PROTENIX_OPT": " OFF "}, {"PROTENIX_OPT_FORCE": "1"}):
            self.assertEqual(self._modules(extra, roots), ["protenix_opt", "protenix_opt._autoload", "protenix_opt._frozen"], extra)   # nothing of the core; _frozen = the leaf frozen-weights rule (stdlib only)
        self.assertEqual(self._modules({"PROTENIX_OPT": "exact"}), ["protenix_opt", "protenix_opt._autoload", "protenix_opt._core", "protenix_opt._core_gate", "protenix_opt._frozen"])   # a selection: + the pre-import core pin gate (stdlib only) and the pinned-copy placement
        self.assertEqual(self._modules({"PROTENIX_OPT": "exact"}, roots), ["opt_core", "opt_core.autoload", "protenix_opt", "protenix_opt._autoload", "protenix_opt._core", "protenix_opt._core_gate", "protenix_opt._frozen"])   # the core's finder module alone
        for raw, mode in (("fast", "fast"), ("FAST", "fast"), (" Exact ", "exact")):
            out = subprocess.run([self.python, "-c", "import sys, protenix_opt._autoload as A; print(A.FINDER is not None and A.FINDER in sys.meta_path, A.FINDER.mode, A.FINDER.armed, A.FINDER.spec.package, " + FINDERS + ")"],
                                 env=dict(self.env, PROTENIX_OPT=raw), capture_output=True, text=True, check=True)
            self.assertEqual(out.stdout.split(), ["True", mode, "True", "protenix_opt", "['opt_core.autoload.Finder']"], raw)
        out = subprocess.run([self.python, "-c", "import protenix_opt._autoload as A; print(A.FINDER)"], env=self.env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "None", "no PROTENIX_OPT: no finder")

    def test_unknown_selection_refuses_at_interpreter_start(self):
        """The installed .pth: an unknown mode or an undeclared PROTENIX_OPT* name exits 3 with the kit's one line, no traceback, the
        program never started; `python -S` processes no .pth and is untouched."""
        for extra, needle in (({"PROTENIX_OPT": "turbo"}, "[protenix-opt] NOT ACTIVE: unknown PROTENIX_OPT='turbo' (expected exact|fast|big|off)"),
                              ({"PROTENIX_OPT_MODE": "exact"}, "[protenix-opt] NOT ACTIVE: undeclared PROTENIX_OPT_MODE")):
            r = subprocess.run([self.python, "-c", "print('STOCK RAN')"], env=dict(self.env, **extra), capture_output=True, text=True)
            self.assertEqual((r.returncode, r.stdout), (3, ""), (extra, r.stderr[-400:]))
            lines = [l for l in r.stderr.splitlines() if l.strip()]
            self.assertEqual(len(lines), 1, r.stderr[-400:]); self.assertTrue(lines[0].startswith(needle), lines[0]); self.assertNotIn("Traceback", r.stderr)
        r = subprocess.run([self.python, "-S", "-c", "import sys; print('STOCK RAN', " + FINDERS + ", 'protenix_opt' in sys.modules)"], env=dict(self.env, PROTENIX_OPT="turbo"), capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "STOCK RAN [] False"), r.stderr[-300:])

    def test_autoload_import_cost(self):
        for extra, budget in (({}, IMPORTTIME_BUDGET_US), ({"PROTENIX_OPT": "exact"}, IMPORTTIME_BUDGET_GATE_US)):
            out = subprocess.run([self.python, "-X", "importtime", "-c", "pass"], env=dict(self.env, **extra), capture_output=True, text=True, check=True)
            rows = [ln.split("|") for ln in out.stderr.splitlines() if ln.startswith("import time:") and ln.count("|") == 2]
            cum = {mod.strip(): int(c) for _, c, mod in rows if c.strip().isdigit()}          # skip the "self | cumulative | name" header
            self.assertIn("protenix_opt._autoload", cum)
            self.assertLess(cum["protenix_opt._autoload"], budget, cum)
            self.assertFalse([m for m in cum if m.split(".")[0] in ("torch", "protenix", "runner")], "heavy imports at interpreter start")

    def test_wall_clock_unchanged(self):
        after = _wall(self.python, self.env)
        after_set = _wall(self.python, dict(self.env, PROTENIX_OPT="exact"))
        sys.stderr.write(f"\n[python -c pass] before .pth {self.wall_before*1e3:.1f} ms | after {after*1e3:.1f} ms | after+PROTENIX_OPT {after_set*1e3:.1f} ms\n")
        self.assertLess(after - self.wall_before, WALL_BUDGET_S)
        self.assertLess(after_set - self.wall_before, WALL_BUDGET_GATE_S)

    def test_wheel_carries_the_pth(self):
        out = os.path.join(self.tmp, "wheel")
        r = subprocess.run([self.python, "-m", "pip", "wheel", "-q", *self.build_args, "--no-deps", "-w", out, stack.opt_home()],
                           env=self.env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, f"setuptools {self.setuptools}; {' '.join(self.build_args)}\n" + r.stderr[-800:])
        whl = glob.glob(os.path.join(out, "protenix_opt-*.whl"))[0]
        with zipfile.ZipFile(whl) as z:
            names = z.namelist()
            self.assertIn("protenix_opt_autoload.pth", names)
            record = z.read([n for n in names if n.endswith(".dist-info/RECORD")][0]).decode()
        self.assertIn("protenix_opt_autoload.pth,sha256=", record)
        self.assertTrue(record.rstrip().splitlines()[-1].endswith("RECORD,,"))
        self.assertEqual(sorted({n.split("/")[0] for n in names if n.endswith(".py")}), ["protenix_opt"], "only the package ships")


if __name__ == "__main__":
    unittest.main()
