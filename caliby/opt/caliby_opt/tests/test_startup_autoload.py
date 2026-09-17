"""Packaging, interpreter start and the CALIBY_OPT route: `pip install -e common/opt_core -e caliby/opt` works in a fresh venv WITHOUT
torch or the upstream packages, ships caliby_opt_autoload.pth (the build backend's generated guard line, byte for byte the tree's) into
site-packages, and that .pth leaves `python -c pass` unchanged (no torch, no upstream, no caliby_opt submodule and nothing of the core
imported) with and without CALIBY_OPT set. Under CALIBY_OPT=fast the first `import caliby` triggers activation BEFORE the package body
runs and, on a box where it cannot activate, prints NOT ACTIVE and exits 3 — stock never runs silently; with another core first on the
path the same route refuses by name at the trigger (the core pin gate: exit 3, before anything of the core is imported); with CALIBY_OPT
unset nothing is activated (the environment route's default is off; the CLI's is the variant's kit mode); CALIBY_OPT=exact on the
single variant is refused by name at the trigger (exit 3, the words of the mode table). The console script answers. Runs in a
throwaway venv created from this interpreter (~15-40 s)."""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from .. import __version__, stack
from ._fixtures import core_dir

OPT = stack.opt_home()
PTH = "caliby_opt_autoload.pth"


class TestStartup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.venv = os.path.join(cls.tmp, "venv")
        r = subprocess.run([sys.executable, "-m", "venv", cls.venv], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("venv unavailable (a precondition of this test, not a reason to skip): " + r.stderr[-300:])
        cls.python = os.path.join(cls.venv, "bin", "python")
        cls.env = {"PATH": os.path.join(cls.venv, "bin") + os.pathsep + os.environ.get("PATH", ""), "HOME": cls.tmp, "MODEL_OPT": stack.tree_home()}
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST"):
            if k in os.environ:
                cls.env[k] = os.environ[k]
        r = subprocess.run([cls.python, "-m", "pip", "install", "-q", "-e", core_dir(), "-e", OPT], capture_output=True, text=True, env=cls.env)
        if r.returncode != 0:
            raise AssertionError("pip install -e failed: " + (r.stdout + r.stderr)[-1500:])
        cls.site = glob.glob(os.path.join(cls.venv, "lib", "python*", "site-packages"))[0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, code, extra_env=None, args=None):
        env = dict(self.env)
        env.update(extra_env or {})
        return subprocess.run([self.python] + (args or ["-c", code]), capture_output=True, text=True, env=env)

    def test_pth_installed_and_inert(self):
        self.assertTrue(os.path.isfile(os.path.join(self.site, PTH)))
        with open(os.path.join(self.site, PTH)) as installed, open(os.path.join(OPT, PTH)) as shipped:
            self.assertEqual(installed.read(), shipped.read())                 # the backend placed the tree's generated .pth (its text is locked in test_carry_and_pins)
        code = "import sys; print(sorted(m for m in sys.modules if m.startswith('caliby_opt') or m in ('torch','caliby')))"
        for extra in ({}, {"CALIBY_OPT": "exact"}, {"CALIBY_OPT": "off"}):
            r = self._run(code, extra)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "['caliby_opt', 'caliby_opt._autoload']", (extra, r.stdout))
            r2 = self._run("import sys; print(sorted(m for m in sys.modules if m == 'opt_core' or m.startswith('opt_core.')))", extra)
            self.assertEqual(r2.stdout.strip(), "[]", (extra, r2.stdout))   # nothing of the shared core at interpreter start
        r = self._run("import sys; print([type(f).__name__ for f in sys.meta_path if type(f).__name__=='Finder'])", {"CALIBY_OPT": "exact"})
        self.assertEqual(r.stdout.strip(), "['Finder']")
        r = self._run("import sys; print([type(f).__name__ for f in sys.meta_path if type(f).__name__=='Finder'])", {"CALIBY_OPT": "off"})
        self.assertEqual(r.stdout.strip(), "[]")
        r = self._run("pass", {"CALIBY_OPT": "turbo"})
        self.assertEqual((r.returncode, r.stderr.strip()), (0, ""), r.stderr)          # an unknown mode: a process that never imports the model is untouched (refused at the trigger, below)

    def test_env_route_refuses_before_the_package_body(self):
        fake = os.path.join(self.tmp, "fakesp")
        os.makedirs(os.path.join(fake, "caliby"), exist_ok=True)
        with open(os.path.join(fake, "caliby", "__init__.py"), "w") as fh:
            fh.write("print('BODY RAN')\n")
        code = "import caliby; print('after import')"
        r = self._run(code, {"CALIBY_OPT": "fast", "PYTHONPATH": fake})
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("[caliby-opt] NOT ACTIVE:", r.stderr)
        self.assertNotIn("BODY RAN", r.stdout)                         # activation fires before the package body
        self.assertNotIn("after import", r.stdout)
        r = self._run(code, {"PYTHONPATH": fake})                      # without CALIBY_OPT: untouched
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("BODY RAN", r.stdout)
        r = self._run(code, {"CALIBY_OPT": "turbo", "PYTHONPATH": fake})  # an unknown mode is refused at the model's import, never a silent stock run
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("[caliby-opt] NOT ACTIVE: unknown CALIBY_OPT='turbo'", r.stderr)
        self.assertNotIn("BODY RAN", r.stdout)
        r = self._run(code, {"CALIBY_OPT": "exact", "PYTHONPATH": fake})  # exact on the single variant (CALIBY_VARIANT unset): refused by name at the trigger
        self.assertEqual((r.returncode, r.stderr.count("NOT ACTIVE")), (3, 1), r.stderr)
        self.assertIn("[caliby-opt] NOT ACTIVE: exact refused: the single-sequence route differs from stock deterministically", r.stderr)
        self.assertNotIn("BODY RAN", r.stdout)
        self.assertNotIn("Traceback", r.stderr)
        hide = os.path.join(self.tmp, "nocore")                        # another `opt_core` first on the path (no __version__ literal, a body that must never run): the core pin gate at the trigger
        os.makedirs(os.path.join(hide, "opt_core"), exist_ok=True)
        with open(os.path.join(hide, "opt_core", "__init__.py"), "w") as fh:
            fh.write("raise ImportError('opt_core hidden by the test')\n")
        r = self._run(code, {"CALIBY_OPT": "fast", "PYTHONPATH": hide + os.pathsep + fake})
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertEqual(r.stderr.count("NOT ACTIVE"), 1, r.stderr)
        self.assertIn("[caliby-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned ", r.stderr)
        self.assertIn("installed v? at " + hide, r.stderr)   # located, never imported: its ImportError body did not run; no __version__ literal found
        self.assertNotIn("Traceback", r.stderr)
        self.assertNotIn("BODY RAN", r.stdout)

    def test_console_script_and_check(self):
        r = subprocess.run([os.path.join(self.venv, "bin", "caliby-opt"), "--version"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"caliby_opt {__version__}", r.stdout)
        r = subprocess.run([os.path.join(self.venv, "bin", "caliby-opt"), "check", "--mode", "fast", "--no-gpu"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r.returncode, 3, r.stderr)                    # would refuse: EXIT_NOT_ACTIVE, the code design gives
        self.assertIn("[caliby-opt] CHECK mode=fast variant=single tier=2 row=X", r.stdout)
        self.assertIn("would refuse: upstream not at its pin", r.stdout)
        self.assertIn("also stated: clean lever", r.stdout)
        r = subprocess.run([os.path.join(self.venv, "bin", "caliby-opt"), "check", "--no-gpu"], capture_output=True, text=True, env=self.env)
        self.assertEqual(r.returncode, 3, r.stderr)                    # no --mode: the variant's kit mode (fast on single) on this box: pins not met
        self.assertIn("[caliby-opt] CHECK mode=fast variant=single tier=2 row=X", r.stdout)
        r = subprocess.run([os.path.join(self.venv, "bin", "caliby-opt"), "check", "--no-gpu"], capture_output=True, text=True, env={**self.env, "CALIBY_OPT": "off"})
        self.assertIn("[caliby-opt] CHECK mode=off variant=single tier=None row=None", r.stdout)


if __name__ == "__main__":
    unittest.main()
