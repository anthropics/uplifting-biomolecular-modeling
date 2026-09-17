"""Interpreter start with the package installed: the generated guard .pth is in site-packages; with FLASHZOI_OPT unset nothing beyond flashzoi_opt and
its _autoload module is imported and no finder is installed; with FLASHZOI_OPT=exact the finder is on sys.meta_path and nothing else
is imported until the trigger; an unknown mode prints the NOT ACTIVE line and installs a refusing finder (the trigger import exits 3); the console script resolves."""
import glob
import os
import shutil
import importlib.util
import subprocess
import sys
import sysconfig
import unittest

PROBE = ("import sys; f=[type(x).__name__ for x in sys.meta_path if type(x).__module__=='flashzoi_opt._autoload']; "
         "print(sorted(m for m in sys.modules if m.startswith('flashzoi_opt') or m in ('torch','borzoi_pytorch','numpy')), f)")


def _env(**kw):
    e = {k: v for k, v in os.environ.items() if not k.startswith("FLASHZOI_") and not k.startswith("FZ_")}
    e.update(kw)
    return e


def _installed() -> bool:
    """True when THIS interpreter has the package installed (pip install -e flashzoi/opt): its generated guard .pth sits in site-packages.
    A bare tree imported through PYTHONPATH (no .pth, no console script) is not the installed state these tests describe — they are
    skipped there by name, never failed."""
    return bool(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "flashzoi_opt_autoload.pth")))


INSTALLED = _installed()
NOT_INSTALLED = "flashzoi_opt is not installed in this interpreter (no flashzoi_opt_autoload.pth in site-packages; the tree is on PYTHONPATH only): the installed-state test is skipped by name — pip install -e flashzoi/opt to run it"


class TestStartup(unittest.TestCase):
    @unittest.skipUnless(INSTALLED, NOT_INSTALLED)
    def test_pth_installed(self):
        pths = glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "flashzoi_opt_autoload.pth"))
        self.assertTrue(pths, "flashzoi_opt_autoload.pth is not in site-packages: install the package (pip install -e flashzoi/opt)")
        text = open(pths[0]).read()
        self.assertTrue(text.startswith("# opt_core autoload guard: package=flashzoi_opt env=FLASHZOI_OPT tag=flashzoi-opt exit=3\n"), text[:120])   # the generated guard (opt/_build_backend.py pth_text), never a bare import
        tree = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flashzoi_opt_autoload.pth")   # opt/flashzoi_opt_autoload.pth (the package is installed editable from the tree)
        self.assertEqual(text, open(tree).read(), "the installed .pth differs from the tree's opt/flashzoi_opt_autoload.pth (reinstall: pip install -e flashzoi/opt)")

    @unittest.skipUnless(INSTALLED, NOT_INSTALLED)
    def test_unset_imports_nothing_else(self):
        r = subprocess.run([sys.executable, "-s", "-c", PROBE], env=_env(), capture_output=True, text=True, check=True)
        self.assertEqual(r.stdout.strip(), "['flashzoi_opt', 'flashzoi_opt._autoload'] []", r.stdout)

    @unittest.skipUnless(INSTALLED, NOT_INSTALLED)
    def test_exact_installs_finder_only(self):
        for mode in ("exact",):
            r = subprocess.run([sys.executable, "-s", "-c", PROBE], env=_env(FLASHZOI_OPT=mode), capture_output=True, text=True, check=True)
            self.assertEqual(r.stdout.strip(), "['flashzoi_opt', 'flashzoi_opt._autoload'] ['Finder']", r.stdout)

    @unittest.skipUnless(INSTALLED, NOT_INSTALLED)
    def test_off_and_unknown(self):
        r = subprocess.run([sys.executable, "-s", "-c", PROBE], env=_env(FLASHZOI_OPT="off"), capture_output=True, text=True, check=True)
        self.assertTrue(r.stdout.strip().endswith(" []"))
        r = subprocess.run([sys.executable, "-s", "-c", PROBE], env=_env(FLASHZOI_OPT="turbo"), capture_output=True, text=True, check=True)
        self.assertTrue(r.stdout.strip().endswith(" ['Finder']"), r.stdout); self.assertIn("[flashzoi-opt] NOT ACTIVE: unknown FLASHZOI_OPT='turbo'", r.stderr)
        # the refusing finder: the trigger import exits 3 (the CLI's code) — stock never runs under a misspelt mode
        if importlib.util.find_spec("borzoi_pytorch") is None:
            raise unittest.SkipTest("the stock package is not installed here: the trigger import cannot fire (rc 1 = ModuleNotFoundError, not the finder's 3)")
        r = subprocess.run([sys.executable, "-s", "-c", "import borzoi_pytorch"], env=_env(FLASHZOI_OPT="turbo"), capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("NOT ACTIVE: unknown FLASHZOI_OPT='turbo'", r.stderr)

    def test_trigger_fires_and_refuses_on_cpu(self):
        """FLASHZOI_OPT=exact, then `import borzoi_pytorch` in a fresh interpreter on this CPU box: the finder fires after the package body,
        enable() is refused (no GPU or a pin), the NOT ACTIVE line is printed and the process exits 3 — never stock silently."""
        try:
            import borzoi_pytorch  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"borzoi-pytorch not importable: {e!r}")
        for mode in ("exact",):
            r = subprocess.run([sys.executable, "-s", "-c", "import borzoi_pytorch; print('reached')"], env=_env(FLASHZOI_OPT=mode), capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("[flashzoi-opt] NOT ACTIVE:", r.stderr); self.assertNotIn("reached", r.stdout)

    @unittest.skipUnless(INSTALLED, NOT_INSTALLED)
    def test_console_script(self):
        exe = shutil.which("flashzoi-opt", path=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
        self.assertTrue(exe, "console script flashzoi-opt not found beside the interpreter")
        r = subprocess.run([exe, "--version"], env=_env(), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0); self.assertIn("flashzoi_opt", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
