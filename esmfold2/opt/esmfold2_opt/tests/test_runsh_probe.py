"""run.sh's / configs/h100.env's "is the package installed?" probe never masks the interpreter's own refusal: a present esmfold2_opt that refuses at
interpreter start (the .pth hook: an undeclared ESMFOLD2_OPT_* name) or fails otherwise keeps ITS stderr text and ITS exit code; the 'not installed'
word names only a genuinely absent package — decided by the absence probe's own exit 41, never by 'any non-zero'. CPU only; the hook cases need the package installed with its .pth (skipped otherwise); every
subprocess runs from a scratch cwd so the package is never importable from the working directory by accident."""
import os
import subprocess
import sys
import tempfile
import unittest

from .test_core_missing import KIT, OPT_DIR, CORE_SRC, python_S_dir

RUN_SH = os.path.join(KIT, "run.sh"); ENV_FILE = os.path.join(KIT, "configs", "h100.env")


def _pth_installed() -> bool:
    r = subprocess.run([sys.executable, "-c", "import esmfold2_opt.pth_gate as g, sys; sys.exit(0 if g.installed() else 1)"], capture_output=True, text=True, cwd=tempfile.gettempdir())
    return r.returncode == 0


def _base_env(home):
    return {"PATH": os.pathsep.join([os.path.dirname(sys.executable), os.environ.get("PATH", "")]), "PYTHONPATH": os.pathsep.join([OPT_DIR, CORE_SRC]),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "HOME": home}


class TestProbeNeverMasksTheInterpretersRefusal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_undeclared_switch_name_is_refused_in_the_hooks_words_not_as_not_installed(self):
        if not _pth_installed():
            self.skipTest("esmfold2_opt's .pth hook is not installed in this interpreter")
        env = dict(_base_env(self.tmp.name), ESMFOLD2_OPT_BOGUS_SWITCH="1")
        for label, argv, extra in (("cli route", ["check", "--variant", "fast", "--mode", "fast"], {}), ("env route", ["check"], {"ESMFOLD2_OPT": "fast", "ESMFOLD2_VARIANT": "fast"})):
            r = subprocess.run(["bash", RUN_SH, *argv], env=dict(env, **extra), capture_output=True, text=True, cwd=self.tmp.name)
            self.assertEqual(r.returncode, 3, (label, r.stderr[-600:]))
            self.assertIn("undeclared ESMFOLD2_OPT_BOGUS_SWITCH", r.stderr, label)
            self.assertNotIn("not installed", r.stderr, label); self.assertNotIn("predates the autoload gate", r.stderr, label); self.assertNotIn("Traceback", r.stderr, label)
        r = subprocess.run(["bash", "-c", f'source "{ENV_FILE}" || exit $?; echo REACHED'], env=env, capture_output=True, text=True, cwd=self.tmp.name)
        self.assertEqual(r.returncode, 3, r.stderr[-600:]); self.assertIn("undeclared ESMFOLD2_OPT_BOGUS_SWITCH", r.stderr); self.assertNotIn("not installed", r.stderr); self.assertNotIn("REACHED", r.stdout)

    def test_a_genuinely_absent_package_says_not_installed(self):
        pyS = python_S_dir(self.tmp.name)                                                # `python` = this interpreter with -S: nothing installed; PYTHONPATH without the package
        env = dict(_base_env(self.tmp.name), PATH=os.pathsep.join([pyS, os.environ.get("PATH", "")]), PYTHONPATH=CORE_SRC)
        r = subprocess.run(["bash", RUN_SH, "check", "--variant", "fast", "--mode", "fast"], env=env, capture_output=True, text=True, cwd=self.tmp.name)
        self.assertEqual(r.returncode, 3, r.stderr[-600:]); self.assertIn("NOT ACTIVE: esmfold2_opt is not installed", r.stderr)
        r = subprocess.run(["bash", "-c", f'source "{ENV_FILE}" || exit $?; echo REACHED'], env=env, capture_output=True, text=True, cwd=self.tmp.name)
        self.assertEqual(r.returncode, 3, r.stderr[-600:]); self.assertIn("NOT ACTIVE: esmfold2_opt is not installed", r.stderr); self.assertNotIn("REACHED", r.stdout)

    def test_any_other_failure_keeps_the_interpreters_text_and_exit_code(self):
        shim = os.path.join(self.tmp.name, "shim"); os.makedirs(shim)
        with open(os.path.join(shim, "python"), "w") as fh:                               # `python -m esmfold2_opt._core_gate` dies with its own words and rc 5; anything else is the real interpreter
            fh.write('#!/usr/bin/env bash\ncase "$*" in *"esmfold2_opt._core_gate"*) echo "boom: the gate interpreter failed (simulated)" >&2; exit 5 ;; esac\nexec "%s" "$@"\n' % sys.executable)
        os.chmod(os.path.join(shim, "python"), 0o755)
        env = dict(_base_env(self.tmp.name), PATH=os.pathsep.join([shim, os.environ.get("PATH", "")]))
        r = subprocess.run(["bash", RUN_SH, "check", "--variant", "fast", "--mode", "fast"], env=env, capture_output=True, text=True, cwd=self.tmp.name)
        self.assertEqual(r.returncode, 5, r.stderr[-600:]); self.assertIn("boom: the gate interpreter failed", r.stderr); self.assertNotIn("not installed", r.stderr)
        r = subprocess.run(["bash", "-c", f'source "{ENV_FILE}" || exit $?; echo REACHED'], env=env, capture_output=True, text=True, cwd=self.tmp.name)
        self.assertEqual(r.returncode, 5, r.stderr[-600:]); self.assertIn("boom", r.stderr); self.assertNotIn("not installed", r.stderr); self.assertNotIn("REACHED", r.stdout)

    # ------------------------------------------------------------------ a LIVE start-up hook (what an installed .pth does): the shadow package refuses at interpreter start
    def _shadow(self, refuse: bool) -> str:
        """A directory first on PYTHONPATH: sitecustomize.py imports esmfold2_opt at interpreter start (as the installed .pth does); with `refuse`, a fake
        esmfold2_opt writes a refusal sentence and raises SystemExit(5) — surfaced as exit 5 the way the kit's .pth surfaces its own SystemExit (os._exit)."""
        d = os.path.join(self.tmp.name, "shadow_refuse" if refuse else "shadow_guard"); os.makedirs(d)
        if refuse:
            os.makedirs(os.path.join(d, "esmfold2_opt"))
            with open(os.path.join(d, "esmfold2_opt", "__init__.py"), "w") as fh:
                fh.write('import sys\nsys.stderr.write("[shadow] esmfold2_opt refused at interpreter start (simulated start-up hook)\\n"); sys.stderr.flush()\nraise SystemExit(5)\n')
            with open(os.path.join(d, "sitecustomize.py"), "w") as fh:
                fh.write('import os, sys\ntry:\n    import esmfold2_opt\nexcept SystemExit as x:\n    sys.stderr.flush(); os._exit(x.code if isinstance(x.code, int) else 3)\n')
        else:
            with open(os.path.join(d, "sitecustomize.py"), "w") as fh:
                fh.write('try:\n    import esmfold2_opt\nexcept ImportError:\n    pass\n')
        return d

    def test_a_startup_hook_refusal_keeps_its_text_and_exit_code_never_not_installed(self):
        shadow = self._shadow(refuse=True)
        env = dict(_base_env(self.tmp.name), PYTHONPATH=os.pathsep.join([shadow, OPT_DIR, CORE_SRC]))
        probe = subprocess.run(["python", "-c", "print('unreachable')"], env=env, capture_output=True, text=True, cwd=self.tmp.name)
        self.assertEqual(probe.returncode, 5, "the shadow hook is live in every interpreter of this environment: " + probe.stderr[-300:])
        for label, cmd in (("run.sh", ["bash", RUN_SH, "check", "--variant", "fast", "--mode", "fast"]), ("configs/h100.env", ["bash", "-c", f'source "{ENV_FILE}" || exit $?; echo REACHED'])):
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=self.tmp.name)
            self.assertEqual(r.returncode, 5, (label, r.stderr[-600:]))
            self.assertIn("[shadow] esmfold2_opt refused at interpreter start", r.stderr, label)
            self.assertNotIn("not installed", r.stderr + r.stdout, label); self.assertNotIn("REACHED", r.stdout, label)

    def test_genuine_absence_under_a_guarded_startup_hook_still_says_not_installed(self):
        shadow = self._shadow(refuse=False)                                             # sitecustomize guards its import (ImportError -> pass); no package anywhere
        pyS = python_S_dir(self.tmp.name)                                                # this interpreter without its installed packages
        env = dict(_base_env(self.tmp.name), PATH=os.pathsep.join([pyS, os.environ.get("PATH", "")]), PYTHONPATH=os.pathsep.join([shadow, CORE_SRC]))
        for label, cmd in (("run.sh", ["bash", RUN_SH, "check", "--variant", "fast", "--mode", "fast"]), ("configs/h100.env", ["bash", "-c", f'source "{ENV_FILE}" || exit $?; echo REACHED'])):
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=self.tmp.name)
            self.assertEqual(r.returncode, 3, (label, r.stderr[-600:])); self.assertIn("NOT ACTIVE: esmfold2_opt is not installed", r.stderr, label); self.assertNotIn("REACHED", r.stdout, label)

    def test_the_absence_sentinel_is_41_and_only_41(self):
        src = open(RUN_SH).read() + open(ENV_FILE).read()
        self.assertEqual(src.count("is not None else 41)"), 2, "run.sh and configs/h100.env carry the same sentinel probe")
        self.assertIn('[ "$rc" -eq 41 ]', src); self.assertIn('[ "$_ef2_probe" -eq 41 ]', src); self.assertNotIn("is not None else 1)", src)

if __name__ == "__main__":
    unittest.main()
