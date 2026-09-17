"""No core, the wrong core, or a core without a module this package imports = the kit's NOT ACTIVE line and exit 3 on EVERY documented
entry route — never a traceback, never a silent stock (or one-GPU) run:

(1) the core pin gate (``_core_gate.py``, the byte-identical copy of the release tree's ``kit_template/_core_gate.py``; statement one of
    ``cli.main`` / the console script, the ``.pth`` hook, ``enable()``, ``run.sh``, ``configs/h100.env``): the importable ``opt_core`` is the
    one ``opt/pyproject.toml [tool.opt_core]`` pins (a MINIMUM version) — ``reason=core_missing:opt_core`` when nothing is importable,
    ``reason=core_mismatch: …`` for a core older than the pin (floor semantics: a newer core passes, an older one refuses);
(2) the producer census (``_producers.py``, statement two): every core module this package imports is on disk under that core —
    ``producer_missing:<module>,…`` (a core at or above the pinned version but missing files);
(3) activation with a core module failing to import → ``core_missing:<module>`` (stack's import guard)."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from esmfold2_opt._producers import MIN_CORE, REQUIRED_PRODUCERS

OPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # <kit>/opt: esmfold2_opt importable from here without the core
KIT = os.path.dirname(OPT_DIR)
CORE_SRC = os.path.join(KIT, "..", "common", "opt_core")                                     # the release tree's core beside the kit (its package dir = CORE_SRC/opt_core)
ITEMS = '[{"id": "t", "sequences": [{"type": "protein", "id": "A", "sequence": "MKV", "msa": null}]}]'


def kit_pin() -> dict:
    """The kit's [tool.opt_core] pin as the gate reads it (path, version — version is a MINIMUM)."""
    sys.path.insert(0, OPT_DIR)
    try:
        from esmfold2_opt._core_gate import read_table
    finally:
        sys.path.remove(OPT_DIR)
    return read_table(os.path.join(OPT_DIR, "pyproject.toml"), "tool.opt_core")


def shadow_core(root: str, *, version: str, without=()) -> str:
    """A copy of the release tree's core PACKAGE under ``root`` (first on PYTHONPATH = the core this interpreter would import), with its
    ``__version__`` rewritten to ``version`` and the files in ``without`` removed. Returns ``root``."""
    dst = os.path.join(root, "opt_core")
    shutil.copytree(os.path.join(CORE_SRC, "opt_core"), dst, ignore=shutil.ignore_patterns("__pycache__"))
    init_py = os.path.join(dst, "__init__.py")
    with open(init_py, encoding="utf-8") as fh:
        src = fh.read()
    new_src, n = re.subn(r'^__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"', src, count=1, flags=re.M)
    assert n == 1, "no __version__ literal found to rewrite in opt_core/__init__.py"
    with open(init_py, "w", encoding="utf-8") as fh:
        fh.write(new_src)
    for rel in without:
        p = os.path.join(dst, rel)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    return root


def python_S_dir(d: str) -> str:
    """A directory holding a `python` that runs this interpreter with -S (no site-packages: an installed core is not importable) — for the
    shell entry points, which invoke `python` themselves."""
    b = os.path.join(d, "bin"); os.makedirs(b, exist_ok=True)
    with open(os.path.join(b, "python"), "w") as fh:
        fh.write(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
    os.chmod(os.path.join(b, "python"), 0o755)
    return b


class TestCoreGateOnEveryRoute(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = self.tmp.name
        self.inp = os.path.join(d, "in.json")
        with open(self.inp, "w") as fh:
            fh.write(ITEMS)
        pin = kit_pin(); self.pin = pin
        self.absent_env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": OPT_DIR, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "HOME": d}
        self.stale = shadow_core(os.path.join(d, "stale"), version="0.2.5")                                    # older than the pin: core_mismatch (floor semantics)
        self.holed = shadow_core(os.path.join(d, "holed"), version=pin["version"], without=("mem", "arch.py"))  # at the pin, but files of this package's producers absent
        self.words_absent = "[esmfold2-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v%s at %s; nothing importable as opt_core on sys.path)" % (pin["version"], pin["path"])
        self.words_stale = "[esmfold2-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v%s at %s, installed v0.2.5 at " % (pin["version"], pin["path"])
        holed_names = [m for m, f in REQUIRED_PRODUCERS if f.split(os.sep)[0] == "mem" or f == "arch.py"]        # the producers the holed core lacks, in the table's order
        self.assertEqual(holed_names[:2] + holed_names[-2:], ["opt_core.mem", "opt_core.mem.torch_alloc", "opt_core.mem.rowpair.rankdata", "opt_core.arch"])
        self.words_holed = f"[esmfold2-opt] NOT ACTIVE: producer_missing:{','.join(holed_names)} — this package imports opt_core >= {MIN_CORE}"

    def tearDown(self):
        self.tmp.cleanup()

    def env_with(self, core_root: str) -> dict:
        return dict(self.absent_env, PYTHONPATH=os.pathsep.join([core_root, OPT_DIR]))

    def assert_refused(self, r, words, what):
        self.assertEqual(r.returncode, 3, (what, r.stderr[-700:])); self.assertIn(words, r.stderr, what)
        self.assertNotIn("Traceback", r.stderr, what); self.assertNotIn("REACHED", r.stdout, what)

    # ---------------------------------------------------------------------------------------------------------------- python -m / console script
    def test_cli_routes(self):
        pred = ["pred", "--mode", "big", "--variant", "fast", "--n_gpu", "2", "--input", self.inp, "--out_dir", os.path.join(self.tmp.name, "o")]
        for argv in (["--help"], ["check", "--mode", "fast", "--variant", "fast"], pred):
            r = subprocess.run([sys.executable, "-S", "-m", "esmfold2_opt", *argv], env=self.absent_env, capture_output=True, text=True)      # -S: no installed core importable
            self.assert_refused(r, self.words_absent, ("absent", argv))
            r = subprocess.run([sys.executable, "-m", "esmfold2_opt", *argv], env=self.env_with(self.stale), capture_output=True, text=True)
            self.assert_refused(r, self.words_stale, ("stale", argv))
            r = subprocess.run([sys.executable, "-m", "esmfold2_opt", *argv], env=self.env_with(self.holed), capture_output=True, text=True)
            self.assert_refused(r, self.words_holed, ("holed", argv))
        exe = shutil.which("esmfold2-opt", path=os.path.dirname(sys.executable))                          # the console script = esmfold2_opt.__main__:main — the same statement one, when installed here
        if exe:
            r = subprocess.run([exe, "check", "--mode", "fast", "--variant", "fast"], env=self.env_with(self.stale), capture_output=True, text=True)
            self.assert_refused(r, self.words_stale, "console script, stale")

    # ---------------------------------------------------------------------------------------------------------------- the .pth hook (interpreter start)
    def test_pth_hook_route(self):
        probe = [ "-c", "import esmfold2_opt._autoload; print('REACHED')"]
        for mode in ("fast", "big"):
            r = subprocess.run([sys.executable, "-S", *probe], env=dict(self.absent_env, ESMFOLD2_OPT=mode, ESMFOLD2_VARIANT="fast"), capture_output=True, text=True)
            self.assert_refused(r, self.words_absent, ("absent", mode))
            r = subprocess.run([sys.executable, *probe], env=dict(self.env_with(self.stale), ESMFOLD2_OPT=mode, ESMFOLD2_VARIANT="fast"), capture_output=True, text=True)
            self.assert_refused(r, self.words_stale, ("stale", mode))
            r = subprocess.run([sys.executable, *probe], env=dict(self.env_with(self.holed), ESMFOLD2_OPT=mode, ESMFOLD2_VARIANT="fast"), capture_output=True, text=True)
            self.assert_refused(r, self.words_holed, ("holed", mode))
        r = subprocess.run([sys.executable, "-S", *probe], env=self.absent_env, capture_output=True, text=True)                          # no mode named: the hook imports nothing, stock runs
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "REACHED"), r.stderr[-400:])

    # ---------------------------------------------------------------------------------------------------------------- in-process enable()
    def test_enable_route(self):
        code = ("import sys, esmfold2_opt as k\n"
                "rep = k.enable(sys.argv[1], 'fast')\n"
                "print('REASON=' + str(rep.get('reason')))\n"
                "sys.exit(0 if not rep.get('active') else 1)\n")
        for mode in ("fast", "big", "exact"):
            r = subprocess.run([sys.executable, "-c", code, mode], env=self.env_with(self.stale), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, (mode, r.stderr[-600:])); self.assertNotIn("Traceback", r.stderr, mode)
            self.assertIn("REASON=reason=core_mismatch: opt_core pinned >= v%s at %s, installed v0.2.5 at " % (self.pin["version"], self.pin["path"]), r.stdout, mode)
            r = subprocess.run([sys.executable, "-c", code, mode], env=self.env_with(self.holed), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, (mode, r.stderr[-600:])); self.assertIn("REASON=producer_missing:opt_core.mem,", r.stdout, mode); self.assertNotIn("Traceback", r.stderr, mode)

    # ---------------------------------------------------------------------------------------------------------------- run.sh and configs/h100.env
    def test_shell_routes(self):
        run_sh, env_file = os.path.join(KIT, "run.sh"), os.path.join(KIT, "configs", "h100.env")
        pyS = python_S_dir(self.tmp.name)                                                                    # `python` = this interpreter with -S: no installed core
        py = os.path.dirname(sys.executable)
        absent = dict(self.absent_env, PATH=os.pathsep.join([pyS, os.environ.get("PATH", "")]))
        stale = dict(self.env_with(self.stale), PATH=os.pathsep.join([py, os.environ.get("PATH", "")]))
        holed = dict(self.env_with(self.holed), PATH=os.pathsep.join([py, os.environ.get("PATH", "")]))
        for env, words, what in ((absent, self.words_absent, "absent"), (stale, self.words_stale, "stale"), (holed, self.words_holed, "holed")):
            r = subprocess.run(["bash", run_sh, "check", "--variant", "fast", "--mode", "fast"], env=env, capture_output=True, text=True)
            self.assert_refused(r, words, ("run.sh", what))
            r = subprocess.run(["bash", "-c", f'source "{env_file}" || exit $?; echo REACHED'], env=env, capture_output=True, text=True)
            self.assert_refused(r, words, ("h100.env", what))


class TestActivationImportGuard(unittest.TestCase):
    def test_activation_with_a_core_module_missing_is_named_not_a_traceback(self):
        """A core module that fails to import DURING activation (past both gates) is the report's `core_missing:<module>` reason, exit 3 —
        stack's import guard; never a traceback."""
        code = ("import builtins, sys\n"
                "real = builtins.__import__\n"
                "def fake(name, *a, **k):\n"
                "    if name == 'opt_core.gates': raise ImportError(\"No module named 'opt_core.gates'\", name='opt_core.gates')\n"
                "    return real(name, *a, **k)\n"
                "import esmfold2_opt, esmfold2_opt.stack as S\n"
                "S._stack_gates_orig = S._stack_gates\n"
                "builtins.__import__ = fake\n"
                "try:\n"
                "    rep = esmfold2_opt.enable('fast', 'fast', strict=True)\n"
                "except esmfold2_opt.ActivationError as e:\n"
                "    print('REFUSED', str(e)[:120]); sys.exit(3)\n"
                "sys.exit(0)\n")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([OPT_DIR, CORE_SRC]), PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, (r.stdout + r.stderr)[-900:])
        self.assertNotIn("Traceback", r.stderr)
        self.assertTrue("core_missing:opt_core.gates" in (r.stdout + r.stderr) or "reason=core_mismatch" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-600:])


if __name__ == "__main__":
    unittest.main()
