"""The shared core (common/opt_core) as this kit stands on it: the pin in opt/pyproject.toml is a version floor the imported core must
clear (the first gate of every activation), the build backend is the core's template byte for byte, the stock proof carries
the core's version block — and the kit's OWN autoload hook (kept: the core's finder is not routed) fails loud on an unknown selection and
survives a find_spec probe of its trigger."""
import importlib
import importlib.util
import io
import shutil
import stat
import subprocess
import re
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

from chai1_opt import _autoload, _core, stack, stock_fold

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TREE = os.path.dirname(OPT)


class TestCorePin(unittest.TestCase):
    def test_pin_is_a_floor_the_imported_core_clears(self):
        facts = _core.core_gate()                                     # THE pin gate (the template _core_gate.py): the facts on a match
        self.assertIn("package_dir", facts["installed"])               # stable subset: version + package_dir, whatever else the core carries
        from chai1_opt._core_gate import version_tuple                # the pin is a FLOOR (>=), as the runtime gate reads it — never equality
        self.assertGreaterEqual(version_tuple(facts["installed"]["version"]), version_tuple(facts["pinned"]["version"]),
                                (facts["installed"]["version"], facts["pinned"]["version"]))
        self.assertEqual(facts["tag"], "chai1-opt")
        from opt_core.gates import core_pin_check                     # the core's own live comparison agrees with the gate's verdict
        g = core_pin_check(_core.PYPROJECT)
        self.assertTrue(g.ok, g.reason)

    def test_core_gate_is_the_house_template_byte_for_byte(self):
        import hashlib
        kit_copy = os.path.join(os.path.dirname(_core.__file__), "_core_gate.py")
        template = os.path.join(_core.tree_core_dir(), "kit_template", "_core_gate.py")
        self.assertTrue(os.path.isfile(template), template)
        self.assertEqual(hashlib.sha256(open(kit_copy, "rb").read()).hexdigest(), hashlib.sha256(open(template, "rb").read()).hexdigest())

    def test_backend_is_the_template(self):
        tpl = os.path.join(_core.tree_core_dir(), "kit_template", "_build_backend.py")
        self.assertEqual(open(os.path.join(OPT, "_build_backend.py"), "rb").read(), open(tpl, "rb").read())

    def test_stock_proof_carries_the_core(self):
        env = {"PATH": "/bin", "CHAI_DOWNLOADS_DIR": "/data/chai_downloads"}
        proof = stock_fold.env_proof(environ=env, path=["/usr/lib/python3"], modules={"os": None, "sys": None})
        self.assertTrue(proof["clean"]); self.assertTrue(proof["core"]["ok"], proof["core"])
        with self.assertRaises(RuntimeError):     # a kit directory on the path is a violation for both proofs
            stock_fold.env_proof(environ=env, path=[os.path.join(OPT, "forward", "fast_inference", "kit")], modules={})


class TestAutoloadHook(unittest.TestCase):
    def test_core_missing_at_activation_is_not_active_exit_3(self):
        """The .pth route with the shared core neither installed nor at the pin's path: the trigger's import fires the hook, the core
        pin gate names the missing core, the process exits 3 with the NOT ACTIVE line — never a traceback, never stock in silence."""
        from chai1_opt.tests import _stubs
        torch, chai1, esm, saved = _stubs.install()
        keep = (_core.ensure_importable, sys.modules.pop("opt_core", None), sys.modules.pop("opt_core.gates", None))
        core_mods = {k: sys.modules.pop(k) for k in [k for k in sys.modules if k == "opt_core" or k.startswith("opt_core.")]}
        def absent():
            raise ImportError("opt_core is neither installed nor at the pin's path /nonexistent/common/opt_core (install the core beside this kit: pip install -e <tree>/common/opt_core)")
        _core.ensure_importable = absent
        stack.reset_for_tests()
        err = io.StringIO()
        try:
            f = _autoload.Finder("exact")
            with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                f._fire("chai_lab.chai1")
            self.assertEqual(cm.exception.code, _autoload.EXIT_NOT_ACTIVE)
            self.assertIn("[chai1-opt] NOT ACTIVE: reason=core_missing:opt_core (opt_core is neither installed nor at the pin's path", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())
        finally:
            _core.ensure_importable = keep[0]
            sys.modules.update(core_mods)
            _stubs.remove(saved); stack.reset_for_tests()

    def test_package_import_failure_at_activation_is_named_core_missing(self):
        """An ImportError raised while the hook imports the package itself (a broken install) is the named `core_missing:` refusal, exit 3."""
        import builtins
        real_import = builtins.__import__
        def fail_det(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "chai1_opt" and fromlist and "det" in fromlist:
                raise ImportError("No module named 'opt_core'", name="opt_core")
            return real_import(name, globals, locals, fromlist, level)
        err = io.StringIO()
        builtins.__import__ = fail_det
        try:
            f = _autoload.Finder("exact")
            with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                f._fire("chai_lab.chai1")
        finally:
            builtins.__import__ = real_import
        self.assertEqual(cm.exception.code, _autoload.EXIT_NOT_ACTIVE)
        self.assertIn("[chai1-opt] NOT ACTIVE: reason=core_missing:opt_core (No module named 'opt_core')", err.getvalue())

    def test_unknown_selection_exits_not_active(self):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            _autoload.install({"CHAI1_OPT": "bogus"})
        self.assertEqual(cm.exception.code, _autoload.EXIT_NOT_ACTIVE)
        self.assertIn("[chai1-opt] NOT ACTIVE: unknown CHAI1_OPT='bogus' (expected exact|fast|big|off)", err.getvalue())

    def test_undeclared_variable_is_refused_in_both_routes(self):
        self.assertEqual(tuple(_autoload.DECLARED_ENV), tuple(stack.DECLARED_ENV))
        env = {"CHAI1_OPT_MODE": "exact", "PATH": "/bin"}
        self.assertEqual(_autoload.undeclared(env), ["CHAI1_OPT_MODE"])
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            _autoload.install(env)                                       # the .pth route: exit 3 even with CHAI1_OPT unset
        self.assertEqual(cm.exception.code, _autoload.EXIT_NOT_ACTIVE)
        self.assertIn("NOT ACTIVE: undeclared variable(s) CHAI1_OPT_MODE (this package reads CHAI1_OPT, CHAI1_OPT_HOME", err.getvalue())
        saved = os.environ.get("CHAI1_OPT_MODE")
        os.environ["CHAI1_OPT_MODE"] = "exact"
        try:
            self.assertTrue(stack.gates({"mode": "exact"}, need_gpu=False).startswith("undeclared variable(s) CHAI1_OPT_MODE"))   # the CLI route: the first refusal
        finally:
            if saved is None: os.environ.pop("CHAI1_OPT_MODE")
            else: os.environ["CHAI1_OPT_MODE"] = saved

    def test_find_spec_probe_then_import_fires_once(self):
        """A bare importlib.util.find_spec of the trigger must not disarm the hook: the following real import still fires it, once."""
        tmp = tempfile.mkdtemp()
        pkg = os.path.join(tmp, "chai_lab"); os.makedirs(pkg)
        open(os.path.join(pkg, "__init__.py"), "w").close(); open(os.path.join(pkg, "chai1.py"), "w").write("MARK = 1\n")
        fired = []
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "chai_lab" or k.startswith("chai_lab.")}
        sys.path.insert(0, tmp)
        try:
            f = _autoload.Finder("exact"); f._fire = lambda trigger: (fired.append(trigger), setattr(f, "fired", trigger))
            sys.meta_path.insert(0, f)
            self.assertIsNotNone(importlib.util.find_spec("chai_lab.chai1"))      # the probe
            self.assertEqual(fired, [])
            mod = importlib.import_module("chai_lab.chai1")                         # the real import
            self.assertEqual(mod.MARK, 1); self.assertEqual(fired, ["chai_lab.chai1"])
            importlib.reload(mod)
            self.assertEqual(fired, ["chai_lab.chai1"])                             # once
        finally:
            if f in sys.meta_path: sys.meta_path.remove(f)
            sys.path.remove(tmp)
            for k in list(sys.modules):
                if k == "chai_lab" or k.startswith("chai_lab."): sys.modules.pop(k)
            sys.modules.update(saved)


OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # <tree>/chai1/opt
CHAI1 = os.path.dirname(OPT)


# The in-process API route in a fresh interpreter: enable() returns the inactive report naming the reason (the NOT ACTIVE line on stderr);
# enable(strict=True) raises chai1_opt.ActivationError carrying the same reason — the probe maps that documented exception to exit 3.
API_PROBE = ("import chai1_opt, sys\n"
             "r = chai1_opt.enable('{mode}')\n"
             "print('ACTIVE' if r['active'] else 'INACTIVE ' + r['reason'])\n"
             "try:\n"
             "    chai1_opt.enable('{mode}', strict=True)\n"
             "except chai1_opt.ActivationError as e:\n"
             "    sys.exit(3 if str(e) == r.get('reason') else 5)\n"
             "sys.exit(4)\n")


def _coreless_tree():
    """A copy of the engine directory's package, run.sh, configs and stock/ under a temp tree that has NO common/opt_core beside it."""
    tmp = tempfile.mkdtemp(prefix="chai1_opt_nocore_")
    dst = os.path.join(tmp, "tree", "chai1")
    shutil.copytree(OPT, os.path.join(dst, "opt"), ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", ".pytest_cache"))
    for name in ("run.sh",):
        shutil.copy2(os.path.join(CHAI1, name), os.path.join(dst, name))
    for name in ("configs", "stock"):
        shutil.copytree(os.path.join(CHAI1, name), os.path.join(dst, name), ignore=shutil.ignore_patterns("src", "*.whl"))
    return tmp, dst


def _shadow_older_core():
    """A temp dir carrying an ``opt_core`` package with a version and nothing else: an older core that lacks every module this package imports."""
    tmp = tempfile.mkdtemp(prefix="chai1_opt_oldcore_")
    os.makedirs(os.path.join(tmp, "opt_core"))
    with open(os.path.join(tmp, "opt_core", "__init__.py"), "w") as fh:
        fh.write('__version__ = "0.2.5"\n')
    return tmp


class TestCoreAbsentOrOlderRefusesByName(unittest.TestCase):
    """Every real entry route refuses BY NAME with exit 3 before anything resolves when the shared core is absent (``core_missing:opt_core``)
    or older than this package's imports (``producer_missing:<module>,…``) — `python -m chai1_opt <verb>`, `bash run.sh <verb>`, the driver,
    the in-process API and the .pth hook; never a traceback, never stock in silence."""

    def _run(self, argv, env_extra, cwd):
        env = {k: v for k, v in os.environ.items() if not k.startswith("CHAI1_OPT")}
        env.update(env_extra)
        return subprocess.run(argv, capture_output=True, text=True, env=env, cwd=cwd)

    def test_cli_and_run_sh_at_an_absent_core_exit_3_named(self):
        tmp, dst = _coreless_tree()
        try:
            bindir = os.path.join(tmp, "bin"); os.makedirs(bindir)
            stub = os.path.join(bindir, "python")                    # `python` for run.sh: -S hides any installed core (site off), the pin check answers 0, PYTHONPATH carries the copied package
            with open(stub, "w") as fh:
                fh.write("#!/bin/bash\n" 'if [ "$1" = -I ] && [[ "$2" == *check_pins.py ]]; then exit 0; fi\n' f'exec "{sys.executable}" -S "$@"\n')
            os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC)
            env = {"PYTHONPATH": os.path.join(dst, "opt"), "PATH": bindir + os.pathsep + os.environ.get("PATH", "")}
            probes = [([sys.executable, "-S", "-m", "chai1_opt", "check", "--mode", "fast"], "python -m chai1_opt check"),
                      ([sys.executable, "-S", "-m", "chai1_opt", "pred", "--mode", "big", "--input", "x.fasta", "--out_dir", "q"], "python -m chai1_opt pred"),
                      ([sys.executable, "-S", "-m", "chai1_opt.driver", "--mode", "fast", "--no-low-memory", "--uids", "x", "--seeds", "0", "--out_dir", "q", "--tag", "t", "--msa_dir", "m"], "python -m chai1_opt.driver"),
                      (["bash", os.path.join(dst, "run.sh"), "check", "--mode", "fast"], "run.sh check"),
                      (["bash", os.path.join(dst, "run.sh"), "pred", "--mode", "exact", "--input", "x.fasta", "--out_dir", "q"], "run.sh pred"),
                      (["bash", os.path.join(dst, "run.sh"), "check", "--config", "h100", "--mode", "big"], "run.sh check --config h100 (configs/h100.env's own guard)"),
                      ([sys.executable, "-S", "-c", "import chai1_opt._autoload as a; a.Finder('fast')._fire('chai_lab.chai1')"], "the .pth hook's activation"),
                      ([sys.executable, "-S", "-c", API_PROBE.format(mode="exact")], "in-process enable() / enable(strict=True)")]
            for argv, label in probes:
                r = self._run(argv, env, tmp)
                self.assertEqual(r.returncode, 3, (label, r.stdout[-400:], r.stderr[-600:]))
                self.assertIn("[chai1-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v", r.stderr, label)   # the template gate's words
                self.assertIn("; nothing importable as opt_core on sys.path)", r.stderr, label)
                self.assertNotIn("Traceback", r.stderr, label); self.assertNotIn("ModuleNotFoundError", r.stderr, label)
                if "enable()" in label:
                    self.assertIn("INACTIVE reason=core_missing:opt_core (pinned >= v", r.stdout)
                self.assertFalse(os.path.exists(os.path.join(tmp, "q")), label)      # nothing resolved: no output directory was made
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_every_route_at_a_stale_core_names_the_mismatch_exit_3(self):
        """A core PRESENT but not the pinned one (a shadow ``opt_core`` 0.2.5 with its MANIFEST and no submodule, first on the path, any
        installed core hidden with -S): every documented route prints the gate's core_mismatch line and exits 3 before anything resolves —
        `python -m chai1_opt <verb>`, the driver, `bash run.sh <verb>` with and without --config, the .pth hook's activation, and the
        in-process enable() (report inactive with the reason; strict → exit 3)."""
        shadow = _shadow_older_core(); tmp, dst = _coreless_tree()
        try:
            bindir = os.path.join(tmp, "bin"); os.makedirs(bindir)
            stub = os.path.join(bindir, "python")
            with open(stub, "w") as fh:
                fh.write("#!/bin/bash\n" 'if [ "$1" = -I ] && [[ "$2" == *check_pins.py ]]; then exit 0; fi\n' f'exec "{sys.executable}" -S "$@"\n')
            os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC)
            env = {"PYTHONPATH": shadow + os.pathsep + os.path.join(dst, "opt"), "PATH": bindir + os.pathsep + os.environ.get("PATH", "")}
            hook = "import chai1_opt._autoload as a; a.Finder('big')._fire('chai_lab.chai1')"
            api = API_PROBE.format(mode="fast")
            probes = [([sys.executable, "-S", "-m", "chai1_opt", "check", "--mode", "exact"], "python -m chai1_opt check"),
                      ([sys.executable, "-S", "-m", "chai1_opt", "check", "--mode", "big", "--n_gpu", "2"], "python -m chai1_opt check big P=2"),
                      ([sys.executable, "-S", "-m", "chai1_opt", "pred", "--mode", "fast", "--input", "x.fasta", "--out_dir", "q"], "python -m chai1_opt pred"),
                      ([sys.executable, "-S", "-m", "chai1_opt", "warm", "--mode", "fast"], "python -m chai1_opt warm"),
                      ([sys.executable, "-S", "-m", "chai1_opt.driver", "--mode", "fast", "--no-low-memory", "--uids", "x", "--seeds", "0", "--out_dir", "q", "--tag", "t", "--msa_dir", "m"], "python -m chai1_opt.driver"),
                      (["bash", os.path.join(dst, "run.sh"), "check", "--mode", "fast"], "run.sh check"),
                      (["bash", os.path.join(dst, "run.sh"), "check", "--config", "h100", "--mode", "big"], "run.sh check --config h100"),
                      (["bash", os.path.join(dst, "run.sh"), "pred", "--mode", "exact", "--input", "x.fasta", "--out_dir", "q"], "run.sh pred"),
                      ([sys.executable, "-S", "-c", hook], "the .pth hook's activation"),
                      ([sys.executable, "-S", "-c", api], "in-process enable() / enable(strict=True)")]
            for argv, label in probes:
                r = self._run(argv, env, tmp)
                self.assertEqual(r.returncode, 3, (label, r.stdout[-400:], r.stderr[-600:]))
                self.assertIn("[chai1-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned ", r.stderr, label)   # the template gate's words
                self.assertIn(", installed v0.2.5 at " + shadow, r.stderr, label)
                self.assertEqual(r.stderr.count("NOT ACTIVE"), 2 if "enable()" in label else 1, label)   # one line per entry call (the API probe calls enable() twice)
                self.assertNotIn("Traceback", r.stderr, label); self.assertNotIn("ModuleNotFoundError", r.stderr, label)
                self.assertFalse(os.path.exists(os.path.join(tmp, "q")), label)
                if "enable()" in label:
                    self.assertIn("INACTIVE reason=core_mismatch: opt_core pinned ", r.stdout)  # the non-strict call returned the report first
        finally:
            shutil.rmtree(shadow, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)

    def test_in_process_api_and_pth_hook_at_an_older_core(self):
        keep = _core.missing_producers
        _core.missing_producers = lambda: ["opt_core.mem.ngpu"]
        stack.reset_for_tests()
        try:
            rep = stack.activate("fast")                                                     # enable()'s body, strict=False: the report names it
            self.assertFalse(rep["active"]); self.assertTrue(rep["reason"].startswith("reason=producer_missing:opt_core.mem.ngpu ("), rep["reason"])   # the second statement: a MATCHING core that lacks a module
            err = io.StringIO()
            f = _autoload.Finder("big")
            with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                f._fire("chai_lab.chai1")
            self.assertEqual(cm.exception.code, _autoload.EXIT_NOT_ACTIVE)
            self.assertIn("[chai1-opt] NOT ACTIVE: reason=producer_missing:opt_core.mem.ngpu (", err.getvalue()); self.assertNotIn("Traceback", err.getvalue())
        finally:
            _core.missing_producers = keep; stack.reset_for_tests()

    def test_required_producers_is_the_packages_import_list(self):
        """_core.REQUIRED_PRODUCERS names exactly the opt_core modules the package's source imports (both ways), sorted, each present in the
        core beside this tree."""
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        found = set()
        pat = re.compile(r"^\s*(?:from (opt_core(?:\.\w+)*) import ([\w, ]+)|import (opt_core(?:\.\w+)+))", re.M)
        modpat = re.compile(r"""(?:CORE_MODULE|SERVE_MODULE)\s*=\s*["'](opt_core[.\w]+)["']""")
        for name in os.listdir(pkg):
            if not name.endswith(".py"):
                continue
            src = open(os.path.join(pkg, name), encoding="utf-8").read()
            for m in pat.finditer(src):
                if m.group(3):
                    found.add(m.group(3)); continue
                base = m.group(1)
                for leaf in (x.strip().split(" ")[0] for x in m.group(2).split(",") if x.strip()):
                    cand = f"{base}.{leaf}"
                    found.add(cand if cand in _core.REQUIRED_PRODUCERS else base)           # `from opt_core.mem import chunk` names a module; `from opt_core.gates import core_pin` a symbol
            found.update(modpat.findall(src))
        found.discard("opt_core")
        self.assertEqual(sorted(found), list(_core.REQUIRED_PRODUCERS))
        self.assertEqual(_core.missing_producers(), [])                                      # the core beside this tree carries them all

class TestInstallProbe(unittest.TestCase):
    """run.sh and configs/h100.env probe `python -c "import chai1_opt"` and relay any non-zero exit verbatim — the import's own words and its
    own code (an undeclared-variable refusal at interpreter start exits 3 by name; a broken import keeps its traceback and rc) — and say
    'not installed' only for the package's own ModuleNotFoundError."""
    SCENARIOS = {   # what the fake interpreter does on `-c "import chai1_opt"`: (stderr, rc)
        "refused": ("[chai1-opt] NOT ACTIVE: undeclared variable(s) CHAI1_OPT_MODE (this package reads CHAI1_OPT, CHAI1_OPT_HOME)", 3),
        "broken": ("Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nImportError: cannot import name 'thing' from 'torch'", 1),
        "absent": ("Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nModuleNotFoundError: No module named 'chai1_opt'", 1),
    }

    def _run(self, argv, scenario):
        import subprocess, tempfile, textwrap
        text, rc = self.SCENARIOS[scenario]
        with tempfile.TemporaryDirectory() as bindir:
            stub = os.path.join(bindir, "python")
            with open(stub, "w") as fh:
                fh.write(textwrap.dedent(f"""\
                    #!/bin/bash
                    # a fake interpreter: the import probe answers the scenario; anything else must not be reached
                    if [ "$1" = "-c" ] && [ "$2" = "import chai1_opt" ]; then printf '%s\\n' {text!r} >&2; exit {rc}; fi
                    echo "fake python reached past the probe: $*" >&2; exit 97
                    """).replace(repr(text), "'" + text.replace("'", "'\\''") + "'"))
            os.chmod(stub, 0o755)
            env = {"PATH": bindir + os.pathsep + "/usr/bin:/bin", "HOME": bindir}
            r = subprocess.run(argv, cwd=TREE, env=env, capture_output=True, text=True)
            return r.returncode, r.stderr

    def test_run_sh_relays_the_imports_refusal_and_rc(self):
        rc, err = self._run(["bash", os.path.join(TREE, "run.sh"), "check", "--mode", "fast"], "refused")
        self.assertEqual(rc, 3, err); self.assertIn("NOT ACTIVE: undeclared variable(s) CHAI1_OPT_MODE", err); self.assertNotIn("not installed", err)
        rc, err = self._run(["bash", os.path.join(TREE, "run.sh"), "check", "--mode", "fast"], "broken")
        self.assertEqual(rc, 1, err); self.assertIn("ImportError: cannot import name 'thing' from 'torch'", err); self.assertNotIn("not installed", err)
        rc, err = self._run(["bash", os.path.join(TREE, "run.sh"), "check", "--mode", "fast"], "absent")
        self.assertEqual(rc, 3, err); self.assertIn("run.sh: chai1_opt is not installed on", err); self.assertNotIn("fake python reached", err)

    def test_the_config_relays_the_imports_refusal_and_rc(self):
        cfg = os.path.join(TREE, "configs", "h100.env")
        rc, err = self._run(["bash", "-c", f'source "{cfg}"'], "refused")
        self.assertEqual(rc, 3, err); self.assertIn("NOT ACTIVE: undeclared variable(s) CHAI1_OPT_MODE", err); self.assertNotIn("not installed", err)
        rc, err = self._run(["bash", "-c", f'source "{cfg}"'], "broken")
        self.assertEqual(rc, 1, err); self.assertIn("ImportError: cannot import name 'thing' from 'torch'", err); self.assertNotIn("not installed", err)
        rc, err = self._run(["bash", "-c", f'source "{cfg}"'], "absent")
        self.assertEqual(rc, 2, err); self.assertIn("[chai1-opt] NOT ACTIVE: chai1_opt is not installed on", err)
        rc, err = self._run(["bash", os.path.join(TREE, "run.sh"), "check", "--config", "h100", "--mode", "fast"], "broken")   # through run.sh --config: the config's relay, rc 1
        self.assertEqual(rc, 1, err); self.assertIn("ImportError", err); self.assertNotIn("not installed", err)


if __name__ == "__main__":
    unittest.main()
